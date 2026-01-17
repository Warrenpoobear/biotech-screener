#!/usr/bin/env python3
"""
Module 5: Composite Ranker (v2)

Enhanced composite scoring with:
- Decomposable score breakdown object for full auditability
- Confidence-weighted scoring (module-level confidence affects weights)
- Monotonic caps (risk gates can't be "outvoted")
- Robust normalization (winsorized rank percentile)
- Weakest-link hybrid aggregation (prevents masking disasters)
- Determinism hash for audit parity with Module 2 v2

Design Philosophy (Governed Over Smart):
- DETERMINISTIC: No datetime.now(), no randomness
- STDLIB-ONLY: No external dependencies
- DECIMAL-ONLY: Pure Decimal arithmetic for all scoring
- FAIL LOUDLY: Clear error states
- AUDITABLE: Full provenance chain with score breakdown

DETERMINISM CONTRACT:
----------------------
This module guarantees deterministic output for identical inputs:
1. All arithmetic uses Decimal with explicit quantization rules
2. Hash computation uses stable key ordering (sorted dict keys)
3. Hash includes: version, weights, inputs, subscores, caps, penalties
4. No floating-point intermediate calculations
5. No datetime.now() or random calls
6. Output field ordering is stable

To verify determinism: same inputs MUST produce identical determinism_hash values.

BREAKING CHANGES FROM v1:
-------------------------
- Added score_breakdown object with full decomposition
- Added confidence_* fields (clinical, financial, catalyst, pos)
- Added effective_weights (confidence-weighted and regime-adjusted)
- Added monotonic_caps_applied list
- Added determinism_hash for audit
- Normalization now uses winsorized rank percentile
- Aggregation uses hybrid weighted-sum + weakest-link

Weights (default):
- Clinical: 40%
- Financial: 35%
- Catalyst: 25%

Weights (enhanced with PoS):
- Clinical: 30%
- Financial: 30%
- Catalyst: 20%
- PoS: 20%

Author: Wake Robin Capital Management
Version: 2.0.0
Last Modified: 2026-01-11
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from common.provenance import create_provenance
from common.types import Severity

__version__ = "2.0.0"
RULESET_VERSION = "2.0.0-V2"
SCHEMA_VERSION = "v2.0"

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

SCORE_PRECISION = Decimal("0.01")
WEIGHT_PRECISION = Decimal("0.0001")
EPS = Decimal("0.000001")

DEFAULT_WEIGHTS = {
    "clinical": Decimal("0.40"),
    "financial": Decimal("0.35"),
    "catalyst": Decimal("0.25"),
}

ENHANCED_WEIGHTS = {
    "clinical": Decimal("0.30"),
    "financial": Decimal("0.30"),
    "catalyst": Decimal("0.20"),
    "pos": Decimal("0.20"),
}

SEVERITY_MULTIPLIERS = {
    Severity.NONE: Decimal("1.0"),
    Severity.SEV1: Decimal("0.90"),
    Severity.SEV2: Decimal("0.50"),
    Severity.SEV3: Decimal("0.0"),
}

SEVERITY_GATE_LABELS = {
    Severity.NONE: "NONE",
    Severity.SEV1: "SEV1_10PCT",
    Severity.SEV2: "SEV2_HALF",
    Severity.SEV3: "SEV3_EXCLUDE",
}

MIN_COHORT_SIZE = 5
MAX_UNCERTAINTY_PENALTY = Decimal("0.30")
WINSOR_LOW = Decimal("5")
WINSOR_HIGH = Decimal("95")
HYBRID_ALPHA = Decimal("0.85")
CRITICAL_COMPONENTS = ["financial", "clinical"]

# Volatility adjustment parameters
VOLATILITY_BASELINE = Decimal("0.50")  # 50% annualized vol as baseline
VOLATILITY_LOW_THRESHOLD = Decimal("0.30")  # Below this = low vol
VOLATILITY_HIGH_THRESHOLD = Decimal("0.80")  # Above this = high vol
VOLATILITY_MAX_ADJUSTMENT = Decimal("0.25")  # Max +/- 25% weight adjustment


class MonotonicCap:
    """Monotonic cap thresholds."""
    LIQUIDITY_FAIL_CAP = Decimal("35")
    LIQUIDITY_WARN_CAP = Decimal("60")
    RUNWAY_CRITICAL_CAP = Decimal("40")
    RUNWAY_WARNING_CAP = Decimal("55")
    DILUTION_SEVERE_CAP = Decimal("45")
    DILUTION_HIGH_CAP = Decimal("60")


class ScoringMode(str, Enum):
    DEFAULT = "default"
    ENHANCED = "enhanced"


class NormalizationMethod(str, Enum):
    COHORT = "cohort"
    COHORT_WINSORIZED = "cohort_winsorized"
    STAGE_FALLBACK = "stage_fallback"
    GLOBAL_FALLBACK = "global_fallback"
    NONE = "none"


# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class ComponentScore:
    name: str
    raw: Optional[Decimal]
    normalized: Optional[Decimal]
    confidence: Decimal
    weight_base: Decimal
    weight_effective: Decimal
    contribution: Decimal
    notes: List[str] = field(default_factory=list)


@dataclass
class ScoreBreakdown:
    version: str
    mode: str
    base_weights: Dict[str, str]
    regime_adjustments: Dict[str, str]
    effective_weights: Dict[str, str]
    components: List[Dict[str, Any]]
    penalties_and_gates: Dict[str, Any]
    bonuses: Dict[str, Any]
    final: Dict[str, str]
    normalization_method: str
    cohort_info: Dict[str, Any]
    hybrid_aggregation: Dict[str, str]


# ============================================================================
# HELPERS
# ============================================================================

def _to_decimal(value: Any, default: Optional[Decimal] = None) -> Optional[Decimal]:
    if value is None:
        return default
    try:
        if isinstance(value, Decimal):
            return value
        if isinstance(value, (int, float)):
            return Decimal(str(value))
        if isinstance(value, str):
            stripped = value.strip()
            return Decimal(stripped) if stripped else default
        return default
    except (InvalidOperation, ValueError):
        return default


def _quantize_score(value: Decimal) -> Decimal:
    return value.quantize(SCORE_PRECISION, rounding=ROUND_HALF_UP)


def _quantize_weight(value: Decimal) -> Decimal:
    return value.quantize(WEIGHT_PRECISION, rounding=ROUND_HALF_UP)


def _clamp(value: Decimal, min_val: Decimal, max_val: Decimal) -> Decimal:
    return max(min_val, min(max_val, value))


def _quarter_from_date(d: date) -> str:
    q = (d.month - 1) // 3 + 1
    return f"{d.year}Q{q}"


def _market_cap_bucket(market_cap_mm: Optional[Any]) -> str:
    mcap = _to_decimal(market_cap_mm)
    if mcap is None:
        return "unknown"
    if mcap >= 10000:
        return "large"
    elif mcap >= 2000:
        return "mid"
    elif mcap >= 300:
        return "small"
    return "micro"


def _stage_bucket(lead_phase: Optional[str]) -> str:
    if not lead_phase:
        return "early"
    phase = lead_phase.lower()
    if "3" in phase or "approved" in phase:
        return "late"
    elif "2" in phase:
        return "mid"
    return "early"


def _get_worst_severity(severities: List[str]) -> Severity:
    priority = {Severity.SEV3: 3, Severity.SEV2: 2, Severity.SEV1: 1, Severity.NONE: 0}
    worst = Severity.NONE
    for s in severities:
        try:
            sev = Severity(s)
            if priority[sev] > priority[worst]:
                worst = sev
        except (ValueError, KeyError):
            continue
    return worst


# ============================================================================
# WINSORIZED NORMALIZATION
# ============================================================================

def _rank_normalize_winsorized(values: List[Decimal]) -> Tuple[List[Decimal], bool]:
    """Winsorized percentile rank normalization."""
    n = len(values)
    if n == 0:
        return [], False
    if n == 1:
        return [Decimal("50")], False

    indexed = [(v, i) for i, v in enumerate(values)]
    indexed.sort(key=lambda x: x[0])

    ranks = [Decimal("0")] * n
    i = 0
    while i < n:
        j = i
        while j < n and indexed[j][0] == indexed[i][0]:
            j += 1
        avg_rank = Decimal(str((i + j - 1) / 2))
        for k in range(i, j):
            ranks[indexed[k][1]] = avg_rank
        i = j

    raw_percentiles = []
    for r in ranks:
        pct = (r / Decimal(str(n - 1))) * Decimal("100") if n > 1 else Decimal("50")
        raw_percentiles.append(pct)

    winsorization_applied = False
    result = []
    for pct in raw_percentiles:
        if pct < WINSOR_LOW or pct > WINSOR_HIGH:
            winsorization_applied = True
        clipped = _clamp(pct, WINSOR_LOW, WINSOR_HIGH)
        rescaled = ((clipped - WINSOR_LOW) / (WINSOR_HIGH - WINSOR_LOW)) * Decimal("100")
        result.append(_quantize_score(rescaled))

    return result, winsorization_applied


# ============================================================================
# CONFIDENCE EXTRACTION
# ============================================================================

def _extract_confidence_financial(fin_data: Dict) -> Decimal:
    conf = _to_decimal(fin_data.get("confidence"))
    if conf is not None:
        return _clamp(conf, Decimal("0"), Decimal("1"))
    state = fin_data.get("financial_data_state", "NONE")
    return {"FULL": Decimal("1.0"), "PARTIAL": Decimal("0.7"),
            "MINIMAL": Decimal("0.4"), "NONE": Decimal("0.1")}.get(state, Decimal("0.5"))


def _extract_confidence_clinical(clin_data: Dict) -> Decimal:
    conf = _to_decimal(clin_data.get("confidence"))
    if conf is not None:
        return _clamp(conf, Decimal("0"), Decimal("1"))
    trial_count = clin_data.get("trial_count", 0)
    has_lead = clin_data.get("lead_phase") is not None
    has_score = clin_data.get("clinical_score") is not None
    base = Decimal("0.3")
    if has_score:
        base = Decimal("0.5")
    if trial_count > 0:
        base += Decimal("0.2")
    if trial_count > 3:
        base += Decimal("0.1")
    if has_lead:
        base += Decimal("0.2")
    return _clamp(base, Decimal("0"), Decimal("1"))


def _extract_confidence_catalyst(cat_data: Any) -> Decimal:
    if not cat_data:
        return Decimal("0.3")
    if hasattr(cat_data, 'n_high_confidence'):
        n_high = cat_data.n_high_confidence
        n_events = cat_data.events_detected_total
    elif isinstance(cat_data, dict):
        scores = cat_data.get("scores", cat_data)
        n_high = scores.get("n_high_confidence", 0)
        n_events = scores.get("events_detected_total", 0)
    else:
        return Decimal("0.3")
    if n_events == 0:
        return Decimal("0.3")
    high_ratio = Decimal(str(n_high)) / Decimal(str(max(n_events, 1)))
    return _clamp(Decimal("0.4") + high_ratio * Decimal("0.5"), Decimal("0"), Decimal("1"))


def _extract_confidence_pos(pos_data: Optional[Dict]) -> Decimal:
    if not pos_data:
        return Decimal("0")
    conf = _to_decimal(pos_data.get("confidence"))
    if conf is not None:
        return _clamp(conf, Decimal("0"), Decimal("1"))
    return Decimal("0.7") if pos_data.get("pos_score") is not None else Decimal("0")


# ============================================================================
# VOLATILITY ADJUSTMENT
# ============================================================================

@dataclass
class VolatilityAdjustment:
    """Volatility-based weight adjustment result."""
    annualized_vol: Optional[Decimal]
    vol_bucket: str  # "low", "normal", "high", "unknown"
    adjustment_factor: Decimal  # Multiplier for weights (0.75 - 1.25)
    confidence_penalty: Decimal  # Additional confidence reduction for high vol


def _extract_volatility(
    fin_data: Dict,
    si_data: Optional[Dict],
    market_data: Optional[Dict] = None,
) -> VolatilityAdjustment:
    """
    Extract and compute volatility adjustment factor.

    Volatility can come from multiple sources:
    1. Market data (252-day realized vol)
    2. Short interest data (implied vol if available)
    3. Financial data (price volatility metrics)

    Higher volatility -> lower weight adjustment (more uncertainty)
    Lower volatility -> higher weight adjustment (more confidence)
    """
    vol = None

    # Try market data first (most reliable)
    if market_data:
        vol = _to_decimal(market_data.get("volatility_252d"))
        if vol is None:
            vol = _to_decimal(market_data.get("annualized_volatility"))

    # Try short interest data (may have implied vol)
    if vol is None and si_data:
        vol = _to_decimal(si_data.get("implied_volatility"))
        if vol is None:
            vol = _to_decimal(si_data.get("realized_volatility"))

    # Try financial data
    if vol is None and fin_data:
        vol = _to_decimal(fin_data.get("price_volatility"))
        if vol is None:
            vol = _to_decimal(fin_data.get("volatility_252d"))

    # If no volatility data, return neutral adjustment
    if vol is None:
        return VolatilityAdjustment(
            annualized_vol=None,
            vol_bucket="unknown",
            adjustment_factor=Decimal("1.0"),
            confidence_penalty=Decimal("0.05"),  # Small penalty for unknown
        )

    # Determine bucket and adjustment
    if vol < VOLATILITY_LOW_THRESHOLD:
        vol_bucket = "low"
        # Low vol = more reliable signal = boost weights slightly
        # Linear interpolation: at 0% vol -> +25% adjustment, at 30% vol -> 0%
        adjustment = Decimal("1.0") + VOLATILITY_MAX_ADJUSTMENT * (
            (VOLATILITY_LOW_THRESHOLD - vol) / VOLATILITY_LOW_THRESHOLD
        )
        confidence_penalty = Decimal("0")
    elif vol <= VOLATILITY_HIGH_THRESHOLD:
        vol_bucket = "normal"
        # Normal vol = neutral
        adjustment = Decimal("1.0")
        # Small confidence penalty that increases with volatility
        vol_ratio = (vol - VOLATILITY_LOW_THRESHOLD) / (VOLATILITY_HIGH_THRESHOLD - VOLATILITY_LOW_THRESHOLD)
        confidence_penalty = vol_ratio * Decimal("0.10")  # Up to 10% penalty
    else:
        vol_bucket = "high"
        # High vol = less reliable signal = reduce weight influence
        # Linear interpolation: at 80% vol -> 0%, capped at -25%
        excess_vol = vol - VOLATILITY_HIGH_THRESHOLD
        adjustment = Decimal("1.0") - min(
            VOLATILITY_MAX_ADJUSTMENT,
            VOLATILITY_MAX_ADJUSTMENT * (excess_vol / VOLATILITY_BASELINE)
        )
        confidence_penalty = Decimal("0.15")  # Significant penalty for high vol

    return VolatilityAdjustment(
        annualized_vol=_quantize_score(vol * Decimal("100")),  # Store as percentage
        vol_bucket=vol_bucket,
        adjustment_factor=_quantize_weight(adjustment),
        confidence_penalty=_quantize_weight(confidence_penalty),
    )


# ============================================================================
# MONOTONIC CAPS
# ============================================================================

def _apply_monotonic_caps(
    score: Decimal,
    liquidity_gate_status: Optional[str],
    runway_months: Optional[Decimal],
    dilution_risk_bucket: Optional[str],
) -> Tuple[Decimal, List[Dict[str, Any]]]:
    """Apply monotonic caps based on risk gates."""
    caps_applied = []
    capped = score

    if liquidity_gate_status == "FAIL":
        if score > MonotonicCap.LIQUIDITY_FAIL_CAP:
            caps_applied.append({"reason": "liquidity_gate_fail", "cap": str(MonotonicCap.LIQUIDITY_FAIL_CAP)})
            capped = min(capped, MonotonicCap.LIQUIDITY_FAIL_CAP)
    elif liquidity_gate_status == "WARN":
        if score > MonotonicCap.LIQUIDITY_WARN_CAP:
            caps_applied.append({"reason": "liquidity_gate_warn", "cap": str(MonotonicCap.LIQUIDITY_WARN_CAP)})
            capped = min(capped, MonotonicCap.LIQUIDITY_WARN_CAP)

    if runway_months is not None:
        if runway_months < Decimal("6") and capped > MonotonicCap.RUNWAY_CRITICAL_CAP:
            caps_applied.append({"reason": "runway_critical", "cap": str(MonotonicCap.RUNWAY_CRITICAL_CAP)})
            capped = min(capped, MonotonicCap.RUNWAY_CRITICAL_CAP)
        elif runway_months < Decimal("12") and capped > MonotonicCap.RUNWAY_WARNING_CAP:
            caps_applied.append({"reason": "runway_warning", "cap": str(MonotonicCap.RUNWAY_WARNING_CAP)})
            capped = min(capped, MonotonicCap.RUNWAY_WARNING_CAP)

    if dilution_risk_bucket == "SEVERE" and capped > MonotonicCap.DILUTION_SEVERE_CAP:
        caps_applied.append({"reason": "dilution_severe", "cap": str(MonotonicCap.DILUTION_SEVERE_CAP)})
        capped = min(capped, MonotonicCap.DILUTION_SEVERE_CAP)
    elif dilution_risk_bucket == "HIGH" and capped > MonotonicCap.DILUTION_HIGH_CAP:
        caps_applied.append({"reason": "dilution_high", "cap": str(MonotonicCap.DILUTION_HIGH_CAP)})
        capped = min(capped, MonotonicCap.DILUTION_HIGH_CAP)

    return _quantize_score(capped), caps_applied


# ============================================================================
# DETERMINISM HASH
# ============================================================================

def _compute_determinism_hash(
    ticker: str,
    version: str,
    mode: str,
    base_weights: Dict[str, Decimal],
    effective_weights: Dict[str, Decimal],
    regime_adjustments: Dict[str, Decimal],
    component_scores: List[ComponentScore],
    uncertainty_penalty: Decimal,
    severity_gate: str,
    caps: List[Dict],
    cohort_key: str,
    final_score: Decimal,
) -> str:
    payload = {
        "ticker": ticker,
        "version": version,
        "mode": mode,
        "base_weights": {k: str(v) for k, v in sorted(base_weights.items())},
        "effective_weights": {k: str(v) for k, v in sorted(effective_weights.items())},
        "regime_adjustments": {k: str(v) for k, v in sorted(regime_adjustments.items())},
        "components": sorted([
            {"name": c.name, "raw": str(c.raw) if c.raw else None,
             "normalized": str(c.normalized) if c.normalized else None,
             "contribution": str(c.contribution)}
            for c in component_scores
        ], key=lambda x: x["name"]),
        "uncertainty_penalty": str(uncertainty_penalty),
        "severity_gate": severity_gate,
        "caps": sorted([str(c) for c in caps]),
        "cohort_key": cohort_key,
        "final_score": str(final_score),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# ============================================================================
# CO-INVEST OVERLAY
# ============================================================================

def _enrich_with_coinvest(ticker: str, coinvest_signals: dict, as_of_date: date) -> dict:
    signal = coinvest_signals.get(ticker)
    if not signal:
        return {"coinvest_overlap_count": 0, "coinvest_holders": [], "coinvest_usable": False}

    pit_positions = [p for p in signal.positions if p.filing_date < as_of_date]
    if not pit_positions:
        return {"coinvest_overlap_count": 0, "coinvest_holders": [], "coinvest_usable": False}

    holders = sorted(set(p.manager_name for p in pit_positions))
    return {
        "coinvest_overlap_count": len(holders),
        "coinvest_holders": holders,
        "coinvest_quarter": _quarter_from_date(pit_positions[0].report_date),
        "coinvest_usable": True,
    }


# ============================================================================
# COHORT NORMALIZATION
# ============================================================================

def _apply_cohort_normalization_v2(members: List[Dict], include_pos: bool = False) -> NormalizationMethod:
    if not members:
        return NormalizationMethod.NONE

    clin_scores = [m["clinical_raw"] or Decimal("0") for m in members]
    fin_scores = [m["financial_raw"] or Decimal("0") for m in members]
    cat_scores = [m["catalyst_raw"] or Decimal("0") for m in members]

    clin_norm, clin_w = _rank_normalize_winsorized(clin_scores)
    fin_norm, fin_w = _rank_normalize_winsorized(fin_scores)
    cat_norm, cat_w = _rank_normalize_winsorized(cat_scores)

    winsor_applied = clin_w or fin_w or cat_w

    pos_norm = None
    if include_pos:
        pos_scores = [m.get("pos_raw") or Decimal("0") for m in members]
        if any(p > 0 for p in pos_scores):
            pos_norm, pos_w = _rank_normalize_winsorized(pos_scores)
            winsor_applied = winsor_applied or pos_w

    for i, m in enumerate(members):
        m["clinical_normalized"] = clin_norm[i]
        m["financial_normalized"] = fin_norm[i]
        m["catalyst_normalized"] = cat_norm[i]
        m["pos_normalized"] = pos_norm[i] if pos_norm else (m.get("pos_raw") or Decimal("0"))

    return NormalizationMethod.COHORT_WINSORIZED if winsor_applied else NormalizationMethod.COHORT


# ============================================================================
# SINGLE TICKER SCORING
# ============================================================================

def _score_single_ticker_v2(
    ticker: str,
    fin_data: Dict,
    cat_data: Any,
    clin_data: Dict,
    pos_data: Optional[Dict],
    si_data: Optional[Dict],
    base_weights: Dict[str, Decimal],
    regime_adjustments: Dict[str, Decimal],
    mode: ScoringMode,
    normalized_scores: Dict[str, Optional[Decimal]],
    cohort_key: str,
    normalization_method: NormalizationMethod,
    market_data: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Score a single ticker with full breakdown."""

    # Extract volatility adjustment
    vol_adj = _extract_volatility(fin_data, si_data, market_data)

    # Extract raw scores
    fin_raw = _to_decimal(fin_data.get("financial_normalized"))
    clin_raw = _to_decimal(clin_data.get("clinical_score"))
    pos_raw = _to_decimal(pos_data.get("pos_score")) if pos_data else None

    if hasattr(cat_data, 'score_blended'):
        cat_raw = _to_decimal(cat_data.score_blended)
        cat_proximity = _to_decimal(getattr(cat_data, 'catalyst_proximity_score', 0)) or Decimal("0")
        cat_delta = _to_decimal(getattr(cat_data, 'catalyst_delta_score', 0)) or Decimal("0")
    elif isinstance(cat_data, dict):
        scores = cat_data.get("scores", cat_data)
        cat_raw = _to_decimal(scores.get("score_blended", scores.get("catalyst_score_net")))
        cat_proximity = _to_decimal(scores.get("catalyst_proximity_score", 0)) or Decimal("0")
        cat_delta = _to_decimal(scores.get("catalyst_delta_score", 0)) or Decimal("0")
    else:
        cat_raw = None
        cat_proximity = Decimal("0")
        cat_delta = Decimal("0")

    # Get normalized scores
    fin_norm = normalized_scores.get("financial") or fin_raw or Decimal("0")
    clin_norm = normalized_scores.get("clinical") or clin_raw or Decimal("0")
    cat_norm = normalized_scores.get("catalyst") or cat_raw or Decimal("0")
    pos_norm = normalized_scores.get("pos") or pos_raw or Decimal("0")

    # Extract confidences
    conf_fin = _extract_confidence_financial(fin_data)
    conf_clin = _extract_confidence_clinical(clin_data)
    conf_cat = _extract_confidence_catalyst(cat_data)
    conf_pos = _extract_confidence_pos(pos_data)

    # Apply regime adjustments
    quality_adj = regime_adjustments.get("quality", Decimal("1.0"))
    momentum_adj = regime_adjustments.get("momentum", Decimal("1.0"))
    catalyst_adj = regime_adjustments.get("catalyst", Decimal("1.0"))

    # Apply volatility adjustment to regime factors
    # High volatility reduces the impact of all adjustments (converges to baseline)
    vol_factor = vol_adj.adjustment_factor

    adjusted_weights = {
        "clinical": base_weights.get("clinical", Decimal("0.40")) * vol_factor,
        "financial": base_weights.get("financial", Decimal("0.35")) * quality_adj * vol_factor,
        "catalyst": base_weights.get("catalyst", Decimal("0.25")) * catalyst_adj * momentum_adj * vol_factor,
    }
    if mode == ScoringMode.ENHANCED:
        adjusted_weights["pos"] = base_weights.get("pos", Decimal("0.20")) * vol_factor

    # Confidence weighting with volatility penalty
    confidences = {
        "clinical": _clamp(conf_clin - vol_adj.confidence_penalty, Decimal("0"), Decimal("1")),
        "financial": _clamp(conf_fin - vol_adj.confidence_penalty, Decimal("0"), Decimal("1")),
        "catalyst": _clamp(conf_cat - vol_adj.confidence_penalty, Decimal("0"), Decimal("1")),
    }
    if mode == ScoringMode.ENHANCED:
        confidences["pos"] = _clamp(conf_pos - vol_adj.confidence_penalty, Decimal("0"), Decimal("1"))

    effective_weights = {}
    total_weight = Decimal("0")
    for comp, base_w in adjusted_weights.items():
        conf = confidences.get(comp, Decimal("0.5"))
        eff_w = base_w * conf if conf >= Decimal("0.15") else base_w * Decimal("0.3")
        effective_weights[comp] = eff_w
        total_weight += eff_w

    if total_weight > EPS:
        effective_weights = {k: _quantize_weight(v / total_weight) for k, v in effective_weights.items()}

    # Build component scores
    component_scores = []
    contributions = {}
    raw_scores = {"clinical": clin_raw, "financial": fin_raw, "catalyst": cat_raw}
    norm_scores = {"clinical": clin_norm, "financial": fin_norm, "catalyst": cat_norm}

    for name in ["clinical", "financial", "catalyst"]:
        w_eff = effective_weights.get(name, Decimal("0"))
        contrib = norm_scores[name] * w_eff
        contributions[name] = contrib
        component_scores.append(ComponentScore(
            name=name, raw=raw_scores[name], normalized=norm_scores[name],
            confidence=confidences.get(name, Decimal("0")),
            weight_base=base_weights.get(name, Decimal("0")),
            weight_effective=w_eff, contribution=_quantize_score(contrib),
            notes=["missing_raw"] if raw_scores[name] is None else [],
        ))

    if mode == ScoringMode.ENHANCED:
        w_eff = effective_weights.get("pos", Decimal("0"))
        contrib = pos_norm * w_eff
        contributions["pos"] = contrib
        component_scores.append(ComponentScore(
            name="pos", raw=pos_raw, normalized=pos_norm, confidence=conf_pos,
            weight_base=base_weights.get("pos", Decimal("0")),
            weight_effective=w_eff, contribution=_quantize_score(contrib),
            notes=["missing_raw"] if pos_raw is None else [],
        ))

    # Weighted sum
    weighted_sum = sum(contributions.values())

    # Weakest-link
    critical_scores = [
        contributions.get(c, Decimal("0")) / max(effective_weights.get(c, Decimal("1")), EPS)
        for c in CRITICAL_COMPONENTS if c in contributions
    ]
    min_critical = min(critical_scores) if critical_scores else weighted_sum

    # Hybrid aggregation
    pre_penalty = HYBRID_ALPHA * weighted_sum + (Decimal("1") - HYBRID_ALPHA) * min_critical
    pre_penalty = _quantize_score(pre_penalty)

    # Uncertainty penalty
    subfactors = [fin_raw, clin_raw, cat_raw]
    if mode == ScoringMode.ENHANCED:
        subfactors.append(pos_raw)
    missing_count = sum(1 for x in subfactors if x is None)
    missing_pct = Decimal(str(missing_count / len(subfactors)))
    uncertainty_penalty = min(MAX_UNCERTAINTY_PENALTY, missing_pct)

    # Severity
    severities = [fin_data.get("severity", "none"), clin_data.get("severity", "none")]
    if hasattr(cat_data, 'severe_negative_flag') and cat_data.severe_negative_flag:
        severities.append("sev1")
    elif isinstance(cat_data, dict):
        flags_dict = cat_data.get("flags", {})
        if isinstance(flags_dict, dict) and flags_dict.get("severe_negative_flag"):
            severities.append("sev1")
    worst_severity = _get_worst_severity(severities)
    severity_multiplier = SEVERITY_MULTIPLIERS[worst_severity]

    # Apply penalties
    post_penalty = pre_penalty * (Decimal("1") - uncertainty_penalty) * severity_multiplier
    post_penalty = _quantize_score(post_penalty)

    # Monotonic caps
    liquidity_status = fin_data.get("liquidity_gate_status")
    runway = _to_decimal(fin_data.get("runway_months"))
    dilution_bucket = fin_data.get("dilution_risk_bucket")
    post_cap, caps_applied = _apply_monotonic_caps(post_penalty, liquidity_status, runway, dilution_bucket)

    # Bonuses
    proximity_bonus = (cat_proximity / Decimal("20")).quantize(SCORE_PRECISION)
    delta_bonus = (cat_delta / Decimal("20")).quantize(SCORE_PRECISION)
    post_bonus = _clamp(post_cap + proximity_bonus + delta_bonus, Decimal("0"), Decimal("100"))
    post_bonus = _quantize_score(post_bonus)

    # Determinism hash
    determinism_hash = _compute_determinism_hash(
        ticker, SCHEMA_VERSION, mode.value, base_weights, effective_weights,
        regime_adjustments, component_scores, uncertainty_penalty,
        SEVERITY_GATE_LABELS[worst_severity], caps_applied, cohort_key, post_bonus,
    )

    # Build breakdown
    breakdown = ScoreBreakdown(
        version=SCHEMA_VERSION, mode=mode.value,
        base_weights={k: str(v) for k, v in base_weights.items()},
        regime_adjustments={k: str(v) for k, v in regime_adjustments.items()},
        effective_weights={k: str(v) for k, v in effective_weights.items()},
        components=[{
            "name": c.name, "raw": str(c.raw) if c.raw else None,
            "normalized": str(c.normalized) if c.normalized else None,
            "confidence": str(c.confidence), "weight_base": str(c.weight_base),
            "weight_effective": str(c.weight_effective), "contribution": str(c.contribution),
            "notes": c.notes,
        } for c in component_scores],
        penalties_and_gates={
            "uncertainty_penalty_pct": str(_quantize_score(uncertainty_penalty * Decimal("100"))),
            "severity_gate": SEVERITY_GATE_LABELS[worst_severity],
            "severity_multiplier": str(severity_multiplier),
            "monotonic_caps_applied": caps_applied,
        },
        bonuses={"proximity_bonus": str(proximity_bonus), "delta_bonus": str(delta_bonus)},
        final={
            "pre_penalty_score": str(pre_penalty), "post_penalty_score": str(post_penalty),
            "post_cap_score": str(post_cap), "post_bonus_score": str(post_bonus),
            "composite_score": str(post_bonus),
        },
        normalization_method=normalization_method.value,
        cohort_info={"cohort_key": cohort_key},
        hybrid_aggregation={
            "alpha": str(HYBRID_ALPHA), "weighted_sum": str(_quantize_score(weighted_sum)),
            "min_critical": str(_quantize_score(min_critical)),
        },
    )

    # Flags
    flags = list(fin_data.get("flags", [])) + list(clin_data.get("flags", []))
    if isinstance(cat_data, dict):
        flags.extend(cat_data.get("flags", []))
    if uncertainty_penalty > 0:
        flags.append("uncertainty_penalty_applied")
    if worst_severity == Severity.SEV2:
        flags.append("sev2_penalty_applied")
    elif worst_severity == Severity.SEV1:
        flags.append("sev1_penalty_applied")
    if caps_applied:
        flags.append("monotonic_cap_applied")
    if mode == ScoringMode.ENHANCED and pos_raw is not None:
        flags.append("pos_score_applied")
    if vol_adj.vol_bucket == "high":
        flags.append("high_volatility_adjustment")
    elif vol_adj.vol_bucket == "low":
        flags.append("low_volatility_boost")

    return {
        "ticker": ticker,
        "composite_score": post_bonus,
        "severity": worst_severity,
        "flags": sorted(set(flags)),
        "determinism_hash": determinism_hash,
        "score_breakdown": breakdown,
        "confidence_clinical": conf_clin,
        "confidence_financial": conf_fin,
        "confidence_catalyst": conf_cat,
        "confidence_pos": conf_pos if mode == ScoringMode.ENHANCED else None,
        "effective_weights": effective_weights,
        "normalization_method": normalization_method.value,
        "caps_applied": caps_applied,
        "component_scores": component_scores,
        "volatility_adjustment": {
            "annualized_vol_pct": str(vol_adj.annualized_vol) if vol_adj.annualized_vol else None,
            "vol_bucket": vol_adj.vol_bucket,
            "adjustment_factor": str(vol_adj.adjustment_factor),
            "confidence_penalty": str(vol_adj.confidence_penalty),
        },
    }


# ============================================================================
# MAIN COMPOSITE FUNCTION
# ============================================================================

def compute_module_5_composite_v2(
    universe_result: Dict[str, Any],
    financial_result: Dict[str, Any],
    catalyst_result: Dict[str, Any],
    clinical_result: Dict[str, Any],
    as_of_date: str,
    weights: Optional[Dict[str, Decimal]] = None,
    coinvest_signals: Optional[Dict] = None,
    enhancement_result: Optional[Dict[str, Any]] = None,
    market_data_by_ticker: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Any]:
    """
    Compute composite scores with all v2 enhancements.

    Args:
        universe_result: Module 1 output with active/excluded securities
        financial_result: Module 2 output with financial scores
        catalyst_result: Module 3 output with catalyst summaries
        clinical_result: Module 4 output with clinical scores
        as_of_date: ISO date string for the scoring date
        weights: Optional custom weights (defaults to DEFAULT_WEIGHTS or ENHANCED_WEIGHTS)
        coinvest_signals: Optional co-invest overlay signals
        enhancement_result: Optional PoS/SI/regime enhancement data
        market_data_by_ticker: Optional dict mapping ticker to market data
            (including volatility_252d for volatility-adjusted weighting)

    Returns:
        Dict with ranked_securities, excluded_securities, and diagnostics
    """

    # Handle empty universe gracefully
    active_securities = universe_result.get("active_securities", [])
    if not active_securities or len(active_securities) == 0:
        logger.warning("Module 5 v2: Empty universe provided - returning empty results")
        return {
            "as_of_date": as_of_date,
            "scoring_mode": ScoringMode.DEFAULT.value,
            "weights_used": {k: str(v) for k, v in DEFAULT_WEIGHTS.items()},
            "ranked_securities": [],
            "excluded_securities": [],
            "cohort_stats": {},
            "diagnostic_counts": {
                "total_input": 0,
                "rankable": 0,
                "excluded": 0,
                "cohort_count": 0,
                "with_pos_scores": 0,
                "with_caps_applied": 0,
            },
            "enhancement_applied": False,
            "enhancement_diagnostics": None,
            "schema_version": SCHEMA_VERSION,
            "provenance": create_provenance(RULESET_VERSION, {"tickers": [], "weights": {}}, as_of_date),
        }

    logger.info(f"Module 5 v2: Computing composite scores for {as_of_date}")

    # Enhancement data
    enhancement_applied = enhancement_result is not None
    pos_by_ticker = {}
    si_by_ticker = {}
    regime_adjustments = {}
    regime_name = "UNKNOWN"

    if enhancement_result:
        for ps in enhancement_result.get("pos_scores", {}).get("scores", []):
            if ps.get("ticker"):
                pos_by_ticker[ps["ticker"].upper()] = ps
        si_data = enhancement_result.get("short_interest_scores") or {}
        for si in si_data.get("scores", []):
            if si.get("ticker"):
                si_by_ticker[si["ticker"].upper()] = si
        regime_data = enhancement_result.get("regime", {})
        regime_name = regime_data.get("regime", "UNKNOWN")
        regime_adjustments = {k: _to_decimal(v, Decimal("1.0")) for k, v in regime_data.get("signal_adjustments", {}).items()}

    mode = ScoringMode.ENHANCED if pos_by_ticker else ScoringMode.DEFAULT
    base_weights = (ENHANCED_WEIGHTS.copy() if mode == ScoringMode.ENHANCED else DEFAULT_WEIGHTS.copy()) if weights is None else {k: _to_decimal(v, Decimal("0")) for k, v in weights.items()}

    # Index outputs
    active_tickers = {s["ticker"] for s in universe_result.get("active_securities", [])}
    financial_by_ticker = {s["ticker"]: s for s in financial_result.get("scores", [])}
    catalyst_by_ticker = catalyst_result.get("summaries", {})
    clinical_by_ticker = {s["ticker"]: s for s in clinical_result.get("scores", [])}

    combined = []
    excluded = []

    for ticker in active_tickers:
        fin = financial_by_ticker.get(ticker, {})
        cat = catalyst_by_ticker.get(ticker, {})
        clin = clinical_by_ticker.get(ticker, {})
        pos = pos_by_ticker.get(ticker.upper())
        si = si_by_ticker.get(ticker.upper())

        fin_score = _to_decimal(fin.get("financial_normalized"))
        clin_score = _to_decimal(clin.get("clinical_score"))
        pos_score = _to_decimal(pos.get("pos_score")) if pos else None

        if hasattr(cat, 'score_blended'):
            cat_score = _to_decimal(cat.score_blended)
        elif isinstance(cat, dict):
            scores = cat.get("scores", cat)
            cat_score = _to_decimal(scores.get("score_blended", scores.get("catalyst_score_net")))
        else:
            cat_score = None

        severities = [fin.get("severity", "none"), clin.get("severity", "none")]
        if hasattr(cat, 'severe_negative_flag') and cat.severe_negative_flag:
            severities.append("sev1")
        worst_severity = _get_worst_severity(severities)

        if worst_severity == Severity.SEV3:
            excluded.append({"ticker": ticker, "reason": "sev3_gate", "severity": worst_severity.value})
            continue

        combined.append({
            "ticker": ticker,
            "clinical_raw": clin_score,
            "financial_raw": fin_score,
            "catalyst_raw": cat_score,
            "pos_raw": pos_score,
            "market_cap_bucket": _market_cap_bucket(fin.get("market_cap_mm")),
            "stage_bucket": _stage_bucket(clin.get("lead_phase")),
            "fin_data": fin,
            "cat_data": cat,
            "clin_data": clin,
            "pos_data": pos,
            "si_data": si,
        })

    # Cohort grouping
    cohorts: Dict[str, List[Dict]] = {}
    for rec in combined:
        key = rec["stage_bucket"]
        rec["cohort_key"] = key
        cohorts.setdefault(key, []).append(rec)

    cohort_stats = {}
    for cohort_key, members in cohorts.items():
        if len(members) >= MIN_COHORT_SIZE:
            method = _apply_cohort_normalization_v2(members, include_pos=enhancement_applied)
            for m in members:
                m["normalization_method"] = method
        else:
            for m in members:
                m["clinical_normalized"] = m["clinical_raw"] or Decimal("0")
                m["financial_normalized"] = m["financial_raw"] or Decimal("0")
                m["catalyst_normalized"] = m["catalyst_raw"] or Decimal("0")
                m["pos_normalized"] = m.get("pos_raw") or Decimal("0")
                m["normalization_method"] = NormalizationMethod.GLOBAL_FALLBACK
        cohort_stats[cohort_key] = {"count": len(members)}

    # Score each ticker
    market_data_dict = market_data_by_ticker or {}
    scored = []
    for rec in combined:
        normalized_scores = {
            "clinical": rec.get("clinical_normalized"),
            "financial": rec.get("financial_normalized"),
            "catalyst": rec.get("catalyst_normalized"),
            "pos": rec.get("pos_normalized"),
        }
        ticker_market_data = market_data_dict.get(rec["ticker"])
        result = _score_single_ticker_v2(
            ticker=rec["ticker"], fin_data=rec["fin_data"], cat_data=rec["cat_data"],
            clin_data=rec["clin_data"], pos_data=rec["pos_data"], si_data=rec["si_data"],
            base_weights=base_weights, regime_adjustments=regime_adjustments,
            mode=mode, normalized_scores=normalized_scores,
            cohort_key=rec["cohort_key"], normalization_method=rec["normalization_method"],
            market_data=ticker_market_data,
        )
        result["market_cap_bucket"] = rec["market_cap_bucket"]
        result["stage_bucket"] = rec["stage_bucket"]
        result["cohort_key"] = rec["cohort_key"]
        scored.append(result)

    # Co-invest
    if coinvest_signals:
        as_of_dt = datetime.strptime(as_of_date, "%Y-%m-%d").date()
        for rec in scored:
            rec["coinvest"] = _enrich_with_coinvest(rec["ticker"], coinvest_signals, as_of_dt)
    else:
        for rec in scored:
            rec["coinvest"] = None

    # Sort and rank
    scored.sort(key=lambda x: (-x["composite_score"], -(x["coinvest"]["coinvest_overlap_count"] if x["coinvest"] else 0), x["ticker"]))
    for i, rec in enumerate(scored):
        rec["composite_rank"] = i + 1

    # Format output
    ranked_securities = []
    for rec in scored:
        bd = rec["score_breakdown"]
        coinvest = rec.get("coinvest") or {}
        ranked_securities.append({
            "ticker": rec["ticker"],
            "composite_score": str(rec["composite_score"]),
            "composite_rank": rec["composite_rank"],
            "severity": rec["severity"].value,
            "flags": rec["flags"],
            "rankable": True,
            "market_cap_bucket": rec["market_cap_bucket"],
            "stage_bucket": rec["stage_bucket"],
            "cohort_key": rec["cohort_key"],
            "normalization_method": rec["normalization_method"],
            "confidence_clinical": str(rec["confidence_clinical"]),
            "confidence_financial": str(rec["confidence_financial"]),
            "confidence_catalyst": str(rec["confidence_catalyst"]),
            "confidence_pos": str(rec["confidence_pos"]) if rec["confidence_pos"] else None,
            "effective_weights": {k: str(v) for k, v in rec["effective_weights"].items()},
            "monotonic_caps_applied": rec["caps_applied"],
            "determinism_hash": rec["determinism_hash"],
            "schema_version": SCHEMA_VERSION,
            "score_breakdown": {
                "version": bd.version, "mode": bd.mode,
                "base_weights": bd.base_weights, "regime_adjustments": bd.regime_adjustments,
                "effective_weights": bd.effective_weights, "components": bd.components,
                "penalties_and_gates": bd.penalties_and_gates, "bonuses": bd.bonuses,
                "final": bd.final, "normalization_method": bd.normalization_method,
                "cohort_info": bd.cohort_info, "hybrid_aggregation": bd.hybrid_aggregation,
            },
            "coinvest_overlap_count": coinvest.get("coinvest_overlap_count", 0),
            "coinvest_holders": coinvest.get("coinvest_holders", []),
            "coinvest_usable": coinvest.get("coinvest_usable", False),
            "volatility_adjustment": rec.get("volatility_adjustment"),
        })

    return {
        "as_of_date": as_of_date,
        "scoring_mode": mode.value,
        "weights_used": {k: str(v) for k, v in base_weights.items()},
        "ranked_securities": ranked_securities,
        "excluded_securities": excluded,
        "cohort_stats": cohort_stats,
        "diagnostic_counts": {
            "total_input": len(active_tickers),
            "rankable": len(ranked_securities),
            "excluded": len(excluded),
            "cohort_count": len(cohorts),
            "with_pos_scores": sum(1 for r in ranked_securities if r.get("confidence_pos")),
            "with_caps_applied": sum(1 for r in ranked_securities if r.get("monotonic_caps_applied")),
            "with_volatility_data": sum(
                1 for r in ranked_securities
                if r.get("volatility_adjustment", {}).get("annualized_vol_pct") is not None
            ),
            "high_volatility_count": sum(
                1 for r in ranked_securities
                if r.get("volatility_adjustment", {}).get("vol_bucket") == "high"
            ),
            "low_volatility_count": sum(
                1 for r in ranked_securities
                if r.get("volatility_adjustment", {}).get("vol_bucket") == "low"
            ),
        },
        "enhancement_applied": enhancement_applied,
        "enhancement_diagnostics": {
            "regime": regime_name,
            "regime_adjustments": {k: str(v) for k, v in regime_adjustments.items()},
        } if enhancement_applied else None,
        "schema_version": SCHEMA_VERSION,
        "provenance": create_provenance(
            RULESET_VERSION,
            {"tickers": list(active_tickers), "weights": {k: str(v) for k, v in base_weights.items()}},
            as_of_date,
        ),
    }
