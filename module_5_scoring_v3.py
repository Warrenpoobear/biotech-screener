#!/usr/bin/env python3
"""
Module 5 Scoring: Per-ticker scoring logic extracted from module_5_composite_v3.

This module contains:
- Types: MonotonicCap, ScoringMode, RunStatus, NormalizationMethod, ComponentScore, ScoreBreakdown, V3ScoringResult
- Helper functions: coalesce, bucket classifiers, confidence extractors, normalization
- Main scoring function: _score_single_ticker_v3

Extracted to reduce module_5_composite_v3.py complexity while preserving:
- Deterministic behavior (no logic changes during extraction)
- Import stability (re-exported from composite)
- Clean dependency graph (scoring does not import composite)

Author: Wake Robin Capital Management
Version: 3.0.0
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from common.types import Severity

# Import IC enhancement utilities
from src.modules.ic_enhancements import (
    # Core enhancement functions
    compute_volatility_adjustment,
    apply_volatility_to_score,
    compute_momentum_signal_with_fallback,
    compute_valuation_signal,
    compute_catalyst_decay,
    apply_catalyst_decay,
    compute_smart_money_signal,
    compute_interaction_terms,
    shrinkage_normalize,
    apply_regime_to_weights,
    # Types
    VolatilityAdjustment,
    VolatilityBucket,
    MomentumSignal,
    MultiWindowMomentumInput,
    ValuationSignal,
    CatalystDecayResult,
    SmartMoneySignal,
    InteractionTerms,
    # Helpers
    _to_decimal,
    _quantize_score,
    _quantize_weight,
    _clamp,
    EPS,
    SCORE_PRECISION,
)

# =============================================================================
# VERSION CONSTANTS (used by scoring for determinism hash)
# =============================================================================

SCHEMA_VERSION = "v3.0"

# =============================================================================
# SCORING CONSTANTS
# =============================================================================

# Severity multipliers
SEVERITY_MULTIPLIERS = {
    Severity.NONE: Decimal("1.0"),
    Severity.SEV1: Decimal("0.90"),  # 10% penalty
    Severity.SEV2: Decimal("0.50"),  # 50% penalty (soft gate)
    Severity.SEV3: Decimal("0.0"),   # Hard gate (excluded)
}

SEVERITY_GATE_LABELS = {
    Severity.NONE: "NONE",
    Severity.SEV1: "SEV1_10PCT",
    Severity.SEV2: "SEV2_HALF",
    Severity.SEV3: "SEV3_EXCLUDE",
}

# Cohort parameters
MIN_COHORT_SIZE = 5
MAX_UNCERTAINTY_PENALTY = Decimal("0.30")

# Winsorization bounds
WINSOR_LOW = Decimal("5")
WINSOR_HIGH = Decimal("95")

# Hybrid aggregation
HYBRID_ALPHA = Decimal("0.85")
CRITICAL_COMPONENTS = ["financial", "clinical"]

# PoS contribution cap
POS_DELTA_CAP = Decimal("6.0")

# Catalyst effective blending weights
CATALYST_WINDOW_WEIGHT = Decimal("0.70")
CATALYST_PROXIMITY_WEIGHT = Decimal("0.30")
CATALYST_DEFAULT_BASE = Decimal("0.85")
CATALYST_DEFAULT_SCORE = Decimal("50")

# Confidence gate threshold
CONFIDENCE_GATE_THRESHOLD = Decimal("0.4")


# =============================================================================
# TYPES
# =============================================================================

class MonotonicCap:
    LIQUIDITY_FAIL_CAP = Decimal("35")
    LIQUIDITY_WARN_CAP = Decimal("60")
    RUNWAY_CRITICAL_CAP = Decimal("40")
    RUNWAY_WARNING_CAP = Decimal("55")
    DILUTION_SEVERE_CAP = Decimal("45")
    DILUTION_HIGH_CAP = Decimal("60")


class ScoringMode(str, Enum):
    """Scoring mode based on available data."""
    DEFAULT = "default"
    PARTIAL = "partial"
    ENHANCED = "enhanced"
    ADAPTIVE = "adaptive"


class RunStatus(str, Enum):
    """Pipeline run status based on data coverage health."""
    OK = "OK"
    DEGRADED = "DEGRADED"
    FAIL = "FAIL"


class NormalizationMethod(str, Enum):
    """Normalization method applied."""
    COHORT = "cohort"
    COHORT_WINSORIZED = "cohort_winsorized"
    COHORT_SHRINKAGE = "cohort_shrinkage"
    STAGE_FALLBACK = "stage_fallback"
    GLOBAL_FALLBACK = "global_fallback"
    NONE = "none"


@dataclass
class ComponentScore:
    """Individual component score with full breakdown."""
    name: str
    raw: Optional[Decimal]
    normalized: Optional[Decimal]
    confidence: Decimal
    weight_base: Decimal
    weight_effective: Decimal
    contribution: Decimal
    decay_factor: Optional[Decimal] = None
    regime_adjustment: Optional[Decimal] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class ScoreBreakdown:
    """Complete score breakdown for auditability."""
    version: str
    mode: str
    base_weights: Dict[str, str]
    regime_adjustments: Dict[str, str]
    effective_weights: Dict[str, str]
    components: List[Dict[str, Any]]
    enhancements: Dict[str, Any]
    penalties_and_gates: Dict[str, Any]
    interaction_terms: Dict[str, Any]
    final: Dict[str, str]
    normalization_method: str
    cohort_info: Dict[str, Any]
    hybrid_aggregation: Dict[str, str]


@dataclass
class V3ScoringResult:
    """Complete v3 scoring result for a single ticker."""
    ticker: str
    composite_score: Decimal
    composite_rank: int
    clinical_normalized: Decimal
    financial_normalized: Decimal
    catalyst_normalized: Decimal
    momentum_normalized: Optional[Decimal]
    valuation_normalized: Optional[Decimal]
    pos_normalized: Optional[Decimal]
    vol_adjustment: VolatilityAdjustment
    catalyst_decay: CatalystDecayResult
    interaction_terms: InteractionTerms
    smart_money_signal: SmartMoneySignal
    severity: Severity
    confidence_overall: Decimal
    uncertainty_penalty: Decimal
    caps_applied: List[Dict[str, Any]]
    effective_weights: Dict[str, Decimal]
    cohort_key: str
    normalization_method: NormalizationMethod
    flags: List[str]
    determinism_hash: str
    score_breakdown: ScoreBreakdown


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _coalesce(*vals: Any, default: Optional[Any] = None) -> Any:
    """Return the first value that is not None.

    Unlike `or` chains, this correctly handles falsy values like Decimal("0")
    which should be treated as valid scores, not missing data.

    Args:
        *vals: Variable number of values to check
        default: Value to return if all vals are None

    Returns:
        First non-None value, or default if all are None
    """
    for v in vals:
        if v is not None:
            return v
    return default


def _compute_catalyst_effective(
    catalyst_score_window: Optional[Decimal],
    catalyst_proximity_score: Optional[Decimal],
) -> Tuple[Decimal, bool, str]:
    """Compute effective catalyst score by blending window and proximity scores."""
    has_window = catalyst_score_window is not None
    has_proximity = catalyst_proximity_score is not None and catalyst_proximity_score > Decimal("0")

    if has_window and has_proximity:
        effective = (
            CATALYST_WINDOW_WEIGHT * catalyst_score_window +
            CATALYST_PROXIMITY_WEIGHT * catalyst_proximity_score
        )
        proximity_blended = True
        blend_mode = "full_blend"
    elif has_proximity and not has_window:
        effective = (
            CATALYST_DEFAULT_BASE * CATALYST_DEFAULT_SCORE +
            (Decimal("1") - CATALYST_DEFAULT_BASE) * catalyst_proximity_score
        )
        proximity_blended = True
        blend_mode = "proximity_only"
    elif has_window and not has_proximity:
        effective = catalyst_score_window
        proximity_blended = False
        blend_mode = "window_only"
    else:
        effective = CATALYST_DEFAULT_SCORE
        proximity_blended = False
        blend_mode = "default"

    effective = _clamp(effective, Decimal("0"), Decimal("100"))
    effective = _quantize_score(effective)

    return (effective, proximity_blended, blend_mode)


def _market_cap_bucket(market_cap_mm: Optional[Any]) -> str:
    """Classify market cap into bucket."""
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
    """Classify development stage into bucket."""
    if not lead_phase:
        return "early"
    phase = lead_phase.lower()
    if "3" in phase or "approved" in phase:
        return "late"
    elif "2" in phase:
        return "mid"
    return "early"


def _quarter_from_date(d: date) -> str:
    """Convert date to quarter string."""
    q = (d.month - 1) // 3 + 1
    return f"{d.year}Q{q}"


def _get_worst_severity(severities: List[str]) -> Severity:
    """Get worst severity from list."""
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


def _extract_confidence_financial(fin_data: Dict) -> Decimal:
    """Extract confidence for financial score."""
    conf = _to_decimal(fin_data.get("confidence"))
    if conf is not None:
        return _clamp(conf, Decimal("0"), Decimal("1"))
    state = fin_data.get("financial_data_state", "NONE")
    return {"FULL": Decimal("1.0"), "PARTIAL": Decimal("0.7"),
            "MINIMAL": Decimal("0.4"), "NONE": Decimal("0.1")}.get(state, Decimal("0.5"))


def _extract_confidence_clinical(clin_data: Dict) -> Decimal:
    """Extract confidence for clinical score."""
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
    """Extract confidence for catalyst score."""
    if not cat_data:
        return Decimal("0.3")

    CONFIDENCE_MAP = {
        "HIGH": Decimal("0.70"),
        "MED": Decimal("0.50"),
        "LOW": Decimal("0.35"),
        "UNKNOWN": Decimal("0.30"),
    }

    if hasattr(cat_data, 'catalyst_confidence') and hasattr(cat_data.catalyst_confidence, 'value'):
        conf_str = cat_data.catalyst_confidence.value
        if conf_str in CONFIDENCE_MAP:
            return CONFIDENCE_MAP[conf_str]

    if isinstance(cat_data, dict):
        integration = cat_data.get("integration", {})
        conf_str = integration.get("catalyst_confidence")
        if conf_str and conf_str in CONFIDENCE_MAP:
            return CONFIDENCE_MAP[conf_str]

        event_summary = cat_data.get("event_summary", {})
        events_total = event_summary.get("events_total", 0)
        if events_total > 0 and conf_str is None:
            return Decimal("0.50")

    if hasattr(cat_data, 'n_high_confidence'):
        n_high = cat_data.n_high_confidence
        n_events = cat_data.events_detected_total
        if n_events > 0:
            high_ratio = Decimal(str(n_high)) / Decimal(str(max(n_events, 1)))
            return _clamp(Decimal("0.4") + high_ratio * Decimal("0.5"), Decimal("0"), Decimal("1"))

    if isinstance(cat_data, dict):
        scores = cat_data.get("scores", cat_data)
        n_high = scores.get("n_high_confidence", 0)
        n_events = scores.get("events_detected_total", 0)
        if n_events > 0:
            high_ratio = Decimal(str(n_high)) / Decimal(str(max(n_events, 1)))
            return _clamp(Decimal("0.4") + high_ratio * Decimal("0.5"), Decimal("0"), Decimal("1"))

    return Decimal("0.3")


def _extract_confidence_pos(pos_data: Optional[Dict]) -> Decimal:
    """Extract confidence for PoS score."""
    if not pos_data:
        return Decimal("0")
    conf = _to_decimal(pos_data.get("confidence"))
    if conf is not None:
        return _clamp(conf, Decimal("0"), Decimal("1"))
    return Decimal("0.7") if pos_data.get("pos_score") is not None else Decimal("0")


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


def _compute_determinism_hash(
    ticker: str,
    version: str,
    mode: str,
    base_weights: Dict[str, Decimal],
    effective_weights: Dict[str, Decimal],
    component_scores: List[ComponentScore],
    enhancements: Dict[str, Any],
    final_score: Decimal,
) -> str:
    """Compute deterministic hash for audit trail."""
    payload = {
        "ticker": ticker,
        "version": version,
        "mode": mode,
        "base_weights": {k: str(v) for k, v in sorted(base_weights.items())},
        "effective_weights": {k: str(v) for k, v in sorted(effective_weights.items())},
        "components": sorted([
            {"name": c.name, "raw": str(c.raw) if c.raw is not None else None,
             "normalized": str(c.normalized) if c.normalized is not None else None,
             "contribution": str(c.contribution)}
            for c in component_scores
        ], key=lambda x: x["name"]),
        "enhancements": {k: str(v) if isinstance(v, Decimal) else v for k, v in sorted(enhancements.items())},
        "final_score": str(final_score),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _enrich_with_coinvest(ticker: str, coinvest_signals: dict, as_of_date: date) -> dict:
    """Look up co-invest signal and return overlay fields with PIT safety."""
    signal = coinvest_signals.get(ticker)
    if not signal:
        return {
            "coinvest_overlap_count": 0,
            "coinvest_holders": [],
            "coinvest_usable": False,
            "position_changes": {},
            "holder_tiers": {},
        }

    if isinstance(signal, dict) and "coinvest_overlap_count" in signal:
        raw_tiers = signal.get("holder_tiers", {})
        normalized_tiers = {}
        for holder_id, tier_info in raw_tiers.items():
            if isinstance(tier_info, dict):
                normalized_tiers[holder_id] = tier_info.get("tier", 2)
            elif isinstance(tier_info, int):
                normalized_tiers[holder_id] = tier_info
            else:
                normalized_tiers[holder_id] = 2

        return {
            "coinvest_overlap_count": signal.get("coinvest_overlap_count", 0),
            "coinvest_holders": signal.get("coinvest_holders", []),
            "coinvest_usable": signal.get("coinvest_overlap_count", 0) > 0,
            "position_changes": signal.get("position_changes", {}),
            "holder_tiers": normalized_tiers,
        }

    if not hasattr(signal, 'positions'):
        return {
            "coinvest_overlap_count": 0,
            "coinvest_holders": [],
            "coinvest_usable": False,
            "position_changes": {},
            "holder_tiers": {},
        }

    pit_positions = [p for p in signal.positions if p.filing_date < as_of_date]
    if not pit_positions:
        return {
            "coinvest_overlap_count": 0,
            "coinvest_holders": [],
            "coinvest_usable": False,
            "position_changes": {},
            "holder_tiers": {},
        }

    holders = sorted(set(p.manager_name for p in pit_positions))

    holder_tiers: Dict[str, int] = {}
    for p in pit_positions:
        manager_name = p.manager_name
        if manager_name not in holder_tiers:
            tier = getattr(p, 'manager_tier', None)
            if tier is not None:
                holder_tiers[manager_name] = tier

    position_changes = {}
    for p in pit_positions:
        change_type = getattr(p, 'change_type', None)
        if change_type is not None:
            position_changes[p.manager_name] = change_type

    return {
        "coinvest_overlap_count": len(holders),
        "coinvest_holders": holders,
        "coinvest_quarter": _quarter_from_date(pit_positions[0].report_date),
        "coinvest_usable": True,
        "position_changes": position_changes,
        "holder_tiers": holder_tiers,
    }


def _apply_cohort_normalization_v3(
    members: List[Dict],
    global_stats: Dict[str, Tuple[Decimal, Decimal]],
    include_pos: bool = False,
    use_shrinkage: bool = True,
) -> NormalizationMethod:
    """Apply V3 cohort normalization with optional shrinkage."""
    if not members:
        return NormalizationMethod.NONE

    n = len(members)

    clin_scores = [m["clinical_raw"] or Decimal("0") for m in members]
    fin_scores = [m["financial_raw"] or Decimal("0") for m in members]
    cat_scores = [m["catalyst_raw"] or Decimal("0") for m in members]

    if n >= MIN_COHORT_SIZE:
        if use_shrinkage and n < 20:
            clin_mean, clin_std = global_stats.get("clinical", (Decimal("50"), Decimal("20")))
            fin_mean, fin_std = global_stats.get("financial", (Decimal("50"), Decimal("20")))
            cat_mean, cat_std = global_stats.get("catalyst", (Decimal("50"), Decimal("20")))

            clin_norm, _ = shrinkage_normalize(clin_scores, clin_mean, clin_std)
            fin_norm, _ = shrinkage_normalize(fin_scores, fin_mean, fin_std)
            cat_norm, _ = shrinkage_normalize(cat_scores, cat_mean, cat_std)

            method = NormalizationMethod.COHORT_SHRINKAGE
        else:
            clin_norm, clin_w = _rank_normalize_winsorized(clin_scores)
            fin_norm, fin_w = _rank_normalize_winsorized(fin_scores)
            cat_norm, cat_w = _rank_normalize_winsorized(cat_scores)

            method = NormalizationMethod.COHORT_WINSORIZED if (clin_w or fin_w or cat_w) else NormalizationMethod.COHORT
    else:
        clin_mean, clin_std = global_stats.get("clinical", (Decimal("50"), Decimal("20")))
        fin_mean, fin_std = global_stats.get("financial", (Decimal("50"), Decimal("20")))
        cat_mean, cat_std = global_stats.get("catalyst", (Decimal("50"), Decimal("20")))

        clin_norm, _ = shrinkage_normalize(clin_scores, clin_mean, clin_std)
        fin_norm, _ = shrinkage_normalize(fin_scores, fin_mean, fin_std)
        cat_norm, _ = shrinkage_normalize(cat_scores, cat_mean, cat_std)

        method = NormalizationMethod.GLOBAL_FALLBACK

    pos_norm = None
    if include_pos:
        pos_scores = [m.get("pos_raw") or Decimal("0") for m in members]
        if any(p > 0 for p in pos_scores):
            if n >= MIN_COHORT_SIZE:
                pos_norm, _ = _rank_normalize_winsorized(pos_scores)
            else:
                pos_mean, pos_std = global_stats.get("pos", (Decimal("50"), Decimal("20")))
                pos_norm, _ = shrinkage_normalize(pos_scores, pos_mean, pos_std)

    for i, m in enumerate(members):
        m["clinical_normalized"] = clin_norm[i]
        m["financial_normalized"] = fin_norm[i]
        m["catalyst_normalized"] = cat_norm[i]
        m["pos_normalized"] = pos_norm[i] if pos_norm else (m.get("pos_raw") or Decimal("0"))
        m["normalization_method"] = method

    return method


def _compute_global_stats(combined: List[Dict]) -> Dict[str, Tuple[Decimal, Decimal]]:
    """Compute global mean and std for each component."""
    stats = {}

    for component in ["clinical", "financial", "catalyst", "pos"]:
        key = f"{component}_raw"
        values = [_to_decimal(m.get(key)) for m in combined if m.get(key) is not None]

        if values:
            mean = sum(values) / Decimal(len(values))
            variance = sum((v - mean) ** 2 for v in values) / Decimal(len(values))
            std = variance.sqrt() if variance > 0 else Decimal("1")
            stats[component] = (mean, std)
        else:
            stats[component] = (Decimal("50"), Decimal("20"))

    return stats


# =============================================================================
# MAIN SCORING FUNCTION
# =============================================================================

def _score_single_ticker_v3(
    ticker: str,
    fin_data: Dict,
    cat_data: Any,
    clin_data: Dict,
    pos_data: Optional[Dict],
    si_data: Optional[Dict],
    market_data: Optional[Dict],
    coinvest_data: Dict,
    base_weights: Dict[str, Decimal],
    regime: str,
    mode: ScoringMode,
    normalized_scores: Dict[str, Optional[Decimal]],
    cohort_key: str,
    normalization_method: NormalizationMethod,
    peer_valuations: List[Dict],
) -> Dict[str, Any]:
    """Score a single ticker with all v3 enhancements."""

    flags = []

    # Extract raw scores
    fin_raw = _to_decimal(_coalesce(fin_data.get("financial_normalized"), fin_data.get("financial_score")))
    clin_raw = _to_decimal(clin_data.get("clinical_score"))
    pos_raw = _to_decimal(pos_data.get("pos_score")) if pos_data else None

    # Extract catalyst scores and metadata
    if hasattr(cat_data, 'score_blended'):
        cat_window = _to_decimal(cat_data.score_blended)
        cat_proximity = _to_decimal(getattr(cat_data, 'catalyst_proximity_score', 0)) or Decimal("0")
        cat_delta = _to_decimal(getattr(cat_data, 'catalyst_delta_score', 0)) or Decimal("0")
        days_to_cat = _coalesce(
            getattr(cat_data, 'catalyst_window_days', None),
            getattr(cat_data, 'days_to_nearest_catalyst', None),
        )
        cat_event_type = getattr(cat_data, 'nearest_catalyst_type', "DEFAULT")
    elif isinstance(cat_data, dict):
        scores = cat_data.get("scores", cat_data)
        cat_window = _to_decimal(scores.get("score_blended", scores.get("catalyst_score_net")))
        cat_proximity = _to_decimal(scores.get("catalyst_proximity_score", 0)) or Decimal("0")
        cat_delta = _to_decimal(scores.get("catalyst_delta_score", 0)) or Decimal("0")
        integration = cat_data.get("integration", {})
        days_to_cat = _coalesce(
            integration.get("catalyst_window_days"),
            scores.get("days_to_nearest_catalyst"),
            scores.get("catalyst_window_days"),
        )
        cat_event_type = scores.get("nearest_catalyst_type", "DEFAULT")
    else:
        cat_window = None
        cat_proximity = Decimal("0")
        cat_delta = Decimal("0")
        days_to_cat = None
        cat_event_type = "DEFAULT"

    # Compute effective catalyst score
    cat_effective, cat_proximity_blended, cat_blend_mode = _compute_catalyst_effective(
        cat_window, cat_proximity
    )
    cat_raw = cat_effective
    if cat_proximity_blended:
        flags.append("catalyst_proximity_blended")

    # Get normalized scores
    clin_norm = _coalesce(normalized_scores.get("clinical"), clin_raw, default=Decimal("50"))
    fin_norm = _coalesce(normalized_scores.get("financial"), fin_raw, default=Decimal("50"))
    cat_norm = _coalesce(normalized_scores.get("catalyst"), cat_raw, default=Decimal("50"))
    pos_norm = _coalesce(normalized_scores.get("pos"), pos_raw, default=Decimal("0"))

    # Extract financial metadata
    runway_months = _to_decimal(fin_data.get("runway_months"))
    liquidity_status = fin_data.get("liquidity_gate_status")
    dilution_bucket = fin_data.get("dilution_risk_bucket")
    market_cap_mm = _to_decimal(fin_data.get("market_cap_mm"))

    # Get stage and trial count
    lead_phase = clin_data.get("lead_phase")
    stage = _stage_bucket(lead_phase)
    trial_count = clin_data.get("trial_count", 0)

    # Extract market data
    annualized_vol = None
    momentum_input = MultiWindowMomentumInput()
    momentum_source = "prices"

    if market_data:
        vol_val = market_data.get("volatility_252d")
        annualized_vol = _to_decimal(vol_val if vol_val is not None else market_data.get("annualized_volatility"))

        if market_data.get("_momentum_source") == "13f":
            momentum_source = "13f"

        momentum_input = MultiWindowMomentumInput(
            return_20d=_to_decimal(market_data.get("return_20d")),
            return_60d=_to_decimal(market_data.get("return_60d")),
            return_120d=_to_decimal(market_data.get("return_120d")),
            benchmark_20d=_to_decimal(market_data.get("xbi_return_20d")),
            benchmark_60d=_to_decimal(market_data.get("xbi_return_60d") or market_data.get("benchmark_return_60d")),
            benchmark_120d=_to_decimal(market_data.get("xbi_return_120d")),
            trading_days_available=market_data.get("trading_days_available"),
            annualized_vol=annualized_vol,
        )

    # =========================================================================
    # COMPUTE ALL ENHANCEMENTS
    # =========================================================================

    # 1. Volatility adjustment
    vol_adj = compute_volatility_adjustment(annualized_vol)
    if vol_adj.vol_bucket == VolatilityBucket.HIGH:
        flags.append("high_volatility")
    elif vol_adj.vol_bucket == VolatilityBucket.LOW:
        flags.append("low_volatility_boost")

    # 2. Momentum signal
    momentum = compute_momentum_signal_with_fallback(
        momentum_input,
        use_vol_adjusted_alpha=False,
    )
    momentum_norm = momentum.momentum_score

    if momentum.data_status == "missing_prices":
        flags.append("momentum_missing_prices")
    elif momentum.data_status == "computed_low_conf":
        flags.append("momentum_low_confidence")

    if momentum.alpha_60d is not None:
        if momentum.alpha_60d >= Decimal("0.10"):
            flags.append("strong_positive_momentum")
        elif momentum.alpha_60d <= Decimal("-0.10"):
            flags.append("strong_negative_momentum")

    if momentum.window_used:
        flags.append(f"momentum_window_{momentum.window_used}d")

    if momentum.data_completeness < Decimal("0.5"):
        flags.append("momentum_data_incomplete")

    # 3. Valuation signal
    valuation = compute_valuation_signal(market_cap_mm, trial_count, lead_phase, peer_valuations)
    valuation_norm = valuation.valuation_score
    if valuation.valuation_score >= Decimal("70"):
        flags.append("undervalued_vs_peers")
    elif valuation.valuation_score <= Decimal("30"):
        flags.append("overvalued_vs_peers")

    # 4. Catalyst decay
    decay = compute_catalyst_decay(days_to_cat, cat_event_type or "DEFAULT")
    if decay.in_optimal_window:
        flags.append("catalyst_optimal_window")

    cat_norm_decayed = apply_catalyst_decay(cat_norm, decay)

    # 5. Smart money signal
    smart_money = compute_smart_money_signal(
        coinvest_data.get("coinvest_overlap_count", 0),
        coinvest_data.get("coinvest_holders", []),
        coinvest_data.get("position_changes"),
        coinvest_data.get("holder_tiers"),
    )
    if smart_money.holders_increasing:
        flags.append("smart_money_buying")
    if smart_money.holders_decreasing:
        flags.append("smart_money_selling")
    if smart_money.tier1_holders:
        flags.append("smart_money_tier1_present")

    # 6. Interaction terms
    if runway_months is None:
        runway_gate = "UNKNOWN"
    elif runway_months < Decimal("6"):
        runway_gate = "FAIL"
    elif runway_months >= Decimal("12"):
        runway_gate = "PASS"
    else:
        runway_gate = "UNKNOWN"
    dilution_gate = "PASS" if dilution_bucket in ("LOW", "MEDIUM") else (
        "FAIL" if dilution_bucket in ("HIGH", "SEVERE") else "UNKNOWN"
    )

    interactions = compute_interaction_terms(
        clin_norm,
        fin_data,
        cat_norm,
        stage,
        vol_adj,
        runway_gate_status=runway_gate,
        dilution_gate_status=dilution_gate,
    )
    flags.extend(interactions.interaction_flags)

    # =========================================================================
    # CONFIDENCE EXTRACTION
    # =========================================================================

    conf_clin = _extract_confidence_clinical(clin_data)
    conf_fin = _extract_confidence_financial(fin_data)
    conf_cat = _extract_confidence_catalyst(cat_data)
    conf_pos = _extract_confidence_pos(pos_data)

    conf_clin = _clamp(conf_clin - vol_adj.confidence_penalty, Decimal("0.1"), Decimal("1"))
    conf_fin = _clamp(conf_fin - vol_adj.confidence_penalty, Decimal("0.1"), Decimal("1"))
    conf_cat = _clamp(conf_cat - vol_adj.confidence_penalty, Decimal("0.1"), Decimal("1"))

    # =========================================================================
    # WEIGHT COMPUTATION
    # =========================================================================

    regime_weights = apply_regime_to_weights(base_weights, regime)

    vol_adjusted_weights = {
        k: v * vol_adj.weight_adjustment_factor
        for k, v in regime_weights.items()
    }

    confidences = {
        "clinical": conf_clin,
        "financial": conf_fin,
        "catalyst": conf_cat,
        "pos": conf_pos,
        "momentum": momentum.confidence,
        "valuation": valuation.confidence,
    }

    gated_components = []

    effective_weights = {}
    for comp, base_w in vol_adjusted_weights.items():
        conf = confidences.get(comp, Decimal("0.5"))

        if conf < CONFIDENCE_GATE_THRESHOLD:
            effective_weights[comp] = Decimal("0")
            gated_components.append(comp)
            flags.append(f"{comp}_confidence_gated")
        else:
            eff_w = base_w * (Decimal("0.5") + conf * Decimal("0.5"))
            effective_weights[comp] = eff_w

    total = sum(effective_weights.values())
    if total > EPS:
        effective_weights = {k: _quantize_weight(v / total) for k, v in effective_weights.items()}

    # =========================================================================
    # COMPOSITE SCORE COMPUTATION
    # =========================================================================

    component_scores = []
    contributions = {}

    core_scores = {
        "clinical": (clin_norm, clin_raw, conf_clin),
        "financial": (fin_norm, fin_raw, conf_fin),
        "catalyst": (cat_norm_decayed, cat_raw, conf_cat),
    }

    for name, (norm, raw, conf) in core_scores.items():
        w_eff = effective_weights.get(name, Decimal("0"))
        contrib = norm * w_eff
        contributions[name] = contrib

        notes = []
        if raw is None:
            notes.append("missing_raw")
        if name == "catalyst" and decay.decay_factor < Decimal("0.9"):
            notes.append(f"decay_factor_{decay.decay_factor}")

        component_scores.append(ComponentScore(
            name=name,
            raw=raw,
            normalized=norm,
            confidence=conf,
            weight_base=base_weights.get(name, Decimal("0")),
            weight_effective=w_eff,
            contribution=_quantize_score(contrib),
            decay_factor=decay.decay_factor if name == "catalyst" else None,
            notes=notes,
        ))

    if mode in (ScoringMode.ENHANCED, ScoringMode.PARTIAL):
        if "momentum" in effective_weights:
            w_eff = effective_weights["momentum"]
            contrib = momentum_norm * w_eff
            contributions["momentum"] = contrib
            component_scores.append(ComponentScore(
                name="momentum",
                raw=momentum.alpha_60d,
                normalized=momentum_norm,
                confidence=momentum.confidence,
                weight_base=base_weights.get("momentum", Decimal("0")),
                weight_effective=w_eff,
                contribution=_quantize_score(contrib),
                notes=[] if momentum.alpha_60d is not None else ["missing_price_data"],
            ))

        if "valuation" in effective_weights:
            w_eff = effective_weights["valuation"]
            contrib = valuation_norm * w_eff
            contributions["valuation"] = contrib
            component_scores.append(ComponentScore(
                name="valuation",
                raw=valuation.mcap_per_asset,
                normalized=valuation_norm,
                confidence=valuation.confidence,
                weight_base=base_weights.get("valuation", Decimal("0")),
                weight_effective=w_eff,
                contribution=_quantize_score(contrib),
                notes=[] if valuation.peer_count >= 5 else ["insufficient_peers"],
            ))

    pos_contrib_raw = Decimal("0")
    pos_contrib_capped = Decimal("0")
    pos_delta_was_capped = False

    if mode == ScoringMode.ENHANCED and pos_raw is not None:
        w_eff = effective_weights.get("pos", Decimal("0"))
        pos_contrib_raw = pos_norm * w_eff

        pos_contrib_capped = _clamp(pos_contrib_raw, -POS_DELTA_CAP, POS_DELTA_CAP)

        if pos_contrib_raw != pos_contrib_capped:
            pos_delta_was_capped = True
            flags.append("pos_delta_capped")

        contributions["pos"] = pos_contrib_capped

        component_scores.append(ComponentScore(
            name="pos",
            raw=pos_raw,
            normalized=pos_norm,
            confidence=conf_pos,
            weight_base=base_weights.get("pos", Decimal("0")),
            weight_effective=w_eff,
            contribution=_quantize_score(pos_contrib_capped),
            notes=["delta_capped"] if pos_delta_was_capped else [],
        ))
        flags.append("pos_score_applied")

    # =========================================================================
    # AGGREGATION
    # =========================================================================

    weighted_sum = sum(contributions.values())

    critical_scores = []
    for c in CRITICAL_COMPONENTS:
        if c in contributions and c in effective_weights and effective_weights[c] > EPS:
            underlying = contributions[c] / effective_weights[c]
            critical_scores.append(underlying)

    min_critical = min(critical_scores) if critical_scores else weighted_sum

    pre_penalty = HYBRID_ALPHA * weighted_sum + (Decimal("1") - HYBRID_ALPHA) * min_critical

    pre_penalty = pre_penalty + interactions.total_interaction_adjustment
    pre_penalty = _quantize_score(pre_penalty)

    # =========================================================================
    # PENALTIES AND CAPS
    # =========================================================================

    subfactors = [clin_raw, fin_raw, cat_raw]
    if mode == ScoringMode.ENHANCED:
        subfactors.append(pos_raw)
    missing_count = sum(1 for x in subfactors if x is None)
    missing_pct = Decimal(str(missing_count / len(subfactors)))
    uncertainty_penalty = min(MAX_UNCERTAINTY_PENALTY, missing_pct)

    post_uncertainty = pre_penalty * (Decimal("1") - uncertainty_penalty)

    severities = [fin_data.get("severity", "none"), clin_data.get("severity", "none")]
    if hasattr(cat_data, 'severe_negative_flag') and cat_data.severe_negative_flag:
        severities.append("sev1")
    elif isinstance(cat_data, dict):
        flags_dict = cat_data.get("flags", {})
        if isinstance(flags_dict, dict) and flags_dict.get("severe_negative_flag"):
            severities.append("sev1")
    worst_severity = _get_worst_severity(severities)
    severity_multiplier = SEVERITY_MULTIPLIERS[worst_severity]

    post_severity = post_uncertainty * severity_multiplier
    post_severity = _quantize_score(post_severity)

    post_cap, caps_applied = _apply_monotonic_caps(
        post_severity,
        liquidity_status,
        runway_months,
        dilution_bucket
    )

    post_vol = apply_volatility_to_score(post_cap, vol_adj)

    delta_bonus = (cat_delta / Decimal("25")).quantize(SCORE_PRECISION)

    final_score = _clamp(post_vol + delta_bonus, Decimal("0"), Decimal("100"))
    final_score = _quantize_score(final_score)

    # =========================================================================
    # BUILD OUTPUT
    # =========================================================================

    if uncertainty_penalty > 0:
        flags.append("uncertainty_penalty_applied")
    if worst_severity == Severity.SEV2:
        flags.append("sev2_penalty_applied")
    elif worst_severity == Severity.SEV1:
        flags.append("sev1_penalty_applied")
    if caps_applied:
        flags.append("monotonic_cap_applied")

    enhancements_dict = {
        "momentum_score": momentum.momentum_score,
        "valuation_score": valuation.valuation_score,
        "smart_money_score": smart_money.smart_money_score,
        "vol_bucket": vol_adj.vol_bucket.value,
        "catalyst_decay": decay.decay_factor,
        "interaction_adjustment": interactions.total_interaction_adjustment,
    }

    determinism_hash = _compute_determinism_hash(
        ticker, SCHEMA_VERSION, mode.value, base_weights, effective_weights,
        component_scores, enhancements_dict, final_score
    )

    confidence_overall = (
        conf_clin * Decimal("0.3") +
        conf_fin * Decimal("0.3") +
        conf_cat * Decimal("0.2") +
        momentum.confidence * Decimal("0.1") +
        valuation.confidence * Decimal("0.1")
    )
    confidence_overall = _clamp(confidence_overall, Decimal("0.1"), Decimal("0.9"))

    breakdown = ScoreBreakdown(
        version=SCHEMA_VERSION,
        mode=mode.value,
        base_weights={k: str(v) for k, v in base_weights.items()},
        regime_adjustments={"regime": regime},
        effective_weights={k: str(v) for k, v in effective_weights.items()},
        components=[{
            "name": c.name, "raw": str(c.raw) if c.raw is not None else None,
            "normalized": str(c.normalized) if c.normalized is not None else None,
            "confidence": str(c.confidence), "weight_base": str(c.weight_base),
            "weight_effective": str(c.weight_effective), "contribution": str(c.contribution),
            "decay_factor": str(c.decay_factor) if c.decay_factor is not None else None,
            "notes": c.notes,
        } for c in component_scores],
        enhancements={
            "momentum": {"score": str(momentum.momentum_score), "alpha_60d": str(momentum.alpha_60d) if momentum.alpha_60d else None, "confidence": str(momentum.confidence)},
            "valuation": {"score": str(valuation.valuation_score), "peer_count": valuation.peer_count, "confidence": str(valuation.confidence)},
            "smart_money": {"score": str(smart_money.smart_money_score), "overlap_count": smart_money.overlap_count},
            "volatility": {"bucket": vol_adj.vol_bucket.value, "weight_factor": str(vol_adj.weight_adjustment_factor), "score_factor": str(vol_adj.score_adjustment_factor)},
            "catalyst_decay": {"factor": str(decay.decay_factor), "in_optimal_window": decay.in_optimal_window},
        },
        penalties_and_gates={
            "uncertainty_penalty_pct": str(_quantize_score(uncertainty_penalty * Decimal("100"))),
            "severity_gate": SEVERITY_GATE_LABELS[worst_severity],
            "severity_multiplier": str(severity_multiplier),
            "monotonic_caps_applied": caps_applied,
        },
        interaction_terms={
            "clinical_financial_synergy": str(interactions.clinical_financial_synergy),
            "stage_financial_interaction": str(interactions.stage_financial_interaction),
            "catalyst_volatility_dampening": str(interactions.catalyst_volatility_dampening),
            "total_adjustment": str(interactions.total_interaction_adjustment),
            "flags": interactions.interaction_flags,
        },
        final={
            "pre_penalty_score": str(pre_penalty),
            "post_uncertainty_score": str(_quantize_score(post_uncertainty)),
            "post_severity_score": str(post_severity),
            "post_cap_score": str(post_cap),
            "post_vol_score": str(post_vol),
            "delta_bonus": str(delta_bonus),
            "composite_score": str(final_score),
        },
        normalization_method=normalization_method.value,
        cohort_info={"cohort_key": cohort_key},
        hybrid_aggregation={
            "alpha": str(HYBRID_ALPHA),
            "weighted_sum": str(_quantize_score(weighted_sum)),
            "min_critical": str(_quantize_score(min_critical)),
        },
    )

    return {
        "ticker": ticker,
        "composite_score": final_score,
        "severity": worst_severity,
        "flags": sorted(set(flags)),
        "determinism_hash": determinism_hash,
        "score_breakdown": breakdown,
        "confidence_clinical": conf_clin,
        "confidence_financial": conf_fin,
        "confidence_catalyst": conf_cat,
        "confidence_pos": conf_pos if mode == ScoringMode.ENHANCED else None,
        "confidence_overall": confidence_overall,
        "effective_weights": effective_weights,
        "normalization_method": normalization_method.value,
        "caps_applied": caps_applied,
        "component_scores": component_scores,
        "uncertainty_penalty": uncertainty_penalty,
        "momentum_signal": {
            "momentum_score": str(momentum.momentum_score),
            "alpha_60d": str(momentum.alpha_60d) if momentum.alpha_60d else None,
            "confidence": str(momentum.confidence),
            "data_completeness": str(momentum.data_completeness),
            "window_used": momentum.window_used,
            "data_status": momentum.data_status,
            "source": momentum_source,
        },
        "valuation_signal": {
            "score": str(valuation.valuation_score),
            "peer_count": valuation.peer_count,
            "confidence": str(valuation.confidence),
        },
        "smart_money_signal": {
            "score": str(smart_money.smart_money_score),
            "overlap_count": smart_money.overlap_count,
            "weighted_overlap": str(smart_money.weighted_overlap),
            "overlap_bonus": str(smart_money.overlap_bonus),
            "change_adjustment": str(smart_money.position_change_adjustment),
            "confidence": str(smart_money.confidence),
            "tier_breakdown": smart_money.tier_breakdown,
            "tier1_holders": smart_money.tier1_holders,
            "holders_increasing": smart_money.holders_increasing,
            "holders_decreasing": smart_money.holders_decreasing,
        },
        "volatility_adjustment": {
            "annualized_vol_pct": str(vol_adj.annualized_vol) if vol_adj.annualized_vol else None,
            "vol_bucket": vol_adj.vol_bucket.value,
            "weight_factor": str(vol_adj.weight_adjustment_factor),
            "score_factor": str(vol_adj.score_adjustment_factor),
        },
        "catalyst_decay": {
            "factor": str(decay.decay_factor),
            "days_to_catalyst": decay.days_to_catalyst,
            "in_optimal_window": decay.in_optimal_window,
        },
        "catalyst_effective": {
            "catalyst_score_window": str(cat_window) if cat_window is not None else None,
            "catalyst_proximity_score": str(cat_proximity) if cat_proximity else "0",
            "catalyst_score_effective": str(cat_effective),
            "catalyst_proximity_blended": cat_proximity_blended,
            "blend_mode": cat_blend_mode,
        },
        "interaction_terms": {
            "total_adjustment": str(interactions.total_interaction_adjustment),
            "flags": interactions.interaction_flags,
        },
    }
