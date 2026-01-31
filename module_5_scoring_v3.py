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

# Import Financial Module 2: Survivability scoring
from financial_module_2_survivability import compute_survivability_score

# Import IC enhancement utilities
from src.modules.ic_enhancements import (
    # Core enhancement functions
    compute_volatility_adjustment,
    apply_volatility_to_score,
    compute_momentum_signal_with_fallback,
    compute_momentum_signal_multiwindow,
    compute_valuation_signal,
    compute_catalyst_decay,
    apply_catalyst_decay,
    compute_smart_money_signal,
    compute_interaction_terms,
    shrinkage_normalize,
    apply_regime_to_weights,
    detect_contradictions,  # Enhancement 6: Contradiction detector
    # Types
    VolatilityAdjustment,
    VolatilityBucket,
    MomentumSignal,
    MultiWindowMomentumInput,
    ValuationSignal,
    ValuationRegime,
    CatalystDecayResult,
    SmartMoneySignal,
    InteractionTerms,
    ContradictionResult,  # Enhancement 6: Contradiction result type
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
# ENHANCEMENT 1: HARD REGIME GATING CONFIGURATION
# =============================================================================

REGIME_GATE_CONFIG = {
    "BEAR": {
        "momentum_cap_pct": Decimal("0.30"),      # Cap momentum at 30% of deviation from neutral
        "valuation_upside_cap": Decimal("55"),    # Cap valuation score at 55
        "financial_penalty_mult": Decimal("1.25"), # Amplify financial penalties 25%
    },
    "BULL": {
        "momentum_cap_pct": Decimal("1.0"),       # Full momentum
        "catalyst_boost": Decimal("1.15"),        # 15% catalyst boost
        "financial_penalty_mult": Decimal("0.85"), # Soften penalties 15%
    },
    "NEUTRAL": {},  # No gating
}

# =============================================================================
# ENHANCEMENT 2: EXISTENTIAL FLAW CONFIGURATION
# =============================================================================

EXISTENTIAL_FLAW_CONFIG = {
    "runway_critical_months": Decimal("9"),
    "binary_clinical_risk_phases": ["phase_1", "phase_2", "phase 1", "phase 2", "p1", "p2"],
    "existential_cap": Decimal("65"),
    "existential_penalty": Decimal("0.20"),  # 20% penalty alternative
}

# =============================================================================
# ENHANCEMENT 3: CONFIDENCE-WEIGHTED AGGREGATION
# =============================================================================

CONFIDENCE_WEIGHTED_CONFIG = {
    "confidence_floor": Decimal("0.3"),  # Minimum confidence factor applied
}

# =============================================================================
# ENHANCEMENT 4: DYNAMIC SCORE CEILINGS
# =============================================================================

STAGE_CEILING_CONFIG = {
    "preclinical": Decimal("65"),
    "phase_1": Decimal("70"),
    "phase 1": Decimal("70"),
    "p1": Decimal("70"),
    # phase_2+ has no stage ceiling
}

CATALYST_CEILING_CONFIG = {
    "no_catalyst_12mo": Decimal("75"),
    "no_catalyst_6mo": Decimal("70"),
}

COMMERCIAL_CEILING_CONFIG = {
    "revenue_declining": Decimal("70"),  # YoY revenue decline
}

# =============================================================================
# ENHANCEMENT 5: ASYMMETRIC TRANSFORM CONFIGURATION
# =============================================================================

ASYMMETRY_CONFIG = {
    "upside_dampening": Decimal("0.6"),       # +10 → +6
    "downside_amplification": Decimal("1.2"), # -10 → -12
    "neutral_threshold": Decimal("50"),       # Scores above/below this
}

# =============================================================================
# COVERAGE-GATED SMART MONEY WEIGHTING (OPTION A)
# =============================================================================
# If a security has 13F coverage (any elite manager holds it), include smart_money
# at the configured weight. If no coverage, set smart_money weight to 0 and
# renormalize other weights. This prevents penalizing uncovered securities while
# still allowing SM signal to influence rankings for covered names.
#
# Key invariant: Weights always sum to 1.0 after renormalization.

SMART_MONEY_COVERAGE_GATED_WEIGHT = Decimal("0.05")  # 5% when coverage exists


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
    """Extract confidence for PoS score.

    Reads pos_confidence from PoS engine (stage + indication adjusted).
    Falls back to 0.7 if pos_score exists but confidence missing.
    """
    if not pos_data:
        return Decimal("0")
    # Primary: read pos_confidence from PoS engine
    conf = _to_decimal(pos_data.get("pos_confidence"))
    if conf is not None:
        return _clamp(conf, Decimal("0"), Decimal("1"))
    # Fallback: legacy "confidence" field
    conf = _to_decimal(pos_data.get("confidence"))
    if conf is not None:
        return _clamp(conf, Decimal("0"), Decimal("1"))
    return Decimal("0.7") if pos_data.get("pos_score") is not None else Decimal("0")


def _extract_confidence_short_interest(si_data: Optional[Dict]) -> Decimal:
    """Extract confidence for short interest signal.

    Confidence is based on:
    - Data availability (base 0.5 if we have any SI data)
    - Signal strength (higher SI% = higher confidence in the signal)
    - Data freshness (if status is SUCCESS)
    """
    if not si_data:
        return Decimal("0")

    if si_data.get("status") == "INSUFFICIENT_DATA":
        return Decimal("0.1")

    # Base confidence if we have data
    conf = Decimal("0.5")

    # Boost for high squeeze potential (strong signal)
    squeeze = si_data.get("squeeze_potential", "LOW")
    if squeeze == "EXTREME":
        conf += Decimal("0.3")
    elif squeeze == "HIGH":
        conf += Decimal("0.2")
    elif squeeze == "MODERATE":
        conf += Decimal("0.1")

    # Boost if we have trend data
    if si_data.get("trend_direction") not in (None, "UNKNOWN"):
        conf += Decimal("0.1")

    return _clamp(conf, Decimal("0"), Decimal("1"))


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
# ENHANCEMENT 1: HARD REGIME GATING
# =============================================================================

def apply_regime_gates(
    normalized_scores: Dict[str, Decimal],
    regime: str,
) -> Tuple[Dict[str, Decimal], Dict[str, Any], List[str]]:
    """
    Apply hard regime-based gating to normalized scores.

    In BEAR regime: Cap momentum upside, cap valuation, amplify financial penalties
    In BULL regime: Allow full momentum, boost catalysts, soften penalties

    Args:
        normalized_scores: Dict of component name -> normalized score (0-100)
        regime: Current market regime ("BULL", "BEAR", "NEUTRAL")

    Returns:
        Tuple of (gated_scores, config_applied, flags)
    """
    gated = normalized_scores.copy()
    config = REGIME_GATE_CONFIG.get(regime.upper(), {})
    flags = []

    if not config:
        return gated, {}, flags

    # Gate momentum in BEAR regime
    if "momentum_cap_pct" in config and "momentum" in gated and gated["momentum"] is not None:
        cap_pct = config["momentum_cap_pct"]
        if cap_pct < Decimal("1.0"):
            original = gated["momentum"]
            # Cap deviation from neutral: max_deviation = (score - 50) * cap_pct
            deviation = original - Decimal("50")
            if deviation > 0:  # Only cap upside momentum
                max_deviation = deviation * cap_pct
                max_momentum = Decimal("50") + max_deviation
                if original > max_momentum:
                    gated["momentum"] = _quantize_score(max_momentum)
                    flags.append("regime_momentum_gated")

    # Cap valuation upside in BEAR regime
    if "valuation_upside_cap" in config and "valuation" in gated and gated["valuation"] is not None:
        cap = config["valuation_upside_cap"]
        if gated["valuation"] > cap:
            gated["valuation"] = cap
            flags.append("regime_valuation_capped")

    # Track financial penalty multiplier for downstream use
    if "financial_penalty_mult" in config:
        flags.append("regime_financial_amplified" if config["financial_penalty_mult"] > Decimal("1.0") else "regime_financial_softened")

    return gated, config, flags


# =============================================================================
# ENHANCEMENT 2: EXISTENTIAL FLAW DETECTION
# =============================================================================

def detect_existential_flaws(
    fin_data: Dict,
    clin_data: Dict,
    cat_data: Any,
) -> List[str]:
    """
    Detect existential flaws that warrant hard score caps.

    Existential flaws are binary risk factors that, if present, should prevent
    a security from scoring too high regardless of other positive factors.

    Args:
        fin_data: Financial data dict
        clin_data: Clinical data dict
        cat_data: Catalyst data (object or dict)

    Returns:
        List of existential flaw identifiers
    """
    flaws = []
    config = EXISTENTIAL_FLAW_CONFIG

    # Flaw 1: Critical runway (< 9 months)
    runway = _to_decimal(fin_data.get("runway_months"))
    if runway is not None and runway < config["runway_critical_months"]:
        flaws.append("runway_existential")

    # Flaw 2: Binary clinical risk (single asset in early phase)
    trial_count = clin_data.get("trial_count", 0)
    lead_phase = clin_data.get("lead_phase", "").lower() if clin_data.get("lead_phase") else ""

    # Check if in risky early phases
    is_early_phase = any(phase in lead_phase for phase in config["binary_clinical_risk_phases"])

    if trial_count <= 1 and is_early_phase:
        flaws.append("binary_clinical_risk")

    # Flaw 3: Debt maturity before next catalyst (if data available)
    # This requires debt_maturity_date and next_catalyst_date comparison
    debt_maturity = fin_data.get("debt_maturity_months")
    days_to_cat = None

    if hasattr(cat_data, 'days_to_nearest_catalyst'):
        days_to_cat = cat_data.days_to_nearest_catalyst
    elif isinstance(cat_data, dict):
        scores = cat_data.get("scores", cat_data)
        days_to_cat = scores.get("days_to_nearest_catalyst")

    if debt_maturity is not None and days_to_cat is not None:
        debt_maturity_dec = _to_decimal(debt_maturity)
        days_to_cat_months = Decimal(str(days_to_cat)) / Decimal("30") if days_to_cat else None

        if debt_maturity_dec is not None and days_to_cat_months is not None:
            if debt_maturity_dec < days_to_cat_months:
                flaws.append("debt_before_catalyst")

    return flaws


def apply_existential_cap(
    score: Decimal,
    flaws: List[str],
) -> Tuple[Decimal, List[str]]:
    """
    Apply existential flaw cap to score.

    Args:
        score: Pre-cap composite score
        flaws: List of existential flaw identifiers

    Returns:
        Tuple of (capped_score, flags)
    """
    flags = []
    if flaws:
        cap = EXISTENTIAL_FLAW_CONFIG["existential_cap"]
        if score > cap:
            score = cap
            flags.append("existential_capped")
        # Add individual flaw flags
        for flaw in flaws:
            flags.append(f"existential_{flaw}")

    return score, flags


# =============================================================================
# ENHANCEMENT 3: CONFIDENCE-WEIGHTED AGGREGATION
# =============================================================================

def compute_confidence_weighted_contribution(
    normalized_score: Decimal,
    weight: Decimal,
    confidence: Decimal,
    confidence_floor: Decimal = None,
) -> Tuple[Decimal, Decimal]:
    """
    Compute contribution with confidence as a binding factor.

    The effective score is reduced based on confidence level, making
    low-confidence signals contribute less to the final score.

    Formula:
        conf_factor = confidence_floor + (1 - confidence_floor) * confidence
        effective_score = score * conf_factor
        contribution = effective_score * weight

    Args:
        normalized_score: Normalized component score (0-100)
        weight: Effective weight for this component
        confidence: Confidence level (0-1)
        confidence_floor: Minimum confidence factor (default from config)

    Returns:
        Tuple of (contribution, conf_factor)
    """
    if confidence_floor is None:
        confidence_floor = CONFIDENCE_WEIGHTED_CONFIG["confidence_floor"]

    # Confidence factor ranges from confidence_floor (low conf) to 1.0 (high conf)
    conf_factor = confidence_floor + (Decimal("1") - confidence_floor) * confidence
    conf_factor = _clamp(conf_factor, confidence_floor, Decimal("1.0"))

    effective_score = normalized_score * conf_factor
    contribution = effective_score * weight

    return _quantize_score(contribution), conf_factor


# =============================================================================
# ENHANCEMENT 4: DYNAMIC SCORE CEILINGS
# =============================================================================

def apply_dynamic_ceilings(
    score: Decimal,
    lead_phase: Optional[str],
    days_to_catalyst: Optional[int],
    revenue_growth: Optional[Decimal],
    is_commercial: bool,
) -> Tuple[Decimal, List[str]]:
    """
    Apply dynamic score ceilings based on stage, catalyst timing, and commercial status.

    Ceilings prevent scores from being too high in structurally disadvantaged situations:
    - Early-stage companies (preclinical, phase 1) have lower ceilings
    - No near-term catalyst means limited upside
    - Commercial companies with declining revenue face ceiling

    Args:
        score: Pre-ceiling composite score
        lead_phase: Lead program phase (e.g., "preclinical", "phase_1", "phase_2")
        days_to_catalyst: Days until next catalyst (None if no catalyst)
        revenue_growth: YoY revenue growth rate (negative = decline)
        is_commercial: Whether company is revenue-generating commercial stage

    Returns:
        Tuple of (capped_score, ceilings_applied)
    """
    ceilings_applied = []
    result = score

    # Normalize lead_phase for matching
    phase_lower = lead_phase.lower() if lead_phase else ""

    # Stage ceiling for early-stage companies
    for phase_key, cap in STAGE_CEILING_CONFIG.items():
        if phase_key in phase_lower:
            if result > cap:
                result = cap
                ceilings_applied.append(f"stage_ceiling_{phase_key.replace(' ', '_')}")
            break  # Only apply one stage ceiling

    # Catalyst ceiling - no near-term catalyst limits upside
    if days_to_catalyst is None or days_to_catalyst > 365:
        cap = CATALYST_CEILING_CONFIG["no_catalyst_12mo"]
        if result > cap:
            result = cap
            ceilings_applied.append("no_catalyst_12mo_ceiling")
    elif days_to_catalyst > 180:
        cap = CATALYST_CEILING_CONFIG["no_catalyst_6mo"]
        if result > cap:
            result = cap
            ceilings_applied.append("no_catalyst_6mo_ceiling")

    # Commercial revenue declining ceiling
    if is_commercial and revenue_growth is not None and revenue_growth < Decimal("0"):
        cap = COMMERCIAL_CEILING_CONFIG["revenue_declining"]
        if result > cap:
            result = cap
            ceilings_applied.append("commercial_revenue_declining_ceiling")

    return _quantize_score(result), ceilings_applied


# =============================================================================
# ENHANCEMENT 5: ASYMMETRIC TRANSFORM (CONVEX DOWNSIDE, CONCAVE UPSIDE)
# =============================================================================

def apply_asymmetric_transform(
    normalized_score: Decimal,
    component: str,
) -> Decimal:
    """
    Apply asymmetric transformation to a normalized score.

    This transformation creates:
    - Concave upside: Positive deviations from neutral are dampened (+10 → +6)
    - Convex downside: Negative deviations from neutral are amplified (-10 → -12)

    This reflects the asymmetric nature of biotech investing where downside risks
    are often more severe than upside surprises.

    Args:
        normalized_score: Normalized component score (0-100)
        component: Component name (for potential future component-specific transforms)

    Returns:
        Transformed score (0-100)
    """
    config = ASYMMETRY_CONFIG
    neutral = config["neutral_threshold"]

    delta = normalized_score - neutral

    if delta > 0:
        # Concave upside: dampen positive deviations
        transformed_delta = delta * config["upside_dampening"]
    elif delta < 0:
        # Convex downside: amplify negative deviations
        transformed_delta = delta * config["downside_amplification"]
    else:
        transformed_delta = Decimal("0")

    transformed_score = neutral + transformed_delta

    # Clamp to valid range
    return _clamp(_quantize_score(transformed_score), Decimal("0"), Decimal("100"))


def apply_asymmetric_transform_to_contribution(
    contribution_delta: Decimal,
    component: str,
) -> Tuple[Decimal, List[str]]:
    """
    Apply asymmetric transform to contribution delta from neutral.

    Args:
        contribution_delta: Deviation of contribution from neutral
        component: Component name

    Returns:
        Tuple of (transformed_delta, flags)
    """
    flags = []
    config = ASYMMETRY_CONFIG

    if contribution_delta > 0:
        # Concave upside: dampen positive contributions
        transformed = contribution_delta * config["upside_dampening"]
        if abs(transformed - contribution_delta) > Decimal("0.5"):
            flags.append("asymmetric_upside_dampened")
    elif contribution_delta < 0:
        # Convex downside: amplify negative contributions
        transformed = contribution_delta * config["downside_amplification"]
        if abs(transformed - contribution_delta) > Decimal("0.5"):
            flags.append("asymmetric_downside_amplified")
    else:
        transformed = Decimal("0")

    return _quantize_score(transformed), flags


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
    fda_data: Optional[Dict] = None,
    diversity_data: Optional[Dict] = None,
    intensity_data: Optional[Dict] = None,
    partnership_data: Optional[Dict] = None,
    cash_burn_data: Optional[Dict] = None,
    phase_momentum_data: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Score a single ticker with all v3 enhancements."""

    flags = []

    # Extract raw scores
    fin_raw = _to_decimal(_coalesce(fin_data.get("financial_normalized"), fin_data.get("financial_score")))
    clin_raw = _to_decimal(clin_data.get("clinical_score"))
    pos_raw = _to_decimal(pos_data.get("pos_score")) if pos_data else None

    # Extract short interest score
    si_raw = _to_decimal(si_data.get("short_signal_score")) if si_data else None
    si_squeeze_potential = si_data.get("squeeze_potential", "UNKNOWN") if si_data else "UNKNOWN"
    si_crowding_risk = si_data.get("crowding_risk", "UNKNOWN") if si_data else "UNKNOWN"
    si_signal_direction = si_data.get("signal_direction", "NEUTRAL") if si_data else "NEUTRAL"
    si_trend_direction = si_data.get("trend_direction", "UNKNOWN") if si_data else "UNKNOWN"

    # Extract FDA designation data
    fda_designation_score = _to_decimal(fda_data.get("designation_score")) if fda_data else None
    fda_pos_multiplier = _to_decimal(fda_data.get("pos_multiplier")) if fda_data else Decimal("1.0")
    fda_timeline_acceleration = fda_data.get("timeline_acceleration_months", 0) if fda_data else 0
    fda_designation_types = fda_data.get("designation_types", []) if fda_data else []

    # Extract pipeline diversity data
    diversity_score = _to_decimal(diversity_data.get("diversity_score")) if diversity_data else None
    diversity_risk_profile = diversity_data.get("risk_profile", "unknown") if diversity_data else "unknown"
    diversity_program_count = diversity_data.get("program_count", 0) if diversity_data else 0
    diversity_platform_validated = diversity_data.get("platform_validated", False) if diversity_data else False

    # Extract competitive intensity data
    intensity_score = _to_decimal(intensity_data.get("competitive_intensity_score")) if intensity_data else None
    intensity_crowding = intensity_data.get("crowding_level", "unknown") if intensity_data else "unknown"
    intensity_position = intensity_data.get("competitive_position", "unknown") if intensity_data else "unknown"
    intensity_competitor_count = intensity_data.get("competitor_count", 0) if intensity_data else 0
    intensity_has_approved = intensity_data.get("has_approved_competition", False) if intensity_data else False

    # Extract partnership validation data
    partnership_score = _to_decimal(partnership_data.get("partnership_score")) if partnership_data else None
    partnership_strength = partnership_data.get("partnership_strength", "unknown") if partnership_data else "unknown"
    partnership_count = partnership_data.get("partnership_count", 0) if partnership_data else 0
    partnership_top_tier = partnership_data.get("top_tier_partners", 0) if partnership_data else 0
    partnership_total_value = _to_decimal(partnership_data.get("total_deal_value", 0)) if partnership_data else Decimal("0")
    partnership_top_partners = partnership_data.get("top_partners", []) if partnership_data else []

    # Extract cash burn trajectory data
    cash_burn_trajectory = cash_burn_data.get("trajectory", "unknown") if cash_burn_data else "unknown"
    cash_burn_risk = cash_burn_data.get("risk_level", "unknown") if cash_burn_data else "unknown"

    # Extract phase momentum data
    phase_momentum_value = phase_momentum_data.get("momentum", "unknown") if phase_momentum_data else "unknown"
    phase_momentum_confidence = _to_decimal(phase_momentum_data.get("confidence", 0)) if phase_momentum_data else Decimal("0")
    phase_momentum_lead_phase = phase_momentum_data.get("current_lead_phase", "unknown") if phase_momentum_data else "unknown"

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

    # V2: Extract financial data for valuation regime routing
    has_revenue = fin_data.get("has_revenue", False)
    revenue_mm = None
    cfo_mm = None
    enterprise_value_mm = None

    # Extract actual CFO and Revenue from financial data (raw values in dollars)
    raw_cfo = fin_data.get("CFO")
    raw_revenue = fin_data.get("Revenue")

    # Convert to millions
    if raw_revenue is not None:
        revenue_mm = _to_decimal(raw_revenue) / Decimal("1000000")
    elif fin_data.get("revenue_scale_bucket") in ("medium", "large", "mega"):
        # Fallback: Use market_cap as proxy if exact revenue not available
        revenue_mm = market_cap_mm

    if raw_cfo is not None:
        cfo_mm = _to_decimal(raw_cfo) / Decimal("1000000")
    elif fin_data.get("burn_source") == "profitable":
        # Fallback: Profitable = positive CFO, use liquid_assets as proxy
        cfo_mm = _to_decimal(fin_data.get("liquid_assets")) / Decimal("1000000") if fin_data.get("liquid_assets") else Decimal("1")
    elif fin_data.get("monthly_burn") == 0 or fin_data.get("monthly_burn") == 0.0:
        # Fallback: No burn = likely profitable
        cfo_mm = Decimal("1")

    # Enterprise value from market data if available
    if market_data:
        enterprise_value_mm = _to_decimal(market_data.get("enterprise_value"))
        if enterprise_value_mm:
            enterprise_value_mm = enterprise_value_mm / Decimal("1000000")  # Convert to millions

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

        # Extract benchmark values with None-safe fallback (0.0 is a valid value)
        # CRITICAL: Do NOT use `val or fallback` pattern - 0.0 is falsy but valid!
        _xbi_20 = market_data.get("xbi_return_20d")
        _xbi_60 = market_data.get("xbi_return_60d")
        _xbi_120 = market_data.get("xbi_return_120d")

        momentum_input = MultiWindowMomentumInput(
            return_20d=_to_decimal(market_data.get("return_20d")),
            return_60d=_to_decimal(market_data.get("return_60d")),
            return_120d=_to_decimal(market_data.get("return_120d")),
            benchmark_20d=_to_decimal(_xbi_20 if _xbi_20 is not None else market_data.get("benchmark_return_20d")),
            benchmark_60d=_to_decimal(_xbi_60 if _xbi_60 is not None else market_data.get("benchmark_return_60d")),
            benchmark_120d=_to_decimal(_xbi_120 if _xbi_120 is not None else market_data.get("benchmark_return_120d")),
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

    # 2. Momentum signal (multi-window blended)
    # Uses weighted blend of 20d (20%), 60d (50%), 120d (30%) for robust signal
    momentum = compute_momentum_signal_multiwindow(
        momentum_input,
        use_vol_adjusted_alpha=False,
        blend_mode="weighted",  # Use multi-window blend instead of fallback
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

    # 3. Valuation signal (V2: with regime routing)
    valuation = compute_valuation_signal(
        market_cap_mm, trial_count, lead_phase, peer_valuations,
        revenue_mm=revenue_mm,
        cfo_mm=cfo_mm,
        enterprise_value_mm=enterprise_value_mm,
        has_revenue=has_revenue,
    )
    valuation_norm = valuation.valuation_score
    if valuation.valuation_score >= Decimal("70"):
        flags.append("undervalued_vs_peers")
    elif valuation.valuation_score <= Decimal("30"):
        flags.append("overvalued_vs_peers")

    # Add valuation regime and method flags
    if valuation.regime:
        flags.append(f"valuation_regime_{valuation.regime.value}")
    if valuation.flags:
        flags.extend(valuation.flags)

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
        competitive_crowding=intensity_crowding,
        partnership_strength=partnership_strength,
        cash_burn_trajectory=cash_burn_trajectory,
        cash_burn_risk=cash_burn_risk,
        phase_momentum=phase_momentum_value,
        phase_momentum_confidence=phase_momentum_confidence,
        phase_momentum_lead_phase=phase_momentum_lead_phase,
        days_to_nearest_catalyst=days_to_cat,
    )
    flags.extend(interactions.interaction_flags)

    # 7. Financial Module 2: Survivability & Capital Discipline
    # Bounded additive term [-10, +5] with 0.06 weight
    survivability_result = compute_survivability_score(fin_data)
    survivability_score = Decimal(str(survivability_result.get("score", 0)))
    survivability_subscores = survivability_result.get("subscores", {})
    survivability_coverage = survivability_result.get("coverage", [])

    if survivability_score < Decimal("-3"):
        flags.append("survivability_critical")
    elif survivability_score < Decimal("0"):
        flags.append("survivability_warning")
    elif survivability_score >= Decimal("3"):
        flags.append("survivability_strong")

    if "missing_cash" in survivability_coverage:
        flags.append("survivability_missing_cash")
    if "missing_burn_data" in survivability_coverage:
        flags.append("survivability_missing_burn")

    # =========================================================================
    # CONFIDENCE EXTRACTION
    # =========================================================================

    conf_clin = _extract_confidence_clinical(clin_data)
    conf_fin = _extract_confidence_financial(fin_data)
    conf_cat = _extract_confidence_catalyst(cat_data)
    conf_pos = _extract_confidence_pos(pos_data)
    conf_si = _extract_confidence_short_interest(si_data)

    conf_clin = _clamp(conf_clin - vol_adj.confidence_penalty, Decimal("0.1"), Decimal("1"))
    conf_fin = _clamp(conf_fin - vol_adj.confidence_penalty, Decimal("0.1"), Decimal("1"))
    conf_cat = _clamp(conf_cat - vol_adj.confidence_penalty, Decimal("0.1"), Decimal("1"))

    # =========================================================================
    # WEIGHT COMPUTATION
    # =========================================================================

    regime_weights = apply_regime_to_weights(base_weights, regime)

    # =========================================================================
    # ENHANCEMENT 1: HARD REGIME GATING
    # Apply regime-specific gates to normalized scores before contribution calc
    # =========================================================================

    pre_gate_scores = {
        "clinical": clin_norm,
        "financial": fin_norm,
        "catalyst": cat_norm,
        "momentum": momentum_norm,
        "valuation": valuation_norm,
    }

    gated_scores, regime_config, regime_gate_flags = apply_regime_gates(pre_gate_scores, regime)
    flags.extend(regime_gate_flags)

    # Update normalized scores with gated values
    if gated_scores.get("momentum") is not None and gated_scores["momentum"] != momentum_norm:
        momentum_norm = gated_scores["momentum"]
    if gated_scores.get("valuation") is not None and gated_scores["valuation"] != valuation_norm:
        valuation_norm = gated_scores["valuation"]

    # Store financial penalty multiplier for later use
    financial_penalty_mult = regime_config.get("financial_penalty_mult", Decimal("1.0"))

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
        "short_interest": conf_si,
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
    # COVERAGE-GATED SMART MONEY WEIGHTING (OPTION A)
    # =========================================================================
    # If security has 13F coverage, include smart_money at 5% and renormalize.
    # If no coverage, smart_money weight is 0 (no penalty for missing data).
    # This allows SM signal to influence covered names without taxing uncovered ones.

    has_smart_money_coverage = smart_money.overlap_count > 0
    if has_smart_money_coverage:
        # Add smart_money at configured weight, then renormalize
        effective_weights["smart_money"] = SMART_MONEY_COVERAGE_GATED_WEIGHT
        flags.append("smart_money_coverage_included")
    else:
        effective_weights["smart_money"] = Decimal("0")
        flags.append("smart_money_no_coverage")

    # Renormalize weights to sum to 1.0 after adding smart_money
    total = sum(effective_weights.values())
    if total > EPS:
        effective_weights = {k: _quantize_weight(v / total) for k, v in effective_weights.items()}

    # =========================================================================
    # COMPOSITE SCORE COMPUTATION
    # =========================================================================

    component_scores = []
    contributions = {}
    confidence_factors = {}  # Track confidence factors for each component
    asymmetric_flags = []  # Track asymmetric transform flags

    core_scores = {
        "clinical": (clin_norm, clin_raw, conf_clin),
        "financial": (fin_norm, fin_raw, conf_fin),
        "catalyst": (cat_norm_decayed, cat_raw, conf_cat),
    }

    for name, (norm, raw, conf) in core_scores.items():
        w_eff = effective_weights.get(name, Decimal("0"))

        # ENHANCEMENT 5: Apply asymmetric transform to normalized score
        transformed_norm = apply_asymmetric_transform(norm, name)
        delta_from_transform = transformed_norm - norm
        if abs(delta_from_transform) > Decimal("0.5"):
            if delta_from_transform < 0:
                asymmetric_flags.append(f"{name}_asymmetric_upside_dampened")
            else:
                asymmetric_flags.append(f"{name}_asymmetric_downside_amplified")

        # ENHANCEMENT 3: Confidence-weighted contribution
        contrib, conf_factor = compute_confidence_weighted_contribution(
            transformed_norm, w_eff, conf
        )
        contributions[name] = contrib
        confidence_factors[name] = conf_factor

        notes = []
        if raw is None:
            notes.append("missing_raw")
        if name == "catalyst" and decay.decay_factor < Decimal("0.9"):
            notes.append(f"decay_factor_{decay.decay_factor}")
        if conf_factor < Decimal("0.7"):
            notes.append(f"conf_weighted_{conf_factor}")

        component_scores.append(ComponentScore(
            name=name,
            raw=raw,
            normalized=transformed_norm,  # Store transformed value
            confidence=conf,
            weight_base=base_weights.get(name, Decimal("0")),
            weight_effective=w_eff,
            contribution=_quantize_score(contrib),
            decay_factor=decay.decay_factor if name == "catalyst" else None,
            notes=notes,
        ))

    # Add asymmetric transform flags
    flags.extend(asymmetric_flags)

    if mode in (ScoringMode.ENHANCED, ScoringMode.PARTIAL):
        if "momentum" in effective_weights:
            w_eff = effective_weights["momentum"]

            # ENHANCEMENT 5: Apply asymmetric transform
            transformed_momentum = apply_asymmetric_transform(momentum_norm, "momentum")
            delta_from_transform = transformed_momentum - momentum_norm
            if abs(delta_from_transform) > Decimal("0.5"):
                if delta_from_transform < 0:
                    flags.append("momentum_asymmetric_upside_dampened")
                else:
                    flags.append("momentum_asymmetric_downside_amplified")

            # ENHANCEMENT 3: Confidence-weighted contribution
            contrib, conf_factor = compute_confidence_weighted_contribution(
                transformed_momentum, w_eff, momentum.confidence
            )
            contributions["momentum"] = contrib
            confidence_factors["momentum"] = conf_factor

            mom_notes = [] if momentum.alpha_60d is not None else ["missing_price_data"]
            if conf_factor < Decimal("0.7"):
                mom_notes.append(f"conf_weighted_{conf_factor}")

            component_scores.append(ComponentScore(
                name="momentum",
                raw=momentum.alpha_60d,
                normalized=transformed_momentum,
                confidence=momentum.confidence,
                weight_base=base_weights.get("momentum", Decimal("0")),
                weight_effective=w_eff,
                contribution=_quantize_score(contrib),
                notes=mom_notes,
            ))

        if "valuation" in effective_weights:
            w_eff = effective_weights["valuation"]

            # ENHANCEMENT 5: Apply asymmetric transform
            transformed_valuation = apply_asymmetric_transform(valuation_norm, "valuation")
            delta_from_transform = transformed_valuation - valuation_norm
            if abs(delta_from_transform) > Decimal("0.5"):
                if delta_from_transform < 0:
                    flags.append("valuation_asymmetric_upside_dampened")
                else:
                    flags.append("valuation_asymmetric_downside_amplified")

            # ENHANCEMENT 3: Confidence-weighted contribution
            contrib, conf_factor = compute_confidence_weighted_contribution(
                transformed_valuation, w_eff, valuation.confidence
            )
            contributions["valuation"] = contrib
            confidence_factors["valuation"] = conf_factor

            val_notes = [] if valuation.peer_count >= 5 else ["insufficient_peers"]
            if conf_factor < Decimal("0.7"):
                val_notes.append(f"conf_weighted_{conf_factor}")

            component_scores.append(ComponentScore(
                name="valuation",
                raw=valuation.mcap_per_asset,
                normalized=transformed_valuation,
                confidence=valuation.confidence,
                weight_base=base_weights.get("valuation", Decimal("0")),
                weight_effective=w_eff,
                contribution=_quantize_score(contrib),
                notes=val_notes,
            ))

        # Short interest signal integration
        if "short_interest" in effective_weights and si_raw is not None:
            w_eff = effective_weights["short_interest"]
            # Short interest score is already 0-100, use directly as normalized
            si_norm = si_raw
            contrib = si_norm * w_eff
            contributions["short_interest"] = contrib

            si_notes = []
            if si_squeeze_potential == "EXTREME":
                flags.append("extreme_squeeze_potential")
                si_notes.append("extreme_squeeze")
            elif si_squeeze_potential == "HIGH":
                flags.append("high_squeeze_potential")
                si_notes.append("high_squeeze")
            if si_crowding_risk == "HIGH":
                flags.append("high_short_crowding")
                si_notes.append("high_crowding")
            if si_signal_direction == "BULLISH":
                flags.append("si_bullish_signal")
            elif si_signal_direction == "BEARISH":
                flags.append("si_bearish_signal")
            if si_trend_direction == "COVERING":
                flags.append("shorts_covering")
            elif si_trend_direction == "BUILDING":
                flags.append("shorts_building")

            component_scores.append(ComponentScore(
                name="short_interest",
                raw=si_raw,
                normalized=si_norm,
                confidence=conf_si,
                weight_base=base_weights.get("short_interest", Decimal("0")),
                weight_effective=w_eff,
                contribution=_quantize_score(contrib),
                notes=si_notes if si_notes else [],
            ))
            flags.append("short_interest_applied")

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
    # SMART MONEY CONTRIBUTION (Coverage-Gated)
    # =========================================================================
    # Only add contribution if coverage exists (weight > 0 after gating)

    sm_w_eff = effective_weights.get("smart_money", Decimal("0"))
    if sm_w_eff > EPS and has_smart_money_coverage:
        # smart_money.smart_money_score is already 20-80 range, use directly
        sm_norm = smart_money.smart_money_score
        sm_contrib = sm_norm * sm_w_eff
        contributions["smart_money"] = sm_contrib

        sm_notes = []
        if smart_money.holders_increasing:
            sm_notes.append(f"buying:{len(smart_money.holders_increasing)}")
        if smart_money.holders_decreasing:
            sm_notes.append(f"selling:{len(smart_money.holders_decreasing)}")
        if smart_money.tier1_holders:
            sm_notes.append(f"tier1:{len(smart_money.tier1_holders)}")
        if smart_money.conditional_capped:
            sm_notes.append("conditional_capped")

        component_scores.append(ComponentScore(
            name="smart_money",
            raw=Decimal(str(smart_money.overlap_count)),  # Raw overlap count
            normalized=sm_norm,
            confidence=smart_money.confidence,
            weight_base=SMART_MONEY_COVERAGE_GATED_WEIGHT,  # Base weight when covered
            weight_effective=sm_w_eff,
            contribution=_quantize_score(sm_contrib),
            notes=sm_notes if sm_notes else [],
        ))
        flags.append("smart_money_applied")

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
    # ENHANCEMENT 6: CONTRADICTION DETECTOR
    # Detect conflicting signals that reduce confidence in the score
    # =========================================================================

    contradiction_data = {
        "momentum_score": momentum_norm,
        "valuation_score": valuation_norm,
        "clinical_score": clin_norm,
        "financial_score": fin_norm,
        "liquidity_gate": liquidity_status,
        "runway_months": runway_months,
        "dilution_bucket": dilution_bucket,
    }

    contradictions = detect_contradictions(contradiction_data)

    if contradictions.contradictions:
        # Apply contradiction score penalty
        pre_penalty = pre_penalty - contradictions.score_penalty
        pre_penalty = _quantize_score(max(pre_penalty, Decimal("0")))

        # Add contradiction flags
        for contradiction in contradictions.contradictions:
            flags.append(f"contradiction_{contradiction}")

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

    # =========================================================================
    # ENHANCEMENT 4: DYNAMIC SCORE CEILINGS
    # Apply stage, catalyst, and commercial revenue ceilings
    # =========================================================================

    # Determine if commercial (has revenue)
    is_commercial = has_revenue or (revenue_mm is not None and revenue_mm > Decimal("0"))

    # Calculate revenue growth if available
    revenue_growth = None
    if fin_data.get("revenue_yoy_growth") is not None:
        revenue_growth = _to_decimal(fin_data.get("revenue_yoy_growth"))
    elif fin_data.get("revenue_growth_rate") is not None:
        revenue_growth = _to_decimal(fin_data.get("revenue_growth_rate"))

    post_ceiling, ceilings_applied = apply_dynamic_ceilings(
        post_cap,
        lead_phase,
        days_to_cat,
        revenue_growth,
        is_commercial,
    )

    if ceilings_applied:
        flags.extend(ceilings_applied)
        caps_applied.extend([{"reason": c, "cap": str(post_ceiling)} for c in ceilings_applied])

    # =========================================================================
    # ENHANCEMENT 2: EXISTENTIAL FLAW CAPS
    # Detect and cap for existential risks (runway, binary clinical, debt)
    # =========================================================================

    existential_flaws = detect_existential_flaws(fin_data, clin_data, cat_data)
    post_existential, existential_flags = apply_existential_cap(post_ceiling, existential_flaws)

    if existential_flags:
        flags.extend(existential_flags)
        if post_existential < post_ceiling:
            caps_applied.append({
                "reason": "existential_cap",
                "cap": str(EXISTENTIAL_FLAW_CONFIG["existential_cap"]),
                "flaws": existential_flaws,
            })

    post_vol = apply_volatility_to_score(post_existential, vol_adj)

    delta_bonus = (cat_delta / Decimal("25")).quantize(SCORE_PRECISION)

    # Financial Module 2: Survivability contribution (bounded additive term)
    # Weight 0.06, score range [-10, +5] -> contribution range [-0.6, +0.3]
    SURVIVABILITY_WEIGHT = Decimal("0.06")
    survivability_contribution = (survivability_score * SURVIVABILITY_WEIGHT).quantize(SCORE_PRECISION)

    final_score = _clamp(post_vol + delta_bonus + survivability_contribution, Decimal("0"), Decimal("100"))
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
        "survivability_score": survivability_score,
        "survivability_contribution": survivability_contribution,
        # New enhancement tracking
        "regime_gates_applied": bool(regime_gate_flags),
        "contradiction_penalty": contradictions.score_penalty,
        "contradictions_detected": contradictions.contradictions,
        "ceilings_applied": ceilings_applied,
        "existential_flaws": existential_flaws,
        "confidence_factors": {k: str(v) for k, v in confidence_factors.items()},
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
            "confidence_factors": {k: str(v) for k, v in confidence_factors.items()},
        },
        penalties_and_gates={
            "uncertainty_penalty_pct": str(_quantize_score(uncertainty_penalty * Decimal("100"))),
            "severity_gate": SEVERITY_GATE_LABELS[worst_severity],
            "severity_multiplier": str(severity_multiplier),
            "monotonic_caps_applied": caps_applied,
            # New enhancement penalties
            "contradiction_penalty": str(contradictions.score_penalty),
            "contradictions": contradictions.contradictions,
            "dynamic_ceilings": ceilings_applied,
            "existential_flaws": existential_flaws,
            "regime_gates": regime_gate_flags,
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
            "post_contradiction_score": str(_quantize_score(pre_penalty - contradictions.score_penalty)),
            "post_uncertainty_score": str(_quantize_score(post_uncertainty)),
            "post_severity_score": str(post_severity),
            "post_cap_score": str(post_cap),
            "post_ceiling_score": str(post_ceiling),
            "post_existential_score": str(post_existential),
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
        "confidence_short_interest": conf_si if si_raw is not None else None,
        "confidence_overall": confidence_overall,
        "effective_weights": effective_weights,
        "normalization_method": normalization_method.value,
        "caps_applied": caps_applied,
        # NOTE: component_scores kept as raw dataclass list for internal use.
        # Serialization happens in module_5_composite_v3.py via score_breakdown.components
        # which is the single source of truth for JSON output.
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
            "valuation_score": str(valuation.valuation_score),
            "peer_count": valuation.peer_count,
            "confidence": str(valuation.confidence),
            # V2 regime routing fields
            "regime": valuation.regime.value if hasattr(valuation.regime, 'value') else str(valuation.regime),
            "method": valuation.method,
            "ev_multiple": str(valuation.ev_multiple) if valuation.ev_multiple else None,
            "peer_median_ev_multiple": str(valuation.peer_median_ev_multiple) if valuation.peer_median_ev_multiple else None,
            "mcap_per_asset": str(valuation.mcap_per_asset) if valuation.mcap_per_asset else None,
            "peer_median_mcap_per_asset": str(valuation.peer_median_mcap_per_asset) if valuation.peer_median_mcap_per_asset else None,
            "flags": valuation.flags if valuation.flags else [],
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
        "short_interest_signal": {
            "score": str(si_raw) if si_raw is not None else None,
            "squeeze_potential": si_squeeze_potential,
            "crowding_risk": si_crowding_risk,
            "signal_direction": si_signal_direction,
            "trend_direction": si_trend_direction,
            "confidence": str(conf_si),
            "applied": si_raw is not None and "short_interest" in effective_weights,
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
        "fda_designation_signal": {
            "designation_score": str(fda_designation_score) if fda_designation_score else None,
            "pos_multiplier": str(fda_pos_multiplier),
            "timeline_acceleration_months": fda_timeline_acceleration,
            "designation_types": fda_designation_types,
            "has_designations": len(fda_designation_types) > 0,
        },
        "pipeline_diversity_signal": {
            "diversity_score": str(diversity_score) if diversity_score else None,
            "risk_profile": diversity_risk_profile,
            "program_count": diversity_program_count,
            "platform_validated": diversity_platform_validated,
        },
        "competitive_intensity_signal": {
            "intensity_score": str(intensity_score) if intensity_score else None,
            "crowding_level": intensity_crowding,
            "competitive_position": intensity_position,
            "competitor_count": intensity_competitor_count,
            "has_approved_competition": intensity_has_approved,
        },
        "partnership_signal": {
            "partnership_score": str(partnership_score) if partnership_score else None,
            "partnership_strength": partnership_strength,
            "partnership_count": partnership_count,
            "top_tier_partners": partnership_top_tier,
            "total_deal_value_mm": str(partnership_total_value),
            "top_partners": partnership_top_partners,
        },
        "survivability_signal": {
            "score": str(survivability_score),
            "contribution": str(survivability_contribution),
            "subscores": {k: str(v) for k, v in survivability_subscores.items()},
            "coverage": survivability_coverage,
            "metrics": survivability_result.get("metrics", {}),
        },
        "interaction_terms": {
            "total_adjustment": str(interactions.total_interaction_adjustment),
            "flags": interactions.interaction_flags,
        },
        # New composite scoring enhancements tracking
        "regime_gating": {
            "regime": regime,
            "gates_applied": regime_gate_flags,
            "financial_penalty_mult": str(financial_penalty_mult),
        },
        "contradiction_detection": {
            "contradictions": contradictions.contradictions,
            "confidence_penalty": str(contradictions.confidence_penalty),
            "score_penalty": str(contradictions.score_penalty),
            "diagnostics": contradictions.diagnostics,
        },
        "dynamic_ceilings": {
            "ceilings_applied": ceilings_applied,
            "lead_phase": lead_phase,
            "days_to_catalyst": days_to_cat,
            "is_commercial": is_commercial,
            "revenue_growth": str(revenue_growth) if revenue_growth is not None else None,
        },
        "existential_flaw_detection": {
            "flaws": existential_flaws,
            "cap_applied": "existential_capped" in existential_flags,
            "cap_value": str(EXISTENTIAL_FLAW_CONFIG["existential_cap"]) if existential_flaws else None,
        },
        "confidence_weighting": {
            "factors": {k: str(v) for k, v in confidence_factors.items()},
            "floor": str(CONFIDENCE_WEIGHTED_CONFIG["confidence_floor"]),
        },
        "asymmetric_transform": {
            "upside_dampening": str(ASYMMETRY_CONFIG["upside_dampening"]),
            "downside_amplification": str(ASYMMETRY_CONFIG["downside_amplification"]),
            "flags_triggered": [f for f in flags if "asymmetric" in f],
        },
    }
