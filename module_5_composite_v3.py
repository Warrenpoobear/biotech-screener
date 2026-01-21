#!/usr/bin/env python3
"""
Module 5: Composite Ranker (v3) - IC-Enhanced Edition

Production-ready composite scoring with all IC enhancement features:
- Adaptive weight learning (historical IC optimization)
- Non-linear signal interactions (cross-factor synergies/penalties)
- Peer-relative valuation signal
- Catalyst signal decay (time-based IC modeling)
- Price momentum signal (relative strength vs XBI)
- Shrinkage normalization (Bayesian cohort adjustment)
- Smart money signal (13F position changes)
- Volatility-adjusted scoring
- Regime-adaptive component selection
- Monotonic caps (from v2)
- Confidence weighting (from v2)
- Hybrid aggregation with weakest-link (from v2)

Design Philosophy (Governed Over Smart):
- DETERMINISTIC: No datetime.now(), no randomness
- STDLIB-ONLY: No external dependencies
- DECIMAL-ONLY: Pure Decimal arithmetic for all scoring
- FAIL LOUDLY: Clear error states with validation
- AUDITABLE: Full provenance chain with score breakdown
- IC-OPTIMIZED: Every feature designed to maximize predictive power

Weight Structure (v3 Enhanced):
- Clinical Development: 28%
- Financial Health:     25%
- Catalyst Momentum:    17%
- Probability of Success: 15%
- Price Momentum:       10%
- Valuation:            5%
(When enhancement data available; falls back to v1 weights otherwise)

Expected IC Improvement: +0.08 to +0.15 vs v1 baseline

Author: Wake Robin Capital Management
Version: 3.0.0
Last Modified: 2026-01-17
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set, Union

from common.provenance import create_provenance
from common.types import Severity
from common.integration_contracts import (
    extract_financial_score,
    extract_catalyst_score,
    extract_clinical_score,
    validate_module_1_output,
    validate_module_2_output,
    validate_module_3_output,
    validate_module_4_output,
    SchemaValidationError,
)

# Import IC enhancement utilities
from src.modules.ic_enhancements import (
    # Core enhancement functions
    compute_volatility_adjustment,
    apply_volatility_to_score,
    compute_momentum_signal,
    compute_momentum_signal_with_fallback,  # V2: Multi-window with fallback
    compute_valuation_signal,
    compute_catalyst_decay,
    apply_catalyst_decay,
    compute_smart_money_signal,
    compute_interaction_terms,
    shrinkage_normalize,
    apply_regime_to_weights,
    compute_adaptive_weights,
    compute_enhanced_score,
    get_regime_signal_importance,
    # Types
    VolatilityAdjustment,
    VolatilityBucket,
    MomentumSignal,
    MultiWindowMomentumInput,  # V2: Multi-window input
    ValuationSignal,
    CatalystDecayResult,
    SmartMoneySignal,
    InteractionTerms,
    AdaptiveWeights,
    RegimeType,
    EnhancedScoringResult,
    # Helpers
    _to_decimal,
    _quantize_score,
    _quantize_weight,
    _clamp,
    _safe_divide,
    EPS,
    SCORE_PRECISION,
    WEIGHT_PRECISION,
)

# Import PIT validation
from src.modules.ic_pit_validation import (
    run_production_gate,
    create_weight_provenance,
    PITValidationError,
    WeightStabilityError,
    ProductionGateResult,
    WeightProvenance,
)

__version__ = "3.0.0"
RULESET_VERSION = "3.0.0-IC-ENHANCED"
SCHEMA_VERSION = "v3.0"

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# V3 Enhanced weights (with all new signals)
V3_ENHANCED_WEIGHTS = {
    "clinical": Decimal("0.28"),
    "financial": Decimal("0.25"),
    "catalyst": Decimal("0.17"),
    "pos": Decimal("0.15"),
    "momentum": Decimal("0.10"),
    "valuation": Decimal("0.05"),
}

# V3 Default weights (without enhancement data)
V3_DEFAULT_WEIGHTS = {
    "clinical": Decimal("0.40"),
    "financial": Decimal("0.35"),
    "catalyst": Decimal("0.25"),
}

# V3 Partial weights (with some enhancement data)
V3_PARTIAL_WEIGHTS = {
    "clinical": Decimal("0.35"),
    "financial": Decimal("0.30"),
    "catalyst": Decimal("0.20"),
    "momentum": Decimal("0.10"),
    "valuation": Decimal("0.05"),
}

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

# PoS contribution cap - prevents any single mapping surprise from
# moving composite more than ±6 points (protects against data quality issues)
POS_DELTA_CAP = Decimal("6.0")

# Catalyst effective blending weights
# When both window score and proximity score present:
#   catalyst_effective = CATALYST_WINDOW_WEIGHT * window_score + CATALYST_PROXIMITY_WEIGHT * proximity_score
# When only proximity present:
#   catalyst_effective = CATALYST_DEFAULT_BASE * 50 + (1 - CATALYST_DEFAULT_BASE) * proximity_score
CATALYST_WINDOW_WEIGHT = Decimal("0.70")
CATALYST_PROXIMITY_WEIGHT = Decimal("0.30")
CATALYST_DEFAULT_BASE = Decimal("0.85")  # Weight toward neutral 50 when only proximity present
CATALYST_DEFAULT_SCORE = Decimal("50")

# Confidence gate threshold - components with confidence below this contribute 0
# This prevents missing data from being masked by neutral (50) defaults
CONFIDENCE_GATE_THRESHOLD = Decimal("0.4")

# Pipeline health thresholds (fraction of universe)
# NOTE: Biotech-adjusted thresholds - sparse coverage is normal for optional enhancement components
# These components enhance but don't drive the core scoring (clinical + financial + catalyst)
HEALTH_GATE_THRESHOLDS = {
    "catalyst": Decimal("0.10"),    # Fail if <10% have catalyst events (core component)
    "momentum": Decimal("0.00"),    # Optional: 13F fallback provides sparse coverage by design
    "smart_money": Decimal("0.00"), # Optional: 13F data only covers subset of universe
}

# Monotonic cap thresholds
class MonotonicCap:
    LIQUIDITY_FAIL_CAP = Decimal("35")
    LIQUIDITY_WARN_CAP = Decimal("60")
    RUNWAY_CRITICAL_CAP = Decimal("40")
    RUNWAY_WARNING_CAP = Decimal("55")
    DILUTION_SEVERE_CAP = Decimal("45")
    DILUTION_HIGH_CAP = Decimal("60")


class ScoringMode(str, Enum):
    """Scoring mode based on available data."""
    DEFAULT = "default"           # No enhancement data
    PARTIAL = "partial"           # Some enhancement data (momentum, valuation)
    ENHANCED = "enhanced"         # Full enhancement data including PoS
    ADAPTIVE = "adaptive"         # Using adaptive weights from historical data


class RunStatus(str, Enum):
    """Pipeline run status based on data coverage health."""
    OK = "OK"                     # All pipelines healthy
    DEGRADED = "DEGRADED"         # Some pipelines below threshold, components gated
    FAIL = "FAIL"                 # Critical pipeline failure, run should be rejected


class NormalizationMethod(str, Enum):
    """Normalization method applied."""
    COHORT = "cohort"
    COHORT_WINSORIZED = "cohort_winsorized"
    COHORT_SHRINKAGE = "cohort_shrinkage"
    STAGE_FALLBACK = "stage_fallback"
    GLOBAL_FALLBACK = "global_fallback"
    NONE = "none"


# =============================================================================
# DATACLASSES
# =============================================================================

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

    # Component scores
    clinical_normalized: Decimal
    financial_normalized: Decimal
    catalyst_normalized: Decimal
    momentum_normalized: Optional[Decimal]
    valuation_normalized: Optional[Decimal]
    pos_normalized: Optional[Decimal]

    # Enhancements applied
    vol_adjustment: VolatilityAdjustment
    catalyst_decay: CatalystDecayResult
    interaction_terms: InteractionTerms
    smart_money_signal: SmartMoneySignal

    # Quality metrics
    severity: Severity
    confidence_overall: Decimal
    uncertainty_penalty: Decimal

    # Caps applied
    caps_applied: List[Dict[str, Any]]

    # Weights
    effective_weights: Dict[str, Decimal]

    # Cohort info
    cohort_key: str
    normalization_method: NormalizationMethod

    # Flags
    flags: List[str]

    # Audit
    determinism_hash: str
    score_breakdown: ScoreBreakdown


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _coalesce(*vals, default=None):
    """Return the first value that is not None.

    Unlike `or` chains, this correctly handles falsy values like Decimal("0")
    which should be treated as valid scores, not missing data.

    Example:
        _coalesce(Decimal("0"), Decimal("50"))  -> Decimal("0")  # Correct
        Decimal("0") or Decimal("50")           -> Decimal("50")  # Wrong!
    """
    for v in vals:
        if v is not None:
            return v
    return default


def _compute_catalyst_effective(
    catalyst_score_window: Optional[Decimal],
    catalyst_proximity_score: Optional[Decimal],
) -> Tuple[Decimal, bool, str]:
    """
    Compute effective catalyst score by blending window and proximity scores.

    Blending rules:
    1. If both scores present:
       catalyst_effective = 0.70 * catalyst_score + 0.30 * proximity_score
    2. If only proximity present (window is None or effectively missing):
       catalyst_effective = 0.85 * 50 + 0.15 * proximity_score
    3. If neither: default 50

    The proximity score captures future calendar events, ensuring that names
    with upcoming catalysts outside the optimal 15-45d window still contribute
    to the composite score.

    Args:
        catalyst_score_window: Window/importance-based catalyst score (Module 3 score_blended)
        catalyst_proximity_score: Proximity score from calendar events

    Returns:
        (catalyst_effective, proximity_blended_flag, blend_mode)
        - catalyst_effective: The blended score [0, 100]
        - proximity_blended_flag: True if proximity was factored into the score
        - blend_mode: String describing how the blend was computed
    """
    # Determine which scores are usable
    has_window = catalyst_score_window is not None
    has_proximity = catalyst_proximity_score is not None and catalyst_proximity_score > Decimal("0")

    if has_window and has_proximity:
        # Case 1: Both present - full blend
        effective = (
            CATALYST_WINDOW_WEIGHT * catalyst_score_window +
            CATALYST_PROXIMITY_WEIGHT * catalyst_proximity_score
        )
        proximity_blended = True
        blend_mode = "full_blend"

    elif has_proximity and not has_window:
        # Case 2: Only proximity - anchor to neutral 50 with proximity contribution
        effective = (
            CATALYST_DEFAULT_BASE * CATALYST_DEFAULT_SCORE +
            (Decimal("1") - CATALYST_DEFAULT_BASE) * catalyst_proximity_score
        )
        proximity_blended = True
        blend_mode = "proximity_only"

    elif has_window and not has_proximity:
        # Case 3: Only window score - use as-is
        effective = catalyst_score_window
        proximity_blended = False
        blend_mode = "window_only"

    else:
        # Case 4: Neither present - default neutral
        effective = CATALYST_DEFAULT_SCORE
        proximity_blended = False
        blend_mode = "default"

    # Clamp to valid range
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


# =============================================================================
# WINSORIZED NORMALIZATION (from v2)
# =============================================================================

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


# =============================================================================
# CONFIDENCE EXTRACTION (from v2)
# =============================================================================

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
    """Extract confidence for catalyst score.

    Handles:
    1. TickerCatalystSummaryV2 dataclass (catalyst_confidence enum attribute)
    2. vNext dict schema (integration.catalyst_confidence string)
    3. Legacy schema (n_high_confidence/events_detected_total counts)

    vNext confidence mapping:
        HIGH -> 0.70
        MED  -> 0.50
        LOW  -> 0.35
        UNKNOWN/None -> 0.30 (default, below gate threshold)
    """
    if not cat_data:
        return Decimal("0.3")

    # vNext schema mapping: string confidence to numeric
    CONFIDENCE_MAP = {
        "HIGH": Decimal("0.70"),
        "MED": Decimal("0.50"),
        "LOW": Decimal("0.35"),
        "UNKNOWN": Decimal("0.30"),
    }

    # 1. Try TickerCatalystSummaryV2 dataclass (catalyst_confidence is an enum)
    if hasattr(cat_data, 'catalyst_confidence') and hasattr(cat_data.catalyst_confidence, 'value'):
        conf_str = cat_data.catalyst_confidence.value  # Get enum value as string
        if conf_str in CONFIDENCE_MAP:
            return CONFIDENCE_MAP[conf_str]

    # 2. Try vNext dict schema: integration.catalyst_confidence (string)
    if isinstance(cat_data, dict):
        integration = cat_data.get("integration", {})
        conf_str = integration.get("catalyst_confidence")
        if conf_str and conf_str in CONFIDENCE_MAP:
            return CONFIDENCE_MAP[conf_str]

        # Also check for vNext event_summary.events_total
        event_summary = cat_data.get("event_summary", {})
        events_total = event_summary.get("events_total", 0)
        if events_total > 0 and conf_str is None:
            # Has events but no confidence string - use MED as default
            return Decimal("0.50")

    # 3. Try legacy dataclass with n_high_confidence
    if hasattr(cat_data, 'n_high_confidence'):
        n_high = cat_data.n_high_confidence
        n_events = cat_data.events_detected_total
        if n_events > 0:
            high_ratio = Decimal(str(n_high)) / Decimal(str(max(n_events, 1)))
            return _clamp(Decimal("0.4") + high_ratio * Decimal("0.5"), Decimal("0"), Decimal("1"))

    # 4. Try legacy dict schema (scores.n_high_confidence)
    if isinstance(cat_data, dict):
        scores = cat_data.get("scores", cat_data)
        n_high = scores.get("n_high_confidence", 0)
        n_events = scores.get("events_detected_total", 0)
        if n_events > 0:
            high_ratio = Decimal(str(n_high)) / Decimal(str(max(n_events, 1)))
            return _clamp(Decimal("0.4") + high_ratio * Decimal("0.5"), Decimal("0"), Decimal("1"))

    # Default: low confidence (below gate threshold)
    return Decimal("0.3")


def _extract_confidence_pos(pos_data: Optional[Dict]) -> Decimal:
    """Extract confidence for PoS score."""
    if not pos_data:
        return Decimal("0")
    conf = _to_decimal(pos_data.get("confidence"))
    if conf is not None:
        return _clamp(conf, Decimal("0"), Decimal("1"))
    return Decimal("0.7") if pos_data.get("pos_score") is not None else Decimal("0")


# =============================================================================
# MONOTONIC CAPS (from v2)
# =============================================================================

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


# =============================================================================
# DETERMINISM HASH
# =============================================================================

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


# =============================================================================
# CO-INVEST OVERLAY
# =============================================================================

def _enrich_with_coinvest(ticker: str, coinvest_signals: dict, as_of_date: date) -> dict:
    """Look up co-invest signal and return overlay fields with PIT safety.

    V2 ENHANCEMENT: Now extracts holder tier metadata for weighted scoring.

    Supports two input formats:
    1. Object format with .positions attribute (from live 13F aggregator)
    2. Pre-computed dict format from holdings_snapshots.json conversion
    """
    signal = coinvest_signals.get(ticker)
    if not signal:
        return {
            "coinvest_overlap_count": 0,
            "coinvest_holders": [],
            "coinvest_usable": False,
            "position_changes": {},
            "holder_tiers": {},
        }

    # Handle pre-computed dict format from _convert_holdings_to_coinvest
    # This format already has coinvest_overlap_count, coinvest_holders, etc.
    if isinstance(signal, dict) and "coinvest_overlap_count" in signal:
        # Convert holder_tiers from nested dict format {"cik": {"tier": 1, "name": "..."}}
        # to simple format {"cik": 1} expected by compute_smart_money_signal
        raw_tiers = signal.get("holder_tiers", {})
        normalized_tiers = {}
        for holder_id, tier_info in raw_tiers.items():
            if isinstance(tier_info, dict):
                normalized_tiers[holder_id] = tier_info.get("tier", 2)
            elif isinstance(tier_info, int):
                normalized_tiers[holder_id] = tier_info
            else:
                normalized_tiers[holder_id] = 2  # Default to tier 2

        return {
            "coinvest_overlap_count": signal.get("coinvest_overlap_count", 0),
            "coinvest_holders": signal.get("coinvest_holders", []),
            "coinvest_usable": signal.get("coinvest_overlap_count", 0) > 0,
            "position_changes": signal.get("position_changes", {}),
            "holder_tiers": normalized_tiers,
        }

    # Handle object format with .positions attribute (original behavior)
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

    # Extract holder tiers from position metadata if available
    # This comes from the elite_managers registry during 13F parsing
    holder_tiers: Dict[str, int] = {}
    for p in pit_positions:
        manager_name = p.manager_name
        if manager_name not in holder_tiers:
            # Check if position has tier metadata
            tier = getattr(p, 'manager_tier', None)
            if tier is not None:
                holder_tiers[manager_name] = tier
            # Otherwise, compute_smart_money_signal will use name-based lookup

    # Compute position changes (if previous quarter data available)
    position_changes = {}
    # Check if positions have change_type metadata (from 13F comparison)
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


# =============================================================================
# COHORT NORMALIZATION (V3 with Shrinkage)
# =============================================================================

def _apply_cohort_normalization_v3(
    members: List[Dict],
    global_stats: Dict[str, Tuple[Decimal, Decimal]],
    include_pos: bool = False,
    use_shrinkage: bool = True,
) -> NormalizationMethod:
    """
    Apply V3 cohort normalization with optional shrinkage.

    Args:
        members: List of member records to normalize
        global_stats: Dict of component -> (mean, std) from global population
        include_pos: Whether to normalize PoS scores
        use_shrinkage: Whether to apply Bayesian shrinkage for small cohorts

    Returns:
        NormalizationMethod indicating what was applied
    """
    if not members:
        return NormalizationMethod.NONE

    n = len(members)

    # Extract raw scores
    clin_scores = [m["clinical_raw"] or Decimal("0") for m in members]
    fin_scores = [m["financial_raw"] or Decimal("0") for m in members]
    cat_scores = [m["catalyst_raw"] or Decimal("0") for m in members]

    # Decide normalization method based on cohort size
    if n >= MIN_COHORT_SIZE:
        if use_shrinkage and n < 20:
            # Use shrinkage for medium cohorts
            clin_mean, clin_std = global_stats.get("clinical", (Decimal("50"), Decimal("20")))
            fin_mean, fin_std = global_stats.get("financial", (Decimal("50"), Decimal("20")))
            cat_mean, cat_std = global_stats.get("catalyst", (Decimal("50"), Decimal("20")))

            clin_norm, _ = shrinkage_normalize(clin_scores, clin_mean, clin_std)
            fin_norm, _ = shrinkage_normalize(fin_scores, fin_mean, fin_std)
            cat_norm, _ = shrinkage_normalize(cat_scores, cat_mean, cat_std)

            method = NormalizationMethod.COHORT_SHRINKAGE
        else:
            # Use winsorized rank for large cohorts
            clin_norm, clin_w = _rank_normalize_winsorized(clin_scores)
            fin_norm, fin_w = _rank_normalize_winsorized(fin_scores)
            cat_norm, cat_w = _rank_normalize_winsorized(cat_scores)

            method = NormalizationMethod.COHORT_WINSORIZED if (clin_w or fin_w or cat_w) else NormalizationMethod.COHORT
    else:
        # Small cohort - fall back to global stats with heavy shrinkage
        clin_mean, clin_std = global_stats.get("clinical", (Decimal("50"), Decimal("20")))
        fin_mean, fin_std = global_stats.get("financial", (Decimal("50"), Decimal("20")))
        cat_mean, cat_std = global_stats.get("catalyst", (Decimal("50"), Decimal("20")))

        clin_norm, _ = shrinkage_normalize(clin_scores, clin_mean, clin_std)
        fin_norm, _ = shrinkage_normalize(fin_scores, fin_mean, fin_std)
        cat_norm, _ = shrinkage_normalize(cat_scores, cat_mean, cat_std)

        method = NormalizationMethod.GLOBAL_FALLBACK

    # Handle PoS normalization
    pos_norm = None
    if include_pos:
        pos_scores = [m.get("pos_raw") or Decimal("0") for m in members]
        if any(p > 0 for p in pos_scores):
            if n >= MIN_COHORT_SIZE:
                pos_norm, _ = _rank_normalize_winsorized(pos_scores)
            else:
                pos_mean, pos_std = global_stats.get("pos", (Decimal("50"), Decimal("20")))
                pos_norm, _ = shrinkage_normalize(pos_scores, pos_mean, pos_std)

    # Apply to members
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
# SINGLE TICKER SCORING
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
    # cat_window = the window/importance-based score from Module 3 (score_blended)
    # cat_proximity = the proximity score from calendar events
    if hasattr(cat_data, 'score_blended'):
        cat_window = _to_decimal(cat_data.score_blended)
        cat_proximity = _to_decimal(getattr(cat_data, 'catalyst_proximity_score', 0)) or Decimal("0")
        cat_delta = _to_decimal(getattr(cat_data, 'catalyst_delta_score', 0)) or Decimal("0")
        # Try both field names for days to catalyst (schema uses catalyst_window_days)
        # Use _coalesce to correctly handle days_to_cat=0 (catalyst today) as valid
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
        # Try both field names (integration.catalyst_window_days or scores.days_to_nearest_catalyst)
        # Use _coalesce to correctly handle days_to_cat=0 (catalyst today) as valid
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

    # Compute effective catalyst score by blending window and proximity
    # This ensures upcoming calendar events contribute even when no events fall in the optimal window
    cat_effective, cat_proximity_blended, cat_blend_mode = _compute_catalyst_effective(
        cat_window, cat_proximity
    )
    # Use the effective score as the raw catalyst input
    cat_raw = cat_effective
    if cat_proximity_blended:
        flags.append("catalyst_proximity_blended")

    # Get normalized scores
    # Use _coalesce to correctly handle Decimal("0") as a valid score, not missing
    clin_norm = _coalesce(normalized_scores.get("clinical"), clin_raw, default=Decimal("50"))
    fin_norm = _coalesce(normalized_scores.get("financial"), fin_raw, default=Decimal("50"))
    cat_norm = _coalesce(normalized_scores.get("catalyst"), cat_raw, default=Decimal("50"))
    pos_norm = _coalesce(normalized_scores.get("pos"), pos_raw, default=Decimal("0"))

    # Extract financial metadata for interactions
    runway_months = _to_decimal(fin_data.get("runway_months"))
    liquidity_status = fin_data.get("liquidity_gate_status")
    dilution_bucket = fin_data.get("dilution_risk_bucket")
    market_cap_mm = _to_decimal(fin_data.get("market_cap_mm"))

    # Get stage and trial count for valuation
    lead_phase = clin_data.get("lead_phase")
    stage = _stage_bucket(lead_phase)
    trial_count = clin_data.get("trial_count", 0)

    # Extract market data for volatility and momentum (multi-window)
    annualized_vol = None
    momentum_input = MultiWindowMomentumInput()
    # Track momentum source for observability (prices vs 13f)
    momentum_source = "prices"  # Default: derived from daily prices

    if market_data:
        # Volatility
        vol_val = market_data.get("volatility_252d")
        annualized_vol = _to_decimal(vol_val if vol_val is not None else market_data.get("annualized_volatility"))

        # Check if momentum came from 13F injection (vs price-derived)
        if market_data.get("_momentum_source") == "13f":
            momentum_source = "13f"

        # Multi-window returns (20d, 60d, 120d)
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

    # 2. Momentum signal (with multi-window fallback for improved coverage)
    momentum = compute_momentum_signal_with_fallback(
        momentum_input,
        use_vol_adjusted_alpha=False,  # Default to raw alpha (can enable for vol-adjusted)
    )
    momentum_norm = momentum.momentum_score

    # Track momentum data status for diagnostics
    if momentum.data_status == "missing_prices":
        flags.append("momentum_missing_prices")
    elif momentum.data_status == "computed_low_conf":
        flags.append("momentum_low_confidence")

    # Flag strong momentum (only when data is available)
    if momentum.alpha_60d is not None:
        if momentum.alpha_60d >= Decimal("0.10"):
            flags.append("strong_positive_momentum")
        elif momentum.alpha_60d <= Decimal("-0.10"):
            flags.append("strong_negative_momentum")

    # Track which window was used
    if momentum.window_used:
        flags.append(f"momentum_window_{momentum.window_used}d")

    # Flag incomplete momentum data (backwards compat)
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

    # Apply decay to catalyst score
    cat_norm_decayed = apply_catalyst_decay(cat_norm, decay)

    # 5. Smart money signal (V2: tier-weighted with saturation)
    smart_money = compute_smart_money_signal(
        coinvest_data.get("coinvest_overlap_count", 0),
        coinvest_data.get("coinvest_holders", []),
        coinvest_data.get("position_changes"),
        coinvest_data.get("holder_tiers"),  # V2: pass tier metadata
    )
    if smart_money.holders_increasing:
        flags.append("smart_money_buying")
    if smart_money.holders_decreasing:
        flags.append("smart_money_selling")
    # V2: Flag high-conviction Tier1 holdings
    if smart_money.tier1_holders:
        flags.append("smart_money_tier1_present")

    # 6. Interaction terms (with gate status to prevent double-counting)
    # Compute runway_gate from runway_months directly (not from liquidity_status)
    # This ensures the "late-stage distress / runway" interaction fires on the correct condition
    if runway_months is None:
        runway_gate = "UNKNOWN"
    elif runway_months < Decimal("6"):
        runway_gate = "FAIL"
    elif runway_months >= Decimal("12"):
        runway_gate = "PASS"
    else:
        runway_gate = "UNKNOWN"  # 6-12 months is borderline
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

    # Apply volatility confidence penalty
    conf_clin = _clamp(conf_clin - vol_adj.confidence_penalty, Decimal("0.1"), Decimal("1"))
    conf_fin = _clamp(conf_fin - vol_adj.confidence_penalty, Decimal("0.1"), Decimal("1"))
    conf_cat = _clamp(conf_cat - vol_adj.confidence_penalty, Decimal("0.1"), Decimal("1"))

    # =========================================================================
    # WEIGHT COMPUTATION
    # =========================================================================

    # Apply regime adjustments
    regime_weights = apply_regime_to_weights(base_weights, regime)

    # Apply volatility adjustment to weights
    vol_adjusted_weights = {
        k: v * vol_adj.weight_adjustment_factor
        for k, v in regime_weights.items()
    }

    # Apply confidence weighting
    confidences = {
        "clinical": conf_clin,
        "financial": conf_fin,
        "catalyst": conf_cat,
        "pos": conf_pos,
        "momentum": momentum.confidence,
        "valuation": valuation.confidence,
    }

    # Track gated components (below confidence threshold)
    gated_components = []

    effective_weights = {}
    for comp, base_w in vol_adjusted_weights.items():
        conf = confidences.get(comp, Decimal("0.5"))

        # CONFIDENCE GATE: If confidence is below threshold, component contributes ZERO
        # This prevents missing data from being masked by neutral (50) defaults
        if conf < CONFIDENCE_GATE_THRESHOLD:
            effective_weights[comp] = Decimal("0")
            gated_components.append(comp)
            flags.append(f"{comp}_confidence_gated")
        else:
            # Confidence-weighted: low confidence reduces weight (range: 50-100% of base)
            eff_w = base_w * (Decimal("0.5") + conf * Decimal("0.5"))
            effective_weights[comp] = eff_w

    # Renormalize (only non-gated components contribute)
    total = sum(effective_weights.values())
    if total > EPS:
        effective_weights = {k: _quantize_weight(v / total) for k, v in effective_weights.items()}

    # =========================================================================
    # COMPOSITE SCORE COMPUTATION
    # =========================================================================

    # Build component contributions
    component_scores = []
    contributions = {}

    # Core components
    core_scores = {
        "clinical": (clin_norm, clin_raw, conf_clin),
        "financial": (fin_norm, fin_raw, conf_fin),
        "catalyst": (cat_norm_decayed, cat_raw, conf_cat),  # Use decayed catalyst
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

    # Enhancement components (if in enhanced mode)
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

    # Track PoS contribution separately for delta capping
    pos_contrib_raw = Decimal("0")
    pos_contrib_capped = Decimal("0")
    pos_delta_was_capped = False

    if mode == ScoringMode.ENHANCED and pos_raw is not None:
        w_eff = effective_weights.get("pos", Decimal("0"))
        pos_contrib_raw = pos_norm * w_eff

        # Cap PoS contribution to prevent mapping surprises from
        # dominating the composite score (±POS_DELTA_CAP points max impact)
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
    # AGGREGATION (Hybrid weighted-sum + weakest-link)
    # =========================================================================

    weighted_sum = sum(contributions.values())

    # Weakest-link for critical components
    critical_scores = []
    for c in CRITICAL_COMPONENTS:
        if c in contributions and c in effective_weights and effective_weights[c] > EPS:
            # Normalize contribution by weight to get underlying score
            underlying = contributions[c] / effective_weights[c]
            critical_scores.append(underlying)

    min_critical = min(critical_scores) if critical_scores else weighted_sum

    # Hybrid: alpha * weighted_sum + (1-alpha) * weakest_link
    pre_penalty = HYBRID_ALPHA * weighted_sum + (Decimal("1") - HYBRID_ALPHA) * min_critical

    # Add interaction adjustment
    pre_penalty = pre_penalty + interactions.total_interaction_adjustment
    pre_penalty = _quantize_score(pre_penalty)

    # =========================================================================
    # PENALTIES AND CAPS
    # =========================================================================

    # Uncertainty penalty
    subfactors = [clin_raw, fin_raw, cat_raw]
    if mode == ScoringMode.ENHANCED:
        subfactors.append(pos_raw)
    missing_count = sum(1 for x in subfactors if x is None)
    missing_pct = Decimal(str(missing_count / len(subfactors)))
    uncertainty_penalty = min(MAX_UNCERTAINTY_PENALTY, missing_pct)

    post_uncertainty = pre_penalty * (Decimal("1") - uncertainty_penalty)

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

    post_severity = post_uncertainty * severity_multiplier
    post_severity = _quantize_score(post_severity)

    # Monotonic caps
    post_cap, caps_applied = _apply_monotonic_caps(
        post_severity,
        liquidity_status,
        runway_months,
        dilution_bucket
    )

    # Volatility score adjustment
    post_vol = apply_volatility_to_score(post_cap, vol_adj)

    # Delta bonus only (proximity is now blended into catalyst_effective)
    # Proximity bonus removed to avoid double-counting since it's already in the catalyst component
    delta_bonus = (cat_delta / Decimal("25")).quantize(SCORE_PRECISION)  # Max ±2 pts

    final_score = _clamp(post_vol + delta_bonus, Decimal("0"), Decimal("100"))
    final_score = _quantize_score(final_score)

    # =========================================================================
    # BUILD OUTPUT
    # =========================================================================

    # Add penalty flags
    if uncertainty_penalty > 0:
        flags.append("uncertainty_penalty_applied")
    if worst_severity == Severity.SEV2:
        flags.append("sev2_penalty_applied")
    elif worst_severity == Severity.SEV1:
        flags.append("sev1_penalty_applied")
    if caps_applied:
        flags.append("monotonic_cap_applied")

    # Determinism hash
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

    # Compute overall confidence
    confidence_overall = (
        conf_clin * Decimal("0.3") +
        conf_fin * Decimal("0.3") +
        conf_cat * Decimal("0.2") +
        momentum.confidence * Decimal("0.1") +
        valuation.confidence * Decimal("0.1")
    )
    confidence_overall = _clamp(confidence_overall, Decimal("0.1"), Decimal("0.9"))

    # Build score breakdown
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
        # Enhancement results for output
        "momentum_signal": {
            "score": str(momentum.momentum_score),
            "alpha_60d": str(momentum.alpha_60d) if momentum.alpha_60d else None,
            "confidence": str(momentum.confidence),
            "data_completeness": str(momentum.data_completeness),
            # V2 multi-window fields
            "window_used": momentum.window_used,
            "data_status": momentum.data_status,
            # V3.2: Source tracking for observability (prices vs 13f)
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
            # V2 fields
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
        # Catalyst effective blending debug fields
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


# =============================================================================
# MAIN COMPOSITE FUNCTION
# =============================================================================

def compute_module_5_composite_v3(
    universe_result: Dict[str, Any],
    financial_result: Dict[str, Any],
    catalyst_result: Dict[str, Any],
    clinical_result: Dict[str, Any],
    as_of_date: str,
    weights: Optional[Dict[str, Decimal]] = None,
    coinvest_signals: Optional[Dict] = None,
    enhancement_result: Optional[Dict[str, Any]] = None,
    market_data_by_ticker: Optional[Dict[str, Dict]] = None,
    historical_scores: Optional[List[Dict]] = None,
    historical_returns: Optional[Dict[Tuple[date, str], Decimal]] = None,
    use_adaptive_weights: bool = False,
    validate_inputs: bool = True,
    enforce_pit_gates: bool = True,
    previous_weights: Optional[Dict[str, Decimal]] = None,
    embargo_months: int = 1,
    shrinkage_lambda: Decimal = Decimal("0.70"),
    smooth_gamma: Decimal = Decimal("0.80"),
) -> Dict[str, Any]:
    """
    Compute composite scores with all v3 IC enhancements.

    This is the production-ready v3 composite scorer with:
    - Adaptive weight learning
    - Non-linear signal interactions
    - Peer-relative valuation
    - Catalyst signal decay
    - Price momentum
    - Shrinkage normalization
    - Smart money signals
    - Volatility-adjusted scoring
    - Regime-adaptive components
    - Monotonic caps
    - Confidence weighting
    - Hybrid aggregation

    Args:
        universe_result: Module 1 output with active/excluded securities
        financial_result: Module 2 output with financial scores
        catalyst_result: Module 3 output with catalyst summaries
        clinical_result: Module 4 output with clinical scores
        as_of_date: ISO date string for the scoring date
        weights: Optional custom weights (defaults based on available data)
        coinvest_signals: Optional co-invest overlay signals
        enhancement_result: Optional PoS/SI/regime enhancement data
        market_data_by_ticker: Optional dict mapping ticker to market data
            (volatility_252d, return_60d, xbi_return_60d)
        historical_scores: Optional historical scores for adaptive weights.
            Each dict must have 'as_of_date' field (date or ISO string).
        historical_returns: Optional Dict keyed by (as_of_date, ticker) -> forward return.
            CRITICAL FOR PIT SAFETY: The as_of_date in the key must be when the
            return period STARTS, not when it ends.
        use_adaptive_weights: Whether to compute adaptive weights
        validate_inputs: If True (default), validate upstream outputs
        embargo_months: Minimum months between score date and return measurement.
            Default 1 month ensures returns are fully realized before use.
        shrinkage_lambda: 0-1, higher = more shrinkage toward base_weights.
            Default 0.70 provides strong regularization to prevent overfitting.
        smooth_gamma: 0-1, higher = more smoothing toward prev_weights.
            Default 0.80 reduces weight volatility period-to-period.

    Returns:
        Dict with ranked_securities, excluded_securities, and diagnostics
    """
    # Validate upstream module outputs
    if validate_inputs:
        validate_module_1_output(universe_result)
        validate_module_2_output(financial_result)
        validate_module_3_output(catalyst_result)
        validate_module_4_output(clinical_result)

    # Parse as_of_date for PIT validation
    as_of_dt = datetime.strptime(as_of_date, "%Y-%m-%d").date()

    # Run PIT production gate (if enforcement enabled)
    production_gate_result = None
    if enforce_pit_gates and (use_adaptive_weights or historical_scores):
        production_gate_result = run_production_gate(
            as_of_date=as_of_dt,
            historical_scores=historical_scores,
            historical_returns=historical_returns,
            use_adaptive_weights=use_adaptive_weights,
            current_weights=weights,
            previous_weights=previous_weights,
        )

        if not production_gate_result.passed:
            logger.warning(
                f"Production gate FAILED: {production_gate_result.blocking_violations}. "
                f"Falling back to non-adaptive mode."
            )
            use_adaptive_weights = False  # Disable adaptive weights on failure
            historical_scores = None
            historical_returns = None

        if production_gate_result.warnings:
            logger.warning(f"Production gate warnings: {production_gate_result.warnings}")

    # Handle empty universe
    active_securities = universe_result.get("active_securities", [])
    if not active_securities:
        logger.warning("Module 5 v3: Empty universe provided")
        return _empty_result(as_of_date)

    logger.info(f"Module 5 v3: Computing IC-enhanced composite scores for {as_of_date}")

    # =========================================================================
    # EXTRACT ENHANCEMENT DATA
    # =========================================================================

    enhancement_applied = enhancement_result is not None
    pos_by_ticker = {}
    si_by_ticker = {}
    regime_adjustments = {}
    regime_name = "NEUTRAL"

    if enhancement_result:
        # Extract PoS scores
        for ps in enhancement_result.get("pos_scores", {}).get("scores", []):
            if ps.get("ticker"):
                pos_by_ticker[ps["ticker"].upper()] = ps

        # Extract SI signals
        si_data = enhancement_result.get("short_interest_scores") or {}
        for si in si_data.get("scores", []):
            if si.get("ticker"):
                si_by_ticker[si["ticker"].upper()] = si

        # Extract regime
        regime_data = enhancement_result.get("regime", {})
        regime_name = regime_data.get("regime", "NEUTRAL")
        regime_adjustments = regime_data.get("signal_adjustments", {})

        # Extract accuracy enhancements
        accuracy_data = enhancement_result.get("accuracy_enhancements") or {}
        accuracy_by_ticker = accuracy_data.get("adjustments", {})

    # =========================================================================
    # DETERMINE SCORING MODE AND WEIGHTS
    # =========================================================================

    market_data_dict = market_data_by_ticker or {}
    has_market_data = bool(market_data_dict)
    has_pos_data = bool(pos_by_ticker)

    if has_pos_data:
        mode = ScoringMode.ENHANCED
        base_weights = V3_ENHANCED_WEIGHTS.copy() if weights is None else weights
    elif has_market_data:
        mode = ScoringMode.PARTIAL
        base_weights = V3_PARTIAL_WEIGHTS.copy() if weights is None else weights
    else:
        mode = ScoringMode.DEFAULT
        base_weights = V3_DEFAULT_WEIGHTS.copy() if weights is None else weights

    # Adaptive weight learning (if enabled and data available)
    # Uses PIT-safe signature: historical_returns keyed by (as_of_date, ticker)
    adaptive_weights_result = None
    if use_adaptive_weights and historical_scores and historical_returns:
        adaptive_weights_result = compute_adaptive_weights(
            historical_scores,
            historical_returns,  # Dict[(date, ticker), Decimal]
            base_weights,
            asof_date=as_of_dt,
            embargo_months=embargo_months,
            shrinkage_lambda=shrinkage_lambda,
            smooth_gamma=smooth_gamma,
            prev_weights=previous_weights,
        )
        if adaptive_weights_result.confidence >= Decimal("0.4"):
            base_weights = adaptive_weights_result.weights
            mode = ScoringMode.ADAPTIVE
            logger.info(
                f"Using adaptive weights (method={adaptive_weights_result.optimization_method}, "
                f"confidence={adaptive_weights_result.confidence}, "
                f"training_periods={adaptive_weights_result.training_periods})"
            )

    # =========================================================================
    # INDEX MODULE OUTPUTS
    # =========================================================================

    # DETERMINISM: Sort active_tickers to ensure consistent iteration order
    # (set iteration order is non-deterministic due to Python hash randomization)
    active_tickers = sorted({s["ticker"] for s in universe_result.get("active_securities", [])})
    financial_by_ticker = {s["ticker"]: s for s in financial_result.get("scores", [])}
    catalyst_by_ticker = catalyst_result.get("summaries", {})
    clinical_by_ticker = {s["ticker"]: s for s in clinical_result.get("scores", [])}

    # =========================================================================
    # BUILD COMBINED RECORDS
    # =========================================================================

    combined = []
    excluded = []

    for ticker in active_tickers:
        fin = financial_by_ticker.get(ticker, {})
        cat = catalyst_by_ticker.get(ticker, {})
        clin = clinical_by_ticker.get(ticker, {})
        pos = pos_by_ticker.get(ticker.upper())
        si = si_by_ticker.get(ticker.upper())
        market = market_data_dict.get(ticker, {})

        # Extract raw scores
        fin_score = _to_decimal(extract_financial_score(fin))
        clin_score = _to_decimal(extract_clinical_score(clin))

        # Extract catalyst score
        if hasattr(cat, 'score_blended'):
            cat_score = _to_decimal(cat.score_blended)
        elif isinstance(cat, dict):
            scores = cat.get("scores", cat)
            cat_score = _to_decimal(scores.get("score_blended", scores.get("catalyst_score_net")))
        else:
            cat_score = None

        pos_score = _to_decimal(pos.get("pos_score")) if pos else None

        # Get severities
        severities = [
            fin.get("severity", "none"),
            clin.get("severity", "none"),
        ]
        if hasattr(cat, 'severe_negative_flag') and cat.severe_negative_flag:
            severities.append("sev1")
        elif isinstance(cat, dict) and cat.get("flags", {}).get("severe_negative_flag"):
            severities.append("sev1")

        worst_severity = _get_worst_severity(severities)

        # Exclude sev3
        if worst_severity == Severity.SEV3:
            excluded.append({
                "ticker": ticker,
                "reason": "sev3_gate",
                "severity": worst_severity.value,
            })
            continue

        # Get cohort info
        lead_phase = clin.get("lead_phase")
        market_cap_mm = fin.get("market_cap_mm")
        trial_count = clin.get("trial_count", 0)

        combined.append({
            "ticker": ticker,
            "clinical_raw": clin_score,
            "financial_raw": fin_score,
            "catalyst_raw": cat_score,
            "pos_raw": pos_score,
            "market_cap_bucket": _market_cap_bucket(market_cap_mm),
            "stage_bucket": _stage_bucket(lead_phase),
            "lead_phase": lead_phase,
            "market_cap_mm": _to_decimal(market_cap_mm),
            "trial_count": trial_count,
            "fin_data": fin,
            "cat_data": cat,
            "clin_data": clin,
            "pos_data": pos,
            "si_data": si,
            "market_data": market,
        })

    # =========================================================================
    # COMPUTE GLOBAL STATS FOR SHRINKAGE
    # =========================================================================

    global_stats = _compute_global_stats(combined)

    # =========================================================================
    # BUILD PEER VALUATION DATA
    # =========================================================================

    peer_valuations = [
        {
            "ticker": r["ticker"],
            "market_cap_mm": r["market_cap_mm"],
            "trial_count": r["trial_count"],
            "stage_bucket": r["stage_bucket"],
        }
        for r in combined
        if r["market_cap_mm"] is not None and r["trial_count"] > 0
    ]

    # =========================================================================
    # COHORT GROUPING AND NORMALIZATION
    # =========================================================================

    cohorts: Dict[str, List[Dict]] = {}
    for rec in combined:
        key = rec["stage_bucket"]
        rec["cohort_key"] = key
        cohorts.setdefault(key, []).append(rec)

    cohort_stats = {}
    for cohort_key, members in cohorts.items():
        method = _apply_cohort_normalization_v3(
            members,
            global_stats,
            include_pos=enhancement_applied,
            use_shrinkage=True,
        )
        cohort_stats[cohort_key] = {
            "count": len(members),
            "normalization_method": method.value,
        }

    # =========================================================================
    # ENRICH WITH CO-INVEST
    # =========================================================================

    as_of_dt = datetime.strptime(as_of_date, "%Y-%m-%d").date()
    for rec in combined:
        if coinvest_signals:
            rec["coinvest"] = _enrich_with_coinvest(rec["ticker"], coinvest_signals, as_of_dt)
        else:
            rec["coinvest"] = {"coinvest_overlap_count": 0, "coinvest_holders": [], "coinvest_usable": False, "position_changes": {}}

    # =========================================================================
    # APPLY ACCURACY ENHANCEMENTS (if available)
    # =========================================================================

    accuracy_by_ticker = accuracy_by_ticker if 'accuracy_by_ticker' in dir() else {}
    if accuracy_by_ticker:
        for rec in combined:
            ticker = rec["ticker"].upper()
            acc_adj = accuracy_by_ticker.get(ticker)
            if acc_adj:
                # Apply multipliers to normalized scores
                clin_mult = Decimal(acc_adj.get("clinical_adjustment", "1.00"))
                fin_mult = Decimal(acc_adj.get("financial_adjustment", "1.00"))
                cat_mult = Decimal(acc_adj.get("catalyst_adjustment", "1.00"))
                reg_bonus = Decimal(acc_adj.get("regulatory_bonus", "0"))

                if rec.get("clinical_normalized"):
                    adj_clinical = rec["clinical_normalized"] * clin_mult + reg_bonus
                    rec["clinical_normalized"] = max(Decimal("0"), min(Decimal("100"), adj_clinical))

                if rec.get("financial_normalized"):
                    adj_financial = rec["financial_normalized"] * fin_mult
                    rec["financial_normalized"] = max(Decimal("0"), min(Decimal("100"), adj_financial))

                if rec.get("catalyst_normalized"):
                    adj_catalyst = rec["catalyst_normalized"] * cat_mult
                    rec["catalyst_normalized"] = max(Decimal("0"), min(Decimal("100"), adj_catalyst))

                # Track adjustments in rec for audit
                rec["accuracy_adjustments_applied"] = acc_adj.get("adjustments_applied", [])

    # =========================================================================
    # SCORE EACH TICKER
    # =========================================================================

    scored = []
    for rec in combined:
        normalized_scores = {
            "clinical": rec.get("clinical_normalized"),
            "financial": rec.get("financial_normalized"),
            "catalyst": rec.get("catalyst_normalized"),
            "pos": rec.get("pos_normalized"),
        }

        result = _score_single_ticker_v3(
            ticker=rec["ticker"],
            fin_data=rec["fin_data"],
            cat_data=rec["cat_data"],
            clin_data=rec["clin_data"],
            pos_data=rec["pos_data"],
            si_data=rec["si_data"],
            market_data=rec["market_data"],
            coinvest_data=rec["coinvest"],
            base_weights=base_weights,
            regime=regime_name,
            mode=mode,
            normalized_scores=normalized_scores,
            cohort_key=rec["cohort_key"],
            normalization_method=rec.get("normalization_method", NormalizationMethod.COHORT),
            peer_valuations=peer_valuations,
        )

        result["market_cap_bucket"] = rec["market_cap_bucket"]
        result["stage_bucket"] = rec["stage_bucket"]
        result["cohort_key"] = rec["cohort_key"]
        result["coinvest"] = rec["coinvest"]
        scored.append(result)

    # =========================================================================
    # SORT AND RANK
    # =========================================================================

    # Sort by composite score (desc), then coinvest (desc), then ticker (asc)
    scored.sort(key=lambda x: (
        -x["composite_score"],
        -(x["coinvest"]["coinvest_overlap_count"] if x["coinvest"] else 0),
        x["ticker"]
    ))

    for i, rec in enumerate(scored):
        rec["composite_rank"] = i + 1

    # =========================================================================
    # FORMAT OUTPUT
    # =========================================================================

    ranked_securities = []
    for rec in scored:
        bd = rec["score_breakdown"]
        coinvest = rec.get("coinvest") or {}

        security_data = {
            "ticker": rec["ticker"],
            "composite_score": str(rec["composite_score"]),
            "composite_rank": rec["composite_rank"],
            "severity": rec["severity"].value,
            "flags": rec["flags"],
            "rankable": True,

            # Cohort info
            "market_cap_bucket": rec["market_cap_bucket"],
            "stage_bucket": rec["stage_bucket"],
            "cohort_key": rec["cohort_key"],
            "normalization_method": rec["normalization_method"],

            # Confidence
            "confidence_clinical": str(rec["confidence_clinical"]),
            "confidence_financial": str(rec["confidence_financial"]),
            "confidence_catalyst": str(rec["confidence_catalyst"]),
            "confidence_pos": str(rec["confidence_pos"]) if rec["confidence_pos"] else None,
            "confidence_overall": str(rec["confidence_overall"]),

            # Weights
            "effective_weights": {k: str(v) for k, v in rec["effective_weights"].items()},

            # Caps and penalties
            "monotonic_caps_applied": rec["caps_applied"],
            "uncertainty_penalty": str(rec["uncertainty_penalty"]),

            # Audit
            "determinism_hash": rec["determinism_hash"],
            "schema_version": SCHEMA_VERSION,

            # Score breakdown
            "score_breakdown": {
                "version": bd.version, "mode": bd.mode,
                "base_weights": bd.base_weights,
                "regime_adjustments": bd.regime_adjustments,
                "effective_weights": bd.effective_weights,
                "components": bd.components,
                "enhancements": bd.enhancements,
                "penalties_and_gates": bd.penalties_and_gates,
                "interaction_terms": bd.interaction_terms,
                "final": bd.final,
                "normalization_method": bd.normalization_method,
                "cohort_info": bd.cohort_info,
                "hybrid_aggregation": bd.hybrid_aggregation,
            },

            # Co-invest
            "coinvest_overlap_count": coinvest.get("coinvest_overlap_count", 0),
            "coinvest_holders": coinvest.get("coinvest_holders", []),
            "coinvest_usable": coinvest.get("coinvest_usable", False),

            # V3 Enhancement signals
            "momentum_signal": rec.get("momentum_signal"),
            "valuation_signal": rec.get("valuation_signal"),
            "smart_money_signal": rec.get("smart_money_signal"),
            "volatility_adjustment": rec.get("volatility_adjustment"),
            "catalyst_decay": rec.get("catalyst_decay"),
            "interaction_terms": rec.get("interaction_terms"),
            "catalyst_effective": rec.get("catalyst_effective"),
        }

        ranked_securities.append(security_data)

    # =========================================================================
    # BUILD DIAGNOSTIC COUNTS
    # =========================================================================

    diagnostic_counts = {
        "total_input": len(active_tickers),
        "rankable": len(ranked_securities),
        "excluded": len(excluded),
        "cohort_count": len(cohorts),

        # Enhancement coverage
        "with_pos_scores": sum(1 for r in ranked_securities if r.get("confidence_pos")),
        "with_market_data": sum(1 for r in ranked_securities if r.get("volatility_adjustment", {}).get("annualized_vol_pct")),
        "with_momentum_signal": sum(1 for r in ranked_securities if r.get("momentum_signal", {}).get("alpha_60d")),
        "with_valuation_signal": sum(1 for r in ranked_securities if r.get("valuation_signal", {}).get("peer_count", 0) >= 5),
        "with_smart_money": sum(1 for r in ranked_securities if r.get("coinvest_overlap_count", 0) > 0),

        # Momentum state breakdown (for debugging/attribution)
        # Categories are MUTUALLY EXCLUSIVE and sum to total_rankable:
        # 1. missing_prices: No price data available for any window
        # 2. computed_low_conf: Computed but confidence < 0.5
        # 3. applied_negative: Strong negative signal
        # 4. applied_positive: Strong positive signal
        # 5. applied_neutral: Signal computed but not strong either way
        "momentum_missing_prices": sum(
            1 for r in ranked_securities
            if "momentum_missing_prices" in r.get("flags", [])
        ),
        "momentum_computed_low_conf": sum(
            1 for r in ranked_securities
            if "momentum_low_confidence" in r.get("flags", [])
        ),
        "momentum_applied_negative": sum(
            1 for r in ranked_securities
            if "strong_negative_momentum" in r.get("flags", [])
            and "momentum_missing_prices" not in r.get("flags", [])
        ),
        "momentum_applied_positive": sum(
            1 for r in ranked_securities
            if "strong_positive_momentum" in r.get("flags", [])
            and "momentum_missing_prices" not in r.get("flags", [])
        ),

        # Window usage breakdown
        "momentum_window_20d": sum(
            1 for r in ranked_securities
            if "momentum_window_20d" in r.get("flags", [])
        ),
        "momentum_window_60d": sum(
            1 for r in ranked_securities
            if "momentum_window_60d" in r.get("flags", [])
        ),
        "momentum_window_120d": sum(
            1 for r in ranked_securities
            if "momentum_window_120d" in r.get("flags", [])
        ),

        # Legacy compat: "no_alpha" = missing_prices + computed_low_conf
        "momentum_no_alpha": sum(
            1 for r in ranked_securities
            if r.get("momentum_signal", {}).get("alpha_60d") is None
        ),
        "momentum_gated_with_data": sum(
            1 for r in ranked_securities
            if r.get("momentum_signal", {}).get("alpha_60d") is not None
            and "momentum_confidence_gated" in r.get("flags", [])
        ),

        # NEW v3.2: Stable coverage metrics for momentum
        # These three metrics are stable and avoid "coverage inflation"
        #
        # 1. momentum_computable: Any window computed (low_conf + applied)
        #    = tickers where we have at least one return window
        "momentum_computable": sum(
            1 for r in ranked_securities
            if r.get("momentum_signal", {}).get("window_used") is not None
        ),
        # 2. momentum_meaningful: Confidence >= 0.5 threshold
        #    = signals strong enough to be trusted
        "momentum_meaningful": sum(
            1 for r in ranked_securities
            if _to_decimal(r.get("momentum_signal", {}).get("confidence", "0")) >= Decimal("0.5")
        ),
        # 3. momentum_strong_signal: Score moved away from 50 by at least 2.5 points
        #    = signals strong enough to meaningfully affect rankings
        #    Note: "applied" total (neg + pos + neutral) is computed separately for consistency
        #
        #    Alpha anchoring (score = 50 + conf * alpha * 150):
        #    - With conf=0.7 (typical): |score-50| >= 2.5 requires |alpha| >= ~2.4%
        #    - With conf=0.9 (high):    |score-50| >= 2.5 requires |alpha| >= ~1.85%
        #    - Raw (no shrinkage):      |score-50| >= 2.5 requires |alpha| >= ~1.67%
        #    Inclusive boundary: score=47.5 or score=52.5 counts as strong.
        "momentum_strong_signal": sum(
            1 for r in ranked_securities
            if abs(_to_decimal(r.get("momentum_signal", {}).get("momentum_score", "50")) - Decimal("50")) >= Decimal("2.5")
            and r.get("momentum_signal", {}).get("window_used") is not None
        ),
        # 4. momentum_strong_and_effective: Strong signal AND high enough confidence to matter
        #    = signals that are both strong (|score-50| >= 2.5) AND have confidence >= 0.6
        #    This is the "portfolio impact" metric: signals likely to move composites
        "momentum_strong_and_effective": sum(
            1 for r in ranked_securities
            if abs(_to_decimal(r.get("momentum_signal", {}).get("momentum_score", "50")) - Decimal("50")) >= Decimal("2.5")
            and r.get("momentum_signal", {}).get("window_used") is not None
            and _to_decimal(r.get("momentum_signal", {}).get("confidence", "0")) >= Decimal("0.6")
        ),
        # 5. Momentum source breakdown (for observability)
        #    prices = computed from daily price returns
        #    13f = injected from institutional momentum data
        "momentum_source_prices": sum(
            1 for r in ranked_securities
            if r.get("momentum_signal", {}).get("source") == "prices"
            and r.get("momentum_signal", {}).get("window_used") is not None
        ),
        "momentum_source_13f": sum(
            1 for r in ranked_securities
            if r.get("momentum_signal", {}).get("source") == "13f"
            and r.get("momentum_signal", {}).get("window_used") is not None
        ),

        # Quality metrics
        "with_caps_applied": sum(1 for r in ranked_securities if r.get("monotonic_caps_applied")),
        "with_interaction_flags": sum(1 for r in ranked_securities if r.get("interaction_terms", {}).get("flags")),
        "high_volatility_count": sum(1 for r in ranked_securities if r.get("volatility_adjustment", {}).get("vol_bucket") == "high"),
        "low_volatility_count": sum(1 for r in ranked_securities if r.get("volatility_adjustment", {}).get("vol_bucket") == "low"),
        "in_catalyst_window": sum(1 for r in ranked_securities if r.get("catalyst_decay", {}).get("in_optimal_window")),

        # Catalyst coverage breakdown (new in v3.1)
        # Raw coverage = has any catalyst events (confidence > default 0.3)
        # Window coverage = in optimal catalyst window (15-45 days from event)
        "with_catalyst_events": sum(
            1 for r in ranked_securities
            if _to_decimal(r.get("confidence_catalyst")) and _to_decimal(r.get("confidence_catalyst")) > Decimal("0.3")
        ),
        # New in v3.2: Proximity-blended catalyst tracking
        # Counts tickers where proximity score was factored into catalyst_effective
        "with_catalyst_proximity_blended": sum(
            1 for r in ranked_securities
            if r.get("catalyst_effective", {}).get("catalyst_proximity_blended", False)
        ),
    }

    # =========================================================================
    # ENHANCEMENT DIAGNOSTICS
    # =========================================================================

    enhancement_diagnostics = {
        "regime": regime_name,
        "regime_adjustments": {k: str(v) for k, v in regime_adjustments.items()} if regime_adjustments else {},
        "mode": mode.value,
        "pos_coverage": len(pos_by_ticker),
        "market_data_coverage": len(market_data_dict),
    }

    # =========================================================================
    # PIPELINE HEALTH CHECK
    # =========================================================================

    total_rankable = len(ranked_securities) if ranked_securities else 1  # Avoid div/0

    # Calculate component coverage as fraction of universe
    component_coverage = {
        # catalyst_window = in optimal 15-45 day window (stricter)
        "catalyst": Decimal(str(diagnostic_counts.get("in_catalyst_window", 0))) / Decimal(str(total_rankable)),
        # catalyst_raw = has any catalyst events (confidence > 0.3 default)
        "catalyst_raw": Decimal(str(diagnostic_counts.get("with_catalyst_events", 0))) / Decimal(str(total_rankable)),
        "momentum": Decimal(str(diagnostic_counts.get("with_momentum_signal", 0))) / Decimal(str(total_rankable)),
        "smart_money": Decimal(str(diagnostic_counts.get("with_smart_money", 0))) / Decimal(str(total_rankable)),
        "market_data": Decimal(str(diagnostic_counts.get("with_market_data", 0))) / Decimal(str(total_rankable)),
        "pos": Decimal(str(diagnostic_counts.get("with_pos_scores", 0))) / Decimal(str(total_rankable)),
        "valuation": Decimal(str(diagnostic_counts.get("with_valuation_signal", 0))) / Decimal(str(total_rankable)),
    }

    # Collect components with confidence-gated entries (from individual security scoring)
    gated_component_counts = {}
    for sec in ranked_securities:
        for flag in sec.get("flags", []):
            if flag.endswith("_confidence_gated"):
                comp = flag.replace("_confidence_gated", "")
                gated_component_counts[comp] = gated_component_counts.get(comp, 0) + 1

    # Determine run status
    run_status = RunStatus.OK
    degraded_components = []
    health_warnings = []
    health_errors = []

    # Check each pipeline against thresholds
    for component, threshold in HEALTH_GATE_THRESHOLDS.items():
        coverage = component_coverage.get(component, Decimal("0"))
        if coverage < threshold:
            # Catalyst: check raw coverage for pipeline health, not window coverage
            if component == "catalyst":
                raw_coverage = component_coverage.get("catalyst_raw", Decimal("0"))
                if raw_coverage < Decimal("0.05"):  # <5% with events = pipeline broken
                    run_status = RunStatus.FAIL
                    health_errors.append(
                        f"CRITICAL: catalyst_raw coverage {raw_coverage*100:.1f}% - pipeline broken "
                        f"(window coverage: {coverage*100:.1f}%)"
                    )
                else:
                    # Raw coverage OK but window coverage low - just informational
                    health_warnings.append(
                        f"INFO: catalyst window coverage {coverage*100:.1f}% (raw: {raw_coverage*100:.1f}%) - "
                        f"events exist but few in optimal 15-45d window"
                    )
            elif component in ("momentum", "market_data"):
                # Momentum/market data below threshold degrades run
                if run_status != RunStatus.FAIL:
                    run_status = RunStatus.DEGRADED
                degraded_components.append(component)
                health_warnings.append(f"DEGRADED: {component} coverage {coverage*100:.1f}% < {threshold*100:.0f}% threshold")
            else:
                # Other components just warn
                degraded_components.append(component)
                health_warnings.append(f"WARNING: {component} coverage {coverage*100:.1f}% < {threshold*100:.0f}% threshold")

    # Check gated components (high rate of confidence gating indicates data issue)
    # NOTE: In biotech, high gating rates are common for optional enhancement components:
    # - valuation: pre-revenue companies lack comparable peers
    # - momentum: 13F-based momentum provides sparse coverage by design
    # - smart_money: 13F data only covers subset of universe
    # Only log as INFO since sparse coverage is expected, not a data pipeline failure
    for comp, count in gated_component_counts.items():
        gated_pct = Decimal(str(count)) / Decimal(str(total_rankable))
        if gated_pct > Decimal("0.5"):  # >50% of universe gated for this component
            # Log as info, not degradation - sparse optional component coverage is normal
            health_warnings.append(f"INFO: {comp} confidence-gated for {gated_pct*100:.1f}% of universe (sparse coverage expected)")

    # Log detailed momentum state breakdown for debugging/attribution
    # V2: Now tracks data status (missing_prices, computed_low_conf) and window usage
    mom_missing = diagnostic_counts.get("momentum_missing_prices", 0)
    mom_low_conf = diagnostic_counts.get("momentum_computed_low_conf", 0)
    mom_neg = diagnostic_counts.get("momentum_applied_negative", 0)
    mom_pos = diagnostic_counts.get("momentum_applied_positive", 0)

    # Window usage stats
    mom_20d = diagnostic_counts.get("momentum_window_20d", 0)
    mom_60d = diagnostic_counts.get("momentum_window_60d", 0)
    mom_120d = diagnostic_counts.get("momentum_window_120d", 0)

    # Neutral = has data, not low conf, not strong either way
    mom_with_data = mom_60d + mom_20d + mom_120d
    mom_neutral = mom_with_data - mom_neg - mom_pos - mom_low_conf
    if mom_neutral < 0:
        mom_neutral = 0

    # Active = actually contributed to scoring
    mom_active = mom_with_data - mom_low_conf
    mom_coverage_pct = (Decimal(str(mom_active)) / Decimal(str(total_rankable)) * 100) if total_rankable > 0 else Decimal("0")

    # Calculate average effective momentum weight across active securities
    avg_mom_weight = Decimal("0")
    total_mom_contribution = Decimal("0")
    if mom_active > 0:
        for sec in ranked_securities:
            flags = sec.get("flags", [])
            if "momentum_missing_prices" not in flags and "momentum_low_confidence" not in flags:
                eff_weights = sec.get("effective_weights", {})
                mom_weight = _to_decimal(eff_weights.get("momentum", "0")) or Decimal("0")
                avg_mom_weight += mom_weight
                # Track total momentum contribution to composite (for impact assessment)
                mom_score = _to_decimal(sec.get("momentum_signal", {}).get("score", "50")) or Decimal("50")
                total_mom_contribution += mom_score * mom_weight
        avg_mom_weight = (avg_mom_weight / Decimal(str(mom_active))).quantize(Decimal("0.001")) if mom_active > 0 else Decimal("0")

    # V3.2: Stable coverage metrics (avoids coverage inflation)
    mom_computable = diagnostic_counts.get("momentum_computable", 0)
    mom_meaningful = diagnostic_counts.get("momentum_meaningful", 0)
    mom_strong_signal = diagnostic_counts.get("momentum_strong_signal", 0)
    mom_strong_effective = diagnostic_counts.get("momentum_strong_and_effective", 0)
    mom_source_prices = diagnostic_counts.get("momentum_source_prices", 0)
    mom_source_13f = diagnostic_counts.get("momentum_source_13f", 0)
    # Fix: applied should equal neg + pos + neutral (all signals with data, not low_conf)
    # This matches the breakdown shown in applied[neg:X, pos:X, neutral:X]
    mom_applied_stable = mom_neg + mom_pos + mom_neutral

    # GUARDRAIL: Coverage collapsed alert
    # If coverage drops below 20%, emit loud warning (likely data pipeline issue)
    coverage_pct = (mom_applied_stable / total_rankable * 100) if total_rankable > 0 else 0
    if total_rankable > 0 and coverage_pct < 20:
        health_warnings.append(
            f"WARN: momentum coverage collapsed ({coverage_pct:.1f}%) - "
            f"check enrich_market_data_momentum outputs / market_data.json keys"
        )

    if mom_with_data > 0 or mom_missing > 0:
        health_warnings.append(
            f"INFO: momentum breakdown - "
            f"missing:{mom_missing}, low_conf:{mom_low_conf}, "
            f"applied[neg:{mom_neg}, pos:{mom_pos}, neutral:{mom_neutral}] | "
            f"windows[20d:{mom_20d}, 60d:{mom_60d}, 120d:{mom_120d}] | "
            f"coverage={mom_active}/{total_rankable} ({mom_coverage_pct:.1f}%), avg_weight={avg_mom_weight}"
        )
        # Single-line signal health dashboard for monitoring:
        # - applied: coverage (has data, not low_conf)
        # - meaningful: confidence >= 0.5
        # - strong: |score-50| >= 2.5
        # - strong+effective: strong AND confidence >= 0.6 (actually moves composite)
        # - sources: prices vs 13f breakdown
        health_warnings.append(
            f"INFO: momentum stable metrics - "
            f"computable:{mom_computable}, meaningful:{mom_meaningful}, "
            f"coverage_applied:{mom_applied_stable}/{total_rankable}, "
            f"strong:{mom_strong_signal}, strong_and_effective:{mom_strong_effective}, "
            f"sources[prices:{mom_source_prices}, 13f:{mom_source_13f}]"
        )

    # Log health status
    if run_status == RunStatus.FAIL:
        for err in health_errors:
            logger.error(err)
    for warn in health_warnings:
        logger.warning(warn)

    if run_status != RunStatus.OK:
        logger.warning(f"Run status: {run_status.value} | Degraded components: {degraded_components}")

    if adaptive_weights_result:
        enhancement_diagnostics["adaptive_weights"] = {
            "method": adaptive_weights_result.optimization_method,
            "confidence": str(adaptive_weights_result.confidence),
            "historical_ic": adaptive_weights_result.historical_ic_by_component,
        }

    # =========================================================================
    # RETURN RESULT
    # =========================================================================

    # Build production gate diagnostics
    pit_gate_diagnostics = None
    if production_gate_result:
        pit_gate_diagnostics = {
            "passed": production_gate_result.passed,
            "checks": [
                {
                    "name": c.check_name,
                    "status": c.status.value,
                    "violations": c.violations,
                }
                for c in production_gate_result.checks
            ],
            "blocking_violations": production_gate_result.blocking_violations,
            "warnings": production_gate_result.warnings,
            "recommendation": production_gate_result.recommendation,
        }

    # DETERMINISM: Sort excluded_securities by ticker for consistent output order
    excluded_sorted = sorted(excluded, key=lambda x: x["ticker"])

    # Build momentum_health for results JSON (scriptable A/B comparisons)
    momentum_health = {
        "coverage_applied": mom_applied_stable,
        "coverage_pct": round(coverage_pct, 1),
        "computable": mom_computable,
        "meaningful": mom_meaningful,
        "strong_signal": mom_strong_signal,
        "strong_and_effective": mom_strong_effective,
        "avg_weight": str(avg_mom_weight),
        "windows_used": {
            "20d": mom_20d,
            "60d": mom_60d,
            "120d": mom_120d,
        },
        "sources": {
            "prices": mom_source_prices,
            "13f": mom_source_13f,
        },
        "total_rankable": total_rankable,
        "as_of_date": as_of_date,
    }

    return {
        "as_of_date": as_of_date,
        "scoring_mode": mode.value,
        "run_status": run_status.value,
        "degraded_components": degraded_components,
        "component_coverage": {k: str(v) for k, v in sorted(component_coverage.items())},
        "gated_component_counts": gated_component_counts,
        "health_warnings": health_warnings,
        "health_errors": health_errors,
        "weights_used": {k: str(v) for k, v in sorted(base_weights.items())},
        "ranked_securities": ranked_securities,
        "excluded_securities": excluded_sorted,
        "cohort_stats": {k: v for k, v in sorted(cohort_stats.items())},
        "global_stats": {k: {"mean": str(v[0]), "std": str(v[1])} for k, v in sorted(global_stats.items())},
        "diagnostic_counts": diagnostic_counts,
        "enhancement_applied": enhancement_applied,
        "enhancement_diagnostics": enhancement_diagnostics,
        "momentum_health": momentum_health,  # V3.2: Persisted for A/B comparisons
        "pit_gate_diagnostics": pit_gate_diagnostics,
        "schema_version": SCHEMA_VERSION,
        "provenance": create_provenance(
            RULESET_VERSION,
            {"tickers": sorted(active_tickers), "weights": {k: str(v) for k, v in sorted(base_weights.items())}, "mode": mode.value},
            as_of_date,
        ),
    }


def _empty_result(as_of_date: str) -> Dict[str, Any]:
    """Return empty result structure."""
    return {
        "as_of_date": as_of_date,
        "scoring_mode": ScoringMode.DEFAULT.value,
        "run_status": RunStatus.OK.value,
        "degraded_components": [],
        "component_coverage": {},
        "gated_component_counts": {},
        "health_warnings": [],
        "health_errors": [],
        "weights_used": {k: str(v) for k, v in V3_DEFAULT_WEIGHTS.items()},
        "ranked_securities": [],
        "excluded_securities": [],
        "cohort_stats": {},
        "global_stats": {},
        "diagnostic_counts": {
            "total_input": 0, "rankable": 0, "excluded": 0, "cohort_count": 0,
            "with_pos_scores": 0, "with_market_data": 0, "with_momentum_signal": 0,
            "with_valuation_signal": 0, "with_smart_money": 0, "with_caps_applied": 0,
            "with_interaction_flags": 0, "high_volatility_count": 0, "low_volatility_count": 0,
            "in_catalyst_window": 0,
        },
        "enhancement_applied": False,
        "enhancement_diagnostics": None,
        "schema_version": SCHEMA_VERSION,
        "provenance": create_provenance(RULESET_VERSION, {"tickers": [], "weights": {}}, as_of_date),
    }


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Module 5 v3: IC-Enhanced Composite Ranker")
    parser.add_argument("--as-of-date", required=True, help="Analysis date (YYYY-MM-DD)")
    parser.add_argument("--universe", required=True, help="Path to Module 1 output JSON")
    parser.add_argument("--financial", required=True, help="Path to Module 2 output JSON")
    parser.add_argument("--catalyst", required=True, help="Path to Module 3 output JSON")
    parser.add_argument("--clinical", required=True, help="Path to Module 4 output JSON")
    parser.add_argument("--enhancement", help="Path to enhancement data JSON")
    parser.add_argument("--market-data", help="Path to market data JSON")
    parser.add_argument("--output", required=True, help="Output path for results")
    parser.add_argument("--adaptive-weights", action="store_true", help="Enable adaptive weight learning")

    args = parser.parse_args()

    # Load inputs
    with open(args.universe) as f:
        universe_result = json.load(f)
    with open(args.financial) as f:
        financial_result = json.load(f)
    with open(args.catalyst) as f:
        catalyst_result = json.load(f)
    with open(args.clinical) as f:
        clinical_result = json.load(f)

    enhancement_result = None
    if args.enhancement:
        with open(args.enhancement) as f:
            enhancement_result = json.load(f)

    market_data = None
    if args.market_data:
        with open(args.market_data) as f:
            market_data = json.load(f)

    # Compute
    result = compute_module_5_composite_v3(
        universe_result=universe_result,
        financial_result=financial_result,
        catalyst_result=catalyst_result,
        clinical_result=clinical_result,
        as_of_date=args.as_of_date,
        enhancement_result=enhancement_result,
        market_data_by_ticker=market_data,
        use_adaptive_weights=args.adaptive_weights,
    )

    # Write output
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"Module 5 v3 complete: {result['diagnostic_counts']['rankable']} securities ranked")
    sys.exit(0)
