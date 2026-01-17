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
    """Extract confidence for catalyst score."""
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
            {"name": c.name, "raw": str(c.raw) if c.raw else None,
             "normalized": str(c.normalized) if c.normalized else None,
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
    """Look up co-invest signal and return overlay fields with PIT safety."""
    signal = coinvest_signals.get(ticker)
    if not signal:
        return {"coinvest_overlap_count": 0, "coinvest_holders": [], "coinvest_usable": False, "position_changes": {}}

    pit_positions = [p for p in signal.positions if p.filing_date < as_of_date]
    if not pit_positions:
        return {"coinvest_overlap_count": 0, "coinvest_holders": [], "coinvest_usable": False, "position_changes": {}}

    holders = sorted(set(p.manager_name for p in pit_positions))

    # Compute position changes (if previous quarter data available)
    position_changes = {}
    # This would require historical 13F data - simplified here
    # In production, compare current quarter to previous quarter

    return {
        "coinvest_overlap_count": len(holders),
        "coinvest_holders": holders,
        "coinvest_quarter": _quarter_from_date(pit_positions[0].report_date),
        "coinvest_usable": True,
        "position_changes": position_changes,
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
    fin_raw = _to_decimal(fin_data.get("financial_normalized") or fin_data.get("financial_score"))
    clin_raw = _to_decimal(clin_data.get("clinical_score"))
    pos_raw = _to_decimal(pos_data.get("pos_score")) if pos_data else None

    # Extract catalyst scores and metadata
    if hasattr(cat_data, 'score_blended'):
        cat_raw = _to_decimal(cat_data.score_blended)
        cat_proximity = _to_decimal(getattr(cat_data, 'catalyst_proximity_score', 0)) or Decimal("0")
        cat_delta = _to_decimal(getattr(cat_data, 'catalyst_delta_score', 0)) or Decimal("0")
        days_to_cat = getattr(cat_data, 'days_to_nearest_catalyst', None)
        cat_event_type = getattr(cat_data, 'nearest_catalyst_type', "DEFAULT")
    elif isinstance(cat_data, dict):
        scores = cat_data.get("scores", cat_data)
        cat_raw = _to_decimal(scores.get("score_blended", scores.get("catalyst_score_net")))
        cat_proximity = _to_decimal(scores.get("catalyst_proximity_score", 0)) or Decimal("0")
        cat_delta = _to_decimal(scores.get("catalyst_delta_score", 0)) or Decimal("0")
        days_to_cat = scores.get("days_to_nearest_catalyst")
        cat_event_type = scores.get("nearest_catalyst_type", "DEFAULT")
    else:
        cat_raw = None
        cat_proximity = Decimal("0")
        cat_delta = Decimal("0")
        days_to_cat = None
        cat_event_type = "DEFAULT"

    # Get normalized scores
    clin_norm = normalized_scores.get("clinical") or clin_raw or Decimal("50")
    fin_norm = normalized_scores.get("financial") or fin_raw or Decimal("50")
    cat_norm = normalized_scores.get("catalyst") or cat_raw or Decimal("50")
    pos_norm = normalized_scores.get("pos") or pos_raw or Decimal("0")

    # Extract financial metadata for interactions
    runway_months = _to_decimal(fin_data.get("runway_months"))
    liquidity_status = fin_data.get("liquidity_gate_status")
    dilution_bucket = fin_data.get("dilution_risk_bucket")
    market_cap_mm = _to_decimal(fin_data.get("market_cap_mm"))

    # Get stage and trial count for valuation
    lead_phase = clin_data.get("lead_phase")
    stage = _stage_bucket(lead_phase)
    trial_count = clin_data.get("trial_count", 0)

    # Extract market data for volatility and momentum
    annualized_vol = None
    return_60d = None
    benchmark_return_60d = None

    if market_data:
        annualized_vol = _to_decimal(market_data.get("volatility_252d") or market_data.get("annualized_volatility"))
        return_60d = _to_decimal(market_data.get("return_60d"))
        benchmark_return_60d = _to_decimal(market_data.get("xbi_return_60d") or market_data.get("benchmark_return_60d"))

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
    momentum = compute_momentum_signal(return_60d, benchmark_return_60d)
    momentum_norm = momentum.momentum_score
    if momentum.alpha_60d is not None:
        if momentum.alpha_60d >= Decimal("0.10"):
            flags.append("strong_positive_momentum")
        elif momentum.alpha_60d <= Decimal("-0.10"):
            flags.append("strong_negative_momentum")

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

    # 5. Smart money signal
    smart_money = compute_smart_money_signal(
        coinvest_data.get("coinvest_overlap_count", 0),
        coinvest_data.get("coinvest_holders", []),
        coinvest_data.get("position_changes"),
    )
    if smart_money.holders_increasing:
        flags.append("smart_money_buying")
    if smart_money.holders_decreasing:
        flags.append("smart_money_selling")

    # 6. Interaction terms (with gate status to prevent double-counting)
    # Convert gate statuses from v2 format
    runway_gate = "PASS" if liquidity_status == "PASS" else (
        "FAIL" if liquidity_status == "FAIL" else "UNKNOWN"
    )
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

    effective_weights = {}
    for comp, base_w in vol_adjusted_weights.items():
        conf = confidences.get(comp, Decimal("0.5"))
        # Confidence-weighted: low confidence reduces weight
        eff_w = base_w * (Decimal("0.5") + conf * Decimal("0.5"))  # Range: 50-100% of base
        effective_weights[comp] = eff_w

    # Renormalize
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

    if mode == ScoringMode.ENHANCED and pos_raw is not None:
        w_eff = effective_weights.get("pos", Decimal("0"))
        contrib = pos_norm * w_eff
        contributions["pos"] = contrib
        component_scores.append(ComponentScore(
            name="pos",
            raw=pos_raw,
            normalized=pos_norm,
            confidence=conf_pos,
            weight_base=base_weights.get("pos", Decimal("0")),
            weight_effective=w_eff,
            contribution=_quantize_score(contrib),
            notes=[],
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

    # Proximity and delta bonuses (scaled down in v3)
    proximity_bonus = (cat_proximity / Decimal("25")).quantize(SCORE_PRECISION)  # Max +4 pts
    delta_bonus = (cat_delta / Decimal("25")).quantize(SCORE_PRECISION)  # Max ±2 pts

    final_score = _clamp(post_vol + proximity_bonus + delta_bonus, Decimal("0"), Decimal("100"))
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
            "name": c.name, "raw": str(c.raw) if c.raw else None,
            "normalized": str(c.normalized) if c.normalized else None,
            "confidence": str(c.confidence), "weight_base": str(c.weight_base),
            "weight_effective": str(c.weight_effective), "contribution": str(c.contribution),
            "decay_factor": str(c.decay_factor) if c.decay_factor else None,
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
            "proximity_bonus": str(proximity_bonus),
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
        },
        "valuation_signal": {
            "score": str(valuation.valuation_score),
            "peer_count": valuation.peer_count,
            "confidence": str(valuation.confidence),
        },
        "smart_money_signal": {
            "score": str(smart_money.smart_money_score),
            "overlap_count": smart_money.overlap_count,
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

    active_tickers = {s["ticker"] for s in universe_result.get("active_securities", [])}
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

        # Quality metrics
        "with_caps_applied": sum(1 for r in ranked_securities if r.get("monotonic_caps_applied")),
        "with_interaction_flags": sum(1 for r in ranked_securities if r.get("interaction_terms", {}).get("flags")),
        "high_volatility_count": sum(1 for r in ranked_securities if r.get("volatility_adjustment", {}).get("vol_bucket") == "high"),
        "low_volatility_count": sum(1 for r in ranked_securities if r.get("volatility_adjustment", {}).get("vol_bucket") == "low"),
        "in_catalyst_window": sum(1 for r in ranked_securities if r.get("catalyst_decay", {}).get("in_optimal_window")),
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

    return {
        "as_of_date": as_of_date,
        "scoring_mode": mode.value,
        "weights_used": {k: str(v) for k, v in base_weights.items()},
        "ranked_securities": ranked_securities,
        "excluded_securities": excluded,
        "cohort_stats": cohort_stats,
        "global_stats": {k: {"mean": str(v[0]), "std": str(v[1])} for k, v in global_stats.items()},
        "diagnostic_counts": diagnostic_counts,
        "enhancement_applied": enhancement_applied,
        "enhancement_diagnostics": enhancement_diagnostics,
        "pit_gate_diagnostics": pit_gate_diagnostics,
        "schema_version": SCHEMA_VERSION,
        "provenance": create_provenance(
            RULESET_VERSION,
            {"tickers": list(active_tickers), "weights": {k: str(v) for k, v in base_weights.items()}, "mode": mode.value},
            as_of_date,
        ),
    }


def _empty_result(as_of_date: str) -> Dict[str, Any]:
    """Return empty result structure."""
    return {
        "as_of_date": as_of_date,
        "scoring_mode": ScoringMode.DEFAULT.value,
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
