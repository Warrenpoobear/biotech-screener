"""
V3 Production Integration Spec - Module 5 Composite Ranker

Based on backtest analysis showing:
- v3 wins NET-OF-COSTS with lower turnover + lower cost drag
- v3 has LOW single-name dependency (robust return distribution)
- v2's performance dominated by single-name lottery ticket (207% top-1 contribution)

This configuration defines:
1. Feature flag defaults (what's ON/OFF)
2. Logging requirements
3. Fallback thresholds
4. Sanity override mechanism

Author: Wake Robin Capital Management
Version: 1.0.0
Created: 2026-01-18
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Set
from enum import Enum


# =============================================================================
# FEATURE FLAG DEFAULTS
# =============================================================================

class FeatureFlags:
    """
    V3 Feature Flags - Default Configuration

    Features are grouped into three tiers:
    - STABLE: Battle-tested, enabled by default, no live-cycle needed
    - MONITORED: Enabled by default, but logged for watchlist
    - EXPERIMENTAL: Disabled by default, requires explicit opt-in
    """

    # =========================================================================
    # TIER 1: STABLE FEATURES (ON by default, no special logging)
    # =========================================================================

    # Core v2 features carried forward (proven stable)
    MONOTONIC_CAPS: bool = True              # Risk gates can't be "outvoted"
    CONFIDENCE_WEIGHTING: bool = True        # Module confidence affects weights
    HYBRID_AGGREGATION: bool = True          # weighted_sum + weakest_link blend
    DETERMINISM_HASH: bool = True            # SHA256 audit trail
    WINSORIZED_NORMALIZATION: bool = True    # Percentile rank with winsorization

    # V3 features proven stable in backtest
    CATALYST_DECAY: bool = True              # Time-based IC decay for catalyst events
    PRICE_MOMENTUM: bool = True              # 60-day relative strength vs XBI
    PEER_VALUATION: bool = True              # MCap-per-asset peer comparison
    SMART_MONEY_OVERLAP: bool = True         # 13F overlap counting

    # =========================================================================
    # TIER 2: MONITORED FEATURES (ON by default, logged for watchlist)
    # =========================================================================

    # These features are enabled but require logging for the first N cycles
    # If anomalies detected, can be disabled without code change

    VOLATILITY_ADJUSTED_SCORING: bool = True  # Vol adjustment to scores/weights
    SHRINKAGE_NORMALIZATION: bool = True      # Bayesian cohort adjustment

    # Smart money V2 enhancements
    SMART_MONEY_TIER_WEIGHTING: bool = True   # Tier1/Tier2/Tier3 weighting
    SMART_MONEY_POSITION_CHANGES: bool = True # NEW/INCREASE/HOLD/DECREASE/EXIT

    # =========================================================================
    # TIER 3: EXPERIMENTAL FEATURES (OFF by default, require explicit opt-in)
    # =========================================================================

    # These features showed promise but need more live validation
    # Can be enabled via --enable-experimental or config override

    INTERACTION_TERMS: bool = False           # Cross-factor synergies/penalties
    ADAPTIVE_WEIGHT_LEARNING: bool = False    # Historical IC optimization
    REGIME_ADAPTIVE_WEIGHTS: bool = False     # Bull/Bear regime adjustments

    # Cap for interaction term impact (even when enabled)
    INTERACTION_TERMS_MAX_ADJUSTMENT: Decimal = Decimal("3.0")  # Max ±3 points


# =============================================================================
# LOGGING REQUIREMENTS
# =============================================================================

class LoggingConfig:
    """
    What gets logged for audit and monitoring.
    """

    # Always log these fields in production output
    MANDATORY_OUTPUT_FIELDS: Set[str] = {
        "ticker",
        "composite_score",
        "composite_rank",
        "severity",
        "determinism_hash",
        "schema_version",
        "effective_weights",
    }

    # Log these when MONITORED features are active
    MONITORED_FEATURE_FIELDS: Dict[str, Set[str]] = {
        "VOLATILITY_ADJUSTED_SCORING": {
            "volatility_adjustment.vol_bucket",
            "volatility_adjustment.weight_factor",
            "volatility_adjustment.score_factor",
        },
        "SHRINKAGE_NORMALIZATION": {
            "normalization_method",
            "cohort_key",
        },
        "SMART_MONEY_TIER_WEIGHTING": {
            "smart_money_signal.weighted_overlap",
            "smart_money_signal.tier_breakdown",
            "smart_money_signal.tier1_holders",
        },
        "SMART_MONEY_POSITION_CHANGES": {
            "smart_money_signal.change_adjustment",
            "smart_money_signal.holders_increasing",
            "smart_money_signal.holders_decreasing",
        },
    }

    # Special logging for experimental features (verbose)
    EXPERIMENTAL_FEATURE_FIELDS: Dict[str, Set[str]] = {
        "INTERACTION_TERMS": {
            "interaction_terms.total_adjustment",
            "interaction_terms.flags",
            "interaction_terms.clinical_financial_synergy",
            "interaction_terms.stage_financial_interaction",
            "interaction_terms.catalyst_volatility_dampening",
        },
        "ADAPTIVE_WEIGHT_LEARNING": {
            "enhancement_diagnostics.adaptive_weights.method",
            "enhancement_diagnostics.adaptive_weights.confidence",
            "enhancement_diagnostics.adaptive_weights.historical_ic",
        },
    }

    # Alert thresholds for monitoring
    ALERT_THRESHOLDS: Dict[str, Decimal] = {
        # If >N% of tickers hit this cap, raise alert
        "monotonic_caps_pct_threshold": Decimal("0.30"),

        # If interaction adjustment exceeds this, raise alert
        "interaction_terms_max_alert": Decimal("2.5"),

        # If single ticker jumps more than N ranks vs v2, log warning
        "rank_divergence_threshold": 10,
    }


# =============================================================================
# FALLBACK THRESHOLDS
# =============================================================================

class FallbackConfig:
    """
    Thresholds that trigger fallback to safer configurations.
    """

    # =========================================================================
    # DATA QUALITY FALLBACKS
    # =========================================================================

    # If market data coverage < N%, disable volatility adjustment
    MIN_MARKET_DATA_COVERAGE_PCT: Decimal = Decimal("0.50")

    # If momentum data completeness < N, reduce momentum weight
    MIN_MOMENTUM_DATA_COMPLETENESS: Decimal = Decimal("0.50")

    # If peer count < N, fall back to global valuation stats
    MIN_PEER_COUNT_FOR_VALUATION: int = 5

    # If cohort size < N, use shrinkage normalization
    MIN_COHORT_SIZE_FOR_RANK_NORM: int = 10

    # =========================================================================
    # ADAPTIVE WEIGHTS FALLBACKS
    # =========================================================================

    # If adaptive weight confidence < N, fall back to base weights
    MIN_ADAPTIVE_WEIGHT_CONFIDENCE: Decimal = Decimal("0.40")

    # If training periods < N, don't use adaptive weights
    MIN_TRAINING_PERIODS_FOR_ADAPTIVE: int = 12  # ~1 year monthly

    # If weight L1 change > N from previous, reject and use previous
    MAX_WEIGHT_L1_CHANGE: Decimal = Decimal("0.15")

    # =========================================================================
    # PIT GATE FALLBACKS
    # =========================================================================

    # Minimum embargo months for historical returns
    MIN_EMBARGO_MONTHS: int = 1

    # If PIT gate fails, disable these features entirely
    FEATURES_DISABLED_ON_PIT_FAILURE: List[str] = [
        "ADAPTIVE_WEIGHT_LEARNING",
        "SMART_MONEY_POSITION_CHANGES",  # Relies on historical 13F comparison
    ]


# =============================================================================
# SANITY OVERRIDE MECHANISM
# =============================================================================

@dataclass
class SanityOverrideResult:
    """Result of sanity override check."""
    ticker: str
    v3_rank: int
    v2_rank: int
    rank_divergence: int
    override_applied: bool
    override_reason: Optional[str]
    driving_factor: Optional[str]
    confidence_level: Decimal
    fallback_to_v2: bool


class SanityOverrideConfig:
    """
    Sanity override mechanism to catch pathological v3 rankings.

    The 86% → 77% sanity score drop from v2 to v3 is acceptable IF we
    prevent pathological cases. This mechanism:

    1. Flags tickers with massive v2/v3 rank divergence
    2. Requires an explanation tag (momentum/valuation/smart-money/interaction)
    3. Falls back to v2 for that ticker if driven by LOW-CONFIDENCE signals
    """

    # =========================================================================
    # DIVERGENCE DETECTION
    # =========================================================================

    # If v3 rank - v2 rank exceeds this, trigger sanity check
    RANK_DIVERGENCE_THRESHOLD: int = 25  # Jumps >25 ranks require explanation

    # If ticker jumps INTO top-5 from outside top-30 in v2, always check
    TOP_BUCKET_JUMP_THRESHOLD: int = 30
    TOP_BUCKET_SIZE: int = 5

    # =========================================================================
    # DRIVING FACTOR ANALYSIS
    # =========================================================================

    # Minimum contribution % to be considered "driving" the rank change
    MIN_DRIVING_FACTOR_CONTRIBUTION: Decimal = Decimal("0.30")  # 30%

    # Valid driving factors (in order of trust)
    TRUSTED_DRIVING_FACTORS: List[str] = [
        "clinical",      # Strong clinical signal
        "financial",     # Clear financial improvement
        "catalyst",      # Near-term catalyst
    ]

    MONITORED_DRIVING_FACTORS: List[str] = [
        "momentum",      # Price momentum - can be noise
        "valuation",     # Peer valuation - can be stale
        "smart_money",   # 13F overlap - 45-day lag
    ]

    EXPERIMENTAL_DRIVING_FACTORS: List[str] = [
        "interaction",   # Interaction terms - high variance
        "adaptive",      # Adaptive weights - can overfit
    ]

    # =========================================================================
    # FALLBACK RULES
    # =========================================================================

    # If driving factor is EXPERIMENTAL and confidence < N, fall back
    EXPERIMENTAL_CONFIDENCE_THRESHOLD: Decimal = Decimal("0.60")

    # If driving factor is MONITORED and confidence < N, fall back
    MONITORED_CONFIDENCE_THRESHOLD: Decimal = Decimal("0.50")

    # Never fall back if driving factor is TRUSTED (regardless of confidence)
    TRUSTED_NEVER_FALLBACK: bool = True


def check_sanity_override(
    ticker: str,
    v3_rank: int,
    v2_rank: int,
    score_breakdown: Dict,
    config: SanityOverrideConfig = SanityOverrideConfig(),
) -> SanityOverrideResult:
    """
    Check if a ticker's v3 ranking requires sanity override.

    This is the "don't get burned" mechanism that prevents v3 from
    producing pathological rankings based on low-confidence or
    experimental signals.

    Args:
        ticker: The ticker symbol
        v3_rank: Rank in v3 scoring
        v2_rank: Rank in v2 scoring (for comparison)
        score_breakdown: Full score breakdown from v3
        config: Override configuration

    Returns:
        SanityOverrideResult with decision and explanation
    """
    rank_divergence = abs(v3_rank - v2_rank)

    # Check if sanity check is needed
    needs_check = False
    check_reason = None

    if rank_divergence > config.RANK_DIVERGENCE_THRESHOLD:
        needs_check = True
        check_reason = f"rank_divergence_{rank_divergence}"

    if v3_rank <= config.TOP_BUCKET_SIZE and v2_rank > config.TOP_BUCKET_JUMP_THRESHOLD:
        needs_check = True
        check_reason = f"top_bucket_jump_from_{v2_rank}"

    if not needs_check:
        return SanityOverrideResult(
            ticker=ticker,
            v3_rank=v3_rank,
            v2_rank=v2_rank,
            rank_divergence=rank_divergence,
            override_applied=False,
            override_reason=None,
            driving_factor=None,
            confidence_level=Decimal("1.0"),
            fallback_to_v2=False,
        )

    # Identify driving factor from score breakdown
    driving_factor = None
    max_contribution = Decimal("0")

    # Check component contributions
    components = score_breakdown.get("components", [])
    for comp in components:
        contrib = Decimal(str(comp.get("contribution", "0")))
        weight = Decimal(str(comp.get("weight_effective", "0")))

        # Normalize contribution by weight to get relative impact
        if weight > Decimal("0.01"):
            norm_contrib = contrib / weight
            if norm_contrib > max_contribution:
                max_contribution = norm_contrib
                driving_factor = comp.get("name")

    # Check enhancement signals
    enhancements = score_breakdown.get("enhancements", {})

    # Check interaction terms
    interaction_adj = Decimal(str(
        score_breakdown.get("interaction_terms", {}).get("total_adjustment", "0")
    ))
    if abs(interaction_adj) > Decimal("1.5"):
        driving_factor = "interaction"

    # Get confidence from overall or from driving component
    confidence_level = Decimal(str(score_breakdown.get("confidence_overall", "0.5")))

    # Determine if fallback is needed
    fallback_to_v2 = False
    override_reason = None

    if driving_factor in config.EXPERIMENTAL_DRIVING_FACTORS:
        if confidence_level < config.EXPERIMENTAL_CONFIDENCE_THRESHOLD:
            fallback_to_v2 = True
            override_reason = f"experimental_driver_{driving_factor}_low_confidence_{confidence_level}"

    elif driving_factor in config.MONITORED_DRIVING_FACTORS:
        if confidence_level < config.MONITORED_CONFIDENCE_THRESHOLD:
            fallback_to_v2 = True
            override_reason = f"monitored_driver_{driving_factor}_low_confidence_{confidence_level}"

    elif driving_factor in config.TRUSTED_DRIVING_FACTORS:
        if not config.TRUSTED_NEVER_FALLBACK:
            # Even trusted factors can be overridden if confidence is abysmal
            if confidence_level < Decimal("0.30"):
                fallback_to_v2 = True
                override_reason = f"trusted_driver_{driving_factor}_very_low_confidence_{confidence_level}"

    return SanityOverrideResult(
        ticker=ticker,
        v3_rank=v3_rank,
        v2_rank=v2_rank,
        rank_divergence=rank_divergence,
        override_applied=True,
        override_reason=override_reason or f"sanity_check_{check_reason}",
        driving_factor=driving_factor,
        confidence_level=confidence_level,
        fallback_to_v2=fallback_to_v2,
    )


# =============================================================================
# PRODUCTION DEFAULTS SUMMARY
# =============================================================================

V3_PRODUCTION_DEFAULTS = {
    # =========================================================================
    # MODULE 5 V3 CONFIGURATION
    # =========================================================================

    # Scoring mode: Always start with ENHANCED if PoS data available
    "default_scoring_mode": "enhanced",

    # Feature flags
    "feature_flags": {
        # TIER 1: ON by default
        "monotonic_caps": True,
        "confidence_weighting": True,
        "hybrid_aggregation": True,
        "determinism_hash": True,
        "catalyst_decay": True,
        "momentum": True,
        "valuation": True,
        "smart_money": True,

        # TIER 2: ON with monitoring
        "volatility_adjustment": True,
        "shrinkage_normalization": True,
        "smart_money_tiers": True,
        "smart_money_changes": True,

        # TIER 3: OFF until explicit opt-in
        "interaction_terms": False,
        "adaptive_weights": False,
        "regime_adaptation": False,
    },

    # =========================================================================
    # WEIGHTS
    # =========================================================================

    "v3_enhanced_weights": {
        "clinical": "0.28",
        "financial": "0.25",
        "catalyst": "0.17",
        "pos": "0.15",
        "momentum": "0.10",
        "valuation": "0.05",
    },

    "v3_partial_weights": {
        "clinical": "0.35",
        "financial": "0.30",
        "catalyst": "0.20",
        "momentum": "0.10",
        "valuation": "0.05",
    },

    "v3_default_weights": {
        "clinical": "0.40",
        "financial": "0.35",
        "catalyst": "0.25",
    },

    # =========================================================================
    # FALLBACKS
    # =========================================================================

    "min_market_data_coverage": "0.50",
    "min_momentum_completeness": "0.50",
    "min_peer_count": 5,
    "min_cohort_size": 10,
    "min_adaptive_confidence": "0.40",
    "max_weight_l1_change": "0.15",

    # =========================================================================
    # SANITY OVERRIDES
    # =========================================================================

    "sanity_rank_divergence_threshold": 25,
    "sanity_top_bucket_jump_threshold": 30,
    "sanity_experimental_confidence_threshold": "0.60",
    "sanity_monitored_confidence_threshold": "0.50",

    # =========================================================================
    # PIT GATES
    # =========================================================================

    "enforce_pit_gates": True,
    "embargo_months": 1,
    "shrinkage_lambda": "0.70",
    "smooth_gamma": "0.80",
}


# =============================================================================
# REGRESSION TESTS TO ADD
# =============================================================================

REQUIRED_REGRESSION_TESTS = [
    {
        "name": "test_ic_sign_spread_sign_agreement",
        "description": "IC sign and spread sign must agree - the earlier v3 bug should never regress",
        "assertion": "sign(IC) == sign(Top-Bottom Spread) for all periods",
    },
    {
        "name": "test_single_name_robustness",
        "description": "Removing any single name from top bucket should not flip conclusion",
        "assertion": "For each ticker in top-5, removing it still results in positive spread",
    },
    {
        "name": "test_decimal_zero_coalesce",
        "description": "Decimal('0') must be treated as valid score, not missing data",
        "assertion": "_coalesce(Decimal('0'), Decimal('50')) == Decimal('0')",
    },
    {
        "name": "test_runway_gate_independence",
        "description": "Runway gate computed from runway_months, not liquidity_status",
        "assertion": "runway_gate fires on runway_months < 6, not liquidity_status == 'FAIL'",
    },
    {
        "name": "test_determinism_hash_stability",
        "description": "Zero scores must not cause hash instability",
        "assertion": "hash(score=0) == hash(score=0) across runs",
    },
]


# Export all for clean imports
__all__ = [
    "FeatureFlags",
    "LoggingConfig",
    "FallbackConfig",
    "SanityOverrideConfig",
    "SanityOverrideResult",
    "check_sanity_override",
    "V3_PRODUCTION_DEFAULTS",
    "REQUIRED_REGRESSION_TESTS",
]
