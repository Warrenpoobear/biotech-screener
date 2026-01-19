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
5. Intelligent governance layer configuration (NEW)

The Intelligent Governance Layer provides:
- Sharpe-optimized weight learning
- Non-linear interaction effects with business logic
- Ensemble ranking (multiple perspectives)
- Regime-adaptive weight orchestration
- Smartness control knob for governance/intelligence tradeoff

Author: Wake Robin Capital Management
Version: 1.1.0
Created: 2026-01-18
Updated: 2026-01-18 - Added intelligent governance integration
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set
from enum import Enum


# =============================================================================
# OPTIMIZED COMPONENT WEIGHTS (Scipy Differential Evolution - 2026-01-19)
# =============================================================================

COMPONENT_WEIGHTS_V3 = {
    'clinical': 0.223,     # was 0.280 (-5.7pp)
    'financial': 0.257,    # was 0.250 (+0.7pp)
    'catalyst': 0.156,     # was 0.170 (-1.4pp)
    'pos': 0.232,          # was 0.150 (+8.2pp) ← KEY CHANGE
    'momentum': 0.102,     # was 0.100 (+0.2pp)
    'valuation': 0.030     # was 0.050 (-2.0pp)
}

# Optimization metadata for audit trail
WEIGHT_OPTIMIZATION_METADATA = {
    'optimization_date': '2026-01-19',
    'method': 'scipy_differential_evolution',
    'training_period': '2022-01-01 to 2024-12-31',
    'baseline_sharpe': 3.34,
    'optimized_sharpe': 4.26,
    'improvement_pct': 27.6,
    'oos_validated': True,
    'deployment_approved_by': 'Darren Schulz',
    'next_review_date': '2026-04-19',
    'key_finding': 'PoS should be weighted 23% (was 15%)'
}


# =============================================================================
# POS CONFIDENCE GATING (Critical for safe PoS weight increase)
# =============================================================================
# PoS coverage is ~75%. Without confidence gating, increasing PoS weight
# could penalize names with missing/weak PoS mappings or let low-quality
# fallbacks move ranks.

POS_CONFIDENCE_CONFIG = {
    # Minimum confidence to apply full PoS weight
    'min_confidence_full_weight': 0.60,

    # Below this threshold, set effective PoS weight to 0 and renormalize
    'min_confidence_threshold': 0.40,

    # Effective weight formula: w_pos_eff = w_pos * pos_conf (when conf >= threshold)
    'scale_by_confidence': True,

    # Log coverage per run to detect drift
    'log_coverage_metrics': True,

    # Alert if coverage drops below this
    'min_coverage_alert_threshold': 0.70,
}


# =============================================================================
# V3 AS DEFAULT, V2 AS SHADOW/FALLBACK
# =============================================================================
# Backtest evidence strongly favors v3:
#   - Mean IC: 0.088 vs 0.047 (almost 2x)
#   - Turnover: 5.9% vs 41.2% (massive reduction)
#   - Max DD: 28.2% vs 39.0%
#   - Cum return: +87.1% vs -27.4%
#   - IC/spread consistency: 94.1% vs 76.5%

SCORING_VERSION_CONFIG = {
    # Primary scorer (production rankings)
    'default_version': 'v3',

    # Shadow scorer (logged for diff monitoring, not used for ranking)
    'shadow_version': 'v2',

    # Enable shadow scoring (runs v2 in parallel, logs diffs)
    'enable_shadow_scoring': True,

    # Fallback to shadow version if primary fails validation
    'fallback_on_primary_failure': True,

    # Log top-N rank differences between v3 and v2
    'diff_report_top_n': 10,

    # Alert if >N tickers have rank divergence > threshold
    'rank_divergence_alert_count': 5,
    'rank_divergence_threshold': 15,
}


# =============================================================================
# DIFF MONITORING AND REASON CODES
# =============================================================================
# Automated report: "top-10 diffs + reason codes" to explain v3 rank changes

DIFF_MONITORING_CONFIG = {
    # Generate diff report each run
    'enabled': True,

    # Number of top rank differences to report
    'top_n_diffs': 10,

    # Include reason codes explaining the rank change
    'include_reason_codes': True,

    # Reason code categories (in order of priority)
    'reason_code_priority': [
        'clinical_signal_change',     # Clinical score moved significantly
        'financial_gate_change',      # Financial severity changed
        'catalyst_event',             # New catalyst or catalyst decay
        'pos_mapping_change',         # PoS indication mapping changed
        'momentum_signal',            # Price momentum moved ranks
        'valuation_peer_change',      # Peer valuation comparison changed
        'smart_money_signal',         # 13F overlap change
        'interaction_effect',         # Non-linear interaction triggered
    ],

    # Minimum contribution % to be flagged as reason
    'min_contribution_for_reason': 0.20,

    # Output format
    'report_format': 'json',  # 'json' or 'markdown'

    # Log to separate file for easy monitoring
    'log_to_separate_file': True,
    'diff_report_path': 'logs/v3_v2_diff_report.json',
}


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

    # =========================================================================
    # TIER 4: INTELLIGENT GOVERNANCE (NEW - Advanced features with explicit control)
    # =========================================================================

    # Master toggle for intelligent governance layer
    INTELLIGENT_GOVERNANCE_ENABLED: bool = True  # ON - master switch

    # Sharpe-ratio weight optimization (learns from historical returns)
    # Requires: historical_scores + forward_returns data
    SHARPE_WEIGHT_OPTIMIZATION: bool = False  # OFF - needs sufficient historical data

    # Non-linear interaction effects (business logic synergies/conflicts)
    # More sophisticated than INTERACTION_TERMS above - uses smooth ramps, not thresholds
    BUSINESS_LOGIC_INTERACTIONS: bool = True  # ON - proven stable

    # Ensemble ranking (multiple ranking perspectives: composite, momentum, value)
    ENSEMBLE_RANKING: bool = False            # OFF - experimental

    # Regime-adaptive weight orchestration (combines base + Sharpe + regime multipliers)
    REGIME_WEIGHT_ORCHESTRATION: bool = True  # ON - proven stable

    # Smartness knob (0.0 conservative to 1.0 aggressive)
    SMARTNESS_LEVEL: Decimal = Decimal("0.5")  # Balanced for production


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
# INTELLIGENT GOVERNANCE CONFIGURATION (NEW)
# =============================================================================

class IntelligentGovernanceConfig:
    """
    Configuration for the Intelligent Governance Layer.

    The governance layer sits on top of the deterministic feature/data layers,
    enabling adaptive optimization while maintaining full auditability and
    IC-defensibility.

    Architecture:
        INTELLIGENCE LAYER (Governed by this config)
        - Sharpe-optimized weight learning
        - Non-linear interaction effects
        - Ensemble ranking
        - Regime-adaptive weights

        FEATURE LAYER (Existing modules - unchanged)
        - Institutional signals (13F)
        - Financial health (SEC)
        - Clinical catalysts (CT.gov)
        - Momentum (prices)

        DATA LAYER (Deterministic - unchanged)
        - Raw filings, trials, prices
        - SHA256 integrity checks
        - Point-in-time discipline
    """

    # =========================================================================
    # MASTER CONTROL: SMARTNESS KNOB
    # =========================================================================
    # Control knob from 0 (conservative/governed) to 1 (aggressive/smart)
    # - 0.0: Max shrinkage, min interactions, strict missing data handling
    # - 0.5: Balanced defaults (RECOMMENDED FOR PRODUCTION)
    # - 1.0: Min shrinkage, max interactions, looser caps

    SMARTNESS_DEFAULT: Decimal = Decimal("0.5")  # Balanced for production
    SMARTNESS_MIN: Decimal = Decimal("0.0")
    SMARTNESS_MAX: Decimal = Decimal("1.0")

    # =========================================================================
    # FEATURE TOGGLES
    # =========================================================================

    # Sharpe-ratio weight optimization (learns from historical returns)
    ENABLE_SHARPE_OPTIMIZATION: bool = False  # OFF by default - needs historical data

    # Non-linear interaction effects (business logic bonuses/penalties)
    ENABLE_INTERACTION_EFFECTS: bool = True  # ON by default

    # Ensemble ranking (multiple ranking perspectives)
    ENABLE_ENSEMBLE_RANKING: bool = False  # OFF by default - experimental

    # Regime-adaptive weights (Bull/Bear/Volatility adjustments)
    ENABLE_REGIME_ADAPTATION: bool = True  # ON by default

    # =========================================================================
    # SHARPE OPTIMIZATION PARAMETERS
    # =========================================================================

    # Minimum periods required for Sharpe optimization
    SHARPE_MIN_PERIODS: int = 12  # ~1 year of monthly data

    # Shrinkage toward base weights (higher = more conservative)
    SHARPE_SHRINKAGE_LAMBDA: Decimal = Decimal("0.70")

    # Smoothing toward previous weights (higher = more stable)
    SHARPE_SMOOTHING_GAMMA: Decimal = Decimal("0.80")

    # PIT embargo for forward returns (months)
    SHARPE_EMBARGO_MONTHS: int = 1

    # Weight bounds
    SHARPE_MIN_WEIGHT: Decimal = Decimal("0.02")  # No component < 2%
    SHARPE_MAX_WEIGHT: Decimal = Decimal("0.60")  # No component > 60%

    # =========================================================================
    # INTERACTION EFFECTS PARAMETERS
    # =========================================================================

    # Maximum total adjustment from interaction effects
    INTERACTION_MAX_ADJUSTMENT: Decimal = Decimal("3.0")  # ±3 points

    # Individual effect caps
    INTERACTION_SYNERGY_CAP: Decimal = Decimal("2.0")
    INTERACTION_CONFLICT_CAP: Decimal = Decimal("2.0")

    # =========================================================================
    # ENSEMBLE RANKING PARAMETERS
    # =========================================================================

    # Weights for ensemble ranking perspectives
    ENSEMBLE_COMPOSITE_WEIGHT: Decimal = Decimal("0.50")  # Standard weighted
    ENSEMBLE_MOMENTUM_WEIGHT: Decimal = Decimal("0.25")   # Trend-following
    ENSEMBLE_VALUE_WEIGHT: Decimal = Decimal("0.25")      # Value/catalyst

    # =========================================================================
    # REGIME ADAPTATION PARAMETERS
    # =========================================================================

    # Maximum weight change per regime
    REGIME_MAX_WEIGHT_DELTA: Decimal = Decimal("0.15")

    # Regime multipliers (applied after base weights)
    REGIME_MULTIPLIERS: Dict[str, Dict[str, Decimal]] = {
        "BULL": {
            "clinical": Decimal("1.0"),
            "financial": Decimal("0.85"),
            "catalyst": Decimal("1.20"),
            "momentum": Decimal("1.25"),
            "pos": Decimal("1.0"),
            "valuation": Decimal("0.80"),
        },
        "BEAR": {
            "clinical": Decimal("1.0"),
            "financial": Decimal("1.30"),
            "catalyst": Decimal("0.80"),
            "momentum": Decimal("0.60"),
            "pos": Decimal("1.10"),
            "valuation": Decimal("1.15"),
        },
        "VOLATILITY_SPIKE": {
            "clinical": Decimal("0.90"),
            "financial": Decimal("1.40"),
            "catalyst": Decimal("0.70"),
            "momentum": Decimal("0.50"),
            "pos": Decimal("1.0"),
            "valuation": Decimal("1.0"),
        },
        "NEUTRAL": {
            "clinical": Decimal("1.0"),
            "financial": Decimal("1.0"),
            "catalyst": Decimal("1.0"),
            "momentum": Decimal("1.0"),
            "pos": Decimal("1.0"),
            "valuation": Decimal("1.0"),
        },
    }


class IntelligentGovernanceLogging:
    """
    Logging requirements for intelligent governance features.
    """

    # Fields to log when governance features are active
    GOVERNANCE_OUTPUT_FIELDS: Set[str] = {
        "governance_flags",
        "effective_weights",
        "smartness_level",
        "governance_audit_hash",
    }

    # Sharpe optimization logging
    SHARPE_LOGGING_FIELDS: Set[str] = {
        "sharpe_optimization.method",
        "sharpe_optimization.historical_sharpe",
        "sharpe_optimization.training_periods",
        "sharpe_optimization.confidence",
        "sharpe_optimization.l1_change_from_base",
        "sharpe_optimization.shrinkage_applied",
        "sharpe_optimization.weights_clamped",
    }

    # Interaction effects logging
    INTERACTION_LOGGING_FIELDS: Set[str] = {
        "interaction_effects.total_adjustment",
        "interaction_effects.net_synergy",
        "interaction_effects.net_conflict",
        "interaction_effects.triggered_effects",
        "interaction_effects.flags",
    }

    # Ensemble ranking logging
    ENSEMBLE_LOGGING_FIELDS: Set[str] = {
        "ensemble_rank.composite_rank",
        "ensemble_rank.momentum_rank",
        "ensemble_rank.value_rank",
        "ensemble_rank.final_rank",
        "ensemble_rank.rank_agreement",
        "ensemble_rank.max_divergence",
    }

    # Regime adaptation logging
    REGIME_LOGGING_FIELDS: Set[str] = {
        "regime_adaptation.regime",
        "regime_adaptation.multipliers_applied",
        "regime_adaptation.weight_delta",
    }

    # Alert thresholds
    GOVERNANCE_ALERT_THRESHOLDS: Dict[str, Decimal] = {
        # Alert if interaction adjustment exceeds this
        "interaction_max_alert": Decimal("2.5"),

        # Alert if Sharpe optimization confidence below this
        "sharpe_min_confidence_alert": Decimal("0.30"),

        # Alert if ensemble rank divergence exceeds this
        "ensemble_max_divergence_alert": 15,

        # Alert if regime weight delta exceeds this
        "regime_delta_alert": Decimal("0.12"),
    }


class IntelligentGovernanceFallbacks:
    """
    Fallback configurations for intelligent governance features.
    """

    # =========================================================================
    # SHARPE OPTIMIZATION FALLBACKS
    # =========================================================================

    # If training periods < N, disable Sharpe optimization
    SHARPE_MIN_TRAINING_PERIODS: int = 6

    # If Sharpe confidence < N, fall back to base weights
    SHARPE_MIN_CONFIDENCE: Decimal = Decimal("0.40")

    # If historical Sharpe < N, don't trust optimization
    SHARPE_MIN_RATIO: Decimal = Decimal("0.10")

    # If weight L1 change > N, reject optimization
    SHARPE_MAX_L1_CHANGE: Decimal = Decimal("0.25")

    # =========================================================================
    # INTERACTION EFFECTS FALLBACKS
    # =========================================================================

    # If missing metadata fields > N%, disable interactions
    INTERACTION_MIN_METADATA_COVERAGE: Decimal = Decimal("0.50")

    # If interaction confidence < N, reduce effect magnitude
    INTERACTION_MIN_CONFIDENCE: Decimal = Decimal("0.40")

    # =========================================================================
    # ENSEMBLE FALLBACKS
    # =========================================================================

    # If method ranks diverge > N, log warning
    ENSEMBLE_MAX_DIVERGENCE_WARNING: int = 20

    # If agreement < N, flag for review
    ENSEMBLE_MIN_AGREEMENT: Decimal = Decimal("0.30")

    # =========================================================================
    # REGIME FALLBACKS
    # =========================================================================

    # If regime signal unclear, fall back to NEUTRAL
    REGIME_DEFAULT_ON_UNCLEAR: str = "NEUTRAL"

    # Features disabled on regime detection failure
    FEATURES_DISABLED_ON_REGIME_FAILURE: List[str] = [
        "REGIME_ADAPTATION",
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

    # Optimized weights from scipy differential evolution (2026-01-19)
    # Sharpe improvement: 3.34 -> 4.26 (+27.6%)
    "v3_enhanced_weights": {
        "clinical": "0.223",    # was 0.28 (-5.7pp)
        "financial": "0.257",   # was 0.25 (+0.7pp)
        "catalyst": "0.156",    # was 0.17 (-1.4pp)
        "pos": "0.232",         # was 0.15 (+8.2pp) KEY CHANGE
        "momentum": "0.102",    # was 0.10 (+0.2pp)
        "valuation": "0.030",   # was 0.05 (-2.0pp)
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

    # =========================================================================
    # INTELLIGENT GOVERNANCE (NEW)
    # =========================================================================

    "intelligent_governance": {
        # Master control knob: 0.0 (conservative) to 1.0 (aggressive)
        # Production default: 0.5 (balanced)
        "smartness": "0.5",

        # Feature toggles
        "enable_sharpe_optimization": False,  # OFF - needs historical data
        "enable_interaction_effects": True,   # ON - proven stable
        "enable_ensemble_ranking": False,     # OFF - experimental
        "enable_regime_adaptation": True,     # ON - proven stable

        # Sharpe optimization (when enabled)
        "sharpe": {
            "min_periods": 12,
            "shrinkage_lambda": "0.70",
            "smoothing_gamma": "0.80",
            "embargo_months": 1,
            "min_weight": "0.02",
            "max_weight": "0.60",
            "min_confidence": "0.40",
        },

        # Interaction effects
        "interactions": {
            "max_adjustment": "3.0",
            "synergy_cap": "2.0",
            "conflict_cap": "2.0",
            "min_confidence": "0.40",
        },

        # Ensemble ranking (when enabled)
        "ensemble": {
            "composite_weight": "0.50",
            "momentum_weight": "0.25",
            "value_weight": "0.25",
            "min_agreement": "0.30",
        },

        # Regime adaptation
        "regime": {
            "max_weight_delta": "0.15",
            "default_on_unclear": "NEUTRAL",
        },
    },
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
    # Intelligent Governance regression tests (NEW)
    {
        "name": "test_governance_missing_data_renormalization",
        "description": "Missing scores should be excluded and weights renormalized, not imputed with 50",
        "assertion": "Missing pos/valuation scores result in governance_flags containing 'excluded_*_missing'",
    },
    {
        "name": "test_interaction_effects_bounded",
        "description": "Interaction effects total adjustment must be capped",
        "assertion": "abs(interaction_result.total_adjustment) <= INTERACTION_MAX_ADJUSTMENT",
    },
    {
        "name": "test_smartness_knob_clamped",
        "description": "Smartness level must be clamped to [0, 1]",
        "assertion": "smartness == clamp(input_smartness, 0, 1)",
    },
    {
        "name": "test_sharpe_pit_safety",
        "description": "Sharpe optimization must respect embargo period",
        "assertion": "No forward returns used from dates < as_of_date - embargo_months",
    },
    {
        "name": "test_governance_audit_hash_determinism",
        "description": "Governance audit hash must be deterministic",
        "assertion": "audit_hash(run1) == audit_hash(run2) for same inputs",
    },
]


# Export all for clean imports
__all__ = [
    # Optimized weights (2026-01-19)
    "COMPONENT_WEIGHTS_V3",
    "WEIGHT_OPTIMIZATION_METADATA",

    # PoS confidence gating
    "POS_CONFIDENCE_CONFIG",

    # V3 default / V2 shadow configuration
    "SCORING_VERSION_CONFIG",

    # Diff monitoring
    "DIFF_MONITORING_CONFIG",

    # Existing exports
    "FeatureFlags",
    "LoggingConfig",
    "FallbackConfig",
    "SanityOverrideConfig",
    "SanityOverrideResult",
    "check_sanity_override",
    "V3_PRODUCTION_DEFAULTS",
    "REQUIRED_REGRESSION_TESTS",

    # Intelligent Governance exports (NEW)
    "IntelligentGovernanceConfig",
    "IntelligentGovernanceLogging",
    "IntelligentGovernanceFallbacks",
    "create_intelligent_governance_layer",
]


# =============================================================================
# FACTORY FUNCTION FOR INTELLIGENT GOVERNANCE LAYER
# =============================================================================

def create_intelligent_governance_layer(
    config: Optional[Dict[str, Any]] = None,
    smartness: Optional[Decimal] = None,
):
    """
    Factory function to create an IntelligentGovernanceLayer with production config.

    This provides a convenient way to instantiate the governance layer with
    the production defaults while allowing overrides.

    Args:
        config: Optional config dict (defaults to V3_PRODUCTION_DEFAULTS["intelligent_governance"])
        smartness: Optional override for smartness level (0.0 to 1.0)

    Returns:
        Configured IntelligentGovernanceLayer instance

    Example:
        # Use production defaults
        layer = create_intelligent_governance_layer()

        # Override smartness for more conservative behavior
        layer = create_intelligent_governance_layer(smartness=Decimal("0.3"))

        # Full custom config
        layer = create_intelligent_governance_layer(config={
            "smartness": "0.7",
            "enable_sharpe_optimization": True,
            ...
        })
    """
    # Import here to avoid circular imports
    from src.modules.intelligent_governance import IntelligentGovernanceLayer

    # Get default config
    gov_config = config or V3_PRODUCTION_DEFAULTS.get("intelligent_governance", {})

    # Determine smartness level
    if smartness is not None:
        smart_level = smartness
    else:
        smart_level = Decimal(str(gov_config.get("smartness", "0.5")))

    # Create layer with configured settings
    return IntelligentGovernanceLayer(
        enable_sharpe_optimization=gov_config.get("enable_sharpe_optimization", False),
        enable_interaction_effects=gov_config.get("enable_interaction_effects", True),
        enable_regime_adaptation=gov_config.get("enable_regime_adaptation", True),
        smartness=smart_level,
    )
