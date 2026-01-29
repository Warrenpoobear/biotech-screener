"""
IC Enhancement Utilities for Module 5 Composite Scoring

This module provides advanced signal processing and scoring enhancements
designed to increase the Information Coefficient (IC) of the composite ranker.

Features:
- Adaptive weight learning (historical IC optimization)
- Non-linear signal interactions (cross-factor synergies/penalties)
- Peer-relative valuation signal
- Catalyst signal decay (time-based IC modeling)
- Price momentum signal (relative strength)
- Shrinkage normalization (Bayesian cohort adjustment)
- Smart money signal (13F position changes)
- Volatility-adjusted scoring
- Regime-adaptive component selection

Design Philosophy:
- DETERMINISTIC: No datetime.now(), no randomness
- STDLIB-ONLY: No external dependencies (except typing)
- DECIMAL-ONLY: Pure Decimal arithmetic for all scoring
- FAIL LOUDLY: Clear error states with explicit None handling
- PIT-SAFE: All lookback calculations respect point-in-time constraints

Author: Wake Robin Capital Management
Version: 1.0.0
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

__version__ = "1.2.0"  # V3 smart money signal with canonical manager registry


# =============================================================================
# CONSTANTS
# =============================================================================

SCORE_PRECISION = Decimal("0.01")
WEIGHT_PRECISION = Decimal("0.0001")
EPS = Decimal("0.000001")

# Volatility thresholds
VOLATILITY_BASELINE = Decimal("0.50")  # 50% annualized vol as baseline
VOLATILITY_LOW_THRESHOLD = Decimal("0.30")
VOLATILITY_HIGH_THRESHOLD = Decimal("0.80")
VOLATILITY_MAX_ADJUSTMENT = Decimal("0.25")

# Momentum calculation parameters
MOMENTUM_LOOKBACK_DAYS = 60
MOMENTUM_OPTIMAL_ALPHA_BPS = 1000  # 10% alpha = 65 score (with slope 150)
# Saturation parameters - reduced to improve rank resolution
# Old: slope=200, clamp 10-90 caused too many ties at extremes
# New: slope=150, clamp 5-95 for better rank granularity
MOMENTUM_SLOPE = Decimal("150")  # +10% alpha → +15 points (vs old +20)
MOMENTUM_SCORE_MIN = Decimal("5")
MOMENTUM_SCORE_MAX = Decimal("95")
# Epsilon for volatility normalization
MOMENTUM_VOL_EPS = Decimal("0.0001")

# Strong signal threshold (for diagnostic metrics)
# A signal is "strong" if |score - 50| >= this threshold
MOMENTUM_STRONG_THRESHOLD = Decimal("2.5")


def compute_alpha_for_score_delta(
    score_delta: Decimal,
    confidence: Decimal,
    slope: Decimal = MOMENTUM_SLOPE,
) -> Decimal:
    """Compute the alpha required to produce a given score delta after shrinkage.

    The momentum score formula is:
        final_score = 50 + confidence * (raw_score - 50)
        raw_score = 50 + alpha * slope

    Combining: final_score = 50 + confidence * alpha * slope
    So: |score_delta| = confidence * |alpha| * slope
    Therefore: |alpha| = |score_delta| / (confidence * slope)

    Args:
        score_delta: The score deviation from 50 (e.g., 2.5 for "strong" threshold)
        confidence: Shrinkage confidence (0.0-1.0)
        slope: Score mapping slope (default MOMENTUM_SLOPE=150)

    Returns:
        The absolute alpha (as Decimal) required to achieve that score delta.
        For score_delta=2.5, conf=0.7: returns ~0.0238 (2.38% alpha)
    """
    if confidence <= 0:
        return Decimal("Inf")
    return (abs(score_delta) / (confidence * slope)).quantize(Decimal("0.0001"))


# Catalyst decay parameters
# Uses exponential decay with asymmetric shape:
# - Slow rise as event approaches (from 90+ days out)
# - Peak at ~30 days before event
# - Moderate decay after event (information gets priced, but catalyst history matters)
CATALYST_OPTIMAL_WINDOW_DAYS = 30
# Optimal window half-width: events within ±CATALYST_OPTIMAL_HALF_WIDTH of peak are "in window"
# Widened from 15 to 30 days to improve coverage (0-60 day window vs 15-45)
CATALYST_OPTIMAL_HALF_WIDTH = 30
# Decay rates - RELAXED by ~40% to keep signals stronger for longer
# Original values were: PDUFA=0.05, DATA_READOUT=0.08, PHASE_COMPLETION=0.10, etc.
CATALYST_DECAY_RATES = {
    "PDUFA": Decimal("0.03"),       # Slowest decay - high anticipation events
    "DATA_READOUT": Decimal("0.05"),
    "PHASE_COMPLETION": Decimal("0.06"),
    "ENROLLMENT_COMPLETE": Decimal("0.04"),
    "CT_PRIMARY_COMPLETION": Decimal("0.04"),  # Clinical trial primary completion
    "CT_STUDY_COMPLETION": Decimal("0.05"),    # Clinical trial study completion
    "DEFAULT": Decimal("0.04"),
}
# Post-event decay multiplier: tau_effective = tau * POST_EVENT_DECAY_MULT
# Reduced from 2.0 to 1.5 - past catalysts retain more signal value
CATALYST_POST_EVENT_DECAY_MULT = Decimal("1.5")
# Minimum decay floor - even distant catalysts contribute meaningfully
# Raised from 0.05 to 0.25 to improve coverage
CATALYST_DECAY_FLOOR = Decimal("0.25")
# Decay precision: 0.0001 matches WEIGHT_PRECISION to avoid tie artifacts
CATALYST_DECAY_PRECISION = Decimal("0.0001")

# Shrinkage normalization parameters
SHRINKAGE_MIN_COHORT = Decimal("5")
SHRINKAGE_PRIOR_STRENGTH = Decimal("5")  # Equivalent sample size of prior

# Smart money signal parameters (legacy - kept for backwards compatibility)
SMART_MONEY_OVERLAP_BONUS_PER_HOLDER = Decimal("5")
SMART_MONEY_MAX_OVERLAP_BONUS = Decimal("20")
SMART_MONEY_POSITION_CHANGE_WEIGHTS = {
    "NEW": Decimal("3"),
    "INCREASE": Decimal("2"),
    "HOLD": Decimal("0"),
    "DECREASE": Decimal("-2"),
    "EXIT": Decimal("-5"),
}

# Smart money V2 parameters - tier-weighted with saturation
# =============================================================================
# V2 IMPROVEMENTS:
# 1. Weight by holder tier (Tier1=1.0, Tier2=0.6, Unknown=0.2)
# 2. Apply saturating function for overlap (diminishing returns)
# 3. Cap per-holder contribution to prevent one noisy filing dominating
# 4. Reduce EXIT weight unless high confidence (data quality issue)
# =============================================================================

# Tier weights: Tier1 pure biotech specialists, Tier2 diversified healthcare
SMART_MONEY_TIER_WEIGHTS = {
    1: Decimal("1.0"),   # Elite Core biotech specialists
    2: Decimal("0.6"),   # Elite Core diversified healthcare
    3: Decimal("0.4"),   # Elite Core smaller/narrower focus
}
SMART_MONEY_UNKNOWN_TIER_WEIGHT = Decimal("0.2")

# Conditional managers (multi-strategy, quant) - capped signal contribution
# These are tracked for breadth signal but shouldn't dominate due to AUM scale
SMART_MONEY_CONDITIONAL_WEIGHT = Decimal("0.3")  # Lower base weight
SMART_MONEY_CONDITIONAL_SIGNAL_CAP = Decimal("0.30")  # Max 30% of total signal

# =============================================================================
# SMART MONEY MANAGER REGISTRY - Loaded from canonical source
# =============================================================================
# CANONICAL SOURCE: production_data/manager_registry.json
# This replaces hardcoded manager lists with dynamically loaded data.
# Elite Core (Tier 1) = Primary biotech specialists
# Conditional (Tier 2) = Multi-strategy/quant platforms with healthcare exposure
# =============================================================================

def _load_manager_registry() -> Dict[str, Any]:
    """
    Load manager registry from canonical JSON source.

    Searches in multiple locations for flexibility across project structure.
    Returns empty dict if not found (graceful degradation).
    """
    search_paths = [
        Path(__file__).parent.parent.parent / "production_data" / "manager_registry.json",
        Path(__file__).parent.parent.parent.parent / "production_data" / "manager_registry.json",
        Path.cwd() / "production_data" / "manager_registry.json",
        Path.cwd().parent / "production_data" / "manager_registry.json",
    ]

    for path in search_paths:
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

    # Fallback: return empty structure (will use empty lookups)
    return {"elite_core": [], "conditional": []}


def _build_tier_by_name(registry: Dict[str, Any]) -> Dict[str, int]:
    """Build name -> tier mapping from registry."""
    result = {}

    # Elite Core managers get tier 1 (primary signal)
    for mgr in registry.get("elite_core", []):
        name = mgr.get("name", "").lower()
        if name:
            result[name] = 1
            # Add common variations
            words = name.split()
            if len(words) >= 2:
                # First two words (e.g., "baker bros" from "Baker Bros Advisors")
                result[" ".join(words[:2]).lower()] = 1
                # First word only for distinctive names
                if words[0].lower() not in {"the", "a", "an"}:
                    result[words[0].lower()] = 1

    # Conditional managers get tier 2 (breadth signal, capped)
    for mgr in registry.get("conditional", []):
        name = mgr.get("name", "").lower()
        if name:
            result[name] = 2
            words = name.split()
            if len(words) >= 2:
                result[" ".join(words[:2]).lower()] = 2
                if words[0].lower() not in {"the", "a", "an"}:
                    result[words[0].lower()] = 2

    return result


def _build_tier_by_cik(registry: Dict[str, Any]) -> Dict[str, int]:
    """Build CIK -> tier mapping from registry."""
    result = {}

    # Elite Core managers get tier 1
    for mgr in registry.get("elite_core", []):
        cik = mgr.get("cik", "")
        if cik:
            # Store both with and without leading zeros
            cik_clean = cik.lstrip("0")
            result[cik] = 1
            result[cik_clean] = 1

    # Conditional managers get tier 2
    for mgr in registry.get("conditional", []):
        cik = mgr.get("cik", "")
        if cik:
            cik_clean = cik.lstrip("0")
            result[cik] = 2
            result[cik_clean] = 2

    return result


def _build_conditional_ciks(registry: Dict[str, Any]) -> set:
    """Build set of Conditional manager CIKs."""
    result = set()
    for mgr in registry.get("conditional", []):
        cik = mgr.get("cik", "")
        if cik:
            result.add(cik)
            result.add(cik.lstrip("0"))
    return result


def _build_conditional_names(registry: Dict[str, Any]) -> set:
    """Build set of Conditional manager name patterns."""
    result = set()
    for mgr in registry.get("conditional", []):
        name = mgr.get("name", "").lower()
        if name:
            result.add(name)
            words = name.split()
            if len(words) >= 2:
                result.add(" ".join(words[:2]).lower())
            if words and words[0].lower() not in {"the", "a", "an"}:
                result.add(words[0].lower())
    return result


# Load registry at module initialization
_MANAGER_REGISTRY = _load_manager_registry()

# Build lookup dictionaries from canonical registry
# These replace the previously hardcoded dictionaries
SMART_MONEY_TIER_BY_NAME = _build_tier_by_name(_MANAGER_REGISTRY)
SMART_MONEY_TIER_BY_CIK = _build_tier_by_cik(_MANAGER_REGISTRY)
SMART_MONEY_CONDITIONAL_CIKS = _build_conditional_ciks(_MANAGER_REGISTRY)
SMART_MONEY_CONDITIONAL_MANAGERS = _build_conditional_names(_MANAGER_REGISTRY)

# Overlap bonus: saturating function applied to weighted overlap
# Piecewise: linear 0-1.5 weighted overlap, then diminishing above
SMART_MONEY_OVERLAP_SATURATION_THRESHOLD = Decimal("1.5")  # Weighted overlap
SMART_MONEY_OVERLAP_LINEAR_BONUS = Decimal("12")  # Max bonus in linear region (per 1.0 weight)
SMART_MONEY_OVERLAP_MAX_BONUS = Decimal("20")  # Hard cap after saturation

# Position change V2: reduced weights, per-holder cap
SMART_MONEY_V2_CHANGE_WEIGHTS = {
    "NEW": Decimal("3"),
    "INCREASE": Decimal("2"),
    "HOLD": Decimal("0"),
    "DECREASE": Decimal("-2"),
    "EXIT": Decimal("-3"),  # Reduced from -5 (data quality sensitivity)
}
SMART_MONEY_PER_HOLDER_CAP = Decimal("5")  # Max |contribution| per holder
SMART_MONEY_CHANGE_MAX_BONUS = Decimal("15")
SMART_MONEY_CHANGE_MIN_PENALTY = Decimal("-15")

# Interaction term parameters - using SMOOTH RAMPS to avoid rank churn
# All inputs must be Decimal with explicit ranges:
#   clinical_norm, catalyst_norm, financial_norm: [0, 100]
#   vol_norm: [0, 1] (fraction, not percentage)
#   runway_months: raw months (no normalization assumed)

# Synergy ramp: bonus ramps linearly from 0 at clinical=60 to max at clinical=80
# AND from 0 at runway=12 to max at runway=24
INTERACTION_SYNERGY_CLINICAL_LOW = Decimal("60")
INTERACTION_SYNERGY_CLINICAL_HIGH = Decimal("80")
INTERACTION_SYNERGY_RUNWAY_LOW = Decimal("12")  # months
INTERACTION_SYNERGY_RUNWAY_HIGH = Decimal("24")  # months
INTERACTION_SYNERGY_MAX_BONUS = Decimal("1.5")  # Max +1.5 points (was 3.0)

# Distress ramp: penalty ramps from 0 at runway=12 to max at runway=6
INTERACTION_DISTRESS_RUNWAY_HIGH = Decimal("12")  # months - no penalty above
INTERACTION_DISTRESS_RUNWAY_LOW = Decimal("6")    # months - max penalty below
INTERACTION_DISTRESS_MAX_PENALTY = Decimal("2.0")  # Max -2.0 points (was 5.0)

# Catalyst dampening in high-vol: multiplicative factor [0.7, 1.0]
# Only applies when vol_bucket == HIGH
INTERACTION_CATALYST_VOL_DAMPENING_MIN = Decimal("0.7")  # Floor multiplier

# Peer-relative valuation parameters
# Winsorization bounds to prevent extreme values from dominating
VALUATION_TRIAL_COUNT_MIN = 1      # Minimum trials (avoid divide-by-zero)
VALUATION_TRIAL_COUNT_MAX = 30     # Cap trials to reduce denominator gaming
VALUATION_MCAP_MIN_MM = Decimal("50")     # Floor at $50M
VALUATION_MCAP_MAX_MM = Decimal("50000")  # Cap at $50B
# Minimum peers for reliable signal
VALUATION_MIN_PEERS = 5
# Confidence ramp: conf = min(0.8, base + slope * N)
VALUATION_CONFIDENCE_BASE = Decimal("0.20")
VALUATION_CONFIDENCE_SLOPE = Decimal("0.04")  # +4% per peer
VALUATION_CONFIDENCE_MAX = Decimal("0.80")
# Shrinkage toward neutral (50) when sample is small
# shrunk_score = raw_score * (1 - shrink) + 50 * shrink
VALUATION_SHRINKAGE_FULL_AT_N = 20  # No shrinkage at 20+ peers
VALUATION_SHRINKAGE_MAX = Decimal("0.50")  # Max 50% shrinkage at N=5

# Commercial vs Development regime classification thresholds
# A company is COMMERCIAL if ANY of these conditions are met
VALUATION_COMMERCIAL_REVENUE_THRESHOLD_MM = Decimal("200")  # $200M revenue
VALUATION_COMMERCIAL_CFO_POSITIVE = True  # Positive operating cash flow

# Development-stage trial count gate
# If trial_count < this threshold, mcap/trial is unreliable -> neutral score
VALUATION_DEV_MIN_TRIALS = 3

# Commercial valuation parameters (EV/CFO or MCAP/Revenue)
VALUATION_COMMERCIAL_MIN_PEERS = 5
VALUATION_COMMERCIAL_EV_CFO_MIN = Decimal("2")    # Floor EV/CFO at 2x
VALUATION_COMMERCIAL_EV_CFO_MAX = Decimal("50")   # Cap EV/CFO at 50x
VALUATION_COMMERCIAL_EV_REV_MIN = Decimal("0.5")  # Floor EV/Revenue at 0.5x
VALUATION_COMMERCIAL_EV_REV_MAX = Decimal("20")   # Cap EV/Revenue at 20x


# =============================================================================
# ENUMS
# =============================================================================

class RegimeType(str, Enum):
    """Market regime classification."""
    BULL = "BULL"
    BEAR = "BEAR"
    NEUTRAL = "NEUTRAL"
    UNKNOWN = "UNKNOWN"


class VolatilityBucket(str, Enum):
    """Volatility classification."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    UNKNOWN = "unknown"


class PositionChangeType(str, Enum):
    """13F position change classification."""
    NEW = "NEW"
    INCREASE = "INCREASE"
    HOLD = "HOLD"
    DECREASE = "DECREASE"
    EXIT = "EXIT"


class ValuationRegime(str, Enum):
    """Valuation model regime classification.

    COMMERCIAL: Revenue-generating companies where cashflow-based metrics apply
    DEVELOPMENT: Pipeline companies where mcap/trial_count is meaningful
    UNKNOWN: Insufficient data to classify
    """
    COMMERCIAL = "commercial"
    DEVELOPMENT = "development"
    UNKNOWN = "unknown"


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class VolatilityAdjustment:
    """Result of volatility-based adjustment calculation."""
    annualized_vol: Optional[Decimal]
    vol_bucket: VolatilityBucket
    weight_adjustment_factor: Decimal  # Multiplier for weights (0.75 - 1.25)
    score_adjustment_factor: Decimal  # Multiplier for final score
    confidence_penalty: Decimal  # Additional confidence reduction


@dataclass
class MomentumSignal:
    """Price momentum signal result.

    Fields:
        momentum_score: 0-100 normalized score (5-95 effective range)
        alpha_60d: Raw excess return vs benchmark (ret - bench)
        alpha_vol_adjusted: Volatility-adjusted alpha (alpha / vol) for
            "persistent outperformance" - rewards low-vol outperformers
        return_60d: Stock's 60-day compounded return
        benchmark_return_60d: XBI's 60-day compounded return (same window)
        confidence: Signal confidence (0-1) based on alpha magnitude
        data_completeness: Data availability flag (0-1) indicating if
            all required inputs were present (1.0 = complete, 0.0 = missing)

    V2 Fields (multi-window fallback):
        window_used: Which window was selected (20, 60, 120, or None)
        data_status: Classification for diagnostics

    V3 Fields (guardrails):
        benchmark_missing: Set True when benchmark was missing for any window
        return_clipped: Set True when return was clipped due to outlier
        guardrail_flags: List of any guardrail events that affected the signal
    """
    momentum_score: Decimal  # 0-100 normalized score
    alpha_60d: Optional[Decimal]  # Excess return vs benchmark
    alpha_vol_adjusted: Optional[Decimal]  # Vol-adjusted alpha (alpha/vol)
    return_60d: Optional[Decimal]
    benchmark_return_60d: Optional[Decimal]
    confidence: Decimal
    data_completeness: Decimal  # 1.0 = full data, 0.0 = missing
    # V2 multi-window fields (optional for backwards compatibility)
    window_used: Optional[int] = None  # Which window was used (20, 60, 120)
    data_status: str = "unknown"  # "missing_prices" | "computed_low_conf" | "applied"
    # V3 guardrail fields
    benchmark_missing: bool = False  # True if benchmark was unavailable
    return_clipped: bool = False  # True if return was clipped for outlier
    guardrail_flags: List[str] = field(default_factory=list)  # All guardrail events


@dataclass
class MultiWindowMomentumInput:
    """Input for multi-window momentum calculation.

    Provides returns for multiple lookback windows (20d, 60d, 120d) with
    corresponding benchmark returns. The momentum calculation will use
    the longest available window, falling back to shorter ones.

    V3 Fields (guardrails):
        return_clipped_20d, etc.: Set True if corresponding return was clipped
    """
    # Stock returns by window (trading days)
    return_20d: Optional[Decimal] = None
    return_60d: Optional[Decimal] = None
    return_120d: Optional[Decimal] = None

    # Benchmark (XBI) returns by window
    benchmark_20d: Optional[Decimal] = None
    benchmark_60d: Optional[Decimal] = None
    benchmark_120d: Optional[Decimal] = None

    # Trading days available (for confidence calculation)
    trading_days_available: Optional[int] = None

    # Volatility (optional, for vol-adjusted alpha)
    annualized_vol: Optional[Decimal] = None

    # V3 guardrail tracking: which returns were clipped for outliers
    return_clipped_20d: bool = False
    return_clipped_60d: bool = False
    return_clipped_120d: bool = False


@dataclass
class ValuationSignal:
    """Peer-relative valuation signal result.

    Supports two valuation regimes:
    - DEVELOPMENT: Uses mcap/trial_count vs stage-matched peers
    - COMMERCIAL: Uses EV/CFO or MCAP/Revenue vs commercial peers
    """
    valuation_score: Decimal  # 0-100 (higher = cheaper)
    mcap_per_asset: Optional[Decimal]  # For dev-stage only
    peer_median_mcap_per_asset: Optional[Decimal]  # For dev-stage only
    peer_count: int
    confidence: Decimal
    # V2 regime routing fields
    regime: ValuationRegime = ValuationRegime.UNKNOWN
    method: str = "unknown"  # "mcap_per_trial", "ev_cfo", "ev_revenue", "sector_mcap", "neutral"
    ev_multiple: Optional[Decimal] = None  # EV/CFO or EV/Revenue for commercial
    peer_median_ev_multiple: Optional[Decimal] = None  # Commercial peer median
    flags: List[str] = field(default_factory=list)  # Diagnostic flags


@dataclass
class CatalystDecayResult:
    """Catalyst signal decay calculation result."""
    decay_factor: Decimal  # 0-1 multiplier
    days_to_catalyst: Optional[int]
    event_type: str
    in_optimal_window: bool


@dataclass
class SmartMoneySignal:
    """Smart money (13F) signal result.

    V2 ENHANCEMENTS:
    - weighted_overlap: Sum of tier weights (not raw count)
    - tier_breakdown: Dict of tier -> count for diagnostics
    - per_holder_contributions: Detailed breakdown by holder
    - Uses saturation to prevent gaming by many small holders

    V3 ENHANCEMENTS:
    - Elite Core vs Conditional separation
    - Conditional signal capped at 30% to prevent AUM dominance
    """
    smart_money_score: Decimal  # 20-80 range
    overlap_count: int  # Raw holder count (for backwards compat)
    overlap_bonus: Decimal  # Now tier-weighted
    position_change_adjustment: Decimal  # Now tier-weighted with per-holder caps
    holders_increasing: List[str]
    holders_decreasing: List[str]
    confidence: Decimal
    # V2 fields
    weighted_overlap: Decimal = Decimal("0")  # Sum of tier weights
    tier_breakdown: Dict[int, int] = field(default_factory=dict)  # tier -> count
    tier1_holders: List[str] = field(default_factory=list)  # For diagnostics
    per_holder_contributions: Dict[str, Decimal] = field(default_factory=dict)
    # V3 fields - Elite Core vs Conditional separation
    elite_core_contribution: Decimal = Decimal("0")  # Signal from Elite Core managers
    conditional_contribution: Decimal = Decimal("0")  # Signal from Conditional (may be capped)
    conditional_raw_contribution: Decimal = Decimal("0")  # Pre-cap Conditional signal
    conditional_capped: bool = False  # True if Conditional signal was capped


@dataclass
class InteractionTerms:
    """Non-linear interaction term calculations.

    All adjustments use smooth ramps to avoid rank churn from discontinuities.
    Magnitudes are bounded to ±3 points to prevent leaderboard rewrites.
    """
    clinical_financial_synergy: Decimal  # [0, +1.5] smooth ramp
    stage_financial_interaction: Decimal  # [-2.0, 0] smooth ramp
    catalyst_volatility_dampening: Decimal  # Dampening applied (informational)
    competitive_partnership_interaction: Decimal  # [-3.0, +3.0] CI+partnership modifier
    cash_burn_interaction: Decimal  # [-1.5, +1.5] burn trajectory modifier
    phase_momentum_interaction: Decimal  # [-1.5, +1.5] phase momentum modifier
    total_interaction_adjustment: Decimal  # Sum of above (bounded)
    interaction_flags: List[str]
    # Gate status tracking to prevent double-counting
    runway_gate_already_applied: bool = False
    dilution_gate_already_applied: bool = False


@dataclass
class AdaptiveWeights:
    """Result of adaptive weight optimization."""
    weights: Dict[str, Decimal]
    historical_ic_by_component: Dict[str, Decimal]
    optimization_method: str
    lookback_months: int
    confidence: Decimal
    # PIT-safety fields
    embargo_months: int = 1
    training_periods: int = 0  # Number of as_of_date periods used
    shrinkage_applied: Decimal = Decimal("0")
    smoothing_applied: Decimal = Decimal("0")


@dataclass
class RegimeSignalImportance:
    """Regime-specific signal importance multipliers."""
    clinical: Decimal
    financial: Decimal
    catalyst: Decimal
    momentum: Decimal
    valuation: Decimal
    smart_money: Decimal


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _to_decimal(value: Any, default: Optional[Decimal] = None) -> Optional[Decimal]:
    """Convert various types to Decimal with safe handling."""
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
    """Quantize to standard score precision."""
    return value.quantize(SCORE_PRECISION, rounding=ROUND_HALF_UP)


def _quantize_weight(value: Decimal) -> Decimal:
    """Quantize to standard weight precision."""
    return value.quantize(WEIGHT_PRECISION, rounding=ROUND_HALF_UP)


def _clamp(value: Decimal, min_val: Decimal, max_val: Decimal) -> Decimal:
    """Clamp value to range."""
    return max(min_val, min(max_val, value))


def _safe_divide(numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0")) -> Decimal:
    """Safe division with zero handling."""
    if denominator == 0 or abs(denominator) < EPS:
        return default
    return numerator / denominator


# =============================================================================
# VOLATILITY-ADJUSTED SCORING
# =============================================================================

def compute_volatility_adjustment(
    annualized_vol: Optional[Decimal],
    *,
    target_vol: Decimal = VOLATILITY_BASELINE,
    low_threshold: Decimal = VOLATILITY_LOW_THRESHOLD,
    high_threshold: Decimal = VOLATILITY_HIGH_THRESHOLD,
    max_adjustment: Decimal = VOLATILITY_MAX_ADJUSTMENT,
    max_score_penalty: Decimal = Decimal("0.30"),
) -> VolatilityAdjustment:
    """
    Compute volatility-based adjustment factors.

    SCORE PENALTY LOGIC (v2 - asymmetric, penalize only high vol):
    - At or below target vol (50%): No score penalty (score_adjustment_factor = 1.0)
    - Above target vol: Linear penalty = (vol_ratio - 1) * 0.15, capped at max_score_penalty
    - This avoids penalizing low-vol names which often deliver more reliable returns

    WEIGHT ADJUSTMENT LOGIC (unchanged):
    - Low volatility (<30%): Boost weight influence (more reliable signal)
    - Normal volatility (30-80%): Neutral weight adjustment
    - High volatility (>80%): Reduce weight influence (less reliable signal)

    RATIONALE:
    - Low-vol, institutionally owned biotech names often have better IC
    - Penalizing low-vol suppresses the best compounders
    - Only high-vol "lottery ticket" names should be dampened

    Test vectors (raw_score=80, target=0.50):
    - vol=0.50 → 80 (no penalty)
    - vol=0.25 → 80 (no penalty - low vol is not penalized)
    - vol=0.75 → ratio 1.5 → penalty 0.075 → 74
    - vol=1.00 → ratio 2.0 → penalty 0.15 → 68

    Args:
        annualized_vol: Annualized volatility as decimal (e.g., 0.50 for 50%)
        target_vol: Target volatility - no penalty at or below (default 0.50)
        low_threshold: Below this = low vol bucket for weight boost
        high_threshold: Above this = high vol bucket for max weight reduction
        max_adjustment: Maximum weight adjustment factor
        max_score_penalty: Maximum score penalty (default 0.30 = 30%)

    Returns:
        VolatilityAdjustment with all computed factors
    """
    if annualized_vol is None:
        return VolatilityAdjustment(
            annualized_vol=None,
            vol_bucket=VolatilityBucket.UNKNOWN,
            weight_adjustment_factor=Decimal("1.0"),
            score_adjustment_factor=Decimal("1.0"),
            confidence_penalty=Decimal("0.05"),  # Small penalty for unknown
        )

    vol = _to_decimal(annualized_vol, Decimal("0.50"))

    # Ensure vol is positive
    if vol <= Decimal("0"):
        return VolatilityAdjustment(
            annualized_vol=Decimal("0"),
            vol_bucket=VolatilityBucket.UNKNOWN,
            weight_adjustment_factor=Decimal("1.0"),
            score_adjustment_factor=Decimal("1.0"),
            confidence_penalty=Decimal("0.05"),
        )

    # =========================================================================
    # SCORE ADJUSTMENT: Penalize only high vol (v2 asymmetric)
    # =========================================================================
    # Key insight: low-vol names often deliver better forward returns in biotech
    # Penalizing them hurts IC. Only penalize above target vol.

    vol_ratio = vol / target_vol

    if vol_ratio <= Decimal("1"):
        # At or below target vol: NO score penalty
        score_adj = Decimal("1.0")
    else:
        # Above target vol: linear penalty, capped
        # penalty = (vol_ratio - 1) * 0.15, capped at max_score_penalty
        penalty = (vol_ratio - Decimal("1")) * Decimal("0.15")
        penalty = _clamp(penalty, Decimal("0"), max_score_penalty)
        score_adj = Decimal("1.0") - penalty

    # =========================================================================
    # WEIGHT ADJUSTMENT: Boost low vol, reduce high vol (for signal reliability)
    # =========================================================================
    # This is separate from score adjustment - it affects how much we trust
    # the signal, not the final ranking directly.

    if vol < low_threshold:
        # Low vol = more reliable signal = boost weights
        vol_bucket = VolatilityBucket.LOW
        # Linear interpolation: at 0% vol -> +max_adjustment, at threshold -> 0%
        weight_adj = Decimal("1.0") + max_adjustment * (
            (low_threshold - vol) / low_threshold
        )
        confidence_penalty = Decimal("0")

    elif vol <= high_threshold:
        # Normal vol = neutral weight adjustment
        vol_bucket = VolatilityBucket.NORMAL
        weight_adj = Decimal("1.0")
        # Confidence penalty increases with volatility
        vol_ratio_norm = (vol - low_threshold) / (high_threshold - low_threshold)
        confidence_penalty = vol_ratio_norm * Decimal("0.10")

    else:
        # High vol = less reliable signal = reduce weight influence
        vol_bucket = VolatilityBucket.HIGH
        # Reduce weight influence
        excess_vol = vol - high_threshold
        weight_adj = Decimal("1.0") - min(
            max_adjustment,
            max_adjustment * (excess_vol / target_vol)
        )
        confidence_penalty = Decimal("0.15")

    return VolatilityAdjustment(
        annualized_vol=_quantize_score(vol * Decimal("100")),  # Store as percentage
        vol_bucket=vol_bucket,
        weight_adjustment_factor=_quantize_weight(weight_adj),
        score_adjustment_factor=_quantize_weight(score_adj),
        confidence_penalty=_quantize_weight(confidence_penalty),
    )


def apply_volatility_to_score(
    raw_score: Decimal,
    vol_adjustment: VolatilityAdjustment,
) -> Decimal:
    """
    Apply volatility adjustment to a raw composite score.

    Args:
        raw_score: Pre-adjustment composite score (0-100)
        vol_adjustment: VolatilityAdjustment from compute_volatility_adjustment

    Returns:
        Volatility-adjusted score (0-100)
    """
    adjusted = raw_score * vol_adjustment.score_adjustment_factor
    return _clamp(_quantize_score(adjusted), Decimal("0"), Decimal("100"))


# =============================================================================
# MOMENTUM SIGNAL
# =============================================================================

def compute_momentum_signal(
    return_60d: Optional[Decimal],
    benchmark_return_60d: Optional[Decimal],
    *,
    annualized_vol: Optional[Decimal] = None,
    use_vol_adjusted_alpha: bool = False,
    lookback_days: int = MOMENTUM_LOOKBACK_DAYS,
) -> MomentumSignal:
    """
    Compute price momentum signal relative to benchmark (XBI).

    Research shows 60-day momentum relative to sector ETF is predictive
    in biotech. Captures both absolute and relative strength.

    COMPOUNDING REQUIREMENT (PIT-critical):
        Both return_60d and benchmark_return_60d MUST be computed as
        compounded returns: P_t / P_{t-60} - 1, NOT arithmetic sum
        of daily returns. Using summed daily returns introduces noise
        and can bias toward volatile names.

    PIT INTEGRITY:
        Both returns MUST use:
        - Same as-of close date (e.g., both as of 2026-01-15)
        - Same lookback count (trading days, not calendar days)
        - Aligned data sources (both from same price feed)
        A common bug is stock return using trading days while XBI
        uses calendar days - this breaks alpha calculation.

    SATURATION (v2 improvement):
        Uses slope=150 and clamp 5-95 to reduce ties at extremes.
        +10% alpha → 65 score (not 70 as in v1)
        -10% alpha → 35 score (not 30 as in v1)
        This improves rank granularity and IC.

    VOL-ADJUSTED ALPHA (optional):
        When use_vol_adjusted_alpha=True and annualized_vol is provided,
        computes alpha_adj = alpha / max(eps, vol). This rewards
        "persistent outperformance" vs "lucky squeeze" and typically
        improves IC in biotech where vol varies widely.

    Args:
        return_60d: Ticker's 60-day COMPOUNDED return as decimal (0.10 = 10%)
        benchmark_return_60d: XBI's 60-day COMPOUNDED return (same window!)
        annualized_vol: Optional annualized volatility for vol adjustment
        use_vol_adjusted_alpha: If True, use vol-adjusted alpha for scoring
        lookback_days: Lookback period (for documentation)

    Returns:
        MomentumSignal with normalized score and components
    """
    ret = _to_decimal(return_60d)
    bench = _to_decimal(benchmark_return_60d)
    vol = _to_decimal(annualized_vol)

    # Determine data completeness (for downstream confidence weighting)
    # 1.0 = all data present, 0.5 = partial, 0.0 = missing critical data
    if ret is not None and bench is not None:
        data_completeness = Decimal("1.0")
        if use_vol_adjusted_alpha and vol is None:
            data_completeness = Decimal("0.8")  # Missing vol for adjustment
    elif ret is not None or bench is not None:
        data_completeness = Decimal("0.3")  # Partial data
    else:
        data_completeness = Decimal("0.0")  # No data

    if ret is None or bench is None:
        return MomentumSignal(
            momentum_score=Decimal("50"),  # Neutral
            alpha_60d=None,
            alpha_vol_adjusted=None,
            return_60d=ret,
            benchmark_return_60d=bench,
            confidence=Decimal("0.3"),
            data_completeness=data_completeness,
        )

    # Compute alpha (excess return)
    alpha = ret - bench

    # Compute volatility-adjusted alpha if requested
    alpha_vol_adjusted: Optional[Decimal] = None
    if vol is not None and vol > MOMENTUM_VOL_EPS:
        alpha_vol_adjusted = alpha / vol

    # Select which alpha to use for scoring
    scoring_alpha = alpha
    if use_vol_adjusted_alpha and alpha_vol_adjusted is not None:
        # Vol-adjusted: scale back to similar range as raw alpha
        # Typical vol is 0.50 (50%), so vol-adjusted alpha ≈ 2x raw alpha
        # Normalize by baseline vol to keep score scale consistent
        scoring_alpha = alpha_vol_adjusted * VOLATILITY_BASELINE

    # Convert alpha to 0-100 score with reduced saturation
    # SLOPE=150: +10% alpha (1000bps) = +15 points → 65 score
    # Old SLOPE=200: +10% alpha = +20 points → 70 score (too aggressive)
    # This reduces ties at extremes and improves rank granularity
    score_delta = scoring_alpha * MOMENTUM_SLOPE

    momentum_score_raw = Decimal("50") + score_delta
    momentum_score_raw = _clamp(momentum_score_raw, MOMENTUM_SCORE_MIN, MOMENTUM_SCORE_MAX)

    # Confidence based on magnitude (larger moves = more confident)
    abs_alpha = abs(alpha)
    if abs_alpha >= Decimal("0.20"):
        confidence = Decimal("0.9")
    elif abs_alpha >= Decimal("0.10"):
        confidence = Decimal("0.7")
    elif abs_alpha >= Decimal("0.05"):
        confidence = Decimal("0.5")
    else:
        confidence = Decimal("0.4")

    # Apply confidence shrinkage: score_final = 50 + conf * (score_raw - 50)
    # This ensures low-confidence signals don't dominate
    # IPOs and sparse tickers will have their scores pulled toward neutral
    momentum_score = Decimal("50") + confidence * (momentum_score_raw - Decimal("50"))
    momentum_score = _clamp(momentum_score, MOMENTUM_SCORE_MIN, MOMENTUM_SCORE_MAX)

    return MomentumSignal(
        momentum_score=_quantize_score(momentum_score),
        alpha_60d=_quantize_weight(alpha),
        alpha_vol_adjusted=_quantize_weight(alpha_vol_adjusted) if alpha_vol_adjusted is not None else None,
        return_60d=_quantize_weight(ret),
        benchmark_return_60d=_quantize_weight(bench),
        confidence=confidence,
        data_completeness=data_completeness,
    )


def compute_momentum_signal_with_fallback(
    inputs: MultiWindowMomentumInput,
    *,
    use_vol_adjusted_alpha: bool = False,
) -> MomentumSignal:
    """
    Compute momentum signal with multi-window fallback for improved coverage.

    FALLBACK STRATEGY:
        1. Prefer 60d window (optimal for biotech IC per research)
        2. If 60d unavailable, try 120d (longer-term trend)
        3. If 120d unavailable, use 20d (short-term, lower confidence)
        4. If no windows available, return neutral with "missing_prices" status

    CONFIDENCE RULES (deterministic):
        - 60d window: conf = 0.7 (baseline)
        - 120d window: conf = 0.6 (longer term, slower signal)
        - 20d window: conf = 0.5 (short term, noisier)
        - No data: conf = 0.3 (neutral fallback)

        Additional adjustment based on trading_days_available:
        - >= 60 days: no penalty
        - 40-59 days: -0.1
        - 20-39 days: -0.2
        - < 20 days: -0.3

    DATA STATUS TRACKING:
        - "missing_prices": No return windows available
        - "computed_low_conf": Computed but confidence < 0.5
        - "applied": Signal computed and applied

    Args:
        inputs: MultiWindowMomentumInput with returns for each window
        use_vol_adjusted_alpha: Whether to use vol-adjusted alpha

    Returns:
        MomentumSignal with window_used and data_status populated
    """
    # Try windows in order of preference: 60d -> 120d -> 20d
    # Each tuple: (window_days, return, benchmark, is_return_clipped)
    window_order = [
        (60, inputs.return_60d, inputs.benchmark_60d, inputs.return_clipped_60d),
        (120, inputs.return_120d, inputs.benchmark_120d, inputs.return_clipped_120d),
        (20, inputs.return_20d, inputs.benchmark_20d, inputs.return_clipped_20d),
    ]

    selected_window = None
    selected_return = None
    selected_benchmark = None
    selected_was_clipped = False

    # Track guardrail events
    guardrail_flags: List[str] = []

    # Track benchmark availability for diagnostics
    benchmark_missing_for_available_return = False

    for window_days, ret, bench, was_clipped in window_order:
        ret_dec = _to_decimal(ret)
        bench_dec = _to_decimal(bench)

        if ret_dec is not None and bench_dec is not None:
            selected_window = window_days
            selected_return = ret_dec
            selected_benchmark = bench_dec
            selected_was_clipped = was_clipped
            break
        elif ret_dec is not None and bench_dec is None:
            # Return exists but benchmark missing - track for diagnostics
            benchmark_missing_for_available_return = True
            guardrail_flags.append(f"benchmark_missing_{window_days}d")

    # No windows available - missing prices
    if selected_window is None:
        return MomentumSignal(
            momentum_score=Decimal("50"),  # Neutral
            alpha_60d=None,
            alpha_vol_adjusted=None,
            return_60d=None,
            benchmark_return_60d=None,
            confidence=Decimal("0.3"),
            data_completeness=Decimal("0.0"),
            window_used=None,
            data_status="missing_prices",
            benchmark_missing=benchmark_missing_for_available_return,
            return_clipped=False,
            guardrail_flags=guardrail_flags,
        )

    # Track if the selected return was clipped
    if selected_was_clipped:
        guardrail_flags.append(f"return_clipped_{selected_window}d")

    # Compute alpha
    alpha = selected_return - selected_benchmark

    # Compute vol-adjusted alpha if requested
    vol = _to_decimal(inputs.annualized_vol)
    alpha_vol_adjusted: Optional[Decimal] = None
    if vol is not None and vol > MOMENTUM_VOL_EPS:
        alpha_vol_adjusted = alpha / vol

    # Select scoring alpha
    scoring_alpha = alpha
    if use_vol_adjusted_alpha and alpha_vol_adjusted is not None:
        scoring_alpha = alpha_vol_adjusted * VOLATILITY_BASELINE

    # Convert alpha to score (raw, before shrinkage)
    score_delta = scoring_alpha * MOMENTUM_SLOPE
    momentum_score_raw = Decimal("50") + score_delta
    momentum_score_raw = _clamp(momentum_score_raw, MOMENTUM_SCORE_MIN, MOMENTUM_SCORE_MAX)

    # Base confidence by window
    window_confidence = {
        60: Decimal("0.7"),
        120: Decimal("0.6"),
        20: Decimal("0.5"),
    }
    base_confidence = window_confidence.get(selected_window, Decimal("0.5"))

    # Adjust confidence based on trading days available
    trading_days = inputs.trading_days_available
    if trading_days is not None:
        if trading_days >= 60:
            days_penalty = Decimal("0.0")
        elif trading_days >= 40:
            days_penalty = Decimal("0.1")
        elif trading_days >= 20:
            days_penalty = Decimal("0.2")
        else:
            days_penalty = Decimal("0.3")
        base_confidence = max(Decimal("0.3"), base_confidence - days_penalty)

    # Boost confidence for strong signals
    abs_alpha = abs(alpha)
    if abs_alpha >= Decimal("0.20"):
        base_confidence = min(Decimal("0.9"), base_confidence + Decimal("0.2"))
    elif abs_alpha >= Decimal("0.10"):
        base_confidence = min(Decimal("0.8"), base_confidence + Decimal("0.1"))

    # Apply confidence shrinkage: score_final = 50 + conf * (score_raw - 50)
    # This ensures low-confidence signals (IPOs, sparse data) don't dominate
    momentum_score = Decimal("50") + base_confidence * (momentum_score_raw - Decimal("50"))
    momentum_score = _clamp(momentum_score, MOMENTUM_SCORE_MIN, MOMENTUM_SCORE_MAX)

    # Determine data status
    if base_confidence < Decimal("0.5"):
        data_status = "computed_low_conf"
    else:
        data_status = "applied"

    # Data completeness: 1.0 for 60d, 0.8 for 120d, 0.6 for 20d
    completeness_by_window = {
        60: Decimal("1.0"),
        120: Decimal("0.8"),
        20: Decimal("0.6"),
    }
    data_completeness = completeness_by_window.get(selected_window, Decimal("0.5"))

    return MomentumSignal(
        momentum_score=_quantize_score(momentum_score),
        alpha_60d=_quantize_weight(alpha),  # Always report as "alpha_60d" for compat
        alpha_vol_adjusted=_quantize_weight(alpha_vol_adjusted) if alpha_vol_adjusted else None,
        return_60d=_quantize_weight(selected_return),  # Report the window actually used
        benchmark_return_60d=_quantize_weight(selected_benchmark),
        confidence=_quantize_weight(base_confidence),
        data_completeness=data_completeness,
        window_used=selected_window,
        data_status=data_status,
        benchmark_missing=benchmark_missing_for_available_return,
        return_clipped=selected_was_clipped,
        guardrail_flags=guardrail_flags,
    )


# =============================================================================
# MULTI-WINDOW MOMENTUM BLENDING
# =============================================================================

# Multi-window blending weights (research-based)
# 60d is primary signal, 120d confirms trend, 20d captures recent moves
MULTIWINDOW_WEIGHTS = {
    20: Decimal("0.20"),   # Short-term: 20%
    60: Decimal("0.50"),   # Medium-term (primary): 50%
    120: Decimal("0.30"),  # Long-term (trend): 30%
}


def compute_momentum_signal_multiwindow(
    inputs: MultiWindowMomentumInput,
    *,
    use_vol_adjusted_alpha: bool = False,
    blend_mode: str = "weighted",  # "weighted" or "fallback"
) -> MomentumSignal:
    """
    Compute momentum signal using multi-window blending for robust signals.

    Unlike the fallback approach, this function blends available windows
    using a weighted average for a more stable momentum estimate.

    BLENDING STRATEGY:
        - Computes alpha for each available window
        - Blends alphas using normalized weights: 60d=50%, 120d=30%, 20d=20%
        - Re-normalizes weights based on available windows
        - Falls back to single-window if only one is available

    RESEARCH RATIONALE:
        - 20d captures short-term momentum reversals (1 month)
        - 60d is the primary signal (3 months, optimal IC in biotech)
        - 120d confirms longer-term trend (6 months)
        - Blending reduces noise and whipsaws from single-window approach

    CONFIDENCE RULES:
        - 3 windows: conf = 0.9 (full data)
        - 2 windows: conf = 0.8 (good coverage)
        - 1 window: conf = 0.6-0.7 (single window, use fallback logic)
        - 0 windows: conf = 0.3 (neutral, missing prices)

    Args:
        inputs: MultiWindowMomentumInput with returns for each window
        use_vol_adjusted_alpha: Whether to use vol-adjusted alpha
        blend_mode: "weighted" for multi-window blend, "fallback" for waterfall

    Returns:
        MomentumSignal with blended score and diagnostics
    """
    if blend_mode == "fallback":
        return compute_momentum_signal_with_fallback(inputs, use_vol_adjusted_alpha=use_vol_adjusted_alpha)

    # Collect available windows with their alphas
    available_windows = []
    guardrail_flags: List[str] = []

    window_data = [
        (20, inputs.return_20d, inputs.benchmark_20d, inputs.return_clipped_20d),
        (60, inputs.return_60d, inputs.benchmark_60d, inputs.return_clipped_60d),
        (120, inputs.return_120d, inputs.benchmark_120d, inputs.return_clipped_120d),
    ]

    for window_days, ret, bench, was_clipped in window_data:
        ret_dec = _to_decimal(ret)
        bench_dec = _to_decimal(bench)

        if ret_dec is not None and bench_dec is not None:
            alpha = ret_dec - bench_dec
            available_windows.append({
                "window": window_days,
                "alpha": alpha,
                "return": ret_dec,
                "benchmark": bench_dec,
                "was_clipped": was_clipped,
            })
            if was_clipped:
                guardrail_flags.append(f"return_clipped_{window_days}d")

    # No windows available - missing prices
    if not available_windows:
        return MomentumSignal(
            momentum_score=Decimal("50"),
            alpha_60d=None,
            alpha_vol_adjusted=None,
            return_60d=None,
            benchmark_return_60d=None,
            confidence=Decimal("0.3"),
            data_completeness=Decimal("0.0"),
            window_used=None,
            data_status="missing_prices",
            benchmark_missing=False,
            return_clipped=False,
            guardrail_flags=guardrail_flags,
        )

    # Single window - use that window directly
    if len(available_windows) == 1:
        w = available_windows[0]
        window_days = w["window"]
        alpha = w["alpha"]

        # Compute score from alpha
        score_delta = alpha * MOMENTUM_SLOPE
        momentum_score_raw = Decimal("50") + score_delta
        momentum_score_raw = _clamp(momentum_score_raw, MOMENTUM_SCORE_MIN, MOMENTUM_SCORE_MAX)

        # Single window confidence
        window_confidence = {20: Decimal("0.5"), 60: Decimal("0.7"), 120: Decimal("0.6")}
        base_confidence = window_confidence.get(window_days, Decimal("0.5"))

        # Apply shrinkage
        momentum_score = Decimal("50") + base_confidence * (momentum_score_raw - Decimal("50"))
        momentum_score = _clamp(momentum_score, MOMENTUM_SCORE_MIN, MOMENTUM_SCORE_MAX)

        guardrail_flags.append(f"single_window_{window_days}d")

        return MomentumSignal(
            momentum_score=_quantize_score(momentum_score),
            alpha_60d=_quantize_weight(alpha),
            alpha_vol_adjusted=None,
            return_60d=_quantize_weight(w["return"]),
            benchmark_return_60d=_quantize_weight(w["benchmark"]),
            confidence=_quantize_weight(base_confidence),
            data_completeness=Decimal("0.6"),
            window_used=window_days,
            data_status="applied",
            benchmark_missing=False,
            return_clipped=w["was_clipped"],
            guardrail_flags=guardrail_flags,
        )

    # Multiple windows - compute weighted blend
    # Normalize weights to sum to 1.0 based on available windows
    available_weight_sum = sum(
        MULTIWINDOW_WEIGHTS[w["window"]] for w in available_windows
    )
    normalized_weights = {
        w["window"]: MULTIWINDOW_WEIGHTS[w["window"]] / available_weight_sum
        for w in available_windows
    }

    # Compute weighted alpha blend
    blended_alpha = sum(
        w["alpha"] * normalized_weights[w["window"]]
        for w in available_windows
    )

    # Track which windows were used
    windows_used = sorted([w["window"] for w in available_windows])
    guardrail_flags.append(f"blended_windows_{'+'.join(map(str, windows_used))}")

    # Compute vol-adjusted alpha if requested
    vol = _to_decimal(inputs.annualized_vol)
    alpha_vol_adjusted: Optional[Decimal] = None
    if vol is not None and vol > MOMENTUM_VOL_EPS:
        alpha_vol_adjusted = blended_alpha / vol

    # Select scoring alpha
    scoring_alpha = blended_alpha
    if use_vol_adjusted_alpha and alpha_vol_adjusted is not None:
        scoring_alpha = alpha_vol_adjusted * VOLATILITY_BASELINE

    # Convert alpha to score
    score_delta = scoring_alpha * MOMENTUM_SLOPE
    momentum_score_raw = Decimal("50") + score_delta
    momentum_score_raw = _clamp(momentum_score_raw, MOMENTUM_SCORE_MIN, MOMENTUM_SCORE_MAX)

    # Confidence based on number of windows
    if len(available_windows) >= 3:
        base_confidence = Decimal("0.9")  # Full data
    else:
        base_confidence = Decimal("0.8")  # 2 windows

    # Boost confidence for strong blended signals
    abs_alpha = abs(blended_alpha)
    if abs_alpha >= Decimal("0.15"):
        base_confidence = min(Decimal("0.95"), base_confidence + Decimal("0.05"))

    # Apply shrinkage
    momentum_score = Decimal("50") + base_confidence * (momentum_score_raw - Decimal("50"))
    momentum_score = _clamp(momentum_score, MOMENTUM_SCORE_MIN, MOMENTUM_SCORE_MAX)

    # Data completeness: higher for more windows
    completeness = Decimal("0.7") + Decimal("0.1") * len(available_windows)
    completeness = min(completeness, Decimal("1.0"))

    # Get 60d values for backward compatibility (or best available)
    w60 = next((w for w in available_windows if w["window"] == 60), available_windows[0])

    return MomentumSignal(
        momentum_score=_quantize_score(momentum_score),
        alpha_60d=_quantize_weight(blended_alpha),  # Report blended alpha
        alpha_vol_adjusted=_quantize_weight(alpha_vol_adjusted) if alpha_vol_adjusted else None,
        return_60d=_quantize_weight(w60["return"]),
        benchmark_return_60d=_quantize_weight(w60["benchmark"]),
        confidence=_quantize_weight(base_confidence),
        data_completeness=completeness,
        window_used=60 if 60 in windows_used else windows_used[0],  # Primary or first
        data_status="applied",
        benchmark_missing=False,
        return_clipped=any(w["was_clipped"] for w in available_windows),
        guardrail_flags=guardrail_flags,
    )


# =============================================================================
# PEER-RELATIVE VALUATION
# =============================================================================

def _classify_valuation_regime(
    revenue_mm: Optional[Decimal],
    cfo_mm: Optional[Decimal],
    has_revenue: bool = False,
) -> Tuple[ValuationRegime, List[str]]:
    """
    Classify company into valuation regime based on financial characteristics.

    COMMERCIAL if ANY of:
    - revenue_mm >= $200M
    - cfo_mm > 0 (positive operating cash flow)

    DEVELOPMENT otherwise (pipeline companies where mcap/trial matters)

    Args:
        revenue_mm: Trailing twelve month revenue in millions
        cfo_mm: Cash flow from operations in millions
        has_revenue: Flag indicating any material revenue

    Returns:
        (ValuationRegime, list of classification flags)
    """
    flags: List[str] = []

    revenue = _to_decimal(revenue_mm)
    cfo = _to_decimal(cfo_mm)

    # Check commercial conditions
    is_commercial = False

    if revenue is not None and revenue >= VALUATION_COMMERCIAL_REVENUE_THRESHOLD_MM:
        is_commercial = True
        flags.append("commercial_revenue_threshold")

    if cfo is not None and cfo > Decimal("0"):
        is_commercial = True
        flags.append("commercial_cfo_positive")

    if is_commercial:
        return ValuationRegime.COMMERCIAL, flags

    # Default to development
    return ValuationRegime.DEVELOPMENT, flags


def _compute_commercial_valuation(
    market_cap_mm: Decimal,
    enterprise_value_mm: Optional[Decimal],
    cfo_mm: Optional[Decimal],
    revenue_mm: Optional[Decimal],
    peer_valuations: List[Dict[str, Any]],
) -> ValuationSignal:
    """
    Compute valuation signal for commercial-stage companies.

    Uses EV/CFO (preferred) or MCAP/Revenue as valuation metric.
    Compares to commercial peers (revenue > threshold or CFO > 0).

    Args:
        market_cap_mm: Market cap in millions
        enterprise_value_mm: Enterprise value in millions (optional)
        cfo_mm: Cash flow from operations in millions
        revenue_mm: Revenue in millions
        peer_valuations: List of peer dicts

    Returns:
        ValuationSignal with commercial valuation score
    """
    flags: List[str] = []
    ev = _to_decimal(enterprise_value_mm) or market_cap_mm
    cfo = _to_decimal(cfo_mm)
    revenue = _to_decimal(revenue_mm)

    # Filter to commercial peers
    commercial_peers = [
        p for p in peer_valuations
        if (_to_decimal(p.get("revenue_mm")) is not None and
            _to_decimal(p.get("revenue_mm")) >= VALUATION_COMMERCIAL_REVENUE_THRESHOLD_MM) or
           (_to_decimal(p.get("cfo_mm")) is not None and
            _to_decimal(p.get("cfo_mm")) > Decimal("0"))
    ]

    # Determine which metric to use: EV/CFO (preferred) or EV/Revenue
    use_cfo = cfo is not None and cfo > Decimal("0")
    method = "ev_cfo" if use_cfo else "ev_revenue"

    if use_cfo:
        # EV/CFO valuation
        ev_multiple = ev / cfo
        ev_multiple_clamped = _clamp(ev_multiple, VALUATION_COMMERCIAL_EV_CFO_MIN, VALUATION_COMMERCIAL_EV_CFO_MAX)

        # Get peer EV/CFO multiples
        peer_multiples: List[Decimal] = []
        for p in commercial_peers:
            p_ev = _to_decimal(p.get("enterprise_value_mm")) or _to_decimal(p.get("market_cap_mm"))
            p_cfo = _to_decimal(p.get("cfo_mm"))
            if p_ev is not None and p_cfo is not None and p_cfo > Decimal("0"):
                p_mult = _clamp(p_ev / p_cfo, VALUATION_COMMERCIAL_EV_CFO_MIN, VALUATION_COMMERCIAL_EV_CFO_MAX)
                peer_multiples.append(p_mult)

        flags.append("valuation_ev_cfo")

    elif revenue is not None and revenue > Decimal("0"):
        # EV/Revenue valuation (fallback)
        ev_multiple = ev / revenue
        ev_multiple_clamped = _clamp(ev_multiple, VALUATION_COMMERCIAL_EV_REV_MIN, VALUATION_COMMERCIAL_EV_REV_MAX)
        method = "ev_revenue"

        # Get peer EV/Revenue multiples
        peer_multiples = []
        for p in commercial_peers:
            p_ev = _to_decimal(p.get("enterprise_value_mm")) or _to_decimal(p.get("market_cap_mm"))
            p_rev = _to_decimal(p.get("revenue_mm"))
            if p_ev is not None and p_rev is not None and p_rev > Decimal("0"):
                p_mult = _clamp(p_ev / p_rev, VALUATION_COMMERCIAL_EV_REV_MIN, VALUATION_COMMERCIAL_EV_REV_MAX)
                peer_multiples.append(p_mult)

        flags.append("valuation_ev_revenue")

    else:
        # No valid commercial metric available
        flags.append("valuation_commercial_no_metric")
        return ValuationSignal(
            valuation_score=Decimal("50"),
            mcap_per_asset=None,
            peer_median_mcap_per_asset=None,
            peer_count=0,
            confidence=Decimal("0.2"),
            regime=ValuationRegime.COMMERCIAL,
            method="neutral",
            ev_multiple=None,
            peer_median_ev_multiple=None,
            flags=flags,
        )

    # Need minimum peers for reliable comparison
    if len(peer_multiples) < VALUATION_COMMERCIAL_MIN_PEERS:
        flags.append("valuation_insufficient_commercial_peers")
        return ValuationSignal(
            valuation_score=Decimal("50"),
            mcap_per_asset=None,
            peer_median_mcap_per_asset=None,
            peer_count=len(peer_multiples),
            confidence=Decimal("0.3"),
            regime=ValuationRegime.COMMERCIAL,
            method=method,
            ev_multiple=_quantize_score(ev_multiple_clamped),
            peer_median_ev_multiple=None,
            flags=flags,
        )

    n_peers = len(peer_multiples)

    # Compute percentile (lower multiple = cheaper = higher score)
    lt_count = sum(1 for p in peer_multiples if p < ev_multiple_clamped)
    eq_count = sum(1 for p in peer_multiples if p == ev_multiple_clamped)

    percentile = (
        (Decimal(lt_count) + Decimal("0.5") * Decimal(eq_count))
        / Decimal(n_peers)
        * Decimal("100")
    )

    # Invert: lower multiple = higher score (cheaper is better)
    raw_valuation_score = Decimal("100") - percentile

    # Compute peer median
    sorted_multiples = sorted(peer_multiples)
    mid = n_peers // 2
    if n_peers % 2 == 0:
        peer_median_mult = (sorted_multiples[mid - 1] + sorted_multiples[mid]) / Decimal("2")
    else:
        peer_median_mult = sorted_multiples[mid]

    # Confidence ramp
    confidence = _clamp(
        VALUATION_CONFIDENCE_BASE + VALUATION_CONFIDENCE_SLOPE * Decimal(n_peers),
        Decimal("0.2"),
        VALUATION_CONFIDENCE_MAX,
    )

    # Shrinkage for small samples
    if n_peers < VALUATION_SHRINKAGE_FULL_AT_N:
        shrink_range = VALUATION_SHRINKAGE_FULL_AT_N - VALUATION_COMMERCIAL_MIN_PEERS
        if shrink_range > 0:
            progress = Decimal(n_peers - VALUATION_COMMERCIAL_MIN_PEERS) / Decimal(shrink_range)
            shrink_factor = VALUATION_SHRINKAGE_MAX * (Decimal("1") - progress)
        else:
            shrink_factor = VALUATION_SHRINKAGE_MAX
        raw_valuation_score = (
            raw_valuation_score * (Decimal("1") - shrink_factor)
            + Decimal("50") * shrink_factor
        )

    # Clamp final score
    valuation_score = _clamp(raw_valuation_score, Decimal("10"), Decimal("90"))

    return ValuationSignal(
        valuation_score=_quantize_score(valuation_score),
        mcap_per_asset=None,
        peer_median_mcap_per_asset=None,
        peer_count=n_peers,
        confidence=_quantize_weight(confidence),
        regime=ValuationRegime.COMMERCIAL,
        method=method,
        ev_multiple=_quantize_score(ev_multiple_clamped),
        peer_median_ev_multiple=_quantize_score(peer_median_mult),
        flags=flags,
    )


def _compute_development_valuation(
    market_cap_mm: Decimal,
    trial_count: int,
    lead_phase: Optional[str],
    peer_valuations: List[Dict[str, Any]],
) -> ValuationSignal:
    """
    Compute valuation signal for development-stage companies.

    Uses mcap/trial_count vs stage-matched peers.
    Includes trial_count < 3 gate for unreliable denominators.

    Args:
        market_cap_mm: Market cap in millions
        trial_count: Number of active clinical trials
        lead_phase: Lead development phase
        peer_valuations: List of peer dicts

    Returns:
        ValuationSignal with development-stage valuation score
    """
    flags: List[str] = []

    # GATE: If trial_count < 3, mcap/trial is unreliable -> neutral score
    if trial_count < VALUATION_DEV_MIN_TRIALS:
        flags.append("trial_count_low")
        flags.append(f"trial_count_{trial_count}")
        return ValuationSignal(
            valuation_score=Decimal("50"),
            mcap_per_asset=market_cap_mm / Decimal(max(1, trial_count)) if trial_count > 0 else None,
            peer_median_mcap_per_asset=None,
            peer_count=0,
            confidence=Decimal("0.2"),
            regime=ValuationRegime.DEVELOPMENT,
            method="neutral",
            ev_multiple=None,
            peer_median_ev_multiple=None,
            flags=flags,
        )

    # Winsorize inputs
    mcap_winsorized = _clamp(market_cap_mm, VALUATION_MCAP_MIN_MM, VALUATION_MCAP_MAX_MM)
    trials_winsorized = max(VALUATION_TRIAL_COUNT_MIN, min(trial_count, VALUATION_TRIAL_COUNT_MAX))

    # Compute mcap per asset
    mcap_per_asset = mcap_winsorized / Decimal(trials_winsorized)

    # Determine stage bucket
    stage = _stage_bucket(lead_phase)

    # Filter to same-stage development peers with valid trial data
    same_stage_peers = [
        p for p in peer_valuations
        if p.get("stage_bucket") == stage and p.get("trial_count", 0) >= VALUATION_DEV_MIN_TRIALS
    ]

    if len(same_stage_peers) < VALUATION_MIN_PEERS:
        flags.append("insufficient_stage_peers")
        return ValuationSignal(
            valuation_score=Decimal("50"),
            mcap_per_asset=_quantize_score(mcap_per_asset),
            peer_median_mcap_per_asset=None,
            peer_count=len(same_stage_peers),
            confidence=Decimal("0.3"),
            regime=ValuationRegime.DEVELOPMENT,
            method="mcap_per_trial",
            ev_multiple=None,
            peer_median_ev_multiple=None,
            flags=flags,
        )

    # Compute peer mcap/asset values
    peer_mcap_per_asset: List[Decimal] = []
    for p in same_stage_peers:
        p_mcap = _to_decimal(p.get("market_cap_mm"))
        p_trials = p.get("trial_count", 0)
        if p_mcap is not None and p_trials >= VALUATION_DEV_MIN_TRIALS:
            p_mcap_w = _clamp(p_mcap, VALUATION_MCAP_MIN_MM, VALUATION_MCAP_MAX_MM)
            p_trials_w = max(VALUATION_TRIAL_COUNT_MIN, min(p_trials, VALUATION_TRIAL_COUNT_MAX))
            peer_mcap_per_asset.append(p_mcap_w / Decimal(p_trials_w))

    if not peer_mcap_per_asset:
        flags.append("no_valid_peers")
        return ValuationSignal(
            valuation_score=Decimal("50"),
            mcap_per_asset=_quantize_score(mcap_per_asset),
            peer_median_mcap_per_asset=None,
            peer_count=0,
            confidence=Decimal("0.2"),
            regime=ValuationRegime.DEVELOPMENT,
            method="mcap_per_trial",
            ev_multiple=None,
            peer_median_ev_multiple=None,
            flags=flags,
        )

    n_peers = len(peer_mcap_per_asset)

    # Compute percentile (lower mcap/asset = cheaper = higher score)
    lt_count = sum(1 for p in peer_mcap_per_asset if p < mcap_per_asset)
    eq_count = sum(1 for p in peer_mcap_per_asset if p == mcap_per_asset)

    percentile = (
        (Decimal(lt_count) + Decimal("0.5") * Decimal(eq_count))
        / Decimal(n_peers)
        * Decimal("100")
    )

    raw_valuation_score = Decimal("100") - percentile

    # Peer median
    sorted_peers = sorted(peer_mcap_per_asset)
    mid = n_peers // 2
    if n_peers % 2 == 0:
        peer_median = (sorted_peers[mid - 1] + sorted_peers[mid]) / Decimal("2")
    else:
        peer_median = sorted_peers[mid]

    # Confidence ramp
    confidence = _clamp(
        VALUATION_CONFIDENCE_BASE + VALUATION_CONFIDENCE_SLOPE * Decimal(n_peers),
        Decimal("0.2"),
        VALUATION_CONFIDENCE_MAX,
    )

    # CV penalty for tight distributions
    if n_peers >= 3:
        peer_mean = sum(peer_mcap_per_asset) / Decimal(n_peers)
        if peer_mean > EPS:
            variance = sum((p - peer_mean) ** 2 for p in peer_mcap_per_asset) / Decimal(n_peers)
            std_approx = _decimal_sqrt_approx(variance)
            cv = std_approx / peer_mean
            if cv < Decimal("0.3"):
                cv_penalty = (Decimal("0.3") - cv) / Decimal("0.3") * Decimal("0.2")
                confidence = _clamp(confidence - cv_penalty, Decimal("0.2"), VALUATION_CONFIDENCE_MAX)

    # Shrinkage for small samples
    if n_peers < VALUATION_SHRINKAGE_FULL_AT_N:
        shrink_range = VALUATION_SHRINKAGE_FULL_AT_N - VALUATION_MIN_PEERS
        if shrink_range > 0:
            progress = Decimal(n_peers - VALUATION_MIN_PEERS) / Decimal(shrink_range)
            shrink_factor = VALUATION_SHRINKAGE_MAX * (Decimal("1") - progress)
        else:
            shrink_factor = VALUATION_SHRINKAGE_MAX
        raw_valuation_score = (
            raw_valuation_score * (Decimal("1") - shrink_factor)
            + Decimal("50") * shrink_factor
        )

    # Clamp
    valuation_score = _clamp(raw_valuation_score, Decimal("10"), Decimal("90"))

    flags.append("valuation_mcap_per_trial")

    return ValuationSignal(
        valuation_score=_quantize_score(valuation_score),
        mcap_per_asset=_quantize_score(mcap_per_asset),
        peer_median_mcap_per_asset=_quantize_score(peer_median),
        peer_count=n_peers,
        confidence=_quantize_weight(confidence),
        regime=ValuationRegime.DEVELOPMENT,
        method="mcap_per_trial",
        ev_multiple=None,
        peer_median_ev_multiple=None,
        flags=flags,
    )


def compute_valuation_signal(
    market_cap_mm: Optional[Decimal],
    trial_count: int,
    lead_phase: Optional[str],
    peer_valuations: List[Dict[str, Any]],
    # V2: Financial data for regime routing (optional for backwards compatibility)
    revenue_mm: Optional[Decimal] = None,
    cfo_mm: Optional[Decimal] = None,
    enterprise_value_mm: Optional[Decimal] = None,
    has_revenue: bool = False,
) -> ValuationSignal:
    """
    Compute peer-relative valuation signal with regime-aware routing.

    V2 ENHANCEMENT: Routes to different valuation models based on company stage:
    - COMMERCIAL (revenue >= $200M or CFO > 0): Uses EV/CFO or EV/Revenue
    - DEVELOPMENT (pipeline companies): Uses mcap/trial_count with trial >= 3 gate

    DETERMINISM: All calculations use pure Decimal arithmetic.
    ROBUSTNESS: Winsorizes inputs, gates unreliable denominators.
    TIE-HANDLING: Uses midrank for ties to avoid bias.
    SHRINKAGE: Shrinks toward neutral (50) when peer sample is small.

    Args:
        market_cap_mm: Market cap in millions
        trial_count: Number of active clinical trials
        lead_phase: Lead development phase (must be PIT-safe)
        peer_valuations: List of peer dicts with market_cap_mm, trial_count, stage_bucket,
                         and optionally revenue_mm, cfo_mm, enterprise_value_mm
        revenue_mm: Company's TTM revenue in millions (for regime classification)
        cfo_mm: Company's TTM operating cash flow in millions (for regime classification)
        enterprise_value_mm: Company's enterprise value in millions (for commercial valuation)
        has_revenue: Flag indicating any material revenue

    Returns:
        ValuationSignal with normalized score, regime, method, and diagnostic flags
    """
    mcap = _to_decimal(market_cap_mm)

    if mcap is None:
        return ValuationSignal(
            valuation_score=Decimal("50"),
            mcap_per_asset=None,
            peer_median_mcap_per_asset=None,
            peer_count=0,
            confidence=Decimal("0.2"),
            regime=ValuationRegime.UNKNOWN,
            method="neutral",
            ev_multiple=None,
            peer_median_ev_multiple=None,
            flags=["missing_market_cap"],
        )

    # Step 1: Classify valuation regime
    regime, regime_flags = _classify_valuation_regime(revenue_mm, cfo_mm, has_revenue)

    # Step 2: Route to appropriate valuation model
    if regime == ValuationRegime.COMMERCIAL:
        result = _compute_commercial_valuation(
            mcap, enterprise_value_mm, cfo_mm, revenue_mm, peer_valuations
        )
        # Add regime classification flags
        result.flags = regime_flags + result.flags
        return result

    else:
        # DEVELOPMENT or UNKNOWN -> use development model
        result = _compute_development_valuation(
            mcap, trial_count, lead_phase, peer_valuations
        )
        # Add regime classification flags
        result.flags = regime_flags + result.flags
        return result


def _decimal_sqrt_approx(value: Decimal, iterations: int = 10) -> Decimal:
    """
    Approximate square root using Newton-Raphson method.

    Pure Decimal implementation for determinism (no math.sqrt).

    Args:
        value: Non-negative Decimal
        iterations: Number of Newton-Raphson iterations

    Returns:
        Approximate square root as Decimal
    """
    if value <= Decimal("0"):
        return Decimal("0")

    # Initial guess: value / 2 (or 1 if value < 1)
    x = value / Decimal("2") if value > Decimal("1") else Decimal("1")

    # Newton-Raphson: x_new = (x + value/x) / 2
    for _ in range(iterations):
        x = (x + value / x) / Decimal("2")

    return x


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


# =============================================================================
# CATALYST SIGNAL DECAY
# =============================================================================

def compute_catalyst_decay(
    days_to_catalyst: Optional[int],
    event_type: str,
    *,
    optimal_window: int = CATALYST_OPTIMAL_WINDOW_DAYS,
) -> CatalystDecayResult:
    """
    Compute time-based decay factor for catalyst signals.

    Research shows IC peaks ~30 days before catalyst event, then decays
    both before (too early) and after (already priced in).

    Uses ASYMMETRIC exponential decay:
    - Before event: slow rise as event approaches (standard tau)
    - After event: fast decay (tau * POST_EVENT_DECAY_MULT)

    This models the biotech reality that post-event information gets
    priced quickly while pre-event anticipation builds slowly.

    DETERMINISM: Uses Decimal.exp() exclusively - no floats.

    Args:
        days_to_catalyst: Days until catalyst event (negative = past)
        event_type: Type of catalyst (PDUFA, DATA_READOUT, etc.)
            Will be normalized to uppercase for matching.
        optimal_window: Days before event for peak signal (default 30)

    Returns:
        CatalystDecayResult with decay factor and metadata
    """
    # Normalize event_type for robust matching
    event_type_normalized = event_type.strip().upper() if event_type else "DEFAULT"

    if days_to_catalyst is None:
        return CatalystDecayResult(
            decay_factor=Decimal("0.5"),  # Neutral
            days_to_catalyst=None,
            event_type=event_type_normalized,
            in_optimal_window=False,
        )

    # Get decay rate for event type (normalized lookup)
    tau = CATALYST_DECAY_RATES.get(event_type_normalized, CATALYST_DECAY_RATES["DEFAULT"])

    # Convert to Decimal for all arithmetic
    days = Decimal(days_to_catalyst)
    optimal = Decimal(optimal_window)

    # Distance from optimal window (signed: negative = past optimal)
    # days_to_catalyst=30 at optimal_window=30 -> d=0 (peak)
    # days_to_catalyst=60 at optimal_window=30 -> d=30 (30 days before optimal)
    # days_to_catalyst=0 at optimal_window=30 -> d=-30 (at event, past optimal)
    # days_to_catalyst=-10 at optimal_window=30 -> d=-40 (event 10 days ago)
    d = days - optimal

    # Check if in optimal window (within CATALYST_OPTIMAL_HALF_WIDTH days of peak)
    # Widened from 15 to 30 days for better coverage
    in_optimal_window = abs(d) <= Decimal(str(CATALYST_OPTIMAL_HALF_WIDTH))

    # ASYMMETRIC DECAY:
    # - d > 0: event is far out, signal building (use standard tau)
    # - d < 0: past optimal peak, signal decaying (use tau * POST_EVENT_DECAY_MULT)
    #
    # Note: d < 0 includes both "approaching event" (days_to_catalyst < optimal)
    # and "event already happened" (days_to_catalyst < 0). Both should decay
    # faster because uncertainty is resolving.
    if d < Decimal("0"):
        tau_effective = tau * CATALYST_POST_EVENT_DECAY_MULT
    else:
        tau_effective = tau

    # Compute decay using Decimal.exp() for determinism
    # decay = exp(-tau_effective * |distance|)
    distance = abs(d)
    exponent = -(tau_effective * distance)

    # Decimal.exp() is deterministic across platforms
    # Clamp exponent to avoid underflow (exp(-50) ≈ 2e-22, effectively 0)
    if exponent < Decimal("-50"):
        decay_factor = Decimal("0")
    else:
        decay_factor = exponent.exp()

    # Clamp to valid range - use CATALYST_DECAY_FLOOR for minimum
    decay_factor = _clamp(decay_factor, CATALYST_DECAY_FLOOR, Decimal("1.0"))

    # Quantize with fine precision to avoid tie artifacts
    decay_factor = decay_factor.quantize(CATALYST_DECAY_PRECISION, rounding=ROUND_HALF_UP)

    return CatalystDecayResult(
        decay_factor=decay_factor,
        days_to_catalyst=days_to_catalyst,
        event_type=event_type_normalized,
        in_optimal_window=in_optimal_window,
    )


def apply_catalyst_decay(
    catalyst_score: Decimal,
    decay_result: CatalystDecayResult,
) -> Decimal:
    """
    Apply decay factor to catalyst score.

    Args:
        catalyst_score: Raw catalyst score (0-100)
        decay_result: CatalystDecayResult from compute_catalyst_decay

    Returns:
        Decay-adjusted catalyst score
    """
    # Decay toward neutral (50) rather than toward 0
    neutral = Decimal("50")
    deviation_from_neutral = catalyst_score - neutral
    decayed_deviation = deviation_from_neutral * decay_result.decay_factor

    adjusted_score = neutral + decayed_deviation
    return _clamp(_quantize_score(adjusted_score), Decimal("0"), Decimal("100"))


# =============================================================================
# SMART MONEY SIGNAL
# =============================================================================


@dataclass
class _NormalizedHolderId:
    """Normalized holder identifier for consistent lookup."""
    raw: str           # Original input, whitespace-stripped
    name_lower: str    # Lowercase for name matching
    is_cik: bool       # True if input is numeric (CIK)
    cik_raw: str       # Original CIK string (if is_cik)
    cik_clean: str     # CIK without leading zeros (if is_cik)
    is_invalid: bool   # True if empty or malformed


def _normalize_holder_id(holder_name: str) -> _NormalizedHolderId:
    """
    Normalize a holder identifier for consistent lookup.

    Handles:
    - Whitespace stripping
    - Empty/malformed input detection
    - CIK detection (all digits)
    - Edge case: "0000000000" → cik_clean = "0" (not empty)

    DETERMINISM: Pure function, no external calls.

    Args:
        holder_name: Raw holder name or CIK string

    Returns:
        _NormalizedHolderId with all normalized forms
    """
    raw = holder_name.strip() if holder_name else ""

    # Handle empty/invalid input
    if not raw:
        return _NormalizedHolderId(
            raw="", name_lower="", is_cik=False,
            cik_raw="", cik_clean="", is_invalid=True
        )

    name_lower = raw.lower()

    # Check if it's a CIK (all digits)
    if raw.isdigit():
        # Handle edge case: "0000000000" → "0", not ""
        cik_clean = raw.lstrip('0') or "0"
        return _NormalizedHolderId(
            raw=raw, name_lower=name_lower, is_cik=True,
            cik_raw=raw, cik_clean=cik_clean, is_invalid=False
        )

    # It's a name, not a CIK
    return _NormalizedHolderId(
        raw=raw, name_lower=name_lower, is_cik=False,
        cik_raw="", cik_clean="", is_invalid=False
    )


def _get_holder_tier(holder_name: str, holder_tiers: Optional[Dict[str, int]] = None) -> int:
    """
    Get tier for a holder by name or CIK.

    Lookup order:
    1. Explicit holder_tiers dict (if provided)
    2. CIK-based lookup (if holder_name looks like a CIK)
    3. Name-based pattern matching

    Returns tier 0 for unknown holders (will use SMART_MONEY_UNKNOWN_TIER_WEIGHT).

    DETERMINISM: Uses _normalize_holder_id for consistent normalization.

    Args:
        holder_name: Name or CIK of the holder/manager
        holder_tiers: Optional dict mapping holder name -> tier (1, 2, or 3)

    Returns:
        Tier number (1, 2, 3) or 0 for unknown
    """
    nid = _normalize_holder_id(holder_name)

    # Invalid input → unknown
    if nid.is_invalid:
        return 0

    # First check explicit mapping (e.g., from 13F metadata)
    if holder_tiers:
        for name_variant, tier in holder_tiers.items():
            if name_variant.lower() == nid.name_lower:
                return tier

    # Check if it's a CIK
    if nid.is_cik:
        # Check CIK mapping (try both raw and cleaned)
        if nid.cik_raw in SMART_MONEY_TIER_BY_CIK:
            return SMART_MONEY_TIER_BY_CIK[nid.cik_raw]
        if nid.cik_clean in SMART_MONEY_TIER_BY_CIK:
            return SMART_MONEY_TIER_BY_CIK[nid.cik_clean]

    # Fall back to built-in name mapping
    for name_pattern, tier in SMART_MONEY_TIER_BY_NAME.items():
        if name_pattern in nid.name_lower or nid.name_lower in name_pattern:
            return tier

    return 0  # Unknown


def _get_holder_weight(tier: int) -> Decimal:
    """Get weight for a holder tier. Unknown (tier 0) gets reduced weight."""
    if tier == 0:
        return SMART_MONEY_UNKNOWN_TIER_WEIGHT
    return SMART_MONEY_TIER_WEIGHTS.get(tier, SMART_MONEY_UNKNOWN_TIER_WEIGHT)


def _is_conditional_manager(holder_name: str) -> bool:
    """
    Check if a holder is a Conditional (multi-strategy/quant) manager.

    Conditional managers have their signal contribution capped to prevent
    their large AUM from dominating the biotech-specialist signal.

    DETERMINISM: Uses _normalize_holder_id for consistent normalization.

    Args:
        holder_name: Name or CIK of the holder/manager

    Returns:
        True if conditional manager, False otherwise
    """
    nid = _normalize_holder_id(holder_name)

    # Invalid input → not conditional
    if nid.is_invalid:
        return False

    # Check if it's a CIK
    if nid.is_cik:
        # Check CIK-based Conditional set (try both raw and cleaned)
        if nid.cik_raw in SMART_MONEY_CONDITIONAL_CIKS:
            return True
        if nid.cik_clean in SMART_MONEY_CONDITIONAL_CIKS:
            return True

    # Fall back to name-based lookup
    for pattern in SMART_MONEY_CONDITIONAL_MANAGERS:
        if pattern in nid.name_lower or nid.name_lower in pattern:
            return True
    return False


def _saturating_bonus(weighted_sum: Decimal) -> Decimal:
    """
    Apply saturating function to weighted overlap.

    Piecewise linear:
    - 0 to threshold: linear bonus
    - Above threshold: diminishing returns (sqrt-like)

    This prevents gaming by many small holders.

    Args:
        weighted_sum: Sum of holder tier weights

    Returns:
        Bonus score (bounded by SMART_MONEY_OVERLAP_MAX_BONUS)
    """
    threshold = SMART_MONEY_OVERLAP_SATURATION_THRESHOLD
    linear_rate = SMART_MONEY_OVERLAP_LINEAR_BONUS
    max_bonus = SMART_MONEY_OVERLAP_MAX_BONUS

    if weighted_sum <= Decimal("0"):
        return Decimal("0")

    if weighted_sum <= threshold:
        # Linear region: full credit
        bonus = weighted_sum * linear_rate
    else:
        # Saturation region: diminishing returns
        # Linear portion up to threshold
        linear_part = threshold * linear_rate
        # Excess gets sqrt-like treatment (approximated as log-like)
        excess = weighted_sum - threshold
        # Diminishing returns: each additional 1.0 weight adds less
        # Use: bonus_excess = linear_rate * (1 - e^(-excess))
        # Approximated as: bonus_excess = linear_rate * min(excess, 1.0) * 0.5
        diminishing = linear_rate * _clamp(excess, Decimal("0"), Decimal("2")) * Decimal("0.3")
        bonus = linear_part + diminishing

    return _clamp(bonus, Decimal("0"), max_bonus)


def compute_smart_money_signal(
    overlap_count: int,
    holders: List[str],
    position_changes: Optional[Dict[str, str]] = None,
    holder_tiers: Optional[Dict[str, int]] = None,
) -> SmartMoneySignal:
    """
    Compute smart money signal from 13F co-invest data.

    V2 IMPROVEMENTS:
    1. Weights by holder tier (Tier1=1.0, Tier2=0.6, Unknown=0.2)
    2. Applies saturating function for overlap (diminishing returns)
    3. Caps per-holder contribution to prevent one noisy filing dominating
    4. Reduces EXIT weight (data quality sensitive)
    5. Deterministic: sorts holders before processing

    V3 IMPROVEMENTS:
    6. Separates Elite Core vs Conditional manager contributions
    7. Caps Conditional signal at 30% to prevent AUM dominance
    8. Elite Core (~$65B) drives signal, Conditional (~$1.1T) provides breadth

    BREADTH vs DIRECTION:
    - Overlap bonus (breadth): How many high-quality co-investors
    - Change bonus (direction): Net weighted change (new/increase vs decrease/exit)

    TIER SENSITIVITY:
    - 2 Tier1 holders beats 4 unknown holders
    - 1 Tier1 + 1 Tier2 = 1.6 weighted overlap vs 2 unknowns = 0.4

    CONDITIONAL CAP (V3):
    - Conditional managers (multi-strategy, quant) capped at 30% of total signal
    - Prevents $1.1T AUM from drowning out $65B of biotech-specialist signal
    - Elite Core always contributes ≥70% of smart money signal

    PIT SAFETY:
    - position_changes should use filing dates, not quarter-end
    - "NEW" must mean "new to tracked managers", not "new to your dataset"

    Args:
        overlap_count: Number of tracked managers holding the position (raw count)
        holders: List of holder names (used for tier lookup)
        position_changes: Dict mapping holder -> change type (NEW, INCREASE, etc.)
        holder_tiers: Optional dict mapping holder name -> tier (1, 2, 3)
            If not provided, uses name-based lookup from SMART_MONEY_TIER_BY_NAME

    Returns:
        SmartMoneySignal with normalized score and components
    """
    base_signal = Decimal("50")

    # DETERMINISM: Sort holders for consistent iteration order
    sorted_holders = sorted(holders) if holders else []

    # =========================================================================
    # STEP 1: Compute tier-weighted overlap (breadth signal)
    # V3: Separate Elite Core vs Conditional contributions
    # =========================================================================
    tier_breakdown: Dict[int, int] = {1: 0, 2: 0, 3: 0, 0: 0}
    weighted_overlap = Decimal("0")
    tier1_holders: List[str] = []
    per_holder_contributions: Dict[str, Decimal] = {}

    # V3: Track Elite Core vs Conditional separately
    elite_core_overlap = Decimal("0")
    conditional_overlap = Decimal("0")

    for holder in sorted_holders:
        tier = _get_holder_tier(holder, holder_tiers)
        tier_breakdown[tier] = tier_breakdown.get(tier, 0) + 1

        # V3: Check if Conditional manager (uses reduced weight)
        is_conditional = _is_conditional_manager(holder)
        if is_conditional:
            holder_weight = SMART_MONEY_CONDITIONAL_WEIGHT
            conditional_overlap += holder_weight
        else:
            holder_weight = _get_holder_weight(tier)
            elite_core_overlap += holder_weight

        weighted_overlap += holder_weight

        if tier == 1:
            tier1_holders.append(holder)

        # Track base contribution (before change adjustment)
        per_holder_contributions[holder] = holder_weight

    # Apply saturating function to weighted overlap
    overlap_bonus = _saturating_bonus(weighted_overlap)

    # =========================================================================
    # STEP 2: Compute tier-weighted position changes (direction signal)
    # V3: Separate Elite Core vs Conditional change contributions
    # =========================================================================
    change_bonus = Decimal("0")
    elite_core_change = Decimal("0")
    conditional_change = Decimal("0")
    holders_increasing: List[str] = []
    holders_decreasing: List[str] = []

    if position_changes:
        # DETERMINISM: Sort keys for consistent iteration order
        for holder in sorted(position_changes.keys()):
            change = position_changes[holder]
            change_upper = change.upper() if isinstance(change, str) else "HOLD"

            # V3: Check if Conditional manager
            is_conditional = _is_conditional_manager(holder)

            # Get tier-based weight (or Conditional weight)
            if is_conditional:
                tier_weight = SMART_MONEY_CONDITIONAL_WEIGHT
            else:
                tier = _get_holder_tier(holder, holder_tiers)
                tier_weight = _get_holder_weight(tier)

            # Get change weight (using V2 weights with reduced EXIT)
            base_change_weight = SMART_MONEY_V2_CHANGE_WEIGHTS.get(change_upper, Decimal("0"))

            # Compute holder's contribution: tier_weight * change_weight
            holder_contribution = tier_weight * base_change_weight

            # Apply per-holder cap to prevent one noisy filing from dominating
            holder_contribution = _clamp(
                holder_contribution,
                -SMART_MONEY_PER_HOLDER_CAP,
                SMART_MONEY_PER_HOLDER_CAP
            )

            # V3: Track by category
            if is_conditional:
                conditional_change += holder_contribution
            else:
                elite_core_change += holder_contribution

            change_bonus += holder_contribution

            # Track contribution for diagnostics
            if holder in per_holder_contributions:
                per_holder_contributions[holder] += holder_contribution
            else:
                per_holder_contributions[holder] = holder_contribution

            # Track direction
            if change_upper in ("NEW", "INCREASE"):
                holders_increasing.append(holder)
            elif change_upper in ("DECREASE", "EXIT"):
                holders_decreasing.append(holder)

    # Clamp total change bonus
    change_bonus = _clamp(
        change_bonus,
        SMART_MONEY_CHANGE_MIN_PENALTY,
        SMART_MONEY_CHANGE_MAX_BONUS
    )

    # =========================================================================
    # STEP 3: Compute Elite Core vs Conditional contributions
    # V3: Apply 30% cap to Conditional signal
    # =========================================================================
    elite_core_contribution = elite_core_overlap + elite_core_change
    conditional_raw_contribution = conditional_overlap + conditional_change

    # Apply Conditional cap: max 30% of total signal above base
    # If Conditional would exceed 30%, cap it and recalculate
    total_raw = elite_core_contribution + conditional_raw_contribution
    conditional_capped = False

    if total_raw > Decimal("0"):
        max_conditional = elite_core_contribution * (
            SMART_MONEY_CONDITIONAL_SIGNAL_CAP / (Decimal("1") - SMART_MONEY_CONDITIONAL_SIGNAL_CAP)
        )
        if conditional_raw_contribution > max_conditional:
            conditional_contribution = max_conditional
            conditional_capped = True
        else:
            conditional_contribution = conditional_raw_contribution
    else:
        conditional_contribution = conditional_raw_contribution

    # Recompute total signal with capped Conditional
    total_signal = elite_core_contribution + conditional_contribution

    # =========================================================================
    # STEP 4: Compute final score
    # =========================================================================
    smart_money_score = base_signal + total_signal
    smart_money_score = _clamp(smart_money_score, Decimal("20"), Decimal("80"))

    # =========================================================================
    # STEP 5: Compute confidence based on tier coverage
    # =========================================================================
    # Confidence scales with:
    # - Number of Tier1 holders (highest signal quality)
    # - Presence of position change data
    # - Overall weighted overlap
    # V3: Boost confidence for Elite Core coverage

    num_tier1 = tier_breakdown.get(1, 0)
    num_tier2 = tier_breakdown.get(2, 0)
    has_changes = bool(position_changes)

    if num_tier1 >= 2:
        confidence = Decimal("0.8")  # Strong: multiple top-tier holders
    elif num_tier1 >= 1 and num_tier2 >= 1:
        confidence = Decimal("0.7")  # Good: Tier1 + Tier2 coverage
    elif num_tier1 >= 1:
        confidence = Decimal("0.6")  # Moderate: single Tier1
    elif weighted_overlap >= Decimal("1.0"):
        confidence = Decimal("0.5")  # Some signal: meaningful weighted overlap
    elif overlap_count >= 2:
        confidence = Decimal("0.4")  # Weak: multiple unknowns
    elif overlap_count >= 1:
        confidence = Decimal("0.3")  # Very weak: single unknown
    else:
        confidence = Decimal("0.2")  # No signal

    # Boost confidence if we have change data
    if has_changes and overlap_count >= 2:
        confidence = _clamp(confidence + Decimal("0.1"), Decimal("0"), Decimal("0.9"))

    # V3: Slight confidence boost if signal is Elite Core dominated (not capped)
    if elite_core_contribution > Decimal("0") and not conditional_capped:
        confidence = _clamp(confidence + Decimal("0.05"), Decimal("0"), Decimal("0.95"))

    return SmartMoneySignal(
        smart_money_score=_quantize_score(smart_money_score),
        overlap_count=overlap_count,
        overlap_bonus=_quantize_score(overlap_bonus),
        position_change_adjustment=_quantize_score(change_bonus),
        holders_increasing=sorted(holders_increasing),
        holders_decreasing=sorted(holders_decreasing),
        confidence=confidence,
        # V2 fields
        weighted_overlap=_quantize_score(weighted_overlap),
        tier_breakdown=tier_breakdown,
        tier1_holders=sorted(tier1_holders),
        per_holder_contributions={k: _quantize_score(v) for k, v in sorted(per_holder_contributions.items())},
        # V3 fields - Elite Core vs Conditional separation
        elite_core_contribution=_quantize_score(elite_core_contribution),
        conditional_contribution=_quantize_score(conditional_contribution),
        conditional_raw_contribution=_quantize_score(conditional_raw_contribution),
        conditional_capped=conditional_capped,
    )


# =============================================================================
# NON-LINEAR INTERACTION TERMS
# =============================================================================

def compute_interaction_terms(
    clinical_normalized: Decimal,
    financial_data: Dict[str, Any],
    catalyst_normalized: Decimal,
    stage_bucket: str,
    vol_adjustment: Optional[VolatilityAdjustment] = None,
    *,
    runway_gate_status: str = "UNKNOWN",
    dilution_gate_status: str = "UNKNOWN",
    competitive_crowding: str = "unknown",
    partnership_strength: str = "unknown",
    cash_burn_trajectory: str = "unknown",
    cash_burn_risk: str = "unknown",
    phase_momentum: str = "unknown",
    phase_momentum_confidence: Decimal = Decimal("0"),
    phase_momentum_lead_phase: str = "unknown",
    days_to_nearest_catalyst: Optional[int] = None,
) -> InteractionTerms:
    """
    Compute non-linear interaction terms between signals.

    Uses SMOOTH RAMPS (piecewise linear) instead of hard thresholds to
    avoid rank churn from discontinuities. All adjustments are bounded
    to ±2 points to prevent leaderboard rewrites.

    SCALE CONTRACTS (enforced):
    - clinical_normalized: Decimal in [0, 100]
    - catalyst_normalized: Decimal in [0, 100]
    - runway_months: Decimal, raw months (not normalized)
    - vol_adjustment.annualized_vol: Decimal in [0, 1] or percentage

    DOUBLE-COUNTING PREVENTION:
    - If runway_gate_status == "FAIL", distress penalty is reduced by 50%
      (v2 gate already penalized this)
    - Synergy bonus only applies if runway_gate_status == "PASS"

    Args:
        clinical_normalized: Normalized clinical score (0-100)
        financial_data: Dict with runway_months, financial_score, etc.
        catalyst_normalized: Normalized catalyst score (0-100)
        stage_bucket: Development stage (early, mid, late)
        vol_adjustment: Optional volatility adjustment result
        runway_gate_status: "PASS", "FAIL", or "UNKNOWN" from v2 gates
        dilution_gate_status: "PASS", "FAIL", or "UNKNOWN" from v2 gates

    Returns:
        InteractionTerms with all computed interactions (bounded ±2)
    """
    flags = []

    # Enforce scale contracts with clamping
    clinical_norm = _clamp(_to_decimal(clinical_normalized, Decimal("50")), Decimal("0"), Decimal("100"))
    catalyst_norm = _clamp(_to_decimal(catalyst_normalized, Decimal("50")), Decimal("0"), Decimal("100"))

    # Extract financial metrics
    runway_months = _to_decimal(financial_data.get("runway_months"), Decimal("24"))
    runway_months = _clamp(runway_months, Decimal("0"), Decimal("120"))  # Cap at 10 years

    # Track if gates already applied
    runway_gate_applied = runway_gate_status == "FAIL"
    dilution_gate_applied = dilution_gate_status == "FAIL"

    # =========================================================================
    # 1. Clinical x Financial SYNERGY (smooth ramp)
    # =========================================================================
    # Bonus ramps from 0 to max as clinical goes 60→80 AND runway goes 12→24
    # Only applies if runway gate passed (to avoid rewarding already-gated names)

    clinical_financial_synergy = Decimal("0")

    if not runway_gate_applied:  # Don't give synergy if already gated
        # Compute clinical contribution: 0 at 60, 1.0 at 80
        clinical_factor = _smooth_ramp(
            clinical_norm,
            INTERACTION_SYNERGY_CLINICAL_LOW,
            INTERACTION_SYNERGY_CLINICAL_HIGH,
        )

        # Compute runway contribution: 0 at 12mo, 1.0 at 24mo
        runway_factor = _smooth_ramp(
            runway_months,
            INTERACTION_SYNERGY_RUNWAY_LOW,
            INTERACTION_SYNERGY_RUNWAY_HIGH,
        )

        # Multiplicative: both must be strong for full bonus
        synergy_factor = clinical_factor * runway_factor
        clinical_financial_synergy = synergy_factor * INTERACTION_SYNERGY_MAX_BONUS

        if clinical_financial_synergy >= Decimal("0.5"):
            flags.append("clinical_financial_synergy")

    # =========================================================================
    # 2. Stage x Financial DISTRESS (smooth ramp)
    # =========================================================================
    # Penalty ramps from 0 at runway=12 to max at runway=6
    # Only applies to late/mid stage
    # If runway gate already failed, reduce penalty by 50% (already penalized)

    stage_financial_interaction = Decimal("0")

    if stage_bucket in ("late", "mid"):
        # Compute distress factor: 0 at 12mo, 1.0 at 6mo (inverted ramp)
        distress_factor = _smooth_ramp_inverted(
            runway_months,
            INTERACTION_DISTRESS_RUNWAY_LOW,
            INTERACTION_DISTRESS_RUNWAY_HIGH,
        )

        # Stage multiplier: late = 1.0, mid = 0.5
        stage_mult = Decimal("1.0") if stage_bucket == "late" else Decimal("0.5")

        # Double-counting adjustment: if gate already failed, halve the penalty
        gate_mult = Decimal("0.5") if runway_gate_applied else Decimal("1.0")

        stage_financial_interaction = -(
            distress_factor * stage_mult * gate_mult * INTERACTION_DISTRESS_MAX_PENALTY
        )

        if stage_financial_interaction <= Decimal("-0.5"):
            flags.append("late_stage_distress" if stage_bucket == "late" else "mid_stage_runway_warning")

    # =========================================================================
    # 3. Catalyst x Volatility DAMPENING (multiplicative, not additive)
    # =========================================================================
    # In high-vol names, extreme catalyst signals are less reliable
    # We return the dampening factor for informational purposes
    # The actual dampening is applied in the main scoring function

    catalyst_volatility_dampening = Decimal("0")

    if vol_adjustment and vol_adjustment.vol_bucket == VolatilityBucket.HIGH:
        # Compute how far catalyst is from neutral (50)
        catalyst_excess = abs(catalyst_norm - Decimal("50"))

        # Dampening increases with catalyst extremity
        # At |excess| = 0: no dampening
        # At |excess| = 50: max dampening (30%)
        dampening_pct = (catalyst_excess / Decimal("50")) * Decimal("0.30")
        dampening_pct = _clamp(dampening_pct, Decimal("0"), Decimal("0.30"))

        # Store as the score-point reduction (informational)
        # This represents how much the catalyst signal is discounted
        catalyst_volatility_dampening = dampening_pct * Decimal("10")  # Scale to ~0-3 points

        if dampening_pct >= Decimal("0.10"):
            flags.append("catalyst_vol_dampening")

    # =========================================================================
    # 4. Competitive Intensity x Partnership Interaction (bounded ±3)
    # =========================================================================
    # Rule 1: Low/moderate competition + strong/exceptional partnership → +2.5
    # Rule 2: Intense competition + exceptional partnership → +1.0 (validated but crowded)
    # Rule 3: Intense competition + no/weak partnership → -1.5 (me-too economics risk)

    competitive_partnership_interaction = Decimal("0")

    # Normalize CI labels (handle potential aliases from different sources)
    ci_normalized = competitive_crowding.lower() if competitive_crowding else "unknown"
    ci_alias_map = {
        "low": "uncrowded",
        "high": "crowded",
        "intense": "highly_crowded",
    }
    ci_normalized = ci_alias_map.get(ci_normalized, ci_normalized)

    # Normalize partnership labels
    ps_normalized = partnership_strength.lower() if partnership_strength else "unknown"

    ci_low = ci_normalized in ("uncrowded", "moderate")
    ci_intense = ci_normalized in ("highly_crowded", "crowded")
    ps_strong = ps_normalized in ("strong", "exceptional")
    ps_exceptional = ps_normalized == "exceptional"
    ps_weak = ps_normalized in ("unknown", "weak", "none", "")

    if ci_low and ps_strong:
        # Best combo: differentiated position + external validation
        competitive_partnership_interaction = Decimal("2.5")
        flags.append("validated_uncrowded_boost")
    elif ci_intense and ps_exceptional:
        # Validated but competitive - small boost
        competitive_partnership_interaction = Decimal("1.0")
        flags.append("validated_crowded_soften")
    elif ci_intense and ps_weak:
        # Crowded with no validation - penalty
        competitive_partnership_interaction = Decimal("-1.5")
        flags.append("unvalidated_crowded_penalty")

    # Clamp CI+partnership interaction to ±3
    competitive_partnership_interaction = _clamp(
        competitive_partnership_interaction, Decimal("-3.0"), Decimal("3.0")
    )

    # =========================================================================
    # 5. Cash Burn Trajectory Interaction (bounded ±1.5)
    # =========================================================================
    # Decelerating burn (financial discipline) → positive modifier
    # Accelerating burn + high risk → negative modifier
    # Only apply when confidence is sufficient

    cash_burn_interaction = Decimal("0")
    burn_traj = cash_burn_trajectory.lower() if cash_burn_trajectory else "unknown"
    burn_risk = cash_burn_risk.lower() if cash_burn_risk else "unknown"

    if burn_traj != "unknown":
        if burn_traj == "decelerating":
            # Financial discipline - reward
            cash_burn_interaction = Decimal("1.0")
            flags.append("burn_discipline_bonus")
        elif burn_traj == "accelerating" and burn_risk in ("high", "critical"):
            # Accelerating burn + short runway - penalty
            cash_burn_interaction = Decimal("-1.5")
            flags.append("burn_acceleration_penalty")
        elif burn_traj == "accelerating_justified":
            # Phase 3 justified - small penalty (already gated by risk)
            if burn_risk == "critical":
                cash_burn_interaction = Decimal("-0.75")
                flags.append("burn_justified_but_critical")

    cash_burn_interaction = _clamp(cash_burn_interaction, Decimal("-1.5"), Decimal("1.5"))

    # =========================================================================
    # 6. Phase Momentum Interaction (bounded ±2.5 with double jeopardy)
    # =========================================================================
    # Governance rules:
    # - Rule A: Skip penalty for approved/commercial (lead_phase in {approved, marketed})
    # - Rule B: Double jeopardy adds -1.0 if strong_neg AND funding pressure AND no catalyst
    # - Rule C: Confidence ramp instead of hard threshold (multiplier = (conf-0.5)/0.5)
    # - Commercial boost cap: max +0.5 for commercial stage (momentum less meaningful)

    phase_momentum_interaction = Decimal("0")
    pm_normalized = phase_momentum.lower() if phase_momentum else "unknown"
    pm_conf = _to_decimal(phase_momentum_confidence, Decimal("0"))

    # Rule C: Confidence ramp (0 at conf=0.5, 1.0 at conf=1.0)
    conf_multiplier = _clamp((pm_conf - Decimal("0.5")) / Decimal("0.5"), Decimal("0"), Decimal("1"))

    # Rule A: Check if approved/commercial stage using lead_phase string (not numeric)
    # This avoids magic number issues if phase mapper changes
    lead_phase_lower = phase_momentum_lead_phase.lower() if phase_momentum_lead_phase else "unknown"
    COMMERCIAL_PHASES = {"approved", "marketed", "phase 4", "commercial"}
    is_commercial_stage = lead_phase_lower in COMMERCIAL_PHASES

    # Extract runway_months from financial_data for Rule B
    runway_months_local = _to_decimal(financial_data.get("runway_months"), Decimal("0"))

    if pm_normalized != "unknown" and conf_multiplier > Decimal("0"):
        if pm_normalized == "strong_positive":
            # Boost applies to all stages, but capped for commercial
            if is_commercial_stage:
                # Cap boost at +0.5 for commercial (momentum less meaningful when approved)
                phase_momentum_interaction = Decimal("0.5") * conf_multiplier
                flags.append("phase_momentum_boost_capped_commercial")
            else:
                phase_momentum_interaction = Decimal("1.5") * conf_multiplier
                flags.append("phase_momentum_boost")

        elif pm_normalized == "positive":
            if is_commercial_stage:
                phase_momentum_interaction = Decimal("0.25") * conf_multiplier
                flags.append("phase_momentum_positive_capped_commercial")
            else:
                phase_momentum_interaction = Decimal("0.75") * conf_multiplier
                flags.append("phase_momentum_positive")

        elif pm_normalized == "negative":
            # Rule A: Skip penalty for commercial stage
            if not is_commercial_stage:
                phase_momentum_interaction = Decimal("-0.75") * conf_multiplier
                flags.append("phase_momentum_negative")
            else:
                flags.append("phase_momentum_negative_skipped_commercial")

        elif pm_normalized == "strong_negative":
            # Rule A: Skip penalty for commercial stage
            if not is_commercial_stage:
                phase_momentum_interaction = Decimal("-1.5") * conf_multiplier
                flags.append("phase_momentum_penalty")

                # Rule B: Double jeopardy for development-stage stagnation
                # Conditions: strong_neg AND (high/critical burn OR short runway) AND no near-term catalyst
                burn_risk_lower = cash_burn_risk.lower() if cash_burn_risk else "unknown"
                has_funding_pressure = (
                    burn_risk_lower in ("high", "critical")
                    or runway_months_local < Decimal("12")
                )
                has_near_catalyst = days_to_nearest_catalyst is not None and days_to_nearest_catalyst <= 90

                if has_funding_pressure and not has_near_catalyst:
                    phase_momentum_interaction -= Decimal("1.0") * conf_multiplier
                    flags.append("phase_momentum_double_jeopardy")
            else:
                flags.append("phase_momentum_penalty_skipped_commercial")

    phase_momentum_interaction = _clamp(phase_momentum_interaction, Decimal("-2.5"), Decimal("1.5"))

    # =========================================================================
    # Total with bounds
    # =========================================================================
    # Clamp total to prevent extreme adjustments
    # CI+partnership: ±3, burn: ±1.5, momentum: ±1.5, synergy: +1.5, distress: -2
    # Max theoretical: +6.5 to -8.0, but clamped to ±3
    total = (
        clinical_financial_synergy
        + stage_financial_interaction
        + competitive_partnership_interaction
        + cash_burn_interaction
        + phase_momentum_interaction
    )
    # Note: catalyst dampening is informational, not added to total
    # (it's applied as a multiplier in the main function)

    total = _clamp(total, Decimal("-3.0"), Decimal("3.0"))

    return InteractionTerms(
        clinical_financial_synergy=_quantize_score(clinical_financial_synergy),
        stage_financial_interaction=_quantize_score(stage_financial_interaction),
        catalyst_volatility_dampening=_quantize_score(catalyst_volatility_dampening),
        competitive_partnership_interaction=_quantize_score(competitive_partnership_interaction),
        cash_burn_interaction=_quantize_score(cash_burn_interaction),
        phase_momentum_interaction=_quantize_score(phase_momentum_interaction),
        total_interaction_adjustment=_quantize_score(total),
        interaction_flags=flags,
        runway_gate_already_applied=runway_gate_applied,
        dilution_gate_already_applied=dilution_gate_applied,
    )


def _smooth_ramp(value: Decimal, low: Decimal, high: Decimal) -> Decimal:
    """
    Compute smooth ramp from 0 at low to 1 at high.

    Returns:
        0 if value <= low
        1 if value >= high
        Linear interpolation between
    """
    if value <= low:
        return Decimal("0")
    if value >= high:
        return Decimal("1")
    return (value - low) / (high - low)


def _smooth_ramp_inverted(value: Decimal, low: Decimal, high: Decimal) -> Decimal:
    """
    Compute inverted smooth ramp from 1 at low to 0 at high.

    Returns:
        1 if value <= low
        0 if value >= high
        Linear interpolation between
    """
    if value <= low:
        return Decimal("1")
    if value >= high:
        return Decimal("0")
    return (high - value) / (high - low)


# =============================================================================
# SHRINKAGE NORMALIZATION
# =============================================================================

def shrinkage_normalize(
    values: List[Decimal],
    global_mean: Decimal,
    global_std: Decimal,
    *,
    prior_strength: Decimal = SHRINKAGE_PRIOR_STRENGTH,
) -> Tuple[List[Decimal], Decimal]:
    """
    Bayesian shrinkage normalization for small cohorts.

    Shrinks cohort statistics toward global distribution to reduce
    noise from small sample sizes. Larger cohorts get less shrinkage.

    Args:
        values: List of raw scores to normalize
        global_mean: Global (across all cohorts) mean
        global_std: Global standard deviation
        prior_strength: Equivalent sample size of prior (higher = more shrinkage)

    Returns:
        Tuple of (normalized_values, shrinkage_factor_used)
    """
    n = len(values)
    if n == 0:
        return [], Decimal("0")

    if n == 1:
        return [Decimal("50")], Decimal("1.0")

    # Compute cohort statistics
    cohort_mean = sum(values) / Decimal(n)

    # Shrinkage factor: prior_strength / (prior_strength + n)
    # Small cohort -> high shrinkage toward global
    # Large cohort -> low shrinkage (use cohort stats)
    shrinkage_factor = prior_strength / (prior_strength + Decimal(n))

    # Shrink mean toward global
    adjusted_mean = (
        cohort_mean * (Decimal("1") - shrinkage_factor) +
        global_mean * shrinkage_factor
    )

    # Shrink std toward global with SAME weight as mean
    # This is critical: small cohorts have unstable std estimates
    # Using same shrinkage weight stabilizes percentile ranks significantly
    cohort_variance = sum((v - cohort_mean) ** 2 for v in values) / Decimal(n)
    cohort_std = cohort_variance.sqrt() if cohort_variance > 0 else Decimal("1")
    adjusted_std = (
        cohort_std * (Decimal("1") - shrinkage_factor) +
        global_std * shrinkage_factor
    )
    adjusted_std = max(adjusted_std, Decimal("0.01"))  # Prevent division by zero

    # Normalize with shrunk parameters and convert to percentile-like scale
    result = []
    for v in values:
        z = (v - adjusted_mean) / adjusted_std
        # Clamp z-score to [-3, +3] before percentile conversion
        # This prevents extreme tail values from distorting ranks
        z = _clamp(z, Decimal("-3"), Decimal("3"))
        # Convert z-score to 0-100 scale (z=0 -> 50, z=±2 -> ~10/90)
        percentile = Decimal("50") + z * Decimal("15")
        percentile = _clamp(percentile, Decimal("5"), Decimal("95"))
        result.append(_quantize_score(percentile))

    return result, _quantize_weight(shrinkage_factor)


# =============================================================================
# REGIME-ADAPTIVE COMPONENTS
# =============================================================================

# Regime-specific signal importance multipliers
REGIME_SIGNAL_IMPORTANCE: Dict[RegimeType, RegimeSignalImportance] = {
    RegimeType.BULL: RegimeSignalImportance(
        clinical=Decimal("1.0"),
        financial=Decimal("0.8"),   # Less important in bull
        catalyst=Decimal("1.2"),    # Catalysts drive in bull
        momentum=Decimal("1.3"),    # Momentum works well
        valuation=Decimal("0.7"),   # Valuation less important
        smart_money=Decimal("1.0"),
    ),
    RegimeType.BEAR: RegimeSignalImportance(
        clinical=Decimal("1.0"),
        financial=Decimal("1.4"),   # Cash is king
        catalyst=Decimal("0.7"),    # Catalysts less reliable
        momentum=Decimal("0.5"),    # Momentum reverses
        valuation=Decimal("1.2"),   # Value matters more
        smart_money=Decimal("1.1"), # Follow smart money
    ),
    RegimeType.NEUTRAL: RegimeSignalImportance(
        clinical=Decimal("1.0"),
        financial=Decimal("1.0"),
        catalyst=Decimal("1.0"),
        momentum=Decimal("1.0"),
        valuation=Decimal("1.0"),
        smart_money=Decimal("1.0"),
    ),
    RegimeType.UNKNOWN: RegimeSignalImportance(
        clinical=Decimal("1.0"),
        financial=Decimal("1.0"),
        catalyst=Decimal("1.0"),
        momentum=Decimal("0.8"),    # Reduce momentum when uncertain
        valuation=Decimal("1.0"),
        smart_money=Decimal("1.0"),
    ),
}


def get_regime_signal_importance(regime: str) -> RegimeSignalImportance:
    """
    Get signal importance multipliers for given regime.

    Args:
        regime: Regime string (BULL, BEAR, NEUTRAL, UNKNOWN)

    Returns:
        RegimeSignalImportance with multipliers for each component
    """
    try:
        regime_type = RegimeType(regime.upper())
    except ValueError:
        regime_type = RegimeType.UNKNOWN

    return REGIME_SIGNAL_IMPORTANCE[regime_type]


def apply_regime_to_weights(
    base_weights: Dict[str, Decimal],
    regime: str,
) -> Dict[str, Decimal]:
    """
    Apply regime-specific adjustments to component weights.

    Multiplies base weights by regime importance factors and renormalizes.

    Args:
        base_weights: Dict of component -> weight (should sum to 1.0)
        regime: Current market regime

    Returns:
        Adjusted weights (sum to 1.0)
    """
    importance = get_regime_signal_importance(regime)

    # Map weight keys to importance attributes
    importance_map = {
        "clinical": importance.clinical,
        "clinical_dev": importance.clinical,
        "financial": importance.financial,
        "catalyst": importance.catalyst,
        "momentum": importance.momentum,
        "valuation": importance.valuation,
        "smart_money": importance.smart_money,
        "pos": Decimal("1.0"),  # PoS not regime-adjusted
    }

    # Apply multipliers
    adjusted = {}
    for key, weight in base_weights.items():
        multiplier = importance_map.get(key, Decimal("1.0"))
        adjusted[key] = weight * multiplier

    # Renormalize to sum to 1.0
    total = sum(adjusted.values())
    if total > EPS:
        adjusted = {k: _quantize_weight(v / total) for k, v in adjusted.items()}

    return adjusted


# =============================================================================
# ADAPTIVE WEIGHT LEARNING (PIT-SAFE)
# =============================================================================

def compute_adaptive_weights(
    historical_scores: List[Dict[str, Any]],
    forward_returns: Dict[Tuple[date, str], Decimal],
    base_weights: Dict[str, Decimal],
    *,
    asof_date: date,
    lookback_months: int = 12,
    embargo_months: int = 1,
    min_weight: Decimal = Decimal("0.02"),
    max_weight: Decimal = Decimal("0.60"),
    shrinkage_lambda: Decimal = Decimal("0.70"),
    smooth_gamma: Decimal = Decimal("0.80"),
    prev_weights: Optional[Dict[str, Decimal]] = None,
) -> AdaptiveWeights:
    """
    Compute weights that maximize historical IC using PIT-safe rank correlation.

    CRITICAL PIT SAFETY:
    - forward_returns is keyed by (asof_date, ticker), NOT just ticker
    - For each historical as_of_date, we only use returns that were realized
      AFTER the embargo period (asof_date + embargo_months)
    - This prevents look-ahead bias in weight optimization

    The function computes per-period cross-sectional IC for each component,
    aggregates across periods, then applies:
    1. Shrinkage toward base_weights (controlled by shrinkage_lambda)
    2. Smoothing toward prev_weights (controlled by smooth_gamma)

    Args:
        historical_scores: List of dicts with ticker, component scores, as_of_date
            Each dict must have 'as_of_date' field (date or ISO string)
        forward_returns: Dict keyed by (as_of_date, ticker) -> forward return
            The as_of_date in the key is when the return period STARTS
        base_weights: Prior weights to shrink toward
        asof_date: Current date for PIT cutoff
        lookback_months: Months of history to use for IC estimation
        embargo_months: Minimum months between score date and return measurement
        min_weight: Minimum weight per component
        max_weight: Maximum weight per component
        shrinkage_lambda: 0-1, higher = more shrinkage toward base_weights
        smooth_gamma: 0-1, higher = more smoothing toward prev_weights
        prev_weights: Previous period's weights for smoothing

    Returns:
        AdaptiveWeights with optimized weights and diagnostics
    """
    if not historical_scores or not forward_returns:
        return AdaptiveWeights(
            weights=base_weights.copy(),
            historical_ic_by_component={},
            optimization_method="fallback_no_data",
            lookback_months=lookback_months,
            confidence=Decimal("0.1"),
            embargo_months=embargo_months,
            training_periods=0,
        )

    # Compute lookback cutoff
    lookback_cutoff = asof_date - timedelta(days=lookback_months * 30)
    embargo_cutoff = asof_date - timedelta(days=embargo_months * 30)

    # Group historical scores by as_of_date
    scores_by_date: Dict[date, List[Dict[str, Any]]] = {}
    for rec in historical_scores:
        rec_date = _parse_date(rec.get("as_of_date"))
        if rec_date is None:
            continue
        # Only use dates within lookback window AND before embargo cutoff
        if rec_date < lookback_cutoff:
            continue
        if rec_date > embargo_cutoff:
            # Too recent - returns not yet realized
            continue
        if rec_date not in scores_by_date:
            scores_by_date[rec_date] = []
        scores_by_date[rec_date].append(rec)

    if not scores_by_date:
        return AdaptiveWeights(
            weights=base_weights.copy(),
            historical_ic_by_component={},
            optimization_method="fallback_no_valid_periods",
            lookback_months=lookback_months,
            confidence=Decimal("0.1"),
            embargo_months=embargo_months,
            training_periods=0,
        )

    # Extract component names from first record
    sample = next(iter(next(iter(scores_by_date.values()))))
    component_names = [k for k in base_weights.keys() if k in sample or f"{k}_normalized" in sample]

    if not component_names:
        return AdaptiveWeights(
            weights=base_weights.copy(),
            historical_ic_by_component={},
            optimization_method="fallback_no_components",
            lookback_months=lookback_months,
            confidence=Decimal("0.1"),
            embargo_months=embargo_months,
            training_periods=0,
        )

    # Compute per-period IC for each component
    # This is the CRITICAL PIT-safe loop
    ic_by_component_by_date: Dict[str, List[Decimal]] = {comp: [] for comp in component_names}

    for score_date, records in sorted(scores_by_date.items()):
        # Get all tickers that have both scores and returns for this date
        tickers_with_returns = [
            rec.get("ticker") for rec in records
            if (score_date, rec.get("ticker")) in forward_returns
        ]

        if len(tickers_with_returns) < 10:
            # Not enough cross-section for reliable IC
            continue

        # For each component, compute cross-sectional IC for this date
        for comp in component_names:
            pairs = []
            for rec in records:
                ticker = rec.get("ticker")
                ret_key = (score_date, ticker)
                if ret_key not in forward_returns:
                    continue

                score = _to_decimal(rec.get(f"{comp}_normalized") or rec.get(comp))
                if score is None:
                    continue

                ret = forward_returns[ret_key]
                pairs.append((ticker, score, ret))

            if len(pairs) < 10:
                continue

            # Compute rank correlation with deterministic tie-breaking
            ic = _compute_rank_correlation_with_tiebreak(pairs)
            ic_by_component_by_date[comp].append(ic)

    # Aggregate ICs across periods (simple mean)
    ic_by_component: Dict[str, Decimal] = {}
    for comp, ics in ic_by_component_by_date.items():
        if ics:
            ic_by_component[comp] = sum(ics) / Decimal(len(ics))
        else:
            ic_by_component[comp] = Decimal("0")

    training_periods = max(len(ics) for ics in ic_by_component_by_date.values()) if ic_by_component_by_date else 0

    if training_periods < 3:
        # Not enough periods for reliable estimation
        return AdaptiveWeights(
            weights=base_weights.copy(),
            historical_ic_by_component={k: str(v) for k, v in ic_by_component.items()},
            optimization_method="fallback_insufficient_periods",
            lookback_months=lookback_months,
            confidence=Decimal("0.15"),
            embargo_months=embargo_months,
            training_periods=training_periods,
        )

    # Compute raw IC-based weights
    # Higher IC -> higher weight
    raw_weights: Dict[str, Decimal] = {}
    total_positive_ic = sum(max(ic, Decimal("0")) for ic in ic_by_component.values())

    if total_positive_ic <= EPS:
        # No positive IC found
        raw_weights = base_weights.copy()
    else:
        for comp, ic in ic_by_component.items():
            base = base_weights.get(comp, Decimal("0.20"))
            # Scale adjustment by IC
            ic_adjustment = _clamp(ic * Decimal("2"), Decimal("-0.5"), Decimal("0.5"))
            adjusted = base * (Decimal("1") + ic_adjustment)
            adjusted = _clamp(adjusted, min_weight, max_weight)
            raw_weights[comp] = adjusted

    # Step 1: Shrinkage toward base_weights
    # shrunk_w = (1 - lambda) * raw_w + lambda * base_w
    shrunk_weights: Dict[str, Decimal] = {}
    for comp in component_names:
        raw_w = raw_weights.get(comp, base_weights.get(comp, Decimal("0.20")))
        base_w = base_weights.get(comp, Decimal("0.20"))
        shrunk = (Decimal("1") - shrinkage_lambda) * raw_w + shrinkage_lambda * base_w
        shrunk_weights[comp] = shrunk

    # Step 2: Smoothing toward prev_weights (if available)
    # smoothed_w = (1 - gamma) * shrunk_w + gamma * prev_w
    if prev_weights:
        smoothed_weights: Dict[str, Decimal] = {}
        for comp in component_names:
            shrunk_w = shrunk_weights.get(comp, Decimal("0.20"))
            prev_w = prev_weights.get(comp, shrunk_w)
            smoothed = (Decimal("1") - smooth_gamma) * shrunk_w + smooth_gamma * prev_w
            smoothed_weights[comp] = smoothed
        final_weights = smoothed_weights
        smoothing_applied = smooth_gamma
    else:
        final_weights = shrunk_weights
        smoothing_applied = Decimal("0")

    # Renormalize to sum to 1.0
    total = sum(final_weights.values())
    if total > EPS:
        final_weights = {k: _quantize_weight(v / total) for k, v in final_weights.items()}

    # Ensure we have all components from base_weights
    for comp in base_weights:
        if comp not in final_weights:
            final_weights[comp] = base_weights[comp]

    # Re-normalize after adding missing components
    total = sum(final_weights.values())
    if total > EPS:
        final_weights = {k: _quantize_weight(v / total) for k, v in final_weights.items()}

    # Compute confidence based on sample size and period count
    if training_periods >= 12:
        confidence = Decimal("0.7")
    elif training_periods >= 6:
        confidence = Decimal("0.5")
    else:
        confidence = Decimal("0.3")

    return AdaptiveWeights(
        weights=final_weights,
        historical_ic_by_component={k: str(v) for k, v in ic_by_component.items()},
        optimization_method="pit_safe_rank_correlation",
        lookback_months=lookback_months,
        confidence=confidence,
        embargo_months=embargo_months,
        training_periods=training_periods,
        shrinkage_applied=shrinkage_lambda,
        smoothing_applied=smoothing_applied,
    )


def _parse_date(value: Any) -> Optional[date]:
    """Parse date from various formats."""
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _compute_rank_correlation_with_tiebreak(
    pairs: List[Tuple[str, Decimal, Decimal]]
) -> Decimal:
    """
    Compute Spearman rank correlation with deterministic tie-breaking.

    Uses ticker as secondary sort key to ensure deterministic ranks when
    scores or returns are tied. This is critical for reproducibility.

    Args:
        pairs: List of (ticker, score, return) tuples

    Returns:
        Rank correlation coefficient (-1 to 1)
    """
    n = len(pairs)
    if n < 2:
        return Decimal("0")

    # Extract data preserving ticker for tiebreaking
    tickers = [p[0] for p in pairs]
    scores = [p[1] for p in pairs]
    returns = [p[2] for p in pairs]

    # Compute ranks with deterministic tiebreak
    score_ranks = _compute_ranks_deterministic(scores, tickers)
    return_ranks = _compute_ranks_deterministic(returns, tickers)

    # Compute Spearman correlation
    # rho = 1 - (6 * sum(d^2)) / (n * (n^2 - 1))
    d_squared_sum = sum(
        (sr - rr) ** 2
        for sr, rr in zip(score_ranks, return_ranks)
    )

    denominator = Decimal(n) * (Decimal(n) ** 2 - Decimal("1"))
    if denominator == 0:
        return Decimal("0")

    rho = Decimal("1") - (Decimal("6") * d_squared_sum) / denominator
    return _quantize_weight(rho)


def _compute_ranks_deterministic(
    values: List[Decimal],
    tiebreakers: List[str],
) -> List[Decimal]:
    """
    Compute ranks with deterministic tie-breaking using secondary key.

    When values are tied, uses tiebreakers (e.g., ticker) as secondary
    sort key to ensure consistent ranking across runs.

    Args:
        values: List of values to rank
        tiebreakers: List of secondary sort keys (same length as values)

    Returns:
        List of ranks (1-indexed)
    """
    n = len(values)
    if n == 0:
        return []

    # Create indexed list with tiebreaker
    indexed = [(v, tb, i) for i, (v, tb) in enumerate(zip(values, tiebreakers))]

    # Sort by value first, then by tiebreaker for determinism
    indexed.sort(key=lambda x: (x[0], x[1]))

    # Assign ranks (1-indexed, no ties due to tiebreaker)
    ranks = [Decimal("0")] * n
    for rank_idx, (_, _, orig_idx) in enumerate(indexed):
        ranks[orig_idx] = Decimal(rank_idx + 1)

    return ranks


def _compute_rank_correlation(pairs: List[Tuple[Decimal, Decimal]]) -> Decimal:
    """
    Compute Spearman rank correlation between score and return pairs.

    Pure Python implementation without scipy.
    NOTE: This uses average ranks for ties. For deterministic behavior,
    use _compute_rank_correlation_with_tiebreak instead.

    Args:
        pairs: List of (score, return) tuples

    Returns:
        Rank correlation coefficient (-1 to 1)
    """
    n = len(pairs)
    if n < 2:
        return Decimal("0")

    # Extract scores and returns
    scores = [p[0] for p in pairs]
    returns = [p[1] for p in pairs]

    # Compute ranks
    score_ranks = _compute_ranks(scores)
    return_ranks = _compute_ranks(returns)

    # Compute Spearman correlation
    # rho = 1 - (6 * sum(d^2)) / (n * (n^2 - 1))
    d_squared_sum = sum(
        (sr - rr) ** 2
        for sr, rr in zip(score_ranks, return_ranks)
    )

    denominator = Decimal(n) * (Decimal(n) ** 2 - Decimal("1"))
    if denominator == 0:
        return Decimal("0")

    rho = Decimal("1") - (Decimal("6") * d_squared_sum) / denominator
    return _quantize_weight(rho)


def _compute_ranks(values: List[Decimal]) -> List[Decimal]:
    """
    Compute ranks with average rank for ties.

    NOTE: This is non-deterministic when values are tied. Use
    _compute_ranks_deterministic for reproducible results.
    """
    n = len(values)
    indexed = [(v, i) for i, v in enumerate(values)]
    indexed.sort(key=lambda x: x[0])

    ranks = [Decimal("0")] * n
    i = 0
    while i < n:
        j = i
        while j < n and indexed[j][0] == indexed[i][0]:
            j += 1
        avg_rank = Decimal(str((i + j + 1) / 2))  # 1-indexed average
        for k in range(i, j):
            ranks[indexed[k][1]] = avg_rank
        i = j

    return ranks


# =============================================================================
# COMPOSITE ENHANCEMENT ORCHESTRATOR
# =============================================================================

@dataclass
class EnhancedScoringResult:
    """Complete enhanced scoring result for a single ticker."""
    ticker: str

    # Core scores (normalized)
    clinical_normalized: Decimal
    financial_normalized: Decimal
    catalyst_normalized: Decimal

    # Enhancement scores
    momentum_signal: MomentumSignal
    valuation_signal: ValuationSignal
    smart_money_signal: SmartMoneySignal

    # Adjustments
    vol_adjustment: VolatilityAdjustment
    catalyst_decay: CatalystDecayResult
    interaction_terms: InteractionTerms

    # Final composite
    base_composite: Decimal  # Before enhancements
    enhanced_composite: Decimal  # After all enhancements
    composite_delta: Decimal  # Enhancement contribution

    # Weights used
    effective_weights: Dict[str, Decimal]

    # Metadata
    enhancement_flags: List[str]
    confidence_overall: Decimal


def compute_enhanced_score(
    ticker: str,
    clinical_normalized: Decimal,
    financial_normalized: Decimal,
    catalyst_normalized: Decimal,
    financial_data: Dict[str, Any],
    stage_bucket: str,
    base_weights: Dict[str, Decimal],
    *,
    # Enhancement inputs (optional)
    return_60d: Optional[Decimal] = None,
    benchmark_return_60d: Optional[Decimal] = None,
    market_cap_mm: Optional[Decimal] = None,
    trial_count: int = 0,
    lead_phase: Optional[str] = None,
    peer_valuations: Optional[List[Dict]] = None,
    annualized_vol: Optional[Decimal] = None,
    days_to_catalyst: Optional[int] = None,
    catalyst_event_type: str = "DEFAULT",
    coinvest_overlap_count: int = 0,
    coinvest_holders: Optional[List[str]] = None,
    position_changes: Optional[Dict[str, str]] = None,
    regime: str = "NEUTRAL",
) -> EnhancedScoringResult:
    """
    Compute fully enhanced composite score with all IC improvements.

    Orchestrates all enhancement modules and combines into final score.

    Args:
        ticker: Security ticker
        clinical_normalized: Normalized clinical score (0-100)
        financial_normalized: Normalized financial score (0-100)
        catalyst_normalized: Normalized catalyst score (0-100)
        financial_data: Dict with runway_months, financial_score, etc.
        stage_bucket: Development stage
        base_weights: Starting component weights
        [... enhancement inputs ...]

    Returns:
        EnhancedScoringResult with complete scoring breakdown
    """
    flags = []

    # 1. Compute volatility adjustment
    vol_adj = compute_volatility_adjustment(annualized_vol)
    if vol_adj.vol_bucket == VolatilityBucket.HIGH:
        flags.append("high_volatility")
    elif vol_adj.vol_bucket == VolatilityBucket.LOW:
        flags.append("low_volatility")

    # 2. Compute momentum signal (with optional vol adjustment)
    momentum = compute_momentum_signal(
        return_60d,
        benchmark_return_60d,
        annualized_vol=annualized_vol,
        use_vol_adjusted_alpha=False,  # Default to raw alpha for now
    )
    if momentum.alpha_60d and abs(momentum.alpha_60d) >= Decimal("0.10"):
        flags.append("strong_momentum" if momentum.alpha_60d > 0 else "weak_momentum")
    # Flag low data completeness
    if momentum.data_completeness < Decimal("0.5"):
        flags.append("momentum_data_incomplete")

    # 3. Compute valuation signal
    valuation = compute_valuation_signal(
        market_cap_mm,
        trial_count,
        lead_phase,
        peer_valuations or []
    )
    if valuation.valuation_score >= Decimal("70"):
        flags.append("undervalued")
    elif valuation.valuation_score <= Decimal("30"):
        flags.append("overvalued")

    # 4. Compute catalyst decay
    decay = compute_catalyst_decay(days_to_catalyst, catalyst_event_type)
    if decay.in_optimal_window:
        flags.append("catalyst_optimal_window")

    # 5. Compute smart money signal
    smart_money = compute_smart_money_signal(
        coinvest_overlap_count,
        coinvest_holders or [],
        position_changes
    )
    if smart_money.holders_increasing:
        flags.append("smart_money_buying")
    if smart_money.holders_decreasing:
        flags.append("smart_money_selling")

    # 6. Compute interaction terms
    interactions = compute_interaction_terms(
        clinical_normalized,
        financial_data,
        catalyst_normalized,
        stage_bucket,
        vol_adj
    )
    flags.extend(interactions.interaction_flags)

    # 7. Apply regime to weights
    regime_weights = apply_regime_to_weights(base_weights, regime)

    # 8. Apply volatility adjustment to weights
    vol_adjusted_weights = {
        k: v * vol_adj.weight_adjustment_factor
        for k, v in regime_weights.items()
    }

    # Renormalize
    total = sum(vol_adjusted_weights.values())
    if total > EPS:
        effective_weights = {k: _quantize_weight(v / total) for k, v in vol_adjusted_weights.items()}
    else:
        effective_weights = regime_weights.copy()

    # 9. Compute base composite (without enhancements)
    base_composite = (
        clinical_normalized * effective_weights.get("clinical", effective_weights.get("clinical_dev", Decimal("0.40"))) +
        financial_normalized * effective_weights.get("financial", Decimal("0.35")) +
        catalyst_normalized * effective_weights.get("catalyst", Decimal("0.25"))
    )
    base_composite = _quantize_score(base_composite)

    # 10. Apply catalyst decay to catalyst component
    decayed_catalyst = apply_catalyst_decay(catalyst_normalized, decay)

    # 11. Compute enhanced composite
    # Include momentum, valuation, smart money with small weights
    enhancement_weight = Decimal("0.15")  # 15% total for enhancements
    core_weight = Decimal("1.0") - enhancement_weight

    # Core with decayed catalyst
    core_composite = (
        clinical_normalized * effective_weights.get("clinical", effective_weights.get("clinical_dev", Decimal("0.40"))) +
        financial_normalized * effective_weights.get("financial", Decimal("0.35")) +
        decayed_catalyst * effective_weights.get("catalyst", Decimal("0.25"))
    )

    # Enhancement contributions
    enhancement_composite = (
        momentum.momentum_score * Decimal("0.05") +
        valuation.valuation_score * Decimal("0.05") +
        smart_money.smart_money_score * Decimal("0.05")
    )

    # Combine with interaction adjustment
    enhanced_composite = (
        core_composite * core_weight +
        enhancement_composite +
        interactions.total_interaction_adjustment
    )

    # Apply volatility score adjustment
    enhanced_composite = apply_volatility_to_score(enhanced_composite, vol_adj)

    # Clamp to valid range
    enhanced_composite = _clamp(enhanced_composite, Decimal("0"), Decimal("100"))

    # Compute delta
    composite_delta = enhanced_composite - base_composite

    # Compute overall confidence (weighted average)
    confidence_overall = (
        momentum.confidence * Decimal("0.2") +
        valuation.confidence * Decimal("0.2") +
        smart_money.confidence * Decimal("0.2") +
        (Decimal("1.0") - vol_adj.confidence_penalty) * Decimal("0.4")
    )
    confidence_overall = _clamp(confidence_overall, Decimal("0.1"), Decimal("0.9"))

    return EnhancedScoringResult(
        ticker=ticker,
        clinical_normalized=clinical_normalized,
        financial_normalized=financial_normalized,
        catalyst_normalized=catalyst_normalized,
        momentum_signal=momentum,
        valuation_signal=valuation,
        smart_money_signal=smart_money,
        vol_adjustment=vol_adj,
        catalyst_decay=decay,
        interaction_terms=interactions,
        base_composite=base_composite,
        enhanced_composite=_quantize_score(enhanced_composite),
        composite_delta=_quantize_score(composite_delta),
        effective_weights=effective_weights,
        enhancement_flags=sorted(set(flags)),
        confidence_overall=_quantize_weight(confidence_overall),
    )


# =============================================================================
# ENHANCEMENT 6: CONTRADICTION DETECTOR
# =============================================================================

@dataclass
class ContradictionResult:
    """Result of contradiction detection between signals.

    Contradictions arise when different signals provide conflicting information,
    suggesting measurement error or structural inconsistency that should reduce
    confidence in the overall score.

    Example contradictions:
    - Strong momentum but failed liquidity gate (price moves without depth)
    - High valuation score but short runway (cheap for a reason)
    - Strong clinical score but severe dilution (market doesn't believe the data)
    """
    contradictions: List[str]
    confidence_penalty: Decimal
    score_penalty: Decimal
    diagnostics: Dict[str, Any]


# Contradiction detection rules
# Each rule specifies a condition and the penalties to apply when triggered
CONTRADICTION_RULES = [
    {
        "name": "momentum_liquidity_conflict",
        "description": "Strong momentum but liquidity gate failed - price moves without depth",
        "confidence_penalty": Decimal("0.15"),
        "score_penalty": Decimal("5"),
    },
    {
        "name": "valuation_financing_conflict",
        "description": "High valuation score but short runway - cheap for a reason",
        "confidence_penalty": Decimal("0.10"),
        "score_penalty": Decimal("3"),
    },
    {
        "name": "clinical_dilution_conflict",
        "description": "Strong clinical score but severe dilution - market skepticism",
        "confidence_penalty": Decimal("0.10"),
        "score_penalty": Decimal("4"),
    },
    {
        "name": "momentum_fundamental_divergence",
        "description": "Strong momentum but poor financial health - unsustainable rally",
        "confidence_penalty": Decimal("0.12"),
        "score_penalty": Decimal("4"),
    },
]

# Penalty caps to prevent excessive penalization
CONTRADICTION_MAX_CONF_PENALTY = Decimal("0.30")
CONTRADICTION_MAX_SCORE_PENALTY = Decimal("10")


def detect_contradictions(data: Dict[str, Any]) -> ContradictionResult:
    """
    Detect contradictions between scoring signals.

    Examines relationships between different signals to identify cases where
    signals conflict, suggesting measurement error or structural inconsistency.

    Args:
        data: Dict containing scoring signals with keys:
            - momentum_score: Decimal (0-100)
            - valuation_score: Decimal (0-100)
            - clinical_score: Decimal (0-100)
            - financial_score: Decimal (0-100)
            - liquidity_gate: str ("PASS", "WARN", "FAIL")
            - runway_months: Decimal or int
            - dilution_bucket: str ("LOW", "MEDIUM", "HIGH", "SEVERE")

    Returns:
        ContradictionResult with detected contradictions and penalties
    """
    contradictions = []
    total_conf_penalty = Decimal("0")
    total_score_penalty = Decimal("0")
    diagnostics: Dict[str, Any] = {"rules_checked": 0, "rules_triggered": []}

    # Helper to safely get Decimal value
    def get_decimal(key: str, default: Decimal = Decimal("50")) -> Optional[Decimal]:
        val = data.get(key)
        if val is None:
            return None
        try:
            return Decimal(str(val))
        except (ValueError, InvalidOperation):
            return default

    # Extract values
    momentum_score = get_decimal("momentum_score")
    valuation_score = get_decimal("valuation_score")
    clinical_score = get_decimal("clinical_score")
    financial_score = get_decimal("financial_score")
    liquidity_gate = data.get("liquidity_gate", "UNKNOWN")
    runway_months = get_decimal("runway_months")
    dilution_bucket = data.get("dilution_bucket", "UNKNOWN")

    # Rule 1: momentum_liquidity_conflict
    # Strong momentum (>70) but liquidity gate failed
    diagnostics["rules_checked"] += 1
    if momentum_score is not None and momentum_score > Decimal("70") and liquidity_gate == "FAIL":
        rule = CONTRADICTION_RULES[0]
        contradictions.append(rule["name"])
        total_conf_penalty += rule["confidence_penalty"]
        total_score_penalty += rule["score_penalty"]
        diagnostics["rules_triggered"].append(rule["name"])

    # Rule 2: valuation_financing_conflict
    # High valuation score (>70) but short runway (<12 months)
    diagnostics["rules_checked"] += 1
    if (valuation_score is not None and valuation_score > Decimal("70") and
            runway_months is not None and runway_months < Decimal("12")):
        rule = CONTRADICTION_RULES[1]
        contradictions.append(rule["name"])
        total_conf_penalty += rule["confidence_penalty"]
        total_score_penalty += rule["score_penalty"]
        diagnostics["rules_triggered"].append(rule["name"])

    # Rule 3: clinical_dilution_conflict
    # Strong clinical score (>70) but severe dilution
    diagnostics["rules_checked"] += 1
    if (clinical_score is not None and clinical_score > Decimal("70") and
            dilution_bucket in ("HIGH", "SEVERE")):
        rule = CONTRADICTION_RULES[2]
        contradictions.append(rule["name"])
        total_conf_penalty += rule["confidence_penalty"]
        total_score_penalty += rule["score_penalty"]
        diagnostics["rules_triggered"].append(rule["name"])

    # Rule 4: momentum_fundamental_divergence
    # Strong momentum (>70) but poor financial health (<40)
    diagnostics["rules_checked"] += 1
    if (momentum_score is not None and momentum_score > Decimal("70") and
            financial_score is not None and financial_score < Decimal("40")):
        rule = CONTRADICTION_RULES[3]
        contradictions.append(rule["name"])
        total_conf_penalty += rule["confidence_penalty"]
        total_score_penalty += rule["score_penalty"]
        diagnostics["rules_triggered"].append(rule["name"])

    # Cap penalties
    total_conf_penalty = min(total_conf_penalty, CONTRADICTION_MAX_CONF_PENALTY)
    total_score_penalty = min(total_score_penalty, CONTRADICTION_MAX_SCORE_PENALTY)

    return ContradictionResult(
        contradictions=contradictions,
        confidence_penalty=_quantize_weight(total_conf_penalty),
        score_penalty=_quantize_score(total_score_penalty),
        diagnostics=diagnostics,
    )
