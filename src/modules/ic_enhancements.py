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

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable

__version__ = "1.0.0"


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
MOMENTUM_OPTIMAL_ALPHA_BPS = 1000  # 10% alpha = 70 score

# Catalyst decay parameters
CATALYST_OPTIMAL_WINDOW_DAYS = 30
CATALYST_DECAY_RATES = {
    "PDUFA": Decimal("0.05"),
    "DATA_READOUT": Decimal("0.08"),
    "PHASE_COMPLETION": Decimal("0.10"),
    "ENROLLMENT_COMPLETE": Decimal("0.07"),
    "DEFAULT": Decimal("0.07"),
}

# Shrinkage normalization parameters
SHRINKAGE_MIN_COHORT = Decimal("5")
SHRINKAGE_PRIOR_STRENGTH = Decimal("5")  # Equivalent sample size of prior

# Smart money signal parameters
SMART_MONEY_OVERLAP_BONUS_PER_HOLDER = Decimal("5")
SMART_MONEY_MAX_OVERLAP_BONUS = Decimal("20")
SMART_MONEY_POSITION_CHANGE_WEIGHTS = {
    "NEW": Decimal("3"),
    "INCREASE": Decimal("2"),
    "HOLD": Decimal("0"),
    "DECREASE": Decimal("-2"),
    "EXIT": Decimal("-5"),
}

# Interaction term parameters
INTERACTION_SYNERGY_THRESHOLD_CLINICAL = Decimal("70")
INTERACTION_SYNERGY_THRESHOLD_RUNWAY = 18  # months
INTERACTION_SYNERGY_BONUS = Decimal("3.0")
INTERACTION_DISTRESS_THRESHOLD_RUNWAY = 12  # months
INTERACTION_DISTRESS_PENALTY = Decimal("5.0")


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
    """Price momentum signal result."""
    momentum_score: Decimal  # 0-100 normalized score
    alpha_60d: Optional[Decimal]  # Excess return vs benchmark
    return_60d: Optional[Decimal]
    benchmark_return_60d: Optional[Decimal]
    confidence: Decimal


@dataclass
class ValuationSignal:
    """Peer-relative valuation signal result."""
    valuation_score: Decimal  # 0-100 (higher = cheaper)
    mcap_per_asset: Optional[Decimal]
    peer_median_mcap_per_asset: Optional[Decimal]
    peer_count: int
    confidence: Decimal


@dataclass
class CatalystDecayResult:
    """Catalyst signal decay calculation result."""
    decay_factor: Decimal  # 0-1 multiplier
    days_to_catalyst: Optional[int]
    event_type: str
    in_optimal_window: bool


@dataclass
class SmartMoneySignal:
    """Smart money (13F) signal result."""
    smart_money_score: Decimal  # 20-80 range
    overlap_count: int
    overlap_bonus: Decimal
    position_change_adjustment: Decimal
    holders_increasing: List[str]
    holders_decreasing: List[str]
    confidence: Decimal


@dataclass
class InteractionTerms:
    """Non-linear interaction term calculations."""
    clinical_financial_synergy: Decimal
    stage_financial_interaction: Decimal
    catalyst_volatility_dampening: Decimal
    total_interaction_adjustment: Decimal
    interaction_flags: List[str]


@dataclass
class AdaptiveWeights:
    """Result of adaptive weight optimization."""
    weights: Dict[str, Decimal]
    historical_ic_by_component: Dict[str, Decimal]
    optimization_method: str
    lookback_months: int
    confidence: Decimal


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
    baseline_vol: Decimal = VOLATILITY_BASELINE,
    low_threshold: Decimal = VOLATILITY_LOW_THRESHOLD,
    high_threshold: Decimal = VOLATILITY_HIGH_THRESHOLD,
    max_adjustment: Decimal = VOLATILITY_MAX_ADJUSTMENT,
) -> VolatilityAdjustment:
    """
    Compute volatility-based adjustment factors.

    Logic:
    - Low volatility (<30%): Boost weight influence (more reliable signal)
    - Normal volatility (30-80%): Neutral adjustment
    - High volatility (>80%): Reduce weight influence (less reliable signal)

    Additionally applies score dampening for high-vol names to approximate
    risk-adjusted returns.

    Args:
        annualized_vol: Annualized volatility as decimal (e.g., 0.50 for 50%)
        baseline_vol: Target volatility for neutral adjustment
        low_threshold: Below this = low vol bucket
        high_threshold: Above this = high vol bucket
        max_adjustment: Maximum +/- adjustment factor

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

    if vol < low_threshold:
        # Low vol = more reliable signal = boost weights
        vol_bucket = VolatilityBucket.LOW
        # Linear interpolation: at 0% vol -> +max_adjustment, at threshold -> 0%
        weight_adj = Decimal("1.0") + max_adjustment * (
            (low_threshold - vol) / low_threshold
        )
        # Score boost for low vol (less risky)
        vol_ratio = vol / baseline_vol
        score_adj = Decimal("1.0") + (Decimal("1.0") - vol_ratio) * Decimal("0.10")
        confidence_penalty = Decimal("0")

    elif vol <= high_threshold:
        # Normal vol = neutral
        vol_bucket = VolatilityBucket.NORMAL
        weight_adj = Decimal("1.0")
        # Mild score dampening as vol increases within normal range
        vol_ratio = (vol - low_threshold) / (high_threshold - low_threshold)
        score_adj = Decimal("1.0") - vol_ratio * Decimal("0.05")
        # Confidence penalty increases with volatility
        confidence_penalty = vol_ratio * Decimal("0.10")

    else:
        # High vol = less reliable signal = reduce weight influence
        vol_bucket = VolatilityBucket.HIGH
        # Reduce weight influence
        excess_vol = vol - high_threshold
        weight_adj = Decimal("1.0") - min(
            max_adjustment,
            max_adjustment * (excess_vol / baseline_vol)
        )
        # Significant score dampening for high vol
        vol_ratio = min(vol / baseline_vol, Decimal("2.0"))
        score_adj = Decimal("1.0") - (vol_ratio - Decimal("1.0")) * Decimal("0.15")
        score_adj = max(score_adj, Decimal("0.70"))  # Floor at 30% penalty
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
    lookback_days: int = MOMENTUM_LOOKBACK_DAYS,
) -> MomentumSignal:
    """
    Compute price momentum signal relative to benchmark (XBI).

    Research shows 60-day momentum relative to sector ETF is predictive
    in biotech. Captures both absolute and relative strength.

    Args:
        return_60d: Ticker's 60-day return as decimal (0.10 = 10%)
        benchmark_return_60d: XBI's 60-day return as decimal
        lookback_days: Lookback period (for documentation)

    Returns:
        MomentumSignal with normalized score and components
    """
    ret = _to_decimal(return_60d)
    bench = _to_decimal(benchmark_return_60d)

    if ret is None or bench is None:
        return MomentumSignal(
            momentum_score=Decimal("50"),  # Neutral
            alpha_60d=None,
            return_60d=ret,
            benchmark_return_60d=bench,
            confidence=Decimal("0.3"),
        )

    # Compute alpha (excess return)
    alpha = ret - bench

    # Convert alpha to 0-100 score
    # +10% alpha (1000bps) = 70 score
    # -10% alpha = 30 score
    # Scale: 20bps of alpha = 1 point of score
    alpha_bps = alpha * Decimal("10000")  # Convert to basis points
    score_delta = alpha_bps / Decimal("50")  # 50bps = 1 point

    momentum_score = Decimal("50") + score_delta
    momentum_score = _clamp(momentum_score, Decimal("10"), Decimal("90"))

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

    return MomentumSignal(
        momentum_score=_quantize_score(momentum_score),
        alpha_60d=_quantize_weight(alpha),
        return_60d=_quantize_weight(ret),
        benchmark_return_60d=_quantize_weight(bench),
        confidence=confidence,
    )


# =============================================================================
# PEER-RELATIVE VALUATION
# =============================================================================

def compute_valuation_signal(
    market_cap_mm: Optional[Decimal],
    trial_count: int,
    lead_phase: Optional[str],
    peer_valuations: List[Dict[str, Any]],
) -> ValuationSignal:
    """
    Compute peer-relative valuation signal.

    Compares market cap per pipeline asset to peers at same development stage.
    Undervalued names (low mcap/trial) tend to outperform.

    Args:
        market_cap_mm: Market cap in millions
        trial_count: Number of active clinical trials
        lead_phase: Lead development phase
        peer_valuations: List of peer dicts with market_cap_mm, trial_count, stage_bucket

    Returns:
        ValuationSignal with normalized score and components
    """
    mcap = _to_decimal(market_cap_mm)

    if mcap is None or trial_count <= 0:
        return ValuationSignal(
            valuation_score=Decimal("50"),
            mcap_per_asset=None,
            peer_median_mcap_per_asset=None,
            peer_count=0,
            confidence=Decimal("0.2"),
        )

    # Compute mcap per asset
    mcap_per_asset = mcap / Decimal(max(trial_count, 1))

    # Determine stage bucket for peer comparison
    stage = _stage_bucket(lead_phase)

    # Filter peers to same stage
    same_stage_peers = [
        p for p in peer_valuations
        if p.get("stage_bucket") == stage and p.get("trial_count", 0) > 0
    ]

    if len(same_stage_peers) < 5:
        return ValuationSignal(
            valuation_score=Decimal("50"),
            mcap_per_asset=_quantize_score(mcap_per_asset),
            peer_median_mcap_per_asset=None,
            peer_count=len(same_stage_peers),
            confidence=Decimal("0.3"),
        )

    # Compute peer mcap/asset values
    peer_mcap_per_asset = []
    for p in same_stage_peers:
        p_mcap = _to_decimal(p.get("market_cap_mm"))
        p_trials = p.get("trial_count", 0)
        if p_mcap is not None and p_trials > 0:
            peer_mcap_per_asset.append(p_mcap / Decimal(p_trials))

    if not peer_mcap_per_asset:
        return ValuationSignal(
            valuation_score=Decimal("50"),
            mcap_per_asset=_quantize_score(mcap_per_asset),
            peer_median_mcap_per_asset=None,
            peer_count=0,
            confidence=Decimal("0.2"),
        )

    # Compute percentile rank (lower mcap/asset = cheaper = better)
    cheaper_count = sum(1 for p in peer_mcap_per_asset if p < mcap_per_asset)
    percentile = Decimal(str(cheaper_count / len(peer_mcap_per_asset) * 100))

    # Invert: cheap is high signal (100 - percentile = valuation_score)
    valuation_score = Decimal("100") - percentile
    valuation_score = _clamp(valuation_score, Decimal("10"), Decimal("90"))

    # Compute peer median for reference
    sorted_peers = sorted(peer_mcap_per_asset)
    mid = len(sorted_peers) // 2
    if len(sorted_peers) % 2 == 0:
        peer_median = (sorted_peers[mid - 1] + sorted_peers[mid]) / 2
    else:
        peer_median = sorted_peers[mid]

    # Confidence based on peer count
    if len(peer_mcap_per_asset) >= 20:
        confidence = Decimal("0.8")
    elif len(peer_mcap_per_asset) >= 10:
        confidence = Decimal("0.6")
    else:
        confidence = Decimal("0.4")

    return ValuationSignal(
        valuation_score=_quantize_score(valuation_score),
        mcap_per_asset=_quantize_score(mcap_per_asset),
        peer_median_mcap_per_asset=_quantize_score(peer_median),
        peer_count=len(peer_mcap_per_asset),
        confidence=confidence,
    )


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

    Uses exponential decay model with event-specific decay rates.

    Args:
        days_to_catalyst: Days until catalyst event (negative = past)
        event_type: Type of catalyst (PDUFA, DATA_READOUT, etc.)
        optimal_window: Days before event for peak signal

    Returns:
        CatalystDecayResult with decay factor and metadata
    """
    if days_to_catalyst is None:
        return CatalystDecayResult(
            decay_factor=Decimal("0.5"),  # Neutral
            days_to_catalyst=None,
            event_type=event_type,
            in_optimal_window=False,
        )

    # Get decay rate for event type
    tau = CATALYST_DECAY_RATES.get(event_type, CATALYST_DECAY_RATES["DEFAULT"])

    # Compute distance from optimal window
    distance = abs(days_to_catalyst - optimal_window)

    # Check if in optimal window (within 15 days of peak)
    in_optimal_window = distance <= 15

    # Exponential decay
    # decay_factor = exp(-tau * distance)
    try:
        decay_float = math.exp(-float(tau) * distance)
        decay_factor = Decimal(str(decay_float))
    except (OverflowError, ValueError):
        decay_factor = Decimal("0.01")

    # Apply asymmetric adjustment: past events decay faster
    if days_to_catalyst < 0:
        # Event already happened - faster decay
        decay_factor = decay_factor * Decimal("0.7")

    decay_factor = _clamp(decay_factor, Decimal("0.05"), Decimal("1.0"))

    return CatalystDecayResult(
        decay_factor=_quantize_weight(decay_factor),
        days_to_catalyst=days_to_catalyst,
        event_type=event_type,
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

def compute_smart_money_signal(
    overlap_count: int,
    holders: List[str],
    position_changes: Optional[Dict[str, str]] = None,
) -> SmartMoneySignal:
    """
    Compute smart money signal from 13F co-invest data.

    Converts co-invest overlay into a scoring component.
    New positions and increases are more predictive than static holdings.

    Args:
        overlap_count: Number of tracked managers holding the position
        holders: List of holder names
        position_changes: Dict mapping holder -> change type (NEW, INCREASE, etc.)

    Returns:
        SmartMoneySignal with normalized score and components
    """
    base_signal = Decimal("50")

    # Overlap count bonus
    overlap_bonus = min(
        Decimal(overlap_count) * SMART_MONEY_OVERLAP_BONUS_PER_HOLDER,
        SMART_MONEY_MAX_OVERLAP_BONUS
    )

    # Position change adjustments
    change_bonus = Decimal("0")
    holders_increasing = []
    holders_decreasing = []

    if position_changes:
        for holder, change in position_changes.items():
            change_upper = change.upper() if isinstance(change, str) else "HOLD"
            weight = SMART_MONEY_POSITION_CHANGE_WEIGHTS.get(change_upper, Decimal("0"))
            change_bonus += weight

            if change_upper in ("NEW", "INCREASE"):
                holders_increasing.append(holder)
            elif change_upper in ("DECREASE", "EXIT"):
                holders_decreasing.append(holder)

    # Clamp change bonus
    change_bonus = _clamp(change_bonus, Decimal("-15"), Decimal("15"))

    # Compute final score
    smart_money_score = base_signal + overlap_bonus + change_bonus
    smart_money_score = _clamp(smart_money_score, Decimal("20"), Decimal("80"))

    # Confidence based on data availability
    if position_changes and overlap_count >= 3:
        confidence = Decimal("0.7")
    elif overlap_count >= 2:
        confidence = Decimal("0.5")
    elif overlap_count >= 1:
        confidence = Decimal("0.4")
    else:
        confidence = Decimal("0.2")

    return SmartMoneySignal(
        smart_money_score=_quantize_score(smart_money_score),
        overlap_count=overlap_count,
        overlap_bonus=_quantize_score(overlap_bonus),
        position_change_adjustment=_quantize_score(change_bonus),
        holders_increasing=sorted(holders_increasing),
        holders_decreasing=sorted(holders_decreasing),
        confidence=confidence,
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
) -> InteractionTerms:
    """
    Compute non-linear interaction terms between signals.

    Captures synergies and distress patterns that linear combination misses:
    - High clinical + strong financial = synergy bonus
    - Late-stage + weak runway = distress penalty
    - High catalyst + high vol = dampened catalyst

    Args:
        clinical_normalized: Normalized clinical score (0-100)
        financial_data: Dict with runway_months, financial_score, etc.
        catalyst_normalized: Normalized catalyst score (0-100)
        stage_bucket: Development stage (early, mid, late)
        vol_adjustment: Optional volatility adjustment result

    Returns:
        InteractionTerms with all computed interactions
    """
    flags = []

    # Extract financial metrics
    runway_months = _to_decimal(financial_data.get("runway_months"), Decimal("24"))
    financial_score = _to_decimal(financial_data.get("financial_score"), Decimal("50"))

    # 1. Clinical x Financial synergy
    # Strong clinical + strong financial = premium company
    clinical_financial_synergy = Decimal("0")
    if (clinical_normalized >= INTERACTION_SYNERGY_THRESHOLD_CLINICAL and
        runway_months >= INTERACTION_SYNERGY_THRESHOLD_RUNWAY):
        clinical_financial_synergy = INTERACTION_SYNERGY_BONUS
        flags.append("clinical_financial_synergy")

    # 2. Stage x Financial distress
    # Late-stage with weak runway = dilution risk, severe penalty
    stage_financial_interaction = Decimal("0")
    if stage_bucket == "late" and runway_months < INTERACTION_DISTRESS_THRESHOLD_RUNWAY:
        stage_financial_interaction = -INTERACTION_DISTRESS_PENALTY
        flags.append("late_stage_distress")
    elif stage_bucket == "mid" and runway_months < Decimal("9"):
        stage_financial_interaction = -INTERACTION_DISTRESS_PENALTY * Decimal("0.5")
        flags.append("mid_stage_runway_warning")

    # 3. Catalyst x Volatility dampening
    # High catalyst in high-vol name = less reliable signal
    catalyst_volatility_dampening = Decimal("0")
    if vol_adjustment and vol_adjustment.vol_bucket == VolatilityBucket.HIGH:
        # Dampen catalyst deviation from neutral
        catalyst_excess = catalyst_normalized - Decimal("50")
        if abs(catalyst_excess) > Decimal("15"):
            # Reduce impact of extreme catalyst signals in high-vol names
            dampening = catalyst_excess * Decimal("0.2")  # 20% dampening
            catalyst_volatility_dampening = -dampening
            flags.append("catalyst_vol_dampening")

    # Total interaction adjustment
    total = clinical_financial_synergy + stage_financial_interaction + catalyst_volatility_dampening

    return InteractionTerms(
        clinical_financial_synergy=_quantize_score(clinical_financial_synergy),
        stage_financial_interaction=_quantize_score(stage_financial_interaction),
        catalyst_volatility_dampening=_quantize_score(catalyst_volatility_dampening),
        total_interaction_adjustment=_quantize_score(total),
        interaction_flags=flags,
    )


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

    # Use global std with mild shrinkage (std estimation is noisier)
    cohort_variance = sum((v - cohort_mean) ** 2 for v in values) / Decimal(n)
    cohort_std = cohort_variance.sqrt() if cohort_variance > 0 else Decimal("1")
    adjusted_std = (
        cohort_std * (Decimal("1") - shrinkage_factor * Decimal("0.5")) +
        global_std * shrinkage_factor * Decimal("0.5")
    )
    adjusted_std = max(adjusted_std, Decimal("0.01"))  # Prevent division by zero

    # Normalize with shrunk parameters and convert to percentile-like scale
    result = []
    for v in values:
        z = (v - adjusted_mean) / adjusted_std
        # Convert z-score to 0-100 scale (z=0 -> 50, z=±2 -> ~5/95)
        percentile = Decimal("50") + z * Decimal("20")
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
# ADAPTIVE WEIGHT LEARNING
# =============================================================================

def compute_adaptive_weights(
    historical_scores: List[Dict[str, Any]],
    historical_returns: Dict[str, Decimal],
    base_weights: Dict[str, Decimal],
    *,
    lookback_months: int = 12,
    min_weight: Decimal = Decimal("0.05"),
    max_weight: Decimal = Decimal("0.50"),
) -> AdaptiveWeights:
    """
    Compute weights that maximize historical IC using simple rank correlation.

    Uses PIT-safe historical data to estimate component-level IC contribution,
    then adjusts weights to overweight high-IC components.

    This is a simplified version suitable for production without scipy.
    Uses rank correlation (Spearman-like) computed with pure Python.

    Args:
        historical_scores: List of dicts with ticker, component scores, as_of_date
        historical_returns: Dict of ticker -> forward return (1-month or 3-month)
        base_weights: Starting weights to adjust from
        lookback_months: Months of history to use
        min_weight: Minimum weight per component
        max_weight: Maximum weight per component

    Returns:
        AdaptiveWeights with optimized weights and diagnostics
    """
    if not historical_scores or not historical_returns:
        return AdaptiveWeights(
            weights=base_weights.copy(),
            historical_ic_by_component={},
            optimization_method="fallback_no_data",
            lookback_months=lookback_months,
            confidence=Decimal("0.1"),
        )

    # Extract component names from first record
    sample = historical_scores[0]
    component_names = [k for k in base_weights.keys() if k in sample or f"{k}_normalized" in sample]

    if not component_names:
        return AdaptiveWeights(
            weights=base_weights.copy(),
            historical_ic_by_component={},
            optimization_method="fallback_no_components",
            lookback_months=lookback_months,
            confidence=Decimal("0.1"),
        )

    # Compute IC for each component
    ic_by_component: Dict[str, Decimal] = {}

    for comp in component_names:
        # Extract (score, return) pairs
        pairs = []
        for rec in historical_scores:
            ticker = rec.get("ticker")
            if ticker not in historical_returns:
                continue

            # Try both normalized and raw score names
            score = _to_decimal(rec.get(f"{comp}_normalized") or rec.get(comp))
            if score is None:
                continue

            ret = historical_returns[ticker]
            pairs.append((score, ret))

        if len(pairs) < 20:
            # Not enough data for reliable IC
            ic_by_component[comp] = Decimal("0")
            continue

        # Compute rank correlation
        ic = _compute_rank_correlation(pairs)
        ic_by_component[comp] = ic

    # Adjust weights based on IC
    # Higher IC -> higher weight (but bounded)
    adjusted_weights = {}
    total_positive_ic = sum(max(ic, Decimal("0")) for ic in ic_by_component.values())

    if total_positive_ic <= EPS:
        # No positive IC found, use base weights
        return AdaptiveWeights(
            weights=base_weights.copy(),
            historical_ic_by_component={k: str(v) for k, v in ic_by_component.items()},
            optimization_method="fallback_no_positive_ic",
            lookback_months=lookback_months,
            confidence=Decimal("0.2"),
        )

    for comp, ic in ic_by_component.items():
        base = base_weights.get(comp, Decimal("0.20"))

        # Scale adjustment by IC (positive IC increases weight)
        # ic_adjustment = (ic / 0.10) * 0.10 = ic itself as percentage adjustment
        ic_adjustment = ic  # IC of 0.05 = 5% weight increase

        adjusted = base * (Decimal("1") + ic_adjustment * Decimal("2"))
        adjusted = _clamp(adjusted, min_weight, max_weight)
        adjusted_weights[comp] = adjusted

    # Renormalize
    total = sum(adjusted_weights.values())
    if total > EPS:
        adjusted_weights = {k: _quantize_weight(v / total) for k, v in adjusted_weights.items()}

    # Compute confidence based on sample size and IC variance
    sample_size = len(historical_scores)
    if sample_size >= 500:
        confidence = Decimal("0.8")
    elif sample_size >= 200:
        confidence = Decimal("0.6")
    elif sample_size >= 100:
        confidence = Decimal("0.4")
    else:
        confidence = Decimal("0.3")

    return AdaptiveWeights(
        weights=adjusted_weights,
        historical_ic_by_component={k: str(v) for k, v in ic_by_component.items()},
        optimization_method="rank_correlation",
        lookback_months=lookback_months,
        confidence=confidence,
    )


def _compute_rank_correlation(pairs: List[Tuple[Decimal, Decimal]]) -> Decimal:
    """
    Compute Spearman rank correlation between score and return pairs.

    Pure Python implementation without scipy.

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
    """Compute ranks with average rank for ties."""
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

    # 2. Compute momentum signal
    momentum = compute_momentum_signal(return_60d, benchmark_return_60d)
    if momentum.alpha_60d and abs(momentum.alpha_60d) >= Decimal("0.10"):
        flags.append("strong_momentum" if momentum.alpha_60d > 0 else "weak_momentum")

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
