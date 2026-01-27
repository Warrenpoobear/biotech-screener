"""
Scoring Robustness Enhancements for Module 5 Composite Scoring

This module implements advanced robustness features for the composite scoring system:

1. Winsorization at Component Level - Clip extreme scores to reduce outlier impact
2. Confidence-Weighted Shrinkage - Shrink low-confidence scores toward cohort mean
3. Rank Stability Regularization - Penalize volatile rankings to reduce churn
4. Multi-Timeframe Signal Blending - Blend signals across different time horizons
5. Asymmetric Interaction Bounds - Different caps for positive vs negative adjustments
6. Regime-Conditional Weight Floors - Minimum weights for critical components by regime
7. Defensive Override Triggers - Automatic defensive posture detection
8. Score Distribution Health Checks - Monitor for problematic distributions

Design Philosophy:
- DETERMINISTIC: No datetime.now(), no randomness
- DECIMAL-ONLY: Pure Decimal arithmetic for precision
- BOUNDED: All adjustments are capped to prevent extreme outcomes
- AUDITABLE: Clear logging of all adjustments made

Author: Wake Robin Capital Management
Version: 1.0.0
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

__version__ = "1.0.0"


# =============================================================================
# CONSTANTS
# =============================================================================

SCORE_PRECISION = Decimal("0.01")
WEIGHT_PRECISION = Decimal("0.0001")
EPS = Decimal("0.000001")

# Winsorization parameters
COMPONENT_WINSOR_LOW = Decimal("2.5")   # 2.5th percentile
COMPONENT_WINSOR_HIGH = Decimal("97.5")  # 97.5th percentile
WINSOR_ABSOLUTE_FLOOR = Decimal("5")     # Never below 5
WINSOR_ABSOLUTE_CAP = Decimal("95")      # Never above 95

# Confidence-weighted shrinkage parameters
SHRINKAGE_LOW_CONF_THRESHOLD = Decimal("0.5")  # Below this, apply shrinkage
SHRINKAGE_MAX_FACTOR = Decimal("0.5")  # Max 50% shrinkage toward cohort mean

# Rank stability parameters
RANK_STABILITY_LOOKBACK = 5  # Number of prior rankings to consider
RANK_STABILITY_PENALTY_PER_POSITION = Decimal("0.02")  # 2% penalty per position moved
RANK_STABILITY_MAX_PENALTY = Decimal("3.0")  # Max 3 points penalty

# Multi-timeframe blending weights
TIMEFRAME_WEIGHTS = {
    "short": Decimal("0.20"),   # 20d signals
    "medium": Decimal("0.50"),  # 60d signals (primary)
    "long": Decimal("0.30"),    # 120d signals (trend confirmation)
}

# Asymmetric interaction bounds
# Rationale: Negative signals (risks) should have tighter bounds than positive
# to preserve defensive posture. Positive signals allow more upside capture.
INTERACTION_POSITIVE_CAP = Decimal("3.5")   # Max positive adjustment
INTERACTION_NEGATIVE_FLOOR = Decimal("-2.5")  # Max negative adjustment (tighter)

# Regime-conditional weight floors
# Ensures critical components always have meaningful weight
REGIME_WEIGHT_FLOORS = {
    "BULL": {
        "clinical": Decimal("0.15"),
        "financial": Decimal("0.10"),
        "catalyst": Decimal("0.10"),
    },
    "BEAR": {
        "clinical": Decimal("0.10"),
        "financial": Decimal("0.25"),  # Higher floor in bear market
        "catalyst": Decimal("0.05"),
    },
    "RECESSION_RISK": {
        "clinical": Decimal("0.10"),
        "financial": Decimal("0.30"),  # Highest floor - cash is king
        "catalyst": Decimal("0.05"),
    },
    "CREDIT_CRISIS": {
        "clinical": Decimal("0.10"),
        "financial": Decimal("0.35"),  # Financial health critical
        "catalyst": Decimal("0.05"),
    },
    "VOLATILITY_SPIKE": {
        "clinical": Decimal("0.15"),
        "financial": Decimal("0.20"),
        "catalyst": Decimal("0.05"),
    },
    "SECTOR_ROTATION": {
        "clinical": Decimal("0.15"),
        "financial": Decimal("0.15"),
        "catalyst": Decimal("0.10"),
    },
    "SECTOR_DISLOCATION": {
        "clinical": Decimal("0.15"),
        "financial": Decimal("0.20"),
        "catalyst": Decimal("0.10"),
    },
    "NEUTRAL": {
        "clinical": Decimal("0.12"),
        "financial": Decimal("0.12"),
        "catalyst": Decimal("0.08"),
    },
}

# Defensive override thresholds
DEFENSIVE_TRIGGER_CONDITIONS = {
    "max_severity_ratio": Decimal("0.30"),      # >30% SEV2+ triggers defense
    "min_avg_runway_months": Decimal("9"),       # Avg runway <9mo triggers
    "max_high_vol_ratio": Decimal("0.40"),       # >40% high-vol triggers
    "min_positive_momentum_ratio": Decimal("0.25"),  # <25% positive momentum triggers
}

# Score distribution health thresholds
DISTRIBUTION_HEALTH_THRESHOLDS = {
    "min_std": Decimal("8"),      # Scores too clustered if std < 8
    "max_std": Decimal("25"),     # Scores too dispersed if std > 25
    "min_iqr": Decimal("10"),     # IQR < 10 indicates lack of differentiation
    "max_skew": Decimal("1.5"),   # Skew > 1.5 indicates distributional problem
    "max_zero_ratio": Decimal("0.20"),  # >20% zeros is problematic
    "min_range": Decimal("40"),   # Score range < 40 lacks differentiation
}


# =============================================================================
# ENUMS
# =============================================================================

class DefensivePosture(str, Enum):
    """Defensive posture classification."""
    NONE = "none"
    LIGHT = "light"           # Minor defensive adjustments
    MODERATE = "moderate"     # Significant defensive bias
    HEAVY = "heavy"           # Maximum defensive mode


class DistributionHealth(str, Enum):
    """Score distribution health status."""
    HEALTHY = "healthy"
    CLUSTERED = "clustered"   # Scores too similar
    DISPERSED = "dispersed"   # Scores too spread
    SKEWED = "skewed"         # Asymmetric distribution
    DEGRADED = "degraded"     # Multiple issues


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class WinsorizedScore:
    """Result of component-level winsorization."""
    original: Decimal
    winsorized: Decimal
    was_clipped: bool
    clip_direction: Optional[str]  # "low", "high", or None
    percentile_rank: Optional[Decimal]


@dataclass
class ShrinkageResult:
    """Result of confidence-weighted shrinkage."""
    original: Decimal
    shrunk: Decimal
    shrinkage_factor: Decimal  # 0 = no shrinkage, 1 = full shrinkage to mean
    cohort_mean: Decimal
    confidence_used: Decimal


@dataclass
class RankStabilityAdjustment:
    """Rank stability regularization result."""
    original_score: Decimal
    adjusted_score: Decimal
    penalty_applied: Decimal
    prior_rank: Optional[int]
    current_rank: int
    rank_change: int
    flags: List[str] = field(default_factory=list)


@dataclass
class AsymmetricBounds:
    """Asymmetric interaction bounds configuration."""
    positive_cap: Decimal
    negative_floor: Decimal
    applied_value: Decimal
    original_value: Decimal
    was_capped: bool
    cap_type: Optional[str]  # "positive", "negative", or None


@dataclass
class WeightFloorResult:
    """Result of applying regime-conditional weight floors."""
    original_weights: Dict[str, Decimal]
    adjusted_weights: Dict[str, Decimal]
    floors_applied: Dict[str, Decimal]
    regime: str
    weight_redistributed: Decimal
    flags: List[str] = field(default_factory=list)


@dataclass
class DefensiveOverrideResult:
    """Result of defensive override trigger evaluation."""
    posture: DefensivePosture
    triggers_hit: List[str]
    trigger_values: Dict[str, Decimal]
    adjustments_applied: Dict[str, Decimal]
    flags: List[str] = field(default_factory=list)


@dataclass
class DistributionHealthCheck:
    """Score distribution health check result."""
    health: DistributionHealth
    mean: Decimal
    std: Decimal
    iqr: Decimal
    skewness: Decimal
    zero_ratio: Decimal
    score_range: Decimal
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class RobustnessEnhancements:
    """Aggregated robustness enhancement results."""
    winsorization_applied: int
    shrinkage_adjustments: int
    rank_stability_penalties: Decimal
    interaction_caps_applied: int
    weight_floors_triggered: int
    defensive_posture: DefensivePosture
    distribution_health: DistributionHealth
    total_adjustment: Decimal
    flags: List[str] = field(default_factory=list)


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
    except Exception:
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


def _compute_percentile(value: Decimal, sorted_values: List[Decimal]) -> Decimal:
    """Compute percentile rank of value in sorted list."""
    if not sorted_values:
        return Decimal("50")
    n = len(sorted_values)
    if n == 1:
        return Decimal("50")

    # Count values below and equal
    below = sum(1 for v in sorted_values if v < value)
    equal = sum(1 for v in sorted_values if v == value)

    # Midrank percentile
    percentile = Decimal(str((below + equal / 2) / n)) * Decimal("100")
    return _quantize_score(percentile)


# =============================================================================
# 1. WINSORIZATION AT COMPONENT LEVEL
# =============================================================================

def winsorize_component_score(
    score: Decimal,
    cohort_scores: List[Decimal],
    *,
    low_percentile: Decimal = COMPONENT_WINSOR_LOW,
    high_percentile: Decimal = COMPONENT_WINSOR_HIGH,
    absolute_floor: Decimal = WINSOR_ABSOLUTE_FLOOR,
    absolute_cap: Decimal = WINSOR_ABSOLUTE_CAP,
) -> WinsorizedScore:
    """
    Apply winsorization to a component score before composite aggregation.

    Clips extreme scores to percentile bounds to reduce outlier impact on
    the composite score. Uses both percentile-based and absolute bounds.

    Args:
        score: Raw component score (0-100)
        cohort_scores: All scores in the cohort for percentile calculation
        low_percentile: Lower percentile bound (default 2.5)
        high_percentile: Upper percentile bound (default 97.5)
        absolute_floor: Absolute minimum score
        absolute_cap: Absolute maximum score

    Returns:
        WinsorizedScore with original, winsorized value, and diagnostics
    """
    original = _to_decimal(score, Decimal("50"))

    if not cohort_scores or len(cohort_scores) < 3:
        # Insufficient data - use absolute bounds only
        winsorized = _clamp(original, absolute_floor, absolute_cap)
        return WinsorizedScore(
            original=original,
            winsorized=winsorized,
            was_clipped=original != winsorized,
            clip_direction="low" if winsorized > original else ("high" if winsorized < original else None),
            percentile_rank=None,
        )

    # Sort for percentile calculation
    sorted_scores = sorted([_to_decimal(s, Decimal("50")) for s in cohort_scores])
    n = len(sorted_scores)

    # Calculate percentile bounds
    low_idx = int(float(low_percentile / Decimal("100")) * n)
    high_idx = min(int(float(high_percentile / Decimal("100")) * n), n - 1)

    percentile_floor = sorted_scores[low_idx]
    percentile_cap = sorted_scores[high_idx]

    # Use more restrictive of percentile and absolute bounds
    effective_floor = max(percentile_floor, absolute_floor)
    effective_cap = min(percentile_cap, absolute_cap)

    # Apply winsorization
    winsorized = _clamp(original, effective_floor, effective_cap)

    # Calculate percentile rank
    percentile_rank = _compute_percentile(original, sorted_scores)

    return WinsorizedScore(
        original=_quantize_score(original),
        winsorized=_quantize_score(winsorized),
        was_clipped=original != winsorized,
        clip_direction="low" if winsorized > original else ("high" if winsorized < original else None),
        percentile_rank=percentile_rank,
    )


def winsorize_cohort(
    scores: Dict[str, Decimal],
    *,
    low_percentile: Decimal = COMPONENT_WINSOR_LOW,
    high_percentile: Decimal = COMPONENT_WINSOR_HIGH,
) -> Tuple[Dict[str, Decimal], int]:
    """
    Winsorize all scores in a cohort.

    Args:
        scores: Dict of ticker -> score
        low_percentile: Lower percentile bound
        high_percentile: Upper percentile bound

    Returns:
        Tuple of (winsorized scores dict, count of clipped scores)
    """
    if not scores:
        return {}, 0

    cohort_values = list(scores.values())
    clipped_count = 0
    winsorized = {}

    for ticker, score in scores.items():
        result = winsorize_component_score(
            score, cohort_values,
            low_percentile=low_percentile,
            high_percentile=high_percentile,
        )
        winsorized[ticker] = result.winsorized
        if result.was_clipped:
            clipped_count += 1

    return winsorized, clipped_count


# =============================================================================
# 2. CONFIDENCE-WEIGHTED SHRINKAGE
# =============================================================================

def apply_confidence_shrinkage(
    score: Decimal,
    confidence: Decimal,
    cohort_mean: Decimal,
    *,
    low_conf_threshold: Decimal = SHRINKAGE_LOW_CONF_THRESHOLD,
    max_shrinkage: Decimal = SHRINKAGE_MAX_FACTOR,
) -> ShrinkageResult:
    """
    Apply confidence-weighted shrinkage toward cohort mean.

    Low-confidence scores are pulled toward the cohort mean to reduce
    noise from unreliable signals. High-confidence scores are preserved.

    Shrinkage formula:
        shrinkage_factor = max_shrinkage * (1 - confidence/threshold) for conf < threshold
        shrunk_score = score * (1 - shrinkage_factor) + cohort_mean * shrinkage_factor

    Args:
        score: Raw score (0-100)
        confidence: Signal confidence (0-1)
        cohort_mean: Mean of cohort scores
        low_conf_threshold: Confidence below which shrinkage applies
        max_shrinkage: Maximum shrinkage factor

    Returns:
        ShrinkageResult with original, shrunk value, and diagnostics
    """
    score_dec = _to_decimal(score, Decimal("50"))
    conf = _clamp(_to_decimal(confidence, Decimal("0.5")), Decimal("0"), Decimal("1"))
    mean = _to_decimal(cohort_mean, Decimal("50"))

    # Calculate shrinkage factor
    if conf >= low_conf_threshold:
        shrinkage_factor = Decimal("0")
    else:
        # Linear ramp: 0 at threshold, max_shrinkage at 0
        conf_ratio = conf / low_conf_threshold
        shrinkage_factor = max_shrinkage * (Decimal("1") - conf_ratio)

    # Apply shrinkage
    shrunk = score_dec * (Decimal("1") - shrinkage_factor) + mean * shrinkage_factor

    return ShrinkageResult(
        original=_quantize_score(score_dec),
        shrunk=_quantize_score(shrunk),
        shrinkage_factor=_quantize_weight(shrinkage_factor),
        cohort_mean=_quantize_score(mean),
        confidence_used=_quantize_weight(conf),
    )


# =============================================================================
# 3. RANK STABILITY REGULARIZATION
# =============================================================================

def compute_rank_stability_penalty(
    current_score: Decimal,
    current_rank: int,
    prior_ranks: List[int],
    *,
    penalty_per_position: Decimal = RANK_STABILITY_PENALTY_PER_POSITION,
    max_penalty: Decimal = RANK_STABILITY_MAX_PENALTY,
) -> RankStabilityAdjustment:
    """
    Compute rank stability penalty to reduce ranking churn.

    Penalizes large rank changes to smooth rankings over time. This reduces
    whipsaw from noisy signals while still allowing genuine rank improvements.

    Penalty formula:
        penalty = min(penalty_per_position * abs(rank_change), max_penalty)

    Applied asymmetrically:
        - Rank improvements: 50% of calculated penalty (allow upward mobility)
        - Rank declines: 100% of calculated penalty (resist downward moves)

    Args:
        current_score: Current composite score
        current_rank: Current ranking position
        prior_ranks: List of prior ranking positions (most recent first)
        penalty_per_position: Penalty per position of rank change
        max_penalty: Maximum penalty cap

    Returns:
        RankStabilityAdjustment with adjusted score and diagnostics
    """
    score = _to_decimal(current_score, Decimal("50"))
    flags = []

    if not prior_ranks:
        return RankStabilityAdjustment(
            original_score=_quantize_score(score),
            adjusted_score=_quantize_score(score),
            penalty_applied=Decimal("0"),
            prior_rank=None,
            current_rank=current_rank,
            rank_change=0,
            flags=["no_prior_ranks"],
        )

    # Use most recent prior rank
    prior_rank = prior_ranks[0]
    rank_change = current_rank - prior_rank  # Positive = worse rank (higher number)

    # Calculate raw penalty
    raw_penalty = penalty_per_position * abs(Decimal(str(rank_change)))

    # Apply asymmetric penalty
    if rank_change < 0:
        # Rank improved (moved up) - apply 50% penalty
        penalty = raw_penalty * Decimal("0.5")
        flags.append("rank_improved_partial_penalty")
    else:
        # Rank declined (moved down) - full penalty
        penalty = raw_penalty
        if rank_change > 5:
            flags.append("significant_rank_decline")

    # Cap penalty
    penalty = min(penalty, max_penalty)

    # Apply penalty (subtract from score)
    adjusted = score - penalty
    adjusted = _clamp(adjusted, Decimal("0"), Decimal("100"))

    return RankStabilityAdjustment(
        original_score=_quantize_score(score),
        adjusted_score=_quantize_score(adjusted),
        penalty_applied=_quantize_score(penalty),
        prior_rank=prior_rank,
        current_rank=current_rank,
        rank_change=rank_change,
        flags=flags,
    )


# =============================================================================
# 4. MULTI-TIMEFRAME SIGNAL BLENDING
# =============================================================================

def blend_timeframe_signals(
    short_signal: Optional[Decimal],
    medium_signal: Optional[Decimal],
    long_signal: Optional[Decimal],
    short_confidence: Decimal = Decimal("0.5"),
    medium_confidence: Decimal = Decimal("0.7"),
    long_confidence: Decimal = Decimal("0.6"),
    *,
    weights: Optional[Dict[str, Decimal]] = None,
) -> Tuple[Decimal, Decimal, List[str]]:
    """
    Blend signals from multiple time horizons with adaptive weighting.

    Combines short (20d), medium (60d), and long-term (120d) signals using
    confidence-weighted averaging. Missing signals are handled by re-normalizing
    weights among available signals.

    Args:
        short_signal: Short-term signal (20d window)
        medium_signal: Medium-term signal (60d window) - primary
        long_signal: Long-term signal (120d window) - trend confirmation
        short_confidence: Confidence for short signal
        medium_confidence: Confidence for medium signal
        long_confidence: Confidence for long signal
        weights: Optional custom weights (default: 20/50/30)

    Returns:
        Tuple of (blended_signal, blended_confidence, flags)
    """
    if weights is None:
        weights = TIMEFRAME_WEIGHTS.copy()

    flags = []
    available_signals = []

    # Collect available signals with their weights and confidences
    if short_signal is not None:
        s = _to_decimal(short_signal, Decimal("50"))
        c = _to_decimal(short_confidence, Decimal("0.5"))
        available_signals.append(("short", s, weights["short"], c))
    else:
        flags.append("short_signal_missing")

    if medium_signal is not None:
        s = _to_decimal(medium_signal, Decimal("50"))
        c = _to_decimal(medium_confidence, Decimal("0.7"))
        available_signals.append(("medium", s, weights["medium"], c))
    else:
        flags.append("medium_signal_missing")

    if long_signal is not None:
        s = _to_decimal(long_signal, Decimal("50"))
        c = _to_decimal(long_confidence, Decimal("0.6"))
        available_signals.append(("long", s, weights["long"], c))
    else:
        flags.append("long_signal_missing")

    # Handle missing signals
    if not available_signals:
        return Decimal("50"), Decimal("0.3"), ["all_signals_missing"]

    if len(available_signals) == 1:
        name, signal, _, conf = available_signals[0]
        flags.append(f"single_signal_{name}")
        return _quantize_score(signal), _quantize_weight(conf), flags

    # Re-normalize weights for available signals
    total_weight = sum(w for _, _, w, _ in available_signals)

    # Compute confidence-weighted blend
    blended_signal = Decimal("0")
    blended_confidence = Decimal("0")

    for name, signal, weight, conf in available_signals:
        normalized_weight = weight / total_weight
        # Weight by both base weight and confidence
        effective_weight = normalized_weight * conf
        blended_signal += signal * effective_weight
        blended_confidence += conf * normalized_weight

    # Re-normalize signal by total effective weight
    total_effective_weight = sum(
        (w / total_weight) * c for _, _, w, c in available_signals
    )
    if total_effective_weight > EPS:
        blended_signal = blended_signal / total_effective_weight

    flags.append(f"blended_{len(available_signals)}_signals")

    return _quantize_score(blended_signal), _quantize_weight(blended_confidence), flags


# =============================================================================
# 5. ASYMMETRIC INTERACTION BOUNDS
# =============================================================================

def apply_asymmetric_bounds(
    interaction_value: Decimal,
    *,
    positive_cap: Decimal = INTERACTION_POSITIVE_CAP,
    negative_floor: Decimal = INTERACTION_NEGATIVE_FLOOR,
) -> AsymmetricBounds:
    """
    Apply asymmetric bounds to interaction adjustments.

    Positive adjustments (upside) have a higher cap than negative adjustments
    (downside) have a floor. This allows more upside capture while maintaining
    defensive posture against risk signals.

    Rationale:
    - Negative signals (risks) are often more predictive of poor outcomes
    - Tighter bounds on negatives prevent over-reaction to temporary issues
    - Higher positive cap allows strong conviction positions

    Args:
        interaction_value: Raw interaction adjustment
        positive_cap: Maximum positive adjustment (default 3.5)
        negative_floor: Maximum negative adjustment (default -2.5)

    Returns:
        AsymmetricBounds with clamped value and diagnostics
    """
    original = _to_decimal(interaction_value, Decimal("0"))

    if original > Decimal("0"):
        clamped = min(original, positive_cap)
        was_capped = original > positive_cap
        cap_type = "positive" if was_capped else None
    else:
        clamped = max(original, negative_floor)
        was_capped = original < negative_floor
        cap_type = "negative" if was_capped else None

    return AsymmetricBounds(
        positive_cap=positive_cap,
        negative_floor=negative_floor,
        applied_value=_quantize_score(clamped),
        original_value=_quantize_score(original),
        was_capped=was_capped,
        cap_type=cap_type,
    )


# =============================================================================
# 6. REGIME-CONDITIONAL WEIGHT FLOORS
# =============================================================================

def apply_weight_floors(
    weights: Dict[str, Decimal],
    regime: str,
    *,
    floor_config: Optional[Dict[str, Dict[str, Decimal]]] = None,
) -> WeightFloorResult:
    """
    Apply regime-conditional minimum weight floors.

    Ensures critical components always have meaningful weight regardless of
    optimization or data quality. Different regimes have different priorities:
    - Bull: Clinical matters most, financials have lower floor
    - Bear/Recession: Financial health has highest floor
    - Volatility: Balanced floors with clinical emphasis

    Weight redistribution:
    - If a component is below its floor, it's raised to the floor
    - The excess weight is proportionally removed from other components
    - Total weights always sum to 1.0

    Args:
        weights: Current component weights (should sum to ~1.0)
        regime: Current market regime
        floor_config: Optional custom floor configuration

    Returns:
        WeightFloorResult with adjusted weights and diagnostics
    """
    if floor_config is None:
        floor_config = REGIME_WEIGHT_FLOORS

    # Get floors for this regime (default to NEUTRAL if unknown)
    regime_upper = regime.upper() if regime else "NEUTRAL"
    floors = floor_config.get(regime_upper, floor_config.get("NEUTRAL", {}))

    adjusted = {k: v for k, v in weights.items()}
    floors_applied = {}
    flags = []
    total_added = Decimal("0")

    # First pass: identify and apply floors
    for component, floor in floors.items():
        if component in adjusted:
            current = adjusted[component]
            if current < floor:
                added = floor - current
                adjusted[component] = floor
                floors_applied[component] = floor
                total_added += added
                flags.append(f"{component}_floor_applied")

    # Second pass: redistribute weight if needed
    if total_added > EPS:
        # Components not at floor share the burden
        unfloored_components = [
            k for k in adjusted
            if k not in floors_applied and adjusted[k] > EPS
        ]

        if unfloored_components:
            total_unfloored = sum(adjusted[k] for k in unfloored_components)
            if total_unfloored > total_added:
                # Reduce unfloored components proportionally
                reduction_factor = (total_unfloored - total_added) / total_unfloored
                for k in unfloored_components:
                    adjusted[k] = adjusted[k] * reduction_factor

        # Normalize to sum to 1.0
        total_weight = sum(adjusted.values())
        if total_weight > EPS and abs(total_weight - Decimal("1")) > EPS:
            adjusted = {k: v / total_weight for k, v in adjusted.items()}

    # Quantize final weights
    adjusted = {k: _quantize_weight(v) for k, v in adjusted.items()}

    return WeightFloorResult(
        original_weights={k: _quantize_weight(v) for k, v in weights.items()},
        adjusted_weights=adjusted,
        floors_applied=floors_applied,
        regime=regime_upper,
        weight_redistributed=_quantize_weight(total_added),
        flags=flags,
    )


# =============================================================================
# 7. DEFENSIVE OVERRIDE TRIGGERS
# =============================================================================

def evaluate_defensive_triggers(
    universe_stats: Dict[str, Any],
    *,
    trigger_config: Optional[Dict[str, Decimal]] = None,
) -> DefensiveOverrideResult:
    """
    Evaluate defensive override triggers based on universe statistics.

    Monitors aggregate universe health to detect conditions requiring
    defensive posture. Triggers include:
    - High severity ratio (many SEV2+ tickers)
    - Low average runway (cash crunch across universe)
    - High volatility ratio (market stress)
    - Low positive momentum ratio (bearish momentum)

    Defensive Postures:
    - NONE: Normal operation
    - LIGHT: Minor defensive bias (1 trigger)
    - MODERATE: Significant defensive adjustments (2 triggers)
    - HEAVY: Maximum defensive mode (3+ triggers)

    Args:
        universe_stats: Dict with severity_ratio, avg_runway_months,
                       high_vol_ratio, positive_momentum_ratio
        trigger_config: Optional custom trigger thresholds

    Returns:
        DefensiveOverrideResult with posture and diagnostics
    """
    if trigger_config is None:
        trigger_config = DEFENSIVE_TRIGGER_CONDITIONS

    triggers_hit = []
    trigger_values = {}
    adjustments = {}
    flags = []

    # Check each trigger condition
    severity_ratio = _to_decimal(universe_stats.get("severity_ratio"), Decimal("0"))
    avg_runway = _to_decimal(universe_stats.get("avg_runway_months"), Decimal("24"))
    high_vol_ratio = _to_decimal(universe_stats.get("high_vol_ratio"), Decimal("0"))
    positive_momentum_ratio = _to_decimal(
        universe_stats.get("positive_momentum_ratio"), Decimal("0.5")
    )

    trigger_values = {
        "severity_ratio": severity_ratio,
        "avg_runway_months": avg_runway,
        "high_vol_ratio": high_vol_ratio,
        "positive_momentum_ratio": positive_momentum_ratio,
    }

    # Evaluate triggers
    if severity_ratio > trigger_config["max_severity_ratio"]:
        triggers_hit.append("high_severity_ratio")
        adjustments["financial_weight_boost"] = Decimal("0.05")

    if avg_runway < trigger_config["min_avg_runway_months"]:
        triggers_hit.append("low_avg_runway")
        adjustments["runway_penalty_multiplier"] = Decimal("1.25")

    if high_vol_ratio > trigger_config["max_high_vol_ratio"]:
        triggers_hit.append("high_volatility_universe")
        adjustments["volatility_penalty_boost"] = Decimal("0.10")

    if positive_momentum_ratio < trigger_config["min_positive_momentum_ratio"]:
        triggers_hit.append("bearish_momentum")
        adjustments["momentum_weight_reduction"] = Decimal("0.30")

    # Determine posture based on trigger count
    num_triggers = len(triggers_hit)
    if num_triggers == 0:
        posture = DefensivePosture.NONE
    elif num_triggers == 1:
        posture = DefensivePosture.LIGHT
        flags.append("defensive_light_triggered")
    elif num_triggers == 2:
        posture = DefensivePosture.MODERATE
        flags.append("defensive_moderate_triggered")
    else:
        posture = DefensivePosture.HEAVY
        flags.append("defensive_heavy_triggered")

    return DefensiveOverrideResult(
        posture=posture,
        triggers_hit=triggers_hit,
        trigger_values={k: _quantize_weight(v) for k, v in trigger_values.items()},
        adjustments_applied={k: _quantize_weight(v) for k, v in adjustments.items()},
        flags=flags,
    )


# =============================================================================
# 8. SCORE DISTRIBUTION HEALTH CHECKS
# =============================================================================

def check_distribution_health(
    scores: List[Decimal],
    *,
    thresholds: Optional[Dict[str, Decimal]] = None,
) -> DistributionHealthCheck:
    """
    Check health of score distribution.

    Monitors for problematic score distributions that may indicate:
    - Data quality issues (too many zeros, NaNs)
    - Model miscalibration (excessive clustering or dispersion)
    - Normalization failures (extreme skewness)

    Args:
        scores: List of composite scores
        thresholds: Optional custom health thresholds

    Returns:
        DistributionHealthCheck with diagnostics and recommendations
    """
    if thresholds is None:
        thresholds = DISTRIBUTION_HEALTH_THRESHOLDS

    issues = []
    recommendations = []

    if not scores or len(scores) < 5:
        return DistributionHealthCheck(
            health=DistributionHealth.DEGRADED,
            mean=Decimal("0"),
            std=Decimal("0"),
            iqr=Decimal("0"),
            skewness=Decimal("0"),
            zero_ratio=Decimal("0"),
            score_range=Decimal("0"),
            issues=["insufficient_data"],
            recommendations=["increase_universe_size"],
        )

    # Convert to Decimal
    dec_scores = [_to_decimal(s, Decimal("0")) for s in scores]
    n = len(dec_scores)

    # Compute statistics
    mean = sum(dec_scores) / Decimal(n)

    variance = sum((s - mean) ** 2 for s in dec_scores) / Decimal(n)
    std = variance.sqrt() if variance > 0 else Decimal("0")

    # Sort for percentiles
    sorted_scores = sorted(dec_scores)
    q1_idx = n // 4
    q3_idx = (3 * n) // 4
    q1 = sorted_scores[q1_idx]
    q3 = sorted_scores[q3_idx]
    iqr = q3 - q1

    # Score range
    score_range = sorted_scores[-1] - sorted_scores[0]

    # Skewness (simplified Pearson's moment coefficient)
    if std > EPS:
        skew_sum = sum((s - mean) ** 3 for s in dec_scores)
        skewness = (skew_sum / Decimal(n)) / (std ** 3)
    else:
        skewness = Decimal("0")

    # Zero ratio
    zero_count = sum(1 for s in dec_scores if s == Decimal("0"))
    zero_ratio = Decimal(str(zero_count)) / Decimal(n)

    # Evaluate health
    if std < thresholds["min_std"]:
        issues.append("scores_too_clustered")
        recommendations.append("review_normalization_method")

    if std > thresholds["max_std"]:
        issues.append("scores_too_dispersed")
        recommendations.append("review_component_weights")

    if iqr < thresholds["min_iqr"]:
        issues.append("insufficient_differentiation")
        recommendations.append("add_discriminative_signals")

    if abs(skewness) > thresholds["max_skew"]:
        issues.append("distribution_skewed")
        if skewness > 0:
            recommendations.append("review_positive_bias_sources")
        else:
            recommendations.append("review_negative_bias_sources")

    if zero_ratio > thresholds["max_zero_ratio"]:
        issues.append("excessive_zeros")
        recommendations.append("review_data_quality_and_coverage")

    if score_range < thresholds["min_range"]:
        issues.append("insufficient_score_range")
        recommendations.append("calibrate_component_scaling")

    # Determine overall health
    if len(issues) == 0:
        health = DistributionHealth.HEALTHY
    elif len(issues) == 1:
        if "clustered" in issues[0]:
            health = DistributionHealth.CLUSTERED
        elif "dispersed" in issues[0]:
            health = DistributionHealth.DISPERSED
        elif "skewed" in issues[0]:
            health = DistributionHealth.SKEWED
        else:
            health = DistributionHealth.DEGRADED
    else:
        health = DistributionHealth.DEGRADED

    return DistributionHealthCheck(
        health=health,
        mean=_quantize_score(mean),
        std=_quantize_score(std),
        iqr=_quantize_score(iqr),
        skewness=_quantize_score(skewness),
        zero_ratio=_quantize_weight(zero_ratio),
        score_range=_quantize_score(score_range),
        issues=issues,
        recommendations=recommendations,
    )


# =============================================================================
# AGGREGATED ENHANCEMENT APPLICATION
# =============================================================================

def apply_robustness_enhancements(
    scores: Dict[str, Dict[str, Any]],
    regime: str,
    universe_stats: Dict[str, Any],
    prior_rankings: Optional[Dict[str, List[int]]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], RobustnessEnhancements]:
    """
    Apply all robustness enhancements to a set of scores.

    Orchestrates the full enhancement pipeline:
    1. Winsorize component scores
    2. Apply confidence shrinkage
    3. Compute rank stability penalties
    4. Apply asymmetric interaction bounds
    5. Apply regime-conditional weight floors
    6. Evaluate defensive triggers
    7. Check distribution health

    Args:
        scores: Dict of ticker -> score data (with components, confidence, etc.)
        regime: Current market regime
        universe_stats: Aggregate universe statistics
        prior_rankings: Optional dict of ticker -> list of prior ranks

    Returns:
        Tuple of (enhanced_scores, enhancement_summary)
    """
    if prior_rankings is None:
        prior_rankings = {}

    enhanced = {}
    flags = []

    winsor_count = 0
    shrinkage_count = 0
    total_rank_penalty = Decimal("0")
    interaction_caps = 0
    floor_triggers = 0
    total_adjustment = Decimal("0")

    # Collect all composite scores for distribution check
    all_composites = [
        _to_decimal(s.get("composite_score"), Decimal("50"))
        for s in scores.values()
    ]

    # Check distribution health
    dist_health = check_distribution_health(all_composites)

    # Evaluate defensive triggers
    defensive = evaluate_defensive_triggers(universe_stats)

    # Process each ticker
    for ticker, score_data in scores.items():
        enhanced_data = score_data.copy()
        ticker_flags = []

        # 1. Winsorize component scores
        for component in ["clinical", "financial", "catalyst"]:
            comp_key = f"{component}_normalized"
            if comp_key in enhanced_data:
                cohort_values = [
                    _to_decimal(s.get(comp_key), Decimal("50"))
                    for s in scores.values()
                ]
                winsor_result = winsorize_component_score(
                    enhanced_data[comp_key],
                    cohort_values,
                )
                if winsor_result.was_clipped:
                    winsor_count += 1
                    ticker_flags.append(f"{component}_winsorized")
                enhanced_data[comp_key] = winsor_result.winsorized

        # 2. Apply confidence shrinkage
        confidence = _to_decimal(enhanced_data.get("confidence_overall"), Decimal("0.5"))
        cohort_mean = sum(all_composites) / Decimal(len(all_composites)) if all_composites else Decimal("50")

        shrinkage = apply_confidence_shrinkage(
            enhanced_data.get("composite_score", Decimal("50")),
            confidence,
            cohort_mean,
        )
        if shrinkage.shrinkage_factor > EPS:
            shrinkage_count += 1
            ticker_flags.append("confidence_shrinkage_applied")
        enhanced_data["composite_score_pre_shrinkage"] = enhanced_data.get("composite_score")
        enhanced_data["composite_score"] = shrinkage.shrunk

        # 3. Apply asymmetric interaction bounds
        interaction_adj = _to_decimal(
            enhanced_data.get("interaction_terms", {}).get("total_adjustment"),
            Decimal("0")
        )
        bounds = apply_asymmetric_bounds(interaction_adj)
        if bounds.was_capped:
            interaction_caps += 1
            ticker_flags.append(f"interaction_{bounds.cap_type}_capped")

        # Update interaction adjustment
        if "interaction_terms" in enhanced_data:
            enhanced_data["interaction_terms"]["total_adjustment"] = str(bounds.applied_value)
            adjusted_score = _to_decimal(enhanced_data["composite_score"], Decimal("50"))
            adjusted_score += (bounds.applied_value - interaction_adj)
            enhanced_data["composite_score"] = _clamp(adjusted_score, Decimal("0"), Decimal("100"))

        # 4. Apply rank stability penalty
        prior_ranks = prior_rankings.get(ticker, [])
        current_rank = list(scores.keys()).index(ticker) + 1

        stability = compute_rank_stability_penalty(
            enhanced_data["composite_score"],
            current_rank,
            prior_ranks,
        )
        if stability.penalty_applied > EPS:
            total_rank_penalty += stability.penalty_applied
            ticker_flags.extend(stability.flags)
        enhanced_data["composite_score"] = stability.adjusted_score

        # Store ticker flags
        existing_flags = enhanced_data.get("flags", [])
        enhanced_data["flags"] = sorted(set(existing_flags + ticker_flags))

        enhanced[ticker] = enhanced_data

    # 5. Apply weight floors (aggregated check)
    sample_weights = next(
        (s.get("effective_weights", {}) for s in scores.values() if s.get("effective_weights")),
        {}
    )
    if sample_weights:
        floor_result = apply_weight_floors(sample_weights, regime)
        floor_triggers = len(floor_result.floors_applied)
        flags.extend(floor_result.flags)

    # Build enhancement summary
    summary = RobustnessEnhancements(
        winsorization_applied=winsor_count,
        shrinkage_adjustments=shrinkage_count,
        rank_stability_penalties=_quantize_score(total_rank_penalty),
        interaction_caps_applied=interaction_caps,
        weight_floors_triggered=floor_triggers,
        defensive_posture=defensive.posture,
        distribution_health=dist_health.health,
        total_adjustment=_quantize_score(total_adjustment),
        flags=flags + defensive.flags + dist_health.issues,
    )

    return enhanced, summary
