"""
common/score_utils.py - Score Manipulation Utilities

Provides standardized utilities for score manipulation:
- Bounds clamping (ensuring scores stay within valid range)
- Score normalization
- Score aggregation helpers

Design Philosophy:
- DECIMAL-ONLY: All score operations use Decimal for precision
- DETERMINISTIC: Same inputs always produce same outputs
- DEFENSIVE: Handles None/invalid values gracefully

Usage:
    from common.score_utils import clamp_score, normalize_to_range, aggregate_scores

    # Clamp a score to 0-100
    safe_score = clamp_score(raw_score)

    # Clamp with custom range
    safe_score = clamp_score(raw_score, min_val=Decimal("0"), max_val=Decimal("100"))

    # Normalize to 0-100 range
    normalized = normalize_to_range(value, input_min=0, input_max=1000)

Author: Wake Robin Capital Management
Version: 1.0.0
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, List, Optional, Dict, Tuple, Union

# Default precision for scores
SCORE_PRECISION = Decimal("0.01")
WEIGHT_PRECISION = Decimal("0.0001")

# Default score bounds
DEFAULT_MIN_SCORE = Decimal("0")
DEFAULT_MAX_SCORE = Decimal("100")

# Small epsilon for division safety
EPS = Decimal("0.000001")


# ============================================================================
# TYPE CONVERSION
# ============================================================================

def to_decimal(
    value: Any,
    default: Optional[Decimal] = None
) -> Optional[Decimal]:
    """
    Safely convert value to Decimal.

    Handles:
    - None -> default
    - Decimal -> pass through
    - int/float -> convert via string (for precision)
    - str -> parse (strips whitespace)

    Args:
        value: Value to convert
        default: Default value if conversion fails

    Returns:
        Decimal value or default
    """
    if value is None:
        return default

    try:
        if isinstance(value, Decimal):
            return value
        if isinstance(value, bool):
            # Prevent True -> Decimal("1")
            return default
        if isinstance(value, (int, float)):
            return Decimal(str(value))
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return default
            return Decimal(stripped)
        return default
    except (InvalidOperation, ValueError, TypeError):
        return default


def to_float_safe(value: Any, default: Optional[float] = None) -> Optional[float]:
    """
    Safely convert value to float for output.

    Args:
        value: Value to convert
        default: Default value if conversion fails

    Returns:
        float value or default
    """
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


# ============================================================================
# SCORE CLAMPING
# ============================================================================

def clamp_score(
    score: Union[Decimal, float, int, str, None],
    min_val: Union[Decimal, float, int] = DEFAULT_MIN_SCORE,
    max_val: Union[Decimal, float, int] = DEFAULT_MAX_SCORE,
    default: Optional[Decimal] = None,
) -> Optional[Decimal]:
    """
    Clamp score to specified range.

    Ensures scores stay within valid bounds (default 0-100).
    Handles None/invalid values gracefully.

    Args:
        score: Score to clamp (Decimal, float, int, str, or None)
        min_val: Minimum allowed value (default 0)
        max_val: Maximum allowed value (default 100)
        default: Value to return if score is None/invalid

    Returns:
        Clamped Decimal score or default if input is None/invalid

    Examples:
        >>> clamp_score(Decimal("150"))
        Decimal('100.00')
        >>> clamp_score(Decimal("-10"))
        Decimal('0.00')
        >>> clamp_score(None, default=Decimal("50"))
        Decimal('50')
    """
    # Convert to Decimal
    dec_score = to_decimal(score)
    if dec_score is None:
        return default

    dec_min = to_decimal(min_val, Decimal("0"))
    dec_max = to_decimal(max_val, Decimal("100"))

    # Clamp
    clamped = max(dec_min, min(dec_max, dec_score))

    # Quantize to standard precision
    return clamped.quantize(SCORE_PRECISION, rounding=ROUND_HALF_UP)


def clamp_weight(
    weight: Union[Decimal, float, int, str, None],
    min_val: Decimal = Decimal("0"),
    max_val: Decimal = Decimal("1"),
) -> Decimal:
    """
    Clamp weight to specified range (default 0-1).

    Args:
        weight: Weight to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Clamped Decimal weight
    """
    dec_weight = to_decimal(weight, Decimal("0"))
    clamped = max(min_val, min(max_val, dec_weight))
    return clamped.quantize(WEIGHT_PRECISION, rounding=ROUND_HALF_UP)


def clamp_in_place(
    scores: Dict[str, Any],
    score_key: str,
    min_val: Decimal = DEFAULT_MIN_SCORE,
    max_val: Decimal = DEFAULT_MAX_SCORE,
) -> None:
    """
    Clamp a score in a dict in place.

    Args:
        scores: Dictionary containing scores
        score_key: Key of score to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value
    """
    if score_key in scores and scores[score_key] is not None:
        scores[score_key] = clamp_score(scores[score_key], min_val, max_val)


# ============================================================================
# SCORE NORMALIZATION
# ============================================================================

def normalize_to_range(
    value: Union[Decimal, float, int, str, None],
    input_min: Union[Decimal, float, int],
    input_max: Union[Decimal, float, int],
    output_min: Decimal = DEFAULT_MIN_SCORE,
    output_max: Decimal = DEFAULT_MAX_SCORE,
    default: Optional[Decimal] = None,
) -> Optional[Decimal]:
    """
    Normalize value from input range to output range.

    Linear transformation: (value - input_min) / (input_max - input_min) * output_range + output_min

    Args:
        value: Value to normalize
        input_min: Minimum of input range
        input_max: Maximum of input range
        output_min: Minimum of output range (default 0)
        output_max: Maximum of output range (default 100)
        default: Value to return if input is None/invalid

    Returns:
        Normalized Decimal score or default

    Examples:
        >>> normalize_to_range(500, input_min=0, input_max=1000)
        Decimal('50.00')
        >>> normalize_to_range(0.75, input_min=0, input_max=1)
        Decimal('75.00')
    """
    dec_value = to_decimal(value)
    if dec_value is None:
        return default

    dec_in_min = to_decimal(input_min, Decimal("0"))
    dec_in_max = to_decimal(input_max, Decimal("100"))
    dec_out_min = to_decimal(output_min, Decimal("0"))
    dec_out_max = to_decimal(output_max, Decimal("100"))

    # Check for zero range
    input_range = dec_in_max - dec_in_min
    if abs(input_range) < EPS:
        return dec_out_min

    output_range = dec_out_max - dec_out_min

    # Normalize
    normalized = (dec_value - dec_in_min) / input_range * output_range + dec_out_min

    # Clamp to output range
    clamped = max(dec_out_min, min(dec_out_max, normalized))

    return clamped.quantize(SCORE_PRECISION, rounding=ROUND_HALF_UP)


def rank_normalize(
    values: List[Union[Decimal, float, int, None]],
    output_min: Decimal = DEFAULT_MIN_SCORE,
    output_max: Decimal = DEFAULT_MAX_SCORE,
) -> List[Optional[Decimal]]:
    """
    Convert values to percentile ranks within the list.

    Handles ties by assigning average rank.

    Args:
        values: List of values to rank-normalize
        output_min: Minimum of output range
        output_max: Maximum of output range

    Returns:
        List of rank-normalized scores (None values remain None)
    """
    # Convert to Decimal, tracking original indices
    dec_values = [(to_decimal(v), i) for i, v in enumerate(values)]

    # Separate None values
    valid = [(v, i) for v, i in dec_values if v is not None]
    n = len(valid)

    if n == 0:
        return [None] * len(values)

    if n == 1:
        result = [None] * len(values)
        result[valid[0][1]] = (output_min + output_max) / 2
        return result

    # Sort by value
    valid.sort(key=lambda x: x[0])

    # Assign ranks (handle ties with average rank)
    ranks = [Decimal("0")] * len(values)
    i = 0
    while i < n:
        j = i
        # Find all values equal to current
        while j < n and valid[j][0] == valid[i][0]:
            j += 1

        # Average rank for ties
        avg_rank = Decimal(str((i + j - 1) / 2))

        for k in range(i, j):
            original_idx = valid[k][1]
            # Convert rank to percentile
            pct = (avg_rank / Decimal(str(n - 1))) * (output_max - output_min) + output_min
            ranks[original_idx] = pct.quantize(SCORE_PRECISION, rounding=ROUND_HALF_UP)

        i = j

    # Set None values back to None
    result = []
    for i, (v, _) in enumerate(dec_values):
        if v is None:
            result.append(None)
        else:
            result.append(ranks[i])

    return result


# ============================================================================
# SCORE AGGREGATION
# ============================================================================

def weighted_average(
    scores: List[Optional[Decimal]],
    weights: List[Decimal],
    skip_none: bool = True,
) -> Tuple[Optional[Decimal], Decimal]:
    """
    Compute weighted average of scores.

    Args:
        scores: List of scores (may contain None)
        weights: List of weights (same length as scores)
        skip_none: If True, skip None values and renormalize weights

    Returns:
        Tuple of (weighted_average, total_weight_used)
    """
    if len(scores) != len(weights):
        raise ValueError(f"Length mismatch: {len(scores)} scores vs {len(weights)} weights")

    total_weight = Decimal("0")
    weighted_sum = Decimal("0")

    for score, weight in zip(scores, weights):
        if score is None:
            if not skip_none:
                return None, Decimal("0")
            continue

        weighted_sum += score * weight
        total_weight += weight

    if total_weight < EPS:
        return None, Decimal("0")

    avg = (weighted_sum / total_weight).quantize(SCORE_PRECISION, rounding=ROUND_HALF_UP)
    return avg, total_weight


def hybrid_aggregate(
    scores: Dict[str, Optional[Decimal]],
    weights: Dict[str, Decimal],
    critical_components: List[str],
    alpha: Decimal = Decimal("0.85"),
) -> Optional[Decimal]:
    """
    Compute hybrid weighted-sum + weakest-link aggregation.

    Combines:
    - Weighted sum (alpha weight)
    - Minimum of critical components (1-alpha weight)

    This prevents masking of critical failures.

    Args:
        scores: Dict of component name -> score
        weights: Dict of component name -> weight
        critical_components: List of component names that are critical
        alpha: Weight for weighted sum vs min (default 0.85)

    Returns:
        Hybrid aggregated score
    """
    # Compute weighted sum
    score_list = []
    weight_list = []

    for name, weight in weights.items():
        score_list.append(scores.get(name))
        weight_list.append(weight)

    weighted_avg, _ = weighted_average(score_list, weight_list)

    if weighted_avg is None:
        return None

    # Compute min of critical components
    critical_scores = [
        scores.get(c) for c in critical_components
        if scores.get(c) is not None
    ]

    if not critical_scores:
        # No critical scores available, use weighted average only
        return weighted_avg

    min_critical = min(critical_scores)

    # Hybrid aggregation
    hybrid = alpha * weighted_avg + (Decimal("1") - alpha) * min_critical
    return hybrid.quantize(SCORE_PRECISION, rounding=ROUND_HALF_UP)


# ============================================================================
# PENALTY APPLICATION
# ============================================================================

def apply_penalty(
    score: Optional[Decimal],
    penalty_pct: Union[Decimal, float],
    floor: Decimal = DEFAULT_MIN_SCORE,
) -> Optional[Decimal]:
    """
    Apply percentage penalty to score.

    new_score = score * (1 - penalty_pct)

    Args:
        score: Score to penalize
        penalty_pct: Penalty as decimal (0.10 = 10% penalty)
        floor: Minimum score after penalty

    Returns:
        Penalized score (clamped to floor)
    """
    if score is None:
        return None

    penalty = to_decimal(penalty_pct, Decimal("0"))
    multiplier = Decimal("1") - penalty
    multiplier = max(Decimal("0"), multiplier)  # Can't have negative multiplier

    penalized = score * multiplier
    return max(floor, penalized.quantize(SCORE_PRECISION, rounding=ROUND_HALF_UP))


def apply_multiplier(
    score: Optional[Decimal],
    multiplier: Union[Decimal, float],
    cap: Decimal = DEFAULT_MAX_SCORE,
) -> Optional[Decimal]:
    """
    Apply multiplier to score with cap.

    new_score = min(score * multiplier, cap)

    Args:
        score: Score to multiply
        multiplier: Multiplier value
        cap: Maximum score after multiplication

    Returns:
        Multiplied score (clamped to cap)
    """
    if score is None:
        return None

    mult = to_decimal(multiplier, Decimal("1"))
    result = score * mult
    return min(cap, result.quantize(SCORE_PRECISION, rounding=ROUND_HALF_UP))


# ============================================================================
# BATCH OPERATIONS
# ============================================================================

def clamp_all_scores(
    records: List[Dict[str, Any]],
    score_keys: List[str],
    min_val: Decimal = DEFAULT_MIN_SCORE,
    max_val: Decimal = DEFAULT_MAX_SCORE,
) -> List[Dict[str, Any]]:
    """
    Clamp specified score fields in all records.

    Args:
        records: List of record dicts
        score_keys: List of keys to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Records with clamped scores (modified in place)
    """
    for record in records:
        for key in score_keys:
            if key in record and record[key] is not None:
                record[key] = clamp_score(record[key], min_val, max_val)

    return records


def validate_score_bounds(
    score: Optional[Decimal],
    min_val: Decimal = DEFAULT_MIN_SCORE,
    max_val: Decimal = DEFAULT_MAX_SCORE,
    score_name: str = "score",
) -> Tuple[bool, Optional[str]]:
    """
    Validate that score is within bounds.

    Args:
        score: Score to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        score_name: Name for error message

    Returns:
        (is_valid, error_message)
    """
    if score is None:
        return True, None  # None is valid

    if score < min_val:
        return False, f"{score_name} {score} is below minimum {min_val}"

    if score > max_val:
        return False, f"{score_name} {score} is above maximum {max_val}"

    return True, None
