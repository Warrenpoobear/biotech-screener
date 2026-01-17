"""
common/null_safety.py - Null Safety Helpers

Provides standardized utilities for handling None/null values across the pipeline.
Prevents common null-related bugs through defensive programming patterns.

Design Philosophy:
- EXPLICIT: All null handling is explicit, no truthy/falsy shortcuts
- TYPE-SAFE: Preserves type information through conversions
- DEFENSIVE: Graceful fallbacks for all operations

Common Gotchas This Prevents:
- `if value:` failing for 0 or empty string (use `if value is not None:`)
- Division by zero (use `safe_divide`)
- Index out of bounds (use `safe_get_index`)
- Key errors on dict access (use `safe_get`)

Usage:
    from common.null_safety import (
        safe_get,
        safe_get_nested,
        safe_get_index,
        safe_divide,
        is_present,
        coalesce,
    )

    # Safe dict access with default
    value = safe_get(record, "key", default=0)

    # Safe nested access
    value = safe_get_nested(data, ["level1", "level2", "key"])

    # Safe division
    ratio = safe_divide(numerator, denominator, default=Decimal("0"))

Author: Wake Robin Capital Management
Version: 1.0.0
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, Sequence

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


# ============================================================================
# NULL CHECKS
# ============================================================================

def is_present(value: Any) -> bool:
    """
    Check if value is not None.

    USE THIS instead of `if value:` to avoid issues with:
    - 0 (int) being falsy
    - 0.0 (float) being falsy
    - Decimal("0") being falsy
    - Empty string "" being falsy (may or may not be desired)

    Args:
        value: Value to check

    Returns:
        True if value is not None

    Examples:
        >>> is_present(0)
        True
        >>> is_present(None)
        False
        >>> is_present(Decimal("0"))
        True
    """
    return value is not None


def is_missing(value: Any) -> bool:
    """
    Check if value is None.

    Inverse of is_present.

    Args:
        value: Value to check

    Returns:
        True if value is None
    """
    return value is None


def is_zero_or_none(value: Any) -> bool:
    """
    Check if value is None or zero.

    Useful for division safety checks.

    Args:
        value: Value to check

    Returns:
        True if value is None or evaluates to zero
    """
    if value is None:
        return True

    try:
        if isinstance(value, (int, float, Decimal)):
            return value == 0
        # Try numeric conversion
        return float(value) == 0
    except (ValueError, TypeError, InvalidOperation):
        return False


def is_empty_or_none(value: Any) -> bool:
    """
    Check if value is None or empty.

    Works for:
    - None
    - Empty string ""
    - Empty list []
    - Empty dict {}
    - Empty set set()

    Args:
        value: Value to check

    Returns:
        True if value is None or empty
    """
    if value is None:
        return True

    # Check for empty collections
    if isinstance(value, (str, list, dict, set, tuple)):
        return len(value) == 0

    return False


# ============================================================================
# SAFE ACCESS
# ============================================================================

def safe_get(
    mapping: Optional[Dict[K, V]],
    key: K,
    default: Optional[V] = None
) -> Optional[V]:
    """
    Safely get value from dict with explicit None handling.

    Args:
        mapping: Dictionary to access (may be None)
        key: Key to look up
        default: Default value if key missing or mapping is None

    Returns:
        Value at key or default

    Examples:
        >>> safe_get({"a": 1}, "a")
        1
        >>> safe_get({"a": 1}, "b", default=0)
        0
        >>> safe_get(None, "a", default=0)
        0
    """
    if mapping is None:
        return default

    return mapping.get(key, default)


def safe_get_nested(
    data: Optional[Dict],
    keys: Sequence[str],
    default: Any = None
) -> Any:
    """
    Safely get nested value from dict.

    Args:
        data: Nested dictionary structure
        keys: Sequence of keys to traverse
        default: Default value if any key missing

    Returns:
        Value at nested path or default

    Examples:
        >>> safe_get_nested({"a": {"b": {"c": 1}}}, ["a", "b", "c"])
        1
        >>> safe_get_nested({"a": {"b": {}}}, ["a", "b", "c"], default=0)
        0
    """
    if data is None:
        return default

    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        if key not in current:
            return default
        current = current[key]

    return current


def safe_get_index(
    sequence: Optional[Sequence[T]],
    index: int,
    default: Optional[T] = None
) -> Optional[T]:
    """
    Safely get value from sequence by index.

    Args:
        sequence: Sequence to access (may be None)
        index: Index to look up (supports negative indices)
        default: Default value if index out of bounds

    Returns:
        Value at index or default

    Examples:
        >>> safe_get_index([1, 2, 3], 0)
        1
        >>> safe_get_index([1, 2, 3], 10, default=0)
        0
        >>> safe_get_index(None, 0, default=0)
        0
    """
    if sequence is None:
        return default

    try:
        return sequence[index]
    except (IndexError, TypeError):
        return default


def safe_get_first(
    sequence: Optional[Sequence[T]],
    default: Optional[T] = None
) -> Optional[T]:
    """
    Safely get first element of sequence.

    Args:
        sequence: Sequence to access
        default: Default if sequence is None or empty

    Returns:
        First element or default
    """
    return safe_get_index(sequence, 0, default)


def safe_get_last(
    sequence: Optional[Sequence[T]],
    default: Optional[T] = None
) -> Optional[T]:
    """
    Safely get last element of sequence.

    Args:
        sequence: Sequence to access
        default: Default if sequence is None or empty

    Returns:
        Last element or default
    """
    return safe_get_index(sequence, -1, default)


# ============================================================================
# COALESCE / DEFAULT VALUES
# ============================================================================

def coalesce(*values: Optional[T]) -> Optional[T]:
    """
    Return first non-None value.

    Similar to SQL COALESCE or ?? operator in other languages.

    Args:
        *values: Values to check in order

    Returns:
        First non-None value, or None if all are None

    Examples:
        >>> coalesce(None, None, 1, 2)
        1
        >>> coalesce(None, None)
        None
    """
    for value in values:
        if value is not None:
            return value
    return None


def default_if_none(value: Optional[T], default: T) -> T:
    """
    Return default if value is None.

    Type-safe version that guarantees non-None return.

    Args:
        value: Value to check
        default: Default to use if value is None

    Returns:
        Value if not None, else default
    """
    return default if value is None else value


def default_if_empty(
    value: Optional[Union[str, List, Dict]],
    default: T
) -> Union[str, List, Dict, T]:
    """
    Return default if value is None or empty.

    Args:
        value: Value to check (string, list, or dict)
        default: Default to use if value is None or empty

    Returns:
        Value if not None/empty, else default
    """
    if is_empty_or_none(value):
        return default
    return value


# ============================================================================
# SAFE ARITHMETIC
# ============================================================================

def safe_divide(
    numerator: Optional[Union[Decimal, float, int]],
    denominator: Optional[Union[Decimal, float, int]],
    default: Optional[Decimal] = None,
    epsilon: Decimal = Decimal("0.000001"),
) -> Optional[Decimal]:
    """
    Safely divide with zero/None handling.

    Args:
        numerator: Division numerator
        denominator: Division denominator
        default: Value to return if division not possible
        epsilon: Values with abs < epsilon treated as zero

    Returns:
        Division result or default

    Examples:
        >>> safe_divide(Decimal("10"), Decimal("2"))
        Decimal('5')
        >>> safe_divide(Decimal("10"), Decimal("0"))
        None
        >>> safe_divide(None, Decimal("2"))
        None
    """
    if numerator is None or denominator is None:
        return default

    try:
        num = Decimal(str(numerator))
        denom = Decimal(str(denominator))

        if abs(denom) < epsilon:
            return default

        return num / denom
    except (InvalidOperation, ValueError, TypeError):
        return default


def safe_multiply(
    *values: Optional[Union[Decimal, float, int]],
    default: Optional[Decimal] = None
) -> Optional[Decimal]:
    """
    Safely multiply multiple values.

    Returns default if any value is None.

    Args:
        *values: Values to multiply
        default: Value to return if any value is None

    Returns:
        Product of values or default
    """
    if not values:
        return default

    result = Decimal("1")

    for value in values:
        if value is None:
            return default
        try:
            result *= Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            return default

    return result


def safe_sum(
    values: Sequence[Optional[Union[Decimal, float, int]]],
    default: Optional[Decimal] = Decimal("0"),
    skip_none: bool = True,
) -> Optional[Decimal]:
    """
    Safely sum values.

    Args:
        values: Values to sum
        default: Default if result would be None
        skip_none: If True, skip None values; if False, return default on any None

    Returns:
        Sum of values or default
    """
    if not values:
        return default

    result = Decimal("0")

    for value in values:
        if value is None:
            if not skip_none:
                return default
            continue

        try:
            result += Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            if not skip_none:
                return default

    return result


# ============================================================================
# SAFE CONVERSION
# ============================================================================

def safe_int(
    value: Any,
    default: Optional[int] = None
) -> Optional[int]:
    """
    Safely convert value to int.

    Args:
        value: Value to convert
        default: Default if conversion fails

    Returns:
        Integer value or default
    """
    if value is None:
        return default

    try:
        # Handle Decimal specially to avoid float precision issues
        if isinstance(value, Decimal):
            return int(value)
        return int(value)
    except (ValueError, TypeError, InvalidOperation):
        return default


def safe_float(
    value: Any,
    default: Optional[float] = None
) -> Optional[float]:
    """
    Safely convert value to float.

    Args:
        value: Value to convert
        default: Default if conversion fails

    Returns:
        Float value or default
    """
    if value is None:
        return default

    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_decimal(
    value: Any,
    default: Optional[Decimal] = None
) -> Optional[Decimal]:
    """
    Safely convert value to Decimal.

    Args:
        value: Value to convert
        default: Default if conversion fails

    Returns:
        Decimal value or default
    """
    if value is None:
        return default

    try:
        if isinstance(value, Decimal):
            return value
        if isinstance(value, bool):
            return default  # Prevent True -> Decimal("1")
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return default


def safe_str(
    value: Any,
    default: Optional[str] = None
) -> Optional[str]:
    """
    Safely convert value to string.

    Args:
        value: Value to convert
        default: Default if value is None

    Returns:
        String value or default
    """
    if value is None:
        return default

    return str(value)


# ============================================================================
# CONDITIONAL EXECUTION
# ============================================================================

def if_present(
    value: Optional[T],
    func: Callable[[T], V],
    default: Optional[V] = None
) -> Optional[V]:
    """
    Apply function if value is not None.

    Args:
        value: Value to check
        func: Function to apply if value is not None
        default: Default if value is None

    Returns:
        Result of func(value) or default

    Examples:
        >>> if_present(5, lambda x: x * 2)
        10
        >>> if_present(None, lambda x: x * 2, default=0)
        0
    """
    if value is None:
        return default
    return func(value)


def map_if_present(
    values: Sequence[Optional[T]],
    func: Callable[[T], V],
) -> List[Optional[V]]:
    """
    Apply function to each non-None value in sequence.

    Args:
        values: Sequence of values (may contain None)
        func: Function to apply to non-None values

    Returns:
        List with func applied to non-None values (None preserved)

    Examples:
        >>> map_if_present([1, None, 3], lambda x: x * 2)
        [2, None, 6]
    """
    return [func(v) if v is not None else None for v in values]


# ============================================================================
# COLLECTION SAFETY
# ============================================================================

def ensure_list(value: Optional[Union[T, List[T]]]) -> List[T]:
    """
    Ensure value is a list.

    Args:
        value: Single value, list, or None

    Returns:
        Empty list if None, [value] if single, list if already list
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def ensure_dict(value: Optional[Dict[K, V]]) -> Dict[K, V]:
    """
    Ensure value is a dict.

    Args:
        value: Dict or None

    Returns:
        Empty dict if None, value if dict
    """
    return {} if value is None else value


def filter_none(values: Sequence[Optional[T]]) -> List[T]:
    """
    Remove None values from sequence.

    Args:
        values: Sequence potentially containing None

    Returns:
        List with None values removed
    """
    return [v for v in values if v is not None]


def count_none(values: Sequence[Optional[Any]]) -> int:
    """
    Count None values in sequence.

    Args:
        values: Sequence to count

    Returns:
        Number of None values
    """
    return sum(1 for v in values if v is None)


def count_present(values: Sequence[Optional[Any]]) -> int:
    """
    Count non-None values in sequence.

    Args:
        values: Sequence to count

    Returns:
        Number of non-None values
    """
    return sum(1 for v in values if v is not None)
