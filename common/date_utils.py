"""
common/date_utils.py - Date normalization utilities

Provides consistent date handling across all modules to avoid type mismatches.

Usage:
    from common.date_utils import normalize_date, to_date_string, to_date_object

    # Normalize to date object
    d = normalize_date("2024-01-15")  # Returns date(2024, 1, 15)
    d = normalize_date(date(2024, 1, 15))  # Returns date(2024, 1, 15)

    # Get string representation
    s = to_date_string("2024-01-15")  # Returns "2024-01-15"
    s = to_date_string(date(2024, 1, 15))  # Returns "2024-01-15"

    # Get date object
    d = to_date_object("2024-01-15")  # Returns date(2024, 1, 15)
"""

from datetime import date
from typing import Union

DateLike = Union[str, date]


def normalize_date(value: DateLike) -> date:
    """
    Normalize a date-like value to a date object.

    Args:
        value: Either a date object or ISO format string (YYYY-MM-DD)

    Returns:
        date object

    Raises:
        ValueError: If string is not valid ISO format
        TypeError: If value is neither str nor date
    """
    if isinstance(value, date):
        return value
    elif isinstance(value, str):
        return date.fromisoformat(value)
    else:
        raise TypeError(f"Expected str or date, got {type(value).__name__}")


def to_date_string(value: DateLike) -> str:
    """
    Convert a date-like value to ISO format string.

    Args:
        value: Either a date object or ISO format string

    Returns:
        ISO format date string (YYYY-MM-DD)
    """
    if isinstance(value, str):
        # Validate and return
        date.fromisoformat(value)
        return value
    elif isinstance(value, date):
        return value.isoformat()
    else:
        raise TypeError(f"Expected str or date, got {type(value).__name__}")


def to_date_object(value: DateLike) -> date:
    """
    Alias for normalize_date for semantic clarity.

    Args:
        value: Either a date object or ISO format string

    Returns:
        date object
    """
    return normalize_date(value)


def validate_as_of_date(value: DateLike) -> None:
    """
    Validate that as_of_date is a valid date.

    Does not compare to current date (to maintain time-invariance).
    Lookahead protection should be enforced via PIT filters.

    Args:
        value: Date to validate

    Raises:
        ValueError: If date is invalid
    """
    try:
        normalize_date(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid as_of_date: {e}") from e
