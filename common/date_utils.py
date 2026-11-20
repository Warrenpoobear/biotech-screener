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

from datetime import date, timedelta
from typing import Union

DateLike = Union[str, date]

# Calendar quarter ends as (month, day). 13F report_date always falls on one of these.
QUARTER_ENDS = ((3, 31), (6, 30), (9, 30), (12, 31))


def normalize_date(value: DateLike) -> date:
    """
    Normalize a date-like value to a date object.

    Handles datetime strings (e.g. "2024-01-15T12:00:00Z") by truncating
    to the date portion. This covers the common case where callers pass
    ISO datetime strings from JSON or API responses.

    Args:
        value: Either a date object or ISO-like date/datetime string

    Returns:
        date object

    Raises:
        ValueError: If string is not valid ISO format
        TypeError: If value is neither str nor date
    """
    if isinstance(value, date):
        return value
    elif isinstance(value, str):
        # Truncate datetime strings: "2024-01-15T12:00:00Z" → "2024-01-15"
        return date.fromisoformat(value[:10])
    else:
        raise TypeError(f"Expected str or date, got {type(value).__name__}")


def normalize_date_or_none(value) -> "date | None":
    """Lenient date normalization — returns None instead of raising.

    Use this for optional date fields where missing/malformed values should
    be treated as absent rather than crashing the caller.
    """
    if value is None:
        return None
    try:
        if isinstance(value, str) and value.strip() == "":
            return None
        return normalize_date(value)
    except (ValueError, TypeError):
        return None


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


def is_quarter_end(value: DateLike) -> bool:
    """
    Return True if value falls on a calendar quarter end.

    Args:
        value: Date to test

    Returns:
        True if the month/day is one of 03-31, 06-30, 09-30, 12-31
    """
    d = normalize_date(value)
    return (d.month, d.day) in QUARTER_ENDS


def parse_quarter_end(value: DateLike) -> str:
    """
    Validate that value is a calendar quarter end and return it as a string.

    Args:
        value: Date to validate

    Returns:
        ISO date string (YYYY-MM-DD)

    Raises:
        ValueError: If value is not a valid date or does not fall on a quarter end
    """
    d = normalize_date(value)
    if not is_quarter_end(d):
        expected = ", ".join(f"{m:02d}-{day:02d}" for m, day in QUARTER_ENDS)
        raise ValueError(f"{to_date_string(d)} is not a quarter end (expected month-day one of: {expected})")
    return to_date_string(d)


def latest_completed_quarter_end(today: "date | None" = None) -> date:
    """
    Return the most recent calendar quarter end strictly before today.

    A quarter is "completed" the day after it ends; this says nothing about
    whether filings for it are available yet.

    Args:
        today: Reference date (defaults to date.today())

    Returns:
        The latest quarter-end date before today
    """
    today = today or date.today()
    candidates = [date(today.year, m, day) for m, day in QUARTER_ENDS]
    candidates.append(date(today.year - 1, 12, 31))
    return max(d for d in candidates if d < today)


def prior_quarter_end(value: DateLike) -> date:
    """
    Return the quarter end immediately preceding the given quarter end.

    Args:
        value: A quarter-end date

    Returns:
        The preceding quarter-end date

    Raises:
        ValueError: If value is not a quarter end
    """
    d = normalize_date(parse_quarter_end(value))
    return latest_completed_quarter_end(d)


def filing_deadline_13f(value: DateLike) -> date:
    """
    Return the SEC 13F filing deadline for a quarter: 45 days after quarter end.

    Args:
        value: A quarter-end date

    Returns:
        The filing deadline date

    Raises:
        ValueError: If value is not a quarter end
    """
    d = normalize_date(parse_quarter_end(value))
    return d + timedelta(days=45)
