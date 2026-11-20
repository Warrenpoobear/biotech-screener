#!/usr/bin/env python3
"""
Tests for common/date_utils.py

Tests date normalization, conversion, and validation utilities.
Covers:
- ISO format parsing (valid and invalid)
- Round-trip consistency (str -> date -> str)
- Leap year handling
- Edge cases and error handling
"""

from datetime import date

import pytest

from common.date_utils import (
    DateLike,
    filing_deadline_13f,
    is_quarter_end,
    latest_completed_quarter_end,
    normalize_date,
    parse_quarter_end,
    prior_quarter_end,
    to_date_object,
    to_date_string,
    validate_as_of_date,
)


class TestNormalizeDate:
    """Tests for normalize_date function."""

    def test_normalize_date_from_date_object(self):
        """Date object should pass through unchanged."""
        d = date(2026, 1, 15)
        result = normalize_date(d)
        assert result == d
        assert isinstance(result, date)

    def test_normalize_date_from_iso_string(self):
        """ISO string should be parsed to date."""
        result = normalize_date("2026-01-15")
        assert result == date(2026, 1, 15)
        assert isinstance(result, date)

    def test_normalize_date_round_trip(self):
        """Converting str -> date -> str should preserve value."""
        original = "2026-06-30"
        d = normalize_date(original)
        back_to_str = d.isoformat()
        assert back_to_str == original

    def test_normalize_date_leap_year_feb_29(self):
        """Feb 29 on leap year should be valid."""
        result = normalize_date("2024-02-29")
        assert result == date(2024, 2, 29)

    def test_normalize_date_non_leap_year_feb_29_raises(self):
        """Feb 29 on non-leap year should raise ValueError."""
        with pytest.raises(ValueError):
            normalize_date("2025-02-29")

    def test_normalize_date_invalid_format_raises(self):
        """Non-ISO format should raise ValueError."""
        with pytest.raises(ValueError):
            normalize_date("01-15-2026")  # US format

    def test_normalize_date_invalid_month_raises(self):
        """Invalid month should raise ValueError."""
        with pytest.raises(ValueError):
            normalize_date("2026-13-01")  # Month 13

    def test_normalize_date_invalid_day_raises(self):
        """Invalid day should raise ValueError."""
        with pytest.raises(ValueError):
            normalize_date("2026-01-32")  # Day 32

    def test_normalize_date_empty_string_raises(self):
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError):
            normalize_date("")

    def test_normalize_date_none_raises(self):
        """None should raise TypeError."""
        with pytest.raises(TypeError):
            normalize_date(None)

    def test_normalize_date_integer_raises(self):
        """Integer should raise TypeError."""
        with pytest.raises(TypeError):
            normalize_date(20260115)

    def test_normalize_date_year_boundary_dec_31(self):
        """December 31 should be valid."""
        result = normalize_date("2025-12-31")
        assert result == date(2025, 12, 31)

    def test_normalize_date_year_boundary_jan_1(self):
        """January 1 should be valid."""
        result = normalize_date("2026-01-01")
        assert result == date(2026, 1, 1)

    def test_normalize_date_with_time_suffix_truncates(self):
        """Date string with time suffix should truncate to date portion."""
        result = normalize_date("2026-01-15T00:00:00")
        assert result == date(2026, 1, 15)

    def test_normalize_date_with_utc_suffix(self):
        """Datetime string with Z suffix should truncate to date."""
        result = normalize_date("2026-01-15T12:30:00Z")
        assert result == date(2026, 1, 15)


class TestNormalizeDateOrNone:
    """Tests for normalize_date_or_none function."""

    def test_valid_date(self):
        from common.date_utils import normalize_date_or_none

        assert normalize_date_or_none("2026-01-15") == date(2026, 1, 15)

    def test_none_returns_none(self):
        from common.date_utils import normalize_date_or_none

        assert normalize_date_or_none(None) is None

    def test_empty_string_returns_none(self):
        from common.date_utils import normalize_date_or_none

        assert normalize_date_or_none("") is None

    def test_invalid_returns_none(self):
        from common.date_utils import normalize_date_or_none

        assert normalize_date_or_none("not-a-date") is None

    def test_datetime_string_truncates(self):
        from common.date_utils import normalize_date_or_none

        assert normalize_date_or_none("2026-01-15T12:00:00Z") == date(2026, 1, 15)


class TestToDateString:
    """Tests for to_date_string function."""

    def test_to_date_string_from_date(self):
        """Date object should convert to ISO string."""
        d = date(2026, 1, 15)
        result = to_date_string(d)
        assert result == "2026-01-15"
        assert isinstance(result, str)

    def test_to_date_string_from_valid_string(self):
        """Valid ISO string should be returned unchanged."""
        result = to_date_string("2026-01-15")
        assert result == "2026-01-15"

    def test_to_date_string_from_invalid_string_raises(self):
        """Invalid date string should raise ValueError."""
        with pytest.raises(ValueError):
            to_date_string("not-a-date")

    def test_to_date_string_validates_string_format(self):
        """String with invalid date should raise ValueError."""
        with pytest.raises(ValueError):
            to_date_string("2026-02-30")  # Feb 30 doesn't exist

    def test_to_date_string_none_raises(self):
        """None should raise TypeError."""
        with pytest.raises(TypeError):
            to_date_string(None)

    def test_to_date_string_round_trip(self):
        """Converting date -> str -> date should preserve value."""
        original = date(2026, 3, 15)
        s = to_date_string(original)
        back_to_date = date.fromisoformat(s)
        assert back_to_date == original


class TestToDateObject:
    """Tests for to_date_object function (alias for normalize_date)."""

    def test_to_date_object_from_string(self):
        """String should convert to date object."""
        result = to_date_object("2026-01-15")
        assert result == date(2026, 1, 15)

    def test_to_date_object_from_date(self):
        """Date object should pass through."""
        d = date(2026, 1, 15)
        result = to_date_object(d)
        assert result == d

    def test_to_date_object_invalid_raises(self):
        """Invalid input should raise error."""
        with pytest.raises(ValueError):
            to_date_object("invalid")


class TestValidateAsOfDate:
    """Tests for validate_as_of_date function."""

    def test_validate_as_of_date_valid_string(self):
        """Valid ISO string should not raise."""
        assert validate_as_of_date("2026-01-15") is None  # returns None on success

    def test_validate_as_of_date_valid_date(self):
        """Valid date object should not raise."""
        assert validate_as_of_date(date(2026, 1, 15)) is None  # returns None on success

    def test_validate_as_of_date_invalid_string_raises(self):
        """Invalid string should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid as_of_date"):
            validate_as_of_date("not-a-date")

    def test_validate_as_of_date_none_raises(self):
        """None should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid as_of_date"):
            validate_as_of_date(None)

    def test_validate_as_of_date_future_date_allowed(self):
        """Future dates should be allowed (PIT enforced elsewhere)."""
        assert validate_as_of_date("2099-12-31") is None  # returns None on success

    def test_validate_as_of_date_past_date_allowed(self):
        """Historical dates should be allowed."""
        assert validate_as_of_date("2020-01-01") is None  # returns None on success


class TestLeapYearHandling:
    """Tests specifically for leap year edge cases."""

    @pytest.mark.parametrize(
        "year,is_leap",
        [
            (2020, True),  # Divisible by 4
            (2024, True),  # Divisible by 4
            (2025, False),  # Not divisible by 4
            (2100, False),  # Divisible by 100 but not 400
            (2000, True),  # Divisible by 400
        ],
    )
    def test_leap_year_detection(self, year, is_leap):
        """Verify leap year handling for Feb 28/29."""
        if is_leap:
            result = normalize_date(f"{year}-02-29")
            assert result == date(year, 2, 29)
        else:
            with pytest.raises(ValueError):
                normalize_date(f"{year}-02-29")

    def test_feb_28_always_valid(self):
        """Feb 28 is valid in all years."""
        for year in [2024, 2025, 2026]:
            result = normalize_date(f"{year}-02-28")
            assert result.day == 28


class TestDateBoundaries:
    """Tests for date boundary conditions."""

    @pytest.mark.parametrize(
        "month,max_day",
        [
            (1, 31),
            (2, 28),
            (3, 31),
            (4, 30),
            (5, 31),
            (6, 30),
            (7, 31),
            (8, 31),
            (9, 30),
            (10, 31),
            (11, 30),
            (12, 31),
        ],
    )
    def test_month_max_days_non_leap(self, month, max_day):
        """Each month's max day should be valid in non-leap year."""
        date_str = f"2025-{month:02d}-{max_day:02d}"
        result = normalize_date(date_str)
        assert result.day == max_day

    def test_min_date_value(self):
        """Minimum date should be parseable."""
        result = normalize_date("0001-01-01")
        assert result == date(1, 1, 1)

    def test_reasonable_future_date(self):
        """Reasonable future date should be valid."""
        result = normalize_date("2099-12-31")
        assert result == date(2099, 12, 31)


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_normalize_date_deterministic(self):
        """Same input should produce same output."""
        input_str = "2026-01-15"
        result1 = normalize_date(input_str)
        result2 = normalize_date(input_str)
        assert result1 == result2

    def test_to_date_string_deterministic(self):
        """Same date should produce same string."""
        d = date(2026, 1, 15)
        result1 = to_date_string(d)
        result2 = to_date_string(d)
        assert result1 == result2


class TestQuarterEnds:
    """Tests for quarter-end recognition and validation."""

    @pytest.mark.parametrize("value", ["2026-03-31", "2026-06-30", "2026-09-30", "2026-12-31"])
    def test_quarter_ends_recognised(self, value):
        """All four calendar quarter ends should be recognised."""
        assert is_quarter_end(value) is True
        assert parse_quarter_end(value) == value

    @pytest.mark.parametrize("value", ["2026-07-01", "2026-06-29", "2026-02-28", "2026-01-31"])
    def test_non_quarter_ends_rejected(self, value):
        """Dates that are not quarter ends should be rejected."""
        assert is_quarter_end(value) is False
        with pytest.raises(ValueError, match="not a quarter end"):
            parse_quarter_end(value)

    def test_quarter_end_accepts_date_object(self):
        """Quarter-end helpers accept date objects, not just strings."""
        assert parse_quarter_end(date(2026, 6, 30)) == "2026-06-30"


class TestLatestCompletedQuarterEnd:
    """Tests for latest_completed_quarter_end."""

    @pytest.mark.parametrize(
        "today,expected",
        [
            (date(2026, 8, 14), date(2026, 6, 30)),  # mid-Q3 -> Q2 just completed
            (date(2026, 7, 1), date(2026, 6, 30)),  # day after quarter end
            (date(2026, 1, 5), date(2025, 12, 31)),  # year rollover
            (date(2026, 4, 1), date(2026, 3, 31)),
        ],
    )
    def test_latest_completed(self, today, expected):
        assert latest_completed_quarter_end(today) == expected

    def test_quarter_end_itself_is_not_yet_completed(self):
        """On the quarter-end date the quarter is not yet complete."""
        assert latest_completed_quarter_end(date(2026, 6, 30)) == date(2026, 3, 31)


class TestPriorQuarterEnd:
    """Tests for prior_quarter_end."""

    @pytest.mark.parametrize(
        "quarter,expected",
        [
            ("2026-06-30", date(2026, 3, 31)),
            ("2026-03-31", date(2025, 12, 31)),  # crosses year boundary
            ("2026-12-31", date(2026, 9, 30)),
        ],
    )
    def test_prior_quarter(self, quarter, expected):
        assert prior_quarter_end(quarter) == expected

    def test_prior_quarter_rejects_non_quarter_end(self):
        with pytest.raises(ValueError, match="not a quarter end"):
            prior_quarter_end("2026-07-15")


class TestFilingDeadline13F:
    """Tests for the 45-day SEC 13F filing deadline."""

    @pytest.mark.parametrize(
        "quarter,expected",
        [
            ("2026-06-30", date(2026, 8, 14)),
            ("2026-03-31", date(2026, 5, 15)),
            ("2025-12-31", date(2026, 2, 14)),
            ("2026-09-30", date(2026, 11, 14)),
        ],
    )
    def test_deadline_is_45_days_after_quarter_end(self, quarter, expected):
        assert filing_deadline_13f(quarter) == expected

    def test_deadline_rejects_non_quarter_end(self):
        with pytest.raises(ValueError, match="not a quarter end"):
            filing_deadline_13f("2026-08-14")
