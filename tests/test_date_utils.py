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

import pytest
from datetime import date
from common.date_utils import (
    normalize_date,
    to_date_string,
    to_date_object,
    validate_as_of_date,
    DateLike,
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

    def test_normalize_date_with_time_suffix_raises(self):
        """Date string with time suffix should raise ValueError."""
        with pytest.raises(ValueError):
            normalize_date("2026-01-15T00:00:00")


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
        validate_as_of_date("2026-01-15")  # Should not raise

    def test_validate_as_of_date_valid_date(self):
        """Valid date object should not raise."""
        validate_as_of_date(date(2026, 1, 15))  # Should not raise

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
        validate_as_of_date("2099-12-31")  # Should not raise

    def test_validate_as_of_date_past_date_allowed(self):
        """Historical dates should be allowed."""
        validate_as_of_date("2020-01-01")  # Should not raise


class TestLeapYearHandling:
    """Tests specifically for leap year edge cases."""

    @pytest.mark.parametrize("year,is_leap", [
        (2020, True),   # Divisible by 4
        (2024, True),   # Divisible by 4
        (2025, False),  # Not divisible by 4
        (2100, False),  # Divisible by 100 but not 400
        (2000, True),   # Divisible by 400
    ])
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

    @pytest.mark.parametrize("month,max_day", [
        (1, 31), (2, 28), (3, 31), (4, 30), (5, 31), (6, 30),
        (7, 31), (8, 31), (9, 30), (10, 31), (11, 30), (12, 31),
    ])
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
