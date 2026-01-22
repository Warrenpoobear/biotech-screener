#!/usr/bin/env python3
"""
Tests for common/pit_enforcement.py

Point-in-Time (PIT) enforcement is CRITICAL to this codebase's determinism
and prevention of look-ahead bias. These tests cover:
- compute_pit_cutoff() date arithmetic
- is_pit_admissible() date comparison logic
- filter_pit_admissible() bulk filtering
- Edge cases: leap years, year boundaries, None handling
"""

import pytest
from datetime import date
from common.pit_enforcement import (
    compute_pit_cutoff,
    is_pit_admissible,
    filter_pit_admissible,
)


class TestComputePitCutoff:
    """Tests for compute_pit_cutoff function."""

    def test_basic_cutoff(self):
        """Cutoff should be as_of_date - 1 day."""
        assert compute_pit_cutoff("2026-01-15") == "2026-01-14"

    def test_first_of_month(self):
        """First day of month should roll back to previous month."""
        assert compute_pit_cutoff("2026-02-01") == "2026-01-31"

    def test_first_of_year(self):
        """First day of year should roll back to Dec 31 of previous year."""
        assert compute_pit_cutoff("2026-01-01") == "2025-12-31"

    def test_leap_year_march_1(self):
        """March 1 in leap year should roll back to Feb 29."""
        assert compute_pit_cutoff("2024-03-01") == "2024-02-29"

    def test_non_leap_year_march_1(self):
        """March 1 in non-leap year should roll back to Feb 28."""
        assert compute_pit_cutoff("2025-03-01") == "2025-02-28"

    def test_leap_year_feb_29(self):
        """Feb 29 should roll back to Feb 28."""
        assert compute_pit_cutoff("2024-02-29") == "2024-02-28"

    def test_year_boundary_crossing(self):
        """Year boundary crossing should work correctly."""
        assert compute_pit_cutoff("2030-01-01") == "2029-12-31"

    def test_mid_month(self):
        """Mid-month date should work correctly."""
        assert compute_pit_cutoff("2026-07-15") == "2026-07-14"

    def test_end_of_month(self):
        """End of month should roll back to previous day."""
        assert compute_pit_cutoff("2026-03-31") == "2026-03-30"

    def test_determinism(self):
        """Same input should always produce same output."""
        result1 = compute_pit_cutoff("2026-01-15")
        result2 = compute_pit_cutoff("2026-01-15")
        assert result1 == result2

    def test_invalid_date_format_raises(self):
        """Invalid date format should raise ValueError."""
        with pytest.raises(ValueError):
            compute_pit_cutoff("01-15-2026")

    def test_invalid_date_value_raises(self):
        """Invalid date value should raise ValueError."""
        with pytest.raises(ValueError):
            compute_pit_cutoff("2026-02-30")


class TestIsPitAdmissible:
    """Tests for is_pit_admissible function."""

    def test_same_day_is_admissible(self):
        """Source date equal to cutoff should be admissible."""
        assert is_pit_admissible("2026-01-14", "2026-01-14") is True

    def test_earlier_date_is_admissible(self):
        """Source date before cutoff should be admissible."""
        assert is_pit_admissible("2026-01-10", "2026-01-14") is True

    def test_much_earlier_date_is_admissible(self):
        """Source date much earlier should be admissible."""
        assert is_pit_admissible("2020-01-01", "2026-01-14") is True

    def test_one_day_after_is_not_admissible(self):
        """Source date one day after cutoff should NOT be admissible."""
        assert is_pit_admissible("2026-01-15", "2026-01-14") is False

    def test_later_date_is_not_admissible(self):
        """Source date after cutoff should NOT be admissible."""
        assert is_pit_admissible("2026-02-01", "2026-01-14") is False

    def test_none_source_date_is_not_admissible(self):
        """None source date should NOT be admissible."""
        assert is_pit_admissible(None, "2026-01-14") is False

    def test_empty_string_source_date_is_not_admissible(self):
        """Empty string source date should NOT be admissible."""
        assert is_pit_admissible("", "2026-01-14") is False

    def test_invalid_source_date_format_is_not_admissible(self):
        """Invalid source date format should NOT be admissible."""
        assert is_pit_admissible("not-a-date", "2026-01-14") is False

    def test_datetime_string_truncated_to_date(self):
        """Datetime string should be truncated to date portion."""
        # is_pit_admissible uses source_date[:10] to handle datetime strings
        assert is_pit_admissible("2026-01-14T12:30:45Z", "2026-01-14") is True
        assert is_pit_admissible("2026-01-15T00:00:00", "2026-01-14") is False

    def test_year_boundary_admissibility(self):
        """Year boundary should be handled correctly."""
        assert is_pit_admissible("2025-12-31", "2025-12-31") is True
        assert is_pit_admissible("2025-12-31", "2026-01-01") is True
        assert is_pit_admissible("2026-01-01", "2025-12-31") is False

    def test_leap_year_admissibility(self):
        """Leap year dates should be handled correctly."""
        assert is_pit_admissible("2024-02-29", "2024-02-29") is True
        assert is_pit_admissible("2024-02-28", "2024-02-29") is True
        assert is_pit_admissible("2024-03-01", "2024-02-29") is False


class TestFilterPitAdmissible:
    """Tests for filter_pit_admissible function."""

    def test_filters_future_records(self):
        """Records with future dates should be filtered out."""
        records = [
            {"id": 1, "source_date": "2026-01-10"},  # Admissible
            {"id": 2, "source_date": "2026-01-14"},  # Admissible (boundary)
            {"id": 3, "source_date": "2026-01-15"},  # NOT admissible
            {"id": 4, "source_date": "2026-01-20"},  # NOT admissible
        ]
        result = filter_pit_admissible(records, "source_date", "2026-01-14")
        assert len(result) == 2
        assert [r["id"] for r in result] == [1, 2]

    def test_filters_none_dates(self):
        """Records with None dates should be filtered out."""
        records = [
            {"id": 1, "source_date": "2026-01-10"},
            {"id": 2, "source_date": None},
            {"id": 3, "source_date": "2026-01-12"},
        ]
        result = filter_pit_admissible(records, "source_date", "2026-01-14")
        assert len(result) == 2
        assert [r["id"] for r in result] == [1, 3]

    def test_filters_missing_date_field(self):
        """Records missing the date field should be filtered out."""
        records = [
            {"id": 1, "source_date": "2026-01-10"},
            {"id": 2, "other_field": "value"},  # Missing source_date
            {"id": 3, "source_date": "2026-01-12"},
        ]
        result = filter_pit_admissible(records, "source_date", "2026-01-14")
        assert len(result) == 2
        assert [r["id"] for r in result] == [1, 3]

    def test_empty_list_returns_empty(self):
        """Empty input list should return empty list."""
        result = filter_pit_admissible([], "source_date", "2026-01-14")
        assert result == []

    def test_all_admissible_returns_all(self):
        """If all records admissible, all should be returned."""
        records = [
            {"id": 1, "source_date": "2026-01-10"},
            {"id": 2, "source_date": "2026-01-11"},
            {"id": 3, "source_date": "2026-01-12"},
        ]
        result = filter_pit_admissible(records, "source_date", "2026-01-14")
        assert len(result) == 3

    def test_all_not_admissible_returns_empty(self):
        """If no records admissible, empty list should be returned."""
        records = [
            {"id": 1, "source_date": "2026-01-20"},
            {"id": 2, "source_date": "2026-01-25"},
            {"id": 3, "source_date": "2026-02-01"},
        ]
        result = filter_pit_admissible(records, "source_date", "2026-01-14")
        assert result == []

    def test_custom_date_field(self):
        """Should work with different date field names."""
        records = [
            {"id": 1, "data_date": "2026-01-10"},
            {"id": 2, "data_date": "2026-01-20"},
        ]
        result = filter_pit_admissible(records, "data_date", "2026-01-14")
        assert len(result) == 1
        assert result[0]["id"] == 1

    def test_preserves_record_content(self):
        """Filtered records should preserve all original content."""
        records = [
            {"id": 1, "source_date": "2026-01-10", "ticker": "ACME", "value": 100},
        ]
        result = filter_pit_admissible(records, "source_date", "2026-01-14")
        assert result[0] == records[0]

    def test_does_not_modify_input(self):
        """Original list should not be modified."""
        records = [
            {"id": 1, "source_date": "2026-01-10"},
            {"id": 2, "source_date": "2026-01-20"},
        ]
        original_len = len(records)
        _ = filter_pit_admissible(records, "source_date", "2026-01-14")
        assert len(records) == original_len


class TestPitEnforcementIntegration:
    """Integration tests for PIT enforcement workflow."""

    def test_typical_workflow(self):
        """Test typical PIT filtering workflow."""
        as_of_date = "2026-01-15"
        pit_cutoff = compute_pit_cutoff(as_of_date)

        assert pit_cutoff == "2026-01-14"

        records = [
            {"ticker": "ACME", "last_update": "2026-01-10"},  # OK
            {"ticker": "BETA", "last_update": "2026-01-14"},  # OK (boundary)
            {"ticker": "GAMMA", "last_update": "2026-01-15"},  # FILTERED
        ]

        filtered = filter_pit_admissible(records, "last_update", pit_cutoff)

        assert len(filtered) == 2
        tickers = [r["ticker"] for r in filtered]
        assert "ACME" in tickers
        assert "BETA" in tickers
        assert "GAMMA" not in tickers

    def test_quarterly_data_cutoff(self):
        """Test filtering quarterly financial data."""
        # Typical scenario: running on Jan 15, 2026
        # Should include Q3 2025 data filed by Dec 31
        # Should exclude any Q4 data filed in January
        as_of_date = "2026-01-15"
        pit_cutoff = compute_pit_cutoff(as_of_date)

        records = [
            {"ticker": "A", "filing_date": "2025-11-15", "quarter": "Q3"},  # OK
            {"ticker": "B", "filing_date": "2025-12-30", "quarter": "Q3"},  # OK
            {"ticker": "C", "filing_date": "2026-01-10", "quarter": "Q4"},  # OK
            {"ticker": "D", "filing_date": "2026-01-14", "quarter": "Q4"},  # OK (boundary)
            {"ticker": "E", "filing_date": "2026-01-15", "quarter": "Q4"},  # FILTERED
        ]

        filtered = filter_pit_admissible(records, "filing_date", pit_cutoff)

        assert len(filtered) == 4
        assert all(r["ticker"] in ["A", "B", "C", "D"] for r in filtered)

    def test_ctgov_trial_update_cutoff(self):
        """Test filtering CT.gov trial updates."""
        as_of_date = "2026-01-15"
        pit_cutoff = compute_pit_cutoff(as_of_date)

        trials = [
            {"nct_id": "NCT001", "last_update_posted": "2025-12-01", "status": "Recruiting"},
            {"nct_id": "NCT002", "last_update_posted": "2026-01-10", "status": "Completed"},
            {"nct_id": "NCT003", "last_update_posted": "2026-01-20", "status": "Terminated"},  # Future
        ]

        filtered = filter_pit_admissible(trials, "last_update_posted", pit_cutoff)

        assert len(filtered) == 2
        assert all(t["nct_id"] in ["NCT001", "NCT002"] for t in filtered)


class TestEdgeCases:
    """Edge case tests for PIT enforcement."""

    def test_century_boundary(self):
        """Test dates around century boundary (historical)."""
        assert compute_pit_cutoff("2000-01-01") == "1999-12-31"
        assert is_pit_admissible("1999-12-31", "1999-12-31") is True

    def test_very_old_dates(self):
        """Test with very old dates."""
        assert is_pit_admissible("1990-01-01", "2026-01-14") is True

    def test_far_future_dates(self):
        """Test with far future dates."""
        assert is_pit_admissible("2050-01-01", "2026-01-14") is False

    def test_malformed_date_gracefully_handled(self):
        """Malformed dates should return False, not raise."""
        # These should all return False gracefully
        assert is_pit_admissible("2026-13-01", "2026-01-14") is False  # Invalid month
        assert is_pit_admissible("2026-01-32", "2026-01-14") is False  # Invalid day
        assert is_pit_admissible("abc", "2026-01-14") is False  # Non-date
        assert is_pit_admissible(123, "2026-01-14") is False  # Wrong type
