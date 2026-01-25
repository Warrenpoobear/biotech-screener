"""
Tests for production hardening patches.

These tests verify that the patches fix the identified issues
and serve as regression tests for future changes.

Run: pytest tests/test_patches.py -v
"""

import json
import pytest
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, Any
import tempfile


# =============================================================================
# PATCH 001: PIT-Safe IC Validation Tests
# =============================================================================

class TestPITSafeICValidation:
    """Tests for patch_001_pit_safe_ic_validation.py"""

    def test_compute_pit_cutoff(self):
        """PIT cutoff should be as_of_date - 1"""
        from patches.patch_001_pit_safe_ic_validation import PITSafeICValidator

        validator = PITSafeICValidator(data_dir=Path('.'))
        cutoff = validator._compute_pit_cutoff('2026-01-15')

        assert cutoff == '2026-01-14', "PIT cutoff should be day before as_of_date"

    def test_lookback_window_ends_before_forward_horizon(self):
        """Lookback window must end before forward horizon to avoid look-ahead"""
        from patches.patch_001_pit_safe_ic_validation import PITSafeICValidator

        validator = PITSafeICValidator(
            data_dir=Path('.'),
            forward_horizon_days=21,
        )
        start, end = validator._compute_lookback_window('2026-01-15')

        # End date should be at least forward_horizon days before PIT cutoff
        end_date = date.fromisoformat(end)
        pit_cutoff = date.fromisoformat('2026-01-14')

        days_gap = (pit_cutoff - end_date).days
        assert days_gap >= 21, f"Gap {days_gap} should be >= forward_horizon (21)"

    def test_validation_fails_with_insufficient_periods(self):
        """Validation should fail if not enough historical IC periods"""
        from patches.patch_001_pit_safe_ic_validation import PITSafeICValidator

        # Create temp directory without ic_history.json
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = PITSafeICValidator(data_dir=Path(tmpdir))
            result = validator.validate_momentum_signal('2026-01-15')

            assert not result.is_significant
            assert not result.passes_threshold
            assert "Insufficient" in result.message


# =============================================================================
# PATCH 002: Deterministic Collection Tests
# =============================================================================

class TestDeterministicCollection:
    """Tests for patch_002_deterministic_collection.py"""

    def test_validate_collection_date_rejects_future(self):
        """Cannot collect data for future dates"""
        from patches.patch_002_deterministic_collection import validate_collection_date

        # Future date should be rejected
        future_date = (date.today() + timedelta(days=7)).isoformat()
        is_valid, message = validate_collection_date(future_date)

        assert not is_valid
        assert "future" in message.lower()

    def test_validate_collection_date_allows_today(self):
        """Today is valid for collection"""
        from patches.patch_002_deterministic_collection import validate_collection_date

        today = date.today().isoformat()
        is_valid, message = validate_collection_date(today)

        assert is_valid
        assert "current" in message.lower()

    def test_validate_collection_date_flags_retrospective(self):
        """Past dates should be flagged as retrospective"""
        from patches.patch_002_deterministic_collection import validate_collection_date

        past_date = (date.today() - timedelta(days=30)).isoformat()
        is_valid, message = validate_collection_date(past_date)

        assert is_valid
        assert "RETROSPECTIVE" in message.upper()

    def test_enforce_explicit_as_of_date_decorator(self):
        """Decorator should require explicit as_of_date"""
        from patches.patch_002_deterministic_collection import (
            enforce_explicit_as_of_date,
            DeterministicCollectionError,
        )

        @enforce_explicit_as_of_date
        def my_func(as_of_date=None):
            return as_of_date

        # Should raise without as_of_date
        with pytest.raises(DeterministicCollectionError):
            my_func()

        # Should work with as_of_date
        result = my_func(as_of_date=date.today().isoformat())
        assert result is not None


# =============================================================================
# PATCH 003: Schema Validation Tests
# =============================================================================

class TestSchemaValidation:
    """Tests for patch_003_schema_validation.py"""

    def test_validate_trial_record_requires_nct_id(self):
        """Trial records must have nct_id"""
        from patches.patch_003_schema_validation import validate_trial_record

        # Missing nct_id
        record = {"ticker": "ACME", "overall_status": "ACTIVE"}
        errors = validate_trial_record(record, 0)

        assert len(errors) > 0
        assert any("nct_id" in e for e in errors)

    def test_validate_trial_record_validates_nct_format(self):
        """NCT ID must start with NCT"""
        from patches.patch_003_schema_validation import validate_trial_record

        record = {
            "nct_id": "INVALID123",  # Should start with NCT
            "ticker": "ACME",
            "overall_status": "ACTIVE",
        }
        errors = validate_trial_record(record, 0)

        assert any("NCT" in e for e in errors)

    def test_validate_trial_record_accepts_valid(self):
        """Valid trial records should pass"""
        from patches.patch_003_schema_validation import validate_trial_record

        record = {
            "nct_id": "NCT12345678",
            "ticker": "ACME",
            "overall_status": "RECRUITING",
            "phase": "Phase 3",
            "primary_completion_date": "2026-06-15",
        }
        errors = validate_trial_record(record, 0)

        assert len(errors) == 0

    def test_load_and_validate_trial_records_strict(self):
        """Strict mode should raise on validation errors"""
        from patches.patch_003_schema_validation import (
            load_and_validate_trial_records,
            SchemaValidationError,
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            # Invalid record - missing nct_id
            json.dump([{"ticker": "ACME", "overall_status": "ACTIVE"}], f)
            f.flush()

            with pytest.raises(SchemaValidationError):
                load_and_validate_trial_records(Path(f.name), strict=True)

    def test_safe_get_with_type_checking(self):
        """safe_get should check types"""
        from patches.patch_003_schema_validation import safe_get

        data = {"score": "not_a_number", "valid_score": 95.5}

        # Wrong type should return default
        result = safe_get(data, "score", default=0.0, expected_type=float)
        assert result == 0.0

        # Correct type should return value
        result = safe_get(data, "valid_score", default=0.0, expected_type=float)
        assert result == 95.5

    def test_safe_list_access(self):
        """safe_list_access should handle out of range"""
        from patches.patch_003_schema_validation import safe_list_access

        data = [1, 2, 3]

        # Valid index
        assert safe_list_access(data, 0) == 1
        assert safe_list_access(data, -1) == 3

        # Out of range
        assert safe_list_access(data, 10, default="N/A") == "N/A"

        # Empty list
        assert safe_list_access([], 0, default="empty") == "empty"


# =============================================================================
# PATCH 004: Exception Handling Tests
# =============================================================================

class TestExceptionHandling:
    """Tests for patch_004_exception_handling.py"""

    def test_safe_decimal_convert_handles_none(self):
        """safe_decimal_convert should handle None"""
        from patches.patch_004_exception_handling import safe_decimal_convert

        result = safe_decimal_convert(None, default=Decimal("42"))
        assert result == Decimal("42")

    def test_safe_decimal_convert_handles_string(self):
        """safe_decimal_convert should parse strings"""
        from patches.patch_004_exception_handling import safe_decimal_convert

        result = safe_decimal_convert("123.45")
        assert result == Decimal("123.45")

    def test_safe_decimal_convert_handles_invalid(self):
        """safe_decimal_convert should return default for invalid input"""
        from patches.patch_004_exception_handling import safe_decimal_convert

        result = safe_decimal_convert("not_a_number", default=Decimal("-1"))
        assert result == Decimal("-1")

    def test_safe_divide_handles_zero(self):
        """safe_divide should handle division by zero"""
        from patches.patch_004_exception_handling import safe_divide

        result = safe_divide(100, 0, default=Decimal("999"))
        assert result == Decimal("999")

    def test_safe_divide_normal_case(self):
        """safe_divide should work for normal cases"""
        from patches.patch_004_exception_handling import safe_divide

        result = safe_divide(100, 4)
        assert result == Decimal("25")

    def test_exception_accumulator_tracks_errors(self):
        """ExceptionAccumulator should track all errors"""
        from patches.patch_004_exception_handling import ExceptionAccumulator

        acc = ExceptionAccumulator("test")

        # Add some exceptions
        acc.increment_processed(10)
        acc.add(ValueError("test1"), context={"ticker": "AAA"})
        acc.add(TypeError("test2"), context={"ticker": "BBB"})

        assert acc.error_count == 2
        assert acc.total_processed == 10
        assert acc.error_rate == 0.2  # 2/10

    def test_exception_accumulator_groups_by_type(self):
        """ExceptionAccumulator report should group by type"""
        from patches.patch_004_exception_handling import ExceptionAccumulator

        acc = ExceptionAccumulator("test")

        acc.add(ValueError("v1"))
        acc.add(ValueError("v2"))
        acc.add(TypeError("t1"))

        report = acc.get_report()

        # Should have 3 exceptions
        assert len(report["exceptions"]) == 3

        # Should have both types
        types = [e["type"] for e in report["exceptions"]]
        assert "ValueError" in types
        assert "TypeError" in types

    def test_sanitize_corr_fixed_logs_errors(self):
        """Fixed sanitize_corr should handle errors properly"""
        from patches.patch_004_exception_handling import sanitize_corr_fixed

        # Missing correlation
        corr, flags = sanitize_corr_fixed({})
        assert corr is None
        assert "def_corr_missing" in flags

        # Invalid value
        corr, flags = sanitize_corr_fixed({"corr_xbi": "invalid"})
        assert corr is None
        assert any("parse_fail" in f for f in flags)

        # Placeholder value
        corr, flags = sanitize_corr_fixed({"corr_xbi": "0.50"})
        assert corr is None
        assert "def_corr_placeholder_0.50" in flags

        # Valid value
        corr, flags = sanitize_corr_fixed({"corr_xbi": "0.35"})
        assert corr == Decimal("0.35")
        assert len(flags) == 0


# =============================================================================
# REGRESSION TESTS FOR KNOWN BUGS
# =============================================================================

class TestRegressionBugs:
    """Regression tests for bugs found during review"""

    def test_runway_zero_not_treated_as_none(self):
        """
        Bug: if runway_months was 0, it was treated as None
        because of `if runway_months:` instead of `if runway_months is not None:`

        This is fixed in module_2_financial.py but we test the pattern here.
        """
        # The bug pattern
        def buggy_check(runway_months):
            if runway_months:  # BUG: 0 is falsy!
                return "has_runway"
            return "no_runway"

        # The fix pattern
        def fixed_check(runway_months):
            if runway_months is not None:
                return "has_runway"
            return "no_runway"

        # Zero runway should be "has_runway", not "no_runway"
        assert buggy_check(0) == "no_runway"  # Bug behavior
        assert fixed_check(0) == "has_runway"  # Correct behavior
        assert fixed_check(None) == "no_runway"  # None is correctly no_runway

    def test_array_access_bounds_checking(self):
        """
        Bug: hist['Close'].iloc[-21] without checking length

        collect_market_data.py:94-95
        """
        # Simulate the bug
        short_list = [1, 2, 3]  # Only 3 elements

        # Bug: direct access fails
        try:
            _ = short_list[-21]  # IndexError
            assert False, "Should have raised IndexError"
        except IndexError:
            pass

        # Fix: check length first
        def safe_access(lst, idx, default=None):
            if abs(idx) < len(lst):
                return lst[idx]
            return default

        assert safe_access(short_list, -21, default=0) == 0
        assert safe_access(short_list, -1, default=0) == 3

    def test_cusip_empty_string_slice(self):
        """
        Bug: cusip = item.get('compositeFIGI', '')[-9:]
        If compositeFIGI is missing, this takes [-9:] of '', returning ''

        map_tickers_to_cusips.py:26
        """
        # Bug demonstration
        item = {}  # No compositeFIGI
        cusip_buggy = item.get('compositeFIGI', '')[-9:]
        assert cusip_buggy == ''  # Empty string, not None

        # The real issue: empty string passes `if cusip:` check
        # because the check happens BEFORE the slice, but in buggy code
        # the slice happens first

        # Fix: validate after slicing
        def get_cusip_fixed(item):
            figi = item.get('compositeFIGI')
            if not figi or len(figi) < 9:
                return None
            return figi[-9:]

        assert get_cusip_fixed({}) is None
        assert get_cusip_fixed({'compositeFIGI': 'too_short'}) is None
        assert get_cusip_fixed({'compositeFIGI': 'BBGXYZ123ABC'}) == '123ABC'[:9]


# =============================================================================
# INTEGRATION TEST: Full Pipeline Validation Flow
# =============================================================================

class TestPipelineValidation:
    """Integration tests for validation flow"""

    def test_module_output_validation(self):
        """Module outputs should be validated"""
        from patches.patch_003_schema_validation import validate_module_output

        # Valid Module 2 output
        m2_output = {
            "scores": [
                {"ticker": "AAA", "financial_score": 85.0},
                {"ticker": "BBB", "financial_score": 72.5},
            ],
            "diagnostic_counts": {"scored": 2, "missing": 0},
        }

        result = validate_module_output(
            m2_output,
            module_name="module_2",
            required_fields={"scores", "diagnostic_counts"},
        )

        assert result.is_valid
        assert result.record_count == 2

    def test_pipeline_data_loading(self):
        """Test loading and validating pipeline data"""
        from patches.patch_003_schema_validation import (
            load_and_validate_trial_records,
            ValidationResult,
        )

        # Create valid test data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_data = [
                {
                    "nct_id": "NCT00000001",
                    "ticker": "AAA",
                    "overall_status": "ACTIVE",
                },
                {
                    "nct_id": "NCT00000002",
                    "ticker": "BBB",
                    "overall_status": "RECRUITING",
                },
            ]
            json.dump(test_data, f)
            f.flush()

            records, result = load_and_validate_trial_records(Path(f.name), strict=True)

            assert len(records) == 2
            assert result.is_valid
            assert result.valid_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
