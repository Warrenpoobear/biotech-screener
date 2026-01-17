"""
Tests for Module 4 Clinical Development v2 Robustness Improvements.

Tests cover:
- Input validation (as_of_date, active_tickers, trial_records)
- Empty ticker and NCT ID tracking
- Null handling in design scoring
- Conditions parsing with deep nesting
- Upper bounds protection
- Malformed input handling
- Edge cases and boundary conditions

Author: Wake Robin Capital Management
Version: 2.1.0
"""

from decimal import Decimal
from datetime import date
import pytest

from module_4_clinical_dev_v2 import (
    compute_module_4_clinical_dev_v2,
    ValidationError,
    _validate_as_of_date,
    _validate_active_tickers,
    _validate_trial_records,
    _normalize_conditions,
    _score_trial_count,
    _score_design,
    _safe_string_field,
    TrialPITRecord,
    TrialStatus,
    MAX_ENDPOINT_LENGTH,
    MAX_CONDITIONS_DEPTH,
    MAX_CONDITION_STRING_LENGTH,
    MAX_TRIALS_PER_TICKER,
)


# ============================================================================
# AS_OF_DATE VALIDATION TESTS
# ============================================================================

class TestAsOfDateValidation:
    """Tests for as_of_date input validation."""

    def test_valid_iso_date_string(self):
        """Valid ISO date string should pass."""
        result = _validate_as_of_date("2026-01-15")
        assert result == "2026-01-15"

    def test_date_object_accepted(self):
        """date object should be converted to ISO string."""
        result = _validate_as_of_date(date(2026, 1, 15))
        assert result == "2026-01-15"

    def test_datetime_string_truncated(self):
        """Datetime string should be truncated to date portion."""
        result = _validate_as_of_date("2026-01-15T12:30:00")
        assert result == "2026-01-15"

    def test_none_raises_validation_error(self):
        """None as_of_date should raise ValidationError."""
        with pytest.raises(ValidationError, match="cannot be None"):
            _validate_as_of_date(None)

    def test_empty_string_raises_validation_error(self):
        """Empty string as_of_date should raise ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            _validate_as_of_date("")

    def test_whitespace_only_raises_validation_error(self):
        """Whitespace-only string should raise ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            _validate_as_of_date("   ")

    def test_invalid_format_raises_validation_error(self):
        """Invalid date format should raise ValidationError."""
        with pytest.raises(ValidationError, match="valid ISO date"):
            _validate_as_of_date("01-15-2026")

    def test_invalid_type_raises_validation_error(self):
        """Non-string/non-date type should raise ValidationError."""
        with pytest.raises(ValidationError, match="must be string or date"):
            _validate_as_of_date(12345)


# ============================================================================
# ACTIVE_TICKERS VALIDATION TESTS
# ============================================================================

class TestActiveTickersValidation:
    """Tests for active_tickers input validation."""

    def test_set_accepted(self):
        """Set of tickers should be accepted."""
        result = _validate_active_tickers({"AAPL", "MSFT", "GOOG"})
        assert result == ["AAPL", "GOOG", "MSFT"]  # Sorted

    def test_list_accepted(self):
        """List of tickers should be accepted."""
        result = _validate_active_tickers(["AAPL", "MSFT"])
        assert result == ["AAPL", "MSFT"]

    def test_tuple_accepted(self):
        """Tuple of tickers should be accepted."""
        result = _validate_active_tickers(("AAPL", "MSFT"))
        assert result == ["AAPL", "MSFT"]

    def test_frozenset_accepted(self):
        """Frozenset of tickers should be accepted."""
        result = _validate_active_tickers(frozenset({"AAPL", "MSFT"}))
        assert set(result) == {"AAPL", "MSFT"}

    def test_none_returns_empty_list(self):
        """None should return empty list."""
        result = _validate_active_tickers(None)
        assert result == []

    def test_empty_set_returns_empty_list(self):
        """Empty set should return empty list."""
        result = _validate_active_tickers(set())
        assert result == []

    def test_uppercase_normalization(self):
        """Lowercase tickers should be uppercased."""
        result = _validate_active_tickers(["aapl", "msft"])
        assert result == ["AAPL", "MSFT"]

    def test_whitespace_stripped(self):
        """Whitespace around tickers should be stripped."""
        result = _validate_active_tickers(["  AAPL  ", "MSFT  "])
        assert result == ["AAPL", "MSFT"]

    def test_empty_strings_filtered(self):
        """Empty strings should be filtered out."""
        result = _validate_active_tickers(["AAPL", "", "  ", "MSFT"])
        assert result == ["AAPL", "MSFT"]

    def test_duplicates_removed(self):
        """Duplicate tickers should be removed."""
        result = _validate_active_tickers(["AAPL", "MSFT", "AAPL", "msft"])
        assert result == ["AAPL", "MSFT"]

    def test_invalid_type_raises_error(self):
        """Non-iterable type should raise ValidationError."""
        with pytest.raises(ValidationError, match="must be set, list"):
            _validate_active_tickers("AAPL")

    def test_non_string_ticker_raises_error(self):
        """Non-string ticker in collection should raise ValidationError."""
        with pytest.raises(ValidationError, match="must be string"):
            _validate_active_tickers(["AAPL", 123, "MSFT"])


# ============================================================================
# TRIAL_RECORDS VALIDATION TESTS
# ============================================================================

class TestTrialRecordsValidation:
    """Tests for trial_records input validation."""

    def test_valid_list_of_dicts(self):
        """Valid list of dicts should pass."""
        records = [{"ticker": "AAPL"}, {"ticker": "MSFT"}]
        result = _validate_trial_records(records)
        assert result == records

    def test_none_returns_empty_list(self):
        """None should return empty list."""
        result = _validate_trial_records(None)
        assert result == []

    def test_empty_list_accepted(self):
        """Empty list should be accepted."""
        result = _validate_trial_records([])
        assert result == []

    def test_non_list_raises_error(self):
        """Non-list type should raise ValidationError."""
        with pytest.raises(ValidationError, match="must be list"):
            _validate_trial_records({"ticker": "AAPL"})

    def test_non_dict_record_raises_error(self):
        """Non-dict record in list should raise ValidationError."""
        with pytest.raises(ValidationError, match="must be dict"):
            _validate_trial_records([{"ticker": "AAPL"}, "invalid"])


# ============================================================================
# CONDITIONS PARSING TESTS
# ============================================================================

class TestConditionsParsing:
    """Tests for _normalize_conditions robustness."""

    def test_simple_string(self):
        """Simple string should be parsed."""
        result = _normalize_conditions("breast cancer")
        assert result == ["breast cancer"]

    def test_comma_separated(self):
        """Comma-separated conditions should be split."""
        result = _normalize_conditions("breast cancer, lung cancer")
        assert result == ["breast cancer", "lung cancer"]

    def test_semicolon_separated(self):
        """Semicolon-separated conditions should be split."""
        result = _normalize_conditions("breast cancer; lung cancer")
        assert result == ["breast cancer", "lung cancer"]

    def test_pipe_separated(self):
        """Pipe-separated conditions should be split."""
        result = _normalize_conditions("breast cancer|lung cancer")
        assert result == ["breast cancer", "lung cancer"]

    def test_list_of_strings(self):
        """List of strings should be parsed."""
        result = _normalize_conditions(["breast cancer", "lung cancer"])
        assert result == ["breast cancer", "lung cancer"]

    def test_nested_list_depth_2(self):
        """Nested list (depth 2) should be flattened."""
        result = _normalize_conditions([["breast cancer"], ["lung cancer"]])
        assert result == ["breast cancer", "lung cancer"]

    def test_nested_list_depth_3(self):
        """Nested list (depth 3) should be handled within limit."""
        result = _normalize_conditions([[["breast cancer"]], ["lung cancer"]])
        assert "breast cancer" in result
        assert "lung cancer" in result

    def test_dict_with_name_key(self):
        """Dict with 'name' key should extract condition."""
        result = _normalize_conditions({"name": "breast cancer"})
        assert result == ["breast cancer"]

    def test_dict_with_condition_key(self):
        """Dict with 'condition' key should extract condition."""
        result = _normalize_conditions({"condition": "breast cancer"})
        assert result == ["breast cancer"]

    def test_none_returns_empty(self):
        """None should return empty list."""
        result = _normalize_conditions(None)
        assert result == []

    def test_integer_ignored(self):
        """Integer values should be ignored."""
        result = _normalize_conditions(123)
        assert result == []

    def test_long_string_truncated(self):
        """Long condition string should be truncated."""
        long_condition = "a" * 1000
        result = _normalize_conditions(long_condition)
        assert len(result[0]) <= MAX_CONDITION_STRING_LENGTH

    def test_depth_limit_enforced(self):
        """Nesting beyond max depth should be ignored."""
        # Create deeply nested structure
        deep = "condition"
        for _ in range(MAX_CONDITIONS_DEPTH + 2):
            deep = [deep]
        result = _normalize_conditions(deep)
        # Should still find something due to partial traversal
        # but not crash

    def test_deduplication(self):
        """Duplicate conditions should be deduplicated."""
        result = _normalize_conditions(["cancer", "Cancer", "CANCER"])
        assert result == ["cancer"]

    def test_deterministic_ordering(self):
        """Output should be sorted for determinism."""
        result = _normalize_conditions(["zebra", "apple", "mango"])
        assert result == ["apple", "mango", "zebra"]


# ============================================================================
# TRIAL COUNT SCORING TESTS
# ============================================================================

class TestTrialCountScoring:
    """Tests for _score_trial_count with upper bounds."""

    def test_zero_trials(self):
        """Zero trials should score 0."""
        assert _score_trial_count(0) == Decimal("0")

    def test_one_trial(self):
        """One trial should score 0.5."""
        assert _score_trial_count(1) == Decimal("0.5")

    def test_two_trials(self):
        """Two trials should score 1.0."""
        assert _score_trial_count(2) == Decimal("1.0")

    def test_five_trials(self):
        """Five trials should score 2.0."""
        assert _score_trial_count(5) == Decimal("2.0")

    def test_twenty_trials(self):
        """Twenty trials should score 4.5."""
        assert _score_trial_count(20) == Decimal("4.5")

    def test_hundred_trials(self):
        """100 trials should score 5.0."""
        assert _score_trial_count(100) == Decimal("5.0")

    def test_pathological_large_number(self):
        """Pathologically large number should still return max score."""
        assert _score_trial_count(10000) == Decimal("5.0")

    def test_negative_number_clamped(self):
        """Negative number should be clamped to 0."""
        assert _score_trial_count(-5) == Decimal("0")

    def test_float_converted(self):
        """Float should be converted to int."""
        assert _score_trial_count(3.7) == Decimal("2.0")  # 3 trials

    def test_invalid_type_returns_zero(self):
        """Invalid type should return 0."""
        assert _score_trial_count("invalid") == Decimal("0")
        assert _score_trial_count(None) == Decimal("0")


# ============================================================================
# DESIGN SCORING TESTS
# ============================================================================

class TestDesignScoring:
    """Tests for _score_design with null handling."""

    def test_basic_trial(self):
        """Basic trial should get base score."""
        trial = TrialPITRecord(
            nct_id="NCT001",
            ticker="TEST",
            phase="phase 2",
            status=TrialStatus.ACTIVE,
            conditions=["cancer"],
            primary_endpoint="safety",
            randomized=False,
            blinded="",
            last_update_posted="2025-01-01",
            pit_date_field_used="first_posted",
            pit_reference_date="2025-01-01",
            pit_admissible=True,
            pit_reason="admissible",
            endpoint_classification="neutral",
            endpoint_matched_pattern=None,
        )
        score = _score_design(trial)
        assert score == Decimal("12")  # Base score

    def test_randomized_bonus(self):
        """Randomized trial should get bonus."""
        trial = TrialPITRecord(
            nct_id="NCT001",
            ticker="TEST",
            phase="phase 2",
            status=TrialStatus.ACTIVE,
            conditions=["cancer"],
            primary_endpoint="safety",
            randomized=True,
            blinded="",
            last_update_posted="2025-01-01",
            pit_date_field_used="first_posted",
            pit_reference_date="2025-01-01",
            pit_admissible=True,
            pit_reason="admissible",
            endpoint_classification="neutral",
            endpoint_matched_pattern=None,
        )
        score = _score_design(trial)
        assert score == Decimal("17")  # Base + randomized bonus

    def test_double_blind_bonus(self):
        """Double-blind trial should get bonus."""
        trial = TrialPITRecord(
            nct_id="NCT001",
            ticker="TEST",
            phase="phase 2",
            status=TrialStatus.ACTIVE,
            conditions=["cancer"],
            primary_endpoint="safety",
            randomized=False,
            blinded="double-blind",
            last_update_posted="2025-01-01",
            pit_date_field_used="first_posted",
            pit_reference_date="2025-01-01",
            pit_admissible=True,
            pit_reason="admissible",
            endpoint_classification="neutral",
            endpoint_matched_pattern=None,
        )
        score = _score_design(trial)
        assert score == Decimal("16")  # Base + double-blind bonus

    def test_single_blind_partial_bonus(self):
        """Single-blind trial should get partial bonus."""
        trial = TrialPITRecord(
            nct_id="NCT001",
            ticker="TEST",
            phase="phase 2",
            status=TrialStatus.ACTIVE,
            conditions=["cancer"],
            primary_endpoint="safety",
            randomized=False,
            blinded="single-blind",
            last_update_posted="2025-01-01",
            pit_date_field_used="first_posted",
            pit_reference_date="2025-01-01",
            pit_admissible=True,
            pit_reason="admissible",
            endpoint_classification="neutral",
            endpoint_matched_pattern=None,
        )
        score = _score_design(trial)
        assert score == Decimal("14")  # Base + single-blind bonus

    def test_empty_blinded_handled(self):
        """Empty blinded field should not crash."""
        trial = TrialPITRecord(
            nct_id="NCT001",
            ticker="TEST",
            phase="phase 2",
            status=TrialStatus.ACTIVE,
            conditions=["cancer"],
            primary_endpoint="safety",
            randomized=False,
            blinded="",
            last_update_posted="2025-01-01",
            pit_date_field_used="first_posted",
            pit_reference_date="2025-01-01",
            pit_admissible=True,
            pit_reason="admissible",
            endpoint_classification="neutral",
            endpoint_matched_pattern=None,
        )
        score = _score_design(trial)
        assert score == Decimal("12")  # Just base score

    def test_strong_endpoint_bonus(self):
        """Strong endpoint should get bonus."""
        trial = TrialPITRecord(
            nct_id="NCT001",
            ticker="TEST",
            phase="phase 2",
            status=TrialStatus.ACTIVE,
            conditions=["cancer"],
            primary_endpoint="overall survival",
            randomized=False,
            blinded="",
            last_update_posted="2025-01-01",
            pit_date_field_used="first_posted",
            pit_reference_date="2025-01-01",
            pit_admissible=True,
            pit_reason="admissible",
            endpoint_classification="strong",
            endpoint_matched_pattern="overall_survival",
        )
        score = _score_design(trial)
        assert score == Decimal("16")  # Base + strong endpoint

    def test_weak_endpoint_penalty(self):
        """Weak endpoint should get penalty."""
        trial = TrialPITRecord(
            nct_id="NCT001",
            ticker="TEST",
            phase="phase 2",
            status=TrialStatus.ACTIVE,
            conditions=["cancer"],
            primary_endpoint="safety",
            randomized=False,
            blinded="",
            last_update_posted="2025-01-01",
            pit_date_field_used="first_posted",
            pit_reference_date="2025-01-01",
            pit_admissible=True,
            pit_reason="admissible",
            endpoint_classification="weak",
            endpoint_matched_pattern="safety",
        )
        score = _score_design(trial)
        assert score == Decimal("9")  # Base - weak penalty

    def test_score_clamped_at_max(self):
        """Score should be clamped at 25."""
        trial = TrialPITRecord(
            nct_id="NCT001",
            ticker="TEST",
            phase="phase 3",
            status=TrialStatus.COMPLETED,
            conditions=["cancer"],
            primary_endpoint="overall survival",
            randomized=True,
            blinded="double-blind",
            last_update_posted="2025-01-01",
            pit_date_field_used="first_posted",
            pit_reference_date="2025-01-01",
            pit_admissible=True,
            pit_reason="admissible",
            endpoint_classification="strong",
            endpoint_matched_pattern="overall_survival",
        )
        score = _score_design(trial)
        assert score <= Decimal("25")


# ============================================================================
# SAFE STRING FIELD TESTS
# ============================================================================

class TestSafeStringField:
    """Tests for _safe_string_field utility."""

    def test_normal_string(self):
        """Normal string should pass through."""
        assert _safe_string_field("hello") == "hello"

    def test_none_returns_empty(self):
        """None should return empty string."""
        assert _safe_string_field(None) == ""

    def test_integer_converted(self):
        """Integer should be converted to string."""
        assert _safe_string_field(123) == "123"

    def test_long_string_truncated(self):
        """Long string should be truncated."""
        long_str = "a" * 1000
        result = _safe_string_field(long_str, max_length=100)
        assert len(result) == 100

    def test_whitespace_stripped(self):
        """Whitespace should be stripped."""
        assert _safe_string_field("  hello  ") == "hello"


# ============================================================================
# FULL MODULE INTEGRATION TESTS
# ============================================================================

class TestModuleIntegration:
    """Integration tests for the full module with robustness checks."""

    def test_invalid_as_of_date_raises(self):
        """Invalid as_of_date should raise ValidationError."""
        with pytest.raises(ValidationError):
            compute_module_4_clinical_dev_v2([], {"TEST"}, "invalid-date")

    def test_invalid_active_tickers_type_raises(self):
        """Invalid active_tickers type should raise ValidationError."""
        with pytest.raises(ValidationError):
            compute_module_4_clinical_dev_v2([], "not-a-set", "2026-01-15")

    def test_invalid_trial_records_type_raises(self):
        """Invalid trial_records type should raise ValidationError."""
        with pytest.raises(ValidationError):
            compute_module_4_clinical_dev_v2("not-a-list", {"TEST"}, "2026-01-15")

    def test_empty_tickers_returns_empty_result(self):
        """Empty active_tickers should return empty result."""
        result = compute_module_4_clinical_dev_v2([], set(), "2026-01-15")
        assert result["scores"] == []
        assert result["diagnostic_counts"]["tickers_scored"] == 0

    def test_validation_issues_tracked(self):
        """Validation issues should be tracked in diagnostics."""
        # Create trial with empty ticker
        trials = [
            {"ticker": "", "nct_id": "NCT001", "phase": "phase 2"},
            {"ticker": "TEST", "nct_id": "", "phase": "phase 2"},  # Empty NCT ID
            {"nct_id": "NCT002", "phase": "phase 2"},  # Missing ticker (None)
        ]
        result = compute_module_4_clinical_dev_v2(
            trials, {"TEST"}, "2026-01-15"
        )
        issues = result["diagnostic_counts"]["validation_issues"]
        assert issues["empty_ticker"] >= 1
        assert issues["empty_nct_id"] >= 1

    def test_date_object_accepted(self):
        """date object should be accepted as as_of_date."""
        result = compute_module_4_clinical_dev_v2(
            [], {"TEST"}, date(2026, 1, 15)
        )
        assert result["as_of_date"] == "2026-01-15"

    def test_deterministic_output(self):
        """Same inputs should produce same output."""
        trials = [
            {
                "ticker": "TEST",
                "nct_id": "NCT001",
                "phase": "phase 2",
                "status": "active",
                "conditions": "breast cancer",
                "primary_endpoint": "overall survival",
                "first_posted": "2025-01-01",
            }
        ]
        result1 = compute_module_4_clinical_dev_v2(trials, {"TEST"}, "2026-01-15")
        result2 = compute_module_4_clinical_dev_v2(trials, {"TEST"}, "2026-01-15")

        # Compare scores (excluding provenance timestamp)
        assert result1["scores"] == result2["scores"]
        assert result1["diagnostic_counts"] == result2["diagnostic_counts"]

    def test_long_endpoint_truncated(self):
        """Long endpoint should be truncated but not crash."""
        long_endpoint = "overall survival " * 1000
        trials = [
            {
                "ticker": "TEST",
                "nct_id": "NCT001",
                "phase": "phase 2",
                "primary_endpoint": long_endpoint,
                "first_posted": "2025-01-01",
            }
        ]
        result = compute_module_4_clinical_dev_v2(trials, {"TEST"}, "2026-01-15")
        # Should complete without error
        assert len(result["scores"]) == 1


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Tests for various edge cases."""

    def test_non_string_ticker_in_trial(self):
        """Non-string ticker in trial should be converted."""
        trials = [
            {
                "ticker": 123,  # Integer ticker
                "nct_id": "NCT001",
                "phase": "phase 2",
                "first_posted": "2025-01-01",
            }
        ]
        result = compute_module_4_clinical_dev_v2(trials, {"123"}, "2026-01-15")
        # Should handle conversion
        issues = result["diagnostic_counts"]["validation_issues"]
        assert issues["invalid_ticker_type"] >= 1

    def test_mixed_case_tickers_normalized(self):
        """Mixed case tickers should be normalized."""
        trials = [
            {
                "ticker": "test",
                "nct_id": "NCT001",
                "phase": "phase 2",
                "first_posted": "2025-01-01",
            }
        ]
        result = compute_module_4_clinical_dev_v2(trials, {"TEST"}, "2026-01-15")
        # Lowercase ticker should match uppercase in active_tickers
        assert result["diagnostic_counts"]["total_trials_raw"] == 1

    def test_whitespace_in_ticker_stripped(self):
        """Whitespace in ticker should be stripped."""
        trials = [
            {
                "ticker": "  TEST  ",
                "nct_id": "NCT001",
                "phase": "phase 2",
                "first_posted": "2025-01-01",
            }
        ]
        result = compute_module_4_clinical_dev_v2(trials, {"TEST"}, "2026-01-15")
        assert result["diagnostic_counts"]["total_trials_raw"] == 1

    def test_deeply_nested_conditions_handled(self):
        """Deeply nested conditions should be handled gracefully."""
        trials = [
            {
                "ticker": "TEST",
                "nct_id": "NCT001",
                "phase": "phase 2",
                "conditions": [[["breast cancer"]]],
                "first_posted": "2025-01-01",
            }
        ]
        result = compute_module_4_clinical_dev_v2(trials, {"TEST"}, "2026-01-15")
        # Should not crash
        assert len(result["scores"]) == 1
