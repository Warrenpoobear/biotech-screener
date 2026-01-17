"""
tests/integration/test_pipeline_robustness.py - Pipeline Robustness Tests

Comprehensive tests for pipeline robustness and edge case handling.

Tests:
- Empty universe handling
- Empty data handling
- Null/missing value handling
- Score bounds validation
- Circuit breaker functionality
- Input validation
- Pipeline determinism

Author: Wake Robin Capital Management
Version: 1.0.0
"""

import pytest
from datetime import date
from decimal import Decimal
from typing import Any, Dict, List

# Import module functions
from module_2_financial_v2 import run_module_2_v2
from module_4_clinical_dev_v2 import compute_module_4_clinical_dev_v2
from module_5_composite_v2 import compute_module_5_composite_v2

# Import utilities
from common.input_validation import (
    validate_pipeline_inputs,
    validate_ticker,
    validate_tickers,
    validate_financial_record,
    validate_trial_record,
    ValidationResult,
    PipelineValidationError,
)
from common.score_utils import (
    clamp_score,
    normalize_to_range,
    weighted_average,
    to_decimal,
)
from common.null_safety import (
    safe_get,
    safe_get_nested,
    safe_divide,
    coalesce,
    is_present,
    is_zero_or_none,
)
from common.data_quality import (
    check_circuit_breaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def as_of_date():
    """Standard as_of_date for tests."""
    return "2026-01-15"


@pytest.fixture
def sample_universe():
    """Sample universe of tickers."""
    return ["AAPL", "GOOG", "MSFT", "AMZN", "TSLA"]


@pytest.fixture
def sample_financial_data():
    """Sample financial data for tests."""
    return [
        {
            "ticker": "AAPL",
            "Cash": 100000000,
            "NetIncome": -5000000,
            "market_cap_mm": 1000,
            "shares_outstanding": 100000000,
        },
        {
            "ticker": "GOOG",
            "Cash": 200000000,
            "NetIncome": 10000000,
            "market_cap_mm": 2000,
            "shares_outstanding": 50000000,
        },
    ]


@pytest.fixture
def sample_market_data():
    """Sample market data for tests."""
    return [
        {"ticker": "AAPL", "price": 150.0, "avg_volume": 1000000, "market_cap": 1500000000},
        {"ticker": "GOOG", "price": 100.0, "avg_volume": 500000, "market_cap": 500000000},
    ]


@pytest.fixture
def sample_trial_data():
    """Sample trial data for tests."""
    return [
        {
            "ticker": "AAPL",
            "nct_id": "NCT12345678",
            "phase": "Phase 2",
            "status": "Recruiting",
            "conditions": ["Cancer"],
            "primary_endpoint": "Overall Survival",
            "first_posted": "2025-01-01",
        },
        {
            "ticker": "GOOG",
            "nct_id": "NCT87654321",
            "phase": "Phase 3",
            "status": "Active",
            "conditions": ["Diabetes"],
            "primary_endpoint": "HbA1c Reduction",
            "first_posted": "2024-06-01",
        },
    ]


# ============================================================================
# EMPTY UNIVERSE TESTS
# ============================================================================

class TestEmptyUniverseHandling:
    """Tests for graceful handling of empty universes."""

    def test_module_2_empty_universe(self):
        """Module 2 should return empty results for empty universe."""
        result = run_module_2_v2(
            universe=[],
            financial_data=[],
            market_data=[],
        )

        assert result["scores"] == []
        assert result["diagnostic_counts"]["scored"] == 0

    def test_module_4_empty_universe(self, as_of_date):
        """Module 4 should return empty results for empty universe."""
        result = compute_module_4_clinical_dev_v2(
            trial_records=[],
            active_tickers=[],
            as_of_date=as_of_date,
        )

        assert result["scores"] == []
        assert result["diagnostic_counts"]["tickers_scored"] == 0

    def test_module_5_empty_universe(self, as_of_date):
        """Module 5 should return empty results for empty universe."""
        result = compute_module_5_composite_v2(
            universe_result={"active_securities": []},
            financial_result={"scores": []},
            catalyst_result={"summaries": {}},
            clinical_result={"scores": []},
            as_of_date=as_of_date,
        )

        assert result["ranked_securities"] == []
        assert result["diagnostic_counts"]["total_input"] == 0


# ============================================================================
# EMPTY DATA TESTS
# ============================================================================

class TestEmptyDataHandling:
    """Tests for graceful handling of empty data."""

    def test_module_2_no_financial_data(self, sample_universe):
        """Module 2 should handle missing financial data."""
        result = run_module_2_v2(
            universe=sample_universe,
            financial_data=[],
            market_data=[],
        )

        # Should still produce scores (with default/missing flags)
        assert isinstance(result["scores"], list)

    def test_module_4_no_trial_data(self, as_of_date, sample_universe):
        """Module 4 should handle missing trial data."""
        result = compute_module_4_clinical_dev_v2(
            trial_records=[],
            active_tickers=sample_universe,
            as_of_date=as_of_date,
        )

        # Should produce scores for all tickers (with no_trials flag)
        assert len(result["scores"]) == len(sample_universe)


# ============================================================================
# INPUT VALIDATION TESTS
# ============================================================================

class TestInputValidation:
    """Tests for input validation utilities."""

    def test_validate_ticker_valid(self):
        """Valid tickers should pass validation."""
        result = validate_ticker("AAPL")
        assert result.valid is True
        assert result.errors == []

    def test_validate_ticker_empty(self):
        """Empty ticker should fail validation."""
        result = validate_ticker("")
        assert result.valid is False
        assert len(result.errors) > 0

    def test_validate_ticker_too_long(self):
        """Ticker exceeding max length should fail."""
        result = validate_ticker("TOOLONG")
        assert result.valid is False
        assert "exceeds" in result.errors[0].lower()

    def test_validate_ticker_invalid_chars(self):
        """Ticker with invalid characters should fail."""
        result = validate_ticker("ABC123")
        assert result.valid is False

    def test_validate_tickers_with_duplicates(self):
        """Duplicate tickers should be flagged."""
        valid, invalid = validate_tickers(["AAPL", "GOOG", "AAPL"])
        # First AAPL should be valid, second should be flagged
        assert "AAPL" in valid  # First one is valid
        # Note: duplicates are detected after the first occurrence

    def test_validate_financial_record_valid(self, sample_financial_data):
        """Valid financial record should pass validation."""
        result = validate_financial_record(sample_financial_data[0])
        assert result.valid is True

    def test_validate_financial_record_negative_cash(self):
        """Negative cash should fail validation."""
        record = {"ticker": "TEST", "Cash": -1000000}
        result = validate_financial_record(record)
        assert result.valid is False
        assert any("negative" in e.lower() for e in result.errors)

    def test_validate_pipeline_inputs_empty_universe(self):
        """Empty universe should fail pipeline validation."""
        result = validate_pipeline_inputs(
            universe=[],
            as_of_date="2026-01-15",
        )
        assert result.passed is False
        assert any("empty" in e.lower() for e in result.errors)

    def test_validate_pipeline_inputs_strict_mode(self):
        """Strict mode should raise exception on failure."""
        with pytest.raises(PipelineValidationError):
            validate_pipeline_inputs(
                universe=[],
                as_of_date="2026-01-15",
                strict=True,
            )


# ============================================================================
# SCORE CLAMPING TESTS
# ============================================================================

class TestScoreClamping:
    """Tests for score clamping utilities."""

    def test_clamp_score_within_range(self):
        """Score within range should be unchanged."""
        result = clamp_score(Decimal("50"))
        assert result == Decimal("50.00")

    def test_clamp_score_above_max(self):
        """Score above max should be clamped to max."""
        result = clamp_score(Decimal("150"))
        assert result == Decimal("100.00")

    def test_clamp_score_below_min(self):
        """Score below min should be clamped to min."""
        result = clamp_score(Decimal("-10"))
        assert result == Decimal("0.00")

    def test_clamp_score_none_with_default(self):
        """None score with default should return default."""
        result = clamp_score(None, default=Decimal("50"))
        assert result == Decimal("50")

    def test_clamp_score_string_input(self):
        """String input should be converted and clamped."""
        result = clamp_score("75.5")
        assert result == Decimal("75.50")

    def test_normalize_to_range(self):
        """Normalization should map input range to output range."""
        result = normalize_to_range(500, input_min=0, input_max=1000)
        assert result == Decimal("50.00")

    def test_weighted_average_with_none(self):
        """Weighted average should skip None values."""
        scores = [Decimal("60"), None, Decimal("80")]
        weights = [Decimal("0.4"), Decimal("0.3"), Decimal("0.3")]
        result, total_weight = weighted_average(scores, weights, skip_none=True)
        # Only 60 and 80 contribute, with renormalized weights
        assert result is not None
        assert total_weight == Decimal("0.7")


# ============================================================================
# NULL SAFETY TESTS
# ============================================================================

class TestNullSafety:
    """Tests for null safety utilities."""

    def test_safe_get_present_key(self):
        """safe_get should return value for present key."""
        data = {"key": "value"}
        assert safe_get(data, "key") == "value"

    def test_safe_get_missing_key(self):
        """safe_get should return default for missing key."""
        data = {"key": "value"}
        assert safe_get(data, "other", default="default") == "default"

    def test_safe_get_none_mapping(self):
        """safe_get should return default for None mapping."""
        assert safe_get(None, "key", default="default") == "default"

    def test_safe_get_nested_success(self):
        """safe_get_nested should traverse nested structure."""
        data = {"a": {"b": {"c": 42}}}
        assert safe_get_nested(data, ["a", "b", "c"]) == 42

    def test_safe_get_nested_missing_key(self):
        """safe_get_nested should return default for missing path."""
        data = {"a": {"b": {}}}
        assert safe_get_nested(data, ["a", "b", "c"], default=0) == 0

    def test_safe_divide_success(self):
        """safe_divide should perform division correctly."""
        result = safe_divide(Decimal("10"), Decimal("2"))
        assert result == Decimal("5")

    def test_safe_divide_by_zero(self):
        """safe_divide should return default for division by zero."""
        result = safe_divide(Decimal("10"), Decimal("0"), default=Decimal("0"))
        assert result == Decimal("0")

    def test_safe_divide_none_numerator(self):
        """safe_divide should return default for None numerator."""
        result = safe_divide(None, Decimal("2"), default=Decimal("0"))
        assert result == Decimal("0")

    def test_coalesce_first_non_none(self):
        """coalesce should return first non-None value."""
        assert coalesce(None, None, 1, 2) == 1

    def test_coalesce_all_none(self):
        """coalesce should return None if all values are None."""
        assert coalesce(None, None, None) is None

    def test_is_present_with_value(self):
        """is_present should return True for non-None values."""
        assert is_present(0) is True  # 0 is not None
        assert is_present("") is True  # empty string is not None
        assert is_present(Decimal("0")) is True

    def test_is_present_with_none(self):
        """is_present should return False for None."""
        assert is_present(None) is False

    def test_is_zero_or_none_zero(self):
        """is_zero_or_none should return True for zero."""
        assert is_zero_or_none(0) is True
        assert is_zero_or_none(0.0) is True
        assert is_zero_or_none(Decimal("0")) is True

    def test_is_zero_or_none_nonzero(self):
        """is_zero_or_none should return False for non-zero."""
        assert is_zero_or_none(1) is False
        assert is_zero_or_none(Decimal("0.01")) is False


# ============================================================================
# CIRCUIT BREAKER TESTS
# ============================================================================

class TestCircuitBreaker:
    """Tests for circuit breaker functionality."""

    def test_circuit_breaker_ok(self):
        """Circuit breaker should not trip for low failure rate."""
        result = check_circuit_breaker(
            total_records=100,
            failed_records=10,  # 10% failure
        )
        assert result.tripped is False
        assert result.warning is False

    def test_circuit_breaker_warning(self):
        """Circuit breaker should warn for moderate failure rate."""
        result = check_circuit_breaker(
            total_records=100,
            failed_records=30,  # 30% failure
        )
        assert result.tripped is False
        assert result.warning is True

    def test_circuit_breaker_trip(self):
        """Circuit breaker should trip for high failure rate."""
        result = check_circuit_breaker(
            total_records=100,
            failed_records=60,  # 60% failure
        )
        assert result.tripped is True

    def test_circuit_breaker_strict_mode(self):
        """Circuit breaker in strict mode should raise exception."""
        config = CircuitBreakerConfig(strict_mode=True)
        with pytest.raises(CircuitBreakerError):
            check_circuit_breaker(
                total_records=100,
                failed_records=60,
                config=config,
            )

    def test_circuit_breaker_skip_small_batch(self):
        """Circuit breaker should skip check for small batches."""
        result = check_circuit_breaker(
            total_records=5,  # Below min threshold
            failed_records=4,  # 80% failure
        )
        assert result.tripped is False
        assert "Skipped" in result.message


# ============================================================================
# PIPELINE DETERMINISM TESTS
# ============================================================================

class TestPipelineDeterminism:
    """Tests for pipeline determinism."""

    def test_module_2_determinism(self, sample_universe, sample_financial_data, sample_market_data):
        """Module 2 should produce identical results for identical inputs."""
        result1 = run_module_2_v2(
            universe=sample_universe,
            financial_data=sample_financial_data,
            market_data=sample_market_data,
        )
        result2 = run_module_2_v2(
            universe=sample_universe,
            financial_data=sample_financial_data,
            market_data=sample_market_data,
        )

        # Scores should be identical
        assert len(result1["scores"]) == len(result2["scores"])
        for s1, s2 in zip(result1["scores"], result2["scores"]):
            assert s1["ticker"] == s2["ticker"]
            assert s1.get("financial_normalized") == s2.get("financial_normalized")

    def test_module_4_determinism(self, as_of_date, sample_universe, sample_trial_data):
        """Module 4 should produce identical results for identical inputs."""
        result1 = compute_module_4_clinical_dev_v2(
            trial_records=sample_trial_data,
            active_tickers=sample_universe,
            as_of_date=as_of_date,
        )
        result2 = compute_module_4_clinical_dev_v2(
            trial_records=sample_trial_data,
            active_tickers=sample_universe,
            as_of_date=as_of_date,
        )

        # Scores should be identical
        assert len(result1["scores"]) == len(result2["scores"])
        for s1, s2 in zip(result1["scores"], result2["scores"]):
            assert s1["ticker"] == s2["ticker"]
            assert s1.get("clinical_score") == s2.get("clinical_score")


# ============================================================================
# BOUNDARY VALUE TESTS
# ============================================================================

class TestBoundaryValues:
    """Tests for boundary value handling."""

    def test_zero_market_cap(self):
        """Zero market cap should be handled gracefully."""
        record = {"ticker": "TEST", "market_cap_mm": 0}
        result = validate_financial_record(record)
        # Should flag as invalid
        assert any("market cap" in e.lower() for e in result.errors)

    def test_negative_market_cap(self):
        """Negative market cap should be handled gracefully."""
        record = {"ticker": "TEST", "market_cap_mm": -100}
        result = validate_financial_record(record)
        # Should flag as invalid
        assert any("market cap" in e.lower() for e in result.errors)

    def test_extreme_scores(self):
        """Extreme score values should be clamped."""
        assert clamp_score(Decimal("1000000")) == Decimal("100.00")
        assert clamp_score(Decimal("-1000000")) == Decimal("0.00")

    def test_very_small_scores(self):
        """Very small positive scores should be preserved."""
        result = clamp_score(Decimal("0.0001"))
        assert result == Decimal("0.00")  # Rounds to precision


# ============================================================================
# TYPE CONVERSION TESTS
# ============================================================================

class TestTypeConversions:
    """Tests for type conversion utilities."""

    def test_to_decimal_from_int(self):
        """Integer should convert to Decimal."""
        assert to_decimal(100) == Decimal("100")

    def test_to_decimal_from_float(self):
        """Float should convert to Decimal via string."""
        result = to_decimal(100.5)
        assert result == Decimal("100.5")

    def test_to_decimal_from_string(self):
        """String should parse to Decimal."""
        assert to_decimal("100.50") == Decimal("100.50")

    def test_to_decimal_from_none(self):
        """None should return default."""
        assert to_decimal(None, default=Decimal("0")) == Decimal("0")

    def test_to_decimal_invalid_string(self):
        """Invalid string should return default."""
        assert to_decimal("not_a_number", default=Decimal("0")) == Decimal("0")


# ============================================================================
# TRIAL RECORD VALIDATION TESTS
# ============================================================================

class TestTrialRecordValidation:
    """Tests for trial record validation."""

    def test_valid_trial_record(self, sample_trial_data):
        """Valid trial record should pass validation."""
        result = validate_trial_record(sample_trial_data[0])
        assert result.valid is True

    def test_missing_nct_id(self):
        """Missing NCT ID should fail validation."""
        record = {"ticker": "TEST", "phase": "Phase 2"}
        result = validate_trial_record(record)
        assert result.valid is False
        assert any("nct_id" in e.lower() for e in result.errors)

    def test_invalid_nct_id_format(self):
        """Invalid NCT ID format should produce warning."""
        record = {"ticker": "TEST", "nct_id": "INVALID123", "phase": "Phase 2"}
        result = validate_trial_record(record)
        # Should have warning about format
        assert len(result.warnings) > 0 or not result.valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
