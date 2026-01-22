#!/usr/bin/env python3
"""
Tests for common/data_quality.py

Data quality gates validate data before pipeline execution.
These tests cover:
- Staleness validation (financial, market, trial data)
- Liquidity validation (ADV thresholds)
- Price validation (penny stock filter)
- Coverage validation (required fields)
- Circuit breaker functionality
"""

import pytest
from datetime import date
from decimal import Decimal

from common.data_quality import (
    DataQualityGates,
    DataQualityConfig,
    QualityGateResult,
    ValidationResult,
    validate_financial_staleness,
    validate_liquidity,
    check_circuit_breaker,
    validate_batch_with_circuit_breaker,
    CircuitBreakerConfig,
    CircuitBreakerResult,
    CircuitBreakerError,
)


class TestDataQualityConfig:
    """Tests for DataQualityConfig."""

    def test_default_values(self):
        """Default configuration values should be set."""
        config = DataQualityConfig()
        assert config.max_financial_age_days == 90
        assert config.max_market_data_age_days == 7
        assert config.min_adv_dollars == 500_000
        assert config.min_price == 5.0
        assert config.min_enrollment == 10

    def test_custom_values(self):
        """Custom configuration values should be accepted."""
        config = DataQualityConfig(
            max_financial_age_days=60,
            min_adv_dollars=1_000_000,
            min_price=10.0,
        )
        assert config.max_financial_age_days == 60
        assert config.min_adv_dollars == 1_000_000
        assert config.min_price == 10.0


class TestValidateFinancialStaleness:
    """Tests for financial staleness validation."""

    def test_fresh_data(self):
        """Recent data should pass."""
        gates = DataQualityGates()
        result = gates.validate_financial_staleness("2026-01-10", "2026-01-15")
        assert result.passed is True
        assert result.value == 5  # 5 days old

    def test_stale_data(self):
        """Data older than threshold should fail."""
        gates = DataQualityGates()
        result = gates.validate_financial_staleness("2025-10-01", "2026-01-15")
        assert result.passed is False
        assert "days old" in result.message

    def test_exactly_at_threshold(self):
        """Data exactly at threshold should pass."""
        config = DataQualityConfig(max_financial_age_days=90)
        gates = DataQualityGates(config)
        # 90 days before 2026-01-15 is 2025-10-17
        result = gates.validate_financial_staleness("2025-10-17", "2026-01-15")
        assert result.passed is True

    def test_one_day_over_threshold(self):
        """Data one day over threshold should fail."""
        config = DataQualityConfig(max_financial_age_days=90)
        gates = DataQualityGates(config)
        result = gates.validate_financial_staleness("2025-10-16", "2026-01-15")
        assert result.passed is False

    def test_none_data_date(self):
        """None data date should fail."""
        gates = DataQualityGates()
        result = gates.validate_financial_staleness(None, "2026-01-15")
        assert result.passed is False
        assert "Missing" in result.message

    def test_invalid_date_format(self):
        """Invalid date format should fail."""
        gates = DataQualityGates()
        result = gates.validate_financial_staleness("not-a-date", "2026-01-15")
        assert result.passed is False
        assert "Invalid date" in result.message

    def test_date_objects(self):
        """Should accept date objects."""
        gates = DataQualityGates()
        result = gates.validate_financial_staleness(date(2026, 1, 10), date(2026, 1, 15))
        assert result.passed is True


class TestValidateLiquidity:
    """Tests for liquidity validation."""

    def test_sufficient_liquidity(self):
        """Sufficient liquidity should pass."""
        gates = DataQualityGates()
        result = gates.validate_liquidity(avg_volume=100_000, price=10.0)
        # ADV = 100,000 * 10 = $1,000,000 > $500,000
        assert result.passed is True

    def test_insufficient_liquidity(self):
        """Insufficient liquidity should fail."""
        gates = DataQualityGates()
        result = gates.validate_liquidity(avg_volume=10_000, price=10.0)
        # ADV = 10,000 * 10 = $100,000 < $500,000
        assert result.passed is False
        assert "below minimum" in result.message

    def test_exactly_at_threshold(self):
        """Liquidity exactly at threshold should pass."""
        config = DataQualityConfig(min_adv_dollars=500_000)
        gates = DataQualityGates(config)
        result = gates.validate_liquidity(avg_volume=50_000, price=10.0)
        # ADV = 50,000 * 10 = $500,000 = threshold
        assert result.passed is True

    def test_missing_volume(self):
        """Missing volume should fail."""
        gates = DataQualityGates()
        result = gates.validate_liquidity(avg_volume=None, price=10.0)
        assert result.passed is False
        assert "Missing" in result.message

    def test_missing_price(self):
        """Missing price should fail."""
        gates = DataQualityGates()
        result = gates.validate_liquidity(avg_volume=100_000, price=None)
        assert result.passed is False

    def test_zero_volume(self):
        """Zero volume should fail."""
        gates = DataQualityGates()
        result = gates.validate_liquidity(avg_volume=0, price=10.0)
        assert result.passed is False
        assert "Invalid" in result.message

    def test_negative_volume(self):
        """Negative volume should fail."""
        gates = DataQualityGates()
        result = gates.validate_liquidity(avg_volume=-100, price=10.0)
        assert result.passed is False


class TestValidatePrice:
    """Tests for price validation."""

    def test_valid_price(self):
        """Price above threshold should pass."""
        gates = DataQualityGates()
        result = gates.validate_price(10.0)
        assert result.passed is True

    def test_penny_stock(self):
        """Price below threshold should fail."""
        gates = DataQualityGates()
        result = gates.validate_price(3.0)
        assert result.passed is False
        assert "below minimum" in result.message

    def test_exactly_at_threshold(self):
        """Price exactly at threshold should pass."""
        config = DataQualityConfig(min_price=5.0)
        gates = DataQualityGates(config)
        result = gates.validate_price(5.0)
        assert result.passed is True

    def test_missing_price(self):
        """Missing price should fail."""
        gates = DataQualityGates()
        result = gates.validate_price(None)
        assert result.passed is False
        assert "Missing" in result.message

    def test_custom_threshold(self):
        """Custom price threshold should be respected."""
        config = DataQualityConfig(min_price=10.0)
        gates = DataQualityGates(config)

        result = gates.validate_price(8.0)
        assert result.passed is False

        result = gates.validate_price(12.0)
        assert result.passed is True


class TestValidateEnrollment:
    """Tests for enrollment validation."""

    def test_sufficient_enrollment(self):
        """Sufficient enrollment should pass."""
        gates = DataQualityGates()
        result = gates.validate_enrollment(100)
        assert result.passed is True

    def test_low_enrollment(self):
        """Low enrollment should fail."""
        gates = DataQualityGates()
        result = gates.validate_enrollment(5)
        assert result.passed is False
        assert "below minimum" in result.message

    def test_exactly_at_threshold(self):
        """Enrollment exactly at threshold should pass."""
        config = DataQualityConfig(min_enrollment=10)
        gates = DataQualityGates(config)
        result = gates.validate_enrollment(10)
        assert result.passed is True

    def test_none_enrollment(self):
        """None enrollment should pass (missing is acceptable)."""
        gates = DataQualityGates()
        result = gates.validate_enrollment(None)
        assert result.passed is True


class TestValidateFinancialCoverage:
    """Tests for financial coverage validation."""

    def test_full_coverage(self):
        """All required fields present should pass."""
        gates = DataQualityGates()
        data = {"Cash": 100000000, "NetIncome": -50000000}
        result = gates.validate_financial_coverage(data)
        assert result.passed is True

    def test_partial_coverage(self):
        """Partial coverage at threshold (50%) passes - coverage >= threshold.

        Note: Implementation uses `<` not `<=`, so exactly 50% passes.
        """
        gates = DataQualityGates()
        data = {"Cash": 100000000}  # Missing NetIncome = 50% coverage
        result = gates.validate_financial_coverage(data)
        # At exactly threshold (50%), it passes because check is `<` not `<=`
        assert result.passed is True
        assert result.value == 0.5

    def test_no_data(self):
        """Empty data should fail."""
        gates = DataQualityGates()
        result = gates.validate_financial_coverage({})
        assert result.passed is False
        assert "No financial data" in result.message

    def test_none_values(self):
        """Fields with None values should not count as present.

        Note: Coverage = 50% (only NetIncome), which equals threshold so passes.
        """
        gates = DataQualityGates()
        data = {"Cash": None, "NetIncome": -50000000}
        result = gates.validate_financial_coverage(data)
        # Coverage = 1/2 = 50% = threshold, so passes (check is `<` not `<=`)
        assert result.passed is True
        assert result.value == 0.5


class TestValidateTickerData:
    """Tests for validate_ticker_data comprehensive validation."""

    def test_all_data_valid(self):
        """All valid data should pass."""
        gates = DataQualityGates()
        result = gates.validate_ticker_data(
            ticker="ACME",
            financial_data={
                "Cash": 100000000,
                "NetIncome": -50000000,
                "data_date": "2026-01-10",
            },
            market_data={
                "price": 45.50,
                "avg_volume": 100000,
            },
            as_of_date="2026-01-15",
        )
        assert result.passed is True
        assert result.ticker == "ACME"
        assert len(result.failures) == 0

    def test_stale_financial_data(self):
        """Stale financial data should generate flag."""
        gates = DataQualityGates()
        result = gates.validate_ticker_data(
            ticker="ACME",
            financial_data={
                "Cash": 100000000,
                "NetIncome": -50000000,
                "data_date": "2025-08-01",  # Very old
            },
            as_of_date="2026-01-15",
        )
        assert result.passed is False
        assert "stale_financial_data" in result.flags

    def test_penny_stock_flag(self):
        """Penny stock should generate flag."""
        gates = DataQualityGates()
        result = gates.validate_ticker_data(
            ticker="PENNY",
            market_data={"price": 2.50, "avg_volume": 100000},
        )
        assert "penny_stock" in result.flags

    def test_low_liquidity_flag(self):
        """Low liquidity should generate flag."""
        gates = DataQualityGates()
        result = gates.validate_ticker_data(
            ticker="ILLIQUID",
            market_data={"price": 10.0, "avg_volume": 1000},  # Very low volume
        )
        assert "low_liquidity" in result.flags

    def test_warnings_property(self):
        """Warnings property should return failure messages."""
        gates = DataQualityGates()
        result = gates.validate_ticker_data(
            ticker="ACME",
            market_data={"price": 2.0},  # Penny stock
        )
        assert len(result.warnings) > 0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_validate_financial_staleness_function(self):
        """Convenience function should work correctly."""
        assert validate_financial_staleness("2026-01-10", "2026-01-15") is True
        assert validate_financial_staleness("2025-08-01", "2026-01-15") is False

    def test_validate_financial_staleness_custom_threshold(self):
        """Convenience function should accept custom threshold."""
        assert validate_financial_staleness(
            "2026-01-10", "2026-01-15", max_age_days=3
        ) is False

    def test_validate_liquidity_function(self):
        """Convenience function should work correctly."""
        assert validate_liquidity(100000, 10.0) is True  # $1M ADV
        assert validate_liquidity(1000, 10.0) is False  # $10K ADV


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig."""

    def test_default_values(self):
        """Default values should be set."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 0.50
        assert config.warning_threshold == 0.20
        assert config.min_records_for_check == 10
        assert config.strict_mode is False


class TestCheckCircuitBreaker:
    """Tests for check_circuit_breaker function."""

    def test_below_warning_threshold(self):
        """Low failure rate should pass without warning."""
        result = check_circuit_breaker(total_records=100, failed_records=10)
        assert result.tripped is False
        assert result.warning is False
        assert result.failure_rate == 0.10

    def test_warning_threshold(self):
        """Failure rate above warning should trigger warning."""
        result = check_circuit_breaker(total_records=100, failed_records=30)
        assert result.tripped is False
        assert result.warning is True
        assert result.failure_rate == 0.30

    def test_failure_threshold(self):
        """Failure rate above failure threshold should trip."""
        result = check_circuit_breaker(total_records=100, failed_records=60)
        assert result.tripped is True
        assert result.warning is True
        assert result.failure_rate == 0.60

    def test_skip_small_batch(self):
        """Small batch should skip check."""
        result = check_circuit_breaker(total_records=5, failed_records=4)
        assert result.tripped is False
        assert "Skipped" in result.message

    def test_custom_thresholds(self):
        """Custom thresholds should be respected."""
        config = CircuitBreakerConfig(
            failure_threshold=0.30,
            warning_threshold=0.10,
        )
        result = check_circuit_breaker(
            total_records=100,
            failed_records=35,
            config=config,
        )
        assert result.tripped is True

    def test_strict_mode_raises(self):
        """Strict mode should raise exception on trip."""
        config = CircuitBreakerConfig(strict_mode=True)
        with pytest.raises(CircuitBreakerError):
            check_circuit_breaker(
                total_records=100,
                failed_records=60,
                config=config,
            )

    def test_zero_records(self):
        """Zero records should not cause division by zero."""
        result = check_circuit_breaker(total_records=0, failed_records=0)
        assert result.tripped is False

    def test_context_in_message(self):
        """Context should appear in message."""
        result = check_circuit_breaker(
            total_records=100,
            failed_records=60,
            context="test validation",
        )
        assert "test validation" in result.message


class TestValidateBatchWithCircuitBreaker:
    """Tests for validate_batch_with_circuit_breaker function."""

    def test_all_valid(self):
        """All valid records should pass."""
        records = [{"value": 1}, {"value": 2}, {"value": 3}]

        def validator(r):
            return True, []

        valid, invalid, cb_result = validate_batch_with_circuit_breaker(
            records, validator
        )
        assert len(valid) == 3
        assert len(invalid) == 0
        assert cb_result.tripped is False

    def test_some_invalid(self):
        """Invalid records should be separated."""
        records = [{"value": 1}, {"value": -1}, {"value": 2}]

        def validator(r):
            if r["value"] < 0:
                return False, ["Negative value"]
            return True, []

        valid, invalid, cb_result = validate_batch_with_circuit_breaker(
            records, validator
        )
        assert len(valid) == 2
        assert len(invalid) == 1
        assert invalid[0]["errors"] == ["Negative value"]

    def test_circuit_breaker_trips(self):
        """Circuit breaker should trip on high failure rate."""
        # Create 20 records, 15 invalid
        records = [{"value": i} for i in range(20)]

        def validator(r):
            if r["value"] < 15:
                return False, ["Too low"]
            return True, []

        valid, invalid, cb_result = validate_batch_with_circuit_breaker(
            records, validator
        )
        assert cb_result.tripped is True
        assert cb_result.failure_rate == 0.75
