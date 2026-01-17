"""
Tests for robustness utilities.
"""
import pytest
import time
from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import patch, MagicMock

from common.robustness import (
    # Data staleness
    DataFreshnessConfig,
    DataFreshnessResult,
    validate_data_freshness,
    validate_record_freshness,
    # Cross-module consistency
    ConsistencyReport,
    validate_ticker_coverage,
    validate_module_handoff,
    # Retry logic
    RetryConfig,
    retry_with_backoff,
    RetryExhaustedError,
    # Memory guards
    MemoryGuardConfig,
    chunk_universe,
    estimate_memory_usage,
    # Structured logging
    CorrelationContext,
    get_correlation_id,
    set_correlation_id,
    with_correlation_id,
    CorrelatedLogger,
    # Graceful degradation
    DegradationReport,
    GracefulDegradationConfig,
    compute_with_degradation,
)


# ============================================================================
# DATA STALENESS TESTS
# ============================================================================

class TestDataFreshness:
    """Tests for data freshness validation."""

    def test_fresh_data_passes(self):
        """Recent data should pass validation."""
        result = validate_data_freshness(
            record_date="2026-01-10",
            as_of_date="2026-01-15",
            data_type="financial",
        )
        assert result.is_fresh is True
        assert result.age_days == 5
        assert result.is_warning is False

    def test_stale_data_fails(self):
        """Old data should fail validation."""
        result = validate_data_freshness(
            record_date="2025-09-01",
            as_of_date="2026-01-15",
            data_type="financial",
        )
        assert result.is_fresh is False
        assert result.age_days > 120

    def test_warning_threshold(self):
        """Data approaching staleness should trigger warning."""
        config = DataFreshnessConfig(
            max_financial_age_days=100,
            warn_threshold_pct=0.75,
        )
        # 80 days old is above 75% of 100 day max
        result = validate_data_freshness(
            record_date="2025-10-27",
            as_of_date="2026-01-15",
            data_type="financial",
            config=config,
        )
        assert result.is_fresh is True
        assert result.is_warning is True

    def test_market_data_stricter(self):
        """Market data should have stricter freshness requirements."""
        config = DataFreshnessConfig(max_market_data_age_days=5)
        result = validate_data_freshness(
            record_date="2026-01-08",
            as_of_date="2026-01-15",
            data_type="market",
            config=config,
        )
        assert result.is_fresh is False  # 7 days > 5 day max

    def test_strict_mode_raises(self):
        """Strict mode should raise exception on stale data."""
        config = DataFreshnessConfig(
            max_financial_age_days=30,
            strict_mode=True,
        )
        with pytest.raises(ValueError, match="stale"):
            validate_data_freshness(
                record_date="2025-11-01",
                as_of_date="2026-01-15",
                data_type="financial",
                config=config,
            )

    def test_invalid_date_format(self):
        """Invalid date format should return not fresh."""
        result = validate_data_freshness(
            record_date="invalid-date",
            as_of_date="2026-01-15",
        )
        assert result.is_fresh is False
        assert result.age_days == -1

    def test_freshness_percentage(self):
        """Freshness percentage should be calculated correctly."""
        result = validate_data_freshness(
            record_date="2026-01-05",
            as_of_date="2026-01-15",
            data_type="financial",
            config=DataFreshnessConfig(max_financial_age_days=100),
        )
        assert result.freshness_pct == 10.0  # 10 days / 100 max


class TestRecordFreshness:
    """Tests for batch record freshness validation."""

    def test_all_fresh_records(self):
        """All fresh records should pass."""
        records = [
            {"ticker": "AAPL", "source_date": "2026-01-10"},
            {"ticker": "GOOG", "source_date": "2026-01-12"},
        ]
        result = validate_record_freshness(
            records, "2026-01-15", data_type="financial"
        )
        assert result["fresh_count"] == 2
        assert result["stale_count"] == 0

    def test_mixed_freshness(self):
        """Mix of fresh and stale records."""
        records = [
            {"ticker": "FRESH", "source_date": "2026-01-10"},
            {"ticker": "STALE", "source_date": "2025-06-01"},
        ]
        result = validate_record_freshness(
            records, "2026-01-15", data_type="financial"
        )
        assert result["fresh_count"] == 1
        assert result["stale_count"] == 1
        assert "STALE" in result["stale_records"]

    def test_missing_date_field(self):
        """Records without date field should be marked stale."""
        records = [
            {"ticker": "NODATE"},
        ]
        result = validate_record_freshness(
            records, "2026-01-15", data_type="financial"
        )
        assert result["stale_count"] == 1

    def test_empty_records(self):
        """Empty records list should return zeros."""
        result = validate_record_freshness([], "2026-01-15")
        assert result["total"] == 0
        assert result["fresh_count"] == 0


# ============================================================================
# CROSS-MODULE CONSISTENCY TESTS
# ============================================================================

class TestTickerCoverage:
    """Tests for cross-module ticker coverage validation."""

    def test_full_coverage(self):
        """Full coverage across all modules."""
        universe = {"A", "B", "C"}
        m2 = {"A", "B", "C"}
        m3 = {"A", "B", "C"}
        m4 = {"A", "B", "C"}

        report = validate_ticker_coverage(universe, m2, m3, m4)
        assert report.is_consistent is True
        assert report.coverage_m2_pct == 100.0
        assert len(report.missing_in_m2) == 0

    def test_partial_coverage(self):
        """Partial coverage should be detected."""
        universe = {"A", "B", "C", "D", "E"}
        m2 = {"A", "B", "C"}  # Missing D, E
        m3 = {"A", "B", "C", "D", "E"}
        m4 = {"A", "B", "C", "D", "E"}

        report = validate_ticker_coverage(universe, m2, m3, m4)
        assert report.is_consistent is False  # 60% < 80% default
        assert report.coverage_m2_pct == 60.0
        assert "D" in report.missing_in_m2
        assert "E" in report.missing_in_m2

    def test_orphan_tickers_detected(self):
        """Tickers in modules but not universe should be detected."""
        universe = {"A", "B"}
        m2 = {"A", "B", "ORPHAN"}
        m3 = {"A", "B"}
        m4 = {"A", "B"}

        report = validate_ticker_coverage(universe, m2, m3, m4)
        assert report.is_consistent is False
        assert "ORPHAN" in report.orphan_in_m2

    def test_custom_coverage_threshold(self):
        """Custom coverage threshold should be respected."""
        universe = {"A", "B", "C", "D"}
        m2 = {"A", "B"}  # 50% coverage
        m3 = {"A", "B", "C", "D"}
        m4 = {"A", "B", "C", "D"}

        # With 50% threshold, should pass
        report = validate_ticker_coverage(universe, m2, m3, m4, min_coverage_pct=0.50)
        assert report.is_consistent is True

    def test_empty_universe(self):
        """Empty universe should return consistent with warning."""
        report = validate_ticker_coverage(set(), set(), set(), set())
        assert report.is_consistent is True
        assert any("Empty" in w for w in report.warnings)


class TestModuleHandoff:
    """Tests for module handoff validation."""

    def test_valid_handoff(self):
        """Valid module output should pass handoff check."""
        output = {
            "scores": [],
            "diagnostic_counts": {},
            "as_of_date": "2026-01-15",
        }
        result = validate_module_handoff(
            "module_2", "module_5",
            output,
            ["scores", "diagnostic_counts", "as_of_date"],
        )
        assert result["is_valid"] is True

    def test_missing_required_field(self):
        """Missing required field should fail handoff."""
        output = {
            "scores": [],
            # Missing diagnostic_counts
        }
        result = validate_module_handoff(
            "module_2", "module_5",
            output,
            ["scores", "diagnostic_counts"],
        )
        assert result["is_valid"] is False
        assert "diagnostic_counts" in result["missing_fields"]


# ============================================================================
# RETRY LOGIC TESTS
# ============================================================================

class TestRetryWithBackoff:
    """Tests for retry with exponential backoff."""

    def test_success_no_retry(self):
        """Successful function should not retry."""
        call_count = 0

        @retry_with_backoff(RetryConfig(max_attempts=3))
        def succeed():
            nonlocal call_count
            call_count += 1
            return "success"

        result = succeed()
        assert result == "success"
        assert call_count == 1

    def test_retry_on_failure(self):
        """Function should retry on retryable exception."""
        call_count = 0

        @retry_with_backoff(RetryConfig(
            max_attempts=3,
            base_delay_seconds=0.01,  # Fast for testing
            retryable_exceptions=(IOError,),
        ))
        def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise IOError("temporary failure")
            return "success"

        result = fail_twice()
        assert result == "success"
        assert call_count == 3

    def test_exhausted_retries(self):
        """Should raise RetryExhaustedError when all attempts fail."""
        @retry_with_backoff(RetryConfig(
            max_attempts=2,
            base_delay_seconds=0.01,
            retryable_exceptions=(IOError,),
        ))
        def always_fail():
            raise IOError("persistent failure")

        with pytest.raises(RetryExhaustedError):
            always_fail()

    def test_non_retryable_exception(self):
        """Non-retryable exceptions should not trigger retry."""
        call_count = 0

        @retry_with_backoff(RetryConfig(
            max_attempts=3,
            retryable_exceptions=(IOError,),  # ValueError not included
        ))
        def raise_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("not retryable")

        with pytest.raises(ValueError):
            raise_value_error()
        assert call_count == 1  # Should not retry


# ============================================================================
# MEMORY GUARDS TESTS
# ============================================================================

class TestMemoryGuards:
    """Tests for memory guard utilities."""

    def test_small_universe_no_chunk(self):
        """Small universe should not be chunked."""
        tickers = ["A", "B", "C"]
        chunks = chunk_universe(tickers)
        assert len(chunks) == 1
        assert chunks[0] == tickers

    def test_large_universe_chunked(self):
        """Large universe should be chunked."""
        tickers = [f"T{i}" for i in range(500)]
        config = MemoryGuardConfig(chunk_size=100)
        chunks = chunk_universe(tickers, config)
        assert len(chunks) == 5
        assert len(chunks[0]) == 100

    def test_chunk_preserves_all_tickers(self):
        """Chunking should preserve all tickers."""
        tickers = [f"T{i}" for i in range(350)]
        config = MemoryGuardConfig(chunk_size=100)
        chunks = chunk_universe(tickers, config)

        all_chunked = [t for chunk in chunks for t in chunk]
        assert set(all_chunked) == set(tickers)

    def test_estimate_memory_usage(self):
        """Memory estimation should return reasonable values."""
        result = estimate_memory_usage(100)
        assert result["ticker_count"] == 100
        assert result["estimated_mb"] > 0
        assert result["recommended_chunks"] >= 1

    def test_memory_warning_large_universe(self):
        """Large universe should generate warning."""
        config = MemoryGuardConfig(max_universe_size=500)
        result = estimate_memory_usage(1000, config)
        assert result["warning"] is not None


# ============================================================================
# STRUCTURED LOGGING TESTS
# ============================================================================

class TestCorrelationId:
    """Tests for correlation ID management."""

    def test_get_generates_id(self):
        """get_correlation_id should generate ID if not set."""
        # Clear any existing ID
        import common.robustness as robustness
        if hasattr(robustness._correlation_context, 'id'):
            delattr(robustness._correlation_context, 'id')

        cid = get_correlation_id()
        assert cid is not None
        assert len(cid) == 8

    def test_set_and_get(self):
        """set_correlation_id should be retrievable."""
        set_correlation_id("test123")
        assert get_correlation_id() == "test123"

    def test_decorator_sets_id(self):
        """with_correlation_id decorator should set ID."""
        captured_id = None

        @with_correlation_id("decorated")
        def capture_id():
            nonlocal captured_id
            captured_id = get_correlation_id()

        capture_id()
        assert captured_id == "decorated"


class TestCorrelatedLogger:
    """Tests for correlated logger."""

    def test_message_includes_correlation_id(self):
        """Log messages should include correlation ID."""
        set_correlation_id("log123")
        logger = CorrelatedLogger("test")

        # Test format
        formatted = logger._format_message("test message")
        assert "[log123]" in formatted
        assert "test message" in formatted


# ============================================================================
# GRACEFUL DEGRADATION TESTS
# ============================================================================

class TestGracefulDegradation:
    """Tests for graceful degradation."""

    def test_no_degradation_when_complete(self):
        """No degradation when all components present."""
        components = {
            "financial": 70.0,
            "catalyst": 60.0,
            "clinical": 80.0,
        }
        report = compute_with_degradation(components)
        assert report.was_degraded is False
        assert len(report.missing_components) == 0
        assert report.confidence_adjustment == Decimal("1.0")

    def test_degradation_with_missing_component(self):
        """Missing component should trigger degradation."""
        components = {
            "financial": 70.0,
            "catalyst": None,  # Missing
            "clinical": 80.0,
        }
        report = compute_with_degradation(components)
        assert report.was_degraded is True
        assert "catalyst" in report.missing_components

    def test_penalty_applied_for_missing(self):
        """Penalty should be applied for missing components."""
        components = {
            "financial": None,
            "catalyst": None,
            "clinical": 80.0,
        }
        config = GracefulDegradationConfig(missing_penalty_pct=Decimal("0.10"))
        report = compute_with_degradation(components, config)

        # 2 missing * 10% = 20% penalty
        assert report.confidence_adjustment == Decimal("0.80")

    def test_critical_component_warning(self):
        """Missing critical component should generate warning."""
        components = {
            "financial": 70.0,
            "catalyst": 60.0,
            "clinical": None,  # Critical!
        }
        config = GracefulDegradationConfig(allow_missing_clinical=False)
        report = compute_with_degradation(components, config)

        assert any("clinical" in w for w in report.warnings)

    def test_insufficient_components_warning(self):
        """Too few components should generate warning."""
        components = {
            "financial": 70.0,
            "catalyst": None,
            "clinical": None,
        }
        config = GracefulDegradationConfig(min_components_required=2)
        report = compute_with_degradation(components, config)

        assert any("Insufficient" in w for w in report.warnings)

    def test_penalty_capped_at_50_percent(self):
        """Penalty should not exceed 50%."""
        components = {
            "a": None,
            "b": None,
            "c": None,
            "d": None,
            "e": None,
        }
        config = GracefulDegradationConfig(missing_penalty_pct=Decimal("0.20"))
        report = compute_with_degradation(components, config)

        # 5 * 20% = 100%, but capped at 50%
        assert report.confidence_adjustment == Decimal("0.50")
