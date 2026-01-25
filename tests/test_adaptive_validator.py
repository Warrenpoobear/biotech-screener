"""
Tests for Adaptive Validator Module

Tests the adaptive validation runner with configurable thresholds,
fallback mechanisms, and qualified pass logic.

Author: Wake Robin Capital Management
"""

import pytest
from decimal import Decimal
from typing import Dict, Any

from validation.validation_config import (
    AdaptiveCoverageConfig,
    CoverageMode,
    FallbackScoringConfig,
    FallbackStrategy,
    ValidationCriteriaConfig,
    ValidationOutcome,
)
from validation.adaptive_validator import (
    AdaptiveValidator,
    CoverageReport,
    FallbackActivation,
    run_adaptive_validation,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_coverage_data() -> Dict[str, Any]:
    """Sample coverage data for testing."""
    return {
        "financial": {
            "coverage_percent": "0.65",
            "missing_tickers": ["ACME", "BETA"],
        },
        "clinical": {
            "coverage_percent": "0.55",
            "missing_tickers": ["GAMMA"],
        },
        "catalyst": {
            "coverage_percent": "0.60",
            "missing_tickers": [],
        },
        "institutional": {
            "coverage_percent": "0.40",
            "missing_tickers": ["DELTA", "EPSILON"],
        },
    }


@pytest.fixture
def sample_backtest_results() -> Dict[str, Any]:
    """Sample backtest results for testing."""
    return {
        "ic_mean": "0.08",
        "hit_rate_top_quintile": "0.58",
        "max_drawdown": "0.35",
        "turnover_annualized": "0.65",
        "n_observations": 450,
    }


@pytest.fixture
def sample_component_metrics() -> Dict[str, Dict[str, Any]]:
    """Sample component metrics for testing."""
    return {
        "pos": {
            "calibration_confidence": "0.55",
            "weight": "0.20",
            "score": "65.0",
        },
        "trial_quality": {
            "expert_agreement": "0.75",
            "weight": "0.10",
            "score": "70.0",
        },
        "competitive_pressure": {
            "ic": "0.04",
            "weight": "0.05",
            "score": "55.0",
        },
        "regime": {
            "drawdown_reduction": "0.08",
            "weight": "0.10",
            "score": "60.0",
        },
        "momentum": {
            "ic": "0.05",
            "weight": "0.10",
            "score": "72.0",
        },
        "smart_money": {
            "coverage": "0.45",
            "weight": "0.05",
            "score": "58.0",
        },
    }


# =============================================================================
# CoverageReport Tests
# =============================================================================

class TestCoverageReport:
    """Tests for CoverageReport."""

    def test_report_creation(self):
        """Report can be created with valid parameters."""
        report = CoverageReport(
            category="financial",
            coverage_pct=Decimal("0.65"),
            confidence_multiplier=Decimal("0.70"),
            tier_name="acceptable",
            meets_threshold=True,
            threshold_used=Decimal("0.50"),
        )
        assert report.category == "financial"
        assert report.coverage_pct == Decimal("0.65")
        assert report.meets_threshold is True

    def test_report_with_missing_tickers(self):
        """Report can include missing tickers."""
        report = CoverageReport(
            category="clinical",
            coverage_pct=Decimal("0.50"),
            confidence_multiplier=Decimal("0.60"),
            tier_name="marginal",
            meets_threshold=False,
            threshold_used=Decimal("0.60"),
            missing_tickers=["ACME", "BETA", "GAMMA"],
        )
        assert len(report.missing_tickers) == 3

    def test_to_dict(self):
        """Report serializes to dict correctly."""
        report = CoverageReport(
            category="catalyst",
            coverage_pct=Decimal("0.75"),
            confidence_multiplier=Decimal("0.80"),
            tier_name="good",
            meets_threshold=True,
            threshold_used=Decimal("0.60"),
        )
        d = report.to_dict()
        assert d["category"] == "catalyst"
        assert d["coverage_pct"] == "0.75"
        assert d["meets_threshold"] is True


class TestFallbackActivation:
    """Tests for FallbackActivation."""

    def test_activation_creation(self):
        """Activation can be created with required fields."""
        activation = FallbackActivation(
            component="pos",
            strategy=FallbackStrategy.STAGE_ONLY,
            reason="calibration_confidence 0.35 < 0.40",
        )
        assert activation.component == "pos"
        assert activation.strategy == FallbackStrategy.STAGE_ONLY

    def test_activation_with_values(self):
        """Activation can include value adjustments."""
        activation = FallbackActivation(
            component="momentum",
            strategy=FallbackStrategy.NEUTRAL_SCORE,
            reason="ic 0.02 < 0.03",
            original_value=Decimal("72.0"),
            fallback_value=Decimal("50.0"),
            weight_adjustment=Decimal("0.05"),
        )
        assert activation.original_value == Decimal("72.0")
        assert activation.fallback_value == Decimal("50.0")

    def test_to_dict(self):
        """Activation serializes to dict correctly."""
        activation = FallbackActivation(
            component="regime",
            strategy=FallbackStrategy.STATIC_WEIGHTS,
            reason="drawdown_reduction 0.03 < 0.05",
        )
        d = activation.to_dict()
        assert d["component"] == "regime"
        assert d["strategy"] == "static_weights"
        assert d["reason"] == "drawdown_reduction 0.03 < 0.05"


# =============================================================================
# AdaptiveValidator Tests
# =============================================================================

class TestAdaptiveValidatorInit:
    """Tests for AdaptiveValidator initialization."""

    def test_default_init(self):
        """Validator initializes with default configs."""
        validator = AdaptiveValidator()
        assert validator.coverage_config.coverage_mode == CoverageMode.ADAPTIVE
        assert validator.active_fallbacks == []

    def test_custom_coverage_config(self):
        """Validator accepts custom coverage config."""
        coverage_config = AdaptiveCoverageConfig(coverage_mode=CoverageMode.LENIENT)
        validator = AdaptiveValidator(coverage_config=coverage_config)
        assert validator.coverage_config.coverage_mode == CoverageMode.LENIENT

    def test_custom_fallback_config(self):
        """Validator accepts custom fallback config."""
        fallback_config = FallbackScoringConfig(
            pos_fallback_strategy=FallbackStrategy.DISABLE
        )
        validator = AdaptiveValidator(fallback_config=fallback_config)
        assert validator.fallback_config.pos_fallback_strategy == FallbackStrategy.DISABLE


class TestValidateCoverage:
    """Tests for validate_coverage method."""

    def test_validate_coverage_adaptive_mode(self, sample_coverage_data):
        """Coverage validation with adaptive mode."""
        validator = AdaptiveValidator()
        reports, overall_pass = validator.validate_coverage(
            sample_coverage_data, universe_size=100
        )

        assert len(reports) == 4  # financial, clinical, catalyst, institutional
        # With 65% financial, 55% clinical, 60% catalyst, 40% institutional
        # in ADAPTIVE mode, most should pass

    def test_validate_coverage_strict_mode(self, sample_coverage_data):
        """Coverage validation with strict mode fails with low coverage."""
        config = AdaptiveCoverageConfig(coverage_mode=CoverageMode.STRICT)
        validator = AdaptiveValidator(coverage_config=config)
        reports, overall_pass = validator.validate_coverage(
            sample_coverage_data, universe_size=100
        )

        # With 65% financial in STRICT mode (requires 90%), should fail
        financial_report = next(r for r in reports if r.category == "financial")
        assert not financial_report.meets_threshold

    def test_validate_coverage_minimal_mode(self, sample_coverage_data):
        """Coverage validation with minimal mode passes with low coverage."""
        config = AdaptiveCoverageConfig(coverage_mode=CoverageMode.MINIMAL)
        validator = AdaptiveValidator(coverage_config=config)
        reports, overall_pass = validator.validate_coverage(
            sample_coverage_data, universe_size=100
        )

        # With minimal mode, all should pass
        assert all(r.meets_threshold for r in reports)


class TestCheckAndActivateFallbacks:
    """Tests for fallback activation."""

    def test_no_fallbacks_when_metrics_good(self, sample_component_metrics):
        """No fallbacks activated when all metrics are good."""
        validator = AdaptiveValidator()
        updated = validator.check_and_activate_fallbacks(sample_component_metrics)

        # Most components should pass without fallback
        assert len(validator.active_fallbacks) <= 1  # At most one marginal

    def test_fallback_activated_for_low_pos_confidence(self):
        """Fallback activated when PoS confidence is low."""
        component_metrics = {
            "pos": {
                "calibration_confidence": "0.30",  # Below 0.40 threshold
                "weight": "0.20",
                "score": "65.0",
            },
        }
        validator = AdaptiveValidator()
        updated = validator.check_and_activate_fallbacks(component_metrics)

        assert len(validator.active_fallbacks) == 1
        assert validator.active_fallbacks[0].component == "pos"
        assert validator.active_fallbacks[0].strategy == FallbackStrategy.STAGE_ONLY

    def test_fallback_activated_for_low_momentum_ic(self):
        """Fallback activated when momentum IC is low."""
        component_metrics = {
            "momentum": {
                "ic": "0.01",  # Below 0.03 threshold
                "weight": "0.10",
                "score": "72.0",
            },
        }
        validator = AdaptiveValidator()
        updated = validator.check_and_activate_fallbacks(component_metrics)

        assert len(validator.active_fallbacks) == 1
        assert validator.active_fallbacks[0].component == "momentum"
        assert validator.active_fallbacks[0].strategy == FallbackStrategy.NEUTRAL_SCORE

    def test_fallback_updates_metrics(self):
        """Fallback activation updates metrics correctly."""
        component_metrics = {
            "smart_money": {
                "coverage": "0.20",  # Below 0.30 threshold
                "weight": "0.05",
                "score": "58.0",
            },
        }
        validator = AdaptiveValidator()
        updated = validator.check_and_activate_fallbacks(component_metrics)

        # Smart money should be disabled
        assert updated["smart_money"]["fallback_active"] is True
        assert updated["smart_money"]["fallback_strategy"] == "disable"
        assert updated["smart_money"]["weight"] == "0.0"


class TestValidateBacktestMetrics:
    """Tests for backtest metrics validation."""

    def test_validate_good_metrics(self, sample_backtest_results):
        """Good backtest metrics pass validation."""
        validator = AdaptiveValidator()
        results = validator.validate_backtest_metrics(sample_backtest_results)

        assert len(results) == 4  # ic, hit_rate, max_drawdown, turnover

        # All should pass with sample data
        ic_result = next(r for r in results if r.metric_name == "ic")
        assert ic_result.outcome == ValidationOutcome.PASS

    def test_validate_marginal_ic(self):
        """Marginal IC returns qualified pass."""
        backtest = {
            "ic_mean": "0.04",  # Between qualified (0.03) and pass (0.05)
            "hit_rate_top_quintile": "0.58",
            "max_drawdown": "0.35",
            "turnover_annualized": "0.65",
        }
        validator = AdaptiveValidator()
        results = validator.validate_backtest_metrics(backtest)

        ic_result = next(r for r in results if r.metric_name == "ic")
        assert ic_result.outcome == ValidationOutcome.QUALIFIED_PASS

    def test_validate_failing_ic(self):
        """Low IC returns fail."""
        backtest = {
            "ic_mean": "0.01",  # Below all thresholds
            "hit_rate_top_quintile": "0.58",
            "max_drawdown": "0.35",
            "turnover_annualized": "0.65",
        }
        validator = AdaptiveValidator()
        results = validator.validate_backtest_metrics(backtest)

        ic_result = next(r for r in results if r.metric_name == "ic")
        assert ic_result.outcome == ValidationOutcome.FAIL

    def test_validate_high_drawdown(self):
        """High drawdown returns warn or fail."""
        backtest = {
            "ic_mean": "0.08",
            "hit_rate_top_quintile": "0.58",
            "max_drawdown": "0.52",  # Above qualified threshold
            "turnover_annualized": "0.65",
        }
        validator = AdaptiveValidator()
        results = validator.validate_backtest_metrics(backtest)

        dd_result = next(r for r in results if r.metric_name == "max_drawdown")
        assert dd_result.outcome in (ValidationOutcome.WARN, ValidationOutcome.FAIL)


class TestValidateSampleSize:
    """Tests for sample size validation."""

    def test_full_sample_size(self):
        """Sample size >= 500 is adequate for full validation."""
        validator = AdaptiveValidator()
        adequate, level = validator.validate_sample_size(600)
        assert adequate is True
        assert level == "full"

    def test_qualified_sample_size(self):
        """Sample size between 200-500 is adequate for qualified validation."""
        validator = AdaptiveValidator()
        adequate, level = validator.validate_sample_size(300)
        assert adequate is True
        assert level == "qualified"

    def test_minimal_sample_size(self):
        """Sample size between 50-200 is adequate for minimal validation."""
        validator = AdaptiveValidator()
        adequate, level = validator.validate_sample_size(75)
        assert adequate is True
        assert level == "minimal"

    def test_insufficient_sample_size(self):
        """Sample size below 50 is insufficient."""
        validator = AdaptiveValidator()
        adequate, level = validator.validate_sample_size(30)
        assert adequate is False
        assert level == "insufficient"


class TestValidatePipelineRun:
    """Tests for full pipeline validation."""

    def test_full_validation_pass(
        self, sample_coverage_data, sample_backtest_results, sample_component_metrics
    ):
        """Full validation with good data passes."""
        validator = AdaptiveValidator()
        summary = validator.validate_pipeline_run(
            coverage_data=sample_coverage_data,
            backtest_results=sample_backtest_results,
            component_metrics=sample_component_metrics,
            universe_size=100,
        )

        assert summary.overall_outcome in (
            ValidationOutcome.PASS,
            ValidationOutcome.QUALIFIED_PASS,
        )

    def test_validation_with_fallbacks(self, sample_coverage_data, sample_backtest_results):
        """Validation tracks active fallbacks."""
        component_metrics = {
            "pos": {"calibration_confidence": "0.30", "weight": "0.20", "score": "65.0"},
            "momentum": {"ic": "0.01", "weight": "0.10", "score": "72.0"},
        }

        validator = AdaptiveValidator()
        summary = validator.validate_pipeline_run(
            coverage_data=sample_coverage_data,
            backtest_results=sample_backtest_results,
            component_metrics=component_metrics,
            universe_size=100,
        )

        assert len(summary.fallbacks_active) == 2
        assert "pos" in summary.fallbacks_active
        assert "momentum" in summary.fallbacks_active

    def test_validation_downgrades_for_small_sample(
        self, sample_coverage_data, sample_component_metrics
    ):
        """Validation downgrades outcome for small sample size."""
        backtest = {
            "ic_mean": "0.08",
            "hit_rate_top_quintile": "0.58",
            "max_drawdown": "0.35",
            "turnover_annualized": "0.65",
            "n_observations": 150,  # Below full but above qualified
        }

        validator = AdaptiveValidator()
        summary = validator.validate_pipeline_run(
            coverage_data=sample_coverage_data,
            backtest_results=backtest,
            component_metrics=sample_component_metrics,
            universe_size=100,
        )

        # Should be at most QUALIFIED_PASS due to sample size
        assert summary.overall_outcome in (
            ValidationOutcome.QUALIFIED_PASS,
            ValidationOutcome.WARN,
        )

    def test_validation_summary_includes_limitations(
        self, sample_coverage_data, sample_backtest_results
    ):
        """Validation summary includes relevant limitations."""
        component_metrics = {
            "pos": {"calibration_confidence": "0.30", "weight": "0.20", "score": "65.0"},
        }

        validator = AdaptiveValidator()
        summary = validator.validate_pipeline_run(
            coverage_data=sample_coverage_data,
            backtest_results=sample_backtest_results,
            component_metrics=component_metrics,
            universe_size=100,
        )

        limitations = summary.get_limitations()
        assert any("fallback" in lim.lower() for lim in limitations)


class TestConvenienceFunction:
    """Tests for run_adaptive_validation convenience function."""

    def test_run_adaptive_validation(
        self, sample_coverage_data, sample_backtest_results
    ):
        """Convenience function runs validation correctly."""
        summary = run_adaptive_validation(
            coverage_data=sample_coverage_data,
            backtest_results=sample_backtest_results,
            coverage_mode=CoverageMode.ADAPTIVE,
            universe_size=100,
        )

        assert summary.coverage_mode == CoverageMode.ADAPTIVE
        assert summary.overall_outcome is not None

    def test_run_with_lenient_mode(
        self, sample_coverage_data, sample_backtest_results
    ):
        """Convenience function works with lenient mode."""
        summary = run_adaptive_validation(
            coverage_data=sample_coverage_data,
            backtest_results=sample_backtest_results,
            coverage_mode=CoverageMode.LENIENT,
        )

        assert summary.coverage_mode == CoverageMode.LENIENT


class TestConfigurationSummary:
    """Tests for configuration summary."""

    def test_get_configuration_summary(self):
        """Configuration summary includes all config sections."""
        validator = AdaptiveValidator()
        summary = validator.get_configuration_summary()

        assert "version" in summary
        assert "coverage_config" in summary
        assert "fallback_config" in summary
        assert "criteria_config" in summary

    def test_get_active_fallbacks_report(self):
        """Active fallbacks report works correctly."""
        validator = AdaptiveValidator()

        # First, activate some fallbacks
        component_metrics = {
            "pos": {"calibration_confidence": "0.30", "weight": "0.20", "score": "65.0"},
        }
        validator.check_and_activate_fallbacks(component_metrics)

        report = validator.get_active_fallbacks_report()
        assert len(report) == 1
        assert report[0]["component"] == "pos"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_coverage_data(self):
        """Handles empty coverage data gracefully."""
        validator = AdaptiveValidator()
        reports, overall_pass = validator.validate_coverage({}, universe_size=100)

        assert len(reports) == 4  # Still creates reports for all categories
        for report in reports:
            assert report.coverage_pct == Decimal("0")

    def test_zero_universe_size(self, sample_coverage_data):
        """Handles zero universe size gracefully."""
        validator = AdaptiveValidator()
        reports, overall_pass = validator.validate_coverage(
            sample_coverage_data, universe_size=0
        )
        # Should not crash

    def test_missing_backtest_fields(self):
        """Handles missing backtest fields gracefully."""
        validator = AdaptiveValidator()
        results = validator.validate_backtest_metrics({})

        # Should create results with zero values
        assert len(results) == 4

    def test_decimal_string_conversion(self):
        """Properly converts decimal strings in metrics."""
        component_metrics = {
            "pos": {
                "calibration_confidence": "0.45",  # String that should be converted
                "weight": "0.20",
                "score": "65.0",
            },
        }
        validator = AdaptiveValidator()
        updated = validator.check_and_activate_fallbacks(component_metrics)

        # Should not crash and should parse correctly
        assert len(validator.active_fallbacks) == 0  # 0.45 > 0.40 threshold
