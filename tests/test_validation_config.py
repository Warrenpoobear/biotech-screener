"""
Tests for Validation Configuration Module

Tests the adaptive coverage configuration, fallback mechanisms, and validation
criteria with qualified pass options.

Author: Wake Robin Capital Management
"""

import pytest
from decimal import Decimal
from typing import Dict, Any

from validation.validation_config import (
    AdaptiveCoverageConfig,
    CoverageMode,
    CoverageTierConfig,
    FallbackScoringConfig,
    FallbackStrategy,
    ValidationCriteriaConfig,
    ValidationOutcome,
    ValidationResult,
    ValidationSummary,
    DEFAULT_COVERAGE_TIERS,
    get_strict_config,
    get_standard_config,
    get_adaptive_config,
    get_lenient_config,
    get_minimal_config,
)


class TestCoverageMode:
    """Tests for CoverageMode enum."""

    def test_all_modes_exist(self):
        """All expected coverage modes exist."""
        assert CoverageMode.STRICT.value == "strict"
        assert CoverageMode.STANDARD.value == "standard"
        assert CoverageMode.ADAPTIVE.value == "adaptive"
        assert CoverageMode.LENIENT.value == "lenient"
        assert CoverageMode.MINIMAL.value == "minimal"


class TestCoverageTierConfig:
    """Tests for CoverageTierConfig."""

    def test_tier_creation(self):
        """Tier can be created with valid parameters."""
        tier = CoverageTierConfig(
            tier_name="excellent",
            min_coverage_pct=Decimal("0.90"),
            confidence_multiplier=Decimal("1.00"),
            description="Full confidence",
        )
        assert tier.tier_name == "excellent"
        assert tier.min_coverage_pct == Decimal("0.90")
        assert tier.confidence_multiplier == Decimal("1.00")

    def test_tier_to_dict(self):
        """Tier converts to dict correctly."""
        tier = CoverageTierConfig(
            tier_name="good",
            min_coverage_pct=Decimal("0.75"),
            confidence_multiplier=Decimal("0.85"),
            description="Minor haircut",
        )
        d = tier.to_dict()
        assert d["tier_name"] == "good"
        assert d["min_coverage_pct"] == "0.75"
        assert d["confidence_multiplier"] == "0.85"


class TestDefaultCoverageTiers:
    """Tests for default coverage tier definitions."""

    def test_all_categories_have_tiers(self):
        """All expected categories have tier definitions."""
        assert "financial" in DEFAULT_COVERAGE_TIERS
        assert "clinical" in DEFAULT_COVERAGE_TIERS
        assert "catalyst" in DEFAULT_COVERAGE_TIERS
        assert "institutional" in DEFAULT_COVERAGE_TIERS

    def test_tiers_are_sorted_descending(self):
        """Tiers are sorted from highest to lowest coverage."""
        for category, tiers in DEFAULT_COVERAGE_TIERS.items():
            for i in range(len(tiers) - 1):
                assert tiers[i].min_coverage_pct >= tiers[i + 1].min_coverage_pct, (
                    f"Tiers in {category} not sorted descending"
                )

    def test_lowest_tier_is_zero(self):
        """Lowest tier in each category starts at 0."""
        for category, tiers in DEFAULT_COVERAGE_TIERS.items():
            assert tiers[-1].min_coverage_pct == Decimal("0.00"), (
                f"Lowest tier in {category} should be 0"
            )


class TestAdaptiveCoverageConfig:
    """Tests for AdaptiveCoverageConfig."""

    def test_default_mode_is_adaptive(self):
        """Default coverage mode is ADAPTIVE."""
        config = AdaptiveCoverageConfig()
        assert config.coverage_mode == CoverageMode.ADAPTIVE

    def test_get_mode_thresholds_strict(self):
        """Strict mode has highest thresholds."""
        config = AdaptiveCoverageConfig(coverage_mode=CoverageMode.STRICT)
        thresholds = config.get_mode_thresholds()
        assert thresholds["financial"] == Decimal("0.90")
        assert thresholds["clinical"] == Decimal("0.70")

    def test_get_mode_thresholds_minimal(self):
        """Minimal mode has lowest thresholds."""
        config = AdaptiveCoverageConfig(coverage_mode=CoverageMode.MINIMAL)
        thresholds = config.get_mode_thresholds()
        assert thresholds["financial"] == Decimal("0.30")
        assert thresholds["clinical"] == Decimal("0.25")

    def test_get_confidence_for_coverage_excellent(self):
        """Excellent coverage returns full confidence."""
        config = AdaptiveCoverageConfig()
        confidence, tier_name = config.get_confidence_for_coverage(
            "financial", Decimal("0.95")
        )
        assert confidence == Decimal("1.00")
        assert tier_name == "excellent"

    def test_get_confidence_for_coverage_marginal(self):
        """Marginal coverage returns reduced confidence."""
        config = AdaptiveCoverageConfig()
        confidence, tier_name = config.get_confidence_for_coverage(
            "financial", Decimal("0.40")
        )
        assert confidence < Decimal("1.00")
        assert tier_name in ("acceptable", "marginal")

    def test_get_confidence_for_coverage_low(self):
        """Low coverage returns minimal confidence."""
        config = AdaptiveCoverageConfig()
        confidence, tier_name = config.get_confidence_for_coverage(
            "financial", Decimal("0.20")
        )
        assert confidence <= Decimal("0.30")
        assert tier_name in ("minimal", "insufficient")

    def test_commercial_stage_multiplier(self):
        """Commercial stage companies require higher effective coverage."""
        config = AdaptiveCoverageConfig(commercial_stage_multiplier=Decimal("1.25"))

        # Same coverage should result in different tiers
        conf_regular, tier_regular = config.get_confidence_for_coverage(
            "financial", Decimal("0.75"), is_commercial=False
        )
        conf_commercial, tier_commercial = config.get_confidence_for_coverage(
            "financial", Decimal("0.75"), is_commercial=True
        )

        # Commercial should have lower effective confidence due to multiplier
        assert conf_commercial <= conf_regular

    def test_to_dict(self):
        """Config serializes to dict correctly."""
        config = AdaptiveCoverageConfig(coverage_mode=CoverageMode.STANDARD)
        d = config.to_dict()
        assert d["coverage_mode"] == "standard"
        assert "mode_thresholds" in d
        assert "absolute_minimums" in d
        assert "alert_thresholds" in d


class TestFallbackStrategy:
    """Tests for FallbackStrategy enum."""

    def test_all_strategies_exist(self):
        """All expected fallback strategies exist."""
        assert FallbackStrategy.DISABLE.value == "disable"
        assert FallbackStrategy.STAGE_ONLY.value == "stage_only"
        assert FallbackStrategy.BINARY_FLAG.value == "binary_flag"
        assert FallbackStrategy.STATIC_WEIGHTS.value == "static_weights"
        assert FallbackStrategy.REDUCE_WEIGHT.value == "reduce_weight"
        assert FallbackStrategy.NEUTRAL_SCORE.value == "neutral_score"


class TestFallbackScoringConfig:
    """Tests for FallbackScoringConfig."""

    def test_default_fallback_strategies(self):
        """Default fallback strategies are set correctly."""
        config = FallbackScoringConfig()
        assert config.pos_fallback_strategy == FallbackStrategy.STAGE_ONLY
        assert config.regime_fallback_strategy == FallbackStrategy.STATIC_WEIGHTS
        assert config.momentum_fallback_strategy == FallbackStrategy.NEUTRAL_SCORE

    def test_get_fallback_for_component_pos(self):
        """PoS fallback returns stage-only strategy."""
        config = FallbackScoringConfig()
        strategy, params = config.get_fallback_for_component("pos")
        assert strategy == FallbackStrategy.STAGE_ONLY
        assert "min_confidence" in params
        assert "weight_multiplier" in params

    def test_get_fallback_for_component_competitive_pressure(self):
        """Competitive pressure fallback returns binary flag strategy."""
        config = FallbackScoringConfig()
        strategy, params = config.get_fallback_for_component("competitive_pressure")
        assert strategy == FallbackStrategy.BINARY_FLAG
        assert "min_ic" in params
        assert "binary_threshold" in params

    def test_get_fallback_for_unknown_component(self):
        """Unknown component returns DISABLE strategy."""
        config = FallbackScoringConfig()
        strategy, params = config.get_fallback_for_component("unknown_component")
        assert strategy == FallbackStrategy.DISABLE
        assert params == {}

    def test_to_dict(self):
        """Config serializes to dict correctly."""
        config = FallbackScoringConfig()
        d = config.to_dict()
        assert "pos" in d
        assert d["pos"]["strategy"] == "stage_only"
        assert "params" in d["pos"]


class TestValidationCriteriaConfig:
    """Tests for ValidationCriteriaConfig."""

    def test_default_thresholds(self):
        """Default thresholds are reasonable."""
        config = ValidationCriteriaConfig()
        assert config.ic_threshold_pass == Decimal("0.05")
        assert config.hit_rate_pass == Decimal("0.55")
        assert config.max_drawdown_pass == Decimal("0.40")
        assert config.turnover_pass == Decimal("0.80")

    def test_qualified_thresholds_lower_than_pass(self):
        """Qualified thresholds are lower than pass thresholds."""
        config = ValidationCriteriaConfig()
        assert config.ic_threshold_qualified < config.ic_threshold_pass
        assert config.hit_rate_qualified < config.hit_rate_pass
        # For drawdown and turnover, qualified is higher (worse) than pass
        assert config.max_drawdown_qualified > config.max_drawdown_pass
        assert config.turnover_qualified > config.turnover_pass

    def test_evaluate_metric_ic_pass(self):
        """IC above threshold returns PASS."""
        config = ValidationCriteriaConfig()
        outcome = config.evaluate_metric("ic", Decimal("0.08"), higher_is_better=True)
        assert outcome == ValidationOutcome.PASS

    def test_evaluate_metric_ic_qualified_pass(self):
        """IC between qualified and pass threshold returns QUALIFIED_PASS."""
        config = ValidationCriteriaConfig()
        outcome = config.evaluate_metric("ic", Decimal("0.04"), higher_is_better=True)
        assert outcome == ValidationOutcome.QUALIFIED_PASS

    def test_evaluate_metric_ic_warn(self):
        """IC between warn and qualified threshold returns WARN."""
        config = ValidationCriteriaConfig()
        outcome = config.evaluate_metric("ic", Decimal("0.025"), higher_is_better=True)
        assert outcome == ValidationOutcome.WARN

    def test_evaluate_metric_ic_fail(self):
        """IC below warn threshold returns FAIL."""
        config = ValidationCriteriaConfig()
        outcome = config.evaluate_metric("ic", Decimal("0.01"), higher_is_better=True)
        assert outcome == ValidationOutcome.FAIL

    def test_evaluate_metric_drawdown_pass(self):
        """Drawdown below threshold returns PASS (lower is better)."""
        config = ValidationCriteriaConfig()
        outcome = config.evaluate_metric("max_drawdown", Decimal("0.30"), higher_is_better=False)
        assert outcome == ValidationOutcome.PASS

    def test_evaluate_metric_drawdown_fail(self):
        """Drawdown above warn threshold returns FAIL."""
        config = ValidationCriteriaConfig()
        outcome = config.evaluate_metric("max_drawdown", Decimal("0.60"), higher_is_better=False)
        assert outcome == ValidationOutcome.FAIL

    def test_get_sample_size_requirement_full(self):
        """Full validation requires 500+ observations."""
        config = ValidationCriteriaConfig()
        assert config.get_sample_size_requirement("full") == 500

    def test_get_sample_size_requirement_qualified(self):
        """Qualified validation requires 200+ observations."""
        config = ValidationCriteriaConfig()
        assert config.get_sample_size_requirement("qualified") == 200

    def test_get_sample_size_requirement_minimal(self):
        """Minimal validation requires 50+ observations."""
        config = ValidationCriteriaConfig()
        assert config.get_sample_size_requirement("minimal") == 50


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_result_creation(self):
        """Result can be created with required fields."""
        result = ValidationResult(
            metric_name="ic",
            value=Decimal("0.08"),
            outcome=ValidationOutcome.PASS,
            threshold_used=Decimal("0.05"),
        )
        assert result.metric_name == "ic"
        assert result.outcome == ValidationOutcome.PASS

    def test_result_with_limitations(self):
        """Result can include limitations."""
        result = ValidationResult(
            metric_name="ic",
            value=Decimal("0.04"),
            outcome=ValidationOutcome.QUALIFIED_PASS,
            threshold_used=Decimal("0.05"),
            limitations=["IC meets qualified but not full threshold"],
        )
        assert len(result.limitations) == 1

    def test_to_dict(self):
        """Result serializes to dict correctly."""
        result = ValidationResult(
            metric_name="hit_rate",
            value=Decimal("0.56"),
            outcome=ValidationOutcome.PASS,
            threshold_used=Decimal("0.55"),
        )
        d = result.to_dict()
        assert d["metric_name"] == "hit_rate"
        assert d["value"] == "0.56"
        assert d["outcome"] == "pass"


class TestValidationSummary:
    """Tests for ValidationSummary."""

    def test_compute_overall_outcome_all_pass(self):
        """All PASS results give overall PASS."""
        summary = ValidationSummary(
            results=[
                ValidationResult("ic", Decimal("0.08"), ValidationOutcome.PASS, Decimal("0.05")),
                ValidationResult("hit_rate", Decimal("0.60"), ValidationOutcome.PASS, Decimal("0.55")),
            ]
        )
        assert summary.compute_overall_outcome() == ValidationOutcome.PASS

    def test_compute_overall_outcome_with_qualified(self):
        """Mix of PASS and QUALIFIED_PASS gives overall QUALIFIED_PASS."""
        summary = ValidationSummary(
            results=[
                ValidationResult("ic", Decimal("0.08"), ValidationOutcome.PASS, Decimal("0.05")),
                ValidationResult("hit_rate", Decimal("0.53"), ValidationOutcome.QUALIFIED_PASS, Decimal("0.55")),
            ]
        )
        assert summary.compute_overall_outcome() == ValidationOutcome.QUALIFIED_PASS

    def test_compute_overall_outcome_with_fail(self):
        """Any FAIL gives overall FAIL."""
        summary = ValidationSummary(
            results=[
                ValidationResult("ic", Decimal("0.08"), ValidationOutcome.PASS, Decimal("0.05")),
                ValidationResult("hit_rate", Decimal("0.45"), ValidationOutcome.FAIL, Decimal("0.55")),
            ]
        )
        assert summary.compute_overall_outcome() == ValidationOutcome.FAIL

    def test_get_limitations(self):
        """Limitations are aggregated from all results."""
        summary = ValidationSummary(
            results=[
                ValidationResult(
                    "ic", Decimal("0.04"), ValidationOutcome.QUALIFIED_PASS, Decimal("0.05"),
                    limitations=["IC meets qualified threshold"]
                ),
                ValidationResult(
                    "hit_rate", Decimal("0.53"), ValidationOutcome.QUALIFIED_PASS, Decimal("0.55"),
                    limitations=["Hit rate meets qualified threshold"]
                ),
            ],
            sample_size=300,
            sample_size_adequate=True,
            fallbacks_active=["pos", "regime"],
        )
        limitations = summary.get_limitations()
        assert len(limitations) >= 3  # 2 from results + 1 from fallbacks

    def test_to_dict(self):
        """Summary serializes to dict correctly."""
        summary = ValidationSummary(
            results=[
                ValidationResult("ic", Decimal("0.08"), ValidationOutcome.PASS, Decimal("0.05")),
            ],
            overall_outcome=ValidationOutcome.PASS,
            sample_size=600,
            sample_size_adequate=True,
            coverage_mode=CoverageMode.ADAPTIVE,
        )
        d = summary.to_dict()
        assert d["overall_outcome"] == "pass"
        assert d["sample_size"] == 600
        assert d["coverage_mode"] == "adaptive"


class TestPresetConfigurations:
    """Tests for preset configuration getters."""

    def test_get_strict_config(self):
        """Strict config has STRICT mode."""
        coverage, fallback, criteria = get_strict_config()
        assert coverage.coverage_mode == CoverageMode.STRICT

    def test_get_standard_config(self):
        """Standard config has STANDARD mode."""
        coverage, fallback, criteria = get_standard_config()
        assert coverage.coverage_mode == CoverageMode.STANDARD

    def test_get_adaptive_config(self):
        """Adaptive config has ADAPTIVE mode."""
        coverage, fallback, criteria = get_adaptive_config()
        assert coverage.coverage_mode == CoverageMode.ADAPTIVE

    def test_get_lenient_config(self):
        """Lenient config has LENIENT mode and relaxed criteria."""
        coverage, fallback, criteria = get_lenient_config()
        assert coverage.coverage_mode == CoverageMode.LENIENT
        assert criteria.ic_threshold_pass == Decimal("0.04")

    def test_get_minimal_config(self):
        """Minimal config has MINIMAL mode and most relaxed criteria."""
        coverage, fallback, criteria = get_minimal_config()
        assert coverage.coverage_mode == CoverageMode.MINIMAL
        assert criteria.ic_threshold_pass == Decimal("0.03")
        assert criteria.min_observations_full == 200

    def test_all_presets_return_three_configs(self):
        """All preset getters return three configuration objects."""
        for getter in [get_strict_config, get_standard_config, get_adaptive_config,
                       get_lenient_config, get_minimal_config]:
            result = getter()
            assert len(result) == 3
            assert isinstance(result[0], AdaptiveCoverageConfig)
            assert isinstance(result[1], FallbackScoringConfig)
            assert isinstance(result[2], ValidationCriteriaConfig)


class TestCoverageModeProgressions:
    """Tests for coverage mode threshold progressions."""

    def test_thresholds_decrease_from_strict_to_minimal(self):
        """Thresholds decrease as mode becomes more lenient."""
        modes = [
            CoverageMode.STRICT,
            CoverageMode.STANDARD,
            CoverageMode.ADAPTIVE,
            CoverageMode.LENIENT,
            CoverageMode.MINIMAL,
        ]

        previous_thresholds = None
        for mode in modes:
            config = AdaptiveCoverageConfig(coverage_mode=mode)
            thresholds = config.get_mode_thresholds()

            if previous_thresholds:
                # Each category threshold should be <= previous
                for category in ["financial", "clinical", "catalyst", "institutional"]:
                    assert thresholds[category] <= previous_thresholds[category], (
                        f"{category} threshold should decrease from {previous_thresholds[category]} "
                        f"in previous mode to {thresholds[category]} in {mode.value}"
                    )

            previous_thresholds = thresholds
