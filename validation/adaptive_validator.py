"""
Adaptive Validation Runner - Configurable Validation with Fallbacks

This module provides a validation runner that uses the configurable validation
framework to evaluate screening results with adaptive thresholds and fallback
mechanisms.

Key Features:
1. Uses AdaptiveCoverageConfig for flexible coverage requirements
2. Applies FallbackScoringConfig when components fail validation
3. Returns ValidationSummary with qualified pass options
4. Tracks limitations and recommendations for transparency

Author: Wake Robin Capital Management
Version: 1.0.0
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from validation.validation_config import (
    AdaptiveCoverageConfig,
    CoverageMode,
    FallbackScoringConfig,
    FallbackStrategy,
    ValidationCriteriaConfig,
    ValidationOutcome,
    ValidationResult,
    ValidationSummary,
    get_adaptive_config,
)


@dataclass
class CoverageReport:
    """Detailed coverage report for a data category."""

    category: str
    coverage_pct: Decimal
    confidence_multiplier: Decimal
    tier_name: str
    meets_threshold: bool
    threshold_used: Decimal
    missing_tickers: List[str] = field(default_factory=list)
    stale_tickers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "coverage_pct": str(self.coverage_pct),
            "confidence_multiplier": str(self.confidence_multiplier),
            "tier_name": self.tier_name,
            "meets_threshold": self.meets_threshold,
            "threshold_used": str(self.threshold_used),
            "missing_tickers_count": len(self.missing_tickers),
            "stale_tickers_count": len(self.stale_tickers),
        }


@dataclass
class FallbackActivation:
    """Record of a fallback activation."""

    component: str
    strategy: FallbackStrategy
    reason: str
    original_value: Optional[Decimal] = None
    fallback_value: Optional[Decimal] = None
    weight_adjustment: Optional[Decimal] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "strategy": self.strategy.value,
            "reason": self.reason,
            "original_value": str(self.original_value) if self.original_value else None,
            "fallback_value": str(self.fallback_value) if self.fallback_value else None,
            "weight_adjustment": str(self.weight_adjustment) if self.weight_adjustment else None,
        }


class AdaptiveValidator:
    """
    Adaptive validation runner with configurable thresholds and fallback mechanisms.

    This validator addresses the challenge of strict validation thresholds when
    actual data coverage is limited (e.g., 35% average coverage with 60% missing
    financial data).

    Usage:
        # Use default adaptive configuration
        validator = AdaptiveValidator()

        # Or use specific configuration
        coverage_config = AdaptiveCoverageConfig(coverage_mode=CoverageMode.LENIENT)
        validator = AdaptiveValidator(coverage_config=coverage_config)

        # Run validation
        summary = validator.validate_pipeline_run(
            coverage_data=coverage_data,
            backtest_results=backtest_results,
        )
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        coverage_config: Optional[AdaptiveCoverageConfig] = None,
        fallback_config: Optional[FallbackScoringConfig] = None,
        criteria_config: Optional[ValidationCriteriaConfig] = None,
    ):
        """
        Initialize adaptive validator.

        Args:
            coverage_config: Coverage threshold configuration
            fallback_config: Fallback strategy configuration
            criteria_config: Validation criteria configuration
        """
        self.coverage_config = coverage_config or AdaptiveCoverageConfig()
        self.fallback_config = fallback_config or FallbackScoringConfig()
        self.criteria_config = criteria_config or ValidationCriteriaConfig()

        # Track active fallbacks
        self.active_fallbacks: List[FallbackActivation] = []

    def validate_coverage(
        self,
        coverage_data: Dict[str, Any],
        universe_size: int,
    ) -> Tuple[List[CoverageReport], bool]:
        """
        Validate data coverage against adaptive thresholds.

        Args:
            coverage_data: Coverage metrics by category
            universe_size: Total universe size

        Returns:
            Tuple of (coverage_reports, overall_pass)
        """
        reports = []
        thresholds = self.coverage_config.get_mode_thresholds()

        for category in ["financial", "clinical", "catalyst", "institutional"]:
            cat_data = coverage_data.get(category, {})

            if isinstance(cat_data, Decimal):
                coverage_pct = cat_data
            elif isinstance(cat_data, dict):
                coverage_pct = Decimal(str(cat_data.get("coverage_percent", "0")))
            else:
                coverage_pct = Decimal("0")

            # Get confidence multiplier from tier
            confidence, tier_name = self.coverage_config.get_confidence_for_coverage(
                category, coverage_pct
            )

            # Check against mode threshold
            threshold = thresholds.get(category, Decimal("0.50"))
            meets_threshold = coverage_pct >= threshold

            # Get missing tickers if available
            missing_tickers = []
            if isinstance(cat_data, dict):
                missing_tickers = cat_data.get("missing_tickers", [])

            reports.append(CoverageReport(
                category=category,
                coverage_pct=coverage_pct,
                confidence_multiplier=confidence,
                tier_name=tier_name,
                meets_threshold=meets_threshold,
                threshold_used=threshold,
                missing_tickers=missing_tickers[:20],  # Limit for reporting
            ))

        # Overall pass if all meet thresholds OR if we can apply fallbacks
        all_meet = all(r.meets_threshold for r in reports)
        return reports, all_meet

    def check_and_activate_fallbacks(
        self,
        component_metrics: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Check component metrics and activate fallbacks as needed.

        Args:
            component_metrics: Metrics for each scoring component

        Returns:
            Updated metrics with fallback adjustments
        """
        self.active_fallbacks = []
        updated_metrics = {}

        for component, metrics in component_metrics.items():
            strategy, params = self.fallback_config.get_fallback_for_component(component)

            # Check if fallback is needed based on component-specific criteria
            needs_fallback, reason = self._check_fallback_needed(
                component, metrics, params
            )

            if needs_fallback:
                # Activate fallback
                activation = self._apply_fallback(component, strategy, metrics, reason)
                self.active_fallbacks.append(activation)

                # Update metrics with fallback values
                updated_metrics[component] = self._get_fallback_metrics(
                    component, strategy, metrics
                )
            else:
                updated_metrics[component] = metrics

        return updated_metrics

    def _check_fallback_needed(
        self,
        component: str,
        metrics: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Check if fallback is needed for a component."""
        if component == "pos":
            confidence = Decimal(str(metrics.get("calibration_confidence", "1.0")))
            min_conf = Decimal(str(params.get("min_confidence", "0.40")))
            if confidence < min_conf:
                return True, f"calibration_confidence {confidence} < {min_conf}"

        elif component == "trial_quality":
            agreement = Decimal(str(metrics.get("expert_agreement", "1.0")))
            min_agreement = Decimal(str(params.get("min_expert_agreement", "0.70")))
            if agreement < min_agreement:
                return True, f"expert_agreement {agreement} < {min_agreement}"

        elif component == "competitive_pressure":
            ic = Decimal(str(metrics.get("ic", "0.0")))
            min_ic = Decimal(str(params.get("min_ic", "0.03")))
            if ic < min_ic:
                return True, f"ic {ic} < {min_ic}"

        elif component == "regime":
            reduction = Decimal(str(metrics.get("drawdown_reduction", "0.0")))
            min_reduction = Decimal(str(params.get("min_drawdown_reduction", "0.05")))
            if reduction < min_reduction:
                return True, f"drawdown_reduction {reduction} < {min_reduction}"

        elif component == "momentum":
            ic = Decimal(str(metrics.get("ic", "0.0")))
            min_ic = Decimal(str(params.get("min_ic", "0.03")))
            if ic < min_ic:
                return True, f"ic {ic} < {min_ic}"

        elif component == "smart_money":
            coverage = Decimal(str(metrics.get("coverage", "0.0")))
            min_coverage = Decimal(str(params.get("min_coverage", "0.30")))
            if coverage < min_coverage:
                return True, f"coverage {coverage} < {min_coverage}"

        return False, ""

    def _apply_fallback(
        self,
        component: str,
        strategy: FallbackStrategy,
        metrics: Dict[str, Any],
        reason: str,
    ) -> FallbackActivation:
        """Apply fallback strategy and return activation record."""
        original_weight = Decimal(str(metrics.get("weight", "0.10")))

        weight_adjustment = None
        if strategy == FallbackStrategy.DISABLE:
            weight_adjustment = Decimal("0.0")
        elif strategy == FallbackStrategy.REDUCE_WEIGHT:
            weight_adjustment = original_weight * Decimal("0.10")
        elif strategy == FallbackStrategy.STAGE_ONLY:
            weight_adjustment = original_weight * Decimal("0.50")

        return FallbackActivation(
            component=component,
            strategy=strategy,
            reason=reason,
            original_value=Decimal(str(metrics.get("score", "0"))),
            weight_adjustment=weight_adjustment,
        )

    def _get_fallback_metrics(
        self,
        component: str,
        strategy: FallbackStrategy,
        original_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Get metrics with fallback applied."""
        metrics = dict(original_metrics)
        metrics["fallback_active"] = True
        metrics["fallback_strategy"] = strategy.value

        if strategy == FallbackStrategy.DISABLE:
            metrics["weight"] = "0.0"
            metrics["score"] = "0.0"
        elif strategy == FallbackStrategy.NEUTRAL_SCORE:
            metrics["score"] = "50.0"
            metrics["confidence"] = "0.30"
        elif strategy == FallbackStrategy.REDUCE_WEIGHT:
            original_weight = Decimal(str(metrics.get("weight", "0.10")))
            metrics["weight"] = str(original_weight * Decimal("0.10"))
        elif strategy == FallbackStrategy.BINARY_FLAG:
            score = Decimal(str(metrics.get("score", "0")))
            metrics["score"] = "100.0" if score >= Decimal("50") else "0.0"
            metrics["is_binary"] = True
        elif strategy == FallbackStrategy.STATIC_WEIGHTS:
            metrics["using_static_weights"] = True

        return metrics

    def validate_backtest_metrics(
        self,
        backtest_results: Dict[str, Any],
    ) -> List[ValidationResult]:
        """
        Validate backtest metrics against criteria.

        Args:
            backtest_results: Results from backtesting

        Returns:
            List of ValidationResult
        """
        results = []

        # IC validation
        ic = Decimal(str(backtest_results.get("ic_mean", "0")))
        ic_outcome = self.criteria_config.evaluate_metric("ic", ic, higher_is_better=True)
        results.append(ValidationResult(
            metric_name="ic",
            value=ic,
            outcome=ic_outcome,
            threshold_used=self.criteria_config.ic_threshold_pass,
            limitations=self._get_limitations("ic", ic_outcome),
            recommendations=self._get_recommendations("ic", ic_outcome, ic),
        ))

        # Hit rate validation
        hit_rate = Decimal(str(backtest_results.get("hit_rate_top_quintile", "0.5")))
        hr_outcome = self.criteria_config.evaluate_metric("hit_rate", hit_rate, higher_is_better=True)
        results.append(ValidationResult(
            metric_name="hit_rate",
            value=hit_rate,
            outcome=hr_outcome,
            threshold_used=self.criteria_config.hit_rate_pass,
            limitations=self._get_limitations("hit_rate", hr_outcome),
            recommendations=self._get_recommendations("hit_rate", hr_outcome, hit_rate),
        ))

        # Max drawdown validation
        max_dd = Decimal(str(backtest_results.get("max_drawdown", "0")))
        dd_outcome = self.criteria_config.evaluate_metric("max_drawdown", max_dd, higher_is_better=False)
        results.append(ValidationResult(
            metric_name="max_drawdown",
            value=max_dd,
            outcome=dd_outcome,
            threshold_used=self.criteria_config.max_drawdown_pass,
            limitations=self._get_limitations("max_drawdown", dd_outcome),
            recommendations=self._get_recommendations("max_drawdown", dd_outcome, max_dd),
        ))

        # Turnover validation
        turnover = Decimal(str(backtest_results.get("turnover_annualized", "0")))
        to_outcome = self.criteria_config.evaluate_metric("turnover", turnover, higher_is_better=False)
        results.append(ValidationResult(
            metric_name="turnover",
            value=turnover,
            outcome=to_outcome,
            threshold_used=self.criteria_config.turnover_pass,
            limitations=self._get_limitations("turnover", to_outcome),
            recommendations=self._get_recommendations("turnover", to_outcome, turnover),
        ))

        return results

    def _get_limitations(
        self,
        metric_name: str,
        outcome: ValidationOutcome,
    ) -> List[str]:
        """Get limitations based on metric outcome."""
        if outcome == ValidationOutcome.PASS:
            return []
        elif outcome == ValidationOutcome.QUALIFIED_PASS:
            return [f"{metric_name} meets qualified but not full threshold"]
        elif outcome == ValidationOutcome.WARN:
            return [f"{metric_name} close to threshold - monitor closely"]
        else:
            return [f"{metric_name} below minimum threshold"]

    def _get_recommendations(
        self,
        metric_name: str,
        outcome: ValidationOutcome,
        value: Decimal,
    ) -> List[str]:
        """Get recommendations based on metric outcome."""
        recommendations = []

        if outcome in (ValidationOutcome.WARN, ValidationOutcome.FAIL):
            if metric_name == "ic":
                recommendations.append("Consider expanding validation sample size")
                recommendations.append("Review component weights for IC optimization")
            elif metric_name == "hit_rate":
                recommendations.append("Analyze hit rate by stage bucket")
                recommendations.append("Check for stage-specific issues")
            elif metric_name == "max_drawdown":
                recommendations.append("Consider enabling defensive overlays")
                recommendations.append("Review position sizing parameters")
            elif metric_name == "turnover":
                recommendations.append("Increase rank stability threshold")
                recommendations.append("Review momentum signal decay parameters")

        return recommendations

    def validate_sample_size(
        self,
        sample_size: int,
    ) -> Tuple[bool, str]:
        """
        Check if sample size is adequate.

        Args:
            sample_size: Number of observations

        Returns:
            Tuple of (is_adequate, validation_level)
        """
        if sample_size >= self.criteria_config.min_observations_full:
            return True, "full"
        elif sample_size >= self.criteria_config.min_observations_qualified:
            return True, "qualified"
        elif sample_size >= self.criteria_config.min_observations_minimal:
            return True, "minimal"
        else:
            return False, "insufficient"

    def validate_pipeline_run(
        self,
        coverage_data: Dict[str, Any],
        backtest_results: Dict[str, Any],
        component_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
        universe_size: int = 100,
    ) -> ValidationSummary:
        """
        Run full validation on pipeline results.

        Args:
            coverage_data: Coverage metrics by category
            backtest_results: Backtest results
            component_metrics: Optional metrics for each scoring component
            universe_size: Total universe size

        Returns:
            ValidationSummary with all results
        """
        # Validate coverage
        coverage_reports, coverage_ok = self.validate_coverage(coverage_data, universe_size)

        # Check and activate fallbacks if component metrics provided
        if component_metrics:
            updated_metrics = self.check_and_activate_fallbacks(component_metrics)

        # Validate backtest metrics
        metric_results = self.validate_backtest_metrics(backtest_results)

        # Check sample size
        sample_size = backtest_results.get("n_observations", 0)
        sample_ok, sample_level = self.validate_sample_size(sample_size)

        # Build summary
        summary = ValidationSummary(
            results=metric_results,
            sample_size=sample_size,
            sample_size_adequate=sample_ok,
            fallbacks_active=[f.component for f in self.active_fallbacks],
            coverage_mode=self.coverage_config.coverage_mode,
        )

        # Compute overall outcome
        summary.overall_outcome = summary.compute_overall_outcome()

        # Downgrade if sample size is only qualified or minimal
        if sample_level == "qualified" and summary.overall_outcome == ValidationOutcome.PASS:
            summary.overall_outcome = ValidationOutcome.QUALIFIED_PASS
        elif sample_level == "minimal":
            if summary.overall_outcome in (ValidationOutcome.PASS, ValidationOutcome.QUALIFIED_PASS):
                summary.overall_outcome = ValidationOutcome.WARN

        # Add coverage as additional check if not ok
        if not coverage_ok:
            summary.results.append(ValidationResult(
                metric_name="coverage",
                value=Decimal("0"),
                outcome=ValidationOutcome.WARN,
                threshold_used=Decimal("0"),
                limitations=["Coverage below thresholds for some categories"],
                recommendations=["Improve data collection for missing categories"],
            ))
            if summary.overall_outcome == ValidationOutcome.PASS:
                summary.overall_outcome = ValidationOutcome.QUALIFIED_PASS

        return summary

    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration."""
        return {
            "version": self.VERSION,
            "coverage_config": self.coverage_config.to_dict(),
            "fallback_config": self.fallback_config.to_dict(),
            "criteria_config": self.criteria_config.to_dict(),
        }

    def get_active_fallbacks_report(self) -> List[Dict[str, Any]]:
        """Get report of all active fallbacks."""
        return [f.to_dict() for f in self.active_fallbacks]


def run_adaptive_validation(
    coverage_data: Dict[str, Any],
    backtest_results: Dict[str, Any],
    component_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
    coverage_mode: CoverageMode = CoverageMode.ADAPTIVE,
    universe_size: int = 100,
) -> ValidationSummary:
    """
    Convenience function to run adaptive validation.

    Args:
        coverage_data: Coverage metrics by category
        backtest_results: Backtest results
        component_metrics: Optional metrics for each scoring component
        coverage_mode: Coverage mode to use
        universe_size: Total universe size

    Returns:
        ValidationSummary
    """
    coverage_config = AdaptiveCoverageConfig(coverage_mode=coverage_mode)
    validator = AdaptiveValidator(coverage_config=coverage_config)

    return validator.validate_pipeline_run(
        coverage_data=coverage_data,
        backtest_results=backtest_results,
        component_metrics=component_metrics,
        universe_size=universe_size,
    )


# Export all
__all__ = [
    "CoverageReport",
    "FallbackActivation",
    "AdaptiveValidator",
    "run_adaptive_validation",
]
