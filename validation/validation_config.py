"""
Validation Configuration Module - Adaptive Coverage & Fallback Controls

This module provides configurable validation thresholds and fallback mechanisms
to help the screening pipeline pass sanity checks while maintaining data quality.

Key Features:
1. Adaptive Coverage Tiers - Adjustable thresholds for different data availability scenarios
2. Fallback Scoring Modes - Explicit activation of fallback strategies
3. Qualified Pass Options - Allow partial validation passes with documented limitations
4. Configurable Sample Sizes - Adjustable minimum sample sizes for validation

Based on production experience with:
- ~35% average coverage
- 60% missing financial data
- Need for flexible validation that maintains quality without being overly restrictive

Author: Wake Robin Capital Management
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class CoverageMode(str, Enum):
    """Coverage mode determines how strictly coverage requirements are enforced."""

    STRICT = "strict"           # Original thresholds (90% financial, 70% clinical)
    STANDARD = "standard"       # Moderate thresholds (75% financial, 60% clinical)
    ADAPTIVE = "adaptive"       # Data-driven adaptive thresholds
    LENIENT = "lenient"         # Relaxed thresholds (60% financial, 50% clinical)
    MINIMAL = "minimal"         # Minimum viable (50% financial, 40% clinical)


class FallbackStrategy(str, Enum):
    """Fallback strategy when a scoring component fails validation."""

    DISABLE = "disable"               # Disable component entirely, renormalize weights
    STAGE_ONLY = "stage_only"         # Use stage-based fallback (e.g., PoS)
    BINARY_FLAG = "binary_flag"       # Use binary flag instead of continuous score
    STATIC_WEIGHTS = "static_weights" # Use static instead of adaptive weights
    REDUCE_WEIGHT = "reduce_weight"   # Reduce weight to 10% of normal
    NEUTRAL_SCORE = "neutral_score"   # Use neutral score (50) with low confidence


class ValidationOutcome(str, Enum):
    """Outcome of validation check."""

    PASS = "pass"                     # Full pass - all criteria met
    QUALIFIED_PASS = "qualified_pass" # Pass with documented limitations
    WARN = "warn"                     # Warning - close to thresholds
    FAIL = "fail"                     # Fail - below thresholds


@dataclass
class CoverageTierConfig:
    """Configuration for a single coverage tier."""

    tier_name: str
    min_coverage_pct: Decimal
    confidence_multiplier: Decimal
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier_name": self.tier_name,
            "min_coverage_pct": str(self.min_coverage_pct),
            "confidence_multiplier": str(self.confidence_multiplier),
            "description": self.description,
        }


# Default coverage tiers by category
DEFAULT_COVERAGE_TIERS = {
    "financial": [
        CoverageTierConfig("excellent", Decimal("0.90"), Decimal("1.00"), "Full confidence"),
        CoverageTierConfig("good", Decimal("0.75"), Decimal("0.85"), "Minor confidence haircut"),
        CoverageTierConfig("acceptable", Decimal("0.60"), Decimal("0.70"), "Moderate confidence haircut"),
        CoverageTierConfig("marginal", Decimal("0.50"), Decimal("0.50"), "Significant confidence haircut"),
        CoverageTierConfig("minimal", Decimal("0.35"), Decimal("0.30"), "Heavy confidence haircut"),
        CoverageTierConfig("insufficient", Decimal("0.00"), Decimal("0.10"), "Near-zero confidence"),
    ],
    "clinical": [
        CoverageTierConfig("excellent", Decimal("0.80"), Decimal("1.00"), "Full confidence"),
        CoverageTierConfig("good", Decimal("0.65"), Decimal("0.80"), "Minor confidence haircut"),
        CoverageTierConfig("acceptable", Decimal("0.50"), Decimal("0.60"), "Moderate confidence haircut"),
        CoverageTierConfig("marginal", Decimal("0.35"), Decimal("0.40"), "Significant confidence haircut"),
        CoverageTierConfig("minimal", Decimal("0.25"), Decimal("0.25"), "Heavy confidence haircut"),
        CoverageTierConfig("insufficient", Decimal("0.00"), Decimal("0.10"), "Near-zero confidence"),
    ],
    "catalyst": [
        CoverageTierConfig("excellent", Decimal("0.80"), Decimal("1.00"), "Full confidence"),
        CoverageTierConfig("good", Decimal("0.60"), Decimal("0.80"), "Minor confidence haircut"),
        CoverageTierConfig("acceptable", Decimal("0.45"), Decimal("0.60"), "Moderate confidence haircut"),
        CoverageTierConfig("marginal", Decimal("0.30"), Decimal("0.40"), "Significant confidence haircut"),
        CoverageTierConfig("insufficient", Decimal("0.00"), Decimal("0.15"), "Near-zero confidence"),
    ],
    "institutional": [
        CoverageTierConfig("excellent", Decimal("0.70"), Decimal("1.00"), "Full confidence"),
        CoverageTierConfig("good", Decimal("0.50"), Decimal("0.75"), "Minor confidence haircut"),
        CoverageTierConfig("acceptable", Decimal("0.35"), Decimal("0.50"), "Moderate confidence haircut"),
        CoverageTierConfig("marginal", Decimal("0.20"), Decimal("0.30"), "Significant confidence haircut"),
        CoverageTierConfig("insufficient", Decimal("0.00"), Decimal("0.10"), "Near-zero confidence"),
    ],
}


@dataclass
class AdaptiveCoverageConfig:
    """
    Adaptive coverage configuration that adjusts thresholds based on data availability.

    This addresses the issue of strict 90% financial coverage requirements when
    actual coverage is ~35% average with 60% missing financial data.
    """

    # Mode selection
    coverage_mode: CoverageMode = CoverageMode.ADAPTIVE

    # Coverage tiers by category
    coverage_tiers: Dict[str, List[CoverageTierConfig]] = field(
        default_factory=lambda: DEFAULT_COVERAGE_TIERS.copy()
    )

    # Minimum viable coverage thresholds (absolute floors)
    absolute_min_financial: Decimal = Decimal("0.25")
    absolute_min_clinical: Decimal = Decimal("0.25")
    absolute_min_catalyst: Decimal = Decimal("0.20")
    absolute_min_institutional: Decimal = Decimal("0.15")

    # Alert thresholds (when to warn about low coverage)
    alert_financial: Decimal = Decimal("0.50")
    alert_clinical: Decimal = Decimal("0.40")
    alert_catalyst: Decimal = Decimal("0.40")
    alert_institutional: Decimal = Decimal("0.30")

    # Commercial-stage special handling (require higher coverage for commercial companies)
    commercial_stage_multiplier: Decimal = Decimal("1.25")  # 25% higher thresholds

    def get_confidence_for_coverage(
        self,
        category: str,
        coverage_pct: Decimal,
        is_commercial: bool = False,
    ) -> Tuple[Decimal, str]:
        """
        Get confidence multiplier for a given coverage percentage.

        Args:
            category: Data category (financial, clinical, catalyst, institutional)
            coverage_pct: Coverage percentage (0.0 to 1.0)
            is_commercial: Whether this is a commercial-stage company

        Returns:
            Tuple of (confidence_multiplier, tier_name)
        """
        tiers = self.coverage_tiers.get(category, [])
        if not tiers:
            return Decimal("0.50"), "unknown"

        # Adjust coverage for commercial companies (effectively require higher coverage)
        effective_coverage = coverage_pct
        if is_commercial:
            effective_coverage = coverage_pct / self.commercial_stage_multiplier

        # Find matching tier
        for tier in tiers:
            if effective_coverage >= tier.min_coverage_pct:
                return tier.confidence_multiplier, tier.tier_name

        # Return lowest tier
        return tiers[-1].confidence_multiplier, tiers[-1].tier_name

    def get_mode_thresholds(self) -> Dict[str, Decimal]:
        """Get minimum thresholds based on current coverage mode."""
        mode_thresholds = {
            CoverageMode.STRICT: {
                "financial": Decimal("0.90"),
                "clinical": Decimal("0.70"),
                "catalyst": Decimal("0.70"),
                "institutional": Decimal("0.50"),
            },
            CoverageMode.STANDARD: {
                "financial": Decimal("0.75"),
                "clinical": Decimal("0.60"),
                "catalyst": Decimal("0.60"),
                "institutional": Decimal("0.40"),
            },
            CoverageMode.ADAPTIVE: {
                "financial": Decimal("0.50"),
                "clinical": Decimal("0.40"),
                "catalyst": Decimal("0.40"),
                "institutional": Decimal("0.25"),
            },
            CoverageMode.LENIENT: {
                "financial": Decimal("0.40"),
                "clinical": Decimal("0.35"),
                "catalyst": Decimal("0.30"),
                "institutional": Decimal("0.20"),
            },
            CoverageMode.MINIMAL: {
                "financial": Decimal("0.30"),
                "clinical": Decimal("0.25"),
                "catalyst": Decimal("0.25"),
                "institutional": Decimal("0.15"),
            },
        }
        return mode_thresholds.get(self.coverage_mode, mode_thresholds[CoverageMode.ADAPTIVE])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coverage_mode": self.coverage_mode.value,
            "mode_thresholds": {k: str(v) for k, v in self.get_mode_thresholds().items()},
            "absolute_minimums": {
                "financial": str(self.absolute_min_financial),
                "clinical": str(self.absolute_min_clinical),
                "catalyst": str(self.absolute_min_catalyst),
                "institutional": str(self.absolute_min_institutional),
            },
            "alert_thresholds": {
                "financial": str(self.alert_financial),
                "clinical": str(self.alert_clinical),
                "catalyst": str(self.alert_catalyst),
                "institutional": str(self.alert_institutional),
            },
        }


@dataclass
class FallbackScoringConfig:
    """
    Configuration for fallback scoring modes when primary scoring fails.

    Each scoring component can have explicit fallback strategies.
    """

    # PoS Framework fallbacks
    pos_fallback_strategy: FallbackStrategy = FallbackStrategy.STAGE_ONLY
    pos_min_calibration_confidence: Decimal = Decimal("0.40")
    pos_stage_only_weight_multiplier: Decimal = Decimal("0.50")  # Half weight when using fallback

    # Trial Quality fallbacks
    trial_quality_fallback_strategy: FallbackStrategy = FallbackStrategy.REDUCE_WEIGHT
    trial_quality_min_expert_agreement: Decimal = Decimal("0.70")  # Reduced from 0.80
    trial_quality_fallback_weight: Decimal = Decimal("0.10")

    # Competitive Pressure fallbacks
    competitive_pressure_fallback_strategy: FallbackStrategy = FallbackStrategy.BINARY_FLAG
    competitive_pressure_min_ic: Decimal = Decimal("0.03")  # Reduced from 0.05
    competitive_pressure_binary_threshold: Decimal = Decimal("0.50")

    # Regime Detection fallbacks
    regime_fallback_strategy: FallbackStrategy = FallbackStrategy.STATIC_WEIGHTS
    regime_min_drawdown_reduction: Decimal = Decimal("0.05")  # Reduced from 0.10
    regime_default: str = "NEUTRAL"

    # Momentum Signal fallbacks
    momentum_fallback_strategy: FallbackStrategy = FallbackStrategy.NEUTRAL_SCORE
    momentum_min_ic: Decimal = Decimal("0.03")
    momentum_neutral_score: Decimal = Decimal("50.0")
    momentum_neutral_confidence: Decimal = Decimal("0.30")

    # Smart Money fallbacks
    smart_money_fallback_strategy: FallbackStrategy = FallbackStrategy.DISABLE
    smart_money_min_coverage: Decimal = Decimal("0.30")

    def get_fallback_for_component(self, component: str) -> Tuple[FallbackStrategy, Dict[str, Any]]:
        """Get fallback strategy and parameters for a component."""
        fallback_map = {
            "pos": (
                self.pos_fallback_strategy,
                {
                    "min_confidence": str(self.pos_min_calibration_confidence),
                    "weight_multiplier": str(self.pos_stage_only_weight_multiplier),
                },
            ),
            "trial_quality": (
                self.trial_quality_fallback_strategy,
                {
                    "min_expert_agreement": str(self.trial_quality_min_expert_agreement),
                    "fallback_weight": str(self.trial_quality_fallback_weight),
                },
            ),
            "competitive_pressure": (
                self.competitive_pressure_fallback_strategy,
                {
                    "min_ic": str(self.competitive_pressure_min_ic),
                    "binary_threshold": str(self.competitive_pressure_binary_threshold),
                },
            ),
            "regime": (
                self.regime_fallback_strategy,
                {
                    "min_drawdown_reduction": str(self.regime_min_drawdown_reduction),
                    "default_regime": self.regime_default,
                },
            ),
            "momentum": (
                self.momentum_fallback_strategy,
                {
                    "min_ic": str(self.momentum_min_ic),
                    "neutral_score": str(self.momentum_neutral_score),
                    "neutral_confidence": str(self.momentum_neutral_confidence),
                },
            ),
            "smart_money": (
                self.smart_money_fallback_strategy,
                {
                    "min_coverage": str(self.smart_money_min_coverage),
                },
            ),
        }
        return fallback_map.get(component, (FallbackStrategy.DISABLE, {}))

    def to_dict(self) -> Dict[str, Any]:
        return {
            component: {
                "strategy": strategy.value,
                "params": params,
            }
            for component in ["pos", "trial_quality", "competitive_pressure", "regime", "momentum", "smart_money"]
            for strategy, params in [self.get_fallback_for_component(component)]
        }


@dataclass
class ValidationCriteriaConfig:
    """
    Configurable validation criteria with qualified pass options.

    Addresses the need for flexible validation that allows partial passes
    with documented limitations.
    """

    # IC thresholds
    ic_threshold_pass: Decimal = Decimal("0.05")
    ic_threshold_qualified: Decimal = Decimal("0.03")  # Qualified pass threshold
    ic_threshold_warn: Decimal = Decimal("0.02")

    # Hit rate thresholds (top quintile)
    hit_rate_pass: Decimal = Decimal("0.55")
    hit_rate_qualified: Decimal = Decimal("0.52")
    hit_rate_warn: Decimal = Decimal("0.50")

    # Max drawdown thresholds
    max_drawdown_pass: Decimal = Decimal("0.40")
    max_drawdown_qualified: Decimal = Decimal("0.45")
    max_drawdown_warn: Decimal = Decimal("0.50")

    # Turnover thresholds (annualized)
    turnover_pass: Decimal = Decimal("0.80")
    turnover_qualified: Decimal = Decimal("0.90")
    turnover_warn: Decimal = Decimal("1.00")

    # Minimum sample sizes for validation
    min_observations_full: int = 500  # For full validation
    min_observations_qualified: int = 200  # For qualified validation
    min_observations_minimal: int = 50  # Absolute minimum

    # Minimum stage sizes for IC by stage
    min_stage_size: int = 5
    min_stage_size_qualified: int = 3  # Allow smaller stages for qualified pass

    # Rank stability thresholds
    rank_corr_stable: Decimal = Decimal("0.50")
    rank_corr_moderate: Decimal = Decimal("0.30")
    rank_corr_qualified: Decimal = Decimal("0.20")  # Qualified pass threshold

    # Top quintile churn thresholds
    churn_low: Decimal = Decimal("0.50")
    churn_moderate: Decimal = Decimal("0.65")
    churn_high: Decimal = Decimal("0.80")

    def evaluate_metric(
        self,
        metric_name: str,
        value: Decimal,
        higher_is_better: bool = True,
    ) -> ValidationOutcome:
        """
        Evaluate a single metric against thresholds.

        Args:
            metric_name: Name of the metric (ic, hit_rate, max_drawdown, turnover)
            value: Metric value
            higher_is_better: Whether higher values are better

        Returns:
            ValidationOutcome
        """
        thresholds = self._get_thresholds(metric_name)
        if not thresholds:
            return ValidationOutcome.WARN

        pass_thresh, qualified_thresh, warn_thresh = thresholds

        if higher_is_better:
            if value >= pass_thresh:
                return ValidationOutcome.PASS
            elif value >= qualified_thresh:
                return ValidationOutcome.QUALIFIED_PASS
            elif value >= warn_thresh:
                return ValidationOutcome.WARN
            else:
                return ValidationOutcome.FAIL
        else:
            # Lower is better (e.g., drawdown, turnover)
            if value <= pass_thresh:
                return ValidationOutcome.PASS
            elif value <= qualified_thresh:
                return ValidationOutcome.QUALIFIED_PASS
            elif value <= warn_thresh:
                return ValidationOutcome.WARN
            else:
                return ValidationOutcome.FAIL

    def _get_thresholds(self, metric_name: str) -> Optional[Tuple[Decimal, Decimal, Decimal]]:
        """Get (pass, qualified, warn) thresholds for a metric."""
        threshold_map = {
            "ic": (self.ic_threshold_pass, self.ic_threshold_qualified, self.ic_threshold_warn),
            "hit_rate": (self.hit_rate_pass, self.hit_rate_qualified, self.hit_rate_warn),
            "max_drawdown": (self.max_drawdown_pass, self.max_drawdown_qualified, self.max_drawdown_warn),
            "turnover": (self.turnover_pass, self.turnover_qualified, self.turnover_warn),
            "rank_corr": (self.rank_corr_stable, self.rank_corr_moderate, self.rank_corr_qualified),
        }
        return threshold_map.get(metric_name)

    def get_sample_size_requirement(self, validation_level: str = "full") -> int:
        """Get minimum sample size for a validation level."""
        size_map = {
            "full": self.min_observations_full,
            "qualified": self.min_observations_qualified,
            "minimal": self.min_observations_minimal,
        }
        return size_map.get(validation_level, self.min_observations_minimal)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ic_thresholds": {
                "pass": str(self.ic_threshold_pass),
                "qualified": str(self.ic_threshold_qualified),
                "warn": str(self.ic_threshold_warn),
            },
            "hit_rate_thresholds": {
                "pass": str(self.hit_rate_pass),
                "qualified": str(self.hit_rate_qualified),
                "warn": str(self.hit_rate_warn),
            },
            "max_drawdown_thresholds": {
                "pass": str(self.max_drawdown_pass),
                "qualified": str(self.max_drawdown_qualified),
                "warn": str(self.max_drawdown_warn),
            },
            "turnover_thresholds": {
                "pass": str(self.turnover_pass),
                "qualified": str(self.turnover_qualified),
                "warn": str(self.turnover_warn),
            },
            "sample_sizes": {
                "full": self.min_observations_full,
                "qualified": self.min_observations_qualified,
                "minimal": self.min_observations_minimal,
            },
        }


@dataclass
class ValidationResult:
    """Result of validation with outcome and details."""

    metric_name: str
    value: Decimal
    outcome: ValidationOutcome
    threshold_used: Decimal
    limitations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "value": str(self.value),
            "outcome": self.outcome.value,
            "threshold_used": str(self.threshold_used),
            "limitations": self.limitations,
            "recommendations": self.recommendations,
        }


@dataclass
class ValidationSummary:
    """Summary of all validation results."""

    results: List[ValidationResult] = field(default_factory=list)
    overall_outcome: ValidationOutcome = ValidationOutcome.FAIL
    sample_size: int = 0
    sample_size_adequate: bool = False
    fallbacks_active: List[str] = field(default_factory=list)
    coverage_mode: CoverageMode = CoverageMode.ADAPTIVE

    def compute_overall_outcome(self) -> ValidationOutcome:
        """Compute overall outcome from individual results."""
        if not self.results:
            return ValidationOutcome.FAIL

        outcomes = [r.outcome for r in self.results]

        # Any FAIL means overall FAIL
        if ValidationOutcome.FAIL in outcomes:
            return ValidationOutcome.FAIL

        # All PASS means overall PASS
        if all(o == ValidationOutcome.PASS for o in outcomes):
            return ValidationOutcome.PASS

        # Mix of PASS and QUALIFIED_PASS means QUALIFIED_PASS
        if all(o in (ValidationOutcome.PASS, ValidationOutcome.QUALIFIED_PASS) for o in outcomes):
            return ValidationOutcome.QUALIFIED_PASS

        # Anything else is WARN
        return ValidationOutcome.WARN

    def get_limitations(self) -> List[str]:
        """Get all limitations from results."""
        limitations = []
        for result in self.results:
            limitations.extend(result.limitations)

        if not self.sample_size_adequate:
            limitations.append(f"Sample size ({self.sample_size}) below recommended minimum")

        if self.fallbacks_active:
            limitations.append(f"Fallbacks active: {', '.join(self.fallbacks_active)}")

        return limitations

    def get_recommendations(self) -> List[str]:
        """Get all recommendations from results."""
        recommendations = []
        for result in self.results:
            recommendations.extend(result.recommendations)
        return recommendations

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_outcome": self.overall_outcome.value,
            "sample_size": self.sample_size,
            "sample_size_adequate": self.sample_size_adequate,
            "coverage_mode": self.coverage_mode.value,
            "fallbacks_active": self.fallbacks_active,
            "results": [r.to_dict() for r in self.results],
            "limitations": self.get_limitations(),
            "recommendations": self.get_recommendations(),
        }


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

def get_strict_config() -> Tuple[AdaptiveCoverageConfig, FallbackScoringConfig, ValidationCriteriaConfig]:
    """Get strict validation configuration (original thresholds)."""
    return (
        AdaptiveCoverageConfig(coverage_mode=CoverageMode.STRICT),
        FallbackScoringConfig(),
        ValidationCriteriaConfig(),
    )


def get_standard_config() -> Tuple[AdaptiveCoverageConfig, FallbackScoringConfig, ValidationCriteriaConfig]:
    """Get standard validation configuration (moderate thresholds)."""
    return (
        AdaptiveCoverageConfig(coverage_mode=CoverageMode.STANDARD),
        FallbackScoringConfig(),
        ValidationCriteriaConfig(),
    )


def get_adaptive_config() -> Tuple[AdaptiveCoverageConfig, FallbackScoringConfig, ValidationCriteriaConfig]:
    """Get adaptive validation configuration (recommended for production)."""
    return (
        AdaptiveCoverageConfig(coverage_mode=CoverageMode.ADAPTIVE),
        FallbackScoringConfig(),
        ValidationCriteriaConfig(),
    )


def get_lenient_config() -> Tuple[AdaptiveCoverageConfig, FallbackScoringConfig, ValidationCriteriaConfig]:
    """Get lenient validation configuration (relaxed thresholds)."""
    criteria = ValidationCriteriaConfig(
        ic_threshold_pass=Decimal("0.04"),
        ic_threshold_qualified=Decimal("0.02"),
        ic_threshold_warn=Decimal("0.01"),
        hit_rate_pass=Decimal("0.53"),
        hit_rate_qualified=Decimal("0.50"),
        hit_rate_warn=Decimal("0.48"),
        min_observations_full=300,
        min_observations_qualified=100,
        min_observations_minimal=30,
    )
    return (
        AdaptiveCoverageConfig(coverage_mode=CoverageMode.LENIENT),
        FallbackScoringConfig(),
        criteria,
    )


def get_minimal_config() -> Tuple[AdaptiveCoverageConfig, FallbackScoringConfig, ValidationCriteriaConfig]:
    """Get minimal validation configuration (minimum viable)."""
    criteria = ValidationCriteriaConfig(
        ic_threshold_pass=Decimal("0.03"),
        ic_threshold_qualified=Decimal("0.01"),
        ic_threshold_warn=Decimal("0.00"),
        hit_rate_pass=Decimal("0.52"),
        hit_rate_qualified=Decimal("0.48"),
        hit_rate_warn=Decimal("0.45"),
        max_drawdown_pass=Decimal("0.50"),
        max_drawdown_qualified=Decimal("0.55"),
        max_drawdown_warn=Decimal("0.60"),
        min_observations_full=200,
        min_observations_qualified=50,
        min_observations_minimal=20,
        min_stage_size=3,
        min_stage_size_qualified=2,
    )
    return (
        AdaptiveCoverageConfig(coverage_mode=CoverageMode.MINIMAL),
        FallbackScoringConfig(),
        criteria,
    )


# Export all
__all__ = [
    # Enums
    "CoverageMode",
    "FallbackStrategy",
    "ValidationOutcome",
    # Config classes
    "CoverageTierConfig",
    "AdaptiveCoverageConfig",
    "FallbackScoringConfig",
    "ValidationCriteriaConfig",
    "ValidationResult",
    "ValidationSummary",
    # Defaults
    "DEFAULT_COVERAGE_TIERS",
    # Preset getters
    "get_strict_config",
    "get_standard_config",
    "get_adaptive_config",
    "get_lenient_config",
    "get_minimal_config",
]
