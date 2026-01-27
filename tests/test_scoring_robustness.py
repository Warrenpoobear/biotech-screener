#!/usr/bin/env python3
"""
test_scoring_robustness.py - Tests for Scoring Robustness Enhancements

Tests the 8 robustness enhancements for the composite scoring system:
1. Winsorization at component level
2. Confidence-weighted shrinkage
3. Rank stability regularization
4. Multi-timeframe signal blending
5. Asymmetric interaction bounds
6. Regime-conditional weight floors
7. Defensive override triggers
8. Score distribution health checks

Usage:
    pytest tests/test_scoring_robustness.py -v
"""

from decimal import Decimal
import pytest

from src.modules.scoring_robustness import (
    # Core functions
    winsorize_component_score,
    winsorize_cohort,
    apply_confidence_shrinkage,
    compute_rank_stability_penalty,
    blend_timeframe_signals,
    apply_asymmetric_bounds,
    apply_weight_floors,
    evaluate_defensive_triggers,
    check_distribution_health,
    apply_robustness_enhancements,
    # Types
    WinsorizedScore,
    ShrinkageResult,
    RankStabilityAdjustment,
    AsymmetricBounds,
    WeightFloorResult,
    DefensiveOverrideResult,
    DistributionHealthCheck,
    RobustnessEnhancements,
    DefensivePosture,
    DistributionHealth,
    # Constants
    COMPONENT_WINSOR_LOW,
    COMPONENT_WINSOR_HIGH,
    INTERACTION_POSITIVE_CAP,
    INTERACTION_NEGATIVE_FLOOR,
)


# =============================================================================
# 1. WINSORIZATION TESTS
# =============================================================================

class TestWinsorization:
    """Tests for component-level winsorization."""

    def test_winsorize_normal_score_unchanged(self):
        """Normal scores within bounds are not clipped."""
        cohort = [Decimal(str(x)) for x in [30, 40, 50, 60, 70]]
        result = winsorize_component_score(Decimal("50"), cohort)

        assert result.was_clipped is False
        assert result.winsorized == result.original

    def test_winsorize_extreme_low_clipped(self):
        """Extremely low scores are clipped."""
        cohort = [Decimal(str(x)) for x in [20, 30, 40, 50, 60, 70, 80, 90, 95, 98]]
        result = winsorize_component_score(Decimal("2"), cohort)

        assert result.was_clipped is True
        assert result.clip_direction == "low"
        assert result.winsorized > result.original

    def test_winsorize_extreme_high_clipped(self):
        """Extremely high scores are clipped."""
        cohort = [Decimal(str(x)) for x in [5, 10, 20, 30, 40, 50, 60, 70, 80]]
        result = winsorize_component_score(Decimal("99"), cohort)

        assert result.was_clipped is True
        assert result.clip_direction == "high"
        assert result.winsorized < result.original

    def test_winsorize_cohort(self):
        """Winsorize entire cohort."""
        scores = {
            "A": Decimal("2"),   # Too low
            "B": Decimal("50"),  # Normal
            "C": Decimal("98"),  # Too high
            "D": Decimal("40"),  # Normal
            "E": Decimal("60"),  # Normal
        }
        winsorized, clipped_count = winsorize_cohort(scores)

        assert clipped_count >= 2  # At least A and C should be clipped
        assert winsorized["A"] > scores["A"]
        assert winsorized["C"] < scores["C"]
        assert winsorized["B"] == scores["B"]


# =============================================================================
# 2. CONFIDENCE SHRINKAGE TESTS
# =============================================================================

class TestConfidenceShrinkage:
    """Tests for confidence-weighted shrinkage."""

    def test_high_confidence_no_shrinkage(self):
        """High confidence scores are not shrunk."""
        result = apply_confidence_shrinkage(
            score=Decimal("80"),
            confidence=Decimal("0.8"),
            cohort_mean=Decimal("50"),
        )

        assert result.shrinkage_factor == Decimal("0")
        assert result.shrunk == result.original

    def test_low_confidence_shrinks_toward_mean(self):
        """Low confidence scores are shrunk toward cohort mean."""
        result = apply_confidence_shrinkage(
            score=Decimal("80"),
            confidence=Decimal("0.3"),
            cohort_mean=Decimal("50"),
        )

        assert result.shrinkage_factor > Decimal("0")
        assert result.shrunk < result.original
        assert result.shrunk > result.cohort_mean  # Still above mean, just pulled toward it

    def test_very_low_confidence_max_shrinkage(self):
        """Very low confidence applies maximum shrinkage."""
        result = apply_confidence_shrinkage(
            score=Decimal("90"),
            confidence=Decimal("0.1"),
            cohort_mean=Decimal("50"),
        )

        assert result.shrinkage_factor > Decimal("0.3")
        # Should be significantly closer to mean
        distance_to_mean = abs(result.shrunk - result.cohort_mean)
        original_distance = abs(result.original - result.cohort_mean)
        assert distance_to_mean < original_distance * Decimal("0.7")


# =============================================================================
# 3. RANK STABILITY TESTS
# =============================================================================

class TestRankStability:
    """Tests for rank stability regularization."""

    def test_no_prior_ranks_no_penalty(self):
        """Without prior ranks, no penalty is applied."""
        result = compute_rank_stability_penalty(
            current_score=Decimal("75"),
            current_rank=5,
            prior_ranks=[],
        )

        assert result.penalty_applied == Decimal("0")
        assert result.adjusted_score == result.original_score
        assert "no_prior_ranks" in result.flags

    def test_rank_decline_full_penalty(self):
        """Rank decline applies full penalty."""
        result = compute_rank_stability_penalty(
            current_score=Decimal("70"),
            current_rank=10,  # Moved down from 5
            prior_ranks=[5],
        )

        assert result.penalty_applied > Decimal("0")
        assert result.adjusted_score < result.original_score
        assert result.rank_change == 5

    def test_rank_improvement_partial_penalty(self):
        """Rank improvement applies partial penalty (50%)."""
        result = compute_rank_stability_penalty(
            current_score=Decimal("70"),
            current_rank=5,  # Moved up from 10
            prior_ranks=[10],
        )

        # Penalty exists but is reduced
        assert result.penalty_applied > Decimal("0")
        assert "rank_improved_partial_penalty" in result.flags

    def test_significant_decline_flagged(self):
        """Significant rank decline is flagged."""
        result = compute_rank_stability_penalty(
            current_score=Decimal("60"),
            current_rank=20,  # Big drop from 5
            prior_ranks=[5],
        )

        assert "significant_rank_decline" in result.flags


# =============================================================================
# 4. MULTI-TIMEFRAME BLENDING TESTS
# =============================================================================

class TestTimeframeBlending:
    """Tests for multi-timeframe signal blending."""

    def test_all_signals_present(self):
        """All three signals blend correctly."""
        blended, conf, flags = blend_timeframe_signals(
            short_signal=Decimal("40"),
            medium_signal=Decimal("60"),
            long_signal=Decimal("50"),
        )

        assert blended > Decimal("40")  # Should be weighted average
        assert blended < Decimal("60")
        assert "blended_3_signals" in flags

    def test_missing_signal_handled(self):
        """Missing signals are handled gracefully."""
        blended, conf, flags = blend_timeframe_signals(
            short_signal=None,
            medium_signal=Decimal("60"),
            long_signal=Decimal("50"),
        )

        assert blended > Decimal("0")
        assert "short_signal_missing" in flags
        assert "blended_2_signals" in flags

    def test_all_missing_returns_neutral(self):
        """All missing signals returns neutral score."""
        blended, conf, flags = blend_timeframe_signals(
            short_signal=None,
            medium_signal=None,
            long_signal=None,
        )

        assert blended == Decimal("50")
        assert conf == Decimal("0.3")
        assert "all_signals_missing" in flags


# =============================================================================
# 5. ASYMMETRIC BOUNDS TESTS
# =============================================================================

class TestAsymmetricBounds:
    """Tests for asymmetric interaction bounds."""

    def test_positive_within_bounds(self):
        """Positive value within cap is unchanged."""
        result = apply_asymmetric_bounds(Decimal("2.0"))

        assert result.was_capped is False
        assert result.applied_value == result.original_value

    def test_positive_exceeds_cap(self):
        """Positive value exceeding cap is capped."""
        result = apply_asymmetric_bounds(Decimal("5.0"))

        assert result.was_capped is True
        assert result.cap_type == "positive"
        assert result.applied_value == INTERACTION_POSITIVE_CAP

    def test_negative_within_floor(self):
        """Negative value within floor is unchanged."""
        result = apply_asymmetric_bounds(Decimal("-2.0"))

        assert result.was_capped is False
        assert result.applied_value == result.original_value

    def test_negative_exceeds_floor(self):
        """Negative value exceeding floor is capped."""
        result = apply_asymmetric_bounds(Decimal("-5.0"))

        assert result.was_capped is True
        assert result.cap_type == "negative"
        assert result.applied_value == INTERACTION_NEGATIVE_FLOOR

    def test_asymmetric_caps_different(self):
        """Positive cap is higher than negative floor (asymmetric)."""
        assert INTERACTION_POSITIVE_CAP > abs(INTERACTION_NEGATIVE_FLOOR)


# =============================================================================
# 6. WEIGHT FLOORS TESTS
# =============================================================================

class TestWeightFloors:
    """Tests for regime-conditional weight floors."""

    def test_bull_regime_floors(self):
        """Bull regime applies appropriate floors."""
        weights = {
            "clinical": Decimal("0.10"),  # Below floor
            "financial": Decimal("0.05"),  # Below floor
            "catalyst": Decimal("0.05"),  # Below floor
            "momentum": Decimal("0.80"),
        }
        result = apply_weight_floors(weights, "BULL")

        assert result.adjusted_weights["clinical"] >= Decimal("0.15")
        assert result.adjusted_weights["financial"] >= Decimal("0.10")

    def test_bear_regime_financial_emphasis(self):
        """Bear regime emphasizes financial health."""
        weights = {
            "clinical": Decimal("0.50"),
            "financial": Decimal("0.10"),  # Below bear floor
            "catalyst": Decimal("0.20"),
            "momentum": Decimal("0.20"),
        }
        result = apply_weight_floors(weights, "BEAR")

        assert result.adjusted_weights["financial"] >= Decimal("0.25")
        assert "financial_floor_applied" in result.flags

    def test_weights_sum_to_one(self):
        """Adjusted weights still sum to approximately 1.0."""
        weights = {
            "clinical": Decimal("0.05"),
            "financial": Decimal("0.05"),
            "catalyst": Decimal("0.05"),
            "momentum": Decimal("0.85"),
        }
        result = apply_weight_floors(weights, "RECESSION_RISK")

        total = sum(result.adjusted_weights.values())
        assert abs(total - Decimal("1.0")) < Decimal("0.01")


# =============================================================================
# 7. DEFENSIVE TRIGGERS TESTS
# =============================================================================

class TestDefensiveTriggers:
    """Tests for defensive override triggers."""

    def test_no_triggers_none_posture(self):
        """No triggers hit results in NONE posture."""
        stats = {
            "severity_ratio": Decimal("0.10"),
            "avg_runway_months": Decimal("18"),
            "high_vol_ratio": Decimal("0.20"),
            "positive_momentum_ratio": Decimal("0.50"),
        }
        result = evaluate_defensive_triggers(stats)

        assert result.posture == DefensivePosture.NONE
        assert len(result.triggers_hit) == 0

    def test_single_trigger_light_posture(self):
        """Single trigger results in LIGHT posture."""
        stats = {
            "severity_ratio": Decimal("0.40"),  # Exceeds threshold
            "avg_runway_months": Decimal("18"),
            "high_vol_ratio": Decimal("0.20"),
            "positive_momentum_ratio": Decimal("0.50"),
        }
        result = evaluate_defensive_triggers(stats)

        assert result.posture == DefensivePosture.LIGHT
        assert "high_severity_ratio" in result.triggers_hit

    def test_multiple_triggers_moderate_posture(self):
        """Multiple triggers result in MODERATE posture."""
        stats = {
            "severity_ratio": Decimal("0.40"),
            "avg_runway_months": Decimal("6"),  # Low runway
            "high_vol_ratio": Decimal("0.20"),
            "positive_momentum_ratio": Decimal("0.50"),
        }
        result = evaluate_defensive_triggers(stats)

        assert result.posture == DefensivePosture.MODERATE
        assert len(result.triggers_hit) == 2

    def test_many_triggers_heavy_posture(self):
        """Many triggers result in HEAVY posture."""
        stats = {
            "severity_ratio": Decimal("0.40"),
            "avg_runway_months": Decimal("6"),
            "high_vol_ratio": Decimal("0.50"),
            "positive_momentum_ratio": Decimal("0.10"),
        }
        result = evaluate_defensive_triggers(stats)

        assert result.posture == DefensivePosture.HEAVY
        assert len(result.triggers_hit) >= 3


# =============================================================================
# 8. DISTRIBUTION HEALTH TESTS
# =============================================================================

class TestDistributionHealth:
    """Tests for score distribution health checks."""

    def test_healthy_distribution(self):
        """Normal distribution is marked healthy."""
        scores = [Decimal(str(x)) for x in [
            30, 35, 40, 42, 45, 48, 50, 52, 55, 58,
            60, 62, 65, 68, 70, 72, 75, 78, 80, 85
        ]]
        result = check_distribution_health(scores)

        assert result.health == DistributionHealth.HEALTHY
        assert len(result.issues) == 0

    def test_clustered_distribution(self):
        """Clustered scores are detected."""
        scores = [Decimal(str(x)) for x in [
            49, 49.5, 50, 50.5, 51, 49, 50, 50, 51, 49.5
        ]]
        result = check_distribution_health(scores)

        assert result.std < Decimal("8")
        assert "scores_too_clustered" in result.issues or "insufficient_score_range" in result.issues

    def test_excessive_zeros(self):
        """Excessive zeros are detected."""
        scores = [Decimal("0")] * 5 + [Decimal(str(x)) for x in [30, 40, 50, 60, 70]]
        result = check_distribution_health(scores)

        assert result.zero_ratio >= Decimal("0.20")

    def test_insufficient_data(self):
        """Insufficient data returns degraded status."""
        scores = [Decimal("50"), Decimal("60")]
        result = check_distribution_health(scores)

        assert result.health == DistributionHealth.DEGRADED
        assert "insufficient_data" in result.issues


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestRobustnessIntegration:
    """Integration tests for full robustness pipeline."""

    def test_apply_robustness_enhancements(self):
        """Full enhancement pipeline runs without error."""
        scores = {
            "TICK1": {
                "composite_score": Decimal("75"),
                "confidence_overall": Decimal("0.8"),
                "clinical_normalized": Decimal("70"),
                "financial_normalized": Decimal("80"),
                "catalyst_normalized": Decimal("60"),
                "flags": [],
                "effective_weights": {
                    "clinical": Decimal("0.30"),
                    "financial": Decimal("0.30"),
                    "catalyst": Decimal("0.20"),
                    "momentum": Decimal("0.20"),
                },
                "interaction_terms": {"total_adjustment": "1.5"},
                "severity": type("MockSev", (), {"value": "none"})(),
            },
            "TICK2": {
                "composite_score": Decimal("60"),
                "confidence_overall": Decimal("0.4"),  # Low confidence
                "clinical_normalized": Decimal("55"),
                "financial_normalized": Decimal("65"),
                "catalyst_normalized": Decimal("58"),
                "flags": [],
                "effective_weights": {
                    "clinical": Decimal("0.30"),
                    "financial": Decimal("0.30"),
                    "catalyst": Decimal("0.20"),
                    "momentum": Decimal("0.20"),
                },
                "interaction_terms": {"total_adjustment": "-1.0"},
                "severity": type("MockSev", (), {"value": "sev1"})(),
            },
        }

        universe_stats = {
            "severity_ratio": Decimal("0.15"),
            "avg_runway_months": Decimal("18"),
            "high_vol_ratio": Decimal("0.25"),
            "positive_momentum_ratio": Decimal("0.40"),
        }

        enhanced, summary = apply_robustness_enhancements(
            scores=scores,
            regime="NEUTRAL",
            universe_stats=universe_stats,
        )

        # Verify structure
        assert len(enhanced) == 2
        assert summary is not None
        assert hasattr(summary, "defensive_posture")
        assert hasattr(summary, "distribution_health")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
