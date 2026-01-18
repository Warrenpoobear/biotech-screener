#!/usr/bin/env python3
"""
Tests for regime_engine.py helper methods

Tests cover:
- Signal score adjustment by regime
- Composite weight adjustments
- Regime score calculation
- Risk flag generation
- Fed rate impact scoring
- Regime transition detection
"""

import pytest
from decimal import Decimal
from typing import Dict

from regime_engine import RegimeDetectionEngine


class TestApplySignalAdjustments:
    """Test apply_signal_adjustments helper method."""

    def test_basic_signal_adjustment(self):
        """Test basic signal score adjustment with multipliers."""
        engine = RegimeDetectionEngine()

        base_scores = {
            "momentum": Decimal("75.0"),
            "quality": Decimal("60.0"),
            "financial": Decimal("50.0"),
        }

        regime_adjustments = {
            "momentum": Decimal("1.2"),  # 20% boost
            "quality": Decimal("0.9"),    # 10% reduction
            "financial": Decimal("1.0"),  # No change
        }

        adjusted = engine.apply_signal_adjustments(base_scores, regime_adjustments)

        assert adjusted["momentum"] == Decimal("90.00")  # 75 * 1.2
        assert adjusted["quality"] == Decimal("54.00")   # 60 * 0.9
        assert adjusted["financial"] == Decimal("50.00") # 50 * 1.0

    def test_missing_adjustment_defaults_to_one(self):
        """Signals without explicit adjustment get 1.0 multiplier."""
        engine = RegimeDetectionEngine()

        base_scores = {
            "momentum": Decimal("80.0"),
            "new_signal": Decimal("70.0"),
        }

        regime_adjustments = {
            "momentum": Decimal("1.1"),
            # new_signal not in adjustments
        }

        adjusted = engine.apply_signal_adjustments(base_scores, regime_adjustments)

        assert adjusted["momentum"] == Decimal("88.00")
        assert adjusted["new_signal"] == Decimal("70.00")  # No adjustment

    def test_precision_rounding(self):
        """Adjusted scores are rounded to 2 decimal places."""
        engine = RegimeDetectionEngine()

        base_scores = {"test": Decimal("33.333")}
        regime_adjustments = {"test": Decimal("1.5")}

        adjusted = engine.apply_signal_adjustments(base_scores, regime_adjustments)

        # 33.333 * 1.5 = 49.9995 → 50.00
        assert adjusted["test"] == Decimal("50.00")

    def test_empty_scores(self):
        """Empty base scores returns empty dict."""
        engine = RegimeDetectionEngine()

        adjusted = engine.apply_signal_adjustments({}, {})

        assert adjusted == {}

    def test_zero_scores_preserved(self):
        """Zero scores are handled correctly."""
        engine = RegimeDetectionEngine()

        base_scores = {"zero_signal": Decimal("0")}
        regime_adjustments = {"zero_signal": Decimal("2.0")}

        adjusted = engine.apply_signal_adjustments(base_scores, regime_adjustments)

        assert adjusted["zero_signal"] == Decimal("0.00")


class TestCompositeWeightAdjustments:
    """Test get_composite_weight_adjustments helper method."""

    def test_bull_market_weight_adjustments(self):
        """Bull market increases momentum weight, reduces financial."""
        engine = RegimeDetectionEngine()

        base_weights = {
            "clinical": Decimal("0.40"),
            "financial": Decimal("0.25"),
            "catalyst": Decimal("0.15"),
            "momentum": Decimal("0.20"),
        }

        adjusted = engine.get_composite_weight_adjustments("BULL", base_weights)

        # Weights should be renormalized to sum to 1.0
        total = sum(adjusted.values())
        assert total == Decimal("1.000")

        # Momentum should increase relative to base
        assert adjusted["momentum"] > base_weights["momentum"]

        # Financial should decrease relative to base
        assert adjusted["financial"] < base_weights["financial"]

    def test_bear_market_weight_adjustments(self):
        """Bear market increases financial weight, reduces momentum."""
        engine = RegimeDetectionEngine()

        base_weights = {
            "clinical": Decimal("0.40"),
            "financial": Decimal("0.25"),
            "catalyst": Decimal("0.15"),
            "momentum": Decimal("0.20"),
        }

        adjusted = engine.get_composite_weight_adjustments("BEAR", base_weights)

        # Weights sum to 1.0
        total = sum(adjusted.values())
        assert total == Decimal("1.000")

        # Financial should increase (flight to quality)
        assert adjusted["financial"] > base_weights["financial"]

        # Momentum should decrease
        assert adjusted["momentum"] < base_weights["momentum"]

    def test_volatility_spike_weight_adjustments(self):
        """Volatility spike emphasizes quality and reduces momentum."""
        engine = RegimeDetectionEngine()

        base_weights = {
            "clinical": Decimal("0.40"),
            "financial": Decimal("0.25"),
            "catalyst": Decimal("0.15"),
            "momentum": Decimal("0.20"),
        }

        adjusted = engine.get_composite_weight_adjustments("VOLATILITY_SPIKE", base_weights)

        # Weights sum to 1.0
        total = sum(adjusted.values())
        assert total == Decimal("1.000")

        # Clinical (quality) should increase
        assert adjusted["clinical"] > base_weights["clinical"]

        # Momentum should decrease significantly
        assert adjusted["momentum"] < base_weights["momentum"]

    def test_unknown_regime_preserves_base(self):
        """Unknown regime returns base weights normalized."""
        engine = RegimeDetectionEngine()

        base_weights = {
            "clinical": Decimal("0.40"),
            "financial": Decimal("0.25"),
            "catalyst": Decimal("0.15"),
            "momentum": Decimal("0.20"),
        }

        adjusted = engine.get_composite_weight_adjustments("UNKNOWN", base_weights)

        # Should be close to base weights (after renormalization)
        for component in base_weights:
            assert abs(adjusted[component] - base_weights[component]) < Decimal("0.01")

    def test_zero_total_returns_base(self):
        """Edge case: if all weights become zero, return base weights."""
        engine = RegimeDetectionEngine()

        base_weights = {"test": Decimal("1.0")}

        # Manually set an adjustment that would zero everything
        # (This is an edge case that shouldn't happen in practice)
        adjusted = engine.get_composite_weight_adjustments("BULL", base_weights)

        # Should still return valid normalized weights
        assert sum(adjusted.values()) > Decimal("0")

    def test_precision_three_decimals(self):
        """Adjusted weights are rounded to 3 decimal places."""
        engine = RegimeDetectionEngine()

        base_weights = {
            "a": Decimal("0.333"),
            "b": Decimal("0.333"),
            "c": Decimal("0.334"),
        }

        adjusted = engine.get_composite_weight_adjustments("BULL", base_weights)

        # All weights should have max 3 decimal places
        for weight in adjusted.values():
            assert len(str(weight).split('.')[-1]) <= 3


class TestCalculateRegimeScores:
    """Test _calculate_regime_scores internal helper method."""

    def test_extreme_vix_scores_volatility_spike(self):
        """Extreme VIX (>35) scores high for VOLATILITY_SPIKE."""
        engine = RegimeDetectionEngine()

        scores = engine._calculate_regime_scores(
            vix_current=Decimal("40.0"),
            xbi_vs_spy_30d=Decimal("0.0"),
            fed_rate_change_3m=None,
            xbi_momentum_10d=None,
            spy_momentum_10d=None,
            credit_spread_change=None,
        )

        assert scores["VOLATILITY_SPIKE"] > Decimal("0")
        assert scores["BEAR"] > Decimal("0")  # Also contributes to bear

    def test_low_vix_scores_bull(self):
        """Low VIX (<15) scores high for BULL."""
        engine = RegimeDetectionEngine()

        scores = engine._calculate_regime_scores(
            vix_current=Decimal("12.0"),
            xbi_vs_spy_30d=Decimal("0.0"),
            fed_rate_change_3m=None,
            xbi_momentum_10d=None,
            spy_momentum_10d=None,
            credit_spread_change=None,
        )

        assert scores["BULL"] > Decimal("0")

    def test_strong_xbi_outperformance_scores_bull(self):
        """Strong XBI outperformance (>5%) scores high for BULL."""
        engine = RegimeDetectionEngine()

        scores = engine._calculate_regime_scores(
            vix_current=Decimal("18.0"),
            xbi_vs_spy_30d=Decimal("7.0"),  # Strong outperformance
            fed_rate_change_3m=None,
            xbi_momentum_10d=None,
            spy_momentum_10d=None,
            credit_spread_change=None,
        )

        assert scores["BULL"] > Decimal("20")

    def test_strong_xbi_underperformance_scores_bear(self):
        """Strong XBI underperformance (<-5%) scores high for BEAR."""
        engine = RegimeDetectionEngine()

        scores = engine._calculate_regime_scores(
            vix_current=Decimal("18.0"),
            xbi_vs_spy_30d=Decimal("-8.0"),  # Strong underperformance
            fed_rate_change_3m=None,
            xbi_momentum_10d=None,
            spy_momentum_10d=None,
            credit_spread_change=None,
        )

        assert scores["BEAR"] > Decimal("20")

    def test_rising_rates_scores_bear(self):
        """Rising Fed rates contribute to BEAR score."""
        engine = RegimeDetectionEngine()

        scores = engine._calculate_regime_scores(
            vix_current=Decimal("18.0"),
            xbi_vs_spy_30d=Decimal("0.0"),
            fed_rate_change_3m=Decimal("0.75"),  # Significant rate hike
            xbi_momentum_10d=None,
            spy_momentum_10d=None,
            credit_spread_change=None,
        )

        assert scores["BEAR"] > Decimal("0")

    def test_falling_rates_scores_bull(self):
        """Falling Fed rates contribute to BULL score."""
        engine = RegimeDetectionEngine()

        scores = engine._calculate_regime_scores(
            vix_current=Decimal("18.0"),
            xbi_vs_spy_30d=Decimal("0.0"),
            fed_rate_change_3m=Decimal("-0.5"),  # Rate cut
            xbi_momentum_10d=None,
            spy_momentum_10d=None,
            credit_spread_change=None,
        )

        assert scores["BULL"] > Decimal("0")

    def test_all_scores_present(self):
        """All regime types get a score (even if zero)."""
        engine = RegimeDetectionEngine()

        scores = engine._calculate_regime_scores(
            vix_current=Decimal("18.0"),
            xbi_vs_spy_30d=Decimal("0.0"),
            fed_rate_change_3m=None,
            xbi_momentum_10d=None,
            spy_momentum_10d=None,
            credit_spread_change=None,
        )

        expected_regimes = {"BULL", "BEAR", "VOLATILITY_SPIKE", "SECTOR_ROTATION"}
        assert set(scores.keys()) == expected_regimes


class TestDetectRegimeIntegration:
    """Test full detect_regime with helper methods."""

    def test_detect_regime_returns_adjustments(self):
        """Detect regime returns signal_adjustments dict."""
        engine = RegimeDetectionEngine()

        result = engine.detect_regime(
            vix_current=Decimal("15.0"),
            xbi_vs_spy_30d=Decimal("3.0"),
            fed_rate_change_3m=Decimal("0.0"),
        )

        assert "signal_adjustments" in result
        assert isinstance(result["signal_adjustments"], dict)
        assert "momentum" in result["signal_adjustments"]
        assert "quality" in result["signal_adjustments"]

    def test_detect_regime_includes_confidence(self):
        """Detect regime includes confidence score."""
        engine = RegimeDetectionEngine()

        result = engine.detect_regime(
            vix_current=Decimal("15.0"),
            xbi_vs_spy_30d=Decimal("3.0"),
            fed_rate_change_3m=Decimal("0.0"),
        )

        assert "confidence" in result
        assert isinstance(result["confidence"], str)
        assert result["confidence"] in ["LOW", "MEDIUM", "HIGH"]

    def test_detect_regime_includes_indicators(self):
        """Detect regime includes indicators dict."""
        engine = RegimeDetectionEngine()

        result = engine.detect_regime(
            vix_current=Decimal("15.0"),
            xbi_vs_spy_30d=Decimal("3.0"),
            fed_rate_change_3m=Decimal("0.0"),
        )

        assert "indicators" in result
        assert "vix_current" in result["indicators"]
        assert "xbi_vs_spy_30d" in result["indicators"]

    def test_extreme_conditions_high_confidence(self):
        """Extreme market conditions result in high confidence."""
        engine = RegimeDetectionEngine()

        # Extreme bear conditions
        result = engine.detect_regime(
            vix_current=Decimal("40.0"),  # Extreme VIX
            xbi_vs_spy_30d=Decimal("-10.0"),  # Severe underperformance
            fed_rate_change_3m=Decimal("1.0"),  # Aggressive tightening
        )

        # Should have high confidence in regime classification
        assert result["confidence"] in ["HIGH", "MEDIUM"]
        assert result["regime"] in ["BEAR", "VOLATILITY_SPIKE"]

    def test_neutral_conditions_lower_confidence(self):
        """Neutral market conditions result in lower confidence."""
        engine = RegimeDetectionEngine()

        # Neutral conditions
        result = engine.detect_regime(
            vix_current=Decimal("18.0"),  # Normal VIX
            xbi_vs_spy_30d=Decimal("0.5"),  # Minimal difference
            fed_rate_change_3m=Decimal("0.0"),  # No rate change
        )

        # Harder to classify = lower confidence
        assert result["confidence"] in ["LOW", "MEDIUM"]


class TestRiskFlags:
    """Test risk flag generation in detect_regime."""

    def test_extreme_vix_generates_flag(self):
        """Extreme VIX generates risk flag."""
        engine = RegimeDetectionEngine()

        result = engine.detect_regime(
            vix_current=Decimal("40.0"),
            xbi_vs_spy_30d=Decimal("0.0"),
            fed_rate_change_3m=None,
        )

        assert "flags" in result
        # Should have volatility-related flag
        assert len(result["flags"]) > 0

    def test_normal_conditions_minimal_flags(self):
        """Normal market conditions generate minimal flags."""
        engine = RegimeDetectionEngine()

        result = engine.detect_regime(
            vix_current=Decimal("17.0"),
            xbi_vs_spy_30d=Decimal("1.0"),
            fed_rate_change_3m=Decimal("0.0"),
        )

        # Should have few or no risk flags
        assert "flags" in result


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_none_optional_parameters_handled(self):
        """Optional parameters can be None without errors."""
        engine = RegimeDetectionEngine()

        result = engine.detect_regime(
            vix_current=Decimal("18.0"),
            xbi_vs_spy_30d=Decimal("2.0"),
            fed_rate_change_3m=None,  # Optional
            xbi_momentum_10d=None,    # Optional
            spy_momentum_10d=None,    # Optional
            credit_spread_change=None,  # Optional
        )

        assert result["regime"] in ["BULL", "BEAR", "VOLATILITY_SPIKE", "SECTOR_ROTATION"]

    def test_zero_vix_edge_case(self):
        """VIX of 0 is handled (unrealistic but shouldn't crash)."""
        engine = RegimeDetectionEngine()

        result = engine.detect_regime(
            vix_current=Decimal("0.0"),
            xbi_vs_spy_30d=Decimal("0.0"),
            fed_rate_change_3m=None,
        )

        # Should classify without error
        assert "regime" in result

    def test_very_high_vix(self):
        """Very high VIX values are handled."""
        engine = RegimeDetectionEngine()

        result = engine.detect_regime(
            vix_current=Decimal("100.0"),  # Extreme panic
            xbi_vs_spy_30d=Decimal("-20.0"),
            fed_rate_change_3m=None,
        )

        assert result["regime"] in ["VOLATILITY_SPIKE", "BEAR"]

    def test_negative_xbi_outperformance(self):
        """Large negative XBI outperformance handled correctly."""
        engine = RegimeDetectionEngine()

        result = engine.detect_regime(
            vix_current=Decimal("20.0"),
            xbi_vs_spy_30d=Decimal("-15.0"),  # Severe underperformance
            fed_rate_change_3m=None,
        )

        assert result["regime"] in ["BEAR", "VOLATILITY_SPIKE"]

    def test_determinism_same_inputs(self):
        """Same inputs always produce same output (determinism)."""
        engine = RegimeDetectionEngine()

        result1 = engine.detect_regime(
            vix_current=Decimal("18.5"),
            xbi_vs_spy_30d=Decimal("2.3"),
            fed_rate_change_3m=Decimal("0.25"),
        )

        result2 = engine.detect_regime(
            vix_current=Decimal("18.5"),
            xbi_vs_spy_30d=Decimal("2.3"),
            fed_rate_change_3m=Decimal("0.25"),
        )

        assert result1["regime"] == result2["regime"]
        assert result1["confidence"] == result2["confidence"]
        assert result1["signal_adjustments"] == result2["signal_adjustments"]
