#!/usr/bin/env python3
"""
test_ic_enhancements.py - Tests for IC enhancement utilities

Tests the momentum signal calculation improvements:
- Reduced saturation (slope=150, clamp 5-95)
- Volatility-adjusted alpha
- Data completeness tracking
- PIT integrity requirements
"""

import unittest
from decimal import Decimal
from datetime import date

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.modules.ic_enhancements import (
    compute_momentum_signal,
    compute_momentum_signal_with_fallback,
    compute_alpha_for_score_delta,
    MomentumSignal,
    MultiWindowMomentumInput,
    MOMENTUM_SLOPE,
    MOMENTUM_SCORE_MIN,
    MOMENTUM_SCORE_MAX,
    MOMENTUM_STRONG_THRESHOLD,
    VOLATILITY_BASELINE,
)


class TestMomentumSignalV2(unittest.TestCase):
    """Tests for improved momentum signal calculation with confidence shrinkage."""

    def test_basic_positive_alpha(self):
        """Positive 10% alpha with conf 0.7 produces shrunk score.

        Raw: 50 + 0.10 * 150 = 65
        Shrunk: 50 + 0.7 * (65 - 50) = 50 + 10.5 = 60.5
        """
        result = compute_momentum_signal(
            return_60d=Decimal("0.15"),
            benchmark_return_60d=Decimal("0.05"),
        )
        self.assertEqual(result.alpha_60d, Decimal("0.1000"))
        # After shrinkage: 50 + 0.7 * (65-50) = 60.5
        self.assertEqual(result.momentum_score, Decimal("60.50"))
        self.assertEqual(result.data_completeness, Decimal("1.0"))

    def test_basic_negative_alpha(self):
        """Negative 10% alpha with conf 0.7 produces shrunk score.

        Raw: 50 - 0.10 * 150 = 35
        Shrunk: 50 + 0.7 * (35 - 50) = 50 - 10.5 = 39.5
        """
        result = compute_momentum_signal(
            return_60d=Decimal("-0.05"),
            benchmark_return_60d=Decimal("0.05"),
        )
        self.assertEqual(result.alpha_60d, Decimal("-0.1000"))
        # After shrinkage: 50 + 0.7 * (35-50) = 39.5
        self.assertEqual(result.momentum_score, Decimal("39.50"))

    def test_neutral_alpha(self):
        """Zero alpha should produce neutral score of 50 (shrinkage has no effect)."""
        result = compute_momentum_signal(
            return_60d=Decimal("0.05"),
            benchmark_return_60d=Decimal("0.05"),
        )
        self.assertEqual(result.alpha_60d, Decimal("0.0000"))
        self.assertEqual(result.momentum_score, Decimal("50.00"))

    def test_clamp_upper_bound(self):
        """Large positive alpha with conf 0.9 after shrinkage.

        Raw: 50 + 0.35 * 150 = 102.5 -> clamped to 95
        Shrunk: 50 + 0.9 * (95 - 50) = 50 + 40.5 = 90.5
        """
        result = compute_momentum_signal(
            return_60d=Decimal("0.40"),
            benchmark_return_60d=Decimal("0.05"),
        )
        # After shrinkage: 50 + 0.9 * (95-50) = 90.5
        self.assertEqual(result.momentum_score, Decimal("90.50"))

    def test_clamp_lower_bound(self):
        """Large negative alpha with conf 0.9 after shrinkage.

        Raw: 50 - 0.40 * 150 = -10 -> clamped to 5
        Shrunk: 50 + 0.9 * (5 - 50) = 50 - 40.5 = 9.5
        """
        result = compute_momentum_signal(
            return_60d=Decimal("-0.35"),
            benchmark_return_60d=Decimal("0.05"),
        )
        # After shrinkage: 50 + 0.9 * (5-50) = 9.5
        self.assertEqual(result.momentum_score, Decimal("9.50"))

    def test_wider_clamp_reduces_saturation(self):
        """Verify the new clamp (5-95) is wider than the old (10-90)."""
        self.assertEqual(MOMENTUM_SCORE_MIN, Decimal("5"))
        self.assertEqual(MOMENTUM_SCORE_MAX, Decimal("95"))
        self.assertEqual(MOMENTUM_SLOPE, Decimal("150"))


class TestMomentumMissingData(unittest.TestCase):
    """Tests for momentum signal with missing data."""

    def test_missing_return(self):
        """Missing stock return should produce neutral score with low completeness."""
        result = compute_momentum_signal(
            return_60d=None,
            benchmark_return_60d=Decimal("0.05"),
        )
        self.assertEqual(result.momentum_score, Decimal("50"))
        self.assertIsNone(result.alpha_60d)
        self.assertEqual(result.data_completeness, Decimal("0.3"))

    def test_missing_benchmark(self):
        """Missing benchmark return should produce neutral score."""
        result = compute_momentum_signal(
            return_60d=Decimal("0.10"),
            benchmark_return_60d=None,
        )
        self.assertEqual(result.momentum_score, Decimal("50"))
        self.assertIsNone(result.alpha_60d)
        self.assertEqual(result.data_completeness, Decimal("0.3"))

    def test_missing_both(self):
        """Missing both returns should produce neutral score with zero completeness."""
        result = compute_momentum_signal(
            return_60d=None,
            benchmark_return_60d=None,
        )
        self.assertEqual(result.momentum_score, Decimal("50"))
        self.assertIsNone(result.alpha_60d)
        self.assertEqual(result.data_completeness, Decimal("0.0"))

    def test_complete_data(self):
        """Complete data should have data_completeness of 1.0."""
        result = compute_momentum_signal(
            return_60d=Decimal("0.10"),
            benchmark_return_60d=Decimal("0.05"),
        )
        self.assertEqual(result.data_completeness, Decimal("1.0"))


class TestVolatilityAdjustedAlpha(unittest.TestCase):
    """Tests for volatility-adjusted momentum alpha."""

    def test_vol_adjusted_alpha_calculation(self):
        """Vol-adjusted alpha should be alpha / vol."""
        result = compute_momentum_signal(
            return_60d=Decimal("0.15"),
            benchmark_return_60d=Decimal("0.05"),
            annualized_vol=Decimal("0.50"),
            use_vol_adjusted_alpha=False,  # Don't use for scoring, but calculate
        )
        # alpha = 0.10, vol = 0.50, vol_adj = 0.10 / 0.50 = 0.20
        self.assertEqual(result.alpha_vol_adjusted, Decimal("0.2000"))

    def test_vol_adjusted_scoring_baseline_vol(self):
        """At baseline vol (50%), vol-adjusted score with shrinkage.

        Vol-adjusted alpha = 0.10 / 0.50 = 0.20
        Scoring alpha = 0.20 * 0.50 (baseline) = 0.10
        Raw score = 50 + 0.10 * 150 = 65
        Shrunk: 50 + 0.7 * (65 - 50) = 60.5
        """
        result = compute_momentum_signal(
            return_60d=Decimal("0.15"),
            benchmark_return_60d=Decimal("0.05"),
            annualized_vol=VOLATILITY_BASELINE,
            use_vol_adjusted_alpha=True,
        )
        # After shrinkage: 50 + 0.7 * (65-50) = 60.5
        self.assertEqual(result.momentum_score, Decimal("60.50"))

    def test_vol_adjusted_penalizes_high_vol(self):
        """High vol names should get lower momentum scores."""
        # Raw alpha case
        raw_result = compute_momentum_signal(
            return_60d=Decimal("0.15"),
            benchmark_return_60d=Decimal("0.05"),
            annualized_vol=Decimal("1.00"),
            use_vol_adjusted_alpha=False,
        )
        # Vol-adjusted case
        vol_adj_result = compute_momentum_signal(
            return_60d=Decimal("0.15"),
            benchmark_return_60d=Decimal("0.05"),
            annualized_vol=Decimal("1.00"),
            use_vol_adjusted_alpha=True,
        )
        # Vol-adjusted should be lower (penalizes high vol)
        self.assertLess(vol_adj_result.momentum_score, raw_result.momentum_score)

    def test_vol_adjusted_rewards_low_vol_outperformance(self):
        """Low vol names with same alpha should get higher scores."""
        # High vol (100%)
        high_vol_result = compute_momentum_signal(
            return_60d=Decimal("0.15"),
            benchmark_return_60d=Decimal("0.05"),
            annualized_vol=Decimal("1.00"),
            use_vol_adjusted_alpha=True,
        )
        # Low vol (25%)
        low_vol_result = compute_momentum_signal(
            return_60d=Decimal("0.15"),
            benchmark_return_60d=Decimal("0.05"),
            annualized_vol=Decimal("0.25"),
            use_vol_adjusted_alpha=True,
        )
        # Low vol should get higher score (rewards persistent outperformance)
        self.assertGreater(low_vol_result.momentum_score, high_vol_result.momentum_score)

    def test_vol_adjusted_missing_vol_reduces_completeness(self):
        """When vol is requested but missing, data_completeness should be reduced."""
        result = compute_momentum_signal(
            return_60d=Decimal("0.15"),
            benchmark_return_60d=Decimal("0.05"),
            annualized_vol=None,
            use_vol_adjusted_alpha=True,
        )
        self.assertEqual(result.data_completeness, Decimal("0.8"))
        self.assertIsNone(result.alpha_vol_adjusted)


class TestMomentumConfidence(unittest.TestCase):
    """Tests for momentum signal confidence levels."""

    def test_high_alpha_high_confidence(self):
        """Large alpha (>=20%) should have confidence 0.9."""
        result = compute_momentum_signal(
            return_60d=Decimal("0.25"),
            benchmark_return_60d=Decimal("0.05"),
        )
        self.assertEqual(result.confidence, Decimal("0.9"))

    def test_medium_alpha_good_confidence(self):
        """Medium alpha (10-20%) should have confidence 0.7."""
        result = compute_momentum_signal(
            return_60d=Decimal("0.15"),
            benchmark_return_60d=Decimal("0.05"),
        )
        self.assertEqual(result.confidence, Decimal("0.7"))

    def test_small_alpha_moderate_confidence(self):
        """Small alpha (5-10%) should have confidence 0.5."""
        result = compute_momentum_signal(
            return_60d=Decimal("0.10"),
            benchmark_return_60d=Decimal("0.05"),
        )
        self.assertEqual(result.confidence, Decimal("0.5"))

    def test_tiny_alpha_low_confidence(self):
        """Tiny alpha (<5%) should have confidence 0.4."""
        result = compute_momentum_signal(
            return_60d=Decimal("0.07"),
            benchmark_return_60d=Decimal("0.05"),
        )
        self.assertEqual(result.confidence, Decimal("0.4"))

    def test_missing_data_low_confidence(self):
        """Missing data should have confidence 0.3."""
        result = compute_momentum_signal(
            return_60d=None,
            benchmark_return_60d=Decimal("0.05"),
        )
        self.assertEqual(result.confidence, Decimal("0.3"))


class TestMomentumDeterminism(unittest.TestCase):
    """Tests for deterministic behavior."""

    def test_identical_inputs_identical_outputs(self):
        """Same inputs should always produce same outputs."""
        result1 = compute_momentum_signal(
            return_60d=Decimal("0.12345"),
            benchmark_return_60d=Decimal("0.06789"),
            annualized_vol=Decimal("0.45678"),
        )
        result2 = compute_momentum_signal(
            return_60d=Decimal("0.12345"),
            benchmark_return_60d=Decimal("0.06789"),
            annualized_vol=Decimal("0.45678"),
        )
        self.assertEqual(result1.momentum_score, result2.momentum_score)
        self.assertEqual(result1.alpha_60d, result2.alpha_60d)
        self.assertEqual(result1.alpha_vol_adjusted, result2.alpha_vol_adjusted)
        self.assertEqual(result1.confidence, result2.confidence)
        self.assertEqual(result1.data_completeness, result2.data_completeness)

    def test_all_decimal_outputs(self):
        """All numeric outputs should be Decimal type."""
        result = compute_momentum_signal(
            return_60d=Decimal("0.10"),
            benchmark_return_60d=Decimal("0.05"),
            annualized_vol=Decimal("0.50"),
        )
        self.assertIsInstance(result.momentum_score, Decimal)
        self.assertIsInstance(result.alpha_60d, Decimal)
        self.assertIsInstance(result.alpha_vol_adjusted, Decimal)
        self.assertIsInstance(result.return_60d, Decimal)
        self.assertIsInstance(result.benchmark_return_60d, Decimal)
        self.assertIsInstance(result.confidence, Decimal)
        self.assertIsInstance(result.data_completeness, Decimal)


class TestMomentumSignalDataclass(unittest.TestCase):
    """Tests for MomentumSignal dataclass fields."""

    def test_all_fields_present(self):
        """MomentumSignal should have all required fields."""
        result = compute_momentum_signal(
            return_60d=Decimal("0.10"),
            benchmark_return_60d=Decimal("0.05"),
            annualized_vol=Decimal("0.50"),
        )
        self.assertTrue(hasattr(result, 'momentum_score'))
        self.assertTrue(hasattr(result, 'alpha_60d'))
        self.assertTrue(hasattr(result, 'alpha_vol_adjusted'))
        self.assertTrue(hasattr(result, 'return_60d'))
        self.assertTrue(hasattr(result, 'benchmark_return_60d'))
        self.assertTrue(hasattr(result, 'confidence'))
        self.assertTrue(hasattr(result, 'data_completeness'))


class TestZeroBenchmarkFalsyBug(unittest.TestCase):
    """
    Regression tests for the 0.0 benchmark falsy bug.

    Bug: Using `or` fallback for numeric values treats 0.0 as falsy,
    causing xbi_return_60d=0.0 to be skipped, resulting in None alpha.

    Fix: Use `x if x is not None else fallback` pattern instead of `or`.

    This test ensures the bug never regresses.
    """

    def test_zero_benchmark_return_computes_alpha(self):
        """
        benchmark_return_60d=0.0 should compute alpha, not return None.

        This is the critical regression test for the falsy bug.
        When XBI return is exactly 0.0 (flat market), alpha should equal
        the stock's return, not be None.

        With shrinkage (conf=0.7): 50 + 0.7 * (65 - 50) = 60.5
        """
        result = compute_momentum_signal(
            return_60d=Decimal("0.10"),  # Stock gained 10%
            benchmark_return_60d=Decimal("0.0"),  # XBI was flat
        )

        # CRITICAL: alpha should be computed, not None
        self.assertIsNotNone(result.alpha_60d, (
            "alpha_60d is None when benchmark=0.0! "
            "This is the falsy bug: 0.0 is being treated as missing data."
        ))

        # Alpha should equal stock return (10% - 0% = 10%)
        self.assertEqual(result.alpha_60d, Decimal("0.1000"))

        # Score should be above neutral (positive alpha with shrinkage)
        # Raw: 65, shrunk: 60.5
        self.assertGreater(result.momentum_score, Decimal("50"))
        self.assertEqual(result.momentum_score, Decimal("60.50"))

        # Data completeness should be 1.0 (both values present)
        self.assertEqual(result.data_completeness, Decimal("1.0"))

    def test_zero_benchmark_different_from_none_benchmark(self):
        """benchmark=0.0 and benchmark=None must produce different results."""
        result_zero = compute_momentum_signal(
            return_60d=Decimal("0.10"),
            benchmark_return_60d=Decimal("0.0"),
        )
        result_none = compute_momentum_signal(
            return_60d=Decimal("0.10"),
            benchmark_return_60d=None,
        )

        # Zero benchmark should have computed alpha
        self.assertIsNotNone(result_zero.alpha_60d)
        self.assertEqual(result_zero.alpha_60d, Decimal("0.1000"))

        # None benchmark should have None alpha
        self.assertIsNone(result_none.alpha_60d)

        # Scores should differ
        self.assertNotEqual(result_zero.momentum_score, result_none.momentum_score)

    def test_zero_return_with_nonzero_benchmark(self):
        """return_60d=0.0 with nonzero benchmark should compute negative alpha."""
        result = compute_momentum_signal(
            return_60d=Decimal("0.0"),  # Stock was flat
            benchmark_return_60d=Decimal("0.10"),  # XBI gained 10%
        )

        # Alpha should be computed: 0% - 10% = -10%
        self.assertIsNotNone(result.alpha_60d)
        self.assertEqual(result.alpha_60d, Decimal("-0.1000"))

        # Score should be below neutral (negative alpha)
        self.assertLess(result.momentum_score, Decimal("50"))

    def test_both_zero_produces_neutral(self):
        """Both return and benchmark at 0.0 should produce neutral score."""
        result = compute_momentum_signal(
            return_60d=Decimal("0.0"),
            benchmark_return_60d=Decimal("0.0"),
        )

        # Alpha should be 0
        self.assertIsNotNone(result.alpha_60d)
        self.assertEqual(result.alpha_60d, Decimal("0.0000"))

        # Score should be exactly neutral
        self.assertEqual(result.momentum_score, Decimal("50.00"))


class TestMultiWindowMomentumFallback(unittest.TestCase):
    """Tests for multi-window momentum with fallback for improved coverage."""

    def test_prefers_60d_window(self):
        """When all windows available, should prefer 60d.

        alpha = 0.05, raw_score = 57.5, conf = 0.7
        shrunk = 50 + 0.7 * (57.5 - 50) = 55.25
        """
        inputs = MultiWindowMomentumInput(
            return_20d=Decimal("0.05"),
            return_60d=Decimal("0.10"),
            return_120d=Decimal("0.15"),
            benchmark_20d=Decimal("0.02"),
            benchmark_60d=Decimal("0.05"),
            benchmark_120d=Decimal("0.08"),
        )
        result = compute_momentum_signal_with_fallback(inputs)

        self.assertEqual(result.window_used, 60)
        # alpha = 0.10 - 0.05 = 0.05
        self.assertEqual(result.alpha_60d, Decimal("0.0500"))
        self.assertEqual(result.data_status, "applied")
        # Verify shrinkage: raw 57.5, conf 0.7 -> 55.25
        self.assertEqual(result.momentum_score, Decimal("55.25"))

    def test_fallback_to_120d(self):
        """When 60d unavailable, should fallback to 120d.

        alpha = 0.07, raw = 60.5, conf = 0.6 (120d base)
        shrunk = 50 + 0.6 * (60.5 - 50) = 56.3
        """
        inputs = MultiWindowMomentumInput(
            return_20d=Decimal("0.05"),
            return_60d=None,
            return_120d=Decimal("0.15"),
            benchmark_20d=Decimal("0.02"),
            benchmark_60d=None,
            benchmark_120d=Decimal("0.08"),
        )
        result = compute_momentum_signal_with_fallback(inputs)

        self.assertEqual(result.window_used, 120)
        # alpha = 0.15 - 0.08 = 0.07
        self.assertEqual(result.alpha_60d, Decimal("0.0700"))
        self.assertEqual(result.data_completeness, Decimal("0.8"))
        # Shrunk: 50 + 0.6 * (60.5 - 50) = 56.30
        self.assertEqual(result.momentum_score, Decimal("56.30"))

    def test_fallback_to_20d(self):
        """When 60d and 120d unavailable, should fallback to 20d.

        alpha = 0.03, raw = 54.5, conf = 0.5 (20d base)
        shrunk = 50 + 0.5 * (54.5 - 50) = 52.25
        """
        inputs = MultiWindowMomentumInput(
            return_20d=Decimal("0.05"),
            return_60d=None,
            return_120d=None,
            benchmark_20d=Decimal("0.02"),
            benchmark_60d=None,
            benchmark_120d=None,
        )
        result = compute_momentum_signal_with_fallback(inputs)

        self.assertEqual(result.window_used, 20)
        # alpha = 0.05 - 0.02 = 0.03
        self.assertEqual(result.alpha_60d, Decimal("0.0300"))
        self.assertEqual(result.data_completeness, Decimal("0.6"))
        # Shrunk: 50 + 0.5 * (54.5 - 50) = 52.25
        self.assertEqual(result.momentum_score, Decimal("52.25"))

    def test_missing_prices_status(self):
        """When no windows available, should return missing_prices status."""
        inputs = MultiWindowMomentumInput(
            return_20d=None,
            return_60d=None,
            return_120d=None,
            benchmark_20d=None,
            benchmark_60d=None,
            benchmark_120d=None,
        )
        result = compute_momentum_signal_with_fallback(inputs)

        self.assertIsNone(result.window_used)
        self.assertEqual(result.data_status, "missing_prices")
        self.assertEqual(result.momentum_score, Decimal("50"))
        self.assertEqual(result.data_completeness, Decimal("0.0"))

    def test_requires_both_stock_and_benchmark(self):
        """Should only use window if both stock and benchmark available."""
        inputs = MultiWindowMomentumInput(
            return_60d=Decimal("0.10"),  # Stock has 60d
            benchmark_60d=None,           # But no XBI 60d
            return_20d=Decimal("0.05"),   # Stock has 20d
            benchmark_20d=Decimal("0.02"), # XBI has 20d
        )
        result = compute_momentum_signal_with_fallback(inputs)

        # Should fallback to 20d since 60d is incomplete
        self.assertEqual(result.window_used, 20)


class TestMultiWindowConfidenceRules(unittest.TestCase):
    """Tests for confidence based on window and data availability."""

    def test_60d_window_base_confidence(self):
        """60d window should have base confidence of 0.7."""
        inputs = MultiWindowMomentumInput(
            return_60d=Decimal("0.05"),
            benchmark_60d=Decimal("0.02"),
            trading_days_available=60,
        )
        result = compute_momentum_signal_with_fallback(inputs)

        # Base 0.7 for 60d window, no days penalty
        # Alpha 0.03 < 0.05, so no boost
        self.assertEqual(result.confidence, Decimal("0.7000"))

    def test_20d_window_lower_confidence(self):
        """20d window should have lower base confidence."""
        inputs = MultiWindowMomentumInput(
            return_20d=Decimal("0.05"),
            benchmark_20d=Decimal("0.02"),
            trading_days_available=20,
        )
        result = compute_momentum_signal_with_fallback(inputs)

        # Base 0.5 for 20d, -0.2 for 20-39 days available
        self.assertEqual(result.confidence, Decimal("0.3000"))

    def test_strong_signal_boosts_confidence(self):
        """Strong alpha should boost confidence."""
        inputs = MultiWindowMomentumInput(
            return_60d=Decimal("0.30"),
            benchmark_60d=Decimal("0.05"),
            trading_days_available=60,
        )
        result = compute_momentum_signal_with_fallback(inputs)

        # alpha = 0.25 >= 0.20, so +0.2 boost
        # Base 0.7 + 0.2 = 0.9
        self.assertEqual(result.confidence, Decimal("0.9000"))

    def test_few_trading_days_penalty(self):
        """Few trading days should penalize confidence."""
        inputs = MultiWindowMomentumInput(
            return_60d=Decimal("0.10"),
            benchmark_60d=Decimal("0.05"),
            trading_days_available=30,  # Only 30 days (20-39 range)
        )
        result = compute_momentum_signal_with_fallback(inputs)

        # Base 0.7 - 0.2 (days penalty) = 0.5
        # alpha 0.05 gives medium boost = +0
        self.assertEqual(result.confidence, Decimal("0.5000"))


class TestMultiWindowDataStatus(unittest.TestCase):
    """Tests for data_status tracking."""

    def test_applied_status(self):
        """Good confidence should result in 'applied' status."""
        inputs = MultiWindowMomentumInput(
            return_60d=Decimal("0.15"),
            benchmark_60d=Decimal("0.05"),
            trading_days_available=60,
        )
        result = compute_momentum_signal_with_fallback(inputs)

        self.assertEqual(result.data_status, "applied")

    def test_low_conf_status(self):
        """Low confidence should result in 'computed_low_conf' status."""
        inputs = MultiWindowMomentumInput(
            return_20d=Decimal("0.03"),  # Small alpha
            benchmark_20d=Decimal("0.02"),
            trading_days_available=15,  # Very few days
        )
        result = compute_momentum_signal_with_fallback(inputs)

        # Base 0.5 - 0.3 (< 20 days) = 0.2 < 0.5 threshold
        # But minimum is 0.3
        self.assertEqual(result.data_status, "computed_low_conf")

    def test_determinism(self):
        """Same inputs should produce identical outputs."""
        inputs = MultiWindowMomentumInput(
            return_20d=Decimal("0.05"),
            return_60d=Decimal("0.10"),
            return_120d=Decimal("0.15"),
            benchmark_20d=Decimal("0.02"),
            benchmark_60d=Decimal("0.05"),
            benchmark_120d=Decimal("0.08"),
            annualized_vol=Decimal("0.45"),
            trading_days_available=100,
        )

        result1 = compute_momentum_signal_with_fallback(inputs)
        result2 = compute_momentum_signal_with_fallback(inputs)

        self.assertEqual(result1.momentum_score, result2.momentum_score)
        self.assertEqual(result1.alpha_60d, result2.alpha_60d)
        self.assertEqual(result1.window_used, result2.window_used)
        self.assertEqual(result1.data_status, result2.data_status)


class TestFallbackRegressionScenarios(unittest.TestCase):
    """
    Regression tests for fallback + confidence behavior.

    These tests ensure the fallback logic behaves correctly in edge cases
    and that confidence is adjusted appropriately based on window and
    trading days available.
    """

    def test_fallback_uses_120d_when_60d_missing(self):
        """When 60d window is missing, fallback to 120d with correct confidence.

        alpha = 0.07, raw = 60.5, conf = 0.6 (120d base)
        shrunk = 50 + 0.6 * (60.5 - 50) = 56.30
        """
        inputs = MultiWindowMomentumInput(
            return_20d=Decimal("0.03"),
            return_60d=None,  # 60d missing
            return_120d=Decimal("0.12"),  # 120d available
            benchmark_20d=Decimal("0.01"),
            benchmark_60d=None,  # 60d missing
            benchmark_120d=Decimal("0.05"),
            trading_days_available=121,
        )
        result = compute_momentum_signal_with_fallback(inputs)

        # Should use 120d window
        self.assertEqual(result.window_used, 120)
        # alpha = 0.12 - 0.05 = 0.07
        self.assertEqual(result.alpha_60d, Decimal("0.0700"))
        # Base confidence for 120d is 0.6, no days penalty for 121 days
        # alpha 0.07 gives +0.0 boost (0.05 <= alpha < 0.10)
        self.assertEqual(result.confidence, Decimal("0.6000"))
        # Data completeness for 120d is 0.8
        self.assertEqual(result.data_completeness, Decimal("0.8"))
        # Should be applied status (0.6 >= 0.5)
        self.assertEqual(result.data_status, "applied")
        # Shrunk: 50 + 0.6 * (60.5 - 50) = 56.30
        self.assertEqual(result.momentum_score, Decimal("56.30"))

    def test_fallback_uses_20d_when_60d_and_120d_missing(self):
        """When both 60d and 120d missing, fallback to 20d with lower confidence.

        alpha = 0.05, raw = 57.5, conf = 0.3 (20d base - 0.2 penalty)
        shrunk = 50 + 0.3 * (57.5 - 50) = 52.25
        """
        inputs = MultiWindowMomentumInput(
            return_20d=Decimal("0.08"),
            return_60d=None,
            return_120d=None,
            benchmark_20d=Decimal("0.03"),
            benchmark_60d=None,
            benchmark_120d=None,
            trading_days_available=25,
        )
        result = compute_momentum_signal_with_fallback(inputs)

        # Should use 20d window
        self.assertEqual(result.window_used, 20)
        # alpha = 0.08 - 0.03 = 0.05
        self.assertEqual(result.alpha_60d, Decimal("0.0500"))
        # Base confidence for 20d is 0.5, -0.2 penalty for 20-39 days
        # 0.5 - 0.2 = 0.3, but alpha 0.05 gives no boost
        self.assertEqual(result.confidence, Decimal("0.3000"))
        # Data completeness for 20d is 0.6
        self.assertEqual(result.data_completeness, Decimal("0.6"))
        # Should be computed_low_conf status (0.3 < 0.5)
        self.assertEqual(result.data_status, "computed_low_conf")
        # Shrunk: 50 + 0.3 * (57.5 - 50) = 52.25
        self.assertEqual(result.momentum_score, Decimal("52.25"))

    def test_confidence_adjusts_with_few_trading_days(self):
        """Verify confidence is penalized when few trading days available.

        With confidence shrinkage, lower confidence means scores are pulled
        closer to 50, so we verify that scores decrease as confidence decreases.
        """
        # Case 1: Full history (60+ days) - no penalty
        inputs_full = MultiWindowMomentumInput(
            return_60d=Decimal("0.10"),
            benchmark_60d=Decimal("0.05"),
            trading_days_available=60,
        )
        result_full = compute_momentum_signal_with_fallback(inputs_full)

        # Case 2: Limited history (40-59 days) - small penalty
        inputs_limited = MultiWindowMomentumInput(
            return_60d=Decimal("0.10"),
            benchmark_60d=Decimal("0.05"),
            trading_days_available=47,
        )
        result_limited = compute_momentum_signal_with_fallback(inputs_limited)

        # Case 3: Very limited history (20-39 days) - larger penalty
        inputs_sparse = MultiWindowMomentumInput(
            return_60d=Decimal("0.10"),
            benchmark_60d=Decimal("0.05"),
            trading_days_available=30,
        )
        result_sparse = compute_momentum_signal_with_fallback(inputs_sparse)

        # Case 4: Minimal history (<20 days) - max penalty
        inputs_minimal = MultiWindowMomentumInput(
            return_60d=Decimal("0.10"),
            benchmark_60d=Decimal("0.05"),
            trading_days_available=15,
        )
        result_minimal = compute_momentum_signal_with_fallback(inputs_minimal)

        # Verify decreasing confidence with fewer trading days
        # Full: 0.7 + 0.0 (boost) = 0.7, Limited: 0.7 - 0.1 = 0.6, etc.
        self.assertGreater(result_full.confidence, result_limited.confidence)
        self.assertGreater(result_limited.confidence, result_sparse.confidence)
        self.assertGreater(result_sparse.confidence, result_minimal.confidence)

        # With shrinkage, scores should also decrease as confidence decreases
        # (positive alpha gets shrunk closer to 50 with lower confidence)
        self.assertGreater(result_full.momentum_score, result_limited.momentum_score)
        self.assertGreater(result_limited.momentum_score, result_sparse.momentum_score)
        self.assertGreater(result_sparse.momentum_score, result_minimal.momentum_score)

        # All scores should still be above neutral (positive alpha)
        for result in [result_full, result_limited, result_sparse, result_minimal]:
            self.assertGreater(result.momentum_score, Decimal("50"))

    def test_confidence_chooses_better_window_when_days_insufficient(self):
        """
        With only 47 trading days available, 60d path should reduce confidence.

        This test verifies that data quality (trading_days_available) appropriately
        penalizes confidence even when a window is technically computable.
        """
        inputs = MultiWindowMomentumInput(
            return_20d=Decimal("0.05"),
            return_60d=Decimal("0.10"),
            return_120d=Decimal("0.15"),
            benchmark_20d=Decimal("0.02"),
            benchmark_60d=Decimal("0.05"),
            benchmark_120d=Decimal("0.08"),
            trading_days_available=47,  # 40-59 range: -0.1 penalty
        )
        result = compute_momentum_signal_with_fallback(inputs)

        # Should still use 60d (preferred window)
        self.assertEqual(result.window_used, 60)
        # Confidence should be 0.7 - 0.1 = 0.6 (days penalty)
        self.assertEqual(result.confidence, Decimal("0.6000"))
        # But still "applied" since 0.6 >= 0.5
        self.assertEqual(result.data_status, "applied")


class TestGoldenFixtureAllWindows(unittest.TestCase):
    """
    Golden fixture test with 1 ticker + XBI with exact 21/61/121 days.

    This ensures all branches (20d, 60d, 120d) are tested with boundary
    values and the window selection is deterministic.
    """

    def setUp(self):
        """Create golden fixture data."""
        # Simulate a ticker with exactly enough data for each window
        self.golden_inputs_21d = MultiWindowMomentumInput(
            return_20d=Decimal("0.04"),
            return_60d=None,
            return_120d=None,
            benchmark_20d=Decimal("0.02"),
            benchmark_60d=None,
            benchmark_120d=None,
            trading_days_available=21,
        )

        self.golden_inputs_61d = MultiWindowMomentumInput(
            return_20d=Decimal("0.04"),
            return_60d=Decimal("0.10"),
            return_120d=None,
            benchmark_20d=Decimal("0.02"),
            benchmark_60d=Decimal("0.05"),
            benchmark_120d=None,
            trading_days_available=61,
        )

        self.golden_inputs_121d = MultiWindowMomentumInput(
            return_20d=Decimal("0.04"),
            return_60d=Decimal("0.10"),
            return_120d=Decimal("0.18"),
            benchmark_20d=Decimal("0.02"),
            benchmark_60d=Decimal("0.05"),
            benchmark_120d=Decimal("0.08"),
            trading_days_available=121,
        )

    def test_golden_21d_uses_20d_window(self):
        """With only 21 trading days, should use 20d window.

        alpha = 0.02, raw = 53, conf = 0.3 (20d base - 0.2 penalty)
        shrunk = 50 + 0.3 * (53 - 50) = 50.9
        """
        result = compute_momentum_signal_with_fallback(self.golden_inputs_21d)

        self.assertEqual(result.window_used, 20)
        # alpha = 0.04 - 0.02 = 0.02
        self.assertEqual(result.alpha_60d, Decimal("0.0200"))
        # 20d base conf 0.5, 21 days is in 20-39 range: -0.2 penalty -> 0.3
        self.assertEqual(result.confidence, Decimal("0.3000"))
        # Shrunk: 50 + 0.3 * (53 - 50) = 50.90
        self.assertEqual(result.momentum_score, Decimal("50.90"))
        self.assertEqual(result.data_completeness, Decimal("0.6"))
        self.assertEqual(result.data_status, "computed_low_conf")

    def test_golden_61d_uses_60d_window(self):
        """With 61 trading days, should use preferred 60d window.

        alpha = 0.05, raw = 57.5, conf = 0.7 (60d base, no penalty)
        shrunk = 50 + 0.7 * (57.5 - 50) = 55.25
        """
        result = compute_momentum_signal_with_fallback(self.golden_inputs_61d)

        self.assertEqual(result.window_used, 60)
        # alpha = 0.10 - 0.05 = 0.05
        self.assertEqual(result.alpha_60d, Decimal("0.0500"))
        # 60d base conf 0.7, 61 days >= 60: no penalty
        self.assertEqual(result.confidence, Decimal("0.7000"))
        # Shrunk: 50 + 0.7 * (57.5 - 50) = 55.25
        self.assertEqual(result.momentum_score, Decimal("55.25"))
        self.assertEqual(result.data_completeness, Decimal("1.0"))
        self.assertEqual(result.data_status, "applied")

    def test_golden_121d_prefers_60d_window(self):
        """With 121 days (all windows available), should prefer 60d.

        alpha = 0.05, raw = 57.5, conf = 0.7 (60d base, no penalty)
        shrunk = 50 + 0.7 * (57.5 - 50) = 55.25
        """
        result = compute_momentum_signal_with_fallback(self.golden_inputs_121d)

        # Should prefer 60d over 120d even though 120d is available
        self.assertEqual(result.window_used, 60)
        # alpha = 0.10 - 0.05 = 0.05 (60d values)
        self.assertEqual(result.alpha_60d, Decimal("0.0500"))
        # Full confidence (60d + no days penalty)
        self.assertEqual(result.confidence, Decimal("0.7000"))
        # Shrunk: 50 + 0.7 * (57.5 - 50) = 55.25
        self.assertEqual(result.momentum_score, Decimal("55.25"))
        self.assertEqual(result.data_status, "applied")

    def test_golden_determinism(self):
        """Golden fixture results should be byte-identical across runs."""
        result1 = compute_momentum_signal_with_fallback(self.golden_inputs_121d)
        result2 = compute_momentum_signal_with_fallback(self.golden_inputs_121d)

        self.assertEqual(result1.momentum_score, result2.momentum_score)
        self.assertEqual(result1.alpha_60d, result2.alpha_60d)
        self.assertEqual(result1.confidence, result2.confidence)
        self.assertEqual(result1.window_used, result2.window_used)
        self.assertEqual(result1.data_status, result2.data_status)


class TestBenchmarkMissingScenarios(unittest.TestCase):
    """Tests for scenarios where benchmark data is missing."""

    def test_benchmark_missing_for_60d_falls_back(self):
        """Missing 60d benchmark should fall back to next window."""
        inputs = MultiWindowMomentumInput(
            return_60d=Decimal("0.10"),
            benchmark_60d=None,  # Benchmark missing!
            return_120d=Decimal("0.15"),
            benchmark_120d=Decimal("0.08"),
        )
        result = compute_momentum_signal_with_fallback(inputs)

        # Should fall back to 120d since 60d benchmark is missing
        self.assertEqual(result.window_used, 120)

    def test_all_benchmarks_missing_returns_neutral(self):
        """All benchmarks missing should return neutral score."""
        inputs = MultiWindowMomentumInput(
            return_20d=Decimal("0.05"),
            return_60d=Decimal("0.10"),
            return_120d=Decimal("0.15"),
            benchmark_20d=None,
            benchmark_60d=None,
            benchmark_120d=None,
        )
        result = compute_momentum_signal_with_fallback(inputs)

        self.assertIsNone(result.window_used)
        self.assertEqual(result.data_status, "missing_prices")
        self.assertEqual(result.momentum_score, Decimal("50"))

    def test_zero_benchmark_not_treated_as_missing(self):
        """Zero benchmark should be valid, not treated as missing.

        alpha = 0.10, raw = 65, conf = 0.7+0.1 (alpha boost) = 0.8
        shrunk = 50 + 0.8 * (65 - 50) = 62.0
        """
        inputs = MultiWindowMomentumInput(
            return_60d=Decimal("0.10"),
            benchmark_60d=Decimal("0.0"),  # Zero, not None
        )
        result = compute_momentum_signal_with_fallback(inputs)

        # Should use 60d (zero is a valid value)
        self.assertEqual(result.window_used, 60)
        # alpha = 0.10 - 0.0 = 0.10
        self.assertEqual(result.alpha_60d, Decimal("0.1000"))
        self.assertEqual(result.data_status, "applied")
        # Shrunk: 50 + 0.8 * (65 - 50) = 62.0
        self.assertEqual(result.momentum_score, Decimal("62.00"))


class TestGuardrailFlags(unittest.TestCase):
    """Tests for guardrail flags (benchmark missing, return clipped)."""

    def test_benchmark_missing_flag_when_return_exists(self):
        """Should flag when stock return exists but benchmark missing."""
        inputs = MultiWindowMomentumInput(
            return_60d=Decimal("0.10"),  # Stock has return
            benchmark_60d=None,  # But no benchmark
            return_20d=Decimal("0.05"),  # Fallback return
            benchmark_20d=Decimal("0.02"),  # Fallback benchmark
        )
        result = compute_momentum_signal_with_fallback(inputs)

        # Should have flagged the 60d benchmark as missing
        self.assertIn("benchmark_missing_60d", result.guardrail_flags)
        # Should have used 20d fallback
        self.assertEqual(result.window_used, 20)
        # benchmark_missing should be True (diagnostic)
        self.assertTrue(result.benchmark_missing)

    def test_return_clipped_flag(self):
        """Should flag when return was clipped for outlier."""
        inputs = MultiWindowMomentumInput(
            return_60d=Decimal("0.50"),
            benchmark_60d=Decimal("0.05"),
            return_clipped_60d=True,  # This was clipped
        )
        result = compute_momentum_signal_with_fallback(inputs)

        self.assertIn("return_clipped_60d", result.guardrail_flags)
        self.assertTrue(result.return_clipped)
        self.assertEqual(result.window_used, 60)

    def test_no_guardrail_flags_for_normal_data(self):
        """Normal data should have no guardrail flags."""
        inputs = MultiWindowMomentumInput(
            return_60d=Decimal("0.10"),
            benchmark_60d=Decimal("0.05"),
        )
        result = compute_momentum_signal_with_fallback(inputs)

        self.assertEqual(result.guardrail_flags, [])
        self.assertFalse(result.benchmark_missing)
        self.assertFalse(result.return_clipped)

    def test_missing_prices_preserves_benchmark_flag(self):
        """When all windows missing, should still track benchmark availability."""
        inputs = MultiWindowMomentumInput(
            return_60d=Decimal("0.10"),  # Has return
            benchmark_60d=None,  # No benchmark
            return_20d=Decimal("0.05"),  # Has return
            benchmark_20d=None,  # No benchmark
            return_120d=Decimal("0.15"),  # Has return
            benchmark_120d=None,  # No benchmark
        )
        result = compute_momentum_signal_with_fallback(inputs)

        # All benchmarks missing
        self.assertEqual(result.data_status, "missing_prices")
        self.assertTrue(result.benchmark_missing)
        # Should flag all missing benchmarks
        self.assertIn("benchmark_missing_60d", result.guardrail_flags)


class TestShrinkageEffect(unittest.TestCase):
    """Tests verifying confidence shrinkage works correctly."""

    def test_low_confidence_pulls_toward_neutral(self):
        """Low confidence should pull score toward 50."""
        # High confidence case (conf=0.9)
        high_conf_inputs = MultiWindowMomentumInput(
            return_60d=Decimal("0.30"),  # High alpha -> conf 0.9
            benchmark_60d=Decimal("0.05"),
            trading_days_available=60,
        )
        high_conf_result = compute_momentum_signal_with_fallback(high_conf_inputs)

        # Low confidence case (conf=0.3 due to 20d + few days)
        low_conf_inputs = MultiWindowMomentumInput(
            return_20d=Decimal("0.30"),
            benchmark_20d=Decimal("0.05"),
            trading_days_available=15,  # Very few days -> penalty
        )
        low_conf_result = compute_momentum_signal_with_fallback(low_conf_inputs)

        # Same raw alpha but very different final scores due to shrinkage
        self.assertEqual(high_conf_result.alpha_60d, Decimal("0.2500"))
        self.assertEqual(low_conf_result.alpha_60d, Decimal("0.2500"))

        # High confidence keeps score far from 50
        self.assertGreater(high_conf_result.momentum_score, Decimal("80"))

        # Low confidence pulls score closer to 50
        self.assertLess(low_conf_result.momentum_score, Decimal("70"))

    def test_neutral_alpha_unaffected_by_shrinkage(self):
        """Zero alpha should be 50 regardless of confidence."""
        inputs = MultiWindowMomentumInput(
            return_60d=Decimal("0.05"),
            benchmark_60d=Decimal("0.05"),  # Zero alpha
        )
        result = compute_momentum_signal_with_fallback(inputs)

        # 50 + conf * (50 - 50) = 50 regardless of confidence
        self.assertEqual(result.momentum_score, Decimal("50.00"))


class TestAlphaThresholdDocumentation(unittest.TestCase):
    """
    Tests that enforce the documented alpha-to-score mapping.

    These tests ensure the documented values in module_5_composite_v3.py
    (alpha anchoring comments) stay in sync with the actual mapping function.

    If the mapping changes, these tests will fail, prompting an update to
    the documentation.
    """

    def test_strong_threshold_constant(self):
        """Verify MOMENTUM_STRONG_THRESHOLD is 2.5."""
        self.assertEqual(MOMENTUM_STRONG_THRESHOLD, Decimal("2.5"))

    def test_alpha_for_strong_signal_conf_0_7(self):
        """With conf=0.7 (typical), strong signal requires |alpha| >= ~2.38%.

        Documented: "With conf=0.7 (typical): |score-50| >= 2.5 requires |alpha| >= ~2.4%"
        """
        alpha_threshold = compute_alpha_for_score_delta(
            score_delta=MOMENTUM_STRONG_THRESHOLD,
            confidence=Decimal("0.7"),
        )
        # 2.5 / (0.7 * 150) = 2.5 / 105 = 0.02381...
        # Allow small tolerance for rounding
        self.assertGreaterEqual(alpha_threshold, Decimal("0.0235"))
        self.assertLessEqual(alpha_threshold, Decimal("0.0240"))

    def test_alpha_for_strong_signal_conf_0_9(self):
        """With conf=0.9 (high), strong signal requires |alpha| >= ~1.85%.

        Documented: "With conf=0.9 (high): |score-50| >= 2.5 requires |alpha| >= ~1.85%"
        """
        alpha_threshold = compute_alpha_for_score_delta(
            score_delta=MOMENTUM_STRONG_THRESHOLD,
            confidence=Decimal("0.9"),
        )
        # 2.5 / (0.9 * 150) = 2.5 / 135 = 0.01852...
        self.assertGreaterEqual(alpha_threshold, Decimal("0.0183"))
        self.assertLessEqual(alpha_threshold, Decimal("0.0188"))

    def test_alpha_for_strong_signal_conf_1_0(self):
        """With conf=1.0 (no shrinkage), strong signal requires |alpha| >= ~1.67%.

        Documented: "Raw (no shrinkage): |score-50| >= 2.5 requires |alpha| >= ~1.67%"
        """
        alpha_threshold = compute_alpha_for_score_delta(
            score_delta=MOMENTUM_STRONG_THRESHOLD,
            confidence=Decimal("1.0"),
        )
        # 2.5 / (1.0 * 150) = 2.5 / 150 = 0.01667...
        self.assertGreaterEqual(alpha_threshold, Decimal("0.0165"))
        self.assertLessEqual(alpha_threshold, Decimal("0.0168"))

    def test_alpha_scales_with_confidence(self):
        """Lower confidence requires higher alpha to achieve same score delta."""
        alpha_high_conf = compute_alpha_for_score_delta(
            score_delta=MOMENTUM_STRONG_THRESHOLD,
            confidence=Decimal("0.9"),
        )
        alpha_low_conf = compute_alpha_for_score_delta(
            score_delta=MOMENTUM_STRONG_THRESHOLD,
            confidence=Decimal("0.5"),
        )
        # Lower confidence requires MORE alpha
        self.assertGreater(alpha_low_conf, alpha_high_conf)

    def test_round_trip_score_calculation(self):
        """Verify alpha -> score -> alpha round-trip consistency.

        Given: alpha = 0.0238 (2.38%), conf = 0.7
        Compute: score = 50 + 0.7 * (50 + 0.0238 * 150 - 50) = 50 + 0.7 * 3.57 = 52.5
        So a score of 52.5 should require ~2.38% alpha.
        """
        # Compute expected alpha for score_delta = 2.5
        expected_alpha = compute_alpha_for_score_delta(
            score_delta=Decimal("2.5"),
            confidence=Decimal("0.7"),
        )

        # Now verify: with this alpha, we get back to score 52.5
        # raw_score = 50 + alpha * 150
        raw_score = Decimal("50") + expected_alpha * MOMENTUM_SLOPE
        # shrunk_score = 50 + conf * (raw - 50)
        shrunk_score = Decimal("50") + Decimal("0.7") * (raw_score - Decimal("50"))

        # Should be close to 52.5 (within precision)
        self.assertAlmostEqual(float(shrunk_score), 52.5, places=1)


if __name__ == "__main__":
    unittest.main()
