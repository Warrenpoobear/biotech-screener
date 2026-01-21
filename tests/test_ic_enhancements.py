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
    MomentumSignal,
    MOMENTUM_SLOPE,
    MOMENTUM_SCORE_MIN,
    MOMENTUM_SCORE_MAX,
    VOLATILITY_BASELINE,
)


class TestMomentumSignalV2(unittest.TestCase):
    """Tests for improved momentum signal calculation."""

    def test_basic_positive_alpha(self):
        """Positive 10% alpha should produce score of 65 (not 70 as in v1)."""
        result = compute_momentum_signal(
            return_60d=Decimal("0.15"),
            benchmark_return_60d=Decimal("0.05"),
        )
        self.assertEqual(result.alpha_60d, Decimal("0.1000"))
        # 50 + 0.10 * 150 = 65
        self.assertEqual(result.momentum_score, Decimal("65.00"))
        self.assertEqual(result.data_completeness, Decimal("1.0"))

    def test_basic_negative_alpha(self):
        """Negative 10% alpha should produce score of 35 (not 30 as in v1)."""
        result = compute_momentum_signal(
            return_60d=Decimal("-0.05"),
            benchmark_return_60d=Decimal("0.05"),
        )
        self.assertEqual(result.alpha_60d, Decimal("-0.1000"))
        # 50 - 0.10 * 150 = 35
        self.assertEqual(result.momentum_score, Decimal("35.00"))

    def test_neutral_alpha(self):
        """Zero alpha should produce neutral score of 50."""
        result = compute_momentum_signal(
            return_60d=Decimal("0.05"),
            benchmark_return_60d=Decimal("0.05"),
        )
        self.assertEqual(result.alpha_60d, Decimal("0.0000"))
        self.assertEqual(result.momentum_score, Decimal("50.00"))

    def test_clamp_upper_bound(self):
        """Large positive alpha should be capped at 95 (not 90 as in v1)."""
        result = compute_momentum_signal(
            return_60d=Decimal("0.40"),
            benchmark_return_60d=Decimal("0.05"),
        )
        # alpha = 0.35, 50 + 0.35 * 150 = 102.5 -> capped at 95
        self.assertEqual(result.momentum_score, MOMENTUM_SCORE_MAX)

    def test_clamp_lower_bound(self):
        """Large negative alpha should be capped at 5 (not 10 as in v1)."""
        result = compute_momentum_signal(
            return_60d=Decimal("-0.35"),
            benchmark_return_60d=Decimal("0.05"),
        )
        # alpha = -0.40, 50 - 0.40 * 150 = -10 -> capped at 5
        self.assertEqual(result.momentum_score, MOMENTUM_SCORE_MIN)

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
        """At baseline vol (50%), vol-adjusted score should match raw alpha."""
        result = compute_momentum_signal(
            return_60d=Decimal("0.15"),
            benchmark_return_60d=Decimal("0.05"),
            annualized_vol=VOLATILITY_BASELINE,
            use_vol_adjusted_alpha=True,
        )
        # Vol-adjusted alpha = 0.10 / 0.50 = 0.20
        # Scoring alpha = 0.20 * 0.50 (baseline) = 0.10
        # Score = 50 + 0.10 * 150 = 65
        self.assertEqual(result.momentum_score, Decimal("65.00"))

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

        # Score should be above neutral (positive alpha)
        self.assertGreater(result.momentum_score, Decimal("50"))

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


if __name__ == "__main__":
    unittest.main()
