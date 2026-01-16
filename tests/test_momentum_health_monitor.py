#!/usr/bin/env python3
"""
test_momentum_health_monitor.py - Tests for momentum health monitoring

Tests the IC-based momentum health monitoring and kill switch functionality.
"""

import unittest
from decimal import Decimal
from datetime import date

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from momentum_health_monitor import (
    MomentumHealthMonitor,
    spearman_rank_correlation,
    calculate_momentum_signal,
    calculate_cross_sectional_ic,
    get_regime_adaptive_momentum_weight,
    assess_momentum_signal_health,
    IC_EXCELLENT,
    IC_GOOD,
    IC_MARGINAL,
    IC_WEAK,
    WEIGHT_FULL,
    WEIGHT_REDUCED,
    WEIGHT_MINIMAL,
    WEIGHT_DISABLED,
)
from regime_engine import RegimeDetectionEngine


class TestSpearmanCorrelation(unittest.TestCase):
    """Tests for Spearman rank correlation calculation."""

    def test_perfect_positive_correlation(self):
        """Perfect positive rank correlation should return 1.0."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 20.0, 30.0, 40.0, 50.0]
        corr = spearman_rank_correlation(x, y)
        self.assertIsNotNone(corr)
        self.assertAlmostEqual(corr, 1.0, places=5)

    def test_perfect_negative_correlation(self):
        """Perfect negative rank correlation should return -1.0."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [50.0, 40.0, 30.0, 20.0, 10.0]
        corr = spearman_rank_correlation(x, y)
        self.assertIsNotNone(corr)
        self.assertAlmostEqual(corr, -1.0, places=5)

    def test_weak_correlation(self):
        """Weak/no correlation for scrambled series."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        y = [4.0, 8.0, 1.0, 5.0, 2.0, 6.0, 3.0, 7.0]  # Scrambled
        corr = spearman_rank_correlation(x, y)
        self.assertIsNotNone(corr)
        # Should not be strongly correlated (may have some correlation due to random chance)
        self.assertLess(abs(corr), 0.8)

    def test_insufficient_data(self):
        """Should return None with insufficient data."""
        x = [1.0, 2.0]
        y = [3.0, 4.0]
        corr = spearman_rank_correlation(x, y)
        self.assertIsNone(corr)

    def test_mismatched_length(self):
        """Should return None with mismatched lengths."""
        x = [1.0, 2.0, 3.0]
        y = [1.0, 2.0]
        corr = spearman_rank_correlation(x, y)
        self.assertIsNone(corr)


class TestMomentumSignal(unittest.TestCase):
    """Tests for momentum signal calculation."""

    def test_positive_momentum(self):
        """Uptrend should produce positive momentum."""
        prices = [100.0] * 63 + [115.0]  # 15% gain
        mom = calculate_momentum_signal(prices, lookback=63)
        self.assertIsNotNone(mom)
        self.assertAlmostEqual(mom, 0.15, places=4)

    def test_negative_momentum(self):
        """Downtrend should produce negative momentum."""
        prices = [100.0] * 63 + [85.0]  # 15% loss
        mom = calculate_momentum_signal(prices, lookback=63)
        self.assertIsNotNone(mom)
        self.assertAlmostEqual(mom, -0.15, places=4)

    def test_insufficient_data(self):
        """Should return None with insufficient data."""
        prices = [100.0] * 30  # Less than 64 needed for 63-day lookback
        mom = calculate_momentum_signal(prices, lookback=63)
        self.assertIsNone(mom)


class TestMomentumHealthMonitor(unittest.TestCase):
    """Tests for MomentumHealthMonitor class."""

    def setUp(self):
        self.monitor = MomentumHealthMonitor()

    def test_initial_state(self):
        """New monitor should have no data."""
        self.assertIsNone(self.monitor.current_ic)
        self.assertEqual(self.monitor.health_status, "UNKNOWN")
        self.assertEqual(len(self.monitor.ic_history), 0)

    def test_update_ic_excellent(self):
        """IC >= 0.15 should be EXCELLENT."""
        result = self.monitor.update_ic(0.18)
        self.assertEqual(result["health_status"], "EXCELLENT")

    def test_update_ic_good(self):
        """IC >= 0.10 should be GOOD."""
        result = self.monitor.update_ic(0.12)
        self.assertEqual(result["health_status"], "GOOD")

    def test_update_ic_marginal(self):
        """IC >= 0.05 should be MARGINAL."""
        result = self.monitor.update_ic(0.07)
        self.assertEqual(result["health_status"], "MARGINAL")

    def test_update_ic_weak(self):
        """IC >= 0.00 should be WEAK."""
        result = self.monitor.update_ic(0.02)
        self.assertEqual(result["health_status"], "WEAK")

    def test_update_ic_inverted(self):
        """IC < 0.00 should be INVERTED."""
        result = self.monitor.update_ic(-0.03)
        self.assertEqual(result["health_status"], "INVERTED")

    def test_update_ic_strongly_inverted(self):
        """IC < -0.05 should be STRONGLY_INVERTED."""
        result = self.monitor.update_ic(-0.10)
        self.assertEqual(result["health_status"], "STRONGLY_INVERTED")

    def test_rolling_average(self):
        """Rolling IC average should be calculated correctly."""
        self.monitor.update_ic(0.10)
        self.monitor.update_ic(0.20)
        self.monitor.update_ic(0.15)
        avg = self.monitor.get_rolling_ic_average()
        self.assertAlmostEqual(float(avg), 0.15, places=4)

    def test_history_limit(self):
        """IC history should be limited to configured length."""
        for i in range(15):
            self.monitor.update_ic(i * 0.01)
        self.assertEqual(len(self.monitor.ic_history), 10)  # Default limit


class TestMomentumKillSwitch(unittest.TestCase):
    """Tests for momentum kill switch behavior."""

    def setUp(self):
        self.monitor = MomentumHealthMonitor()

    def test_kill_switch_disabled_for_negative_ic(self):
        """Negative IC should return 0 weight (disabled)."""
        weight = self.monitor.check_momentum_health(-0.10)
        self.assertEqual(weight, WEIGHT_DISABLED)

    def test_kill_switch_minimal_for_weak_ic(self):
        """IC < 0.05 should return minimal weight."""
        weight = self.monitor.check_momentum_health(0.03)
        self.assertEqual(weight, WEIGHT_MINIMAL)

    def test_kill_switch_full_for_good_ic(self):
        """IC >= 0.05 should return full weight."""
        weight = self.monitor.check_momentum_health(0.10)
        self.assertEqual(weight, WEIGHT_FULL)

    def test_get_momentum_weight_with_ic(self):
        """get_momentum_weight should apply IC adjustment."""
        # Update with negative IC
        self.monitor.update_ic(-0.05)
        self.monitor.update_ic(-0.08)
        self.monitor.update_ic(-0.10)

        # Should return 0 regardless of regime
        weight_bull = self.monitor.get_momentum_weight("BULL")
        weight_bear = self.monitor.get_momentum_weight("BEAR")

        self.assertEqual(weight_bull, WEIGHT_DISABLED)
        self.assertEqual(weight_bear, WEIGHT_DISABLED)

    def test_get_momentum_weight_healthy_ic(self):
        """Healthy IC should return regime-specific weight."""
        # Update with good IC
        self.monitor.update_ic(0.15)
        self.monitor.update_ic(0.12)
        self.monitor.update_ic(0.14)

        # Should return regime-specific weight
        weight_bull = self.monitor.get_momentum_weight("BULL")
        weight_bear = self.monitor.get_momentum_weight("BEAR")

        self.assertEqual(weight_bull, Decimal("1.20"))  # BULL boost
        self.assertEqual(weight_bear, Decimal("0.80"))  # BEAR reduce


class TestRegimeEngineIntegration(unittest.TestCase):
    """Tests for regime engine integration with momentum health."""

    def setUp(self):
        self.engine = RegimeDetectionEngine()

    def test_detect_regime_without_ic(self):
        """detect_regime without IC should work normally."""
        result = self.engine.detect_regime(
            vix_current=Decimal("18.0"),
            xbi_vs_spy_30d=Decimal("3.0"),
            fed_rate_change_3m=Decimal("-0.25")
        )
        self.assertIn("regime", result)
        self.assertIn("signal_adjustments", result)
        self.assertIsNone(result.get("momentum_health"))

    def test_detect_regime_with_good_ic(self):
        """detect_regime with good IC should preserve momentum weight."""
        result = self.engine.detect_regime(
            vix_current=Decimal("18.0"),
            xbi_vs_spy_30d=Decimal("3.0"),
            fed_rate_change_3m=Decimal("-0.25"),
            momentum_ic_3m=Decimal("0.15")
        )

        self.assertIsNotNone(result.get("momentum_health"))
        mh = result["momentum_health"]
        self.assertEqual(mh["health_status"], "HEALTHY")
        self.assertEqual(mh["action"], "FULL")

    def test_detect_regime_with_negative_ic(self):
        """detect_regime with negative IC should disable momentum."""
        result = self.engine.detect_regime(
            vix_current=Decimal("18.0"),
            xbi_vs_spy_30d=Decimal("3.0"),
            fed_rate_change_3m=Decimal("-0.25"),
            momentum_ic_3m=Decimal("-0.10")
        )

        self.assertIsNotNone(result.get("momentum_health"))
        mh = result["momentum_health"]
        self.assertEqual(mh["health_status"], "INVERTED")
        self.assertEqual(mh["action"], "DISABLED")
        self.assertEqual(result["signal_adjustments"]["momentum"], Decimal("0"))

    def test_detect_regime_with_marginal_ic(self):
        """detect_regime with marginal IC should reduce momentum."""
        result = self.engine.detect_regime(
            vix_current=Decimal("18.0"),
            xbi_vs_spy_30d=Decimal("3.0"),
            fed_rate_change_3m=Decimal("-0.25"),
            momentum_ic_3m=Decimal("0.07")
        )

        self.assertIsNotNone(result.get("momentum_health"))
        mh = result["momentum_health"]
        self.assertEqual(mh["health_status"], "MARGINAL")
        self.assertEqual(mh["action"], "REDUCED")


class TestConvenienceFunctions(unittest.TestCase):
    """Tests for convenience functions."""

    def test_get_regime_adaptive_momentum_weight(self):
        """get_regime_adaptive_momentum_weight should return correct weight."""
        # Good IC - full regime weight
        weight = get_regime_adaptive_momentum_weight(0.15, "BULL")
        self.assertEqual(weight, Decimal("1.20"))

        # Negative IC - disabled
        weight = get_regime_adaptive_momentum_weight(-0.10, "BULL")
        self.assertEqual(weight, WEIGHT_DISABLED)

        # Very weak IC (<0.05) - minimal
        weight = get_regime_adaptive_momentum_weight(0.02, "BEAR")
        self.assertEqual(weight, WEIGHT_MINIMAL)

        # Marginal IC (0.05 <= IC < 0.10) - reduced
        weight = get_regime_adaptive_momentum_weight(0.07, "BULL")
        self.assertEqual(weight, WEIGHT_REDUCED)

    def test_assess_momentum_signal_health_insufficient_data(self):
        """assess_momentum_signal_health with insufficient data."""
        prices = {"AAPL": [100.0] * 10}  # Too short
        result = assess_momentum_signal_health(prices)
        self.assertIsNone(result["ic"])
        self.assertEqual(result["health_status"], "UNKNOWN")


class TestSerialization(unittest.TestCase):
    """Tests for serialization/deserialization."""

    def test_to_dict_from_dict(self):
        """Monitor should serialize and deserialize correctly."""
        monitor = MomentumHealthMonitor()
        monitor.update_ic(0.10)
        monitor.update_ic(0.12)

        data = monitor.to_dict()
        restored = MomentumHealthMonitor.from_dict(data)

        self.assertEqual(monitor.current_ic, restored.current_ic)
        self.assertEqual(len(monitor.ic_history), len(restored.ic_history))
        self.assertEqual(monitor.health_status, restored.health_status)


if __name__ == "__main__":
    unittest.main()
