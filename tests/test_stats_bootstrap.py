"""Tests for common.stats.bootstrap module."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.stats.bootstrap import block_bootstrap, compare_strategies, stationary_bootstrap


class TestBlockBootstrap:
    def test_positive_series(self):
        """Bootstrap of clearly positive series should have P(>0) near 1."""
        returns = [
            0.02,
            0.03,
            0.01,
            0.04,
            0.02,
            0.03,
            0.01,
            0.05,
            0.02,
            0.03,
            0.01,
            0.04,
            0.02,
            0.03,
            0.01,
            0.05,
            0.02,
            0.03,
            0.01,
            0.04,
            0.02,
            0.03,
            0.01,
            0.05,
        ]
        result = block_bootstrap(returns, block_length=4, n_bootstrap=1000, seed=42)
        assert result["prob_positive"] > 0.99
        assert result["ci_excludes_zero"]

    def test_noisy_series(self):
        """Bootstrap of zero-mean noisy series should NOT confidently exclude zero."""
        np.random.seed(42)
        returns = np.random.randn(48) * 0.05
        result = block_bootstrap(returns, block_length=6, n_bootstrap=1000, seed=42)
        # The noisy series may happen to lean positive or negative;
        # the key test is that CI should be wide and straddle zero
        assert not result["ci_excludes_zero"]

    def test_reproducibility(self):
        """Same seed should produce same results."""
        returns = [0.01, -0.02, 0.03, -0.01, 0.02, 0.04] * 6
        r1 = block_bootstrap(returns, seed=123)
        r2 = block_bootstrap(returns, seed=123)
        assert r1["boot_mean"] == r2["boot_mean"]

    def test_short_series_error(self):
        """Series shorter than block_length should error."""
        result = block_bootstrap([0.01, 0.02], block_length=6)
        assert "error" in result


class TestStationaryBootstrap:
    def test_basic(self):
        """Stationary bootstrap produces valid results."""
        np.random.seed(42)
        returns = np.random.randn(36) * 0.05 + 0.02
        result = stationary_bootstrap(returns, mean_block_length=6, n_bootstrap=1000, seed=42)
        assert "boot_mean" in result
        assert result["n_obs"] == 36

    def test_very_short_error(self):
        result = stationary_bootstrap([1.0, 2.0], mean_block_length=6)
        # T=2 is below the min threshold of 3
        assert "error" in result


class TestCompareStrategies:
    def test_clearly_better(self):
        """When A is clearly better than B, P(A>B) should be high."""
        np.random.seed(42)
        n = 36
        a = np.random.randn(n) * 0.05 + 0.03
        b = np.random.randn(n) * 0.05 + 0.00
        result = compare_strategies(a, b, block_length=4, n_bootstrap=1000, seed=42)
        assert result["prob_a_better"] > 0.85

    def test_length_mismatch(self):
        result = compare_strategies([1, 2, 3], [1, 2])
        assert "error" in result
