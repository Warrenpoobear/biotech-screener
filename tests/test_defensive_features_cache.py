#!/usr/bin/env python3
"""Unit tests for defensive features cache builder."""

import math

from wake_robin_data_pipeline.defensive_features_cache import (
    compute_returns,
    vol,
    build_cache,
)


class TestDefensiveFeaturesCache:
    """Tests for cache builder core functions."""

    def test_compute_returns_log_returns(self):
        """Log returns computed correctly."""
        closes = [100.0, 110.0, 105.0]
        returns = compute_returns(closes)
        assert len(returns) == 2
        assert abs(returns[0] - math.log(110 / 100)) < 1e-9
        assert abs(returns[1] - math.log(105 / 110)) < 1e-9

    def test_vol_annualized(self):
        """Volatility annualized with sqrt(252)."""
        # Constant returns = zero vol
        returns = [0.01] * 60
        v = vol(returns, 60)
        assert v == 0.0  # No variance in constant series

        # Non-constant returns
        returns = [0.01, -0.01] * 30
        v = vol(returns, 60)
        assert v is not None
        assert v > 0  # Should have positive vol
