"""Tests for Fama-MacBeth cross-sectional regression."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.fmb import (
    cross_sectional_regression,
    fama_macbeth,
    newey_west_se,
    zscore_features,
)


class TestCrossSectionalRegression:
    """Unit tests for single-date OLS."""

    def test_perfect_linear(self):
        """y = 2*x + noise=0 → beta ≈ 2."""
        np.random.seed(42)
        n = 50
        x = np.linspace(0, 1, n)
        y = 2.0 * x + 0.5  # intercept=0.5, beta=2
        features = x.reshape(-1, 1)
        betas = cross_sectional_regression(y, features, add_intercept=True)
        assert abs(betas[0] - 2.0) < 1e-6

    def test_too_few_obs(self):
        """Fewer obs than params → NaN."""
        y = np.array([1.0, 2.0])
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        betas = cross_sectional_regression(y, X, add_intercept=True)
        assert np.all(np.isnan(betas))

    def test_multiple_features(self):
        """Two features, known betas."""
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        y = 1.5 * x1 - 0.5 * x2 + 0.1  # intercept=0.1
        features = np.column_stack([x1, x2])
        betas = cross_sectional_regression(y, features, add_intercept=True)
        assert abs(betas[0] - 1.5) < 0.01
        assert abs(betas[1] - (-0.5)) < 0.01


class TestNeweyWestSE:
    """Test Newey-West standard error computation."""

    def test_iid_series(self):
        """For IID series, NW SE ≈ classical SE."""
        np.random.seed(42)
        series = np.random.randn(100)
        nw_se = newey_west_se(series, lags=0)
        classical_se = np.std(series, ddof=1) / np.sqrt(100)
        assert abs(nw_se - classical_se) < 0.02

    def test_positive_autocorr(self):
        """With positive autocorrelation, NW SE > classical SE."""
        np.random.seed(42)
        T = 200
        e = np.random.randn(T)
        series = np.zeros(T)
        series[0] = e[0]
        for t in range(1, T):
            series[t] = 0.8 * series[t - 1] + e[t]  # AR(1), rho=0.8
        nw_se = newey_west_se(series, lags=10)
        classical_se = np.std(series, ddof=1) / np.sqrt(T)
        assert nw_se > classical_se

    def test_short_series(self):
        """Fewer than 3 obs → NaN."""
        assert np.isnan(newey_west_se(np.array([1.0, 2.0]), lags=1))


class TestZscoreFeatures:
    """Test cross-sectional z-scoring."""

    def test_zero_mean_unit_var(self):
        """After z-scoring, each date should have mean≈0 std≈1."""
        df = pd.DataFrame({
            "date": ["2025-01-03"] * 5 + ["2025-01-10"] * 5,
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0],
        })
        result = zscore_features(df, ["x"], date_col="date")
        for dt in ["2025-01-03", "2025-01-10"]:
            vals = result[result["date"] == dt]["x"]
            assert abs(vals.mean()) < 1e-10
            assert abs(vals.std(ddof=1) - 1.0) < 1e-10


class TestFamaMacBeth:
    """End-to-end FMB tests on synthetic data."""

    def _make_panel(self, n_dates=20, n_stocks=30, true_beta=0.5, seed=42):
        """
        Synthetic panel where feature x predicts returns with known beta.
        """
        np.random.seed(seed)
        rows = []
        for d in range(n_dates):
            dt = f"2025-{d+1:04d}"  # Fake dates
            x = np.random.randn(n_stocks)
            ret = true_beta * x + np.random.randn(n_stocks) * 0.5
            for i in range(n_stocks):
                rows.append({
                    "rebalance_date": dt,
                    "ticker": f"T{i:03d}",
                    "score_x": x[i],
                    "fwd_return": ret[i],
                })
        return pd.DataFrame(rows)

    def test_positive_signal_detected(self):
        """Feature with true beta=0.5 should produce mean_beta > 0 and |t| > 2."""
        panel = self._make_panel(n_dates=50, n_stocks=50, true_beta=0.5)
        summary, betas = fama_macbeth(
            panel, "fwd_return", ["score_x"],
            standardize=True, nw_lags=4,
        )
        assert len(summary) == 1
        row = summary.iloc[0]
        assert row["mean_beta"] > 0
        assert abs(row["t_stat"]) > 2.0

    def test_zero_signal(self):
        """Feature with true beta=0 should produce |t| < 2 (usually)."""
        panel = self._make_panel(n_dates=50, n_stocks=50, true_beta=0.0)
        summary, betas = fama_macbeth(
            panel, "fwd_return", ["score_x"],
            standardize=True, nw_lags=4,
        )
        # With beta=0, t-stat should be small most of the time
        assert abs(summary.iloc[0]["t_stat"]) < 4.0  # Generous bound

    def test_deterministic(self):
        """Same inputs produce identical outputs."""
        panel = self._make_panel()
        s1, b1 = fama_macbeth(panel, "fwd_return", ["score_x"])
        s2, b2 = fama_macbeth(panel, "fwd_return", ["score_x"])
        pd.testing.assert_frame_equal(s1, s2)
        pd.testing.assert_frame_equal(b1, b2)

    def test_multiple_features(self):
        """Two features: one predictive, one noise."""
        np.random.seed(42)
        rows = []
        for d in range(30):
            dt = f"2025-{d+1:04d}"
            n = 40
            x1 = np.random.randn(n)
            x2 = np.random.randn(n)
            ret = 0.8 * x1 + np.random.randn(n) * 0.3  # x2 has no signal
            for i in range(n):
                rows.append({
                    "rebalance_date": dt,
                    "ticker": f"T{i:03d}",
                    "signal": x1[i],
                    "noise": x2[i],
                    "fwd_return": ret[i],
                })
        panel = pd.DataFrame(rows)
        summary, _ = fama_macbeth(
            panel, "fwd_return", ["noise", "signal"],
            standardize=True, nw_lags=4,
        )
        signal_row = summary[summary["feature"] == "signal"].iloc[0]
        noise_row = summary[summary["feature"] == "noise"].iloc[0]
        # Signal should be highly significant
        assert abs(signal_row["t_stat"]) > abs(noise_row["t_stat"])
        assert abs(signal_row["t_stat"]) > 3.0

    def test_empty_panel(self):
        """Empty panel returns empty summary."""
        panel = pd.DataFrame(columns=["rebalance_date", "ticker", "x", "ret"])
        summary, betas = fama_macbeth(panel, "ret", ["x"])
        assert summary.empty
