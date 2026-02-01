"""Tests for Milestone 1 IC metrics."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.metrics_m1 import (
    spearman_rank_ic,
    compute_ic_timeseries,
    summarize_ic,
)


class TestSpearmanRankIC:
    """Unit tests for Spearman rank IC calculation."""

    def test_perfect_positive(self):
        """Score perfectly predicts return ordering → IC = +1."""
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        returns = [0.01, 0.02, 0.03, 0.04, 0.05]
        ic = spearman_rank_ic(scores, returns)
        assert abs(ic - 1.0) < 1e-9

    def test_perfect_negative(self):
        """Score is perfectly anti-correlated → IC = -1."""
        scores = [5.0, 4.0, 3.0, 2.0, 1.0]
        returns = [0.01, 0.02, 0.03, 0.04, 0.05]
        ic = spearman_rank_ic(scores, returns)
        assert abs(ic - (-1.0)) < 1e-9

    def test_no_correlation(self):
        """Uncorrelated data → IC near 0."""
        # Specific permutation with known zero Spearman
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        returns = [3.0, 1.0, 5.0, 2.0, 4.0]
        ic = spearman_rank_ic(scores, returns)
        assert abs(ic) < 0.5  # Not exactly 0 but small

    def test_too_few_points(self):
        """Fewer than 3 valid pairs → NaN."""
        ic = spearman_rank_ic([1.0, 2.0], [0.1, 0.2])
        assert math.isnan(ic)

    def test_nan_filtering(self):
        """NaN values in input are excluded."""
        scores = [1.0, float("nan"), 3.0, 4.0, 5.0]
        returns = [0.01, 0.02, 0.03, 0.04, 0.05]
        ic = spearman_rank_ic(scores, returns)
        assert math.isfinite(ic)

    def test_all_nan_returns_nan(self):
        scores = [float("nan")] * 5
        returns = [0.01, 0.02, 0.03, 0.04, 0.05]
        ic = spearman_rank_ic(scores, returns)
        assert math.isnan(ic)

    def test_tied_values(self):
        """Ties in scores are handled (average rank)."""
        scores = [1.0, 1.0, 3.0, 4.0, 5.0]
        returns = [0.01, 0.02, 0.03, 0.04, 0.05]
        ic = spearman_rank_ic(scores, returns)
        assert math.isfinite(ic)
        assert ic > 0.8  # Still strongly positive


class TestICTimeseries:
    """Test compute_ic_timeseries and summarize_ic on synthetic data."""

    def _make_panel(self):
        """2 dates × 5 tickers, perfect positive correlation."""
        rows = []
        for dt in ["2025-01-03", "2025-01-10"]:
            for i, ticker in enumerate(["A", "B", "C", "D", "E"]):
                rows.append({
                    "rebalance_date": dt,
                    "ticker": ticker,
                    "score_x": float(i + 1),
                    "fwd_21d": float(i + 1) * 0.01,
                })
        return rows

    def test_ic_timeseries_perfect(self):
        rows = self._make_panel()
        ic_ts = compute_ic_timeseries(
            rows, ["score_x"], ["fwd_21d"], date_col="rebalance_date"
        )
        assert len(ic_ts) == 2  # 2 dates
        for entry in ic_ts:
            assert abs(entry["ic"] - 1.0) < 1e-9
            assert entry["n_obs"] == 5

    def test_summarize_ic(self):
        rows = self._make_panel()
        ic_ts = compute_ic_timeseries(
            rows, ["score_x"], ["fwd_21d"], date_col="rebalance_date"
        )
        summary = summarize_ic(ic_ts)
        assert len(summary) == 1
        s = summary[0]
        assert s["score_col"] == "score_x"
        assert s["return_col"] == "fwd_21d"
        assert abs(s["mean_ic"] - 1.0) < 1e-6
        assert s["hit_rate"] == 1.0
        assert s["n_dates"] == 2

    def test_negative_ic_panel(self):
        """Reversed scores → IC = -1."""
        rows = []
        for dt in ["2025-01-03", "2025-01-10"]:
            for i, ticker in enumerate(["A", "B", "C", "D", "E"]):
                rows.append({
                    "rebalance_date": dt,
                    "ticker": ticker,
                    "score_x": float(5 - i),  # reversed
                    "fwd_21d": float(i + 1) * 0.01,
                })
        ic_ts = compute_ic_timeseries(
            rows, ["score_x"], ["fwd_21d"], date_col="rebalance_date"
        )
        summary = summarize_ic(ic_ts)
        assert abs(summary[0]["mean_ic"] - (-1.0)) < 1e-6
        assert summary[0]["hit_rate"] == 0.0

    def test_multiple_scores_and_horizons(self):
        """Multiple score cols × return cols produce correct cartesian."""
        rows = []
        for dt in ["2025-01-03"]:
            for i, ticker in enumerate(["A", "B", "C", "D", "E"]):
                rows.append({
                    "rebalance_date": dt,
                    "ticker": ticker,
                    "score_a": float(i),
                    "score_b": float(4 - i),
                    "fwd_21d": float(i) * 0.01,
                    "fwd_63d": float(i) * 0.02,
                })
        ic_ts = compute_ic_timeseries(
            rows, ["score_a", "score_b"], ["fwd_21d", "fwd_63d"]
        )
        # 1 date × 2 scores × 2 horizons = 4 rows
        assert len(ic_ts) == 4
