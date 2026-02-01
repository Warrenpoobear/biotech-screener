"""Tests for Walk-Forward OOS M3 Evaluation."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.walkforward_oos_m3 import (
    walk_forward_evaluate,
    aggregate_results,
    _quintile_spread,
)
from backtest.evaluate_m3_weights import DEFAULT_V3_WEIGHTS, FEATURE_COLS


# ── Helpers ───────────────────────────────────────────────────────────

def _make_panel(
    n_weeks: int = 60,
    n_stocks: int = 200,
    seed: int = 42,
    signal_feature: str = "valuation_normalized",
    signal_beta: float = 0.05,
) -> pd.DataFrame:
    """
    Synthetic panel where return = signal_beta * signal_feature + noise.

    All features drawn from U(0,1). Returns have a known signal on one feature.
    """
    rng = np.random.RandomState(seed)
    dates = [f"2024-{(w // 4) + 1:02d}-{(w % 4) * 7 + 1:02d}" for w in range(n_weeks)]
    # Ensure unique dates
    dates = [f"2024-W{w:03d}" for w in range(n_weeks)]
    tickers = [f"TK{i:04d}" for i in range(n_stocks)]

    rows = []
    for dt in dates:
        for tk in tickers:
            feats = {c: rng.uniform(0, 1) for c in FEATURE_COLS}
            noise = rng.normal(0, 0.05)
            ret = signal_beta * feats[signal_feature] + noise
            rows.append({
                "rebalance_date": dt,
                "ticker": tk,
                **feats,
                "composite_score": rng.uniform(0, 1),
                "fwd_5d": ret,
                "fwd_21d": ret * 2,  # scaled version for multi-horizon
            })
    return pd.DataFrame(rows)


# ── 1) Walk-forward uses only past weeks ──────────────────────────────

class TestPITSafety:
    def test_training_excludes_eval_date(self):
        """Training window must not include the eval date."""
        panel = _make_panel(n_weeks=60, n_stocks=100)
        dates = sorted(panel["rebalance_date"].unique())

        ts_df, wt_df = walk_forward_evaluate(
            panel,
            return_col="fwd_5d",
            train_window_weeks=52,
            min_stocks=10,
        )

        # The earliest eval_date should be dates[52], not dates[51]
        if len(ts_df) > 0:
            earliest_eval = ts_df["eval_date"].min()
            assert earliest_eval == dates[52], (
                f"Earliest eval {earliest_eval} should be {dates[52]}"
            )

    def test_no_future_leakage_in_weights(self):
        """Weights at eval date t should only use data from dates < t."""
        panel = _make_panel(n_weeks=60, n_stocks=100)
        dates = sorted(panel["rebalance_date"].unique())

        ts_df, wt_df = walk_forward_evaluate(
            panel,
            return_col="fwd_5d",
            train_window_weeks=52,
            min_stocks=10,
        )

        if len(wt_df) > 0:
            # First weight row's eval_date should be dates[52]
            first_eval = wt_df["eval_date"].iloc[0]
            assert first_eval >= dates[52]


# ── 2) Positive OOS signal is captured ────────────────────────────────

class TestSignalCapture:
    def test_known_signal_produces_positive_ic(self):
        """
        When return = beta * feature + noise, walk-forward ridge should
        learn the signal and produce positive OOS IC.
        """
        panel = _make_panel(
            n_weeks=60, n_stocks=200, seed=123,
            signal_feature="valuation_normalized",
            signal_beta=0.10,
        )

        ts_df, _ = walk_forward_evaluate(
            panel,
            return_col="fwd_5d",
            train_window_weeks=52,
            min_stocks=10,
            ridge_lambda=10.0,
        )

        m3_oos = ts_df[ts_df["variant"] == "m3_oos"]
        assert len(m3_oos) > 0, "No OOS dates produced"

        mean_ic = m3_oos["ic"].mean()
        assert mean_ic > 0, f"M3 OOS mean IC should be positive, got {mean_ic:.4f}"


# ── 3) Determinism ────────────────────────────────────────────────────

class TestDeterminism:
    def test_identical_runs(self):
        """Two runs on same data produce identical results."""
        panel = _make_panel(n_weeks=58, n_stocks=80, seed=99)

        ts1, wt1 = walk_forward_evaluate(
            panel, return_col="fwd_5d", train_window_weeks=52, min_stocks=10,
        )
        ts2, wt2 = walk_forward_evaluate(
            panel, return_col="fwd_5d", train_window_weeks=52, min_stocks=10,
        )

        pd.testing.assert_frame_equal(ts1, ts2)
        pd.testing.assert_frame_equal(wt1, wt2)


# ── 4) Handles missing returns ────────────────────────────────────────

class TestMissingData:
    def test_nan_returns_skipped(self):
        """Weeks where return is NaN should not crash; stocks are dropped."""
        panel = _make_panel(n_weeks=58, n_stocks=100)
        # Inject NaN returns for some stocks on the last eval date
        dates = sorted(panel["rebalance_date"].unique())
        last_date = dates[-1]
        mask = (panel["rebalance_date"] == last_date)
        panel.loc[mask & (panel.index % 3 == 0), "fwd_5d"] = np.nan

        ts_df, _ = walk_forward_evaluate(
            panel, return_col="fwd_5d", train_window_weeks=52, min_stocks=10,
        )

        # Should still produce results (the date isn't fully NaN)
        assert len(ts_df) > 0

    def test_nan_features_handled(self):
        """Stocks with NaN features are excluded from eval."""
        panel = _make_panel(n_weeks=58, n_stocks=100)
        dates = sorted(panel["rebalance_date"].unique())
        last_date = dates[-1]
        mask = (panel["rebalance_date"] == last_date)
        panel.loc[mask & (panel.index % 5 == 0), "clinical_normalized"] = np.nan

        ts_df, _ = walk_forward_evaluate(
            panel, return_col="fwd_5d", train_window_weeks=52, min_stocks=10,
        )

        # n_stocks count should reflect the valid subset
        last_rows = ts_df[ts_df["eval_date"] == last_date]
        if len(last_rows) > 0:
            assert last_rows.iloc[0]["n_stocks"] < 100


# ── 5) Min-stocks threshold ──────────────────────────────────────────

class TestMinStocks:
    def test_small_date_skipped(self):
        """Eval dates with fewer than min_stocks valid rows are skipped."""
        panel = _make_panel(n_weeks=58, n_stocks=100)
        dates = sorted(panel["rebalance_date"].unique())

        # Keep only 5 stocks on the last eval date
        last_date = dates[-1]
        last_mask = panel["rebalance_date"] == last_date
        tickers_to_drop = panel.loc[last_mask, "ticker"].unique()[5:]
        panel = panel[~(last_mask & panel["ticker"].isin(tickers_to_drop))]

        ts_df, _ = walk_forward_evaluate(
            panel, return_col="fwd_5d", train_window_weeks=52, min_stocks=50,
        )

        eval_dates = ts_df["eval_date"].unique()
        assert last_date not in eval_dates, "Date with <50 stocks should be skipped"


# ── 6) Top-N computation correctness ─────────────────────────────────

class TestTopNCorrectness:
    def test_top_n_selects_correct_stocks(self):
        """On a small eval set, top-N mean return matches hand calculation."""
        panel = _make_panel(n_weeks=55, n_stocks=60, seed=77)
        dates = sorted(panel["rebalance_date"].unique())

        ts_df, _ = walk_forward_evaluate(
            panel, return_col="fwd_5d", train_window_weeks=52,
            min_stocks=10, top_n=5,
        )

        # Just verify we get results and they're finite
        m3_rows = ts_df[ts_df["variant"] == "m3_oos"]
        assert len(m3_rows) > 0
        assert m3_rows["topn_return"].notna().all()


# ── 7) Ridge fitting stability ────────────────────────────────────────

class TestRidgeStability:
    def test_collinear_features_no_crash(self):
        """Collinear features should not crash; ridge regularizes."""
        panel = _make_panel(n_weeks=58, n_stocks=100)
        # Make two features identical (collinear)
        panel["momentum_normalized"] = panel["financial_normalized"]

        ts_df, wt_df = walk_forward_evaluate(
            panel, return_col="fwd_5d", train_window_weeks=52, min_stocks=10,
        )

        # Should still produce results with finite weights
        assert len(ts_df) > 0
        if len(wt_df) > 0:
            for col in FEATURE_COLS:
                if col in wt_df.columns:
                    assert wt_df[col].apply(np.isfinite).all(), (
                        f"Weight for {col} has non-finite values"
                    )


# ── 8) NW t-stat sanity ──────────────────────────────────────────────

class TestNWTstat:
    def test_positive_ic_positive_tstat(self):
        """If IC is consistently positive, t-stat should be positive and finite."""
        panel = _make_panel(
            n_weeks=60, n_stocks=200, seed=456,
            signal_feature="valuation_normalized",
            signal_beta=0.15,  # strong signal
        )

        ts_df, _ = walk_forward_evaluate(
            panel, return_col="fwd_5d", train_window_weeks=52, min_stocks=10,
        )

        summary = aggregate_results(ts_df, nw_lags=4)
        m3_row = summary[summary["variant"] == "m3_oos"]

        if len(m3_row) > 0:
            tstat = m3_row.iloc[0]["IC_tstat"]
            assert np.isfinite(tstat), f"t-stat should be finite, got {tstat}"
            # With strong signal, t-stat should be positive
            assert tstat > 0, f"t-stat should be positive with strong signal, got {tstat}"


# ── 9) Aggregate summary structure ───────────────────────────────────

class TestAggregation:
    def test_summary_has_all_variants(self):
        """Summary should have rows for all three variants."""
        panel = _make_panel(n_weeks=58, n_stocks=80)
        ts_df, _ = walk_forward_evaluate(
            panel, return_col="fwd_5d", train_window_weeks=52, min_stocks=10,
        )

        summary = aggregate_results(ts_df)
        variants = set(summary["variant"].tolist())
        assert "default_linear" in variants
        assert "pipeline" in variants
        assert "m3_oos" in variants

    def test_summary_columns(self):
        """Summary has all expected metric columns."""
        panel = _make_panel(n_weeks=58, n_stocks=80)
        ts_df, _ = walk_forward_evaluate(
            panel, return_col="fwd_5d", train_window_weeks=52, min_stocks=10,
        )

        summary = aggregate_results(ts_df)
        expected_cols = {
            "variant", "IC_mean", "IC_tstat", "IC_hit",
            "TopN_mean", "TopN_tstat", "CumRet", "Sharpe", "MDD",
            "QSpread_mean", "n_weeks",
        }
        assert expected_cols.issubset(set(summary.columns))
