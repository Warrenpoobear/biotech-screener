"""
Walk-Forward Out-of-Sample M3 Evaluation (Rolling 52-week)

For each rebalance week t after a burn-in window:
  - Train M3 ridge weights on weeks [t-train_window, t-1]
  - Score week t using those weights
  - Compute OOS Spearman IC, top-N returns, quintile spread

Compares: default linear | pipeline composite_score | walk-forward M3

Deterministic: stable sorting, no randomness, stable CSV output.
PIT-safe by construction: training uses only weeks < t.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backtest.calibrate_m3_weights import pooled_ridge
from backtest.evaluate_m3_weights import (
    DEFAULT_V3_WEIGHTS,
    FEATURE_COLS,
    compute_linear_score,
)
from backtest.fmb import newey_west_se
from backtest.metrics_m1 import spearman_rank_ic


# ── Core walk-forward loop ────────────────────────────────────────────

def walk_forward_evaluate(
    panel: pd.DataFrame,
    return_col: str = "fwd_5d",
    feature_cols: Optional[List[str]] = None,
    train_window_weeks: int = 52,
    ridge_lambda: float = 10.0,
    min_stocks: int = 50,
    top_n: int = 20,
    date_col: str = "rebalance_date",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run rolling walk-forward evaluation.

    Returns:
        (timeseries_df, weights_df)
        timeseries_df: per eval_date x variant metrics
        weights_df: per eval_date ridge weights learned
    """
    if feature_cols is None:
        feature_cols = sorted(FEATURE_COLS)
    else:
        feature_cols = sorted(feature_cols)

    # Stable sort
    panel = panel.sort_values([date_col, "ticker"]).reset_index(drop=True)
    dates = sorted(panel[date_col].unique())
    n_dates = len(dates)

    if train_window_weeks >= n_dates:
        raise ValueError(
            f"train_window_weeks ({train_window_weeks}) >= total dates ({n_dates})"
        )

    ts_records: List[Dict[str, Any]] = []
    wt_records: List[Dict[str, Any]] = []

    for i in range(train_window_weeks, n_dates):
        eval_date = dates[i]
        train_dates = dates[i - train_window_weeks : i]

        train_df = panel[panel[date_col].isin(train_dates)].copy()
        eval_df = panel[panel[date_col] == eval_date].copy()

        # Drop rows with NaN in return or features for eval
        eval_valid = eval_df.dropna(subset=[return_col] + feature_cols)
        n_stocks = len(eval_valid)
        if n_stocks < min_stocks:
            continue

        # Fit ridge on training window
        weights, _meta = pooled_ridge(
            train_df,
            return_col=return_col,
            feature_cols=feature_cols,
            date_col=date_col,
            ridge_lambda=ridge_lambda,
            min_stocks_per_week=min_stocks,
        )

        # Check weights are finite
        if all(v == 0.0 for v in weights.values()):
            continue

        # Record weights
        wt_rec = {"eval_date": eval_date}
        wt_rec.update(weights)
        wt_records.append(wt_rec)

        # Score eval set
        score_default = compute_linear_score(eval_valid, DEFAULT_V3_WEIGHTS)
        score_pipeline = eval_valid["composite_score"]
        score_m3_oos = compute_linear_score(eval_valid, weights)

        returns = eval_valid[return_col].values.tolist()

        # IC
        ic_default = spearman_rank_ic(score_default.tolist(), returns)
        ic_pipeline = spearman_rank_ic(score_pipeline.tolist(), returns)
        ic_m3_oos = spearman_rank_ic(score_m3_oos.tolist(), returns)

        # Top-N returns
        for variant_name, scores_s in [
            ("default_linear", score_default),
            ("pipeline", score_pipeline),
            ("m3_oos", score_m3_oos),
        ]:
            eval_scored = eval_valid.copy()
            eval_scored["_score"] = scores_s.values
            eval_scored = eval_scored.dropna(subset=["_score", return_col])
            top = eval_scored.nlargest(min(top_n, len(eval_scored)), "_score")
            topn_ret = top[return_col].mean() if len(top) > 0 else float("nan")

            # Quintile spread
            qspread = _quintile_spread(eval_scored, "_score", return_col)

            ic_val = {"default_linear": ic_default,
                      "pipeline": ic_pipeline,
                      "m3_oos": ic_m3_oos}[variant_name]

            ts_records.append({
                "eval_date": eval_date,
                "variant": variant_name,
                "ic": ic_val,
                "topn_return": topn_ret,
                "quintile_spread": qspread,
                "n_stocks": n_stocks,
            })

    ts_df = pd.DataFrame(ts_records)
    wt_df = pd.DataFrame(wt_records)
    return ts_df, wt_df


def _quintile_spread(
    df: pd.DataFrame,
    score_col: str,
    return_col: str,
) -> float:
    """Top-20% mean return minus bottom-20% mean return."""
    valid = df[[score_col, return_col]].dropna()
    if len(valid) < 5:
        return float("nan")
    valid = valid.copy()
    try:
        valid["q"] = pd.qcut(valid[score_col], 5, labels=False, duplicates="drop")
    except ValueError:
        return float("nan")
    q_means = valid.groupby("q")[return_col].mean()
    if len(q_means) < 2:
        return float("nan")
    return float(q_means.iloc[-1] - q_means.iloc[0])


# ── Aggregation ───────────────────────────────────────────────────────

def aggregate_results(
    ts_df: pd.DataFrame,
    nw_lags: int = 4,
) -> pd.DataFrame:
    """
    Aggregate per-date walk-forward results into summary per variant.

    Columns: variant, IC_mean, IC_tstat, IC_hit, TopN_mean, TopN_tstat,
             CumRet, Sharpe, MDD, QSpread_mean, n_weeks
    """
    rows = []
    for variant, grp in ts_df.groupby("variant"):
        grp = grp.sort_values("eval_date")

        ic_vals = grp["ic"].dropna().values
        topn_vals = grp["topn_return"].dropna().values
        qs_vals = grp["quintile_spread"].dropna().values
        n_weeks = len(grp)

        # IC stats
        ic_mean = np.mean(ic_vals) if len(ic_vals) > 0 else np.nan
        ic_se = newey_west_se(ic_vals, lags=nw_lags)
        ic_tstat = ic_mean / ic_se if ic_se > 0 and np.isfinite(ic_se) else np.nan
        ic_hit = np.mean(ic_vals > 0) if len(ic_vals) > 0 else np.nan

        # Top-N stats
        topn_mean = np.mean(topn_vals) if len(topn_vals) > 0 else np.nan
        topn_se = newey_west_se(topn_vals, lags=nw_lags)
        topn_tstat = (topn_mean / topn_se
                      if topn_se > 0 and np.isfinite(topn_se) else np.nan)

        # Cumulative return (compounded)
        cum = np.prod(1 + topn_vals) if len(topn_vals) > 0 else np.nan
        cum_ret = cum - 1.0 if np.isfinite(cum) else np.nan

        # Sharpe (weekly, annualized)
        if len(topn_vals) > 1:
            topn_std = np.std(topn_vals, ddof=1)
            sharpe = (topn_mean / topn_std * np.sqrt(52)
                      if topn_std > 0 else np.nan)
        else:
            sharpe = np.nan

        # Max drawdown
        if len(topn_vals) > 0:
            wealth = np.cumprod(1 + topn_vals)
            running_max = np.maximum.accumulate(wealth)
            dd = (wealth - running_max) / running_max
            mdd = float(np.min(dd))
        else:
            mdd = np.nan

        # Quintile spread
        qs_mean = np.mean(qs_vals) if len(qs_vals) > 0 else np.nan

        rows.append({
            "variant": variant,
            "IC_mean": ic_mean,
            "IC_tstat": ic_tstat,
            "IC_hit": ic_hit,
            "TopN_mean": topn_mean,
            "TopN_tstat": topn_tstat,
            "CumRet": cum_ret,
            "Sharpe": sharpe,
            "MDD": mdd,
            "QSpread_mean": qs_mean,
            "n_weeks": n_weeks,
        })

    return pd.DataFrame(rows).sort_values("variant").reset_index(drop=True)


# ── Console output ────────────────────────────────────────────────────

def print_walkforward_table(summary_df: pd.DataFrame) -> str:
    """Print side-by-side comparison table."""
    lines = []
    lines.append("")
    lines.append("=" * 95)
    lines.append("WALK-FORWARD OOS EVALUATION — COMPARISON TABLE")
    lines.append("=" * 95)

    variants = summary_df["variant"].tolist()
    metric_cols = [c for c in summary_df.columns if c != "variant"]

    header = f"{'Metric':<20}" + "".join(f"{v:>25}" for v in variants)
    lines.append(header)
    lines.append("-" * 95)

    for col in metric_cols:
        row = f"{col:<20}"
        for _, r in summary_df.iterrows():
            val = r[col]
            if isinstance(val, (int, np.integer)):
                row += f"{val:>25d}"
            elif isinstance(val, float) and np.isfinite(val):
                row += f"{val:>25.4f}"
            else:
                row += f"{'N/A':>25}"
        lines.append(row)

    lines.append("=" * 95)
    output = "\n".join(lines)
    print(output)
    return output


# ── CLI ───────────────────────────────────────────────────────────────

def run_walkforward(
    panel_path: str,
    return_col: str = "fwd_5d",
    train_window_weeks: int = 52,
    nw_lags: int = 4,
    top_n: int = 20,
    min_stocks: int = 50,
    ridge_lambda: float = 10.0,
    feature_cols: Optional[List[str]] = None,
    output_dir: str = "output/backtest_walkforward_m3",
) -> pd.DataFrame:
    """Full walk-forward evaluation with file output."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading panel from {panel_path} ...")
    df = pd.read_csv(panel_path)
    df["rebalance_date"] = df["rebalance_date"].astype(str)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df.sort_values(["rebalance_date", "ticker"]).reset_index(drop=True)

    n_dates = df["rebalance_date"].nunique()
    print(f"  {len(df):,} rows, {n_dates} dates, {df['ticker'].nunique()} tickers")
    print(f"  Train window: {train_window_weeks}w, return: {return_col}, "
          f"lambda: {ridge_lambda}, top-N: {top_n}")
    print(f"  OOS eval dates: {n_dates - train_window_weeks}")

    print("\nRunning walk-forward evaluation ...")
    ts_df, wt_df = walk_forward_evaluate(
        df,
        return_col=return_col,
        feature_cols=feature_cols,
        train_window_weeks=train_window_weeks,
        ridge_lambda=ridge_lambda,
        min_stocks=min_stocks,
        top_n=top_n,
    )

    if len(ts_df) == 0:
        print("  WARNING: No OOS eval dates produced results.")
        return pd.DataFrame()

    # Aggregate
    summary_df = aggregate_results(ts_df, nw_lags=nw_lags)

    # Direction guardrail
    for _, row in summary_df.iterrows():
        if row["IC_mean"] < -0.02:
            print(f"  WARNING: {row['variant']} has mean IC = {row['IC_mean']:.4f} "
                  f"(strongly negative) — check sign/feature mapping")

    # Write outputs
    summary_df.to_csv(
        os.path.join(output_dir, "walkforward_summary.csv"), index=False,
    )
    ts_df.to_csv(
        os.path.join(output_dir, "walkforward_timeseries.csv"), index=False,
    )
    if len(wt_df) > 0:
        wt_df.to_csv(
            os.path.join(output_dir, "walkforward_weights_by_date.csv.gz"),
            index=False, compression="gzip",
        )

    print_walkforward_table(summary_df)
    print(f"\nOutputs written to {output_dir}/")
    return summary_df


def main():
    parser = argparse.ArgumentParser(
        description="Walk-Forward OOS M3 Evaluation (Rolling 52-week)"
    )
    parser.add_argument(
        "--panel", required=True,
        help="Path to M1 panel CSV (gzipped OK)",
    )
    parser.add_argument("--return-col", default="fwd_5d")
    parser.add_argument("--train-window-weeks", type=int, default=52)
    parser.add_argument("--nw-lags", type=int, default=4)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-stocks", type=int, default=50)
    parser.add_argument("--lambda", dest="ridge_lambda", type=float, default=10.0)
    parser.add_argument("--feature-cols", nargs="+", default=None)
    parser.add_argument(
        "--output-dir", default="output/backtest_walkforward_m3",
    )
    args = parser.parse_args()

    run_walkforward(
        panel_path=args.panel,
        return_col=args.return_col,
        train_window_weeks=args.train_window_weeks,
        nw_lags=args.nw_lags,
        top_n=args.top_n,
        min_stocks=args.min_stocks,
        ridge_lambda=args.ridge_lambda,
        feature_cols=args.feature_cols,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
