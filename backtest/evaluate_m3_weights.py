"""
M3 Before/After Weight Evaluation

Compare predictive power of default V3 weights vs M3 ridge-calibrated weights
on the existing 2-year weekly panel (105 dates, ~332 tickers/week).

Three score variants:
  - score_default_linear: V3 4-feature renormalized weights (higher = better)
  - score_m3_calibrated:  M3 ridge pred = X @ w, no transforms (higher = higher expected return)
  - score_pipeline:       Existing composite_score from panel (higher = better)

All variants use "higher score = rank higher" for portfolio selection and IC.
M3 weights may be negative — that's the model's learned direction; no negation applied.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtest.metrics_m1 import spearman_rank_ic

# ── Default V3 weights (4-feature renormalized to sum=1) ──────────────
DEFAULT_V3_WEIGHTS = {
    "clinical_normalized": 0.406,
    "financial_normalized": 0.375,
    "momentum_normalized": 0.141,
    "valuation_normalized": 0.078,
}

FEATURE_COLS = [
    "clinical_normalized",
    "financial_normalized",
    "momentum_normalized",
    "valuation_normalized",
]

RETURN_COLS = ["fwd_5d", "fwd_21d", "fwd_63d", "fwd_126d"]


# ── Weight loading ────────────────────────────────────────────────────
def load_m3_weights(path: str) -> Dict[str, float]:
    """Load M3 ridge-calibrated weights from JSON."""
    with open(path) as f:
        data = json.load(f)
    return data["feature_weights"]


# ── Score computation ─────────────────────────────────────────────────
def compute_linear_score(
    df: pd.DataFrame,
    weights: Dict[str, float],
) -> pd.Series:
    """
    Weighted linear combination of feature columns: pred = Σ(w_i * x_i).

    No sign transforms applied — weights are used as-is. Negative weights
    are the model's learned direction (higher feature → lower expected return).

    Returns NaN for rows where any feature is NaN.
    """
    score = pd.Series(0.0, index=df.index)
    for col, w in weights.items():
        score = score + df[col] * w
    return score


# ── IC computation (per-date) ─────────────────────────────────────────
def compute_ic(
    df: pd.DataFrame,
    score_col: str,
    return_col: str,
    date_col: str = "rebalance_date",
) -> pd.Series:
    """Spearman rank IC per rebalance date. Returns Series indexed by date."""
    results = {}
    for dt, grp in df.groupby(date_col):
        s = grp[score_col].values
        r = grp[return_col].values
        results[dt] = spearman_rank_ic(s.tolist(), r.tolist())
    return pd.Series(results, name=f"ic_{score_col}_{return_col}")


# ── Top-N returns ─────────────────────────────────────────────────────
def compute_topn_returns(
    df: pd.DataFrame,
    score_col: str,
    return_col: str,
    n: int = 20,
    date_col: str = "rebalance_date",
) -> pd.Series:
    """Mean equal-weight return of top-N ranked stocks per date."""
    results = {}
    for dt, grp in df.groupby(date_col):
        valid = grp[[score_col, return_col]].dropna()
        if len(valid) == 0:
            results[dt] = float("nan")
            continue
        top = valid.nlargest(min(n, len(valid)), score_col)
        results[dt] = top[return_col].mean()
    return pd.Series(results, name=f"topn_{n}_{score_col}")


# ── Quintile spread ───────────────────────────────────────────────────
def compute_quintile_spread(
    df: pd.DataFrame,
    score_col: str,
    return_col: str,
    date_col: str = "rebalance_date",
) -> pd.Series:
    """Q5 mean return minus Q1 mean return per date (Q5 = best-ranked)."""
    results = {}
    for dt, grp in df.groupby(date_col):
        valid = grp[[score_col, return_col]].dropna()
        if len(valid) < 5:
            results[dt] = float("nan")
            continue
        valid = valid.copy()
        valid["quintile"] = pd.qcut(
            valid[score_col], 5, labels=False, duplicates="drop"
        )
        q_means = valid.groupby("quintile")[return_col].mean()
        if len(q_means) < 2:
            results[dt] = float("nan")
            continue
        results[dt] = q_means.iloc[-1] - q_means.iloc[0]
    return pd.Series(results, name=f"qspread_{score_col}")


# ── Hit rate ──────────────────────────────────────────────────────────
def compute_hit_rate(
    df: pd.DataFrame,
    score_col: str,
    return_col: str,
    date_col: str = "rebalance_date",
) -> pd.Series:
    """% of top-quintile stocks beating cross-sectional median return."""
    results = {}
    for dt, grp in df.groupby(date_col):
        valid = grp[[score_col, return_col]].dropna()
        if len(valid) < 5:
            results[dt] = float("nan")
            continue
        median_ret = valid[return_col].median()
        threshold = valid[score_col].quantile(0.8)
        top_q = valid[valid[score_col] >= threshold]
        if len(top_q) == 0:
            results[dt] = float("nan")
            continue
        results[dt] = (top_q[return_col] > median_ret).mean()
    return pd.Series(results, name=f"hitrate_{score_col}")


# ── Turnover ──────────────────────────────────────────────────────────
def compute_turnover(
    df: pd.DataFrame,
    score_col: str,
    n: int = 20,
    date_col: str = "rebalance_date",
    ticker_col: str = "ticker",
) -> pd.Series:
    """Week-to-week Jaccard distance of top-N sets (0=stable, 1=complete turnover)."""
    dates = sorted(df[date_col].unique())
    prev_set: Optional[set] = None
    results = {}
    for dt in dates:
        grp = df[df[date_col] == dt][[score_col, ticker_col]].dropna(subset=[score_col])
        top = grp.nlargest(min(n, len(grp)), score_col)
        cur_set = set(top[ticker_col])
        if prev_set is not None and len(cur_set) > 0 and len(prev_set) > 0:
            union = prev_set | cur_set
            inter = prev_set & cur_set
            results[dt] = 1.0 - len(inter) / len(union) if len(union) > 0 else 0.0
        prev_set = cur_set
    return pd.Series(results, name=f"turnover_{score_col}")


# ── Cumulative returns & drawdown ─────────────────────────────────────
def build_cumulative_curve(topn_weekly_returns: pd.Series) -> pd.Series:
    """Chain weekly returns into cumulative wealth curve (starts at 1)."""
    returns = topn_weekly_returns.dropna().sort_index()
    cum = (1 + returns).cumprod()
    return cum


def compute_drawdown(cum_curve: pd.Series) -> Tuple[float, int]:
    """Max drawdown and duration (in periods) from cumulative curve."""
    if len(cum_curve) == 0:
        return 0.0, 0
    running_max = cum_curve.cummax()
    drawdown = (cum_curve - running_max) / running_max
    max_dd = drawdown.min()
    # Duration: longest streak below running max
    below = (cum_curve < running_max).astype(int)
    if below.sum() == 0:
        return float(max_dd), 0
    streaks = below.groupby((below != below.shift()).cumsum())
    max_dur = 0
    for _, streak in streaks:
        if streak.iloc[0] == 1:
            max_dur = max(max_dur, len(streak))
    return float(max_dd), max_dur


# ── IC summary stats ──────────────────────────────────────────────────
def ic_summary(ic_series: pd.Series) -> Dict[str, float]:
    """Compute mean, median, std, t-stat, hit rate from IC time-series."""
    valid = ic_series.dropna()
    n = len(valid)
    if n == 0:
        return {"mean": np.nan, "median": np.nan, "std": np.nan,
                "tstat": np.nan, "hit_rate": np.nan, "n": 0}
    mean = valid.mean()
    std = valid.std(ddof=1) if n > 1 else np.nan
    return {
        "mean": mean,
        "median": valid.median(),
        "std": std,
        "tstat": mean / (std / np.sqrt(n)) if std > 0 else np.nan,
        "hit_rate": (valid > 0).mean(),
        "n": n,
    }


# ── Comparison table printer ──────────────────────────────────────────
def print_comparison_table(metrics: Dict[str, Dict[str, Any]]) -> str:
    """Print side-by-side comparison of all three variants."""
    lines = []
    lines.append("")
    lines.append("=" * 90)
    lines.append("M3 WEIGHT EVALUATION — COMPARISON TABLE (fwd_21d)")
    lines.append("=" * 90)

    variants = list(metrics.keys())
    header = f"{'Metric':<30}" + "".join(f"{v:>20}" for v in variants)
    lines.append(header)
    lines.append("-" * 90)

    # Collect all metric keys from first variant
    first = metrics[variants[0]]
    for key in first:
        row = f"{key:<30}"
        for v in variants:
            val = metrics[v].get(key, "")
            if isinstance(val, float):
                row += f"{val:>20.4f}"
            elif isinstance(val, int):
                row += f"{val:>20d}"
            else:
                row += f"{str(val):>20}"
        lines.append(row)

    lines.append("=" * 90)
    output = "\n".join(lines)
    print(output)
    return output


# ── Main evaluation ───────────────────────────────────────────────────
def run_evaluation(
    panel_path: str,
    weights_path: str,
    output_dir: str,
) -> Dict[str, Any]:
    """Run full M3 vs default weight evaluation."""
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print(f"Loading panel from {panel_path} ...")
    df = pd.read_csv(panel_path)
    print(f"  → {len(df):,} rows, {df['rebalance_date'].nunique()} dates, "
          f"{df['ticker'].nunique()} unique tickers")

    # Load M3 weights
    m3_weights = load_m3_weights(weights_path)
    print(f"M3 weights: {m3_weights}")

    # ── Compute scores ─────────────────────────────────────────────
    df["score_default_linear"] = compute_linear_score(df, DEFAULT_V3_WEIGHTS)
    df["score_m3_calibrated"] = compute_linear_score(df, m3_weights)
    df["score_pipeline"] = df["composite_score"]

    score_variants = ["score_default_linear", "score_m3_calibrated", "score_pipeline"]

    # ── IC time-series ─────────────────────────────────────────────
    print("\nComputing IC time-series ...")
    ic_records = []
    for sv in score_variants:
        for rc in RETURN_COLS:
            ic_ts = compute_ic(df, sv, rc)
            for dt, val in ic_ts.items():
                ic_records.append({
                    "rebalance_date": dt,
                    "variant": sv,
                    "return_col": rc,
                    "ic": val,
                })

    ic_df = pd.DataFrame(ic_records)
    ic_df.to_csv(os.path.join(output_dir, "ic_timeseries.csv"), index=False)

    # ── Top-N returns ──────────────────────────────────────────────
    print("Computing top-N returns ...")
    topn_records = []
    for sv in score_variants:
        for rc in RETURN_COLS:
            for n in [20, 40, 60]:
                topn_ts = compute_topn_returns(df, sv, rc, n=n)
                for dt, val in topn_ts.items():
                    topn_records.append({
                        "rebalance_date": dt,
                        "variant": sv,
                        "return_col": rc,
                        "n": n,
                        "mean_return": val,
                    })

    topn_df = pd.DataFrame(topn_records)
    topn_df.to_csv(os.path.join(output_dir, "topn_returns.csv"), index=False)

    # ── Build summary metrics ──────────────────────────────────────
    print("Building summary metrics ...")
    summary_rows = []
    comparison = {}  # variant -> metrics dict for console table

    for sv in score_variants:
        variant_metrics = {}

        # IC metrics (fwd_21d primary)
        for rc in RETURN_COLS:
            ic_ts = compute_ic(df, sv, rc)
            ics = ic_summary(ic_ts)
            prefix = rc.replace("fwd_", "")
            for k, v in ics.items():
                summary_rows.append({
                    "variant": sv, "horizon": rc,
                    "metric": f"ic_{k}", "value": v,
                })
            if rc == "fwd_21d":
                for k, v in ics.items():
                    variant_metrics[f"IC_{k}"] = v

        # Top-N returns (fwd_21d)
        for n in [20, 40, 60]:
            topn_ts = compute_topn_returns(df, sv, "fwd_21d", n=n)
            mean_ret = topn_ts.mean()
            summary_rows.append({
                "variant": sv, "horizon": "fwd_21d",
                "metric": f"topn_{n}_mean", "value": mean_ret,
            })
            variant_metrics[f"top{n}_mean_21d"] = mean_ret

        # Quintile spread (fwd_21d)
        qspread = compute_quintile_spread(df, sv, "fwd_21d")
        mean_qs = qspread.mean()
        summary_rows.append({
            "variant": sv, "horizon": "fwd_21d",
            "metric": "quintile_spread_mean", "value": mean_qs,
        })
        variant_metrics["qspread_21d"] = mean_qs

        # Hit rate (fwd_21d)
        hr = compute_hit_rate(df, sv, "fwd_21d")
        mean_hr = hr.mean()
        summary_rows.append({
            "variant": sv, "horizon": "fwd_21d",
            "metric": "hit_rate_mean", "value": mean_hr,
        })
        variant_metrics["hit_rate_21d"] = mean_hr

        # Turnover
        for n in [20, 40]:
            turn = compute_turnover(df, sv, n=n)
            mean_turn = turn.mean()
            summary_rows.append({
                "variant": sv, "horizon": "N/A",
                "metric": f"turnover_{n}_mean", "value": mean_turn,
            })
            variant_metrics[f"turnover_{n}"] = mean_turn

        # Cumulative return (fwd_5d weekly)
        topn_5d = compute_topn_returns(df, sv, "fwd_5d", n=20)
        cum = build_cumulative_curve(topn_5d)
        if len(cum) > 0:
            total_ret = cum.iloc[-1] - 1.0
            max_dd, dd_dur = compute_drawdown(cum)
            weekly_rets = topn_5d.dropna()
            sharpe = (weekly_rets.mean() / weekly_rets.std() * np.sqrt(52)
                      if weekly_rets.std() > 0 else np.nan)
        else:
            total_ret, max_dd, dd_dur, sharpe = np.nan, np.nan, 0, np.nan

        variant_metrics["cum_return_5d"] = total_ret
        variant_metrics["max_drawdown"] = max_dd
        variant_metrics["dd_duration"] = dd_dur
        variant_metrics["sharpe_annual"] = sharpe

        for k in ["cum_return_5d", "max_drawdown", "dd_duration", "sharpe_annual"]:
            summary_rows.append({
                "variant": sv, "horizon": "fwd_5d",
                "metric": k, "value": variant_metrics[k],
            })

        comparison[sv] = variant_metrics

    # Write summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, "comparison_summary.csv"), index=False)

    # Print console table
    print_comparison_table(comparison)

    # Direction guardrail: warn if any variant has strongly negative mean IC
    for sv, vm in comparison.items():
        mean_ic = vm.get("IC_mean", 0)
        if mean_ic < -0.02:
            print(f"  ⚠ {sv}: mean IC = {mean_ic:.4f} (strongly negative) "
                  f"— direction may be inverted; check sign/feature mapping")

    print(f"\nOutputs written to {output_dir}/")
    return comparison


# ── CLI ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="M3 Before/After Weight Evaluation"
    )
    parser.add_argument(
        "--panel",
        default="output/backtest_m1_full/panel_weekly.csv.gz",
        help="Path to panel CSV (gzipped OK)",
    )
    parser.add_argument(
        "--weights",
        default="data/module5_weights_m3.json",
        help="Path to M3 weights JSON",
    )
    parser.add_argument(
        "--output-dir",
        default="output/backtest_m3_eval",
        help="Output directory for CSVs",
    )
    args = parser.parse_args()
    run_evaluation(args.panel, args.weights, args.output_dir)


if __name__ == "__main__":
    main()
