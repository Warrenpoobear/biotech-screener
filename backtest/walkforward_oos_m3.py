"""
Walk-Forward Out-of-Sample M3 Evaluation (Rolling 52-week)

For each rebalance week t after a burn-in window:
  - Train M3 ridge weights on weeks [t-train_window, t-1]
  - Score week t using those weights
  - Compute OOS Spearman IC, top-N returns, quintile spread

Compares: default linear | pipeline composite_score | walk-forward M3
Optionally adds regime-conditioned M3 (--use-regime).

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
    regime_by_date: Optional[Dict[str, str]] = None,
    min_fit_weeks: int = 8,
    blend_k: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run rolling walk-forward evaluation.

    Parameters
    ----------
    regime_by_date : if provided, adds m3_oos_regime variant that trains
        ridge weights only on training weeks matching the eval week's regime.
    min_fit_weeks : minimum distinct regime training dates to attempt a
        regime-specific ridge fit (low bar — allows fitting with thin data).
    blend_k : shrinkage anchor for blending. alpha = n_regime / (n_regime + blend_k).
        Higher blend_k → more global shrinkage. Separating from min_fit_weeks
        lets you fit on 8+ weeks but shrink heavily until you have 20+.

    Returns:
        (timeseries_df, weights_df)
        timeseries_df: per eval_date x variant metrics
        weights_df: per eval_date ridge weights learned
    """
    if feature_cols is None:
        feature_cols = sorted(FEATURE_COLS)
    else:
        feature_cols = sorted(feature_cols)

    use_regime = regime_by_date is not None

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

        # Fit global ridge on training window
        weights_global, _meta = pooled_ridge(
            train_df,
            return_col=return_col,
            feature_cols=feature_cols,
            date_col=date_col,
            ridge_lambda=ridge_lambda,
            min_stocks_per_week=min_stocks,
        )

        # Check global weights are finite
        if all(v == 0.0 for v in weights_global.values()):
            continue

        # Record global weights
        wt_rec = {"eval_date": eval_date, "fit_scope": "GLOBAL"}
        wt_rec.update(weights_global)
        wt_records.append(wt_rec)

        # Regime-conditioned weights (blended shrinkage)
        weights_regime = None
        fit_scope = "N/A"
        eval_regime = "N/A"
        blend_alpha = 0.0
        if use_regime:
            eval_regime = regime_by_date.get(eval_date, "CHOP")
            # Filter training dates to same regime
            regime_train_dates = [
                d for d in train_dates
                if regime_by_date.get(d, "CHOP") == eval_regime
            ]
            n_regime = len(regime_train_dates)
            if n_regime >= min_fit_weeks:
                regime_train_df = panel[
                    panel[date_col].isin(regime_train_dates)
                ].copy()
                weights_regime_raw, _ = pooled_ridge(
                    regime_train_df,
                    return_col=return_col,
                    feature_cols=feature_cols,
                    date_col=date_col,
                    ridge_lambda=ridge_lambda,
                    min_stocks_per_week=min_stocks,
                )
                if all(v == 0.0 for v in weights_regime_raw.values()):
                    weights_regime = weights_global
                    fit_scope = "GLOBAL_FALLBACK"
                else:
                    fit_scope = "REGIME"
                    wt_rec_r = {
                        "eval_date": eval_date,
                        "fit_scope": f"REGIME_{eval_regime}",
                    }
                    wt_rec_r.update(weights_regime_raw)
                    wt_records.append(wt_rec_r)
                    weights_regime = weights_regime_raw
            else:
                weights_regime = weights_global
                fit_scope = "GLOBAL_FALLBACK"

            # Blended shrinkage: w_used = alpha * w_regime + (1-alpha) * w_global
            # alpha = n_regime / (n_regime + blend_k), so more regime data → more tilt
            blend_alpha = n_regime / (n_regime + blend_k)
            if weights_regime is not None and fit_scope == "REGIME":
                weights_regime = _blend_weights(
                    weights_regime, weights_global, blend_alpha, feature_cols,
                )
            # For GLOBAL_FALLBACK, alpha is still computed but weights stay global

        # Score eval set
        score_default = compute_linear_score(eval_valid, DEFAULT_V3_WEIGHTS)
        score_pipeline = eval_valid["composite_score"]
        score_m3_oos = compute_linear_score(eval_valid, weights_global)

        variants = [
            ("default_linear", score_default),
            ("pipeline", score_pipeline),
            ("m3_oos", score_m3_oos),
        ]

        if use_regime and weights_regime is not None:
            score_m3_regime = compute_linear_score(eval_valid, weights_regime)
            variants.append(("m3_oos_regime", score_m3_regime))

        returns = eval_valid[return_col].values.tolist()

        # Compute IC for all variants
        ic_map = {}
        for vname, scores_s in variants:
            ic_map[vname] = spearman_rank_ic(scores_s.tolist(), returns)

        # Top-N returns + quintile spread
        for variant_name, scores_s in variants:
            eval_scored = eval_valid.copy()
            eval_scored["_score"] = scores_s.values
            eval_scored = eval_scored.dropna(subset=["_score", return_col])
            top = eval_scored.nlargest(min(top_n, len(eval_scored)), "_score")
            topn_ret = top[return_col].mean() if len(top) > 0 else float("nan")

            qspread = _quintile_spread(eval_scored, "_score", return_col)

            rec = {
                "eval_date": eval_date,
                "variant": variant_name,
                "ic": ic_map[variant_name],
                "topn_return": topn_ret,
                "quintile_spread": qspread,
                "n_stocks": n_stocks,
                "regime": eval_regime if use_regime else "N/A",
            }
            if variant_name == "m3_oos_regime":
                rec["fit_scope"] = fit_scope
                rec["blend_alpha"] = blend_alpha
            else:
                rec["fit_scope"] = "N/A"
                rec["blend_alpha"] = float("nan")
            ts_records.append(rec)

    ts_df = pd.DataFrame(ts_records)
    wt_df = pd.DataFrame(wt_records)
    return ts_df, wt_df


def _blend_weights(
    w_regime: Dict[str, float],
    w_global: Dict[str, float],
    alpha: float,
    feature_cols: List[str],
) -> Dict[str, float]:
    """Blend regime and global weights: w = alpha * w_regime + (1-alpha) * w_global."""
    return {
        c: alpha * w_regime.get(c, 0.0) + (1 - alpha) * w_global.get(c, 0.0)
        for c in feature_cols
    }


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
        rows.append(_summarize_group(variant, grp, nw_lags))

    return pd.DataFrame(rows).sort_values("variant").reset_index(drop=True)


def aggregate_by_regime(
    ts_df: pd.DataFrame,
    nw_lags: int = 4,
) -> pd.DataFrame:
    """
    Aggregate per-date results broken down by variant x regime.

    Returns DataFrame with columns: variant, regime, IC_mean, IC_tstat, ...
    """
    if "regime" not in ts_df.columns or ts_df["regime"].eq("N/A").all():
        return pd.DataFrame()

    rows = []
    for (variant, regime), grp in ts_df.groupby(["variant", "regime"]):
        if regime == "N/A":
            continue
        grp = grp.sort_values("eval_date")
        row = _summarize_group(variant, grp, nw_lags)
        row["regime"] = regime
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    cols = ["variant", "regime"] + [c for c in df.columns if c not in ("variant", "regime")]
    return df[cols].sort_values(["variant", "regime"]).reset_index(drop=True)


def _summarize_group(
    variant: str,
    grp: pd.DataFrame,
    nw_lags: int,
) -> Dict[str, Any]:
    """Compute summary metrics for a single variant group."""
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

    return {
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
    }


# ── Console output ────────────────────────────────────────────────────

def print_walkforward_table(summary_df: pd.DataFrame, title: str = "") -> str:
    """Print side-by-side comparison table."""
    lines = []
    lines.append("")
    lines.append("=" * 95)
    label = title or "WALK-FORWARD OOS EVALUATION — COMPARISON TABLE"
    lines.append(label)
    lines.append("=" * 95)

    variants = summary_df["variant"].tolist()
    metric_cols = [c for c in summary_df.columns if c not in ("variant", "regime")]

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
    use_regime: bool = False,
    price_csv: Optional[str] = None,
    market_ticker: str = "XBI",
    min_fit_weeks: int = 8,
    blend_k: int = 20,
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

    # Regime setup
    regime_by_date = None
    if use_regime:
        from backtest.regime import (
            load_price_history,
            compute_regime_series,
            assign_regime_to_rebalance_dates,
        )
        if price_csv is None:
            price_csv = str(_PROJECT_ROOT / "production_data" / "price_history.csv")
        print(f"\nLoading regime from {price_csv} (ticker={market_ticker}) ...")
        price_df = load_price_history(Path(price_csv))
        regime_series = compute_regime_series(price_df, market_ticker=market_ticker)
        rebalance_dates = sorted(df["rebalance_date"].unique())
        regime_by_date = assign_regime_to_rebalance_dates(
            regime_series, rebalance_dates,
        )
        # Print regime distribution
        from collections import Counter
        counts = Counter(regime_by_date.values())
        print(f"  Regime distribution: {dict(sorted(counts.items()))}")
        print(f"  Min fit weeks: {min_fit_weeks}, blend_k: {blend_k}")

    print("\nRunning walk-forward evaluation ...")
    ts_df, wt_df = walk_forward_evaluate(
        df,
        return_col=return_col,
        feature_cols=feature_cols,
        train_window_weeks=train_window_weeks,
        ridge_lambda=ridge_lambda,
        min_stocks=min_stocks,
        top_n=top_n,
        regime_by_date=regime_by_date,
        min_fit_weeks=min_fit_weeks,
        blend_k=blend_k,
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
        os.path.join(output_dir, "oos_summary.csv"), index=False,
    )
    ts_df.to_csv(
        os.path.join(output_dir, "oos_timeseries.csv"), index=False,
    )
    if len(wt_df) > 0:
        wt_df.to_csv(
            os.path.join(output_dir, "oos_weights_by_date.csv.gz"),
            index=False, compression="gzip",
        )

    print_walkforward_table(summary_df)

    # Per-regime summary
    if use_regime:
        regime_summary = aggregate_by_regime(ts_df, nw_lags=nw_lags)
        if len(regime_summary) > 0:
            regime_summary.to_csv(
                os.path.join(output_dir, "oos_regime_summary.csv"), index=False,
            )
            # Print per-regime breakdown for key variants
            for regime in sorted(regime_summary["regime"].unique()):
                subset = regime_summary[regime_summary["regime"] == regime]
                print_walkforward_table(
                    subset.drop(columns=["regime"]),
                    title=f"REGIME BREAKDOWN: {regime}",
                )

            # Print fit_scope distribution for regime variant
            regime_rows = ts_df[ts_df["variant"] == "m3_oos_regime"]
            if len(regime_rows) > 0:
                scope_counts = regime_rows["fit_scope"].value_counts().to_dict()
                n_total = len(regime_rows)
                n_fallback = scope_counts.get("GLOBAL_FALLBACK", 0)
                fallback_pct = n_fallback / n_total * 100 if n_total > 0 else 0
                n_regime_fit = scope_counts.get("REGIME", 0)
                print(f"\n  Regime fit_scope: {n_regime_fit}/{n_total} REGIME, "
                      f"{n_fallback}/{n_total} GLOBAL_FALLBACK ({fallback_pct:.0f}%)")
                if fallback_pct > 40:
                    print(f"  WARNING: {fallback_pct:.0f}% of weeks used global "
                          f"fallback; consider lowering --min-fit-weeks or "
                          f"increasing --train-window-weeks")
                # Print blend alpha diagnostics
                alphas = regime_rows["blend_alpha"].dropna()
                if len(alphas) > 0:
                    print(f"  Mean blend alpha (all weeks): {alphas.mean():.2f} "
                          f"(1.0 = pure regime, 0.0 = pure global)")
                regime_fit_rows = regime_rows[regime_rows["fit_scope"] == "REGIME"]
                if len(regime_fit_rows) > 0:
                    eff_alphas = regime_fit_rows["blend_alpha"].dropna()
                    if len(eff_alphas) > 0:
                        print(f"  Mean blend alpha (REGIME-fit weeks only): "
                              f"{eff_alphas.mean():.2f}")

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
    # Regime options
    parser.add_argument("--use-regime", action="store_true",
                        help="Enable regime-conditioned M3 variant")
    parser.add_argument("--price-csv", default=None,
                        help="Path to price history CSV (default: production_data/price_history.csv)")
    parser.add_argument("--market-ticker", default="XBI",
                        help="Ticker for regime classification (default: XBI)")
    parser.add_argument("--min-fit-weeks", type=int, default=8,
                        help="Min regime training weeks to attempt fit (default: 8)")
    parser.add_argument("--blend-k", type=int, default=20,
                        help="Shrinkage anchor for regime/global blending (default: 20)")
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
        use_regime=args.use_regime,
        price_csv=args.price_csv,
        market_ticker=args.market_ticker,
        min_fit_weeks=args.min_fit_weeks,
        blend_k=args.blend_k,
    )


if __name__ == "__main__":
    main()
