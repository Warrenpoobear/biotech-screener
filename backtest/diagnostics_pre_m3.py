#!/usr/bin/env python3
"""
Pre-M3 diagnostics: sign stability, rolling IC/FMB, coverage report.

Usage:
    python3 backtest/diagnostics_pre_m3.py \
        --panel output/backtest_m1_full/panel_weekly.csv.gz \
        --outdir output/diagnostics_pre_m3
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backtest.metrics_m1 import spearman_rank_ic
from backtest.fmb import (
    drop_constant_features,
    fama_macbeth,
    zscore_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("diag_pre_m3")

_FEATURE_COLS = [
    "catalyst_normalized",
    "clinical_normalized",
    "composite_score",
    "financial_normalized",
    "momentum_normalized",
    "uncertainty_penalty",
    "valuation_normalized",
]

_RETURN_COLS = ["fwd_5d", "fwd_21d", "fwd_63d", "fwd_126d"]


# ---------------------------------------------------------------------------
# 1. Coverage report
# ---------------------------------------------------------------------------


def coverage_report(
    panel: pd.DataFrame,
    feature_cols: List[str],
    date_col: str = "rebalance_date",
) -> pd.DataFrame:
    """Per-feature: % non-null per week, % weeks dropped as constant."""
    dates = sorted(panel[date_col].unique())
    rows = []
    for col in sorted(feature_cols):
        if col not in panel.columns:
            rows.append({
                "feature": col,
                "pct_nonnull_overall": 0.0,
                "pct_weeks_present": 0.0,
                "pct_weeks_constant": 100.0,
                "median_nonnull_per_week": 0.0,
                "n_weeks_total": len(dates),
            })
            continue
        nonnull_per_week = []
        constant_weeks = 0
        present_weeks = 0
        for dt in dates:
            mask = panel[date_col] == dt
            vals = panel.loc[mask, col].dropna()
            n = len(vals)
            nonnull_per_week.append(n)
            if n > 0:
                present_weeks += 1
                std = vals.std(ddof=1) if n > 1 else 0.0
                if not np.isfinite(std) or std < 1e-12:
                    constant_weeks += 1

        total_cells = len(panel)
        nonnull_total = panel[col].notna().sum()

        rows.append({
            "feature": col,
            "pct_nonnull_overall": round(100.0 * nonnull_total / total_cells, 2),
            "pct_weeks_present": round(100.0 * present_weeks / len(dates), 2),
            "pct_weeks_constant": round(
                100.0 * constant_weeks / max(present_weeks, 1), 2
            ),
            "median_nonnull_per_week": float(np.median(nonnull_per_week)),
            "n_weeks_total": len(dates),
        })
    return pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Subperiod sign stability (H1 vs H2)
# ---------------------------------------------------------------------------


def compute_ic_for_subset(
    panel: pd.DataFrame,
    feature_cols: List[str],
    return_cols: List[str],
    date_col: str = "rebalance_date",
) -> pd.DataFrame:
    """Compute mean IC per (feature, horizon) for a panel subset."""
    dates = sorted(panel[date_col].unique())
    rows = []
    for fcol in sorted(feature_cols):
        if fcol not in panel.columns:
            continue
        for rcol in sorted(return_cols):
            if rcol not in panel.columns:
                continue
            ics = []
            for dt in dates:
                mask = panel[date_col] == dt
                scores = panel.loc[mask, fcol].values
                rets = panel.loc[mask, rcol].values
                ic = spearman_rank_ic(scores, rets)
                if np.isfinite(ic):
                    ics.append(ic)
            if ics:
                rows.append({
                    "feature": fcol,
                    "return_col": rcol,
                    "mean_ic": round(np.mean(ics), 6),
                    "n_dates": len(ics),
                })
    return pd.DataFrame(rows)


def subperiod_sign_stability(
    panel: pd.DataFrame,
    feature_cols: List[str],
    return_cols: List[str],
    date_col: str = "rebalance_date",
) -> pd.DataFrame:
    """Split panel in half, compare IC signs."""
    dates = sorted(panel[date_col].unique())
    mid = len(dates) // 2
    h1_dates = set(dates[:mid])
    h2_dates = set(dates[mid:])

    h1 = panel[panel[date_col].isin(h1_dates)]
    h2 = panel[panel[date_col].isin(h2_dates)]

    ic_h1 = compute_ic_for_subset(h1, feature_cols, return_cols, date_col)
    ic_h2 = compute_ic_for_subset(h2, feature_cols, return_cols, date_col)

    merged = ic_h1.merge(
        ic_h2,
        on=["feature", "return_col"],
        suffixes=("_h1", "_h2"),
        how="outer",
    )
    merged["sign_match"] = (
        np.sign(merged["mean_ic_h1"].fillna(0))
        == np.sign(merged["mean_ic_h2"].fillna(0))
    )
    merged["h1_dates"] = f"{dates[0]}..{dates[mid-1]}"
    merged["h2_dates"] = f"{dates[mid]}..{dates[-1]}"

    return merged.sort_values(["feature", "return_col"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Rolling IC (26-week and 52-week)
# ---------------------------------------------------------------------------


def rolling_ic(
    panel: pd.DataFrame,
    feature_cols: List[str],
    return_cols: List[str],
    window_weeks: int = 26,
    date_col: str = "rebalance_date",
) -> pd.DataFrame:
    """Compute rolling-window mean IC per (feature, horizon) at each end date."""
    dates = sorted(panel[date_col].unique())
    rows = []
    for end_idx in range(window_weeks, len(dates) + 1):
        window_dates = set(dates[end_idx - window_weeks : end_idx])
        subset = panel[panel[date_col].isin(window_dates)]
        end_date = dates[end_idx - 1]

        for fcol in sorted(feature_cols):
            if fcol not in panel.columns:
                continue
            for rcol in sorted(return_cols):
                if rcol not in panel.columns:
                    continue
                ics = []
                for dt in sorted(window_dates):
                    mask = subset[date_col] == dt
                    scores = subset.loc[mask, fcol].values
                    rets = subset.loc[mask, rcol].values
                    ic = spearman_rank_ic(scores, rets)
                    if np.isfinite(ic):
                        ics.append(ic)
                if ics:
                    rows.append({
                        "end_date": end_date,
                        "window_weeks": window_weeks,
                        "feature": fcol,
                        "return_col": rcol,
                        "mean_ic": round(np.mean(ics), 6),
                        "n_dates": len(ics),
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Rolling FMB (52-week)
# ---------------------------------------------------------------------------


def rolling_fmb(
    panel: pd.DataFrame,
    feature_cols: List[str],
    return_col: str = "fwd_21d",
    window_weeks: int = 52,
    date_col: str = "rebalance_date",
) -> pd.DataFrame:
    """Compute rolling-window FMB betas + t-stats at each end date."""
    dates = sorted(panel[date_col].unique())
    present_features = [c for c in feature_cols if c in panel.columns]
    rows = []
    for end_idx in range(window_weeks, len(dates) + 1):
        window_dates = set(dates[end_idx - window_weeks : end_idx])
        subset = panel[panel[date_col].isin(window_dates)].copy()
        end_date = dates[end_idx - 1]

        if return_col not in subset.columns:
            continue
        valid = subset[[date_col, return_col] + present_features].dropna(
            subset=[return_col]
        )
        if len(valid) < 100:
            continue

        try:
            summary, _ = fama_macbeth(
                valid,
                return_col=return_col,
                feature_cols=present_features,
                date_col=date_col,
                standardize=True,
                nw_lags=4,
                prune_collinear=True,
            )
            for _, row in summary.iterrows():
                rows.append({
                    "end_date": end_date,
                    "window_weeks": window_weeks,
                    "feature": row["feature"],
                    "return_col": return_col,
                    "mean_beta": round(row["mean_beta"], 6),
                    "t_stat": round(row["t_stat"], 4),
                    "n_valid_weeks": int(row["n_valid_weeks"]),
                })
        except Exception as e:
            logger.warning(f"FMB failed for window ending {end_date}: {e}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Pre-M3 diagnostics")
    p.add_argument("--panel", required=True, help="Path to M1 panel csv.gz")
    p.add_argument("--outdir", default="output/diagnostics_pre_m3")
    p.add_argument("--date-col", default="rebalance_date")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading panel from {args.panel}...")
    if args.panel.endswith(".csv.gz"):
        panel = pd.read_csv(args.panel, compression="gzip")
    else:
        panel = pd.read_csv(args.panel)
    panel[args.date_col] = panel[args.date_col].astype(str)
    panel["ticker"] = panel["ticker"].astype(str).str.upper().str.strip()
    panel = panel.sort_values([args.date_col, "ticker"]).reset_index(drop=True)

    dates = sorted(panel[args.date_col].unique())
    features_present = [c for c in _FEATURE_COLS if c in panel.columns]
    returns_present = [c for c in _RETURN_COLS if c in panel.columns]

    logger.info(
        f"  Panel: {len(panel)} rows, {len(dates)} dates, "
        f"{panel['ticker'].nunique()} tickers"
    )
    logger.info(f"  Features: {features_present}")
    logger.info(f"  Returns: {returns_present}")

    # 1. Coverage
    logger.info("Computing coverage report...")
    cov = coverage_report(panel, features_present, args.date_col)
    cov_path = outdir / "coverage_report.csv"
    cov.to_csv(cov_path, index=False)
    logger.info(f"  Wrote {cov_path}")
    for _, row in cov.iterrows():
        flag = " *** LOW" if row["pct_weeks_constant"] > 80 else ""
        logger.info(
            f"    {row['feature']:30s}  "
            f"nonnull={row['pct_nonnull_overall']:5.1f}%  "
            f"present={row['pct_weeks_present']:5.1f}%  "
            f"constant={row['pct_weeks_constant']:5.1f}%{flag}"
        )

    # 2. Subperiod sign stability
    logger.info("Computing subperiod sign stability (H1 vs H2)...")
    stability = subperiod_sign_stability(
        panel, features_present, returns_present, args.date_col
    )
    stab_path = outdir / "subperiod_sign_stability.csv"
    stability.to_csv(stab_path, index=False)
    logger.info(f"  Wrote {stab_path}")
    for _, row in stability.iterrows():
        match = "YES" if row["sign_match"] else " NO"
        ic1 = row.get("mean_ic_h1", float("nan"))
        ic2 = row.get("mean_ic_h2", float("nan"))
        ic1_s = f"{ic1:+.4f}" if pd.notna(ic1) else "   N/A"
        ic2_s = f"{ic2:+.4f}" if pd.notna(ic2) else "   N/A"
        logger.info(
            f"    {row['feature']:30s} {row['return_col']:10s}  "
            f"H1={ic1_s}  H2={ic2_s}  match={match}"
        )

    # 3. Rolling IC (26-week and 52-week)
    for window in [26, 52]:
        logger.info(f"Computing {window}-week rolling IC...")
        ric = rolling_ic(
            panel, features_present, returns_present, window, args.date_col
        )
        ric_path = outdir / f"rolling_ic_{window}w.csv.gz"
        ric.to_csv(ric_path, index=False, compression="gzip")
        logger.info(f"  Wrote {ric_path} ({len(ric)} rows)")

        # Summary: min/max mean_ic and sign-flip count per feature×horizon
        if not ric.empty:
            for (feat, rcol), grp in ric.groupby(["feature", "return_col"]):
                vals = grp["mean_ic"].values
                sign_changes = int(np.sum(np.diff(np.sign(vals)) != 0))
                logger.info(
                    f"    {feat:30s} {rcol:10s}  "
                    f"min={vals.min():+.4f}  max={vals.max():+.4f}  "
                    f"sign_flips={sign_changes}"
                )

    # 4. Rolling FMB (52-week on fwd_21d)
    logger.info("Computing 52-week rolling FMB (fwd_21d)...")
    rfmb = rolling_fmb(
        panel, features_present, "fwd_21d", 52, args.date_col
    )
    rfmb_path = outdir / "rolling_fmb_52w_fwd21d.csv.gz"
    rfmb.to_csv(rfmb_path, index=False, compression="gzip")
    logger.info(f"  Wrote {rfmb_path} ({len(rfmb)} rows)")

    if not rfmb.empty:
        for feat, grp in rfmb.groupby("feature"):
            betas = grp["mean_beta"].values
            tstats = grp["t_stat"].values
            sign_changes = int(np.sum(np.diff(np.sign(betas)) != 0))
            sig_count = int(np.sum(np.abs(tstats) > 1.96))
            logger.info(
                f"    {feat:30s}  "
                f"beta=[{betas.min():+.4f}, {betas.max():+.4f}]  "
                f"sign_flips={sign_changes}  sig_windows={sig_count}/{len(tstats)}"
            )

    # Also run FMB on fwd_5d
    logger.info("Computing 52-week rolling FMB (fwd_5d)...")
    rfmb5 = rolling_fmb(
        panel, features_present, "fwd_5d", 52, args.date_col
    )
    rfmb5_path = outdir / "rolling_fmb_52w_fwd5d.csv.gz"
    rfmb5.to_csv(rfmb5_path, index=False, compression="gzip")
    logger.info(f"  Wrote {rfmb5_path} ({len(rfmb5)} rows)")

    if not rfmb5.empty:
        for feat, grp in rfmb5.groupby("feature"):
            betas = grp["mean_beta"].values
            tstats = grp["t_stat"].values
            sign_changes = int(np.sum(np.diff(np.sign(betas)) != 0))
            sig_count = int(np.sum(np.abs(tstats) > 1.96))
            logger.info(
                f"    {feat:30s}  "
                f"beta=[{betas.min():+.4f}, {betas.max():+.4f}]  "
                f"sign_flips={sign_changes}  sig_windows={sig_count}/{len(tstats)}"
            )

    logger.info("============================================================")
    logger.info("Pre-M3 diagnostics complete")
    logger.info(f"  Output: {outdir}")


if __name__ == "__main__":
    main()
