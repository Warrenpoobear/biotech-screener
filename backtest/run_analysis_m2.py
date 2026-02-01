#!/usr/bin/env python3
"""
Milestone 2 Analysis Runner — Fama–MacBeth + Manager Scoring

Reads the M1 weekly panel (panel_weekly.csv.gz) and produces:
  - FMB summary (cross-sectional regression significance)
  - Manager leaderboard (Tier-1 stock-picker scoring)

Usage:
    python3 backtest/run_analysis_m2.py \
        --panel output/backtest_m1/panel_weekly.csv.gz \
        --data-dir production_data \
        --outdir output/backtest_m2 \
        --return-col fwd_21d

    # FMB only (no manager data needed):
    python3 backtest/run_analysis_m2.py \
        --panel output/backtest_m1/panel_weekly.csv.gz \
        --outdir output/backtest_m2 \
        --fmb-only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Ensure project root on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backtest.fmb import fama_macbeth, write_fmb_outputs
from backtest.manager_scoring import (
    assign_holdings_to_weeks,
    build_manager_holdings_panel,
    build_manager_leaderboard,
    compute_manager_returns,
    load_manager_registry,
    write_manager_outputs,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("analysis_m2")

# Default feature columns to test in FMB (core component scores).
# Uses normalized scores (percentile rank, independent across components)
# rather than contribution columns (which are linear functions of
# normalized × weight and would create multicollinearity).
_DEFAULT_FEATURE_COLS = [
    "catalyst_normalized",
    "clinical_normalized",
    "composite_score",
    "financial_normalized",
    "momentum_normalized",
    "uncertainty_penalty",
    "valuation_normalized",
]

# Extended feature set includes raw scores (may be collinear with normalized)
_EXTENDED_FEATURE_COLS = [
    "catalyst_contribution",
    "catalyst_normalized",
    "catalyst_raw",
    "clinical_contribution",
    "clinical_normalized",
    "clinical_raw",
    "composite_score",
    "financial_contribution",
    "financial_normalized",
    "financial_raw",
    "momentum_contribution",
    "momentum_normalized",
    "momentum_raw",
    "uncertainty_penalty",
    "valuation_contribution",
    "valuation_normalized",
    "valuation_raw",
]

# Columns that are confidence measures — test separately if desired
_CONFIDENCE_COLS = [
    "catalyst_confidence",
    "clinical_confidence",
    "financial_confidence",
    "momentum_confidence",
    "valuation_confidence",
]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Milestone 2: Fama-MacBeth + Manager Scoring",
    )
    p.add_argument("--panel", required=True, help="Path to M1 panel (csv.gz or csv)")
    p.add_argument("--data-dir", default="production_data", help="Data directory with holdings + registry")
    p.add_argument("--outdir", default="output/backtest_m2", help="Output directory")
    p.add_argument("--return-col", default=None,
                   help="Forward return column (default: fwd_5d if present, else fwd_21d)")
    p.add_argument("--date-col", default="rebalance_date", help="Date column name")
    p.add_argument("--nw-lags", type=int, default=4, help="Newey-West lags (default: 4)")
    p.add_argument("--no-standardize", action="store_true", help="Skip z-scoring features")
    p.add_argument("--fmb-only", action="store_true", help="Skip manager scoring")
    p.add_argument("--feature-cols", nargs="+", default=None,
                   help="Override feature columns (default: auto-detect from panel)")
    p.add_argument("--include-confidence", action="store_true",
                   help="Include confidence columns in FMB features")
    p.add_argument("--extended-features", action="store_true",
                   help="Use extended feature set (raw + contribution + normalized)")
    p.add_argument("--min-manager-weeks", type=int, default=3,
                   help="Minimum weeks for manager to appear in leaderboard")
    return p.parse_args(argv)


def load_panel(path: str, date_col: str = "rebalance_date") -> pd.DataFrame:
    """Load panel from csv.gz or csv, enforce types, sort deterministically."""
    p = Path(path)
    if p.suffix == ".gz" or str(p).endswith(".csv.gz"):
        df = pd.read_csv(p, compression="gzip")
    else:
        df = pd.read_csv(p)

    if date_col in df.columns:
        df[date_col] = df[date_col].astype(str)

    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    df = df.sort_values([date_col, "ticker"]).reset_index(drop=True)
    return df


def detect_feature_cols(
    panel: pd.DataFrame,
    include_confidence: bool = False,
    extended: bool = False,
) -> List[str]:
    """Auto-detect feature columns present in the panel."""
    candidates = _EXTENDED_FEATURE_COLS[:] if extended else _DEFAULT_FEATURE_COLS[:]
    if include_confidence:
        candidates.extend(_CONFIDENCE_COLS)

    present = [c for c in sorted(set(candidates)) if c in panel.columns]
    return present


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir).resolve()

    t0 = time.monotonic()

    # ── Load panel ──────────────────────────────────────────────────────
    logger.info(f"Loading panel from {args.panel}...")
    panel = load_panel(args.panel, args.date_col)
    n_dates = panel[args.date_col].nunique()
    n_tickers = panel["ticker"].nunique()
    logger.info(f"  Panel: {len(panel)} rows, {n_dates} dates, {n_tickers} tickers")

    # ── Determine feature columns ──────────────────────────────────────
    if args.feature_cols:
        feature_cols = sorted(args.feature_cols)
    else:
        feature_cols = detect_feature_cols(panel, args.include_confidence, args.extended_features)
    logger.info(f"  Feature columns ({len(feature_cols)}): {feature_cols}")

    if not feature_cols:
        logger.error("No feature columns found in panel. Cannot run FMB.")
        return 1

    # ── Part A: Fama–MacBeth ────────────────────────────────────────────
    logger.info("Running Fama-MacBeth...")

    # Auto-detect primary return column
    return_col = args.return_col
    if return_col is None:
        for candidate in ["fwd_5d", "fwd_21d"]:
            if candidate in panel.columns and panel[candidate].notna().sum() > 0:
                return_col = candidate
                break
        if return_col is None:
            logger.error("No usable return column found in panel")
            return 1
    elif return_col not in panel.columns:
        logger.error(f"Return column '{return_col}' not in panel columns: {list(panel.columns)}")
        return 1
    logger.info(f"  Primary return column: {return_col}")

    # Run FMB for each return horizon present
    return_cols_to_test = []
    for rc in ["fwd_5d", "fwd_21d", "fwd_63d", "fwd_126d"]:
        if rc in panel.columns and panel[rc].notna().sum() > 0:
            return_cols_to_test.append(rc)

    if not return_cols_to_test:
        return_cols_to_test = [return_col]

    all_summaries = []
    all_betas = []

    for rc in return_cols_to_test:
        logger.info(f"  FMB for {rc}...")
        summary, betas = fama_macbeth(
            panel=panel,
            return_col=rc,
            feature_cols=feature_cols,
            date_col=args.date_col,
            standardize=not args.no_standardize,
            nw_lags=args.nw_lags,
        )
        if not summary.empty:
            summary["return_col"] = rc
            all_summaries.append(summary)
        if not betas.empty:
            betas["return_col"] = rc
            all_betas.append(betas)

    if all_summaries:
        combined_summary = pd.concat(all_summaries, ignore_index=True)
        combined_summary = combined_summary.sort_values(
            ["return_col", "feature"]
        ).reset_index(drop=True)
    else:
        combined_summary = pd.DataFrame()

    if all_betas:
        combined_betas = pd.concat(all_betas, ignore_index=True)
    else:
        combined_betas = pd.DataFrame()

    fmb_paths = write_fmb_outputs(
        summary_df=combined_summary,
        betas_df=combined_betas,
        outdir=outdir,
        return_col=",".join(return_cols_to_test),
        feature_cols=feature_cols,
        nw_lags=args.nw_lags,
        standardize=not args.no_standardize,
    )
    logger.info(f"  FMB summary: {fmb_paths['summary']}")

    if not combined_summary.empty:
        # Print top significant features
        sig = combined_summary[combined_summary["t_stat"].abs() > 1.96]
        if not sig.empty:
            logger.info(f"  Significant features (|t| > 1.96): {len(sig)}")
            for _, row in sig.iterrows():
                logger.info(
                    f"    {row['feature']:30s} [{row['return_col']}] "
                    f"beta={row['mean_beta']:+.4f}  t={row['t_stat']:+.2f}"
                )
        else:
            logger.info("  No features significant at 5% level")

    # ── Part B: Manager Scoring ─────────────────────────────────────────
    if args.fmb_only:
        logger.info("Skipping manager scoring (--fmb-only)")
    else:
        holdings_path = data_dir / "holdings_snapshots.json"
        registry_path = data_dir / "manager_registry.json"

        if not holdings_path.exists():
            logger.warning(f"Holdings file not found: {holdings_path}. Skipping manager scoring.")
        elif not registry_path.exists():
            logger.warning(f"Registry file not found: {registry_path}. Skipping manager scoring.")
        else:
            logger.info("Running manager scoring...")

            # Load Tier-1 managers
            manager_ciks = load_manager_registry(registry_path, tier="elite_core")
            logger.info(f"  Tier-1 managers: {len(manager_ciks)}")

            # Build holdings panel
            holdings_df = build_manager_holdings_panel(
                holdings_path, manager_ciks,
            )
            logger.info(f"  Holdings panel: {len(holdings_df)} rows")

            if holdings_df.empty:
                logger.warning("  No manager holdings found. Skipping.")
            else:
                # Get rebalance dates from panel
                rebalance_dates = sorted(panel[args.date_col].unique().tolist())

                # Map holdings to weekly dates
                weekly_holdings = assign_holdings_to_weeks(holdings_df, rebalance_dates)
                logger.info(f"  Weekly holdings: {len(weekly_holdings)} rows")

                if weekly_holdings.empty:
                    logger.warning("  No holdings mapped to rebalance dates (PIT filter). Skipping.")
                else:
                    # Use primary return column for manager scoring
                    mgr_return_col = return_col
                    if mgr_return_col not in panel.columns or panel[mgr_return_col].notna().sum() == 0:
                        # Fallback to first available
                        for rc in ["fwd_5d", "fwd_21d", "fwd_63d", "fwd_126d"]:
                            if rc in panel.columns and panel[rc].notna().sum() > 0:
                                mgr_return_col = rc
                                break

                    manager_returns = compute_manager_returns(
                        weekly_holdings=weekly_holdings,
                        panel=panel,
                        return_col=mgr_return_col,
                        date_col=args.date_col,
                    )
                    logger.info(f"  Manager returns: {len(manager_returns)} rows")

                    leaderboard = build_manager_leaderboard(
                        manager_returns,
                        min_weeks=args.min_manager_weeks,
                        nw_lags=args.nw_lags,
                        date_col=args.date_col,
                    )
                    logger.info(f"  Leaderboard: {len(leaderboard)} managers ranked")

                    mgr_paths = write_manager_outputs(leaderboard, manager_returns, outdir)
                    logger.info(f"  Manager ranking: {mgr_paths['ranking']}")

                    if not leaderboard.empty:
                        logger.info("  Top 5 managers:")
                        for _, row in leaderboard.head(5).iterrows():
                            logger.info(
                                f"    #{row['rank']} {row['manager_name']:35s} "
                                f"ret={row['avg_return']:+.4f}  "
                                f"t={row['t_stat']:+.2f}  "
                                f"hit={row['hit_rate']:.0%}  "
                                f"weeks={row['n_weeks']}  "
                                f"holdings={row['avg_holdings']:.0f}"
                            )

    # ── Summary ─────────────────────────────────────────────────────────
    elapsed = time.monotonic() - t0
    logger.info("=" * 60)
    logger.info(f"Analysis M2 complete ({elapsed:.1f}s)")
    logger.info(f"  Output: {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
