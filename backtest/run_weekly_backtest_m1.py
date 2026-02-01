#!/usr/bin/env python3
"""
Milestone 1 Weekly Backtest Harness — Panel + Rank IC

Builds a 2-year weekly panel of component scores and forward raw returns,
then computes Spearman rank IC for each component vs each horizon.

Usage:
    python3 backtest/run_weekly_backtest_m1.py \
        --start-date 2024-02-02 \
        --end-date 2026-02-01 \
        --data-dir production_data \
        --out-dir output/backtest_m1 \
        --pit-mode degrade

Outputs:
    panel_weekly.csv.gz     - full panel dataset
    ic_summary.csv          - mean IC, hit rate per component x horizon
    run_metadata.json       - run config + diagnostics
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path

# Ensure project root on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backtest.panel_builder_m1 import (
    build_panel,
    build_trading_day_index,
    generate_rebalance_dates,
    load_price_history,
    write_panel_csv_gz,
)
from backtest.metrics_m1 import (
    compute_ic_timeseries,
    summarize_ic,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest_m1")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Milestone 1 weekly backtest: panel + rank IC",
    )
    p.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    p.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    p.add_argument("--data-dir", default="production_data", help="Input data directory")
    p.add_argument("--out-dir", default="output/backtest_m1", help="Output directory")
    p.add_argument(
        "--pit-mode",
        choices=["strict", "degrade"],
        default="degrade",
        help="Point-in-time mode (default: degrade)",
    )
    p.add_argument("--weekday", default="FRI", help="Rebalance weekday (default: FRI)")
    p.add_argument(
        "--max-dates",
        type=int,
        default=None,
        help="Process only first N rebalance dates (for dev runs)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()

    # ── Step 1: Load price history ──────────────────────────────────────
    logger.info("Loading price history...")
    price_file = data_dir / "price_history.csv"
    if not price_file.exists():
        # Fallback to data/daily_prices.csv
        alt = _PROJECT_ROOT / "data" / "daily_prices.csv"
        if alt.exists():
            price_file = alt
        else:
            logger.error(f"No price file found at {price_file} or {alt}")
            return 1

    prices = load_price_history(price_file)
    trading_days = build_trading_day_index(prices)
    logger.info(
        f"  Loaded {len(prices)} tickers, "
        f"{len(trading_days)} trading days "
        f"({trading_days[0]} to {trading_days[-1]})"
    )

    # ── Step 2: Generate rebalance schedule ─────────────────────────────
    trading_day_set = set(trading_days)
    rebalance_dates = generate_rebalance_dates(
        start, end, weekday=args.weekday, trading_days=trading_day_set,
    )
    logger.info(f"  {len(rebalance_dates)} rebalance dates generated")

    # ── Step 3: Build panel ─────────────────────────────────────────────
    cache_dir = out_dir / "cache"
    rows, score_cols = build_panel(
        rebalance_dates=rebalance_dates,
        data_dir=data_dir,
        prices=prices,
        trading_days=trading_days,
        pit_mode=args.pit_mode,
        cache_dir=cache_dir,
        max_dates=args.max_dates,
    )
    logger.info(f"  Panel: {len(rows)} rows, {len(score_cols)} score columns")

    # ── Step 4: Write panel ─────────────────────────────────────────────
    panel_path = out_dir / "panel_weekly.csv.gz"
    n_written = write_panel_csv_gz(rows, score_cols, panel_path)
    logger.info(f"  Wrote {n_written} rows to {panel_path}")

    # ── Step 5: Compute IC ──────────────────────────────────────────────
    return_cols = ["fwd_5d", "fwd_21d", "fwd_63d", "fwd_126d"]
    # Filter score_cols to only those with variance (skip rank/metadata)
    ic_score_cols = [c for c in score_cols if c != "composite_rank"]

    ic_ts = compute_ic_timeseries(rows, ic_score_cols, return_cols)
    ic_summary = summarize_ic(ic_ts)

    # Write IC summary
    ic_path = out_dir / "ic_summary.csv"
    if ic_summary:
        with open(ic_path, "w", encoding="utf-8", newline="") as f:
            fieldnames = ["score_col", "return_col", "mean_ic", "median_ic", "hit_rate", "n_dates"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in ic_summary:
                writer.writerow(row)
        logger.info(f"  Wrote {len(ic_summary)} IC summary rows to {ic_path}")
    else:
        logger.warning("  No IC summary rows produced")

    # ── Step 6: Write metadata ──────────────────────────────────────────
    elapsed = time.monotonic() - t0
    metadata = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "data_dir": str(data_dir),
        "pit_mode": args.pit_mode,
        "weekday": args.weekday,
        "max_dates": args.max_dates,
        "n_rebalance_dates": len(rebalance_dates),
        "n_dates_processed": len(set(r["rebalance_date"] for r in rows)) if rows else 0,
        "n_panel_rows": len(rows),
        "score_columns": score_cols,
        "return_columns": return_cols,
        "n_tickers_in_prices": len(prices),
        "price_date_range": [trading_days[0], trading_days[-1]] if trading_days else [],
        "elapsed_seconds": round(elapsed, 1),
    }
    meta_path = out_dir / "run_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  Metadata written to {meta_path}")

    # ── Summary ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Backtest M1 complete")
    logger.info(f"  Panel: {panel_path}")
    logger.info(f"  IC:    {ic_path}")
    logger.info(f"  Meta:  {meta_path}")
    if ic_summary:
        logger.info("  Top ICs:")
        for row in sorted(ic_summary, key=lambda r: abs(r["mean_ic"]), reverse=True)[:5]:
            logger.info(
                f"    {row['score_col']:30s} vs {row['return_col']:10s} "
                f"IC={row['mean_ic']:+.4f}  hit={row['hit_rate']:.1%}  n={row['n_dates']}"
            )
    logger.info(f"  Elapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
