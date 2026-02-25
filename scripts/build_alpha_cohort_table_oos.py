#!/usr/bin/env python3
"""Build PIT-safe out-of-sample alpha cohort table.

Given an as_of_date, discovers archives strictly before that date, filters to
those with mature forward returns, applies a train mode (expanding / trailing-N
/ decay-H), and writes a JSON table in the same schema as v1.json.

This script is designed to be called from run_screen.py in-process via
``build_oos_table()`` or from the CLI for manual rebuilds.

Usage:
    python scripts/build_alpha_cohort_table_oos.py \
        --as-of-date 2026-02-24 \
        --output production_data/alpha_cohort_tables/daily/v1_2026-02-24.json \
        --horizon 84 --train-mode trailing-6 --min-train-dates 6
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Project imports (add project root first)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.ic_measurement import add_trading_days  # noqa: E402
from backtest.returns_provider import CSVReturnsProvider  # noqa: E402
from module_5_alpha_cohort import compute_alpha_cohort_key  # noqa: E402
from run_rank_ic_backtest import (  # noqa: E402
    ARCHIVE_DIR,
    PRICE_CSV,
    RETURNS_JSON,
    ChainedReturnsProvider,
    MorningstarReturnsProvider,
    compute_forward_returns,
    discover_archives,
)
from scripts.build_alpha_cohort_table import (  # noqa: E402
    ALL_KEYS,
    backfill_clinical_z_tier,
    load_rankings_dicts,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_SHRINK_K = 50
ALPHA_CLIP_MIN = -0.10
ALPHA_CLIP_MAX = 0.10


# ---------------------------------------------------------------------------
# Train mode parsing (reuse logic from backtest_signal_robustness.py)
# ---------------------------------------------------------------------------

def parse_train_mode(mode_str: str) -> Tuple[str, Optional[int]]:
    """Parse train mode string.

    Returns (mode, param) where mode is 'expanding', 'trailing', or 'decay'
    and param is the integer argument (None for expanding).
    """
    mode_str = mode_str.strip().lower()
    if mode_str == "expanding":
        return ("expanding", None)
    if mode_str.startswith("trailing-"):
        try:
            n = int(mode_str.split("-", 1)[1])
        except (ValueError, IndexError):
            raise ValueError(f"Invalid train mode: {mode_str!r} — expected trailing-N")
        if n < 1:
            raise ValueError(f"trailing window must be >= 1, got {n}")
        return ("trailing", n)
    if mode_str.startswith("decay-"):
        try:
            h = int(mode_str.split("-", 1)[1])
        except (ValueError, IndexError):
            raise ValueError(f"Invalid train mode: {mode_str!r} — expected decay-H")
        if h < 1:
            raise ValueError(f"decay half-life must be >= 1, got {h}")
        return ("decay", h)
    raise ValueError(f"Unknown train mode: {mode_str!r}")


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_oos_table(
    as_of_date: str,
    train_mode: str = "trailing-6",
    horizon: int = 84,
    min_train_dates: int = 6,
    shrink_k: float = DEFAULT_SHRINK_K,
    archive_dir: Optional[Path] = None,
    price_csv: Optional[Path] = None,
    returns_json: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Build a PIT-safe out-of-sample alpha cohort table.

    Args:
        as_of_date: Reference date (YYYY-MM-DD). Only archives strictly before
            this date are used for training.
        train_mode: Training window specification — "expanding", "trailing-N",
            or "decay-H".
        horizon: Forward return horizon in trading days (63, 84, or 126).
        min_train_dates: Minimum number of usable training dates required.
        shrink_k: James-Stein shrinkage parameter.
        archive_dir: Override default archive directory.
        price_csv: Override default price CSV path.
        returns_json: Override default Morningstar returns JSON path.

    Returns:
        Table dict (v1 schema) or None if insufficient training data.
    """
    _archive_dir = archive_dir or ARCHIVE_DIR
    _price_csv = price_csv or PRICE_CSV
    _returns_json = returns_json or RETURNS_JSON

    mode, param = parse_train_mode(train_mode)

    # Discover all archives
    all_archives = discover_archives(_archive_dir)
    log.info("Discovered %d total archives", len(all_archives))

    # PIT filter: archives strictly before as_of_date
    pit_archives = [(d, p) for d, p in all_archives if d < as_of_date]
    log.info("Archives before %s: %d", as_of_date, len(pit_archives))

    if not pit_archives:
        log.warning("No archives before %s — cannot build table", as_of_date)
        return None

    # Build returns provider
    ms_provider = MorningstarReturnsProvider(_returns_json)
    csv_provider = CSVReturnsProvider(_price_csv, price_col="close")
    provider = ChainedReturnsProvider(ms_provider, csv_provider)

    last_date = provider.get_last_date()
    if last_date:
        log.info("Last available price date: %s", last_date.isoformat())

    # Filter to archives with mature forward returns
    usable: List[Tuple[str, Path]] = []
    for date_str, tar_path in pit_archives:
        horizon_end = add_trading_days(date_str, horizon)
        if last_date and horizon_end > last_date.isoformat():
            log.debug(
                "Skipping %s — horizon end %s beyond price data", date_str, horizon_end
            )
            continue
        usable.append((date_str, tar_path))

    log.info(
        "Usable archives (before %s, %d-day fwd available): %d",
        as_of_date, horizon, len(usable),
    )

    if len(usable) < min_train_dates:
        log.warning(
            "Only %d usable dates (need %d) — cannot build table",
            len(usable), min_train_dates,
        )
        return None

    # Apply train mode windowing
    if mode == "trailing" and param is not None:
        # Keep only the last N dates
        usable = usable[-param:]
        log.info("Trailing-%d: using last %d dates", param, len(usable))

    # Pre-compute per-date data (rows + excess returns)
    train_cache: List[Dict[str, Any]] = []
    for date_str, tar_path in usable:
        rows = load_rankings_dicts(tar_path)
        if not rows:
            log.warning("  No rankings.csv in %s — skipping", tar_path.name)
            continue

        backfill_clinical_z_tier(rows)
        for row in rows:
            row["alpha_cohort_key"] = compute_alpha_cohort_key(row)

        tickers = [r["ticker"] for r in rows if r.get("ticker")]
        fwd_rets = compute_forward_returns(provider, tickers, date_str, horizon)

        if len(fwd_rets) < 5:
            log.warning("  %s: too few fwd returns (%d) — skipping", date_str, len(fwd_rets))
            continue

        median_ret = statistics.median(fwd_rets.values())
        excess = {t: r - median_ret for t, r in fwd_rets.items()}

        train_cache.append({
            "date": date_str,
            "rows": rows,
            "excess": excess,
        })
        log.info(
            "  %s: %d tickers, %d with fwd returns",
            date_str, len(tickers), len(fwd_rets),
        )

    if len(train_cache) < min_train_dates:
        log.warning(
            "Only %d valid dates after loading (need %d) — cannot build table",
            len(train_cache), min_train_dates,
        )
        return None

    # Build table based on mode
    if mode == "decay" and param is not None:
        # Exponential decay weights: w_i = exp(-i * ln(2) / H) where i = distance from newest
        n_dates = len(train_cache)
        half_life = param
        weights = [
            math.exp(-(n_dates - 1 - i) * math.log(2) / half_life)
            for i in range(n_dates)
        ]
        table = _build_weighted_table(train_cache, weights, shrink_k)
        log.info("Built decay-%d table from %d dates", half_life, n_dates)
    else:
        # expanding or trailing: equal weight
        table = _build_equal_weight_table(train_cache, shrink_k)
        log.info("Built %s table from %d dates", train_mode, len(train_cache))

    # Add schema metadata
    table["_schema"] = {"name": "alpha_cohort_table", "version": "1"}
    table["_build_info"] = {
        "as_of_date": as_of_date,
        "train_mode": train_mode,
        "horizon": horizon,
        "n_train_dates": len(train_cache),
        "train_dates": [e["date"] for e in train_cache],
    }

    return table


def _build_equal_weight_table(
    train_cache: List[Dict[str, Any]],
    shrink_k: float,
) -> Dict[str, Any]:
    """Build table with equal-weighted cell means across all dates."""
    cell_returns: Dict[str, List[float]] = defaultdict(list)

    for entry in train_cache:
        rows = entry["rows"]
        excess = entry["excess"]
        for row in rows:
            tk = (row.get("ticker") or "").strip()
            if tk not in excess:
                continue
            key = compute_alpha_cohort_key(row)
            cell_returns[key].append(excess[tk])

    cells: Dict[str, Dict[str, Any]] = {}
    for key in ALL_KEYS:
        rets = cell_returns.get(key, [])
        if rets:
            cells[key] = {
                "mean_excess_ret_6m": round(statistics.mean(rets), 6),
                "n": len(rets),
            }
        else:
            cells[key] = {"mean_excess_ret_6m": 0.0, "n": 0}

    return {
        "cells": cells,
        "shrink_k_default": shrink_k,
        "alpha_clip": {"min": ALPHA_CLIP_MIN, "max": ALPHA_CLIP_MAX},
    }


def _build_weighted_table(
    train_cache: List[Dict[str, Any]],
    weights: List[float],
    shrink_k: float,
) -> Dict[str, Any]:
    """Build table with weighted cell means (for decay mode)."""
    cell_w_sums: Dict[str, float] = defaultdict(float)
    cell_wr_sums: Dict[str, float] = defaultdict(float)
    cell_counts: Dict[str, float] = defaultdict(float)

    for entry, w in zip(train_cache, weights):
        if w <= 0.0:
            continue
        rows = entry["rows"]
        excess = entry["excess"]
        for row in rows:
            tk = (row.get("ticker") or "").strip()
            if tk not in excess:
                continue
            key = compute_alpha_cohort_key(row)
            cell_wr_sums[key] += w * excess[tk]
            cell_w_sums[key] += w
            cell_counts[key] += w

    cells: Dict[str, Dict[str, Any]] = {}
    for key in ALL_KEYS:
        ws = cell_w_sums.get(key, 0.0)
        if ws > 0:
            cells[key] = {
                "mean_excess_ret_6m": round(cell_wr_sums[key] / ws, 6),
                "n": cell_counts[key],
            }
        else:
            cells[key] = {"mean_excess_ret_6m": 0.0, "n": 0}

    return {
        "cells": cells,
        "shrink_k_default": shrink_k,
        "alpha_clip": {"min": ALPHA_CLIP_MIN, "max": ALPHA_CLIP_MAX},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build PIT-safe out-of-sample alpha cohort table."
    )
    parser.add_argument(
        "--as-of-date", required=True,
        help="Reference date (YYYY-MM-DD). Archives strictly before this date are used.",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Output JSON path (default: production_data/alpha_cohort_tables/daily/v1_<date>.json)",
    )
    parser.add_argument("--horizon", type=int, default=84, help="Forward horizon in trading days")
    parser.add_argument(
        "--train-mode", default="trailing-6",
        help="Training window: expanding | trailing-N | decay-H",
    )
    parser.add_argument(
        "--min-train-dates", type=int, default=6,
        help="Minimum usable training dates required (default: 6)",
    )
    parser.add_argument("--shrink-k", type=float, default=DEFAULT_SHRINK_K, help="Shrinkage parameter")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Default output path
    if args.output is None:
        args.output = (
            PROJECT_ROOT / "production_data" / "alpha_cohort_tables"
            / "daily" / f"v1_{args.as_of_date}.json"
        )

    table = build_oos_table(
        as_of_date=args.as_of_date,
        train_mode=args.train_mode,
        horizon=args.horizon,
        min_train_dates=args.min_train_dates,
        shrink_k=args.shrink_k,
    )

    if table is None:
        log.error("Could not build table — insufficient data")
        sys.exit(1)

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)
        f.write("\n")
    log.info("Wrote %s", args.output)

    # Summary
    cells = table.get("cells", {})
    populated = sum(1 for c in cells.values() if c.get("n", 0) > 0)
    log.info("Cell coverage: %d / %d populated", populated, len(ALL_KEYS))


if __name__ == "__main__":
    main()
