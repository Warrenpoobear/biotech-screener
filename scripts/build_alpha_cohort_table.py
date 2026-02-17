#!/usr/bin/env python3
"""Build alpha cohort lookup table from historical archive data.

Iterates backtest archives, assigns each ticker to an alpha cohort cell
(stage × catalyst horizon × clinical z sign), computes 6-month forward
excess returns, pools per cell, and writes a populated v1.json table.

Usage:
    python scripts/build_alpha_cohort_table.py \
        --output production_data/alpha_cohort_tables/v1.json \
        --horizon 126 --start 2024-01-31 --end 2025-07-31
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import statistics
import sys
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.ic_measurement import add_trading_days
from backtest.returns_provider import CSVReturnsProvider
from module_5_alpha_cohort import compute_alpha_cohort_key
from run_rank_ic_backtest import (
    ARCHIVE_DIR,
    PRICE_CSV,
    RETURNS_JSON,
    ChainedReturnsProvider,
    MorningstarReturnsProvider,
    compute_forward_returns,
    discover_archives,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# All 36 valid cohort keys (3 stages × 6 horizons × 2 signs)
# ---------------------------------------------------------------------------
_STAGES = ("early", "mid", "late")
_HORIZONS = (
    "near_0_30", "near_31_90", "near_91_180",
    "near_181_270", "far_271_540", "none",
)
_SIGNS = ("pos", "nonpos")

ALL_KEYS = [
    f"{s}|{h}|{sg}" for s in _STAGES for h in _HORIZONS for sg in _SIGNS
]
assert len(ALL_KEYS) == 36

# Archetypes treated as drug developers for z-score grouping
_DD_ARCHETYPES = {"drug_developer"}
_COMM_ARCHETYPES = {"commercial_biotech", "commercial_pharma"}


# ---------------------------------------------------------------------------
# Archive reader
# ---------------------------------------------------------------------------
def load_rankings_dicts(tar_path: Path) -> List[Dict[str, str]]:
    """Read rankings.csv from a tar.gz archive as raw string dicts."""
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith("/rankings.csv"):
                f = io.TextIOWrapper(tar.extractfile(member), encoding="utf-8")
                return list(csv.DictReader(f))
    return []


# ---------------------------------------------------------------------------
# Clinical z-tier backfill (mirrors compare_rulesets_replay.py logic)
# ---------------------------------------------------------------------------
def backfill_clinical_z_tier(rows: List[Dict[str, str]]) -> None:
    """Backfill clinical_score_z_tier in-place using ddof=0 z-scoring.

    Drug developers: z-score clinical_score within each tier_dev group.
    Commercial / platform / other: set to 0.0.
    """
    # Group drug developers by tier_dev
    dd_by_tier: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for i, row in enumerate(rows):
        arch = (row.get("archetype") or "").strip()
        if arch not in _DD_ARCHETYPES:
            row["clinical_score_z_tier"] = "0.0"
            continue
        tier = (row.get("tier_dev") or "").strip()
        if not tier:
            row["clinical_score_z_tier"] = "0.0"
            continue
        raw = (row.get("clinical_score") or "").strip()
        if not raw:
            row["clinical_score_z_tier"] = "0.0"
            continue
        try:
            score = float(raw)
        except (ValueError, TypeError):
            row["clinical_score_z_tier"] = "0.0"
            continue
        dd_by_tier[tier].append((i, score))

    for tier, members in dd_by_tier.items():
        if len(members) < 2:
            for idx, _ in members:
                rows[idx]["clinical_score_z_tier"] = "0.0"
            continue
        scores = [s for _, s in members]
        mu = statistics.mean(scores)
        # Population std (ddof=0)
        std = statistics.pstdev(scores)
        if std > 0:
            for idx, s in members:
                z = round((s - mu) / std, 4)
                rows[idx]["clinical_score_z_tier"] = str(z)
        else:
            for idx, _ in members:
                rows[idx]["clinical_score_z_tier"] = "0.0"


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------
def build_table(
    archives: List[Tuple[str, Path]],
    provider: Any,
    horizon: int,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Build the alpha cohort table from archive data.

    Returns the full table dict ready for JSON serialization.
    """
    cell_returns: Dict[str, List[float]] = defaultdict(list)
    total_obs = 0
    total_missing = 0

    for date_str, tar_path in archives:
        log.info("Processing archive %s ...", date_str)
        rows = load_rankings_dicts(tar_path)
        if not rows:
            log.warning("  No rankings.csv in %s — skipping", tar_path.name)
            continue

        # Backfill clinical_score_z_tier (no archive has this column)
        backfill_clinical_z_tier(rows)

        # Assign cohort keys
        for row in rows:
            row["alpha_cohort_key"] = compute_alpha_cohort_key(row)

        # Compute forward returns
        tickers = [r["ticker"] for r in rows if r.get("ticker")]
        fwd_rets = compute_forward_returns(provider, tickers, date_str, horizon)

        n_with_ret = len(fwd_rets)
        n_missing = len(tickers) - n_with_ret
        log.info(
            "  %s: %d tickers, %d with fwd returns, %d missing",
            date_str, len(tickers), n_with_ret, n_missing,
        )

        if n_with_ret < 5:
            log.warning("  Too few returns (%d) — skipping date", n_with_ret)
            continue

        # Cross-sectional median for excess return
        median_ret = statistics.median(fwd_rets.values())
        log.info("  Cross-sectional median return: %.4f", median_ret)

        excess = {t: r - median_ret for t, r in fwd_rets.items()}

        # Pool per cell
        date_obs = 0
        for row in rows:
            ticker = row.get("ticker", "")
            if ticker in excess:
                cell_returns[row["alpha_cohort_key"]].append(excess[ticker])
                date_obs += 1

        total_obs += date_obs
        total_missing += n_missing
        log.info("  Pooled %d observations", date_obs)

    # Build cells dict — ensure all 36 keys present
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

    # Log summary
    populated = sum(1 for c in cells.values() if c["n"] > 0)
    ns = [c["n"] for c in cells.values() if c["n"] > 0]
    log.info("=== SUMMARY ===")
    log.info("Total observations pooled: %d", total_obs)
    log.info("Total missing forward returns: %d", total_missing)
    log.info("Cell coverage: %d / %d populated", populated, len(ALL_KEYS))
    if ns:
        log.info(
            "Observations per populated cell: min=%d, median=%.0f, max=%d",
            min(ns), statistics.median(ns), max(ns),
        )

    # Top / bottom cells
    ranked = sorted(cells.items(), key=lambda kv: kv[1]["mean_excess_ret_6m"])
    if ranked:
        worst = ranked[0]
        best = ranked[-1]
        log.info(
            "Most negative cell: %s  mean=%.4f  n=%d",
            worst[0], worst[1]["mean_excess_ret_6m"], worst[1]["n"],
        )
        log.info(
            "Most positive cell: %s  mean=%.4f  n=%d",
            best[0], best[1]["mean_excess_ret_6m"], best[1]["n"],
        )

    table = {
        "_schema": {"name": "alpha_cohort_table", "version": "1"},
        "shrink_k_default": 50,
        "alpha_clip": {"min": -0.10, "max": 0.10},
        "cells": cells,
    }
    return table


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build alpha cohort lookup table from historical archives."
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "production_data" / "alpha_cohort_tables" / "v1.json",
        help="Output JSON path",
    )
    parser.add_argument("--horizon", type=int, default=126, help="Forward horizon in trading days")
    parser.add_argument("--start", type=str, default=None, help="Earliest archive date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="Latest archive date (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing")
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

    # Discover archives
    archives = discover_archives(ARCHIVE_DIR, start=args.start, end=args.end)
    log.info("Discovered %d archives in date range", len(archives))

    # Build returns provider
    log.info("Loading Morningstar HS793 returns from %s ...", RETURNS_JSON)
    ms_provider = MorningstarReturnsProvider(RETURNS_JSON)
    log.info("Loading CSV price data from %s ...", PRICE_CSV)
    csv_provider = CSVReturnsProvider(PRICE_CSV, price_col="close")
    provider = ChainedReturnsProvider(ms_provider, csv_provider)

    # Determine last available price date for horizon filtering
    last_date = provider.get_last_date()
    if last_date:
        log.info("Last available price date: %s", last_date.isoformat())

    # Filter archives: only keep those where horizon end date ≤ last price date
    usable = []
    for date_str, tar_path in archives:
        horizon_end = add_trading_days(date_str, args.horizon)
        if last_date and horizon_end > last_date.isoformat():
            log.debug("Skipping %s — horizon end %s beyond data", date_str, horizon_end)
            continue
        usable.append((date_str, tar_path))

    log.info("Usable archives (with full %d-day forward data): %d", args.horizon, len(usable))
    if not usable:
        log.error("No usable archives — check date range and price data coverage")
        sys.exit(1)

    for d, p in usable:
        log.info("  %s", d)

    # Build table
    table = build_table(usable, provider, args.horizon, dry_run=args.dry_run)

    if args.dry_run:
        log.info("Dry run — not writing output")
        print(json.dumps(table, indent=2))
        return

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)
        f.write("\n")
    log.info("Wrote %s", args.output)

    # Fallback stats
    stats = provider.get_fallback_stats()
    if stats["n_tickers_used_fallback"]:
        log.info(
            "CSV fallback used for %d tickers (%d lookups)",
            stats["n_tickers_used_fallback"], stats["total_fallback_lookups"],
        )
    overrides = provider.get_outlier_overrides()
    if overrides:
        log.warning("Outlier overrides: %d cases", len(overrides))
        for ov in overrides[:5]:
            log.warning(
                "  %s %s→%s: morningstar=%.2f, csv=%.2f",
                ov["ticker"], ov["start"], ov["end"],
                ov["morningstar_ret"], ov["csv_ret"],
            )


if __name__ == "__main__":
    main()
