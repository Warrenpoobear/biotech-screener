#!/usr/bin/env python3
"""
Backfill Module 4 clinical feature cache per business day.

Usage:
    python3 scripts/backfill_clinical_cache.py --from-date 2026-01-15 --to-date 2026-02-16

Produces: cache/clinical/clinical_features_{YYYY-MM-DD}.json per business day.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

# Allow importing project root modules when run as script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from module_4_clinical_dev import compute_module_4_clinical_dev

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _business_days(start: date, end: date):
    """Yield business days (Mon-Fri) in [start, end] inclusive."""
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


def _load_json(path: Path) -> list | dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pit_filter_trials(trials: list[dict], as_of_date_str: str) -> list[dict]:
    """Conservative inline PIT filter: keep trials where last_update_posted <= as_of_date."""
    kept = []
    for t in trials:
        lup = (t.get("last_update_posted") or "")[:10]
        if lup == "" or lup <= as_of_date_str:
            kept.append(t)
    return kept


def _serialize(obj: dict) -> str:
    """Deterministic JSON serialization."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def backfill_one_date(
    as_of: date,
    data_dir: Path,
    cache_dir: Path,
    skip_existing: bool = True,
) -> dict | None:
    """
    Compute and cache clinical features for a single date.

    Returns a summary dict or None if skipped.
    """
    as_of_str = as_of.isoformat()
    out_dir = cache_dir / "clinical"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"clinical_features_{as_of_str}.json"

    if skip_existing and out_path.exists():
        logger.info(f"  {as_of_str}: cached — skip")
        return None

    # --- resolve trial_records ---
    ctgov_cache = cache_dir / "ctgov" / f"trial_records_{as_of_str}.json"
    if ctgov_cache.exists():
        trials_raw = _load_json(ctgov_cache)
        try:
            source = str(ctgov_cache.relative_to(_PROJECT_ROOT))
        except ValueError:
            source = str(ctgov_cache)
        # CTGov cache is already PIT-filtered; pass through
        trials = trials_raw
    else:
        prod_path = data_dir / "trial_records.json"
        trials_raw = _load_json(prod_path)
        try:
            source = str(prod_path.relative_to(_PROJECT_ROOT))
        except ValueError:
            source = str(prod_path)
        trials = _pit_filter_trials(trials_raw, as_of_str)

    # --- load universe tickers ---
    universe = _load_json(data_dir / "universe.json")
    tickers = [rec["ticker"] for rec in universe if isinstance(rec, dict) and "ticker" in rec]

    # --- compute Module 4 ---
    m4_result = compute_module_4_clinical_dev(
        trial_records=trials,
        active_tickers=tickers,
        as_of_date=as_of_str,
    )

    # --- wrap with backfill provenance ---
    output = {
        "as_of_date": as_of_str,
        "provenance": {
            "trial_records_source": source,
            "pit_filter": {
                "field": "last_update_posted",
                "inclusive_cutoff": as_of_str,
            },
            "module4_provenance": m4_result.get("provenance", {}),
        },
        "diagnostic_counts": {
            "tickers": m4_result["diagnostic_counts"].get("scored", 0),
            "trials_raw": m4_result["diagnostic_counts"].get("total_trials_raw", 0),
            "trials_after_pit": m4_result["diagnostic_counts"].get("pit_filtered", 0),
        },
        "scores": m4_result["scores"],
    }

    # Note: trials_after_pit from m4 diagnostic_counts.pit_filtered is the count
    # of trials REMOVED by PIT. The actual count after PIT is total_trials_unique.
    # Fix the label to reflect the actual meaning.
    output["diagnostic_counts"]["trials_after_pit"] = m4_result["diagnostic_counts"].get(
        "total_trials_unique", 0
    )

    out_path.write_text(_serialize(output), encoding="utf-8")
    logger.info(f"  {as_of_str}: wrote {len(m4_result['scores'])} tickers")

    return {
        "date": as_of_str,
        "tickers": output["diagnostic_counts"]["tickers"],
        "trials_raw": output["diagnostic_counts"]["trials_raw"],
        "trials_after_pit": output["diagnostic_counts"]["trials_after_pit"],
        "source": source,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill clinical feature cache")
    parser.add_argument("--from-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--data-dir", default="production_data", help="Data directory")
    parser.add_argument("--cache-dir", default="cache", help="Cache directory")
    parser.add_argument("--no-skip-existing", action="store_true", help="Recompute even if cached")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    start = date.fromisoformat(args.from_date)
    end = date.fromisoformat(args.to_date)
    data_dir = (_PROJECT_ROOT / args.data_dir).resolve()
    cache_dir = (_PROJECT_ROOT / args.cache_dir).resolve()
    skip_existing = not args.no_skip_existing

    logger.info(f"Backfilling clinical cache: {start} → {end}")
    logger.info(f"  data_dir={data_dir}, cache_dir={cache_dir}, skip_existing={skip_existing}")

    rows: list[dict] = []
    for d in _business_days(start, end):
        result = backfill_one_date(d, data_dir, cache_dir, skip_existing)
        if result:
            rows.append(result)

    # --- summary table ---
    if rows:
        print(f"\n{'date':<14} {'tickers':>8} {'trials_raw':>11} {'after_pit':>10}  source")
        print("-" * 75)
        for r in rows:
            src_short = r["source"].split("/")[-1][:30]
            print(
                f"{r['date']:<14} {r['tickers']:>8} {r['trials_raw']:>11} "
                f"{r['trials_after_pit']:>10}  {src_short}"
            )
        print(f"\nTotal: {len(rows)} dates backfilled")
    else:
        print("\nAll dates already cached (nothing to do)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
