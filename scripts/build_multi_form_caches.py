#!/usr/bin/env python3
"""Build SEC multi-form (10-Q, 10-K, 6-K) caches for all archive dates.

Reads data/archives/*.tar.gz to discover dates, then calls
collect_sec_filing_events() for each date that lacks an existing cache.
Results are written to wake_robin_data_pipeline/cache/sec/8k_catalysts/.

Usage:
    python scripts/build_multi_form_caches.py
    python scripts/build_multi_form_caches.py --dates 2025-06-30 2025-07-31
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector import (
    _multi_form_cache_path,
    collect_sec_filing_events,
)

ARCHIVES_DIR = PROJECT_ROOT / "data" / "archives"
CACHE_DIR = PROJECT_ROOT / "wake_robin_data_pipeline" / "cache" / "sec" / "8k_catalysts"
UNIVERSE_PATH = PROJECT_ROOT / "production_data" / "universe.json"


def discover_archive_dates() -> list:
    """Return sorted list of date objects from archive filenames."""
    dates = []
    for p in sorted(ARCHIVES_DIR.glob("*.tar.gz")):
        date_str = p.name.replace(".tar.gz", "")
        try:
            dates.append(datetime.strptime(date_str, "%Y-%m-%d").date())
        except ValueError:
            continue
    return dates


def main():
    parser = argparse.ArgumentParser(description="Build SEC multi-form caches for archive dates")
    parser.add_argument(
        "--dates", nargs="*",
        help="Specific dates to process (YYYY-MM-DD). Default: all archive dates.",
    )
    args = parser.parse_args()

    if args.dates:
        dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in args.dates]
    else:
        dates = discover_archive_dates()

    if not dates:
        print("No archive dates found.")
        return

    # Load universe
    if not UNIVERSE_PATH.exists():
        print(f"ERROR: {UNIVERSE_PATH} not found")
        sys.exit(1)

    with open(UNIVERSE_PATH, "r") as f:
        universe_entries = json.load(f)
    print(f"Universe: {len(universe_entries)} tickers")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    skipped = 0
    built = 0
    errors = 0

    for as_of in dates:
        cache_path = _multi_form_cache_path(CACHE_DIR, as_of)
        if cache_path.exists():
            print(f"  {as_of}  SKIP (cache exists: {cache_path.name})")
            skipped += 1
            continue

        print(f"  {as_of}  collecting...", end="", flush=True)
        try:
            events = collect_sec_filing_events(
                universe=universe_entries,
                as_of_date=as_of,
                cache_dir=CACHE_DIR,
            )
            tickers = {e.get("ticker") for e in events}
            print(f"  {len(events)} events, {len(tickers)} tickers")
            built += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            errors += 1

    print(f"\nDone: {built} built, {skipped} skipped, {errors} errors")


if __name__ == "__main__":
    main()
