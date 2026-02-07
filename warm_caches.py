#!/usr/bin/env python3
"""
warm_caches.py - Pre-populate catalyst data caches for production runs.

Fetches external data sources (FDA ADCOM calendar, SEC 8-K filings) and
writes cache files so the screening pipeline can consume them without
live network access.

Usage:
    python warm_caches.py                          # warm all caches for today
    python warm_caches.py --as-of-date 2026-02-07  # warm for specific date
    python warm_caches.py --sources fda_adcom      # warm only FDA ADCOM
    python warm_caches.py --sources sec_8k         # warm only SEC 8-K
    python warm_caches.py --sources fda_adcom,sec_8k  # both (default)

Cache files are written to:
    cache/fda/adcom_calendar_{date}.json
    cache/sec/8k_catalysts/8k_catalysts_{date}.json

After running, the screening pipeline (with default cache_only mode) will
automatically pick up these cached artifacts.
"""

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("warm_caches")

# Project root (where this script lives)
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "production_data"


def warm_fda_adcom(as_of_date: date, data_dir: Path, cache_dir: Path) -> int:
    """Fetch and cache FDA ADCOM calendar events. Returns count."""
    from wake_robin_data_pipeline.collectors.fda_adcom_collector import (
        collect_fda_adcom_events,
        build_product_ticker_map,
    )

    logger.info("Building product-to-ticker map...")
    product_map = build_product_ticker_map(data_dir)
    if not product_map:
        logger.warning("Empty product map — check pdufa_dates.json / fda_designations.json")
        return 0

    logger.info(f"Fetching FDA ADCOM calendar for {as_of_date}...")
    events = collect_fda_adcom_events(
        drug_to_ticker=product_map,
        as_of_date=as_of_date,
        cache_dir=cache_dir,
    )

    cache_path = cache_dir / f"adcom_calendar_{as_of_date.isoformat()}.json"
    logger.info(f"FDA ADCOM: {len(events)} events cached → {cache_path}")
    return len(events)


def warm_sec_8k(as_of_date: date, data_dir: Path, cache_dir: Path) -> int:
    """Fetch and cache SEC 8-K timing events. Returns count."""
    from wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector import (
        collect_8k_timing_events,
    )

    # Load universe
    universe_path = data_dir / "universe.json"
    if not universe_path.exists():
        logger.warning(f"Universe file not found: {universe_path}")
        return 0

    with open(universe_path, "r", encoding="utf-8") as f:
        universe_data = json.load(f)

    universe = universe_data if isinstance(universe_data, list) else universe_data.get("tickers", [])
    logger.info(f"Fetching SEC 8-K filings for {len(universe)} tickers, as_of {as_of_date}...")

    events = collect_8k_timing_events(
        universe=universe,
        as_of_date=as_of_date,
        cache_dir=cache_dir,
    )

    cache_path = cache_dir / f"8k_catalysts_{as_of_date.isoformat()}.json"
    logger.info(f"SEC 8-K: {len(events)} events cached → {cache_path}")
    return len(events)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-populate catalyst data caches for production screening runs.",
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=date.today().isoformat(),
        help="Date to warm caches for (ISO format, default: today)",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="fda_adcom,sec_8k",
        help="Comma-separated sources to warm (default: fda_adcom,sec_8k)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DATA_DIR),
        help=f"Data directory (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--fda-cache-dir",
        type=str,
        default=str(PROJECT_ROOT / "cache" / "fda"),
        help="FDA ADCOM cache directory",
    )
    parser.add_argument(
        "--sec-cache-dir",
        type=str,
        default=str(PROJECT_ROOT / "cache" / "sec" / "8k_catalysts"),
        help="SEC 8-K cache directory",
    )

    args = parser.parse_args()
    as_of = date.fromisoformat(args.as_of_date)
    sources = [s.strip() for s in args.sources.split(",")]
    data_dir = Path(args.data_dir)

    logger.info(f"Warming caches for as_of_date={as_of}, sources={sources}")

    total = 0

    if "fda_adcom" in sources:
        fda_cache_dir = Path(args.fda_cache_dir)
        fda_cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            total += warm_fda_adcom(as_of, data_dir, fda_cache_dir)
        except Exception as e:
            logger.error(f"FDA ADCOM warm failed: {e}")

    if "sec_8k" in sources:
        sec_cache_dir = Path(args.sec_cache_dir)
        sec_cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            total += warm_sec_8k(as_of, data_dir, sec_cache_dir)
        except Exception as e:
            logger.error(f"SEC 8-K warm failed: {e}")

    logger.info(f"Cache warm complete: {total} total events cached")
    return 0


if __name__ == "__main__":
    sys.exit(main())
