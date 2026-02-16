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
import re
import sys
from datetime import date, timedelta
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("warm_caches")

# Project root (where this script lives)
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "production_data"

# Delta warming: narrow EDGAR search to recent filings only
_DELTA_LOOKBACK_DAYS = 7
_EXPIRE_WINDOW_DAYS = 180


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

    universe = _load_universe(data_dir)
    if not universe:
        return 0
    logger.info(f"Fetching SEC 8-K filings for {len(universe)} tickers, as_of {as_of_date}...")

    events = collect_8k_timing_events(
        universe=universe,
        as_of_date=as_of_date,
        cache_dir=cache_dir,
    )

    cache_path = cache_dir / f"8k_catalysts_{as_of_date.isoformat()}.json"
    logger.info(f"SEC 8-K: {len(events)} events cached → {cache_path}")
    return len(events)


def _load_universe(data_dir: Path) -> list:
    """Load universe from data_dir/universe.json. Returns list of dicts."""
    universe_path = data_dir / "universe.json"
    if not universe_path.exists():
        logger.warning(f"Universe file not found: {universe_path}")
        return []
    with open(universe_path, "r", encoding="utf-8") as f:
        universe_data = json.load(f)
    return universe_data if isinstance(universe_data, list) else universe_data.get("tickers", [])


def _extract_pattern_version(cache_path: Path) -> str | None:
    """Extract PATTERN_VERSION from a versioned cache filename.

    Filenames look like: 8k_catalysts_2026-02-14_249a4353.json
    Returns the version hash or None if not parseable.
    """
    m = re.search(r"8k_catalysts_\d{4}-\d{2}-\d{2}_([a-f0-9]{8})\.json$", cache_path.name)
    return m.group(1) if m else None


def _dedup_events(events: list[dict]) -> list[dict]:
    """Deduplicate events by (ticker, event_type, event_date)."""
    seen: set[tuple] = set()
    deduped = []
    for event in events:
        key = (event["ticker"], event["event_type"], event["event_date"])
        if key not in seen:
            seen.add(key)
            deduped.append(event)
    return deduped


def warm_sec_8k_delta(
    as_of_date: date,
    data_dir: Path,
    cache_dir: Path,
    seed_cache_path: Path,
) -> int:
    """Delta-warm SEC 8-K cache: seed from prior date + narrow EDGAR fetch.

    1. Load seed cache (prior date's events)
    2. Validate PATTERN_VERSION matches current collector version
    3. Run collector with narrow lookback (7 days)
    4. Merge seed + delta, dedup, expire old events
    5. Overwrite cache with merged result

    Falls back to full warm on version mismatch or seed load failure.
    Returns event count.
    """
    from wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector import (
        collect_8k_timing_events,
        PATTERN_VERSION,
        _versioned_cache_path,
    )

    # Validate seed exists
    if not seed_cache_path.exists():
        logger.warning(f"Seed cache not found: {seed_cache_path} — falling back to full warm")
        return warm_sec_8k(as_of_date, data_dir, cache_dir)

    # Validate PATTERN_VERSION in seed filename matches current
    seed_version = _extract_pattern_version(seed_cache_path)
    if seed_version != PATTERN_VERSION:
        logger.warning(
            f"Seed PATTERN_VERSION mismatch: seed={seed_version}, "
            f"current={PATTERN_VERSION} — falling back to full warm"
        )
        return warm_sec_8k(as_of_date, data_dir, cache_dir)

    # Load seed events
    try:
        with open(seed_cache_path, "r", encoding="utf-8") as f:
            seed_events = json.load(f)
        logger.info(f"Loaded {len(seed_events)} seed events from {seed_cache_path.name}")
    except Exception as e:
        logger.warning(f"Seed cache read error: {e} — falling back to full warm")
        return warm_sec_8k(as_of_date, data_dir, cache_dir)

    # Load universe
    universe = _load_universe(data_dir)
    if not universe:
        return 0

    # Remove any existing cache for as_of_date so collector doesn't short-circuit
    target_cache = _versioned_cache_path(cache_dir, as_of_date)
    if target_cache.exists():
        target_cache.unlink()

    # Narrow EDGAR fetch (7-day lookback)
    logger.info(
        f"SEC 8-K delta: fetching {_DELTA_LOOKBACK_DAYS}-day window for "
        f"{len(universe)} tickers, as_of {as_of_date}..."
    )
    delta_events = collect_8k_timing_events(
        universe=universe,
        as_of_date=as_of_date,
        cache_dir=cache_dir,
        lookback_days=_DELTA_LOOKBACK_DAYS,
    )

    # Merge seed + delta, dedup
    merged = _dedup_events(seed_events + delta_events)

    # Expire events with disclosed_at older than window
    expire_cutoff = (as_of_date - timedelta(days=_EXPIRE_WINDOW_DAYS)).isoformat()
    expired_count = 0
    final = []
    for event in merged:
        disclosed = event.get("disclosed_at", "")
        if disclosed and disclosed < expire_cutoff:
            expired_count += 1
        else:
            final.append(event)

    # Overwrite cache file with merged result
    try:
        with open(target_cache, "w", encoding="utf-8") as f:
            json.dump(final, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to write merged cache: {e}")
        return len(delta_events)

    logger.info(
        f"SEC 8-K delta: {len(seed_events)} seed + {len(delta_events)} new "
        f"→ {len(final)} merged ({expired_count} expired)"
    )
    return len(final)


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
    parser.add_argument(
        "--seed-cache",
        type=str,
        default=None,
        help="Path to prior date's 8-K cache for delta warming",
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
            if args.seed_cache and Path(args.seed_cache).exists():
                total += warm_sec_8k_delta(as_of, data_dir, sec_cache_dir, Path(args.seed_cache))
            else:
                total += warm_sec_8k(as_of, data_dir, sec_cache_dir)
        except Exception as e:
            logger.error(f"SEC 8-K warm failed: {e}")

    logger.info(f"Cache warm complete: {total} total events cached")
    return 0


if __name__ == "__main__":
    sys.exit(main())
