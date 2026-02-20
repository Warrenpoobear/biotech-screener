#!/usr/bin/env python3
"""
warm_caches.py - Pre-populate catalyst data caches for production runs.

Fetches external data sources (FDA ADCOM calendar, SEC 8-K filings) and
writes cache files so the screening pipeline can consume them without
live network access.  Also supports local PIT-filtered CTGov snapshots.

Usage:
    python warm_caches.py                          # warm all caches for today
    python warm_caches.py --as-of-date 2026-02-07  # warm for specific date
    python warm_caches.py --sources fda_adcom      # warm only FDA ADCOM
    python warm_caches.py --sources sec_8k         # warm only SEC 8-K
    python warm_caches.py --sources ctgov          # warm only CTGov PIT cache
    python warm_caches.py --sources sec_13f               # warm only SEC 13F
    python warm_caches.py --sources fda_adcom,sec_8k,ctgov  # all three

Cache files are written to:
    cache/fda/adcom_calendar_{date}.json
    cache/sec/8k_catalysts/8k_catalysts_{date}.json
    cache/ctgov/trial_records_{date}.json

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

# CTGov PIT-filtered cache directory
_CTGOV_CACHE_DIR = PROJECT_ROOT / "cache" / "ctgov"

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


def warm_ctgov(as_of_date: date, data_dir: Path, cache_dir: Path | None = None) -> int:
    """Create PIT-filtered trial_records snapshot for as_of_date.

    Keeps only records with last_update_posted <= as_of_date.
    Returns count of records in filtered snapshot.
    """
    cache_dir = cache_dir or _CTGOV_CACHE_DIR
    target = cache_dir / f"trial_records_{as_of_date.isoformat()}.json"
    if target.exists():
        existing = json.loads(target.read_text())
        logger.info(f"CTGov cache exists: {target.name} ({len(existing)} records)")
        return len(existing)

    source = data_dir / "trial_records.json"
    if not source.exists():
        raise FileNotFoundError(f"trial_records.json not found in {data_dir}")

    records = json.loads(source.read_text())
    cutoff = as_of_date.isoformat()

    missing_lup = sum(1 for r in records if not (r.get("last_update_posted") or "").strip())
    filtered = [
        r for r in records
        if (r.get("last_update_posted") or "")[:10] <= cutoff
    ]

    cache_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(filtered))
    logger.info(
        f"CTGov PIT filter: {len(records)} → {len(filtered)} records "
        f"(cutoff={cutoff}, dropped={len(records) - len(filtered)}, "
        f"missing_lup={missing_lup})"
    )
    return len(filtered)


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


def warm_sec_13f(as_of_date: date, data_dir: Path, cache_dir: Path | None = None) -> int:
    """Warm SEC 13F cache for elite managers. Returns count of managers with filings."""
    from tools.warm_13f_cache import warm_13f_cache

    out_dir = cache_dir or (PROJECT_ROOT / "data" / "caches" / "sec_13f" / "PIT" / as_of_date.isoformat())
    result = warm_13f_cache(as_of_date=as_of_date, out_dir=out_dir, elite_only=True)
    return result.get("managers_with_filing", 0)


def warm_event_ledger(as_of_date: date, data_dir: Path, cache_dir: Path) -> int:
    """Build and write event_ledger_{as_of_date}.jsonl from existing caches.

    Reads all cached sources (CTGov, SEC, FDA, PDUFA) and writes a unified
    ledger file for downstream consumption.  Returns entry count.
    """
    from event_ledger import build_event_ledger, write_ledger_jsonl, LedgerConfig

    ledger_config = LedgerConfig(
        ctgov_cache_dir=cache_dir.parent / "ctgov",
        sec_cache_dir=cache_dir.parent / "sec" / "8k_catalysts",
        fda_cache_dir=cache_dir.parent / "fda",
        data_dir=data_dir,
        strict_ctgov=False,
    )
    ledger = build_event_ledger(as_of_date, ledger_config)

    out_dir = cache_dir.parent / "ledger"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"event_ledger_{as_of_date.isoformat()}.jsonl"
    write_ledger_jsonl(ledger, out_path)
    logger.info(f"Event ledger: {len(ledger)} entries → {out_path}")
    return len(ledger)


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
        help="Comma-separated sources to warm: fda_adcom,sec_8k,ctgov,sec_13f,event_ledger (default: fda_adcom,sec_8k)",
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
    parser.add_argument(
        "--sec-13f-cache-dir",
        type=str,
        default=None,
        help="SEC 13F PIT cache output directory (default: data/caches/sec_13f/PIT/{date})",
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

    sec_13f_count = 0
    if "sec_13f" in sources:
        try:
            cache_dir_13f = Path(args.sec_13f_cache_dir) if args.sec_13f_cache_dir else None
            sec_13f_count = warm_sec_13f(as_of, data_dir, cache_dir=cache_dir_13f)
        except Exception as e:
            logger.error(f"SEC 13F warm failed: {e}")

    ctgov_records = 0
    if "ctgov" in sources:
        try:
            ctgov_records = warm_ctgov(as_of, data_dir)
        except Exception as e:
            logger.error(f"CTGov warm failed: {e}")

    ledger_entries = 0
    if "event_ledger" in sources:
        try:
            # Use fda cache dir as reference for cache root
            cache_root = Path(args.fda_cache_dir).parent
            ledger_entries = warm_event_ledger(as_of, data_dir, cache_root / "fda")
        except Exception as e:
            logger.error(f"Event ledger warm failed: {e}")

    parts = [f"{total} events"]
    if sec_13f_count:
        parts.append(f"{sec_13f_count} 13F managers")
    if ctgov_records:
        parts.append(f"{ctgov_records} CTGov records")
    if ledger_entries:
        parts.append(f"{ledger_entries} ledger entries")
    logger.info(f"Cache warm complete: {', '.join(parts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
