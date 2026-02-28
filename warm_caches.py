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
import os
import re
import shutil
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

from cache_health import (
    SEC8K_MIN_RATIO_VS_PRIOR,
    CTGOV_BAD_RATIO_LOW,
    CTGOV_BAD_RATIO_HIGH,
)

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


def validate_cache_refresh(
    source: str,
    new_count: int,
    prior_count: int | None,
) -> tuple[bool, str]:
    """Validate a cache refresh before committing.

    Returns (accepted, reason). reason="" if accepted.
    """
    if source == "sec_8k":
        if new_count == 0:
            return False, "empty_refresh"
        if prior_count is not None and prior_count > 0:
            ratio = new_count / prior_count
            if ratio < SEC8K_MIN_RATIO_VS_PRIOR:
                return False, f"collapse (ratio={ratio:.2f} < {SEC8K_MIN_RATIO_VS_PRIOR})"
        return True, ""

    if source == "ctgov":
        if new_count == 0:
            return False, "empty_refresh"
        if prior_count is not None and prior_count > 0:
            ratio = new_count / prior_count
            if ratio < CTGOV_BAD_RATIO_LOW or ratio > CTGOV_BAD_RATIO_HIGH:
                return False, (
                    f"out_of_band (ratio={ratio:.2f}, "
                    f"band=[{CTGOV_BAD_RATIO_LOW}, {CTGOV_BAD_RATIO_HIGH}])"
                )
        return True, ""

    if source in ("euctr", "ctis", "isrctn", "ema_agenda", "ema_outcomes"):
        # New registries/sources: accept anything non-negative; no hard floor initially
        return True, ""

    if source == "merged_trials":
        if new_count == 0:
            return False, "empty_refresh"
        if prior_count is not None and prior_count > 0:
            from cache_health import TRIAL_MERGE_BAD_RATIO_LOW, TRIAL_MERGE_BAD_RATIO_HIGH
            ratio = new_count / prior_count
            if ratio < TRIAL_MERGE_BAD_RATIO_LOW or ratio > TRIAL_MERGE_BAD_RATIO_HIGH:
                return False, (
                    f"out_of_band (ratio={ratio:.2f}, "
                    f"band=[{TRIAL_MERGE_BAD_RATIO_LOW}, {TRIAL_MERGE_BAD_RATIO_HIGH}])"
                )
        return True, ""

    return True, ""


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


def _find_prior_sec8k_count(cache_dir: Path) -> int | None:
    """Find most recent prior 8-K cache file and return its event count."""
    candidates = sorted(cache_dir.glob("8k_catalysts_*_*.json"), reverse=True)
    for path in candidates:
        try:
            data = json.loads(path.read_text())
            if isinstance(data, list):
                return len(data)
        except (json.JSONDecodeError, OSError):
            continue
    return None


def warm_sec_8k(as_of_date: date, data_dir: Path, cache_dir: Path) -> int:
    """Fetch and cache SEC 8-K timing events. Returns count.

    Uses staging: fetches into a temp dir, validates the result count
    against the prior cache, and only commits to the live path on pass.
    """
    from wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector import (
        collect_8k_timing_events,
        _versioned_cache_path,
    )

    # Short-circuit if live cache already exists
    live_path = _versioned_cache_path(cache_dir, as_of_date)
    if live_path.exists():
        try:
            cached = json.loads(live_path.read_text())
            logger.info(f"SEC 8-K cache exists: {live_path.name} ({len(cached)} events)")
            return len(cached)
        except (json.JSONDecodeError, OSError):
            pass  # fall through to re-fetch

    universe = _load_universe(data_dir)
    if not universe:
        return 0

    prior_count = _find_prior_sec8k_count(cache_dir)
    logger.info(f"Fetching SEC 8-K filings for {len(universe)} tickers, as_of {as_of_date}...")

    # Stage into temp dir (same filesystem for fast move)
    staging_dir = tempfile.mkdtemp(dir=str(cache_dir), prefix=".staging_8k_")
    try:
        events = collect_8k_timing_events(
            universe=universe,
            as_of_date=as_of_date,
            cache_dir=Path(staging_dir),
        )

        accepted, reason = validate_cache_refresh("sec_8k", len(events), prior_count)
        if not accepted:
            logger.warning(
                f"SEC 8-K refresh REJECTED: {reason} "
                f"(new={len(events)}, prior={prior_count}) — keeping prior cache"
            )
            return len(events)

        # Write validated events to live path
        with open(live_path, "w", encoding="utf-8") as f:
            json.dump(events, f, indent=2)

        logger.info(f"SEC 8-K: {len(events)} events cached → {live_path}")
        return len(events)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def _find_prior_ctgov_count(cache_dir: Path) -> int | None:
    """Find most recent prior CTGov cache file and return its record count."""
    candidates = sorted(cache_dir.glob("trial_records_*.json"), reverse=True)
    for path in candidates:
        try:
            data = json.loads(path.read_text())
            if isinstance(data, list):
                return len(data)
        except (json.JSONDecodeError, OSError):
            continue
    return None


def warm_ctgov(as_of_date: date, data_dir: Path, cache_dir: Path | None = None) -> int:
    """Create PIT-filtered trial_records snapshot for as_of_date.

    Keeps only records with last_update_posted <= as_of_date.
    Uses atomic write: temp file + os.replace() after validation.
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

    # Validate before writing
    prior_count = _find_prior_ctgov_count(cache_dir)
    accepted, reason = validate_cache_refresh("ctgov", len(filtered), prior_count)
    if not accepted:
        logger.warning(
            f"CTGov refresh REJECTED: {reason} "
            f"(new={len(filtered)}, prior={prior_count}) — keeping prior cache"
        )
        return len(filtered)

    # Atomic write: temp file + os.replace()
    fd, tmp_path = tempfile.mkstemp(dir=str(cache_dir), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(filtered, f)
        os.replace(tmp_path, str(target))
    except BaseException:
        # Clean up temp file on any failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

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
    3. Run collector with narrow lookback (7 days) into staging dir
    4. Merge seed + delta, dedup, expire old events
    5. Validate merged count, commit to live path on pass

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

    target_cache = _versioned_cache_path(cache_dir, as_of_date)

    # Stage delta fetch into temp dir (collector won't short-circuit on empty dir)
    staging_dir = tempfile.mkdtemp(dir=str(cache_dir), prefix=".staging_8k_delta_")
    try:
        logger.info(
            f"SEC 8-K delta: fetching {_DELTA_LOOKBACK_DAYS}-day window for "
            f"{len(universe)} tickers, as_of {as_of_date}..."
        )
        delta_events = collect_8k_timing_events(
            universe=universe,
            as_of_date=as_of_date,
            cache_dir=Path(staging_dir),
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

        # Validate merged result
        accepted, reason = validate_cache_refresh("sec_8k", len(final), len(seed_events))
        if not accepted:
            logger.warning(
                f"SEC 8-K delta refresh REJECTED: {reason} "
                f"(merged={len(final)}, seed={len(seed_events)}) — keeping prior cache"
            )
            return len(final)

        # Write merged result to staging, then move to live path
        staged_file = Path(staging_dir) / target_cache.name
        with open(staged_file, "w", encoding="utf-8") as f:
            json.dump(final, f, indent=2)
        shutil.move(str(staged_file), str(target_cache))

        logger.info(
            f"SEC 8-K delta: {len(seed_events)} seed + {len(delta_events)} new "
            f"→ {len(final)} merged ({expired_count} expired)"
        )
        return len(final)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


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


def warm_price_pit(
    as_of_date: date,
    data_dir: Path,
    cache_dir: Path | None = None,
    snapshot_dir: Path | None = None,
) -> int:
    """Create PIT price cache anchor for today's snapshot. Returns ticker count."""
    from tools.warm_price_cache import snapshot_price_cache

    snap_dir = snapshot_dir or (PROJECT_ROOT / "data" / "snapshots")
    rankings_csv = snap_dir / as_of_date.isoformat() / "rankings.csv"
    if not rankings_csv.exists():
        logger.warning(f"rankings.csv not found at {rankings_csv} — skipping price_pit")
        return 0

    out_dir = cache_dir or (
        PROJECT_ROOT / "data" / "caches" / "price_pit" / "PIT" / as_of_date.isoformat()
    )
    price_csv = data_dir / "price_history.csv"
    if not price_csv.exists():
        # Fallback to production_data
        price_csv = PROJECT_ROOT / "production_data" / "price_history.csv"

    index = snapshot_price_cache(
        as_of_date=as_of_date.isoformat(),
        price_csv=price_csv,
        rankings_csv=rankings_csv,
        out_dir=out_dir,
    )
    return index.get("ticker_count", 0)


def warm_ema_committee_events(as_of_date: date, data_dir: Path, cache_dir: Path) -> int:
    """Fetch and cache EMA committee agenda events. Returns count."""
    from wake_robin_data_pipeline.collectors.ema_committee_collector import (
        collect_ema_committee_events,
    )
    from wake_robin_data_pipeline.collectors.fda_adcom_collector import (
        build_product_ticker_map,
    )

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    product_map = build_product_ticker_map(data_dir)
    if not product_map:
        logger.warning("Empty product map — check pdufa_dates.json / fda_designations.json")
        return 0

    logger.info("Fetching EMA committee events for %s...", as_of_date)
    events = collect_ema_committee_events(
        as_of_date=as_of_date,
        cache_dir=cache_dir,
        product_ticker_map=product_map,
    )
    logger.info("EMA committee events: %d events cached", len(events))
    return len(events)


def warm_ema_meeting_outcomes(as_of_date: date, data_dir: Path, cache_dir: Path) -> int:
    """Fetch and cache EMA meeting outcome events. Returns count."""
    from wake_robin_data_pipeline.collectors.ema_committee_collector import (
        collect_ema_meeting_outcomes,
    )
    from wake_robin_data_pipeline.collectors.fda_adcom_collector import (
        build_product_ticker_map,
    )

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    product_map = build_product_ticker_map(data_dir)
    if not product_map:
        logger.warning("Empty product map — check pdufa_dates.json / fda_designations.json")
        return 0

    logger.info("Fetching EMA meeting outcomes for %s...", as_of_date)
    events = collect_ema_meeting_outcomes(
        as_of_date=as_of_date,
        cache_dir=cache_dir,
        product_ticker_map=product_map,
    )
    logger.info("EMA meeting outcomes: %d events cached", len(events))
    return len(events)


def warm_euctr(as_of_date: date, data_dir: Path, cache_dir: Path) -> int:
    """Fetch and cache EUCTR trial records. Returns count."""
    from wake_robin_data_pipeline.collectors.euctr_collector import collect_euctr_trials

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    universe = _load_universe(data_dir)
    if not universe:
        return 0

    logger.info("Fetching EUCTR trials for %d tickers...", len(universe))
    records = collect_euctr_trials(
        universe=universe,
        as_of_date=as_of_date,
        cache_dir=cache_dir,
    )
    logger.info("EUCTR: %d records cached", len(records))
    return len(records)


def warm_ctis(
    as_of_date: date, data_dir: Path, cache_dir: Path,
    enrich_detail: bool = True,
) -> int:
    """Fetch and cache CTIS trial records. Returns count."""
    from wake_robin_data_pipeline.collectors.ctis_collector import collect_ctis_trials

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    universe = _load_universe(data_dir)
    if not universe:
        return 0

    logger.info("Fetching CTIS trials for %d tickers...", len(universe))
    records = collect_ctis_trials(
        universe=universe,
        as_of_date=as_of_date,
        cache_dir=cache_dir,
        enrich_detail=enrich_detail,
    )
    logger.info("CTIS: %d records cached", len(records))
    return len(records)


def warm_isrctn(as_of_date: date, data_dir: Path, cache_dir: Path) -> int:
    """Fetch and cache ISRCTN trial records. Returns count."""
    from wake_robin_data_pipeline.collectors.isrctn_collector import collect_isrctn_trials

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    universe = _load_universe(data_dir)
    if not universe:
        return 0

    logger.info("Fetching ISRCTN trials for %d tickers...", len(universe))
    records = collect_isrctn_trials(
        universe=universe,
        as_of_date=as_of_date,
        cache_dir=cache_dir,
    )
    logger.info("ISRCTN: %d records cached", len(records))
    return len(records)


def warm_merged_trials(as_of_date: date, data_dir: Path, cache_root: Path) -> int:
    """Merge all per-registry trial caches into unified merged output. Returns count."""
    from wake_robin_data_pipeline.collectors.trial_registry_merger import (
        merge_trial_registries,
        write_merged_cache,
    )

    cache_root = Path(cache_root)

    # Load per-registry caches (empty list if absent)
    ctgov_records = _load_registry_cache(
        _CTGOV_CACHE_DIR / f"trial_records_{as_of_date.isoformat()}.json"
    )
    euctr_records = _load_registry_cache(
        cache_root / "euctr" / f"euctr_{as_of_date.isoformat()}.json"
    )
    ctis_records = _load_registry_cache(
        cache_root / "ctis" / f"ctis_{as_of_date.isoformat()}.json"
    )
    isrctn_records = _load_registry_cache(
        cache_root / "isrctn" / f"isrctn_{as_of_date.isoformat()}.json"
    )

    total_input = len(ctgov_records) + len(euctr_records) + len(ctis_records) + len(isrctn_records)
    if total_input == 0:
        logger.warning("No per-registry caches found for %s — skipping merge", as_of_date)
        return 0

    logger.info(
        "Merging trials: ctgov=%d, euctr=%d, ctis=%d, isrctn=%d",
        len(ctgov_records), len(euctr_records), len(ctis_records), len(isrctn_records),
    )

    merged = merge_trial_registries(
        ctgov_records, euctr_records, ctis_records, isrctn_records, as_of_date,
    )

    merged_dir = cache_root / "merged"
    write_merged_cache(merged, as_of_date, merged_dir)

    logger.info("Merged trials: %d records -> %s", len(merged), merged_dir)
    return len(merged)


def _load_registry_cache(path: Path) -> list:
    """Load a per-registry JSON cache file. Returns empty list on error."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


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
        help="Comma-separated sources to warm: fda_adcom,sec_8k,ctgov,sec_13f,event_ledger,price_pit (default: fda_adcom,sec_8k)",
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
    parser.add_argument(
        "--ema-cache-dir",
        type=str,
        default=str(PROJECT_ROOT / "cache" / "ema"),
        help="EMA committee cache directory",
    )
    parser.add_argument(
        "--euctr-cache-dir",
        type=str,
        default=str(PROJECT_ROOT / "cache" / "trials" / "euctr"),
        help="EUCTR cache directory",
    )
    parser.add_argument(
        "--ctis-cache-dir",
        type=str,
        default=str(PROJECT_ROOT / "cache" / "trials" / "ctis"),
        help="CTIS cache directory",
    )
    parser.add_argument(
        "--isrctn-cache-dir",
        type=str,
        default=str(PROJECT_ROOT / "cache" / "trials" / "isrctn"),
        help="ISRCTN cache directory",
    )
    parser.add_argument(
        "--trials-cache-root",
        type=str,
        default=str(PROJECT_ROOT / "cache" / "trials"),
        help="Root directory for trial registry caches (merged output goes here)",
    )
    parser.add_argument(
        "--no-ctis-enrich-detail",
        action="store_true",
        default=False,
        help="Disable CTIS detail endpoint enrichment (default: enrichment ON)",
    )

    args = parser.parse_args()
    as_of = date.fromisoformat(args.as_of_date)
    sources = [s.strip() for s in args.sources.split(",")]
    data_dir = Path(args.data_dir)

    logger.info(f"Warming caches for as_of_date={as_of}, sources={sources}")

    total = 0
    refresh_results: list[dict] = []

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
        prior_8k = _find_prior_sec8k_count(sec_cache_dir)
        sec_8k_count = 0
        try:
            if args.seed_cache and Path(args.seed_cache).exists():
                sec_8k_count = warm_sec_8k_delta(as_of, data_dir, sec_cache_dir, Path(args.seed_cache))
            else:
                sec_8k_count = warm_sec_8k(as_of, data_dir, sec_cache_dir)
        except Exception as e:
            logger.error(f"SEC 8-K warm failed: {e}")
        accepted, reason = validate_cache_refresh("sec_8k", sec_8k_count, prior_8k)
        if accepted:
            total += sec_8k_count
        refresh_results.append({
            "source": "sec_8k",
            "count": sec_8k_count,
            "prior_count": prior_8k,
            "accepted": accepted,
            "committed": accepted,
            "reason": reason,
        })

    sec_13f_count = 0
    if "sec_13f" in sources:
        try:
            cache_dir_13f = Path(args.sec_13f_cache_dir) if args.sec_13f_cache_dir else None
            sec_13f_count = warm_sec_13f(as_of, data_dir, cache_dir=cache_dir_13f)
        except Exception as e:
            logger.error(f"SEC 13F warm failed: {e}")

    ctgov_records = 0
    if "ctgov" in sources:
        prior_ctgov = _find_prior_ctgov_count(_CTGOV_CACHE_DIR)
        try:
            ctgov_records = warm_ctgov(as_of, data_dir)
        except Exception as e:
            logger.error(f"CTGov warm failed: {e}")
        accepted, reason = validate_cache_refresh("ctgov", ctgov_records, prior_ctgov)
        refresh_results.append({
            "source": "ctgov",
            "count": ctgov_records,
            "prior_count": prior_ctgov,
            "accepted": accepted,
            "committed": accepted,
            "reason": reason,
        })

    ledger_entries = 0
    if "event_ledger" in sources:
        try:
            # Use fda cache dir as reference for cache root
            cache_root = Path(args.fda_cache_dir).parent
            ledger_entries = warm_event_ledger(as_of, data_dir, cache_root / "fda")
        except Exception as e:
            logger.error(f"Event ledger warm failed: {e}")

    price_pit_count = 0
    if "price_pit" in sources:
        try:
            price_pit_count = warm_price_pit(as_of, data_dir)
        except Exception as e:
            logger.error(f"Price PIT warm failed: {e}")

    ema_agenda_count = 0
    if "ema_agenda" in sources:
        try:
            ema_cache_dir = Path(args.ema_cache_dir)
            ema_agenda_count = warm_ema_committee_events(as_of, data_dir, ema_cache_dir)
        except Exception as e:
            logger.error(f"EMA committee events warm failed: {e}")

    ema_outcomes_count = 0
    if "ema_outcomes" in sources:
        try:
            ema_cache_dir = Path(args.ema_cache_dir)
            ema_outcomes_count = warm_ema_meeting_outcomes(as_of, data_dir, ema_cache_dir)
        except Exception as e:
            logger.error(f"EMA meeting outcomes warm failed: {e}")

    euctr_count = 0
    if "euctr" in sources:
        try:
            euctr_cache_dir = Path(args.euctr_cache_dir)
            euctr_count = warm_euctr(as_of, data_dir, euctr_cache_dir)
        except Exception as e:
            logger.error(f"EUCTR warm failed: {e}")

    ctis_count = 0
    if "ctis" in sources:
        try:
            ctis_cache_dir = Path(args.ctis_cache_dir)
            ctis_count = warm_ctis(
                as_of, data_dir, ctis_cache_dir,
                enrich_detail=not args.no_ctis_enrich_detail,
            )
        except Exception as e:
            logger.error(f"CTIS warm failed: {e}")

    isrctn_count = 0
    if "isrctn" in sources:
        try:
            isrctn_cache_dir = Path(args.isrctn_cache_dir)
            isrctn_count = warm_isrctn(as_of, data_dir, isrctn_cache_dir)
        except Exception as e:
            logger.error(f"ISRCTN warm failed: {e}")

    merged_trials_count = 0
    if "merged_trials" in sources:
        try:
            trials_cache_root = Path(args.trials_cache_root)
            merged_trials_count = warm_merged_trials(as_of, data_dir, trials_cache_root)
        except Exception as e:
            logger.error(f"Merged trials warm failed: {e}")

    # Write refresh diagnostics sidecar
    if refresh_results:
        cache_root = PROJECT_ROOT / "cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        sidecar_path = cache_root / f"cache_refresh_{as_of.isoformat()}.json"
        try:
            sidecar_path.write_text(json.dumps({
                "schema": "cache_refresh.v1",
                "as_of_date": as_of.isoformat(),
                "results": refresh_results,
            }, indent=2))
            logger.info(f"Refresh diagnostics → {sidecar_path}")
        except OSError as e:
            logger.warning(f"Could not write refresh sidecar: {e}")

    parts = [f"{total} events"]
    if sec_13f_count:
        parts.append(f"{sec_13f_count} 13F managers")
    if ctgov_records:
        parts.append(f"{ctgov_records} CTGov records")
    if ledger_entries:
        parts.append(f"{ledger_entries} ledger entries")
    if price_pit_count:
        parts.append(f"{price_pit_count} price PIT tickers")
    if ema_agenda_count:
        parts.append(f"{ema_agenda_count} EMA agenda events")
    if ema_outcomes_count:
        parts.append(f"{ema_outcomes_count} EMA outcome events")
    if euctr_count:
        parts.append(f"{euctr_count} EUCTR records")
    if ctis_count:
        parts.append(f"{ctis_count} CTIS records")
    if isrctn_count:
        parts.append(f"{isrctn_count} ISRCTN records")
    if merged_trials_count:
        parts.append(f"{merged_trials_count} merged trial records")
    logger.info(f"Cache warm complete: {', '.join(parts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
