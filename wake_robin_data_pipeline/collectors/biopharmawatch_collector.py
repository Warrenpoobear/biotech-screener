"""
biopharmawatch_collector.py - BiopharmaWatch FDA catalyst collector.

Fetches PDUFA, ADCOM, approval, and CRL events from the BiopharmaWatch
/api/v2/fdacalendar endpoint and writes them to the cache/fda/ directory
in the exact formats that event_ledger.py loaders expect.

Environment variable: BPW_API_KEY (never hardcode, never log, never commit)

Writes:
    cache/fda/adcom_calendar_{as_of_date}.json
    cache/fda/fda_regulatory_{as_of_date}.json
"""

import json
import logging
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

BPW_BASE_URL = "https://api.biopharmawatch.com"
BPW_ENDPOINT = "/api/v2/fdacalendar"
DEFAULT_CACHE_DIR = Path("cache") / "fda"
DEFAULT_LOOKFORWARD_DAYS = 365

# BPW eventType → canonical event_type for cache records
_EVENT_TYPE_MAP = {
    "pdufa-date": "PDUFA",
    "fda-advisory-committee": "FDA_ADCOM",
    "fda-approval": "FDA_APPROVAL",
    "fda-supplement-approval": "FDA_APPROVAL",
    "fda-crl": "FDA_CRL",
}

# ADCOM events go to adcom_calendar; all others go to fda_regulatory
_ADCOM_EVENT_TYPES = {"FDA_ADCOM"}


def _load_pdufa_dates(pdufa_dates_path: Path) -> list:
    """Load curated pdufa_dates.json for dedup. Returns empty list on error."""
    if not pdufa_dates_path.exists():
        return []
    try:
        data = json.loads(pdufa_dates_path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not load pdufa_dates: %s", e)
        return []


def _build_pdufa_dedup_set(pdufa_entries: list) -> set:
    """Build (ticker, date_str) set from pdufa_dates.json for dedup checking."""
    result = set()
    for entry in pdufa_entries:
        ticker = entry.get("ticker", "").upper().strip()
        pdufa_date = entry.get("pdufa_date", "")
        if ticker and pdufa_date:
            result.add((ticker, pdufa_date))
    return result


def _is_pdufa_duplicate(ticker: str, event_date: str, dedup_set: set, window_days: int = 3) -> bool:
    """Return True if pdufa_dates.json already has this (ticker, ~date) entry."""
    try:
        ev = date.fromisoformat(event_date)
    except ValueError:
        return False
    for delta in range(-window_days, window_days + 1):
        candidate = (ev + timedelta(days=delta)).isoformat()
        if (ticker.upper(), candidate) in dedup_set:
            return True
    return False


def _normalize_date(value: Optional[str]) -> Optional[str]:
    """Normalize a date string to YYYY-MM-DD. Returns None on failure."""
    if not value:
        return None
    # Accept YYYY-MM-DD directly
    if len(value) == 10 and value[4] == "-" and value[7] == "-":
        return value
    # Try ISO with time component
    try:
        return value[:10]
    except Exception:
        return None


def _map_bpw_record(raw: dict, as_of_date: date) -> Optional[dict]:
    """
    Map a BPW API response record to a cache dict.

    Returns None if:
    - ticker is missing or empty
    - event_date is missing or malformed
    - disclosed_at > as_of_date (leakage gate)
    - event_type is not in the mapping
    """
    ticker = (raw.get("ticker") or "").upper().strip()
    if not ticker:
        return None

    bpw_event_type = raw.get("eventType") or raw.get("event_type") or ""
    event_type = _EVENT_TYPE_MAP.get(bpw_event_type)
    if not event_type:
        logger.debug("Unknown BPW eventType=%r — skipping", bpw_event_type)
        return None

    raw_event_date = raw.get("pdufaDate") or raw.get("date") or raw.get("eventDate") or ""
    event_date = _normalize_date(raw_event_date)
    if not event_date:
        logger.debug("Missing/malformed event_date for %s — skipping", ticker)
        return None

    # PIT anchor: use the earlier of announcedOn / filedAt as disclosed_at
    announced = _normalize_date(raw.get("announcedOn") or raw.get("announced_on"))
    filed = _normalize_date(raw.get("filedAt") or raw.get("filed_at"))
    candidates = [d for d in [announced, filed] if d]
    if candidates:
        disclosed_at = min(candidates)
    else:
        # Fall back to as_of_date as the PIT anchor (conservative)
        disclosed_at = as_of_date.isoformat()

    # Leakage gate: skip if disclosed_at > as_of_date
    if disclosed_at > as_of_date.isoformat():
        logger.debug("Leakage gate: %s disclosed_at=%s > as_of=%s — skipping", ticker, disclosed_at, as_of_date)
        return None

    # Build tags
    tags = [event_type.lower(), "source:biopharmawatch", "pit_anchor:filing_date"]
    app_number = raw.get("applicationNumber") or raw.get("application_number")
    if app_number:
        tags.append(f"nda:{app_number}")

    # Event name: title or drug name, truncated to 120 chars
    title = raw.get("title") or raw.get("drugName") or raw.get("drug_name") or ""
    if title:
        prefix = event_type.replace("_", " ").title()
        event_name = f"{prefix}: {title}"[:120]
    else:
        event_name = event_type

    record = {
        "ticker": ticker,
        "event_type": event_type,
        "event_date": event_date,
        "disclosed_at": disclosed_at,
        "event_name": event_name,
        "tags": tags,
    }
    return record


def _fetch_bpw_events(api_key: str, as_of_date: date, lookforward_days: int) -> list:
    """Fetch raw events from BPW API. Returns list of raw dicts."""
    try:
        from common.robustness import create_resilient_session
    except ImportError:
        import requests

        session = requests.Session()
    else:
        session = create_resilient_session()

    end_date = (as_of_date + timedelta(days=lookforward_days)).isoformat()
    params = {
        "startDate": as_of_date.isoformat(),
        "endDate": end_date,
        "apiKey": api_key,
    }
    headers = {"Accept": "application/json"}

    url = BPW_BASE_URL + BPW_ENDPOINT
    try:
        resp = session.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error("BPW API request failed: %s", e)
        return []

    if isinstance(data, list):
        return data
    # Some endpoints return {"data": [...]}
    if isinstance(data, dict):
        return data.get("data") or data.get("events") or []
    return []


def _atomic_write(path: Path, records: list) -> None:
    """Write records to path atomically via .tmp + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def collect_biopharmawatch_fda_events(
    as_of_date: date,
    cache_dir: Path,
    pdufa_dates_path: Path,
    lookforward_days: int = DEFAULT_LOOKFORWARD_DAYS,
    api_key: Optional[str] = None,
) -> Tuple[List[dict], List[dict]]:
    """
    Fetch BPW FDA calendar and write cache files.

    Returns (fda_regulatory_records, adcom_records).

    If BPW_API_KEY is absent and api_key is not provided, logs INFO and
    returns ([], []) without writing any files.

    Cache files written (atomic):
        cache_dir/fda_regulatory_{as_of_date}.json
        cache_dir/adcom_calendar_{as_of_date}.json
    """
    cache_dir = Path(cache_dir)

    # Resolve API key — never log it
    resolved_key = api_key or os.environ.get("BPW_API_KEY", "")
    if not resolved_key:
        logger.info("BPW_API_KEY not set — skipping BiopharmaWatch cache warm")
        return [], []

    # Short-circuit if both cache files already exist
    reg_path = cache_dir / f"fda_regulatory_{as_of_date.isoformat()}.json"
    adcom_path = cache_dir / f"adcom_calendar_{as_of_date.isoformat()}.json"
    if reg_path.exists() and adcom_path.exists():
        try:
            reg_records = json.loads(reg_path.read_text(encoding="utf-8"))
            adcom_records = json.loads(adcom_path.read_text(encoding="utf-8"))
            logger.info(
                "BPW cache exists: fda_regulatory=%d adcom=%d", len(reg_records), len(adcom_records)
            )
            return reg_records, adcom_records
        except (json.JSONDecodeError, OSError):
            pass  # re-fetch on corrupt cache

    # Load curated PDUFA dates for dedup
    pdufa_entries = _load_pdufa_dates(pdufa_dates_path)
    pdufa_dedup_set = _build_pdufa_dedup_set(pdufa_entries)
    logger.debug("Loaded %d curated PDUFA entries for dedup", len(pdufa_entries))

    # Fetch from BPW
    logger.info("Fetching BiopharmaWatch FDA calendar (as_of=%s, +%dd)...", as_of_date, lookforward_days)
    raw_events = _fetch_bpw_events(resolved_key, as_of_date, lookforward_days)
    logger.info("BPW returned %d raw records", len(raw_events))

    reg_records: List[dict] = []
    adcom_records: List[dict] = []
    skipped_leakage = 0
    skipped_unknown = 0
    skipped_dedup = 0

    for raw in raw_events:
        record = _map_bpw_record(raw, as_of_date)
        if record is None:
            # _map_bpw_record logs at DEBUG; count unknowns vs leakage separately
            skipped_unknown += 1
            continue

        event_type = record["event_type"]

        if event_type in _ADCOM_EVENT_TYPES:
            adcom_records.append(record)
        else:
            # Dedup PDUFA records against curated pdufa_dates.json
            if event_type == "PDUFA" and _is_pdufa_duplicate(
                record["ticker"], record["event_date"], pdufa_dedup_set
            ):
                logger.debug(
                    "BPW PDUFA dedup: %s %s already in pdufa_dates.json — skipping",
                    record["ticker"],
                    record["event_date"],
                )
                skipped_dedup += 1
                continue
            reg_records.append(record)

    # Stable sort for deterministic output
    reg_records.sort(key=lambda r: (r.get("event_date", ""), r.get("ticker", "")))
    adcom_records.sort(key=lambda r: (r.get("event_date", ""), r.get("ticker", "")))

    logger.info(
        "BPW mapped: fda_regulatory=%d adcom=%d skipped(dedup=%d unknown/leakage=%d)",
        len(reg_records),
        len(adcom_records),
        skipped_dedup,
        skipped_unknown,
    )

    # Atomic write
    cache_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(reg_path, reg_records)
    _atomic_write(adcom_path, adcom_records)
    logger.info("BPW cache written: %s (%d), %s (%d)", reg_path.name, len(reg_records), adcom_path.name, len(adcom_records))

    return reg_records, adcom_records
