"""
pdufa_bio_collector.py - pdufa.bio free FDA catalyst collector.

Fetches PDUFA dates and ADCOM meetings from the pdufa.bio API (free, no key
required at anonymous tier — 1,000 req/day, 10/min burst).

Writes to the same cache files that biopharmawatch_collector.py writes.
Includes a cache-exists check at the top, so if BPW already populated the
cache for this date, this collector returns early without a network call.
This makes pdufa.bio a natural no-cost fallback when BPW_API_KEY is absent.

PIT note: pdufa.bio does not expose the original SEC filing date.
disclosed_at = updated_at[:10] when available, else as_of_date.isoformat().
Every record is tagged pit_anchor:updated_at or pit_anchor:as_of_date to
flag the limitation. Conservative (not contaminating) — updated_at <= as_of_date.

Writes:
    cache/fda/fda_regulatory_{as_of_date}.json  (PDUFA records)
    cache/fda/adcom_calendar_{as_of_date}.json  (ADCOM records)
"""

import json
import logging
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

PDUFA_BIO_BASE = "https://www.pdufa.bio/api/v1"
DEFAULT_LOOKFORWARD_DAYS = 365
_REQUEST_TIMEOUT = 20
_MAX_LIMIT = 1000


def _load_pdufa_dates(pdufa_dates_path: Path) -> list:
    if not pdufa_dates_path.exists():
        return []
    try:
        data = json.loads(pdufa_dates_path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not load pdufa_dates: %s", e)
        return []


def _build_pdufa_dedup_set(entries: list) -> set:
    result = set()
    for e in entries:
        ticker = (e.get("ticker") or "").upper().strip()
        pdufa_date = e.get("pdufa_date", "")
        if ticker and pdufa_date:
            result.add((ticker, pdufa_date))
    return result


def _is_pdufa_duplicate(ticker: str, event_date: str, dedup_set: set, window_days: int = 3) -> bool:
    try:
        ev = date.fromisoformat(event_date)
    except ValueError:
        return False
    for delta in range(-window_days, window_days + 1):
        if (ticker.upper(), (ev + timedelta(days=delta)).isoformat()) in dedup_set:
            return True
    return False


def _get_session():
    try:
        from common.robustness import create_resilient_session
        return create_resilient_session()
    except ImportError:
        import requests
        return requests.Session()


def _fetch_endpoint(session, endpoint: str, params: dict) -> list:
    """GET a pdufa.bio endpoint. Returns list of records or [] on error."""
    url = PDUFA_BIO_BASE + endpoint
    try:
        resp = session.get(url, params=params, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        body = resp.json()
    except Exception as e:
        logger.error("pdufa.bio %s request failed: %s", endpoint, e)
        return []

    if isinstance(body, list):
        return body
    if isinstance(body, dict):
        return body.get("data") or body.get("events") or []
    return []


def _map_record(raw: dict, event_type: str, as_of_date: date) -> Optional[dict]:
    """
    Map a pdufa.bio record to a cache dict.

    Returns None if:
    - ticker is missing / empty
    - date is None or date_precision != "day"
    - disclosed_at > as_of_date (leakage gate)
    """
    ticker = (raw.get("ticker") or "").upper().strip()
    if not ticker:
        return None

    # Only use records with a confirmed day-precision date
    if raw.get("date_precision") != "day":
        logger.debug("pdufa.bio: %s skipped — date_precision=%r", ticker, raw.get("date_precision"))
        return None

    event_date = raw.get("date")
    if not event_date or not isinstance(event_date, str) or len(event_date) < 10:
        return None
    event_date = event_date[:10]

    # PIT anchor: updated_at is the best available proxy for disclosure time
    updated_at = raw.get("updated_at") or raw.get("updatedAt") or ""
    if updated_at and len(updated_at) >= 10:
        disclosed_at = updated_at[:10]
        pit_tag = "pit_anchor:updated_at"
    else:
        disclosed_at = as_of_date.isoformat()
        pit_tag = "pit_anchor:as_of_date"

    # Leakage gate: skip if disclosed_at > as_of_date
    if disclosed_at > as_of_date.isoformat():
        logger.debug("pdufa.bio leakage gate: %s disclosed_at=%s > as_of=%s", ticker, disclosed_at, as_of_date)
        return None

    # Build tags
    tags = [event_type.lower().replace("_", "-"), "source:pdufa_bio", pit_tag]
    nct_id = raw.get("nct_id") or raw.get("nctId")
    if nct_id:
        tags.append(f"nct:{nct_id}")

    # Event name: name field, truncated to 120 chars
    name = (raw.get("name") or raw.get("indication") or "").strip()
    if name:
        prefix = event_type.replace("_", " ").title()
        event_name = f"{prefix}: {name}"[:120]
    else:
        event_name = event_type

    record: dict = {
        "ticker": ticker,
        "event_type": event_type,
        "event_date": event_date,
        "disclosed_at": disclosed_at,
        "event_name": event_name,
        "tags": tags,
    }
    return record


def _atomic_write(path: Path, records: list) -> None:
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


def collect_pdufa_bio_events(
    as_of_date: date,
    cache_dir: Path,
    pdufa_dates_path: Path,
    lookforward_days: int = DEFAULT_LOOKFORWARD_DAYS,
) -> Tuple[List[dict], List[dict]]:
    """
    Fetch pdufa.bio PDUFA and ADCOM events, write cache files.

    Returns (fda_regulatory_records, adcom_records).

    Short-circuits if both cache files already exist (e.g. BPW already ran).
    No API key required. Free anonymous tier: 1,000 req/day, 10/min burst.
    """
    cache_dir = Path(cache_dir)

    reg_path = cache_dir / f"fda_regulatory_{as_of_date.isoformat()}.json"
    adcom_path = cache_dir / f"adcom_calendar_{as_of_date.isoformat()}.json"

    # Short-circuit if both cache files already exist
    if reg_path.exists() and adcom_path.exists():
        try:
            reg_records = json.loads(reg_path.read_text(encoding="utf-8"))
            adcom_records = json.loads(adcom_path.read_text(encoding="utf-8"))
            logger.info(
                "pdufa.bio cache exists (or populated by BPW): fda_regulatory=%d adcom=%d",
                len(reg_records),
                len(adcom_records),
            )
            return reg_records, adcom_records
        except (json.JSONDecodeError, OSError):
            pass

    pdufa_entries = _load_pdufa_dates(pdufa_dates_path)
    pdufa_dedup_set = _build_pdufa_dedup_set(pdufa_entries)

    end_date = (as_of_date + timedelta(days=lookforward_days)).isoformat()
    base_params = {
        "from": as_of_date.isoformat(),
        "to": end_date,
        "limit": _MAX_LIMIT,
    }

    session = _get_session()

    # Fetch PDUFA dates
    logger.info("Fetching pdufa.bio /pdufa (as_of=%s, +%dd)...", as_of_date, lookforward_days)
    raw_pdufa = _fetch_endpoint(session, "/pdufa", base_params)
    logger.info("pdufa.bio /pdufa returned %d records", len(raw_pdufa))

    # Fetch ADCOM meetings
    logger.info("Fetching pdufa.bio /adcomm...")
    raw_adcomm = _fetch_endpoint(session, "/adcomm", base_params)
    logger.info("pdufa.bio /adcomm returned %d records", len(raw_adcomm))

    reg_records: List[dict] = []
    adcom_records: List[dict] = []
    skipped = 0

    for raw in raw_pdufa:
        record = _map_record(raw, "PDUFA", as_of_date)
        if record is None:
            skipped += 1
            continue
        if _is_pdufa_duplicate(record["ticker"], record["event_date"], pdufa_dedup_set):
            logger.debug(
                "pdufa.bio PDUFA dedup: %s %s already in pdufa_dates.json",
                record["ticker"], record["event_date"],
            )
            skipped += 1
            continue
        reg_records.append(record)

    for raw in raw_adcomm:
        record = _map_record(raw, "FDA_ADCOM", as_of_date)
        if record is None:
            skipped += 1
            continue
        adcom_records.append(record)

    # Stable sort for deterministic output
    reg_records.sort(key=lambda r: (r.get("event_date", ""), r.get("ticker", "")))
    adcom_records.sort(key=lambda r: (r.get("event_date", ""), r.get("ticker", "")))

    logger.info(
        "pdufa.bio mapped: fda_regulatory=%d adcom=%d skipped=%d",
        len(reg_records), len(adcom_records), skipped,
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(reg_path, reg_records)
    _atomic_write(adcom_path, adcom_records)
    logger.info(
        "pdufa.bio cache written: %s (%d), %s (%d)",
        reg_path.name, len(reg_records), adcom_path.name, len(adcom_records),
    )

    return reg_records, adcom_records
