"""
euctr_collector.py - Collect clinical trial data from EU Clinical Trials Register (EUCTR).

Searches EUCTR via RSS feed endpoint (structured XML), parses results,
and normalizes to the unified TrialRecord schema.

Rate limit: 1.0s between requests. Timeout: 30s.
"""

import hashlib
import json
import logging
import os
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote_plus

import requests

from wake_robin_data_pipeline.collectors.trials_collector import (
    SPONSOR_ALIASES,
    _clean_company_name,
)

logger = logging.getLogger(__name__)

# Rate limiting
_RATE_LIMIT_SECS = 1.0
_REQUEST_TIMEOUT = 30

_EUCTR_RSS_URL = "https://www.clinicaltrialsregister.eu/ctr-search/rest/feed/bydates"

# Phase mapping: EUCTR text -> normalized
_PHASE_MAP = {
    "phase i": "PHASE1",
    "phase 1": "PHASE1",
    "phase ii": "PHASE2",
    "phase 2": "PHASE2",
    "phase iii": "PHASE3",
    "phase 3": "PHASE3",
    "phase iv": "PHASE4",
    "phase 4": "PHASE4",
}

# Status mapping: EUCTR text -> normalized
_STATUS_MAP = {
    "ongoing": "RECRUITING",
    "recruiting": "RECRUITING",
    "not yet recruiting": "RECRUITING",
    "active, not recruiting": "ACTIVE_NOT_RECRUITING",
    "completed": "COMPLETED",
    "terminated": "TERMINATED",
    "withdrawn": "TERMINATED",
    "suspended": "TERMINATED",
    "prematurely ended": "TERMINATED",
    "prohibited by ca": "TERMINATED",
}


def collect_euctr_trials(
    universe: list,
    as_of_date: date,
    cache_dir: Path,
    cache_only: bool = False,
) -> List[dict]:
    """Collect EUCTR trials for tickers in universe.

    Args:
        universe: List of ticker dicts with 'ticker' and 'company_name' keys.
        as_of_date: PIT date for cache naming.
        cache_dir: Root directory for EUCTR cache files.
        cache_only: If True, only read from cache (no network).

    Returns:
        List of normalized TrialRecord dicts.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    live_path = cache_dir / f"euctr_{as_of_date.isoformat()}.json"
    meta_path = cache_dir / f"euctr_{as_of_date.isoformat()}.meta.json"

    # Short-circuit if cache exists
    if live_path.exists():
        try:
            records = json.loads(live_path.read_text())
            logger.info("EUCTR cache hit: %s (%d records)", live_path.name, len(records))
            return records
        except (json.JSONDecodeError, OSError):
            pass

    if cache_only:
        logger.info("EUCTR cache_only=True, no cache found — returning empty")
        return []

    # Build universe map: lowercase sponsor name -> ticker
    universe_map = _build_universe_map(universe)

    fetched_at = datetime.now(timezone.utc).isoformat()
    raw_dir = cache_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_records: Dict[str, dict] = {}  # dedup by EudraCT number

    for ticker_dict in universe:
        ticker = ticker_dict.get("ticker", "")
        company_name = ticker_dict.get("company_name", "")
        if not ticker:
            continue

        # Get search terms: SPONSOR_ALIASES + cleaned company name
        search_terms = _get_search_terms(ticker, company_name)

        for term in search_terms:
            try:
                items = _search_euctr(term, raw_dir=raw_dir)
            except Exception as e:
                logger.warning("EUCTR search failed for '%s': %s", term, e)
                continue

            for raw_item in items:
                eudract = raw_item.get("eudract_number", "")
                if not eudract or eudract in all_records:
                    continue

                # Match sponsor to ticker
                matched_ticker = _match_ticker(
                    raw_item.get("sponsor", ""), universe_map
                )
                if matched_ticker is None:
                    matched_ticker = ticker  # searched by this ticker's alias

                record = _normalize_euctr_record(raw_item, matched_ticker, fetched_at)
                all_records[eudract] = record

    records = sorted(all_records.values(), key=lambda r: (r["ticker"], r["primary_id"]))

    # Atomic write
    fd, tmp_path = tempfile.mkstemp(dir=str(cache_dir), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(records, f, indent=2)
        os.replace(tmp_path, str(live_path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    # Write meta sidecar
    meta = {
        "schema": "euctr_cache.v1",
        "as_of_date": as_of_date.isoformat(),
        "record_count": len(records),
        "fetched_at_utc": fetched_at,
        "tickers_searched": len(universe),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info("EUCTR: %d records cached -> %s", len(records), live_path)
    return records


def _get_search_terms(ticker: str, company_name: str) -> List[str]:
    """Get search terms for a ticker from SPONSOR_ALIASES + company name."""
    terms = []
    if ticker in SPONSOR_ALIASES:
        terms.extend(SPONSOR_ALIASES[ticker])
    if company_name:
        cleaned = _clean_company_name(company_name)
        if cleaned and cleaned not in terms:
            terms.append(cleaned)
        if company_name not in terms:
            terms.append(company_name)
    return terms[:5]  # cap to avoid excessive queries


def _search_euctr(sponsor_name: str, page: int = 1, raw_dir: Path | None = None) -> List[dict]:
    """Search EUCTR RSS feed by sponsor name.

    Returns list of parsed item dicts from RSS XML.
    """
    params = {"query": sponsor_name, "page": page}
    time.sleep(_RATE_LIMIT_SECS)

    resp = requests.get(_EUCTR_RSS_URL, params=params, timeout=_REQUEST_TIMEOUT)
    resp.raise_for_status()

    # Cache raw response
    if raw_dir is not None:
        query_hash = hashlib.md5(f"{sponsor_name}_{page}".encode()).hexdigest()[:8]
        raw_path = raw_dir / f"euctr_rss_{query_hash}.xml"
        raw_path.write_text(resp.text, encoding="utf-8")

    return _parse_rss_response(resp.text)


def _parse_rss_response(xml_text: str) -> List[dict]:
    """Parse EUCTR RSS XML response into list of item dicts."""
    items = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.warning("EUCTR RSS parse error: %s", e)
        return []

    for item in root.iter("item"):
        parsed = _parse_rss_item(item)
        if parsed:
            items.append(parsed)
    return items


def _parse_rss_item(item: ET.Element) -> Optional[dict]:
    """Extract fields from a single RSS <item> element."""
    title = (item.findtext("title") or "").strip()
    link = (item.findtext("link") or "").strip()
    description = (item.findtext("description") or "").strip()

    # Extract EudraCT number from link or title
    eudract = _extract_eudract_number(link) or _extract_eudract_number(title)
    if not eudract:
        return None

    # Parse description fields (semi-structured text)
    fields = _parse_description(description)

    return {
        "eudract_number": eudract,
        "title": title,
        "link": link,
        "sponsor": fields.get("sponsor", ""),
        "sponsor_protocol": fields.get("sponsor_protocol", ""),
        "status": fields.get("status", ""),
        "phase": fields.get("phase", ""),
        "start_date": fields.get("start_date", ""),
        "medical_condition": fields.get("medical_condition", ""),
        "countries": fields.get("countries", []),
    }


def _extract_eudract_number(text: str) -> Optional[str]:
    """Extract EudraCT number (YYYY-NNNNNN-CC) from text."""
    m = re.search(r"(\d{4}-\d{6}-\d{2})", text)
    return m.group(1) if m else None


def _parse_description(desc: str) -> dict:
    """Parse semi-structured description text from RSS item."""
    fields: dict = {}
    for line in desc.split("\n"):
        line = line.strip()
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()

        if "sponsor" in key and "protocol" not in key:
            fields["sponsor"] = value
        elif "protocol" in key:
            fields["sponsor_protocol"] = value
        elif "status" in key or "trial status" in key:
            fields["status"] = value
        elif "phase" in key:
            fields["phase"] = value
        elif "start date" in key:
            fields["start_date"] = value
        elif "condition" in key or "disease" in key:
            fields["medical_condition"] = value
        elif "countr" in key:
            fields["countries"] = [c.strip() for c in value.split(",") if c.strip()]

    return fields


def _normalize_euctr_record(raw: dict, ticker: str, fetched_at: str) -> dict:
    """Normalize a raw EUCTR item to the unified TrialRecord schema."""
    eudract = raw.get("eudract_number", "")
    phase_raw = (raw.get("phase") or "").lower().strip()
    phase = _PHASE_MAP.get(phase_raw, "NA")

    status_raw = (raw.get("status") or "").lower().strip()
    status = _STATUS_MAP.get(status_raw, "UNKNOWN")

    start_date = _normalize_date(raw.get("start_date", ""))
    link = raw.get("link", "")
    if not link and eudract:
        link = f"https://www.clinicaltrialsregister.eu/ctr-search/trial/{eudract}/results"

    conditions = []
    mc = raw.get("medical_condition", "")
    if mc:
        conditions = [c.strip() for c in mc.split(";") if c.strip()]

    secondary_ids = []
    sp = raw.get("sponsor_protocol", "")
    if sp:
        secondary_ids.append(f"protocol:{sp}")

    return {
        "registry": "euctr",
        "primary_id": eudract,
        "secondary_ids": secondary_ids,
        "ticker": ticker,
        "sponsor": {"name": raw.get("sponsor", ""), "country": ""},
        "title": raw.get("title", ""),
        "conditions": conditions,
        "phase": phase,
        "status": status,
        "study_type": "INTERVENTIONAL",
        "countries": raw.get("countries", []),
        "start_date": start_date,
        "primary_completion_date": None,
        "completion_date": None,
        "results_posted_date": None,
        "first_posted": start_date,
        "last_update_posted": start_date,  # best available PIT anchor
        "source_url": link,
        "fetched_at_utc": fetched_at,
    }


def _build_universe_map(universe: list) -> Dict[str, str]:
    """Build lowercase sponsor name -> ticker map from universe + SPONSOR_ALIASES."""
    umap: Dict[str, str] = {}
    for ticker_dict in universe:
        ticker = ticker_dict.get("ticker", "")
        company_name = ticker_dict.get("company_name", "")
        if not ticker:
            continue
        if company_name:
            umap[company_name.lower()] = ticker
            cleaned = _clean_company_name(company_name).lower()
            if cleaned:
                umap[cleaned] = ticker
        if ticker in SPONSOR_ALIASES:
            for alias in SPONSOR_ALIASES[ticker]:
                umap[alias.lower()] = ticker
    return umap


def _match_ticker(sponsor_name: str, universe_map: Dict[str, str]) -> Optional[str]:
    """Match sponsor name against universe map (exact lowercase match)."""
    if not sponsor_name:
        return None
    key = sponsor_name.strip().lower()
    if key in universe_map:
        return universe_map[key]
    # Try cleaning the sponsor name
    cleaned = _clean_company_name(sponsor_name).lower()
    if cleaned in universe_map:
        return universe_map[cleaned]
    return None


def _normalize_date(raw: str) -> Optional[str]:
    """Normalize a date string to YYYY-MM-DD or return None."""
    if not raw:
        return None
    raw = raw.strip()
    # Try common formats
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d %B %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    # Try ISO prefix
    if re.match(r"\d{4}-\d{2}-\d{2}", raw):
        return raw[:10]
    return None
