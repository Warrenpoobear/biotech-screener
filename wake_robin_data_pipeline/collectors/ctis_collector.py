"""
ctis_collector.py - Collect clinical trial data from CTIS (Clinical Trials Information System).

Searches CTIS public API via POST search endpoint, paginates results,
and normalizes to the unified TrialRecord schema.

Rate limit: 0.5s between requests. Timeout: 30s.
"""

import hashlib
import json
import logging
import os
import re
import tempfile
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests

from wake_robin_data_pipeline.collectors.trials_collector import (
    SPONSOR_ALIASES,
    _clean_company_name,
)

logger = logging.getLogger(__name__)

# Rate limiting
_RATE_LIMIT_SECS = 0.5
_REQUEST_TIMEOUT = 30
_MAX_PAGES_PER_SPONSOR = 10
_PAGE_SIZE = 100

_CTIS_SEARCH_URL = "https://euclinicaltrials.eu/ctis-public-api/search"
_CTIS_DETAIL_URL = "https://euclinicaltrials.eu/ctis-public-api/retrieve"

# Phase mapping: CTIS text -> normalized
_PHASE_MAP = {
    "phase i": "PHASE1",
    "phase 1": "PHASE1",
    "phase ii": "PHASE2",
    "phase 2": "PHASE2",
    "phase iii": "PHASE3",
    "phase 3": "PHASE3",
    "phase iv": "PHASE4",
    "phase 4": "PHASE4",
    "i": "PHASE1",
    "ii": "PHASE2",
    "iii": "PHASE3",
    "iv": "PHASE4",
}

# Status integer codes observed from CTIS API -> normalized strings
_STATUS_CODE_MAP = {
    1: "RECRUITING",
    2: "ACTIVE_NOT_RECRUITING",
    3: "COMPLETED",
    4: "TERMINATED",
    5: "UNKNOWN",
}

# Status text mapping
_STATUS_TEXT_MAP = {
    "ongoing": "RECRUITING",
    "authorised": "RECRUITING",
    "recruiting": "RECRUITING",
    "active, not recruiting": "ACTIVE_NOT_RECRUITING",
    "not recruiting": "ACTIVE_NOT_RECRUITING",
    "completed": "COMPLETED",
    "ended": "TERMINATED",
    "terminated": "TERMINATED",
    "withdrawn": "TERMINATED",
    "suspended": "TERMINATED",
    "lapsed": "TERMINATED",
    "halted": "TERMINATED",
}


def collect_ctis_trials(
    universe: list,
    as_of_date: date,
    cache_dir: Path,
    cache_only: bool = False,
) -> List[dict]:
    """Collect CTIS trials for tickers in universe.

    Args:
        universe: List of ticker dicts with 'ticker' and 'company_name' keys.
        as_of_date: PIT date for cache naming.
        cache_dir: Root directory for CTIS cache files.
        cache_only: If True, only read from cache (no network).

    Returns:
        List of normalized TrialRecord dicts.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    live_path = cache_dir / f"ctis_{as_of_date.isoformat()}.json"
    meta_path = cache_dir / f"ctis_{as_of_date.isoformat()}.meta.json"

    # Short-circuit if cache exists
    if live_path.exists():
        try:
            records = json.loads(live_path.read_text())
            logger.info("CTIS cache hit: %s (%d records)", live_path.name, len(records))
            return records
        except (json.JSONDecodeError, OSError):
            pass

    if cache_only:
        logger.info("CTIS cache_only=True, no cache found — returning empty")
        return []

    # Build universe map
    universe_map = _build_universe_map(universe)

    fetched_at = datetime.now(timezone.utc).isoformat()
    raw_dir = cache_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_records: Dict[str, dict] = {}  # dedup by CT number

    for ticker_dict in universe:
        ticker = ticker_dict.get("ticker", "")
        company_name = ticker_dict.get("company_name", "")
        if not ticker:
            continue

        search_terms = _get_search_terms(ticker, company_name)

        for term in search_terms:
            page = 1
            while page <= _MAX_PAGES_PER_SPONSOR:
                try:
                    response = _search_ctis(term, page=page, page_size=_PAGE_SIZE,
                                            raw_dir=raw_dir)
                except Exception as e:
                    logger.warning("CTIS search failed for '%s' page %d: %s", term, page, e)
                    break

                data_list = response.get("data", [])
                if not data_list:
                    break

                for raw_item in data_list:
                    ct_number = raw_item.get("ctNumber", "")
                    if not ct_number or ct_number in all_records:
                        continue

                    matched_ticker = _match_ticker(
                        raw_item.get("sponsorName", ""), universe_map
                    )
                    if matched_ticker is None:
                        matched_ticker = ticker

                    record = _normalize_ctis_record(raw_item, matched_ticker, fetched_at)
                    all_records[ct_number] = record

                # Check pagination
                pagination = response.get("pagination", {})
                total_pages = pagination.get("totalPages", 1)
                if page >= total_pages:
                    break
                page += 1

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
        "schema": "ctis_cache.v1",
        "as_of_date": as_of_date.isoformat(),
        "record_count": len(records),
        "fetched_at_utc": fetched_at,
        "tickers_searched": len(universe),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info("CTIS: %d records cached -> %s", len(records), live_path)
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
    return terms[:5]


def _search_ctis(
    sponsor_name: str,
    page: int = 1,
    page_size: int = 100,
    raw_dir: Path | None = None,
) -> dict:
    """Search CTIS public API by sponsor name.

    POST to search endpoint with pagination and sponsor filter.
    Returns parsed JSON response with 'data' list + 'pagination' metadata.
    """
    body = {
        "pagination": {"page": page, "size": page_size},
        "sort": {"property": "decisionDate", "direction": "DESC"},
        "searchCriteria": {"sponsor": sponsor_name},
    }

    time.sleep(_RATE_LIMIT_SECS)

    resp = requests.post(
        _CTIS_SEARCH_URL,
        json=body,
        timeout=_REQUEST_TIMEOUT,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    resp.raise_for_status()

    result = resp.json()

    # Cache raw response
    if raw_dir is not None:
        query_hash = hashlib.md5(f"{sponsor_name}_{page}".encode()).hexdigest()[:8]
        raw_path = raw_dir / f"ctis_search_{query_hash}.json"
        raw_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


def _fetch_ctis_detail(ct_number: str, raw_dir: Path | None = None) -> dict:
    """Fetch full trial detail from CTIS API.

    GET /retrieve/{ct_number}
    Returns parsed JSON with full trial metadata.
    """
    time.sleep(_RATE_LIMIT_SECS)

    resp = requests.get(
        f"{_CTIS_DETAIL_URL}/{ct_number}",
        timeout=_REQUEST_TIMEOUT,
        headers={"Accept": "application/json"},
    )
    resp.raise_for_status()

    result = resp.json()

    if raw_dir is not None:
        raw_path = raw_dir / f"ctis_detail_{ct_number.replace('-', '_')}.json"
        raw_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


def _normalize_ctis_record(raw: dict, ticker: str, fetched_at: str) -> dict:
    """Normalize a CTIS search result to the unified TrialRecord schema."""
    ct_number = raw.get("ctNumber", "")

    # Phase normalization
    phase_raw = (raw.get("trialPhase") or raw.get("phase") or "").lower().strip()
    phase = _PHASE_MAP.get(phase_raw, "NA")

    # Status normalization - try integer code first, then text
    status_raw = raw.get("trialStatus") or raw.get("overallStatus") or ""
    if isinstance(status_raw, int):
        status = _STATUS_CODE_MAP.get(status_raw, "UNKNOWN")
    else:
        status = _STATUS_TEXT_MAP.get(str(status_raw).lower().strip(), "UNKNOWN")

    # Date parsing: CTIS uses "dd/mm/yyyy" or ISO
    decision_date = _normalize_ctis_date(raw.get("decisionDate", ""))
    start_date = _normalize_ctis_date(raw.get("startDate", ""))
    end_date = _normalize_ctis_date(raw.get("endDate", ""))

    # Sponsor info
    sponsor_name = raw.get("sponsorName", "")
    sponsor_country = raw.get("sponsorCountry", "")

    # Secondary IDs
    secondary_ids = []
    eudract = raw.get("eudractNumber", "")
    if eudract:
        secondary_ids.append(eudract)
    nct_id = raw.get("nctNumber") or raw.get("nctId") or ""
    if nct_id:
        secondary_ids.append(nct_id)

    # Countries
    countries = raw.get("memberStates", [])
    if isinstance(countries, str):
        countries = [c.strip() for c in countries.split(",") if c.strip()]

    # Conditions
    conditions = []
    condition_raw = raw.get("medicalCondition") or raw.get("therapeuticArea") or ""
    if condition_raw:
        conditions = [c.strip() for c in condition_raw.split(";") if c.strip()]

    title = raw.get("title") or raw.get("trialTitle") or ""

    return {
        "registry": "ctis",
        "primary_id": ct_number,
        "secondary_ids": secondary_ids,
        "ticker": ticker,
        "sponsor": {"name": sponsor_name, "country": sponsor_country},
        "title": title,
        "conditions": conditions,
        "phase": phase,
        "status": status,
        "study_type": "INTERVENTIONAL",
        "countries": countries,
        "start_date": start_date,
        "primary_completion_date": end_date,
        "completion_date": end_date,
        "results_posted_date": None,
        "first_posted": decision_date,
        "last_update_posted": decision_date or start_date,
        "source_url": f"https://euclinicaltrials.eu/ctis-public/view/{ct_number}" if ct_number else "",
        "fetched_at_utc": fetched_at,
    }


def _normalize_ctis_date(raw: str) -> Optional[str]:
    """Normalize CTIS date formats to YYYY-MM-DD."""
    if not raw:
        return None
    raw = str(raw).strip()
    # Try dd/mm/yyyy
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%d-%m-%Y"):
        try:
            return datetime.strptime(raw[:10], fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    if re.match(r"\d{4}-\d{2}-\d{2}", raw):
        return raw[:10]
    return None


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
    """Match sponsor name against universe map."""
    if not sponsor_name:
        return None
    key = sponsor_name.strip().lower()
    if key in universe_map:
        return universe_map[key]
    cleaned = _clean_company_name(sponsor_name).lower()
    if cleaned in universe_map:
        return universe_map[cleaned]
    return None
