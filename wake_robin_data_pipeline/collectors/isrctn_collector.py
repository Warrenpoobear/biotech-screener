"""
isrctn_collector.py - Collect clinical trial data from ISRCTN registry.

Searches ISRCTN by sponsor name via GET with query params, parses HTML
results, extracts structured data from detail pages (JSON-LD preferred,
HTML table fallback), normalizes to unified TrialRecord schema.

Rate limit: 1.0s between requests. Timeout: 30s.
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
_RATE_LIMIT_SECS = 1.0
_REQUEST_TIMEOUT = 30

_ISRCTN_SEARCH_URL = "https://www.isrctn.com/search"
_ISRCTN_BASE_URL = "https://www.isrctn.com"

# Phase mapping
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
    "0": "NA",
    "not applicable": "NA",
}

# Status mapping
_STATUS_MAP = {
    "recruiting": "RECRUITING",
    "no longer recruiting": "ACTIVE_NOT_RECRUITING",
    "suspended": "TERMINATED",
    "stopped": "TERMINATED",
    "completed": "COMPLETED",
}


def collect_isrctn_trials(
    universe: list,
    as_of_date: date,
    cache_dir: Path,
    cache_only: bool = False,
) -> List[dict]:
    """Collect ISRCTN trials for tickers in universe.

    Args:
        universe: List of ticker dicts with 'ticker' and 'company_name' keys.
        as_of_date: PIT date for cache naming.
        cache_dir: Root directory for ISRCTN cache files.
        cache_only: If True, only read from cache (no network).

    Returns:
        List of normalized TrialRecord dicts.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    live_path = cache_dir / f"isrctn_{as_of_date.isoformat()}.json"
    meta_path = cache_dir / f"isrctn_{as_of_date.isoformat()}.meta.json"

    # Short-circuit if cache exists
    if live_path.exists():
        try:
            records = json.loads(live_path.read_text())
            logger.info("ISRCTN cache hit: %s (%d records)", live_path.name, len(records))
            return records
        except (json.JSONDecodeError, OSError):
            pass

    if cache_only:
        logger.info("ISRCTN cache_only=True, no cache found — returning empty")
        return []

    # Build universe map
    universe_map = _build_universe_map(universe)

    fetched_at = datetime.now(timezone.utc).isoformat()
    raw_dir = cache_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_records: Dict[str, dict] = {}  # dedup by ISRCTN ID

    for ticker_dict in universe:
        ticker = ticker_dict.get("ticker", "")
        company_name = ticker_dict.get("company_name", "")
        if not ticker:
            continue

        search_terms = _get_search_terms(ticker, company_name)

        for term in search_terms:
            try:
                isrctn_ids = _search_isrctn(term, raw_dir=raw_dir)
            except Exception as e:
                logger.warning("ISRCTN search failed for '%s': %s", term, e)
                continue

            for isrctn_id in isrctn_ids:
                if isrctn_id in all_records:
                    continue

                try:
                    detail = _fetch_isrctn_detail(isrctn_id, raw_dir=raw_dir)
                except Exception as e:
                    logger.warning("ISRCTN detail fetch failed for %s: %s", isrctn_id, e)
                    continue

                if not detail:
                    continue

                matched_ticker = _match_ticker(
                    detail.get("sponsor", ""), universe_map
                )
                if matched_ticker is None:
                    matched_ticker = ticker

                record = _normalize_isrctn_record(detail, matched_ticker, fetched_at)
                all_records[isrctn_id] = record

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
        "schema": "isrctn_cache.v1",
        "as_of_date": as_of_date.isoformat(),
        "record_count": len(records),
        "fetched_at_utc": fetched_at,
        "tickers_searched": len(universe),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info("ISRCTN: %d records cached -> %s", len(records), live_path)
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


def _search_isrctn(sponsor_name: str, page: int = 1, raw_dir: Path | None = None) -> List[str]:
    """Search ISRCTN by sponsor name.

    Returns list of ISRCTN IDs from search results.
    """
    params = {
        "q": sponsor_name,
        "filters": "primaryStudyDesign:Interventional",
        "page": page,
        "pageSize": 100,
    }

    time.sleep(_RATE_LIMIT_SECS)

    resp = requests.get(
        _ISRCTN_SEARCH_URL,
        params=params,
        timeout=_REQUEST_TIMEOUT,
        allow_redirects=True,
    )
    resp.raise_for_status()

    # Cache raw response
    if raw_dir is not None:
        query_hash = hashlib.md5(f"{sponsor_name}_{page}".encode()).hexdigest()[:8]
        raw_path = raw_dir / f"isrctn_search_{query_hash}.html"
        raw_path.write_text(resp.text, encoding="utf-8")

    return _parse_search_results(resp.text)


def _parse_search_results(html: str) -> List[str]:
    """Extract ISRCTN IDs from search results HTML."""
    # ISRCTN IDs appear as links: /ISRCTN12345678
    pattern = r'href="[^"]*/(ISRCTN\d+)"'
    matches = re.findall(pattern, html)
    # Deduplicate while preserving order
    seen = set()
    result = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            result.append(m)
    return result


def _fetch_isrctn_detail(isrctn_id: str, raw_dir: Path | None = None) -> Optional[dict]:
    """Fetch trial detail from ISRCTN.

    Looks for JSON-LD first, falls back to HTML table parsing.
    """
    time.sleep(_RATE_LIMIT_SECS)

    resp = requests.get(
        f"{_ISRCTN_BASE_URL}/{isrctn_id}",
        timeout=_REQUEST_TIMEOUT,
        allow_redirects=True,
    )
    resp.raise_for_status()

    # Cache raw response
    if raw_dir is not None:
        raw_path = raw_dir / f"isrctn_detail_{isrctn_id}.html"
        raw_path.write_text(resp.text, encoding="utf-8")

    # Try JSON-LD first
    detail = _extract_json_ld(resp.text, isrctn_id)
    if detail:
        return detail

    # Fallback to HTML table parsing
    return _parse_html_detail(resp.text, isrctn_id)


def _extract_json_ld(html: str, isrctn_id: str) -> Optional[dict]:
    """Extract trial data from JSON-LD script tag."""
    pattern = r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>'
    matches = re.findall(pattern, html, re.DOTALL)

    for match in matches:
        try:
            data = json.loads(match)
        except json.JSONDecodeError:
            continue

        # Handle both single object and array
        items = data if isinstance(data, list) else [data]
        for item in items:
            if item.get("@type") in ("MedicalTrial", "ClinicalTrial", "MedicalStudy"):
                return _normalize_json_ld(item, isrctn_id)

    return None


def _normalize_json_ld(data: dict, isrctn_id: str) -> dict:
    """Normalize JSON-LD data to intermediate detail dict."""
    sponsor = ""
    sponsor_obj = data.get("sponsor", {})
    if isinstance(sponsor_obj, dict):
        sponsor = sponsor_obj.get("name", "")
    elif isinstance(sponsor_obj, str):
        sponsor = sponsor_obj

    status = (data.get("status") or data.get("trialStatus") or "").lower()

    phase_raw = data.get("phase") or data.get("trialDesign", {}).get("phase", "")
    if isinstance(phase_raw, dict):
        phase_raw = phase_raw.get("name", "")

    conditions = []
    hc = data.get("healthCondition") or data.get("about", [])
    if isinstance(hc, list):
        for c in hc:
            if isinstance(c, dict):
                conditions.append(c.get("name", ""))
            elif isinstance(c, str):
                conditions.append(c)
    elif isinstance(hc, str):
        conditions = [hc]

    countries = []
    loc = data.get("studyLocation") or data.get("location", [])
    if isinstance(loc, list):
        for l in loc:
            if isinstance(l, dict):
                countries.append(l.get("name", ""))
            elif isinstance(l, str):
                countries.append(l)

    return {
        "isrctn_id": isrctn_id,
        "title": data.get("name", ""),
        "sponsor": sponsor,
        "status": status,
        "phase": str(phase_raw).lower() if phase_raw else "",
        "conditions": [c for c in conditions if c],
        "countries": countries,
        "start_date": data.get("startDate", ""),
        "end_date": data.get("endDate", ""),
        "date_registered": data.get("datePublished", ""),
        "last_edited": data.get("dateModified", ""),
        "secondary_ids": _extract_secondary_ids(data),
        "source_url": data.get("url", f"{_ISRCTN_BASE_URL}/{isrctn_id}"),
    }


def _extract_secondary_ids(data: dict) -> List[str]:
    """Extract secondary IDs (NCT, EudraCT, etc.) from JSON-LD."""
    ids = []
    for field in ("identifier", "secondaryId", "sameAs"):
        val = data.get(field, [])
        if isinstance(val, str):
            val = [val]
        for v in val:
            if isinstance(v, dict):
                v = v.get("value", "") or v.get("@value", "")
            if isinstance(v, str) and v:
                ids.append(v)
    return ids


def _parse_html_detail(html: str, isrctn_id: str) -> Optional[dict]:
    """Parse trial detail from HTML tables (fallback)."""
    detail: dict = {
        "isrctn_id": isrctn_id,
        "title": "",
        "sponsor": "",
        "status": "",
        "phase": "",
        "conditions": [],
        "countries": [],
        "start_date": "",
        "end_date": "",
        "date_registered": "",
        "last_edited": "",
        "secondary_ids": [],
        "source_url": f"{_ISRCTN_BASE_URL}/{isrctn_id}",
    }

    # Extract title from <h1>
    title_match = re.search(r'<h1[^>]*>(.*?)</h1>', html, re.DOTALL)
    if title_match:
        detail["title"] = re.sub(r'<[^>]+>', '', title_match.group(1)).strip()

    # Extract table rows: <dt>Label</dt><dd>Value</dd> or <th>Label</th><td>Value</td>
    for pattern in [
        r'<dt[^>]*>(.*?)</dt>\s*<dd[^>]*>(.*?)</dd>',
        r'<th[^>]*>(.*?)</th>\s*<td[^>]*>(.*?)</td>',
    ]:
        for match in re.finditer(pattern, html, re.DOTALL):
            label = re.sub(r'<[^>]+>', '', match.group(1)).strip().lower()
            value = re.sub(r'<[^>]+>', '', match.group(2)).strip()

            if "sponsor" in label:
                detail["sponsor"] = value
            elif "status" in label and "overall" in label:
                detail["status"] = value.lower()
            elif "phase" in label:
                detail["phase"] = value.lower()
            elif "condition" in label:
                detail["conditions"] = [c.strip() for c in value.split(",") if c.strip()]
            elif "countr" in label:
                detail["countries"] = [c.strip() for c in value.split(",") if c.strip()]
            elif "date" in label and "registered" in label:
                detail["date_registered"] = value
            elif "last edited" in label:
                detail["last_edited"] = value
            elif "start date" in label or "overall start" in label:
                detail["start_date"] = value
            elif "end date" in label or "overall end" in label:
                detail["end_date"] = value

    # Extract NCT or EudraCT IDs from page text
    nct_matches = re.findall(r'(NCT\d{8})', html)
    eudract_matches = re.findall(r'(\d{4}-\d{6}-\d{2})', html)
    detail["secondary_ids"] = list(set(nct_matches + eudract_matches))

    return detail if detail["title"] else None


def _normalize_isrctn_record(raw: dict, ticker: str, fetched_at: str) -> dict:
    """Normalize an ISRCTN detail dict to the unified TrialRecord schema."""
    isrctn_id = raw.get("isrctn_id", "")

    phase_raw = (raw.get("phase") or "").lower().strip()
    phase = _PHASE_MAP.get(phase_raw, "NA")

    status_raw = (raw.get("status") or "").lower().strip()
    status = _STATUS_MAP.get(status_raw, "UNKNOWN")

    start_date = _normalize_date(raw.get("start_date", ""))
    end_date = _normalize_date(raw.get("end_date", ""))
    date_registered = _normalize_date(raw.get("date_registered", ""))
    last_edited = _normalize_date(raw.get("last_edited", ""))

    return {
        "registry": "isrctn",
        "primary_id": isrctn_id,
        "secondary_ids": raw.get("secondary_ids", []),
        "ticker": ticker,
        "sponsor": {"name": raw.get("sponsor", ""), "country": ""},
        "title": raw.get("title", ""),
        "conditions": raw.get("conditions", []),
        "phase": phase,
        "status": status,
        "study_type": "INTERVENTIONAL",
        "countries": raw.get("countries", []),
        "start_date": start_date,
        "primary_completion_date": end_date,
        "completion_date": end_date,
        "results_posted_date": None,
        "first_posted": date_registered,
        "last_update_posted": last_edited or date_registered,
        "source_url": raw.get("source_url", f"{_ISRCTN_BASE_URL}/{isrctn_id}"),
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


def _normalize_date(raw: str) -> Optional[str]:
    """Normalize a date string to YYYY-MM-DD or return None."""
    if not raw:
        return None
    raw = raw.strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d %B %Y", "%B %d, %Y",
                "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(raw[:max(10, len(raw))], fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    if re.match(r"\d{4}-\d{2}-\d{2}", raw):
        return raw[:10]
    return None
