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
from datetime import date, datetime, timedelta, timezone
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

# Phase mapping: CTIS trialPhase field uses descriptive text
# e.g. "Therapeutic exploratory (Phase II)", "Therapeutic confirmatory (Phase III)"
_PHASE_PATTERNS = [
    (re.compile(r"phase\s*i(?:v|4)", re.IGNORECASE), "PHASE4"),
    (re.compile(r"phase\s*iii|phase\s*3|confirmatory", re.IGNORECASE), "PHASE3"),
    (re.compile(r"phase\s*ii|phase\s*2|exploratory", re.IGNORECASE), "PHASE2"),
    (re.compile(r"phase\s*i|phase\s*1|human\s*pharmacology", re.IGNORECASE), "PHASE1"),
]

# Status integer codes observed from CTIS API -> normalized strings
# Reverse-engineered from live responses: ctStatus field
_STATUS_CODE_MAP = {
    1: "RECRUITING",       # Authorised - recruiting
    2: "RECRUITING",       # Authorised - not yet recruiting
    3: "ACTIVE_NOT_RECRUITING",  # Authorised - no longer recruiting
    4: "COMPLETED",        # Completed
    5: "TERMINATED",       # Ended / Terminated
    6: "TERMINATED",       # Lapsed
    7: "TERMINATED",       # Suspended / Halted
    8: "RECRUITING",       # Authorised (general)
    9: "TERMINATED",       # Withdrawn
}

# Status text mapping (fallback)
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

# Detail endpoint ctStatus text -> normalized status
# The detail endpoint returns descriptive text (not integer codes like search)
_DETAIL_STATUS_MAP = {
    "authorised": "RECRUITING",
    "ongoing": "RECRUITING",
    "recruiting": "RECRUITING",
    "not yet recruiting": "RECRUITING",
    "no longer recruiting": "ACTIVE_NOT_RECRUITING",
    "active, not recruiting": "ACTIVE_NOT_RECRUITING",
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
    enrich_detail: bool = True,
) -> List[dict]:
    """Collect CTIS trials for tickers in universe.

    Args:
        universe: List of ticker dicts with 'ticker' and 'company_name' keys.
        as_of_date: PIT date for cache naming.
        cache_dir: Root directory for CTIS cache files.
        cache_only: If True, only read from cache (no network).
        enrich_detail: If True, call detail endpoint for records missing
            primary_completion_date to get corrected status, NCT IDs,
            and future completion dates.

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
                        raw_item.get("sponsor") or raw_item.get("sponsorName") or "",
                        universe_map,
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

    # --- Detail enrichment pass ---
    enrich_candidates = 0
    enrich_ok = 0
    enrich_errors = 0
    if enrich_detail:
        candidates = [
            r for r in records if not r.get("primary_completion_date")
        ]
        enrich_candidates = len(candidates)
        logger.info(
            "CTIS enrichment: %d/%d records missing primary_completion_date",
            enrich_candidates, len(records),
        )
        for rec in candidates:
            ct_number = rec["primary_id"]
            try:
                detail = _fetch_ctis_detail(ct_number, raw_dir=raw_dir)
                _enrich_record_from_detail(rec, detail, as_of_date)
                enrich_ok += 1
            except Exception as e:
                enrich_errors += 1
                logger.warning(
                    "CTIS detail enrichment failed for %s: %s", ct_number, e,
                )
        logger.info(
            "CTIS enrichment done: %d enriched, %d errors (of %d candidates)",
            enrich_ok, enrich_errors, enrich_candidates,
        )

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
        "enrichment": {
            "enabled": enrich_detail,
            "candidates": enrich_candidates,
            "enriched": enrich_ok,
            "errors": enrich_errors,
        },
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


# ---------------------------------------------------------------------------
# Detail enrichment helpers
# ---------------------------------------------------------------------------


def _extract_detail_status(detail: dict) -> str:
    """Map detail endpoint ctStatus text to normalized status.

    The detail endpoint returns descriptive text like "Authorised", "Ended".
    Case-insensitive lookup via _DETAIL_STATUS_MAP.
    """
    raw = detail.get("ctStatus") or ""
    if isinstance(raw, int):
        return _STATUS_CODE_MAP.get(raw, "UNKNOWN")
    return _DETAIL_STATUS_MAP.get(str(raw).lower().strip(), "UNKNOWN")


def _extract_nct_id(detail: dict) -> Optional[str]:
    """Extract NCT number from nested detail structure.

    Path: authorizedApplication.authorizedPartI.trialDetails
          .clinicalTrialIdentifiers.secondaryIdentifyingNumbers
          .nctNumber.number
    """
    nct = (
        detail
        .get("authorizedApplication", {})
        .get("authorizedPartI", {})
        .get("trialDetails", {})
        .get("clinicalTrialIdentifiers", {})
        .get("secondaryIdentifyingNumbers", {})
        .get("nctNumber", {})
        .get("number")
    )
    if nct and isinstance(nct, str) and nct.strip():
        return nct.strip()
    return None


def _collect_milestone_dates(
    events_block: dict, notification_type: str,
) -> List[str]:
    """Gather normalized dates for a milestone type across all countries.

    Scans events_block["trialEvents"] list for entries matching the given
    notificationType and collects their dates (YYYY-MM-DD normalized).
    """
    dates = []
    trial_events = events_block.get("trialEvents") or []
    for event in trial_events:
        if not isinstance(event, dict):
            continue
        if (event.get("notificationType") or "").upper() != notification_type.upper():
            continue
        raw_date = event.get("date") or event.get("eventDate") or ""
        normalized = _normalize_ctis_date(str(raw_date))
        if normalized:
            dates.append(normalized)
    return sorted(set(dates))


def _parse_treatment_duration_days(detail: dict) -> Optional[int]:
    """Parse study treatment duration from periodDetails descriptions.

    Looks for patterns like "12 weeks", "6 months", "180 days" in the
    description fields of periodDetails entries.
    """
    period_details = detail.get("periodDetails") or []
    for period in period_details:
        desc = (period.get("description") or "").lower()
        if not desc:
            continue
        m_weeks = re.search(r"(\d+)\s*week", desc)
        if m_weeks:
            return int(m_weeks.group(1)) * 7
        m_months = re.search(r"(\d+)\s*month", desc)
        if m_months:
            return int(m_months.group(1)) * 30
        m_days = re.search(r"(\d+)\s*day", desc)
        if m_days:
            return int(m_days.group(1))
    return None


def _estimate_primary_completion_date(
    detail: dict, as_of_date: date,
) -> Optional[str]:
    """Estimate primary completion date from detail endpoint data.

    Cascade:
    1. Latest END_OF_TRIAL date across countries (prefer future dates)
    2. trialGlobalEndDate if present
    3. Latest END_OF_RECRUITMENT + treatment duration (heuristic)
    4. None
    """
    as_of_str = as_of_date.isoformat()
    events_block = detail.get("events") or {}

    # 1. END_OF_TRIAL milestone dates
    eot_dates = _collect_milestone_dates(events_block, "END_OF_TRIAL")
    if eot_dates:
        future = [d for d in eot_dates if d > as_of_str]
        if future:
            return future[-1]  # latest future
        return eot_dates[-1]  # latest past

    # 2. trialGlobalEndDate
    global_end = _normalize_ctis_date(detail.get("trialGlobalEndDate") or "")
    if global_end:
        return global_end

    # 3. END_OF_RECRUITMENT + treatment duration
    eor_dates = _collect_milestone_dates(events_block, "END_OF_RECRUITMENT")
    if eor_dates:
        duration_days = _parse_treatment_duration_days(detail)
        if duration_days:
            latest_eor = eor_dates[-1]
            try:
                eor_dt = datetime.strptime(latest_eor, "%Y-%m-%d").date()
                pcd = eor_dt + timedelta(days=duration_days)
                return pcd.isoformat()
            except ValueError:
                pass

    return None


def _enrich_record_from_detail(
    record: dict, detail: dict, as_of_date: date,
) -> dict:
    """Enrich a TrialRecord dict using detail endpoint data.

    Overwrites: status, primary_completion_date, completion_date.
    Appends: NCT ID to secondary_ids.
    Updates: last_update_posted if publishDate is newer.
    Returns the record (modified in place).
    """
    # Status
    detail_status = _extract_detail_status(detail)
    if detail_status != "UNKNOWN":
        record["status"] = detail_status

    # NCT cross-reference
    nct_id = _extract_nct_id(detail)
    if nct_id and nct_id not in record.get("secondary_ids", []):
        record.setdefault("secondary_ids", []).append(nct_id)

    # Primary completion date
    pcd = _estimate_primary_completion_date(detail, as_of_date)
    if pcd:
        record["primary_completion_date"] = pcd
        if not record.get("completion_date"):
            record["completion_date"] = pcd

    # PIT anchor: update last_update_posted if publishDate is newer
    publish_date = _normalize_ctis_date(detail.get("publishDate") or "")
    if publish_date:
        current_lup = record.get("last_update_posted") or ""
        if not current_lup or publish_date > current_lup:
            record["last_update_posted"] = publish_date

    return record


def _normalize_ctis_record(raw: dict, ticker: str, fetched_at: str) -> dict:
    """Normalize a CTIS search result to the unified TrialRecord schema.

    Actual CTIS API fields (from live response):
        ctNumber, ctStatus (int), ctTitle, shortTitle, startDateEU, endDateEU,
        conditions (str), trialCountries (list of "Country:code"), sponsor (str),
        trialPhase (descriptive text), decisionDateOverall, lastUpdated,
        lastPublicationUpdate, therapeuticAreas (list), product, etc.
    """
    ct_number = raw.get("ctNumber", "")

    # Phase normalization — trialPhase uses descriptive text like
    # "Therapeutic exploratory (Phase II)" or "Therapeutic confirmatory (Phase III)"
    phase_raw = raw.get("trialPhase") or ""
    phase = "NA"
    for pattern, phase_val in _PHASE_PATTERNS:
        if pattern.search(phase_raw):
            phase = phase_val
            break

    # Status normalization — ctStatus is an integer code
    status_raw = raw.get("ctStatus")
    if isinstance(status_raw, int):
        status = _STATUS_CODE_MAP.get(status_raw, "UNKNOWN")
    else:
        status = _STATUS_TEXT_MAP.get(str(status_raw or "").lower().strip(), "UNKNOWN")

    # Date parsing: CTIS uses "dd/mm/yyyy"
    # decisionDateOverall is the single overall date; decisionDate has per-country breakdown
    decision_date = _normalize_ctis_date(raw.get("decisionDateOverall") or "")
    start_date = _normalize_ctis_date(raw.get("startDateEU") or "")
    end_date = _normalize_ctis_date(raw.get("endDateEU") or raw.get("endDate") or "")
    last_updated = _normalize_ctis_date(raw.get("lastUpdated") or "")
    last_pub_update = _normalize_ctis_date(raw.get("lastPublicationUpdate") or "")

    # Sponsor info — "sponsor" is a plain string in search results
    sponsor_name = raw.get("sponsor") or raw.get("sponsorName") or ""
    sponsor_country = raw.get("sponsorCountry") or ""

    # Secondary IDs
    secondary_ids = []
    eudract = raw.get("eudractNumber") or ""
    if eudract:
        secondary_ids.append(eudract)
    nct_id = raw.get("nctNumber") or raw.get("nctId") or ""
    if nct_id:
        secondary_ids.append(nct_id)
    protocol = raw.get("shortTitle") or ""
    if protocol:
        secondary_ids.append(f"protocol:{protocol}")

    # Countries — trialCountries is list of "Country:statusCode" pairs
    countries_raw = raw.get("trialCountries") or raw.get("memberStates") or []
    if isinstance(countries_raw, str):
        countries_raw = [countries_raw]
    countries = []
    for entry in countries_raw:
        if isinstance(entry, str):
            # "Germany:8" -> "Germany"
            country = entry.split(":")[0].strip()
            if country:
                countries.append(country)

    # Conditions — plain string in search results
    conditions = []
    cond_raw = raw.get("conditions") or ""
    if isinstance(cond_raw, str) and cond_raw:
        conditions = [c.strip() for c in cond_raw.split(";") if c.strip()]
    # Also capture therapeutic areas
    ta_raw = raw.get("therapeuticAreas") or []
    for ta in ta_raw:
        if isinstance(ta, str) and ta and ta not in conditions:
            conditions.append(ta)

    title = raw.get("ctTitle") or raw.get("title") or raw.get("trialTitle") or ""
    # Clean up embedded newlines in title
    title = " ".join(title.split())

    # PIT anchor: prefer lastUpdated > lastPublicationUpdate > decisionDate > startDate
    pit_anchor = last_updated or last_pub_update or decision_date or start_date

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
        "last_update_posted": pit_anchor,
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
