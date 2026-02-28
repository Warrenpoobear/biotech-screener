"""
ema_committee_collector.py - Collect EMA committee agendas + meeting highlights.

Two AdCom-equivalent catalyst streams:
  1) Committee events (agenda/annex-derived, forward-looking, MED confidence)
  2) Meeting outcomes (meeting highlights-derived, backward-looking, HIGH confidence)

Endpoints:
  - CHMP landing: /en/committees/committee-medicinal-products-human-use-chmp
  - PRAC landing: /en/committees/pharmacovigilance-risk-assessment-committee-prac
  - Meeting highlights pages: /en/news/meeting-highlights-...

Cache: cache/ema/ema_committee_events_{date}.json
       cache/ema/ema_meeting_outcomes_{date}.json

Rate limit: 1.0s between requests. Timeout: 30s.
"""

import hashlib
import json
import logging
import os
import re
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RATE_LIMIT_SECS = 1.0
_REQUEST_TIMEOUT = 30
_COLLECTOR_VERSION = "ema_committee_collector.v1"
_MAX_UNMATCHED_LOG = 100

_BASE_URL = "https://www.ema.europa.eu"

_COMMITTEE_URLS = {
    "CHMP": f"{_BASE_URL}/en/committees/committee-medicinal-products-human-use-chmp",
    "PRAC": f"{_BASE_URL}/en/committees/pharmacovigilance-risk-assessment-committee-prac",
}

# Meeting date range regex: "8 - 11 December 2025" or "8-11 December 2025"
# Also handles "10 November 2025" (single day) and "27 - 30 January 2026"
_DATE_RANGE_RE = re.compile(
    r"(\d{1,2})\s*[-–]\s*(\d{1,2})\s+"
    r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{4})",
    re.IGNORECASE,
)
_SINGLE_DATE_RE = re.compile(
    r"(\d{1,2})\s+"
    r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{4})",
    re.IGNORECASE,
)

# Outcome section headers → outcome_code mapping
_OUTCOME_SECTION_MAP = {
    "recommended for approval": "positive_opinion",
    "positive opinion": "positive_opinion",
    "recommended for marketing authorisation": "positive_opinion",
    "new indication": "positive_opinion",
    "extension of indication": "positive_opinion",
    "negative opinion": "negative_opinion",
    "refused": "refusal",
    "refusal": "refusal",
    "withdrawn": "withdrawn",
    "withdrawal": "withdrawn",
    "referral": "referral_started",
    "referrals started": "referral_started",
    "referrals concluded": "referral_concluded",
    "safety signal": "safety_signal",
    "signal": "safety_signal",
}

# Procedure type keywords for agenda items
_PROCEDURE_TYPE_MAP = {
    "marketing authorisation application": "MAA",
    "maa": "MAA",
    "initial authorisation": "MAA",
    "variation": "VARIATION",
    "type ii variation": "VARIATION",
    "referral": "REFERRAL",
    "periodic safety update report": "PSUR",
    "psur": "PSUR",
    "signal": "SIGNAL",
    "safety signal": "SIGNAL",
}


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------


def _fetch(url: str) -> str:
    """Fetch URL content with rate limiting. Returns HTML text."""
    time.sleep(_RATE_LIMIT_SECS)
    resp = requests.get(
        url,
        timeout=_REQUEST_TIMEOUT,
        headers={
            "Accept": "text/html,application/xhtml+xml",
            "User-Agent": "Wake Robin Research (biotech screener)",
        },
    )
    resp.raise_for_status()
    return resp.text


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_meeting_dates(text: str) -> Optional[Tuple[str, str]]:
    """Extract meeting date range from text like '8 - 11 December 2025'.

    Returns (start_date, end_date) as ISO strings, or None.
    """
    m = _DATE_RANGE_RE.search(text)
    if m:
        day_start, day_end, month_str, year = m.groups()
        try:
            end_dt = datetime.strptime(f"{day_end} {month_str} {year}", "%d %B %Y")
            start_dt = datetime.strptime(f"{day_start} {month_str} {year}", "%d %B %Y")
            return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    m = _SINGLE_DATE_RE.search(text)
    if m:
        day, month_str, year = m.groups()
        try:
            dt = datetime.strptime(f"{day} {month_str} {year}", "%d %B %Y")
            iso = dt.strftime("%Y-%m-%d")
            return iso, iso
        except ValueError:
            pass

    return None


def _normalize_name(s: str) -> str:
    """Normalize a medicine/substance name for matching."""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _match_medicine_to_ticker(
    medicine_name: str,
    active_substance: Optional[str],
    product_ticker_map: Dict[str, str],
) -> Optional[str]:
    """Match medicine name or active substance to a ticker.

    Matching order:
      1) exact match on normalized medicine_name
      2) exact match on normalized active_substance
      3) substring match on normalized medicine_name
      4) substring match with hyphens removed
    """
    norm_med = _normalize_name(medicine_name) if medicine_name else ""
    norm_sub = _normalize_name(active_substance) if active_substance else ""

    # Exact matches first
    if norm_med and norm_med in product_ticker_map:
        return product_ticker_map[norm_med]
    if norm_sub and norm_sub in product_ticker_map:
        return product_ticker_map[norm_sub]

    # Substring matches
    for drug_name, ticker in product_ticker_map.items():
        if norm_med and drug_name in norm_med:
            return ticker
        drug_no_hyphen = drug_name.replace("-", "")
        if norm_med and drug_no_hyphen in norm_med.replace("-", ""):
            return ticker
    for drug_name, ticker in product_ticker_map.items():
        if norm_sub and drug_name in norm_sub:
            return ticker

    return None


def _make_event_id(prefix: str, *parts: str) -> str:
    """Generate stable 10-char hex hash from key parts."""
    key = "|".join(str(p).lower() for p in parts)
    return prefix + "_" + hashlib.sha1(key.encode()).hexdigest()[:10]


def _classify_procedure_type(text: str) -> str:
    """Classify procedure type from agenda text."""
    lower = text.lower()
    for keyword, ptype in _PROCEDURE_TYPE_MAP.items():
        if keyword in lower:
            return ptype
    return "OTHER"


def _classify_outcome_code(section_heading: str) -> str:
    """Map a section heading to an outcome code."""
    lower = section_heading.lower().strip()
    for keyword, code in _OUTCOME_SECTION_MAP.items():
        if keyword in lower:
            return code
    return "other"


# ---------------------------------------------------------------------------
# Discovery: scrape committee landing page for links
# ---------------------------------------------------------------------------


def _discover_links(
    html: str, base_url: str,
) -> Dict[str, List[Dict[str, str]]]:
    """Discover agenda docs and meeting highlights links from a committee page.

    Returns dict with keys:
      - agenda_docs: list of {url, title}
      - highlights: list of {url, title}
    """
    soup = BeautifulSoup(html, "html.parser")

    agenda_docs: List[Dict[str, str]] = []
    highlights: List[Dict[str, str]] = []

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        text = a_tag.get_text(strip=True)
        if not text:
            continue

        # Resolve relative URLs
        if href.startswith("/"):
            href = base_url.rstrip("/") + href

        text_lower = text.lower()

        # Agenda / Annex documents
        if "agenda" in text_lower and ("meeting" in text_lower or "annex" in text_lower):
            agenda_docs.append({"url": href, "title": text})

        # Meeting highlights
        if "/en/news/meeting-highlights" in href.lower() or (
            "meeting highlights" in text_lower
        ):
            highlights.append({"url": href, "title": text})

    return {"agenda_docs": agenda_docs, "highlights": highlights}


# ---------------------------------------------------------------------------
# Parsing: agenda documents (HTML link-based, minimal)
# ---------------------------------------------------------------------------


def _parse_agenda_page(
    html: str,
    url: str,
    committee: str,
    meeting_start: str,
    meeting_end: str,
    product_ticker_map: Dict[str, str],
) -> Tuple[List[dict], List[str]]:
    """Parse an agenda/annex page for medicine items.

    Returns (events, unmatched) where unmatched is list of medicine names
    that couldn't be mapped to tickers.
    """
    soup = BeautifulSoup(html, "html.parser")
    events = []
    unmatched = []

    # Extract text content and look for medicine-like items in lists
    # Agenda pages typically have structured lists or tables
    items = []

    # Try list items first
    for li in soup.find_all("li"):
        text = li.get_text(strip=True)
        if len(text) > 10 and len(text) < 500:
            items.append(text)

    # Also try table rows
    for tr in soup.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        if cells:
            items.append(" | ".join(cells))

    # Also try paragraphs with capitalized drug-like names
    for p in soup.find_all("p"):
        text = p.get_text(strip=True)
        if len(text) > 10 and len(text) < 500:
            items.append(text)

    seen_ids = set()
    for item_text in items:
        # Try to extract a medicine name — look for capitalized words
        # that match the product map
        procedure_type = _classify_procedure_type(item_text)

        # Try matching the whole text against the product map
        ticker = _match_medicine_to_ticker(item_text, None, product_ticker_map)
        medicine_name = item_text[:100]  # truncate for display

        if ticker is None:
            # Try extracting just capitalized words as potential drug names
            words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", item_text)
            for word in words:
                ticker = _match_medicine_to_ticker(word, None, product_ticker_map)
                if ticker:
                    medicine_name = word
                    break

        if ticker is None:
            if len(unmatched) < _MAX_UNMATCHED_LOG:
                unmatched.append(item_text[:200])
            continue

        event_id = _make_event_id(
            f"EMA_{committee}_AGENDA_{meeting_end}",
            medicine_name, procedure_type, committee, meeting_start, meeting_end,
        )
        if event_id in seen_ids:
            continue
        seen_ids.add(event_id)

        events.append({
            "schema": "ema_committee_event.v1",
            "id": event_id,
            "ticker": ticker,
            "jurisdiction": "EU",
            "committee": committee,
            "meeting_start": meeting_start,
            "meeting_end": meeting_end,
            "event_date": meeting_end,
            "disclosed_at": None,
            "item_type": "agenda_item",
            "procedure_type": procedure_type,
            "medicine_name": medicine_name,
            "active_substance": None,
            "sponsor": None,
            "title": f"{committee} agenda: {procedure_type} — {medicine_name}",
            "source": f"EMA_{committee}_AGENDA",
            "url": url,
            "confidence": "MED",
            "raw": {"snippet": item_text[:300]},
        })

    return events, unmatched


# ---------------------------------------------------------------------------
# Parsing: meeting highlights (outcomes)
# ---------------------------------------------------------------------------


def _parse_meeting_highlights(
    html: str,
    url: str,
    committee: str,
    product_ticker_map: Dict[str, str],
) -> Tuple[List[dict], List[str]]:
    """Parse a meeting highlights page for outcome records.

    Returns (outcomes, unmatched).
    """
    soup = BeautifulSoup(html, "html.parser")
    outcomes = []
    unmatched = []

    # Extract meeting dates from the page title or first heading
    title_tag = soup.find("title")
    h1_tag = soup.find("h1")
    title_text = ""
    if h1_tag:
        title_text = h1_tag.get_text(strip=True)
    elif title_tag:
        title_text = title_tag.get_text(strip=True)

    dates = _parse_meeting_dates(title_text)
    if dates is None:
        # Try from URL
        dates = _parse_meeting_dates(url)
    meeting_start = dates[0] if dates else ""
    meeting_end = dates[1] if dates else ""

    # Extract "First published" date as disclosed_at
    disclosed_at = None
    for meta in soup.find_all("meta"):
        if meta.get("name", "").lower() in ("dcterms.issued", "dc.date", "article:published_time"):
            raw = meta.get("content", "")
            if raw:
                disclosed_at = raw[:10]
    if not disclosed_at:
        # Look for text like "First published: DD/MM/YYYY"
        for text_node in soup.find_all(string=re.compile(r"first published", re.IGNORECASE)):
            parent_text = text_node.parent.get_text(strip=True) if text_node.parent else str(text_node)
            date_match = re.search(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})", parent_text)
            if date_match:
                d, mo, y = date_match.groups()
                try:
                    disclosed_at = datetime.strptime(f"{d}/{mo}/{y}", "%d/%m/%Y").strftime("%Y-%m-%d")
                except ValueError:
                    pass
    if not disclosed_at:
        disclosed_at = meeting_end or None

    # Walk headings to find outcome sections
    current_outcome_code = None
    seen_ids = set()

    for element in soup.find_all(["h2", "h3", "h4", "li", "p"]):
        tag_name = element.name
        text = element.get_text(strip=True)

        if tag_name in ("h2", "h3", "h4"):
            current_outcome_code = _classify_outcome_code(text)
            continue

        if current_outcome_code is None or not text or len(text) < 3:
            continue

        # Try to match text as a medicine name
        ticker = _match_medicine_to_ticker(text, None, product_ticker_map)
        medicine_name = text

        if ticker is None:
            # Try extracting capitalized words
            words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
            for word in words:
                ticker = _match_medicine_to_ticker(word, None, product_ticker_map)
                if ticker:
                    medicine_name = word
                    break

        if ticker is None:
            if len(unmatched) < _MAX_UNMATCHED_LOG:
                unmatched.append(text[:200])
            continue

        event_id = _make_event_id(
            f"EMA_{committee}_OUTCOME_{meeting_end}_{current_outcome_code}",
            medicine_name, committee, meeting_start, meeting_end,
        )
        if event_id in seen_ids:
            continue
        seen_ids.add(event_id)

        outcomes.append({
            "schema": "ema_meeting_outcome.v1",
            "id": event_id,
            "ticker": ticker,
            "jurisdiction": "EU",
            "committee": committee,
            "meeting_start": meeting_start,
            "meeting_end": meeting_end,
            "event_date": meeting_end,
            "disclosed_at": disclosed_at,
            "outcome_code": current_outcome_code,
            "medicine_name": medicine_name,
            "active_substance": None,
            "indication": None,
            "title": f"{committee} meeting highlights: {current_outcome_code} — {medicine_name}",
            "source": "EMA_MEETING_HIGHLIGHTS",
            "url": url,
            "confidence": "HIGH",
            "raw": {"section": current_outcome_code, "snippet": text[:300]},
        })

    return outcomes, unmatched


# ---------------------------------------------------------------------------
# Public entrypoints
# ---------------------------------------------------------------------------


def collect_ema_committee_events(
    *,
    as_of_date: date,
    cache_dir: Path,
    product_ticker_map: Dict[str, str],
    committees: Tuple[str, ...] = ("CHMP",),
    lookback_days: int = 365,
) -> List[dict]:
    """Collect EMA committee agenda events.

    Scrapes committee landing pages for agenda/annex links, parses them,
    and maps medicines to tickers.

    Args:
        as_of_date: PIT date for cache naming.
        cache_dir: Root directory for EMA cache files.
        product_ticker_map: Dict mapping lowercase drug names to ticker symbols.
        committees: Tuple of committee codes to scrape.
        lookback_days: Only include meetings within this window.

    Returns:
        List of event dicts (ema_committee_event.v1 schema).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / f"ema_committee_events_{as_of_date.isoformat()}.json"

    # Short-circuit if cache exists
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text())
            events = payload.get("events", payload) if isinstance(payload, dict) else payload
            logger.info("EMA committee events cache hit: %s (%d events)", cache_path.name, len(events))
            return events
        except (json.JSONDecodeError, OSError):
            pass

    cutoff = (as_of_date - timedelta(days=lookback_days)).isoformat()
    all_events: List[dict] = []
    all_unmatched: List[str] = []
    stats = {"fetched_pages": 0, "parsed_docs": 0, "matched_tickers": 0, "unmatched_items": 0}

    for committee in committees:
        landing_url = _COMMITTEE_URLS.get(committee)
        if not landing_url:
            logger.warning("Unknown EMA committee: %s", committee)
            continue

        try:
            landing_html = _fetch(landing_url)
            stats["fetched_pages"] += 1
        except Exception as e:
            logger.warning("EMA %s landing page fetch failed: %s", committee, e)
            continue

        links = _discover_links(landing_html, _BASE_URL)

        for doc in links["agenda_docs"]:
            dates = _parse_meeting_dates(doc["title"])
            if not dates:
                continue
            meeting_start, meeting_end = dates

            # Lookback + PIT filter
            if meeting_end < cutoff or meeting_end > as_of_date.isoformat():
                continue

            try:
                doc_html = _fetch(doc["url"])
                stats["fetched_pages"] += 1
                stats["parsed_docs"] += 1
            except Exception as e:
                logger.warning("EMA agenda fetch failed for %s: %s", doc["url"], e)
                continue

            events, unmatched = _parse_agenda_page(
                doc_html, doc["url"], committee,
                meeting_start, meeting_end, product_ticker_map,
            )
            # Set disclosed_at based on as_of_date for agenda items
            for ev in events:
                if ev["disclosed_at"] is None:
                    ev["disclosed_at"] = as_of_date.isoformat()

            all_events.extend(events)
            all_unmatched.extend(unmatched)

    stats["matched_tickers"] = len(set(ev["ticker"] for ev in all_events))
    stats["unmatched_items"] = len(all_unmatched)

    # Deterministic sort
    all_events.sort(key=lambda e: (e["event_date"], e["committee"], e["ticker"], e["id"]))

    # Write cache
    payload = {
        "schema": "ema_committee_events.v1",
        "as_of_date": as_of_date.isoformat(),
        "collector_version": _COLLECTOR_VERSION,
        "events": all_events,
        "stats": stats,
    }
    _write_cache(cache_path, payload)

    logger.info("EMA committee events: %d events cached -> %s", len(all_events), cache_path)
    return all_events


def collect_ema_meeting_outcomes(
    *,
    as_of_date: date,
    cache_dir: Path,
    product_ticker_map: Dict[str, str],
    committees: Tuple[str, ...] = ("CHMP", "PRAC"),
    lookback_days: int = 365,
) -> List[dict]:
    """Collect EMA meeting outcome events from meeting highlights pages.

    Scrapes committee landing pages for meeting highlights links, parses them,
    and maps medicines to tickers.

    Args:
        as_of_date: PIT date for cache naming.
        cache_dir: Root directory for EMA cache files.
        product_ticker_map: Dict mapping lowercase drug names to ticker symbols.
        committees: Tuple of committee codes to scrape.
        lookback_days: Only include meetings within this window.

    Returns:
        List of outcome dicts (ema_meeting_outcome.v1 schema).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / f"ema_meeting_outcomes_{as_of_date.isoformat()}.json"

    # Short-circuit if cache exists
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text())
            events = payload.get("events", payload) if isinstance(payload, dict) else payload
            logger.info("EMA meeting outcomes cache hit: %s (%d events)", cache_path.name, len(events))
            return events
        except (json.JSONDecodeError, OSError):
            pass

    cutoff = (as_of_date - timedelta(days=lookback_days)).isoformat()
    all_outcomes: List[dict] = []
    all_unmatched: List[str] = []
    stats = {"fetched_pages": 0, "parsed_docs": 0, "matched_tickers": 0, "unmatched_items": 0}

    for committee in committees:
        landing_url = _COMMITTEE_URLS.get(committee)
        if not landing_url:
            logger.warning("Unknown EMA committee: %s", committee)
            continue

        try:
            landing_html = _fetch(landing_url)
            stats["fetched_pages"] += 1
        except Exception as e:
            logger.warning("EMA %s landing page fetch failed: %s", committee, e)
            continue

        links = _discover_links(landing_html, _BASE_URL)

        for hl in links["highlights"]:
            dates = _parse_meeting_dates(hl["title"])
            if not dates:
                continue
            meeting_start, meeting_end = dates

            # Lookback + PIT filter
            if meeting_end < cutoff:
                continue
            # Outcomes are backward-looking: meeting_end <= as_of_date
            if meeting_end > as_of_date.isoformat():
                continue

            try:
                hl_html = _fetch(hl["url"])
                stats["fetched_pages"] += 1
                stats["parsed_docs"] += 1
            except Exception as e:
                logger.warning("EMA highlights fetch failed for %s: %s", hl["url"], e)
                continue

            outcomes, unmatched = _parse_meeting_highlights(
                hl_html, hl["url"], committee, product_ticker_map,
            )
            all_outcomes.extend(outcomes)
            all_unmatched.extend(unmatched)

    stats["matched_tickers"] = len(set(ev["ticker"] for ev in all_outcomes))
    stats["unmatched_items"] = len(all_unmatched)

    # Deterministic sort
    all_outcomes.sort(key=lambda e: (e["event_date"], e["committee"], e["ticker"], e["id"]))

    # Write cache
    payload = {
        "schema": "ema_meeting_outcomes.v1",
        "as_of_date": as_of_date.isoformat(),
        "collector_version": _COLLECTOR_VERSION,
        "events": all_outcomes,
        "stats": stats,
    }
    _write_cache(cache_path, payload)

    logger.info("EMA meeting outcomes: %d events cached -> %s", len(all_outcomes), cache_path)
    return all_outcomes


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------


def _write_cache(cache_path: Path, payload: dict) -> None:
    """Atomic JSON write with sort_keys for determinism."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(cache_path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp_path, str(cache_path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
