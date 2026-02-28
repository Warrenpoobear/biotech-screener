"""
press_release_collector.py - Press Release Future-Date Extractor

Extracts future-dated event announcements from company press releases
(via RSS/Atom feeds or pre-fetched HTML).

Produces PIT-friendly records with clear provenance.

Source: Company PR feeds (RSS/Atom XML + HTML detail pages)
Cache: cache/press/press_release_events_{as_of_date}.json
Schema: press_release_events.v1

Design:
  - Only emits events with DAY precision (YYYY-MM-DD) on or after as_of_date.
  - Requires "future intent" phrase near extracted date.
  - Stable IDs via sha1(ticker|event_date|title|url).
  - Deterministic output: stable sort by (event_date, ticker, source, id).
  - No live web search; requires pr_rss_url in universe or explicit config.
  - Cap of 3 events per press release to limit noise.
"""

import hashlib
import json
import logging
import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Schema and version constants
SCHEMA = "press_release_events.v1"
COLLECTOR_VERSION = "press_release_collector.v1"

# Rate limit between PR page fetches
_PR_FETCH_DELAY = 0.5

# Cap on events per single press release
_MAX_EVENTS_PER_PR = 3

# Cap on unmatched samples in telemetry
_UNMATCHED_SAMPLES_CAP = 100

# Suffixes to strip during company name normalization
_COMPANY_SUFFIXES = re.compile(
    r"\b(inc|corp|corporation|ltd|limited|plc|co|company|sa|nv|ag|se|therapeutics|"
    r"pharmaceutical|pharmaceuticals|biosciences|biopharma|biotech|biotechnology)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Date extraction patterns
# ---------------------------------------------------------------------------

# Named month + day + year: "March 15, 2026" or "March 15 2026"
_MONTH_DAY_YEAR = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b"
)
# Abbreviated month: "Mar 15, 2026"
_ABBR_MONTH_DAY_YEAR = re.compile(
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"\s+(\d{1,2}),?\s+(\d{4})\b"
)
# Month + day without year: "March 15" or "Mar 15"
_MONTH_DAY_NO_YEAR = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"\s+(\d{1,2})\b"
)

# Future intent phrases — must appear near a date to qualify
_FUTURE_INTENT_PHRASES = [
    "will present",
    "to present",
    "will host",
    "will participate",
    "to participate",
    "scheduled for",
    "scheduled to",
    "plans to present",
    "plans to host",
    "expects to present",
    "expects to report",
    "will report",
    "will announce",
    "to be held",
    "to be presented",
    "will be presented",
    "to report",
    "will be held",
]

# Historical context phrases — date associated with past event
_HISTORICAL_PHRASES = [
    "presented on",
    "presented at",
    "reported on",
    "announced on",
    "held on",
    "took place",
    "was held",
    "were presented",
]

# Event kind classification from title/text
_PR_EVENT_KIND_MAP = [
    ("data readout", "data_readout"),
    ("top-line data", "data_readout"),
    ("topline data", "data_readout"),
    ("top line data", "data_readout"),
    ("interim data", "data_readout"),
    ("pivotal data", "data_readout"),
    ("phase 3 data", "data_readout"),
    ("phase 2 data", "data_readout"),
    ("phase 1 data", "data_readout"),
    ("abstract accepted", "abstract_accepted"),
    ("abstract selected", "abstract_accepted"),
    ("oral presentation", "conference_presentation"),
    ("poster presentation", "conference_presentation"),
    ("conference presentation", "conference_presentation"),
    ("will present", "conference_presentation"),
    ("to present", "conference_presentation"),
    ("earnings", "earnings_date"),
    ("financial results", "earnings_date"),
    ("quarterly results", "earnings_date"),
    ("guidance", "guidance"),
]

# Proximity window (chars) for future-intent phrase to be near a date
_PHRASE_PROXIMITY_CHARS = 200


def normalize_company(name: str) -> str:
    """Normalize company name for matching.

    Lower-case, strip punctuation, collapse whitespace, drop common suffixes.
    """
    s = name.lower()
    s = re.sub(r"[^\w\s-]", " ", s)
    s = _COMPANY_SUFFIXES.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _build_universe_map(
    universe: List[Dict],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build name -> ticker mapping and ticker -> pr_rss_url mapping.

    Returns:
        (name_map, rss_map)
    """
    name_map: Dict[str, str] = {}
    rss_map: Dict[str, str] = {}

    for entry in universe:
        ticker = entry.get("ticker", "").strip().upper()
        if not ticker:
            continue

        names = set()
        if entry.get("name"):
            names.add(entry["name"])
        md = entry.get("market_data") or {}
        if md.get("company_name"):
            names.add(md["company_name"])

        for name in names:
            norm = normalize_company(name)
            if norm and norm not in name_map:
                name_map[norm] = ticker

        if entry.get("pr_rss_url"):
            rss_map[ticker] = entry["pr_rss_url"]

    return name_map, rss_map


def _stable_id(ticker: str, event_date: str, title: str, url: str) -> str:
    """Generate stable 10-char hex ID from event attributes."""
    raw = f"{ticker.lower()}|{event_date}|{title.lower()}|{url.lower()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def _classify_event_kind(text: str) -> str:
    """Classify press release event kind from text."""
    text_lower = text.lower()
    for keyword, kind in _PR_EVENT_KIND_MAP:
        if keyword in text_lower:
            return kind
    return "other"


def _parse_month_day_year(month_str: str, day_str: str, year_str: str) -> Optional[date]:
    """Parse month/day/year strings into a date."""
    for fmt_m in ("%B", "%b"):
        try:
            dt = datetime.strptime(f"{month_str} {day_str} {year_str}", f"{fmt_m} %d %Y")
            return dt.date()
        except ValueError:
            continue
    return None


def _parse_month_day_infer_year(
    month_str: str, day_str: str, disclosed_at: date
) -> Optional[date]:
    """Parse month/day without year, inferring year from disclosed_at.

    If the resulting date would be in the past relative to disclosed_at,
    roll forward one year.
    """
    for fmt_m in ("%B", "%b"):
        try:
            dt = datetime.strptime(f"{month_str} {day_str}", f"{fmt_m} %d")
            candidate = date(disclosed_at.year, dt.month, dt.day)
            if candidate < disclosed_at:
                candidate = date(disclosed_at.year + 1, dt.month, dt.day)
            return candidate
        except ValueError:
            continue
    return None


def _has_future_intent_near(text: str, date_pos: int) -> Tuple[bool, List[str]]:
    """Check if a future-intent phrase appears within proximity of date position.

    Returns (is_future, matched_phrases).
    """
    text_lower = text.lower()
    window_start = max(0, date_pos - _PHRASE_PROXIMITY_CHARS)
    window_end = min(len(text_lower), date_pos + _PHRASE_PROXIMITY_CHARS)
    window = text_lower[window_start:window_end]

    matched = []
    for phrase in _FUTURE_INTENT_PHRASES:
        if phrase in window:
            matched.append(phrase)

    return bool(matched), matched


def _is_historical_context(text: str, date_pos: int) -> bool:
    """Check if date appears in historical context."""
    text_lower = text.lower()
    window_start = max(0, date_pos - _PHRASE_PROXIMITY_CHARS)
    window_end = min(len(text_lower), date_pos + _PHRASE_PROXIMITY_CHARS)
    window = text_lower[window_start:window_end]

    for phrase in _HISTORICAL_PHRASES:
        if phrase in window:
            return True
    return False


def extract_future_dates_from_text(
    text: str,
    disclosed_at: date,
    as_of_date: date,
) -> List[Dict[str, Any]]:
    """Extract future-dated events from press release text.

    Returns list of dicts with keys: event_date, matched_phrases, date_text, position.
    Only returns dates that:
    - Have DAY precision
    - Are >= as_of_date
    - Have a future-intent phrase nearby
    - Are NOT in historical context
    """
    results: List[Dict[str, Any]] = []
    seen_dates: set = set()

    # Full month + day + year
    for m in _MONTH_DAY_YEAR.finditer(text):
        d = _parse_month_day_year(m.group(1), m.group(2), m.group(3))
        if d is None or d < as_of_date:
            continue
        if d.isoformat() in seen_dates:
            continue

        if _is_historical_context(text, m.start()):
            continue

        is_future, phrases = _has_future_intent_near(text, m.start())
        if not is_future:
            continue

        seen_dates.add(d.isoformat())
        results.append({
            "event_date": d.isoformat(),
            "matched_phrases": phrases,
            "date_text": m.group(0),
            "position": m.start(),
        })

    # Abbreviated month + day + year
    for m in _ABBR_MONTH_DAY_YEAR.finditer(text):
        d = _parse_month_day_year(m.group(1), m.group(2), m.group(3))
        if d is None or d < as_of_date:
            continue
        if d.isoformat() in seen_dates:
            continue

        if _is_historical_context(text, m.start()):
            continue

        is_future, phrases = _has_future_intent_near(text, m.start())
        if not is_future:
            continue

        seen_dates.add(d.isoformat())
        results.append({
            "event_date": d.isoformat(),
            "matched_phrases": phrases,
            "date_text": m.group(0),
            "position": m.start(),
        })

    # Month + day without year (infer year)
    for m in _MONTH_DAY_NO_YEAR.finditer(text):
        # Skip if this match is part of a longer "Month DD, YYYY" already captured
        after_match = text[m.end():m.end() + 10].strip()
        if re.match(r",?\s*\d{4}", after_match):
            continue

        d = _parse_month_day_infer_year(m.group(1), m.group(2), disclosed_at)
        if d is None or d < as_of_date:
            continue
        if d.isoformat() in seen_dates:
            continue

        if _is_historical_context(text, m.start()):
            continue

        is_future, phrases = _has_future_intent_near(text, m.start())
        if not is_future:
            continue

        seen_dates.add(d.isoformat())
        results.append({
            "event_date": d.isoformat(),
            "matched_phrases": phrases,
            "date_text": m.group(0),
            "position": m.start(),
        })

    return results


def _parse_rss_items(xml_text: str) -> List[Dict[str, str]]:
    """Parse RSS/Atom feed XML into list of items with title, link, pubDate."""
    items: List[Dict[str, str]] = []

    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError as e:
        logger.warning("RSS XML parse error: %s", e)
        return []

    # RSS 2.0
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        items.append({"title": title, "link": link, "pubDate": pub_date})

    # Atom
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    for entry in root.iter("{http://www.w3.org/2005/Atom}entry"):
        title = (entry.findtext("atom:title", namespaces=ns) or
                 entry.findtext("{http://www.w3.org/2005/Atom}title") or "").strip()
        link_el = entry.find("{http://www.w3.org/2005/Atom}link")
        link = (link_el.get("href", "") if link_el is not None else "").strip()
        pub_date = (entry.findtext("{http://www.w3.org/2005/Atom}published") or
                    entry.findtext("{http://www.w3.org/2005/Atom}updated") or "").strip()
        items.append({"title": title, "link": link, "pubDate": pub_date})

    return items


def _parse_pub_date(pub_date_str: str) -> Optional[date]:
    """Parse RSS pubDate or Atom published date to date object."""
    if not pub_date_str:
        return None

    # Try ISO format first
    try:
        return date.fromisoformat(pub_date_str[:10])
    except ValueError:
        pass

    # RFC 2822 format: "Mon, 01 Mar 2026 12:00:00 GMT"
    for fmt in (
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S %z",
        "%d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
    ):
        try:
            return datetime.strptime(pub_date_str.strip(), fmt).date()
        except ValueError:
            continue

    return None


def collect_press_release_events(
    *,
    universe: List[Dict],
    as_of_date: date,
    cache_dir: Path,
    cache_only: bool = False,
) -> List[Dict[str, Any]]:
    """Collect future-dated events from press release feeds.

    Args:
        universe: List of universe dicts (must have 'ticker'; optionally 'pr_rss_url').
        as_of_date: PIT date boundary.
        cache_dir: Directory for cache output.
        cache_only: If True, load from cache only (no network).

    Returns:
        List of press_release_event.v1 records.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"press_release_events_{as_of_date.isoformat()}.json"

    # Check cache
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            events = cached.get("events", [])
            logger.info("Press release events: loaded %d from cache", len(events))
            return events
        except Exception as e:
            logger.warning("Press release cache read error: %s", e)

    if cache_only:
        logger.debug("Press releases: cache_only mode, no cache for %s", as_of_date)
        return []

    _, rss_map = _build_universe_map(universe)

    if not rss_map:
        logger.info("Press releases: no pr_rss_url entries in universe — returning []")
        return []

    all_events: List[Dict[str, Any]] = []
    stats = {
        "fetched_pages": 0,
        "parsed_items": 0,
        "matched": 0,
        "unmatched": 0,
        "unmatched_samples": [],
    }

    if requests is None:
        logger.warning("requests library not available")
        return []

    for ticker, rss_url in sorted(rss_map.items()):
        try:
            time.sleep(_PR_FETCH_DELAY)
            resp = requests.get(rss_url, timeout=30)
            if resp.status_code != 200:
                logger.debug("PR RSS fetch %s returned %d", ticker, resp.status_code)
                continue
            stats["fetched_pages"] += 1
        except Exception as e:
            logger.warning("PR RSS fetch error for %s: %s", ticker, e)
            continue

        rss_items = _parse_rss_items(resp.text)
        stats["parsed_items"] += len(rss_items)

        for item in rss_items:
            disclosed_at = _parse_pub_date(item.get("pubDate", ""))
            if disclosed_at is None:
                disclosed_at = as_of_date

            # PIT: skip items published after as_of_date
            if disclosed_at > as_of_date:
                continue

            title = item.get("title", "")
            link = item.get("link", "")
            text = title  # Use title as primary text; HTML fetch is optional

            # Extract future dates from title text
            future_dates = extract_future_dates_from_text(
                text, disclosed_at, as_of_date
            )

            # Cap events per PR
            for fd in future_dates[:_MAX_EVENTS_PER_PR]:
                record_id = f"PR_{ticker}_{_stable_id(ticker, fd['event_date'], title, link)}"
                event_kind = _classify_event_kind(title)

                all_events.append({
                    "schema": "press_release_event.v1",
                    "id": record_id,
                    "ticker": ticker,
                    "jurisdiction": "US",
                    "event_date": fd["event_date"],
                    "disclosed_at": disclosed_at.isoformat(),
                    "event_kind": event_kind,
                    "title": title[:200],
                    "url": link,
                    "source": "PRESS_RELEASE",
                    "confidence": "MED",
                    "raw": {
                        "company": ticker,
                        "matched_phrases": fd["matched_phrases"],
                        "date_text": fd["date_text"],
                        "source_url": link,
                    },
                })
                stats["matched"] += 1

    # Stable sort
    all_events.sort(key=lambda e: (
        e.get("event_date", ""),
        e.get("ticker", ""),
        e.get("source", ""),
        e.get("id", ""),
    ))

    # Write cache wrapper
    wrapper = {
        "schema": SCHEMA,
        "as_of_date": as_of_date.isoformat(),
        "collector_version": COLLECTOR_VERSION,
        "events": all_events,
        "stats": stats,
    }
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(wrapper, f, sort_keys=True, indent=2)
    except Exception as e:
        logger.warning("Press release cache write error: %s", e)

    logger.info("Press release events: %d events collected", len(all_events))
    return all_events
