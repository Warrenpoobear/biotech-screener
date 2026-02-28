"""
ir_events_collector.py - Investor Relations Events Page Collector

Extracts future-dated events from company IR/Events pages.
Produces PIT-friendly records with clear provenance.

Source: Company IR websites (HTML scraping)
Cache: cache/ir/ir_events_{as_of_date}.json
Schema: ir_events.v1

Design:
  - Only emits events with DAY precision (YYYY-MM-DD) on or after as_of_date.
  - Stable IDs via sha1(ticker|event_date|title|url).
  - Deterministic output: stable sort by (event_date, ticker, source, id).
  - No live web search; requires ir_url in universe or explicit config.
"""

import hashlib
import json
import logging
import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Schema and version constants
SCHEMA = "ir_events.v1"
COLLECTOR_VERSION = "ir_events_collector.v1"

# Rate limit between IR page fetches
_IR_FETCH_DELAY = 1.0

# Cap on unmatched samples in telemetry
_UNMATCHED_SAMPLES_CAP = 100

# Suffixes to strip during company name normalization
_COMPANY_SUFFIXES = re.compile(
    r"\b(inc|corp|corporation|ltd|limited|plc|co|company|sa|nv|ag|se|therapeutics|"
    r"pharmaceutical|pharmaceuticals|biosciences|biopharma|biotech|biotechnology)\b",
    re.IGNORECASE,
)

# Patterns for extracting dates from HTML event pages
_DATE_PATTERNS = [
    # ISO format: 2026-03-15
    re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),
    # Month DD, YYYY: March 15, 2026
    re.compile(r"\b(January|February|March|April|May|June|July|August|September|"
               r"October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b"),
    # Mon DD, YYYY: Mar 15, 2026
    re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
               r"\s+(\d{1,2}),?\s+(\d{4})\b"),
    # MM/DD/YYYY
    re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"),
]

# Event kind keywords
_EVENT_KIND_MAP = [
    ("earnings", "earnings"),
    ("quarterly results", "earnings"),
    ("financial results", "earnings"),
    ("investor day", "investor_day"),
    ("investor meeting", "investor_day"),
    ("r&d day", "investor_day"),
    ("analyst day", "investor_day"),
    ("webcast", "webcast"),
    ("web cast", "webcast"),
    ("conference", "conference"),
    ("symposium", "conference"),
    ("summit", "conference"),
    ("congress", "conference"),
    ("presentation", "conference"),
]


def normalize_company(name: str) -> str:
    """Normalize company name for matching.

    Lower-case, strip punctuation, collapse whitespace, drop common suffixes.
    """
    s = name.lower()
    # Strip punctuation except hyphens
    s = re.sub(r"[^\w\s-]", " ", s)
    # Remove common suffixes
    s = _COMPANY_SUFFIXES.sub("", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _build_universe_map(universe: List[Dict]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build normalized company name -> ticker mappings from universe.

    Returns:
        (exact_map, ir_url_map) where exact_map is {normalized_name: ticker}
        and ir_url_map is {ticker: ir_url}.
    """
    exact_map: Dict[str, str] = {}
    ir_url_map: Dict[str, str] = {}

    for entry in universe:
        ticker = entry.get("ticker", "").strip().upper()
        if not ticker:
            continue

        # Collect all name variants
        names = set()
        if entry.get("name"):
            names.add(entry["name"])
        md = entry.get("market_data") or {}
        if md.get("company_name"):
            names.add(md["company_name"])

        for name in names:
            norm = normalize_company(name)
            if norm and norm not in exact_map:
                exact_map[norm] = ticker

        # IR URL if present
        if entry.get("ir_url"):
            ir_url_map[ticker] = entry["ir_url"]

    return exact_map, ir_url_map


def _stable_id(ticker: str, event_date: str, title: str, url: str) -> str:
    """Generate stable 10-char hex ID from event attributes."""
    raw = f"{ticker.lower()}|{event_date}|{title.lower()}|{url.lower()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def _classify_event_kind(title: str) -> str:
    """Classify event kind from title text using keyword heuristics."""
    title_lower = title.lower()
    for keyword, kind in _EVENT_KIND_MAP:
        if keyword in title_lower:
            return kind
    return "other"


def _parse_date_from_text(text: str) -> Optional[date]:
    """Try to parse a YYYY-MM-DD date from text.

    Only returns dates with DAY precision. Returns None for
    ambiguous or unparseable dates.
    """
    # ISO format first
    m = _DATE_PATTERNS[0].search(text)
    if m:
        try:
            return date.fromisoformat(m.group(0))
        except ValueError:
            pass

    # Month DD, YYYY
    m = _DATE_PATTERNS[1].search(text)
    if m:
        try:
            return datetime.strptime(
                f"{m.group(1)} {m.group(2)} {m.group(3)}", "%B %d %Y"
            ).date()
        except ValueError:
            pass

    # Mon DD, YYYY
    m = _DATE_PATTERNS[2].search(text)
    if m:
        try:
            return datetime.strptime(
                f"{m.group(1)} {m.group(2)} {m.group(3)}", "%b %d %Y"
            ).date()
        except ValueError:
            pass

    # MM/DD/YYYY
    m = _DATE_PATTERNS[3].search(text)
    if m:
        try:
            return date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
        except ValueError:
            pass

    return None


def _extract_events_from_html(
    html: str,
    ticker: str,
    source_url: str,
    as_of_date: date,
) -> List[Dict[str, Any]]:
    """Extract future-dated events from IR page HTML.

    Looks for:
    - <time datetime="..."> elements
    - Table rows with date columns
    - List items with date prefixes

    Only emits records where event_date >= as_of_date.
    """
    events: List[Dict[str, Any]] = []
    seen_ids: set = set()

    # Strategy 1: <time datetime="YYYY-MM-DD"> or data-date attributes
    time_pattern = re.compile(
        r'(?:datetime|data-date)\s*=\s*["\'](\d{4}-\d{2}-\d{2})["\']',
        re.IGNORECASE,
    )
    # Find surrounding text for title extraction
    # We look for the time tag and then grab nearby text
    block_pattern = re.compile(
        r'(?:<[^>]*(?:datetime|data-date)\s*=\s*["\'](\d{4}-\d{2}-\d{2})["\'][^>]*>)'
        r'(.*?)(?=<(?:time|tr|li|div\s+class)|$)',
        re.IGNORECASE | re.DOTALL,
    )

    for m in block_pattern.finditer(html):
        date_str = m.group(1)
        context = m.group(2) if m.group(2) else ""
        # Strip HTML tags from context
        title = re.sub(r"<[^>]+>", " ", context)
        title = re.sub(r"\s+", " ", title).strip()[:200]

        try:
            event_date = date.fromisoformat(date_str)
        except ValueError:
            continue

        if event_date < as_of_date:
            continue

        if not title:
            title = f"IR event on {date_str}"

        record_id = f"IR_{ticker}_{_stable_id(ticker, date_str, title, source_url)}"
        if record_id in seen_ids:
            continue
        seen_ids.add(record_id)

        events.append({
            "schema": "ir_event.v1",
            "id": record_id,
            "ticker": ticker,
            "jurisdiction": "US",
            "event_date": date_str,
            "disclosed_at": as_of_date.isoformat(),
            "event_kind": _classify_event_kind(title),
            "title": title,
            "url": source_url,
            "source": "IR_EVENTS",
            "confidence": "MED",
            "raw": {
                "company": ticker,
                "source_url": source_url,
                "date_text": date_str,
            },
        })

    # Strategy 2: Scan for date patterns in text blocks
    # Strip HTML to plain text for broad date scanning
    plain = re.sub(r"<[^>]+>", "\n", html)
    plain = re.sub(r"&nbsp;|&#\d+;|&\w+;", " ", plain)

    for line in plain.split("\n"):
        line = line.strip()
        if not line or len(line) < 10:
            continue

        event_date = _parse_date_from_text(line)
        if event_date is None or event_date < as_of_date:
            continue

        date_str = event_date.isoformat()
        title = re.sub(r"\s+", " ", line).strip()[:200]

        record_id = f"IR_{ticker}_{_stable_id(ticker, date_str, title, source_url)}"
        if record_id in seen_ids:
            continue
        seen_ids.add(record_id)

        events.append({
            "schema": "ir_event.v1",
            "id": record_id,
            "ticker": ticker,
            "jurisdiction": "US",
            "event_date": date_str,
            "disclosed_at": as_of_date.isoformat(),
            "event_kind": _classify_event_kind(title),
            "title": title,
            "url": source_url,
            "source": "IR_EVENTS",
            "confidence": "MED",
            "raw": {
                "company": ticker,
                "source_url": source_url,
                "date_text": date_str,
            },
        })

    return events


def collect_ir_events(
    *,
    universe: List[Dict],
    as_of_date: date,
    cache_dir: Path,
    cache_only: bool = False,
) -> List[Dict[str, Any]]:
    """Collect IR events from company events pages.

    Args:
        universe: List of universe dicts (must have 'ticker'; optionally 'ir_url').
        as_of_date: PIT date boundary.
        cache_dir: Directory for cache output.
        cache_only: If True, load from cache only (no network).

    Returns:
        List of ir_event.v1 records.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"ir_events_{as_of_date.isoformat()}.json"

    # Check cache
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            events = cached.get("events", [])
            logger.info("IR events: loaded %d from cache", len(events))
            return events
        except Exception as e:
            logger.warning("IR events cache read error: %s", e)

    if cache_only:
        logger.debug("IR events: cache_only mode, no cache for %s", as_of_date)
        return []

    _, ir_url_map = _build_universe_map(universe)

    if not ir_url_map:
        logger.info("IR events: no ir_url entries in universe — returning []")
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

    for ticker, ir_url in sorted(ir_url_map.items()):
        try:
            time.sleep(_IR_FETCH_DELAY)
            resp = requests.get(ir_url, timeout=30)
            if resp.status_code != 200:
                logger.debug("IR fetch %s returned %d", ticker, resp.status_code)
                continue
            stats["fetched_pages"] += 1
        except Exception as e:
            logger.warning("IR fetch error for %s: %s", ticker, e)
            continue

        try:
            events = _extract_events_from_html(
                resp.text, ticker, ir_url, as_of_date
            )
            stats["parsed_items"] += len(events)
            stats["matched"] += len(events)
            all_events.extend(events)
        except Exception as e:
            logger.warning("IR parse error for %s: %s", ticker, e)
            continue

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
        logger.warning("IR events cache write error: %s", e)

    logger.info("IR events: %d events collected", len(all_events))
    return all_events
