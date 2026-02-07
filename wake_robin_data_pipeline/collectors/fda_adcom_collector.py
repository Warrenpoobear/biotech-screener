"""
fda_adcom_collector.py - FDA Advisory Committee Calendar Collector

Extracts FDA Advisory Committee (ADCOM) meeting dates from two sources:

  1. FDA.gov calendar page (HTML scraping)
     - The page uses JavaScript DataTables; static HTML has empty <tbody>
     - Works when the FDA page renders data server-side (intermittent)
     - Fallback: if no meetings parsed, tries alternative sources

  2. SEC EDGAR 8-K filings mentioning "advisory committee" for universe tickers
     - Searches EDGAR for 8-K filings containing ADCOM-related keywords
     - Extracts meeting dates from press release text
     - More reliable than scraping since it uses structured SEC API

Source: https://www.fda.gov/advisory-committees/advisory-committee-calendar
        https://efts.sec.gov/LATEST/search-index (fallback)
Rate limiting: 1 req/sec for FDA, 10 req/sec for SEC EDGAR
Cache: results cached per as_of_date to avoid repeated fetches
"""

import json
import logging
import os
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# FDA Advisory Committee Calendar URL
FDA_ADCOM_CALENDAR_URL = (
    "https://www.fda.gov/advisory-committees/advisory-committee-calendar"
)

# SEC EDGAR search for ADCOM-related 8-K filings
SEC_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"

# User-Agent for requests
USER_AGENT = os.environ.get(
    "FDA_USER_AGENT",
    "Wake Robin Research contact@wakerobincapital.com"
)

# Default cache directory
DEFAULT_CACHE_DIR = Path(__file__).parent.parent / "cache" / "fda"

# Regex to extract ticker from EDGAR display_names
_TICKER_FROM_DISPLAY = re.compile(r"\(([A-Z]{2,5})\)")

# ADCOM date patterns in SEC filings
_ADCOM_DATE_PATTERNS = [
    # "Advisory Committee meeting scheduled for March 15, 2026"
    r"[Aa]dvisory\s+[Cc]ommittee\s+(?:meeting\s+)?(?:scheduled\s+for|on|date[:\s]+|is\s+)\s*(\w+\s+\d{1,2},?\s+\d{4})",
    # "ADCOM meeting on March 15, 2026"
    r"ADCOM\s+(?:meeting\s+)?(?:scheduled\s+for|on|date[:\s]+)\s*(\w+\s+\d{1,2},?\s+\d{4})",
    # "the FDA has scheduled an advisory committee meeting for March 15, 2026"
    r"FDA\s+(?:has\s+)?scheduled\s+an?\s+[Aa]dvisory\s+[Cc]ommittee\s+(?:meeting\s+)?(?:for\s+)?(\w+\s+\d{1,2},?\s+\d{4})",
]


def build_product_ticker_map(data_dir: Path) -> Dict[str, str]:
    """
    Build a product-name-to-ticker mapping from PDUFA dates and FDA designations.

    Reads pdufa_dates.json and fda_designations.json to build a lookup
    of {drug_name_lower: ticker}.

    Args:
        data_dir: Directory containing data files (e.g. production_data/)

    Returns:
        Dict mapping lowercase drug names to ticker symbols
    """
    product_map: Dict[str, str] = {}

    # Read PDUFA dates
    pdufa_path = data_dir / "pdufa_dates.json"
    if pdufa_path.exists():
        try:
            with open(pdufa_path, "r", encoding="utf-8") as f:
                pdufa_data = json.load(f)
            # pdufa_dates.json is a list of dicts
            entries = pdufa_data if isinstance(pdufa_data, list) else pdufa_data.get("events", [])
            for entry in entries:
                drug = entry.get("drug_name", "").strip()
                ticker = entry.get("ticker", "").strip()
                if drug and ticker:
                    product_map[drug.lower()] = ticker
        except Exception as e:
            logger.warning(f"Error reading PDUFA dates: {e}")

    # Read FDA designations
    desig_path = data_dir / "fda_designations.json"
    if desig_path.exists():
        try:
            with open(desig_path, "r", encoding="utf-8") as f:
                desig_data = json.load(f)
            designations = desig_data.get("designations", [])
            for entry in designations:
                drug = entry.get("drug_name", "").strip()
                ticker = entry.get("ticker", "").strip()
                if drug and ticker:
                    product_map[drug.lower()] = ticker
        except Exception as e:
            logger.warning(f"Error reading FDA designations: {e}")

    logger.info(f"Built product-to-ticker map with {len(product_map)} entries")
    return product_map


def _parse_adcom_html(html_content: str) -> List[Dict[str, Any]]:
    """
    Parse FDA advisory committee calendar HTML to extract meeting entries.

    Note: The FDA page uses JavaScript DataTables for rendering. Static HTML
    may have empty <tbody>. This parser handles both cases:
    - Server-rendered rows (within <tr> tags)
    - Embedded JSON data (if available)

    Returns:
        List of raw meeting dicts with keys: date_str, committee, topic
    """
    meetings = []

    # Pattern to match meeting date entries in the FDA calendar page
    date_patterns = [
        r"(\w+ \d{1,2},? \d{4})",      # "March 15, 2026"
        r"(\d{1,2}/\d{1,2}/\d{4})",    # "03/15/2026"
    ]

    # Look for table rows or structured content blocks with ADCOM info
    row_pattern = re.compile(
        r'(?:<tr[^>]*>|<div[^>]*class="[^"]*views-row[^"]*"[^>]*>)'
        r'(.*?)'
        r'(?:</tr>|</div>)',
        re.DOTALL | re.IGNORECASE,
    )

    rows = row_pattern.findall(html_content)
    if not rows:
        # Fallback: try splitting on common separators
        rows = re.split(r'<(?:tr|article|div\s+class="[^"]*row)[^>]*>', html_content)

    for row in rows:
        # Extract date
        found_date = None
        for dp in date_patterns:
            m = re.search(dp, row)
            if m:
                found_date = m.group(1)
                break

        if not found_date:
            continue

        # Parse to actual date
        parsed_date = None
        for fmt in ("%B %d, %Y", "%B %d %Y", "%m/%d/%Y"):
            try:
                parsed_date = datetime.strptime(found_date, fmt).date()
                break
            except ValueError:
                continue

        if not parsed_date:
            continue

        # Extract text content (strip HTML tags)
        text = re.sub(r'<[^>]+>', ' ', row)
        text = re.sub(r'\s+', ' ', text).strip()

        # Try to identify committee and topic
        committee = ""
        topic = text

        # Common committee name patterns
        committee_match = re.search(
            r'((?:Oncologic|Cardiovascular|Psychopharmacologic|Anti[- ]?Infective|'
            r'Endocrinologic|Dermatologic|Gastrointestinal|Pulmonary|'
            r'Peripheral and Central Nervous System|Arthritis|'
            r'Drug Safety|Pediatric|Bone|Reproductive|'
            r'Medical Imaging|Anesthetic|Cellular)[^,;]*(?:Drugs?|Advisory)?\s*'
            r'(?:Advisory\s+)?(?:Committee)?)',
            text,
            re.IGNORECASE,
        )
        if committee_match:
            committee = committee_match.group(1).strip()

        meetings.append({
            "date": parsed_date.isoformat(),
            "committee": committee,
            "topic": topic,
        })

    return meetings


def _match_product_to_ticker(
    topic: str,
    product_map: Dict[str, str],
) -> Optional[Dict[str, str]]:
    """
    Try to match a meeting topic string against known product names.

    Returns:
        Dict with 'ticker' and 'drug_name' if matched, else None
    """
    topic_lower = topic.lower()

    for drug_name, ticker in product_map.items():
        # Check if drug name appears in the topic text
        if drug_name in topic_lower:
            return {"ticker": ticker, "drug_name": drug_name}

        # Also check without hyphens (e.g., "karxt" vs "kar-xt")
        drug_no_hyphen = drug_name.replace("-", "")
        if drug_no_hyphen in topic_lower.replace("-", ""):
            return {"ticker": ticker, "drug_name": drug_name}

    return None


def _parse_date_str(date_str: str) -> Optional[date]:
    """Parse a date string in common formats to a date object."""
    for fmt in ("%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _collect_adcom_from_edgar(
    ticker_set: set,
    as_of_date: date,
    lookback_days: int = 365,
) -> List[Dict[str, Any]]:
    """
    Search SEC EDGAR for 8-K filings mentioning advisory committee meetings
    and extract ADCOM dates for tickers in our universe.

    Args:
        ticker_set: Set of uppercase ticker symbols
        as_of_date: Current analysis date
        lookback_days: How far back to search

    Returns:
        List of ADCOM event dicts
    """
    try:
        import requests
    except ImportError:
        logger.warning("requests library not available")
        return []

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    start_date = (as_of_date - timedelta(days=lookback_days)).isoformat()
    end_date = as_of_date.isoformat()

    events = []

    # Search EDGAR for 8-K filings mentioning advisory committee
    queries = [
        '"advisory committee" AND ("FDA" OR "ADCOM")',
    ]

    for query in queries:
        try:
            params = {
                "q": query,
                "dateRange": "custom",
                "startdt": start_date,
                "enddt": end_date,
                "forms": "8-K",
            }

            time.sleep(0.1)
            response = session.get(SEC_SEARCH_URL, params=params, timeout=30)

            if response.status_code != 200:
                logger.warning(f"EDGAR ADCOM search returned {response.status_code}")
                continue

            results = response.json()
            hits = results.get("hits", {}).get("hits", [])
            logger.info(f"EDGAR ADCOM search: {len(hits)} hits")

            for hit in hits:
                source = hit.get("_source", {})
                display_names = source.get("display_names", [])
                file_date = source.get("file_date", "")
                adsh = source.get("adsh", "")
                ciks = source.get("ciks", [])

                if not adsh or not file_date or file_date > end_date:
                    continue

                # Extract ticker from display_names
                ticker = None
                for name in display_names:
                    for m in _TICKER_FROM_DISPLAY.finditer(name):
                        candidate = m.group(1)
                        if candidate != "CIK" and candidate.upper() in ticker_set:
                            ticker = candidate.upper()
                            break
                    if ticker:
                        break

                if not ticker:
                    continue

                # Fetch the actual filing text to extract ADCOM dates
                cik = ciks[0] if ciks else ""
                adsh_nodash = adsh.replace("-", "")
                cik_stripped = cik.lstrip("0")

                time.sleep(0.1)
                index_url = f"{SEC_ARCHIVES_BASE}/{cik_stripped}/{adsh_nodash}/"
                try:
                    idx_resp = session.get(index_url, timeout=30)
                    if idx_resp.status_code != 200:
                        continue
                except Exception:
                    continue

                # Find exhibit documents
                doc_links = re.findall(
                    rf'href="(/Archives/edgar/data/{cik_stripped}/{adsh_nodash}/[^"]+\.htm)"',
                    idx_resp.text,
                )
                exhibit_links = [
                    link for link in doc_links
                    if any(kw in link.split("/")[-1].lower() for kw in ("ex99", "ex-99", "pr"))
                ][:3]

                # Fetch and search exhibit text
                for doc_link in exhibit_links:
                    time.sleep(0.1)
                    try:
                        doc_resp = session.get(f"https://www.sec.gov{doc_link}", timeout=30)
                        if doc_resp.status_code != 200:
                            continue
                    except Exception:
                        continue

                    text = re.sub(r"<[^>]+>", " ", doc_resp.text)
                    text = re.sub(r"&nbsp;|&#\d+;|&\w+;", " ", text)
                    text = re.sub(r"\s+", " ", text).strip()

                    # Search for ADCOM date patterns
                    for pattern in _ADCOM_DATE_PATTERNS:
                        for m in re.finditer(pattern, text):
                            adcom_date = _parse_date_str(m.group(1))
                            if adcom_date and adcom_date >= as_of_date - timedelta(days=30):
                                events.append({
                                    "ticker": ticker,
                                    "event_type": "FDA_ADCOM",
                                    "event_date": adcom_date.isoformat(),
                                    "event_name": f"ADCOM: {m.group(0)[:60]}",
                                    "drug_name": "",
                                    "confidence": "HIGH",
                                    "source": "SEC_8K_ADCOM",
                                    "disclosed_at": file_date,
                                    "committee": "",
                                    "tags": ["adcom", "sec_8k"],
                                })

        except Exception as e:
            logger.warning(f"EDGAR ADCOM search error: {e}")

    # Dedup by (ticker, event_date)
    seen = set()
    deduped = []
    for event in events:
        key = (event["ticker"], event["event_date"])
        if key not in seen:
            seen.add(key)
            deduped.append(event)

    return deduped


def collect_fda_adcom_events(
    drug_to_ticker: Dict[str, str],
    as_of_date: date,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    lookback_days: int = 365,
) -> List[Dict[str, Any]]:
    """
    Collect FDA Advisory Committee calendar events and match to universe tickers.

    Strategy:
      1. Try scraping the FDA ADCOM calendar page (may return 0 if JS-rendered)
      2. Always also search SEC EDGAR for 8-K filings mentioning ADCOM meetings
      3. Merge and deduplicate

    Args:
        drug_to_ticker: Mapping of lowercase drug names to ticker symbols
        as_of_date: Current analysis date
        cache_dir: Directory for caching fetched data
        lookback_days: How far back to look for events

    Returns:
        List of event dicts matching corporate catalyst schema
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"adcom_calendar_{as_of_date.isoformat()}.json"

    # Check cache
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            logger.info(f"Loaded {len(cached)} FDA ADCOM events from cache")
            return cached
        except Exception as e:
            logger.warning(f"Cache read error: {e}")

    try:
        import requests
    except ImportError:
        logger.warning("requests library not available; skipping FDA ADCOM collection")
        return []

    events = []

    # ── Source 1: FDA calendar page ──
    try:
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(FDA_ADCOM_CALENDAR_URL, headers=headers, timeout=30)
        response.raise_for_status()

        meetings = _parse_adcom_html(response.text)
        logger.info(f"Parsed {len(meetings)} meetings from FDA ADCOM calendar page")

        cutoff_date = as_of_date - timedelta(days=lookback_days)

        for meeting in meetings:
            meeting_date = date.fromisoformat(meeting["date"])

            if meeting_date < cutoff_date:
                continue

            match = _match_product_to_ticker(meeting["topic"], drug_to_ticker)
            if not match:
                continue

            events.append({
                "ticker": match["ticker"],
                "event_type": "FDA_ADCOM",
                "event_date": meeting["date"],
                "event_name": f"ADCOM: {match['drug_name']}",
                "drug_name": match["drug_name"],
                "confidence": "HIGH",
                "source": "FDA_ADCOM_CALENDAR",
                "disclosed_at": as_of_date.isoformat(),
                "committee": meeting.get("committee", ""),
                "tags": ["adcom"],
            })

    except Exception as e:
        logger.warning(f"FDA ADCOM calendar scrape: {e}")

    fda_count = len(events)

    # ── Source 2: EDGAR 8-K filings mentioning ADCOM ──
    ticker_set = set(drug_to_ticker.values())
    edgar_events = _collect_adcom_from_edgar(ticker_set, as_of_date, lookback_days)
    logger.info(f"EDGAR ADCOM search: {len(edgar_events)} events from 8-K filings")

    # Merge, dedup by (ticker, event_date) keeping FDA source over EDGAR
    seen = set()
    for event in events:
        seen.add((event["ticker"], event["event_date"]))

    for event in edgar_events:
        key = (event["ticker"], event["event_date"])
        if key not in seen:
            seen.add(key)
            events.append(event)

    logger.info(
        f"FDA ADCOM total: {len(events)} events "
        f"({fda_count} from FDA page, {len(events) - fda_count} from EDGAR)"
    )

    # Cache results
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(events, f, indent=2)
    except Exception as e:
        logger.warning(f"Cache write error: {e}")

    return events
