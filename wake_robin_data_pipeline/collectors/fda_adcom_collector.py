"""
fda_adcom_collector.py - FDA Advisory Committee Calendar Collector

Extracts FDA Advisory Committee (ADCOM) meeting dates from two sources:

  1. Federal Register API (primary, free, structured JSON)
     - FDA is legally required to publish ADCOM notices ≥15 days before meetings
     - Returns meeting date, committee name, drug/product, NDA/BLA numbers
     - Endpoint: https://www.federalregister.gov/api/v1/documents.json

  2. SEC EDGAR 8-K filings mentioning "advisory committee" (supplementary)
     - Catches ADCOM announcements from company press releases
     - Lower yield: filings rarely state the meeting date explicitly

Rate limiting: Federal Register has no documented rate limit; we add 0.2s courtesy delay
Cache: results cached per as_of_date to avoid repeated fetches
"""

import json
import logging
import os
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Federal Register API
FEDERAL_REGISTER_API = "https://www.federalregister.gov/api/v1/documents.json"

# SEC EDGAR search for ADCOM-related 8-K filings (supplementary)
SEC_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"

# User-Agent for requests
USER_AGENT = os.environ.get(
    "FDA_USER_AGENT",
    "Wake Robin Research contact@wakerobincapital.com"
)

# Default cache directory — matches Module3Config's default path (relative to project root)
DEFAULT_CACHE_DIR = Path("cache") / "fda"

# Regex to extract ticker from EDGAR display_names (handles comma-separated warrants)
_TICKER_FROM_DISPLAY = re.compile(r"\(([A-Z]{2,5})(?:\)|[,\s])")

# Extract meeting date from Federal Register `dates` field
# e.g. "The meeting will be held on July 17, 2025, from 8 a.m. ..."
# Also handles: "held virtually on ...", "held on virtually on ...", "take place on ..."
_FR_MEETING_DATE = re.compile(
    r"(?:held|scheduled|take\s+place)\s+(?:(?:on\s+)?virtually\s+(?:on\s+)?|on\s+)?(\w+\s+\d{1,2},?\s+\d{4})"
)

# Extract drug/product names from Federal Register title
# Pattern 1: "...NDA/BLA 761440 for Belantamab Mafodotin" or "...for REXULTI (brexpiprazole) Tablets"
_FR_DRUG_AFTER_NDA = re.compile(
    r"(?:NDA|BLA|sNDA|sBLA)\s*\)?\s*[\d/S-]+,?\s+for\s+([A-Z][A-Za-z]+(?:\s+[A-Za-z]+)*)"
)
# Pattern 2: "BRANDNAME (generic)" e.g. "COLUMVI (glofitamab)"
_FR_BRAND_GENERIC = re.compile(
    r"([A-Z]{3,})\s+\(([a-z][a-z\s]+)\)"
)
# Pattern 3: "Comments-Midomafetamine Capsules" (drug name right after Comments-)
_FR_DRUG_AFTER_COMMENTS = re.compile(
    r"Comments-([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Capsules?|Tablets?|Injection|Solution))?)"
)

# ADCOM date patterns in SEC 8-K filings
_ADCOM_DATE_PATTERNS = [
    r"[Aa]dvisory\s+[Cc]ommittee\s+(?:meeting\s+)?(?:scheduled\s+for|on|date[:\s]+|is\s+)\s*(\w+\s+\d{1,2},?\s+\d{4})",
    r"ADCOM\s+(?:meeting\s+)?(?:scheduled\s+for|on|date[:\s]+)\s*(\w+\s+\d{1,2},?\s+\d{4})",
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


def _collect_adcom_from_federal_register(
    product_map: Dict[str, str],
    as_of_date: date,
    lookback_days: int = 365,
) -> List[Dict[str, Any]]:
    """
    Query the Federal Register API for FDA drug advisory committee meeting notices.

    FDA is legally required to publish meeting notices at least 15 days before
    the meeting date. The Federal Register API is free, requires no API key,
    and returns structured JSON with meeting dates, committee names, and
    drug/product details.

    Args:
        product_map: {drug_name_lower: ticker} for matching
        as_of_date: Current analysis date
        lookback_days: How far back to search for notices

    Returns:
        List of ADCOM event dicts
    """
    try:
        import requests
    except ImportError:
        logger.warning("requests library not available")
        return []

    start_date = (as_of_date - timedelta(days=lookback_days)).isoformat()
    end_date = as_of_date.isoformat()

    events = []
    page = 1

    while True:
        params = {
            "conditions[agencies][]": "food-and-drug-administration",
            "conditions[term]": '"advisory committee" "notice of meeting"',
            "conditions[publication_date][gte]": start_date,
            "conditions[publication_date][lte]": end_date,
            "fields[]": ["title", "dates", "abstract", "publication_date", "html_url"],
            "per_page": 100,
            "page": page,
        }

        try:
            time.sleep(0.2)
            resp = requests.get(FEDERAL_REGISTER_API, params=params, timeout=30)
            if resp.status_code != 200:
                logger.warning(f"Federal Register API returned {resp.status_code}")
                break

            data = resp.json()
            results = data.get("results", [])
            total = data.get("total_pages", 1)

            if page == 1:
                count = data.get("count", 0)
                logger.info(f"Federal Register: {count} ADCOM notices found")

            if not results:
                break

        except Exception as e:
            logger.warning(f"Federal Register API error: {e}")
            break

        for doc in results:
            title = doc.get("title", "")
            dates_str = doc.get("dates", "")
            pub_date = doc.get("publication_date", "")

            # Skip non-drug committees (tobacco, devices, veterinary, food)
            title_lower = title.lower()
            skip_keywords = ["tobacco", "device", "veterinary", "food safety", "blood products"]
            if any(kw in title_lower for kw in skip_keywords):
                continue

            # Extract meeting date from `dates` field
            meeting_date = None
            date_match = _FR_MEETING_DATE.search(dates_str)
            if date_match:
                meeting_date = _parse_date_str(date_match.group(1))

            if not meeting_date:
                continue

            # PIT safety: skip meetings announced after as_of_date
            if pub_date > end_date:
                continue

            # Extract committee name (before first semicolon)
            committee = title.split(";")[0].strip() if ";" in title else ""

            # Match drug name against product map using full title (substring)
            match = _match_product_to_ticker(title, product_map)

            # Extract drug names from title via multiple regex patterns
            extracted_drugs = []
            for m in _FR_DRUG_AFTER_NDA.finditer(title):
                extracted_drugs.append(m.group(1).strip())
            for m in _FR_BRAND_GENERIC.finditer(title):
                extracted_drugs.append(m.group(1).strip())  # brand
                extracted_drugs.append(m.group(2).strip())  # generic
            for m in _FR_DRUG_AFTER_COMMENTS.finditer(title):
                extracted_drugs.append(m.group(1).strip())

            drug_name = ""
            ticker = ""
            if match:
                drug_name = match["drug_name"]
                ticker = match["ticker"]
            else:
                # Try each extracted drug name against product map
                for candidate in extracted_drugs:
                    match2 = _match_product_to_ticker(candidate, product_map)
                    if match2:
                        ticker = match2["ticker"]
                        drug_name = match2["drug_name"]
                        break

                if not ticker:
                    # Log extracted drugs for debugging
                    if extracted_drugs:
                        logger.debug(
                            f"Federal Register ADCOM: unmatched drugs {extracted_drugs} "
                            f"in '{title[:80]}'"
                        )
                    continue

            events.append({
                "ticker": ticker,
                "event_type": "FDA_ADCOM",
                "event_date": meeting_date.isoformat(),
                "event_name": f"ADCOM: {drug_name} ({committee})",
                "drug_name": drug_name,
                "confidence": "HIGH",
                "source": "FEDERAL_REGISTER",
                "disclosed_at": pub_date,
                "committee": committee,
                "tags": ["adcom", "federal_register"],
            })

        if page >= total:
            break
        page += 1

    return events


def _collect_adcom_from_edgar(
    ticker_set: set,
    cik_to_ticker: Dict[str, str],
    as_of_date: date,
    lookback_days: int = 365,
) -> List[Dict[str, Any]]:
    """
    Search SEC EDGAR for 8-K filings mentioning advisory committee meetings
    and extract ADCOM dates for tickers in our universe.

    Supplementary source — most filings mention ADCOM without stating dates.

    Args:
        ticker_set: Set of uppercase ticker symbols
        cik_to_ticker: CIK-to-ticker reverse map for fallback matching
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

    # EDGAR uses implicit AND (space-separated); explicit AND keyword doesn't work
    queries = [
        '"advisory committee" "FDA"',
        '"ADCOM" OR "FDA advisory committee"',
    ]

    # Collect unique filings keyed by accession number
    filings_to_fetch: Dict[str, Dict[str, str]] = {}

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
            total = results.get("hits", {}).get("total", {}).get("value", 0)
            hits = results.get("hits", {}).get("hits", [])
            logger.info(f"EDGAR ADCOM search '{query[:40]}': {total} total, {len(hits)} returned")

            for hit in hits:
                source = hit.get("_source", {})
                display_names = source.get("display_names", [])
                file_date = source.get("file_date", "")
                adsh = source.get("adsh", "")
                ciks = source.get("ciks", [])

                if not adsh or not file_date or file_date > end_date:
                    continue

                # Extract ticker: try display_names first, then CIK fallback
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
                    for cik_val in ciks:
                        cik_stripped = str(cik_val).lstrip("0")
                        if cik_stripped in cik_to_ticker:
                            ticker = cik_to_ticker[cik_stripped]
                            break

                if not ticker:
                    continue

                cik = ciks[0] if ciks else ""
                if adsh not in filings_to_fetch:
                    filings_to_fetch[adsh] = {
                        "ticker": ticker,
                        "cik": str(cik).lstrip("0"),
                        "file_date": file_date,
                    }

        except Exception as e:
            logger.warning(f"EDGAR ADCOM search error: {e}")

    logger.info(f"EDGAR ADCOM: {len(filings_to_fetch)} universe filings to fetch")

    # Phase 2: Fetch filing text and extract ADCOM dates
    for adsh, meta in filings_to_fetch.items():
        ticker = meta["ticker"]
        cik_stripped = meta["cik"]
        file_date = meta["file_date"]
        adsh_nodash = adsh.replace("-", "")

        base_url = f"{SEC_ARCHIVES_BASE}/{cik_stripped}/{adsh_nodash}"

        try:
            time.sleep(0.1)
            idx_resp = session.get(f"{base_url}/index.json", timeout=30)
            if idx_resp.status_code != 200:
                continue
            items = idx_resp.json().get("directory", {}).get("item", [])
        except Exception:
            continue

        exhibit_files = []
        main_files = []
        for item in items:
            name = item.get("name", "")
            name_lower = name.lower()
            if not name_lower.endswith(".htm"):
                continue
            if "ex99" in name_lower or "ex-99" in name_lower or "pr" in name_lower:
                exhibit_files.append(name)
            elif "index" not in name_lower and "R1" not in name:
                main_files.append(name)

        docs_to_fetch = exhibit_files[:3] + main_files[:1]

        for filename in docs_to_fetch:
            doc_url = f"{base_url}/{filename}"
            try:
                time.sleep(0.1)
                doc_resp = session.get(doc_url, timeout=30)
                if doc_resp.status_code != 200:
                    continue
            except Exception:
                continue

            text = re.sub(r"<[^>]+>", " ", doc_resp.text)
            text = re.sub(r"&nbsp;|&#\d+;|&\w+;", " ", text)
            text = re.sub(r"\s+", " ", text).strip()

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

    # Dedup by (ticker, event_date)
    seen: Set[tuple] = set()
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
    universe_tickers: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """
    Collect FDA Advisory Committee calendar events and match to universe tickers.

    Strategy:
      1. Federal Register API (primary) — structured, free, legally mandated
      2. SEC EDGAR 8-K filings (supplementary) — catches company-announced ADCOM
      3. Merge and deduplicate

    Args:
        drug_to_ticker: Mapping of lowercase drug names to ticker symbols
        as_of_date: Current analysis date
        cache_dir: Directory for caching fetched data
        lookback_days: How far back to look for events
        universe_tickers: Full set of universe tickers for EDGAR matching
                          (defaults to drug_to_ticker values if not provided)

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
        import requests  # noqa: F401 — verify requests is available
    except ImportError:
        logger.warning("requests library not available; skipping FDA ADCOM collection")
        return []

    # ── Source 1: Federal Register API ──
    fr_events = _collect_adcom_from_federal_register(drug_to_ticker, as_of_date, lookback_days)
    logger.info(f"Federal Register: {len(fr_events)} matched ADCOM events")

    events = list(fr_events)
    fr_count = len(events)

    # ── Source 2: EDGAR 8-K filings mentioning ADCOM (supplementary) ──
    ticker_set = universe_tickers if universe_tickers else set(drug_to_ticker.values())

    # Build CIK-to-ticker reverse map for fallback matching
    cik_to_ticker: Dict[str, str] = {}
    try:
        import requests as _req
        _session = _req.Session()
        _session.headers.update({"User-Agent": USER_AGENT})
        ct_resp = _session.get("https://www.sec.gov/files/company_tickers.json", timeout=30)
        if ct_resp.status_code == 200:
            for _k, entry in ct_resp.json().items():
                t = entry.get("ticker", "").upper()
                c = str(entry.get("cik_str", "")).lstrip("0")
                if t in ticker_set and c:
                    cik_to_ticker[c] = t
            logger.info(f"Built CIK-to-ticker map: {len(cik_to_ticker)} entries")
    except Exception as e:
        logger.warning(f"Could not build CIK map: {e}")

    edgar_events = _collect_adcom_from_edgar(ticker_set, cik_to_ticker, as_of_date, lookback_days)
    logger.info(f"EDGAR ADCOM: {len(edgar_events)} events from 8-K filings")

    # Merge, dedup by (ticker, event_date) keeping Federal Register over EDGAR
    seen: Set[tuple] = set()
    for event in events:
        seen.add((event["ticker"], event["event_date"]))

    for event in edgar_events:
        key = (event["ticker"], event["event_date"])
        if key not in seen:
            seen.add(key)
            events.append(event)

    logger.info(
        f"FDA ADCOM total: {len(events)} events "
        f"({fr_count} from Federal Register, {len(events) - fr_count} from EDGAR)"
    )

    # Cache results
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(events, f, indent=2)
    except Exception as e:
        logger.warning(f"Cache write error: {e}")

    return events
