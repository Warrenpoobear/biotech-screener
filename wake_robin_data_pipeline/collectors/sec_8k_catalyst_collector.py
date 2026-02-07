"""
sec_8k_catalyst_collector.py - SEC 8-K Timing Phrase Extractor

Searches SEC EDGAR for 8-K filings and extracts timing phrases related to
clinical data readouts and FDA milestones.

Two-phase approach:
  1. Search EDGAR full-text search-index for 8-K filings matching timing keywords
     → returns metadata (accession number, CIK, display_names, file_date) but NO text
  2. For filings matching our universe tickers, fetch the actual filing documents
     (main 8-K + press release exhibits) from SEC Archives to extract timing phrases

Source: SEC EDGAR (https://efts.sec.gov/LATEST/search-index + Archives)
Rate limiting: 10 req/sec per SEC EDGAR fair access guidelines
Cache: results cached per as_of_date
"""

import json
import logging
import os
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# SEC EDGAR endpoints
SEC_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"

# SEC requires User-Agent header with contact info
USER_AGENT = os.environ.get(
    "SEC_USER_AGENT",
    "Wake Robin Research contact@wakerobincapital.com"
)

# Default cache directory
DEFAULT_CACHE_DIR = Path(__file__).parent.parent / "cache" / "sec" / "8k_catalysts"

# Rate limit: 10 req/sec for SEC EDGAR
_SEC_MIN_INTERVAL = 0.1

# Hard cap on filings to fetch per run (safety against runaway costs)
_MAX_FILINGS_PER_RUN = 200

# SEC official ticker-to-CIK mapping
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# Timing phrase patterns: (regex, event_type, date_precision)
TIMING_PATTERNS = [
    # "expects topline data in Q2 2026" / "anticipates top-line results in Q1 2026"
    (
        r"(?:expects?|anticipates?)\s+(?:topline|top-line|interim)\s+(?:data|results?)\s+.*?(Q[1-4]\s*\d{4})",
        "DATA_READOUT",
        "QUARTER",
    ),
    # "topline results in Q2 2026" (without expects/anticipates prefix)
    (
        r"(?:topline|top-line)\s+(?:data|results?)\s+(?:in|by|during)\s+.*?(Q[1-4]\s*\d{4})",
        "DATA_READOUT",
        "QUARTER",
    ),
    # "PDUFA date of March 15, 2026" or "PDUFA action date set for March 15, 2026"
    (
        r"PDUFA\s+(?:date|action date)\s+(?:of|is|set for)\s+(\w+\s+\d{1,2},?\s+\d{4})",
        "FDA_PDUFA_DATE",
        "DAY",
    ),
    # "expects to report topline results ... second half of 2026"
    (
        r"(?:expects?|anticipates?)\s+.*?(?:results?|data)\s+.*?(?:first|second)\s+half\s+(?:of\s+)?(\d{4})",
        "DATA_READOUT",
        "HALF_YEAR",
    ),
    # "data expected in mid-2026" / "results expected by year-end 2026"
    (
        r"(?:data|results?)\s+expected\s+.*?(?:mid|year[- ]?end)[- ]?(\d{4})",
        "DATA_READOUT",
        "HALF_YEAR",
    ),
]

# Regex to extract ticker from EDGAR display_names like "AGIOS PHARMACEUTICALS, INC.  (AGIO)  (CIK 0001439222)"
_TICKER_FROM_DISPLAY = re.compile(r"\(([A-Z]{2,5})\)")


def _quarter_to_date_range(quarter_str: str) -> Tuple[str, str]:
    """
    Convert quarter string to date range.

    Args:
        quarter_str: e.g., "Q2 2026" or "Q2 2026"

    Returns:
        (start_date, end_date) as ISO strings
    """
    match = re.match(r"Q([1-4])\s*(\d{4})", quarter_str.strip())
    if not match:
        raise ValueError(f"Invalid quarter string: {quarter_str}")

    q = int(match.group(1))
    year = int(match.group(2))

    quarter_starts = {1: (1, 1), 2: (4, 1), 3: (7, 1), 4: (10, 1)}
    quarter_ends = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}

    start_month, start_day = quarter_starts[q]
    end_month, end_day = quarter_ends[q]

    return (
        date(year, start_month, start_day).isoformat(),
        date(year, end_month, end_day).isoformat(),
    )


def _half_year_to_date_range(year_str: str, half: str = "first") -> Tuple[str, str]:
    """
    Convert half-year string to date range.

    Args:
        year_str: e.g., "2026"
        half: "first" or "second"

    Returns:
        (start_date, end_date) as ISO strings
    """
    year = int(year_str)
    if half == "first":
        return (
            date(year, 1, 1).isoformat(),
            date(year, 6, 30).isoformat(),
        )
    else:
        return (
            date(year, 7, 1).isoformat(),
            date(year, 12, 31).isoformat(),
        )


def _parse_exact_date(date_str: str) -> Optional[str]:
    """
    Parse an exact date string like "March 15, 2026" to ISO format.

    Returns:
        ISO date string or None if parse fails
    """
    for fmt in ("%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _extract_timing_events(
    text: str,
    ticker: str,
    filing_date: str,
    as_of_date: Optional[date] = None,
) -> List[Dict[str, Any]]:
    """
    Extract timing events from 8-K filing text.

    Args:
        text: Filing text content
        ticker: Associated ticker symbol
        filing_date: ISO date string of the 8-K filing
        as_of_date: Current analysis date (used for year guard + staleness filter)

    Returns:
        List of event dicts
    """
    events = []

    # Year guard: reject extracted dates outside [as_of_year - 1, as_of_year + 3]
    if as_of_date is not None:
        min_year = as_of_date.year - 1
        max_year = as_of_date.year + 3
    else:
        min_year = 0
        max_year = 9999

    # Staleness cutoff: reject events whose end date is >180d before as_of_date
    staleness_cutoff = (as_of_date - timedelta(days=180)).isoformat() if as_of_date else None

    for pattern, event_type, precision in TIMING_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            captured = match.group(1).strip()

            if precision == "DAY":
                event_date = _parse_exact_date(captured)
                if not event_date:
                    continue
                event_date_end = None
                confidence = "HIGH"

            elif precision == "QUARTER":
                try:
                    event_date, event_date_end = _quarter_to_date_range(captured)
                except ValueError:
                    continue
                confidence = "MED"

            elif precision == "HALF_YEAR":
                # Determine which half from surrounding context
                context_start = max(0, match.start() - 100)
                context = text[context_start:match.end()].lower()
                half = "second" if "second" in context or "mid" in context or "year-end" in context else "first"
                try:
                    event_date, event_date_end = _half_year_to_date_range(captured, half)
                except (ValueError, TypeError):
                    continue
                confidence = "LOW"

            else:
                continue

            # Year guard: reject dates outside plausible range
            try:
                event_year = int(event_date[:4])
            except (ValueError, TypeError):
                continue
            if event_year < min_year or event_year > max_year:
                logger.debug(
                    f"Year guard: skipping {ticker} {event_type} {event_date} "
                    f"(year {event_year} outside [{min_year}, {max_year}])"
                )
                continue

            # Staleness filter: reject events whose end date is >180d before as_of_date
            end_for_staleness = event_date_end or event_date
            if staleness_cutoff and end_for_staleness < staleness_cutoff:
                logger.debug(
                    f"Staleness filter: skipping {ticker} {event_type} {end_for_staleness} "
                    f"(older than {staleness_cutoff})"
                )
                continue

            events.append({
                "ticker": ticker,
                "event_type": event_type,
                "event_date": event_date,
                "event_date_end": event_date_end,
                "date_precision": precision,
                "event_name": f"8-K: {match.group(0)[:80]}",
                "confidence": confidence,
                "source": "SEC_8K_FILING",
                "disclosed_at": filing_date,
                "tags": ["sec_8k"],
            })

    return events


def _extract_ticker_from_display_names(display_names: List[str]) -> Optional[str]:
    """
    Extract ticker symbol from EDGAR display_names field.

    EDGAR display_names look like:
      "AGIOS PHARMACEUTICALS, INC.  (AGIO)  (CIK 0001439222)"

    Returns first non-CIK parenthesized match, or None.
    """
    for name in display_names:
        for m in _TICKER_FROM_DISPLAY.finditer(name):
            candidate = m.group(1)
            if candidate != "CIK":
                return candidate
    return None


def _rate_limit():
    """Sleep to respect SEC rate limit."""
    time.sleep(_SEC_MIN_INTERVAL)


def _sec_get(session: Any, url: str, **kwargs) -> Any:
    """GET with backoff on 403/429."""
    import requests as _requests
    _rate_limit()
    resp = session.get(url, timeout=30, **kwargs)
    if resp.status_code in (403, 429):
        wait = int(resp.headers.get("Retry-After", 10))
        logger.warning(f"SEC {resp.status_code} on {url}, backing off {wait}s")
        time.sleep(wait)
        _rate_limit()
        resp = session.get(url, timeout=30, **kwargs)
    return resp


def _fetch_filing_text(
    cik: str,
    adsh: str,
    session: Any,
) -> str:
    """
    Fetch the actual text of an 8-K filing from SEC EDGAR Archives.

    Uses index.json (structured) to find filing documents instead of
    scraping the directory listing HTML.

    Args:
        cik: CIK number (with leading zeros)
        adsh: Accession number (e.g., "0001439222-26-000018")
        session: requests.Session with User-Agent set

    Returns:
        Concatenated plain text from the filing documents
    """
    adsh_nodash = adsh.replace("-", "")
    cik_stripped = cik.lstrip("0")
    base_url = f"{SEC_ARCHIVES_BASE}/{cik_stripped}/{adsh_nodash}"

    # Use index.json for structured file listing
    try:
        resp = _sec_get(session, f"{base_url}/index.json")
        if resp.status_code != 200:
            logger.debug(f"Filing index.json {resp.status_code}: {base_url}")
            return ""
        items = resp.json().get("directory", {}).get("item", [])
    except Exception as e:
        logger.debug(f"Filing index.json fetch error: {e}")
        return ""

    # Categorize .htm files from the index
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

    # Fetch exhibits first (press releases have the timing language), then main doc
    all_text_parts = []
    docs_to_fetch = exhibit_files[:3] + main_files[:2]  # Cap at 5 docs per filing

    for filename in docs_to_fetch:
        doc_url = f"{base_url}/{filename}"
        try:
            doc_resp = _sec_get(session, doc_url)
            if doc_resp.status_code == 200:
                # Strip HTML tags to get plain text
                text = re.sub(r"<[^>]+>", " ", doc_resp.text)
                text = re.sub(r"&nbsp;", " ", text)
                text = re.sub(r"&#\d+;", " ", text)
                text = re.sub(r"&\w+;", " ", text)
                text = re.sub(r"\s+", " ", text).strip()
                all_text_parts.append(text)
        except Exception as e:
            logger.debug(f"Doc fetch error {doc_url}: {e}")

    return " ".join(all_text_parts)


def collect_8k_timing_events(
    universe: List[Dict[str, Any]],
    as_of_date: date,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    lookback_days: int = 180,
) -> List[Dict[str, Any]]:
    """
    Collect timing events from SEC 8-K filings for universe tickers.

    Two-phase approach:
      Phase 1: Search EDGAR full-text index for 8-K filings with timing keywords.
               Extract ticker from display_names, filter to our universe.
      Phase 2: For matched filings, fetch actual filing text from EDGAR Archives
               and extract timing phrases with date ranges.

    Args:
        universe: List of universe entries (must have 'ticker' and optionally 'cik')
        as_of_date: Current analysis date
        cache_dir: Directory for caching results
        lookback_days: How far back to search for 8-K filings

    Returns:
        List of event dicts matching corporate catalyst schema
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"8k_catalysts_{as_of_date.isoformat()}.json"

    # Check cache
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            logger.info(f"Loaded {len(cached)} SEC 8-K catalyst events from cache")
            return cached
        except Exception as e:
            logger.warning(f"Cache read error: {e}")

    try:
        import requests
    except ImportError:
        logger.warning("requests library not available; skipping SEC 8-K collection")
        return []

    # Build universe ticker set and ticker→CIK mapping
    ticker_set: Set[str] = set()
    ticker_to_cik: Dict[str, str] = {}
    for entry in universe:
        ticker = entry.get("ticker", "")
        cik = entry.get("cik", "")
        if ticker:
            ticker_set.add(ticker.upper())
        if ticker and cik:
            ticker_to_cik[ticker.upper()] = str(cik)

    # Supplement with SEC's official company_tickers.json for reliable CIK mapping
    try:
        _rate_limit()
        ct_resp = session.get(SEC_COMPANY_TICKERS_URL, timeout=15)
        if ct_resp.status_code == 200:
            for _k, entry in ct_resp.json().items():
                t = entry.get("ticker", "").upper()
                if t in ticker_set and t not in ticker_to_cik:
                    ticker_to_cik[t] = str(entry.get("cik_str", ""))
            logger.info(f"Loaded SEC company_tickers.json: {len(ticker_to_cik)} CIK mappings for universe")
    except Exception as e:
        logger.warning(f"Could not load company_tickers.json: {e}")

    start_date = (as_of_date - timedelta(days=lookback_days)).isoformat()
    end_date = as_of_date.isoformat()

    # ── Phase 1: Search EDGAR for 8-K filings with timing keywords ──
    search_queries = [
        '"topline data" OR "top-line data"',
        '"PDUFA date" OR "PDUFA action date"',
        '"expects results" OR "anticipates data"',
    ]

    # Collect unique filings keyed by accession number
    # {adsh: {ticker, cik, file_date}}
    filings_to_fetch: Dict[str, Dict[str, str]] = {}

    for query in search_queries:
        try:
            params = {
                "q": query,
                "dateRange": "custom",
                "startdt": start_date,
                "enddt": end_date,
                "forms": "8-K",
            }

            resp = _sec_get(session, SEC_SEARCH_URL, params=params)

            if resp.status_code != 200:
                logger.warning(f"SEC search returned {resp.status_code} for: {query[:40]}")
                continue

            results = resp.json()
            hits = results.get("hits", {}).get("hits", [])
            logger.info(f"EDGAR search '{query[:30]}...': {len(hits)} hits")

            for hit in hits:
                source = hit.get("_source", {})
                display_names = source.get("display_names", [])
                file_date = source.get("file_date", "")
                adsh = source.get("adsh", "")
                ciks = source.get("ciks", [])

                if not adsh or not file_date:
                    continue

                # PIT safety: skip filings after as_of_date
                if file_date > end_date:
                    continue

                # Extract ticker from display_names
                ticker = _extract_ticker_from_display_names(display_names)
                if not ticker or ticker.upper() not in ticker_set:
                    continue

                # Use our ticker→CIK mapping (more reliable than search-index ciks)
                ticker_upper = ticker.upper()
                cik = ticker_to_cik.get(ticker_upper) or (ciks[0] if ciks else "")

                # Deduplicate by accession number
                if adsh not in filings_to_fetch:
                    filings_to_fetch[adsh] = {
                        "ticker": ticker_upper,
                        "cik": cik,
                        "file_date": file_date,
                    }

        except Exception as e:
            logger.warning(f"SEC 8-K search error for '{query[:40]}': {e}")

    logger.info(f"Phase 1: {len(filings_to_fetch)} unique filings from universe to fetch")

    # Hard cap: don't fetch more than _MAX_FILINGS_PER_RUN filings
    if len(filings_to_fetch) > _MAX_FILINGS_PER_RUN:
        logger.warning(
            f"Capping filing fetches at {_MAX_FILINGS_PER_RUN} "
            f"(found {len(filings_to_fetch)})"
        )
        filings_to_fetch = dict(list(filings_to_fetch.items())[:_MAX_FILINGS_PER_RUN])

    # ── Phase 2: Fetch filing text and extract timing phrases ──
    all_events = []
    fetch_errors = 0

    for adsh, meta in filings_to_fetch.items():
        ticker = meta["ticker"]
        cik = meta["cik"]
        file_date = meta["file_date"]

        try:
            text = _fetch_filing_text(cik, adsh, session)
            if not text:
                fetch_errors += 1
                continue

            extracted = _extract_timing_events(text, ticker, file_date, as_of_date)
            if extracted:
                logger.info(
                    f"  {ticker} ({adsh}, {file_date}): {len(extracted)} timing events"
                )
            all_events.extend(extracted)

        except Exception as e:
            logger.warning(f"Error processing {ticker} {adsh}: {e}")
            fetch_errors += 1

    if fetch_errors:
        logger.info(f"Phase 2: {fetch_errors} filing fetch errors")

    # Deduplicate events by (ticker, event_type, event_date)
    seen = set()
    deduped = []
    for event in all_events:
        key = (event["ticker"], event["event_type"], event["event_date"])
        if key not in seen:
            seen.add(key)
            deduped.append(event)

    logger.info(
        f"Collected {len(deduped)} SEC 8-K timing events "
        f"({len(all_events) - len(deduped)} duplicates removed)"
    )

    # Cache results
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(deduped, f, indent=2)
    except Exception as e:
        logger.warning(f"Cache write error: {e}")

    return deduped
