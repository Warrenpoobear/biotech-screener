"""
edgar_13f_extractor.py (PRODUCTION-CORRECTED)

SEC EDGAR 13F Holdings Extractor for Elite Biotech Managers
Extracts holdings data from 13F-HR XML filings and maps to ticker universe.

CRITICAL FIXES APPLIED:
- Gzip decompression handling
- data.sec.gov submissions API (official endpoint)
- Deterministic filed_at from EDGAR metadata
- Corrected put/call parsing (removed investmentDiscretion)

Author: Wake Robin Capital Management
Date: 2026-01-09 (Production-Corrected)
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
import urllib.request
import urllib.error
import gzip

# ==============================================================================
# NAMESPACE HELPER
# ==============================================================================

def strip_ns(tag: str) -> str:
    """Strip XML namespace prefix from tag (e.g., '{...}value' -> 'value')"""
    return tag.split('}')[-1].lower()

# ==============================================================================
# CONFIGURATION
# ==============================================================================

SEC_EDGAR_BASE_URL = "https://www.sec.gov"
SEC_DATA_API_BASE = "https://data.sec.gov"
USER_AGENT = "Wake Robin Capital Management institutional.validation@wakerobincapital.com"

# Rate limiting (SEC requires <10 requests/second)
REQUEST_DELAY_SECONDS = 0.15  # ~6.7 requests/second

# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass
class RawHolding:
    """Raw holding from 13F XML"""
    cusip: str
    shares: int
    value_kusd: int  # In thousands (native 13F units)
    put_call: str  # '', 'PUT', or 'CALL'
    
@dataclass
class FilingInfo:
    """13F filing metadata"""
    cik: str
    manager_name: str
    quarter_end: date
    accession: str
    total_value_kusd: int
    filed_at: datetime
    is_amendment: bool
    holdings_count: int

# ==============================================================================
# SEC EDGAR API FUNCTIONS (PRODUCTION-CORRECTED)
# ==============================================================================

def fetch_url_with_rate_limit(url: str) -> str:
    """
    Fetch URL content with proper SEC rate limiting, user agent, and gzip handling.
    
    CRITICAL FIX: Decompress gzip responses before decoding.
    
    SEC Requirements:
    - User-Agent header with contact info
    - <10 requests per second
    - Handle gzip compression
    """
    headers = {
        'User-Agent': USER_AGENT,
        'Accept-Encoding': 'gzip, deflate'
    }
    
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read()
            
            # CRITICAL FIX: Decompress if gzipped
            if response.headers.get('Content-Encoding', '').lower() == 'gzip':
                raw = gzip.decompress(raw)
            
            # Use 'replace' for robustness with malformed characters
            content = raw.decode('utf-8', errors='replace')
        
        # Rate limiting
        time.sleep(REQUEST_DELAY_SECONDS)
        
        return content
        
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code} fetching {url}: {e.reason}")
        return ""
    except urllib.error.URLError as e:
        print(f"URL Error fetching {url}: {e.reason}")
        return ""
    except Exception as e:
        print(f"Unexpected error fetching {url}: {e}")
        return ""


def find_13f_filing_via_submissions_api(
    cik: str,
    quarter_end: date
) -> Optional[Tuple[str, date, str, bool]]:
    """
    Find 13F-HR filing using official data.sec.gov submissions API.

    PRODUCTION FIX: Use submissions JSON API instead of Atom feed parsing.

    Args:
        cik: 10-digit CIK (e.g., '0001263508')
        quarter_end: Quarter end date (e.g., 2024-09-30)

    Returns:
        (accession, filing_date, primary_document_url, is_amendment) or None
        is_amendment is True for 13F-HR/A filings, False for 13F-HR

    API Reference:
    https://www.sec.gov/edgar/sec-api-documentation
    """
    # CIK must be 10 digits with leading zeros
    assert len(cik) == 10 and cik.startswith('000'), f"Invalid CIK: {cik}"
    
    # Submissions endpoint
    url = f"{SEC_DATA_API_BASE}/submissions/CIK{cik}.json"
    
    print(f"  Querying submissions API for CIK {cik}...")
    
    content = fetch_url_with_rate_limit(url)
    if not content:
        return None
    
    try:
        data = json.loads(content)
        
        # Recent filings in 'filings.recent'
        recent = data.get('filings', {}).get('recent', {})
        
        if not recent:
            print(f"    No recent filings found")
            return None
        
        accession_list = recent.get('accessionNumber', [])
        filing_date_list = recent.get('filingDate', [])
        form_list = recent.get('form', [])
        primary_doc_list = recent.get('primaryDocument', [])
        
        # Find 13F-HR filings within 45 days of quarter end
        # Match both original filings (13F-HR) and amendments (13F-HR/A)
        for i, form in enumerate(form_list):
            if form not in ('13F-HR', '13F-HR/A'):
                continue

            filing_date_str = filing_date_list[i]
            filing_date = datetime.strptime(filing_date_str, '%Y-%m-%d').date()

            # Check if filing is for our quarter (within 45 days after quarter end)
            days_after_quarter = (filing_date - quarter_end).days

            if 0 <= days_after_quarter <= 45:
                accession = accession_list[i]
                primary_document = primary_doc_list[i].split('/')[-1]  # Strip xslForm13F_X02/ prefix

                is_amendment = form == '13F-HR/A'
                amendment_str = " (AMENDMENT)" if is_amendment else ""
                print(f"    Found {form}: {accession} (filed {filing_date}){amendment_str}")

                # Construct primary document URL
                # Format: https://www.sec.gov/Archives/edgar/data/{cik_no_leading_zeros}/{accession_no_dashes}/{primary_doc}
                cik_no_zeros = cik.lstrip('0')
                accession_no_dashes = accession.replace('-', '')

                doc_url = f"{SEC_EDGAR_BASE_URL}/Archives/edgar/data/{cik_no_zeros}/{accession_no_dashes}/{primary_document}"

                return accession, filing_date, doc_url, is_amendment

        print(f"    No 13F-HR found for quarter ending {quarter_end}")
        return None
        
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error parsing submissions: {e}")
        return None


def fetch_information_table_xml(
    cik: str,
    accession: str,
    primary_doc_url: str
) -> Optional[Tuple[str, str]]:
    """
    Fetch 13F information table XML document.
    
    PRODUCTION FIX: Fetch actual filing documents, not viewer page.
    
    Strategy:
    1. Try primary document from submissions API
    2. If that's not XML, look for "informationTable.xml" in filing directory
    
    Args:
        cik: 10-digit CIK
        accession: Accession number
        primary_doc_url: URL from submissions API
        
    Returns:
        (xml_content, document_url) or None
    """
    print(f"    Fetching information table XML...")
    
    # Try 1: Primary document (often the information table)
    content = fetch_url_with_rate_limit(primary_doc_url)
    
    if content and ('<informationTable' in content or '<ns1:informationTable' in content):
        print(f"      Found information table in primary document")
        return content, primary_doc_url
    
    # Try 2: Look for informationTable.xml in filing directory
    # Extract base directory from primary_doc_url
    base_dir = '/'.join(primary_doc_url.rsplit('/', 1)[:-1])
    
    # Common 13F information table filenames
    common_xml_names = [
        'informationTable.xml',
        'form13fInfoTable.xml',
        'infotable.xml'
    ]
    
    for xml_name in common_xml_names:
        xml_url = f"{base_dir}/{xml_name}"
        content = fetch_url_with_rate_limit(xml_url)
        
        if content and ('<informationTable' in content or '<ns1:informationTable' in content):
            print(f"      Found information table: {xml_name}")
            return content, xml_url
    
    # Try 3: Fetch filing index to find information table
    cik_no_zeros = cik.lstrip('0')
    accession_no_dashes = accession.replace('-', '')
    index_url = f"{SEC_EDGAR_BASE_URL}/cgi-bin/viewer?action=view&cik={cik_no_zeros}&accession_number={accession}&xbrl_type=v"
    
    index_content = fetch_url_with_rate_limit(index_url)
    
    if index_content:
        # Try to extract XML document link from index
        # This is a fallback - production systems should use submissions API properly
        pass
    
    print(f"      Could not locate information table XML")
    return None


def parse_13f_xml(xml_content: str) -> Tuple[List[RawHolding], int]:
    """
    Parse 13F-HR XML to extract holdings.
    
    CRITICAL FIX: Removed investmentDiscretion from put/call detection.
    
    Args:
        xml_content: XML string from SEC filing
        
    Returns:
        (holdings_list, total_value_kusd)
    """
    holdings = []
    total_value = 0
    
    try:
        root = ET.fromstring(xml_content)
        
        # Find information table (namespace-agnostic)
        for table in root.iter():
            tag_lower = strip_ns(table.tag)
            if 'informationtable' not in tag_lower:
                continue
            
            for info_entry in table:
                entry_tag = strip_ns(info_entry.tag)
                if 'infotable' not in entry_tag:
                    continue
                
                # Extract holding details
                cusip = None
                shares = 0
                value_kusd = 0
                put_call = ''
                
                for elem in info_entry:
                    tag = strip_ns(elem.tag)
                    
                    if 'cusip' in tag:
                        cusip = elem.text.strip() if elem.text else None
                    
                    elif 'shrsorbrnamt' in tag or 'sharesorbonds' in tag:
                        # Navigate to sshPrnAmt (shares or principal amount)
                        for sub in elem:
                            if 'sshprnamt' in strip_ns(sub.tag):
                                try:
                                    shares = int(sub.text or 0)
                                except ValueError:
                                    shares = 0
                    
                    elif tag == 'value':
                        # Only match exact 'value' tag, not substrings
                        try:
                            value_kusd = int(elem.text or 0)
                        except ValueError:
                            value_kusd = 0
                    
                    # CRITICAL FIX: Only check 'putcall', NOT 'investmentDiscretion'
                    elif 'putcall' in tag and tag.endswith('putcall'):
                        pc = (elem.text or '').strip().upper()
                        if pc in ('PUT', 'CALL'):
                            put_call = pc
                
                # Valid holding must have CUSIP and value
                if cusip and value_kusd > 0:
                    holdings.append(RawHolding(
                        cusip=cusip,
                        shares=shares,
                        value_kusd=value_kusd,
                        put_call=put_call
                    ))
                    total_value += value_kusd
        
        return holdings, total_value
        
    except ET.ParseError as e:
        print(f"XML parse error: {e}")
        return [], 0


# ==============================================================================
# CUSIP MAPPING
# ==============================================================================

def load_cusip_ticker_map(cache_path: Path) -> Dict[str, str]:
    """
    Load CUSIP→Ticker mapping from cache.
    
    Cache format:
    {
      "037833100": "AAPL",
      "459200101": "IBM",
      ...
    }
    """
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


def map_cusip_to_ticker(cusip: str, cusip_map: Dict[str, object]) -> Optional[str]:
    """
    Map CUSIP to ticker using cached mapping.
    
    Supports these cusip_map value shapes:
      1) "AAPL"
      2) {"ticker": "AAPL", ...}
      3) [{"ticker": "AAPL"}, ...]  (takes first with ticker)
    
    Returns:
        Ticker string (uppercase) or None
    """
    v = cusip_map.get(cusip)
    if not v:
        return None
    
    # Most common: direct string mapping
    if isinstance(v, str):
        t = v.strip().upper()
        return t or None
    
    # Dict mapping: {"ticker": "..."}
    if isinstance(v, dict):
        t = v.get("ticker") or v.get("symbol")
        if isinstance(t, str):
            t = t.strip().upper()
            return t or None
        return None
    
    # List mapping: [{"ticker": "..."}, ...]
    if isinstance(v, list):
        for item in v:
            if isinstance(item, dict):
                t = item.get("ticker") or item.get("symbol")
                if isinstance(t, str) and t.strip():
                    return t.strip().upper()
        return None
    
    return None


# ==============================================================================
# MAIN EXTRACTION LOGIC (PRODUCTION-CORRECTED)
# ==============================================================================

def extract_manager_holdings(
    cik: str,
    manager_name: str,
    quarter_end: date,
    cusip_map: Dict[str, str],
    universe_tickers: set[str]
) -> Tuple[Optional[FilingInfo], Dict[str, RawHolding]]:
    """
    Extract holdings for one manager for one quarter.
    
    PRODUCTION FIX: Uses deterministic filed_at from EDGAR metadata.
    
    Args:
        cik: Manager CIK
        manager_name: Manager name (for logging)
        quarter_end: Quarter end date
        cusip_map: CUSIP→Ticker mapping
        universe_tickers: Set of tickers in screening universe
        
    Returns:
        (filing_info, holdings_dict) where holdings_dict = {ticker: RawHolding}
    """
    print(f"\nProcessing {manager_name} ({cik}) for Q ending {quarter_end}...")
    
    # Step 1: Find 13F filing via submissions API
    found = find_13f_filing_via_submissions_api(cik, quarter_end)
    if not found:
        print(f"  No filing found for {manager_name}")
        return None, {}

    accession, filing_date, primary_doc_url, is_amendment = found

    # Step 2: Fetch information table XML
    xml_result = fetch_information_table_xml(cik, accession, primary_doc_url)
    if not xml_result:
        print(f"  Failed to fetch information table XML for {manager_name}")
        return None, {}
    
    xml_content, doc_url = xml_result
    
    # Step 3: Parse holdings
    raw_holdings, total_value_kusd = parse_13f_xml(xml_content)
    print(f"  Parsed {len(raw_holdings)} holdings, total value: ${total_value_kusd:,}K")
    
    # Step 4: Map CUSIPs to tickers and filter to universe
    ticker_holdings = {}
    
    for holding in raw_holdings:
        ticker = map_cusip_to_ticker(holding.cusip, cusip_map)
        
        if ticker and ticker in universe_tickers:
            ticker_holdings[ticker] = holding
    
    print(f"  Matched {len(ticker_holdings)} holdings to universe")
    
    # Step 5: Create filing info with DETERMINISTIC filed_at
    # CRITICAL FIX: Use filing_date from EDGAR, not datetime.now()
    filing_info = FilingInfo(
        cik=cik,
        manager_name=manager_name,
        quarter_end=quarter_end,
        accession=accession,
        total_value_kusd=total_value_kusd,
        filed_at=datetime.combine(filing_date, datetime.min.time()),  # Deterministic!
        is_amendment=is_amendment,  # Detected from SEC form type (13F-HR vs 13F-HR/A)
        holdings_count=len(raw_holdings)
    )
    
    return filing_info, ticker_holdings


def extract_all_elite_holdings(
    manager_registry_path: Path,
    universe_path: Path,
    cusip_map_path: Path,
    quarter_end: date,
    output_path: Path
) -> None:
    """
    Extract 13F holdings for all elite managers.
    
    Creates holdings_snapshots.json with structure:
    {
      "ARGX": {
        "market_cap_usd": 12000000000,
        "holdings": {
          "current": {
            "0001263508": {...},
            ...
          },
          "prior": {
            "0001263508": {...},
            ...
          }
        },
        "filings_metadata": {
          "0001263508": {...},
          ...
        }
      },
      ...
    }
    """
    print(f"\n{'='*80}")
    print(f"EXTRACTING 13F HOLDINGS FOR Q ENDING {quarter_end}")
    print(f"{'='*80}\n")
    
    # Load inputs
    with open(manager_registry_path) as f:
        registry = json.load(f)
    
    with open(universe_path) as f:
        universe = json.load(f)
    
    universe_tickers = {
        s.get("ticker").strip().upper()
        for s in universe
        if isinstance(s.get("ticker"), str) and s.get("ticker").strip() and s.get("ticker") != "_XBI_BENCHMARK_"
    }
    
    cusip_map = load_cusip_ticker_map(cusip_map_path)
    
    # Calculate prior quarter
    # Q1 (3/31) → prior Q4 (12/31 prev year)
    # Q2 (6/30) → prior Q1 (3/31)
    # Q3 (9/30) → prior Q2 (6/30)
    # Q4 (12/31) → prior Q3 (9/30)
    
    quarter_map = {
        (3, 31): (12, 31, -1),  # Q1 → Q4 prev year
        (6, 30): (3, 31, 0),    # Q2 → Q1
        (9, 30): (6, 30, 0),    # Q3 → Q2
        (12, 31): (9, 30, 0)    # Q4 → Q3
    }
    
    prior_month, prior_day, year_offset = quarter_map[(quarter_end.month, quarter_end.day)]
    prior_quarter_end = date(quarter_end.year + year_offset, prior_month, prior_day)
    
    print(f"Current quarter: {quarter_end}")
    print(f"Prior quarter: {prior_quarter_end}\n")
    
    # Extract holdings for each manager
    all_current_holdings = {}  # {cik: {ticker: RawHolding}}
    all_prior_holdings = {}
    all_filings_metadata = {}  # {cik: FilingInfo}
    
    elite_managers = registry['elite_core']
    
    for manager in elite_managers:
        cik = manager['cik']
        name = manager['name']
        
        # Extract current quarter
        filing_current, holdings_current = extract_manager_holdings(
            cik=cik,
            manager_name=name,
            quarter_end=quarter_end,
            cusip_map=cusip_map,
            universe_tickers=universe_tickers
        )
        
        if filing_current:
            all_current_holdings[cik] = holdings_current
            all_filings_metadata[cik] = filing_current
        
        # Extract prior quarter
        filing_prior, holdings_prior = extract_manager_holdings(
            cik=cik,
            manager_name=name,
            quarter_end=prior_quarter_end,
            cusip_map=cusip_map,
            universe_tickers=universe_tickers
        )
        
        if filing_prior:
            all_prior_holdings[cik] = holdings_prior
    
    # Organize by ticker
    ticker_snapshots = {}
    
    for ticker in universe_tickers:
        # Get market cap from universe
        ticker_info = next((s for s in universe if s.get('ticker') == ticker), None)
        market_cap = ticker_info.get('market_cap_usd', 0) if ticker_info else 0
        
        current_holdings = {}
        prior_holdings = {}
        
        for cik in all_current_holdings:
            if ticker in all_current_holdings[cik]:
                holding = all_current_holdings[cik][ticker]
                current_holdings[cik] = {
                    'quarter_end': quarter_end.isoformat(),
                    'state': 'KNOWN',
                    'shares': holding.shares,
                    'value_kusd': holding.value_kusd,
                    'put_call': holding.put_call
                }
        
        for cik in all_prior_holdings:
            if ticker in all_prior_holdings[cik]:
                holding = all_prior_holdings[cik][ticker]
                prior_holdings[cik] = {
                    'quarter_end': prior_quarter_end.isoformat(),
                    'state': 'KNOWN',
                    'shares': holding.shares,
                    'value_kusd': holding.value_kusd,
                    'put_call': holding.put_call
                }
        
        # Only include ticker if at least one manager holds it
        if current_holdings or prior_holdings:
            ticker_snapshots[ticker] = {
                'market_cap_usd': market_cap,
                'holdings': {
                    'current': current_holdings,
                    'prior': prior_holdings
                },
                'filings_metadata': {
                    cik: {
                        'quarter_end': info.quarter_end.isoformat(),
                        'accession': info.accession,
                        'total_value_kusd': info.total_value_kusd,
                        'filed_at': info.filed_at.isoformat(),
                        'is_amendment': info.is_amendment
                    }
                    for cik, info in all_filings_metadata.items()
                }
            }
    
    # Write output
    with open(output_path, 'w') as f:
        json.dump(ticker_snapshots, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"Tickers with institutional coverage: {len(ticker_snapshots)}")
    print(f"Output saved to: {output_path}")
    print(f"{'='*80}\n")


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract 13F holdings from SEC EDGAR for elite biotech managers"
    )
    parser.add_argument(
        '--quarter-end',
        type=lambda s: datetime.strptime(s, '%Y-%m-%d').date(),
        required=True,
        help='Quarter end date (YYYY-MM-DD), e.g., 2024-09-30'
    )
    parser.add_argument(
        '--manager-registry',
        type=Path,
        required=True,
        help='Path to manager_registry.json'
    )
    parser.add_argument(
        '--universe',
        type=Path,
        required=True,
        help='Path to universe.json'
    )
    parser.add_argument(
        '--cusip-map',
        type=Path,
        required=True,
        help='Path to CUSIP→Ticker mapping JSON'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output path for holdings_snapshots.json'
    )
    
    args = parser.parse_args()
    
    extract_all_elite_holdings(
        manager_registry_path=args.manager_registry,
        universe_path=args.universe,
        cusip_map_path=args.cusip_map,
        quarter_end=args.quarter_end,
        output_path=args.output
    )
