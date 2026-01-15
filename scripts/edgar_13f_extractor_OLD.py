"""
edgar_13f_extractor.py

SEC EDGAR 13F Holdings Extractor for Elite Biotech Managers
Extracts holdings data from 13F-HR XML filings and maps to ticker universe.

Author: Wake Robin Capital Management
Date: 2026-01-09
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import time
import urllib.request
import urllib.error
from decimal import Decimal

# ==============================================================================
# CONFIGURATION
# ==============================================================================

SEC_EDGAR_BASE_URL = "https://www.sec.gov"
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
# SEC EDGAR API FUNCTIONS
# ==============================================================================

def fetch_url_with_rate_limit(url: str) -> str:
    """
    Fetch URL content with proper SEC rate limiting and user agent.
    
    SEC Requirements:
    - User-Agent header with contact info
    - <10 requests per second
    """
    headers = {
        'User-Agent': USER_AGENT,
        'Accept-Encoding': 'gzip, deflate'
    }
    
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read().decode('utf-8')
        
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


def find_13f_filing(cik: str, quarter_end: date) -> Optional[str]:
    """
    Find 13F-HR accession number for given CIK and quarter.
    
    Args:
        cik: 10-digit CIK (e.g., '0001263508')
        quarter_end: Quarter end date (e.g., 2024-09-30)
        
    Returns:
        Accession number or None if not found
    """
    # CIK must be 10 digits with leading zeros
    assert len(cik) == 10 and cik.startswith('000')
    
    # Search filings for this CIK
    # SEC EDGAR submissions JSON endpoint
    url = f"{SEC_EDGAR_BASE_URL}/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=13F-HR&dateb=&owner=exclude&count=40&output=atom"
    
    print(f"  Searching 13F filings for CIK {cik}...")
    
    content = fetch_url_with_rate_limit(url)
    if not content:
        return None
    
    # Parse Atom feed
    try:
        root = ET.fromstring(content)
        
        # Namespace handling
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        # Find all entries
        for entry in root.findall('atom:entry', ns):
            # Get filing date
            filing_date_elem = entry.find('atom:filing-date', ns)
            if filing_date_elem is None:
                continue
            
            filing_date = datetime.strptime(filing_date_elem.text, '%Y-%m-%d').date()
            
            # Check if filing is for our quarter
            # 13F-HR must be filed within 45 days of quarter end
            days_after_quarter = (filing_date - quarter_end).days
            
            if 0 <= days_after_quarter <= 45:
                # Found matching filing - extract accession
                link_elem = entry.find('atom:link[@rel="alternate"]', ns)
                if link_elem is not None:
                    href = link_elem.get('href', '')
                    # Extract accession from URL
                    # Format: .../Archives/edgar/data/1263508/0001193125-24-123456.txt
                    parts = href.split('/')
                    if len(parts) > 0:
                        accession_txt = parts[-1]
                        accession = accession_txt.replace('.txt', '').replace('-index.html', '')
                        print(f"    Found filing: {accession} (filed {filing_date})")
                        return accession
        
        print(f"    No 13F-HR found for quarter ending {quarter_end}")
        return None
        
    except ET.ParseError as e:
        print(f"XML parse error: {e}")
        return None


def fetch_13f_xml(cik: str, accession: str) -> Optional[str]:
    """
    Fetch 13F-HR XML document content.
    
    Args:
        cik: 10-digit CIK
        accession: Accession number (e.g., '0001193125-24-123456')
        
    Returns:
        XML content as string or None
    """
    # Remove dashes from accession for URL
    accession_no_dash = accession.replace('-', '')
    
    # Construct URL
    # Format: https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dash}/{accession}-xslForm13F_X01/primary_doc.xml
    # Note: The exact path varies - we need to try multiple patterns
    
    # Pattern 1: Standard information table XML
    url = f"{SEC_EDGAR_BASE_URL}/cgi-bin/viewer?action=view&cik={cik}&accession_number={accession}&xbrl_type=v"
    
    print(f"    Fetching 13F XML...")
    content = fetch_url_with_rate_limit(url)
    
    return content if content else None


def parse_13f_xml(xml_content: str) -> Tuple[List[RawHolding], int]:
    """
    Parse 13F-HR XML to extract holdings.
    
    Args:
        xml_content: XML string from SEC filing
        
    Returns:
        (holdings_list, total_value_kusd)
    """
    holdings = []
    total_value = 0
    
    try:
        root = ET.fromstring(xml_content)
        
        # Find information table
        # Namespace varies by filing, so search broadly
        for table in root.iter():
            if 'informationTable' in table.tag.lower():
                for info_entry in table:
                    if 'infotable' in info_entry.tag.lower():
                        # Extract holding details
                        cusip = None
                        shares = 0
                        value_kusd = 0
                        put_call = ''
                        
                        for elem in info_entry:
                            tag = elem.tag.lower()
                            
                            if 'cusip' in tag:
                                cusip = elem.text.strip() if elem.text else None
                            elif 'shrsorbrnamt' in tag or 'shares' in tag:
                                for sub in elem:
                                    if 'sshprnamt' in sub.tag.lower():
                                        try:
                                            shares = int(sub.text or 0)
                                        except ValueError:
                                            shares = 0
                            elif 'value' in tag:
                                try:
                                    value_kusd = int(elem.text or 0)
                                except ValueError:
                                    value_kusd = 0
                            elif 'putcall' in tag or 'investmentdiscretion' in tag:
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


def map_cusip_to_ticker(cusip: str, cusip_map: Dict[str, str]) -> Optional[str]:
    """
    Map CUSIP to ticker using cached mapping.
    
    For production, you would integrate with:
    - OpenFIGI API (free, requires registration)
    - Your existing CUSIP mapping from SEC work
    - Static mapping for known biotech CUSIPs
    """
    return cusip_map.get(cusip)


# ==============================================================================
# MAIN EXTRACTION LOGIC
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
    
    # Step 1: Find 13F filing
    accession = find_13f_filing(cik, quarter_end)
    if not accession:
        print(f"  No filing found for {manager_name}")
        return None, {}
    
    # Step 2: Fetch XML
    xml_content = fetch_13f_xml(cik, accession)
    if not xml_content:
        print(f"  Failed to fetch XML for {manager_name}")
        return None, {}
    
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
    
    # Step 5: Create filing info
    filing_info = FilingInfo(
        cik=cik,
        manager_name=manager_name,
        quarter_end=quarter_end,
        accession=accession,
        total_value_kusd=total_value_kusd,
        filed_at=datetime.now(),  # In production, extract from filing
        is_amendment=False,  # In production, detect amendments
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
    
    universe_tickers = {s['ticker'] for s in universe if s.get('ticker') != '_XBI_BENCHMARK_'}
    
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
