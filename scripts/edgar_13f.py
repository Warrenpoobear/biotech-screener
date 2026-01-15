"""
SEC EDGAR 13F Fetcher and Parser for Wake Robin Biotech Alpha System

This module fetches and parses 13F-HR filings from SEC EDGAR.

13F-HR (Holdings Report) contains:
- Cover page: filer info, report date, filing date
- Information table: detailed holdings (CUSIP, name, shares, value)

Point-in-time safety:
- filing_date: when the 13F was filed with SEC (public availability date)
- report_date: quarter-end date the holdings represent (e.g., 2025-09-30)
- For backtesting, use filing_date as the "knowledge date"

Usage:
    from wake_robin.providers.sec_13f.edgar_13f import SEC13FFetcher
    
    fetcher = SEC13FFetcher()
    filings = fetcher.get_recent_filings(cik='1074999', count=4)  # Baker Bros
    
    for filing in filings:
        holdings = fetcher.parse_holdings(filing)
        for h in holdings:
            print(f"{h['ticker']}: {h['shares']:,} shares (${h['value']:,.0f})")
"""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, Generator
from pathlib import Path
import json
import hashlib
import time

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from cusip_resolver import CUSIPResolver, resolve_cusip


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Filing13F:
    """Represents a single 13F-HR filing."""
    cik: str
    accession_number: str
    filing_date: date
    report_date: date
    form_type: str
    primary_doc_url: str
    info_table_url: Optional[str] = None
    filer_name: Optional[str] = None
    
    @property
    def filing_id(self) -> str:
        """Unique identifier for this filing."""
        return f"{self.cik}_{self.accession_number}"
    
    def to_dict(self) -> dict:
        return {
            'cik': self.cik,
            'accession_number': self.accession_number,
            'filing_date': self.filing_date.isoformat(),
            'report_date': self.report_date.isoformat(),
            'form_type': self.form_type,
            'filer_name': self.filer_name,
        }


@dataclass
class Holding:
    """Represents a single position from 13F information table."""
    cusip: str
    issuer_name: str
    class_title: str
    shares: int
    value: int  # In dollars (13F reports in thousands, we convert)
    shares_type: str  # 'SH' for shares, 'PRN' for principal amount
    investment_discretion: str  # 'SOLE', 'SHARED', 'NONE'
    voting_authority_sole: int = 0
    voting_authority_shared: int = 0
    voting_authority_none: int = 0
    put_call: Optional[str] = None  # 'PUT', 'CALL', or None for common stock
    
    # Enriched fields (added after parsing)
    ticker: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            'cusip': self.cusip,
            'ticker': self.ticker,
            'issuer_name': self.issuer_name,
            'class_title': self.class_title,
            'shares': self.shares,
            'value': self.value,
            'shares_type': self.shares_type,
            'put_call': self.put_call,
        }


# =============================================================================
# SEC EDGAR 13F FETCHER
# =============================================================================

class SEC13FFetcher:
    """
    Fetches 13F filings from SEC EDGAR.
    
    Uses the EDGAR full-text search and filing index APIs.
    """
    
    EDGAR_BASE = 'https://www.sec.gov'
    EDGAR_ARCHIVES = 'https://www.sec.gov/cgi-bin/browse-edgar'
    EDGAR_FILINGS = 'https://data.sec.gov/submissions'
    
    # SEC requires a User-Agent header
    USER_AGENT = 'Wake Robin Capital claude@anthropic.com'
    
    def __init__(
        self,
        cusip_resolver: Optional[CUSIPResolver] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize fetcher.
        
        Args:
            cusip_resolver: Resolver for CUSIP→ticker mapping
            cache_dir: Directory to cache downloaded filings
        """
        self.resolver = cusip_resolver or CUSIPResolver()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._session = None
    
    @property
    def session(self):
        """Lazy-init requests session with proper headers."""
        if self._session is None and HAS_REQUESTS:
            self._session = requests.Session()
            self._session.headers.update({
                'User-Agent': self.USER_AGENT,
                'Accept-Encoding': 'gzip, deflate',
            })
        return self._session
    
    def get_recent_filings(
        self,
        cik: str,
        count: int = 4,
        form_type: str = '13F-HR',
    ) -> list[Filing13F]:
        """
        Get recent 13F filings for a CIK.
        
        Args:
            cik: SEC Central Index Key
            count: Number of recent filings to retrieve
            form_type: '13F-HR' for regular, '13F-HR/A' for amendments
            
        Returns:
            List of Filing13F objects, most recent first
        """
        if not HAS_REQUESTS:
            raise RuntimeError("requests library required for EDGAR access")
        
        # Normalize CIK (remove leading zeros for API, keep for URLs)
        cik_int = int(cik)
        cik_padded = str(cik_int).zfill(10)
        
        # Fetch filing history from EDGAR JSON API
        url = f"{self.EDGAR_FILINGS}/CIK{cik_padded}.json"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error fetching filing index for CIK {cik}: {e}")
            return []
        
        filings = []
        filer_name = data.get('name', '')
        
        # Parse recent filings
        recent = data.get('filings', {}).get('recent', {})
        forms = recent.get('form', [])
        accessions = recent.get('accessionNumber', [])
        filing_dates = recent.get('filingDate', [])
        report_dates = recent.get('reportDate', [])
        primary_docs = recent.get('primaryDocument', [])
        
        for i, form in enumerate(forms):
            if form == form_type or (form_type == '13F-HR' and form in ('13F-HR', '13F-HR/A')):
                if len(filings) >= count:
                    break
                
                accession = accessions[i].replace('-', '')
                filing_date_str = filing_dates[i]
                report_date_str = report_dates[i] if i < len(report_dates) else filing_dates[i]
                primary_doc = primary_docs[i] if i < len(primary_docs) else ''
                
                # Construct URLs
                accession_formatted = accessions[i]
                base_url = f"{self.EDGAR_BASE}/Archives/edgar/data/{cik_int}/{accession}"
                primary_url = f"{base_url}/{primary_doc}"
                
                filing = Filing13F(
                    cik=cik,
                    accession_number=accession,
                    filing_date=datetime.strptime(filing_date_str, '%Y-%m-%d').date(),
                    report_date=datetime.strptime(report_date_str, '%Y-%m-%d').date(),
                    form_type=form,
                    primary_doc_url=primary_url,
                    filer_name=filer_name,
                )
                
                filings.append(filing)
        
        return filings
    
    def find_info_table_url(self, filing: Filing13F) -> Optional[str]:
        """
        Find the information table XML URL for a filing.
        
        The info table is usually named something like:
        - infotable.xml
        - form13fInfoTable.xml
        - *_infotable.xml
        """
        if not HAS_REQUESTS:
            return None
        
        # Fetch filing index
        cik_int = int(filing.cik)
        accession = filing.accession_number
        index_url = f"{self.EDGAR_BASE}/Archives/edgar/data/{cik_int}/{accession}/index.json"
        
        try:
            response = self.session.get(index_url, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error fetching filing index: {e}")
            return None
        
        # Look for info table
        for item in data.get('directory', {}).get('item', []):
            name = item.get('name', '').lower()
            if 'infotable' in name and name.endswith('.xml'):
                return f"{self.EDGAR_BASE}/Archives/edgar/data/{cik_int}/{accession}/{item['name']}"
        
        return None
    
    def fetch_info_table_xml(self, filing: Filing13F) -> Optional[str]:
        """
        Fetch the raw XML content of the information table.
        
        Returns cached version if available.
        """
        if not HAS_REQUESTS:
            return None
        
        # Check cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{filing.filing_id}_infotable.xml"
            if cache_file.exists():
                return cache_file.read_text(encoding='utf-8')
        
        # Find URL
        info_url = filing.info_table_url or self.find_info_table_url(filing)
        if not info_url:
            print(f"Could not find info table for {filing.filing_id}")
            return None
        
        # Fetch
        try:
            # Respect SEC rate limits
            time.sleep(0.1)
            response = self.session.get(info_url, timeout=30)
            response.raise_for_status()
            xml_content = response.text
            
            # Cache
            if self.cache_dir:
                cache_file.write_text(xml_content, encoding='utf-8')
            
            return xml_content
            
        except Exception as e:
            print(f"Error fetching info table: {e}")
            return None
    
    def parse_holdings(
        self,
        filing: Filing13F,
        resolve_tickers: bool = True,
    ) -> list[Holding]:
        """
        Parse holdings from a 13F filing's information table.
        
        Args:
            filing: Filing13F object
            resolve_tickers: Whether to resolve CUSIP→ticker
            
        Returns:
            List of Holding objects
        """
        xml_content = self.fetch_info_table_xml(filing)
        if not xml_content:
            return []
        
        holdings = []
        
        try:
            # Parse XML
            root = ET.fromstring(xml_content)
            
            # Handle namespace
            ns = {}
            if root.tag.startswith('{'):
                ns_uri = root.tag.split('}')[0] + '}'
                ns = {'ns': ns_uri[1:-1]}
            
            # Find info table entries
            # Common paths: //infoTable or //ns:infoTable
            entries = root.findall('.//infoTable', ns) or root.findall('.//{*}infoTable')
            
            if not entries:
                # Try without namespace
                entries = [e for e in root.iter() if 'infotable' in e.tag.lower()]
            
            for entry in entries:
                holding = self._parse_info_table_entry(entry)
                if holding:
                    holdings.append(holding)
            
            # Resolve tickers
            if resolve_tickers and holdings:
                cusips = [h.cusip for h in holdings]
                mappings = self.resolver.resolve_batch(cusips)
                for holding in holdings:
                    holding.ticker = mappings.get(holding.cusip)
            
        except ET.ParseError as e:
            print(f"XML parse error for {filing.filing_id}: {e}")
        
        return holdings
    
    def _parse_info_table_entry(self, entry: ET.Element) -> Optional[Holding]:
        """Parse a single infoTable entry."""
        def get_text(tag: str) -> str:
            """Get text from child element, handling namespaces."""
            for child in entry.iter():
                if tag.lower() in child.tag.lower():
                    return (child.text or '').strip()
            return ''
        
        def get_int(tag: str) -> int:
            """Get integer from child element."""
            text = get_text(tag)
            try:
                return int(text.replace(',', ''))
            except ValueError:
                return 0
        
        cusip = get_text('cusip')
        if not cusip:
            return None
        
        # Value is in thousands, convert to dollars
        value_thousands = get_int('value')
        value = value_thousands * 1000
        
        shares = get_int('sshPrnamt') or get_int('shares')
        shares_type = get_text('sshPrnamtType') or 'SH'
        
        # Voting authority
        voting = {}
        for child in entry.iter():
            tag_lower = child.tag.lower()
            if 'sole' in tag_lower and 'voting' in tag_lower:
                voting['sole'] = int((child.text or '0').replace(',', ''))
            elif 'shared' in tag_lower and 'voting' in tag_lower:
                voting['shared'] = int((child.text or '0').replace(',', ''))
            elif 'none' in tag_lower and 'voting' in tag_lower:
                voting['none'] = int((child.text or '0').replace(',', ''))
        
        return Holding(
            cusip=cusip,
            issuer_name=get_text('nameOfIssuer'),
            class_title=get_text('titleOfClass'),
            shares=shares,
            value=value,
            shares_type=shares_type,
            investment_discretion=get_text('investmentDiscretion') or 'SOLE',
            voting_authority_sole=voting.get('sole', 0),
            voting_authority_shared=voting.get('shared', 0),
            voting_authority_none=voting.get('none', 0),
            put_call=get_text('putCall') or None,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_manager_holdings(
    cik: str,
    quarters_back: int = 1,
    cache_dir: str = 'data/13f_cache',
) -> dict:
    """
    Get holdings for a manager, most recent filing.
    
    Returns dict with:
        - filing: Filing13F metadata
        - holdings: List of Holding objects
        - summary: Aggregated stats
    """
    fetcher = SEC13FFetcher(cache_dir=cache_dir)
    
    filings = fetcher.get_recent_filings(cik, count=quarters_back)
    if not filings:
        return {'error': f'No 13F filings found for CIK {cik}'}
    
    filing = filings[0]  # Most recent
    holdings = fetcher.parse_holdings(filing)
    
    # Compute summary
    total_value = sum(h.value for h in holdings)
    positions_count = len(holdings)
    top_10_value = sum(h.value for h in sorted(holdings, key=lambda x: -x.value)[:10])
    
    return {
        'filing': filing.to_dict(),
        'holdings': [h.to_dict() for h in holdings],
        'summary': {
            'total_value': total_value,
            'positions_count': positions_count,
            'top_10_concentration': top_10_value / total_value if total_value else 0,
        }
    }


def holdings_hash(holdings: list[Holding]) -> str:
    """
    Generate deterministic hash of holdings.
    
    Use for point-in-time safety verification.
    """
    # Sort by CUSIP for determinism
    data = [(h.cusip, h.shares, h.value) for h in sorted(holdings, key=lambda x: x.cusip)]
    content = json.dumps(data, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# =============================================================================
# CLI FOR TESTING
# =============================================================================

if __name__ == '__main__':
    import sys
    
    # Default to Baker Bros if no CIK provided
    cik = sys.argv[1] if len(sys.argv) > 1 else '1074999'
    
    print(f"Fetching 13F holdings for CIK {cik}")
    print("=" * 60)
    
    result = get_manager_holdings(cik, quarters_back=1)
    
    if 'error' in result:
        print(result['error'])
        sys.exit(1)
    
    filing = result['filing']
    print(f"Filer: {filing.get('filer_name', 'Unknown')}")
    print(f"Report Date: {filing['report_date']}")
    print(f"Filing Date: {filing['filing_date']}")
    print()
    
    summary = result['summary']
    print(f"Total Value: ${summary['total_value']:,.0f}")
    print(f"Positions: {summary['positions_count']}")
    print(f"Top 10 Concentration: {summary['top_10_concentration']:.1%}")
    print()
    
    print("TOP 20 HOLDINGS:")
    print("-" * 60)
    
    holdings = sorted(result['holdings'], key=lambda x: -x['value'])[:20]
    for h in holdings:
        ticker = h['ticker'] or h['cusip']
        pct = h['value'] / summary['total_value'] * 100
        print(f"  {ticker:8} {h['issuer_name'][:30]:30} ${h['value']:>12,.0f} ({pct:5.2f}%)")
