#!/usr/bin/env python3
"""
sec_filing_downloader.py - Automated SEC Filing Downloader

Downloads 10-Q and 10-K filings from SEC EDGAR with proper rate limiting.

Usage:
    python sec_filing_downloader.py --tickers CVAC,RYTM,IMMP --count 8

Environment Variables:
    SEC_USER_AGENT: Your User-Agent string (required by SEC)
                   Format: "YourName/1.0 (your.email@example.com)"
"""

import os
import requests
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import re
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# SEC requires User-Agent with contact info
# Set via SEC_USER_AGENT environment variable or pass explicitly
_DEFAULT_USER_AGENT_PLACEHOLDER = "YourName/1.0 (your.email@example.com)"


def get_sec_user_agent() -> str:
    """
    Get SEC User-Agent from environment variable.

    SEC requires all EDGAR requests to include a User-Agent header with:
    - Company or application name
    - Contact email address

    Returns:
        User-Agent string from SEC_USER_AGENT env var, or placeholder if not set
    """
    user_agent = os.environ.get("SEC_USER_AGENT", "").strip()

    if not user_agent:
        logger.warning(
            "SEC_USER_AGENT environment variable not set. "
            "SEC requires a valid User-Agent with your contact info. "
            "Set it with: export SEC_USER_AGENT='YourName/1.0 (your.email@example.com)'"
        )
        return _DEFAULT_USER_AGENT_PLACEHOLDER

    # Validate format (should contain email-like pattern)
    if "@" not in user_agent:
        logger.warning(
            f"SEC_USER_AGENT '{user_agent}' may be invalid - should include email address"
        )

    return user_agent


USER_AGENT = get_sec_user_agent()

# SEC rate limit: 10 requests per second
RATE_LIMIT_DELAY = 0.11  # 110ms between requests (slightly over 100ms to be safe)

# Base URLs
SEC_SEARCH_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}.txt"


class SECDownloader:
    """Automated SEC filing downloader with rate limiting"""
    
    def __init__(self, output_dir: Path = Path("filings"), user_agent: str = USER_AGENT):
        self.output_dir = output_dir
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': user_agent,
            'Accept-Encoding': 'gzip, deflate',
        })
        self.last_request_time = 0
        self.download_log = {}
        
    def _rate_limit(self):
        """Enforce SEC rate limit of 10 requests/second"""
        elapsed = time.time() - self.last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self.last_request_time = time.time()
    
    def get_cik(self, ticker: str) -> Optional[str]:
        """
        Get CIK (Central Index Key) for a ticker.
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            CIK as string (zero-padded to 10 digits) or None if not found
        """
        # Try to look up CIK from SEC's ticker list
        # For now, we'll use a simple mapping approach
        # In production, you'd want to maintain a ticker->CIK mapping file
        
        ticker = ticker.upper().strip()
        
        # SEC company tickers JSON (updated daily)
        url = "https://www.sec.gov/files/company_tickers.json"
        
        self._rate_limit()
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            companies = response.json()
            
            # Find matching ticker
            for company_data in companies.values():
                if company_data.get('ticker', '').upper() == ticker:
                    cik = str(company_data['cik_str']).zfill(10)
                    logger.info(f"Found CIK for {ticker}: {cik}")
                    return cik
            
            logger.warning(f"CIK not found for ticker: {ticker}")
            return None
            
        except Exception as e:
            logger.error(f"Error looking up CIK for {ticker}: {e}")
            return None
    
    def get_filings_list(
        self,
        cik: str,
        form_types: List[str] = ["10-Q", "10-K"],
        count: int = 8
    ) -> List[Dict]:
        """
        Get list of filings for a CIK.
        
        Args:
            cik: Central Index Key (10 digits, zero-padded)
            form_types: List of form types to download
            count: Maximum number of filings to retrieve
        
        Returns:
            List of filing metadata dicts
        """
        url = SEC_SEARCH_URL.format(cik=cik)
        
        self._rate_limit()
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract recent filings
            filings = data.get('filings', {}).get('recent', {})
            
            if not filings:
                logger.warning(f"No filings found for CIK {cik}")
                return []
            
            # Parse filings
            form_list = filings.get('form', [])
            filing_dates = filings.get('filingDate', [])
            accession_numbers = filings.get('accessionNumber', [])
            primary_documents = filings.get('primaryDocument', [])
            
            results = []
            for i in range(len(form_list)):
                form_type = form_list[i]
                
                # Filter by form type
                if form_type not in form_types:
                    continue
                
                # Remove dashes from accession number for URL
                accession = accession_numbers[i].replace('-', '')
                
                filing_info = {
                    'form_type': form_type,
                    'filing_date': filing_dates[i],
                    'accession_number': accession_numbers[i],
                    'accession_number_clean': accession,
                    'primary_document': primary_documents[i],
                    'cik': cik
                }
                
                results.append(filing_info)
                
                if len(results) >= count:
                    break
            
            logger.info(f"Found {len(results)} filings for CIK {cik}")
            return results
            
        except Exception as e:
            logger.error(f"Error fetching filings list for CIK {cik}: {e}")
            return []
    
    def download_filing(
        self,
        ticker: str,
        filing_info: Dict,
        overwrite: bool = False
    ) -> Optional[Path]:
        """
        Download a single filing.
        
        Args:
            ticker: Stock ticker
            filing_info: Filing metadata from get_filings_list()
            overwrite: Whether to overwrite existing files
        
        Returns:
            Path to downloaded file or None if failed
        """
        cik = filing_info['cik']
        accession = filing_info['accession_number_clean']
        form_type = filing_info['form_type']
        filing_date = filing_info['filing_date']
        primary_doc = filing_info.get('primary_document', '')
        
        # Create output directory
        ticker_dir = self.output_dir / ticker.upper()
        ticker_dir.mkdir(parents=True, exist_ok=True)
        
        # Output filename
        filename = f"{form_type}_{filing_date}_{filing_info['accession_number']}.txt"
        output_path = ticker_dir / filename
        
        # Check if already downloaded
        if output_path.exists() and not overwrite:
            logger.info(f"Already downloaded: {filename}")
            return output_path
        
        # SEC EDGAR filing URL structure:
        # https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{primary_document}
        # The accession number acts as a folder name
        if primary_doc:
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary_doc}"
        else:
            # Fallback: try without primary document (some older filings)
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}.txt"
        
        self._rate_limit()
        try:
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8', errors='ignore') as f:
                f.write(response.text)
            
            logger.info(f"Downloaded: {ticker} {form_type} {filing_date}")
            
            # Track download
            if ticker not in self.download_log:
                self.download_log[ticker] = []
            self.download_log[ticker].append({
                'form_type': form_type,
                'filing_date': filing_date,
                'filename': filename,
                'downloaded_at': datetime.now().isoformat()
            })
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error downloading {ticker} {form_type} {filing_date}: {e}")
            return None
    
    def download_ticker(
        self,
        ticker: str,
        form_types: List[str] = ["10-Q", "10-K"],
        count: int = 8,
        overwrite: bool = False
    ) -> List[Path]:
        """
        Download filings for a single ticker.
        
        Args:
            ticker: Stock ticker
            form_types: List of form types to download
            count: Maximum number of filings
            overwrite: Whether to overwrite existing files
        
        Returns:
            List of paths to downloaded files
        """
        logger.info(f"Processing ticker: {ticker}")
        
        # Get CIK
        cik = self.get_cik(ticker)
        if not cik:
            logger.error(f"Cannot download {ticker} - CIK not found")
            return []
        
        # Get filings list
        filings = self.get_filings_list(cik, form_types, count)
        if not filings:
            logger.warning(f"No filings found for {ticker}")
            return []
        
        # Download each filing
        downloaded = []
        for filing_info in filings:
            path = self.download_filing(ticker, filing_info, overwrite)
            if path:
                downloaded.append(path)
        
        logger.info(f"Downloaded {len(downloaded)}/{len(filings)} filings for {ticker}")
        return downloaded
    
    def download_batch(
        self,
        tickers: List[str],
        form_types: List[str] = ["10-Q", "10-K"],
        count: int = 8,
        overwrite: bool = False
    ) -> Dict[str, List[Path]]:
        """
        Download filings for multiple tickers.
        
        Args:
            tickers: List of stock tickers
            form_types: List of form types to download
            count: Maximum number of filings per ticker
            overwrite: Whether to overwrite existing files
        
        Returns:
            Dict of ticker -> list of downloaded file paths
        """
        results = {}
        
        logger.info(f"Starting batch download for {len(tickers)} tickers")
        
        for i, ticker in enumerate(tickers, 1):
            logger.info(f"Progress: {i}/{len(tickers)} - {ticker}")
            
            downloaded = self.download_ticker(ticker, form_types, count, overwrite)
            results[ticker] = downloaded
            
            # Small delay between tickers to be polite
            if i < len(tickers):
                time.sleep(0.5)
        
        logger.info(f"Batch download complete: {sum(len(v) for v in results.values())} files")
        return results
    
    def save_download_log(self, output_path: Path = Path("download_log.json")):
        """Save download log to JSON"""
        with open(output_path, 'w') as f:
            json.dump(self.download_log, f, indent=2)
        logger.info(f"Download log saved to {output_path}")


def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download SEC filings automatically'
    )
    parser.add_argument(
        '--tickers',
        type=str,
        required=True,
        help='Comma-separated list of tickers (e.g., CVAC,RYTM,IMMP)'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=8,
        help='Number of filings per ticker (default: 8)'
    )
    parser.add_argument(
        '--forms',
        type=str,
        default='10-Q,10-K',
        help='Comma-separated form types (default: 10-Q,10-K)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='filings',
        help='Output directory (default: filings)'
    )
    parser.add_argument(
        '--user-agent',
        type=str,
        default=USER_AGENT,
        help='User-Agent string (REQUIRED by SEC - include your contact info)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing files'
    )
    
    args = parser.parse_args()
    
    # Validate user agent
    if args.user_agent == _DEFAULT_USER_AGENT_PLACEHOLDER:
        print("\n⚠️  WARNING: Please set a proper User-Agent with your contact information!")
        print("   SEC requires this for rate limiting and to contact you if needed.")
        print("   Set environment variable: export SEC_USER_AGENT='YourName/1.0 (your.email@example.com)'")
        print("   Or use: --user-agent 'YourName/1.0 (your.email@example.com)'")
        print("\n   Proceeding with placeholder, but you MUST update this for production.\n")
    
    # Parse inputs
    tickers = [t.strip().upper() for t in args.tickers.split(',')]
    form_types = [f.strip() for f in args.forms.split(',')]
    output_dir = Path(args.output_dir)
    
    # Create downloader
    downloader = SECDownloader(output_dir, args.user_agent)
    
    # Download
    print("\n" + "="*80)
    print("SEC FILING DOWNLOADER")
    print("="*80)
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Forms: {', '.join(form_types)}")
    print(f"Count per ticker: {args.count}")
    print(f"Output: {output_dir}")
    print("="*80 + "\n")
    
    results = downloader.download_batch(
        tickers,
        form_types,
        args.count,
        args.overwrite
    )
    
    # Save log
    downloader.save_download_log(output_dir / "download_log.json")
    
    # Summary
    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE")
    print("="*80)
    for ticker, paths in results.items():
        print(f"{ticker:6s}: {len(paths)} files")
    print(f"\nTotal: {sum(len(v) for v in results.values())} files downloaded")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Next steps
    print("\n💡 Next Steps:")
    print(f"   1. Extract CFO data:")
    print(f"      python cfo_extractor.py --filings-dir {output_dir} --as-of-date 2024-12-31")
    print(f"   2. Integrate with Module 2")
    print()


if __name__ == "__main__":
    main()
