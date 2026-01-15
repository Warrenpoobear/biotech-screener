#!/usr/bin/env python3
"""
download_etf_holdings.py - Download Complete ETF Holdings

Fetches ALL current holdings from XBI, IBB, NBI including small caps.

Usage:
    python download_etf_holdings.py --output etf_holdings.json --api alphavantage --key YOUR_API_KEY
"""

import json
import requests
from pathlib import Path
from typing import List, Dict, Set
import argparse
import time


# ============================================================================
# METHOD 1: MANUAL CSV IMPORT (MOST ACCURATE)
# ============================================================================

def load_from_csv(csv_path: Path, ticker_column: str = 'Ticker') -> List[str]:
    """
    Load tickers from downloaded ETF holdings CSV.
    
    Args:
        csv_path: Path to downloaded CSV (from SPDR/iShares website)
        ticker_column: Name of ticker column (varies by provider)
    
    Returns:
        List of ticker symbols
    """
    import csv
    
    tickers = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row.get(ticker_column, '').strip()
            if ticker and ticker not in ['', 'Cash', 'CASH']:
                tickers.append(ticker)
    
    return tickers


# ============================================================================
# METHOD 2: ALPHA VANTAGE API (FREE)
# ============================================================================

def get_holdings_alphavantage(symbol: str, api_key: str) -> List[str]:
    """
    Get ETF holdings from Alpha Vantage.
    
    Free tier: 25 requests/day, 5 requests/minute
    Get key: https://www.alphavantage.co/support/#api-key
    """
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "ETF_PROFILE",
        "symbol": symbol,
        "apikey": api_key
    }
    
    print(f"  Fetching {symbol} from Alpha Vantage...")
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print(f"    ❌ Error: HTTP {response.status_code}")
        return []
    
    data = response.json()
    
    # Parse response
    # Note: Alpha Vantage format varies, adjust as needed
    if "Error Message" in data:
        print(f"    ❌ API Error: {data['Error Message']}")
        return []
    
    # Extract holdings (format varies by API response)
    holdings = data.get('holdings', [])
    tickers = [h.get('symbol', '').replace('.', '-') for h in holdings if h.get('symbol')]
    
    print(f"    ✅ Found {len(tickers)} holdings")
    return tickers


# ============================================================================
# METHOD 3: FINANCIAL MODELING PREP (FREE TIER)
# ============================================================================

def get_holdings_fmp(symbol: str, api_key: str) -> List[str]:
    """
    Get ETF holdings from Financial Modeling Prep.
    
    Free tier: 250 requests/day
    Get key: https://financialmodelingprep.com/developer/docs/
    """
    url = f"https://financialmodelingprep.com/api/v3/etf-holder/{symbol}"
    params = {"apikey": api_key}
    
    print(f"  Fetching {symbol} from Financial Modeling Prep...")
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print(f"    ❌ Error: HTTP {response.status_code}")
        return []
    
    data = response.json()
    
    if not data:
        print(f"    ❌ No data returned")
        return []
    
    # FMP returns list of holdings
    tickers = [item.get('asset', '').replace('.', '-') for item in data if item.get('asset')]
    
    print(f"    ✅ Found {len(tickers)} holdings")
    return tickers


# ============================================================================
# METHOD 4: WEB SCRAPING (FALLBACK)
# ============================================================================

def scrape_xbi_holdings() -> List[str]:
    """
    Scrape XBI holdings from SPDR website.
    
    WARNING: Web scraping is fragile. Prefer CSV download or API.
    """
    print("  ⚠️  Web scraping not implemented (use CSV download instead)")
    print("     Download from: https://www.ssga.com/us/en/individual/etfs/funds/xbi")
    return []


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def download_all_etf_holdings(
    method: str = 'csv',
    api_key: str = None,
    csv_dir: Path = None
) -> Dict[str, List[str]]:
    """
    Download complete holdings for XBI, IBB, NBI.
    
    Args:
        method: 'csv', 'alphavantage', 'fmp'
        api_key: API key (if using API method)
        csv_dir: Directory with downloaded CSVs (if using CSV method)
    
    Returns:
        {
            'xbi': ['VRTX', 'GILD', ...],
            'ibb': ['VRTX', 'GILD', ...],
            'nbi': ['VRTX', 'GILD', ...]
        }
    """
    
    holdings = {}
    
    print("\n" + "="*80)
    print("DOWNLOADING ETF HOLDINGS")
    print("="*80)
    
    if method == 'csv':
        print("\n📥 Loading from CSV files...")
        
        if not csv_dir or not csv_dir.exists():
            print(f"❌ CSV directory not found: {csv_dir}")
            print("\nTo use CSV method:")
            print("1. Download holdings CSVs from ETF websites:")
            print("   • XBI: https://www.ssga.com/us/en/individual/etfs/funds/xbi")
            print("   • IBB: https://www.ishares.com/us/products/239699/")
            print("2. Save as: XBI_holdings.csv, IBB_holdings.csv")
            print("3. Run: python download_etf_holdings.py --method csv --csv-dir /path/to/csvs")
            return {}
        
        # Load XBI
        xbi_csv = csv_dir / "XBI_holdings.csv"
        if xbi_csv.exists():
            holdings['xbi'] = load_from_csv(xbi_csv, ticker_column='Ticker')  # Adjust column name
            print(f"  ✅ XBI: {len(holdings['xbi'])} holdings")
        else:
            print(f"  ❌ XBI CSV not found: {xbi_csv}")
        
        # Load IBB
        ibb_csv = csv_dir / "IBB_holdings.csv"
        if ibb_csv.exists():
            holdings['ibb'] = load_from_csv(ibb_csv, ticker_column='Ticker')  # Adjust column name
            print(f"  ✅ IBB: {len(holdings['ibb'])} holdings")
        else:
            print(f"  ❌ IBB CSV not found: {ibb_csv}")
        
        # Load NBI
        nbi_csv = csv_dir / "NBI_holdings.csv"
        if nbi_csv.exists():
            holdings['nbi'] = load_from_csv(nbi_csv, ticker_column='Symbol')  # NBI uses "Symbol"
            print(f"  ✅ NBI: {len(holdings['nbi'])} holdings")
        else:
            print(f"  ⚠️  NBI CSV not found: {nbi_csv}")
            print(f"     Download from: https://indexes.nasdaqomx.com/Index/Weighting/NBI")
    
    elif method == 'alphavantage':
        if not api_key:
            print("❌ Alpha Vantage requires API key")
            print("   Get free key: https://www.alphavantage.co/support/#api-key")
            return {}
        
        print("\n📥 Fetching from Alpha Vantage API...")
        holdings['xbi'] = get_holdings_alphavantage('XBI', api_key)
        time.sleep(12)  # Rate limit: 5 calls/minute
        holdings['ibb'] = get_holdings_alphavantage('IBB', api_key)
    
    elif method == 'fmp':
        if not api_key:
            print("❌ Financial Modeling Prep requires API key")
            print("   Get free key: https://financialmodelingprep.com/developer/docs/")
            return {}
        
        print("\n📥 Fetching from Financial Modeling Prep API...")
        holdings['xbi'] = get_holdings_fmp('XBI', api_key)
        holdings['ibb'] = get_holdings_fmp('IBB', api_key)
    
    else:
        print(f"❌ Unknown method: {method}")
        return {}
    
    # Compute unique constituents
    all_tickers = set()
    for etf_holdings in holdings.values():
        all_tickers.update(etf_holdings)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"XBI holdings: {len(holdings.get('xbi', []))}")
    print(f"IBB holdings: {len(holdings.get('ibb', []))}")
    print(f"NBI holdings: {len(holdings.get('nbi', []))}")
    print(f"Total unique: {len(all_tickers)}")
    print("="*80 + "\n")
    
    return holdings


def main():
    parser = argparse.ArgumentParser(
        description="Download complete ETF holdings (including small caps)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Method 1: CSV (Most Accurate - RECOMMENDED)
  # 1. Download CSVs from ETF websites
  # 2. Save to etf_csvs/ directory
  python download_etf_holdings.py --method csv --csv-dir etf_csvs/ --output etf_holdings.json
  
  # Method 2: Alpha Vantage API (Free)
  python download_etf_holdings.py --method alphavantage --api-key YOUR_KEY --output etf_holdings.json
  
  # Method 3: Financial Modeling Prep API
  python download_etf_holdings.py --method fmp --api-key YOUR_KEY --output etf_holdings.json

Recommended approach:
  1. Use CSV method for accuracy
  2. Download holdings from:
     • XBI: https://www.ssga.com/us/en/individual/etfs/funds/xbi
     • IBB: https://www.ishares.com/us/products/239699/
  3. Import into your universe
        """
    )
    
    parser.add_argument(
        '--method',
        choices=['csv', 'alphavantage', 'fmp'],
        default='csv',
        help='Download method'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        help='API key (for alphavantage or fmp methods)'
    )
    
    parser.add_argument(
        '--csv-dir',
        type=Path,
        help='Directory with downloaded CSV files (for csv method)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output JSON file'
    )
    
    args = parser.parse_args()
    
    # Download holdings
    holdings = download_all_etf_holdings(
        method=args.method,
        api_key=args.api_key,
        csv_dir=args.csv_dir
    )
    
    if not holdings:
        print("❌ No holdings downloaded")
        return 1
    
    # Save to JSON
    with open(args.output, 'w') as f:
        json.dump(holdings, f, indent=2, sort_keys=True)
        f.write('\n')
    
    print(f"✅ Saved holdings to: {args.output}")
    
    # Print instructions for next steps
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Review downloaded holdings in", args.output)
    print("2. Update check_etf_coverage.py with complete lists:")
    print("   python update_etf_constituents.py --input", args.output)
    print("3. Expand your universe:")
    print("   python expand_universe_to_etfs.py --universe production_data/universe.json")
    print("="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
