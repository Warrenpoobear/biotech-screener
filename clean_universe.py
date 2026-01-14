"""
Clean biotech universe by removing ineligible securities.

Identifies and removes:
- Currency tickers (USD, EUR, etc.)
- ETFs (XBI, IBB, LABU, LABD, etc.)
- Indices
- Leveraged/inverse products
- Non-biotech companies

Usage:
    python clean_universe.py --input outputs/rankings_FIXED.csv --output data/universe_clean.csv --check-all
    python clean_universe.py --input outputs/rankings_FIXED.csv --quick  # Just remove known bad tickers
"""

import pandas as pd
import argparse
from pathlib import Path
import yfinance as yf
from time import sleep
import sys


# Known ineligible securities to ALWAYS remove
INELIGIBLE_TICKERS = {
    # Currencies
    'USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'CNY', 'INR',
    
    # Major Biotech ETFs
    'XBI',   # SPDR S&P Biotech ETF
    'IBB',   # iShares Biotechnology ETF  
    'BBH',   # VanEck Biotech ETF
    'FBT',   # First Trust Biotech ETF
    'PBE',   # Invesco Dynamic Biotech ETF
    'SBIO',  # ALPS Medical Breakthroughs ETF
    
    # Healthcare ETFs
    'XLV',   # Health Care Select Sector SPDR
    'VHT',   # Vanguard Health Care ETF
    'IHI',   # iShares Medical Devices ETF
    'IHF',   # iShares U.S. Healthcare Providers ETF
    'XHS',   # SPDR S&P Health Care Services ETF
    
    # Leveraged/Inverse ETFs
    'LABU',  # Direxion 3X Bull Biotech
    'LABD',  # Direxion 3X Bear Biotech
    'CURE',  # Direxion 3X Bull Healthcare
    
    # ARK ETFs
    'ARKG',  # ARK Genomic Revolution
    'ARKK',  # ARK Innovation (not pure biotech)
    
    # Indices (not tradeable)
    'NBI',   # NASDAQ Biotech Index
    '^NBI',  # NASDAQ Biotech Index (alternate)
    
    # PowerShares/Invesco duplicates
    'POWL',  # Powell Industries (industrial, not biotech)
}

# Suspicious patterns in ticker names
SUSPICIOUS_PATTERNS = [
    'POWER',  # PowerShares products
    'PRO',    # ProShares products  
    '^',      # Index symbols
    '=',      # Index symbols
]

# Known non-biotech sectors
NON_BIOTECH_SECTORS = {
    'Financial Services',
    'Real Estate', 
    'Energy',
    'Utilities',
    'Basic Materials',
    'Communication Services',
    'Consumer Cyclical',
    'Consumer Defensive',
    'Industrials',
    'Technology',  # Unless biotech-focused
}


def quick_clean(df: pd.DataFrame, ticker_col: str = 'ticker') -> pd.DataFrame:
    """
    Quick removal of known ineligible tickers without API calls.
    """
    
    initial_count = len(df)
    
    # Remove known ineligible tickers
    df_clean = df[~df[ticker_col].isin(INELIGIBLE_TICKERS)].copy()
    
    # Remove tickers with suspicious patterns
    for pattern in SUSPICIOUS_PATTERNS:
        df_clean = df_clean[~df_clean[ticker_col].str.contains(pattern, case=False, na=False)]
    
    removed_count = initial_count - len(df_clean)
    
    print(f"\n{'='*60}")
    print("QUICK CLEAN RESULTS")
    print(f"{'='*60}")
    print(f"Initial universe: {initial_count} tickers")
    print(f"Removed: {removed_count} tickers")
    print(f"Clean universe: {len(df_clean)} tickers")
    
    if removed_count > 0:
        removed = set(df[ticker_col]) - set(df_clean[ticker_col])
        print(f"\nRemoved tickers:")
        for ticker in sorted(removed):
            print(f"  - {ticker}")
    
    return df_clean


def check_ticker_with_yfinance(ticker: str) -> dict:
    """Check ticker validity using Yahoo Finance."""
    
    result = {
        'ticker': ticker,
        'valid': True,
        'reason': None,
        'sector': None,
        'industry': None,
        'name': None,
        'quote_type': None
    }
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        result['name'] = info.get('longName', info.get('shortName', ''))
        result['sector'] = info.get('sector', '')
        result['industry'] = info.get('industry', '')
        result['quote_type'] = info.get('quoteType', '')
        
        # No company name = likely delisted/invalid
        if not result['name']:
            result['valid'] = False
            result['reason'] = 'No data (delisted/invalid)'
            return result
        
        # ETF check
        if result['quote_type'] == 'ETF':
            result['valid'] = False
            result['reason'] = 'ETF (not individual stock)'
            return result
        
        # Sector check
        if result['sector'] in NON_BIOTECH_SECTORS:
            result['valid'] = False
            result['reason'] = f"Non-biotech sector: {result['sector']}"
            return result
        
        # Healthcare check (should be Healthcare or Biotechnology)
        if result['sector'] and 'Healthcare' not in result['sector']:
            result['valid'] = False
            result['reason'] = f"Non-healthcare sector: {result['sector']}"
            return result
            
    except Exception as e:
        result['valid'] = False
        result['reason'] = f"Error fetching data: {str(e)}"
    
    return result


def deep_clean(df: pd.DataFrame, ticker_col: str = 'ticker') -> pd.DataFrame:
    """
    Deep clean using Yahoo Finance API to verify each ticker.
    WARNING: Slow! Uses API rate limits.
    """
    
    print(f"\n{'='*60}")
    print("DEEP CLEAN (checking all tickers with Yahoo Finance)")
    print(f"{'='*60}")
    print(f"Checking {len(df)} tickers... this will take a few minutes")
    print()
    
    valid_tickers = []
    invalid_tickers = []
    
    for i, ticker in enumerate(df[ticker_col], 1):
        print(f"[{i}/{len(df)}] Checking {ticker}...", end=' ')
        
        result = check_ticker_with_yfinance(ticker)
        
        if result['valid']:
            valid_tickers.append(ticker)
            print(f"✅ {result['name']}")
        else:
            invalid_tickers.append((ticker, result['reason']))
            print(f"❌ {result['reason']}")
        
        # Rate limiting
        if i % 10 == 0:
            sleep(1)
    
    # Create clean dataframe
    df_clean = df[df[ticker_col].isin(valid_tickers)].copy()
    
    print(f"\n{'='*60}")
    print("DEEP CLEAN RESULTS")
    print(f"{'='*60}")
    print(f"Initial universe: {len(df)} tickers")
    print(f"Valid: {len(valid_tickers)} tickers")
    print(f"Invalid: {len(invalid_tickers)} tickers")
    
    if invalid_tickers:
        print(f"\nInvalid tickers removed:")
        for ticker, reason in sorted(invalid_tickers):
            print(f"  - {ticker}: {reason}")
    
    return df_clean


def main():
    parser = argparse.ArgumentParser(
        description='Clean biotech universe by removing ineligible securities'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Input CSV file (e.g., outputs/rankings_FIXED.csv)'
    )
    parser.add_argument(
        '--output',
        default='data/universe_clean.csv',
        help='Output CSV file for cleaned universe'
    )
    parser.add_argument(
        '--ticker-col',
        default='ticker',
        help='Name of ticker column (default: ticker)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick clean (just remove known bad tickers, no API calls)'
    )
    parser.add_argument(
        '--check-all',
        action='store_true',
        help='Deep clean (check all tickers with Yahoo Finance API - SLOW!)'
    )
    
    args = parser.parse_args()
    
    # Load universe
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {args.input}")
        sys.exit(1)
    
    print(f"Loading universe from: {args.input}")
    df = pd.read_csv(input_path)
    
    if args.ticker_col not in df.columns:
        print(f"❌ Error: Column '{args.ticker_col}' not found in CSV")
        print(f"   Available columns: {', '.join(df.columns)}")
        sys.exit(1)
    
    print(f"Loaded {len(df)} tickers")
    
    # Clean based on mode
    if args.check_all:
        df_clean = deep_clean(df, args.ticker_col)
    else:
        # Default: quick clean (or if --quick specified)
        df_clean = quick_clean(df, args.ticker_col)
    
    # Save cleaned universe
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_clean.to_csv(output_path, index=False)
    
    print(f"\n✅ Cleaned universe saved to: {args.output}")
    print(f"   Removed: {len(df) - len(df_clean)} tickers")
    print(f"   Remaining: {len(df_clean)} tickers")
    
    # Summary
    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print(f"{'='*60}")
    print(f"1. Review cleaned universe: {args.output}")
    print(f"2. Use cleaned universe for future snapshots:")
    print(f"   python historical_fetchers/reconstruct_snapshot.py \\")
    print(f"     --date YYYY-MM-DD \\")
    print(f"     --tickers-file {args.output} \\")
    print(f"     --generate-rankings")


if __name__ == '__main__':
    main()
