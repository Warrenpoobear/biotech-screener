"""
Clean biotech universe by removing ineligible securities.

Identifies and removes:
- Currency tickers (USD, EUR, etc.)
- ETFs (XBI, IBB, LABU, LABD, etc.)
- Indices
- Leveraged/inverse products
- Non-biotech companies

Usage:
    python clean_universe.py --input outputs/rankings_FIXED.csv --output data/universe_clean.csv --quick
"""

import pandas as pd
import argparse
from pathlib import Path
import sys

# Known ineligible securities to ALWAYS remove
INELIGIBLE_TICKERS = {
    # Currencies
    'USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'CNY', 'INR',
    
    # Major Biotech ETFs
    'XBI', 'IBB', 'BBH', 'FBT', 'PBE', 'SBIO',
    
    # Healthcare ETFs
    'XLV', 'VHT', 'IHI', 'IHF', 'XHS',
    
    # Leveraged/Inverse ETFs
    'LABU', 'LABD', 'CURE',
    
    # ARK ETFs
    'ARKG', 'ARKK',
    
    # Indices
    'NBI', '^NBI',
    
    # Other
    'POWL',
}

def quick_clean(df, ticker_col='ticker'):
    """Quick removal of known ineligible tickers."""
    
    initial_count = len(df)
    
    # Remove known ineligible tickers
    df_clean = df[~df[ticker_col].isin(INELIGIBLE_TICKERS)].copy()
    
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

def main():
    parser = argparse.ArgumentParser(description='Clean biotech universe')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='data/universe_clean.csv')
    parser.add_argument('--ticker-col', default='ticker')
    parser.add_argument('--quick', action='store_true')
    
    args = parser.parse_args()
    
    # Load universe
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    print(f"Loading universe from: {args.input}")
    df = pd.read_csv(input_path)
    
    if args.ticker_col not in df.columns:
        print(f"Error: Column '{args.ticker_col}' not found")
        print(f"Available columns: {', '.join(df.columns)}")
        sys.exit(1)
    
    print(f"Loaded {len(df)} tickers")
    
    # Clean
    df_clean = quick_clean(df, args.ticker_col)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    
    print(f"\nCleaned universe saved to: {args.output}")
    print(f"Removed: {len(df) - len(df_clean)} tickers")
    print(f"Remaining: {len(df_clean)} tickers")

if __name__ == '__main__':
    main()
