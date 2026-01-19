"""
Fetch historical price data for tickers in screening history.

Downloads price data from Yahoo Finance to enable forward return calculations.
"""

import json
import csv
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import sys

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("WARNING: yfinance not installed")
    print("Install with: pip install yfinance")


class PriceDataFetcher:
    """Fetch historical price data for optimization."""

    def __init__(self, checkpoint_dir='checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir)

    def get_all_tickers(self):
        """Extract all tickers from checkpoint files."""
        tickers = set()

        for filepath in self.checkpoint_dir.glob('module_5_*.json'):
            try:
                with open(filepath) as f:
                    data = json.load(f)

                # Extract tickers
                securities = data.get('ranked_securities', data.get('results', []))
                for security in securities:
                    ticker = security.get('ticker')
                    if ticker:
                        tickers.add(ticker)
            except Exception as e:
                print(f"Warning: Error reading {filepath.name}: {e}")
                continue

        return sorted(tickers)

    def get_date_range(self):
        """Get date range from checkpoint files."""
        dates = []

        for filepath in self.checkpoint_dir.glob('module_5_*.json'):
            try:
                date_str = filepath.stem.replace('module_5_', '')
                date = datetime.strptime(date_str, '%Y-%m-%d')
                dates.append(date)
            except ValueError:
                continue

        if not dates:
            return None, None

        return min(dates), max(dates)

    def fetch_yahoo_prices(self, tickers, start_date, end_date, output_file='production_data/price_history.csv'):
        """
        Fetch historical prices from Yahoo Finance.

        Parameters:
        - tickers: List of ticker symbols
        - start_date: Start date (datetime)
        - end_date: End date (datetime)
        - output_file: Where to save the CSV
        """
        if not YFINANCE_AVAILABLE:
            print("ERROR: yfinance package required")
            print("Install with: pip install yfinance")
            return

        print(f"\nFetching price data for {len(tickers)} tickers...")
        print(f"Date range: {start_date.date()} to {end_date.date()}")

        # Extend date range by 30 days on each end for forward returns
        fetch_start = start_date - timedelta(days=30)
        fetch_end = end_date + timedelta(days=30)

        price_data = []
        failed_tickers = []

        for i, ticker in enumerate(tickers, 1):
            print(f"  [{i}/{len(tickers)}] Fetching {ticker}...", end='', flush=True)

            try:
                # Download data
                # Set auto_adjust=False to use traditional column names ('Adj Close')
                # In yfinance v1.0+, auto_adjust defaults to True which changes column names
                df = yf.download(ticker, start=fetch_start, end=fetch_end, progress=False, auto_adjust=False)

                if df.empty:
                    print(" ❌ No data")
                    failed_tickers.append(ticker)
                    continue

                # Determine which column to use for adjusted close price
                # - yfinance < 1.0 or auto_adjust=False: 'Adj Close'
                # - yfinance >= 1.0 with auto_adjust=True: 'Close' (already adjusted)
                if 'Adj Close' in df.columns:
                    close_col = 'Adj Close'
                elif 'Close' in df.columns:
                    close_col = 'Close'
                else:
                    print(" ❌ No price column found")
                    failed_tickers.append(ticker)
                    continue

                # Extract data
                for date, row in df.iterrows():
                    price_data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'ticker': ticker,
                        'close': float(row[close_col])
                    })

                print(f" ✓ {len(df)} days")

            except Exception as e:
                print(f" ❌ Error: {e}")
                failed_tickers.append(ticker)
                continue

        # Save to CSV
        if price_data:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', newline='') as f:
                fieldnames = ['date', 'ticker', 'close']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(price_data)

            print(f"\n✓ Saved {len(price_data)} price observations to {output_file}")
            print(f"  Success: {len(tickers) - len(failed_tickers)}/{len(tickers)} tickers")

            if failed_tickers:
                print(f"\n  Failed tickers: {', '.join(failed_tickers[:10])}")
                if len(failed_tickers) > 10:
                    print(f"  ... and {len(failed_tickers) - 10} more")
        else:
            print("\n❌ No price data fetched")

    def fetch_from_yahoo_cache(self, output_file='production_data/price_history.csv'):
        """
        Check for existing Yahoo Finance cache and convert to CSV.

        Some systems cache Yahoo data in JSON format. This converts it.
        """
        cache_files = [
            'production_data/yahoo_cache.json',
            'production_data/yahoo_prices.json',
            'data/yahoo_cache.json'
        ]

        for cache_file in cache_files:
            cache_path = Path(cache_file)
            if cache_path.exists():
                print(f"Found Yahoo cache: {cache_file}")

                try:
                    with open(cache_path) as f:
                        cache_data = json.load(f)

                    # Convert to CSV format
                    price_data = []
                    for ticker, ticker_data in cache_data.items():
                        if isinstance(ticker_data, dict):
                            for date_str, price in ticker_data.items():
                                price_data.append({
                                    'date': date_str,
                                    'ticker': ticker,
                                    'close': float(price)
                                })

                    if price_data:
                        output_path = Path(output_file)
                        output_path.parent.mkdir(parents=True, exist_ok=True)

                        with open(output_path, 'w', newline='') as f:
                            fieldnames = ['date', 'ticker', 'close']
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(price_data)

                        print(f"✓ Converted {len(price_data)} price observations to {output_file}")
                        return True

                except Exception as e:
                    print(f"Error reading cache: {e}")
                    continue

        return False


def main():
    """Main price fetching workflow."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Fetch historical price data for optimization'
    )
    parser.add_argument(
        '--checkpoint-dir',
        default='checkpoints',
        help='Directory containing module_5_*.json checkpoint files'
    )
    parser.add_argument(
        '--output',
        default='production_data/price_history.csv',
        help='Output CSV file'
    )
    parser.add_argument(
        '--check-cache',
        action='store_true',
        help='Check for existing Yahoo Finance cache first'
    )

    args = parser.parse_args()

    print("="*60)
    print("HISTORICAL PRICE DATA FETCHER")
    print("="*60)

    fetcher = PriceDataFetcher(checkpoint_dir=args.checkpoint_dir)

    # Check for existing cache first
    if args.check_cache:
        print("\nChecking for existing Yahoo Finance cache...")
        if fetcher.fetch_from_yahoo_cache(output_file=args.output):
            print("\n✓ Price data ready from cache")
            sys.exit(0)

    # Get tickers from checkpoints
    print("\nScanning checkpoints for tickers...")
    tickers = fetcher.get_all_tickers()

    if not tickers:
        print("ERROR: No tickers found in checkpoint files")
        sys.exit(1)

    print(f"Found {len(tickers)} unique tickers")

    # Get date range
    start_date, end_date = fetcher.get_date_range()

    if not start_date:
        print("ERROR: Could not determine date range from checkpoints")
        sys.exit(1)

    # Fetch prices
    fetcher.fetch_yahoo_prices(tickers, start_date, end_date, output_file=args.output)

    print("\n" + "="*60)
    print("PRICE DATA READY")
    print("="*60)
    print("\nNext step: Extract training data")
    print("  python optimization/extract_historical_data.py")


if __name__ == '__main__':
    main()
