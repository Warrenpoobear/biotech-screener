#!/usr/bin/env python3
"""
Simple price fetcher using Yahoo Finance CSV download API.

No external dependencies required - uses only standard library.
"""

import csv
import json
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from pathlib import Path
from http.cookiejar import CookieJar


class YahooFinanceSession:
    """Handle Yahoo Finance authentication with cookies and crumb."""

    def __init__(self):
        self.cookie_jar = CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.cookie_jar)
        )
        self.crumb = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }

    def _get_crumb(self):
        """Fetch crumb token from Yahoo Finance."""
        if self.crumb:
            return self.crumb

        # First, visit the main page to get cookies
        try:
            request = urllib.request.Request(
                'https://finance.yahoo.com',
                headers=self.headers
            )
            self.opener.open(request, timeout=30)
        except Exception:
            pass

        # Then fetch the crumb
        try:
            request = urllib.request.Request(
                'https://query1.finance.yahoo.com/v1/test/getcrumb',
                headers=self.headers
            )
            with self.opener.open(request, timeout=30) as response:
                self.crumb = response.read().decode('utf-8').strip()
                return self.crumb
        except Exception:
            return None

    def fetch_prices(self, ticker, start_date, end_date):
        """
        Fetch historical prices from Yahoo Finance.

        Returns list of (date, close_price) tuples.
        """
        # Get crumb if we don't have one
        crumb = self._get_crumb()

        # Convert dates to Unix timestamps
        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())

        # Build URL with crumb
        url = (
            f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
            f"?period1={start_ts}&period2={end_ts}&interval=1d&events=history"
        )
        if crumb:
            url += f"&crumb={crumb}"

        try:
            request = urllib.request.Request(url, headers=self.headers)
            with self.opener.open(request, timeout=30) as response:
                content = response.read().decode('utf-8')

            # Parse CSV
            lines = content.strip().split('\n')
            if len(lines) < 2:
                return []

            prices = []
            for line in lines[1:]:  # Skip header
                parts = line.split(',')
                if len(parts) >= 6:
                    date_str = parts[0]
                    adj_close = parts[5]  # Adj Close column
                    try:
                        price = float(adj_close)
                        if price > 0:
                            prices.append((date_str, price))
                    except (ValueError, IndexError):
                        continue

            return prices

        except urllib.error.HTTPError as e:
            if e.code == 401:
                self.crumb = None
                return []
            if e.code == 404:
                return []
            raise
        except Exception:
            return []


def get_tickers_from_checkpoints(checkpoint_dir='checkpoints'):
    """Extract all tickers from checkpoint files."""
    tickers = set()
    checkpoint_path = Path(checkpoint_dir)

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        return []

    for filepath in checkpoint_path.glob('module_5_*.json'):
        try:
            with open(filepath) as f:
                data = json.load(f)

            # Handle nested structure
            if 'data' in data:
                securities = data['data'].get('ranked_securities', [])
            else:
                securities = data.get('ranked_securities', data.get('results', []))

            for security in securities:
                ticker = security.get('ticker')
                if ticker:
                    tickers.add(ticker)
        except Exception as e:
            print(f"Warning: Error reading {filepath.name}: {e}")

    return sorted(tickers)


def get_date_range_from_checkpoints(checkpoint_dir='checkpoints'):
    """Get date range from checkpoint files."""
    dates = []
    checkpoint_path = Path(checkpoint_dir)

    for filepath in checkpoint_path.glob('module_5_*.json'):
        try:
            date_str = filepath.stem.replace('module_5_', '')
            date = datetime.strptime(date_str, '%Y-%m-%d')
            dates.append(date)
        except ValueError:
            continue

    if not dates:
        return None, None

    return min(dates), max(dates)


def fetch_all_prices(tickers, start_date, end_date, output_file, delay=0.5):
    """
    Fetch prices for all tickers and save to CSV.

    Args:
        tickers: List of ticker symbols
        start_date: Start datetime
        end_date: End datetime
        output_file: Output CSV path
        delay: Delay between requests (seconds)
    """
    print(f"\nFetching prices for {len(tickers)} tickers...")
    print(f"Date range: {start_date.date()} to {end_date.date()}")

    # Extend date range for forward returns
    fetch_start = start_date - timedelta(days=7)
    fetch_end = end_date + timedelta(days=35)

    # Create session for authenticated requests
    session = YahooFinanceSession()
    print("Initializing Yahoo Finance session...")

    all_prices = []
    success_count = 0
    failed_tickers = []

    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i}/{len(tickers)}] {ticker}...", end='', flush=True)

        try:
            prices = session.fetch_prices(ticker, fetch_start, fetch_end)

            if prices:
                for date_str, close in prices:
                    all_prices.append({
                        'date': date_str,
                        'ticker': ticker,
                        'close': close
                    })
                print(f" ✓ {len(prices)} days")
                success_count += 1
            else:
                print(" ✗ No data")
                failed_tickers.append(ticker)

        except Exception as e:
            print(f" ✗ Error: {e}")
            failed_tickers.append(ticker)

        # Rate limiting
        if delay > 0 and i < len(tickers):
            time.sleep(delay)

    # Save to CSV
    if all_prices:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['date', 'ticker', 'close'])
            writer.writeheader()
            writer.writerows(all_prices)

        print(f"\n✓ Saved {len(all_prices)} price observations to {output_file}")
        print(f"  Success: {success_count}/{len(tickers)} tickers")

        if failed_tickers:
            print(f"\n  Failed tickers ({len(failed_tickers)}):")
            for t in failed_tickers[:15]:
                print(f"    - {t}")
            if len(failed_tickers) > 15:
                print(f"    ... and {len(failed_tickers) - 15} more")
    else:
        print("\n✗ No price data fetched")
        return False

    return True


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Fetch historical prices from Yahoo Finance (no dependencies)'
    )
    parser.add_argument(
        '--checkpoint-dir',
        default='checkpoints',
        help='Directory containing checkpoint files'
    )
    parser.add_argument(
        '--output',
        default='production_data/price_history.csv',
        help='Output CSV file'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.3,
        help='Delay between requests in seconds (default: 0.3)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("YAHOO FINANCE PRICE FETCHER (Simple)")
    print("=" * 60)

    # Get tickers
    print("\nScanning checkpoints for tickers...")
    tickers = get_tickers_from_checkpoints(args.checkpoint_dir)

    if not tickers:
        print("Error: No tickers found")
        sys.exit(1)

    print(f"Found {len(tickers)} unique tickers")

    # Get date range
    start_date, end_date = get_date_range_from_checkpoints(args.checkpoint_dir)

    if not start_date:
        print("Error: Could not determine date range")
        sys.exit(1)

    # Fetch prices
    success = fetch_all_prices(
        tickers,
        start_date,
        end_date,
        args.output,
        delay=args.delay
    )

    if success:
        print("\n" + "=" * 60)
        print("PRICE DATA READY")
        print("=" * 60)
        print("\nNext step: Extract training data with returns")
        print(f"  python -m optimization.extract_historical_data --price-file {args.output}")
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
