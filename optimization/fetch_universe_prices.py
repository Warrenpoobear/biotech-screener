#!/usr/bin/env python3
"""
Fetch historical price data for the full biotech universe.

Downloads adjusted close prices from Yahoo Finance for all eligible securities
in production_data/universe.json.

No external dependencies required - uses only standard library.

Usage:
    python optimization/fetch_universe_prices.py --start-date 2020-01-01
"""

import csv
import json
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from pathlib import Path


def load_universe_tickers(universe_file='production_data/universe.json'):
    """Load tickers from universe file, excluding benchmark placeholders."""
    with open(universe_file) as f:
        data = json.load(f)

    tickers = []
    for item in data:
        ticker = item.get('ticker', '')
        # Skip benchmark placeholders and invalid tickers
        if ticker and not ticker.startswith('_') and not ticker.endswith('_'):
            tickers.append(ticker)

    return sorted(set(tickers))


def fetch_yahoo_csv(ticker, start_date, end_date):
    """
    Fetch historical prices from Yahoo Finance CSV API.

    Returns list of (date, close_price) tuples.
    """
    # Convert dates to Unix timestamps
    start_ts = int(start_date.timestamp())
    end_ts = int(end_date.timestamp())

    # Yahoo Finance CSV download URL
    url = (
        f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
        f"?period1={start_ts}&period2={end_ts}&interval=1d&events=history"
    )

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=30) as response:
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
                    if price > 0:  # Skip invalid prices
                        prices.append((date_str, price))
                except (ValueError, IndexError):
                    continue

        return prices

    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []  # Ticker not found
        raise
    except Exception:
        return []


def fetch_all_prices(tickers, start_date, end_date, output_file, delay=0.3):
    """
    Fetch prices for all tickers and save to CSV.

    Args:
        tickers: List of ticker symbols
        start_date: Start datetime
        end_date: End datetime
        output_file: Output CSV path
        delay: Delay between requests (seconds)
    """
    print(f"\nFetching prices for {len(tickers)} tickers")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Output: {output_file}")
    print()

    all_prices = []
    success_count = 0
    failed_tickers = []
    start_time = time.time()

    for i, ticker in enumerate(tickers, 1):
        # Calculate ETA
        elapsed = time.time() - start_time
        if i > 1:
            avg_time = elapsed / (i - 1)
            remaining = avg_time * (len(tickers) - i + 1)
            eta = f"{int(remaining // 60)}m {int(remaining % 60)}s"
        else:
            eta = "calculating..."

        print(f"[{i:3d}/{len(tickers)}] {ticker:8s} (ETA: {eta})...", end='', flush=True)

        try:
            prices = fetch_yahoo_csv(ticker, start_date, end_date)

            if prices:
                for date_str, close in prices:
                    all_prices.append({
                        'date': date_str,
                        'ticker': ticker,
                        'close': close
                    })
                print(f" {len(prices)} days")
                success_count += 1
            else:
                print(" No data")
                failed_tickers.append(ticker)

        except Exception as e:
            print(f" Error: {str(e)[:40]}")
            failed_tickers.append(ticker)

        # Rate limiting
        if delay > 0 and i < len(tickers):
            time.sleep(delay)

    # Sort by date then ticker
    all_prices.sort(key=lambda x: (x['date'], x['ticker']))

    # Save to CSV
    if all_prices:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['date', 'ticker', 'close'])
            writer.writeheader()
            writer.writerows(all_prices)

    # Summary
    elapsed = time.time() - start_time
    print()
    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print()
    print(f"Total time: {int(elapsed // 60)}m {int(elapsed % 60)}s")
    print(f"Success: {success_count}/{len(tickers)} tickers ({100*success_count/len(tickers):.1f}%)")
    print(f"Total price observations: {len(all_prices):,}")

    if all_prices:
        print(f"Output saved to: {output_file}")

    if failed_tickers:
        print()
        print(f"Failed tickers ({len(failed_tickers)}):")
        for ticker in failed_tickers[:20]:
            print(f"  - {ticker}")
        if len(failed_tickers) > 20:
            print(f"  ... and {len(failed_tickers) - 20} more")

    return success_count > 0


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Fetch historical prices for biotech universe (no external dependencies)'
    )
    parser.add_argument(
        '--universe',
        default='production_data/universe.json',
        help='Universe JSON file (default: production_data/universe.json)'
    )
    parser.add_argument(
        '--start-date',
        default='2020-01-01',
        help='Start date YYYY-MM-DD (default: 2020-01-01)'
    )
    parser.add_argument(
        '--end-date',
        default=datetime.now().strftime('%Y-%m-%d'),
        help='End date YYYY-MM-DD (default: today)'
    )
    parser.add_argument(
        '--output',
        default='production_data/price_history.csv',
        help='Output CSV file (default: production_data/price_history.csv)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.3,
        help='Delay between requests in seconds (default: 0.3)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("BIOTECH UNIVERSE PRICE FETCHER")
    print("=" * 70)

    # Parse dates
    try:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    except ValueError as e:
        print(f"Error: Invalid date format. Use YYYY-MM-DD. {e}")
        sys.exit(1)

    # Load tickers
    print(f"\nLoading universe from: {args.universe}")
    try:
        tickers = load_universe_tickers(args.universe)
    except FileNotFoundError:
        print(f"Error: Universe file not found: {args.universe}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in universe file: {e}")
        sys.exit(1)

    print(f"Found {len(tickers)} eligible tickers")

    if not tickers:
        print("Error: No tickers found in universe")
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
        print()
        print("Next steps:")
        print("  1. Extract training data:")
        print("     python -m optimization.extract_historical_data")
        print()
        print("  2. Optimize weights:")
        print("     python -m optimization.optimize_weights_scipy")
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
