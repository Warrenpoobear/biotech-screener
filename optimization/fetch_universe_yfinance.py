#!/usr/bin/env python3
"""
Fetch historical price data using yfinance library.

This script uses the yfinance library which handles Yahoo Finance authentication internally.

Usage:
    pip install yfinance
    python optimization/fetch_universe_yfinance.py --start-date 2020-01-01 --end-date 2026-01-16
"""

import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("ERROR: yfinance not installed")
    print("Install with: pip install yfinance")
    sys.exit(1)


def load_universe_tickers(universe_file='production_data/universe.json', include_xbi=True):
    """Load tickers from universe file."""
    with open(universe_file) as f:
        data = json.load(f)

    tickers = []
    for item in data:
        ticker = item.get('ticker', '')
        if ticker and not ticker.startswith('_') and not ticker.endswith('_'):
            tickers.append(ticker)

    if include_xbi:
        tickers.append('XBI')

    return sorted(set(tickers))


def fetch_prices_yfinance(tickers, start_date, end_date, output_file, delay=0.5):
    """Fetch prices using yfinance library."""
    print(f"\nFetching prices for {len(tickers)} tickers")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Output: {output_file}")
    print()

    all_prices = []
    success_count = 0
    failed_tickers = []
    start_time = time.time()

    for i, ticker in enumerate(tickers, 1):
        elapsed = time.time() - start_time
        if i > 1:
            avg_time = elapsed / (i - 1)
            remaining = avg_time * (len(tickers) - i + 1)
            eta = f"{int(remaining // 60)}m {int(remaining % 60)}s"
        else:
            eta = "calculating..."

        print(f"[{i:3d}/{len(tickers)}] {ticker:8s} (ETA: {eta})...", end='', flush=True)

        try:
            # Use yfinance to download data
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False
            )

            if df.empty:
                print(" No data")
                failed_tickers.append(ticker)
                continue

            # Handle column names (yfinance v1.0+ vs older)
            if 'Adj Close' in df.columns:
                close_col = 'Adj Close'
            elif 'Close' in df.columns:
                close_col = 'Close'
            else:
                # Handle MultiIndex columns from batch download
                if hasattr(df.columns, 'get_level_values'):
                    cols = df.columns.get_level_values(0)
                    if 'Adj Close' in cols:
                        close_col = 'Adj Close'
                    elif 'Close' in cols:
                        close_col = 'Close'
                    else:
                        print(" No price column")
                        failed_tickers.append(ticker)
                        continue
                else:
                    print(" No price column")
                    failed_tickers.append(ticker)
                    continue

            # Extract prices
            count = 0
            for date_idx, row in df.iterrows():
                try:
                    if hasattr(df.columns, 'get_level_values'):
                        price = float(row[(close_col, ticker)])
                    else:
                        price = float(row[close_col])

                    if price > 0:
                        all_prices.append({
                            'date': date_idx.strftime('%Y-%m-%d'),
                            'ticker': ticker,
                            'close': price
                        })
                        count += 1
                except (KeyError, ValueError, TypeError):
                    continue

            if count > 0:
                print(f" {count} days")
                success_count += 1
            else:
                print(" No valid prices")
                failed_tickers.append(ticker)

        except Exception as e:
            print(f" Error: {str(e)[:40]}")
            failed_tickers.append(ticker)

        if delay > 0 and i < len(tickers):
            time.sleep(delay)

    # Sort and save
    all_prices.sort(key=lambda x: (x['date'], x['ticker']))

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
    print(f"Success: {success_count}/{len(tickers)} ({100*success_count/len(tickers):.1f}%)")
    print(f"Total observations: {len(all_prices):,}")
    print(f"Output: {output_file}")

    if failed_tickers:
        print()
        print(f"Failed ({len(failed_tickers)}):")
        for t in failed_tickers[:15]:
            print(f"  - {t}")
        if len(failed_tickers) > 15:
            print(f"  ... and {len(failed_tickers) - 15} more")

    return success_count > 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Fetch prices using yfinance')
    parser.add_argument('--universe', default='production_data/universe.json')
    parser.add_argument('--start-date', default='2020-01-01')
    parser.add_argument('--end-date', default='2026-01-16')
    parser.add_argument('--output', default='production_data/price_history.csv')
    parser.add_argument('--delay', type=float, default=0.5)

    args = parser.parse_args()

    print("=" * 70)
    print("BIOTECH UNIVERSE PRICE FETCHER (yfinance)")
    print("=" * 70)

    print(f"\nLoading universe from: {args.universe}")
    tickers = load_universe_tickers(args.universe)
    print(f"Found {len(tickers)} tickers (including XBI)")

    success = fetch_prices_yfinance(
        tickers,
        args.start_date,
        args.end_date,
        args.output,
        args.delay
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
