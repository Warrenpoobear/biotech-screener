#!/usr/bin/env python3
"""
Fetch daily returns from Yahoo Finance for the biotech universe.

This is a fallback script if Morningstar Direct is unavailable.

Usage:
    pip install yfinance
    python scripts/fetch_daily_returns_yfinance.py

Options:
    --years N          Number of years of daily data to fetch (default: 2)
    --output PATH      Output file path (default: data/returns/returns_db_daily.json)
    --universe PATH    Universe file path (default: wake_robin_data_pipeline/universe/full_universe.json)
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

try:
    import yfinance as yf
except ImportError:
    print("Error: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_universe(universe_path: str) -> list[str]:
    """Load tickers from universe file."""
    with open(universe_path) as f:
        data = json.load(f)

    # Extract tickers, filtering out invalid ones
    invalid_tickers = {'_XBI_BENCHMARK_', '-', 'NAME', 'USD', 'XTSLA', 'RTYH6', 'IXCH6', 'SGAFT'}
    tickers = []

    for t in data.get('tickers', []):
        ticker = t.get('ticker', '')
        # Only include valid stock tickers (alphabetic, <= 5 chars)
        if ticker not in invalid_tickers and len(ticker) <= 5 and ticker.isalpha():
            tickers.append(ticker)

    return tickers


def fetch_ticker_returns(ticker: str, start_date: datetime, end_date: datetime) -> dict[str, str]:
    """Fetch daily returns for a single ticker."""
    returns = {}

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)

        if hist.empty:
            return returns

        # Calculate daily returns
        prev_close = None
        for date_idx, row in hist.iterrows():
            curr_close = row['Close']
            if prev_close is not None:
                daily_return = (curr_close / prev_close) - 1.0
                returns[date_idx.strftime('%Y-%m-%d')] = str(daily_return)
            prev_close = curr_close

    except Exception as e:
        logger.debug(f"Error fetching {ticker}: {e}")

    return returns


def main():
    parser = argparse.ArgumentParser(description='Fetch daily returns from Yahoo Finance')
    parser.add_argument('--years', type=int, default=2, help='Years of data to fetch')
    parser.add_argument('--output', type=str, default='data/returns/returns_db_daily.json',
                       help='Output file path')
    parser.add_argument('--universe', type=str,
                       default='wake_robin_data_pipeline/universe/full_universe.json',
                       help='Universe file path')
    parser.add_argument('--include-xbi', action='store_true', default=True,
                       help='Include XBI benchmark')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * args.years)

    logger.info(f"Fetching daily returns from {start_date.date()} to {end_date.date()}")

    # Load universe
    logger.info(f"Loading universe from {args.universe}...")
    tickers = load_universe(args.universe)

    # Add XBI benchmark
    if args.include_xbi:
        tickers.append('XBI')

    logger.info(f"Processing {len(tickers)} tickers")

    # Fetch returns for each ticker
    all_returns = {}
    failed = []

    for i, ticker in enumerate(tickers):
        returns = fetch_ticker_returns(ticker, start_date, end_date)

        if returns:
            all_returns[ticker] = returns
            logger.debug(f"  {ticker}: {len(returns)} days")
        else:
            failed.append(ticker)
            logger.debug(f"  {ticker}: No data")

        # Progress update every 20 tickers
        if (i + 1) % 20 == 0:
            logger.info(f"  Processed {i + 1}/{len(tickers)} tickers...")

        # Small delay to be nice to Yahoo Finance
        time.sleep(0.2)

    # Separate XBI as benchmark
    xbi_returns = all_returns.pop('XBI', {})
    ticker_returns = all_returns

    # Summary
    logger.info("")
    logger.info("=== Results Summary ===")
    logger.info(f"Total tickers with data: {len(ticker_returns)}")
    logger.info(f"Failed tickers: {len(failed)}")
    logger.info(f"XBI benchmark days: {len(xbi_returns)}")

    if failed:
        logger.info(f"Failed tickers (first 20): {failed[:20]}")

    # Save output
    output = {
        'tickers': ticker_returns,
        'benchmark': {'XBI': xbi_returns},
        'metadata': {
            'source': 'yahoo_finance',
            'fetch_date': datetime.now().isoformat(),
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'frequency': 'daily',
            'total_tickers': len(ticker_returns),
            'failed_tickers': failed,
            'universe_file': args.universe
        }
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nSaved to: {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
