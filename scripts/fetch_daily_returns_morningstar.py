#!/usr/bin/env python3
"""
Fetch daily returns from Morningstar Direct for the full biotech universe.

Run locally with your Morningstar credentials:

    # Windows PowerShell:
    $env:MD_AUTH_TOKEN="your-token-here"
    python scripts/fetch_daily_returns_morningstar.py

    # Mac/Linux:
    export MD_AUTH_TOKEN="your-token-here"
    python scripts/fetch_daily_returns_morningstar.py

Options:
    --years N          Number of years of daily data to fetch (default: 2)
    --batch-size N     Number of tickers per batch (default: 50)
    --output PATH      Output file path (default: data/returns/returns_db_daily.json)
    --universe PATH    Universe file path (default: wake_robin_data_pipeline/universe/full_universe.json)
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

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


def lookup_secids(tickers: list[str]) -> dict[str, str]:
    """Look up Morningstar SecIds for each ticker."""
    import morningstar_data as md

    ticker_to_secid = {}
    failed = []

    logger.info(f"Looking up SecIds for {len(tickers)} tickers...")

    for i, ticker in enumerate(tickers):
        try:
            inv_df = md.direct.investments(ticker)
            if inv_df is not None and not inv_df.empty:
                # Filter for US listings
                us_rows = inv_df[
                    (inv_df['Country'] == 'USA') &
                    (inv_df['Base Currency'] == 'USD')
                ]
                if not us_rows.empty:
                    sec_id = str(us_rows.iloc[0]['SecId'])
                    ticker_to_secid[ticker] = sec_id
                    logger.debug(f"  {ticker}: {sec_id}")
                else:
                    failed.append(ticker)
                    logger.debug(f"  {ticker}: No US listing found")
            else:
                failed.append(ticker)
                logger.debug(f"  {ticker}: No data")
        except Exception as e:
            failed.append(ticker)
            logger.debug(f"  {ticker}: ERROR - {str(e)[:50]}")

        # Progress update every 50 tickers
        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i + 1}/{len(tickers)} tickers...")

        # Small delay to avoid rate limiting
        time.sleep(0.1)

    logger.info(f"Found SecIds for {len(ticker_to_secid)} tickers, {len(failed)} failed")
    if failed:
        logger.debug(f"Failed tickers: {failed[:20]}...")

    return ticker_to_secid


def fetch_returns_batch(
    sec_ids: list[str],
    secid_to_ticker: dict[str, str],
    start_date: str,
    end_date: str
) -> dict[str, dict[str, str]]:
    """Fetch daily returns for a batch of SecIds."""
    import morningstar_data as md
    from morningstar_data.direct.data_type import Frequency

    all_returns = {}

    try:
        df = md.direct.get_returns(
            investments=sec_ids,
            start_date=start_date,
            end_date=end_date,
            freq=Frequency.daily,
        )

        if df is None or df.empty:
            logger.warning(f"No data returned for batch")
            return all_returns

        logger.debug(f"Got DataFrame with {len(df)} rows, columns: {list(df.columns)}")

        # Find columns
        id_col = None
        date_col = None
        return_col = None

        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['id', 'secid', 'sec_id']:
                id_col = col
            elif col_lower in ['date', 'enddate', 'end_date']:
                date_col = col
            elif 'return' in col_lower:
                return_col = col

        if not all([id_col, date_col, return_col]):
            logger.error(f"Could not identify required columns. Available: {list(df.columns)}")
            return all_returns

        # Process rows
        for idx, row in df.iterrows():
            sec_id = str(row[id_col])
            ticker = secid_to_ticker.get(sec_id)
            if not ticker:
                continue

            if ticker not in all_returns:
                all_returns[ticker] = {}

            date_val = row[date_col]
            if hasattr(date_val, 'strftime'):
                date_str = date_val.strftime('%Y-%m-%d')
            else:
                date_str = str(date_val).split('T')[0].split(' ')[0]

            ret_val = row[return_col]
            if ret_val is not None and str(ret_val) not in ['nan', 'None', '']:
                # Morningstar returns are in percentage, convert to decimal
                all_returns[ticker][date_str] = str(float(ret_val) / 100)

    except Exception as e:
        logger.error(f"Error fetching batch: {e}")

    return all_returns


def fetch_all_returns(
    ticker_to_secid: dict[str, str],
    start_date: str,
    end_date: str,
    batch_size: int = 50
) -> dict[str, dict[str, str]]:
    """Fetch daily returns for all tickers in batches."""

    secid_to_ticker = {v: k for k, v in ticker_to_secid.items()}
    all_sec_ids = list(ticker_to_secid.values())
    all_returns = {}

    # Process in batches
    num_batches = (len(all_sec_ids) + batch_size - 1) // batch_size
    logger.info(f"Fetching returns in {num_batches} batches of {batch_size}...")

    for i in range(0, len(all_sec_ids), batch_size):
        batch_num = i // batch_size + 1
        batch_sec_ids = all_sec_ids[i:i + batch_size]

        logger.info(f"Processing batch {batch_num}/{num_batches} ({len(batch_sec_ids)} tickers)...")

        batch_returns = fetch_returns_batch(
            batch_sec_ids, secid_to_ticker, start_date, end_date
        )

        # Merge results
        for ticker, returns in batch_returns.items():
            if ticker not in all_returns:
                all_returns[ticker] = {}
            all_returns[ticker].update(returns)

        logger.info(f"  Batch {batch_num}: got data for {len(batch_returns)} tickers")

        # Delay between batches
        if i + batch_size < len(all_sec_ids):
            time.sleep(1)

    return all_returns


def main():
    parser = argparse.ArgumentParser(description='Fetch daily returns from Morningstar Direct')
    parser.add_argument('--years', type=int, default=2, help='Years of data to fetch')
    parser.add_argument('--batch-size', type=int, default=50, help='Tickers per batch')
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

    # Check for auth token
    if not os.environ.get('MD_AUTH_TOKEN'):
        logger.error("MD_AUTH_TOKEN environment variable not set!")
        logger.error("Set it with:")
        logger.error('  Windows: $env:MD_AUTH_TOKEN="your-token"')
        logger.error('  Mac/Linux: export MD_AUTH_TOKEN="your-token"')
        sys.exit(1)

    # Date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365 * args.years)).strftime('%Y-%m-%d')

    logger.info(f"Fetching daily returns from {start_date} to {end_date}")

    # Load universe
    logger.info(f"Loading universe from {args.universe}...")
    tickers = load_universe(args.universe)

    # Add XBI benchmark
    if args.include_xbi:
        tickers.append('XBI')

    logger.info(f"Processing {len(tickers)} tickers")

    # Look up SecIds
    ticker_to_secid = lookup_secids(tickers)

    if not ticker_to_secid:
        logger.error("Failed to find any SecIds!")
        sys.exit(1)

    # Fetch returns
    all_returns = fetch_all_returns(
        ticker_to_secid, start_date, end_date, args.batch_size
    )

    # Separate XBI as benchmark
    xbi_returns = all_returns.pop('XBI', {})
    ticker_returns = all_returns

    # Summary
    logger.info("")
    logger.info("=== Results Summary ===")
    for ticker in sorted(ticker_returns.keys())[:20]:
        logger.info(f"  {ticker}: {len(ticker_returns[ticker])} days")
    if len(ticker_returns) > 20:
        logger.info(f"  ... and {len(ticker_returns) - 20} more tickers")
    logger.info(f"  XBI (benchmark): {len(xbi_returns)} days")

    # Save output
    output = {
        'tickers': ticker_returns,
        'benchmark': {'XBI': xbi_returns},
        'metadata': {
            'source': 'morningstar_direct',
            'fetch_date': datetime.now().isoformat(),
            'start_date': start_date,
            'end_date': end_date,
            'frequency': 'daily',
            'total_tickers': len(ticker_returns),
            'universe_file': args.universe
        }
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nSaved to: {output_path}")
    logger.info(f"Total tickers with data: {len(ticker_returns)}")
    logger.info(f"XBI benchmark days: {len(xbi_returns)}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
