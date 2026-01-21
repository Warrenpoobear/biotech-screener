#!/usr/bin/env python3
"""
compute_momentum_returns.py - Add 60-day momentum returns to market_data.json

Uses only Python standard library (no yfinance dependency).
Fetches 60-day historical prices from Yahoo Finance and computes:
- return_60d: Stock's 60 trading day compounded return
- xbi_return_60d: XBI benchmark 60-day return
- volatility_252d: Annualized volatility (if enough history)

Usage:
    python scripts/compute_momentum_returns.py \
        --market-data production_data/market_data.json \
        --output production_data/market_data.json

Point-in-Time Safety:
    Uses collected_at date from market_data as reference to avoid lookahead bias.
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
from typing import Dict, List, Optional, Tuple
import argparse


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

    def _get_crumb(self) -> Optional[str]:
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

    def fetch_prices(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Tuple[str, float]]:
        """
        Fetch historical prices from Yahoo Finance.

        Returns list of (date_str, close_price) tuples sorted by date ascending.
        """
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
            return []
        except Exception:
            return []


def calculate_return_60d(prices: List[Tuple[str, float]]) -> Optional[float]:
    """
    Calculate 60 trading day compounded return.

    Formula: (P_t / P_{t-60}) - 1

    Args:
        prices: List of (date_str, close) tuples sorted by date ascending

    Returns:
        Compounded return as decimal, or None if insufficient data
    """
    if len(prices) < 61:  # Need 61 prices for 60-day return
        return None

    # Most recent price and price 60 trading days ago
    current_price = prices[-1][1]
    past_price = prices[-61][1]

    if past_price <= 0:
        return None

    return (current_price / past_price) - 1


def calculate_volatility_252d(prices: List[Tuple[str, float]]) -> Optional[float]:
    """
    Calculate 252-day annualized volatility.

    Formula: std(daily_returns) * sqrt(252)

    Args:
        prices: List of (date_str, close) tuples sorted by date ascending

    Returns:
        Annualized volatility as decimal, or None if insufficient data
    """
    if len(prices) < 100:  # Need at least ~100 days
        return None

    # Use most recent 252 days (or all available)
    recent_prices = prices[-253:] if len(prices) >= 253 else prices

    # Calculate daily returns
    returns = []
    for i in range(1, len(recent_prices)):
        if recent_prices[i-1][1] > 0:
            daily_return = (recent_prices[i][1] / recent_prices[i-1][1]) - 1
            returns.append(daily_return)

    if len(returns) < 20:
        return None

    # Calculate standard deviation
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    std_dev = variance ** 0.5

    # Annualize
    return std_dev * (252 ** 0.5)


def main():
    parser = argparse.ArgumentParser(
        description="Add 60-day momentum returns to market_data.json"
    )
    parser.add_argument(
        "--market-data",
        type=Path,
        required=True,
        help="Path to market_data.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path (defaults to overwriting input)"
    )
    parser.add_argument(
        "--reference-date",
        type=str,
        help="Reference date (YYYY-MM-DD), defaults to collected_at from data"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Delay between API calls in seconds (default: 0.3)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of tickers to process (for testing)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without making changes"
    )

    args = parser.parse_args()

    # Load existing market data
    if not args.market_data.exists():
        print(f"Error: {args.market_data} not found")
        sys.exit(1)

    print(f"Loading {args.market_data}...")
    with open(args.market_data) as f:
        market_data = json.load(f)

    if not isinstance(market_data, list):
        print("Error: market_data.json must be a list of records")
        sys.exit(1)

    print(f"Loaded {len(market_data)} records")

    # Determine reference date
    if args.reference_date:
        ref_date = datetime.strptime(args.reference_date, "%Y-%m-%d")
    else:
        collected_dates = [
            r.get("collected_at") for r in market_data
            if r.get("collected_at")
        ]
        if collected_dates:
            ref_date = datetime.strptime(max(collected_dates), "%Y-%m-%d")
        else:
            ref_date = datetime.now()

    print(f"Reference date: {ref_date.strftime('%Y-%m-%d')}")

    # Calculate date range (need ~300 days for 252-day vol)
    start_date = ref_date - timedelta(days=370)
    end_date = ref_date + timedelta(days=1)

    # Check current state
    has_return_60d = sum(1 for r in market_data if r.get("return_60d") is not None)
    has_xbi = sum(1 for r in market_data if r.get("xbi_return_60d") is not None)
    has_vol = sum(1 for r in market_data if r.get("volatility_252d") is not None)

    print(f"\nCurrent state:")
    print(f"  With return_60d: {has_return_60d}/{len(market_data)}")
    print(f"  With xbi_return_60d: {has_xbi}/{len(market_data)}")
    print(f"  With volatility_252d: {has_vol}/{len(market_data)}")

    if args.dry_run:
        print("\n[DRY RUN] Would fetch prices and compute momentum data")
        sys.exit(0)

    # Initialize Yahoo Finance session
    print("\nInitializing Yahoo Finance session...")
    session = YahooFinanceSession()

    # Fetch XBI benchmark first
    print("Fetching XBI benchmark prices...")
    xbi_prices = session.fetch_prices("XBI", start_date, end_date)
    xbi_return_60d = None

    if xbi_prices:
        xbi_return_60d = calculate_return_60d(xbi_prices)
        if xbi_return_60d is not None:
            print(f"  XBI 60-day return: {xbi_return_60d:.4f} ({xbi_return_60d*100:.2f}%)")
            print(f"  XBI data points: {len(xbi_prices)}")
        else:
            print(f"  Warning: Insufficient XBI data ({len(xbi_prices)} points)")
    else:
        print("  Warning: Could not fetch XBI prices")

    time.sleep(args.delay)

    # Process each ticker
    tickers = [r.get("ticker") for r in market_data if r.get("ticker")]
    if args.limit:
        tickers = tickers[:args.limit]

    print(f"\nProcessing {len(tickers)} tickers...")
    print("="*60)

    success_count = 0
    ticker_to_data = {}

    for i, ticker in enumerate(tickers):
        if (i + 1) % 25 == 0:
            print(f"  Progress: {i + 1}/{len(tickers)} ({success_count} successful)")

        prices = session.fetch_prices(ticker, start_date, end_date)

        if prices:
            return_60d = calculate_return_60d(prices)
            volatility_252d = calculate_volatility_252d(prices)

            if return_60d is not None:
                success_count += 1
                ticker_to_data[ticker] = {
                    "return_60d": round(return_60d, 6),
                    "xbi_return_60d": round(xbi_return_60d, 6) if xbi_return_60d else None,
                    "benchmark_return_60d": round(xbi_return_60d, 6) if xbi_return_60d else None,
                    "volatility_252d": round(volatility_252d, 6) if volatility_252d else None,
                    "annualized_volatility": round(volatility_252d, 6) if volatility_252d else None,
                    "price_data_points": len(prices),
                }

        time.sleep(args.delay)

    print("="*60)
    print(f"Fetched momentum data for {success_count}/{len(tickers)} tickers")

    # Update market data records
    for record in market_data:
        ticker = record.get("ticker")
        if ticker and ticker in ticker_to_data:
            record.update(ticker_to_data[ticker])

    # Report final state
    has_return_60d = sum(1 for r in market_data if r.get("return_60d") is not None)
    has_xbi = sum(1 for r in market_data if r.get("xbi_return_60d") is not None)
    has_vol = sum(1 for r in market_data if r.get("volatility_252d") is not None)

    print(f"\nFinal state:")
    print(f"  With return_60d: {has_return_60d}/{len(market_data)}")
    print(f"  With xbi_return_60d: {has_xbi}/{len(market_data)}")
    print(f"  With volatility_252d: {has_vol}/{len(market_data)}")

    # Write output
    output_path = args.output or args.market_data
    print(f"\nWriting to {output_path}...")

    with open(output_path, "w") as f:
        json.dump(market_data, f, indent=2, sort_keys=False)

    print("Done!")

    # Print sample records
    print("\nSample enriched records:")
    for r in market_data[:3]:
        if r.get("return_60d"):
            print(f"  {r['ticker']}: return_60d={r.get('return_60d'):.4f}, "
                  f"xbi={r.get('xbi_return_60d'):.4f if r.get('xbi_return_60d') else 'N/A'}, "
                  f"vol={r.get('volatility_252d'):.4f if r.get('volatility_252d') else 'N/A'}")


if __name__ == "__main__":
    main()
