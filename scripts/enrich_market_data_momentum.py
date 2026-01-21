#!/usr/bin/env python3
"""
enrich_market_data_momentum.py - Add 60-day momentum fields to market_data.json

This script enriches the existing market_data.json with the following fields required
for momentum scoring in Module 5:
- return_60d: 60 trading day compounded return (P_t / P_{t-60} - 1)
- xbi_return_60d: XBI benchmark 60-day return (same calculation)
- volatility_252d: Annualized 252-day volatility (optional but improves scoring)

The fields are calculated using yfinance historical price data.

Usage:
    python scripts/enrich_market_data_momentum.py \
        --market-data production_data/market_data.json \
        --output production_data/market_data.json

Point-in-Time Safety:
    - Uses the collected_at date from each record as the reference date
    - Ensures consistent lookback windows across all tickers
"""

import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import sys


def fetch_historical_prices(
    ticker: str,
    end_date: datetime,
    lookback_days: int = 300,  # Fetch extra days to ensure we have 252 trading days
) -> Optional[List[Dict]]:
    """
    Fetch historical daily prices for a ticker.

    Returns list of {date, close} dicts sorted by date ascending.
    """
    try:
        import yfinance as yf

        start_date = end_date - timedelta(days=lookback_days)
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)

        if hist.empty:
            return None

        prices = []
        for idx, row in hist.iterrows():
            prices.append({
                "date": idx.strftime("%Y-%m-%d"),
                "close": float(row["Close"])
            })

        return prices

    except Exception as e:
        print(f"  Warning: Failed to fetch {ticker}: {e}")
        return None


def calculate_return(prices: List[Dict], lookback_trading_days: int) -> Optional[float]:
    """
    Calculate compounded return over N trading days.

    Formula: (P_t / P_{t-N}) - 1

    Args:
        prices: List of {date, close} dicts sorted by date ascending
        lookback_trading_days: Number of trading days to look back (e.g., 60)

    Returns:
        Compounded return as decimal (e.g., 0.15 = 15%), or None if insufficient data
    """
    if not prices or len(prices) < lookback_trading_days + 1:
        return None

    # Get the most recent price and the price N trading days ago
    current_price = prices[-1]["close"]
    past_price = prices[-(lookback_trading_days + 1)]["close"]

    if past_price <= 0:
        return None

    return (current_price / past_price) - 1


def calculate_volatility(prices: List[Dict], lookback_trading_days: int = 252) -> Optional[float]:
    """
    Calculate annualized volatility over N trading days.

    Formula: std(daily_returns) * sqrt(252)

    Args:
        prices: List of {date, close} dicts sorted by date ascending
        lookback_trading_days: Number of trading days (default 252 for annual)

    Returns:
        Annualized volatility as decimal (e.g., 0.50 = 50%)
    """
    if not prices or len(prices) < lookback_trading_days + 1:
        return None

    # Use the most recent N trading days
    recent_prices = prices[-(lookback_trading_days + 1):]

    # Calculate daily returns
    returns = []
    for i in range(1, len(recent_prices)):
        if recent_prices[i-1]["close"] > 0:
            daily_return = (recent_prices[i]["close"] / recent_prices[i-1]["close"]) - 1
            returns.append(daily_return)

    if len(returns) < 20:  # Need at least 20 data points
        return None

    # Calculate standard deviation
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    std_dev = variance ** 0.5

    # Annualize
    annualized_vol = std_dev * (252 ** 0.5)

    return annualized_vol


def enrich_market_data(
    market_data: List[Dict],
    reference_date: Optional[str] = None,
    rate_limit_delay: float = 0.2,
) -> List[Dict]:
    """
    Enrich market data records with momentum fields.

    Args:
        market_data: List of market data records
        reference_date: Optional override date (YYYY-MM-DD), otherwise uses collected_at
        rate_limit_delay: Delay between API calls in seconds

    Returns:
        Enriched market data with return_60d, xbi_return_60d, volatility_252d
    """
    import time

    # Determine reference date
    if reference_date:
        ref_date = datetime.strptime(reference_date, "%Y-%m-%d")
    else:
        # Use the most recent collected_at from the data
        collected_dates = [
            r.get("collected_at") for r in market_data
            if r.get("collected_at")
        ]
        if collected_dates:
            ref_date = datetime.strptime(max(collected_dates), "%Y-%m-%d")
        else:
            ref_date = datetime.now()

    print(f"Reference date: {ref_date.strftime('%Y-%m-%d')}")

    # Fetch XBI (benchmark) prices once
    print("Fetching XBI benchmark data...")
    xbi_prices = fetch_historical_prices("XBI", ref_date, lookback_days=300)
    xbi_return_60d = None

    if xbi_prices:
        xbi_return_60d = calculate_return(xbi_prices, 60)
        if xbi_return_60d is not None:
            print(f"  XBI 60-day return: {xbi_return_60d:.4f} ({xbi_return_60d*100:.2f}%)")
        else:
            print("  Warning: Could not calculate XBI 60-day return")
    else:
        print("  Warning: Could not fetch XBI prices")

    time.sleep(rate_limit_delay)

    # Process each ticker
    enriched = []
    total = len(market_data)
    success_count = 0

    print(f"\nProcessing {total} tickers...")

    for i, record in enumerate(market_data):
        ticker = record.get("ticker")
        if not ticker:
            enriched.append(record)
            continue

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{total} ({success_count} with momentum data)")

        # Fetch historical prices
        prices = fetch_historical_prices(ticker, ref_date, lookback_days=300)

        if prices:
            return_60d = calculate_return(prices, 60)
            volatility_252d = calculate_volatility(prices, 252)

            if return_60d is not None:
                success_count += 1
                record["return_60d"] = round(return_60d, 6)
                record["xbi_return_60d"] = round(xbi_return_60d, 6) if xbi_return_60d else None
                record["benchmark_return_60d"] = record["xbi_return_60d"]  # Alias

            if volatility_252d is not None:
                record["volatility_252d"] = round(volatility_252d, 6)
                record["annualized_volatility"] = record["volatility_252d"]  # Alias

        enriched.append(record)
        time.sleep(rate_limit_delay)

    print(f"\nCompleted: {success_count}/{total} tickers enriched with momentum data")

    return enriched


def main():
    parser = argparse.ArgumentParser(
        description="Enrich market_data.json with 60-day momentum fields"
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
        "--rate-limit",
        type=float,
        default=0.2,
        help="Delay between API calls in seconds (default: 0.2)"
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

    # Check current state
    has_return_60d = sum(1 for r in market_data if r.get("return_60d") is not None)
    has_xbi = sum(1 for r in market_data if r.get("xbi_return_60d") is not None)
    has_vol_252d = sum(1 for r in market_data if r.get("volatility_252d") is not None)

    print(f"\nCurrent state:")
    print(f"  With return_60d: {has_return_60d}/{len(market_data)}")
    print(f"  With xbi_return_60d: {has_xbi}/{len(market_data)}")
    print(f"  With volatility_252d: {has_vol_252d}/{len(market_data)}")

    if args.dry_run:
        print("\n[DRY RUN] Would enrich market data with momentum fields")
        sys.exit(0)

    # Check for yfinance
    try:
        import yfinance
        print(f"\nyfinance version: {yfinance.__version__}")
    except ImportError:
        print("\nError: yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    # Enrich the data
    print("\n" + "="*60)
    enriched = enrich_market_data(
        market_data,
        reference_date=args.reference_date,
        rate_limit_delay=args.rate_limit,
    )
    print("="*60)

    # Report final state
    has_return_60d = sum(1 for r in enriched if r.get("return_60d") is not None)
    has_xbi = sum(1 for r in enriched if r.get("xbi_return_60d") is not None)
    has_vol_252d = sum(1 for r in enriched if r.get("volatility_252d") is not None)

    print(f"\nFinal state:")
    print(f"  With return_60d: {has_return_60d}/{len(enriched)}")
    print(f"  With xbi_return_60d: {has_xbi}/{len(enriched)}")
    print(f"  With volatility_252d: {has_vol_252d}/{len(enriched)}")

    # Write output
    output_path = args.output or args.market_data
    print(f"\nWriting to {output_path}...")

    with open(output_path, "w") as f:
        json.dump(enriched, f, indent=2, sort_keys=False)

    print("Done!")


if __name__ == "__main__":
    main()
