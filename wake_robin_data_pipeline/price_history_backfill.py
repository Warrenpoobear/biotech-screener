#!/usr/bin/env python3
"""Offline Price History Backfill CLI (PIT-safe)."""

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Tuple

from wake_robin_data_pipeline.price_gap_report import (
    load_universe_tickers,
    load_price_coverage,
)

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


def get_blocking_tickers(
    universe_tickers: List[str],
    price_coverage: Dict[str, Dict],
    min_rows: int,
) -> List[str]:
    """Get tickers that need backfill (missing or insufficient rows)."""
    blocking = []
    for ticker in sorted(universe_tickers):
        if ticker.startswith("_"):
            continue  # Skip synthetic
        cov = price_coverage.get(ticker)
        if not cov or cov["rows_total"] < min_rows:
            blocking.append(ticker)
    return blocking


def load_existing_dates(price_file: str, ticker: str) -> Set[str]:
    """Load existing dates for a ticker from price file."""
    dates = set()
    with open(price_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("ticker") == ticker and row.get("date"):
                dates.add(row["date"])
    return dates


def fetch_prices_yfinance(ticker: str, start_date: str, end_date: str) -> List[Dict]:
    """Fetch daily prices from yfinance."""
    if not HAS_YFINANCE:
        return []
    try:
        t = yf.Ticker(ticker)
        df = t.history(start=start_date, end=end_date, interval="1d")
        if df.empty:
            return []
        rows = []
        for idx, row in df.iterrows():
            date_str = idx.strftime("%Y-%m-%d")
            rows.append({
                "date": date_str,
                "ticker": ticker,
                "close": row.get("Close"),
                "open": row.get("Open"),
                "high": row.get("High"),
                "low": row.get("Low"),
                "volume": row.get("Volume"),
            })
        return rows
    except Exception as e:
        print(f"  Warning: fetch failed for {ticker}: {e}")
        return []


def backfill_ticker(
    ticker: str,
    price_file: str,
    as_of: str,
    min_rows: int,
    dry_run: bool,
) -> Tuple[int, int]:
    """Backfill prices for a single ticker. Returns (fetched, new)."""
    existing = load_existing_dates(price_file, ticker)

    # Compute start date: go back enough to get min_rows trading days
    # ~1.5x trading days to calendar days ratio, plus buffer
    as_of_dt = datetime.strptime(as_of, "%Y-%m-%d")
    days_needed = int(min_rows * 1.5) + 30
    start_dt = as_of_dt - timedelta(days=days_needed)
    start_date = start_dt.strftime("%Y-%m-%d")

    # Fetch
    rows = fetch_prices_yfinance(ticker, start_date, as_of)
    if not rows:
        return 0, 0

    # Filter: only dates <= as_of and not already existing
    new_rows = [r for r in rows if r["date"] <= as_of and r["date"] not in existing]

    if dry_run:
        return len(rows), len(new_rows)

    # Append to file
    if new_rows:
        with open(price_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["date", "ticker", "close", "open", "high", "low", "volume"])
            for row in sorted(new_rows, key=lambda x: x["date"]):
                writer.writerow(row)

    return len(rows), len(new_rows)


def main():
    parser = argparse.ArgumentParser(description="Backfill missing price history")
    parser.add_argument("--as-of", required=True, help="As-of date (YYYY-MM-DD)")
    parser.add_argument("--price-file", required=True, help="Price history CSV")
    parser.add_argument("--universe", default=None, help="Universe JSON (optional)")
    parser.add_argument("--min-rows", type=int, default=120, help="Min rows needed (default 120)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fetched")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to backfill (overrides gap detection)")
    args = parser.parse_args()

    if not HAS_YFINANCE:
        print("Error: yfinance not installed. Run: pip install yfinance")
        return

    # Load universe and coverage
    universe = load_universe_tickers(args.universe)
    coverage = load_price_coverage(args.price_file, args.as_of)

    # Determine which tickers to backfill
    if args.tickers:
        blocking = [t for t in args.tickers if not t.startswith("_")]
    else:
        blocking = get_blocking_tickers(universe, coverage, args.min_rows)

    if not blocking:
        print(f"No tickers need backfill (all have >= {args.min_rows} rows)")
        return

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Backfill for as_of={args.as_of}, min_rows={args.min_rows}")
    print(f"  Blocking tickers: {len(blocking)}")

    total_fetched = 0
    total_new = 0

    for ticker in blocking:
        cov = coverage.get(ticker, {})
        current = cov.get("rows_total", 0)
        fetched, new = backfill_ticker(ticker, args.price_file, args.as_of, args.min_rows, args.dry_run)
        total_fetched += fetched
        total_new += new
        status = f"fetched={fetched}, new={new}"
        if args.dry_run:
            print(f"  {ticker}: current={current}, would_fetch={fetched}, would_add={new}")
        else:
            print(f"  {ticker}: current={current} -> +{new} rows")

    print(f"\nTotal: fetched={total_fetched}, {'would add' if args.dry_run else 'added'}={total_new} rows")


if __name__ == "__main__":
    main()
