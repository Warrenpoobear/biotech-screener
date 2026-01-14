#!/usr/bin/env python3
"""
Fetch Morningstar returns data for Wake Robin universe.

Handles your actual data source format and converts to the
standard format expected by the momentum calculator.
"""

from decimal import Decimal
from pathlib import Path
import json
from datetime import datetime, timedelta
import sys


def fetch_morningstar_returns(ticker_list, start_date, end_date):
    """
    Fetch Morningstar daily returns for ticker list.

    TODO: Replace this with your actual Morningstar data fetcher.
    Options:
    1. Morningstar Direct API
    2. CSV export from Morningstar Direct
    3. Bloomberg/FactSet as alternative
    4. yfinance as fallback (free but lower quality)

    Returns:
        dict: {ticker: {date_str: Decimal(return)}}
    """
    # First try to load from existing returns database
    returns_db_path = Path("data/returns")
    if returns_db_path.exists():
        for db_file in sorted(returns_db_path.glob("returns_db_*.json"), reverse=True):
            print(f"Found existing returns database: {db_file}")
            return load_from_returns_db(db_file, ticker_list)

    # Fallback to yfinance
    print("WARNING: Using yfinance fallback - consider using Morningstar Direct for production")
    return fetch_via_yfinance(ticker_list, start_date, end_date)


def load_from_returns_db(db_file, ticker_list):
    """Load returns from existing Morningstar returns database."""
    with open(db_file, 'r') as f:
        data = json.load(f)

    result = {}
    tickers_data = data.get("tickers", {})

    for ticker in ticker_list:
        if ticker in tickers_data:
            # Convert to Decimal
            result[ticker] = {
                date: Decimal(str(ret))
                for date, ret in tickers_data[ticker].items()
            }

    return result


def load_xbi_from_returns_db(db_file):
    """Load XBI benchmark from existing returns database."""
    with open(db_file, 'r') as f:
        data = json.load(f)

    benchmark_data = data.get("benchmark", {}).get("XBI", {})

    return {
        date: Decimal(str(ret))
        for date, ret in benchmark_data.items()
    }


def fetch_via_yfinance(ticker_list, start_date, end_date):
    """
    Fallback: Fetch via yfinance (free but survivorship bias risk).

    NOTE: This is ONLY for initial testing. Production should use
    Morningstar Direct which includes delisted tickers.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed. Run: pip install yfinance")
        return {}

    all_returns = {}

    for i, ticker in enumerate(ticker_list, 1):
        if i % 10 == 0:
            print(f"  Fetching {i}/{len(ticker_list)}...")

        try:
            data = yf.Ticker(ticker)
            hist = data.history(start=start_date, end=end_date)

            if hist.empty:
                continue

            # Calculate daily returns
            returns = {}
            prev_close = None

            for date_idx, row in hist.iterrows():
                curr_close = row['Close']

                if prev_close is not None:
                    daily_return = (curr_close / prev_close) - 1.0
                    returns[date_idx.strftime("%Y-%m-%d")] = Decimal(str(daily_return))

                prev_close = curr_close

            all_returns[ticker] = returns

        except Exception as e:
            print(f"  Warning: Failed to fetch {ticker}: {e}")
            continue

    return all_returns


def save_returns_data(returns_data, output_file):
    """Save returns data to JSON with Decimal→string conversion."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert Decimals to strings for JSON
    data_json = {}
    for ticker, returns in returns_data.items():
        data_json[ticker] = {
            date: str(ret)
            for date, ret in returns.items()
        }

    with open(output_path, 'w') as f:
        json.dump(data_json, f, indent=2)

    print(f"Returns data saved to: {output_path}")


def main():
    """Fetch data for your biotech universe."""
    import argparse

    parser = argparse.ArgumentParser(description="Fetch Morningstar returns data")
    parser.add_argument(
        "--universe-file",
        type=str,
        default="data/universe/biotech_universe.json",
        help="Path to universe file with ticker list"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/momentum",
        help="Output directory for returns data"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=730,
        help="Number of days of history to fetch (default: 730 = 2 years)"
    )

    args = parser.parse_args()

    # Load your universe
    universe_file = Path(args.universe_file)

    if not universe_file.exists():
        # Try to get tickers from existing returns database
        returns_db_path = Path("data/returns")
        if returns_db_path.exists():
            for db_file in sorted(returns_db_path.glob("returns_db_*.json"), reverse=True):
                print(f"Loading tickers from existing database: {db_file}")
                with open(db_file, 'r') as f:
                    data = json.load(f)
                ticker_list = list(data.get("tickers", {}).keys())
                break
        else:
            print("ERROR: Universe file not found and no existing returns database")
            print(f"Create {universe_file} with your ticker list or provide --universe-file")
            return
    else:
        with open(universe_file, 'r') as f:
            universe_data = json.load(f)
        ticker_list = universe_data.get("tickers", [])

    if not ticker_list:
        print("ERROR: No tickers in universe file")
        return

    print(f"Fetching returns data for {len(ticker_list)} tickers...")

    # Fetch data
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")

    returns_data = fetch_morningstar_returns(ticker_list, start_date, end_date)

    print(f"Successfully fetched data for {len(returns_data)} tickers")

    # Also fetch XBI benchmark
    print("\nFetching XBI benchmark data...")

    # Try existing database first
    returns_db_path = Path("data/returns")
    xbi_data = None
    if returns_db_path.exists():
        for db_file in sorted(returns_db_path.glob("returns_db_*.json"), reverse=True):
            xbi_data = {"XBI": load_xbi_from_returns_db(db_file)}
            break

    if not xbi_data:
        xbi_data = fetch_via_yfinance(["XBI"], start_date, end_date)

    # Save both
    output_dir = Path(args.output_dir)
    save_returns_data(returns_data, output_dir / "ticker_returns.json")
    save_returns_data(xbi_data, output_dir / "xbi_returns.json")

    print("\n✅ Data fetch complete")


if __name__ == "__main__":
    main()
