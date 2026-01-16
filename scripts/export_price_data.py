#!/usr/bin/env python3
"""
export_price_data.py - Export historical price data for validation

Generates CSV files for Stage 2 momentum validation:
- data/universe_prices.csv: Daily prices for all universe tickers
- data/indices_prices.csv: XBI and SPY benchmark prices

Uses existing data infrastructure (Morningstar with yfinance fallback).

Usage:
    python scripts/export_price_data.py
    python scripts/export_price_data.py --lookback 504  # 2 years
    python scripts/export_price_data.py --output-dir validation_data
"""

import sys
import json
import csv
from pathlib import Path
from datetime import date, timedelta
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import data providers
try:
    from wake_robin_data_pipeline.market_data_provider import PriceDataProvider
    HAS_MARKET_DATA = True
except ImportError:
    HAS_MARKET_DATA = False

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


def load_universe_tickers() -> List[str]:
    """Load tickers from universe files."""
    tickers = set()

    # Check multiple universe file locations (full universe first!)
    universe_paths = [
        Path("production_data/universe.json"),  # Full 307 tickers - check first!
        Path("data/universe.json"),
        Path("wake_robin_data_pipeline/universe/biotech_universe.json"),
        Path("wake_robin_data_pipeline/universe/pilot_universe.json"),  # Small pilot last
    ]

    for path in universe_paths:
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)

                # Handle different formats
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, str):
                            tickers.add(item)
                        elif isinstance(item, dict) and 'ticker' in item:
                            tickers.add(item['ticker'])
                elif isinstance(data, dict):
                    if 'tickers' in data:
                        for item in data['tickers']:
                            if isinstance(item, str):
                                tickers.add(item)
                            elif isinstance(item, dict) and 'ticker' in item:
                                tickers.add(item['ticker'])
                    elif 'universe' in data:
                        for item in data['universe']:
                            if isinstance(item, str):
                                tickers.add(item)
                            elif isinstance(item, dict) and 'ticker' in item:
                                tickers.add(item['ticker'])

                print(f"   Loaded {len(tickers)} tickers from {path}")
            except Exception as e:
                print(f"   Warning: Could not load {path}: {e}")

    # Filter out internal tickers
    tickers = {t for t in tickers if not t.startswith('_')}

    return sorted(tickers)


def generate_sample_prices(tickers: List[str], n_days: int = 252) -> tuple:
    """Generate synthetic price data for testing validation framework."""
    import random
    random.seed(42)  # Deterministic

    print(f"   Generating synthetic data for {len(tickers)} tickers, {n_days} days...")
    print("   WARNING: This is SAMPLE DATA for testing only!")
    print("   Run on local machine with yfinance for real validation.")

    prices_by_ticker = {}
    dates = []

    # Generate dates (trading days only)
    current = date.today() - timedelta(days=int(n_days * 1.5))
    day_count = 0
    while day_count < n_days:
        if current.weekday() < 5:  # Mon-Fri
            dates.append(current.isoformat())
            day_count += 1
        current += timedelta(days=1)

    # Generate prices for each ticker
    for ticker in tickers:
        # Random starting price between 10 and 200
        price = random.uniform(10, 200)
        # Random daily volatility between 2% and 5%
        vol = random.uniform(0.02, 0.05)
        # Random drift between -10% and +30% annualized
        drift = random.uniform(-0.10, 0.30) / 252

        prices = []
        for _ in range(n_days):
            # Geometric Brownian Motion
            ret = drift + vol * random.gauss(0, 1)
            price = price * (1 + ret)
            prices.append(max(0.01, price))  # Floor at $0.01

        prices_by_ticker[ticker] = prices

    print(f"   Generated {len(prices_by_ticker)} tickers")
    return prices_by_ticker, dates


def generate_sample_indices(n_days: int = 252) -> tuple:
    """Generate synthetic XBI and SPY data."""
    import random
    random.seed(43)

    dates = []
    current = date.today() - timedelta(days=int(n_days * 1.5))
    day_count = 0
    while day_count < n_days:
        if current.weekday() < 5:
            dates.append(current.isoformat())
            day_count += 1
        current += timedelta(days=1)

    # XBI: Higher vol biotech ETF
    xbi = [100.0]
    for _ in range(n_days - 1):
        ret = 0.0005 + 0.025 * random.gauss(0, 1)
        xbi.append(xbi[-1] * (1 + ret))

    # SPY: Lower vol broad market
    spy = [450.0]
    for _ in range(n_days - 1):
        ret = 0.0003 + 0.012 * random.gauss(0, 1)
        spy.append(spy[-1] * (1 + ret))

    print(f"   Generated {len(dates)} days of synthetic benchmark data")
    return dates, xbi, spy


def fetch_prices_morningstar(tickers: List[str], start_date: date, end_date: date) -> tuple:
    """Fetch historical prices using Morningstar provider."""
    try:
        from wake_robin_data_pipeline.market_data_provider import PriceDataProvider
        provider = PriceDataProvider()
    except Exception as e:
        print(f"   Morningstar data provider not available: {e}")
        return generate_sample_prices(tickers)
    prices_by_ticker = {}
    all_dates = set()

    print(f"   Fetching {len(tickers)} tickers from Morningstar...")

    for i, ticker in enumerate(tickers):
        if (i + 1) % 20 == 0:
            print(f"   Progress: {i+1}/{len(tickers)}")
        try:
            prices = provider.get_prices(ticker, start_date, end_date)
            if prices:
                prices_by_ticker[ticker] = [p for _, p in sorted(prices.items())]
                all_dates.update(prices.keys())
        except Exception as e:
            pass  # Skip failed tickers

    dates = sorted(all_dates)
    print(f"   Successfully fetched {len(prices_by_ticker)} tickers, {len(dates)} days")
    return prices_by_ticker, [d.isoformat() if hasattr(d, 'isoformat') else str(d) for d in dates]


def fetch_prices_yfinance(tickers: List[str], start_date: date, end_date: date) -> tuple:
    """Fetch historical prices using yfinance."""
    if not HAS_YFINANCE:
        print("   yfinance not installed - trying Morningstar provider...")
        return fetch_prices_morningstar(tickers, start_date, end_date)

    prices_by_ticker = {}
    dates = []

    print(f"   Fetching {len(tickers)} tickers from Yahoo Finance...")

    # Download all at once for efficiency
    try:
        data = yf.download(
            tickers,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            progress=True,
            threads=True
        )

        if data.empty:
            print("   No data returned from Yahoo Finance")
            return {}

        # Extract close prices
        if 'Close' in data.columns:
            close_data = data['Close']
        elif 'Adj Close' in data.columns:
            close_data = data['Adj Close']
        else:
            close_data = data

        # Get dates
        dates = [d.strftime('%Y-%m-%d') for d in close_data.index]

        # Extract per-ticker prices
        if isinstance(close_data, dict) or hasattr(close_data, 'columns'):
            for ticker in tickers:
                if ticker in close_data.columns:
                    prices = close_data[ticker].tolist()
                    # Filter out NaN values
                    prices_by_ticker[ticker] = [p for p in prices if p == p]  # NaN != NaN
        else:
            # Single ticker case
            if len(tickers) == 1:
                prices_by_ticker[tickers[0]] = close_data.tolist()

        print(f"   Successfully fetched {len(prices_by_ticker)} tickers, {len(dates)} days")

    except Exception as e:
        print(f"   Error fetching batch: {e}")
        # Fall back to individual downloads
        for i, ticker in enumerate(tickers):
            if (i + 1) % 50 == 0:
                print(f"   Progress: {i+1}/{len(tickers)}")
            try:
                data = yf.download(ticker, start=start_date.isoformat(), end=end_date.isoformat(), progress=False)
                if not data.empty:
                    prices_by_ticker[ticker] = data['Close'].tolist()
                    if not dates:
                        dates = [d.strftime('%Y-%m-%d') for d in data.index]
            except Exception as e2:
                print(f"   Failed {ticker}: {e2}")

    return prices_by_ticker, dates


def fetch_indices_morningstar(start_date: date, end_date: date) -> tuple:
    """Fetch XBI and SPY prices using Morningstar."""
    try:
        from wake_robin_data_pipeline.market_data_provider import PriceDataProvider
        provider = PriceDataProvider()
    except Exception as e:
        print(f"   Morningstar not available: {e}")
        return generate_sample_indices()
    dates = []
    xbi = []
    spy = []

    try:
        xbi_prices = provider.get_prices('XBI', start_date, end_date)
        spy_prices = provider.get_prices('SPY', start_date, end_date)

        if xbi_prices and spy_prices:
            # Align dates
            common_dates = sorted(set(xbi_prices.keys()) & set(spy_prices.keys()))
            dates = [d.isoformat() if hasattr(d, 'isoformat') else str(d) for d in common_dates]
            xbi = [xbi_prices[d] for d in common_dates]
            spy = [spy_prices[d] for d in common_dates]
            print(f"   Loaded {len(dates)} days of benchmark data")

    except Exception as e:
        print(f"   Error fetching indices: {e}")

    return dates, xbi, spy


def fetch_indices_yfinance(start_date: date, end_date: date) -> tuple:
    """Fetch XBI and SPY prices."""
    if not HAS_YFINANCE:
        print("   Trying Morningstar for benchmarks...")
        return fetch_indices_morningstar(start_date, end_date)

    print("   Fetching XBI and SPY benchmarks...")

    try:
        data = yf.download(['XBI', 'SPY'], start=start_date.isoformat(), end=end_date.isoformat(), progress=False)

        if data.empty:
            return [], [], []

        dates = [d.strftime('%Y-%m-%d') for d in data.index]
        xbi = data['Close']['XBI'].tolist()
        spy = data['Close']['SPY'].tolist()

        print(f"   Loaded {len(dates)} days of benchmark data")
        return dates, xbi, spy

    except Exception as e:
        print(f"   Error fetching indices: {e}")
        return [], [], []


def export_universe_prices(prices_by_ticker: Dict, dates: List[str], output_path: Path):
    """Export universe prices to CSV."""
    if not prices_by_ticker or not dates:
        print(f"   No data to export for universe prices")
        return

    # Get tickers with complete data
    tickers = sorted(prices_by_ticker.keys())

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(['date'] + tickers)

        # Data rows
        for i, d in enumerate(dates):
            row = [d]
            for ticker in tickers:
                prices = prices_by_ticker.get(ticker, [])
                if i < len(prices):
                    row.append(f"{prices[i]:.4f}")
                else:
                    row.append('')
            writer.writerow(row)

    print(f"   Exported {len(tickers)} tickers, {len(dates)} days to {output_path}")


def export_indices_prices(dates: List[str], xbi: List[float], spy: List[float], output_path: Path):
    """Export index prices to CSV."""
    if not dates:
        print(f"   No data to export for indices")
        return

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['date', 'XBI', 'SPY'])

        for i, d in enumerate(dates):
            xbi_val = f"{xbi[i]:.4f}" if i < len(xbi) and xbi[i] == xbi[i] else ''
            spy_val = f"{spy[i]:.4f}" if i < len(spy) and spy[i] == spy[i] else ''
            writer.writerow([d, xbi_val, spy_val])

    print(f"   Exported {len(dates)} days to {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Export price data for validation")
    parser.add_argument('--lookback', type=int, default=252, help="Days of history (default: 252 = 1 year)")
    parser.add_argument('--output-dir', type=Path, default=Path('data'), help="Output directory")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("PRICE DATA EXPORT FOR VALIDATION")
    print("=" * 60)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load universe
    print("\n1. Loading universe tickers...")
    tickers = load_universe_tickers()

    if not tickers:
        print("   No tickers found in universe files")
        sys.exit(1)

    print(f"   Found {len(tickers)} tickers")

    # Calculate date range
    end_date = date.today()
    start_date = end_date - timedelta(days=int(args.lookback * 1.5))  # Extra buffer for weekends/holidays

    print(f"\n2. Fetching price data ({start_date} to {end_date})...")

    # Fetch universe prices
    prices_by_ticker, dates = fetch_prices_yfinance(tickers, start_date, end_date)

    # Fetch index prices
    print("\n3. Fetching benchmark indices...")
    idx_dates, xbi, spy = fetch_indices_yfinance(start_date, end_date)

    # Export CSVs
    print("\n4. Exporting CSV files...")

    universe_path = args.output_dir / "universe_prices.csv"
    indices_path = args.output_dir / "indices_prices.csv"

    export_universe_prices(prices_by_ticker, dates, universe_path)
    export_indices_prices(idx_dates, xbi, spy, indices_path)

    # Summary
    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  - {universe_path}")
    print(f"  - {indices_path}")
    print(f"\nRun Stage 2 validation:")
    print(f"  python scripts/validate_momentum_signal.py --full --data-dir {args.output_dir}")


if __name__ == "__main__":
    main()
