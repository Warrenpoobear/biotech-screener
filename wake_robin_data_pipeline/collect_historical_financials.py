#!/usr/bin/env python3
"""
collect_historical_financials.py - Collect 5 years of historical SEC financial data

This script fetches historical financial snapshots from SEC EDGAR for all tickers
in the universe, enabling proper point-in-time (PIT) backtesting.

Usage:
    python collect_historical_financials.py
"""
import json
import sys
from pathlib import Path
from datetime import datetime

# Add collectors to path
sys.path.insert(0, str(Path(__file__).parent))

from collectors import sec_collector


def load_universe():
    """Load ticker list from full universe."""
    universe_file = Path(__file__).parent / "universe" / "full_universe.json"
    if not universe_file.exists():
        # Fall back to pilot
        universe_file = Path(__file__).parent / "universe" / "pilot_universe.json"

    with open(universe_file) as f:
        data = json.load(f)

    return [t['ticker'] for t in data.get('tickers', [])]


def main():
    print("\n" + "=" * 60)
    print("HISTORICAL FINANCIAL DATA COLLECTION (5 YEARS)")
    print("=" * 60)

    # Load universe
    tickers = load_universe()
    print(f"\nLoaded {len(tickers)} tickers from universe")

    # Collect historical data
    results = sec_collector.collect_historical_batch(tickers, years=5, delay_seconds=0.5)

    # Save results
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save full results
    full_file = output_dir / f"historical_financials_{timestamp}.json"
    with open(full_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved full results to: {full_file.name}")

    # Create flattened records for backtest
    all_snapshots = []
    for ticker, data in results.items():
        if data.get('success'):
            for snapshot in data.get('snapshots', []):
                all_snapshots.append(snapshot)

    # Sort by date
    all_snapshots.sort(key=lambda x: (x['date'], x['ticker']))

    # Save flattened snapshots
    snapshots_file = output_dir / f"financial_snapshots_{timestamp}.json"
    with open(snapshots_file, 'w') as f:
        json.dump(all_snapshots, f, indent=2)
    print(f"✓ Saved {len(all_snapshots)} snapshots to: {snapshots_file.name}")

    # Also save to production_data for backtest
    prod_data_dir = Path(__file__).parent.parent / "production_data"
    prod_data_dir.mkdir(exist_ok=True)

    prod_file = prod_data_dir / "historical_financial_snapshots.json"
    with open(prod_file, 'w') as f:
        json.dump(all_snapshots, f, indent=2)
    print(f"✓ Copied to: {prod_file}")

    # Summary
    print("\n" + "=" * 60)
    print("COLLECTION SUMMARY")
    print("=" * 60)
    successful = sum(1 for d in results.values() if d.get('success'))
    print(f"Tickers processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Total snapshots: {len(all_snapshots)}")

    # Date range
    if all_snapshots:
        dates = [s['date'] for s in all_snapshots]
        print(f"Date range: {min(dates)} to {max(dates)}")

    print("\n✅ Historical financial data collection complete!")
    print("\nTo run backtest with historical data:")
    print("  python run_backtest.py --use-production-scorer --start-date 2020-01-01 --end-date 2024-12-31")


if __name__ == "__main__":
    main()
