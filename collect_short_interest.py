#!/usr/bin/env python3
"""
collect_short_interest.py - Short Interest Data Collector

Collects short interest data from Yahoo Finance for all tickers in the universe.

Fields collected:
- short_interest_pct: Short interest as % of float
- days_to_cover: Short ratio (days to cover based on avg volume)
- short_interest_change_pct: Change from prior month
- institutional_long_pct: Institutional ownership %
- avg_daily_volume: Average daily trading volume
- report_date: Date of short interest data

Usage:
    python collect_short_interest.py
    python collect_short_interest.py --universe production_data/universe.json
    python collect_short_interest.py --output production_data/short_interest.json
"""

import argparse
import json
import sys
import time
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any

# Check for yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


def get_short_interest_data(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Get short interest data for a single ticker from Yahoo Finance.

    Returns dict with:
        - ticker: Stock ticker
        - short_interest_pct: Short % of float (0-100 scale)
        - days_to_cover: Short ratio
        - short_interest_change_pct: Change from prior month
        - institutional_long_pct: Institutional ownership %
        - avg_daily_volume: Average daily volume
        - report_date: Date of short interest data (YYYY-MM-DD)
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Extract short interest fields
        short_pct_float = info.get('shortPercentOfFloat')
        short_ratio = info.get('shortRatio')
        shares_short = info.get('sharesShort')
        shares_short_prior = info.get('sharesShortPriorMonth')
        inst_pct = info.get('heldPercentInstitutions')
        avg_volume = info.get('averageVolume')
        date_short_interest = info.get('dateShortInterest')

        # Skip if no short interest data
        if short_pct_float is None and shares_short is None:
            return None

        # Calculate change percentage
        change_pct = None
        if shares_short and shares_short_prior and shares_short_prior > 0:
            change_pct = round(((shares_short - shares_short_prior) / shares_short_prior) * 100, 1)

        # Convert date from unix timestamp
        report_date = None
        if date_short_interest:
            try:
                report_date = datetime.fromtimestamp(date_short_interest).strftime('%Y-%m-%d')
            except (ValueError, OSError):
                pass

        # Use today's date if no report date available
        if not report_date:
            report_date = date.today().isoformat()

        return {
            'ticker': ticker,
            'short_interest_pct': round(short_pct_float * 100, 1) if short_pct_float else None,
            'days_to_cover': round(short_ratio, 1) if short_ratio else None,
            'short_interest_change_pct': change_pct,
            'institutional_long_pct': round(inst_pct * 100, 1) if inst_pct else None,
            'avg_daily_volume': avg_volume,
            'report_date': report_date,
        }

    except Exception as e:
        return None


def load_cached_data(output_file: Path) -> tuple[List[Dict], str]:
    """Load existing short interest data if available."""
    if not output_file.exists():
        return [], ""

    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not data or not isinstance(data, list):
            return [], ""

        # Get report date from first valid record
        report_date = ""
        for item in data:
            if isinstance(item, dict) and item.get('report_date'):
                report_date = item['report_date']
                break

        return data, report_date
    except Exception:
        return [], ""


def collect_short_interest(
    universe_file: Path,
    output_file: Path,
    force_refresh: bool = False,
) -> int:
    """
    Collect short interest data for all tickers in universe.

    Args:
        universe_file: Path to universe.json
        output_file: Path to output short_interest.json
        force_refresh: If True, ignore cache and refresh all data

    Returns:
        Exit code (0 for success, 1 for error)
    """
    print("=" * 80)
    print("SHORT INTEREST DATA COLLECTION (Yahoo Finance)")
    print("=" * 80)
    print(f"Date: {date.today()}")

    # Check yfinance
    if not YFINANCE_AVAILABLE:
        print("\n❌ ERROR: yfinance not installed")
        print("Install with: pip install yfinance")
        return 1

    print("✅ yfinance library found")

    # Check for cached data
    cached_data, cache_date = load_cached_data(output_file)
    cache_available = len(cached_data) > 0

    if cache_available:
        cache_age = (date.today() - date.fromisoformat(cache_date)).days if cache_date else -1
        print(f"📦 Cached data: {len(cached_data)} records from {cache_date} ({cache_age} days old)")

        if not force_refresh and cache_age >= 0 and cache_age < 7:
            print(f"\n⚠️  Cache is only {cache_age} days old.")
            print("   Short interest is reported bi-weekly.")
            print("   Use --force to refresh anyway.")

    # Load universe
    if not universe_file.exists():
        print(f"\n❌ ERROR: Universe file not found: {universe_file}")
        return 1

    with open(universe_file, 'r', encoding='utf-8') as f:
        universe = json.load(f)

    # Extract tickers
    if isinstance(universe, list):
        tickers = [s.get('ticker') for s in universe if s.get('ticker')]
    elif isinstance(universe, dict):
        securities = universe.get('active_securities', universe.get('securities', []))
        tickers = [s.get('ticker') for s in securities if s.get('ticker')]
    else:
        print(f"\n❌ ERROR: Invalid universe format")
        return 1

    print(f"\nUniverse: {len(tickers)} tickers")
    print(f"Output: {output_file}")
    print(f"Estimated time: ~{len(tickers) // 3} seconds")

    print("\n" + "=" * 80)
    print("COLLECTING SHORT INTEREST DATA")
    print("=" * 80 + "\n")

    # Collect data
    results = []
    success_count = 0
    error_count = 0

    for i, ticker in enumerate(tickers):
        data = get_short_interest_data(ticker)

        if data:
            results.append(data)
            success_count += 1
            si_pct = data.get('short_interest_pct')
            si_str = f"{si_pct:>5.1f}%" if si_pct else "  N/A"
            days = data.get('days_to_cover')
            days_str = f"{days:>4.1f}d" if days else " N/A"
            print(f"[{i+1:>3}/{len(tickers)}] {ticker:<6} ✅ SI: {si_str}  DTC: {days_str}")
        else:
            # Create placeholder record
            results.append({
                'ticker': ticker,
                'short_interest_pct': None,
                'days_to_cover': None,
                'short_interest_change_pct': None,
                'institutional_long_pct': None,
                'avg_daily_volume': None,
                'report_date': date.today().isoformat(),
            })
            error_count += 1
            print(f"[{i+1:>3}/{len(tickers)}] {ticker:<6} ⚠️  No data")

        # Rate limiting - Yahoo Finance can be sensitive
        if (i + 1) % 5 == 0:
            time.sleep(0.3)

    print("\n" + "=" * 80)
    print("COLLECTION SUMMARY")
    print("=" * 80)
    print(f"Total tickers:    {len(tickers)}")
    print(f"Data retrieved:   {success_count} ({100*success_count/len(tickers):.1f}%)")
    print(f"No data:          {error_count}")

    # Sort by ticker
    results.sort(key=lambda x: x.get('ticker', ''))

    # Calculate statistics
    si_values = [r['short_interest_pct'] for r in results if r.get('short_interest_pct')]
    if si_values:
        avg_si = sum(si_values) / len(si_values)
        max_si = max(si_values)
        high_si_count = sum(1 for v in si_values if v > 20)
        print(f"\nShort Interest Stats:")
        print(f"  Average SI%:    {avg_si:.1f}%")
        print(f"  Max SI%:        {max_si:.1f}%")
        print(f"  High SI (>20%): {high_si_count} tickers")

    # Write output
    print(f"\nWriting to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"✅ Saved {len(results)} records")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Collect short interest data from Yahoo Finance"
    )
    parser.add_argument(
        "--universe",
        type=Path,
        default=Path("production_data/universe.json"),
        help="Path to universe JSON file (default: production_data/universe.json)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("production_data/short_interest.json"),
        help="Output path for short interest data (default: production_data/short_interest.json)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force refresh even if cache is recent"
    )

    args = parser.parse_args()

    return collect_short_interest(
        universe_file=args.universe,
        output_file=args.output,
        force_refresh=args.force,
    )


if __name__ == "__main__":
    sys.exit(main())
