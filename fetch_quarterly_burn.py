#!/usr/bin/env python3
"""
Fetch Quarterly Burn Rate History

Fetches quarterly operating cash flow data from Yahoo Finance to enable
cash burn trajectory analysis.

Usage:
    python fetch_quarterly_burn.py [--output PATH] [--tickers TICKER...]

Requirements:
    pip install yfinance
"""

import argparse
import json
import sys
import time
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


def load_universe_tickers(universe_file: str = 'production_data/universe.json') -> List[str]:
    """Load tickers from universe file."""
    with open(universe_file) as f:
        data = json.load(f)

    tickers = []
    for item in data:
        ticker = item.get('ticker', '')
        if ticker and not ticker.startswith('_') and not ticker.endswith('_'):
            tickers.append(ticker)

    return sorted(set(tickers))


def fetch_quarterly_cashflow(ticker: str) -> Optional[List[Dict[str, Any]]]:
    """
    Fetch quarterly cash flow data for a ticker.

    Returns list of {period_end, operating_cash_flow} dicts in $MM.
    """
    try:
        stock = yf.Ticker(ticker)

        # Get quarterly cash flow statement
        cf = stock.quarterly_cashflow

        if cf is None or cf.empty:
            return None

        # Look for operating cash flow row
        # Common names: 'Operating Cash Flow', 'Total Cash From Operating Activities',
        # 'Cash Flow From Operating Activities', 'Net Cash Provided By Operating Activities'
        ocf_row = None
        for row_name in cf.index:
            row_lower = str(row_name).lower()
            if 'operating' in row_lower and ('cash' in row_lower or 'flow' in row_lower):
                ocf_row = row_name
                break
            if 'cash from operating' in row_lower:
                ocf_row = row_name
                break

        if ocf_row is None:
            # Try 'Free Cash Flow' as fallback
            for row_name in cf.index:
                if 'free cash flow' in str(row_name).lower():
                    ocf_row = row_name
                    break

        if ocf_row is None:
            return None

        quarterly_data = []
        for col in cf.columns:
            period_end = col
            if hasattr(period_end, 'strftime'):
                period_str = period_end.strftime('%Y-%m-%d')
            else:
                period_str = str(period_end)[:10]

            value = cf.loc[ocf_row, col]

            # Skip NaN values
            if value is None or (hasattr(value, 'isna') and value.isna()):
                continue

            # Convert to millions
            ocf_mm = float(value) / 1_000_000

            quarterly_data.append({
                'period_end': period_str,
                'operating_cash_flow': round(ocf_mm, 2)
            })

        # Sort by period_end descending (most recent first)
        quarterly_data.sort(key=lambda x: x['period_end'], reverse=True)

        return quarterly_data if quarterly_data else None

    except Exception as e:
        print(f"    Error: {e}")
        return None


def fetch_all_quarterly_burn(
    tickers: List[str],
    delay: float = 0.3,
    progress: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Fetch quarterly burn data for all tickers.

    Args:
        tickers: List of ticker symbols
        delay: Delay between requests in seconds
        progress: Show progress output

    Returns:
        Dict keyed by ticker with quarterly burn data
    """
    burn_history = {}
    success_count = 0
    failed_tickers = []
    start_time = time.time()

    for i, ticker in enumerate(tickers, 1):
        if progress:
            elapsed = time.time() - start_time
            if i > 1:
                avg_time = elapsed / (i - 1)
                remaining = avg_time * (len(tickers) - i + 1)
                eta = f"{int(remaining // 60)}m {int(remaining % 60)}s"
            else:
                eta = "calculating..."

            print(f"[{i:3d}/{len(tickers)}] {ticker:8s} (ETA: {eta})...", end='', flush=True)

        quarterly_data = fetch_quarterly_cashflow(ticker)

        if quarterly_data:
            burn_history[ticker] = {
                'ticker': ticker,
                'quarterly_burn': quarterly_data,
                'data_source': 'yahoo_finance',
                'last_updated': datetime.now().strftime('%Y-%m-%d')
            }
            success_count += 1
            if progress:
                # Show burn trend
                if len(quarterly_data) >= 2:
                    recent = quarterly_data[0]['operating_cash_flow']
                    prior = quarterly_data[1]['operating_cash_flow']
                    trend = "↓" if recent > prior else "↑" if recent < prior else "→"
                    print(f" OK ({len(quarterly_data)} quarters, {recent:.1f}MM {trend})")
                else:
                    print(f" OK ({len(quarterly_data)} quarters)")
        else:
            failed_tickers.append(ticker)
            if progress:
                print(" No data")

        if delay > 0 and i < len(tickers):
            time.sleep(delay)

    if progress:
        print(f"\nComplete: {success_count}/{len(tickers)} tickers")
        if failed_tickers:
            print(f"Failed: {', '.join(failed_tickers[:20])}")
            if len(failed_tickers) > 20:
                print(f"  ... and {len(failed_tickers) - 20} more")

    return burn_history


def main():
    parser = argparse.ArgumentParser(
        description='Fetch quarterly cash flow data for burn trajectory analysis'
    )
    parser.add_argument(
        '--output', '-o',
        default='production_data/quarterly_burn_history.json',
        help='Output file path (default: production_data/quarterly_burn_history.json)'
    )
    parser.add_argument(
        '--tickers', '-t',
        nargs='+',
        help='Specific tickers to fetch (default: all universe tickers)'
    )
    parser.add_argument(
        '--universe', '-u',
        default='production_data/universe.json',
        help='Universe file to load tickers from'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.3,
        help='Delay between API requests in seconds (default: 0.3)'
    )
    parser.add_argument(
        '--merge',
        action='store_true',
        help='Merge with existing data instead of replacing'
    )

    args = parser.parse_args()

    if not YFINANCE_AVAILABLE:
        print("ERROR: yfinance not installed")
        print("Install with: pip install yfinance")
        sys.exit(1)

    # Get tickers
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
        print(f"Fetching data for {len(tickers)} specified tickers")
    else:
        tickers = load_universe_tickers(args.universe)
        print(f"Loaded {len(tickers)} tickers from {args.universe}")

    # Fetch data
    print(f"\nFetching quarterly cash flow data...")
    print(f"Output: {args.output}")
    print()

    burn_history = fetch_all_quarterly_burn(tickers, delay=args.delay)

    # Merge with existing if requested
    if args.merge and Path(args.output).exists():
        with open(args.output) as f:
            existing = json.load(f)
        existing_burn = existing.get('burn_history', {})
        # Update existing with new data
        existing_burn.update(burn_history)
        burn_history = existing_burn
        print(f"\nMerged with existing data ({len(burn_history)} total tickers)")

    # Get quarters included
    all_quarters = set()
    for data in burn_history.values():
        for q in data.get('quarterly_burn', []):
            all_quarters.add(q['period_end'])

    # Build output
    output = {
        'metadata': {
            'description': 'Quarterly operating cash flow history for burn trajectory analysis',
            'units': 'USD millions (negative = cash burn, positive = cash generation)',
            'quarters_included': sorted(all_quarters, reverse=True)[:8],
            'ticker_count': len(burn_history),
            'generated_date': datetime.now().strftime('%Y-%m-%d'),
            'data_source': 'yahoo_finance',
            'version': '1.0.0'
        },
        'burn_history': burn_history
    }

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {len(burn_history)} tickers to {args.output}")

    # Show summary statistics
    decelerating = 0
    accelerating = 0
    stable = 0
    insufficient = 0

    for data in burn_history.values():
        quarters = data.get('quarterly_burn', [])
        if len(quarters) >= 2:
            recent = quarters[0]['operating_cash_flow']
            prior = quarters[1]['operating_cash_flow']

            # Calculate change (for burn, more negative = accelerating)
            if prior != 0:
                change_pct = (recent - prior) / abs(prior)
                if change_pct <= -0.15:  # Burn decreased by 15%+ (good)
                    decelerating += 1
                elif change_pct >= 0.15:  # Burn increased by 15%+ (bad)
                    accelerating += 1
                else:
                    stable += 1
            else:
                stable += 1
        else:
            insufficient += 1

    print(f"\nTrajectory preview:")
    print(f"  Decelerating: {decelerating}")
    print(f"  Stable: {stable}")
    print(f"  Accelerating: {accelerating}")
    print(f"  Insufficient data: {insufficient}")


if __name__ == '__main__':
    main()
