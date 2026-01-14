#!/usr/bin/env python3
"""
Universe Quality Filter for Biotech Screener

Filters universe to investable names by removing:
- Micro-caps (market cap < $100M)
- Illiquid stocks (avg volume < 50k shares/day)
- Penny stocks (price < $2)
- Delistings/distressed names

Usage:
    python scripts/filter_universe.py --input data/universe_322_biotech.csv --output data/universe_investable.csv
    python scripts/filter_universe.py --input data/universe_322_biotech.csv --dry-run

Author: Wake Robin Capital
Version: 1.0
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Try to import yfinance
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("Warning: yfinance not installed. Install with: pip install yfinance")


# =============================================================================
# Configuration
# =============================================================================

# Quality filter thresholds
FILTERS = {
    'min_market_cap': 100_000_000,    # $100M minimum
    'min_price': 2.00,                 # $2 minimum (no penny stocks)
    'min_avg_volume': 50_000,          # 50k shares/day minimum
}

# Known problematic tickers to exclude
EXCLUDE_TICKERS = {
    'XBI',   # ETF, not a stock
    'IBB',   # ETF
    'LABU',  # Leveraged ETF
    'LABD',  # Leveraged ETF
}


# =============================================================================
# Quality Checks
# =============================================================================

def get_stock_quality(ticker: str) -> Dict:
    """
    Get quality metrics for a ticker.

    Returns dict with:
    - market_cap: Market capitalization in dollars
    - price: Current stock price
    - avg_volume: Average daily volume (3-month)
    - passes_all: True if passes all quality filters
    - failed_checks: List of failed checks
    """
    result = {
        'ticker': ticker,
        'market_cap': 0,
        'price': 0,
        'avg_volume': 0,
        'passes_all': False,
        'failed_checks': [],
        'error': None
    }

    if not HAS_YFINANCE:
        result['error'] = 'yfinance not installed'
        return result

    if ticker in EXCLUDE_TICKERS:
        result['failed_checks'].append('excluded_ticker')
        result['error'] = 'Ticker in exclusion list'
        return result

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Get metrics
        market_cap = info.get('marketCap', 0) or 0
        price = info.get('currentPrice') or info.get('regularMarketPrice') or 0

        # Get average volume from history
        try:
            hist = stock.history(period='3mo')
            avg_volume = hist['Volume'].mean() if not hist.empty else 0
        except:
            avg_volume = info.get('averageVolume', 0) or 0

        result['market_cap'] = market_cap
        result['price'] = price
        result['avg_volume'] = avg_volume

        # Check filters
        failed = []

        if market_cap < FILTERS['min_market_cap']:
            failed.append(f"market_cap ${market_cap/1e6:.1f}M < ${FILTERS['min_market_cap']/1e6:.0f}M")

        if price < FILTERS['min_price']:
            failed.append(f"price ${price:.2f} < ${FILTERS['min_price']:.2f}")

        if avg_volume < FILTERS['min_avg_volume']:
            failed.append(f"volume {avg_volume/1000:.0f}k < {FILTERS['min_avg_volume']/1000:.0f}k")

        if market_cap == 0 and price == 0:
            failed.append("no_data")

        result['failed_checks'] = failed
        result['passes_all'] = len(failed) == 0

    except Exception as e:
        result['error'] = str(e)
        result['failed_checks'].append('error')

    return result


def load_tickers(input_file: str) -> List[str]:
    """Load tickers from CSV file."""
    tickers = []

    with open(input_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader, None)

        # Find ticker column
        ticker_col = 0
        if header:
            lower_header = [h.lower() for h in header]
            if 'ticker' in lower_header:
                ticker_col = lower_header.index('ticker')
            elif 'symbol' in lower_header:
                ticker_col = lower_header.index('symbol')

        for row in reader:
            if row and len(row) > ticker_col:
                ticker = row[ticker_col].strip().upper()
                if ticker and ticker.isalpha() and len(ticker) <= 6:
                    tickers.append(ticker)

    return tickers


def filter_universe(
    tickers: List[str],
    delay: float = 0.3
) -> Tuple[List[str], List[Dict]]:
    """
    Filter universe to quality names only.

    Returns:
    - passed: List of tickers that pass all filters
    - results: Full results for all tickers
    """
    results = []
    passed = []
    failed = []

    total = len(tickers)
    print(f"\nChecking quality for {total} tickers...")
    print(f"Filters: market_cap >= ${FILTERS['min_market_cap']/1e6:.0f}M, "
          f"price >= ${FILTERS['min_price']:.2f}, "
          f"volume >= {FILTERS['min_avg_volume']/1000:.0f}k/day")
    print()

    for i, ticker in enumerate(tickers, 1):
        result = get_stock_quality(ticker)
        results.append(result)

        if result['passes_all']:
            passed.append(ticker)
            status = "✓ PASS"
        else:
            failed.append(ticker)
            status = f"✗ FAIL: {', '.join(result['failed_checks'])}"

        if i % 10 == 0 or i == total:
            print(f"  [{i}/{total}] {ticker}: {status}")

        time.sleep(delay)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Total checked: {total}")
    print(f"  Passed: {len(passed)} ({100*len(passed)/total:.1f}%)")
    print(f"  Failed: {len(failed)} ({100*len(failed)/total:.1f}%)")
    print()

    # Breakdown of failures
    failure_reasons = {}
    for r in results:
        for check in r['failed_checks']:
            reason = check.split()[0] if ' ' in check else check
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

    if failure_reasons:
        print("  Failure breakdown:")
        for reason, count in sorted(failure_reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")

    return passed, results


def save_filtered_universe(
    tickers: List[str],
    output_file: str,
    results: List[Dict] = None
):
    """Save filtered universe to CSV."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ticker'])
        for ticker in sorted(tickers):
            writer.writerow([ticker])

    print(f"\nSaved {len(tickers)} tickers to: {output_file}")

    # Also save detailed results
    if results:
        details_file = output_path.with_suffix('.details.json')
        with open(details_file, 'w', encoding='utf-8') as f:
            json.dump({
                'generated_at': datetime.now().isoformat(),
                'filters': FILTERS,
                'total_checked': len(results),
                'passed': len(tickers),
                'results': results
            }, f, indent=2)
        print(f"Saved details to: {details_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Filter universe to investable biotech stocks'
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input CSV file with tickers')
    parser.add_argument('--output', '-o', type=str,
                        help='Output CSV file for filtered universe')
    parser.add_argument('--dry-run', action='store_true',
                        help='Check quality but do not save output')
    parser.add_argument('--delay', type=float, default=0.3,
                        help='Delay between API calls (default: 0.3s)')
    parser.add_argument('--min-market-cap', type=float,
                        help=f'Override min market cap (default: ${FILTERS["min_market_cap"]/1e6:.0f}M)')
    parser.add_argument('--min-price', type=float,
                        help=f'Override min price (default: ${FILTERS["min_price"]:.2f})')
    parser.add_argument('--min-volume', type=float,
                        help=f'Override min volume (default: {FILTERS["min_avg_volume"]/1000:.0f}k)')

    args = parser.parse_args()

    # Override filters if specified
    if args.min_market_cap:
        FILTERS['min_market_cap'] = args.min_market_cap * 1_000_000
    if args.min_price:
        FILTERS['min_price'] = args.min_price
    if args.min_volume:
        FILTERS['min_avg_volume'] = args.min_volume * 1000

    # Load tickers
    tickers = load_tickers(args.input)
    print(f"Loaded {len(tickers)} tickers from {args.input}")

    # Filter
    passed, results = filter_universe(tickers, delay=args.delay)

    # Save
    if not args.dry_run and args.output:
        save_filtered_universe(passed, args.output, results)
    elif args.dry_run:
        print("\n[DRY RUN] Would save to:", args.output or "data/universe_investable.csv")
        print(f"[DRY RUN] Investable tickers: {', '.join(sorted(passed)[:20])}...")


if __name__ == "__main__":
    main()
