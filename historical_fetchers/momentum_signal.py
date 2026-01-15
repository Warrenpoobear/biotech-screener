#!/usr/bin/env python3
"""
Momentum Signal Calculator for Biotech Screener

Calculates 3-month price momentum as an additional scoring factor.

Momentum is one of the most robust factors in equity investing:
- Stocks that have been rising tend to continue rising
- Stocks that have been falling tend to continue falling

Scoring (penalty-style, lower = better):
- Strong positive momentum (>+20%): score = 10 (best)
- Positive momentum (+5% to +20%): score = 20
- Neutral momentum (-5% to +5%): score = 30
- Negative momentum (<-5%): score = 40 (worst)

Usage:
    from momentum_signal import get_momentum_score, get_momentum_batch

    # Single ticker
    result = get_momentum_score('VRTX', as_of_date='2024-01-15')

    # Batch
    results = get_momentum_batch(['VRTX', 'REGN', 'ALNY'], as_of_date='2024-01-15')

Author: Wake Robin Capital
Version: 1.0
"""

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

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

# Momentum lookback period (months)
LOOKBACK_MONTHS = 3

# Score thresholds
MOMENTUM_THRESHOLDS = {
    'STRONG_POSITIVE': 20,    # >= +20% = score 10
    'POSITIVE': 5,            # >= +5% = score 20
    'NEUTRAL_LOW': -5,        # >= -5% = score 30
    # < -5% = score 40
}

# Scores (penalty-style)
MOMENTUM_SCORES = {
    'STRONG_POSITIVE': 10,
    'POSITIVE': 20,
    'NEUTRAL': 30,
    'NEGATIVE': 40,
}

# Neutral score for unknown/error cases
NEUTRAL_SCORE = 30


# =============================================================================
# Momentum Calculation
# =============================================================================

def get_momentum_score(
    ticker: str,
    as_of_date: str = None,
    lookback_months: int = LOOKBACK_MONTHS
) -> Dict:
    """
    Calculate momentum score for a single ticker.

    Args:
        ticker: Stock ticker symbol
        as_of_date: End date for momentum calculation (YYYY-MM-DD)
        lookback_months: Number of months to look back (default: 3)

    Returns:
        Dict with:
        - ticker: Ticker symbol
        - momentum_pct: Percentage return over lookback period
        - momentum_bucket: 'STRONG_POSITIVE', 'POSITIVE', 'NEUTRAL', 'NEGATIVE'
        - momentum_score: Score (10, 20, 30, or 40)
        - start_price: Price at start of period
        - end_price: Price at end of period
        - confidence: 'HIGH', 'MEDIUM', 'LOW', or 'UNKNOWN'
    """
    result = {
        'ticker': ticker,
        'momentum_pct': 0,
        'momentum_bucket': 'NEUTRAL',
        'momentum_score': NEUTRAL_SCORE,
        'start_price': None,
        'end_price': None,
        'confidence': 'UNKNOWN',
        'error': None
    }

    if not HAS_YFINANCE:
        result['error'] = 'yfinance not installed'
        return result

    try:
        # Parse dates
        if as_of_date:
            end_date = datetime.strptime(as_of_date, '%Y-%m-%d')
        else:
            end_date = datetime.now()

        start_date = end_date - timedelta(days=lookback_months * 30)

        # Fetch historical data
        stock = yf.Ticker(ticker)
        hist = stock.history(
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d')
        )

        if hist.empty or len(hist) < 10:
            result['error'] = 'Insufficient history'
            result['confidence'] = 'UNKNOWN'
            return result

        # Calculate momentum
        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        momentum_pct = ((end_price / start_price) - 1) * 100

        result['start_price'] = float(start_price)
        result['end_price'] = float(end_price)
        result['momentum_pct'] = float(momentum_pct)

        # Determine bucket and score
        if momentum_pct >= MOMENTUM_THRESHOLDS['STRONG_POSITIVE']:
            bucket = 'STRONG_POSITIVE'
            score = MOMENTUM_SCORES['STRONG_POSITIVE']
        elif momentum_pct >= MOMENTUM_THRESHOLDS['POSITIVE']:
            bucket = 'POSITIVE'
            score = MOMENTUM_SCORES['POSITIVE']
        elif momentum_pct >= MOMENTUM_THRESHOLDS['NEUTRAL_LOW']:
            bucket = 'NEUTRAL'
            score = MOMENTUM_SCORES['NEUTRAL']
        else:
            bucket = 'NEGATIVE'
            score = MOMENTUM_SCORES['NEGATIVE']

        result['momentum_bucket'] = bucket
        result['momentum_score'] = score

        # Confidence based on data quality
        days_of_data = len(hist)
        if days_of_data >= 60:
            result['confidence'] = 'HIGH'
        elif days_of_data >= 30:
            result['confidence'] = 'MEDIUM'
        else:
            result['confidence'] = 'LOW'

    except Exception as e:
        result['error'] = str(e)

    return result


def get_momentum_batch(
    tickers: List[str],
    as_of_date: str = None,
    lookback_months: int = LOOKBACK_MONTHS,
    delay: float = 0.2
) -> Dict[str, Dict]:
    """
    Calculate momentum scores for multiple tickers.

    Args:
        tickers: List of ticker symbols
        as_of_date: End date for momentum calculation (YYYY-MM-DD)
        lookback_months: Number of months to look back
        delay: Delay between API calls (seconds)

    Returns:
        Dict mapping ticker -> momentum result
    """
    results = {}
    total = len(tickers)

    bucket_counts = {
        'STRONG_POSITIVE': 0,
        'POSITIVE': 0,
        'NEUTRAL': 0,
        'NEGATIVE': 0
    }

    print(f"\nCalculating {lookback_months}-month momentum for {total} tickers...")
    if as_of_date:
        print(f"As of date: {as_of_date}")

    for i, ticker in enumerate(tickers, 1):
        result = get_momentum_score(ticker, as_of_date, lookback_months)
        results[ticker] = result

        bucket = result['momentum_bucket']
        bucket_counts[bucket] += 1

        if i % 25 == 0 or i == total:
            print(f"  [{i}/{total}] {ticker}: {result['momentum_pct']:+.1f}% ({bucket})")

        time.sleep(delay)

    # Summary
    print(f"\n{'='*60}")
    print(f"MOMENTUM SUMMARY")
    print(f"{'='*60}")
    print(f"  STRONG_POSITIVE (>=+20%): {bucket_counts['STRONG_POSITIVE']} ({100*bucket_counts['STRONG_POSITIVE']/total:.1f}%)")
    print(f"  POSITIVE (+5% to +20%): {bucket_counts['POSITIVE']} ({100*bucket_counts['POSITIVE']/total:.1f}%)")
    print(f"  NEUTRAL (-5% to +5%): {bucket_counts['NEUTRAL']} ({100*bucket_counts['NEUTRAL']/total:.1f}%)")
    print(f"  NEGATIVE (<-5%): {bucket_counts['NEGATIVE']} ({100*bucket_counts['NEGATIVE']/total:.1f}%)")

    return results


# =============================================================================
# Composite Score Integration
# =============================================================================

def integrate_momentum_into_ranking(
    rankings_file: str,
    as_of_date: str,
    momentum_weight: float = 0.15,
    output_file: str = None
) -> List[Dict]:
    """
    Add momentum scores to existing rankings and recalculate composite.

    Args:
        rankings_file: Path to rankings.csv
        as_of_date: Date for momentum calculation
        momentum_weight: Weight for momentum in composite (default: 0.15)
        output_file: Output file for updated rankings (optional)

    Returns:
        List of updated ticker rankings
    """
    import csv

    # Load existing rankings
    rankings = []
    with open(rankings_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rankings.append(row)

    # Get tickers
    tickers = [r.get('Ticker', r.get('ticker', '')) for r in rankings]
    tickers = [t for t in tickers if t]

    # Calculate momentum
    momentum_data = get_momentum_batch(tickers, as_of_date)

    # Update rankings with momentum
    for r in rankings:
        ticker = r.get('Ticker', r.get('ticker', ''))
        if ticker in momentum_data:
            m = momentum_data[ticker]
            r['Momentum_Pct'] = f"{m['momentum_pct']:.1f}"
            r['Momentum_Score'] = str(m['momentum_score'])
            r['Momentum_Bucket'] = m['momentum_bucket']

            # Recalculate composite with momentum
            try:
                old_composite = float(r.get('Composite_Score', 30))
                momentum_score = float(m['momentum_score'])

                # Blend: reduce other factors to make room for momentum
                # Old: financial(50%) + clinical(50%)
                # New: financial(42.5%) + clinical(42.5%) + momentum(15%)
                scale_factor = 1 - momentum_weight
                new_composite = (old_composite * scale_factor) + (momentum_score * momentum_weight)

                r['Composite_Score_With_Momentum'] = f"{new_composite:.2f}"
            except (ValueError, TypeError):
                r['Composite_Score_With_Momentum'] = r.get('Composite_Score', '30')

    # Re-sort by new composite
    rankings.sort(key=lambda x: float(x.get('Composite_Score_With_Momentum', x.get('Composite_Score', 30))))

    # Update ranks
    for i, r in enumerate(rankings, 1):
        r['Rank_With_Momentum'] = str(i)

    # Save if output specified
    if output_file:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            if rankings:
                writer = csv.DictWriter(f, fieldnames=rankings[0].keys())
                writer.writeheader()
                writer.writerows(rankings)
        print(f"\nSaved updated rankings to: {output_file}")

    return rankings


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Calculate momentum scores for biotech tickers'
    )
    parser.add_argument('--ticker', type=str, help='Single ticker to check')
    parser.add_argument('--tickers', type=str, help='Comma-separated tickers')
    parser.add_argument('--file', type=str, help='CSV file with tickers')
    parser.add_argument('--as-of', type=str, help='As-of date (YYYY-MM-DD)')
    parser.add_argument('--months', type=int, default=3, help='Lookback months (default: 3)')
    parser.add_argument('--output', type=str, help='Output JSON file')
    parser.add_argument('--integrate', type=str,
                        help='Integrate momentum into existing rankings.csv')
    parser.add_argument('--weight', type=float, default=0.15,
                        help='Momentum weight in composite (default: 0.15)')

    args = parser.parse_args()

    # Collect tickers
    tickers = []
    if args.ticker:
        tickers = [args.ticker]
    elif args.tickers:
        tickers = [t.strip() for t in args.tickers.split(',')]
    elif args.file:
        import csv
        with open(args.file, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if row:
                    tickers.append(row[0].strip())

    # Integration mode
    if args.integrate:
        if not args.as_of:
            parser.error("--as-of required for integration mode")

        rankings = integrate_momentum_into_ranking(
            args.integrate,
            args.as_of,
            momentum_weight=args.weight,
            output_file=args.output or args.integrate.replace('.csv', '_with_momentum.csv')
        )
        return

    # Standard mode
    if not tickers:
        parser.error("Specify --ticker, --tickers, --file, or --integrate")

    if len(tickers) == 1:
        result = get_momentum_score(tickers[0], args.as_of, args.months)
        print(json.dumps(result, indent=2))
    else:
        results = get_momentum_batch(tickers, args.as_of, args.months)

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump({
                    'as_of_date': args.as_of or datetime.now().strftime('%Y-%m-%d'),
                    'lookback_months': args.months,
                    'tickers': results
                }, f, indent=2)
            print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
