#!/usr/bin/env python3
"""
Calculate momentum signals for all tickers using daily returns data.

Usage:
    python scripts/calculate_momentum_batch.py
    python scripts/calculate_momentum_batch.py --returns-db data/returns/returns_db_daily.json
    python scripts/calculate_momentum_batch.py --output outputs/momentum_signals.json
"""

import argparse
import json
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.signals.morningstar_momentum_signals_v2 import MorningstarMomentumSignals


def load_daily_returns(returns_path: Path) -> tuple[dict, dict]:
    """
    Load daily returns from JSON database.

    Returns:
        (ticker_returns, xbi_returns) - both as {date: Decimal(return)}
    """
    with open(returns_path) as f:
        data = json.load(f)

    # Convert string returns to Decimal for precise calculations
    ticker_returns = {}
    for ticker, returns in data.get('tickers', {}).items():
        ticker_returns[ticker] = {
            date: Decimal(str(ret)) for date, ret in returns.items()
        }

    # Get XBI benchmark
    xbi_returns = {}
    benchmark = data.get('benchmark', {}).get('XBI', {})
    xbi_returns = {
        date: Decimal(str(ret)) for date, ret in benchmark.items()
    }

    return ticker_returns, xbi_returns


def main():
    parser = argparse.ArgumentParser(description='Calculate momentum signals for all tickers')
    parser.add_argument('--returns-db', type=str, default='data/returns/returns_db_daily.json',
                       help='Path to daily returns database')
    parser.add_argument('--output', type=str, default='outputs/momentum_signals.json',
                       help='Output file for momentum signals')
    parser.add_argument('--calc-date', type=str, default=None,
                       help='Calculation date (default: today)')
    args = parser.parse_args()

    returns_path = Path(args.returns_db)
    if not returns_path.exists():
        print(f"Error: Returns database not found: {returns_path}")
        print("Run scripts/fetch_daily_returns_yfinance.py first to fetch daily data")
        sys.exit(1)

    # Load returns data
    print(f"Loading returns from: {returns_path}")
    ticker_returns, xbi_returns = load_daily_returns(returns_path)

    print(f"  Loaded {len(ticker_returns)} tickers")
    print(f"  XBI benchmark: {len(xbi_returns)} days")

    # Calculation date
    calc_date = args.calc_date or datetime.now().strftime('%Y-%m-%d')
    print(f"  Calculation date: {calc_date}")

    # Initialize momentum calculator
    momentum_calc = MorningstarMomentumSignals()

    # Calculate signals for all tickers
    print(f"\nCalculating momentum signals...")
    results = {}
    errors = []

    for i, (ticker, returns) in enumerate(sorted(ticker_returns.items())):
        try:
            signals = momentum_calc.calculate_all_signals(
                ticker=ticker,
                returns_data=returns,
                xbi_returns_data=xbi_returns,
                calc_date=calc_date
            )

            # Convert Decimal to float for JSON serialization
            results[ticker] = {
                'composite_momentum_score': float(signals['composite_momentum_score']),
                'confidence_tier': signals['confidence_tier'],
                'confidence_multiplier': float(signals['confidence_multiplier']),
                'multi_horizon_sharpe': {
                    k: float(v) if isinstance(v, Decimal) else v
                    for k, v in signals['multi_horizon_sharpe'].items()
                },
                'relative_strength_vs_xbi': {
                    k: float(v) if isinstance(v, Decimal) else v
                    for k, v in signals['relative_strength_vs_xbi'].items()
                },
                'idiosyncratic_momentum': {
                    k: float(v) if isinstance(v, Decimal) else v
                    for k, v in signals['idiosyncratic_momentum'].items()
                },
                'drawdown_gate': {
                    k: float(v) if isinstance(v, Decimal) else v
                    for k, v in signals['drawdown_gate'].items()
                },
            }

        except Exception as e:
            errors.append((ticker, str(e)))
            continue

        # Progress update
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(ticker_returns)} tickers...")

    print(f"\n=== Results Summary ===")
    print(f"Successful: {len(results)}")
    print(f"Errors: {len(errors)}")

    if errors:
        print(f"\nFirst 10 errors:")
        for ticker, err in errors[:10]:
            print(f"  {ticker}: {err[:60]}")

    # Sort by momentum score
    sorted_by_score = sorted(
        results.items(),
        key=lambda x: x[1]['composite_momentum_score'],
        reverse=True
    )

    print(f"\nTop 20 by Momentum Score:")
    print("-" * 60)
    for ticker, signals in sorted_by_score[:20]:
        score = signals['composite_momentum_score']
        tier = signals['confidence_tier']
        print(f"  {ticker:6s}  Score: {score:5.1f}  Confidence: {tier}")

    print(f"\nBottom 10 by Momentum Score:")
    print("-" * 60)
    for ticker, signals in sorted_by_score[-10:]:
        score = signals['composite_momentum_score']
        tier = signals['confidence_tier']
        print(f"  {ticker:6s}  Score: {score:5.1f}  Confidence: {tier}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        'metadata': {
            'calculation_date': calc_date,
            'generated_at': datetime.now().isoformat(),
            'returns_source': str(returns_path),
            'total_tickers': len(results),
            'errors': len(errors)
        },
        'signals': results,
        'ranked_list': [
            {'ticker': t, 'momentum_score': s['composite_momentum_score'], 'confidence': s['confidence_tier']}
            for t, s in sorted_by_score
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to: {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
