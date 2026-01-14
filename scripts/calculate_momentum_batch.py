#!/usr/bin/env python3
"""
Calculate momentum signals for entire biotech universe.

Run this weekly (Tuesdays) as part of your screening workflow.
"""

from decimal import Decimal
from pathlib import Path
import json
from datetime import datetime
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.signals.morningstar_momentum_signals_v2 import MorningstarMomentumSignals


def load_returns_data(filepath):
    """Load returns data from JSON."""
    with open(filepath, 'r') as f:
        data_json = json.load(f)

    # Convert strings back to Decimal
    data_decimal = {}
    for ticker, returns in data_json.items():
        data_decimal[ticker] = {
            date: Decimal(ret)
            for date, ret in returns.items()
        }

    return data_decimal


def load_from_returns_db(db_file):
    """Load returns from existing Morningstar returns database format."""
    with open(db_file, 'r') as f:
        data = json.load(f)

    ticker_returns = {}

    # Handle both 'tickers' and 'returns' keys
    returns_data = data.get("tickers", data.get("returns", {}))

    for ticker, returns in returns_data.items():
        if isinstance(returns, list):
            # List of {date, return} dicts (monthly format)
            ticker_returns[ticker] = {
                _normalize_date(item["date"]): Decimal(str(item["return"] / 100))  # Convert % to decimal
                for item in returns
                if "date" in item and "return" in item
            }
        elif isinstance(returns, dict):
            # Dict format {date: return}
            ticker_returns[ticker] = {
                date: Decimal(str(ret))
                for date, ret in returns.items()
            }

    # Load XBI benchmark
    xbi_returns = {}
    xbi_data = data.get("benchmark", {}).get("XBI", [])

    if isinstance(xbi_data, list):
        # List of {date, return} dicts
        xbi_returns = {
            _normalize_date(item["date"]): Decimal(str(item["return"] / 100))  # Convert % to decimal
            for item in xbi_data
            if "date" in item and "return" in item
        }
    elif isinstance(xbi_data, dict):
        xbi_returns = {
            date: Decimal(str(ret))
            for date, ret in xbi_data.items()
        }

    return ticker_returns, xbi_returns


def _normalize_date(date_str):
    """Normalize date string to YYYY-MM-DD format."""
    if "T" in date_str:
        return date_str.split("T")[0]
    return date_str


def calculate_universe_momentum(calc_date, data_source=None):
    """
    Calculate momentum signals for entire universe.

    Args:
        calc_date: Calculation date (YYYY-MM-DD)
        data_source: Optional path to data source (returns_db or momentum dir)

    Returns:
        dict: {ticker: momentum_signals}
    """
    print(f"Calculating momentum signals as-of {calc_date}")
    print("=" * 70)

    # Load data - try multiple sources
    print("\n1. Loading returns data...")

    ticker_returns_all = None
    xbi_returns = None

    # Try existing returns database first
    returns_db_path = Path("data/returns")
    if returns_db_path.exists():
        for db_file in sorted(returns_db_path.glob("returns_db_*.json"), reverse=True):
            print(f"   Loading from: {db_file}")
            ticker_returns_all, xbi_returns = load_from_returns_db(db_file)
            break

    # Try momentum data directory
    if not ticker_returns_all:
        momentum_dir = Path(data_source) if data_source else Path("data/momentum")
        ticker_file = momentum_dir / "ticker_returns.json"
        xbi_file = momentum_dir / "xbi_returns.json"

        if ticker_file.exists() and xbi_file.exists():
            print(f"   Loading from: {momentum_dir}")
            ticker_returns_all = load_returns_data(ticker_file)
            xbi_data = load_returns_data(xbi_file)
            xbi_returns = xbi_data.get("XBI", {})

    if not ticker_returns_all:
        print("ERROR: No returns data found")
        print("Run scripts/fetch_morningstar_data.py first")
        return {}

    print(f"   Loaded {len(ticker_returns_all)} tickers")
    print(f"   XBI benchmark: {len(xbi_returns)} days")

    # Calculate signals
    print("\n2. Calculating momentum signals...")
    calculator = MorningstarMomentumSignals()
    results = {}

    confidence_stats = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN": 0}
    errors = []

    for i, ticker in enumerate(sorted(ticker_returns_all.keys()), 1):
        if i % 25 == 0:
            print(f"   Progress: {i}/{len(ticker_returns_all)} ({i/len(ticker_returns_all)*100:.1f}%)")

        try:
            ticker_returns = ticker_returns_all[ticker]

            signals = calculator.calculate_all_signals(
                ticker, ticker_returns, xbi_returns, calc_date
            )

            results[ticker] = signals
            confidence_stats[signals["confidence_tier"]] += 1

        except Exception as e:
            errors.append((ticker, str(e)))
            continue

    print(f"\n3. Completed {len(results)}/{len(ticker_returns_all)} tickers")

    if errors:
        print(f"   Errors: {len(errors)}")
        for ticker, err in errors[:5]:
            print(f"     {ticker}: {err}")
        if len(errors) > 5:
            print(f"     ... and {len(errors) - 5} more")

    print("\n   Confidence Distribution:")
    for tier in ["HIGH", "MEDIUM", "LOW", "UNKNOWN"]:
        count = confidence_stats[tier]
        pct = count / len(results) * 100 if results else 0
        print(f"     {tier:8s}: {count:3d} ({pct:5.1f}%)")

    # Save results
    output_dir = Path("data/momentum/signals")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"momentum_signals_{calc_date}.json"

    # Convert Decimals to strings
    results_json = {}
    for ticker, signals in results.items():
        results_json[ticker] = _convert_decimals_recursive(signals)

    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\n4. Results saved to: {output_file}")

    # Save audit trail
    audit_file = output_dir / f"momentum_audit_{calc_date}.json"
    calculator.write_audit_trail(audit_file)
    print(f"   Audit trail: {audit_file}")

    # Print top 10 by momentum score
    print("\n5. Top 10 by Momentum Score:")
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]["composite_momentum_score"],
        reverse=True
    )[:10]

    for rank, (ticker, signals) in enumerate(sorted_results, 1):
        score = signals["composite_momentum_score"]
        conf = signals["confidence_tier"]
        sharpe = signals["multi_horizon_sharpe"]["composite"]
        print(f"   {rank:2d}. {ticker:6s}  Score={score:6.2f}  Conf={conf:7s}  Sharpe={sharpe:+7.2f}")

    return results


def _convert_decimals_recursive(obj):
    """Recursively convert Decimals to strings."""
    if isinstance(obj, Decimal):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: _convert_decimals_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_decimals_recursive(item) for item in obj]
    else:
        return obj


def main():
    """Run momentum calculation."""
    import argparse

    parser = argparse.ArgumentParser(description="Calculate momentum signals")
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Calculation date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default=None,
        help="Path to data source directory"
    )

    args = parser.parse_args()

    results = calculate_universe_momentum(args.date, args.data_source)

    print("\n" + "=" * 70)
    print(f"✅ Momentum calculation complete: {len(results)} tickers")
    print("=" * 70)


if __name__ == "__main__":
    main()
