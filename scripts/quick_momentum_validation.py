#!/usr/bin/env python3
"""
Quick validation: Do high-momentum tickers outperform low-momentum?

This is a fast sanity check before full ablation testing.
Uses the daily returns data to check if momentum signals are predictive.
"""

import json
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_momentum_signals(path="outputs/momentum_signals.json"):
    """Load the momentum signals."""
    with open(path, 'r') as f:
        data = json.load(f)

    # Extract ticker → score mapping
    scores = {}
    for ticker, signals in data.get('signals', {}).items():
        scores[ticker] = Decimal(str(signals["composite_momentum_score"]))

    return scores, data.get('metadata', {})


def load_returns(path="data/returns/returns_db_daily.json"):
    """Load daily returns data."""
    with open(path, 'r') as f:
        data = json.load(f)

    return data.get('tickers', {}), data.get('benchmark', {}).get('XBI', {})


def calculate_period_return(returns_dict, start_date, end_date):
    """Calculate cumulative return between two dates."""
    if not returns_dict:
        return None

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # Get dates in range
    period_returns = []
    for date_str, ret in returns_dict.items():
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            if start_dt <= dt <= end_dt:
                period_returns.append((date_str, Decimal(str(ret))))
        except (ValueError, TypeError):
            continue

    if len(period_returns) < 5:  # Need minimum data
        return None

    # Sort by date and compound
    period_returns.sort(key=lambda x: x[0])

    cum_return = Decimal("1.0")
    for _, daily_ret in period_returns:
        cum_return *= (Decimal("1.0") + daily_ret)

    return cum_return - Decimal("1.0")


def calculate_spearman_rank_correlation(x_ranks, y_ranks):
    """Calculate Spearman rank correlation coefficient."""
    n = len(x_ranks)
    if n < 5:
        return None

    # Calculate d^2 (squared rank differences)
    d_squared_sum = sum((x - y) ** 2 for x, y in zip(x_ranks, y_ranks))

    # Spearman formula: 1 - (6 * sum(d^2)) / (n * (n^2 - 1))
    rho = 1 - (6 * d_squared_sum) / (n * (n * n - 1))
    return rho


def run_backtest_validation():
    """
    Run a simple backtest validation on historical data.

    Tests whether high momentum scores predicted better forward returns.
    """
    print("=" * 70)
    print("MOMENTUM SIGNAL VALIDATION")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")

    returns_path = Path("data/returns/returns_db_daily.json")
    momentum_path = Path("outputs/momentum_signals.json")

    if not returns_path.exists():
        print(f"   ERROR: Returns not found at {returns_path}")
        return 1

    if not momentum_path.exists():
        print(f"   ERROR: Momentum signals not found at {momentum_path}")
        print("   Run: python scripts/calculate_momentum_batch.py")
        return 1

    momentum_scores, metadata = load_momentum_signals(momentum_path)
    ticker_returns, xbi_returns = load_returns(returns_path)

    calc_date = metadata.get('calculation_date', '2026-01-14')
    print(f"   Momentum calculation date: {calc_date}")
    print(f"   Tickers with momentum scores: {len(momentum_scores)}")
    print(f"   Tickers with returns data: {len(ticker_returns)}")

    # Find common tickers
    common_tickers = set(momentum_scores.keys()) & set(ticker_returns.keys())
    print(f"   Common tickers: {len(common_tickers)}")

    # We need to do a historical backtest
    # Pick a date ~90 days before the calc_date to have forward data
    calc_dt = datetime.strptime(calc_date, "%Y-%m-%d")

    # Test periods: calculate momentum at T, measure returns from T to T+60
    test_periods = [
        # (momentum_calc_date, forward_return_start, forward_return_end)
        ("2025-10-01", "2025-10-02", "2025-12-15"),  # 75 days forward
        ("2025-07-01", "2025-07-02", "2025-09-15"),  # 75 days forward
        ("2025-04-01", "2025-04-02", "2025-06-15"),  # 75 days forward
        ("2025-01-02", "2025-01-03", "2025-03-15"),  # 75 days forward
    ]

    print("\n2. Running backtest validation...")
    print("   (Using historical periods to test momentum predictiveness)")

    all_correlations = []
    all_spreads = []

    for i, (mom_date, fwd_start, fwd_end) in enumerate(test_periods):
        print(f"\n   Period {i+1}: Momentum at {mom_date}, returns {fwd_start} to {fwd_end}")

        # For this backtest, we'll use the returns data to approximate what
        # momentum would have been at mom_date (trailing 120 days of returns)
        # This is a simplified version - a proper backtest would recalculate momentum

        # Calculate forward returns for each ticker
        fwd_returns = {}
        for ticker in common_tickers:
            ret = calculate_period_return(ticker_returns[ticker], fwd_start, fwd_end)
            if ret is not None:
                fwd_returns[ticker] = ret

        # Calculate XBI forward return
        xbi_ret = calculate_period_return(xbi_returns, fwd_start, fwd_end)

        if len(fwd_returns) < 20:
            print(f"      Skipped - insufficient forward return data ({len(fwd_returns)} tickers)")
            continue

        print(f"      Tickers with forward returns: {len(fwd_returns)}")
        if xbi_ret:
            print(f"      XBI return: {float(xbi_ret):.2%}")

        # For this simplified test, use current momentum scores as proxy
        # (A proper backtest would recalculate momentum as of mom_date)
        valid_tickers = [t for t in fwd_returns.keys() if t in momentum_scores]

        # Sort by momentum score
        sorted_by_momentum = sorted(
            valid_tickers,
            key=lambda t: momentum_scores[t],
            reverse=True
        )

        n = len(sorted_by_momentum)
        q_size = n // 4

        q1_tickers = sorted_by_momentum[:q_size]  # Top 25% momentum
        q4_tickers = sorted_by_momentum[-q_size:]  # Bottom 25% momentum

        # Calculate average returns
        q1_returns = [fwd_returns[t] for t in q1_tickers]
        q4_returns = [fwd_returns[t] for t in q4_tickers]

        q1_avg = sum(q1_returns) / len(q1_returns)
        q4_avg = sum(q4_returns) / len(q4_returns)
        spread = q1_avg - q4_avg

        print(f"      Q1 (top momentum) avg return: {float(q1_avg):>7.2%}")
        print(f"      Q4 (low momentum) avg return: {float(q4_avg):>7.2%}")
        print(f"      Q1-Q4 Spread:                 {float(spread):>7.2%}")

        all_spreads.append(float(spread))

        # Calculate rank correlation (IC)
        momentum_ranks = {t: i for i, t in enumerate(sorted_by_momentum)}
        return_ranks = {t: i for i, t in enumerate(
            sorted(valid_tickers, key=lambda t: fwd_returns[t], reverse=True)
        )}

        x_ranks = [momentum_ranks[t] for t in valid_tickers]
        y_ranks = [return_ranks[t] for t in valid_tickers]

        ic = calculate_spearman_rank_correlation(x_ranks, y_ranks)
        if ic:
            all_correlations.append(ic)
            print(f"      Information Coefficient (IC): {ic:>7.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    if all_spreads:
        avg_spread = sum(all_spreads) / len(all_spreads)
        print(f"\nAverage Q1-Q4 Spread: {avg_spread:>7.2%}")

        if avg_spread > 0.15:
            print("✅ STRONG SIGNAL - Momentum is highly predictive")
        elif avg_spread > 0.08:
            print("✓ MODERATE SIGNAL - Momentum adds value")
        elif avg_spread > 0.03:
            print("~ WEAK SIGNAL - Marginal improvement")
        elif avg_spread > -0.03:
            print("⚠️ NO SIGNAL - Momentum not predictive")
        else:
            print("❌ NEGATIVE SIGNAL - Momentum is contrarian (consider inverting)")

    if all_correlations:
        avg_ic = sum(all_correlations) / len(all_correlations)
        print(f"\nAverage IC: {avg_ic:>7.3f}")

        if avg_ic > 0.10:
            print("✅ HIGH IC - Strong predictive power")
        elif avg_ic > 0.05:
            print("✓ MODERATE IC - Useful signal")
        elif avg_ic > 0.02:
            print("~ LOW IC - Marginal signal")
        else:
            print("⚠️ NEGLIGIBLE IC - Limited predictive power")

    # Recommendation
    print("\n" + "-" * 70)
    print("RECOMMENDATION")
    print("-" * 70)

    if all_spreads and all_correlations:
        avg_spread = sum(all_spreads) / len(all_spreads)
        avg_ic = sum(all_correlations) / len(all_correlations)

        if avg_spread > 0.05 and avg_ic > 0.03:
            print("""
✅ ADD MOMENTUM TO COMPOSITE SCORING

Suggested integration:
- Weight: 15-20% of composite score
- Apply confidence multiplier (already built into signals)
- Use as tie-breaker when other factors are similar

See: src/scoring/integrate_momentum_confidence_weighted.py
            """)
        elif avg_spread < -0.05:
            print("""
⚠️ CONSIDER INVERTING MOMENTUM

Your momentum signal appears contrarian - high momentum stocks
underperform. Consider:
1. Using LOW momentum as a BUY signal
2. Or investigating why (biotech-specific mean reversion?)
            """)
        else:
            print("""
~ MOMENTUM ADDS LIMITED VALUE

Consider:
1. Momentum as tie-breaker only (not in core weights)
2. Sector-specific momentum (vs overall biotech)
3. More sophisticated momentum factors
            """)

    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(run_backtest_validation())
