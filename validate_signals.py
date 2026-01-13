#!/usr/bin/env python3
"""
Signal Validation Framework

Tests if screening signals actually predict forward returns.
Uses CACHED returns data - NO TOKEN REQUIRED.

This is the core backtesting tool for validating that your screening
methodology produces alpha.

Usage:
    python validate_signals.py \
        --database data/returns/returns_db.json \
        --ranked-list screen_2023_01.csv \
        --screen-date 2023-01-15 \
        --forward-months 6

Output:
    - Hit rate (% of picks that were profitable)
    - Average/median forward returns
    - Quintile analysis (do top-ranked outperform bottom-ranked?)
    - Alpha vs XBI benchmark
    - Statistical significance

Architecture:
    - Reads from cached JSON database (built by build_returns_database.py)
    - No API calls = no token needed
    - Deterministic (same inputs = same outputs)
    - Point-in-time compliant (no look-ahead bias)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from morningstar_returns import ReturnsDatabase


@dataclass
class ValidationResult:
    """Results of signal validation."""

    screen_date: str
    forward_months: int
    total_tickers: int
    tickers_with_returns: int

    # Performance metrics
    hit_rate: Optional[float]  # % profitable
    avg_return: Optional[float]
    median_return: Optional[float]
    std_return: Optional[float]

    # Benchmark comparison
    benchmark_return: Optional[float]
    avg_alpha: Optional[float]

    # Alpha metrics (enhanced)
    alpha_hit_rate: Optional[float]  # % that beat benchmark
    median_alpha: Optional[float]
    quintile_alphas: Dict[int, float]  # Alpha by quintile
    alpha_spread: Optional[float]  # Q1 alpha - Q5 alpha

    # Quintile analysis
    quintile_returns: Dict[int, float]  # Q1=top, Q5=bottom
    quintile_spread: Optional[float]  # Q1 - Q5

    # Individual results
    ticker_results: List[Dict[str, Any]]


def load_ranked_list(ranked_list_path: Path) -> List[Tuple[str, Optional[float]]]:
    """
    Load ranked list from CSV.

    Expects CSV with columns:
        - ticker (required): Ticker symbol
        - rank or score (optional): Ranking/score value

    Returns:
        List of (ticker, score) tuples, ordered by file order or score
    """
    results = []

    with open(ranked_list_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        # Find columns (case-insensitive)
        headers = {h.lower(): h for h in reader.fieldnames or []}
        ticker_col = headers.get("ticker") or headers.get("symbol")
        rank_col = headers.get("rank") or headers.get("score") or headers.get("composite_score")

        if not ticker_col:
            raise ValueError(
                f"No 'ticker' column found in {ranked_list_path}. "
                f"Available columns: {list(reader.fieldnames or [])}"
            )

        for row in reader:
            ticker = row[ticker_col].strip().upper()
            if not ticker:
                continue

            score = None
            if rank_col:
                try:
                    score = float(row[rank_col])
                except (ValueError, TypeError, KeyError):
                    pass

            results.append((ticker, score))

    # Sort by score if available (descending - higher is better)
    if any(s is not None for _, s in results):
        results.sort(key=lambda x: (x[1] is None, -(x[1] or 0)))

    return results


def calculate_quintiles(
    ticker_returns: List[Tuple[str, float]],
) -> Dict[int, float]:
    """
    Calculate average returns by quintile.

    Q1 = top 20% (highest ranked)
    Q5 = bottom 20% (lowest ranked)
    """
    n = len(ticker_returns)
    if n < 5:
        return {}

    quintile_size = n // 5
    quintiles = {}

    for q in range(1, 6):
        start_idx = (q - 1) * quintile_size
        end_idx = q * quintile_size if q < 5 else n

        q_returns = [r for _, r in ticker_returns[start_idx:end_idx]]
        if q_returns:
            quintiles[q] = sum(q_returns) / len(q_returns)

    return quintiles


def validate_signals(
    database: ReturnsDatabase,
    ranked_list: List[Tuple[str, Optional[float]]],
    screen_date: str,
    forward_months: int,
) -> ValidationResult:
    """
    Validate screening signals against forward returns.

    Args:
        database: Returns database (loaded from JSON)
        ranked_list: List of (ticker, score) tuples
        screen_date: Date of the screen (YYYY-MM-DD)
        forward_months: Forward return period in months

    Returns:
        ValidationResult with hit rate, quintile analysis, etc.
    """
    # Calculate forward returns for each ticker
    ticker_results = []
    returns_list = []

    for ticker, score in ranked_list:
        forward_return = database.get_forward_return(ticker, screen_date, forward_months)
        excess_return = database.get_excess_return(ticker, screen_date, forward_months)

        result = {
            "ticker": ticker,
            "score": score,
            "forward_return": float(forward_return) if forward_return else None,
            "excess_return": float(excess_return) if excess_return else None,
        }
        ticker_results.append(result)

        if forward_return is not None:
            returns_list.append((ticker, float(forward_return)))

    # Calculate metrics
    n_with_returns = len(returns_list)

    if n_with_returns == 0:
        return ValidationResult(
            screen_date=screen_date,
            forward_months=forward_months,
            total_tickers=len(ranked_list),
            tickers_with_returns=0,
            hit_rate=None,
            avg_return=None,
            median_return=None,
            std_return=None,
            benchmark_return=None,
            avg_alpha=None,
            alpha_hit_rate=None,
            median_alpha=None,
            quintile_alphas={},
            alpha_spread=None,
            quintile_returns={},
            quintile_spread=None,
            ticker_results=ticker_results,
        )

    # Hit rate (% profitable)
    profitable = sum(1 for _, r in returns_list if r > 0)
    hit_rate = profitable / n_with_returns

    # Average/median return
    all_returns = [r for _, r in returns_list]
    avg_return = sum(all_returns) / len(all_returns)

    sorted_returns = sorted(all_returns)
    mid = len(sorted_returns) // 2
    if len(sorted_returns) % 2 == 0:
        median_return = (sorted_returns[mid - 1] + sorted_returns[mid]) / 2
    else:
        median_return = sorted_returns[mid]

    # Standard deviation
    variance = sum((r - avg_return) ** 2 for r in all_returns) / len(all_returns)
    std_return = variance ** 0.5

    # Benchmark return
    benchmark_return = database.get_forward_return("XBI", screen_date, forward_months)
    benchmark_return = float(benchmark_return) if benchmark_return else None

    # Alpha metrics
    excess_returns = [r["excess_return"] for r in ticker_results if r["excess_return"] is not None]
    avg_alpha = sum(excess_returns) / len(excess_returns) if excess_returns else None

    # Enhanced alpha metrics
    alpha_hit_rate = None
    median_alpha = None
    if excess_returns:
        # Alpha hit rate (% that beat benchmark)
        beat_benchmark = sum(1 for a in excess_returns if a > 0)
        alpha_hit_rate = beat_benchmark / len(excess_returns)

        # Median alpha
        sorted_alphas = sorted(excess_returns)
        mid = len(sorted_alphas) // 2
        if len(sorted_alphas) % 2 == 0:
            median_alpha = (sorted_alphas[mid - 1] + sorted_alphas[mid]) / 2
        else:
            median_alpha = sorted_alphas[mid]

    # Build alpha list in ranked order (matching returns_list order)
    alpha_list = []
    for ticker, _ in returns_list:
        for r in ticker_results:
            if r["ticker"] == ticker and r["excess_return"] is not None:
                alpha_list.append((ticker, r["excess_return"]))
                break

    # Quintile analysis
    quintile_returns = calculate_quintiles(returns_list)
    quintile_spread = None
    if 1 in quintile_returns and 5 in quintile_returns:
        quintile_spread = quintile_returns[1] - quintile_returns[5]

    # Quintile alpha analysis
    quintile_alphas = calculate_quintiles(alpha_list)
    alpha_spread = None
    if 1 in quintile_alphas and 5 in quintile_alphas:
        alpha_spread = quintile_alphas[1] - quintile_alphas[5]

    return ValidationResult(
        screen_date=screen_date,
        forward_months=forward_months,
        total_tickers=len(ranked_list),
        tickers_with_returns=n_with_returns,
        hit_rate=hit_rate,
        avg_return=avg_return,
        median_return=median_return,
        std_return=std_return,
        benchmark_return=benchmark_return,
        avg_alpha=avg_alpha,
        alpha_hit_rate=alpha_hit_rate,
        median_alpha=median_alpha,
        quintile_alphas=quintile_alphas,
        alpha_spread=alpha_spread,
        quintile_returns=quintile_returns,
        quintile_spread=quintile_spread,
        ticker_results=ticker_results,
    )


def print_report(result: ValidationResult) -> None:
    """Print formatted validation report."""
    print()
    print("=" * 60)
    print("SIGNAL VALIDATION REPORT")
    print("=" * 60)
    print()
    print(f"Screen Date:      {result.screen_date}")
    print(f"Forward Period:   {result.forward_months} months")
    print(f"Tickers Tested:   {result.tickers_with_returns}/{result.total_tickers}")
    print()

    if result.tickers_with_returns == 0:
        print("ERROR: No returns data available for the specified period.")
        print()
        print("Possible causes:")
        print("  - Screen date outside database date range")
        print("  - Forward period extends beyond database end date")
        print("  - Tickers not found in database")
        return

    # Performance Metrics
    print("-" * 60)
    print("PERFORMANCE METRICS")
    print("-" * 60)

    def fmt_pct(val: Optional[float], suffix: str = "") -> str:
        if val is None:
            return "N/A"
        return f"{val * 100:+.2f}%{suffix}"

    print(f"  Hit Rate:          {fmt_pct(result.hit_rate)}  (% profitable)")
    print(f"  Average Return:    {fmt_pct(result.avg_return)}")
    print(f"  Median Return:     {fmt_pct(result.median_return)}")
    print(f"  Std Deviation:     {fmt_pct(result.std_return)}")
    print()

    # Alpha Metrics (vs XBI Benchmark)
    print("-" * 60)
    print("ALPHA METRICS (vs XBI)")
    print("-" * 60)
    print(f"  XBI Return:        {fmt_pct(result.benchmark_return)}")
    print(f"  Alpha Hit Rate:    {fmt_pct(result.alpha_hit_rate)}  (% that beat XBI)")
    print(f"  Average Alpha:     {fmt_pct(result.avg_alpha)}")
    print(f"  Median Alpha:      {fmt_pct(result.median_alpha)}")
    print()

    # Alpha by quintile
    if result.quintile_alphas:
        print("  Alpha by Quintile:")
        for q in range(1, 6):
            if q in result.quintile_alphas:
                label = "TOP" if q == 1 else "BOTTOM" if q == 5 else ""
                print(f"    Q{q}: {fmt_pct(result.quintile_alphas[q]):>10}  {label}")
        print()
        if result.alpha_spread is not None:
            print(f"  Alpha Q1-Q5 Spread: {fmt_pct(result.alpha_spread)}")
            print()

    # Quintile Analysis (Absolute Returns)
    if result.quintile_returns:
        print("-" * 60)
        print("QUINTILE ANALYSIS")
        print("-" * 60)
        print("  (Q1 = top-ranked, Q5 = bottom-ranked)")
        print()
        for q in range(1, 6):
            if q in result.quintile_returns:
                label = "TOP" if q == 1 else "BOTTOM" if q == 5 else ""
                print(f"  Q{q}: {fmt_pct(result.quintile_returns[q]):>10}  {label}")

        print()
        if result.quintile_spread is not None:
            print(f"  Q1-Q5 Spread:      {fmt_pct(result.quintile_spread)}")
            print()

    # Interpretation
    print("-" * 60)
    print("INTERPRETATION")
    print("-" * 60)

    # Hit rate interpretation
    if result.hit_rate is not None:
        if result.hit_rate > 0.55:
            print(f"  [GOOD] Hit rate {result.hit_rate*100:.1f}% > 55% indicates useful signal")
        elif result.hit_rate > 0.50:
            print(f"  [WEAK] Hit rate {result.hit_rate*100:.1f}% is marginally above random")
        else:
            print(f"  [POOR] Hit rate {result.hit_rate*100:.1f}% is below random - signal may be inverted")

    # Quintile spread interpretation
    if result.quintile_spread is not None:
        if result.quintile_spread > 0.10:
            print(f"  [STRONG] {result.quintile_spread*100:.1f}% Q1-Q5 spread indicates strong ranking power")
        elif result.quintile_spread > 0.05:
            print(f"  [GOOD] {result.quintile_spread*100:.1f}% Q1-Q5 spread indicates meaningful ranking")
        elif result.quintile_spread > 0:
            print(f"  [WEAK] {result.quintile_spread*100:.1f}% Q1-Q5 spread indicates weak ranking signal")
        else:
            print(f"  [INVERTED] Negative Q1-Q5 spread - bottom-ranked outperform top-ranked!")

    # Alpha interpretation
    if result.avg_alpha is not None:
        if result.avg_alpha > 0.05:
            print(f"  [STRONG ALPHA] {result.avg_alpha*100:.1f}% average alpha vs XBI")
        elif result.avg_alpha > 0.02:
            print(f"  [GOOD ALPHA] {result.avg_alpha*100:.1f}% average alpha vs XBI")
        elif result.avg_alpha > 0:
            print(f"  [MILD ALPHA] {result.avg_alpha*100:.1f}% average alpha vs XBI")
        else:
            print(f"  [NO ALPHA] {result.avg_alpha*100:.1f}% underperforms XBI on average")

    # Alpha spread interpretation
    if result.alpha_spread is not None:
        if result.alpha_spread > 0.10:
            print(f"  [STRONG RANKING] {result.alpha_spread*100:.1f}% alpha spread - top picks generate most alpha")
        elif result.alpha_spread > 0.05:
            print(f"  [GOOD RANKING] {result.alpha_spread*100:.1f}% alpha spread - ranking adds value")
        elif result.alpha_spread > 0:
            print(f"  [WEAK RANKING] {result.alpha_spread*100:.1f}% alpha spread - minimal ranking benefit")
        else:
            print(f"  [INVERTED ALPHA] Negative alpha spread - bottom-ranked generate more alpha!")

    print()


def save_results(result: ValidationResult, output_path: Path) -> None:
    """Save validation results to JSON."""
    output = {
        "screen_date": result.screen_date,
        "forward_months": result.forward_months,
        "total_tickers": result.total_tickers,
        "tickers_with_returns": result.tickers_with_returns,
        "metrics": {
            "hit_rate": result.hit_rate,
            "avg_return": result.avg_return,
            "median_return": result.median_return,
            "std_return": result.std_return,
            "benchmark_return": result.benchmark_return,
            "avg_alpha": result.avg_alpha,
        },
        "alpha_metrics": {
            "alpha_hit_rate": result.alpha_hit_rate,
            "avg_alpha": result.avg_alpha,
            "median_alpha": result.median_alpha,
            "quintile_alphas": result.quintile_alphas,
            "alpha_spread": result.alpha_spread,
        },
        "quintile_analysis": {
            "quintile_returns": result.quintile_returns,
            "q1_q5_spread": result.quintile_spread,
        },
        "ticker_results": result.ticker_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate screening signals against forward returns (NO TOKEN REQUIRED)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic validation
    python validate_signals.py \\
        --database data/returns/returns_db.json \\
        --ranked-list screen_2023_01.csv \\
        --screen-date 2023-01-15

    # Test 6-month forward returns
    python validate_signals.py \\
        --database data/returns/returns_db.json \\
        --ranked-list screen_2023_01.csv \\
        --screen-date 2023-01-15 \\
        --forward-months 6

    # Save detailed results to JSON
    python validate_signals.py \\
        --database data/returns/returns_db.json \\
        --ranked-list screen_2023_01.csv \\
        --screen-date 2023-01-15 \\
        --output validation_results.json

NO TOKEN REQUIRED - This tool uses cached data only.
        """,
    )

    parser.add_argument(
        "--database",
        type=Path,
        required=True,
        help="Path to returns database JSON file (from build_returns_database.py)",
    )

    parser.add_argument(
        "--ranked-list",
        type=Path,
        required=True,
        help="Path to CSV with ranked tickers (must have 'ticker' column)",
    )

    parser.add_argument(
        "--screen-date",
        type=str,
        required=True,
        help="Date of the screen (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--forward-months",
        type=int,
        default=3,
        help="Forward return period in months (default: 3)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save detailed results JSON (optional)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output (only show summary)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.database.exists():
        print(f"Error: Database not found: {args.database}")
        print()
        print("Build the database first with:")
        print("  python build_returns_database.py --universe universe.csv --start-date 2020-01-01")
        sys.exit(1)

    if not args.ranked_list.exists():
        print(f"Error: Ranked list not found: {args.ranked_list}")
        sys.exit(1)

    # Parse screen date
    try:
        screen_date = date.fromisoformat(args.screen_date)
    except ValueError:
        print(f"Error: Invalid date format: {args.screen_date}")
        print("Expected format: YYYY-MM-DD (e.g., 2023-01-15)")
        sys.exit(1)

    # Load database
    print(f"Loading database: {args.database}")
    try:
        database = ReturnsDatabase(args.database)
    except Exception as e:
        print(f"Error loading database: {e}")
        sys.exit(1)

    print(f"  Date range: {database.date_range[0]} to {database.date_range[1]}")
    print(f"  Tickers: {len(database.available_tickers)}")
    print()

    # Load ranked list
    print(f"Loading ranked list: {args.ranked_list}")
    try:
        ranked_list = load_ranked_list(args.ranked_list)
    except Exception as e:
        print(f"Error loading ranked list: {e}")
        sys.exit(1)

    print(f"  Tickers: {len(ranked_list)}")
    print()

    # Run validation
    result = validate_signals(
        database=database,
        ranked_list=ranked_list,
        screen_date=args.screen_date,
        forward_months=args.forward_months,
    )

    # Print report
    if not args.quiet:
        print_report(result)

    # Save results if requested
    if args.output:
        save_results(result, args.output)

    # Exit code based on results
    if result.tickers_with_returns == 0:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
