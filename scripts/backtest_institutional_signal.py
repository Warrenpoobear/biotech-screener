#!/usr/bin/env python3
"""
Backtest Harness for Institutional Signal

Point-in-time safe backtesting:
- Uses only data that was available at signal generation time
- Computes forward returns from first trading day AFTER quarter end
- Measures hit rate, mean excess return vs benchmark

Outputs:
- backtest_results.json (canonical JSON with governance metadata)
- BACKTEST_SUMMARY.txt (human-readable summary)
- BACKTEST_PER_QUARTER_TABLE.txt (tabular per-quarter results)

Usage:
    python scripts/backtest_institutional_signal.py \
        --holdings-history-dir production_data/holdings_history \
        --prices-csv data/daily_prices.csv \
        --benchmark XBI \
        --horizons 30,60,90 \
        --topk 10,25 \
        --min-managers 4 \
        --score-version v1 \
        --out-dir backtests/
"""

import argparse
import csv
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.history.snapshots import (
    list_quarters,
    load_snapshot,
    load_manifest,
    get_prior_quarter,
    SNAPSHOT_SCHEMA_VERSION,
)
from governance.canonical_json import canonical_dumps
from governance.hashing import hash_file, hash_bytes
from governance.run_id import compute_run_id
from governance.schema_registry import PIPELINE_VERSION


# =============================================================================
# PRICE DATA LOADING
# =============================================================================

def load_prices(prices_path: Path) -> Dict[str, Dict[str, float]]:
    """
    Load price data from CSV.

    Args:
        prices_path: Path to prices CSV (date,ticker,adj_close)

    Returns:
        Dict of {ticker: {date_str: price}}
    """
    prices = {}

    with open(prices_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row['ticker'].strip().upper()
            date_str = row['date'].strip()
            price = float(row['adj_close'])

            if ticker not in prices:
                prices[ticker] = {}
            prices[ticker][date_str] = price

    return prices


def get_all_dates(prices: Dict[str, Dict[str, float]]) -> List[str]:
    """Get sorted list of all trading dates."""
    all_dates = set()
    for ticker_prices in prices.values():
        all_dates.update(ticker_prices.keys())
    return sorted(all_dates)


def find_price_on_or_after(
    ticker: str,
    target_date: date,
    prices: Dict[str, Dict[str, float]],
    trading_dates: List[str],
    max_days: int = 10,
) -> Optional[Tuple[str, float]]:
    """
    Find price for ticker on or after target date.

    Returns:
        (date_str, price) or None if not found within max_days
    """
    if ticker not in prices:
        return None

    ticker_prices = prices[ticker]
    target_str = target_date.isoformat()

    # Find first date >= target
    for date_str in trading_dates:
        if date_str >= target_str:
            if date_str in ticker_prices:
                return date_str, ticker_prices[date_str]

            # Check if within max_days
            d = datetime.strptime(date_str, '%Y-%m-%d').date()
            if (d - target_date).days > max_days:
                return None

    return None


# =============================================================================
# SCORING LOGIC (Simplified for backtest)
# =============================================================================

def compute_signal_score(
    ticker: str,
    snapshot: Dict[str, Any],
    min_managers: int,
) -> Optional[Dict[str, Any]]:
    """
    Compute institutional signal score for a ticker.

    Scoring logic (simplified from institutional_signal_report.py):
    - Count managers with KNOWN holdings
    - Check for position increases (QoQ)
    - Apply conviction weighting

    Returns:
        Score dict or None if insufficient data
    """
    ticker_data = snapshot.get('tickers', {}).get(ticker)
    if not ticker_data:
        return None

    holdings = ticker_data.get('holdings', {})
    current = holdings.get('current', {})
    prior = holdings.get('prior', {})

    if len(current) < min_managers:
        return None

    # Count position changes
    new_positions = 0
    increased = 0
    decreased = 0
    unchanged = 0

    for cik, holding in current.items():
        if cik not in prior:
            new_positions += 1
        else:
            curr_shares = holding.get('shares', 0)
            prior_shares = prior[cik].get('shares', 0)

            if curr_shares > prior_shares * 1.05:
                increased += 1
            elif curr_shares < prior_shares * 0.95:
                decreased += 1
            else:
                unchanged += 1

    # Simple score: managers + increases - decreases
    score = len(current) * 10 + new_positions * 5 + increased * 3 - decreased * 2

    return {
        'ticker': ticker,
        'score': score,
        'managers_count': len(current),
        'new_positions': new_positions,
        'increased': increased,
        'decreased': decreased,
        'unchanged': unchanged,
    }


def rank_tickers(
    snapshot: Dict[str, Any],
    min_managers: int,
) -> List[Dict[str, Any]]:
    """
    Rank all tickers by signal score.

    Returns:
        Sorted list of score dicts (highest first)
    """
    scores = []

    for ticker in snapshot.get('tickers', {}).keys():
        score = compute_signal_score(ticker, snapshot, min_managers)
        if score:
            scores.append(score)

    # Sort by score desc, then managers desc, then ticker asc
    scores.sort(key=lambda x: (-x['score'], -x['managers_count'], x['ticker']))

    return scores


# =============================================================================
# RETURN CALCULATION
# =============================================================================

def compute_forward_return(
    ticker: str,
    entry_date: date,
    horizon_days: int,
    prices: Dict[str, Dict[str, float]],
    trading_dates: List[str],
) -> Optional[float]:
    """
    Compute forward return for a ticker.

    Args:
        ticker: Stock ticker
        entry_date: Entry date (first trading day after signal)
        horizon_days: Holding period in days
        prices: Price data
        trading_dates: Sorted list of trading dates

    Returns:
        Return as decimal (0.05 = 5%) or None if data missing
    """
    # Find entry price
    entry = find_price_on_or_after(ticker, entry_date, prices, trading_dates)
    if not entry:
        return None

    entry_date_str, entry_price = entry

    # Find exit date and price
    exit_target = datetime.strptime(entry_date_str, '%Y-%m-%d').date() + timedelta(days=horizon_days)
    exit = find_price_on_or_after(ticker, exit_target, prices, trading_dates)
    if not exit:
        return None

    exit_date_str, exit_price = exit

    # Compute return
    return (exit_price / entry_price) - 1.0


def compute_horizon_metrics(
    tickers: List[str],
    entry_date: date,
    horizon_days: int,
    benchmark: str,
    prices: Dict[str, Dict[str, float]],
    trading_dates: List[str],
) -> Dict[str, Any]:
    """
    Compute metrics for a horizon.

    Returns:
        Dict with mean_return, median_return, benchmark_return, mean_excess, hit_rate
    """
    returns = []
    excess_returns = []

    # Get benchmark return
    benchmark_return = compute_forward_return(
        benchmark, entry_date, horizon_days, prices, trading_dates
    )

    for ticker in tickers:
        ret = compute_forward_return(ticker, entry_date, horizon_days, prices, trading_dates)
        if ret is not None:
            returns.append(ret)
            if benchmark_return is not None:
                excess_returns.append(ret - benchmark_return)

    if not returns:
        return {
            'mean_return': None,
            'median_return': None,
            'benchmark_return': benchmark_return,
            'mean_excess': None,
            'hit_rate': None,
            'tickers_measured': 0,
        }

    # Sort for median
    sorted_returns = sorted(returns)
    n = len(sorted_returns)
    if n % 2 == 0:
        median_return = (sorted_returns[n//2 - 1] + sorted_returns[n//2]) / 2
    else:
        median_return = sorted_returns[n//2]

    mean_return = sum(returns) / len(returns)

    # Compute hit rate and mean excess
    if excess_returns:
        mean_excess = sum(excess_returns) / len(excess_returns)
        hit_rate = sum(1 for x in excess_returns if x > 0) / len(excess_returns)
    else:
        mean_excess = None
        hit_rate = None

    return {
        'mean_return': round(mean_return, 6),
        'median_return': round(median_return, 6),
        'benchmark_return': round(benchmark_return, 6) if benchmark_return else None,
        'mean_excess': round(mean_excess, 6) if mean_excess else None,
        'hit_rate': round(hit_rate, 4) if hit_rate else None,
        'tickers_measured': len(returns),
    }


# =============================================================================
# BACKTEST CORE
# =============================================================================

def run_backtest(
    holdings_dir: Path,
    prices_path: Path,
    benchmark: str,
    horizons: List[int],
    topk_values: List[int],
    min_managers: int,
    score_version: str,
) -> Dict[str, Any]:
    """
    Run full backtest across all quarters.

    Returns:
        Backtest results dict
    """
    print(f"\n{'='*80}")
    print("INSTITUTIONAL SIGNAL BACKTEST")
    print(f"{'='*80}")
    print(f"Holdings dir: {holdings_dir}")
    print(f"Benchmark: {benchmark}")
    print(f"Horizons: {horizons}")
    print(f"Top-K: {topk_values}")
    print(f"Min managers: {min_managers}")
    print(f"{'='*80}\n")

    # Load prices
    print("Loading price data...")
    prices = load_prices(prices_path)
    trading_dates = get_all_dates(prices)
    print(f"  Loaded {len(prices)} tickers, {len(trading_dates)} trading days")
    print(f"  Date range: {trading_dates[0]} to {trading_dates[-1]}")

    # Load manifest
    manifest = load_manifest(holdings_dir)
    run_id_source = manifest.get('run_id', 'unknown')

    # Get available quarters
    quarters = list_quarters(holdings_dir)
    print(f"\n  Available quarters: {len(quarters)}")

    # For backtesting, we need quarters with price data available
    # Skip most recent quarter (incomplete forward data)
    usable_quarters = []
    for q in quarters[1:]:  # Skip most recent
        signal_date = q + timedelta(days=1)  # First day after quarter end

        # Check if we have enough forward price data
        max_horizon = max(horizons)
        end_target = signal_date + timedelta(days=max_horizon + 30)

        if end_target.isoformat() <= trading_dates[-1]:
            usable_quarters.append(q)

    print(f"  Usable quarters (with forward data): {len(usable_quarters)}")

    if not usable_quarters:
        raise ValueError("No quarters with sufficient forward price data")

    # Run backtest per quarter
    per_quarter_results = []

    for i, quarter_end in enumerate(usable_quarters):
        print(f"\n--- Quarter {i+1}/{len(usable_quarters)}: {quarter_end} ---")

        # Load snapshot
        snapshot = load_snapshot(quarter_end, holdings_dir)

        # Compute rankings
        rankings = rank_tickers(snapshot, min_managers)
        print(f"  Ranked {len(rankings)} tickers")

        # Signal date: first trading day after quarter end
        signal_target = quarter_end + timedelta(days=1)
        signal_entry = find_price_on_or_after(
            benchmark, signal_target, prices, trading_dates
        )

        if not signal_entry:
            print(f"  WARNING: No trading day found after {quarter_end}")
            continue

        signal_date = datetime.strptime(signal_entry[0], '%Y-%m-%d').date()

        quarter_result = {
            'quarter_end': quarter_end.isoformat(),
            'signal_date': signal_date.isoformat(),
        }

        # Compute metrics for each topk
        for k in topk_values:
            top_k_rankings = rankings[:k]
            top_k_tickers = [r['ticker'] for r in top_k_rankings]
            top_k_scores = [r['score'] for r in top_k_rankings]

            k_key = f'topk_{k}'
            quarter_result[k_key] = {
                'tickers': top_k_tickers,
                'scores': top_k_scores,
                'horizons': {},
            }

            for horizon in horizons:
                metrics = compute_horizon_metrics(
                    tickers=top_k_tickers,
                    entry_date=signal_date,
                    horizon_days=horizon,
                    benchmark=benchmark,
                    prices=prices,
                    trading_dates=trading_dates,
                )
                quarter_result[k_key]['horizons'][str(horizon)] = metrics

                print(f"  top{k} @ {horizon}d: mean_ret={metrics.get('mean_return', 'N/A'):.4f}, "
                      f"excess={metrics.get('mean_excess', 'N/A'):.4f if metrics.get('mean_excess') else 'N/A'}, "
                      f"hit={metrics.get('hit_rate', 'N/A'):.2f if metrics.get('hit_rate') else 'N/A'}")

        # Compute turnover vs prior quarter
        prior_quarter = get_prior_quarter(quarter_end)
        try:
            prior_snapshot = load_snapshot(prior_quarter, holdings_dir)
            prior_rankings = rank_tickers(prior_snapshot, min_managers)

            # Compare top-k sets
            for k in topk_values:
                curr_set = set(r['ticker'] for r in rankings[:k])
                prior_set = set(r['ticker'] for r in prior_rankings[:k])

                if prior_set:
                    turnover = len(curr_set - prior_set) / k
                else:
                    turnover = 1.0

                quarter_result[f'topk_{k}']['turnover_vs_prior'] = round(turnover, 4)

        except Exception:
            # Prior quarter not available
            for k in topk_values:
                quarter_result[f'topk_{k}']['turnover_vs_prior'] = None

        per_quarter_results.append(quarter_result)

    # Compute aggregate metrics
    print(f"\n--- Aggregate Metrics ---")
    aggregate = {}

    for k in topk_values:
        k_key = f'topk_{k}'
        aggregate[k_key] = {'horizons': {}}

        for horizon in horizons:
            h_key = str(horizon)

            # Collect all quarter metrics for this topk/horizon
            all_returns = []
            all_excess = []
            all_hits = []

            for qr in per_quarter_results:
                metrics = qr.get(k_key, {}).get('horizons', {}).get(h_key, {})
                if metrics.get('mean_return') is not None:
                    all_returns.append(metrics['mean_return'])
                if metrics.get('mean_excess') is not None:
                    all_excess.append(metrics['mean_excess'])
                if metrics.get('hit_rate') is not None:
                    all_hits.append(metrics['hit_rate'])

            if all_returns:
                aggregate[k_key]['horizons'][h_key] = {
                    'mean_return': round(sum(all_returns) / len(all_returns), 6),
                    'median_return': round(sorted(all_returns)[len(all_returns)//2], 6),
                    'mean_excess': round(sum(all_excess) / len(all_excess), 6) if all_excess else None,
                    'hit_rate': round(sum(all_hits) / len(all_hits), 4) if all_hits else None,
                    'quarters_count': len(all_returns),
                }

                print(f"  top{k} @ {horizon}d: mean_ret={aggregate[k_key]['horizons'][h_key]['mean_return']:.4f}, "
                      f"hit_rate={aggregate[k_key]['horizons'][h_key].get('hit_rate', 'N/A')}")

    # Build final results
    prices_hash = hash_file(prices_path)
    manifest_hash = hash_file(holdings_dir / 'manifest.json')

    params = {
        'holdings_history_dir': str(holdings_dir),
        'prices_csv': str(prices_path),
        'benchmark': benchmark,
        'horizons': horizons,
        'topk': topk_values,
        'min_managers': min_managers,
    }
    params_hash = hash_bytes(canonical_dumps(params).encode())[:16]

    # Compute backtest run_id
    backtest_run_id = compute_run_id(
        as_of_date=usable_quarters[0].isoformat(),
        score_version=score_version,
        parameters_hash=params_hash,
        input_hashes=[
            {'path': 'manifest.json', 'sha256': manifest_hash},
            {'path': str(prices_path.name), 'sha256': prices_hash},
        ],
        pipeline_version=PIPELINE_VERSION,
    )

    results = {
        '_schema': {
            'version': 'backtest_results_v1',
        },
        '_governance': {
            'run_id': backtest_run_id,
            'score_version': score_version,
            'parameters_hash': params_hash,
            'manifest_hash': manifest_hash,
            'prices_hash': prices_hash,
            'source_run_id': run_id_source,
        },
        'params': params,
        'quarters_used': len(per_quarter_results),
        'per_quarter': per_quarter_results,
        'aggregate': aggregate,
    }

    return results


# =============================================================================
# REPORT GENERATION
# =============================================================================

def write_summary_txt(results: Dict[str, Any], output_path: Path) -> None:
    """Write human-readable summary."""
    lines = [
        "=" * 70,
        "INSTITUTIONAL SIGNAL BACKTEST SUMMARY",
        "=" * 70,
        "",
        f"Run ID: {results['_governance']['run_id']}",
        f"Score Version: {results['_governance']['score_version']}",
        f"Quarters Analyzed: {results['quarters_used']}",
        "",
        "PARAMETERS:",
        f"  Benchmark: {results['params']['benchmark']}",
        f"  Horizons: {results['params']['horizons']}",
        f"  Top-K: {results['params']['topk']}",
        f"  Min Managers: {results['params']['min_managers']}",
        "",
        "=" * 70,
        "AGGREGATE RESULTS",
        "=" * 70,
        "",
    ]

    for k in results['params']['topk']:
        k_key = f'topk_{k}'
        if k_key not in results['aggregate']:
            continue

        lines.append(f"TOP {k}:")
        lines.append("-" * 40)

        for horizon in results['params']['horizons']:
            h_key = str(horizon)
            metrics = results['aggregate'][k_key]['horizons'].get(h_key, {})

            if not metrics:
                continue

            lines.append(f"  {horizon}-day horizon:")
            lines.append(f"    Mean Return:    {metrics.get('mean_return', 0)*100:+.2f}%")
            lines.append(f"    Median Return:  {metrics.get('median_return', 0)*100:+.2f}%")

            if metrics.get('mean_excess') is not None:
                lines.append(f"    Mean Excess:    {metrics['mean_excess']*100:+.2f}%")

            if metrics.get('hit_rate') is not None:
                lines.append(f"    Hit Rate:       {metrics['hit_rate']*100:.1f}%")

            lines.append(f"    Quarters:       {metrics.get('quarters_count', 0)}")
            lines.append("")

        lines.append("")

    lines.extend([
        "=" * 70,
        "Generated by backtest_institutional_signal.py",
        "=" * 70,
    ])

    output_path.write_text('\n'.join(lines))


def write_per_quarter_table(results: Dict[str, Any], output_path: Path) -> None:
    """Write per-quarter tabular results."""
    lines = [
        "=" * 100,
        "BACKTEST PER-QUARTER RESULTS",
        "=" * 100,
        "",
    ]

    # Get first topk value for the table
    topk = results['params']['topk'][0]
    k_key = f'topk_{topk}'

    # Header
    horizons = results['params']['horizons']
    header = f"{'Quarter':<12} {'Signal':<12}"
    for h in horizons:
        header += f" {'Ret'+str(h)+'d':>10} {'Exc'+str(h)+'d':>10}"
    header += f" {'Turnover':>10}"
    lines.append(header)
    lines.append("-" * 100)

    # Per-quarter rows
    for qr in results['per_quarter']:
        row = f"{qr['quarter_end']:<12} {qr['signal_date']:<12}"

        q_data = qr.get(k_key, {})

        for h in horizons:
            h_key = str(h)
            metrics = q_data.get('horizons', {}).get(h_key, {})

            ret = metrics.get('mean_return')
            exc = metrics.get('mean_excess')

            row += f" {ret*100:>9.2f}%" if ret is not None else f" {'N/A':>10}"
            row += f" {exc*100:>9.2f}%" if exc is not None else f" {'N/A':>10}"

        turnover = q_data.get('turnover_vs_prior')
        row += f" {turnover*100:>9.1f}%" if turnover is not None else f" {'N/A':>10}"

        lines.append(row)

    lines.extend([
        "",
        "=" * 100,
    ])

    output_path.write_text('\n'.join(lines))


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run point-in-time backtest of institutional signal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        '--holdings-history-dir',
        type=Path,
        required=True,
        help='Directory with per-quarter holdings snapshots',
    )

    parser.add_argument(
        '--prices-csv',
        type=Path,
        required=True,
        help='Path to daily prices CSV (date,ticker,adj_close)',
    )

    parser.add_argument(
        '--benchmark',
        default='XBI',
        help='Benchmark ticker (default: XBI)',
    )

    parser.add_argument(
        '--horizons',
        default='30,60,90',
        help='Comma-separated holding periods in days (default: 30,60,90)',
    )

    parser.add_argument(
        '--topk',
        default='10,25',
        help='Comma-separated top-k values (default: 10,25)',
    )

    parser.add_argument(
        '--min-managers',
        type=int,
        default=4,
        help='Minimum managers for inclusion (default: 4)',
    )

    parser.add_argument(
        '--score-version',
        default='v1',
        help='Score version for governance (default: v1)',
    )

    parser.add_argument(
        '--out-dir',
        type=Path,
        required=True,
        help='Output directory for backtest results',
    )

    args = parser.parse_args()

    # Parse list arguments
    horizons = [int(x.strip()) for x in args.horizons.split(',')]
    topk_values = [int(x.strip()) for x in args.topk.split(',')]

    # Validate inputs
    if not args.holdings_history_dir.exists():
        print(f"ERROR: Holdings directory not found: {args.holdings_history_dir}")
        sys.exit(1)

    if not args.prices_csv.exists():
        print(f"ERROR: Prices CSV not found: {args.prices_csv}")
        sys.exit(1)

    # Create output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Run backtest
    try:
        results = run_backtest(
            holdings_dir=args.holdings_history_dir,
            prices_path=args.prices_csv,
            benchmark=args.benchmark,
            horizons=horizons,
            topk_values=topk_values,
            min_managers=args.min_managers,
            score_version=args.score_version,
        )

        # Write outputs
        results_path = args.out_dir / 'backtest_results.json'
        summary_path = args.out_dir / 'BACKTEST_SUMMARY.txt'
        table_path = args.out_dir / 'BACKTEST_PER_QUARTER_TABLE.txt'

        # Write canonical JSON
        results_json = canonical_dumps(results)
        results_path.write_text(results_json)
        print(f"\nWritten: {results_path}")

        # Write summary
        write_summary_txt(results, summary_path)
        print(f"Written: {summary_path}")

        # Write per-quarter table
        write_per_quarter_table(results, table_path)
        print(f"Written: {table_path}")

        print(f"\n{'='*80}")
        print("BACKTEST COMPLETE")
        print(f"{'='*80}")
        print(f"Run ID: {results['_governance']['run_id']}")
        print(f"Quarters analyzed: {results['quarters_used']}")
        print(f"Output directory: {args.out_dir}")
        print(f"{'='*80}\n")

        sys.exit(0)

    except Exception as e:
        print(f"ERROR: Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
