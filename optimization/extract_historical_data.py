#!/usr/bin/env python3
"""
Extract historical screening data from checkpoint files for weight optimization.

Reads module_5 checkpoint files and extracts component scores with forward returns.
Outputs CSV in the format required by optimize_weights_scipy.py.

Usage:
    # Extract from checkpoints with price data
    python -m optimization.extract_historical_data \
        --checkpoints-dir checkpoints \
        --prices-file production_data/prices.csv \
        --output optimization/optimization_data/training_dataset.csv

    # Extract without prices (scores only, no returns)
    python -m optimization.extract_historical_data \
        --checkpoints-dir checkpoints \
        --output optimization/optimization_data/scores_only.csv \
        --no-returns

Author: Wake Robin Capital Management
Version: 1.0.0
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__version__ = "1.0.0"

# Component mapping from checkpoint to optimization format
# The checkpoint has 4 core components; momentum/valuation may be added separately
COMPONENT_MAPPING = {
    'clinical': 'clinical',
    'financial': 'financial',
    'catalyst': 'catalyst',
    'pos': 'pos',
}

# Default values for missing components
DEFAULT_MOMENTUM = 50.0  # Neutral
DEFAULT_VALUATION = 50.0  # Neutral


def parse_date(date_str: str) -> datetime:
    """Parse ISO date string to datetime."""
    return datetime.strptime(date_str, '%Y-%m-%d')


def load_checkpoint(filepath: str) -> Dict[str, Any]:
    """Load a module_5 checkpoint file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_scores_from_checkpoint(
    checkpoint: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Extract component scores from a module_5 checkpoint.

    Returns list of {ticker, clinical, financial, catalyst, pos, momentum, valuation}
    """
    as_of_date = checkpoint.get('as_of_date', '')
    data = checkpoint.get('data', {})
    ranked_securities = data.get('ranked_securities', [])

    scores = []

    for security in ranked_securities:
        ticker = security.get('ticker')
        if not ticker:
            continue

        # Extract component scores from score_breakdown
        breakdown = security.get('score_breakdown', {})
        components = breakdown.get('components', [])

        # Build component dict
        component_scores = {}
        for comp in components:
            name = comp.get('name')
            # Use normalized score (0-100 scale)
            normalized = comp.get('normalized', '50.0')
            try:
                component_scores[name] = float(normalized)
            except (ValueError, TypeError):
                component_scores[name] = 50.0

        # Map to output format
        score_row = {
            'date': as_of_date,
            'ticker': ticker,
            'clinical': component_scores.get('clinical', 50.0),
            'financial': component_scores.get('financial', 50.0),
            'catalyst': component_scores.get('catalyst', 50.0),
            'pos': component_scores.get('pos', 50.0),
            'momentum': DEFAULT_MOMENTUM,  # Not in checkpoint, use default
            'valuation': DEFAULT_VALUATION,  # Not in checkpoint, use default
            'composite_score': float(security.get('composite_score', 50.0)),
            'composite_rank': security.get('composite_rank', 999),
        }

        scores.append(score_row)

    return scores


def load_prices(prices_file: str) -> Dict[str, Dict[str, float]]:
    """
    Load price data for forward return calculation.

    Expected format: date,ticker,close
    Returns: {ticker: {date: price}}
    """
    prices = defaultdict(dict)

    if not os.path.exists(prices_file):
        print(f"Warning: Prices file not found: {prices_file}")
        return prices

    with open(prices_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row.get('ticker', row.get('symbol', ''))
            date = row.get('date', '')
            close = row.get('close', row.get('adj_close', row.get('price', '')))

            if ticker and date and close:
                try:
                    prices[ticker][date] = float(close)
                except ValueError:
                    pass

    print(f"Loaded prices for {len(prices)} tickers")
    return prices


def calculate_forward_return(
    ticker: str,
    date: str,
    prices: Dict[str, Dict[str, float]],
    forward_weeks: int = 4
) -> Optional[float]:
    """
    Calculate forward return for a ticker from date.

    Returns None if prices not available.
    """
    if ticker not in prices:
        return None

    ticker_prices = prices[ticker]

    # Find start price
    start_date = parse_date(date)
    start_price = None

    # Look for price on or near start date (within 3 days)
    for delta in range(4):
        check_date = (start_date + timedelta(days=delta)).strftime('%Y-%m-%d')
        if check_date in ticker_prices:
            start_price = ticker_prices[check_date]
            break

    if start_price is None:
        return None

    # Find end price (forward_weeks later)
    end_date = start_date + timedelta(weeks=forward_weeks)
    end_price = None

    for delta in range(4):
        check_date = (end_date + timedelta(days=delta)).strftime('%Y-%m-%d')
        if check_date in ticker_prices:
            end_price = ticker_prices[check_date]
            break

    if end_price is None:
        return None

    # Calculate return
    return (end_price - start_price) / start_price


def find_checkpoint_files(checkpoints_dir: str) -> List[Tuple[str, str]]:
    """
    Find all module_5 checkpoint files.

    Returns: List of (filepath, date) tuples sorted by date
    """
    files = []
    pattern = 'module_5_'

    for filename in os.listdir(checkpoints_dir):
        if filename.startswith(pattern) and filename.endswith('.json'):
            # Extract date from filename (module_5_YYYY-MM-DD.json)
            date_part = filename[len(pattern):-5]
            filepath = os.path.join(checkpoints_dir, filename)
            files.append((filepath, date_part))

    # Sort by date
    files.sort(key=lambda x: x[1])
    return files


def extract_all_data(
    checkpoints_dir: str,
    prices_file: Optional[str] = None,
    forward_weeks: int = 4,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Extract historical data from all checkpoint files.

    Args:
        checkpoints_dir: Directory containing module_5_*.json files
        prices_file: Optional path to prices CSV for forward returns
        forward_weeks: Weeks ahead for forward return calculation
        verbose: Print progress

    Returns:
        List of observation dicts with scores and optional returns
    """
    # Find checkpoint files
    checkpoint_files = find_checkpoint_files(checkpoints_dir)

    if verbose:
        print(f"Found {len(checkpoint_files)} checkpoint files")

    if not checkpoint_files:
        raise ValueError(f"No module_5_*.json files found in {checkpoints_dir}")

    # Load prices if provided
    prices = {}
    if prices_file:
        prices = load_prices(prices_file)

    # Extract scores from each checkpoint
    all_observations = []
    returns_available = 0
    returns_missing = 0

    for filepath, date in checkpoint_files:
        if verbose:
            print(f"  Processing {date}...")

        try:
            checkpoint = load_checkpoint(filepath)
            scores = extract_scores_from_checkpoint(checkpoint)

            for score in scores:
                # Calculate forward return if prices available
                if prices:
                    fwd_return = calculate_forward_return(
                        score['ticker'], date, prices, forward_weeks
                    )
                    if fwd_return is not None:
                        score['fwd_4w'] = fwd_return
                        returns_available += 1
                    else:
                        score['fwd_4w'] = None
                        returns_missing += 1
                else:
                    score['fwd_4w'] = None

                all_observations.append(score)

        except Exception as e:
            print(f"  Error processing {filepath}: {e}")

    if verbose:
        print(f"\nExtracted {len(all_observations)} observations")
        if prices:
            print(f"  Returns available: {returns_available}")
            print(f"  Returns missing:   {returns_missing}")

    return all_observations


def save_to_csv(
    observations: List[Dict[str, Any]],
    output_path: str,
    include_returns: bool = True,
    verbose: bool = True
):
    """
    Save observations to CSV in optimization format.

    Args:
        observations: List of observation dicts
        output_path: Output CSV path
        include_returns: Whether to include forward returns column
        verbose: Print progress
    """
    # Filter to observations with returns if required
    if include_returns:
        valid_obs = [o for o in observations if o.get('fwd_4w') is not None]
        if verbose:
            print(f"Writing {len(valid_obs)} observations with returns")
    else:
        valid_obs = observations
        if verbose:
            print(f"Writing {len(valid_obs)} observations (no returns)")

    if not valid_obs:
        raise ValueError("No valid observations to write")

    # Determine columns
    columns = ['date', 'ticker', 'clinical', 'financial', 'catalyst',
               'pos', 'momentum', 'valuation']
    if include_returns:
        columns.append('fwd_4w')

    # Write CSV
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()

        for obs in valid_obs:
            # Format numeric values
            row = {
                'date': obs['date'],
                'ticker': obs['ticker'],
                'clinical': f"{obs['clinical']:.2f}",
                'financial': f"{obs['financial']:.2f}",
                'catalyst': f"{obs['catalyst']:.2f}",
                'pos': f"{obs['pos']:.2f}",
                'momentum': f"{obs['momentum']:.2f}",
                'valuation': f"{obs['valuation']:.2f}",
            }
            if include_returns and obs.get('fwd_4w') is not None:
                row['fwd_4w'] = f"{obs['fwd_4w']:.6f}"

            writer.writerow(row)

    if verbose:
        print(f"Saved to {output_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Extract historical screening data for weight optimization'
    )
    parser.add_argument(
        '--checkpoints-dir',
        default='checkpoints',
        help='Directory containing module_5_*.json checkpoint files'
    )
    parser.add_argument(
        '--prices-file',
        help='Path to prices CSV (date,ticker,close) for forward returns'
    )
    parser.add_argument(
        '--output',
        default='optimization/optimization_data/training_dataset.csv',
        help='Output CSV path'
    )
    parser.add_argument(
        '--forward-weeks',
        type=int,
        default=4,
        help='Weeks ahead for forward return calculation'
    )
    parser.add_argument(
        '--no-returns',
        action='store_true',
        help='Skip return calculation (scores only)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    # Resolve paths
    if not os.path.isabs(args.checkpoints_dir):
        # Try relative to project root
        if not os.path.exists(args.checkpoints_dir):
            alt_path = os.path.join('..', args.checkpoints_dir)
            if os.path.exists(alt_path):
                args.checkpoints_dir = alt_path

    try:
        # Extract data
        observations = extract_all_data(
            checkpoints_dir=args.checkpoints_dir,
            prices_file=args.prices_file,
            forward_weeks=args.forward_weeks,
            verbose=not args.quiet
        )

        # Save to CSV
        save_to_csv(
            observations,
            output_path=args.output,
            include_returns=not args.no_returns and args.prices_file is not None,
            verbose=not args.quiet
        )

        # Print summary
        if not args.quiet:
            print("\n" + "="*60)
            print("EXTRACTION COMPLETE")
            print("="*60)

            dates = sorted(set(o['date'] for o in observations))
            print(f"\nDate range: {dates[0]} to {dates[-1]}")
            print(f"Unique dates: {len(dates)}")
            print(f"Total observations: {len(observations)}")

            if args.prices_file:
                with_returns = sum(1 for o in observations if o.get('fwd_4w') is not None)
                print(f"With forward returns: {with_returns}")

            print(f"\nOutput: {args.output}")

            if not args.prices_file:
                print("\nNOTE: No prices file provided.")
                print("To calculate forward returns, run with --prices-file:")
                print(f"  python -m optimization.extract_historical_data \\")
                print(f"    --checkpoints-dir {args.checkpoints_dir} \\")
                print(f"    --prices-file production_data/prices.csv \\")
                print(f"    --output {args.output}")

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == '__main__':
    sys.exit(main())
