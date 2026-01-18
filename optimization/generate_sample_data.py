#!/usr/bin/env python3
"""
Generate sample training data for weight optimization testing.

Creates synthetic but realistic biotech scoring data with known factor returns.
Useful for testing the optimizer before real backtest data is available.

Usage:
    python optimization/generate_sample_data.py
    python optimization/generate_sample_data.py --output optimization_data/training_dataset.csv

Author: Wake Robin Capital Management
Version: 1.0.0
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np

# Component names (must match optimize_weights_scipy.py)
COMPONENT_NAMES = ['clinical', 'financial', 'catalyst', 'pos', 'momentum', 'valuation']

# Sample biotech tickers
SAMPLE_TICKERS = [
    'ACAD', 'ALNY', 'AMGN', 'BIIB', 'BMRN', 'BNTX', 'CRSP', 'EXEL',
    'GILD', 'HZNP', 'ILMN', 'INCY', 'IONS', 'JAZZ', 'MRNA', 'NBIX',
    'NTLA', 'PCVX', 'REGN', 'SGEN', 'SRPT', 'UTHR', 'VRTX', 'XLRN',
    'ABCL', 'ALKS', 'ARCT', 'BEAM', 'BGNE', 'BLUE', 'CCXI', 'CERE',
    'DAWN', 'DRNA', 'EDIT', 'FATE', 'FOLD', 'GERN', 'HALO', 'IMVT',
    'IOVA', 'KPTI', 'LEGN', 'MCRB', 'MIRM', 'NRIX', 'OCGN', 'PTGX',
    'RARE', 'RCKT', 'RXRX', 'SAGE', 'SNDX', 'TCRT', 'TGTX', 'VCNX',
    'VIR', 'XNCR', 'YMAB', 'ZLAB'
]

# True factor returns (hidden from optimizer, used to generate returns)
# These determine the "optimal" weights
TRUE_FACTOR_RETURNS = {
    'clinical': 0.003,      # 0.3% per period contribution
    'financial': 0.002,     # 0.2% per period contribution
    'catalyst': 0.0015,     # 0.15% per period contribution
    'pos': 0.0012,          # 0.12% per period contribution
    'momentum': 0.0008,     # 0.08% per period contribution
    'valuation': 0.0004,    # 0.04% per period contribution
}

# Noise parameters
IDIOSYNCRATIC_VOL = 0.10   # 10% annualized idiosyncratic volatility
SCORE_NOISE_STD = 10.0     # Standard deviation of score noise


def generate_dates(
    start_date: str = '2021-01-01',
    end_date: str = '2025-12-31',
    freq_weeks: int = 4
) -> List[str]:
    """Generate list of rebalance dates (every N weeks)."""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    dates = []
    current = start

    while current <= end:
        dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(weeks=freq_weeks)

    return dates


def generate_scores(
    n_tickers: int,
    seed: int
) -> np.ndarray:
    """
    Generate random component scores for a cross-section.

    Returns:
        Array of shape (n_tickers, 6) with scores in [0, 100]
    """
    rng = np.random.RandomState(seed)

    # Generate correlated scores (components tend to be correlated in practice)
    correlation_matrix = np.array([
        [1.0,  0.3,  0.2,  0.4,  0.1,  0.1],  # clinical
        [0.3,  1.0,  0.2,  0.2,  0.1,  0.2],  # financial
        [0.2,  0.2,  1.0,  0.3,  0.2,  0.1],  # catalyst
        [0.4,  0.2,  0.3,  1.0,  0.1,  0.1],  # pos
        [0.1,  0.1,  0.2,  0.1,  1.0,  0.3],  # momentum
        [0.1,  0.2,  0.1,  0.1,  0.3,  1.0],  # valuation
    ])

    # Cholesky decomposition for correlated samples
    L = np.linalg.cholesky(correlation_matrix)

    # Generate uncorrelated standard normal
    z = rng.randn(n_tickers, 6)

    # Apply correlation structure
    correlated = z @ L.T

    # Transform to [0, 100] using CDF
    from scipy.stats import norm
    scores = norm.cdf(correlated) * 100

    return scores


def generate_returns(
    scores: np.ndarray,
    factor_returns: dict,
    idio_vol: float,
    seed: int
) -> np.ndarray:
    """
    Generate forward returns based on scores and factor model.

    return = sum(score_i * factor_return_i) + noise

    Args:
        scores: Array of shape (n_tickers, 6)
        factor_returns: Dict of factor return contributions
        idio_vol: Idiosyncratic volatility (annualized)
        seed: Random seed

    Returns:
        Array of forward returns
    """
    rng = np.random.RandomState(seed)
    n_tickers = scores.shape[0]

    # Factor contribution (normalize scores to z-scores for factor model)
    factor_contrib = np.zeros(n_tickers)

    for i, name in enumerate(COMPONENT_NAMES):
        # Z-score the component
        z = (scores[:, i] - 50) / 25  # Rough z-score
        factor_contrib += z * factor_returns[name]

    # Add idiosyncratic noise
    # Convert annual vol to 4-week vol: vol * sqrt(4/52)
    period_vol = idio_vol * np.sqrt(4/52)
    noise = rng.randn(n_tickers) * period_vol

    returns = factor_contrib + noise

    return returns


def generate_training_data(
    output_path: str,
    start_date: str = '2021-01-01',
    end_date: str = '2025-12-31',
    n_tickers: int = 50,
    seed: int = 42,
    verbose: bool = True
) -> int:
    """
    Generate complete training dataset.

    Args:
        output_path: Path to save CSV
        start_date: Start date for data
        end_date: End date for data
        n_tickers: Number of tickers per cross-section
        seed: Master random seed
        verbose: Print progress

    Returns:
        Number of observations generated
    """
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    dates = generate_dates(start_date, end_date)
    tickers = SAMPLE_TICKERS[:n_tickers]

    if verbose:
        print(f"Generating {len(dates)} cross-sections with {len(tickers)} tickers each")
        print(f"Total observations: {len(dates) * len(tickers)}")

    # Track for determinism
    np.random.seed(seed)

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'date', 'ticker', 'clinical', 'financial', 'catalyst',
            'pos', 'momentum', 'valuation', 'fwd_4w'
        ])
        writer.writeheader()

        n_obs = 0

        for i, date in enumerate(dates):
            # Generate scores for this cross-section
            period_seed = seed + i * 1000
            scores = generate_scores(len(tickers), period_seed)

            # Generate returns
            returns = generate_returns(
                scores, TRUE_FACTOR_RETURNS, IDIOSYNCRATIC_VOL, period_seed + 1
            )

            # Write rows
            for j, ticker in enumerate(tickers):
                row = {
                    'date': date,
                    'ticker': ticker,
                    'clinical': f"{scores[j, 0]:.2f}",
                    'financial': f"{scores[j, 1]:.2f}",
                    'catalyst': f"{scores[j, 2]:.2f}",
                    'pos': f"{scores[j, 3]:.2f}",
                    'momentum': f"{scores[j, 4]:.2f}",
                    'valuation': f"{scores[j, 5]:.2f}",
                    'fwd_4w': f"{returns[j]:.6f}"
                }
                writer.writerow(row)
                n_obs += 1

    if verbose:
        print(f"Generated {n_obs} observations")
        print(f"Saved to {output_path}")

        # Print true factor returns for reference
        print("\nTrue factor returns (hidden from optimizer):")
        for name, ret in TRUE_FACTOR_RETURNS.items():
            print(f"  {name:12s}: {ret*100:.3f}% per period")

    return n_obs


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Generate sample training data for weight optimization'
    )
    parser.add_argument(
        '--output',
        default='optimization_data/training_dataset.csv',
        help='Output path for CSV file'
    )
    parser.add_argument(
        '--start-date',
        default='2021-01-01',
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        default='2025-12-31',
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--n-tickers',
        type=int,
        default=50,
        help='Number of tickers per cross-section'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    # Handle relative path - assume we're in project root or optimization dir
    output_path = args.output
    if not os.path.isabs(output_path):
        # Check if we're in the optimization directory
        if os.path.basename(os.getcwd()) == 'optimization':
            output_path = output_path
        else:
            # Assume we're in project root
            output_path = os.path.join('optimization', output_path)

    n_obs = generate_training_data(
        output_path=output_path,
        start_date=args.start_date,
        end_date=args.end_date,
        n_tickers=args.n_tickers,
        seed=args.seed,
        verbose=not args.quiet
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
