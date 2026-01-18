#!/usr/bin/env python3
"""
Scipy-based weight optimization for biotech-screener.

Faster than grid search (minutes vs hours).
Uses SLSQP (Sequential Least Squares Programming) for constrained optimization,
plus Differential Evolution for global optimization.

Design Philosophy:
- DETERMINISTIC: Fixed random seeds for reproducibility
- STDLIB-FRIENDLY: Uses scipy.optimize + numpy only
- FAIL-LOUD: Clear validation and error reporting
- AUDITABLE: Full provenance chain in output

Author: Wake Robin Capital Management
Version: 1.0.0
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import rankdata, spearmanr

__version__ = "1.0.0"

# =============================================================================
# CONSTANTS
# =============================================================================

# Component names (must match module_5_composite_v3.py)
COMPONENT_NAMES = ['clinical', 'financial', 'catalyst', 'pos', 'momentum', 'valuation']

# Default baseline weights (from V3_ENHANCED_WEIGHTS)
BASELINE_WEIGHTS = np.array([0.28, 0.25, 0.17, 0.15, 0.10, 0.05])

# Default weight bounds
DEFAULT_BOUNDS = [
    (0.20, 0.40),  # clinical
    (0.15, 0.35),  # financial
    (0.10, 0.25),  # catalyst
    (0.10, 0.25),  # pos
    (0.05, 0.20),  # momentum
    (0.00, 0.15),  # valuation
]

# Annualization factor for weekly data (52 weeks/year, assuming 4-week periods = 13)
ANNUALIZATION_FACTOR = np.sqrt(13)

# Default random seed for reproducibility
DEFAULT_SEED = 42


# =============================================================================
# DATA LOADING
# =============================================================================

def load_training_data(csv_file: str) -> List[Dict[str, Any]]:
    """
    Load prepared optimization dataset from CSV.

    Expected CSV columns:
    - date: ISO date string (YYYY-MM-DD)
    - ticker: Stock ticker symbol
    - clinical, financial, catalyst, pos, momentum, valuation: Component scores (0-100)
    - fwd_4w: Forward 4-week return

    Returns:
        List of observation dictionaries with 'components' as numpy array

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV has missing/invalid data
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Training data not found: {csv_file}")

    data = []
    line_num = 0

    with open(csv_file, 'r', newline='') as f:
        reader = csv.DictReader(f)

        # Validate header
        required_cols = {'date', 'ticker', 'clinical', 'financial', 'catalyst',
                        'pos', 'momentum', 'valuation', 'fwd_4w'}
        if not required_cols.issubset(set(reader.fieldnames or [])):
            missing = required_cols - set(reader.fieldnames or [])
            raise ValueError(f"Missing required columns: {missing}")

        for row in reader:
            line_num += 1
            try:
                components = np.array([
                    float(row['clinical']),
                    float(row['financial']),
                    float(row['catalyst']),
                    float(row['pos']),
                    float(row['momentum']),
                    float(row['valuation'])
                ])

                # Validate component scores are in [0, 100] range
                if np.any(components < 0) or np.any(components > 100):
                    print(f"Warning: Line {line_num} has component scores outside [0, 100]")

                data.append({
                    'date': row['date'],
                    'ticker': row['ticker'],
                    'components': components,
                    'fwd_return': float(row['fwd_4w'])
                })
            except (ValueError, KeyError) as e:
                raise ValueError(f"Invalid data at line {line_num}: {e}")

    if not data:
        raise ValueError("No valid data rows found in CSV")

    return data


# =============================================================================
# OBJECTIVE FUNCTIONS
# =============================================================================

def spearman_corr_numpy(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate Spearman rank correlation using numpy.

    Returns NaN if inputs have zero variance.
    """
    if len(x) != len(y) or len(x) < 2:
        return np.nan

    x_ranks = rankdata(x)
    y_ranks = rankdata(y)

    # Check for zero variance
    if np.std(x_ranks) == 0 or np.std(y_ranks) == 0:
        return np.nan

    return np.corrcoef(x_ranks, y_ranks)[0, 1]


def objective_function(
    weights: np.ndarray,
    data: List[Dict[str, Any]],
    metric: str = 'sharpe'
) -> float:
    """
    Objective function to minimize (returns negative Sharpe or negative IC).

    Args:
        weights: numpy array [clinical, financial, catalyst, pos, momentum, valuation]
        data: Training observations
        metric: 'sharpe' for Sharpe ratio, 'ic' for information coefficient

    Returns:
        Negative metric value (for minimization)
    """
    # Group observations by date
    by_date: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for obs in data:
        score = np.dot(obs['components'], weights)
        by_date[obs['date']].append((score, obs['fwd_return']))

    if metric == 'sharpe':
        return _compute_negative_sharpe(by_date)
    else:  # IC
        return _compute_negative_ic(by_date)


def _compute_negative_sharpe(
    by_date: Dict[str, List[Tuple[float, float]]]
) -> float:
    """Compute negative Sharpe ratio from long-short portfolio returns."""
    returns = []

    for date, scored in by_date.items():
        # Sort by score (high to low)
        scored_sorted = sorted(scored, key=lambda x: x[0], reverse=True)
        n = len(scored_sorted)

        if n < 3:
            continue  # Need at least 3 for meaningful decile

        # Top/bottom decile (at least 1)
        top_n = max(1, n // 10)

        long_rets = [ret for _, ret in scored_sorted[:top_n]]
        short_rets = [ret for _, ret in scored_sorted[-top_n:]]

        ls_return = np.mean(long_rets) - np.mean(short_rets)
        returns.append(ls_return)

    # Calculate Sharpe ratio
    if len(returns) < 2:
        return 999.0  # Penalty for insufficient data

    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)

    if std_ret == 0 or np.isnan(std_ret):
        return 999.0  # Penalty for zero volatility

    sharpe = (mean_ret / std_ret) * ANNUALIZATION_FACTOR
    return -sharpe  # Negative because we minimize


def _compute_negative_ic(
    by_date: Dict[str, List[Tuple[float, float]]]
) -> float:
    """Compute negative Information Coefficient (Spearman correlation)."""
    all_scores = []
    all_returns = []

    for date, scored in by_date.items():
        for score, ret in scored:
            all_scores.append(score)
            all_returns.append(ret)

    if len(all_scores) < 10:
        return 999.0  # Penalty for insufficient data

    ic = spearman_corr_numpy(np.array(all_scores), np.array(all_returns))

    if np.isnan(ic):
        return 999.0

    return -ic  # Negative because we minimize


# =============================================================================
# OPTIMIZATION METHODS
# =============================================================================

def optimize_slsqp(
    data: List[Dict[str, Any]],
    initial_weights: Optional[np.ndarray] = None,
    bounds: Optional[List[Tuple[float, float]]] = None,
    metric: str = 'sharpe'
) -> Tuple[np.ndarray, float, Any]:
    """
    Optimize using SLSQP (Sequential Least Squares Programming).

    Fast, deterministic (given same starting point), handles constraints well.

    Args:
        data: Training observations
        initial_weights: Starting point (defaults to BASELINE_WEIGHTS)
        bounds: Weight bounds per component (defaults to DEFAULT_BOUNDS)
        metric: Optimization target ('sharpe' or 'ic')

    Returns:
        Tuple of (optimal_weights, optimal_metric_value, scipy_result)
    """
    if initial_weights is None:
        initial_weights = BASELINE_WEIGHTS.copy()

    if bounds is None:
        bounds = DEFAULT_BOUNDS

    # Constraint: weights must sum to 1.0
    constraints = {
        'type': 'eq',
        'fun': lambda w: np.sum(w) - 1.0
    }

    result = minimize(
        objective_function,
        x0=initial_weights,
        args=(data, metric),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-9, 'disp': False}
    )

    if not result.success:
        print(f"Warning: SLSQP did not converge: {result.message}")

    optimal_weights = result.x
    optimal_metric = -result.fun  # Negate back to positive

    return optimal_weights, optimal_metric, result


def optimize_differential_evolution(
    data: List[Dict[str, Any]],
    bounds: Optional[List[Tuple[float, float]]] = None,
    metric: str = 'sharpe',
    seed: int = DEFAULT_SEED,
    maxiter: int = 1000,
    popsize: int = 15,
    verbose: bool = False
) -> Tuple[np.ndarray, float, Any]:
    """
    Global optimization using Differential Evolution.

    Slower than SLSQP but explores solution space more thoroughly.
    Good for finding global optimum vs local optimum.

    Args:
        data: Training observations
        bounds: Weight bounds per component
        metric: Optimization target
        seed: Random seed for reproducibility
        maxiter: Maximum iterations
        popsize: Population size multiplier
        verbose: Print progress

    Returns:
        Tuple of (optimal_weights, optimal_metric_value, scipy_result)
    """
    if bounds is None:
        bounds = DEFAULT_BOUNDS

    # Penalty method: add penalty when sum != 1
    def penalized_objective(weights: np.ndarray, data: List[Dict]) -> float:
        penalty = 1000 * abs(np.sum(weights) - 1.0)
        return objective_function(weights, data, metric) + penalty

    if verbose:
        print("Running Differential Evolution optimization...")
        print("This may take 5-10 minutes...")

    result = differential_evolution(
        penalized_objective,
        bounds=bounds,
        args=(data,),
        strategy='best1bin',
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-7,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=seed,
        workers=1,  # Single-threaded for determinism
        disp=verbose
    )

    optimal_weights = result.x

    # Normalize to ensure sum = 1.0 (may drift slightly due to penalty method)
    optimal_weights = optimal_weights / np.sum(optimal_weights)

    # Re-evaluate with normalized weights
    optimal_metric = -objective_function(optimal_weights, data, metric)

    return optimal_weights, optimal_metric, result


def multi_start_optimization(
    data: List[Dict[str, Any]],
    n_starts: int = 10,
    bounds: Optional[List[Tuple[float, float]]] = None,
    metric: str = 'sharpe',
    seed: int = DEFAULT_SEED,
    verbose: bool = True
) -> Tuple[np.ndarray, float, List[Dict[str, Any]]]:
    """
    Run SLSQP from multiple random starting points.

    Helps avoid local optima while remaining faster than Differential Evolution.

    Args:
        data: Training observations
        n_starts: Number of random starting points
        bounds: Weight bounds per component
        metric: Optimization target
        seed: Random seed for reproducibility
        verbose: Print progress

    Returns:
        Tuple of (best_weights, best_metric, all_results)
    """
    if bounds is None:
        bounds = DEFAULT_BOUNDS

    np.random.seed(seed)

    best_weights = None
    best_metric = -999.0
    all_results = []

    if verbose:
        print(f"Running {n_starts} optimization attempts from different starting points...")

    for i in range(n_starts):
        # Random starting point using Dirichlet distribution (sums to 1)
        start_weights = np.random.dirichlet(np.ones(len(COMPONENT_NAMES)))

        # Clip to bounds
        lower_bounds = np.array([b[0] for b in bounds])
        upper_bounds = np.array([b[1] for b in bounds])
        start_weights = np.clip(start_weights, lower_bounds, upper_bounds)

        # Re-normalize to sum to 1
        start_weights = start_weights / np.sum(start_weights)

        # Optimize from this starting point
        weights, metric_val, result = optimize_slsqp(
            data, initial_weights=start_weights, bounds=bounds, metric=metric
        )

        all_results.append({
            'start': start_weights.tolist(),
            'optimal': weights.tolist(),
            metric: float(metric_val),
            'success': result.success
        })

        if metric_val > best_metric:
            best_metric = metric_val
            best_weights = weights

        if verbose:
            print(f"  Start {i+1}/{n_starts}: {metric.capitalize()} = {metric_val:.4f}")

    return best_weights, best_metric, all_results


def compare_methods(
    data: List[Dict[str, Any]],
    verbose: bool = True
) -> Tuple[np.ndarray, float, Dict[str, Dict[str, Any]]]:
    """
    Compare all optimization methods and return best result.

    Runs:
    1. SLSQP from current production weights
    2. Multi-start SLSQP (10 random starts)
    3. Differential Evolution (global optimizer)

    Args:
        data: Training observations
        verbose: Print progress

    Returns:
        Tuple of (best_weights, best_sharpe, all_results_dict)
    """
    results = {}

    # Method 1: SLSQP from current weights
    if verbose:
        print("\n" + "="*60)
        print("Method 1: SLSQP (single start from baseline)")
        print("="*60)

    w1, s1, r1 = optimize_slsqp(data)
    results['slsqp_single'] = {
        'weights': w1.tolist(),
        'sharpe': float(s1),
        'success': r1.success
    }

    if verbose:
        print(f"Sharpe: {s1:.4f}")

    # Method 2: Multi-start SLSQP
    if verbose:
        print("\n" + "="*60)
        print("Method 2: Multi-start SLSQP (10 starts)")
        print("="*60)

    w2, s2, r2 = multi_start_optimization(data, n_starts=10, verbose=verbose)
    results['slsqp_multi'] = {
        'weights': w2.tolist(),
        'sharpe': float(s2),
        'all_starts': r2
    }

    if verbose:
        print(f"Best Sharpe: {s2:.4f}")

    # Method 3: Differential Evolution
    if verbose:
        print("\n" + "="*60)
        print("Method 3: Differential Evolution (global optimizer)")
        print("="*60)

    w3, s3, r3 = optimize_differential_evolution(data, verbose=verbose)
    results['diff_evolution'] = {
        'weights': w3.tolist(),
        'sharpe': float(s3),
        'success': r3.success
    }

    if verbose:
        print(f"Sharpe: {s3:.4f}")

    # Select best method
    method_scores = {
        'slsqp_single': s1,
        'slsqp_multi': s2,
        'diff_evolution': s3
    }
    best_method = max(method_scores, key=method_scores.get)
    best_weights = results[best_method]['weights']
    best_sharpe = results[best_method]['sharpe']

    if verbose:
        print("\n" + "="*60)
        print("BEST METHOD")
        print("="*60)
        print(f"Winner: {best_method}")
        print(f"Sharpe: {best_sharpe:.4f}")

    return np.array(best_weights), best_sharpe, results


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def format_weights_output(
    optimal_weights: np.ndarray,
    optimal_sharpe: float,
    baseline_sharpe: float,
    method: str = 'scipy_comparison',
    all_results: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Format optimization results for output/storage.

    Returns dictionary suitable for JSON serialization with provenance.
    """
    weights_dict = {
        name: float(weight)
        for name, weight in zip(COMPONENT_NAMES, optimal_weights)
    }

    improvement_pct = (optimal_sharpe / baseline_sharpe - 1) * 100 if baseline_sharpe > 0 else 0.0

    output = {
        'weights': weights_dict,
        'sharpe': float(optimal_sharpe),
        'baseline_sharpe': float(baseline_sharpe),
        'improvement_pct': float(improvement_pct),
        'method': method,
        'version': __version__,
        'component_names': COMPONENT_NAMES,
        'bounds': {
            name: {'min': bounds[0], 'max': bounds[1]}
            for name, bounds in zip(COMPONENT_NAMES, DEFAULT_BOUNDS)
        },
        'provenance': {
            'optimizer_version': __version__,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'baseline_weights': {
                name: float(w) for name, w in zip(COMPONENT_NAMES, BASELINE_WEIGHTS)
            }
        }
    }

    if all_results:
        output['all_methods'] = {
            k: {'sharpe': float(v.get('sharpe', 0))}
            for k, v in all_results.items()
        }

    return output


def print_weights_comparison(
    optimal_weights: np.ndarray,
    baseline_weights: np.ndarray = BASELINE_WEIGHTS
) -> None:
    """Print formatted comparison of optimal vs baseline weights."""
    print("\n" + "="*60)
    print("OPTIMAL WEIGHTS")
    print("="*60)

    for name, opt, base in zip(COMPONENT_NAMES, optimal_weights, baseline_weights):
        delta = opt - base
        print(f"{name:12s}: {opt:.4f} (baseline: {base:.4f}, delta: {delta:+.4f})")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_optimization(
    training_data_path: str,
    output_path: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run complete scipy optimization workflow.

    Args:
        training_data_path: Path to training CSV
        output_path: Optional path to save results JSON
        verbose: Print progress

    Returns:
        Optimization results dictionary
    """
    # Load data
    if verbose:
        print("Loading training data...")

    data = load_training_data(training_data_path)

    if verbose:
        print(f"Loaded {len(data)} observations")

        # Show date range
        dates = sorted(set(d['date'] for d in data))
        print(f"Date range: {dates[0]} to {dates[-1]}")
        print(f"Unique dates: {len(dates)}")

    # Calculate baseline
    baseline_sharpe = -objective_function(BASELINE_WEIGHTS, data, 'sharpe')

    if verbose:
        print(f"\nBaseline Sharpe: {baseline_sharpe:.4f}")

    # Compare methods
    optimal_weights, optimal_sharpe, all_results = compare_methods(data, verbose=verbose)

    # Print comparison
    if verbose:
        print_weights_comparison(optimal_weights)
        print(f"\nSharpe improvement: {(optimal_sharpe/baseline_sharpe - 1)*100:+.1f}%")

    # Format output
    output = format_weights_output(
        optimal_weights,
        optimal_sharpe,
        baseline_sharpe,
        method='scipy_comparison',
        all_results=all_results
    )

    # Save if path provided
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        if verbose:
            print(f"\nSaved to {output_path}")

    return output


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Scipy-based weight optimization for biotech-screener'
    )
    parser.add_argument(
        '--training-data',
        default='optimization_data/training_dataset.csv',
        help='Path to training CSV file'
    )
    parser.add_argument(
        '--output',
        default='optimization_data/optimal_weights_scipy.json',
        help='Output path for results JSON'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    try:
        result = run_optimization(
            training_data_path=args.training_data,
            output_path=args.output,
            verbose=not args.quiet
        )

        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETE")
        print("="*60)

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nTo create sample training data, run:")
        print("  python optimization/generate_sample_data.py")
        return 1

    except Exception as e:
        print(f"Error during optimization: {e}")
        raise


if __name__ == '__main__':
    sys.exit(main())
