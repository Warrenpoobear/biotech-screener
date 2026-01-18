#!/usr/bin/env python3
"""
Validate optimized weights with out-of-sample testing and stability analysis.

Provides:
1. Train/test split validation (temporal)
2. K-fold cross-validation (rolling windows)
3. Weight stability analysis across periods
4. Deployment readiness assessment

Usage:
    python -m optimization.validate_weights
    python -m optimization.validate_weights --weights-file optimal_weights_scipy.json

Author: Wake Robin Capital Management
Version: 1.0.0
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr

try:
    from .optimize_weights_scipy import (
        BASELINE_WEIGHTS,
        COMPONENT_NAMES,
        DEFAULT_BOUNDS,
        load_training_data,
        objective_function,
        optimize_slsqp,
        _resolve_data_path,
    )
except ImportError:
    # Running as script directly
    from optimize_weights_scipy import (
        BASELINE_WEIGHTS,
        COMPONENT_NAMES,
        DEFAULT_BOUNDS,
        load_training_data,
        objective_function,
        optimize_slsqp,
        _resolve_data_path,
    )

__version__ = "1.0.0"

# Validation thresholds
MIN_SHARPE_IMPROVEMENT_PCT = 10.0  # Minimum 10% improvement required
MIN_WIN_RATE_PCT = 55.0            # At least 55% of periods should improve
MAX_WEIGHT_STD = 0.05              # Maximum weight standard deviation across folds
MIN_OOS_SHARPE = 0.5               # Minimum out-of-sample Sharpe ratio


@dataclass
class ValidationResult:
    """Results from a single validation run."""
    train_sharpe: float
    test_sharpe: float
    train_ic: float
    test_ic: float
    weights: Dict[str, float]
    train_periods: int
    test_periods: int
    improvement_pct: float


@dataclass
class StabilityMetrics:
    """Weight stability metrics across validation folds."""
    mean_weights: Dict[str, float]
    std_weights: Dict[str, float]
    min_weights: Dict[str, float]
    max_weights: Dict[str, float]
    weight_ranges: Dict[str, float]


@dataclass
class DeploymentAssessment:
    """Final deployment readiness assessment."""
    is_ready: bool
    reasons: List[str]
    warnings: List[str]
    metrics: Dict[str, float]
    recommended_weights: Optional[Dict[str, float]]


def compute_ic(
    weights: np.ndarray,
    data: List[Dict[str, Any]]
) -> float:
    """Compute Information Coefficient (Spearman correlation)."""
    scores = []
    returns = []

    for obs in data:
        score = np.dot(obs['components'], weights)
        scores.append(score)
        returns.append(obs['fwd_return'])

    if len(scores) < 10:
        return 0.0

    ic, _ = spearmanr(scores, returns)
    return ic if not np.isnan(ic) else 0.0


def temporal_train_test_split(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.7
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split data temporally (earlier dates for training, later for testing).

    Args:
        data: Full dataset
        train_ratio: Fraction of dates to use for training

    Returns:
        (train_data, test_data)
    """
    # Get unique dates sorted
    dates = sorted(set(d['date'] for d in data))
    n_train_dates = int(len(dates) * train_ratio)

    train_dates = set(dates[:n_train_dates])
    test_dates = set(dates[n_train_dates:])

    train_data = [d for d in data if d['date'] in train_dates]
    test_data = [d for d in data if d['date'] in test_dates]

    return train_data, test_data


def rolling_window_splits(
    data: List[Dict[str, Any]],
    n_folds: int = 5,
    train_ratio: float = 0.6
) -> List[Tuple[List[Dict], List[Dict]]]:
    """
    Create rolling window train/test splits for cross-validation.

    Args:
        data: Full dataset
        n_folds: Number of folds
        train_ratio: Ratio of each window used for training

    Returns:
        List of (train_data, test_data) tuples
    """
    dates = sorted(set(d['date'] for d in data))
    n_dates = len(dates)

    # Calculate window size
    window_size = n_dates // n_folds
    train_size = int(window_size * train_ratio)

    splits = []

    for i in range(n_folds):
        start_idx = i * (n_dates - window_size) // (n_folds - 1) if n_folds > 1 else 0
        end_idx = min(start_idx + window_size, n_dates)

        window_dates = dates[start_idx:end_idx]
        train_dates = set(window_dates[:train_size])
        test_dates = set(window_dates[train_size:])

        if not train_dates or not test_dates:
            continue

        train_data = [d for d in data if d['date'] in train_dates]
        test_data = [d for d in data if d['date'] in test_dates]

        if len(train_data) > 50 and len(test_data) > 20:
            splits.append((train_data, test_data))

    return splits


def evaluate_fixed_weights(
    weights: np.ndarray,
    test_data: List[Dict],
    baseline_weights: np.ndarray = BASELINE_WEIGHTS
) -> Dict[str, float]:
    """
    Evaluate fixed weights on test data (no re-optimization).

    This tests the SAVED weights directly, not re-optimized weights.
    """
    test_sharpe_optimal = -objective_function(weights, test_data, 'sharpe')
    test_sharpe_baseline = -objective_function(baseline_weights, test_data, 'sharpe')

    test_ic_optimal = compute_ic(weights, test_data)
    test_ic_baseline = compute_ic(baseline_weights, test_data)

    improvement = (test_sharpe_optimal / test_sharpe_baseline - 1) * 100 if test_sharpe_baseline > 0 else 0

    return {
        'optimized_sharpe': float(test_sharpe_optimal),
        'baseline_sharpe': float(test_sharpe_baseline),
        'optimized_ic': float(test_ic_optimal),
        'baseline_ic': float(test_ic_baseline),
        'improvement_pct': float(improvement),
        'n_observations': len(test_data),
        'n_periods': len(set(d['date'] for d in test_data))
    }


def validate_single_split(
    train_data: List[Dict],
    test_data: List[Dict],
    baseline_weights: np.ndarray = BASELINE_WEIGHTS
) -> ValidationResult:
    """
    Run validation on a single train/test split.

    Optimizes on training data, evaluates on test data.
    """
    # Optimize on training data
    optimal_weights, train_sharpe, _ = optimize_slsqp(train_data)

    # Evaluate on test data
    test_sharpe_optimal = -objective_function(optimal_weights, test_data, 'sharpe')
    test_sharpe_baseline = -objective_function(baseline_weights, test_data, 'sharpe')

    # Compute ICs
    train_ic = compute_ic(optimal_weights, train_data)
    test_ic = compute_ic(optimal_weights, test_data)

    # Count periods
    train_periods = len(set(d['date'] for d in train_data))
    test_periods = len(set(d['date'] for d in test_data))

    # Improvement
    improvement = (test_sharpe_optimal / test_sharpe_baseline - 1) * 100 if test_sharpe_baseline > 0 else 0

    return ValidationResult(
        train_sharpe=train_sharpe,
        test_sharpe=test_sharpe_optimal,
        train_ic=train_ic,
        test_ic=test_ic,
        weights={name: float(w) for name, w in zip(COMPONENT_NAMES, optimal_weights)},
        train_periods=train_periods,
        test_periods=test_periods,
        improvement_pct=improvement
    )


def run_cross_validation(
    data: List[Dict[str, Any]],
    n_folds: int = 5,
    verbose: bool = True
) -> Tuple[List[ValidationResult], StabilityMetrics]:
    """
    Run k-fold cross-validation with rolling windows.

    Returns:
        (list of ValidationResults, StabilityMetrics)
    """
    splits = rolling_window_splits(data, n_folds=n_folds)

    if verbose:
        print(f"\nRunning {len(splits)}-fold cross-validation...")

    results = []
    all_weights = {name: [] for name in COMPONENT_NAMES}

    for i, (train_data, test_data) in enumerate(splits):
        if verbose:
            print(f"\n  Fold {i+1}/{len(splits)}:")
            print(f"    Train: {len(train_data)} obs, Test: {len(test_data)} obs")

        result = validate_single_split(train_data, test_data)
        results.append(result)

        # Collect weights
        for name in COMPONENT_NAMES:
            all_weights[name].append(result.weights[name])

        if verbose:
            print(f"    Train Sharpe: {result.train_sharpe:.4f}")
            print(f"    Test Sharpe:  {result.test_sharpe:.4f}")
            print(f"    Test IC:      {result.test_ic:.4f}")
            print(f"    Improvement:  {result.improvement_pct:+.1f}%")

    # Compute stability metrics
    stability = StabilityMetrics(
        mean_weights={name: float(np.mean(weights)) for name, weights in all_weights.items()},
        std_weights={name: float(np.std(weights)) for name, weights in all_weights.items()},
        min_weights={name: float(np.min(weights)) for name, weights in all_weights.items()},
        max_weights={name: float(np.max(weights)) for name, weights in all_weights.items()},
        weight_ranges={name: float(np.max(weights) - np.min(weights)) for name, weights in all_weights.items()}
    )

    return results, stability


def assess_deployment_readiness(
    cv_results: List[ValidationResult],
    stability: StabilityMetrics,
    oos_result: Optional[ValidationResult] = None,
    direct_result: Optional[Dict[str, float]] = None,
    saved_weights: Optional[Dict[str, float]] = None
) -> DeploymentAssessment:
    """
    Assess whether optimized weights are ready for deployment.

    PRIMARY criteria (from direct_result - testing saved weights):
    1. Direct OOS Sharpe improvement >= 10%
    2. Direct OOS Sharpe >= 0.5

    SECONDARY criteria (from CV - informational):
    3. Weight stability across CV folds
    4. Mean CV Sharpe
    """
    reasons = []
    warnings = []
    is_ready = True

    # PRIMARY: Use direct validation of saved weights
    if direct_result:
        direct_improvement = direct_result['improvement_pct']
        direct_sharpe = direct_result['optimized_sharpe']
        direct_ic = direct_result['optimized_ic']

        # Check 1: Direct improvement (PRIMARY)
        if direct_improvement >= MIN_SHARPE_IMPROVEMENT_PCT:
            reasons.append(f"✓ Direct OOS improvement: {direct_improvement:+.1f}% (threshold: {MIN_SHARPE_IMPROVEMENT_PCT}%)")
        else:
            reasons.append(f"✗ Direct OOS improvement: {direct_improvement:+.1f}% (threshold: {MIN_SHARPE_IMPROVEMENT_PCT}%)")
            is_ready = False

        # Check 2: Direct Sharpe (PRIMARY)
        if direct_sharpe >= MIN_OOS_SHARPE:
            reasons.append(f"✓ Direct OOS Sharpe: {direct_sharpe:.4f} (threshold: {MIN_OOS_SHARPE})")
        else:
            reasons.append(f"✗ Direct OOS Sharpe: {direct_sharpe:.4f} (threshold: {MIN_OOS_SHARPE})")
            is_ready = False
    else:
        # Fallback to CV results if no direct result
        improvements = [r.improvement_pct for r in cv_results]
        mean_improvement = np.mean(improvements)
        if mean_improvement >= MIN_SHARPE_IMPROVEMENT_PCT:
            reasons.append(f"✓ Mean CV improvement: {mean_improvement:+.1f}%")
        else:
            reasons.append(f"✗ Mean CV improvement: {mean_improvement:+.1f}%")
            is_ready = False

    # SECONDARY: CV metrics (informational)
    oos_sharpes = [r.test_sharpe for r in cv_results]
    mean_oos_sharpe = np.mean(oos_sharpes)
    max_weight_std = max(stability.std_weights.values())

    # Check if CV optimizer failed to find improvements (all baseline)
    cv_all_baseline = max_weight_std == 0.0
    if cv_all_baseline:
        warnings.append("CV re-optimization returned baseline weights (insufficient data per fold)")
        # Don't fail on stability if CV couldn't optimize
    else:
        if max_weight_std <= MAX_WEIGHT_STD:
            reasons.append(f"✓ CV weight stability: max std {max_weight_std:.4f}")
        else:
            reasons.append(f"⚠ CV weight stability: max std {max_weight_std:.4f} (high variance)")
            warnings.append("High CV weight variance - consider larger training windows")

    reasons.append(f"ℹ Mean CV Sharpe: {mean_oos_sharpe:.4f}")

    # Build metrics
    metrics = {
        'direct_improvement_pct': float(direct_result['improvement_pct']) if direct_result else 0.0,
        'direct_sharpe': float(direct_result['optimized_sharpe']) if direct_result else 0.0,
        'mean_cv_sharpe': float(mean_oos_sharpe),
        'max_weight_std': float(max_weight_std),
        'n_folds': len(cv_results)
    }

    # Recommended weights: use the SAVED weights if passing, not CV mean
    recommended_weights = None
    if is_ready and saved_weights:
        recommended_weights = saved_weights

    return DeploymentAssessment(
        is_ready=is_ready,
        reasons=reasons,
        warnings=warnings,
        metrics=metrics,
        recommended_weights=recommended_weights
    )


def validate_from_file(
    weights_file: str,
    training_data_file: str,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Validate weights from a saved optimization result file.

    Args:
        weights_file: Path to optimal_weights_scipy.json
        training_data_file: Path to training CSV
        verbose: Print progress

    Returns:
        Validation results dictionary
    """
    # Load weights
    with open(weights_file) as f:
        saved_result = json.load(f)

    weights = saved_result['weights']
    weight_array = np.array([weights[name] for name in COMPONENT_NAMES])

    if verbose:
        print("="*60)
        print("WEIGHT VALIDATION")
        print("="*60)
        print(f"\nLoaded weights from: {weights_file}")
        print(f"Original Sharpe: {saved_result['sharpe']:.4f}")
        print(f"Reported improvement: {saved_result['improvement_pct']:.1f}%")

    # Load data
    data = load_training_data(training_data_file)

    if verbose:
        print(f"\nLoaded {len(data)} observations from {training_data_file}")
        print(f"\nSaved optimal weights:")
        for name in COMPONENT_NAMES:
            baseline = BASELINE_WEIGHTS[COMPONENT_NAMES.index(name)]
            delta = weights[name] - baseline
            print(f"  {name:12s}: {weights[name]:.4f} (baseline: {baseline:.2f}, delta: {delta:+.4f})")

    # SECTION 1: Direct test of SAVED weights on held-out data
    if verbose:
        print("\n" + "-"*60)
        print("1. DIRECT VALIDATION OF SAVED WEIGHTS (70/30 split)")
        print("-"*60)

    train_data, test_data = temporal_train_test_split(data, train_ratio=0.7)

    # Test the SAVED weights directly (no re-optimization)
    direct_result = evaluate_fixed_weights(weight_array, test_data)

    if verbose:
        print(f"\n  Test periods:       {direct_result['n_periods']}")
        print(f"  Test observations:  {direct_result['n_observations']}")
        print(f"  Optimized Sharpe:   {direct_result['optimized_sharpe']:.4f}")
        print(f"  Baseline Sharpe:    {direct_result['baseline_sharpe']:.4f}")
        print(f"  Improvement:        {direct_result['improvement_pct']:+.1f}%")
        print(f"  Optimized IC:       {direct_result['optimized_ic']:.4f}")
        print(f"  Baseline IC:        {direct_result['baseline_ic']:.4f}")

    # SECTION 2: Cross-validation (re-optimizes to test stability)
    if verbose:
        print("\n" + "-"*60)
        print("2. CROSS-VALIDATION STABILITY (re-optimizes each fold)")
        print("-"*60)

    # Also run the re-optimization validation for comparison
    oos_result = validate_single_split(train_data, test_data)

    if verbose:
        print(f"  Test Sharpe:   {oos_result.test_sharpe:.4f}")
        print(f"  Improvement:   {oos_result.improvement_pct:+.1f}%")

    # Run cross-validation
    if verbose:
        print("\n" + "-"*60)
        print("3. ROLLING WINDOW CROSS-VALIDATION")
        print("-"*60)

    cv_results, stability = run_cross_validation(data, n_folds=5, verbose=verbose)

    # Print stability analysis
    if verbose:
        print("\n" + "-"*60)
        print("4. WEIGHT STABILITY ANALYSIS (from CV re-optimization)")
        print("-"*60)
        print("\n  Component       Mean     Std      Range")
        print("  " + "-"*45)
        for name in COMPONENT_NAMES:
            mean = stability.mean_weights[name]
            std = stability.std_weights[name]
            rng = stability.weight_ranges[name]
            baseline = BASELINE_WEIGHTS[COMPONENT_NAMES.index(name)]
            print(f"  {name:12s}  {mean:.4f}   {std:.4f}   {rng:.4f}  (baseline: {baseline:.2f})")

    # Assess deployment readiness (use direct validation as primary metric)
    assessment = assess_deployment_readiness(
        cv_results, stability, oos_result,
        direct_result=direct_result,
        saved_weights=weights
    )

    if verbose:
        print("\n" + "-"*60)
        print("5. DEPLOYMENT ASSESSMENT")
        print("-"*60)

        # First show the direct validation result (most important)
        print(f"\n  PRIMARY METRIC (saved weights on held-out data):")
        print(f"    Sharpe improvement: {direct_result['improvement_pct']:+.1f}%")

        for reason in assessment.reasons:
            print(f"  {reason}")

        if assessment.warnings:
            print("\n  Warnings:")
            for warning in assessment.warnings:
                print(f"    ⚠ {warning}")

        print("\n" + "="*60)
        if assessment.is_ready:
            print("RECOMMENDATION: ✓ DEPLOY OPTIMIZED WEIGHTS")
            print("="*60)
            print("\nRecommended weights (validated on held-out data):")
            for name, weight in assessment.recommended_weights.items():
                baseline = BASELINE_WEIGHTS[COMPONENT_NAMES.index(name)]
                delta = weight - baseline
                print(f"  {name:12s}: {weight:.4f} (delta: {delta:+.4f})")
        else:
            print("RECOMMENDATION: ✗ KEEP BASELINE WEIGHTS")
            print("="*60)
            print("\nOptimization did not meet deployment criteria.")
            print("Consider:")
            print("  - Collecting more training data")
            print("  - Adjusting weight bounds")
            print("  - Using more conservative optimization")

    # Build output
    output = {
        'validation_version': __version__,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'source_weights_file': weights_file,
        'training_data_file': training_data_file,
        'saved_weights': weights,
        'direct_validation': direct_result,  # Testing SAVED weights on held-out data
        'temporal_split_reoptimized': {
            'train_sharpe': oos_result.train_sharpe,
            'test_sharpe': oos_result.test_sharpe,
            'test_ic': oos_result.test_ic,
            'improvement_pct': oos_result.improvement_pct
        },
        'cross_validation': {
            'n_folds': len(cv_results),
            'mean_test_sharpe': float(np.mean([r.test_sharpe for r in cv_results])),
            'mean_test_ic': float(np.mean([r.test_ic for r in cv_results])),
            'mean_improvement_pct': float(np.mean([r.improvement_pct for r in cv_results])),
            'improvements_by_fold': [r.improvement_pct for r in cv_results]
        },
        'stability': {
            'mean_weights': stability.mean_weights,
            'std_weights': stability.std_weights,
            'weight_ranges': stability.weight_ranges
        },
        'assessment': {
            'is_ready': assessment.is_ready,
            'reasons': assessment.reasons,
            'warnings': assessment.warnings,
            'metrics': assessment.metrics,
            'recommended_weights': assessment.recommended_weights
        }
    }

    return output


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Validate optimized weights with out-of-sample testing'
    )
    parser.add_argument(
        '--weights-file',
        default='optimization_data/optimal_weights_scipy.json',
        help='Path to optimized weights JSON'
    )
    parser.add_argument(
        '--training-data',
        default='optimization_data/training_dataset.csv',
        help='Path to training CSV file'
    )
    parser.add_argument(
        '--output',
        default='optimization_data/validation_results.json',
        help='Output path for validation results'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    # Resolve paths
    weights_path = _resolve_data_path(args.weights_file)
    training_path = _resolve_data_path(args.training_data)
    output_path = _resolve_data_path(args.output)

    try:
        results = validate_from_file(
            weights_file=weights_path,
            training_data_file=training_path,
            verbose=not args.quiet
        )

        # Save results
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        if not args.quiet:
            print(f"\nSaved validation results to {output_path}")

        # Return exit code based on deployment readiness
        return 0 if results['assessment']['is_ready'] else 1

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure you have run optimization first:")
        print("  python -m optimization.generate_sample_data")
        print("  python -m optimization.optimize_weights_scipy")
        return 1

    except Exception as e:
        print(f"Error during validation: {e}")
        raise


if __name__ == '__main__':
    sys.exit(main())
