"""
Optimization package for biotech-screener.

Provides scipy-based weight optimization tools for the composite scoring module.

Modules:
    optimize_weights_scipy: Fast optimization using SLSQP and Differential Evolution
    generate_sample_data: Generate synthetic training data for testing
    validate_weights: Out-of-sample validation and deployment readiness
    extract_historical_data: Extract scores from checkpoint files for training
    fetch_price_data: Fetch historical price data from Yahoo Finance

Usage:
    # From project root
    pip install -e ".[optimization]"

    # Generate sample data
    python -m optimization.generate_sample_data

    # Run optimization
    python -m optimization.optimize_weights_scipy

    # Validate optimized weights
    python -m optimization.validate_weights

    # Fetch price data (requires yfinance)
    python -m optimization.fetch_price_data --checkpoint-dir checkpoints

    # Extract real data from checkpoints
    python -m optimization.extract_historical_data --checkpoint-dir checkpoints

Author: Wake Robin Capital Management
Version: 1.0.0
"""

__version__ = "1.0.0"

# Lazy imports to avoid requiring scipy unless actually used
__all__ = [
    'run_optimization',
    'optimize_slsqp',
    'optimize_differential_evolution',
    'multi_start_optimization',
    'compare_methods',
    'load_training_data',
    'generate_training_data',
    'validate_from_file',
    'run_cross_validation',
    'HistoricalDataExtractor',
    'PriceDataFetcher',
]


def __getattr__(name):
    """Lazy import to avoid requiring scipy at package import time."""
    if name in ('run_optimization', 'optimize_slsqp', 'optimize_differential_evolution',
                'multi_start_optimization', 'compare_methods', 'load_training_data'):
        from . import optimize_weights_scipy as opt
        return getattr(opt, name)
    elif name == 'generate_training_data':
        from . import generate_sample_data as gen
        return gen.generate_training_data
    elif name in ('validate_from_file', 'run_cross_validation'):
        from . import validate_weights as val
        return getattr(val, name)
    elif name == 'HistoricalDataExtractor':
        from . import extract_historical_data as ext
        return ext.HistoricalDataExtractor
    elif name == 'PriceDataFetcher':
        from . import fetch_price_data as fetch
        return fetch.PriceDataFetcher
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
