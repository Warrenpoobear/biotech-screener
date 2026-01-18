"""
Optimization package for biotech-screener.

Provides scipy-based weight optimization tools for the composite scoring module.

Modules:
    optimize_weights_scipy: Fast optimization using SLSQP and Differential Evolution
    generate_sample_data: Generate synthetic training data for testing

Usage:
    # From project root
    pip install -e ".[optimization]"

    # Generate sample data
    python -m optimization.generate_sample_data

    # Run optimization
    python -m optimization.optimize_weights_scipy

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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
