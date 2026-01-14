"""
Scoring module for composite score integration.

Exports:
    integrate_momentum_signals_with_confidence: Confidence-weighted momentum integration
    integrate_momentum_batch: Batch integration for multiple tickers
"""

from .integrate_momentum_confidence_weighted import (
    integrate_momentum_signals_with_confidence,
    integrate_momentum_batch,
)

__all__ = [
    "integrate_momentum_signals_with_confidence",
    "integrate_momentum_batch",
]
