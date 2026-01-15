"""
Scoring module for composite score integration.

Exports:
    integrate_momentum_signals_with_confidence: Confidence-weighted momentum integration
"""

from .integrate_momentum_confidence_weighted import integrate_momentum_signals_with_confidence

__all__ = ["integrate_momentum_signals_with_confidence"]
