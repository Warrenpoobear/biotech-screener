"""
Signals module for momentum and technical indicators.

Exports:
    MorningstarMomentumSignals: Multi-horizon momentum signal generator
"""

from .morningstar_momentum_signals_v2 import MorningstarMomentumSignals

__all__ = ["MorningstarMomentumSignals"]
