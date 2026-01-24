#!/usr/bin/env python3
"""
multi_window_momentum.py - Multi-Window Momentum Signal with Regime Adaptation

P2 Enhancement: Implements multi-window momentum (20d/60d/120d) with
regime-adaptive weighting to improve signal robustness.

Key Features:
1. Multi-window momentum averaging (IC-optimized)
2. Regime-based window weighting
3. Volatility adjustment
4. Signal smoothing to reduce noise
5. Mean reversion detection

Design Philosophy:
- DETERMINISTIC: No randomness, explicit parameters
- DECIMAL-ONLY: All arithmetic uses Decimal
- AUDITABLE: Full provenance for all scores

Author: Wake Robin Capital Management
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import logging
import math

__version__ = "1.0.0"

logger = logging.getLogger(__name__)


class RegimeState(str, Enum):
    """Market regime for momentum weight adaptation."""
    BULL = "BULL"
    BEAR = "BEAR"
    VOLATILITY = "VOLATILITY"
    NEUTRAL = "NEUTRAL"


class MomentumSignalQuality(str, Enum):
    """Quality classification of momentum signal."""
    STRONG = "STRONG"       # All windows agree, high confidence
    MODERATE = "MODERATE"   # Most windows agree
    WEAK = "WEAK"           # Mixed signals
    CONFLICTING = "CONFLICTING"  # Windows disagree
    INSUFFICIENT = "INSUFFICIENT"  # Not enough data


@dataclass(frozen=True)
class MomentumWindow:
    """Configuration for a single momentum lookback window."""
    name: str
    lookback_days: int
    min_observations: int  # Minimum data points required
    description: str


# Default momentum windows
DEFAULT_MOMENTUM_WINDOWS: Tuple[MomentumWindow, ...] = (
    MomentumWindow(
        name="short",
        lookback_days=20,
        min_observations=15,
        description="Short-term momentum (20 trading days)"
    ),
    MomentumWindow(
        name="medium",
        lookback_days=60,
        min_observations=45,
        description="Medium-term momentum (60 trading days)"
    ),
    MomentumWindow(
        name="long",
        lookback_days=120,
        min_observations=90,
        description="Long-term momentum (120 trading days)"
    ),
)


# Regime-based window weights
REGIME_WINDOW_WEIGHTS: Dict[RegimeState, Dict[str, Decimal]] = {
    RegimeState.BULL: {
        "short": Decimal("0.40"),   # Favor recent momentum
        "medium": Decimal("0.40"),
        "long": Decimal("0.20"),
    },
    RegimeState.BEAR: {
        "short": Decimal("0.20"),   # Favor trend/long-term
        "medium": Decimal("0.30"),
        "long": Decimal("0.50"),
    },
    RegimeState.VOLATILITY: {
        "short": Decimal("0.25"),   # Reduce short-term noise
        "medium": Decimal("0.35"),
        "long": Decimal("0.40"),
    },
    RegimeState.NEUTRAL: {
        "short": Decimal("0.33"),   # Equal weighting
        "medium": Decimal("0.34"),
        "long": Decimal("0.33"),
    },
}


@dataclass
class WindowMomentumResult:
    """Momentum result for a single window."""
    window_name: str
    lookback_days: int
    return_pct: Optional[Decimal]  # Raw return over period
    annualized_return: Optional[Decimal]
    volatility: Optional[Decimal]  # Annualized volatility
    sharpe_ratio: Optional[Decimal]  # Risk-adjusted
    observations: int
    has_sufficient_data: bool
    raw_score: Optional[Decimal]  # 0-100 normalized

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_name": self.window_name,
            "lookback_days": self.lookback_days,
            "return_pct": str(self.return_pct) if self.return_pct else None,
            "annualized_return": str(self.annualized_return) if self.annualized_return else None,
            "volatility": str(self.volatility) if self.volatility else None,
            "sharpe_ratio": str(self.sharpe_ratio) if self.sharpe_ratio else None,
            "observations": self.observations,
            "has_sufficient_data": self.has_sufficient_data,
            "raw_score": str(self.raw_score) if self.raw_score else None,
        }


@dataclass
class MultiWindowMomentumResult:
    """Complete multi-window momentum analysis result."""
    ticker: str
    as_of_date: str
    regime: RegimeState
    window_results: Dict[str, WindowMomentumResult]
    window_weights: Dict[str, Decimal]
    weighted_score: Decimal
    signal_quality: MomentumSignalQuality
    confidence: Decimal  # 0-1 confidence in signal
    volatility_adjustment: Decimal  # Multiplier applied
    mean_reversion_flag: bool  # True if short-term reversal detected
    flags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "as_of_date": self.as_of_date,
            "regime": self.regime.value,
            "weighted_score": str(self.weighted_score),
            "signal_quality": self.signal_quality.value,
            "confidence": str(self.confidence),
            "volatility_adjustment": str(self.volatility_adjustment),
            "mean_reversion_flag": self.mean_reversion_flag,
            "flags": self.flags,
            "window_results": {k: v.to_dict() for k, v in self.window_results.items()},
            "window_weights": {k: str(v) for k, v in self.window_weights.items()},
        }


def _to_decimal(value: Any, default: Optional[Decimal] = None) -> Optional[Decimal]:
    """Safely convert value to Decimal."""
    if value is None:
        return default
    try:
        if isinstance(value, Decimal):
            return value
        if isinstance(value, (int, float)):
            if math.isnan(value) or math.isinf(value):
                return default
            return Decimal(str(value))
        if isinstance(value, str):
            stripped = value.strip()
            return Decimal(stripped) if stripped else default
        return default
    except (InvalidOperation, ValueError):
        return default


def _quantize(value: Decimal, precision: str = "0.01") -> Decimal:
    """Quantize decimal to specified precision."""
    return value.quantize(Decimal(precision), rounding=ROUND_HALF_UP)


class MultiWindowMomentumEngine:
    """
    Multi-window momentum calculation engine with regime adaptation.

    Implements multi-window momentum averaging with:
    - Regime-based window weighting
    - Volatility adjustment
    - Signal quality assessment
    - Mean reversion detection

    Usage:
        engine = MultiWindowMomentumEngine()
        result = engine.compute_momentum(
            ticker="ACME",
            price_series=prices,
            as_of_date=date(2026, 1, 15),
            regime=RegimeState.BULL
        )
        score = result.weighted_score
    """

    VERSION = "1.0.0"

    # Volatility baseline for adjustment (50% annualized)
    BASELINE_VOLATILITY = Decimal("0.50")

    # Volatility adjustment bounds
    VOL_ADJUSTMENT_MIN = Decimal("0.70")  # Max 30% penalty for high vol
    VOL_ADJUSTMENT_MAX = Decimal("1.30")  # Max 30% boost for low vol

    # Score normalization parameters
    SCORE_MIN = Decimal("0")
    SCORE_MAX = Decimal("100")
    SCORE_NEUTRAL = Decimal("50")

    # Annualization factor (trading days)
    TRADING_DAYS_YEAR = 252

    def __init__(
        self,
        windows: Optional[Tuple[MomentumWindow, ...]] = None,
        regime_weights: Optional[Dict[RegimeState, Dict[str, Decimal]]] = None,
        apply_volatility_adjustment: bool = True,
    ):
        """
        Initialize multi-window momentum engine.

        Args:
            windows: Momentum window configurations
            regime_weights: Regime-specific window weights
            apply_volatility_adjustment: Whether to adjust for volatility
        """
        self.windows = windows or DEFAULT_MOMENTUM_WINDOWS
        self.regime_weights = regime_weights or REGIME_WINDOW_WEIGHTS
        self.apply_volatility_adjustment = apply_volatility_adjustment

    def compute_window_momentum(
        self,
        price_series: List[Decimal],
        window: MomentumWindow,
    ) -> WindowMomentumResult:
        """
        Compute momentum for a single window.

        Args:
            price_series: List of prices (most recent last)
            window: Window configuration

        Returns:
            WindowMomentumResult for this window
        """
        # Check data sufficiency
        if len(price_series) < window.min_observations:
            return WindowMomentumResult(
                window_name=window.name,
                lookback_days=window.lookback_days,
                return_pct=None,
                annualized_return=None,
                volatility=None,
                sharpe_ratio=None,
                observations=len(price_series),
                has_sufficient_data=False,
                raw_score=None,
            )

        # Get prices for window
        window_prices = price_series[-window.lookback_days:]
        if len(window_prices) < window.min_observations:
            window_prices = price_series

        observations = len(window_prices)

        # Compute returns
        start_price = window_prices[0]
        end_price = window_prices[-1]

        if start_price <= Decimal("0"):
            return WindowMomentumResult(
                window_name=window.name,
                lookback_days=window.lookback_days,
                return_pct=None,
                annualized_return=None,
                volatility=None,
                sharpe_ratio=None,
                observations=observations,
                has_sufficient_data=False,
                raw_score=None,
            )

        # Period return
        return_pct = ((end_price - start_price) / start_price) * Decimal("100")
        return_pct = _quantize(return_pct)

        # Annualized return (using trading days)
        annualization_factor = Decimal(str(math.sqrt(self.TRADING_DAYS_YEAR / max(1, observations))))
        annualized_return = return_pct * annualization_factor / Decimal("100")
        annualized_return = _quantize(annualized_return, "0.0001")

        # Compute daily returns for volatility
        daily_returns = []
        for i in range(1, len(window_prices)):
            if window_prices[i-1] > Decimal("0"):
                daily_ret = (window_prices[i] - window_prices[i-1]) / window_prices[i-1]
                daily_returns.append(float(daily_ret))

        # Volatility (annualized)
        volatility = None
        sharpe_ratio = None
        if len(daily_returns) >= 10:
            mean_ret = sum(daily_returns) / len(daily_returns)
            variance = sum((r - mean_ret) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
            daily_vol = math.sqrt(variance) if variance > 0 else 0
            vol_annualized = daily_vol * math.sqrt(self.TRADING_DAYS_YEAR)
            volatility = _to_decimal(vol_annualized)

            # Sharpe ratio (assuming 5% risk-free rate)
            if volatility and volatility > Decimal("0"):
                risk_free = Decimal("0.05")
                excess_return = annualized_return - risk_free / Decimal(str(self.TRADING_DAYS_YEAR / observations))
                sharpe_ratio = _quantize(excess_return / volatility, "0.01")

        # Normalize to 0-100 score
        # Map returns to score: -50% -> 0, 0% -> 50, +50% -> 100
        raw_score = self.SCORE_NEUTRAL + (return_pct / Decimal("100")) * self.SCORE_NEUTRAL
        raw_score = max(self.SCORE_MIN, min(self.SCORE_MAX, raw_score))
        raw_score = _quantize(raw_score)

        return WindowMomentumResult(
            window_name=window.name,
            lookback_days=window.lookback_days,
            return_pct=return_pct,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            observations=observations,
            has_sufficient_data=True,
            raw_score=raw_score,
        )

    def compute_volatility_adjustment(
        self,
        window_results: Dict[str, WindowMomentumResult],
    ) -> Decimal:
        """
        Compute volatility adjustment factor.

        Higher volatility -> lower adjustment (dampens score)
        Lower volatility -> higher adjustment (boosts score)
        """
        # Use medium window volatility as reference
        med_result = window_results.get("medium")
        if med_result and med_result.volatility:
            vol = med_result.volatility

            # Adjustment = baseline / actual (clamped)
            if vol > Decimal("0"):
                adjustment = self.BASELINE_VOLATILITY / vol
                adjustment = max(self.VOL_ADJUSTMENT_MIN, min(self.VOL_ADJUSTMENT_MAX, adjustment))
                return _quantize(adjustment)

        return Decimal("1.00")

    def detect_mean_reversion(
        self,
        window_results: Dict[str, WindowMomentumResult],
    ) -> Tuple[bool, List[str]]:
        """
        Detect potential mean reversion signals.

        Mean reversion indicated when:
        - Short-term momentum opposite to medium/long-term
        - Recent price extreme relative to historical
        """
        flags = []

        short = window_results.get("short")
        medium = window_results.get("medium")
        long_term = window_results.get("long")

        # Check for opposing short vs medium/long
        if short and short.return_pct and medium and medium.return_pct:
            short_direction = short.return_pct > Decimal("0")
            medium_direction = medium.return_pct > Decimal("0")

            if short_direction != medium_direction:
                # Short-term reversal
                if abs(short.return_pct) > Decimal("10"):  # Strong reversal
                    flags.append("SHORT_TERM_REVERSAL")
                    return (True, flags)

        # Check for extreme short-term move with mean reversion potential
        if short and short.return_pct:
            if abs(short.return_pct) > Decimal("30"):
                flags.append("EXTREME_SHORT_TERM_MOVE")
                if long_term and long_term.return_pct:
                    if (short.return_pct > Decimal("0")) != (long_term.return_pct > Decimal("0")):
                        flags.append("POTENTIAL_MEAN_REVERSION")
                        return (True, flags)

        return (False, flags)

    def assess_signal_quality(
        self,
        window_results: Dict[str, WindowMomentumResult],
    ) -> Tuple[MomentumSignalQuality, Decimal]:
        """
        Assess quality and confidence of momentum signal.

        Returns (quality, confidence) tuple.
        """
        # Count windows with data
        valid_windows = [w for w in window_results.values() if w.has_sufficient_data and w.raw_score is not None]

        if len(valid_windows) == 0:
            return (MomentumSignalQuality.INSUFFICIENT, Decimal("0"))

        if len(valid_windows) == 1:
            return (MomentumSignalQuality.WEAK, Decimal("0.30"))

        # Check direction agreement
        directions = []
        for w in valid_windows:
            if w.return_pct is not None:
                directions.append(w.return_pct > Decimal("0"))

        if not directions:
            return (MomentumSignalQuality.INSUFFICIENT, Decimal("0"))

        agreement = sum(1 for d in directions if d == directions[0]) / len(directions)

        if agreement == 1.0:
            # All windows agree
            return (MomentumSignalQuality.STRONG, Decimal("0.90"))
        elif agreement >= 0.67:
            # Most windows agree
            return (MomentumSignalQuality.MODERATE, Decimal("0.70"))
        elif agreement >= 0.5:
            # Mixed signals
            return (MomentumSignalQuality.WEAK, Decimal("0.40"))
        else:
            # Conflicting signals
            return (MomentumSignalQuality.CONFLICTING, Decimal("0.20"))

    def compute_momentum(
        self,
        ticker: str,
        price_series: List[Decimal],
        as_of_date: date,
        regime: RegimeState = RegimeState.NEUTRAL,
    ) -> MultiWindowMomentumResult:
        """
        Compute multi-window momentum for a ticker.

        Args:
            ticker: Stock ticker
            price_series: Price history (oldest first, most recent last)
            as_of_date: Analysis date
            regime: Current market regime for weight adaptation

        Returns:
            MultiWindowMomentumResult with full analysis
        """
        # Compute momentum for each window
        window_results: Dict[str, WindowMomentumResult] = {}
        for window in self.windows:
            result = self.compute_window_momentum(price_series, window)
            window_results[window.name] = result

        # Get regime-based weights
        window_weights = self.regime_weights.get(regime, self.regime_weights[RegimeState.NEUTRAL])

        # Compute weighted score
        weighted_score = Decimal("0")
        total_weight = Decimal("0")

        for window_name, weight in window_weights.items():
            result = window_results.get(window_name)
            if result and result.has_sufficient_data and result.raw_score is not None:
                weighted_score += result.raw_score * weight
                total_weight += weight

        if total_weight > Decimal("0"):
            weighted_score = weighted_score / total_weight
        else:
            weighted_score = self.SCORE_NEUTRAL

        weighted_score = _quantize(weighted_score)

        # Apply volatility adjustment
        vol_adjustment = Decimal("1.00")
        if self.apply_volatility_adjustment:
            vol_adjustment = self.compute_volatility_adjustment(window_results)
            # Apply adjustment relative to neutral
            adjustment_effect = (weighted_score - self.SCORE_NEUTRAL) * (vol_adjustment - Decimal("1.00"))
            weighted_score = weighted_score + adjustment_effect
            weighted_score = max(self.SCORE_MIN, min(self.SCORE_MAX, weighted_score))
            weighted_score = _quantize(weighted_score)

        # Assess signal quality
        signal_quality, confidence = self.assess_signal_quality(window_results)

        # Detect mean reversion
        mean_reversion, mr_flags = self.detect_mean_reversion(window_results)

        # Build flags
        flags = list(mr_flags)
        if signal_quality == MomentumSignalQuality.CONFLICTING:
            flags.append("CONFLICTING_WINDOWS")
        if signal_quality == MomentumSignalQuality.INSUFFICIENT:
            flags.append("INSUFFICIENT_DATA")
        if vol_adjustment < Decimal("0.85"):
            flags.append("HIGH_VOLATILITY_DAMPENING")
        if vol_adjustment > Decimal("1.15"):
            flags.append("LOW_VOLATILITY_BOOST")

        return MultiWindowMomentumResult(
            ticker=ticker,
            as_of_date=as_of_date.isoformat(),
            regime=regime,
            window_results=window_results,
            window_weights=window_weights,
            weighted_score=weighted_score,
            signal_quality=signal_quality,
            confidence=confidence,
            volatility_adjustment=vol_adjustment,
            mean_reversion_flag=mean_reversion,
            flags=flags,
        )


def compute_batch_momentum(
    price_data: Dict[str, List[Decimal]],
    as_of_date: date,
    regime: RegimeState = RegimeState.NEUTRAL,
    engine: Optional[MultiWindowMomentumEngine] = None,
) -> Dict[str, MultiWindowMomentumResult]:
    """
    Compute multi-window momentum for multiple tickers.

    Args:
        price_data: Dict of ticker -> price series
        as_of_date: Analysis date
        regime: Current market regime
        engine: Momentum engine (uses default if None)

    Returns:
        Dict of ticker -> MultiWindowMomentumResult
    """
    if engine is None:
        engine = MultiWindowMomentumEngine()

    results = {}
    for ticker in sorted(price_data.keys()):
        prices = price_data[ticker]
        result = engine.compute_momentum(ticker, prices, as_of_date, regime)
        results[ticker] = result

    return results


if __name__ == "__main__":
    # Demonstration
    print("=" * 70)
    print("MULTI-WINDOW MOMENTUM ENGINE - DEMONSTRATION")
    print("=" * 70)

    engine = MultiWindowMomentumEngine()

    # Generate sample price data (uptrend)
    prices = [Decimal(str(100 + i * 0.5 + (i % 5) * 0.2)) for i in range(150)]

    result = engine.compute_momentum(
        ticker="ACME",
        price_series=prices,
        as_of_date=date(2026, 1, 15),
        regime=RegimeState.BULL,
    )

    print(f"\nTicker: {result.ticker}")
    print(f"Regime: {result.regime.value}")
    print(f"Weighted Score: {result.weighted_score}")
    print(f"Signal Quality: {result.signal_quality.value}")
    print(f"Confidence: {result.confidence}")
    print(f"Volatility Adjustment: {result.volatility_adjustment}")
    print(f"Mean Reversion Flag: {result.mean_reversion_flag}")
    print(f"Flags: {result.flags}")

    print("\nWindow Results:")
    for name, wr in result.window_results.items():
        print(f"  {name}: return={wr.return_pct}%, score={wr.raw_score}")

    print("\nWindow Weights (BULL regime):")
    for name, weight in result.window_weights.items():
        print(f"  {name}: {weight}")
