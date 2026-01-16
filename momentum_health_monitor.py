#!/usr/bin/env python3
"""
momentum_health_monitor.py - Price Momentum Signal Health Monitoring

Monitors the effectiveness of price momentum signals using rolling Information
Coefficient (IC) and provides adaptive weight recommendations.

Key Features:
1. Rolling IC calculation (3-month window)
2. Automatic momentum weight adjustment based on IC health
3. Kill switch when momentum signal shows persistent negative IC
4. Integration with regime_engine.py for regime-aware adjustments

Based on validation findings:
- Momentum works well in BULL regime (IC > 0.15 typical)
- Momentum may invert in BEAR regime (IC < 0)
- Current NEUTRAL regime shows IC ~ -0.10

Weight Recommendations:
- IC >= 0.10: Full regime-adaptive weight (momentum is working)
- IC >= 0.05: Reduced weight (0.10) - marginal signal
- IC >= 0.00: Minimal weight (0.05) - weak signal
- IC < 0.00: Disable momentum (0.00) - inverted signal

Design Philosophy:
- Deterministic: Same inputs produce same outputs
- Stdlib-only: No external dependencies beyond scipy for correlation
- Decimal arithmetic for precision
- Full audit trail

Author: Wake Robin Capital Management
Version: 1.0.0
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any
from datetime import date, timedelta
import math

__version__ = "1.0.0"
SCHEMA_VERSION = "v1.0"

# Quantization precision
Q = Decimal("0.0001")


def _dq(x: float) -> Decimal:
    """Quantize float to 4 decimal places."""
    return Decimal(str(x)).quantize(Q, rounding=ROUND_HALF_UP)


# ============================================================================
# IC CALCULATION
# ============================================================================

def spearman_rank_correlation(x: List[float], y: List[float]) -> Optional[float]:
    """
    Calculate Spearman rank correlation coefficient.

    Used for Information Coefficient (IC) calculation between momentum
    signals and forward returns.

    Args:
        x: First series (e.g., momentum ranks)
        y: Second series (e.g., forward returns)

    Returns:
        Correlation coefficient [-1, 1], or None if insufficient data
    """
    if len(x) != len(y) or len(x) < 3:
        return None

    n = len(x)

    # Convert to ranks
    def rank(values: List[float]) -> List[float]:
        sorted_pairs = sorted(enumerate(values), key=lambda p: p[1])
        ranks = [0.0] * n
        for rank_val, (orig_idx, _) in enumerate(sorted_pairs, 1):
            ranks[orig_idx] = float(rank_val)
        return ranks

    rx = rank(x)
    ry = rank(y)

    # Pearson correlation on ranks
    mx = sum(rx) / n
    my = sum(ry) / n

    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n)) / (n - 1)
    sx = math.sqrt(sum((v - mx) ** 2 for v in rx) / (n - 1))
    sy = math.sqrt(sum((v - my) ** 2 for v in ry) / (n - 1))

    if sx == 0 or sy == 0:
        return None

    return cov / (sx * sy)


def calculate_momentum_signal(prices: List[float], lookback: int = 63) -> Optional[float]:
    """
    Calculate momentum signal from price series.

    Uses total return over lookback period (typically 63 days = 3 months).

    Args:
        prices: Daily price series
        lookback: Momentum lookback period in trading days

    Returns:
        Momentum return as decimal (e.g., 0.15 = 15%), or None
    """
    if prices is None or len(prices) < lookback + 1:
        return None

    past_price = prices[-(lookback + 1)]
    current_price = prices[-1]

    if past_price <= 0:
        return None

    return (current_price / past_price) - 1.0


def calculate_forward_return(prices: List[float], horizon: int = 21) -> Optional[float]:
    """
    Calculate forward return for IC measurement.

    Args:
        prices: Full price series (including future data for backtesting)
        horizon: Forward return horizon in trading days

    Returns:
        Forward return as decimal, or None
    """
    # This is for backtesting - we need prices that extend into the future
    if prices is None or len(prices) < horizon + 1:
        return None

    current_price = prices[-(horizon + 1)]
    future_price = prices[-1]

    if current_price <= 0:
        return None

    return (future_price / current_price) - 1.0


def calculate_cross_sectional_ic(
    prices_by_ticker: Dict[str, List[float]],
    momentum_lookback: int = 63,
    forward_horizon: int = 21,
) -> Optional[float]:
    """
    Calculate cross-sectional Information Coefficient.

    IC measures the Spearman rank correlation between momentum signals
    across tickers and their subsequent forward returns.

    High IC (>0.1): Momentum signal is predictive
    Low IC (~0): No predictive power
    Negative IC (<0): Momentum is inverting (mean reversion)

    Args:
        prices_by_ticker: Dict mapping ticker to price series
        momentum_lookback: Lookback for momentum calculation
        forward_horizon: Forward return horizon

    Returns:
        IC value, or None if insufficient data
    """
    momentum_signals = []
    forward_returns = []

    for ticker, prices in prices_by_ticker.items():
        if ticker.startswith('_'):  # Skip internal tickers
            continue

        # Need enough data for both lookback and forward horizon
        min_required = momentum_lookback + forward_horizon + 2
        if len(prices) < min_required:
            continue

        # Split prices: history for momentum, future for forward returns
        history_end = len(prices) - forward_horizon
        history_prices = prices[:history_end]
        future_prices = prices[history_end - 1:]  # Include overlap point

        # Calculate momentum at history_end
        mom = calculate_momentum_signal(history_prices, momentum_lookback)
        if mom is None:
            continue

        # Calculate forward return from history_end
        if len(future_prices) < 2:
            continue
        fwd = (future_prices[-1] / future_prices[0]) - 1.0

        momentum_signals.append(mom)
        forward_returns.append(fwd)

    if len(momentum_signals) < 10:  # Need minimum sample size
        return None

    return spearman_rank_correlation(momentum_signals, forward_returns)


def calculate_rolling_ic(
    prices_by_ticker: Dict[str, List[float]],
    window_days: int = 63,  # 3 months
    momentum_lookback: int = 63,
    forward_horizon: int = 21,
) -> List[Tuple[int, float]]:
    """
    Calculate rolling IC over time.

    Returns IC at multiple points to understand signal stability.

    Args:
        prices_by_ticker: Dict mapping ticker to price series
        window_days: Rolling window size
        momentum_lookback: Lookback for momentum calculation
        forward_horizon: Forward return horizon

    Returns:
        List of (day_offset, ic_value) tuples
    """
    results = []

    # Find minimum series length
    min_len = min(len(p) for p in prices_by_ticker.values() if len(p) > 0)

    # Need enough data for rolling calculation
    min_required = momentum_lookback + forward_horizon + window_days
    if min_len < min_required:
        return results

    # Calculate IC at each point
    for offset in range(0, min_len - min_required, forward_horizon):
        # Slice prices to this window
        window_prices = {
            ticker: prices[offset:offset + min_required]
            for ticker, prices in prices_by_ticker.items()
        }

        ic = calculate_cross_sectional_ic(
            window_prices,
            momentum_lookback,
            forward_horizon
        )

        if ic is not None:
            results.append((offset, ic))

    return results


# ============================================================================
# MOMENTUM HEALTH ASSESSMENT
# ============================================================================

# IC thresholds for weight adjustment
IC_EXCELLENT = Decimal("0.15")   # Strong predictive power
IC_GOOD = Decimal("0.10")        # Good predictive power
IC_MARGINAL = Decimal("0.05")    # Marginal signal
IC_WEAK = Decimal("0.00")        # No predictive power
IC_INVERTED = Decimal("-0.05")   # Mean reversion regime

# Weight levels
WEIGHT_FULL = Decimal("1.00")     # Full regime-adaptive weight
WEIGHT_REDUCED = Decimal("0.10")  # Reduced weight for marginal signal
WEIGHT_MINIMAL = Decimal("0.05")  # Minimal weight for weak signal
WEIGHT_DISABLED = Decimal("0.00") # Momentum disabled


class MomentumHealthMonitor:
    """
    Monitor momentum signal health and provide adaptive weights.

    Uses rolling IC to determine if momentum signal is working and
    adjusts weights accordingly.

    Usage:
        monitor = MomentumHealthMonitor()
        monitor.update_ic(0.12)  # From validation
        weight = monitor.get_momentum_weight("BULL")
    """

    VERSION = "1.0.0"

    # Regime base weights (from regime_engine.py)
    REGIME_BASE_WEIGHTS = {
        "BULL": Decimal("1.20"),
        "BEAR": Decimal("0.80"),
        "VOLATILITY_SPIKE": Decimal("0.70"),
        "SECTOR_ROTATION": Decimal("1.00"),
        "NEUTRAL": Decimal("1.00"),
        "UNKNOWN": Decimal("1.00"),
    }

    def __init__(self, ic_history_length: int = 10):
        """
        Initialize momentum health monitor.

        Args:
            ic_history_length: Number of IC observations to track
        """
        self.ic_history: List[Decimal] = []
        self.ic_history_length = ic_history_length
        self.current_ic: Optional[Decimal] = None
        self.health_status: str = "UNKNOWN"
        self.audit_trail: List[Dict[str, Any]] = []

    def update_ic(self, ic_value: float, as_of_date: Optional[date] = None) -> Dict[str, Any]:
        """
        Update with new IC observation.

        Args:
            ic_value: New IC measurement
            as_of_date: Date of measurement

        Returns:
            Health assessment dict
        """
        ic_decimal = _dq(ic_value)
        self.current_ic = ic_decimal

        # Update history
        self.ic_history.append(ic_decimal)
        if len(self.ic_history) > self.ic_history_length:
            self.ic_history = self.ic_history[-self.ic_history_length:]

        # Assess health
        self.health_status = self._assess_health(ic_decimal)

        # Audit entry
        audit = {
            "as_of_date": as_of_date.isoformat() if as_of_date else None,
            "ic_value": str(ic_decimal),
            "health_status": self.health_status,
            "ic_3m_avg": str(self.get_rolling_ic_average()),
            "module_version": self.VERSION,
        }
        self.audit_trail.append(audit)

        return {
            "ic": ic_decimal,
            "health_status": self.health_status,
            "ic_3m_avg": self.get_rolling_ic_average(),
            "recommendation": self._get_recommendation(),
        }

    def _assess_health(self, ic: Decimal) -> str:
        """Classify momentum health based on IC."""
        if ic >= IC_EXCELLENT:
            return "EXCELLENT"
        elif ic >= IC_GOOD:
            return "GOOD"
        elif ic >= IC_MARGINAL:
            return "MARGINAL"
        elif ic >= IC_WEAK:
            return "WEAK"
        elif ic >= IC_INVERTED:
            return "INVERTED"
        else:
            return "STRONGLY_INVERTED"

    def _get_recommendation(self) -> str:
        """Get action recommendation based on health."""
        if self.health_status in ("EXCELLENT", "GOOD"):
            return "Use full regime-adaptive momentum weight"
        elif self.health_status == "MARGINAL":
            return "Reduce momentum weight to 10%"
        elif self.health_status == "WEAK":
            return "Reduce momentum weight to 5%"
        else:
            return "DISABLE momentum signal - consider mean reversion"

    def get_rolling_ic_average(self) -> Decimal:
        """Get rolling average IC."""
        if not self.ic_history:
            return Decimal("0")
        return sum(self.ic_history) / len(self.ic_history)

    def get_momentum_weight(self, current_regime: str) -> Decimal:
        """
        Get adaptive momentum weight based on IC health and regime.

        This is the main integration point with the screening pipeline.

        Args:
            current_regime: Current market regime (BULL, BEAR, etc.)

        Returns:
            Momentum weight multiplier
        """
        if self.current_ic is None:
            # No IC data - use regime default
            return self.REGIME_BASE_WEIGHTS.get(current_regime, Decimal("1.00"))

        # Get rolling average for stability
        ic_3m = self.get_rolling_ic_average()

        # Apply IC-based adjustment
        if ic_3m >= IC_GOOD:
            # Momentum is working - use full regime weight
            return self.REGIME_BASE_WEIGHTS.get(current_regime, Decimal("1.00"))
        elif ic_3m >= IC_MARGINAL:
            # Marginal signal - reduce weight
            return WEIGHT_REDUCED
        elif ic_3m >= IC_WEAK:
            # Weak signal - minimal weight
            return WEIGHT_MINIMAL
        else:
            # Inverted signal - disable momentum
            return WEIGHT_DISABLED

    def check_momentum_health(self, ic_3m_rolling: float) -> Decimal:
        """
        Check momentum health and return adaptive weight.

        This implements the user's suggested kill switch logic:
        - If IC < 0.00: disable momentum (inverted)
        - If IC < 0.05: minimal weight (weak)
        - If IC < 0.10: reduced weight (marginal)
        - If IC >= 0.10: use full regime-adaptive weight

        Args:
            ic_3m_rolling: 3-month rolling IC value

        Returns:
            Adaptive momentum weight
        """
        ic = _dq(ic_3m_rolling)

        if ic < IC_WEAK:
            # IC < 0: Disable momentum (signal is inverted)
            return WEIGHT_DISABLED
        elif ic < IC_MARGINAL:
            # IC < 0.05: Minimal weight (very weak signal)
            return WEIGHT_MINIMAL
        elif ic < IC_GOOD:
            # IC < 0.10: Reduced weight (marginal signal)
            return WEIGHT_REDUCED
        else:
            # IC >= 0.10: Use full regime-adaptive weight
            # Caller should apply regime adjustment
            return WEIGHT_FULL

    def get_status_summary(self) -> Dict[str, Any]:
        """Get current momentum health status summary."""
        return {
            "current_ic": str(self.current_ic) if self.current_ic else None,
            "rolling_ic_3m": str(self.get_rolling_ic_average()),
            "health_status": self.health_status,
            "ic_history_count": len(self.ic_history),
            "recommendation": self._get_recommendation(),
            "version": self.VERSION,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "current_ic": str(self.current_ic) if self.current_ic else None,
            "ic_history": [str(ic) for ic in self.ic_history],
            "health_status": self.health_status,
            "schema_version": SCHEMA_VERSION,
            "module_version": self.VERSION,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MomentumHealthMonitor":
        """Deserialize from dictionary."""
        monitor = cls()
        if data.get("current_ic"):
            monitor.current_ic = Decimal(data["current_ic"])
        if data.get("ic_history"):
            monitor.ic_history = [Decimal(ic) for ic in data["ic_history"]]
        if data.get("health_status"):
            monitor.health_status = data["health_status"]
        return monitor


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_regime_adaptive_momentum_weight(
    ic_3m_rolling: float,
    current_regime: str,
) -> Decimal:
    """
    Get regime-adaptive momentum weight with IC kill switch.

    Main integration function for screening pipeline.

    Args:
        ic_3m_rolling: 3-month rolling IC value
        current_regime: Current market regime

    Returns:
        Final momentum weight multiplier
    """
    monitor = MomentumHealthMonitor()
    base_health_weight = monitor.check_momentum_health(ic_3m_rolling)

    if base_health_weight == WEIGHT_DISABLED:
        return WEIGHT_DISABLED
    elif base_health_weight == WEIGHT_MINIMAL:
        return WEIGHT_MINIMAL
    elif base_health_weight == WEIGHT_REDUCED:
        return WEIGHT_REDUCED
    else:
        # Full weight - apply regime adjustment
        regime_weight = monitor.REGIME_BASE_WEIGHTS.get(current_regime, Decimal("1.00"))
        return regime_weight


def assess_momentum_signal_health(
    prices_by_ticker: Dict[str, List[float]],
    current_regime: str = "NEUTRAL",
) -> Dict[str, Any]:
    """
    Full momentum signal health assessment.

    Calculates IC and provides weight recommendation.

    Args:
        prices_by_ticker: Historical prices by ticker
        current_regime: Current market regime

    Returns:
        Complete health assessment
    """
    # Calculate current IC
    ic = calculate_cross_sectional_ic(prices_by_ticker)

    if ic is None:
        return {
            "ic": None,
            "health_status": "UNKNOWN",
            "recommended_weight": Decimal("1.00"),
            "message": "Insufficient data to calculate IC",
        }

    # Get adaptive weight
    weight = get_regime_adaptive_momentum_weight(ic, current_regime)

    # Determine health status
    ic_decimal = _dq(ic)
    if ic_decimal >= IC_GOOD:
        status = "HEALTHY"
    elif ic_decimal >= IC_MARGINAL:
        status = "MARGINAL"
    elif ic_decimal >= IC_WEAK:
        status = "WEAK"
    else:
        status = "INVERTED"

    return {
        "ic": ic_decimal,
        "health_status": status,
        "recommended_weight": weight,
        "current_regime": current_regime,
        "message": _get_health_message(status, ic_decimal, weight),
    }


def _get_health_message(status: str, ic: Decimal, weight: Decimal) -> str:
    """Generate human-readable health message."""
    if status == "HEALTHY":
        return f"Momentum signal is working (IC={ic}). Using regime-adjusted weight {weight}."
    elif status == "MARGINAL":
        return f"Momentum signal is marginal (IC={ic}). Reducing weight to {weight}."
    elif status == "WEAK":
        return f"Momentum signal is weak (IC={ic}). Using minimal weight {weight}."
    else:
        return f"Momentum signal is INVERTED (IC={ic}). DISABLING momentum (weight=0)."


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MOMENTUM HEALTH MONITOR - DEMONSTRATION")
    print("=" * 70)
    print()

    monitor = MomentumHealthMonitor()

    # Simulate IC observations over time
    test_ics = [
        (0.15, "BULL market - strong momentum"),
        (0.12, "Still BULL - momentum working"),
        (0.08, "Transition - momentum weakening"),
        (0.03, "NEUTRAL - marginal signal"),
        (-0.02, "Rotation - momentum inverting"),
        (-0.10, "BEAR market - mean reversion"),
    ]

    print("Simulating IC observations over time:")
    print("-" * 70)

    for ic, description in test_ics:
        result = monitor.update_ic(ic)
        weight_bull = monitor.get_momentum_weight("BULL")
        weight_bear = monitor.get_momentum_weight("BEAR")

        print(f"\nIC = {ic:+.2f} ({description})")
        print(f"  Health Status: {result['health_status']}")
        print(f"  Rolling IC (3m): {result['ic_3m_avg']}")
        print(f"  Weight (BULL regime): {weight_bull}")
        print(f"  Weight (BEAR regime): {weight_bear}")
        print(f"  Recommendation: {result['recommendation']}")

    print()
    print("=" * 70)
    print("FINAL STATUS")
    print("=" * 70)
    print()

    status = monitor.get_status_summary()
    for key, value in status.items():
        print(f"  {key}: {value}")

    print()
    print("Kill switch function test:")
    print("-" * 70)

    for ic_val in [0.20, 0.08, 0.03, -0.05, -0.15]:
        weight = monitor.check_momentum_health(ic_val)
        print(f"  IC = {ic_val:+.2f} -> Weight = {weight}")
