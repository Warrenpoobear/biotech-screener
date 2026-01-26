#!/usr/bin/env python3
"""
regime_hysteresis.py - Regime Hysteresis for Stable Regime Classification

P2 Enhancement: Adds hysteresis to regime detection to prevent unstable
regime switching around threshold boundaries.

Key Features:
1. Hysteresis buffers for VIX thresholds
2. Regime persistence tracking
3. Transition risk assessment
4. Regime stability scoring

Design Philosophy:
- DETERMINISTIC: Explicit state machine with clear transitions
- AUDITABLE: Full history of regime changes
- STABLE: Requires threshold + buffer to switch regimes

Author: Wake Robin Capital Management
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import logging

__version__ = "1.0.0"

logger = logging.getLogger(__name__)


class RegimeState(str, Enum):
    """Market regime states."""
    BULL = "BULL"
    BEAR = "BEAR"
    VOLATILITY_SPIKE = "VOLATILITY_SPIKE"
    SECTOR_ROTATION = "SECTOR_ROTATION"
    UNKNOWN = "UNKNOWN"


class TransitionType(str, Enum):
    """Type of regime transition."""
    NONE = "NONE"           # No transition
    CONFIRMED = "CONFIRMED" # Transition confirmed (crossed buffer)
    PENDING = "PENDING"     # Threshold crossed but within buffer
    REJECTED = "REJECTED"   # Returned to original regime


@dataclass
class RegimeTransition:
    """Record of a regime transition."""
    from_regime: RegimeState
    to_regime: RegimeState
    transition_date: str
    transition_type: TransitionType
    days_in_prior_regime: int
    trigger_values: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_regime": self.from_regime.value,
            "to_regime": self.to_regime.value,
            "transition_date": self.transition_date,
            "transition_type": self.transition_type.value,
            "days_in_prior_regime": self.days_in_prior_regime,
            "trigger_values": self.trigger_values,
        }


@dataclass
class HysteresisState:
    """Current hysteresis state for regime tracking."""
    current_regime: RegimeState
    regime_since: str  # ISO date
    days_in_regime: int
    pending_transition: Optional[RegimeState]
    pending_since: Optional[str]
    stability_score: Decimal  # 0-1, higher = more stable
    transition_history: List[RegimeTransition] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_regime": self.current_regime.value,
            "regime_since": self.regime_since,
            "days_in_regime": self.days_in_regime,
            "pending_transition": self.pending_transition.value if self.pending_transition else None,
            "pending_since": self.pending_since,
            "stability_score": str(self.stability_score),
            "transition_count_30d": sum(1 for t in self.transition_history
                                        if t.transition_type == TransitionType.CONFIRMED),
        }


class RegimeHysteresisEngine:
    """
    Regime detection with hysteresis for stability.

    Prevents noisy regime switching by requiring thresholds to be
    crossed by a buffer amount before confirming a transition.

    Usage:
        engine = RegimeHysteresisEngine()

        # First call establishes baseline
        result = engine.update(
            vix=Decimal("22.0"),
            xbi_momentum=Decimal("0.02"),
            as_of_date=date(2026, 1, 15)
        )

        # Subsequent calls track transitions with hysteresis
        result = engine.update(
            vix=Decimal("31.0"),  # Crosses VIX=30 threshold
            xbi_momentum=Decimal("-0.05"),
            as_of_date=date(2026, 1, 16)
        )
        # Transition may be PENDING until buffer is also crossed
    """

    VERSION = "1.0.0"

    # Base thresholds (same as RegimeDetectionEngine)
    VIX_BULL_THRESHOLD = Decimal("20")
    VIX_BEAR_THRESHOLD = Decimal("25")
    VIX_SPIKE_THRESHOLD = Decimal("30")

    # Hysteresis buffers (require this much beyond threshold to switch)
    VIX_HYSTERESIS_BUFFER = Decimal("2.0")

    # Momentum thresholds
    XBI_BULL_THRESHOLD = Decimal("0.02")   # 2% outperformance
    XBI_BEAR_THRESHOLD = Decimal("-0.02")  # 2% underperformance
    XBI_HYSTERESIS_BUFFER = Decimal("0.01")

    # Minimum days in regime before transition allowed
    MIN_DAYS_BEFORE_TRANSITION = 3

    # Days threshold must be crossed before confirming
    CONFIRMATION_DAYS = 2

    def __init__(
        self,
        vix_hysteresis: Optional[Decimal] = None,
        xbi_hysteresis: Optional[Decimal] = None,
        min_days_before_transition: int = 3,
        confirmation_days: int = 2,
    ):
        """
        Initialize hysteresis engine.

        Args:
            vix_hysteresis: Buffer for VIX thresholds
            xbi_hysteresis: Buffer for XBI momentum thresholds
            min_days_before_transition: Minimum days in regime before switch allowed
            confirmation_days: Days threshold must be crossed to confirm
        """
        self.vix_hysteresis = vix_hysteresis or self.VIX_HYSTERESIS_BUFFER
        self.xbi_hysteresis = xbi_hysteresis or self.XBI_HYSTERESIS_BUFFER
        self.min_days_before_transition = min_days_before_transition
        self.confirmation_days = confirmation_days

        # State
        self._state: Optional[HysteresisState] = None
        self._pending_count: int = 0

    def _get_effective_thresholds(
        self,
        current_regime: RegimeState,
    ) -> Dict[str, Dict[str, Decimal]]:
        """
        Get effective thresholds with hysteresis applied.

        Thresholds differ based on current regime to create hysteresis.
        """
        # If currently in BULL, require higher VIX to switch to BEAR/SPIKE
        # If currently in BEAR, require lower VIX to switch to BULL

        if current_regime == RegimeState.BULL:
            return {
                "vix_bear": self.VIX_BEAR_THRESHOLD + self.vix_hysteresis,
                "vix_spike": self.VIX_SPIKE_THRESHOLD + self.vix_hysteresis,
                "vix_bull": self.VIX_BULL_THRESHOLD,  # No buffer to stay
                "xbi_bear": self.XBI_BEAR_THRESHOLD - self.xbi_hysteresis,
                "xbi_bull": self.XBI_BULL_THRESHOLD,
            }
        elif current_regime == RegimeState.BEAR:
            return {
                "vix_bear": self.VIX_BEAR_THRESHOLD,  # No buffer to stay
                "vix_spike": self.VIX_SPIKE_THRESHOLD + self.vix_hysteresis,
                "vix_bull": self.VIX_BULL_THRESHOLD - self.vix_hysteresis,
                "xbi_bear": self.XBI_BEAR_THRESHOLD,
                "xbi_bull": self.XBI_BULL_THRESHOLD + self.xbi_hysteresis,
            }
        elif current_regime == RegimeState.VOLATILITY_SPIKE:
            return {
                "vix_bear": self.VIX_BEAR_THRESHOLD,
                "vix_spike": self.VIX_SPIKE_THRESHOLD - self.vix_hysteresis,  # Require drop to exit
                "vix_bull": self.VIX_BULL_THRESHOLD - self.vix_hysteresis,
                "xbi_bear": self.XBI_BEAR_THRESHOLD,
                "xbi_bull": self.XBI_BULL_THRESHOLD,
            }
        else:  # SECTOR_ROTATION or UNKNOWN
            return {
                "vix_bear": self.VIX_BEAR_THRESHOLD,
                "vix_spike": self.VIX_SPIKE_THRESHOLD,
                "vix_bull": self.VIX_BULL_THRESHOLD,
                "xbi_bear": self.XBI_BEAR_THRESHOLD,
                "xbi_bull": self.XBI_BULL_THRESHOLD,
            }

    def _classify_regime_raw(
        self,
        vix: Decimal,
        xbi_momentum: Decimal,
        thresholds: Dict[str, Decimal],
    ) -> RegimeState:
        """Classify regime based on inputs and thresholds."""
        # Volatility spike takes priority
        if vix >= thresholds["vix_spike"]:
            return RegimeState.VOLATILITY_SPIKE

        # BEAR conditions
        if vix >= thresholds["vix_bear"] or xbi_momentum <= thresholds["xbi_bear"]:
            return RegimeState.BEAR

        # BULL conditions
        if vix <= thresholds["vix_bull"] and xbi_momentum >= thresholds["xbi_bull"]:
            return RegimeState.BULL

        # Default to rotation
        return RegimeState.SECTOR_ROTATION

    def _compute_stability_score(
        self,
        days_in_regime: int,
        transitions_30d: int,
    ) -> Decimal:
        """
        Compute regime stability score (0-1).

        Higher score = more stable (longer in regime, fewer transitions)
        """
        # Days component: 0-30 days maps to 0-0.5
        days_score = min(Decimal("0.5"), Decimal(str(days_in_regime)) / Decimal("60"))

        # Transitions component: 0 transitions = 0.5, 3+ = 0
        trans_score = max(Decimal("0"), Decimal("0.5") - Decimal(str(transitions_30d)) * Decimal("0.17"))

        stability = (days_score + trans_score).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return min(Decimal("1.0"), stability)

    def update(
        self,
        vix: Decimal,
        xbi_momentum: Decimal,
        as_of_date: date,
    ) -> Dict[str, Any]:
        """
        Update regime detection with new data.

        Applies hysteresis to prevent unstable switching.

        Args:
            vix: Current VIX level
            xbi_momentum: XBI vs SPY momentum (e.g., 0.05 = 5% outperformance)
            as_of_date: Current analysis date

        Returns:
            Dict with regime info, transition details, and stability metrics

        Raises:
            TypeError: If vix or xbi_momentum are not Decimal types
            ValueError: If vix is negative or as_of_date is None
        """
        # Input validation
        if not isinstance(vix, Decimal):
            raise TypeError(f"vix must be Decimal, got {type(vix).__name__}")
        if not isinstance(xbi_momentum, Decimal):
            raise TypeError(f"xbi_momentum must be Decimal, got {type(xbi_momentum).__name__}")
        if as_of_date is None:
            raise ValueError("as_of_date cannot be None")
        if vix < Decimal("0"):
            raise ValueError(f"vix cannot be negative, got {vix}")

        # Initialize state if first call
        if self._state is None:
            initial_regime = self._classify_regime_raw(
                vix, xbi_momentum,
                self._get_effective_thresholds(RegimeState.SECTOR_ROTATION)
            )
            self._state = HysteresisState(
                current_regime=initial_regime,
                regime_since=as_of_date.isoformat(),
                days_in_regime=0,
                pending_transition=None,
                pending_since=None,
                stability_score=Decimal("0.50"),
                transition_history=[],
            )
            self._pending_count = 0

            return {
                "regime": initial_regime.value,
                "regime_since": as_of_date.isoformat(),
                "days_in_regime": 0,
                "transition_type": TransitionType.NONE.value,
                "pending_transition": None,
                "stability_score": str(self._state.stability_score),
                "is_initial": True,
            }

        # Get thresholds based on current regime
        thresholds = self._get_effective_thresholds(self._state.current_regime)

        # Classify new regime
        new_regime = self._classify_regime_raw(vix, xbi_momentum, thresholds)

        # Update days in regime
        regime_since = date.fromisoformat(self._state.regime_since)
        days_in_regime = (as_of_date - regime_since).days

        # Check for transition
        transition_type = TransitionType.NONE
        transition = None

        if new_regime != self._state.current_regime:
            # Potential transition detected

            # Check minimum days requirement
            if days_in_regime < self.min_days_before_transition:
                # Too soon to transition - reject
                transition_type = TransitionType.REJECTED
                new_regime = self._state.current_regime  # Stay in current
            elif self._state.pending_transition == new_regime:
                # Same transition pending - increment counter
                self._pending_count += 1
                if self._pending_count >= self.confirmation_days:
                    # Confirmed transition
                    transition_type = TransitionType.CONFIRMED
                    transition = RegimeTransition(
                        from_regime=self._state.current_regime,
                        to_regime=new_regime,
                        transition_date=as_of_date.isoformat(),
                        transition_type=TransitionType.CONFIRMED,
                        days_in_prior_regime=days_in_regime,
                        trigger_values={
                            "vix": str(vix),
                            "xbi_momentum": str(xbi_momentum),
                        },
                    )
                    self._state.transition_history.append(transition)

                    # Update state
                    self._state.current_regime = new_regime
                    self._state.regime_since = as_of_date.isoformat()
                    self._state.pending_transition = None
                    self._state.pending_since = None
                    self._pending_count = 0
                    days_in_regime = 0
                else:
                    # Still pending
                    transition_type = TransitionType.PENDING
                    new_regime = self._state.current_regime  # Stay for now
            else:
                # New pending transition
                transition_type = TransitionType.PENDING
                self._state.pending_transition = new_regime
                self._state.pending_since = as_of_date.isoformat()
                self._pending_count = 1
                new_regime = self._state.current_regime  # Stay for now
        else:
            # No transition - clear any pending
            if self._state.pending_transition is not None:
                # Pending transition cancelled (returned to current regime)
                transition_type = TransitionType.REJECTED
            self._state.pending_transition = None
            self._state.pending_since = None
            self._pending_count = 0

        # Update days
        self._state.days_in_regime = days_in_regime

        # Compute stability
        recent_transitions = sum(1 for t in self._state.transition_history[-10:]
                                 if t.transition_type == TransitionType.CONFIRMED)
        self._state.stability_score = self._compute_stability_score(
            days_in_regime, recent_transitions
        )

        return {
            "regime": self._state.current_regime.value,
            "regime_since": self._state.regime_since,
            "days_in_regime": self._state.days_in_regime,
            "transition_type": transition_type.value,
            "pending_transition": self._state.pending_transition.value if self._state.pending_transition else None,
            "pending_days": self._pending_count if self._state.pending_transition else 0,
            "stability_score": str(self._state.stability_score),
            "transition": transition.to_dict() if transition else None,
            "is_initial": False,
            "input": {
                "vix": str(vix),
                "xbi_momentum": str(xbi_momentum),
            },
            "effective_thresholds": {k: str(v) for k, v in thresholds.items()},
        }

    def get_state(self) -> Optional[HysteresisState]:
        """Get current hysteresis state."""
        return self._state

    def get_transition_history(self) -> List[RegimeTransition]:
        """Get transition history."""
        if self._state:
            return self._state.transition_history.copy()
        return []

    def reset(self) -> None:
        """Reset engine state."""
        self._state = None
        self._pending_count = 0


def integrate_with_regime_engine(
    regime_engine_result: Dict[str, Any],
    hysteresis_engine: RegimeHysteresisEngine,
    as_of_date: date,
) -> Dict[str, Any]:
    """
    Integrate hysteresis with existing RegimeDetectionEngine output.

    Args:
        regime_engine_result: Output from RegimeDetectionEngine.detect_regime()
        hysteresis_engine: Hysteresis engine instance
        as_of_date: Current analysis date

    Returns:
        Enhanced result with hysteresis-stabilized regime
    """
    # Extract VIX and XBI values from regime engine result
    indicators = regime_engine_result.get("indicators", {})
    vix_str = indicators.get("vix_value")
    xbi_str = indicators.get("xbi_vs_spy_30d_value")

    if vix_str is None or xbi_str is None:
        # Can't apply hysteresis without values
        regime_engine_result["hysteresis"] = {
            "applied": False,
            "reason": "Missing VIX or XBI values",
        }
        return regime_engine_result

    # Safe Decimal conversion with validation
    try:
        vix = Decimal(vix_str)
        xbi = Decimal(xbi_str) / Decimal("100")  # Convert to decimal ratio
    except (ValueError, TypeError, ArithmeticError) as e:
        regime_engine_result["hysteresis"] = {
            "applied": False,
            "reason": f"Invalid VIX or XBI value: {e}",
        }
        return regime_engine_result

    # Update hysteresis
    hysteresis_result = hysteresis_engine.update(vix, xbi, as_of_date)

    # Replace regime with hysteresis-stabilized version
    raw_regime = regime_engine_result["regime"]
    stabilized_regime = hysteresis_result["regime"]

    regime_engine_result["raw_regime"] = raw_regime
    regime_engine_result["regime"] = stabilized_regime
    regime_engine_result["hysteresis"] = {
        "applied": True,
        "stabilized_regime": stabilized_regime,
        "raw_regime": raw_regime,
        "regime_changed": raw_regime != stabilized_regime,
        "stability_score": hysteresis_result["stability_score"],
        "days_in_regime": hysteresis_result["days_in_regime"],
        "pending_transition": hysteresis_result["pending_transition"],
        "transition_type": hysteresis_result["transition_type"],
    }

    return regime_engine_result


if __name__ == "__main__":
    # Demonstration
    print("=" * 70)
    print("REGIME HYSTERESIS ENGINE - DEMONSTRATION")
    print("=" * 70)

    engine = RegimeHysteresisEngine()

    # Simulate regime transitions
    test_data = [
        (date(2026, 1, 1), Decimal("18.0"), Decimal("0.03")),   # BULL
        (date(2026, 1, 2), Decimal("19.0"), Decimal("0.02")),   # Still BULL
        (date(2026, 1, 3), Decimal("23.0"), Decimal("0.01")),   # Approaching BEAR threshold
        (date(2026, 1, 4), Decimal("26.0"), Decimal("-0.01")),  # Crosses threshold
        (date(2026, 1, 5), Decimal("27.0"), Decimal("-0.02")),  # Still above (pending)
        (date(2026, 1, 6), Decimal("28.0"), Decimal("-0.03")),  # Confirmed BEAR
        (date(2026, 1, 7), Decimal("27.0"), Decimal("-0.02")),  # Still BEAR
        (date(2026, 1, 8), Decimal("24.0"), Decimal("-0.01")),  # Improving
        (date(2026, 1, 9), Decimal("22.0"), Decimal("0.01")),   # Back to neutral
    ]

    print("\nSimulating regime transitions with hysteresis:\n")
    for as_of, vix, xbi in test_data:
        result = engine.update(vix, xbi, as_of)
        print(f"{as_of}: VIX={vix}, XBI={xbi*100:.0f}%")
        print(f"  Regime: {result['regime']}")
        print(f"  Transition: {result['transition_type']}")
        if result['pending_transition']:
            print(f"  Pending: {result['pending_transition']} (day {result['pending_days']})")
        print(f"  Stability: {result['stability_score']}")
        print()
