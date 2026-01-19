#!/usr/bin/env python3
"""
regime_engine.py

Market Regime Detection Engine for Biotech Screener

Detects market regime and provides signal weight adjustments to adapt
screening behavior to different market conditions.

Regimes:
1. BULL (Risk-On): Low VIX, XBI outperforming SPY, falling rates
2. BEAR (Risk-Off): High VIX, XBI underperforming, rising rates
3. VOLATILITY_SPIKE: VIX >30, extreme moves, uncertainty
4. SECTOR_ROTATION: Mixed signals, relative strength varies

Signal Adjustments by Regime:
- BULL: Momentum +20%, Fundamental -10% (chase growth)
- BEAR: Momentum -20%, Fundamental +10%, Quality +15% (defensive)
- VOLATILITY_SPIKE: All signals dampened, quality premium
- SECTOR_ROTATION: Neutral weights, slight quality bias

Design Philosophy:
- Deterministic regime classification with explicit thresholds
- Stdlib-only for corporate safety
- Decimal arithmetic for precision
- Full audit trail for every regime determination

Author: Wake Robin Capital Management
Version: 1.0.0
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any, Tuple
from datetime import date
from enum import Enum

# Import momentum health monitor for IC-based adjustments
try:
    from momentum_health_monitor import (
        MomentumHealthMonitor,
        IC_GOOD,
        IC_MARGINAL,
        IC_WEAK,
        WEIGHT_DISABLED,
        WEIGHT_MINIMAL,
        WEIGHT_REDUCED,
    )
    HAS_MOMENTUM_MONITOR = True
except ImportError:
    HAS_MOMENTUM_MONITOR = False


# Module metadata
__version__ = "1.2.0"  # Added staleness gating for data freshness
__author__ = "Wake Robin Capital Management"


class MarketRegime(Enum):
    """Market regime classifications."""
    BULL = "BULL"
    BEAR = "BEAR"
    VOLATILITY_SPIKE = "VOLATILITY_SPIKE"
    SECTOR_ROTATION = "SECTOR_ROTATION"
    UNKNOWN = "UNKNOWN"


class RegimeDetectionEngine:
    """
    Market regime detection and signal weight adjustment engine.

    Classifies market conditions and provides optimal signal weights
    for each regime based on historical performance patterns.

    Now includes IC-based momentum health monitoring to disable momentum
    when the signal shows persistent negative IC (mean reversion).

    Usage:
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("18.5"),
            xbi_vs_spy_30d=Decimal("3.2"),
            fed_rate_change_3m=Decimal("-0.25"),
            momentum_ic_3m=Decimal("0.12")  # Optional IC for health check
        )
        print(result["regime"])  # "BULL"
        print(result["signal_adjustments"])  # Weights for each signal type
    """

    VERSION = "1.2.0"  # Added staleness gating

    # VIX thresholds for regime classification
    VIX_LOW = Decimal("15")      # Below = calm markets
    VIX_NORMAL = Decimal("20")   # Normal range
    VIX_ELEVATED = Decimal("25") # Elevated concern
    VIX_HIGH = Decimal("30")     # High volatility
    VIX_EXTREME = Decimal("40")  # Crisis levels

    # XBI vs SPY relative performance thresholds (30-day)
    XBI_OUTPERFORM_STRONG = Decimal("5")   # Strong biotech rally
    XBI_OUTPERFORM_MODERATE = Decimal("2") # Moderate outperformance
    XBI_UNDERPERFORM_MODERATE = Decimal("-2")  # Moderate underperformance
    XBI_UNDERPERFORM_STRONG = Decimal("-5")    # Biotech selloff

    # Fed rate change thresholds (3-month basis points)
    RATE_HIKE_AGGRESSIVE = Decimal("0.50")   # Aggressive tightening
    RATE_HIKE_MODERATE = Decimal("0.25")     # Moderate tightening
    RATE_CUT_MODERATE = Decimal("-0.25")     # Moderate easing
    RATE_CUT_AGGRESSIVE = Decimal("-0.50")   # Aggressive easing

    # Data staleness thresholds and confidence haircuts
    # Regime is time-sensitive; stale data should not drive full weight tilts
    STALENESS_THRESHOLDS: Dict[int, Decimal] = {
        2: Decimal("1.00"),   # ≤2 days: full confidence
        5: Decimal("0.85"),   # 3-5 days: 15% haircut
        10: Decimal("0.65"),  # 6-10 days: 35% haircut
    }
    STALENESS_MAX_DAYS = 10  # >10 days: force UNKNOWN regime

    # Signal weight adjustments by regime
    REGIME_ADJUSTMENTS: Dict[str, Dict[str, Decimal]] = {
        "BULL": {
            "momentum": Decimal("1.20"),      # Boost momentum (chase winners)
            "fundamental": Decimal("0.90"),   # Slightly reduce fundamental focus
            "quality": Decimal("1.00"),       # Neutral quality
            "catalyst": Decimal("1.15"),      # Boost catalyst plays
            "institutional": Decimal("1.05"), # Slight institutional boost
            "clinical": Decimal("1.10"),      # Boost clinical progress
            "financial": Decimal("0.95")      # Slightly reduce financial focus
        },
        "BEAR": {
            "momentum": Decimal("0.80"),      # Reduce momentum (avoid falling knives)
            "fundamental": Decimal("1.15"),   # Boost fundamental focus
            "quality": Decimal("1.20"),       # Strong quality premium
            "catalyst": Decimal("0.90"),      # Reduce catalyst speculation
            "institutional": Decimal("1.15"), # Boost institutional backing
            "clinical": Decimal("1.05"),      # Slight clinical boost
            "financial": Decimal("1.20")      # Strong financial focus (runway)
        },
        "VOLATILITY_SPIKE": {
            "momentum": Decimal("0.70"),      # Strongly reduce momentum
            "fundamental": Decimal("0.95"),   # Slight fundamental reduction
            "quality": Decimal("1.30"),       # Maximum quality premium
            "catalyst": Decimal("0.80"),      # Reduce catalyst (uncertainty)
            "institutional": Decimal("1.10"), # Institutional stability premium
            "clinical": Decimal("0.90"),      # Reduce clinical speculation
            "financial": Decimal("1.25")      # Strong runway focus
        },
        "SECTOR_ROTATION": {
            "momentum": Decimal("1.00"),      # Neutral momentum
            "fundamental": Decimal("1.05"),   # Slight fundamental bias
            "quality": Decimal("1.10"),       # Slight quality premium
            "catalyst": Decimal("1.00"),      # Neutral catalyst
            "institutional": Decimal("1.05"), # Slight institutional premium
            "clinical": Decimal("1.00"),      # Neutral clinical
            "financial": Decimal("1.05")      # Slight financial focus
        },
        "UNKNOWN": {
            "momentum": Decimal("1.00"),
            "fundamental": Decimal("1.00"),
            "quality": Decimal("1.00"),
            "catalyst": Decimal("1.00"),
            "institutional": Decimal("1.00"),
            "clinical": Decimal("1.00"),
            "financial": Decimal("1.00")
        }
    }

    # Regime descriptions for reporting
    REGIME_DESCRIPTIONS: Dict[str, str] = {
        "BULL": "Risk-on environment favorable for growth/momentum strategies",
        "BEAR": "Risk-off environment requiring defensive positioning",
        "VOLATILITY_SPIKE": "High uncertainty requiring maximum quality focus",
        "SECTOR_ROTATION": "Mixed signals, balanced approach recommended",
        "UNKNOWN": "Insufficient data for regime classification"
    }

    def __init__(self):
        """Initialize the regime detection engine."""
        self.current_regime: Optional[str] = None
        self.regime_history: List[Dict[str, Any]] = []
        self.audit_trail: List[Dict[str, Any]] = []
        self.momentum_monitor: Optional[Any] = None

        # Initialize momentum health monitor if available
        if HAS_MOMENTUM_MONITOR:
            self.momentum_monitor = MomentumHealthMonitor()

    def compute_staleness_haircut(
        self,
        data_as_of_date: date,
        run_as_of_date: date
    ) -> Tuple[int, Decimal, bool]:
        """
        Compute confidence haircut based on data staleness.

        Args:
            data_as_of_date: Date of the market snapshot data
            run_as_of_date: Date of the screening run (as_of_date)

        Returns:
            Tuple of (age_days, haircut_multiplier, is_stale)
            - age_days: Number of days between data and run
            - haircut_multiplier: Decimal multiplier (1.0 = no haircut, 0.0 = fully stale)
            - is_stale: True if data is too old and regime should be UNKNOWN
        """
        age_days = (run_as_of_date - data_as_of_date).days

        # Check if too stale
        if age_days > self.STALENESS_MAX_DAYS:
            return (age_days, Decimal("0.00"), True)

        # Find applicable haircut
        haircut = Decimal("1.00")
        for threshold_days, multiplier in sorted(self.STALENESS_THRESHOLDS.items()):
            if age_days <= threshold_days:
                haircut = multiplier
                break
        else:
            # Beyond all thresholds but under max - use last threshold
            haircut = Decimal("0.65")

        return (age_days, haircut, False)

    def detect_regime(
        self,
        vix_current: Decimal,
        xbi_vs_spy_30d: Decimal,  # % relative performance last 30 days
        fed_rate_change_3m: Optional[Decimal] = None,  # Rate change last 3 months
        xbi_momentum_10d: Optional[Decimal] = None,    # 10-day XBI momentum
        spy_momentum_10d: Optional[Decimal] = None,    # 10-day SPY momentum
        credit_spread_change: Optional[Decimal] = None, # HY spread change
        momentum_ic_3m: Optional[Decimal] = None,      # 3-month rolling momentum IC
        as_of_date: Optional[date] = None,
        data_as_of_date: Optional[date] = None         # Date of market snapshot data
    ) -> Dict[str, Any]:
        """
        Detect current market regime and return signal adjustments.

        Args:
            vix_current: Current VIX level
            xbi_vs_spy_30d: XBI vs SPY relative performance (30 days, %)
            fed_rate_change_3m: Fed funds rate change (3 months, %)
            xbi_momentum_10d: Optional XBI 10-day momentum
            spy_momentum_10d: Optional SPY 10-day momentum
            credit_spread_change: Optional HY credit spread change
            momentum_ic_3m: Optional 3-month rolling momentum IC for health check
            as_of_date: Point-in-time date

        Returns:
            Dict containing:
            - regime: str (BULL, BEAR, VOLATILITY_SPIKE, SECTOR_ROTATION)
            - regime_description: str
            - confidence: Decimal (0-1 classification confidence)
            - signal_adjustments: Dict[str, Decimal] (weight multipliers)
            - momentum_health: Dict (IC-based momentum health if provided)
            - indicators: Dict (contributing factors)
            - audit_entry: Dict
        """

        # Calculate regime scores for each classification
        regime_scores = self._calculate_regime_scores(
            vix_current=vix_current,
            xbi_vs_spy_30d=xbi_vs_spy_30d,
            fed_rate_change_3m=fed_rate_change_3m,
            xbi_momentum_10d=xbi_momentum_10d,
            spy_momentum_10d=spy_momentum_10d,
            credit_spread_change=credit_spread_change
        )

        # Determine primary regime and confidence
        regime, confidence = self._select_regime(regime_scores)
        raw_confidence = confidence  # Preserve pre-haircut value

        # Apply staleness gating if both dates provided
        staleness_info = None
        if data_as_of_date is not None and as_of_date is not None:
            age_days, haircut, is_stale = self.compute_staleness_haircut(
                data_as_of_date, as_of_date
            )

            staleness_info = {
                "data_as_of_date": data_as_of_date.isoformat(),
                "run_as_of_date": as_of_date.isoformat(),
                "age_days": age_days,
                "haircut_multiplier": str(haircut),
                "is_stale": is_stale,
                "raw_confidence": str(raw_confidence),
            }

            if is_stale:
                # Data too old - force UNKNOWN regime with neutral weights
                regime = "UNKNOWN"
                confidence = Decimal("0.00")
                staleness_info["action"] = "FORCED_UNKNOWN"
                staleness_info["reason"] = f"Data age ({age_days} days) exceeds max ({self.STALENESS_MAX_DAYS} days)"
            else:
                # Apply haircut to confidence
                confidence = (confidence * haircut).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
                staleness_info["adjusted_confidence"] = str(confidence)
                staleness_info["action"] = "HAIRCUT_APPLIED" if haircut < Decimal("1.00") else "FULL_CONFIDENCE"

        # Get signal adjustments for this regime
        signal_adjustments = self.REGIME_ADJUSTMENTS.get(
            regime,
            self.REGIME_ADJUSTMENTS["UNKNOWN"]
        ).copy()

        # Apply IC-based momentum health adjustment (kill switch)
        momentum_health = None
        if momentum_ic_3m is not None and HAS_MOMENTUM_MONITOR:
            momentum_health = self._apply_momentum_health_adjustment(
                signal_adjustments,
                momentum_ic_3m,
                regime
            )

        # Compile indicator summary
        indicators = {
            "vix_level": self._classify_vix(vix_current),
            "biotech_relative_performance": self._classify_xbi_performance(xbi_vs_spy_30d),
            "rate_environment": self._classify_rate_environment(fed_rate_change_3m),
            "vix_value": str(vix_current),
            "xbi_vs_spy_30d_value": str(xbi_vs_spy_30d),
            "fed_rate_change_3m_value": str(fed_rate_change_3m) if fed_rate_change_3m else None
        }

        # Generate flags
        flags = self._generate_flags(vix_current, xbi_vs_spy_30d, regime)

        # Update state
        self.current_regime = regime

        # Deterministic timestamp from as_of_date
        deterministic_timestamp = f"{as_of_date.isoformat()}T00:00:00Z" if as_of_date else None

        # Audit trail entry (deterministic!)
        audit_entry = {
            "timestamp": deterministic_timestamp,
            "as_of_date": as_of_date.isoformat() if as_of_date else None,
            "input": {
                "vix_current": str(vix_current),
                "xbi_vs_spy_30d": str(xbi_vs_spy_30d),
                "fed_rate_change_3m": str(fed_rate_change_3m) if fed_rate_change_3m else None,
                "xbi_momentum_10d": str(xbi_momentum_10d) if xbi_momentum_10d else None,
                "spy_momentum_10d": str(spy_momentum_10d) if spy_momentum_10d else None,
                "credit_spread_change": str(credit_spread_change) if credit_spread_change else None,
                "momentum_ic_3m": str(momentum_ic_3m) if momentum_ic_3m else None,
                "data_as_of_date": data_as_of_date.isoformat() if data_as_of_date else None
            },
            "staleness": staleness_info,
            "momentum_health": momentum_health,
            "regime_scores": {k: str(v) for k, v in regime_scores.items()},
            "regime": regime,
            "confidence": str(confidence),
            "signal_adjustments": {k: str(v) for k, v in signal_adjustments.items()},
            "indicators": indicators,
            "flags": flags,
            "module_version": self.VERSION
        }

        self.audit_trail.append(audit_entry)
        self.regime_history.append({
            "as_of_date": as_of_date.isoformat() if as_of_date else None,
            "regime": regime,
            "confidence": confidence
        })

        return {
            "regime": regime,
            "regime_description": self.REGIME_DESCRIPTIONS.get(regime, ""),
            "confidence": confidence,
            "signal_adjustments": signal_adjustments,
            "staleness": staleness_info,
            "momentum_health": momentum_health,
            "regime_scores": regime_scores,
            "indicators": indicators,
            "flags": flags,
            "audit_entry": audit_entry
        }

    def apply_regime_weights(
        self,
        base_scores: Dict[str, Decimal],
        regime_adjustments: Dict[str, Decimal]
    ) -> Dict[str, Decimal]:
        """
        Apply regime-based weight adjustments to base scores.

        Args:
            base_scores: Dict of signal_name → raw_score
            regime_adjustments: Dict of signal_name → multiplier

        Returns:
            Dict of signal_name → adjusted_score
        """
        adjusted_scores = {}

        for signal_name, base_score in base_scores.items():
            multiplier = regime_adjustments.get(signal_name, Decimal("1.0"))
            adjusted = base_score * multiplier
            adjusted_scores[signal_name] = adjusted.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        return adjusted_scores

    def get_composite_weight_adjustments(
        self,
        regime: str,
        base_weights: Dict[str, Decimal]
    ) -> Dict[str, Decimal]:
        """
        Get adjusted composite weights for a given regime.

        Adjusts the base module weights (clinical, financial, catalyst, etc.)
        based on the current regime.

        Args:
            regime: Current regime classification
            base_weights: Base weight allocation (must sum to 1.0)

        Returns:
            Adjusted weights (renormalized to sum to 1.0)
        """
        adjustments = self.REGIME_ADJUSTMENTS.get(regime, self.REGIME_ADJUSTMENTS["UNKNOWN"])

        # Apply multipliers
        adjusted = {}
        for component, base_weight in base_weights.items():
            multiplier = adjustments.get(component, Decimal("1.0"))
            adjusted[component] = base_weight * multiplier

        # Renormalize to sum to 1.0
        total = sum(adjusted.values())
        if total > Decimal("0"):
            normalized = {
                k: (v / total).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
                for k, v in adjusted.items()
            }
        else:
            normalized = base_weights.copy()

        return normalized

    def _calculate_regime_scores(
        self,
        vix_current: Decimal,
        xbi_vs_spy_30d: Decimal,
        fed_rate_change_3m: Optional[Decimal],
        xbi_momentum_10d: Optional[Decimal],
        spy_momentum_10d: Optional[Decimal],
        credit_spread_change: Optional[Decimal]
    ) -> Dict[str, Decimal]:
        """Calculate classification scores for each regime."""

        scores = {
            "BULL": Decimal("0"),
            "BEAR": Decimal("0"),
            "VOLATILITY_SPIKE": Decimal("0"),
            "SECTOR_ROTATION": Decimal("0")
        }

        # VIX contribution
        if vix_current >= self.VIX_EXTREME:
            scores["VOLATILITY_SPIKE"] += Decimal("40")
            scores["BEAR"] += Decimal("20")
        elif vix_current >= self.VIX_HIGH:
            scores["VOLATILITY_SPIKE"] += Decimal("30")
            scores["BEAR"] += Decimal("15")
        elif vix_current >= self.VIX_ELEVATED:
            scores["BEAR"] += Decimal("15")
            scores["SECTOR_ROTATION"] += Decimal("10")
        elif vix_current >= self.VIX_NORMAL:
            scores["SECTOR_ROTATION"] += Decimal("15")
        elif vix_current <= self.VIX_LOW:
            scores["BULL"] += Decimal("25")
        else:
            scores["SECTOR_ROTATION"] += Decimal("10")

        # XBI relative performance contribution
        if xbi_vs_spy_30d >= self.XBI_OUTPERFORM_STRONG:
            scores["BULL"] += Decimal("30")
        elif xbi_vs_spy_30d >= self.XBI_OUTPERFORM_MODERATE:
            scores["BULL"] += Decimal("20")
        elif xbi_vs_spy_30d <= self.XBI_UNDERPERFORM_STRONG:
            scores["BEAR"] += Decimal("30")
        elif xbi_vs_spy_30d <= self.XBI_UNDERPERFORM_MODERATE:
            scores["BEAR"] += Decimal("20")
        else:
            scores["SECTOR_ROTATION"] += Decimal("15")

        # Rate environment contribution
        if fed_rate_change_3m is not None:
            if fed_rate_change_3m >= self.RATE_HIKE_AGGRESSIVE:
                scores["BEAR"] += Decimal("20")
            elif fed_rate_change_3m >= self.RATE_HIKE_MODERATE:
                scores["BEAR"] += Decimal("10")
            elif fed_rate_change_3m <= self.RATE_CUT_AGGRESSIVE:
                scores["BULL"] += Decimal("20")
            elif fed_rate_change_3m <= self.RATE_CUT_MODERATE:
                scores["BULL"] += Decimal("10")
            else:
                scores["SECTOR_ROTATION"] += Decimal("5")

        # Credit spread contribution (if available)
        if credit_spread_change is not None:
            if credit_spread_change >= Decimal("50"):  # Spreads widening 50+ bps
                scores["BEAR"] += Decimal("15")
                scores["VOLATILITY_SPIKE"] += Decimal("10")
            elif credit_spread_change <= Decimal("-30"):  # Spreads tightening
                scores["BULL"] += Decimal("10")

        # Momentum confirmation (if available)
        if xbi_momentum_10d is not None and spy_momentum_10d is not None:
            if xbi_momentum_10d > Decimal("0") and spy_momentum_10d > Decimal("0"):
                scores["BULL"] += Decimal("10")
            elif xbi_momentum_10d < Decimal("0") and spy_momentum_10d < Decimal("0"):
                scores["BEAR"] += Decimal("10")
            else:
                scores["SECTOR_ROTATION"] += Decimal("10")

        return scores

    def _select_regime(
        self,
        regime_scores: Dict[str, Decimal]
    ) -> Tuple[str, Decimal]:
        """Select the regime with highest score and calculate confidence."""

        # Find max score
        max_regime = max(regime_scores, key=regime_scores.get)
        max_score = regime_scores[max_regime]

        # Calculate total for confidence
        total_score = sum(regime_scores.values())

        if total_score > Decimal("0"):
            confidence = (max_score / total_score).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        else:
            return "UNKNOWN", Decimal("0.00")

        # Require minimum confidence threshold
        if confidence < Decimal("0.30"):
            return "SECTOR_ROTATION", confidence

        return max_regime, confidence

    def _classify_vix(self, vix: Decimal) -> str:
        """Classify VIX level."""
        if vix >= self.VIX_EXTREME:
            return "CRISIS"
        elif vix >= self.VIX_HIGH:
            return "HIGH"
        elif vix >= self.VIX_ELEVATED:
            return "ELEVATED"
        elif vix >= self.VIX_NORMAL:
            return "NORMAL"
        elif vix >= self.VIX_LOW:
            return "LOW"
        else:
            return "VERY_LOW"

    def _classify_xbi_performance(self, xbi_vs_spy: Decimal) -> str:
        """Classify XBI relative performance."""
        if xbi_vs_spy >= self.XBI_OUTPERFORM_STRONG:
            return "STRONG_OUTPERFORMANCE"
        elif xbi_vs_spy >= self.XBI_OUTPERFORM_MODERATE:
            return "MODERATE_OUTPERFORMANCE"
        elif xbi_vs_spy <= self.XBI_UNDERPERFORM_STRONG:
            return "STRONG_UNDERPERFORMANCE"
        elif xbi_vs_spy <= self.XBI_UNDERPERFORM_MODERATE:
            return "MODERATE_UNDERPERFORMANCE"
        else:
            return "NEUTRAL"

    def _classify_rate_environment(self, rate_change: Optional[Decimal]) -> str:
        """Classify rate environment."""
        if rate_change is None:
            return "UNKNOWN"
        if rate_change >= self.RATE_HIKE_AGGRESSIVE:
            return "AGGRESSIVE_TIGHTENING"
        elif rate_change >= self.RATE_HIKE_MODERATE:
            return "MODERATE_TIGHTENING"
        elif rate_change <= self.RATE_CUT_AGGRESSIVE:
            return "AGGRESSIVE_EASING"
        elif rate_change <= self.RATE_CUT_MODERATE:
            return "MODERATE_EASING"
        else:
            return "STABLE"

    def _generate_flags(
        self,
        vix: Decimal,
        xbi_vs_spy: Decimal,
        regime: str
    ) -> List[str]:
        """Generate warning/info flags."""
        flags = []

        if vix >= self.VIX_EXTREME:
            flags.append("CRISIS_VOLATILITY")
        elif vix >= self.VIX_HIGH:
            flags.append("HIGH_VOLATILITY")

        if xbi_vs_spy <= Decimal("-10"):
            flags.append("BIOTECH_SELLOFF")
        elif xbi_vs_spy >= Decimal("10"):
            flags.append("BIOTECH_RALLY")

        if regime == "VOLATILITY_SPIKE":
            flags.append("REDUCE_POSITION_SIZE")
            flags.append("QUALITY_FOCUS_REQUIRED")
        elif regime == "BEAR":
            flags.append("DEFENSIVE_POSITIONING")

        return flags

    def _apply_momentum_health_adjustment(
        self,
        signal_adjustments: Dict[str, Decimal],
        momentum_ic_3m: Decimal,
        regime: str
    ) -> Dict[str, Any]:
        """
        Apply IC-based momentum health adjustment (kill switch).

        Modifies signal_adjustments in-place to reduce or disable
        momentum when the IC indicates the signal isn't working.

        Args:
            signal_adjustments: Dict of signal weights (modified in-place)
            momentum_ic_3m: 3-month rolling IC
            regime: Current market regime

        Returns:
            Dict with momentum health assessment
        """
        original_momentum = signal_adjustments.get("momentum", Decimal("1.00"))

        # Use the monitor if available
        if self.momentum_monitor:
            self.momentum_monitor.update_ic(float(momentum_ic_3m))
            adjusted_weight = self.momentum_monitor.check_momentum_health(float(momentum_ic_3m))

            # If health check returns full weight, use regime weight
            if adjusted_weight == Decimal("1.00"):
                final_weight = original_momentum
            else:
                final_weight = adjusted_weight
        else:
            # Fallback: Simple threshold check
            if momentum_ic_3m < IC_WEAK:
                final_weight = WEIGHT_DISABLED
            elif momentum_ic_3m < IC_MARGINAL:
                final_weight = WEIGHT_MINIMAL
            elif momentum_ic_3m < IC_GOOD:
                final_weight = WEIGHT_REDUCED
            else:
                final_weight = original_momentum

        # Update signal_adjustments in place
        signal_adjustments["momentum"] = final_weight

        # Classify health status
        if momentum_ic_3m >= IC_GOOD:
            health_status = "HEALTHY"
        elif momentum_ic_3m >= IC_MARGINAL:
            health_status = "MARGINAL"
        elif momentum_ic_3m >= IC_WEAK:
            health_status = "WEAK"
        else:
            health_status = "INVERTED"

        return {
            "ic_3m": str(momentum_ic_3m),
            "health_status": health_status,
            "original_weight": str(original_momentum),
            "adjusted_weight": str(final_weight),
            "regime": regime,
            "action": "DISABLED" if final_weight == Decimal("0") else
                      "REDUCED" if final_weight < original_momentum else "FULL",
        }

    def get_regime_history(self) -> List[Dict[str, Any]]:
        """Return regime classification history."""
        return self.regime_history.copy()

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Return full audit trail."""
        return self.audit_trail.copy()

    def clear_state(self) -> None:
        """Clear all state (for testing/reset)."""
        self.current_regime = None
        self.regime_history = []
        self.audit_trail = []
        if HAS_MOMENTUM_MONITOR and self.momentum_monitor:
            self.momentum_monitor = MomentumHealthMonitor()


def demonstration() -> None:
    """Demonstrate the regime detection engine capabilities."""
    print("=" * 70)
    print("MARKET REGIME DETECTION ENGINE - DEMONSTRATION")
    print("=" * 70)
    print()

    engine = RegimeDetectionEngine()

    # Example 1: Bull market conditions
    print("Example 1: Bull Market Conditions")
    print("-" * 70)

    result1 = engine.detect_regime(
        vix_current=Decimal("14.5"),
        xbi_vs_spy_30d=Decimal("6.2"),
        fed_rate_change_3m=Decimal("-0.25")
    )

    print(f"Regime: {result1['regime']}")
    print(f"Description: {result1['regime_description']}")
    print(f"Confidence: {result1['confidence']}")
    print(f"Indicators: {result1['indicators']}")
    print("Signal Adjustments:")
    for signal, mult in result1['signal_adjustments'].items():
        print(f"  {signal}: {mult}x")
    print()

    # Example 2: Bear market conditions
    print("Example 2: Bear Market Conditions")
    print("-" * 70)

    result2 = engine.detect_regime(
        vix_current=Decimal("28.5"),
        xbi_vs_spy_30d=Decimal("-7.3"),
        fed_rate_change_3m=Decimal("0.75")
    )

    print(f"Regime: {result2['regime']}")
    print(f"Confidence: {result2['confidence']}")
    print(f"Flags: {result2['flags']}")
    print("Key Adjustments:")
    print(f"  Momentum: {result2['signal_adjustments']['momentum']}x")
    print(f"  Quality: {result2['signal_adjustments']['quality']}x")
    print(f"  Financial: {result2['signal_adjustments']['financial']}x")
    print()

    # Example 3: Volatility spike
    print("Example 3: Volatility Spike")
    print("-" * 70)

    result3 = engine.detect_regime(
        vix_current=Decimal("42.0"),
        xbi_vs_spy_30d=Decimal("-3.5"),
        fed_rate_change_3m=Decimal("0.00")
    )

    print(f"Regime: {result3['regime']}")
    print(f"Confidence: {result3['confidence']}")
    print(f"Flags: {result3['flags']}")
    print()

    # Example 4: Weight adjustment demonstration
    print("Example 4: Composite Weight Adjustment")
    print("-" * 70)

    base_weights = {
        "clinical": Decimal("0.40"),
        "financial": Decimal("0.35"),
        "catalyst": Decimal("0.25")
    }

    adjusted_weights = engine.get_composite_weight_adjustments(
        regime="BEAR",
        base_weights=base_weights
    )

    print("Base Weights → Adjusted Weights (BEAR regime):")
    for component in base_weights:
        print(f"  {component}: {base_weights[component]} → {adjusted_weights[component]}")
    print()

    # Example 5: IC-based momentum kill switch
    print("Example 5: IC-Based Momentum Health (Kill Switch)")
    print("-" * 70)

    # Create fresh engine
    engine2 = RegimeDetectionEngine()

    # Test with different IC values
    ic_tests = [
        (Decimal("0.15"), "Strong momentum signal"),
        (Decimal("0.08"), "Marginal momentum signal"),
        (Decimal("0.02"), "Weak momentum signal"),
        (Decimal("-0.10"), "Inverted (mean reversion) - DISABLED"),
    ]

    for ic, description in ic_tests:
        result = engine2.detect_regime(
            vix_current=Decimal("18.0"),
            xbi_vs_spy_30d=Decimal("1.0"),
            fed_rate_change_3m=Decimal("0.00"),
            momentum_ic_3m=ic
        )

        print(f"\nIC = {ic} ({description})")
        if result.get("momentum_health"):
            mh = result["momentum_health"]
            print(f"  Health Status: {mh['health_status']}")
            print(f"  Original Weight: {mh['original_weight']}")
            print(f"  Adjusted Weight: {mh['adjusted_weight']}")
            print(f"  Action: {mh['action']}")
        print(f"  Final Momentum Weight: {result['signal_adjustments']['momentum']}x")

    print()


if __name__ == "__main__":
    demonstration()
