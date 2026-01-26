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
from typing import Dict, List, Optional, Any, Tuple, Protocol, runtime_checkable, Callable
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from functools import lru_cache, cached_property

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
__version__ = "1.3.0"  # Added new regimes, signals, callbacks, Kalman, HMM, ensemble
__author__ = "Wake Robin Capital Management"


@runtime_checkable
class RegimeTransitionCallback(Protocol):
    """Protocol for regime transition callbacks."""

    def on_transition(
        self,
        old_regime: str,
        new_regime: str,
        transition_date: date,
        days_in_prior_regime: int,
        trigger_values: Dict[str, str],
    ) -> None:
        """
        Called when a regime transition is confirmed.

        Args:
            old_regime: The previous regime
            new_regime: The new regime
            transition_date: Date of the transition
            days_in_prior_regime: Number of days spent in the prior regime
            trigger_values: Dict of trigger values that caused the transition
        """
        ...


@dataclass
class RegimeTransitionEvent:
    """Record of a regime transition event."""
    old_regime: str
    new_regime: str
    transition_date: date
    days_in_prior_regime: int
    trigger_values: Dict[str, str]
    confidence: Decimal = Decimal("0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "old_regime": self.old_regime,
            "new_regime": self.new_regime,
            "transition_date": self.transition_date.isoformat(),
            "days_in_prior_regime": self.days_in_prior_regime,
            "trigger_values": self.trigger_values,
            "confidence": str(self.confidence),
        }


class MarketRegime(Enum):
    """Market regime classifications."""
    BULL = "BULL"
    BEAR = "BEAR"
    VOLATILITY_SPIKE = "VOLATILITY_SPIKE"
    SECTOR_ROTATION = "SECTOR_ROTATION"
    RECESSION_RISK = "RECESSION_RISK"          # Inverted yield curve + widening credit
    CREDIT_CRISIS = "CREDIT_CRISIS"            # Extreme credit spread (>600bps OAS)
    SECTOR_DISLOCATION = "SECTOR_DISLOCATION"  # Biotech divergence >15%
    UNKNOWN = "UNKNOWN"


class VIXKalmanFilter:
    """
    Simple Kalman filter for VIX smoothing.

    Uses a univariate Kalman filter to smooth VIX measurements and
    reduce noise in volatility signals.

    Usage:
        kf = VIXKalmanFilter()
        smoothed_vix = kf.update(Decimal("22.5"))
    """

    def __init__(
        self,
        process_noise: Decimal = Decimal("0.1"),
        measurement_noise: Decimal = Decimal("1.0"),
        initial_estimate: Decimal = Decimal("20.0"),
        initial_error: Decimal = Decimal("1.0"),
    ):
        """
        Initialize the Kalman filter.

        Args:
            process_noise: Process noise covariance (Q)
            measurement_noise: Measurement noise covariance (R)
            initial_estimate: Initial state estimate
            initial_error: Initial estimation error covariance
        """
        self.Q = process_noise  # Process noise
        self.R = measurement_noise  # Measurement noise
        self.x = initial_estimate  # State estimate
        self.P = initial_error  # Estimation error covariance
        self._initialized = False

    def update(self, measurement: Decimal) -> Decimal:
        """
        Update filter with new measurement and return smoothed value.

        Args:
            measurement: New VIX measurement

        Returns:
            Smoothed VIX value
        """
        if not self._initialized:
            # First measurement initializes the filter
            self.x = measurement
            self._initialized = True
            return self.x

        # Prediction step
        x_pred = self.x  # State prediction (assume constant model)
        P_pred = self.P + self.Q  # Error covariance prediction

        # Update step
        K = P_pred / (P_pred + self.R)  # Kalman gain
        self.x = x_pred + K * (measurement - x_pred)  # State update
        self.P = (Decimal("1") - K) * P_pred  # Error covariance update

        return self.x.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def reset(self) -> None:
        """Reset the filter to initial state."""
        self.x = Decimal("20.0")
        self.P = Decimal("1.0")
        self._initialized = False

    def get_state(self) -> Dict[str, str]:
        """Get current filter state."""
        return {
            "estimate": str(self.x),
            "error_covariance": str(self.P),
            "initialized": str(self._initialized),
        }


class RegimeHMM:
    """
    Hidden Markov Model for regime probability estimation.

    Uses transition probabilities and observation likelihoods to
    estimate the probability distribution over regimes.

    Regimes tend to persist, so diagonal transition probabilities
    are higher than off-diagonal.
    """

    REGIMES = [
        "BULL", "BEAR", "VOLATILITY_SPIKE", "SECTOR_ROTATION",
        "RECESSION_RISK", "CREDIT_CRISIS", "SECTOR_DISLOCATION"
    ]

    # Transition probability matrix (regimes tend to persist)
    # Row = from regime, Column = to regime
    DEFAULT_TRANSITION_PROBS = {
        "BULL": {"BULL": Decimal("0.85"), "BEAR": Decimal("0.05"),
                 "VOLATILITY_SPIKE": Decimal("0.02"), "SECTOR_ROTATION": Decimal("0.05"),
                 "RECESSION_RISK": Decimal("0.01"), "CREDIT_CRISIS": Decimal("0.01"),
                 "SECTOR_DISLOCATION": Decimal("0.01")},
        "BEAR": {"BULL": Decimal("0.05"), "BEAR": Decimal("0.80"),
                 "VOLATILITY_SPIKE": Decimal("0.05"), "SECTOR_ROTATION": Decimal("0.05"),
                 "RECESSION_RISK": Decimal("0.03"), "CREDIT_CRISIS": Decimal("0.01"),
                 "SECTOR_DISLOCATION": Decimal("0.01")},
        "VOLATILITY_SPIKE": {"BULL": Decimal("0.05"), "BEAR": Decimal("0.15"),
                             "VOLATILITY_SPIKE": Decimal("0.60"), "SECTOR_ROTATION": Decimal("0.10"),
                             "RECESSION_RISK": Decimal("0.05"), "CREDIT_CRISIS": Decimal("0.03"),
                             "SECTOR_DISLOCATION": Decimal("0.02")},
        "SECTOR_ROTATION": {"BULL": Decimal("0.15"), "BEAR": Decimal("0.15"),
                            "VOLATILITY_SPIKE": Decimal("0.05"), "SECTOR_ROTATION": Decimal("0.55"),
                            "RECESSION_RISK": Decimal("0.04"), "CREDIT_CRISIS": Decimal("0.02"),
                            "SECTOR_DISLOCATION": Decimal("0.04")},
        "RECESSION_RISK": {"BULL": Decimal("0.02"), "BEAR": Decimal("0.15"),
                           "VOLATILITY_SPIKE": Decimal("0.08"), "SECTOR_ROTATION": Decimal("0.05"),
                           "RECESSION_RISK": Decimal("0.60"), "CREDIT_CRISIS": Decimal("0.08"),
                           "SECTOR_DISLOCATION": Decimal("0.02")},
        "CREDIT_CRISIS": {"BULL": Decimal("0.01"), "BEAR": Decimal("0.10"),
                          "VOLATILITY_SPIKE": Decimal("0.15"), "SECTOR_ROTATION": Decimal("0.04"),
                          "RECESSION_RISK": Decimal("0.10"), "CREDIT_CRISIS": Decimal("0.58"),
                          "SECTOR_DISLOCATION": Decimal("0.02")},
        "SECTOR_DISLOCATION": {"BULL": Decimal("0.10"), "BEAR": Decimal("0.10"),
                               "VOLATILITY_SPIKE": Decimal("0.05"), "SECTOR_ROTATION": Decimal("0.15"),
                               "RECESSION_RISK": Decimal("0.03"), "CREDIT_CRISIS": Decimal("0.02"),
                               "SECTOR_DISLOCATION": Decimal("0.55")},
    }

    def __init__(
        self,
        transition_probs: Optional[Dict[str, Dict[str, Decimal]]] = None,
        initial_probs: Optional[Dict[str, Decimal]] = None,
    ):
        """
        Initialize the HMM.

        Args:
            transition_probs: Custom transition probability matrix
            initial_probs: Initial regime probabilities
        """
        self.transition_probs = transition_probs or self.DEFAULT_TRANSITION_PROBS

        # Default to uniform initial probabilities
        if initial_probs:
            self.state_probs = initial_probs.copy()
        else:
            uniform = Decimal("1") / Decimal(str(len(self.REGIMES)))
            self.state_probs = {regime: uniform for regime in self.REGIMES}

    def update(self, observation_likelihoods: Dict[str, Decimal]) -> Dict[str, Decimal]:
        """
        Update state probabilities based on observation likelihoods.

        Args:
            observation_likelihoods: P(observation | regime) for each regime
                                    (typically normalized regime scores)

        Returns:
            Updated probability distribution over regimes
        """
        # Prediction step: P(s_t | observations_{1:t-1})
        predicted_probs = {}
        for to_regime in self.REGIMES:
            prob = Decimal("0")
            for from_regime in self.REGIMES:
                prob += (self.state_probs.get(from_regime, Decimal("0")) *
                        self.transition_probs.get(from_regime, {}).get(to_regime, Decimal("0")))
            predicted_probs[to_regime] = prob

        # Update step: P(s_t | observations_{1:t})
        updated_probs = {}
        for regime in self.REGIMES:
            likelihood = observation_likelihoods.get(regime, Decimal("0.01"))
            updated_probs[regime] = predicted_probs[regime] * likelihood

        # Normalize
        total = sum(updated_probs.values())
        if total > Decimal("0"):
            for regime in self.REGIMES:
                updated_probs[regime] = (updated_probs[regime] / total).quantize(
                    Decimal("0.0001"), rounding=ROUND_HALF_UP
                )
        else:
            # Fallback to uniform
            uniform = Decimal("1") / Decimal(str(len(self.REGIMES)))
            updated_probs = {regime: uniform for regime in self.REGIMES}

        self.state_probs = updated_probs
        return updated_probs.copy()

    def get_most_likely_regime(self) -> Tuple[str, Decimal]:
        """
        Get the most likely regime and its probability.

        Returns:
            Tuple of (regime_name, probability)
        """
        max_regime = max(self.state_probs, key=self.state_probs.get)
        return (max_regime, self.state_probs[max_regime])

    def get_state_probabilities(self) -> Dict[str, Decimal]:
        """Get current state probability distribution."""
        return self.state_probs.copy()

    def reset(self) -> None:
        """Reset to uniform probabilities."""
        uniform = Decimal("1") / Decimal(str(len(self.REGIMES)))
        self.state_probs = {regime: uniform for regime in self.REGIMES}


class EnsembleRegimeClassifier:
    """
    Ensemble classifier combining multiple regime classification methods.

    Combines score-based classification with Kalman-filtered VIX
    and HMM-based probability updates.
    """

    # Default weights for ensemble components
    DEFAULT_WEIGHTS = {
        "score_based": Decimal("0.40"),
        "vix_hmm": Decimal("0.25"),
        "credit_hmm": Decimal("0.20"),
        "yield_hmm": Decimal("0.15"),
    }

    def __init__(
        self,
        weights: Optional[Dict[str, Decimal]] = None,
    ):
        """
        Initialize ensemble classifier.

        Args:
            weights: Custom weights for ensemble components
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.kalman_filter = VIXKalmanFilter()
        self.regime_hmm = RegimeHMM()

    def classify(
        self,
        score_result: Dict[str, Decimal],
        vix_current: Optional[Decimal] = None,
        hy_spread: Optional[Decimal] = None,
        yield_slope: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        """
        Classify regime using ensemble of methods.

        Args:
            score_result: Raw regime scores from _calculate_regime_scores
            vix_current: Current VIX level
            hy_spread: Current HY credit spread
            yield_slope: Current yield curve slope

        Returns:
            Dict with ensemble classification results
        """
        # Get smoothed VIX if available
        vix_smoothed = None
        if vix_current is not None:
            vix_smoothed = self.kalman_filter.update(vix_current)

        # Normalize score_result to probabilities
        total_score = sum(score_result.values())
        if total_score > Decimal("0"):
            score_probs = {
                k: v / total_score for k, v in score_result.items()
            }
        else:
            score_probs = {k: Decimal("0") for k in score_result}

        # Update HMM with score-based likelihoods
        hmm_probs = self.regime_hmm.update(score_probs)

        # Combine methods with weights
        ensemble_probs: Dict[str, Decimal] = {}
        for regime in RegimeHMM.REGIMES:
            score_prob = score_probs.get(regime, Decimal("0"))
            hmm_prob = hmm_probs.get(regime, Decimal("0"))

            # Weighted combination
            ensemble_probs[regime] = (
                self.weights["score_based"] * score_prob +
                (Decimal("1") - self.weights["score_based"]) * hmm_prob
            )

        # Normalize ensemble probabilities
        total_ensemble = sum(ensemble_probs.values())
        if total_ensemble > Decimal("0"):
            for regime in ensemble_probs:
                ensemble_probs[regime] = (ensemble_probs[regime] / total_ensemble).quantize(
                    Decimal("0.0001"), rounding=ROUND_HALF_UP
                )

        # Determine final regime
        final_regime = max(ensemble_probs, key=ensemble_probs.get)
        final_confidence = ensemble_probs[final_regime]

        return {
            "regime": final_regime,
            "confidence": final_confidence,
            "ensemble_probabilities": {k: str(v) for k, v in ensemble_probs.items()},
            "score_probabilities": {k: str(v) for k, v in score_probs.items()},
            "hmm_probabilities": {k: str(v) for k, v in hmm_probs.items()},
            "vix_smoothed": str(vix_smoothed) if vix_smoothed else None,
            "method": "ensemble",
        }

    def reset(self) -> None:
        """Reset all components."""
        self.kalman_filter.reset()
        self.regime_hmm.reset()


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

    # Yield curve thresholds (10Y-2Y spread in basis points)
    YIELD_CURVE_INVERTED = Decimal("-10")        # bps - mildly inverted
    YIELD_CURVE_DEEPLY_INVERTED = Decimal("-50") # bps - deeply inverted

    # High yield credit spread thresholds (OAS in basis points)
    HY_SPREAD_ELEVATED = Decimal("400")   # bps OAS - elevated stress
    HY_SPREAD_CRISIS = Decimal("600")     # bps OAS - crisis levels

    # Sector dislocation threshold (biotech divergence from market)
    SECTOR_DISLOCATION_THRESHOLD = Decimal("15")  # % divergence

    # Fund flow thresholds (weekly ETF flows in $MM)
    FUND_FLOW_STRONG_INFLOWS = Decimal("200")
    FUND_FLOW_MODERATE_INFLOWS = Decimal("50")
    FUND_FLOW_MODERATE_OUTFLOWS = Decimal("-50")
    FUND_FLOW_HEAVY_OUTFLOWS = Decimal("-200")

    # Data staleness thresholds and confidence haircuts
    # Regime is time-sensitive; stale data should not drive full weight tilts
    STALENESS_THRESHOLDS: Dict[int, Decimal] = {
        2: Decimal("1.00"),   # ≤2 days: full confidence
        5: Decimal("0.85"),   # 3-5 days: 15% haircut
        10: Decimal("0.65"),  # 6-10 days: 35% haircut
    }
    STALENESS_MAX_DAYS = 10  # >10 days: force UNKNOWN regime

    # Minimum confidence threshold for regime classification
    # Below this, default to SECTOR_ROTATION for safety
    MIN_CONFIDENCE_THRESHOLD = Decimal("0.30")

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
        },
        "RECESSION_RISK": {
            "momentum": Decimal("0.60"),      # Strongly reduce momentum
            "fundamental": Decimal("1.10"),   # Boost fundamentals
            "quality": Decimal("1.35"),       # Strong quality premium
            "catalyst": Decimal("0.85"),      # Reduce speculation
            "institutional": Decimal("1.10"), # Institutional stability
            "clinical": Decimal("0.95"),      # Slight clinical reduction
            "financial": Decimal("1.35")      # Maximum runway focus
        },
        "CREDIT_CRISIS": {
            "momentum": Decimal("0.50"),      # Severely reduce momentum
            "fundamental": Decimal("1.15"),   # Boost fundamentals
            "quality": Decimal("1.40"),       # Maximum quality premium
            "catalyst": Decimal("0.70"),      # Strongly reduce speculation
            "institutional": Decimal("1.15"), # Strong institutional premium
            "clinical": Decimal("0.85"),      # Reduce clinical speculation
            "financial": Decimal("1.45")      # Critical runway focus
        },
        "SECTOR_DISLOCATION": {
            "momentum": Decimal("0.85"),      # Moderately reduce momentum
            "fundamental": Decimal("1.05"),   # Slight fundamental boost
            "quality": Decimal("1.10"),       # Quality premium
            "catalyst": Decimal("1.05"),      # Neutral catalyst
            "institutional": Decimal("1.20"), # Strong institutional premium
            "clinical": Decimal("1.00"),      # Neutral clinical
            "financial": Decimal("1.05")      # Slight financial focus
        }
    }

    # Regime descriptions for reporting
    REGIME_DESCRIPTIONS: Dict[str, str] = {
        "BULL": "Risk-on environment favorable for growth/momentum strategies",
        "BEAR": "Risk-off environment requiring defensive positioning",
        "VOLATILITY_SPIKE": "High uncertainty requiring maximum quality focus",
        "SECTOR_ROTATION": "Mixed signals, balanced approach recommended",
        "RECESSION_RISK": "Yield curve inversion signals recession risk - maximum defensive posture",
        "CREDIT_CRISIS": "Extreme credit stress - focus on balance sheet strength and liquidity",
        "SECTOR_DISLOCATION": "Biotech sector divergence from broader market - institutional focus",
        "UNKNOWN": "Insufficient data for regime classification"
    }

    @cached_property
    def _precomputed_thresholds(self) -> Dict[str, Dict[str, Decimal]]:
        """
        Pre-compute all thresholds for faster classification.

        Returns:
            Dict mapping threshold category to threshold values
        """
        return {
            "vix": {
                "very_low": self.VIX_LOW,
                "low": self.VIX_LOW,
                "normal": self.VIX_NORMAL,
                "elevated": self.VIX_ELEVATED,
                "high": self.VIX_HIGH,
                "extreme": self.VIX_EXTREME,
            },
            "xbi": {
                "strong_outperform": self.XBI_OUTPERFORM_STRONG,
                "moderate_outperform": self.XBI_OUTPERFORM_MODERATE,
                "moderate_underperform": self.XBI_UNDERPERFORM_MODERATE,
                "strong_underperform": self.XBI_UNDERPERFORM_STRONG,
            },
            "rate": {
                "hike_aggressive": self.RATE_HIKE_AGGRESSIVE,
                "hike_moderate": self.RATE_HIKE_MODERATE,
                "cut_moderate": self.RATE_CUT_MODERATE,
                "cut_aggressive": self.RATE_CUT_AGGRESSIVE,
            },
            "yield_curve": {
                "deeply_inverted": self.YIELD_CURVE_DEEPLY_INVERTED,
                "inverted": self.YIELD_CURVE_INVERTED,
            },
            "credit": {
                "crisis": self.HY_SPREAD_CRISIS,
                "elevated": self.HY_SPREAD_ELEVATED,
            },
        }

    @cached_property
    def _regime_score_weights(self) -> Dict[str, Decimal]:
        """
        Pre-compute score weights for regime determination.

        Returns:
            Dict of weights for each score component
        """
        return {
            "vix_extreme_vol_spike": Decimal("40"),
            "vix_extreme_bear": Decimal("20"),
            "vix_high_vol_spike": Decimal("30"),
            "vix_high_bear": Decimal("15"),
            "vix_elevated_bear": Decimal("15"),
            "vix_elevated_rotation": Decimal("10"),
            "vix_normal_rotation": Decimal("15"),
            "vix_low_bull": Decimal("25"),
            "vix_neutral_rotation": Decimal("10"),
            "xbi_strong_out_bull": Decimal("30"),
            "xbi_mod_out_bull": Decimal("20"),
            "xbi_strong_under_bear": Decimal("30"),
            "xbi_mod_under_bear": Decimal("20"),
            "xbi_neutral_rotation": Decimal("15"),
        }

    def __init__(self):
        """Initialize the regime detection engine."""
        self.current_regime: Optional[str] = None
        self.regime_history: List[Dict[str, Any]] = []
        self.audit_trail: List[Dict[str, Any]] = []
        self.momentum_monitor: Optional[Any] = None

        # Callback system for regime transitions
        self._callbacks: List[RegimeTransitionCallback] = []
        self._transition_history: List[RegimeTransitionEvent] = []
        self._regime_start_date: Optional[date] = None

        # Ensemble classifier for advanced regime detection
        self._ensemble_classifier = EnsembleRegimeClassifier()

        # Kalman filter for VIX smoothing (standalone)
        self._vix_kalman = VIXKalmanFilter()

        # Initialize momentum health monitor if available
        if HAS_MOMENTUM_MONITOR:
            self.momentum_monitor = MomentumHealthMonitor()

    def add_callback(self, callback: RegimeTransitionCallback) -> None:
        """
        Register a callback to be notified on regime transitions.

        Args:
            callback: Object implementing RegimeTransitionCallback protocol
        """
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def remove_callback(self, callback: RegimeTransitionCallback) -> bool:
        """
        Remove a callback from the registry.

        Args:
            callback: The callback to remove

        Returns:
            True if callback was found and removed, False otherwise
        """
        try:
            self._callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def _notify_transition(
        self,
        old_regime: str,
        new_regime: str,
        transition_date: date,
        days_in_prior_regime: int,
        trigger_values: Dict[str, str],
        confidence: Decimal,
    ) -> None:
        """
        Notify all registered callbacks of a regime transition.

        Args:
            old_regime: The previous regime
            new_regime: The new regime
            transition_date: Date of the transition
            days_in_prior_regime: Number of days in the prior regime
            trigger_values: Dict of values that triggered the transition
            confidence: Confidence level of the new regime
        """
        # Record the transition event
        event = RegimeTransitionEvent(
            old_regime=old_regime,
            new_regime=new_regime,
            transition_date=transition_date,
            days_in_prior_regime=days_in_prior_regime,
            trigger_values=trigger_values,
            confidence=confidence,
        )
        self._transition_history.append(event)

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback.on_transition(
                    old_regime=old_regime,
                    new_regime=new_regime,
                    transition_date=transition_date,
                    days_in_prior_regime=days_in_prior_regime,
                    trigger_values=trigger_values,
                )
            except Exception:
                # Don't let callback errors break regime detection
                pass

    def _compute_regime_duration_metrics(self) -> Dict[str, Any]:
        """
        Compute metrics about regime durations and transitions.

        Returns:
            Dict with avg duration, transition counts, regime counts
        """
        if not self._transition_history:
            return {
                "total_transitions": 0,
                "avg_duration_days": None,
                "regime_counts": {},
                "recent_transitions": [],
            }

        # Count transitions by regime pair
        regime_counts: Dict[str, int] = {}
        total_days = 0
        for event in self._transition_history:
            regime_counts[event.new_regime] = regime_counts.get(event.new_regime, 0) + 1
            total_days += event.days_in_prior_regime

        avg_duration = Decimal(str(total_days / len(self._transition_history))).quantize(
            Decimal("0.1"), rounding=ROUND_HALF_UP
        )

        # Get recent transitions (last 5)
        recent = [e.to_dict() for e in self._transition_history[-5:]]

        return {
            "total_transitions": len(self._transition_history),
            "avg_duration_days": str(avg_duration),
            "regime_counts": regime_counts,
            "recent_transitions": recent,
        }

    def get_transition_history(self) -> List[RegimeTransitionEvent]:
        """Return the list of regime transition events."""
        return self._transition_history.copy()

    @staticmethod
    @lru_cache(maxsize=128)
    def _compute_staleness_haircut_cached(age_days: int, max_days: int = 10) -> Tuple[str, bool]:
        """
        Cached computation of staleness haircut based on age.

        Args:
            age_days: Number of days old the data is
            max_days: Maximum allowed staleness

        Returns:
            Tuple of (haircut_multiplier_str, is_stale)
        """
        if age_days > max_days:
            return ("0.00", True)

        # Thresholds: 2 days = 1.00, 5 days = 0.85, 10 days = 0.65
        if age_days <= 2:
            return ("1.00", False)
        elif age_days <= 5:
            return ("0.85", False)
        elif age_days <= 10:
            return ("0.65", False)
        else:
            return ("0.65", False)

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

        # Use cached computation
        haircut_str, is_stale = self._compute_staleness_haircut_cached(
            age_days, self.STALENESS_MAX_DAYS
        )
        haircut = Decimal(haircut_str)

        if is_stale:
            return (age_days, Decimal("0.00"), True)

        # Find applicable haircut using instance thresholds for flexibility
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
        data_as_of_date: Optional[date] = None,        # Date of market snapshot data
        # New signals (all optional for backward compatibility)
        yield_curve_slope: Optional[Decimal] = None,   # 10Y-2Y spread in bps
        hy_credit_spread: Optional[Decimal] = None,    # HY OAS in bps
        biotech_fund_flows: Optional[Decimal] = None,  # Weekly ETF flows in $MM
        use_ensemble: bool = False                     # Enable ensemble classification
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
            data_as_of_date: Date of market snapshot data for staleness gating
            yield_curve_slope: 10Y-2Y treasury spread in basis points
            hy_credit_spread: High yield credit spread (OAS) in basis points
            biotech_fund_flows: Weekly biotech ETF flows in $MM
            use_ensemble: Enable ensemble classification (Kalman + HMM)

        Returns:
            Dict containing:
            - regime: str (BULL, BEAR, VOLATILITY_SPIKE, SECTOR_ROTATION,
                          RECESSION_RISK, CREDIT_CRISIS, SECTOR_DISLOCATION)
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
            credit_spread_change=credit_spread_change,
            yield_curve_slope=yield_curve_slope,
            hy_credit_spread=hy_credit_spread,
            biotech_fund_flows=biotech_fund_flows
        )

        # Determine primary regime and confidence
        ensemble_result = None
        if use_ensemble:
            # Use ensemble classification
            ensemble_result = self._ensemble_classifier.classify(
                score_result=regime_scores,
                vix_current=vix_current,
                hy_spread=hy_credit_spread,
                yield_slope=yield_curve_slope,
            )
            regime = ensemble_result["regime"]
            confidence = ensemble_result["confidence"]
        else:
            # Use standard score-based classification
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
            "fed_rate_change_3m_value": str(fed_rate_change_3m) if fed_rate_change_3m else None,
            # New signal classifications
            "yield_curve_state": self._classify_yield_curve(yield_curve_slope),
            "credit_environment": self._classify_credit_environment(hy_credit_spread),
            "fund_flow_state": self._classify_fund_flows(biotech_fund_flows),
            # Raw values for new signals
            "yield_curve_slope_value": str(yield_curve_slope) if yield_curve_slope else None,
            "hy_credit_spread_value": str(hy_credit_spread) if hy_credit_spread else None,
            "biotech_fund_flows_value": str(biotech_fund_flows) if biotech_fund_flows else None
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
                "data_as_of_date": data_as_of_date.isoformat() if data_as_of_date else None,
                # New signals
                "yield_curve_slope": str(yield_curve_slope) if yield_curve_slope else None,
                "hy_credit_spread": str(hy_credit_spread) if hy_credit_spread else None,
                "biotech_fund_flows": str(biotech_fund_flows) if biotech_fund_flows else None,
                "use_ensemble": use_ensemble
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
            "ensemble": ensemble_result,
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
        credit_spread_change: Optional[Decimal],
        yield_curve_slope: Optional[Decimal] = None,
        hy_credit_spread: Optional[Decimal] = None,
        biotech_fund_flows: Optional[Decimal] = None
    ) -> Dict[str, Decimal]:
        """Calculate classification scores for each regime."""

        scores = {
            "BULL": Decimal("0"),
            "BEAR": Decimal("0"),
            "VOLATILITY_SPIKE": Decimal("0"),
            "SECTOR_ROTATION": Decimal("0"),
            "RECESSION_RISK": Decimal("0"),
            "CREDIT_CRISIS": Decimal("0"),
            "SECTOR_DISLOCATION": Decimal("0")
        }

        # VIX contribution - VOLATILITY_SPIKE gets priority at extreme levels
        if vix_current >= self.VIX_EXTREME:
            scores["VOLATILITY_SPIKE"] += Decimal("60")  # Increased from 40
            scores["BEAR"] += Decimal("10")              # Reduced from 20
        elif vix_current >= self.VIX_HIGH:
            scores["VOLATILITY_SPIKE"] += Decimal("45")  # Increased from 30
            scores["BEAR"] += Decimal("10")              # Reduced from 15
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

        # Yield curve contribution -> RECESSION_RISK gets priority when inverted
        if yield_curve_slope is not None:
            if yield_curve_slope <= self.YIELD_CURVE_DEEPLY_INVERTED:
                scores["RECESSION_RISK"] += Decimal("55")  # Increased from 40
                scores["BEAR"] += Decimal("10")            # Reduced from 15
            elif yield_curve_slope <= self.YIELD_CURVE_INVERTED:
                scores["RECESSION_RISK"] += Decimal("35")  # Increased from 25
                scores["BEAR"] += Decimal("5")             # Reduced from 10
            elif yield_curve_slope >= Decimal("100"):  # Steep curve (>100bps)
                scores["BULL"] += Decimal("10")

        # High yield credit spread contribution -> CREDIT_CRISIS gets priority at crisis levels
        if hy_credit_spread is not None:
            if hy_credit_spread >= self.HY_SPREAD_CRISIS:
                scores["CREDIT_CRISIS"] += Decimal("85")  # Increased further for clear signal
                scores["BEAR"] += Decimal("5")            # Minimal BEAR contribution
                scores["VOLATILITY_SPIKE"] += Decimal("10")
            elif hy_credit_spread >= self.HY_SPREAD_ELEVATED:
                scores["CREDIT_CRISIS"] += Decimal("40")  # Increased from 35
                scores["BEAR"] += Decimal("8")
                scores["RECESSION_RISK"] += Decimal("10")
            elif hy_credit_spread <= Decimal("250"):  # Tight spreads
                scores["BULL"] += Decimal("10")

            # Combined RECESSION_RISK signal: inverted curve + widening credit
            if yield_curve_slope is not None:
                if yield_curve_slope <= self.YIELD_CURVE_INVERTED and hy_credit_spread >= self.HY_SPREAD_ELEVATED:
                    scores["RECESSION_RISK"] += Decimal("25")  # Increased from 20

        # Sector dislocation contribution - priority when extreme divergence
        if xbi_vs_spy_30d is not None:
            abs_divergence = abs(xbi_vs_spy_30d)
            if abs_divergence >= self.SECTOR_DISLOCATION_THRESHOLD:
                scores["SECTOR_DISLOCATION"] += Decimal("55")  # Increased from 35
                scores["SECTOR_ROTATION"] += Decimal("10")     # Reduced from 15
            elif abs_divergence >= Decimal("10"):
                scores["SECTOR_DISLOCATION"] += Decimal("25")  # Increased from 15

        # Fund flow contribution
        if biotech_fund_flows is not None:
            if biotech_fund_flows >= self.FUND_FLOW_STRONG_INFLOWS:
                scores["BULL"] += Decimal("15")
            elif biotech_fund_flows >= self.FUND_FLOW_MODERATE_INFLOWS:
                scores["BULL"] += Decimal("8")
            elif biotech_fund_flows <= self.FUND_FLOW_HEAVY_OUTFLOWS:
                scores["BEAR"] += Decimal("10")            # Reduced from 15
                scores["SECTOR_DISLOCATION"] += Decimal("20")  # Increased from 10
            elif biotech_fund_flows <= self.FUND_FLOW_MODERATE_OUTFLOWS:
                scores["BEAR"] += Decimal("5")             # Reduced from 8

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
        if confidence < self.MIN_CONFIDENCE_THRESHOLD:
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

    def _classify_yield_curve(self, yield_curve_slope: Optional[Decimal]) -> str:
        """Classify yield curve state based on 10Y-2Y spread."""
        if yield_curve_slope is None:
            return "UNKNOWN"
        if yield_curve_slope <= self.YIELD_CURVE_DEEPLY_INVERTED:
            return "DEEPLY_INVERTED"
        elif yield_curve_slope <= self.YIELD_CURVE_INVERTED:
            return "INVERTED"
        elif yield_curve_slope <= Decimal("25"):
            return "FLAT"
        elif yield_curve_slope <= Decimal("100"):
            return "NORMAL"
        else:
            return "STEEP"

    def _classify_credit_environment(self, hy_credit_spread: Optional[Decimal]) -> str:
        """Classify credit environment based on HY OAS spread."""
        if hy_credit_spread is None:
            return "UNKNOWN"
        if hy_credit_spread >= self.HY_SPREAD_CRISIS:
            return "CRISIS"
        elif hy_credit_spread >= self.HY_SPREAD_ELEVATED:
            return "STRESSED"
        elif hy_credit_spread >= Decimal("300"):
            return "ELEVATED"
        else:
            return "NORMAL"

    def _classify_fund_flows(self, biotech_fund_flows: Optional[Decimal]) -> str:
        """Classify biotech fund flow state."""
        if biotech_fund_flows is None:
            return "UNKNOWN"
        if biotech_fund_flows >= self.FUND_FLOW_STRONG_INFLOWS:
            return "STRONG_INFLOWS"
        elif biotech_fund_flows >= self.FUND_FLOW_MODERATE_INFLOWS:
            return "MODERATE_INFLOWS"
        elif biotech_fund_flows <= self.FUND_FLOW_HEAVY_OUTFLOWS:
            return "HEAVY_OUTFLOWS"
        elif biotech_fund_flows <= self.FUND_FLOW_MODERATE_OUTFLOWS:
            return "MODERATE_OUTFLOWS"
        else:
            return "NEUTRAL"

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
        elif regime == "RECESSION_RISK":
            flags.append("RECESSION_WARNING")
            flags.append("DEFENSIVE_POSITIONING")
            flags.append("QUALITY_FOCUS_REQUIRED")
        elif regime == "CREDIT_CRISIS":
            flags.append("CREDIT_CRISIS_WARNING")
            flags.append("LIQUIDITY_FOCUS_REQUIRED")
            flags.append("REDUCE_POSITION_SIZE")
        elif regime == "SECTOR_DISLOCATION":
            flags.append("SECTOR_DISLOCATION_WARNING")
            flags.append("INSTITUTIONAL_FOCUS")

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
        self._transition_history = []
        self._regime_start_date = None
        # Note: callbacks are not cleared - they're considered configuration

        # Reset signal processing components
        self._vix_kalman.reset()
        self._ensemble_classifier.reset()

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
