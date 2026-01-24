#!/usr/bin/env python3
"""
regime_adaptive_decay.py - Regime-Adaptive Time Decay for Catalyst Signals

P3 Enhancement: Makes time decay windows adaptive to market regime,
improving signal quality in different market conditions.

Key Features:
1. Regime-based decay rate adjustments
2. Event-type specific decay modification
3. Volatility-responsive decay acceleration
4. Integration with existing time_decay_scoring.py

Design Philosophy:
- DETERMINISTIC: Explicit decay functions with no randomness
- AUDITABLE: Full provenance for decay calculations
- CONFIGURABLE: All parameters externally adjustable

Author: Wake Robin Capital Management
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import logging
import math

__version__ = "1.0.0"

logger = logging.getLogger(__name__)


class RegimeState(str, Enum):
    """Market regime states."""
    BULL = "BULL"
    BEAR = "BEAR"
    VOLATILITY_SPIKE = "VOLATILITY_SPIKE"
    SECTOR_ROTATION = "SECTOR_ROTATION"
    UNKNOWN = "UNKNOWN"


class EventCategory(str, Enum):
    """Categories of catalyst events for decay purposes."""
    CRITICAL_POSITIVE = "CRITICAL_POSITIVE"  # FDA approvals, major readouts
    POSITIVE = "POSITIVE"                     # Upgrades, timeline pulls
    NEUTRAL = "NEUTRAL"                       # Activity signals
    NEGATIVE = "NEGATIVE"                     # Downgrades, pushouts
    SEVERE_NEGATIVE = "SEVERE_NEGATIVE"       # Trial stops, failures


@dataclass(frozen=True)
class DecayParameters:
    """Parameters for exponential decay calculation."""
    half_life_days: int
    decay_floor: Decimal  # Minimum weight (never decays below this)
    decay_cap: Decimal    # Maximum weight


# Base half-lives by event category (days until signal halves)
BASE_HALF_LIVES: Dict[EventCategory, int] = {
    EventCategory.CRITICAL_POSITIVE: 60,    # Major events decay slowly
    EventCategory.POSITIVE: 30,
    EventCategory.NEUTRAL: 14,
    EventCategory.NEGATIVE: 21,             # Negative events remembered longer
    EventCategory.SEVERE_NEGATIVE: 90,      # Major negatives very sticky
}


# Regime multipliers for half-life
# >1.0 = slower decay (longer half-life), <1.0 = faster decay
REGIME_DECAY_MULTIPLIERS: Dict[RegimeState, Dict[EventCategory, Decimal]] = {
    RegimeState.BULL: {
        EventCategory.CRITICAL_POSITIVE: Decimal("1.2"),   # Positive events last longer
        EventCategory.POSITIVE: Decimal("1.3"),
        EventCategory.NEUTRAL: Decimal("1.0"),
        EventCategory.NEGATIVE: Decimal("0.7"),            # Negatives forgotten faster
        EventCategory.SEVERE_NEGATIVE: Decimal("0.8"),
    },
    RegimeState.BEAR: {
        EventCategory.CRITICAL_POSITIVE: Decimal("0.8"),   # Positive events forgotten faster
        EventCategory.POSITIVE: Decimal("0.7"),
        EventCategory.NEUTRAL: Decimal("0.8"),
        EventCategory.NEGATIVE: Decimal("1.3"),            # Negatives remembered longer
        EventCategory.SEVERE_NEGATIVE: Decimal("1.5"),
    },
    RegimeState.VOLATILITY_SPIKE: {
        EventCategory.CRITICAL_POSITIVE: Decimal("0.6"),   # Everything decays faster
        EventCategory.POSITIVE: Decimal("0.5"),
        EventCategory.NEUTRAL: Decimal("0.4"),
        EventCategory.NEGATIVE: Decimal("0.6"),
        EventCategory.SEVERE_NEGATIVE: Decimal("0.8"),
    },
    RegimeState.SECTOR_ROTATION: {
        EventCategory.CRITICAL_POSITIVE: Decimal("1.0"),
        EventCategory.POSITIVE: Decimal("0.9"),
        EventCategory.NEUTRAL: Decimal("0.8"),
        EventCategory.NEGATIVE: Decimal("1.1"),
        EventCategory.SEVERE_NEGATIVE: Decimal("1.0"),
    },
    RegimeState.UNKNOWN: {
        EventCategory.CRITICAL_POSITIVE: Decimal("1.0"),
        EventCategory.POSITIVE: Decimal("1.0"),
        EventCategory.NEUTRAL: Decimal("1.0"),
        EventCategory.NEGATIVE: Decimal("1.0"),
        EventCategory.SEVERE_NEGATIVE: Decimal("1.0"),
    },
}


@dataclass
class DecayResult:
    """Result of decay calculation."""
    event_age_days: int
    base_half_life: int
    adjusted_half_life: int
    regime: RegimeState
    event_category: EventCategory
    decay_weight: Decimal  # 0.0 - 1.0
    decay_multiplier: Decimal
    volatility_adjustment: Optional[Decimal]
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_age_days": self.event_age_days,
            "base_half_life": self.base_half_life,
            "adjusted_half_life": self.adjusted_half_life,
            "regime": self.regime.value,
            "event_category": self.event_category.value,
            "decay_weight": str(self.decay_weight),
            "decay_multiplier": str(self.decay_multiplier),
            "volatility_adjustment": str(self.volatility_adjustment) if self.volatility_adjustment else None,
            "explanation": self.explanation,
        }


class RegimeAdaptiveDecayEngine:
    """
    Engine for computing regime-adaptive signal decay.

    Adjusts time decay based on market regime, event category,
    and volatility to improve signal quality.

    Usage:
        engine = RegimeAdaptiveDecayEngine()
        result = engine.compute_decay(
            event_age_days=15,
            event_category=EventCategory.POSITIVE,
            regime=RegimeState.BULL,
            vix_level=Decimal("22.0")
        )
        weight = result.decay_weight
    """

    VERSION = "1.0.0"

    # Decay bounds
    DEFAULT_DECAY_FLOOR = Decimal("0.05")  # Never fully decay
    DEFAULT_DECAY_CAP = Decimal("1.00")

    # Volatility baseline for adjustment
    VIX_BASELINE = Decimal("20.0")
    VIX_HIGH = Decimal("30.0")
    VIX_EXTREME = Decimal("40.0")

    def __init__(
        self,
        base_half_lives: Optional[Dict[EventCategory, int]] = None,
        regime_multipliers: Optional[Dict[RegimeState, Dict[EventCategory, Decimal]]] = None,
        decay_floor: Decimal = DEFAULT_DECAY_FLOOR,
        decay_cap: Decimal = DEFAULT_DECAY_CAP,
        apply_volatility_adjustment: bool = True,
    ):
        """
        Initialize regime-adaptive decay engine.

        Args:
            base_half_lives: Base half-lives by event category
            regime_multipliers: Regime-specific multipliers
            decay_floor: Minimum decay weight
            decay_cap: Maximum decay weight
            apply_volatility_adjustment: Whether to adjust for VIX
        """
        self.base_half_lives = base_half_lives or BASE_HALF_LIVES
        self.regime_multipliers = regime_multipliers or REGIME_DECAY_MULTIPLIERS
        self.decay_floor = decay_floor
        self.decay_cap = decay_cap
        self.apply_volatility_adjustment = apply_volatility_adjustment

    def _compute_volatility_adjustment(
        self,
        vix_level: Optional[Decimal],
    ) -> Tuple[Decimal, str]:
        """
        Compute decay rate adjustment based on VIX.

        Higher VIX = faster decay (shorter effective half-life)
        """
        if vix_level is None or not self.apply_volatility_adjustment:
            return (Decimal("1.0"), "No VIX adjustment")

        # VIX ratio to baseline
        vix_ratio = vix_level / self.VIX_BASELINE

        # Convert to decay multiplier
        # VIX 20 -> 1.0x (no change)
        # VIX 30 -> 0.75x (faster decay)
        # VIX 40 -> 0.5x (much faster decay)
        if vix_level <= self.VIX_BASELINE:
            # Low vol: slower decay
            adjustment = Decimal("1.0") + (self.VIX_BASELINE - vix_level) / Decimal("40")
            adjustment = min(Decimal("1.3"), adjustment)
            reason = f"Low VIX ({vix_level}) -> slower decay"
        elif vix_level <= self.VIX_HIGH:
            # Elevated vol: faster decay
            adjustment = Decimal("1.0") - (vix_level - self.VIX_BASELINE) / Decimal("40")
            adjustment = max(Decimal("0.70"), adjustment)
            reason = f"Elevated VIX ({vix_level}) -> faster decay"
        else:
            # High vol: much faster decay
            adjustment = Decimal("0.50")
            reason = f"High VIX ({vix_level}) -> rapid decay"

        return (adjustment.quantize(Decimal("0.01")), reason)

    def compute_decay(
        self,
        event_age_days: int,
        event_category: EventCategory,
        regime: RegimeState = RegimeState.UNKNOWN,
        vix_level: Optional[Decimal] = None,
    ) -> DecayResult:
        """
        Compute regime-adaptive decay weight for an event.

        Args:
            event_age_days: Days since event occurred
            event_category: Category of the event
            regime: Current market regime
            vix_level: Current VIX level (optional)

        Returns:
            DecayResult with decay weight and explanation
        """
        # Base half-life
        base_half_life = self.base_half_lives.get(event_category, 30)

        # Regime multiplier
        regime_mults = self.regime_multipliers.get(regime, self.regime_multipliers[RegimeState.UNKNOWN])
        regime_mult = regime_mults.get(event_category, Decimal("1.0"))

        # Volatility adjustment
        vol_adjustment, vol_reason = self._compute_volatility_adjustment(vix_level)

        # Adjusted half-life
        adjusted_half_life = int(float(base_half_life) * float(regime_mult) * float(vol_adjustment))
        adjusted_half_life = max(7, adjusted_half_life)  # Minimum 7 days

        # Exponential decay: weight = 2^(-age/half_life)
        decay_exponent = -event_age_days / adjusted_half_life
        decay_weight = Decimal(str(math.pow(2, decay_exponent)))
        decay_weight = decay_weight.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

        # Apply bounds
        decay_weight = max(self.decay_floor, min(self.decay_cap, decay_weight))

        # Build explanation
        explanation_parts = [
            f"Base half-life: {base_half_life}d",
            f"Regime ({regime.value}) mult: {regime_mult}x",
        ]
        if self.apply_volatility_adjustment and vix_level:
            explanation_parts.append(f"VIX ({vix_level}) adj: {vol_adjustment}x")
        explanation_parts.append(f"Adjusted half-life: {adjusted_half_life}d")
        explanation_parts.append(f"Age {event_age_days}d -> weight {decay_weight}")

        return DecayResult(
            event_age_days=event_age_days,
            base_half_life=base_half_life,
            adjusted_half_life=adjusted_half_life,
            regime=regime,
            event_category=event_category,
            decay_weight=decay_weight,
            decay_multiplier=regime_mult,
            volatility_adjustment=vol_adjustment if self.apply_volatility_adjustment else None,
            explanation="; ".join(explanation_parts),
        )

    def compute_event_weight(
        self,
        event_date: date,
        as_of_date: date,
        event_category: EventCategory,
        base_score: Decimal,
        regime: RegimeState = RegimeState.UNKNOWN,
        vix_level: Optional[Decimal] = None,
    ) -> Tuple[Decimal, DecayResult]:
        """
        Compute decayed weight for a scored event.

        Args:
            event_date: Date of the event
            as_of_date: Analysis date
            event_category: Event category
            base_score: Raw score before decay
            regime: Market regime
            vix_level: VIX level

        Returns:
            (decayed_score, decay_result)
        """
        age_days = (as_of_date - event_date).days

        if age_days < 0:
            # Future event - no decay (shouldn't happen in PIT-safe system)
            return (base_score, DecayResult(
                event_age_days=age_days,
                base_half_life=0,
                adjusted_half_life=0,
                regime=regime,
                event_category=event_category,
                decay_weight=Decimal("1.0"),
                decay_multiplier=Decimal("1.0"),
                volatility_adjustment=None,
                explanation="Future event - no decay applied",
            ))

        decay_result = self.compute_decay(
            event_age_days=age_days,
            event_category=event_category,
            regime=regime,
            vix_level=vix_level,
        )

        decayed_score = (base_score * decay_result.decay_weight).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        return (decayed_score, decay_result)


def integrate_with_time_decay_scoring(
    events: List[Dict[str, Any]],
    as_of_date: date,
    regime: RegimeState,
    vix_level: Optional[Decimal] = None,
    engine: Optional[RegimeAdaptiveDecayEngine] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Apply regime-adaptive decay to a list of events.

    Enhances events with regime-adaptive decay weights.

    Args:
        events: List of event dicts with 'event_date', 'event_category', 'score'
        as_of_date: Analysis date
        regime: Current market regime
        vix_level: VIX level
        engine: Decay engine (uses default if None)

    Returns:
        (enhanced_events, diagnostics)
    """
    if engine is None:
        engine = RegimeAdaptiveDecayEngine()

    enhanced_events = []
    total_raw_score = Decimal("0")
    total_decayed_score = Decimal("0")

    for event in events:
        event_date_str = event.get("event_date") or event.get("disclosed_at")
        if not event_date_str:
            enhanced_events.append(event)
            continue

        try:
            event_date = date.fromisoformat(event_date_str[:10])
        except (ValueError, TypeError):
            enhanced_events.append(event)
            continue

        # Map event severity/type to category
        event_severity = event.get("event_severity", "NEUTRAL")
        if event_severity == "CRITICAL_POSITIVE":
            category = EventCategory.CRITICAL_POSITIVE
        elif event_severity == "POSITIVE":
            category = EventCategory.POSITIVE
        elif event_severity == "NEGATIVE":
            category = EventCategory.NEGATIVE
        elif event_severity == "SEVERE_NEGATIVE":
            category = EventCategory.SEVERE_NEGATIVE
        else:
            category = EventCategory.NEUTRAL

        # Get base score
        base_score = Decimal(str(event.get("score", "0")))
        total_raw_score += abs(base_score)

        # Compute decay
        decayed_score, decay_result = engine.compute_event_weight(
            event_date=event_date,
            as_of_date=as_of_date,
            event_category=category,
            base_score=base_score,
            regime=regime,
            vix_level=vix_level,
        )

        total_decayed_score += abs(decayed_score)

        # Enhance event
        enhanced_event = dict(event)
        enhanced_event["regime_decay"] = {
            "raw_score": str(base_score),
            "decayed_score": str(decayed_score),
            "decay_weight": str(decay_result.decay_weight),
            "adjusted_half_life": decay_result.adjusted_half_life,
            "regime": regime.value,
        }
        enhanced_event["score"] = str(decayed_score)
        enhanced_events.append(enhanced_event)

    # Diagnostics
    diagnostics = {
        "events_processed": len(events),
        "regime": regime.value,
        "vix_level": str(vix_level) if vix_level else None,
        "total_raw_score": str(total_raw_score),
        "total_decayed_score": str(total_decayed_score),
        "decay_ratio": str(
            (total_decayed_score / total_raw_score).quantize(Decimal("0.01"))
            if total_raw_score > 0 else Decimal("1.0")
        ),
    }

    return (enhanced_events, diagnostics)


if __name__ == "__main__":
    # Demonstration
    print("=" * 70)
    print("REGIME-ADAPTIVE DECAY ENGINE - DEMONSTRATION")
    print("=" * 70)

    engine = RegimeAdaptiveDecayEngine()

    # Test different regime/category combinations
    test_cases = [
        (RegimeState.BULL, EventCategory.POSITIVE, Decimal("22.0"), 15),
        (RegimeState.BEAR, EventCategory.POSITIVE, Decimal("22.0"), 15),
        (RegimeState.VOLATILITY_SPIKE, EventCategory.POSITIVE, Decimal("35.0"), 15),
        (RegimeState.BULL, EventCategory.NEGATIVE, Decimal("22.0"), 15),
        (RegimeState.BEAR, EventCategory.NEGATIVE, Decimal("22.0"), 15),
    ]

    print("\nDecay comparison (15-day old event):\n")
    print(f"{'Regime':<20} {'Category':<20} {'VIX':<8} {'Weight':<10} {'Half-life':<10}")
    print("-" * 70)

    for regime, category, vix, age in test_cases:
        result = engine.compute_decay(age, category, regime, vix)
        print(f"{regime.value:<20} {category.value:<20} {vix:<8} {result.decay_weight:<10} {result.adjusted_half_life}d")

    # Test decay curve
    print("\n\nDecay curve for POSITIVE event in BULL market:\n")
    print(f"{'Age (days)':<12} {'Weight':<10} {'Explanation'}")
    print("-" * 60)

    for age in [0, 7, 14, 30, 60, 90]:
        result = engine.compute_decay(age, EventCategory.POSITIVE, RegimeState.BULL, Decimal("20.0"))
        print(f"{age:<12} {result.decay_weight:<10} adj_hl={result.adjusted_half_life}d")
