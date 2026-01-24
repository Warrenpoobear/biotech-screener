#!/usr/bin/env python3
"""
forward_looking_separation.py - Separate Historical vs Forward-Looking Signals

P0 Enhancement: Prevents lookahead bias by explicitly separating:
1. Detected events (historical, PIT-safe)
2. Calendar events (forward-looking, requires disclosure)

When forward-looking signals contribute >50% of catalyst score,
a LOOKAHEAD_RISK warning is raised for pipeline review.

Design Philosophy:
- DETERMINISTIC: Explicit classification rules
- AUDITABLE: Full provenance of signal source
- FAIL-LOUD: Warnings logged when forward-looking dominates

Author: Wake Robin Capital Management
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import logging

__version__ = "1.0.0"

logger = logging.getLogger(__name__)


class SignalSourceType(str, Enum):
    """Classification of signal source for PIT safety."""
    HISTORICAL = "HISTORICAL"           # Detected from past events (PIT-safe)
    FORWARD_LOOKING = "FORWARD_LOOKING" # Calendar/predicted dates (not PIT-safe for backtesting)
    MIXED = "MIXED"                     # Contains both types


class LookaheadRiskLevel(str, Enum):
    """Risk level for lookahead bias contamination."""
    NONE = "NONE"       # <20% forward-looking
    LOW = "LOW"         # 20-35% forward-looking
    MEDIUM = "MEDIUM"   # 35-50% forward-looking
    HIGH = "HIGH"       # >50% forward-looking (WARNING)


@dataclass
class SignalContribution:
    """Track contribution of historical vs forward-looking signals."""
    ticker: str
    historical_score: Decimal
    forward_looking_score: Decimal
    total_score: Decimal
    historical_event_count: int
    forward_looking_event_count: int
    forward_looking_weight: Decimal  # 0.0 - 1.0
    lookahead_risk: LookaheadRiskLevel
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "historical_score": str(self.historical_score),
            "forward_looking_score": str(self.forward_looking_score),
            "total_score": str(self.total_score),
            "historical_event_count": self.historical_event_count,
            "forward_looking_event_count": self.forward_looking_event_count,
            "forward_looking_weight": str(self.forward_looking_weight),
            "lookahead_risk": self.lookahead_risk.value,
            "warnings": self.warnings,
        }


@dataclass
class SeparatedCatalystResult:
    """
    Catalyst results with explicit historical/forward-looking separation.

    This is the P0 enhancement that prevents calendar catalysts
    from contaminating PIT-safe backtesting.
    """
    ticker: str
    as_of_date: str

    # Separated scores (use historical_score for backtesting)
    historical_score: Decimal          # PIT-safe score from detected events
    forward_looking_score: Decimal     # Score from calendar events (NOT PIT-safe)
    blended_score: Decimal             # Combined score (for production, not backtest)

    # Separated event lists
    historical_events: List[Dict[str, Any]]
    forward_looking_events: List[Dict[str, Any]]

    # Risk assessment
    signal_contribution: SignalContribution

    # Configuration used
    forward_looking_weight: Decimal    # Weight applied to forward-looking in blend

    def get_backtest_safe_score(self) -> Decimal:
        """Return only the PIT-safe historical score for backtesting."""
        return self.historical_score

    def get_production_score(self) -> Decimal:
        """Return blended score for production use."""
        return self.blended_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "as_of_date": self.as_of_date,
            "scores": {
                "historical_score": str(self.historical_score),
                "forward_looking_score": str(self.forward_looking_score),
                "blended_score": str(self.blended_score),
                "backtest_safe_score": str(self.get_backtest_safe_score()),
            },
            "event_counts": {
                "historical": len(self.historical_events),
                "forward_looking": len(self.forward_looking_events),
            },
            "signal_contribution": self.signal_contribution.to_dict(),
            "forward_looking_weight": str(self.forward_looking_weight),
        }


class ForwardLookingSeparator:
    """
    Engine for separating historical and forward-looking catalyst signals.

    This is a critical PIT-safety component that ensures backtesting
    uses only historically available information.

    Usage:
        separator = ForwardLookingSeparator()
        result = separator.separate_signals(
            ticker="ACME",
            as_of_date=date(2026, 1, 15),
            detected_events=historical_events,
            calendar_events=calendar_catalysts,
        )

        # For backtesting (PIT-safe)
        backtest_score = result.get_backtest_safe_score()

        # For production
        production_score = result.get_production_score()
    """

    VERSION = "1.0.0"

    # Weight applied to forward-looking signals in production blending
    DEFAULT_FORWARD_LOOKING_WEIGHT = Decimal("0.5")  # 50% weight reduction

    # Thresholds for lookahead risk classification
    RISK_THRESHOLDS = {
        LookaheadRiskLevel.NONE: Decimal("0.20"),
        LookaheadRiskLevel.LOW: Decimal("0.35"),
        LookaheadRiskLevel.MEDIUM: Decimal("0.50"),
        LookaheadRiskLevel.HIGH: Decimal("1.00"),
    }

    def __init__(
        self,
        forward_looking_weight: Decimal = DEFAULT_FORWARD_LOOKING_WEIGHT,
        warn_on_high_forward_looking: bool = True,
        calendar_event_cap: Decimal = Decimal("15.0"),  # Max contribution from calendar events
    ):
        """
        Initialize separator.

        Args:
            forward_looking_weight: Weight for forward-looking signals in blend (0.0-1.0)
            warn_on_high_forward_looking: Log warning when >50% forward-looking
            calendar_event_cap: Maximum score contribution from calendar events
        """
        self.forward_looking_weight = forward_looking_weight
        self.warn_on_high_forward_looking = warn_on_high_forward_looking
        self.calendar_event_cap = calendar_event_cap

    def classify_lookahead_risk(
        self,
        forward_looking_weight: Decimal,
    ) -> LookaheadRiskLevel:
        """Classify lookahead risk based on forward-looking weight."""
        if forward_looking_weight <= self.RISK_THRESHOLDS[LookaheadRiskLevel.NONE]:
            return LookaheadRiskLevel.NONE
        elif forward_looking_weight <= self.RISK_THRESHOLDS[LookaheadRiskLevel.LOW]:
            return LookaheadRiskLevel.LOW
        elif forward_looking_weight <= self.RISK_THRESHOLDS[LookaheadRiskLevel.MEDIUM]:
            return LookaheadRiskLevel.MEDIUM
        else:
            return LookaheadRiskLevel.HIGH

    def compute_event_score(
        self,
        events: List[Dict[str, Any]],
        score_field: str = "score",
    ) -> Decimal:
        """Sum scores from events."""
        total = Decimal("0")
        for event in events:
            score = event.get(score_field)
            if score is not None:
                if isinstance(score, str):
                    score = Decimal(score)
                elif isinstance(score, (int, float)):
                    score = Decimal(str(score))
                total += score
        return total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def separate_signals(
        self,
        ticker: str,
        as_of_date: date,
        detected_events: List[Dict[str, Any]],
        calendar_events: List[Dict[str, Any]],
        historical_score_field: str = "score",
        calendar_score_field: str = "confidence",  # Calendar uses confidence as weight
    ) -> SeparatedCatalystResult:
        """
        Separate historical and forward-looking catalyst signals.

        Args:
            ticker: Stock ticker
            as_of_date: Analysis date
            detected_events: Events detected from snapshot deltas (PIT-safe)
            calendar_events: Events from trial calendar dates (forward-looking)
            historical_score_field: Field name for historical event scores
            calendar_score_field: Field name for calendar event scores

        Returns:
            SeparatedCatalystResult with explicit separation
        """
        # Compute historical score (PIT-safe)
        historical_score = self.compute_event_score(detected_events, historical_score_field)

        # Compute forward-looking score with cap
        raw_forward_score = Decimal("0")
        for event in calendar_events:
            # Calendar events use confidence as a weight multiplier
            confidence = event.get(calendar_score_field, 0.5)
            if isinstance(confidence, str):
                confidence = Decimal(confidence)
            elif isinstance(confidence, (int, float)):
                confidence = Decimal(str(confidence))

            # Base score based on days until event
            days_until = event.get("days_until", 90)
            if days_until <= 30:
                base_score = Decimal("5.0")
            elif days_until <= 60:
                base_score = Decimal("3.0")
            else:
                base_score = Decimal("1.5")

            raw_forward_score += base_score * confidence

        # Apply cap to forward-looking score
        forward_looking_score = min(raw_forward_score, self.calendar_event_cap)
        forward_looking_score = forward_looking_score.quantize(Decimal("0.01"))

        # Compute total and weights
        total_unweighted = historical_score + forward_looking_score

        if total_unweighted > Decimal("0"):
            fl_weight = forward_looking_score / total_unweighted
        else:
            fl_weight = Decimal("0")

        fl_weight = fl_weight.quantize(Decimal("0.01"))

        # Compute blended score for production
        # Historical at full weight, forward-looking at reduced weight
        blended_score = (
            historical_score +
            forward_looking_score * self.forward_looking_weight
        ).quantize(Decimal("0.01"))

        # Classify risk
        risk = self.classify_lookahead_risk(fl_weight)

        # Build warnings
        warnings = []
        if risk == LookaheadRiskLevel.HIGH:
            warnings.append(
                f"LOOKAHEAD_RISK: {fl_weight:.0%} of catalyst score from forward-looking signals. "
                "Backtest results may be unreliable."
            )
            if self.warn_on_high_forward_looking:
                logger.warning(f"{ticker}: {warnings[0]}")
        elif risk == LookaheadRiskLevel.MEDIUM:
            warnings.append(
                f"Moderate forward-looking contribution ({fl_weight:.0%}). "
                "Consider using historical_score for backtesting."
            )

        # Build contribution summary
        contribution = SignalContribution(
            ticker=ticker,
            historical_score=historical_score,
            forward_looking_score=forward_looking_score,
            total_score=total_unweighted,
            historical_event_count=len(detected_events),
            forward_looking_event_count=len(calendar_events),
            forward_looking_weight=fl_weight,
            lookahead_risk=risk,
            warnings=warnings,
        )

        return SeparatedCatalystResult(
            ticker=ticker,
            as_of_date=as_of_date.isoformat(),
            historical_score=historical_score,
            forward_looking_score=forward_looking_score,
            blended_score=blended_score,
            historical_events=detected_events,
            forward_looking_events=calendar_events,
            signal_contribution=contribution,
            forward_looking_weight=self.forward_looking_weight,
        )


@dataclass
class BatchSeparationResult:
    """Results from separating signals for multiple tickers."""
    total_tickers: int
    tickers_high_risk: List[str]
    tickers_medium_risk: List[str]
    avg_forward_looking_weight: Decimal
    results: Dict[str, SeparatedCatalystResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "total_tickers": self.total_tickers,
                "high_risk_count": len(self.tickers_high_risk),
                "medium_risk_count": len(self.tickers_medium_risk),
                "avg_forward_looking_weight": str(self.avg_forward_looking_weight),
                "tickers_high_risk": sorted(self.tickers_high_risk),
                "tickers_medium_risk": sorted(self.tickers_medium_risk),
            },
            "results": {t: r.to_dict() for t, r in self.results.items()},
        }


def separate_batch_signals(
    detected_events_by_ticker: Dict[str, List[Dict[str, Any]]],
    calendar_events_by_ticker: Dict[str, List[Dict[str, Any]]],
    as_of_date: date,
    separator: Optional[ForwardLookingSeparator] = None,
) -> BatchSeparationResult:
    """
    Separate signals for multiple tickers.

    Args:
        detected_events_by_ticker: Historical events by ticker
        calendar_events_by_ticker: Calendar events by ticker
        as_of_date: Analysis date
        separator: Separator instance (uses default if None)

    Returns:
        BatchSeparationResult with all tickers
    """
    if separator is None:
        separator = ForwardLookingSeparator()

    all_tickers = set(detected_events_by_ticker.keys()) | set(calendar_events_by_ticker.keys())
    results = {}
    high_risk = []
    medium_risk = []
    total_fl_weight = Decimal("0")

    for ticker in sorted(all_tickers):
        detected = detected_events_by_ticker.get(ticker, [])
        calendar = calendar_events_by_ticker.get(ticker, [])

        result = separator.separate_signals(
            ticker=ticker,
            as_of_date=as_of_date,
            detected_events=detected,
            calendar_events=calendar,
        )
        results[ticker] = result

        if result.signal_contribution.lookahead_risk == LookaheadRiskLevel.HIGH:
            high_risk.append(ticker)
        elif result.signal_contribution.lookahead_risk == LookaheadRiskLevel.MEDIUM:
            medium_risk.append(ticker)

        total_fl_weight += result.signal_contribution.forward_looking_weight

    avg_fl_weight = (
        total_fl_weight / Decimal(len(all_tickers))
        if all_tickers else Decimal("0")
    ).quantize(Decimal("0.01"))

    # Log summary warning if many high-risk tickers
    if len(high_risk) > len(all_tickers) * 0.2:  # >20% high risk
        logger.warning(
            f"BATCH LOOKAHEAD WARNING: {len(high_risk)}/{len(all_tickers)} tickers "
            f"({len(high_risk)/len(all_tickers):.0%}) have HIGH lookahead risk. "
            "Backtest reliability may be compromised."
        )

    return BatchSeparationResult(
        total_tickers=len(all_tickers),
        tickers_high_risk=high_risk,
        tickers_medium_risk=medium_risk,
        avg_forward_looking_weight=avg_fl_weight,
        results=results,
    )


if __name__ == "__main__":
    # Demonstration
    print("=" * 70)
    print("FORWARD-LOOKING SIGNAL SEPARATOR - DEMONSTRATION")
    print("=" * 70)

    separator = ForwardLookingSeparator()

    # Simulate events
    detected = [
        {"event_type": "CT_STATUS_UPGRADE", "score": "5.0"},
        {"event_type": "CT_TIMELINE_PULLIN", "score": "3.0"},
    ]
    calendar = [
        {"event_type": "UPCOMING_PCD", "days_until": 25, "confidence": 0.85},
        {"event_type": "UPCOMING_SCD", "days_until": 45, "confidence": 0.70},
    ]

    result = separator.separate_signals(
        ticker="ACME",
        as_of_date=date(2026, 1, 15),
        detected_events=detected,
        calendar_events=calendar,
    )

    print(f"\nTicker: {result.ticker}")
    print(f"Historical Score (PIT-safe): {result.historical_score}")
    print(f"Forward-Looking Score: {result.forward_looking_score}")
    print(f"Blended Score (production): {result.blended_score}")
    print(f"Lookahead Risk: {result.signal_contribution.lookahead_risk.value}")
    print(f"Forward-Looking Weight: {result.signal_contribution.forward_looking_weight:.0%}")

    if result.signal_contribution.warnings:
        print("\nWarnings:")
        for w in result.signal_contribution.warnings:
            print(f"  - {w}")
