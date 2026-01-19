#!/usr/bin/env python3
"""
time_decay_scoring.py - Multi-Window Time-Decay Scoring for Catalyst Events

Implements time-decay weighting across multiple lookback windows to capture
both fresh events (high weight) and persistent momentum (decayed weight).

Design Philosophy:
- DETERMINISTIC: No datetime.now(), no randomness
- PIT-SAFE: All dates explicit, no wall-clock
- STDLIB-ONLY: No external dependencies

Window Configuration:
- 7d:  Fresh events get 100% weight
- 30d: Recent events get 50% weight
- 90d: Historical context at 25% weight

Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any
import logging

from module_3_schema_v2 import (
    CatalystEventV2,
    EventSeverity,
    ConfidenceLevel,
    SEVERITY_SCORE_CONTRIBUTION,
    CONFIDENCE_WEIGHTS,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# TIME-DECAY WINDOW CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class TimeDecayWindow:
    """Configuration for a single lookback window."""
    name: str
    lookback_days: int
    weight: Decimal
    description: str

    def get_lookback_date(self, as_of_date: date) -> date:
        """Calculate the lookback date for this window."""
        return as_of_date - timedelta(days=self.lookback_days)


# Default time-decay windows
DEFAULT_TIME_DECAY_WINDOWS: Tuple[TimeDecayWindow, ...] = (
    TimeDecayWindow(
        name="7d",
        lookback_days=7,
        weight=Decimal("1.00"),
        description="Fresh events - full weight"
    ),
    TimeDecayWindow(
        name="30d",
        lookback_days=30,
        weight=Decimal("0.50"),
        description="Recent events - 50% weight"
    ),
    TimeDecayWindow(
        name="90d",
        lookback_days=90,
        weight=Decimal("0.25"),
        description="Historical context - 25% weight"
    ),
)


@dataclass
class TimeDecayConfig:
    """Configuration for time-decay scoring."""
    windows: Tuple[TimeDecayWindow, ...] = DEFAULT_TIME_DECAY_WINDOWS
    use_max_across_windows: bool = True  # Take max vs sum
    enable_cluster_bonus: bool = True    # Bonus for events across multiple windows
    cluster_bonus_factor: Decimal = Decimal("1.15")  # 15% bonus for cluster

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TimeDecayConfig':
        """Create config from dictionary."""
        config = cls()
        if 'use_max_across_windows' in config_dict:
            config.use_max_across_windows = config_dict['use_max_across_windows']
        if 'enable_cluster_bonus' in config_dict:
            config.enable_cluster_bonus = config_dict['enable_cluster_bonus']
        if 'cluster_bonus_factor' in config_dict:
            config.cluster_bonus_factor = Decimal(str(config_dict['cluster_bonus_factor']))

        # Custom windows
        if 'windows' in config_dict:
            custom_windows = []
            for w in config_dict['windows']:
                custom_windows.append(TimeDecayWindow(
                    name=w['name'],
                    lookback_days=w['lookback_days'],
                    weight=Decimal(str(w['weight'])),
                    description=w.get('description', '')
                ))
            config.windows = tuple(custom_windows)

        return config


# =============================================================================
# WINDOW SCORE RESULT
# =============================================================================

@dataclass
class WindowScoreResult:
    """Score result for a single time window."""
    window_name: str
    lookback_days: int
    window_weight: Decimal
    events_in_window: int
    raw_score: Decimal
    weighted_score: Decimal
    event_ids: List[str] = field(default_factory=list)
    dominant_severity: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_name": self.window_name,
            "lookback_days": self.lookback_days,
            "window_weight": str(self.window_weight),
            "events_in_window": self.events_in_window,
            "raw_score": str(self.raw_score),
            "weighted_score": str(self.weighted_score),
            "event_ids": self.event_ids,
            "dominant_severity": self.dominant_severity,
        }


@dataclass
class TimeDecayScoreResult:
    """Complete time-decay scoring result for a ticker."""
    ticker: str
    as_of_date: str
    final_score: Decimal
    contributing_window: str  # Which window contributed the final score
    window_scores: List[WindowScoreResult] = field(default_factory=list)
    cluster_detected: bool = False
    cluster_bonus_applied: bool = False
    unique_events_total: int = 0
    windows_with_events: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "as_of_date": self.as_of_date,
            "final_score": str(self.final_score),
            "contributing_window": self.contributing_window,
            "window_scores": [w.to_dict() for w in self.window_scores],
            "cluster_detected": self.cluster_detected,
            "cluster_bonus_applied": self.cluster_bonus_applied,
            "unique_events_total": self.unique_events_total,
            "windows_with_events": self.windows_with_events,
        }


# =============================================================================
# TIME-DECAY SCORING ENGINE
# =============================================================================

def compute_event_score(
    event: CatalystEventV2,
    as_of_date: date,
) -> Decimal:
    """
    Compute individual event contribution score.

    Uses severity contribution weighted by confidence and recency.
    """
    # Base severity contribution
    severity_contrib = SEVERITY_SCORE_CONTRIBUTION.get(
        event.event_severity, Decimal("0.0")
    )

    # Confidence weight
    confidence_weight = CONFIDENCE_WEIGHTS.get(
        event.confidence, Decimal("0.5")
    )

    # Certainty score (includes source reliability, date specificity, staleness)
    certainty = event.compute_certainty_score(as_of_date)

    # Combined score
    score = severity_contrib * confidence_weight * certainty

    return score.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def filter_events_for_window(
    events: List[CatalystEventV2],
    window: TimeDecayWindow,
    as_of_date: date,
) -> List[CatalystEventV2]:
    """
    Filter events that fall within the specified time window.

    An event is in-window if:
    - event.disclosed_at >= window_start_date
    - event.disclosed_at <= as_of_date (PIT safety)
    """
    window_start = window.get_lookback_date(as_of_date)
    in_window = []

    for event in events:
        if not event.disclosed_at:
            continue

        try:
            disclosed_date = date.fromisoformat(event.disclosed_at)
        except (ValueError, TypeError):
            continue

        # PIT check: event must be disclosed on or before as_of_date
        if disclosed_date > as_of_date:
            continue

        # Window check: event must be within the lookback window
        if disclosed_date >= window_start:
            in_window.append(event)

    return in_window


def score_window(
    events: List[CatalystEventV2],
    window: TimeDecayWindow,
    as_of_date: date,
) -> WindowScoreResult:
    """
    Score events for a single time window.
    """
    # Filter events for this window
    window_events = filter_events_for_window(events, window, as_of_date)

    if not window_events:
        return WindowScoreResult(
            window_name=window.name,
            lookback_days=window.lookback_days,
            window_weight=window.weight,
            events_in_window=0,
            raw_score=Decimal("0"),
            weighted_score=Decimal("0"),
            event_ids=[],
            dominant_severity=None,
        )

    # Sum event contributions
    raw_score = Decimal("0")
    severity_counts: Dict[EventSeverity, int] = {}

    for event in window_events:
        contrib = compute_event_score(event, as_of_date)
        raw_score += contrib

        sev = event.event_severity
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    # Find dominant severity
    dominant_severity = None
    if severity_counts:
        dominant_severity = max(severity_counts, key=severity_counts.get).value

    # Apply window weight
    weighted_score = (raw_score * window.weight).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )

    return WindowScoreResult(
        window_name=window.name,
        lookback_days=window.lookback_days,
        window_weight=window.weight,
        events_in_window=len(window_events),
        raw_score=raw_score.quantize(Decimal("0.01")),
        weighted_score=weighted_score,
        event_ids=[e.event_id_short for e in window_events],
        dominant_severity=dominant_severity,
    )


def compute_time_decay_score(
    ticker: str,
    events: List[CatalystEventV2],
    as_of_date: date,
    config: Optional[TimeDecayConfig] = None,
) -> TimeDecayScoreResult:
    """
    Compute time-decay weighted catalyst score for a ticker.

    Strategy:
    1. Score events in each time window
    2. Apply window-specific weights
    3. Take MAX across windows (or SUM based on config)
    4. Apply cluster bonus if events span multiple windows

    Args:
        ticker: Ticker symbol
        events: List of catalyst events for this ticker
        as_of_date: Point-in-time date
        config: Time-decay configuration

    Returns:
        TimeDecayScoreResult with full breakdown
    """
    if config is None:
        config = TimeDecayConfig()

    # Score each window
    window_scores: List[WindowScoreResult] = []
    for window in config.windows:
        window_result = score_window(events, window, as_of_date)
        window_scores.append(window_result)

    # Track unique events across all windows
    all_event_ids = set()
    windows_with_events = 0
    for ws in window_scores:
        all_event_ids.update(ws.event_ids)
        if ws.events_in_window > 0:
            windows_with_events += 1

    # Determine final score
    if config.use_max_across_windows:
        # Take maximum weighted score
        best_window = max(window_scores, key=lambda w: w.weighted_score)
        final_score = best_window.weighted_score
        contributing_window = best_window.window_name
    else:
        # Sum weighted scores (with deduplication penalty)
        final_score = sum(w.weighted_score for w in window_scores)
        final_score = final_score.quantize(Decimal("0.01"))
        contributing_window = "sum_all"

    # Cluster detection and bonus
    cluster_detected = windows_with_events >= 2
    cluster_bonus_applied = False

    if config.enable_cluster_bonus and cluster_detected:
        # Apply cluster bonus for sustained momentum
        final_score = (final_score * config.cluster_bonus_factor).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        cluster_bonus_applied = True

    return TimeDecayScoreResult(
        ticker=ticker,
        as_of_date=as_of_date.isoformat(),
        final_score=final_score,
        contributing_window=contributing_window,
        window_scores=window_scores,
        cluster_detected=cluster_detected,
        cluster_bonus_applied=cluster_bonus_applied,
        unique_events_total=len(all_event_ids),
        windows_with_events=windows_with_events,
    )


# =============================================================================
# BATCH SCORING
# =============================================================================

def score_all_tickers_with_time_decay(
    events_by_ticker: Dict[str, List[CatalystEventV2]],
    active_tickers: List[str],
    as_of_date: date,
    config: Optional[TimeDecayConfig] = None,
) -> Tuple[Dict[str, TimeDecayScoreResult], Dict[str, Any]]:
    """
    Score all tickers using time-decay weighting.

    Args:
        events_by_ticker: Events grouped by ticker
        active_tickers: List of active tickers to score
        as_of_date: Point-in-time date
        config: Time-decay configuration

    Returns:
        (results_by_ticker, diagnostics)
    """
    if config is None:
        config = TimeDecayConfig()

    results: Dict[str, TimeDecayScoreResult] = {}
    diagnostics = {
        "tickers_scored": 0,
        "tickers_with_cluster": 0,
        "total_events_scored": 0,
        "avg_score": Decimal("0"),
        "score_distribution": {
            "positive": 0,
            "neutral": 0,
            "negative": 0,
        },
        "window_contributions": {w.name: 0 for w in config.windows},
    }

    total_score = Decimal("0")

    for ticker in sorted(active_tickers):
        events = events_by_ticker.get(ticker, [])
        result = compute_time_decay_score(ticker, events, as_of_date, config)
        results[ticker] = result

        # Update diagnostics
        diagnostics["tickers_scored"] += 1
        diagnostics["total_events_scored"] += result.unique_events_total
        total_score += result.final_score

        if result.cluster_detected:
            diagnostics["tickers_with_cluster"] += 1

        # Score distribution
        if result.final_score > Decimal("0"):
            diagnostics["score_distribution"]["positive"] += 1
        elif result.final_score < Decimal("0"):
            diagnostics["score_distribution"]["negative"] += 1
        else:
            diagnostics["score_distribution"]["neutral"] += 1

        # Track which window contributed
        diagnostics["window_contributions"][result.contributing_window] = (
            diagnostics["window_contributions"].get(result.contributing_window, 0) + 1
        )

    # Compute average
    if diagnostics["tickers_scored"] > 0:
        diagnostics["avg_score"] = (
            total_score / Decimal(diagnostics["tickers_scored"])
        ).quantize(Decimal("0.01"))

    # Convert Decimals for JSON serialization
    diagnostics["avg_score"] = str(diagnostics["avg_score"])

    logger.info(f"Time-decay scoring complete: {diagnostics['tickers_scored']} tickers, "
                f"{diagnostics['tickers_with_cluster']} with clusters")

    return (results, diagnostics)


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

def integrate_time_decay_into_summary(
    time_decay_result: TimeDecayScoreResult,
    base_score: Decimal,
    time_decay_weight: Decimal = Decimal("0.15"),
) -> Tuple[Decimal, str]:
    """
    Integrate time-decay score into composite catalyst score.

    Formula:
        adjusted_score = base_score * (1 - time_decay_weight) + time_decay_adjustment

    Where time_decay_adjustment adds/subtracts based on recent event momentum.

    Args:
        time_decay_result: Time-decay scoring result
        base_score: Original catalyst score (0-100)
        time_decay_weight: Weight for time-decay contribution (default 15%)

    Returns:
        (adjusted_score, explanation)
    """
    # Normalize time-decay score to a modifier (-10 to +10 points)
    # Positive momentum adds to score, negative subtracts
    td_score = time_decay_result.final_score

    # Scale to -10 to +10 range
    if td_score > Decimal("0"):
        modifier = min(Decimal("10"), td_score / Decimal("5"))
    elif td_score < Decimal("0"):
        modifier = max(Decimal("-10"), td_score / Decimal("5"))
    else:
        modifier = Decimal("0")

    # Apply cluster bonus amplification
    if time_decay_result.cluster_bonus_applied:
        modifier = (modifier * Decimal("1.20")).quantize(Decimal("0.01"))

    # Blend with base score
    adjustment = modifier * time_decay_weight * Decimal("100")
    adjusted_score = base_score + adjustment

    # Clamp to 0-100
    adjusted_score = max(Decimal("0"), min(Decimal("100"), adjusted_score))
    adjusted_score = adjusted_score.quantize(Decimal("0.01"))

    # Build explanation
    if modifier > Decimal("0"):
        explanation = f"Boosted by recent catalyst momentum ({time_decay_result.contributing_window} window)"
    elif modifier < Decimal("0"):
        explanation = f"Penalized by recent negative catalysts ({time_decay_result.contributing_window} window)"
    else:
        explanation = "No time-decay adjustment"

    if time_decay_result.cluster_bonus_applied:
        explanation += " [cluster bonus]"

    return (adjusted_score, explanation)


# =============================================================================
# CLI / TESTING
# =============================================================================

if __name__ == "__main__":
    print("Time-Decay Scoring Module loaded successfully")
    print(f"Default windows: {[w.name for w in DEFAULT_TIME_DECAY_WINDOWS]}")
