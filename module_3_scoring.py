"""
Module 3 Catalyst Scoring System vNext

Converts catalyst events into numeric impact scores (0-100 scale).

Score Modes:
- score_override: Hierarchical logic (severe_neg=20, critical_pos=75, else blend)
- score_blended: Confidence-weighted, recency-decayed scoring

Determinism Guarantees:
- All computations use Decimal for stable results
- No datetime.now() calls - only uses as_of_date
- Sorting uses explicit keys
"""

import logging
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any

from module_3_schema import (
    SCHEMA_VERSION,
    SCORE_VERSION,
    EventType,
    EventSeverity,
    ConfidenceLevel,
    CatalystWindowBucket,
    CatalystEventV2,
    TickerCatalystSummaryV2,
    DiagnosticCounts,
    EVENT_SEVERITY_MAP,
    EVENT_DEFAULT_CONFIDENCE,
    CONFIDENCE_WEIGHTS,
    SEVERITY_SCORE_CONTRIBUTION,
    compute_catalyst_window_bucket,
    decimal_to_str,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# SCORE CONSTANTS
# =============================================================================

# Override scores (hierarchical logic)
SCORE_OVERRIDE_SEVERE_NEGATIVE = Decimal("20")
SCORE_OVERRIDE_CRITICAL_POSITIVE = Decimal("75")
SCORE_DEFAULT = Decimal("50")

# Blended scoring base
SCORE_BLENDED_BASE = Decimal("50")
SCORE_MIN = Decimal("0")
SCORE_MAX = Decimal("100")

# Recency decay parameters
DECAY_HALF_LIFE_DAYS = 90  # Score contribution halves every 90 days
STALENESS_THRESHOLD_DAYS = 180  # Penalty if no events newer than this
STALENESS_PENALTY_FACTOR = Decimal("0.8")


# =============================================================================
# RECENCY / DECAY CALCULATION
# =============================================================================

def compute_recency_weight(
    event_date: Optional[str],
    as_of_date: date,
) -> Decimal:
    """
    Compute recency weight using exponential decay.

    Args:
        event_date: ISO date string or None
        as_of_date: Point-in-time date

    Returns:
        Weight between 0.0 and 1.0 (Decimal)
        - 1.0 for events on as_of_date
        - 0.5 for events DECAY_HALF_LIFE_DAYS old
        - 0.5 (conservative) for events with null date
    """
    if event_date is None:
        # Conservative weight for unknown dates
        return Decimal("0.5")

    try:
        event_d = date.fromisoformat(event_date)
    except (ValueError, TypeError):
        return Decimal("0.5")

    days_old = (as_of_date - event_d).days

    if days_old < 0:
        # Future event - should not be scored (PIT violation)
        return Decimal("0")

    if days_old == 0:
        return Decimal("1.0")

    # Exponential decay: weight = 2^(-days / half_life)
    # Using approximation for Decimal arithmetic
    decay_factor = Decimal(days_old) / Decimal(DECAY_HALF_LIFE_DAYS)
    weight = Decimal("0.5") ** decay_factor

    return weight.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


def compute_staleness_factor(
    events: List[CatalystEventV2],
    as_of_date: date,
) -> Decimal:
    """
    Compute staleness penalty factor.

    If no events are newer than STALENESS_THRESHOLD_DAYS, apply penalty.

    Returns:
        1.0 if not stale, STALENESS_PENALTY_FACTOR if stale
    """
    if not events:
        return Decimal("1.0")

    newest_days = None

    for event in events:
        if event.event_date:
            try:
                event_d = date.fromisoformat(event.event_date)
                days_old = (as_of_date - event_d).days
                if days_old >= 0:  # Only consider past events
                    if newest_days is None or days_old < newest_days:
                        newest_days = days_old
            except (ValueError, TypeError):
                continue

    if newest_days is None:
        # All events have null dates - no penalty
        return Decimal("1.0")

    if newest_days > STALENESS_THRESHOLD_DAYS:
        return STALENESS_PENALTY_FACTOR

    return Decimal("1.0")


# =============================================================================
# OVERRIDE SCORING (HIERARCHICAL)
# =============================================================================

def calculate_score_override(
    events: List[CatalystEventV2],
) -> Tuple[Decimal, str]:
    """
    Calculate override score using hierarchical logic.

    Rules:
    1. If ANY severe negative -> 20
    2. If ANY critical positive -> 75
    3. Otherwise, blend based on severity counts

    Args:
        events: List of catalyst events

    Returns:
        (score, reason)
    """
    if not events:
        return (SCORE_DEFAULT, "NO_EVENTS")

    # Count severities
    severity_counts: Dict[EventSeverity, int] = {}
    for event in events:
        sev = event.event_severity
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    # Rule 1: Severe negatives override everything
    if severity_counts.get(EventSeverity.SEVERE_NEGATIVE, 0) > 0:
        return (SCORE_OVERRIDE_SEVERE_NEGATIVE, "SEVERE_NEGATIVE_EVENTS")

    # Rule 2: Critical positives (in absence of severe negatives)
    if severity_counts.get(EventSeverity.CRITICAL_POSITIVE, 0) > 0:
        return (SCORE_OVERRIDE_CRITICAL_POSITIVE, "CRITICAL_POSITIVE_EVENTS")

    # Rule 3: Blend positive and negative events
    positive_count = (
        severity_counts.get(EventSeverity.POSITIVE, 0) +
        severity_counts.get(EventSeverity.CRITICAL_POSITIVE, 0)
    )
    negative_count = (
        severity_counts.get(EventSeverity.NEGATIVE, 0) +
        severity_counts.get(EventSeverity.SEVERE_NEGATIVE, 0)
    )

    if positive_count > 0 and negative_count == 0:
        # Pure positive
        score = Decimal("60") + min(Decimal("10"), Decimal(positive_count) * Decimal("5"))
        return (score, "POSITIVE_EVENTS")

    if negative_count > 0 and positive_count == 0:
        # Pure negative
        score = Decimal("40") - min(Decimal("5"), Decimal(negative_count) * Decimal("2.5"))
        return (score, "NEGATIVE_EVENTS")

    if positive_count > 0 and negative_count > 0:
        # Mixed
        net = SCORE_DEFAULT + (Decimal(positive_count) * Decimal("3")) - (Decimal(negative_count) * Decimal("2"))
        score = max(Decimal("35"), min(Decimal("65"), net))
        return (score, "MIXED_EVENTS")

    # No classified events
    return (SCORE_DEFAULT, "NO_CLASSIFIED_EVENTS")


# =============================================================================
# BLENDED SCORING (CONFIDENCE + RECENCY WEIGHTED)
# =============================================================================

def calculate_score_blended(
    events: List[CatalystEventV2],
    as_of_date: date,
) -> Tuple[Decimal, Dict[str, str]]:
    """
    Calculate blended score with confidence and recency weighting.

    Formula:
    - For each event: contribution = severity_score * confidence_weight * recency_weight
    - Sum all contributions
    - Apply staleness penalty if applicable
    - Clamp to [0, 100]

    Args:
        events: List of catalyst events
        as_of_date: Point-in-time date

    Returns:
        (score, weighted_counts_by_severity)
    """
    if not events:
        return (SCORE_DEFAULT, {})

    # Accumulate weighted contributions
    weighted_sums: Dict[EventSeverity, Decimal] = {sev: Decimal("0") for sev in EventSeverity}
    total_contribution = Decimal("0")

    for event in events:
        # Skip future events (PIT safety)
        if event.event_date:
            try:
                event_d = date.fromisoformat(event.event_date)
                if event_d > as_of_date:
                    continue
            except (ValueError, TypeError):
                pass

        severity = event.event_severity
        confidence = event.confidence

        # Weights
        confidence_weight = CONFIDENCE_WEIGHTS.get(confidence, Decimal("0.5"))
        recency_weight = compute_recency_weight(event.event_date, as_of_date)

        # Severity contribution (can be negative for negative events)
        severity_contrib = SEVERITY_SCORE_CONTRIBUTION.get(severity, Decimal("0"))

        # Combined contribution
        contribution = severity_contrib * confidence_weight * recency_weight

        total_contribution += contribution
        weighted_sums[severity] += confidence_weight * recency_weight

    # Apply staleness penalty
    staleness = compute_staleness_factor(events, as_of_date)

    # Final score: base + weighted contribution, with staleness
    score = SCORE_BLENDED_BASE + (total_contribution * staleness)

    # Clamp
    score = max(SCORE_MIN, min(SCORE_MAX, score))

    # Convert weighted sums to strings for JSON
    weighted_counts = {
        sev.value: decimal_to_str(weighted_sums[sev])
        for sev in EventSeverity
        if weighted_sums[sev] > 0
    }

    return (score, weighted_counts)


# =============================================================================
# TICKER-LEVEL SCORING
# =============================================================================

def calculate_ticker_catalyst_score(
    ticker: str,
    events: List[CatalystEventV2],
    as_of_date: date,
) -> TickerCatalystSummaryV2:
    """
    Calculate complete catalyst score for a ticker.

    Args:
        ticker: Ticker symbol
        events: List of catalyst events for this ticker
        as_of_date: Point-in-time date

    Returns:
        TickerCatalystSummaryV2 with both override and blended scores
    """
    # Sort events deterministically
    sorted_events = sorted(events, key=lambda e: e.sort_key())

    # Calculate both scores
    score_override, override_reason = calculate_score_override(sorted_events)
    score_blended, weighted_counts = calculate_score_blended(sorted_events, as_of_date)

    # Determine which mode to use
    # Use override when it differs materially (>5 points) from blended
    score_diff = abs(score_override - score_blended)
    if score_diff > Decimal("5"):
        score_mode = "override"
    else:
        score_mode = "blended"

    # Check for severe negative flag
    severe_negative_flag = any(
        e.event_severity == EventSeverity.SEVERE_NEGATIVE
        for e in sorted_events
    )

    # Compute integration hooks
    next_catalyst_date, catalyst_window_days = _compute_next_catalyst(sorted_events, as_of_date)
    catalyst_window_bucket = compute_catalyst_window_bucket(catalyst_window_days)
    catalyst_confidence = _compute_catalyst_confidence(sorted_events, next_catalyst_date)

    # Event summary counts
    events_by_severity: Dict[str, int] = {}
    events_by_type: Dict[str, int] = {}
    for e in sorted_events:
        sev = e.event_severity.value
        events_by_severity[sev] = events_by_severity.get(sev, 0) + 1
        typ = e.event_type.value
        events_by_type[typ] = events_by_type.get(typ, 0) + 1

    # Top 3 events (highest severity, most recent, highest confidence)
    top_3 = _select_top_3_events(sorted_events, as_of_date)

    return TickerCatalystSummaryV2(
        ticker=ticker,
        as_of_date=as_of_date.isoformat(),
        score_override=score_override.quantize(Decimal("0.01")),
        score_blended=score_blended.quantize(Decimal("0.01")),
        score_mode_used=score_mode,
        severe_negative_flag=severe_negative_flag,
        next_catalyst_date=next_catalyst_date,
        catalyst_window_days=catalyst_window_days,
        catalyst_window_bucket=catalyst_window_bucket,
        catalyst_confidence=catalyst_confidence,
        events_total=len(sorted_events),
        events_by_severity=dict(sorted(events_by_severity.items())),
        events_by_type=dict(sorted(events_by_type.items())),
        weighted_counts_by_severity=dict(sorted(weighted_counts.items())),
        top_3_events=[e.to_dict() for e in top_3],
        events=sorted_events,
    )


def _compute_next_catalyst(
    events: List[CatalystEventV2],
    as_of_date: date,
) -> Tuple[Optional[str], Optional[int]]:
    """
    Compute next catalyst date from events.

    Returns:
        (next_date_iso, days_until)
    """
    future_dates = []

    for event in events:
        if event.event_date:
            try:
                event_d = date.fromisoformat(event.event_date)
                if event_d > as_of_date:
                    future_dates.append(event_d)
            except (ValueError, TypeError):
                continue

    if not future_dates:
        return (None, None)

    next_date = min(future_dates)
    days_until = (next_date - as_of_date).days

    return (next_date.isoformat(), days_until)


def _compute_catalyst_confidence(
    events: List[CatalystEventV2],
    next_catalyst_date: Optional[str],
) -> ConfidenceLevel:
    """
    Compute confidence level for catalyst window.

    Uses highest confidence among upcoming catalyst events.
    """
    if not next_catalyst_date:
        return ConfidenceLevel.MED

    # Find events matching next catalyst date
    matching_confidences = []
    for event in events:
        if event.event_date == next_catalyst_date:
            matching_confidences.append(event.confidence)

    if not matching_confidences:
        return ConfidenceLevel.MED

    # Return highest confidence
    priority = {ConfidenceLevel.HIGH: 0, ConfidenceLevel.MED: 1, ConfidenceLevel.LOW: 2}
    return min(matching_confidences, key=lambda c: priority.get(c, 99))


def _select_top_3_events(
    events: List[CatalystEventV2],
    as_of_date: date,
) -> List[CatalystEventV2]:
    """
    Select top 3 events for downstream consumers.

    Priority:
    1. Highest severity (SEVERE_NEGATIVE > CRITICAL_POSITIVE > ...)
    2. Most recent (within as_of_date)
    3. Highest confidence
    """
    severity_priority = {
        EventSeverity.SEVERE_NEGATIVE: 0,
        EventSeverity.CRITICAL_POSITIVE: 1,
        EventSeverity.NEGATIVE: 2,
        EventSeverity.POSITIVE: 3,
        EventSeverity.NEUTRAL: 4,
    }

    confidence_priority = {
        ConfidenceLevel.HIGH: 0,
        ConfidenceLevel.MED: 1,
        ConfidenceLevel.LOW: 2,
    }

    def event_priority(e: CatalystEventV2) -> Tuple:
        sev_pri = severity_priority.get(e.event_severity, 99)
        conf_pri = confidence_priority.get(e.confidence, 99)

        # Recency: more recent = lower value = higher priority
        if e.event_date:
            try:
                event_d = date.fromisoformat(e.event_date)
                days_old = (as_of_date - event_d).days
                if days_old < 0:
                    days_old = 9999  # Future events lower priority
            except (ValueError, TypeError):
                days_old = 9999
        else:
            days_old = 9999

        return (sev_pri, days_old, conf_pri, e.event_id)

    sorted_by_priority = sorted(events, key=event_priority)
    return sorted_by_priority[:3]


# =============================================================================
# BATCH SCORING
# =============================================================================

def score_catalyst_events(
    events_by_ticker: Dict[str, List[CatalystEventV2]],
    active_tickers: List[str],
    as_of_date: date,
) -> Tuple[Dict[str, TickerCatalystSummaryV2], DiagnosticCounts]:
    """
    Score catalyst events for all active tickers.

    Args:
        events_by_ticker: Dict of ticker -> list of events
        active_tickers: List of active tickers to score
        as_of_date: Point-in-time date

    Returns:
        (summaries_dict, diagnostics)
    """
    logger.info(f"Scoring catalyst events for {len(active_tickers)} tickers")

    summaries = {}
    diagnostics = DiagnosticCounts()
    diagnostics.tickers_analyzed = len(active_tickers)

    for ticker in sorted(active_tickers):  # Sorted for determinism
        events = events_by_ticker.get(ticker, [])

        summary = calculate_ticker_catalyst_score(ticker, events, as_of_date)
        summaries[ticker] = summary

        # Update diagnostics
        diagnostics.events_detected_total += summary.events_total

        if summary.events_total > 0:
            diagnostics.tickers_with_events += 1

        if summary.severe_negative_flag:
            diagnostics.tickers_with_severe_negative += 1

        for typ, count in summary.events_by_type.items():
            diagnostics.events_by_type[typ] = diagnostics.events_by_type.get(typ, 0) + count

        for sev, count in summary.events_by_severity.items():
            diagnostics.events_by_severity[sev] = diagnostics.events_by_severity.get(sev, 0) + count

        for conf_level in ConfidenceLevel:
            count = sum(1 for e in events if e.confidence == conf_level)
            if count > 0:
                key = conf_level.value
                diagnostics.events_by_confidence[key] = diagnostics.events_by_confidence.get(key, 0) + count

    logger.info(f"Scoring complete. Generated {len(summaries)} summaries")
    logger.info(f"Diagnostics: {diagnostics.to_dict()}")

    return (summaries, diagnostics)


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

# Keep old EVENT_SEVERITY for backwards compatibility with existing code
EVENT_SEVERITY = {
    'PHASE_ADVANCE_P2_TO_P3': 'CRITICAL_POSITIVE',
    'PHASE_ADVANCE_P3_TO_NDA': 'CRITICAL_POSITIVE',
    'FDA_APPROVAL': 'CRITICAL_POSITIVE',
    'BREAKTHROUGH_DESIGNATION': 'CRITICAL_POSITIVE',
    'TRIAL_TERMINATION': 'SEVERE_NEGATIVE',
    'TRIAL_SUSPENDED': 'SEVERE_NEGATIVE',
    'FDA_REJECTION': 'SEVERE_NEGATIVE',
    'SAFETY_HOLD': 'SEVERE_NEGATIVE',
    'ENROLLMENT_COMPLETE': 'POSITIVE',
    'ENROLLMENT_STARTED': 'POSITIVE',
    'FAST_TRACK_GRANTED': 'POSITIVE',
    'ORPHAN_DESIGNATION': 'POSITIVE',
    'ENROLLMENT_DELAY': 'NEGATIVE',
    'TRIAL_DELAYED': 'NEGATIVE',
}

SEVERITY_SCORES = {
    'CRITICAL_POSITIVE': 75.0,
    'POSITIVE': 60.0,
    'NEGATIVE': 40.0,
    'SEVERE_NEGATIVE': 20.0,
}

DEFAULT_SCORE = 50.0


def calculate_ticker_catalyst_score_legacy(ticker: str, summary) -> Dict:
    """
    Legacy scoring function for backwards compatibility.

    Use calculate_ticker_catalyst_score() for new code.
    """
    # Handle both dict and dataclass
    if isinstance(summary, dict):
        events = summary.get('events', [])
    else:
        events = getattr(summary, 'events', [])

    if not events:
        return {
            'ticker': ticker,
            'score': DEFAULT_SCORE,
            'reason': 'NO_EVENTS',
            'event_count': 0,
            'severity_breakdown': {}
        }

    severity_counts = {}
    event_details = []

    for event in events:
        if isinstance(event, dict):
            event_type = event.get('event_type', 'UNKNOWN')
        else:
            event_type = getattr(event, 'event_type', 'UNKNOWN')
            if hasattr(event_type, 'value'):
                event_type = event_type.value

        severity = EVENT_SEVERITY.get(event_type, 'NEUTRAL')
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        event_details.append({
            'event_type': event_type,
            'severity': severity
        })

    # Hierarchical scoring
    if severity_counts.get('SEVERE_NEGATIVE', 0) > 0:
        score = 20.0
    elif severity_counts.get('CRITICAL_POSITIVE', 0) > 0:
        score = 75.0
    else:
        positive_count = severity_counts.get('POSITIVE', 0)
        negative_count = severity_counts.get('NEGATIVE', 0)

        if positive_count > 0 and negative_count == 0:
            score = 60.0 + min(10.0, positive_count * 5.0)
        elif negative_count > 0 and positive_count == 0:
            score = 40.0 - min(5.0, negative_count * 2.5)
        elif positive_count > 0 and negative_count > 0:
            score = 50.0 + (positive_count * 3.0) - (negative_count * 2.0)
            score = max(35.0, min(65.0, score))
        else:
            score = DEFAULT_SCORE

    reason = 'SEVERE_NEGATIVE_EVENTS' if severity_counts.get('SEVERE_NEGATIVE', 0) > 0 else \
             'CRITICAL_POSITIVE_EVENTS' if severity_counts.get('CRITICAL_POSITIVE', 0) > 0 else \
             'POSITIVE_EVENTS' if severity_counts.get('POSITIVE', 0) > 0 and severity_counts.get('NEGATIVE', 0) == 0 else \
             'NEGATIVE_EVENTS' if severity_counts.get('NEGATIVE', 0) > 0 and severity_counts.get('POSITIVE', 0) == 0 else \
             'MIXED_EVENTS' if severity_counts.get('POSITIVE', 0) > 0 and severity_counts.get('NEGATIVE', 0) > 0 else \
             'NO_CLASSIFIED_EVENTS'

    return {
        'ticker': ticker,
        'score': score,
        'reason': reason,
        'event_count': len(events),
        'severity_breakdown': severity_counts,
        'event_details': event_details
    }


def validate_scores(scores: Dict[str, Dict]) -> Dict:
    """Validate score outputs for sanity."""
    if not scores:
        return {'valid': False, 'reason': 'NO_SCORES'}

    score_values = [s['score'] if isinstance(s, dict) else float(s.score_override) for s in scores.values()]

    return {
        'valid': True,
        'total_tickers': len(scores),
        'unique_scores': len(set(score_values)),
        'min_score': min(score_values) if score_values else None,
        'max_score': max(score_values) if score_values else None,
        'mean_score': sum(score_values) / len(score_values) if score_values else None,
    }
