#!/usr/bin/env python3
"""
module_3_scoring_v2.py - Module 3 Catalyst Scoring System v2 (Robust)

Enhanced scoring with:
- catalyst_proximity_score with exponential decay
- certainty_score from source reliability, date specificity, corroboration, staleness
- Delta engine: event_added/removed, date_shift_days, window_widening, status_change
- catalyst_delta_score computation
- Rolling baseline velocity: catalyst_velocity_4w
- Negative catalyst scoring (holds, terminations, CRL, slips)
- Enhanced diagnostics

Design Philosophy (Governed Over Smart):
- DETERMINISTIC: No datetime.now(), no randomness
- PIT-SAFE: All dates explicit, no wall-clock
- STDLIB-ONLY: No external dependencies

Version: 2.0.0
"""

import logging
import math
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any, Set

from module_3_schema_v2 import (
    SCHEMA_VERSION,
    SCORE_VERSION,
    EventType,
    EventSeverity,
    ConfidenceLevel,
    CatalystWindowBucket,
    SourceReliability,
    DateSpecificity,
    DeltaType,
    CatalystEventV2,
    DeltaEvent,
    TickerCatalystSummaryV2,
    DiagnosticCountsV2,
    NEGATIVE_CATALYST_TYPES,
    SEVERE_NEGATIVE_TYPES,
    CONFIDENCE_WEIGHTS,
    SEVERITY_SCORE_CONTRIBUTION,
    EVENT_TYPE_WEIGHT,
    EVENT_DEFAULT_CONFIDENCE,
    compute_catalyst_window_bucket,
    decimal_to_str,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# SCORE CONSTANTS
# =============================================================================

SCORE_OVERRIDE_SEVERE_NEGATIVE = Decimal("20")
SCORE_OVERRIDE_CRITICAL_POSITIVE = Decimal("75")
SCORE_DEFAULT = Decimal("50")
SCORE_MIN = Decimal("0")
SCORE_MAX = Decimal("100")

# Recency decay
DECAY_HALF_LIFE_DAYS = 90
STALENESS_THRESHOLD_DAYS = 180
STALENESS_PENALTY_FACTOR = Decimal("0.8")

# Proximity scoring
PROXIMITY_HORIZON_DAYS = 270
PROXIMITY_HALF_LIFE_DAYS = 120
PROXIMITY_SCALE_FACTOR = Decimal("2.0")

# Delta scoring
DELTA_SCALE_FACTOR = Decimal("1.0")

# Negative catalyst scoring
NEGATIVE_SCALE_FACTOR = Decimal("1.5")


# =============================================================================
# RECENCY / DECAY
# =============================================================================

def compute_recency_weight(
    event_date: Optional[str],
    as_of_date: date,
) -> Decimal:
    """Compute recency weight using exponential decay."""
    if event_date is None:
        return Decimal("0.5")

    try:
        event_d = date.fromisoformat(event_date)
    except (ValueError, TypeError):
        return Decimal("0.5")

    days_old = (as_of_date - event_d).days

    if days_old < 0:
        return Decimal("0")  # Future event

    if days_old == 0:
        return Decimal("1.0")

    decay_factor = Decimal(days_old) / Decimal(DECAY_HALF_LIFE_DAYS)
    weight = Decimal("0.5") ** decay_factor

    return weight.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


def compute_staleness_factor(
    events: List[CatalystEventV2],
    as_of_date: date,
) -> Decimal:
    """Compute staleness penalty factor."""
    if not events:
        return Decimal("1.0")

    newest_days = None
    for event in events:
        if event.event_date:
            try:
                event_d = date.fromisoformat(event.event_date)
                days_old = (as_of_date - event_d).days
                if days_old >= 0:
                    if newest_days is None or days_old < newest_days:
                        newest_days = days_old
            except (ValueError, TypeError):
                continue

    if newest_days is None:
        return Decimal("1.0")

    if newest_days > STALENESS_THRESHOLD_DAYS:
        return STALENESS_PENALTY_FACTOR

    return Decimal("1.0")


# =============================================================================
# PROXIMITY SCORING (ENHANCED)
# =============================================================================

def compute_proximity_score(
    events: List[CatalystEventV2],
    as_of_date: date,
    horizon_days: int = PROXIMITY_HORIZON_DAYS,
    half_life_days: int = PROXIMITY_HALF_LIFE_DAYS,
) -> Tuple[Decimal, int]:
    """
    Compute proximity score for upcoming catalyst events.

    Formula per event:
        contrib = type_weight * certainty_weight * exp(-days_to_event / half_life)

    Args:
        events: List of catalyst events
        as_of_date: Point-in-time date
        horizon_days: Look-ahead window (default 270)
        half_life_days: Decay half-life (default 120)

    Returns:
        (proximity_score, n_events_upcoming)
    """
    if not events:
        return (Decimal("0"), 0)

    total_contrib = Decimal("0")
    n_upcoming = 0

    for event in events:
        if not event.event_date:
            continue

        try:
            event_d = date.fromisoformat(event.event_date)
        except (ValueError, TypeError):
            continue

        days_to_event = (event_d - as_of_date).days

        # Only future events within horizon
        if days_to_event <= 0 or days_to_event > horizon_days:
            continue

        n_upcoming += 1

        # Type weight (can be negative for negative catalysts)
        type_weight = EVENT_TYPE_WEIGHT.get(event.event_type, Decimal("1.0"))

        # Certainty weight (combines confidence + source reliability + date specificity)
        certainty = event.compute_certainty_score(as_of_date)

        # Exponential decay: closer events score higher
        decay_factor = Decimal(days_to_event) / Decimal(half_life_days)
        time_weight = Decimal("0.5") ** decay_factor

        # Combined contribution
        contrib = type_weight * certainty * time_weight
        total_contrib += contrib

    # Scale and clamp
    scaled_score = total_contrib * PROXIMITY_SCALE_FACTOR
    clamped_score = max(Decimal("-50"), min(Decimal("100"), scaled_score))

    return (clamped_score.quantize(Decimal("0.01")), n_upcoming)


# =============================================================================
# DELTA ENGINE (ENHANCED)
# =============================================================================

def detect_deltas(
    current_events: List[CatalystEventV2],
    prior_events: List[CatalystEventV2],
    as_of_date: date,
) -> List[DeltaEvent]:
    """
    Detect deltas between current and prior event snapshots.

    Detects:
    - EVENT_ADDED: New event_id in current
    - EVENT_REMOVED: Event_id in prior but not current
    - DATE_SHIFT: Same stable_id, different event_date
    - WINDOW_WIDENING: Completion date moved later
    - STATUS_CHANGE: Status field changed
    """
    deltas = []
    source_date = as_of_date.isoformat()

    # Build indices by stable_id (ticker + nct_id + event_type)
    def stable_id(e: CatalystEventV2) -> str:
        return f"{e.ticker}|{e.nct_id}|{e.event_type.value}"

    current_by_stable = {stable_id(e): e for e in current_events}
    prior_by_stable = {stable_id(e): e for e in prior_events}

    # Build indices by event_id (full hash)
    current_ids = {e.event_id for e in current_events}
    prior_ids = {e.event_id for e in prior_events}

    # EVENT_ADDED: new event_id
    for e in current_events:
        if e.event_id not in prior_ids:
            deltas.append(DeltaEvent(
                ticker=e.ticker,
                nct_id=e.nct_id,
                delta_type=DeltaType.EVENT_ADDED,
                prior_value=None,
                new_value=e.event_type.value,
                shift_days=None,
                source_date=source_date,
            ))

    # EVENT_REMOVED: event_id in prior but not current
    for e in prior_events:
        if e.event_id not in current_ids:
            deltas.append(DeltaEvent(
                ticker=e.ticker,
                nct_id=e.nct_id,
                delta_type=DeltaType.EVENT_REMOVED,
                prior_value=e.event_type.value,
                new_value=None,
                shift_days=None,
                source_date=source_date,
            ))

    # DATE_SHIFT / WINDOW_WIDENING / STATUS_CHANGE for matched stable_ids
    matched_stable_ids = set(current_by_stable.keys()) & set(prior_by_stable.keys())

    for sid in matched_stable_ids:
        curr = current_by_stable[sid]
        prior = prior_by_stable[sid]

        # DATE_SHIFT
        if curr.event_date and prior.event_date and curr.event_date != prior.event_date:
            try:
                curr_d = date.fromisoformat(curr.event_date)
                prior_d = date.fromisoformat(prior.event_date)
                shift_days = (curr_d - prior_d).days

                delta_type = DeltaType.DATE_SHIFT
                # If shift is positive and this is a completion event, it's WINDOW_WIDENING
                if shift_days > 0 and curr.event_type in (
                    EventType.CT_PRIMARY_COMPLETION,
                    EventType.CT_STUDY_COMPLETION,
                ):
                    delta_type = DeltaType.WINDOW_WIDENING

                deltas.append(DeltaEvent(
                    ticker=curr.ticker,
                    nct_id=curr.nct_id,
                    delta_type=delta_type,
                    prior_value=prior.event_date,
                    new_value=curr.event_date,
                    shift_days=shift_days,
                    source_date=source_date,
                ))
            except (ValueError, TypeError):
                pass

        # STATUS_CHANGE (via field_changed containing "status")
        if "status" in curr.field_changed.lower() and curr.new_value != prior.new_value:
            deltas.append(DeltaEvent(
                ticker=curr.ticker,
                nct_id=curr.nct_id,
                delta_type=DeltaType.STATUS_CHANGE,
                prior_value=prior.new_value,
                new_value=curr.new_value,
                shift_days=None,
                source_date=source_date,
            ))

    return deltas


def compute_delta_score(
    deltas: List[DeltaEvent],
    as_of_date: date,
) -> Tuple[Decimal, int, int, int, int, int, Optional[int]]:
    """
    Compute delta score from detected deltas.

    Returns:
        (delta_score, n_added, n_removed, n_date_shifts, n_window_widenings, n_status_changes, max_slip_days)
    """
    if not deltas:
        return (Decimal("0"), 0, 0, 0, 0, 0, None)

    score = Decimal("0")
    n_added = 0
    n_removed = 0
    n_date_shifts = 0
    n_window_widenings = 0
    n_status_changes = 0
    max_slip = None

    for delta in deltas:
        if delta.delta_type == DeltaType.EVENT_ADDED:
            n_added += 1
            score += Decimal("3")  # Positive: new event visibility

        elif delta.delta_type == DeltaType.EVENT_REMOVED:
            n_removed += 1
            score -= Decimal("2")  # Negative: lost visibility

        elif delta.delta_type == DeltaType.DATE_SHIFT:
            n_date_shifts += 1
            if delta.shift_days is not None:
                if max_slip is None or delta.shift_days > max_slip:
                    max_slip = delta.shift_days

                # Magnitude-weighted score
                magnitude = min(Decimal("1"), Decimal(abs(delta.shift_days)) / Decimal("90"))
                if delta.shift_days < 0:
                    score += Decimal("5") * magnitude  # Earlier = positive
                else:
                    score -= Decimal("5") * magnitude  # Later = negative

        elif delta.delta_type == DeltaType.WINDOW_WIDENING:
            n_window_widenings += 1
            if delta.shift_days is not None:
                if max_slip is None or delta.shift_days > max_slip:
                    max_slip = delta.shift_days
                magnitude = min(Decimal("1"), Decimal(delta.shift_days) / Decimal("90"))
                score -= Decimal("8") * magnitude  # Window widening is worse

        elif delta.delta_type == DeltaType.STATUS_CHANGE:
            n_status_changes += 1
            # Status changes can be positive or negative depending on content
            # For now, neutral contribution
            score += Decimal("0")

    # Scale and clamp
    scaled = score * DELTA_SCALE_FACTOR
    clamped = max(Decimal("-50"), min(Decimal("50"), scaled))

    return (
        clamped.quantize(Decimal("0.01")),
        n_added,
        n_removed,
        n_date_shifts,
        n_window_widenings,
        n_status_changes,
        max_slip,
    )


# =============================================================================
# NEGATIVE CATALYST SCORING
# =============================================================================

def compute_negative_catalyst_score(
    events: List[CatalystEventV2],
    as_of_date: date,
) -> Tuple[Decimal, int, int]:
    """
    Compute separate score for negative catalysts.

    Considers: holds, terminations, CRL, slips, etc.

    Returns:
        (negative_score, n_negative, n_severe_negative)
    """
    if not events:
        return (Decimal("0"), 0, 0)

    score = Decimal("0")
    n_negative = 0
    n_severe_negative = 0

    for event in events:
        if not event.is_negative:
            continue

        n_negative += 1
        if event.is_severe_negative:
            n_severe_negative += 1

        # Get type weight (negative values for negative events)
        type_weight = EVENT_TYPE_WEIGHT.get(event.event_type, Decimal("-5.0"))

        # Certainty weight
        certainty = event.compute_certainty_score(as_of_date)

        # Recency weight (more recent = higher impact)
        recency = compute_recency_weight(event.event_date, as_of_date)

        # Combined contribution (type_weight is already negative)
        contrib = type_weight * certainty * recency
        score += contrib

    # Scale (make negative score a positive number representing risk)
    scaled = abs(score) * NEGATIVE_SCALE_FACTOR
    clamped = min(Decimal("100"), scaled)

    return (clamped.quantize(Decimal("0.01")), n_negative, n_severe_negative)


# =============================================================================
# CERTAINTY SCORING
# =============================================================================

def compute_avg_certainty(
    events: List[CatalystEventV2],
    as_of_date: date,
) -> Tuple[Decimal, int, Decimal]:
    """
    Compute average certainty score and uncertainty penalty.

    Returns:
        (avg_certainty, n_high_conf, uncertainty_penalty)
    """
    if not events:
        return (Decimal("0.5"), 0, Decimal("0"))

    certainties = []
    n_high = 0

    for event in events:
        cert = event.compute_certainty_score(as_of_date)
        certainties.append(cert)
        if event.confidence == ConfidenceLevel.HIGH:
            n_high += 1

    avg = sum(certainties) / len(certainties)

    # Uncertainty penalty: if avg certainty < 0.5, apply penalty
    if avg < Decimal("0.5"):
        penalty = (Decimal("0.5") - avg) * Decimal("10")
    else:
        penalty = Decimal("0")

    return (
        avg.quantize(Decimal("0.001")),
        n_high,
        penalty.quantize(Decimal("0.01")),
    )


# =============================================================================
# VELOCITY SCORING
# =============================================================================

def compute_velocity(
    current_proximity: Decimal,
    historical_proximities: List[Decimal],
) -> Optional[Decimal]:
    """
    Compute velocity as current proximity minus rolling median of last 4.
    """
    if len(historical_proximities) < 4:
        return None

    recent_4 = historical_proximities[:4]
    sorted_scores = sorted(recent_4)
    n = len(sorted_scores)

    if n % 2 == 0:
        median = (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
    else:
        median = sorted_scores[n // 2]

    velocity = current_proximity - median
    return velocity.quantize(Decimal("0.01"))


# =============================================================================
# OVERRIDE SCORING
# =============================================================================

def calculate_score_override(
    events: List[CatalystEventV2],
) -> Tuple[Decimal, str]:
    """Calculate override score using hierarchical logic."""
    if not events:
        return (SCORE_DEFAULT, "NO_EVENTS")

    severity_counts: Dict[EventSeverity, int] = {}
    for event in events:
        sev = event.event_severity
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    if severity_counts.get(EventSeverity.SEVERE_NEGATIVE, 0) > 0:
        return (SCORE_OVERRIDE_SEVERE_NEGATIVE, "SEVERE_NEGATIVE_EVENTS")

    if severity_counts.get(EventSeverity.CRITICAL_POSITIVE, 0) > 0:
        return (SCORE_OVERRIDE_CRITICAL_POSITIVE, "CRITICAL_POSITIVE_EVENTS")

    positive_count = (
        severity_counts.get(EventSeverity.POSITIVE, 0) +
        severity_counts.get(EventSeverity.CRITICAL_POSITIVE, 0)
    )
    negative_count = (
        severity_counts.get(EventSeverity.NEGATIVE, 0) +
        severity_counts.get(EventSeverity.SEVERE_NEGATIVE, 0)
    )

    if positive_count > 0 and negative_count == 0:
        score = Decimal("60") + min(Decimal("10"), Decimal(positive_count) * Decimal("5"))
        return (score, "POSITIVE_EVENTS")

    if negative_count > 0 and positive_count == 0:
        score = Decimal("40") - min(Decimal("5"), Decimal(negative_count) * Decimal("2.5"))
        return (score, "NEGATIVE_EVENTS")

    if positive_count > 0 and negative_count > 0:
        net = SCORE_DEFAULT + (Decimal(positive_count) * Decimal("3")) - (Decimal(negative_count) * Decimal("2"))
        score = max(Decimal("35"), min(Decimal("65"), net))
        return (score, "MIXED_EVENTS")

    return (SCORE_DEFAULT, "NO_CLASSIFIED_EVENTS")


# =============================================================================
# BLENDED SCORING
# =============================================================================

def calculate_score_blended(
    events: List[CatalystEventV2],
    as_of_date: date,
) -> Tuple[Decimal, Dict[str, str]]:
    """Calculate blended score with confidence and recency weighting."""
    if not events:
        return (SCORE_DEFAULT, {})

    weighted_sums: Dict[EventSeverity, Decimal] = {sev: Decimal("0") for sev in EventSeverity}
    total_contribution = Decimal("0")

    for event in events:
        # PIT safety
        if event.event_date:
            try:
                event_d = date.fromisoformat(event.event_date)
                if event_d > as_of_date:
                    continue
            except (ValueError, TypeError):
                pass

        severity = event.event_severity
        confidence = event.confidence

        confidence_weight = CONFIDENCE_WEIGHTS.get(confidence, Decimal("0.5"))
        recency_weight = compute_recency_weight(event.event_date, as_of_date)

        severity_contrib = SEVERITY_SCORE_CONTRIBUTION.get(severity, Decimal("0"))
        contribution = severity_contrib * confidence_weight * recency_weight

        total_contribution += contribution
        weighted_sums[severity] += confidence_weight * recency_weight

    staleness = compute_staleness_factor(events, as_of_date)
    score = SCORE_DEFAULT + (total_contribution * staleness)
    score = max(SCORE_MIN, min(SCORE_MAX, score))

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
    prior_events: Optional[List[CatalystEventV2]] = None,
    historical_proximities: Optional[List[Decimal]] = None,
) -> TickerCatalystSummaryV2:
    """
    Calculate complete catalyst score for a ticker.

    Includes:
    - Override and blended scores
    - Proximity score for upcoming catalysts
    - Delta score from event changes
    - Velocity (4-week rolling)
    - Negative catalyst score
    - Certainty metrics
    """
    sorted_events = sorted(events, key=lambda e: e.sort_key())

    # Base scores
    score_override, override_reason = calculate_score_override(sorted_events)
    score_blended, weighted_counts = calculate_score_blended(sorted_events, as_of_date)

    # Proximity score
    proximity_score, n_upcoming = compute_proximity_score(sorted_events, as_of_date)

    # Delta scoring
    delta_events = []
    delta_score = Decimal("0")
    n_added = 0
    n_removed = 0
    n_date_shifts = 0
    n_window_widenings = 0
    n_status_changes = 0
    max_slip = None

    if prior_events is not None:
        sorted_prior = sorted(prior_events, key=lambda e: e.sort_key())
        delta_events = detect_deltas(sorted_events, sorted_prior, as_of_date)
        (
            delta_score, n_added, n_removed, n_date_shifts,
            n_window_widenings, n_status_changes, max_slip
        ) = compute_delta_score(delta_events, as_of_date)

    # Velocity
    velocity = None
    if historical_proximities:
        velocity = compute_velocity(proximity_score, historical_proximities)

    # Negative catalyst score
    negative_score, n_negative, n_severe_negative = compute_negative_catalyst_score(
        sorted_events, as_of_date
    )

    # Certainty
    avg_certainty, n_high_conf, uncertainty_penalty = compute_avg_certainty(
        sorted_events, as_of_date
    )

    # Mode selection
    score_diff = abs(score_override - score_blended)
    score_mode = "override" if score_diff > Decimal("5") else "blended"

    # Flags
    severe_negative_flag = any(e.is_severe_negative for e in sorted_events)

    # Integration hooks
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

    # Top 3 events
    top_3 = _select_top_3_events(sorted_events, as_of_date)

    # Coverage state
    if len(sorted_events) > 0:
        coverage_state = "FULL"
    else:
        coverage_state = "NONE"

    return TickerCatalystSummaryV2(
        ticker=ticker,
        as_of_date=as_of_date.isoformat(),
        score_override=score_override.quantize(Decimal("0.01")),
        score_blended=score_blended.quantize(Decimal("0.01")),
        score_mode_used=score_mode,
        catalyst_proximity_score=proximity_score,
        n_events_upcoming=n_upcoming,
        catalyst_delta_score=delta_score,
        n_events_added=n_added,
        n_events_removed=n_removed,
        n_date_shifts=n_date_shifts,
        n_window_widenings=n_window_widenings,
        n_status_changes=n_status_changes,
        max_slip_days=max_slip,
        catalyst_velocity_4w=velocity,
        negative_catalyst_score=negative_score,
        n_negative_events=n_negative,
        n_severe_negative_events=n_severe_negative,
        avg_certainty_score=avg_certainty,
        n_high_confidence=n_high_conf,
        uncertainty_penalty=uncertainty_penalty,
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
        delta_events=delta_events,
        coverage_state=coverage_state,
    )


def _compute_next_catalyst(
    events: List[CatalystEventV2],
    as_of_date: date,
) -> Tuple[Optional[str], Optional[int]]:
    """Compute next catalyst date from events."""
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
    """Compute confidence level for catalyst window."""
    if not next_catalyst_date:
        return ConfidenceLevel.MED

    matching_confidences = []
    for event in events:
        if event.event_date == next_catalyst_date:
            matching_confidences.append(event.confidence)

    if not matching_confidences:
        return ConfidenceLevel.MED

    priority = {ConfidenceLevel.HIGH: 0, ConfidenceLevel.MED: 1, ConfidenceLevel.LOW: 2}
    return min(matching_confidences, key=lambda c: priority.get(c, 99))


def _select_top_3_events(
    events: List[CatalystEventV2],
    as_of_date: date,
) -> List[CatalystEventV2]:
    """Select top 3 events by severity, recency, confidence."""
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

        if e.event_date:
            try:
                event_d = date.fromisoformat(e.event_date)
                days_old = (as_of_date - event_d).days
                if days_old < 0:
                    days_old = 9999
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
    prior_events_by_ticker: Optional[Dict[str, List[CatalystEventV2]]] = None,
    historical_proximities_by_ticker: Optional[Dict[str, List[Decimal]]] = None,
) -> Tuple[Dict[str, TickerCatalystSummaryV2], DiagnosticCountsV2]:
    """Score catalyst events for all active tickers."""
    logger.info(f"Scoring catalyst events for {len(active_tickers)} tickers")

    summaries = {}
    diagnostics = DiagnosticCountsV2()
    diagnostics.tickers_analyzed = len(active_tickers)

    prior_events_by_ticker = prior_events_by_ticker or {}
    historical_proximities_by_ticker = historical_proximities_by_ticker or {}

    total_certainty = Decimal("0")
    certainty_count = 0

    for ticker in sorted(active_tickers):
        events = events_by_ticker.get(ticker, [])
        prior_events = prior_events_by_ticker.get(ticker)
        historical_proximities = historical_proximities_by_ticker.get(ticker)

        summary = calculate_ticker_catalyst_score(
            ticker, events, as_of_date,
            prior_events=prior_events,
            historical_proximities=historical_proximities,
        )
        summaries[ticker] = summary

        # Update diagnostics
        diagnostics.events_detected_total += summary.events_total
        diagnostics.n_events_upcoming += summary.n_events_upcoming
        diagnostics.n_high_conf += summary.n_high_confidence
        diagnostics.n_negative_total += summary.n_negative_events
        diagnostics.n_severe_negative_total += summary.n_severe_negative_events

        if summary.max_slip_days is not None:
            if diagnostics.max_slip_days is None or summary.max_slip_days > diagnostics.max_slip_days:
                diagnostics.max_slip_days = summary.max_slip_days
            diagnostics.total_slip_events += summary.n_date_shifts + summary.n_window_widenings

        diagnostics.uncertainty_penalty_total += summary.uncertainty_penalty

        if summary.events_total > 0:
            total_certainty += summary.avg_certainty_score
            certainty_count += 1

        if summary.events_total > 0:
            diagnostics.tickers_with_events += 1

        if summary.severe_negative_flag:
            diagnostics.tickers_with_severe_negative += 1

        for typ, count in summary.events_by_type.items():
            diagnostics.events_by_type[typ] = diagnostics.events_by_type.get(typ, 0) + count

        for sev, count in summary.events_by_severity.items():
            diagnostics.events_by_severity[sev] = diagnostics.events_by_severity.get(sev, 0) + count

        for event in events:
            key = event.confidence.value
            diagnostics.events_by_confidence[key] = diagnostics.events_by_confidence.get(key, 0) + 1

    # Compute average certainty
    if certainty_count > 0:
        diagnostics.avg_certainty = (total_certainty / Decimal(certainty_count)).quantize(Decimal("0.001"))

    # Coverage state
    if diagnostics.tickers_with_events == diagnostics.tickers_analyzed:
        diagnostics.coverage_state = "FULL"
    elif diagnostics.tickers_with_events > 0:
        diagnostics.coverage_state = "PARTIAL"
    else:
        diagnostics.coverage_state = "NONE"

    diagnostics.coverage_pct = (
        Decimal(diagnostics.tickers_with_events) / Decimal(diagnostics.tickers_analyzed) * Decimal("100")
    ).quantize(Decimal("0.1")) if diagnostics.tickers_analyzed > 0 else Decimal("0")

    logger.info(f"Scoring complete. Generated {len(summaries)} summaries")

    return (summaries, diagnostics)
