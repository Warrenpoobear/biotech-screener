#!/usr/bin/env python3
"""
event_detector.py - CT.gov Catalyst Event Detection

Detects status changes, timeline shifts, and date confirmations from trial deltas.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional, Protocol
from enum import Enum
import logging

from ctgov_adapter import CanonicalTrialRecord, CTGovStatus, CompletionType

logger = logging.getLogger(__name__)


# ============================================================================
# MARKET CALENDAR PROTOCOL
# ============================================================================

class MarketCalendar(Protocol):
    """Interface for market calendar"""
    def next_trading_day(self, d: date) -> date:
        """Returns next NYSE trading day after given date"""
        ...


class SimpleMarketCalendar:
    """Simple market calendar that skips weekends (for testing)"""
    def next_trading_day(self, d: date) -> date:
        """Skip weekends, no holiday handling"""
        next_day = d + timedelta(days=1)
        while next_day.weekday() >= 5:  # Saturday=5, Sunday=6
            next_day += timedelta(days=1)
        return next_day


# ============================================================================
# EVENT TYPES
# ============================================================================

class EventType(Enum):
    """Catalyst event types from CT.gov deltas"""
    CT_STATUS_SEVERE_NEG = "CT_STATUS_SEVERE_NEG"
    CT_STATUS_DOWNGRADE = "CT_STATUS_DOWNGRADE"
    CT_STATUS_UPGRADE = "CT_STATUS_UPGRADE"
    CT_TIMELINE_PUSHOUT = "CT_TIMELINE_PUSHOUT"
    CT_TIMELINE_PULLIN = "CT_TIMELINE_PULLIN"
    CT_DATE_CONFIRMED_ACTUAL = "CT_DATE_CONFIRMED_ACTUAL"
    CT_RESULTS_POSTED = "CT_RESULTS_POSTED"
    # Activity proxy: trial was updated but no specific status/date change detected
    # Used as historical data workaround when CT.gov API only shows current state
    CT_ACTIVITY_PROXY = "CT_ACTIVITY_PROXY"


# ============================================================================
# CATALYST EVENT
# ============================================================================

@dataclass
class CatalystEvent:
    """Single catalyst event from CT.gov delta with full explainability"""
    source: str = "CTGOV"
    nct_id: str = ""
    event_type: EventType = EventType.CT_STATUS_UPGRADE
    direction: str = "POS"  # 'POS', 'NEG', 'NEUTRAL'
    impact: int = 1  # 1-3
    confidence: float = 0.85
    disclosed_at: date = date.today()
    fields_changed: dict = None
    actual_date: Optional[date] = None
    # Explainability fields (new)
    event_rule_id: str = ""  # e.g., "M3_DIFF_STATUS_SEVERE_NEG"
    confidence_reason: str = ""  # Human-readable explanation

    def __post_init__(self):
        if self.fields_changed is None:
            self.fields_changed = {}
        # Auto-generate rule_id if not set
        if not self.event_rule_id:
            self.event_rule_id = f"M3_DIFF_{self.event_type.name}"

    def effective_trading_date(self, calendar: MarketCalendar) -> date:
        """
        Conservative: treat disclosed_at as effective next trading day
        Prevents same-day lookahead and handles weekends/holidays
        """
        return calendar.next_trading_day(self.disclosed_at)

    def days_to_event(self, as_of_date: date, calendar: MarketCalendar) -> int:
        """Returns unsigned days to event (for nearest_*_days calculation)"""
        effective_date = self.effective_trading_date(calendar)
        return max(1, (effective_date - as_of_date).days)

    def to_dict(self) -> dict:
        """Serialize for JSON with full explainability"""
        return {
            'source': self.source,
            'nct_id': self.nct_id,
            'event_type': self.event_type.value,
            'direction': self.direction,
            'impact': self.impact,
            'confidence': self.confidence,
            'disclosed_at': self.disclosed_at.isoformat(),
            'fields_changed': self.fields_changed,
            'actual_date': self.actual_date.isoformat() if self.actual_date else None,
            'event_rule_id': self.event_rule_id,
            'confidence_reason': self.confidence_reason,
        }


# ============================================================================
# EVENT CLASSIFICATION
# ============================================================================

def classify_status_change(
    old_status: CTGovStatus, 
    new_status: CTGovStatus
) -> tuple[Optional[EventType], int, str]:
    """
    Classify status change event
    
    Returns: (event_type, impact, direction)
    """
    # Severe negative: trial stopped
    if new_status in {CTGovStatus.SUSPENDED, CTGovStatus.TERMINATED, CTGovStatus.WITHDRAWN}:
        return (EventType.CT_STATUS_SEVERE_NEG, 3, 'NEG')
    
    # Directional change based on status ordering
    if new_status.value < old_status.value:
        # Downgrade
        delta = old_status.value - new_status.value
        impact = min(3, 1 + delta // 2)
        return (EventType.CT_STATUS_DOWNGRADE, impact, 'NEG')
    
    elif new_status.value > old_status.value:
        # Upgrade
        delta = new_status.value - old_status.value
        impact = min(3, 1 + delta // 2)
        return (EventType.CT_STATUS_UPGRADE, impact, 'POS')
    
    else:
        # No change
        return (None, 0, 'NEUTRAL')


def classify_timeline_change(
    old_date: date,
    new_date: date,
    noise_band_days: int = 14
) -> tuple[Optional[EventType], int, str]:
    """
    Classify timeline change event
    
    Returns: (event_type, impact, direction)
    Ignores changes <14 days as noise
    """
    delta_days = (new_date - old_date).days
    
    if abs(delta_days) < noise_band_days:
        return (None, 0, 'NEUTRAL')
    
    # Severity scaling
    if abs(delta_days) < 60:
        impact = 1
    elif abs(delta_days) < 180:
        impact = 2
    else:
        impact = 3
    
    if delta_days >= noise_band_days:
        return (EventType.CT_TIMELINE_PUSHOUT, impact, 'NEG')
    else:  # delta_days <= -noise_band_days
        return (EventType.CT_TIMELINE_PULLIN, impact, 'POS')


def classify_date_confirmation(
    old_type: Optional[CompletionType],
    new_type: Optional[CompletionType],
    actual_date: date,
    as_of_date: date,
    recency_threshold_days: int = 90
) -> tuple[Optional[EventType], int, str, Optional[date]]:
    """
    Classify date confirmation event
    
    Returns: (event_type, impact, direction, actual_date)
    """
    # Detect ANTICIPATED/ESTIMATED → ACTUAL transitions
    if (old_type in {CompletionType.ANTICIPATED, CompletionType.ESTIMATED} and 
        new_type == CompletionType.ACTUAL):
        
        days_since_actual = (as_of_date - actual_date).days
        
        if days_since_actual <= recency_threshold_days:
            return (EventType.CT_DATE_CONFIRMED_ACTUAL, 1, 'POS', actual_date)
    
    return (None, 0, 'NEUTRAL', None)


def classify_results_posted(
    old_results_date: Optional[date],
    new_results_date: Optional[date]
) -> tuple[Optional[EventType], int, str]:
    """
    Classify results posted event
    
    Returns: (event_type, impact, direction)
    """
    if old_results_date is None and new_results_date is not None:
        return (EventType.CT_RESULTS_POSTED, 1, 'NEUTRAL')
    elif (old_results_date != new_results_date and 
          new_results_date is not None):
        return (EventType.CT_RESULTS_POSTED, 1, 'NEUTRAL')
    
    return (None, 0, 'NEUTRAL')


# ============================================================================
# EVENT DETECTOR
# ============================================================================

@dataclass
class EventDetectorConfig:
    """Configuration for event detection"""
    noise_band_days: int = 14
    recency_threshold_days: int = 90
    confidence_scores: dict = None
    
    def __post_init__(self):
        if self.confidence_scores is None:
            self.confidence_scores = {
                EventType.CT_STATUS_SEVERE_NEG: 0.95,
                EventType.CT_STATUS_DOWNGRADE: 0.85,
                EventType.CT_STATUS_UPGRADE: 0.80,
                EventType.CT_TIMELINE_PUSHOUT: 0.75,
                EventType.CT_TIMELINE_PULLIN: 0.70,
                EventType.CT_DATE_CONFIRMED_ACTUAL: 0.85,
                EventType.CT_RESULTS_POSTED: 0.90,
                EventType.CT_ACTIVITY_PROXY: 0.30,  # Low confidence - unknown change type
            }


class EventDetector:
    """Detects catalyst events from trial record deltas"""
    
    def __init__(self, config: EventDetectorConfig = EventDetectorConfig()):
        self.config = config
    
    def detect_events(
        self,
        current_record: CanonicalTrialRecord,
        prior_record: Optional[CanonicalTrialRecord],
        as_of_date: date
    ) -> list[CatalystEvent]:
        """
        Detect all catalyst events for a single trial
        
        Args:
            current_record: Current state
            prior_record: Prior state (None if first ingest)
            as_of_date: Date for PIT validation
        
        Returns:
            List of catalyst events
        """
        events = []
        
        # If no prior record, this is initial ingest (not a delta)
        if prior_record is None:
            return []
        
        disclosed_at = current_record.last_update_posted
        
        # Status change detection
        if current_record.overall_status != prior_record.overall_status:
            event_type, impact, direction = classify_status_change(
                prior_record.overall_status,
                current_record.overall_status
            )
            if event_type:
                # Generate confidence reason
                if event_type == EventType.CT_STATUS_SEVERE_NEG:
                    confidence_reason = f"Trial status changed to {current_record.overall_status.name} (terminal negative)"
                elif direction == 'NEG':
                    confidence_reason = f"Status downgrade: {prior_record.overall_status.name} → {current_record.overall_status.name}"
                else:
                    confidence_reason = f"Status upgrade: {prior_record.overall_status.name} → {current_record.overall_status.name}"

                events.append(CatalystEvent(
                    nct_id=current_record.nct_id,
                    event_type=event_type,
                    direction=direction,
                    impact=impact,
                    confidence=self.config.confidence_scores[event_type],
                    disclosed_at=disclosed_at,
                    fields_changed={
                        'overallStatus': [
                            prior_record.overall_status.name,
                            current_record.overall_status.name
                        ]
                    },
                    confidence_reason=confidence_reason,
                ))
        
        # Primary completion date change
        if (current_record.primary_completion_date and
            prior_record.primary_completion_date and
            current_record.primary_completion_date != prior_record.primary_completion_date):

            event_type, impact, direction = classify_timeline_change(
                prior_record.primary_completion_date,
                current_record.primary_completion_date,
                self.config.noise_band_days
            )
            if event_type:
                delta_days = (current_record.primary_completion_date - prior_record.primary_completion_date).days
                if delta_days > 0:
                    confidence_reason = f"Primary completion date pushed out by {delta_days} days"
                else:
                    confidence_reason = f"Primary completion date pulled in by {abs(delta_days)} days"

                events.append(CatalystEvent(
                    nct_id=current_record.nct_id,
                    event_type=event_type,
                    direction=direction,
                    impact=impact,
                    confidence=self.config.confidence_scores[event_type],
                    disclosed_at=disclosed_at,
                    fields_changed={
                        'primaryCompletionDate': [
                            prior_record.primary_completion_date.isoformat(),
                            current_record.primary_completion_date.isoformat()
                        ]
                    },
                    confidence_reason=confidence_reason,
                ))
        
        # Study completion date change
        if (current_record.completion_date and
            prior_record.completion_date and
            current_record.completion_date != prior_record.completion_date):

            event_type, impact, direction = classify_timeline_change(
                prior_record.completion_date,
                current_record.completion_date,
                self.config.noise_band_days
            )
            if event_type:
                delta_days = (current_record.completion_date - prior_record.completion_date).days
                if delta_days > 0:
                    confidence_reason = f"Study completion date pushed out by {delta_days} days"
                else:
                    confidence_reason = f"Study completion date pulled in by {abs(delta_days)} days"

                events.append(CatalystEvent(
                    nct_id=current_record.nct_id,
                    event_type=event_type,
                    direction=direction,
                    impact=impact,
                    confidence=self.config.confidence_scores[event_type],
                    disclosed_at=disclosed_at,
                    fields_changed={
                        'completionDate': [
                            prior_record.completion_date.isoformat(),
                            current_record.completion_date.isoformat()
                        ]
                    },
                    confidence_reason=confidence_reason,
                ))

        # Primary completion type change (ANTICIPATED → ACTUAL)
        if (current_record.primary_completion_type != prior_record.primary_completion_type and
            current_record.primary_completion_date):

            event_type, impact, direction, actual_date = classify_date_confirmation(
                prior_record.primary_completion_type,
                current_record.primary_completion_type,
                current_record.primary_completion_date,
                as_of_date,
                self.config.recency_threshold_days
            )
            if event_type:
                prior_type = prior_record.primary_completion_type.value if prior_record.primary_completion_type else 'None'
                current_type = current_record.primary_completion_type.value if current_record.primary_completion_type else 'None'
                confidence_reason = f"Primary completion type changed: {prior_type} → {current_type}"

                events.append(CatalystEvent(
                    nct_id=current_record.nct_id,
                    event_type=event_type,
                    direction=direction,
                    impact=impact,
                    confidence=self.config.confidence_scores[event_type],
                    disclosed_at=disclosed_at,
                    actual_date=actual_date,
                    fields_changed={
                        'primaryCompletionType': [
                            prior_record.primary_completion_type.value if prior_record.primary_completion_type else None,
                            current_record.primary_completion_type.value if current_record.primary_completion_type else None
                        ]
                    },
                    confidence_reason=confidence_reason,
                ))

        # Study completion type change
        if (current_record.completion_type != prior_record.completion_type and
            current_record.completion_date):

            event_type, impact, direction, actual_date = classify_date_confirmation(
                prior_record.completion_type,
                current_record.completion_type,
                current_record.completion_date,
                as_of_date,
                self.config.recency_threshold_days
            )
            if event_type:
                prior_type = prior_record.completion_type.value if prior_record.completion_type else 'None'
                current_type = current_record.completion_type.value if current_record.completion_type else 'None'
                confidence_reason = f"Study completion type changed: {prior_type} → {current_type}"

                events.append(CatalystEvent(
                    nct_id=current_record.nct_id,
                    event_type=event_type,
                    direction=direction,
                    impact=impact,
                    confidence=self.config.confidence_scores[event_type],
                    disclosed_at=disclosed_at,
                    actual_date=actual_date,
                    fields_changed={
                        'completionType': [
                            prior_record.completion_type.value if prior_record.completion_type else None,
                            current_record.completion_type.value if current_record.completion_type else None
                        ]
                    },
                    confidence_reason=confidence_reason,
                ))

        # Results posted detection
        if current_record.results_first_posted != prior_record.results_first_posted:
            event_type, impact, direction = classify_results_posted(
                prior_record.results_first_posted,
                current_record.results_first_posted
            )
            if event_type:
                if prior_record.results_first_posted is None:
                    confidence_reason = f"Results first posted on {current_record.results_first_posted}"
                else:
                    confidence_reason = f"Results posting date updated: {prior_record.results_first_posted} → {current_record.results_first_posted}"

                events.append(CatalystEvent(
                    nct_id=current_record.nct_id,
                    event_type=event_type,
                    direction=direction,
                    impact=impact,
                    confidence=self.config.confidence_scores[event_type],
                    disclosed_at=disclosed_at,
                    fields_changed={
                        'resultsFirstPosted': [
                            prior_record.results_first_posted.isoformat() if prior_record.results_first_posted else None,
                            current_record.results_first_posted.isoformat() if current_record.results_first_posted else None
                        ]
                    },
                    confidence_reason=confidence_reason,
                ))

        # Activity proxy detection: trial was updated but no specific event type detected
        # This is a historical data workaround - CT.gov API only shows current state,
        # so we can't know what specifically changed. But an update indicates engagement.
        if (len(events) == 0 and
            current_record.last_update_posted is not None and
            prior_record.last_update_posted is not None and
            current_record.last_update_posted != prior_record.last_update_posted):
            # Only generate activity proxy if no other events were detected
            days_since_update = (as_of_date - current_record.last_update_posted).days

            # Only count updates within 90 days as meaningful activity
            if days_since_update <= 90:
                # Scale impact by recency (more recent = higher impact)
                if days_since_update <= 7:
                    impact = 2
                elif days_since_update <= 30:
                    impact = 1
                else:
                    impact = 1

                confidence_reason = (
                    f"Trial record updated {days_since_update} days ago "
                    f"({prior_record.last_update_posted} → {current_record.last_update_posted}) "
                    f"but no specific status/date change detected. May indicate "
                    f"administrative update, protocol amendment, or enrollment activity."
                )

                events.append(CatalystEvent(
                    nct_id=current_record.nct_id,
                    event_type=EventType.CT_ACTIVITY_PROXY,
                    direction='NEUTRAL',
                    impact=impact,
                    confidence=self.config.confidence_scores[EventType.CT_ACTIVITY_PROXY],
                    disclosed_at=disclosed_at,
                    fields_changed={
                        'lastUpdatePosted': [
                            prior_record.last_update_posted.isoformat() if prior_record.last_update_posted else None,
                            current_record.last_update_posted.isoformat() if current_record.last_update_posted else None
                        ]
                    },
                    confidence_reason=confidence_reason,
                ))

        return events


# ============================================================================
# EVENT SCORING
# ============================================================================

def compute_event_score(
    event: CatalystEvent,
    as_of_date: date,
    calendar: MarketCalendar,
    decay_constant: float = 30.0
) -> float:
    """
    Score = impact × confidence × proximity
    
    Proximity decay:
    - CT.gov deltas are "now" events (proximity = 1.0 at disclosure)
    - Date confirmations decay from actual_date, not disclosed_at
    """
    if event.event_type == EventType.CT_DATE_CONFIRMED_ACTUAL:
        if event.actual_date is None:
            logger.warning(f"CT_DATE_CONFIRMED_ACTUAL missing actual_date for {event.nct_id}")
            proximity = 1.0
        else:
            days_since_actual = (as_of_date - event.actual_date).days
            proximity = 1.0 / (1.0 + days_since_actual / decay_constant)
    else:
        # For other events, full proximity at disclosure
        proximity = 1.0
    
    return event.impact * event.confidence * proximity


# ============================================================================
# ACTIVITY PROXY DETECTION (HISTORICAL DATA WORKAROUND)
# ============================================================================

def detect_activity_proxy_from_lookback(
    trials: list[CanonicalTrialRecord],
    as_of_date: date,
    lookback_days: int = 90
) -> dict[str, list[CatalystEvent]]:
    """
    Detect activity proxy events from last_update_posted within lookback window.

    This is a workaround for lack of historical CT.gov snapshots. When we can't
    determine what changed, we can at least identify trials with recent activity.

    Args:
        trials: List of canonical trial records for a single ticker
        as_of_date: Reference date for lookback calculation
        lookback_days: Number of days to look back (default 90)

    Returns:
        Dict mapping ticker to list of activity proxy events

    Note: This should be used in addition to diff-based detection when
    historical snapshots are incomplete or unavailable.
    """
    cutoff_date = as_of_date - timedelta(days=lookback_days)
    events_by_ticker: dict[str, list[CatalystEvent]] = {}

    for trial in trials:
        if not trial.last_update_posted:
            continue

        if trial.last_update_posted < cutoff_date:
            continue

        days_since_update = (as_of_date - trial.last_update_posted).days

        # Scale impact by recency and phase (if available)
        if days_since_update <= 7:
            impact = 2
        elif days_since_update <= 30:
            impact = 1
        else:
            impact = 1

        # Time decay factor for scoring
        decay = max(0.1, 1.0 - (days_since_update / lookback_days))

        event = CatalystEvent(
            nct_id=trial.nct_id,
            event_type=EventType.CT_ACTIVITY_PROXY,
            direction='NEUTRAL',
            impact=impact,
            confidence=0.30 * decay,  # Decayed confidence
            disclosed_at=trial.last_update_posted,
            fields_changed={
                'lastUpdatePosted': [None, trial.last_update_posted.isoformat()],
                'days_ago': days_since_update,
            },
            confidence_reason=(
                f"Trial updated {days_since_update} days ago. "
                f"Activity detected via last_update_posted (historical proxy)."
            ),
            event_rule_id="M3_ACTIVITY_PROXY_LOOKBACK",
        )

        ticker = trial.ticker
        if ticker not in events_by_ticker:
            events_by_ticker[ticker] = []
        events_by_ticker[ticker].append(event)

    return events_by_ticker


def compute_activity_proxy_score(
    trials: list[CanonicalTrialRecord],
    as_of_date: date,
    lookback_days: int = 90,
    phase_weights: dict[str, float] = None
) -> dict[str, any]:
    """
    Compute activity proxy score based on recent trial updates.

    This supplements diff-based scoring when historical data is limited.

    Args:
        trials: List of canonical trial records
        as_of_date: Reference date
        lookback_days: Lookback window (default 90)
        phase_weights: Optional phase-based weighting

    Returns:
        Dict with:
        - activity_proxy_score: Aggregated score
        - activity_count_90d: Trials updated in past 90 days
        - activity_count_30d: Trials updated in past 30 days
        - recent_nct_ids: List of NCT IDs with recent activity
    """
    if phase_weights is None:
        phase_weights = {
            'PHASE3': 10.0,
            'PHASE2': 7.0,
            'PHASE1': 4.0,
            'PHASE4': 3.0,  # Post-marketing
            'NA': 2.0,
        }

    cutoff_90d = as_of_date - timedelta(days=90)
    cutoff_30d = as_of_date - timedelta(days=30)

    activity_count_90d = 0
    activity_count_30d = 0
    total_score = 0.0
    recent_nct_ids = []

    for trial in trials:
        if not trial.last_update_posted:
            continue

        if trial.last_update_posted < cutoff_90d:
            continue

        activity_count_90d += 1
        recent_nct_ids.append(trial.nct_id)

        if trial.last_update_posted >= cutoff_30d:
            activity_count_30d += 1

        days_ago = (as_of_date - trial.last_update_posted).days

        # Time decay (exponential)
        decay = max(0.1, 1.0 - (days_ago / lookback_days))

        # Phase-based weighting (extract phase from trial if available)
        phase_key = 'NA'  # Default
        if hasattr(trial, 'phase') and trial.phase:
            phase_str = str(trial.phase).upper()
            if 'PHASE3' in phase_str or 'PHASE 3' in phase_str:
                phase_key = 'PHASE3'
            elif 'PHASE2' in phase_str or 'PHASE 2' in phase_str:
                phase_key = 'PHASE2'
            elif 'PHASE1' in phase_str or 'PHASE 1' in phase_str:
                phase_key = 'PHASE1'
            elif 'PHASE4' in phase_str or 'PHASE 4' in phase_str:
                phase_key = 'PHASE4'

        base_score = phase_weights.get(phase_key, 2.0)
        trial_score = base_score * decay

        total_score += trial_score

    return {
        'activity_proxy_score': round(total_score, 2),
        'activity_count_90d': activity_count_90d,
        'activity_count_30d': activity_count_30d,
        'recent_nct_ids': recent_nct_ids,
    }


if __name__ == "__main__":
    print("Event Detector loaded successfully")
    print("Use EventDetector.detect_events() to detect catalyst events")
