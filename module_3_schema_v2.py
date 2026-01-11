#!/usr/bin/env python3
"""
module_3_schema_v2.py - Module 3 Catalyst Event Schema v2 (Robust)

Enhanced schema with:
- Atomic events with stable event_id (sha256 of deterministic fields)
- Full PIT tracking with source_date and pit_date_field_used
- Certainty scoring framework
- Negative catalyst taxonomy
- Enhanced diagnostics

Design Philosophy (Governed Over Smart):
- DETERMINISTIC: All hashes from stable fields, no randomness
- PIT-SAFE: Every event has source_date and pit_date_field_used
- STDLIB-ONLY: No external dependencies
- FAIL LOUDLY: Validation on all inputs

Version: 2.0.0 - Robust Edition
"""

import hashlib
import json
import math
from dataclasses import dataclass, field, asdict
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple, Set


# =============================================================================
# VERSION CONSTANTS
# =============================================================================

SCHEMA_VERSION = "m3catalyst_v2_20260111"
SCORE_VERSION = "m3score_v2_20260111"


# =============================================================================
# EVENT TYPE TAXONOMY (EXPANDED WITH NEGATIVE CATALYSTS)
# =============================================================================

class EventType(str, Enum):
    """
    Catalyst event types from CT.gov deltas.

    Expanded taxonomy with explicit negative catalyst categories.
    """
    # === POSITIVE STATUS CHANGES ===
    CT_STATUS_UPGRADE = "CT_STATUS_UPGRADE"
    CT_ENROLLMENT_STARTED = "CT_ENROLLMENT_STARTED"
    CT_ENROLLMENT_COMPLETE = "CT_ENROLLMENT_COMPLETE"
    CT_ENROLLMENT_RESUMED = "CT_ENROLLMENT_RESUMED"

    # === POSITIVE TIMELINE CHANGES ===
    CT_TIMELINE_PULLIN = "CT_TIMELINE_PULLIN"
    CT_DATE_CONFIRMED_ACTUAL = "CT_DATE_CONFIRMED_ACTUAL"

    # === CRITICAL POSITIVE (MILESTONE) ===
    CT_PRIMARY_COMPLETION = "CT_PRIMARY_COMPLETION"
    CT_STUDY_COMPLETION = "CT_STUDY_COMPLETION"
    CT_RESULTS_POSTED = "CT_RESULTS_POSTED"

    # === NEUTRAL (INFORMATIONAL) ===
    CT_PROTOCOL_AMENDMENT = "CT_PROTOCOL_AMENDMENT"
    CT_ARM_ADDED = "CT_ARM_ADDED"
    CT_ENDPOINT_CHANGED = "CT_ENDPOINT_CHANGED"

    # === NEGATIVE STATUS CHANGES ===
    CT_STATUS_DOWNGRADE = "CT_STATUS_DOWNGRADE"
    CT_ARM_REMOVED = "CT_ARM_REMOVED"
    CT_ENROLLMENT_PAUSED = "CT_ENROLLMENT_PAUSED"

    # === NEGATIVE TIMELINE CHANGES ===
    CT_TIMELINE_PUSHOUT = "CT_TIMELINE_PUSHOUT"
    CT_TIMELINE_SLIP = "CT_TIMELINE_SLIP"  # Date moved later
    CT_WINDOW_WIDENING = "CT_WINDOW_WIDENING"  # Completion window expanded

    # === SEVERE NEGATIVE (TERMINATIONS/HOLDS) ===
    CT_STATUS_SEVERE_NEG = "CT_STATUS_SEVERE_NEG"
    CT_TRIAL_TERMINATED = "CT_TRIAL_TERMINATED"
    CT_TRIAL_WITHDRAWN = "CT_TRIAL_WITHDRAWN"
    CT_TRIAL_SUSPENDED = "CT_TRIAL_SUSPENDED"
    CT_CLINICAL_HOLD = "CT_CLINICAL_HOLD"  # FDA clinical hold
    CT_CRL_RECEIVED = "CT_CRL_RECEIVED"  # Complete Response Letter

    # === UNKNOWN ===
    UNKNOWN = "UNKNOWN"


class EventSeverity(str, Enum):
    """Event severity categories for scoring."""
    CRITICAL_POSITIVE = "CRITICAL_POSITIVE"
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"
    SEVERE_NEGATIVE = "SEVERE_NEGATIVE"


class ConfidenceLevel(str, Enum):
    """Confidence levels for events."""
    HIGH = "HIGH"
    MED = "MED"
    LOW = "LOW"


class CatalystWindowBucket(str, Enum):
    """Catalyst window time buckets."""
    DAYS_0_30 = "0_30"
    DAYS_31_90 = "31_90"
    DAYS_91_180 = "91_180"
    DAYS_181_365 = "181_365"
    DAYS_GT_365 = ">365"
    UNKNOWN = "UNKNOWN"


class SourceReliability(str, Enum):
    """Source reliability for certainty scoring."""
    OFFICIAL = "OFFICIAL"  # CT.gov official status
    REGULATORY = "REGULATORY"  # FDA, EMA filings
    COMPANY = "COMPANY"  # Press release, SEC filing
    INFERRED = "INFERRED"  # Computed from delta
    UNKNOWN = "UNKNOWN"


class DateSpecificity(str, Enum):
    """Date specificity for certainty scoring."""
    EXACT = "EXACT"  # Exact date known
    MONTH = "MONTH"  # Month-level precision
    QUARTER = "QUARTER"  # Quarter-level
    YEAR = "YEAR"  # Year-level only
    UNKNOWN = "UNKNOWN"


# =============================================================================
# EVENT TYPE → SEVERITY MAPPING (EXPANDED)
# =============================================================================

EVENT_SEVERITY_MAP: Dict[EventType, EventSeverity] = {
    # Critical positive
    EventType.CT_PRIMARY_COMPLETION: EventSeverity.CRITICAL_POSITIVE,
    EventType.CT_STUDY_COMPLETION: EventSeverity.POSITIVE,
    EventType.CT_RESULTS_POSTED: EventSeverity.CRITICAL_POSITIVE,

    # Positive
    EventType.CT_STATUS_UPGRADE: EventSeverity.POSITIVE,
    EventType.CT_TIMELINE_PULLIN: EventSeverity.POSITIVE,
    EventType.CT_DATE_CONFIRMED_ACTUAL: EventSeverity.POSITIVE,
    EventType.CT_ENROLLMENT_COMPLETE: EventSeverity.POSITIVE,
    EventType.CT_ENROLLMENT_STARTED: EventSeverity.POSITIVE,
    EventType.CT_ENROLLMENT_RESUMED: EventSeverity.POSITIVE,

    # Neutral
    EventType.CT_ARM_ADDED: EventSeverity.NEUTRAL,
    EventType.CT_PROTOCOL_AMENDMENT: EventSeverity.NEUTRAL,
    EventType.CT_ENDPOINT_CHANGED: EventSeverity.NEUTRAL,
    EventType.UNKNOWN: EventSeverity.NEUTRAL,

    # Negative
    EventType.CT_STATUS_DOWNGRADE: EventSeverity.NEGATIVE,
    EventType.CT_TIMELINE_PUSHOUT: EventSeverity.NEGATIVE,
    EventType.CT_TIMELINE_SLIP: EventSeverity.NEGATIVE,
    EventType.CT_WINDOW_WIDENING: EventSeverity.NEGATIVE,
    EventType.CT_ARM_REMOVED: EventSeverity.NEGATIVE,
    EventType.CT_ENROLLMENT_PAUSED: EventSeverity.NEGATIVE,

    # Severe negative
    EventType.CT_STATUS_SEVERE_NEG: EventSeverity.SEVERE_NEGATIVE,
    EventType.CT_TRIAL_TERMINATED: EventSeverity.SEVERE_NEGATIVE,
    EventType.CT_TRIAL_WITHDRAWN: EventSeverity.SEVERE_NEGATIVE,
    EventType.CT_TRIAL_SUSPENDED: EventSeverity.SEVERE_NEGATIVE,
    EventType.CT_CLINICAL_HOLD: EventSeverity.SEVERE_NEGATIVE,
    EventType.CT_CRL_RECEIVED: EventSeverity.SEVERE_NEGATIVE,
}


# =============================================================================
# NEGATIVE CATALYST TAXONOMY
# =============================================================================

NEGATIVE_CATALYST_TYPES: Set[EventType] = {
    EventType.CT_STATUS_DOWNGRADE,
    EventType.CT_TIMELINE_PUSHOUT,
    EventType.CT_TIMELINE_SLIP,
    EventType.CT_WINDOW_WIDENING,
    EventType.CT_ARM_REMOVED,
    EventType.CT_ENROLLMENT_PAUSED,
    EventType.CT_STATUS_SEVERE_NEG,
    EventType.CT_TRIAL_TERMINATED,
    EventType.CT_TRIAL_WITHDRAWN,
    EventType.CT_TRIAL_SUSPENDED,
    EventType.CT_CLINICAL_HOLD,
    EventType.CT_CRL_RECEIVED,
}

SEVERE_NEGATIVE_TYPES: Set[EventType] = {
    EventType.CT_STATUS_SEVERE_NEG,
    EventType.CT_TRIAL_TERMINATED,
    EventType.CT_TRIAL_WITHDRAWN,
    EventType.CT_TRIAL_SUSPENDED,
    EventType.CT_CLINICAL_HOLD,
    EventType.CT_CRL_RECEIVED,
}


# =============================================================================
# CONFIDENCE + CERTAINTY WEIGHTS
# =============================================================================

CONFIDENCE_WEIGHTS: Dict[ConfidenceLevel, Decimal] = {
    ConfidenceLevel.HIGH: Decimal("1.0"),
    ConfidenceLevel.MED: Decimal("0.6"),
    ConfidenceLevel.LOW: Decimal("0.3"),
}

SOURCE_RELIABILITY_WEIGHTS: Dict[SourceReliability, Decimal] = {
    SourceReliability.OFFICIAL: Decimal("1.0"),
    SourceReliability.REGULATORY: Decimal("0.95"),
    SourceReliability.COMPANY: Decimal("0.8"),
    SourceReliability.INFERRED: Decimal("0.5"),
    SourceReliability.UNKNOWN: Decimal("0.3"),
}

DATE_SPECIFICITY_WEIGHTS: Dict[DateSpecificity, Decimal] = {
    DateSpecificity.EXACT: Decimal("1.0"),
    DateSpecificity.MONTH: Decimal("0.8"),
    DateSpecificity.QUARTER: Decimal("0.6"),
    DateSpecificity.YEAR: Decimal("0.4"),
    DateSpecificity.UNKNOWN: Decimal("0.2"),
}


# =============================================================================
# SEVERITY → SCORE CONTRIBUTION
# =============================================================================

SEVERITY_SCORE_CONTRIBUTION: Dict[EventSeverity, Decimal] = {
    EventSeverity.CRITICAL_POSITIVE: Decimal("15.0"),
    EventSeverity.POSITIVE: Decimal("8.0"),
    EventSeverity.NEUTRAL: Decimal("0.0"),
    EventSeverity.NEGATIVE: Decimal("-5.0"),
    EventSeverity.SEVERE_NEGATIVE: Decimal("-20.0"),
}


# =============================================================================
# EVENT TYPE → PROXIMITY WEIGHT
# =============================================================================

EVENT_TYPE_WEIGHT: Dict[EventType, Decimal] = {
    # High-impact
    EventType.CT_PRIMARY_COMPLETION: Decimal("20.0"),
    EventType.CT_RESULTS_POSTED: Decimal("18.0"),
    EventType.CT_STUDY_COMPLETION: Decimal("15.0"),
    EventType.CT_DATE_CONFIRMED_ACTUAL: Decimal("12.0"),

    # Medium-impact
    EventType.CT_ENROLLMENT_COMPLETE: Decimal("10.0"),
    EventType.CT_STATUS_UPGRADE: Decimal("8.0"),
    EventType.CT_TIMELINE_PULLIN: Decimal("6.0"),

    # Lower-impact
    EventType.CT_ENROLLMENT_STARTED: Decimal("5.0"),
    EventType.CT_ENROLLMENT_RESUMED: Decimal("4.0"),
    EventType.CT_ARM_ADDED: Decimal("3.0"),
    EventType.CT_PROTOCOL_AMENDMENT: Decimal("2.0"),
    EventType.CT_ENDPOINT_CHANGED: Decimal("1.0"),

    # Negative events (contribute to negative score)
    EventType.CT_STATUS_SEVERE_NEG: Decimal("-20.0"),
    EventType.CT_TRIAL_TERMINATED: Decimal("-20.0"),
    EventType.CT_TRIAL_SUSPENDED: Decimal("-18.0"),
    EventType.CT_CLINICAL_HOLD: Decimal("-18.0"),
    EventType.CT_CRL_RECEIVED: Decimal("-15.0"),
    EventType.CT_TRIAL_WITHDRAWN: Decimal("-12.0"),
    EventType.CT_STATUS_DOWNGRADE: Decimal("-8.0"),
    EventType.CT_TIMELINE_PUSHOUT: Decimal("-5.0"),
    EventType.CT_TIMELINE_SLIP: Decimal("-5.0"),
    EventType.CT_WINDOW_WIDENING: Decimal("-3.0"),
    EventType.CT_ARM_REMOVED: Decimal("-3.0"),
    EventType.CT_ENROLLMENT_PAUSED: Decimal("-4.0"),

    EventType.UNKNOWN: Decimal("0.0"),
}


# =============================================================================
# DEFAULT CONFIDENCE BY EVENT TYPE
# =============================================================================

EVENT_DEFAULT_CONFIDENCE: Dict[EventType, ConfidenceLevel] = {
    EventType.CT_STATUS_SEVERE_NEG: ConfidenceLevel.HIGH,
    EventType.CT_TRIAL_TERMINATED: ConfidenceLevel.HIGH,
    EventType.CT_TRIAL_WITHDRAWN: ConfidenceLevel.HIGH,
    EventType.CT_TRIAL_SUSPENDED: ConfidenceLevel.HIGH,
    EventType.CT_CLINICAL_HOLD: ConfidenceLevel.HIGH,
    EventType.CT_CRL_RECEIVED: ConfidenceLevel.HIGH,
    EventType.CT_STATUS_DOWNGRADE: ConfidenceLevel.HIGH,
    EventType.CT_STATUS_UPGRADE: ConfidenceLevel.HIGH,
    EventType.CT_TIMELINE_PUSHOUT: ConfidenceLevel.MED,
    EventType.CT_TIMELINE_PULLIN: ConfidenceLevel.MED,
    EventType.CT_TIMELINE_SLIP: ConfidenceLevel.MED,
    EventType.CT_WINDOW_WIDENING: ConfidenceLevel.MED,
    EventType.CT_DATE_CONFIRMED_ACTUAL: ConfidenceLevel.HIGH,
    EventType.CT_PRIMARY_COMPLETION: ConfidenceLevel.HIGH,
    EventType.CT_STUDY_COMPLETION: ConfidenceLevel.HIGH,
    EventType.CT_RESULTS_POSTED: ConfidenceLevel.HIGH,
    EventType.CT_PROTOCOL_AMENDMENT: ConfidenceLevel.MED,
    EventType.CT_ARM_ADDED: ConfidenceLevel.MED,
    EventType.CT_ARM_REMOVED: ConfidenceLevel.MED,
    EventType.CT_ENDPOINT_CHANGED: ConfidenceLevel.LOW,
    EventType.CT_ENROLLMENT_STARTED: ConfidenceLevel.HIGH,
    EventType.CT_ENROLLMENT_COMPLETE: ConfidenceLevel.HIGH,
    EventType.CT_ENROLLMENT_PAUSED: ConfidenceLevel.HIGH,
    EventType.CT_ENROLLMENT_RESUMED: ConfidenceLevel.HIGH,
    EventType.UNKNOWN: ConfidenceLevel.LOW,
}


# =============================================================================
# CATALYST EVENT V2 (ROBUST)
# =============================================================================

@dataclass(frozen=True)
class CatalystEventV2:
    """
    Atomic catalyst event with stable event_id and full PIT tracking.

    Key guarantees:
    - event_id: SHA256 hash of deterministic fields (stable across runs)
    - source_date: When the data was collected/observed
    - pit_date_field_used: Which date field was used for PIT filtering
    """
    ticker: str
    nct_id: str
    event_type: EventType
    event_severity: EventSeverity
    event_date: Optional[str]  # ISO date string, the actual/expected event date
    field_changed: str
    prior_value: Optional[str]
    new_value: Optional[str]
    source: str
    confidence: ConfidenceLevel
    disclosed_at: str  # ISO date string, when event was disclosed

    # === PIT FIELDS (REQUIRED) ===
    source_date: str  # ISO date: when data was collected
    pit_date_field_used: str  # e.g., "last_update_posted", "first_posted"

    # === CERTAINTY FIELDS ===
    source_reliability: SourceReliability = SourceReliability.INFERRED
    date_specificity: DateSpecificity = DateSpecificity.UNKNOWN
    corroboration_count: int = 0  # Number of corroborating sources

    @property
    def event_id(self) -> str:
        """
        Stable event ID from SHA256 of deterministic fields.

        Uses: ticker, nct_id, event_type, event_date, field_changed,
              prior_value, new_value, source

        Does NOT include: disclosed_at, source_date (vary by observation)
        """
        canonical_tuple = (
            self.ticker,
            self.nct_id,
            self.event_type.value,
            self.event_date or "",
            self.field_changed,
            self.prior_value or "",
            self.new_value or "",
            self.source,
        )
        canonical_str = "|".join(str(x) for x in canonical_tuple)
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    @property
    def event_id_short(self) -> str:
        """Short form of event_id (first 16 chars)."""
        return self.event_id[:16]

    @property
    def is_negative(self) -> bool:
        """Check if event is a negative catalyst."""
        return self.event_type in NEGATIVE_CATALYST_TYPES

    @property
    def is_severe_negative(self) -> bool:
        """Check if event is a severe negative catalyst."""
        return self.event_type in SEVERE_NEGATIVE_TYPES

    def compute_certainty_score(self, as_of_date: date) -> Decimal:
        """
        Compute certainty score from source reliability, date specificity,
        corroboration count, and staleness.

        Formula:
            certainty = source_weight * date_weight * corroboration_factor * freshness_factor

        Returns:
            Decimal between 0.0 and 1.0
        """
        # Source reliability weight
        source_weight = SOURCE_RELIABILITY_WEIGHTS.get(
            self.source_reliability, Decimal("0.3")
        )

        # Date specificity weight
        date_weight = DATE_SPECIFICITY_WEIGHTS.get(
            self.date_specificity, Decimal("0.2")
        )

        # Corroboration factor: 1.0 + 0.1 * min(corroboration_count, 3)
        corroboration_factor = Decimal("1.0") + Decimal("0.1") * Decimal(
            min(self.corroboration_count, 3)
        )

        # Staleness factor: decay based on source_date age
        try:
            source_d = date.fromisoformat(self.source_date)
            days_old = (as_of_date - source_d).days
            if days_old < 0:
                freshness_factor = Decimal("0")  # Future data = invalid
            elif days_old <= 7:
                freshness_factor = Decimal("1.0")
            elif days_old <= 30:
                freshness_factor = Decimal("0.9")
            elif days_old <= 90:
                freshness_factor = Decimal("0.7")
            elif days_old <= 180:
                freshness_factor = Decimal("0.5")
            else:
                freshness_factor = Decimal("0.3")
        except (ValueError, TypeError):
            freshness_factor = Decimal("0.5")

        certainty = source_weight * date_weight * corroboration_factor * freshness_factor
        return min(Decimal("1.0"), certainty.quantize(Decimal("0.001")))

    def sort_key(self) -> Tuple:
        """Deterministic sort key for event ordering."""
        return (
            self.event_date or "9999-99-99",
            self.event_type.value,
            self.nct_id,
            self.field_changed,
            self.prior_value or "",
            self.new_value or "",
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "event_id": self.event_id_short,
            "event_id_full": self.event_id,
            "ticker": self.ticker,
            "nct_id": self.nct_id,
            "event_type": self.event_type.value,
            "event_severity": self.event_severity.value,
            "event_date": self.event_date,
            "field_changed": self.field_changed,
            "prior_value": self.prior_value,
            "new_value": self.new_value,
            "source": self.source,
            "confidence": self.confidence.value,
            "disclosed_at": self.disclosed_at,
            "source_date": self.source_date,
            "pit_date_field_used": self.pit_date_field_used,
            "source_reliability": self.source_reliability.value,
            "date_specificity": self.date_specificity.value,
            "corroboration_count": self.corroboration_count,
            "is_negative": self.is_negative,
            "is_severe_negative": self.is_severe_negative,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CatalystEventV2":
        """Deserialize from dict."""
        return cls(
            ticker=data["ticker"],
            nct_id=data["nct_id"],
            event_type=EventType(data["event_type"]),
            event_severity=EventSeverity(data["event_severity"]),
            event_date=data.get("event_date"),
            field_changed=data["field_changed"],
            prior_value=data.get("prior_value"),
            new_value=data.get("new_value"),
            source=data["source"],
            confidence=ConfidenceLevel(data["confidence"]),
            disclosed_at=data["disclosed_at"],
            source_date=data.get("source_date", data["disclosed_at"]),
            pit_date_field_used=data.get("pit_date_field_used", "disclosed_at"),
            source_reliability=SourceReliability(data.get("source_reliability", "INFERRED")),
            date_specificity=DateSpecificity(data.get("date_specificity", "UNKNOWN")),
            corroboration_count=data.get("corroboration_count", 0),
        )


# =============================================================================
# DELTA EVENT (FOR CHANGE DETECTION)
# =============================================================================

class DeltaType(str, Enum):
    """Types of deltas detected between snapshots."""
    EVENT_ADDED = "EVENT_ADDED"
    EVENT_REMOVED = "EVENT_REMOVED"
    DATE_SHIFT = "DATE_SHIFT"
    WINDOW_WIDENING = "WINDOW_WIDENING"
    STATUS_CHANGE = "STATUS_CHANGE"


@dataclass(frozen=True)
class DeltaEvent:
    """
    Represents a change detected between two snapshots.
    """
    ticker: str
    nct_id: str
    delta_type: DeltaType
    prior_value: Optional[str]
    new_value: Optional[str]
    shift_days: Optional[int]  # For DATE_SHIFT: positive = later, negative = earlier
    source_date: str

    @property
    def delta_id(self) -> str:
        """Stable delta ID."""
        canonical = f"{self.ticker}|{self.nct_id}|{self.delta_type.value}|{self.prior_value}|{self.new_value}"
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "delta_id": self.delta_id,
            "ticker": self.ticker,
            "nct_id": self.nct_id,
            "delta_type": self.delta_type.value,
            "prior_value": self.prior_value,
            "new_value": self.new_value,
            "shift_days": self.shift_days,
            "source_date": self.source_date,
        }


# =============================================================================
# TICKER CATALYST SUMMARY V2 (ROBUST)
# =============================================================================

@dataclass
class TickerCatalystSummaryV2:
    """
    Aggregated catalyst summary for a ticker with full scoring.
    """
    ticker: str
    as_of_date: str

    # === SCORES ===
    score_override: Decimal
    score_blended: Decimal
    score_mode_used: str

    # === PROXIMITY SCORING ===
    catalyst_proximity_score: Decimal = Decimal("0")
    n_events_upcoming: int = 0

    # === DELTA SCORING ===
    catalyst_delta_score: Decimal = Decimal("0")
    n_events_added: int = 0
    n_events_removed: int = 0
    n_date_shifts: int = 0
    n_window_widenings: int = 0
    n_status_changes: int = 0
    max_slip_days: Optional[int] = None

    # === VELOCITY ===
    catalyst_velocity_4w: Optional[Decimal] = None

    # === NEGATIVE CATALYST SCORING ===
    negative_catalyst_score: Decimal = Decimal("0")
    n_negative_events: int = 0
    n_severe_negative_events: int = 0

    # === CERTAINTY ===
    avg_certainty_score: Decimal = Decimal("0.5")
    n_high_confidence: int = 0
    uncertainty_penalty: Decimal = Decimal("0")

    # === FLAGS ===
    severe_negative_flag: bool = False

    # === INTEGRATION HOOKS ===
    next_catalyst_date: Optional[str] = None
    catalyst_window_days: Optional[int] = None
    catalyst_window_bucket: CatalystWindowBucket = CatalystWindowBucket.UNKNOWN
    catalyst_confidence: ConfidenceLevel = ConfidenceLevel.MED

    # === EVENT SUMMARY ===
    events_total: int = 0
    events_by_severity: Dict[str, int] = field(default_factory=dict)
    events_by_type: Dict[str, int] = field(default_factory=dict)
    weighted_counts_by_severity: Dict[str, str] = field(default_factory=dict)

    # === TOP EVENTS ===
    top_3_events: List[Dict[str, Any]] = field(default_factory=list)

    # === ALL EVENTS ===
    events: List[CatalystEventV2] = field(default_factory=list)

    # === DELTA EVENTS ===
    delta_events: List[DeltaEvent] = field(default_factory=list)

    # === COVERAGE STATE ===
    coverage_state: str = "FULL"  # FULL, PARTIAL, NONE

    # === SCHEMA METADATA ===
    schema_version: str = SCHEMA_VERSION
    score_version: str = SCORE_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict with deterministic ordering."""
        return {
            "_schema": {
                "schema_version": self.schema_version,
                "score_version": self.score_version,
            },
            "ticker": self.ticker,
            "as_of_date": self.as_of_date,
            "scores": {
                "score_override": str(self.score_override),
                "score_blended": str(self.score_blended),
                "score_mode_used": self.score_mode_used,
                "catalyst_proximity_score": str(self.catalyst_proximity_score),
                "catalyst_delta_score": str(self.catalyst_delta_score),
                "catalyst_velocity_4w": str(self.catalyst_velocity_4w) if self.catalyst_velocity_4w is not None else None,
                "negative_catalyst_score": str(self.negative_catalyst_score),
                "avg_certainty_score": str(self.avg_certainty_score),
                "uncertainty_penalty": str(self.uncertainty_penalty),
            },
            "flags": {
                "severe_negative_flag": self.severe_negative_flag,
            },
            "integration": {
                "next_catalyst_date": self.next_catalyst_date,
                "catalyst_window_days": self.catalyst_window_days,
                "catalyst_window_bucket": self.catalyst_window_bucket.value,
                "catalyst_confidence": self.catalyst_confidence.value,
            },
            "event_summary": {
                "events_total": self.events_total,
                "events_by_severity": self.events_by_severity,
                "events_by_type": self.events_by_type,
                "weighted_counts_by_severity": self.weighted_counts_by_severity,
                "n_events_upcoming": self.n_events_upcoming,
                "n_events_added": self.n_events_added,
                "n_events_removed": self.n_events_removed,
                "n_date_shifts": self.n_date_shifts,
                "n_window_widenings": self.n_window_widenings,
                "n_status_changes": self.n_status_changes,
                "max_slip_days": self.max_slip_days,
                "n_negative_events": self.n_negative_events,
                "n_severe_negative_events": self.n_severe_negative_events,
                "n_high_confidence": self.n_high_confidence,
                "coverage_state": self.coverage_state,
            },
            "top_3_events": self.top_3_events,
            "events": [e.to_dict() for e in self.events],
            "delta_events": [d.to_dict() for d in self.delta_events],
        }


# =============================================================================
# ENHANCED DIAGNOSTICS
# =============================================================================

@dataclass
class DiagnosticCountsV2:
    """Enhanced diagnostic counters."""
    # Event counts
    events_detected_total: int = 0
    events_deduped: int = 0
    events_by_type: Dict[str, int] = field(default_factory=dict)
    events_by_confidence: Dict[str, int] = field(default_factory=dict)
    events_by_severity: Dict[str, int] = field(default_factory=dict)

    # Ticker counts
    tickers_with_severe_negative: int = 0
    tickers_analyzed: int = 0
    tickers_with_events: int = 0

    # NEW: Upcoming events
    n_events_upcoming: int = 0
    n_high_conf: int = 0

    # NEW: Slip tracking
    max_slip_days: Optional[int] = None
    total_slip_events: int = 0

    # NEW: Uncertainty
    uncertainty_penalty_total: Decimal = Decimal("0")
    avg_certainty: Decimal = Decimal("0.5")

    # NEW: Coverage
    coverage_state: str = "UNKNOWN"  # FULL, PARTIAL, NONE
    coverage_pct: Decimal = Decimal("0")

    # NEW: Negative catalyst counts
    n_negative_total: int = 0
    n_severe_negative_total: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize with sorted keys for determinism."""
        return {
            "events_detected_total": self.events_detected_total,
            "events_deduped": self.events_deduped,
            "events_by_type": dict(sorted(self.events_by_type.items())),
            "events_by_confidence": dict(sorted(self.events_by_confidence.items())),
            "events_by_severity": dict(sorted(self.events_by_severity.items())),
            "tickers_with_severe_negative": self.tickers_with_severe_negative,
            "tickers_analyzed": self.tickers_analyzed,
            "tickers_with_events": self.tickers_with_events,
            "n_events_upcoming": self.n_events_upcoming,
            "n_high_conf": self.n_high_conf,
            "max_slip_days": self.max_slip_days,
            "total_slip_events": self.total_slip_events,
            "uncertainty_penalty_total": str(self.uncertainty_penalty_total),
            "avg_certainty": str(self.avg_certainty),
            "coverage_state": self.coverage_state,
            "coverage_pct": str(self.coverage_pct),
            "n_negative_total": self.n_negative_total,
            "n_severe_negative_total": self.n_severe_negative_total,
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def decimal_to_str(d: Decimal, places: int = 4) -> str:
    """Convert Decimal to string with fixed precision."""
    quantizer = Decimal(10) ** -places
    return str(d.quantize(quantizer, rounding=ROUND_HALF_UP))


def compute_catalyst_window_bucket(days: Optional[int]) -> CatalystWindowBucket:
    """Compute catalyst window bucket from days."""
    if days is None:
        return CatalystWindowBucket.UNKNOWN
    if days <= 30:
        return CatalystWindowBucket.DAYS_0_30
    if days <= 90:
        return CatalystWindowBucket.DAYS_31_90
    if days <= 180:
        return CatalystWindowBucket.DAYS_91_180
    if days <= 365:
        return CatalystWindowBucket.DAYS_181_365
    return CatalystWindowBucket.DAYS_GT_365


def canonical_json_dumps(obj: Any) -> str:
    """Serialize to canonical JSON (sorted keys, deterministic)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"


def validate_event_schema(event_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate event against schema."""
    errors = []

    required_fields = [
        "ticker", "nct_id", "event_type", "event_severity",
        "field_changed", "source", "confidence", "disclosed_at",
        "source_date", "pit_date_field_used"
    ]

    for fld in required_fields:
        if fld not in event_dict:
            errors.append(f"Missing required field: {fld}")

    # Validate enums
    if "event_type" in event_dict:
        try:
            EventType(event_dict["event_type"])
        except ValueError:
            errors.append(f"Invalid event_type: {event_dict['event_type']}")

    if "event_severity" in event_dict:
        try:
            EventSeverity(event_dict["event_severity"])
        except ValueError:
            errors.append(f"Invalid event_severity: {event_dict['event_severity']}")

    return (len(errors) == 0, errors)


def validate_summary_schema(summary_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate ticker summary against schema."""
    errors = []

    schema = summary_dict.get("_schema", {})
    if "schema_version" not in schema:
        errors.append("Missing _schema.schema_version")

    required_sections = ["ticker", "as_of_date", "scores", "flags", "integration", "event_summary"]
    for section in required_sections:
        if section not in summary_dict:
            errors.append(f"Missing required section: {section}")

    return (len(errors) == 0, errors)
