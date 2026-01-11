#!/usr/bin/env python3
"""
module_3_schema.py - Module 3 Catalyst Event Schema vNext

Defines versioned, deterministic schemas for catalyst events and scoring.

Schema Versions:
- SCHEMA_VERSION: Event data structure version
- SCORE_VERSION: Scoring algorithm version

Determinism Guarantees:
- All event_ids are SHA256 hashes of canonical tuples
- All enums have explicit string values
- All floats use Decimal for stable serialization
- All orderings use explicit sort keys
"""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple

# =============================================================================
# VERSION CONSTANTS
# =============================================================================

SCHEMA_VERSION = "m3catalyst_vnext_20260111"
SCORE_VERSION = "m3score_vnext_20260111"

# =============================================================================
# EVENT TYPE TAXONOMY (EXPANDED)
# =============================================================================

class EventType(str, Enum):
    """
    Catalyst event types from CT.gov deltas.

    Taxonomy expanded to include:
    - Status changes (severe neg, upgrade, downgrade)
    - Timeline changes (pushout, pullin)
    - Confirmation events
    - Results events
    - Protocol amendments
    - Enrollment changes
    """
    # Status changes
    CT_STATUS_SEVERE_NEG = "CT_STATUS_SEVERE_NEG"
    CT_STATUS_DOWNGRADE = "CT_STATUS_DOWNGRADE"
    CT_STATUS_UPGRADE = "CT_STATUS_UPGRADE"

    # Timeline changes
    CT_TIMELINE_PUSHOUT = "CT_TIMELINE_PUSHOUT"
    CT_TIMELINE_PULLIN = "CT_TIMELINE_PULLIN"

    # Confirmation events
    CT_DATE_CONFIRMED_ACTUAL = "CT_DATE_CONFIRMED_ACTUAL"
    CT_PRIMARY_COMPLETION = "CT_PRIMARY_COMPLETION"
    CT_STUDY_COMPLETION = "CT_STUDY_COMPLETION"

    # Results events
    CT_RESULTS_POSTED = "CT_RESULTS_POSTED"

    # Protocol amendments (new)
    CT_PROTOCOL_AMENDMENT = "CT_PROTOCOL_AMENDMENT"
    CT_ARM_ADDED = "CT_ARM_ADDED"
    CT_ARM_REMOVED = "CT_ARM_REMOVED"
    CT_ENDPOINT_CHANGED = "CT_ENDPOINT_CHANGED"

    # Enrollment changes (new)
    CT_ENROLLMENT_STARTED = "CT_ENROLLMENT_STARTED"
    CT_ENROLLMENT_COMPLETE = "CT_ENROLLMENT_COMPLETE"
    CT_ENROLLMENT_PAUSED = "CT_ENROLLMENT_PAUSED"
    CT_ENROLLMENT_RESUMED = "CT_ENROLLMENT_RESUMED"

    # Unknown (catch-all, zero score)
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


# =============================================================================
# EVENT TYPE → SEVERITY MAPPING
# =============================================================================

EVENT_SEVERITY_MAP: Dict[EventType, EventSeverity] = {
    # Critical positive
    EventType.CT_STATUS_UPGRADE: EventSeverity.POSITIVE,  # Can be critical depending on phase
    EventType.CT_TIMELINE_PULLIN: EventSeverity.POSITIVE,
    EventType.CT_DATE_CONFIRMED_ACTUAL: EventSeverity.POSITIVE,
    EventType.CT_PRIMARY_COMPLETION: EventSeverity.CRITICAL_POSITIVE,
    EventType.CT_STUDY_COMPLETION: EventSeverity.POSITIVE,
    EventType.CT_ENROLLMENT_COMPLETE: EventSeverity.POSITIVE,
    EventType.CT_ENROLLMENT_STARTED: EventSeverity.POSITIVE,
    EventType.CT_ENROLLMENT_RESUMED: EventSeverity.POSITIVE,
    EventType.CT_ARM_ADDED: EventSeverity.NEUTRAL,

    # Neutral
    EventType.CT_RESULTS_POSTED: EventSeverity.NEUTRAL,
    EventType.CT_PROTOCOL_AMENDMENT: EventSeverity.NEUTRAL,
    EventType.CT_ENDPOINT_CHANGED: EventSeverity.NEUTRAL,
    EventType.UNKNOWN: EventSeverity.NEUTRAL,

    # Negative
    EventType.CT_STATUS_DOWNGRADE: EventSeverity.NEGATIVE,
    EventType.CT_TIMELINE_PUSHOUT: EventSeverity.NEGATIVE,
    EventType.CT_ARM_REMOVED: EventSeverity.NEGATIVE,
    EventType.CT_ENROLLMENT_PAUSED: EventSeverity.NEGATIVE,

    # Severe negative
    EventType.CT_STATUS_SEVERE_NEG: EventSeverity.SEVERE_NEGATIVE,
}


# =============================================================================
# EVENT TYPE → DEFAULT CONFIDENCE
# =============================================================================

EVENT_DEFAULT_CONFIDENCE: Dict[EventType, ConfidenceLevel] = {
    EventType.CT_STATUS_SEVERE_NEG: ConfidenceLevel.HIGH,
    EventType.CT_STATUS_DOWNGRADE: ConfidenceLevel.HIGH,
    EventType.CT_STATUS_UPGRADE: ConfidenceLevel.HIGH,
    EventType.CT_TIMELINE_PUSHOUT: ConfidenceLevel.MED,
    EventType.CT_TIMELINE_PULLIN: ConfidenceLevel.MED,
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
# CONFIDENCE → WEIGHT MAPPING
# =============================================================================

CONFIDENCE_WEIGHTS: Dict[ConfidenceLevel, Decimal] = {
    ConfidenceLevel.HIGH: Decimal("1.0"),
    ConfidenceLevel.MED: Decimal("0.6"),
    ConfidenceLevel.LOW: Decimal("0.3"),
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
# EVENT TYPE → PROXIMITY WEIGHT (for upcoming event scoring)
# =============================================================================

EVENT_TYPE_WEIGHT: Dict[EventType, Decimal] = {
    # High-impact upcoming events
    EventType.CT_PRIMARY_COMPLETION: Decimal("20.0"),
    EventType.CT_STUDY_COMPLETION: Decimal("15.0"),
    EventType.CT_RESULTS_POSTED: Decimal("18.0"),
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

    # Neutral/negative (shouldn't affect upcoming score positively)
    EventType.CT_STATUS_SEVERE_NEG: Decimal("0.0"),
    EventType.CT_STATUS_DOWNGRADE: Decimal("0.0"),
    EventType.CT_TIMELINE_PUSHOUT: Decimal("0.0"),
    EventType.CT_ARM_REMOVED: Decimal("0.0"),
    EventType.CT_ENROLLMENT_PAUSED: Decimal("0.0"),
    EventType.CT_ENDPOINT_CHANGED: Decimal("1.0"),
    EventType.UNKNOWN: Decimal("0.0"),
}


# =============================================================================
# CATALYST EVENT (VERSIONED)
# =============================================================================

@dataclass(frozen=True)
class CatalystEventV2:
    """
    Single catalyst event with deterministic event_id.

    Frozen dataclass ensures immutability for hashing.
    """
    ticker: str
    nct_id: str
    event_type: EventType
    event_severity: EventSeverity
    event_date: Optional[str]  # ISO date string or None
    field_changed: str
    prior_value: Optional[str]
    new_value: Optional[str]
    source: str
    confidence: ConfidenceLevel
    disclosed_at: str  # ISO date string

    @property
    def event_id(self) -> str:
        """
        Deterministic event ID from canonical tuple.

        SHA256 of: (ticker, nct_id, event_type, event_date, field_changed,
                    prior_value, new_value, source)
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
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()[:16]

    def sort_key(self) -> Tuple:
        """Deterministic sort key for event ordering."""
        return (
            self.event_date or "9999-99-99",  # Null dates sort last
            self.event_type.value,
            self.nct_id,
            self.field_changed,
            self.prior_value or "",
            self.new_value or "",
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "event_id": self.event_id,
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
        )


# =============================================================================
# TICKER CATALYST SUMMARY (VERSIONED)
# =============================================================================

@dataclass
class TickerCatalystSummaryV2:
    """
    Aggregated catalyst summary for a ticker.

    Includes both override and blended scores for flexibility.
    """
    ticker: str
    as_of_date: str  # ISO date string

    # Scores (override uses hierarchical logic, blended uses confidence weighting)
    score_override: Decimal
    score_blended: Decimal
    score_mode_used: str  # "override" or "blended"

    # Flags
    severe_negative_flag: bool

    # Integration hooks
    next_catalyst_date: Optional[str]  # ISO date or None
    catalyst_window_days: Optional[int]
    catalyst_window_bucket: CatalystWindowBucket
    catalyst_confidence: ConfidenceLevel

    # Event summary counts
    events_total: int
    events_by_severity: Dict[str, int]
    events_by_type: Dict[str, int]
    weighted_counts_by_severity: Dict[str, str]  # String decimals for JSON

    # Top 3 events for downstream consumers
    top_3_events: List[Dict[str, Any]]

    # All events (sorted deterministically)
    events: List[CatalystEventV2]

    # NEW: Proximity scoring (exponential decay for upcoming events)
    catalyst_proximity_score: Decimal = Decimal("0")
    n_events_upcoming: int = 0

    # NEW: Delta scoring (event-based changes)
    catalyst_delta_score: Decimal = Decimal("0")
    n_events_added: int = 0
    n_events_removed: int = 0
    max_slip_days: Optional[int] = None  # Worst delay (positive = later)

    # NEW: Velocity (rolling baseline comparison)
    catalyst_velocity_4w: Optional[Decimal] = None  # Current - median of last 4

    # Schema metadata
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
                "max_slip_days": self.max_slip_days,
            },
            "top_3_events": self.top_3_events,
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TickerCatalystSummaryV2":
        """Deserialize from dict with schema migration support."""
        schema = data.get("_schema", {})

        # Handle legacy schema if needed
        if "schema_version" not in schema:
            return cls._migrate_from_legacy(data)

        events = [CatalystEventV2.from_dict(e) for e in data.get("events", [])]

        return cls(
            ticker=data["ticker"],
            as_of_date=data["as_of_date"],
            score_override=Decimal(data["scores"]["score_override"]),
            score_blended=Decimal(data["scores"]["score_blended"]),
            score_mode_used=data["scores"]["score_mode_used"],
            severe_negative_flag=data["flags"]["severe_negative_flag"],
            next_catalyst_date=data["integration"]["next_catalyst_date"],
            catalyst_window_days=data["integration"]["catalyst_window_days"],
            catalyst_window_bucket=CatalystWindowBucket(data["integration"]["catalyst_window_bucket"]),
            catalyst_confidence=ConfidenceLevel(data["integration"]["catalyst_confidence"]),
            events_total=data["event_summary"]["events_total"],
            events_by_severity=data["event_summary"]["events_by_severity"],
            events_by_type=data["event_summary"]["events_by_type"],
            weighted_counts_by_severity=data["event_summary"]["weighted_counts_by_severity"],
            top_3_events=data["top_3_events"],
            events=events,
            schema_version=schema.get("schema_version", SCHEMA_VERSION),
            score_version=schema.get("score_version", SCORE_VERSION),
        )

    @classmethod
    def _migrate_from_legacy(cls, data: Dict[str, Any]) -> "TickerCatalystSummaryV2":
        """Migrate from legacy TickerCatalystSummary format."""
        # This handles the old format from catalyst_summary.py
        ticker = data.get("ticker", "UNKNOWN")
        as_of_date = data.get("as_of_date", "1970-01-01")

        # Legacy scores
        net_score = Decimal(str(data.get("catalyst_score_net", 50)))

        # Map to new scale (legacy net score to 0-100)
        # Legacy: positive means good, negative means bad
        # New: 50 is neutral, >50 positive, <50 negative
        legacy_score = Decimal("50") + (net_score * Decimal("10"))
        legacy_score = max(Decimal("0"), min(Decimal("100"), legacy_score))

        # Flags
        severe_neg = data.get("severe_negative_flag", False)

        # Events - need to convert from legacy format
        legacy_events = data.get("events", [])
        events = []

        for le in legacy_events:
            # Handle both dict and dataclass
            if hasattr(le, "to_dict"):
                le = le.to_dict()

            # Map legacy event_type to new EventType
            legacy_type = le.get("event_type", "UNKNOWN")
            try:
                event_type = EventType(legacy_type)
            except ValueError:
                event_type = EventType.UNKNOWN

            severity = EVENT_SEVERITY_MAP.get(event_type, EventSeverity.NEUTRAL)
            confidence = EVENT_DEFAULT_CONFIDENCE.get(event_type, ConfidenceLevel.MED)

            events.append(CatalystEventV2(
                ticker=ticker,
                nct_id=le.get("nct_id", ""),
                event_type=event_type,
                event_severity=severity,
                event_date=le.get("actual_date") or le.get("disclosed_at"),
                field_changed=list(le.get("fields_changed", {}).keys())[0] if le.get("fields_changed") else "",
                prior_value=None,
                new_value=None,
                source=le.get("source", "CTGOV"),
                confidence=confidence,
                disclosed_at=le.get("disclosed_at", as_of_date),
            ))

        # Sort events
        events.sort(key=lambda e: e.sort_key())

        # Compute summary stats
        events_by_severity = {}
        events_by_type = {}
        for e in events:
            sev = e.event_severity.value
            events_by_severity[sev] = events_by_severity.get(sev, 0) + 1
            typ = e.event_type.value
            events_by_type[typ] = events_by_type.get(typ, 0) + 1

        return cls(
            ticker=ticker,
            as_of_date=as_of_date,
            score_override=legacy_score,
            score_blended=legacy_score,
            score_mode_used="override",
            severe_negative_flag=severe_neg,
            next_catalyst_date=None,
            catalyst_window_days=None,
            catalyst_window_bucket=CatalystWindowBucket.UNKNOWN,
            catalyst_confidence=ConfidenceLevel.MED,
            events_total=len(events),
            events_by_severity=events_by_severity,
            events_by_type=events_by_type,
            weighted_counts_by_severity={},
            top_3_events=[e.to_dict() for e in events[:3]],
            events=events,
        )


# =============================================================================
# DIAGNOSTICS
# =============================================================================

@dataclass
class DiagnosticCounts:
    """Deterministic diagnostic counters."""
    events_detected_total: int = 0
    events_deduped: int = 0
    events_by_type: Dict[str, int] = field(default_factory=dict)
    events_by_confidence: Dict[str, int] = field(default_factory=dict)
    events_by_severity: Dict[str, int] = field(default_factory=dict)
    tickers_with_severe_negative: int = 0
    tickers_analyzed: int = 0
    tickers_with_events: int = 0

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
    """
    Serialize to canonical JSON.

    Guarantees:
    - Sorted keys
    - No trailing whitespace
    - Consistent separators
    - Trailing newline
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"


# =============================================================================
# SCHEMA VALIDATION
# =============================================================================

def validate_event_schema(event_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate event against schema."""
    errors = []

    required_fields = [
        "event_id", "ticker", "nct_id", "event_type", "event_severity",
        "field_changed", "source", "confidence", "disclosed_at"
    ]

    for field in required_fields:
        if field not in event_dict:
            errors.append(f"Missing required field: {field}")

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

    if "confidence" in event_dict:
        try:
            ConfidenceLevel(event_dict["confidence"])
        except ValueError:
            errors.append(f"Invalid confidence: {event_dict['confidence']}")

    return (len(errors) == 0, errors)


def validate_summary_schema(summary_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate ticker summary against schema."""
    errors = []

    # Check schema version
    schema = summary_dict.get("_schema", {})
    if "schema_version" not in schema:
        errors.append("Missing _schema.schema_version")

    # Check required sections
    required_sections = ["ticker", "as_of_date", "scores", "flags", "integration", "event_summary"]
    for section in required_sections:
        if section not in summary_dict:
            errors.append(f"Missing required section: {section}")

    # Validate events
    for i, event in enumerate(summary_dict.get("events", [])):
        valid, event_errors = validate_event_schema(event)
        if not valid:
            errors.extend([f"events[{i}]: {e}" for e in event_errors])

    return (len(errors) == 0, errors)
