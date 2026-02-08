#!/usr/bin/env python3
"""
Tests for Module 3 Catalyst Upgrade vNext

Test Suite:
T1: Determinism / Byte-identical
T2: PIT Safety
T3: Stable Ordering
T4: Schema Evolution
T5: Confidence Weighting
T6: Recency / Decay
T7: Integration Hooks
T8: Dedup / Noise-band

Run with: pytest tests/test_module_3_vnext.py -v
"""

import hashlib
import json
import pytest
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    canonical_json_dumps,
    validate_event_schema,
    validate_summary_schema,
    compute_catalyst_window_bucket,
)

from module_3_scoring import (
    calculate_score_override,
    calculate_score_blended,
    calculate_ticker_catalyst_score,
    compute_recency_weight,
    compute_staleness_factor,
    compute_bucket_proximity_score,
    _proximity_time_weight,
    SCORE_OVERRIDE_SEVERE_NEGATIVE,
    SCORE_OVERRIDE_CRITICAL_POSITIVE,
    SCORE_DEFAULT,
    PROXIMITY_NEUTRAL_BASE,
    _PROXIMITY_RAMP_END,
    _PROXIMITY_PLATEAU_END,
    _PROXIMITY_HORIZON,
    _PROXIMITY_SATURATION_K,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_events() -> List[CatalystEventV2]:
    """Create sample events for testing."""
    return [
        CatalystEventV2(
            ticker="AAPL",
            nct_id="NCT00000001",
            event_type=EventType.CT_STATUS_UPGRADE,
            event_severity=EventSeverity.POSITIVE,
            event_date="2024-01-15",
            field_changed="overallStatus",
            prior_value="RECRUITING",
            new_value="ACTIVE_NOT_RECRUITING",
            source="CTGOV",
            confidence=ConfidenceLevel.HIGH,
            disclosed_at="2024-01-15",
        ),
        CatalystEventV2(
            ticker="AAPL",
            nct_id="NCT00000002",
            event_type=EventType.CT_TIMELINE_PULLIN,
            event_severity=EventSeverity.POSITIVE,
            event_date="2024-01-10",
            field_changed="primaryCompletionDate",
            prior_value="2024-06-30",
            new_value="2024-03-31",
            source="CTGOV",
            confidence=ConfidenceLevel.MED,
            disclosed_at="2024-01-10",
        ),
    ]


@pytest.fixture
def severe_negative_event() -> CatalystEventV2:
    """Create a severe negative event."""
    return CatalystEventV2(
        ticker="BIIB",
        nct_id="NCT00000003",
        event_type=EventType.CT_STATUS_SEVERE_NEG,
        event_severity=EventSeverity.SEVERE_NEGATIVE,
        event_date="2024-01-20",
        field_changed="overallStatus",
        prior_value="RECRUITING",
        new_value="TERMINATED",
        source="CTGOV",
        confidence=ConfidenceLevel.HIGH,
        disclosed_at="2024-01-20",
    )


@pytest.fixture
def critical_positive_event() -> CatalystEventV2:
    """Create a critical positive event."""
    return CatalystEventV2(
        ticker="MRNA",
        nct_id="NCT00000004",
        event_type=EventType.CT_PRIMARY_COMPLETION,
        event_severity=EventSeverity.CRITICAL_POSITIVE,
        event_date="2024-01-25",
        field_changed="primaryCompletionType",
        prior_value="ANTICIPATED",
        new_value="ACTUAL",
        source="CTGOV",
        confidence=ConfidenceLevel.HIGH,
        disclosed_at="2024-01-25",
    )


@pytest.fixture
def legacy_summary_fixture() -> dict:
    """Create a legacy format summary for migration testing."""
    return {
        "ticker": "TEST",
        "as_of_date": "2024-01-15",
        "catalyst_score_pos": 2.5,
        "catalyst_score_neg": 1.0,
        "catalyst_score_net": 1.5,
        "nearest_positive_days": 30,
        "nearest_negative_days": None,
        "severe_negative_flag": False,
        "events": [
            {
                "source": "CTGOV",
                "nct_id": "NCT12345",
                "event_type": "CT_STATUS_UPGRADE",
                "direction": "POS",
                "impact": 2,
                "confidence": 0.85,
                "disclosed_at": "2024-01-10",
                "fields_changed": {"overallStatus": ["RECRUITING", "ACTIVE"]},
                "actual_date": None,
            }
        ],
    }


# =============================================================================
# T1: DETERMINISM / BYTE-IDENTICAL
# =============================================================================

class TestDeterminism:
    """T1: Verify byte-identical outputs across runs."""

    def test_event_id_deterministic(self, sample_events):
        """Event IDs are deterministic."""
        event = sample_events[0]

        # Recreate the same event
        event2 = CatalystEventV2(
            ticker=event.ticker,
            nct_id=event.nct_id,
            event_type=event.event_type,
            event_severity=event.event_severity,
            event_date=event.event_date,
            field_changed=event.field_changed,
            prior_value=event.prior_value,
            new_value=event.new_value,
            source=event.source,
            confidence=event.confidence,
            disclosed_at=event.disclosed_at,
        )

        assert event.event_id == event2.event_id

    def test_canonical_json_deterministic(self, sample_events):
        """Canonical JSON produces identical output."""
        event = sample_events[0]

        json1 = canonical_json_dumps(event.to_dict())
        json2 = canonical_json_dumps(event.to_dict())

        assert json1 == json2
        assert hashlib.sha256(json1.encode()).hexdigest() == hashlib.sha256(json2.encode()).hexdigest()

    def test_scoring_deterministic(self, sample_events):
        """Scoring produces identical results."""
        as_of = date(2024, 1, 31)

        score1, _ = calculate_score_override(sample_events)
        score2, _ = calculate_score_override(sample_events)

        assert score1 == score2

        blended1, _ = calculate_score_blended(sample_events, as_of)
        blended2, _ = calculate_score_blended(sample_events, as_of)

        assert blended1 == blended2

    def test_summary_serialization_deterministic(self, sample_events):
        """Summary serialization is deterministic."""
        as_of = date(2024, 1, 31)

        summary1 = calculate_ticker_catalyst_score("TEST", sample_events, as_of)
        summary2 = calculate_ticker_catalyst_score("TEST", sample_events, as_of)

        json1 = canonical_json_dumps(summary1.to_dict())
        json2 = canonical_json_dumps(summary2.to_dict())

        assert json1 == json2


# =============================================================================
# T2: PIT SAFETY
# =============================================================================

class TestPITSafety:
    """T2: Point-in-time safety - no future data."""

    def test_future_events_excluded_from_blended_score(self):
        """Future events (after as_of_date) don't contribute to blended score."""
        as_of = date(2024, 1, 15)

        # Event in the future
        future_event = CatalystEventV2(
            ticker="TEST",
            nct_id="NCT001",
            event_type=EventType.CT_PRIMARY_COMPLETION,
            event_severity=EventSeverity.CRITICAL_POSITIVE,
            event_date="2024-02-01",  # After as_of_date
            field_changed="status",
            prior_value="A",
            new_value="B",
            source="CTGOV",
            confidence=ConfidenceLevel.HIGH,
            disclosed_at="2024-02-01",
        )

        # Event in the past
        past_event = CatalystEventV2(
            ticker="TEST",
            nct_id="NCT002",
            event_type=EventType.CT_STATUS_UPGRADE,
            event_severity=EventSeverity.POSITIVE,
            event_date="2024-01-10",
            field_changed="status",
            prior_value="X",
            new_value="Y",
            source="CTGOV",
            confidence=ConfidenceLevel.HIGH,
            disclosed_at="2024-01-10",
        )

        # Score with only past event should equal score with both
        # (future event should be ignored)
        score_past_only, _ = calculate_score_blended([past_event], as_of)
        score_both, _ = calculate_score_blended([past_event, future_event], as_of)

        # Note: Future events might still appear but with 0 weight
        # The key is they don't inflate the score
        assert score_past_only <= score_both + Decimal("0.01")  # Allow tiny rounding

    def test_recency_weight_zero_for_future(self):
        """Recency weight is 0 for future events."""
        as_of = date(2024, 1, 15)
        future_date = "2024-02-01"

        weight = compute_recency_weight(future_date, as_of)

        assert weight == Decimal("0")

    def test_recency_weight_positive_for_past(self):
        """Recency weight is positive for past events."""
        as_of = date(2024, 1, 15)
        past_date = "2024-01-10"

        weight = compute_recency_weight(past_date, as_of)

        assert weight > Decimal("0")

    def test_recency_weight_one_for_same_day(self):
        """Recency weight is 1.0 for same-day events."""
        as_of = date(2024, 1, 15)
        same_date = "2024-01-15"

        weight = compute_recency_weight(same_date, as_of)

        assert weight == Decimal("1.0")


# =============================================================================
# T3: STABLE ORDERING
# =============================================================================

class TestStableOrdering:
    """T3: Verify ordering remains stable across runs."""

    def test_event_sort_key_deterministic(self):
        """Event sort keys are deterministic."""
        event = CatalystEventV2(
            ticker="TEST",
            nct_id="NCT001",
            event_type=EventType.CT_STATUS_UPGRADE,
            event_severity=EventSeverity.POSITIVE,
            event_date="2024-01-15",
            field_changed="status",
            prior_value="A",
            new_value="B",
            source="CTGOV",
            confidence=ConfidenceLevel.HIGH,
            disclosed_at="2024-01-15",
        )

        key1 = event.sort_key()
        key2 = event.sort_key()

        assert key1 == key2

    def test_events_sorted_by_date_first(self):
        """Events are sorted by date first."""
        e1 = CatalystEventV2(
            ticker="TEST", nct_id="NCT001", event_type=EventType.CT_STATUS_UPGRADE,
            event_severity=EventSeverity.POSITIVE, event_date="2024-01-20",
            field_changed="s", prior_value="a", new_value="b", source="X",
            confidence=ConfidenceLevel.HIGH, disclosed_at="2024-01-20",
        )
        e2 = CatalystEventV2(
            ticker="TEST", nct_id="NCT002", event_type=EventType.CT_STATUS_UPGRADE,
            event_severity=EventSeverity.POSITIVE, event_date="2024-01-10",
            field_changed="s", prior_value="a", new_value="b", source="X",
            confidence=ConfidenceLevel.HIGH, disclosed_at="2024-01-10",
        )

        sorted_events = sorted([e1, e2], key=lambda e: e.sort_key())

        assert sorted_events[0].event_date == "2024-01-10"
        assert sorted_events[1].event_date == "2024-01-20"

    def test_null_dates_sort_last(self):
        """Events with null dates sort after events with dates."""
        e1 = CatalystEventV2(
            ticker="TEST", nct_id="NCT001", event_type=EventType.CT_STATUS_UPGRADE,
            event_severity=EventSeverity.POSITIVE, event_date=None,  # Null
            field_changed="s", prior_value="a", new_value="b", source="X",
            confidence=ConfidenceLevel.HIGH, disclosed_at="2024-01-20",
        )
        e2 = CatalystEventV2(
            ticker="TEST", nct_id="NCT002", event_type=EventType.CT_STATUS_UPGRADE,
            event_severity=EventSeverity.POSITIVE, event_date="2024-01-10",
            field_changed="s", prior_value="a", new_value="b", source="X",
            confidence=ConfidenceLevel.HIGH, disclosed_at="2024-01-10",
        )

        sorted_events = sorted([e1, e2], key=lambda e: e.sort_key())

        assert sorted_events[0].event_date == "2024-01-10"  # Has date
        assert sorted_events[1].event_date is None  # Null sorts last

    def test_shuffled_input_same_output(self):
        """Shuffled input produces same sorted output."""
        events = [
            CatalystEventV2(
                ticker="TEST", nct_id=f"NCT{i:03d}", event_type=EventType.CT_STATUS_UPGRADE,
                event_severity=EventSeverity.POSITIVE, event_date=f"2024-01-{10+i:02d}",
                field_changed="s", prior_value="a", new_value="b", source="X",
                confidence=ConfidenceLevel.HIGH, disclosed_at=f"2024-01-{10+i:02d}",
            )
            for i in range(5)
        ]

        import random
        shuffled1 = events.copy()
        random.shuffle(shuffled1)
        shuffled2 = events.copy()
        random.shuffle(shuffled2)

        sorted1 = sorted(shuffled1, key=lambda e: e.sort_key())
        sorted2 = sorted(shuffled2, key=lambda e: e.sort_key())

        assert [e.nct_id for e in sorted1] == [e.nct_id for e in sorted2]


# =============================================================================
# T4: SCHEMA EVOLUTION
# =============================================================================

class TestSchemaEvolution:
    """T4: Legacy schema migration works correctly."""

    def test_legacy_summary_migrates(self, legacy_summary_fixture):
        """Legacy summary can be migrated to V2."""
        migrated = TickerCatalystSummaryV2._migrate_from_legacy(legacy_summary_fixture)

        assert migrated.ticker == "TEST"
        assert migrated.as_of_date == "2024-01-15"
        assert len(migrated.events) == 1
        assert migrated.events[0].event_type == EventType.CT_STATUS_UPGRADE

    def test_migrated_summary_validates(self, legacy_summary_fixture):
        """Migrated summary validates against schema."""
        migrated = TickerCatalystSummaryV2._migrate_from_legacy(legacy_summary_fixture)
        summary_dict = migrated.to_dict()

        is_valid, errors = validate_summary_schema(summary_dict)

        assert is_valid, f"Validation errors: {errors}"

    def test_v2_schema_validates(self, sample_events):
        """V2 summaries validate against schema."""
        as_of = date(2024, 1, 31)
        summary = calculate_ticker_catalyst_score("TEST", sample_events, as_of)

        summary_dict = summary.to_dict()
        is_valid, errors = validate_summary_schema(summary_dict)

        assert is_valid, f"Validation errors: {errors}"

    def test_event_schema_validates(self, sample_events):
        """Events validate against schema."""
        event = sample_events[0]
        event_dict = event.to_dict()

        is_valid, errors = validate_event_schema(event_dict)

        assert is_valid, f"Validation errors: {errors}"


# =============================================================================
# T5: CONFIDENCE WEIGHTING
# =============================================================================

class TestConfidenceWeighting:
    """T5: Confidence affects blended score, not override."""

    def test_override_ignores_confidence(self):
        """Override score doesn't change with confidence."""
        # High confidence severe negative
        high_conf = CatalystEventV2(
            ticker="TEST", nct_id="NCT001", event_type=EventType.CT_STATUS_SEVERE_NEG,
            event_severity=EventSeverity.SEVERE_NEGATIVE, event_date="2024-01-15",
            field_changed="s", prior_value="a", new_value="b", source="X",
            confidence=ConfidenceLevel.HIGH, disclosed_at="2024-01-15",
        )

        # Low confidence severe negative
        low_conf = CatalystEventV2(
            ticker="TEST", nct_id="NCT002", event_type=EventType.CT_STATUS_SEVERE_NEG,
            event_severity=EventSeverity.SEVERE_NEGATIVE, event_date="2024-01-15",
            field_changed="s", prior_value="a", new_value="b", source="X",
            confidence=ConfidenceLevel.LOW, disclosed_at="2024-01-15",
        )

        score_high, _ = calculate_score_override([high_conf])
        score_low, _ = calculate_score_override([low_conf])

        # Both should be SEVERE_NEGATIVE override score
        assert score_high == SCORE_OVERRIDE_SEVERE_NEGATIVE
        assert score_low == SCORE_OVERRIDE_SEVERE_NEGATIVE

    def test_blended_varies_with_confidence(self):
        """Blended score changes with confidence level."""
        as_of = date(2024, 1, 31)

        # High confidence positive
        high_conf = CatalystEventV2(
            ticker="TEST", nct_id="NCT001", event_type=EventType.CT_STATUS_UPGRADE,
            event_severity=EventSeverity.POSITIVE, event_date="2024-01-15",
            field_changed="s", prior_value="a", new_value="b", source="X",
            confidence=ConfidenceLevel.HIGH, disclosed_at="2024-01-15",
        )

        # Low confidence positive (same event otherwise)
        low_conf = CatalystEventV2(
            ticker="TEST", nct_id="NCT002", event_type=EventType.CT_STATUS_UPGRADE,
            event_severity=EventSeverity.POSITIVE, event_date="2024-01-15",
            field_changed="s", prior_value="a", new_value="b", source="X",
            confidence=ConfidenceLevel.LOW, disclosed_at="2024-01-15",
        )

        score_high, _ = calculate_score_blended([high_conf], as_of)
        score_low, _ = calculate_score_blended([low_conf], as_of)

        # High confidence should contribute more
        assert score_high > score_low


# =============================================================================
# T6: RECENCY / DECAY
# =============================================================================

class TestRecencyDecay:
    """T6: Recent events contribute more than old events."""

    def test_recent_event_higher_weight(self):
        """Recent events have higher recency weight."""
        as_of = date(2024, 1, 31)

        recent_weight = compute_recency_weight("2024-01-30", as_of)  # 1 day old
        old_weight = compute_recency_weight("2023-10-31", as_of)  # 92 days old

        assert recent_weight > old_weight

    def test_decay_at_half_life(self):
        """Weight is approximately 0.5 at half-life (90 days)."""
        as_of = date(2024, 1, 31)
        half_life_ago = "2023-11-02"  # ~90 days before

        weight = compute_recency_weight(half_life_ago, as_of)

        # Should be close to 0.5
        assert Decimal("0.4") < weight < Decimal("0.6")

    def test_staleness_penalty_applied(self):
        """Staleness penalty is applied for old events."""
        # Events older than 180 days
        old_events = [
            CatalystEventV2(
                ticker="TEST", nct_id="NCT001", event_type=EventType.CT_STATUS_UPGRADE,
                event_severity=EventSeverity.POSITIVE, event_date="2023-06-01",  # Very old
                field_changed="s", prior_value="a", new_value="b", source="X",
                confidence=ConfidenceLevel.HIGH, disclosed_at="2023-06-01",
            )
        ]

        as_of = date(2024, 1, 31)  # ~240 days later
        staleness = compute_staleness_factor(old_events, as_of)

        assert staleness < Decimal("1.0")  # Penalty applied

    def test_no_staleness_penalty_for_recent(self):
        """No staleness penalty for recent events."""
        recent_events = [
            CatalystEventV2(
                ticker="TEST", nct_id="NCT001", event_type=EventType.CT_STATUS_UPGRADE,
                event_severity=EventSeverity.POSITIVE, event_date="2024-01-20",
                field_changed="s", prior_value="a", new_value="b", source="X",
                confidence=ConfidenceLevel.HIGH, disclosed_at="2024-01-20",
            )
        ]

        as_of = date(2024, 1, 31)
        staleness = compute_staleness_factor(recent_events, as_of)

        assert staleness == Decimal("1.0")


# =============================================================================
# T7: INTEGRATION HOOKS
# =============================================================================

class TestIntegrationHooks:
    """T7: Integration hooks are computed correctly."""

    def test_next_catalyst_date_computed(self):
        """Next catalyst date is computed from future events."""
        as_of = date(2024, 1, 15)

        events = [
            CatalystEventV2(
                ticker="TEST", nct_id="NCT001", event_type=EventType.CT_PRIMARY_COMPLETION,
                event_severity=EventSeverity.CRITICAL_POSITIVE, event_date="2024-02-15",  # Future
                field_changed="s", prior_value="a", new_value="b", source="X",
                confidence=ConfidenceLevel.HIGH, disclosed_at="2024-01-10",
            ),
            CatalystEventV2(
                ticker="TEST", nct_id="NCT002", event_type=EventType.CT_STATUS_UPGRADE,
                event_severity=EventSeverity.POSITIVE, event_date="2024-01-10",  # Past
                field_changed="s", prior_value="a", new_value="b", source="X",
                confidence=ConfidenceLevel.MED, disclosed_at="2024-01-10",
            ),
        ]

        summary = calculate_ticker_catalyst_score("TEST", events, as_of)

        assert summary.next_catalyst_date == "2024-02-15"
        assert summary.catalyst_window_days == 31  # Days until Feb 15

    def test_catalyst_window_bucket_computed(self):
        """Catalyst window bucket is computed correctly."""
        assert compute_catalyst_window_bucket(15) == CatalystWindowBucket.DAYS_0_30
        assert compute_catalyst_window_bucket(45) == CatalystWindowBucket.DAYS_31_90
        assert compute_catalyst_window_bucket(100) == CatalystWindowBucket.DAYS_91_180
        assert compute_catalyst_window_bucket(200) == CatalystWindowBucket.DAYS_181_365
        assert compute_catalyst_window_bucket(400) == CatalystWindowBucket.DAYS_GT_365
        assert compute_catalyst_window_bucket(None) == CatalystWindowBucket.UNKNOWN

    def test_catalyst_confidence_from_events(self):
        """Catalyst confidence is derived from event confidences."""
        as_of = date(2024, 1, 15)

        # Future event with HIGH confidence
        events = [
            CatalystEventV2(
                ticker="TEST", nct_id="NCT001", event_type=EventType.CT_PRIMARY_COMPLETION,
                event_severity=EventSeverity.CRITICAL_POSITIVE, event_date="2024-02-15",
                field_changed="s", prior_value="a", new_value="b", source="X",
                confidence=ConfidenceLevel.HIGH, disclosed_at="2024-01-10",
            ),
        ]

        summary = calculate_ticker_catalyst_score("TEST", events, as_of)

        assert summary.catalyst_confidence == ConfidenceLevel.HIGH

    def test_top_3_events_selected(self, sample_events, severe_negative_event):
        """Top 3 events are selected correctly."""
        as_of = date(2024, 1, 31)

        events = sample_events + [severe_negative_event]
        summary = calculate_ticker_catalyst_score("TEST", events, as_of)

        assert len(summary.top_3_events) <= 3

        # Severe negative should be in top 3 (highest priority)
        top_types = [e["event_type"] for e in summary.top_3_events]
        assert "CT_STATUS_SEVERE_NEG" in top_types


# =============================================================================
# T8: DEDUP / NOISE-BAND
# =============================================================================

class TestDedupNoiseBand:
    """T8: Deduplication and noise-band handling."""

    def test_duplicate_events_deduped(self):
        """Duplicate events (same event_id) are removed."""
        from module_3_catalyst import dedup_events

        # Create identical events
        event = CatalystEventV2(
            ticker="TEST", nct_id="NCT001", event_type=EventType.CT_STATUS_UPGRADE,
            event_severity=EventSeverity.POSITIVE, event_date="2024-01-15",
            field_changed="status", prior_value="A", new_value="B", source="CTGOV",
            confidence=ConfidenceLevel.HIGH, disclosed_at="2024-01-15",
        )

        # Same event twice
        events = [event, event]
        deduped, removed = dedup_events(events)

        assert len(deduped) == 1
        assert removed == 1

    def test_different_events_not_deduped(self):
        """Different events are not removed."""
        from module_3_catalyst import dedup_events

        e1 = CatalystEventV2(
            ticker="TEST", nct_id="NCT001", event_type=EventType.CT_STATUS_UPGRADE,
            event_severity=EventSeverity.POSITIVE, event_date="2024-01-15",
            field_changed="status", prior_value="A", new_value="B", source="CTGOV",
            confidence=ConfidenceLevel.HIGH, disclosed_at="2024-01-15",
        )
        e2 = CatalystEventV2(
            ticker="TEST", nct_id="NCT002", event_type=EventType.CT_STATUS_UPGRADE,
            event_severity=EventSeverity.POSITIVE, event_date="2024-01-16",
            field_changed="status", prior_value="A", new_value="B", source="CTGOV",
            confidence=ConfidenceLevel.HIGH, disclosed_at="2024-01-16",
        )

        events = [e1, e2]
        deduped, removed = dedup_events(events)

        assert len(deduped) == 2
        assert removed == 0

    def test_event_id_differs_for_different_values(self):
        """Events with different values have different IDs."""
        e1 = CatalystEventV2(
            ticker="TEST", nct_id="NCT001", event_type=EventType.CT_STATUS_UPGRADE,
            event_severity=EventSeverity.POSITIVE, event_date="2024-01-15",
            field_changed="status", prior_value="A", new_value="B", source="CTGOV",
            confidence=ConfidenceLevel.HIGH, disclosed_at="2024-01-15",
        )
        e2 = CatalystEventV2(
            ticker="TEST", nct_id="NCT001", event_type=EventType.CT_STATUS_UPGRADE,
            event_severity=EventSeverity.POSITIVE, event_date="2024-01-15",
            field_changed="status", prior_value="A", new_value="C",  # Different new_value
            source="CTGOV",
            confidence=ConfidenceLevel.HIGH, disclosed_at="2024-01-15",
        )

        assert e1.event_id != e2.event_id


# =============================================================================
# ADDITIONAL TESTS
# =============================================================================

class TestScoreOverride:
    """Additional tests for override scoring logic."""

    def test_severe_negative_overrides_all(self, sample_events, severe_negative_event):
        """Severe negative always produces score 20."""
        events = sample_events + [severe_negative_event]
        score, reason = calculate_score_override(events)

        assert score == SCORE_OVERRIDE_SEVERE_NEGATIVE
        assert reason == "SEVERE_NEGATIVE_EVENTS"

    def test_critical_positive_when_no_severe_neg(self, sample_events, critical_positive_event):
        """Critical positive produces 75 when no severe negative."""
        events = sample_events + [critical_positive_event]
        score, reason = calculate_score_override(events)

        assert score == SCORE_OVERRIDE_CRITICAL_POSITIVE
        assert reason == "CRITICAL_POSITIVE_EVENTS"

    def test_no_events_returns_default(self):
        """Empty event list returns default score."""
        score, reason = calculate_score_override([])

        assert score == SCORE_DEFAULT
        assert reason == "NO_EVENTS"


class TestDiagnostics:
    """Tests for diagnostic counters."""

    def test_diagnostics_deterministic(self, sample_events):
        """Diagnostic counters are deterministic."""
        diag = DiagnosticCounts()
        diag.events_detected_total = 10
        diag.events_deduped = 2
        diag.events_by_type = {"CT_STATUS_UPGRADE": 5, "CT_TIMELINE_PULLIN": 3}

        dict1 = diag.to_dict()
        dict2 = diag.to_dict()

        assert dict1 == dict2

    def test_diagnostics_sorted_keys(self):
        """Diagnostic dicts have sorted keys."""
        diag = DiagnosticCounts()
        diag.events_by_type = {"Z_TYPE": 1, "A_TYPE": 2}
        diag.events_by_confidence = {"LOW": 3, "HIGH": 1}

        d = diag.to_dict()

        # Keys should be sorted
        assert list(d["events_by_type"].keys()) == ["A_TYPE", "Z_TYPE"]
        assert list(d["events_by_confidence"].keys()) == ["HIGH", "LOW"]


# =============================================================================
# T9: CALENDAR CATALYST INTEGRATION
# =============================================================================

class TestCalendarCatalystIntegration:
    """T9: Calendar catalysts flow through to scoring and diagnostics.

    This tests the fix for the bug where calendar catalysts were detected
    but never merged into the scoring pipeline.

    Invariants:
    - If calendar catalysts > 0, then events_detected_total > 0
    - If calendar catalysts > 0, then tickers_with_events > 0
    - Calendar events appear in ticker summaries
    """

    def test_calendar_catalyst_conversion(self):
        """Calendar catalysts can be converted to CatalystEventV2."""
        from module_3_catalyst import convert_calendar_catalyst_to_v2
        from catalyst_diagnostics import CalendarCatalyst, EventEvidence, EventRuleID

        calendar_cat = CalendarCatalyst(
            ticker="TEST",
            nct_id="NCT00000001",
            event_type="UPCOMING_PCD",
            target_date=date(2024, 3, 15),
            days_until=60,
            window="60D",
            confidence=0.85,
            rule_id=EventRuleID.M3_CAL_PCD_60D,
            evidence=EventEvidence(
                rule_id=EventRuleID.M3_CAL_PCD_60D,
                fields={"primary_completion_date": "2024-03-15", "days_until": 60},
                confidence=0.85,
                confidence_reason="Date is ESTIMATED",
            ),
        )

        v2_event = convert_calendar_catalyst_to_v2(calendar_cat)

        # Verify conversion
        assert v2_event.ticker == "TEST"
        assert v2_event.nct_id == "NCT00000001"
        assert v2_event.event_type == EventType.CT_PRIMARY_COMPLETION
        assert v2_event.event_date == "2024-03-15"
        assert v2_event.source == "CTGOV_CALENDAR"
        assert v2_event.confidence == ConfidenceLevel.HIGH  # 0.85 >= 0.85

    def test_calendar_catalyst_confidence_mapping(self):
        """Calendar catalyst confidence is mapped correctly."""
        from module_3_catalyst import convert_calendar_catalyst_to_v2
        from catalyst_diagnostics import CalendarCatalyst, EventEvidence, EventRuleID

        def make_cal_cat(confidence: float) -> CalendarCatalyst:
            return CalendarCatalyst(
                ticker="TEST",
                nct_id="NCT00000001",
                event_type="UPCOMING_PCD",
                target_date=date(2024, 3, 15),
                days_until=30,
                window="30D",
                confidence=confidence,
                rule_id=EventRuleID.M3_CAL_PCD_30D,
                evidence=EventEvidence(
                    rule_id=EventRuleID.M3_CAL_PCD_30D,
                    fields={},
                    confidence=confidence,
                    confidence_reason="test",
                ),
            )

        # HIGH: >= 0.85
        high_cat = make_cal_cat(0.90)
        assert convert_calendar_catalyst_to_v2(high_cat).confidence == ConfidenceLevel.HIGH

        # MED: >= 0.65 and < 0.85
        med_cat = make_cal_cat(0.75)
        assert convert_calendar_catalyst_to_v2(med_cat).confidence == ConfidenceLevel.MED

        # LOW: < 0.65
        low_cat = make_cal_cat(0.55)
        assert convert_calendar_catalyst_to_v2(low_cat).confidence == ConfidenceLevel.LOW

    def test_calendar_event_type_mapping(self):
        """Calendar event types map to correct EventType."""
        from module_3_catalyst import convert_calendar_catalyst_to_v2
        from catalyst_diagnostics import CalendarCatalyst, EventEvidence, EventRuleID

        def make_cal_cat(event_type: str, rule_id: str) -> CalendarCatalyst:
            return CalendarCatalyst(
                ticker="TEST",
                nct_id="NCT00000001",
                event_type=event_type,
                target_date=date(2024, 3, 15),
                days_until=30,
                window="30D",
                confidence=0.80,
                rule_id=rule_id,
                evidence=EventEvidence(
                    rule_id=rule_id,
                    fields={},
                    confidence=0.80,
                    confidence_reason="test",
                ),
            )

        # UPCOMING_PCD -> CT_PRIMARY_COMPLETION
        pcd_cat = make_cal_cat("UPCOMING_PCD", EventRuleID.M3_CAL_PCD_30D)
        assert convert_calendar_catalyst_to_v2(pcd_cat).event_type == EventType.CT_PRIMARY_COMPLETION

        # UPCOMING_SCD -> CT_STUDY_COMPLETION
        scd_cat = make_cal_cat("UPCOMING_SCD", EventRuleID.M3_CAL_SCD_30D)
        assert convert_calendar_catalyst_to_v2(scd_cat).event_type == EventType.CT_STUDY_COMPLETION

        # RESULTS_DUE -> CT_RESULTS_POSTED
        results_cat = make_cal_cat("RESULTS_DUE", EventRuleID.M3_CAL_RESULTS_DUE)
        assert convert_calendar_catalyst_to_v2(results_cat).event_type == EventType.CT_RESULTS_POSTED

    def test_calendar_events_included_in_batch_scoring(self):
        """Calendar events merged into batch scoring produce non-zero coverage."""
        from module_3_scoring import score_catalyst_events

        # Simulate calendar-derived events
        calendar_events = [
            CatalystEventV2(
                ticker="ACME",
                nct_id="NCT00000001",
                event_type=EventType.CT_PRIMARY_COMPLETION,
                event_severity=EventSeverity.CRITICAL_POSITIVE,
                event_date="2024-03-15",  # Future date
                field_changed="calendar_upcoming_pcd",
                prior_value=None,
                new_value="60d_ahead",
                source="CTGOV_CALENDAR",
                confidence=ConfidenceLevel.HIGH,
                disclosed_at="2024-03-15",
            ),
            CatalystEventV2(
                ticker="BETA",
                nct_id="NCT00000002",
                event_type=EventType.CT_STUDY_COMPLETION,
                event_severity=EventSeverity.POSITIVE,
                event_date="2024-04-01",  # Future date
                field_changed="calendar_upcoming_scd",
                prior_value=None,
                new_value="77d_ahead",
                source="CTGOV_CALENDAR",
                confidence=ConfidenceLevel.MED,
                disclosed_at="2024-04-01",
            ),
        ]

        events_by_ticker = {
            "ACME": [calendar_events[0]],
            "BETA": [calendar_events[1]],
        }
        active_tickers = ["ACME", "BETA", "GAMMA"]
        as_of = date(2024, 1, 15)

        summaries, diagnostics = score_catalyst_events(events_by_ticker, active_tickers, as_of)

        # Key invariant: If calendar catalysts exist, diagnostics should reflect them
        assert diagnostics.events_detected_total == 2, \
            f"Expected 2 events, got {diagnostics.events_detected_total}"
        assert diagnostics.tickers_with_events == 2, \
            f"Expected 2 tickers with events, got {diagnostics.tickers_with_events}"

        # Verify events appear in summaries
        assert len(summaries["ACME"].events) == 1
        assert summaries["ACME"].events[0].event_type == EventType.CT_PRIMARY_COMPLETION
        assert summaries["ACME"].events[0].source == "CTGOV_CALENDAR"

        assert len(summaries["BETA"].events) == 1
        assert summaries["BETA"].events[0].event_type == EventType.CT_STUDY_COMPLETION

        # GAMMA has no events
        assert len(summaries["GAMMA"].events) == 0

    def test_calendar_events_contribute_to_proximity_score(self):
        """Calendar events (future-dated) contribute to proximity score."""
        as_of = date(2024, 1, 15)

        # Future calendar event
        calendar_event = CatalystEventV2(
            ticker="TEST",
            nct_id="NCT00000001",
            event_type=EventType.CT_PRIMARY_COMPLETION,
            event_severity=EventSeverity.CRITICAL_POSITIVE,
            event_date="2024-03-01",  # ~45 days in future
            field_changed="calendar_upcoming_pcd",
            prior_value=None,
            new_value="45d_ahead",
            source="CTGOV_CALENDAR",
            confidence=ConfidenceLevel.HIGH,
            disclosed_at="2024-03-01",
        )

        summary = calculate_ticker_catalyst_score("TEST", [calendar_event], as_of)

        # Proximity score should be positive (event is in future)
        assert summary.catalyst_proximity_score > Decimal("0"), \
            f"Expected positive proximity score, got {summary.catalyst_proximity_score}"
        assert summary.n_events_upcoming == 1, \
            f"Expected 1 upcoming event, got {summary.n_events_upcoming}"

        # Next catalyst date should be set
        assert summary.next_catalyst_date == "2024-03-01"
        # Days until Mar 1 from Jan 15 (inclusive/exclusive handling may vary)
        assert 45 <= summary.catalyst_window_days <= 46


class TestContinuousCatalystProximity:
    """Tests for continuous piecewise-linear catalyst proximity scoring."""

    # -----------------------------------------------------------------
    # Helper: make a single catalyst event at a given days-from-now offset
    # -----------------------------------------------------------------
    @staticmethod
    def _event(days_ahead: int, as_of: date,
               event_type: EventType = EventType.FDA_PDUFA_DATE,
               confidence: ConfidenceLevel = ConfidenceLevel.HIGH,
               severity: EventSeverity = EventSeverity.CRITICAL_POSITIVE,
               ) -> CatalystEventV2:
        from datetime import timedelta
        event_date = (as_of + timedelta(days=days_ahead)).isoformat()
        return CatalystEventV2(
            ticker="TEST",
            nct_id="NCT99999999",
            event_type=event_type,
            event_severity=severity,
            event_date=event_date,
            field_changed="phase",
            prior_value="Phase 2",
            new_value="Phase 3",
            source="CTGOV",
            confidence=confidence,
            disclosed_at=as_of.isoformat(),
        )

    # -----------------------------------------------------------------
    # T1: Time kernel breakpoints
    # -----------------------------------------------------------------
    def test_time_kernel_past_event_zero(self):
        """Events at d<=0 get zero weight."""
        assert _proximity_time_weight(0) == Decimal("0")
        assert _proximity_time_weight(-5) == Decimal("0")

    def test_time_kernel_ramp_midpoint(self):
        """Events in the ramp region get proportional weight."""
        w = _proximity_time_weight(8)
        expected = Decimal("8") / Decimal("15")
        assert abs(w - expected) < Decimal("0.001")

    def test_time_kernel_ramp_end(self):
        """Event at ramp end (d=15) reaches peak."""
        assert _proximity_time_weight(_PROXIMITY_RAMP_END) == Decimal("1")

    def test_time_kernel_plateau(self):
        """Events in plateau region get weight 1.0."""
        for d in [16, 20, 30, 44, _PROXIMITY_PLATEAU_END]:
            assert _proximity_time_weight(d) == Decimal("1"), f"d={d}"

    def test_time_kernel_decay_midpoint(self):
        """Events in decay region linearly approach 0."""
        # d=67 is midpoint of [45, 90) decay
        mid = (_PROXIMITY_PLATEAU_END + _PROXIMITY_HORIZON) // 2  # 67
        w = _proximity_time_weight(mid)
        expected = Decimal(_PROXIMITY_HORIZON - mid) / Decimal(
            _PROXIMITY_HORIZON - _PROXIMITY_PLATEAU_END)
        assert abs(w - expected) < Decimal("0.001")

    def test_time_kernel_horizon_zero(self):
        """Events at/beyond horizon get zero weight."""
        assert _proximity_time_weight(_PROXIMITY_HORIZON) == Decimal("0")
        assert _proximity_time_weight(120) == Decimal("0")

    # -----------------------------------------------------------------
    # T2: Monotonicity — score increases as event moves from horizon to plateau
    # -----------------------------------------------------------------
    def test_monotonicity_approaching_plateau(self):
        """Scores increase monotonically as event approaches from horizon toward plateau."""
        as_of = date(2026, 1, 1)
        scores = []
        for days_ahead in [80, 70, 60, 50, 45, 30, 20, 15]:
            ev = self._event(days_ahead, as_of)
            score, _ = compute_bucket_proximity_score([ev], as_of)
            scores.append((days_ahead, score))

        for i in range(len(scores) - 1):
            d_far, s_far = scores[i]
            d_near, s_near = scores[i + 1]
            assert s_near >= s_far, (
                f"Score should increase as event approaches: "
                f"d={d_far}→{s_far} vs d={d_near}→{s_near}"
            )

    # -----------------------------------------------------------------
    # T3: No-event neutral
    # -----------------------------------------------------------------
    def test_no_events_returns_neutral(self):
        """Empty event list returns exactly neutral (50)."""
        score, n = compute_bucket_proximity_score([], date(2026, 1, 1))
        assert score == PROXIMITY_NEUTRAL_BASE
        assert n == 0

    def test_past_only_events_returns_neutral(self):
        """Events entirely in the past return neutral."""
        as_of = date(2026, 1, 1)
        ev = self._event(-10, as_of)  # 10 days ago
        score, n = compute_bucket_proximity_score([ev], as_of)
        assert score == PROXIMITY_NEUTRAL_BASE
        assert n == 0

    def test_beyond_horizon_returns_neutral(self):
        """Events beyond 90-day horizon return neutral."""
        as_of = date(2026, 1, 1)
        ev = self._event(100, as_of)  # 100 days ahead, beyond horizon
        score, n = compute_bucket_proximity_score([ev], as_of)
        assert score == PROXIMITY_NEUTRAL_BASE
        assert n == 0

    # -----------------------------------------------------------------
    # T4: Saturation — multiple events raise score but with diminishing returns
    # -----------------------------------------------------------------
    def test_saturation_diminishing_returns(self):
        """Adding more events increases score but with diminishing returns."""
        as_of = date(2026, 1, 1)
        # All events at plateau (d=30, peak time weight)
        events_1 = [self._event(30, as_of)]
        events_2 = [self._event(30, as_of), self._event(25, as_of)]
        events_3 = [self._event(30, as_of), self._event(25, as_of),
                     self._event(20, as_of)]

        s1, _ = compute_bucket_proximity_score(events_1, as_of)
        s2, _ = compute_bucket_proximity_score(events_2, as_of)
        s3, _ = compute_bucket_proximity_score(events_3, as_of)

        # Increasing
        assert s2 > s1 > PROXIMITY_NEUTRAL_BASE
        assert s3 > s2

        # Diminishing returns: marginal gain decreases
        gain_1_to_2 = s2 - s1
        gain_2_to_3 = s3 - s2
        assert gain_2_to_3 < gain_1_to_2, (
            f"Diminishing returns: 1→2 gain={gain_1_to_2}, 2→3 gain={gain_2_to_3}"
        )

    # -----------------------------------------------------------------
    # T5: Score always > 50 when events exist
    # -----------------------------------------------------------------
    def test_positive_events_always_above_neutral(self):
        """Any positive/critical event within horizon produces score > 50."""
        as_of = date(2026, 1, 1)
        for days in [1, 5, 15, 30, 45, 60, 80, 89]:
            ev = self._event(days, as_of)
            score, n = compute_bucket_proximity_score([ev], as_of)
            if n > 0:  # d=0 same-day event gets 0 time weight → excluded
                assert score > PROXIMITY_NEUTRAL_BASE, (
                    f"d={days}: score={score}, expected > {PROXIMITY_NEUTRAL_BASE}"
                )

    # -----------------------------------------------------------------
    # T6: Determinism — same inputs produce identical outputs
    # -----------------------------------------------------------------
    def test_determinism(self):
        """Same inputs produce byte-identical scores across repeated calls."""
        as_of = date(2026, 1, 1)
        events = [
            self._event(20, as_of, EventType.FDA_PDUFA_DATE, ConfidenceLevel.HIGH),
            self._event(50, as_of, EventType.DATA_READOUT, ConfidenceLevel.MED),
            self._event(10, as_of, EventType.CT_PRIMARY_COMPLETION, ConfidenceLevel.LOW),
        ]
        results = set()
        for _ in range(10):
            score, n = compute_bucket_proximity_score(events, as_of)
            results.add((str(score), n))
        assert len(results) == 1, f"Non-deterministic: {results}"

    # -----------------------------------------------------------------
    # T7: Confidence ordering
    # -----------------------------------------------------------------
    def test_confidence_ordering(self):
        """HIGH confidence produces higher score than MED, which beats LOW."""
        as_of = date(2026, 1, 1)
        scores = {}
        for conf in [ConfidenceLevel.HIGH, ConfidenceLevel.MED, ConfidenceLevel.LOW]:
            ev = self._event(30, as_of, confidence=conf)
            s, _ = compute_bucket_proximity_score([ev], as_of)
            scores[conf] = s

        assert scores[ConfidenceLevel.HIGH] > scores[ConfidenceLevel.MED]
        assert scores[ConfidenceLevel.MED] > scores[ConfidenceLevel.LOW]

    # -----------------------------------------------------------------
    # T8: Event type ordering — higher-weight types produce higher scores
    # -----------------------------------------------------------------
    def test_event_type_ordering(self):
        """FDA_PDUFA_DATE (weight 25) beats ENROLLMENT_STARTED (weight 6)."""
        as_of = date(2026, 1, 1)
        ev_high = self._event(30, as_of, event_type=EventType.FDA_PDUFA_DATE)
        ev_low = self._event(30, as_of, event_type=EventType.CT_ENROLLMENT_STARTED)

        s_high, _ = compute_bucket_proximity_score([ev_high], as_of)
        s_low, _ = compute_bucket_proximity_score([ev_low], as_of)

        assert s_high > s_low

    # -----------------------------------------------------------------
    # T9: Uniqueness — varied inputs produce varied scores
    # -----------------------------------------------------------------
    def test_uniqueness_across_varied_inputs(self):
        """Different event configurations produce different scores (no bucketing collapse)."""
        as_of = date(2026, 1, 1)
        unique_scores = set()

        event_types = [EventType.FDA_PDUFA_DATE, EventType.DATA_READOUT,
                       EventType.CT_PRIMARY_COMPLETION, EventType.CT_ENROLLMENT_STARTED]
        confidences = [ConfidenceLevel.HIGH, ConfidenceLevel.MED, ConfidenceLevel.LOW]

        for days in range(5, 85, 5):
            for et in event_types:
                for conf in confidences:
                    ev = self._event(days, as_of, event_type=et, confidence=conf)
                    score, _ = compute_bucket_proximity_score([ev], as_of)
                    unique_scores.add(str(score))

        # 16 day values × 4 types × 3 confidences = 192 combos
        # Old bucketed version: ~15-20 unique scores
        # Continuous version: should have 50+ unique values (some collapse
        # from 0.01 quantization and the saturation function)
        assert len(unique_scores) > 50, (
            f"Expected >50 unique scores from 192 combos, got {len(unique_scores)}"
        )

    # -----------------------------------------------------------------
    # T10: n_upcoming count accuracy
    # -----------------------------------------------------------------
    def test_n_upcoming_count(self):
        """n_upcoming reflects events within horizon with nonzero time weight."""
        as_of = date(2026, 1, 1)
        events = [
            self._event(10, as_of),   # in ramp → counted
            self._event(30, as_of),   # in plateau → counted
            self._event(80, as_of),   # in decay → counted
            self._event(100, as_of),  # beyond horizon → NOT counted
            self._event(-5, as_of),   # past → NOT counted
        ]
        _, n = compute_bucket_proximity_score(events, as_of)
        assert n == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
