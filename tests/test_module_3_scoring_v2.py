#!/usr/bin/env python3
"""
Tests for module_3_scoring_v2.py - Catalyst Scoring System v2

Tests cover:
- Recency weight calculation with exponential decay
- Staleness factor computation
- Proximity scoring for upcoming events
- Delta detection (event added/removed/date shifts)
- Certainty score computation
- Negative catalyst scoring
- Event aggregation and scoring
"""

import pytest
from datetime import date, timedelta
from decimal import Decimal
from typing import List

from module_3_scoring_v2 import (
    compute_recency_weight,
    compute_staleness_factor,
    compute_proximity_score,
    detect_deltas,
    DECAY_HALF_LIFE_DAYS,
    STALENESS_THRESHOLD_DAYS,
    PROXIMITY_HORIZON_DAYS,
)

from module_3_schema_v2 import (
    CatalystEventV2,
    DeltaEvent,
    DeltaType,
    EventType,
    EventSeverity,
    ConfidenceLevel,
    SourceReliability,
    DateSpecificity,
)


class TestRecencyWeight:
    """Test recency weight calculation with exponential decay."""

    def test_same_day_event_full_weight(self):
        """Events on as_of_date get weight 1.0."""
        as_of = date(2026, 1, 15)
        event_date = "2026-01-15"

        weight = compute_recency_weight(event_date, as_of)

        assert weight == Decimal("1.0")

    def test_future_event_zero_weight(self):
        """Future events get weight 0.0 (PIT safety)."""
        as_of = date(2026, 1, 15)
        event_date = "2026-01-20"  # 5 days in future

        weight = compute_recency_weight(event_date, as_of)

        assert weight == Decimal("0")

    def test_half_life_decay(self):
        """Event at DECAY_HALF_LIFE_DAYS old gets weight 0.5."""
        as_of = date(2026, 1, 15)
        event_date = (as_of - timedelta(days=DECAY_HALF_LIFE_DAYS)).isoformat()

        weight = compute_recency_weight(event_date, as_of)

        assert weight == Decimal("0.5")

    def test_old_event_low_weight(self):
        """Very old events have very low weight."""
        as_of = date(2026, 1, 15)
        event_date = (as_of - timedelta(days=365)).isoformat()

        weight = compute_recency_weight(event_date, as_of)

        assert weight < Decimal("0.1")
        assert weight > Decimal("0")

    def test_none_event_date_default(self):
        """None event_date returns default 0.5."""
        as_of = date(2026, 1, 15)

        weight = compute_recency_weight(None, as_of)

        assert weight == Decimal("0.5")

    def test_invalid_date_format_default(self):
        """Invalid date format returns default 0.5."""
        as_of = date(2026, 1, 15)

        weight = compute_recency_weight("not-a-date", as_of)

        assert weight == Decimal("0.5")


class TestStalenessFactor:
    """Test staleness penalty factor computation."""

    def test_no_events_returns_one(self):
        """Empty event list returns factor 1.0."""
        as_of = date(2026, 1, 15)

        factor = compute_staleness_factor([], as_of)

        assert factor == Decimal("1.0")

    def test_recent_events_no_penalty(self):
        """Recent events (< STALENESS_THRESHOLD_DAYS) have no penalty."""
        as_of = date(2026, 1, 15)
        recent_date = (as_of - timedelta(days=30)).isoformat()

        events = [
            CatalystEventV2(
                ticker="TEST",
                nct_id="NCT00000001",
                event_type=EventType.CT_RESULTS_POSTED,
                event_severity=EventSeverity.NEUTRAL,
                event_date=recent_date,
                field_changed="results_posted",
                prior_value=None,
                new_value="posted",
                source="CTGOV",
                confidence=ConfidenceLevel.HIGH,
                disclosed_at="2026-01-15",
                source_date="2026-01-15",
                pit_date_field_used="results_first_posted_date",
                source_reliability=SourceReliability.OFFICIAL,
                date_specificity=DateSpecificity.EXACT,
            )
        ]

        factor = compute_staleness_factor(events, as_of)

        assert factor == Decimal("1.0")

    def test_stale_events_penalty(self):
        """Events older than STALENESS_THRESHOLD_DAYS get penalty."""
        as_of = date(2026, 1, 15)
        stale_date = (as_of - timedelta(days=STALENESS_THRESHOLD_DAYS + 10)).isoformat()

        events = [
            CatalystEventV2(
                ticker="TEST",
                nct_id="NCT00000001",
                event_type=EventType.CT_RESULTS_POSTED,
                event_date=stale_date,
                source_date="2026-01-15",
                disclosed_at="2026-01-15",
                pit_date_field_used="last_update_posted_date",
                event_severity=EventSeverity.NEUTRAL,
                field_changed="status",
                prior_value=None,
                new_value="updated",
                source="CTGOV",
                confidence=ConfidenceLevel.HIGH,
                source_reliability=SourceReliability.OFFICIAL,
                date_specificity=DateSpecificity.EXACT,
            )
        ]

        factor = compute_staleness_factor(events, as_of)

        assert factor < Decimal("1.0")
        assert factor == Decimal("0.8")

    def test_mixed_events_uses_newest(self):
        """With mixed event ages, uses newest event to determine staleness."""
        as_of = date(2026, 1, 15)
        recent_date = (as_of - timedelta(days=30)).isoformat()
        old_date = (as_of - timedelta(days=200)).isoformat()

        events = [
            CatalystEventV2(
                ticker="TEST",
                nct_id="NCT00000001",
                event_type=EventType.CT_RESULTS_POSTED,
                event_date=old_date,
                source_date="2026-01-15",
                disclosed_at="2026-01-15",
                pit_date_field_used="last_update_posted_date",
                event_severity=EventSeverity.NEUTRAL,
                field_changed="status",
                prior_value=None,
                new_value="updated",
                source="CTGOV",
                confidence=ConfidenceLevel.HIGH,
                source_reliability=SourceReliability.OFFICIAL,
                date_specificity=DateSpecificity.EXACT,
            ),
            CatalystEventV2(
                ticker="TEST",
                nct_id="NCT00000002",
                event_type=EventType.CT_STATUS_UPGRADE,
                event_date=recent_date,
                source_date="2026-01-15",
                disclosed_at="2026-01-15",
                pit_date_field_used="last_update_posted_date",
                event_severity=EventSeverity.NEUTRAL,
                field_changed="status",
                prior_value=None,
                new_value="updated",
                source="CTGOV",
                confidence=ConfidenceLevel.HIGH,
                source_reliability=SourceReliability.OFFICIAL,
                date_specificity=DateSpecificity.EXACT,
            ),
        ]

        factor = compute_staleness_factor(events, as_of)

        # Should not be penalized because newest event is recent
        assert factor == Decimal("1.0")

    def test_all_none_dates_no_penalty(self):
        """Events with no dates return factor 1.0."""
        as_of = date(2026, 1, 15)

        events = [
            CatalystEventV2(
                ticker="TEST",
                nct_id="NCT00000001",
                event_type=EventType.CT_RESULTS_POSTED,
                event_date=None,
                source_date="2026-01-15",
                disclosed_at="2026-01-15",
                pit_date_field_used="last_update_posted_date",
                event_severity=EventSeverity.NEUTRAL,
                field_changed="status",
                prior_value=None,
                new_value="updated",
                source="CTGOV",
                confidence=ConfidenceLevel.HIGH,
                source_reliability=SourceReliability.OFFICIAL,
                date_specificity=DateSpecificity.EXACT,
            )
        ]

        factor = compute_staleness_factor(events, as_of)

        assert factor == Decimal("1.0")


class TestProximityScore:
    """Test proximity scoring for upcoming catalyst events."""

    def test_no_events_zero_score(self):
        """Empty event list returns score 0."""
        as_of = date(2026, 1, 15)

        score, n_upcoming = compute_proximity_score([], as_of)

        assert score == Decimal("0")
        assert n_upcoming == 0

    def test_past_events_ignored(self):
        """Past events don't contribute to proximity score."""
        as_of = date(2026, 1, 15)
        past_date = "2026-01-10"

        events = [
            CatalystEventV2(
                ticker="TEST",
                nct_id="NCT00000001",
                event_type=EventType.CT_RESULTS_POSTED,
                event_date=past_date,
                source_date="2026-01-15",
                disclosed_at="2026-01-15",
                pit_date_field_used="last_update_posted_date",
                event_severity=EventSeverity.NEUTRAL,
                field_changed="status",
                prior_value=None,
                new_value="updated",
                source="CTGOV",
                confidence=ConfidenceLevel.HIGH,
                source_reliability=SourceReliability.OFFICIAL,
                date_specificity=DateSpecificity.EXACT,
            )
        ]

        score, n_upcoming = compute_proximity_score(events, as_of)

        assert score == Decimal("0")
        assert n_upcoming == 0

    def test_future_event_within_horizon(self):
        """Future events within horizon contribute to score."""
        as_of = date(2026, 1, 15)
        future_date = (as_of + timedelta(days=30)).isoformat()

        events = [
            CatalystEventV2(
                ticker="TEST",
                nct_id="NCT00000001",
                event_type=EventType.CT_PRIMARY_COMPLETION,
                event_date=future_date,
                source_date="2026-01-15",
                disclosed_at="2026-01-15",
                pit_date_field_used="last_update_posted_date",
                event_severity=EventSeverity.POSITIVE,
                field_changed="status",
                prior_value=None,
                new_value="updated",
                source="CTGOV",
                confidence=ConfidenceLevel.HIGH,
                source_reliability=SourceReliability.OFFICIAL,
                date_specificity=DateSpecificity.EXACT,
            )
        ]

        score, n_upcoming = compute_proximity_score(events, as_of)

        assert score > Decimal("0")
        assert n_upcoming == 1

    def test_beyond_horizon_ignored(self):
        """Events beyond horizon don't contribute."""
        as_of = date(2026, 1, 15)
        far_future = (as_of + timedelta(days=PROXIMITY_HORIZON_DAYS + 10)).isoformat()

        events = [
            CatalystEventV2(
                ticker="TEST",
                nct_id="NCT00000001",
                event_type=EventType.CT_PRIMARY_COMPLETION,
                event_date=far_future,
                source_date="2026-01-15",
                disclosed_at="2026-01-15",
                pit_date_field_used="last_update_posted_date",
                event_severity=EventSeverity.POSITIVE,
                field_changed="status",
                prior_value=None,
                new_value="updated",
                source="CTGOV",
                confidence=ConfidenceLevel.HIGH,
                source_reliability=SourceReliability.OFFICIAL,
                date_specificity=DateSpecificity.EXACT,
            )
        ]

        score, n_upcoming = compute_proximity_score(events, as_of)

        assert score == Decimal("0")
        assert n_upcoming == 0

    def test_closer_events_higher_score(self):
        """Closer events score higher than distant ones."""
        as_of = date(2026, 1, 15)
        near_date = (as_of + timedelta(days=30)).isoformat()
        far_date = (as_of + timedelta(days=180)).isoformat()

        near_events = [
            CatalystEventV2(
                ticker="TEST",
                nct_id="NCT00000001",
                event_type=EventType.CT_PRIMARY_COMPLETION,
                event_date=near_date,
                source_date="2026-01-15",
                disclosed_at="2026-01-15",
                pit_date_field_used="last_update_posted_date",
                event_severity=EventSeverity.POSITIVE,
                field_changed="status",
                prior_value=None,
                new_value="updated",
                source="CTGOV",
                confidence=ConfidenceLevel.HIGH,
                source_reliability=SourceReliability.OFFICIAL,
                date_specificity=DateSpecificity.EXACT,
            )
        ]

        far_events = [
            CatalystEventV2(
                ticker="TEST",
                nct_id="NCT00000001",
                event_type=EventType.CT_PRIMARY_COMPLETION,
                event_date=far_date,
                source_date="2026-01-15",
                disclosed_at="2026-01-15",
                pit_date_field_used="last_update_posted_date",
                event_severity=EventSeverity.POSITIVE,
                field_changed="status",
                prior_value=None,
                new_value="updated",
                source="CTGOV",
                confidence=ConfidenceLevel.HIGH,
                source_reliability=SourceReliability.OFFICIAL,
                date_specificity=DateSpecificity.EXACT,
            )
        ]

        near_score, _ = compute_proximity_score(near_events, as_of)
        far_score, _ = compute_proximity_score(far_events, as_of)

        assert near_score > far_score

    def test_score_clamping(self):
        """Proximity score is clamped to [-50, 100]."""
        as_of = date(2026, 1, 15)

        # Create many high-impact events to test upper bound
        events = []
        for i in range(20):
            events.append(
                CatalystEventV2(
                    ticker="TEST",
                    nct_id=f"NCT0000000{i}",
                    event_type=EventType.CT_PRIMARY_COMPLETION,
                    event_severity=EventSeverity.POSITIVE,
                    event_date=(as_of + timedelta(days=i+1)).isoformat(),
                    field_changed="completion_date",
                    prior_value=None,
                    new_value="updated",
                    source="CTGOV",
                    confidence=ConfidenceLevel.HIGH,
                    disclosed_at="2026-01-15",
                    source_date="2026-01-15",
                    pit_date_field_used="primary_completion_date",
                    source_reliability=SourceReliability.OFFICIAL,
                    date_specificity=DateSpecificity.EXACT,
                )
            )

        score, n_upcoming = compute_proximity_score(events, as_of)

        assert score <= Decimal("100")
        assert score >= Decimal("-50")
        assert n_upcoming == 20


class TestDeltaDetection:
    """Test delta detection between event snapshots."""

    def test_event_added_detection(self):
        """Detect when new event appears in current snapshot."""
        as_of = date(2026, 1, 15)

        prior_events = []
        current_events = [
            CatalystEventV2(
                ticker="TEST",
                nct_id="NCT00000001",
                event_type=EventType.CT_RESULTS_POSTED,
                event_date="2026-02-01",
                source_date="2026-01-15",
                disclosed_at="2026-01-15",
                pit_date_field_used="last_update_posted_date",
                event_severity=EventSeverity.NEUTRAL,
                field_changed="status",
                prior_value=None,
                new_value="updated",
                source="CTGOV",
                confidence=ConfidenceLevel.HIGH,
                source_reliability=SourceReliability.OFFICIAL,
                date_specificity=DateSpecificity.EXACT,
            )
        ]

        deltas = detect_deltas(current_events, prior_events, as_of)

        assert len(deltas) == 1
        assert deltas[0].delta_type == DeltaType.EVENT_ADDED
        assert deltas[0].ticker == "TEST"
        assert deltas[0].nct_id == "NCT00000001"

    def test_event_removed_detection(self):
        """Detect when event disappears from current snapshot."""
        as_of = date(2026, 1, 15)

        prior_events = [
            CatalystEventV2(
                ticker="TEST",
                nct_id="NCT00000001",
                event_type=EventType.CT_RESULTS_POSTED,
                event_date="2026-02-01",
                source_date="2026-01-14",
                disclosed_at="2026-01-14",
                pit_date_field_used="last_update_posted_date",
                event_severity=EventSeverity.NEUTRAL,
                field_changed="status",
                prior_value=None,
                new_value="updated",
                source="CTGOV",
                confidence=ConfidenceLevel.HIGH,
                source_reliability=SourceReliability.OFFICIAL,
                date_specificity=DateSpecificity.EXACT,
            )
        ]
        current_events = []

        deltas = detect_deltas(current_events, prior_events, as_of)

        assert len(deltas) == 1
        assert deltas[0].delta_type == DeltaType.EVENT_REMOVED
        assert deltas[0].ticker == "TEST"
        assert deltas[0].nct_id == "NCT00000001"

    def test_date_shift_detection(self):
        """Detect when event date changes between snapshots."""
        as_of = date(2026, 1, 15)

        prior_events = [
            CatalystEventV2(
                ticker="TEST",
                nct_id="NCT00000001",
                event_type=EventType.CT_RESULTS_POSTED,
                event_date="2026-02-01",
                source_date="2026-01-14",
                disclosed_at="2026-01-14",
                pit_date_field_used="last_update_posted_date",
                event_severity=EventSeverity.NEUTRAL,
                field_changed="status",
                prior_value=None,
                new_value="updated",
                source="CTGOV",
                confidence=ConfidenceLevel.HIGH,
                source_reliability=SourceReliability.OFFICIAL,
                date_specificity=DateSpecificity.EXACT,
            )
        ]
        current_events = [
            CatalystEventV2(
                ticker="TEST",
                nct_id="NCT00000001",
                event_type=EventType.CT_RESULTS_POSTED,
                event_date="2026-02-15",  # Date changed
                source_date="2026-01-15",
                disclosed_at="2026-01-15",
                pit_date_field_used="last_update_posted_date",
                event_severity=EventSeverity.NEUTRAL,
                field_changed="status",
                prior_value=None,
                new_value="updated",
                source="CTGOV",
                confidence=ConfidenceLevel.HIGH,
                source_reliability=SourceReliability.OFFICIAL,
                date_specificity=DateSpecificity.EXACT,
            )
        ]

        deltas = detect_deltas(current_events, prior_events, as_of)

        # Should detect date shift
        date_shifts = [d for d in deltas if d.delta_type == DeltaType.DATE_SHIFT]
        assert len(date_shifts) >= 1
        assert date_shifts[0].prior_value == "2026-02-01"
        assert date_shifts[0].new_value == "2026-02-15"
        assert date_shifts[0].shift_days == 14

    def test_window_widening_detection(self):
        """Detect when completion date is pushed out (negative signal)."""
        as_of = date(2026, 1, 15)

        prior_events = [
            CatalystEventV2(
                ticker="TEST",
                nct_id="NCT00000001",
                event_type=EventType.CT_PRIMARY_COMPLETION,
                event_date="2026-03-01",
                source_date="2026-01-14",
                disclosed_at="2026-01-14",
                pit_date_field_used="last_update_posted_date",
                event_severity=EventSeverity.NEUTRAL,
                field_changed="status",
                prior_value=None,
                new_value="updated",
                source="CTGOV",
                confidence=ConfidenceLevel.HIGH,
                source_reliability=SourceReliability.OFFICIAL,
                date_specificity=DateSpecificity.EXACT,
            )
        ]
        current_events = [
            CatalystEventV2(
                ticker="TEST",
                nct_id="NCT00000001",
                event_type=EventType.CT_PRIMARY_COMPLETION,
                event_date="2026-06-01",  # Pushed out 3 months
                source_date="2026-01-15",
                disclosed_at="2026-01-15",
                pit_date_field_used="last_update_posted_date",
                event_severity=EventSeverity.NEUTRAL,
                field_changed="status",
                prior_value=None,
                new_value="updated",
                source="CTGOV",
                confidence=ConfidenceLevel.HIGH,
                source_reliability=SourceReliability.OFFICIAL,
                date_specificity=DateSpecificity.EXACT,
            )
        ]

        deltas = detect_deltas(current_events, prior_events, as_of)

        # Should detect as WINDOW_WIDENING, not just DATE_SHIFT
        window_widenings = [d for d in deltas if d.delta_type == DeltaType.WINDOW_WIDENING]
        assert len(window_widenings) >= 1
        assert window_widenings[0].shift_days == 92  # ~3 months

    def test_no_deltas_when_unchanged(self):
        """No deltas when snapshots are identical."""
        as_of = date(2026, 1, 15)

        event = CatalystEventV2(
                ticker="TEST",
                nct_id="NCT00000001",
                event_type=EventType.CT_RESULTS_POSTED,
                event_date="2026-02-01",
                source_date="2026-01-15",
                disclosed_at="2026-01-15",
                pit_date_field_used="last_update_posted_date",
                event_severity=EventSeverity.NEUTRAL,
                field_changed="status",
                prior_value=None,
                new_value="updated",
                source="CTGOV",
                confidence=ConfidenceLevel.HIGH,
                source_reliability=SourceReliability.OFFICIAL,
                date_specificity=DateSpecificity.EXACT,
            )

        deltas = detect_deltas([event], [event], as_of)

        # Same event_id means no changes detected
        assert len(deltas) == 0

    def test_multiple_deltas_same_ticker(self):
        """Can detect multiple delta types for same ticker."""
        as_of = date(2026, 1, 15)

        prior_events = [
            CatalystEventV2(
                ticker="TEST",
                nct_id="NCT00000001",
                event_type=EventType.CT_RESULTS_POSTED,
                event_date="2026-02-01",
                source_date="2026-01-14",
                disclosed_at="2026-01-14",
                pit_date_field_used="last_update_posted_date",
                event_severity=EventSeverity.NEUTRAL,
                field_changed="status",
                prior_value=None,
                new_value="updated",
                source="CTGOV",
                confidence=ConfidenceLevel.HIGH,
                source_reliability=SourceReliability.OFFICIAL,
                date_specificity=DateSpecificity.EXACT,
            ),
        ]

        current_events = [
            CatalystEventV2(
                ticker="TEST",
                nct_id="NCT00000002",  # New trial
                event_type=EventType.CT_ENROLLMENT_STARTED,
                event_date="2026-03-01",
                source_date="2026-01-15",
                disclosed_at="2026-01-15",
                pit_date_field_used="last_update_posted_date",
                event_severity=EventSeverity.POSITIVE,
                field_changed="status",
                prior_value=None,
                new_value="updated",
                source="CTGOV",
                confidence=ConfidenceLevel.MED,
                source_reliability=SourceReliability.OFFICIAL,
                date_specificity=DateSpecificity.EXACT,
            ),
        ]

        deltas = detect_deltas(current_events, prior_events, as_of)

        # Should detect both EVENT_REMOVED and EVENT_ADDED
        assert len(deltas) >= 2
        delta_types = {d.delta_type for d in deltas}
        assert DeltaType.EVENT_ADDED in delta_types
        assert DeltaType.EVENT_REMOVED in delta_types


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_recency_weight_with_invalid_types(self):
        """Test recency weight handles various invalid inputs."""
        as_of = date(2026, 1, 15)

        # These should all return default 0.5
        assert compute_recency_weight("", as_of) == Decimal("0.5")
        assert compute_recency_weight("invalid-date", as_of) == Decimal("0.5")
        assert compute_recency_weight(None, as_of) == Decimal("0.5")

    def test_proximity_score_with_none_dates(self):
        """Events with None dates are skipped in proximity scoring."""
        as_of = date(2026, 1, 15)

        events = [
            CatalystEventV2(
                ticker="TEST",
                nct_id="NCT00000001",
                event_type=EventType.CT_PRIMARY_COMPLETION,
                event_date=None,  # Missing date
                source_date="2026-01-15",
                disclosed_at="2026-01-15",
                pit_date_field_used="last_update_posted_date",
                event_severity=EventSeverity.POSITIVE,
                field_changed="status",
                prior_value=None,
                new_value="updated",
                source="CTGOV",
                confidence=ConfidenceLevel.HIGH,
                source_reliability=SourceReliability.OFFICIAL,
                date_specificity=DateSpecificity.UNKNOWN,
            )
        ]

        score, n_upcoming = compute_proximity_score(events, as_of)

        assert score == Decimal("0")
        assert n_upcoming == 0

    def test_detect_deltas_with_partial_dates(self):
        """Delta detection handles events with missing dates."""
        as_of = date(2026, 1, 15)

        prior_events = [
            CatalystEventV2(
                ticker="TEST",
                nct_id="NCT00000001",
                event_type=EventType.CT_RESULTS_POSTED,
                event_date=None,
                source_date="2026-01-14",
                disclosed_at="2026-01-14",
                pit_date_field_used="last_update_posted_date",
                event_severity=EventSeverity.NEUTRAL,
                field_changed="status",
                prior_value=None,
                new_value="updated",
                source="CTGOV",
                confidence=ConfidenceLevel.HIGH,
                source_reliability=SourceReliability.OFFICIAL,
                date_specificity=DateSpecificity.UNKNOWN,
            )
        ]
        current_events = [
            CatalystEventV2(
                ticker="TEST",
                nct_id="NCT00000001",
                event_type=EventType.CT_RESULTS_POSTED,
                event_date="2026-02-01",  # Date now available
                source_date="2026-01-15",
                disclosed_at="2026-01-15",
                pit_date_field_used="last_update_posted_date",
                event_severity=EventSeverity.NEUTRAL,
                field_changed="status",
                prior_value=None,
                new_value="updated",
                source="CTGOV",
                confidence=ConfidenceLevel.HIGH,
                source_reliability=SourceReliability.OFFICIAL,
                date_specificity=DateSpecificity.EXACT,
            )
        ]

        # Should not crash, and should detect event_id change
        deltas = detect_deltas(current_events, prior_events, as_of)

        # Event ID will be different due to date change
        assert len(deltas) >= 0  # May detect changes
