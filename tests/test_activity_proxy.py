#!/usr/bin/env python3
"""
Tests for Activity Proxy Detection (Historical Data Workaround)

This tests the CT_ACTIVITY_PROXY event type and related functionality
that supplements diff-based detection when historical CT.gov snapshots
are limited or unavailable.

Test Coverage:
T1: Activity proxy event detection when only last_update_posted changes
T2: Activity proxy score computation with time decay
T3: Activity proxy lookback detection from current records
T4: Schema integration (EventType, severity mappings)
T5: Integration with TickerCatalystSummaryV2

Run with: pytest tests/test_activity_proxy.py -v
"""

import pytest
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from event_detector import (
    EventDetector,
    EventDetectorConfig,
    EventType,
    CatalystEvent,
    detect_activity_proxy_from_lookback,
    compute_activity_proxy_score,
)
from ctgov_adapter import CanonicalTrialRecord, CTGovStatus, CompletionType
from module_3_schema import (
    EventType as SchemaEventType,
    EventSeverity,
    ConfidenceLevel,
    EVENT_SEVERITY_MAP,
    EVENT_DEFAULT_CONFIDENCE,
    EVENT_TYPE_WEIGHT,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def as_of_date() -> date:
    """Standard as_of_date for tests."""
    return date(2026, 1, 15)


@pytest.fixture
def event_detector() -> EventDetector:
    """Event detector with default config."""
    return EventDetector(EventDetectorConfig())


@pytest.fixture
def sample_trial_record(as_of_date) -> CanonicalTrialRecord:
    """Sample canonical trial record."""
    return CanonicalTrialRecord(
        ticker="ACME",
        nct_id="NCT12345678",
        overall_status=CTGovStatus.RECRUITING,
        last_update_posted=as_of_date - timedelta(days=5),
        primary_completion_date=as_of_date + timedelta(days=90),
        primary_completion_type=CompletionType.ESTIMATED,
        completion_date=as_of_date + timedelta(days=180),
        completion_type=CompletionType.ESTIMATED,
        results_first_posted=None,
    )


@pytest.fixture
def prior_trial_record(as_of_date) -> CanonicalTrialRecord:
    """Prior trial record with older last_update_posted but same status/dates."""
    return CanonicalTrialRecord(
        ticker="ACME",
        nct_id="NCT12345678",
        overall_status=CTGovStatus.RECRUITING,  # Same status
        last_update_posted=as_of_date - timedelta(days=30),  # Older update
        primary_completion_date=as_of_date + timedelta(days=90),  # Same date
        primary_completion_type=CompletionType.ESTIMATED,
        completion_date=as_of_date + timedelta(days=180),  # Same date
        completion_type=CompletionType.ESTIMATED,
        results_first_posted=None,
    )


@pytest.fixture
def sample_trial_records_for_lookback(as_of_date) -> list[CanonicalTrialRecord]:
    """Multiple trial records for lookback testing."""
    return [
        # Recent update (5 days ago)
        CanonicalTrialRecord(
            ticker="ACME",
            nct_id="NCT00000001",
            overall_status=CTGovStatus.RECRUITING,
            last_update_posted=as_of_date - timedelta(days=5),
            primary_completion_date=as_of_date + timedelta(days=90),
            primary_completion_type=CompletionType.ESTIMATED,
            completion_date=None,
            completion_type=None,
            results_first_posted=None,
        ),
        # Medium recent (30 days ago)
        CanonicalTrialRecord(
            ticker="ACME",
            nct_id="NCT00000002",
            overall_status=CTGovStatus.ACTIVE_NOT_RECRUITING,
            last_update_posted=as_of_date - timedelta(days=30),
            primary_completion_date=as_of_date + timedelta(days=60),
            primary_completion_type=CompletionType.ESTIMATED,
            completion_date=None,
            completion_type=None,
            results_first_posted=None,
        ),
        # Old update (60 days ago)
        CanonicalTrialRecord(
            ticker="ACME",
            nct_id="NCT00000003",
            overall_status=CTGovStatus.COMPLETED,
            last_update_posted=as_of_date - timedelta(days=60),
            primary_completion_date=as_of_date - timedelta(days=10),
            primary_completion_type=CompletionType.ACTUAL,
            completion_date=None,
            completion_type=None,
            results_first_posted=None,
        ),
        # Very old (100 days ago - outside 90d window)
        CanonicalTrialRecord(
            ticker="ACME",
            nct_id="NCT00000004",
            overall_status=CTGovStatus.COMPLETED,
            last_update_posted=as_of_date - timedelta(days=100),
            primary_completion_date=as_of_date - timedelta(days=50),
            primary_completion_type=CompletionType.ACTUAL,
            completion_date=None,
            completion_type=None,
            results_first_posted=None,
        ),
    ]


# =============================================================================
# T1: ACTIVITY PROXY EVENT DETECTION
# =============================================================================

class TestActivityProxyEventDetection:
    """Tests for activity proxy event detection when only last_update_posted changes."""

    def test_activity_proxy_detected_when_only_update_posted_changes(
        self, event_detector, sample_trial_record, prior_trial_record, as_of_date
    ):
        """Activity proxy event should be generated when only last_update_posted changed."""
        events = event_detector.detect_events(
            sample_trial_record,
            prior_trial_record,
            as_of_date
        )

        assert len(events) == 1
        assert events[0].event_type == EventType.CT_ACTIVITY_PROXY
        assert events[0].direction == 'NEUTRAL'
        assert events[0].nct_id == "NCT12345678"

    def test_activity_proxy_not_detected_when_status_changes(
        self, event_detector, as_of_date
    ):
        """Activity proxy should NOT be generated when status changes (other event takes precedence)."""
        current = CanonicalTrialRecord(
            ticker="ACME",
            nct_id="NCT12345678",
            overall_status=CTGovStatus.COMPLETED,  # Status changed
            last_update_posted=as_of_date - timedelta(days=5),
            primary_completion_date=as_of_date - timedelta(days=10),
            primary_completion_type=CompletionType.ACTUAL,
            completion_date=None,
            completion_type=None,
            results_first_posted=None,
        )
        prior = CanonicalTrialRecord(
            ticker="ACME",
            nct_id="NCT12345678",
            overall_status=CTGovStatus.RECRUITING,  # Was recruiting
            last_update_posted=as_of_date - timedelta(days=30),
            primary_completion_date=as_of_date + timedelta(days=90),
            primary_completion_type=CompletionType.ESTIMATED,
            completion_date=None,
            completion_type=None,
            results_first_posted=None,
        )

        events = event_detector.detect_events(current, prior, as_of_date)

        # Should have status event, NOT activity proxy
        assert len(events) >= 1
        event_types = [e.event_type for e in events]
        assert EventType.CT_ACTIVITY_PROXY not in event_types
        assert EventType.CT_STATUS_UPGRADE in event_types

    def test_activity_proxy_not_detected_for_stale_update(
        self, event_detector, as_of_date
    ):
        """Activity proxy should NOT be generated for updates > 90 days old."""
        current = CanonicalTrialRecord(
            ticker="ACME",
            nct_id="NCT12345678",
            overall_status=CTGovStatus.RECRUITING,
            last_update_posted=as_of_date - timedelta(days=100),  # Too old
            primary_completion_date=as_of_date + timedelta(days=90),
            primary_completion_type=CompletionType.ESTIMATED,
            completion_date=None,
            completion_type=None,
            results_first_posted=None,
        )
        prior = CanonicalTrialRecord(
            ticker="ACME",
            nct_id="NCT12345678",
            overall_status=CTGovStatus.RECRUITING,
            last_update_posted=as_of_date - timedelta(days=150),
            primary_completion_date=as_of_date + timedelta(days=90),
            primary_completion_type=CompletionType.ESTIMATED,
            completion_date=None,
            completion_type=None,
            results_first_posted=None,
        )

        events = event_detector.detect_events(current, prior, as_of_date)

        # Should have no events (activity proxy not detected for stale updates)
        assert len(events) == 0

    def test_activity_proxy_confidence_is_low(
        self, event_detector, sample_trial_record, prior_trial_record, as_of_date
    ):
        """Activity proxy events should have low confidence (0.30)."""
        events = event_detector.detect_events(
            sample_trial_record,
            prior_trial_record,
            as_of_date
        )

        assert len(events) == 1
        assert events[0].confidence == 0.30

    def test_activity_proxy_impact_scales_with_recency(
        self, event_detector, as_of_date
    ):
        """Activity proxy impact should be higher for more recent updates."""
        # Very recent (3 days ago) - should have impact 2
        recent_current = CanonicalTrialRecord(
            ticker="ACME",
            nct_id="NCT12345678",
            overall_status=CTGovStatus.RECRUITING,
            last_update_posted=as_of_date - timedelta(days=3),
            primary_completion_date=as_of_date + timedelta(days=90),
            primary_completion_type=CompletionType.ESTIMATED,
            completion_date=None,
            completion_type=None,
            results_first_posted=None,
        )
        prior = CanonicalTrialRecord(
            ticker="ACME",
            nct_id="NCT12345678",
            overall_status=CTGovStatus.RECRUITING,
            last_update_posted=as_of_date - timedelta(days=30),
            primary_completion_date=as_of_date + timedelta(days=90),
            primary_completion_type=CompletionType.ESTIMATED,
            completion_date=None,
            completion_type=None,
            results_first_posted=None,
        )

        events = event_detector.detect_events(recent_current, prior, as_of_date)

        assert len(events) == 1
        assert events[0].impact == 2  # Recent update = higher impact


# =============================================================================
# T2: ACTIVITY PROXY SCORE COMPUTATION
# =============================================================================

class TestActivityProxyScoreComputation:
    """Tests for activity proxy score computation with time decay."""

    def test_activity_proxy_score_basic(
        self, sample_trial_records_for_lookback, as_of_date
    ):
        """Basic activity proxy score computation."""
        result = compute_activity_proxy_score(
            sample_trial_records_for_lookback,
            as_of_date,
            lookback_days=90,
        )

        # Should count 3 trials within 90 days (NCT1, NCT2, NCT3)
        assert result['activity_count_90d'] == 3
        # Should count 2 trials within 30 days (NCT1, NCT2)
        assert result['activity_count_30d'] == 2
        # Score should be positive
        assert result['activity_proxy_score'] > 0
        # Should have 3 NCT IDs
        assert len(result['recent_nct_ids']) == 3

    def test_activity_proxy_score_time_decay(
        self, as_of_date
    ):
        """Activity proxy score should decay with time."""
        # Recent trial (5 days ago)
        recent = [CanonicalTrialRecord(
            ticker="ACME",
            nct_id="NCT00000001",
            overall_status=CTGovStatus.RECRUITING,
            last_update_posted=as_of_date - timedelta(days=5),
            primary_completion_date=as_of_date + timedelta(days=90),
            primary_completion_type=CompletionType.ESTIMATED,
            completion_date=None,
            completion_type=None,
            results_first_posted=None,
        )]

        # Old trial (85 days ago)
        old = [CanonicalTrialRecord(
            ticker="ACME",
            nct_id="NCT00000002",
            overall_status=CTGovStatus.RECRUITING,
            last_update_posted=as_of_date - timedelta(days=85),
            primary_completion_date=as_of_date + timedelta(days=90),
            primary_completion_type=CompletionType.ESTIMATED,
            completion_date=None,
            completion_type=None,
            results_first_posted=None,
        )]

        recent_result = compute_activity_proxy_score(recent, as_of_date, 90)
        old_result = compute_activity_proxy_score(old, as_of_date, 90)

        # Recent should score higher due to time decay
        assert recent_result['activity_proxy_score'] > old_result['activity_proxy_score']

    def test_activity_proxy_score_empty_trials(self, as_of_date):
        """Empty trial list should return zero score."""
        result = compute_activity_proxy_score([], as_of_date, 90)

        assert result['activity_count_90d'] == 0
        assert result['activity_count_30d'] == 0
        assert result['activity_proxy_score'] == 0


# =============================================================================
# T3: ACTIVITY PROXY LOOKBACK DETECTION
# =============================================================================

class TestActivityProxyLookbackDetection:
    """Tests for activity proxy detection from current records (lookback method)."""

    def test_detect_activity_proxy_from_lookback(
        self, sample_trial_records_for_lookback, as_of_date
    ):
        """Detect activity proxy events from lookback window."""
        events_by_ticker = detect_activity_proxy_from_lookback(
            sample_trial_records_for_lookback,
            as_of_date,
            lookback_days=90,
        )

        # Should have events for ACME ticker
        assert "ACME" in events_by_ticker
        events = events_by_ticker["ACME"]

        # Should have 3 events (trials within 90 days)
        assert len(events) == 3

        # All events should be activity proxy type
        for event in events:
            assert event.event_type == EventType.CT_ACTIVITY_PROXY

    def test_lookback_excludes_old_trials(
        self, sample_trial_records_for_lookback, as_of_date
    ):
        """Lookback should exclude trials older than lookback window."""
        events_by_ticker = detect_activity_proxy_from_lookback(
            sample_trial_records_for_lookback,
            as_of_date,
            lookback_days=90,
        )

        # NCT00000004 (100 days old) should be excluded
        events = events_by_ticker.get("ACME", [])
        nct_ids = [e.nct_id for e in events]
        assert "NCT00000004" not in nct_ids

    def test_lookback_with_custom_window(
        self, sample_trial_records_for_lookback, as_of_date
    ):
        """Lookback with shorter window should return fewer events."""
        events_30d = detect_activity_proxy_from_lookback(
            sample_trial_records_for_lookback,
            as_of_date,
            lookback_days=30,
        )
        events_90d = detect_activity_proxy_from_lookback(
            sample_trial_records_for_lookback,
            as_of_date,
            lookback_days=90,
        )

        # 30d window should have fewer events than 90d
        assert len(events_30d.get("ACME", [])) <= len(events_90d.get("ACME", []))


# =============================================================================
# T4: SCHEMA INTEGRATION
# =============================================================================

class TestActivityProxySchemaIntegration:
    """Tests for activity proxy integration with module_3_schema."""

    def test_activity_proxy_event_type_exists(self):
        """CT_ACTIVITY_PROXY should exist in schema EventType."""
        assert hasattr(SchemaEventType, 'CT_ACTIVITY_PROXY')
        assert SchemaEventType.CT_ACTIVITY_PROXY.value == "CT_ACTIVITY_PROXY"

    def test_activity_proxy_severity_mapping(self):
        """CT_ACTIVITY_PROXY should map to NEUTRAL severity."""
        severity = EVENT_SEVERITY_MAP.get(SchemaEventType.CT_ACTIVITY_PROXY)
        assert severity == EventSeverity.NEUTRAL

    def test_activity_proxy_confidence_mapping(self):
        """CT_ACTIVITY_PROXY should have LOW default confidence."""
        confidence = EVENT_DEFAULT_CONFIDENCE.get(SchemaEventType.CT_ACTIVITY_PROXY)
        assert confidence == ConfidenceLevel.LOW

    def test_activity_proxy_weight_mapping(self):
        """CT_ACTIVITY_PROXY should have low positive weight."""
        weight = EVENT_TYPE_WEIGHT.get(SchemaEventType.CT_ACTIVITY_PROXY)
        assert weight == Decimal("2.0")


# =============================================================================
# T5: INTEGRATION WITH TICKER SUMMARY
# =============================================================================

class TestActivityProxySummaryIntegration:
    """Tests for activity proxy integration with TickerCatalystSummaryV2."""

    def test_summary_has_activity_proxy_fields(self):
        """TickerCatalystSummaryV2 should have activity proxy fields."""
        from module_3_schema import TickerCatalystSummaryV2, CatalystWindowBucket

        summary = TickerCatalystSummaryV2(
            ticker="ACME",
            as_of_date="2026-01-15",
            score_override=Decimal("50"),
            score_blended=Decimal("50"),
            score_mode_used="blended",
            severe_negative_flag=False,
            next_catalyst_date=None,
            catalyst_window_days=None,
            catalyst_window_bucket=CatalystWindowBucket.UNKNOWN,
            catalyst_confidence=ConfidenceLevel.MED,
            events_total=0,
            events_by_severity={},
            events_by_type={},
            weighted_counts_by_severity={},
            top_3_events=[],
            events=[],
        )

        # Check activity proxy fields exist
        assert hasattr(summary, 'activity_proxy_score')
        assert hasattr(summary, 'activity_proxy_count_90d')
        assert hasattr(summary, 'activity_proxy_count_30d')

        # Check default values
        assert summary.activity_proxy_score == Decimal("0")
        assert summary.activity_proxy_count_90d == 0
        assert summary.activity_proxy_count_30d == 0

    def test_summary_serialization_includes_activity_proxy(self):
        """Summary to_dict() should include activity proxy fields."""
        from module_3_schema import TickerCatalystSummaryV2, CatalystWindowBucket

        summary = TickerCatalystSummaryV2(
            ticker="ACME",
            as_of_date="2026-01-15",
            score_override=Decimal("50"),
            score_blended=Decimal("50"),
            score_mode_used="blended",
            severe_negative_flag=False,
            next_catalyst_date=None,
            catalyst_window_days=None,
            catalyst_window_bucket=CatalystWindowBucket.UNKNOWN,
            catalyst_confidence=ConfidenceLevel.MED,
            events_total=0,
            events_by_severity={},
            events_by_type={},
            weighted_counts_by_severity={},
            top_3_events=[],
            events=[],
            activity_proxy_score=Decimal("5.5"),
            activity_proxy_count_90d=3,
            activity_proxy_count_30d=2,
        )

        serialized = summary.to_dict()

        # Check scores section
        assert serialized['scores']['activity_proxy_score'] == "5.5"

        # Check event_summary section
        assert serialized['event_summary']['activity_proxy_count_90d'] == 3
        assert serialized['event_summary']['activity_proxy_count_30d'] == 2


# =============================================================================
# EDGE CASES
# =============================================================================

class TestActivityProxyEdgeCases:
    """Edge case tests for activity proxy detection."""

    def test_no_prior_record(self, event_detector, sample_trial_record, as_of_date):
        """Should not generate activity proxy when no prior record exists."""
        events = event_detector.detect_events(sample_trial_record, None, as_of_date)
        assert len(events) == 0

    def test_identical_records(self, event_detector, sample_trial_record, as_of_date):
        """Should not generate activity proxy when records are identical."""
        events = event_detector.detect_events(
            sample_trial_record,
            sample_trial_record,  # Same record
            as_of_date
        )
        assert len(events) == 0

    def test_missing_last_update_posted(self, event_detector, as_of_date):
        """Should handle missing last_update_posted gracefully."""
        current = CanonicalTrialRecord(
            ticker="ACME",
            nct_id="NCT12345678",
            overall_status=CTGovStatus.RECRUITING,
            last_update_posted=None,  # Missing
            primary_completion_date=as_of_date + timedelta(days=90),
            primary_completion_type=CompletionType.ESTIMATED,
            completion_date=None,
            completion_type=None,
            results_first_posted=None,
        )
        prior = CanonicalTrialRecord(
            ticker="ACME",
            nct_id="NCT12345678",
            overall_status=CTGovStatus.RECRUITING,
            last_update_posted=as_of_date - timedelta(days=30),
            primary_completion_date=as_of_date + timedelta(days=90),
            primary_completion_type=CompletionType.ESTIMATED,
            completion_date=None,
            completion_type=None,
            results_first_posted=None,
        )

        # Should not crash, but may not generate event
        events = event_detector.detect_events(current, prior, as_of_date)
        # Result depends on implementation - just verify no crash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
