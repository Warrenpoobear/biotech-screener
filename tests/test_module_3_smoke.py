#!/usr/bin/env python3
"""
test_module_3_smoke.py - Smoke Tests for Module 3 Catalyst Detection

Verifies that Module 3 correctly detects:
1. Status changes (upgrade, downgrade, severe negative)
2. Timeline shifts (date push, date pull)
3. Results posted events
4. Calendar-based catalysts (upcoming PCD)
5. Delta diagnostics with sample diffs
6. Staleness gating

These tests use synthetic fixtures with known events to prevent regressions.
"""

import pytest
from datetime import date, timedelta
from decimal import Decimal
import tempfile
import json
from pathlib import Path

from ctgov_adapter import (
    CTGovStatus,
    CompletionType,
    CanonicalTrialRecord,
    CTGovAdapter,
    AdapterConfig,
)
from event_detector import (
    EventDetector,
    EventDetectorConfig,
    EventType,
    CatalystEvent,
    SimpleMarketCalendar,
    classify_status_change,
    classify_timeline_change,
)
from state_management import StateStore, StateSnapshot
from catalyst_diagnostics import (
    compute_delta_diagnostics,
    check_trial_records_staleness,
    detect_calendar_catalysts,
    DeltaDiagnostics,
    StalenessResult,
    CalendarCatalyst,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def as_of_date():
    """Standard test date"""
    return date(2026, 1, 15)


@pytest.fixture
def prior_date():
    """Prior snapshot date"""
    return date(2026, 1, 10)


@pytest.fixture
def market_calendar():
    """Simple market calendar for tests"""
    return SimpleMarketCalendar()


@pytest.fixture
def event_detector():
    """Event detector with default config"""
    return EventDetector(EventDetectorConfig())


@pytest.fixture
def sample_prior_record():
    """Sample prior trial record"""
    return CanonicalTrialRecord(
        ticker="TEST",
        nct_id="NCT00000001",
        overall_status=CTGovStatus.RECRUITING,
        last_update_posted=date(2026, 1, 5),
        primary_completion_date=date(2026, 6, 15),
        primary_completion_type=CompletionType.ESTIMATED,
        completion_date=date(2026, 12, 15),
        completion_type=CompletionType.ESTIMATED,
        results_first_posted=None,
    )


# ============================================================================
# STATUS CHANGE TESTS
# ============================================================================

class TestStatusChangeDetection:
    """Test status change event detection"""

    def test_severe_negative_terminated(self, event_detector, sample_prior_record, as_of_date):
        """Terminated status should trigger severe negative event"""
        current = CanonicalTrialRecord(
            ticker="TEST",
            nct_id="NCT00000001",
            overall_status=CTGovStatus.TERMINATED,
            last_update_posted=date(2026, 1, 14),
            primary_completion_date=sample_prior_record.primary_completion_date,
            primary_completion_type=sample_prior_record.primary_completion_type,
            completion_date=sample_prior_record.completion_date,
            completion_type=sample_prior_record.completion_type,
            results_first_posted=None,
        )

        events = event_detector.detect_events(current, sample_prior_record, as_of_date)

        assert len(events) == 1
        assert events[0].event_type == EventType.CT_STATUS_SEVERE_NEG
        assert events[0].direction == 'NEG'
        assert events[0].impact == 3
        assert 'terminal negative' in events[0].confidence_reason.lower()
        assert events[0].event_rule_id == "M3_DIFF_CT_STATUS_SEVERE_NEG"

    def test_status_upgrade(self, event_detector, as_of_date):
        """Status upgrade should trigger positive event"""
        prior = CanonicalTrialRecord(
            ticker="TEST",
            nct_id="NCT00000002",
            overall_status=CTGovStatus.NOT_YET_RECRUITING,
            last_update_posted=date(2026, 1, 5),
            primary_completion_date=date(2026, 6, 15),
            primary_completion_type=CompletionType.ESTIMATED,
            completion_date=None,
            completion_type=None,
            results_first_posted=None,
        )
        current = CanonicalTrialRecord(
            ticker="TEST",
            nct_id="NCT00000002",
            overall_status=CTGovStatus.RECRUITING,
            last_update_posted=date(2026, 1, 14),
            primary_completion_date=date(2026, 6, 15),
            primary_completion_type=CompletionType.ESTIMATED,
            completion_date=None,
            completion_type=None,
            results_first_posted=None,
        )

        events = event_detector.detect_events(current, prior, as_of_date)

        assert len(events) == 1
        assert events[0].event_type == EventType.CT_STATUS_UPGRADE
        assert events[0].direction == 'POS'
        assert 'upgrade' in events[0].confidence_reason.lower()

    def test_status_downgrade(self, event_detector, as_of_date):
        """Status downgrade should trigger negative event"""
        prior = CanonicalTrialRecord(
            ticker="TEST",
            nct_id="NCT00000003",
            overall_status=CTGovStatus.RECRUITING,
            last_update_posted=date(2026, 1, 5),
            primary_completion_date=date(2026, 6, 15),
            primary_completion_type=CompletionType.ESTIMATED,
            completion_date=None,
            completion_type=None,
            results_first_posted=None,
        )
        current = CanonicalTrialRecord(
            ticker="TEST",
            nct_id="NCT00000003",
            overall_status=CTGovStatus.SUSPENDED,
            last_update_posted=date(2026, 1, 14),
            primary_completion_date=date(2026, 6, 15),
            primary_completion_type=CompletionType.ESTIMATED,
            completion_date=None,
            completion_type=None,
            results_first_posted=None,
        )

        events = event_detector.detect_events(current, prior, as_of_date)

        # SUSPENDED triggers CT_STATUS_SEVERE_NEG, not just DOWNGRADE
        assert len(events) == 1
        assert events[0].event_type == EventType.CT_STATUS_SEVERE_NEG
        assert events[0].direction == 'NEG'


# ============================================================================
# TIMELINE SHIFT TESTS
# ============================================================================

class TestTimelineShiftDetection:
    """Test timeline shift event detection"""

    def test_date_pushout_30_days(self, event_detector, sample_prior_record, as_of_date):
        """30+ day pushout should trigger negative timeline event"""
        current = CanonicalTrialRecord(
            ticker="TEST",
            nct_id="NCT00000001",
            overall_status=sample_prior_record.overall_status,
            last_update_posted=date(2026, 1, 14),
            primary_completion_date=date(2026, 7, 20),  # +35 days
            primary_completion_type=CompletionType.ESTIMATED,
            completion_date=sample_prior_record.completion_date,
            completion_type=sample_prior_record.completion_type,
            results_first_posted=None,
        )

        events = event_detector.detect_events(current, sample_prior_record, as_of_date)

        assert len(events) == 1
        assert events[0].event_type == EventType.CT_TIMELINE_PUSHOUT
        assert events[0].direction == 'NEG'
        assert 'pushed out' in events[0].confidence_reason.lower()
        assert '35 days' in events[0].confidence_reason

    def test_date_pullin_30_days(self, event_detector, sample_prior_record, as_of_date):
        """30+ day pull-in should trigger positive timeline event"""
        current = CanonicalTrialRecord(
            ticker="TEST",
            nct_id="NCT00000001",
            overall_status=sample_prior_record.overall_status,
            last_update_posted=date(2026, 1, 14),
            primary_completion_date=date(2026, 5, 10),  # -36 days
            primary_completion_type=CompletionType.ESTIMATED,
            completion_date=sample_prior_record.completion_date,
            completion_type=sample_prior_record.completion_type,
            results_first_posted=None,
        )

        events = event_detector.detect_events(current, sample_prior_record, as_of_date)

        assert len(events) == 1
        assert events[0].event_type == EventType.CT_TIMELINE_PULLIN
        assert events[0].direction == 'POS'
        assert 'pulled in' in events[0].confidence_reason.lower()

    def test_small_date_change_ignored(self, event_detector, sample_prior_record, as_of_date):
        """Small date changes (<14 days) should be ignored as noise"""
        current = CanonicalTrialRecord(
            ticker="TEST",
            nct_id="NCT00000001",
            overall_status=sample_prior_record.overall_status,
            last_update_posted=date(2026, 1, 14),
            primary_completion_date=date(2026, 6, 20),  # +5 days
            primary_completion_type=CompletionType.ESTIMATED,
            completion_date=sample_prior_record.completion_date,
            completion_type=sample_prior_record.completion_type,
            results_first_posted=None,
        )

        events = event_detector.detect_events(current, sample_prior_record, as_of_date)

        # No events - small change is filtered as noise
        assert len(events) == 0


# ============================================================================
# RESULTS POSTED TESTS
# ============================================================================

class TestResultsPostedDetection:
    """Test results posted event detection"""

    def test_results_first_posted(self, event_detector, sample_prior_record, as_of_date):
        """Results being posted should trigger event"""
        current = CanonicalTrialRecord(
            ticker="TEST",
            nct_id="NCT00000001",
            overall_status=CTGovStatus.COMPLETED,
            last_update_posted=date(2026, 1, 14),
            primary_completion_date=sample_prior_record.primary_completion_date,
            primary_completion_type=CompletionType.ACTUAL,
            completion_date=sample_prior_record.completion_date,
            completion_type=CompletionType.ACTUAL,
            results_first_posted=date(2026, 1, 10),  # Results posted
        )

        events = event_detector.detect_events(current, sample_prior_record, as_of_date)

        # Should have results_posted and possibly status/type changes
        results_events = [e for e in events if e.event_type == EventType.CT_RESULTS_POSTED]
        assert len(results_events) == 1
        assert 'first posted' in results_events[0].confidence_reason.lower()


# ============================================================================
# STATUS NORMALIZATION TESTS
# ============================================================================

class TestStatusNormalization:
    """Test that uncommon statuses are properly normalized"""

    def test_approved_for_marketing(self):
        """APPROVED_FOR_MARKETING should map to specific enum value"""
        status = CTGovStatus.from_string("APPROVED_FOR_MARKETING")
        assert status == CTGovStatus.APPROVED_FOR_MARKETING
        assert status.is_terminal_positive

    def test_available(self):
        """AVAILABLE should map to specific enum value"""
        status = CTGovStatus.from_string("AVAILABLE")
        assert status == CTGovStatus.AVAILABLE
        assert status.is_active

    def test_no_longer_available(self):
        """NO_LONGER_AVAILABLE should map to specific enum value"""
        status = CTGovStatus.from_string("NO_LONGER_AVAILABLE")
        assert status == CTGovStatus.NO_LONGER_AVAILABLE

    def test_withheld(self):
        """WITHHELD should map to specific enum value"""
        status = CTGovStatus.from_string("WITHHELD")
        assert status == CTGovStatus.WITHHELD

    def test_status_ordering(self):
        """Verify status ordering for lifecycle progression"""
        assert CTGovStatus.WITHDRAWN.value < CTGovStatus.RECRUITING.value
        assert CTGovStatus.RECRUITING.value < CTGovStatus.COMPLETED.value
        # APPROVED_FOR_MARKETING is placed before COMPLETED since COMPLETED is final trial state
        # Both are terminal positive, but COMPLETED represents the highest trial completion state
        assert CTGovStatus.APPROVED_FOR_MARKETING.value < CTGovStatus.COMPLETED.value
        assert CTGovStatus.APPROVED_FOR_MARKETING.is_terminal_positive
        assert CTGovStatus.COMPLETED.is_terminal_positive


# ============================================================================
# DELTA DIAGNOSTICS TESTS
# ============================================================================

class TestDeltaDiagnostics:
    """Test delta diagnostics computation"""

    def test_delta_with_changes(self, prior_date, as_of_date):
        """Delta diagnostics should capture field changes"""
        prior_records = [
            CanonicalTrialRecord(
                ticker="TEST",
                nct_id="NCT00000001",
                overall_status=CTGovStatus.RECRUITING,
                last_update_posted=prior_date,
                primary_completion_date=date(2026, 6, 15),
                primary_completion_type=CompletionType.ESTIMATED,
                completion_date=None,
                completion_type=None,
                results_first_posted=None,
            )
        ]
        current_records = [
            CanonicalTrialRecord(
                ticker="TEST",
                nct_id="NCT00000001",
                overall_status=CTGovStatus.COMPLETED,  # Changed
                last_update_posted=as_of_date,
                primary_completion_date=date(2026, 6, 20),  # Changed
                primary_completion_type=CompletionType.ACTUAL,  # Changed
                completion_date=None,
                completion_type=None,
                results_first_posted=None,
            )
        ]

        prior_snapshot = StateSnapshot(snapshot_date=prior_date, records=prior_records)
        current_snapshot = StateSnapshot(snapshot_date=as_of_date, records=current_records)

        diag = compute_delta_diagnostics(current_snapshot, prior_snapshot)

        assert diag.records_changed_count == 1
        assert diag.no_changes_detected is False
        assert 'overall_status' in diag.fields_changed_histogram
        assert 'primary_completion_date' in diag.fields_changed_histogram
        assert len(diag.sample_diffs) > 0

    def test_delta_no_changes(self, prior_date, as_of_date):
        """Delta diagnostics should flag when no changes detected"""
        records = [
            CanonicalTrialRecord(
                ticker="TEST",
                nct_id="NCT00000001",
                overall_status=CTGovStatus.RECRUITING,
                last_update_posted=prior_date,
                primary_completion_date=date(2026, 6, 15),
                primary_completion_type=CompletionType.ESTIMATED,
                completion_date=None,
                completion_type=None,
                results_first_posted=None,
            )
        ]

        prior_snapshot = StateSnapshot(snapshot_date=prior_date, records=records)
        current_snapshot = StateSnapshot(snapshot_date=as_of_date, records=records)

        diag = compute_delta_diagnostics(current_snapshot, prior_snapshot)

        assert diag.records_changed_count == 0
        assert diag.no_changes_detected is True

    def test_delta_no_prior_snapshot(self, as_of_date):
        """Delta diagnostics should handle missing prior snapshot"""
        current_records = [
            CanonicalTrialRecord(
                ticker="TEST",
                nct_id="NCT00000001",
                overall_status=CTGovStatus.RECRUITING,
                last_update_posted=as_of_date,
                primary_completion_date=date(2026, 6, 15),
                primary_completion_type=CompletionType.ESTIMATED,
                completion_date=None,
                completion_type=None,
                results_first_posted=None,
            )
        ]
        current_snapshot = StateSnapshot(snapshot_date=as_of_date, records=current_records)

        diag = compute_delta_diagnostics(current_snapshot, None)

        assert diag.prior_snapshot_missing is True


# ============================================================================
# STALENESS GATING TESTS
# ============================================================================

class TestStalenessGating:
    """Test staleness gating for trial records"""

    def test_fresh_data(self, as_of_date):
        """Fresh data should have HIGH confidence"""
        records = [
            {"ticker": "TEST", "nct_id": "NCT00000001", "last_update_posted": as_of_date.isoformat()}
        ]

        result = check_trial_records_staleness(records, as_of_date)

        assert result.is_stale is False
        assert result.confidence_level == 'HIGH'
        assert result.age_days == 0

    def test_stale_data(self, as_of_date):
        """Stale data (>5 days) should have LOW confidence"""
        old_date = as_of_date - timedelta(days=10)
        records = [
            {"ticker": "TEST", "nct_id": "NCT00000001", "last_update_posted": old_date.isoformat()}
        ]

        result = check_trial_records_staleness(records, as_of_date)

        assert result.is_stale is True
        assert result.confidence_level == 'LOW'
        assert result.age_days == 10

    def test_future_data(self, as_of_date):
        """Future data should be flagged as DEGRADED"""
        future_date = as_of_date + timedelta(days=5)
        records = [
            {"ticker": "TEST", "nct_id": "NCT00000001", "last_update_posted": future_date.isoformat()}
        ]

        result = check_trial_records_staleness(records, as_of_date)

        assert result.is_stale is True
        assert result.confidence_level == 'DEGRADED'
        assert 'lookahead' in result.recommendation.lower()


# ============================================================================
# CALENDAR CATALYST TESTS
# ============================================================================

class TestCalendarCatalysts:
    """Test calendar-based catalyst detection"""

    def test_upcoming_pcd_30d(self, as_of_date):
        """PCD within 30 days should be detected"""
        records = [
            CanonicalTrialRecord(
                ticker="TEST",
                nct_id="NCT00000001",
                overall_status=CTGovStatus.ACTIVE_NOT_RECRUITING,
                last_update_posted=as_of_date,
                primary_completion_date=as_of_date + timedelta(days=25),
                primary_completion_type=CompletionType.ESTIMATED,
                completion_date=None,
                completion_type=None,
                results_first_posted=None,
            )
        ]
        snapshot = StateSnapshot(snapshot_date=as_of_date, records=records)

        catalysts = detect_calendar_catalysts(snapshot, as_of_date)

        assert len(catalysts) == 1
        assert catalysts[0].event_type == 'UPCOMING_PCD'
        assert catalysts[0].window == '30D'
        assert catalysts[0].days_until == 25

    def test_upcoming_pcd_60d(self, as_of_date):
        """PCD within 60 days should be detected"""
        records = [
            CanonicalTrialRecord(
                ticker="TEST",
                nct_id="NCT00000001",
                overall_status=CTGovStatus.RECRUITING,
                last_update_posted=as_of_date,
                primary_completion_date=as_of_date + timedelta(days=45),
                primary_completion_type=CompletionType.ESTIMATED,
                completion_date=None,
                completion_type=None,
                results_first_posted=None,
            )
        ]
        snapshot = StateSnapshot(snapshot_date=as_of_date, records=records)

        catalysts = detect_calendar_catalysts(snapshot, as_of_date)

        assert len(catalysts) == 1
        assert catalysts[0].window == '60D'

    def test_terminal_trial_excluded(self, as_of_date):
        """Terminal negative trials should not have calendar catalysts"""
        records = [
            CanonicalTrialRecord(
                ticker="TEST",
                nct_id="NCT00000001",
                overall_status=CTGovStatus.TERMINATED,
                last_update_posted=as_of_date,
                primary_completion_date=as_of_date + timedelta(days=25),
                primary_completion_type=CompletionType.ESTIMATED,
                completion_date=None,
                completion_type=None,
                results_first_posted=None,
            )
        ]
        snapshot = StateSnapshot(snapshot_date=as_of_date, records=records)

        catalysts = detect_calendar_catalysts(snapshot, as_of_date)

        assert len(catalysts) == 0


# ============================================================================
# INTEGRATION SMOKE TEST
# ============================================================================

class TestModule3IntegrationSmoke:
    """End-to-end smoke test with synthetic data"""

    def test_full_event_detection_pipeline(self, as_of_date, prior_date):
        """Full pipeline should detect all expected event types"""
        # Create synthetic snapshots with known events
        prior_records = [
            # Will have status change to COMPLETED
            CanonicalTrialRecord(
                ticker="AAAA",
                nct_id="NCT00000001",
                overall_status=CTGovStatus.ACTIVE_NOT_RECRUITING,
                last_update_posted=prior_date,
                primary_completion_date=date(2026, 1, 10),
                primary_completion_type=CompletionType.ANTICIPATED,
                completion_date=None,
                completion_type=None,
                results_first_posted=None,
            ),
            # Will have date pushout
            CanonicalTrialRecord(
                ticker="BBBB",
                nct_id="NCT00000002",
                overall_status=CTGovStatus.RECRUITING,
                last_update_posted=prior_date,
                primary_completion_date=date(2026, 6, 15),
                primary_completion_type=CompletionType.ESTIMATED,
                completion_date=None,
                completion_type=None,
                results_first_posted=None,
            ),
            # Will have results posted
            CanonicalTrialRecord(
                ticker="CCCC",
                nct_id="NCT00000003",
                overall_status=CTGovStatus.COMPLETED,
                last_update_posted=prior_date,
                primary_completion_date=date(2025, 12, 1),
                primary_completion_type=CompletionType.ACTUAL,
                completion_date=None,
                completion_type=None,
                results_first_posted=None,
            ),
        ]

        current_records = [
            # Status: ACTIVE_NOT_RECRUITING → COMPLETED + date confirmed ACTUAL
            CanonicalTrialRecord(
                ticker="AAAA",
                nct_id="NCT00000001",
                overall_status=CTGovStatus.COMPLETED,
                last_update_posted=as_of_date,
                primary_completion_date=date(2026, 1, 10),
                primary_completion_type=CompletionType.ACTUAL,
                completion_date=None,
                completion_type=None,
                results_first_posted=None,
            ),
            # Date push: 2026-06-15 → 2026-08-15 (+61 days)
            CanonicalTrialRecord(
                ticker="BBBB",
                nct_id="NCT00000002",
                overall_status=CTGovStatus.RECRUITING,
                last_update_posted=as_of_date,
                primary_completion_date=date(2026, 8, 15),
                primary_completion_type=CompletionType.ESTIMATED,
                completion_date=None,
                completion_type=None,
                results_first_posted=None,
            ),
            # Results posted
            CanonicalTrialRecord(
                ticker="CCCC",
                nct_id="NCT00000003",
                overall_status=CTGovStatus.COMPLETED,
                last_update_posted=as_of_date,
                primary_completion_date=date(2025, 12, 1),
                primary_completion_type=CompletionType.ACTUAL,
                completion_date=None,
                completion_type=None,
                results_first_posted=date(2026, 1, 12),
            ),
        ]

        # Create snapshots
        prior_snapshot = StateSnapshot(snapshot_date=prior_date, records=prior_records)
        current_snapshot = StateSnapshot(snapshot_date=as_of_date, records=current_records)

        # Run event detection
        event_detector = EventDetector(EventDetectorConfig())
        all_events = []

        for current_record in current_records:
            prior_record = prior_snapshot.get_record(current_record.ticker, current_record.nct_id)
            events = event_detector.detect_events(current_record, prior_record, as_of_date)
            all_events.extend(events)

        # Verify expected events
        event_types = [e.event_type for e in all_events]

        # Should have status upgrade (ACTIVE_NOT_RECRUITING → COMPLETED)
        assert EventType.CT_STATUS_UPGRADE in event_types

        # Should have timeline pushout
        assert EventType.CT_TIMELINE_PUSHOUT in event_types

        # Should have results posted
        assert EventType.CT_RESULTS_POSTED in event_types

        # Should have date confirmed (ANTICIPATED → ACTUAL)
        assert EventType.CT_DATE_CONFIRMED_ACTUAL in event_types

        # Verify delta diagnostics
        diag = compute_delta_diagnostics(current_snapshot, prior_snapshot)
        assert diag.records_changed_count == 3
        assert diag.no_changes_detected is False

        # Verify all events have explainability fields
        for event in all_events:
            assert event.event_rule_id != ""
            assert event.confidence_reason != ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
