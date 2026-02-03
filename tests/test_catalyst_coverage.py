#!/usr/bin/env python3
"""
test_catalyst_coverage.py - Tests for improved catalyst coverage

Tests:
1. Extended calendar windows (180D, 270D) produce events for distant trials
2. Recent results_first_posted generates RESULTS_RECENT events
3. PIT exclusion: no event if last_update_posted > as_of_date
4. Proximity score is nonzero for events at 180-day horizon
5. Monotonicity: closer events produce higher proximity scores
"""

import sys
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ctgov_adapter import CanonicalTrialRecord, CTGovStatus, CompletionType
from state_management import StateSnapshot
from catalyst_diagnostics import detect_calendar_catalysts, EventRuleID
from module_3_scoring import compute_proximity_score
from module_3_schema import (
    CatalystEventV2,
    EventType,
    EventSeverity,
    ConfidenceLevel,
)


# ============================================================================
# HELPERS
# ============================================================================

def _make_record(
    ticker: str = "ACME",
    nct_id: str = "NCT00000001",
    status: CTGovStatus = CTGovStatus.RECRUITING,
    last_update_posted: date = date(2026, 1, 10),
    primary_completion_date: date = None,
    primary_completion_type: CompletionType = CompletionType.ESTIMATED,
    completion_date: date = None,
    completion_type: CompletionType = None,
    results_first_posted: date = None,
) -> CanonicalTrialRecord:
    return CanonicalTrialRecord(
        ticker=ticker,
        nct_id=nct_id,
        overall_status=status,
        last_update_posted=last_update_posted,
        primary_completion_date=primary_completion_date,
        primary_completion_type=primary_completion_type,
        completion_date=completion_date,
        completion_type=completion_type,
        results_first_posted=results_first_posted,
    )


def _make_snapshot(records, as_of_date):
    return StateSnapshot(snapshot_date=as_of_date, records=records)


def _make_event_v2(
    ticker: str = "ACME",
    nct_id: str = "NCT00000001",
    event_type: EventType = EventType.CT_PRIMARY_COMPLETION,
    event_date: str = "2026-07-15",
    confidence: ConfidenceLevel = ConfidenceLevel.MED,
) -> CatalystEventV2:
    return CatalystEventV2(
        ticker=ticker,
        nct_id=nct_id,
        event_type=event_type,
        event_severity=EventSeverity.POSITIVE,
        event_date=event_date,
        field_changed="calendar_upcoming_pcd",
        prior_value=None,
        new_value=f"ahead",
        source="CTGOV_CALENDAR",
        confidence=confidence,
        disclosed_at=event_date,
    )


# ============================================================================
# TEST: Extended windows produce events for 180D and 270D trials
# ============================================================================

class TestExtendedCalendarWindows:

    def test_180d_pcd_generates_event(self):
        """Trial with PCD 150 days out should generate a 180D event."""
        as_of = date(2026, 1, 15)
        pcd = as_of + timedelta(days=150)
        record = _make_record(primary_completion_date=pcd)
        snapshot = _make_snapshot([record], as_of)

        catalysts = detect_calendar_catalysts(snapshot, as_of)

        assert len(catalysts) >= 1
        pcd_cats = [c for c in catalysts if c.event_type == 'UPCOMING_PCD']
        assert len(pcd_cats) == 1
        assert pcd_cats[0].window == '180D'
        assert pcd_cats[0].rule_id == EventRuleID.M3_CAL_PCD_180D

    def test_270d_pcd_generates_event(self):
        """Trial with PCD 250 days out should generate a 270D event."""
        as_of = date(2026, 1, 15)
        pcd = as_of + timedelta(days=250)
        record = _make_record(primary_completion_date=pcd)
        snapshot = _make_snapshot([record], as_of)

        catalysts = detect_calendar_catalysts(snapshot, as_of)

        pcd_cats = [c for c in catalysts if c.event_type == 'UPCOMING_PCD']
        assert len(pcd_cats) == 1
        assert pcd_cats[0].window == '270D'
        assert pcd_cats[0].rule_id == EventRuleID.M3_CAL_PCD_270D

    def test_180d_scd_generates_event(self):
        """Trial with SCD 120 days out (no PCD) should generate a 180D SCD event."""
        as_of = date(2026, 1, 15)
        scd = as_of + timedelta(days=120)
        record = _make_record(
            completion_date=scd,
            completion_type=CompletionType.ESTIMATED,
        )
        snapshot = _make_snapshot([record], as_of)

        catalysts = detect_calendar_catalysts(snapshot, as_of)

        scd_cats = [c for c in catalysts if c.event_type == 'UPCOMING_SCD']
        assert len(scd_cats) == 1
        assert scd_cats[0].window == '180D'
        assert scd_cats[0].rule_id == EventRuleID.M3_CAL_SCD_180D

    def test_270d_scd_generates_event(self):
        """Trial with SCD 200 days out should generate a 270D SCD event."""
        as_of = date(2026, 1, 15)
        scd = as_of + timedelta(days=200)
        record = _make_record(
            completion_date=scd,
            completion_type=CompletionType.ESTIMATED,
        )
        snapshot = _make_snapshot([record], as_of)

        catalysts = detect_calendar_catalysts(snapshot, as_of)

        scd_cats = [c for c in catalysts if c.event_type == 'UPCOMING_SCD']
        assert len(scd_cats) == 1
        assert scd_cats[0].window == '270D'

    def test_no_event_beyond_270d(self):
        """Trial with PCD 300 days out should NOT generate an event."""
        as_of = date(2026, 1, 15)
        pcd = as_of + timedelta(days=300)
        record = _make_record(primary_completion_date=pcd)
        snapshot = _make_snapshot([record], as_of)

        catalysts = detect_calendar_catalysts(snapshot, as_of)

        pcd_cats = [c for c in catalysts if c.event_type == 'UPCOMING_PCD']
        assert len(pcd_cats) == 0

    def test_old_windows_still_work(self):
        """30D, 60D, 90D windows still produce correct events."""
        as_of = date(2026, 1, 15)

        results = {}
        for days, expected_window in [(20, '30D'), (45, '60D'), (75, '90D')]:
            pcd = as_of + timedelta(days=days)
            record = _make_record(
                nct_id=f"NCT{days:08d}",
                primary_completion_date=pcd,
            )
            snapshot = _make_snapshot([record], as_of)
            cats = detect_calendar_catalysts(snapshot, as_of)
            pcd_cats = [c for c in cats if c.event_type == 'UPCOMING_PCD']
            assert len(pcd_cats) == 1, f"Expected 1 PCD event at {days}d"
            results[days] = pcd_cats[0].window

        assert results[20] == '30D'
        assert results[45] == '60D'
        assert results[75] == '90D'

    def test_confidence_decreases_with_distance(self):
        """Confidence should decrease: 30D > 60D > 90D > 180D > 270D."""
        as_of = date(2026, 1, 15)
        confidences = {}

        for days, window_label in [(15, '30D'), (45, '60D'), (75, '90D'),
                                    (120, '180D'), (200, '270D')]:
            pcd = as_of + timedelta(days=days)
            record = _make_record(
                nct_id=f"NCT{days:08d}",
                primary_completion_date=pcd,
            )
            snapshot = _make_snapshot([record], as_of)
            cats = detect_calendar_catalysts(snapshot, as_of)
            pcd_cats = [c for c in cats if c.event_type == 'UPCOMING_PCD']
            assert len(pcd_cats) == 1
            confidences[window_label] = pcd_cats[0].confidence

        assert confidences['30D'] > confidences['60D']
        assert confidences['60D'] > confidences['90D']
        assert confidences['90D'] > confidences['180D']
        assert confidences['180D'] > confidences['270D']


# ============================================================================
# TEST: Recent results detection
# ============================================================================

class TestRecentResultsDetection:

    def test_recent_results_generates_event(self):
        """results_first_posted within 60 days should generate RESULTS_RECENT."""
        as_of = date(2026, 1, 15)
        results_posted = as_of - timedelta(days=30)
        record = _make_record(
            results_first_posted=results_posted,
            last_update_posted=as_of - timedelta(days=5),
            status=CTGovStatus.COMPLETED,
        )
        snapshot = _make_snapshot([record], as_of)

        catalysts = detect_calendar_catalysts(snapshot, as_of)

        results_cats = [c for c in catalysts if c.event_type == 'RESULTS_RECENT']
        assert len(results_cats) == 1
        assert results_cats[0].rule_id == EventRuleID.M3_CAL_RESULTS_RECENT
        assert results_cats[0].confidence == 0.90
        assert results_cats[0].target_date == results_posted
        assert results_cats[0].days_until == -30  # Negative: past event

    def test_old_results_no_event(self):
        """results_first_posted > 60 days ago should NOT generate an event."""
        as_of = date(2026, 1, 15)
        results_posted = as_of - timedelta(days=90)
        record = _make_record(
            results_first_posted=results_posted,
            last_update_posted=as_of - timedelta(days=5),
            status=CTGovStatus.COMPLETED,
        )
        snapshot = _make_snapshot([record], as_of)

        catalysts = detect_calendar_catalysts(snapshot, as_of)

        results_cats = [c for c in catalysts if c.event_type == 'RESULTS_RECENT']
        assert len(results_cats) == 0

    def test_results_today_generates_event(self):
        """results_first_posted on as_of_date (0 days ago) should generate event."""
        as_of = date(2026, 1, 15)
        record = _make_record(
            results_first_posted=as_of,
            last_update_posted=as_of,
            status=CTGovStatus.COMPLETED,
        )
        snapshot = _make_snapshot([record], as_of)

        catalysts = detect_calendar_catalysts(snapshot, as_of)

        results_cats = [c for c in catalysts if c.event_type == 'RESULTS_RECENT']
        assert len(results_cats) == 1

    def test_terminal_negative_skipped(self):
        """Terminal negative trials should not generate any events."""
        as_of = date(2026, 1, 15)
        record = _make_record(
            results_first_posted=as_of - timedelta(days=10),
            last_update_posted=as_of - timedelta(days=5),
            status=CTGovStatus.TERMINATED,
            primary_completion_date=as_of + timedelta(days=100),
        )
        snapshot = _make_snapshot([record], as_of)

        catalysts = detect_calendar_catalysts(snapshot, as_of)
        assert len(catalysts) == 0


# ============================================================================
# TEST: PIT exclusion
# ============================================================================

class TestPITExclusion:

    def test_results_from_future_excluded(self):
        """If last_update_posted > as_of_date, results_recent event must not appear."""
        as_of = date(2026, 1, 15)
        record = _make_record(
            results_first_posted=as_of - timedelta(days=10),
            last_update_posted=as_of + timedelta(days=1),  # Future!
            status=CTGovStatus.COMPLETED,
        )
        snapshot = _make_snapshot([record], as_of)

        catalysts = detect_calendar_catalysts(snapshot, as_of)

        results_cats = [c for c in catalysts if c.event_type == 'RESULTS_RECENT']
        assert len(results_cats) == 0


# ============================================================================
# TEST: Proximity score nonzero at 180-day horizon
# ============================================================================

class TestProximityScoreExtended:

    def test_proximity_nonzero_at_180d(self):
        """Events at 180d horizon should produce nonzero proximity score."""
        as_of = date(2026, 1, 15)
        event_date = (as_of + timedelta(days=180)).isoformat()

        events = [_make_event_v2(event_date=event_date)]
        score, n_upcoming = compute_proximity_score(events, as_of)

        assert n_upcoming == 1
        assert score > Decimal("0")

    def test_proximity_nonzero_at_270d(self):
        """Events at 270d should still produce nonzero proximity score."""
        as_of = date(2026, 1, 15)
        event_date = (as_of + timedelta(days=269)).isoformat()

        events = [_make_event_v2(event_date=event_date)]
        score, n_upcoming = compute_proximity_score(events, as_of)

        assert n_upcoming == 1
        assert score > Decimal("0")

    def test_proximity_zero_beyond_horizon(self):
        """Events beyond 270d should produce zero proximity score."""
        as_of = date(2026, 1, 15)
        event_date = (as_of + timedelta(days=271)).isoformat()

        events = [_make_event_v2(event_date=event_date)]
        score, n_upcoming = compute_proximity_score(events, as_of)

        assert n_upcoming == 0
        assert score == Decimal("0")


# ============================================================================
# TEST: Monotonicity — closer events produce higher proximity scores
# ============================================================================

class TestProximityMonotonicity:

    def test_closer_events_score_higher(self):
        """30d event should score higher than 90d, which scores higher than 180d."""
        as_of = date(2026, 1, 15)

        scores = {}
        for days in [30, 60, 90, 120, 180, 250]:
            event_date = (as_of + timedelta(days=days)).isoformat()
            events = [_make_event_v2(event_date=event_date)]
            score, _ = compute_proximity_score(events, as_of)
            scores[days] = score

        # Strict monotonic decrease
        for d1, d2 in zip([30, 60, 90, 120, 180], [60, 90, 120, 180, 250]):
            assert scores[d1] > scores[d2], (
                f"Expected score at {d1}d ({scores[d1]}) > score at {d2}d ({scores[d2]})"
            )

    def test_multiple_events_score_higher_than_single(self):
        """Two upcoming events should score higher than one at same distance."""
        as_of = date(2026, 1, 15)
        event_date_1 = (as_of + timedelta(days=60)).isoformat()
        event_date_2 = (as_of + timedelta(days=90)).isoformat()

        single = [_make_event_v2(event_date=event_date_1)]
        double = [
            _make_event_v2(event_date=event_date_1, nct_id="NCT00000001"),
            _make_event_v2(event_date=event_date_2, nct_id="NCT00000002"),
        ]

        score_single, _ = compute_proximity_score(single, as_of)
        score_double, _ = compute_proximity_score(double, as_of)

        assert score_double > score_single
