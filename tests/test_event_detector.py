#!/usr/bin/env python3
"""
Unit tests for event_detector.py

Tests CT.gov catalyst event detection:
- Event types and classification
- CatalystEvent dataclass
- Status change classification
- Timeline change classification
- Date confirmation classification
- Results posted detection
- EventDetector class
- Activity proxy detection
- Event scoring
"""

import pytest
from datetime import date, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from event_detector import (
    EventType,
    CatalystEvent,
    EventDetectorConfig,
    EventDetector,
    SimpleMarketCalendar,
    classify_status_change,
    classify_timeline_change,
    classify_date_confirmation,
    classify_results_posted,
    compute_event_score,
    detect_activity_proxy_from_lookback,
    compute_activity_proxy_score,
)
from ctgov_adapter import CTGovStatus, CompletionType, CanonicalTrialRecord


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def as_of_date():
    """Standard as_of_date for tests."""
    return date(2026, 1, 15)


@pytest.fixture
def calendar():
    """Simple market calendar for tests."""
    return SimpleMarketCalendar()


@pytest.fixture
def config():
    """Default event detector config."""
    return EventDetectorConfig()


@pytest.fixture
def detector(config):
    """Event detector instance."""
    return EventDetector(config)


@pytest.fixture
def sample_trial_record():
    """Sample canonical trial record."""
    return CanonicalTrialRecord(
        ticker="ACME",
        nct_id="NCT12345678",
        overall_status=CTGovStatus.ACTIVE_NOT_RECRUITING,
        last_update_posted=date(2026, 1, 10),
        primary_completion_date=date(2026, 6, 15),
        primary_completion_type=CompletionType.ESTIMATED,
        completion_date=date(2026, 9, 30),
        completion_type=CompletionType.ESTIMATED,
        results_first_posted=None,
    )


# ============================================================================
# EVENT TYPE TESTS
# ============================================================================

class TestEventType:
    """Tests for EventType enum."""

    def test_all_event_types_defined(self):
        """All expected event types should be defined."""
        expected = [
            "CT_STATUS_SEVERE_NEG",
            "CT_STATUS_DOWNGRADE",
            "CT_STATUS_UPGRADE",
            "CT_TIMELINE_PUSHOUT",
            "CT_TIMELINE_PULLIN",
            "CT_DATE_CONFIRMED_ACTUAL",
            "CT_RESULTS_POSTED",
            "CT_ACTIVITY_PROXY",
        ]
        for et in expected:
            assert hasattr(EventType, et)

    def test_event_type_values(self):
        """Event type values should match names."""
        assert EventType.CT_STATUS_SEVERE_NEG.value == "CT_STATUS_SEVERE_NEG"
        assert EventType.CT_TIMELINE_PUSHOUT.value == "CT_TIMELINE_PUSHOUT"


# ============================================================================
# CATALYST EVENT TESTS
# ============================================================================

class TestCatalystEvent:
    """Tests for CatalystEvent dataclass."""

    def test_requires_disclosed_at(self):
        """CatalystEvent should require disclosed_at to be set."""
        with pytest.raises(ValueError, match="disclosed_at must be explicitly set"):
            CatalystEvent(
                nct_id="NCT12345678",
                event_type=EventType.CT_STATUS_UPGRADE,
                # disclosed_at not provided
            )

    def test_basic_creation(self, as_of_date):
        """Should create event with required fields."""
        event = CatalystEvent(
            nct_id="NCT12345678",
            event_type=EventType.CT_STATUS_UPGRADE,
            direction="POS",
            impact=2,
            confidence=0.85,
            disclosed_at=as_of_date,
        )

        assert event.nct_id == "NCT12345678"
        assert event.event_type == EventType.CT_STATUS_UPGRADE
        assert event.direction == "POS"
        assert event.impact == 2
        assert event.disclosed_at == as_of_date

    def test_auto_generates_rule_id(self, as_of_date):
        """Should auto-generate rule ID from event type."""
        event = CatalystEvent(
            nct_id="NCT12345678",
            event_type=EventType.CT_TIMELINE_PUSHOUT,
            disclosed_at=as_of_date,
        )

        assert event.event_rule_id == "M3_DIFF_CT_TIMELINE_PUSHOUT"

    def test_fields_changed_default(self, as_of_date):
        """Fields changed should default to empty dict."""
        event = CatalystEvent(
            nct_id="NCT12345678",
            event_type=EventType.CT_STATUS_UPGRADE,
            disclosed_at=as_of_date,
        )

        assert event.fields_changed == {}

    def test_to_dict(self, as_of_date):
        """Should serialize to dict correctly."""
        event = CatalystEvent(
            nct_id="NCT12345678",
            event_type=EventType.CT_STATUS_UPGRADE,
            direction="POS",
            impact=2,
            confidence=0.85,
            disclosed_at=as_of_date,
            actual_date=as_of_date,
            confidence_reason="Test reason",
        )

        d = event.to_dict()

        assert d["nct_id"] == "NCT12345678"
        assert d["event_type"] == "CT_STATUS_UPGRADE"
        assert d["direction"] == "POS"
        assert d["impact"] == 2
        assert d["disclosed_at"] == as_of_date.isoformat()
        assert d["actual_date"] == as_of_date.isoformat()
        assert d["confidence_reason"] == "Test reason"

    def test_effective_trading_date(self, as_of_date, calendar):
        """Should return next trading day after disclosure."""
        # Friday disclosure
        friday = date(2026, 1, 10)  # A Friday
        event = CatalystEvent(
            nct_id="NCT12345678",
            event_type=EventType.CT_STATUS_UPGRADE,
            disclosed_at=friday,
        )

        effective = event.effective_trading_date(calendar)

        # Should skip to Monday
        assert effective.weekday() < 5  # Weekday

    def test_days_to_event(self, as_of_date, calendar):
        """Should calculate days to event correctly."""
        event = CatalystEvent(
            nct_id="NCT12345678",
            event_type=EventType.CT_STATUS_UPGRADE,
            disclosed_at=as_of_date + timedelta(days=5),
        )

        days = event.days_to_event(as_of_date, calendar)

        assert days >= 1


# ============================================================================
# STATUS CHANGE CLASSIFICATION TESTS
# ============================================================================

class TestClassifyStatusChange:
    """Tests for classify_status_change function."""

    def test_severe_neg_suspended(self):
        """Suspended status should be severe negative."""
        event_type, impact, direction = classify_status_change(
            CTGovStatus.RECRUITING,
            CTGovStatus.SUSPENDED,
        )

        assert event_type == EventType.CT_STATUS_SEVERE_NEG
        assert impact == 3
        assert direction == "NEG"

    def test_severe_neg_terminated(self):
        """Terminated status should be severe negative."""
        event_type, impact, direction = classify_status_change(
            CTGovStatus.RECRUITING,
            CTGovStatus.TERMINATED,
        )

        assert event_type == EventType.CT_STATUS_SEVERE_NEG
        assert direction == "NEG"

    def test_severe_neg_withdrawn(self):
        """Withdrawn status should be severe negative."""
        event_type, impact, direction = classify_status_change(
            CTGovStatus.NOT_YET_RECRUITING,
            CTGovStatus.WITHDRAWN,
        )

        assert event_type == EventType.CT_STATUS_SEVERE_NEG

    def test_upgrade(self):
        """Status improvement should be upgrade."""
        event_type, impact, direction = classify_status_change(
            CTGovStatus.NOT_YET_RECRUITING,
            CTGovStatus.RECRUITING,
        )

        assert event_type == EventType.CT_STATUS_UPGRADE
        assert direction == "POS"

    def test_downgrade(self):
        """Status worsening should be downgrade."""
        event_type, impact, direction = classify_status_change(
            CTGovStatus.RECRUITING,
            CTGovStatus.NOT_YET_RECRUITING,
        )

        assert event_type == EventType.CT_STATUS_DOWNGRADE
        assert direction == "NEG"

    def test_no_change(self):
        """Same status should return None."""
        event_type, impact, direction = classify_status_change(
            CTGovStatus.RECRUITING,
            CTGovStatus.RECRUITING,
        )

        assert event_type is None
        assert impact == 0
        assert direction == "NEUTRAL"


# ============================================================================
# TIMELINE CHANGE CLASSIFICATION TESTS
# ============================================================================

class TestClassifyTimelineChange:
    """Tests for classify_timeline_change function."""

    def test_pushout_detected(self):
        """Timeline delay should be pushout."""
        old_date = date(2026, 6, 1)
        new_date = date(2026, 8, 1)  # 61 days later

        event_type, impact, direction = classify_timeline_change(old_date, new_date)

        assert event_type == EventType.CT_TIMELINE_PUSHOUT
        assert direction == "NEG"
        assert impact == 2  # 60-180 days

    def test_pullin_detected(self):
        """Timeline acceleration should be pullin."""
        old_date = date(2026, 8, 1)
        new_date = date(2026, 6, 1)  # 61 days earlier

        event_type, impact, direction = classify_timeline_change(old_date, new_date)

        assert event_type == EventType.CT_TIMELINE_PULLIN
        assert direction == "POS"

    def test_noise_band_filters_small_changes(self):
        """Small changes should be filtered as noise."""
        old_date = date(2026, 6, 1)
        new_date = date(2026, 6, 10)  # 9 days - within noise band

        event_type, impact, direction = classify_timeline_change(old_date, new_date)

        assert event_type is None
        assert direction == "NEUTRAL"

    def test_custom_noise_band(self):
        """Custom noise band should be respected."""
        old_date = date(2026, 6, 1)
        new_date = date(2026, 6, 10)  # 9 days

        # With small noise band, this should be detected
        event_type, impact, direction = classify_timeline_change(
            old_date, new_date, noise_band_days=5
        )

        assert event_type is not None

    def test_impact_scaling_small(self):
        """Small delays should have low impact."""
        old_date = date(2026, 6, 1)
        new_date = date(2026, 6, 30)  # 29 days

        event_type, impact, direction = classify_timeline_change(old_date, new_date)

        assert impact == 1  # < 60 days

    def test_impact_scaling_medium(self):
        """Medium delays should have medium impact."""
        old_date = date(2026, 6, 1)
        new_date = date(2026, 9, 1)  # ~90 days

        event_type, impact, direction = classify_timeline_change(old_date, new_date)

        assert impact == 2  # 60-180 days

    def test_impact_scaling_large(self):
        """Large delays should have high impact."""
        old_date = date(2026, 6, 1)
        new_date = date(2027, 1, 1)  # ~200 days

        event_type, impact, direction = classify_timeline_change(old_date, new_date)

        assert impact == 3  # > 180 days


# ============================================================================
# DATE CONFIRMATION CLASSIFICATION TESTS
# ============================================================================

class TestClassifyDateConfirmation:
    """Tests for classify_date_confirmation function."""

    def test_anticipated_to_actual(self):
        """ANTICIPATED to ACTUAL should be confirmed."""
        actual_date = date(2026, 1, 1)
        as_of_date = date(2026, 1, 15)  # 14 days after

        event_type, impact, direction, returned_date = classify_date_confirmation(
            old_type=CompletionType.ANTICIPATED,
            new_type=CompletionType.ACTUAL,
            actual_date=actual_date,
            as_of_date=as_of_date,
        )

        assert event_type == EventType.CT_DATE_CONFIRMED_ACTUAL
        assert direction == "POS"
        assert returned_date == actual_date

    def test_estimated_to_actual(self):
        """ESTIMATED to ACTUAL should be confirmed."""
        actual_date = date(2026, 1, 1)
        as_of_date = date(2026, 1, 15)

        event_type, impact, direction, returned_date = classify_date_confirmation(
            old_type=CompletionType.ESTIMATED,
            new_type=CompletionType.ACTUAL,
            actual_date=actual_date,
            as_of_date=as_of_date,
        )

        assert event_type == EventType.CT_DATE_CONFIRMED_ACTUAL

    def test_stale_actual_ignored(self):
        """Old actual dates should not trigger event."""
        actual_date = date(2025, 6, 1)  # 7+ months ago
        as_of_date = date(2026, 1, 15)

        event_type, impact, direction, returned_date = classify_date_confirmation(
            old_type=CompletionType.ESTIMATED,
            new_type=CompletionType.ACTUAL,
            actual_date=actual_date,
            as_of_date=as_of_date,
            recency_threshold_days=90,
        )

        assert event_type is None

    def test_no_type_change(self):
        """Same type should not trigger event."""
        actual_date = date(2026, 1, 1)
        as_of_date = date(2026, 1, 15)

        event_type, impact, direction, returned_date = classify_date_confirmation(
            old_type=CompletionType.ESTIMATED,
            new_type=CompletionType.ESTIMATED,
            actual_date=actual_date,
            as_of_date=as_of_date,
        )

        assert event_type is None


# ============================================================================
# RESULTS POSTED CLASSIFICATION TESTS
# ============================================================================

class TestClassifyResultsPosted:
    """Tests for classify_results_posted function."""

    def test_first_results_posted(self):
        """First results posting should trigger event."""
        event_type, impact, direction = classify_results_posted(
            old_results_date=None,
            new_results_date=date(2026, 1, 10),
        )

        assert event_type == EventType.CT_RESULTS_POSTED
        assert direction == "NEUTRAL"  # Results are informational

    def test_results_date_updated(self):
        """Results date update should trigger event."""
        event_type, impact, direction = classify_results_posted(
            old_results_date=date(2026, 1, 5),
            new_results_date=date(2026, 1, 10),
        )

        assert event_type == EventType.CT_RESULTS_POSTED

    def test_no_results_change(self):
        """Same results date should not trigger event."""
        event_type, impact, direction = classify_results_posted(
            old_results_date=date(2026, 1, 10),
            new_results_date=date(2026, 1, 10),
        )

        assert event_type is None

    def test_both_none(self):
        """Both None should not trigger event."""
        event_type, impact, direction = classify_results_posted(
            old_results_date=None,
            new_results_date=None,
        )

        assert event_type is None


# ============================================================================
# EVENT DETECTOR CONFIG TESTS
# ============================================================================

class TestEventDetectorConfig:
    """Tests for EventDetectorConfig."""

    def test_default_config(self):
        """Default config should have expected values."""
        config = EventDetectorConfig()

        assert config.noise_band_days == 14
        assert config.recency_threshold_days == 90
        assert config.confidence_scores is not None

    def test_confidence_scores_populated(self):
        """Confidence scores should be populated for all event types."""
        config = EventDetectorConfig()

        assert EventType.CT_STATUS_SEVERE_NEG in config.confidence_scores
        assert EventType.CT_TIMELINE_PUSHOUT in config.confidence_scores
        assert EventType.CT_ACTIVITY_PROXY in config.confidence_scores

    def test_custom_noise_band(self):
        """Custom noise band should be accepted."""
        config = EventDetectorConfig(noise_band_days=7)

        assert config.noise_band_days == 7


# ============================================================================
# EVENT DETECTOR TESTS
# ============================================================================

class TestEventDetector:
    """Tests for EventDetector class."""

    def test_no_prior_record_returns_empty(self, detector, sample_trial_record, as_of_date):
        """No prior record should return empty list."""
        events = detector.detect_events(
            current_record=sample_trial_record,
            prior_record=None,
            as_of_date=as_of_date,
        )

        assert events == []

    def test_detects_status_change(self, detector, as_of_date):
        """Should detect status change events."""
        prior = CanonicalTrialRecord(
            ticker="ACME",
            nct_id="NCT12345678",
            overall_status=CTGovStatus.RECRUITING,
            last_update_posted=date(2026, 1, 1),
            primary_completion_date=None,
            primary_completion_type=None,
            completion_date=None,
            completion_type=None,
            results_first_posted=None,
        )
        current = CanonicalTrialRecord(
            ticker="ACME",
            nct_id="NCT12345678",
            overall_status=CTGovStatus.SUSPENDED,
            last_update_posted=date(2026, 1, 10),
            primary_completion_date=None,
            primary_completion_type=None,
            completion_date=None,
            completion_type=None,
            results_first_posted=None,
        )

        events = detector.detect_events(current, prior, as_of_date)

        assert len(events) == 1
        assert events[0].event_type == EventType.CT_STATUS_SEVERE_NEG

    def test_detects_timeline_change(self, detector, as_of_date):
        """Should detect timeline change events."""
        prior = CanonicalTrialRecord(
            ticker="ACME",
            nct_id="NCT12345678",
            overall_status=CTGovStatus.RECRUITING,
            last_update_posted=date(2026, 1, 1),
            primary_completion_date=date(2026, 6, 1),
            primary_completion_type=CompletionType.ESTIMATED,
            completion_date=None,
            completion_type=None,
            results_first_posted=None,
        )
        current = CanonicalTrialRecord(
            ticker="ACME",
            nct_id="NCT12345678",
            overall_status=CTGovStatus.RECRUITING,
            last_update_posted=date(2026, 1, 10),
            primary_completion_date=date(2026, 9, 1),  # 3 months later
            primary_completion_type=CompletionType.ESTIMATED,
            completion_date=None,
            completion_type=None,
            results_first_posted=None,
        )

        events = detector.detect_events(current, prior, as_of_date)

        assert len(events) == 1
        assert events[0].event_type == EventType.CT_TIMELINE_PUSHOUT

    def test_detects_results_posted(self, detector, as_of_date):
        """Should detect results posted events."""
        prior = CanonicalTrialRecord(
            ticker="ACME",
            nct_id="NCT12345678",
            overall_status=CTGovStatus.COMPLETED,
            last_update_posted=date(2026, 1, 1),
            primary_completion_date=None,
            primary_completion_type=None,
            completion_date=None,
            completion_type=None,
            results_first_posted=None,
        )
        current = CanonicalTrialRecord(
            ticker="ACME",
            nct_id="NCT12345678",
            overall_status=CTGovStatus.COMPLETED,
            last_update_posted=date(2026, 1, 10),
            primary_completion_date=None,
            primary_completion_type=None,
            completion_date=None,
            completion_type=None,
            results_first_posted=date(2026, 1, 10),
        )

        events = detector.detect_events(current, prior, as_of_date)

        assert any(e.event_type == EventType.CT_RESULTS_POSTED for e in events)

    def test_detects_multiple_events(self, detector, as_of_date):
        """Should detect multiple events from single delta."""
        prior = CanonicalTrialRecord(
            ticker="ACME",
            nct_id="NCT12345678",
            overall_status=CTGovStatus.RECRUITING,
            last_update_posted=date(2026, 1, 1),
            primary_completion_date=date(2026, 6, 1),
            primary_completion_type=CompletionType.ESTIMATED,
            completion_date=None,
            completion_type=None,
            results_first_posted=None,
        )
        current = CanonicalTrialRecord(
            ticker="ACME",
            nct_id="NCT12345678",
            overall_status=CTGovStatus.SUSPENDED,  # Status change
            last_update_posted=date(2026, 1, 10),
            primary_completion_date=date(2026, 9, 1),  # Timeline change
            primary_completion_type=CompletionType.ESTIMATED,
            completion_date=None,
            completion_type=None,
            results_first_posted=None,
        )

        events = detector.detect_events(current, prior, as_of_date)

        assert len(events) >= 2
        event_types = {e.event_type for e in events}
        assert EventType.CT_STATUS_SEVERE_NEG in event_types

    def test_activity_proxy_when_no_other_events(self, detector, as_of_date):
        """Should detect activity proxy when no other events detected."""
        prior = CanonicalTrialRecord(
            ticker="ACME",
            nct_id="NCT12345678",
            overall_status=CTGovStatus.RECRUITING,
            last_update_posted=date(2026, 1, 1),
            primary_completion_date=None,
            primary_completion_type=None,
            completion_date=None,
            completion_type=None,
            results_first_posted=None,
        )
        current = CanonicalTrialRecord(
            ticker="ACME",
            nct_id="NCT12345678",
            overall_status=CTGovStatus.RECRUITING,  # Same status
            last_update_posted=date(2026, 1, 10),  # Update posted changed
            primary_completion_date=None,
            primary_completion_type=None,
            completion_date=None,
            completion_type=None,
            results_first_posted=None,
        )

        events = detector.detect_events(current, prior, as_of_date)

        assert len(events) == 1
        assert events[0].event_type == EventType.CT_ACTIVITY_PROXY


# ============================================================================
# EVENT SCORING TESTS
# ============================================================================

class TestComputeEventScore:
    """Tests for compute_event_score function."""

    def test_basic_score(self, as_of_date, calendar):
        """Should compute basic event score."""
        event = CatalystEvent(
            nct_id="NCT12345678",
            event_type=EventType.CT_STATUS_UPGRADE,
            impact=2,
            confidence=0.85,
            disclosed_at=as_of_date,
        )

        score = compute_event_score(event, as_of_date, calendar)

        # Score = impact * confidence * proximity
        # proximity = 1.0 for non-date-confirmed events
        expected = 2 * 0.85 * 1.0
        assert score == pytest.approx(expected)

    def test_date_confirmed_decays(self, as_of_date, calendar):
        """Date confirmed events should decay from actual date."""
        actual_date = as_of_date - timedelta(days=30)
        event = CatalystEvent(
            nct_id="NCT12345678",
            event_type=EventType.CT_DATE_CONFIRMED_ACTUAL,
            impact=1,
            confidence=0.85,
            disclosed_at=as_of_date,
            actual_date=actual_date,
        )

        score = compute_event_score(event, as_of_date, calendar, decay_constant=30.0)

        # Score should be less than non-decayed
        max_score = 1 * 0.85 * 1.0
        assert score < max_score


# ============================================================================
# ACTIVITY PROXY DETECTION TESTS
# ============================================================================

class TestDetectActivityProxyFromLookback:
    """Tests for detect_activity_proxy_from_lookback function."""

    def test_detects_recent_activity(self, as_of_date):
        """Should detect trials with recent updates."""
        trials = [
            CanonicalTrialRecord(
                ticker="ACME",
                nct_id="NCT12345678",
                overall_status=CTGovStatus.RECRUITING,
                last_update_posted=as_of_date - timedelta(days=30),
                primary_completion_date=None,
                primary_completion_type=None,
                completion_date=None,
                completion_type=None,
                results_first_posted=None,
            ),
        ]

        events = detect_activity_proxy_from_lookback(trials, as_of_date, lookback_days=90)

        assert "ACME" in events
        assert len(events["ACME"]) == 1
        assert events["ACME"][0].event_type == EventType.CT_ACTIVITY_PROXY

    def test_filters_old_updates(self, as_of_date):
        """Should filter trials with old updates."""
        trials = [
            CanonicalTrialRecord(
                ticker="ACME",
                nct_id="NCT12345678",
                overall_status=CTGovStatus.RECRUITING,
                last_update_posted=as_of_date - timedelta(days=200),  # Too old
                primary_completion_date=None,
                primary_completion_type=None,
                completion_date=None,
                completion_type=None,
                results_first_posted=None,
            ),
        ]

        events = detect_activity_proxy_from_lookback(trials, as_of_date, lookback_days=90)

        assert len(events) == 0


class TestComputeActivityProxyScore:
    """Tests for compute_activity_proxy_score function."""

    def test_computes_score(self, as_of_date):
        """Should compute activity proxy score."""
        trials = [
            CanonicalTrialRecord(
                ticker="ACME",
                nct_id="NCT12345678",
                overall_status=CTGovStatus.RECRUITING,
                last_update_posted=as_of_date - timedelta(days=15),
                primary_completion_date=None,
                primary_completion_type=None,
                completion_date=None,
                completion_type=None,
                results_first_posted=None,
            ),
        ]

        result = compute_activity_proxy_score(trials, as_of_date)

        assert "activity_proxy_score" in result
        assert result["activity_proxy_score"] > 0
        assert result["activity_count_120d"] == 1

    def test_counts_30d_activity(self, as_of_date):
        """Should count 30-day activity separately."""
        trials = [
            CanonicalTrialRecord(
                ticker="ACME",
                nct_id="NCT11111111",
                overall_status=CTGovStatus.RECRUITING,
                last_update_posted=as_of_date - timedelta(days=15),  # Within 30d
                primary_completion_date=None,
                primary_completion_type=None,
                completion_date=None,
                completion_type=None,
                results_first_posted=None,
            ),
            CanonicalTrialRecord(
                ticker="ACME",
                nct_id="NCT22222222",
                overall_status=CTGovStatus.RECRUITING,
                last_update_posted=as_of_date - timedelta(days=60),  # Outside 30d
                primary_completion_date=None,
                primary_completion_type=None,
                completion_date=None,
                completion_type=None,
                results_first_posted=None,
            ),
        ]

        result = compute_activity_proxy_score(trials, as_of_date)

        assert result["activity_count_30d"] == 1
        assert result["activity_count_120d"] == 2


# ============================================================================
# SIMPLE MARKET CALENDAR TESTS
# ============================================================================

class TestSimpleMarketCalendar:
    """Tests for SimpleMarketCalendar."""

    def test_skips_saturday(self, calendar):
        """Should skip Saturday."""
        friday = date(2026, 1, 9)  # A Friday
        next_day = calendar.next_trading_day(friday)

        assert next_day.weekday() == 0  # Monday

    def test_skips_sunday(self, calendar):
        """Should skip Sunday."""
        saturday = date(2026, 1, 10)  # A Saturday
        next_day = calendar.next_trading_day(saturday)

        assert next_day.weekday() == 0  # Monday

    def test_weekday_returns_next_weekday(self, calendar):
        """Weekday should return next weekday."""
        wednesday = date(2026, 1, 14)  # A Wednesday
        next_day = calendar.next_trading_day(wednesday)

        assert next_day.weekday() == 3  # Thursday


# ============================================================================
# DETERMINISM TESTS
# ============================================================================

class TestDeterminism:
    """Tests verifying deterministic behavior."""

    def test_detect_events_deterministic(self, detector, as_of_date):
        """detect_events should be deterministic."""
        prior = CanonicalTrialRecord(
            ticker="ACME",
            nct_id="NCT12345678",
            overall_status=CTGovStatus.RECRUITING,
            last_update_posted=date(2026, 1, 1),
            primary_completion_date=date(2026, 6, 1),
            primary_completion_type=CompletionType.ESTIMATED,
            completion_date=None,
            completion_type=None,
            results_first_posted=None,
        )
        current = CanonicalTrialRecord(
            ticker="ACME",
            nct_id="NCT12345678",
            overall_status=CTGovStatus.SUSPENDED,
            last_update_posted=date(2026, 1, 10),
            primary_completion_date=date(2026, 9, 1),
            primary_completion_type=CompletionType.ESTIMATED,
            completion_date=None,
            completion_type=None,
            results_first_posted=None,
        )

        results = [
            detector.detect_events(current, prior, as_of_date)
            for _ in range(5)
        ]

        # All results should be identical
        for i in range(1, len(results)):
            assert len(results[0]) == len(results[i])
            for j in range(len(results[0])):
                assert results[0][j].event_type == results[i][j].event_type
                assert results[0][j].impact == results[i][j].impact
