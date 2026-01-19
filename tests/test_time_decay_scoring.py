#!/usr/bin/env python3
"""
test_time_decay_scoring.py - Tests for time-decay scoring module

Tests the multi-window time-decay scoring system that weights catalyst
events based on recency.
"""

import pytest
from datetime import date, timedelta
from decimal import Decimal

from time_decay_scoring import (
    TimeDecayConfig,
    TimeDecayWindow,
    WindowScoreResult,
    TimeDecayScoreResult,
    DEFAULT_TIME_DECAY_WINDOWS,
    compute_event_score,
    filter_events_for_window,
    score_window,
    compute_time_decay_score,
    score_all_tickers_with_time_decay,
    integrate_time_decay_into_summary,
)
from module_3_schema_v2 import (
    CatalystEventV2,
    EventType,
    EventSeverity,
    ConfidenceLevel,
    SourceReliability,
    DateSpecificity,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def as_of_date():
    """Standard test as_of_date."""
    return date(2026, 1, 15)


@pytest.fixture
def sample_event_positive(as_of_date):
    """Sample positive catalyst event disclosed 3 days ago."""
    disclosed = (as_of_date - timedelta(days=3)).isoformat()
    return CatalystEventV2(
        ticker="SIGA",
        nct_id="NCT12345678",
        event_type=EventType.CT_STATUS_UPGRADE,
        event_severity=EventSeverity.POSITIVE,
        event_date=(as_of_date + timedelta(days=30)).isoformat(),
        field_changed="overall_status",
        prior_value="NOT_YET_RECRUITING",
        new_value="RECRUITING",
        source="ctgov",
        confidence=ConfidenceLevel.HIGH,
        disclosed_at=disclosed,
        source_date=disclosed,
        pit_date_field_used="last_update_posted",
        source_reliability=SourceReliability.OFFICIAL,
        date_specificity=DateSpecificity.EXACT,
    )


@pytest.fixture
def sample_event_negative(as_of_date):
    """Sample negative catalyst event disclosed 10 days ago."""
    disclosed = (as_of_date - timedelta(days=10)).isoformat()
    return CatalystEventV2(
        ticker="SIGA",
        nct_id="NCT12345679",
        event_type=EventType.CT_TIMELINE_PUSHOUT,
        event_severity=EventSeverity.NEGATIVE,
        event_date=(as_of_date + timedelta(days=90)).isoformat(),
        field_changed="primary_completion_date",
        prior_value="2026-03-01",
        new_value="2026-06-01",
        source="ctgov",
        confidence=ConfidenceLevel.MED,
        disclosed_at=disclosed,
        source_date=disclosed,
        pit_date_field_used="last_update_posted",
        source_reliability=SourceReliability.OFFICIAL,
        date_specificity=DateSpecificity.MONTH,
    )


@pytest.fixture
def sample_event_old(as_of_date):
    """Sample event disclosed 45 days ago (outside 30d window)."""
    disclosed = (as_of_date - timedelta(days=45)).isoformat()
    return CatalystEventV2(
        ticker="SIGA",
        nct_id="NCT12345680",
        event_type=EventType.CT_RESULTS_POSTED,
        event_severity=EventSeverity.CRITICAL_POSITIVE,
        event_date=(as_of_date - timedelta(days=40)).isoformat(),
        field_changed="results_first_posted",
        prior_value=None,
        new_value=disclosed,
        source="ctgov",
        confidence=ConfidenceLevel.HIGH,
        disclosed_at=disclosed,
        source_date=disclosed,
        pit_date_field_used="results_first_posted",
        source_reliability=SourceReliability.OFFICIAL,
        date_specificity=DateSpecificity.EXACT,
    )


# =============================================================================
# TEST TIME DECAY WINDOW
# =============================================================================

class TestTimeDecayWindow:
    """Tests for TimeDecayWindow configuration."""

    def test_default_windows(self):
        """Test default window configuration."""
        assert len(DEFAULT_TIME_DECAY_WINDOWS) == 3
        assert DEFAULT_TIME_DECAY_WINDOWS[0].name == "7d"
        assert DEFAULT_TIME_DECAY_WINDOWS[0].weight == Decimal("1.00")
        assert DEFAULT_TIME_DECAY_WINDOWS[1].name == "30d"
        assert DEFAULT_TIME_DECAY_WINDOWS[1].weight == Decimal("0.50")
        assert DEFAULT_TIME_DECAY_WINDOWS[2].name == "90d"
        assert DEFAULT_TIME_DECAY_WINDOWS[2].weight == Decimal("0.25")

    def test_get_lookback_date(self, as_of_date):
        """Test lookback date calculation."""
        window = TimeDecayWindow("7d", 7, Decimal("1.0"), "test")
        lookback = window.get_lookback_date(as_of_date)
        assert lookback == as_of_date - timedelta(days=7)


class TestTimeDecayConfig:
    """Tests for TimeDecayConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = TimeDecayConfig()
        assert config.use_max_across_windows is True
        assert config.enable_cluster_bonus is True
        assert config.cluster_bonus_factor == Decimal("1.15")
        assert len(config.windows) == 3

    def test_from_dict(self):
        """Test config creation from dictionary."""
        config_dict = {
            "use_max_across_windows": False,
            "enable_cluster_bonus": False,
            "cluster_bonus_factor": "1.25",
            "windows": [
                {"name": "14d", "lookback_days": 14, "weight": "0.8"},
                {"name": "60d", "lookback_days": 60, "weight": "0.4"},
            ]
        }
        config = TimeDecayConfig.from_dict(config_dict)
        assert config.use_max_across_windows is False
        assert config.enable_cluster_bonus is False
        assert config.cluster_bonus_factor == Decimal("1.25")
        assert len(config.windows) == 2
        assert config.windows[0].name == "14d"


# =============================================================================
# TEST EVENT SCORING
# =============================================================================

class TestEventScoring:
    """Tests for individual event scoring."""

    def test_compute_event_score_positive(self, sample_event_positive, as_of_date):
        """Test scoring for positive event."""
        score = compute_event_score(sample_event_positive, as_of_date)
        # Positive event with HIGH confidence should have positive score
        assert score > Decimal("0")

    def test_compute_event_score_negative(self, sample_event_negative, as_of_date):
        """Test scoring for negative event."""
        score = compute_event_score(sample_event_negative, as_of_date)
        # Negative event should have negative score
        assert score < Decimal("0")


# =============================================================================
# TEST WINDOW FILTERING
# =============================================================================

class TestWindowFiltering:
    """Tests for event filtering by time window."""

    def test_filter_events_7d_window(
        self,
        sample_event_positive,
        sample_event_negative,
        sample_event_old,
        as_of_date,
    ):
        """Test filtering for 7-day window."""
        window = DEFAULT_TIME_DECAY_WINDOWS[0]  # 7d window
        events = [sample_event_positive, sample_event_negative, sample_event_old]

        filtered = filter_events_for_window(events, window, as_of_date)

        # Only sample_event_positive (3 days old) should be in 7d window
        assert len(filtered) == 1
        assert filtered[0].nct_id == "NCT12345678"

    def test_filter_events_30d_window(
        self,
        sample_event_positive,
        sample_event_negative,
        sample_event_old,
        as_of_date,
    ):
        """Test filtering for 30-day window."""
        window = DEFAULT_TIME_DECAY_WINDOWS[1]  # 30d window
        events = [sample_event_positive, sample_event_negative, sample_event_old]

        filtered = filter_events_for_window(events, window, as_of_date)

        # sample_event_positive (3 days) and sample_event_negative (10 days)
        # should be in 30d window
        assert len(filtered) == 2

    def test_filter_events_90d_window(
        self,
        sample_event_positive,
        sample_event_negative,
        sample_event_old,
        as_of_date,
    ):
        """Test filtering for 90-day window."""
        window = DEFAULT_TIME_DECAY_WINDOWS[2]  # 90d window
        events = [sample_event_positive, sample_event_negative, sample_event_old]

        filtered = filter_events_for_window(events, window, as_of_date)

        # All events should be in 90d window
        assert len(filtered) == 3

    def test_filter_excludes_future_events(self, as_of_date):
        """Test that events disclosed in the future are excluded (PIT safety)."""
        future_disclosed = (as_of_date + timedelta(days=1)).isoformat()
        future_event = CatalystEventV2(
            ticker="TEST",
            nct_id="NCT99999999",
            event_type=EventType.CT_STATUS_UPGRADE,
            event_severity=EventSeverity.POSITIVE,
            event_date=future_disclosed,
            field_changed="overall_status",
            prior_value="NOT_YET_RECRUITING",
            new_value="RECRUITING",
            source="ctgov",
            confidence=ConfidenceLevel.HIGH,
            disclosed_at=future_disclosed,  # Future date
            source_date=future_disclosed,
            pit_date_field_used="last_update_posted",
        )

        window = DEFAULT_TIME_DECAY_WINDOWS[0]  # 7d window
        filtered = filter_events_for_window([future_event], window, as_of_date)

        assert len(filtered) == 0  # Future event should be excluded


# =============================================================================
# TEST WINDOW SCORING
# =============================================================================

class TestWindowScoring:
    """Tests for scoring events within a window."""

    def test_score_window_with_events(
        self,
        sample_event_positive,
        as_of_date,
    ):
        """Test window scoring with events."""
        window = DEFAULT_TIME_DECAY_WINDOWS[0]  # 7d window
        result = score_window([sample_event_positive], window, as_of_date)

        assert result.window_name == "7d"
        assert result.events_in_window == 1
        assert result.raw_score > Decimal("0")
        assert result.weighted_score == result.raw_score * window.weight
        assert len(result.event_ids) == 1

    def test_score_empty_window(self, as_of_date):
        """Test window scoring with no events."""
        window = DEFAULT_TIME_DECAY_WINDOWS[0]
        result = score_window([], window, as_of_date)

        assert result.events_in_window == 0
        assert result.raw_score == Decimal("0")
        assert result.weighted_score == Decimal("0")
        assert len(result.event_ids) == 0


# =============================================================================
# TEST TIME DECAY SCORING
# =============================================================================

class TestTimeDecayScoring:
    """Tests for complete time-decay scoring."""

    def test_compute_time_decay_score_single_window(
        self,
        sample_event_positive,
        as_of_date,
    ):
        """Test time-decay scoring with events in single window."""
        result = compute_time_decay_score(
            "SIGA",
            [sample_event_positive],
            as_of_date,
        )

        assert result.ticker == "SIGA"
        assert result.final_score > Decimal("0")
        assert result.contributing_window == "7d"  # Highest weight for recent event
        assert result.unique_events_total == 1
        # A 3-day-old event falls within all windows (7d, 30d, 90d)
        assert result.windows_with_events == 3
        # Cluster detected since event appears in multiple windows
        assert result.cluster_detected is True

    def test_compute_time_decay_score_cluster(
        self,
        sample_event_positive,
        sample_event_old,
        as_of_date,
    ):
        """Test time-decay scoring with cluster across windows."""
        events = [sample_event_positive, sample_event_old]
        result = compute_time_decay_score("SIGA", events, as_of_date)

        # Events span 7d and 90d windows - should detect cluster
        assert result.windows_with_events >= 2
        assert result.cluster_detected is True
        assert result.cluster_bonus_applied is True

    def test_compute_time_decay_score_no_events(self, as_of_date):
        """Test time-decay scoring with no events."""
        result = compute_time_decay_score("TEST", [], as_of_date)

        assert result.ticker == "TEST"
        assert result.final_score == Decimal("0")
        assert result.unique_events_total == 0
        assert result.windows_with_events == 0
        assert result.cluster_detected is False

    def test_compute_time_decay_score_max_window(
        self,
        sample_event_positive,
        sample_event_old,
        as_of_date,
    ):
        """Test that max window is used by default."""
        config = TimeDecayConfig()
        assert config.use_max_across_windows is True

        events = [sample_event_positive, sample_event_old]
        result = compute_time_decay_score("SIGA", events, as_of_date, config)

        # Should use max, not sum
        assert result.contributing_window in ["7d", "30d", "90d"]


# =============================================================================
# TEST BATCH SCORING
# =============================================================================

class TestBatchScoring:
    """Tests for batch scoring all tickers."""

    def test_score_all_tickers(
        self,
        sample_event_positive,
        sample_event_negative,
        as_of_date,
    ):
        """Test batch scoring multiple tickers."""
        events_by_ticker = {
            "SIGA": [sample_event_positive],
            "TEST": [],
        }
        active_tickers = ["SIGA", "TEST", "EMPTY"]

        results, diagnostics = score_all_tickers_with_time_decay(
            events_by_ticker,
            active_tickers,
            as_of_date,
        )

        assert len(results) == 3
        assert "SIGA" in results
        assert "TEST" in results
        assert "EMPTY" in results

        assert results["SIGA"].final_score > Decimal("0")
        assert results["TEST"].final_score == Decimal("0")
        assert results["EMPTY"].final_score == Decimal("0")

        assert diagnostics["tickers_scored"] == 3


# =============================================================================
# TEST INTEGRATION HELPER
# =============================================================================

class TestIntegrationHelper:
    """Tests for integration with composite scoring."""

    def test_integrate_positive_momentum(self, as_of_date):
        """Test integration with positive momentum."""
        td_result = TimeDecayScoreResult(
            ticker="TEST",
            as_of_date=as_of_date.isoformat(),
            final_score=Decimal("20.0"),
            contributing_window="7d",
            cluster_detected=False,
            cluster_bonus_applied=False,
            unique_events_total=2,
            windows_with_events=1,
        )

        base_score = Decimal("50.0")
        adjusted, explanation = integrate_time_decay_into_summary(
            td_result, base_score
        )

        # Positive momentum should increase score
        assert adjusted > base_score
        assert "Boosted" in explanation

    def test_integrate_negative_momentum(self, as_of_date):
        """Test integration with negative momentum."""
        td_result = TimeDecayScoreResult(
            ticker="TEST",
            as_of_date=as_of_date.isoformat(),
            final_score=Decimal("-15.0"),
            contributing_window="7d",
            cluster_detected=False,
            cluster_bonus_applied=False,
            unique_events_total=1,
            windows_with_events=1,
        )

        base_score = Decimal("50.0")
        adjusted, explanation = integrate_time_decay_into_summary(
            td_result, base_score
        )

        # Negative momentum should decrease score
        assert adjusted < base_score
        assert "Penalized" in explanation

    def test_integrate_with_cluster_bonus(self, as_of_date):
        """Test integration with cluster bonus amplification."""
        td_result = TimeDecayScoreResult(
            ticker="TEST",
            as_of_date=as_of_date.isoformat(),
            final_score=Decimal("10.0"),
            contributing_window="7d",
            cluster_detected=True,
            cluster_bonus_applied=True,
            unique_events_total=3,
            windows_with_events=2,
        )

        base_score = Decimal("50.0")
        adjusted, explanation = integrate_time_decay_into_summary(
            td_result, base_score
        )

        assert "[cluster bonus]" in explanation


# =============================================================================
# TEST DETERMINISM
# =============================================================================

class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_scoring_is_deterministic(
        self,
        sample_event_positive,
        sample_event_negative,
        as_of_date,
    ):
        """Test that scoring produces identical results across runs."""
        events = [sample_event_positive, sample_event_negative]

        result1 = compute_time_decay_score("SIGA", events, as_of_date)
        result2 = compute_time_decay_score("SIGA", events, as_of_date)

        assert result1.final_score == result2.final_score
        assert result1.contributing_window == result2.contributing_window
        assert result1.unique_events_total == result2.unique_events_total

    def test_batch_scoring_is_deterministic(
        self,
        sample_event_positive,
        as_of_date,
    ):
        """Test that batch scoring is deterministic."""
        events_by_ticker = {"SIGA": [sample_event_positive]}
        active_tickers = ["SIGA", "TEST"]

        results1, _ = score_all_tickers_with_time_decay(
            events_by_ticker, active_tickers, as_of_date
        )
        results2, _ = score_all_tickers_with_time_decay(
            events_by_ticker, active_tickers, as_of_date
        )

        assert results1["SIGA"].final_score == results2["SIGA"].final_score


# =============================================================================
# TEST SERIALIZATION
# =============================================================================

class TestSerialization:
    """Tests for result serialization."""

    def test_window_score_result_to_dict(self):
        """Test WindowScoreResult serialization."""
        result = WindowScoreResult(
            window_name="7d",
            lookback_days=7,
            window_weight=Decimal("1.0"),
            events_in_window=2,
            raw_score=Decimal("15.50"),
            weighted_score=Decimal("15.50"),
            event_ids=["abc123", "def456"],
            dominant_severity="POSITIVE",
        )

        d = result.to_dict()
        assert d["window_name"] == "7d"
        assert d["window_weight"] == "1.0"
        assert d["raw_score"] == "15.50"
        assert d["events_in_window"] == 2
        assert len(d["event_ids"]) == 2

    def test_time_decay_result_to_dict(self, as_of_date):
        """Test TimeDecayScoreResult serialization."""
        result = TimeDecayScoreResult(
            ticker="TEST",
            as_of_date=as_of_date.isoformat(),
            final_score=Decimal("25.00"),
            contributing_window="7d",
            cluster_detected=True,
            cluster_bonus_applied=True,
            unique_events_total=3,
            windows_with_events=2,
        )

        d = result.to_dict()
        assert d["ticker"] == "TEST"
        assert d["final_score"] == "25.00"
        assert d["cluster_detected"] is True
        assert d["cluster_bonus_applied"] is True
