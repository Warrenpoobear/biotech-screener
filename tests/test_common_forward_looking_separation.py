#!/usr/bin/env python3
"""
Unit tests for common/forward_looking_separation.py

Tests forward-looking signal separation for PIT safety:
- SignalSourceType and LookaheadRiskLevel enums
- SignalContribution and SeparatedCatalystResult dataclasses
- ForwardLookingSeparator class
- Lookahead risk classification
- Batch signal separation
"""

import pytest
from datetime import date
from decimal import Decimal
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.forward_looking_separation import (
    SignalSourceType,
    LookaheadRiskLevel,
    SignalContribution,
    SeparatedCatalystResult,
    BatchSeparationResult,
    ForwardLookingSeparator,
    separate_batch_signals,
    __version__,
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def as_of_date():
    """Standard as_of_date for tests."""
    return date(2026, 1, 15)


@pytest.fixture
def separator():
    """Default separator instance."""
    return ForwardLookingSeparator()


@pytest.fixture
def sample_detected_events():
    """Sample historical detected events (PIT-safe)."""
    return [
        {"event_type": "CT_STATUS_UPGRADE", "score": "5.0", "nct_id": "NCT11111111"},
        {"event_type": "CT_TIMELINE_PULLIN", "score": "3.0", "nct_id": "NCT22222222"},
    ]


@pytest.fixture
def sample_calendar_events():
    """Sample calendar events (forward-looking)."""
    return [
        {"event_type": "UPCOMING_PCD", "days_until": 25, "confidence": 0.85},
        {"event_type": "UPCOMING_SCD", "days_until": 45, "confidence": 0.70},
    ]


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestSignalSourceType:
    """Tests for SignalSourceType enum."""

    def test_all_types_defined(self):
        """All expected source types should be defined."""
        assert SignalSourceType.HISTORICAL == "HISTORICAL"
        assert SignalSourceType.FORWARD_LOOKING == "FORWARD_LOOKING"
        assert SignalSourceType.MIXED == "MIXED"


class TestLookaheadRiskLevel:
    """Tests for LookaheadRiskLevel enum."""

    def test_all_levels_defined(self):
        """All risk levels should be defined."""
        assert LookaheadRiskLevel.NONE == "NONE"
        assert LookaheadRiskLevel.LOW == "LOW"
        assert LookaheadRiskLevel.MEDIUM == "MEDIUM"
        assert LookaheadRiskLevel.HIGH == "HIGH"


# ============================================================================
# SIGNAL CONTRIBUTION TESTS
# ============================================================================

class TestSignalContribution:
    """Tests for SignalContribution dataclass."""

    def test_basic_creation(self):
        """Should create contribution with required fields."""
        contribution = SignalContribution(
            ticker="ACME",
            historical_score=Decimal("8.0"),
            forward_looking_score=Decimal("4.0"),
            total_score=Decimal("12.0"),
            historical_event_count=2,
            forward_looking_event_count=2,
            forward_looking_weight=Decimal("0.33"),
            lookahead_risk=LookaheadRiskLevel.LOW,
        )

        assert contribution.ticker == "ACME"
        assert contribution.historical_score == Decimal("8.0")
        assert contribution.lookahead_risk == LookaheadRiskLevel.LOW

    def test_to_dict(self):
        """Should serialize to dict correctly."""
        contribution = SignalContribution(
            ticker="ACME",
            historical_score=Decimal("8.0"),
            forward_looking_score=Decimal("4.0"),
            total_score=Decimal("12.0"),
            historical_event_count=2,
            forward_looking_event_count=2,
            forward_looking_weight=Decimal("0.33"),
            lookahead_risk=LookaheadRiskLevel.LOW,
            warnings=["Test warning"],
        )

        d = contribution.to_dict()

        assert d["ticker"] == "ACME"
        assert d["historical_score"] == "8.0"
        assert d["lookahead_risk"] == "LOW"
        assert d["warnings"] == ["Test warning"]


# ============================================================================
# SEPARATED CATALYST RESULT TESTS
# ============================================================================

class TestSeparatedCatalystResult:
    """Tests for SeparatedCatalystResult dataclass."""

    def test_get_backtest_safe_score(self, as_of_date):
        """Should return historical score for backtesting."""
        contribution = SignalContribution(
            ticker="ACME",
            historical_score=Decimal("8.0"),
            forward_looking_score=Decimal("4.0"),
            total_score=Decimal("12.0"),
            historical_event_count=2,
            forward_looking_event_count=2,
            forward_looking_weight=Decimal("0.33"),
            lookahead_risk=LookaheadRiskLevel.LOW,
        )

        result = SeparatedCatalystResult(
            ticker="ACME",
            as_of_date=as_of_date.isoformat(),
            historical_score=Decimal("8.0"),
            forward_looking_score=Decimal("4.0"),
            blended_score=Decimal("10.0"),
            historical_events=[],
            forward_looking_events=[],
            signal_contribution=contribution,
            forward_looking_weight=Decimal("0.5"),
        )

        assert result.get_backtest_safe_score() == Decimal("8.0")

    def test_get_production_score(self, as_of_date):
        """Should return blended score for production."""
        contribution = SignalContribution(
            ticker="ACME",
            historical_score=Decimal("8.0"),
            forward_looking_score=Decimal("4.0"),
            total_score=Decimal("12.0"),
            historical_event_count=2,
            forward_looking_event_count=2,
            forward_looking_weight=Decimal("0.33"),
            lookahead_risk=LookaheadRiskLevel.LOW,
        )

        result = SeparatedCatalystResult(
            ticker="ACME",
            as_of_date=as_of_date.isoformat(),
            historical_score=Decimal("8.0"),
            forward_looking_score=Decimal("4.0"),
            blended_score=Decimal("10.0"),
            historical_events=[],
            forward_looking_events=[],
            signal_contribution=contribution,
            forward_looking_weight=Decimal("0.5"),
        )

        assert result.get_production_score() == Decimal("10.0")

    def test_to_dict(self, as_of_date):
        """Should serialize to dict correctly."""
        contribution = SignalContribution(
            ticker="ACME",
            historical_score=Decimal("8.0"),
            forward_looking_score=Decimal("4.0"),
            total_score=Decimal("12.0"),
            historical_event_count=2,
            forward_looking_event_count=2,
            forward_looking_weight=Decimal("0.33"),
            lookahead_risk=LookaheadRiskLevel.LOW,
        )

        result = SeparatedCatalystResult(
            ticker="ACME",
            as_of_date=as_of_date.isoformat(),
            historical_score=Decimal("8.0"),
            forward_looking_score=Decimal("4.0"),
            blended_score=Decimal("10.0"),
            historical_events=[{"event": "test"}],
            forward_looking_events=[{"calendar": "test"}],
            signal_contribution=contribution,
            forward_looking_weight=Decimal("0.5"),
        )

        d = result.to_dict()

        assert d["ticker"] == "ACME"
        assert d["scores"]["historical_score"] == "8.0"
        assert d["scores"]["blended_score"] == "10.0"
        assert d["event_counts"]["historical"] == 1
        assert d["event_counts"]["forward_looking"] == 1


# ============================================================================
# FORWARD LOOKING SEPARATOR TESTS
# ============================================================================

class TestForwardLookingSeparator:
    """Tests for ForwardLookingSeparator class."""

    def test_default_initialization(self, separator):
        """Should initialize with default values."""
        assert separator.forward_looking_weight == Decimal("0.5")
        assert separator.warn_on_high_forward_looking is True
        assert separator.calendar_event_cap == Decimal("15.0")

    def test_custom_initialization(self):
        """Should accept custom parameters."""
        custom = ForwardLookingSeparator(
            forward_looking_weight=Decimal("0.3"),
            warn_on_high_forward_looking=False,
            calendar_event_cap=Decimal("20.0"),
        )

        assert custom.forward_looking_weight == Decimal("0.3")
        assert custom.warn_on_high_forward_looking is False
        assert custom.calendar_event_cap == Decimal("20.0")

    def test_classify_lookahead_risk_none(self, separator):
        """Low forward-looking weight should be NONE risk."""
        risk = separator.classify_lookahead_risk(Decimal("0.15"))
        assert risk == LookaheadRiskLevel.NONE

    def test_classify_lookahead_risk_low(self, separator):
        """Moderate forward-looking weight should be LOW risk."""
        risk = separator.classify_lookahead_risk(Decimal("0.30"))
        assert risk == LookaheadRiskLevel.LOW

    def test_classify_lookahead_risk_medium(self, separator):
        """Higher forward-looking weight should be MEDIUM risk."""
        risk = separator.classify_lookahead_risk(Decimal("0.45"))
        assert risk == LookaheadRiskLevel.MEDIUM

    def test_classify_lookahead_risk_high(self, separator):
        """Very high forward-looking weight should be HIGH risk."""
        risk = separator.classify_lookahead_risk(Decimal("0.60"))
        assert risk == LookaheadRiskLevel.HIGH

    def test_compute_event_score(self, separator):
        """Should sum scores from events."""
        events = [
            {"score": "5.0"},
            {"score": "3.0"},
            {"score": 2.0},  # Also handles numeric
        ]

        total = separator.compute_event_score(events)

        assert total == Decimal("10.0")

    def test_compute_event_score_custom_field(self, separator):
        """Should use custom score field."""
        events = [
            {"confidence": "0.85"},
            {"confidence": "0.70"},
        ]

        total = separator.compute_event_score(events, score_field="confidence")

        assert total == Decimal("1.55")

    def test_compute_event_score_empty(self, separator):
        """Should return 0 for empty events."""
        total = separator.compute_event_score([])
        assert total == Decimal("0")


# ============================================================================
# SEPARATE SIGNALS TESTS
# ============================================================================

class TestSeparateSignals:
    """Tests for separate_signals method."""

    def test_basic_separation(self, separator, as_of_date, sample_detected_events, sample_calendar_events):
        """Should separate historical and forward-looking signals."""
        result = separator.separate_signals(
            ticker="ACME",
            as_of_date=as_of_date,
            detected_events=sample_detected_events,
            calendar_events=sample_calendar_events,
        )

        assert result.ticker == "ACME"
        assert result.historical_score == Decimal("8.0")  # 5 + 3
        assert result.forward_looking_score > Decimal("0")

    def test_historical_only(self, separator, as_of_date, sample_detected_events):
        """Should handle historical-only events."""
        result = separator.separate_signals(
            ticker="ACME",
            as_of_date=as_of_date,
            detected_events=sample_detected_events,
            calendar_events=[],
        )

        assert result.historical_score == Decimal("8.0")
        assert result.forward_looking_score == Decimal("0")
        assert result.signal_contribution.lookahead_risk == LookaheadRiskLevel.NONE

    def test_forward_looking_only(self, separator, as_of_date, sample_calendar_events):
        """Should handle forward-looking only events."""
        result = separator.separate_signals(
            ticker="ACME",
            as_of_date=as_of_date,
            detected_events=[],
            calendar_events=sample_calendar_events,
        )

        assert result.historical_score == Decimal("0")
        assert result.forward_looking_score > Decimal("0")
        assert result.signal_contribution.lookahead_risk == LookaheadRiskLevel.HIGH

    def test_blended_score_calculation(self, separator, as_of_date, sample_detected_events, sample_calendar_events):
        """Blended score should apply weight to forward-looking."""
        result = separator.separate_signals(
            ticker="ACME",
            as_of_date=as_of_date,
            detected_events=sample_detected_events,
            calendar_events=sample_calendar_events,
        )

        # Blended = historical + (forward_looking * 0.5)
        expected_blended = result.historical_score + result.forward_looking_score * Decimal("0.5")
        assert result.blended_score == expected_blended.quantize(Decimal("0.01"))

    def test_calendar_event_cap(self, as_of_date):
        """Should cap forward-looking score."""
        separator = ForwardLookingSeparator(calendar_event_cap=Decimal("5.0"))

        many_calendar_events = [
            {"days_until": 10, "confidence": 1.0},
            {"days_until": 10, "confidence": 1.0},
            {"days_until": 10, "confidence": 1.0},
        ]

        result = separator.separate_signals(
            ticker="ACME",
            as_of_date=as_of_date,
            detected_events=[],
            calendar_events=many_calendar_events,
        )

        assert result.forward_looking_score <= Decimal("5.0")

    def test_proximity_scoring(self, separator, as_of_date):
        """Nearer calendar events should score higher."""
        near_events = [{"days_until": 20, "confidence": 1.0}]
        far_events = [{"days_until": 80, "confidence": 1.0}]

        near_result = separator.separate_signals(
            ticker="ACME",
            as_of_date=as_of_date,
            detected_events=[],
            calendar_events=near_events,
        )

        far_result = separator.separate_signals(
            ticker="ACME",
            as_of_date=as_of_date,
            detected_events=[],
            calendar_events=far_events,
        )

        assert near_result.forward_looking_score > far_result.forward_looking_score

    def test_high_risk_generates_warning(self, separator, as_of_date):
        """High lookahead risk should generate warning."""
        # Forward-looking only = 100% forward-looking weight = HIGH risk
        result = separator.separate_signals(
            ticker="ACME",
            as_of_date=as_of_date,
            detected_events=[],
            calendar_events=[{"days_until": 30, "confidence": 0.9}],
        )

        assert result.signal_contribution.lookahead_risk == LookaheadRiskLevel.HIGH
        assert len(result.signal_contribution.warnings) > 0
        assert "LOOKAHEAD_RISK" in result.signal_contribution.warnings[0]

    def test_no_events(self, separator, as_of_date):
        """Should handle no events."""
        result = separator.separate_signals(
            ticker="ACME",
            as_of_date=as_of_date,
            detected_events=[],
            calendar_events=[],
        )

        assert result.historical_score == Decimal("0")
        assert result.forward_looking_score == Decimal("0")
        assert result.blended_score == Decimal("0")


# ============================================================================
# BATCH SEPARATION TESTS
# ============================================================================

class TestBatchSeparation:
    """Tests for separate_batch_signals function."""

    def test_basic_batch(self, as_of_date, sample_detected_events, sample_calendar_events):
        """Should separate signals for multiple tickers."""
        detected_by_ticker = {
            "ACME": sample_detected_events,
            "BETA": [{"score": "10.0"}],
        }
        calendar_by_ticker = {
            "ACME": sample_calendar_events,
            "BETA": [],
        }

        result = separate_batch_signals(
            detected_events_by_ticker=detected_by_ticker,
            calendar_events_by_ticker=calendar_by_ticker,
            as_of_date=as_of_date,
        )

        assert result.total_tickers == 2
        assert "ACME" in result.results
        assert "BETA" in result.results

    def test_identifies_high_risk_tickers(self, as_of_date):
        """Should identify high-risk tickers."""
        detected_by_ticker = {
            "ACME": [],  # No historical
            "BETA": [{"score": "20.0"}],  # Historical heavy
        }
        calendar_by_ticker = {
            "ACME": [{"days_until": 30, "confidence": 0.9}],  # Forward-looking only
            "BETA": [],
        }

        result = separate_batch_signals(
            detected_events_by_ticker=detected_by_ticker,
            calendar_events_by_ticker=calendar_by_ticker,
            as_of_date=as_of_date,
        )

        assert "ACME" in result.tickers_high_risk
        assert "BETA" not in result.tickers_high_risk

    def test_computes_average_fl_weight(self, as_of_date):
        """Should compute average forward-looking weight."""
        detected_by_ticker = {"ACME": [{"score": "10.0"}], "BETA": [{"score": "10.0"}]}
        calendar_by_ticker = {"ACME": [], "BETA": []}

        result = separate_batch_signals(
            detected_events_by_ticker=detected_by_ticker,
            calendar_events_by_ticker=calendar_by_ticker,
            as_of_date=as_of_date,
        )

        assert result.avg_forward_looking_weight == Decimal("0.0")

    def test_handles_ticker_union(self, as_of_date):
        """Should handle tickers that appear in only one dict."""
        detected_by_ticker = {"ACME": [{"score": "5.0"}]}
        calendar_by_ticker = {"BETA": [{"days_until": 30, "confidence": 0.5}]}

        result = separate_batch_signals(
            detected_events_by_ticker=detected_by_ticker,
            calendar_events_by_ticker=calendar_by_ticker,
            as_of_date=as_of_date,
        )

        assert result.total_tickers == 2
        assert "ACME" in result.results
        assert "BETA" in result.results

    def test_to_dict(self, as_of_date, sample_detected_events, sample_calendar_events):
        """Should serialize batch result to dict."""
        detected_by_ticker = {"ACME": sample_detected_events}
        calendar_by_ticker = {"ACME": sample_calendar_events}

        result = separate_batch_signals(
            detected_events_by_ticker=detected_by_ticker,
            calendar_events_by_ticker=calendar_by_ticker,
            as_of_date=as_of_date,
        )

        d = result.to_dict()

        assert "summary" in d
        assert d["summary"]["total_tickers"] == 1
        assert "results" in d
        assert "ACME" in d["results"]


# ============================================================================
# BATCH SEPARATION RESULT TESTS
# ============================================================================

class TestBatchSeparationResult:
    """Tests for BatchSeparationResult dataclass."""

    def test_to_dict(self):
        """Should serialize to dict correctly."""
        result = BatchSeparationResult(
            total_tickers=5,
            tickers_high_risk=["ACME"],
            tickers_medium_risk=["BETA", "GAMMA"],
            avg_forward_looking_weight=Decimal("0.25"),
            results={},
        )

        d = result.to_dict()

        assert d["summary"]["total_tickers"] == 5
        assert d["summary"]["high_risk_count"] == 1
        assert d["summary"]["medium_risk_count"] == 2
        assert d["summary"]["avg_forward_looking_weight"] == "0.25"


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Edge case tests for forward-looking separation."""

    def test_zero_total_score(self, separator, as_of_date):
        """Should handle zero total score."""
        result = separator.separate_signals(
            ticker="ACME",
            as_of_date=as_of_date,
            detected_events=[],
            calendar_events=[],
        )

        assert result.signal_contribution.forward_looking_weight == Decimal("0")

    def test_very_small_historical(self, separator, as_of_date):
        """Should handle very small historical contribution."""
        result = separator.separate_signals(
            ticker="ACME",
            as_of_date=as_of_date,
            detected_events=[{"score": "0.01"}],
            calendar_events=[{"days_until": 10, "confidence": 1.0}],
        )

        # Should still compute without errors
        assert result.historical_score == Decimal("0.01")

    def test_string_confidence(self, separator, as_of_date):
        """Should handle string confidence values."""
        result = separator.separate_signals(
            ticker="ACME",
            as_of_date=as_of_date,
            detected_events=[],
            calendar_events=[{"days_until": 30, "confidence": "0.85"}],
        )

        assert result.forward_looking_score > Decimal("0")

    def test_missing_score_field(self, separator, as_of_date):
        """Should handle events missing score field."""
        result = separator.separate_signals(
            ticker="ACME",
            as_of_date=as_of_date,
            detected_events=[{"event_type": "TEST"}],  # No score
            calendar_events=[],
        )

        assert result.historical_score == Decimal("0")


# ============================================================================
# DETERMINISM TESTS
# ============================================================================

class TestDeterminism:
    """Tests verifying deterministic behavior."""

    def test_separate_signals_deterministic(self, separator, as_of_date, sample_detected_events, sample_calendar_events):
        """separate_signals should be deterministic."""
        results = [
            separator.separate_signals(
                ticker="ACME",
                as_of_date=as_of_date,
                detected_events=sample_detected_events,
                calendar_events=sample_calendar_events,
            )
            for _ in range(5)
        ]

        for i in range(1, len(results)):
            assert results[0].historical_score == results[i].historical_score
            assert results[0].forward_looking_score == results[i].forward_looking_score
            assert results[0].blended_score == results[i].blended_score

    def test_batch_separation_deterministic(self, as_of_date, sample_detected_events, sample_calendar_events):
        """separate_batch_signals should be deterministic."""
        detected_by_ticker = {"ACME": sample_detected_events, "BETA": [{"score": "5.0"}]}
        calendar_by_ticker = {"ACME": sample_calendar_events, "BETA": []}

        results = [
            separate_batch_signals(
                detected_events_by_ticker=detected_by_ticker,
                calendar_events_by_ticker=calendar_by_ticker,
                as_of_date=as_of_date,
            )
            for _ in range(5)
        ]

        for i in range(1, len(results)):
            assert results[0].avg_forward_looking_weight == results[i].avg_forward_looking_weight
            assert results[0].tickers_high_risk == results[i].tickers_high_risk


# ============================================================================
# VERSION TESTS
# ============================================================================

class TestVersion:
    """Tests for module version."""

    def test_version_defined(self):
        """Module version should be defined."""
        assert __version__ is not None
        assert isinstance(__version__, str)
