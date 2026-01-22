#!/usr/bin/env python3
"""
Tests for timeline_slippage_engine.py

Timeline slippage detection identifies execution risk from trial delays.
These tests cover:
- Pushout detection (delays)
- Pullin detection (accelerations)
- Severity classification
- Score calculation
- Repeat offender detection
- Universe scoring
"""

import pytest
from datetime import date
from decimal import Decimal

from timeline_slippage_engine import (
    TimelineSlippageEngine,
    SlippageDirection,
    SlippageSeverity,
    TickerSlippageScore,
    SlippageResult,
)


class TestSlippageDirectionDetection:
    """Tests for slippage direction detection."""

    def test_pushout_detection(self):
        """Delay should be detected as pushout."""
        engine = TimelineSlippageEngine()
        current = [{"nct_id": "NCT001", "primary_completion_date": "2026-09-01"}]
        prior = [{"nct_id": "NCT001", "primary_completion_date": "2026-03-01"}]

        result = engine.calculate_slippage_score(
            ticker="ACME",
            current_trials=current,
            prior_trials=prior,
            as_of_date=date(2026, 1, 15),
        )

        assert result.trial_results[0].direction == SlippageDirection.PUSHOUT
        assert result.trial_results[0].days_slipped == 184

    def test_pullin_detection(self):
        """Acceleration should be detected as pullin."""
        engine = TimelineSlippageEngine()
        current = [{"nct_id": "NCT001", "primary_completion_date": "2026-03-01"}]
        prior = [{"nct_id": "NCT001", "primary_completion_date": "2026-09-01"}]

        result = engine.calculate_slippage_score(
            ticker="ACME",
            current_trials=current,
            prior_trials=prior,
            as_of_date=date(2026, 1, 15),
        )

        assert result.trial_results[0].direction == SlippageDirection.PULLIN
        assert "TIMELINE_ACCELERATED" in result.trial_results[0].flags

    def test_stable_detection(self):
        """Small changes (<30 days) should be stable."""
        engine = TimelineSlippageEngine()
        current = [{"nct_id": "NCT001", "primary_completion_date": "2026-06-15"}]
        prior = [{"nct_id": "NCT001", "primary_completion_date": "2026-06-01"}]

        result = engine.calculate_slippage_score(
            ticker="ACME",
            current_trials=current,
            prior_trials=prior,
            as_of_date=date(2026, 1, 15),
        )

        assert result.trial_results[0].direction == SlippageDirection.STABLE

    def test_unknown_with_missing_prior(self):
        """Missing prior data should result in unknown direction."""
        engine = TimelineSlippageEngine()
        current = [{"nct_id": "NCT001", "primary_completion_date": "2026-06-01"}]
        prior = []  # No prior data

        result = engine.calculate_slippage_score(
            ticker="ACME",
            current_trials=current,
            prior_trials=prior,
            as_of_date=date(2026, 1, 15),
        )

        assert result.trial_results[0].direction == SlippageDirection.UNKNOWN
        assert "MISSING_PRIOR_DATE" in result.trial_results[0].flags


class TestSeverityClassification:
    """Tests for slippage severity classification."""

    def test_severe_pushout(self):
        """Pushout >180 days should be severe."""
        engine = TimelineSlippageEngine()
        current = [{"nct_id": "NCT001", "primary_completion_date": "2027-01-01"}]
        prior = [{"nct_id": "NCT001", "primary_completion_date": "2026-01-01"}]

        result = engine.calculate_slippage_score(
            ticker="ACME",
            current_trials=current,
            prior_trials=prior,
            as_of_date=date(2026, 1, 15),
        )

        assert result.trial_results[0].severity == SlippageSeverity.SEVERE
        assert "SEVERE_PUSHOUT" in result.trial_results[0].flags

    def test_moderate_pushout(self):
        """Pushout 60-180 days should be moderate."""
        engine = TimelineSlippageEngine()
        current = [{"nct_id": "NCT001", "primary_completion_date": "2026-05-01"}]
        prior = [{"nct_id": "NCT001", "primary_completion_date": "2026-02-01"}]  # ~90 days

        result = engine.calculate_slippage_score(
            ticker="ACME",
            current_trials=current,
            prior_trials=prior,
            as_of_date=date(2026, 1, 15),
        )

        assert result.trial_results[0].severity == SlippageSeverity.MODERATE
        assert "MODERATE_PUSHOUT" in result.trial_results[0].flags

    def test_minor_pushout(self):
        """Pushout 30-60 days should be minor."""
        engine = TimelineSlippageEngine()
        current = [{"nct_id": "NCT001", "primary_completion_date": "2026-04-15"}]
        prior = [{"nct_id": "NCT001", "primary_completion_date": "2026-03-01"}]  # ~45 days

        result = engine.calculate_slippage_score(
            ticker="ACME",
            current_trials=current,
            prior_trials=prior,
            as_of_date=date(2026, 1, 15),
        )

        assert result.trial_results[0].severity == SlippageSeverity.MINOR
        assert "MINOR_PUSHOUT" in result.trial_results[0].flags

    def test_accelerated_severity(self):
        """Pullin should be marked as accelerated."""
        engine = TimelineSlippageEngine()
        current = [{"nct_id": "NCT001", "primary_completion_date": "2026-03-01"}]
        prior = [{"nct_id": "NCT001", "primary_completion_date": "2026-06-01"}]

        result = engine.calculate_slippage_score(
            ticker="ACME",
            current_trials=current,
            prior_trials=prior,
            as_of_date=date(2026, 1, 15),
        )

        assert result.trial_results[0].severity == SlippageSeverity.ACCELERATED


class TestScoreCalculation:
    """Tests for slippage score calculation."""

    def test_neutral_baseline_score(self):
        """Neutral/stable trials should score around 50."""
        engine = TimelineSlippageEngine()
        current = [{"nct_id": "NCT001", "primary_completion_date": "2026-06-01"}]
        prior = [{"nct_id": "NCT001", "primary_completion_date": "2026-06-10"}]

        result = engine.calculate_slippage_score(
            ticker="ACME",
            current_trials=current,
            prior_trials=prior,
            as_of_date=date(2026, 1, 15),
        )

        assert result.trial_results[0].slippage_score == Decimal("50.00")

    def test_severe_pushout_penalty(self):
        """Severe pushout should reduce score by 20."""
        engine = TimelineSlippageEngine()
        current = [{"nct_id": "NCT001", "primary_completion_date": "2027-01-01"}]
        prior = [{"nct_id": "NCT001", "primary_completion_date": "2026-01-01"}]

        result = engine.calculate_slippage_score(
            ticker="ACME",
            current_trials=current,
            prior_trials=prior,
            as_of_date=date(2026, 1, 15),
        )

        assert result.trial_results[0].slippage_score == Decimal("30.00")  # 50 - 20

    def test_pullin_bonus(self):
        """Pullin should add 5 to score."""
        engine = TimelineSlippageEngine()
        current = [{"nct_id": "NCT001", "primary_completion_date": "2026-03-01"}]
        prior = [{"nct_id": "NCT001", "primary_completion_date": "2026-06-01"}]

        result = engine.calculate_slippage_score(
            ticker="ACME",
            current_trials=current,
            prior_trials=prior,
            as_of_date=date(2026, 1, 15),
        )

        assert result.trial_results[0].slippage_score == Decimal("55.00")  # 50 + 5

    def test_score_bounded_0_100(self):
        """Score should always be between 0 and 100."""
        engine = TimelineSlippageEngine()
        # Create multiple severe pushouts
        current = [
            {"nct_id": f"NCT00{i}", "primary_completion_date": "2028-01-01"}
            for i in range(5)
        ]
        prior = [
            {"nct_id": f"NCT00{i}", "primary_completion_date": "2026-01-01"}
            for i in range(5)
        ]

        result = engine.calculate_slippage_score(
            ticker="ACME",
            current_trials=current,
            prior_trials=prior,
            as_of_date=date(2026, 1, 15),
        )

        assert result.execution_risk_score >= Decimal("0")
        assert result.execution_risk_score <= Decimal("100")


class TestRepeatOffenderDetection:
    """Tests for repeat offender pattern detection."""

    def test_repeat_offender_detected(self):
        """Multiple severe/moderate pushouts should flag repeat offender."""
        engine = TimelineSlippageEngine()
        current = [
            {"nct_id": "NCT001", "primary_completion_date": "2027-01-01"},  # Severe
            {"nct_id": "NCT002", "primary_completion_date": "2027-06-01"},  # Severe
        ]
        prior = [
            {"nct_id": "NCT001", "primary_completion_date": "2026-01-01"},
            {"nct_id": "NCT002", "primary_completion_date": "2026-06-01"},
        ]

        result = engine.calculate_slippage_score(
            ticker="ACME",
            current_trials=current,
            prior_trials=prior,
            as_of_date=date(2026, 1, 15),
        )

        assert result.repeat_offender is True
        assert "REPEAT_SLIPPAGE_OFFENDER" in result.flags

    def test_single_pushout_not_repeat_offender(self):
        """Single pushout should not flag repeat offender."""
        engine = TimelineSlippageEngine()
        current = [
            {"nct_id": "NCT001", "primary_completion_date": "2027-01-01"},  # Severe
            {"nct_id": "NCT002", "primary_completion_date": "2026-06-10"},  # Stable
        ]
        prior = [
            {"nct_id": "NCT001", "primary_completion_date": "2026-01-01"},
            {"nct_id": "NCT002", "primary_completion_date": "2026-06-01"},
        ]

        result = engine.calculate_slippage_score(
            ticker="ACME",
            current_trials=current,
            prior_trials=prior,
            as_of_date=date(2026, 1, 15),
        )

        assert result.repeat_offender is False

    def test_repeat_offender_penalty(self):
        """Repeat offender should receive additional penalty."""
        engine = TimelineSlippageEngine()
        current = [
            {"nct_id": "NCT001", "primary_completion_date": "2027-01-01"},
            {"nct_id": "NCT002", "primary_completion_date": "2027-06-01"},
        ]
        prior = [
            {"nct_id": "NCT001", "primary_completion_date": "2026-01-01"},
            {"nct_id": "NCT002", "primary_completion_date": "2026-06-01"},
        ]

        result = engine.calculate_slippage_score(
            ticker="ACME",
            current_trials=current,
            prior_trials=prior,
            as_of_date=date(2026, 1, 15),
        )

        # Without repeat penalty: avg of two scores at 30 = 30
        # With repeat penalty: 30 - 15 = 15
        assert result.execution_risk_score == Decimal("15.00")


class TestAggregatedResults:
    """Tests for aggregated ticker results."""

    def test_counts_calculated(self):
        """Direction counts should be calculated correctly."""
        engine = TimelineSlippageEngine()
        current = [
            {"nct_id": "NCT001", "primary_completion_date": "2027-01-01"},  # Pushout
            {"nct_id": "NCT002", "primary_completion_date": "2026-03-01"},  # Pullin
            {"nct_id": "NCT003", "primary_completion_date": "2026-06-05"},  # Stable
        ]
        prior = [
            {"nct_id": "NCT001", "primary_completion_date": "2026-01-01"},
            {"nct_id": "NCT002", "primary_completion_date": "2026-09-01"},
            {"nct_id": "NCT003", "primary_completion_date": "2026-06-01"},
        ]

        result = engine.calculate_slippage_score(
            ticker="ACME",
            current_trials=current,
            prior_trials=prior,
            as_of_date=date(2026, 1, 15),
        )

        assert result.pushout_count == 1
        assert result.pullin_count == 1
        assert result.stable_count == 1

    def test_average_slippage_days(self):
        """Average slippage should be calculated correctly."""
        engine = TimelineSlippageEngine()
        current = [
            {"nct_id": "NCT001", "primary_completion_date": "2026-09-01"},  # ~180 days pushout
            {"nct_id": "NCT002", "primary_completion_date": "2026-04-01"},  # ~60 days pushout
        ]
        prior = [
            {"nct_id": "NCT001", "primary_completion_date": "2026-03-01"},
            {"nct_id": "NCT002", "primary_completion_date": "2026-02-01"},
        ]

        result = engine.calculate_slippage_score(
            ticker="ACME",
            current_trials=current,
            prior_trials=prior,
            as_of_date=date(2026, 1, 15),
        )

        # Total pushout days / total trials
        assert result.avg_slippage_days > Decimal("0")

    def test_no_trials_returns_neutral(self):
        """No trials should return neutral score."""
        engine = TimelineSlippageEngine()
        result = engine.calculate_slippage_score(
            ticker="ACME",
            current_trials=[],
            prior_trials=[],
            as_of_date=date(2026, 1, 15),
        )

        assert result.execution_risk_score == Decimal("50")
        assert result.confidence == "low"
        assert "NO_TRIALS" in result.flags


class TestConfidenceLevel:
    """Tests for confidence level determination."""

    def test_high_confidence_many_trials(self):
        """3+ trials should yield high confidence."""
        engine = TimelineSlippageEngine()
        current = [
            {"nct_id": f"NCT00{i}", "primary_completion_date": "2026-06-01"}
            for i in range(3)
        ]
        prior = [
            {"nct_id": f"NCT00{i}", "primary_completion_date": "2026-05-01"}
            for i in range(3)
        ]

        result = engine.calculate_slippage_score(
            ticker="ACME",
            current_trials=current,
            prior_trials=prior,
            as_of_date=date(2026, 1, 15),
        )

        assert result.confidence == "high"

    def test_medium_confidence_two_trials(self):
        """2 trials should yield medium confidence."""
        engine = TimelineSlippageEngine()
        current = [
            {"nct_id": "NCT001", "primary_completion_date": "2026-06-01"},
            {"nct_id": "NCT002", "primary_completion_date": "2026-07-01"},
        ]
        prior = [
            {"nct_id": "NCT001", "primary_completion_date": "2026-05-01"},
            {"nct_id": "NCT002", "primary_completion_date": "2026-06-01"},
        ]

        result = engine.calculate_slippage_score(
            ticker="ACME",
            current_trials=current,
            prior_trials=prior,
            as_of_date=date(2026, 1, 15),
        )

        assert result.confidence == "medium"

    def test_low_confidence_one_trial(self):
        """1 trial should yield low confidence."""
        engine = TimelineSlippageEngine()
        current = [{"nct_id": "NCT001", "primary_completion_date": "2026-06-01"}]
        prior = [{"nct_id": "NCT001", "primary_completion_date": "2026-05-01"}]

        result = engine.calculate_slippage_score(
            ticker="ACME",
            current_trials=current,
            prior_trials=prior,
            as_of_date=date(2026, 1, 15),
        )

        assert result.confidence == "low"


class TestScoreUniverse:
    """Tests for universe scoring."""

    def test_scores_multiple_tickers(self):
        """Should score multiple tickers."""
        engine = TimelineSlippageEngine()
        universe = [
            {"ticker": "ACME"},
            {"ticker": "BETA"},
        ]
        current_by_ticker = {
            "ACME": [{"nct_id": "NCT001", "primary_completion_date": "2026-09-01"}],
            "BETA": [{"nct_id": "NCT002", "primary_completion_date": "2026-06-01"}],
        }
        prior_by_ticker = {
            "ACME": [{"nct_id": "NCT001", "primary_completion_date": "2026-03-01"}],
            "BETA": [{"nct_id": "NCT002", "primary_completion_date": "2026-05-15"}],
        }

        result = engine.score_universe(
            universe=universe,
            current_trials_by_ticker=current_by_ticker,
            prior_trials_by_ticker=prior_by_ticker,
            as_of_date=date(2026, 1, 15),
        )

        assert result["diagnostic_counts"]["total_scored"] == 2
        assert len(result["scores"]) == 2

    def test_missing_ticker_trials_handled(self):
        """Missing trials for ticker should be handled gracefully."""
        engine = TimelineSlippageEngine()
        universe = [{"ticker": "ACME"}, {"ticker": "MISSING"}]
        current_by_ticker = {
            "ACME": [{"nct_id": "NCT001", "primary_completion_date": "2026-06-01"}],
            # MISSING has no trials
        }

        result = engine.score_universe(
            universe=universe,
            current_trials_by_ticker=current_by_ticker,
            as_of_date=date(2026, 1, 15),
        )

        assert result["diagnostic_counts"]["total_scored"] == 2
        # MISSING should have NO_TRIALS flag
        missing_score = [s for s in result["scores"] if s["ticker"] == "MISSING"][0]
        assert "NO_TRIALS" in missing_score["flags"]


class TestDateParsing:
    """Tests for date parsing."""

    def test_iso_string_parsing(self):
        """Should parse ISO date strings."""
        engine = TimelineSlippageEngine()
        current = [{"nct_id": "NCT001", "primary_completion_date": "2026-06-01"}]
        prior = [{"nct_id": "NCT001", "primary_completion_date": "2026-03-01"}]

        result = engine.calculate_slippage_score(
            ticker="ACME",
            current_trials=current,
            prior_trials=prior,
            as_of_date="2026-01-15",  # String date
        )

        assert result.trial_results[0].current_date == "2026-06-01"
        assert result.trial_results[0].prior_date == "2026-03-01"

    def test_date_object_handling(self):
        """Should handle date objects directly."""
        engine = TimelineSlippageEngine()
        current = [{"nct_id": "NCT001", "primary_completion_date": date(2026, 6, 1)}]
        prior = [{"nct_id": "NCT001", "primary_completion_date": date(2026, 3, 1)}]

        result = engine.calculate_slippage_score(
            ticker="ACME",
            current_trials=current,
            prior_trials=prior,
            as_of_date=date(2026, 1, 15),
        )

        assert result.trial_results[0].current_date == "2026-06-01"

    def test_completion_date_fallback(self):
        """Should fallback to completion_date if primary_completion_date missing."""
        engine = TimelineSlippageEngine()
        current = [{"nct_id": "NCT001", "completion_date": "2026-06-01"}]
        prior = [{"nct_id": "NCT001", "completion_date": "2026-03-01"}]

        result = engine.calculate_slippage_score(
            ticker="ACME",
            current_trials=current,
            prior_trials=prior,
            as_of_date=date(2026, 1, 15),
        )

        assert result.trial_results[0].current_date == "2026-06-01"


class TestRequiredAsOfDate:
    """Tests for required as_of_date parameter."""

    def test_raises_without_as_of_date(self):
        """Should raise if as_of_date not provided."""
        engine = TimelineSlippageEngine()
        with pytest.raises(ValueError) as exc_info:
            engine.calculate_slippage_score(
                ticker="ACME",
                current_trials=[],
                as_of_date=None,
            )
        assert "as_of_date is required" in str(exc_info.value)


class TestAuditTrail:
    """Tests for audit trail."""

    def test_audit_recorded(self):
        """Calculations should be recorded in audit trail."""
        engine = TimelineSlippageEngine()
        engine.calculate_slippage_score(
            ticker="ACME",
            current_trials=[{"nct_id": "NCT001", "primary_completion_date": "2026-06-01"}],
            as_of_date=date(2026, 1, 15),
        )

        assert len(engine.audit_trail) == 1
        assert engine.audit_trail[0]["ticker"] == "ACME"

    def test_diagnostic_counts(self):
        """Diagnostic counts should be available."""
        engine = TimelineSlippageEngine()
        engine.calculate_slippage_score(
            ticker="ACME",
            current_trials=[{"nct_id": "NCT001", "primary_completion_date": "2027-01-01"}],
            prior_trials=[{"nct_id": "NCT001", "primary_completion_date": "2026-01-01"}],
            as_of_date=date(2026, 1, 15),
        )

        counts = engine.get_diagnostic_counts()
        assert counts["total"] == 1
        assert counts["with_pushouts"] == 1
