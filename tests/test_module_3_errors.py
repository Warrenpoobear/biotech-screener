#!/usr/bin/env python3
"""
Error handling tests for Module 3: Catalyst Scoring (v2)

Tests edge cases and error scenarios including:
- Invalid date formats
- Boundary conditions for scoring functions
"""

import pytest
from datetime import date, timedelta
from decimal import Decimal

from module_3_scoring_v2 import (
    compute_recency_weight,
    compute_staleness_factor,
    SCORE_MIN,
    SCORE_MAX,
    DECAY_HALF_LIFE_DAYS,
    STALENESS_THRESHOLD_DAYS,
)


class TestComputeRecencyWeightErrors:
    """Tests for compute_recency_weight error handling."""

    @pytest.fixture
    def as_of_date(self):
        return date(2026, 1, 15)

    def test_none_event_date(self, as_of_date):
        """None event date should return default weight."""
        result = compute_recency_weight(None, as_of_date)
        assert result == Decimal("0.5")

    def test_invalid_date_format(self, as_of_date):
        """Invalid date format should return default weight."""
        result = compute_recency_weight("not-a-date", as_of_date)
        assert result == Decimal("0.5")

    def test_invalid_date_format_us_style(self, as_of_date):
        """US date format should return default weight."""
        result = compute_recency_weight("01-15-2026", as_of_date)
        assert result == Decimal("0.5")

    def test_invalid_date_format_partial(self, as_of_date):
        """Partial date should return default weight."""
        result = compute_recency_weight("2026-01", as_of_date)
        assert result == Decimal("0.5")

    def test_future_event_date(self, as_of_date):
        """Future event date should return 0 weight."""
        future = (as_of_date + timedelta(days=30)).isoformat()
        result = compute_recency_weight(future, as_of_date)
        assert result == Decimal("0")

    def test_same_day_event(self, as_of_date):
        """Same day event should return 1.0 weight."""
        result = compute_recency_weight(as_of_date.isoformat(), as_of_date)
        assert result == Decimal("1.0")

    def test_old_event(self, as_of_date):
        """Old event should have low weight."""
        old_date = (as_of_date - timedelta(days=365)).isoformat()
        result = compute_recency_weight(old_date, as_of_date)
        assert result < Decimal("0.5")

    def test_recent_event(self, as_of_date):
        """Recent event should have high weight."""
        recent = (as_of_date - timedelta(days=7)).isoformat()
        result = compute_recency_weight(recent, as_of_date)
        assert result > Decimal("0.9")


class TestComputeStalenessFactor:
    """Tests for compute_staleness_factor error handling."""

    @pytest.fixture
    def as_of_date(self):
        return date(2026, 1, 15)

    def test_empty_events(self, as_of_date):
        """Empty events list should return 1.0."""
        result = compute_staleness_factor([], as_of_date)
        assert result == Decimal("1.0")


class TestScoringConstants:
    """Tests for scoring constant values."""

    def test_score_min_is_zero(self):
        """SCORE_MIN should be 0."""
        assert SCORE_MIN == Decimal("0")

    def test_score_max_is_hundred(self):
        """SCORE_MAX should be 100."""
        assert SCORE_MAX == Decimal("100")

    def test_decay_half_life_positive(self):
        """DECAY_HALF_LIFE_DAYS should be positive."""
        assert DECAY_HALF_LIFE_DAYS > 0

    def test_staleness_threshold_positive(self):
        """STALENESS_THRESHOLD_DAYS should be positive."""
        assert STALENESS_THRESHOLD_DAYS > 0


class TestPITCutoffEnforcement:
    """Tests for PIT cutoff enforcement."""

    @pytest.fixture
    def as_of_date(self):
        return date(2026, 1, 15)

    def test_future_events_excluded_from_historical(self, as_of_date):
        """Events dated after as_of_date should not affect historical calculations."""
        future_date = (as_of_date + timedelta(days=30)).isoformat()
        weight = compute_recency_weight(future_date, as_of_date)
        assert weight == Decimal("0")

    def test_events_exactly_on_cutoff(self, as_of_date):
        """Events exactly on as_of_date should be handled correctly."""
        weight = compute_recency_weight(as_of_date.isoformat(), as_of_date)
        assert weight == Decimal("1.0")


class TestDeterminism:
    """Tests for deterministic behavior."""

    @pytest.fixture
    def as_of_date(self):
        return date(2026, 1, 15)

    def test_same_inputs_same_weight(self, as_of_date):
        """Same inputs should produce same weight."""
        event_date = (as_of_date - timedelta(days=30)).isoformat()
        result1 = compute_recency_weight(event_date, as_of_date)
        result2 = compute_recency_weight(event_date, as_of_date)
        assert result1 == result2
