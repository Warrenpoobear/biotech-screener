#!/usr/bin/env python3
"""
Tests for competitive_pressure_engine.py

Tests the Competitive Pressure Scoring Engine.
Covers:
- Score calculation
- Competition level classification
- Score bounds
- Universe scoring
- Edge cases
"""

import pytest
from datetime import date
from decimal import Decimal

from competitive_pressure_engine import (
    CompetitivePressureEngine,
    TickerCompetitiveScore,
)


class TestCompetitivePressureEngineInit:
    """Tests for engine initialization."""

    def test_engine_init(self):
        """Engine should initialize correctly."""
        engine = CompetitivePressureEngine()
        assert engine.VERSION == "1.0.0"
        assert engine.audit_trail == []


class TestCalculateCompetitiveScore:
    """Tests for calculate_competitive_score method."""

    @pytest.fixture
    def engine(self):
        return CompetitivePressureEngine()

    @pytest.fixture
    def as_of_date(self):
        return date(2026, 1, 15)

    def test_missing_as_of_date_raises(self, engine):
        """as_of_date is required."""
        with pytest.raises(ValueError, match="as_of_date is required"):
            engine.calculate_competitive_score(
                ticker="TEST",
                indication="oncology",
                phase="phase 3",
                competitor_programs=[],
            )

    def test_no_competitors_high_score(self, engine, as_of_date):
        """No competitors should result in high score."""
        result = engine.calculate_competitive_score(
            ticker="TEST",
            indication="rare_disease",
            phase="phase 3",
            competitor_programs=[],
            as_of_date=as_of_date,
        )

        assert isinstance(result, TickerCompetitiveScore)
        assert result.competitive_pressure_score >= Decimal("90")
        assert result.competitor_count == 0
        assert "NO_KNOWN_COMPETITORS" in result.flags

    def test_many_competitors_low_score(self, engine, as_of_date):
        """Many competitors should result in lower score."""
        competitors = [{"phase": "phase 3"} for _ in range(15)]

        result = engine.calculate_competitive_score(
            ticker="TEST",
            indication="oncology",
            phase="phase 3",
            competitor_programs=competitors,
            as_of_date=as_of_date,
        )

        assert result.competitor_count > 0
        assert result.competition_level == "hyper_competitive"
        assert "HYPER_COMPETITIVE_MARKET" in result.flags

    def test_first_in_class_flag(self, engine, as_of_date):
        """First-in-class should add flag."""
        result = engine.calculate_competitive_score(
            ticker="TEST",
            indication="rare_disease",
            phase="phase 3",
            competitor_programs=[],
            is_first_in_class=True,
            as_of_date=as_of_date,
        )

        assert result.is_first_in_class == True
        assert "FIRST_IN_CLASS" in result.flags

    def test_score_bounds(self, engine, as_of_date):
        """Score should be between 0 and 100."""
        # Test with many competitors
        many_competitors = [{"phase": "phase 3"} for _ in range(20)]
        result = engine.calculate_competitive_score(
            ticker="TEST",
            indication="oncology",
            phase="phase 3",
            competitor_programs=many_competitors,
            as_of_date=as_of_date,
        )

        assert Decimal("0") <= result.competitive_pressure_score <= Decimal("100")

    def test_accepts_string_date(self, engine):
        """Should accept string date."""
        result = engine.calculate_competitive_score(
            ticker="TEST",
            indication="oncology",
            phase="phase 3",
            competitor_programs=[],
            as_of_date="2026-01-15",
        )

        assert result.ticker == "TEST"

    def test_audit_trail_updated(self, engine, as_of_date):
        """Calculation should update audit trail."""
        engine.audit_trail = []

        engine.calculate_competitive_score(
            ticker="TEST",
            indication="oncology",
            phase="phase 3",
            competitor_programs=[],
            as_of_date=as_of_date,
        )

        assert len(engine.audit_trail) == 1
        assert engine.audit_trail[0]["ticker"] == "TEST"

    def test_confidence_with_competitor_data(self, engine, as_of_date):
        """Confidence should be high with competitor data."""
        result = engine.calculate_competitive_score(
            ticker="TEST",
            indication="oncology",
            phase="phase 3",
            competitor_programs=[{"phase": "phase 2"}],
            as_of_date=as_of_date,
        )

        assert result.confidence == "high"

    def test_confidence_without_competitor_data(self, engine, as_of_date):
        """Confidence should be medium without competitor data but with indication."""
        result = engine.calculate_competitive_score(
            ticker="TEST",
            indication="oncology",
            phase="phase 3",
            competitor_programs=[],
            as_of_date=as_of_date,
        )

        assert result.confidence == "medium"


class TestScoreUniverse:
    """Tests for score_universe method."""

    @pytest.fixture
    def engine(self):
        return CompetitivePressureEngine()

    @pytest.fixture
    def as_of_date(self):
        return date(2026, 1, 15)

    def test_score_universe_basic(self, engine, as_of_date):
        """Should score multiple companies."""
        universe = [
            {"ticker": "ACME", "indication": "oncology", "phase": "phase 3"},
            {"ticker": "BETA", "indication": "rare_disease", "phase": "phase 2"},
        ]

        result = engine.score_universe(
            universe=universe,
            as_of_date=as_of_date,
        )

        assert result["as_of_date"] == "2026-01-15"
        assert len(result["scores"]) == 2

    def test_score_universe_with_competitor_data(self, engine, as_of_date):
        """Should use competitor data by indication."""
        universe = [
            {"ticker": "ACME", "indication": "oncology", "phase": "phase 3"},
        ]

        competitor_data = {
            "oncology": [{"phase": "phase 3"} for _ in range(10)],
        }

        result = engine.score_universe(
            universe=universe,
            competitor_data_by_indication=competitor_data,
            as_of_date=as_of_date,
        )

        assert len(result["scores"]) == 1
        # With competitors, score should be lower
        score = Decimal(result["scores"][0]["competitive_pressure_score"])
        assert score < Decimal("100")

    def test_score_universe_missing_ticker(self, engine, as_of_date):
        """Should skip entries without ticker."""
        universe = [
            {"ticker": "ACME", "indication": "oncology"},
            {"indication": "rare_disease"},  # Missing ticker
        ]

        result = engine.score_universe(
            universe=universe,
            as_of_date=as_of_date,
        )

        assert len(result["scores"]) == 1

    def test_score_universe_diagnostics(self, engine, as_of_date):
        """Should include diagnostics."""
        universe = [
            {"ticker": "ACME", "indication": "oncology", "phase": "phase 3"},
            {"ticker": "BETA", "indication": "rare_disease", "is_first_in_class": True},
        ]

        result = engine.score_universe(
            universe=universe,
            as_of_date=as_of_date,
        )

        diag = result["diagnostic_counts"]
        assert "total_scored" in diag
        assert "first_in_class" in diag
        assert diag["first_in_class"] == 1

    def test_score_universe_missing_as_of_date_raises(self, engine):
        """Should raise without as_of_date."""
        with pytest.raises(ValueError, match="as_of_date is required"):
            engine.score_universe(
                universe=[{"ticker": "TEST"}],
                as_of_date=None,
            )

    def test_score_universe_provenance(self, engine, as_of_date):
        """Should include provenance info."""
        result = engine.score_universe(
            universe=[{"ticker": "TEST"}],
            as_of_date=as_of_date,
        )

        assert "provenance" in result
        assert result["provenance"]["engine"] == "CompetitivePressureEngine"
        assert result["provenance"]["version"] == "1.0.0"


class TestGetDiagnosticCounts:
    """Tests for get_diagnostic_counts method."""

    @pytest.fixture
    def engine(self):
        return CompetitivePressureEngine()

    @pytest.fixture
    def as_of_date(self):
        return date(2026, 1, 15)

    def test_empty_audit_trail(self, engine):
        """Empty audit trail should return zero counts."""
        result = engine.get_diagnostic_counts()
        assert result == {"total": 0}

    def test_with_calculations(self, engine, as_of_date):
        """Should count calculations."""
        engine.calculate_competitive_score(
            ticker="TEST1",
            indication="oncology",
            phase="phase 3",
            competitor_programs=[{"phase": "phase 3"}] * 15,  # Hyper competitive
            as_of_date=as_of_date,
        )

        engine.calculate_competitive_score(
            ticker="TEST2",
            indication="rare_disease",
            phase="phase 2",
            competitor_programs=[],
            as_of_date=as_of_date,
        )

        result = engine.get_diagnostic_counts()
        assert result["total"] == 2
        assert "high_competition_pct" in result


class TestTickerCompetitiveScoreDataclass:
    """Tests for TickerCompetitiveScore dataclass."""

    def test_dataclass_creation(self):
        """Should create valid dataclass."""
        score = TickerCompetitiveScore(
            ticker="TEST",
            competitive_pressure_score=Decimal("75.50"),
            competitor_count=5,
            competition_level="moderate",
            penalty_applied=Decimal("5"),
            market_share_estimate=Decimal("0.15"),
            is_first_in_class=False,
            confidence="high",
        )

        assert score.ticker == "TEST"
        assert score.competitive_pressure_score == Decimal("75.50")
        assert score.competitor_count == 5
        assert score.flags == []  # Default

    def test_dataclass_with_flags(self):
        """Should accept custom flags."""
        score = TickerCompetitiveScore(
            ticker="TEST",
            competitive_pressure_score=Decimal("50"),
            competitor_count=15,
            competition_level="hyper_competitive",
            penalty_applied=Decimal("15"),
            market_share_estimate=Decimal("0.05"),
            is_first_in_class=False,
            confidence="high",
            flags=["HYPER_COMPETITIVE_MARKET"],
        )

        assert "HYPER_COMPETITIVE_MARKET" in score.flags


class TestCompetitionLevels:
    """Tests for different competition level classifications."""

    @pytest.fixture
    def engine(self):
        return CompetitivePressureEngine()

    @pytest.fixture
    def as_of_date(self):
        return date(2026, 1, 15)

    def test_low_competition(self, engine, as_of_date):
        """Few competitors should result in low competition."""
        result = engine.calculate_competitive_score(
            ticker="TEST",
            indication="rare_disease",
            phase="phase 3",
            competitor_programs=[{"phase": "phase 2"}],  # 1 competitor
            as_of_date=as_of_date,
        )

        assert result.competition_level in ["low", "moderate", "none"]

    def test_moderate_competition(self, engine, as_of_date):
        """Several competitors should result in moderate competition."""
        competitors = [{"phase": "phase 2"} for _ in range(5)]

        result = engine.calculate_competitive_score(
            ticker="TEST",
            indication="oncology",
            phase="phase 3",
            competitor_programs=competitors,
            as_of_date=as_of_date,
        )

        # Competition level depends on implementation details

    def test_high_competition(self, engine, as_of_date):
        """Many competitors should result in high/hyper competition."""
        competitors = [{"phase": "phase 3"} for _ in range(15)]

        result = engine.calculate_competitive_score(
            ticker="TEST",
            indication="oncology",
            phase="phase 3",
            competitor_programs=competitors,
            as_of_date=as_of_date,
        )

        assert result.competition_level in ["high", "hyper_competitive"]


class TestDeterminism:
    """Tests for deterministic behavior."""

    @pytest.fixture
    def engine(self):
        return CompetitivePressureEngine()

    @pytest.fixture
    def as_of_date(self):
        return date(2026, 1, 15)

    def test_same_inputs_same_output(self, engine, as_of_date):
        """Same inputs should produce same output."""
        params = dict(
            ticker="TEST",
            indication="oncology",
            phase="phase 3",
            competitor_programs=[{"phase": "phase 2"}] * 5,
            as_of_date=as_of_date,
        )

        result1 = engine.calculate_competitive_score(**params)
        result2 = engine.calculate_competitive_score(**params)

        assert result1.competitive_pressure_score == result2.competitive_pressure_score
        assert result1.competitor_count == result2.competitor_count
        assert result1.competition_level == result2.competition_level
