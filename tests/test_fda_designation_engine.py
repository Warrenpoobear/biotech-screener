#!/usr/bin/env python3
"""
Tests for FDA Designation Engine
"""

import pytest
from decimal import Decimal
from datetime import date

from fda_designation_engine import (
    FDADesignationEngine,
    DesignationType,
    generate_sample_designations,
)


class TestFDADesignationEngine:
    """Tests for FDADesignationEngine."""

    @pytest.fixture
    def engine(self):
        return FDADesignationEngine()

    @pytest.fixture
    def engine_with_data(self):
        engine = FDADesignationEngine()
        engine.load_designations(generate_sample_designations())
        return engine

    def test_initialization(self, engine):
        """Engine initializes with empty state."""
        assert engine.designations == {}
        assert engine.audit_trail == []

    def test_load_designations(self, engine):
        """Load designations from sample data."""
        data = generate_sample_designations()
        loaded = engine.load_designations(data)
        assert loaded > 0
        assert len(engine.designations) > 0

    def test_score_ticker_with_designations(self, engine_with_data):
        """Score ticker that has designations."""
        as_of = date(2026, 1, 26)
        result = engine_with_data.score_ticker("MRNA", as_of)

        assert result["ticker"] == "MRNA"
        assert result["designation_count"] > 0
        assert result["designation_score"] > Decimal("50")  # Above neutral
        assert result["pos_multiplier"] > Decimal("1.0")
        assert "BTD" in result["designation_types"] or "FT" in result["designation_types"]

    def test_score_ticker_without_designations(self, engine_with_data):
        """Score ticker without designations."""
        as_of = date(2026, 1, 26)
        result = engine_with_data.score_ticker("UNKNOWN", as_of)

        assert result["ticker"] == "UNKNOWN"
        assert result["designation_count"] == 0
        assert result["designation_score"] == Decimal("50")  # Neutral
        assert result["pos_multiplier"] == Decimal("1.0")

    def test_pos_adjustment(self, engine_with_data):
        """Verify PoS adjustment calculation."""
        as_of = date(2026, 1, 26)
        base_pos = Decimal("50")

        result = engine_with_data.score_ticker("MRNA", as_of, base_pos=base_pos)

        assert result["adjusted_pos"] is not None
        assert result["adjusted_pos"] > base_pos  # Should be boosted
        assert result["adjusted_pos"] <= Decimal("100")  # Capped at 100

    def test_breakthrough_therapy_boost(self, engine):
        """Breakthrough therapy should provide significant boost."""
        engine.load_designations([
            {"ticker": "TEST", "designation_type": "BTD", "indication": "oncology",
             "grant_date": "2025-01-01", "source": "fda", "confidence": "confirmed"},
        ])

        as_of = date(2026, 1, 26)
        result = engine.score_ticker("TEST", as_of, base_pos=Decimal("50"))

        # BTD has 1.25 multiplier
        assert result["pos_multiplier"] >= Decimal("1.20")
        assert result["designation_score"] >= Decimal("60")

    def test_multiple_designations_diminishing_returns(self, engine):
        """Multiple designations should have diminishing returns."""
        engine.load_designations([
            {"ticker": "MULTI", "designation_type": "BTD", "indication": "oncology",
             "grant_date": "2025-01-01", "source": "fda", "confidence": "confirmed"},
            {"ticker": "MULTI", "designation_type": "ODD", "indication": "rare_disease",
             "grant_date": "2025-02-01", "source": "fda", "confidence": "confirmed"},
            {"ticker": "MULTI", "designation_type": "FT", "indication": "oncology",
             "grant_date": "2025-03-01", "source": "fda", "confidence": "confirmed"},
        ])

        as_of = date(2026, 1, 26)
        result = engine.score_ticker("MULTI", as_of)

        # Should have bonus but capped
        assert result["designation_count"] == 3
        assert result["pos_multiplier"] <= Decimal("1.50")  # Capped
        assert result["designation_score"] <= Decimal("100")

    def test_timeline_acceleration(self, engine_with_data):
        """Timeline acceleration calculated correctly."""
        as_of = date(2026, 1, 26)
        result = engine_with_data.score_ticker("MRNA", as_of)

        # MRNA has BTD and FT
        assert result["timeline_acceleration_months"] > 0
        assert result["timeline_acceleration_months"] <= 18  # Capped

    def test_score_universe(self, engine_with_data):
        """Score entire universe."""
        universe = [
            {"ticker": "MRNA"},
            {"ticker": "BEAM"},
            {"ticker": "UNKNOWN"},
        ]
        as_of = date(2026, 1, 26)

        result = engine_with_data.score_universe(universe, as_of)

        assert result["diagnostic_counts"]["total_scored"] == 3
        assert result["diagnostic_counts"]["with_designations"] >= 2
        assert "provenance" in result

    def test_date_filtering(self, engine):
        """Designations after as_of_date should be excluded."""
        engine.load_designations([
            {"ticker": "FUTURE", "designation_type": "BTD", "indication": "oncology",
             "grant_date": "2027-01-01", "source": "fda", "confidence": "confirmed"},
        ])

        as_of = date(2026, 1, 26)
        result = engine.score_ticker("FUTURE", as_of)

        # Future designation should be excluded
        assert result["designation_count"] == 0

    def test_confidence_multiplier(self, engine):
        """Confidence level affects multiplier."""
        engine.load_designations([
            {"ticker": "CONFIRMED", "designation_type": "BTD", "indication": "oncology",
             "grant_date": "2025-01-01", "source": "fda", "confidence": "confirmed"},
        ])
        engine.load_designations([
            {"ticker": "INFERRED", "designation_type": "BTD", "indication": "oncology",
             "grant_date": "2025-01-01", "source": "inferred", "confidence": "inferred"},
        ])

        as_of = date(2026, 1, 26)
        confirmed = engine.score_ticker("CONFIRMED", as_of)
        inferred = engine.score_ticker("INFERRED", as_of)

        # Confirmed should have higher multiplier
        assert confirmed["pos_multiplier"] > inferred["pos_multiplier"]

    def test_audit_trail(self, engine_with_data):
        """Audit trail is maintained."""
        as_of = date(2026, 1, 26)
        engine_with_data.score_ticker("MRNA", as_of)

        trail = engine_with_data.get_audit_trail()
        assert len(trail) > 0
        assert trail[0]["ticker"] == "MRNA"

        engine_with_data.clear_audit_trail()
        assert len(engine_with_data.get_audit_trail()) == 0


class TestDesignationTypes:
    """Tests for designation type handling."""

    @pytest.fixture
    def engine(self):
        return FDADesignationEngine()

    def test_all_designation_types_have_factors(self):
        """All designation types have PoS adjustment factors."""
        for des_type in DesignationType:
            assert des_type in FDADesignationEngine.POS_ADJUSTMENT_FACTORS

    def test_all_designation_types_have_timeline(self):
        """All designation types have timeline acceleration values."""
        for des_type in DesignationType:
            assert des_type in FDADesignationEngine.TIMELINE_ACCELERATION

    def test_invalid_designation_type_skipped(self, engine):
        """Invalid designation types are skipped during load."""
        loaded = engine.load_designations([
            {"ticker": "TEST", "designation_type": "INVALID", "indication": "oncology"},
        ])
        assert loaded == 0
