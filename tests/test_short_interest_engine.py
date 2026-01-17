#!/usr/bin/env python3
"""
Unit tests for short_interest_engine.py - Short Interest Signal Engine.

Tests:
1. Squeeze potential assessment (EXTREME, HIGH, MODERATE, LOW)
2. Crowding risk classification
3. Trend contribution calculation
4. Signal direction determination
5. Insufficient data handling
6. Universe scoring with content hash determinism

Run: pytest tests/test_short_interest_engine.py -v
"""

import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from short_interest_engine import ShortInterestSignalEngine


# ============================================================================
# ENGINE INITIALIZATION TESTS
# ============================================================================

class TestEngineInitialization:
    """Tests for engine initialization."""

    def test_engine_creates_successfully(self):
        """Engine should initialize with default values."""
        engine = ShortInterestSignalEngine()

        assert engine.VERSION == "1.0.0"
        assert engine.audit_trail == []

    def test_squeeze_thresholds_defined(self):
        """Squeeze thresholds should be properly defined."""
        engine = ShortInterestSignalEngine()

        assert "extreme" in engine.SQUEEZE_THRESHOLDS
        assert "high" in engine.SQUEEZE_THRESHOLDS
        assert "moderate" in engine.SQUEEZE_THRESHOLDS


# ============================================================================
# SQUEEZE POTENTIAL TESTS
# ============================================================================

class TestSqueezePotenial:
    """Tests for squeeze potential assessment."""

    @pytest.fixture
    def engine(self):
        return ShortInterestSignalEngine()

    def test_extreme_squeeze_potential(self, engine):
        """High SI% and high DTC should trigger EXTREME."""
        result = engine.calculate_short_signal(
            ticker="SQUEEZE",
            short_interest_pct=Decimal("45"),
            days_to_cover=Decimal("12"),
            as_of_date=date(2026, 1, 15)
        )

        assert result["squeeze_potential"] == "EXTREME"

    def test_high_squeeze_potential(self, engine):
        """Moderate-high SI% and DTC should trigger HIGH."""
        result = engine.calculate_short_signal(
            ticker="HIGH",
            short_interest_pct=Decimal("25"),
            days_to_cover=Decimal("8"),
            as_of_date=date(2026, 1, 15)
        )

        assert result["squeeze_potential"] == "HIGH"

    def test_moderate_squeeze_potential(self, engine):
        """Moderate SI% and DTC should trigger MODERATE."""
        result = engine.calculate_short_signal(
            ticker="MOD",
            short_interest_pct=Decimal("12"),
            days_to_cover=Decimal("6"),
            as_of_date=date(2026, 1, 15)
        )

        assert result["squeeze_potential"] == "MODERATE"

    def test_low_squeeze_potential(self, engine):
        """Low SI% or DTC should trigger LOW."""
        result = engine.calculate_short_signal(
            ticker="LOW",
            short_interest_pct=Decimal("5"),
            days_to_cover=Decimal("2"),
            as_of_date=date(2026, 1, 15)
        )

        assert result["squeeze_potential"] == "LOW"

    def test_squeeze_requires_both_conditions(self, engine):
        """EXTREME requires both high SI% AND high DTC."""
        # High SI but low DTC
        result1 = engine.calculate_short_signal(
            ticker="TEST1",
            short_interest_pct=Decimal("50"),
            days_to_cover=Decimal("2"),  # Low DTC
            as_of_date=date(2026, 1, 15)
        )
        assert result1["squeeze_potential"] != "EXTREME"

        # Low SI but high DTC
        result2 = engine.calculate_short_signal(
            ticker="TEST2",
            short_interest_pct=Decimal("5"),  # Low SI
            days_to_cover=Decimal("15"),
            as_of_date=date(2026, 1, 15)
        )
        assert result2["squeeze_potential"] != "EXTREME"


# ============================================================================
# CROWDING RISK TESTS
# ============================================================================

class TestCrowdingRisk:
    """Tests for crowding risk assessment."""

    @pytest.fixture
    def engine(self):
        return ShortInterestSignalEngine()

    def test_high_crowding_risk(self, engine):
        """SI% >= 30% should trigger HIGH crowding."""
        result = engine.calculate_short_signal(
            ticker="CROWD",
            short_interest_pct=Decimal("35"),
            days_to_cover=Decimal("5"),
            as_of_date=date(2026, 1, 15)
        )

        assert result["crowding_risk"] == "HIGH"

    def test_medium_crowding_risk(self, engine):
        """SI% 15-30% should trigger MEDIUM crowding."""
        result = engine.calculate_short_signal(
            ticker="MED",
            short_interest_pct=Decimal("20"),
            days_to_cover=Decimal("5"),
            as_of_date=date(2026, 1, 15)
        )

        assert result["crowding_risk"] == "MEDIUM"

    def test_low_crowding_risk(self, engine):
        """SI% < 15% should trigger LOW crowding."""
        result = engine.calculate_short_signal(
            ticker="LOW",
            short_interest_pct=Decimal("10"),
            days_to_cover=Decimal("5"),
            as_of_date=date(2026, 1, 15)
        )

        assert result["crowding_risk"] == "LOW"


# ============================================================================
# TREND CONTRIBUTION TESTS
# ============================================================================

class TestTrendContribution:
    """Tests for short interest trend contribution."""

    @pytest.fixture
    def engine(self):
        return ShortInterestSignalEngine()

    def test_strong_covering_bullish(self, engine):
        """Rapidly covering shorts (negative change) should be bullish."""
        result = engine.calculate_short_signal(
            ticker="COVER",
            short_interest_pct=Decimal("20"),
            days_to_cover=Decimal("8"),
            short_interest_change_pct=Decimal("-25"),  # Strong covering
            as_of_date=date(2026, 1, 15)
        )

        assert result["trend_direction"] == "COVERING"
        assert result["component_contributions"]["trend"] > Decimal("0")

    def test_strong_buildup_bearish(self, engine):
        """Rapidly increasing shorts (positive change) should be bearish."""
        result = engine.calculate_short_signal(
            ticker="BUILD",
            short_interest_pct=Decimal("20"),
            days_to_cover=Decimal("8"),
            short_interest_change_pct=Decimal("25"),  # Strong buildup
            as_of_date=date(2026, 1, 15)
        )

        assert result["trend_direction"] == "BUILDING"
        assert result["component_contributions"]["trend"] < Decimal("0")

    def test_stable_neutral(self, engine):
        """Stable short interest should be neutral."""
        result = engine.calculate_short_signal(
            ticker="STABLE",
            short_interest_pct=Decimal("20"),
            days_to_cover=Decimal("8"),
            short_interest_change_pct=Decimal("0"),
            as_of_date=date(2026, 1, 15)
        )

        assert result["trend_direction"] == "STABLE"
        assert result["component_contributions"]["trend"] == Decimal("0")


# ============================================================================
# SIGNAL DIRECTION TESTS
# ============================================================================

class TestSignalDirection:
    """Tests for overall signal direction determination."""

    @pytest.fixture
    def engine(self):
        return ShortInterestSignalEngine()

    def test_bullish_signal_from_high_score(self, engine):
        """Score >= 60 should generate BULLISH signal."""
        # High squeeze + covering = bullish
        result = engine.calculate_short_signal(
            ticker="BULL",
            short_interest_pct=Decimal("45"),
            days_to_cover=Decimal("12"),
            short_interest_change_pct=Decimal("-20"),
            institutional_long_pct=Decimal("60"),
            as_of_date=date(2026, 1, 15)
        )

        assert result["signal_direction"] == "BULLISH"
        assert result["short_signal_score"] >= Decimal("60")

    def test_bearish_signal_from_low_score(self, engine):
        """Score <= 40 should generate BEARISH signal."""
        # Low everything + building = bearish
        result = engine.calculate_short_signal(
            ticker="BEAR",
            short_interest_pct=Decimal("5"),
            days_to_cover=Decimal("1"),
            short_interest_change_pct=Decimal("25"),
            institutional_long_pct=Decimal("10"),
            as_of_date=date(2026, 1, 15)
        )

        assert result["signal_direction"] == "BEARISH"
        assert result["short_signal_score"] <= Decimal("40")

    def test_neutral_signal_from_middle_score(self, engine):
        """Score 40-60 should generate NEUTRAL signal."""
        result = engine.calculate_short_signal(
            ticker="NEUTRAL",
            short_interest_pct=Decimal("15"),
            days_to_cover=Decimal("5"),
            as_of_date=date(2026, 1, 15)
        )

        assert result["signal_direction"] == "NEUTRAL"
        assert Decimal("40") < result["short_signal_score"] < Decimal("60")


# ============================================================================
# INSUFFICIENT DATA TESTS
# ============================================================================

class TestInsufficientData:
    """Tests for insufficient data handling."""

    @pytest.fixture
    def engine(self):
        return ShortInterestSignalEngine()

    def test_both_none_returns_insufficient(self, engine):
        """Both SI% and DTC as None should return INSUFFICIENT_DATA."""
        result = engine.calculate_short_signal(
            ticker="MISSING",
            short_interest_pct=None,
            days_to_cover=None,
            as_of_date=date(2026, 1, 15)
        )

        assert result["status"] == "INSUFFICIENT_DATA"
        assert result["squeeze_potential"] == "UNKNOWN"
        assert result["crowding_risk"] == "UNKNOWN"
        assert "SI_DATA_MISSING" in result["flags"]

    def test_insufficient_data_returns_neutral_score(self, engine):
        """Insufficient data should return neutral score of 50."""
        result = engine.calculate_short_signal(
            ticker="MISSING",
            short_interest_pct=None,
            days_to_cover=None,
            as_of_date=date(2026, 1, 15)
        )

        assert result["short_signal_score"] == Decimal("50")
        assert result["signal_direction"] == "NEUTRAL"

    def test_only_si_pct_provided(self, engine):
        """Only SI% provided should still calculate (DTC defaults to 0)."""
        result = engine.calculate_short_signal(
            ticker="PARTIAL",
            short_interest_pct=Decimal("25"),
            days_to_cover=None,
            as_of_date=date(2026, 1, 15)
        )

        assert result["status"] == "SUCCESS"
        # With DTC=0, squeeze potential will be limited
        assert result["squeeze_potential"] in ["HIGH", "MODERATE", "LOW"]


# ============================================================================
# SCORE BOUNDING TESTS
# ============================================================================

class TestScoreBounding:
    """Tests for score value bounding."""

    @pytest.fixture
    def engine(self):
        return ShortInterestSignalEngine()

    def test_score_bounded_0_100(self, engine):
        """Score should always be between 0 and 100."""
        # Test extreme positive conditions
        result_high = engine.calculate_short_signal(
            ticker="HIGH",
            short_interest_pct=Decimal("80"),
            days_to_cover=Decimal("30"),
            short_interest_change_pct=Decimal("-50"),
            institutional_long_pct=Decimal("90"),
            as_of_date=date(2026, 1, 15)
        )
        assert Decimal("0") <= result_high["short_signal_score"] <= Decimal("100")

        # Test extreme negative conditions
        result_low = engine.calculate_short_signal(
            ticker="LOW",
            short_interest_pct=Decimal("0"),
            days_to_cover=Decimal("0"),
            short_interest_change_pct=Decimal("50"),
            institutional_long_pct=Decimal("0"),
            as_of_date=date(2026, 1, 15)
        )
        assert Decimal("0") <= result_low["short_signal_score"] <= Decimal("100")


# ============================================================================
# INSTITUTIONAL SUPPORT TESTS
# ============================================================================

class TestInstitutionalSupport:
    """Tests for institutional support contribution."""

    @pytest.fixture
    def engine(self):
        return ShortInterestSignalEngine()

    def test_high_institutional_support_bullish(self, engine):
        """High institutional ownership should add positive contribution."""
        base = engine.calculate_short_signal(
            ticker="BASE",
            short_interest_pct=Decimal("20"),
            days_to_cover=Decimal("8"),
            as_of_date=date(2026, 1, 15)
        )

        with_inst = engine.calculate_short_signal(
            ticker="INST",
            short_interest_pct=Decimal("20"),
            days_to_cover=Decimal("8"),
            institutional_long_pct=Decimal("75"),
            as_of_date=date(2026, 1, 15)
        )

        assert with_inst["short_signal_score"] > base["short_signal_score"]
        assert with_inst["component_contributions"]["institutional"] > Decimal("0")


# ============================================================================
# DAYS TO COVER CONTRIBUTION TESTS
# ============================================================================

class TestDaysTooCoverContribution:
    """Tests for days-to-cover contribution."""

    @pytest.fixture
    def engine(self):
        return ShortInterestSignalEngine()

    def test_high_dtc_adds_contribution(self, engine):
        """High DTC should add positive contribution (squeeze pressure)."""
        low_dtc = engine.calculate_short_signal(
            ticker="LOW",
            short_interest_pct=Decimal("20"),
            days_to_cover=Decimal("2"),
            as_of_date=date(2026, 1, 15)
        )

        high_dtc = engine.calculate_short_signal(
            ticker="HIGH",
            short_interest_pct=Decimal("20"),
            days_to_cover=Decimal("15"),
            as_of_date=date(2026, 1, 15)
        )

        assert high_dtc["component_contributions"]["days_to_cover"] > \
               low_dtc["component_contributions"]["days_to_cover"]


# ============================================================================
# UNIVERSE SCORING TESTS
# ============================================================================

class TestUniverseScoring:
    """Tests for scoring an entire universe."""

    @pytest.fixture
    def engine(self):
        return ShortInterestSignalEngine()

    @pytest.fixture
    def si_universe(self):
        return [
            {"ticker": "SQUEEZE", "short_interest_pct": "42", "days_to_cover": "12"},
            {"ticker": "NORMAL", "short_interest_pct": "10", "days_to_cover": "4"},
            {"ticker": "MISSING"},  # No data
        ]

    def test_score_universe_returns_all_tickers(self, engine, si_universe, as_of_date):
        """Universe scoring should return scores for all tickers."""
        result = engine.score_universe(si_universe, as_of_date)

        assert len(result["scores"]) == 3
        tickers = {s["ticker"] for s in result["scores"]}
        assert tickers == {"SQUEEZE", "NORMAL", "MISSING"}

    def test_score_universe_tracks_data_coverage(self, engine, si_universe, as_of_date):
        """Universe scoring should track data coverage correctly."""
        result = engine.score_universe(si_universe, as_of_date)

        diag = result["diagnostic_counts"]
        # 2 of 3 have data
        assert diag["data_coverage"] == 2
        assert diag["data_coverage_pct"] == "66.7%"

    def test_score_universe_tracks_squeeze_distribution(self, engine, si_universe, as_of_date):
        """Universe scoring should track squeeze potential distribution."""
        result = engine.score_universe(si_universe, as_of_date)

        diag = result["diagnostic_counts"]
        assert "squeeze_distribution" in diag
        assert diag["squeeze_distribution"]["EXTREME"] >= 0

    def test_score_universe_deterministic_hash(self, engine, si_universe, as_of_date):
        """Two runs with same input should produce same content hash."""
        result1 = engine.score_universe(si_universe, as_of_date)
        result2 = engine.score_universe(si_universe, as_of_date)

        assert result1["provenance"]["content_hash"] == result2["provenance"]["content_hash"]

    def test_score_universe_includes_provenance(self, engine, si_universe, as_of_date):
        """Universe result should include provenance metadata."""
        result = engine.score_universe(si_universe, as_of_date)

        assert result["provenance"]["module"] == "short_interest_engine"
        assert result["provenance"]["module_version"] == engine.VERSION
        assert result["provenance"]["pit_cutoff"] == as_of_date.isoformat()


# ============================================================================
# AUDIT TRAIL TESTS
# ============================================================================

class TestAuditTrail:
    """Tests for audit trail functionality."""

    @pytest.fixture
    def engine(self):
        return ShortInterestSignalEngine()

    def test_audit_trail_populated(self, engine):
        """Calculating scores should add entries to audit trail."""
        engine.calculate_short_signal(
            ticker="TEST1",
            short_interest_pct=Decimal("20"),
            days_to_cover=Decimal("8"),
            as_of_date=date(2026, 1, 15)
        )
        engine.calculate_short_signal(
            ticker="TEST2",
            short_interest_pct=Decimal("10"),
            days_to_cover=Decimal("4"),
            as_of_date=date(2026, 1, 15)
        )

        assert len(engine.audit_trail) == 2

    def test_audit_contains_deterministic_timestamp(self, engine):
        """Audit entry should have deterministic timestamp."""
        result = engine.calculate_short_signal(
            ticker="TEST",
            short_interest_pct=Decimal("20"),
            days_to_cover=Decimal("8"),
            as_of_date=date(2026, 1, 15)
        )

        audit = result["audit_entry"]
        assert audit["timestamp"] == "2026-01-15T00:00:00Z"

    def test_audit_contains_calculation_details(self, engine):
        """Audit entry should contain calculation breakdown."""
        result = engine.calculate_short_signal(
            ticker="TEST",
            short_interest_pct=Decimal("20"),
            days_to_cover=Decimal("8"),
            as_of_date=date(2026, 1, 15)
        )

        audit = result["audit_entry"]
        calc = audit["calculation"]
        assert "squeeze_potential" in calc
        assert "crowding_risk" in calc
        assert "composite_score" in calc


# ============================================================================
# COMPONENT CONTRIBUTION TESTS
# ============================================================================

class TestComponentContributions:
    """Tests for component contribution tracking."""

    @pytest.fixture
    def engine(self):
        return ShortInterestSignalEngine()

    def test_all_components_tracked(self, engine):
        """All four components should be tracked in contributions."""
        result = engine.calculate_short_signal(
            ticker="TEST",
            short_interest_pct=Decimal("30"),
            days_to_cover=Decimal("10"),
            short_interest_change_pct=Decimal("-10"),
            institutional_long_pct=Decimal("50"),
            as_of_date=date(2026, 1, 15)
        )

        contribs = result["component_contributions"]
        assert "squeeze" in contribs
        assert "trend" in contribs
        assert "institutional" in contribs
        assert "days_to_cover" in contribs

    def test_contributions_are_decimals(self, engine):
        """All contributions should be Decimal values."""
        result = engine.calculate_short_signal(
            ticker="TEST",
            short_interest_pct=Decimal("30"),
            days_to_cover=Decimal("10"),
            as_of_date=date(2026, 1, 15)
        )

        for key, value in result["component_contributions"].items():
            assert isinstance(value, Decimal), f"{key} should be Decimal"
