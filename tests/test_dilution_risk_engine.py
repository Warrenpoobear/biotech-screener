#!/usr/bin/env python3
"""
Tests for dilution_risk_engine.py

Tests the Dilution Risk Scoring Engine.
Covers:
- Risk bucket classification
- Cash gap calculations
- Data quality assessment
- Confidence scoring
- Edge cases (zero runway, negative cash, etc.)
- Integration helper function
"""

import pytest
from datetime import date, timedelta
from decimal import Decimal

from dilution_risk_engine import (
    DilutionRiskEngine,
    DataQualityState,
    RiskBucket,
    integrate_dilution_risk,
)


class TestDilutionRiskEngineInit:
    """Tests for engine initialization."""

    def test_engine_init(self):
        """Engine should initialize correctly."""
        engine = DilutionRiskEngine()
        assert engine.VERSION == "1.0.0"
        assert engine.audit_trail == []

    def test_engine_constants(self):
        """Engine constants should be set correctly."""
        engine = DilutionRiskEngine()
        assert engine.RISK_SCORE_MIN == Decimal("0")
        assert engine.RISK_SCORE_MAX == Decimal("1")
        assert engine.LOW_RISK_THRESHOLD == Decimal("0.40")
        assert engine.MEDIUM_RISK_THRESHOLD == Decimal("0.70")


class TestCalculateDilutionRisk:
    """Tests for calculate_dilution_risk method."""

    @pytest.fixture
    def engine(self):
        return DilutionRiskEngine()

    @pytest.fixture
    def as_of_date(self):
        return date(2026, 1, 15)

    def test_missing_as_of_date_raises(self, engine):
        """as_of_date is required."""
        with pytest.raises(ValueError, match="as_of_date is REQUIRED"):
            engine.calculate_dilution_risk(
                ticker="TEST",
                quarterly_cash=Decimal("100000000"),
                quarterly_burn=Decimal("-15000000"),
                next_catalyst_date="2026-07-15",
            )

    def test_missing_cash_returns_insufficient_data(self, engine, as_of_date):
        """Missing cash should return INSUFFICIENT_DATA."""
        result = engine.calculate_dilution_risk(
            ticker="TEST",
            quarterly_cash=None,
            quarterly_burn=Decimal("-15000000"),
            next_catalyst_date="2026-07-15",
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "INSUFFICIENT_DATA"
        assert result["dilution_risk_score"] is None
        assert "quarterly_cash" in result["missing_fields"]

    def test_missing_burn_returns_insufficient_data(self, engine, as_of_date):
        """Missing burn should return INSUFFICIENT_DATA."""
        result = engine.calculate_dilution_risk(
            ticker="TEST",
            quarterly_cash=Decimal("100000000"),
            quarterly_burn=None,
            next_catalyst_date="2026-07-15",
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "INSUFFICIENT_DATA"
        assert "quarterly_burn" in result["missing_fields"]

    def test_missing_catalyst_date_returns_insufficient_data(self, engine, as_of_date):
        """Missing catalyst date should return INSUFFICIENT_DATA."""
        result = engine.calculate_dilution_risk(
            ticker="TEST",
            quarterly_cash=Decimal("100000000"),
            quarterly_burn=Decimal("-15000000"),
            next_catalyst_date=None,
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "INSUFFICIENT_DATA"
        assert "next_catalyst_date" in result["missing_fields"]

    def test_invalid_catalyst_date_format(self, engine, as_of_date):
        """Invalid date format should return error."""
        result = engine.calculate_dilution_risk(
            ticker="TEST",
            quarterly_cash=Decimal("100000000"),
            quarterly_burn=Decimal("-15000000"),
            next_catalyst_date="not-a-date",
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "INVALID_CATALYST_DATE"

    def test_catalyst_date_in_past(self, engine, as_of_date):
        """Catalyst date in past should return error."""
        result = engine.calculate_dilution_risk(
            ticker="TEST",
            quarterly_cash=Decimal("100000000"),
            quarterly_burn=Decimal("-15000000"),
            next_catalyst_date="2025-01-01",  # Before as_of_date
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "INVALID_CATALYST_DATE"
        assert "must be in future" in result.get("details", "")

    def test_well_funded_company_no_risk(self, engine, as_of_date):
        """Well-funded company should have NO_RISK."""
        result = engine.calculate_dilution_risk(
            ticker="FUNDED",
            quarterly_cash=Decimal("500000000"),  # $500M
            quarterly_burn=Decimal("-30000000"),  # $30M/quarter = $10M/month
            next_catalyst_date="2026-07-15",       # 6 months away
            market_cap=Decimal("2000000000"),
            avg_daily_volume_90d=2_000_000,
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "SUCCESS"
        assert result["risk_bucket"] == "NO_RISK"
        assert result["dilution_risk_score"] == Decimal("0.000")
        assert result["components"]["cash_gap"] <= Decimal("0")

    def test_underfunded_company_high_risk(self, engine, as_of_date):
        """Underfunded company should have HIGH_RISK."""
        result = engine.calculate_dilution_risk(
            ticker="BURNING",
            quarterly_cash=Decimal("30000000"),   # $30M cash
            quarterly_burn=Decimal("-45000000"),  # $15M/month burn
            next_catalyst_date="2026-12-15",      # 11 months away
            market_cap=Decimal("100000000"),      # $100M market cap
            avg_daily_volume_90d=500_000,
            shelf_capacity=Decimal("0"),
            atm_remaining=Decimal("0"),
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "SUCCESS"
        assert result["risk_bucket"] == "HIGH_RISK"
        assert result["dilution_risk_score"] > Decimal("0.70")

    def test_medium_risk_with_atm(self, engine, as_of_date):
        """Company with ATM capacity should have reduced risk."""
        result = engine.calculate_dilution_risk(
            ticker="ATMUSER",
            quarterly_cash=Decimal("40000000"),   # $40M
            quarterly_burn=Decimal("-36000000"),  # $12M/month
            next_catalyst_date="2026-09-15",      # 8 months away
            market_cap=Decimal("300000000"),
            atm_remaining=Decimal("50000000"),    # $50M ATM capacity
            atm_active=True,
            avg_daily_volume_90d=1_000_000,
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "SUCCESS"
        # ATM reduces risk
        assert result["components"]["usable_capacity"] > Decimal("0")

    def test_risk_score_bounds(self, engine, as_of_date):
        """Risk score should always be between 0 and 1."""
        # Test low risk scenario
        result_low = engine.calculate_dilution_risk(
            ticker="LOW",
            quarterly_cash=Decimal("1000000000"),
            quarterly_burn=Decimal("-10000000"),
            next_catalyst_date="2026-03-15",
            as_of_date=as_of_date,
        )
        if result_low["dilution_risk_score"] is not None:
            assert Decimal("0") <= result_low["dilution_risk_score"] <= Decimal("1")

        # Test high risk scenario
        result_high = engine.calculate_dilution_risk(
            ticker="HIGH",
            quarterly_cash=Decimal("10000000"),
            quarterly_burn=Decimal("-50000000"),
            next_catalyst_date="2026-12-15",
            market_cap=Decimal("50000000"),
            as_of_date=as_of_date,
        )
        if result_high["dilution_risk_score"] is not None:
            assert Decimal("0") <= result_high["dilution_risk_score"] <= Decimal("1")

    def test_determinism(self, engine, as_of_date):
        """Same inputs should produce same outputs."""
        params = dict(
            ticker="ACME",
            quarterly_cash=Decimal("100000000"),
            quarterly_burn=Decimal("-20000000"),
            next_catalyst_date="2026-07-15",
            market_cap=Decimal("500000000"),
            avg_daily_volume_90d=1_000_000,
            as_of_date=as_of_date,
        )

        result1 = engine.calculate_dilution_risk(**params)
        result2 = engine.calculate_dilution_risk(**params)

        assert result1["dilution_risk_score"] == result2["dilution_risk_score"]
        assert result1["risk_bucket"] == result2["risk_bucket"]
        assert result1["hash"] == result2["hash"]

    def test_hash_is_computed(self, engine, as_of_date):
        """Result should include content hash."""
        result = engine.calculate_dilution_risk(
            ticker="TEST",
            quarterly_cash=Decimal("100000000"),
            quarterly_burn=Decimal("-15000000"),
            next_catalyst_date="2026-07-15",
            as_of_date=as_of_date,
        )

        assert result["hash"] is not None
        assert len(result["hash"]) == 16  # First 16 chars of SHA256

    def test_audit_entry_created(self, engine, as_of_date):
        """Calculation should create audit entry."""
        engine.clear_audit_trail()

        engine.calculate_dilution_risk(
            ticker="TEST",
            quarterly_cash=Decimal("100000000"),
            quarterly_burn=Decimal("-15000000"),
            next_catalyst_date="2026-07-15",
            as_of_date=as_of_date,
        )

        audit = engine.get_audit_trail()
        assert len(audit) == 1
        assert audit[0]["ticker"] == "TEST"


class TestDataQualityAssessment:
    """Tests for data quality assessment."""

    @pytest.fixture
    def engine(self):
        return DilutionRiskEngine()

    @pytest.fixture
    def as_of_date(self):
        return date(2026, 1, 15)

    def test_full_data_quality(self, engine, as_of_date):
        """All fields present should result in FULL quality."""
        result = engine.calculate_dilution_risk(
            ticker="TEST",
            quarterly_cash=Decimal("100000000"),
            quarterly_burn=Decimal("-15000000"),
            next_catalyst_date="2026-07-15",
            market_cap=Decimal("500000000"),
            shelf_capacity=Decimal("100000000"),
            atm_remaining=Decimal("50000000"),
            avg_daily_volume_90d=1_000_000,
            shelf_filed=True,
            atm_active=True,
            as_of_date=as_of_date,
        )

        assert result["data_quality_state"] == "FULL"

    def test_minimal_data_quality(self, engine, as_of_date):
        """Only required fields should result in MINIMAL quality."""
        result = engine.calculate_dilution_risk(
            ticker="TEST",
            quarterly_cash=Decimal("100000000"),
            quarterly_burn=Decimal("-15000000"),
            next_catalyst_date="2026-07-15",
            as_of_date=as_of_date,
        )

        # Missing all optional fields
        assert result["data_quality_state"] in ["MINIMAL", "PARTIAL"]

    def test_none_quality_missing_required(self, engine, as_of_date):
        """Missing required fields should result in NONE quality."""
        result = engine.calculate_dilution_risk(
            ticker="TEST",
            quarterly_cash=None,
            quarterly_burn=Decimal("-15000000"),
            next_catalyst_date="2026-07-15",
            as_of_date=as_of_date,
        )

        assert result["data_quality_state"] == "NONE"


class TestConfidenceScoring:
    """Tests for confidence score calculation."""

    @pytest.fixture
    def engine(self):
        return DilutionRiskEngine()

    @pytest.fixture
    def as_of_date(self):
        return date(2026, 1, 15)

    def test_confidence_with_all_optional(self, engine, as_of_date):
        """Full optional data should give high confidence."""
        result = engine.calculate_dilution_risk(
            ticker="TEST",
            quarterly_cash=Decimal("100000000"),
            quarterly_burn=Decimal("-15000000"),
            next_catalyst_date="2026-07-15",
            shelf_capacity=Decimal("100000000"),
            atm_remaining=Decimal("50000000"),
            avg_daily_volume_90d=1_000_000,
            as_of_date=as_of_date,
        )

        assert result["confidence"] == Decimal("1.00")

    def test_confidence_without_optional(self, engine, as_of_date):
        """Missing optional data should give base confidence."""
        result = engine.calculate_dilution_risk(
            ticker="TEST",
            quarterly_cash=Decimal("100000000"),
            quarterly_burn=Decimal("-15000000"),
            next_catalyst_date="2026-07-15",
            as_of_date=as_of_date,
        )

        assert result["confidence"] == Decimal("0.50")  # Base only

    def test_confidence_bounds(self, engine, as_of_date):
        """Confidence should be between 0 and 1."""
        result = engine.calculate_dilution_risk(
            ticker="TEST",
            quarterly_cash=Decimal("100000000"),
            quarterly_burn=Decimal("-15000000"),
            next_catalyst_date="2026-07-15",
            as_of_date=as_of_date,
        )

        if result["confidence"] is not None:
            assert Decimal("0") <= result["confidence"] <= Decimal("1")


class TestScoreUniverse:
    """Tests for score_universe method."""

    @pytest.fixture
    def engine(self):
        return DilutionRiskEngine()

    @pytest.fixture
    def as_of_date(self):
        return date(2026, 1, 15)

    def test_score_universe_basic(self, engine, as_of_date):
        """Should score multiple companies."""
        universe = [
            {
                "ticker": "ACME",
                "quarterly_cash": Decimal("500000000"),
                "quarterly_burn": Decimal("-30000000"),
                "next_catalyst_date": "2026-07-15",
            },
            {
                "ticker": "BETA",
                "quarterly_cash": Decimal("50000000"),
                "quarterly_burn": Decimal("-40000000"),
                "next_catalyst_date": "2026-12-15",
            },
        ]

        result = engine.score_universe(universe, as_of_date)

        assert result["as_of_date"] == "2026-01-15"
        assert len(result["scores"]) == 2
        assert result["diagnostic_counts"]["total_scored"] == 2

    def test_score_universe_diagnostics(self, engine, as_of_date):
        """Should include proper diagnostics."""
        universe = [
            {
                "ticker": "SAFE",
                "quarterly_cash": Decimal("500000000"),
                "quarterly_burn": Decimal("-10000000"),
                "next_catalyst_date": "2026-07-15",
            },
            {
                "ticker": "RISKY",
                "quarterly_cash": Decimal("10000000"),
                "quarterly_burn": Decimal("-50000000"),
                "next_catalyst_date": "2026-12-15",
                "market_cap": Decimal("50000000"),
            },
        ]

        result = engine.score_universe(universe, as_of_date)

        diag = result["diagnostic_counts"]
        assert "risk_distribution" in diag
        assert "data_quality_distribution" in diag

    def test_score_universe_with_errors(self, engine, as_of_date):
        """Should handle companies with missing data."""
        universe = [
            {
                "ticker": "GOOD",
                "quarterly_cash": Decimal("100000000"),
                "quarterly_burn": Decimal("-15000000"),
                "next_catalyst_date": "2026-07-15",
            },
            {
                "ticker": "BAD",
                "quarterly_cash": None,  # Missing required
                "quarterly_burn": Decimal("-15000000"),
                "next_catalyst_date": "2026-07-15",
            },
        ]

        result = engine.score_universe(universe, as_of_date)

        assert result["diagnostic_counts"]["success_count"] == 1
        assert result["diagnostic_counts"]["error_count"] == 1


class TestRiskBucketClassification:
    """Tests for risk bucket classification logic."""

    @pytest.fixture
    def engine(self):
        return DilutionRiskEngine()

    @pytest.fixture
    def as_of_date(self):
        return date(2026, 1, 15)

    def test_no_risk_classification(self, engine, as_of_date):
        """Company with negative cash gap should be NO_RISK."""
        # Very well funded - cash exceeds needs
        result = engine.calculate_dilution_risk(
            ticker="TEST",
            quarterly_cash=Decimal("1000000000"),  # $1B cash
            quarterly_burn=Decimal("-10000000"),    # $3.3M/month
            next_catalyst_date="2026-03-15",        # 2 months
            as_of_date=as_of_date,
        )

        assert result["risk_bucket"] == "NO_RISK"

    def test_unknown_risk_on_error(self, engine, as_of_date):
        """Error cases should return UNKNOWN risk bucket."""
        result = engine.calculate_dilution_risk(
            ticker="TEST",
            quarterly_cash=None,
            quarterly_burn=Decimal("-15000000"),
            next_catalyst_date="2026-07-15",
            as_of_date=as_of_date,
        )

        assert result["risk_bucket"] == "UNKNOWN"


class TestTypeConversions:
    """Tests for type conversion helpers."""

    @pytest.fixture
    def engine(self):
        return DilutionRiskEngine()

    @pytest.fixture
    def as_of_date(self):
        return date(2026, 1, 15)

    def test_accepts_float_inputs(self, engine, as_of_date):
        """Should accept float inputs."""
        result = engine.calculate_dilution_risk(
            ticker="TEST",
            quarterly_cash=100000000.0,
            quarterly_burn=-15000000.0,
            next_catalyst_date="2026-07-15",
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "SUCCESS"

    def test_accepts_int_inputs(self, engine, as_of_date):
        """Should accept integer inputs."""
        result = engine.calculate_dilution_risk(
            ticker="TEST",
            quarterly_cash=100000000,
            quarterly_burn=-15000000,
            next_catalyst_date="2026-07-15",
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "SUCCESS"

    def test_accepts_string_inputs(self, engine, as_of_date):
        """Should accept string numeric inputs."""
        result = engine.calculate_dilution_risk(
            ticker="TEST",
            quarterly_cash="100000000",
            quarterly_burn="-15000000",
            next_catalyst_date="2026-07-15",
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "SUCCESS"


class TestIntegrateDilutionRisk:
    """Tests for integrate_dilution_risk helper function."""

    def test_integration_below_confidence_threshold(self):
        """Low confidence should not apply penalty."""
        dilution_data = {
            "dilution_risk_score": Decimal("0.80"),
            "confidence": Decimal("0.40"),  # Below 0.60 threshold
        }

        result = integrate_dilution_risk(
            base_score=Decimal("75.00"),
            dilution_data=dilution_data,
        )

        assert result == Decimal("75.00")  # No change

    def test_integration_above_confidence_threshold(self):
        """High confidence should apply penalty."""
        dilution_data = {
            "dilution_risk_score": Decimal("0.80"),
            "confidence": Decimal("0.80"),  # Above 0.60 threshold
        }

        result = integrate_dilution_risk(
            base_score=Decimal("75.00"),
            dilution_data=dilution_data,
        )

        # 0.80 * 15 (max penalty) = 12 points penalty
        expected = Decimal("75.00") - (Decimal("0.80") * Decimal("15"))
        assert result == expected.quantize(Decimal("0.01"))

    def test_integration_with_none_risk_score(self):
        """None risk score should not apply penalty."""
        dilution_data = {
            "dilution_risk_score": None,
            "confidence": Decimal("0.80"),
        }

        result = integrate_dilution_risk(
            base_score=Decimal("75.00"),
            dilution_data=dilution_data,
        )

        assert result == Decimal("75.00")

    def test_integration_floor_at_zero(self):
        """Result should not go below zero."""
        dilution_data = {
            "dilution_risk_score": Decimal("1.0"),  # Max risk
            "confidence": Decimal("1.0"),
        }

        result = integrate_dilution_risk(
            base_score=Decimal("10.00"),  # Low base score
            dilution_data=dilution_data,
            max_penalty=Decimal("20"),  # Penalty would exceed base
        )

        assert result == Decimal("0")

    def test_integration_custom_max_penalty(self):
        """Should respect custom max penalty."""
        dilution_data = {
            "dilution_risk_score": Decimal("1.0"),
            "confidence": Decimal("1.0"),
        }

        result = integrate_dilution_risk(
            base_score=Decimal("75.00"),
            dilution_data=dilution_data,
            max_penalty=Decimal("10"),
        )

        expected = Decimal("65.00")  # 75 - (1.0 * 10)
        assert result == expected

    def test_integration_custom_confidence_threshold(self):
        """Should respect custom confidence threshold."""
        dilution_data = {
            "dilution_risk_score": Decimal("0.80"),
            "confidence": Decimal("0.50"),
        }

        # With default threshold 0.60, this would not apply
        result_default = integrate_dilution_risk(
            base_score=Decimal("75.00"),
            dilution_data=dilution_data,
        )
        assert result_default == Decimal("75.00")

        # With lower threshold, it should apply
        result_lower = integrate_dilution_risk(
            base_score=Decimal("75.00"),
            dilution_data=dilution_data,
            confidence_threshold=Decimal("0.40"),
        )
        assert result_lower < Decimal("75.00")


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def engine(self):
        return DilutionRiskEngine()

    @pytest.fixture
    def as_of_date(self):
        return date(2026, 1, 15)

    def test_zero_burn_rate(self, engine, as_of_date):
        """Zero burn rate (profitable company) should work."""
        result = engine.calculate_dilution_risk(
            ticker="PROFITABLE",
            quarterly_cash=Decimal("100000000"),
            quarterly_burn=Decimal("0"),  # No burn
            next_catalyst_date="2026-07-15",
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "SUCCESS"
        assert result["risk_bucket"] == "NO_RISK"

    def test_positive_burn_profitable(self, engine, as_of_date):
        """Positive burn (cash generation) should result in no risk."""
        result = engine.calculate_dilution_risk(
            ticker="CASHGEN",
            quarterly_cash=Decimal("100000000"),
            quarterly_burn=Decimal("10000000"),  # Generating cash
            next_catalyst_date="2026-07-15",
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "SUCCESS"
        # Profitable company should have low/no risk
        assert result["risk_bucket"] in ["NO_RISK", "LOW_RISK"]

    def test_very_small_burn(self, engine, as_of_date):
        """Very small burn rate should handle correctly."""
        result = engine.calculate_dilution_risk(
            ticker="SMALL",
            quarterly_cash=Decimal("100000000"),
            quarterly_burn=Decimal("-100"),  # $100 quarterly burn
            next_catalyst_date="2026-07-15",
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "SUCCESS"

    def test_catalyst_just_after_as_of_date(self, engine, as_of_date):
        """Catalyst 1 day in future should work."""
        result = engine.calculate_dilution_risk(
            ticker="IMMINENT",
            quarterly_cash=Decimal("100000000"),
            quarterly_burn=Decimal("-15000000"),
            next_catalyst_date="2026-01-16",  # Tomorrow
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "SUCCESS"

    def test_catalyst_same_day_invalid(self, engine, as_of_date):
        """Catalyst on same day as as_of_date should be invalid."""
        result = engine.calculate_dilution_risk(
            ticker="TODAY",
            quarterly_cash=Decimal("100000000"),
            quarterly_burn=Decimal("-15000000"),
            next_catalyst_date="2026-01-15",  # Same as as_of_date
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "INVALID_CATALYST_DATE"

    def test_very_large_numbers(self, engine, as_of_date):
        """Very large numbers should handle correctly."""
        result = engine.calculate_dilution_risk(
            ticker="MEGA",
            quarterly_cash=Decimal("999999999999"),  # ~$1T
            quarterly_burn=Decimal("-999999999"),
            next_catalyst_date="2026-07-15",
            market_cap=Decimal("9999999999999"),
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "SUCCESS"
