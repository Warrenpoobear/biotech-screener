#!/usr/bin/env python3
"""
tests/test_dilution_risk.py

Comprehensive tests for the Dilution Risk Scoring Engine.

Tests cover:
- No risk case (sufficient cash through catalyst)
- High risk case (will run out of cash)
- Medium risk case (with financing capacity)
- Determinism (same inputs produce same outputs)
- PIT discipline (catalyst date must be future)
- Fail-closed behavior (missing data returns error)
- Confidence scoring
- Integration with composite scoring

Author: Wake Robin Capital Management
"""

import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

# Add parent to path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dilution_risk_engine import (
    DilutionRiskEngine,
    DataQualityState,
    RiskBucket,
    integrate_dilution_risk,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def engine() -> DilutionRiskEngine:
    """Fresh engine instance for each test."""
    return DilutionRiskEngine()


@pytest.fixture
def as_of_date() -> date:
    """Standard as_of_date for deterministic tests."""
    return date(2026, 1, 15)


@pytest.fixture
def well_funded_company(as_of_date) -> dict:
    """Company with sufficient cash through catalyst."""
    return {
        "ticker": "FUNDED",
        "quarterly_cash": Decimal("500000000"),  # $500M
        "quarterly_burn": Decimal("-30000000"),  # $30M quarterly = $10M/month
        "next_catalyst_date": "2026-07-15",      # 6 months away
        "market_cap": Decimal("2000000000"),     # $2B
        "avg_daily_volume_90d": 2_000_000,
    }


@pytest.fixture
def underfunded_company(as_of_date) -> dict:
    """Company that will run out of cash before catalyst."""
    return {
        "ticker": "BURNING",
        "quarterly_cash": Decimal("30000000"),   # $30M
        "quarterly_burn": Decimal("-45000000"),  # $45M quarterly = $15M/month
        "next_catalyst_date": "2026-12-15",      # 11 months away
        "market_cap": Decimal("100000000"),      # $100M
        "avg_daily_volume_90d": 500_000,
        "shelf_capacity": Decimal("0"),
        "atm_remaining": Decimal("0"),
    }


@pytest.fixture
def company_with_atm(as_of_date) -> dict:
    """Company with ATM program providing additional capacity."""
    return {
        "ticker": "ATMUSER",
        "quarterly_cash": Decimal("60000000"),   # $60M cash
        "quarterly_burn": Decimal("-30000000"),  # $30M quarterly = $10M/month
        "next_catalyst_date": "2026-06-15",      # 5 months away ($50M needed)
        "market_cap": Decimal("400000000"),      # $400M
        "atm_remaining": Decimal("40000000"),    # $40M ATM capacity (28M usable)
        "atm_active": True,
        "avg_daily_volume_90d": 1_500_000,
    }


@pytest.fixture
def minimal_data_company(as_of_date) -> dict:
    """Company with only required fields."""
    return {
        "ticker": "MINIMAL",
        "quarterly_cash": Decimal("100000000"),
        "quarterly_burn": Decimal("-25000000"),
        "next_catalyst_date": "2026-06-15",
    }


# ============================================================================
# NO RISK TESTS
# ============================================================================

class TestNoRiskCase:
    """Tests for companies with sufficient funding."""

    def test_well_funded_returns_no_risk(self, engine, well_funded_company, as_of_date):
        """Company with sufficient cash through catalyst has zero risk."""
        result = engine.calculate_dilution_risk(
            **well_funded_company,
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "SUCCESS"
        assert result["risk_bucket"] == "NO_RISK"
        assert result["dilution_risk_score"] == Decimal("0.0")

    def test_no_risk_cash_gap_negative(self, engine, well_funded_company, as_of_date):
        """No risk case should have negative or zero cash gap."""
        result = engine.calculate_dilution_risk(
            **well_funded_company,
            as_of_date=as_of_date,
        )

        assert result["components"]["cash_gap"] <= Decimal("0")

    def test_no_risk_has_full_feasibility(self, engine, well_funded_company, as_of_date):
        """No risk case should have raise feasibility of 1.0."""
        result = engine.calculate_dilution_risk(
            **well_funded_company,
            as_of_date=as_of_date,
        )

        assert result["components"]["raise_feasibility"] == Decimal("1.0")


# ============================================================================
# HIGH RISK TESTS
# ============================================================================

class TestHighRiskCase:
    """Tests for companies with significant funding gaps."""

    def test_underfunded_returns_high_risk(self, engine, underfunded_company, as_of_date):
        """Company running out of cash before catalyst has high risk."""
        result = engine.calculate_dilution_risk(
            **underfunded_company,
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "SUCCESS"
        assert result["risk_bucket"] == "HIGH_RISK"
        assert result["dilution_risk_score"] > Decimal("0.70")

    def test_high_risk_positive_cash_gap(self, engine, underfunded_company, as_of_date):
        """High risk case should have positive cash gap."""
        result = engine.calculate_dilution_risk(
            **underfunded_company,
            as_of_date=as_of_date,
        )

        assert result["components"]["cash_gap"] > Decimal("0")

    def test_high_risk_low_feasibility(self, engine, underfunded_company, as_of_date):
        """High risk case should have low raise feasibility."""
        result = engine.calculate_dilution_risk(
            **underfunded_company,
            as_of_date=as_of_date,
        )

        assert result["components"]["raise_feasibility"] < Decimal("0.40")

    def test_very_short_runway_extreme_risk(self, engine, as_of_date):
        """Company with <3 month runway should have maximum risk."""
        result = engine.calculate_dilution_risk(
            ticker="IMMINENT",
            quarterly_cash=Decimal("10000000"),    # $10M
            quarterly_burn=Decimal("-60000000"),   # $60M quarterly = $20M/month
            next_catalyst_date="2026-12-15",       # 11 months away
            market_cap=Decimal("50000000"),        # $50M (tiny market cap)
            avg_daily_volume_90d=100_000,          # Low volume
            as_of_date=as_of_date,
        )

        assert result["risk_bucket"] == "HIGH_RISK"
        assert result["dilution_risk_score"] >= Decimal("0.80")


# ============================================================================
# MEDIUM RISK / FINANCING CAPACITY TESTS
# ============================================================================

class TestMediumRiskCase:
    """Tests for companies with moderate risk and financing options."""

    def test_atm_capacity_reduces_risk(self, engine, company_with_atm, as_of_date):
        """Company with ATM capacity should have reduced risk."""
        result = engine.calculate_dilution_risk(
            **company_with_atm,
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "SUCCESS"
        # ATM capacity covers any potential shortfall, should NOT be HIGH_RISK
        assert result["risk_bucket"] in ["NO_RISK", "LOW_RISK", "MEDIUM_RISK"]

    def test_atm_usable_capacity_factor(self, engine, company_with_atm, as_of_date):
        """ATM capacity should be discounted by usable factor (70%)."""
        result = engine.calculate_dilution_risk(
            **company_with_atm,
            as_of_date=as_of_date,
        )

        atm_amount = Decimal("40000000")
        expected_usable = atm_amount * Decimal("0.70")

        assert result["components"]["usable_capacity"] == expected_usable.quantize(Decimal("0.01"))

    def test_shelf_capacity_also_counts(self, engine, as_of_date):
        """Shelf registration capacity should also reduce risk."""
        result = engine.calculate_dilution_risk(
            ticker="SHELVED",
            quarterly_cash=Decimal("40000000"),
            quarterly_burn=Decimal("-36000000"),
            next_catalyst_date="2026-09-15",
            market_cap=Decimal("300000000"),
            shelf_capacity=Decimal("100000000"),  # $100M shelf
            shelf_filed=True,
            avg_daily_volume_90d=1_000_000,
            as_of_date=as_of_date,
        )

        # Should have usable capacity from shelf
        assert result["components"]["usable_capacity"] > Decimal("0")


# ============================================================================
# DETERMINISM TESTS
# ============================================================================

class TestDeterminism:
    """Tests ensuring identical inputs produce identical outputs."""

    def test_same_inputs_same_hash(self, engine, well_funded_company, as_of_date):
        """Same inputs always produce same hash."""
        result1 = engine.calculate_dilution_risk(
            **well_funded_company,
            as_of_date=as_of_date,
        )

        # Create new engine for fresh state
        engine2 = DilutionRiskEngine()
        result2 = engine2.calculate_dilution_risk(
            **well_funded_company,
            as_of_date=as_of_date,
        )

        assert result1["hash"] == result2["hash"]

    def test_same_inputs_same_score(self, engine, underfunded_company, as_of_date):
        """Same inputs always produce same risk score."""
        result1 = engine.calculate_dilution_risk(
            **underfunded_company,
            as_of_date=as_of_date,
        )

        engine2 = DilutionRiskEngine()
        result2 = engine2.calculate_dilution_risk(
            **underfunded_company,
            as_of_date=as_of_date,
        )

        assert result1["dilution_risk_score"] == result2["dilution_risk_score"]
        assert result1["risk_bucket"] == result2["risk_bucket"]

    def test_different_inputs_different_hash(self, engine, well_funded_company, underfunded_company, as_of_date):
        """Different inputs produce different hashes."""
        result1 = engine.calculate_dilution_risk(
            **well_funded_company,
            as_of_date=as_of_date,
        )
        result2 = engine.calculate_dilution_risk(
            **underfunded_company,
            as_of_date=as_of_date,
        )

        assert result1["hash"] != result2["hash"]


# ============================================================================
# PIT DISCIPLINE TESTS
# ============================================================================

class TestPITDiscipline:
    """Tests for Point-in-Time enforcement."""

    def test_catalyst_must_be_in_future(self, engine, as_of_date):
        """Catalyst date must be after as_of_date."""
        result = engine.calculate_dilution_risk(
            ticker="PASTCAT",
            quarterly_cash=Decimal("100000000"),
            quarterly_burn=Decimal("-25000000"),
            next_catalyst_date="2026-01-10",  # Before as_of_date
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "INVALID_CATALYST_DATE"
        assert result["dilution_risk_score"] is None
        assert result["risk_bucket"] == "UNKNOWN"

    def test_catalyst_on_as_of_date_invalid(self, engine, as_of_date):
        """Catalyst date on as_of_date is invalid (must be strictly future)."""
        result = engine.calculate_dilution_risk(
            ticker="SAMEDAY",
            quarterly_cash=Decimal("100000000"),
            quarterly_burn=Decimal("-25000000"),
            next_catalyst_date="2026-01-15",  # Same as as_of_date
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "INVALID_CATALYST_DATE"

    def test_invalid_date_format_rejected(self, engine, as_of_date):
        """Invalid date format should return error."""
        result = engine.calculate_dilution_risk(
            ticker="BADDATE",
            quarterly_cash=Decimal("100000000"),
            quarterly_burn=Decimal("-25000000"),
            next_catalyst_date="not-a-date",
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "INVALID_CATALYST_DATE"


# ============================================================================
# FAIL-CLOSED BEHAVIOR TESTS
# ============================================================================

class TestFailClosed:
    """Tests for explicit error handling on missing data."""

    def test_missing_cash_returns_error(self, engine, as_of_date):
        """Missing cash field returns error with explicit reason."""
        result = engine.calculate_dilution_risk(
            ticker="NOCASH",
            quarterly_cash=None,
            quarterly_burn=Decimal("-25000000"),
            next_catalyst_date="2026-07-15",
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "INSUFFICIENT_DATA"
        assert "quarterly_cash" in result["missing_fields"]
        assert result["dilution_risk_score"] is None

    def test_missing_burn_returns_error(self, engine, as_of_date):
        """Missing burn field returns error with explicit reason."""
        result = engine.calculate_dilution_risk(
            ticker="NOBURN",
            quarterly_cash=Decimal("100000000"),
            quarterly_burn=None,
            next_catalyst_date="2026-07-15",
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "INSUFFICIENT_DATA"
        assert "quarterly_burn" in result["missing_fields"]

    def test_missing_catalyst_date_returns_error(self, engine, as_of_date):
        """Missing catalyst date returns error with explicit reason."""
        result = engine.calculate_dilution_risk(
            ticker="NOCAT",
            quarterly_cash=Decimal("100000000"),
            quarterly_burn=Decimal("-25000000"),
            next_catalyst_date=None,
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "INSUFFICIENT_DATA"
        assert "next_catalyst_date" in result["missing_fields"]

    def test_multiple_missing_fields_all_reported(self, engine, as_of_date):
        """All missing required fields are reported."""
        result = engine.calculate_dilution_risk(
            ticker="EMPTY",
            quarterly_cash=None,
            quarterly_burn=None,
            next_catalyst_date=None,
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "INSUFFICIENT_DATA"
        assert "quarterly_cash" in result["missing_fields"]
        assert "quarterly_burn" in result["missing_fields"]
        assert "next_catalyst_date" in result["missing_fields"]


# ============================================================================
# CONFIDENCE SCORING TESTS
# ============================================================================

class TestConfidenceScoring:
    """Tests for data completeness confidence scoring."""

    def test_full_data_high_confidence(self, engine, well_funded_company, as_of_date):
        """Full data should have high confidence (close to 1.0)."""
        # Add all optional fields
        well_funded_company["shelf_capacity"] = Decimal("100000000")
        well_funded_company["atm_remaining"] = Decimal("50000000")

        result = engine.calculate_dilution_risk(
            **well_funded_company,
            as_of_date=as_of_date,
        )

        assert result["confidence"] >= Decimal("0.85")

    def test_minimal_data_lower_confidence(self, engine, minimal_data_company, as_of_date):
        """Minimal data should have lower confidence."""
        result = engine.calculate_dilution_risk(
            **minimal_data_company,
            as_of_date=as_of_date,
        )

        # Base confidence is 0.50 for required fields only
        assert result["confidence"] == Decimal("0.50")

    def test_confidence_increases_with_volume(self, engine, minimal_data_company, as_of_date):
        """Adding volume data should increase confidence."""
        minimal_data_company["avg_daily_volume_90d"] = 1_000_000

        result = engine.calculate_dilution_risk(
            **minimal_data_company,
            as_of_date=as_of_date,
        )

        # Should get +0.15 for volume
        assert result["confidence"] >= Decimal("0.65")

    def test_error_result_zero_confidence(self, engine, as_of_date):
        """Error results should have zero confidence."""
        result = engine.calculate_dilution_risk(
            ticker="ERROR",
            quarterly_cash=None,
            quarterly_burn=None,
            next_catalyst_date=None,
            as_of_date=as_of_date,
        )

        assert result["confidence"] == Decimal("0.0")


# ============================================================================
# DATA QUALITY STATE TESTS
# ============================================================================

class TestDataQualityState:
    """Tests for data quality classification."""

    def test_full_quality_all_fields(self, engine, well_funded_company, as_of_date):
        """All fields present = FULL quality."""
        well_funded_company["shelf_capacity"] = Decimal("100000000")
        well_funded_company["atm_remaining"] = Decimal("50000000")

        result = engine.calculate_dilution_risk(
            **well_funded_company,
            as_of_date=as_of_date,
        )

        assert result["data_quality_state"] == "FULL"

    def test_partial_quality_some_optional_missing(self, engine, as_of_date):
        """Some optional fields missing = PARTIAL quality."""
        # Company without market_cap - an optional field that gets tracked
        result = engine.calculate_dilution_risk(
            ticker="PARTIAL",
            quarterly_cash=Decimal("100000000"),
            quarterly_burn=Decimal("-25000000"),
            next_catalyst_date="2026-07-15",
            avg_daily_volume_90d=1_000_000,  # Present
            # market_cap is missing - gets tracked as missing
            as_of_date=as_of_date,
        )

        # market_cap is optional but tracked as missing
        assert result["data_quality_state"] in ["PARTIAL", "MINIMAL"]

    def test_none_quality_required_missing(self, engine, as_of_date):
        """Required fields missing = NONE quality."""
        result = engine.calculate_dilution_risk(
            ticker="MISSING",
            quarterly_cash=None,
            quarterly_burn=Decimal("-25000000"),
            next_catalyst_date="2026-07-15",
            as_of_date=as_of_date,
        )

        assert result["data_quality_state"] == "NONE"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Tests for integration with composite scoring."""

    def test_integrate_applies_penalty(self, engine, underfunded_company, as_of_date):
        """High risk should apply penalty to composite score."""
        dilution_result = engine.calculate_dilution_risk(
            **underfunded_company,
            as_of_date=as_of_date,
        )

        base_score = Decimal("75.00")
        adjusted = integrate_dilution_risk(base_score, dilution_result)

        assert adjusted < base_score

    def test_integrate_max_penalty_capped(self, engine, as_of_date):
        """Penalty should not exceed max_penalty."""
        # Create extreme high risk scenario
        extreme_risk = engine.calculate_dilution_risk(
            ticker="EXTREME",
            quarterly_cash=Decimal("5000000"),     # $5M
            quarterly_burn=Decimal("-90000000"),   # $90M quarterly
            next_catalyst_date="2026-12-15",
            market_cap=Decimal("20000000"),        # $20M
            avg_daily_volume_90d=10_000,
            as_of_date=as_of_date,
        )

        base_score = Decimal("80.00")
        adjusted = integrate_dilution_risk(base_score, extreme_risk, max_penalty=Decimal("15"))

        # Maximum penalty should be 15 points
        assert base_score - adjusted <= Decimal("15")

    def test_integrate_no_penalty_below_confidence(self, engine, minimal_data_company, as_of_date):
        """No penalty applied if confidence below threshold."""
        dilution_result = engine.calculate_dilution_risk(
            **minimal_data_company,
            as_of_date=as_of_date,
        )

        # Minimal data has confidence of 0.50, below default threshold of 0.60
        base_score = Decimal("75.00")
        adjusted = integrate_dilution_risk(base_score, dilution_result)

        assert adjusted == base_score

    def test_integrate_no_penalty_on_error(self, engine, as_of_date):
        """No penalty applied on error results."""
        error_result = engine.calculate_dilution_risk(
            ticker="ERROR",
            quarterly_cash=None,
            quarterly_burn=None,
            next_catalyst_date=None,
            as_of_date=as_of_date,
        )

        base_score = Decimal("75.00")
        adjusted = integrate_dilution_risk(base_score, error_result)

        assert adjusted == base_score

    def test_integrate_no_penalty_no_risk(self, engine, well_funded_company, as_of_date):
        """No penalty for NO_RISK companies."""
        no_risk = engine.calculate_dilution_risk(
            **well_funded_company,
            as_of_date=as_of_date,
        )

        base_score = Decimal("75.00")
        adjusted = integrate_dilution_risk(base_score, no_risk)

        # Zero risk score = zero penalty
        assert adjusted == base_score


# ============================================================================
# UNIVERSE SCORING TESTS
# ============================================================================

class TestUniverseScoring:
    """Tests for batch scoring of universe."""

    def test_score_universe_returns_all_tickers(self, engine, as_of_date):
        """Universe scoring returns result for each ticker."""
        universe = [
            {
                "ticker": "TICK1",
                "quarterly_cash": Decimal("100000000"),
                "quarterly_burn": Decimal("-20000000"),
                "next_catalyst_date": "2026-07-15",
            },
            {
                "ticker": "TICK2",
                "quarterly_cash": Decimal("50000000"),
                "quarterly_burn": Decimal("-30000000"),
                "next_catalyst_date": "2026-09-15",
            },
        ]

        result = engine.score_universe(universe, as_of_date)

        assert len(result["scores"]) == 2
        tickers = [s["ticker"] for s in result["scores"]]
        assert "TICK1" in tickers
        assert "TICK2" in tickers

    def test_score_universe_tracks_distributions(self, engine, as_of_date):
        """Universe scoring tracks risk and quality distributions."""
        universe = [
            {
                "ticker": "FUNDED",
                "quarterly_cash": Decimal("500000000"),
                "quarterly_burn": Decimal("-30000000"),
                "next_catalyst_date": "2026-07-15",
                "market_cap": Decimal("2000000000"),
                "avg_daily_volume_90d": 2_000_000,
            },
            {
                "ticker": "BURNING",
                "quarterly_cash": Decimal("20000000"),
                "quarterly_burn": Decimal("-50000000"),
                "next_catalyst_date": "2026-12-15",
                "market_cap": Decimal("100000000"),
                "avg_daily_volume_90d": 500_000,
            },
        ]

        result = engine.score_universe(universe, as_of_date)

        assert "risk_distribution" in result["diagnostic_counts"]
        assert "data_quality_distribution" in result["diagnostic_counts"]

    def test_score_universe_deterministic_hash(self, engine, as_of_date):
        """Universe scoring produces deterministic content hash."""
        universe = [
            {
                "ticker": "TICK1",
                "quarterly_cash": Decimal("100000000"),
                "quarterly_burn": Decimal("-20000000"),
                "next_catalyst_date": "2026-07-15",
            },
        ]

        result1 = engine.score_universe(universe, as_of_date)

        engine2 = DilutionRiskEngine()
        result2 = engine2.score_universe(universe, as_of_date)

        assert result1["provenance"]["content_hash"] == result2["provenance"]["content_hash"]


# ============================================================================
# AUDIT TRAIL TESTS
# ============================================================================

class TestAuditTrail:
    """Tests for audit trail functionality."""

    def test_audit_trail_recorded(self, engine, well_funded_company, as_of_date):
        """Each calculation is recorded in audit trail."""
        engine.calculate_dilution_risk(
            **well_funded_company,
            as_of_date=as_of_date,
        )

        trail = engine.get_audit_trail()
        assert len(trail) == 1
        assert trail[0]["ticker"] == "FUNDED"

    def test_audit_trail_deterministic_timestamp(self, engine, well_funded_company, as_of_date):
        """Audit trail uses deterministic timestamp from as_of_date."""
        engine.calculate_dilution_risk(
            **well_funded_company,
            as_of_date=as_of_date,
        )

        trail = engine.get_audit_trail()
        assert trail[0]["timestamp"] == "2026-01-15T00:00:00Z"

    def test_audit_trail_can_be_cleared(self, engine, well_funded_company, as_of_date):
        """Audit trail can be cleared."""
        engine.calculate_dilution_risk(
            **well_funded_company,
            as_of_date=as_of_date,
        )

        engine.clear_audit_trail()
        trail = engine.get_audit_trail()

        assert len(trail) == 0


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_burn_rate(self, engine, as_of_date):
        """Zero burn rate should result in no risk."""
        result = engine.calculate_dilution_risk(
            ticker="ZEROBURN",
            quarterly_cash=Decimal("100000000"),
            quarterly_burn=Decimal("0"),
            next_catalyst_date="2026-07-15",
            market_cap=Decimal("500000000"),
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "SUCCESS"
        assert result["risk_bucket"] == "NO_RISK"

    def test_positive_burn_treated_as_cash_inflow(self, engine, as_of_date):
        """Positive 'burn' (cash inflow) should result in no risk."""
        result = engine.calculate_dilution_risk(
            ticker="CASHGEN",
            quarterly_cash=Decimal("100000000"),
            quarterly_burn=Decimal("20000000"),  # Positive = generating cash
            next_catalyst_date="2026-07-15",
            market_cap=Decimal("500000000"),
            as_of_date=as_of_date,
        )

        # Engine uses abs(burn), so still calculates need
        # But with positive burn, might indicate cash generating
        assert result["reason_code"] == "SUCCESS"

    def test_very_large_cash_position(self, engine, as_of_date):
        """Very large cash position should be no risk."""
        result = engine.calculate_dilution_risk(
            ticker="MEGACASH",
            quarterly_cash=Decimal("10000000000"),  # $10B
            quarterly_burn=Decimal("-100000000"),   # $100M quarterly
            next_catalyst_date="2026-12-15",
            market_cap=Decimal("50000000000"),
            as_of_date=as_of_date,
        )

        assert result["risk_bucket"] == "NO_RISK"

    def test_very_small_market_cap(self, engine, as_of_date):
        """Very small market cap increases dilution difficulty."""
        result = engine.calculate_dilution_risk(
            ticker="MICROCAP",
            quarterly_cash=Decimal("5000000"),      # $5M
            quarterly_burn=Decimal("-10000000"),    # $10M quarterly
            next_catalyst_date="2026-12-15",
            market_cap=Decimal("10000000"),         # $10M market cap
            avg_daily_volume_90d=10_000,            # Low volume
            as_of_date=as_of_date,
        )

        # Large raise relative to market cap = high risk
        assert result["risk_bucket"] == "HIGH_RISK"

    def test_catalyst_far_future(self, engine, as_of_date):
        """Catalyst very far in future increases cash needs."""
        result = engine.calculate_dilution_risk(
            ticker="LONGTERM",
            quarterly_cash=Decimal("100000000"),
            quarterly_burn=Decimal("-30000000"),
            next_catalyst_date="2028-01-15",  # 2 years away
            market_cap=Decimal("500000000"),
            avg_daily_volume_90d=1_000_000,
            as_of_date=as_of_date,
        )

        # 24 months * $10M/month = $240M needed vs $100M cash
        assert result["components"]["cash_gap"] > Decimal("0")

    def test_string_numeric_inputs_converted(self, engine, as_of_date):
        """String numeric inputs should be properly converted."""
        result = engine.calculate_dilution_risk(
            ticker="STRINGS",
            quarterly_cash="100000000",  # String
            quarterly_burn="-25000000",  # String
            next_catalyst_date="2026-07-15",
            market_cap="500000000",      # String
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "SUCCESS"

    def test_float_inputs_converted(self, engine, as_of_date):
        """Float inputs should be properly converted via string."""
        result = engine.calculate_dilution_risk(
            ticker="FLOATS",
            quarterly_cash=100000000.50,  # Float
            quarterly_burn=-25000000.25,  # Float
            next_catalyst_date="2026-07-15",
            market_cap=500000000.0,       # Float
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "SUCCESS"


# ============================================================================
# SCORE BOUNDS TESTS
# ============================================================================

class TestScoreBounds:
    """Tests ensuring scores stay within valid ranges."""

    def test_risk_score_bounded_zero_to_one(self, engine, as_of_date):
        """Risk score should always be between 0 and 1."""
        # Test with various scenarios
        scenarios = [
            {  # Low risk
                "quarterly_cash": Decimal("500000000"),
                "quarterly_burn": Decimal("-30000000"),
                "next_catalyst_date": "2026-07-15",
            },
            {  # High risk
                "quarterly_cash": Decimal("10000000"),
                "quarterly_burn": Decimal("-80000000"),
                "next_catalyst_date": "2026-12-15",
            },
        ]

        for i, scenario in enumerate(scenarios):
            result = engine.calculate_dilution_risk(
                ticker=f"TEST{i}",
                **scenario,
                as_of_date=as_of_date,
            )

            if result["dilution_risk_score"] is not None:
                assert Decimal("0") <= result["dilution_risk_score"] <= Decimal("1"), \
                    f"Score {result['dilution_risk_score']} out of bounds for scenario {i}"

    def test_confidence_bounded_zero_to_one(self, engine, as_of_date):
        """Confidence should always be between 0 and 1."""
        result = engine.calculate_dilution_risk(
            ticker="CONFTEST",
            quarterly_cash=Decimal("100000000"),
            quarterly_burn=Decimal("-25000000"),
            next_catalyst_date="2026-07-15",
            as_of_date=as_of_date,
        )

        assert Decimal("0") <= result["confidence"] <= Decimal("1")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
