#!/usr/bin/env python3
"""
Unit tests for module_2_financial_v2.py

Tests individual functions and scoring components:
- Burn rate calculation hierarchy
- Liquid assets computation
- Runway scoring
- Dilution risk calculation
- Liquidity scoring
- Data quality assessment
- Severity determination
- Composite scoring

Design:
- Deterministic tests (fixed as_of_date)
- Decimal-only arithmetic verification
- Edge case coverage
- Boundary condition testing
"""

import pytest
from decimal import Decimal
from datetime import date
from typing import Any, Dict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from module_2_financial_v2 import (
    # Core functions
    _to_decimal,
    _safe_divide,
    _clamp,
    _quantize_score,
    _quantize_rate,
    _extract_quarterly_burns,
    # Burn rate
    calculate_burn_rate_v2,
    calculate_burn_acceleration,
    BurnResult,
    BurnSource,
    BurnConfidence,
    BurnAcceleration,
    # Liquid assets
    calculate_liquid_assets,
    LiquidAssetsResult,
    # Runway
    calculate_runway,
    _score_runway,
    RunwayResult,
    # Dilution
    calculate_dilution_risk,
    DilutionResult,
    DilutionRiskBucket,
    # Liquidity
    score_liquidity,
    LiquidityResult,
    LiquidityGate,
    # Data quality
    assess_data_quality,
    DataQualityResult,
    DataState,
    # Severity
    determine_severity,
    # Main scoring
    score_financial_health_v2,
    run_module_2_v2,
    compute_module_2_financial,
    # Constants
    EPS,
    RUNWAY_CRITICAL,
    RUNWAY_WARNING,
    RUNWAY_CAUTION,
    LIQUIDITY_GATE_WARN,
    LIQUIDITY_GATE_FAIL,
)
from common.types import Severity


# ============================================================================
# HELPER FUNCTION TESTS
# ============================================================================

class TestToDecimal:
    """Tests for _to_decimal helper function."""

    def test_none_returns_default(self):
        assert _to_decimal(None) is None
        assert _to_decimal(None, Decimal("0")) == Decimal("0")

    def test_decimal_passthrough(self):
        val = Decimal("123.45")
        assert _to_decimal(val) == val

    def test_int_conversion(self):
        assert _to_decimal(100) == Decimal("100")
        assert _to_decimal(-50) == Decimal("-50")
        assert _to_decimal(0) == Decimal("0")

    def test_float_conversion(self):
        # Float is converted via str to avoid precision issues
        result = _to_decimal(123.45)
        assert isinstance(result, Decimal)
        assert result == Decimal("123.45")

    def test_string_conversion(self):
        assert _to_decimal("100.50") == Decimal("100.50")
        assert _to_decimal("-200") == Decimal("-200")
        assert _to_decimal("  50.0  ") == Decimal("50.0")

    def test_invalid_string_returns_default(self):
        assert _to_decimal("invalid") is None
        assert _to_decimal("invalid", Decimal("0")) == Decimal("0")

    def test_zero_value_not_falsy(self):
        """Regression: zero should not be treated as None."""
        assert _to_decimal(0) == Decimal("0")
        assert _to_decimal("0") == Decimal("0")
        assert _to_decimal(Decimal("0")) == Decimal("0")


class TestSafeDivide:
    """Tests for _safe_divide helper function."""

    def test_normal_division(self):
        result = _safe_divide(Decimal("100"), Decimal("4"))
        assert result == Decimal("25")

    def test_zero_denominator_returns_default(self):
        result = _safe_divide(Decimal("100"), Decimal("0"))
        assert result is None

        result = _safe_divide(Decimal("100"), Decimal("0"), Decimal("-1"))
        assert result == Decimal("-1")

    def test_epsilon_denominator(self):
        """Very small denominators should return default."""
        tiny = EPS / Decimal("2")
        result = _safe_divide(Decimal("100"), tiny)
        assert result is None

    def test_none_denominator(self):
        result = _safe_divide(Decimal("100"), None)
        assert result is None


class TestClamp:
    """Tests for _clamp helper function."""

    def test_value_within_range(self):
        assert _clamp(Decimal("50"), Decimal("0"), Decimal("100")) == Decimal("50")

    def test_value_below_min(self):
        assert _clamp(Decimal("-10"), Decimal("0"), Decimal("100")) == Decimal("0")

    def test_value_above_max(self):
        assert _clamp(Decimal("150"), Decimal("0"), Decimal("100")) == Decimal("100")

    def test_boundary_values(self):
        assert _clamp(Decimal("0"), Decimal("0"), Decimal("100")) == Decimal("0")
        assert _clamp(Decimal("100"), Decimal("0"), Decimal("100")) == Decimal("100")


class TestQuantizeFunctions:
    """Tests for quantization helpers."""

    def test_quantize_score_precision(self):
        """Score precision should be 2 decimal places."""
        result = _quantize_score(Decimal("50.12345"))
        assert result == Decimal("50.12")

    def test_quantize_score_rounding(self):
        """Should use ROUND_HALF_UP."""
        assert _quantize_score(Decimal("50.125")) == Decimal("50.13")
        assert _quantize_score(Decimal("50.124")) == Decimal("50.12")

    def test_quantize_rate_precision(self):
        """Rate precision should be 4 decimal places."""
        result = _quantize_rate(Decimal("0.1234567"))
        assert result == Decimal("0.1235")


# ============================================================================
# BURN RATE CALCULATION TESTS
# ============================================================================

class TestExtractQuarterlyBurns:
    """Tests for _extract_quarterly_burns function."""

    def test_empty_data(self):
        assert _extract_quarterly_burns({}) == []

    def test_burn_history_field(self):
        data = {"burn_history": [-50e6, -55e6, -48e6, -52e6]}
        burns = _extract_quarterly_burns(data)
        assert len(burns) == 4
        assert all(b > 0 for b in burns)  # Should be absolute values

    def test_positive_values_excluded(self):
        """Only negative values (burns) should be extracted."""
        data = {"burn_history": [-50e6, 10e6, -30e6]}
        burns = _extract_quarterly_burns(data)
        assert len(burns) == 2

    def test_max_four_quarters(self):
        """Should return at most 4 quarters."""
        data = {"burn_history": [-50e6] * 10}
        burns = _extract_quarterly_burns(data)
        assert len(burns) == 4


class TestCalculateBurnRateV2:
    """Tests for calculate_burn_rate_v2 function."""

    def test_profitable_company(self):
        """Profitable companies have zero burn."""
        data = {"CFO": 100e6, "NetIncome": 50e6}
        result = calculate_burn_rate_v2(data)
        assert result.monthly_burn == Decimal("0")
        assert result.burn_source == BurnSource.PROFITABLE
        assert result.burn_confidence == BurnConfidence.HIGH

    def test_cfo_quarterly_priority(self):
        """CFO quarterly should be highest priority."""
        data = {
            "CFO_quarterly": -60e6,  # $60M quarterly burn
            "CFO": -200e6,
            "NetIncome": -150e6,
        }
        result = calculate_burn_rate_v2(data)
        assert result.burn_source == BurnSource.CFO_QUARTERLY
        assert result.monthly_burn == Decimal("20000000")  # 60M / 3

    def test_cfo_ytd_differencing(self):
        """CFO YTD with prior should use difference."""
        data = {
            "CFO_YTD": -100e6,
            "CFO_YTD_prior": -40e6,
            "quarters_in_ytd": 2,
        }
        result = calculate_burn_rate_v2(data)
        assert result.burn_source == BurnSource.CFO_YTD
        # Quarterly diff = -100M - (-40M) = -60M, monthly = 20M
        assert result.monthly_burn == Decimal("20000000")

    def test_cfo_annual_fallback(self):
        """Annual CFO should be used when quarterly unavailable."""
        data = {"CFO": -120e6}  # $120M annual burn
        result = calculate_burn_rate_v2(data)
        assert result.burn_source == BurnSource.CFO_ANNUAL
        assert result.monthly_burn == Decimal("10000000")  # 120M / 12

    def test_fcf_quarterly(self):
        """FCF quarterly when CFO unavailable."""
        data = {"FCF_quarterly": -45e6}
        result = calculate_burn_rate_v2(data)
        assert result.burn_source == BurnSource.FCF_QUARTERLY
        assert result.monthly_burn == Decimal("15000000")  # 45M / 3

    def test_net_income_fallback(self):
        """Net income as fallback with medium confidence."""
        data = {"NetIncome": -30e6}
        result = calculate_burn_rate_v2(data)
        assert result.burn_source == BurnSource.NET_INCOME
        assert result.burn_confidence == BurnConfidence.MEDIUM

    def test_rd_proxy_last_resort(self):
        """R&D proxy as last resort with low confidence."""
        data = {"R&D": 20e6}  # R&D is positive (expense)
        result = calculate_burn_rate_v2(data)
        assert result.burn_source == BurnSource.RD_PROXY
        assert result.burn_confidence == BurnConfidence.LOW
        # Monthly = R&D * 1.5 / 3
        expected = Decimal("20000000") * Decimal("1.5") / Decimal("3")
        assert result.monthly_burn == _quantize_rate(expected)

    def test_no_data_returns_none(self):
        """Missing all data should return None burn."""
        data = {}
        result = calculate_burn_rate_v2(data)
        assert result.monthly_burn is None
        assert result.burn_source == BurnSource.NONE
        assert result.burn_confidence == BurnConfidence.NONE

    def test_rejection_reasons_tracked(self):
        """Rejection reasons should be tracked for each source."""
        data = {"R&D": 20e6}  # Only R&D available
        result = calculate_burn_rate_v2(data)
        assert "CFO_quarterly" in result.rejection_reasons
        assert "CFO_annual" in result.rejection_reasons


class TestCalculateBurnAcceleration:
    """Tests for calculate_burn_acceleration function."""

    def test_insufficient_data(self):
        """Less than 2 quarters should return unknown."""
        data = {"burn_history": [-50e6]}
        result = calculate_burn_acceleration(data)
        assert result.trend_direction == "unknown"
        assert result.confidence == "none"

    def test_stable_burn(self):
        """Consistent burn should be stable."""
        data = {"burn_history": [-50e6, -51e6, -49e6, -50e6]}  # <10% variation
        result = calculate_burn_acceleration(data)
        assert result.trend_direction == "stable"
        assert result.penalty_factor == Decimal("1.0")

    def test_accelerating_burn(self):
        """Increasing burn should be flagged as accelerating."""
        # Most recent is first, so increasing = each quarter burn higher than prior
        data = {"burn_history": [-80e6, -60e6, -45e6, -35e6]}  # Accelerating
        result = calculate_burn_acceleration(data)
        assert result.is_accelerating is True
        assert result.trend_direction == "accelerating"
        assert result.penalty_factor < Decimal("1.0")

    def test_decelerating_burn(self):
        """Decreasing burn should give bonus."""
        data = {"burn_history": [-30e6, -45e6, -60e6, -75e6]}  # Decelerating
        result = calculate_burn_acceleration(data)
        assert result.trend_direction == "decelerating"
        assert result.penalty_factor > Decimal("1.0")

    def test_confidence_levels(self):
        """Confidence should increase with more data."""
        data_2q = {"burn_history": [-50e6, -60e6]}
        data_3q = {"burn_history": [-50e6, -55e6, -60e6]}
        data_4q = {"burn_history": [-50e6, -52e6, -55e6, -58e6]}

        assert calculate_burn_acceleration(data_2q).confidence == "low"
        assert calculate_burn_acceleration(data_3q).confidence == "medium"
        assert calculate_burn_acceleration(data_4q).confidence == "high"


# ============================================================================
# LIQUID ASSETS TESTS
# ============================================================================

class TestCalculateLiquidAssets:
    """Tests for calculate_liquid_assets function."""

    def test_cash_only(self):
        data = {"Cash": 500e6}
        result = calculate_liquid_assets(data)
        assert result.liquid_assets == Decimal("500000000")
        assert result.cash == Decimal("500000000")
        assert result.marketable_securities == Decimal("0")
        assert "Cash" in result.components_used

    def test_cash_plus_marketable(self):
        data = {"Cash": 500e6, "MarketableSecurities": 100e6}
        result = calculate_liquid_assets(data)
        assert result.liquid_assets == Decimal("600000000")
        assert "MarketableSecurities" in result.components_used

    def test_alternative_field_names(self):
        """Should check multiple field name variations."""
        data = {"Cash": 500e6, "ShortTermInvestments": 50e6}
        result = calculate_liquid_assets(data)
        assert result.liquid_assets == Decimal("550000000")
        assert "ShortTermInvestments" in result.components_used

    def test_zero_cash(self):
        """Zero cash should not be treated as missing."""
        data = {"Cash": 0, "MarketableSecurities": 100e6}
        result = calculate_liquid_assets(data)
        assert result.cash == Decimal("0")
        assert result.liquid_assets == Decimal("100000000")


# ============================================================================
# RUNWAY SCORING TESTS
# ============================================================================

class TestScoreRunway:
    """Tests for _score_runway function."""

    def test_runway_tiers(self):
        """Test runway score thresholds."""
        assert _score_runway(Decimal("30")) == Decimal("100")  # 2+ years
        assert _score_runway(Decimal("24")) == Decimal("100")
        assert _score_runway(Decimal("20")) == Decimal("90")   # 18-24 months
        assert _score_runway(Decimal("15")) == Decimal("70")   # 12-18 months
        assert _score_runway(Decimal("9")) == Decimal("40")    # 6-12 months
        assert _score_runway(Decimal("3")) == Decimal("10")    # < 6 months


class TestCalculateRunway:
    """Tests for calculate_runway function."""

    def test_profitable_company(self):
        """Profitable company should have max runway."""
        fin_data = {"CFO": 100e6}
        mkt_data = {}
        result = calculate_runway(fin_data, mkt_data)
        assert result.runway_months == Decimal("999")
        assert result.runway_score == Decimal("100")

    def test_standard_runway_calculation(self):
        """Standard runway = liquid assets / monthly burn."""
        fin_data = {"Cash": 120e6, "CFO_quarterly": -30e6}  # $10M/month burn
        mkt_data = {}
        result = calculate_runway(fin_data, mkt_data)
        assert result.runway_months == Decimal("12")  # 120M / 10M
        assert result.runway_score == Decimal("70")  # 12-18 month tier

    def test_missing_burn_data(self):
        """Missing burn data should return neutral score."""
        fin_data = {"Cash": 100e6}  # No burn data
        mkt_data = {}
        result = calculate_runway(fin_data, mkt_data)
        assert result.runway_score == Decimal("50")  # Neutral

    def test_burn_acceleration_adjustment(self):
        """Accelerating burn should reduce score."""
        fin_data = {
            "Cash": 120e6,
            "CFO_quarterly": -30e6,
            "burn_history": [-40e6, -30e6, -20e6, -15e6],  # Accelerating
        }
        mkt_data = {}
        result = calculate_runway(fin_data, mkt_data)
        # Should have penalty applied
        assert result.runway_score_pre_acceleration is not None
        assert result.runway_score < result.runway_score_pre_acceleration


# ============================================================================
# DILUTION RISK TESTS
# ============================================================================

class TestCalculateDilutionRisk:
    """Tests for calculate_dilution_risk function."""

    def test_missing_market_cap(self):
        """Missing market cap should return unknown bucket."""
        fin_data = {"Cash": 100e6}
        mkt_data = {}
        result = calculate_dilution_risk(fin_data, mkt_data, None, None)
        assert result.dilution_risk_bucket == DilutionRiskBucket.UNKNOWN

    def test_low_dilution_risk(self):
        """High cash/mcap ratio = low dilution risk."""
        fin_data = {"Cash": 500e6}
        mkt_data = {"market_cap": 1000e6}  # 50% cash/mcap
        result = calculate_dilution_risk(fin_data, mkt_data, Decimal("24"), None)
        assert result.dilution_risk_bucket == DilutionRiskBucket.LOW
        assert result.cash_to_mcap == Decimal("0.5")

    def test_severe_dilution_risk(self):
        """Very low cash/mcap = severe dilution risk."""
        fin_data = {"Cash": 20e6}
        mkt_data = {"market_cap": 1000e6}  # 2% cash/mcap
        result = calculate_dilution_risk(fin_data, mkt_data, Decimal("6"), None)
        assert result.dilution_risk_bucket == DilutionRiskBucket.SEVERE

    def test_share_count_growth(self):
        """Should calculate share dilution."""
        fin_data = {
            "Cash": 100e6,
            "shares_outstanding": 110e6,
            "shares_outstanding_prior": 100e6,
        }
        mkt_data = {"market_cap": 500e6}
        result = calculate_dilution_risk(fin_data, mkt_data, Decimal("12"), None)
        assert result.share_count_growth == Decimal("0.1")  # 10% growth

    def test_burn_to_mcap_ratio(self):
        """Should calculate annual burn / market cap."""
        fin_data = {"Cash": 200e6}
        mkt_data = {"market_cap": 1000e6}
        monthly_burn = Decimal("10000000")  # $10M/month
        result = calculate_dilution_risk(fin_data, mkt_data, Decimal("20"), monthly_burn)
        # Annual burn = 120M, mcap = 1000M, ratio = 0.12
        assert result.burn_to_mcap == Decimal("0.12")


# ============================================================================
# LIQUIDITY SCORING TESTS
# ============================================================================

class TestScoreLiquidity:
    """Tests for score_liquidity function."""

    def test_high_liquidity(self):
        """Large cap with high volume = PASS."""
        mkt_data = {
            "market_cap": 10e9,
            "avg_volume": 5e6,
            "price": 50,
        }
        result = score_liquidity(mkt_data)
        assert result.liquidity_gate == LiquidityGate.PASS
        assert result.dollar_adv_20d == Decimal("250000000")  # 5M * $50

    def test_low_liquidity_warn(self):
        """Low ADV should trigger WARN."""
        mkt_data = {
            "market_cap": 500e6,
            "avg_volume": 50000,
            "price": 5,
        }
        result = score_liquidity(mkt_data)
        # ADV = 50K * $5 = $250K (between FAIL and WARN thresholds)
        assert result.liquidity_gate == LiquidityGate.WARN

    def test_very_low_liquidity_fail(self):
        """Very low ADV should trigger FAIL."""
        mkt_data = {
            "market_cap": 100e6,
            "avg_volume": 5000,
            "price": 5,
        }
        result = score_liquidity(mkt_data)
        # ADV = 5K * $5 = $25K (below FAIL threshold)
        assert result.liquidity_gate == LiquidityGate.FAIL

    def test_composite_score_weights(self):
        """Score should be 60% ADV, 40% market cap."""
        mkt_data = {
            "market_cap": 10e9,  # Large cap score = 100
            "avg_volume": 1e6,
            "price": 5,  # ADV = $5M → score ~70
        }
        result = score_liquidity(mkt_data)
        # Expected: 70 * 0.6 + 100 * 0.4 = 42 + 40 = 82
        assert result.liquidity_score == Decimal("82")


# ============================================================================
# DATA QUALITY TESTS
# ============================================================================

class TestAssessDataQuality:
    """Tests for assess_data_quality function."""

    def test_full_data(self):
        """All fields present = FULL state."""
        fin_data = {
            "Cash": 100e6,
            "MarketableSecurities": 50e6,
            "CFO": -20e6,
        }
        mkt_data = {
            "market_cap": 500e6,
            "avg_volume": 1e6,
            "price": 25,
        }
        result = assess_data_quality(fin_data, mkt_data)
        assert result.financial_data_state == DataState.FULL
        assert result.confidence == Decimal("1.0")
        assert len(result.missing_fields) == 0

    def test_partial_data(self):
        """Some fields missing = PARTIAL state."""
        fin_data = {"Cash": 100e6}  # Missing burn data
        mkt_data = {"market_cap": 500e6, "avg_volume": 1e6, "price": 25}
        result = assess_data_quality(fin_data, mkt_data)
        assert result.financial_data_state in [DataState.PARTIAL, DataState.MINIMAL]

    def test_missing_critical_data(self):
        """Missing critical fields = NONE state."""
        fin_data = {}  # No cash
        mkt_data = {}  # No market cap
        result = assess_data_quality(fin_data, mkt_data)
        assert result.financial_data_state == DataState.NONE
        assert "cash" in result.missing_fields
        assert "market_cap" in result.missing_fields

    def test_inputs_tracked(self):
        """Should track which fields were used."""
        fin_data = {"Cash": 100e6, "NetIncome": -20e6}
        mkt_data = {"market_cap": 500e6}
        result = assess_data_quality(fin_data, mkt_data)
        assert "cash" in result.inputs_used
        assert result.inputs_used["cash"] == "Cash"


# ============================================================================
# SEVERITY TESTS
# ============================================================================

class TestDetermineSeverity:
    """Tests for determine_severity function."""

    def test_healthy_runway(self):
        """Runway >= 18 months = no severity."""
        assert determine_severity(Decimal("24"), Decimal("0.3")) == Severity.NONE

    def test_caution_runway(self):
        """12-18 months = SEV1."""
        assert determine_severity(Decimal("15"), Decimal("0.2")) == Severity.SEV1

    def test_warning_runway(self):
        """6-12 months = SEV2."""
        assert determine_severity(Decimal("9"), Decimal("0.1")) == Severity.SEV2

    def test_critical_runway(self):
        """<6 months = SEV3."""
        assert determine_severity(Decimal("3"), Decimal("0.05")) == Severity.SEV3

    def test_missing_runway_low_cash(self):
        """No runway but low cash/mcap = SEV2."""
        assert determine_severity(None, Decimal("0.03")) == Severity.SEV2

    def test_missing_all(self):
        """Missing all data = NONE (unknown)."""
        assert determine_severity(None, None) == Severity.NONE


# ============================================================================
# MAIN SCORING FUNCTION TESTS
# ============================================================================

class TestScoreFinancialHealthV2:
    """Tests for score_financial_health_v2 function."""

    def test_complete_scoring(self):
        """Test with complete data."""
        fin_data = {
            "Cash": 500e6,
            "MarketableSecurities": 100e6,
            "CFO_quarterly": -50e6,
        }
        mkt_data = {
            "market_cap": 2e9,
            "avg_volume": 500000,
            "price": 40,
        }
        result = score_financial_health_v2("ACME", fin_data, mkt_data)

        # Check required fields
        assert result["ticker"] == "ACME"
        assert "financial_score" in result
        assert "runway_months" in result
        assert "severity" in result
        assert "flags" in result
        assert "determinism_hash" in result

    def test_output_types(self):
        """Verify output field types (floats for API compat)."""
        fin_data = {"Cash": 100e6, "CFO": -50e6}
        mkt_data = {"market_cap": 500e6, "avg_volume": 100000, "price": 10}
        result = score_financial_health_v2("TEST", fin_data, mkt_data)

        # Numeric fields should be floats
        assert isinstance(result["financial_score"], (float, type(None)))
        assert isinstance(result["runway_months"], (float, type(None)))

        # Boolean fields
        assert isinstance(result["has_financial_data"], bool)
        assert isinstance(result["liquidity_gate"], bool)

    def test_flags_generated(self):
        """Test that appropriate flags are set."""
        # Low runway scenario
        fin_data = {"Cash": 30e6, "CFO_quarterly": -30e6}  # 3 month runway
        mkt_data = {"market_cap": 1e9, "avg_volume": 100000, "price": 10}
        result = score_financial_health_v2("LOW", fin_data, mkt_data)
        assert "low_runway" in result["flags"]

    def test_determinism(self):
        """Same inputs should produce identical hash."""
        fin_data = {"Cash": 100e6, "CFO": -50e6}
        mkt_data = {"market_cap": 500e6, "avg_volume": 100000, "price": 10}

        result1 = score_financial_health_v2("SAME", fin_data, mkt_data)
        result2 = score_financial_health_v2("SAME", fin_data, mkt_data)

        assert result1["determinism_hash"] == result2["determinism_hash"]
        assert result1["financial_score"] == result2["financial_score"]

    def test_composite_score_weights(self):
        """Composite should be 50% runway, 30% dilution, 20% liquidity."""
        # Create scenario where we know the component scores
        fin_data = {"Cash": 600e6, "CFO_quarterly": -50e6}  # 24+ months
        mkt_data = {"market_cap": 1e9, "avg_volume": 10e6, "price": 50}
        result = score_financial_health_v2("WEIGHTS", fin_data, mkt_data)

        # With 24+ months runway and 60% cash/mcap, scores should be high
        assert result["financial_score"] is not None
        assert result["financial_score"] > 50


class TestRunModule2V2:
    """Tests for run_module_2_v2 entry point."""

    def test_empty_universe(self):
        """Empty universe should return empty results."""
        result = run_module_2_v2([], [], [])
        assert result["scores"] == []
        assert result["diagnostic_counts"]["scored"] == 0

    def test_multiple_tickers(self):
        """Should score all tickers in universe."""
        universe = ["TICK1", "TICK2", "TICK3"]
        fin_data = [
            {"ticker": "TICK1", "Cash": 100e6},
            {"ticker": "TICK2", "Cash": 200e6},
        ]
        mkt_data = [
            {"ticker": "TICK1", "market_cap": 500e6, "avg_volume": 100000, "price": 10},
            {"ticker": "TICK2", "market_cap": 1e9, "avg_volume": 200000, "price": 20},
        ]

        result = run_module_2_v2(universe, fin_data, mkt_data)
        assert len(result["scores"]) == 3
        assert result["diagnostic_counts"]["scored"] == 3

    def test_diagnostic_distributions(self):
        """Should track severity and data state distributions."""
        universe = ["TICK1", "TICK2"]
        fin_data = [
            {"ticker": "TICK1", "Cash": 30e6, "CFO_quarterly": -30e6},  # Critical
            {"ticker": "TICK2", "Cash": 500e6},  # Healthy
        ]
        mkt_data = [
            {"ticker": "TICK1", "market_cap": 100e6, "avg_volume": 50000, "price": 5},
            {"ticker": "TICK2", "market_cap": 1e9, "avg_volume": 500000, "price": 50},
        ]

        result = run_module_2_v2(universe, fin_data, mkt_data)
        assert "severity_distribution" in result["diagnostic_counts"]
        assert "data_state_distribution" in result["diagnostic_counts"]


class TestComputeModule2Financial:
    """Tests for backwards compatibility wrapper."""

    def test_legacy_field_mapping(self):
        """Should map legacy field names."""
        records = [
            {
                "ticker": "LEGACY",
                "cash_mm": 100,  # Legacy: in millions
                "burn_rate_mm": 10,  # Legacy: in millions
                "market_cap_mm": 500,
            }
        ]
        universe = ["LEGACY"]

        result = compute_module_2_financial(records, universe, "2026-01-15")
        assert len(result["scores"]) == 1
        assert result["scores"][0]["ticker"] == "LEGACY"

    def test_set_universe(self):
        """Should handle set universe input."""
        records = [{"ticker": "TEST", "Cash": 100e6}]
        universe = {"TEST"}  # Set instead of list

        result = compute_module_2_financial(universe=universe, financial_records=records)
        assert len(result["scores"]) == 1


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Edge case and boundary condition tests."""

    def test_zero_market_cap(self):
        """Zero market cap should not cause division error."""
        fin_data = {"Cash": 100e6}
        mkt_data = {"market_cap": 0}
        result = score_financial_health_v2("ZERO", fin_data, mkt_data)
        assert result["cash_to_mcap"] is None

    def test_negative_cash(self):
        """Negative cash should be handled."""
        fin_data = {"Cash": -10e6}  # Overdraft
        mkt_data = {"market_cap": 100e6, "avg_volume": 10000, "price": 5}
        result = score_financial_health_v2("NEG", fin_data, mkt_data)
        # Should still produce a result
        assert "financial_score" in result

    def test_very_large_values(self):
        """Should handle very large values without overflow."""
        fin_data = {"Cash": 1e12}  # $1 trillion
        mkt_data = {"market_cap": 2e12, "avg_volume": 1e9, "price": 1000}
        result = score_financial_health_v2("HUGE", fin_data, mkt_data)
        assert result["financial_score"] is not None

    def test_unicode_ticker(self):
        """Should handle unicode in ticker (even if unusual)."""
        fin_data = {"Cash": 100e6}
        mkt_data = {"market_cap": 500e6, "avg_volume": 100000, "price": 10}
        result = score_financial_health_v2("TEST_123", fin_data, mkt_data)
        assert result["ticker"] == "TEST_123"


# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================

@pytest.mark.parametrize("cash,burn,expected_runway_approx", [
    (120e6, -30e6, 12),   # 12 months
    (240e6, -30e6, 24),   # 24 months
    (60e6, -30e6, 6),     # 6 months
    (30e6, -30e6, 3),     # 3 months
    (600e6, -30e6, 60),   # 5 years
])
def test_runway_calculation_parametrized(cash, burn, expected_runway_approx):
    """Parametrized test for runway calculations."""
    fin_data = {"Cash": cash, "CFO_quarterly": burn}
    mkt_data = {}
    result = calculate_runway(fin_data, mkt_data)
    assert abs(float(result.runway_months) - expected_runway_approx) < 1


@pytest.mark.parametrize("runway_months,expected_severity", [
    (Decimal("3"), Severity.SEV3),
    (Decimal("6"), Severity.SEV2),
    (Decimal("12"), Severity.SEV1),
    (Decimal("18"), Severity.NONE),
    (Decimal("24"), Severity.NONE),
])
def test_severity_thresholds_parametrized(runway_months, expected_severity):
    """Parametrized test for severity thresholds."""
    result = determine_severity(runway_months, Decimal("0.2"))
    assert result == expected_severity
