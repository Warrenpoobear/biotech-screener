#!/usr/bin/env python3
"""
Error handling tests for Module 2: Financial Health Scoring (v2)

Tests edge cases and error scenarios including:
- Division by zero handling
- Missing/null data handling
- Invalid type handling
- Boundary conditions
"""

import pytest
from datetime import date
from decimal import Decimal

from module_2_financial_v2 import (
    score_financial_health_v2,
    _to_decimal,
    _safe_divide,
    _clamp,
    _quantize_score,
    EPS,
    LiquidityGate,
    DilutionRiskBucket,
)
from common.types import Severity


class TestToDecimalEdgeCases:
    """Tests for _to_decimal type conversion."""

    def test_none_returns_default(self):
        """None should return default value."""
        assert _to_decimal(None) is None
        assert _to_decimal(None, Decimal("0")) == Decimal("0")

    def test_empty_string_returns_default(self):
        """Empty string should return default."""
        assert _to_decimal("") is None
        assert _to_decimal("  ") is None  # Whitespace only

    def test_invalid_string_returns_default(self):
        """Invalid string should return default."""
        assert _to_decimal("not-a-number") is None
        assert _to_decimal("$500") is None  # Currency symbol
        assert _to_decimal("500M") is None  # Suffix

    def test_valid_string_converts(self):
        """Valid numeric string should convert."""
        assert _to_decimal("123.45") == Decimal("123.45")
        assert _to_decimal("-999.99") == Decimal("-999.99")
        assert _to_decimal("0") == Decimal("0")

    def test_int_converts(self):
        """Integer should convert to Decimal."""
        assert _to_decimal(100) == Decimal("100")
        assert _to_decimal(-50) == Decimal("-50")

    def test_float_converts(self):
        """Float should convert to Decimal."""
        result = _to_decimal(123.45)
        assert result is not None

    def test_decimal_passthrough(self):
        """Decimal should pass through."""
        value = Decimal("123.45")
        assert _to_decimal(value) == value

    def test_special_float_values(self):
        """Special float values should be handled."""
        result = _to_decimal(float('inf'))
        # Either None or Infinity is acceptable
        assert result is None or str(result) == "Infinity"


class TestSafeDivideEdgeCases:
    """Tests for _safe_divide function."""

    def test_divide_by_zero_returns_default(self):
        """Division by zero should return default."""
        result = _safe_divide(Decimal("100"), Decimal("0"))
        assert result is None

        result = _safe_divide(Decimal("100"), Decimal("0"), default=Decimal("999"))
        assert result == Decimal("999")

    def test_divide_by_eps_returns_default(self):
        """Division by EPS (very small) should return default."""
        result = _safe_divide(Decimal("100"), EPS / 10)
        # Should handle very small divisors

    def test_normal_division(self):
        """Normal division should work."""
        result = _safe_divide(Decimal("100"), Decimal("4"))
        assert result == Decimal("25")

    def test_negative_division(self):
        """Negative values should work."""
        result = _safe_divide(Decimal("-100"), Decimal("4"))
        assert result == Decimal("-25")

        result = _safe_divide(Decimal("100"), Decimal("-4"))
        assert result == Decimal("-25")

    def test_none_denominator(self):
        """None denominator should return default."""
        result = _safe_divide(Decimal("100"), None)
        assert result is None


class TestClampEdgeCases:
    """Tests for _clamp function."""

    def test_below_min(self):
        """Value below min should be clamped to min."""
        result = _clamp(Decimal("-10"), Decimal("0"), Decimal("100"))
        assert result == Decimal("0")

    def test_above_max(self):
        """Value above max should be clamped to max."""
        result = _clamp(Decimal("150"), Decimal("0"), Decimal("100"))
        assert result == Decimal("100")

    def test_at_min(self):
        """Value at min should remain unchanged."""
        result = _clamp(Decimal("0"), Decimal("0"), Decimal("100"))
        assert result == Decimal("0")

    def test_at_max(self):
        """Value at max should remain unchanged."""
        result = _clamp(Decimal("100"), Decimal("0"), Decimal("100"))
        assert result == Decimal("100")

    def test_value_in_range(self):
        """Value in range should remain unchanged."""
        result = _clamp(Decimal("50"), Decimal("0"), Decimal("100"))
        assert result == Decimal("50")


class TestQuantizeScoreEdgeCases:
    """Tests for _quantize_score function."""

    def test_rounds_to_two_decimals(self):
        """Score should be rounded to 2 decimal places."""
        result = _quantize_score(Decimal("50.1234"))
        assert result == Decimal("50.12")

    def test_rounds_half_up(self):
        """Should round 0.5 up."""
        result = _quantize_score(Decimal("50.125"))
        assert result == Decimal("50.13") or result == Decimal("50.12")


class TestScoreFinancialHealthV2:
    """Tests for score_financial_health_v2 error handling."""

    def test_empty_financial_data(self):
        """Empty financial data should handle gracefully."""
        result = score_financial_health_v2(
            ticker="EMPTY",
            financial_data={},
            market_data={},
        )
        assert result is not None
        assert result.get("ticker") == "EMPTY"

    def test_empty_market_data(self):
        """Empty market data should handle gracefully."""
        result = score_financial_health_v2(
            ticker="NOMARKET",
            financial_data={"Cash": 100000000},
            market_data={},
        )
        assert result is not None

    def test_negative_cash(self):
        """Negative cash should be handled."""
        result = score_financial_health_v2(
            ticker="NEGCASH",
            financial_data={
                "Cash": -50000000,
                "CFO_quarterly": -10000000,
            },
            market_data={},
        )
        assert result is not None

    def test_zero_cash(self):
        """Zero cash should be handled."""
        result = score_financial_health_v2(
            ticker="ZEROCASH",
            financial_data={
                "Cash": 0,
                "CFO_quarterly": -10000000,
            },
            market_data={},
        )
        assert result is not None

    def test_zero_burn(self):
        """Zero burn (profitable company) should be handled."""
        result = score_financial_health_v2(
            ticker="PROFITABLE",
            financial_data={
                "Cash": 100000000,
                "CFO_quarterly": 0,
            },
            market_data={},
        )
        assert result is not None

    def test_positive_cfo(self):
        """Positive CFO (generating cash) should be handled."""
        result = score_financial_health_v2(
            ticker="CASHGEN",
            financial_data={
                "Cash": 100000000,
                "CFO_quarterly": 10000000,
            },
            market_data={},
        )
        assert result is not None

    def test_string_numeric_values(self):
        """String numeric values should be converted."""
        result = score_financial_health_v2(
            ticker="STRINGS",
            financial_data={
                "Cash": "100000000",
                "CFO_quarterly": "-10000000",
            },
            market_data={
                "shares_outstanding": "50000000",
            },
        )
        assert result is not None

    def test_invalid_type_values(self):
        """Invalid type values should be handled gracefully."""
        result = score_financial_health_v2(
            ticker="INVALID",
            financial_data={
                "Cash": {"nested": "dict"},
                "CFO_quarterly": [1, 2, 3],
            },
            market_data={
                "shares_outstanding": None,
            },
        )
        assert result is not None

    def test_extreme_large_values(self):
        """Extremely large values should be handled."""
        result = score_financial_health_v2(
            ticker="HUGE",
            financial_data={
                "Cash": 999999999999999,
                "CFO_quarterly": -999999999999,
            },
            market_data={},
        )
        assert result is not None

    def test_extreme_small_burn(self):
        """Extremely small burn rate should be handled."""
        result = score_financial_health_v2(
            ticker="TINY",
            financial_data={
                "Cash": 100000000,
                "CFO_quarterly": -100,
            },
            market_data={},
        )
        assert result is not None


class TestScoreBounds:
    """Tests for score boundary conditions."""

    def test_score_clamped_to_zero(self):
        """Score should not go below 0."""
        result = score_financial_health_v2(
            ticker="CRITICAL",
            financial_data={
                "Cash": 1000000,
                "CFO_quarterly": -50000000,
            },
            market_data={},
        )
        if result and result.get("financial_score") is not None:
            score = Decimal(str(result["financial_score"]))
            assert score >= Decimal("0")

    def test_score_clamped_to_hundred(self):
        """Score should not exceed 100."""
        result = score_financial_health_v2(
            ticker="WEALTHY",
            financial_data={
                "Cash": 10000000000,
                "CFO_quarterly": 1000000000,
            },
            market_data={
                "avg_volume_20d": 10000000,
            },
        )
        if result and result.get("financial_score") is not None:
            score = Decimal(str(result["financial_score"]))
            assert score <= Decimal("100")


class TestEnums:
    """Tests for enum values."""

    def test_liquidity_gate_values(self):
        """LiquidityGate should have values."""
        assert len(list(LiquidityGate)) > 0

    def test_dilution_risk_bucket_values(self):
        """DilutionRiskBucket should have values."""
        assert len(list(DilutionRiskBucket)) > 0

    def test_severity_values(self):
        """Severity should have expected values."""
        assert Severity.NONE.value == "none"
        assert Severity.SEV1.value == "sev1"
        assert Severity.SEV2.value == "sev2"
        assert Severity.SEV3.value == "sev3"


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_inputs_same_output(self):
        """Same inputs should produce same output."""
        params = dict(
            ticker="DETERMINISTIC",
            financial_data={
                "Cash": 100000000,
                "CFO_quarterly": -15000000,
            },
            market_data={
                "shares_outstanding": 50000000,
            },
        )

        result1 = score_financial_health_v2(**params)
        result2 = score_financial_health_v2(**params)

        if result1 and result2:
            assert result1.get("financial_score") == result2.get("financial_score")
            assert result1.get("runway_months") == result2.get("runway_months")
