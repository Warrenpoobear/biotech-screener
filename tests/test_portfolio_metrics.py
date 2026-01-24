#!/usr/bin/env python3
"""
Tests for backtest/portfolio_metrics.py

Tests cover:
- Dataclass construction and serialization
- Utility functions (_to_decimal, _quantize)
- DrawdownEvent and DrawdownAnalysis
- RiskMetrics and PerformanceSummary
"""

import pytest
from decimal import Decimal

from backtest.portfolio_metrics import (
    DrawdownEvent,
    DrawdownAnalysis,
    RiskMetrics,
    PerformanceSummary,
    _to_decimal,
    _quantize,
)


class TestDrawdownEvent:
    """Tests for DrawdownEvent dataclass."""

    def test_create_recovered_drawdown(self):
        """Should create a recovered drawdown event."""
        event = DrawdownEvent(
            start_date="2025-03-01",
            trough_date="2025-03-15",
            recovery_date="2025-04-01",
            peak_value=Decimal("100.00"),
            trough_value=Decimal("80.00"),
            drawdown_pct=Decimal("-20.00"),
            duration_days=14,
            recovery_days=17,
            is_recovered=True,
            regime_at_start="BULL",
        )
        assert event.start_date == "2025-03-01"
        assert event.is_recovered is True
        assert event.drawdown_pct == Decimal("-20.00")

    def test_create_unrecovered_drawdown(self):
        """Should create an unrecovered drawdown event."""
        event = DrawdownEvent(
            start_date="2025-06-01",
            trough_date="2025-06-20",
            recovery_date=None,
            peak_value=Decimal("150.00"),
            trough_value=Decimal("120.00"),
            drawdown_pct=Decimal("-20.00"),
            duration_days=19,
            recovery_days=None,
            is_recovered=False,
            regime_at_start="BEAR",
        )
        assert event.recovery_date is None
        assert event.is_recovered is False

    def test_to_dict(self):
        """Should serialize to dict correctly."""
        event = DrawdownEvent(
            start_date="2025-03-01",
            trough_date="2025-03-15",
            recovery_date="2025-04-01",
            peak_value=Decimal("100.00"),
            trough_value=Decimal("85.00"),
            drawdown_pct=Decimal("-15.00"),
            duration_days=14,
            recovery_days=17,
            is_recovered=True,
            regime_at_start="NEUTRAL",
        )

        result = event.to_dict()

        assert result["start_date"] == "2025-03-01"
        assert result["trough_date"] == "2025-03-15"
        assert result["recovery_date"] == "2025-04-01"
        assert result["peak_value"] == "100.00"
        assert result["trough_value"] == "85.00"
        assert result["drawdown_pct"] == "-15.00"
        assert result["duration_days"] == 14
        assert result["recovery_days"] == 17
        assert result["is_recovered"] is True
        assert result["regime_at_start"] == "NEUTRAL"


class TestDrawdownAnalysis:
    """Tests for DrawdownAnalysis dataclass."""

    def test_create_analysis(self):
        """Should create drawdown analysis."""
        event = DrawdownEvent(
            start_date="2025-03-01",
            trough_date="2025-03-15",
            recovery_date="2025-04-01",
            peak_value=Decimal("100.00"),
            trough_value=Decimal("85.00"),
            drawdown_pct=Decimal("-15.00"),
            duration_days=14,
            recovery_days=17,
            is_recovered=True,
            regime_at_start="BULL",
        )

        analysis = DrawdownAnalysis(
            max_drawdown_pct=Decimal("-15.00"),
            max_drawdown_event=event,
            avg_drawdown_pct=Decimal("-10.00"),
            avg_recovery_days=15,
            drawdown_events=[event],
            current_drawdown_pct=Decimal("0"),
            days_since_peak=0,
            regime_drawdowns={"BULL": [event]},
        )

        assert analysis.max_drawdown_pct == Decimal("-15.00")
        assert analysis.avg_recovery_days == 15
        assert len(analysis.drawdown_events) == 1

    def test_to_dict(self):
        """Should serialize to dict."""
        event = DrawdownEvent(
            start_date="2025-03-01",
            trough_date="2025-03-15",
            recovery_date="2025-04-01",
            peak_value=Decimal("100.00"),
            trough_value=Decimal("85.00"),
            drawdown_pct=Decimal("-15.00"),
            duration_days=14,
            recovery_days=17,
            is_recovered=True,
            regime_at_start="BULL",
        )

        analysis = DrawdownAnalysis(
            max_drawdown_pct=Decimal("-15.00"),
            max_drawdown_event=event,
            avg_drawdown_pct=Decimal("-15.00"),
            avg_recovery_days=17,
            drawdown_events=[event],
            current_drawdown_pct=Decimal("-5.00"),
            days_since_peak=10,
            regime_drawdowns={"BULL": [event]},
        )

        result = analysis.to_dict()

        assert result["max_drawdown_pct"] == "-15.00"
        assert result["n_drawdown_events"] == 1
        assert result["current_drawdown_pct"] == "-5.00"
        assert result["days_since_peak"] == 10
        assert "BULL" in result["regime_summary"]


class TestRiskMetrics:
    """Tests for RiskMetrics dataclass."""

    def test_create_risk_metrics(self):
        """Should create risk metrics."""
        metrics = RiskMetrics(
            sharpe_ratio=Decimal("1.25"),
            sortino_ratio=Decimal("1.80"),
            calmar_ratio=Decimal("2.50"),
            volatility_annualized=Decimal("0.18"),
            downside_deviation=Decimal("0.10"),
            var_95=Decimal("-0.05"),
            cvar_95=Decimal("-0.08"),
        )
        assert metrics.sharpe_ratio == Decimal("1.25")
        assert metrics.sortino_ratio == Decimal("1.80")

    def test_create_with_none_values(self):
        """Should allow None values."""
        metrics = RiskMetrics(
            sharpe_ratio=None,
            sortino_ratio=None,
            calmar_ratio=None,
            volatility_annualized=None,
            downside_deviation=None,
            var_95=None,
            cvar_95=None,
        )
        assert metrics.sharpe_ratio is None

    def test_to_dict(self):
        """Should serialize to dict."""
        metrics = RiskMetrics(
            sharpe_ratio=Decimal("1.25"),
            sortino_ratio=Decimal("1.80"),
            calmar_ratio=Decimal("2.50"),
            volatility_annualized=Decimal("0.18"),
            downside_deviation=Decimal("0.10"),
            var_95=Decimal("-0.05"),
            cvar_95=Decimal("-0.08"),
        )

        result = metrics.to_dict()

        assert result["sharpe_ratio"] == "1.25"
        assert result["sortino_ratio"] == "1.80"
        assert result["calmar_ratio"] == "2.50"
        assert result["volatility_annualized"] == "0.18"
        assert result["var_95"] == "-0.05"

    def test_to_dict_with_none(self):
        """Should handle None values in to_dict."""
        metrics = RiskMetrics(
            sharpe_ratio=None,
            sortino_ratio=Decimal("1.50"),
            calmar_ratio=None,
            volatility_annualized=Decimal("0.20"),
            downside_deviation=None,
            var_95=None,
            cvar_95=None,
        )

        result = metrics.to_dict()

        assert result["sharpe_ratio"] is None
        assert result["sortino_ratio"] == "1.50"


class TestToDecimal:
    """Tests for _to_decimal utility function."""

    def test_none_returns_default(self):
        """None should return default value."""
        assert _to_decimal(None) is None
        assert _to_decimal(None, Decimal("0")) == Decimal("0")

    def test_decimal_passthrough(self):
        """Decimal should pass through unchanged."""
        val = Decimal("123.45")
        assert _to_decimal(val) == val

    def test_int_conversion(self):
        """Int should convert to Decimal."""
        assert _to_decimal(42) == Decimal("42")

    def test_float_conversion(self):
        """Float should convert to Decimal via string."""
        result = _to_decimal(3.14)
        assert result == Decimal("3.14")

    def test_string_conversion(self):
        """String should convert to Decimal."""
        assert _to_decimal("99.99") == Decimal("99.99")

    def test_invalid_returns_default(self):
        """Invalid value should return default."""
        assert _to_decimal("not a number", Decimal("0")) == Decimal("0")


class TestQuantize:
    """Tests for _quantize utility function."""

    def test_default_precision(self):
        """Default precision should be 0.01."""
        result = _quantize(Decimal("123.456789"))
        assert result == Decimal("123.46")

    def test_custom_precision(self):
        """Should respect custom precision."""
        result = _quantize(Decimal("123.456789"), "0.001")
        assert result == Decimal("123.457")

    def test_rounds_half_up(self):
        """Should round half up."""
        result = _quantize(Decimal("0.125"), "0.01")
        assert result == Decimal("0.13")


class TestDeterminism:
    """Tests ensuring all functions are deterministic."""

    def test_to_decimal_deterministic(self):
        """_to_decimal should be deterministic."""
        results = [_to_decimal("123.456") for _ in range(5)]
        assert all(r == results[0] for r in results)

    def test_quantize_deterministic(self):
        """_quantize should be deterministic."""
        results = [_quantize(Decimal("123.456789")) for _ in range(5)]
        assert all(r == results[0] for r in results)

    def test_drawdown_event_to_dict_deterministic(self):
        """DrawdownEvent.to_dict should be deterministic."""
        event = DrawdownEvent(
            start_date="2025-03-01",
            trough_date="2025-03-15",
            recovery_date="2025-04-01",
            peak_value=Decimal("100.00"),
            trough_value=Decimal("85.00"),
            drawdown_pct=Decimal("-15.00"),
            duration_days=14,
            recovery_days=17,
            is_recovered=True,
            regime_at_start="BULL",
        )

        results = [event.to_dict() for _ in range(5)]
        assert all(r == results[0] for r in results)
