#!/usr/bin/env python3
"""
Tests for module_5_diagnostics_v3.py

Tests cover:
- compute_momentum_breakdown invariants
- build_momentum_health JSON schema
- format_momentum_log_lines output
- check_coverage_guardrail warnings
"""

import pytest
from decimal import Decimal

from module_5_diagnostics_v3 import (
    compute_momentum_breakdown,
    build_momentum_health,
    format_momentum_log_lines,
    check_coverage_guardrail,
)


class TestComputeMomentumBreakdown:
    """Tests for compute_momentum_breakdown function."""

    def test_basic_breakdown(self):
        """Should compute basic breakdown correctly."""
        ranked_securities = [
            {"ticker": "ACME", "flags": []},
            {"ticker": "BETA", "flags": ["momentum_missing_prices"]},
            {"ticker": "GAMMA", "flags": []},
        ]
        diagnostic_counts = {
            "momentum_missing_prices": 1,
            "momentum_computed_low_conf": 0,
            "momentum_applied_negative": 1,
            "momentum_applied_positive": 1,
            "momentum_window_20d": 0,
            "momentum_window_60d": 2,
            "momentum_window_120d": 0,
            "momentum_computable": 2,
            "momentum_meaningful": 2,
            "momentum_strong_signal": 1,
            "momentum_strong_and_effective": 1,
            "momentum_source_prices": 2,
            "momentum_source_13f": 0,
        }
        total_rankable = 3

        result = compute_momentum_breakdown(
            ranked_securities, diagnostic_counts, total_rankable
        )

        assert result["missing"] == 1
        assert result["low_conf"] == 0
        assert result["neg"] == 1
        assert result["pos"] == 1
        assert result["total_rankable"] == 3
        assert result["with_data"] == 2
        assert result["applied"] == 2
        # neutral = applied - neg - pos = 2 - 1 - 1 = 0
        assert result["neutral"] == 0

    def test_invariant_applied_equals_with_data_minus_low_conf(self):
        """Invariant: applied = with_data - low_conf."""
        diagnostic_counts = {
            "momentum_missing_prices": 2,
            "momentum_computed_low_conf": 3,
            "momentum_applied_negative": 2,
            "momentum_applied_positive": 3,
            "momentum_window_20d": 2,
            "momentum_window_60d": 5,
            "momentum_window_120d": 1,
            "momentum_computable": 8,
            "momentum_meaningful": 8,
            "momentum_strong_signal": 4,
            "momentum_strong_and_effective": 3,
            "momentum_source_prices": 8,
            "momentum_source_13f": 0,
        }

        result = compute_momentum_breakdown([], diagnostic_counts, 10)

        with_data = 2 + 5 + 1  # 8
        low_conf = 3
        expected_applied = with_data - low_conf  # 5

        assert result["with_data"] == with_data
        assert result["applied"] == expected_applied

    def test_invariant_neutral_nonnegative(self):
        """Invariant: neutral = max(0, applied - neg - pos)."""
        # Case where neg + pos > applied (shouldn't happen, but should be safe)
        diagnostic_counts = {
            "momentum_missing_prices": 5,
            "momentum_computed_low_conf": 3,
            "momentum_applied_negative": 5,
            "momentum_applied_positive": 5,
            "momentum_window_20d": 0,
            "momentum_window_60d": 5,
            "momentum_window_120d": 0,
        }

        result = compute_momentum_breakdown([], diagnostic_counts, 10)

        assert result["neutral"] >= 0

    def test_coverage_pct_calculation(self):
        """Coverage percentage should be computed correctly."""
        diagnostic_counts = {
            "momentum_missing_prices": 0,
            "momentum_computed_low_conf": 0,
            "momentum_applied_negative": 5,
            "momentum_applied_positive": 5,
            "momentum_window_20d": 0,
            "momentum_window_60d": 10,
            "momentum_window_120d": 0,
        }

        result = compute_momentum_breakdown([], diagnostic_counts, 20)

        # applied = 10, total_rankable = 20, coverage = 50%
        assert result["coverage_pct"] == Decimal("50")

    def test_coverage_pct_zero_total_rankable(self):
        """Coverage should be 0 when total_rankable is 0."""
        diagnostic_counts = {
            "momentum_missing_prices": 0,
            "momentum_computed_low_conf": 0,
            "momentum_applied_negative": 0,
            "momentum_applied_positive": 0,
            "momentum_window_20d": 0,
            "momentum_window_60d": 0,
            "momentum_window_120d": 0,
        }

        result = compute_momentum_breakdown([], diagnostic_counts, 0)

        assert result["coverage_pct"] == Decimal("0")

    def test_windows_used_dict(self):
        """Should include windows_used breakdown."""
        diagnostic_counts = {
            "momentum_window_20d": 3,
            "momentum_window_60d": 10,
            "momentum_window_120d": 5,
        }

        result = compute_momentum_breakdown([], diagnostic_counts, 20)

        assert result["windows_used"]["20d"] == 3
        assert result["windows_used"]["60d"] == 10
        assert result["windows_used"]["120d"] == 5

    def test_sources_dict(self):
        """Should include sources breakdown."""
        diagnostic_counts = {
            "momentum_source_prices": 15,
            "momentum_source_13f": 3,
        }

        result = compute_momentum_breakdown([], diagnostic_counts, 20)

        assert result["sources"]["prices"] == 15
        assert result["sources"]["13f"] == 3

    def test_avg_weight_calculation(self):
        """Should calculate average momentum weight."""
        ranked_securities = [
            {"ticker": "A", "flags": [], "effective_weights": {"momentum": "0.15"}},
            {"ticker": "B", "flags": [], "effective_weights": {"momentum": "0.20"}},
            {"ticker": "C", "flags": [], "effective_weights": {"momentum": "0.10"}},
            {"ticker": "D", "flags": ["momentum_missing_prices"], "effective_weights": {}},
        ]
        diagnostic_counts = {}

        result = compute_momentum_breakdown(ranked_securities, diagnostic_counts, 4)

        # Average of 0.15, 0.20, 0.10 = 0.45 / 3 = 0.15
        assert result["avg_weight"] == Decimal("0.150")

    def test_avg_weight_excludes_missing_and_low_conf(self):
        """Average weight should exclude securities with missing/low_conf flags."""
        ranked_securities = [
            {"ticker": "A", "flags": [], "effective_weights": {"momentum": "0.20"}},
            {"ticker": "B", "flags": ["momentum_missing_prices"], "effective_weights": {"momentum": "0.50"}},
            {"ticker": "C", "flags": ["momentum_low_confidence"], "effective_weights": {"momentum": "0.50"}},
            {"ticker": "D", "flags": [], "effective_weights": {"momentum": "0.10"}},
        ]
        diagnostic_counts = {}

        result = compute_momentum_breakdown(ranked_securities, diagnostic_counts, 4)

        # Only A and D should be included: (0.20 + 0.10) / 2 = 0.15
        assert result["avg_weight"] == Decimal("0.150")


class TestBuildMomentumHealth:
    """Tests for build_momentum_health function."""

    def test_basic_health_output(self):
        """Should build momentum health dict."""
        breakdown = {
            "applied": 10,
            "coverage_pct": Decimal("50.0"),
            "computable": 12,
            "meaningful": 10,
            "strong_signal": 5,
            "strong_and_effective": 4,
            "avg_weight": Decimal("0.150"),
            "windows_used": {"20d": 2, "60d": 8, "120d": 0},
            "sources": {"prices": 10, "13f": 0},
            "total_rankable": 20,
        }
        as_of_date = "2026-01-15"

        result = build_momentum_health(breakdown, as_of_date)

        assert result["coverage_applied"] == 10
        assert result["coverage_pct"] == 50.0
        assert result["computable"] == 12
        assert result["meaningful"] == 10
        assert result["strong_signal"] == 5
        assert result["strong_and_effective"] == 4
        assert result["avg_weight"] == "0.150"
        assert result["windows_used"] == {"20d": 2, "60d": 8, "120d": 0}
        assert result["sources"] == {"prices": 10, "13f": 0}
        assert result["total_rankable"] == 20
        assert result["as_of_date"] == "2026-01-15"

    def test_coverage_pct_rounded(self):
        """Coverage pct should be rounded to 1 decimal."""
        breakdown = {
            "applied": 10,
            "coverage_pct": Decimal("33.333333"),
            "computable": 10,
            "meaningful": 10,
            "strong_signal": 5,
            "strong_and_effective": 4,
            "avg_weight": Decimal("0.150"),
            "windows_used": {"20d": 0, "60d": 10, "120d": 0},
            "sources": {"prices": 10, "13f": 0},
            "total_rankable": 30,
        }

        result = build_momentum_health(breakdown, "2026-01-15")

        assert result["coverage_pct"] == 33.3


class TestFormatMomentumLogLines:
    """Tests for format_momentum_log_lines function."""

    def test_returns_two_lines(self):
        """Should return exactly two log lines."""
        breakdown = {
            "total_rankable": 50,
            "missing": 5,
            "low_conf": 3,
            "neg": 10,
            "pos": 15,
            "neutral": 17,
            "applied": 42,
            "windows_used": {"20d": 5, "60d": 30, "120d": 7},
            "coverage_pct": Decimal("84.0"),
            "avg_weight": Decimal("0.150"),
            "computable": 45,
            "meaningful": 42,
            "strong_signal": 20,
            "strong_and_effective": 18,
            "sources": {"prices": 42, "13f": 0},
        }

        lines = format_momentum_log_lines(breakdown)

        assert len(lines) == 2

    def test_line_contains_missing_count(self):
        """First line should contain missing count."""
        breakdown = {
            "total_rankable": 50,
            "missing": 7,
            "low_conf": 3,
            "neg": 10,
            "pos": 15,
            "neutral": 15,
            "applied": 40,
            "windows_used": {"20d": 5, "60d": 30, "120d": 5},
            "coverage_pct": Decimal("80.0"),
            "avg_weight": Decimal("0.150"),
            "computable": 43,
            "meaningful": 40,
            "strong_signal": 20,
            "strong_and_effective": 18,
            "sources": {"prices": 40, "13f": 0},
        }

        lines = format_momentum_log_lines(breakdown)

        assert "missing:7" in lines[0]

    def test_line_contains_window_breakdown(self):
        """First line should contain window breakdown."""
        breakdown = {
            "total_rankable": 50,
            "missing": 5,
            "low_conf": 3,
            "neg": 10,
            "pos": 15,
            "neutral": 17,
            "applied": 42,
            "windows_used": {"20d": 5, "60d": 30, "120d": 7},
            "coverage_pct": Decimal("84.0"),
            "avg_weight": Decimal("0.150"),
            "computable": 45,
            "meaningful": 42,
            "strong_signal": 20,
            "strong_and_effective": 18,
            "sources": {"prices": 42, "13f": 0},
        }

        lines = format_momentum_log_lines(breakdown)

        assert "20d:5" in lines[0]
        assert "60d:30" in lines[0]
        assert "120d:7" in lines[0]

    def test_stable_metrics_line(self):
        """Second line should contain stable metrics."""
        breakdown = {
            "total_rankable": 50,
            "missing": 5,
            "low_conf": 3,
            "neg": 10,
            "pos": 15,
            "neutral": 17,
            "applied": 42,
            "windows_used": {"20d": 5, "60d": 30, "120d": 7},
            "coverage_pct": Decimal("84.0"),
            "avg_weight": Decimal("0.150"),
            "computable": 45,
            "meaningful": 42,
            "strong_signal": 20,
            "strong_and_effective": 18,
            "sources": {"prices": 42, "13f": 0},
        }

        lines = format_momentum_log_lines(breakdown)

        assert "computable:45" in lines[1]
        assert "meaningful:42" in lines[1]
        assert "strong:20" in lines[1]


class TestCheckCoverageGuardrail:
    """Tests for check_coverage_guardrail function."""

    def test_no_warning_above_threshold(self):
        """No warning when coverage is above threshold."""
        breakdown = {
            "total_rankable": 100,
            "coverage_pct": Decimal("50.0"),
        }

        result = check_coverage_guardrail(breakdown, threshold_pct=20.0)

        assert result is None

    def test_warning_below_threshold(self):
        """Warning when coverage is below threshold."""
        breakdown = {
            "total_rankable": 100,
            "coverage_pct": Decimal("15.0"),
        }

        result = check_coverage_guardrail(breakdown, threshold_pct=20.0)

        assert result is not None
        assert "WARN" in result
        assert "15.0%" in result

    def test_no_warning_when_zero_total_rankable(self):
        """No warning when total_rankable is 0 (no universe)."""
        breakdown = {
            "total_rankable": 0,
            "coverage_pct": Decimal("0"),
        }

        result = check_coverage_guardrail(breakdown, threshold_pct=20.0)

        assert result is None

    def test_custom_threshold(self):
        """Should respect custom threshold."""
        breakdown = {
            "total_rankable": 100,
            "coverage_pct": Decimal("30.0"),
        }

        # Should pass with 20% threshold
        result_pass = check_coverage_guardrail(breakdown, threshold_pct=20.0)
        assert result_pass is None

        # Should fail with 40% threshold
        result_fail = check_coverage_guardrail(breakdown, threshold_pct=40.0)
        assert result_fail is not None


class TestDeterminism:
    """Tests ensuring all functions are deterministic."""

    def test_breakdown_deterministic(self):
        """compute_momentum_breakdown should be deterministic."""
        ranked_securities = [
            {"ticker": "A", "flags": [], "effective_weights": {"momentum": "0.15"}},
            {"ticker": "B", "flags": [], "effective_weights": {"momentum": "0.20"}},
        ]
        diagnostic_counts = {
            "momentum_missing_prices": 2,
            "momentum_computed_low_conf": 1,
            "momentum_applied_negative": 3,
            "momentum_applied_positive": 4,
            "momentum_window_20d": 1,
            "momentum_window_60d": 5,
            "momentum_window_120d": 2,
        }

        result1 = compute_momentum_breakdown(ranked_securities, diagnostic_counts, 10)
        result2 = compute_momentum_breakdown(ranked_securities, diagnostic_counts, 10)

        assert result1 == result2

    def test_health_output_deterministic(self):
        """build_momentum_health should be deterministic."""
        breakdown = {
            "applied": 10,
            "coverage_pct": Decimal("50.0"),
            "computable": 12,
            "meaningful": 10,
            "strong_signal": 5,
            "strong_and_effective": 4,
            "avg_weight": Decimal("0.150"),
            "windows_used": {"20d": 2, "60d": 8, "120d": 0},
            "sources": {"prices": 10, "13f": 0},
            "total_rankable": 20,
        }

        result1 = build_momentum_health(breakdown, "2026-01-15")
        result2 = build_momentum_health(breakdown, "2026-01-15")

        assert result1 == result2
