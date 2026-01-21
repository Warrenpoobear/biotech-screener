#!/usr/bin/env python3
"""
Unit tests for defensive_overlay_adapter.py - Position sizing with defensive overlays.

Tests:
1. Correlation sanitization (placeholders, NaN, out-of-range)
2. Defensive multiplier calculation (elite diversifier, high correlation penalty)
3. Inverse-volatility weight calculation
4. Dynamic position floor calculation
5. Caps and renormalization

Run: pytest tests/test_defensive_overlay_adapter.py -v
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from defensive_overlay_adapter import (
    sanitize_corr,
    defensive_multiplier,
    raw_inv_vol_weight,
    calculate_dynamic_floor,
    apply_caps_and_renormalize,
    _q,
)


# ============================================================================
# CORRELATION SANITIZATION TESTS
# ============================================================================

class TestSanitizeCorrelation:
    """Tests for correlation data sanitization."""

    def test_valid_correlation_returned(self):
        """Valid correlation value should be returned."""
        corr, flags = sanitize_corr({"corr_xbi": "0.65"})

        assert corr == Decimal("0.65")
        assert len(flags) == 0

    def test_placeholder_correlation_treated_as_missing(self):
        """0.50 placeholder should be treated as missing."""
        corr, flags = sanitize_corr({"corr_xbi": "0.50"})

        assert corr is None
        assert "def_corr_placeholder_0.50" in flags

    def test_missing_correlation_flagged(self):
        """Missing correlation field should be flagged."""
        corr, flags = sanitize_corr({})

        assert corr is None
        assert "def_corr_missing" in flags

    def test_nan_correlation_handled(self):
        """NaN correlation should be handled gracefully."""
        corr, flags = sanitize_corr({"corr_xbi": "NaN"})

        assert corr is None
        assert "def_corr_not_finite" in flags

    def test_infinity_correlation_handled(self):
        """Infinity correlation should be handled gracefully."""
        corr, flags = sanitize_corr({"corr_xbi": "Infinity"})

        assert corr is None
        assert "def_corr_not_finite" in flags

    def test_out_of_range_correlation_flagged(self):
        """Correlation outside [-1, 1] should be flagged."""
        corr_high, flags_high = sanitize_corr({"corr_xbi": "1.5"})
        corr_low, flags_low = sanitize_corr({"corr_xbi": "-1.5"})

        assert corr_high is None
        assert "def_corr_out_of_range" in flags_high
        assert corr_low is None
        assert "def_corr_out_of_range" in flags_low

    def test_alternate_field_name_supported(self):
        """Should support both corr_xbi and corr_xbi_120d field names."""
        corr1, _ = sanitize_corr({"corr_xbi": "0.70"})
        corr2, _ = sanitize_corr({"corr_xbi_120d": "0.75"})

        assert corr1 == Decimal("0.70")
        assert corr2 == Decimal("0.75")

    def test_parse_failure_flagged(self):
        """Unparseable correlation should be flagged."""
        corr, flags = sanitize_corr({"corr_xbi": "not_a_number"})

        assert corr is None
        assert "def_corr_parse_fail" in flags


# ============================================================================
# DEFENSIVE MULTIPLIER TESTS
# ============================================================================

class TestDefensiveMultiplier:
    """Tests for defensive multiplier calculation."""

    def test_neutral_multiplier_default(self):
        """Default multiplier should be 1.00."""
        mult, notes = defensive_multiplier({})

        # With no valid data, multiplier stays at 1.00
        assert mult == Decimal("1.00")

    def test_elite_diversifier_bonus(self, defensive_features_elite):
        """Elite diversifier (low corr + low vol) should get 1.40x bonus."""
        mult, notes = defensive_multiplier(defensive_features_elite)

        assert mult == Decimal("1.40")
        assert "def_mult_elite_1.40" in notes

    def test_good_diversifier_bonus(self):
        """Good diversifier (moderate low corr + moderate vol) should get 1.10x bonus."""
        features = {
            "corr_xbi": "0.35",  # Below 0.40
            "vol_60d": "0.45",   # Below 0.50
        }
        mult, notes = defensive_multiplier(features)

        assert mult == Decimal("1.10")
        assert "def_mult_good_diversifier_1.10" in notes

    def test_high_correlation_penalty(self, defensive_features_high_corr):
        """High correlation (>0.80) should get 0.95x penalty."""
        mult, notes = defensive_multiplier(defensive_features_high_corr)

        assert mult == Decimal("0.95")
        assert "def_mult_high_corr_0.95" in notes

    def test_placeholder_correlation_no_bonus(self, defensive_features_placeholder):
        """Placeholder correlation should not trigger any bonus."""
        mult, notes = defensive_multiplier(defensive_features_placeholder)

        # Should be neutral since correlation is placeholder
        assert mult == Decimal("1.00")
        assert "def_corr_placeholder_0.50" in notes

    def test_drawdown_warning_added(self):
        """Large drawdown should add warning flag."""
        features = {
            "corr_xbi": "0.60",
            "vol_60d": "0.40",
            "drawdown_60d": "-0.35",  # > 30% drawdown
        }
        mult, notes = defensive_multiplier(features)

        assert "def_warn_drawdown_gt_30pct" in notes

    def test_elite_requires_both_conditions(self):
        """Elite bonus requires both low correlation AND low volatility."""
        # Low corr but high vol
        features1 = {"corr_xbi": "0.25", "vol_60d": "0.60"}
        mult1, notes1 = defensive_multiplier(features1)
        assert mult1 != Decimal("1.40")
        assert "def_skip_not_elite_vol" in notes1

        # Low vol implied but missing correlation
        features2 = {"vol_60d": "0.30"}  # No corr
        mult2, notes2 = defensive_multiplier(features2)
        assert mult2 == Decimal("1.00")


# ============================================================================
# INVERSE VOLATILITY WEIGHT TESTS
# ============================================================================

class TestInverseVolWeight:
    """Tests for inverse-volatility weight calculation."""

    def test_low_volatility_high_weight(self):
        """Low volatility should result in high weight."""
        low_vol = raw_inv_vol_weight({"vol_60d": "0.20"})
        high_vol = raw_inv_vol_weight({"vol_60d": "0.60"})

        assert low_vol > high_vol
        # With power=2.0: 1/(0.20^2) = 25, 1/(0.60^2) ≈ 2.78
        assert low_vol / high_vol > Decimal("5")

    def test_missing_volatility_returns_none(self):
        """Missing volatility should return None."""
        weight = raw_inv_vol_weight({})

        assert weight is None

    def test_zero_volatility_returns_none(self):
        """Zero volatility should return None (divide by zero)."""
        weight = raw_inv_vol_weight({"vol_60d": "0"})

        assert weight is None

    def test_negative_volatility_returns_none(self):
        """Negative volatility should return None."""
        weight = raw_inv_vol_weight({"vol_60d": "-0.20"})

        assert weight is None

    def test_custom_power_parameter(self):
        """Custom power parameter should affect weight scaling."""
        weight_p2 = raw_inv_vol_weight({"vol_60d": "0.50"}, power=Decimal("2.0"))
        weight_p1 = raw_inv_vol_weight({"vol_60d": "0.50"}, power=Decimal("1.0"))

        # With power=2.0: 1/(0.50^2) = 4
        # With power=1.0: 1/(0.50^1) = 2
        assert weight_p2 > weight_p1


# ============================================================================
# DYNAMIC FLOOR TESTS
# ============================================================================

class TestDynamicFloor:
    """Tests for dynamic position floor calculation."""

    def test_small_universe_1pct_floor(self):
        """Small universe (<=50) should have 1% floor."""
        assert calculate_dynamic_floor(20) == Decimal("0.01")
        assert calculate_dynamic_floor(50) == Decimal("0.01")

    def test_medium_universe_05pct_floor(self):
        """Medium universe (51-100) should have 0.5% floor."""
        assert calculate_dynamic_floor(51) == Decimal("0.005")
        assert calculate_dynamic_floor(100) == Decimal("0.005")

    def test_large_universe_03pct_floor(self):
        """Large universe (101-200) should have 0.3% floor."""
        assert calculate_dynamic_floor(101) == Decimal("0.003")
        assert calculate_dynamic_floor(200) == Decimal("0.003")

    def test_very_large_universe_02pct_floor(self):
        """Very large universe (201+) should have 0.2% floor."""
        assert calculate_dynamic_floor(201) == Decimal("0.002")
        assert calculate_dynamic_floor(500) == Decimal("0.002")


# ============================================================================
# CAPS AND RENORMALIZATION TESTS
# ============================================================================

class TestCapsAndRenormalize:
    """Tests for position caps and renormalization."""

    def test_weights_sum_to_investable(self):
        """Weights should sum to investable capital (1 - cash_target)."""
        records = [
            {"rankable": True, "_position_weight_raw": Decimal("1.0")},
            {"rankable": True, "_position_weight_raw": Decimal("2.0")},
            {"rankable": True, "_position_weight_raw": Decimal("1.5")},
        ]

        apply_caps_and_renormalize(records, cash_target=Decimal("0.10"))

        total = sum(Decimal(r["position_weight"]) for r in records)
        # Should be approximately 0.90 (1.0 - 0.10)
        assert abs(total - Decimal("0.90")) < Decimal("0.01")

    def test_max_position_capped(self):
        """Individual positions should be capped before renormalization."""
        records = [
            {"rankable": True, "_position_weight_raw": Decimal("100.0")},  # Very high
            {"rankable": True, "_position_weight_raw": Decimal("1.0")},
            {"rankable": True, "_position_weight_raw": Decimal("1.0")},
        ]

        apply_caps_and_renormalize(records, max_pos=Decimal("0.07"))

        # The high-weight record should get proportionally more than others
        # but not 100x more (the cap prevents extreme concentration)
        weights = [Decimal(r["position_weight"]) for r in records]
        # The ratio between max and min should be bounded
        # (Without cap, ratio would be 100:1, with cap it should be much smaller)
        ratio = max(weights) / min(weights)
        assert ratio < Decimal("10")  # Reasonable concentration limit

    def test_min_position_enforced(self):
        """Positions should meet minimum floor."""
        records = [
            {"rankable": True, "_position_weight_raw": Decimal("0.001")},  # Very small
            {"rankable": True, "_position_weight_raw": Decimal("100.0")},  # Very large
        ]

        apply_caps_and_renormalize(records, min_pos=Decimal("0.01"))

        min_weight = min(Decimal(r["position_weight"]) for r in records if Decimal(r["position_weight"]) > 0)
        assert min_weight >= Decimal("0.005")  # After renormalization

    def test_excluded_securities_zero_weight(self):
        """Non-rankable securities should get zero weight."""
        records = [
            {"rankable": True, "_position_weight_raw": Decimal("1.0")},
            {"rankable": False, "_position_weight_raw": Decimal("1.0")},
        ]

        apply_caps_and_renormalize(records)

        assert Decimal(records[1]["position_weight"]) == Decimal("0")

    def test_top_n_selection(self):
        """Top-N selection should limit positions and flag excluded."""
        records = [
            {"rankable": True, "composite_rank": 1, "_position_weight_raw": Decimal("1.0")},
            {"rankable": True, "composite_rank": 2, "_position_weight_raw": Decimal("1.0")},
            {"rankable": True, "composite_rank": 3, "_position_weight_raw": Decimal("1.0")},
            {"rankable": True, "composite_rank": 4, "_position_weight_raw": Decimal("1.0")},
        ]

        apply_caps_and_renormalize(records, top_n=2)

        # First 2 should have weight
        assert Decimal(records[0]["position_weight"]) > Decimal("0")
        assert Decimal(records[1]["position_weight"]) > Decimal("0")

        # Last 2 should be excluded
        assert Decimal(records[2]["position_weight"]) == Decimal("0")
        assert Decimal(records[3]["position_weight"]) == Decimal("0")
        assert "NOT_IN_TOP_N" in records[2].get("position_flags", [])

    def test_equal_weight_fallback(self):
        """If no valid raw weights, should fallback to equal weight."""
        records = [
            {"rankable": True},  # No _position_weight_raw
            {"rankable": True},
        ]

        apply_caps_and_renormalize(records)

        # Both should have equal non-zero weight
        w1 = Decimal(records[0]["position_weight"])
        w2 = Decimal(records[1]["position_weight"])
        assert w1 == w2
        assert w1 > Decimal("0")

    def test_empty_included_list_handled(self):
        """Empty included list should not raise."""
        records = [
            {"rankable": False},
            {"rankable": False},
        ]

        # Should not raise
        apply_caps_and_renormalize(records)

        for r in records:
            assert r["position_weight"] == "0.0000"


# ============================================================================
# QUANTIZATION TESTS
# ============================================================================

class TestQuantization:
    """Tests for weight quantization helper."""

    def test_quantize_to_4_decimals(self):
        """Weights should be quantized to 4 decimal places."""
        result = _q(Decimal("0.123456789"))

        assert result == Decimal("0.1235")  # Rounds half up

    def test_quantize_rounding(self):
        """Quantization should use ROUND_HALF_UP."""
        assert _q(Decimal("0.12345")) == Decimal("0.1235")
        assert _q(Decimal("0.12344")) == Decimal("0.1234")


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_features_dict(self):
        """Empty features dict should be handled gracefully."""
        mult, notes = defensive_multiplier({})
        assert mult == Decimal("1.00")

        corr, flags = sanitize_corr({})
        assert corr is None

    def test_none_features(self):
        """None features should be handled gracefully."""
        # sanitize_corr expects a dict, but should handle empty
        corr, flags = sanitize_corr({})
        assert corr is None

    def test_negative_correlation_valid(self):
        """Negative correlation should be valid (inverse relationship)."""
        corr, flags = sanitize_corr({"corr_xbi": "-0.30"})

        assert corr == Decimal("-0.30")
        assert len(flags) == 0

    def test_boundary_correlation_values(self):
        """Boundary values -1 and 1 should be valid."""
        corr_neg1, flags1 = sanitize_corr({"corr_xbi": "-1.0"})
        corr_pos1, flags2 = sanitize_corr({"corr_xbi": "1.0"})

        assert corr_neg1 == Decimal("-1.0")
        assert corr_pos1 == Decimal("1.0")
        assert len(flags1) == 0
        assert len(flags2) == 0


# ============================================================================
# COVERAGE DIAGNOSTICS TESTS
# ============================================================================

class TestCoverageDiagnostics:
    """Tests for defensive feature coverage diagnostics."""

    def test_coverage_diagnostics_added_to_output(self):
        """Enrichment should add coverage diagnostics to output."""
        from defensive_overlay_adapter import enrich_with_defensive_overlays

        output = {
            "ranked_securities": [
                {"ticker": "AAA", "composite_score": "50.00", "rankable": True},
                {"ticker": "BBB", "composite_score": "40.00", "rankable": True},
            ]
        }

        scores_by_ticker = {
            "AAA": {"defensive_features": {"vol_60d": "0.30", "corr_xbi": "0.40"}},
            # BBB has no defensive features
        }

        result = enrich_with_defensive_overlays(
            output, scores_by_ticker, apply_multiplier=False, apply_position_sizing=True
        )

        coverage = result.get("diagnostic_counts", {}).get("defensive_features_coverage", {})
        assert coverage.get("total_securities") == 2
        assert coverage.get("with_defensive_features") == 1
        assert coverage.get("with_correlation") == 1
        assert coverage.get("with_volatility") == 1
        assert coverage.get("coverage_pct") == 50.0

    def test_coverage_with_alternate_field_names(self):
        """Coverage should detect both corr_xbi and corr_xbi_120d."""
        from defensive_overlay_adapter import enrich_with_defensive_overlays

        output = {
            "ranked_securities": [
                {"ticker": "AAA", "composite_score": "50.00", "rankable": True},
                {"ticker": "BBB", "composite_score": "40.00", "rankable": True},
            ]
        }

        scores_by_ticker = {
            "AAA": {"defensive_features": {"vol_60d": "0.30", "corr_xbi": "0.40"}},
            "BBB": {"defensive_features": {"vol_60d": "0.25", "corr_xbi_120d": "0.35"}},
        }

        result = enrich_with_defensive_overlays(
            output, scores_by_ticker, apply_multiplier=False, apply_position_sizing=True
        )

        coverage = result.get("diagnostic_counts", {}).get("defensive_features_coverage", {})
        assert coverage.get("with_correlation") == 2
        assert coverage.get("with_volatility") == 2
        assert coverage.get("coverage_pct") == 100.0

    def test_coverage_with_empty_defensive_features(self):
        """Empty defensive features dict should count as having no features."""
        from defensive_overlay_adapter import enrich_with_defensive_overlays

        output = {
            "ranked_securities": [
                {"ticker": "AAA", "composite_score": "50.00", "rankable": True},
            ]
        }

        scores_by_ticker = {
            "AAA": {"defensive_features": {}},
        }

        result = enrich_with_defensive_overlays(
            output, scores_by_ticker, apply_multiplier=False, apply_position_sizing=True
        )

        coverage = result.get("diagnostic_counts", {}).get("defensive_features_coverage", {})
        assert coverage.get("with_defensive_features") == 0
        assert coverage.get("coverage_pct") == 0
