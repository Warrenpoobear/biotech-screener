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
        assert "def_mult_good_1.10" in notes

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
# MULTI-FACTOR FEATURE TESTS (v2.0)
# ============================================================================

class TestMultiFactorFeatures:
    """Tests for multi-factor defensive multiplier (momentum, RSI, drawdown)."""

    def test_momentum_bonus(self):
        """Strong positive momentum (>10% 21d return) should get bonus."""
        features = {
            "corr_xbi": "0.50",  # Neutral correlation
            "vol_60d": "0.50",
            "ret_21d": "0.15",  # 15% 21-day return
        }
        mult, notes = defensive_multiplier(features)

        assert mult == Decimal("1.05")
        assert "def_mult_momentum_bonus_1.05" in notes

    def test_momentum_penalty(self):
        """Large negative momentum (<-20%) should get penalty."""
        features = {
            "corr_xbi": "0.50",
            "vol_60d": "0.50",
            "ret_21d": "-0.25",  # -25% 21-day return
        }
        mult, notes = defensive_multiplier(features)

        assert mult == Decimal("0.95")
        assert "def_mult_momentum_penalty_0.95" in notes

    def test_rsi_oversold_bonus(self):
        """RSI < 30 (oversold) should get small bonus."""
        features = {
            "corr_xbi": "0.50",
            "vol_60d": "0.50",
            "rsi_14d": "25",  # Oversold
        }
        mult, notes = defensive_multiplier(features)

        assert mult == Decimal("1.03")
        assert "def_mult_rsi_oversold_1.03" in notes

    def test_rsi_overbought_penalty(self):
        """RSI > 70 (overbought) should get small penalty."""
        features = {
            "corr_xbi": "0.50",
            "vol_60d": "0.50",
            "rsi_14d": "75",  # Overbought
        }
        mult, notes = defensive_multiplier(features)

        assert mult == Decimal("0.98")
        assert "def_mult_rsi_overbought_0.98" in notes

    def test_drawdown_penalty(self):
        """Deep drawdown (<-40%) should get penalty (not just warning)."""
        features = {
            "corr_xbi": "0.50",
            "vol_60d": "0.50",
            "drawdown_current": "-0.45",  # -45% drawdown
        }
        mult, notes = defensive_multiplier(features)

        assert mult == Decimal("0.92")
        assert "def_mult_drawdown_penalty_0.92" in notes

    def test_stacking_multiple_factors(self):
        """Multiple factors should stack multiplicatively."""
        features = {
            "corr_xbi": "0.25",  # Elite corr
            "vol_60d": "0.35",   # Elite vol
            "ret_21d": "0.15",   # Momentum bonus
            "rsi_14d": "28",     # Oversold bonus
        }
        mult, notes = defensive_multiplier(features)

        # 1.40 * 1.05 * 1.03 = 1.5141
        expected = Decimal("1.40") * Decimal("1.05") * Decimal("1.03")
        assert mult == expected
        assert "def_mult_elite_1.40" in notes
        assert "def_mult_momentum_bonus_1.05" in notes
        assert "def_mult_rsi_oversold_1.03" in notes

    def test_high_vol_penalty(self):
        """High volatility (>80% ann) should get penalty regardless of correlation."""
        features = {
            "corr_xbi": "0.50",
            "vol_60d": "0.90",  # Very high vol
        }
        mult, notes = defensive_multiplier(features)

        assert mult == Decimal("0.97")
        assert "def_mult_high_vol_0.97" in notes

    def test_config_can_disable_factors(self):
        """Config flags should disable specific factors."""
        from defensive_overlay_adapter import DefensiveConfig

        # Create config with momentum disabled
        config = DefensiveConfig(enable_momentum=False, enable_rsi=False)

        features = {
            "corr_xbi": "0.50",
            "vol_60d": "0.50",
            "ret_21d": "0.20",  # Would trigger momentum bonus
            "rsi_14d": "25",    # Would trigger RSI bonus
        }
        mult, notes = defensive_multiplier(features, config=config)

        # No momentum or RSI bonuses should apply
        assert mult == Decimal("1.00")
        assert "def_mult_momentum" not in str(notes)
        assert "def_mult_rsi" not in str(notes)

    def test_multiplier_clamping_ceiling(self):
        """Extreme stacking should be clamped to ceiling."""
        from defensive_overlay_adapter import DefensiveConfig

        # Config with low ceiling to test clamping
        config = DefensiveConfig(mult_ceiling=Decimal("1.20"))

        features = {
            "corr_xbi": "0.25",  # Elite corr
            "vol_60d": "0.35",   # Elite vol → 1.40x
            "ret_21d": "0.15",   # Momentum → 1.05x
        }
        mult, notes = defensive_multiplier(features, config=config)

        # Would be 1.40 * 1.05 = 1.47, but clamped to 1.20
        assert mult == Decimal("1.20")
        assert any("def_mult_clamped" in n for n in notes)

    def test_multiplier_clamping_floor(self):
        """Extreme penalties should be clamped to floor."""
        from defensive_overlay_adapter import DefensiveConfig

        # Config with high floor to test clamping
        config = DefensiveConfig(mult_floor=Decimal("0.90"))

        features = {
            "corr_xbi": "0.85",       # High corr penalty → 0.95x
            "ret_21d": "-0.25",       # Momentum penalty → 0.95x
            "drawdown_current": "-0.45",  # Drawdown penalty → 0.92x
        }
        mult, notes = defensive_multiplier(features, config=config)

        # Would be 0.95 * 0.95 * 0.92 = 0.830, but clamped to 0.90
        assert mult == Decimal("0.90")
        assert any("def_mult_clamped" in n for n in notes)

    def test_config_provenance(self):
        """Config should provide provenance for audit trail."""
        from defensive_overlay_adapter import DefensiveConfig, DEFAULT_DEFENSIVE_CONFIG

        prov = DEFAULT_DEFENSIVE_CONFIG.to_provenance()

        assert prov["config_id"] == "default"
        assert prov["config_version"] == "2.0.0"
        assert "config_hash" in prov
        assert len(prov["config_hash"]) == 8  # 8-char hex hash
        assert "mult_bounds" in prov
        assert prov["enabled_factors"]["momentum"] is True
        assert prov["enabled_factors"]["vol_ratio"] is False


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
        # New format: by_feature breakdown
        by_feature = coverage.get("by_feature", {})
        assert by_feature.get("corr_xbi_120d", {}).get("count") == 1
        assert by_feature.get("vol_60d", {}).get("count") == 1
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
        # New format: by_feature breakdown (corr_xbi aliased to corr_xbi_120d)
        by_feature = coverage.get("by_feature", {})
        assert by_feature.get("corr_xbi_120d", {}).get("count") == 2
        assert by_feature.get("vol_60d", {}).get("count") == 2
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


# ============================================================================
# STRUCTURED TAGS AND AUDIT FEATURES TESTS
# ============================================================================

class TestStructuredTags:
    """Tests for machine-safe defensive tags extraction."""

    def test_elite_tag_extracted(self):
        """Elite multiplier note should produce 'elite' tag."""
        from defensive_overlay_adapter import _extract_defensive_tags

        notes = ["def_mult_elite_1.40"]
        tags = _extract_defensive_tags(notes)

        assert "elite" in tags
        assert len(tags) == 1

    def test_multiple_tags_extracted(self):
        """Multiple notes should produce multiple tags."""
        from defensive_overlay_adapter import _extract_defensive_tags

        notes = ["def_skip_not_elite_vol", "def_mult_high_vol_0.97", "def_mult_momentum_bonus_1.05"]
        tags = _extract_defensive_tags(notes)

        assert "not_elite" in tags
        assert "high_vol_penalty" in tags
        assert "momentum_bonus" in tags
        assert len(tags) == 3

    def test_warning_tag_extracted(self):
        """Warning notes should produce warning tags."""
        from defensive_overlay_adapter import _extract_defensive_tags

        notes = ["def_warn_drawdown_gt_30pct"]
        tags = _extract_defensive_tags(notes)

        assert "drawdown_warning" in tags

    def test_empty_notes_produce_empty_tags(self):
        """Empty notes list should produce empty tags list."""
        from defensive_overlay_adapter import _extract_defensive_tags

        tags = _extract_defensive_tags([])

        assert tags == []

    def test_unknown_notes_ignored(self):
        """Unknown note patterns should be ignored."""
        from defensive_overlay_adapter import _extract_defensive_tags

        notes = ["some_unknown_note", "another_random_note"]
        tags = _extract_defensive_tags(notes)

        assert tags == []


class TestAuditFeatures:
    """Tests for defensive features audit extraction."""

    def test_all_features_extracted(self):
        """All key features should be extracted when present."""
        from defensive_overlay_adapter import _extract_audit_features

        features = {
            "corr_xbi": "0.25",
            "vol_60d": "0.35",
            "rsi_14d": "45",
            "ret_21d": "0.10",
            "drawdown_current": "-0.15",
        }

        audit = _extract_audit_features(features)

        assert audit["corr_xbi"] == "0.25"
        assert audit["vol_60d"] == "0.35"
        assert audit["rsi_14d"] == "45"
        assert audit["ret_21d"] == "0.10"
        assert audit["drawdown"] == "-0.15"

    def test_alias_corr_xbi_120d_works(self):
        """corr_xbi_120d alias should be extracted as corr_xbi."""
        from defensive_overlay_adapter import _extract_audit_features

        features = {"corr_xbi_120d": "0.30"}
        audit = _extract_audit_features(features)

        assert audit["corr_xbi"] == "0.30"

    def test_alias_drawdown_60d_works(self):
        """drawdown_60d alias should be extracted as drawdown."""
        from defensive_overlay_adapter import _extract_audit_features

        features = {"drawdown_60d": "-0.25"}
        audit = _extract_audit_features(features)

        assert audit["drawdown"] == "-0.25"

    def test_missing_features_return_none(self):
        """Missing features should return None values."""
        from defensive_overlay_adapter import _extract_audit_features

        features = {"vol_60d": "0.40"}
        audit = _extract_audit_features(features)

        assert audit["corr_xbi"] is None
        assert audit["vol_60d"] == "0.40"
        assert audit["rsi_14d"] is None
        assert audit["ret_21d"] is None
        assert audit["drawdown"] is None

    def test_empty_features_all_none(self):
        """Empty features dict should produce all None values."""
        from defensive_overlay_adapter import _extract_audit_features

        audit = _extract_audit_features({})

        assert all(v is None for v in audit.values())


class TestEnrichOutputFields:
    """Tests for new fields added to enriched output."""

    def test_tags_added_to_output(self):
        """defensive_tags should be added to each record."""
        from defensive_overlay_adapter import enrich_with_defensive_overlays

        output = {
            "ranked_securities": [
                {"ticker": "AAA", "composite_score": "50.00", "rankable": True},
            ]
        }

        scores_by_ticker = {
            "AAA": {"defensive_features": {"corr_xbi": "0.25", "vol_60d": "0.35"}},
        }

        result = enrich_with_defensive_overlays(
            output, scores_by_ticker, apply_multiplier=True
        )

        rec = result["ranked_securities"][0]
        assert "defensive_tags" in rec
        assert "elite" in rec["defensive_tags"]

    def test_features_added_to_output(self):
        """defensive_features should be added to each record."""
        from defensive_overlay_adapter import enrich_with_defensive_overlays

        output = {
            "ranked_securities": [
                {"ticker": "AAA", "composite_score": "50.00", "rankable": True},
            ]
        }

        scores_by_ticker = {
            "AAA": {"defensive_features": {"corr_xbi": "0.28", "vol_60d": "0.32", "ret_21d": "0.05"}},
        }

        result = enrich_with_defensive_overlays(
            output, scores_by_ticker, apply_multiplier=True
        )

        rec = result["ranked_securities"][0]
        assert "defensive_features" in rec
        assert rec["defensive_features"]["corr_xbi"] == "0.28"
        assert rec["defensive_features"]["vol_60d"] == "0.32"
        assert rec["defensive_features"]["ret_21d"] == "0.05"


# ============================================================================
# NEW: FIELDS ALWAYS PRESENT TESTS
# ============================================================================

class TestFieldsAlwaysPresent:
    """Tests for fields present even when apply_multiplier=False."""

    def test_fields_present_when_multiplier_disabled(self):
        """All defensive fields should be present even when multiplier disabled."""
        from defensive_overlay_adapter import enrich_with_defensive_overlays

        output = {
            "ranked_securities": [
                {"ticker": "AAA", "composite_score": "50.00", "composite_rank": 1, "rankable": True},
            ]
        }

        scores_by_ticker = {
            "AAA": {"defensive_features": {"corr_xbi": "0.25", "vol_60d": "0.35"}},
        }

        result = enrich_with_defensive_overlays(
            output, scores_by_ticker, apply_multiplier=False
        )

        rec = result["ranked_securities"][0]
        # All fields present
        assert "defensive_multiplier" in rec
        assert "defensive_notes" in rec
        assert "defensive_tags" in rec
        assert "defensive_features" in rec
        assert "defensive_bucket" in rec
        assert "risk_adjusted_score" in rec
        # Multiplier is 1.00 (not applied)
        assert rec["defensive_multiplier"] == "1.00"
        # Note indicates not applied
        assert "def_not_applied" in rec["defensive_notes"]
        # Tag indicates disabled
        assert "multiplier_disabled" in rec["defensive_tags"]

    def test_ranks_unchanged_when_multiplier_disabled(self):
        """Ranks should not change when apply_multiplier=False."""
        from defensive_overlay_adapter import enrich_with_defensive_overlays

        output = {
            "ranked_securities": [
                {"ticker": "AAA", "composite_score": "50.00", "composite_rank": 1, "rankable": True},
                {"ticker": "BBB", "composite_score": "60.00", "composite_rank": 2, "rankable": True},
            ]
        }

        scores_by_ticker = {
            "AAA": {"defensive_features": {"corr_xbi": "0.25", "vol_60d": "0.35"}},  # Would be elite
            "BBB": {"defensive_features": {"corr_xbi": "0.85", "vol_60d": "0.90"}},  # Would be penalty
        }

        result = enrich_with_defensive_overlays(
            output, scores_by_ticker, apply_multiplier=False
        )

        # Ranks unchanged
        assert result["ranked_securities"][0]["composite_rank"] == 1
        assert result["ranked_securities"][1]["composite_rank"] == 2
        # Scores unchanged
        assert result["ranked_securities"][0]["composite_score"] == "50.00"
        assert result["ranked_securities"][1]["composite_score"] == "60.00"

    def test_risk_adjusted_score_computed_when_disabled(self):
        """risk_adjusted_score should be computed even when multiplier disabled."""
        from defensive_overlay_adapter import enrich_with_defensive_overlays
        from decimal import Decimal

        output = {
            "ranked_securities": [
                {"ticker": "AAA", "composite_score": "50.00", "rankable": True},
            ]
        }

        scores_by_ticker = {
            "AAA": {"defensive_features": {"corr_xbi": "0.25", "vol_60d": "0.35"}},  # Elite = 1.40x
        }

        result = enrich_with_defensive_overlays(
            output, scores_by_ticker, apply_multiplier=False
        )

        rec = result["ranked_securities"][0]
        # risk_adjusted_score = 50 * 1.40 = 70
        assert Decimal(rec["risk_adjusted_score"]) == Decimal("70.00")
        # But composite_score unchanged
        assert rec["composite_score"] == "50.00"


class TestDefensiveBucket:
    """Tests for defensive_bucket derivation."""

    def test_elite_bucket(self):
        """Multiplier > 1.20 should be 'elite' bucket."""
        from defensive_overlay_adapter import _derive_defensive_bucket
        from decimal import Decimal

        assert _derive_defensive_bucket(Decimal("1.40")) == "elite"
        assert _derive_defensive_bucket(Decimal("1.21")) == "elite"

    def test_good_bucket(self):
        """Multiplier 1.05 < m <= 1.20 should be 'good' bucket."""
        from defensive_overlay_adapter import _derive_defensive_bucket
        from decimal import Decimal

        assert _derive_defensive_bucket(Decimal("1.10")) == "good"
        assert _derive_defensive_bucket(Decimal("1.20")) == "good"
        assert _derive_defensive_bucket(Decimal("1.06")) == "good"

    def test_penalty_bucket(self):
        """Multiplier < 0.98 should be 'penalty' bucket."""
        from defensive_overlay_adapter import _derive_defensive_bucket
        from decimal import Decimal

        assert _derive_defensive_bucket(Decimal("0.95")) == "penalty"
        assert _derive_defensive_bucket(Decimal("0.75")) == "penalty"
        assert _derive_defensive_bucket(Decimal("0.97")) == "penalty"

    def test_neutral_bucket(self):
        """Multiplier 0.98 <= m <= 1.05 should be 'neutral' bucket."""
        from defensive_overlay_adapter import _derive_defensive_bucket
        from decimal import Decimal

        assert _derive_defensive_bucket(Decimal("1.00")) == "neutral"
        assert _derive_defensive_bucket(Decimal("0.98")) == "neutral"
        assert _derive_defensive_bucket(Decimal("1.05")) == "neutral"


class TestNullEquivalentDetection:
    """Tests for consistent null/placeholder detection in coverage."""

    def test_empty_string_not_counted_as_valid(self):
        """Empty string should not be counted as valid coverage."""
        from defensive_overlay_adapter import _is_valid_value

        assert _is_valid_value("") is False
        assert _is_valid_value("  ") is False

    def test_na_variants_not_counted(self):
        """N/A, Unknown, NaN, etc. should not count as valid."""
        from defensive_overlay_adapter import _is_valid_value

        assert _is_valid_value("N/A") is False
        assert _is_valid_value("Unknown") is False
        assert _is_valid_value("NaN") is False
        assert _is_valid_value("-") is False
        assert _is_valid_value("None") is False
        assert _is_valid_value("null") is False

    def test_zero_is_valid(self):
        """Zero should be counted as valid (it's a real value)."""
        from defensive_overlay_adapter import _is_valid_value

        assert _is_valid_value("0") is True
        assert _is_valid_value("0.0") is True
        assert _is_valid_value(0) is True

    def test_numeric_strings_valid(self):
        """Numeric strings should be valid."""
        from defensive_overlay_adapter import _is_valid_value

        assert _is_valid_value("0.50") is True
        assert _is_valid_value("-0.25") is True
        assert _is_valid_value("1.5") is True

    def test_coverage_excludes_null_equivalents(self):
        """Coverage counts should exclude null-equivalent values."""
        from defensive_overlay_adapter import enrich_with_defensive_overlays

        output = {
            "ranked_securities": [
                {"ticker": "AAA", "composite_score": "50.00", "rankable": True},
                {"ticker": "BBB", "composite_score": "40.00", "rankable": True},
            ]
        }

        scores_by_ticker = {
            "AAA": {"defensive_features": {"vol_60d": "0.30", "corr_xbi": "N/A"}},  # corr is null-eq
            "BBB": {"defensive_features": {"vol_60d": "", "corr_xbi": "0.40"}},     # vol is null-eq
        }

        result = enrich_with_defensive_overlays(
            output, scores_by_ticker, apply_multiplier=False
        )

        coverage = result["diagnostic_counts"]["defensive_features_coverage"]
        by_feature = coverage["by_feature"]
        # corr_xbi_120d: only BBB has valid (AAA has "N/A")
        assert by_feature["corr_xbi_120d"]["count"] == 1
        # vol_60d: only AAA has valid (BBB has "")
        assert by_feature["vol_60d"]["count"] == 1


class TestAliasCoverage:
    """Tests for alias field coverage diagnostics."""

    def test_alias_coverage_tracked_separately(self):
        """Alias fields should be tracked in separate alias_coverage dict."""
        from defensive_overlay_adapter import enrich_with_defensive_overlays

        output = {
            "ranked_securities": [
                {"ticker": "AAA", "composite_score": "50.00", "rankable": True},
                {"ticker": "BBB", "composite_score": "40.00", "rankable": True},
            ]
        }

        scores_by_ticker = {
            "AAA": {"defensive_features": {"corr_xbi": "0.30", "drawdown_60d": "-0.10"}},
            "BBB": {"defensive_features": {"corr_xbi_120d": "0.40", "drawdown_current": "-0.15"}},
        }

        result = enrich_with_defensive_overlays(
            output, scores_by_ticker, apply_multiplier=False
        )

        coverage = result["diagnostic_counts"]["defensive_features_coverage"]

        # Alias coverage tracked separately
        assert "alias_coverage" in coverage
        assert coverage["alias_coverage"]["corr_xbi"]["count"] == 1  # AAA
        assert coverage["alias_coverage"]["drawdown_60d"]["count"] == 1  # AAA

        # Canonical includes aliased values
        assert coverage["by_feature"]["corr_xbi_120d"]["count"] == 2  # Both via alias
        assert coverage["by_feature"]["drawdown_current"]["count"] == 2  # Both via alias

    def test_n_with_sufficient_features(self):
        """n_with_sufficient_features_for_multiplier should count correctly."""
        from defensive_overlay_adapter import enrich_with_defensive_overlays

        output = {
            "ranked_securities": [
                {"ticker": "AAA", "composite_score": "50.00", "rankable": True},
                {"ticker": "BBB", "composite_score": "40.00", "rankable": True},
                {"ticker": "CCC", "composite_score": "30.00", "rankable": True},
            ]
        }

        scores_by_ticker = {
            "AAA": {"defensive_features": {"corr_xbi": "0.30", "vol_60d": "0.40"}},  # Has corr+vol
            "BBB": {"defensive_features": {"ret_21d": "0.10"}},  # Has momentum only
            "CCC": {"defensive_features": {}},  # No features
        }

        result = enrich_with_defensive_overlays(
            output, scores_by_ticker, apply_multiplier=False
        )

        coverage = result["diagnostic_counts"]["defensive_features_coverage"]
        # AAA and BBB have sufficient, CCC does not
        assert coverage["n_with_sufficient_features_for_multiplier"] == 2


class TestOutputSchemaColumns:
    """Tests for attach_output_schema_columns() - extended output schema."""

    def test_output_schema_extension(self):
        """Output schema extension should add all required columns."""
        from defensive_overlay_adapter import attach_output_schema_columns, OUTPUT_SCHEMA_VERSION

        output = {
            "ranked_securities": [{
                "ticker": "AAA",
                "score_z": 1.2,
                "expected_excess_return_annual": 0.096,
                "defensive_features": {"vol_60d": "0.40", "drawdown_current": "-0.15"},
                "cluster_id": 2,
                "component_scores": {"clinical": "70.0", "nested": {"skip": "me"}},
            }]
        }
        coverage = attach_output_schema_columns(output)

        assert output["output_schema_version"] == OUTPUT_SCHEMA_VERSION
        rec = output["ranked_securities"][0]
        assert rec["expected_excess_return"] == 0.096  # Alias created
        assert rec["volatility"] == "0.40"             # Extracted from defensive_features
        assert rec["drawdown"] == "-0.15"              # Extracted from defensive_features
        assert rec["module_scores"] == {"clinical": "70.0"}  # Scalars only
        assert coverage["score_z"] == 1


# ============================================================================
# CACHE MERGE HELPERS TESTS
# ============================================================================

class TestCacheMergeHelpers:
    """Tests for load_defensive_cache() and merge_cache_into_scores()."""

    def test_load_and_merge_fills_gaps(self, tmp_path):
        """Load cache file and merge should fill missing features."""
        import json
        from defensive_overlay_adapter import load_defensive_cache, merge_cache_into_scores

        # Create cache file
        cache_file = tmp_path / "cache.json"
        with open(cache_file, "w") as f:
            json.dump({"data": {"features_by_ticker": {
                "AMGN": {"vol_60d": "0.35", "corr_xbi_120d": "0.25"},
            }}}, f)

        # Load and merge
        cache = load_defensive_cache(str(cache_file))
        scores = {"AMGN": {"defensive_features": {"vol_60d": "0.99"}}}  # Has vol_60d
        merged = merge_cache_into_scores(scores, cache, overwrite=False)

        # vol_60d NOT overwritten, corr filled
        assert scores["AMGN"]["defensive_features"]["vol_60d"] == "0.99"
        assert scores["AMGN"]["defensive_features"]["corr_xbi_120d"] == "0.25"
        assert merged == 1  # Only corr was merged

    def test_load_missing_file_raises(self, tmp_path):
        """Should raise FileNotFoundError for missing file."""
        from defensive_overlay_adapter import load_defensive_cache
        with pytest.raises(FileNotFoundError):
            load_defensive_cache(str(tmp_path / "nonexistent.json"))
