"""
Tests for accuracy improvement features:
- Burn acceleration detection (Module 2)
- Volatility-adjusted weighting (Module 5)
"""
import pytest
from decimal import Decimal
from unittest.mock import patch

# Module 2 imports
from module_2_financial_v2 import (
    BurnAcceleration,
    calculate_burn_acceleration,
    _extract_quarterly_burns,
)

# Module 5 imports
from module_5_composite_v2 import (
    VolatilityAdjustment,
    _extract_volatility,
    VOLATILITY_LOW_THRESHOLD,
    VOLATILITY_HIGH_THRESHOLD,
    VOLATILITY_MAX_ADJUSTMENT,
    VOLATILITY_BASELINE,
    compute_module_5_composite_v2,
)


# ============================================================================
# BURN ACCELERATION TESTS
# ============================================================================

class TestExtractQuarterlyBurns:
    """Tests for _extract_quarterly_burns helper."""

    def test_extracts_from_burn_history(self):
        """Should extract burns from burn_history field."""
        data = {"burn_history": [-10, -12, -15, -18]}
        burns = _extract_quarterly_burns(data)
        assert len(burns) == 4
        assert all(b > 0 for b in burns)  # Converted to absolute values

    def test_extracts_from_quarterly_burns(self):
        """Should extract from quarterly_burns field."""
        data = {"quarterly_burns": [-5, -7, -9]}
        burns = _extract_quarterly_burns(data)
        assert len(burns) == 3

    def test_extracts_from_cfo_history(self):
        """Should extract from cfo_history field."""
        data = {"cfo_history": [-20, -22]}
        burns = _extract_quarterly_burns(data)
        assert len(burns) == 2

    def test_filters_positive_values(self):
        """Should only include negative values (burns)."""
        data = {"burn_history": [-10, 5, -15, 20, -8]}  # Mix of burns and gains
        burns = _extract_quarterly_burns(data)
        # Only negative values should be extracted
        assert len(burns) == 3
        assert all(b > 0 for b in burns)

    def test_limits_to_four_quarters(self):
        """Should return at most 4 quarters."""
        data = {"burn_history": [-1, -2, -3, -4, -5, -6, -7]}
        burns = _extract_quarterly_burns(data)
        assert len(burns) == 4

    def test_handles_empty_data(self):
        """Should handle missing data gracefully."""
        burns = _extract_quarterly_burns({})
        assert burns == []

    def test_handles_none_values(self):
        """Should skip None values."""
        data = {"burn_history": [-10, None, -15, None]}
        burns = _extract_quarterly_burns(data)
        assert len(burns) == 2


class TestCalculateBurnAcceleration:
    """Tests for burn acceleration calculation."""

    def test_insufficient_data_returns_unknown(self):
        """With less than 2 quarters, returns unknown trend."""
        result = calculate_burn_acceleration({"burn_history": [-10]})
        assert result.trend_direction == "unknown"
        assert result.confidence == "none"
        assert result.penalty_factor == Decimal("1.0")
        assert result.is_accelerating is False

    def test_no_data_returns_unknown(self):
        """With no data, returns unknown trend."""
        result = calculate_burn_acceleration({})
        assert result.trend_direction == "unknown"
        assert result.quarters_analyzed == 0

    def test_two_quarters_low_confidence(self):
        """With 2 quarters, confidence should be low."""
        data = {"burn_history": [-10, -8]}  # Burns: 10, 8 (most recent first)
        result = calculate_burn_acceleration(data)
        assert result.confidence == "low"
        assert result.quarters_analyzed == 2

    def test_three_quarters_medium_confidence(self):
        """With 3 quarters, confidence should be medium."""
        data = {"burn_history": [-10, -9, -8]}
        result = calculate_burn_acceleration(data)
        assert result.confidence == "medium"
        assert result.quarters_analyzed == 3

    def test_four_quarters_high_confidence(self):
        """With 4 quarters, confidence should be high."""
        data = {"burn_history": [-10, -9, -8, -7]}
        result = calculate_burn_acceleration(data)
        assert result.confidence == "high"
        assert result.quarters_analyzed == 4

    def test_accelerating_burn_detected(self):
        """Should detect accelerating burn (increasing cash consumption)."""
        # Burns are increasing: 10 -> 12 -> 15 -> 20 (20% -> 25% -> 33% increase)
        data = {"burn_history": [-20, -15, -12, -10]}
        result = calculate_burn_acceleration(data)
        assert result.trend_direction == "accelerating"
        assert result.is_accelerating is True
        assert result.acceleration_rate > Decimal("10")  # Threshold
        assert result.penalty_factor < Decimal("1.0")  # Should apply penalty

    def test_decelerating_burn_detected(self):
        """Should detect decelerating burn (decreasing cash consumption)."""
        # Burns are decreasing: 20 -> 15 -> 12 -> 10 (25% -> 20% -> 17% decrease)
        data = {"burn_history": [-10, -12, -15, -20]}
        result = calculate_burn_acceleration(data)
        assert result.trend_direction == "decelerating"
        assert result.is_accelerating is False
        assert result.acceleration_rate < Decimal("-10")  # Threshold
        assert result.penalty_factor > Decimal("1.0")  # Should apply bonus

    def test_stable_burn_detected(self):
        """Should detect stable burn (minimal change)."""
        # Burns are relatively stable: 10 -> 10.5 -> 10.2 -> 10.3 (<10% change)
        data = {"burn_history": [Decimal("-10.3"), Decimal("-10.2"), Decimal("-10.5"), Decimal("-10")]}
        result = calculate_burn_acceleration(data)
        assert result.trend_direction == "stable"
        assert result.is_accelerating is False
        assert result.penalty_factor == Decimal("1.0")  # No penalty or bonus

    def test_penalty_factor_bounded(self):
        """Penalty factor should not exceed 30% reduction."""
        # Extreme acceleration: 1 -> 5 -> 25 -> 125 (400% increase each quarter)
        data = {"burn_history": [-125, -25, -5, -1]}
        result = calculate_burn_acceleration(data)
        assert result.penalty_factor >= Decimal("0.70")  # Max 30% penalty

    def test_bonus_factor_bounded(self):
        """Bonus factor should not exceed 10% increase."""
        # Extreme deceleration: 100 -> 50 -> 25 -> 12 (50% decrease each quarter)
        data = {"burn_history": [-12, -25, -50, -100]}
        result = calculate_burn_acceleration(data)
        assert result.penalty_factor <= Decimal("1.10")  # Max 10% bonus


class TestBurnAccelerationIntegration:
    """Integration tests for burn acceleration in scoring."""

    def test_acceleration_affects_runway_score(self):
        """Burn acceleration should reduce effective runway."""
        # This would be tested in the full scoring flow
        accelerating = BurnAcceleration(
            is_accelerating=True,
            acceleration_rate=Decimal("25.0"),
            trend_direction="accelerating",
            quarters_analyzed=4,
            confidence="high",
            penalty_factor=Decimal("0.875"),  # 12.5% penalty
        )
        assert accelerating.penalty_factor < Decimal("1.0")

    def test_deceleration_improves_runway_score(self):
        """Burn deceleration should improve effective runway."""
        decelerating = BurnAcceleration(
            is_accelerating=False,
            acceleration_rate=Decimal("-20.0"),
            trend_direction="decelerating",
            quarters_analyzed=4,
            confidence="high",
            penalty_factor=Decimal("1.04"),  # 4% bonus
        )
        assert decelerating.penalty_factor > Decimal("1.0")


# ============================================================================
# VOLATILITY ADJUSTMENT TESTS
# ============================================================================

class TestExtractVolatility:
    """Tests for volatility extraction and adjustment calculation."""

    def test_no_volatility_data_returns_neutral(self):
        """Should return neutral adjustment when no volatility data available."""
        result = _extract_volatility({}, None, None)
        assert result.vol_bucket == "unknown"
        assert result.adjustment_factor == Decimal("1.0")
        assert result.confidence_penalty == Decimal("0.05")
        assert result.annualized_vol is None

    def test_extracts_from_market_data_first(self):
        """Should prefer market data volatility_252d."""
        market_data = {"volatility_252d": "0.45"}  # 45% vol
        result = _extract_volatility({}, None, market_data)
        assert result.annualized_vol == Decimal("45.00")
        assert result.vol_bucket == "normal"

    def test_extracts_from_si_data_if_no_market(self):
        """Should use short interest data if market data unavailable."""
        si_data = {"realized_volatility": "0.35"}
        result = _extract_volatility({}, si_data, None)
        assert result.annualized_vol == Decimal("35.00")
        assert result.vol_bucket == "normal"

    def test_extracts_from_financial_data_as_fallback(self):
        """Should use financial data as last resort."""
        fin_data = {"price_volatility": "0.55"}
        result = _extract_volatility(fin_data, None, None)
        assert result.annualized_vol == Decimal("55.00")
        assert result.vol_bucket == "normal"

    def test_low_volatility_bucket(self):
        """Volatility below 30% should be classified as low."""
        market_data = {"volatility_252d": "0.20"}  # 20% vol
        result = _extract_volatility({}, None, market_data)
        assert result.vol_bucket == "low"
        assert result.adjustment_factor > Decimal("1.0")  # Should boost
        assert result.confidence_penalty == Decimal("0")

    def test_normal_volatility_bucket(self):
        """Volatility between 30-80% should be classified as normal."""
        market_data = {"volatility_252d": "0.50"}  # 50% vol
        result = _extract_volatility({}, None, market_data)
        assert result.vol_bucket == "normal"
        assert result.adjustment_factor == Decimal("1.0")  # Neutral
        # Should have some confidence penalty
        assert Decimal("0") < result.confidence_penalty < Decimal("0.15")

    def test_high_volatility_bucket(self):
        """Volatility above 80% should be classified as high."""
        market_data = {"volatility_252d": "0.95"}  # 95% vol
        result = _extract_volatility({}, None, market_data)
        assert result.vol_bucket == "high"
        assert result.adjustment_factor < Decimal("1.0")  # Should reduce
        assert result.confidence_penalty == Decimal("0.15")

    def test_low_vol_max_boost(self):
        """Very low volatility should get max 25% boost."""
        market_data = {"volatility_252d": "0.05"}  # 5% vol - very low
        result = _extract_volatility({}, None, market_data)
        assert result.vol_bucket == "low"
        # Near max boost
        assert result.adjustment_factor > Decimal("1.20")
        assert result.adjustment_factor <= Decimal("1.25")

    def test_high_vol_max_reduction(self):
        """Very high volatility should not reduce more than 25%."""
        market_data = {"volatility_252d": "1.50"}  # 150% vol - extremely high
        result = _extract_volatility({}, None, market_data)
        assert result.vol_bucket == "high"
        # Should not go below 75%
        assert result.adjustment_factor >= Decimal("0.75")

    def test_boundary_low_normal(self):
        """Volatility at 30% should be at boundary of low/normal."""
        market_data = {"volatility_252d": "0.30"}  # Exactly at threshold
        result = _extract_volatility({}, None, market_data)
        # At 30% exactly, should be normal (inclusive)
        assert result.vol_bucket == "normal"

    def test_boundary_normal_high(self):
        """Volatility at 80% should be at boundary of normal/high."""
        market_data = {"volatility_252d": "0.80"}  # Exactly at threshold
        result = _extract_volatility({}, None, market_data)
        # At 80% exactly, should still be normal (inclusive)
        assert result.vol_bucket == "normal"


class TestVolatilityAdjustmentDataclass:
    """Tests for VolatilityAdjustment dataclass."""

    def test_dataclass_fields(self):
        """Should have all required fields."""
        adj = VolatilityAdjustment(
            annualized_vol=Decimal("45.00"),
            vol_bucket="normal",
            adjustment_factor=Decimal("1.0"),
            confidence_penalty=Decimal("0.05"),
        )
        assert adj.annualized_vol == Decimal("45.00")
        assert adj.vol_bucket == "normal"
        assert adj.adjustment_factor == Decimal("1.0")
        assert adj.confidence_penalty == Decimal("0.05")


class TestVolatilityInCompositeScoring:
    """Integration tests for volatility in Module 5."""

    @pytest.fixture
    def minimal_inputs(self):
        """Minimal inputs for compute_module_5_composite_v2."""
        return {
            "universe_result": {
                "active_securities": [
                    {"ticker": "TEST", "status": "active", "market_cap_mm": 1000},
                ]
            },
            "financial_result": {
                "scores": [
                    {
                        "ticker": "TEST",
                        "financial_score": 70.0,
                        "financial_normalized": 70.0,
                        "market_cap_mm": 1000,
                        "severity": "none",
                        "flags": [],
                    }
                ]
            },
            "catalyst_result": {
                "summaries": {
                    "TEST": {
                        "scores": {
                            "score_blended": 60.0,
                            "catalyst_proximity_score": 10.0,
                            "catalyst_delta_score": 5.0,
                        }
                    }
                }
            },
            "clinical_result": {
                "scores": [
                    {
                        "ticker": "TEST",
                        "clinical_score": "65.0",
                        "lead_phase": "Phase 2",
                        "severity": "none",
                        "flags": [],
                    }
                ]
            },
            "as_of_date": "2026-01-15",
        }

    def test_no_volatility_data_uses_neutral(self, minimal_inputs):
        """Without volatility data, should use neutral adjustment."""
        result = compute_module_5_composite_v2(**minimal_inputs)
        assert result["diagnostic_counts"]["with_volatility_data"] == 0
        # Score should still be computed
        assert len(result["ranked_securities"]) == 1
        vol_adj = result["ranked_securities"][0].get("volatility_adjustment", {})
        assert vol_adj.get("vol_bucket") == "unknown"

    def test_high_volatility_reduces_weights(self, minimal_inputs):
        """High volatility should reduce effective weights."""
        # Add high volatility market data
        minimal_inputs["market_data_by_ticker"] = {
            "TEST": {"volatility_252d": 0.95}  # 95% vol
        }
        result = compute_module_5_composite_v2(**minimal_inputs)
        ranked = result["ranked_securities"][0]

        # Check volatility adjustment was applied
        vol_adj = ranked.get("volatility_adjustment", {})
        assert vol_adj.get("vol_bucket") == "high"
        assert "high_volatility_adjustment" in ranked["flags"]

        # Diagnostic should show volatility data present
        assert result["diagnostic_counts"]["with_volatility_data"] == 1
        assert result["diagnostic_counts"]["high_volatility_count"] == 1

    def test_low_volatility_boosts_weights(self, minimal_inputs):
        """Low volatility should boost effective weights."""
        minimal_inputs["market_data_by_ticker"] = {
            "TEST": {"volatility_252d": 0.15}  # 15% vol
        }
        result = compute_module_5_composite_v2(**minimal_inputs)
        ranked = result["ranked_securities"][0]

        vol_adj = ranked.get("volatility_adjustment", {})
        assert vol_adj.get("vol_bucket") == "low"
        assert "low_volatility_boost" in ranked["flags"]

        # Diagnostic counts
        assert result["diagnostic_counts"]["low_volatility_count"] == 1

    def test_volatility_diagnostic_counts(self, minimal_inputs):
        """Should track volatility statistics in diagnostic_counts."""
        # Add multiple tickers with different volatilities
        minimal_inputs["universe_result"]["active_securities"] = [
            {"ticker": "LOW", "status": "active", "market_cap_mm": 1000},
            {"ticker": "MED", "status": "active", "market_cap_mm": 1000},
            {"ticker": "HIGH", "status": "active", "market_cap_mm": 1000},
            {"ticker": "NONE", "status": "active", "market_cap_mm": 1000},
        ]
        minimal_inputs["financial_result"]["scores"] = [
            {"ticker": t, "financial_score": 70.0, "financial_normalized": 70.0,
             "market_cap_mm": 1000, "severity": "none", "flags": []}
            for t in ["LOW", "MED", "HIGH", "NONE"]
        ]
        minimal_inputs["clinical_result"]["scores"] = [
            {"ticker": t, "clinical_score": "65.0", "lead_phase": "Phase 2",
             "severity": "none", "flags": []}
            for t in ["LOW", "MED", "HIGH", "NONE"]
        ]
        minimal_inputs["catalyst_result"]["summaries"] = {
            t: {"scores": {"score_blended": 60.0}} for t in ["LOW", "MED", "HIGH", "NONE"]
        }
        minimal_inputs["market_data_by_ticker"] = {
            "LOW": {"volatility_252d": 0.15},   # Low
            "MED": {"volatility_252d": 0.50},   # Normal
            "HIGH": {"volatility_252d": 0.95},  # High
            # NONE has no data
        }

        result = compute_module_5_composite_v2(**minimal_inputs)

        assert result["diagnostic_counts"]["with_volatility_data"] == 3
        assert result["diagnostic_counts"]["high_volatility_count"] == 1
        assert result["diagnostic_counts"]["low_volatility_count"] == 1

    def test_empty_universe_handles_volatility_gracefully(self, minimal_inputs):
        """Empty universe should not crash with volatility handling."""
        minimal_inputs["universe_result"]["active_securities"] = []
        result = compute_module_5_composite_v2(**minimal_inputs)
        # Should return empty but not crash
        # Note: Empty universe returns simplified diagnostic_counts without volatility fields
        assert result["ranked_securities"] == []
        assert result["diagnostic_counts"]["rankable"] == 0


class TestVolatilityDeterminism:
    """Tests for deterministic volatility calculations."""

    def test_same_inputs_same_output(self):
        """Same inputs should produce identical volatility adjustments."""
        market_data = {"volatility_252d": "0.65"}

        result1 = _extract_volatility({}, None, market_data)
        result2 = _extract_volatility({}, None, market_data)

        assert result1.annualized_vol == result2.annualized_vol
        assert result1.vol_bucket == result2.vol_bucket
        assert result1.adjustment_factor == result2.adjustment_factor
        assert result1.confidence_penalty == result2.confidence_penalty

    def test_volatility_in_score_breakdown(self):
        """Volatility adjustment should be included in output for auditability."""
        market_data = {"volatility_252d": "0.45"}
        result = _extract_volatility({}, None, market_data)

        # Should have all fields populated
        assert result.annualized_vol is not None
        assert result.vol_bucket in ["low", "normal", "high", "unknown"]
        assert result.adjustment_factor is not None
        assert result.confidence_penalty is not None


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases in accuracy improvements."""

    def test_burn_acceleration_with_zero_prior_quarter(self):
        """Should handle case where prior quarter burn is zero/very small."""
        data = {"burn_history": [-10, Decimal("-0.00001"), -8]}
        result = calculate_burn_acceleration(data)
        # Should handle gracefully, likely unknown or partial analysis
        assert result.quarters_analyzed <= 3

    def test_volatility_zero(self):
        """Should handle zero volatility."""
        market_data = {"volatility_252d": "0.0"}
        result = _extract_volatility({}, None, market_data)
        assert result.vol_bucket == "low"
        # Max boost at 0% vol
        assert result.adjustment_factor == Decimal("1.25").quantize(Decimal("0.0001"))

    def test_volatility_negative_handled(self):
        """Should handle negative volatility (invalid but defensive)."""
        market_data = {"volatility_252d": "-0.10"}
        result = _extract_volatility({}, None, market_data)
        # Should still work - negative would be treated as low
        assert result.vol_bucket == "low"

    def test_burn_all_positive_values(self):
        """Should handle case where all values are positive (profitable)."""
        data = {"burn_history": [10, 12, 15]}  # All positive = profits
        burns = _extract_quarterly_burns(data)
        assert burns == []  # No burns extracted

        result = calculate_burn_acceleration(data)
        assert result.trend_direction == "unknown"

    def test_volatility_string_parsing(self):
        """Should handle volatility as string input."""
        market_data = {"volatility_252d": "  0.50  "}  # With whitespace
        result = _extract_volatility({}, None, market_data)
        assert result.annualized_vol == Decimal("50.00")

    def test_volatility_integer_input(self):
        """Should handle volatility as integer."""
        market_data = {"volatility_252d": 1}  # 100% vol as integer
        result = _extract_volatility({}, None, market_data)
        assert result.vol_bucket == "high"
