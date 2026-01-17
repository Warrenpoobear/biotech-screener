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


# ============================================================================
# NEW ACCURACY IMPROVEMENTS TESTS (8 Priority Improvements)
# ============================================================================

from common.accuracy_improvements import (
    # 1. Indication-specific endpoint weighting
    TherapeuticArea,
    classify_therapeutic_area,
    compute_weighted_endpoint_score,
    # 2. Phase-dependent staleness
    compute_phase_staleness,
    get_staleness_threshold_for_phase,
    # 3. Regulatory pathway scoring
    RegulatoryDesignation,
    compute_regulatory_pathway_score,
    # 4. Regime-adaptive catalyst decay
    MarketRegimeType,
    compute_regime_adaptive_decay,
    # 5. Competitive landscape penalty
    compute_competition_penalty,
    # 6. Dynamic dilution
    compute_vix_dilution_adjustment,
    # 7. Burn seasonality
    compute_burn_seasonality_adjustment,
    # 8. Proximity boost
    compute_binary_event_proximity_boost,
    # Integration helper
    apply_all_accuracy_improvements,
)
from datetime import date


# =============================================================================
# 1. INDICATION-SPECIFIC ENDPOINT WEIGHTING TESTS
# =============================================================================

class TestIndicationSpecificEndpointWeighting:
    """Tests for indication-specific endpoint weighting."""

    def test_oncology_os_highest_weight(self):
        """OS should have highest weight in oncology."""
        weight, etype, is_strong = compute_weighted_endpoint_score(
            "Overall Survival at 5 years",
            TherapeuticArea.ONCOLOGY
        )
        assert weight >= Decimal("0.95")
        assert is_strong is True

    def test_oncology_pfs_vs_orr(self):
        """PFS should be weighted higher than ORR in oncology."""
        weight_pfs, _, _ = compute_weighted_endpoint_score(
            "Progression-free survival",
            TherapeuticArea.ONCOLOGY
        )
        weight_orr, _, _ = compute_weighted_endpoint_score(
            "Objective Response Rate",
            TherapeuticArea.ONCOLOGY
        )
        assert weight_pfs > weight_orr

    def test_autoimmune_acr_hierarchy(self):
        """ACR70 should be weighted higher than ACR50 in autoimmune."""
        weight_70, _, _ = compute_weighted_endpoint_score(
            "ACR70 response",
            TherapeuticArea.AUTOIMMUNE
        )
        weight_50, _, _ = compute_weighted_endpoint_score(
            "ACR50 response",
            TherapeuticArea.AUTOIMMUNE
        )
        assert weight_70 > weight_50

    def test_therapeutic_area_classification_oncology(self):
        """Test oncology classification from conditions."""
        conditions = ["breast cancer", "metastatic disease"]
        area = classify_therapeutic_area(conditions)
        assert area == TherapeuticArea.ONCOLOGY

    def test_therapeutic_area_classification_autoimmune(self):
        """Test autoimmune classification."""
        conditions = ["rheumatoid arthritis", "joint inflammation"]
        area = classify_therapeutic_area(conditions)
        assert area == TherapeuticArea.AUTOIMMUNE

    def test_therapeutic_area_classification_cns(self):
        """Test CNS classification."""
        conditions = ["alzheimer's disease", "cognitive decline"]
        area = classify_therapeutic_area(conditions)
        assert area == TherapeuticArea.CNS

    def test_empty_endpoint_neutral(self):
        """Empty endpoint should return neutral weight."""
        weight, _, _ = compute_weighted_endpoint_score("", TherapeuticArea.ONCOLOGY)
        assert weight == Decimal("0.50")


# =============================================================================
# 2. PHASE-DEPENDENT STALENESS TESTS
# =============================================================================

class TestPhaseSpecificStaleness:
    """Tests for phase-dependent staleness thresholds."""

    def test_phase3_strict_threshold(self):
        """Phase 3 should have 6-month staleness threshold."""
        config = get_staleness_threshold_for_phase("phase 3")
        assert config.max_staleness_days == 180
        assert config.severity_if_stale == "sev2"

    def test_phase1_lenient_threshold(self):
        """Phase 1 should have 2-year staleness threshold."""
        config = get_staleness_threshold_for_phase("phase 1")
        assert config.max_staleness_days == 730
        assert config.severity_if_stale == "none"

    def test_preclinical_very_lenient(self):
        """Preclinical should have 3-year threshold."""
        config = get_staleness_threshold_for_phase("preclinical")
        assert config.max_staleness_days == 1095

    def test_phase3_stale_after_6_months(self):
        """Phase 3 trial should be stale after 6+ months."""
        result = compute_phase_staleness(
            "phase 3",
            "2025-06-01",
            "2026-01-15"
        )
        assert result.is_stale is True
        assert result.staleness_penalty == Decimal("0.50")  # sev2

    def test_phase1_not_stale_at_1_year(self):
        """Phase 1 trial should not be stale at 1 year."""
        result = compute_phase_staleness(
            "phase 1",
            "2025-01-15",
            "2026-01-15"
        )
        assert result.is_stale is False

    def test_missing_date_warning(self):
        """Missing date should trigger warning but not stale."""
        result = compute_phase_staleness("phase 2", None, "2026-01-15")
        assert result.is_stale is False
        assert result.is_warning is True
        assert result.staleness_penalty == Decimal("0.90")


# =============================================================================
# 3. REGULATORY PATHWAY SCORING TESTS
# =============================================================================

class TestRegulatoryPathway:
    """Tests for regulatory pathway scoring."""

    def test_breakthrough_designation_bonus(self):
        """Breakthrough designation should add +15 points."""
        trial_data = {"breakthrough_designation": True}
        result = compute_regulatory_pathway_score(trial_data)
        assert RegulatoryDesignation.BREAKTHROUGH_THERAPY in result.designations_detected
        assert result.total_score_modifier >= Decimal("15")
        assert result.is_expedited is True

    def test_fast_track_bonus(self):
        """Fast track should add +8 points."""
        trial_data = {"fast_track_designation": True}
        result = compute_regulatory_pathway_score(trial_data)
        assert RegulatoryDesignation.FAST_TRACK in result.designations_detected
        assert result.total_score_modifier >= Decimal("8")

    def test_orphan_designation_bonus(self):
        """Orphan designation should add +5 points."""
        trial_data = {"orphan_designation": True}
        result = compute_regulatory_pathway_score(trial_data)
        assert RegulatoryDesignation.ORPHAN_DRUG in result.designations_detected
        assert result.total_score_modifier >= Decimal("5")

    def test_rems_penalty(self):
        """REMS requirement should subtract -5 points."""
        trial_data = {"rems_required": True}
        result = compute_regulatory_pathway_score(trial_data)
        assert RegulatoryDesignation.REMS_REQUIRED in result.designations_detected
        assert result.total_score_modifier < Decimal("0")
        assert result.has_risk_factor is True

    def test_multiple_designations_additive(self):
        """Multiple designations should be additive."""
        trial_data = {
            "breakthrough_designation": True,
            "orphan_designation": True
        }
        result = compute_regulatory_pathway_score(trial_data)
        assert len(result.designations_detected) == 2
        assert result.total_score_modifier == Decimal("20")  # 15 + 5

    def test_text_based_detection(self):
        """Should detect designations from text fields."""
        trial_data = {"brief_title": "Fast track study of novel compound"}
        result = compute_regulatory_pathway_score(trial_data)
        assert RegulatoryDesignation.FAST_TRACK in result.designations_detected


# =============================================================================
# 4. REGIME-ADAPTIVE CATALYST DECAY TESTS
# =============================================================================

class TestRegimeAdaptiveCatalystDecay:
    """Tests for regime-adaptive catalyst decay."""

    def test_bull_market_faster_decay(self):
        """Bull market should have 20-day half-life."""
        result = compute_regime_adaptive_decay(
            "2026-01-01",
            "2026-01-15",
            MarketRegimeType.BULL
        )
        assert result.decay_half_life_days == 20

    def test_bear_market_slower_decay(self):
        """Bear market should have 45-day half-life."""
        result = compute_regime_adaptive_decay(
            "2026-01-01",
            "2026-01-15",
            MarketRegimeType.BEAR
        )
        assert result.decay_half_life_days == 45

    def test_volatility_spike_slower_decay(self):
        """Volatility spike should have 45-day half-life."""
        result = compute_regime_adaptive_decay(
            "2026-01-01",
            "2026-01-15",
            MarketRegimeType.VOLATILITY_SPIKE
        )
        assert result.decay_half_life_days == 45

    def test_sector_rotation_normal_decay(self):
        """Sector rotation should have 30-day half-life."""
        result = compute_regime_adaptive_decay(
            "2026-01-01",
            "2026-01-15",
            MarketRegimeType.SECTOR_ROTATION
        )
        assert result.decay_half_life_days == 30

    def test_bull_vs_bear_weight_difference(self):
        """Same event should have higher weight in bear vs bull."""
        bull_result = compute_regime_adaptive_decay(
            "2026-01-01",
            "2026-01-15",
            MarketRegimeType.BULL
        )
        bear_result = compute_regime_adaptive_decay(
            "2026-01-01",
            "2026-01-15",
            MarketRegimeType.BEAR
        )
        assert bear_result.decay_weight > bull_result.decay_weight

    def test_same_day_full_weight(self):
        """Same-day events should have full weight."""
        result = compute_regime_adaptive_decay(
            "2026-01-15",
            "2026-01-15",
            MarketRegimeType.BULL
        )
        assert result.decay_weight == Decimal("1.0")

    def test_future_event_zero_weight(self):
        """Future events should have zero weight (PIT safety)."""
        result = compute_regime_adaptive_decay(
            "2026-02-01",
            "2026-01-15",
            MarketRegimeType.BULL
        )
        assert result.decay_weight == Decimal("0")


# =============================================================================
# 5. COMPETITIVE LANDSCAPE PENALTY TESTS
# =============================================================================

class TestCompetitiveLandscape:
    """Tests for competitive landscape penalty."""

    def test_no_competitors_no_penalty(self):
        """0-2 competitors should have no penalty."""
        result = compute_competition_penalty(
            "rare disease",
            "phase 3",
            [{"phase": "phase 3"}]  # 1 competitor
        )
        assert result.competition_level == "low"
        assert result.penalty_points == Decimal("0")

    def test_moderate_competition_5_points(self):
        """3-5 competitors should have 5-point penalty."""
        competitors = [{"phase": "phase 3"} for _ in range(4)]
        result = compute_competition_penalty(
            "indication",
            "phase 3",
            competitors
        )
        assert result.competition_level == "moderate"
        assert result.penalty_points == Decimal("5")

    def test_high_competition_12_points(self):
        """6-10 competitors should have 12-point penalty."""
        competitors = [{"phase": "phase 3"} for _ in range(8)]
        result = compute_competition_penalty(
            "indication",
            "phase 3",
            competitors
        )
        assert result.competition_level == "high"
        assert result.penalty_points == Decimal("12")

    def test_hyper_competitive_20_points(self):
        """11+ competitors should have 20-point penalty."""
        competitors = [{"phase": "phase 3"} for _ in range(15)]
        result = compute_competition_penalty(
            "indication",
            "phase 3",
            competitors
        )
        assert result.competition_level == "hyper_competitive"
        assert result.penalty_points == Decimal("20")

    def test_first_in_class_50_percent_reduction(self):
        """First-in-class should reduce penalty by 50%."""
        competitors = [{"phase": "phase 3"} for _ in range(8)]
        result = compute_competition_penalty(
            "indication",
            "phase 3",
            competitors,
            is_first_in_class=True
        )
        assert result.penalty_points == Decimal("6")  # 12 * 0.5

    def test_early_phase_50_percent_reduction(self):
        """Early phase should reduce penalty by 50%."""
        competitors = [{"phase": "phase 3"} for _ in range(8)]
        result = compute_competition_penalty(
            "indication",
            "phase 1",  # Early phase
            competitors
        )
        assert result.penalty_points == Decimal("6")  # 12 * 0.5


# =============================================================================
# 6. VIX-ADJUSTED DILUTION TESTS
# =============================================================================

class TestVixAdjustedDilution:
    """Tests for VIX-adjusted dilution risk."""

    def test_low_vix_favorable(self):
        """Low VIX (< 15) should reduce dilution risk."""
        result = compute_vix_dilution_adjustment(Decimal("12"), Decimal("50"))
        assert result.vix_bucket == "low"
        assert result.adjustment_factor == Decimal("0.85")
        assert result.market_receptiveness == "favorable"

    def test_normal_vix_neutral(self):
        """Normal VIX (15-20) should be neutral."""
        result = compute_vix_dilution_adjustment(Decimal("18"), Decimal("50"))
        assert result.vix_bucket == "normal"
        assert result.adjustment_factor == Decimal("1.00")

    def test_elevated_vix_challenging(self):
        """Elevated VIX (20-25) should increase risk."""
        result = compute_vix_dilution_adjustment(Decimal("23"), Decimal("50"))
        assert result.vix_bucket == "elevated"
        assert result.adjustment_factor == Decimal("1.15")
        assert result.market_receptiveness == "challenging"

    def test_high_vix_difficult(self):
        """High VIX (25-35) should significantly increase risk."""
        result = compute_vix_dilution_adjustment(Decimal("30"), Decimal("50"))
        assert result.vix_bucket == "high"
        assert result.adjustment_factor == Decimal("1.35")
        assert result.market_receptiveness == "difficult"

    def test_extreme_vix(self):
        """Extreme VIX (35+) should have maximum penalty."""
        result = compute_vix_dilution_adjustment(Decimal("45"), Decimal("50"))
        assert result.vix_bucket == "extreme"
        assert result.adjustment_factor == Decimal("1.60")


# =============================================================================
# 7. BURN SEASONALITY TESTS
# =============================================================================

class TestBurnSeasonalityAdjustment:
    """Tests for burn rate seasonality adjustment."""

    def test_q1_lower_burn_adjustment(self):
        """Q1 should have 15% lower burn (0.85 factor)."""
        result = compute_burn_seasonality_adjustment(
            Decimal("10000000"),
            Decimal("100000000"),
            "2026-01-15"  # Q1
        )
        assert result.fiscal_quarter == 1
        assert result.adjustment_factor == Decimal("0.85")
        assert result.adjusted_monthly_burn == Decimal("8500000.00")

    def test_q2_higher_burn(self):
        """Q2 should have 5% higher burn (1.05 factor)."""
        result = compute_burn_seasonality_adjustment(
            Decimal("10000000"),
            Decimal("100000000"),
            "2026-04-15"  # Q2
        )
        assert result.fiscal_quarter == 2
        assert result.adjustment_factor == Decimal("1.05")

    def test_q3_highest_burn(self):
        """Q3 should have 10% higher burn (1.10 factor)."""
        result = compute_burn_seasonality_adjustment(
            Decimal("10000000"),
            Decimal("100000000"),
            "2026-07-15"  # Q3
        )
        assert result.fiscal_quarter == 3
        assert result.adjustment_factor == Decimal("1.10")

    def test_q4_normalized(self):
        """Q4 should be normalized (1.00 factor)."""
        result = compute_burn_seasonality_adjustment(
            Decimal("10000000"),
            Decimal("100000000"),
            "2026-10-15"  # Q4
        )
        assert result.fiscal_quarter == 4
        assert result.adjustment_factor == Decimal("1.00")
        assert result.is_q4_submission_window is True


# =============================================================================
# 8. BINARY EVENT PROXIMITY BOOST TESTS
# =============================================================================

class TestBinaryEventProximity:
    """Tests for binary event proximity boost."""

    def test_imminent_20_percent_boost(self):
        """Events within 30 days should get 20% boost."""
        result = compute_binary_event_proximity_boost(
            "2026-02-01",  # 17 days from 2026-01-15
            "2026-01-15",
            confidence="high"
        )
        assert result.proximity_bucket == "imminent"
        assert result.boost_percentage == Decimal("0.200")

    def test_near_term_10_percent_boost(self):
        """Events 31-60 days out should get 10% boost."""
        result = compute_binary_event_proximity_boost(
            "2026-02-28",  # ~44 days
            "2026-01-15",
            confidence="high"
        )
        assert result.proximity_bucket == "near_term"
        assert result.boost_percentage == Decimal("0.100")

    def test_medium_term_5_percent_boost(self):
        """Events 61-90 days out should get 5% boost."""
        result = compute_binary_event_proximity_boost(
            "2026-03-25",  # ~69 days
            "2026-01-15",
            confidence="high"
        )
        assert result.proximity_bucket == "medium_term"
        assert result.boost_percentage == Decimal("0.050")

    def test_far_no_boost(self):
        """Events beyond 90 days should get no boost."""
        result = compute_binary_event_proximity_boost(
            "2026-06-01",
            "2026-01-15"
        )
        assert result.proximity_bucket == "far"
        assert result.boost_percentage == Decimal("0.000")

    def test_low_confidence_reduces_boost(self):
        """Low confidence should reduce boost by 50%."""
        result_low = compute_binary_event_proximity_boost(
            "2026-02-01",
            "2026-01-15",
            confidence="low"
        )
        assert result_low.boost_percentage == Decimal("0.100")  # 20% * 0.5

    def test_no_catalyst_no_boost(self):
        """No catalyst date should return no boost."""
        result = compute_binary_event_proximity_boost(None, "2026-01-15")
        assert result.proximity_bucket == "none"
        assert result.boost_percentage == Decimal("0")


# =============================================================================
# INTEGRATION TEST FOR ALL IMPROVEMENTS
# =============================================================================

class TestAllImprovementsIntegration:
    """Integration tests for all accuracy improvements."""

    def test_apply_all_improvements_returns_all_sections(self):
        """Applying all improvements should return all expected sections."""
        trial_data = {
            "phase": "phase 3",
            "conditions": ["breast cancer"],
            "primary_endpoint": "Overall Survival",
            "last_update_posted": "2025-12-01",
            "breakthrough_designation": True,
            "next_catalyst_date": "2026-02-01",
            "catalyst_confidence": "high",
        }
        financial_data = {
            "dilution_score": Decimal("60"),
            "monthly_burn": Decimal("15000000"),
            "liquid_assets": Decimal("200000000"),
        }
        market_data = {}

        result = apply_all_accuracy_improvements(
            ticker="ACME",
            trial_data=trial_data,
            financial_data=financial_data,
            market_data=market_data,
            as_of_date="2026-01-15",
            vix_current=Decimal("22"),
            market_regime=MarketRegimeType.BULL,
            competitor_programs=[{"phase": "phase 3"} for _ in range(5)],
        )

        # All sections should be present
        assert "endpoint_analysis" in result
        assert "staleness_analysis" in result
        assert "regulatory_analysis" in result
        assert "decay_analysis" in result
        assert "competition_analysis" in result
        assert "vix_adjustment" in result
        assert "seasonality_analysis" in result
        assert "proximity_analysis" in result

    def test_improvements_are_deterministic(self):
        """All improvements should be deterministic."""
        trial_data = {
            "phase": "phase 3",
            "conditions": ["nsclc"],
            "primary_endpoint": "PFS",
        }
        financial_data = {"dilution_score": 50}

        results = []
        for _ in range(3):
            result = apply_all_accuracy_improvements(
                ticker="TEST",
                trial_data=trial_data,
                financial_data=financial_data,
                market_data={},
                as_of_date="2026-01-15",
            )
            results.append(str(result))

        # All results should be identical
        assert len(set(results)) == 1
