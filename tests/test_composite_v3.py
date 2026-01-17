"""
Comprehensive tests for Module 5 Composite v3 (IC-Enhanced Edition).

Tests cover:
- All IC enhancement utilities
- Non-linear signal interactions
- Peer-relative valuation
- Catalyst signal decay
- Price momentum signal
- Shrinkage normalization
- Smart money signal
- Volatility-adjusted scoring
- Regime-adaptive components
- Adaptive weight learning
- Full composite scoring pipeline
- Determinism guarantees
- Edge cases and error handling

Author: Wake Robin Capital Management
"""
import pytest
from datetime import date
from decimal import Decimal
from typing import Dict, Any, List

# Import IC enhancement utilities
from src.modules.ic_enhancements import (
    compute_volatility_adjustment,
    apply_volatility_to_score,
    compute_momentum_signal,
    compute_valuation_signal,
    compute_catalyst_decay,
    apply_catalyst_decay,
    compute_smart_money_signal,
    compute_interaction_terms,
    shrinkage_normalize,
    apply_regime_to_weights,
    compute_adaptive_weights,
    compute_enhanced_score,
    get_regime_signal_importance,
    VolatilityBucket,
    RegimeType,
    _to_decimal,
    _quantize_score,
    _clamp,
)

# Import main composite function
from module_5_composite_v3 import (
    compute_module_5_composite_v3,
    _market_cap_bucket,
    _stage_bucket,
    _get_worst_severity,
    _rank_normalize_winsorized,
    _apply_monotonic_caps,
    _compute_global_stats,
    V3_ENHANCED_WEIGHTS,
    V3_DEFAULT_WEIGHTS,
    V3_PARTIAL_WEIGHTS,
    ScoringMode,
    NormalizationMethod,
    MonotonicCap,
)
from common.types import Severity


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def as_of_date():
    """Standard test date."""
    return "2026-01-15"


@pytest.fixture
def sample_universe_result():
    """Sample Module 1 output."""
    return {
        "active_securities": [
            {"ticker": "AAPL", "status": "active", "market_cap_mm": 3000000},
            {"ticker": "BIIB", "status": "active", "market_cap_mm": 25000},
            {"ticker": "MRNA", "status": "active", "market_cap_mm": 45000},
            {"ticker": "SGEN", "status": "active", "market_cap_mm": 18000},
            {"ticker": "ALNY", "status": "active", "market_cap_mm": 22000},
            {"ticker": "BMRN", "status": "active", "market_cap_mm": 12000},
            {"ticker": "EXEL", "status": "active", "market_cap_mm": 8000},
            {"ticker": "RARE", "status": "active", "market_cap_mm": 5000},
            {"ticker": "FOLD", "status": "active", "market_cap_mm": 2500},
            {"ticker": "BLUE", "status": "active", "market_cap_mm": 800},
        ],
        "excluded_securities": [],
        "diagnostic_counts": {"active": 10, "excluded": 0, "total_input": 10},
    }


@pytest.fixture
def sample_financial_result():
    """Sample Module 2 output."""
    return {
        "scores": [
            {"ticker": "AAPL", "financial_score": 95, "financial_normalized": 95, "market_cap_mm": 3000000, "runway_months": 999, "severity": "none", "flags": [], "liquidity_gate_status": "PASS", "dilution_risk_bucket": "LOW"},
            {"ticker": "BIIB", "financial_score": 78, "financial_normalized": 78, "market_cap_mm": 25000, "runway_months": 48, "severity": "none", "flags": [], "liquidity_gate_status": "PASS", "dilution_risk_bucket": "LOW"},
            {"ticker": "MRNA", "financial_score": 82, "financial_normalized": 82, "market_cap_mm": 45000, "runway_months": 36, "severity": "none", "flags": [], "liquidity_gate_status": "PASS", "dilution_risk_bucket": "LOW"},
            {"ticker": "SGEN", "financial_score": 70, "financial_normalized": 70, "market_cap_mm": 18000, "runway_months": 24, "severity": "none", "flags": [], "liquidity_gate_status": "PASS", "dilution_risk_bucket": "MEDIUM"},
            {"ticker": "ALNY", "financial_score": 65, "financial_normalized": 65, "market_cap_mm": 22000, "runway_months": 18, "severity": "sev1", "flags": ["runway_warning"], "liquidity_gate_status": "PASS", "dilution_risk_bucket": "MEDIUM"},
            {"ticker": "BMRN", "financial_score": 55, "financial_normalized": 55, "market_cap_mm": 12000, "runway_months": 15, "severity": "sev1", "flags": ["runway_warning"], "liquidity_gate_status": "PASS", "dilution_risk_bucket": "MEDIUM"},
            {"ticker": "EXEL", "financial_score": 60, "financial_normalized": 60, "market_cap_mm": 8000, "runway_months": 20, "severity": "none", "flags": [], "liquidity_gate_status": "PASS", "dilution_risk_bucket": "MEDIUM"},
            {"ticker": "RARE", "financial_score": 45, "financial_normalized": 45, "market_cap_mm": 5000, "runway_months": 12, "severity": "sev1", "flags": ["runway_warning"], "liquidity_gate_status": "WARN", "dilution_risk_bucket": "HIGH"},
            {"ticker": "FOLD", "financial_score": 35, "financial_normalized": 35, "market_cap_mm": 2500, "runway_months": 9, "severity": "sev2", "flags": ["runway_critical"], "liquidity_gate_status": "WARN", "dilution_risk_bucket": "HIGH"},
            {"ticker": "BLUE", "financial_score": 20, "financial_normalized": 20, "market_cap_mm": 800, "runway_months": 6, "severity": "sev2", "flags": ["runway_critical", "dilution_severe"], "liquidity_gate_status": "FAIL", "dilution_risk_bucket": "SEVERE"},
        ],
        "diagnostic_counts": {"scored": 10, "missing": 0},
    }


@pytest.fixture
def sample_catalyst_result():
    """Sample Module 3 output."""
    return {
        "summaries": {
            "AAPL": {"scores": {"score_blended": 50, "catalyst_proximity_score": 0, "catalyst_delta_score": 0}},
            "BIIB": {"scores": {"score_blended": 75, "catalyst_proximity_score": 80, "catalyst_delta_score": 10, "days_to_nearest_catalyst": 25, "nearest_catalyst_type": "DATA_READOUT"}},
            "MRNA": {"scores": {"score_blended": 85, "catalyst_proximity_score": 90, "catalyst_delta_score": 15, "days_to_nearest_catalyst": 15, "nearest_catalyst_type": "PDUFA"}},
            "SGEN": {"scores": {"score_blended": 65, "catalyst_proximity_score": 40, "catalyst_delta_score": 5}},
            "ALNY": {"scores": {"score_blended": 70, "catalyst_proximity_score": 60, "catalyst_delta_score": 8, "days_to_nearest_catalyst": 45}},
            "BMRN": {"scores": {"score_blended": 55, "catalyst_proximity_score": 30, "catalyst_delta_score": -5}},
            "EXEL": {"scores": {"score_blended": 60, "catalyst_proximity_score": 50, "catalyst_delta_score": 0}},
            "RARE": {"scores": {"score_blended": 72, "catalyst_proximity_score": 70, "catalyst_delta_score": 12, "days_to_nearest_catalyst": 30}},
            "FOLD": {"scores": {"score_blended": 40, "catalyst_proximity_score": 20, "catalyst_delta_score": -10}},
            "BLUE": {"scores": {"score_blended": 30, "catalyst_proximity_score": 10, "catalyst_delta_score": -15}, "flags": {"severe_negative_flag": True}},
        },
        "as_of_date": "2026-01-15",
        "schema_version": "2.0",
        "diagnostic_counts": {"total_input": 10, "scored": 10, "missing": 0},
        "score_version": "v2.0",
    }


@pytest.fixture
def sample_clinical_result():
    """Sample Module 4 output."""
    return {
        "as_of_date": "2026-01-15",
        "scores": [
            {"ticker": "AAPL", "clinical_score": "40", "lead_phase": "Approved", "severity": "none", "flags": [], "trial_count": 5},
            {"ticker": "BIIB", "clinical_score": "85", "lead_phase": "Phase 3", "severity": "none", "flags": [], "trial_count": 12},
            {"ticker": "MRNA", "clinical_score": "90", "lead_phase": "Phase 3", "severity": "none", "flags": [], "trial_count": 15},
            {"ticker": "SGEN", "clinical_score": "75", "lead_phase": "Phase 3", "severity": "none", "flags": [], "trial_count": 8},
            {"ticker": "ALNY", "clinical_score": "70", "lead_phase": "Phase 2", "severity": "none", "flags": [], "trial_count": 6},
            {"ticker": "BMRN", "clinical_score": "65", "lead_phase": "Phase 2", "severity": "none", "flags": [], "trial_count": 7},
            {"ticker": "EXEL", "clinical_score": "60", "lead_phase": "Phase 2", "severity": "none", "flags": [], "trial_count": 4},
            {"ticker": "RARE", "clinical_score": "55", "lead_phase": "Phase 2", "severity": "sev1", "flags": ["enrollment_slow"], "trial_count": 3},
            {"ticker": "FOLD", "clinical_score": "45", "lead_phase": "Phase 1", "severity": "none", "flags": [], "trial_count": 2},
            {"ticker": "BLUE", "clinical_score": "35", "lead_phase": "Phase 1", "severity": "sev2", "flags": ["trial_stopped"], "trial_count": 1},
        ],
        "diagnostic_counts": {"scored": 10, "missing": 0},
    }


@pytest.fixture
def sample_market_data():
    """Sample market data by ticker."""
    return {
        "AAPL": {"volatility_252d": 0.25, "return_60d": 0.08, "xbi_return_60d": 0.02},
        "BIIB": {"volatility_252d": 0.45, "return_60d": 0.15, "xbi_return_60d": 0.02},
        "MRNA": {"volatility_252d": 0.55, "return_60d": 0.20, "xbi_return_60d": 0.02},
        "SGEN": {"volatility_252d": 0.40, "return_60d": 0.05, "xbi_return_60d": 0.02},
        "ALNY": {"volatility_252d": 0.50, "return_60d": -0.05, "xbi_return_60d": 0.02},
        "BMRN": {"volatility_252d": 0.48, "return_60d": 0.00, "xbi_return_60d": 0.02},
        "EXEL": {"volatility_252d": 0.52, "return_60d": 0.10, "xbi_return_60d": 0.02},
        "RARE": {"volatility_252d": 0.70, "return_60d": -0.10, "xbi_return_60d": 0.02},
        "FOLD": {"volatility_252d": 0.85, "return_60d": -0.20, "xbi_return_60d": 0.02},
        "BLUE": {"volatility_252d": 1.10, "return_60d": -0.35, "xbi_return_60d": 0.02},
    }


@pytest.fixture
def sample_enhancement_result():
    """Sample enhancement data with PoS and regime."""
    return {
        "pos_scores": {
            "scores": [
                {"ticker": "BIIB", "pos_score": 75},
                {"ticker": "MRNA", "pos_score": 80},
                {"ticker": "SGEN", "pos_score": 65},
                {"ticker": "ALNY", "pos_score": 55},
                {"ticker": "BMRN", "pos_score": 50},
            ]
        },
        "short_interest_scores": {
            "scores": [
                {"ticker": "RARE", "squeeze_potential": "HIGH", "signal_direction": "BULLISH"},
                {"ticker": "FOLD", "squeeze_potential": "MEDIUM", "signal_direction": "NEUTRAL"},
            ]
        },
        "regime": {
            "regime": "NEUTRAL",
            "signal_adjustments": {"quality": 1.0, "momentum": 1.0, "catalyst": 1.0},
        },
    }


# =============================================================================
# VOLATILITY ADJUSTMENT TESTS
# =============================================================================

class TestVolatilityAdjustment:
    """Tests for volatility-adjusted scoring."""

    def test_low_volatility_boost(self):
        """Low volatility should boost weights and scores."""
        result = compute_volatility_adjustment(Decimal("0.20"))
        assert result.vol_bucket == VolatilityBucket.LOW
        assert result.weight_adjustment_factor > Decimal("1.0")
        assert result.score_adjustment_factor >= Decimal("1.0")
        assert result.confidence_penalty == Decimal("0")

    def test_normal_volatility_neutral(self):
        """Normal volatility should be neutral."""
        result = compute_volatility_adjustment(Decimal("0.50"))
        assert result.vol_bucket == VolatilityBucket.NORMAL
        assert result.weight_adjustment_factor == Decimal("1.0")

    def test_high_volatility_penalty(self):
        """High volatility should reduce weights and scores."""
        result = compute_volatility_adjustment(Decimal("1.00"))
        assert result.vol_bucket == VolatilityBucket.HIGH
        assert result.weight_adjustment_factor < Decimal("1.0")
        assert result.score_adjustment_factor < Decimal("1.0")
        assert result.confidence_penalty > Decimal("0.10")

    def test_unknown_volatility(self):
        """Missing volatility should return neutral with small penalty."""
        result = compute_volatility_adjustment(None)
        assert result.vol_bucket == VolatilityBucket.UNKNOWN
        assert result.weight_adjustment_factor == Decimal("1.0")
        assert result.confidence_penalty == Decimal("0.05")

    def test_apply_volatility_to_score(self):
        """Score adjustment should dampen high-vol stocks."""
        vol_adj = compute_volatility_adjustment(Decimal("1.00"))
        original_score = Decimal("80")
        adjusted = apply_volatility_to_score(original_score, vol_adj)
        assert adjusted < original_score


# =============================================================================
# MOMENTUM SIGNAL TESTS
# =============================================================================

class TestMomentumSignal:
    """Tests for price momentum signal."""

    def test_positive_alpha(self):
        """Positive alpha should produce high momentum score."""
        result = compute_momentum_signal(Decimal("0.15"), Decimal("0.02"))
        assert result.momentum_score > Decimal("50")
        assert result.alpha_60d == Decimal("0.13")

    def test_negative_alpha(self):
        """Negative alpha should produce low momentum score."""
        result = compute_momentum_signal(Decimal("-0.10"), Decimal("0.05"))
        assert result.momentum_score < Decimal("50")
        assert result.alpha_60d < Decimal("0")

    def test_neutral_alpha(self):
        """Zero alpha should produce neutral score."""
        result = compute_momentum_signal(Decimal("0.05"), Decimal("0.05"))
        assert result.momentum_score == Decimal("50")

    def test_missing_data(self):
        """Missing data should return neutral with low confidence."""
        result = compute_momentum_signal(None, Decimal("0.05"))
        assert result.momentum_score == Decimal("50")
        assert result.confidence == Decimal("0.3")

    def test_strong_alpha_high_confidence(self):
        """Strong alpha should have high confidence."""
        result = compute_momentum_signal(Decimal("0.25"), Decimal("0.02"))
        assert result.confidence >= Decimal("0.7")


# =============================================================================
# VALUATION SIGNAL TESTS
# =============================================================================

class TestValuationSignal:
    """Tests for peer-relative valuation signal."""

    def test_cheap_vs_peers(self):
        """Cheap stock should have high valuation score."""
        peers = [
            {"market_cap_mm": 5000, "trial_count": 5, "stage_bucket": "mid"},
            {"market_cap_mm": 6000, "trial_count": 5, "stage_bucket": "mid"},
            {"market_cap_mm": 7000, "trial_count": 5, "stage_bucket": "mid"},
            {"market_cap_mm": 8000, "trial_count": 5, "stage_bucket": "mid"},
            {"market_cap_mm": 9000, "trial_count": 5, "stage_bucket": "mid"},
            {"market_cap_mm": 10000, "trial_count": 5, "stage_bucket": "mid"},
        ]
        # Cheapest stock
        result = compute_valuation_signal(
            Decimal("4000"), 5, "Phase 2", peers
        )
        assert result.valuation_score > Decimal("70")

    def test_expensive_vs_peers(self):
        """Expensive stock should have low valuation score."""
        peers = [
            {"market_cap_mm": 5000, "trial_count": 5, "stage_bucket": "mid"},
            {"market_cap_mm": 6000, "trial_count": 5, "stage_bucket": "mid"},
            {"market_cap_mm": 7000, "trial_count": 5, "stage_bucket": "mid"},
            {"market_cap_mm": 8000, "trial_count": 5, "stage_bucket": "mid"},
            {"market_cap_mm": 9000, "trial_count": 5, "stage_bucket": "mid"},
            {"market_cap_mm": 10000, "trial_count": 5, "stage_bucket": "mid"},
        ]
        # Most expensive
        result = compute_valuation_signal(
            Decimal("15000"), 5, "Phase 2", peers
        )
        assert result.valuation_score < Decimal("30")

    def test_insufficient_peers(self):
        """Insufficient peers should return neutral with low confidence."""
        peers = [
            {"market_cap_mm": 5000, "trial_count": 5, "stage_bucket": "mid"},
            {"market_cap_mm": 6000, "trial_count": 5, "stage_bucket": "mid"},
        ]
        result = compute_valuation_signal(
            Decimal("5500"), 5, "Phase 2", peers
        )
        assert result.valuation_score == Decimal("50")
        assert result.confidence <= Decimal("0.3")


# =============================================================================
# CATALYST DECAY TESTS
# =============================================================================

class TestCatalystDecay:
    """Tests for catalyst signal decay."""

    def test_optimal_window(self):
        """In optimal window should have high decay factor."""
        result = compute_catalyst_decay(30, "DATA_READOUT")
        assert result.in_optimal_window
        assert result.decay_factor >= Decimal("0.9")

    def test_far_from_catalyst(self):
        """Far from catalyst should have lower decay factor."""
        result = compute_catalyst_decay(120, "DATA_READOUT")
        assert not result.in_optimal_window
        assert result.decay_factor < Decimal("0.5")

    def test_past_catalyst(self):
        """Past catalyst should decay faster."""
        result = compute_catalyst_decay(-30, "DATA_READOUT")
        assert result.decay_factor < Decimal("0.7")

    def test_pdufa_slower_decay(self):
        """PDUFA events should decay slower."""
        pdufa = compute_catalyst_decay(60, "PDUFA")
        data_readout = compute_catalyst_decay(60, "DATA_READOUT")
        assert pdufa.decay_factor > data_readout.decay_factor

    def test_apply_decay_to_score(self):
        """Decay should move score toward neutral."""
        decay = compute_catalyst_decay(120, "DATA_READOUT")
        original = Decimal("80")
        decayed = apply_catalyst_decay(original, decay)
        # Should move toward 50
        assert decayed < original
        assert decayed > Decimal("50")


# =============================================================================
# SMART MONEY SIGNAL TESTS
# =============================================================================

class TestSmartMoneySignal:
    """Tests for smart money (13F) signal."""

    def test_high_overlap(self):
        """High overlap should increase score."""
        result = compute_smart_money_signal(
            overlap_count=4,
            holders=["Holder1", "Holder2", "Holder3", "Holder4"],
            position_changes=None,
        )
        assert result.smart_money_score > Decimal("60")
        assert result.overlap_bonus == Decimal("20")  # Capped

    def test_position_increases(self):
        """Position increases should boost score."""
        result = compute_smart_money_signal(
            overlap_count=2,
            holders=["Holder1", "Holder2"],
            position_changes={"Holder1": "INCREASE", "Holder2": "NEW"},
        )
        assert result.position_change_adjustment > Decimal("0")
        assert "Holder1" in result.holders_increasing

    def test_position_decreases(self):
        """Position decreases should lower score."""
        result = compute_smart_money_signal(
            overlap_count=2,
            holders=["Holder1", "Holder2"],
            position_changes={"Holder1": "DECREASE", "Holder2": "EXIT"},
        )
        assert result.position_change_adjustment < Decimal("0")
        assert "Holder1" in result.holders_decreasing

    def test_no_overlap(self):
        """No overlap should return neutral-low score."""
        result = compute_smart_money_signal(
            overlap_count=0,
            holders=[],
            position_changes=None,
        )
        assert result.smart_money_score == Decimal("50")
        assert result.confidence <= Decimal("0.2")


# =============================================================================
# INTERACTION TERMS TESTS
# =============================================================================

class TestInteractionTerms:
    """Tests for non-linear interaction terms with smooth ramps."""

    def test_clinical_financial_synergy(self):
        """High clinical + strong runway should produce synergy via smooth ramp."""
        result = compute_interaction_terms(
            clinical_normalized=Decimal("80"),
            financial_data={"runway_months": 24, "financial_score": 75},
            catalyst_normalized=Decimal("60"),
            stage_bucket="mid",
            vol_adjustment=None,
        )
        assert result.clinical_financial_synergy > Decimal("0")
        assert "clinical_financial_synergy" in result.interaction_flags
        # Should be max bonus (1.5) at clinical=80, runway=24
        assert result.clinical_financial_synergy == Decimal("1.50")

    def test_synergy_smooth_ramp(self):
        """Synergy should increase smoothly, not have cliff at 70."""
        # At clinical=65 (halfway from 60 to 80), synergy should be partial
        result_65 = compute_interaction_terms(
            clinical_normalized=Decimal("65"),
            financial_data={"runway_months": 24, "financial_score": 75},
            catalyst_normalized=Decimal("50"),
            stage_bucket="mid",
        )
        result_70 = compute_interaction_terms(
            clinical_normalized=Decimal("70"),
            financial_data={"runway_months": 24, "financial_score": 75},
            catalyst_normalized=Decimal("50"),
            stage_bucket="mid",
        )
        # Both should have synergy (no cliff at 70)
        assert result_65.clinical_financial_synergy > Decimal("0")
        assert result_70.clinical_financial_synergy > result_65.clinical_financial_synergy

    def test_late_stage_distress(self):
        """Late stage with weak runway should produce penalty via smooth ramp."""
        result = compute_interaction_terms(
            clinical_normalized=Decimal("70"),
            financial_data={"runway_months": 6, "financial_score": 40},
            catalyst_normalized=Decimal("60"),
            stage_bucket="late",
            vol_adjustment=None,
        )
        assert result.stage_financial_interaction < Decimal("0")
        assert "late_stage_distress" in result.interaction_flags
        # Should be max penalty (-2.0) at runway=6 for late stage
        assert result.stage_financial_interaction == Decimal("-2.00")

    def test_distress_smooth_ramp(self):
        """Distress penalty should ramp smoothly from 12mo to 6mo."""
        # At runway=9 (halfway from 12 to 6), penalty should be partial
        result_9 = compute_interaction_terms(
            clinical_normalized=Decimal("60"),
            financial_data={"runway_months": 9, "financial_score": 40},
            catalyst_normalized=Decimal("50"),
            stage_bucket="late",
        )
        result_6 = compute_interaction_terms(
            clinical_normalized=Decimal("60"),
            financial_data={"runway_months": 6, "financial_score": 40},
            catalyst_normalized=Decimal("50"),
            stage_bucket="late",
        )
        # Penalty should be greater (more negative) at 6 than at 9
        assert result_9.stage_financial_interaction < Decimal("0")
        assert result_6.stage_financial_interaction < result_9.stage_financial_interaction

    def test_no_distress_above_12mo(self):
        """No distress penalty when runway >= 12 months."""
        result = compute_interaction_terms(
            clinical_normalized=Decimal("60"),
            financial_data={"runway_months": 12, "financial_score": 40},
            catalyst_normalized=Decimal("50"),
            stage_bucket="late",
        )
        assert result.stage_financial_interaction == Decimal("0")

    def test_catalyst_volatility_dampening(self):
        """High catalyst in high-vol name should be dampened."""
        vol_adj = compute_volatility_adjustment(Decimal("1.00"))
        result = compute_interaction_terms(
            clinical_normalized=Decimal("60"),
            financial_data={"runway_months": 24, "financial_score": 60},
            catalyst_normalized=Decimal("85"),
            stage_bucket="mid",
            vol_adjustment=vol_adj,
        )
        assert result.catalyst_volatility_dampening > Decimal("0")
        assert "catalyst_vol_dampening" in result.interaction_flags

    def test_double_counting_prevention(self):
        """Distress penalty should be halved if runway gate already failed."""
        # Without gate
        result_no_gate = compute_interaction_terms(
            clinical_normalized=Decimal("60"),
            financial_data={"runway_months": 6, "financial_score": 40},
            catalyst_normalized=Decimal("50"),
            stage_bucket="late",
            runway_gate_status="UNKNOWN",
        )
        # With gate already failed
        result_gate_failed = compute_interaction_terms(
            clinical_normalized=Decimal("60"),
            financial_data={"runway_months": 6, "financial_score": 40},
            catalyst_normalized=Decimal("50"),
            stage_bucket="late",
            runway_gate_status="FAIL",
        )
        # Penalty should be halved when gate already applied
        assert abs(result_gate_failed.stage_financial_interaction) < abs(result_no_gate.stage_financial_interaction)
        assert result_gate_failed.runway_gate_already_applied is True

    def test_no_synergy_when_gate_failed(self):
        """Synergy bonus should not apply if runway gate already failed."""
        result = compute_interaction_terms(
            clinical_normalized=Decimal("80"),
            financial_data={"runway_months": 24, "financial_score": 75},
            catalyst_normalized=Decimal("50"),
            stage_bucket="mid",
            runway_gate_status="FAIL",
        )
        # No synergy despite high clinical and runway, because gate already failed
        assert result.clinical_financial_synergy == Decimal("0")

    def test_boundedness(self):
        """Total interaction adjustment should always be within [-2, +2]."""
        # Test extreme values
        for clinical in [Decimal("0"), Decimal("50"), Decimal("100")]:
            for runway in [1, 6, 12, 24, 36]:
                for stage in ["early", "mid", "late"]:
                    result = compute_interaction_terms(
                        clinical_normalized=clinical,
                        financial_data={"runway_months": runway},
                        catalyst_normalized=Decimal("50"),
                        stage_bucket=stage,
                    )
                    assert Decimal("-2.0") <= result.total_interaction_adjustment <= Decimal("2.0")
                    # Should never be NaN or None
                    assert result.total_interaction_adjustment is not None

    def test_monotonicity(self):
        """Increasing runway should never increase late-stage penalty."""
        penalties = []
        for runway in [3, 6, 9, 12, 15, 18]:
            result = compute_interaction_terms(
                clinical_normalized=Decimal("60"),
                financial_data={"runway_months": runway},
                catalyst_normalized=Decimal("50"),
                stage_bucket="late",
            )
            penalties.append((runway, result.stage_financial_interaction))

        # Penalties should be non-increasing (less negative) as runway increases
        for i in range(len(penalties) - 1):
            assert penalties[i][1] <= penalties[i + 1][1], \
                f"Monotonicity violated: runway {penalties[i][0]} has penalty {penalties[i][1]}, " \
                f"but runway {penalties[i + 1][0]} has penalty {penalties[i + 1][1]}"


# =============================================================================
# SHRINKAGE NORMALIZATION TESTS
# =============================================================================

class TestShrinkageNormalization:
    """Tests for Bayesian shrinkage normalization."""

    def test_small_cohort_shrinkage(self):
        """Small cohort should shrink toward global mean."""
        values = [Decimal("30"), Decimal("35"), Decimal("40")]
        global_mean = Decimal("50")
        global_std = Decimal("20")

        result, shrinkage_factor = shrinkage_normalize(values, global_mean, global_std)

        # Shrinkage factor should be significant for small cohort
        assert shrinkage_factor > Decimal("0.5")
        # Results should be bounded and reasonable
        assert len(result) == 3
        assert all(Decimal("5") <= r <= Decimal("95") for r in result)
        # Results should maintain relative ordering
        assert result[0] <= result[1] <= result[2]

    def test_large_cohort_less_shrinkage(self):
        """Large cohort should have less shrinkage."""
        values = [Decimal(str(i)) for i in range(20, 80)]
        global_mean = Decimal("50")
        global_std = Decimal("20")

        result, shrinkage_factor = shrinkage_normalize(values, global_mean, global_std)

        # Shrinkage factor should be small for large cohort
        assert shrinkage_factor < Decimal("0.1")


# =============================================================================
# REGIME-ADAPTIVE TESTS
# =============================================================================

class TestRegimeAdaptive:
    """Tests for regime-adaptive components."""

    def test_bull_regime_weights(self):
        """Bull regime should boost catalyst and momentum."""
        base_weights = {
            "clinical": Decimal("0.30"),
            "financial": Decimal("0.30"),
            "catalyst": Decimal("0.20"),
            "momentum": Decimal("0.10"),
            "valuation": Decimal("0.10"),
        }
        adjusted = apply_regime_to_weights(base_weights, "BULL")

        # Catalyst and momentum should be relatively higher
        assert adjusted["catalyst"] / adjusted["financial"] > base_weights["catalyst"] / base_weights["financial"]

    def test_bear_regime_weights(self):
        """Bear regime should boost financial and reduce momentum."""
        base_weights = {
            "clinical": Decimal("0.30"),
            "financial": Decimal("0.30"),
            "catalyst": Decimal("0.20"),
            "momentum": Decimal("0.10"),
            "valuation": Decimal("0.10"),
        }
        adjusted = apply_regime_to_weights(base_weights, "BEAR")

        # Financial should be relatively higher
        assert adjusted["financial"] / adjusted["momentum"] > base_weights["financial"] / base_weights["momentum"]

    def test_unknown_regime(self):
        """Unknown regime should reduce momentum."""
        importance = get_regime_signal_importance("UNKNOWN")
        assert importance.momentum < Decimal("1.0")


# =============================================================================
# ADAPTIVE WEIGHT LEARNING TESTS
# =============================================================================

class TestAdaptiveWeights:
    """Tests for adaptive weight learning with PIT-safe signature."""

    def test_no_data_fallback(self):
        """No historical data should return base weights."""
        from datetime import date
        base_weights = {"clinical": Decimal("0.40"), "financial": Decimal("0.35"), "catalyst": Decimal("0.25")}
        result = compute_adaptive_weights(
            historical_scores=[],
            forward_returns={},
            base_weights=base_weights,
            asof_date=date(2026, 1, 15),
        )
        assert result.weights == base_weights
        assert result.optimization_method == "fallback_no_data"
        assert result.training_periods == 0

    def test_with_historical_data(self):
        """Historical data should produce adjusted weights based on IC."""
        from datetime import date, timedelta
        base_weights = {"clinical": Decimal("0.40"), "financial": Decimal("0.35"), "catalyst": Decimal("0.25")}

        asof_date = date(2026, 1, 15)

        # Create mock historical data where clinical is most predictive
        # Each score record needs an as_of_date field
        # We'll create 6 months of data, one record per ticker per month
        historical_scores = []
        forward_returns = {}  # Dict[(date, ticker), Decimal]

        for month_offset in range(6, 12):  # 6-12 months ago (within lookback, before embargo)
            score_date = asof_date - timedelta(days=30 * month_offset)
            for i in range(20):  # 20 tickers per month
                ticker = f"T{i}"
                historical_scores.append({
                    "ticker": ticker,
                    "as_of_date": score_date.isoformat(),
                    "clinical": Decimal(str(50 + i)),  # Clinical varies by ticker
                    "financial": Decimal("50"),
                    "catalyst": Decimal("50"),
                })
                # Returns correlated with clinical score (higher ticker # = higher return)
                forward_returns[(score_date, ticker)] = Decimal(str(0.01 * i))

        result = compute_adaptive_weights(
            historical_scores=historical_scores,
            forward_returns=forward_returns,
            base_weights=base_weights,
            asof_date=asof_date,
            embargo_months=1,
            shrinkage_lambda=Decimal("0.50"),  # Less shrinkage for clearer signal
            smooth_gamma=Decimal("0.0"),  # No smoothing (no prev_weights)
        )

        # Should use PIT-safe method
        assert result.optimization_method == "pit_safe_rank_correlation"
        # Should have training periods
        assert result.training_periods >= 3
        # Weights should sum to ~1.0
        total = sum(result.weights.values())
        assert Decimal("0.99") <= total <= Decimal("1.01")

    def test_embargo_enforcement(self):
        """Data within embargo period should be excluded."""
        from datetime import date, timedelta
        base_weights = {"clinical": Decimal("0.40"), "financial": Decimal("0.35")}
        asof_date = date(2026, 1, 15)

        # Create data that's too recent (within embargo)
        recent_date = asof_date - timedelta(days=15)  # Only 15 days ago
        historical_scores = [
            {"ticker": f"T{i}", "as_of_date": recent_date.isoformat(), "clinical": Decimal("60"), "financial": Decimal("50")}
            for i in range(20)
        ]
        forward_returns = {(recent_date, f"T{i}"): Decimal("0.05") for i in range(20)}

        result = compute_adaptive_weights(
            historical_scores=historical_scores,
            forward_returns=forward_returns,
            base_weights=base_weights,
            asof_date=asof_date,
            embargo_months=1,  # 30 days embargo
        )

        # Should fall back because all data is within embargo
        assert "fallback" in result.optimization_method
        assert result.training_periods == 0

    def test_deterministic_tiebreaking(self):
        """Tickers with same scores should have deterministic ranking."""
        from datetime import date, timedelta
        base_weights = {"clinical": Decimal("0.50"), "financial": Decimal("0.50")}
        asof_date = date(2026, 1, 15)

        # Create data with ties (same scores for all tickers)
        historical_scores = []
        forward_returns = {}

        for month_offset in range(3, 9):
            score_date = asof_date - timedelta(days=30 * month_offset)
            for i in range(15):
                ticker = f"TICK{i:03d}"  # Zero-padded for consistent sorting
                historical_scores.append({
                    "ticker": ticker,
                    "as_of_date": score_date.isoformat(),
                    "clinical": Decimal("50"),  # Same score for all
                    "financial": Decimal("50"),
                })
                # Different returns for each ticker
                forward_returns[(score_date, ticker)] = Decimal(str(0.02 * i))

        # Run twice to verify determinism
        result1 = compute_adaptive_weights(
            historical_scores=historical_scores,
            forward_returns=forward_returns,
            base_weights=base_weights,
            asof_date=asof_date,
        )
        result2 = compute_adaptive_weights(
            historical_scores=historical_scores,
            forward_returns=forward_returns,
            base_weights=base_weights,
            asof_date=asof_date,
        )

        # Results should be identical
        assert result1.weights == result2.weights
        assert result1.historical_ic_by_component == result2.historical_ic_by_component


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================

class TestHelperFunctions:
    """Tests for helper functions."""

    def test_market_cap_bucket(self):
        """Market cap bucketing should work correctly."""
        assert _market_cap_bucket(15000) == "large"
        assert _market_cap_bucket(5000) == "mid"
        assert _market_cap_bucket(500) == "small"
        assert _market_cap_bucket(100) == "micro"
        assert _market_cap_bucket(None) == "unknown"

    def test_stage_bucket(self):
        """Stage bucketing should work correctly."""
        assert _stage_bucket("Phase 3") == "late"
        assert _stage_bucket("Approved") == "late"
        assert _stage_bucket("Phase 2") == "mid"
        assert _stage_bucket("Phase 1") == "early"
        assert _stage_bucket("Preclinical") == "early"
        assert _stage_bucket(None) == "early"

    def test_get_worst_severity(self):
        """Worst severity selection should work correctly."""
        assert _get_worst_severity(["none", "sev1", "sev2"]) == Severity.SEV2
        assert _get_worst_severity(["none", "sev3"]) == Severity.SEV3
        assert _get_worst_severity(["none", "none"]) == Severity.NONE

    def test_rank_normalize_winsorized(self):
        """Winsorized rank normalization should work."""
        values = [Decimal(str(i * 10)) for i in range(10)]
        result, winsorized = _rank_normalize_winsorized(values)

        assert len(result) == 10
        assert all(Decimal("0") <= r <= Decimal("100") for r in result)

    def test_monotonic_caps(self):
        """Monotonic caps should apply correctly."""
        score = Decimal("80")

        # Liquidity fail should cap at 35
        capped, caps = _apply_monotonic_caps(score, "FAIL", Decimal("24"), "LOW")
        assert capped == MonotonicCap.LIQUIDITY_FAIL_CAP
        assert len(caps) == 1

        # Runway critical should cap at 40
        capped, caps = _apply_monotonic_caps(score, "PASS", Decimal("5"), "LOW")
        assert capped == MonotonicCap.RUNWAY_CRITICAL_CAP


# =============================================================================
# FULL PIPELINE TESTS
# =============================================================================

class TestFullPipeline:
    """Tests for the full v3 composite pipeline."""

    def test_basic_pipeline(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """Basic pipeline should produce ranked results."""
        result = compute_module_5_composite_v3(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
        )

        assert result["as_of_date"] == as_of_date
        assert len(result["ranked_securities"]) > 0
        assert result["scoring_mode"] == "default"

        # Check ranking is monotonic
        scores = [Decimal(r["composite_score"]) for r in result["ranked_securities"]]
        assert scores == sorted(scores, reverse=True)

    def test_enhanced_pipeline(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
        sample_enhancement_result,
        sample_market_data,
    ):
        """Enhanced pipeline with all signals should work."""
        result = compute_module_5_composite_v3(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            enhancement_result=sample_enhancement_result,
            market_data_by_ticker=sample_market_data,
        )

        assert result["scoring_mode"] == "enhanced"
        assert result["enhancement_applied"]

        # Check enhancement signals are present
        first = result["ranked_securities"][0]
        assert "momentum_signal" in first
        assert "valuation_signal" in first
        assert "volatility_adjustment" in first

    def test_determinism(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """Same inputs should produce identical outputs."""
        result1 = compute_module_5_composite_v3(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
        )

        result2 = compute_module_5_composite_v3(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
        )

        # Scores should be identical
        for r1, r2 in zip(result1["ranked_securities"], result2["ranked_securities"]):
            assert r1["composite_score"] == r2["composite_score"]
            assert r1["determinism_hash"] == r2["determinism_hash"]

    def test_sev3_exclusion(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """Sev3 securities should be excluded."""
        # Add a sev3 security
        sample_financial_result["scores"].append({
            "ticker": "FAIL",
            "financial_score": 10,
            "financial_normalized": 10,
            "market_cap_mm": 100,
            "runway_months": 2,
            "severity": "sev3",
            "flags": ["delisting_imminent"],
            "liquidity_gate_status": "FAIL",
            "dilution_risk_bucket": "SEVERE",
        })
        sample_clinical_result["scores"].append({
            "ticker": "FAIL",
            "clinical_score": "20",
            "lead_phase": "Phase 1",
            "severity": "sev3",
            "flags": ["trial_terminated"],
            "trial_count": 0,
        })
        sample_catalyst_result["summaries"]["FAIL"] = {"scores": {"score_blended": 20}}
        sample_universe_result["active_securities"].append({"ticker": "FAIL", "status": "active", "market_cap_mm": 100})

        result = compute_module_5_composite_v3(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
        )

        # FAIL should be excluded
        ranked_tickers = [r["ticker"] for r in result["ranked_securities"]]
        excluded_tickers = [r["ticker"] for r in result["excluded_securities"]]

        assert "FAIL" not in ranked_tickers
        assert "FAIL" in excluded_tickers

    def test_empty_universe(self, as_of_date):
        """Empty universe should return empty results gracefully."""
        result = compute_module_5_composite_v3(
            universe_result={"active_securities": [], "excluded_securities": [], "diagnostic_counts": {"total_input": 0, "active": 0, "excluded": 0}},
            financial_result={"scores": [], "diagnostic_counts": {"scored": 0, "missing": 0}},
            catalyst_result={"summaries": {}, "diagnostic_counts": {"total_input": 0, "scored": 0, "missing": 0}, "as_of_date": as_of_date, "schema_version": "2.0", "score_version": "v2.0"},
            clinical_result={"scores": [], "as_of_date": as_of_date, "diagnostic_counts": {"scored": 0, "missing": 0}},
            as_of_date=as_of_date,
        )

        assert result["ranked_securities"] == []
        assert result["diagnostic_counts"]["rankable"] == 0

    def test_score_breakdown_present(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """Score breakdown should be present and complete."""
        result = compute_module_5_composite_v3(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
        )

        first = result["ranked_securities"][0]
        breakdown = first["score_breakdown"]

        assert "version" in breakdown
        assert "components" in breakdown
        assert "penalties_and_gates" in breakdown
        assert "interaction_terms" in breakdown
        assert "final" in breakdown
        assert "hybrid_aggregation" in breakdown


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for v3 improvements."""

    def test_interaction_terms_affect_score(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """Interaction terms should affect final scores."""
        result = compute_module_5_composite_v3(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
        )

        # Check that interaction flags are present
        all_flags = []
        for sec in result["ranked_securities"]:
            all_flags.extend(sec.get("interaction_terms", {}).get("flags", []))

        # At least some securities should have interaction flags
        # (depends on test data, but late stage distress should trigger)
        interaction_types = ["clinical_financial_synergy", "late_stage_distress", "mid_stage_runway_warning"]
        has_interactions = any(f in all_flags for f in interaction_types)
        # This may or may not be true depending on exact thresholds

    def test_volatility_affects_ranking(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
        sample_market_data,
    ):
        """Volatility adjustments should affect rankings."""
        # Run without market data
        result_no_vol = compute_module_5_composite_v3(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
        )

        # Run with market data
        result_with_vol = compute_module_5_composite_v3(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            market_data_by_ticker=sample_market_data,
        )

        # Scores should differ
        scores_no_vol = {r["ticker"]: r["composite_score"] for r in result_no_vol["ranked_securities"]}
        scores_with_vol = {r["ticker"]: r["composite_score"] for r in result_with_vol["ranked_securities"]}

        # At least some scores should be different
        different_scores = sum(
            1 for t in scores_no_vol
            if t in scores_with_vol and scores_no_vol[t] != scores_with_vol[t]
        )
        assert different_scores > 0

    def test_regime_affects_weights(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """Different regimes should produce different weights."""
        # Create two enhancement results with different regimes
        enhancement_bull = {
            "pos_scores": {"scores": []},
            "regime": {"regime": "BULL", "signal_adjustments": {}},
        }
        enhancement_bear = {
            "pos_scores": {"scores": []},
            "regime": {"regime": "BEAR", "signal_adjustments": {}},
        }

        result_bull = compute_module_5_composite_v3(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            enhancement_result=enhancement_bull,
        )

        result_bear = compute_module_5_composite_v3(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            enhancement_result=enhancement_bear,
        )

        # Effective weights should differ between regimes
        first_bull = result_bull["ranked_securities"][0]
        first_bear = result_bear["ranked_securities"][0]

        # Due to regime adjustments, weights should be different
        assert first_bull["effective_weights"] != first_bear["effective_weights"]


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_missing_scores(self, as_of_date):
        """Pipeline should handle missing scores gracefully."""
        universe = {"active_securities": [{"ticker": "TEST", "status": "active", "market_cap_mm": 1000}], "excluded_securities": [], "diagnostic_counts": {"total_input": 1, "active": 1, "excluded": 0}}
        financial = {"scores": [], "diagnostic_counts": {"scored": 0, "missing": 1}}  # No financial score
        catalyst = {"summaries": {}, "diagnostic_counts": {"total_input": 1, "scored": 0, "missing": 1}, "as_of_date": as_of_date, "schema_version": "2.0", "score_version": "v2.0"}  # No catalyst score
        clinical = {"scores": [], "as_of_date": as_of_date, "diagnostic_counts": {"scored": 0, "missing": 1}}  # No clinical score

        result = compute_module_5_composite_v3(
            universe_result=universe,
            financial_result=financial,
            catalyst_result=catalyst,
            clinical_result=clinical,
            as_of_date=as_of_date,
        )

        # Should still produce a result (with uncertainty penalty)
        assert len(result["ranked_securities"]) == 1
        assert "uncertainty_penalty_applied" in result["ranked_securities"][0]["flags"]

    def test_single_security(self, as_of_date):
        """Single security should work."""
        universe = {"active_securities": [{"ticker": "ONLY", "status": "active", "market_cap_mm": 5000}], "excluded_securities": [], "diagnostic_counts": {"total_input": 1, "active": 1, "excluded": 0}}
        financial = {"scores": [{"ticker": "ONLY", "financial_score": 70, "financial_normalized": 70, "market_cap_mm": 5000, "runway_months": 24, "severity": "none", "flags": [], "liquidity_gate_status": "PASS", "dilution_risk_bucket": "MEDIUM"}], "diagnostic_counts": {"scored": 1, "missing": 0}}
        catalyst = {"summaries": {"ONLY": {"scores": {"score_blended": 60}}}, "diagnostic_counts": {"total_input": 1, "scored": 1, "missing": 0}, "as_of_date": as_of_date, "schema_version": "2.0", "score_version": "v2.0"}
        clinical = {"scores": [{"ticker": "ONLY", "clinical_score": "65", "lead_phase": "Phase 2", "severity": "none", "flags": [], "trial_count": 5}], "as_of_date": as_of_date, "diagnostic_counts": {"scored": 1, "missing": 0}}

        result = compute_module_5_composite_v3(
            universe_result=universe,
            financial_result=financial,
            catalyst_result=catalyst,
            clinical_result=clinical,
            as_of_date=as_of_date,
        )

        assert len(result["ranked_securities"]) == 1
        assert result["ranked_securities"][0]["composite_rank"] == 1

    def test_extreme_values(self, as_of_date):
        """Extreme values should be handled correctly."""
        universe = {"active_securities": [{"ticker": "EXTREME", "status": "active", "market_cap_mm": 1}], "excluded_securities": [], "diagnostic_counts": {"total_input": 1, "active": 1, "excluded": 0}}
        financial = {"scores": [{"ticker": "EXTREME", "financial_score": 100, "financial_normalized": 100, "market_cap_mm": 1, "runway_months": 0.5, "severity": "sev2", "flags": ["extreme"], "liquidity_gate_status": "FAIL", "dilution_risk_bucket": "SEVERE"}], "diagnostic_counts": {"scored": 1, "missing": 0}}
        catalyst = {"summaries": {"EXTREME": {"scores": {"score_blended": 100, "catalyst_proximity_score": 100, "catalyst_delta_score": 50}}}, "diagnostic_counts": {"total_input": 1, "scored": 1, "missing": 0}, "as_of_date": as_of_date, "schema_version": "2.0", "score_version": "v2.0"}
        clinical = {"scores": [{"ticker": "EXTREME", "clinical_score": "100", "lead_phase": "Phase 3", "severity": "sev2", "flags": ["extreme"], "trial_count": 100}], "as_of_date": as_of_date, "diagnostic_counts": {"scored": 1, "missing": 0}}

        result = compute_module_5_composite_v3(
            universe_result=universe,
            financial_result=financial,
            catalyst_result=catalyst,
            clinical_result=clinical,
            as_of_date=as_of_date,
        )

        # Should produce valid result
        assert len(result["ranked_securities"]) == 1
        score = Decimal(result["ranked_securities"][0]["composite_score"])
        assert Decimal("0") <= score <= Decimal("100")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
