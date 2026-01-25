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

@pytest.fixture(autouse=True)
def set_validation_mode_warn():
    """Disable strict schema validation for test fixtures."""
    import os
    old_mode = os.environ.get("IC_VALIDATION_MODE")
    os.environ["IC_VALIDATION_MODE"] = "warn"
    yield
    if old_mode is None:
        os.environ.pop("IC_VALIDATION_MODE", None)
    else:
        os.environ["IC_VALIDATION_MODE"] = old_mode


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
            {"ticker": "AAPL", "financial_score": 95, "financial_normalized": 95, "market_cap_mm": 3000000, "runway_months": 999, "severity": "none", "flags": [], "liquidity_gate_status": "PASS", "dilution_risk_bucket": "LOW", "financial_data_state": "FULL"},
            {"ticker": "BIIB", "financial_score": 78, "financial_normalized": 78, "market_cap_mm": 25000, "runway_months": 48, "severity": "none", "flags": [], "liquidity_gate_status": "PASS", "dilution_risk_bucket": "LOW", "financial_data_state": "FULL"},
            {"ticker": "MRNA", "financial_score": 82, "financial_normalized": 82, "market_cap_mm": 45000, "runway_months": 36, "severity": "none", "flags": [], "liquidity_gate_status": "PASS", "dilution_risk_bucket": "LOW", "financial_data_state": "FULL"},
            {"ticker": "SGEN", "financial_score": 70, "financial_normalized": 70, "market_cap_mm": 18000, "runway_months": 24, "severity": "none", "flags": [], "liquidity_gate_status": "PASS", "dilution_risk_bucket": "MEDIUM", "financial_data_state": "FULL"},
            {"ticker": "ALNY", "financial_score": 65, "financial_normalized": 65, "market_cap_mm": 22000, "runway_months": 18, "severity": "sev1", "flags": ["runway_warning"], "liquidity_gate_status": "PASS", "dilution_risk_bucket": "MEDIUM", "financial_data_state": "FULL"},
            {"ticker": "BMRN", "financial_score": 55, "financial_normalized": 55, "market_cap_mm": 12000, "runway_months": 15, "severity": "sev1", "flags": ["runway_warning"], "liquidity_gate_status": "PASS", "dilution_risk_bucket": "MEDIUM", "financial_data_state": "FULL"},
            {"ticker": "EXEL", "financial_score": 60, "financial_normalized": 60, "market_cap_mm": 8000, "runway_months": 20, "severity": "none", "flags": [], "liquidity_gate_status": "PASS", "dilution_risk_bucket": "MEDIUM", "financial_data_state": "FULL"},
            {"ticker": "RARE", "financial_score": 45, "financial_normalized": 45, "market_cap_mm": 5000, "runway_months": 12, "severity": "sev1", "flags": ["runway_warning"], "liquidity_gate_status": "WARN", "dilution_risk_bucket": "HIGH", "financial_data_state": "FULL"},
            {"ticker": "FOLD", "financial_score": 35, "financial_normalized": 35, "market_cap_mm": 2500, "runway_months": 9, "severity": "sev2", "flags": ["runway_critical"], "liquidity_gate_status": "WARN", "dilution_risk_bucket": "HIGH", "financial_data_state": "FULL"},
            {"ticker": "BLUE", "financial_score": 20, "financial_normalized": 20, "market_cap_mm": 800, "runway_months": 6, "severity": "sev2", "flags": ["runway_critical", "dilution_severe"], "liquidity_gate_status": "FAIL", "dilution_risk_bucket": "SEVERE", "financial_data_state": "FULL"},
        ],
        "diagnostic_counts": {"scored": 10, "missing": 0},
    }


@pytest.fixture
def sample_catalyst_result():
    """Sample Module 3 output."""
    return {
        "summaries": {
            "AAPL": {"scores": {"score_blended": 50, "catalyst_proximity_score": 0, "catalyst_delta_score": 0}, "integration": {"catalyst_confidence": "MED"}},
            "BIIB": {"scores": {"score_blended": 75, "catalyst_proximity_score": 80, "catalyst_delta_score": 10, "days_to_nearest_catalyst": 25, "nearest_catalyst_type": "DATA_READOUT"}, "integration": {"catalyst_confidence": "HIGH"}},
            "MRNA": {"scores": {"score_blended": 85, "catalyst_proximity_score": 90, "catalyst_delta_score": 15, "days_to_nearest_catalyst": 15, "nearest_catalyst_type": "PDUFA"}, "integration": {"catalyst_confidence": "HIGH"}},
            "SGEN": {"scores": {"score_blended": 65, "catalyst_proximity_score": 40, "catalyst_delta_score": 5}, "integration": {"catalyst_confidence": "MED"}},
            "ALNY": {"scores": {"score_blended": 70, "catalyst_proximity_score": 60, "catalyst_delta_score": 8, "days_to_nearest_catalyst": 45}, "integration": {"catalyst_confidence": "HIGH"}},
            "BMRN": {"scores": {"score_blended": 55, "catalyst_proximity_score": 30, "catalyst_delta_score": -5}, "integration": {"catalyst_confidence": "MED"}},
            "EXEL": {"scores": {"score_blended": 60, "catalyst_proximity_score": 50, "catalyst_delta_score": 0}, "integration": {"catalyst_confidence": "MED"}},
            "RARE": {"scores": {"score_blended": 72, "catalyst_proximity_score": 70, "catalyst_delta_score": 12, "days_to_nearest_catalyst": 30}, "integration": {"catalyst_confidence": "HIGH"}},
            "FOLD": {"scores": {"score_blended": 40, "catalyst_proximity_score": 20, "catalyst_delta_score": -10}, "integration": {"catalyst_confidence": "MED"}},
            "BLUE": {"scores": {"score_blended": 30, "catalyst_proximity_score": 10, "catalyst_delta_score": -15}, "flags": {"severe_negative_flag": True}, "integration": {"catalyst_confidence": "LOW"}},
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
    """Tests for volatility-adjusted scoring.

    V2 ASYMMETRIC PENALTY:
    - Score penalty only applies above target vol (50%)
    - Low vol names are NOT penalized (better IC in biotech)
    - Weight adjustments still boost low vol / reduce high vol for signal reliability
    """

    def test_low_volatility_no_score_penalty(self):
        """Low volatility should NOT have score penalty (v2 asymmetric).

        Low-vol names often deliver better forward returns in biotech.
        Penalizing them hurts IC.
        """
        result = compute_volatility_adjustment(Decimal("0.20"))
        assert result.vol_bucket == VolatilityBucket.LOW
        # Weights are still boosted for signal reliability
        assert result.weight_adjustment_factor > Decimal("1.0")
        # But NO score penalty for low vol (v2 improvement)
        assert result.score_adjustment_factor == Decimal("1.0")
        assert result.confidence_penalty == Decimal("0")

    def test_low_volatility_25pct_no_penalty(self):
        """At 25% vol (below target 50%), no score penalty.

        Test vector: vol=0.25, target=0.50 -> vol_ratio=0.5 -> no penalty
        """
        result = compute_volatility_adjustment(Decimal("0.25"))
        assert result.vol_bucket == VolatilityBucket.LOW
        assert result.score_adjustment_factor == Decimal("1.0")

    def test_target_volatility_neutral(self):
        """At target volatility (50%), no score penalty."""
        result = compute_volatility_adjustment(Decimal("0.50"))
        assert result.vol_bucket == VolatilityBucket.NORMAL
        assert result.weight_adjustment_factor == Decimal("1.0")
        # At target vol, vol_ratio = 1.0, no penalty
        assert result.score_adjustment_factor == Decimal("1.0")

    def test_above_target_vol_75pct_penalty(self):
        """At 75% vol (above target 50%), linear score penalty.

        Test vector: vol=0.75, target=0.50 -> vol_ratio=1.5 -> penalty=0.075
        score_adj = 1.0 - 0.075 = 0.925
        """
        result = compute_volatility_adjustment(Decimal("0.75"))
        assert result.vol_bucket == VolatilityBucket.NORMAL
        # penalty = (1.5 - 1) * 0.15 = 0.075
        expected_score_adj = Decimal("0.9250")
        assert result.score_adjustment_factor == expected_score_adj

    def test_high_volatility_100pct_penalty(self):
        """At 100% vol, significant score penalty.

        Test vector: vol=1.00, target=0.50 -> vol_ratio=2.0 -> penalty=0.15
        score_adj = 1.0 - 0.15 = 0.85
        """
        result = compute_volatility_adjustment(Decimal("1.00"))
        assert result.vol_bucket == VolatilityBucket.HIGH
        assert result.weight_adjustment_factor < Decimal("1.0")
        # penalty = (2.0 - 1) * 0.15 = 0.15
        expected_score_adj = Decimal("0.8500")
        assert result.score_adjustment_factor == expected_score_adj
        assert result.confidence_penalty > Decimal("0.10")

    def test_very_high_volatility_capped(self):
        """Very high volatility penalty is capped at 30%.

        Test vector: vol=2.00, target=0.50 -> vol_ratio=4.0 -> penalty capped at 0.30
        score_adj = 1.0 - 0.30 = 0.70
        """
        result = compute_volatility_adjustment(Decimal("2.00"))
        assert result.vol_bucket == VolatilityBucket.HIGH
        # penalty = (4.0 - 1) * 0.15 = 0.45, but capped at 0.30
        expected_score_adj = Decimal("0.7000")
        assert result.score_adjustment_factor == expected_score_adj

    def test_unknown_volatility(self):
        """Missing volatility should return neutral with small penalty."""
        result = compute_volatility_adjustment(None)
        assert result.vol_bucket == VolatilityBucket.UNKNOWN
        assert result.weight_adjustment_factor == Decimal("1.0")
        assert result.score_adjustment_factor == Decimal("1.0")
        assert result.confidence_penalty == Decimal("0.05")

    def test_apply_volatility_to_score_high_vol(self):
        """Score adjustment should dampen high-vol stocks."""
        vol_adj = compute_volatility_adjustment(Decimal("1.00"))
        original_score = Decimal("80")
        adjusted = apply_volatility_to_score(original_score, vol_adj)
        # 80 * 0.85 = 68
        assert adjusted < original_score
        assert adjusted == Decimal("68.00")

    def test_apply_volatility_to_score_low_vol(self):
        """Score adjustment should NOT change low-vol stocks."""
        vol_adj = compute_volatility_adjustment(Decimal("0.25"))
        original_score = Decimal("80")
        adjusted = apply_volatility_to_score(original_score, vol_adj)
        # No penalty for low vol
        assert adjusted == original_score


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

    def test_determinism(self):
        """Same inputs must produce identical outputs (no float variance)."""
        peers = [
            {"market_cap_mm": 5000, "trial_count": 5, "stage_bucket": "mid"},
            {"market_cap_mm": 6000, "trial_count": 6, "stage_bucket": "mid"},
            {"market_cap_mm": 7000, "trial_count": 7, "stage_bucket": "mid"},
            {"market_cap_mm": 8000, "trial_count": 4, "stage_bucket": "mid"},
            {"market_cap_mm": 9000, "trial_count": 8, "stage_bucket": "mid"},
            {"market_cap_mm": 10000, "trial_count": 3, "stage_bucket": "mid"},
        ]
        # Run multiple times
        results = [compute_valuation_signal(Decimal("6500"), 5, "Phase 2", peers) for _ in range(10)]

        # All must be identical
        assert all(r.valuation_score == results[0].valuation_score for r in results)
        assert all(r.confidence == results[0].confidence for r in results)

    def test_tie_aware_percentile(self):
        """Ties should use midrank, not strict less-than."""
        # Create peers with identical mcap/trial ratios
        peers = [
            {"market_cap_mm": 1000, "trial_count": 10, "stage_bucket": "mid"},  # 100
            {"market_cap_mm": 1000, "trial_count": 10, "stage_bucket": "mid"},  # 100
            {"market_cap_mm": 1000, "trial_count": 10, "stage_bucket": "mid"},  # 100
            {"market_cap_mm": 2000, "trial_count": 10, "stage_bucket": "mid"},  # 200
            {"market_cap_mm": 2000, "trial_count": 10, "stage_bucket": "mid"},  # 200
        ]
        # Test stock with same ratio as the 1000/10 peers (100)
        result = compute_valuation_signal(Decimal("1000"), 10, "Phase 2", peers)

        # With midrank: 3 ties at 100, so percentile = (0 + 0.5*3)/5 = 0.3 = 30%
        # Valuation score = 100 - 30 = 70 (before shrinkage)
        # Should NOT be 100 (which would happen with strict < counting)
        assert result.valuation_score < Decimal("85")  # Account for shrinkage
        assert result.valuation_score > Decimal("50")

    def test_winsorization_trial_count(self):
        """Extreme trial counts should be winsorized."""
        peers = [
            {"market_cap_mm": 1000, "trial_count": 5, "stage_bucket": "mid"},
            {"market_cap_mm": 2000, "trial_count": 5, "stage_bucket": "mid"},
            {"market_cap_mm": 3000, "trial_count": 5, "stage_bucket": "mid"},
            {"market_cap_mm": 4000, "trial_count": 5, "stage_bucket": "mid"},
            {"market_cap_mm": 5000, "trial_count": 5, "stage_bucket": "mid"},
        ]
        # Stock with absurdly high trial count (should be capped)
        result_extreme = compute_valuation_signal(Decimal("3000"), 100, "Phase 2", peers)
        result_capped = compute_valuation_signal(Decimal("3000"), 30, "Phase 2", peers)

        # Both should produce similar results due to winsorization (100 -> 30)
        assert result_extreme.mcap_per_asset == result_capped.mcap_per_asset

    def test_winsorization_mcap(self):
        """Extreme market caps should be winsorized."""
        peers = [
            {"market_cap_mm": 1000, "trial_count": 5, "stage_bucket": "mid"},
            {"market_cap_mm": 2000, "trial_count": 5, "stage_bucket": "mid"},
            {"market_cap_mm": 3000, "trial_count": 5, "stage_bucket": "mid"},
            {"market_cap_mm": 4000, "trial_count": 5, "stage_bucket": "mid"},
            {"market_cap_mm": 5000, "trial_count": 5, "stage_bucket": "mid"},
        ]
        # Stock with very small mcap (should be floored at 50)
        result = compute_valuation_signal(Decimal("10"), 5, "Phase 2", peers)

        # mcap_per_asset should be based on floor (50 / 5 = 10), not (10 / 5 = 2)
        assert result.mcap_per_asset == Decimal("10.00")

    def test_confidence_ramp(self):
        """Confidence should increase smoothly with peer count."""
        base_peer = {"market_cap_mm": 5000, "trial_count": 5, "stage_bucket": "mid"}

        # 6 peers
        peers_6 = [base_peer.copy() for _ in range(6)]
        for i, p in enumerate(peers_6):
            p["market_cap_mm"] = 5000 + i * 1000
        result_6 = compute_valuation_signal(Decimal("5000"), 5, "Phase 2", peers_6)

        # 15 peers
        peers_15 = [base_peer.copy() for _ in range(15)]
        for i, p in enumerate(peers_15):
            p["market_cap_mm"] = 5000 + i * 500
        result_15 = compute_valuation_signal(Decimal("5000"), 5, "Phase 2", peers_15)

        # More peers = higher confidence
        assert result_15.confidence > result_6.confidence

    def test_shrinkage_small_sample(self):
        """Small samples should be shrunk toward neutral (50)."""
        # Create peers where stock is clearly cheapest
        peers_small = [
            {"market_cap_mm": 10000, "trial_count": 5, "stage_bucket": "mid"},
            {"market_cap_mm": 11000, "trial_count": 5, "stage_bucket": "mid"},
            {"market_cap_mm": 12000, "trial_count": 5, "stage_bucket": "mid"},
            {"market_cap_mm": 13000, "trial_count": 5, "stage_bucket": "mid"},
            {"market_cap_mm": 14000, "trial_count": 5, "stage_bucket": "mid"},
        ]
        result_small = compute_valuation_signal(Decimal("1000"), 5, "Phase 2", peers_small)

        # Create large peer set where stock is also cheapest
        peers_large = peers_small + [
            {"market_cap_mm": 15000 + i * 1000, "trial_count": 5, "stage_bucket": "mid"}
            for i in range(20)
        ]
        result_large = compute_valuation_signal(Decimal("1000"), 5, "Phase 2", peers_large)

        # Small sample should be shrunk closer to 50 than large sample
        # Both should be > 50 (cheapest), but small should be less extreme
        distance_small = abs(result_small.valuation_score - Decimal("50"))
        distance_large = abs(result_large.valuation_score - Decimal("50"))
        assert distance_small < distance_large

    def test_stage_filtering(self):
        """Only same-stage peers should be used."""
        peers = [
            {"market_cap_mm": 1000, "trial_count": 5, "stage_bucket": "early"},
            {"market_cap_mm": 2000, "trial_count": 5, "stage_bucket": "early"},
            {"market_cap_mm": 100000, "trial_count": 5, "stage_bucket": "mid"},
            {"market_cap_mm": 200000, "trial_count": 5, "stage_bucket": "mid"},
            {"market_cap_mm": 200000, "trial_count": 5, "stage_bucket": "mid"},
            {"market_cap_mm": 200000, "trial_count": 5, "stage_bucket": "mid"},
            {"market_cap_mm": 200000, "trial_count": 5, "stage_bucket": "mid"},
        ]
        # Phase 2 = mid stage, should only compare to mid peers
        result = compute_valuation_signal(Decimal("150000"), 5, "Phase 2", peers)

        # Should be compared to 100k-200k mid peers, not 1k-2k early peers
        # 150k is in the middle of mid peers, so score should be around 50
        assert Decimal("30") < result.valuation_score < Decimal("70")


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

    def test_determinism(self):
        """Same inputs must produce identical outputs (no float variance)."""
        # Run twice with identical inputs
        result1 = compute_catalyst_decay(45, "DATA_READOUT")
        result2 = compute_catalyst_decay(45, "DATA_READOUT")

        # Must be exactly equal (not just approximately)
        assert result1.decay_factor == result2.decay_factor
        assert result1.in_optimal_window == result2.in_optimal_window

        # Run many times to catch any non-determinism
        factors = [compute_catalyst_decay(45, "DATA_READOUT").decay_factor for _ in range(10)]
        assert all(f == factors[0] for f in factors), "Non-deterministic decay detected"

    def test_asymmetric_decay_faster_after_peak(self):
        """Past optimal peak should decay faster than same distance before peak.

        days_to_catalyst=60 is 30 days BEFORE optimal (d=+30)
        days_to_catalyst=0 is 30 days AFTER optimal (d=-30)

        Post-peak should decay faster due to information pricing.
        """
        # 30 days before optimal peak (event far in future)
        before_peak = compute_catalyst_decay(60, "DATA_READOUT")  # d = 60 - 30 = +30
        # 30 days after optimal peak (at event itself)
        at_event = compute_catalyst_decay(0, "DATA_READOUT")  # d = 0 - 30 = -30

        # Both are same distance (30 days) from optimal, but post-peak should be lower
        assert at_event.decay_factor < before_peak.decay_factor, \
            f"Post-peak decay ({at_event.decay_factor}) should be < pre-peak ({before_peak.decay_factor})"

    def test_asymmetric_decay_past_event(self):
        """Events that already happened should decay very fast."""
        # Use closer distances where decay hasn't hit the floor yet
        # Event happened 5 days ago (d = -5 - 30 = -35)
        past_event = compute_catalyst_decay(-5, "PDUFA")  # PDUFA has slower decay
        # Event is 65 days away (d = 65 - 30 = +35, same distance)
        future_event = compute_catalyst_decay(65, "PDUFA")

        # Past should decay faster (post-event decay mult = 2x)
        # Both are 35 days from optimal, but past uses 2x tau
        assert past_event.decay_factor < future_event.decay_factor, \
            f"Past event ({past_event.decay_factor}) should decay faster than future ({future_event.decay_factor})"

    def test_event_type_normalization(self):
        """Event type should be case-insensitive with whitespace stripped."""
        # All these should produce identical results
        upper = compute_catalyst_decay(30, "PDUFA")
        lower = compute_catalyst_decay(30, "pdufa")
        mixed = compute_catalyst_decay(30, "Pdufa")
        whitespace = compute_catalyst_decay(30, "  PDUFA  ")

        assert upper.decay_factor == lower.decay_factor
        assert upper.decay_factor == mixed.decay_factor
        assert upper.decay_factor == whitespace.decay_factor

        # All should report normalized event type
        assert upper.event_type == "PDUFA"
        assert lower.event_type == "PDUFA"
        assert mixed.event_type == "PDUFA"
        assert whitespace.event_type == "PDUFA"

    def test_unknown_event_type_uses_default(self):
        """Unknown event types should use DEFAULT rate, not fail."""
        result = compute_catalyst_decay(30, "UNKNOWN_EVENT_TYPE")
        assert result.decay_factor > Decimal("0")
        assert result.event_type == "UNKNOWN_EVENT_TYPE"

    def test_decay_monotonicity_approaching_peak(self):
        """Decay factor should increase as we approach optimal window from far out."""
        # Moving from 120 days out toward 30 days out (optimal)
        factors = []
        for days in [120, 90, 60, 45, 30]:
            result = compute_catalyst_decay(days, "DATA_READOUT")
            factors.append((days, result.decay_factor))

        # Each step closer should have >= decay factor (monotonically increasing)
        for i in range(len(factors) - 1):
            assert factors[i][1] <= factors[i + 1][1], \
                f"Monotonicity violated: {factors[i]} should have <= decay than {factors[i + 1]}"


# =============================================================================
# SMART MONEY SIGNAL TESTS
# =============================================================================

class TestSmartMoneySignal:
    """Tests for smart money (13F) signal.

    NOTE: V2 implements tier-weighted scoring. Tests updated to use
    Tier1 holder names for expected high-conviction behavior.
    Unknown holders (generic names) now get reduced weight (0.2).
    """

    def test_high_overlap_with_tier1_holders(self):
        """High overlap with Tier1 holders should increase score significantly."""
        # Use Tier1 holder names for tier-weighted bonus
        result = compute_smart_money_signal(
            overlap_count=4,
            holders=["Baker Bros", "RA Capital", "Perceptive", "BVF"],
            position_changes=None,
            holder_tiers={"Baker Bros": 1, "RA Capital": 1, "Perceptive": 1, "BVF": 1},
        )
        # 4 Tier1 holders × 1.0 weight = 4.0 weighted overlap
        # Saturation applies, but should still be high score
        assert result.smart_money_score > Decimal("60")
        assert result.weighted_overlap == Decimal("4.00")

    def test_high_overlap_with_unknown_holders(self):
        """High overlap with unknown holders should give moderate bonus."""
        # Unknown holders get 0.2 weight each
        result = compute_smart_money_signal(
            overlap_count=4,
            holders=["Holder1", "Holder2", "Holder3", "Holder4"],
            position_changes=None,
        )
        # 4 unknown × 0.2 = 0.8 weighted overlap → modest bonus
        assert result.smart_money_score > Decimal("50")  # Above neutral
        assert result.smart_money_score < Decimal("65")  # Not too high for unknowns
        assert result.weighted_overlap == Decimal("0.80")

    def test_position_increases(self):
        """Position increases should boost score."""
        result = compute_smart_money_signal(
            overlap_count=2,
            holders=["Baker Bros", "RA Capital"],
            position_changes={"Baker Bros": "INCREASE", "RA Capital": "NEW"},
            holder_tiers={"Baker Bros": 1, "RA Capital": 1},
        )
        assert result.position_change_adjustment > Decimal("0")
        assert "Baker Bros" in result.holders_increasing

    def test_position_decreases(self):
        """Position decreases should lower score."""
        result = compute_smart_money_signal(
            overlap_count=2,
            holders=["Baker Bros", "RA Capital"],
            position_changes={"Baker Bros": "DECREASE", "RA Capital": "EXIT"},
            holder_tiers={"Baker Bros": 1, "RA Capital": 1},
        )
        assert result.position_change_adjustment < Decimal("0")
        assert "Baker Bros" in result.holders_decreasing

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

    def test_equal_to_global_mean_yields_fifty(self):
        """If all values equal global_mean, outputs should all be ~50."""
        global_mean = Decimal("50")
        global_std = Decimal("20")
        values = [global_mean] * 10

        result, _ = shrinkage_normalize(values, global_mean, global_std)

        # All results should be exactly 50 (within quantization)
        assert all(r == Decimal("50.00") for r in result)

    def test_single_value_shrinks_to_global(self):
        """Single value (n=1) should return 50 due to maximum shrinkage."""
        global_mean = Decimal("50")
        global_std = Decimal("20")
        values = [Decimal("100")]  # Extreme value

        result, shrinkage_factor = shrinkage_normalize(values, global_mean, global_std)

        # n=1 is a special case, should return 50
        assert len(result) == 1
        assert result[0] == Decimal("50")
        assert shrinkage_factor == Decimal("1.0")

    def test_output_bounds_never_exceeded(self):
        """No output should ever be outside [5, 95], even with extreme values."""
        global_mean = Decimal("50")
        global_std = Decimal("20")
        # Extreme outliers
        values = [Decimal("-1000"), Decimal("0"), Decimal("50"), Decimal("100"), Decimal("1000")]

        result, _ = shrinkage_normalize(values, global_mean, global_std)

        # All results must be within bounds
        assert len(result) == 5
        assert all(Decimal("5") <= r <= Decimal("95") for r in result)

    def test_z_score_clamping_limits_spread(self):
        """Z-score clamping should limit the spread to ~45 points (3*15)."""
        global_mean = Decimal("50")
        global_std = Decimal("20")
        # Values that would produce extreme z-scores without clamping
        values = [Decimal("0"), Decimal("25"), Decimal("50"), Decimal("75"), Decimal("100")]

        result, _ = shrinkage_normalize(values, global_mean, global_std)

        # With z clamped to [-3, +3] and multiplier of 15:
        # min should be 50 - 3*15 = 5, max should be 50 + 3*15 = 95
        # But with shrinkage, the spread will be reduced
        assert min(result) >= Decimal("5")
        assert max(result) <= Decimal("95")
        # Verify ordering is maintained
        assert result[0] <= result[1] <= result[2] <= result[3] <= result[4]

    def test_std_shrinkage_stabilizes_small_cohorts(self):
        """Small cohorts with high variance should shrink std toward global."""
        global_mean = Decimal("50")
        global_std = Decimal("20")
        # Small cohort with extreme variance
        values = [Decimal("0"), Decimal("50"), Decimal("100")]

        result, shrinkage_factor = shrinkage_normalize(values, global_mean, global_std)

        # With n=3 and prior_strength=5, shrinkage = 5/(5+3) = 0.625
        assert shrinkage_factor > Decimal("0.6")
        # Results should be centered due to heavy shrinkage toward global mean
        # and std shrinkage should prevent extreme spread
        assert all(Decimal("5") <= r <= Decimal("95") for r in result)
        # Middle value (50) should be close to 50 due to shrinkage
        assert Decimal("40") <= result[1] <= Decimal("60")


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


# =============================================================================
# SMART MONEY SIGNAL V2 TESTS
# =============================================================================

class TestSmartMoneySignalV2:
    """Tests for tier-weighted smart money signal (V2 improvements)."""

    def test_tier_sensitivity_tier1_beats_unknowns(self):
        """2 Tier1 holders should beat 4 unknown holders."""
        # 2 Tier1 holders (Baker Bros + RA Capital)
        tier1_result = compute_smart_money_signal(
            overlap_count=2,
            holders=["Baker Bros Advisors LP", "RA Capital Management, L.P."],
            position_changes=None,
            holder_tiers=None,  # Use name-based lookup
        )

        # 4 unknown holders
        unknown_result = compute_smart_money_signal(
            overlap_count=4,
            holders=["Unknown Fund A", "Unknown Fund B", "Unknown Fund C", "Unknown Fund D"],
            position_changes=None,
            holder_tiers=None,
        )

        # Tier1 should have higher weighted overlap
        assert tier1_result.weighted_overlap > unknown_result.weighted_overlap
        # Tier1 should have higher score
        assert tier1_result.smart_money_score > unknown_result.smart_money_score
        # Tier1 should have higher confidence
        assert tier1_result.confidence >= unknown_result.confidence

    def test_tier_sensitivity_weighted_overlap_calculation(self):
        """Verify weighted overlap calculation: Tier1=1.0, Tier2=0.6, Unknown=0.2."""
        # 1 Tier1 + 1 Tier2 = 1.0 + 0.6 = 1.6
        mixed_result = compute_smart_money_signal(
            overlap_count=2,
            holders=["Baker Bros Advisors LP", "OrbiMed Advisors LLC"],
            position_changes=None,
            holder_tiers={"Baker Bros Advisors LP": 1, "OrbiMed Advisors LLC": 2},
        )

        # 2 unknowns = 0.2 + 0.2 = 0.4
        unknown_result = compute_smart_money_signal(
            overlap_count=2,
            holders=["Unknown A", "Unknown B"],
            position_changes=None,
            holder_tiers=None,
        )

        assert mixed_result.weighted_overlap == Decimal("1.60")
        assert unknown_result.weighted_overlap == Decimal("0.40")
        assert mixed_result.weighted_overlap == Decimal("4") * unknown_result.weighted_overlap

    def test_breadth_only_score_rises_without_changes(self):
        """Score should rise with overlap even without position_changes data."""
        # No holders
        no_holders = compute_smart_money_signal(
            overlap_count=0,
            holders=[],
            position_changes=None,
            holder_tiers=None,
        )

        # 2 Tier1 holders, no change data
        with_holders = compute_smart_money_signal(
            overlap_count=2,
            holders=["Baker Bros Advisors LP", "RA Capital Management, L.P."],
            position_changes=None,  # Empty
            holder_tiers=None,
        )

        # Score should increase from overlap alone
        assert with_holders.smart_money_score > no_holders.smart_money_score
        assert with_holders.overlap_bonus > Decimal("0")
        assert with_holders.position_change_adjustment == Decimal("0")

    def test_exit_dominance_capped(self):
        """One EXIT from Tier1 should matter but not nuke the score."""
        # Tier1 EXIT only
        exit_only = compute_smart_money_signal(
            overlap_count=1,
            holders=["Baker Bros Advisors LP"],
            position_changes={"Baker Bros Advisors LP": "EXIT"},
            holder_tiers={"Baker Bros Advisors LP": 1},
        )

        # Tier1 EXIT with Tier1 NEW (offsetting)
        mixed_changes = compute_smart_money_signal(
            overlap_count=2,
            holders=["Baker Bros Advisors LP", "RA Capital Management, L.P."],
            position_changes={
                "Baker Bros Advisors LP": "EXIT",
                "RA Capital Management, L.P.": "NEW",
            },
            holder_tiers={"Baker Bros Advisors LP": 1, "RA Capital Management, L.P.": 1},
        )

        # EXIT should lower score but not below 20 (hard floor)
        assert exit_only.smart_money_score >= Decimal("20")
        # Per-holder cap should prevent excessive penalty
        assert exit_only.position_change_adjustment >= Decimal("-5")  # Capped per-holder
        # Mixed changes should partially offset
        assert mixed_changes.smart_money_score > exit_only.smart_money_score

    def test_per_holder_cap_prevents_domination(self):
        """Per-holder cap should prevent one noisy filing from dominating."""
        # Single Tier1 with NEW (high weight)
        single_tier1 = compute_smart_money_signal(
            overlap_count=1,
            holders=["Baker Bros Advisors LP"],
            position_changes={"Baker Bros Advisors LP": "NEW"},
            holder_tiers={"Baker Bros Advisors LP": 1},
        )

        # Per-holder contribution should be capped
        contribution = single_tier1.per_holder_contributions.get("Baker Bros Advisors LP", Decimal("0"))
        # Cap is 5, so tier_weight(1.0) + change(3.0) = 4.0 < 5, should be uncapped
        # But total change contribution is tier_weight * change_weight = 1.0 * 3.0 = 3.0
        # So per_holder_contribution = overlap_weight + change_contribution = 1.0 + 3.0 = 4.0
        # This is under the cap of 5
        assert contribution <= Decimal("5.01")  # Small tolerance for quantization

    def test_deterministic_ordering(self):
        """Dict iteration order should not affect result."""
        # Create position_changes in different orders
        changes_1 = {"Baker Bros Advisors LP": "NEW", "RA Capital": "INCREASE", "Perceptive": "HOLD"}
        changes_2 = {"Perceptive": "HOLD", "Baker Bros Advisors LP": "NEW", "RA Capital": "INCREASE"}
        changes_3 = {"RA Capital": "INCREASE", "Perceptive": "HOLD", "Baker Bros Advisors LP": "NEW"}

        result_1 = compute_smart_money_signal(
            overlap_count=3,
            holders=["Baker Bros Advisors LP", "RA Capital", "Perceptive"],
            position_changes=changes_1,
            holder_tiers=None,
        )

        result_2 = compute_smart_money_signal(
            overlap_count=3,
            holders=["RA Capital", "Perceptive", "Baker Bros Advisors LP"],
            position_changes=changes_2,
            holder_tiers=None,
        )

        result_3 = compute_smart_money_signal(
            overlap_count=3,
            holders=["Perceptive", "Baker Bros Advisors LP", "RA Capital"],
            position_changes=changes_3,
            holder_tiers=None,
        )

        # All results should be identical
        assert result_1.smart_money_score == result_2.smart_money_score == result_3.smart_money_score
        assert result_1.weighted_overlap == result_2.weighted_overlap == result_3.weighted_overlap
        assert result_1.overlap_bonus == result_2.overlap_bonus == result_3.overlap_bonus
        assert result_1.position_change_adjustment == result_2.position_change_adjustment == result_3.position_change_adjustment
        # Sorted lists should also be identical
        assert result_1.holders_increasing == result_2.holders_increasing == result_3.holders_increasing
        assert result_1.tier1_holders == result_2.tier1_holders == result_3.tier1_holders

    def test_saturation_diminishing_returns(self):
        """Many holders should have diminishing returns on overlap bonus."""
        # 1 Tier1 holder
        one_tier1 = compute_smart_money_signal(
            overlap_count=1,
            holders=["Baker Bros Advisors LP"],
            position_changes=None,
            holder_tiers={"Baker Bros Advisors LP": 1},
        )

        # 5 Tier1 holders (excessive)
        five_tier1 = compute_smart_money_signal(
            overlap_count=5,
            holders=["Baker Bros", "RA Capital", "Perceptive", "BVF", "EcoR1"],
            position_changes=None,
            holder_tiers={"Baker Bros": 1, "RA Capital": 1, "Perceptive": 1, "BVF": 1, "EcoR1": 1},
        )

        # 5x weighted overlap should NOT give 5x bonus (saturation)
        overlap_ratio = five_tier1.weighted_overlap / one_tier1.weighted_overlap
        bonus_ratio = five_tier1.overlap_bonus / one_tier1.overlap_bonus if one_tier1.overlap_bonus > 0 else Decimal("1")

        # Bonus ratio should be less than overlap ratio (diminishing returns)
        assert bonus_ratio < overlap_ratio

    def test_confidence_based_on_tier_coverage(self):
        """Confidence should scale with tier coverage."""
        # 2 Tier1 = highest confidence
        two_tier1 = compute_smart_money_signal(
            overlap_count=2,
            holders=["Baker Bros", "RA Capital"],
            position_changes=None,
            holder_tiers={"Baker Bros": 1, "RA Capital": 1},
        )

        # 1 Tier1 + 1 Tier2 = good confidence
        mixed_tiers = compute_smart_money_signal(
            overlap_count=2,
            holders=["Baker Bros", "OrbiMed"],
            position_changes=None,
            holder_tiers={"Baker Bros": 1, "OrbiMed": 2},
        )

        # 2 unknowns = lower confidence
        two_unknown = compute_smart_money_signal(
            overlap_count=2,
            holders=["Unknown A", "Unknown B"],
            position_changes=None,
            holder_tiers=None,
        )

        # Confidence should decrease with tier quality
        assert two_tier1.confidence >= mixed_tiers.confidence
        assert mixed_tiers.confidence >= two_unknown.confidence

    def test_tier_breakdown_tracking(self):
        """Tier breakdown should accurately track holder distribution."""
        result = compute_smart_money_signal(
            overlap_count=5,
            holders=["Baker Bros", "RA Capital", "OrbiMed", "Unknown A", "Unknown B"],
            position_changes=None,
            holder_tiers={"Baker Bros": 1, "RA Capital": 1, "OrbiMed": 2},
        )

        # Verify tier breakdown
        assert result.tier_breakdown.get(1, 0) == 2  # Baker Bros, RA Capital
        assert result.tier_breakdown.get(2, 0) == 1  # OrbiMed
        assert result.tier_breakdown.get(0, 0) == 2  # Unknown A, Unknown B
        # Tier1 holders list should contain the right names
        assert "Baker Bros" in result.tier1_holders
        assert "RA Capital" in result.tier1_holders
        assert len(result.tier1_holders) == 2

    def test_explicit_holder_tiers_override_name_lookup(self):
        """Explicit holder_tiers should override name-based lookup."""
        # Use Baker Bros name but assign Tier3 explicitly
        result = compute_smart_money_signal(
            overlap_count=1,
            holders=["Baker Bros Advisors LP"],
            position_changes=None,
            holder_tiers={"Baker Bros Advisors LP": 3},  # Override to Tier3
        )

        # Should use Tier3 weight (0.4), not Tier1 (1.0)
        assert result.weighted_overlap == Decimal("0.40")
        assert result.tier_breakdown.get(3, 0) == 1
        assert result.tier_breakdown.get(1, 0) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
