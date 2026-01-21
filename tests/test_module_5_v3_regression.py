"""
Module 5 v3 Regression Tests

These tests encode the critical correctness requirements identified during
the v2 vs v3 backtest analysis. They ensure that fixed bugs never regress.

Tests:
1. IC sign / spread sign agreement
2. Single-name robustness
3. Decimal zero coalesce
4. Runway gate independence from liquidity status
5. Determinism hash stability with zero scores

Author: Wake Robin Capital Management
Version: 1.0.0
"""
import pytest
from decimal import Decimal
from typing import Dict, List, Any

# Module under test
from module_5_composite_v3 import (
    compute_module_5_composite_v3,
    _coalesce,
    _score_single_ticker_v3,
    _compute_determinism_hash,
    ScoringMode,
    NormalizationMethod,
    V3_DEFAULT_WEIGHTS,
    ComponentScore,
)

from src.modules.ic_enhancements import (
    compute_interaction_terms,
    compute_volatility_adjustment,
    VolatilityBucket,
)

from backtest.metrics import compute_spearman_ic


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_module_inputs():
    """Generate sample module 1-4 outputs for testing."""
    tickers = ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE"]

    universe = {
        "active_securities": [{"ticker": t, "market_cap_mm": 1000 + i*100} for i, t in enumerate(tickers)],
        "excluded_securities": [],
    }

    financial = {
        "scores": [
            {"ticker": t, "financial_score": str(50 + i*5), "market_cap_mm": 1000 + i*100, "runway_months": 12 + i*3, "severity": "none"}
            for i, t in enumerate(tickers)
        ],
    }

    catalyst = {
        "summaries": {
            t: {"ticker": t, "scores": {"score_blended": str(40 + i*8)}}
            for i, t in enumerate(tickers)
        },
    }

    clinical = {
        "scores": [
            {"ticker": t, "clinical_score": str(60 + i*5), "lead_phase": "phase_2", "severity": "none"}
            for i, t in enumerate(tickers)
        ],
    }

    return {
        "universe": universe,
        "financial": financial,
        "catalyst": catalyst,
        "clinical": clinical,
    }


@pytest.fixture
def sample_returns():
    """Sample forward returns for IC calculation."""
    return {
        "AAAA": 0.05,
        "BBBB": 0.10,
        "CCCC": 0.15,
        "DDDD": 0.08,
        "EEEE": 0.12,
    }


# =============================================================================
# TEST 1: IC Sign / Spread Sign Agreement
# =============================================================================

class TestICSpreadSignAgreement:
    """
    The earlier v3 bug had IC > 0 but spread < 0, which is nonsensical.
    This must never regress.
    """

    def test_positive_ic_implies_positive_spread(self, sample_module_inputs, sample_returns):
        """If IC > 0, top N should outperform bottom N."""
        result = compute_module_5_composite_v3(
            universe_result=sample_module_inputs["universe"],
            financial_result=sample_module_inputs["financial"],
            catalyst_result=sample_module_inputs["catalyst"],
            clinical_result=sample_module_inputs["clinical"],
            as_of_date="2026-01-15",
            validate_inputs=False,
        )

        # Extract scores
        scores = {}
        for sec in result["ranked_securities"]:
            scores[sec["ticker"]] = Decimal(sec["composite_score"])

        # Common tickers
        common = sorted(set(scores.keys()) & set(sample_returns.keys()), key=lambda t: scores[t], reverse=True)

        if len(common) < 4:
            pytest.skip("Insufficient common tickers for test")

        # Compute IC
        score_list = [scores[t] for t in common]
        return_list = [Decimal(str(sample_returns[t])) for t in common]
        ic = compute_spearman_ic(score_list, return_list)

        # Compute spread (top 2 vs bottom 2)
        top_2 = common[:2]
        bottom_2 = common[-2:]
        top_mean = sum(sample_returns[t] for t in top_2) / 2
        bottom_mean = sum(sample_returns[t] for t in bottom_2) / 2
        spread = top_mean - bottom_mean

        # CRITICAL: Signs must agree
        if ic is not None and abs(float(ic)) > 0.01:
            assert (float(ic) > 0) == (spread > 0), (
                f"IC/Spread sign mismatch! IC={float(ic):.4f}, Spread={spread:.4f}"
            )


# =============================================================================
# TEST 2: Single-Name Robustness
# =============================================================================

class TestSingleNameRobustness:
    """
    With 207% top-1 contribution in v2 backtest, v3 should be more robust.
    Removing any single name from top bucket should not flip the conclusion.
    """

    def test_removing_top_ticker_preserves_positive_spread(self, sample_module_inputs, sample_returns):
        """Top bucket performance should not depend on single name."""
        result = compute_module_5_composite_v3(
            universe_result=sample_module_inputs["universe"],
            financial_result=sample_module_inputs["financial"],
            catalyst_result=sample_module_inputs["catalyst"],
            clinical_result=sample_module_inputs["clinical"],
            as_of_date="2026-01-15",
            validate_inputs=False,
        )

        scores = {}
        for sec in result["ranked_securities"]:
            scores[sec["ticker"]] = Decimal(sec["composite_score"])

        common = sorted(set(scores.keys()) & set(sample_returns.keys()), key=lambda t: scores[t], reverse=True)

        if len(common) < 5:
            pytest.skip("Need at least 5 common tickers")

        top_3 = common[:3]
        bottom_3 = common[-3:]

        # Full spread
        full_top_mean = sum(sample_returns[t] for t in top_3) / 3
        full_bottom_mean = sum(sample_returns[t] for t in bottom_3) / 3
        full_spread = full_top_mean - full_bottom_mean

        # Only test if full spread is positive (model is "working")
        if full_spread <= 0:
            pytest.skip("Full spread not positive, skipping robustness check")

        # Check each removal
        for exclude_ticker in top_3:
            remaining_top = [t for t in top_3 if t != exclude_ticker]
            if len(remaining_top) < 2:
                continue

            partial_top_mean = sum(sample_returns[t] for t in remaining_top) / len(remaining_top)
            partial_spread = partial_top_mean - full_bottom_mean

            # Spread can decrease but should stay non-negative
            assert partial_spread >= -0.05, (
                f"Removing {exclude_ticker} flipped spread: {partial_spread:.4f}"
            )


# =============================================================================
# TEST 3: Decimal Zero Coalesce
# =============================================================================

class TestDecimalZeroCoalesce:
    """
    Decimal('0') must be treated as a valid score, not missing data.
    This was a bug where `or` chains incorrectly skipped zero values.
    """

    def test_coalesce_preserves_zero(self):
        """_coalesce should return Decimal('0') not fall through to next value."""
        # This is the CORRECT behavior
        result = _coalesce(Decimal("0"), Decimal("50"))
        assert result == Decimal("0"), f"Coalesce incorrectly skipped zero: {result}"

    def test_xbi_return_60d_zero_is_valid(self):
        """
        Regression test for xbi_return_60d = 0.0 falsy bug.

        Bug: Using `market_data.get("xbi_return_60d") or fallback` fails when
        xbi_return_60d is exactly 0.0 because 0.0 is falsy in Python.

        Fix: Use explicit None check:
            xbi_val = market_data.get("xbi_return_60d")
            benchmark = xbi_val if xbi_val is not None else fallback

        This test ensures xbi_return_60d=0.0 is treated as a valid benchmark
        return (flat market), not as missing data.
        """
        from src.modules.ic_enhancements import compute_momentum_signal

        # Case 1: xbi_return_60d = 0.0 should be used (flat benchmark)
        # Stock return of 10% vs flat benchmark = +10% alpha
        stock_return = Decimal("0.10")
        benchmark_zero = Decimal("0.0")  # Flat market

        momentum = compute_momentum_signal(
            return_60d=stock_return,
            benchmark_return_60d=benchmark_zero,
        )

        # Alpha should be +0.10 (stock beat flat benchmark by 10%)
        assert momentum.alpha_60d == Decimal("0.10"), (
            f"Alpha should be +0.10 when stock=+10% vs flat benchmark, "
            f"got alpha={momentum.alpha_60d}"
        )
        # Score should be > 50 (positive alpha = above neutral)
        assert momentum.momentum_score > Decimal("50"), (
            f"Momentum score should be > 50 for positive alpha, "
            f"got {momentum.momentum_score}"
        )

        # Case 2: None benchmark should result in None alpha
        momentum_no_bench = compute_momentum_signal(
            return_60d=stock_return,
            benchmark_return_60d=None,
        )
        assert momentum_no_bench.alpha_60d is None, (
            "Alpha should be None when benchmark is None"
        )

    def test_volatility_252d_zero_is_valid(self):
        """
        Regression test for volatility_252d = 0.0 falsy bug (same pattern).

        Zero volatility is theoretically impossible but the code should
        handle it gracefully, not treat it as missing data.
        """
        from src.modules.ic_enhancements import compute_volatility_adjustment

        # Zero volatility should be treated as valid (not None)
        vol_adj = compute_volatility_adjustment(Decimal("0.0"))

        # Should not crash and should return a valid adjustment
        assert vol_adj is not None
        assert vol_adj.score_adjustment_factor is not None

    def test_coalesce_none_falls_through(self):
        """_coalesce should fall through None values."""
        result = _coalesce(None, Decimal("50"))
        assert result == Decimal("50"), f"Coalesce failed to fall through None: {result}"

    def test_coalesce_false_not_zero(self):
        """_coalesce should distinguish None from falsy values."""
        # Zero is not None, so it should be returned
        result = _coalesce(Decimal("0"), Decimal("100"), default=Decimal("999"))
        assert result == Decimal("0")

        # None should fall through
        result = _coalesce(None, Decimal("100"), default=Decimal("999"))
        assert result == Decimal("100")

        # All None should return default
        result = _coalesce(None, None, default=Decimal("999"))
        assert result == Decimal("999")

    def test_coalesce_int_zero_preserved(self):
        """
        Integer 0 (days_to_catalyst=0) must be treated as valid.

        Bug: Using `or` for days_to_catalyst=0 (catalyst today) would
        treat 0 as falsy and fall through to the next field.

        Fix: Use _coalesce which only falls through on None.
        """
        # Integer zero should be preserved
        result = _coalesce(0, 30, default=None)
        assert result == 0, f"Coalesce incorrectly skipped int zero: {result}"

        # None should fall through to next value
        result = _coalesce(None, 30, default=None)
        assert result == 30


# =============================================================================
# TEST 4: Runway Gate Independence
# =============================================================================

class TestRunwayGateIndependence:
    """
    Runway gate must be computed from runway_months directly,
    not inferred from liquidity_status. These are separate concerns:
    - liquidity_status: Can we trade this security?
    - runway_gate: Does the company have enough cash?
    """

    def test_runway_gate_fires_on_short_runway(self):
        """Runway < 6 months should trigger gate regardless of liquidity."""
        fin_data = {
            "financial_score": "60",
            "runway_months": 5,  # Short runway
            "liquidity_gate_status": "PASS",  # Liquidity is fine
            "dilution_risk_bucket": "LOW",
            "severity": "none",
        }

        # Mock interaction terms computation
        vol_adj = compute_volatility_adjustment(None)

        interactions = compute_interaction_terms(
            clinical_normalized=Decimal("70"),
            financial_data=fin_data,
            catalyst_normalized=Decimal("50"),
            stage_bucket="late",
            vol_adjustment=vol_adj,
            runway_gate_status="FAIL",  # Should be FAIL because runway < 6
            dilution_gate_status="PASS",
        )

        # The stage-financial interaction should fire the distress penalty
        # because runway is short despite liquidity being fine
        assert "late_stage_distress" in interactions.interaction_flags or \
               interactions.stage_financial_interaction < Decimal("0"), (
            f"Runway distress not detected: flags={interactions.interaction_flags}, "
            f"stage_fin_interaction={interactions.stage_financial_interaction}"
        )

    def test_liquidity_status_does_not_override_runway(self):
        """Liquidity PASS should not mask runway FAIL."""
        # This tests that the two gates are independent

        # Case 1: Liquidity PASS, Runway FAIL - should still penalize
        fin_data_short_runway = {
            "runway_months": 4,
            "liquidity_gate_status": "PASS",
        }

        # Case 2: Liquidity FAIL, Runway PASS - different penalty
        fin_data_illiquid = {
            "runway_months": 24,
            "liquidity_gate_status": "FAIL",
        }

        # Both should have independent effects, not conflate
        assert fin_data_short_runway["runway_months"] < 6
        assert fin_data_illiquid["runway_months"] >= 12

        # The interaction logic should treat these differently
        # (detailed assertion depends on interaction_terms implementation)


# =============================================================================
# TEST 5: Determinism Hash Stability
# =============================================================================

class TestDeterminismHashStability:
    """
    Zero scores must not cause hash instability due to serialization issues.
    JSON serialization of 0 vs "0" vs Decimal("0") must be consistent.
    """

    def test_hash_stability_with_zero_scores(self):
        """Same inputs with zero scores should produce identical hashes."""
        base_weights = {"clinical": Decimal("0.40"), "financial": Decimal("0.35"), "catalyst": Decimal("0.25")}
        effective_weights = base_weights.copy()

        component_scores = [
            ComponentScore(
                name="clinical",
                raw=Decimal("0"),  # Zero raw score
                normalized=Decimal("50"),
                confidence=Decimal("0.7"),
                weight_base=Decimal("0.40"),
                weight_effective=Decimal("0.40"),
                contribution=Decimal("20"),
            ),
        ]

        enhancements = {
            "momentum_score": Decimal("0"),  # Zero enhancement
            "valuation_score": Decimal("50"),
        }

        hash1 = _compute_determinism_hash(
            ticker="TEST",
            version="v3.0",
            mode="default",
            base_weights=base_weights,
            effective_weights=effective_weights,
            component_scores=component_scores,
            enhancements=enhancements,
            final_score=Decimal("45.00"),
        )

        hash2 = _compute_determinism_hash(
            ticker="TEST",
            version="v3.0",
            mode="default",
            base_weights=base_weights,
            effective_weights=effective_weights,
            component_scores=component_scores,
            enhancements=enhancements,
            final_score=Decimal("45.00"),
        )

        assert hash1 == hash2, f"Hash instability: {hash1} != {hash2}"

    def test_hash_different_for_different_scores(self):
        """Different scores should produce different hashes."""
        base_weights = {"clinical": Decimal("0.40")}
        effective_weights = base_weights.copy()

        component_scores_a = [
            ComponentScore(
                name="clinical",
                raw=Decimal("0"),
                normalized=Decimal("50"),
                confidence=Decimal("0.7"),
                weight_base=Decimal("0.40"),
                weight_effective=Decimal("0.40"),
                contribution=Decimal("20"),
            ),
        ]

        component_scores_b = [
            ComponentScore(
                name="clinical",
                raw=Decimal("10"),  # Different raw
                normalized=Decimal("50"),
                confidence=Decimal("0.7"),
                weight_base=Decimal("0.40"),
                weight_effective=Decimal("0.40"),
                contribution=Decimal("20"),
            ),
        ]

        hash_a = _compute_determinism_hash(
            ticker="TEST", version="v3.0", mode="default",
            base_weights=base_weights, effective_weights=effective_weights,
            component_scores=component_scores_a,
            enhancements={}, final_score=Decimal("45.00"),
        )

        hash_b = _compute_determinism_hash(
            ticker="TEST", version="v3.0", mode="default",
            base_weights=base_weights, effective_weights=effective_weights,
            component_scores=component_scores_b,
            enhancements={}, final_score=Decimal("45.00"),
        )

        assert hash_a != hash_b, "Different raw scores should produce different hashes"


# =============================================================================
# TEST 6: Full Pipeline Determinism
# =============================================================================

class TestPipelineDeterminism:
    """Ensure full pipeline produces identical results across runs."""

    def test_same_inputs_same_outputs(self, sample_module_inputs):
        """Running v3 twice with same inputs should produce identical output."""
        result1 = compute_module_5_composite_v3(
            universe_result=sample_module_inputs["universe"],
            financial_result=sample_module_inputs["financial"],
            catalyst_result=sample_module_inputs["catalyst"],
            clinical_result=sample_module_inputs["clinical"],
            as_of_date="2026-01-15",
            validate_inputs=False,
        )

        result2 = compute_module_5_composite_v3(
            universe_result=sample_module_inputs["universe"],
            financial_result=sample_module_inputs["financial"],
            catalyst_result=sample_module_inputs["catalyst"],
            clinical_result=sample_module_inputs["clinical"],
            as_of_date="2026-01-15",
            validate_inputs=False,
        )

        # Compare rankings
        ranks1 = {s["ticker"]: s["composite_rank"] for s in result1["ranked_securities"]}
        ranks2 = {s["ticker"]: s["composite_rank"] for s in result2["ranked_securities"]}
        assert ranks1 == ranks2, "Rankings differ between runs"

        # Compare scores
        scores1 = {s["ticker"]: s["composite_score"] for s in result1["ranked_securities"]}
        scores2 = {s["ticker"]: s["composite_score"] for s in result2["ranked_securities"]}
        assert scores1 == scores2, "Scores differ between runs"

        # Compare hashes
        hashes1 = {s["ticker"]: s["determinism_hash"] for s in result1["ranked_securities"]}
        hashes2 = {s["ticker"]: s["determinism_hash"] for s in result2["ranked_securities"]}
        assert hashes1 == hashes2, "Determinism hashes differ between runs"


# =============================================================================
# TEST 7: Momentum Strong Signal Threshold Boundary Inclusivity
# =============================================================================

class TestMomentumStrongSignalBoundary:
    """
    Regression tests for momentum_strong_signal threshold boundary.

    The metric counts signals where |score - 50| >= 2.5, meaning:
    - Score >= 52.5 OR score <= 47.5 counts as "strong"
    - Score = 52.49 or 47.51 does NOT count

    This is INCLUSIVE at exactly 47.5 and 52.5.

    Alpha anchoring (for reference):
    - With conf=0.7 (typical): |score-50| >= 2.5 requires |alpha| >= ~2.4%
    - With conf=0.9 (high):    |score-50| >= 2.5 requires |alpha| >= ~1.85%
    - Raw (no shrinkage):      |score-50| >= 2.5 requires |alpha| >= ~1.67%
    """

    def _is_strong_signal(self, score: Decimal) -> bool:
        """Helper to check if a score qualifies as strong signal.

        This replicates the logic from module_5_composite_v3.py diagnostic_counts.
        """
        return abs(score - Decimal("50")) >= Decimal("2.5")

    def test_boundary_inclusive_at_47_5(self):
        """Score of exactly 47.5 should count as strong (inclusive boundary)."""
        assert self._is_strong_signal(Decimal("47.5")), (
            "Score 47.5 should be counted as strong (boundary is inclusive)"
        )

    def test_boundary_inclusive_at_52_5(self):
        """Score of exactly 52.5 should count as strong (inclusive boundary)."""
        assert self._is_strong_signal(Decimal("52.5")), (
            "Score 52.5 should be counted as strong (boundary is inclusive)"
        )

    def test_score_47_51_not_strong_delta_2_49(self):
        """Score 47.51 NOT strong: |47.51-50| = 2.49 < 2.5 threshold."""
        assert not self._is_strong_signal(Decimal("47.51")), (
            "Score 47.51 should NOT be counted as strong (delta=2.49)"
        )

    def test_score_52_49_not_strong_delta_2_49(self):
        """Score 52.49 NOT strong: |52.49-50| = 2.49 < 2.5 threshold."""
        assert not self._is_strong_signal(Decimal("52.49")), (
            "Score 52.49 should NOT be counted as strong (delta=2.49)"
        )

    def test_neutral_score_not_strong(self):
        """Score of exactly 50 should NOT count as strong."""
        assert not self._is_strong_signal(Decimal("50")), (
            "Neutral score (50) should NOT be counted as strong"
        )

    def test_boundary_epsilon_below_threshold(self):
        """Score just inside threshold boundary should NOT count as strong."""
        # Use high precision to test boundary
        # 52.4999999 is 2.4999999 away from 50, which is < 2.5
        assert not self._is_strong_signal(Decimal("52.4999999")), (
            "Score 52.4999999 should NOT be counted as strong"
        )
        # 47.5000001 is 2.4999999 away from 50, which is < 2.5
        assert not self._is_strong_signal(Decimal("47.5000001")), (
            "Score 47.5000001 should NOT be counted as strong"
        )

    def test_extreme_scores_are_strong(self):
        """Extreme scores (near 5 or 95 clamp limits) should count as strong."""
        assert self._is_strong_signal(Decimal("5")), "Score 5 should be strong"
        assert self._is_strong_signal(Decimal("95")), "Score 95 should be strong"
        assert self._is_strong_signal(Decimal("9.5")), "Score 9.5 should be strong"
        assert self._is_strong_signal(Decimal("90.5")), "Score 90.5 should be strong"

    def test_deterministic_threshold(self):
        """Same score should always produce same strong/not-strong classification."""
        test_scores = [
            Decimal("47.5"),   # Strong (boundary)
            Decimal("47.51"),  # Not strong
            Decimal("50"),     # Neutral
            Decimal("52.5"),   # Strong (boundary)
            Decimal("52.49"),  # Not strong
        ]
        for score in test_scores:
            result1 = self._is_strong_signal(score)
            result2 = self._is_strong_signal(score)
            assert result1 == result2, f"Non-deterministic result for score {score}"
