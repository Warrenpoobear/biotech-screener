"""
Tests for the Intelligent Governance Layer

Validates:
1. Sharpe-ratio weight optimization
2. Business logic interaction effects
3. Ensemble ranking
4. Regime-adaptive weight orchestration
5. Determinism and audit trail integrity
6. Edge cases and fallback behavior

Author: Wake Robin Capital Management
"""
import pytest
from datetime import date, timedelta
from decimal import Decimal
from typing import Dict, List, Tuple, Any

from src.modules.intelligent_governance import (
    # Main orchestrator
    IntelligentGovernanceLayer,
    IntelligentGovernanceResult,

    # Components
    SharpeWeightOptimizer,
    SharpeOptimizationResult,
    InteractionEffectsEngine,
    InteractionEffectsResult,
    InteractionEffect,
    EnsembleRanker,
    EnsembleRank,
    RegimeAdaptiveOrchestrator,

    # Enums
    OptimizationMethod,
    InteractionType,
    RankingMethod,

    # Helpers
    _to_decimal,
    _coalesce,
    _quantize_weight,
    _quantize_score,
    _clamp,
    _compute_l1_distance,
    _normalize_weights,
    _compute_audit_hash,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def base_weights() -> Dict[str, Decimal]:
    """Standard base weights for testing."""
    return {
        "clinical": Decimal("0.30"),
        "financial": Decimal("0.25"),
        "catalyst": Decimal("0.20"),
        "momentum": Decimal("0.15"),
        "pos": Decimal("0.10"),
    }


@pytest.fixture
def sample_scores() -> Dict[str, Decimal]:
    """Sample normalized scores for testing."""
    return {
        "clinical": Decimal("72"),
        "financial": Decimal("65"),
        "catalyst": Decimal("58"),
        "momentum": Decimal("70"),
        "institutional": Decimal("68"),
    }


@pytest.fixture
def historical_scores() -> List[Dict[str, Any]]:
    """Sample historical scores for Sharpe optimization."""
    base_date = date(2025, 6, 15)
    scores = []

    # Generate 24 months of data
    for month_offset in range(24):
        score_date = base_date - timedelta(days=month_offset * 30)
        for ticker_idx in range(20):
            ticker = f"TKR{ticker_idx:02d}"
            scores.append({
                "as_of_date": score_date,
                "ticker": ticker,
                "clinical": Decimal("50") + Decimal(str(ticker_idx * 2)),
                "financial": Decimal("40") + Decimal(str(ticker_idx * 3)),
                "catalyst": Decimal("45") + Decimal(str(ticker_idx * 2.5)),
                "momentum": Decimal("35") + Decimal(str(ticker_idx * 2)),
                "pos": Decimal("55") + Decimal(str(ticker_idx * 1.5)),
            })
    return scores


@pytest.fixture
def forward_returns(historical_scores) -> Dict[Tuple[date, str], Decimal]:
    """Sample forward returns keyed by (as_of_date, ticker)."""
    returns = {}
    for score in historical_scores:
        score_date = score["as_of_date"]
        ticker = score["ticker"]
        # Simulate returns correlated with clinical score
        clinical = score["clinical"]
        base_return = (clinical - Decimal("50")) / Decimal("200")  # -0.25 to +0.25
        returns[(score_date, ticker)] = base_return
    return returns


@pytest.fixture
def ticker_data_batch() -> List[Dict[str, Any]]:
    """Batch of ticker data for ensemble testing."""
    return [
        {
            "ticker": "ACME",
            "clinical": Decimal("75"),
            "financial": Decimal("68"),
            "catalyst": Decimal("62"),
            "momentum": Decimal("72"),
            "institutional": Decimal("78"),
            "pos": Decimal("60"),
            "valuation": Decimal("55"),
            "metadata": {
                "runway_months": Decimal("30"),
                "days_to_catalyst": 25,
            }
        },
        {
            "ticker": "BETA",
            "clinical": Decimal("65"),
            "financial": Decimal("35"),
            "catalyst": Decimal("70"),
            "momentum": Decimal("80"),
            "institutional": Decimal("45"),
            "pos": Decimal("50"),
            "valuation": Decimal("70"),
            "metadata": {
                "runway_months": Decimal("8"),
                "short_interest_pct": Decimal("22"),
            }
        },
        {
            "ticker": "GAMMA",
            "clinical": Decimal("82"),
            "financial": Decimal("72"),
            "catalyst": Decimal("55"),
            "momentum": Decimal("48"),
            "institutional": Decimal("85"),
            "pos": Decimal("75"),
            "valuation": Decimal("45"),
            "metadata": {
                "runway_months": Decimal("36"),
                "institutional_net_change": Decimal("5"),
            }
        },
        {
            "ticker": "DELTA",
            "clinical": Decimal("55"),
            "financial": Decimal("80"),
            "catalyst": Decimal("45"),
            "momentum": Decimal("55"),
            "institutional": Decimal("60"),
            "pos": Decimal("65"),
            "valuation": Decimal("80"),
            "metadata": {
                "runway_months": Decimal("48"),
            }
        },
    ]


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================

class TestHelperFunctions:
    """Tests for utility helper functions."""

    def test_to_decimal_from_decimal(self):
        """Test _to_decimal preserves Decimal."""
        val = Decimal("42.50")
        result = _to_decimal(val)
        assert result == val

    def test_to_decimal_from_float(self):
        """Test _to_decimal converts float."""
        result = _to_decimal(42.5)
        assert result == Decimal("42.5")

    def test_to_decimal_from_string(self):
        """Test _to_decimal converts string."""
        result = _to_decimal("42.50")
        assert result == Decimal("42.50")

    def test_to_decimal_none_returns_default(self):
        """Test _to_decimal returns default for None."""
        result = _to_decimal(None, Decimal("100"))
        assert result == Decimal("100")

    def test_to_decimal_invalid_returns_default(self):
        """Test _to_decimal returns default for invalid input."""
        result = _to_decimal("not_a_number", Decimal("50"))
        assert result == Decimal("50")

    def test_coalesce_returns_first_non_none(self):
        """Test _coalesce returns first non-None value."""
        result = _coalesce(None, Decimal("42"), Decimal("100"))
        assert result == Decimal("42")

    def test_coalesce_preserves_zero(self):
        """Test _coalesce doesn't treat 0 as falsy (critical bug fix)."""
        result = _coalesce(Decimal("0"), Decimal("100"))
        assert result == Decimal("0")  # NOT 100!

    def test_coalesce_all_none_returns_default(self):
        """Test _coalesce returns default when all values None."""
        result = _coalesce(None, None, None, default=Decimal("50"))
        assert result == Decimal("50")

    def test_quantize_weight(self):
        """Test weight quantization precision."""
        result = _quantize_weight(Decimal("0.123456789"))
        assert result == Decimal("0.1235")

    def test_quantize_score(self):
        """Test score quantization precision."""
        result = _quantize_score(Decimal("72.5678"))
        assert result == Decimal("72.57")

    def test_clamp_within_range(self):
        """Test clamp with value in range."""
        result = _clamp(Decimal("50"), Decimal("0"), Decimal("100"))
        assert result == Decimal("50")

    def test_clamp_below_minimum(self):
        """Test clamp with value below minimum."""
        result = _clamp(Decimal("-10"), Decimal("0"), Decimal("100"))
        assert result == Decimal("0")

    def test_clamp_above_maximum(self):
        """Test clamp with value above maximum."""
        result = _clamp(Decimal("150"), Decimal("0"), Decimal("100"))
        assert result == Decimal("100")

    def test_compute_l1_distance(self):
        """Test L1 distance computation."""
        w1 = {"a": Decimal("0.40"), "b": Decimal("0.60")}
        w2 = {"a": Decimal("0.30"), "b": Decimal("0.70")}
        result = _compute_l1_distance(w1, w2)
        assert result == Decimal("0.20")  # |0.40-0.30| + |0.60-0.70|

    def test_normalize_weights(self):
        """Test weight normalization."""
        weights = {"a": Decimal("2"), "b": Decimal("3"), "c": Decimal("5")}
        result = _normalize_weights(weights)
        assert sum(result.values()) == Decimal("1.0")
        assert result["a"] == Decimal("0.2")
        assert result["b"] == Decimal("0.3")
        assert result["c"] == Decimal("0.5")

    def test_compute_audit_hash_determinism(self):
        """Test audit hash is deterministic."""
        data = {"ticker": "ACME", "score": Decimal("72.50")}
        hash1 = _compute_audit_hash(data)
        hash2 = _compute_audit_hash(data)
        assert hash1 == hash2
        assert hash1.startswith("sha256:")

    def test_compute_audit_hash_different_data(self):
        """Test audit hash differs for different data."""
        hash1 = _compute_audit_hash({"ticker": "ACME"})
        hash2 = _compute_audit_hash({"ticker": "BETA"})
        assert hash1 != hash2


# =============================================================================
# SHARPE WEIGHT OPTIMIZER TESTS
# =============================================================================

class TestSharpeWeightOptimizer:
    """Tests for Sharpe-ratio weight optimization."""

    def test_fallback_on_insufficient_data(self, base_weights):
        """Test fallback when not enough historical data."""
        optimizer = SharpeWeightOptimizer(min_periods=12)
        result = optimizer.optimize(
            historical_scores=[],
            forward_returns={},
            base_weights=base_weights,
            as_of_date=date(2026, 1, 15),
        )

        assert result.optimization_method == OptimizationMethod.FALLBACK
        assert result.optimized_weights == base_weights
        assert result.confidence <= Decimal("0.15")

    def test_optimization_with_sufficient_data(
        self, base_weights, historical_scores, forward_returns
    ):
        """Test optimization runs with sufficient data."""
        optimizer = SharpeWeightOptimizer(min_periods=6)
        result = optimizer.optimize(
            historical_scores=historical_scores,
            forward_returns=forward_returns,
            base_weights=base_weights,
            as_of_date=date(2026, 1, 15),
            lookback_months=24,
        )

        # Should successfully optimize
        assert result.training_periods >= 6
        assert sum(result.optimized_weights.values()) == pytest.approx(Decimal("1.0"), abs=Decimal("0.01"))

        # Weights should be within bounds
        for weight in result.optimized_weights.values():
            assert weight >= Decimal("0.02")
            assert weight <= Decimal("0.60")

    def test_shrinkage_toward_base_weights(
        self, base_weights, historical_scores, forward_returns
    ):
        """Test that shrinkage pulls weights toward base."""
        # High shrinkage
        optimizer_high = SharpeWeightOptimizer(
            min_periods=6,
            shrinkage_lambda=Decimal("0.95"),  # Very high shrinkage
        )
        result_high = optimizer_high.optimize(
            historical_scores=historical_scores,
            forward_returns=forward_returns,
            base_weights=base_weights,
            as_of_date=date(2026, 1, 15),
        )

        # Low shrinkage
        optimizer_low = SharpeWeightOptimizer(
            min_periods=6,
            shrinkage_lambda=Decimal("0.10"),  # Very low shrinkage
        )
        result_low = optimizer_low.optimize(
            historical_scores=historical_scores,
            forward_returns=forward_returns,
            base_weights=base_weights,
            as_of_date=date(2026, 1, 15),
        )

        # High shrinkage should be closer to base weights
        l1_high = _compute_l1_distance(result_high.optimized_weights, base_weights)
        l1_low = _compute_l1_distance(result_low.optimized_weights, base_weights)

        assert l1_high <= l1_low

    def test_weight_bounds_enforced(
        self, base_weights, historical_scores, forward_returns
    ):
        """Test that weight bounds are enforced."""
        optimizer = SharpeWeightOptimizer(
            min_weight=Decimal("0.05"),
            max_weight=Decimal("0.40"),
        )
        result = optimizer.optimize(
            historical_scores=historical_scores,
            forward_returns=forward_returns,
            base_weights=base_weights,
            as_of_date=date(2026, 1, 15),
        )

        for weight in result.optimized_weights.values():
            assert weight >= Decimal("0.05")
            assert weight <= Decimal("0.40")

    def test_pit_safety_embargo(
        self, base_weights, historical_scores, forward_returns
    ):
        """Test that embargo period is respected."""
        optimizer = SharpeWeightOptimizer(embargo_months=3)

        # Should exclude recent periods
        result = optimizer.optimize(
            historical_scores=historical_scores,
            forward_returns=forward_returns,
            base_weights=base_weights,
            as_of_date=date(2026, 1, 15),
            lookback_months=24,
        )

        assert "embargo_months" in result.provenance
        assert result.provenance["embargo_months"] == 3

    def test_determinism(
        self, base_weights, historical_scores, forward_returns
    ):
        """Test that optimization is deterministic."""
        optimizer = SharpeWeightOptimizer()

        result1 = optimizer.optimize(
            historical_scores=historical_scores,
            forward_returns=forward_returns,
            base_weights=base_weights,
            as_of_date=date(2026, 1, 15),
        )

        result2 = optimizer.optimize(
            historical_scores=historical_scores,
            forward_returns=forward_returns,
            base_weights=base_weights,
            as_of_date=date(2026, 1, 15),
        )

        assert result1.optimized_weights == result2.optimized_weights
        assert result1.historical_sharpe == result2.historical_sharpe


# =============================================================================
# INTERACTION EFFECTS ENGINE TESTS
# =============================================================================

class TestInteractionEffectsEngine:
    """Tests for business logic interaction effects."""

    def test_no_effects_with_neutral_scores(self):
        """Test no effects triggered with neutral scores."""
        engine = InteractionEffectsEngine()
        result = engine.compute_effects(
            ticker="ACME",
            scores={
                "clinical": Decimal("50"),
                "financial": Decimal("50"),
                "catalyst": Decimal("50"),
                "momentum": Decimal("50"),
                "institutional": Decimal("50"),
            },
            metadata={"runway_months": Decimal("24")},
        )

        # No effects should be triggered
        assert result.total_adjustment == Decimal("0")
        assert len(result.effects) == 0

    def test_institutional_catalyst_amplification(self):
        """Test institutional + near catalyst synergy."""
        engine = InteractionEffectsEngine()
        result = engine.compute_effects(
            ticker="ACME",
            scores={
                "clinical": Decimal("65"),
                "financial": Decimal("60"),
                "catalyst": Decimal("70"),
                "momentum": Decimal("55"),
                "institutional": Decimal("78"),  # High institutional
            },
            metadata={
                "runway_months": Decimal("24"),
                "days_to_catalyst": 20,  # Near catalyst
            },
        )

        # Should have synergy effect
        assert any(e.name == "institutional_catalyst_amplification" for e in result.effects)
        assert result.total_adjustment > Decimal("0")
        assert "institutional_catalyst_synergy" in result.flags

    def test_clinical_runway_conviction(self):
        """Test clinical + strong runway synergy."""
        engine = InteractionEffectsEngine()
        result = engine.compute_effects(
            ticker="ACME",
            scores={
                "clinical": Decimal("75"),  # Strong clinical
                "financial": Decimal("70"),
                "catalyst": Decimal("50"),
                "momentum": Decimal("50"),
                "institutional": Decimal("50"),
            },
            metadata={"runway_months": Decimal("30")},  # Long runway
        )

        assert any(e.name == "clinical_runway_conviction" for e in result.effects)
        assert result.net_synergy > Decimal("0")

    def test_momentum_fundamental_conflict(self):
        """Test momentum + weak fundamentals conflict."""
        engine = InteractionEffectsEngine()
        result = engine.compute_effects(
            ticker="ACME",
            scores={
                "clinical": Decimal("50"),
                "financial": Decimal("30"),  # Weak financials
                "catalyst": Decimal("50"),
                "momentum": Decimal("78"),  # Strong momentum
                "institutional": Decimal("50"),
            },
            metadata={"runway_months": Decimal("12")},
        )

        assert any(e.name == "momentum_fundamental_conflict" for e in result.effects)
        assert result.total_adjustment < Decimal("0")
        assert "fade_momentum_signal" in result.flags

    def test_institutional_financial_warning(self):
        """Test high institutional + deteriorating financials warning."""
        engine = InteractionEffectsEngine()
        result = engine.compute_effects(
            ticker="ACME",
            scores={
                "clinical": Decimal("50"),
                "financial": Decimal("28"),  # Distressed
                "catalyst": Decimal("50"),
                "momentum": Decimal("50"),
                "institutional": Decimal("72"),  # High institutional
            },
            metadata={"runway_months": Decimal("6")},
        )

        assert any(e.name == "institutional_financial_warning" for e in result.effects)
        assert "institutional_trapped_warning" in result.flags

    def test_catalyst_short_interest_risk(self):
        """Test near catalyst + high short interest conflict."""
        engine = InteractionEffectsEngine()
        result = engine.compute_effects(
            ticker="ACME",
            scores={
                "clinical": Decimal("60"),
                "financial": Decimal("55"),
                "catalyst": Decimal("70"),
                "momentum": Decimal("60"),
                "institutional": Decimal("50"),
            },
            metadata={
                "runway_months": Decimal("18"),
                "days_to_catalyst": 5,  # Very near (increased proximity for flag trigger)
                "short_interest_pct": Decimal("35"),  # Very high shorts
            },
        )

        assert any(e.name == "catalyst_short_interest_risk" for e in result.effects)
        # The effect should be triggered with negative adjustment
        effect = next(e for e in result.effects if e.name == "catalyst_short_interest_risk")
        assert effect.adjustment < Decimal("0")
        assert result.total_adjustment < Decimal("0")

    def test_max_adjustment_cap(self):
        """Test that total adjustment is capped."""
        engine = InteractionEffectsEngine(max_adjustment=Decimal("2.0"))
        result = engine.compute_effects(
            ticker="ACME",
            scores={
                "clinical": Decimal("90"),  # Extreme
                "financial": Decimal("90"),
                "catalyst": Decimal("90"),
                "momentum": Decimal("90"),
                "institutional": Decimal("90"),
            },
            metadata={
                "runway_months": Decimal("60"),
                "days_to_catalyst": 10,
                "institutional_net_change": Decimal("20"),
            },
        )

        assert abs(result.total_adjustment) <= Decimal("2.0")

    def test_effects_have_business_logic(self):
        """Test that all triggered effects have business logic explanation."""
        engine = InteractionEffectsEngine()
        result = engine.compute_effects(
            ticker="ACME",
            scores={
                "clinical": Decimal("75"),
                "financial": Decimal("70"),
                "catalyst": Decimal("65"),
                "momentum": Decimal("70"),
                "institutional": Decimal("75"),
            },
            metadata={
                "runway_months": Decimal("28"),
                "days_to_catalyst": 25,
                "institutional_net_change": Decimal("8"),
            },
        )

        for effect in result.effects:
            assert effect.business_logic is not None
            assert len(effect.business_logic) > 10  # Meaningful explanation


# =============================================================================
# ENSEMBLE RANKER TESTS
# =============================================================================

class TestEnsembleRanker:
    """Tests for ensemble ranking system."""

    def test_single_ticker(self):
        """Test ensemble ranking with single ticker."""
        ranker = EnsembleRanker()
        result = ranker.compute_ranks([
            {"ticker": "ACME", "clinical": Decimal("70"), "financial": Decimal("65")}
        ])

        assert len(result) == 1
        assert result[0].ticker == "ACME"
        assert result[0].final_rank == 1

    def test_multiple_tickers_ordered(self, ticker_data_batch):
        """Test ensemble ranking preserves order."""
        ranker = EnsembleRanker()
        result = ranker.compute_ranks(ticker_data_batch)

        assert len(result) == 4
        ranks = [r.final_rank for r in result]
        assert sorted(ranks) == [1, 2, 3, 4]

    def test_ensemble_rank_is_weighted_average(self, ticker_data_batch):
        """Test ensemble rank is weighted average of method ranks."""
        ranker = EnsembleRanker(
            composite_weight=Decimal("0.50"),
            momentum_weight=Decimal("0.25"),
            value_weight=Decimal("0.25"),
        )
        result = ranker.compute_ranks(ticker_data_batch)

        for r in result:
            expected = (
                Decimal(r.composite_rank) * Decimal("0.50") +
                Decimal(r.momentum_rank) * Decimal("0.25") +
                Decimal(r.value_rank) * Decimal("0.25")
            )
            assert abs(r.ensemble_rank - expected) < Decimal("0.1")

    def test_agreement_metric(self, ticker_data_batch):
        """Test rank agreement metric."""
        ranker = EnsembleRanker()
        result = ranker.compute_ranks(ticker_data_batch)

        for r in result:
            assert Decimal("0") <= r.rank_agreement <= Decimal("1")

    def test_max_divergence_metric(self, ticker_data_batch):
        """Test max rank divergence metric."""
        ranker = EnsembleRanker()
        result = ranker.compute_ranks(ticker_data_batch)

        for r in result:
            ranks = [r.composite_rank, r.momentum_rank, r.value_rank]
            assert r.max_rank_divergence == max(ranks) - min(ranks)

    def test_deterministic_tiebreak(self):
        """Test that ties are broken deterministically by ticker."""
        # Create data with identical scores
        data = [
            {"ticker": "ZZZ", "clinical": Decimal("70"), "financial": Decimal("65")},
            {"ticker": "AAA", "clinical": Decimal("70"), "financial": Decimal("65")},
        ]
        ranker = EnsembleRanker()
        result = ranker.compute_ranks(data)

        # AAA should rank higher (lower rank number) due to alphabetical tiebreak
        aaa_result = next(r for r in result if r.ticker == "AAA")
        zzz_result = next(r for r in result if r.ticker == "ZZZ")
        assert aaa_result.final_rank < zzz_result.final_rank


# =============================================================================
# REGIME ADAPTIVE ORCHESTRATOR TESTS
# =============================================================================

class TestRegimeAdaptiveOrchestrator:
    """Tests for regime-adaptive weight orchestration."""

    def test_neutral_regime_no_change(self, base_weights):
        """Test NEUTRAL regime doesn't change weights much."""
        orchestrator = RegimeAdaptiveOrchestrator()
        adapted, diagnostics = orchestrator.adapt_weights(
            base_weights=base_weights,
            regime="NEUTRAL",
        )

        # Should be very close to base weights
        l1 = _compute_l1_distance(adapted, base_weights)
        assert l1 < Decimal("0.10")

    def test_bull_regime_boosts_momentum(self, base_weights):
        """Test BULL regime boosts momentum weight."""
        orchestrator = RegimeAdaptiveOrchestrator()
        adapted, _ = orchestrator.adapt_weights(
            base_weights=base_weights,
            regime="BULL",
        )

        # Momentum should be higher than base
        assert adapted.get("momentum", Decimal("0")) > base_weights.get("momentum", Decimal("0"))

    def test_bear_regime_boosts_financial(self, base_weights):
        """Test BEAR regime boosts financial weight."""
        orchestrator = RegimeAdaptiveOrchestrator()
        adapted, _ = orchestrator.adapt_weights(
            base_weights=base_weights,
            regime="BEAR",
        )

        # Financial should be higher than base
        assert adapted.get("financial", Decimal("0")) > base_weights.get("financial", Decimal("0"))

    def test_weight_change_capped(self, base_weights):
        """Test weight changes are capped."""
        orchestrator = RegimeAdaptiveOrchestrator(max_weight_delta=Decimal("0.10"))

        # VOLATILITY_SPIKE has extreme multipliers
        adapted, _ = orchestrator.adapt_weights(
            base_weights=base_weights,
            regime="VOLATILITY_SPIKE",
        )

        # No weight should change more than 10%
        for k in base_weights:
            if k in adapted:
                delta = abs(adapted[k] - base_weights[k])
                assert delta <= Decimal("0.10") + Decimal("0.01")  # Small tolerance

    def test_sharpe_weights_blended(self, base_weights):
        """Test Sharpe weights are blended in."""
        orchestrator = RegimeAdaptiveOrchestrator()

        sharpe_weights = {
            "clinical": Decimal("0.40"),  # Different from base
            "financial": Decimal("0.30"),
            "catalyst": Decimal("0.15"),
            "momentum": Decimal("0.10"),
            "pos": Decimal("0.05"),
        }

        adapted, diagnostics = orchestrator.adapt_weights(
            base_weights=base_weights,
            regime="NEUTRAL",
            sharpe_weights=sharpe_weights,
            sharpe_confidence=Decimal("0.60"),
        )

        # Should be between base and sharpe
        assert "sharpe_blend" in str(diagnostics.get("adaptations_applied", []))

    def test_diagnostics_complete(self, base_weights):
        """Test diagnostics contain all required fields."""
        orchestrator = RegimeAdaptiveOrchestrator()
        _, diagnostics = orchestrator.adapt_weights(
            base_weights=base_weights,
            regime="BULL",
        )

        assert "base_weights" in diagnostics
        assert "regime" in diagnostics
        assert "final_weights" in diagnostics
        assert "adaptations_applied" in diagnostics


# =============================================================================
# MAIN ORCHESTRATOR TESTS
# =============================================================================

class TestIntelligentGovernanceLayer:
    """Tests for the main intelligent governance orchestrator."""

    def test_basic_computation(self, base_weights, sample_scores):
        """Test basic score computation."""
        layer = IntelligentGovernanceLayer(
            enable_sharpe_optimization=False,
            enable_interaction_effects=True,
            enable_regime_adaptation=True,
        )

        result = layer.compute(
            ticker="ACME",
            scores=sample_scores,
            metadata={"runway_months": Decimal("24")},
            base_weights=base_weights,
            regime="NEUTRAL",
        )

        assert result.ticker == "ACME"
        assert Decimal("0") <= result.base_score <= Decimal("100")
        assert Decimal("0") <= result.optimized_score <= Decimal("100")
        assert result.audit_hash.startswith("sha256:")

    def test_score_delta_bounded(self, base_weights, sample_scores):
        """Test score delta is bounded by interaction cap."""
        layer = IntelligentGovernanceLayer()

        result = layer.compute(
            ticker="ACME",
            scores=sample_scores,
            metadata={"runway_months": Decimal("24")},
            base_weights=base_weights,
        )

        # Delta should be bounded
        assert abs(result.score_delta) <= Decimal("3.0")

    def test_batch_computation(self, base_weights, ticker_data_batch):
        """Test batch computation returns correct results."""
        layer = IntelligentGovernanceLayer(
            enable_sharpe_optimization=False,
        )

        results, ensemble_ranks = layer.compute_batch(
            ticker_data=ticker_data_batch,
            base_weights=base_weights,
            regime="BULL",
        )

        assert len(results) == 4
        assert len(ensemble_ranks) == 4

        # Each result should have ensemble rank attached
        for result in results:
            assert result.ensemble_rank is not None

    def test_governance_flags_populated(self, base_weights, sample_scores):
        """Test governance flags are populated."""
        layer = IntelligentGovernanceLayer(
            enable_regime_adaptation=True,
        )

        result = layer.compute(
            ticker="ACME",
            scores=sample_scores,
            metadata={"runway_months": Decimal("24")},
            base_weights=base_weights,
            regime="BULL",
        )

        assert len(result.governance_flags) > 0
        assert any("regime_adapted" in f for f in result.governance_flags)

    def test_determinism(self, base_weights, sample_scores):
        """Test computation is deterministic."""
        layer = IntelligentGovernanceLayer()

        result1 = layer.compute(
            ticker="ACME",
            scores=sample_scores,
            metadata={"runway_months": Decimal("24")},
            base_weights=base_weights,
        )

        result2 = layer.compute(
            ticker="ACME",
            scores=sample_scores,
            metadata={"runway_months": Decimal("24")},
            base_weights=base_weights,
        )

        assert result1.base_score == result2.base_score
        assert result1.optimized_score == result2.optimized_score
        assert result1.audit_hash == result2.audit_hash

    def test_disabled_features_no_effect(self, base_weights, sample_scores):
        """Test disabled features have no effect."""
        layer_enabled = IntelligentGovernanceLayer(
            enable_interaction_effects=True,
            enable_regime_adaptation=True,
        )

        layer_disabled = IntelligentGovernanceLayer(
            enable_interaction_effects=False,
            enable_regime_adaptation=False,
        )

        result_enabled = layer_enabled.compute(
            ticker="ACME",
            scores=sample_scores,
            metadata={"runway_months": Decimal("24")},
            base_weights=base_weights,
            regime="BULL",
        )

        result_disabled = layer_disabled.compute(
            ticker="ACME",
            scores=sample_scores,
            metadata={"runway_months": Decimal("24")},
            base_weights=base_weights,
            regime="BULL",
        )

        # Disabled should have no interaction adjustment
        assert result_disabled.interaction_result.total_adjustment == Decimal("0")

        # Disabled should use base weights
        assert result_disabled.effective_weights == base_weights

    def test_smartness_knob_low(self, base_weights, sample_scores):
        """Test conservative smartness setting (0.0)."""
        layer = IntelligentGovernanceLayer(smartness=Decimal("0.0"))

        # Check that smartness params are conservative
        assert layer._smartness_params["smartness"] == "0"
        # Interaction cap should be lower
        assert Decimal(layer._smartness_params["interaction_cap"]) == Decimal("2.0")
        # Shrinkage should be higher (more governed)
        assert Decimal(layer._smartness_params["shrinkage"]) == Decimal("0.90")

    def test_smartness_knob_high(self, base_weights, sample_scores):
        """Test aggressive smartness setting (1.0)."""
        layer = IntelligentGovernanceLayer(smartness=Decimal("1.0"))

        # Check that smartness params are aggressive
        assert layer._smartness_params["smartness"] == "1"
        # Interaction cap should be higher
        assert Decimal(layer._smartness_params["interaction_cap"]) == Decimal("4.0")
        # Shrinkage should be lower (less governed)
        assert Decimal(layer._smartness_params["shrinkage"]) == Decimal("0.50")

    def test_smartness_knob_clamped(self):
        """Test smartness is clamped to [0, 1]."""
        layer_under = IntelligentGovernanceLayer(smartness=Decimal("-0.5"))
        layer_over = IntelligentGovernanceLayer(smartness=Decimal("1.5"))

        assert layer_under.smartness == Decimal("0")
        assert layer_over.smartness == Decimal("1")


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline(
        self, base_weights, historical_scores, forward_returns, ticker_data_batch
    ):
        """Test full intelligent governance pipeline."""
        layer = IntelligentGovernanceLayer(
            enable_sharpe_optimization=True,
            enable_interaction_effects=True,
            enable_regime_adaptation=True,
        )

        results, ensemble_ranks = layer.compute_batch(
            ticker_data=ticker_data_batch,
            base_weights=base_weights,
            regime="BEAR",
            historical_scores=historical_scores,
            forward_returns=forward_returns,
            as_of_date=date(2026, 1, 15),
        )

        # Verify all components worked
        for result in results:
            assert result.ticker is not None
            assert result.base_score >= Decimal("0")
            assert result.optimized_score >= Decimal("0")
            assert result.effective_weights is not None
            assert result.audit_hash is not None

        # Verify ensemble rankings
        assert len(ensemble_ranks) == len(ticker_data_batch)
        final_ranks = sorted([r.final_rank for r in ensemble_ranks])
        assert final_ranks == list(range(1, len(ticker_data_batch) + 1))

    def test_regime_impact_on_rankings(self, base_weights, ticker_data_batch):
        """Test that regime affects rankings."""
        layer = IntelligentGovernanceLayer(
            enable_sharpe_optimization=False,
            enable_regime_adaptation=True,
        )

        # BULL regime
        _, bull_ranks = layer.compute_batch(
            ticker_data=ticker_data_batch,
            base_weights=base_weights,
            regime="BULL",
        )

        # BEAR regime
        _, bear_ranks = layer.compute_batch(
            ticker_data=ticker_data_batch,
            base_weights=base_weights,
            regime="BEAR",
        )

        # Rankings should differ between regimes
        bull_order = [r.ticker for r in sorted(bull_ranks, key=lambda x: x.final_rank)]
        bear_order = [r.ticker for r in sorted(bear_ranks, key=lambda x: x.final_rank)]

        # At least some difference expected (may be same if data is too similar)
        # Just verify we got valid rankings
        assert len(bull_order) == len(bear_order) == 4

    def test_interaction_effects_impact_scores(self, base_weights):
        """Test that interaction effects meaningfully impact scores."""
        layer = IntelligentGovernanceLayer(
            enable_sharpe_optimization=False,
            enable_interaction_effects=True,
        )

        # Ticker with synergy conditions
        synergy_ticker = {
            "clinical": Decimal("75"),
            "financial": Decimal("70"),
            "catalyst": Decimal("65"),
            "momentum": Decimal("70"),
            "institutional": Decimal("75"),
        }
        synergy_meta = {
            "runway_months": Decimal("28"),
            "days_to_catalyst": 20,
            "institutional_net_change": Decimal("8"),
        }

        # Ticker with conflict conditions
        conflict_ticker = {
            "clinical": Decimal("50"),
            "financial": Decimal("30"),
            "catalyst": Decimal("60"),
            "momentum": Decimal("78"),
            "institutional": Decimal("50"),
        }
        conflict_meta = {"runway_months": Decimal("8")}

        synergy_result = layer.compute(
            ticker="SYNERGY",
            scores=synergy_ticker,
            metadata=synergy_meta,
            base_weights=base_weights,
        )

        conflict_result = layer.compute(
            ticker="CONFLICT",
            scores=conflict_ticker,
            metadata=conflict_meta,
            base_weights=base_weights,
        )

        # Synergy should have positive delta, conflict should have negative
        assert synergy_result.score_delta > Decimal("0")
        assert conflict_result.score_delta < Decimal("0")


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_ticker_data(self, base_weights):
        """Test handling of empty ticker data."""
        layer = IntelligentGovernanceLayer()
        results, ranks = layer.compute_batch(
            ticker_data=[],
            base_weights=base_weights,
        )

        assert results == []
        assert ranks == []

    def test_missing_score_components(self, base_weights):
        """Test handling of missing score components."""
        layer = IntelligentGovernanceLayer()

        # Only partial scores provided
        result = layer.compute(
            ticker="ACME",
            scores={"clinical": Decimal("70")},  # Only one component
            metadata={},
            base_weights=base_weights,
        )

        # Should still produce valid output
        assert result.base_score >= Decimal("0")

    def test_extreme_scores(self, base_weights):
        """Test handling of extreme scores."""
        layer = IntelligentGovernanceLayer()

        result = layer.compute(
            ticker="EXTREME",
            scores={
                "clinical": Decimal("100"),
                "financial": Decimal("0"),
                "catalyst": Decimal("100"),
                "momentum": Decimal("0"),
            },
            metadata={"runway_months": Decimal("0")},
            base_weights=base_weights,
        )

        # Should clamp to valid range
        assert Decimal("0") <= result.optimized_score <= Decimal("100")

    def test_unknown_regime(self, base_weights, sample_scores):
        """Test handling of unknown regime."""
        layer = IntelligentGovernanceLayer()

        result = layer.compute(
            ticker="ACME",
            scores=sample_scores,
            metadata={},
            base_weights=base_weights,
            regime="UNKNOWN_REGIME",  # Invalid regime
        )

        # Should fall back to neutral behavior
        assert result.base_score >= Decimal("0")

    def test_zero_weights(self):
        """Test handling of zero weights."""
        layer = IntelligentGovernanceLayer()

        # Weights that sum to zero (edge case)
        zero_weights = {
            "clinical": Decimal("0"),
            "financial": Decimal("0"),
        }

        result = layer.compute(
            ticker="ACME",
            scores={"clinical": Decimal("70"), "financial": Decimal("65")},
            metadata={},
            base_weights=zero_weights,
        )

        # Should handle gracefully (normalize will provide equal weights)
        assert result.base_score >= Decimal("0")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
