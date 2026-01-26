#!/usr/bin/env python3
"""
Tests for backtest/compare_module5_versions.py

Module 5 v2 vs v3 backtest comparison utilities.
Tests cover:
- generate_module_inputs (synthetic input generation)
- extract_scores (score extraction from results)
- compute_ic (Spearman IC calculation)
- compute_ic_tstat (IC t-statistic)
- bootstrap_ic_ci (bootstrap confidence intervals)
- compute_rank_correlation (rank correlation between versions)
- compute_top_bottom_spread (top-N vs bottom-N spread)
- compute_turnover (portfolio turnover)
- compute_drawdown (max drawdown)
- compute_costed_return (net returns after costs)
- compute_concentration_metrics (concentration analysis)
"""

import pytest
from decimal import Decimal
from statistics import mean

from backtest.compare_module5_versions import (
    # Data generation
    generate_module_inputs,
    DEFAULT_UNIVERSE,
    CLINICAL_DATA,
    MARKET_CAP_DATA,
    # Score utilities
    extract_scores,
    compute_ic,
    compute_ic_tstat,
    bootstrap_ic_ci,
    # Rank correlation
    compute_rank_correlation,
    # Portfolio metrics
    compute_top_bottom_spread,
    compute_turnover,
    compute_drawdown,
    compute_costed_return,
    compute_concentration_metrics,
)


class TestGenerateModuleInputs:
    """Tests for generate_module_inputs function."""

    def test_returns_dict(self):
        """Should return a dict with required keys."""
        result = generate_module_inputs(["AMGN", "GILD"], "2024-01-01")

        assert isinstance(result, dict)
        assert "universe" in result
        assert "financial" in result
        assert "catalyst" in result
        assert "clinical" in result

    def test_includes_tickers_in_universe(self):
        """Should include tickers in universe result."""
        tickers = ["AMGN", "GILD"]
        result = generate_module_inputs(tickers, "2024-01-01")

        universe = result["universe"]
        universe_tickers = [s["ticker"] for s in universe["active_securities"]]

        assert "AMGN" in universe_tickers
        assert "GILD" in universe_tickers

    def test_deterministic_with_same_seed(self):
        """Same seed should produce same results."""
        tickers = ["AMGN", "GILD"]

        result1 = generate_module_inputs(tickers, "2024-01-01", seed=42)
        result2 = generate_module_inputs(tickers, "2024-01-01", seed=42)

        # Financial scores should match
        scores1 = {s["ticker"]: s["financial_score"] for s in result1["financial"]["scores"]}
        scores2 = {s["ticker"]: s["financial_score"] for s in result2["financial"]["scores"]}

        assert scores1 == scores2

    def test_different_seeds_produce_different_results(self):
        """Different seeds should produce different results."""
        tickers = ["AMGN", "GILD"]

        result1 = generate_module_inputs(tickers, "2024-01-01", seed=1)
        result2 = generate_module_inputs(tickers, "2024-01-01", seed=2)

        scores1 = {s["ticker"]: s["financial_score"] for s in result1["financial"]["scores"]}
        scores2 = {s["ticker"]: s["financial_score"] for s in result2["financial"]["scores"]}

        assert scores1 != scores2


class TestExtractScores:
    """Tests for extract_scores function."""

    def test_extracts_scores(self):
        """Should extract ticker -> score mapping."""
        result = {
            "ranked_securities": [
                {"ticker": "A", "composite_score": "75.00"},
                {"ticker": "B", "composite_score": "80.00"},
            ]
        }

        scores = extract_scores(result)

        assert scores["A"] == Decimal("75.00")
        assert scores["B"] == Decimal("80.00")

    def test_handles_decimal_scores(self):
        """Should handle Decimal composite scores."""
        result = {
            "ranked_securities": [
                {"ticker": "A", "composite_score": Decimal("75.50")},
            ]
        }

        scores = extract_scores(result)
        assert scores["A"] == Decimal("75.50")

    def test_handles_empty_result(self):
        """Should handle empty result."""
        scores = extract_scores({})
        assert scores == {}

    def test_skips_missing_ticker(self):
        """Should skip entries without ticker."""
        result = {
            "ranked_securities": [
                {"ticker": "A", "composite_score": "75.00"},
                {"composite_score": "80.00"},  # Missing ticker
            ]
        }

        scores = extract_scores(result)
        assert len(scores) == 1

    def test_skips_none_score(self):
        """Should skip entries with None score."""
        result = {
            "ranked_securities": [
                {"ticker": "A", "composite_score": "75.00"},
                {"ticker": "B", "composite_score": None},
            ]
        }

        scores = extract_scores(result)
        assert "A" in scores
        assert "B" not in scores


class TestComputeIcTstat:
    """Tests for compute_ic_tstat function."""

    def test_insufficient_data(self):
        """Should return None for < 2 values."""
        assert compute_ic_tstat([]) is None
        assert compute_ic_tstat([0.05]) is None

    def test_computes_tstat(self):
        """Should compute t-statistic correctly."""
        # Mean = 0.05, Std = 0.01, N = 4
        # t-stat = 0.05 / (0.01 / sqrt(4)) = 0.05 / 0.005 = 10
        ic_values = [0.04, 0.05, 0.05, 0.06]

        result = compute_ic_tstat(ic_values)
        assert result is not None
        assert result > 0

    def test_zero_std_returns_none(self):
        """Should return None for zero standard deviation."""
        ic_values = [0.05, 0.05, 0.05, 0.05]
        assert compute_ic_tstat(ic_values) is None


class TestBootstrapIcCi:
    """Tests for bootstrap_ic_ci function."""

    def test_returns_tuple(self):
        """Should return (lower, upper) tuple."""
        ic_values = [0.05, 0.06, 0.04, 0.07]
        result = bootstrap_ic_ci(ic_values)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_lower_less_than_upper(self):
        """Lower bound should be less than upper bound."""
        ic_values = [0.05, 0.06, 0.04, 0.07, 0.05, 0.06]
        lower, upper = bootstrap_ic_ci(ic_values)

        assert lower < upper

    def test_mean_within_ci(self):
        """Mean should be within CI."""
        ic_values = [0.05, 0.06, 0.04, 0.07, 0.05, 0.06]
        lower, upper = bootstrap_ic_ci(ic_values)
        ic_mean = mean(ic_values)

        assert lower <= ic_mean <= upper

    def test_insufficient_data_returns_nan(self):
        """Should return NaN for insufficient data."""
        import math
        lower, upper = bootstrap_ic_ci([0.05])

        assert math.isnan(lower)
        assert math.isnan(upper)


class TestComputeRankCorrelation:
    """Tests for compute_rank_correlation function."""

    def test_perfect_correlation(self):
        """Identical rankings should have correlation 1.0."""
        # Need >= 5 common tickers for rank correlation
        scores = {
            "A": Decimal("90"),
            "B": Decimal("80"),
            "C": Decimal("70"),
            "D": Decimal("60"),
            "E": Decimal("50"),
        }

        result = compute_rank_correlation(scores, scores)
        assert result == pytest.approx(1.0)

    def test_inverse_correlation(self):
        """Inverse rankings should have correlation -1.0."""
        scores_a = {
            "A": Decimal("90"),
            "B": Decimal("80"),
            "C": Decimal("70"),
            "D": Decimal("60"),
            "E": Decimal("50"),
        }
        scores_b = {
            "A": Decimal("50"),
            "B": Decimal("60"),
            "C": Decimal("70"),
            "D": Decimal("80"),
            "E": Decimal("90"),
        }

        result = compute_rank_correlation(scores_a, scores_b)
        assert result == pytest.approx(-1.0)

    def test_insufficient_data(self):
        """Should return None for < 5 common tickers."""
        scores_a = {"A": Decimal("90"), "B": Decimal("80")}
        scores_b = {"A": Decimal("70"), "B": Decimal("80")}

        result = compute_rank_correlation(scores_a, scores_b)
        assert result is None


class TestComputeTopBottomSpread:
    """Tests for compute_top_bottom_spread function."""

    def test_positive_spread(self):
        """Top outperforming bottom should have positive spread."""
        scores = {
            "A": Decimal("100"),  # Top
            "B": Decimal("90"),
            "C": Decimal("80"),
            "D": Decimal("70"),
            "E": Decimal("60"),
            "F": Decimal("50"),
            "G": Decimal("40"),
            "H": Decimal("30"),
            "I": Decimal("20"),
            "J": Decimal("10"),  # Bottom
        }
        returns = {
            "A": 0.20, "B": 0.15, "C": 0.10, "D": 0.05, "E": 0.02,  # Top performers
            "F": -0.02, "G": -0.05, "H": -0.10, "I": -0.15, "J": -0.20,  # Bottom performers
        }

        top_mean, bottom_mean, spread, sign_consistent = compute_top_bottom_spread(
            scores, returns, top_n=2
        )

        assert top_mean > bottom_mean
        assert spread > 0

    def test_insufficient_data(self):
        """Should return None values for insufficient data."""
        scores = {"A": Decimal("100"), "B": Decimal("90")}
        returns = {"A": 0.10, "B": 0.05}

        top_mean, bottom_mean, spread, sign_consistent = compute_top_bottom_spread(
            scores, returns, top_n=5
        )

        assert top_mean is None
        assert bottom_mean is None
        assert spread is None


class TestComputeTurnover:
    """Tests for compute_turnover function."""

    def test_no_turnover(self):
        """Same sets should have zero turnover."""
        prev_top = {"A", "B", "C"}
        curr_top = {"A", "B", "C"}

        result = compute_turnover(prev_top, curr_top)
        assert result == 0.0

    def test_complete_turnover(self):
        """Completely different sets should have 100% turnover."""
        prev_top = {"A", "B", "C"}
        curr_top = {"D", "E", "F"}

        result = compute_turnover(prev_top, curr_top)
        assert result == 1.0

    def test_partial_turnover(self):
        """Partial overlap should have proportional turnover."""
        prev_top = {"A", "B", "C", "D"}
        curr_top = {"C", "D", "E", "F"}  # 2 same, 2 different

        result = compute_turnover(prev_top, curr_top)
        # Symmetric difference = {A, B, E, F} = 4
        # Average size = 4
        # Turnover = 4 / (2 * 4) = 0.5
        assert result == 0.5

    def test_empty_sets(self):
        """Empty sets should return None."""
        assert compute_turnover(set(), {"A"}) is None
        assert compute_turnover({"A"}, set()) is None


class TestComputeDrawdown:
    """Tests for compute_drawdown function."""

    def test_no_drawdown(self):
        """Monotonically increasing should have zero drawdown."""
        returns = [1.0, 1.1, 1.2, 1.3]
        result = compute_drawdown(returns)
        assert result == 0.0

    def test_computes_max_drawdown(self):
        """Should compute maximum drawdown."""
        # Peak at 1.2, trough at 0.9
        # Drawdown = (1.2 - 0.9) / 1.2 = 0.25
        returns = [1.0, 1.2, 0.9, 1.0]
        result = compute_drawdown(returns)
        assert result == pytest.approx(0.25)

    def test_empty_returns(self):
        """Empty returns should return 0."""
        result = compute_drawdown([])
        assert result == 0.0


class TestComputeCostedReturn:
    """Tests for compute_costed_return function."""

    def test_no_turnover_no_cost(self):
        """Zero turnover should have no cost drag."""
        result = compute_costed_return(0.10, 0.0, cost_bps=50)
        assert result == 0.10

    def test_full_turnover_applies_cost(self):
        """100% turnover should apply full cost."""
        # 100% turnover * 50 bps = 0.50%
        result = compute_costed_return(0.10, 1.0, cost_bps=50)
        assert result == pytest.approx(0.10 - 0.0050)

    def test_partial_turnover(self):
        """50% turnover should apply half cost."""
        result = compute_costed_return(0.10, 0.5, cost_bps=50)
        assert result == pytest.approx(0.10 - 0.0025)


class TestComputeConcentrationMetrics:
    """Tests for compute_concentration_metrics function."""

    def test_returns_dict(self):
        """Should return a dict with expected keys."""
        scores = {f"T{i}": Decimal(str(100 - i)) for i in range(10)}
        returns = {f"T{i}": 0.05 - i * 0.01 for i in range(10)}

        result = compute_concentration_metrics(scores, returns, top_n=5)

        assert "median_return" in result
        assert "mean_return" in result
        assert "win_rate" in result
        assert "top1_contribution" in result
        assert "top1_ticker" in result
        assert "returns_by_ticker" in result

    def test_insufficient_data(self):
        """Should return None values for insufficient data."""
        scores = {"A": Decimal("100")}
        returns = {"A": 0.10}

        result = compute_concentration_metrics(scores, returns, top_n=5)

        assert result["median_return"] is None
        assert result["win_rate"] is None


class TestConstants:
    """Tests for module constants."""

    def test_default_universe_not_empty(self):
        """Default universe should have tickers."""
        assert len(DEFAULT_UNIVERSE) > 0

    def test_clinical_data_has_phases(self):
        """Clinical data should have phase info."""
        for ticker, data in CLINICAL_DATA.items():
            assert "phase" in data
            assert "lead_phase" in data

    def test_market_cap_data_positive(self):
        """Market cap values should be positive."""
        for ticker, mcap in MARKET_CAP_DATA.items():
            assert mcap > 0
