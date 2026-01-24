#!/usr/bin/env python3
"""
Tests for backtest/factor_attribution.py

Tests cover:
- Dataclass construction and serialization
- Utility functions (_to_decimal, _quantize, _rank_data)
- Spearman IC calculation
- T-statistic computation
"""

import pytest
from decimal import Decimal

from backtest.factor_attribution import (
    FactorIC,
    FactorContribution,
    DecayPoint,
    FactorDecayCurve,
    RegimePerformance,
    FactorAttributionResult,
    _to_decimal,
    _quantize,
    _rank_data,
    compute_spearman_ic,
    compute_t_statistic,
)


class TestFactorICDataclass:
    """Tests for FactorIC dataclass."""

    def test_create_positive_ic(self):
        """Should create FactorIC with positive direction."""
        factor_ic = FactorIC(
            factor_name="clinical",
            ic_value=Decimal("0.15"),
            t_statistic=Decimal("2.5"),
            n_observations=50,
            is_significant=True,
            direction="POSITIVE",
        )
        assert factor_ic.factor_name == "clinical"
        assert factor_ic.ic_value == Decimal("0.15")
        assert factor_ic.is_significant is True
        assert factor_ic.direction == "POSITIVE"

    def test_create_negative_ic(self):
        """Should create FactorIC with negative direction."""
        factor_ic = FactorIC(
            factor_name="momentum",
            ic_value=Decimal("-0.10"),
            t_statistic=Decimal("-1.8"),
            n_observations=50,
            is_significant=False,
            direction="NEGATIVE",
        )
        assert factor_ic.direction == "NEGATIVE"
        assert factor_ic.is_significant is False

    def test_to_dict(self):
        """Should serialize to dict correctly."""
        factor_ic = FactorIC(
            factor_name="financial",
            ic_value=Decimal("0.12"),
            t_statistic=Decimal("2.1"),
            n_observations=45,
            is_significant=True,
            direction="POSITIVE",
        )

        result = factor_ic.to_dict()

        assert result["factor_name"] == "financial"
        assert result["ic_value"] == "0.12"
        assert result["t_statistic"] == "2.1"
        assert result["n_observations"] == 45
        assert result["is_significant"] is True
        assert result["direction"] == "POSITIVE"

    def test_to_dict_with_none_values(self):
        """Should handle None values in to_dict."""
        factor_ic = FactorIC(
            factor_name="clinical",
            ic_value=None,
            t_statistic=None,
            n_observations=5,
            is_significant=False,
            direction="NEUTRAL",
        )

        result = factor_ic.to_dict()

        assert result["ic_value"] is None
        assert result["t_statistic"] is None


class TestFactorContributionDataclass:
    """Tests for FactorContribution dataclass."""

    def test_create_contribution(self):
        """Should create FactorContribution."""
        contrib = FactorContribution(
            factor_name="catalyst",
            standalone_ic=Decimal("0.08"),
            marginal_ic=Decimal("0.05"),
            contribution_pct=Decimal("25.0"),
            is_additive=True,
        )
        assert contrib.factor_name == "catalyst"
        assert contrib.is_additive is True

    def test_to_dict(self):
        """Should serialize to dict."""
        contrib = FactorContribution(
            factor_name="momentum",
            standalone_ic=Decimal("0.10"),
            marginal_ic=Decimal("-0.02"),
            contribution_pct=Decimal("-10.0"),
            is_additive=False,
        )

        result = contrib.to_dict()

        assert result["factor_name"] == "momentum"
        assert result["marginal_ic"] == "-0.02"
        assert result["is_additive"] is False


class TestDecayPointAndCurve:
    """Tests for DecayPoint and FactorDecayCurve dataclasses."""

    def test_create_decay_point(self):
        """Should create DecayPoint."""
        point = DecayPoint(
            age_days=7,
            ic_value=Decimal("0.15"),
            n_observations=100,
        )
        assert point.age_days == 7
        assert point.ic_value == Decimal("0.15")

    def test_create_decay_curve(self):
        """Should create FactorDecayCurve."""
        points = [
            DecayPoint(0, Decimal("0.20"), 100),
            DecayPoint(7, Decimal("0.15"), 95),
            DecayPoint(14, Decimal("0.10"), 90),
            DecayPoint(21, Decimal("0.05"), 85),
        ]
        curve = FactorDecayCurve(
            factor_name="clinical",
            decay_points=points,
            half_life_days=14,
            decay_rate=Decimal("0.05"),
        )
        assert curve.factor_name == "clinical"
        assert curve.half_life_days == 14
        assert len(curve.decay_points) == 4

    def test_decay_curve_to_dict(self):
        """Should serialize decay curve to dict."""
        points = [
            DecayPoint(0, Decimal("0.20"), 100),
            DecayPoint(7, Decimal("0.15"), 95),
        ]
        curve = FactorDecayCurve(
            factor_name="momentum",
            decay_points=points,
            half_life_days=10,
            decay_rate=Decimal("0.07"),
        )

        result = curve.to_dict()

        assert result["factor_name"] == "momentum"
        assert result["half_life_days"] == 10
        assert result["decay_rate"] == "0.07"
        assert len(result["decay_points"]) == 2
        assert result["decay_points"][0]["age_days"] == 0
        assert result["decay_points"][0]["ic"] == "0.20"


class TestRegimePerformance:
    """Tests for RegimePerformance dataclass."""

    def test_create_regime_performance(self):
        """Should create RegimePerformance."""
        perf = RegimePerformance(
            regime="BULL",
            ic_mean=Decimal("0.18"),
            ic_median=Decimal("0.15"),
            hit_rate=Decimal("0.70"),
            n_periods=24,
            spread_q5_q1=Decimal("0.08"),
        )
        assert perf.regime == "BULL"
        assert perf.n_periods == 24

    def test_to_dict(self):
        """Should serialize to dict."""
        perf = RegimePerformance(
            regime="BEAR",
            ic_mean=Decimal("0.05"),
            ic_median=Decimal("0.03"),
            hit_rate=Decimal("0.55"),
            n_periods=12,
            spread_q5_q1=Decimal("0.02"),
        )

        result = perf.to_dict()

        assert result["regime"] == "BEAR"
        assert result["ic_mean"] == "0.05"
        assert result["hit_rate"] == "0.55"


class TestToDecimal:
    """Tests for _to_decimal utility function."""

    def test_none_returns_default(self):
        """None should return default value."""
        assert _to_decimal(None) is None
        assert _to_decimal(None, Decimal("0")) == Decimal("0")

    def test_decimal_passthrough(self):
        """Decimal should pass through unchanged."""
        val = Decimal("123.45")
        assert _to_decimal(val) == val

    def test_int_conversion(self):
        """Int should convert to Decimal."""
        assert _to_decimal(42) == Decimal("42")

    def test_float_conversion(self):
        """Float should convert to Decimal via string."""
        result = _to_decimal(3.14)
        assert result == Decimal("3.14")

    def test_string_conversion(self):
        """String should convert to Decimal."""
        assert _to_decimal("99.99") == Decimal("99.99")

    def test_invalid_returns_default(self):
        """Invalid value should return default."""
        assert _to_decimal("not a number", Decimal("0")) == Decimal("0")
        assert _to_decimal("abc") is None


class TestQuantize:
    """Tests for _quantize utility function."""

    def test_default_precision(self):
        """Default precision should be 0.0001."""
        result = _quantize(Decimal("0.123456789"))
        assert result == Decimal("0.1235")

    def test_custom_precision(self):
        """Should respect custom precision."""
        result = _quantize(Decimal("0.123456789"), "0.01")
        assert result == Decimal("0.12")

    def test_rounds_half_up(self):
        """Should round half up."""
        result = _quantize(Decimal("0.1235"), "0.001")
        assert result == Decimal("0.124")


class TestRankData:
    """Tests for _rank_data utility function."""

    def test_empty_list(self):
        """Empty list should return empty list."""
        assert _rank_data([]) == []

    def test_simple_ranking(self):
        """Should rank values correctly."""
        values = [Decimal("10"), Decimal("30"), Decimal("20")]
        ranks = _rank_data(values)
        # 10 -> rank 1, 20 -> rank 2, 30 -> rank 3
        assert ranks == [1.0, 3.0, 2.0]

    def test_tie_handling(self):
        """Should handle ties with average ranks."""
        values = [Decimal("10"), Decimal("20"), Decimal("20"), Decimal("30")]
        ranks = _rank_data(values)
        # 10 -> 1, 20 and 20 -> average of 2,3 = 2.5, 30 -> 4
        assert ranks == [1.0, 2.5, 2.5, 4.0]

    def test_all_same_values(self):
        """All same values should get same rank."""
        values = [Decimal("50"), Decimal("50"), Decimal("50")]
        ranks = _rank_data(values)
        # All get average rank of (1+2+3)/3 = 2
        assert all(r == 2.0 for r in ranks)

    def test_deterministic(self):
        """Ranking should be deterministic."""
        values = [Decimal("5"), Decimal("3"), Decimal("7"), Decimal("1")]
        ranks1 = _rank_data(values)
        ranks2 = _rank_data(values)
        assert ranks1 == ranks2


class TestComputeSpearmanIC:
    """Tests for compute_spearman_ic function."""

    def test_insufficient_data(self):
        """Should return None for < 10 observations."""
        scores = [Decimal(str(i)) for i in range(9)]
        returns = [Decimal(str(i)) for i in range(9)]
        assert compute_spearman_ic(scores, returns) is None

    def test_mismatched_lengths(self):
        """Should return None for mismatched lengths."""
        scores = [Decimal(str(i)) for i in range(15)]
        returns = [Decimal(str(i)) for i in range(10)]
        assert compute_spearman_ic(scores, returns) is None

    def test_perfect_positive_correlation(self):
        """Perfect positive correlation should return 1.0."""
        scores = [Decimal(str(i)) for i in range(20)]
        returns = [Decimal(str(i)) for i in range(20)]
        ic = compute_spearman_ic(scores, returns)
        assert ic is not None
        assert abs(ic - Decimal("1.0")) < Decimal("0.01")

    def test_perfect_negative_correlation(self):
        """Perfect negative correlation should return -1.0."""
        scores = [Decimal(str(i)) for i in range(20)]
        returns = [Decimal(str(19 - i)) for i in range(20)]
        ic = compute_spearman_ic(scores, returns)
        assert ic is not None
        assert abs(ic - Decimal("-1.0")) < Decimal("0.01")

    def test_zero_variance(self):
        """Zero variance should return 0."""
        scores = [Decimal("50")] * 20
        returns = [Decimal(str(i)) for i in range(20)]
        ic = compute_spearman_ic(scores, returns)
        assert ic == Decimal("0")


class TestComputeTStatistic:
    """Tests for compute_t_statistic function."""

    def test_insufficient_data(self):
        """Should return None for < 10 observations."""
        assert compute_t_statistic(Decimal("0.15"), 9) is None

    def test_ic_of_one(self):
        """Should return None for IC = 1."""
        assert compute_t_statistic(Decimal("1.0"), 50) is None

    def test_ic_of_negative_one(self):
        """Should return None for IC = -1."""
        assert compute_t_statistic(Decimal("-1.0"), 50) is None

    def test_positive_ic_positive_t(self):
        """Positive IC should give positive t-statistic."""
        t = compute_t_statistic(Decimal("0.15"), 50)
        assert t is not None
        assert t > Decimal("0")

    def test_negative_ic_negative_t(self):
        """Negative IC should give negative t-statistic."""
        t = compute_t_statistic(Decimal("-0.15"), 50)
        assert t is not None
        assert t < Decimal("0")

    def test_zero_ic_zero_t(self):
        """Zero IC should give zero t-statistic."""
        t = compute_t_statistic(Decimal("0"), 50)
        assert t == Decimal("0")

    def test_larger_n_larger_t(self):
        """Larger n should give larger absolute t for same IC."""
        t_small = compute_t_statistic(Decimal("0.15"), 20)
        t_large = compute_t_statistic(Decimal("0.15"), 100)
        assert t_small is not None and t_large is not None
        assert abs(t_large) > abs(t_small)


class TestDeterminism:
    """Tests ensuring all functions are deterministic."""

    def test_rank_data_deterministic(self):
        """_rank_data should be deterministic."""
        values = [Decimal("5"), Decimal("3"), Decimal("7"), Decimal("3"), Decimal("9")]
        results = [_rank_data(values) for _ in range(5)]
        assert all(r == results[0] for r in results)

    def test_spearman_ic_deterministic(self):
        """compute_spearman_ic should be deterministic."""
        scores = [Decimal(str(i % 10)) for i in range(50)]
        returns = [Decimal(str((i + 3) % 10)) for i in range(50)]

        results = [compute_spearman_ic(scores, returns) for _ in range(5)]
        assert all(r == results[0] for r in results)

    def test_t_statistic_deterministic(self):
        """compute_t_statistic should be deterministic."""
        results = [compute_t_statistic(Decimal("0.15"), 50) for _ in range(5)]
        assert all(r == results[0] for r in results)
