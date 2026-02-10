"""Tests for backtest/cost_model.py — deterministic transaction cost model."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.cost_model import CostEstimate, CostSchedule, DEFAULT_SCHEDULE, estimate_trade_cost


class TestMonotonic:
    """Cost should increase with trade size and decrease with ADV."""

    def test_monotonic_in_trade_size(self):
        """Larger weight → higher cost (fixed ADV)."""
        adv = 5_000_000
        c3 = estimate_trade_cost(3.0, adv)
        c5 = estimate_trade_cost(5.0, adv)
        c10 = estimate_trade_cost(10.0, adv)
        assert c3.one_way_bps <= c5.one_way_bps <= c10.one_way_bps

    def test_monotonic_in_adv(self):
        """Lower ADV → higher cost (fixed weight)."""
        weight = 5.0
        c_low = estimate_trade_cost(weight, 100_000)
        c_mid = estimate_trade_cost(weight, 1_000_000)
        c_high = estimate_trade_cost(weight, 10_000_000)
        assert c_high.one_way_bps <= c_mid.one_way_bps <= c_low.one_way_bps


class TestEdgeCases:
    """Zero-weight, zero-ADV, and cap behaviour."""

    def test_zero_weight_zero_cost(self):
        """weight=0 → all costs = 0."""
        c = estimate_trade_cost(0.0, 5_000_000)
        assert c.spread_bps == 0.0
        assert c.impact_bps == 0.0
        assert c.one_way_bps == 0.0
        assert c.round_trip_bps == 0.0
        assert c.participation_pct == 0.0

    def test_zero_adv_zero_cost(self):
        """adv=0 → all costs = 0 (guard against division by zero)."""
        c = estimate_trade_cost(5.0, 0.0)
        assert c.one_way_bps == 0.0
        assert c.participation_pct == 0.0

    def test_high_participation_capped(self):
        """Tiny ADV → impact capped at impact_cap_bps."""
        c = estimate_trade_cost(5.0, 1.0)  # extreme: $2.5M trade / $1 ADV
        assert c.impact_bps == DEFAULT_SCHEDULE.impact_cap_bps


class TestSpreadSchedule:
    """Verify each ADV bucket maps to the correct spread."""

    @pytest.mark.parametrize("adv, expected_bps", [
        (15_000_000, 3),   # >= $10M
        (10_000_000, 3),   # exactly $10M
        (7_000_000, 5),    # >= $5M
        (5_000_000, 5),    # exactly $5M
        (2_000_000, 10),   # >= $1M
        (1_000_000, 10),   # exactly $1M
        (750_000, 18),     # >= $500K
        (500_000, 18),     # exactly $500K
        (200_000, 25),     # < $500K
        (1, 25),           # minimal ADV
    ])
    def test_spread_schedule_boundaries(self, adv, expected_bps):
        c = estimate_trade_cost(5.0, adv)
        assert c.spread_bps == expected_bps


class TestInvariants:
    """Algebraic invariants that should always hold."""

    def test_round_trip_is_double_one_way(self):
        c = estimate_trade_cost(5.0, 3_000_000)
        assert c.round_trip_bps == pytest.approx(2 * c.one_way_bps)

    def test_one_way_is_spread_plus_impact(self):
        c = estimate_trade_cost(5.0, 3_000_000)
        assert c.one_way_bps == pytest.approx(c.spread_bps + c.impact_bps)


class TestScheduleId:
    """Schedule identity hashing."""

    def test_schedule_id_deterministic(self):
        """Same params → same id across invocations."""
        s1 = CostSchedule()
        s2 = CostSchedule()
        assert s1.schedule_id == s2.schedule_id

    def test_schedule_id_changes_with_params(self):
        """Different params → different id."""
        s1 = CostSchedule()
        s2 = CostSchedule(impact_coeff=0.20)
        assert s1.schedule_id != s2.schedule_id
