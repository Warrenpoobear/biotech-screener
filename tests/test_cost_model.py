"""Tests for backtest/cost_model.py — deterministic transaction cost model."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.cost_model import (
    CostEstimate,
    CostSchedule,
    DEFAULT_SCHEDULE,
    compute_cost_telemetry,
    estimate_trade_cost,
)
from decision_engine import DecisionRuleset


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


class TestCapDispersion:
    """Cap degeneracy: low cap compresses costs, high cap reveals dispersion."""

    def test_high_cap_reveals_dispersion(self):
        """Two ADVs ($20M vs $500K) at 5% weight with high cap → costs diverge > 100 bps."""
        high_cap = CostSchedule(impact_cap_bps=2000.0)
        c_liquid = estimate_trade_cost(5.0, 20_000_000, high_cap)
        c_illiquid = estimate_trade_cost(5.0, 500_000, high_cap)
        gap = c_illiquid.round_trip_bps - c_liquid.round_trip_bps
        assert gap > 100, f"Expected >100 bps gap with high cap, got {gap}"

    def test_low_cap_compresses_costs(self):
        """Same pair with cap=200 → both hit cap, costs converge."""
        low_cap = CostSchedule(impact_cap_bps=200.0)
        c_liquid = estimate_trade_cost(5.0, 20_000_000, low_cap)
        c_illiquid = estimate_trade_cost(5.0, 500_000, low_cap)
        # Both should be cap-bound on impact — gap comes only from spread difference
        # Spread gap: 18 bps ($500K) - 3 bps ($20M) = 15 bps one-way = 30 bps round-trip
        gap = c_illiquid.round_trip_bps - c_liquid.round_trip_bps
        assert gap <= 50, f"Expected <=50 bps gap with low cap, got {gap}"


class TestCostTelemetry:
    """Tests for compute_cost_telemetry()."""

    def test_telemetry_basics(self):
        """Mixed estimates: some cap-binding, some not."""
        sched = CostSchedule(impact_cap_bps=200.0)
        estimates = [
            CostEstimate(spread_bps=5, impact_bps=200.0, one_way_bps=205,
                         round_trip_bps=410, participation_pct=10.0, adv_dollars=1e6),
            CostEstimate(spread_bps=3, impact_bps=50.0, one_way_bps=53,
                         round_trip_bps=106, participation_pct=2.0, adv_dollars=5e6),
            CostEstimate(spread_bps=10, impact_bps=199.99, one_way_bps=209.99,
                         round_trip_bps=419.98, participation_pct=8.0, adv_dollars=2e6),
        ]
        telem = compute_cost_telemetry(n_positions=20, estimates=estimates, schedule=sched)
        assert telem["n_costed"] == 3
        assert telem["n_positions"] == 20
        assert telem["cost_coverage_pct"] == 15.0  # 3/20 * 100
        # estimates[0]: 200.0 >= 199.99 → cap-binding
        # estimates[2]: 199.99 >= 199.99 → cap-binding (within 0.01 tolerance)
        assert telem["cap_binding_pct"] == pytest.approx(66.7, abs=0.1)  # 2/3

    def test_telemetry_no_estimates(self):
        """No estimates → coverage 0%, cap_binding 0%."""
        sched = DEFAULT_SCHEDULE
        telem = compute_cost_telemetry(n_positions=10, estimates=[], schedule=sched)
        assert telem["cost_coverage_pct"] == 0.0
        assert telem["cap_binding_pct"] == 0.0
        assert telem["n_costed"] == 0

    def test_telemetry_zero_positions(self):
        """Zero positions → coverage 0%."""
        sched = DEFAULT_SCHEDULE
        telem = compute_cost_telemetry(n_positions=0, estimates=[], schedule=sched)
        assert telem["cost_coverage_pct"] == 0.0


class TestDefaultCapInvariant:
    """DecisionRuleset default cap must match DEFAULT_SCHEDULE cap."""

    def test_default_cap_matches_schedule(self):
        """cost_impact_cap_bps default == DEFAULT_SCHEDULE.impact_cap_bps."""
        assert DecisionRuleset().cost_impact_cap_bps == DEFAULT_SCHEDULE.impact_cap_bps
