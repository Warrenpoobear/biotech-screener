#!/usr/bin/env python3
"""Regression tests: est_cost_bps wiring in the production path (run_screen.py).

Exercises the same logic as save_validation_snapshot()'s inner loop:
  1. Initialize CostSchedule when ruleset.enable_cost_haircut is True
  2. Compute est_cost_bps from market_data avg_volume * price
  3. Pass est_cost_bps to compute_decision_fields()

Three scenarios:
  - Haircut enabled + ADV present  → cost_mult < 1.0
  - Haircut disabled               → cost_mult == 1.0
  - Haircut enabled, no market data → graceful degradation (cost_mult == 1.0)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.cost_model import CostSchedule, estimate_trade_cost
from decision_engine import DecisionRuleset, compute_decision_fields


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ENABLED_RS = DecisionRuleset(enable_cost_haircut=True, cost_impact_cap_bps=1000.0)
DISABLED_RS = DecisionRuleset(enable_cost_haircut=False)

# Representative weight matching production: 100 / top_k(20) = 5.0%
REP_WEIGHT = 5.0


def _minimal_rec(drawdown=-0.10):
    """Minimal rec that passes eligibility gates."""
    return {
        "defensive_features": {
            "drawdown": drawdown,
            "drawdown_xbi": -0.05,
            "drawdown_rel_xbi": drawdown - (-0.05),
            "vol_60d": "0.30",
            "beta_xbi_60d": "0.80",
            "rsi_14d": "50",
        },
        "fundamental_red_flag": False,
        "severity": "",
        "confidence_overall": 0.5,
        "smart_money_signal": {},
        "coinvest": {},
        "catalyst_decay": {},
        "score_breakdown": {"enhancements": {}},
        "momentum_signal": {},
        "data_quality_flags": [],
    }


def _run_production_cost_logic(ticker, ruleset, market_data_by_ticker):
    """Replicate the production cost-wiring logic from save_validation_snapshot."""
    cost_schedule = None
    rep_weight = 0.0
    if ruleset and ruleset.enable_cost_haircut and market_data_by_ticker:
        cost_schedule = CostSchedule(
            impact_cap_bps=ruleset.cost_impact_cap_bps,
        )
        rep_weight = REP_WEIGHT

    est_cost_bps = None
    if cost_schedule is not None:
        md = market_data_by_ticker.get(ticker, {})
        adv = (md.get("avg_volume") or 0) * (md.get("price") or 0)
        if adv > 0:
            cost_est = estimate_trade_cost(rep_weight, adv, cost_schedule)
            est_cost_bps = cost_est.round_trip_bps

    rec = _minimal_rec()
    return compute_decision_fields(
        rec, "drug_developer", 0.80, ruleset=ruleset, est_cost_bps=est_cost_bps,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestCostProductionWiring:
    """Verify est_cost_bps flows through the production path."""

    def test_haircut_enabled_with_adv(self):
        """Low-ADV ticker gets cost haircut when enabled."""
        # $100K ADV → very illiquid → high cost → haircut
        market_data = {
            "XTST": {"avg_volume": 10_000, "price": 10.0},  # ADV = $100K
        }
        fields = _run_production_cost_logic("XTST", ENABLED_RS, market_data)

        assert fields["cost_mult"] < 1.0, "Expected cost haircut for low-ADV ticker"
        assert fields["cost_haircut_applied"] == "1"
        assert fields["cost_bucket"] != ""

    def test_haircut_disabled(self):
        """Disabled ruleset produces no haircut even with ADV data."""
        market_data = {
            "XTST": {"avg_volume": 10_000, "price": 10.0},
        }
        fields = _run_production_cost_logic("XTST", DISABLED_RS, market_data)

        assert fields["cost_mult"] == 1.0
        assert fields["cost_haircut_applied"] == "0"
        assert fields["cost_bucket"] == ""

    def test_haircut_enabled_no_market_data(self):
        """Enabled but no market data → graceful degradation, no crash."""
        fields = _run_production_cost_logic("XTST", ENABLED_RS, None)

        assert fields["cost_mult"] == 1.0
        assert fields["cost_haircut_applied"] == "0"
        assert fields["cost_bucket"] == ""
