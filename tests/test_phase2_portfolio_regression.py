#!/usr/bin/env python3
"""
Phase-2 Portfolio Regression Test

Pinned test that verifies the strategy portfolio construction produces
deterministic, known-good output for a specific snapshot + ruleset.

This prevents silent ordering/weight drift from code changes to the
decision engine, sort key, or sizing logic.

Pinned on: 2026-02-08
Snapshot:  2025-10-31 archive
Ruleset:   v2_phase2_default (ID: d3cdf5c8, a_floor=0.55)
Policy:    tier_filter=[A,B], top_k=20
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from decision_engine import DecisionRuleset
from run_decision_ruleset_sweep import load_archive_data
from run_decision_strategy_backtest import build_strategy_portfolio
from run_rank_ic_backtest import ARCHIVE_DIR


# =============================================================================
# PINNED EXPECTED VALUES
# =============================================================================

SNAPSHOT_DATE = "2025-10-31"
RULESET_PATH = PROJECT_ROOT / "production_data" / "decision_rulesets" / "v2_phase2_default.json"
ARCHIVE_PATH = ARCHIVE_DIR / f"{SNAPSHOT_DATE}.tar.gz"
TIER_FILTER = ["A", "B"]
TOP_K = 20

# Pinned top-10 tickers in exact actionable order
EXPECTED_TOP_10 = [
    {"ticker": "CNTX",  "rank": 1,  "tier": "A", "band": "L", "weight": 5.78},
    {"ticker": "KALV",  "rank": 2,  "tier": "A", "band": "L", "weight": 5.78},
    {"ticker": "XENE",  "rank": 3,  "tier": "B", "band": "M", "weight": 3.47},
    {"ticker": "PVLA",  "rank": 4,  "tier": "B", "band": "L", "weight": 5.78},
    {"ticker": "VRDN",  "rank": 5,  "tier": "B", "band": "M", "weight": 3.47},
    {"ticker": "AKRO",  "rank": 6,  "tier": "B", "band": "L", "weight": 5.78},
    {"ticker": "DNTH",  "rank": 7,  "tier": "B", "band": "L", "weight": 5.78},
    {"ticker": "NUVL",  "rank": 8,  "tier": "B", "band": "L", "weight": 5.78},
    {"ticker": "NAUT",  "rank": 9,  "tier": "B", "band": "L", "weight": 5.78},
    {"ticker": "CELC",  "rank": 10, "tier": "B", "band": "L", "weight": 5.78},
]

EXPECTED_N_POSITIONS = 20


# =============================================================================
# TESTS
# =============================================================================

@pytest.fixture(scope="module")
def portfolio():
    """Load archive once and build the pinned portfolio."""
    if not ARCHIVE_PATH.exists():
        pytest.skip(f"Archive not found: {ARCHIVE_PATH}")
    if not RULESET_PATH.exists():
        pytest.skip(f"Ruleset not found: {RULESET_PATH}")

    ruleset = DecisionRuleset.from_json(str(RULESET_PATH))
    ad = load_archive_data(ARCHIVE_PATH, SNAPSHOT_DATE)
    return build_strategy_portfolio(ad, ruleset, TIER_FILTER, TOP_K)


class TestPhase2PortfolioRegression:

    def test_position_count(self, portfolio):
        """Portfolio should have exactly the expected number of positions."""
        assert len(portfolio) == EXPECTED_N_POSITIONS

    def test_top_10_tickers(self, portfolio):
        """Top-10 tickers should match in exact order."""
        actual_top10 = [p["ticker"] for p in portfolio[:10]]
        expected_top10 = [e["ticker"] for e in EXPECTED_TOP_10]
        assert actual_top10 == expected_top10, (
            f"Top-10 ticker order changed!\n"
            f"  Expected: {expected_top10}\n"
            f"  Actual:   {actual_top10}"
        )

    def test_top_10_ranks(self, portfolio):
        """Actionable ranks should be 1-indexed and sequential."""
        for i, pos in enumerate(portfolio[:10]):
            expected = EXPECTED_TOP_10[i]
            assert pos["actionable_rank"] == expected["rank"], (
                f"Rank mismatch for {pos['ticker']}: "
                f"expected {expected['rank']}, got {pos['actionable_rank']}"
            )

    def test_top_10_tiers(self, portfolio):
        """Each top-10 position should have the expected tier."""
        for i, pos in enumerate(portfolio[:10]):
            expected = EXPECTED_TOP_10[i]
            assert pos["tier_dev"] == expected["tier"], (
                f"Tier mismatch for {pos['ticker']}: "
                f"expected {expected['tier']}, got {pos['tier_dev']}"
            )

    def test_top_10_size_bands(self, portfolio):
        """Each top-10 position should have the expected size band."""
        for i, pos in enumerate(portfolio[:10]):
            expected = EXPECTED_TOP_10[i]
            assert pos["size_band"] == expected["band"], (
                f"Size band mismatch for {pos['ticker']}: "
                f"expected {expected['band']}, got {pos['size_band']}"
            )

    def test_top_10_weights(self, portfolio):
        """Each top-10 position should have the expected weight (within 0.01%)."""
        for i, pos in enumerate(portfolio[:10]):
            expected = EXPECTED_TOP_10[i]
            assert abs(pos["weight_pct"] - expected["weight"]) < 0.01, (
                f"Weight mismatch for {pos['ticker']}: "
                f"expected {expected['weight']}, got {pos['weight_pct']}"
            )

    def test_weights_sum_to_100(self, portfolio):
        """All weights should sum to approximately 100%."""
        total = sum(p["weight_pct"] for p in portfolio)
        assert abs(total - 100.0) < 1.0, (
            f"Weight sum = {total:.2f}%, expected ~100%"
        )

    def test_only_ab_tiers(self, portfolio):
        """All positions should be tier A or B."""
        for pos in portfolio:
            assert pos["tier_dev"] in ("A", "B"), (
                f"{pos['ticker']} has tier {pos['tier_dev']}, expected A or B"
            )

    def test_ruleset_id(self):
        """The Phase-2 ruleset should produce the expected ID."""
        if not RULESET_PATH.exists():
            pytest.skip(f"Ruleset not found: {RULESET_PATH}")
        ruleset = DecisionRuleset.from_json(str(RULESET_PATH))
        assert ruleset.ruleset_id == "d3cdf5c8", (
            f"Ruleset ID changed: expected d3cdf5c8, got {ruleset.ruleset_id}"
        )

    def test_a_floor_055(self):
        """Phase-2 ruleset should have a_floor=0.55 (key change from v1)."""
        if not RULESET_PATH.exists():
            pytest.skip(f"Ruleset not found: {RULESET_PATH}")
        ruleset = DecisionRuleset.from_json(str(RULESET_PATH))
        assert ruleset.tier_a_optionality_floor == 0.55

    def test_hard_drawdown_mode(self):
        """Phase-2 should use hard drawdown (not soft)."""
        if not RULESET_PATH.exists():
            pytest.skip(f"Ruleset not found: {RULESET_PATH}")
        ruleset = DecisionRuleset.from_json(str(RULESET_PATH))
        assert ruleset.drawdown_gate_mode == "hard"
