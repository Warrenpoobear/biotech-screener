#!/usr/bin/env python3
"""
Tests for baker_overlay.py — Baker Bros CIK attribution, IC review queues, alignment metrics.
"""
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from decimal import Decimal
from baker_overlay import (
    compute_baker_overlay,
    _classify_activity,
    _risk_gates_pass,
    _guardrail_blocks_bonus,
    _extract_disagreement_reasons,
    BAKER_CIK,
    Q1_WEIGHT_PCT,
    Q1_WEIGHT_RANK,
    Q1_VALUE_USD,
    Q1_VALUE_RANK,
    Q1_ACTIVITY_RANK,
    Q1_ACTIVITY_MIN_WEIGHT,
    Q2_CATALYST_MIN_DAYS,
    Q2_CATALYST_MAX_DAYS,
    Q2_RANK,
)


# =============================================================================
# Helpers: build minimal test data
# =============================================================================

def _make_holdings(baker_positions: dict, other_positions: dict = None):
    """Build holdings_snapshots dict.

    baker_positions: {ticker: (current_usd, prior_usd)}
    other_positions: {ticker: {cik: (current_usd, prior_usd)}}
    """
    holdings = {}
    for ticker, (curr, prior) in baker_positions.items():
        holdings[ticker] = {
            "holdings": {
                "current": {BAKER_CIK: {"value_kusd": curr}} if curr > 0 else {},
                "prior": {BAKER_CIK: {"value_kusd": prior}} if prior > 0 else {},
            }
        }
    if other_positions:
        for ticker, cik_data in other_positions.items():
            if ticker not in holdings:
                holdings[ticker] = {"holdings": {"current": {}, "prior": {}}}
            for cik, (curr, prior) in cik_data.items():
                if curr > 0:
                    holdings[ticker]["holdings"]["current"][cik] = {"value_kusd": curr}
                if prior > 0:
                    holdings[ticker]["holdings"]["prior"][cik] = {"value_kusd": prior}
    return holdings


def _make_ranked(tickers_and_ranks, severity="none", flags=None, catalyst_days=None):
    """Build ranked_securities list.

    tickers_and_ranks: [(ticker, composite_rank), ...]
    """
    result = []
    for ticker, rank in tickers_and_ranks:
        sec = {
            "ticker": ticker,
            "composite_rank": rank,
            "composite_score": str(100 - rank),
            "severity": severity,
            "flags": flags or [],
            "catalyst_decay": {"days_to_catalyst": catalyst_days} if catalyst_days is not None else {},
        }
        result.append(sec)
    return result


# =============================================================================
# Activity classification
# =============================================================================

class TestActivityClassification:
    def test_new_position(self):
        activity, pct = _classify_activity(100_000, 0)
        assert activity == "new"
        assert pct is None

    def test_exited_position(self):
        activity, pct = _classify_activity(0, 100_000)
        assert activity == "exited"
        assert pct == -1.0

    def test_unchanged_position(self):
        activity, pct = _classify_activity(100_000, 95_000)
        assert activity == "unchanged"
        assert abs(pct - 0.0526) < 0.01

    def test_add_position(self):
        activity, pct = _classify_activity(150_000, 100_000)
        assert activity == "add"
        assert pct == 0.5

    def test_trim_position(self):
        activity, pct = _classify_activity(70_000, 100_000)
        assert activity == "trim"
        assert pct == -0.3

    def test_both_zero(self):
        activity, pct = _classify_activity(0, 0)
        assert activity == "unchanged"
        assert pct == 0.0


# =============================================================================
# Baker signal extraction
# =============================================================================

class TestBakerSignals:
    def test_baker_held_true(self):
        holdings = _make_holdings({"AAAA": (500_000, 400_000)})
        ranked = _make_ranked([("AAAA", 1)])
        result = compute_baker_overlay(ranked, holdings)
        sec = result["securities"][0]
        assert sec["held"] is True
        assert sec["portfolio_weight_pct"] == 100.0
        assert sec["value_usd"] == 500_000
        assert sec["activity"] == "add"

    def test_baker_not_held(self):
        holdings = _make_holdings({"AAAA": (500_000, 400_000)})
        ranked = _make_ranked([("BBBB", 1)])
        result = compute_baker_overlay(ranked, holdings)
        sec = result["securities"][0]
        assert sec["held"] is False
        assert sec["portfolio_weight_pct"] == 0.0

    def test_exited_position_not_held(self):
        holdings = _make_holdings({"AAAA": (0, 400_000)})
        ranked = _make_ranked([("AAAA", 1)])
        result = compute_baker_overlay(ranked, holdings)
        sec = result["securities"][0]
        assert sec["held"] is False
        assert sec["activity"] == "exited"

    def test_portfolio_weight_computation(self):
        holdings = _make_holdings({
            "AAAA": (300_000, 200_000),  # 30%
            "BBBB": (700_000, 500_000),  # 70%
        })
        ranked = _make_ranked([("AAAA", 1), ("BBBB", 2)])
        result = compute_baker_overlay(ranked, holdings)
        sec_a = result["securities"][0]
        sec_b = result["securities"][1]
        assert abs(sec_a["portfolio_weight_pct"] - 30.0) < 0.01
        assert abs(sec_b["portfolio_weight_pct"] - 70.0) < 0.01

    def test_empty_inputs(self):
        result = compute_baker_overlay([], {})
        assert result["securities"] == []
        assert result["queue_1_core_divergence"] == []

    def test_total_value_b_correct_units(self):
        """baker_total_value_b divides by 1 billion, not 1 million."""
        # 10B total in raw USD
        holdings = _make_holdings({"AAAA": (10_000_000_000, 8_000_000_000)})
        ranked = _make_ranked([("AAAA", 1)])
        result = compute_baker_overlay(ranked, holdings)
        assert result["metrics"]["baker_total_value_b"] == 10.0


# =============================================================================
# Ranking
# =============================================================================

class TestBakerRanking:
    def test_rank_by_weight(self):
        holdings = _make_holdings({
            "AAAA": (100_000, 80_000),
            "BBBB": (300_000, 250_000),
            "CCCC": (200_000, 180_000),
        })
        ranked = _make_ranked([("AAAA", 1), ("BBBB", 2), ("CCCC", 3)])
        result = compute_baker_overlay(ranked, holdings)
        # BBBB has highest weight -> rank 1
        assert result["securities"][1]["rank_by_weight"] == 1  # BBBB
        assert result["securities"][2]["rank_by_weight"] == 2  # CCCC
        assert result["securities"][0]["rank_by_weight"] == 3  # AAAA

    def test_rank_by_activity(self):
        holdings = _make_holdings({
            "AAAA": (100_000, 0),          # new (priority 0)
            "BBBB": (300_000, 250_000),    # add (priority 1)
            "CCCC": (200_000, 195_000),    # unchanged (priority 2)
        })
        ranked = _make_ranked([("AAAA", 1), ("BBBB", 2), ("CCCC", 3)])
        result = compute_baker_overlay(ranked, holdings)
        assert result["securities"][0]["rank_by_activity"] == 1  # AAAA (new)
        assert result["securities"][1]["rank_by_activity"] == 2  # BBBB (add)
        assert result["securities"][2]["rank_by_activity"] == 3  # CCCC (unchanged)

    def test_non_held_gets_none_rank(self):
        holdings = _make_holdings({"AAAA": (100_000, 80_000)})
        ranked = _make_ranked([("AAAA", 1), ("BBBB", 2)])
        result = compute_baker_overlay(ranked, holdings)
        assert result["securities"][1]["rank_by_weight"] is None
        assert result["securities"][1]["rank_by_activity"] is None


# =============================================================================
# Queue 1: Baker Core Divergence
# =============================================================================

class TestQueue1:
    def test_weight_trigger(self):
        """weight >= 1% AND composite_rank > 75 -> Queue 1."""
        # Total = 1M, AAAA = 20K (2%) at rank 80
        holdings = _make_holdings({
            "AAAA": (20_000, 15_000),   # 2% weight
            "BBBB": (980_000, 900_000),  # 98% weight
        })
        ranked = _make_ranked([("AAAA", 80), ("BBBB", 5)])
        result = compute_baker_overlay(ranked, holdings)
        q1_tickers = [q["ticker"] for q in result["queue_1_core_divergence"]]
        # AAAA: weight 2% > 1%, rank 80 > 75 -> in queue
        assert "AAAA" in q1_tickers
        # BBBB: weight 98% > 1%, rank 5 <= 75 -> NOT in queue (rank too good)
        assert "BBBB" not in q1_tickers

    def test_value_trigger_with_weight_floor(self):
        """value >= $250M AND weight >= 0.25% AND rank > 75 -> Queue 1."""
        # Total = 10B. AAAA = 300M (3%), BBBB = 100M (1%), CCCC = 10M (0.1%)
        holdings = _make_holdings({
            "AAAA": (300_000_000, 250_000_000),  # $300M, 3% weight
            "BBBB": (100_000_000, 80_000_000),   # $100M, 1% weight
            "CCCC": (10_000_000, 5_000_000),     # $10M, 0.1% weight (below floor)
            "DDDD": (9_590_000_000, 9_000_000_000),  # filler
        })
        ranked = _make_ranked([("AAAA", 80), ("BBBB", 80), ("CCCC", 80), ("DDDD", 5)])
        result = compute_baker_overlay(ranked, holdings)
        q1_tickers = [q["ticker"] for q in result["queue_1_core_divergence"]]
        # AAAA: value $300M >= $250M, weight 3% >= 0.25%, rank 80 > 75 -> in queue
        assert "AAAA" in q1_tickers
        # BBBB: value $100M < $250M -> value trigger doesn't fire.
        #        weight 1% >= 1% -> weight trigger fires at rank 80 > 75
        assert "BBBB" in q1_tickers  # via weight trigger
        # CCCC: weight 0.1% < 0.25% floor -> excluded from value AND activity triggers
        #        weight 0.1% < 1% -> weight trigger doesn't fire either
        assert "CCCC" not in q1_tickers

    def test_activity_trigger_with_weight_floor(self):
        """activity in {new, add} AND weight >= 0.25% AND rank > 75 -> Queue 1."""
        # Total = 10B. AAAA = 50M new (0.5%), BBBB = 5M new (0.05%)
        holdings = _make_holdings({
            "AAAA": (50_000_000, 0),              # new, 0.5% weight
            "BBBB": (5_000_000, 0),               # new, 0.05% weight (below floor)
            "CCCC": (9_945_000_000, 9_000_000_000),  # filler
        })
        ranked = _make_ranked([("AAAA", 80), ("BBBB", 80), ("CCCC", 10)])
        result = compute_baker_overlay(ranked, holdings)
        q1_tickers = [q["ticker"] for q in result["queue_1_core_divergence"]]
        q1_reasons = {q["ticker"]: q["queue_reasons"] for q in result["queue_1_core_divergence"]}
        # AAAA: new + weight 0.5% >= 0.25% + rank 80 > 75 -> in queue
        assert "AAAA" in q1_tickers
        assert any("activity=new" in r for r in q1_reasons.get("AAAA", []))
        # BBBB: new but weight 0.05% < 0.25% floor -> excluded
        assert "BBBB" not in q1_tickers

    def test_good_rank_excluded(self):
        """Baker-held at rank 10 -> no divergence, not in Queue 1."""
        holdings = _make_holdings({"AAAA": (500_000, 400_000)})
        ranked = _make_ranked([("AAAA", 10)])
        result = compute_baker_overlay(ranked, holdings)
        assert len(result["queue_1_core_divergence"]) == 0

    def test_rank_75_boundary_excluded(self):
        """rank == 75 (not > 75) -> NOT in Queue 1."""
        holdings = _make_holdings({"AAAA": (500_000, 400_000)})
        ranked = _make_ranked([("AAAA", 75)])
        result = compute_baker_overlay(ranked, holdings)
        assert len(result["queue_1_core_divergence"]) == 0

    def test_queue_sorted_by_weight_desc(self):
        """Queue 1 entries sorted by baker weight descending."""
        holdings = _make_holdings({
            "AAAA": (100_000_000, 80_000_000),
            "BBBB": (300_000_000, 250_000_000),
            "CCCC": (200_000_000, 180_000_000),
        })
        ranked = _make_ranked([("AAAA", 80), ("BBBB", 80), ("CCCC", 80)])
        result = compute_baker_overlay(ranked, holdings)
        q1 = result["queue_1_core_divergence"]
        weights = [q["baker_portfolio_weight_pct"] for q in q1]
        assert weights == sorted(weights, reverse=True)

    def test_disagreement_reasons_present(self):
        """Queue 1 entries have disagreement_reasons field."""
        holdings = _make_holdings({"AAAA": (500_000, 400_000)})
        ranked = _make_ranked([("AAAA", 80)], severity="sev1")
        result = compute_baker_overlay(ranked, holdings)
        q1 = result["queue_1_core_divergence"]
        assert len(q1) == 1
        assert "disagreement_reasons" in q1[0]
        assert "severity_sev1" in q1[0]["disagreement_reasons"]


# =============================================================================
# Queue 2: Baker Near-Term Catalyst
# =============================================================================

class TestQueue2:
    def test_catalyst_in_window(self):
        """Baker-held, catalyst in 15-120 days, rank > 50, risk OK -> Queue 2."""
        holdings = _make_holdings({"AAAA": (100_000, 80_000)})
        ranked = _make_ranked([("AAAA", 60)], catalyst_days=30)
        result = compute_baker_overlay(ranked, holdings)
        q2_tickers = [q["ticker"] for q in result["queue_2_near_term_catalyst"]]
        assert "AAAA" in q2_tickers

    def test_catalyst_too_soon(self):
        """days_to_catalyst < 15 -> NOT in Queue 2."""
        holdings = _make_holdings({"AAAA": (100_000, 80_000)})
        ranked = _make_ranked([("AAAA", 60)], catalyst_days=10)
        result = compute_baker_overlay(ranked, holdings)
        assert len(result["queue_2_near_term_catalyst"]) == 0

    def test_catalyst_too_far(self):
        """days_to_catalyst > 120 -> NOT in Queue 2."""
        holdings = _make_holdings({"AAAA": (100_000, 80_000)})
        ranked = _make_ranked([("AAAA", 60)], catalyst_days=150)
        result = compute_baker_overlay(ranked, holdings)
        assert len(result["queue_2_near_term_catalyst"]) == 0

    def test_good_rank_excluded(self):
        """composite_rank <= 50 -> NOT in Queue 2."""
        holdings = _make_holdings({"AAAA": (100_000, 80_000)})
        ranked = _make_ranked([("AAAA", 30)], catalyst_days=30)
        result = compute_baker_overlay(ranked, holdings)
        assert len(result["queue_2_near_term_catalyst"]) == 0

    def test_risk_gate_blocks(self):
        """sev2 severity -> risk gates fail -> NOT in Queue 2."""
        holdings = _make_holdings({"AAAA": (100_000, 80_000)})
        ranked = _make_ranked([("AAAA", 60)], severity="sev2", catalyst_days=30)
        result = compute_baker_overlay(ranked, holdings)
        assert len(result["queue_2_near_term_catalyst"]) == 0

    def test_survivability_critical_blocks(self):
        """guardrail_a_survivability_critical flag -> NOT in Queue 2."""
        holdings = _make_holdings({"AAAA": (100_000, 80_000)})
        ranked = _make_ranked(
            [("AAAA", 60)],
            flags=["guardrail_a_survivability_critical"],
            catalyst_days=30,
        )
        result = compute_baker_overlay(ranked, holdings)
        assert len(result["queue_2_near_term_catalyst"]) == 0

    def test_catalyst_from_module3_summaries(self):
        """Falls back to module 3 summaries when catalyst_decay is empty."""
        holdings = _make_holdings({"AAAA": (100_000, 80_000)})
        ranked = _make_ranked([("AAAA", 60)])  # no catalyst_days in catalyst_decay
        m3_summaries = {
            "AAAA": {
                "integration": {"catalyst_window_days": 45, "next_catalyst_date": "2026-03-20"},
            }
        }
        result = compute_baker_overlay(ranked, holdings, catalyst_summaries=m3_summaries)
        q2_tickers = [q["ticker"] for q in result["queue_2_near_term_catalyst"]]
        assert "AAAA" in q2_tickers

    def test_queue_sorted_by_days_asc(self):
        """Queue 2 sorted by days_to_catalyst ascending."""
        holdings = _make_holdings({
            "AAAA": (100_000, 80_000),
            "BBBB": (200_000, 150_000),
        })
        ranked = [
            {**_make_ranked([("AAAA", 60)], catalyst_days=60)[0]},
            {**_make_ranked([("BBBB", 70)], catalyst_days=20)[0]},
        ]
        result = compute_baker_overlay(ranked, holdings)
        q2 = result["queue_2_near_term_catalyst"]
        assert len(q2) == 2
        assert q2[0]["ticker"] == "BBBB"  # 20 days, soonest
        assert q2[1]["ticker"] == "AAAA"  # 60 days


# =============================================================================
# Alignment metrics
# =============================================================================

class TestAlignmentMetrics:
    def test_recall_at_k(self):
        """Recall@50: fraction of Baker top-20 in composite top-50."""
        # Create 100 tickers, Baker holds top 20 by value
        baker_tickers = {f"BK{i:02d}": (100_000 * (21 - i), 80_000 * (21 - i)) for i in range(1, 21)}
        holdings = _make_holdings(baker_tickers)
        # Place 15 of Baker's top-20 in composite top-50, 5 outside
        ranked_list = []
        for i in range(1, 21):
            ticker = f"BK{i:02d}"
            if i <= 15:
                ranked_list.append((ticker, i * 3))   # ranks 3-45, all in top-50
            else:
                ranked_list.append((ticker, 50 + i))   # ranks 66-70, outside top-50
        # Add filler to get to 100
        for j in range(80):
            ranked_list.append((f"FL{j:02d}", 21 + j))
        ranked_list.sort(key=lambda x: x[1])
        ranked = _make_ranked(ranked_list)
        result = compute_baker_overlay(ranked, holdings)
        # 15 of Baker's top-20 are in composite top-50
        assert result["metrics"]["recall_at_50"] == 0.75
        # All 20 should be in composite top-100
        assert result["metrics"]["recall_at_100"] == 1.0

    def test_divergence_pct(self):
        """baker_divergence_pct measures rank divergence normalized by universe size."""
        holdings = _make_holdings({
            "AAAA": (500_000, 400_000),  # weight rank 1
            "BBBB": (300_000, 250_000),  # weight rank 2
        })
        # AAAA at composite rank 100, BBBB at composite rank 1
        ranked = _make_ranked([("BBBB", 1), ("AAAA", 100)])
        result = compute_baker_overlay(ranked, holdings)
        assert result["metrics"]["baker_divergence_pct"] > 0
        assert result["metrics"]["baker_avg_rank_divergence"] > 0

    def test_baker_held_count(self):
        """Metrics include correct held count."""
        holdings = _make_holdings({
            "AAAA": (100_000, 80_000),
            "BBBB": (0, 50_000),  # exited
        })
        ranked = _make_ranked([("AAAA", 1), ("BBBB", 2), ("CCCC", 3)])
        result = compute_baker_overlay(ranked, holdings)
        assert result["metrics"]["baker_held_count"] == 1  # BBBB is exited


# =============================================================================
# Bonus computation
# =============================================================================

class TestBonus:
    def test_bonus_weight_proportional(self):
        """Bonus = min(weight_pct, 2.0), capped at 2."""
        holdings = _make_holdings({
            "AAAA": (500_000, 400_000),  # 50% weight
            "BBBB": (500_000, 400_000),  # 50% weight
        })
        ranked = _make_ranked([("AAAA", 1), ("BBBB", 2)])
        result = compute_baker_overlay(ranked, holdings)
        # 50% weight -> min(50, 2) = 2.00
        assert result["securities"][0]["bonus"] == "2.00"

    def test_bonus_small_weight(self):
        """Small weight -> proportional bonus."""
        holdings = _make_holdings({
            "AAAA": (5_000, 4_000),        # 0.5% weight
            "BBBB": (995_000, 900_000),    # 99.5% weight
        })
        ranked = _make_ranked([("AAAA", 1), ("BBBB", 2)])
        result = compute_baker_overlay(ranked, holdings)
        assert result["securities"][0]["bonus"] == "0.50"

    def test_bonus_blocked_by_sev2(self):
        """Guardrail A blocks bonus for sev2."""
        holdings = _make_holdings({"AAAA": (100_000, 80_000)})
        ranked = _make_ranked(
            [("AAAA", 1)],
            flags=["guardrail_a_sev_sev2"],
        )
        result = compute_baker_overlay(ranked, holdings)
        assert result["securities"][0]["bonus"] == "0"
        assert result["securities"][0].get("bonus_blocked") is True

    def test_bonus_blocked_by_red_flag(self):
        """Guardrail A2 blocks bonus for red-flag-eligible."""
        holdings = _make_holdings({"AAAA": (100_000, 80_000)})
        ranked = _make_ranked(
            [("AAAA", 1)],
            flags=["guardrail_a_red_flag_eligible:single_asset_early"],
        )
        result = compute_baker_overlay(ranked, holdings)
        assert result["securities"][0]["bonus"] == "0"
        assert result["securities"][0].get("bonus_blocked") is True

    def test_bonus_not_held(self):
        """Non-held ticker gets zero bonus."""
        holdings = _make_holdings({"AAAA": (100_000, 80_000)})
        ranked = _make_ranked([("BBBB", 1)])
        result = compute_baker_overlay(ranked, holdings)
        assert result["securities"][0]["bonus"] == "0"

    def test_bonus_eligible_count(self):
        """Metrics track bonus eligible count."""
        holdings = _make_holdings({
            "AAAA": (100_000, 80_000),
            "BBBB": (200_000, 150_000),
        })
        ranked = _make_ranked([("AAAA", 1), ("BBBB", 2)])
        result = compute_baker_overlay(ranked, holdings)
        assert result["metrics"]["baker_bonus_eligible_count"] == 2


# =============================================================================
# Risk gate helpers
# =============================================================================

class TestRiskGates:
    def test_risk_gates_pass_clean(self):
        assert _risk_gates_pass({"severity": "none", "flags": []}) is True

    def test_risk_gates_fail_sev2(self):
        assert _risk_gates_pass({"severity": "sev2", "flags": []}) is False

    def test_risk_gates_fail_sev3(self):
        assert _risk_gates_pass({"severity": "sev3", "flags": []}) is False

    def test_risk_gates_fail_survivability(self):
        assert _risk_gates_pass({
            "severity": "none",
            "flags": ["guardrail_a_survivability_critical"],
        }) is False

    def test_risk_gates_fail_runway(self):
        assert _risk_gates_pass({
            "severity": "none",
            "flags": ["guardrail_a_cash_runway_lt_6m"],
        }) is False

    def test_guardrail_blocks_bonus_sev(self):
        assert _guardrail_blocks_bonus({"flags": ["guardrail_a_sev_sev2"]}) is True

    def test_guardrail_blocks_bonus_red_flag(self):
        assert _guardrail_blocks_bonus({
            "flags": ["guardrail_a_red_flag_eligible:dilution_high"],
        }) is True

    def test_guardrail_does_not_block_clean(self):
        assert _guardrail_blocks_bonus({"flags": []}) is False

    def test_guardrail_sev1_does_not_block_bonus(self):
        """sev1_reduced_cap does NOT block bonus (only sev2+ and red-flag do)."""
        assert _guardrail_blocks_bonus({"flags": ["guardrail_a_sev1_reduced_cap"]}) is False


# =============================================================================
# Disagreement reasons
# =============================================================================

class TestDisagreementReasons:
    def test_severity_sev2(self):
        reasons = _extract_disagreement_reasons({"severity": "sev2", "flags": []})
        assert "severity_sev2" in reasons

    def test_severity_sev1(self):
        reasons = _extract_disagreement_reasons({"severity": "sev1", "flags": []})
        assert "severity_sev1" in reasons

    def test_thesis_gate(self):
        reasons = _extract_disagreement_reasons({
            "severity": "none",
            "flags": ["thesis_gate_penalty_0.80"],
        })
        assert "thesis_gate" in reasons

    def test_sm_reinforcement_blocked(self):
        reasons = _extract_disagreement_reasons({
            "severity": "none",
            "flags": ["sm_reinforcement_blocked_by_thesis_gate"],
        })
        assert "sm_reinforcement_blocked" in reasons

    def test_low_components(self):
        reasons = _extract_disagreement_reasons({
            "severity": "none",
            "flags": [],
            "component_scores": [
                {"name": "financial", "normalized": "22.5", "raw": "69"},
                {"name": "clinical", "normalized": "60", "raw": "70"},
            ],
        })
        assert any("low_financial=22" in r for r in reasons)
        assert not any("low_clinical" in r for r in reasons)

    def test_negative_momentum(self):
        reasons = _extract_disagreement_reasons({
            "severity": "none",
            "flags": [],
            "score_breakdown": {
                "enhancements": {"momentum": {"score": "28.5"}},
            },
        })
        assert any("neg_momentum=28" in r for r in reasons)

    def test_low_valuation(self):
        reasons = _extract_disagreement_reasons({
            "severity": "none",
            "flags": [],
            "score_breakdown": {
                "enhancements": {"valuation": {"score": "15.2"}},
            },
        })
        assert any("low_valuation=15" in r for r in reasons)

    def test_clean_security_no_reasons(self):
        reasons = _extract_disagreement_reasons({"severity": "none", "flags": []})
        assert reasons == []

    def test_survivability_warning(self):
        reasons = _extract_disagreement_reasons({
            "severity": "none",
            "flags": ["survivability_warning"],
        })
        assert "survivability_warning" in reasons

    def test_late_stage_distress(self):
        reasons = _extract_disagreement_reasons({
            "severity": "none",
            "flags": ["late_stage_distress"],
        })
        assert "late_stage_distress" in reasons


# =============================================================================
# Diagnostics
# =============================================================================

# =============================================================================
# Schema contract — prevent KeyError regressions
# =============================================================================

class TestSchemaContract:
    """Ensure queue entries always contain the required key set."""

    QUEUE_1_REQUIRED = {
        "ticker", "baker_activity", "baker_portfolio_weight_pct",
        "baker_value_usd", "baker_rank_by_weight", "composite_rank",
        "severity", "queue_reasons", "disagreement_reasons",
    }
    QUEUE_2_REQUIRED = {
        "ticker", "baker_activity", "baker_portfolio_weight_pct",
        "baker_rank_by_weight", "composite_rank", "severity",
        "days_to_catalyst",
    }

    def test_queue_1_entry_keys(self):
        """Every Queue 1 entry must contain all required fields."""
        holdings = _make_holdings({"AAAA": (500_000_000, 0)})  # >$250M, new
        ranked = _make_ranked([("AAAA", 200)])  # rank > 75
        result = compute_baker_overlay(ranked, holdings)
        q1 = result["queue_1_core_divergence"]
        assert len(q1) >= 1, "Expected at least one Queue 1 entry"
        for entry in q1:
            missing = self.QUEUE_1_REQUIRED - set(entry.keys())
            assert not missing, f"{entry['ticker']} missing keys: {missing}"

    def test_queue_2_entry_keys(self):
        """Every Queue 2 entry must contain all required fields."""
        holdings = _make_holdings({"AAAA": (100_000, 0)})
        ranked = _make_ranked([("AAAA", 80)])  # rank > Q2_RANK (50)
        ranked[0]["catalyst_decay"] = {"days_to_catalyst": 30}
        result = compute_baker_overlay(ranked, holdings)
        q2 = result["queue_2_near_term_catalyst"]
        assert len(q2) >= 1, "Expected at least one Queue 2 entry"
        for entry in q2:
            missing = self.QUEUE_2_REQUIRED - set(entry.keys())
            assert not missing, f"{entry['ticker']} missing keys: {missing}"


class TestDiagnostics:
    def test_diagnostics_present(self):
        holdings = _make_holdings({"AAAA": (100_000, 0)})
        ranked = _make_ranked([("AAAA", 1)])
        result = compute_baker_overlay(ranked, holdings)
        diag = result["diagnostics"]
        assert diag["baker_cik"] == BAKER_CIK
        assert diag["version"] == "1.1.0"
        assert "queue_1_thresholds" in diag
        assert "queue_2_thresholds" in diag
        assert diag["baker_activity_counts"]["new"] == 1
        # Check weight floor thresholds are in diagnostics
        assert diag["queue_1_thresholds"]["value_min_weight_pct"] == 0.25
        assert diag["queue_1_thresholds"]["activity_min_weight_pct"] == 0.25
