#!/usr/bin/env python3
"""
Unit tests for run_gate_ablation.py

Tests gate ablation logic, promoted-name detection, frequency counting,
inert gate detection, and percentile computation.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from decision_engine import DecisionRuleset, compute_decision_fields
from run_gate_ablation import (
    KNOWN_GATES,
    _patch_rec_for_scenario,
    _percentile,
    _safe_div,
    count_gate_frequencies,
    find_promoted_names,
    _compute_tier_distribution,
    _make_scenario_ruleset,
)
from run_decision_ruleset_sweep import ArchiveData, ForwardReturnData


# =============================================================================
# FIXTURES: minimal rec builders
# =============================================================================

def _make_rec(
    severity: str = "",
    drawdown: Optional[float] = None,
    fundamental_red_flag: bool = False,
) -> Dict[str, Any]:
    """Build a minimal rec dict for testing."""
    rec: Dict[str, Any] = {
        "severity": severity,
        "confidence_overall": 0.5,
        "fundamental_red_flag": fundamental_red_flag,
        "catalyst_decay": {"days_to_catalyst": 30, "in_optimal_window": True},
        "smart_money_signal": {},
        "coinvest": {"tier1_count": 2},
        "defensive_features": {},
        "score_breakdown": {},
        "momentum_signal": {},
    }
    if drawdown is not None:
        rec["defensive_features"]["drawdown"] = drawdown
    return rec


def _make_archive_data(
    tickers_data: Dict[str, Dict[str, Any]],
    date_str: str = "2024-06-30",
) -> ArchiveData:
    """Build a minimal ArchiveData for testing.

    tickers_data: {ticker: {"rec": rec_dict, "archetype": str, "opt": float|None}}
    """
    tickers = list(tickers_data.keys())
    recs = {t: d["rec"] for t, d in tickers_data.items()}
    archetypes = {t: d.get("archetype", "drug_developer") for t, d in tickers_data.items()}
    optionalities = {t: d.get("opt", 0.70) for t, d in tickers_data.items()}
    rows = [{"ticker": t, "archetype": archetypes[t]} for t in tickers]

    return ArchiveData(
        date_str=date_str,
        rows=rows,
        recs=recs,
        archetypes=archetypes,
        optionalities=optionalities,
        tickers=tickers,
    )


def _make_forward_data(
    tickers: Dict[str, float],
) -> ForwardReturnData:
    """Build minimal ForwardReturnData. tickers = {ticker: raw_return}."""
    return ForwardReturnData(
        raw_returns=tickers,
        residual_returns=tickers,  # simplified: resid = raw
        xbi_return=0.0,
        betas={t: 0.0 for t in tickers},
    )


# =============================================================================
# TEST: drawdown ablation disables gate
# =============================================================================

class TestDrawdownAblation:
    def test_drawdown_gate_fires_at_default(self):
        """A ticker with -50% drawdown should be ineligible under default ruleset."""
        rec = _make_rec(drawdown=-0.50)
        fields = compute_decision_fields(
            rec, "drug_developer", 0.70, ruleset=DecisionRuleset()
        )
        assert fields["eligible"] == "0"
        assert "deep_drawdown" in fields["ineligible_reasons"]
        assert fields["tier_dev"] == "D"

    def test_drawdown_ablation_disables_gate(self):
        """Setting drawdown_gate=-999.0 should prevent the gate from firing."""
        rec = _make_rec(drawdown=-0.50)
        ablated = DecisionRuleset(drawdown_gate=-999.0)
        fields = compute_decision_fields(
            rec, "drug_developer", 0.70, ruleset=ablated
        )
        assert fields["eligible"] == "1"
        assert "deep_drawdown" not in fields["ineligible_reasons"]
        assert fields["tier_dev"] != "D"

    def test_drawdown_at_threshold_is_eligible(self):
        """A ticker exactly at -0.40 should be eligible (dd < gate, not dd <= gate)."""
        rec = _make_rec(drawdown=-0.40)
        fields = compute_decision_fields(
            rec, "drug_developer", 0.70, ruleset=DecisionRuleset()
        )
        assert fields["eligible"] == "1"

    def test_drawdown_just_below_threshold_ineligible(self):
        """A ticker at -0.41 should be ineligible (below -0.40 gate)."""
        rec = _make_rec(drawdown=-0.41)
        fields = compute_decision_fields(
            rec, "drug_developer", 0.70, ruleset=DecisionRuleset()
        )
        assert fields["eligible"] == "0"
        assert "deep_drawdown" in fields["ineligible_reasons"]


# =============================================================================
# TEST: SEV3 ablation disables gate
# =============================================================================

class TestSev3Ablation:
    def test_sev3_gate_fires(self):
        """A ticker with severity=SEV3 should be ineligible."""
        rec = _make_rec(severity="SEV3")
        fields = compute_decision_fields(
            rec, "drug_developer", 0.70, ruleset=DecisionRuleset()
        )
        assert fields["eligible"] == "0"
        assert "sev3" in fields["ineligible_reasons"]

    def test_sev3_ablation_disables_gate(self):
        """Patching severity="" should prevent SEV3 gate from firing."""
        rec = _make_rec(severity="SEV3")
        patched = _patch_rec_for_scenario(rec, "no_sev3")
        fields = compute_decision_fields(
            patched, "drug_developer", 0.70, ruleset=DecisionRuleset()
        )
        assert fields["eligible"] == "1"
        assert "sev3" not in fields["ineligible_reasons"]

    def test_sev3_patch_does_not_affect_other_gates(self):
        """Patching for sev3 should not disable drawdown gate."""
        rec = _make_rec(severity="SEV3", drawdown=-0.50)
        patched = _patch_rec_for_scenario(rec, "no_sev3")
        fields = compute_decision_fields(
            patched, "drug_developer", 0.70, ruleset=DecisionRuleset()
        )
        # SEV3 gone but drawdown still fires
        assert "sev3" not in fields["ineligible_reasons"]
        assert "deep_drawdown" in fields["ineligible_reasons"]
        assert fields["eligible"] == "0"


# =============================================================================
# TEST: promoted names detection
# =============================================================================

class TestPromotedNames:
    def test_promoted_names_detection(self):
        """Tickers that move from D to non-D should be detected as promoted."""
        baseline_tiers = {"AAA": "D", "BBB": "C", "CCC": "D", "DDD": "A"}
        ablated_tiers = {"AAA": "C", "BBB": "C", "CCC": "D", "DDD": "A"}
        fd = _make_forward_data({"AAA": 0.10, "BBB": 0.05, "CCC": -0.05, "DDD": 0.20})

        promoted = find_promoted_names(baseline_tiers, ablated_tiers, fd, "2024-06-30")

        assert len(promoted) == 1
        assert promoted[0]["ticker"] == "AAA"
        assert promoted[0]["old_tier"] == "D"
        assert promoted[0]["new_tier"] == "C"
        assert promoted[0]["raw_return_pct"] == 10.0

    def test_no_promotions_when_all_stay_d(self):
        """No promotions if ablated tiers are the same."""
        baseline_tiers = {"AAA": "D", "BBB": "D"}
        ablated_tiers = {"AAA": "D", "BBB": "D"}
        fd = _make_forward_data({"AAA": 0.10, "BBB": -0.05})

        promoted = find_promoted_names(baseline_tiers, ablated_tiers, fd, "2024-06-30")
        assert len(promoted) == 0

    def test_promoted_tracks_new_tier_correctly(self):
        """Promoted ticker's new tier should match ablated assignment."""
        baseline_tiers = {"AAA": "D"}
        ablated_tiers = {"AAA": "B"}
        fd = _make_forward_data({"AAA": 0.15})

        promoted = find_promoted_names(baseline_tiers, ablated_tiers, fd, "2024-01-31")
        assert len(promoted) == 1
        assert promoted[0]["new_tier"] == "B"


# =============================================================================
# TEST: gate frequency counting
# =============================================================================

class TestGateFrequency:
    def test_gate_frequency_counting(self):
        """Count gates from synthetic recs with known ineligibility."""
        tickers_data = {
            "T1": {"rec": _make_rec(drawdown=-0.50), "opt": 0.70},  # deep_drawdown
            "T2": {"rec": _make_rec(severity="SEV3"), "opt": 0.70},  # sev3
            "T3": {"rec": _make_rec(), "opt": 0.70},                 # eligible
            "T4": {"rec": _make_rec(drawdown=-0.60, severity="SEV3"), "opt": 0.70},  # both
        }
        ad = _make_archive_data(tickers_data, "2024-06-30")
        cache = {"2024-06-30": ad}

        result = count_gate_frequencies(cache, DecisionRuleset())

        pooled = result["pooled"]
        assert pooled["n_dev"] == 4
        assert pooled["n_ineligible"] == 3

        # deep_drawdown: T1 + T4 = 2
        assert pooled["gates"]["deep_drawdown"]["n"] == 2
        # sev3: T2 + T4 = 2
        assert pooled["gates"]["sev3"]["n"] == 2
        # fundamental_red_flag: 0
        assert pooled["gates"]["fundamental_red_flag"]["n"] == 0

    def test_inert_gate_detection(self):
        """Gates with zero hits should be identified as inert."""
        tickers_data = {
            "T1": {"rec": _make_rec(drawdown=-0.50), "opt": 0.70},
            "T2": {"rec": _make_rec(), "opt": 0.70},
        }
        ad = _make_archive_data(tickers_data, "2024-06-30")
        cache = {"2024-06-30": ad}

        result = count_gate_frequencies(cache, DecisionRuleset())

        # sev3, fundamental_red_flag, and adv_fail should be inert
        assert "fundamental_red_flag" in result["inert_gates"]
        assert "adv_fail" in result["inert_gates"]
        assert "sev3" in result["inert_gates"]
        assert "deep_drawdown" not in result["inert_gates"]

    def test_overlap_counting(self):
        """Both gates firing on same ticker should appear in overlap."""
        tickers_data = {
            "T1": {"rec": _make_rec(drawdown=-0.50, severity="SEV3"), "opt": 0.70},
        }
        ad = _make_archive_data(tickers_data, "2024-06-30")
        cache = {"2024-06-30": ad}

        result = count_gate_frequencies(cache, DecisionRuleset())
        overlap = result["pooled"]["overlap"]
        assert any("deep_drawdown" in k and "sev3" in k for k in overlap)


# =============================================================================
# TEST: percentile computation
# =============================================================================

class TestPercentile:
    def test_percentile_p5_known(self):
        """P5 of [0..99] should be ~4.95."""
        vals = list(range(100))
        assert abs(_percentile(vals, 5) - 4.95) < 0.01

    def test_percentile_p10_known(self):
        """P10 of [0..99] should be ~9.9."""
        vals = list(range(100))
        assert abs(_percentile(vals, 10) - 9.9) < 0.01

    def test_percentile_p50_is_median(self):
        """P50 should be close to median for odd-length list."""
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile(vals, 50) == 3.0

    def test_percentile_empty(self):
        """Empty list returns 0.0."""
        assert _percentile([], 50) == 0.0

    def test_percentile_single_value(self):
        """Single value returns that value for any percentile."""
        assert _percentile([42.0], 5) == 42.0
        assert _percentile([42.0], 95) == 42.0


# =============================================================================
# TEST: scenario ruleset construction
# =============================================================================

class TestScenarioRuleset:
    def test_baseline_uses_default(self):
        rs = _make_scenario_ruleset("baseline")
        assert rs.drawdown_gate == -0.40

    def test_no_drawdown_disables_gate(self):
        rs = _make_scenario_ruleset("no_drawdown")
        assert rs.drawdown_gate == -999.0

    def test_no_gates_disables_drawdown(self):
        rs = _make_scenario_ruleset("no_gates")
        assert rs.drawdown_gate == -999.0


# =============================================================================
# TEST: tier distribution helper
# =============================================================================

class TestTierDistribution:
    def test_basic_distribution(self):
        tiers = {"T1": "A", "T2": "B", "T3": "C", "T4": "D", "T5": "C"}
        dist = _compute_tier_distribution(tiers)
        assert dist["A"]["n"] == 1
        assert dist["B"]["n"] == 1
        assert dist["C"]["n"] == 2
        assert dist["D"]["n"] == 1
        assert dist["A"]["pct"] == 20.0

    def test_all_same_tier(self):
        tiers = {"T1": "C", "T2": "C", "T3": "C"}
        dist = _compute_tier_distribution(tiers)
        assert dist["C"]["pct"] == 100.0
        assert dist["A"]["n"] == 0


# =============================================================================
# TEST: safe_div helper
# =============================================================================

class TestSafeDiv:
    def test_normal_division(self):
        assert _safe_div(10, 5) == 2.0

    def test_zero_denominator(self):
        assert _safe_div(10, 0) == 0.0
