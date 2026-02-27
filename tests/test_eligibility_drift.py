"""
Unit tests for eligibility drift scripts:
  - build_eligibility_delta.py  (normalize + delta computation)
  - eval_eligibility_gate.py    (gate evaluation logic)
  - triage_eligibility_drift.py (mover detection)

Run: pytest tests/test_eligibility_drift.py -v
"""
import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from scripts.build_eligibility_delta import (
    NormSummary,
    normalize_summary,
    build_delta,
    render_delta_md,
    list_snapshot_dates,
    pick_prior_date,
)
from scripts.eval_eligibility_gate import evaluate_eligibility_gate
from scripts.triage_eligibility_drift import (
    find_movers,
    render_triage_md,
    _split_reasons,
    _safe_bool01,
)


# =============================================================================
# normalize_summary
# =============================================================================


class TestNormalizeSummary:
    """Tests for eligibility summary normalisation."""

    def test_canonical_v1_keys(self):
        """Standard eligibility_summary.v1 keys are parsed correctly."""
        raw = {
            "schema": "eligibility_summary.v1",
            "n_total": 300,
            "n_eligible": 250,
            "n_ineligible": 50,
            "counts_by_reason": {"deep_drawdown": 30, "adv_fail": 20},
        }
        ns = normalize_summary(raw, "2026-02-27")
        assert ns.total == 300
        assert ns.eligible == 250
        assert ns.ineligible == 50
        assert ns.reasons == {"deep_drawdown": 30, "adv_fail": 20}
        assert abs(ns.pct_ineligible - 50 / 300) < 1e-6

    def test_alternative_keys(self):
        """Falls back to 'total', 'eligible', 'ineligible', 'reason_counts'."""
        raw = {
            "total": 200,
            "eligible": 180,
            "ineligible": 20,
            "reason_counts": {"sev3": 5},
        }
        ns = normalize_summary(raw, "2026-01-01")
        assert ns.total == 200
        assert ns.eligible == 180
        assert ns.reasons == {"sev3": 5}

    def test_empty_raw(self):
        ns = normalize_summary({}, "2026-01-01")
        assert ns.total == 0
        assert ns.eligible == 0
        assert ns.ineligible == 0
        assert ns.pct_ineligible == 0.0
        assert ns.reasons == {}

    def test_total_inferred_from_parts(self):
        """If total missing, sum eligible + ineligible."""
        raw = {"n_eligible": 100, "n_ineligible": 25}
        ns = normalize_summary(raw, "2026-01-01")
        assert ns.total == 125

    def test_to_dict_deterministic(self):
        raw = {
            "n_total": 10,
            "n_eligible": 8,
            "n_ineligible": 2,
            "counts_by_reason": {"z_reason": 1, "a_reason": 1},
        }
        ns = normalize_summary(raw, "2026-01-01")
        d = ns.to_dict()
        # Reasons should be alphabetically sorted in dict
        keys = list(d["reasons"].keys())
        assert keys == sorted(keys)


# =============================================================================
# build_delta
# =============================================================================


class TestBuildDelta:
    """Tests for delta computation."""

    def test_basic_delta(self):
        cur = {"n_total": 300, "n_eligible": 250, "n_ineligible": 50,
               "counts_by_reason": {"deep_drawdown": 30, "adv_fail": 20}}
        pri = {"n_total": 300, "n_eligible": 260, "n_ineligible": 40,
               "counts_by_reason": {"deep_drawdown": 25, "adv_fail": 15}}
        out = build_delta(cur, pri, "2026-02-27", "2026-02-26")
        assert out["schema"] == "eligibility_delta.v1"
        assert out["delta"]["eligible"] == -10
        assert out["delta"]["ineligible"] == 10
        assert out["delta"]["reasons"]["deep_drawdown"] == 5
        assert out["delta"]["reasons"]["adv_fail"] == 5

    def test_new_reason_appears(self):
        cur = {"n_total": 100, "n_eligible": 90, "n_ineligible": 10,
               "counts_by_reason": {"sev3": 10}}
        pri = {"n_total": 100, "n_eligible": 100, "n_ineligible": 0,
               "counts_by_reason": {}}
        out = build_delta(cur, pri, "2026-02-27", "2026-02-26")
        assert out["delta"]["reasons"]["sev3"] == 10

    def test_reason_disappears(self):
        cur = {"n_total": 100, "n_eligible": 100, "n_ineligible": 0,
               "counts_by_reason": {}}
        pri = {"n_total": 100, "n_eligible": 95, "n_ineligible": 5,
               "counts_by_reason": {"adv_fail": 5}}
        out = build_delta(cur, pri, "2026-02-27", "2026-02-26")
        assert out["delta"]["reasons"]["adv_fail"] == -5

    def test_total_change_noted(self):
        cur = {"n_total": 310, "n_eligible": 260, "n_ineligible": 50,
               "counts_by_reason": {}}
        pri = {"n_total": 300, "n_eligible": 260, "n_ineligible": 40,
               "counts_by_reason": {}}
        out = build_delta(cur, pri, "2026-02-27", "2026-02-26")
        assert any("total changed" in n for n in out["notes"])

    def test_md_rendering(self):
        cur = {"n_total": 100, "n_eligible": 90, "n_ineligible": 10,
               "counts_by_reason": {"deep_drawdown": 10}}
        pri = {"n_total": 100, "n_eligible": 95, "n_ineligible": 5,
               "counts_by_reason": {"deep_drawdown": 5}}
        out = build_delta(cur, pri, "2026-02-27", "2026-02-26")
        md = render_delta_md(out)
        assert "Eligibility Delta" in md
        assert "deep_drawdown" in md
        assert "+5" in md


# =============================================================================
# evaluate_eligibility_gate
# =============================================================================


class TestEvaluateGate:
    """Tests for gate evaluation logic."""

    def _make_delta(self, pct=0.0, inel=0, reasons=None):
        return {
            "delta": {
                "pct_ineligible": pct,
                "ineligible": inel,
                "eligible": -inel,
                "reasons": reasons or {},
            },
        }

    def test_pass_no_drift(self):
        verdict, triggers = evaluate_eligibility_gate(self._make_delta())
        assert verdict == "PASS"

    def test_warn_pct_threshold(self):
        verdict, _ = evaluate_eligibility_gate(
            self._make_delta(pct=0.04), warn_pct=0.03, fail_pct=0.06,
        )
        assert verdict == "WARN"

    def test_fail_pct_threshold(self):
        verdict, _ = evaluate_eligibility_gate(
            self._make_delta(pct=0.07), warn_pct=0.03, fail_pct=0.06,
        )
        assert verdict == "FAIL"

    def test_warn_abs_threshold(self):
        verdict, _ = evaluate_eligibility_gate(
            self._make_delta(inel=12), warn_abs=10, fail_abs=25,
        )
        assert verdict == "WARN"

    def test_fail_abs_threshold(self):
        verdict, _ = evaluate_eligibility_gate(
            self._make_delta(inel=30), warn_abs=10, fail_abs=25,
        )
        assert verdict == "FAIL"

    def test_negative_delta_also_triggers(self):
        """Large drop in ineligible is also suspicious."""
        verdict, _ = evaluate_eligibility_gate(
            self._make_delta(inel=-15), warn_abs=10, fail_abs=25,
        )
        assert verdict == "WARN"

    def test_triggers_include_reason_deltas(self):
        delta = self._make_delta(reasons={"deep_drawdown": 8, "sev3": 2})
        _, triggers = evaluate_eligibility_gate(delta)
        reason_trig = [t for t in triggers if "top_reason_deltas" in t]
        assert len(reason_trig) == 1
        assert "deep_drawdown" in reason_trig[0]

    def test_combined_pct_and_abs(self):
        """Both pct and abs can fire simultaneously."""
        verdict, triggers = evaluate_eligibility_gate(
            self._make_delta(pct=0.08, inel=30),
            warn_pct=0.03, fail_pct=0.06, warn_abs=10, fail_abs=25,
        )
        assert verdict == "FAIL"
        assert len([t for t in triggers if t.startswith("pct_ineligible_delta")]) == 1
        assert len([t for t in triggers if t.startswith("ineligible_delta")]) == 1


# =============================================================================
# triage: helpers
# =============================================================================


class TestTriageHelpers:
    """Tests for triage helper functions."""

    def test_split_reasons_pipe(self):
        assert _split_reasons("deep_drawdown|adv_fail") == ["adv_fail", "deep_drawdown"]

    def test_split_reasons_empty(self):
        assert _split_reasons("") == []
        assert _split_reasons(None) == []

    def test_split_reasons_dedup(self):
        assert _split_reasons("sev3|sev3|adv_fail") == ["adv_fail", "sev3"]

    def test_safe_bool01(self):
        assert _safe_bool01("1") == 1
        assert _safe_bool01("0") == 0
        assert _safe_bool01("") is None
        assert _safe_bool01("True") == 1


# =============================================================================
# triage: find_movers
# =============================================================================


class TestFindMovers:
    """Tests for mover detection."""

    def test_no_changes(self):
        cur = {"AAA": {"eligible": "1", "ineligible_reasons": ""}}
        pri = {"AAA": {"eligible": "1", "ineligible_reasons": ""}}
        movers = find_movers(cur, pri)
        assert len(movers) == 0

    def test_newly_ineligible(self):
        cur = {"AAA": {"eligible": "0", "ineligible_reasons": "deep_drawdown"}}
        pri = {"AAA": {"eligible": "1", "ineligible_reasons": ""}}
        movers = find_movers(cur, pri)
        assert len(movers) == 1
        assert movers[0].ticker == "AAA"
        assert movers[0].prior_eligible == 1
        assert movers[0].cur_eligible == 0

    def test_newly_eligible(self):
        cur = {"BBB": {"eligible": "1", "ineligible_reasons": ""}}
        pri = {"BBB": {"eligible": "0", "ineligible_reasons": "sev3"}}
        movers = find_movers(cur, pri)
        assert len(movers) == 1
        assert movers[0].cur_eligible == 1

    def test_reason_only_excluded_by_default(self):
        cur = {"CCC": {"eligible": "0", "ineligible_reasons": "deep_drawdown|sev3"}}
        pri = {"CCC": {"eligible": "0", "ineligible_reasons": "deep_drawdown"}}
        movers = find_movers(cur, pri, include_reason_only=False)
        assert len(movers) == 0

    def test_reason_only_included_when_flag_set(self):
        cur = {"CCC": {"eligible": "0", "ineligible_reasons": "deep_drawdown|sev3"}}
        pri = {"CCC": {"eligible": "0", "ineligible_reasons": "deep_drawdown"}}
        movers = find_movers(cur, pri, include_reason_only=True)
        assert len(movers) == 1

    def test_ordering_newly_ineligible_first(self):
        cur = {
            "AAA": {"eligible": "1", "ineligible_reasons": ""},
            "BBB": {"eligible": "0", "ineligible_reasons": "sev3"},
        }
        pri = {
            "AAA": {"eligible": "0", "ineligible_reasons": "adv_fail"},
            "BBB": {"eligible": "1", "ineligible_reasons": ""},
        }
        movers = find_movers(cur, pri)
        assert movers[0].ticker == "BBB"  # newly ineligible first
        assert movers[1].ticker == "AAA"  # newly eligible second

    def test_max_tickers_cap(self):
        cur = {f"T{i:03d}": {"eligible": "0", "ineligible_reasons": "sev3"} for i in range(20)}
        pri = {f"T{i:03d}": {"eligible": "1", "ineligible_reasons": ""} for i in range(20)}
        movers = find_movers(cur, pri, max_tickers=5)
        assert len(movers) == 5

    def test_new_ticker_in_current(self):
        """Ticker only in current (not in prior) — no flip possible (prior_eligible=None)."""
        cur = {"NEW": {"eligible": "1", "ineligible_reasons": ""}}
        pri = {}
        movers = find_movers(cur, pri)
        assert len(movers) == 0  # prior_eligible is None, no flip

    def test_md_rendering(self):
        cur = {"AAA": {"eligible": "0", "ineligible_reasons": "deep_drawdown",
                        "archetype": "drug_developer", "tier_any": "B"}}
        pri = {"AAA": {"eligible": "1", "ineligible_reasons": ""}}
        movers = find_movers(cur, pri)
        md = render_triage_md({"as_of_date": "2026-02-27", "prior_date": "2026-02-26"}, movers)
        assert "Newly ineligible" in md
        assert "AAA" in md
        assert "deep_drawdown" in md


# =============================================================================
# Snapshot date discovery (build_eligibility_delta)
# =============================================================================


class TestSnapshotDiscovery:
    """Tests for snapshot date listing and prior selection."""

    def test_list_snapshot_dates(self, tmp_path):
        (tmp_path / "2026-02-25").mkdir()
        (tmp_path / "2026-02-26").mkdir()
        (tmp_path / "2026-02-27").mkdir()
        (tmp_path / "not-a-date").mkdir()
        dates = list_snapshot_dates(tmp_path)
        assert dates == ["2026-02-25", "2026-02-26", "2026-02-27"]

    def test_pick_prior_date(self, tmp_path):
        for d in ["2026-02-25", "2026-02-26", "2026-02-27"]:
            sd = tmp_path / d
            sd.mkdir()
            (sd / "eligibility_summary.json").write_text("{}")
        prior = pick_prior_date(tmp_path, "2026-02-27")
        assert prior == "2026-02-26"

    def test_pick_prior_skips_degraded(self, tmp_path):
        for d in ["2026-02-25", "2026-02-26", "2026-02-27"]:
            sd = tmp_path / d
            sd.mkdir()
            (sd / "eligibility_summary.json").write_text("{}")
        # Mark 2026-02-26 as degraded
        (tmp_path / "2026-02-26" / "cache_health.json").write_text(
            json.dumps({"degraded_run": True})
        )
        prior = pick_prior_date(tmp_path, "2026-02-27", include_degraded=False)
        assert prior == "2026-02-25"

    def test_pick_prior_none_when_first(self, tmp_path):
        sd = tmp_path / "2026-02-27"
        sd.mkdir()
        (sd / "eligibility_summary.json").write_text("{}")
        prior = pick_prior_date(tmp_path, "2026-02-27")
        assert prior is None
