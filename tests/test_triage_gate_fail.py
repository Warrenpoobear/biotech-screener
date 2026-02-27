"""Tests for gate fail triage drilldown (synthetic, no network)."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
import sys

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from decision_engine import DecisionRuleset
from scripts.triage_gate_fail import (
    collect_top_movers,
    find_prior_date,
    reconstruct_movers_from_snapshots,
    select_worst_date,
    triage_gate_fail,
)
from scripts.explain_rank_shift import (
    explain_ticker_shift,
    _feature_deltas,
    _feature_snapshot,
    _module_attribution,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _make_snapshot(snap_dir: Path, date: str, rows: List[Dict]) -> None:
    d = snap_dir / date
    d.mkdir(parents=True, exist_ok=True)
    _write_csv(d / "rankings.csv", rows)


def _rich_rows(tickers: List[str]) -> List[Dict[str, str]]:
    """Create rows with varied module scores that affect rerank ordering."""
    rows = []
    n = len(tickers)
    for i, t in enumerate(tickers):
        rows.append({
            "ticker": t,
            "eligible": "1",
            "archetype": "drug_developer",
            "tier_dev": "A" if i < n // 2 else "B",
            "composite_rank": str(i + 1),
            "clinical_optionality_pct_dev": str(round((n - i) / n, 4)),
            "clinical_score": str(round((n - i) * 10 / n, 2)),
            "catalyst_mode": "specific_days",
            "catalyst_days": str(90 + i * 10),
            "mom_state": "neutral",
            "momentum_score": str(round(0.5 + i * 0.02, 4)),
            "financial_score": str(round(0.3 + i * 0.01, 4)),
            "catalyst_score": str(round(0.6 - i * 0.01, 4)),
            "smart_money_score": str(round(0.4 + i * 0.005, 4)),
            "valuation_score": str(round(0.5, 4)),
            "composite_score": str(round(1.5 + i * 0.05, 4)),
            "coinvest_score_z": str(round(0.1 * i, 4)),
            "alpha_cohort_raw": str(round(0.05 * (n - i), 4)),
            "alpha_cohort_pct": str(round((n - i) / n, 4)),
        })
    return rows


def _make_eval_json(
    dates_evaluated: List[str],
    per_date: List[Dict],
    top_movers: List[Dict],
    candidate_id: str = "abc12345",
    baseline_id: str = "def67890",
    verdict: str = "FAIL",
) -> Dict:
    return {
        "schema": "ruleset_eval.v1",
        "mode": "rerank_only",
        "candidate": {"id": candidate_id, "file": "test.json"},
        "baseline": {"id": baseline_id, "file": "base.json"},
        "config": {
            "start": dates_evaluated[0] if dates_evaluated else "",
            "end": dates_evaluated[-1] if dates_evaluated else "",
            "rerank_only": True,
        },
        "gate": {
            "verdict": verdict,
            "fail_reasons": [],
            "warn_reasons": [],
        },
        "dates_evaluated": dates_evaluated,
        "n_evaluated": len(dates_evaluated),
        "n_skipped": 0,
        "dates_skipped": [],
        "per_date": per_date,
        "temporal_stability": {
            "mean_top60_overlap": 85.0,
            "worst_top60_overlap": 75.0,
            "mean_spearman": 0.95,
            "max_rank_shift": top_movers[0] if top_movers else {},
            "top_movers": top_movers,
        },
    }


def _standard_setup(tmp_path):
    """Create synthetic eval JSON + snapshots.  Returns (snap_dir, eval_json)."""
    snap_dir = tmp_path / "snaps"
    tickers = [f"T{i:02d}" for i in range(1, 11)]
    _make_snapshot(snap_dir, "2026-02-10", _rich_rows(tickers))
    tickers_d2 = tickers[:7] + list(reversed(tickers[7:]))
    _make_snapshot(snap_dir, "2026-02-11", _rich_rows(tickers_d2))
    _make_snapshot(snap_dir, "2026-02-12", _rich_rows(tickers))

    dates = ["2026-02-10", "2026-02-11", "2026-02-12"]
    per_date = [
        {"date": "2026-02-10", "n_eligible": 10},
        {
            "date": "2026-02-11",
            "n_eligible": 10,
            "temporal_max_shift": 15,
            "temporal_max_shift_ticker": "T09",
            "temporal_overlap_60": 80.0,
            "temporal_spearman": 0.92,
        },
        {
            "date": "2026-02-12",
            "n_eligible": 10,
            "temporal_max_shift": 8,
            "temporal_max_shift_ticker": "T08",
            "temporal_overlap_60": 90.0,
            "temporal_spearman": 0.96,
        },
    ]
    top_movers = [
        {"ticker": "T09", "shift": 15, "from": 2, "to": 17, "date": "2026-02-11"},
        {"ticker": "T10", "shift": 12, "from": 3, "to": 15, "date": "2026-02-11"},
        {"ticker": "T08", "shift": 8, "from": 5, "to": 13, "date": "2026-02-12"},
    ]
    eval_data = _make_eval_json(dates, per_date, top_movers)
    eval_json = tmp_path / "eval" / "ruleset_eval.json"
    _write_json(eval_json, eval_data)
    return snap_dir, eval_json


# ---------------------------------------------------------------------------
# Tests: select_worst_date
# ---------------------------------------------------------------------------


class TestSelectWorstDate:
    def test_picks_highest_shift(self):
        per_date = [
            {"date": "2026-02-10", "temporal_max_shift": 5, "temporal_overlap_60": 95.0},
            {"date": "2026-02-11", "temporal_max_shift": 20, "temporal_overlap_60": 80.0},
            {"date": "2026-02-12", "temporal_max_shift": 10, "temporal_overlap_60": 90.0},
        ]
        assert select_worst_date(per_date)["date"] == "2026-02-11"

    def test_tiebreaks_by_lowest_overlap(self):
        per_date = [
            {"date": "2026-02-10", "temporal_max_shift": 20, "temporal_overlap_60": 90.0},
            {"date": "2026-02-11", "temporal_max_shift": 20, "temporal_overlap_60": 80.0},
        ]
        assert select_worst_date(per_date)["date"] == "2026-02-11"

    def test_tiebreaks_by_latest_date(self):
        per_date = [
            {"date": "2026-02-10", "temporal_max_shift": 20, "temporal_overlap_60": 80.0},
            {"date": "2026-02-11", "temporal_max_shift": 20, "temporal_overlap_60": 80.0},
        ]
        assert select_worst_date(per_date)["date"] == "2026-02-11"

    def test_empty_returns_none(self):
        assert select_worst_date([]) is None

    def test_skips_first_date_without_temporal(self):
        per_date = [
            {"date": "2026-02-10", "n_eligible": 100},
            {"date": "2026-02-11", "temporal_max_shift": 10, "temporal_overlap_60": 90.0},
        ]
        assert select_worst_date(per_date)["date"] == "2026-02-11"

    def test_fallback_to_last_date_when_no_temporal(self):
        per_date = [
            {"date": "2026-02-10", "n_eligible": 100},
            {"date": "2026-02-11", "n_eligible": 100},
        ]
        assert select_worst_date(per_date)["date"] == "2026-02-11"


# ---------------------------------------------------------------------------
# Tests: find_prior_date
# ---------------------------------------------------------------------------


class TestFindPriorDate:
    def test_finds_previous(self):
        dates = ["2026-02-10", "2026-02-11", "2026-02-12"]
        assert find_prior_date("2026-02-12", dates) == "2026-02-11"
        assert find_prior_date("2026-02-11", dates) == "2026-02-10"

    def test_no_prior_for_first(self):
        assert find_prior_date("2026-02-10", ["2026-02-10", "2026-02-11"]) is None

    def test_missing_date_returns_none(self):
        assert find_prior_date("2026-02-15", ["2026-02-10", "2026-02-11"]) is None


# ---------------------------------------------------------------------------
# Tests: collect_top_movers
# ---------------------------------------------------------------------------


class TestCollectTopMovers:
    def test_filters_to_worst_date(self):
        eval_data = {
            "temporal_stability": {
                "top_movers": [
                    {"ticker": "A", "shift": 20, "from": 1, "to": 21, "date": "2026-02-11"},
                    {"ticker": "B", "shift": 15, "from": 5, "to": 20, "date": "2026-02-12"},
                    {"ticker": "C", "shift": 10, "from": 3, "to": 13, "date": "2026-02-11"},
                ],
            },
        }
        movers = collect_top_movers(eval_data, "2026-02-11", max_tickers=10)
        assert len(movers) == 2
        assert movers[0]["ticker"] == "A"
        assert movers[1]["ticker"] == "C"

    def test_caps_at_max_tickers(self):
        eval_data = {
            "temporal_stability": {
                "top_movers": [
                    {"ticker": f"T{i}", "shift": 20 - i, "from": 1, "to": i + 1, "date": "2026-02-11"}
                    for i in range(10)
                ],
            },
        }
        movers = collect_top_movers(eval_data, "2026-02-11", max_tickers=3)
        assert len(movers) == 3

    def test_empty_movers(self):
        assert collect_top_movers({"temporal_stability": {}}, "2026-02-11", 10) == []


# ---------------------------------------------------------------------------
# Tests: explain_ticker_shift
# ---------------------------------------------------------------------------


class TestExplainTickerShift:
    def test_basic_explain(self, tmp_path):
        snap_dir = tmp_path / "snaps"
        tickers = ["A", "B", "C", "D", "E"]
        _make_snapshot(snap_dir, "2026-02-10", _rich_rows(tickers))
        _make_snapshot(snap_dir, "2026-02-11", _rich_rows(list(reversed(tickers))))

        rs = DecisionRuleset()
        result = explain_ticker_shift(
            ticker="A",
            current_date="2026-02-11",
            prior_date="2026-02-10",
            snapshot_dir=snap_dir,
            ruleset=rs,
        )
        assert result["schema"] == "explain_rank_shift.v1"
        assert result["ticker"] == "A"
        assert result["current_rank"] is not None
        assert result["prior_rank"] is not None
        assert "delta_rank" in result
        assert "module_attribution" in result
        assert "feature_deltas" in result

    def test_missing_snapshot(self, tmp_path):
        snap_dir = tmp_path / "snaps"
        snap_dir.mkdir(parents=True)
        rs = DecisionRuleset()
        result = explain_ticker_shift(
            ticker="A",
            current_date="2026-02-11",
            prior_date="2026-02-10",
            snapshot_dir=snap_dir,
            ruleset=rs,
        )
        assert "error" in result

    def test_missing_fields_no_crash(self, tmp_path):
        """Ticker with minimal fields (no alpha, coinvest, etc) still works."""
        snap_dir = tmp_path / "snaps"
        rows = [
            {"ticker": "X", "eligible": "1", "tier_dev": "A"},
            {"ticker": "Y", "eligible": "1", "tier_dev": "B"},
        ]
        _make_snapshot(snap_dir, "2026-02-10", rows)
        _make_snapshot(snap_dir, "2026-02-11", rows)

        rs = DecisionRuleset()
        result = explain_ticker_shift(
            ticker="X",
            current_date="2026-02-11",
            prior_date="2026-02-10",
            snapshot_dir=snap_dir,
            ruleset=rs,
        )
        assert result["ticker"] == "X"
        assert "error" not in result
        assert len(result["module_attribution"]) == 6  # 6 module score columns

    def test_ticker_not_in_snapshot(self, tmp_path):
        """Ticker missing from a snapshot produces notes, not crash."""
        snap_dir = tmp_path / "snaps"
        _make_snapshot(snap_dir, "2026-02-10", _rich_rows(["A", "B"]))
        _make_snapshot(snap_dir, "2026-02-11", _rich_rows(["A", "B"]))

        rs = DecisionRuleset()
        result = explain_ticker_shift(
            ticker="MISSING",
            current_date="2026-02-11",
            prior_date="2026-02-10",
            snapshot_dir=snap_dir,
            ruleset=rs,
        )
        assert result["current_rank"] is None
        assert result["prior_rank"] is None
        assert any("not eligible" in n or "not found" in n for n in result["notes"])

    def test_deterministic_output(self, tmp_path):
        snap_dir = tmp_path / "snaps"
        _make_snapshot(snap_dir, "2026-02-10", _rich_rows(["A", "B", "C"]))
        _make_snapshot(snap_dir, "2026-02-11", _rich_rows(["C", "B", "A"]))

        rs = DecisionRuleset()
        r1 = explain_ticker_shift("A", "2026-02-11", "2026-02-10", snap_dir, rs)
        r2 = explain_ticker_shift("A", "2026-02-11", "2026-02-10", snap_dir, rs)
        assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)

    def test_module_attribution_has_delta(self, tmp_path):
        """Module scores that differ between dates produce delta values."""
        snap_dir = tmp_path / "snaps"
        tickers = ["A", "B", "C"]
        _make_snapshot(snap_dir, "2026-02-10", _rich_rows(tickers))
        _make_snapshot(snap_dir, "2026-02-11", _rich_rows(list(reversed(tickers))))

        rs = DecisionRuleset()
        result = explain_ticker_shift("A", "2026-02-11", "2026-02-10", snap_dir, rs)
        attribs = result["module_attribution"]
        # At least one module should have a delta (scores differ because row position differs)
        deltas = [a for a in attribs if "delta" in a]
        assert len(deltas) >= 1


# ---------------------------------------------------------------------------
# Tests: feature helpers
# ---------------------------------------------------------------------------


class TestFeatureHelpers:
    def test_feature_snapshot_filters_empty(self):
        row = {"ticker": "X", "eligible": "1", "catalyst_days": "90"}
        snap = _feature_snapshot(row)
        assert "catalyst" in snap
        assert snap["catalyst"]["catalyst_days"] == "90"
        # momentum has no values in this row
        assert "momentum" not in snap or not snap.get("momentum")

    def test_feature_deltas_only_changed(self):
        cur = {"catalyst": {"catalyst_days": "90", "catalyst_mode": "specific_days"}}
        pri = {"catalyst": {"catalyst_days": "120", "catalyst_mode": "specific_days"}}
        deltas = _feature_deltas(cur, pri)
        assert "catalyst" in deltas
        assert "catalyst_days" in deltas["catalyst"]
        # catalyst_mode unchanged — should not appear
        assert "catalyst_mode" not in deltas["catalyst"]

    def test_feature_deltas_numeric_delta(self):
        cur = {"alpha": {"alpha_cohort_raw": "0.25"}}
        pri = {"alpha": {"alpha_cohort_raw": "0.15"}}
        deltas = _feature_deltas(cur, pri)
        assert deltas["alpha"]["alpha_cohort_raw"]["delta"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Tests: triage_gate_fail (full orchestration)
# ---------------------------------------------------------------------------


class TestTriageGateFail:
    def test_produces_triage_bundle(self, tmp_path):
        snap_dir, eval_json = _standard_setup(tmp_path)
        out_dir = tmp_path / "triage"
        rs = DecisionRuleset()

        summary = triage_gate_fail(
            eval_json_path=eval_json,
            snapshot_dir=snap_dir,
            out_dir=out_dir,
            max_tickers=5,
            ruleset=rs,
        )

        assert (out_dir / "triage_summary.json").exists()
        assert (out_dir / "triage_summary.md").exists()
        assert summary["schema"] == "triage.v1"
        assert summary["worst_date"] == "2026-02-11"
        assert len(summary["top_movers"]) >= 1

    def test_explain_files_written(self, tmp_path):
        snap_dir, eval_json = _standard_setup(tmp_path)
        out_dir = tmp_path / "triage"
        rs = DecisionRuleset()

        summary = triage_gate_fail(
            eval_json_path=eval_json,
            snapshot_dir=snap_dir,
            out_dir=out_dir,
            max_tickers=5,
            ruleset=rs,
        )

        explain_dir = out_dir / "explain"
        explain_files = list(explain_dir.glob("explain_*.json"))
        assert len(explain_files) >= 1
        assert len(summary["explain_files"]) == len(explain_files)

        # Verify an explain file is valid JSON with expected schema
        ex = json.loads(explain_files[0].read_text())
        assert ex["schema"] == "explain_rank_shift.v1"
        assert "ticker" in ex
        assert "module_attribution" in ex

    def test_worst_date_deterministic(self, tmp_path):
        snap_dir, eval_json = _standard_setup(tmp_path)
        out1 = tmp_path / "t1"
        out2 = tmp_path / "t2"
        rs = DecisionRuleset()

        s1 = triage_gate_fail(eval_json, snap_dir, out1, ruleset=rs)
        s2 = triage_gate_fail(eval_json, snap_dir, out2, ruleset=rs)
        assert s1["worst_date"] == s2["worst_date"]
        assert s1["top_movers"] == s2["top_movers"]

    def test_no_ruleset_skips_explains(self, tmp_path):
        snap_dir, eval_json = _standard_setup(tmp_path)
        out_dir = tmp_path / "triage"

        summary = triage_gate_fail(
            eval_json_path=eval_json,
            snapshot_dir=snap_dir,
            out_dir=out_dir,
            ruleset=None,
        )
        assert summary["explain_files"] == []
        assert any("no ruleset" in n for n in summary["notes"])

    def test_summary_json_schema_keys(self, tmp_path):
        snap_dir, eval_json = _standard_setup(tmp_path)
        out_dir = tmp_path / "triage"

        summary = triage_gate_fail(eval_json, snap_dir, out_dir)
        required_keys = {
            "schema", "version", "candidate_id", "baseline_id",
            "eval_mode", "verdict", "date_range", "n_evaluated",
            "worst_date", "worst_metric_name", "worst_metric_value",
            "top_movers", "explain_files", "notes",
        }
        assert required_keys.issubset(set(summary.keys()))

    def test_movers_sorted_by_shift_desc(self, tmp_path):
        snap_dir, eval_json = _standard_setup(tmp_path)
        out_dir = tmp_path / "triage"

        summary = triage_gate_fail(eval_json, snap_dir, out_dir)
        movers = summary["top_movers"]
        shifts = [m["shift"] for m in movers]
        assert shifts == sorted(shifts, reverse=True)

    def test_markdown_output_readable(self, tmp_path):
        snap_dir, eval_json = _standard_setup(tmp_path)
        out_dir = tmp_path / "triage"
        rs = DecisionRuleset()

        triage_gate_fail(eval_json, snap_dir, out_dir, ruleset=rs)
        md = (out_dir / "triage_summary.md").read_text(encoding="utf-8")
        assert "# Gate Fail Triage" in md
        assert "Worst Date" in md
        assert "Top Movers" in md

    def test_fallback_movers_from_per_date(self, tmp_path):
        """When top_movers is empty, falls back to per-date shift info."""
        snap_dir = tmp_path / "snaps"
        _make_snapshot(snap_dir, "2026-02-10", _rich_rows(["A", "B"]))
        _make_snapshot(snap_dir, "2026-02-11", _rich_rows(["B", "A"]))

        eval_data = _make_eval_json(
            dates_evaluated=["2026-02-10", "2026-02-11"],
            per_date=[
                {"date": "2026-02-10", "n_eligible": 2},
                {
                    "date": "2026-02-11",
                    "n_eligible": 2,
                    "temporal_max_shift": 5,
                    "temporal_max_shift_ticker": "A",
                    "temporal_max_shift_from": 1,
                    "temporal_max_shift_to": 6,
                    "temporal_overlap_60": 90.0,
                },
            ],
            top_movers=[],  # Empty!
        )
        eval_json = tmp_path / "eval.json"
        _write_json(eval_json, eval_data)

        summary = triage_gate_fail(eval_json, snap_dir, tmp_path / "out")
        assert len(summary["top_movers"]) == 1
        assert summary["top_movers"][0]["ticker"] == "A"

    def test_reconstruct_movers_from_snapshots(self, tmp_path):
        """When eval JSON has NO top_movers AND no per-date shift ticker,
        movers are reconstructed from snapshots."""
        snap_dir = tmp_path / "snaps"
        tickers = [f"T{i:02d}" for i in range(1, 11)]
        _make_snapshot(snap_dir, "2026-02-10", _rich_rows(tickers))
        tickers_d2 = tickers[:7] + list(reversed(tickers[7:]))
        _make_snapshot(snap_dir, "2026-02-11", _rich_rows(tickers_d2))

        eval_data = _make_eval_json(
            dates_evaluated=["2026-02-10", "2026-02-11"],
            per_date=[
                {"date": "2026-02-10", "n_eligible": 10},
                {
                    "date": "2026-02-11",
                    "n_eligible": 10,
                    "temporal_overlap_60": 80.0,
                    "temporal_spearman": 0.92,
                    # No temporal_max_shift_ticker — only overlap!
                },
            ],
            top_movers=[],
        )
        eval_json = tmp_path / "eval.json"
        _write_json(eval_json, eval_data)
        rs = DecisionRuleset()

        summary = triage_gate_fail(
            eval_json, snap_dir, tmp_path / "out",
            max_tickers=5, ruleset=rs,
        )
        assert len(summary["top_movers"]) >= 1
        assert any(
            "reconstructed" in n for n in summary["notes"]
        )
        # Explains should also be produced
        assert len(summary["explain_files"]) >= 1

    def test_reconstruct_without_ruleset_uses_existing_ranks(self, tmp_path):
        """Reconstruction without ruleset uses actionable_rank column."""
        snap_dir = tmp_path / "snaps"
        rows_d1 = [
            {"ticker": "A", "eligible": "1", "actionable_rank": "1"},
            {"ticker": "B", "eligible": "1", "actionable_rank": "2"},
            {"ticker": "C", "eligible": "1", "actionable_rank": "3"},
        ]
        rows_d2 = [
            {"ticker": "A", "eligible": "1", "actionable_rank": "3"},
            {"ticker": "B", "eligible": "1", "actionable_rank": "1"},
            {"ticker": "C", "eligible": "1", "actionable_rank": "2"},
        ]
        _make_snapshot(snap_dir, "2026-02-10", rows_d1)
        _make_snapshot(snap_dir, "2026-02-11", rows_d2)

        eval_data = _make_eval_json(
            dates_evaluated=["2026-02-10", "2026-02-11"],
            per_date=[
                {"date": "2026-02-10", "n_eligible": 3},
                {
                    "date": "2026-02-11",
                    "n_eligible": 3,
                    "temporal_overlap_60": 80.0,
                },
            ],
            top_movers=[],
        )
        eval_json = tmp_path / "eval.json"
        _write_json(eval_json, eval_data)

        # No ruleset — should use actionable_rank
        summary = triage_gate_fail(
            eval_json, snap_dir, tmp_path / "out",
            max_tickers=5, ruleset=None,
        )
        assert len(summary["top_movers"]) >= 1
        assert summary["top_movers"][0]["ticker"] == "A"  # shifted by 2
        assert summary["top_movers"][0]["shift"] == 2

    def test_aggregate_max_shift_fallback(self, tmp_path):
        """Falls back to aggregate max_rank_shift when no snapshots exist."""
        eval_data = _make_eval_json(
            dates_evaluated=["2026-02-10", "2026-02-11"],
            per_date=[
                {"date": "2026-02-10", "n_eligible": 10},
                {
                    "date": "2026-02-11",
                    "n_eligible": 10,
                    "temporal_overlap_60": 80.0,
                },
            ],
            top_movers=[],
        )
        # Add aggregate max_rank_shift
        eval_data["temporal_stability"]["max_rank_shift"] = {
            "ticker": "IBRX", "shift": 201, "from": 29, "to": 230,
            "date": "2026-02-11",
        }
        eval_json = tmp_path / "eval.json"
        _write_json(eval_json, eval_data)

        # No snapshots exist, no ruleset — should fall back to aggregate
        summary = triage_gate_fail(
            eval_json, tmp_path / "empty_snaps", tmp_path / "out",
        )
        assert len(summary["top_movers"]) == 1
        assert summary["top_movers"][0]["ticker"] == "IBRX"
        assert summary["top_movers"][0]["shift"] == 201
