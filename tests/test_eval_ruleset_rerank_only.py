"""Tests for rerank-only evaluation mode."""
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
from scripts.eval_ruleset import (
    compute_top_shifts,
    evaluate_ruleset_rerank_only,
    rerank_rows,
    main,
)


# ---------------------------------------------------------------------------
# Fixtures
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


def _make_ruleset_file(path: Path, **overrides) -> Path:
    """Create a ruleset JSON file with optional field overrides."""
    # Minimal valid ruleset JSON (all defaults come from the dataclass)
    data = dict(overrides) if overrides else {}
    _write_json(path, data)
    return path


def _default_ruleset() -> DecisionRuleset:
    """DecisionRuleset with all defaults."""
    return DecisionRuleset()


def _make_snapshot(
    snap_dir: Path,
    date: str,
    rows: List[Dict[str, str]],
    degraded: bool = False,
) -> Path:
    """Create a snapshot directory with rankings.csv + optional cache_health."""
    d = snap_dir / date
    d.mkdir(parents=True, exist_ok=True)
    _write_csv(d / "rankings.csv", rows)
    if degraded:
        _write_json(d / "cache_health.json", {
            "overall_status": "bad",
            "degraded_run": True,
        })
    return d


def _minimal_rows(tickers: List[str], eligible: bool = True) -> List[Dict[str, str]]:
    """Create minimal rankings rows with only ticker + eligible + tier_dev."""
    rows = []
    for i, t in enumerate(tickers):
        rows.append({
            "ticker": t,
            "eligible": "1" if eligible else "0",
            "tier_dev": "A" if i < len(tickers) // 2 else "B",
        })
    return rows


def _rich_rows(tickers: List[str]) -> List[Dict[str, str]]:
    """Create rows with columns that affect sort order differently per ruleset."""
    rows = []
    n = len(tickers)
    for i, t in enumerate(tickers):
        rows.append({
            "ticker": t,
            "eligible": "1",
            "archetype": "drug_developer",
            "tier_dev": "A" if i < n // 2 else "B",
            "composite_rank": str(i + 1),
            # Optionality in reverse order of composite_rank
            "clinical_optionality_pct_dev": str(round((n - i) / n, 4)),
            "clinical_score": str(round((n - i) * 10 / n, 2)),
            "catalyst_mode": "specific_days",
            "catalyst_days": str(90 + i * 10),
            "mom_state": "neutral",
        })
    return rows


# ---------------------------------------------------------------------------
# Tests: compute_top_shifts
# ---------------------------------------------------------------------------


class TestComputeTopShifts:
    def test_returns_top_n_sorted(self):
        curr = {"A": 1, "B": 5, "C": 3, "D": 4, "E": 2}
        prior = {"A": 2, "B": 1, "C": 5, "D": 4, "E": 3}
        result = compute_top_shifts(curr, prior, n=3)
        assert len(result) == 3
        shifts = [r["shift"] for r in result]
        assert shifts == sorted(shifts, reverse=True)

    def test_empty_on_no_overlap(self):
        assert compute_top_shifts({"A": 1}, {"B": 1}, n=3) == []

    def test_excludes_zero_shifts(self):
        curr = {"A": 1, "B": 2}
        prior = {"A": 1, "B": 3}
        result = compute_top_shifts(curr, prior, n=5)
        assert len(result) == 1
        assert result[0]["ticker"] == "B"

    def test_deterministic_tiebreak_by_ticker(self):
        curr = {"A": 2, "B": 3}
        prior = {"A": 1, "B": 2}
        # Both shift by 1
        result = compute_top_shifts(curr, prior, n=5)
        assert len(result) == 2
        assert result[0]["ticker"] == "A"  # alphabetical tiebreak
        assert result[1]["ticker"] == "B"


# ---------------------------------------------------------------------------
# Tests: rerank_rows helper
# ---------------------------------------------------------------------------


class TestRerankRows:
    def test_returns_eligible_tickers_only(self):
        rows = [
            {"ticker": "A", "eligible": "1", "tier_dev": "A"},
            {"ticker": "B", "eligible": "0", "tier_dev": "A"},
            {"ticker": "C", "eligible": "1", "tier_dev": "B"},
        ]
        rs = _default_ruleset()
        tickers, ranks = rerank_rows(rows, rs)
        assert "A" in tickers
        assert "C" in tickers
        assert "B" not in tickers
        assert len(ranks) == 2

    def test_does_not_mutate_input(self):
        rows = [
            {"ticker": "X", "eligible": "1", "tier_dev": "A"},
        ]
        original_keys = set(rows[0].keys())
        rs = _default_ruleset()
        rerank_rows(rows, rs)
        # Input rows should not have extra backfilled columns
        assert set(rows[0].keys()) == original_keys

    def test_deterministic(self):
        rows = _rich_rows(["A", "B", "C", "D", "E"])
        rs = _default_ruleset()
        t1, r1 = rerank_rows(rows, rs)
        t2, r2 = rerank_rows(rows, rs)
        assert t1 == t2
        assert r1 == r2


# ---------------------------------------------------------------------------
# Tests: evaluate_ruleset_rerank_only
# ---------------------------------------------------------------------------


class TestRerankOnlyEval:
    def test_evaluates_old_schema(self, tmp_path):
        """Rankings with only ticker + tier_dev are evaluated without crash."""
        snap_dir = tmp_path / "snaps"
        tickers = ["A", "B", "C", "D", "E"]
        _make_snapshot(snap_dir, "2026-02-10", _minimal_rows(tickers))

        rs = _default_ruleset()
        result = evaluate_ruleset_rerank_only(
            candidate_ruleset=rs,
            baseline_ruleset=rs,
            dates=["2026-02-10"],
            snapshot_dir=snap_dir,
            k=5,
        )
        assert result["n_evaluated"] == 1
        assert result["n_skipped"] == 0

    def test_fills_defaults_deterministically(self, tmp_path):
        """Run twice with missing alpha/catalyst columns → identical output."""
        snap_dir = tmp_path / "snaps"
        tickers = ["A", "B", "C", "D", "E", "F"]
        rows = _minimal_rows(tickers)
        _make_snapshot(snap_dir, "2026-02-10", rows)
        _make_snapshot(snap_dir, "2026-02-11", rows)

        rs = _default_ruleset()
        r1 = evaluate_ruleset_rerank_only(
            candidate_ruleset=rs,
            baseline_ruleset=rs,
            dates=["2026-02-10", "2026-02-11"],
            snapshot_dir=snap_dir,
            k=5,
        )
        r2 = evaluate_ruleset_rerank_only(
            candidate_ruleset=rs,
            baseline_ruleset=rs,
            dates=["2026-02-10", "2026-02-11"],
            snapshot_dir=snap_dir,
            k=5,
        )
        j1 = json.dumps(r1, sort_keys=True, indent=2)
        j2 = json.dumps(r2, sort_keys=True, indent=2)
        assert j1 == j2

    def test_skips_missing_rankings(self, tmp_path):
        """Date dir exists but no rankings.csv → skipped."""
        snap_dir = tmp_path / "snaps"
        d = snap_dir / "2026-02-10"
        d.mkdir(parents=True)
        # No rankings.csv written

        rs = _default_ruleset()
        result = evaluate_ruleset_rerank_only(
            candidate_ruleset=rs,
            baseline_ruleset=rs,
            dates=["2026-02-10"],
            snapshot_dir=snap_dir,
            k=5,
        )
        assert result["n_evaluated"] == 0
        assert result["n_skipped"] == 1
        assert result["dates_skipped"][0]["reason"] == "missing_rankings"

    def test_skips_missing_snapshot(self, tmp_path):
        """Snapshot directory does not exist → skipped."""
        snap_dir = tmp_path / "snaps"
        snap_dir.mkdir()

        rs = _default_ruleset()
        result = evaluate_ruleset_rerank_only(
            candidate_ruleset=rs,
            baseline_ruleset=rs,
            dates=["2026-02-10"],
            snapshot_dir=snap_dir,
            k=5,
        )
        assert result["n_evaluated"] == 0
        assert result["dates_skipped"][0]["reason"] == "missing_snapshot"

    def test_topk_overlap_computed(self, tmp_path):
        """Two rulesets with different sort anchors produce measurable overlap."""
        snap_dir = tmp_path / "snaps"
        rows = _rich_rows([f"T{i:02d}" for i in range(1, 21)])
        _make_snapshot(snap_dir, "2026-02-10", rows)

        # Candidate: default sort by composite_rank
        cand_rs = DecisionRuleset()
        # Baseline: sort by optionality_pct (reversed from composite_rank)
        base_rs = DecisionRuleset(sort_anchor="optionality_pct")

        result = evaluate_ruleset_rerank_only(
            candidate_ruleset=cand_rs,
            baseline_ruleset=base_rs,
            dates=["2026-02-10"],
            snapshot_dir=snap_dir,
            k=10,
        )
        assert result["n_evaluated"] == 1
        # Cross-comparison should show measurable difference
        cross = result.get("cross_comparison", {})
        assert "mean_top60_overlap" in cross or "mean_top20_overlap" in cross
        # With reversed sort, overlap should be < 100%
        per = result["per_date"][0]
        assert "cross_overlap_60" in per

    def test_temporal_stability_across_dates(self, tmp_path):
        """Two dates produce temporal stability metrics."""
        snap_dir = tmp_path / "snaps"
        tickers = [f"T{i:02d}" for i in range(1, 11)]
        _make_snapshot(snap_dir, "2026-02-10", _rich_rows(tickers))
        # Slightly different order on day 2
        tickers_d2 = tickers[:8] + list(reversed(tickers[8:]))
        _make_snapshot(snap_dir, "2026-02-11", _rich_rows(tickers_d2))

        rs = _default_ruleset()
        result = evaluate_ruleset_rerank_only(
            candidate_ruleset=rs,
            baseline_ruleset=rs,
            dates=["2026-02-10", "2026-02-11"],
            snapshot_dir=snap_dir,
            k=10,
        )
        assert result["n_evaluated"] == 2
        ts = result["temporal_stability"]
        assert "mean_top60_overlap" in ts
        assert "mean_spearman" in ts

    def test_per_date_shift_details(self, tmp_path):
        """Per-date entries include max shift ticker/from/to when shifts occur."""
        snap_dir = tmp_path / "snaps"
        tickers = [f"T{i:02d}" for i in range(1, 11)]
        _make_snapshot(snap_dir, "2026-02-10", _rich_rows(tickers))
        tickers_d2 = tickers[:8] + list(reversed(tickers[8:]))
        _make_snapshot(snap_dir, "2026-02-11", _rich_rows(tickers_d2))

        rs = _default_ruleset()
        result = evaluate_ruleset_rerank_only(
            candidate_ruleset=rs,
            baseline_ruleset=rs,
            dates=["2026-02-10", "2026-02-11"],
            snapshot_dir=snap_dir,
            k=10,
        )
        # Day 1 has no prior — no shift fields
        assert "temporal_max_shift_ticker" not in result["per_date"][0]
        # Day 2 should have shift details
        day2 = result["per_date"][1]
        assert "temporal_max_shift_ticker" in day2
        assert isinstance(day2["temporal_max_shift"], int)
        assert isinstance(day2["temporal_max_shift_from"], int)
        assert isinstance(day2["temporal_max_shift_to"], int)

    def test_top_movers_in_temporal_stability(self, tmp_path):
        """temporal_stability.top_movers lists worst shifts across dates."""
        snap_dir = tmp_path / "snaps"
        tickers = [f"T{i:02d}" for i in range(1, 11)]
        _make_snapshot(snap_dir, "2026-02-10", _rich_rows(tickers))
        tickers_d2 = tickers[:8] + list(reversed(tickers[8:]))
        _make_snapshot(snap_dir, "2026-02-11", _rich_rows(tickers_d2))
        _make_snapshot(snap_dir, "2026-02-12", _rich_rows(tickers))

        rs = _default_ruleset()
        result = evaluate_ruleset_rerank_only(
            candidate_ruleset=rs,
            baseline_ruleset=rs,
            dates=["2026-02-10", "2026-02-11", "2026-02-12"],
            snapshot_dir=snap_dir,
            k=10,
        )
        ts = result["temporal_stability"]
        assert "top_movers" in ts
        movers = ts["top_movers"]
        assert len(movers) >= 1
        assert len(movers) <= 20
        # Sorted descending by shift, then ascending ticker for ties
        for i in range(len(movers) - 1):
            a, b = movers[i], movers[i + 1]
            assert (-a["shift"], a["ticker"]) <= (-b["shift"], b["ticker"])
        # Each entry has core + enrichment keys
        for m in movers:
            assert "ticker" in m
            assert "date" in m
            assert "from" in m
            assert "to" in m
            assert "shift" in m
            # Enrichment columns
            assert "tier" in m
            assert "archetype" in m
            assert "composite_rank" in m
            assert "catalyst_days" in m
            assert "missing_count" in m
            assert isinstance(m["missing_count"], int)

    def test_top_movers_enrichment_values(self, tmp_path):
        """Enrichment columns reflect the CSV row data for shifted tickers."""
        snap_dir = tmp_path / "snaps"
        tickers = [f"T{i:02d}" for i in range(1, 11)]
        rows_d1 = _rich_rows(tickers)
        _make_snapshot(snap_dir, "2026-02-10", rows_d1)
        tickers_d2 = tickers[:8] + list(reversed(tickers[8:]))
        _make_snapshot(snap_dir, "2026-02-11", _rich_rows(tickers_d2))

        rs = _default_ruleset()
        result = evaluate_ruleset_rerank_only(
            candidate_ruleset=rs,
            baseline_ruleset=rs,
            dates=["2026-02-10", "2026-02-11"],
            snapshot_dir=snap_dir,
            k=10,
        )
        movers = result["temporal_stability"]["top_movers"]
        assert len(movers) >= 1
        top = movers[0]
        # Rich rows set archetype="drug_developer" for all
        assert top["archetype"] == "drug_developer"
        # Tier is A or B
        assert top["tier"] in ("", "A", "B")  # tier_any may not be set; tier_dev is
        # missing_count should be 0 (no missing_components in rich rows)
        assert top["missing_count"] == 0

    def test_degraded_snapshot_still_evaluated(self, tmp_path):
        """Degraded snapshots are evaluated in rerank-only (tagged, not skipped)."""
        snap_dir = tmp_path / "snaps"
        tickers = ["A", "B", "C", "D", "E"]
        _make_snapshot(snap_dir, "2026-02-10", _minimal_rows(tickers), degraded=True)

        rs = _default_ruleset()
        result = evaluate_ruleset_rerank_only(
            candidate_ruleset=rs,
            baseline_ruleset=rs,
            dates=["2026-02-10"],
            snapshot_dir=snap_dir,
            k=5,
        )
        assert result["n_evaluated"] == 1
        assert result["per_date"][0].get("degraded_snapshot") is True

    def test_self_eval_no_cross_comparison(self, tmp_path):
        """Self-eval skips cross-comparison (both rulesets identical)."""
        snap_dir = tmp_path / "snaps"
        _make_snapshot(snap_dir, "2026-02-10", _rich_rows(["A", "B", "C", "D", "E"]))

        rs = _default_ruleset()
        result = evaluate_ruleset_rerank_only(
            candidate_ruleset=rs,
            baseline_ruleset=rs,
            dates=["2026-02-10"],
            snapshot_dir=snap_dir,
            k=5,
        )
        assert result["cross_comparison"] == {}
        per = result["per_date"][0]
        assert "cross_overlap_60" not in per


# ---------------------------------------------------------------------------
# Tests: CLI rerank-only mode
# ---------------------------------------------------------------------------


class TestCLIRerankOnly:
    def test_rerank_only_writes_outputs(self, tmp_path):
        """CLI --rerank-only writes JSON, MD, and CSV."""
        snap_dir = tmp_path / "snaps"
        tickers = [f"T{i:02d}" for i in range(1, 11)]
        _make_snapshot(snap_dir, "2026-02-10", _rich_rows(tickers))
        _make_snapshot(snap_dir, "2026-02-11", _rich_rows(tickers))

        rs_path = _make_ruleset_file(tmp_path / "ruleset.json")
        out_dir = tmp_path / "output"

        rc = main([
            "--candidate", str(rs_path),
            "--baseline", str(rs_path),
            "--start", "2026-02-10",
            "--end", "2026-02-11",
            "--k", "5",
            "--snapshot-dir", str(snap_dir),
            "--out-dir", str(out_dir),
            "--rerank-only",
        ])
        assert rc == 0

        json_files = list(out_dir.glob("**/ruleset_eval.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text())
        assert data["mode"] == "rerank_only"
        assert data["config"]["rerank_only"] is True
        assert data["n_evaluated"] == 2

        md_files = list(out_dir.glob("**/ruleset_eval.md"))
        assert len(md_files) == 1

        csv_files = list(out_dir.glob("**/per_date_metrics.csv"))
        assert len(csv_files) == 1

    def test_rerank_only_gate_pass(self, tmp_path):
        """CLI --rerank-only --gate produces PASS for stable rankings."""
        snap_dir = tmp_path / "snaps"
        tickers = [f"T{i:02d}" for i in range(1, 11)]
        _make_snapshot(snap_dir, "2026-02-10", _rich_rows(tickers))
        _make_snapshot(snap_dir, "2026-02-11", _rich_rows(tickers))

        rs_path = _make_ruleset_file(tmp_path / "ruleset.json")
        out_dir = tmp_path / "output"

        rc = main([
            "--candidate", str(rs_path),
            "--baseline", str(rs_path),
            "--start", "2026-02-10",
            "--end", "2026-02-11",
            "--k", "5",
            "--snapshot-dir", str(snap_dir),
            "--out-dir", str(out_dir),
            "--rerank-only",
            "--gate",
        ])
        assert rc == 0

        data = json.loads(
            list(out_dir.glob("**/ruleset_eval.json"))[0].read_text()
        )
        assert data["gate"]["verdict"] == "PASS"

    def test_rerank_only_no_dates_exits_1(self, tmp_path):
        """CLI --rerank-only with no snapshots in range exits 1."""
        snap_dir = tmp_path / "snaps"
        snap_dir.mkdir()

        rs_path = _make_ruleset_file(tmp_path / "ruleset.json")

        rc = main([
            "--candidate", str(rs_path),
            "--start", "2099-01-01",
            "--end", "2099-12-31",
            "--snapshot-dir", str(snap_dir),
            "--out-dir", str(tmp_path / "out"),
            "--rerank-only",
        ])
        assert rc == 1
