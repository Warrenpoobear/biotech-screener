#!/usr/bin/env python3
"""Tests for scripts/rollback_drill.py.

Covers:
  - No rollback when health OK + cross-compare healthy
  - Rollback recommended via ruleset_health flag
  - Rollback recommended via cross-compare low overlap
  - No LKG in manifest → cross-compare error, health-flag still works
  - Reasons list contents
  - Exit code: 2 if recommended, 0 otherwise
  - render_text includes rollback command when recommended
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from rollback_drill import (
    _build_reasons,
    _find_lkg,
    run_drill,
    render_markdown,
    render_text,
    write_drill_artifacts,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_ACTIVE_ID = "82982998"
_LKG_ID = "4f12a7f8"

_ACTIVE_FILE = "v1.8.3_buffer30_candidate.json"
_LKG_FILE = "v1.8.2_clinical_sort_off_candidate.json"


def _make_manifest(lkg_present: bool = True) -> Dict:
    rulesets = []
    if lkg_present:
        rulesets.append({
            "id": _LKG_ID,
            "status": "retired",
            "file": _LKG_FILE,
            "description": "LKG ruleset",
            "updated_by": "promote_ruleset.py --rollback",
        })
    rulesets.append({
        "id": _ACTIVE_ID,
        "status": "active",
        "file": _ACTIVE_FILE,
        "description": "Active ruleset",
        "updated_by": "promote_ruleset.py",
    })
    return {"rulesets": rulesets}


def _write_manifest(path: Path, manifest: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


def _ruleset_dict(ruleset_id: str) -> Dict:
    """Minimal valid DecisionRuleset-compatible dict."""
    return {
        "ruleset_id": ruleset_id,
        "top_k": 20,
        "a_floor": 0.60,
        "b_floor": 0.30,
        "tier_filter": ["A", "B"],
        "sort_anchor": "optionality_pct",
        "composite_engine": "alpha_cohort",
        "alpha_cohort_tiebreak_weight": 0.0,
        "rebalance_buffer_ranks": 0,
        "catalyst_near": 120,
        "catalyst_mid": 180,
        "tiering_priority_mode": "dev_first",
        "catalyst_priority_mode": "off",
        "coinvest_sort_weight": 0.0,
        "coinvest_positive_only": True,
        "alpha_modifier_mode": "off",
        "alpha_modifier_weight": 0.0,
        "alpha_cohort_sort_weight": 0.0,
        "enable_calendar_alpha_sort": True,
        "calendar_alpha_sort_weight": 0.3,
        "enable_clinical_sort_signal": False,
        "clinical_sort_weight": 0.0,
        "enable_clinical_sizing": False,
        "missingness_sort_penalty": 0.0,
        "missingness_size_penalty": 0.0,
        "institutional_delta_sort_weight": 0.3,
        "alpha_cohort_tb": 0.0,
    }


def _write_ruleset(rulesets_dir: Path, filename: str, ruleset_id: str) -> None:
    rulesets_dir.mkdir(parents=True, exist_ok=True)
    (rulesets_dir / filename).write_text(json.dumps(_ruleset_dict(ruleset_id)))


def _write_rankings_csv(path: Path, n: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ticker", "optionality_pct", "eligible",
                                           "alpha_cohort_pct", "alpha_raw",
                                           "tier_dev", "composite_rank"])
        w.writeheader()
        for i in range(1, n + 1):
            w.writerow({
                "ticker": f"T{i:03d}",
                "optionality_pct": str(round((n - i) / n, 4)),
                "eligible": "1",
                "alpha_cohort_pct": str(round((n - i) / n, 4)),
                "alpha_raw": "0.5",
                "tier_dev": "A",
                "composite_rank": str(i),
            })


def _write_ruleset_health(snap_dir: Path, recommend: bool = False,
                           consecutive: int = 0, status: str = "OK",
                           warn_reasons: Optional[List[str]] = None) -> None:
    snap_dir.mkdir(parents=True, exist_ok=True)
    (snap_dir / "ruleset_health.json").write_text(json.dumps({
        "schema": "ruleset_health.v1",
        "active_ruleset_id": _ACTIVE_ID,
        "status": status,
        "consecutive_warn_days": consecutive,
        "recommend_rollback": recommend,
        "detail": "test",
        "warn_reasons": warn_reasons or [],
    }))


def _setup_env(
    tmp_path: Path,
    dates: List[str],
    *,
    recommend_rollback: bool = False,
    consecutive_warns: int = 0,
    lkg_present: bool = True,
    monkeypatch,
) -> None:
    """Wire up tmp_path as project root for the drill."""
    import rollback_drill as rd

    snap_root = tmp_path / "snapshots"
    rulesets_dir = tmp_path / "rulesets"
    manifest_path = tmp_path / "manifest.json"

    manifest = _make_manifest(lkg_present=lkg_present)
    _write_manifest(manifest_path, manifest)
    _write_ruleset(rulesets_dir, _ACTIVE_FILE, _ACTIVE_ID)
    if lkg_present:
        _write_ruleset(rulesets_dir, _LKG_FILE, _LKG_ID)

    for date in dates:
        snap_dir = snap_root / date
        snap_dir.mkdir(parents=True)
        _write_rankings_csv(snap_dir / "rankings.csv")
        _write_ruleset_health(snap_dir, recommend=recommend_rollback,
                               consecutive=consecutive_warns)

    monkeypatch.setattr(rd, "SNAPSHOTS_ROOT", snap_root)
    monkeypatch.setattr(rd, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(rd, "RULESETS_DIR", rulesets_dir)


# ---------------------------------------------------------------------------
# _find_lkg
# ---------------------------------------------------------------------------

class TestFindLkg:
    def test_finds_last_retired_promoted(self):
        manifest = _make_manifest(lkg_present=True)
        result = _find_lkg(manifest)
        assert result is not None
        assert result["id"] == _LKG_ID

    def test_returns_none_when_no_retired_promoted(self):
        manifest = {"rulesets": [
            {"id": "abc", "status": "active", "updated_by": "promote_ruleset.py"},
        ]}
        assert _find_lkg(manifest) is None

    def test_skips_non_promoted_retired(self):
        manifest = {"rulesets": [
            {"id": "xyz", "status": "retired", "updated_by": "manual"},
            {"id": "abc", "status": "active", "updated_by": "promote_ruleset.py"},
        ]}
        assert _find_lkg(manifest) is None


# ---------------------------------------------------------------------------
# _build_reasons
# ---------------------------------------------------------------------------

class TestBuildReasons:
    def _healthy_rh(self) -> Dict:
        return {"recommend_rollback": False, "consecutive_warn_days": 0,
                "warn_reasons": []}

    def test_no_reasons_when_all_healthy(self):
        cross = {"mean_top20_overlap": 98.0, "mean_top60_overlap": 95.0}
        reasons = _build_reasons(self._healthy_rh(), cross, n_evaluated=5)
        assert reasons == []

    def test_health_flag_adds_reason(self):
        rh = {"recommend_rollback": True, "consecutive_warn_days": 3, "warn_reasons": []}
        reasons = _build_reasons(rh, {}, n_evaluated=0)
        assert any("3 consecutive WARN" in r for r in reasons)

    def test_low_top20_overlap_adds_reason(self):
        cross = {"mean_top20_overlap": 82.0, "mean_top60_overlap": 95.0}
        reasons = _build_reasons(self._healthy_rh(), cross, n_evaluated=5)
        assert any("top-20 overlap" in r for r in reasons)

    def test_low_top60_overlap_adds_reason(self):
        cross = {"mean_top20_overlap": 95.0, "mean_top60_overlap": 80.0}
        reasons = _build_reasons(self._healthy_rh(), cross, n_evaluated=5)
        assert any("top-60 overlap" in r for r in reasons)

    def test_zero_evaluated_suppresses_cross_reasons(self):
        # If no dates evaluated, no cross reasons (just empty)
        cross = {"mean_top20_overlap": 80.0, "mean_top60_overlap": 75.0}
        reasons = _build_reasons(self._healthy_rh(), cross, n_evaluated=0)
        assert reasons == []

    def test_warn_reasons_from_rh_propagated(self):
        rh = {"recommend_rollback": True, "consecutive_warn_days": 3,
              "warn_reasons": ["top60_overlap 78% < floor"]}
        reasons = _build_reasons(rh, {}, n_evaluated=0)
        assert any("top60_overlap" in r for r in reasons)


# ---------------------------------------------------------------------------
# run_drill (integration)
# ---------------------------------------------------------------------------

class TestRunDrill:
    def test_no_rollback_healthy(self, tmp_path, monkeypatch):
        _setup_env(tmp_path, ["2026-03-01", "2026-03-03", "2026-03-05"],
                   monkeypatch=monkeypatch)
        drill = run_drill("2026-03-05", n_snapshots=5)
        assert drill["recommended"] is False
        assert drill["reasons"] == []
        assert drill["active"]["id"] == _ACTIVE_ID
        assert drill["lkg"]["id"] == _LKG_ID

    def test_rollback_from_health_flag(self, tmp_path, monkeypatch):
        _setup_env(tmp_path, ["2026-03-03", "2026-03-05"],
                   recommend_rollback=True, consecutive_warns=3,
                   monkeypatch=monkeypatch)
        drill = run_drill("2026-03-05")
        assert drill["recommended"] is True
        assert any("3 consecutive" in r for r in drill["reasons"])

    def test_no_lkg_cross_error_but_health_works(self, tmp_path, monkeypatch):
        _setup_env(tmp_path, ["2026-03-05"],
                   lkg_present=False, recommend_rollback=True, consecutive_warns=3,
                   monkeypatch=monkeypatch)
        drill = run_drill("2026-03-05")
        assert drill["cross_compare"]["error"] is not None
        assert drill["recommended"] is True  # health flag still fires

    def test_no_lkg_and_health_ok_no_rollback(self, tmp_path, monkeypatch):
        _setup_env(tmp_path, ["2026-03-05"], lkg_present=False, monkeypatch=monkeypatch)
        drill = run_drill("2026-03-05")
        assert drill["recommended"] is False

    def test_schema_field_present(self, tmp_path, monkeypatch):
        _setup_env(tmp_path, ["2026-03-05"], monkeypatch=monkeypatch)
        drill = run_drill("2026-03-05")
        assert drill["schema"] == "rollback_drill.v1"

    def test_cross_compare_evaluates_dates(self, tmp_path, monkeypatch):
        dates = ["2026-03-01", "2026-03-02", "2026-03-03"]
        _setup_env(tmp_path, dates, monkeypatch=monkeypatch)
        drill = run_drill("2026-03-03", n_snapshots=5)
        assert drill["cross_compare"]["n_evaluated"] == len(dates)

    def test_window_capped_to_n_snapshots(self, tmp_path, monkeypatch):
        dates = [f"2026-03-{i:02d}" for i in range(1, 8)]
        _setup_env(tmp_path, dates, monkeypatch=monkeypatch)
        drill = run_drill("2026-03-07", n_snapshots=3)
        assert len(drill["cross_compare"]["window"]) == 3


# ---------------------------------------------------------------------------
# render_text
# ---------------------------------------------------------------------------

class TestRenderText:
    def _drill(self, recommended: bool, reasons: Optional[List[str]] = None) -> Dict:
        return {
            "schema": "rollback_drill.v1",
            "as_of_date": "2026-03-05",
            "generated_at": "2026-03-05T12:00:00+00:00",
            "active": {"id": _ACTIVE_ID, "file": _ACTIVE_FILE, "description": ""},
            "lkg": {"id": _LKG_ID, "file": _LKG_FILE, "description": ""},
            "ruleset_health": {
                "status": "OK" if not recommended else "WARN",
                "consecutive_warn_days": 3 if recommended else 0,
                "recommend_rollback": recommended,
                "detail": "test",
                "warn_reasons": [],
            },
            "cross_compare": {
                "window": ["2026-03-01", "2026-03-05"],
                "n_snapshots_requested": 5,
                "n_evaluated": 5,
                "n_skipped": 0,
                "mean_top20_overlap_pct": 98.0,
                "mean_top60_overlap_pct": 95.0,
                "worst_top60_overlap_pct": 90.0,
                "mean_spearman": 0.999,
                "mean_pct_rank_changed": 2.0,
                "error": None,
            },
            "reasons": reasons or (["health: 3d WARN streak"] if recommended else []),
            "recommended": recommended,
        }

    def test_no_rollback_shows_checkmark(self):
        text = render_text(self._drill(recommended=False))
        assert "No rollback needed" in text
        assert "ROLLBACK RECOMMENDED" not in text

    def test_rollback_shows_warning(self):
        text = render_text(self._drill(recommended=True))
        assert "ROLLBACK RECOMMENDED" in text

    def test_rollback_includes_command(self):
        text = render_text(self._drill(recommended=True))
        assert "promote_ruleset.py --rollback" in text

    def test_no_rollback_no_command(self):
        text = render_text(self._drill(recommended=False))
        assert "promote_ruleset.py --rollback" not in text

    def test_cross_compare_error_shown(self):
        drill = self._drill(recommended=False)
        drill["cross_compare"]["error"] = "No LKG ruleset available"
        text = render_text(drill)
        assert "No LKG ruleset available" in text


# ---------------------------------------------------------------------------
# render_markdown + write_drill_artifacts
# ---------------------------------------------------------------------------

class TestRenderMarkdown:
    def _drill(self, recommended: bool) -> Dict:
        return {
            "schema": "rollback_drill.v1",
            "as_of_date": "2026-03-05",
            "generated_at": "2026-03-05T12:00:00+00:00",
            "active": {"id": _ACTIVE_ID, "file": _ACTIVE_FILE, "description": ""},
            "lkg": {"id": _LKG_ID, "file": _LKG_FILE, "description": ""},
            "ruleset_health": {
                "status": "WARN" if recommended else "OK",
                "consecutive_warn_days": 3 if recommended else 0,
                "recommend_rollback": recommended,
                "detail": "test detail",
                "warn_reasons": ["top60_overlap degraded"] if recommended else [],
            },
            "cross_compare": {
                "window": ["2026-03-01", "2026-03-05"],
                "n_snapshots_requested": 5,
                "n_evaluated": 5,
                "n_skipped": 0,
                "mean_top20_overlap_pct": 98.0,
                "mean_top60_overlap_pct": 95.0,
                "worst_top60_overlap_pct": 90.0,
                "mean_spearman": 0.999,
                "mean_pct_rank_changed": 2.0,
                "error": None,
            },
            "reasons": ["health: 3d WARN streak"] if recommended else [],
            "recommended": recommended,
        }

    def test_md_h1_contains_date(self):
        md = render_markdown(self._drill(recommended=False))
        assert "# Rollback Drill — 2026-03-05" in md

    def test_no_rollback_shows_checkmark(self):
        md = render_markdown(self._drill(recommended=False))
        assert "No rollback needed" in md
        assert "ROLLBACK RECOMMENDED" not in md

    def test_rollback_shows_warning(self):
        md = render_markdown(self._drill(recommended=True))
        assert "ROLLBACK RECOMMENDED" in md

    def test_rollback_includes_execute_section(self):
        md = render_markdown(self._drill(recommended=True))
        assert "## Execute Rollback" in md
        assert "promote_ruleset.py --rollback" in md

    def test_no_rollback_no_execute_section(self):
        md = render_markdown(self._drill(recommended=False))
        assert "## Execute Rollback" not in md

    def test_active_lkg_table_present(self):
        md = render_markdown(self._drill(recommended=False))
        assert _ACTIVE_ID in md
        assert _LKG_ID in md

    def test_cross_compare_error_shown(self):
        drill = self._drill(recommended=False)
        drill["cross_compare"]["error"] = "No LKG available"
        md = render_markdown(drill)
        assert "No LKG available" in md


class TestWriteDrillArtifacts:
    def _drill(self, recommended: bool = False) -> Dict:
        return {
            "schema": "rollback_drill.v1",
            "as_of_date": "2026-03-05",
            "generated_at": "2026-03-05T12:00:00+00:00",
            "active": {"id": _ACTIVE_ID, "file": _ACTIVE_FILE, "description": ""},
            "lkg": {"id": _LKG_ID, "file": _LKG_FILE, "description": ""},
            "ruleset_health": {"status": "OK", "consecutive_warn_days": 0,
                               "recommend_rollback": False, "detail": "", "warn_reasons": []},
            "cross_compare": {"window": [], "n_snapshots_requested": 5, "n_evaluated": 0,
                              "n_skipped": 0, "mean_top20_overlap_pct": None,
                              "mean_top60_overlap_pct": None, "worst_top60_overlap_pct": None,
                              "mean_spearman": None, "mean_pct_rank_changed": None, "error": None},
            "reasons": [],
            "recommended": recommended,
        }

    def test_creates_both_files(self, tmp_path):
        md_path, json_path = write_drill_artifacts(self._drill(), tmp_path)
        assert md_path.is_file()
        assert json_path.is_file()

    def test_filenames(self, tmp_path):
        md_path, json_path = write_drill_artifacts(self._drill(), tmp_path)
        assert md_path.name == "ROLLBACK.md"
        assert json_path.name == "ROLLBACK.json"

    def test_json_parseable(self, tmp_path):
        _, json_path = write_drill_artifacts(self._drill(), tmp_path)
        d = json.loads(json_path.read_text())
        assert d["schema"] == "rollback_drill.v1"

    def test_md_nonempty(self, tmp_path):
        md_path, _ = write_drill_artifacts(self._drill(), tmp_path)
        assert len(md_path.read_text()) > 100

    def test_creates_dir_if_missing(self, tmp_path):
        out = tmp_path / "nested" / "dir"
        write_drill_artifacts(self._drill(), out)
        assert out.is_dir()
