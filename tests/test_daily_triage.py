"""Tests for scripts/build_daily_triage.py — triage bundle builder."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_daily_triage import (
    _worst_verdict,
    assemble_bundle,
    build_triage_context,
    collect_gate_artifacts,
    render_health_summary_md,
    render_summary_md,
    main,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _make_monitoring(snap: Path, **overrides) -> dict:
    data = {
        "schema": "monitoring.v1",
        "context": {"as_of_date": "2026-02-27", "cache_degraded_run": False},
        "coverage": {"n_universe": 300, "n_eligible": 240, "sec_8k_count": 200, "ctgov_count": 1000},
        "stability": {
            "suppressed": False,
            "top_60": {"overlap_pct": 85.0},
            "rank_correlation": {"spearman": 0.92},
            "name_turnover_pct": 12.0,
        },
    }
    data.update(overrides)
    _write_json(snap / "monitoring.json", data)
    return data


def _make_eligibility_gate(snap: Path, verdict: str = "PASS", **overrides) -> dict:
    data = {
        "schema": "eligibility_gate.v1",
        "verdict": verdict,
        "triggers": [f"test_{verdict.lower()}_trigger"] if verdict != "PASS" else [],
        "metrics": {"eligible_delta": -3, "ineligible_delta": 3},
    }
    data.update(overrides)
    _write_json(snap / "eligibility_gate.json", data)
    return data


def _make_eligibility_delta(snap: Path, **overrides) -> dict:
    data = {
        "schema": "eligibility_delta.v1",
        "delta": {"eligible": -3, "ineligible": 3, "pct_ineligible": 0.01},
        "notes": ["ineligible_count increased by 3"],
    }
    data.update(overrides)
    _write_json(snap / "eligibility_delta.json", data)
    return data


def _make_eligibility_triage(snap: Path) -> dict:
    triage_dir = snap / "eligibility_triage"
    triage_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "schema": "eligibility_triage.v1",
        "movers": [
            {"ticker": "AAPL", "prior_eligible": 1, "cur_eligible": 0,
             "prior_reasons": [], "cur_reasons": ["deep_drawdown"]},
            {"ticker": "GOOG", "prior_eligible": 0, "cur_eligible": 1,
             "prior_reasons": ["adv_fail"], "cur_reasons": []},
        ],
    }
    _write_json(triage_dir / "triage_summary.json", data)
    (triage_dir / "triage_summary.md").write_text("# Triage\ntest", encoding="utf-8")
    (triage_dir / "movers.csv").write_text("ticker,prior_eligible\nAAPL,1\n", encoding="utf-8")
    return data


def _make_ruleset_eval(snap: Path, verdict: str = "PASS", subdir: str = "") -> dict:
    data = {
        "schema": "ruleset_eval.v1",
        "candidate": {"id": "abc12345", "file": "v2.json"},
        "baseline": {"id": "old12345", "file": "v1.json"},
        "n_evaluated": 10,
        "n_skipped": 0,
        "gate": {
            "verdict": verdict,
            "fail_reasons": ["test_fail"] if verdict == "FAIL" else [],
            "warn_reasons": ["test_warn"] if verdict == "WARN" else [],
        },
        "temporal_stability": {"mean_top60_overlap": 92.0},
    }
    target = snap / subdir if subdir else snap
    _write_json(target / "ruleset_eval.json", data)
    return data


def _make_all(snap: Path, *, monitoring: bool = True, eligibility: bool = True,
              triage: bool = True, ruleset: bool = True, rs_subdir: str = "") -> None:
    if monitoring:
        _make_monitoring(snap)
    if eligibility:
        _make_eligibility_gate(snap)
        _make_eligibility_delta(snap)
    if triage:
        _make_eligibility_triage(snap)
    if ruleset:
        _make_ruleset_eval(snap, subdir=rs_subdir)


# ---------------------------------------------------------------------------
# TestCollectGateArtifacts
# ---------------------------------------------------------------------------


class TestCollectGateArtifacts:
    def test_all_present(self, tmp_path):
        snap = tmp_path / "2026-02-27"
        snap.mkdir()
        _make_all(snap)
        arts = collect_gate_artifacts(snap)
        assert arts["monitoring_json"] is not None
        assert arts["eligibility_gate_json"] is not None
        assert arts["eligibility_delta_json"] is not None
        assert arts["eligibility_triage_json"] is not None
        assert arts["ruleset_eval_json"] is not None

    def test_missing_monitoring(self, tmp_path):
        snap = tmp_path / "2026-02-27"
        snap.mkdir()
        _make_all(snap, monitoring=False)
        arts = collect_gate_artifacts(snap)
        assert arts["monitoring_json"] is None
        assert arts["eligibility_gate_json"] is not None

    def test_missing_eligibility(self, tmp_path):
        snap = tmp_path / "2026-02-27"
        snap.mkdir()
        _make_all(snap, eligibility=False)
        arts = collect_gate_artifacts(snap)
        assert arts["eligibility_gate_json"] is None
        assert arts["eligibility_delta_json"] is None
        assert arts["monitoring_json"] is not None

    def test_all_missing(self, tmp_path):
        snap = tmp_path / "2026-02-27"
        snap.mkdir()
        arts = collect_gate_artifacts(snap)
        assert arts["monitoring_json"] is None
        assert arts["eligibility_gate_json"] is None
        assert arts["eligibility_delta_json"] is None
        assert arts["eligibility_triage_json"] is None
        assert arts["ruleset_eval_json"] is None


# ---------------------------------------------------------------------------
# TestWorstVerdict
# ---------------------------------------------------------------------------


class TestWorstVerdict:
    def test_pass_only(self):
        assert _worst_verdict(["PASS", "PASS", "PASS"]) == "PASS"

    def test_warn_beats_pass(self):
        assert _worst_verdict(["PASS", "WARN", "PASS"]) == "WARN"

    def test_fail_beats_all(self):
        assert _worst_verdict(["PASS", "WARN", "FAIL"]) == "FAIL"


# ---------------------------------------------------------------------------
# TestBuildTriageContext
# ---------------------------------------------------------------------------


class TestBuildTriageContext:
    def test_all_pass(self, tmp_path, monkeypatch):
        snap = tmp_path / "2026-02-27"
        snap.mkdir()
        _make_all(snap)
        # Stub thresholds so monitoring evaluate works without real file
        arts = collect_gate_artifacts(snap)
        # Patch _derive_monitoring_verdict to avoid needing thresholds file
        import scripts.build_daily_triage as mod
        monkeypatch.setattr(
            mod, "_derive_monitoring_verdict",
            lambda a: {"verdict": "PASS", "key_metrics": {"eligible_pct": 80.0}},
        )
        ctx = build_triage_context("2026-02-27", arts, git_sha="abc123")
        assert ctx["overall_verdict"] == "PASS"
        assert ctx["non_pass_gates"] == []
        assert ctx["gates"]["monitoring"]["verdict"] == "PASS"

    def test_warn_eligibility(self, tmp_path, monkeypatch):
        snap = tmp_path / "2026-02-27"
        snap.mkdir()
        _make_monitoring(snap)
        _make_eligibility_gate(snap, verdict="WARN")
        _make_eligibility_delta(snap)
        _make_ruleset_eval(snap)
        arts = collect_gate_artifacts(snap)
        import scripts.build_daily_triage as mod
        monkeypatch.setattr(
            mod, "_derive_monitoring_verdict",
            lambda a: {"verdict": "PASS", "key_metrics": {"eligible_pct": 80.0}},
        )
        ctx = build_triage_context("2026-02-27", arts)
        assert ctx["overall_verdict"] == "WARN"
        assert "eligibility" in ctx["non_pass_gates"]

    def test_fail_monitoring(self, tmp_path, monkeypatch):
        snap = tmp_path / "2026-02-27"
        snap.mkdir()
        _make_monitoring(snap)
        _make_eligibility_gate(snap)
        _make_eligibility_delta(snap)
        _make_ruleset_eval(snap)
        arts = collect_gate_artifacts(snap)
        import scripts.build_daily_triage as mod
        monkeypatch.setattr(
            mod, "_derive_monitoring_verdict",
            lambda a: {"verdict": "FAIL", "fail_reasons": ["eligible_pct too low"], "key_metrics": {}},
        )
        ctx = build_triage_context("2026-02-27", arts)
        assert ctx["overall_verdict"] == "FAIL"
        assert "monitoring" in ctx["non_pass_gates"]

    def test_degraded_skip(self, tmp_path, monkeypatch):
        snap = tmp_path / "2026-02-27"
        snap.mkdir()
        # No eligibility gate, only delta → DEGRADED_SKIP
        _make_monitoring(snap)
        _make_eligibility_delta(snap)
        _make_ruleset_eval(snap)
        arts = collect_gate_artifacts(snap)
        import scripts.build_daily_triage as mod
        monkeypatch.setattr(
            mod, "_derive_monitoring_verdict",
            lambda a: {"verdict": "PASS", "key_metrics": {}},
        )
        ctx = build_triage_context("2026-02-27", arts)
        assert ctx["gates"]["eligibility"]["verdict"] == "DEGRADED_SKIP"
        assert ctx["overall_verdict"] == "DEGRADED_SKIP"


# ---------------------------------------------------------------------------
# TestRenderSummaryMd
# ---------------------------------------------------------------------------


class TestRenderSummaryMd:
    def _make_ctx(self, overall="PASS", **gates):
        default_gates = {
            "monitoring": {"verdict": "PASS", "key_metrics": {"eligible_pct": 80.0}},
            "eligibility": {"verdict": "PASS", "triggers": [], "key_metrics": {"eligible_delta": -2}},
            "ruleset_eval": {"verdict": "PASS", "key_metrics": {"n_evaluated": 10, "mean_top60_overlap": 92.0}},
        }
        default_gates.update(gates)
        return {
            "schema": "daily_triage_context.v1",
            "as_of_date": "2026-02-27",
            "git_sha": "abc12345",
            "overall_verdict": overall,
            "non_pass_gates": [n for n, g in default_gates.items() if g["verdict"] != "PASS"],
            "gates": default_gates,
        }

    def test_includes_all_gates(self):
        ctx = self._make_ctx()
        md = render_summary_md(ctx, {})
        assert "monitoring" in md
        assert "eligibility" in md
        assert "ruleset_eval" in md
        assert "PASS" in md

    def test_includes_movers(self):
        ctx = self._make_ctx()
        arts = {
            "eligibility_triage_json": {
                "movers": [
                    {"ticker": "XYZ", "prior_eligible": 1, "cur_eligible": 0},
                    {"ticker": "ABC", "prior_eligible": 0, "cur_eligible": 1},
                ],
            },
        }
        md = render_summary_md(ctx, arts)
        assert "Eligibility Movers" in md
        assert "XYZ" in md

    def test_action_recommendation(self):
        ctx_fail = self._make_ctx(overall="FAIL")
        md = render_summary_md(ctx_fail, {})
        assert "BLOCKED" in md

        ctx_warn = self._make_ctx(overall="WARN")
        md = render_summary_md(ctx_warn, {})
        assert "Investigate" in md

        ctx_pass = self._make_ctx(overall="PASS")
        md = render_summary_md(ctx_pass, {})
        assert "Proceed" in md


# ---------------------------------------------------------------------------
# TestRenderHealthSummaryMd
# ---------------------------------------------------------------------------


class TestRenderHealthSummaryMd:
    def _make_ctx(self, overall="PASS"):
        return {
            "as_of_date": "2026-02-27",
            "overall_verdict": overall,
            "non_pass_gates": ["monitoring"] if overall != "PASS" else [],
            "gates": {
                "monitoring": {"verdict": overall},
                "eligibility": {"verdict": "PASS"},
                "ruleset_eval": {"verdict": "PASS"},
            },
        }

    def test_compact_format(self):
        ctx = self._make_ctx()
        md = render_health_summary_md(ctx)
        lines = md.strip().split("\n")
        # Should be compact — header + blank + verdict + blank + table header(3) + 3 rows + blank + pointer
        assert len(lines) <= 15
        assert "Daily Health Summary" in md

    def test_includes_verdict_icons(self):
        ctx = self._make_ctx(overall="WARN")
        md = render_health_summary_md(ctx)
        assert "⚠️" in md
        assert "WARN" in md


# ---------------------------------------------------------------------------
# TestAssembleBundle
# ---------------------------------------------------------------------------


class TestAssembleBundle:
    def test_writes_all_files(self, tmp_path, monkeypatch):
        snap = tmp_path / "2026-02-27"
        snap.mkdir()
        _make_all(snap)
        arts = collect_gate_artifacts(snap)
        import scripts.build_daily_triage as mod
        monkeypatch.setattr(
            mod, "_derive_monitoring_verdict",
            lambda a: {"verdict": "PASS", "key_metrics": {"eligible_pct": 80.0}},
        )
        ctx = build_triage_context("2026-02-27", arts)
        out = snap / "daily_triage"
        assemble_bundle(snap, out, arts, ctx)

        assert (out / "context.json").exists()
        assert (out / "summary.md").exists()
        assert (out / "health_summary.md").exists()

    def test_explains_populated(self, tmp_path, monkeypatch):
        snap = tmp_path / "2026-02-27"
        snap.mkdir()
        _make_all(snap)
        arts = collect_gate_artifacts(snap)
        import scripts.build_daily_triage as mod
        monkeypatch.setattr(
            mod, "_derive_monitoring_verdict",
            lambda a: {"verdict": "PASS", "key_metrics": {}},
        )
        ctx = build_triage_context("2026-02-27", arts)
        out = snap / "daily_triage"
        assemble_bundle(snap, out, arts, ctx)

        explains = out / "explains"
        assert (explains / "monitoring.json").exists()
        assert (explains / "eligibility_gate.json").exists()
        assert (explains / "eligibility_delta.json").exists()
        assert (explains / "ruleset_eval.json").exists()
        # Generated markdowns
        assert (explains / "monitoring.md").exists()
        assert (explains / "eligibility_delta.md").exists()
        assert (explains / "ruleset_eval.md").exists()

    def test_movers_copied(self, tmp_path, monkeypatch):
        snap = tmp_path / "2026-02-27"
        snap.mkdir()
        _make_all(snap)
        arts = collect_gate_artifacts(snap)
        import scripts.build_daily_triage as mod
        monkeypatch.setattr(
            mod, "_derive_monitoring_verdict",
            lambda a: {"verdict": "PASS", "key_metrics": {}},
        )
        ctx = build_triage_context("2026-02-27", arts)
        out = snap / "daily_triage"
        assemble_bundle(snap, out, arts, ctx)

        assert (out / "movers.csv").exists()
        content = (out / "movers.csv").read_text()
        assert "AAPL" in content


# ---------------------------------------------------------------------------
# TestCLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_success_exit_0(self, tmp_path, monkeypatch):
        snap = tmp_path / "2026-02-27"
        snap.mkdir()
        _make_all(snap)
        import scripts.build_daily_triage as mod
        monkeypatch.setattr(
            mod, "_derive_monitoring_verdict",
            lambda a: {"verdict": "PASS", "key_metrics": {}},
        )
        rc = main([
            "--as-of-date", "2026-02-27",
            "--snapshot-dir", str(tmp_path),
            "--git-sha", "test123",
        ])
        assert rc == 0
        assert (snap / "daily_triage" / "context.json").exists()

    def test_missing_snapshot_exit_1(self, tmp_path):
        rc = main([
            "--as-of-date", "2026-02-27",
            "--snapshot-dir", str(tmp_path),
        ])
        assert rc == 1
