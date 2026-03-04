"""
Tests for scripts/research/submit_research.py

10 tests covering:
  - Baseline auto-discovery from manifest + audited backtests
  - Correct defaults forwarded to run_audited_backtest
  - Exit codes match verdict (0=PROMOTE, 1=ARCHIVE, 2=NEEDS_MORE)
  - Missing baseline → NEEDS_MORE
  - RESEARCH_WORKFLOW.md exists
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.research.submit_research import (
    _find_active_ruleset_id,
    _find_baseline_summary,
    _VERDICT_EXIT,
    submit,
    _DEFAULT_HORIZONS,
    _DEFAULT_TOP_K,
    _DEFAULT_COST_BPS,
    _DEFAULT_ANCHOR_MODE,
    _DEFAULT_BENCHMARK,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_manifest(tmp_path: Path, entries: list) -> Path:
    manifest = {"schema_version": 1, "rulesets": entries}
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest))
    return p


def _write_verdict(run_dir: Path, ruleset_id: str, verdict: str = "PROMOTE") -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "eval").mkdir(exist_ok=True)
    vdata = {"schema": "verdict.v1", "ruleset_id": ruleset_id, "verdict": verdict}
    vpath = run_dir / "VERDICT.json"
    vpath.write_text(json.dumps(vdata))
    summary = run_dir / "eval" / "summary.json"
    summary.write_text(json.dumps({"n_evaluated": 50, "by_horizon": {}}))
    return vpath


# ---------------------------------------------------------------------------
# 1. Baseline auto-discovery
# ---------------------------------------------------------------------------

class TestBaselineDiscovery:

    def test_auto_discover_from_active_ruleset(self, tmp_path):
        """Auto-discovery finds baseline summary when active ruleset has a VERDICT.json."""
        manifest_path = _write_manifest(tmp_path, [
            {"id": "abc12345", "file": "active.json", "status": "active"},
        ])
        audited_root = tmp_path / "audited"
        _write_verdict(audited_root / "baseline_run", "abc12345")

        with (
            patch("scripts.research.submit_research._MANIFEST_PATH", manifest_path),
            patch("scripts.research.submit_research._AUDITED_ROOT", audited_root),
        ):
            summary = _find_baseline_summary()

        assert summary is not None
        assert summary.name == "summary.json"
        assert summary.is_file()

    def test_explicit_baseline_dir(self, tmp_path):
        """Explicit --baseline-dir overrides auto-discovery."""
        run_dir = tmp_path / "my_run"
        run_dir.mkdir()
        (run_dir / "eval").mkdir()
        summary = run_dir / "eval" / "summary.json"
        summary.write_text(json.dumps({"n_evaluated": 40}))

        result = _find_baseline_summary(baseline_dir=run_dir)
        assert result == summary

    def test_missing_baseline_dir_returns_none(self, tmp_path):
        """Non-existent explicit baseline dir → returns None."""
        result = _find_baseline_summary(baseline_dir=tmp_path / "nonexistent")
        assert result is None

    def test_no_active_ruleset_returns_none(self, tmp_path):
        """No active ruleset in manifest → auto-discovery returns None."""
        manifest_path = _write_manifest(tmp_path, [
            {"id": "abc12345", "file": "old.json", "status": "retired"},
        ])
        audited_root = tmp_path / "audited"
        audited_root.mkdir()

        with (
            patch("scripts.research.submit_research._MANIFEST_PATH", manifest_path),
            patch("scripts.research.submit_research._AUDITED_ROOT", audited_root),
        ):
            result = _find_baseline_summary()
        assert result is None

    def test_active_ruleset_id_found(self, tmp_path):
        """_find_active_ruleset_id() returns correct ID."""
        manifest_path = _write_manifest(tmp_path, [
            {"id": "retired_id", "file": "old.json", "status": "retired"},
            {"id": "live_id", "file": "active.json", "status": "active"},
        ])
        with patch("scripts.research.submit_research._MANIFEST_PATH", manifest_path):
            active_id = _find_active_ruleset_id()
        assert active_id == "live_id"


# ---------------------------------------------------------------------------
# 2. Correct defaults forwarded
# ---------------------------------------------------------------------------

class TestDefaultsForwarded:

    def test_production_defaults_used(self, tmp_path):
        """submit() passes production defaults to run_audited_backtest."""
        ruleset = tmp_path / "candidate.json"
        ruleset.write_text(json.dumps({"version": "test"}))

        with patch("scripts.research.submit_research.run_audited_backtest") as mock_run:
            mock_run.return_value = 0
            # Also need to mock verdict loading
            with patch("scripts.research.submit_research._find_baseline_summary", return_value=None):
                submit(
                    ruleset_path=ruleset,
                    name="test_run",
                    baseline_dir=None,
                    out_root=tmp_path,
                )

        assert mock_run.called
        _, kwargs = mock_run.call_args
        assert kwargs["horizons"] == _DEFAULT_HORIZONS
        assert kwargs["top_k"] == _DEFAULT_TOP_K
        assert kwargs["cost_bps"] == _DEFAULT_COST_BPS
        assert kwargs["anchor_mode"] == _DEFAULT_ANCHOR_MODE
        assert kwargs["benchmark"] == _DEFAULT_BENCHMARK
        assert kwargs["rerank"] is True
        assert kwargs["preflight_strict"] is True

    def test_run_id_derived_from_name(self, tmp_path):
        """submit() uses name (lowercase, underscored) as run_id."""
        ruleset = tmp_path / "candidate.json"
        ruleset.write_text(json.dumps({}))

        with patch("scripts.research.submit_research.run_audited_backtest") as mock_run:
            mock_run.return_value = 0
            with patch("scripts.research.submit_research._find_baseline_summary", return_value=None):
                submit(ruleset_path=ruleset, name="My Experiment v2", out_root=tmp_path)

        _, kwargs = mock_run.call_args
        assert kwargs["run_id"] == "my_experiment_v2"


# ---------------------------------------------------------------------------
# 3. Exit codes match verdict
# ---------------------------------------------------------------------------

class TestExitCodes:

    def _make_run_with_verdict(self, tmp_path: Path, verdict: str) -> Path:
        """Create a fake run directory with a VERDICT.json."""
        run_dir = tmp_path / "test_run"
        run_dir.mkdir()
        vdata = {"schema": "verdict.v1", "verdict": verdict, "verdict_reasons": []}
        (run_dir / "VERDICT.json").write_text(json.dumps(vdata))
        (run_dir / "VERDICT.md").write_text(f"# {verdict}\n")
        return run_dir

    def test_promote_verdict_exit_0(self, tmp_path):
        self._make_run_with_verdict(tmp_path, "PROMOTE")
        ruleset = tmp_path / "c.json"
        ruleset.write_text("{}")
        with (
            patch("scripts.research.submit_research.run_audited_backtest", return_value=0),
            patch("scripts.research.submit_research._find_baseline_summary", return_value=None),
            patch("scripts.research.submit_research._DEFAULT_OUT_ROOT", tmp_path),
        ):
            code = submit(ruleset_path=ruleset, name="test_run", out_root=tmp_path)
        assert code == 0

    def test_archive_verdict_exit_1(self, tmp_path):
        self._make_run_with_verdict(tmp_path, "ARCHIVE")
        ruleset = tmp_path / "c.json"
        ruleset.write_text("{}")
        with (
            patch("scripts.research.submit_research.run_audited_backtest", return_value=0),
            patch("scripts.research.submit_research._find_baseline_summary", return_value=None),
        ):
            code = submit(ruleset_path=ruleset, name="test_run", out_root=tmp_path)
        assert code == 1

    def test_needs_more_verdict_exit_2(self, tmp_path):
        self._make_run_with_verdict(tmp_path, "NEEDS_MORE")
        ruleset = tmp_path / "c.json"
        ruleset.write_text("{}")
        with (
            patch("scripts.research.submit_research.run_audited_backtest", return_value=0),
            patch("scripts.research.submit_research._find_baseline_summary", return_value=None),
        ):
            code = submit(ruleset_path=ruleset, name="test_run", out_root=tmp_path)
        assert code == 2


# ---------------------------------------------------------------------------
# 4. Missing baseline → NEEDS_MORE
# ---------------------------------------------------------------------------

class TestMissingBaseline:

    def test_no_baseline_still_runs_backtest(self, tmp_path):
        """Missing baseline causes no crash — backtest runs with baseline_summary_path=None."""
        ruleset = tmp_path / "c.json"
        ruleset.write_text("{}")
        with (
            patch("scripts.research.submit_research.run_audited_backtest") as mock_run,
            patch("scripts.research.submit_research._find_baseline_summary", return_value=None),
        ):
            mock_run.return_value = 0
            submit(ruleset_path=ruleset, name="no_baseline_run", out_root=tmp_path)
        _, kwargs = mock_run.call_args
        assert kwargs["baseline_summary_path"] is None


# ---------------------------------------------------------------------------
# 5. RESEARCH_WORKFLOW.md exists
# ---------------------------------------------------------------------------

class TestResearchWorkflowDoc:

    def test_workflow_doc_exists(self):
        """RESEARCH_WORKFLOW.md is present in scripts/research/."""
        doc = Path(_project_root) / "scripts" / "research" / "RESEARCH_WORKFLOW.md"
        assert doc.is_file(), f"RESEARCH_WORKFLOW.md not found at {doc}"

    def test_workflow_doc_has_key_sections(self):
        """RESEARCH_WORKFLOW.md covers submit, verdict, and promote steps."""
        doc = Path(_project_root) / "scripts" / "research" / "RESEARCH_WORKFLOW.md"
        content = doc.read_text()
        assert "submit_research.py" in content
        assert "VERDICT" in content
        assert "promote" in content.lower()
