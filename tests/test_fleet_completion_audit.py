"""Tests for tools/fleet_completion_audit.py."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


@pytest.fixture
def audit_mod():
    import tools.fleet_completion_audit as mod

    return importlib.reload(mod)


def test_build_audit_passes_on_repo(audit_mod):
    report = audit_mod.build_audit()
    assert report["schema"] == "fleet_completion_audit.v1"
    assert report["fail_count"] == 0, json.dumps(
        [c for c in report["checks"] if c.get("status") == "FAIL"], indent=2
    )
    assert report["overall"] == "PASS"


def test_check_cron_llm_free_flags_run_agent_direct(audit_mod, tmp_path, monkeypatch):
    cron = tmp_path / "tools" / "cron_evening_catchup.sh"
    cron.parent.mkdir(parents=True)
    cron.write_text('run_tool x "$PYTHON tools/run_agent_direct.py --agent ops"\n', encoding="utf-8")
    monkeypatch.setattr(audit_mod, "REPO", tmp_path)
    monkeypatch.setattr(audit_mod, "PRODUCTION_CRON_SCRIPTS", ["tools/cron_evening_catchup.sh"])

    findings = audit_mod.check_cron_llm_free()
    assert any(f["status"] == "FAIL" and "run_agent_direct" in f.get("detail", "") for f in findings)


def test_main_json_exit_code(audit_mod, monkeypatch):
    monkeypatch.setattr(
        audit_mod,
        "build_audit",
        lambda: {"overall": "PASS", "pass_count": 1, "fail_count": 0, "checks": []},
    )
    monkeypatch.setattr(sys, "argv", ["fleet_completion_audit.py", "--json"])
    assert audit_mod.main() == 0
