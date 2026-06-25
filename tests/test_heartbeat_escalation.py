"""Tests for heartbeat artifact escalation (phase 4)."""

from __future__ import annotations

import importlib
import json
import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


@pytest.fixture
def hb_mod(tmp_path, monkeypatch):
    import tools.agent_heartbeat_checks as mod

    mod = importlib.reload(mod)
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "ARTIFACTS_DIR", tmp_path / "artifacts")
    (tmp_path / "artifacts" / "heartbeat").mkdir(parents=True)
    return mod


def _anomaly_result(hb_mod, agent="qa", status="WARN"):
    return hb_mod.CheckResult(agent, status, "test detail", ["snapshot drift"])


def test_escalate_anomalies_default_is_artifact_only(hb_mod, monkeypatch):
    monkeypatch.delenv("HEARTBEAT_LLM_ESCALATE", raising=False)
    dt = date.fromisoformat("2026-06-24")
    results = [_anomaly_result(hb_mod)]

    with patch("subprocess.run") as mock_run:
        hb_mod.escalate_anomalies(results, dry_run=False, dt=dt)

    mock_run.assert_not_called()

    ds = "2026-06-24"
    anomalies = hb_mod.ARTIFACTS_DIR / "heartbeat" / f"{ds}_anomalies.md"
    escalation = hb_mod.ARTIFACTS_DIR / "heartbeat" / f"{ds}_escalation.json"
    assert anomalies.is_file()
    assert escalation.is_file()
    payload = json.loads(escalation.read_text())
    assert payload["mode"] == "artifact_only"
    assert payload["llm_requested"] is False
    assert payload["agent_count"] == 1


def test_escalate_anomalies_llm_when_env_set(hb_mod, monkeypatch):
    monkeypatch.setenv("HEARTBEAT_LLM_ESCALATE", "1")
    dt = date.fromisoformat("2026-06-24")
    results = [_anomaly_result(hb_mod)]

    mock_proc = MagicMock()
    mock_proc.stdout = "ops triage complete"
    mock_proc.returncode = 0

    with patch("subprocess.run", return_value=mock_proc) as mock_run:
        hb_mod.escalate_anomalies(results, dry_run=False, dt=dt)

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert any("run_agent_direct.py" in str(part) for part in cmd)
    assert "ops" in cmd

    payload = json.loads(
        (hb_mod.ARTIFACTS_DIR / "heartbeat" / "2026-06-24_escalation.json").read_text()
    )
    assert payload["mode"] == "llm"
    assert payload["llm_requested"] is True
    assert payload["llm_status"] == "ok"


def test_escalate_anomalies_dry_run_skips_subprocess(hb_mod, monkeypatch):
    monkeypatch.setenv("HEARTBEAT_LLM_ESCALATE", "1")
    dt = date.fromisoformat("2026-06-24")

    with patch("subprocess.run") as mock_run:
        hb_mod.escalate_anomalies([_anomaly_result(hb_mod)], dry_run=True, dt=dt)

    mock_run.assert_not_called()
    payload = json.loads(
        (hb_mod.ARTIFACTS_DIR / "heartbeat" / "2026-06-24_escalation.json").read_text()
    )
    assert payload["mode"] == "dry_run"
    assert payload["llm_status"] == "skipped_dry_run"


def test_escalate_anomalies_skips_when_no_actionable_anomalies(hb_mod, tmp_path):
    carried = hb_mod.CheckResult(
        "ic_health_monitor",
        "WARN",
        "attention=HIGH",
        ["[CARRIED] SIGNAL_ALERT: inst_delta_z (expected)"],
    )
    dt = date.fromisoformat("2026-06-24")

    hb_mod.escalate_anomalies([carried], dry_run=False, dt=dt)

    assert not (hb_mod.ARTIFACTS_DIR / "heartbeat" / "2026-06-24_escalation.json").exists()
