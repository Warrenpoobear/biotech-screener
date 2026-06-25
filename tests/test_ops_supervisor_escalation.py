"""Tests for ops_supervisor heartbeat escalation.json integration."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SUPERVISOR_PATH = REPO / "agents" / "ops_supervisor" / "supervisor.py"


def _load_supervisor():
    spec = importlib.util.spec_from_file_location("ops_supervisor_module", SUPERVISOR_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_parse_heartbeat_escalation_json(tmp_path):
    mod = _load_supervisor()
    path = tmp_path / "2026-06-24_escalation.json"
    path.write_text(
        json.dumps(
            {
                "agents": [
                    {
                        "agent": "qa",
                        "status": "WARN",
                        "detail": "schema drift",
                        "anomalies": ["ROW_COUNT_DRIFT"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    parsed = mod.parse_heartbeat_escalation_json(path)

    assert len(parsed) == 1
    assert parsed[0]["agent"] == "qa"
    assert parsed[0]["raw_status"] == "WARN"
    assert "ROW_COUNT_DRIFT" in parsed[0]["raw_text"]


def test_load_heartbeat_anomalies_prefers_json(tmp_path, monkeypatch):
    mod = _load_supervisor()
    hb_dir = tmp_path / "heartbeat"
    hb_dir.mkdir(parents=True)
    monkeypatch.setattr(mod, "HEARTBEAT_DIR", hb_dir)

    (hb_dir / "2026-06-24_escalation.json").write_text(
        json.dumps(
            {
                "agents": [
                    {
                        "agent": "ic_health_monitor",
                        "status": "FAIL",
                        "detail": "attention=HIGH",
                        "anomalies": ["SIGNAL_ALERT: foo"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (hb_dir / "2026-06-24_anomalies.md").write_text(
        "## [qa] WARN — old md only\n- STALE\n",
        encoding="utf-8",
    )

    anomalies, status = mod.load_heartbeat_anomalies("2026-06-24")

    assert status["heartbeat_escalation_json"] == "found"
    assert len(anomalies) == 1
    assert anomalies[0]["agent"] == "ic_health_monitor"


def test_load_heartbeat_anomalies_falls_back_to_md(tmp_path, monkeypatch):
    mod = _load_supervisor()
    hb_dir = tmp_path / "heartbeat"
    hb_dir.mkdir(parents=True)
    monkeypatch.setattr(mod, "HEARTBEAT_DIR", hb_dir)

    (hb_dir / "2026-06-24_anomalies.md").write_text(
        "## [qa] WARN — detail\n- CODE: issue\n",
        encoding="utf-8",
    )

    anomalies, status = mod.load_heartbeat_anomalies("2026-06-24")

    assert status["heartbeat_escalation_json"] == "missing"
    assert anomalies[0]["agent"] == "qa"
