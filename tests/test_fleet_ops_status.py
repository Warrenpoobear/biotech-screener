"""Tests for tools/fleet_ops_status.py."""

from __future__ import annotations

import importlib
import json
import sys
from datetime import date
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


@pytest.fixture
def fleet_mod(tmp_path, monkeypatch):
    import tools.fleet_ops_status as mod

    mod = importlib.reload(mod)
    monkeypatch.setattr(mod, "REPO", tmp_path)
    (tmp_path / "artifacts" / "heartbeat").mkdir(parents=True)
    (tmp_path / "logs").mkdir()
    (tmp_path / "data" / "snapshots" / "2026-06-24").mkdir(parents=True)
    return mod


def test_build_status_includes_heartbeat_receipt_verdict(fleet_mod, tmp_path, monkeypatch):
    ds = "2026-06-24"
    receipt = tmp_path / "artifacts" / "heartbeat" / f"{ds}_receipt.md"
    receipt.write_text("# Fleet Receipt\nVerdict: YELLOW\n", encoding="utf-8")
    (tmp_path / "data" / "snapshots" / ds / "rankings.csv").write_text("ticker\n", encoding="utf-8")

    monkeypatch.setattr(
        "tools.herald_health_check.run_check",
        lambda _as_of: {
            "verdict": "HEALTHY",
            "herald_done": True,
            "latest_classified_date": ds,
            "source_age_days": 0,
            "issues": [],
        },
    )

    report = fleet_mod.build_status(date.fromisoformat(ds))
    assert report["heartbeat"]["verdict"] == "YELLOW"
    assert report["snapshot"]["exists"] is True
    assert report["schema"] == "fleet_ops_status.v1"
    assert any(loop["id"] == "F-2026-005" for loop in report["stalled_loops"])


def test_build_status_json_serializable(fleet_mod, monkeypatch):
    monkeypatch.setattr(
        "tools.herald_health_check.run_check",
        lambda _as_of: {
            "verdict": "WARN",
            "herald_done": False,
            "latest_classified_date": None,
            "source_age_days": 10,
            "issues": ["STALE_SOURCE"],
        },
    )
    report = fleet_mod.build_status(date.fromisoformat("2026-06-24"))
    json.dumps(report)
    assert report["overall"] in ("WARN", "FAIL")


def test_main_exit_code_warn_on_open_stalled_loops(fleet_mod, monkeypatch):
    monkeypatch.setattr(
        fleet_mod,
        "build_status",
        lambda _as_of=None: {
            "schema": "fleet_ops_status.v1",
            "as_of_date": "2026-06-24",
            "overall": "WARN",
            "herald": {},
            "heartbeat": {},
            "snapshot": {},
            "fleet_jobs": {},
            "stalled_loops": [],
            "crontab_install": "bash tools/install_agent_fleet_crontab.sh",
            "crontab_hints": [],
            "rule_12_checklist": "docs/governance/RULE_12_PROMOTION_CHECKLIST.md",
        },
    )
    monkeypatch.setattr(sys, "argv", ["fleet_ops_status.py", "--json"])
    assert fleet_mod.main() == 1
