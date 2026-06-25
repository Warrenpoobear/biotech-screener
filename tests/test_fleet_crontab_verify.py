"""Tests for tools/fleet_crontab_verify.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


@pytest.fixture
def verify_mod():
    import tools.fleet_crontab_verify as mod

    return mod


def test_verify_skips_when_crontab_unavailable(verify_mod, monkeypatch):
    monkeypatch.setattr(verify_mod, "read_active_crontab_lines", lambda: ("UNAVAILABLE", []))
    report = verify_mod.verify_crontab()
    assert report["overall"] == "SKIP"
    assert report["skip_count"] == len(verify_mod.FLEET_CRON_JOBS)


def test_verify_passes_when_all_jobs_present(verify_mod, monkeypatch):
    lines = [
        "40 14 * * 1-5 cd /repo && python3 tools/herald_health_check.py",
        "30 10 * * 1-5 /repo/tools/cron_intraday_mover.sh poll",
        "30 19 * * 5 cd /repo && bash tools/cron_weekly_skills_review.sh",
        "30 17 * * 1-5 cd /repo && bash tools/cron_data_auditor.sh",
        "30 17 * * 1-5 cd /repo && python3 tools/agent_heartbeat_checks.py",
        "0 18 * * 1-5 cd /repo && python3 tools/build_hermes_knowledge_layer.py",
        "5 18 * * 1-5 cd /repo && python3 agents/hermes-contradiction-detector/run_job.py",
        "0 22 * * 1-5 /repo/tools/cron_evening_catchup.sh",
        "@reboot /repo/tools/cron_watchdog.sh",
        "30 12 * * 1-5 /repo/tools/cron_watchdog.sh",
    ]
    monkeypatch.setattr(verify_mod, "read_active_crontab_lines", lambda: ("OPERATOR_HOST", lines))
    report = verify_mod.verify_crontab()
    assert report["overall"] == "PASS"
    assert report["fail_count"] == 0


def test_verify_fails_on_missing_watchdog_reboot(verify_mod, monkeypatch):
    lines = [
        "30 12 * * 1-5 /repo/tools/cron_watchdog.sh",
        "40 14 * * 1-5 python3 tools/herald_health_check.py",
        "30 10 * * 1-5 /repo/tools/cron_intraday_mover.sh poll",
        "30 19 * * 5 bash tools/cron_weekly_skills_review.sh",
        "30 17 * * 1-5 bash tools/cron_data_auditor.sh",
        "30 17 * * 1-5 python3 tools/agent_heartbeat_checks.py",
        "0 18 * * 1-5 python3 tools/build_hermes_knowledge_layer.py",
        "5 18 * * 1-5 agents/hermes-contradiction-detector/run_job.py",
        "0 22 * * 1-5 /repo/tools/cron_evening_catchup.sh",
    ]
    monkeypatch.setattr(verify_mod, "read_active_crontab_lines", lambda: ("OPERATOR_HOST", lines))
    report = verify_mod.verify_crontab()
    assert report["overall"] == "FAIL"
    failed = [c["job"] for c in report["checks"] if c["status"] == "FAIL"]
    assert "watchdog_reboot" in failed


def test_main_writes_artifact(verify_mod, tmp_path, monkeypatch):
    monkeypatch.setattr(verify_mod, "OUT_DIR", tmp_path / "artifacts" / "fleet_ops")
    monkeypatch.setattr(verify_mod, "read_active_crontab_lines", lambda: ("UNAVAILABLE", []))
    monkeypatch.setattr(sys, "argv", ["fleet_crontab_verify.py", "--write", "--as-of-date", "2026-06-24"])

    assert verify_mod.main() == 0
    out = tmp_path / "artifacts" / "fleet_ops" / "2026-06-24_crontab_verify.json"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "fleet_crontab_verify.v1"
    assert payload["overall"] == "SKIP"
