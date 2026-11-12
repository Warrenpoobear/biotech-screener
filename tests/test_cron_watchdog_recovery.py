"""Static checks for tools/cron_watchdog.sh recovery paths."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

WATCHDOG = Path(__file__).resolve().parent.parent / "tools" / "cron_watchdog.sh"


def test_watchdog_phase2_uses_deterministic_builders():
    text = WATCHDOG.read_text(encoding="utf-8")
    assert "run_agent_direct.py" not in text
    assert "build_price_action_watch.py" in text
    assert "build_options_watch.py" in text
    assert "run_postmortem.py" in text
    assert "run_review_queue_steward.py" in text


def test_watchdog_evening_delegates_to_catchup():
    text = WATCHDOG.read_text(encoding="utf-8")
    assert "cron_evening_catchup.sh" in text


def test_watchdog_recovers_heartbeat_and_ops_supervisor():
    text = WATCHDOG.read_text(encoding="utf-8")
    assert "artifacts/heartbeat/${TODAY}_receipt.md" in text
    assert "agent_heartbeat_checks.py" in text
    assert "ops_supervisor/supervisor.py" in text


def test_watchdog_recovers_herald_health_on_fail():
    text = WATCHDOG.read_text(encoding="utf-8")
    assert "artifacts/herald/health_check_${TODAY}.json" in text
    assert "herald_health_check.py" in text
    assert "--recover" in text
    assert "recovery_done_${TODAY}.complete" in text
    assert "F-2026-005" in text


def _run_trapops_block(tmp_path: Path, snapshot_promoted: bool) -> tuple[bool, str]:
    """Execute only the watchdog's trapops block against a temp repo.

    Returns (trapops_monitor_invoked, watchdog_log_text).
    """
    text = WATCHDOG.read_text(encoding="utf-8")
    start = text.index('TRAPOPS_LOG="$REPO/logs/trapops_cron.log"')
    end = text.index("# Evening cron recovery")
    block = text[start:end]

    repo = tmp_path / "repo"
    (repo / "logs").mkdir(parents=True)
    (repo / "data" / "snapshots").mkdir(parents=True)
    if snapshot_promoted:
        snap = repo / "data" / "snapshots" / "2026-11-12"
        snap.mkdir()
        (snap / "rankings.csv").write_text("ticker\n", encoding="utf-8")
    marker = tmp_path / "trapops_called"
    stub = tmp_path / "python_stub.sh"
    stub.write_text(f'#!/bin/bash\ntouch "{marker}"\n', encoding="utf-8")
    stub.chmod(0o755)
    log_file = tmp_path / "watchdog.log"

    script = f'REPO="{repo}"\nTODAY=2026-11-12\nPYTHON="{stub}"\n' f'log() {{ echo "$*" >> "{log_file}"; }}\n' + block
    subprocess.run(["bash", "-c", script], check=True)
    return marker.exists(), log_file.read_text(encoding="utf-8") if log_file.exists() else ""


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash required")
def test_trapops_not_recovered_before_snapshot_exists(tmp_path):
    # 2026-11-12: watchdog ran at 08:09/08:30, before the 16:30 production run
    # created today's snapshot dir, and logged ~50 "MISSED trapops" + FAILED
    # recoveries per day (trapops_monitor cannot write into a missing dir).
    called, log = _run_trapops_block(tmp_path, snapshot_promoted=False)
    assert not called
    assert "MISSED trapops" not in log
    assert "FAILED" not in log


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash required")
def test_trapops_recovered_when_snapshot_promoted_but_artifact_missing(tmp_path):
    called, log = _run_trapops_block(tmp_path, snapshot_promoted=True)
    assert called
    assert "MISSED trapops artifact for 2026-11-12" in log


def test_watchdog_phase2_checks_artifacts_not_agents_direct():
    text = WATCHDOG.read_text(encoding="utf-8")
    assert "artifacts/price_action_watch/" in text
    code_lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    assert "agents_direct" not in "\n".join(code_lines)
