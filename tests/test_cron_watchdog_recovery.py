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


def _run_gate_fail_block(tmp_path: Path, n_gate_fails: int, max_fails: str | None = None) -> tuple[str, int, str]:
    """Execute only the watchdog's gate-failure block against a temp repo.

    Returns (PROD_RAN after the block, number of notifications sent, watchdog log text).
    """
    text = WATCHDOG.read_text(encoding="utf-8")
    start = text.index("# A run that finished and was refused promotion by a gate is not a missed run.")
    end = text.index('    log "MISSED: No production run for $TODAY')
    block = text[start:end].rsplit('if [ "$PROD_RAN" = false ]; then', 1)[0]

    repo = tmp_path / "repo"
    (repo / "logs").mkdir(parents=True, exist_ok=True)
    daily = repo / "logs" / "daily_production_2026-11-13.log"
    lines = []
    for _ in range(n_gate_fails):
        lines.append("2026-11-13 16:47:16,607 - INFO - RESULT: FAIL — snapshot NOT promoted")
        lines.append("2026-11-13 16:47:16,608 - INFO - [FAIL] drift_monitoring: vs 2026-11-12: BLOCKED")
    daily.write_text("\n".join(lines) + "\n", encoding="utf-8")
    calls = tmp_path / "notify_calls"
    stub = tmp_path / "python_stub.sh"
    stub.write_text(f'#!/bin/bash\necho call >> "{calls}"\n', encoding="utf-8")
    stub.chmod(0o755)
    log_file = tmp_path / "watchdog.log"
    out = tmp_path / "prod_ran"
    env_line = f'WATCHDOG_MAX_GATE_FAILS="{max_fails}"\n' if max_fails else ""
    script = (
        f'REPO="{repo}"\nTODAY=2026-11-13\nPYTHON="{stub}"\nLOG="{log_file}"\nPROD_RAN=false\n'
        f'log() {{ echo "$*" >> "{log_file}"; }}\n' + env_line + block + f'\necho "$PROD_RAN" > "{out}"\n'
    )
    subprocess.run(["bash", "-c", script], check=True)
    n_calls = len(calls.read_text().splitlines()) if calls.exists() else 0
    return out.read_text().strip(), n_calls, log_file.read_text(encoding="utf-8") if log_file.exists() else ""


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash required")
def test_gate_failure_gets_one_retry(tmp_path):
    # One refused promotion: still treated as recoverable (the failure may be transient).
    prod_ran, calls, log = _run_gate_fail_block(tmp_path, n_gate_fails=1)
    assert prod_ran == "false"
    assert calls == 0
    assert "GATE_FAIL" not in log


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash required")
def test_repeated_gate_failure_stops_retrying_and_notifies_once(tmp_path):
    # 2026-11-13: the drift gate refused promotion three times; the watchdog re-ran the
    # whole pipeline every 30 minutes. After the second refusal it must stop and notify.
    prod_ran, calls, log = _run_gate_fail_block(tmp_path, n_gate_fails=2)
    assert prod_ran == "gatefail"
    assert calls == 1
    assert "GATE_FAIL" in log and "drift_monitoring" in log
    # A later watchdog pass the same day does not notify again.
    prod_ran, calls, _ = _run_gate_fail_block(tmp_path, n_gate_fails=3)
    assert prod_ran == "gatefail"
    assert calls == 1


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash required")
def test_no_daily_log_is_a_missed_run(tmp_path):
    prod_ran, calls, _ = _run_gate_fail_block(tmp_path, n_gate_fails=0)
    assert prod_ran == "false" and calls == 0


def test_gatefail_state_skips_recovery_branch():
    text = WATCHDOG.read_text(encoding="utf-8")
    assert '[ "$PROD_RAN" = "gatefail" ]' in text
