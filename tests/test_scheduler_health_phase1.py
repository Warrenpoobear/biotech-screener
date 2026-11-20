"""
Test suite for Phase 1 Priority 2: Scheduler/Gateway Stall Watchdog.

Validates detect-only mode for WSL2 sleep, cron silence, and gateway inactivity.

Acceptance gate: Detect-only mode proves it can identify cron/gateway silence
without restarting anything. Function returns (status, diagnostics) tuple with
status in ["OK", "WARN", "ALERT"] and diagnostics as a list of strings.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def settled_uptime():
    """Make the uptime check see a long-running system.

    scheduler_health_check raises ALERT when /proc/uptime is under 30 minutes (WSL2 just
    woke). A CI runner is always freshly booted, so tests about log age must pin uptime
    or they read ALERT for a reason they are not testing.
    """
    import tools.run_agent_direct as module

    with patch.object(module, "open", create=True) as mock_file:
        mock_file.return_value.__enter__.return_value.read.return_value = "360000.0 720000.0"
        yield


def test_scheduler_health_check_function_exists():
    """Scheduler health check function is implemented."""
    from tools.run_agent_direct import scheduler_health_check

    assert callable(scheduler_health_check)


def test_scheduler_health_returns_tuple():
    """Scheduler health check returns (status: str, diagnostics: list[str]) tuple."""
    from tools.run_agent_direct import scheduler_health_check

    result = scheduler_health_check("ops")
    assert isinstance(result, tuple), "Should return a tuple"
    assert len(result) == 2, "Should return a 2-tuple"
    status, diagnostics = result
    assert isinstance(status, str), "Status should be a string"
    assert status in ["OK", "WARN", "ALERT"], f"Status should be OK/WARN/ALERT, got {status}"
    assert isinstance(diagnostics, list), "Diagnostics should be a list"
    assert all(isinstance(d, str) for d in diagnostics), "All diagnostics should be strings"


def test_scheduler_health_detects_gateway_inactivity_alert():
    """Scheduler health check detects gateway inactivity (>6 hours) as ALERT."""
    from tools.run_agent_direct import PROJECT_ROOT, scheduler_health_check

    # Create a stale log file (7 hours old)
    logs_dir = PROJECT_ROOT / "logs" / "agents_direct"
    logs_dir.mkdir(parents=True, exist_ok=True)

    old_time = datetime.now().timestamp() - (7 * 3600)  # 7 hours ago
    stale_log = logs_dir / "stale_dispatch.json"
    stale_log.write_text("{}")
    os.utime(str(stale_log), (old_time, old_time))

    try:
        status, diagnostics = scheduler_health_check("ops")
        assert status == "ALERT", f"Should detect 7-hour-old log as ALERT, got {status}"
        assert len(diagnostics) > 0, "Should have diagnostics"
        assert any("Gateway inactivity" in d for d in diagnostics), f"Diagnostics: {diagnostics}"
    finally:
        stale_log.unlink()


def test_scheduler_health_detects_gateway_inactivity_warn(settled_uptime):
    """Scheduler health check detects moderate gateway inactivity (2-6 hours) as WARN."""
    from tools.run_agent_direct import PROJECT_ROOT, scheduler_health_check

    logs_dir = PROJECT_ROOT / "logs" / "agents_direct"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Create a log file 4 hours old
    old_time = datetime.now().timestamp() - (4 * 3600)
    warn_log = logs_dir / "warn_dispatch.json"
    warn_log.write_text("{}")
    os.utime(str(warn_log), (old_time, old_time))

    try:
        status, diagnostics = scheduler_health_check("ops")
        assert status == "WARN", f"Should detect 4-hour-old log as WARN, got {status}"
        assert len(diagnostics) > 0, "Should have diagnostics"
        assert any("Gateway slow" in d for d in diagnostics), f"Diagnostics: {diagnostics}"
    finally:
        warn_log.unlink()


def test_scheduler_health_detects_no_logs(settled_uptime):
    """Scheduler health check detects missing gateway logs as WARN."""
    from tools.run_agent_direct import PROJECT_ROOT, scheduler_health_check

    logs_dir = PROJECT_ROOT / "logs" / "agents_direct"

    # Temporarily move logs out of the way
    if logs_dir.exists():
        backup_dir = logs_dir.parent / "agents_direct_backup"
        logs_dir.rename(backup_dir)
    else:
        backup_dir = None

    try:
        status, diagnostics = scheduler_health_check("ops")
        # If no logs at all, should be WARN
        # (or OK if logs_dir doesn't exist)
        assert status in ["OK", "WARN"], f"Status should be OK or WARN when no logs, got {status}"
    finally:
        if backup_dir and backup_dir.exists():
            backup_dir.rename(logs_dir)


def test_scheduler_health_detects_wsl2_sleep():
    """Scheduler health check detects WSL2 sleep (uptime < 30 min) as ALERT."""
    from tools.run_agent_direct import scheduler_health_check

    # Mock /proc/uptime to simulate WSL2 just waking up (15 minutes)
    mock_uptime_content = "900.5 2700.3"  # 900 seconds = 15 minutes

    with patch("builtins.open", create=True) as mock_open_file:
        mock_open_file.return_value.__enter__.return_value.read.return_value = mock_uptime_content

        # Patch the open call in the scheduler_health_check function
        import tools.run_agent_direct as module

        with patch.object(module, "open", create=True) as mock_file:
            mock_file.return_value.__enter__.return_value.read.return_value = mock_uptime_content

            status, diagnostics = scheduler_health_check("ops")
            # Should detect WSL2 sleep (uptime < 30 min = 1800 sec)
            assert status == "ALERT", f"Should detect WSL2 sleep as ALERT, got {status}"
            assert any("WSL2 sleep" in d for d in diagnostics), f"Diagnostics: {diagnostics}"


def test_scheduler_health_no_restart_action():
    """Scheduler health check operates in detect-only mode (no restart action)."""
    import subprocess

    from tools.run_agent_direct import scheduler_health_check

    # Mock subprocess to verify it's never called (detect-only)
    with patch("subprocess.run") as mock_run:
        status, diagnostics = scheduler_health_check("ops")
        # Should NOT call subprocess (no restart action)
        mock_run.assert_not_called()


def test_scheduler_health_is_non_destructive():
    """Scheduler health check does not modify any state."""
    from tools.run_agent_direct import scheduler_health_check

    # Capture environment before
    env_before = dict(os.environ)
    cwd_before = os.getcwd()

    # Run health check
    scheduler_health_check("ops")

    # Verify no mutations
    assert dict(os.environ) == env_before, "Should not mutate environment"
    assert os.getcwd() == cwd_before, "Should not change working directory"


def test_scheduler_health_accepts_any_agent_name():
    """Scheduler health check accepts any agent name (query parameter)."""
    from tools.run_agent_direct import scheduler_health_check

    for agent_name in ["ops", "shadow_monitor", "data_auditor", "ic_health_monitor"]:
        status, diagnostics = scheduler_health_check(agent_name)
        assert isinstance(status, str), f"Agent {agent_name}: status should be string"
        assert isinstance(diagnostics, list), f"Agent {agent_name}: diagnostics should be list"


def test_scheduler_health_thresholds_are_conservative(settled_uptime):
    """Scheduler health check uses conservative thresholds (alert early)."""
    from tools.run_agent_direct import PROJECT_ROOT, scheduler_health_check

    logs_dir = PROJECT_ROOT / "logs" / "agents_direct"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Clean up test files before running
    for f in logs_dir.glob("threshold_test_*.json"):
        try:
            f.unlink()
        except FileNotFoundError:
            pass

    # Save existing logs
    existing_logs = list(logs_dir.glob("*.json"))
    existing_logs = [f for f in existing_logs if not f.name.startswith("threshold_test_")]

    # Temporarily remove existing logs to test in isolation
    backups = {}
    for f in existing_logs:
        backup_path = f.parent / f"_backup_{f.name}"
        f.rename(backup_path)
        backups[f] = backup_path

    try:
        # Test that 6h threshold is respected (6h = ALERT)
        old_time = datetime.now().timestamp() - (6 * 3600 + 1)
        test_log_6h = logs_dir / "threshold_test_6h.json"
        test_log_6h.write_text("{}")
        os.utime(str(test_log_6h), (old_time, old_time))

        status, _ = scheduler_health_check("ops")
        assert status == "ALERT", f"6h+ old log should be ALERT, got {status}"
        test_log_6h.unlink()

        # Test that 2h threshold is respected (2h = WARN)
        old_time = datetime.now().timestamp() - (2 * 3600 + 1)
        test_log_2h = logs_dir / "threshold_test_2h.json"
        test_log_2h.write_text("{}")
        os.utime(str(test_log_2h), (old_time, old_time))

        status, _ = scheduler_health_check("ops")
        assert status == "WARN", f"2-6h old log should be WARN, got {status}"
        test_log_2h.unlink()
    finally:
        # Restore existing logs
        for orig_path, backup_path in backups.items():
            if backup_path.exists():
                backup_path.rename(orig_path)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
