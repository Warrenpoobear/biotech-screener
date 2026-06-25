"""Static checks for tools/install_agent_fleet_crontab.sh."""

from __future__ import annotations

from pathlib import Path

INSTALL = Path(__file__).resolve().parent.parent / "tools" / "install_agent_fleet_crontab.sh"


def test_install_script_includes_heartbeat_and_watchdog():
    text = INSTALL.read_text(encoding="utf-8")
    assert "agent_heartbeat_checks.py" in text
    assert "logs/heartbeat_checks.log" in text
    assert "cron_watchdog.sh" in text
    assert "logs/watchdog.log" in text
    assert "fleet_ops_status.py" in text
    assert "fleet_completion_audit.py" in text
    assert "fleet_crontab_verify.py" in text


def test_install_script_has_evening_catchup_and_herald_health():
    text = INSTALL.read_text(encoding="utf-8")
    assert "cron_evening_catchup.sh" in text
    assert "herald_health_check.py" in text
