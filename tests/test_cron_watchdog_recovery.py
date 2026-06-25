"""Static checks for tools/cron_watchdog.sh recovery paths."""

from __future__ import annotations

from pathlib import Path

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


def test_watchdog_phase2_checks_artifacts_not_agents_direct():
    text = WATCHDOG.read_text(encoding="utf-8")
    assert "artifacts/price_action_watch/" in text
    code_lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    assert "agents_direct" not in "\n".join(code_lines)
