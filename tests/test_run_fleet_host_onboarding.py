"""Static checks for tools/run_fleet_host_onboarding.sh."""

from __future__ import annotations

from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "tools" / "run_fleet_host_onboarding.sh"


def test_host_onboarding_invokes_checklist():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "run_fleet_operator_checklist.sh" in text
    assert "install_agent_fleet_crontab.sh" in text
    assert "F-2026-005" in text
    assert "F-2026-006" in text
    assert "AGENT_FLEET_ARCHITECTURE_INDEX.md" in text
    assert "run_research_host_battery.sh" in text
    assert "phases 2–15" in text
