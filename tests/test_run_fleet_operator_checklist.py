"""Static checks for tools/run_fleet_operator_checklist.sh."""

from __future__ import annotations

from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "tools" / "run_fleet_operator_checklist.sh"


def test_operator_checklist_runs_core_fleet_tools():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "herald_health_check.py" in text
    assert "fleet_ops_status.py" in text
    assert "fleet_completion_audit.py" in text
    assert "fleet_crontab_verify.py" in text
    assert "selfimprove_gates_status" in text
    assert "install_agent_fleet_crontab.sh" in text


def test_operator_checklist_runs_audit_before_fleet_ops():
    text = SCRIPT.read_text(encoding="utf-8")
    audit_pos = text.find("fleet_completion_audit.py")
    fleet_pos = text.find("fleet_ops_status.py")
    assert audit_pos != -1 and fleet_pos != -1
    assert audit_pos < fleet_pos
