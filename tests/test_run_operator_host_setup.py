"""Static checks for tools/run_operator_host_setup.sh."""

from __future__ import annotations

from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "tools" / "run_operator_host_setup.sh"


def test_operator_host_setup_chains_fleet_and_research():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "run_fleet_host_onboarding.sh" in text
    assert "run_research_host_battery.sh" in text
    assert "--skip-research" in text
    assert "--research-only" in text
    assert "snapshots_pit_v2" in text
    assert "CHECKLIST_V2_FINAL_SCORE_BLOCKER" in text
    assert "run_forward_evidence_package.sh" in text
