"""Static checks for tools/cron_weekly_skills_review.sh."""

from __future__ import annotations

from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "tools" / "cron_weekly_skills_review.sh"


def test_weekly_skills_review_runs_fleet_ops_status():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "fleet_ops_status.py" in text
    assert "--write" in text
    assert "selfimprove_gates_status" in text
