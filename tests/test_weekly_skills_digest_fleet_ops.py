"""Fleet ops section in weekly skills digest."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


@pytest.fixture
def digest_mod():
    import tools.weekly_skills_digest as mod

    return mod


def test_fleet_ops_section_reads_artifact(digest_mod, tmp_path, monkeypatch):
    fleet_dir = tmp_path / "artifacts" / "fleet_ops"
    fleet_dir.mkdir(parents=True)
    ds = "2026-06-24"
    (fleet_dir / f"{ds}_status.json").write_text(
        json.dumps(
            {
                "overall": "WARN",
                "herald": {"verdict": "WARN", "herald_done": False},
                "heartbeat": {"receipt_exists": True, "verdict": "YELLOW", "escalation_mode": "artifact_only"},
                "selfimprove_gates": {"message": "blocked"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(digest_mod, "REPO", tmp_path)

    lines = digest_mod._fleet_ops_section(date.fromisoformat(ds))
    text = "\n".join(lines)

    assert "Overall: **WARN**" in text
    assert "Herald: WARN" in text
    assert "escalation=artifact_only" in text
    assert "blocked" in text


def test_fleet_ops_section_missing_artifact(digest_mod, tmp_path, monkeypatch):
    monkeypatch.setattr(digest_mod, "REPO", tmp_path)
    lines = digest_mod._fleet_ops_section(date.fromisoformat("2026-06-24"))
    assert "fleet_ops_status.py --write" in "\n".join(lines)
