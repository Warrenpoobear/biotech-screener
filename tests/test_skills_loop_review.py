"""Tests for tools/skills_loop_review.py and contradiction gate."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

from tools.pattern_to_skillpatch import draft_patch, parse_learnings
from tools.skills_loop_review import (
    check_skill_contradiction,
    efficacy_overdue,
    format_loop_review_sections,
    stalled_loop_entries,
    trim_candidates,
)

CONTRADICTION_SAMPLE = """
## [LRN-20260625-001] float_scores

**Status**: pending
**Area**: hermes_ops

### Summary
Use float for scoring speed.

### Metadata
- Pattern-Key: float_scores
- Recurrence-Count: 3
- Skill-Path: screener_ops
"""


def test_check_skill_contradiction_detects_prohibition(tmp_path):
    skill = "Never use float in scoring paths.\n"
    lesson = "Use float in scoring paths for speed."
    conflicts = check_skill_contradiction(skill, lesson)
    assert conflicts
    assert "float" in conflicts[0].lower()


def test_draft_patch_contradiction_review():
    entries = list(parse_learnings(CONTRADICTION_SAMPLE))
    assert len(entries) == 1
    conflicts = ["Lesson may contradict skill prohibition: never float"]
    body = draft_patch(entries[0], contradictions=conflicts)
    assert "CONTRADICTION_REVIEW" in body
    assert "never float" in body


def test_trim_candidates_zero_loads(tmp_path, monkeypatch):
    meta = tmp_path / "_meta.json"
    meta.write_text(
        json.dumps({"skills": {"alpha-skill": {}, "beta-skill": {}}}),
        encoding="utf-8",
    )
    logs = tmp_path / "logs"
    logs.mkdir()
    # Only alpha-skill has a recent execution
    log = logs / "execution_log_prod_2026-06.jsonl"
    ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    log.write_text(
        json.dumps({"skill_name": "alpha-skill", "timestamp": ts, "environment": "prod"}) + "\n",
        encoding="utf-8",
    )
    import tools.skills_loop_review as mod

    monkeypatch.setattr(mod, "META_JSON", meta)
    monkeypatch.setattr(mod, "LOGS_DIR", logs)
    trim = trim_candidates(as_of=date.today(), logs_dir=logs)
    skills = {t["skill"] for t in trim}
    assert "beta-skill" in skills
    assert "alpha-skill" not in skills


def test_efficacy_overdue_blocked_when_stalled_open():
    overdue = efficacy_overdue(stalled_open=True)
    assert len(overdue) == 1
    assert overdue[0]["reason"] == "stalled-loop"


def test_efficacy_overdue_finds_unverified_patch(tmp_path):
    harvest = tmp_path / "harvest_log.md"
    old = (date.today() - timedelta(days=20)).isoformat()
    harvest.write_text(
        f"""## {old} — test patch

### Skill patches
- **self-improving** (`skills/self-improving/SKILL.md`): added rule

---

## {date.today().isoformat()} — unrelated
""",
        encoding="utf-8",
    )
    overdue = efficacy_overdue(as_of=date.today(), harvest_path=harvest, stalled_open=False)
    assert any(o.get("skill") == "self-improving" for o in overdue)


def test_stalled_loop_entries_from_memory():
    entries = stalled_loop_entries(REPO / ".learnings" / "memory.md")
    ids = {e["id"] for e in entries}
    assert "F-2026-005" in ids
    assert "F-2026-006" in ids


def test_format_loop_review_sections_includes_trim(tmp_path, monkeypatch):
    import tools.skills_loop_review as mod

    monkeypatch.setattr(mod, "trim_candidates", lambda **kw: [{"skill": "unused-skill", "action": "review"}])
    lines = format_loop_review_sections(as_of=date.today())
    text = "\n".join(lines)
    assert "Trim candidates" in text
    assert "unused-skill" in text
