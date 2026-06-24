"""Tests for tools/pattern_to_skillpatch.py."""

from __future__ import annotations

import os
from pathlib import Path

from tools.pattern_to_skillpatch import infer_promotion_lane, parse_learnings

SAMPLE = """
## [LRN-20260601-001] ops_pattern

**Status**: pending
**Area**: hermes_ops

### Summary
Ops lesson.

### Metadata
- Pattern-Key: ops_pattern
- Recurrence-Count: 3
- Skill-Path: screener_ops

## [LRN-20260601-002] research_signal

**Status**: pending
**Area**: research

### Summary
Signal finding.

### Metadata
- Pattern-Key: research_signal
- Recurrence-Count: 4
- Promotion-lane: spec
"""


def test_parse_learnings_skill_path():
    entries = list(parse_learnings(SAMPLE))
    assert len(entries) == 2
    assert entries[0]["skill_path"] == "screener_ops"
    assert entries[1]["promotion_lane"] == "spec"


def test_infer_promotion_lane_defaults():
    assert infer_promotion_lane("research", None) == "spec"
    assert infer_promotion_lane("portfolio", None) == "spec"
    assert infer_promotion_lane("hermes_ops", None) == "skill"
    assert infer_promotion_lane("tooling", "none") == "none"


def test_pattern_to_skillpatch_runs_with_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("SELFIMPROVE_GATES_MET", "1")
    learnings = tmp_path / "LEARNINGS.md"
    learnings.write_text(SAMPLE, encoding="utf-8")
    out = tmp_path / "drafts"
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent.parent / "tools" / "pattern_to_skillpatch.py"),
            "--learnings",
            str(learnings),
            "--min-recurrence",
            "3",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "SELFIMPROVE_GATES_MET": "1"},
    )
    assert result.returncode == 0
    drafts = list(out.glob("skill_patch_drafts_*.md"))
    assert drafts
    body = drafts[0].read_text(encoding="utf-8")
    assert "ops_pattern" in body
    assert "BLOCKED" in body or "research_signal" in body
