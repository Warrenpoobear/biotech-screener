"""Lane-refusal tests for tools/pattern_to_skillpatch.py (Rule 12)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tools.pattern_to_skillpatch import (
    draft_patch,
    infer_promotion_lane,
    parse_learnings,
    refuse_spec_lane_entries,
)

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

## [LRN-20260601-003] log_only

**Status**: pending
**Area**: hermes_ops

### Summary
History only.

### Metadata
- Pattern-Key: log_only
- Recurrence-Count: 5
- Promotion-lane: none
"""


def test_infer_promotion_lane_defaults():
    assert infer_promotion_lane("research", None) == "spec"
    assert infer_promotion_lane("portfolio", None) == "spec"
    assert infer_promotion_lane("hermes_ops", None) == "skill"
    assert infer_promotion_lane("tooling", "none") == "none"


def test_parse_learnings_skill_path_and_lanes():
    entries = list(parse_learnings(SAMPLE))
    assert len(entries) == 3
    assert entries[0]["skill_path"] == "screener_ops"
    assert entries[0]["promotion_lane"] == "skill"
    assert entries[1]["promotion_lane"] == "spec"
    assert entries[2]["promotion_lane"] == "none"


def test_refuse_spec_lane_entries():
    entries = list(parse_learnings(SAMPLE))
    eligible = [e for e in entries if e["recurrence"] >= 3 and e["pattern_key"]]
    refused = refuse_spec_lane_entries(eligible)
    assert len(refused) == 1
    assert refused[0]["id"] == "LRN-20260601-002"


def test_draft_patch_blocks_spec_lane():
    entries = list(parse_learnings(SAMPLE))
    spec_entry = next(e for e in entries if e["promotion_lane"] == "spec")
    body = draft_patch(spec_entry)
    assert "BLOCKED (spec lane)" in body
    assert "governance Spec" in body


def test_pattern_to_skillpatch_cli_with_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("SELFIMPROVE_GATES_MET", "1")
    learnings = tmp_path / "LEARNINGS.md"
    learnings.write_text(SAMPLE, encoding="utf-8")
    out = tmp_path / "drafts"
    tool = Path(__file__).resolve().parent.parent / "tools" / "pattern_to_skillpatch.py"
    result = subprocess.run(
        [
            sys.executable,
            str(tool),
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
    assert "BLOCKED (spec lane)" in body
    assert "1 spec-lane entry refused" in result.stdout
