"""Tests for Rule 12 self-improve gate helpers."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.skills_loop_review import selfimprove_gates_status, stalled_loops_open


def test_selfimprove_gates_blocked_when_memory_has_open_loops(tmp_path, monkeypatch):
    memory = tmp_path / "memory.md"
    memory.write_text(
        """
## Stalled-loop verdicts

| ID | System | Status | Evidence | Close when | Target |
| F-2026-005 | Herald Digest | **OPEN** | dark | host fix | 2026-07-01 |
| F-2026-006 | GitHub CI | **OPEN** | budget | green CI | 2026-07-01 |
""",
        encoding="utf-8",
    )
    monkeypatch.setattr("tools.skills_loop_review.MEMORY_MD", memory)

    gates = selfimprove_gates_status(memory)
    assert gates["stalled_loops_open"] is True
    assert gates["selfimprove_gates_met_allowed"] is False
    assert "F-2026-005" in gates["message"]
    assert stalled_loops_open(memory) is True


def test_selfimprove_gates_allowed_when_no_open_loops(tmp_path, monkeypatch):
    memory = tmp_path / "memory.md"
    memory.write_text(
        """
## Stalled-loop verdicts

| ID | System | Status | Evidence | Close when | Target |
| F-2026-005 | Herald Digest | RESOLVED | fixed | done | 2026-07-01 |
""",
        encoding="utf-8",
    )
    monkeypatch.setattr("tools.skills_loop_review.MEMORY_MD", memory)

    gates = selfimprove_gates_status(memory)
    assert gates["selfimprove_gates_met_allowed"] is True
    assert "may set SELFIMPROVE_GATES_MET=1" in gates["message"]
