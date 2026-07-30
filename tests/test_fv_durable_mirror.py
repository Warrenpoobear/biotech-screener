"""Tests for the out-of-tree durable mirror of the forward-validation ledger.

Classification: FORWARD_VALIDATION_EVIDENCE_DURABILITY / NO_MODEL_CHANGE

Spec 115 Phase 2a. On 2026-07-23 a mandate-eligible capture was written
successfully and then destroyed by a git working-tree revert of the tracked
ledger, unnoticed for five days. The mirror lives outside the git working tree,
so a revert cannot touch it, and divergence between the two is detectable and
repairable.

The tracked ledger remains the published audit record. The mirror exists only
so evidence loss is recoverable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from tools.fv_durable_mirror import dates_in, default_mirror_path, mirror_append, missing_from_tracked, restore_missing


def _rec(date: str, mode: str = "LIVE") -> dict:
    return {"date": date, "capture_mode": mode, "eligible_for_mandate": True}


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


class TestMirrorLocation:
    def test_default_mirror_is_outside_the_repo(self):
        """A path inside the working tree would still be reachable by git."""
        mirror = default_mirror_path()
        assert PROJECT_ROOT not in mirror.parents
        assert mirror.is_absolute()


class TestMirrorAppend:
    def test_append_creates_and_writes(self, tmp_path):
        m = tmp_path / "nested" / "captures.jsonl"
        mirror_append(_rec("2026-07-28"), mirror=m)
        assert dates_in(m) == ["2026-07-28"]

    def test_append_is_append_only(self, tmp_path):
        m = tmp_path / "captures.jsonl"
        mirror_append(_rec("2026-07-28"), mirror=m)
        mirror_append(_rec("2026-07-29"), mirror=m)
        assert dates_in(m) == ["2026-07-28", "2026-07-29"]

    def test_append_never_raises_on_unwritable_target(self, tmp_path):
        """Durability is best-effort — it must never fail a production capture."""
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        mirror_append(_rec("2026-07-28"), mirror=blocker / "captures.jsonl")

    def test_deterministic_serialization(self, tmp_path):
        m1, m2 = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
        mirror_append(_rec("2026-07-28"), mirror=m1)
        mirror_append(_rec("2026-07-28"), mirror=m2)
        assert m1.read_bytes() == m2.read_bytes()


class TestDivergenceDetection:
    def test_detects_the_2026_07_23_scenario(self, tmp_path):
        """Mirror holds 07-23; tracked ledger was reverted and lost it."""
        mirror = tmp_path / "mirror.jsonl"
        tracked = tmp_path / "tracked.jsonl"
        _write_jsonl(mirror, [_rec("2026-07-22"), _rec("2026-07-23"), _rec("2026-07-24")])
        _write_jsonl(tracked, [_rec("2026-07-22"), _rec("2026-07-24")])

        assert missing_from_tracked(tracked=tracked, mirror=mirror) == ["2026-07-23"]

    def test_no_divergence_when_in_sync(self, tmp_path):
        mirror = tmp_path / "mirror.jsonl"
        tracked = tmp_path / "tracked.jsonl"
        recs = [_rec("2026-07-28"), _rec("2026-07-29")]
        _write_jsonl(mirror, recs)
        _write_jsonl(tracked, recs)
        assert missing_from_tracked(tracked=tracked, mirror=mirror) == []

    def test_tracked_ahead_is_not_reported_as_loss(self, tmp_path):
        """The tracked ledger having extra rows is not evidence loss."""
        mirror = tmp_path / "mirror.jsonl"
        tracked = tmp_path / "tracked.jsonl"
        _write_jsonl(mirror, [_rec("2026-07-28")])
        _write_jsonl(tracked, [_rec("2026-07-28"), _rec("2026-07-29")])
        assert missing_from_tracked(tracked=tracked, mirror=mirror) == []

    def test_absent_mirror_reports_nothing(self, tmp_path):
        tracked = tmp_path / "tracked.jsonl"
        _write_jsonl(tracked, [_rec("2026-07-28")])
        assert missing_from_tracked(tracked=tracked, mirror=tmp_path / "nope.jsonl") == []


class TestRestore:
    def test_restore_reappends_lost_rows_in_date_order(self, tmp_path):
        mirror = tmp_path / "mirror.jsonl"
        tracked = tmp_path / "tracked.jsonl"
        _write_jsonl(mirror, [_rec("2026-07-22"), _rec("2026-07-23"), _rec("2026-07-24")])
        _write_jsonl(tracked, [_rec("2026-07-22"), _rec("2026-07-24")])

        restored = restore_missing(tracked=tracked, mirror=mirror)

        assert restored == ["2026-07-23"]
        assert sorted(dates_in(tracked)) == ["2026-07-22", "2026-07-23", "2026-07-24"]

    def test_restore_is_idempotent(self, tmp_path):
        mirror = tmp_path / "mirror.jsonl"
        tracked = tmp_path / "tracked.jsonl"
        _write_jsonl(mirror, [_rec("2026-07-23")])
        _write_jsonl(tracked, [])

        assert restore_missing(tracked=tracked, mirror=mirror) == ["2026-07-23"]
        assert restore_missing(tracked=tracked, mirror=mirror) == []

    def test_restore_does_not_fabricate_from_an_absent_mirror(self, tmp_path):
        tracked = tmp_path / "tracked.jsonl"
        _write_jsonl(tracked, [_rec("2026-07-28")])
        assert restore_missing(tracked=tracked, mirror=tmp_path / "nope.jsonl") == []
        assert dates_in(tracked) == ["2026-07-28"]
