"""The forward-validation capture pins the production order mode.

``model_hash`` does not cover the ``--ranker-mode`` dispatch in ``run_screen.py``, so the
switch to ``selector_only`` on 2026-11-13 left the hash unchanged. The capture therefore
compares the snapshot's recorded mode with ``CANDIDATE.json.ranker_mode``.
"""

from __future__ import annotations

import json

from tools.run_forward_validation import check_ranker_mode, run_dq_checks, snapshot_ranker_mode


def _snapshot(tmp_path, mode=None, raw=None):
    snap = tmp_path / "2026-11-16"
    snap.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        (snap / "metadata.json").write_text(raw, encoding="utf-8")
    elif mode is not None:
        (snap / "metadata.json").write_text(json.dumps({"ranker_mode": mode}), encoding="utf-8")
    return snap


def test_matching_mode_passes(tmp_path):
    ok, detail = check_ranker_mode({"ranker_mode": "selector_only"}, _snapshot(tmp_path, "selector_only"))
    assert ok is True
    assert "selector_only" in detail


def test_different_mode_fails(tmp_path):
    ok, detail = check_ranker_mode({"ranker_mode": "selector_only"}, _snapshot(tmp_path, "pairwise_minimal"))
    assert ok is False
    assert "pairwise_minimal" in detail


def test_pinned_mode_with_unrecorded_snapshot_fails(tmp_path):
    assert check_ranker_mode({"ranker_mode": "selector_only"}, _snapshot(tmp_path))[0] is False
    assert check_ranker_mode({"ranker_mode": "selector_only"}, _snapshot(tmp_path, raw="{not json"))[0] is False


def test_candidate_without_pin_passes(tmp_path):
    assert check_ranker_mode({"model_hash": "x"}, _snapshot(tmp_path, "pairwise_minimal"))[0] is True
    assert check_ranker_mode(None, _snapshot(tmp_path))[0] is True


def test_snapshot_ranker_mode_reads_metadata(tmp_path):
    assert snapshot_ranker_mode(_snapshot(tmp_path, "selector_only")) == "selector_only"
    assert snapshot_ranker_mode(tmp_path / "absent") is None


def test_mismatch_is_a_hard_dq_failure(tmp_path):
    snap = _snapshot(tmp_path, "pairwise_minimal")
    candidate = {"model_hash": "abc", "ranker_mode": "selector_only"}
    quality, checks = run_dq_checks("2026-11-16", [], snap, "abc", candidate)
    by_name = {c["check"]: c for c in checks}
    assert by_name["ranker_mode_match"]["pass"] is False
    assert by_name["ranker_mode_match"]["hard"] is True
    assert quality == "FAIL"
