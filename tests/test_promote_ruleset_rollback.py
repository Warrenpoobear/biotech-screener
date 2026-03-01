"""Tests for first-class rollback in scripts/promote_ruleset.py."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.promote_ruleset import (
    _find_active,
    _find_last_known_good,
    main,
    write_receipt,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CANDIDATE_DATA = {"test_param": "value", "version": "candidate"}
_CANDIDATE_ID = hashlib.sha256(
    json.dumps(_CANDIDATE_DATA, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()[:8]

_ACTIVE_DATA = {"test_param": "old", "version": "active"}
_ACTIVE_ID = hashlib.sha256(
    json.dumps(_ACTIVE_DATA, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()[:8]

_RETIRED_DATA = {"test_param": "ancient", "version": "retired"}
_RETIRED_ID = hashlib.sha256(
    json.dumps(_RETIRED_DATA, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()[:8]


def _make_repo_with_retired(tmp_path: Path) -> dict:
    """Build a repo with active + retired rulesets for rollback testing."""
    rulesets_dir = tmp_path / "production_data" / "decision_rulesets"
    rulesets_dir.mkdir(parents=True)

    # Active ruleset file
    active_file = rulesets_dir / "v2_active.json"
    active_file.write_text(json.dumps(_ACTIVE_DATA), encoding="utf-8")

    # Retired ruleset file (was previously promoted)
    retired_file = rulesets_dir / "v1_retired.json"
    retired_file.write_text(json.dumps(_RETIRED_DATA), encoding="utf-8")

    # Candidate ruleset file (for forward promotion tests)
    candidate_file = rulesets_dir / "v3_candidate.json"
    candidate_file.write_text(json.dumps(_CANDIDATE_DATA), encoding="utf-8")

    manifest = {
        "rulesets": [
            {
                "id": _RETIRED_ID,
                "file": "v1_retired.json",
                "status": "retired",
                "engine_version": "v1.0.0",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_by": "promote_ruleset.py",
                "updated_at": "2026-01-15T00:00:00Z",
            },
            {
                "id": _ACTIVE_ID,
                "file": "v2_active.json",
                "status": "active",
                "engine_version": "v1.0.0",
                "created_at": "2026-01-15T00:00:00Z",
                "updated_by": "promote_ruleset.py",
                "updated_at": "2026-01-15T00:00:00Z",
            },
            {
                "id": _CANDIDATE_ID,
                "file": "v3_candidate.json",
                "status": "candidate",
                "engine_version": "v1.0.0",
                "created_at": "2026-02-01T00:00:00Z",
            },
        ]
    }
    manifest_path = rulesets_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Changelog (only has candidate entry, not retired — rollback skips changelog)
    changelog = tmp_path / "RULESET_CHANGELOG.md"
    changelog.write_text(
        f"# Changelog\n\n## [v3.0.0] {_CANDIDATE_ID} — 2026-02-27\n\nTest entry.\n",
        encoding="utf-8",
    )

    # Pin files
    run_screen = tmp_path / "run_screen.py"
    run_screen.write_text(
        f'PHASE2_PINNED_RULESET_ID = "{_ACTIVE_ID}"\n'
        f'PHASE2_DEFAULT_RULESET_PATH = (\n'
        f'    Path(__file__).resolve().parent\n'
        f'    / "production_data" / "decision_rulesets" / "v2_active.json"\n'
        f')\n',
        encoding="utf-8",
    )

    delta = tmp_path / "run_phase2_snapshot_delta.py"
    delta.write_text(
        f'PHASE2_PINNED_RULESET_ID = "{_ACTIVE_ID}"\n',
        encoding="utf-8",
    )

    return {
        "root": tmp_path,
        "manifest": manifest_path,
        "changelog": changelog,
        "run_screen": run_screen,
        "delta": delta,
        "rulesets_dir": rulesets_dir,
    }


def _patch_module(monkeypatch, repo):
    """Monkeypatch promote_ruleset module-level paths."""
    import scripts.promote_ruleset as mod
    monkeypatch.setattr(mod, "RULESETS_DIR", repo["rulesets_dir"])
    monkeypatch.setattr(mod, "MANIFEST_PATH", repo["manifest"])
    monkeypatch.setattr(mod, "CHANGELOG_PATH", repo["changelog"])
    monkeypatch.setattr(mod, "RECEIPTS_DIR", repo["root"] / "receipts")
    monkeypatch.setattr(mod, "PINNED_FILES", [repo["run_screen"], repo["delta"]])
    monkeypatch.setattr(mod, "PROJECT_ROOT", repo["root"])


def _make_gate_json(tmp_path: Path, candidate_id: str, verdict: str = "PASS") -> Path:
    gate_path = tmp_path / "ruleset_eval.json"
    data = {
        "schema": "ruleset_eval.v1",
        "candidate": {"id": candidate_id, "file": "test.json"},
        "baseline": {"id": _ACTIVE_ID, "file": "active.json"},
        "n_evaluated": 5,
        "n_skipped": 0,
        "gate": {
            "verdict": verdict,
            "fail_reasons": [],
            "warn_reasons": [],
        },
        "temporal_stability": {
            "mean_top60_overlap": 95.0,
            "max_rank_shift": {"ticker": "X", "shift": 5},
        },
        "performance": {
            "candidate": {"mean_turnover": 0.12},
        },
    }
    gate_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return gate_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRollbackWithoutForce:
    def test_rollback_with_reason_succeeds(self, tmp_path, monkeypatch):
        """--rollback --reason works without --force."""
        repo = _make_repo_with_retired(tmp_path)
        _patch_module(monkeypatch, repo)

        rc = main([
            _RETIRED_ID, "--rollback", "--reason", "drift spike detected",
            "--manifest", str(repo["manifest"]),
            "--changelog", str(repo["changelog"]),
        ])
        assert rc == 0

        m = json.loads(repo["manifest"].read_text())
        assert _find_active(m)["id"] == _RETIRED_ID

    def test_rollback_without_reason_or_force_fails(self, tmp_path, monkeypatch):
        """--rollback without --reason, --gate-summary, or --force → error."""
        repo = _make_repo_with_retired(tmp_path)
        _patch_module(monkeypatch, repo)

        rc = main([
            _RETIRED_ID, "--rollback",
            "--manifest", str(repo["manifest"]),
        ])
        assert rc == 1

        # Manifest unchanged
        m = json.loads(repo["manifest"].read_text())
        assert _find_active(m)["id"] == _ACTIVE_ID


class TestRollbackAutoDiscover:
    def test_auto_discover_last_known_good(self, tmp_path, monkeypatch):
        """--rollback without ruleset_id auto-discovers the LKG retired entry."""
        repo = _make_repo_with_retired(tmp_path)
        _patch_module(monkeypatch, repo)

        rc = main([
            "--rollback", "--reason", "auto rollback test",
            "--manifest", str(repo["manifest"]),
            "--changelog", str(repo["changelog"]),
        ])
        assert rc == 0

        m = json.loads(repo["manifest"].read_text())
        active = _find_active(m)
        assert active["id"] == _RETIRED_ID

    def test_find_last_known_good_helper(self):
        """_find_last_known_good walks manifest in reverse."""
        manifest = {
            "rulesets": [
                {"id": "aaa", "status": "retired", "updated_by": "manual"},
                {"id": "bbb", "status": "retired", "updated_by": "promote_ruleset.py"},
                {"id": "ccc", "status": "active", "updated_by": "promote_ruleset.py"},
            ]
        }
        lkg = _find_last_known_good(manifest)
        assert lkg["id"] == "bbb"

    def test_find_last_known_good_none(self):
        """No retired entry with promote_ruleset.py → None."""
        manifest = {
            "rulesets": [
                {"id": "aaa", "status": "retired", "updated_by": "manual"},
                {"id": "ccc", "status": "active"},
            ]
        }
        assert _find_last_known_good(manifest) is None


class TestRollbackReceipt:
    def test_rollback_receipt_has_action_field(self, tmp_path, monkeypatch):
        """Rollback receipt has action='rollback' and reason."""
        repo = _make_repo_with_retired(tmp_path)
        _patch_module(monkeypatch, repo)

        rc = main([
            _RETIRED_ID, "--rollback", "--reason", "testing receipt",
            "--manifest", str(repo["manifest"]),
            "--changelog", str(repo["changelog"]),
        ])
        assert rc == 0

        receipts = list((repo["root"] / "receipts").glob("rollback_*.json"))
        assert len(receipts) == 1
        data = json.loads(receipts[0].read_text())
        assert data["action"] == "rollback"
        assert data["reason"] == "testing receipt"
        assert data["new_active_id"] == _RETIRED_ID

    def test_rollback_updates_pins(self, tmp_path, monkeypatch):
        """Rollback updates PHASE2_PINNED_RULESET_ID in pin files."""
        repo = _make_repo_with_retired(tmp_path)
        _patch_module(monkeypatch, repo)

        rc = main([
            _RETIRED_ID, "--rollback", "--reason", "pin test",
            "--manifest", str(repo["manifest"]),
            "--changelog", str(repo["changelog"]),
        ])
        assert rc == 0

        assert _RETIRED_ID in repo["run_screen"].read_text()
        assert _RETIRED_ID in repo["delta"].read_text()
        assert _ACTIVE_ID not in repo["run_screen"].read_text()

    def test_rollback_to_specific_id(self, tmp_path, monkeypatch):
        """Explicit ruleset_id targets the correct retired entry."""
        repo = _make_repo_with_retired(tmp_path)
        _patch_module(monkeypatch, repo)

        rc = main([
            _RETIRED_ID, "--rollback", "--reason", "specific target",
            "--manifest", str(repo["manifest"]),
            "--changelog", str(repo["changelog"]),
        ])
        assert rc == 0

        m = json.loads(repo["manifest"].read_text())
        assert _find_active(m)["id"] == _RETIRED_ID
        # Old active is now retired
        for e in m["rulesets"]:
            if e["id"] == _ACTIVE_ID:
                assert e["status"] == "retired"


class TestForwardPromotionReceipt:
    def test_forward_promotion_receipt_has_action_field(self, tmp_path, monkeypatch):
        """Forward promotion writes action='promote'."""
        repo = _make_repo_with_retired(tmp_path)
        _patch_module(monkeypatch, repo)
        gate = _make_gate_json(tmp_path, _CANDIDATE_ID)

        rc = main([
            _CANDIDATE_ID, "--gate-summary", str(gate),
            "--manifest", str(repo["manifest"]),
            "--changelog", str(repo["changelog"]),
        ])
        assert rc == 0

        receipts = list((repo["root"] / "receipts").glob("promotion_*.json"))
        assert len(receipts) == 1
        data = json.loads(receipts[0].read_text())
        assert data["action"] == "promote"
        assert data.get("reason", "") == ""


class TestForceRollbackBackwardCompat:
    def test_force_rollback_still_works(self, tmp_path, monkeypatch):
        """--rollback --force still works (backward compat)."""
        repo = _make_repo_with_retired(tmp_path)
        _patch_module(monkeypatch, repo)

        rc = main([
            _RETIRED_ID, "--rollback", "--force",
            "--manifest", str(repo["manifest"]),
            "--changelog", str(repo["changelog"]),
        ])
        assert rc == 0

        m = json.loads(repo["manifest"].read_text())
        assert _find_active(m)["id"] == _RETIRED_ID
