"""Tests for --rollback support in scripts/promote_ruleset.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMOTE_SCRIPT = PROJECT_ROOT / "scripts" / "promote_ruleset.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_manifest(tmp_path: Path, rulesets: list) -> Path:
    """Write a manifest.json and return its path."""
    mpath = tmp_path / "production_data" / "decision_rulesets" / "manifest.json"
    mpath.parent.mkdir(parents=True, exist_ok=True)
    with open(mpath, "w") as f:
        json.dump({"schema_version": 1, "rulesets": rulesets}, f)
    return mpath


def _write_changelog(tmp_path: Path, content: str = "") -> Path:
    cpath = tmp_path / "RULESET_CHANGELOG.md"
    cpath.write_text(content, encoding="utf-8")
    return cpath


def _run_promote(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    """Run promote_ruleset.py with the project root overridden to tmp_path."""
    # We patch PROJECT_ROOT by running via subprocess with modified script
    env_script = (
        f"import sys; sys.path.insert(0, '{PROJECT_ROOT}'); "
        f"sys.path.insert(0, '{PROJECT_ROOT / 'scripts'}'); "
        f"import promote_ruleset; "
        f"promote_ruleset.MANIFEST_PATH = __import__('pathlib').Path('{tmp_path / 'production_data' / 'decision_rulesets' / 'manifest.json'}'); "
        f"promote_ruleset.CHANGELOG_PATH = __import__('pathlib').Path('{tmp_path / 'RULESET_CHANGELOG.md'}'); "
        f"sys.argv = ['promote_ruleset.py'] + {list(args)}; "
        f"sys.exit(promote_ruleset.main())"
    )
    return subprocess.run(
        [sys.executable, "-c", env_script],
        capture_output=True, text=True,
    )


def _load_manifest(tmp_path: Path) -> dict:
    mpath = tmp_path / "production_data" / "decision_rulesets" / "manifest.json"
    with open(mpath) as f:
        return json.load(f)


# ============================================================================
# Tests
# ============================================================================
class TestPromoteRollback:
    def test_rollback_promotes_retired_to_active(self, tmp_path):
        """retired → active, old active → retired."""
        _write_manifest(tmp_path, [
            {"id": "active1", "file": "active.json", "status": "active"},
            {"id": "retired1", "file": "old.json", "status": "retired",
             "updated_at": "2026-01-01T00:00:00Z"},
        ])
        _write_changelog(tmp_path, "retired1 entry")

        result = _run_promote(tmp_path, "retired1", "--rollback", "--force")
        assert result.returncode == 0, result.stderr

        manifest = _load_manifest(tmp_path)
        statuses = {r["id"]: r["status"] for r in manifest["rulesets"]}
        assert statuses["retired1"] == "active"
        assert statuses["active1"] == "retired"

    def test_rollback_requires_force(self, tmp_path):
        """--rollback without --force → exit 1."""
        _write_manifest(tmp_path, [
            {"id": "active1", "file": "active.json", "status": "active"},
            {"id": "retired1", "file": "old.json", "status": "retired"},
        ])
        _write_changelog(tmp_path)

        result = _run_promote(tmp_path, "retired1", "--rollback")
        assert result.returncode == 1
        assert "--force" in result.stderr

    def test_rollback_rejects_non_retired(self, tmp_path):
        """candidate ID + --rollback → error about wrong status."""
        _write_manifest(tmp_path, [
            {"id": "active1", "file": "active.json", "status": "active"},
            {"id": "cand1", "file": "cand.json", "status": "candidate"},
        ])
        _write_changelog(tmp_path)

        result = _run_promote(tmp_path, "cand1", "--rollback", "--force")
        assert result.returncode == 1
        assert "candidate" in result.stderr

    def test_rollback_rejects_unknown_id(self, tmp_path):
        """Unknown ID → error."""
        _write_manifest(tmp_path, [
            {"id": "active1", "file": "active.json", "status": "active"},
        ])
        _write_changelog(tmp_path)

        result = _run_promote(tmp_path, "unknown99", "--rollback", "--force")
        assert result.returncode == 1
        assert "unknown99" in result.stderr

    def test_rollback_audit_trail(self, tmp_path):
        """updated_by contains '--rollback'."""
        _write_manifest(tmp_path, [
            {"id": "active1", "file": "active.json", "status": "active"},
            {"id": "retired1", "file": "old.json", "status": "retired"},
        ])
        _write_changelog(tmp_path)

        result = _run_promote(tmp_path, "retired1", "--rollback", "--force")
        assert result.returncode == 0, result.stderr

        manifest = _load_manifest(tmp_path)
        for entry in manifest["rulesets"]:
            if entry["id"] == "retired1":
                assert "--rollback" in entry["updated_by"]
                break

    def test_rollback_already_active_noop(self, tmp_path):
        """Target already active → exit 0 (noop)."""
        _write_manifest(tmp_path, [
            {"id": "active1", "file": "active.json", "status": "active"},
        ])
        _write_changelog(tmp_path)

        result = _run_promote(tmp_path, "active1", "--rollback", "--force")
        assert result.returncode == 0
        assert "already active" in result.stdout.lower()

    def test_rollback_dry_run(self, tmp_path):
        """--dry-run shows changes without writing."""
        _write_manifest(tmp_path, [
            {"id": "active1", "file": "active.json", "status": "active"},
            {"id": "retired1", "file": "old.json", "status": "retired"},
        ])
        _write_changelog(tmp_path)

        result = _run_promote(tmp_path, "retired1", "--rollback", "--force", "--dry-run")
        assert result.returncode == 0
        assert "dry-run" in result.stdout.lower()

        # Manifest should be unchanged
        manifest = _load_manifest(tmp_path)
        statuses = {r["id"]: r["status"] for r in manifest["rulesets"]}
        assert statuses["retired1"] == "retired"
        assert statuses["active1"] == "active"

    def test_normal_promote_still_works(self, tmp_path):
        """Existing candidate→active path unchanged."""
        _write_manifest(tmp_path, [
            {"id": "active1", "file": "active.json", "status": "active"},
            {"id": "cand1", "file": "cand.json", "status": "candidate"},
        ])
        _write_changelog(tmp_path, "cand1 entry here")

        result = _run_promote(tmp_path, "cand1", "--force")
        assert result.returncode == 0, result.stderr

        manifest = _load_manifest(tmp_path)
        statuses = {r["id"]: r["status"] for r in manifest["rulesets"]}
        assert statuses["cand1"] == "active"
        assert statuses["active1"] == "retired"
