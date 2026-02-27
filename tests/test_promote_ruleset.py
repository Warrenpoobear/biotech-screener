"""Tests for scripts/promote_ruleset.py — gate requirement + receipt + pin updates."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts.promote_ruleset import (
    _extract_from_artifact,
    _find_active,
    _validate_changelog,
    update_default_path,
    update_pinned_ids,
    validate_gate,
    write_receipt,
    main,
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


def _make_repo(tmp_path: Path) -> dict:
    """Build a minimal fake repo layout. Returns paths dict."""
    rulesets_dir = tmp_path / "production_data" / "decision_rulesets"
    rulesets_dir.mkdir(parents=True)

    # Active ruleset file
    active_file = rulesets_dir / "v1_active.json"
    active_file.write_text(json.dumps(_ACTIVE_DATA), encoding="utf-8")

    # Candidate ruleset file
    candidate_file = rulesets_dir / "v2_candidate.json"
    candidate_file.write_text(json.dumps(_CANDIDATE_DATA), encoding="utf-8")

    # Manifest
    manifest = {
        "rulesets": [
            {
                "id": _ACTIVE_ID,
                "file": "v1_active.json",
                "status": "active",
                "engine_version": "v1.0.0",
                "created_at": "2026-01-01T00:00:00Z",
            },
            {
                "id": _CANDIDATE_ID,
                "file": "v2_candidate.json",
                "status": "candidate",
                "engine_version": "v1.0.0",
                "created_at": "2026-02-01T00:00:00Z",
            },
        ]
    }
    manifest_path = rulesets_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Changelog
    changelog = tmp_path / "RULESET_CHANGELOG.md"
    changelog.write_text(
        f"# Changelog\n\n## [v1.0.0] {_CANDIDATE_ID} — 2026-02-27 — Test\n\nTest entry.\n",
        encoding="utf-8",
    )

    # Minimal run_screen.py stub
    run_screen = tmp_path / "run_screen.py"
    run_screen.write_text(
        f'PHASE2_PINNED_RULESET_ID = "{_ACTIVE_ID}"\n'
        f'PHASE2_DEFAULT_RULESET_PATH = (\n'
        f'    Path(__file__).resolve().parent\n'
        f'    / "production_data" / "decision_rulesets" / "v1_active.json"\n'
        f')\n',
        encoding="utf-8",
    )

    # Minimal delta stub
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
        "candidate_file": candidate_file,
    }


def _make_gate_json(
    tmp_path: Path,
    verdict: str = "PASS",
    candidate_id: str = _CANDIDATE_ID,
    n_evaluated: int = 5,
) -> Path:
    gate_path = tmp_path / "ruleset_eval.json"
    data = {
        "schema": "ruleset_eval.v1",
        "candidate": {"id": candidate_id, "file": "v2_candidate.json"},
        "baseline": {"id": _ACTIVE_ID, "file": "v1_active.json"},
        "n_evaluated": n_evaluated,
        "n_skipped": 0,
        "gate": {
            "verdict": verdict,
            "fail_reasons": ["test_fail"] if verdict == "FAIL" else [],
            "warn_reasons": ["test_warn"] if verdict == "WARN" else [],
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
# Tests: validate_gate
# ---------------------------------------------------------------------------


class TestValidateGate:
    def test_pass_accepted(self, tmp_path):
        gate = _make_gate_json(tmp_path)
        ok, msg, data = validate_gate(gate, _CANDIDATE_ID)
        assert ok
        assert "PASS" in msg

    def test_fail_rejected(self, tmp_path):
        gate = _make_gate_json(tmp_path, verdict="FAIL")
        ok, msg, _ = validate_gate(gate, _CANDIDATE_ID)
        assert not ok
        assert "FAIL" in msg

    def test_warn_rejected(self, tmp_path):
        gate = _make_gate_json(tmp_path, verdict="WARN")
        ok, msg, _ = validate_gate(gate, _CANDIDATE_ID)
        assert not ok
        assert "WARN" in msg

    def test_id_mismatch_rejected(self, tmp_path):
        gate = _make_gate_json(tmp_path, candidate_id="wrong123")
        ok, msg, _ = validate_gate(gate, _CANDIDATE_ID)
        assert not ok
        assert "does not match" in msg

    def test_zero_evaluated_rejected(self, tmp_path):
        gate = _make_gate_json(tmp_path, n_evaluated=0)
        ok, msg, _ = validate_gate(gate, _CANDIDATE_ID)
        assert not ok
        assert "0 dates" in msg

    def test_missing_file_rejected(self, tmp_path):
        ok, msg, _ = validate_gate(tmp_path / "nonexistent.json", _CANDIDATE_ID)
        assert not ok


# ---------------------------------------------------------------------------
# Tests: pin updates
# ---------------------------------------------------------------------------


class TestPinUpdates:
    def test_update_pinned_ids(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text(f'PHASE2_PINNED_RULESET_ID = "{_ACTIVE_ID}"\n')
        updated = update_pinned_ids(_ACTIVE_ID, _CANDIDATE_ID, [f])
        assert len(updated) == 1
        assert _CANDIDATE_ID in f.read_text()

    def test_dry_run_no_write(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text(f'PHASE2_PINNED_RULESET_ID = "{_ACTIVE_ID}"\n')
        updated = update_pinned_ids(_ACTIVE_ID, _CANDIDATE_ID, [f], dry_run=True)
        assert len(updated) == 1
        assert _ACTIVE_ID in f.read_text()  # unchanged

    def test_update_default_path(self, tmp_path):
        f = tmp_path / "run_screen.py"
        f.write_text(
            'x = (\n    Path()\n'
            '    / "production_data" / "decision_rulesets" / "old_file.json"\n)\n'
        )
        ok = update_default_path("new_file.json", f)
        assert ok
        assert "new_file.json" in f.read_text()
        assert "old_file.json" not in f.read_text()


# ---------------------------------------------------------------------------
# Tests: receipt
# ---------------------------------------------------------------------------


class TestReceipt:
    def test_receipt_written(self, tmp_path):
        path = write_receipt(
            old_active_id=_ACTIVE_ID,
            new_active_id=_CANDIDATE_ID,
            candidate_path="v2_candidate.json",
            promoted_file="v2_candidate.json",
            gate_data={"gate": {"verdict": "PASS"}, "n_evaluated": 5},
            gate_summary_path="/tmp/eval.json",
            gate_run_url="https://example.com/run/123",
            forced=False,
            receipts_dir=tmp_path / "receipts",
        )
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["schema"] == "promote_receipt.v1"
        assert data["old_active_id"] == _ACTIVE_ID
        assert data["new_active_id"] == _CANDIDATE_ID
        assert data["forced"] is False
        assert data["gate_run_url"] == "https://example.com/run/123"


# ---------------------------------------------------------------------------
# Tests: main (integration)
# ---------------------------------------------------------------------------


class TestMain:
    def test_pass_gate_promotes(self, tmp_path, monkeypatch):
        repo = _make_repo(tmp_path)
        gate = _make_gate_json(tmp_path)

        # Patch module-level paths
        import scripts.promote_ruleset as mod
        monkeypatch.setattr(mod, "RULESETS_DIR", repo["rulesets_dir"])
        monkeypatch.setattr(mod, "MANIFEST_PATH", repo["manifest"])
        monkeypatch.setattr(mod, "CHANGELOG_PATH", repo["changelog"])
        monkeypatch.setattr(mod, "RECEIPTS_DIR", tmp_path / "receipts")
        monkeypatch.setattr(mod, "PINNED_FILES", [repo["run_screen"], repo["delta"]])
        monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)

        rc = main([_CANDIDATE_ID, "--gate-summary", str(gate), "--manifest", str(repo["manifest"]), "--changelog", str(repo["changelog"])])
        assert rc == 0

        # Check manifest
        m = json.loads(repo["manifest"].read_text())
        new_active = _find_active(m)
        assert new_active["id"] == _CANDIDATE_ID
        # Old active is retired
        for e in m["rulesets"]:
            if e["id"] == _ACTIVE_ID:
                assert e["status"] == "retired"

        # Check pins updated
        assert _CANDIDATE_ID in repo["run_screen"].read_text()
        assert _CANDIDATE_ID in repo["delta"].read_text()

        # Check receipt
        receipts = list((tmp_path / "receipts").glob("*.json"))
        assert len(receipts) == 1

    def test_fail_gate_blocks(self, tmp_path, monkeypatch):
        repo = _make_repo(tmp_path)
        gate = _make_gate_json(tmp_path, verdict="FAIL")

        import scripts.promote_ruleset as mod
        monkeypatch.setattr(mod, "RULESETS_DIR", repo["rulesets_dir"])
        monkeypatch.setattr(mod, "MANIFEST_PATH", repo["manifest"])

        rc = main([_CANDIDATE_ID, "--gate-summary", str(gate), "--manifest", str(repo["manifest"])])
        assert rc == 1

        # Manifest unchanged
        m = json.loads(repo["manifest"].read_text())
        assert _find_active(m)["id"] == _ACTIVE_ID

    def test_zero_evaluated_blocks(self, tmp_path, monkeypatch):
        repo = _make_repo(tmp_path)
        gate = _make_gate_json(tmp_path, n_evaluated=0)

        import scripts.promote_ruleset as mod
        monkeypatch.setattr(mod, "RULESETS_DIR", repo["rulesets_dir"])
        monkeypatch.setattr(mod, "MANIFEST_PATH", repo["manifest"])

        rc = main([_CANDIDATE_ID, "--gate-summary", str(gate), "--manifest", str(repo["manifest"])])
        assert rc == 1

    def test_force_bypasses_gate(self, tmp_path, monkeypatch):
        repo = _make_repo(tmp_path)

        import scripts.promote_ruleset as mod
        monkeypatch.setattr(mod, "RULESETS_DIR", repo["rulesets_dir"])
        monkeypatch.setattr(mod, "MANIFEST_PATH", repo["manifest"])
        monkeypatch.setattr(mod, "CHANGELOG_PATH", repo["changelog"])
        monkeypatch.setattr(mod, "RECEIPTS_DIR", tmp_path / "receipts")
        monkeypatch.setattr(mod, "PINNED_FILES", [repo["run_screen"], repo["delta"]])
        monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)

        rc = main([_CANDIDATE_ID, "--force", "--manifest", str(repo["manifest"]), "--changelog", str(repo["changelog"])])
        assert rc == 0

        m = json.loads(repo["manifest"].read_text())
        assert _find_active(m)["id"] == _CANDIDATE_ID

    def test_no_gate_summary_without_force_fails(self, tmp_path, monkeypatch):
        repo = _make_repo(tmp_path)

        import scripts.promote_ruleset as mod
        monkeypatch.setattr(mod, "MANIFEST_PATH", repo["manifest"])

        rc = main([_CANDIDATE_ID, "--manifest", str(repo["manifest"])])
        assert rc == 1

    def test_id_mismatch_blocks(self, tmp_path, monkeypatch):
        repo = _make_repo(tmp_path)
        gate = _make_gate_json(tmp_path, candidate_id="wrong123")

        import scripts.promote_ruleset as mod
        monkeypatch.setattr(mod, "RULESETS_DIR", repo["rulesets_dir"])
        monkeypatch.setattr(mod, "MANIFEST_PATH", repo["manifest"])

        rc = main([_CANDIDATE_ID, "--gate-summary", str(gate), "--manifest", str(repo["manifest"])])
        assert rc == 1

    def test_dry_run_no_writes(self, tmp_path, monkeypatch):
        repo = _make_repo(tmp_path)
        gate = _make_gate_json(tmp_path)

        import scripts.promote_ruleset as mod
        monkeypatch.setattr(mod, "RULESETS_DIR", repo["rulesets_dir"])
        monkeypatch.setattr(mod, "MANIFEST_PATH", repo["manifest"])
        monkeypatch.setattr(mod, "CHANGELOG_PATH", repo["changelog"])
        monkeypatch.setattr(mod, "PINNED_FILES", [repo["run_screen"], repo["delta"]])
        monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)

        rc = main([_CANDIDATE_ID, "--gate-summary", str(gate), "--dry-run", "--manifest", str(repo["manifest"]), "--changelog", str(repo["changelog"])])
        assert rc == 0

        # Manifest unchanged
        m = json.loads(repo["manifest"].read_text())
        assert _find_active(m)["id"] == _ACTIVE_ID
        # Pins unchanged
        assert _ACTIVE_ID in repo["run_screen"].read_text()


# ---------------------------------------------------------------------------
# Tests: gate artifact auto-fetch
# ---------------------------------------------------------------------------


class TestGateAutoFetch:
    def test_artifact_zip_extraction(self, tmp_path):
        """Extracts ruleset_eval.json from a zip artifact."""
        eval_data = {"gate": {"verdict": "PASS"}, "n_evaluated": 5}
        zip_path = tmp_path / "artifact.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr(
                "phase2-snapshot-2026-02-27/ruleset_eval.json",
                json.dumps(eval_data),
            )
        work_dir = tmp_path / "extract"
        work_dir.mkdir()
        result = _extract_from_artifact(zip_path, work_dir)
        assert result.exists()
        loaded = json.loads(result.read_text())
        assert loaded["gate"]["verdict"] == "PASS"

    def test_multiple_sources_error(self, tmp_path, monkeypatch):
        """Specifying >1 gate source produces an error."""
        repo = _make_repo(tmp_path)
        gate = _make_gate_json(tmp_path)

        import scripts.promote_ruleset as mod
        monkeypatch.setattr(mod, "RULESETS_DIR", repo["rulesets_dir"])
        monkeypatch.setattr(mod, "MANIFEST_PATH", repo["manifest"])

        rc = main([
            _CANDIDATE_ID,
            "--gate-summary", str(gate),
            "--gate-run-id", "12345",
            "--manifest", str(repo["manifest"]),
        ])
        assert rc == 1

    def test_missing_artifact_file_error(self, tmp_path, monkeypatch):
        """Non-existent zip path errors cleanly."""
        repo = _make_repo(tmp_path)

        import scripts.promote_ruleset as mod
        monkeypatch.setattr(mod, "RULESETS_DIR", repo["rulesets_dir"])
        monkeypatch.setattr(mod, "MANIFEST_PATH", repo["manifest"])

        rc = main([
            _CANDIDATE_ID,
            "--gate-artifact", str(tmp_path / "nonexistent.zip"),
            "--manifest", str(repo["manifest"]),
        ])
        assert rc == 1
