"""Tests for replay bundle packaging and verification."""
from __future__ import annotations

import csv
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

sys_path_added = False


def _ensure_path():
    global sys_path_added
    if not sys_path_added:
        import sys

        root = str(Path(__file__).resolve().parent.parent)
        if root not in sys.path:
            sys.path.insert(0, root)
        sys_path_added = True


_ensure_path()

from tools.make_replay_bundle import (
    FINGERPRINT_KEYS,
    SCHEMA_VERSION,
    _extract_topk_from_rankings,
    _sha256,
    _sha256_bytes,
    _stable_json,
    make_replay_bundle,
)
from tools.replay_bundle import (
    extract_bundle,
    verify_bundle_integrity,
    verify_invariants,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_synthetic_snapshot(
    snap_dir: Path,
    as_of_date: str = "2026-01-15",
    n_tickers: int = 10,
) -> Path:
    """Create a minimal synthetic snapshot for testing."""
    date_dir = snap_dir / as_of_date
    date_dir.mkdir(parents=True, exist_ok=True)

    # Metadata
    metadata = {
        "as_of_date": as_of_date,
        "saved_at": f"{as_of_date}T12:00:00Z",
        "decision_mode": "phase2",
        "ranking_mode": "decision",
        "version": "1.6.0",
        "ticker_count": n_tickers,
        "source_type": "live_pipeline_v3",
        "scoring_model": "module_5_v3_ic_enhanced",
        "total_evaluated": n_tickers + 5,
        "active_universe": n_tickers + 2,
        "clinical_excluded": 3,
        "clinical_exempted": 2,
        "archetype_distribution": {"drug_developer": n_tickers},
        "alpha_modifier_telemetry": {
            "mode": "within_tier",
            "weight": 0.05,
            "table_policy": "if_missing",
            "table_path": "",
            "alpha_table_source": "static",
        },
        "clinical_sort_telemetry": {
            "ruleset_id": "test1234",
        },
        "cache_health_overall_status": "ok",
        "cache_degraded_run": False,
        "cache_refresh_had_rejections": False,
        "cache_refresh_rejected_sources": [],
    }
    _write_json(date_dir / "metadata.json", metadata)

    # Ruleset
    ruleset = {
        "id": "test1234",
        "version": "v1.0.0_test",
        "alpha_cohort_table_path": "production_data/alpha_cohort_tables/v1.json",
    }
    _write_json(date_dir / "decision_ruleset.json", ruleset)

    # Rankings
    tickers = [f"TK{i:02d}" for i in range(1, n_tickers + 1)]
    rows = []
    for i, ticker in enumerate(tickers):
        rows.append({
            "ticker": ticker,
            "company_name": f"Company {ticker}",
            "actionable_rank": str(i + 1),
            "target_weight_pct": str(round(100 / n_tickers, 2)),
            "tier_dev": "A" if i < 5 else "B",
        })
    _write_csv(date_dir / "rankings.csv", rows)

    # Cache health
    _write_json(date_dir / "cache_health.json", {
        "schema": "cache_health.v1",
        "overall_status": "ok",
    })

    return date_dir


def _make_project_structure(root: Path, as_of_date: str = "2026-01-15") -> None:
    """Create minimal project files outside the snapshot."""
    # Universe
    universe = [{"ticker": f"TK{i:02d}"} for i in range(1, 11)]
    _write_json(root / "production_data" / "universe.json", universe)

    # Price history (tiny)
    price_path = root / "production_data" / "price_history.csv"
    _write_csv(price_path, [
        {"date": as_of_date, "ticker": "TK01", "close": "10.0"},
        {"date": as_of_date, "ticker": "TK02", "close": "20.0"},
    ])


# ---------------------------------------------------------------------------
# Tests: _stable_json determinism
# ---------------------------------------------------------------------------


class TestStableJson:
    def test_sorted_keys(self):
        obj = {"z": 1, "a": 2, "m": 3}
        result = json.loads(_stable_json(obj))
        keys = list(result.keys())
        assert keys == sorted(keys)

    def test_trailing_newline(self):
        result = _stable_json({"key": "value"})
        assert result.endswith(b"\n")

    def test_deterministic(self):
        obj = {"b": [3, 2, 1], "a": {"y": 1, "x": 2}}
        assert _stable_json(obj) == _stable_json(obj)


# ---------------------------------------------------------------------------
# Tests: _extract_topk_from_rankings
# ---------------------------------------------------------------------------


class TestExtractTopK:
    def test_extracts_ranked_tickers(self, tmp_path):
        rows = [
            {"ticker": "AAAA", "actionable_rank": "3", "other": "x"},
            {"ticker": "BBBB", "actionable_rank": "1", "other": "y"},
            {"ticker": "CCCC", "actionable_rank": "2", "other": "z"},
        ]
        csv_path = tmp_path / "rankings.csv"
        _write_csv(csv_path, rows)
        result = _extract_topk_from_rankings(csv_path, top_k=2)
        assert result == ["BBBB", "CCCC"]

    def test_respects_top_k_limit(self, tmp_path):
        rows = [{"ticker": f"T{i}", "actionable_rank": str(i)} for i in range(1, 20)]
        csv_path = tmp_path / "rankings.csv"
        _write_csv(csv_path, rows)
        result = _extract_topk_from_rankings(csv_path, top_k=5)
        assert len(result) == 5
        assert result[0] == "T1"

    def test_skips_empty_rank(self, tmp_path):
        rows = [
            {"ticker": "A", "actionable_rank": "1"},
            {"ticker": "B", "actionable_rank": ""},
            {"ticker": "C", "actionable_rank": "2"},
        ]
        csv_path = tmp_path / "rankings.csv"
        _write_csv(csv_path, rows)
        result = _extract_topk_from_rankings(csv_path, top_k=10)
        assert result == ["A", "C"]


# ---------------------------------------------------------------------------
# Tests: make_replay_bundle
# ---------------------------------------------------------------------------


class TestMakeReplayBundle:
    def test_creates_tgz(self, tmp_path):
        root = tmp_path / "project"
        snap_dir = root / "data" / "snapshots"
        _make_synthetic_snapshot(snap_dir, "2026-01-15")
        _make_project_structure(root, "2026-01-15")

        out = tmp_path / "out" / "bundle.tgz"
        result = make_replay_bundle(
            as_of_date="2026-01-15",
            snapshot_dir=snap_dir,
            out_path=out,
            top_k=5,
            project_root=root,
        )
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_manifest_schema(self, tmp_path):
        root = tmp_path / "project"
        snap_dir = root / "data" / "snapshots"
        _make_synthetic_snapshot(snap_dir, "2026-01-15")
        _make_project_structure(root, "2026-01-15")

        out = tmp_path / "bundle.tgz"
        make_replay_bundle(
            as_of_date="2026-01-15",
            snapshot_dir=snap_dir,
            out_path=out,
            top_k=5,
            project_root=root,
        )

        # Extract and check manifest
        extract_dir = tmp_path / "extracted"
        bundle_dir = extract_bundle(out, extract_dir)
        manifest = json.loads((bundle_dir / "manifest.json").read_text())

        assert manifest["schema"] == SCHEMA_VERSION
        assert manifest["as_of_date"] == "2026-01-15"
        assert manifest["decision_mode"] == "phase2"
        assert manifest["ruleset_id"] == "test1234"
        assert isinstance(manifest["included_files"], list)
        assert len(manifest["included_files"]) > 0

    def test_included_files_have_sha256(self, tmp_path):
        root = tmp_path / "project"
        snap_dir = root / "data" / "snapshots"
        _make_synthetic_snapshot(snap_dir, "2026-01-15")
        _make_project_structure(root, "2026-01-15")

        out = tmp_path / "bundle.tgz"
        make_replay_bundle(
            as_of_date="2026-01-15",
            snapshot_dir=snap_dir,
            out_path=out,
            top_k=5,
            project_root=root,
        )

        extract_dir = tmp_path / "extracted"
        bundle_dir = extract_bundle(out, extract_dir)
        manifest = json.loads((bundle_dir / "manifest.json").read_text())

        for entry in manifest["included_files"]:
            assert "relpath" in entry
            assert "sha256" in entry
            assert len(entry["sha256"]) == 64  # hex-encoded SHA-256
            assert "bytes" in entry

    def test_topk_tickers_in_bundle(self, tmp_path):
        root = tmp_path / "project"
        snap_dir = root / "data" / "snapshots"
        _make_synthetic_snapshot(snap_dir, "2026-01-15", n_tickers=10)
        _make_project_structure(root, "2026-01-15")

        out = tmp_path / "bundle.tgz"
        make_replay_bundle(
            as_of_date="2026-01-15",
            snapshot_dir=snap_dir,
            out_path=out,
            top_k=5,
            project_root=root,
        )

        extract_dir = tmp_path / "extracted"
        bundle_dir = extract_bundle(out, extract_dir)
        topk = json.loads(
            (bundle_dir / "replay" / "expected" / "topk_tickers.json").read_text()
        )
        assert topk["top_k"] == 5
        assert len(topk["tickers"]) == 5
        assert topk["tickers"][0] == "TK01"

    def test_metadata_fingerprint_in_bundle(self, tmp_path):
        root = tmp_path / "project"
        snap_dir = root / "data" / "snapshots"
        _make_synthetic_snapshot(snap_dir, "2026-01-15")
        _make_project_structure(root, "2026-01-15")

        out = tmp_path / "bundle.tgz"
        make_replay_bundle(
            as_of_date="2026-01-15",
            snapshot_dir=snap_dir,
            out_path=out,
            top_k=5,
            project_root=root,
        )

        extract_dir = tmp_path / "extracted"
        bundle_dir = extract_bundle(out, extract_dir)
        fp = json.loads(
            (bundle_dir / "replay" / "expected" / "metadata_fingerprint.json").read_text()
        )
        assert "fingerprint_sha256" in fp
        assert len(fp["fingerprint_sha256"]) == 64
        assert fp["keys"] == FINGERPRINT_KEYS
        assert "values" in fp

    def test_run_args_in_bundle(self, tmp_path):
        root = tmp_path / "project"
        snap_dir = root / "data" / "snapshots"
        _make_synthetic_snapshot(snap_dir, "2026-01-15")
        _make_project_structure(root, "2026-01-15")

        out = tmp_path / "bundle.tgz"
        make_replay_bundle(
            as_of_date="2026-01-15",
            snapshot_dir=snap_dir,
            out_path=out,
            top_k=5,
            project_root=root,
        )

        extract_dir = tmp_path / "extracted"
        bundle_dir = extract_bundle(out, extract_dir)
        args = json.loads((bundle_dir / "replay" / "run_args.json").read_text())
        assert args["as_of_date"] == "2026-01-15"
        assert args["decision_mode"] == "phase2"
        assert args["ruleset"] == "ruleset.json"

    def test_missing_snapshot_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Snapshot directory"):
            make_replay_bundle(
                as_of_date="2099-01-01",
                snapshot_dir=tmp_path / "snapshots",
                out_path=tmp_path / "out.tgz",
                project_root=tmp_path,
            )

    def test_missing_metadata_raises(self, tmp_path):
        snap_dir = tmp_path / "snapshots" / "2026-01-15"
        snap_dir.mkdir(parents=True)
        # Create rankings but no metadata
        _write_csv(snap_dir / "rankings.csv", [{"ticker": "X", "actionable_rank": "1"}])
        _write_json(snap_dir / "decision_ruleset.json", {"id": "x"})

        with pytest.raises(FileNotFoundError, match="metadata.json"):
            make_replay_bundle(
                as_of_date="2026-01-15",
                snapshot_dir=tmp_path / "snapshots",
                out_path=tmp_path / "out.tgz",
                project_root=tmp_path,
            )

    def test_stable_tar_ordering(self, tmp_path):
        """Two builds of the same snapshot should have identical file ordering."""
        root = tmp_path / "project"
        snap_dir = root / "data" / "snapshots"
        _make_synthetic_snapshot(snap_dir, "2026-01-15")
        _make_project_structure(root, "2026-01-15")

        out1 = tmp_path / "bundle1.tgz"
        out2 = tmp_path / "bundle2.tgz"

        make_replay_bundle("2026-01-15", snap_dir, out1, top_k=5, project_root=root)
        make_replay_bundle("2026-01-15", snap_dir, out2, top_k=5, project_root=root)

        with tarfile.open(str(out1), "r:gz") as t1, tarfile.open(str(out2), "r:gz") as t2:
            names1 = t1.getnames()
            names2 = t2.getnames()

        assert names1 == names2

    def test_includes_cache_refresh_when_present(self, tmp_path):
        root = tmp_path / "project"
        snap_dir = root / "data" / "snapshots"
        _make_synthetic_snapshot(snap_dir, "2026-01-15")
        _make_project_structure(root, "2026-01-15")

        # Create cache refresh sidecar
        _write_json(root / "cache" / "cache_refresh_2026-01-15.json", {
            "schema": "cache_refresh.v1",
            "results": [],
        })

        out = tmp_path / "bundle.tgz"
        make_replay_bundle("2026-01-15", snap_dir, out, top_k=5, project_root=root)

        extract_dir = tmp_path / "extracted"
        bundle_dir = extract_bundle(out, extract_dir)
        assert (bundle_dir / "inputs" / "cache_refresh.json").exists()

    def test_includes_sec_8k_cache(self, tmp_path):
        root = tmp_path / "project"
        snap_dir = root / "data" / "snapshots"
        _make_synthetic_snapshot(snap_dir, "2026-01-15")
        _make_project_structure(root, "2026-01-15")

        # Create SEC 8-K cache file
        sec_dir = root / "cache" / "sec" / "8k_catalysts"
        sec_dir.mkdir(parents=True)
        _write_json(sec_dir / "8k_catalysts_2026-01-15_abcd1234.json", [
            {"ticker": "TK01", "event": "8-K"},
        ])

        out = tmp_path / "bundle.tgz"
        make_replay_bundle("2026-01-15", snap_dir, out, top_k=5, project_root=root)

        extract_dir = tmp_path / "extracted"
        bundle_dir = extract_bundle(out, extract_dir)
        sec_files = list((bundle_dir / "inputs" / "sec_8k").iterdir())
        assert len(sec_files) == 1
        assert "8k_catalysts_2026-01-15" in sec_files[0].name

    def test_includes_ctgov_cache(self, tmp_path):
        root = tmp_path / "project"
        snap_dir = root / "data" / "snapshots"
        _make_synthetic_snapshot(snap_dir, "2026-01-15")
        _make_project_structure(root, "2026-01-15")

        # Create CTGov cache file
        ctgov_dir = root / "cache" / "ctgov"
        ctgov_dir.mkdir(parents=True)
        _write_json(ctgov_dir / "trial_records_2026-01-15.json", [{"nct_id": "NCT001"}])

        out = tmp_path / "bundle.tgz"
        make_replay_bundle("2026-01-15", snap_dir, out, top_k=5, project_root=root)

        extract_dir = tmp_path / "extracted"
        bundle_dir = extract_bundle(out, extract_dir)
        ctgov_files = list((bundle_dir / "inputs" / "ctgov").iterdir())
        assert len(ctgov_files) == 1
        assert "trial_records_2026-01-15" in ctgov_files[0].name


# ---------------------------------------------------------------------------
# Tests: verify_bundle_integrity
# ---------------------------------------------------------------------------


class TestVerifyBundleIntegrity:
    def test_intact_bundle_passes(self, tmp_path):
        root = tmp_path / "project"
        snap_dir = root / "data" / "snapshots"
        _make_synthetic_snapshot(snap_dir, "2026-01-15")
        _make_project_structure(root, "2026-01-15")

        out = tmp_path / "bundle.tgz"
        make_replay_bundle("2026-01-15", snap_dir, out, top_k=5, project_root=root)

        extract_dir = tmp_path / "extracted"
        bundle_dir = extract_bundle(out, extract_dir)
        errors = verify_bundle_integrity(bundle_dir)
        assert errors == []

    def test_corrupted_file_detected(self, tmp_path):
        root = tmp_path / "project"
        snap_dir = root / "data" / "snapshots"
        _make_synthetic_snapshot(snap_dir, "2026-01-15")
        _make_project_structure(root, "2026-01-15")

        out = tmp_path / "bundle.tgz"
        make_replay_bundle("2026-01-15", snap_dir, out, top_k=5, project_root=root)

        extract_dir = tmp_path / "extracted"
        bundle_dir = extract_bundle(out, extract_dir)

        # Corrupt a file
        ruleset_path = bundle_dir / "ruleset.json"
        ruleset_path.write_text("corrupted!", encoding="utf-8")

        errors = verify_bundle_integrity(bundle_dir)
        assert len(errors) > 0
        assert any("SHA mismatch" in e and "ruleset.json" in e for e in errors)

    def test_missing_file_detected(self, tmp_path):
        root = tmp_path / "project"
        snap_dir = root / "data" / "snapshots"
        _make_synthetic_snapshot(snap_dir, "2026-01-15")
        _make_project_structure(root, "2026-01-15")

        out = tmp_path / "bundle.tgz"
        make_replay_bundle("2026-01-15", snap_dir, out, top_k=5, project_root=root)

        extract_dir = tmp_path / "extracted"
        bundle_dir = extract_bundle(out, extract_dir)

        # Delete a file
        (bundle_dir / "ruleset.json").unlink()

        errors = verify_bundle_integrity(bundle_dir)
        assert len(errors) > 0
        assert any("Missing file" in e for e in errors)


# ---------------------------------------------------------------------------
# Tests: verify_invariants
# ---------------------------------------------------------------------------


class TestVerifyInvariants:
    def test_valid_bundle_passes(self, tmp_path):
        root = tmp_path / "project"
        snap_dir = root / "data" / "snapshots"
        _make_synthetic_snapshot(snap_dir, "2026-01-15")
        _make_project_structure(root, "2026-01-15")

        out = tmp_path / "bundle.tgz"
        make_replay_bundle("2026-01-15", snap_dir, out, top_k=5, project_root=root)

        extract_dir = tmp_path / "extracted"
        bundle_dir = extract_bundle(out, extract_dir)
        result = verify_invariants(bundle_dir, top_k=5)

        assert result["passed"] is True
        assert all(c["passed"] for c in result["checks"])

    def test_corrupted_bundle_fails(self, tmp_path):
        root = tmp_path / "project"
        snap_dir = root / "data" / "snapshots"
        _make_synthetic_snapshot(snap_dir, "2026-01-15")
        _make_project_structure(root, "2026-01-15")

        out = tmp_path / "bundle.tgz"
        make_replay_bundle("2026-01-15", snap_dir, out, top_k=5, project_root=root)

        extract_dir = tmp_path / "extracted"
        bundle_dir = extract_bundle(out, extract_dir)

        # Corrupt a file to fail integrity check
        (bundle_dir / "ruleset.json").write_text("{}", encoding="utf-8")

        result = verify_invariants(bundle_dir, top_k=5)
        assert result["passed"] is False

        integrity_check = next(c for c in result["checks"] if c["name"] == "file_integrity")
        assert integrity_check["passed"] is False


# ---------------------------------------------------------------------------
# Tests: extract_bundle
# ---------------------------------------------------------------------------


class TestExtractBundle:
    def test_roundtrip(self, tmp_path):
        root = tmp_path / "project"
        snap_dir = root / "data" / "snapshots"
        _make_synthetic_snapshot(snap_dir, "2026-01-15")
        _make_project_structure(root, "2026-01-15")

        out = tmp_path / "bundle.tgz"
        make_replay_bundle("2026-01-15", snap_dir, out, top_k=5, project_root=root)

        extract_dir = tmp_path / "extracted"
        bundle_dir = extract_bundle(out, extract_dir)

        assert bundle_dir.name == "replay_bundle_v1"
        assert (bundle_dir / "manifest.json").exists()
        assert (bundle_dir / "ruleset.json").exists()
        assert (bundle_dir / "replay" / "run_args.json").exists()
        assert (bundle_dir / "replay" / "expected" / "topk_tickers.json").exists()
        assert (bundle_dir / "replay" / "expected" / "metadata_fingerprint.json").exists()


# ---------------------------------------------------------------------------
# Tests: CLI entry points (main)
# ---------------------------------------------------------------------------


class TestMakeBundleCLI:
    def test_missing_snapshot_exits_1(self, tmp_path):
        from tools.make_replay_bundle import main

        rc = main([
            "--as-of-date", "2099-01-01",
            "--snapshot-dir", str(tmp_path / "nonexistent"),
            "--out", str(tmp_path / "out.tgz"),
            "--project-root", str(tmp_path),
        ])
        assert rc == 1

    def test_valid_snapshot_exits_0(self, tmp_path):
        root = tmp_path / "project"
        snap_dir = root / "data" / "snapshots"
        _make_synthetic_snapshot(snap_dir, "2026-01-15")
        _make_project_structure(root, "2026-01-15")

        from tools.make_replay_bundle import main

        rc = main([
            "--as-of-date", "2026-01-15",
            "--snapshot-dir", str(snap_dir),
            "--out", str(tmp_path / "out.tgz"),
            "--project-root", str(root),
            "--top-k", "3",
        ])
        assert rc == 0
        assert (tmp_path / "out.tgz").exists()


class TestReplayBundleCLI:
    def test_missing_bundle_exits_2(self, tmp_path):
        from tools.replay_bundle import main

        rc = main(["--bundle", str(tmp_path / "nonexistent.tgz")])
        assert rc == 2

    def test_valid_bundle_exits_0(self, tmp_path):
        root = tmp_path / "project"
        snap_dir = root / "data" / "snapshots"
        _make_synthetic_snapshot(snap_dir, "2026-01-15")
        _make_project_structure(root, "2026-01-15")

        bundle_path = tmp_path / "bundle.tgz"
        make_replay_bundle("2026-01-15", snap_dir, bundle_path, top_k=5, project_root=root)

        from tools.replay_bundle import main

        rc = main([
            "--bundle", str(bundle_path),
            "--workdir", str(tmp_path / "work"),
        ])
        assert rc == 0
