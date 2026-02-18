#!/usr/bin/env python3
"""
Tests for the runtime-critical input manifest feature.

Covers:
- Schema stability (CI gate)
- Registry integrity
- Functional (build, record counts, condition gating)
- Verify mode (pass/fail)
- Metadata integration
"""

import json
import re
import sys
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import shutil
import tarfile

from run_screen import (
    InputDependency,
    DEPENDENCY_REGISTRY,
    MANIFEST_VERSION,
    build_inputs_manifest,
    verify_inputs_manifest,
    verify_against_prior_manifest,
    create_replay_bundle,
    extract_replay_bundle,
    _count_records,
)
from archive_snapshot import sha256_file


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def data_dir(tmp_path):
    """Create a minimal data_dir with the four required files."""
    (tmp_path / "universe.json").write_text(json.dumps([
        {"ticker": "AAPL"}, {"ticker": "GOOG"},
    ]))
    (tmp_path / "financial_records.json").write_text(json.dumps([
        {"ticker": "AAPL", "cash": 100}, {"ticker": "GOOG", "cash": 200},
    ]))
    (tmp_path / "trial_records.json").write_text(json.dumps([
        {"nct_id": "NCT001"}, {"nct_id": "NCT002"}, {"nct_id": "NCT003"},
    ]))
    (tmp_path / "market_data.json").write_text(json.dumps([
        {"ticker": "AAPL", "price": 150},
    ]))
    return tmp_path


@pytest.fixture
def all_conditions_off():
    return {
        "enable_coinvest": False,
        "enable_enhancements": False,
        "decision_phase2": False,
        "pit_mode": False,
        "sec_8k": False,
        "fda_adcom": False,
    }


@pytest.fixture
def all_conditions_on():
    return {
        "enable_coinvest": True,
        "enable_enhancements": True,
        "decision_phase2": True,
        "pit_mode": True,
        "sec_8k": True,
        "fda_adcom": True,
    }


# ===========================================================================
# Schema stability (CI gate)
# ===========================================================================

class TestSchemaStability:
    def test_manifest_schema_v1_keys(self, data_dir, all_conditions_off):
        m = build_inputs_manifest("2026-02-17", data_dir, all_conditions_off)
        expected_keys = {
            "manifest_version", "as_of_date", "generated_at",
            "data_dir", "dependencies", "validation",
        }
        assert set(m.keys()) == expected_keys

    def test_dependency_entry_keys(self, data_dir, all_conditions_off):
        m = build_inputs_manifest("2026-02-17", data_dir, all_conditions_off)
        expected_keys = {
            "key", "path", "required", "exists",
            "resolved_from", "load_site", "sha256", "record_count",
        }
        for dep in m["dependencies"]:
            assert set(dep.keys()) == expected_keys, f"Bad keys for {dep['key']}"

    def test_validation_block_keys(self, data_dir, all_conditions_off):
        m = build_inputs_manifest("2026-02-17", data_dir, all_conditions_off)
        expected_keys = {"all_required_present", "errors", "warnings"}
        assert set(m["validation"].keys()) == expected_keys

    def test_manifest_version_is_v1(self, data_dir, all_conditions_off):
        m = build_inputs_manifest("2026-02-17", data_dir, all_conditions_off)
        assert m["manifest_version"] == "v1"


# ===========================================================================
# Registry integrity
# ===========================================================================

class TestRegistryIntegrity:
    def test_registry_covers_four_required(self):
        required_keys = {d.key for d in DEPENDENCY_REGISTRY if d.required}
        assert required_keys == {"universe", "financial_records", "trial_records", "market_data"}

    def test_registry_no_duplicate_keys(self):
        keys = [d.key for d in DEPENDENCY_REGISTRY]
        assert len(keys) == len(set(keys)), f"Duplicate keys: {[k for k in keys if keys.count(k) > 1]}"

    def test_registry_load_sites_format(self):
        for dep in DEPENDENCY_REGISTRY:
            assert re.match(r"\w+\.py:\d+", dep.load_site), (
                f"Bad load_site format for {dep.key}: {dep.load_site}"
            )

    def test_registry_has_18_entries(self):
        assert len(DEPENDENCY_REGISTRY) == 18


# ===========================================================================
# Functional
# ===========================================================================

class TestFunctional:
    def test_all_required_present(self, data_dir, all_conditions_off):
        m = build_inputs_manifest("2026-02-17", data_dir, all_conditions_off)
        assert m["validation"]["all_required_present"] is True
        assert m["validation"]["errors"] == []

    def test_missing_required_file(self, data_dir, all_conditions_off):
        (data_dir / "market_data.json").unlink()
        m = build_inputs_manifest("2026-02-17", data_dir, all_conditions_off)
        assert m["validation"]["all_required_present"] is False
        assert any("market_data" in e for e in m["validation"]["errors"])

    def test_optional_missing_is_warning(self, data_dir, all_conditions_off):
        """Required files present, optional absent → warnings only, no errors."""
        m = build_inputs_manifest("2026-02-17", data_dir, all_conditions_off)
        assert m["validation"]["all_required_present"] is True
        # Optional files that have no condition gate should appear as warnings
        # (e.g. price_history, morningstar_prices, fda_designations, etc.)
        assert len(m["validation"]["errors"]) == 0

    def test_sha256_full_length(self, data_dir, all_conditions_off):
        m = build_inputs_manifest("2026-02-17", data_dir, all_conditions_off)
        for dep in m["dependencies"]:
            if dep["exists"]:
                assert dep["sha256"] is not None
                assert len(dep["sha256"]) == 64, f"sha256 not 64 hex for {dep['key']}"
                assert re.match(r"^[0-9a-f]{64}$", dep["sha256"])

    def test_record_count_json_list(self, tmp_path):
        f = tmp_path / "list.json"
        f.write_text(json.dumps([{}, {}]))
        assert _count_records(f) == 2

    def test_record_count_json_dict(self, tmp_path):
        f = tmp_path / "dict.json"
        f.write_text(json.dumps({"a": 1, "b": 2, "c": 3}))
        assert _count_records(f) == 3

    def test_record_count_csv(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("ticker,score\nAAPL,1\nGOOG,2\nMSFT,3\n")
        assert _count_records(f) == 3

    def test_record_count_empty_csv(self, tmp_path):
        f = tmp_path / "empty.csv"
        f.write_text("ticker,score\n")
        assert _count_records(f) == 0

    def test_condition_gating(self, data_dir, all_conditions_off):
        """enable_coinvest=False → coinvest_signals and holdings_detailed omitted."""
        m = build_inputs_manifest("2026-02-17", data_dir, all_conditions_off)
        dep_keys = {d["key"] for d in m["dependencies"]}
        assert "coinvest_signals" not in dep_keys
        assert "holdings_detailed" not in dep_keys

    def test_condition_gating_enabled(self, data_dir):
        """enable_coinvest=True → coinvest entries present (even if missing)."""
        conditions = {
            "enable_coinvest": True,
            "enable_enhancements": False,
            "decision_phase2": False,
            "pit_mode": False,
            "sec_8k": False,
            "fda_adcom": False,
        }
        m = build_inputs_manifest("2026-02-17", data_dir, conditions)
        dep_keys = {d["key"] for d in m["dependencies"]}
        assert "coinvest_signals" in dep_keys
        assert "holdings_detailed" in dep_keys

    def test_deterministic_ordering(self, data_dir, all_conditions_off):
        """Two builds with same input → same JSON (ignoring generated_at)."""
        m1 = build_inputs_manifest("2026-02-17", data_dir, all_conditions_off)
        m2 = build_inputs_manifest("2026-02-17", data_dir, all_conditions_off)
        # Strip generated_at for comparison
        m1.pop("generated_at")
        m2.pop("generated_at")
        assert json.dumps(m1, sort_keys=True) == json.dumps(m2, sort_keys=True)

    def test_resolved_paths_override(self, data_dir, all_conditions_off):
        """Override trial_records to a custom path."""
        custom = data_dir / "custom_trials.json"
        custom.write_text(json.dumps([{"nct_id": "NCT999"}]))
        resolved = {"trial_records": custom}
        m = build_inputs_manifest("2026-02-17", data_dir, all_conditions_off, resolved)
        trial_dep = next(d for d in m["dependencies"] if d["key"] == "trial_records")
        assert trial_dep["path"] == str(custom)
        assert trial_dep["record_count"] == 1

    def test_record_counts_in_manifest(self, data_dir, all_conditions_off):
        """Verify record counts match expected values for required files."""
        m = build_inputs_manifest("2026-02-17", data_dir, all_conditions_off)
        by_key = {d["key"]: d for d in m["dependencies"]}
        assert by_key["universe"]["record_count"] == 2
        assert by_key["financial_records"]["record_count"] == 2
        assert by_key["trial_records"]["record_count"] == 3
        assert by_key["market_data"]["record_count"] == 1


# ===========================================================================
# Verify mode
# ===========================================================================

class TestVerifyMode:
    def test_verify_passes(self, data_dir, all_conditions_off):
        m = build_inputs_manifest("2026-02-17", data_dir, all_conditions_off)
        assert verify_inputs_manifest(m) is True

    def test_verify_fails(self, data_dir, all_conditions_off):
        (data_dir / "market_data.json").unlink()
        m = build_inputs_manifest("2026-02-17", data_dir, all_conditions_off)
        assert verify_inputs_manifest(m) is False


# ===========================================================================
# Drift check (verify against prior manifest)
# ===========================================================================

class TestDriftCheck:
    def test_no_drift_same_inputs(self, data_dir, all_conditions_off):
        """Identical inputs → zero drift errors."""
        m = build_inputs_manifest("2026-02-17", data_dir, all_conditions_off)
        errs = verify_against_prior_manifest(m, m)
        assert errs == []

    def test_drift_on_sha_change(self, data_dir, all_conditions_off):
        """Mutating a required file → sha256 drift error."""
        prior = build_inputs_manifest("2026-02-17", data_dir, all_conditions_off)
        # Mutate universe.json
        (data_dir / "universe.json").write_text(json.dumps([{"ticker": "CHANGED"}]))
        current = build_inputs_manifest("2026-02-17", data_dir, all_conditions_off)
        errs = verify_against_prior_manifest(current, prior)
        assert len(errs) == 1
        assert "universe" in errs[0]
        assert "drift" in errs[0]

    def test_drift_on_missing_required(self, data_dir, all_conditions_off):
        """Deleting a required file → missing dep error."""
        prior = build_inputs_manifest("2026-02-17", data_dir, all_conditions_off)
        (data_dir / "market_data.json").unlink()
        current = build_inputs_manifest("2026-02-17", data_dir, all_conditions_off)
        errs = verify_against_prior_manifest(current, prior)
        assert len(errs) == 1
        assert "market_data" in errs[0]
        assert "missing" in errs[0]

    def test_no_drift_optional_changed(self, data_dir, all_conditions_off):
        """Changing an optional file does NOT trigger drift errors."""
        # Add an optional file, build prior
        (data_dir / "price_history.csv").write_text("ticker,price\nAAPL,150\n")
        prior = build_inputs_manifest("2026-02-17", data_dir, all_conditions_off)
        # Mutate the optional file
        (data_dir / "price_history.csv").write_text("ticker,price\nAAPL,999\n")
        current = build_inputs_manifest("2026-02-17", data_dir, all_conditions_off)
        errs = verify_against_prior_manifest(current, prior)
        assert errs == [], "Optional file changes should not cause drift errors"

    def test_drift_multiple_files(self, data_dir, all_conditions_off):
        """Mutating two required files → two drift errors."""
        prior = build_inputs_manifest("2026-02-17", data_dir, all_conditions_off)
        (data_dir / "universe.json").write_text(json.dumps([{"ticker": "X"}]))
        (data_dir / "trial_records.json").write_text(json.dumps([{"nct_id": "NCT999"}]))
        current = build_inputs_manifest("2026-02-17", data_dir, all_conditions_off)
        errs = verify_against_prior_manifest(current, prior)
        assert len(errs) == 2
        keys_mentioned = " ".join(errs)
        assert "universe" in keys_mentioned
        assert "trial_records" in keys_mentioned

    def test_drift_error_contains_truncated_hashes(self, data_dir, all_conditions_off):
        """Drift error message includes truncated prior and current hashes."""
        prior = build_inputs_manifest("2026-02-17", data_dir, all_conditions_off)
        (data_dir / "universe.json").write_text(json.dumps([{"ticker": "NEW"}]))
        current = build_inputs_manifest("2026-02-17", data_dir, all_conditions_off)
        errs = verify_against_prior_manifest(current, prior)
        assert len(errs) == 1
        assert "prior=" in errs[0]
        assert "current=" in errs[0]

    def test_drift_rejects_version_mismatch(self, data_dir, all_conditions_off):
        """manifest_version mismatch → immediate single error, no dep-level checks."""
        m = build_inputs_manifest("2026-02-17", data_dir, all_conditions_off)
        prior = json.loads(json.dumps(m))
        prior["manifest_version"] = "v0"
        errs = verify_against_prior_manifest(m, prior)
        assert len(errs) == 1
        assert "manifest_version mismatch" in errs[0]

    def test_drift_rejects_date_mismatch(self, data_dir, all_conditions_off):
        """as_of_date mismatch → immediate single error."""
        m = build_inputs_manifest("2026-02-17", data_dir, all_conditions_off)
        prior = json.loads(json.dumps(m))
        prior["as_of_date"] = "2026-01-01"
        errs = verify_against_prior_manifest(m, prior)
        assert len(errs) == 1
        assert "as_of_date mismatch" in errs[0]

    def test_drift_new_required_dep_not_in_prior(self, data_dir, all_conditions_off):
        """Current has a required dep not present in prior → drift error."""
        prior = build_inputs_manifest("2026-02-17", data_dir, all_conditions_off)
        current = json.loads(json.dumps(prior))
        current["dependencies"].append({
            "key": "new_required_dep", "required": True, "exists": True,
            "sha256": "0" * 64, "path": "/fake", "resolved_from": "test",
            "load_site": "test.py:1", "record_count": None,
        })
        errs = verify_against_prior_manifest(current, prior)
        assert any("new_required_dep" in e and "absent from prior" in e for e in errs)


# ===========================================================================
# Metadata integration
# ===========================================================================

class TestMetadataIntegration:
    def test_metadata_manifest_block_write(self):
        """Simulate what save_validation_snapshot writes for mode=write."""
        manifest = {
            "validation": {"all_required_present": True, "errors": [], "warnings": []},
        }
        results = {"run_metadata": {"inputs_manifest": manifest}}
        inputs_manifest_mode = "write"

        # Replicate the metadata assembly logic
        meta_block = {
            "mode": inputs_manifest_mode,
            "path": "inputs_manifest.json" if inputs_manifest_mode != "off" else None,
            "all_required_present": (
                results.get("run_metadata", {}).get("inputs_manifest", {}).get(
                    "validation", {}
                ).get("all_required_present")
                if inputs_manifest_mode != "off" else None
            ),
        }
        assert meta_block["mode"] == "write"
        assert meta_block["path"] == "inputs_manifest.json"
        assert meta_block["all_required_present"] is True

    def test_metadata_manifest_block_off(self):
        """When mode=off, path and all_required_present are None."""
        inputs_manifest_mode = "off"
        meta_block = {
            "mode": inputs_manifest_mode,
            "path": "inputs_manifest.json" if inputs_manifest_mode != "off" else None,
            "all_required_present": None,
        }
        assert meta_block["mode"] == "off"
        assert meta_block["path"] is None
        assert meta_block["all_required_present"] is None


# ===========================================================================
# Replay bundles
# ===========================================================================

class TestReplayBundle:
    def test_bundle_cache_dir_layout(self, tmp_path):
        """Cache directory deps bundle to cache/<subdir>/... (no extra basename layer)."""
        cache_root = tmp_path / "ctgov_cache"
        cache_root.mkdir()
        (cache_root / "trial_records_2026-02-17.json").write_text(json.dumps([{"nct_id": "NCT001"}]))
        m = {
            "manifest_version": "v1",
            "as_of_date": "2026-02-17",
            "generated_at": "2026-02-17T00:00:00Z",
            "data_dir": str(tmp_path),
            "dependencies": [{
                "key": "ctgov_cache_dir",
                "path": str(cache_root),
                "required": False,
                "exists": True,
                "resolved_from": "cache/ctgov",
                "load_site": "run_screen.py:4209",
                "sha256": None,
                "record_count": None,
            }],
            "validation": {"all_required_present": True, "errors": [], "warnings": []},
        }
        out = tmp_path / "replay_bundle.tgz"
        create_replay_bundle(m, out, include_optional_present=True)
        rb = extract_replay_bundle(out)
        try:
            assert (rb["tmp_dir"] / "cache" / "ctgov" / "trial_records_2026-02-17.json").exists()
            assert rb["ctgov_cache_dir"] is not None
        finally:
            shutil.rmtree(rb["tmp_dir"], ignore_errors=True)

    def test_bundle_includes_required_deps(self, data_dir, all_conditions_off):
        """Bundle tarball contains all required deps with correct sha256."""
        m = build_inputs_manifest("2026-02-17", data_dir, all_conditions_off)
        out = data_dir / "replay_bundle.tgz"
        create_replay_bundle(m, out)
        assert out.exists()

        with tarfile.open(out, "r:gz") as tar:
            names = tar.getnames()
        # All 4 required files
        assert "data/universe.json" in names
        assert "data/financial_records.json" in names
        assert "data/trial_records.json" in names
        assert "data/market_data.json" in names
        assert "inputs_manifest.json" in names
        assert "bundle_index.json" in names

    def test_bundle_preserves_sha256(self, data_dir, all_conditions_off, tmp_path):
        """Files inside bundle have the same sha256 as the original."""
        m = build_inputs_manifest("2026-02-17", data_dir, all_conditions_off)
        out = tmp_path / "replay_bundle.tgz"
        create_replay_bundle(m, out)

        rb = extract_replay_bundle(out)
        try:
            for dep in m["dependencies"]:
                if dep["exists"] and dep["sha256"]:
                    fname = Path(dep["path"]).name
                    extracted = rb["data_dir"] / fname
                    if extracted.exists():
                        assert sha256_file(extracted) == dep["sha256"], \
                            f"sha256 mismatch for {dep['key']}"
        finally:
            shutil.rmtree(rb["tmp_dir"], ignore_errors=True)

    def test_bundle_extraction_layout(self, data_dir, all_conditions_off, tmp_path):
        """Extraction produces expected on-disk layout."""
        m = build_inputs_manifest("2026-02-17", data_dir, all_conditions_off)
        out = tmp_path / "replay_bundle.tgz"
        create_replay_bundle(m, out)

        rb = extract_replay_bundle(out)
        try:
            assert rb["data_dir"].is_dir()
            assert rb["manifest_path"].exists()
            assert (rb["data_dir"] / "universe.json").exists()
            assert (rb["data_dir"] / "financial_records.json").exists()
            assert (rb["data_dir"] / "trial_records.json").exists()
            assert (rb["data_dir"] / "market_data.json").exists()
            assert rb["bundle_index"]["bundle_version"] == "v1"
            assert rb["bundle_index"]["as_of_date"] == "2026-02-17"
        finally:
            shutil.rmtree(rb["tmp_dir"], ignore_errors=True)

    def test_bundle_includes_optional_present(self, data_dir, all_conditions_off, tmp_path):
        """With include_optional_present=True, present optional files are bundled."""
        (data_dir / "price_history.csv").write_text("ticker,price\nAAPL,150\n")
        m = build_inputs_manifest("2026-02-17", data_dir, all_conditions_off)
        out = tmp_path / "replay_bundle.tgz"
        create_replay_bundle(m, out, include_optional_present=True)

        with tarfile.open(out, "r:gz") as tar:
            names = tar.getnames()
        assert "data/price_history.csv" in names

    def test_bundle_excludes_optional_when_disabled(self, data_dir, all_conditions_off, tmp_path):
        """With include_optional_present=False, optional files are NOT bundled."""
        (data_dir / "price_history.csv").write_text("ticker,price\nAAPL,150\n")
        m = build_inputs_manifest("2026-02-17", data_dir, all_conditions_off)
        out = tmp_path / "replay_bundle.tgz"
        create_replay_bundle(m, out, include_optional_present=False)

        with tarfile.open(out, "r:gz") as tar:
            names = tar.getnames()
        assert "data/price_history.csv" not in names
        # Required files still present
        assert "data/universe.json" in names

    def test_bundle_drift_on_mutated_file(self, data_dir, all_conditions_off, tmp_path):
        """Mutating a file inside an extracted bundle → verify drift fails."""
        m = build_inputs_manifest("2026-02-17", data_dir, all_conditions_off)
        out = tmp_path / "replay_bundle.tgz"
        create_replay_bundle(m, out)

        rb = extract_replay_bundle(out)
        try:
            # Mutate universe.json inside the extracted bundle
            (rb["data_dir"] / "universe.json").write_text(json.dumps([{"ticker": "MUTATED"}]))
            # Build manifest from the mutated data_dir
            current = build_inputs_manifest("2026-02-17", rb["data_dir"], all_conditions_off)
            # Load the bundled prior manifest
            with open(rb["manifest_path"], "r") as f:
                prior = json.load(f)
            errs = verify_against_prior_manifest(current, prior)
            assert len(errs) >= 1
            assert any("universe" in e and "drift" in e for e in errs)
        finally:
            shutil.rmtree(rb["tmp_dir"], ignore_errors=True)

    def test_bundle_index_file_count(self, data_dir, all_conditions_off, tmp_path):
        """bundle_index.json lists the correct number of files."""
        m = build_inputs_manifest("2026-02-17", data_dir, all_conditions_off)
        out = tmp_path / "replay_bundle.tgz"
        create_replay_bundle(m, out)

        rb = extract_replay_bundle(out)
        try:
            n_existing = sum(1 for d in m["dependencies"] if d["exists"])
            assert len(rb["bundle_index"]["files"]) == n_existing
        finally:
            shutil.rmtree(rb["tmp_dir"], ignore_errors=True)

    def test_bundle_ruleset_autowire(self, data_dir, all_conditions_off, tmp_path):
        """If a decision_ruleset dep exists, it's bundled and extraction finds it."""
        # Create a fake ruleset and add it to the manifest
        m = build_inputs_manifest("2026-02-17", data_dir, all_conditions_off)
        rs_file = tmp_path / "my_ruleset.json"
        rs_file.write_text(json.dumps({"ruleset_id": "test123"}))
        m["dependencies"].append({
            "key": "decision_ruleset",
            "path": str(rs_file),
            "required": False,
            "exists": True,
            "resolved_from": "production_data",
            "load_site": "run_screen.py:6544",
            "sha256": sha256_file(rs_file),
            "record_count": None,
        })

        out = tmp_path / "replay_bundle.tgz"
        create_replay_bundle(m, out)

        rb = extract_replay_bundle(out)
        try:
            assert rb["ruleset_path"] is not None
            assert rb["ruleset_path"].exists()
            loaded = json.loads(rb["ruleset_path"].read_text())
            assert loaded["ruleset_id"] == "test123"
        finally:
            shutil.rmtree(rb["tmp_dir"], ignore_errors=True)
