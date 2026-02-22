"""Tests for scripts/build_pit_bundle.py — PIT feature bundle builder."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from scripts.build_pit_bundle import (
    SCHEMA_VERSION,
    _sha256_file,
    build_single_bundle,
    discover_buildable_dates,
    validate_bundle,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _make_trial(
    ticker="ACAD",
    first_posted="2025-01-15",
    primary_completion_date="2026-06-15",
    phase="PHASE2",
    study_type="INTERVENTIONAL",
    status="RECRUITING",
    nct_id="NCT12345678",
):
    return {
        "ticker": ticker,
        "study_type": study_type,
        "phase": phase,
        "first_posted": first_posted,
        "last_update_posted": "2025-06-01",
        "primary_completion_date": primary_completion_date,
        "status": status,
        "nct_id": nct_id,
    }


AS_OF = "2026-01-15"


# ===================================================================
# TestSingleBundle
# ===================================================================

class TestSingleBundle:
    """Single bundle creation tests."""

    def test_creates_clinical_and_manifest(self, tmp_path):
        """Creates clinical features + manifest (skip catalyst/coinvest)."""
        trial_path = tmp_path / "trials.json"
        _write_json(trial_path, [_make_trial(ticker="ACAD")])

        manifest = build_single_bundle(
            as_of_date=AS_OF,
            bundle_root=tmp_path / "bundles",
            universe_tickers={"ACAD", "AARD"},
            trial_records_path=trial_path,
            skip_catalyst=True,
            skip_coinvest=True,
        )

        # Manifest created
        assert manifest["schema_version"] == SCHEMA_VERSION
        assert manifest["as_of_date"] == AS_OF
        assert manifest["universe_size"] == 2
        assert manifest["build_duration_seconds"] >= 0

        # Clinical component present
        assert "clinical_features" in manifest["components"]
        clin = manifest["components"]["clinical_features"]
        assert clin["file"] == "clinical_features.json"
        assert clin["schema_version"] == "clinical_features_pit.v1"

        # File exists
        clin_path = tmp_path / "bundles" / AS_OF / "clinical_features.json"
        assert clin_path.exists()

    def test_manifest_hashes_match(self, tmp_path):
        """Manifest SHA-256 hashes match actual file contents."""
        trial_path = tmp_path / "trials.json"
        _write_json(trial_path, [_make_trial()])

        manifest = build_single_bundle(
            as_of_date=AS_OF,
            bundle_root=tmp_path / "bundles",
            universe_tickers={"ACAD"},
            trial_records_path=trial_path,
            skip_catalyst=True,
            skip_coinvest=True,
        )

        for name, info in manifest["components"].items():
            fpath = tmp_path / "bundles" / AS_OF / info["file"]
            actual = _sha256_file(fpath)
            assert actual == info["sha256"], f"hash mismatch for {name}"

    def test_schema_versions_correct(self, tmp_path):
        """Schema versions are correct in manifest components."""
        trial_path = tmp_path / "trials.json"
        _write_json(trial_path, [_make_trial()])

        manifest = build_single_bundle(
            as_of_date=AS_OF,
            bundle_root=tmp_path / "bundles",
            universe_tickers={"ACAD"},
            trial_records_path=trial_path,
            skip_catalyst=True,
            skip_coinvest=True,
        )

        if "clinical_features" in manifest["components"]:
            assert (manifest["components"]["clinical_features"]["schema_version"]
                    == "clinical_features_pit.v1")

    def test_build_duration_recorded(self, tmp_path):
        """Build duration is recorded in manifest."""
        trial_path = tmp_path / "trials.json"
        _write_json(trial_path, [])

        manifest = build_single_bundle(
            as_of_date=AS_OF,
            bundle_root=tmp_path / "bundles",
            universe_tickers={"ACAD"},
            trial_records_path=trial_path,
            skip_catalyst=True,
            skip_coinvest=True,
        )
        assert isinstance(manifest["build_duration_seconds"], float)


# ===================================================================
# TestValidation
# ===================================================================

class TestValidation:
    """Bundle validation tests."""

    def test_valid_bundle_passes(self, tmp_path):
        """Valid bundle passes validation."""
        trial_path = tmp_path / "trials.json"
        _write_json(trial_path, [_make_trial()])

        build_single_bundle(
            as_of_date=AS_OF,
            bundle_root=tmp_path / "bundles",
            universe_tickers={"ACAD"},
            trial_records_path=trial_path,
            skip_catalyst=True,
            skip_coinvest=True,
        )

        bundle_dir = tmp_path / "bundles" / AS_OF
        ok, reason = validate_bundle(bundle_dir)
        assert ok, reason

    def test_tampered_file_fails(self, tmp_path):
        """Tampered file (wrong hash) → validation fails."""
        trial_path = tmp_path / "trials.json"
        _write_json(trial_path, [_make_trial()])

        build_single_bundle(
            as_of_date=AS_OF,
            bundle_root=tmp_path / "bundles",
            universe_tickers={"ACAD"},
            trial_records_path=trial_path,
            skip_catalyst=True,
            skip_coinvest=True,
        )

        # Tamper with clinical file
        clin_path = tmp_path / "bundles" / AS_OF / "clinical_features.json"
        data = json.loads(clin_path.read_text())
        data["tampered"] = True
        clin_path.write_text(json.dumps(data))

        bundle_dir = tmp_path / "bundles" / AS_OF
        ok, reason = validate_bundle(bundle_dir)
        assert not ok
        assert "hash mismatch" in reason

    def test_missing_component_fails(self, tmp_path):
        """Missing component file → validation fails."""
        trial_path = tmp_path / "trials.json"
        _write_json(trial_path, [_make_trial()])

        build_single_bundle(
            as_of_date=AS_OF,
            bundle_root=tmp_path / "bundles",
            universe_tickers={"ACAD"},
            trial_records_path=trial_path,
            skip_catalyst=True,
            skip_coinvest=True,
        )

        # Remove clinical file
        clin_path = tmp_path / "bundles" / AS_OF / "clinical_features.json"
        clin_path.unlink()

        bundle_dir = tmp_path / "bundles" / AS_OF
        ok, reason = validate_bundle(bundle_dir)
        assert not ok
        assert "missing component" in reason


# ===================================================================
# TestDiscoverDates
# ===================================================================

class TestDiscoverDates:
    """Date discovery tests."""

    def test_discovers_date_dirs(self, tmp_path):
        """Discovers correct dates from cache dirs."""
        cache_root = tmp_path / "caches"
        (cache_root / "2026-01-15").mkdir(parents=True)
        (cache_root / "2026-02-01").mkdir(parents=True)
        (cache_root / "not-a-date").mkdir(parents=True)

        dates = discover_buildable_dates(cache_root)
        assert dates == ["2026-01-15", "2026-02-01"]

    def test_empty_root(self, tmp_path):
        """Missing root → empty list."""
        dates = discover_buildable_dates(tmp_path / "nonexistent")
        assert dates == []


# ===================================================================
# TestEdgeCases
# ===================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_skip_clinical_flag(self, tmp_path):
        """--skip-clinical → no clinical component in manifest."""
        trial_path = tmp_path / "trials.json"
        _write_json(trial_path, [_make_trial()])

        manifest = build_single_bundle(
            as_of_date=AS_OF,
            bundle_root=tmp_path / "bundles",
            universe_tickers={"ACAD"},
            trial_records_path=trial_path,
            skip_clinical=True,
            skip_catalyst=True,
            skip_coinvest=True,
        )
        assert "clinical_features" not in manifest["components"]

    def test_all_skipped_empty_components(self, tmp_path):
        """All components skipped → empty components dict."""
        manifest = build_single_bundle(
            as_of_date=AS_OF,
            bundle_root=tmp_path / "bundles",
            universe_tickers={"ACAD"},
            skip_clinical=True,
            skip_catalyst=True,
            skip_coinvest=True,
        )
        assert manifest["components"] == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
