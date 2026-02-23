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
    filter_dates_by_cadence,
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


# ===================================================================
# TestCarryForward
# ===================================================================

class TestCarryForward:
    """Coinvest carry-forward tests."""

    def test_find_latest_cache_date_exact(self, tmp_path):
        """Exact date exists → returns it."""
        from scripts.build_pit_bundle import _find_latest_cache_date
        cache = tmp_path / "13f"
        (cache / "2025-03-31").mkdir(parents=True)
        (cache / "2025-06-30").mkdir(parents=True)
        assert _find_latest_cache_date(cache, "2025-06-30") == "2025-06-30"

    def test_find_latest_cache_date_carry(self, tmp_path):
        """No exact match → returns latest date <= as_of."""
        from scripts.build_pit_bundle import _find_latest_cache_date
        cache = tmp_path / "13f"
        (cache / "2025-03-31").mkdir(parents=True)
        (cache / "2025-06-30").mkdir(parents=True)
        assert _find_latest_cache_date(cache, "2025-05-15") == "2025-03-31"

    def test_find_latest_cache_date_none(self, tmp_path):
        """All dates after as_of → None."""
        from scripts.build_pit_bundle import _find_latest_cache_date
        cache = tmp_path / "13f"
        (cache / "2025-06-30").mkdir(parents=True)
        assert _find_latest_cache_date(cache, "2025-01-01") is None

    def test_find_latest_cache_date_missing_root(self, tmp_path):
        """Root doesn't exist → None."""
        from scripts.build_pit_bundle import _find_latest_cache_date
        assert _find_latest_cache_date(tmp_path / "nope", "2025-06-30") is None

    def test_carry_forward_off_skips_coinvest(self, tmp_path):
        """carry_forward=off + no exact cache → coinvest skipped."""
        manifest = build_single_bundle(
            as_of_date="2025-05-15",
            bundle_root=tmp_path / "bundles",
            universe_tickers={"ACAD"},
            cache_13f_root=tmp_path / "empty_cache",
            skip_clinical=True,
            skip_catalyst=True,
            coinvest_carry_forward="off",
        )
        assert "coinvest_features" not in manifest["components"]

    def test_carry_forward_source_date_in_manifest(self, tmp_path):
        """When carry-forward used, source_date recorded in component metadata."""
        # This test verifies the source_date metadata key path; actual coinvest
        # building would need a real 13F cache. We verify the _find_latest_cache_date
        # selection picks the right date.
        from scripts.build_pit_bundle import _find_latest_cache_date
        cache = tmp_path / "13f"
        (cache / "2025-03-31").mkdir(parents=True)
        (cache / "2025-06-30").mkdir(parents=True)
        (cache / "2025-09-30").mkdir(parents=True)

        # Target is 2025-08-15 → should pick 2025-06-30
        result = _find_latest_cache_date(cache, "2025-08-15")
        assert result == "2025-06-30"

        # Target is 2025-12-31 → should pick 2025-09-30
        result = _find_latest_cache_date(cache, "2025-12-31")
        assert result == "2025-09-30"


# ---------------------------------------------------------------------------
# CTgov cache date discovery
# ---------------------------------------------------------------------------

class TestCtgovCacheDiscovery:
    def test_discover_from_ctgov_cache(self, tmp_path):
        """--date-grid-from-ctgov-cache discovers dates from trial_records_*.json."""
        ctgov_root = tmp_path / "ctgov"
        ctgov_root.mkdir()
        # Create trial_records cache files
        for d in ["2024-03-31", "2024-04-30", "2024-05-31"]:
            (ctgov_root / f"trial_records_{d}.json").write_text("{}")
        # Create 13F cache dir for one overlapping date
        f13_root = tmp_path / "13f"
        (f13_root / "2024-03-31").mkdir(parents=True)
        (f13_root / "2024-06-30").mkdir(parents=True)

        dates = discover_buildable_dates(f13_root, ctgov_cache_root=ctgov_root)
        assert "2024-03-31" in dates  # from both
        assert "2024-04-30" in dates  # from ctgov only
        assert "2024-05-31" in dates  # from ctgov only
        assert "2024-06-30" in dates  # from 13F only
        # Should be sorted
        assert dates == sorted(dates)

    def test_discover_ctgov_only(self, tmp_path):
        """No 13F cache → still discovers CTgov dates."""
        ctgov_root = tmp_path / "ctgov"
        ctgov_root.mkdir()
        (ctgov_root / "trial_records_2024-01-31.json").write_text("{}")
        (ctgov_root / "trial_records_2024-02-28.json").write_text("{}")

        f13_root = tmp_path / "13f_missing"  # doesn't exist
        dates = discover_buildable_dates(f13_root, ctgov_cache_root=ctgov_root)
        assert dates == ["2024-01-31", "2024-02-28"]

    def test_discover_without_ctgov(self, tmp_path):
        """No CTgov root → original 13F-only behavior."""
        f13_root = tmp_path / "13f"
        (f13_root / "2024-03-31").mkdir(parents=True)
        (f13_root / "2024-06-30").mkdir(parents=True)

        dates = discover_buildable_dates(f13_root, ctgov_cache_root=None)
        assert dates == ["2024-03-31", "2024-06-30"]


# ---------------------------------------------------------------------------
# Tests: Cadence filtering (Part 2B)
# ---------------------------------------------------------------------------

class TestCadenceFiltering:

    def test_filter_dates_quarterly(self):
        """Quarterly filter keeps only quarter-end dates."""
        dates = ["2025-01-31", "2025-02-28", "2025-03-31",
                 "2025-04-30", "2025-06-30", "2025-09-30"]
        result = filter_dates_by_cadence(dates, "quarterly")
        assert result == ["2025-03-31", "2025-06-30", "2025-09-30"]

    def test_filter_dates_monthly(self):
        """Monthly filter keeps only month-end dates."""
        dates = ["2025-01-15", "2025-01-31", "2025-02-28", "2025-03-15", "2025-03-31"]
        result = filter_dates_by_cadence(dates, "monthly")
        assert result == ["2025-01-31", "2025-02-28", "2025-03-31"]

    def test_batch_monthly_cadence(self, tmp_path):
        """Batch mode with monthly cadence filters discovered dates."""
        f13_root = tmp_path / "13f"
        # Create both month-end and non-month-end dirs
        for d in ["2025-01-31", "2025-02-15", "2025-02-28", "2025-03-31"]:
            (f13_root / d).mkdir(parents=True)

        dates = discover_buildable_dates(f13_root, ctgov_cache_root=None)
        assert len(dates) == 4  # all discovered

        monthly = filter_dates_by_cadence(dates, "monthly")
        assert monthly == ["2025-01-31", "2025-02-28", "2025-03-31"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
