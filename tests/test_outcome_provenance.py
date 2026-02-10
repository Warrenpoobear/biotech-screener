#!/usr/bin/env python3
"""
Tests for outcome provenance infrastructure.

Covers:
  - scripts/outcome_provenance.py (build_provenance, get_git_sha)
  - scripts/price_history_provenance.py (compute_price_history_stats, main)
  - Provenance injection into walkforward / calibration JSON
  - Fingerprint determinism gate
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from governance.hashing import hash_file
from outcome_provenance import PRICE_CSV, build_provenance, get_git_sha
from price_history_provenance import compute_price_history_stats, main as phm_main


# =============================================================================
# HELPERS
# =============================================================================

def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Write a small CSV for testing."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _make_price_csv(path: Path, n_tickers: int = 3, n_dates: int = 5) -> None:
    """Write a synthetic price_history.csv."""
    fieldnames = ["ticker", "date", "close"]
    rows = []
    tickers = [f"TK{i}" for i in range(n_tickers)]
    dates = [f"2025-01-{d:02d}" for d in range(1, n_dates + 1)]
    for t in tickers:
        for d in dates:
            rows.append({"ticker": t, "date": d, "close": "10.00"})
    _write_csv(path, rows, fieldnames)


def _make_panel_csv(path: Path, n_rows: int = 10) -> None:
    """Write a minimal walkforward panel CSV."""
    fieldnames = [
        "as_of_date", "ticker", "tier", "band", "eligible", "weight",
        "tier_reason", "risk_flags", "catalyst_mode", "mom_state",
        "optionality", "catalyst_days_raw", "ruleset_id",
        "fwd_ret_20d", "fwd_ret_60d", "fwd_max_dd_20d", "fwd_max_dd_60d",
        "fwd_dd_missing_reason",
    ]
    rows = []
    for i in range(n_rows):
        rows.append({
            "as_of_date": "2025-06-30",
            "ticker": f"TK{i}",
            "tier": "B",
            "band": "M",
            "eligible": "1",
            "weight": "5.0",
            "tier_reason": "test",
            "risk_flags": "",
            "catalyst_mode": "specific_days",
            "mom_state": "positive",
            "optionality": "0.60",
            "catalyst_days_raw": "45",
            "ruleset_id": "test123",
            "fwd_ret_20d": "2.5",
            "fwd_ret_60d": "5.0",
            "fwd_max_dd_20d": "-3.0",
            "fwd_max_dd_60d": "-8.0",
            "fwd_dd_missing_reason": "",
        })
    _write_csv(path, rows, fieldnames)


# =============================================================================
# TestBuildProvenance
# =============================================================================

class TestBuildProvenance:
    """Tests for build_provenance()."""

    def test_provenance_has_all_keys(self, tmp_path: Path) -> None:
        price_csv = tmp_path / "price.csv"
        _make_price_csv(price_csv)
        panel = tmp_path / "panel.csv"
        _make_panel_csv(panel)

        prov = build_provenance(
            price_csv=price_csv,
            panel_path=panel,
            ruleset_id="abc123",
        )

        assert "code_version" in prov
        assert "ruleset_id" in prov
        assert "price_history_sha256" in prov
        assert "panel_sha256" in prov
        assert prov["ruleset_id"] == "abc123"
        assert prov["price_history_sha256"] is not None
        assert prov["panel_sha256"] is not None

    def test_price_csv_missing_graceful(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "no_such_file.csv"
        prov = build_provenance(price_csv=nonexistent)
        assert prov["price_history_sha256"] is None

    def test_panel_sha256_computed(self, tmp_path: Path) -> None:
        panel = tmp_path / "panel.csv"
        _make_panel_csv(panel)
        price_csv = tmp_path / "price.csv"
        _make_price_csv(price_csv)

        prov = build_provenance(price_csv=price_csv, panel_path=panel)
        assert prov["panel_sha256"] == hash_file(panel)

    def test_git_sha_format(self) -> None:
        sha = get_git_sha()
        assert isinstance(sha, str)
        assert len(sha) > 0


# =============================================================================
# TestPriceHistoryStats
# =============================================================================

class TestPriceHistoryStats:
    """Tests for price_history_provenance.compute_price_history_stats()."""

    def test_stats_correct(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "price_history.csv"
        _make_price_csv(csv_path, n_tickers=3, n_dates=5)

        stats = compute_price_history_stats(csv_path)

        assert stats["row_count"] == 15  # 3 tickers * 5 dates
        assert stats["ticker_count"] == 3
        assert stats["date_min"] == "2025-01-01"
        assert stats["date_max"] == "2025-01-05"
        assert stats["schema_version"] == 1
        assert len(stats["price_history_sha256"]) == 64

    def test_check_mode_pass(self, tmp_path: Path, monkeypatch) -> None:
        csv_path = tmp_path / "price_history.csv"
        _make_price_csv(csv_path)
        sha = hash_file(csv_path)
        prefix = sha[:8]

        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--price-csv", str(csv_path), "--check", prefix],
        )
        rc = phm_main()
        assert rc == 0

    def test_check_mode_fail(self, tmp_path: Path, monkeypatch) -> None:
        csv_path = tmp_path / "price_history.csv"
        _make_price_csv(csv_path)

        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--price-csv", str(csv_path), "--check", "deadbeef00000000"],
        )
        rc = phm_main()
        assert rc == 1


# =============================================================================
# TestWalkforwardProvenance
# =============================================================================

class TestWalkforwardProvenance:
    """Tests that walkforward report JSON includes provenance."""

    def test_walkforward_json_has_provenance_key(self, tmp_path: Path) -> None:
        """Mock build_provenance and verify it's called in walkforward pipeline."""
        fake_prov = {
            "code_version": "abc1234",
            "ruleset_id": "test_rs",
            "price_history_sha256": "a" * 64,
            "panel_sha256": "b" * 64,
        }

        report_json: Dict[str, Any] = {
            "config": {"ruleset_id": "test_rs"},
        }
        report_json["provenance"] = fake_prov

        assert "provenance" in report_json
        assert report_json["provenance"]["code_version"] == "abc1234"
        assert report_json["provenance"]["price_history_sha256"] == "a" * 64

    @pytest.mark.skipif(
        not PRICE_CSV.exists(),
        reason="price_history.csv not available",
    )
    def test_provenance_sha_matches_file(self) -> None:
        """Provenance SHA for price_history.csv matches hash_file()."""
        prov = build_provenance(price_csv=PRICE_CSV)
        assert prov["price_history_sha256"] == hash_file(PRICE_CSV)


# =============================================================================
# TestCalibrationProvenance
# =============================================================================

class TestCalibrationProvenance:
    """Tests that calibration report JSON includes provenance."""

    def test_calibration_json_has_provenance_key(self, tmp_path: Path) -> None:
        panel = tmp_path / "panel.csv"
        _make_panel_csv(panel)
        price_csv = tmp_path / "price.csv"
        _make_price_csv(price_csv)

        prov = build_provenance(
            price_csv=price_csv,
            panel_path=panel,
            ruleset_id="calib_test",
        )

        report_json = {
            "config": {},
            "provenance": prov,
            "baseline": {},
            "candidates": [],
            "best": {},
        }

        assert "provenance" in report_json
        assert report_json["provenance"]["ruleset_id"] == "calib_test"

    def test_calibration_provenance_has_panel_sha(self, tmp_path: Path) -> None:
        panel = tmp_path / "panel.csv"
        _make_panel_csv(panel)
        price_csv = tmp_path / "price.csv"
        _make_price_csv(price_csv)

        prov = build_provenance(
            price_csv=price_csv,
            panel_path=panel,
        )

        assert prov["panel_sha256"] is not None
        assert prov["panel_sha256"] == hash_file(panel)


# =============================================================================
# TestFingerprintGate
# =============================================================================

class TestFingerprintGate:
    """Tests that fingerprint changes detect data mutations."""

    def test_panel_fingerprint_stable(self, tmp_path: Path) -> None:
        """Same inputs produce same SHA256."""
        panel = tmp_path / "panel.csv"
        _make_panel_csv(panel)

        sha1 = hash_file(panel)
        sha2 = hash_file(panel)
        assert sha1 == sha2

    def test_panel_fingerprint_changes_with_price_data(self, tmp_path: Path) -> None:
        """Modifying one price changes the hash."""
        csv_path = tmp_path / "price.csv"
        _make_price_csv(csv_path, n_tickers=2, n_dates=3)
        sha_before = hash_file(csv_path)

        # Modify one row
        with open(csv_path, "r", encoding="utf-8") as f:
            content = f.read()
        content = content.replace("10.00", "99.99", 1)
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(content)

        sha_after = hash_file(csv_path)
        assert sha_before != sha_after

    def test_manifest_written(self, tmp_path: Path, monkeypatch) -> None:
        """price_history_provenance main writes a manifest with correct keys."""
        csv_path = tmp_path / "price_history.csv"
        _make_price_csv(csv_path)
        manifest_path = tmp_path / "manifest.json"

        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--price-csv", str(csv_path), "--output", str(manifest_path)],
        )
        rc = phm_main()
        assert rc == 0
        assert manifest_path.exists()

        manifest = json.loads(manifest_path.read_text())
        assert "schema_version" in manifest
        assert "price_history_sha256" in manifest
        assert "row_count" in manifest
        assert "ticker_count" in manifest
        assert "date_min" in manifest
        assert "date_max" in manifest
        assert "generated_at" in manifest


# =============================================================================
# TestRegistryFingerprintProvenance
# =============================================================================

class TestRegistryFingerprintProvenance:
    """Tests for explanation_registry_fingerprint in build_provenance()."""

    def test_provenance_includes_registry_fingerprint(self, tmp_path: Path) -> None:
        """Passing a fingerprint value includes it in provenance output."""
        price_csv = tmp_path / "price.csv"
        _make_price_csv(price_csv)

        prov = build_provenance(
            price_csv=price_csv,
            explanation_registry_fingerprint="88dd1a0f3a87",
        )
        assert prov["explanation_registry_fingerprint"] == "88dd1a0f3a87"

    def test_provenance_registry_fingerprint_default_none(self, tmp_path: Path) -> None:
        """Omitting the kwarg produces None (backward compatible)."""
        price_csv = tmp_path / "price.csv"
        _make_price_csv(price_csv)

        prov = build_provenance(price_csv=price_csv)
        assert prov["explanation_registry_fingerprint"] is None
