"""Tests for pit_bundle_health gate in tools/run_daily_production.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from tools.run_daily_production import check_pit_bundle_health, GATE_ALLOWLIST


class TestPitBundleHealthGate:
    def test_gate_in_allowlist(self):
        assert "pit_bundle_health" in GATE_ALLOWLIST

    def test_pass_when_all_caches_exist(self, tmp_path):
        # Create CTgov cache
        ctgov_dir = tmp_path / "ctgov"
        ctgov_dir.mkdir()
        (ctgov_dir / "trial_records_2026-02-20.json").write_text("[]")

        # Create 13F cache
        cache_13f = tmp_path / "13f" / "2026-02-20"
        cache_13f.mkdir(parents=True)
        (cache_13f / "index.json").write_text("{}")

        # Create trial_records.json
        prod = tmp_path / "production_data"
        prod.mkdir()
        (prod / "trial_records.json").write_text("[]")

        # Monkey-patch REPO_ROOT for trial_records check
        import tools.run_daily_production as mod
        orig_root = mod.REPO_ROOT
        mod.REPO_ROOT = tmp_path
        try:
            result = check_pit_bundle_health(
                "2026-02-20",
                ctgov_cache_dir=ctgov_dir,
                cache_13f_root=tmp_path / "13f",
            )
            assert result.status == "PASS"
            assert result.name == "pit_bundle_health"
        finally:
            mod.REPO_ROOT = orig_root

    def test_warn_when_ctgov_missing(self, tmp_path):
        ctgov_dir = tmp_path / "ctgov"
        ctgov_dir.mkdir()
        # No trial_records_{date}.json

        cache_13f = tmp_path / "13f" / "2026-02-20"
        cache_13f.mkdir(parents=True)
        (cache_13f / "index.json").write_text("{}")

        prod = tmp_path / "production_data"
        prod.mkdir()
        (prod / "trial_records.json").write_text("[]")

        import tools.run_daily_production as mod
        orig_root = mod.REPO_ROOT
        mod.REPO_ROOT = tmp_path
        try:
            result = check_pit_bundle_health(
                "2026-02-20",
                ctgov_cache_dir=ctgov_dir,
                cache_13f_root=tmp_path / "13f",
            )
            assert result.status == "WARN"
            assert "CTgov" in result.detail
        finally:
            mod.REPO_ROOT = orig_root

    def test_warn_when_13f_missing(self, tmp_path):
        ctgov_dir = tmp_path / "ctgov"
        ctgov_dir.mkdir()
        (ctgov_dir / "trial_records_2026-02-20.json").write_text("[]")

        # No 13F cache

        prod = tmp_path / "production_data"
        prod.mkdir()
        (prod / "trial_records.json").write_text("[]")

        import tools.run_daily_production as mod
        orig_root = mod.REPO_ROOT
        mod.REPO_ROOT = tmp_path
        try:
            result = check_pit_bundle_health(
                "2026-02-20",
                ctgov_cache_dir=ctgov_dir,
                cache_13f_root=tmp_path / "13f",
            )
            assert result.status == "WARN"
            assert "13F" in result.detail
        finally:
            mod.REPO_ROOT = orig_root

    def test_warn_when_trial_records_missing(self, tmp_path):
        ctgov_dir = tmp_path / "ctgov"
        ctgov_dir.mkdir()
        (ctgov_dir / "trial_records_2026-02-20.json").write_text("[]")

        cache_13f = tmp_path / "13f" / "2026-02-20"
        cache_13f.mkdir(parents=True)
        (cache_13f / "index.json").write_text("{}")

        # No production_data/trial_records.json

        import tools.run_daily_production as mod
        orig_root = mod.REPO_ROOT
        mod.REPO_ROOT = tmp_path
        try:
            result = check_pit_bundle_health(
                "2026-02-20",
                ctgov_cache_dir=ctgov_dir,
                cache_13f_root=tmp_path / "13f",
            )
            assert result.status == "WARN"
            assert "trial_records.json" in result.detail
        finally:
            mod.REPO_ROOT = orig_root

    def test_never_returns_fail(self, tmp_path):
        """Gate is WARN-only, never FAIL."""
        ctgov_dir = tmp_path / "empty_ctgov"
        ctgov_dir.mkdir()

        import tools.run_daily_production as mod
        orig_root = mod.REPO_ROOT
        mod.REPO_ROOT = tmp_path
        try:
            result = check_pit_bundle_health(
                "2026-02-20",
                ctgov_cache_dir=ctgov_dir,
                cache_13f_root=tmp_path / "13f",
            )
            assert result.status in ("PASS", "WARN")
            assert result.status != "FAIL"
        finally:
            mod.REPO_ROOT = orig_root

    def test_warn_aggregates_multiple_issues(self, tmp_path):
        """All caches missing → multiple issues in detail."""
        ctgov_dir = tmp_path / "ctgov"
        ctgov_dir.mkdir()

        import tools.run_daily_production as mod
        orig_root = mod.REPO_ROOT
        mod.REPO_ROOT = tmp_path
        try:
            result = check_pit_bundle_health(
                "2026-02-20",
                ctgov_cache_dir=ctgov_dir,
                cache_13f_root=tmp_path / "13f",
            )
            assert result.status == "WARN"
            # Should mention multiple issues
            assert "CTgov" in result.detail or "13F" in result.detail
        finally:
            mod.REPO_ROOT = orig_root
