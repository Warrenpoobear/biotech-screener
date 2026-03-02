"""Regression tests for common.alpha_table resolver.

Guards against the forward-looking alpha table leak in PIT bundles:
  - Historical bundle dates must resolve to per-date OOS tables (not static)
  - Nearest-earlier fallback works when exact date is missing
  - Static fallback when no per-date tables exist at all
  - Policy "never" always returns static
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pytest

from common.alpha_table import (
    check_artifact_marker,
    resolve_alpha_table_path,
    _find_nearest_earlier_table,
)


# ---------------------------------------------------------------------------
# Minimal ruleset stub (avoids importing DecisionRuleset for unit tests)
# ---------------------------------------------------------------------------

class _StubRuleset:
    """Minimal ruleset with the 3 fields the resolver uses."""

    def __init__(
        self,
        policy: str = "if_missing",
        table_path: str = "production_data/alpha_cohort_tables/v1.json",
        train_mode: str = "trailing-6",
        train_horizon: int = 84,
        train_min_dates: int = 6,
        shrink_k: int = 50,
    ):
        self.alpha_table_rebuild_policy = policy
        self.alpha_cohort_table_path = table_path
        self.alpha_train_mode = train_mode
        self.alpha_train_horizon = train_horizon
        self.alpha_train_min_train_dates = train_min_dates
        self.alpha_cohort_shrink_k = shrink_k


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """Create a temp project root with static + daily alpha tables."""
    # Static table
    static_dir = tmp_path / "production_data" / "alpha_cohort_tables"
    static_dir.mkdir(parents=True)
    static_path = static_dir / "v1.json"
    static_path.write_text(json.dumps({
        "cells": {"early|near|pos": {"mean_excess_ret_6m": 0.05, "n": 10}},
        "_build_info": {"as_of_date": "2026-02-17", "source": "static"},
    }))

    # Daily tables
    daily_dir = static_dir / "daily"
    daily_dir.mkdir()

    for dt in ["2024-06-14", "2024-06-21", "2024-06-28", "2025-01-31"]:
        table = {
            "cells": {"early|near|pos": {"mean_excess_ret_6m": 0.03, "n": 8}},
            "_build_info": {"as_of_date": dt, "source": "oos"},
        }
        (daily_dir / f"v1_{dt}.json").write_text(json.dumps(table))

    # One table with artifact marker
    dt_with_marker = "2025-01-31"
    marker = {
        "sha256": _sha256_hex(daily_dir / f"v1_{dt_with_marker}.json"),
        "source": "ci_pipeline",
    }
    (daily_dir / f"v1_{dt_with_marker}.artifact.json").write_text(
        json.dumps(marker)
    )

    return tmp_path


def _sha256_hex(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def logger() -> logging.Logger:
    return logging.getLogger("test_alpha_table_resolver")


# ---------------------------------------------------------------------------
# Tests: policy = "never" → always static
# ---------------------------------------------------------------------------

class TestPolicyNever:
    def test_always_returns_static(self, project_root: Path, logger: logging.Logger):
        ruleset = _StubRuleset(policy="never")
        path, source = resolve_alpha_table_path(
            ruleset, "2024-06-28", project_root, logger,
        )
        assert source == "static"
        assert path.name == "v1.json"

    def test_ignores_dated_tables(self, project_root: Path, logger: logging.Logger):
        """Even when a per-date table exists, policy=never returns static."""
        ruleset = _StubRuleset(policy="never")
        path, source = resolve_alpha_table_path(
            ruleset, "2025-01-31", project_root, logger,
        )
        assert source == "static"


# ---------------------------------------------------------------------------
# Tests: policy = "if_missing" → prefer dated, then nearest, then static
# ---------------------------------------------------------------------------

class TestPolicyIfMissing:
    def test_exact_match_with_artifact(self, project_root: Path, logger: logging.Logger):
        """Dated table with valid artifact marker → source='artifact'."""
        ruleset = _StubRuleset(policy="if_missing")
        path, source = resolve_alpha_table_path(
            ruleset, "2025-01-31", project_root, logger,
        )
        assert source == "artifact"
        assert path.name == "v1_2025-01-31.json"

    def test_exact_match_without_marker(self, project_root: Path, logger: logging.Logger):
        """Dated table without artifact marker → source='preexisting_local'."""
        ruleset = _StubRuleset(policy="if_missing")
        path, source = resolve_alpha_table_path(
            ruleset, "2024-06-28", project_root, logger,
        )
        assert source == "preexisting_local"
        assert path.name == "v1_2024-06-28.json"

    def test_nearest_earlier_fallback(self, project_root: Path, logger: logging.Logger):
        """No exact match but earlier dated tables exist → nearest earlier."""
        ruleset = _StubRuleset(policy="if_missing")
        path, source = resolve_alpha_table_path(
            ruleset, "2024-06-30", project_root, logger, allow_rebuild=False,
        )
        assert source == "nearest_earlier:2024-06-28"
        assert path.name == "v1_2024-06-28.json"

    def test_skips_future_tables(self, project_root: Path, logger: logging.Logger):
        """Nearest earlier must not select a table after as_of_date."""
        ruleset = _StubRuleset(policy="if_missing")
        path, source = resolve_alpha_table_path(
            ruleset, "2024-06-10", project_root, logger, allow_rebuild=False,
        )
        # No tables exist on or before 2024-06-10 → static fallback
        assert "static" in source
        assert path.name == "v1.json"

    def test_static_fallback_when_no_daily_dir(self, tmp_path: Path, logger: logging.Logger):
        """No daily/ directory at all → static fallback."""
        # Create only static table
        static_dir = tmp_path / "production_data" / "alpha_cohort_tables"
        static_dir.mkdir(parents=True)
        (static_dir / "v1.json").write_text("{}")

        ruleset = _StubRuleset(policy="if_missing")
        path, source = resolve_alpha_table_path(
            ruleset, "2024-06-28", tmp_path, logger, allow_rebuild=False,
        )
        assert source == "static_fallback"
        assert path.name == "v1.json"


# ---------------------------------------------------------------------------
# Tests: allow_rebuild=False (bundle mode)
# ---------------------------------------------------------------------------

class TestAllowRebuildFalse:
    def test_no_rebuild_in_bundle_mode(self, project_root: Path, logger: logging.Logger):
        """With allow_rebuild=False and daily policy, should NOT attempt OOS build."""
        ruleset = _StubRuleset(policy="daily")
        # Date that doesn't have exact table → would normally rebuild
        path, source = resolve_alpha_table_path(
            ruleset, "2024-06-30", project_root, logger, allow_rebuild=False,
        )
        # Should fall back to nearest earlier, not rebuild
        assert source == "nearest_earlier:2024-06-28"
        assert "rebuilt" not in source


# ---------------------------------------------------------------------------
# Tests: check_artifact_marker
# ---------------------------------------------------------------------------

class TestCheckArtifactMarker:
    def test_valid_marker(self, project_root: Path):
        daily_dir = project_root / "production_data" / "alpha_cohort_tables" / "daily"
        dated_path = daily_dir / "v1_2025-01-31.json"
        present, valid, data = check_artifact_marker(dated_path)
        assert present is True
        assert valid is True
        assert data["source"] == "ci_pipeline"

    def test_no_marker(self, project_root: Path):
        daily_dir = project_root / "production_data" / "alpha_cohort_tables" / "daily"
        dated_path = daily_dir / "v1_2024-06-28.json"
        present, valid, data = check_artifact_marker(dated_path)
        assert present is False
        assert valid is False

    def test_corrupted_marker(self, project_root: Path):
        daily_dir = project_root / "production_data" / "alpha_cohort_tables" / "daily"
        dated_path = daily_dir / "v1_2024-06-14.json"
        # Write corrupt marker
        marker_path = dated_path.with_suffix(".artifact.json")
        marker_path.write_text("not valid json{{{")
        present, valid, data = check_artifact_marker(dated_path)
        assert present is True
        assert valid is False


# ---------------------------------------------------------------------------
# Tests: _find_nearest_earlier_table
# ---------------------------------------------------------------------------

class TestFindNearestEarlier:
    def test_finds_nearest(self, project_root: Path):
        daily_dir = project_root / "production_data" / "alpha_cohort_tables" / "daily"
        result = _find_nearest_earlier_table(daily_dir, "2024-06-30")
        assert result is not None
        path, dt = result
        assert dt == "2024-06-28"

    def test_returns_none_when_no_earlier(self, project_root: Path):
        daily_dir = project_root / "production_data" / "alpha_cohort_tables" / "daily"
        result = _find_nearest_earlier_table(daily_dir, "2020-01-01")
        assert result is None

    def test_returns_none_when_no_dir(self, tmp_path: Path):
        result = _find_nearest_earlier_table(tmp_path / "nonexistent", "2024-06-30")
        assert result is None


# ---------------------------------------------------------------------------
# Tests: bundle pipeline integration (end-to-end with _resolve_alpha_table_for_bundle)
# ---------------------------------------------------------------------------

class TestBundlePipelineIntegration:
    """Verify the bundle pipeline's _resolve_alpha_table_for_bundle
    delegates to the shared resolver correctly."""

    def test_import_works(self):
        """The bundle module can import the resolver."""
        from scripts.run_screen_from_bundle import _resolve_alpha_table_for_bundle
        assert callable(_resolve_alpha_table_for_bundle)

    def test_resolver_uses_if_missing_policy(self, monkeypatch, logger: logging.Logger):
        """When policy is if_missing, dated tables are preferred."""
        from scripts import run_screen_from_bundle as mod
        from decision_engine import DecisionRuleset
        import dataclasses

        ruleset = dataclasses.replace(
            DecisionRuleset(),
            alpha_table_rebuild_policy="if_missing",
        )

        # Use real project root (has actual daily tables)
        project_root = Path(mod._PROJECT_ROOT)
        daily_dir = project_root / "production_data" / "alpha_cohort_tables" / "daily"

        if not (daily_dir / "v1_2024-06-28.json").exists():
            pytest.skip("No daily alpha tables in project")

        path, source = mod._resolve_alpha_table_for_bundle("2024-06-28", ruleset)
        assert "static" not in source
        assert "v1_2024" in path.name or "v1_2025" in path.name
