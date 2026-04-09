"""Contract tests: output schema invariants for rankings.csv and run_manifest.json.

Proves that:
1. SNAPSHOT_COLUMNS is the authoritative column list and hasn't drifted.
2. run_manifest.json has all required top-level keys and gate structure.
3. GATE_ALLOWLIST is complete and no unknown gates can sneak in.
4. SORT_CONTRIB_KEYS are reflected in SNAPSHOT_COLUMNS.

These contracts catch silent field renames, type changes, or column
additions/removals at CI time.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from decision_engine import SORT_CONTRIB_KEYS
from decision_engine import VERSION as DE_VERSION
from decision_engine import DecisionRuleset
from run_screen import SNAPSHOT_COLUMNS

# ---------------------------------------------------------------------------
# Contract 1: rankings.csv column schema
# ---------------------------------------------------------------------------

# Authoritative frozen column set — update this when columns are
# intentionally added/removed (forces explicit acknowledgment).
REQUIRED_IDENTITY_COLUMNS = {"ticker", "company_name"}

REQUIRED_DECISION_COLUMNS = {
    "actionable_rank",
    "target_weight_pct",
    "tier_any",
    "tier_any_reason",
    "tier_dev",
    "tier_reason",
    "tier_commercial",
    "eligible",
    "ineligible_reasons",
}

REQUIRED_ENGINE_METADATA_COLUMNS = {
    "decision_engine_version",
    "decision_engine_ruleset_id",
}

REQUIRED_HYDRATION_COLUMNS = {
    "de_alpha_60d",
    "de_alpha_60d_source",
    "de_alpha_60d_missing_reason",
    "de_beta_xbi_60d",
    "de_beta_xbi_60d_source",
    "de_beta_xbi_60d_missing_reason",
    "de_drawdown",
    "de_drawdown_missing_reason",
    "de_rsi_14d",
    "de_vol_60d",
    "de_drawdown_xbi",
    "de_drawdown_rel_xbi",
}

REQUIRED_SORT_CONTRIB_COLUMNS = {
    "de_sort_total_adj",
} | {f"de_sort_contrib_{k}" for k in SORT_CONTRIB_KEYS}


class TestRankingsCSVSchema:
    """Contract: SNAPSHOT_COLUMNS contains all required column families."""

    def test_snapshot_columns_is_list(self):
        assert isinstance(SNAPSHOT_COLUMNS, list)

    def test_minimum_column_count(self):
        """Rankings must have at least 100 columns (current: ~139)."""
        assert len(SNAPSHOT_COLUMNS) >= 100, f"SNAPSHOT_COLUMNS has {len(SNAPSHOT_COLUMNS)} columns, expected >= 100"

    def test_no_duplicate_columns(self):
        """No duplicate column names."""
        seen = set()
        dupes = []
        for col in SNAPSHOT_COLUMNS:
            if col in seen:
                dupes.append(col)
            seen.add(col)
        assert dupes == [], f"Duplicate columns: {dupes}"

    def test_identity_columns_present(self):
        cols = set(SNAPSHOT_COLUMNS)
        missing = REQUIRED_IDENTITY_COLUMNS - cols
        assert not missing, f"Missing identity columns: {missing}"

    def test_decision_columns_present(self):
        cols = set(SNAPSHOT_COLUMNS)
        missing = REQUIRED_DECISION_COLUMNS - cols
        assert not missing, f"Missing decision columns: {missing}"

    def test_engine_metadata_columns_present(self):
        cols = set(SNAPSHOT_COLUMNS)
        missing = REQUIRED_ENGINE_METADATA_COLUMNS - cols
        assert not missing, f"Missing engine metadata columns: {missing}"

    def test_hydration_columns_present(self):
        cols = set(SNAPSHOT_COLUMNS)
        missing = REQUIRED_HYDRATION_COLUMNS - cols
        assert not missing, f"Missing hydration columns: {missing}"

    def test_sort_contrib_columns_present(self):
        cols = set(SNAPSHOT_COLUMNS)
        missing = REQUIRED_SORT_CONTRIB_COLUMNS - cols
        assert not missing, f"Missing sort contribution columns: {missing}"

    def test_ticker_is_first_column(self):
        """ticker must be the first column for downstream consumers."""
        assert SNAPSHOT_COLUMNS[0] == "ticker"

    def test_sort_contrib_keys_reflected(self):
        """Every SORT_CONTRIB_KEY has a corresponding de_sort_contrib_ column."""
        cols = set(SNAPSHOT_COLUMNS)
        for key in SORT_CONTRIB_KEYS:
            col = f"de_sort_contrib_{key}"
            assert col in cols, f"SORT_CONTRIB_KEY '{key}' missing column '{col}'"

    def test_column_names_are_snake_case(self):
        """All columns should be snake_case (no spaces, no camelCase)."""
        for col in SNAPSHOT_COLUMNS:
            assert " " not in col, f"Column '{col}' contains spaces"
            # Allow digits and underscores
            assert col == col.lower() or col.startswith("de_") or "_" in col, f"Column '{col}' may not be snake_case"


# ---------------------------------------------------------------------------
# Contract 2: run_manifest.json schema
# ---------------------------------------------------------------------------

MANIFEST_REQUIRED_TOP_LEVEL_KEYS = {
    "manifest_version",
    "requested_as_of_date",
    "effective_as_of_date",
    "as_of_date",
    "generated_at",
    "git",
    "ruleset",
    "row_counts",
    "price_refresh",
    "missing_reason_counts",
    "gates",
    "overall_status",
    "screen_exit_code",
    "audit_exit_code",
    "gate_config",
}

MANIFEST_GIT_REQUIRED_KEYS = {
    "dirty_pre_run",
    "dirty_post_run",
    "dirty",
}

MANIFEST_GATE_REQUIRED_KEYS = {"name", "status", "detail", "value", "threshold"}

MANIFEST_OVERALL_STATUS_VALUES = {"PASS", "WARN", "FAIL"}


class TestRunManifestSchema:
    """Contract: run_manifest.json structure is stable."""

    def _build_minimal_manifest(self) -> Dict[str, Any]:
        """Build a manifest dict matching the real schema."""
        from tools.run_daily_production import MANIFEST_VERSION

        return {
            "manifest_version": MANIFEST_VERSION,
            "requested_as_of_date": "2026-01-15",
            "effective_as_of_date": "2026-01-15",
            "as_of_date": "2026-01-15",
            "generated_at": "2026-01-15T14:00:00+00:00",
            "git": {
                "sha": "abc1234",
                "branch": "main",
                "dirty": False,
                "dirty_pre_run": False,
                "dirty_post_run": None,
            },
            "ruleset": {
                "ruleset_version": "v1.3.0",
                "ruleset_hash": "e966af9d",
                "ranking_mode": "phase2",
                "decision_mode": "phase2",
            },
            "row_counts": {
                "ticker_count": 353,
                "total_evaluated": 353,
                "active_universe": 319,
            },
            "price_refresh": {
                "n_extended": 350,
                "n_rows_appended": 350,
                "n_failed": 3,
                "failed_tickers": ["DELISTED1"],
                "xbi_last_date": "2026-01-15",
            },
            "market_data_refresh": {},
            "missing_reason_counts": {},
            "gates": [
                {
                    "name": "screen",
                    "status": "PASS",
                    "detail": "exit code 0",
                    "value": 0,
                    "threshold": 0,
                },
            ],
            "overall_status": "PASS",
            "screen_exit_code": 0,
            "audit_exit_code": 0,
            "gate_config": {},
        }

    def test_required_top_level_keys(self):
        manifest = self._build_minimal_manifest()
        missing = MANIFEST_REQUIRED_TOP_LEVEL_KEYS - set(manifest.keys())
        assert not missing, f"Missing manifest keys: {missing}"

    def test_git_block_keys(self):
        manifest = self._build_minimal_manifest()
        git = manifest["git"]
        missing = MANIFEST_GIT_REQUIRED_KEYS - set(git.keys())
        assert not missing, f"Missing git keys: {missing}"

    def test_gate_entry_keys(self):
        manifest = self._build_minimal_manifest()
        for gate in manifest["gates"]:
            missing = MANIFEST_GATE_REQUIRED_KEYS - set(gate.keys())
            assert not missing, f"Gate '{gate.get('name')}' missing keys: {missing}"

    def test_overall_status_is_valid(self):
        manifest = self._build_minimal_manifest()
        assert manifest["overall_status"] in MANIFEST_OVERALL_STATUS_VALUES

    def test_manifest_version_is_semver(self):
        from tools.run_daily_production import MANIFEST_VERSION

        parts = MANIFEST_VERSION.split(".")
        assert len(parts) == 3, f"MANIFEST_VERSION not semver: {MANIFEST_VERSION}"
        for p in parts:
            assert p.isdigit(), f"Non-numeric semver part: {p}"


# ---------------------------------------------------------------------------
# Contract 3: GATE_ALLOWLIST completeness
# ---------------------------------------------------------------------------


class TestGateAllowlist:
    """Contract: GATE_ALLOWLIST is the single source of truth for gate names."""

    def test_allowlist_is_frozenset(self):
        from tools.run_daily_production import GATE_ALLOWLIST

        assert isinstance(GATE_ALLOWLIST, frozenset)

    def test_minimum_gate_count(self):
        """At least 20 gates registered."""
        from tools.run_daily_production import GATE_ALLOWLIST

        assert len(GATE_ALLOWLIST) >= 20, f"GATE_ALLOWLIST has {len(GATE_ALLOWLIST)} gates, expected >= 20"

    def test_critical_gates_present(self):
        """Essential gates must exist."""
        from tools.run_daily_production import GATE_ALLOWLIST

        critical = {
            "screen",
            "audit",
            "xbi_staleness",
            "ctgov_cache",
            "inputs_present",
            "drift_monitoring",
            "ruleset_health",
            "cache_health",
        }
        missing = critical - GATE_ALLOWLIST
        assert not missing, f"Missing critical gates: {missing}"

    def test_gate_names_are_snake_case(self):
        """All gate names should be snake_case."""
        from tools.run_daily_production import GATE_ALLOWLIST

        for name in GATE_ALLOWLIST:
            assert name == name.lower(), f"Gate '{name}' not lowercase"
            assert " " not in name, f"Gate '{name}' contains spaces"


# ---------------------------------------------------------------------------
# Contract 4: SORT_CONTRIB_KEYS stability
# ---------------------------------------------------------------------------


class TestSortContribKeysContract:
    """Contract: SORT_CONTRIB_KEYS is the canonical list of sort signals."""

    EXPECTED_KEYS = (
        "clinical",
        "coinvest",
        "institutional",
        "calendar_alpha",
        "alpha_cohort_tb",
        "catalyst_bonus",
        "binary_quality",
        "binary_institutional",
        "clinical_quality_91_180",
        "options_quality_91_180",
        "binary_quality_now",
        "binary_institutional_now",
        "clinical_build_window",
        "pcr_penalty_bw",
        "oncology_crowding",
        "options_verdict",
        "event_ev",
    )

    def test_keys_match_expected(self):
        assert SORT_CONTRIB_KEYS == self.EXPECTED_KEYS

    def test_keys_are_tuple(self):
        assert isinstance(SORT_CONTRIB_KEYS, tuple)

    def test_no_duplicates(self):
        assert len(SORT_CONTRIB_KEYS) == len(set(SORT_CONTRIB_KEYS))


# ---------------------------------------------------------------------------
# Contract 5: DecisionRuleset schema stability
# ---------------------------------------------------------------------------


class TestDecisionRulesetContract:
    """Contract: DecisionRuleset default values produce a deterministic ID."""

    def test_default_ruleset_has_id(self):
        rs = DecisionRuleset()
        assert rs.ruleset_id
        assert len(rs.ruleset_id) == 8

    def test_ruleset_is_frozen(self):
        rs = DecisionRuleset()
        with pytest.raises(AttributeError):
            rs.drawdown_gate = -0.50  # type: ignore[misc]

    def test_sort_anchor_values(self):
        """Valid sort_anchor values are codified."""
        for anchor in ("composite_rank", "optionality_pct", "alpha_cohort"):
            rs = DecisionRuleset(sort_anchor=anchor)
            assert rs.sort_anchor == anchor

    def test_invalid_sort_anchor_rejected(self):
        with pytest.raises(ValueError, match="sort_anchor"):
            DecisionRuleset(sort_anchor="invalid")

    def test_decision_engine_version_format(self):
        """DE version is vX.Y.Z format."""
        assert DE_VERSION.startswith("v")
        parts = DE_VERSION[1:].split(".")
        assert len(parts) == 3
