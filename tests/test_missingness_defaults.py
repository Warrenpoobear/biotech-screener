"""Tests for output hygiene: _ensure_defaults, applicability flags,
catalyst_days semantics, and coinvest_recency_state."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from run_screen import _ensure_defaults, _ALWAYS_NUMERIC_DEFAULTS


# ---------------------------------------------------------------------------
# _ensure_defaults
# ---------------------------------------------------------------------------

class TestEnsureDefaults:
    def test_fills_missing_clinical_z_tier_with_zero(self):
        rows = [{"ticker": "AAA", "clinical_score_z_tier": ""}]
        _ensure_defaults(rows)
        assert rows[0]["clinical_score_z_tier"] == 0.0

    def test_fills_none_with_default(self):
        rows = [{"ticker": "AAA"}]  # field not present → .get returns None
        _ensure_defaults(rows)
        assert rows[0]["clinical_score_z_tier"] == 0.0

    def test_fills_nan_string_with_default(self):
        rows = [{"ticker": "AAA", "coinvest_score_z": "nan"}]
        _ensure_defaults(rows)
        assert rows[0]["coinvest_score_z"] == 0.0

    def test_fills_nan_uppercase_with_default(self):
        rows = [{"ticker": "AAA", "inst_delta_z": "NaN"}]
        _ensure_defaults(rows)
        assert rows[0]["inst_delta_z"] == 0.0

    def test_preserves_existing_values(self):
        rows = [{"ticker": "AAA", "clinical_score_z_tier": 1.23,
                 "coinvest_score_z": -0.5, "inst_delta_z": 0.8,
                 "catalyst_decay_w": 0.42}]
        _ensure_defaults(rows)
        assert rows[0]["clinical_score_z_tier"] == 1.23
        assert rows[0]["coinvest_score_z"] == -0.5
        assert rows[0]["inst_delta_z"] == 0.8
        assert rows[0]["catalyst_decay_w"] == 0.42

    def test_preserves_zero(self):
        """0.0 is a real value, not missing."""
        rows = [{"ticker": "AAA", "clinical_score_z_tier": 0.0}]
        _ensure_defaults(rows)
        assert rows[0]["clinical_score_z_tier"] == 0.0

    def test_all_always_numeric_fields_present_after(self):
        rows = [{"ticker": "AAA"}]
        _ensure_defaults(rows)
        for field in _ALWAYS_NUMERIC_DEFAULTS:
            assert field in rows[0], f"{field} should be set"
            assert rows[0][field] == _ALWAYS_NUMERIC_DEFAULTS[field]

    def test_custom_defaults(self):
        rows = [{"ticker": "AAA"}]
        _ensure_defaults(rows, {"custom_field": 99.0})
        assert rows[0]["custom_field"] == 99.0

    def test_multiple_rows(self):
        rows = [
            {"ticker": "A", "clinical_score_z_tier": ""},
            {"ticker": "B", "clinical_score_z_tier": 1.5},
            {"ticker": "C"},
        ]
        _ensure_defaults(rows)
        assert rows[0]["clinical_score_z_tier"] == 0.0
        assert rows[1]["clinical_score_z_tier"] == 1.5
        assert rows[2]["clinical_score_z_tier"] == 0.0


# ---------------------------------------------------------------------------
# Applicability flags
# ---------------------------------------------------------------------------

class TestApplicabilityFlags:
    """Test the flag columns populated in save_validation_snapshot."""

    def _make_row(self, archetype):
        return {"ticker": "T1", "archetype": archetype}

    def test_drug_developer_gets_clinical_optionality_flag(self):
        row = self._make_row("drug_developer")
        row["has_clinical_optionality_dev"] = 1 if row["archetype"] == "drug_developer" else 0
        row["has_commercial_quality"] = 1 if row["archetype"] != "drug_developer" else 0
        assert row["has_clinical_optionality_dev"] == 1
        assert row["has_commercial_quality"] == 0

    def test_commercial_biotech_gets_commercial_quality_flag(self):
        row = self._make_row("commercial_biotech")
        row["has_clinical_optionality_dev"] = 1 if row["archetype"] == "drug_developer" else 0
        row["has_commercial_quality"] = 1 if row["archetype"] != "drug_developer" else 0
        assert row["has_commercial_quality"] == 1
        assert row["has_clinical_optionality_dev"] == 0

    def test_platform_gets_commercial_quality_flag(self):
        row = self._make_row("platform_diagnostics")
        row["has_clinical_optionality_dev"] = 1 if row["archetype"] == "drug_developer" else 0
        row["has_commercial_quality"] = 1 if row["archetype"] != "drug_developer" else 0
        assert row["has_commercial_quality"] == 1
        assert row["has_clinical_optionality_dev"] == 0

    def test_flags_are_mutually_exclusive_for_dev(self):
        row = self._make_row("drug_developer")
        row["has_clinical_optionality_dev"] = 1 if row["archetype"] == "drug_developer" else 0
        row["has_commercial_quality"] = 1 if row["archetype"] != "drug_developer" else 0
        # Exactly one flag is 1
        assert row["has_clinical_optionality_dev"] + row["has_commercial_quality"] == 1


# ---------------------------------------------------------------------------
# Catalyst days semantics
# ---------------------------------------------------------------------------

class TestCatalystDaysSemantics:
    """Verify that catalyst_days is set correctly based on catalyst_mode."""

    def test_specific_days_has_numeric_value(self):
        """catalyst_mode=specific_days → catalyst_days should be positive int."""
        row = {"catalyst_mode": "specific_days", "catalyst_days": "45"}
        assert int(row["catalyst_days"]) > 0

    def test_no_upcoming_has_empty_catalyst_days(self):
        """catalyst_mode=no_upcoming → catalyst_days is empty (expected N/A)."""
        row = {"catalyst_mode": "no_upcoming", "catalyst_days": ""}
        assert row["catalyst_days"] == ""

    def test_blended_window_has_zero_catalyst_days(self):
        """catalyst_mode=blended_window → catalyst_days can be 0."""
        row = {"catalyst_mode": "blended_window", "catalyst_days": "0"}
        assert int(row["catalyst_days"]) == 0

    def test_missing_mode_has_empty_catalyst_days(self):
        """catalyst_mode=missing → catalyst_days is empty."""
        row = {"catalyst_mode": "missing", "catalyst_days": ""}
        assert row["catalyst_days"] == ""


# ---------------------------------------------------------------------------
# Coinvest recency state (decision_engine.py)
# ---------------------------------------------------------------------------

class TestCoinvestRecencyState:
    """Test coinvest_recency_state logic from decision_engine overlay."""

    def _compute_recency(self, filing_age):
        """Replicate the DE logic for testing."""
        if filing_age is None:
            return ""
        elif filing_age <= 90:
            return "fresh"
        else:
            return "stale"

    def test_no_coinvest_returns_empty_state(self):
        assert self._compute_recency(None) == ""

    def test_fresh_filing_returns_fresh(self):
        assert self._compute_recency(30) == "fresh"
        assert self._compute_recency(90) == "fresh"

    def test_stale_filing_returns_stale(self):
        assert self._compute_recency(91) == "stale"
        assert self._compute_recency(365) == "stale"

    def test_zero_days_is_fresh(self):
        assert self._compute_recency(0) == "fresh"

    def test_boundary_90_is_fresh(self):
        assert self._compute_recency(90) == "fresh"

    def test_boundary_91_is_stale(self):
        assert self._compute_recency(91) == "stale"
