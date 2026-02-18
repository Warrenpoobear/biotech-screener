#!/usr/bin/env python3
"""
Tests for alpha_signal_contract.py - Alpha Signal Contract + Schema Enforcement

Tests cover:
- Minimal valid rec passes validation
- Missing required fields are flagged as errors
- Missing recommended fields are flagged as warnings (not errors)
- Trace fields present or allowed missing when optional
- warn vs fail mode behavior
- Output validation for alpha cohort fields
- Diagnostics structure
"""

import pytest
from typing import Any, Dict, List

from alpha_signal_contract import (
    ALPHA_CONTRACT_VERSION,
    CATALYST_TRACE_KEYS,
    AlphaContractDiagnostics,
    AlphaContractViolation,
    validate_alpha_inputs,
    validate_alpha_outputs,
    _resolve_nested,
    _check_rec_fields,
    _check_row_fields,
)


# =============================================================================
# FIXTURES
# =============================================================================

def _make_valid_rec(ticker: str = "ACAD") -> Dict[str, Any]:
    """Build a minimal valid rec dict that passes all required checks."""
    return {
        "ticker": ticker,
        "catalyst_decay": {
            "days_to_catalyst": 45,
            "in_optimal_window": True,
        },
        "defensive_features": {
            "drawdown": -0.25,
            "drawdown_rel_xbi": -0.10,
            "vol_60d": 0.55,
            "beta_xbi_60d": 1.2,
            "rsi_14d": 52.0,
        },
        "severity": "NONE",
        "smart_money_signal": {"overlap_count": 3},
        "coinvest": {"tier1_count": 2},
        "score_breakdown": {
            "enhancements": {
                "momentum": {"alpha_60d": 0.05},
            },
        },
        "confidence_overall": 0.72,
    }


def _make_valid_row(ticker: str = "ACAD", archetype: str = "drug_developer") -> Dict[str, Any]:
    """Build a minimal valid csv_row dict that passes all required checks."""
    return {
        "ticker": ticker,
        "archetype": archetype,
        "clinical_score_z": 0.85,
        "clinical_alpha_z": 1.2,
        "clinical_readout_days": 45,
        "clinical_coverage_flag": "full",
        "clinical_score_z_tier": 0.65,
        "alpha_cohort_key": "mid|near_31_90|pos",
        "alpha_cohort_raw": 0.035,
        "alpha_cohort_pct": 0.72,
    }


# =============================================================================
# TestResolveNested
# =============================================================================

class TestResolveNested:
    def test_simple_path(self):
        d = {"a": {"b": 42}}
        found, val = _resolve_nested(d, ("a", "b"))
        assert found is True
        assert val == 42

    def test_missing_key(self):
        d = {"a": {"b": 42}}
        found, val = _resolve_nested(d, ("a", "c"))
        assert found is False

    def test_missing_parent(self):
        d = {"x": 1}
        found, val = _resolve_nested(d, ("a", "b"))
        assert found is False

    def test_single_key(self):
        d = {"severity": "SEV1"}
        found, val = _resolve_nested(d, ("severity",))
        assert found is True
        assert val == "SEV1"

    def test_none_value_is_found(self):
        """None value counts as present — key exists."""
        d = {"catalyst_decay": {"days_to_catalyst": None}}
        found, val = _resolve_nested(d, ("catalyst_decay", "days_to_catalyst"))
        assert found is True
        assert val is None


# =============================================================================
# TestInputValidation
# =============================================================================

class TestInputValidation:
    """Tests for validate_alpha_inputs."""

    def test_minimal_valid_rec_passes(self):
        """Fully populated rec + row passes with zero errors."""
        rec = _make_valid_rec("ACAD")
        row = _make_valid_row("ACAD")

        diag = validate_alpha_inputs(
            {"ACAD": rec}, [row], schema_mode="warn",
        )

        assert diag.total_errors == 0
        assert diag.n_tickers_checked == 1
        assert not diag.has_errors

    def test_missing_catalyst_decay_days_to_catalyst_is_error(self):
        """Missing catalyst_decay.days_to_catalyst flagged as error."""
        rec = _make_valid_rec("ACAD")
        del rec["catalyst_decay"]["days_to_catalyst"]
        row = _make_valid_row("ACAD")

        diag = validate_alpha_inputs(
            {"ACAD": rec}, [row], schema_mode="warn",
        )

        assert diag.total_errors >= 1
        assert "catalyst_decay.days_to_catalyst" in diag.error_field_counts

    def test_missing_catalyst_decay_entirely_is_error(self):
        """Missing catalyst_decay dict entirely -> both sub-keys error."""
        rec = _make_valid_rec("ACAD")
        del rec["catalyst_decay"]
        row = _make_valid_row("ACAD")

        diag = validate_alpha_inputs(
            {"ACAD": rec}, [row], schema_mode="warn",
        )

        assert diag.total_errors >= 2
        assert "catalyst_decay.days_to_catalyst" in diag.error_field_counts
        assert "catalyst_decay.in_optimal_window" in diag.error_field_counts

    def test_missing_clinical_score_z_is_error_for_drug_developer(self):
        """Missing clinical_score_z is an error for drug_developer archetype."""
        rec = _make_valid_rec("ACAD")
        row = _make_valid_row("ACAD", archetype="drug_developer")
        del row["clinical_score_z"]

        diag = validate_alpha_inputs(
            {"ACAD": rec}, [row], schema_mode="warn",
        )

        assert diag.total_errors >= 1
        assert "clinical_score_z" in diag.error_field_counts

    def test_missing_clinical_score_z_is_warning_for_platform(self):
        """Missing clinical_score_z is only a warning for platform archetypes."""
        rec = _make_valid_rec("PLAT")
        row = _make_valid_row("PLAT", archetype="platform_diagnostics")
        del row["clinical_score_z"]

        diag = validate_alpha_inputs(
            {"PLAT": rec}, [row], schema_mode="warn",
        )

        assert diag.total_errors == 0
        assert "clinical_score_z" in diag.warning_field_counts

    def test_clinical_score_z_required_for_commercial_biotech(self):
        """clinical_score_z is required for commercial_biotech (not just drug_developer)."""
        rec = _make_valid_rec("MRNA")
        row = _make_valid_row("MRNA", archetype="commercial_biotech")
        del row["clinical_score_z"]

        diag = validate_alpha_inputs(
            {"MRNA": rec}, [row], schema_mode="warn",
        )

        assert "clinical_score_z" in diag.error_field_counts

    def test_missing_severity_is_error(self):
        """Missing severity on rec is an error."""
        rec = _make_valid_rec("ACAD")
        del rec["severity"]
        row = _make_valid_row("ACAD")

        diag = validate_alpha_inputs(
            {"ACAD": rec}, [row], schema_mode="warn",
        )

        assert "severity" in diag.error_field_counts

    def test_trace_fields_missing_is_warning_not_error(self):
        """Traceability fields (nearest_catalyst_event_id etc.) are recommended,
        not required — missing is a warning, not an error."""
        rec = _make_valid_rec("ACAD")
        # Ensure no trace fields
        assert "nearest_catalyst_event_id" not in rec.get("catalyst_decay", {})

        row = _make_valid_row("ACAD")

        diag = validate_alpha_inputs(
            {"ACAD": rec}, [row], schema_mode="warn",
        )

        assert diag.total_errors == 0  # no errors
        assert diag.total_warnings >= 1  # trace fields are warnings
        assert "catalyst_decay.nearest_catalyst_event_id" in diag.warning_field_counts

        # Derived n_missing_catalyst_trace reflects the 3 trace fields
        d = diag.to_dict()
        assert d["n_missing_catalyst_trace"] == 3  # 1 ticker × 3 trace fields
        assert d["n_missing_catalyst_trace"] == sum(
            d["top_missing_recommended"].get(k, 0) for k in CATALYST_TRACE_KEYS
        )

    def test_recommended_clinical_fields_missing_is_warning(self):
        """Missing clinical_alpha_z (recommended) -> warning, not error."""
        rec = _make_valid_rec("ACAD")
        row = _make_valid_row("ACAD")
        del row["clinical_alpha_z"]

        diag = validate_alpha_inputs(
            {"ACAD": rec}, [row], schema_mode="warn",
        )

        assert diag.total_errors == 0
        assert "clinical_alpha_z" in diag.warning_field_counts

    def test_none_values_are_acceptable(self):
        """None values for required fields are OK — key must exist, value may be None."""
        rec = _make_valid_rec("ACAD")
        rec["catalyst_decay"]["days_to_catalyst"] = None
        rec["catalyst_decay"]["in_optimal_window"] = None
        row = _make_valid_row("ACAD")

        diag = validate_alpha_inputs(
            {"ACAD": rec}, [row], schema_mode="warn",
        )

        assert diag.total_errors == 0

    def test_multiple_tickers(self):
        """Contract checks all tickers, accumulates counts."""
        recs = {
            "ACAD": _make_valid_rec("ACAD"),
            "MRNA": _make_valid_rec("MRNA"),
        }
        # Break MRNA's catalyst
        del recs["MRNA"]["catalyst_decay"]["days_to_catalyst"]

        rows = [_make_valid_row("ACAD"), _make_valid_row("MRNA")]

        diag = validate_alpha_inputs(recs, rows, schema_mode="warn")

        assert diag.n_tickers_checked == 2
        assert diag.n_tickers_with_errors == 1
        assert diag.total_errors == 1

    def test_missing_rec_is_warning_not_error(self):
        """Rows with no matching rec surface as 'missing_rec' warning."""
        recs = {"ACAD": _make_valid_rec("ACAD")}
        rows = [_make_valid_row("ACAD"), _make_valid_row("MRNA")]  # MRNA has no rec

        diag = validate_alpha_inputs(recs, rows, schema_mode="warn")

        assert diag.n_tickers_checked == 2  # both ACAD and MRNA checked
        assert diag.total_errors == 0
        assert diag.total_warnings >= 1
        assert "alpha_contract.missing_rec" in diag.warning_field_counts
        assert diag.warning_field_counts["alpha_contract.missing_rec"] == 1

        # Derived convenience key in to_dict() must match source of truth
        d = diag.to_dict()
        assert d["n_missing_rec"] == d["top_missing_recommended"].get("alpha_contract.missing_rec", 0)

    def test_empty_ticker_skipped(self):
        """Rows with empty or whitespace-only ticker are silently skipped."""
        recs = {"ACAD": _make_valid_rec("ACAD")}
        row_empty = _make_valid_row("ACAD")
        row_empty["ticker"] = ""
        row_blank = _make_valid_row("ACAD")
        row_blank["ticker"] = "  "
        rows = [_make_valid_row("ACAD"), row_empty, row_blank]

        diag = validate_alpha_inputs(recs, rows, schema_mode="warn")

        assert diag.n_tickers_checked == 1  # only real ACAD
        assert diag.total_errors == 0

    def test_fail_mode_does_not_raise_on_missing_rec(self):
        """schema_mode='fail' does not raise on missing-rec (warning only)."""
        recs = {}  # no recs at all
        rows = [_make_valid_row("MRNA")]

        # Should not raise
        diag = validate_alpha_inputs(recs, rows, schema_mode="fail")

        assert diag.total_errors == 0
        assert "alpha_contract.missing_rec" in diag.warning_field_counts


# =============================================================================
# TestFailMode
# =============================================================================

class TestFailMode:
    """Tests for schema_mode='fail' behavior."""

    def test_fail_mode_raises_on_error(self):
        """schema_mode='fail' raises AlphaContractViolation on required field missing."""
        rec = _make_valid_rec("ACAD")
        del rec["catalyst_decay"]
        row = _make_valid_row("ACAD")

        with pytest.raises(AlphaContractViolation) as exc_info:
            validate_alpha_inputs(
                {"ACAD": rec}, [row], schema_mode="fail",
            )

        assert exc_info.value.diagnostics.total_errors >= 1

    def test_fail_mode_passes_when_valid(self):
        """schema_mode='fail' does not raise when all required fields present."""
        rec = _make_valid_rec("ACAD")
        row = _make_valid_row("ACAD")

        diag = validate_alpha_inputs(
            {"ACAD": rec}, [row], schema_mode="fail",
        )

        assert diag.total_errors == 0

    def test_fail_mode_does_not_raise_on_warnings_only(self):
        """Warnings alone (missing recommended fields) do not trigger failure."""
        rec = _make_valid_rec("ACAD")
        row = _make_valid_row("ACAD")
        del row["clinical_alpha_z"]  # recommended, not required

        # Should not raise
        diag = validate_alpha_inputs(
            {"ACAD": rec}, [row], schema_mode="fail",
        )

        assert diag.total_errors == 0
        assert diag.total_warnings >= 1


# =============================================================================
# TestOutputValidation
# =============================================================================

class TestOutputValidation:
    """Tests for validate_alpha_outputs."""

    def test_valid_outputs_pass(self):
        """Row with all alpha output fields passes."""
        row = _make_valid_row("ACAD")

        diag = validate_alpha_outputs([row], schema_mode="warn", alpha_cohort_enabled=True)

        assert diag.total_errors == 0

    def test_missing_alpha_cohort_key_is_error(self):
        """Missing alpha_cohort_key flagged as error."""
        row = _make_valid_row("ACAD")
        del row["alpha_cohort_key"]

        diag = validate_alpha_outputs([row], schema_mode="warn", alpha_cohort_enabled=True)

        assert "alpha_cohort_key" in diag.error_field_counts

    def test_missing_alpha_cohort_pct_is_error(self):
        row = _make_valid_row("ACAD")
        del row["alpha_cohort_pct"]

        diag = validate_alpha_outputs([row], schema_mode="warn", alpha_cohort_enabled=True)

        assert "alpha_cohort_pct" in diag.error_field_counts

    def test_disabled_alpha_cohort_skips_validation(self):
        """When alpha_cohort_enabled=False, output validation is skipped."""
        row = {"ticker": "ACAD"}  # no alpha fields at all

        diag = validate_alpha_outputs([row], schema_mode="warn", alpha_cohort_enabled=False)

        assert diag.n_tickers_checked == 0
        assert diag.total_errors == 0

    def test_fail_mode_raises_on_missing_output(self):
        row = _make_valid_row("ACAD")
        del row["alpha_cohort_raw"]

        with pytest.raises(AlphaContractViolation):
            validate_alpha_outputs([row], schema_mode="fail", alpha_cohort_enabled=True)

    def test_missing_clinical_score_z_tier_is_warning(self):
        """clinical_score_z_tier is recommended, not required."""
        row = _make_valid_row("ACAD")
        del row["clinical_score_z_tier"]

        diag = validate_alpha_outputs([row], schema_mode="warn", alpha_cohort_enabled=True)

        assert diag.total_errors == 0
        assert "clinical_score_z_tier" in diag.warning_field_counts


# =============================================================================
# TestDiagnostics
# =============================================================================

class TestDiagnostics:
    """Tests for AlphaContractDiagnostics structure."""

    def test_diagnostics_to_dict_structure(self):
        """to_dict() returns expected keys."""
        diag = AlphaContractDiagnostics()
        diag.record("ACAD", ["catalyst_decay.days_to_catalyst"], ["coinvest"])

        d = diag.to_dict()

        assert d["alpha_contract_version"] == ALPHA_CONTRACT_VERSION
        assert d["n_tickers_checked"] == 1
        assert d["n_tickers_with_errors"] == 1
        assert d["total_errors"] == 1
        assert d["total_warnings"] == 1
        assert "catalyst_decay.days_to_catalyst" in d["top_missing_required"]
        assert "coinvest" in d["top_missing_recommended"]
        assert len(d["sample_errors"]) == 1
        assert d["sample_errors"][0]["ticker"] == "ACAD"
        # Derived convenience keys are consistent with source of truth
        assert d["n_missing_rec"] == 0
        assert d["n_missing_rec"] == d["top_missing_recommended"].get("alpha_contract.missing_rec", 0)
        assert d["n_missing_catalyst_trace"] == sum(
            d["top_missing_recommended"].get(k, 0) for k in CATALYST_TRACE_KEYS
        )

    def test_diagnostics_has_errors_property(self):
        diag = AlphaContractDiagnostics()
        assert not diag.has_errors

        diag.record("ACAD", ["some_field"], [])
        assert diag.has_errors

    def test_sample_errors_capped(self):
        """Only first N sample errors are stored."""
        diag = AlphaContractDiagnostics()
        for i in range(10):
            diag.record(f"TICK{i}", ["field_a"], [], max_samples=3)

        assert len(diag.sample_errors) == 3
        assert diag.n_tickers_with_errors == 10

    def test_contract_version_set(self):
        diag = AlphaContractDiagnostics()
        assert diag.contract_version == ALPHA_CONTRACT_VERSION


# =============================================================================
# TestMetadataIntegration
# =============================================================================

class TestMetadataIntegration:
    """Tests that contract diagnostics produce the expected metadata shape."""

    def test_telemetry_shape_for_metadata_json(self):
        """Simulates the alpha_contract_telemetry block written to metadata.json.

        Verifies that input + output diagnostics together produce the expected
        structure: {input: {...}, output: {...}} with stable keys.
        """
        rec = _make_valid_rec("ACAD")
        row = _make_valid_row("ACAD")

        input_diag = validate_alpha_inputs(
            {"ACAD": rec}, [row], schema_mode="warn",
        )
        output_diag = validate_alpha_outputs(
            [row], schema_mode="warn", alpha_cohort_enabled=True,
        )

        # Build the telemetry block exactly as run_screen.py does
        telemetry = {
            "input": input_diag.to_dict(),
            "output": output_diag.to_dict(),
        }

        # Stable keys present
        for section in ("input", "output"):
            d = telemetry[section]
            assert "alpha_contract_version" in d
            assert d["alpha_contract_version"] == ALPHA_CONTRACT_VERSION
            assert "n_tickers_checked" in d
            assert "total_errors" in d
            assert "total_warnings" in d
            assert "n_missing_rec" in d
            assert isinstance(d["n_missing_rec"], int)
            assert "n_missing_catalyst_trace" in d
            assert isinstance(d["n_missing_catalyst_trace"], int)
            assert "top_missing_required" in d
            assert "top_missing_recommended" in d
            assert "sample_errors" in d

        # Valid input -> no errors
        assert telemetry["input"]["total_errors"] == 0
        assert telemetry["output"]["total_errors"] == 0

    def test_telemetry_skipped_shape(self):
        """When alpha_cohort is disabled, output diagnostics show 0 tickers checked."""
        row = {"ticker": "ACAD"}

        output_diag = validate_alpha_outputs(
            [row], schema_mode="warn", alpha_cohort_enabled=False,
        )

        d = output_diag.to_dict()
        assert d["n_tickers_checked"] == 0
        assert d["total_errors"] == 0
