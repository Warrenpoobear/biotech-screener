"""Tests for clinical_alpha_z, clinical_readout_days, and clinical_coverage_flag."""
from __future__ import annotations

import math

import pytest

from run_screen import (
    _clinical_proximity,
    _compute_clinical_readout_days,
    _compute_clinical_alpha_z,
    _PHASE_NUM,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trial(
    ticker: str = "ACME",
    phase: str = "PHASE2",
    study_type: str = "INTERVENTIONAL",
    first_posted: str = "2024-01-01",
    primary_completion_date: str | None = None,
    completion_date: str | None = None,
    results_first_posted: str | None = None,
    last_update_posted: str | None = None,
    status: str = "RECRUITING",
) -> dict:
    """Build a minimal trial record."""
    t: dict = {
        "ticker": ticker,
        "phase": phase,
        "study_type": study_type,
        "first_posted": first_posted,
        "status": status,
    }
    if primary_completion_date is not None:
        t["primary_completion_date"] = primary_completion_date
    if completion_date is not None:
        t["completion_date"] = completion_date
    if results_first_posted is not None:
        t["results_first_posted"] = results_first_posted
    if last_update_posted is not None:
        t["last_update_posted"] = last_update_posted
    return t


def _make_dev_row(ticker: str, clinical_score=70.0, archetype="drug_developer"):
    """Build a minimal csv_row dict for a dev ticker."""
    return {
        "ticker": ticker,
        "archetype": archetype,
        "clinical_score": clinical_score,
    }


def _make_m4(ticker: str, lead_phase: str = "phase 2", clinical_score=70.0):
    """Build a minimal module_4 score entry."""
    return {
        "ticker": ticker,
        "lead_phase": lead_phase,
        "clinical_score": clinical_score,
    }


# ===========================================================================
# _clinical_proximity tests
# ===========================================================================
class TestClinicalProximity:
    def test_zero_days(self):
        assert _clinical_proximity(0) == 1.0

    def test_negative_days(self):
        assert _clinical_proximity(-5) == 1.0

    def test_none_days(self):
        assert _clinical_proximity(None) == 0.0

    def test_90_days(self):
        expected = math.exp(-1)  # e^(-90/90) ≈ 0.3679
        assert abs(_clinical_proximity(90) - expected) < 1e-6

    def test_30_days(self):
        expected = math.exp(-30 / 90)  # ≈ 0.7165
        assert abs(_clinical_proximity(30) - expected) < 1e-6

    def test_180_days(self):
        expected = math.exp(-180 / 90)  # ≈ 0.1353
        assert abs(_clinical_proximity(180) - expected) < 1e-6

    def test_monotonically_decreasing(self):
        vals = [_clinical_proximity(d) for d in [0, 30, 60, 90, 180, 365]]
        for a, b in zip(vals, vals[1:]):
            assert a > b


# ===========================================================================
# _compute_clinical_readout_days tests
# ===========================================================================
class TestComputeClinicalReadoutDays:
    def test_future_pcd(self):
        """Future PCD → correct positive days."""
        trials = [_make_trial(primary_completion_date="2026-06-15")]
        result = _compute_clinical_readout_days(trials, "2026-01-15")
        assert result["ACME"] == (
            __import__("datetime").date(2026, 6, 15)
            - __import__("datetime").date(2026, 1, 15)
        ).days
        assert result["ACME"] == 151

    def test_past_pcd_no_results_imminent(self):
        """Past PCD + no results_first_posted → 0 (imminent)."""
        trials = [_make_trial(primary_completion_date="2025-12-01")]
        result = _compute_clinical_readout_days(trials, "2026-01-15")
        assert result["ACME"] == 0

    def test_past_pcd_with_results_skipped(self):
        """Past PCD + results posted → skip (already known)."""
        trials = [_make_trial(
            primary_completion_date="2025-10-01",
            results_first_posted="2025-12-01",
        )]
        result = _compute_clinical_readout_days(trials, "2026-01-15")
        assert "ACME" not in result

    def test_pit_filter_excludes_future_trial(self):
        """Trial posted after as_of_date should be excluded."""
        trials = [_make_trial(
            first_posted="2026-06-01",
            primary_completion_date="2026-09-01",
        )]
        result = _compute_clinical_readout_days(trials, "2026-01-15")
        assert "ACME" not in result

    def test_pit_last_update_posted_fallback(self):
        """If first_posted missing, last_update_posted used for PIT gate."""
        trials = [_make_trial(
            first_posted=None,
            last_update_posted="2025-06-01",
            primary_completion_date="2026-06-15",
        )]
        # Remove first_posted entirely
        del trials[0]["first_posted"]
        result = _compute_clinical_readout_days(trials, "2026-01-15")
        assert "ACME" in result

    def test_no_pcd_future_cd_fallback(self):
        """No PCD but future completion_date → use that."""
        trials = [_make_trial(completion_date="2026-08-01")]
        result = _compute_clinical_readout_days(trials, "2026-01-15")
        assert result["ACME"] == (
            __import__("datetime").date(2026, 8, 1)
            - __import__("datetime").date(2026, 1, 15)
        ).days

    def test_no_qualifying_trials(self):
        """Ticker with no qualifying trials → not in result."""
        trials = [_make_trial(study_type="OBSERVATIONAL",
                              primary_completion_date="2026-06-01")]
        result = _compute_clinical_readout_days(trials, "2026-01-15")
        assert "ACME" not in result

    def test_phase4_excluded(self):
        """Phase 4 trials should be excluded."""
        trials = [_make_trial(phase="PHASE4",
                              primary_completion_date="2026-06-01")]
        result = _compute_clinical_readout_days(trials, "2026-01-15")
        assert "ACME" not in result

    def test_min_across_trials(self):
        """Takes minimum days across qualifying trials."""
        trials = [
            _make_trial(primary_completion_date="2026-09-01"),
            _make_trial(primary_completion_date="2026-03-01"),
        ]
        result = _compute_clinical_readout_days(trials, "2026-01-15")
        # March 1 is closer → 45 days
        assert result["ACME"] == 45

    def test_empty_trial_records(self):
        """Empty list → empty dict, no crash."""
        result = _compute_clinical_readout_days([], "2026-01-15")
        assert result == {}


# ===========================================================================
# _compute_clinical_alpha_z tests
# ===========================================================================
class TestComputeClinicalAlphaZ:
    def test_z_scores_mean_near_zero(self):
        """Z-scores within dev cohort should have mean ≈ 0."""
        rows = [_make_dev_row(f"T{i}", clinical_score=20 + i * 3) for i in range(30)]
        m4 = {f"T{i}": _make_m4(f"T{i}", clinical_score=20 + i * 3)
               for i in range(30)}
        readout = {f"T{i}": 30 + i * 10 for i in range(30)}

        _compute_clinical_alpha_z(rows, m4, readout)

        zs = [r["clinical_alpha_z"] for r in rows if r.get("clinical_alpha_z") is not None]
        assert len(zs) == 30
        assert abs(sum(zs) / len(zs)) < 0.1  # mean ≈ 0

    def test_winsorization_clips_outliers(self):
        """Extreme outliers should be clipped by winsorization."""
        # Create 25 rows with varying scores (spread) + 1 extreme outlier
        rows = [_make_dev_row(f"T{i}", clinical_score=30 + i * 2) for i in range(25)]
        rows.append(_make_dev_row("OUTLIER", clinical_score=100))

        m4 = {r["ticker"]: _make_m4(r["ticker"], lead_phase="phase 2",
                                     clinical_score=r["clinical_score"])
               for r in rows}
        # Vary readout days for spread; outlier gets extreme proximity
        readout = {f"T{i}": 60 + i * 10 for i in range(25)}
        readout["OUTLIER"] = 0

        _compute_clinical_alpha_z(rows, m4, readout)

        zs = [r["clinical_alpha_z"] for r in rows]
        # Outlier z should still be highest (even after clipping)
        outlier_z = rows[-1]["clinical_alpha_z"]
        normal_zs = [r["clinical_alpha_z"] for r in rows[:-1]]
        assert outlier_z >= max(normal_zs)

    def test_non_dev_tickers_blank(self):
        """Non drug_developer tickers should get blank values."""
        rows = [
            _make_dev_row("DEV1"),
            _make_dev_row("DEV2"),
            {"ticker": "COMM1", "archetype": "commercial_biotech",
             "clinical_score": 80.0},
        ]
        m4 = {"DEV1": _make_m4("DEV1"), "DEV2": _make_m4("DEV2"),
               "COMM1": _make_m4("COMM1")}
        readout = {"DEV1": 30, "DEV2": 60}

        _compute_clinical_alpha_z(rows, m4, readout)

        assert "clinical_alpha_z" in rows[0]
        assert "clinical_alpha_z" not in rows[2]

    def test_coverage_flag_zero_no_readout(self):
        """coverage_flag = 0 when no readout data for a dev ticker."""
        rows = [_make_dev_row("T1"), _make_dev_row("T2")]
        m4 = {"T1": _make_m4("T1"), "T2": _make_m4("T2")}
        readout = {"T1": 30}  # T2 missing

        _compute_clinical_alpha_z(rows, m4, readout)

        assert rows[0]["clinical_coverage_flag"] == 1
        assert rows[1]["clinical_coverage_flag"] == 0
        assert rows[0]["clinical_readout_days"] == 30
        assert rows[1]["clinical_readout_days"] == ""

    def test_all_same_raw_score_z_zero(self):
        """When all raw scores identical → std=0 → z=0.0 for all."""
        rows = [_make_dev_row(f"T{i}", clinical_score=50) for i in range(10)]
        m4 = {f"T{i}": _make_m4(f"T{i}", lead_phase="phase 2", clinical_score=50)
               for i in range(10)}
        readout = {f"T{i}": 45 for i in range(10)}

        _compute_clinical_alpha_z(rows, m4, readout)

        for r in rows:
            assert r["clinical_alpha_z"] == 0.0

    def test_empty_trial_records_graceful(self):
        """No m4_lookup entries → no crash, no z written."""
        rows = [_make_dev_row("T1")]
        _compute_clinical_alpha_z(rows, {}, {})
        assert "clinical_alpha_z" not in rows[0]

    def test_readout_days_written_as_int(self):
        """clinical_readout_days should be integer, not float."""
        rows = [_make_dev_row("T1"), _make_dev_row("T2")]
        m4 = {"T1": _make_m4("T1"), "T2": _make_m4("T2")}
        readout = {"T1": 45, "T2": 0}

        _compute_clinical_alpha_z(rows, m4, readout)

        assert rows[0]["clinical_readout_days"] == 45
        assert isinstance(rows[0]["clinical_readout_days"], int)
        assert rows[1]["clinical_readout_days"] == 0

    def test_phase_num_mapping_coverage(self):
        """All expected lead_phase values should map to a non-zero value."""
        for phase in ["phase 1", "phase 1/2", "phase 2", "phase 2/3",
                       "phase 3", "approved", "preclinical"]:
            assert phase in _PHASE_NUM
            assert _PHASE_NUM[phase] > 0
