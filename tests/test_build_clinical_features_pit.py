"""Tests for scripts/build_clinical_features_pit.py — PIT-safe clinical features."""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from datetime import date
from pathlib import Path

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from scripts.build_clinical_features_pit import (
    DEFAULT_QUALITY,
    SCHEMA_VERSION,
    _CE_WEIGHTS,
    _CLINICAL_PHASES,
    _PHASE_NUM,
    _clinical_proximity,
    _collected_at_range,
    _compute_far_horizon_days,
    _compute_readout_for_ticker,
    _has_active_interventional,
    _highest_phase,
    _is_pit_admitted,
    _load_universe,
    _parse_trial_date,
    _phase_label_to_num,
    build_clinical_features,
    validate_clinical_features_schema,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_trial(
    ticker="ACAD",
    study_type="INTERVENTIONAL",
    phase="PHASE2",
    first_posted="2025-01-15",
    last_update_posted="2025-06-01",
    primary_completion_date="2026-06-15",
    completion_date=None,
    results_first_posted=None,
    status="RECRUITING",
    nct_id="NCT12345678",
    collected_at=None,
):
    t = {
        "ticker": ticker,
        "study_type": study_type,
        "phase": phase,
        "first_posted": first_posted,
        "last_update_posted": last_update_posted,
        "primary_completion_date": primary_completion_date,
        "status": status,
        "nct_id": nct_id,
    }
    if completion_date is not None:
        t["completion_date"] = completion_date
    if results_first_posted is not None:
        t["results_first_posted"] = results_first_posted
    if collected_at is not None:
        t["collected_at"] = collected_at
    return t


AS_OF = "2026-01-15"
AS_OF_DATE = date(2026, 1, 15)


# ===================================================================
# TestPITFiltering
# ===================================================================

class TestPITFiltering:
    """PIT admission gate tests."""

    def test_first_posted_after_as_of_filtered(self):
        """Trial with first_posted after as_of_date → filtered out."""
        trial = _make_trial(first_posted="2026-02-01")
        admitted, field = _is_pit_admitted(trial, AS_OF_DATE)
        assert not admitted
        assert field == "first_posted"

    def test_first_posted_on_as_of_admitted(self):
        """Trial with first_posted on as_of_date → admitted (<=)."""
        trial = _make_trial(first_posted="2026-01-15")
        admitted, field = _is_pit_admitted(trial, AS_OF_DATE)
        assert admitted
        assert field == "first_posted"

    def test_first_posted_before_as_of_admitted(self):
        """Trial with first_posted before as_of_date → admitted."""
        trial = _make_trial(first_posted="2025-06-01")
        admitted, field = _is_pit_admitted(trial, AS_OF_DATE)
        assert admitted
        assert field == "first_posted"

    def test_last_update_posted_fallback(self):
        """Trial with only last_update_posted → falls back correctly."""
        trial = _make_trial(first_posted=None, last_update_posted="2025-12-01")
        admitted, field = _is_pit_admitted(trial, AS_OF_DATE)
        assert admitted
        assert field == "last_update_posted"

    def test_no_date_fields_filtered(self):
        """Trial with neither date field → filtered (None→excluded)."""
        trial = _make_trial(first_posted=None, last_update_posted=None)
        admitted, field = _is_pit_admitted(trial, AS_OF_DATE)
        assert not admitted
        assert field == ""

    def test_non_interventional_excluded_from_readout(self):
        """Non-INTERVENTIONAL study type → admitted but no readout days."""
        trials = [_make_trial(study_type="OBSERVATIONAL")]
        rd, nct_id, n_adm, n_filt, pit_field = _compute_readout_for_ticker(
            trials, AS_OF_DATE,
        )
        assert rd is None
        assert n_adm == 1  # admitted but no readout contribution
        assert n_filt == 0

    def test_terminal_status_included_in_trial_count(self):
        """Terminal status (WITHDRAWN) → admitted (counts) but excluded from far horizon."""
        trials = [_make_trial(status="WITHDRAWN")]
        # Terminal trials are still admitted/counted for readout_days
        # (only far_horizon excludes them)
        rd, nct_id, n_adm, n_filt, pit_field = _compute_readout_for_ticker(
            trials, AS_OF_DATE,
        )
        assert n_adm == 1
        assert n_filt == 0


# ===================================================================
# TestReadoutDays
# ===================================================================

class TestReadoutDays:
    """Readout days computation tests."""

    def test_future_pcd_correct_days(self):
        """Future PCD → correct days delta."""
        # PCD = 2026-06-15, as_of = 2026-01-15 → 151 days
        trials = [_make_trial(primary_completion_date="2026-06-15")]
        rd, nct_id, *_ = _compute_readout_for_ticker(trials, AS_OF_DATE)
        assert rd == 151

    def test_past_pcd_no_results_imminent(self):
        """Past PCD with no results_first_posted → 0 (imminent)."""
        trials = [_make_trial(
            primary_completion_date="2025-12-01",
            results_first_posted=None,
        )]
        rd, *_ = _compute_readout_for_ticker(trials, AS_OF_DATE)
        assert rd == 0

    def test_past_pcd_with_results_skipped(self):
        """Past PCD with results_first_posted → skipped."""
        trials = [_make_trial(
            primary_completion_date="2025-12-01",
            results_first_posted="2025-12-15",
        )]
        rd, *_ = _compute_readout_for_ticker(trials, AS_OF_DATE)
        assert rd is None

    def test_completion_date_fallback(self):
        """completion_date fallback when no PCD."""
        trials = [_make_trial(
            primary_completion_date=None,
            completion_date="2026-04-01",
        )]
        rd, *_ = _compute_readout_for_ticker(trials, AS_OF_DATE)
        expected = (date(2026, 4, 1) - AS_OF_DATE).days
        assert rd == expected

    def test_multiple_trials_min_selected(self):
        """Multiple trials → min days selected."""
        trials = [
            _make_trial(primary_completion_date="2026-06-15", nct_id="NCT1"),
            _make_trial(primary_completion_date="2026-03-01", nct_id="NCT2"),
        ]
        rd, nct_id, *_ = _compute_readout_for_ticker(trials, AS_OF_DATE)
        expected = (date(2026, 3, 1) - AS_OF_DATE).days
        assert rd == expected
        assert nct_id == "NCT2"

    def test_no_qualifying_trials_none(self):
        """No qualifying trials → None."""
        trials = [_make_trial(study_type="OBSERVATIONAL")]
        rd, *_ = _compute_readout_for_ticker(trials, AS_OF_DATE)
        assert rd is None


# ===================================================================
# TestClinicalAlphaRaw
# ===================================================================

class TestClinicalAlphaRaw:
    """Clinical alpha raw score computation tests."""

    def test_correct_weighted_formula(self):
        """Correct weighted formula: phase*0.35 + proximity*0.35 + quality*0.30."""
        phase_num = 0.83
        proximity = 0.5
        quality = DEFAULT_QUALITY
        expected = 0.35 * phase_num + 0.35 * proximity + 0.30 * quality
        # Build via actual function
        trials = [_make_trial(
            phase="PHASE3",
            primary_completion_date="2026-04-01",  # ~76 days → proximity
        )]
        features = build_clinical_features(
            AS_OF, trials, {"ACAD"}, quality_value=quality,
        )
        ticker_data = features["tickers"]["ACAD"]
        # Verify proximity ~ exp(-76/90)
        days = (date(2026, 4, 1) - AS_OF_DATE).days
        exp_prox = math.exp(-days / 90)
        exp_raw = 0.35 * 0.83 + 0.35 * exp_prox + 0.30 * quality
        assert abs(ticker_data["clinical_alpha_raw"] - exp_raw) < 0.001

    def test_phase3_phase_num(self):
        """Phase 3 trial → phase_num=0.83."""
        trials = [_make_trial(phase="PHASE3")]
        features = build_clinical_features(AS_OF, trials, {"ACAD"})
        assert features["tickers"]["ACAD"]["phase_num"] == 0.83

    def test_proximity_exp_decay_90d(self):
        """Proximity exp decay: 90 days → exp(-1) ≈ 0.368."""
        prox = _clinical_proximity(90)
        assert abs(prox - math.exp(-1)) < 0.001

    def test_zero_readout_days_proximity_one(self):
        """Zero readout_days → proximity=1.0."""
        prox = _clinical_proximity(0)
        assert prox == 1.0

    def test_none_readout_days_proximity_zero(self):
        """None readout_days → proximity=0.0."""
        prox = _clinical_proximity(None)
        assert prox == 0.0


# ===================================================================
# TestFarHorizonPIT
# ===================================================================

class TestFarHorizonPIT:
    """Far horizon PIT safety tests."""

    def test_respects_pit_gate(self):
        """Far horizon respects PIT gate (trial first_posted after as_of → excluded)."""
        trials = [_make_trial(
            first_posted="2026-02-01",  # after as_of
            primary_completion_date="2026-06-15",
        )]
        result = _compute_far_horizon_days(trials, AS_OF_DATE)
        assert result is None

    def test_within_far_window(self):
        """Trial within far_window_days → recorded."""
        trials = [_make_trial(
            first_posted="2025-01-01",
            primary_completion_date="2027-01-01",  # ~351 days
        )]
        result = _compute_far_horizon_days(trials, AS_OF_DATE, far_max=730)
        expected = (date(2027, 1, 1) - AS_OF_DATE).days
        assert result == expected

    def test_beyond_far_window_excluded(self):
        """Trial beyond far_window_days → excluded."""
        trials = [_make_trial(
            first_posted="2025-01-01",
            primary_completion_date="2030-01-01",  # way beyond 730d
        )]
        result = _compute_far_horizon_days(trials, AS_OF_DATE, far_max=730)
        assert result is None

    def test_terminal_status_excluded(self):
        """Only INTERVENTIONAL, non-terminal trials qualify."""
        trials = [_make_trial(
            first_posted="2025-01-01",
            status="TERMINATED",
            primary_completion_date="2026-06-15",
        )]
        result = _compute_far_horizon_days(trials, AS_OF_DATE)
        assert result is None


# ===================================================================
# TestHighestPhase
# ===================================================================

class TestHighestPhase:
    """Phase ranking tests."""

    def test_picks_highest(self):
        """Picks highest phase across multiple trials."""
        trials = [
            _make_trial(phase="PHASE1"),
            _make_trial(phase="PHASE3"),
            _make_trial(phase="PHASE2"),
        ]
        num, label = _highest_phase(trials, AS_OF_DATE)
        assert num == 0.83
        assert label == "phase 3"

    def test_combination_phase(self):
        """Phase 2/3 combo recognized."""
        trials = [_make_trial(phase="PHASE2/PHASE3")]
        num, label = _highest_phase(trials, AS_OF_DATE)
        assert num == 0.67
        assert label == "phase 2/3"


# ===================================================================
# TestSchemaValidation
# ===================================================================

class TestSchemaValidation:
    """Schema validation tests."""

    def test_full_output_passes(self):
        """Full output passes schema validation."""
        trials = [
            _make_trial(ticker="ACAD"),
            _make_trial(ticker="AARD", nct_id="NCT999"),
        ]
        features = build_clinical_features(AS_OF, trials, {"ACAD", "AARD"})
        ok, reason = validate_clinical_features_schema(features)
        assert ok, reason

    def test_coverage_consistency(self):
        """signal_coverage_pct matches counts."""
        trials = [_make_trial(ticker="ACAD")]
        features = build_clinical_features(AS_OF, trials, {"ACAD", "AARD"})
        n_uni = features["tickers_in_universe"]
        n_sig = features["tickers_with_signal"]
        expected = round(n_sig / n_uni * 100, 1)
        assert abs(features["signal_coverage_pct"] - expected) < 0.2

    def test_provenance_complete(self):
        """Provenance block complete."""
        features = build_clinical_features(AS_OF, [], {"ACAD"})
        prov = features["provenance"]
        assert prov["builder"] == "build_clinical_features_pit.py"
        assert prov["builder_version"] == "1.0.0"
        assert prov["quality_mode"] == "neutral"
        assert prov["quality_value"] == DEFAULT_QUALITY
        assert prov["pit_date_priority"] == ["first_posted", "last_update_posted"]

    def test_missing_field_detected(self):
        """Missing required field → validation fails."""
        features = build_clinical_features(AS_OF, [], {"ACAD"})
        del features["schema_version"]
        ok, reason = validate_clinical_features_schema(features)
        assert not ok
        assert "schema_version" in reason


# ===================================================================
# TestEdgeCases
# ===================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_empty_universe(self):
        """Empty universe → empty tickers dict."""
        features = build_clinical_features(AS_OF, [_make_trial()], set())
        assert features["tickers"] == {}
        assert features["tickers_in_universe"] == 0
        assert features["tickers_with_signal"] == 0

    def test_empty_trial_records(self):
        """Empty trial_records → no tickers with signal."""
        features = build_clinical_features(AS_OF, [], {"ACAD", "AARD"})
        assert features["tickers_with_signal"] == 0
        assert features["tickers"] == {}

    def test_partial_date_month_only(self):
        """Trial with partial date ('2024-06') → parsed as '2024-06-01'."""
        d = _parse_trial_date("2024-06")
        assert d == date(2024, 6, 1)

    def test_multiple_tickers_correct_aggregation(self):
        """Multiple tickers from same trial set → correct per-ticker aggregation."""
        trials = [
            _make_trial(ticker="ACAD", phase="PHASE3",
                        primary_completion_date="2026-06-15"),
            _make_trial(ticker="AARD", phase="PHASE1",
                        primary_completion_date="2026-03-01"),
        ]
        features = build_clinical_features(AS_OF, trials, {"ACAD", "AARD"})
        assert features["tickers"]["ACAD"]["phase_num"] == 0.83
        assert features["tickers"]["AARD"]["phase_num"] == 0.17

    def test_study_type_none_handled(self):
        """study_type None → (t.get('study_type') or '') pattern works."""
        trial = _make_trial(study_type=None)
        trials = [trial]
        # Should not crash
        rd, *_ = _compute_readout_for_ticker(trials, AS_OF_DATE)
        assert rd is None

    def test_has_active_interventional_true(self):
        """Active interventional trial detected."""
        trials = [_make_trial(status="RECRUITING")]
        assert _has_active_interventional(trials, AS_OF_DATE) is True

    def test_has_active_interventional_false_terminal(self):
        """All terminal → no active interventional."""
        trials = [_make_trial(status="WITHDRAWN")]
        assert _has_active_interventional(trials, AS_OF_DATE) is False

    def test_collected_at_range(self):
        """Collected-at range computed from trial records."""
        trials = [
            _make_trial(collected_at="2026-02-01"),
            _make_trial(collected_at="2026-02-18"),
            _make_trial(collected_at="2026-02-10"),
        ]
        mn, mx = _collected_at_range(trials)
        assert mn == "2026-02-01"
        assert mx == "2026-02-18"


# ===================================================================
# TestCLI
# ===================================================================

class TestCLI:
    """CLI integration tests."""

    def test_main_roundtrip(self, tmp_path):
        """CLI produces valid output file."""
        from scripts.build_clinical_features_pit import main

        # Write trial records
        trials = [
            _make_trial(ticker="ACAD"),
            _make_trial(ticker="AARD", nct_id="NCT999"),
        ]
        trial_path = tmp_path / "trials.json"
        trial_path.write_text(json.dumps(trials))

        # Write universe
        uni_path = tmp_path / "universe.json"
        uni_path.write_text(json.dumps(["ACAD", "AARD", "ALNY"]))

        out_path = tmp_path / "output.json"
        rc = main([
            "--as-of-date", AS_OF,
            "--trial-records", str(trial_path),
            "--universe", str(uni_path),
            "--out", str(out_path),
        ])
        assert rc == 0
        data = json.loads(out_path.read_text())
        ok, reason = validate_clinical_features_schema(data)
        assert ok, reason
        assert data["tickers_in_universe"] == 3
        assert data["tickers_with_signal"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
