"""Tests for scripts/backfill_archive_catalysts.py — catalyst backfill from trial milestones."""
from __future__ import annotations

import csv
import json
import os
import random
import shutil
import tarfile
import tempfile
from datetime import date
from pathlib import Path

import pytest

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from backfill_archive_catalysts import (
    ACTIVE_STATUSES,
    BackfillResult,
    backfill_archive,
    compute_backfill_catalyst,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic trial builder
# ---------------------------------------------------------------------------
def _make_trial(
    ticker: str = "ACME",
    phase: str = "PHASE2",
    status: str = "RECRUITING",
    first_posted: str = "2024-01-01",
    start_date: str = "2024-02-01",
    primary_completion_date: str | None = None,
    completion_date: str | None = None,
) -> dict:
    """Build a minimal trial record."""
    t = {
        "ticker": ticker,
        "phase": phase,
        "status": status,
        "first_posted": first_posted,
        "start_date": start_date,
    }
    if primary_completion_date is not None:
        t["primary_completion_date"] = primary_completion_date
    if completion_date is not None:
        t["completion_date"] = completion_date
    return t


def _make_archive(
    tmp_path: Path,
    date_str: str,
    dev_tickers: list[str],
    trials: list[dict],
    decision_inputs: dict | None = None,
) -> Path:
    """Build a minimal .tar.gz archive suitable for backfill_archive().

    Returns path to the created .tar.gz file.
    """
    archive_root = tmp_path / date_str
    inputs_dir = archive_root / "inputs"
    inputs_dir.mkdir(parents=True)

    # rankings.csv with drug_developer archetype
    csv_path = archive_root / "rankings.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ticker", "archetype", "tier_dev"])
        writer.writeheader()
        for t in dev_tickers:
            writer.writerow({"ticker": t, "archetype": "drug_developer", "tier_dev": "B"})

    # trial_records.json
    with open(inputs_dir / "trial_records.json", "w") as f:
        json.dump(trials, f)

    # decision_inputs.json
    di = decision_inputs or {}
    with open(inputs_dir / "decision_inputs.json", "w") as f:
        json.dump(di, f)

    # Pack into tar.gz
    tar_path = tmp_path / f"{date_str}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(str(archive_root), arcname=date_str)

    # Clean up extracted dir to avoid interference
    shutil.rmtree(archive_root)

    return tar_path


# ===========================================================================
# PIT Safety Tests
# ===========================================================================
class TestPITSafety:
    """Ensure point-in-time correctness."""

    def test_future_first_posted_excluded(self):
        """Trial with first_posted > as_of should never be used."""
        trials = [
            _make_trial(
                first_posted="2025-06-01",  # after as_of
                completion_date="2025-09-01",
            ),
        ]
        result = compute_backfill_catalyst("ACME", trials, date(2025, 1, 31))
        assert result is None

    def test_pcd_before_as_of_excluded(self):
        """Completion date in the past should not produce a signal."""
        trials = [
            _make_trial(
                first_posted="2024-01-01",
                completion_date="2024-06-01",  # before as_of
            ),
        ]
        result = compute_backfill_catalyst("ACME", trials, date(2025, 1, 31))
        # CD is in the past, so trial_cd won't fire; but trial_active WILL fire
        # since it's RECRUITING with start_date <= as_of
        assert result is not None
        assert result["catalyst_source"] == "trial_active"

    def test_pit_boundary_exact(self):
        """first_posted == as_of should be included (<=)."""
        trials = [
            _make_trial(
                first_posted="2025-01-31",  # exactly as_of
                completion_date="2025-06-15",
            ),
        ]
        result = compute_backfill_catalyst("ACME", trials, date(2025, 1, 31))
        assert result is not None
        assert result["catalyst_source"] == "trial_cd"


# ===========================================================================
# Determinism Tests
# ===========================================================================
class TestDeterminism:
    """Ensure reproducibility."""

    def test_same_input_same_output(self):
        """Running twice with identical inputs produces identical results."""
        trials = [
            _make_trial(ticker="ACME", completion_date="2025-06-15"),
            _make_trial(ticker="ACME", completion_date="2025-09-30", status="ACTIVE_NOT_RECRUITING"),
        ]
        r1 = compute_backfill_catalyst("ACME", trials, date(2025, 1, 31))
        r2 = compute_backfill_catalyst("ACME", trials, date(2025, 1, 31))
        assert r1 == r2

    def test_order_independent(self):
        """Shuffled trial list produces the same catalyst signal."""
        trials = [
            _make_trial(ticker="ACME", completion_date="2025-09-30", status="RECRUITING"),
            _make_trial(ticker="ACME", completion_date="2025-06-15", status="ACTIVE_NOT_RECRUITING"),
            _make_trial(ticker="ACME", phase="PHASE3", completion_date="2025-12-01"),
        ]
        r_original = compute_backfill_catalyst("ACME", trials, date(2025, 1, 31))

        shuffled = list(trials)
        rng = random.Random(99)
        rng.shuffle(shuffled)
        r_shuffled = compute_backfill_catalyst("ACME", shuffled, date(2025, 1, 31))
        assert r_original == r_shuffled


# ===========================================================================
# Source Priority Tests
# ===========================================================================
class TestSourcePriority:
    """Ensure correct source selection and priority chain."""

    def test_existing_catalyst_never_overwritten(self, tmp_path):
        """Ticker with existing catalyst signal is untouched by backfill."""
        existing_di = {
            "ACME": {
                "days_to_catalyst": 45,
                "in_optimal_window": True,
                "catalyst_source": "trial_pcd",
                "catalyst_event_type": "DATA_READOUT",
            },
        }
        trials = [
            _make_trial(ticker="ACME", completion_date="2025-03-15"),
        ]
        tar_path = _make_archive(
            tmp_path, "2025-01-31", ["ACME"], trials, decision_inputs=existing_di
        )
        result = backfill_archive(tar_path)
        assert result.n_existing == 1
        assert result.n_backfilled == 0

        # Verify the archive wasn't modified (existing signal preserved)
        with tarfile.open(tar_path, "r:gz") as tar:
            for m in tar.getmembers():
                if "decision_inputs" in m.name:
                    di = json.load(tar.extractfile(m))
                    assert di["ACME"]["catalyst_source"] == "trial_pcd"
                    assert di["ACME"]["days_to_catalyst"] == 45

    def test_trial_cd_preferred_over_trial_active(self):
        """When both CD fallback and active presence are available, CD wins."""
        trials = [
            # Trial 1: has CD (no PCD), active → qualifies for trial_cd
            _make_trial(
                ticker="ACME",
                completion_date="2025-06-15",
                status="RECRUITING",
            ),
            # Trial 2: active but no dates in window → would give trial_active
            _make_trial(
                ticker="ACME",
                status="RECRUITING",
                start_date="2024-06-01",
            ),
        ]
        result = compute_backfill_catalyst("ACME", trials, date(2025, 1, 31))
        assert result is not None
        assert result["catalyst_source"] == "trial_cd"

    def test_trial_cd_uses_completion_date_when_pcd_missing(self):
        """trial_cd uses completion_date when primary_completion_date is absent."""
        trials = [
            _make_trial(
                ticker="ACME",
                primary_completion_date=None,
                completion_date="2025-08-20",
            ),
        ]
        result = compute_backfill_catalyst("ACME", trials, date(2025, 1, 31))
        assert result is not None
        assert result["catalyst_source"] == "trial_cd"
        assert result["days_to_catalyst"] == (date(2025, 8, 20) - date(2025, 1, 31)).days

    def test_trial_active_requires_active_status(self):
        """COMPLETED trials should not produce trial_active signals."""
        trials = [
            _make_trial(
                ticker="ACME",
                status="COMPLETED",
                start_date="2023-01-01",
            ),
        ]
        result = compute_backfill_catalyst("ACME", trials, date(2025, 1, 31))
        assert result is None

    def test_trial_active_requires_start_before_as_of(self):
        """trial_active requires start_date <= as_of."""
        trials = [
            _make_trial(
                ticker="ACME",
                status="RECRUITING",
                start_date="2025-06-01",  # after as_of
                first_posted="2024-01-01",
            ),
        ]
        result = compute_backfill_catalyst("ACME", trials, date(2025, 1, 31))
        assert result is None

    def test_trial_active_synthetic_days(self):
        """trial_active emits days_to_catalyst = window_days + 1."""
        window = 365
        trials = [
            _make_trial(
                ticker="ACME",
                status="RECRUITING",
                start_date="2024-06-01",
                first_posted="2024-01-01",
                # no PCD, no CD → only trial_active
            ),
        ]
        result = compute_backfill_catalyst("ACME", trials, date(2025, 1, 31), window_days=window)
        assert result is not None
        assert result["catalyst_source"] == "trial_active"
        assert result["days_to_catalyst"] == window + 1
        assert result["in_optimal_window"] is False
        assert result["catalyst_event_type"] == "TRIAL_ONGOING"


# ===========================================================================
# Structural Tests
# ===========================================================================
class TestStructural:
    """Verify invariants and edge cases."""

    def test_phase_filter(self):
        """Only Phase 2/3 trials should produce signals."""
        trials = [
            _make_trial(ticker="ACME", phase="PHASE1", completion_date="2025-06-15"),
            _make_trial(ticker="ACME", phase="PHASE4", completion_date="2025-06-15"),
            _make_trial(ticker="ACME", phase="N/A", completion_date="2025-06-15"),
        ]
        result = compute_backfill_catalyst("ACME", trials, date(2025, 1, 31))
        assert result is None

    def test_backfill_result_counts(self, tmp_path):
        """n_existing + n_backfilled + n_still_missing == n_dev."""
        # 3 dev tickers: 1 has existing catalyst, 1 gets backfilled, 1 stays missing
        existing_di = {
            "EXIST": {
                "days_to_catalyst": 90,
                "in_optimal_window": False,
                "catalyst_source": "trial_pcd",
                "catalyst_event_type": "DATA_READOUT",
            },
        }
        trials = [
            # BKFILL gets a trial_cd signal
            _make_trial(ticker="BKFILL", completion_date="2025-06-15"),
        ]
        # MISS has no trials at all
        tar_path = _make_archive(
            tmp_path, "2025-01-31",
            ["EXIST", "BKFILL", "MISS"],
            trials,
            decision_inputs=existing_di,
        )
        result = backfill_archive(tar_path)
        assert result.n_dev == 3
        assert result.n_existing == 1
        assert result.n_backfilled == 1
        assert result.n_still_missing == 1
        assert result.n_existing + result.n_backfilled + result.n_still_missing == result.n_dev

    def test_dry_run_no_modification(self, tmp_path):
        """dry_run mode should not modify the archive."""
        trials = [
            _make_trial(ticker="ACME", completion_date="2025-06-15"),
        ]
        tar_path = _make_archive(tmp_path, "2025-01-31", ["ACME"], trials)

        # Record original file content
        original_data = tar_path.read_bytes()

        result = backfill_archive(tar_path, dry_run=True)
        assert result.status == "dry_run"
        assert result.n_backfilled == 1  # would have backfilled

        # File should be unchanged
        assert tar_path.read_bytes() == original_data


# ===========================================================================
# Integration Test
# ===========================================================================
class TestIntegration:
    """End-to-end: backfilled catalyst consumed by decision engine."""

    def test_backfill_round_trip_with_decision_engine(self, tmp_path):
        """Backfilled catalyst_decay dict is correctly consumed by compute_decision_fields."""
        from decision_engine import compute_decision_fields

        # Simulate a backfilled signal
        signal = compute_backfill_catalyst(
            "ACME",
            [_make_trial(ticker="ACME", completion_date="2025-04-15")],
            date(2025, 1, 31),
        )
        assert signal is not None

        # Build a minimal rec dict that decision engine expects
        rec = {
            "ticker": "ACME",
            "composite_score": 55.0,
            "score_rank_pct": 0.75,
            "catalyst_decay": signal,
            "smart_money_signal": {},
            "coinvest": {},
            "defensive_features": {},
            "score_breakdown": {"enhancements": {}},
        }

        result = compute_decision_fields(
            rec, archetype="drug_developer", optionality_pct_dev=0.65,
        )

        # The decision engine should see a real catalyst signal
        assert result["catalyst_mode"] == "specific_days"
        assert result["catalyst_days"] == signal["days_to_catalyst"]
        # trial_cd should produce near/mid/far strength (not missing)
        assert result["catalyst_strength"] != "missing"


# ===========================================================================
# trial_cd: PCD present should be skipped (no PCD override)
# ===========================================================================
class TestTrialCdSkipsPcdPresent:
    """trial_cd only fires when PCD is missing but CD exists."""

    def test_trial_with_pcd_not_used_for_cd_fallback(self):
        """If PCD is present, trial_cd never fires (even if CD is closer)."""
        trials = [
            _make_trial(
                ticker="ACME",
                primary_completion_date="2025-12-01",  # PCD present
                completion_date="2025-03-01",           # CD is closer
                status="RECRUITING",
            ),
        ]
        # Since PCD is present, trial_cd skips this trial. trial_active fires.
        result = compute_backfill_catalyst("ACME", trials, date(2025, 1, 31))
        assert result is not None
        # Should get trial_active (not trial_cd), because PCD is present
        assert result["catalyst_source"] == "trial_active"
