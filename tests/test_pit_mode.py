"""Strict-PIT regression test: ensures future-dated trial_records are rejected."""

import json
from datetime import date
from pathlib import Path

import pytest

from catalyst_diagnostics import PITViolationError
from module_3_catalyst import compute_module_3_catalyst


def test_strict_pit_rejects_future_trial_records(tmp_path: Path):
    """pit_mode='strict' must raise PITViolationError when
    trial_records contain data dated AFTER as_of_date."""

    trial_records = [
        {
            "nct_id": "NCT00000001",
            "ticker": "ACME",
            "overall_status": "Recruiting",
            "phase": "Phase 2",
            "conditions": ["Test Condition"],
            "last_update_posted": "2026-01-30",
        }
    ]
    trial_path = tmp_path / "trial_records.json"
    trial_path.write_text(json.dumps(trial_records))

    state_dir = tmp_path / "ctgov_state"
    state_dir.mkdir()

    with pytest.raises(PITViolationError):
        compute_module_3_catalyst(
            trial_records_path=trial_path,
            state_dir=state_dir,
            active_tickers={"ACME"},
            as_of_date=date(2026, 1, 28),
            pit_mode="strict",
        )


def test_degrade_pit_does_not_raise_on_future_trial_records(tmp_path: Path):
    """pit_mode='degrade' must NOT raise on future-dated trial_records."""

    trial_records = [
        {
            "nct_id": "NCT00000001",
            "ticker": "ACME",
            "overall_status": "Recruiting",
            "phase": "Phase 2",
            "conditions": ["Test Condition"],
            "last_update_posted": "2026-01-30",
        }
    ]
    trial_path = tmp_path / "trial_records.json"
    trial_path.write_text(json.dumps(trial_records))

    state_dir = tmp_path / "ctgov_state"
    state_dir.mkdir()

    # Should not raise — degrade mode continues with a warning
    result = compute_module_3_catalyst(
        trial_records_path=trial_path,
        state_dir=state_dir,
        active_tickers={"ACME"},
        as_of_date=date(2026, 1, 28),
        pit_mode="degrade",
    )
    assert "summaries" in result
