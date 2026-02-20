"""Tests for ctgov_pit_dates gate in run_daily_production.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from run_daily_production import check_ctgov_pit_dates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_trial_records(
    path: Path,
    records: list[dict],
) -> None:
    """Write trial records JSON to the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f)


def _make_records(
    n: int,
    *,
    first_posted_frac: float = 1.0,
    last_update_frac: float = 1.0,
    results_posted_frac: float = 0.0,
) -> list[dict]:
    """Generate n trial records with specified coverage fractions."""
    records = []
    for i in range(n):
        rec = {"nct_id": f"NCT{i:08d}"}
        if i < int(n * first_posted_frac):
            rec["first_posted"] = "2025-01-15"
        else:
            rec["first_posted"] = ""
        if i < int(n * last_update_frac):
            rec["last_update_posted"] = "2025-06-10"
        else:
            rec["last_update_posted"] = ""
        if i < int(n * results_posted_frac):
            rec["results_first_posted"] = "2025-09-01"
        else:
            rec["results_first_posted"] = ""
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCtgovPitDatesGate:
    def test_full_coverage_pass(self, tmp_path):
        """100% first_posted + last_update → PASS."""
        records = _make_records(100, first_posted_frac=1.0, last_update_frac=1.0)
        _write_trial_records(tmp_path / "trial_records_2026-02-19.json", records)

        gate = check_ctgov_pit_dates(tmp_path, "2026-02-19")
        assert gate.status == "PASS"
        assert gate.value["first_posted"] == 1.0
        assert gate.value["last_update_posted"] == 1.0

    def test_low_first_posted_warn(self, tmp_path):
        """70% first_posted → WARN."""
        records = _make_records(100, first_posted_frac=0.70, last_update_frac=1.0)
        _write_trial_records(tmp_path / "trial_records_2026-02-19.json", records)

        gate = check_ctgov_pit_dates(tmp_path, "2026-02-19")
        assert gate.status == "WARN"
        assert "first_posted" in gate.detail

    def test_low_last_update_warn(self, tmp_path):
        """75% last_update_posted → WARN."""
        records = _make_records(100, first_posted_frac=1.0, last_update_frac=0.75)
        _write_trial_records(tmp_path / "trial_records_2026-02-19.json", records)

        gate = check_ctgov_pit_dates(tmp_path, "2026-02-19")
        assert gate.status == "WARN"
        assert "last_update" in gate.detail

    def test_results_posted_no_warn(self, tmp_path):
        """Low results_first_posted → still PASS (informational only)."""
        records = _make_records(
            100,
            first_posted_frac=1.0,
            last_update_frac=1.0,
            results_posted_frac=0.10,
        )
        _write_trial_records(tmp_path / "trial_records_2026-02-19.json", records)

        gate = check_ctgov_pit_dates(tmp_path, "2026-02-19")
        assert gate.status == "PASS"
        # results coverage is still reported
        assert gate.value["results_first_posted"] == 0.1

    def test_missing_cache_file(self, tmp_path):
        """Cache file not found → WARN with detail."""
        gate = check_ctgov_pit_dates(tmp_path, "2026-02-19")
        assert gate.status == "WARN"
        assert "No trial_records cache found" in gate.detail

    def test_empty_records(self, tmp_path):
        """Empty trial list → WARN."""
        _write_trial_records(tmp_path / "trial_records_2026-02-19.json", [])

        gate = check_ctgov_pit_dates(tmp_path, "2026-02-19")
        assert gate.status == "WARN"
        assert "Empty" in gate.detail or "invalid" in gate.detail.lower()

    def test_coverage_values_in_gate_result(self, tmp_path):
        """GateResult.value contains coverage dict with expected keys."""
        records = _make_records(50, first_posted_frac=0.90, last_update_frac=0.95, results_posted_frac=0.30)
        _write_trial_records(tmp_path / "trial_records_2026-02-19.json", records)

        gate = check_ctgov_pit_dates(tmp_path, "2026-02-19")
        assert gate.status == "PASS"
        assert "first_posted" in gate.value
        assert "last_update_posted" in gate.value
        assert "results_first_posted" in gate.value
        assert "total_records" in gate.value
        assert gate.value["total_records"] == 50

    def test_never_fails(self, tmp_path):
        """Even 0% coverage → WARN, never FAIL."""
        records = _make_records(100, first_posted_frac=0.0, last_update_frac=0.0)
        _write_trial_records(tmp_path / "trial_records_2026-02-19.json", records)

        gate = check_ctgov_pit_dates(tmp_path, "2026-02-19")
        assert gate.status in ("PASS", "WARN")
        assert gate.status != "FAIL"

    def test_fallback_to_prior_date(self, tmp_path):
        """When exact date missing, falls back to latest prior date."""
        records = _make_records(100, first_posted_frac=1.0, last_update_frac=1.0)
        _write_trial_records(tmp_path / "trial_records_2026-02-18.json", records)

        gate = check_ctgov_pit_dates(tmp_path, "2026-02-19")
        assert gate.status == "PASS"
        assert "2026-02-18" in gate.value["cache_file"]

    def test_custom_thresholds(self, tmp_path):
        """Custom thresholds are respected."""
        records = _make_records(100, first_posted_frac=0.95, last_update_frac=0.95)
        _write_trial_records(tmp_path / "trial_records_2026-02-19.json", records)

        # Default thresholds (0.80) → PASS
        gate = check_ctgov_pit_dates(tmp_path, "2026-02-19")
        assert gate.status == "PASS"

        # Stricter thresholds → WARN
        gate = check_ctgov_pit_dates(
            tmp_path, "2026-02-19",
            warn_first_posted_min=0.99,
        )
        assert gate.status == "WARN"
