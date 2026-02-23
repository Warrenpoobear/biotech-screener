"""Tests for scripts/validate_pit_inputs.py — PIT validators."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_pit_inputs import (
    VERSION,
    SCHEMA_VERSION,
    validate_ctgov_pit,
    validate_catalyst_pit,
    build_validation_report,
    main,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_trial(
    nct_id: str = "NCT001",
    ticker: str = "ACME",
    first_posted: str = "2024-01-15",
    last_update_posted: str = "2024-06-01",
) -> dict:
    return {
        "nct_id": nct_id,
        "ticker": ticker,
        "first_posted": first_posted,
        "last_update_posted": last_update_posted,
    }


def _make_catalyst_events(tickers_data: dict) -> dict:
    """Build a catalyst_events_pit.v1-like dict."""
    return {
        "schema_version": "catalyst_events_pit.v1",
        "as_of_date": "2024-06-30",
        "tickers": tickers_data,
    }


# ===================================================================
# CTgov PIT validator tests
# ===================================================================

class TestValidateCtgovPit:
    def test_clean_data(self):
        trials = [
            _make_trial("NCT001", "AAA", "2024-01-10", "2024-03-01"),
            _make_trial("NCT002", "BBB", "2024-03-15", "2024-06-01"),
        ]
        result = validate_ctgov_pit(trials, "2024-06-30", mode="strict")
        assert result["clean"] is True
        assert result["first_posted_violations"] == 0
        assert result["total_records"] == 2

    def test_first_posted_violation(self):
        trials = [
            _make_trial("NCT001", "AAA", "2024-01-10"),  # OK
            _make_trial("NCT002", "BBB", "2024-07-15"),  # violation
            _make_trial("NCT003", "CCC", "2025-01-01"),  # violation
        ]
        result = validate_ctgov_pit(trials, "2024-06-30", mode="strict")
        assert result["clean"] is False
        assert result["first_posted_violations"] == 2
        assert len(result["first_posted_examples"]) == 2
        assert result["first_posted_examples"][0]["nct_id"] == "NCT002"

    def test_last_update_future_flagged(self):
        trials = [
            _make_trial("NCT001", "AAA", "2024-01-10", "2024-08-01"),
        ]
        result = validate_ctgov_pit(trials, "2024-06-30", mode="strict")
        # first_posted is OK, so clean=True in strict mode
        assert result["clean"] is True
        # But last_update_posted is future
        assert result["last_update_future_count"] == 1

    def test_degrade_mode_ignores_first_posted(self):
        trials = [
            _make_trial("NCT001", "AAA", "2024-07-15"),  # future
        ]
        result = validate_ctgov_pit(trials, "2024-06-30", mode="degrade")
        assert result["clean"] is True  # degrade ignores violations

    def test_empty_trials(self):
        result = validate_ctgov_pit([], "2024-06-30")
        assert result["clean"] is True
        assert result["total_records"] == 0

    def test_max_examples_cap(self):
        trials = [_make_trial(f"NCT{i:03d}", "AAA", "2025-01-01") for i in range(50)]
        result = validate_ctgov_pit(trials, "2024-06-30", mode="strict", max_examples=5)
        assert len(result["first_posted_examples"]) == 5
        assert result["first_posted_violations"] == 5  # capped at max_examples

    def test_month_only_date(self):
        trials = [_make_trial("NCT001", "AAA", "2024-03")]
        result = validate_ctgov_pit(trials, "2024-06-30")
        assert result["clean"] is True

    def test_exact_boundary(self):
        trials = [_make_trial("NCT001", "AAA", "2024-06-30")]
        result = validate_ctgov_pit(trials, "2024-06-30")
        assert result["clean"] is True
        assert result["first_posted_violations"] == 0

    def test_one_day_after_boundary(self):
        trials = [_make_trial("NCT001", "AAA", "2024-07-01")]
        result = validate_ctgov_pit(trials, "2024-06-30")
        assert result["clean"] is False
        assert result["first_posted_violations"] == 1


# ===================================================================
# Catalyst PIT validator tests
# ===================================================================

class TestValidateCatalystPit:
    def test_clean_data(self):
        events = _make_catalyst_events({
            "AAA": {
                "catalyst_mode": "specific_days",
                "nearest_disclosed_at": "2024-06-15",
                "nearest_event_source": "SEC_8K",
            },
            "BBB": {
                "catalyst_mode": "missing",  # no catalyst — skipped
            },
        })
        result = validate_catalyst_pit(events, "2024-06-30", mode="strict")
        assert result["clean"] is True
        assert result["missing_disclosed_at"] == 0
        assert result["future_disclosed_at"] == 0

    def test_missing_disclosed_at(self):
        events = _make_catalyst_events({
            "AAA": {
                "catalyst_mode": "specific_days",
                "nearest_disclosed_at": "",
                "nearest_event_source": "CTGOV",
            },
        })
        result = validate_catalyst_pit(events, "2024-06-30", mode="strict")
        assert result["clean"] is False
        assert result["missing_disclosed_at"] == 1
        assert result["missing_examples"][0]["ticker"] == "AAA"

    def test_future_disclosed_at(self):
        events = _make_catalyst_events({
            "AAA": {
                "catalyst_mode": "specific_days",
                "nearest_disclosed_at": "2024-07-15",
                "nearest_event_source": "SEC_8K",
            },
        })
        result = validate_catalyst_pit(events, "2024-06-30", mode="strict")
        assert result["clean"] is False
        assert result["future_disclosed_at"] == 1

    def test_degrade_mode_allows_missing(self):
        events = _make_catalyst_events({
            "AAA": {
                "catalyst_mode": "specific_days",
                "nearest_disclosed_at": "",
            },
        })
        result = validate_catalyst_pit(events, "2024-06-30", mode="degrade")
        assert result["clean"] is True  # degrade allows missing
        assert result["missing_disclosed_at"] == 1  # still counted

    def test_degrade_mode_rejects_future(self):
        events = _make_catalyst_events({
            "AAA": {
                "catalyst_mode": "specific_days",
                "nearest_disclosed_at": "2024-07-15",
            },
        })
        result = validate_catalyst_pit(events, "2024-06-30", mode="degrade")
        assert result["clean"] is False  # future is always a violation

    def test_missing_tickers_skipped(self):
        events = _make_catalyst_events({
            "AAA": {"catalyst_mode": "missing"},
            "BBB": {"catalyst_mode": "missing"},
        })
        result = validate_catalyst_pit(events, "2024-06-30")
        assert result["clean"] is True
        assert result["total_tickers"] == 2

    def test_flat_dict_format(self):
        """Accepts flat ticker dict without 'tickers' wrapper."""
        flat = {
            "AAA": {
                "catalyst_mode": "specific_days",
                "nearest_disclosed_at": "2024-06-15",
            },
        }
        result = validate_catalyst_pit(flat, "2024-06-30")
        assert result["clean"] is True

    def test_empty_events(self):
        events = _make_catalyst_events({})
        result = validate_catalyst_pit(events, "2024-06-30")
        assert result["clean"] is True
        assert result["total_tickers"] == 0

    def test_max_examples_cap(self):
        tickers = {}
        for i in range(50):
            tickers[f"T{i:03d}"] = {
                "catalyst_mode": "specific_days",
                "nearest_disclosed_at": "",
            }
        events = _make_catalyst_events(tickers)
        result = validate_catalyst_pit(events, "2024-06-30", max_examples=3)
        assert len(result["missing_examples"]) == 3
        assert result["missing_disclosed_at"] == 3  # capped at max_examples


# ===================================================================
# Report builder tests
# ===================================================================

class TestBuildValidationReport:
    def test_both_clean(self):
        ctgov = {"source": "ctgov", "clean": True, "first_posted_violations": 0,
                 "total_records": 100, "last_update_future_count": 0,
                 "first_posted_examples": [], "last_update_future_examples": []}
        cat = {"source": "catalyst", "clean": True, "total_tickers": 50,
               "missing_disclosed_at": 0, "future_disclosed_at": 0,
               "mode": "strict", "missing_examples": [], "future_examples": []}

        report = build_validation_report("2024-06-30", "strict", ctgov, cat)
        assert report["schema_version"] == SCHEMA_VERSION
        assert report["overall_clean"] is True
        assert "ctgov" in report["components"]
        assert "catalyst" in report["components"]

    def test_ctgov_violation_propagates(self):
        ctgov = {"source": "ctgov", "clean": False, "first_posted_violations": 3,
                 "total_records": 100, "last_update_future_count": 0,
                 "first_posted_examples": [], "last_update_future_examples": []}

        report = build_validation_report("2024-06-30", "strict", ctgov)
        assert report["overall_clean"] is False

    def test_catalyst_only(self):
        cat = {"source": "catalyst", "clean": True, "total_tickers": 50,
               "missing_disclosed_at": 0, "future_disclosed_at": 0,
               "mode": "strict", "missing_examples": [], "future_examples": []}

        report = build_validation_report("2024-06-30", "strict", catalyst_result=cat)
        assert report["overall_clean"] is True
        assert "ctgov" not in report["components"]

    def test_file_hashes_included(self):
        report = build_validation_report(
            "2024-06-30", "strict",
            file_hashes={"ctgov_cache": "abc123"},
        )
        assert report["file_hashes"]["ctgov_cache"] == "abc123"


# ===================================================================
# CLI tests
# ===================================================================

class TestCLI:
    def test_exit_0_when_clean(self, tmp_path):
        # Write clean CTgov cache
        trials = [_make_trial("NCT001", "AAA", "2024-01-10")]
        cache_path = tmp_path / "trials.json"
        cache_path.write_text(json.dumps(trials))
        out_path = tmp_path / "report.json"

        rc = main([
            "--as-of-date", "2024-06-30",
            "--ctgov-cache", str(cache_path),
            "--out", str(out_path),
        ])
        assert rc == 0
        report = json.loads(out_path.read_text())
        assert report["overall_clean"] is True

    def test_exit_2_when_violations(self, tmp_path):
        trials = [_make_trial("NCT001", "AAA", "2024-07-15")]
        cache_path = tmp_path / "trials.json"
        cache_path.write_text(json.dumps(trials))
        out_path = tmp_path / "report.json"

        rc = main([
            "--as-of-date", "2024-06-30",
            "--ctgov-cache", str(cache_path),
            "--out", str(out_path),
        ])
        assert rc == 2

    def test_both_inputs(self, tmp_path):
        trials = [_make_trial("NCT001", "AAA", "2024-01-10")]
        cache_path = tmp_path / "trials.json"
        cache_path.write_text(json.dumps(trials))

        events = _make_catalyst_events({
            "AAA": {
                "catalyst_mode": "specific_days",
                "nearest_disclosed_at": "2024-06-15",
                "nearest_event_source": "SEC_8K",
            },
        })
        cat_path = tmp_path / "catalyst.json"
        cat_path.write_text(json.dumps(events))
        out_path = tmp_path / "report.json"

        rc = main([
            "--as-of-date", "2024-06-30",
            "--ctgov-cache", str(cache_path),
            "--catalyst-events", str(cat_path),
            "--out", str(out_path),
        ])
        assert rc == 0
        report = json.loads(out_path.read_text())
        assert "ctgov" in report["components"]
        assert "catalyst" in report["components"]

    def test_missing_file_does_not_crash(self, tmp_path):
        out_path = tmp_path / "report.json"
        rc = main([
            "--as-of-date", "2024-06-30",
            "--ctgov-cache", str(tmp_path / "nonexistent.json"),
            "--out", str(out_path),
        ])
        # No components validated, so overall_clean=True (vacuously)
        assert rc == 0
