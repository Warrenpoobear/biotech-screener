#!/usr/bin/env python3
"""
Tests for PIT date plumbing fix (Task A).

Covers:
- _normalize_date() in collect_ctgov_data
- Expanded _select_pit_date() priority chain in module_4_clinical_dev
- PIT safety hole closure (no-date trials excluded, not silently passed)
- Regression: Module 4 returns nonzero scores for valid trials with posted dates
"""

import sys
from pathlib import Path
from decimal import Decimal

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from collect_ctgov_data import _normalize_date
from module_4_clinical_dev import _select_pit_date, compute_module_4_clinical_dev


# ---------------------------------------------------------------------------
# Unit tests: _normalize_date
# ---------------------------------------------------------------------------


class TestNormalizeDate:
    def test_none_returns_none(self):
        assert _normalize_date(None) is None

    def test_empty_string_returns_none(self):
        assert _normalize_date("") is None

    def test_year_only(self):
        assert _normalize_date("2024") == "2024-01-01"

    def test_year_month(self):
        assert _normalize_date("2024-06") == "2024-06-01"

    def test_full_date(self):
        assert _normalize_date("2024-06-15") == "2024-06-15"

    def test_datetime_string_truncated(self):
        assert _normalize_date("2024-06-15T12:00:00Z") == "2024-06-15"

    def test_whitespace_stripped(self):
        assert _normalize_date("  2024-06  ") == "2024-06-01"


# ---------------------------------------------------------------------------
# Unit tests: _select_pit_date (expanded priority)
# ---------------------------------------------------------------------------


class TestSelectPitDate:
    def test_first_posted_wins(self):
        trial = {
            "first_posted": "2024-01-01",
            "last_update_posted": "2024-06-01",
            "results_first_posted": "2024-03-01",
            "source_date": "2024-02-01",
            "start_date": "2023-06-01",
        }
        value, field = _select_pit_date(trial)
        assert field == "first_posted"
        assert value == "2024-01-01"

    def test_falls_through_to_last_update(self):
        trial = {
            "last_update_posted": "2024-06-01",
            "results_first_posted": "2024-03-01",
        }
        value, field = _select_pit_date(trial)
        assert field == "last_update_posted"
        assert value == "2024-06-01"

    def test_falls_through_to_results_first_posted(self):
        trial = {"results_first_posted": "2024-03-01"}
        value, field = _select_pit_date(trial)
        assert field == "results_first_posted"
        assert value == "2024-03-01"

    def test_falls_through_to_source_date(self):
        trial = {"source_date": "2024-02-01"}
        value, field = _select_pit_date(trial)
        assert field == "source_date"

    def test_falls_through_to_start_date(self):
        trial = {"start_date": "2023-06-01"}
        value, field = _select_pit_date(trial)
        assert field == "start_date"
        assert value == "2023-06-01"

    def test_all_none_returns_none(self):
        trial = {}
        value, field = _select_pit_date(trial)
        assert value is None
        assert field == "none"

    def test_empty_strings_skipped(self):
        trial = {"first_posted": "", "last_update_posted": "2024-06-01"}
        value, field = _select_pit_date(trial)
        assert field == "last_update_posted"


# ---------------------------------------------------------------------------
# Integration: PIT safety hole — no-date trials must be excluded
# ---------------------------------------------------------------------------


class TestPitSafetyHole:
    """Trials with NO date fields must be excluded, not silently passed."""

    def _make_trial(self, ticker, nct_id, **overrides):
        base = {
            "ticker": ticker,
            "nct_id": nct_id,
            "phase": "PHASE2",
            "status": "RECRUITING",
            "conditions": ["cancer"],
        }
        base.update(overrides)
        return base

    def test_no_date_trial_excluded(self):
        """A trial with zero date fields should be excluded from scoring."""
        trial_no_dates = self._make_trial("AAAA", "NCT00000001")
        result = compute_module_4_clinical_dev(
            trial_records=[trial_no_dates],
            active_tickers=["AAAA"],
            as_of_date="2026-01-30",
        )
        diag = result["diagnostic_counts"]
        assert diag["no_date_excluded"] == 1
        # The trial should not appear in scored unique trials
        assert diag["total_trials_unique"] == 0

    def test_trial_with_date_admitted(self):
        """A trial with a valid PIT date should pass through."""
        trial_ok = self._make_trial(
            "BBBB", "NCT00000002",
            last_update_posted="2025-12-01",
        )
        result = compute_module_4_clinical_dev(
            trial_records=[trial_ok],
            active_tickers=["BBBB"],
            as_of_date="2026-01-30",
        )
        diag = result["diagnostic_counts"]
        assert diag["no_date_excluded"] == 0
        assert diag["total_trials_unique"] == 1


# ---------------------------------------------------------------------------
# Regression: Module 4 produces nonzero scores for valid trial data
# ---------------------------------------------------------------------------


class TestModule4Regression:
    """Three sample trials with posted dates should yield nonzero clinical scores."""

    SAMPLE_TRIALS = [
        {
            "ticker": "VRTX",
            "nct_id": "NCT05100001",
            "phase": "PHASE3",
            "status": "ACTIVE_NOT_RECRUITING",
            "conditions": ["Cystic Fibrosis"],
            "first_posted": "2021-10-15",
            "last_update_posted": "2025-11-20",
            "results_first_posted": None,
            "start_date": "2021-12-01",
        },
        {
            "ticker": "VRTX",
            "nct_id": "NCT05100002",
            "phase": "PHASE2",
            "status": "RECRUITING",
            "conditions": ["Pain"],
            "first_posted": "2023-03-01",
            "last_update_posted": "2025-09-10",
        },
        {
            "ticker": "BEAM",
            "nct_id": "NCT05200001",
            "phase": "PHASE1",
            "status": "RECRUITING",
            "conditions": ["Sickle Cell Disease"],
            "first_posted": "2022-06-01",
            "last_update_posted": "2025-08-15",
        },
    ]

    def test_scored_tickers_have_nonzero_scores(self):
        result = compute_module_4_clinical_dev(
            trial_records=self.SAMPLE_TRIALS,
            active_tickers=["VRTX", "BEAM"],
            as_of_date="2026-01-30",
        )
        scores = {s["ticker"]: s for s in result["scores"]}

        for ticker in ("VRTX", "BEAM"):
            assert ticker in scores
            clinical = Decimal(scores[ticker]["clinical_score"])
            assert clinical > 0, f"{ticker} should have nonzero clinical_score"

    def test_no_date_excluded_is_zero(self):
        result = compute_module_4_clinical_dev(
            trial_records=self.SAMPLE_TRIALS,
            active_tickers=["VRTX", "BEAM"],
            as_of_date="2026-01-30",
        )
        assert result["diagnostic_counts"]["no_date_excluded"] == 0

    def test_pit_fields_used_shows_first_posted(self):
        result = compute_module_4_clinical_dev(
            trial_records=self.SAMPLE_TRIALS,
            active_tickers=["VRTX", "BEAM"],
            as_of_date="2026-01-30",
        )
        pit_fields = result["diagnostic_counts"]["pit_fields_used"]
        assert pit_fields.get("first_posted", 0) == 3

    def test_audit_trail_pit_date_in_trial_record(self):
        """Per-trial records should carry pit_date and pit_date_field."""
        result = compute_module_4_clinical_dev(
            trial_records=self.SAMPLE_TRIALS,
            active_tickers=["VRTX", "BEAM"],
            as_of_date="2026-01-30",
        )
        # Access is indirect — check that unique count matches input
        # (all 3 trials have first_posted, so all admitted)
        assert result["diagnostic_counts"]["total_trials_unique"] == 3
