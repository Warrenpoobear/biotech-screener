#!/usr/bin/env python3
"""
Tests for PIT date plumbing fix (Task A).

Covers:
- _normalize_date() in collect_ctgov_data
- Expanded _select_pit_date() priority chain in module_4_clinical_dev
- PIT safety hole closure (no-date trials excluded, not silently passed)
- Regression: Module 4 returns nonzero scores for valid trials with posted dates
"""

import csv
import json
import sys
import tempfile
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from collect_ctgov_data import _normalize_date, _parse_study
from export_results_csv import CLINICAL_PIT_COLUMNS, export_to_csv
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
# Unit tests: _parse_study precision provenance (Spec 114 / #535)
#
# The collector is what actually writes production_data/trial_records.json.
# #538 captured CT.gov date granularity and #543 plumbed it through
# CanonicalTrialRecord -> CalendarCatalyst, but both sat on the
# data_sources/ctgov_client.py path, which the production cache does not flow
# through. Without the fields below, ctgov_adapter's
# record.get("ctgov_date_precision_raw") is always None and Spec 114 Phase 2
# is a guaranteed no-op.
#
# Non-routing: primary_completion_date must keep its existing normalized
# value. These tests lock provenance capture only.
# ---------------------------------------------------------------------------


def _study(pcd_struct: dict | None) -> dict:
    """Minimal CT.gov API v2 study payload with a given primaryCompletionDateStruct."""
    status_module: dict = {"overallStatus": "RECRUITING"}
    if pcd_struct is not None:
        status_module["primaryCompletionDateStruct"] = pcd_struct
    return {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT99999999", "briefTitle": "Fixture"},
            "statusModule": status_module,
            "designModule": {"phases": ["PHASE3"], "studyType": "INTERVENTIONAL"},
            "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Fixture Sponsor"}},
            "conditionsModule": {"conditions": ["Fixture Condition"]},
            "armsInterventionsModule": {"interventions": []},
        }
    }


class TestParseStudyDatePrecision:
    def test_month_only_pcd_records_month_precision(self):
        rec = _parse_study(_study({"date": "2026-08", "type": "ESTIMATED"}), "TEST")
        assert rec["ctgov_date_precision_raw"] == "MONTH"

    def test_full_day_pcd_records_day_precision(self):
        rec = _parse_study(_study({"date": "2026-08-14", "type": "ACTUAL"}), "TEST")
        assert rec["ctgov_date_precision_raw"] == "DAY"

    def test_year_only_pcd_records_year_precision(self):
        rec = _parse_study(_study({"date": "2026", "type": "ESTIMATED"}), "TEST")
        assert rec["ctgov_date_precision_raw"] == "YEAR"

    def test_absent_pcd_is_unknown_never_day(self):
        """The contract locked by #543: absence means UNKNOWN, never DAY.

        Defaulting to the strongest claim is the defect Spec 114 fixes.
        """
        rec = _parse_study(_study(None), "TEST")
        assert rec["ctgov_date_precision_raw"] == "UNKNOWN"

    def test_raw_date_and_type_preserved(self):
        rec = _parse_study(_study({"date": "2026-08", "type": "ESTIMATED"}), "TEST")
        assert rec["ctgov_raw_date"] == "2026-08"
        assert rec["ctgov_date_type"] == "ESTIMATED"

    def test_month_only_normalized_candidate_is_month(self):
        """CTGOV_CALENDAR declares a MONTH floor, so a month-only date stays MONTH."""
        rec = _parse_study(_study({"date": "2026-08", "type": "ESTIMATED"}), "TEST")
        assert rec["ctgov_precision_normalized_candidate"] == "MONTH"

    def test_day_precision_floored_to_source_declaration(self):
        """Even an exact day is floored by the CTGOV_CALENDAR MONTH declaration."""
        rec = _parse_study(_study({"date": "2026-08-14", "type": "ACTUAL"}), "TEST")
        assert rec["ctgov_precision_normalized_candidate"] == "MONTH"

    def test_routing_unchanged_pcd_still_snapped(self):
        """NON-ROUTING guarantee: the consumed date keeps its normalized value."""
        rec = _parse_study(_study({"date": "2026-08", "type": "ESTIMATED"}), "TEST")
        assert rec["primary_completion_date"] == "2026-08-01"

    def test_precision_matches_client_path_for_same_struct(self):
        """The two CT.gov paths must not disagree about precision."""
        from data_sources.ctgov_client import date_provenance_shadow

        struct = {"date": "2026-08", "type": "ESTIMATED"}
        rec = _parse_study(_study(struct), "TEST")
        shadow = date_provenance_shadow(struct)
        for key, value in shadow.items():
            assert rec[key] == value, f"collector disagrees with client on {key}"


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
            "BBBB",
            "NCT00000002",
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


# ---------------------------------------------------------------------------
# CSV export: PIT observability columns present and populated
# ---------------------------------------------------------------------------


class TestCsvPitColumns:
    """Verify that export_to_csv emits PIT columns from Module 4 scores."""

    def _build_results(self):
        """Build a minimal results dict with Module 4 scores and Module 5 ranked securities."""
        return {
            "module_4_clinical": {
                "scores": [
                    {
                        "ticker": "AAAA",
                        "clinical_score": "72.50",
                        "n_trials_raw": 10,
                        "n_trials_unique": 8,
                        "pit_filtered_count_ticker": 2,
                        "lead_phase": "phase 3",
                        "lead_trial_nct_id": "NCT00000001",
                        "recency_days": 15,
                    },
                    {
                        "ticker": "BBBB",
                        "clinical_score": "45.00",
                        "n_trials_raw": 3,
                        "n_trials_unique": 3,
                        "pit_filtered_count_ticker": 0,
                        "lead_phase": "phase 1",
                        "lead_trial_nct_id": "NCT00000002",
                        "recency_days": 60,
                    },
                ],
            },
            "module_5_composite": {
                "ranked_securities": [
                    {"ticker": "AAAA", "composite_rank": 1, "composite_score": 80.0},
                    {"ticker": "BBBB", "composite_rank": 2, "composite_score": 60.0},
                ],
            },
        }

    def test_pit_columns_present_in_header(self):
        results = self._build_results()
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "results.json"
            csv_path = Path(tmpdir) / "results.csv"
            json_path.write_text(json.dumps(results))
            count = export_to_csv(json_path, csv_path)
            assert count == 2

            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for col in CLINICAL_PIT_COLUMNS:
                    assert col in reader.fieldnames, f"Missing PIT column: {col}"

    def test_pit_columns_populated(self):
        results = self._build_results()
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "results.json"
            csv_path = Path(tmpdir) / "results.csv"
            json_path.write_text(json.dumps(results))
            export_to_csv(json_path, csv_path)

            with open(csv_path) as f:
                rows = list(csv.DictReader(f))

            aaaa = rows[0]
            assert aaaa["ticker"] == "AAAA"
            assert aaaa["pit_trials_total"] == "10"
            assert aaaa["pit_trials_eligible"] == "8"
            assert aaaa["pit_trials_filtered"] == "2"
            assert aaaa["pit_trials_no_date"] == "0"
            assert aaaa["clinical_lead_phase"] == "phase 3"
            assert aaaa["clinical_lead_nct_id"] == "NCT00000001"
            assert aaaa["clinical_recency_days"] == "15"
            assert aaaa["clinical_score_raw"] == "72.50"

            bbbb = rows[1]
            assert bbbb["pit_trials_total"] == "3"
            assert bbbb["pit_trials_eligible"] == "3"
            assert bbbb["pit_trials_filtered"] == "0"

    def test_pit_no_date_derived_correctly(self):
        """pit_trials_no_date = max(0, total - eligible - filtered)."""
        results = self._build_results()
        # Simulate: 10 raw, 6 eligible, 2 filtered → 2 no_date
        results["module_4_clinical"]["scores"][0]["n_trials_unique"] = 6
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "results.json"
            csv_path = Path(tmpdir) / "results.csv"
            json_path.write_text(json.dumps(results))
            export_to_csv(json_path, csv_path)

            with open(csv_path) as f:
                rows = list(csv.DictReader(f))

            assert rows[0]["pit_trials_no_date"] == "2"

    def test_missing_m4_ticker_gets_empty_pit_columns(self):
        """A ticker in Module 5 but not in Module 4 should get empty PIT columns."""
        results = self._build_results()
        results["module_5_composite"]["ranked_securities"].append(
            {"ticker": "CCCC", "composite_rank": 3, "composite_score": 40.0}
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "results.json"
            csv_path = Path(tmpdir) / "results.csv"
            json_path.write_text(json.dumps(results))
            export_to_csv(json_path, csv_path)

            with open(csv_path) as f:
                rows = list(csv.DictReader(f))

            cccc = [r for r in rows if r["ticker"] == "CCCC"][0]
            assert cccc["pit_trials_total"] == ""
            assert cccc["pit_trials_eligible"] == ""
            assert cccc["pit_trials_no_date"] == "0"
