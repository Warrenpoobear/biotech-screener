"""Tests for tools/backfill_ctgov_pit_history.py — PIT-safe CTgov history backfill."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import List

import pytest

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from tools.backfill_ctgov_pit_history import (
    TOOL_VERSION,
    SCHEMA_VERSION,
    PIT_DATE_PRIORITY,
    generate_quarter_end_dates,
    generate_month_end_dates,
    generate_weekly_dates,
    generate_daily_dates,
    generate_dates,
    is_pit_admitted,
    filter_trials_pit,
    _stable_sort_trials,
    _is_date_complete,
    build_pit_cache_for_date,
    backfill_ctgov_history,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_trial(
    nct_id: str = "NCT00000001",
    ticker: str = "ACME",
    first_posted: str = "2024-01-15",
    last_update_posted: str = "2024-06-01",
    phase: str = "PHASE2",
    status: str = "RECRUITING",
    study_type: str = "INTERVENTIONAL",
    primary_completion_date: str = "2025-12-01",
) -> dict:
    return {
        "nct_id": nct_id,
        "ticker": ticker,
        "first_posted": first_posted,
        "last_update_posted": last_update_posted,
        "phase": phase,
        "status": status,
        "study_type": study_type,
        "primary_completion_date": primary_completion_date,
        "brief_title": f"Trial {nct_id}",
    }


SAMPLE_TRIALS = [
    _make_trial("NCT001", "AAA", "2023-01-10", "2023-06-01"),
    _make_trial("NCT002", "BBB", "2024-03-15", "2024-07-01"),
    _make_trial("NCT003", "CCC", "2024-06-20", "2024-12-01"),
    _make_trial("NCT004", "DDD", "2025-01-05", "2025-03-15"),
    _make_trial("NCT005", "EEE", "2025-07-01", "2025-09-01"),
    _make_trial("NCT006", "FFF", "2026-01-15", "2026-02-01"),
]


# ===================================================================
# Date generation tests
# ===================================================================

class TestQuarterEndDates:
    def test_basic_range(self):
        dates = generate_quarter_end_dates(date(2024, 1, 1), date(2024, 12, 31))
        assert dates == [
            date(2024, 3, 31),
            date(2024, 6, 30),
            date(2024, 9, 30),
            date(2024, 12, 31),
        ]

    def test_partial_year(self):
        dates = generate_quarter_end_dates(date(2024, 4, 1), date(2024, 9, 30))
        assert dates == [date(2024, 6, 30), date(2024, 9, 30)]

    def test_single_quarter(self):
        dates = generate_quarter_end_dates(date(2024, 3, 31), date(2024, 3, 31))
        assert dates == [date(2024, 3, 31)]

    def test_empty_when_from_after_to(self):
        dates = generate_quarter_end_dates(date(2025, 1, 1), date(2024, 1, 1))
        assert dates == []

    def test_cross_year(self):
        dates = generate_quarter_end_dates(date(2024, 10, 1), date(2025, 6, 30))
        assert dates == [date(2024, 12, 31), date(2025, 3, 31), date(2025, 6, 30)]

    def test_inclusive_bounds(self):
        # Quarter-end date is exactly date_from
        dates = generate_quarter_end_dates(date(2024, 6, 30), date(2024, 9, 30))
        assert date(2024, 6, 30) in dates


class TestMonthEndDates:
    def test_basic(self):
        dates = generate_month_end_dates(date(2024, 1, 1), date(2024, 3, 31))
        assert dates == [date(2024, 1, 31), date(2024, 2, 29), date(2024, 3, 31)]

    def test_empty_when_from_after_to(self):
        assert generate_month_end_dates(date(2025, 1, 1), date(2024, 1, 1)) == []

    def test_feb_non_leap_year(self):
        dates = generate_month_end_dates(date(2023, 2, 1), date(2023, 2, 28))
        assert dates == [date(2023, 2, 28)]


class TestWeeklyDates:
    def test_basic(self):
        dates = generate_weekly_dates(date(2024, 1, 1), date(2024, 1, 31))
        # Fridays in Jan 2024: 5, 12, 19, 26
        assert dates == [date(2024, 1, 5), date(2024, 1, 12), date(2024, 1, 19), date(2024, 1, 26)]

    def test_empty(self):
        assert generate_weekly_dates(date(2025, 1, 1), date(2024, 1, 1)) == []


class TestDailyDates:
    def test_skips_weekends(self):
        dates = generate_daily_dates(date(2024, 1, 1), date(2024, 1, 7))
        # Jan 1=Mon, 2=Tue, 3=Wed, 4=Thu, 5=Fri, 6=Sat, 7=Sun
        assert dates == [
            date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3),
            date(2024, 1, 4), date(2024, 1, 5),
        ]


class TestGenerateDates:
    def test_quarterly(self):
        assert generate_dates(date(2024, 1, 1), date(2024, 6, 30), "quarterly") == [
            date(2024, 3, 31), date(2024, 6, 30),
        ]

    def test_unknown_cadence_raises(self):
        with pytest.raises(ValueError, match="Unknown cadence"):
            generate_dates(date(2024, 1, 1), date(2024, 12, 31), "biweekly")


# ===================================================================
# PIT filtering tests
# ===================================================================

class TestIsPitAdmitted:
    def test_admitted_by_first_posted(self):
        trial = _make_trial(first_posted="2024-01-01", last_update_posted="2024-06-01")
        ok, field = is_pit_admitted(trial, date(2024, 3, 15))
        assert ok is True
        assert field == "first_posted"

    def test_rejected_by_first_posted(self):
        trial = _make_trial(first_posted="2024-07-01", last_update_posted="2024-08-01")
        ok, field = is_pit_admitted(trial, date(2024, 6, 30))
        assert ok is False
        assert field == "first_posted"

    def test_admitted_on_exact_date(self):
        trial = _make_trial(first_posted="2024-06-30")
        ok, _ = is_pit_admitted(trial, date(2024, 6, 30))
        assert ok is True

    def test_fallback_to_last_update_posted(self):
        trial = {"last_update_posted": "2024-03-01", "nct_id": "NCT001"}
        ok, field = is_pit_admitted(trial, date(2024, 6, 30))
        assert ok is True
        assert field == "last_update_posted"

    def test_no_dates_returns_false(self):
        trial = {"nct_id": "NCT001"}
        ok, field = is_pit_admitted(trial, date(2024, 6, 30))
        assert ok is False
        assert field == ""

    def test_month_only_date_parsed(self):
        trial = _make_trial(first_posted="2024-03")
        ok, _ = is_pit_admitted(trial, date(2024, 6, 30))
        assert ok is True


class TestFilterTrialsPit:
    def test_filters_future_trials(self):
        as_of = date(2024, 6, 30)
        admitted, n_adm, n_filt = filter_trials_pit(SAMPLE_TRIALS, as_of)
        # NCT001 (2023-01-10) ✓, NCT002 (2024-03-15) ✓, NCT003 (2024-06-20) ✓
        # NCT004 (2025-01-05) ✗, NCT005 (2025-07-01) ✗, NCT006 (2026-01-15) ✗
        assert n_adm == 3
        assert n_filt == 3
        tickers = {t["ticker"] for t in admitted}
        assert tickers == {"AAA", "BBB", "CCC"}

    def test_all_admitted_for_future_as_of(self):
        as_of = date(2030, 1, 1)
        admitted, n_adm, n_filt = filter_trials_pit(SAMPLE_TRIALS, as_of)
        assert n_adm == 6
        assert n_filt == 0

    def test_empty_input(self):
        admitted, n_adm, n_filt = filter_trials_pit([], date(2024, 1, 1))
        assert admitted == []
        assert n_adm == 0
        assert n_filt == 0


class TestStableSort:
    def test_sort_order(self):
        trials = [
            _make_trial("NCT002", "BBB"),
            _make_trial("NCT001", "AAA"),
            _make_trial("NCT003", "AAA"),
        ]
        sorted_trials = _stable_sort_trials(trials)
        assert [t["nct_id"] for t in sorted_trials] == ["NCT001", "NCT003", "NCT002"]

    def test_determinism(self):
        trials = [_make_trial(f"NCT{i:03d}", f"T{i}") for i in range(20)]
        s1 = _stable_sort_trials(list(trials))
        s2 = _stable_sort_trials(list(reversed(trials)))
        assert [t["nct_id"] for t in s1] == [t["nct_id"] for t in s2]


# ===================================================================
# Resume tests
# ===================================================================

class TestIsDateComplete:
    def test_returns_false_for_missing_file(self, tmp_path):
        assert _is_date_complete(tmp_path, date(2024, 3, 31)) is False

    def test_returns_true_for_valid_cache(self, tmp_path):
        cache_file = tmp_path / "trial_records_2024-03-31.json"
        cache_file.write_text(json.dumps([{"nct_id": "NCT001"}]))
        assert _is_date_complete(tmp_path, date(2024, 3, 31)) is True

    def test_returns_false_for_invalid_json(self, tmp_path):
        cache_file = tmp_path / "trial_records_2024-03-31.json"
        cache_file.write_text("not json")
        assert _is_date_complete(tmp_path, date(2024, 3, 31)) is False

    def test_returns_false_for_non_list(self, tmp_path):
        cache_file = tmp_path / "trial_records_2024-03-31.json"
        cache_file.write_text(json.dumps({"not": "a list"}))
        assert _is_date_complete(tmp_path, date(2024, 3, 31)) is False


# ===================================================================
# Build cache for date tests
# ===================================================================

class TestBuildPitCacheForDate:
    def test_writes_cache_and_meta(self, tmp_path):
        result = build_pit_cache_for_date(
            as_of=date(2024, 6, 30),
            all_trials=SAMPLE_TRIALS,
            out_root=tmp_path,
            input_hash="abc12345",
        )
        cache_path = tmp_path / "trial_records_2024-06-30.json"
        meta_path = tmp_path / "trial_records_2024-06-30.meta.json"
        assert cache_path.exists()
        assert meta_path.exists()

        # Verify cache contents
        with open(cache_path) as f:
            cached = json.load(f)
        assert len(cached) == 3  # NCT001, NCT002, NCT003
        assert all(t["first_posted"] <= "2024-06-30" for t in cached)

        # Verify metadata
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["schema_version"] == SCHEMA_VERSION
        assert meta["as_of_date"] == "2024-06-30"
        assert meta["pit_mode"] == "strict"
        assert meta["input_hash"] == "abc12345"
        assert meta["trials_admitted"] == 3
        assert meta["trials_filtered"] == 3

    def test_result_dict(self, tmp_path):
        result = build_pit_cache_for_date(
            as_of=date(2024, 6, 30),
            all_trials=SAMPLE_TRIALS,
            out_root=tmp_path,
            input_hash="abc",
        )
        assert result["as_of_date"] == "2024-06-30"
        assert result["trials_admitted"] == 3
        assert result["trials_filtered"] == 3
        assert result["trials_total"] == 6

    def test_deterministic_output(self, tmp_path):
        """Same input → same output (ignoring created_at)."""
        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        build_pit_cache_for_date(date(2024, 6, 30), SAMPLE_TRIALS, out1, "hash1")
        build_pit_cache_for_date(date(2024, 6, 30), SAMPLE_TRIALS, out2, "hash1")

        with open(out1 / "trial_records_2024-06-30.json") as f:
            d1 = json.load(f)
        with open(out2 / "trial_records_2024-06-30.json") as f:
            d2 = json.load(f)
        # Trial content should be identical
        assert d1 == d2

    def test_empty_trials(self, tmp_path):
        result = build_pit_cache_for_date(
            as_of=date(2024, 6, 30),
            all_trials=[],
            out_root=tmp_path,
            input_hash="empty",
        )
        assert result["trials_admitted"] == 0
        assert result["trials_filtered"] == 0


# ===================================================================
# Backfill orchestrator tests
# ===================================================================

class TestBackfillCtgovHistory:
    def test_basic_quarterly(self, tmp_path):
        trial_path = tmp_path / "trials.json"
        trial_path.write_text(json.dumps(SAMPLE_TRIALS))

        summary = backfill_ctgov_history(
            date_from=date(2024, 1, 1),
            date_to=date(2024, 12, 31),
            trial_records_path=trial_path,
            cadence="quarterly",
            out_root=tmp_path / "cache",
        )
        assert summary["total_dates"] == 4
        assert summary["dates_skipped"] == 0
        assert len(summary["per_date"]) == 4

        # Verify files exist
        for d in ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"]:
            assert (tmp_path / "cache" / f"trial_records_{d}.json").exists()

    def test_resume_skips_existing(self, tmp_path):
        trial_path = tmp_path / "trials.json"
        trial_path.write_text(json.dumps(SAMPLE_TRIALS))
        out_root = tmp_path / "cache"

        # First run
        backfill_ctgov_history(
            date_from=date(2024, 1, 1),
            date_to=date(2024, 6, 30),
            trial_records_path=trial_path,
            cadence="quarterly",
            out_root=out_root,
        )

        # Second run with resume
        summary = backfill_ctgov_history(
            date_from=date(2024, 1, 1),
            date_to=date(2024, 6, 30),
            trial_records_path=trial_path,
            cadence="quarterly",
            out_root=out_root,
            resume=True,
        )
        assert summary["dates_skipped"] == 2
        assert all(d.get("skipped") for d in summary["per_date"])

    def test_max_dates(self, tmp_path):
        trial_path = tmp_path / "trials.json"
        trial_path.write_text(json.dumps(SAMPLE_TRIALS))

        summary = backfill_ctgov_history(
            date_from=date(2024, 1, 1),
            date_to=date(2024, 12, 31),
            trial_records_path=trial_path,
            cadence="quarterly",
            out_root=tmp_path / "cache",
            max_dates=2,
        )
        assert summary["total_dates"] == 2
        assert len(summary["per_date"]) == 2

    def test_report_flag(self, tmp_path):
        trial_path = tmp_path / "trials.json"
        trial_path.write_text(json.dumps(SAMPLE_TRIALS))

        summary = backfill_ctgov_history(
            date_from=date(2024, 1, 1),
            date_to=date(2024, 6, 30),
            trial_records_path=trial_path,
            cadence="quarterly",
            out_root=tmp_path / "cache",
            report=True,
        )
        assert "coverage_report" in summary
        rpt = summary["coverage_report"]
        assert rpt["total_dates"] == 2
        assert rpt["min_admitted"] <= rpt["max_admitted"]

    def test_pit_mode_in_summary(self, tmp_path):
        trial_path = tmp_path / "trials.json"
        trial_path.write_text(json.dumps(SAMPLE_TRIALS))

        summary = backfill_ctgov_history(
            date_from=date(2024, 3, 31),
            date_to=date(2024, 3, 31),
            trial_records_path=trial_path,
            cadence="quarterly",
            out_root=tmp_path / "cache",
            pit_mode="degrade",
        )
        assert summary["pit_mode"] == "degrade"

    def test_non_list_input_raises(self, tmp_path):
        trial_path = tmp_path / "trials.json"
        trial_path.write_text(json.dumps({"not": "a list"}))
        with pytest.raises(ValueError, match="trial_records must be a list"):
            backfill_ctgov_history(
                date_from=date(2024, 1, 1),
                date_to=date(2024, 3, 31),
                trial_records_path=trial_path,
                cadence="quarterly",
                out_root=tmp_path / "cache",
            )

    def test_monotonic_admission_counts(self, tmp_path):
        """Later dates should admit >= earlier dates' trial counts."""
        trial_path = tmp_path / "trials.json"
        trial_path.write_text(json.dumps(SAMPLE_TRIALS))

        summary = backfill_ctgov_history(
            date_from=date(2024, 1, 1),
            date_to=date(2025, 12, 31),
            trial_records_path=trial_path,
            cadence="quarterly",
            out_root=tmp_path / "cache",
        )
        counts = [d["trials_admitted"] for d in summary["per_date"]]
        for i in range(1, len(counts)):
            assert counts[i] >= counts[i - 1], (
                f"Non-monotonic: {counts[i]} < {counts[i-1]} at index {i}"
            )

    def test_pit_invariant_no_future_trials(self, tmp_path):
        """Verify no trial with first_posted > as_of_date in output."""
        trial_path = tmp_path / "trials.json"
        trial_path.write_text(json.dumps(SAMPLE_TRIALS))

        as_of = date(2024, 6, 30)
        backfill_ctgov_history(
            date_from=as_of,
            date_to=as_of,
            trial_records_path=trial_path,
            cadence="quarterly",
            out_root=tmp_path / "cache",
        )

        cache_path = tmp_path / "cache" / f"trial_records_{as_of.isoformat()}.json"
        with open(cache_path) as f:
            cached = json.load(f)

        for t in cached:
            fp = t.get("first_posted", "")
            if fp:
                assert fp <= as_of.isoformat(), (
                    f"Future trial leaked: {t['nct_id']} first_posted={fp}"
                )
