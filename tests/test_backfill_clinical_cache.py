"""Tests for scripts/backfill_clinical_cache.py"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure project root on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.backfill_clinical_cache import (
    _business_days,
    _pit_filter_trials,
    _serialize,
    backfill_one_date,
    main,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _fake_m4_result(trial_records=None, active_tickers=None, as_of_date=None):
    """Deterministic stub for compute_module_4_clinical_dev."""
    tickers = active_tickers or []
    trials = trial_records or []
    scores = [
        {
            "ticker": t,
            "clinical_score": "50.00",
            "phase_score": "15.00",
            "phase_progress": "2.50",
            "trial_count_bonus": "1.00",
            "diversity_bonus": "1.00",
            "recency_bonus": "1.00",
            "design_score": "10.00",
            "execution_score": "10.00",
            "endpoint_score": "9.50",
            "lead_phase": "phase 2",
            "trial_count": 1,
            "flags": [],
            "severity": "none",
            "n_trials_unique": 1,
            "n_trials_raw": 1,
            "pit_filtered_count_ticker": 0,
            "lead_trial_nct_id": f"NCT0000000{i}",
            "recency_days": 10,
            "lead_program_key": f"key_{t}",
            "completion_rate": "0.50",
            "termination_rate": "0.10",
            "status_quality_score": "0.80",
            "n_strong_endpoints": 1,
            "n_weak_endpoints": 0,
            "n_neutral_endpoints": 0,
            "phase_counts": {"phase_2": 1},
        }
        for i, t in enumerate(
            tickers if isinstance(tickers, list) else sorted(tickers)
        )
    ]
    return {
        "as_of_date": as_of_date,
        "scores": scores,
        "diagnostic_counts": {
            "scored": len(scores),
            "total_trials_raw": len(trials),
            "total_trials_unique": max(len(trials) - 1, 0),
            "total_trials": max(len(trials) - 1, 0),
            "pit_filtered": 0,
            "pit_fields_used": {},
        },
        "provenance": {"ruleset_version": "test", "inputs_hash": "abc", "pit_cutoff": as_of_date},
    }


@pytest.fixture()
def fake_env(tmp_path):
    """Set up minimal data_dir and cache_dir with synthetic data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # universe.json — 3 tickers
    universe = [
        {"ticker": "AAA", "name": "AAA Inc"},
        {"ticker": "BBB", "name": "BBB Corp"},
        {"ticker": "CCC", "name": "CCC Ltd"},
    ]
    (data_dir / "universe.json").write_text(json.dumps(universe))

    # trial_records.json — 4 trials, 1 has future last_update_posted
    trials = [
        {"ticker": "AAA", "nct_id": "NCT001", "last_update_posted": "2026-01-10", "phase": "PHASE2", "status": "RECRUITING"},
        {"ticker": "BBB", "nct_id": "NCT002", "last_update_posted": "2026-01-14", "phase": "PHASE3", "status": "COMPLETED"},
        {"ticker": "CCC", "nct_id": "NCT003", "last_update_posted": "2026-01-20", "phase": "PHASE1", "status": "RECRUITING"},
        {"ticker": "AAA", "nct_id": "NCT004", "last_update_posted": "2099-12-31", "phase": "PHASE2", "status": "NOT_YET_RECRUITING"},
    ]
    (data_dir / "trial_records.json").write_text(json.dumps(trials))

    return data_dir, cache_dir, universe, trials


PATCH_TARGET = "scripts.backfill_clinical_cache.compute_module_4_clinical_dev"


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestBusinessDays:
    def test_weekdays_only(self):
        days = list(_business_days(date(2026, 1, 12), date(2026, 1, 18)))
        # Mon 12, Tue 13, Wed 14, Thu 15, Fri 16  (skip Sat 17, Sun 18)
        assert days == [date(2026, 1, 12), date(2026, 1, 13), date(2026, 1, 14),
                        date(2026, 1, 15), date(2026, 1, 16)]

    def test_single_weekend(self):
        days = list(_business_days(date(2026, 1, 17), date(2026, 1, 18)))
        assert days == []

    def test_single_weekday(self):
        days = list(_business_days(date(2026, 1, 15), date(2026, 1, 15)))
        assert days == [date(2026, 1, 15)]


class TestPitFilter:
    def test_removes_future_trials(self):
        trials = [
            {"last_update_posted": "2026-01-10"},
            {"last_update_posted": "2026-01-16"},
            {"last_update_posted": "2026-01-15"},
        ]
        kept = _pit_filter_trials(trials, "2026-01-15")
        assert len(kept) == 2
        assert all(t["last_update_posted"] <= "2026-01-15" for t in kept)

    def test_keeps_trials_with_no_date(self):
        trials = [{"last_update_posted": None}, {"ticker": "X"}]
        kept = _pit_filter_trials(trials, "2026-01-15")
        assert len(kept) == 2


class TestSerialize:
    def test_deterministic(self):
        obj = {"b": 2, "a": 1}
        assert _serialize(obj) == '{"a":1,"b":2}'


# ---------------------------------------------------------------------------
# Integration-ish tests (monkeypatched Module 4)
# ---------------------------------------------------------------------------

class TestBackfillOneDate:
    def test_writes_cache_file(self, fake_env):
        data_dir, cache_dir, _, _ = fake_env
        with patch(PATCH_TARGET, side_effect=_fake_m4_result):
            result = backfill_one_date(date(2026, 1, 15), data_dir, cache_dir)
        assert result is not None
        assert result["tickers"] == 3
        cached = cache_dir / "clinical" / "clinical_features_2026-01-15.json"
        assert cached.exists()

    def test_output_schema(self, fake_env):
        data_dir, cache_dir, _, _ = fake_env
        with patch(PATCH_TARGET, side_effect=_fake_m4_result):
            backfill_one_date(date(2026, 1, 15), data_dir, cache_dir)
        cached = json.loads(
            (cache_dir / "clinical" / "clinical_features_2026-01-15.json").read_text()
        )
        assert "as_of_date" in cached
        assert "provenance" in cached
        assert "diagnostic_counts" in cached
        assert "scores" in cached
        assert cached["as_of_date"] == "2026-01-15"
        assert "trial_records_source" in cached["provenance"]
        assert "pit_filter" in cached["provenance"]
        assert cached["provenance"]["pit_filter"]["inclusive_cutoff"] == "2026-01-15"

    def test_idempotency_skips(self, fake_env):
        data_dir, cache_dir, _, _ = fake_env
        with patch(PATCH_TARGET, side_effect=_fake_m4_result) as mock_m4:
            backfill_one_date(date(2026, 1, 15), data_dir, cache_dir)
            result2 = backfill_one_date(date(2026, 1, 15), data_dir, cache_dir)
        assert result2 is None  # skipped
        assert mock_m4.call_count == 1  # only called once

    def test_no_skip_existing_overwrites(self, fake_env):
        data_dir, cache_dir, _, _ = fake_env
        with patch(PATCH_TARGET, side_effect=_fake_m4_result) as mock_m4:
            backfill_one_date(date(2026, 1, 15), data_dir, cache_dir)
            result2 = backfill_one_date(date(2026, 1, 15), data_dir, cache_dir, skip_existing=False)
        assert result2 is not None
        assert mock_m4.call_count == 2

    def test_prefers_ctgov_cache(self, fake_env):
        data_dir, cache_dir, _, _ = fake_env
        # Create a ctgov cache file for 2026-01-15
        ctgov_dir = cache_dir / "ctgov"
        ctgov_dir.mkdir(parents=True, exist_ok=True)
        ctgov_trials = [
            {"ticker": "AAA", "nct_id": "NCT001", "last_update_posted": "2026-01-10",
             "phase": "PHASE2", "status": "RECRUITING"},
        ]
        (ctgov_dir / "trial_records_2026-01-15.json").write_text(json.dumps(ctgov_trials))

        with patch(PATCH_TARGET, side_effect=_fake_m4_result):
            backfill_one_date(date(2026, 1, 15), data_dir, cache_dir)
        cached = json.loads(
            (cache_dir / "clinical" / "clinical_features_2026-01-15.json").read_text()
        )
        assert "ctgov" in cached["provenance"]["trial_records_source"]

    def test_fallback_applies_pit_filter(self, fake_env):
        """When no ctgov cache exists, inline PIT filter excludes future trials."""
        data_dir, cache_dir, _, _ = fake_env
        calls = []

        def capture_m4(trial_records=None, active_tickers=None, as_of_date=None):
            calls.append({"n_trials": len(trial_records or [])})
            return _fake_m4_result(trial_records=trial_records, active_tickers=active_tickers, as_of_date=as_of_date)

        with patch(PATCH_TARGET, side_effect=capture_m4):
            backfill_one_date(date(2026, 1, 15), data_dir, cache_dir)

        # Original has 4 trials; NCT004 (2099-12-31) should be filtered out
        # NCT003 (2026-01-20) > 2026-01-15, also filtered
        assert calls[0]["n_trials"] == 2  # AAA(NCT001) + BBB(NCT002)


class TestMainCLI:
    def test_business_days_only_no_weekend_files(self, fake_env):
        """Weekends should produce no cache files."""
        data_dir, cache_dir, _, _ = fake_env
        with patch(PATCH_TARGET, side_effect=_fake_m4_result):
            # Sat Jan 17 to Sun Jan 18 — 0 business days
            main(["--from-date", "2026-01-17", "--to-date", "2026-01-18",
                   "--data-dir", str(data_dir), "--cache-dir", str(cache_dir)])
        clinical_dir = cache_dir / "clinical"
        if clinical_dir.exists():
            assert list(clinical_dir.glob("*.json")) == []

    def test_two_weekdays(self, fake_env):
        data_dir, cache_dir, _, _ = fake_env
        with patch(PATCH_TARGET, side_effect=_fake_m4_result):
            main(["--from-date", "2026-01-15", "--to-date", "2026-01-16",
                   "--data-dir", str(data_dir), "--cache-dir", str(cache_dir)])
        files = sorted((cache_dir / "clinical").glob("*.json"))
        assert len(files) == 2
        assert files[0].name == "clinical_features_2026-01-15.json"
        assert files[1].name == "clinical_features_2026-01-16.json"
