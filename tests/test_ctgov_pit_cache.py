"""Integration tests: PIT-filtered CTGov cache → Module 3 PIT check passes."""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from warm_caches import warm_ctgov
from catalyst_diagnostics import check_trial_records_staleness


def _make_trial_records(lup_dates: list[str]) -> list[dict]:
    """Minimal but schema-valid trial records."""
    return [
        {
            "nct_id": f"NCT{i:08d}",
            "ticker": f"TICK{i}",
            "status": "Recruiting",
            "last_update_posted": d,
            "brief_title": f"Study {i}",
        }
        for i, d in enumerate(lup_dates)
    ]


class TestCachedRecordsPassPITCheck:
    """Filtered cache produces data where max(lup) <= as_of_date, so
    Module 3's staleness/PIT check does NOT fire."""

    def test_cached_records_pass_pit_check(self, tmp_path):
        """Module 3 staleness check with filtered file → no PIT violation."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        # Simulate production: most records old, a few newer than as_of_date
        lup_dates = ["2026-01-10", "2026-01-20", "2026-01-28", "2026-02-04"]
        records = _make_trial_records(lup_dates)
        (data_dir / "trial_records.json").write_text(json.dumps(records))

        cache_dir = tmp_path / "cache"
        as_of = date(2026, 1, 28)
        warm_ctgov(as_of, data_dir, cache_dir)

        # Load cached and verify PIT check passes
        cached = json.loads(
            (cache_dir / f"trial_records_{as_of.isoformat()}.json").read_text()
        )
        result = check_trial_records_staleness(cached, as_of)
        # trial_records_date should be <= as_of → NOT stale in forward-looking sense
        assert result.trial_records_date is not None
        assert result.trial_records_date <= as_of
        # age_days >= 0 means no PIT violation (negative age = future data)
        assert result.age_days >= 0

    def test_no_cache_falls_back_to_production(self, tmp_path):
        """No cache file → run_screen would use data_dir/trial_records.json."""
        # This tests the fallback logic conceptually: if no cache exists,
        # the original path is used (and may trigger PIT violation).
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        records = _make_trial_records(["2026-02-04"])
        (data_dir / "trial_records.json").write_text(json.dumps(records))

        cache_dir = tmp_path / "cache"
        cache_path = cache_dir / "trial_records_2026-01-28.json"

        # No cache exists
        assert not cache_path.exists()

        # Direct PIT check on unfiltered data would detect forward data
        result = check_trial_records_staleness(records, date(2026, 1, 28))
        assert result.trial_records_date == date(2026, 2, 4)
        assert result.age_days < 0  # future data → PIT violation

    def test_all_records_within_cutoff(self, tmp_path):
        """Every record in cache has lup <= as_of_date."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        lup_dates = [
            "2026-01-01", "2026-01-15", "2026-01-28",
            "2026-02-01", "2026-02-04",
        ]
        records = _make_trial_records(lup_dates)
        (data_dir / "trial_records.json").write_text(json.dumps(records))

        cache_dir = tmp_path / "cache"
        as_of = date(2026, 1, 28)
        warm_ctgov(as_of, data_dir, cache_dir)

        cached = json.loads(
            (cache_dir / f"trial_records_{as_of.isoformat()}.json").read_text()
        )
        for rec in cached:
            assert rec["last_update_posted"][:10] <= as_of.isoformat()

    def test_determinism(self, tmp_path):
        """Same inputs → identical output (PIT regression guard)."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        lup_dates = ["2026-01-10", "2026-01-20", "2026-02-04"]
        records = _make_trial_records(lup_dates)
        (data_dir / "trial_records.json").write_text(json.dumps(records))

        cache_dir_1 = tmp_path / "cache1"
        cache_dir_2 = tmp_path / "cache2"
        as_of = date(2026, 1, 28)

        warm_ctgov(as_of, data_dir, cache_dir_1)
        warm_ctgov(as_of, data_dir, cache_dir_2)

        out1 = (cache_dir_1 / f"trial_records_{as_of.isoformat()}.json").read_text()
        out2 = (cache_dir_2 / f"trial_records_{as_of.isoformat()}.json").read_text()
        assert out1 == out2
