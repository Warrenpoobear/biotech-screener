"""Tests for warm_caches.py — CTGov PIT-filtered cache."""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from warm_caches import warm_ctgov


def _make_records(lup_dates: list[str]) -> list[dict]:
    """Build minimal trial records with given last_update_posted dates."""
    return [
        {"nct_id": f"NCT{i:08d}", "last_update_posted": d}
        for i, d in enumerate(lup_dates)
    ]


class TestWarmCtgov:
    def test_filters_by_last_update_posted(self, tmp_path):
        """Records with lup > as_of_date are excluded."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        records = _make_records(["2026-01-25", "2026-01-30", "2026-02-05"])
        (data_dir / "trial_records.json").write_text(json.dumps(records))

        cache_dir = tmp_path / "cache"
        count = warm_ctgov(date(2026, 1, 28), data_dir, cache_dir)

        assert count == 1  # only 2026-01-25
        cached = json.loads((cache_dir / "trial_records_2026-01-28.json").read_text())
        assert len(cached) == 1
        assert cached[0]["last_update_posted"] == "2026-01-25"

    def test_keeps_records_at_boundary(self, tmp_path):
        """Records with lup == as_of_date are included (inclusive)."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        records = _make_records(["2026-01-28", "2026-01-29"])
        (data_dir / "trial_records.json").write_text(json.dumps(records))

        cache_dir = tmp_path / "cache"
        count = warm_ctgov(date(2026, 1, 28), data_dir, cache_dir)

        assert count == 1
        cached = json.loads((cache_dir / "trial_records_2026-01-28.json").read_text())
        assert cached[0]["last_update_posted"] == "2026-01-28"

    def test_handles_missing_lup(self, tmp_path):
        """Records without last_update_posted are included (not provably future)."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        records = [
            {"nct_id": "NCT001", "last_update_posted": "2026-01-20"},
            {"nct_id": "NCT002"},  # no lup field → included
            {"nct_id": "NCT003", "last_update_posted": None},  # null → included
            {"nct_id": "NCT004", "last_update_posted": ""},  # empty → included
            {"nct_id": "NCT005", "last_update_posted": "2026-02-05"},  # future → excluded
        ]
        (data_dir / "trial_records.json").write_text(json.dumps(records))

        cache_dir = tmp_path / "cache"
        count = warm_ctgov(date(2026, 2, 1), data_dir, cache_dir)

        # NCT001 (valid lup), NCT002-004 (missing/empty lup), but NOT NCT005 (future)
        assert count == 4

    def test_cache_file_written(self, tmp_path):
        """Output file appears at expected path with correct count."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        records = _make_records(["2026-01-10", "2026-01-15", "2026-01-20"])
        (data_dir / "trial_records.json").write_text(json.dumps(records))

        cache_dir = tmp_path / "cache"
        count = warm_ctgov(date(2026, 1, 20), data_dir, cache_dir)

        target = cache_dir / "trial_records_2026-01-20.json"
        assert target.exists()
        assert count == 3
        assert len(json.loads(target.read_text())) == 3

    def test_existing_cache_skipped(self, tmp_path):
        """Second call returns count without rewriting."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        records = _make_records(["2026-01-10"])
        (data_dir / "trial_records.json").write_text(json.dumps(records))

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        target = cache_dir / "trial_records_2026-01-15.json"
        target.write_text(json.dumps(records))

        # Should read existing, not rewrite
        count = warm_ctgov(date(2026, 1, 15), data_dir, cache_dir)
        assert count == 1

        # Verify source file was NOT re-read (by deleting it — should still work)
        (data_dir / "trial_records.json").unlink()
        count2 = warm_ctgov(date(2026, 1, 15), data_dir, cache_dir)
        assert count2 == 1

    def test_missing_source_file_errors(self, tmp_path):
        """trial_records.json missing → FileNotFoundError."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        # No trial_records.json
        cache_dir = tmp_path / "cache"

        with pytest.raises(FileNotFoundError, match="trial_records.json not found"):
            warm_ctgov(date(2026, 1, 15), data_dir, cache_dir)

    def test_preserves_all_record_fields(self, tmp_path):
        """Filtered records retain all original fields."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        records = [
            {
                "nct_id": "NCT001",
                "last_update_posted": "2026-01-10",
                "brief_title": "Test Study",
                "phase": "Phase 2",
                "extra_field": [1, 2, 3],
            }
        ]
        (data_dir / "trial_records.json").write_text(json.dumps(records))

        cache_dir = tmp_path / "cache"
        warm_ctgov(date(2026, 1, 15), data_dir, cache_dir)

        cached = json.loads((cache_dir / "trial_records_2026-01-15.json").read_text())
        assert cached[0] == records[0]
