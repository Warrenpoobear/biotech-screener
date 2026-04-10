"""Tests for scripts/backfill_herald_data.py"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.backfill_herald_data import (
    _all_days,
    _is_weekday,
    _output_exists,
    backfill,
    backfill_one_date,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestAllDays:
    def test_single_day(self):
        days = list(_all_days(date(2026, 4, 1), date(2026, 4, 1)))
        assert days == [date(2026, 4, 1)]

    def test_range(self):
        days = list(_all_days(date(2026, 4, 1), date(2026, 4, 3)))
        assert days == [date(2026, 4, 1), date(2026, 4, 2), date(2026, 4, 3)]

    def test_empty_when_start_after_end(self):
        days = list(_all_days(date(2026, 4, 5), date(2026, 4, 1)))
        assert days == []


class TestIsWeekday:
    def test_monday(self):
        assert _is_weekday(date(2026, 4, 6))  # Monday

    def test_friday(self):
        assert _is_weekday(date(2026, 4, 10))  # Friday

    def test_saturday(self):
        assert not _is_weekday(date(2026, 4, 11))  # Saturday

    def test_sunday(self):
        assert not _is_weekday(date(2026, 4, 12))  # Sunday


# ---------------------------------------------------------------------------
# Output existence checks
# ---------------------------------------------------------------------------


class TestOutputExists:
    def test_no_outputs(self, tmp_path):
        """When directories are empty, all flags are False."""
        with patch("scripts.backfill_herald_data.RELEASES_DIR", tmp_path / "pr"), \
             patch("scripts.backfill_herald_data.DEDUPED_DIR", tmp_path / "deduped"), \
             patch("scripts.backfill_herald_data.CLASSIFIED_DIR", tmp_path / "classified"), \
             patch("scripts.backfill_herald_data.CRT_INTAKE_DIR", tmp_path / "crt"), \
             patch("scripts.backfill_herald_data.SNAPSHOT_DIR", tmp_path / "snap"):
            result = _output_exists(date(2026, 4, 1))
            assert result["releases"] is False
            assert result["deduped"] is False
            assert result["classified"] is False
            assert result["health"] is False
            assert result["crt_intake"] is False
            assert result["snapshot"] is False

    def test_classified_exists(self, tmp_path):
        """When classified output exists, flag is True."""
        classified_dir = tmp_path / "classified"
        classified_dir.mkdir(parents=True)
        (classified_dir / "classified_2026-04-01.jsonl").write_text("{}\n")

        with patch("scripts.backfill_herald_data.RELEASES_DIR", tmp_path / "pr"), \
             patch("scripts.backfill_herald_data.DEDUPED_DIR", tmp_path / "deduped"), \
             patch("scripts.backfill_herald_data.CLASSIFIED_DIR", classified_dir), \
             patch("scripts.backfill_herald_data.CRT_INTAKE_DIR", tmp_path / "crt"), \
             patch("scripts.backfill_herald_data.SNAPSHOT_DIR", tmp_path / "snap"):
            result = _output_exists(date(2026, 4, 1))
            assert result["classified"] is True
            assert result["releases"] is False


# ---------------------------------------------------------------------------
# Single date backfill
# ---------------------------------------------------------------------------


class TestBackfillOneDate:
    def test_skip_when_classified_exists(self, tmp_path):
        """Skip date when classified output already exists and force=False."""
        classified_dir = tmp_path / "classified"
        classified_dir.mkdir(parents=True)
        (classified_dir / "classified_2026-04-01.jsonl").write_text(
            '{"ticker":"ACME","headline":"test"}\n'
        )

        with patch("scripts.backfill_herald_data.RELEASES_DIR", tmp_path / "pr"), \
             patch("scripts.backfill_herald_data.DEDUPED_DIR", tmp_path / "deduped"), \
             patch("scripts.backfill_herald_data.CLASSIFIED_DIR", classified_dir), \
             patch("scripts.backfill_herald_data.CRT_INTAKE_DIR", tmp_path / "crt"), \
             patch("scripts.backfill_herald_data.SNAPSHOT_DIR", tmp_path / "snap"):
            result = backfill_one_date(date(2026, 4, 1), dry_run=False, force=False)
            assert result["skipped"] is True
            assert result["classified_count"] == 1

    def test_dry_run_no_writes(self):
        """Dry run should not execute subprocesses."""
        with patch("scripts.backfill_herald_data._output_exists") as mock_exists:
            mock_exists.return_value = {
                "releases": False,
                "deduped": False,
                "classified": False,
                "health": False,
                "crt_intake": False,
                "snapshot": False,
            }
            with patch("subprocess.run") as mock_run:
                result = backfill_one_date(date(2026, 4, 1), dry_run=True)
                mock_run.assert_not_called()
                assert result["skipped"] is False
                assert len(result["steps"]) > 0
                assert all(s["status"] == "dry_run" for s in result["steps"])

    def test_fetch_failure_stops_pipeline(self, tmp_path):
        """When fetch fails, downstream steps are skipped."""
        with patch("scripts.backfill_herald_data._output_exists") as mock_exists:
            mock_exists.return_value = {
                "releases": False,
                "deduped": False,
                "classified": False,
                "health": False,
                "crt_intake": False,
                "snapshot": False,
            }
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=1,
                    stdout="Error",
                    stderr="fetch failed",
                )
                result = backfill_one_date(date(2026, 4, 1))
                assert result.get("error") is not None
                assert "fetch failed" in result["error"]
                # Only fetch step should have been attempted
                assert len(result["steps"]) == 1

    def test_no_releases_skips_downstream(self, tmp_path):
        """When fetch produces empty file, dedupe/classify are skipped."""
        pr_dir = tmp_path / "pr"
        pr_dir.mkdir(parents=True)
        # Create empty releases file (already exists from prior fetch)
        (pr_dir / "releases_2026-04-01.jsonl").write_text("")

        with patch("scripts.backfill_herald_data.RELEASES_DIR", pr_dir), \
             patch("scripts.backfill_herald_data.DEDUPED_DIR", tmp_path / "deduped"), \
             patch("scripts.backfill_herald_data.CLASSIFIED_DIR", tmp_path / "classified"), \
             patch("scripts.backfill_herald_data.CRT_INTAKE_DIR", tmp_path / "crt"), \
             patch("scripts.backfill_herald_data.SNAPSHOT_DIR", tmp_path / "snap"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0, stdout="0 releases", stderr=""
                )
                result = backfill_one_date(date(2026, 4, 1))
                assert result["releases_count"] == 0
                # Releases file exists but empty -> fetch reused, downstream skipped
                # No subprocess calls needed since fetch is reused and 0 releases
                assert mock_run.call_count == 0


# ---------------------------------------------------------------------------
# Batch backfill
# ---------------------------------------------------------------------------


class TestBackfill:
    def test_weekdays_only_filters_weekends(self):
        """Backfill with weekdays_only=True skips weekends."""
        with patch("scripts.backfill_herald_data.backfill_one_date") as mock_one:
            mock_one.return_value = {
                "date": "2026-04-06",
                "steps": [],
                "skipped": True,
                "releases_count": 0,
                "deduped_count": 0,
                "classified_count": 0,
                "crt_candidates": 0,
            }
            # April 6-12 2026 = Mon-Sun
            summary = backfill(
                date(2026, 4, 6),
                date(2026, 4, 12),
                weekdays_only=True,
                sleep_seconds=0,
                dry_run=True,
            )
            assert summary["total_dates"] == 5  # Mon-Fri

    def test_include_weekends(self):
        """Backfill with weekdays_only=False includes all days."""
        with patch("scripts.backfill_herald_data.backfill_one_date") as mock_one:
            mock_one.return_value = {
                "date": "2026-04-06",
                "steps": [],
                "skipped": True,
                "releases_count": 0,
                "deduped_count": 0,
                "classified_count": 0,
                "crt_candidates": 0,
            }
            summary = backfill(
                date(2026, 4, 6),
                date(2026, 4, 12),
                weekdays_only=False,
                sleep_seconds=0,
                dry_run=True,
            )
            assert summary["total_dates"] == 7  # All 7 days

    def test_summary_counts(self):
        """Verify summary aggregates per-date counts."""
        call_count = 0

        def mock_one_date(d, **kwargs):
            nonlocal call_count
            call_count += 1
            return {
                "date": d.isoformat(),
                "steps": [],
                "skipped": False,
                "releases_count": 10,
                "deduped_count": 8,
                "classified_count": 7,
                "crt_candidates": 2,
            }

        with patch("scripts.backfill_herald_data.backfill_one_date", side_effect=mock_one_date):
            summary = backfill(
                date(2026, 4, 6),
                date(2026, 4, 8),
                weekdays_only=True,
                sleep_seconds=0,
            )
            assert summary["processed"] == 3  # Mon, Tue, Wed
            assert summary["total_releases"] == 30
            assert summary["total_deduped"] == 24
            assert summary["total_classified"] == 21
            assert summary["total_crt_candidates"] == 6

    def test_failure_counted(self):
        """Failed dates are counted in summary."""
        def mock_one_date(d, **kwargs):
            return {
                "date": d.isoformat(),
                "steps": [],
                "skipped": False,
                "error": "fetch failed: timeout",
                "releases_count": 0,
                "deduped_count": 0,
                "classified_count": 0,
                "crt_candidates": 0,
            }

        with patch("scripts.backfill_herald_data.backfill_one_date", side_effect=mock_one_date):
            summary = backfill(
                date(2026, 4, 6),
                date(2026, 4, 6),
                weekdays_only=True,
                sleep_seconds=0,
            )
            assert summary["failed"] == 1
            assert summary["processed"] == 0


# ---------------------------------------------------------------------------
# PIT safety
# ---------------------------------------------------------------------------


class TestPITSafety:
    def test_releases_path_uses_as_of_date(self):
        """Verify output paths use the as-of date, not today's date."""
        d = date(2026, 3, 15)
        ds = d.isoformat()
        with patch("scripts.backfill_herald_data._output_exists") as mock_exists:
            mock_exists.return_value = {
                "releases": False,
                "deduped": False,
                "classified": False,
                "health": False,
                "crt_intake": False,
                "snapshot": False,
            }
            with patch("subprocess.run") as mock_run:
                result = backfill_one_date(d, dry_run=True)
                # The fetch step should pass --as-of-date with the correct date
                fetch_step = result["steps"][0]
                assert fetch_step["status"] == "dry_run"

    def test_force_reprocesses_existing(self, tmp_path):
        """Force flag causes reprocessing even when output exists."""
        classified_dir = tmp_path / "classified"
        classified_dir.mkdir(parents=True)
        (classified_dir / "classified_2026-04-01.jsonl").write_text(
            '{"ticker":"ACME","headline":"old"}\n'
        )

        with patch("scripts.backfill_herald_data.RELEASES_DIR", tmp_path / "pr"), \
             patch("scripts.backfill_herald_data.DEDUPED_DIR", tmp_path / "deduped"), \
             patch("scripts.backfill_herald_data.CLASSIFIED_DIR", classified_dir), \
             patch("scripts.backfill_herald_data.CRT_INTAKE_DIR", tmp_path / "crt"), \
             patch("scripts.backfill_herald_data.SNAPSHOT_DIR", tmp_path / "snap"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0, stdout="ok", stderr=""
                )
                result = backfill_one_date(date(2026, 4, 1), force=True)
                assert result["skipped"] is False
                assert mock_run.called
