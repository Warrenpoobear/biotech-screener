"""Tests for scripts/rollup_shadow_metrics.py."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.rollup_shadow_metrics import collect_shadow_metrics, write_csv, main, COLUMNS


def _write_shadow(snap_dir: Path, date_str: str, data: dict) -> None:
    d = snap_dir / date_str
    d.mkdir(parents=True, exist_ok=True)
    (d / "catalyst_shadow_metrics.json").write_text(
        json.dumps(data), encoding="utf-8",
    )


def _sample_metrics(as_of_date: str, **overrides) -> dict:
    base = {
        "as_of_date": as_of_date,
        "prior_date": None,
        "bad_to_good_count": 3,
        "good_to_bad_count": 1,
        "median_catalyst_days_good": 45.0,
        "p10_catalyst_days": 10.0,
        "p50_catalyst_days": 45.0,
        "p90_catalyst_days": 150.0,
        "A_tier_count": 37,
        "top60_overlap": 0.875,
        "top100_overlap": 0.96,
        "sec_8k_events": 206,
        "ctgov_events": 1084,
        "fda_events": 11,
    }
    base.update(overrides)
    return base


class TestCollectShadowMetrics:

    def test_collects_sorted_by_date(self, tmp_path):
        snap = tmp_path / "snapshots"
        _write_shadow(snap, "2026-02-14", _sample_metrics("2026-02-14"))
        _write_shadow(snap, "2026-02-11", _sample_metrics("2026-02-11"))

        rows = collect_shadow_metrics(snap)
        assert len(rows) == 2
        assert rows[0]["as_of_date"] == "2026-02-11"
        assert rows[1]["as_of_date"] == "2026-02-14"

    def test_skips_dirs_without_shadow_json(self, tmp_path):
        snap = tmp_path / "snapshots"
        _write_shadow(snap, "2026-02-14", _sample_metrics("2026-02-14"))
        (snap / "2026-02-11").mkdir(parents=True)  # no JSON

        rows = collect_shadow_metrics(snap)
        assert len(rows) == 1

    def test_skips_malformed_json(self, tmp_path):
        snap = tmp_path / "snapshots"
        d = snap / "2026-02-14"
        d.mkdir(parents=True)
        (d / "catalyst_shadow_metrics.json").write_text("{bad json", encoding="utf-8")

        rows = collect_shadow_metrics(snap)
        assert len(rows) == 0

    def test_returns_empty_for_missing_dir(self, tmp_path):
        rows = collect_shadow_metrics(tmp_path / "nonexistent")
        assert rows == []

    def test_skips_non_date_dirs(self, tmp_path):
        snap = tmp_path / "snapshots"
        _write_shadow(snap, "2026-02-14", _sample_metrics("2026-02-14"))
        _write_shadow(snap, "not-a-date", _sample_metrics("not-a-date"))

        rows = collect_shadow_metrics(snap)
        assert len(rows) == 1
        assert rows[0]["as_of_date"] == "2026-02-14"

    def test_fallback_date_from_dirname(self, tmp_path):
        snap = tmp_path / "snapshots"
        data = _sample_metrics("2026-02-14")
        del data["as_of_date"]
        _write_shadow(snap, "2026-02-14", data)

        rows = collect_shadow_metrics(snap)
        assert rows[0]["as_of_date"] == "2026-02-14"

    def test_from_date_filters_earlier(self, tmp_path):
        snap = tmp_path / "snapshots"
        _write_shadow(snap, "2025-06-30", _sample_metrics("2025-06-30"))
        _write_shadow(snap, "2026-01-15", _sample_metrics("2026-01-15"))
        _write_shadow(snap, "2026-02-14", _sample_metrics("2026-02-14"))

        rows = collect_shadow_metrics(snap, from_date="2026-01-01")
        assert len(rows) == 2
        assert rows[0]["as_of_date"] == "2026-01-15"

    def test_to_date_filters_later(self, tmp_path):
        snap = tmp_path / "snapshots"
        _write_shadow(snap, "2026-01-15", _sample_metrics("2026-01-15"))
        _write_shadow(snap, "2026-02-14", _sample_metrics("2026-02-14"))
        _write_shadow(snap, "2026-03-01", _sample_metrics("2026-03-01"))

        rows = collect_shadow_metrics(snap, to_date="2026-02-28")
        assert len(rows) == 2
        assert rows[-1]["as_of_date"] == "2026-02-14"

    def test_from_and_to_date_combined(self, tmp_path):
        snap = tmp_path / "snapshots"
        _write_shadow(snap, "2025-12-31", _sample_metrics("2025-12-31"))
        _write_shadow(snap, "2026-01-15", _sample_metrics("2026-01-15"))
        _write_shadow(snap, "2026-02-14", _sample_metrics("2026-02-14"))
        _write_shadow(snap, "2026-03-01", _sample_metrics("2026-03-01"))

        rows = collect_shadow_metrics(snap, from_date="2026-01-01", to_date="2026-02-28")
        assert len(rows) == 2
        dates = [r["as_of_date"] for r in rows]
        assert dates == ["2026-01-15", "2026-02-14"]

    def test_from_to_date_inclusive_boundaries(self, tmp_path):
        snap = tmp_path / "snapshots"
        _write_shadow(snap, "2026-02-10", _sample_metrics("2026-02-10"))
        _write_shadow(snap, "2026-02-11", _sample_metrics("2026-02-11"))
        _write_shadow(snap, "2026-02-12", _sample_metrics("2026-02-12"))

        rows = collect_shadow_metrics(snap, from_date="2026-02-10", to_date="2026-02-12")
        assert len(rows) == 3


class TestWriteCsv:

    def test_writes_valid_csv(self, tmp_path):
        out = tmp_path / "out.csv"
        rows = [_sample_metrics("2026-02-14")]
        write_csv(rows, out)

        assert out.exists()
        with open(out, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == COLUMNS
            loaded = list(reader)
        assert len(loaded) == 1
        assert loaded[0]["as_of_date"] == "2026-02-14"
        assert loaded[0]["sec_8k_events"] == "206"

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "nested" / "dir" / "out.csv"
        write_csv([_sample_metrics("2026-02-14")], out)
        assert out.exists()

    def test_none_values_written_as_empty(self, tmp_path):
        out = tmp_path / "out.csv"
        data = _sample_metrics("2026-02-14", prior_date=None, top60_overlap=None)
        write_csv([data], out)

        with open(out, encoding="utf-8") as f:
            loaded = list(csv.DictReader(f))
        assert loaded[0]["prior_date"] == ""
        assert loaded[0]["top60_overlap"] == ""


class TestMain:

    def test_main_writes_csv(self, tmp_path):
        snap = tmp_path / "snapshots"
        _write_shadow(snap, "2026-02-14", _sample_metrics("2026-02-14"))
        out = tmp_path / "out.csv"

        rc = main(["--snapshot-dir", str(snap), "--output", str(out)])
        assert rc == 0
        assert out.exists()

    def test_main_no_data_returns_1(self, tmp_path):
        rc = main(["--snapshot-dir", str(tmp_path / "empty"), "--output", str(tmp_path / "out.csv")])
        assert rc == 1

    def test_main_multiple_dates(self, tmp_path):
        snap = tmp_path / "snapshots"
        _write_shadow(snap, "2026-02-11", _sample_metrics("2026-02-11", A_tier_count=33))
        _write_shadow(snap, "2026-02-14", _sample_metrics("2026-02-14", A_tier_count=37))
        out = tmp_path / "out.csv"

        rc = main(["--snapshot-dir", str(snap), "--output", str(out)])
        assert rc == 0

        with open(out, encoding="utf-8") as f:
            loaded = list(csv.DictReader(f))
        assert len(loaded) == 2
        assert loaded[0]["A_tier_count"] == "33"
        assert loaded[1]["A_tier_count"] == "37"

    def test_main_from_to_date_filters(self, tmp_path):
        snap = tmp_path / "snapshots"
        _write_shadow(snap, "2025-06-30", _sample_metrics("2025-06-30"))
        _write_shadow(snap, "2026-01-15", _sample_metrics("2026-01-15"))
        _write_shadow(snap, "2026-02-14", _sample_metrics("2026-02-14"))
        _write_shadow(snap, "2026-03-01", _sample_metrics("2026-03-01"))
        out = tmp_path / "out.csv"

        rc = main([
            "--snapshot-dir", str(snap),
            "--output", str(out),
            "--from-date", "2026-01-01",
            "--to-date", "2026-02-28",
        ])
        assert rc == 0

        with open(out, encoding="utf-8") as f:
            loaded = list(csv.DictReader(f))
        assert len(loaded) == 2
        dates = [r["as_of_date"] for r in loaded]
        assert dates == ["2026-01-15", "2026-02-14"]
