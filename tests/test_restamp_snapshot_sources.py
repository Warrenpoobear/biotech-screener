"""Tests for tools/restamp_snapshot_sources.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.restamp_snapshot_sources import (
    _discover_13f_dates,
    _discover_ctgov_dates,
    _find_latest_cache_date,
    _detect_active_families,
    reconstruct_data_sources,
    restamp_batch,
)

# ── Fixtures ──────────────────────────────────────────────────────────────

# CSV header that contains columns from all three families
_FULL_HEADER = (
    "ticker,composite_score,coinvest_score_z,sponsor_tier1_count,"
    "clinical_score,clinical_score_z,clinical_readout_days,"
    "catalyst_days,catalyst_decay_w,catalyst_source\n"
)

# CSV header with only sec_8k columns (no 13f or ctgov)
_MINIMAL_HEADER = "ticker,composite_score,catalyst_days,catalyst_decay_w\n"


def _write_rankings(snap_dir: Path, header: str = _FULL_HEADER) -> Path:
    """Create a minimal rankings.csv with the given header."""
    csv_path = snap_dir / "rankings.csv"
    csv_path.write_text(header + "ACME,1.0,0.5,3,80,1.2,30,45,0.9,8k\n", encoding="utf-8")
    return csv_path


def _write_metadata(snap_dir: Path, extra: dict | None = None) -> Path:
    meta = {"as_of_date": snap_dir.name, "version": "1.6.0"}
    if extra:
        meta.update(extra)
    meta_path = snap_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta_path


def _make_13f_pit(pit_base: Path, dates: list[str]) -> None:
    """Create PIT cache dirs with index.json files."""
    for d in dates:
        d_dir = pit_base / d
        d_dir.mkdir(parents=True)
        idx = {"as_of_date": d, "schema_version": "sec_13f_pit_index.v1"}
        (d_dir / "index.json").write_text(json.dumps(idx) + "\n", encoding="utf-8")


def _make_ctgov_cache(ctgov_dir: Path, dates: list[str]) -> None:
    """Create trial_records_{date}.json files."""
    ctgov_dir.mkdir(parents=True, exist_ok=True)
    for d in dates:
        (ctgov_dir / f"trial_records_{d}.json").write_text("{}", encoding="utf-8")


# ── TestDiscoverCacheDates ────────────────────────────────────────────────

class TestDiscoverCacheDates:
    def test_discover_13f_dirs(self, tmp_path: Path) -> None:
        pit_base = tmp_path / "PIT"
        dates = ["2020-03-31", "2020-06-30", "2020-09-30"]
        _make_13f_pit(pit_base, dates)
        # Add a non-date dir that should be ignored
        (pit_base / "latest").mkdir()
        result = _discover_13f_dates(pit_base)
        assert result == dates

    def test_discover_ctgov_files(self, tmp_path: Path) -> None:
        ctgov = tmp_path / "ctgov"
        dates = ["2020-03-31", "2020-04-30", "2020-05-31"]
        _make_ctgov_cache(ctgov, dates)
        # Add a non-matching file
        (ctgov / "trial_records_latest.json").write_text("{}")
        # Add a meta file
        (ctgov / "trial_records_2020-03-31.meta.json").write_text("{}")
        result = _discover_ctgov_dates(ctgov)
        assert result == dates


# ── TestFindLatestCacheDate ───────────────────────────────────────────────

class TestFindLatestCacheDate:
    def test_exact_match(self) -> None:
        dates = ["2020-03-31", "2020-06-30", "2020-09-30"]
        assert _find_latest_cache_date(dates, "2020-06-30") == "2020-06-30"

    def test_nearest_prior(self) -> None:
        dates = ["2020-03-31", "2020-06-30", "2020-09-30"]
        assert _find_latest_cache_date(dates, "2020-08-15") == "2020-06-30"

    def test_no_match_before_first(self) -> None:
        dates = ["2020-03-31", "2020-06-30"]
        assert _find_latest_cache_date(dates, "2020-01-01") is None


# ── TestReconstructDataSources ────────────────────────────────────────────

class TestReconstructDataSources:
    def test_all_families_present(self, tmp_path: Path) -> None:
        snap_dir = tmp_path / "2020-06-15"
        snap_dir.mkdir()
        csv_path = _write_rankings(snap_dir)

        pit_base = tmp_path / "PIT"
        _make_13f_pit(pit_base, ["2020-03-31", "2020-06-30"])

        ctgov_dates = ["2020-03-31", "2020-05-31", "2020-06-30"]

        result = reconstruct_data_sources(
            "2020-06-15",
            csv_path,
            sec_13f_cache_dates=["2020-03-31", "2020-06-30"],
            ctgov_cache_dates=ctgov_dates,
            sec_13f_pit_base=pit_base,
        )
        assert "sec_13f" in result
        assert result["sec_13f"]["effective_date"] == "2020-03-31"
        assert result["sec_13f"]["lag_days"] == 45
        assert result["ctgov"]["effective_date"] == "2020-05-31"
        assert result["sec_8k"]["effective_date"] == "2020-06-15"

    def test_missing_13f_pre_cache_era(self, tmp_path: Path) -> None:
        snap_dir = tmp_path / "2020-01-03"
        snap_dir.mkdir()
        csv_path = _write_rankings(snap_dir)

        result = reconstruct_data_sources(
            "2020-01-03",
            csv_path,
            sec_13f_cache_dates=["2020-03-31"],  # first cache is after snap
            ctgov_cache_dates=["2020-03-31"],
        )
        assert "sec_13f" not in result
        assert "ctgov" not in result
        assert "sec_8k" in result

    def test_missing_ctgov(self, tmp_path: Path) -> None:
        snap_dir = tmp_path / "2020-02-14"
        snap_dir.mkdir()
        csv_path = _write_rankings(snap_dir)

        pit_base = tmp_path / "PIT"
        _make_13f_pit(pit_base, ["2020-01-31"])

        result = reconstruct_data_sources(
            "2020-02-14",
            csv_path,
            sec_13f_cache_dates=["2020-01-31"],
            ctgov_cache_dates=["2020-03-31"],  # after snap
            sec_13f_pit_base=pit_base,
        )
        assert "sec_13f" in result
        assert "ctgov" not in result

    def test_family_detection_from_header(self, tmp_path: Path) -> None:
        """Only sec_8k family active when header has only catalyst columns."""
        snap_dir = tmp_path / "2020-06-15"
        snap_dir.mkdir()
        csv_path = _write_rankings(snap_dir, header=_MINIMAL_HEADER)

        result = reconstruct_data_sources(
            "2020-06-15",
            csv_path,
            sec_13f_cache_dates=["2020-03-31"],
            ctgov_cache_dates=["2020-03-31"],
        )
        assert "sec_13f" not in result
        assert "ctgov" not in result
        assert "sec_8k" in result


# ── TestRestampBatch ──────────────────────────────────────────────────────

class TestRestampBatch:
    def _setup_snapshots(
        self, tmp_path: Path, dates: list[str], *, with_data_sources: bool = False
    ) -> tuple[Path, Path, Path]:
        snap_root = tmp_path / "snapshots"
        pit_base = tmp_path / "PIT"
        ctgov_dir = tmp_path / "ctgov"

        _make_13f_pit(pit_base, ["2020-03-31", "2020-06-30"])
        _make_ctgov_cache(ctgov_dir, ["2020-03-31", "2020-05-31"])

        for d in dates:
            sd = snap_root / d
            sd.mkdir(parents=True)
            _write_rankings(sd)
            extra = {}
            if with_data_sources:
                extra["data_sources"] = {"sec_8k": {"effective_date": d}}
            _write_metadata(sd, extra)

        return snap_root, pit_base, ctgov_dir

    def test_multi_date_batch(self, tmp_path: Path) -> None:
        dates = ["2020-01-03", "2020-04-15", "2020-06-15"]
        snap_root, pit_base, ctgov_dir = self._setup_snapshots(tmp_path, dates)

        summary = restamp_batch(
            snap_root,
            sec_13f_pit_base=pit_base,
            ctgov_cache_dir=ctgov_dir,
        )
        assert summary["n_total"] == 3
        assert summary["n_stamped"] == 3
        assert summary["n_skipped_existing"] == 0

        # Pre-cache date: only sec_8k
        meta_early = json.loads((snap_root / "2020-01-03" / "metadata.json").read_text())
        assert "sec_13f" not in meta_early["data_sources"]
        assert "sec_8k" in meta_early["data_sources"]
        assert "data_sources_restamped_at" in meta_early

        # Post-cache date: all three families
        meta_mid = json.loads((snap_root / "2020-06-15" / "metadata.json").read_text())
        assert "sec_13f" in meta_mid["data_sources"]
        assert "ctgov" in meta_mid["data_sources"]
        assert "sec_8k" in meta_mid["data_sources"]

        # sec_8k count = 3 (all dates), sec_13f = 2 (post-cache), ctgov = 2
        assert summary["families"]["sec_8k"] == 3
        assert summary["families"]["sec_13f"] == 2
        assert summary["families"]["ctgov"] == 2

    def test_skip_existing_idempotent(self, tmp_path: Path) -> None:
        dates = ["2020-04-15"]
        snap_root, pit_base, ctgov_dir = self._setup_snapshots(
            tmp_path, dates, with_data_sources=True
        )

        summary = restamp_batch(
            snap_root,
            sec_13f_pit_base=pit_base,
            ctgov_cache_dir=ctgov_dir,
        )
        assert summary["n_stamped"] == 0
        assert summary["n_skipped_existing"] == 1

        # Original data_sources preserved (only sec_8k, no 13f/ctgov)
        meta = json.loads((snap_root / "2020-04-15" / "metadata.json").read_text())
        assert "sec_13f" not in meta["data_sources"]

    def test_force_overwrites(self, tmp_path: Path) -> None:
        dates = ["2020-04-15"]
        snap_root, pit_base, ctgov_dir = self._setup_snapshots(
            tmp_path, dates, with_data_sources=True
        )

        summary = restamp_batch(
            snap_root,
            sec_13f_pit_base=pit_base,
            ctgov_cache_dir=ctgov_dir,
            force=True,
        )
        assert summary["n_stamped"] == 1
        assert summary["n_skipped_existing"] == 0

        # Now has all families
        meta = json.loads((snap_root / "2020-04-15" / "metadata.json").read_text())
        assert "sec_13f" in meta["data_sources"]
        assert "data_sources_restamped_at" in meta
