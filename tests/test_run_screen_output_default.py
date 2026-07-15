"""Tests for snapshot-managed output paths in run_screen.py."""

from pathlib import Path

import pytest

from run_screen import (
    _default_output_path,
    _enforce_snapshot_output_overwrite_policy,
    _managed_snapshot_dir,
    _output_targets_managed_snapshot_dir,
)


class TestDefaultOutputPath:
    def test_with_snapshot_dir(self):
        result = _default_output_path(
            as_of_date="2026-02-28",
            data_dir=Path("/repo/production_data"),
            snapshot_dir=Path("/tmp/snap"),
        )
        assert result == Path("/tmp/snap/2026-02-28/screen_output.json")

    def test_without_snapshot_dir(self):
        result = _default_output_path(
            as_of_date="2026-02-28",
            data_dir=Path("/repo/production_data"),
            snapshot_dir=None,
        )
        assert result == Path("/repo/data/snapshots/2026-02-28/screen_output.json")

    def test_different_date(self):
        result = _default_output_path(
            as_of_date="2025-01-15",
            data_dir=Path("/x/production_data"),
            snapshot_dir=None,
        )
        assert result == Path("/x/data/snapshots/2025-01-15/screen_output.json")


class TestSnapshotOutputOverwritePolicy:
    def test_managed_snapshot_dir_defaults_to_data_snapshots(self):
        result = _managed_snapshot_dir(
            as_of_date="2026-02-28",
            data_dir=Path("/repo/production_data"),
            snapshot_dir=None,
        )
        assert result == Path("/repo/data/snapshots/2026-02-28")

    def test_detects_explicit_output_targeting_managed_snapshot_dir(self):
        data_dir = Path("/repo/production_data")
        output = Path("/repo/data/snapshots/2026-02-28/custom_output.json")

        assert _output_targets_managed_snapshot_dir(
            output,
            as_of_date="2026-02-28",
            data_dir=data_dir,
            snapshot_dir=None,
        )

    def test_allows_explicit_output_outside_managed_snapshot_dir(self, tmp_path):
        data_dir = tmp_path / "production_data"
        data_dir.mkdir()
        existing_snapshot = tmp_path / "data" / "snapshots" / "2026-02-28"
        existing_snapshot.mkdir(parents=True)
        explicit_output = tmp_path / "custom_outputs" / "screen_output.json"

        # Output outside the managed snapshot dir is allowed (no raise; returns None)
        result = _enforce_snapshot_output_overwrite_policy(
            explicit_output,
            as_of_date="2026-02-28",
            data_dir=data_dir,
            snapshot_dir=None,
            force_overwrite=False,
        )
        assert result is None

    def test_blocks_default_output_when_snapshot_dir_already_exists(self, tmp_path):
        data_dir = tmp_path / "production_data"
        data_dir.mkdir()
        snap_dir = tmp_path / "data" / "snapshots" / "2026-02-28"
        snap_dir.mkdir(parents=True)
        rankings = snap_dir / "rankings.csv"
        output = snap_dir / "screen_output.json"
        rankings.write_text("ticker,rank\nOLD,1\n", encoding="utf-8")
        output.write_text('{"old": true}\n', encoding="utf-8")

        with pytest.raises(FileExistsError, match="Snapshot already exists"):
            _enforce_snapshot_output_overwrite_policy(
                output,
                as_of_date="2026-02-28",
                data_dir=data_dir,
                snapshot_dir=None,
                force_overwrite=False,
            )

        assert rankings.read_text(encoding="utf-8") == "ticker,rank\nOLD,1\n"
        assert output.read_text(encoding="utf-8") == '{"old": true}\n'

    def test_blocks_explicit_output_inside_managed_snapshot_dir(self, tmp_path):
        data_dir = tmp_path / "production_data"
        data_dir.mkdir()
        snap_dir = tmp_path / "custom_snapshots" / "2026-02-28"
        snap_dir.mkdir(parents=True)

        with pytest.raises(FileExistsError):
            _enforce_snapshot_output_overwrite_policy(
                snap_dir / "explicit.json",
                as_of_date="2026-02-28",
                data_dir=data_dir,
                snapshot_dir=tmp_path / "custom_snapshots",
                force_overwrite=False,
            )

    def test_force_overwrite_allows_managed_snapshot_output(self, tmp_path):
        data_dir = tmp_path / "production_data"
        data_dir.mkdir()
        snap_dir = tmp_path / "data" / "snapshots" / "2026-02-28"
        snap_dir.mkdir(parents=True)

        # force_overwrite=True permits writing into the managed snapshot dir (no raise; returns None)
        result = _enforce_snapshot_output_overwrite_policy(
            snap_dir / "screen_output.json",
            as_of_date="2026-02-28",
            data_dir=data_dir,
            snapshot_dir=None,
            force_overwrite=True,
        )
        assert result is None
