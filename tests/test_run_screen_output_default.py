"""Tests for _default_output_path in run_screen.py."""
from pathlib import Path

from run_screen import _default_output_path


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
