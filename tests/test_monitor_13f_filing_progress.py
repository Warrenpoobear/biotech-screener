"""Tests for tools/monitor_13f_filing_progress.py — per-quarter filing progress tracking."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from tools.monitor_13f_filing_progress import (  # noqa: E402
    STATUS_FILE_TEMPLATE,
    load_status,
    save_status,
    status_path_for,
)

Q1 = "2026-03-31"
Q2 = "2026-06-30"


def _write_status(path: Path, quarter: str, n_filed: int) -> None:
    filed = {f"{i:010d}": {"name": f"Manager {i}", "filed_date": "2026-05-10"} for i in range(n_filed)}
    path.write_text(json.dumps({"target_quarter": quarter, "last_check": None, "filed": filed, "pending": {}}))


class TestStatusPathForQuarter:
    """Status files must be scoped per quarter so results cannot be conflated."""

    def test_distinct_quarters_get_distinct_paths(self):
        assert status_path_for(Q1) != status_path_for(Q2)

    def test_path_follows_template(self):
        assert status_path_for(Q2) == Path(STATUS_FILE_TEMPLATE.format(quarter=Q2))

    def test_override_is_honoured(self, tmp_path):
        override = tmp_path / "custom.json"
        assert status_path_for(Q2, str(override)) == override


class TestLoadStatusQuarterGuard:
    """Reusing another quarter's `filed` map would skip every manager in it.

    The loop in check_manager_filings() skips any CIK already present in
    status["filed"], so a Q1 file loaded for Q2 would report all Q1 filers as
    having filed for Q2 and immediately trip the 70% milestone.
    """

    def test_same_quarter_state_is_reused(self, tmp_path):
        path = tmp_path / "status.json"
        _write_status(path, Q1, 54)

        status = load_status(path, Q1)

        assert len(status["filed"]) == 54
        assert status["target_quarter"] == Q1

    def test_different_quarter_state_is_discarded(self, tmp_path):
        path = tmp_path / "status.json"
        _write_status(path, Q1, 54)

        status = load_status(path, Q2)

        assert status["filed"] == {}
        assert status["pending"] == {}
        assert status["target_quarter"] == Q2

    def test_status_file_without_recorded_quarter_is_discarded(self, tmp_path):
        """Legacy files predate target_quarter tracking and cannot be trusted."""
        path = tmp_path / "status.json"
        path.write_text(json.dumps({"filed": {"0000000001": {"name": "Legacy"}}, "pending": {}}))

        status = load_status(path, Q2)

        assert status["filed"] == {}
        assert status["target_quarter"] == Q2

    def test_absent_file_yields_empty_state(self, tmp_path):
        status = load_status(tmp_path / "missing.json", Q2)

        assert status == {"target_quarter": Q2, "last_check": None, "filed": {}, "pending": {}}

    @pytest.mark.parametrize("key", ["filed", "pending"])
    def test_missing_keys_are_defaulted(self, tmp_path, key):
        path = tmp_path / "status.json"
        payload = {"target_quarter": Q2, "filed": {}, "pending": {}}
        del payload[key]
        path.write_text(json.dumps(payload))

        status = load_status(path, Q2)

        assert status[key] == {}


class TestSaveStatusRoundTrip:
    def test_save_then_load_preserves_state(self, tmp_path):
        path = tmp_path / "nested" / "status.json"
        status = {"target_quarter": Q2, "last_check": "2026-08-14T12:00:00", "filed": {"0000000001": {}}, "pending": {}}

        save_status(status, path)

        assert path.exists()
        assert load_status(path, Q2)["filed"] == {"0000000001": {}}

    def test_save_creates_parent_directory(self, tmp_path):
        path = tmp_path / "a" / "b" / "status.json"
        save_status({"target_quarter": Q2, "filed": {}, "pending": {}}, path)
        assert path.exists()
