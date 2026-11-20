"""Tests for tools/build_universe_maintenance.py — dead-ticker detection and paths."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from common.corporate_actions import load_actions  # noqa: E402
from tools.build_universe_maintenance import build_universe_maintenance, check_dead_not_retired  # noqa: E402

AS_OF = "2026-08-14"


@pytest.fixture
def registry(tmp_path):
    """Synthetic corporate-actions registry covering each relevant action type."""
    path = tmp_path / "corporate_actions.json"
    path.write_text(
        json.dumps(
            {
                "actions": [
                    {
                        "ticker": "DEADACQ",
                        "action": "acquisition",
                        "effective_date": "2026-07-15",
                        "acquirer": "BigCo",
                    },
                    {"ticker": "DEADBK", "action": "bankruptcy", "effective_date": "2026-06-24"},
                    {"ticker": "DEADDL", "action": "delisted", "effective_date": "2026-05-01"},
                    {
                        "ticker": "FUTURE",
                        "action": "acquisition",
                        "effective_date": "2026-12-31",
                        "acquirer": "LaterCo",
                    },
                    {"ticker": "PENDING", "action": "pending_acquisition", "announced_date": "2026-08-01"},
                    {"ticker": "SPLIT", "action": "forward_split", "effective_date": "2026-03-01", "ratio": "2:1"},
                ]
            }
        )
    )
    return load_actions(path)


def _u(ticker, status="active"):
    return {"ticker": ticker, "status": status}


class TestDeadNotRetired:

    def test_dead_with_active_status_is_flagged(self, registry):
        out = check_dead_not_retired([_u("DEADACQ")], AS_OF, registry=registry)

        assert out == {"DEADACQ": "2026-07-15"}

    @pytest.mark.parametrize(
        "ticker,effective", [("DEADACQ", "2026-07-15"), ("DEADBK", "2026-06-24"), ("DEADDL", "2026-05-01")]
    )
    def test_all_dead_action_types_flagged(self, registry, ticker, effective):
        out = check_dead_not_retired([_u(ticker)], AS_OF, registry=registry)

        assert out[ticker] == effective

    @pytest.mark.parametrize("status", ["delisted", "excluded_acquired", "retired"])
    def test_already_retired_is_not_flagged(self, registry, status):
        """Retirement is the operator's action; once done there is nothing to report."""
        out = check_dead_not_retired([_u("DEADACQ", status)], AS_OF, registry=registry)

        assert out == {}

    def test_live_ticker_not_flagged(self, registry):
        out = check_dead_not_retired([_u("ALIVE")], AS_OF, registry=registry)

        assert out == {}

    def test_pending_acquisition_not_flagged(self, registry):
        """pending_acquisition is deliberately not a dead action type — it still trades."""
        out = check_dead_not_retired([_u("PENDING")], AS_OF, registry=registry)

        assert out == {}

    def test_split_is_not_death(self, registry):
        out = check_dead_not_retired([_u("SPLIT")], AS_OF, registry=registry)

        assert out == {}

    def test_missing_ticker_field_is_skipped(self, registry):
        out = check_dead_not_retired([{"status": "active"}], AS_OF, registry=registry)

        assert out == {}


class TestPointInTimeSafety:
    """A replay must not retire names that were alive at the as-of date."""

    def test_death_after_as_of_is_not_flagged(self, registry):
        out = check_dead_not_retired([_u("FUTURE")], AS_OF, registry=registry)

        assert out == {}

    def test_death_before_as_of_is_flagged(self, registry):
        out = check_dead_not_retired([_u("FUTURE")], "2027-01-15", registry=registry)

        assert out == {"FUTURE": "2026-12-31"}

    def test_as_of_before_all_deaths_flags_nothing(self, registry):
        universe = [_u("DEADACQ"), _u("DEADBK"), _u("DEADDL")]

        out = check_dead_not_retired(universe, "2026-01-01", registry=registry)

        assert out == {}

    def test_as_of_between_deaths_flags_only_the_earlier(self, registry):
        universe = [_u("DEADDL"), _u("DEADACQ")]

        # DEADDL died 2026-05-01, DEADACQ not until 2026-07-15.
        out = check_dead_not_retired(universe, "2026-06-01", registry=registry)

        assert out == {"DEADDL": "2026-05-01"}


class TestFailOpen:

    def test_empty_registry_returns_empty(self, tmp_path):
        empty = load_actions(tmp_path / "does_not_exist.json")

        out = check_dead_not_retired([_u("DEADACQ")], AS_OF, registry=empty)

        assert out == {}


class TestSnapshotsDirDefault:
    """Regression: the default pointed at data/snapshots_pit, dead since 2026-06-10."""

    def test_default_snapshots_dir_is_live_tree(self):
        default = inspect.signature(build_universe_maintenance).parameters["snapshots_dir"].default

        assert default.name == "snapshots"
        assert "snapshots_pit" not in str(default)
