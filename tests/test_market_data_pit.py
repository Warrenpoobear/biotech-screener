"""Tests for market_data.json PIT field + staleness gate (Commit 1 / 2026-04-28 audit).

Covers _load_market_data_with_pit_check in run_screen.py:
  - v1 bare list: loads and warns once, no date validation
  - v2 matching as_of_date: loads cleanly (INFO)
  - v2 stale >MAX_MARKET_DATA_STALE_DAYS: loads but warns
  - v2 as_of_date AFTER screener date: raises ValueError (lookahead block)
  - v2 missing as_of_date: raises ValueError (malformed)
  - v2 boundary conditions: exactly at and one-over max stale days
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_RECORD = {"ticker": "AAAA", "price": 10.0, "collected_at": "2026-04-27"}


def _write_v1(path: Path, records=None):
    if records is None:
        records = [_SAMPLE_RECORD]
    path.write_text(json.dumps(records), encoding="utf-8")


def _write_v2(path: Path, as_of_date: str, records=None):
    if records is None:
        records = [_SAMPLE_RECORD]
    payload = {
        "as_of_date": as_of_date,
        "schema_version": "market_data.v2",
        "records": records,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _get_loader():
    from run_screen import _load_market_data_with_pit_check, MAX_MARKET_DATA_STALE_DAYS
    return _load_market_data_with_pit_check, MAX_MARKET_DATA_STALE_DAYS


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_v1_list_loads_and_warns(tmp_path, caplog):
    """v1 bare-list shape: loads records, warns once about missing as_of_date."""
    load_fn, _ = _get_loader()
    f = tmp_path / "market_data.json"
    _write_v1(f)
    with caplog.at_level(logging.WARNING, logger="root"):
        records = load_fn(f, "2026-04-28")
    assert len(records) == 1
    assert records[0]["ticker"] == "AAAA"
    assert any("v1" in m or "as_of_date unverifiable" in m for m in caplog.messages)


def test_v2_matching_date_loads_cleanly(tmp_path, caplog):
    """v2 with as_of_date == screener date: no warning, returns records."""
    load_fn, _ = _get_loader()
    f = tmp_path / "market_data.json"
    _write_v2(f, "2026-04-28")
    with caplog.at_level(logging.WARNING, logger="root"):
        records = load_fn(f, "2026-04-28")
    assert len(records) == 1
    warning_msgs = [m for m in caplog.messages if "stale" in m.lower() or "lookahead" in m.lower()]
    assert warning_msgs == [], f"Unexpected warnings: {warning_msgs}"


def test_v2_one_day_stale_loads_cleanly(tmp_path, caplog):
    """v2 with as_of_date 1 day before screener: within tolerance, no staleness warning."""
    load_fn, _ = _get_loader()
    f = tmp_path / "market_data.json"
    _write_v2(f, "2026-04-27")
    with caplog.at_level(logging.WARNING, logger="root"):
        records = load_fn(f, "2026-04-28")
    assert len(records) == 1
    stale_msgs = [m for m in caplog.messages if "stale" in m.lower()]
    assert stale_msgs == []


def test_v2_stale_beyond_max_warns(tmp_path, caplog):
    """v2 with as_of_date MAX_MARKET_DATA_STALE_DAYS+1 behind screener: WARNING."""
    load_fn, max_stale = _get_loader()
    f = tmp_path / "market_data.json"
    source = (date.fromisoformat("2026-04-28") - timedelta(days=max_stale + 1)).isoformat()
    _write_v2(f, source)
    with caplog.at_level(logging.WARNING, logger="root"):
        records = load_fn(f, "2026-04-28")
    assert len(records) == 1
    stale_msgs = [m for m in caplog.messages if "stale" in m.lower()]
    assert stale_msgs, "Expected staleness warning but got none"


def test_v2_lookahead_raises(tmp_path):
    """v2 with as_of_date AFTER screener date: raises ValueError (lookahead block)."""
    load_fn, _ = _get_loader()
    f = tmp_path / "market_data.json"
    _write_v2(f, "2026-04-29")  # 1 day after screener
    with pytest.raises(ValueError, match="lookahead"):
        load_fn(f, "2026-04-28")


def test_v2_missing_as_of_date_raises(tmp_path):
    """v2 dict without as_of_date field raises ValueError (malformed)."""
    load_fn, _ = _get_loader()
    f = tmp_path / "market_data.json"
    payload = {"schema_version": "market_data.v2", "records": [_SAMPLE_RECORD]}
    f.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="as_of_date"):
        load_fn(f, "2026-04-28")


def test_v2_stale_exactly_at_boundary_no_warn(tmp_path, caplog):
    """v2 exactly MAX_MARKET_DATA_STALE_DAYS stale: still OK (boundary inclusive)."""
    load_fn, max_stale = _get_loader()
    screener = "2026-04-28"
    source = (date.fromisoformat(screener) - timedelta(days=max_stale)).isoformat()
    f = tmp_path / "market_data.json"
    _write_v2(f, source)
    with caplog.at_level(logging.WARNING, logger="root"):
        records = load_fn(f, screener)
    assert len(records) == 1
    stale_msgs = [m for m in caplog.messages if "stale" in m.lower()]
    assert stale_msgs == [], f"Should not warn at exactly the boundary: {stale_msgs}"


def test_v2_stale_one_over_boundary_warns(tmp_path, caplog):
    """v2 MAX_MARKET_DATA_STALE_DAYS+1 stale: triggers WARNING."""
    load_fn, max_stale = _get_loader()
    screener = "2026-04-28"
    source = (date.fromisoformat(screener) - timedelta(days=max_stale + 1)).isoformat()
    f = tmp_path / "market_data.json"
    _write_v2(f, source)
    with caplog.at_level(logging.WARNING, logger="root"):
        records = load_fn(f, screener)
    assert len(records) == 1
    stale_msgs = [m for m in caplog.messages if "stale" in m.lower()]
    assert stale_msgs, "Expected staleness warning for MAX+1 days"
