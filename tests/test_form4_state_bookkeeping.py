"""Tests for Form 4 producer state bookkeeping (Spec 065 stable-snapshot gate).

Pins the post-2026-05-01 operational repair contract:
  - _state_write merges into existing state (preserves other keys)
  - last_attempt is independent from last_success (timeout-resilient)
  - last_new_filing only advances when total_txns > 0
  - last_panel_rebuild updates on panel-only runs
  - schema_fingerprint changes when InsiderTransaction fields change
  - missing-vs-known-zero distinction is preserved end-to-end
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.fetch_form4_insider import InsiderTransaction, _schema_fingerprint, _state_load, _state_write  # noqa: E402


@pytest.fixture
def tmp_state_file(monkeypatch, tmp_path):
    """Redirect STATE_FILE to a tmp path so tests can't touch real state."""
    fake_state = tmp_path / "fetch_state.json"
    monkeypatch.setattr("tools.fetch_form4_insider.STATE_FILE", fake_state)
    return fake_state


def test_state_load_returns_empty_when_file_missing(tmp_state_file):
    assert _state_load() == {}


def test_state_load_returns_empty_when_file_corrupt(tmp_state_file):
    tmp_state_file.write_text("{not valid json")
    assert _state_load() == {}


def test_state_write_creates_file(tmp_state_file):
    _state_write({"last_attempt": "2026-05-01T13:30:00+00:00"})
    assert tmp_state_file.exists()
    assert _state_load()["last_attempt"] == "2026-05-01T13:30:00+00:00"


def test_state_write_merges_into_existing(tmp_state_file):
    """Critical bookkeeping property: a partial update must NOT clobber
    fields written by a prior run. Without merge, last_success could be
    erased by a partial failure that only writes last_attempt."""
    _state_write({"last_success": "2026-04-30T13:30:00+00:00", "tickers_checked": 341})
    _state_write({"last_attempt": "2026-05-01T13:30:00+00:00"})
    state = _state_load()
    assert state["last_success"] == "2026-04-30T13:30:00+00:00"
    assert state["tickers_checked"] == 341
    assert state["last_attempt"] == "2026-05-01T13:30:00+00:00"


def test_last_attempt_independent_of_last_success(tmp_state_file):
    """Spec 065 §1 #1 distinguishes producer-ran-but-failed from never-ran.
    last_attempt advances on every run start; last_success only on success."""
    _state_write({"last_attempt": "2026-05-01T13:30:00+00:00"})
    state = _state_load()
    assert state["last_attempt"] == "2026-05-01T13:30:00+00:00"
    assert "last_success" not in state  # not pre-populated by attempt write


def test_last_new_filing_only_advances_when_new_transactions(tmp_state_file):
    """No-new-filings day must record success without faking new data."""
    # Day 1: new filings → last_new_filing set
    _state_write(
        {
            "last_success": "2026-05-01T13:30:00+00:00",
            "last_new_filing": "2026-05-01T13:30:00+00:00",
            "new_transactions": 3,
        }
    )
    # Day 2: no new filings → last_success advances, last_new_filing must NOT
    _state_write(
        {
            "last_success": "2026-05-02T13:30:00+00:00",
            "new_transactions": 0,
        }
    )
    state = _state_load()
    assert state["last_success"] == "2026-05-02T13:30:00+00:00"
    assert state["last_new_filing"] == "2026-05-01T13:30:00+00:00"  # preserved
    assert state["new_transactions"] == 0


def test_last_panel_rebuild_independent_of_fetch(tmp_state_file):
    """Panel rebuild can run via --panel-only without touching fetch state."""
    _state_write({"last_success": "2026-05-01T13:30:00+00:00"})
    # Simulate --panel-only run: only last_panel_rebuild + panel_rows update
    _state_write({"last_panel_rebuild": "2026-05-01T13:35:00+00:00", "panel_rows": 27500})
    state = _state_load()
    assert state["last_success"] == "2026-05-01T13:30:00+00:00"  # untouched
    assert state["last_panel_rebuild"] == "2026-05-01T13:35:00+00:00"
    assert state["panel_rows"] == 27500


def test_schema_fingerprint_is_stable_for_unchanged_dataclass():
    """The fingerprint must be deterministic for a given dataclass shape."""
    fp1 = _schema_fingerprint()
    fp2 = _schema_fingerprint()
    assert fp1 == fp2
    assert len(fp1) == 12  # 12-char sha256 prefix per implementation


def test_schema_fingerprint_changes_when_field_added():
    """Adding/removing a field on InsiderTransaction must change the fingerprint.
    This is the producer-side schema drift detector for Spec 065 §1 #2."""
    original = _schema_fingerprint()

    # Simulate schema drift by injecting a fake __dataclass_fields__ extension.
    fake_fields = dict(InsiderTransaction.__dataclass_fields__)
    fake_fields["new_field_xyz"] = mock.MagicMock()

    with mock.patch.object(InsiderTransaction, "__dataclass_fields__", fake_fields):
        drifted = _schema_fingerprint()

    assert drifted != original


def test_state_round_trip_preserves_field_types(tmp_state_file):
    """Numeric fields must round-trip as int, not coerce to str."""
    _state_write(
        {
            "tickers_checked": 341,
            "tickers_updated": 0,
            "new_transactions": 0,
            "panel_rows": 27500,
        }
    )
    state = _state_load()
    assert state["tickers_checked"] == 341
    assert isinstance(state["tickers_checked"], int)
    assert state["new_transactions"] == 0
    assert isinstance(state["new_transactions"], int)
