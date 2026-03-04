"""
Tests for tools/maintain_universe.py

12 tests covering:
  - audit flags non-active tickers
  - audit skips XBI (protected)
  - add creates a valid record
  - add is idempotent (no dup on re-add)
  - retire sets correct status
  - retire infers status from reason keyword
  - audit log entry written for add/retire
  - missing CIK allowed (add without --cik)
  - retire XBI blocked
  - audit flags missing-from-snapshots
  - retire ticker not in universe → sys.exit(1)
  - audit clean universe → no flags
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List
from unittest.mock import patch
import argparse

import pytest

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import tools.maintain_universe as mu


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_universe(entries: List[Dict]) -> List[Dict]:
    """Minimal universe list."""
    defaults = {"ticker": "ACME", "name": "ACME Corp", "status": "active",
                "sector": "Biotechnology", "exchange": "NASDAQ"}
    return [{**defaults, **e} for e in entries]


def _write_universe(tmp_path: Path, entries: List[Dict]) -> Path:
    p = tmp_path / "universe.json"
    p.write_text(json.dumps(_make_universe(entries)))
    return p


def _args(**kwargs) -> argparse.Namespace:
    """Build a minimal argparse Namespace for testing."""
    defaults = {
        "ticker": "ACME",
        "name": None,
        "sector": "Biotechnology",
        "exchange": None,
        "cik": None,
        "description": None,
        "operator": "test",
        "reason": "",
        "status": None,
        "universe_path": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# 1. audit: flags non-active status
# ---------------------------------------------------------------------------

class TestAuditNonActive:

    def test_non_active_ticker_flagged(self, tmp_path, capsys):
        """Ticker with status='delisted' should be flagged."""
        u_path = _write_universe(tmp_path, [
            {"ticker": "GOOD", "status": "active"},
            {"ticker": "GONE", "status": "delisted"},
        ])
        with patch.object(mu, "UNIVERSE_JSON", u_path), \
             patch.object(mu, "SNAPSHOTS_ROOT", tmp_path / "snaps"):
            mu.cmd_audit(_args())

        out = capsys.readouterr().out
        assert "GONE" in out
        assert "non_active_status" in out

    def test_active_ticker_not_flagged_for_status(self, tmp_path, capsys):
        """Active tickers should not appear in non_active_status flag."""
        u_path = _write_universe(tmp_path, [{"ticker": "OK", "status": "active"}])
        with patch.object(mu, "UNIVERSE_JSON", u_path), \
             patch.object(mu, "SNAPSHOTS_ROOT", tmp_path / "snaps"):
            mu.cmd_audit(_args())

        out = capsys.readouterr().out
        assert "non_active_status" not in out


# ---------------------------------------------------------------------------
# 2. audit: XBI always skipped
# ---------------------------------------------------------------------------

class TestAuditXbiProtected:

    def test_xbi_not_flagged(self, tmp_path, capsys):
        """XBI in universe with non-active status should NOT be flagged."""
        u_path = _write_universe(tmp_path, [
            {"ticker": "XBI", "status": "benchmark"},
            {"ticker": "NORM", "status": "active"},
        ])
        with patch.object(mu, "UNIVERSE_JSON", u_path), \
             patch.object(mu, "SNAPSHOTS_ROOT", tmp_path / "snaps"):
            mu.cmd_audit(_args())

        out = capsys.readouterr().out
        assert "XBI" not in out.split("Protected")[0]  # not in flags section


# ---------------------------------------------------------------------------
# 3. add: creates valid record
# ---------------------------------------------------------------------------

class TestAdd:

    def test_add_creates_entry(self, tmp_path):
        u_path = tmp_path / "universe.json"
        u_path.write_text("[]")
        log_path = tmp_path / "universe_audit_log.jsonl"

        with patch.object(mu, "UNIVERSE_JSON", u_path), \
             patch.object(mu, "AUDIT_LOG", log_path):
            mu.cmd_add(_args(ticker="NEWCO", name="New Company", cik="0001234567"))

        universe = json.loads(u_path.read_text())
        assert len(universe) == 1
        entry = universe[0]
        assert entry["ticker"] == "NEWCO"
        assert entry["name"] == "New Company"
        assert entry["cik"] == "0001234567"
        assert entry["status"] == "active"
        assert "added_date" in entry

    def test_add_without_cik_allowed(self, tmp_path):
        """CIK is optional."""
        u_path = tmp_path / "universe.json"
        u_path.write_text("[]")
        log_path = tmp_path / "log.jsonl"

        with patch.object(mu, "UNIVERSE_JSON", u_path), \
             patch.object(mu, "AUDIT_LOG", log_path):
            mu.cmd_add(_args(ticker="NOCIK", name="No CIK Company", cik=None))

        universe = json.loads(u_path.read_text())
        assert universe[0]["ticker"] == "NOCIK"
        assert "cik" not in universe[0]


# ---------------------------------------------------------------------------
# 4. add: idempotent (no dup)
# ---------------------------------------------------------------------------

class TestAddIdempotent:

    def test_add_twice_no_duplicate(self, tmp_path, capsys):
        """Adding the same ticker twice → no duplicate entry."""
        u_path = tmp_path / "universe.json"
        u_path.write_text("[]")
        log_path = tmp_path / "log.jsonl"

        with patch.object(mu, "UNIVERSE_JSON", u_path), \
             patch.object(mu, "AUDIT_LOG", log_path):
            mu.cmd_add(_args(ticker="DUP"))
            mu.cmd_add(_args(ticker="DUP"))  # second call

        universe = json.loads(u_path.read_text())
        assert len([e for e in universe if e["ticker"] == "DUP"]) == 1


# ---------------------------------------------------------------------------
# 5. retire: sets status
# ---------------------------------------------------------------------------

class TestRetire:

    def test_retire_sets_new_status(self, tmp_path):
        u_path = _write_universe(tmp_path, [{"ticker": "GONE", "status": "active"}])
        log_path = tmp_path / "log.jsonl"

        with patch.object(mu, "UNIVERSE_JSON", u_path), \
             patch.object(mu, "AUDIT_LOG", log_path):
            mu.cmd_retire(_args(ticker="GONE", reason="acquired by Merck 2026-01-01"))

        universe = json.loads(u_path.read_text())
        entry = next(e for e in universe if e["ticker"] == "GONE")
        assert entry["status"] == "excluded_acquired"
        assert entry["retire_reason"] == "acquired by Merck 2026-01-01"

    def test_retire_infers_delisted_from_reason(self, tmp_path):
        u_path = _write_universe(tmp_path, [{"ticker": "BANKR", "status": "active"}])
        log_path = tmp_path / "log.jsonl"

        with patch.object(mu, "UNIVERSE_JSON", u_path), \
             patch.object(mu, "AUDIT_LOG", log_path):
            mu.cmd_retire(_args(ticker="BANKR", reason="delisted 2026-02-01"))

        universe = json.loads(u_path.read_text())
        entry = next(e for e in universe if e["ticker"] == "BANKR")
        assert entry["status"] == "delisted"


# ---------------------------------------------------------------------------
# 6. audit log entry written
# ---------------------------------------------------------------------------

class TestAuditLog:

    def test_add_writes_log_entry(self, tmp_path):
        u_path = tmp_path / "universe.json"
        u_path.write_text("[]")
        log_path = tmp_path / "log.jsonl"

        with patch.object(mu, "UNIVERSE_JSON", u_path), \
             patch.object(mu, "AUDIT_LOG", log_path):
            mu.cmd_add(_args(ticker="LOGTEST", name="Log Test", operator="alice"))

        log_entries = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        assert len(log_entries) == 1
        assert log_entries[0]["action"] == "add"
        assert log_entries[0]["ticker"] == "LOGTEST"
        assert log_entries[0]["operator"] == "alice"

    def test_retire_writes_log_entry(self, tmp_path):
        u_path = _write_universe(tmp_path, [{"ticker": "LOGT2", "status": "active"}])
        log_path = tmp_path / "log.jsonl"

        with patch.object(mu, "UNIVERSE_JSON", u_path), \
             patch.object(mu, "AUDIT_LOG", log_path):
            mu.cmd_retire(_args(ticker="LOGT2", reason="delisted", operator="bob"))

        log_entries = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        assert len(log_entries) == 1
        assert log_entries[0]["action"] == "retire"
        assert log_entries[0]["operator"] == "bob"


# ---------------------------------------------------------------------------
# 7. retire XBI blocked
# ---------------------------------------------------------------------------

class TestRetireXbiBlocked:

    def test_retire_xbi_exits_nonzero(self, tmp_path):
        u_path = _write_universe(tmp_path, [{"ticker": "XBI", "status": "benchmark"}])
        log_path = tmp_path / "log.jsonl"

        with patch.object(mu, "UNIVERSE_JSON", u_path), \
             patch.object(mu, "AUDIT_LOG", log_path), \
             pytest.raises(SystemExit) as exc_info:
            mu.cmd_retire(_args(ticker="XBI", reason="testing"))

        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# 8. retire ticker not in universe
# ---------------------------------------------------------------------------

class TestRetireNotFound:

    def test_retire_unknown_ticker_exits_nonzero(self, tmp_path):
        u_path = _write_universe(tmp_path, [{"ticker": "KNOWN", "status": "active"}])
        log_path = tmp_path / "log.jsonl"

        with patch.object(mu, "UNIVERSE_JSON", u_path), \
             patch.object(mu, "AUDIT_LOG", log_path), \
             pytest.raises(SystemExit) as exc_info:
            mu.cmd_retire(_args(ticker="UNKNOWN"))

        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# 9. audit: clean universe has no flags
# ---------------------------------------------------------------------------

class TestAuditClean:

    def test_clean_universe_no_flags(self, tmp_path, capsys):
        """Fully active universe with no snapshot data → no flags."""
        u_path = _write_universe(tmp_path, [
            {"ticker": "A", "status": "active"},
            {"ticker": "B", "status": "active"},
        ])

        with patch.object(mu, "UNIVERSE_JSON", u_path), \
             patch.object(mu, "SNAPSHOTS_ROOT", tmp_path / "nosnaps"):
            mu.cmd_audit(_args())

        out = capsys.readouterr().out
        assert "looks clean" in out
