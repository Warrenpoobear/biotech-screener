"""Tests for the market_snapshot staleness and validity preflight gate.

Classification: REGIME_SNAPSHOT_REFRESH_AND_STALENESS_GATE_NO_MODEL_CHANGE

Acceptance tests covered:
  1. snapshot older than 2 trading days => check_freshness ok=False
  2. VIX=0 in live snapshot => check_freshness ok=False
  3. all regime signals zero => check_freshness ok=False
  4. fresh + valid snapshot => check_freshness ok=True
  5. missing snapshot file => check_freshness ok=False
  6. malformed JSON => check_freshness ok=False
  7. snapshot exactly 2 trading days old => ok=True (boundary: at limit)
  8. snapshot 3 trading days old => ok=False (one over limit)
  9. missing as_of_date field => ok=False
  10. emit_diagnostic does not raise on any result shape
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import date
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from check_market_snapshot_freshness import _count_trading_days, check_freshness, count_present_feeds, emit_diagnostic

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_snap(d: dict, tmp_dir: str) -> Path:
    p = Path(tmp_dir) / "market_snapshot.json"
    p.write_text(json.dumps(d) + "\n")
    return p


def _valid_snap(as_of_date: str) -> dict:
    return {
        "provenance": {"as_of_date": as_of_date, "version": "1.5.0"},
        "vix": "18.50",
        "xbi_vs_spy_30d": "-2.30",
        "xbi_momentum_10d": "1.50",
        "spy_momentum_10d": "2.10",
        "xbi_realized_vol_20d": "22.50",
        "feeds": {"vix": "live", "xbi": "live"},
    }


# ---------------------------------------------------------------------------
# Trading-day counter unit tests
# ---------------------------------------------------------------------------


class TestCountTradingDays:
    def test_same_day(self):
        d = date(2026, 6, 26)  # Friday
        assert _count_trading_days(d, d) == 1

    def test_weekend_skipped(self):
        # Monday Jun 22 to Friday Jun 26 of same week = 5 trading days
        mon = date(2026, 6, 22)
        fri = date(2026, 6, 26)
        assert _count_trading_days(mon, fri) == 5

    def test_crosses_weekend(self):
        # Friday Jun 26 to Mon Jun 29 = 2 trading days (Sat/Sun skipped)
        fri = date(2026, 6, 26)
        mon = date(2026, 6, 29)
        assert _count_trading_days(fri, mon) == 2

    def test_from_after_to_returns_zero(self):
        assert _count_trading_days(date(2026, 6, 26), date(2026, 6, 20)) == 0


# ---------------------------------------------------------------------------
# check_freshness tests
# ---------------------------------------------------------------------------


class TestCheckFreshness:
    def test_fresh_valid_snapshot_ok(self):
        ref = date(2026, 6, 26)
        snap_date = "2026-06-26"
        with tempfile.TemporaryDirectory() as tmp:
            p = _write_snap(_valid_snap(snap_date), tmp)
            result = check_freshness(p, reference_date=ref)
        assert result["ok"] is True
        assert result["age_trading_days"] == 0
        assert result["issues"] == []

    def test_one_trading_day_old_ok(self):
        # Snap from Thursday, check on Friday
        ref = date(2026, 6, 26)  # Friday
        with tempfile.TemporaryDirectory() as tmp:
            p = _write_snap(_valid_snap("2026-06-25"), tmp)  # Thursday
            result = check_freshness(p, reference_date=ref)
        assert result["ok"] is True
        assert result["age_trading_days"] == 1

    def test_exactly_two_trading_days_old_ok(self):
        # Snap from Wednesday Jun 24, check on Friday Jun 26 = 2 trading-day gap
        ref = date(2026, 6, 26)  # Friday
        with tempfile.TemporaryDirectory() as tmp:
            p = _write_snap(_valid_snap("2026-06-24"), tmp)  # Wednesday
            result = check_freshness(p, reference_date=ref)
        assert result["ok"] is True
        assert result["age_trading_days"] == 2

    def test_three_trading_days_old_fails(self):
        # Snap from Tuesday Jun 23, check on Friday Jun 26 = 3 trading-day gap
        ref = date(2026, 6, 26)  # Friday
        with tempfile.TemporaryDirectory() as tmp:
            p = _write_snap(_valid_snap("2026-06-23"), tmp)  # Tuesday
            result = check_freshness(p, reference_date=ref)
        assert result["ok"] is False
        assert result["age_trading_days"] == 3
        assert any("trading days old" in i for i in result["issues"])

    def test_very_stale_snapshot_fails(self):
        """Phase 3 scenario: snapshot from 2026-05-19, checked on 2026-06-09."""
        stale = {
            "provenance": {"as_of_date": "2026-05-19", "version": "1.4.0"},
            "vix": "0",
            "xbi_vs_spy_30d": "0",
            "xbi_momentum_10d": "0",
            "spy_momentum_10d": "0",
            "xbi_realized_vol_20d": "0",
            "feeds": {"vix": "failed", "xbi": "failed"},
        }
        ref = date(2026, 6, 9)
        with tempfile.TemporaryDirectory() as tmp:
            p = _write_snap(stale, tmp)
            result = check_freshness(p, reference_date=ref)
        assert result["ok"] is False
        # Must surface the stale + VIX=0 + all-zero issues
        issue_text = " ".join(result["issues"])
        assert "trading days old" in issue_text
        assert "vix = 0" in issue_text

    def test_vix_zero_fails(self):
        snap = _valid_snap("2026-06-26")
        snap["vix"] = "0"
        with tempfile.TemporaryDirectory() as tmp:
            p = _write_snap(snap, tmp)
            result = check_freshness(p, reference_date=date(2026, 6, 26))
        assert result["ok"] is False
        assert any("vix = 0" in i for i in result["issues"])

    def test_all_signals_zero_fails(self):
        snap = _valid_snap("2026-06-26")
        for f in ("vix", "xbi_vs_spy_30d", "xbi_momentum_10d", "spy_momentum_10d", "xbi_realized_vol_20d"):
            snap[f] = "0"
        with tempfile.TemporaryDirectory() as tmp:
            p = _write_snap(snap, tmp)
            result = check_freshness(p, reference_date=date(2026, 6, 26))
        assert result["ok"] is False
        issue_text = " ".join(result["issues"])
        assert "0.0" in issue_text or "vix = 0" in issue_text

    def test_missing_snapshot_file_fails(self):
        result = check_freshness(
            "/nonexistent/path/market_snapshot.json",
            reference_date=date(2026, 6, 26),
        )
        assert result["ok"] is False
        assert any("not found" in i for i in result["issues"])

    def test_malformed_json_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "market_snapshot.json"
            p.write_text("{ not valid json }")
            result = check_freshness(p, reference_date=date(2026, 6, 26))
        assert result["ok"] is False
        assert any("parse" in i for i in result["issues"])

    def test_missing_as_of_date_fails(self):
        snap = _valid_snap("2026-06-26")
        del snap["provenance"]["as_of_date"]
        with tempfile.TemporaryDirectory() as tmp:
            p = _write_snap(snap, tmp)
            result = check_freshness(p, reference_date=date(2026, 6, 26))
        assert result["ok"] is False
        assert any("as_of_date" in i for i in result["issues"])

    def test_custom_max_stale_days(self):
        """Caller can override the stale threshold."""
        ref = date(2026, 6, 26)  # Friday
        with tempfile.TemporaryDirectory() as tmp:
            p = _write_snap(_valid_snap("2026-06-23"), tmp)  # Tuesday — 3d gap
            result_strict = check_freshness(p, reference_date=ref, max_stale_trading_days=2)
            result_loose = check_freshness(p, reference_date=ref, max_stale_trading_days=5)
        assert result_strict["ok"] is False
        assert result_loose["ok"] is True

    def test_returns_feed_status(self):
        """check_freshness must surface the feeds dict from the snapshot."""
        snap = _valid_snap("2026-06-26")
        snap["feeds"] = {"vix": "live", "xbi": "failed", "spy": "live"}
        with tempfile.TemporaryDirectory() as tmp:
            p = _write_snap(snap, tmp)
            result = check_freshness(p, reference_date=date(2026, 6, 26))
        assert result["feeds"]["xbi"] == "failed"

    def test_reconstructed_pit_feeds_count_as_present(self):
        feeds = {
            "vix": "pit",
            "biotech_fund_flows": "pit_volume_proxy",
            "spy": "failed",
            "xbi": "live",
        }
        assert count_present_feeds(feeds) == 3

    def test_emit_diagnostic_does_not_raise(self):
        """emit_diagnostic must survive any result shape, including all-null."""
        emit_diagnostic(
            {
                "ok": True,
                "age_trading_days": 0,
                "snapshot_as_of_date": "2026-06-26",
                "reference_date": "2026-06-26",
                "issues": [],
                "feeds": {},
                "signal_fields": {},
            }
        )
        emit_diagnostic(
            {
                "ok": False,
                "age_trading_days": None,
                "snapshot_as_of_date": None,
                "reference_date": "2026-06-26",
                "issues": ["file not found"],
                "feeds": {},
                "signal_fields": {},
            }
        )
