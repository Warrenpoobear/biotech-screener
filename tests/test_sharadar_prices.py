"""Sharadar as the production price source (common/sharadar_prices.py)."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from common.sharadar_prices import (  # noqa: E402
    PRICE_COLUMNS,
    fetch_bars,
    last_dates,
    refresh_prices_from_sharadar,
    to_bar,
    update_market_caps_from_daily,
    upsert_price_history,
)


class FakeClient:
    """Serves canned rows per table; records requests."""

    def __init__(self, tables, fail=False):
        self.tables = tables
        self.calls = []
        self.fail = fail

    def fetch(self, table, *, tickers=None, date_range=None, fields=None, **_):
        self.calls.append((table, tuple(tickers or ()), date_range))
        if self.fail:
            raise RuntimeError("network down")
        lo, hi = date_range
        want = {t.upper() for t in (tickers or [])}
        return [r for r in self.tables.get(table, []) if r["ticker"] in want and lo <= r["date"] <= hi]


def sep(t, d, close, unadj=None, **kw):
    return {
        "ticker": t,
        "date": d,
        "open": kw.get("open", close),
        "high": kw.get("high", close),
        "low": kw.get("low", close),
        "close": close,
        "closeunadj": unadj if unadj is not None else close,
        "volume": kw.get("volume", 1000),
    }


def write_prices(path: Path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(PRICE_COLUMNS)
        for r in rows:
            w.writerow(r)


def read_prices(path: Path):
    with open(path, newline="", encoding="utf-8") as fh:
        return {(r["ticker"], r["date"]): r for r in csv.DictReader(fh)}


# --- conventions -------------------------------------------------------------
def test_bar_is_split_adjusted_not_raw():
    """Split-adjusted close 50 (raw 100 before a 2:1 split) -> 50: the series must be
    continuous, because the repair step only detects >+300% / <-75% jumps."""
    b = to_bar(sep("ACME", "2026-01-02", 50.0, unadj=100.0, open=49.0, high=51.0, low=48.0, volume=2000))
    assert (b["close"], b["open"], b["high"], b["low"], b["volume"]) == (50.0, 49.0, 51.0, 48.0, 2000)


def test_bar_rejects_missing_close():
    assert to_bar({"ticker": "X", "date": "2026-01-02", "close": None}) is None


# --- upsert ------------------------------------------------------------------
def test_upsert_revises_provisional_close_and_appends(tmp_path):
    p = tmp_path / "price_history.csv"
    write_prices(p, [["2026-11-09", "ABVX", "87.0", "", "", "", ""], ["2026-11-10", "ABVX", "91.37", "", "", "", ""]])
    stats = upsert_price_history(p, [to_bar(sep("ABVX", "2026-11-10", 90.0)), to_bar(sep("ABVX", "2026-11-11", 92.0))])
    rows = read_prices(p)
    assert float(rows[("ABVX", "2026-11-10")]["close"]) == 90.0  # official close replaced the intraday one
    assert ("ABVX", "2026-11-11") in rows
    assert float(rows[("ABVX", "2026-11-09")]["close"]) == 87.0  # untouched
    assert stats["n_revised"] == 1 and stats["n_appended"] == 1


def test_upsert_is_noop_within_tolerance(tmp_path):
    p = tmp_path / "price_history.csv"
    write_prices(p, [["2026-11-10", "AMGN", "465.2300109863281", "", "", "", ""]])
    before = p.read_bytes()
    stats = upsert_price_history(p, [to_bar(sep("AMGN", "2026-11-10", 465.23))])
    assert stats["n_revised"] == 0 and p.read_bytes() == before


def test_upsert_does_not_write_through_hard_links(tmp_path):
    p = tmp_path / "price_history.csv"
    write_prices(p, [["2026-11-10", "ABVX", "91.0", "", "", "", ""]])
    staged = tmp_path / "staged.csv"
    os.link(p, staged)
    upsert_price_history(p, [to_bar(sep("ABVX", "2026-11-10", 90.0))])
    assert "91.0" in staged.read_text()


# --- fetch / refresh ---------------------------------------------------------
def test_etfs_fall_back_to_sfp():
    c = FakeClient({"SEP": [sep("ABVX", "2026-11-10", 90.0)], "SFP": [sep("XBI", "2026-11-10", 100.0)]})
    bars, missing = fetch_bars(c, ["ABVX", "XBI", "NOPE"], "2026-11-01", "2026-11-10")
    assert {b["ticker"] for b in bars} == {"ABVX", "XBI"}
    assert missing == ["NOPE"]
    assert [call[0] for call in c.calls] == ["SEP", "SFP"]


def test_refresh_backfills_new_tickers_and_windows_existing(tmp_path):
    p = tmp_path / "price_history.csv"
    write_prices(p, [["2026-11-09", "ABVX", "87.0", "", "", "", ""]])
    c = FakeClient(
        {"SEP": [sep("ABVX", "2026-11-10", 90.0), sep("NEW", "2021-06-01", 5.0), sep("NEW", "2026-11-10", 6.0)]}
    )
    stats = refresh_prices_from_sharadar(p, "2026-11-10", ["ABVX", "NEW"], client=c)
    assert stats["status"] == "OK" and stats["has_through_date"]
    assert last_dates(p) == {"ABVX": "2026-11-10", "NEW": "2026-11-10"}
    windows = {call[1]: call[2] for call in c.calls}
    assert windows[("ABVX",)][0] == "2026-10-31"  # trailing 10-day window
    assert windows[("NEW",)][0] == "2020-01-01"  # full backfill for a new ticker


def test_refresh_never_raises_and_leaves_file_for_yfinance(tmp_path):
    p = tmp_path / "price_history.csv"
    write_prices(p, [["2026-11-09", "ABVX", "87.0", "", "", "", ""]])
    before = p.read_bytes()
    stats = refresh_prices_from_sharadar(p, "2026-11-10", ["ABVX"], client=FakeClient({}, fail=True))
    assert stats["status"] == "ERROR" and "network down" in stats["detail"]
    assert p.read_bytes() == before


def test_refresh_reports_partial_when_today_not_published(tmp_path):
    p = tmp_path / "price_history.csv"
    write_prices(p, [["2026-11-09", "ABVX", "87.0", "", "", "", ""]])
    c = FakeClient({"SEP": [sep("ABVX", "2026-11-10", 90.0)]})
    stats = refresh_prices_from_sharadar(p, "2026-11-11", ["ABVX"], client=c)
    assert stats["latest_date"] == "2026-11-10" and stats["has_through_date"] is False


# --- market caps ---------------------------------------------------------------
def test_daily_market_cap_millions_become_usd(tmp_path):
    md = tmp_path / "market_data.json"
    md.write_text(json.dumps([{"ticker": "ABVX", "price": 90.0, "market_cap": 1.0, "collected_at": "2026-11-10"}]))
    c = FakeClient({"DAILY": [{"ticker": "ABVX", "date": "2026-11-10", "marketcap": 7065.6, "ev": 6622.0}]})
    stats = update_market_caps_from_daily(md, "2026-11-10", client=c)
    rec = json.loads(md.read_text())[0]
    assert stats["n_updated"] == 1
    assert rec["market_cap"] == pytest.approx(7_065_600_000.0)
    assert rec["enterprise_value"] == pytest.approx(6_622_000_000.0)
    assert rec["market_cap_source"] == "sharadar_daily"


def test_daily_missing_marketcap_leaves_existing_cap(tmp_path):
    """Ported from the retired scripts/refresh_from_sharadar.py tests: a DAILY row
    with no marketcap/ev must not blank or rescale the existing raw-USD values."""
    md = tmp_path / "market_data.json"
    md.write_text(json.dumps([{"ticker": "X", "price": 5.0, "market_cap": 5e8, "enterprise_value": 4e8}]))
    c = FakeClient({"DAILY": [{"ticker": "X", "date": "2026-11-10", "marketcap": None, "ev": None}]})
    update_market_caps_from_daily(md, "2026-11-10", client=c)
    rec = json.loads(md.read_text())[0]
    assert rec["market_cap"] == 5e8
    assert rec["enterprise_value"] == 4e8


# --- wrapper switch --------------------------------------------------------------
def test_wrapper_switch(monkeypatch):
    from run_daily_production import sharadar_price_source_enabled

    monkeypatch.delenv("BIOTECH_PRICE_SOURCE", raising=False)
    monkeypatch.setenv("SHARADAR_API_KEY", "x" * 32)  # pragma: allowlist secret
    assert sharadar_price_source_enabled() is True
    monkeypatch.setenv("BIOTECH_PRICE_SOURCE", "yfinance")
    assert sharadar_price_source_enabled() is False
    monkeypatch.delenv("BIOTECH_PRICE_SOURCE")
    monkeypatch.delenv("SHARADAR_API_KEY")
    assert sharadar_price_source_enabled() is False
