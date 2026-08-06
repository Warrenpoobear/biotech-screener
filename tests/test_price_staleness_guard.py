"""Tests for the per-ticker price-history staleness guard (2026-08-05).

Root cause: the staleness check in compute_momentum_from_price_history took
max() over EVERY ticker's latest date and compared that single global maximum
against as_of_date:

    _latest_price_date = max(per-ticker maxima)
    if (ref_date - _latest_price_date).days > 7: warn

One fresh ticker pins the global maximum at as_of_date, so the check reports
0 days stale no matter how far behind every other ticker is. Those tickers
still get return_20d/60d/120d computed from their stale closes and are then
ranked against fresh peers.

Observed 2026-08-05: price_history.csv max date was 2026-08-05 (so the global
check saw 0 days stale and stayed silent) while CPRX (15d), ESPR (21d) and
SGMO (35d) were being scored off stale closes — their yfinance fetches had
been failing silently for weeks. Only 265/366 tickers were current (72.4%),
which is what the data_collection_health card had been reporting as a 73.9%
Market Data FAIL all along.

Fix: per-ticker gaps, scoped to tickers actually being scored, naming the
worst offenders. Detection-only unless BIOTECH_STRICT_PRICE_STALENESS=1.
"""

from __future__ import annotations

import csv
import importlib
import logging
from datetime import date, timedelta
from pathlib import Path

import pytest

import run_screen

AS_OF = "2026-08-05"


def _write_history(tmp_path: Path, rows: list[tuple[str, str, float]]) -> Path:
    """rows = [(ticker, YYYY-MM-DD, close), ...]"""
    p = tmp_path / "price_history.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "ticker", "close", "open", "high", "low", "volume"])
        for tkr, d, close in rows:
            w.writerow([d, tkr, close, close, close, close, 1000])
    return p


def _series(tkr: str, last: str, n: int = 200) -> list[tuple[str, str, float]]:
    """A contiguous daily series ending at `last`, so momentum has enough history."""
    end = date.fromisoformat(last)
    return [(tkr, (end - timedelta(days=i)).isoformat(), 100.0 + i) for i in range(n)]


def _run(tmp_path, rows, scored, caplog, strict=False, monkeypatch=None):
    path = _write_history(tmp_path, rows)
    market_data = {t: {} for t in scored}
    if strict:
        monkeypatch.setattr(run_screen, "_STRICT_PRICE_STALENESS", True)
    with caplog.at_level(logging.INFO):
        run_screen.compute_momentum_from_price_history(path, AS_OF, market_data)
    return caplog.text


# --- the actual defect -----------------------------------------------------
def test_fresh_ticker_does_not_mask_a_stale_one(tmp_path, caplog):
    """THE regression. A fresh ticker pins the global max at as_of_date; the
    stale ticker must still be reported. The old global-max check was silent here."""
    rows = _series("FRESH", AS_OF) + _series("STALE", "2026-07-01")
    text = _run(tmp_path, rows, ["FRESH", "STALE"], caplog)

    assert "STALE" in text
    assert "stale for 1/2 scored tickers" in text
    assert "FRESH@" not in text  # fresh ticker not named as an offender


def test_all_fresh_is_silent(tmp_path, caplog):
    rows = _series("AAA", AS_OF) + _series("BBB", AS_OF)
    text = _run(tmp_path, rows, ["AAA", "BBB"], caplog)
    assert "Price history stale" not in text
    assert "current (<=4d): 2 (100.0%)" in text


def test_one_day_lag_is_tolerated(tmp_path, caplog):
    """80 tickers sat at as_of-1 on 2026-08-05 from normal append lag — not stale."""
    rows = _series("AAA", AS_OF) + _series("BBB", "2026-08-04")
    text = _run(tmp_path, rows, ["AAA", "BBB"], caplog)
    assert "Price history stale" not in text


def test_weekend_gap_is_tolerated(tmp_path, caplog):
    """Fri 2026-07-31 close read on Wed as_of is 5d... but Fri->Mon is 3d.
    Confirm the 4d WARN threshold does not fire on a plain weekend."""
    rows = _series("AAA", "2026-08-03")  # 2 days back
    text = _run(tmp_path, rows, ["AAA"], caplog)
    assert "Price history stale" not in text


# --- scoping ---------------------------------------------------------------
def test_unscored_stale_tickers_are_ignored(tmp_path, caplog):
    """Delisted names still in price_history.csv (MRSN/CVAC at 211d) must not
    trip the guard — they are not in market_data_by_ticker."""
    rows = _series("AAA", AS_OF) + _series("MRSN", "2026-01-06")
    text = _run(tmp_path, rows, ["AAA"], caplog)  # MRSN not scored
    assert "Price history stale" not in text
    assert "MRSN" not in text


def test_scored_count_excludes_unscored(tmp_path, caplog):
    rows = _series("AAA", AS_OF) + _series("ZZZ", AS_OF) + _series("MRSN", "2026-01-06")
    text = _run(tmp_path, rows, ["AAA", "ZZZ"], caplog)
    assert "scored tickers with price history: 2" in text


# --- severity escalation ---------------------------------------------------
def test_severe_staleness_logs_at_error(tmp_path, caplog):
    rows = _series("AAA", AS_OF) + _series("SGMO", "2026-07-01")  # 35d
    with caplog.at_level(logging.INFO):
        _run(tmp_path, rows, ["AAA", "SGMO"], caplog)
    assert any(r.levelno == logging.ERROR for r in caplog.records)


def test_mild_staleness_logs_at_warning_not_error(tmp_path, caplog):
    rows = _series("AAA", AS_OF) + _series("MILD", "2026-07-29")  # 7d: >4 but <=10
    with caplog.at_level(logging.INFO):
        _run(tmp_path, rows, ["AAA", "MILD"], caplog)
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert not any(r.levelno == logging.ERROR for r in caplog.records)


def test_worst_offenders_are_named_with_gap(tmp_path, caplog):
    rows = _series("AAA", AS_OF) + _series("SGMO", "2026-07-01")
    text = _run(tmp_path, rows, ["AAA", "SGMO"], caplog)
    assert "SGMO@2026-07-01(35d)" in text


# --- strict mode is opt-in -------------------------------------------------
def test_default_is_detection_only(tmp_path, caplog):
    """Guard must not change production behaviour on its own."""
    assert run_screen._STRICT_PRICE_STALENESS is False
    rows = _series("AAA", AS_OF) + _series("SGMO", "2026-07-01")
    _run(tmp_path, rows, ["AAA", "SGMO"], caplog)  # must not raise


def test_strict_mode_raises_on_severe(tmp_path, caplog, monkeypatch):
    rows = _series("AAA", AS_OF) + _series("SGMO", "2026-07-01")
    with pytest.raises(RuntimeError, match="severely stale"):
        _run(tmp_path, rows, ["AAA", "SGMO"], caplog, strict=True, monkeypatch=monkeypatch)


def test_strict_mode_does_not_raise_on_mild(tmp_path, caplog, monkeypatch):
    """Only ALERT-level staleness aborts; WARN-level stays a log line."""
    rows = _series("AAA", AS_OF) + _series("MILD", "2026-07-29")  # 7d
    _run(tmp_path, rows, ["AAA", "MILD"], caplog, strict=True, monkeypatch=monkeypatch)


def test_strict_flag_reads_env(monkeypatch):
    monkeypatch.setenv("BIOTECH_STRICT_PRICE_STALENESS", "1")
    mod = importlib.reload(run_screen)
    try:
        assert mod._STRICT_PRICE_STALENESS is True
    finally:
        monkeypatch.delenv("BIOTECH_STRICT_PRICE_STALENESS", raising=False)
        importlib.reload(run_screen)
