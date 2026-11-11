"""Regression test for market_implied_cross_section (Step 1.47 daily options IC).

run() normalises price_history dates to "YYYY-MM-DD" strings, so the pivoted
price index is string-typed. The function used to test membership with the
string but index with a pd.Timestamp, raising KeyError(Timestamp(...)) on every
production run that had forward data ("Options IC update failed (non-blocking)").
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.options_shadow_analysis import market_implied_cross_section  # noqa: E402


def _price_df(dates, tickers):
    rng = np.random.default_rng(0)
    rows = []
    for t_i, t in enumerate(tickers):
        px = 10.0 + t_i
        for d in dates:
            px *= 1 + rng.normal(0, 0.02)
            rows.append({"date": d, "ticker": t, "close": px})
    df = pd.DataFrame(rows)
    # Same normalisation as run(): string dates.
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def test_cross_section_with_string_date_index_does_not_raise():
    dates = pd.bdate_range("2026-10-01", periods=40)
    tickers = [f"T{i}" for i in range(12)]
    price_df = _price_df(dates, tickers)
    enrich_df = pd.DataFrame({"ticker": tickers, "rv_20d": np.linspace(0.2, 1.3, len(tickers))})

    snap_date = dates[5].strftime("%Y-%m-%d")
    out = market_implied_cross_section(enrich_df, price_df, snap_date)

    for fwd in ("fwd5d", "fwd10d", "fwd20d"):
        assert "status" not in out[fwd], out[fwd]
        assert out[fwd]["n_pairs"] == len(tickers)
        assert "rv_20d" in out[fwd]["correlations"]


def test_cross_section_reports_insufficient_forward_data():
    dates = pd.bdate_range("2026-10-01", periods=8)
    tickers = ["A", "B", "C"]
    price_df = _price_df(dates, tickers)
    enrich_df = pd.DataFrame({"ticker": tickers, "rv_20d": [0.5, 0.6, 0.7]})

    out = market_implied_cross_section(enrich_df, price_df, dates[5].strftime("%Y-%m-%d"))
    assert out["fwd20d"]["status"] == "insufficient_forward_data"


def test_cross_section_missing_snap_date_is_price_not_found():
    dates = pd.bdate_range("2026-10-01", periods=40)
    tickers = ["A", "B", "C"]
    price_df = _price_df(dates, tickers)
    enrich_df = pd.DataFrame({"ticker": tickers, "rv_20d": [0.5, 0.6, 0.7]})

    # A Saturday: not in the index, forward dates are.
    out = market_implied_cross_section(enrich_df, price_df, "2026-10-03")
    assert out["fwd5d"]["status"] == "price_not_found"
