"""refresh_from_sharadar must use the same price convention as the pipeline.

common/sharadar_prices.py (the nightly path, PR #622) writes Sharadar SEP
``close``: split-adjusted, NOT dividend-adjusted. ``closeadj`` (split+dividend)
is explicitly not used, because price_history.csv feeds momentum and
price_history_split_adj.csv, and dividends are back-adjusted separately from the
corporate-actions registry (#640). The manual script wrote ``closeadj``, so a
run would silently rewrite every past dividend payer's history (e.g. SUPN's
$8.13 special dividend) and set market_data price from it.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _load():
    spec = importlib.util.spec_from_file_location(
        "refresh_from_sharadar", REPO / "scripts" / "refresh_from_sharadar.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# SUPN-like: special dividend ex 2026-11-03, so closeadj < close before it.
SEP = [
    {
        "ticker": "SUPN",
        "date": "2026-10-31",
        "open": 50.0,
        "high": 51.0,
        "low": 49.0,
        "close": 50.0,
        "closeadj": 41.87,
        "volume": 1000.0,
    },
    {
        "ticker": "SUPN",
        "date": "2026-11-11",
        "open": 42.0,
        "high": 43.0,
        "low": 41.5,
        "close": 42.5,
        "closeadj": 42.4,
        "volume": 2000.0,
    },
]


def test_price_history_uses_split_adjusted_close_not_closeadj(tmp_path):
    mod = _load()
    out = tmp_path / "price_history.csv"
    mod._write_price_history(SEP, out)
    rows = {r["date"]: r for r in csv.DictReader(out.open(encoding="utf-8"))}
    assert float(rows["2026-10-31"]["close"]) == 50.0
    assert float(rows["2026-11-11"]["close"]) == 42.5


def test_market_data_price_uses_close_not_closeadj(tmp_path):
    mod = _load()
    md = tmp_path / "market_data.json"
    md.write_text(json.dumps([{"ticker": "SUPN", "price": 40.0, "collected_at": "2026-11-12"}]), encoding="utf-8")
    mod._update_market_data({}, SEP, tmp_path)
    rec = json.loads(md.read_text(encoding="utf-8"))[0]
    assert rec["price"] == 42.5
    assert rec["price_date"] == "2026-11-11"


def test_bars_match_pipeline_converter(tmp_path):
    """Same rows in, same CSV values out as common.sharadar_prices.to_bar."""
    from common.sharadar_prices import to_bar

    mod = _load()
    out = tmp_path / "price_history.csv"
    mod._write_price_history(SEP, out)
    got = list(csv.DictReader(out.open(encoding="utf-8")))
    want = [to_bar(r) for r in SEP]
    assert [(g["date"], float(g["close"]), g["volume"]) for g in got] == [
        (w["date"], w["close"], str(w["volume"])) for w in want
    ]
