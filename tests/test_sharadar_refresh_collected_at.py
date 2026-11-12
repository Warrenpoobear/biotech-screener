"""refresh_from_sharadar must not overwrite collected_at in market_data.json.

Same defect as tools/patch_market_data_prices.py (fixed in #644): the script
updates only price, market_cap and enterprise_value, but stamped collected_at,
which the market_data_staleness gate (#643) reads as the date of the row's
full collect_market_data.py fetch (avg_volume, beta, volatility_90d, ...).
It now records price_date / price_source instead.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "refresh_from_sharadar", REPO / "scripts" / "refresh_from_sharadar.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(tmp_path: Path) -> Path:
    md = tmp_path / "market_data.json"
    md.write_text(
        json.dumps(
            [
                {"ticker": "AAA", "price": 5.0, "avg_volume": 100, "market_cap": 1e8, "collected_at": "2026-02-19"},
                {"ticker": "ZZZ", "price": 9.0, "avg_volume": 300, "market_cap": 2e8, "collected_at": "2026-02-19"},
            ]
        ),
        encoding="utf-8",
    )
    return md


SEP = [
    {"ticker": "AAA", "date": "2026-11-10", "close": 6.0, "closeadj": 6.0},
    {"ticker": "AAA", "date": "2026-11-11", "close": 6.5, "closeadj": 6.5},
]
DAILY = {"AAA": {"date": "2026-11-11", "marketcap": 150.0, "ev": 120.0}}


def test_update_keeps_collected_at_and_records_price_date(tmp_path):
    mod = _load()
    md = _write(tmp_path)
    mod._update_market_data(DAILY, SEP, tmp_path)
    rows = {r["ticker"]: r for r in json.loads(md.read_text(encoding="utf-8"))}
    a = rows["AAA"]
    assert a["price"] == 6.5
    assert a["collected_at"] == "2026-02-19"
    assert a["price_date"] == "2026-11-11"
    assert a["price_source"] == "sharadar_sep"
    assert a["market_cap"] == 150.0 * mod.SHARADAR_DAILY_UNIT_USD
    assert a["avg_volume"] == 100
    # a ticker with no SEP rows is untouched
    assert rows["ZZZ"] == {
        "ticker": "ZZZ",
        "price": 9.0,
        "avg_volume": 300,
        "market_cap": 2e8,
        "collected_at": "2026-02-19",
    }


def test_refreshed_file_still_fails_staleness_gate(tmp_path):
    from run_daily_production import check_market_data_staleness

    mod = _load()
    _write(tmp_path)
    mod._update_market_data(DAILY, SEP, tmp_path)
    result = check_market_data_staleness(tmp_path, "2026-11-12", max_age_days=5)
    assert result.status == "FAIL"
    assert "2026-02-19" in result.detail
