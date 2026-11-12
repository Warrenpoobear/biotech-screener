"""patch_market_data_prices must not overwrite collected_at.

collected_at is when collect_market_data.py last fetched the row's full field
set (avg_volume, beta, volatility_90d, short_percent, shares ...). The price
patch only refreshes ``price``; stamping collected_at with the close date made
months-old rows look fresh to the market_data_staleness gate (2026-11-12: 338 of
344 Feb-dated rows would have been re-stamped). The patch records its own date
in ``price_date`` instead, like the Sharadar updater's ``market_cap_date``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load_patch_module():
    spec = importlib.util.spec_from_file_location(
        "patch_market_data_prices", REPO / "tools" / "patch_market_data_prices.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _setup(tmp_path, monkeypatch):
    mod = _load_patch_module()
    md = tmp_path / "market_data.json"
    md.write_text(
        json.dumps(
            [
                {"ticker": "AAA", "price": 5.0, "avg_volume": 100, "collected_at": "2026-02-19"},
                {"ticker": "BBB", "price": 7.0, "avg_volume": 200, "collected_at": "2026-11-10"},
            ]
        ),
        encoding="utf-8",
    )
    ph = tmp_path / "price_history.csv"
    ph.write_text("date,ticker,close\n2026-11-11,AAA,6.5\n2026-11-11,BBB,7.0\n", encoding="utf-8")
    monkeypatch.setattr(mod, "MARKET_DATA", md)
    monkeypatch.setattr(mod, "PRICE_HISTORY", ph)
    return mod, md


def test_patch_updates_price_but_keeps_collected_at(tmp_path, monkeypatch):
    mod, md = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["patch_market_data_prices.py", "--as-of-date", "2026-11-12"])
    mod.main()
    rows = {r["ticker"]: r for r in json.loads(md.read_text(encoding="utf-8"))}
    assert rows["AAA"]["price"] == 6.5
    assert rows["AAA"]["collected_at"] == "2026-02-19"  # full-field collection date untouched
    assert rows["AAA"]["price_date"] == "2026-11-11"
    assert rows["AAA"]["price_source"] == "price_history"
    assert rows["AAA"]["avg_volume"] == 100


def test_unchanged_price_still_records_price_date(tmp_path, monkeypatch):
    mod, md = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["patch_market_data_prices.py", "--as-of-date", "2026-11-12"])
    mod.main()
    rows = {r["ticker"]: r for r in json.loads(md.read_text(encoding="utf-8"))}
    assert rows["BBB"]["collected_at"] == "2026-11-10"
    assert rows["BBB"]["price_date"] == "2026-11-11"  # price confirmed current as of this close


def test_patched_file_still_fails_staleness_gate(tmp_path, monkeypatch):
    """End to end: Feb rows stay Feb after the patch, so the gate triggers a refresh."""
    sys.path.insert(0, str(REPO / "tools"))
    sys.path.insert(0, str(REPO))
    from run_daily_production import check_market_data_staleness

    mod, md = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["patch_market_data_prices.py", "--as-of-date", "2026-11-12"])
    mod.main()
    result = check_market_data_staleness(tmp_path, "2026-11-12", max_age_days=5)
    assert result.status == "FAIL"
    assert "2026-02-19" in result.detail
