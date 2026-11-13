"""Point-in-time market data for historical runs.

Classification: RESEARCH_TOOLING_PIT_MARKET_DATA_NO_MODEL_CHANGE

The leakage test is the point of this file: a record built for a past date must not
change when prices or filings after that date change.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.pit_market_data import (
    NO_PIT_SOURCE_FIELDS,
    SOURCE_NEXT_FILING,
    SOURCE_PIT,
    SOURCE_SCALED,
    PitMarketDataBuilder,
    PriceData,
    build_market_records,
    latest_fact,
    month_end_ladder,
    pit_share_count,
)


def _daily(start_year: int, end_year: int):
    days = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            for d in (5, 15, 25, 28):
                days.append(f"{y}-{m:02d}-{d:02d}")
    return days


def _prices(path_by_ticker, split=None):
    """Build PriceData. ``split`` = (ticker, date, factor): raw price multiplies by factor from date."""
    raw, adj = {}, {}
    for ticker, fn in path_by_ticker.items():
        raw[ticker], adj[ticker] = [], []
        for day in _daily(2020, 2024):
            px = fn(day)
            if px is None:
                continue
            adj_px = px
            raw_px = px
            if split and split[0] == ticker and day < split[1]:
                raw_px = px / split[2]  # before a 1:N reverse split the as-traded price was lower
            raw[ticker].append((day, raw_px, 1000.0))
            adj[ticker].append((day, adj_px))
    return PriceData(raw, adj)


def _fact(end, filed, val):
    return {"end": end, "filed": filed, "val": val, "form": "10-Q"}


CURRENT = [{"ticker": "AAA", "market_cap": 2000.0, "shares_outstanding": 100, "price": 20.0, "short_percent": 0.1}]


def _aaa_price(day):
    return 10.0 if day < "2023-01-01" else 20.0


def test_latest_fact_uses_filing_date_not_period_end():
    facts = {"shares_outstanding": [_fact("2021-12-31", "2022-03-01", 50), _fact("2022-03-31", "2022-05-10", 80)]}
    assert latest_fact(facts, "shares_outstanding", "2022-04-15")["val"] == 50  # Q1 not filed yet
    assert latest_fact(facts, "shares_outstanding", "2022-05-10")["val"] == 80
    assert latest_fact(facts, "shares_outstanding", "2022-02-01") is None


def test_market_cap_is_pit_shares_times_close_on_the_date():
    prices = _prices({"AAA": _aaa_price})
    facts = {
        "AAA": {"shares_outstanding": [_fact("2021-12-31", "2022-03-01", 80), _fact("2023-12-31", "2024-03-01", 100)]}
    }
    records, counts = build_market_records("2022-06-28", CURRENT, prices, facts)
    rec = records[0]
    assert rec["market_cap"] == pytest.approx(800.0)  # 80 shares x 10.00, not today's 2000
    assert rec["market_cap_source"] == SOURCE_PIT
    assert rec["price"] == 10.0 and rec["price_date"] == "2022-06-28"
    assert rec["collected_at"] == "2022-06-28"
    assert counts[SOURCE_PIT] == 1


def test_no_point_in_time_source_fields_are_blanked():
    prices = _prices({"AAA": _aaa_price})
    records, _ = build_market_records("2022-06-28", CURRENT, prices, {})
    assert all(records[0][name] is None for name in NO_PIT_SOURCE_FIELDS)


def test_leakage_later_prices_and_filings_do_not_change_a_past_record():
    facts_then = {"AAA": {"shares_outstanding": [_fact("2021-12-31", "2022-03-01", 80)]}}
    facts_later = {
        "AAA": {"shares_outstanding": [_fact("2021-12-31", "2022-03-01", 80), _fact("2023-12-31", "2024-03-01", 100)]}
    }
    base, _ = build_market_records("2022-06-28", CURRENT, _prices({"AAA": _aaa_price}), facts_then)
    # A different future: the stock triples after the date, more shares are filed, today's cap is higher.
    moon = _prices({"AAA": lambda d: 10.0 if d < "2023-01-01" else 60.0})
    current_moon = [dict(CURRENT[0], market_cap=6000.0, price=60.0)]
    alt, _ = build_market_records("2022-06-28", current_moon, moon, facts_later)
    for key in (
        "market_cap",
        "price",
        "avg_volume",
        "52w_high",
        "52w_low",
        "volatility_90d",
        "returns_1m",
        "returns_3m",
    ):
        assert alt[0][key] == pytest.approx(base[0][key]), key


def test_ticker_not_trading_on_the_date_is_left_out():
    prices = _prices({"AAA": lambda d: None if d < "2022-01-01" else 10.0})
    records, counts = build_market_records("2021-06-28", CURRENT, prices, {})
    assert records == [] and counts["not_trading"] == 1


def test_no_share_fact_falls_back_to_price_scaled_current_cap():
    prices = _prices({"AAA": _aaa_price})
    records, _ = build_market_records("2022-06-28", CURRENT, prices, {})
    assert records[0]["market_cap_source"] == SOURCE_SCALED
    assert records[0]["market_cap"] == pytest.approx(2000.0 * 10.0 / 20.0)


def test_reverse_split_between_fact_and_date_restates_the_share_count():
    # 1:10 reverse split on 2022-06-01: 1000 old shares at 1.00 become 100 shares at 10.00.
    prices = _prices({"AAA": lambda d: 10.0}, split=("AAA", "2022-06-01", 10.0))
    facts = {"shares_outstanding": [_fact("2022-03-31", "2022-05-10", 1000)]}
    assert pit_share_count(facts, prices, "AAA", "2022-05-15") == pytest.approx(1000)
    assert pit_share_count(facts, prices, "AAA", "2022-06-28") == pytest.approx(100)


def test_foreign_filer_share_basis_is_not_used():
    # Filed ordinary-share count is ~1% of the ADS-equivalent count at every date.
    prices = _prices({"AAA": lambda d: 20.0})
    facts = {"AAA": {"shares_outstanding": [_fact(f"{y}-12-31", f"{y + 1}-03-01", 1) for y in range(2019, 2024)]}}
    records, _ = build_market_records("2022-06-28", CURRENT, prices, facts)
    assert records[0]["market_cap_source"] == SOURCE_SCALED
    assert records[0]["market_cap"] == pytest.approx(2000.0)


def test_pre_listing_share_count_takes_the_next_filing():
    # Until the first post-IPO 10-Q is filed (2022-08-10) the only fact is a tiny pre-conversion count.
    prices = _prices({"AAA": lambda d: None if d < "2022-05-01" else 20.0})
    facts = {
        "AAA": {
            "shares_outstanding": [_fact("2021-12-31", "2022-03-20", 2), _fact("2022-06-30", "2022-08-10", 90)]
            + [_fact(f"{y}-12-31", f"{y + 1}-03-01", 100) for y in (2022, 2023)]
        }
    }
    records, _ = build_market_records("2022-06-28", CURRENT, prices, facts)
    assert records[0]["market_cap_source"] == SOURCE_NEXT_FILING
    assert records[0]["market_cap"] == pytest.approx(90 * 20.0)


def test_month_end_ladder():
    assert month_end_ladder("2019-12-31", "2020-03-15") == ["2019-12-31", "2020-01-31", "2020-02-29"]


def test_builder_writes_market_data_json(tmp_path: Path):
    prod = tmp_path / "production_data"
    (prod / "pit_financials").mkdir(parents=True)
    (prod / "market_data.json").write_text(json.dumps(CURRENT))
    rows = ["date,ticker,close,open,high,low,volume"] + [
        f"{d},AAA,{_aaa_price(d)},0,0,0,1000" for d in _daily(2020, 2024)
    ]
    (prod / "price_history.csv").write_text("\n".join(rows) + "\n")
    (prod / "price_history_split_adj.csv").write_text(
        "ticker,date,close\n" + "\n".join(f"AAA,{d},{_aaa_price(d)}" for d in _daily(2020, 2024)) + "\n"
    )
    (prod / "pit_financials" / "AAA.json").write_text(
        json.dumps({"ticker": "AAA", "facts": {"shares_outstanding": [_fact("2021-12-31", "2022-03-01", 80)]}})
    )
    out = tmp_path / "staging" / "2022-06-28" / "market_data.json"
    counts = PitMarketDataBuilder(prod).write("2022-06-28", out)
    written = json.loads(out.read_text())
    assert counts == {SOURCE_PIT: 1}
    assert written[0]["market_cap"] == pytest.approx(800.0)
    assert not out.with_suffix(".json.tmp").exists()
