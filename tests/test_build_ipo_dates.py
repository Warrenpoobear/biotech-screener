"""tools/build_ipo_dates.py: first dates must not move later, truncated histories are refused.

Defect (2026-11-04, PR #606): ipo_dates.json was regenerated from a price
history that only went back two years. 351 of 373 tickers got a
first_price_date in 2024-11, so every historical survivorship filter dropped
almost the whole universe before that date. Fixtures are synthetic.
"""

import csv
import json
from pathlib import Path

import pytest

import tools.build_ipo_dates as bid


def _write_prices(path: Path, spans: dict) -> None:
    """spans: {ticker: (first_date, last_date)} -> two price rows per ticker."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "ticker", "close"])
        for ticker, (first, last) in spans.items():
            w.writerow([first, ticker, "10.0"])
            w.writerow([last, ticker, "11.0"])


def _existing(tickers: dict) -> dict:
    return {
        "generated_at": "2026-07-02T00:00:00+00:00",
        "source": "test",
        "ticker_count": len(tickers),
        "tickers": {t: {"first_price_date": f, "last_price_date": lst} for t, (f, lst) in tickers.items()},
    }


def _varied(n: int = 40) -> dict:
    return {f"T{i:02d}": (f"20{10 + i % 12:02d}-0{1 + i % 9}-15", "2026-11-11") for i in range(n)}


def test_first_and_last_dates_come_from_the_price_history(tmp_path):
    prices = tmp_path / "price_history.csv"
    _write_prices(prices, {"AAA": ("2013-01-02", "2026-11-11"), "BBB": ("2025-02-13", "2026-11-10")})
    result = bid.build_ipo_dates(prices)
    assert result["tickers"]["AAA"] == {"first_price_date": "2013-01-02", "last_price_date": "2026-11-11"}
    assert result["tickers"]["BBB"] == {"first_price_date": "2025-02-13", "last_price_date": "2026-11-10"}
    assert result["ticker_count"] == 2


def test_merge_never_moves_a_first_date_later(tmp_path):
    prices = tmp_path / "price_history.csv"
    # A price file that only goes back to 2024-11 for a name listed since 2013.
    _write_prices(prices, {"AAA": ("2024-11-04", "2026-11-11")})
    built = bid.build_ipo_dates(prices)
    merged, moved = bid.merge_with_existing(built, _existing({"AAA": ("2013-01-02", "2026-11-03")}))
    assert merged["tickers"]["AAA"]["first_price_date"] == "2013-01-02"
    assert merged["tickers"]["AAA"]["last_price_date"] == "2026-11-11"
    assert moved == {"first_kept_earlier": 1, "first_moved_earlier": 0, "last_kept_later": 0, "dropped": 0}


def test_merge_takes_an_earlier_first_date_and_never_moves_last_date_back(tmp_path):
    prices = tmp_path / "price_history.csv"
    _write_prices(prices, {"AAA": ("2013-01-02", "2026-10-01")})
    built = bid.build_ipo_dates(prices)
    merged, moved = bid.merge_with_existing(built, _existing({"AAA": ("2024-11-04", "2026-11-03")}))
    assert merged["tickers"]["AAA"] == {"first_price_date": "2013-01-02", "last_price_date": "2026-11-03"}
    assert moved["first_moved_earlier"] == 1 and moved["last_kept_later"] == 1


def test_merge_keeps_tickers_missing_from_the_price_file(tmp_path):
    prices = tmp_path / "price_history.csv"
    _write_prices(prices, {"AAA": ("2013-01-02", "2026-11-11")})
    built = bid.build_ipo_dates(prices)
    merged, moved = bid.merge_with_existing(built, _existing({"GONE": ("2015-05-05", "2024-03-01")}))
    assert merged["tickers"]["GONE"] == {"first_price_date": "2015-05-05", "last_price_date": "2024-03-01"}
    assert merged["ticker_count"] == 2 and moved["dropped"] == 0


def test_truncated_history_is_detected():
    tickers = {f"T{i:02d}": {"first_price_date": "2024-11-04", "last_price_date": "2026-11-11"} for i in range(30)}
    tickers.update(
        {f"N{i:02d}": {"first_price_date": f"2025-0{i + 1}-10", "last_price_date": "2026-11-11"} for i in range(5)}
    )
    share, day = bid.first_date_concentration({"tickers": tickers})
    assert day == "2024-11-04" and share == pytest.approx(30 / 35)
    assert bid.looks_truncated({"tickers": tickers}) is True
    # The same concentration at the start of the price file itself is expected.
    floor = {t: {"first_price_date": "2020-01-02", "last_price_date": "2026-11-11"} for t in tickers}
    assert bid.looks_truncated({"tickers": floor}) is False
    assert (
        bid.looks_truncated(
            {"tickers": {t: {"first_price_date": f, "last_price_date": lst} for t, (f, lst) in _varied().items()}}
        )
        is False
    )


def test_main_refuses_a_truncated_history_and_leaves_the_file_alone(tmp_path):
    prices, out = tmp_path / "price_history.csv", tmp_path / "ipo_dates.json"
    _write_prices(prices, {f"T{i:02d}": ("2024-11-04", "2026-11-11") for i in range(40)})
    out.write_text(json.dumps(_existing({"OLD": ("2013-01-02", "2026-11-03")})), encoding="utf-8")
    before = out.read_text(encoding="utf-8")
    assert bid.main(["--price-history", str(prices), "--output", str(out), "--no-merge"]) == 2
    assert out.read_text(encoding="utf-8") == before


def test_main_writes_merged_result(tmp_path):
    prices, out = tmp_path / "price_history.csv", tmp_path / "ipo_dates.json"
    spans = _varied()
    spans["AAA"] = ("2024-11-04", "2026-11-11")
    _write_prices(prices, spans)
    out.write_text(json.dumps(_existing({"AAA": ("2013-01-02", "2026-11-03")})), encoding="utf-8")
    assert bid.main(["--price-history", str(prices), "--output", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["tickers"]["AAA"] == {"first_price_date": "2013-01-02", "last_price_date": "2026-11-11"}
    assert data["ticker_count"] == len(spans)
    assert not list(tmp_path.glob("*.tmp"))
