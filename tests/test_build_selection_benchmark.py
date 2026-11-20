"""scripts/research/build_selection_benchmark.py — return arithmetic and reporting.

Defects fixed 2026-11-13:
  1. Period returns were summed, not compounded.
  2. The win rate was only counted on periods that had constructed-shadow data,
     so it read 0% whenever there was none.
  3. Prices came from the unadjusted price_history.csv.
  4. A snapshot date with no price row on that exact date dropped the period
     (and the next one).
  5. A name with no price was held at a 0% return instead of being left out.
  6. No net-of-cost figures.

Fixtures are synthetic; no live data is read.
"""

import csv
import math
from pathlib import Path

import pytest

import scripts.research.build_selection_benchmark as bsb

DATES = ["2024-01-31", "2024-02-29", "2024-03-29", "2024-04-30"]


def _snapshot(root: Path, day: str, tickers: list) -> None:
    path = root / day
    path.mkdir(parents=True)
    with open(path / "rankings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "actionable_rank", "eligible"])
        for i, t in enumerate(tickers, start=1):
            w.writerow([t, i, 1])


def _prices(path: Path, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "ticker", "close"])
        w.writerows(rows)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    snaps = tmp_path / "snapshots"
    monkeypatch.setattr(bsb, "SNAPSHOT_DIR", snaps)
    monkeypatch.setattr(bsb, "SHADOW_PERF_PATH", tmp_path / "none.csv")
    monkeypatch.setattr(bsb, "_pit_mode", "off")
    return snaps, tmp_path / "prices.csv"


def _two_name_setup(env, aaa, bbb, xbi, baskets=None):
    snaps, prices = env
    baskets = baskets or [["AAA", "BBB"]] * len(DATES)
    for d, b in zip(DATES, baskets):
        _snapshot(snaps, d, b)
    rows = []
    for d, a, b, x in zip(DATES, aaa, bbb, xbi):
        rows += [[d, "AAA", a], [d, "BBB", b], [d, "XBI", x]]
    _prices(prices, rows)
    return prices


def test_returns_are_compounded_not_summed(env):
    # Basket: +10%, +10%, +10%. XBI flat.
    prices = _two_name_setup(env, [100, 110, 121, 133.1], [100, 110, 121, 133.1], [50, 50, 50, 50])
    result = bsb.run_benchmark(top_n=2, price_csv=prices, cost_bps=(50,), exclude=())
    s = result["summary"]
    assert s["n_periods"] == 3
    assert s["gross"]["cumulative_pct"] == pytest.approx(33.1, abs=1e-6)  # not 30.0
    assert s["ew_cumulative_pct"] == pytest.approx(33.1, abs=1e-6)
    assert result["periods"][-1]["cum_ew"] == pytest.approx(33.1, abs=1e-4)


def test_win_rate_is_counted_without_shadow_data(env):
    # Basket beats XBI in two of three periods.
    prices = _two_name_setup(env, [100, 110, 110, 121], [100, 110, 110, 121], [50, 50, 55, 55])
    s = bsb.run_benchmark(top_n=2, price_csv=prices, cost_bps=(50,), exclude=())["summary"]
    assert s["gross"]["hit_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert s["ew_win_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert s["shadow_win_rate"] is None


def test_net_of_cost_uses_actual_turnover(env):
    # Period 1 holds AAA+BBB (initial buy: 100% one-way). Period 2 keeps AAA,
    # swaps BBB for CCC (50% turnover: sell + buy). Period 3 unchanged.
    snaps, prices = env
    baskets = [["AAA", "BBB"], ["AAA", "CCC"], ["AAA", "CCC"], ["AAA", "CCC"]]
    for d, b in zip(DATES, baskets):
        _snapshot(snaps, d, b)
    rows = [[d, t, 100] for d in DATES for t in ("AAA", "BBB", "CCC", "XBI")]
    _prices(prices, rows)
    result = bsb.run_benchmark(top_n=2, price_csv=prices, cost_bps=(50,), exclude=())
    costs = [p["cost_pct_net50"] for p in result["periods"]]
    assert costs == pytest.approx([0.50, 0.50, 0.0], abs=1e-9)
    # Flat prices: net return is just the cost.
    assert result["summary"]["net50"]["cumulative_pct"] == pytest.approx(((1 - 0.005) ** 2 - 1) * 100, abs=1e-6)
    assert result["summary"]["gross"]["cumulative_pct"] == pytest.approx(0.0, abs=1e-9)


def test_snapshot_on_a_non_trading_day_uses_the_last_price(env):
    snaps, prices = env
    for d in DATES[:3]:
        _snapshot(snaps, d, ["AAA", "BBB"])
    rows = []
    # 2024-02-29 has no price row; the last trade was 2024-02-28.
    for d, p in (("2024-01-31", 100), ("2024-02-28", 110), ("2024-03-29", 121)):
        rows += [[d, "AAA", p], [d, "BBB", p], [d, "XBI", 50]]
    _prices(prices, rows)
    result = bsb.run_benchmark(top_n=2, price_csv=prices, cost_bps=(50,), exclude=())
    assert result["summary"]["n_periods"] == 2
    assert result["summary"]["n_skipped_periods"] == 0
    assert [p["ew_pnl_pct"] for p in result["periods"]] == pytest.approx([10.0, 10.0], abs=1e-6)


def test_unpriced_name_is_left_out_not_held_flat(env):
    snaps, prices = env
    for d in DATES[:2]:
        _snapshot(snaps, d, ["AAA", "BBB", "CCC", "DDD", "EEE"])
    rows = [[DATES[0], t, 100] for t in ("AAA", "BBB", "CCC", "DDD", "XBI")]
    rows += [[DATES[1], t, 120] for t in ("AAA", "BBB", "CCC", "DDD")] + [[DATES[1], "XBI", 100]]
    _prices(prices, rows)  # EEE has no prices at all
    period = bsb.run_benchmark(top_n=5, price_csv=prices, cost_bps=(50,), exclude=())["periods"][0]
    assert (period["n_held"], period["n_priced"]) == (5, 4)
    assert period["ew_pnl_pct"] == pytest.approx(20.0, abs=1e-6)  # not 16.0


def test_period_is_skipped_when_too_few_names_are_priced(env):
    snaps, prices = env
    for d in DATES[:2]:
        _snapshot(snaps, d, ["AAA", "BBB", "CCC", "DDD", "EEE"])
    rows = [[d, t, 100] for d in DATES[:2] for t in ("AAA", "XBI")]
    _prices(prices, rows)
    result = bsb.run_benchmark(top_n=5, price_csv=prices, cost_bps=(50,), exclude=())
    assert result["summary"]["n_periods"] == 0
    assert result["summary"]["n_skipped_periods"] == 1


def test_a_name_that_stops_trading_exits_at_its_last_price(env):
    snaps, prices = env
    for d in DATES[:2]:
        _snapshot(snaps, d, ["AAA", "BBB"])
    rows = [[DATES[0], "AAA", 100], [DATES[0], "BBB", 100], [DATES[0], "XBI", 50]]
    rows += [["2024-02-10", "BBB", 150]]  # acquired: last print mid-period
    rows += [[DATES[1], "AAA", 100], [DATES[1], "XBI", 50]]
    _prices(prices, rows)
    period = bsb.run_benchmark(top_n=2, price_csv=prices, cost_bps=(50,), exclude=())["periods"][0]
    assert period["ew_pnl_pct"] == pytest.approx(25.0, abs=1e-6)
    assert period["n_stale_exit"] == 1


def test_excluded_tickers_are_skipped_and_the_next_rank_is_taken(env):
    snaps, prices = env
    for d in DATES[:2]:
        _snapshot(snaps, d, ["JBIO", "AAA", "BBB"])
    rows = [[DATES[0], t, 100] for t in ("JBIO", "AAA", "BBB", "XBI")]
    rows += [[DATES[1], "JBIO", 300], [DATES[1], "AAA", 110], [DATES[1], "BBB", 110], [DATES[1], "XBI", 100]]
    _prices(prices, rows)
    period = bsb.run_benchmark(top_n=2, price_csv=prices, cost_bps=(50,), exclude=("JBIO",))["periods"][0]
    assert period["ew_pnl_pct"] == pytest.approx(10.0, abs=1e-6)


def test_summary_statistics(env):
    prices = _two_name_setup(env, [100, 110, 99, 108.9], [100, 110, 99, 108.9], [50, 50, 50, 50])
    s = bsb.run_benchmark(top_n=2, price_csv=prices, cost_bps=(50,), exclude=())["summary"]
    g = s["gross"]
    ex = [0.10, -0.10, 0.10]
    mean = sum(ex) / 3
    sd = math.sqrt(sum((x - mean) ** 2 for x in ex) / 2)
    assert g["mean_period_excess_pct"] == pytest.approx(mean * 100, abs=1e-3)
    assert g["t_stat"] == pytest.approx(mean / (sd / math.sqrt(3)), abs=1e-2)
    assert g["max_drawdown_pct"] == pytest.approx(-10.0, abs=1e-3)
    assert g["xbi_cumulative_pct"] == pytest.approx(0.0, abs=1e-9)
    assert s["schema"] == "selection_benchmark.v2"
    assert s["price_source"] == "prices.csv"


def test_default_price_file_prefers_split_adjusted(tmp_path):
    raw = tmp_path / "price_history.csv"
    raw.write_text("date,ticker,close\n", encoding="utf-8")
    assert bsb.default_price_path(tmp_path) == raw
    adj = tmp_path / "price_history_split_adj.csv"
    adj.write_text("date,ticker,close\n", encoding="utf-8")
    assert bsb.default_price_path(tmp_path) == adj
