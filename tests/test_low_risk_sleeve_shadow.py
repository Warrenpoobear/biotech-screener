"""Low-risk sleeve shadow: basket construction, capture discipline, return comparison.

Fixtures are synthetic; no live data is read.
"""

import csv
import json
import math
from pathlib import Path

import pytest

import tools.low_risk_sleeve_shadow as lrs

N = 80  # eligible names per snapshot


def _days(n: int) -> list:
    out = []
    for y in (2025, 2026):
        for m in range(1, 13):
            for d in range(1, 29):
                out.append(f"{y}-{m:02d}-{d:02d}")
    return out[:n]


def _write_snapshot(root: Path, day: str, n: int = N, ineligible: int = 3) -> Path:
    """Ticker Txx: production holds T00..T29 (actionable_rank = i + 1)."""
    path = root / day
    path.mkdir(parents=True)
    with open(path / "rankings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "eligible", "actionable_rank", "selector_score"])
        for i in range(n):
            w.writerow([f"T{i:02d}", 1, i + 1, round(1.0 - i / 100, 4)])
        for i in range(ineligible):
            w.writerow([f"X{i:02d}", 0, "", 0.5])
    return path / "rankings.csv"


def _amp(i: int) -> float:
    """Daily swing for ticker i: higher index = calmer stock."""
    return 0.05 / (1 + i / 10)


def _write_prices(path: Path, days: list, drift=lambda i: 0.0, last_day=None) -> None:
    """Alternating +/- swings of size _amp(i) around a drift, so realised vol falls with i."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "date", "close"])
        for i in list(range(N)) + [f"X{k:02d}" for k in range(3)]:
            price = 100.0
            for k, d in enumerate(days):
                if last_day and d > last_day:
                    break
                name = i if isinstance(i, str) else f"T{i:02d}"
                w.writerow([name, d, f"{price:.6f}"])
                a = _amp(i) if isinstance(i, int) else 0.02
                price *= 1 + (a if k % 2 == 0 else -a * 0.98) + (drift(i) if isinstance(i, int) else 0.0)
        price = 100.0
        for k, d in enumerate(days):
            if last_day and d > last_day:
                break
            w.writerow(["XBI", d, f"{price:.6f}"])
            price *= 1 + (0.01 if k % 2 == 0 else -0.0098)


# ── Risk measures and basket ──────────────────────────────────────────


def test_risk_measures_use_only_prices_on_or_before_as_of():
    days = _days(200)
    series = {d: 100.0 * (1.01**k) for k, d in enumerate(days)}
    as_of = days[150]
    poisoned = dict(series)
    for d in days[151:]:
        poisoned[d] = 1.0  # a crash after as_of must not move the measures
    xbi = {d: 100.0 for d in days}
    assert lrs.risk_measures(series, xbi, as_of) == lrs.risk_measures(poisoned, xbi, as_of)


def test_basket_prefers_calm_names_and_excludes_production_top30(tmp_path):
    days = _days(200)
    rankings = _write_snapshot(tmp_path / "snaps", days[-1])
    prices_csv = tmp_path / "prices.csv"
    _write_prices(prices_csv, days)
    rows = lrs._read_rankings(rankings)
    prices = lrs.load_prices(prices_csv, {r["ticker"] for r in rows} | {"XBI"})
    result = lrs.build_sleeve(rows, prices, days[-1])
    sleeve = result["sleeve"]
    assert len(sleeve) == lrs.SLEEVE_N
    assert not set(sleeve) & set(result["production"])
    assert result["production"] == [f"T{i:02d}" for i in range(30)]
    # Calmest eligible names are the highest indices.
    assert sleeve == [f"T{i:02d}" for i in range(N - 1, N - 1 - lrs.SLEEVE_N, -1)]
    assert not any(t.startswith("X") for t in sleeve)


def test_names_without_enough_history_are_not_eligible_for_the_sleeve(tmp_path):
    days = _days(200)
    prices = {"T01": {d: 100.0 for d in days[-10:]}, "XBI": {d: 100.0 for d in days}}
    assert lrs.risk_measures(prices["T01"], prices["XBI"], days[-1])["vol_60d"] is None


# ── Capture discipline ────────────────────────────────────────────────


def _setup(tmp_path, n_days=200, last_price_day=None):
    days = _days(n_days)
    snaps = tmp_path / "snaps"
    _write_snapshot(snaps, days[150])
    prices_csv = tmp_path / "prices.csv"
    _write_prices(prices_csv, days, last_day=last_price_day)
    return days, snaps, prices_csv, tmp_path / "out"


def test_capture_on_snapshot_date_is_live_and_append_only(tmp_path):
    days, snaps, prices_csv, out = _setup(tmp_path)
    d = days[150]
    first = lrs.capture(d, snapshots_dir=snaps, out_dir=out, price_csv=prices_csv, today=d)
    assert first["status"] == "captured" and first["capture_mode"] == lrs.LIVE
    second = lrs.capture(d, snapshots_dir=snaps, out_dir=out, price_csv=prices_csv, today=d)
    assert second["status"] == "exists"
    rows = lrs.load_captures(out)
    assert len(rows) == 1
    row = rows[0]
    assert row["schema"] == lrs.SCHEMA_VERSION
    assert row["price_asof"] == d
    assert len(row["sleeve"]) == lrs.SLEEVE_N and len(row["production"]) == 30
    assert len(row["rankings_sha256"]) == 64
    assert set(row["sleeve_measures"]) == set(row["sleeve"])


def test_capture_written_later_is_replay(tmp_path):
    days, snaps, prices_csv, out = _setup(tmp_path)
    res = lrs.capture(days[150], snapshots_dir=snaps, out_dir=out, price_csv=prices_csv, today=days[160])
    assert res["capture_mode"] == lrs.REPLAY


def test_capture_refuses_stale_prices(tmp_path):
    days, snaps, prices_csv, out = _setup(tmp_path, last_price_day=_days(200)[140])
    res = lrs.capture(days[150], snapshots_dir=snaps, out_dir=out, price_csv=prices_csv, today=days[150])
    assert res["status"] == "skipped" and "stale" in res["reason"]
    assert lrs.load_captures(out) == []


def test_capture_skips_missing_snapshot(tmp_path):
    days, snaps, prices_csv, out = _setup(tmp_path)
    res = lrs.capture(days[151], snapshots_dir=snaps, out_dir=out, price_csv=prices_csv, today=days[151])
    assert res["status"] == "skipped"


# ── Summary ───────────────────────────────────────────────────────────


def test_summary_scores_matured_live_windows_only_once_per_horizon(tmp_path):
    days = _days(260)
    snaps = tmp_path / "snaps"
    out = tmp_path / "out"
    prices_csv = tmp_path / "prices.csv"
    # Calm names (high index) drift up, so the sleeve should beat production.
    _write_prices(prices_csv, days, drift=lambda i: 0.002 if i >= 40 else -0.001)
    for k in range(150, 200, 5):
        _write_snapshot(snaps, days[k])
        lrs.capture(days[k], snapshots_dir=snaps, out_dir=out, price_csv=prices_csv, today=days[k])
    result = lrs.summarize(out_dir=out, price_csv=prices_csv)
    h5 = result["by_mode"][lrs.LIVE]["h5d"]
    assert h5["stats"]["sleeve_minus_xbi"]["n"] == 10
    assert h5["stats"]["sleeve_minus_production"]["mean_pct"] > 0
    assert h5["stats"]["sleeve_minus_xbi"]["reading"] == "insufficient_windows"
    # 21-day windows must not overlap: at most one per 21 trading days.
    starts = [w["as_of_date"] for w in result["by_mode"][lrs.LIVE]["h21d"]["windows"]]
    idx = [days.index(s) for s in starts]
    assert all(b - a >= 21 for a, b in zip(idx, idx[1:]))
    assert (out / "summary.json").exists() and (out / "summary.md").exists()
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert "pre-registered" in md.lower()


def test_summary_with_no_captures(tmp_path):
    prices_csv = tmp_path / "prices.csv"
    _write_prices(prices_csv, _days(30))
    res = lrs.summarize(out_dir=tmp_path / "out", price_csv=prices_csv)
    assert res["status"] == "no_captures"
