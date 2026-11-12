"""Selector-only shadow (Spec 094): capture discipline and return comparison.

Fixtures are synthetic; no live data is read.
"""

import csv
import json
from pathlib import Path

import tools.selector_only_shadow as sos

N = 70  # eligible names per snapshot


def _write_snapshot(root: Path, day: str, n: int = N, ineligible: int = 5) -> Path:
    """Ticker Txx: selector_score falls with xx; the ranker reverses the top-60 cohort."""
    path = root / day
    path.mkdir(parents=True)
    with open(path / "rankings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "eligible", "actionable_rank", "selector_score"])
        for i in range(n):
            # Cohort = first 60 by selector_score; production rank reverses it.
            rank = 60 - i if i < 60 else i + 1
            w.writerow([f"T{i:02d}", 1, rank, round(1.0 - i / 100, 4)])
        for i in range(ineligible):
            w.writerow([f"X{i:02d}", 0, "", 0.99])
    return path / "rankings.csv"


def _write_prices(path: Path, days: list, daily_ret) -> None:
    """daily_ret(ticker_index) -> constant daily return for that ticker."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "ticker", "close"])
        for i in range(N):
            price = 100.0
            for d in days:
                w.writerow([d, f"T{i:02d}", f"{price:.6f}"])
                price *= 1 + daily_ret(i)
        for k, d in enumerate(days):
            w.writerow([d, "XBI", f"{100 + k:.6f}"])


def _days(n: int) -> list:
    out = []
    for m in (3, 4, 5):
        for d in range(1, 29):
            out.append(f"2026-{m:02d}-{d:02d}")
    return out[:n]


# ── Baskets ───────────────────────────────────────────────────────────


def test_baskets_split_selector_from_ranker(tmp_path):
    rankings = _write_snapshot(tmp_path, "2026-03-02")
    baskets = sos.build_baskets(sos._read_rankings(rankings))
    # Selector alone takes the 30 highest selector scores.
    assert baskets["selector_only"] == [f"T{i:02d}" for i in range(30)]
    # Production takes rank 1..30, which the ranker put at cohort positions 59..30.
    assert baskets["production"] == [f"T{i:02d}" for i in range(59, 29, -1)]
    assert not set(baskets["production"]) & set(baskets["selector_only"])


def test_ineligible_rows_never_enter_a_basket(tmp_path):
    rankings = _write_snapshot(tmp_path, "2026-03-02")
    baskets = sos.build_baskets(sos._read_rankings(rankings))
    assert not [t for b in baskets.values() for t in b if t.startswith("X")]


def test_selector_ties_break_on_ticker():
    rows = [
        {"ticker": t, "eligible": "1", "actionable_rank": str(i + 1), "selector_score": "0.5"}
        for i, t in enumerate("DCBA")
    ]
    assert sos.build_baskets(rows, top_n=2)["selector_only"] == ["A", "B"]


# ── Capture discipline ────────────────────────────────────────────────


def test_same_day_capture_is_live_and_later_capture_is_replay(tmp_path):
    snaps, out = tmp_path / "snaps", tmp_path / "out"
    _write_snapshot(snaps, "2026-03-02")
    _write_snapshot(snaps, "2026-03-03")
    live = sos.capture("2026-03-02", snapshots_dir=snaps, out_dir=out, today="2026-03-02")
    late = sos.capture("2026-03-03", snapshots_dir=snaps, out_dir=out, today="2026-03-09")
    assert (live["status"], live["capture_mode"]) == ("captured", sos.LIVE)
    assert (late["status"], late["capture_mode"]) == ("captured", sos.REPLAY)
    rows = sos.load_captures(out)
    assert [r["capture_mode"] for r in rows] == [sos.LIVE, sos.REPLAY]
    assert len(rows[0]["rankings_sha256"]) == 64 and rows[0]["overlap"] == 0


def test_capture_never_rewrites_a_date(tmp_path):
    snaps, out = tmp_path / "snaps", tmp_path / "out"
    rankings = _write_snapshot(snaps, "2026-03-02")
    sos.capture("2026-03-02", snapshots_dir=snaps, out_dir=out, today="2026-03-02")
    before = (out / "captures.jsonl").read_text(encoding="utf-8")
    # The snapshot is regenerated afterwards; the ledger must not follow it.
    rankings.write_text(rankings.read_text(encoding="utf-8").replace("T00", "ZZZ"), encoding="utf-8")
    again = sos.capture("2026-03-02", snapshots_dir=snaps, out_dir=out, today="2026-03-02")
    assert again["status"] == "exists"
    assert (out / "captures.jsonl").read_text(encoding="utf-8") == before


def test_capture_skips_missing_or_thin_snapshots(tmp_path):
    snaps, out = tmp_path / "snaps", tmp_path / "out"
    assert sos.capture("2026-03-02", snapshots_dir=snaps, out_dir=out)["status"] == "skipped"
    _write_snapshot(snaps, "2026-03-03", n=20)
    assert sos.capture("2026-03-03", snapshots_dir=snaps, out_dir=out)["status"] == "skipped"
    assert not (out / "captures.jsonl").exists()


def test_backfill_writes_replay_only_and_respects_exclusions(tmp_path):
    snaps, out = tmp_path / "snaps", tmp_path / "out"
    for d in ("2026-03-02", "2026-03-03", "2026-03-04"):
        _write_snapshot(snaps, d)
    excl = tmp_path / "excl.json"
    excl.write_text(json.dumps({"snapshots": [{"date": "2026-03-03"}]}), encoding="utf-8")
    counts = sos.backfill(snapshots_dir=snaps, out_dir=out, start_date="2026-03-01", exclusions_path=excl)
    assert counts["captured"] == 2
    rows = sos.load_captures(out)
    assert [r["as_of_date"] for r in rows] == ["2026-03-02", "2026-03-04"]
    assert {r["capture_mode"] for r in rows} == {sos.REPLAY}


# ── Return comparison ─────────────────────────────────────────────────


def _setup(tmp_path, n_snap_days: int, n_price_days: int, daily_ret):
    snaps, out = tmp_path / "snaps", tmp_path / "out"
    days = _days(n_price_days)
    for d in days[:n_snap_days]:
        _write_snapshot(snaps, d)
        sos.capture(d, snapshots_dir=snaps, out_dir=out, today=d)
    prices = tmp_path / "prices.csv"
    _write_prices(prices, days, daily_ret)
    return out, prices


def test_diff_is_production_minus_selector_only(tmp_path):
    # Low-index names (the selector's picks) fall; high-index names rise.
    out, prices = _setup(tmp_path, 1, 10, lambda i: -0.001 if i < 30 else 0.001)
    result = sos.summarize(out_dir=out, price_csv=prices, horizons=(5,))
    window = result["by_mode"][sos.LIVE]["h5d"]["windows"][0]
    assert window["production_pct"] > 0 > window["selector_only_pct"]
    assert window["diff_pct"] == round(window["production_pct"] - window["selector_only_pct"], 3)
    assert result["by_mode"][sos.LIVE]["h5d"]["stats"]["ranker_win_rate"] == 1.0


def test_statistics_use_non_overlapping_windows(tmp_path):
    out, prices = _setup(tmp_path, 12, 30, lambda i: 0.001 if i < 30 else -0.001)
    result = sos.summarize(out_dir=out, price_csv=prices, horizons=(5,))
    block = result["by_mode"][sos.LIVE]["h5d"]
    # 12 daily captures, 5-day horizon: starts on day 0, 5 and 10 only.
    assert block["stats"]["n_matured_captures"] == 12
    assert block["stats"]["n"] == 3
    assert [w["as_of_date"] for w in block["windows"]] == [_days(30)[k] for k in (0, 5, 10)]
    assert block["stats"]["reading"] == "insufficient_windows"
    assert block["stats"]["ranker_win_rate"] == 0.0


def test_unmatured_captures_are_not_scored(tmp_path):
    out, prices = _setup(tmp_path, 3, 6, lambda i: 0.001)
    result = sos.summarize(out_dir=out, price_csv=prices, horizons=(5,))
    # Only the first capture has five later trading days in the price file.
    assert result["by_mode"][sos.LIVE]["h5d"]["stats"]["n_matured_captures"] == 1


def test_replay_rows_are_reported_separately(tmp_path):
    snaps, out = tmp_path / "snaps", tmp_path / "out"
    days = _days(10)
    _write_snapshot(snaps, days[0])
    _write_snapshot(snaps, days[1])
    sos.capture(days[0], snapshots_dir=snaps, out_dir=out, today=days[0])
    sos.capture(days[1], snapshots_dir=snaps, out_dir=out, today=days[9])
    prices = tmp_path / "prices.csv"
    _write_prices(prices, days, lambda i: 0.001)
    result = sos.summarize(out_dir=out, price_csv=prices, horizons=(5,))
    assert result["n_captures"] == {sos.LIVE: 1, sos.REPLAY: 1}
    assert result["by_mode"][sos.LIVE]["h5d"]["stats"]["n"] == 1
    assert result["by_mode"][sos.REPLAY]["h5d"]["stats"]["n"] == 1
    assert "Only LIVE captures are evidence" in (out / "summary.md").read_text(encoding="utf-8")


def test_summary_with_no_captures(tmp_path):
    result = sos.summarize(out_dir=tmp_path / "out", price_csv=tmp_path / "none.csv")
    assert result["status"] == "no_captures"
