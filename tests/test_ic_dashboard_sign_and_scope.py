"""IC dashboard: score_rank_pct sign, production-order signal, price source, trend.

Defects found 2026-11-12:

1. ``score_rank_pct`` was negated before the IC (``higher_is_better=False``).
   The percentile rises with ``composite_score`` (best rank -> highest
   percentile, ``common/score_to_er.py``), so every reported IC had the wrong
   sign.
2. The dashboard did not measure the production order (``actionable_rank``).
3. Forward returns came from ``price_history.csv``, which still carries
   unadjusted reverse splits.
4. A healthy lookback mean could hide a recent collapse.

Fixtures are synthetic; no live data is read.
"""

import csv
from pathlib import Path

import scripts.shared_regime_check as regime
import tools.build_ic_dashboard as icd

N = 12


def _dates(n):
    return [f"2026-03-{d:02d}" for d in range(2, 2 + n)]


def _write_snapshots(root: Path, dates, ranked=N):
    """score_rank_pct rises with the ticker index; rank 1 is the highest index."""
    for d in dates:
        day = root / d
        day.mkdir(parents=True)
        with open(day / "rankings.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ticker", "score_rank_pct", "actionable_rank", "eligible"])
            for i in range(N):
                rank = N - i if i >= N - ranked else ""
                w.writerow([f"T{i:02d}", (i + 1) / N, rank, 1 if rank != "" else 0])


def _write_prices(path: Path, dates, sign_by_step):
    """Daily return of ticker i on step k is sign_by_step[k] * 0.001 * (i + 1)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "date", "close"])
        for i in range(N):
            price = 100.0
            for k, d in enumerate(dates):
                w.writerow([f"T{i:02d}", d, f"{price:.6f}"])
                if k < len(sign_by_step):
                    price *= 1 + sign_by_step[k] * 0.001 * (i + 1)


def _build(tmp_path, dates, signs, **kw):
    snaps = tmp_path / "snapshots"
    _write_snapshots(snaps, dates[:-1], **kw)
    prices = tmp_path / "prices.csv"
    _write_prices(prices, dates, signs)
    return icd.build_ic_dashboard(
        "2026-04-01",
        snapshots_dir=snaps,
        price_csv=prices,
        artifacts_dir=tmp_path / "artifacts",
        lookback=60,
        horizon=1,
        min_dates_for_verdict=2,
    )


def test_score_rank_pct_is_higher_is_better(tmp_path):
    dates = _dates(6)
    result = _build(tmp_path, dates, [1] * 5)
    sig = result["signals"]["score_rank_pct"]
    # Higher percentile and higher forward return move together: IC is +1.
    assert sig["mean_ic"] == 1.0
    assert sig["health"] == "HEALTHY"


def test_signal_direction_table():
    directions = dict(icd.SIGNALS)
    assert directions["score_rank_pct"] is True
    assert directions["actionable_rank"] is False
    assert regime.SIGNAL_HIB["score_rank_pct"] is True


def test_actionable_rank_is_tracked_and_skips_unranked_rows(tmp_path):
    dates = _dates(6)
    result = _build(tmp_path, dates, [1] * 5, ranked=10)
    sig = result["signals"]["actionable_rank"]
    # Rank 1 has the highest forward return, so the production order IC is +1.
    assert sig["mean_ic"] == 1.0
    # Two rows carry no rank (ineligible) and are not observations.
    assert {d["n_obs"] for d in sig["per_date"]} == {10}


def test_default_price_source_prefers_split_adjusted(tmp_path):
    raw = tmp_path / "price_history.csv"
    raw.write_text("ticker,date,close\n", encoding="utf-8")
    assert icd.default_price_csv(tmp_path) == raw
    adj = tmp_path / "price_history_split_adj.csv"
    adj.write_text("ticker,date,close\n", encoding="utf-8")
    assert icd.default_price_csv(tmp_path) == adj


def test_price_source_is_recorded(tmp_path):
    result = _build(tmp_path, _dates(6), [1] * 5)
    assert result["price_source"].endswith("prices.csv")


def test_recent_collapse_is_flagged_while_the_mean_is_still_healthy(tmp_path):
    dates = _dates(13)
    # Seven dates at IC +1, then five at IC -1: mean +0.1667, last five -1.
    result = _build(tmp_path, dates, [1] * 7 + [-1] * 5)
    sig = result["signals"]["score_rank_pct"]
    assert sig["n_dates"] == 12
    assert sig["health"] == "HEALTHY"
    assert sig["recent_ic"] == -1.0
    assert sig["trend"] == "decaying"
    assert result["attention"] == "MEDIUM"


def test_steady_signal_has_stable_trend(tmp_path):
    result = _build(tmp_path, _dates(13), [1] * 12)
    sig = result["signals"]["score_rank_pct"]
    assert sig["trend"] == "stable"
    assert result["attention"] == "LOW"


def test_trend_needs_enough_dates(tmp_path):
    result = _build(tmp_path, _dates(4), [1] * 3)
    sig = result["signals"]["score_rank_pct"]
    assert sig["recent_ic"] is None
    assert sig["trend"] == "insufficient_dates"
