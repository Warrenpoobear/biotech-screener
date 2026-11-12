"""IC dashboard coverage gates + priced-move plausibility bound (2026-11-11)."""

import csv
import json
from pathlib import Path

import tools.build_ic_dashboard as icd
from common.straddle_mispricing import MAX_PLAUSIBLE_IMPLIED_MOVE, compute_cheap_vol_score


def test_coinvest_is_monitored():
    assert ("coinvest_score_z", True) in icd.SIGNALS


def test_inst_delta_rows_without_signal_are_dropped(tmp_path):
    rows = [
        {"ticker": "A", "inst_delta_z": "1.0", "has_inst_delta": "True", "inst_delta_regime": "clean"},
        {"ticker": "B", "inst_delta_z": "0.0", "has_inst_delta": "False", "inst_delta_regime": "clean"},
        {"ticker": "C", "inst_delta_z": "-2.0", "has_inst_delta": "True", "inst_delta_regime": "transition"},
    ]
    with open(tmp_path / "rankings.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert icd._load_signal_values(tmp_path, "inst_delta_z") == {"A": 1.0}
    assert set(icd._load_signal_values(tmp_path, "has_inst_delta")) == set()  # non-numeric


def test_excluded_dates_are_skipped(tmp_path):
    for d in ("2026-11-02", "2026-11-03", "2026-11-04"):
        (tmp_path / d).mkdir()
    got = icd._find_prior_snapshots(tmp_path, "2026-11-05", 10, exclude={"2026-11-03"})
    assert got == ["2026-11-02", "2026-11-04"]


def test_provenance_file_loads():
    dates = icd._load_excluded_dates()
    assert "2026-10-20" in dates and "2026-11-04" not in dates


def test_min_dates_gate_constant():
    assert icd.MIN_DATES_FOR_VERDICT >= 8


def _write_ranked_snapshots(root: Path, dates: list, n: int = 12) -> None:
    for d in dates:
        day = root / d
        day.mkdir(parents=True)
        with open(day / "rankings.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ticker", "score_rank_pct", "inst_delta_z", "has_inst_delta", "coinvest_score_z"])
            for i in range(n):
                # score_rank_pct is higher_is_better=False, so a rising score
                # against a falling return is a strongly negative IC.
                w.writerow([f"T{i:02d}", i, "0.0", "False", i])


def _write_prices(path: Path, dates: list, n: int = 12) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "date", "close"])
        for i in range(n):
            for k, d in enumerate(dates):
                # Next-day close rises with the ticker index, so the raw
                # signal and the forward return move together. score_rank_pct
                # is higher_is_better=False, which flips that into a negative IC.
                w.writerow([f"T{i:02d}", d, 100 + k + (i + 1) * k])


def test_min_dates_blocks_alert_and_is_configurable(tmp_path):
    dates = ["2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05", "2026-03-06"]
    snaps = tmp_path / "snapshots"
    _write_ranked_snapshots(snaps, dates)
    prices = tmp_path / "prices.csv"
    _write_prices(prices, dates + ["2026-03-09", "2026-03-10"])
    gated = icd.build_ic_dashboard(
        "2026-03-11",
        snapshots_dir=snaps,
        price_csv=prices,
        artifacts_dir=tmp_path / "a",
        lookback=10,
        horizon=1,
        min_dates_for_verdict=8,
    )
    assert gated["signals"]["score_rank_pct"]["health"] == "INSUFFICIENT_DATES"
    assert gated["signals"]["score_rank_pct"]["n_dates"] < 8
    assert gated["attention"] == "LOW"
    assert "coinvest_score_z" in gated["signals"]
    # The zero column is not a measured inst_delta, so it contributes no dates.
    assert gated["signals"]["inst_delta_z"]["n_dates"] == 0
    assert gated["signals"]["inst_delta_z"]["health"] == "NO_DATA"

    alerted = icd.build_ic_dashboard(
        "2026-03-11",
        snapshots_dir=snaps,
        price_csv=prices,
        artifacts_dir=tmp_path / "b",
        lookback=10,
        horizon=1,
        min_dates_for_verdict=2,
    )
    assert alerted["signals"]["score_rank_pct"]["health"] == "ALERT"
    assert alerted["min_dates_for_verdict"] == 2


def test_implausible_iv_move_is_dropped():
    out = compute_cheap_vol_score(26.78, 355, "CLINICAL", "phase 2", "oncology", {})
    assert out["cheap_vol_score"] is None and out["implied_move"] is None


def test_bound_value():
    assert MAX_PLAUSIBLE_IMPLIED_MOVE == 1.5


def test_exclusions_only_for_production_dir_unless_explicit(tmp_path):
    snaps = tmp_path / "snapshots"
    for d in ("2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05"):
        (snaps / d).mkdir(parents=True)
        (snaps / d / "rankings.csv").write_text("ticker,score_rank_pct\n")
    ex = tmp_path / "ex.json"
    ex.write_text(json.dumps({"snapshots": [{"date": "2026-03-03"}, {"date": "2026-03-04"}]}))
    r = icd.build_ic_dashboard("2026-03-06", snapshots_dir=snaps, artifacts_dir=tmp_path / "a", exclusions_path=ex)
    assert "error" in r and "found 2" in r["error"]
