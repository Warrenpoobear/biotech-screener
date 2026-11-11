"""Registry-confirmed split handling (2026-11-11).

- Momentum lookback prices are split-adjusted via corporate_actions.json, so an
  in-window reverse split is not scored as a return (IMMP 1:20, 2026-09-28).
- The split-adj repair applies only registry-confirmed discontinuities; a real
  crash with no registry entry (EYPT 2026-10-19) is left in the series.
"""

import csv
import json
import sys
from datetime import date, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from repair_price_history_splits import repair_price_history  # noqa: E402

import run_screen as rs  # noqa: E402


def _write_prices(path: Path, series: dict) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "ticker", "close", "open", "high", "low", "volume"])
        for ticker, rows in series.items():
            for d, c in rows:
                w.writerow([d.isoformat(), ticker, c, c, c, c, 1000])


def _write_registry(path: Path, actions: list) -> None:
    path.write_text(json.dumps({"_schema": "corporate_actions.v1", "actions": actions}))


def _bdays(start: date, n: int):
    d, out = start, []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


SPLIT_DAY = date(2026, 9, 28)
AS_OF = "2026-11-11"


def _split_series():
    """0.25 flat before a 1:20 reverse split, 5.00 flat after (no real move)."""
    days = _bdays(date(2026, 4, 1), 160)
    return [(d, 0.25 if d < SPLIT_DAY else 5.0) for d in days if d <= date(2026, 11, 11)]


def _crash_series():
    """3.00 flat, then a genuine -85% crash on 2026-10-19 (no split)."""
    days = _bdays(date(2026, 4, 1), 160)
    return [(d, 3.0 if d < date(2026, 10, 19) else 0.45) for d in days if d <= date(2026, 11, 11)]


REVERSE_SPLIT = {
    "ticker": "AAA",
    "action": "reverse_split",
    "effective_date": SPLIT_DAY.isoformat(),
    "ratio": "1:20",
    "factor": 20.0,
}


def test_momentum_adjusts_registered_split(tmp_path):
    _write_prices(tmp_path / "price_history.csv", {"AAA": _split_series(), "XBI": _crash_series()})
    _write_registry(tmp_path / "corporate_actions.json", [REVERSE_SPLIT])
    md = {"AAA": {}}
    rs.compute_momentum_from_price_history(tmp_path / "price_history.csv", AS_OF, md)
    assert abs(md["AAA"]["return_60d"]) < 1e-9
    assert abs(md["AAA"]["return_120d"]) < 1e-9


def test_momentum_without_registry_is_unchanged_raw(tmp_path):
    _write_prices(tmp_path / "price_history.csv", {"AAA": _split_series(), "XBI": _crash_series()})
    md = {"AAA": {}}
    rs.compute_momentum_from_price_history(tmp_path / "price_history.csv", AS_OF, md)
    assert md["AAA"]["return_60d"] > 18  # raw 0.25 -> 5.00 = +1900%


def test_momentum_keeps_unregistered_crash(tmp_path):
    _write_prices(tmp_path / "price_history.csv", {"BBB": _crash_series(), "XBI": _crash_series()})
    _write_registry(tmp_path / "corporate_actions.json", [REVERSE_SPLIT])
    md = {"BBB": {}}
    rs.compute_momentum_from_price_history(tmp_path / "price_history.csv", AS_OF, md)
    assert abs(md["BBB"]["return_60d"] - (-0.85)) < 1e-9


def _adj_closes(path: Path, ticker: str) -> list:
    return [float(r["close"]) for r in csv.DictReader(open(path)) if r["ticker"] == ticker]


def test_repair_applies_confirmed_skips_unconfirmed(tmp_path):
    _write_prices(tmp_path / "price_history.csv", {"AAA": _split_series(), "BBB": _crash_series()})
    _write_registry(tmp_path / "corporate_actions.json", [REVERSE_SPLIT])
    out = tmp_path / "adj.csv"
    summary = repair_price_history(tmp_path / "price_history.csv", out)
    assert summary["split_mode"] == "registry_confirmed"
    assert set(summary["split_details"]) == {"AAA"}
    assert set(summary["unconfirmed_not_applied"]) == {"BBB"}
    assert set(_adj_closes(out, "AAA")) == {5.0}  # split smoothed
    assert min(_adj_closes(out, "BBB")) == 0.45  # crash preserved


def test_repair_without_registry_falls_back_to_heuristic(tmp_path):
    _write_prices(tmp_path / "price_history.csv", {"BBB": _crash_series()})
    out = tmp_path / "adj.csv"
    summary = repair_price_history(tmp_path / "price_history.csv", out)
    assert summary["split_mode"] == "heuristic_fallback"
    assert set(summary["split_details"]) == {"BBB"}


def test_registry_has_2026_reverse_splits():
    reg = json.loads((REPO_ROOT / "production_data" / "corporate_actions.json").read_text())
    got = {(a["ticker"], a["effective_date"], a["factor"]) for a in reg["actions"] if a["action"] == "reverse_split"}
    for row in [
        ("KALA", "2026-05-08", 50.0),
        ("SKYE", "2026-08-17", 8.0),
        ("GOSS", "2026-08-17", 80.0),
        ("IMMP", "2026-09-28", 20.0),
    ]:
        assert row in got


def test_momentum_does_not_double_adjust_already_adjusted_source(tmp_path):
    """Sharadar SEP close / post-split yfinance history: no jump in the series.

    The registry split must be skipped, not applied a second time.
    """
    days = _bdays(date(2026, 4, 1), 160)
    adjusted = [(d, 5.0) for d in days if d <= date(2026, 11, 11)]
    _write_prices(tmp_path / "price_history.csv", {"AAA": adjusted, "XBI": _crash_series()})
    _write_registry(tmp_path / "corporate_actions.json", [REVERSE_SPLIT])
    md = {"AAA": {}}
    rs.compute_momentum_from_price_history(tmp_path / "price_history.csv", AS_OF, md)
    assert abs(md["AAA"]["return_60d"]) < 1e-9
    assert abs(md["AAA"]["return_120d"]) < 1e-9
