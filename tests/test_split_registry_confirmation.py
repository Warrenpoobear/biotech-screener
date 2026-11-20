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


def test_repair_without_registry_does_not_treat_crash_as_split(tmp_path):
    _write_prices(tmp_path / "price_history.csv", {"BBB": _crash_series()})
    out = tmp_path / "adj.csv"
    summary = repair_price_history(tmp_path / "price_history.csv", out)
    assert summary["split_mode"] == "registry_missing"
    assert summary["split_details"] == {}
    assert set(summary["unconfirmed_not_applied"]) == {"BBB"}
    assert min(_adj_closes(out, "BBB")) == 0.45


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


SUPN_DIV = {"ticker": "DDD", "action": "special_dividend", "effective_date": "2026-11-03", "amount": 8.13}


def _div_series(drop: bool):
    days = _bdays(date(2026, 4, 1), 160)
    after = 28.15 if drop else 36.28
    return [(d, 36.28 if d < date(2026, 11, 3) else after) for d in days if d <= date(2026, 11, 11)]


def test_momentum_adjusts_visible_special_dividend(tmp_path):
    _write_prices(tmp_path / "price_history.csv", {"DDD": _div_series(True), "XBI": _crash_series()})
    _write_registry(tmp_path / "corporate_actions.json", [SUPN_DIV])
    md = {"DDD": {}}
    rs.compute_momentum_from_price_history(tmp_path / "price_history.csv", AS_OF, md)
    assert abs(md["DDD"]["return_20d"]) < 1e-9


def test_momentum_skips_dividend_not_visible(tmp_path):
    """Dividend-adjusted source: no ex-date drop, so no adjustment."""
    _write_prices(tmp_path / "price_history.csv", {"DDD": _div_series(False), "XBI": _crash_series()})
    _write_registry(tmp_path / "corporate_actions.json", [SUPN_DIV])
    md = {"DDD": {}}
    rs.compute_momentum_from_price_history(tmp_path / "price_history.csv", AS_OF, md)
    assert abs(md["DDD"]["return_20d"]) < 1e-9


def test_registry_loads_special_dividend():
    from common.corporate_actions import get_special_dividends, load_actions

    reg = load_actions(REPO_ROOT / "production_data" / "corporate_actions.json")
    divs = get_special_dividends("SUPN", reg)
    assert [(a.effective_date, a.amount) for a in divs] == [("2026-11-03", 8.13)]


# Factor 2.0 (AZN ADR ratio). The old momentum check was
# abs(log(observed) - log(factor)) <= log(2). On a flat already-adjusted
# series observed == 1, so abs(log(1) - log(2)) == log(2) and the lookback
# was doubled again (~-50%). A 15% relative window does not confirm that.


FACTOR_2 = {
    "ticker": "AZ2",
    "action": "reverse_split",
    "effective_date": SPLIT_DAY.isoformat(),
    "ratio": "1:2",
    "factor": 2.0,
}


def _factor2_days():
    return [d for d in _bdays(date(2026, 4, 1), 200) if d <= date(2026, 11, 11)]


def _factor2_jump_series():
    """Exact 2x jump on SPLIT_DAY. Flat on each side, so one adjustment is a zero return."""
    return [(d, 50.0 if d < SPLIT_DAY else 100.0) for d in _factor2_days()]


def _factor2_flat_series():
    """Already on the post-split basis: no jump to confirm."""
    return [(d, 100.0) for d in _factor2_days()]


def _factor2_drawdown_jump_series():
    """2x-matching jump, with an earlier pre-split high that only the adjusted series sees.

    Adjacent bars are 50 → 90 (ratio 1.8, within 15% of 2.0). Prices before
    2026-09-01 are 100. After ×2 those become 200, above the post-split 90,
    so drawdown is -0.55. Raw peak is 100 and drawdown is -0.10.
    """
    rows = []
    for d in _factor2_days():
        if d < date(2026, 9, 1):
            px = 100.0
        elif d < SPLIT_DAY:
            px = 50.0
        else:
            px = 90.0
        rows.append((d, px))
    return rows


def _registry(actions):
    from common.corporate_actions import CorporateAction, CorporateActionRegistry

    reg = CorporateActionRegistry(
        actions=[
            CorporateAction(
                ticker=a["ticker"],
                action=a["action"],
                effective_date=a["effective_date"],
                ratio=a.get("ratio"),
                factor=a.get("factor"),
                amount=a.get("amount"),
            )
            for a in actions
        ]
    )
    reg._build_indices()
    return reg


def test_split_ratio_confirmation_window_rejects_flat_factor_2():
    """The shared 15% window: exact 2x confirms; a flat series (ratio 1) does not."""
    from common.corporate_actions import registry_split_confirmed

    assert registry_split_confirmed(2.0, 2.0)
    assert registry_split_confirmed(1.964435, 2.0)  # AZN 187.80 / 95.60
    assert registry_split_confirmed(2.2, 2.0)  # 10% off, inside the window
    assert not registry_split_confirmed(1.0, 2.0)
    assert not registry_split_confirmed(2.4, 2.0)  # 20% off, outside the window


def test_momentum_factor2_jumped_series_adjusts_once(tmp_path):
    _write_prices(tmp_path / "price_history.csv", {"AZ2": _factor2_jump_series(), "XBI": _crash_series()})
    _write_registry(tmp_path / "corporate_actions.json", [FACTOR_2])
    md = {"AZ2": {}}
    rs.compute_momentum_from_price_history(tmp_path / "price_history.csv", AS_OF, md)
    # Once: lookback ×2 meets the post price. Twice would be about -50%. Unadjusted would be +100%.
    assert abs(md["AZ2"]["return_60d"]) < 1e-9
    assert abs(md["AZ2"]["return_120d"]) < 1e-9


def test_momentum_factor2_flat_series_is_not_adjusted(tmp_path):
    _write_prices(tmp_path / "price_history.csv", {"AZ2": _factor2_flat_series(), "XBI": _crash_series()})
    _write_registry(tmp_path / "corporate_actions.json", [FACTOR_2])
    md = {"AZ2": {}}
    rs.compute_momentum_from_price_history(tmp_path / "price_history.csv", AS_OF, md)
    assert abs(md["AZ2"]["return_60d"]) < 1e-9
    assert abs(md["AZ2"]["return_120d"]) < 1e-9


def test_drawdown_factor2_jumped_series_adjusts_once(tmp_path):
    from run_screen import _hydrate_drawdown

    prices = tmp_path / "price_history.csv"
    _write_prices(prices, {"AZ2": _factor2_drawdown_jump_series()})
    recs = {"AZ2": {"ticker": "AZ2", "defensive_features": {}}}
    _hydrate_drawdown(recs, prices, AS_OF, corporate_actions=_registry([FACTOR_2]))
    dd = recs["AZ2"]["defensive_features"]["drawdown"]
    # Adjusted peak 200, last close 90. Raw peak 100 would be -0.10. Twice would be -0.775.
    assert abs(dd - (-0.55)) < 1e-6


def test_drawdown_factor2_flat_series_is_not_adjusted(tmp_path):
    from run_screen import _hydrate_drawdown

    prices = tmp_path / "price_history.csv"
    _write_prices(prices, {"AZ2": _factor2_flat_series()})
    recs = {"AZ2": {"ticker": "AZ2", "defensive_features": {}}}
    _hydrate_drawdown(recs, prices, AS_OF, corporate_actions=_registry([FACTOR_2]))
    dd = recs["AZ2"]["defensive_features"]["drawdown"]
    assert dd == 0.0


def _beta_prices(jump: bool):
    """Weekdays through AS_OF. The split sits inside the 60-day beta window."""
    days = [d for d in _bdays(date(2026, 7, 1), 120) if d <= date(2026, 11, 11)]
    split_at = days[-40]
    az, xbi_rows = [], []
    for i, d in enumerate(days):
        px = 100.0 if (not jump or d >= split_at) else 50.0
        az.append((d, px))
        # Varying benchmark returns. A constant daily rate has zero variance
        # and beta is undefined.
        xbi_rows.append((d, 100.0 + (i % 5)))
    return az, xbi_rows, split_at


def test_beta_factor2_jumped_series_adjusts_once(tmp_path):
    from run_screen import _hydrate_beta_rsi

    az, xbi, split_at = _beta_prices(jump=True)
    _write_prices(tmp_path / "price_history.csv", {"AZ2": az, "XBI": xbi})
    reg = _registry(
        [
            {
                "ticker": "AZ2",
                "action": "reverse_split",
                "effective_date": split_at.isoformat(),
                "ratio": "1:2",
                "factor": 2.0,
            }
        ]
    )
    recs = {"AZ2": {"ticker": "AZ2", "defensive_features": {}}}
    _hydrate_beta_rsi(recs, tmp_path / "price_history.csv", AS_OF, corporate_actions=reg)
    df = recs["AZ2"]["defensive_features"]
    # Adjusted series is flat, so beta and alpha are ~0. The raw +100% day is not.
    assert abs(df["beta_xbi_60d"]) < 1e-6
    assert abs(df["alpha_60d"]) < 1e-6


def test_beta_factor2_flat_series_is_not_adjusted(tmp_path):
    from run_screen import _hydrate_beta_rsi

    az, xbi, split_at = _beta_prices(jump=False)
    _write_prices(tmp_path / "price_history.csv", {"AZ2": az, "XBI": xbi})
    reg = _registry(
        [
            {
                "ticker": "AZ2",
                "action": "reverse_split",
                "effective_date": split_at.isoformat(),
                "ratio": "1:2",
                "factor": 2.0,
            }
        ]
    )
    recs = {"AZ2": {"ticker": "AZ2", "defensive_features": {}}}
    _hydrate_beta_rsi(recs, tmp_path / "price_history.csv", AS_OF, corporate_actions=reg)
    df = recs["AZ2"]["defensive_features"]
    assert abs(df["beta_xbi_60d"]) < 1e-6
    assert abs(df["alpha_60d"]) < 1e-6
