"""Registry splits below the gap-detection thresholds (2026-11-18).

AZN's ADR ratio changed 2:1 on 2026-01-28 (one ADR went from half an ordinary
share to one share), so the quoted price doubled overnight. The repair only looked
at single-day moves of +300% / -75%, so the +96% jump stayed in
price_history_split_adj.csv as a real return. A registry split whose factor sits
inside the thresholds must be applied when the raw series shows a one-day move
matching that factor; without a registry entry, or with a move that does not
match, nothing changes.
"""

import csv
import json
import sys
from datetime import date, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from repair_price_history_splits import repair_price_history  # noqa: E402

CHANGE_DAY = date(2026, 1, 28)


def _bdays(start: date, end: date):
    d, out = start, []
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _write_prices(path: Path, series: dict) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "ticker", "close", "open", "high", "low", "volume"])
        for ticker, rows in series.items():
            for d, c in rows:
                w.writerow([d.isoformat(), ticker, c, c, c, c, 1000])


def _write_registry(path: Path, actions: list) -> None:
    path.write_text(json.dumps({"_schema": "corporate_actions.v1", "actions": actions}))


def _closes(path: Path, ticker: str) -> list:
    return [float(r["close"]) for r in csv.DictReader(open(path)) if r["ticker"] == ticker]


def _ratio_series(after: float):
    """95.60 flat, then `after` flat from CHANGE_DAY (2x = ADR ratio change, no real move)."""
    days = _bdays(date(2025, 12, 1), date(2026, 2, 27))
    return [(d, 95.6 if d < CHANGE_DAY else after) for d in days]


ADR_RATIO = {
    "ticker": "AAA",
    "action": "reverse_split",
    "effective_date": CHANGE_DAY.isoformat(),
    "ratio": "1:2",
    "factor": 2.0,
    "notes": "ADR ratio change",
}


def test_registry_ratio_change_below_threshold_is_applied(tmp_path):
    _write_prices(tmp_path / "p.csv", {"AAA": _ratio_series(191.2)})
    _write_registry(tmp_path / "corporate_actions.json", [ADR_RATIO])
    out = tmp_path / "adj.csv"
    summary = repair_price_history(tmp_path / "p.csv", out)
    assert set(_closes(out, "AAA")) == {191.2}
    assert summary["split_details"]["AAA"][0]["date"] == CHANGE_DAY.isoformat()


def test_registry_entry_with_mismatched_move_is_not_applied(tmp_path):
    # Registry says 2x, but the series only rose 10%: a real move, not this split.
    _write_prices(tmp_path / "p.csv", {"AAA": _ratio_series(105.16)})
    _write_registry(tmp_path / "corporate_actions.json", [ADR_RATIO])
    out = tmp_path / "adj.csv"
    summary = repair_price_history(tmp_path / "p.csv", out)
    assert max(_closes(out, "AAA")) == 105.16
    assert min(_closes(out, "AAA")) == 95.6
    assert "AAA" not in summary["split_details"]


def test_unregistered_doubling_is_left_as_a_return(tmp_path):
    _write_prices(tmp_path / "p.csv", {"AAA": _ratio_series(191.2)})
    _write_registry(tmp_path / "corporate_actions.json", [])
    out = tmp_path / "adj.csv"
    repair_price_history(tmp_path / "p.csv", out)
    assert min(_closes(out, "AAA")) == 95.6


def test_registry_has_azn_adr_ratio_change():
    reg = json.loads((REPO_ROOT / "production_data" / "corporate_actions.json").read_text())
    got = {(a.get("ticker"), a.get("action"), a.get("effective_date"), a.get("factor")) for a in reg["actions"]}
    assert ("AZN", "reverse_split", "2026-01-28", 2.0) in got
