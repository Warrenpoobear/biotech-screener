"""Tests for the split-adjusted price refresh step and freshness gate.

`production_data/price_history_split_adj.csv` is the price source declared in
docs/FORWARD_VALIDATION_PROTOCOL.md, but nothing in the pipeline refreshed it —
it silently drifted 17 trading days behind the raw series (last 2026-07-10 vs
raw 2026-08-03) before being caught by hand.

The subtle hazard is that split adjustment is RETROACTIVE: a new split rewrites
every earlier price for that ticker, which can change the inputs to
already-completed forward-validation windows. These tests pin the distinction
between a harmless pure append and a retroactive rewrite that an operator must
be told about.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import tools.run_daily_production as rdp  # noqa: E402

DATES = [f"2026-06-{d:02d}" for d in range(1, 26)]


def _write_prices(path: Path, rows) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "ticker", "close", "open", "high", "low", "volume"])
        for d, t, c in rows:
            w.writerow([d, t, c, c, c, c, 1000])


def _rows(split_on: str | None = None, factor: float = 0.1):
    """Flat $100 AAA + $50 XBI; AAA drops by `factor` from `split_on` onward."""
    out = []
    for d in DATES:
        px = 100.0 * factor if (split_on and d >= split_on) else 100.0
        out.append((d, "AAA", px))
        out.append((d, "XBI", 50.0))
    return out


def _seed_incumbent(tmp_path: Path, raw_rows, through: str) -> Path:
    """Build a self-consistent incumbent split-adj file from a truncated raw series."""
    trunc = tmp_path / "trunc.csv"
    adj = tmp_path / "price_history_split_adj.csv"
    _write_prices(trunc, [r for r in raw_rows if r[0] <= through])
    subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "repair_price_history_splits.py"),
            "--prices",
            str(trunc),
            "--out",
            str(adj),
        ],
        capture_output=True,
        cwd=str(REPO),
        check=True,
    )
    return adj


def test_pure_append_reports_no_retroactive_change(tmp_path):
    """No split in the gap → pure append, history untouched."""
    raw_rows = _rows()
    raw = tmp_path / "price_history.csv"
    _write_prices(raw, raw_rows)
    adj = _seed_incumbent(tmp_path, raw_rows, "2026-06-15")

    stats = rdp.refresh_split_adjusted_prices(raw, adj, REPO)

    assert stats["status"] == "APPENDED"
    assert stats["n_retroactive"] == 0
    assert stats["n_lost"] == 0
    assert stats["n_appended"] > 0
    assert stats["last_date"] == "2026-06-25"


def test_split_in_the_gap_is_flagged_as_retroactive(tmp_path):
    """A split inside the gap rewrites earlier prices — must NOT pass silently."""
    raw_rows = _rows(split_on="2026-06-18")
    raw = tmp_path / "price_history.csv"
    _write_prices(raw, raw_rows)
    adj = _seed_incumbent(tmp_path, raw_rows, "2026-06-15")

    stats = rdp.refresh_split_adjusted_prices(raw, adj, REPO)

    assert stats["status"] == "RETROACTIVE_CHANGE"
    assert stats["n_retroactive"] > 0
    assert "AAA" in stats["retroactive_tickers"]
    # The corrected values are still installed — correctness wins, but loudly.
    assert stats["last_date"] == "2026-06-25"


def test_refresh_is_idempotent(tmp_path):
    """Re-running against an already-current file is a no-op."""
    raw_rows = _rows()
    raw = tmp_path / "price_history.csv"
    _write_prices(raw, raw_rows)
    adj = _seed_incumbent(tmp_path, raw_rows, DATES[-1])

    stats = rdp.refresh_split_adjusted_prices(raw, adj, REPO)

    assert stats["status"] == "APPENDED"
    assert stats["n_appended"] == 0
    assert stats["n_retroactive"] == 0


def test_freshness_gate_passes_when_current(tmp_path):
    raw = tmp_path / "price_history.csv"
    adj = tmp_path / "price_history_split_adj.csv"
    _write_prices(raw, _rows())
    _write_prices(adj, _rows())

    gate = rdp.check_split_adj_freshness(raw, adj)

    assert gate.status == "PASS"
    assert gate.value == 0


def test_freshness_gate_warns_when_behind(tmp_path):
    """This is the drift that went unnoticed for 17 trading days."""
    raw = tmp_path / "price_history.csv"
    adj = tmp_path / "price_history_split_adj.csv"
    _write_prices(raw, _rows())
    _write_prices(adj, [r for r in _rows() if r[0] <= "2026-06-05"])

    gate = rdp.check_split_adj_freshness(raw, adj)

    assert gate.status == "WARN"
    assert gate.value > 4
    assert "behind raw" in gate.detail


def test_freshness_gate_warns_when_missing(tmp_path):
    raw = tmp_path / "price_history.csv"
    _write_prices(raw, _rows())

    gate = rdp.check_split_adj_freshness(raw, tmp_path / "nope.csv")

    assert gate.status == "WARN"
    assert "missing" in gate.detail


def test_refresh_never_raises_on_missing_script(tmp_path):
    """The step is non-blocking: a broken environment must not abort the run."""
    raw = tmp_path / "price_history.csv"
    _write_prices(raw, _rows())

    stats = rdp.refresh_split_adjusted_prices(raw, tmp_path / "adj.csv", tmp_path)

    assert stats["status"] == "SCRIPT_MISSING"


@pytest.mark.parametrize("gap_days,expected", [(0, "PASS"), (14, "WARN")])
def test_gate_threshold_boundary(tmp_path, gap_days, expected):
    raw = tmp_path / "price_history.csv"
    adj = tmp_path / "price_history_split_adj.csv"
    _write_prices(raw, _rows())
    cutoff = DATES[-1] if gap_days == 0 else "2026-06-05"
    _write_prices(adj, [r for r in _rows() if r[0] <= cutoff])

    assert rdp.check_split_adj_freshness(raw, adj).status == expected
