"""Universe maintenance must only flag names the screen is expected to cover.

Retired names (delisted / excluded_acquired / retired) and the benchmark row stop
trading and are never ranked, so "stale price" and "missing from rankings" are
expected for them, not maintenance work. ``pending_coverage`` names are excluded
from scoring by module_1 until coverage is built, so they are expected to be
missing from rankings too (their other gaps are still worth reporting).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.build_universe_maintenance import build_universe_maintenance  # noqa: E402

AS_OF = "2026-11-16"


def _setup(tmp_path: Path):
    prod = tmp_path / "production_data"
    prod.mkdir()
    universe = [
        {"ticker": "ACT", "status": "active", "cik": "1"},
        {"ticker": "MISS", "status": "active", "cik": "2"},
        {"ticker": "DEL", "status": "delisted"},
        {"ticker": "ACQ", "status": "excluded_acquired"},
        {"ticker": "RET", "status": "retired"},
        {"ticker": "XBI", "status": "benchmark"},
        {"ticker": "PEND", "status": "pending_coverage"},
    ]
    (prod / "universe.json").write_text(json.dumps(universe))
    prices = ["date,ticker,close"]
    for t in ("ACT", "MISS", "PEND"):
        prices.append(f"{AS_OF},{t},10")
    for t in ("DEL", "ACQ", "RET"):
        prices.append(f"2026-01-02,{t},5")
    price_csv = prod / "price_history.csv"
    price_csv.write_text("\n".join(prices) + "\n")
    snaps = tmp_path / "snapshots" / AS_OF
    snaps.mkdir(parents=True)
    (snaps / "rankings.csv").write_text("ticker\nACT\n")
    ctgov = tmp_path / "ctgov"
    ctgov.mkdir()
    trials = [{"ticker": t} for t in ("ACT", "MISS")]
    (ctgov / "trial_records_2026-11-16.json").write_text(json.dumps(trials))
    return build_universe_maintenance(
        AS_OF,
        production_dir=prod,
        snapshots_dir=tmp_path / "snapshots",
        cache_dir=ctgov,
        price_csv=price_csv,
        out_dir=tmp_path / "out",
    )


def test_retired_and_benchmark_names_are_not_flagged(tmp_path):
    result = _setup(tmp_path)
    flagged = {r["ticker"] for r in result["flagged_tickers"]}
    assert not flagged & {"DEL", "ACQ", "RET", "XBI"}
    assert result["summary"]["price_stale_alert"] == 0
    assert result["summary"]["no_price_data"] == 0
    assert result["n_skipped_inactive"] == 4


def test_pending_coverage_not_flagged_missing_from_rankings(tmp_path):
    result = _setup(tmp_path)
    flags = {r["ticker"]: r["flags"] for r in result["flagged_tickers"]}
    assert "missing_from_rankings" not in flags.get("PEND", [])
    # other real gaps for pending names are still reported
    assert "no_ctgov_trials" in flags["PEND"]
    assert "no_cik" in flags["PEND"]


def test_active_name_missing_from_rankings_still_flagged(tmp_path):
    result = _setup(tmp_path)
    flags = {r["ticker"]: r["flags"] for r in result["flagged_tickers"]}
    assert flags["MISS"] == ["missing_from_rankings"]
    assert "ACT" not in flags
    assert result["summary"]["missing_from_rankings"] == 1


def test_report_written_to_out_dir(tmp_path):
    result = _setup(tmp_path)
    assert Path(result["_json_path"]).parent == tmp_path / "out"
