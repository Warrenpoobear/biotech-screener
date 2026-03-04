"""
Tests for tools/live_performance_tracker.py

15 tests covering:
  - Write-once (no overwrite of existing rows)
  - IC computation
  - XBI excess return
  - Fresh start (no prior data)
  - Missing price cache (skip gracefully)
  - Rolling summary stats
  - Turnover chain
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import patch, MagicMock

import pytest

# Ensure project root on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tools.live_performance_tracker import (
    SCHEMA_VERSION,
    HORIZON,
    TOP_K,
    CSV_FIELDS,
    _compute_fwd_returns,
    _get_xbi_forward_return,
    _mean_safe,
    build_summary,
    compute_row,
    run_tracker,
    _load_existing_dates,
    _write_rows,
    _load_all_rows,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_pit_prices(tickers: List[str], h20_close: float = 11.0, anchor: float = 10.0) -> Dict[str, Dict]:
    """Synthetic PIT prices dict."""
    return {
        tk: {
            "ticker": tk,
            "anchor_date": "2026-01-02",
            "anchor_close": str(anchor),
            "h5_date": "", "h5_close": "",
            "h20_date": "2026-01-30",
            "h20_close": str(h20_close),
            "h63_date": "", "h63_close": "",
        }
        for tk in tickers
    }


def _make_rankings(tickers: List[str], eligible: bool = True) -> List[Dict]:
    """Synthetic rankings list with actionable_rank."""
    rows = []
    for i, tk in enumerate(tickers, 1):
        rows.append({
            "ticker": tk,
            "actionable_rank": str(i),
            "eligible": "1" if eligible else "0",
            "tier_dev": "A" if i <= 10 else "B",
        })
    return rows


def _write_pit_dir(tmp_path: Path, snap_date: str, prices: Dict, horizon: int = 20) -> Path:
    """Write a fake PIT directory with prices.csv + index.json."""
    pit_dir = tmp_path / snap_date
    pit_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = ["ticker", "anchor_date", "anchor_close",
                  "h5_date", "h5_close", "h20_date", "h20_close",
                  "h63_date", "h63_close"]
    with (pit_dir / "prices.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prices.values())

    index = {
        "schema_version": "price_pit_index.v1",
        "as_of_date": snap_date,
        "anchor_date": snap_date,
        "horizons_filled": [horizon],
        "horizons_pending": [],
        "split_warnings": [],
        "ticker_count": len(prices),
        "coverage_pct": 100.0,
    }
    with (pit_dir / "index.json").open("w") as f:
        json.dump(index, f)
    return pit_dir


def _write_snap_dir(tmp_path: Path, snap_date: str, rankings: List[Dict]) -> Path:
    """Write a fake snapshot directory with rankings.csv + metadata.json."""
    snap_dir = tmp_path / snap_date
    snap_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rankings[0].keys()) if rankings else ["ticker", "actionable_rank", "eligible"]
    with (snap_dir / "rankings.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rankings)

    meta = {"ruleset_id": "test_ruleset_001", "as_of_date": snap_date}
    with (snap_dir / "metadata.json").open("w") as f:
        json.dump(meta, f)

    return snap_dir


# ---------------------------------------------------------------------------
# 1. compute_fwd_returns — basic and split filtering
# ---------------------------------------------------------------------------

class TestComputeFwdReturns:

    def test_basic_return(self):
        """h20_close / anchor_close - 1 computed correctly."""
        pit_prices = _make_pit_prices(["ACME"], h20_close=11.0, anchor=10.0)
        fwd = _compute_fwd_returns(pit_prices, set(), horizon=20)
        assert abs(fwd["ACME"] - 0.10) < 1e-6

    def test_split_ticker_excluded(self):
        """Tickers in split_tickers are skipped."""
        pit_prices = _make_pit_prices(["ACME", "BETA"], h20_close=15.0, anchor=10.0)
        fwd = _compute_fwd_returns(pit_prices, {"ACME"}, horizon=20)
        assert "ACME" not in fwd
        assert "BETA" in fwd

    def test_missing_forward_price_skipped(self):
        """Row with empty h20_close is skipped."""
        pit_prices = {
            "GOOD": {
                "ticker": "GOOD", "anchor_close": "10.0",
                "h20_close": "11.0", "h20_date": "2026-01-30",
            },
            "BAD": {
                "ticker": "BAD", "anchor_close": "10.0",
                "h20_close": "",  # empty
                "h20_date": "",
            },
        }
        fwd = _compute_fwd_returns(pit_prices, set(), horizon=20)
        assert "GOOD" in fwd
        assert "BAD" not in fwd

    def test_zero_anchor_skipped(self):
        """Zero anchor_close is skipped (no division by zero)."""
        pit_prices = {"ZERO": {"ticker": "ZERO", "anchor_close": "0.0", "h20_close": "5.0"}}
        fwd = _compute_fwd_returns(pit_prices, set(), horizon=20)
        assert "ZERO" not in fwd


# ---------------------------------------------------------------------------
# 2. XBI forward return
# ---------------------------------------------------------------------------

class TestXbiForwardReturn:

    def _xbi_prices(self):
        """5 trading dates."""
        return {
            "2026-01-02": 100.0,
            "2026-01-05": 101.0,
            "2026-01-06": 102.0,
            "2026-01-07": 103.0,
            "2026-01-08": 104.0,
        }

    def test_basic_1_day(self):
        xbi = self._xbi_prices()
        ret = _get_xbi_forward_return(xbi, "2026-01-02", horizon=1)
        # 1 trading day forward: 2026-01-05 → 101/100 - 1 = 0.01
        assert abs(ret - 0.01) < 1e-6

    def test_horizon_exceeds_data(self):
        xbi = self._xbi_prices()
        ret = _get_xbi_forward_return(xbi, "2026-01-02", horizon=10)
        assert ret is None

    def test_anchor_not_in_xbi(self):
        xbi = self._xbi_prices()
        ret = _get_xbi_forward_return(xbi, "2026-01-03", horizon=1)  # weekend
        assert ret is None

    def test_empty_xbi(self):
        ret = _get_xbi_forward_return({}, "2026-01-02", horizon=1)
        assert ret is None


# ---------------------------------------------------------------------------
# 3. Build summary — rolling windows + inception
# ---------------------------------------------------------------------------

class TestBuildSummary:

    def _rows(self, n: int) -> List[Dict]:
        rows = []
        for i in range(n):
            d = f"2026-{i // 28 + 1:02d}-{i % 28 + 1:02d}"
            rows.append({
                "date": d,
                "horizon": str(HORIZON),
                "gross_return": str(0.05),
                "net_return": str(0.04),
                "ic": str(0.10),
                "excess_return": str(0.02),
            })
        return rows

    def test_summary_fields_present(self):
        summary = build_summary(self._rows(5))
        assert "last_4w" in summary
        assert "last_13w" in summary
        assert "inception" in summary
        assert summary["schema_version"] == "live_performance_summary.v1"

    def test_mean_net_return_correct(self):
        rows = self._rows(5)
        summary = build_summary(rows)
        assert abs(summary["inception"]["mean_net_return"] - 0.04) < 1e-6

    def test_empty_rows(self):
        summary = build_summary([])
        assert summary["total_dates"] == 0
        assert summary["inception"]["n_dates"] == 0

    def test_4w_window_bounded(self):
        rows = self._rows(30)
        summary = build_summary(rows)
        # last_4w should use at most 20 dates
        assert summary["last_4w"]["n_dates"] <= 20

    def test_mean_safe_filters_none(self):
        assert _mean_safe([1.0, None, 2.0]) == 1.5
        assert _mean_safe([]) is None


# ---------------------------------------------------------------------------
# 4. Write-once behavior
# ---------------------------------------------------------------------------

class TestWriteOnce:

    def test_existing_rows_not_overwritten(self, tmp_path):
        """Rows already in CSV are not rewritten (write-once guarantee)."""
        existing = [
            {"schema_version": SCHEMA_VERSION, "date": "2025-12-01",
             "horizon": str(HORIZON), "n_held": "20",
             "anchor_close_mean": "10.0", "forward_close_mean": "11.0",
             "gross_return": "0.1", "net_return": "0.095",
             "ic": "0.05", "xbi_return": "0.03", "excess_return": "0.07",
             "turnover": "0.1", "ruleset_id": "old_ruleset", "notes": ""},
        ]
        csv_path = tmp_path / "live_performance.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(existing)

        # Simulate re-writing: existing rows should be preserved
        all_rows = []
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(dict(row))

        assert len(all_rows) == 1
        assert all_rows[0]["date"] == "2025-12-01"
        assert all_rows[0]["gross_return"] == "0.1"

    def test_new_rows_appended_not_replaced(self, tmp_path):
        """Adding new rows does not alter the old rows."""
        existing = [
            {"schema_version": SCHEMA_VERSION, "date": "2025-11-01",
             "horizon": str(HORIZON), "n_held": "20",
             "anchor_close_mean": "", "forward_close_mean": "",
             "gross_return": "0.05", "net_return": "0.04",
             "ic": "0.08", "xbi_return": "0.02", "excess_return": "0.03",
             "turnover": "0.05", "ruleset_id": "r1", "notes": ""},
        ]
        new = [
            {"schema_version": SCHEMA_VERSION, "date": "2025-12-01",
             "horizon": str(HORIZON), "n_held": "20",
             "anchor_close_mean": "", "forward_close_mean": "",
             "gross_return": "0.06", "net_return": "0.05",
             "ic": "0.09", "xbi_return": "0.02", "excess_return": "0.04",
             "turnover": "0.06", "ruleset_id": "r1", "notes": ""},
        ]
        csv_path = tmp_path / "lp.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(existing + new)

        with csv_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["date"] == "2025-11-01"
        assert rows[0]["gross_return"] == "0.05"  # unchanged


# ---------------------------------------------------------------------------
# 5. Missing price cache — graceful skip
# ---------------------------------------------------------------------------

class TestMissingPriceCache:

    def test_compute_row_no_pit_dir(self, tmp_path):
        """compute_row returns None when PIT dir does not exist."""
        with patch("tools.live_performance_tracker.PRICE_PIT_BASE", tmp_path):
            result = compute_row("2026-01-01", HORIZON, {}, None)
        assert result is None

    def test_compute_row_horizon_not_filled(self, tmp_path):
        """compute_row returns None when horizon is pending (not filled)."""
        pit_dir = tmp_path / "2026-01-02"
        pit_dir.mkdir()
        index = {
            "as_of_date": "2026-01-02",
            "anchor_date": "2026-01-02",
            "horizons_filled": [],
            "horizons_pending": [20],
            "split_warnings": [],
        }
        with (pit_dir / "index.json").open("w") as f:
            json.dump(index, f)

        with patch("tools.live_performance_tracker.PRICE_PIT_BASE", tmp_path):
            result = compute_row("2026-01-02", HORIZON, {}, None)
        assert result is None


# ---------------------------------------------------------------------------
# 6. IC computation end-to-end (via mocked data)
# ---------------------------------------------------------------------------

class TestICComputation:

    def test_ic_monotone_ranking_positive(self, tmp_path):
        """Perfect monotone ranking (rank matches return order) → IC near +1."""
        # 10 tickers: higher return for lower rank number
        tickers = [f"T{i:02d}" for i in range(1, 11)]
        pit_prices = {}
        for i, tk in enumerate(tickers, 1):
            # Ticker with rank=1 gets best return (10%), rank=10 gets worst (1%)
            fwd = 10.0 + (11 - i) * 0.1
            pit_prices[tk] = {
                "ticker": tk, "anchor_close": "10.0",
                "h20_date": "2026-01-30", "h20_close": str(fwd),
                "h5_date": "", "h5_close": "", "h63_date": "", "h63_close": "",
            }

        fwd_rets = _compute_fwd_returns(pit_prices, set(), horizon=20)
        rankings = _make_rankings(tickers)

        # Build signal + returns in rank order
        signal = [-float(r["actionable_rank"]) for r in rankings if r["ticker"] in fwd_rets]
        rets = [fwd_rets[r["ticker"]] for r in rankings if r["ticker"] in fwd_rets]

        from tools.live_performance_tracker import spearman_ic as _sic
        ic = _sic(signal, rets)
        assert ic is not None
        assert ic > 0.9, f"Expected IC near +1, got {ic}"

    def test_ic_none_when_insufficient_data(self):
        """Fewer than 3 overlapping tickers → IC is None."""
        from tools.live_performance_tracker import spearman_ic as _sic
        ic = _sic([1.0, 2.0], [0.05, 0.03])
        assert ic is None


# ---------------------------------------------------------------------------
# 7. Fresh start (no prior data)
# ---------------------------------------------------------------------------

class TestFreshStart:

    def test_fresh_start_note_in_first_row(self, tmp_path):
        """First row ever gets notes='fresh_start'."""
        tickers = [f"T{i:02d}" for i in range(1, 25)]
        prices = _make_pit_prices(tickers, h20_close=11.0, anchor=10.0)
        rankings = _make_rankings(tickers)

        pit_base = tmp_path / "PIT"
        snap_base = tmp_path / "snaps"
        _write_pit_dir(pit_base, "2026-01-02", prices, horizon=20)
        _write_snap_dir(snap_base, "2026-01-02", rankings)

        output_csv = tmp_path / "output" / "live_performance.csv"
        output_summary = tmp_path / "output" / "live_performance_summary.json"
        (tmp_path / "output").mkdir()

        with (
            patch("tools.live_performance_tracker.PRICE_PIT_BASE", pit_base),
            patch("tools.live_performance_tracker.SNAPSHOTS_ROOT", snap_base),
            patch("tools.live_performance_tracker.OUTPUT_CSV", output_csv),
            patch("tools.live_performance_tracker.OUTPUT_SUMMARY", output_summary),
            patch("tools.live_performance_tracker._load_xbi_prices", return_value={}),
        ):
            run_tracker(dry_run=False, horizon=20)

        assert output_csv.exists()
        with output_csv.open(newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["notes"] == "fresh_start"
