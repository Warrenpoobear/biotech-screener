"""Tests for scripts/eval_forward_returns.py."""
from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from scripts.eval_forward_returns import (
    DEFAULT_MIN_PRICE_COVERAGE,
    DateResult,
    EvalSummary,
    _avg_ranks,
    _cumulative,
    _trading_days_after,
    compute_forward_return,
    compute_turnover,
    discover_snapshot_dates,
    evaluate,
    load_price_series,
    net_return,
    spearman_ic,
    top_k_portfolio_return,
    write_by_date_csv,
    write_skips_json,
    write_summary_json,
    write_summary_md,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _write_price_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    """Write a minimal price_history.csv."""
    fieldnames = ["date", "ticker", "close"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_rankings_csv(snap_dir: Path, tickers: List[str]) -> None:
    """Write a minimal rankings.csv with actionable_rank 1..N."""
    snap_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["ticker", "actionable_rank", "eligible", "tier_dev",
                   "composite_rank", "composite_score", "archetype"]
    with open(snap_dir / "rankings.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, t in enumerate(tickers):
            writer.writerow({
                "ticker": t, "actionable_rank": str(i + 1),
                "eligible": "1", "tier_dev": "A",
                "composite_rank": str(i + 1), "composite_score": "50.0",
                "archetype": "drug_developer",
            })


def _write_metadata(snap_dir: Path, date_str: str) -> None:
    """Write metadata.json with as_of_date."""
    snap_dir.mkdir(parents=True, exist_ok=True)
    with open(snap_dir / "metadata.json", "w") as f:
        json.dump({"as_of_date": date_str}, f)


# ---------------------------------------------------------------------------
# Unit tests: turnover
# ---------------------------------------------------------------------------

class TestTurnover:
    def test_identical_sets(self):
        assert compute_turnover(["A", "B", "C"], ["A", "B", "C"]) == 0.0

    def test_completely_different(self):
        turn = compute_turnover(["A", "B"], ["C", "D"])
        # symmetric diff = 4, max(2,2) = 2 → 0.5 * 4/2 = 1.0
        assert turn == 1.0

    def test_partial_overlap(self):
        turn = compute_turnover(["A", "B", "C"], ["A", "B", "D"])
        # symmetric diff = 2, max(3,3) = 3 → 0.5 * 2/3 ≈ 0.3333
        assert abs(turn - 1 / 3) < 1e-6

    def test_empty_both(self):
        assert compute_turnover([], []) == 0.0

    def test_empty_prev(self):
        turn = compute_turnover([], ["A", "B"])
        # symmetric diff = 2, max(0,2) = 2 → 0.5 * 2/2 = 0.5
        assert turn == 0.5


# ---------------------------------------------------------------------------
# Unit tests: net return
# ---------------------------------------------------------------------------

class TestNetReturn:
    def test_zero_cost(self):
        assert net_return(0.05, 0.3, 0) == 0.05

    def test_basic_haircut(self):
        # gross=0.05, turnover=0.5, cost=30bps → net = 0.05 - 0.5*30/10000
        result = net_return(0.05, 0.5, 30)
        expected = 0.05 - 0.5 * 30 / 10_000
        assert abs(result - expected) < 1e-10

    def test_high_turnover(self):
        # 100% turnover, 100 bps cost → haircut = 1.0 * 100/10000 = 0.01
        result = net_return(0.03, 1.0, 100)
        assert abs(result - 0.02) < 1e-10


# ---------------------------------------------------------------------------
# Unit tests: signal direction (IC)
# ---------------------------------------------------------------------------

class TestSpearmanIC:
    def test_perfect_positive(self):
        signal = [1.0, 2.0, 3.0, 4.0, 5.0]
        returns = [0.01, 0.02, 0.03, 0.04, 0.05]
        ic = spearman_ic(signal, returns)
        assert ic is not None
        assert abs(ic - 1.0) < 1e-6

    def test_perfect_negative(self):
        signal = [5.0, 4.0, 3.0, 2.0, 1.0]
        returns = [0.01, 0.02, 0.03, 0.04, 0.05]
        ic = spearman_ic(signal, returns)
        assert ic is not None
        assert abs(ic - (-1.0)) < 1e-6

    def test_too_few(self):
        assert spearman_ic([1.0, 2.0], [0.1, 0.2]) is None

    def test_constant_signal(self):
        ic = spearman_ic([1.0, 1.0, 1.0], [0.1, 0.2, 0.3])
        assert ic is None  # zero std


# ---------------------------------------------------------------------------
# Unit tests: forward return
# ---------------------------------------------------------------------------

class TestForwardReturn:
    def test_basic(self):
        prices = {"2025-01-01": 100.0, "2025-01-02": 105.0, "2025-01-03": 110.0}
        dates = ["2025-01-01", "2025-01-02", "2025-01-03"]
        ret = compute_forward_return(prices, dates, "2025-01-01", 2)
        assert ret is not None
        assert abs(ret - 0.10) < 1e-6

    def test_horizon_beyond_data(self):
        prices = {"2025-01-01": 100.0, "2025-01-02": 105.0}
        dates = ["2025-01-01", "2025-01-02"]
        ret = compute_forward_return(prices, dates, "2025-01-01", 5)
        assert ret is None

    def test_missing_start_price(self):
        prices = {"2025-01-02": 105.0}
        dates = ["2025-01-01", "2025-01-02"]
        ret = compute_forward_return(prices, dates, "2025-01-01", 1)
        assert ret is None


# ---------------------------------------------------------------------------
# Unit tests: top-K portfolio
# ---------------------------------------------------------------------------

class TestTopKPortfolio:
    def test_basic(self):
        tickers = ["A", "B", "C"]
        fwd = {"A": 0.10, "B": 0.05, "C": -0.02}
        ret, n = top_k_portfolio_return(tickers, fwd, 2)
        assert n == 2
        assert ret is not None
        # mean of A=0.10, B=0.05
        assert abs(ret - 0.075) < 1e-6

    def test_no_returns(self):
        ret, n = top_k_portfolio_return(["A", "B"], {}, 2)
        assert ret is None
        assert n == 0


# ---------------------------------------------------------------------------
# Unit tests: coverage threshold skip
# ---------------------------------------------------------------------------

class TestCoverageSkip:
    def test_low_coverage_skips(self, tmp_dir):
        # Create snapshot with 10 tickers but only 3 have prices
        snap_dir = tmp_dir / "snapshots" / "2025-06-01"
        tickers = [f"T{i}" for i in range(10)]
        _write_rankings_csv(snap_dir, tickers)
        _write_metadata(snap_dir, "2025-06-01")

        price_csv = tmp_dir / "prices.csv"
        _write_price_csv(price_csv, [
            {"date": "2025-06-01", "ticker": "T0", "close": "100"},
            {"date": "2025-06-01", "ticker": "T1", "close": "100"},
            {"date": "2025-06-01", "ticker": "T2", "close": "100"},
            {"date": "2025-06-06", "ticker": "T0", "close": "110"},
            {"date": "2025-06-06", "ticker": "T1", "close": "105"},
            {"date": "2025-06-06", "ticker": "T2", "close": "95"},
        ])

        summary, results, skips = evaluate(
            snapshot_root=tmp_dir / "snapshots",
            price_csv=price_csv,
            horizons=[5],
            min_price_coverage=0.50,  # need 50% but only have 30%
        )
        # Coverage = 3/10 = 30%, below 50% → skip
        assert summary.n_skipped == 1
        assert any("LOW_COVERAGE" in s.get("reason", "") for s in skips)


# ---------------------------------------------------------------------------
# Unit tests: skip behavior (empty rankings)
# ---------------------------------------------------------------------------

class TestSkipBehavior:
    def test_empty_rankings_skip(self, tmp_dir):
        snap_dir = tmp_dir / "snapshots" / "2025-06-01"
        snap_dir.mkdir(parents=True)
        _write_metadata(snap_dir, "2025-06-01")
        # No rankings.csv

        price_csv = tmp_dir / "prices.csv"
        _write_price_csv(price_csv, [
            {"date": "2025-06-01", "ticker": "A", "close": "100"},
        ])

        summary, results, skips = evaluate(
            snapshot_root=tmp_dir / "snapshots",
            price_csv=price_csv,
            horizons=[5],
        )
        assert summary.n_skipped == 1
        assert any("EMPTY_RANKINGS" in s.get("reason", "") for s in skips)

    def test_pit_violation_skip(self, tmp_dir):
        snap_dir = tmp_dir / "snapshots" / "2025-06-01"
        _write_rankings_csv(snap_dir, ["A", "B"])
        # Write metadata with WRONG date
        with open(snap_dir / "metadata.json", "w") as f:
            json.dump({"as_of_date": "2025-05-30"}, f)

        price_csv = tmp_dir / "prices.csv"
        _write_price_csv(price_csv, [
            {"date": "2025-06-01", "ticker": "A", "close": "100"},
        ])

        summary, results, skips = evaluate(
            snapshot_root=tmp_dir / "snapshots",
            price_csv=price_csv,
            horizons=[5],
            pit_mode="strict",
        )
        assert summary.n_skipped == 1
        assert any("PIT" in s.get("reason", "") for s in skips)


# ---------------------------------------------------------------------------
# Unit tests: cumulative return
# ---------------------------------------------------------------------------

class TestCumulative:
    def test_basic(self):
        # (1+0.10) * (1+0.05) - 1 = 0.155
        assert abs(_cumulative([0.10, 0.05]) - 0.155) < 1e-6

    def test_empty(self):
        assert _cumulative([]) is None


# ---------------------------------------------------------------------------
# Integration: snapshot discovery
# ---------------------------------------------------------------------------

class TestDiscovery:
    def test_date_filtering(self, tmp_dir):
        snap_root = tmp_dir / "snapshots"
        for d in ["2025-01-01", "2025-02-01", "2025-03-01", "not-a-date", "2025-04-01"]:
            (snap_root / d).mkdir(parents=True)

        dates = discover_snapshot_dates(snap_root, "2025-01-15", "2025-03-15")
        assert dates == ["2025-02-01", "2025-03-01"]


# ---------------------------------------------------------------------------
# Integration: full evaluation with outputs
# ---------------------------------------------------------------------------

class TestFullEvaluation:
    def test_end_to_end(self, tmp_dir):
        snap_root = tmp_dir / "snapshots"
        out_dir = tmp_dir / "output"

        # Create 3 snapshots with 5 tickers
        tickers = ["A", "B", "C", "D", "E"]
        dates = ["2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09", "2025-01-10"]

        for snap_date in ["2025-01-06", "2025-01-07"]:
            _write_rankings_csv(snap_root / snap_date, tickers)
            _write_metadata(snap_root / snap_date, snap_date)

        # Write prices for all dates and tickers
        price_rows = []
        base_prices = {"A": 100, "B": 50, "C": 200, "D": 75, "E": 150}
        for i, d in enumerate(dates):
            for t, base in base_prices.items():
                price_rows.append({
                    "date": d, "ticker": t,
                    "close": str(base * (1 + 0.01 * (i + 1))),
                })

        price_csv = tmp_dir / "prices.csv"
        _write_price_csv(price_csv, price_rows)

        summary, results, skips = evaluate(
            snapshot_root=snap_root,
            price_csv=price_csv,
            horizons=[1, 2],
            top_k=3,
            cost_bps=30,
            min_price_coverage=0.50,
        )

        assert summary.n_dates == 2
        assert summary.n_evaluated >= 1
        assert 1 in summary.by_horizon
        assert 2 in summary.by_horizon

        # Write outputs
        out_dir.mkdir(parents=True, exist_ok=True)
        write_summary_json(summary, out_dir)
        write_summary_md(summary, out_dir)
        write_by_date_csv(results, out_dir)
        write_skips_json(skips, out_dir)

        assert (out_dir / "summary.json").exists()
        assert (out_dir / "summary.md").exists()
        assert (out_dir / "by_date.csv").exists()
        assert (out_dir / "skips.json").exists()

        # Validate JSON roundtrip
        with open(out_dir / "summary.json") as f:
            loaded = json.load(f)
        assert loaded["top_k"] == 3
        assert loaded["cost_bps"] == 30


# ---------------------------------------------------------------------------
# Unit tests: avg_ranks
# ---------------------------------------------------------------------------

class TestAvgRanks:
    def test_no_ties(self):
        ranks = _avg_ranks([10.0, 30.0, 20.0])
        assert ranks == [1.0, 3.0, 2.0]

    def test_ties(self):
        ranks = _avg_ranks([10.0, 10.0, 30.0])
        assert ranks[0] == 1.5
        assert ranks[1] == 1.5
        assert ranks[2] == 3.0
