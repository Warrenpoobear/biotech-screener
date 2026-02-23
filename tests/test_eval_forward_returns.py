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
    _decile_spread,
    _beta_hedged_return,
    _monotonic_slope,
    _multi_ols,
    _trading_days_after,
    assign_splits,
    bottom_k_portfolio_return,
    compute_decile_curve,
    compute_forward_return,
    compute_residual_alpha,
    compute_residual_alpha_multi,
    compute_turnover,
    discover_snapshot_dates,
    evaluate,
    load_price_series,
    net_return,
    resolve_trade_date,
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


# ---------------------------------------------------------------------------
# Unit tests: resolve_trade_date (anchor-mode)
# ---------------------------------------------------------------------------

class TestResolveTradeDate:
    TRADING = ["2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09", "2025-01-10"]

    def test_exact_hit(self):
        assert resolve_trade_date(self.TRADING, "2025-01-07", "exact") == "2025-01-07"

    def test_exact_miss(self):
        """Weekend date with exact mode → None."""
        assert resolve_trade_date(self.TRADING, "2025-01-04", "exact") is None

    def test_next_trading_day_weekday(self):
        """Saturday → next Monday."""
        # 2025-01-04 is Saturday, next trading is 2025-01-06
        assert resolve_trade_date(self.TRADING, "2025-01-05", "next_trading_day") == "2025-01-06"

    def test_next_trading_day_friday(self):
        """Friday snap → Monday trade."""
        assert resolve_trade_date(self.TRADING, "2025-01-03", "next_trading_day") == "2025-01-06"

    def test_next_trading_day_on_trading_day(self):
        """Trade date as snap → NEXT trade date (strictly after)."""
        assert resolve_trade_date(self.TRADING, "2025-01-06", "next_trading_day") == "2025-01-07"

    def test_next_trading_day_past_end(self):
        """Past last date → None."""
        assert resolve_trade_date(self.TRADING, "2025-01-10", "next_trading_day") is None

    def test_prev_trading_day(self):
        """Weekend → last Friday."""
        assert resolve_trade_date(self.TRADING, "2025-01-11", "prev_trading_day") == "2025-01-10"

    def test_prev_trading_day_on_trading_day(self):
        """Trade date → itself (on or before)."""
        assert resolve_trade_date(self.TRADING, "2025-01-08", "prev_trading_day") == "2025-01-08"

    def test_prev_trading_day_before_all(self):
        """Before all dates → None."""
        assert resolve_trade_date(self.TRADING, "2025-01-01", "prev_trading_day") is None


# ---------------------------------------------------------------------------
# Unit tests: anchor-mode integration with evaluate()
# ---------------------------------------------------------------------------

class TestAnchorModeEval:
    """Anchor-mode end-to-end in evaluate()."""

    def test_weekend_exact_skips(self, tmp_dir):
        """Weekend snap with exact mode → ANCHOR_DATE_NOT_FOUND skip."""
        snap_dir = tmp_dir / "snapshots" / "2025-01-04"  # Saturday
        _write_rankings_csv(snap_dir, ["A", "B", "C"])
        _write_metadata(snap_dir, "2025-01-04")

        price_csv = tmp_dir / "prices.csv"
        # Only weekday prices
        _write_price_csv(price_csv, [
            {"date": "2025-01-03", "ticker": "A", "close": "100"},
            {"date": "2025-01-06", "ticker": "A", "close": "105"},
        ])

        summary, results, skips = evaluate(
            snapshot_root=tmp_dir / "snapshots",
            price_csv=price_csv,
            horizons=[1],
            anchor_mode="exact",
        )
        assert summary.n_skipped == 1
        assert any("ANCHOR_DATE_NOT_FOUND" in s["reason"] for s in skips)

    def test_weekend_next_trading_day_succeeds(self, tmp_dir):
        """Weekend snap with next_trading_day mode → anchors on Monday."""
        snap_dir = tmp_dir / "snapshots" / "2025-01-04"  # Saturday
        _write_rankings_csv(snap_dir, ["A", "B", "C"])
        _write_metadata(snap_dir, "2025-01-04")

        price_csv = tmp_dir / "prices.csv"
        dates = ["2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"]
        for d in dates:
            for t in ["A", "B", "C"]:
                _write_price_csv(price_csv, [])  # clear first
        # Write proper prices
        rows = []
        for i, d in enumerate(dates):
            for t in ["A", "B", "C"]:
                rows.append({"date": d, "ticker": t, "close": str(100 + i)})
        _write_price_csv(price_csv, rows)

        summary, results, skips = evaluate(
            snapshot_root=tmp_dir / "snapshots",
            price_csv=price_csv,
            horizons=[1],
            anchor_mode="next_trading_day",
            min_price_coverage=0.50,
        )
        # Should evaluate successfully (anchor on 2025-01-06)
        evaluated = [r for r in results if not r.skipped]
        assert len(evaluated) > 0
        assert evaluated[0].trade_date == "2025-01-06"

    def test_trade_date_in_results(self, tmp_dir):
        """trade_date field populated in DateResult."""
        snap_dir = tmp_dir / "snapshots" / "2025-01-06"
        _write_rankings_csv(snap_dir, ["A", "B"])
        _write_metadata(snap_dir, "2025-01-06")

        rows = []
        for d in ["2025-01-06", "2025-01-07", "2025-01-08"]:
            for t in ["A", "B"]:
                rows.append({"date": d, "ticker": t, "close": "100"})
        price_csv = tmp_dir / "prices.csv"
        _write_price_csv(price_csv, rows)

        _, results, _ = evaluate(
            snapshot_root=tmp_dir / "snapshots",
            price_csv=price_csv,
            horizons=[1],
            anchor_mode="exact",
        )
        for r in results:
            if not r.skipped:
                assert r.trade_date == "2025-01-06"


# ---------------------------------------------------------------------------
# Unit tests: decile spread
# ---------------------------------------------------------------------------

class TestDecileSpread:
    def test_basic_spread(self):
        """Top decile minus bottom decile of 20 tickers."""
        tickers = [f"T{i:02d}" for i in range(20)]
        # Returns: T00=10%, T01=9%, ... T19=-9%
        fwd = {f"T{i:02d}": 0.10 - 0.01 * i for i in range(20)}
        spread = _decile_spread(tickers, fwd)
        assert spread is not None
        # Decile size = 20//10 = 2
        # Top: T00(10%), T01(9%) → mean = 9.5%
        # Bottom: T18(-8%), T19(-9%) → mean = -8.5%
        # Spread = 9.5% - (-8.5%) = 18%
        assert abs(spread - 0.18) < 1e-6

    def test_too_few_tickers(self):
        """< 10 tickers → None."""
        tickers = ["A", "B", "C"]
        fwd = {"A": 0.10, "B": 0.05, "C": -0.02}
        assert _decile_spread(tickers, fwd) is None


# ---------------------------------------------------------------------------
# Unit tests: beta-hedged return
# ---------------------------------------------------------------------------

class TestBetaHedgedReturn:
    def test_basic_hedge(self):
        """Portfolio beta=1.0, benchmark=5% → hedge removes market."""
        held = ["A", "B"]
        fwd = {"A": 0.10, "B": 0.06}  # mean = 8%
        betas = {"A": 1.0, "B": 1.0}  # avg beta = 1.0
        bm_ret = 0.05
        hedged = _beta_hedged_return(held, fwd, bm_ret, betas)
        assert hedged is not None
        # hedged = 8% - 1.0 * 5% = 3%
        assert abs(hedged - 0.03) < 1e-6

    def test_low_beta_preserves_more(self):
        """Beta < 1 → less hedging."""
        held = ["A"]
        fwd = {"A": 0.10}
        betas = {"A": 0.5}
        bm_ret = 0.08
        hedged = _beta_hedged_return(held, fwd, bm_ret, betas)
        # hedged = 10% - 0.5 * 8% = 6%
        assert abs(hedged - 0.06) < 1e-6

    def test_missing_betas(self):
        """No betas → None."""
        held = ["A"]
        fwd = {"A": 0.10}
        betas = {}
        hedged = _beta_hedged_return(held, fwd, 0.05, betas)
        assert hedged is None


# ---------------------------------------------------------------------------
# Integration: benchmark-relative evaluation
# ---------------------------------------------------------------------------

class TestBenchmarkEval:
    def test_excess_return_computed(self, tmp_dir):
        """Benchmark=XBI → excess_return in results."""
        snap_dir = tmp_dir / "snapshots" / "2025-01-06"
        _write_rankings_csv(snap_dir, ["A", "B", "C"])
        _write_metadata(snap_dir, "2025-01-06")

        rows = []
        for d in ["2025-01-06", "2025-01-07"]:
            for t in ["A", "B", "C"]:
                rows.append({"date": d, "ticker": t, "close": str(100 if d == "2025-01-06" else 110)})
            # Benchmark
            rows.append({"date": d, "ticker": "XBI", "close": str(100 if d == "2025-01-06" else 105)})
        price_csv = tmp_dir / "prices.csv"
        _write_price_csv(price_csv, rows)

        summary, results, _ = evaluate(
            snapshot_root=tmp_dir / "snapshots",
            price_csv=price_csv,
            horizons=[1],
            anchor_mode="exact",
            benchmark="XBI",
        )
        evaluated = [r for r in results if not r.skipped]
        assert len(evaluated) > 0
        for r in evaluated:
            assert r.benchmark_return is not None
            assert r.excess_return is not None
            # portfolio=10%, XBI=5% → excess=5%
            assert abs(r.excess_return - 0.05) < 1e-4

    def test_summary_has_excess_fields(self, tmp_dir):
        """Summary by_horizon includes mean_excess_return when benchmark set."""
        snap_dir = tmp_dir / "snapshots" / "2025-01-06"
        _write_rankings_csv(snap_dir, ["A", "B"])
        _write_metadata(snap_dir, "2025-01-06")

        rows = []
        for d in ["2025-01-06", "2025-01-07"]:
            for t in ["A", "B", "XBI"]:
                rows.append({"date": d, "ticker": t, "close": "100"})
        price_csv = tmp_dir / "prices.csv"
        _write_price_csv(price_csv, rows)

        summary, _, _ = evaluate(
            snapshot_root=tmp_dir / "snapshots",
            price_csv=price_csv,
            horizons=[1],
            benchmark="XBI",
        )
        bh = summary.by_horizon.get(1, {})
        assert "mean_excess_return" in bh


# ---------------------------------------------------------------------------
# Integration: long-short decile in evaluate()
# ---------------------------------------------------------------------------

class TestLongShortEval:
    def test_ls_return_present(self, tmp_dir):
        """long_short_deciles=True → ls_return in results."""
        snap_dir = tmp_dir / "snapshots" / "2025-01-06"
        # Need >= 10 tickers for decile calculation
        tickers = [f"T{i:02d}" for i in range(20)]
        _write_rankings_csv(snap_dir, tickers)
        _write_metadata(snap_dir, "2025-01-06")

        rows = []
        for d in ["2025-01-06", "2025-01-07"]:
            for i, t in enumerate(tickers):
                base = 100 + i
                price = base if d == "2025-01-06" else base * (1 + 0.01 * (20 - i))
                rows.append({"date": d, "ticker": t, "close": str(price)})
        price_csv = tmp_dir / "prices.csv"
        _write_price_csv(price_csv, rows)

        summary, results, _ = evaluate(
            snapshot_root=tmp_dir / "snapshots",
            price_csv=price_csv,
            horizons=[1],
            long_short_deciles=True,
        )
        evaluated = [r for r in results if not r.skipped]
        assert len(evaluated) > 0
        for r in evaluated:
            assert r.ls_return is not None


# ---------------------------------------------------------------------------
# Integration: summary markdown with new sections
# ---------------------------------------------------------------------------

class TestSummaryMarkdown:
    def test_anchor_mode_in_md(self, tmp_dir):
        """Summary markdown mentions anchor mode."""
        summary = EvalSummary(
            horizons=[5], anchor_mode="next_trading_day",
            benchmark="XBI", n_dates=3, n_evaluated=2,
        )
        summary.by_horizon[5] = {
            "n_dates": 2, "mean_ic": 0.05, "median_ic": 0.04,
            "std_ic": 0.03, "mean_gross_return": 0.01,
            "mean_bottom_k_return": -0.01,
            "mean_net_return": 0.009, "cumulative_gross": 0.02,
            "cumulative_net": 0.018, "mean_turnover": 0.3,
            "sign_mismatches": 0,
            "mean_excess_return": 0.005, "cumulative_excess": 0.01,
        }
        out = tmp_dir / "out"
        out.mkdir()
        path = write_summary_md(summary, out)
        text = path.read_text()
        assert "next_trading_day" in text
        assert "first trading day after D" in text
        assert "Benchmark-Relative" in text
        assert "Sign Consistency" in text


# ---------------------------------------------------------------------------
# Sign consistency + sort bug fix tests
# ---------------------------------------------------------------------------

def _write_rankings_with_ineligible(snap_dir: Path, ranked: List[str],
                                     ineligible: List[str]) -> None:
    """Write rankings with ranked tickers (rank 1..N) + ineligible (empty rank)."""
    snap_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["ticker", "actionable_rank", "eligible", "tier_dev",
                   "composite_rank", "composite_score", "archetype"]
    with open(snap_dir / "rankings.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, t in enumerate(ranked):
            writer.writerow({
                "ticker": t, "actionable_rank": str(i + 1),
                "eligible": "1", "tier_dev": "A",
                "composite_rank": str(i + 1), "composite_score": "50.0",
                "archetype": "drug_developer",
            })
        for t in ineligible:
            writer.writerow({
                "ticker": t, "actionable_rank": "",
                "eligible": "0", "tier_dev": "",
                "composite_rank": "", "composite_score": "",
                "archetype": "drug_developer",
            })


class TestSortBugFix:
    """Verify the safe-rank sort handles empty actionable_rank."""

    def test_ranked_before_ineligible(self, tmp_dir):
        """Ranked tickers sorted first, ineligible pushed to back."""
        # Ranked tickers get high returns, ineligible get low
        ranked = ["GOOD1", "GOOD2", "GOOD3"]
        ineligible = ["BAD1", "BAD2"]
        snap_dir = tmp_dir / "snapshots" / "2025-01-06"
        _write_rankings_with_ineligible(snap_dir, ranked, ineligible)
        _write_metadata(snap_dir, "2025-01-06")

        # Good tickers go up, bad tickers go down
        rows = []
        for d in ["2025-01-06", "2025-01-07"]:
            for t in ranked:
                rows.append({"date": d, "ticker": t,
                             "close": str(100 if d == "2025-01-06" else 120)})
            for t in ineligible:
                rows.append({"date": d, "ticker": t,
                             "close": str(100 if d == "2025-01-06" else 80)})
        price_csv = tmp_dir / "prices.csv"
        _write_price_csv(price_csv, rows)

        _, results, _ = evaluate(
            snapshot_root=tmp_dir / "snapshots",
            price_csv=price_csv,
            horizons=[1],
            top_k=3,
        )
        evaluated = [r for r in results if not r.skipped]
        assert len(evaluated) == 1
        r = evaluated[0]
        # Top-3 should be GOOD1, GOOD2, GOOD3 with 20% return
        assert r.gross_return is not None
        assert abs(r.gross_return - 0.20) < 1e-4
        # Bottom-3 should include BAD1, BAD2 with -20% return
        assert r.bottom_k_return is not None
        assert r.bottom_k_return < 0


class TestSignConsistency:
    """Sign consistency: top-K should beat bottom-K when IC > 0."""

    def test_positive_signal_positive_ls(self, tmp_dir):
        """Known monotonic ranking → IC>0 and top beats bottom."""
        # 20 tickers: rank 1 has best return, rank 20 has worst
        n = 20
        tickers = [f"T{i:02d}" for i in range(n)]
        snap_dir = tmp_dir / "snapshots" / "2025-01-06"
        _write_rankings_csv(snap_dir, tickers)
        _write_metadata(snap_dir, "2025-01-06")

        rows = []
        for d in ["2025-01-06", "2025-01-07"]:
            for i, t in enumerate(tickers):
                if d == "2025-01-06":
                    rows.append({"date": d, "ticker": t, "close": "100"})
                else:
                    # Rank 1 → +20%, rank 20 → -18%
                    ret = 0.20 - 0.02 * i
                    rows.append({"date": d, "ticker": t,
                                 "close": str(100 * (1 + ret))})
        price_csv = tmp_dir / "prices.csv"
        _write_price_csv(price_csv, rows)

        summary, results, _ = evaluate(
            snapshot_root=tmp_dir / "snapshots",
            price_csv=price_csv,
            horizons=[1],
            top_k=5,
            long_short_deciles=True,
        )
        evaluated = [r for r in results if not r.skipped]
        assert len(evaluated) == 1
        r = evaluated[0]
        # IC should be positive (better rank → better return)
        assert r.ic is not None and r.ic > 0.5
        # Top-K should beat bottom-K
        assert r.gross_return is not None and r.bottom_k_return is not None
        assert r.gross_return > r.bottom_k_return
        # L/S should be positive
        assert r.ls_return is not None and r.ls_return > 0
        # Summary sign check should show 0 mismatches
        bh = summary.by_horizon.get(1, {})
        assert bh.get("sign_mismatches", 0) == 0

    def test_inverted_signal_negative_ls(self, tmp_dir):
        """Inverted ranking → IC<0 and top underperforms bottom."""
        n = 20
        tickers = [f"T{i:02d}" for i in range(n)]
        snap_dir = tmp_dir / "snapshots" / "2025-01-06"
        _write_rankings_csv(snap_dir, tickers)
        _write_metadata(snap_dir, "2025-01-06")

        rows = []
        for d in ["2025-01-06", "2025-01-07"]:
            for i, t in enumerate(tickers):
                if d == "2025-01-06":
                    rows.append({"date": d, "ticker": t, "close": "100"})
                else:
                    # INVERTED: rank 1 → worst return, rank 20 → best
                    ret = -0.18 + 0.02 * i
                    rows.append({"date": d, "ticker": t,
                                 "close": str(100 * (1 + ret))})
        price_csv = tmp_dir / "prices.csv"
        _write_price_csv(price_csv, rows)

        summary, results, _ = evaluate(
            snapshot_root=tmp_dir / "snapshots",
            price_csv=price_csv,
            horizons=[1],
            top_k=5,
            long_short_deciles=True,
        )
        evaluated = [r for r in results if not r.skipped]
        assert len(evaluated) == 1
        r = evaluated[0]
        # IC should be negative (better rank → worse return)
        assert r.ic is not None and r.ic < -0.5
        # Top-K should underperform bottom-K
        assert r.gross_return is not None and r.bottom_k_return is not None
        assert r.gross_return < r.bottom_k_return
        # L/S should be negative
        assert r.ls_return is not None and r.ls_return < 0


# ---------------------------------------------------------------------------
# Unit tests: decile curve + monotonic slope
# ---------------------------------------------------------------------------

class TestDecileCurve:
    def test_basic_curve(self):
        """20 tickers → 10 deciles of 2 each."""
        tickers = [f"T{i:02d}" for i in range(20)]
        fwd = {f"T{i:02d}": 0.20 - 0.02 * i for i in range(20)}
        curve = compute_decile_curve(tickers, fwd)
        assert curve is not None
        assert len(curve) == 10
        assert curve[0]["decile"] == 1
        assert curve[9]["decile"] == 10
        # Decile 1 (best) should have highest return
        assert curve[0]["mean_return"] > curve[9]["mean_return"]

    def test_too_few_tickers(self):
        tickers = ["A", "B", "C"]
        fwd = {"A": 0.10, "B": 0.05, "C": -0.02}
        assert compute_decile_curve(tickers, fwd) is None

    def test_curve_covers_all_tickers(self):
        """Total n across deciles equals eligible count."""
        tickers = [f"T{i:02d}" for i in range(25)]
        fwd = {f"T{i:02d}": 0.01 * i for i in range(25)}
        curve = compute_decile_curve(tickers, fwd)
        assert curve is not None
        total = sum(c["n"] for c in curve)
        assert total == 25


class TestMonotonicSlope:
    def test_positive_slope_monotonic(self):
        """Monotonically decreasing returns → positive slope (signal works)."""
        curve = [
            {"decile": d, "n": 10, "mean_return": 0.10 - 0.02 * (d - 1)}
            for d in range(1, 11)
        ]
        slope = _monotonic_slope(curve)
        assert slope is not None
        # Decile 1 has highest return, signal proxy = 10, so corr is positive
        assert slope > 0.9

    def test_negative_slope_inverted(self):
        """Monotonically increasing returns → negative slope (signal inverted)."""
        curve = [
            {"decile": d, "n": 10, "mean_return": -0.10 + 0.02 * (d - 1)}
            for d in range(1, 11)
        ]
        slope = _monotonic_slope(curve)
        assert slope is not None
        assert slope < -0.9

    def test_none_for_short_curve(self):
        assert _monotonic_slope(None) is None
        assert _monotonic_slope([{"decile": 1, "n": 5, "mean_return": 0.05}]) is None


# ---------------------------------------------------------------------------
# Unit tests: multi-variate OLS
# ---------------------------------------------------------------------------

class TestMultiOLS:
    def test_single_regressor_matches_simple(self):
        """_multi_ols with 1 regressor should match _simple_ols."""
        from scripts.eval_forward_returns import _simple_ols
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        y = [2.1, 4.0, 5.9, 8.1, 10.0, 12.1]
        alpha_s, beta_s, r2_s = _simple_ols(x, y)
        betas, r2_m = _multi_ols([x], y)
        assert abs(betas[0] - alpha_s) < 1e-6
        assert abs(betas[1] - beta_s) < 1e-6
        assert abs(r2_m - r2_s) < 1e-4

    def test_two_regressors(self):
        """Known synthetic: y = 1 + 2*x1 + 3*x2 with independent regressors."""
        n = 20
        x1 = [float(i) for i in range(n)]
        x2 = [float(i % 5) for i in range(n)]  # independent of x1
        y = [1.0 + 2.0 * x1[i] + 3.0 * x2[i] for i in range(n)]
        betas, r2 = _multi_ols([x1, x2], y)
        assert abs(betas[0] - 1.0) < 1e-4
        assert abs(betas[1] - 2.0) < 1e-4
        assert abs(betas[2] - 3.0) < 1e-4
        assert r2 > 0.99

    def test_too_few_observations(self):
        """Fewer than p+2 observations → zeros."""
        betas, r2 = _multi_ols([[1.0, 2.0], [3.0, 4.0]], [5.0, 6.0])
        assert all(b == 0.0 for b in betas)


# ---------------------------------------------------------------------------
# Unit tests: walk-forward splits
# ---------------------------------------------------------------------------

class TestAssignSplits:
    def test_none_mode(self):
        dates = ["2024-01", "2024-02", "2024-03"]
        result = assign_splits(dates, "none")
        assert all(v == (0, False, False) for v in result.values())

    def test_fixed_split(self):
        dates = ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05"]
        result = assign_splits(
            dates, "fixed", train_end="2024-03", test_start="2024-04"
        )
        assert result["2024-01"][1] is True  # is_train
        assert result["2024-01"][2] is False  # is_test
        assert result["2024-03"][1] is True
        assert result["2024-04"][2] is True  # is_test
        assert result["2024-05"][2] is True

    def test_walk_forward(self):
        dates = [f"2024-{m:02d}" for m in range(1, 11)]  # 10 dates
        result = assign_splits(
            dates, "walk_forward", wf_train_months=4, wf_test_months=3
        )
        # First 4 dates = train
        for d in dates[:4]:
            assert result[d][1] is True  # is_train
            assert result[d][2] is False
        # Next 3 = test split 1
        for d in dates[4:7]:
            assert result[d][0] == 1  # split_id
            assert result[d][2] is True  # is_test
        # Next 3 = test split 2
        for d in dates[7:10]:
            assert result[d][0] == 2  # split_id
            assert result[d][2] is True

    def test_walk_forward_train_only(self):
        """More train months than dates → all train."""
        dates = ["2024-01", "2024-02", "2024-03"]
        result = assign_splits(dates, "walk_forward", wf_train_months=5, wf_test_months=3)
        assert all(v[1] is True for v in result.values())  # all train
        assert all(v[2] is False for v in result.values())  # none test


# ---------------------------------------------------------------------------
# Unit tests: residual alpha regression
# ---------------------------------------------------------------------------

class TestResidualAlpha:
    def test_benchmark_explains_returns(self):
        """When L/S is purely driven by benchmark, residual alpha ≈ 0."""
        drs = []
        for i in range(20):
            bm = 0.01 * (i - 10)
            ls = 0.5 * bm  # L/S = 0.5 * benchmark (no alpha)
            drs.append(DateResult(
                date=f"2024-{i+1:02d}-01", horizon=5,
                gross_return=ls + 0.01, bottom_k_return=0.01,
                benchmark_return=bm,
            ))
        result = compute_residual_alpha(drs, 5)
        assert result["n"] == 20
        assert abs(result["alpha"]) < 0.01

    def test_multi_regressor_with_exposure(self):
        """Multi-regressor with known exposure drivers."""
        drs = []
        for i in range(20):
            bm = 0.01 * (i - 10)
            drs.append(DateResult(
                date=f"2024-{i+1:02d}-01", horizon=5,
                gross_return=bm * 0.3 + 0.005,
                bottom_k_return=0.0,
                benchmark_return=bm,
                topk_avg_beta=0.9, bottomk_avg_beta=1.1,
                topk_avg_drawdown=-0.2, bottomk_avg_drawdown=-0.5,
            ))
        result = compute_residual_alpha_multi(drs, 5)
        assert result["n"] == 20
        assert result["alpha"] is not None
        assert result["r2"] is not None


# ---------------------------------------------------------------------------
# Integration: monotonic slope in evaluate
# ---------------------------------------------------------------------------

class TestMonotonicInEval:
    def test_slope_populated(self, tmp_dir):
        """Monotonic slope field populated in DateResult."""
        n = 20
        tickers = [f"T{i:02d}" for i in range(n)]
        snap_dir = tmp_dir / "snapshots" / "2025-01-06"
        _write_rankings_csv(snap_dir, tickers)
        _write_metadata(snap_dir, "2025-01-06")

        rows = []
        for d in ["2025-01-06", "2025-01-07"]:
            for i, t in enumerate(tickers):
                if d == "2025-01-06":
                    rows.append({"date": d, "ticker": t, "close": "100"})
                else:
                    ret = 0.20 - 0.02 * i
                    rows.append({"date": d, "ticker": t,
                                 "close": str(100 * (1 + ret))})
        price_csv = tmp_dir / "prices.csv"
        _write_price_csv(price_csv, rows)

        _, results, _ = evaluate(
            snapshot_root=tmp_dir / "snapshots",
            price_csv=price_csv,
            horizons=[1],
        )
        evaluated = [r for r in results if not r.skipped]
        assert len(evaluated) == 1
        r = evaluated[0]
        assert r.monotonic_slope is not None
        # Monotonic ranking → slope should be positive
        assert r.monotonic_slope > 0.5


# ---------------------------------------------------------------------------
# Integration: sign mismatch JSON dump
# ---------------------------------------------------------------------------

class TestSignMismatchDump:
    def test_mismatch_json_written_deterministic(self, tmp_dir):
        """Deterministic mismatch: IC > 0.1 AND (top-bottom) < -0.05.

        Construction: 80 tickers. Only 3 at each tail inverted (3.75% each),
        middle 74 (92.5%) strongly monotonic → IC dominated by middle.
        Top-3 get -30%, bottom-3 get +35%. top_k=3.
        """
        n = 80
        tickers = [f"T{i:02d}" for i in range(n)]
        snap_dir = tmp_dir / "snapshots" / "2025-01-06"
        _write_rankings_csv(snap_dir, tickers)
        _write_metadata(snap_dir, "2025-01-06")

        rows = []
        for d in ["2025-01-06", "2025-01-07"]:
            for i, t in enumerate(tickers):
                if d == "2025-01-06":
                    rows.append({"date": d, "ticker": t, "close": "100"})
                else:
                    if i < 3:
                        ret = -0.30  # top-3 → very bad
                    elif i >= 77:
                        ret = 0.35  # bottom-3 → very good
                    else:
                        # Middle 74: strongly monotonic
                        # rank 3 → +25%, rank 76 → -15% (wide 40% spread)
                        ret = 0.25 - 0.005405 * (i - 3)
                    rows.append({"date": d, "ticker": t,
                                 "close": str(100 * (1 + ret))})
        price_csv = tmp_dir / "prices.csv"
        _write_price_csv(price_csv, rows)

        out = tmp_dir / "output"
        out.mkdir()

        _, results, _ = evaluate(
            snapshot_root=tmp_dir / "snapshots",
            price_csv=price_csv,
            horizons=[1],
            top_k=3,
            out_dir=out,
        )
        evaluated = [r for r in results if not r.skipped]
        assert len(evaluated) == 1
        r = evaluated[0]
        # Verify the construction works as intended
        assert r.ic is not None and r.ic > 0.1, f"IC={r.ic}, expected > 0.1"
        assert r.gross_return is not None and r.bottom_k_return is not None
        top_bottom = r.gross_return - r.bottom_k_return
        assert top_bottom < -0.05, f"top-bottom={top_bottom}, expected < -0.05"
        # Mismatch JSON MUST exist (no conditional)
        mismatch_files = list(out.glob("sign_mismatch_*.json"))
        assert len(mismatch_files) == 1
        data = json.loads(mismatch_files[0].read_text())
        assert "decile_curve" in data
        assert "top_tickers" in data
        assert "bottom_tickers" in data
        assert data["horizon"] == 1
        assert data["ic"] > 0.1

    def test_no_mismatch_for_correct_ranking(self, tmp_dir):
        """Correct monotonic ranking → no mismatch file."""
        n = 20
        tickers = [f"T{i:02d}" for i in range(n)]
        snap_dir = tmp_dir / "snapshots" / "2025-01-06"
        _write_rankings_csv(snap_dir, tickers)
        _write_metadata(snap_dir, "2025-01-06")

        rows = []
        for d in ["2025-01-06", "2025-01-07"]:
            for i, t in enumerate(tickers):
                if d == "2025-01-06":
                    rows.append({"date": d, "ticker": t, "close": "100"})
                else:
                    ret = 0.20 - 0.02 * i
                    rows.append({"date": d, "ticker": t,
                                 "close": str(100 * (1 + ret))})
        price_csv = tmp_dir / "prices.csv"
        _write_price_csv(price_csv, rows)

        out = tmp_dir / "output"
        out.mkdir()

        _, _, _ = evaluate(
            snapshot_root=tmp_dir / "snapshots",
            price_csv=price_csv,
            horizons=[1],
            out_dir=out,
        )
        mismatch_files = list(out.glob("sign_mismatch_*.json"))
        assert len(mismatch_files) == 0


# ---------------------------------------------------------------------------
# Integration: walk-forward split in evaluate
# ---------------------------------------------------------------------------

class TestWalkForwardEval:
    def test_split_tags_in_results(self, tmp_dir):
        """Walk-forward split tags appear in DateResult."""
        snap_root = tmp_dir / "snapshots"
        tickers = ["A", "B", "C", "D", "E"]
        dates = [f"2025-01-{d:02d}" for d in range(6, 11)]  # 5 trading days

        for snap_date in ["2025-01-06", "2025-01-07", "2025-01-08"]:
            _write_rankings_csv(snap_root / snap_date, tickers)
            _write_metadata(snap_root / snap_date, snap_date)

        rows = []
        for d in dates:
            for t in tickers:
                rows.append({"date": d, "ticker": t, "close": "100"})
        price_csv = tmp_dir / "prices.csv"
        _write_price_csv(price_csv, rows)

        _, results, _ = evaluate(
            snapshot_root=snap_root,
            price_csv=price_csv,
            horizons=[1],
            split_mode="walk_forward",
            wf_train_months=1,
            wf_test_months=1,
        )
        evaluated = [r for r in results if not r.skipped]
        # First date should be train
        train_dates = [r for r in evaluated if r.is_train]
        test_dates = [r for r in evaluated if r.is_test]
        assert len(train_dates) >= 1
        assert len(test_dates) >= 1

    def test_summary_has_split_metrics(self, tmp_dir):
        """Walk-forward summary includes train/test metrics."""
        snap_root = tmp_dir / "snapshots"
        tickers = [f"T{i}" for i in range(10)]
        dates_all = [f"2025-01-{d:02d}" for d in range(6, 16)]  # 10 trading days

        for snap_date in [f"2025-01-{d:02d}" for d in range(6, 12)]:
            _write_rankings_csv(snap_root / snap_date, tickers)
            _write_metadata(snap_root / snap_date, snap_date)

        rows = []
        for d in dates_all:
            for t in tickers:
                rows.append({"date": d, "ticker": t, "close": "100"})
        price_csv = tmp_dir / "prices.csv"
        _write_price_csv(price_csv, rows)

        summary, _, _ = evaluate(
            snapshot_root=snap_root,
            price_csv=price_csv,
            horizons=[1],
            split_mode="walk_forward",
            wf_train_months=2,
            wf_test_months=2,
        )
        bh = summary.by_horizon.get(1, {})
        assert "train_n" in bh
        assert "test_n" in bh


# ---------------------------------------------------------------------------
# Integration: interpretation notes in summary.md
# ---------------------------------------------------------------------------

class TestInterpretationNotes:
    def test_notes_in_md(self, tmp_dir):
        """Summary markdown includes interpretation notes section."""
        summary = EvalSummary(
            horizons=[5], anchor_mode="next_trading_day",
            benchmark="XBI", n_dates=3, n_evaluated=2,
        )
        summary.by_horizon[5] = {
            "n_dates": 2, "mean_ic": 0.05, "median_ic": 0.04,
            "std_ic": 0.03, "mean_gross_return": 0.01,
            "mean_bottom_k_return": -0.01,
            "mean_net_return": 0.009, "cumulative_gross": 0.02,
            "cumulative_net": 0.018, "mean_turnover": 0.3,
            "sign_mismatches": 0,
        }
        out = tmp_dir / "out"
        out.mkdir()
        path = write_summary_md(summary, out)
        text = path.read_text()
        assert "## Interpretation Notes" in text
        assert "Signal Direction" in text
        assert "SIGN_MISMATCH" in text
        assert "PIT Enforcement" in text


# ---------------------------------------------------------------------------
# Part 1: Eval universe diagnostics
# ---------------------------------------------------------------------------

def _write_rankings_with_components(snap_dir: Path, tickers: List[str],
                                     component_scores: Dict[str, Dict[str, str]] = None):
    """Write rankings.csv with component score columns."""
    snap_dir.mkdir(parents=True, exist_ok=True)
    base_fields = ["ticker", "actionable_rank", "eligible", "tier_dev",
                    "composite_rank", "composite_score", "archetype"]
    comp_fields = ["clinical_score_z_tier", "coinvest_score_z", "catalyst_strength",
                   "clinical_alpha_z", "inst_delta_z", "alpha_cohort_raw"]
    fieldnames = base_fields + comp_fields
    with open(snap_dir / "rankings.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, t in enumerate(tickers):
            row = {
                "ticker": t, "actionable_rank": str(i + 1),
                "eligible": "1", "tier_dev": "A",
                "composite_rank": str(i + 1), "composite_score": "50.0",
                "archetype": "drug_developer",
            }
            if component_scores and t in component_scores:
                row.update(component_scores[t])
            writer.writerow(row)


class TestEvalUniverse:
    def test_universe_counts_tracked(self, tmp_dir):
        """n_universe_eval, n_missing_anchor, n_missing_forward populated."""
        n = 10
        tickers = [f"T{i:02d}" for i in range(n)]
        snap_dir = tmp_dir / "snapshots" / "2025-01-06"
        _write_rankings_csv(snap_dir, tickers)
        _write_metadata(snap_dir, "2025-01-06")

        rows = []
        # Only 7 tickers have anchor prices; only 5 have forward prices
        for d in ["2025-01-06", "2025-01-07"]:
            for i in range(7):
                t = tickers[i]
                if d == "2025-01-06":
                    rows.append({"date": d, "ticker": t, "close": "100"})
                elif i < 5:
                    rows.append({"date": d, "ticker": t, "close": "110"})
        price_csv = tmp_dir / "prices.csv"
        _write_price_csv(price_csv, rows)

        _, results, _ = evaluate(
            snapshot_root=tmp_dir / "snapshots",
            price_csv=price_csv,
            horizons=[1],
            min_price_coverage=0.10,  # low threshold to avoid skip
        )
        evaluated = [r for r in results if not r.skipped]
        assert len(evaluated) == 1
        r = evaluated[0]
        assert r.n_universe_eval == 5
        assert r.n_missing_anchor == 3  # T07, T08, T09 have no prices at all
        assert r.n_missing_forward == 2  # T05, T06 have anchor but no forward

    def test_price_available_mode(self, tmp_dir):
        """universe_mode=price_available filters eval universe."""
        n = 20
        tickers = [f"T{i:02d}" for i in range(n)]
        snap_dir = tmp_dir / "snapshots" / "2025-01-06"
        _write_rankings_csv(snap_dir, tickers)
        _write_metadata(snap_dir, "2025-01-06")

        rows = []
        for d in ["2025-01-06", "2025-01-07"]:
            for i, t in enumerate(tickers):
                if d == "2025-01-06":
                    rows.append({"date": d, "ticker": t, "close": "100"})
                else:
                    # rank 0 → 20%, rank 19 → -18% (monotonic)
                    ret = 0.20 - 0.02 * i
                    rows.append({"date": d, "ticker": t,
                                 "close": str(100 * (1 + ret))})
        price_csv = tmp_dir / "prices.csv"
        _write_price_csv(price_csv, rows)

        _, results_curr, _ = evaluate(
            snapshot_root=tmp_dir / "snapshots",
            price_csv=price_csv,
            horizons=[1],
            universe_mode="current",
        )
        _, results_avail, _ = evaluate(
            snapshot_root=tmp_dir / "snapshots",
            price_csv=price_csv,
            horizons=[1],
            universe_mode="price_available",
        )
        # With all prices available, results should be identical
        r_curr = [r for r in results_curr if not r.skipped][0]
        r_avail = [r for r in results_avail if not r.skipped][0]
        assert r_curr.ic == r_avail.ic

    def test_universe_mode_in_summary(self, tmp_dir):
        """EvalSummary has universe_mode field."""
        summary = EvalSummary(horizons=[5], universe_mode="price_available")
        assert summary.universe_mode == "price_available"
        d = summary.to_dict()
        assert d["universe_mode"] == "price_available"


# ---------------------------------------------------------------------------
# Part 3: Component eval
# ---------------------------------------------------------------------------

class TestComponentEval:
    def test_component_eval_outputs(self, tmp_dir):
        """--component-eval produces CSV and MD files."""
        n = 20
        tickers = [f"T{i:02d}" for i in range(n)]
        comp_scores = {}
        for i, t in enumerate(tickers):
            comp_scores[t] = {
                "clinical_score_z_tier": str(1.0 - 0.1 * i),
                "coinvest_score_z": str(0.5 - 0.05 * i),
                "catalyst_strength": str(0.8 - 0.04 * i),
            }
        snap_dir = tmp_dir / "snapshots" / "2025-01-06"
        _write_rankings_with_components(snap_dir, tickers, comp_scores)
        _write_metadata(snap_dir, "2025-01-06")

        rows = []
        for d in ["2025-01-06", "2025-01-07"]:
            for i, t in enumerate(tickers):
                if d == "2025-01-06":
                    rows.append({"date": d, "ticker": t, "close": "100"})
                else:
                    ret = 0.20 - 0.02 * i
                    rows.append({"date": d, "ticker": t,
                                 "close": str(100 * (1 + ret))})
        price_csv = tmp_dir / "prices.csv"
        _write_price_csv(price_csv, rows)

        out = tmp_dir / "output"
        out.mkdir()

        evaluate(
            snapshot_root=tmp_dir / "snapshots",
            price_csv=price_csv,
            horizons=[1],
            component_eval=True,
            out_dir=out,
        )
        assert (out / "component_eval_by_date.csv").exists()
        assert (out / "component_eval_summary.md").exists()

        # Check CSV has rows
        with open(out / "component_eval_by_date.csv") as f:
            reader = list(csv.DictReader(f))
        assert len(reader) >= 3  # at least 3 components with data
        assert reader[0]["component"] in [
            "clinical_score_z_tier", "coinvest_score_z", "catalyst_strength",
        ]

    def test_component_ic_computed(self, tmp_dir):
        """Component IC is computed for available scores."""
        n = 20
        tickers = [f"T{i:02d}" for i in range(n)]
        comp_scores = {}
        for i, t in enumerate(tickers):
            # Clinical score perfectly correlated with rank
            comp_scores[t] = {"clinical_score_z_tier": str(1.0 - 0.1 * i)}
        snap_dir = tmp_dir / "snapshots" / "2025-01-06"
        _write_rankings_with_components(snap_dir, tickers, comp_scores)
        _write_metadata(snap_dir, "2025-01-06")

        rows = []
        for d in ["2025-01-06", "2025-01-07"]:
            for i, t in enumerate(tickers):
                if d == "2025-01-06":
                    rows.append({"date": d, "ticker": t, "close": "100"})
                else:
                    ret = 0.20 - 0.02 * i  # monotonic with rank
                    rows.append({"date": d, "ticker": t,
                                 "close": str(100 * (1 + ret))})
        price_csv = tmp_dir / "prices.csv"
        _write_price_csv(price_csv, rows)

        out = tmp_dir / "output"
        out.mkdir()

        evaluate(
            snapshot_root=tmp_dir / "snapshots",
            price_csv=price_csv,
            horizons=[1],
            component_eval=True,
            out_dir=out,
        )
        with open(out / "component_eval_by_date.csv") as f:
            reader = list(csv.DictReader(f))
        clin_rows = [r for r in reader if r["component"] == "clinical_score_z_tier"]
        assert len(clin_rows) >= 1
        ic = float(clin_rows[0]["ic"])
        # Clinical score is perfectly correlated → IC should be very high
        assert ic > 0.9

    def test_component_eval_summary_md(self, tmp_dir):
        """Component eval summary MD has expected structure."""
        n = 20
        tickers = [f"T{i:02d}" for i in range(n)]
        comp_scores = {t: {"coinvest_score_z": str(0.5 - 0.05 * i)}
                       for i, t in enumerate(tickers)}
        snap_dir = tmp_dir / "snapshots" / "2025-01-06"
        _write_rankings_with_components(snap_dir, tickers, comp_scores)
        _write_metadata(snap_dir, "2025-01-06")

        rows = []
        for d in ["2025-01-06", "2025-01-07"]:
            for i, t in enumerate(tickers):
                rows.append({"date": d, "ticker": t,
                             "close": str(100 if d == "2025-01-06" else 100 + i)})
        price_csv = tmp_dir / "prices.csv"
        _write_price_csv(price_csv, rows)

        out = tmp_dir / "output"
        out.mkdir()

        evaluate(
            snapshot_root=tmp_dir / "snapshots",
            price_csv=price_csv,
            horizons=[1],
            component_eval=True,
            out_dir=out,
        )
        text = (out / "component_eval_summary.md").read_text()
        assert "Component Attribution Summary" in text
        assert "coinvest_score_z" in text
        assert "Mean IC" in text


# ---------------------------------------------------------------------------
# Part 4: Portfolio construction variants
# ---------------------------------------------------------------------------

class TestPortfolioVariants:
    def _setup_monotonic(self, tmp_dir, n=20):
        """Helper: create monotonic ranking test data."""
        tickers = [f"T{i:02d}" for i in range(n)]
        snap_dir = tmp_dir / "snapshots" / "2025-01-06"
        _write_rankings_csv(snap_dir, tickers)
        _write_metadata(snap_dir, "2025-01-06")

        # Add a second date for turnover testing
        snap_dir2 = tmp_dir / "snapshots" / "2025-01-07"
        # Reverse the ranking on day 2 to force turnover
        _write_rankings_csv(snap_dir2, list(reversed(tickers)))
        _write_metadata(snap_dir2, "2025-01-07")

        rows = []
        for d in ["2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09"]:
            for i, t in enumerate(tickers):
                rows.append({"date": d, "ticker": t,
                             "close": str(100 + i)})
        price_csv = tmp_dir / "prices.csv"
        _write_price_csv(price_csv, rows)
        return price_csv

    def test_top_quantile(self, tmp_dir):
        """--top-quantile 0.25 uses top 25% instead of top-K."""
        n = 20
        tickers = [f"T{i:02d}" for i in range(n)]
        snap_dir = tmp_dir / "snapshots" / "2025-01-06"
        _write_rankings_csv(snap_dir, tickers)
        _write_metadata(snap_dir, "2025-01-06")

        rows = []
        for d in ["2025-01-06", "2025-01-07"]:
            for i, t in enumerate(tickers):
                if d == "2025-01-06":
                    rows.append({"date": d, "ticker": t, "close": "100"})
                else:
                    ret = 0.20 - 0.02 * i
                    rows.append({"date": d, "ticker": t,
                                 "close": str(100 * (1 + ret))})
        price_csv = tmp_dir / "prices.csv"
        _write_price_csv(price_csv, rows)

        _, results, _ = evaluate(
            snapshot_root=tmp_dir / "snapshots",
            price_csv=price_csv,
            horizons=[1],
            top_k=20,  # would normally hold all 20
            top_quantile=0.25,  # should hold 5
        )
        evaluated = [r for r in results if not r.skipped]
        assert len(evaluated) == 1
        assert evaluated[0].n_held == 5  # 25% of 20

    def test_rebalance_buffer_reduces_turnover(self, tmp_dir):
        """Rebalance buffer reduces turnover vs no buffer."""
        price_csv = self._setup_monotonic(tmp_dir)

        _, results_no_buf, _ = evaluate(
            snapshot_root=tmp_dir / "snapshots",
            price_csv=price_csv,
            horizons=[1],
            top_k=5,
            rebalance_buffer_ranks=0,
        )
        _, results_buf, _ = evaluate(
            snapshot_root=tmp_dir / "snapshots",
            price_csv=price_csv,
            horizons=[1],
            top_k=5,
            rebalance_buffer_ranks=10,
        )
        # Second date has reversed ranking → high turnover without buffer
        eval_no_buf = [r for r in results_no_buf if not r.skipped]
        eval_buf = [r for r in results_buf if not r.skipped]
        # With buffer, turnover on the second date should be lower
        if len(eval_no_buf) >= 2 and len(eval_buf) >= 2:
            turn_no = eval_no_buf[1].turnover
            turn_buf = eval_buf[1].turnover
            assert turn_buf <= turn_no

    def test_turnover_cap(self, tmp_dir):
        """Turnover cap limits turnover on subsequent rebalances."""
        price_csv = self._setup_monotonic(tmp_dir)

        _, results_no_cap, _ = evaluate(
            snapshot_root=tmp_dir / "snapshots",
            price_csv=price_csv,
            horizons=[1],
            top_k=5,
            turnover_cap=0.0,
        )
        _, results_cap, _ = evaluate(
            snapshot_root=tmp_dir / "snapshots",
            price_csv=price_csv,
            horizons=[1],
            top_k=5,
            turnover_cap=0.25,
        )
        eval_no_cap = [r for r in results_no_cap if not r.skipped]
        eval_cap = [r for r in results_cap if not r.skipped]
        # Skip first date (turnover from empty); check second date
        if len(eval_no_cap) >= 2 and len(eval_cap) >= 2:
            turn_no = eval_no_cap[1].turnover
            turn_cap = eval_cap[1].turnover
            # Capped turnover should be <= uncapped (or equal if already low)
            assert turn_cap <= turn_no + 0.01
