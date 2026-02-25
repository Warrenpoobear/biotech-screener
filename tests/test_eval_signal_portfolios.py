"""Tests for scripts/eval_signal_portfolios.py — portfolio-realistic backtest."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_signal_portfolios import (
    DEFAULT_COST_BPS,
    SIGNAL_COLUMNS,
    DateRow,
    _build_summary,
    _cumulative,
    _portfolio_gross,
    _safe_float,
    evaluate_signal_portfolio,
    extract_signal,
    select_bottom_k,
    select_top_k,
    write_summary_json,
    write_timeseries_csv,
)
from eval_forward_returns import compute_turnover, net_return


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(ticker: str, eligible: str = "1", **fields) -> Dict[str, str]:
    """Build a synthetic rankings row."""
    r = {"ticker": ticker, "eligible": eligible}
    for k, v in fields.items():
        r[k] = str(v)
    return r


# ---------------------------------------------------------------------------
# _safe_float
# ---------------------------------------------------------------------------

class TestSafeFloat:
    def test_normal(self):
        assert _safe_float("3.14") == pytest.approx(3.14)

    def test_none_returns_nan(self):
        assert math.isnan(_safe_float(None))

    def test_empty_returns_nan(self):
        assert math.isnan(_safe_float(""))

    def test_default_override(self):
        assert _safe_float("", default=0.0) == 0.0

    def test_nan_string(self):
        assert math.isnan(_safe_float("nan"))


# ---------------------------------------------------------------------------
# extract_signal
# ---------------------------------------------------------------------------

class TestExtractSignal:

    def test_catalyst_higher_is_better(self):
        rows = [
            _row("AAA", catalyst_decay_w="0.9"),
            _row("BBB", catalyst_decay_w="0.3"),
            _row("CCC", catalyst_decay_w="0.7"),
        ]
        pairs = extract_signal(rows, "catalyst")
        tickers = [t for t, _ in pairs]
        assert tickers == ["AAA", "CCC", "BBB"]

    def test_actionable_rank_lower_is_better(self):
        rows = [
            _row("AAA", actionable_rank="5"),
            _row("BBB", actionable_rank="1"),
            _row("CCC", actionable_rank="3"),
        ]
        pairs = extract_signal(rows, "actionable_rank")
        tickers = [t for t, _ in pairs]
        assert tickers == ["BBB", "CCC", "AAA"]

    def test_ineligible_excluded(self):
        rows = [
            _row("AAA", eligible="1", catalyst_decay_w="0.9"),
            _row("BBB", eligible="0", catalyst_decay_w="0.99"),
        ]
        pairs = extract_signal(rows, "catalyst")
        assert len(pairs) == 1
        assert pairs[0][0] == "AAA"

    def test_missing_signal_excluded(self):
        rows = [
            _row("AAA", catalyst_decay_w="0.9"),
            _row("BBB"),  # no catalyst_decay_w
        ]
        pairs = extract_signal(rows, "catalyst")
        assert len(pairs) == 1

    def test_empty_string_excluded(self):
        rows = [
            _row("AAA", catalyst_decay_w=""),
        ]
        pairs = extract_signal(rows, "catalyst")
        assert len(pairs) == 0


# ---------------------------------------------------------------------------
# select_top_k / select_bottom_k
# ---------------------------------------------------------------------------

class TestSelectTopBottom:

    def test_top_k_basic(self):
        pairs = [("A", 0.9), ("B", 0.7), ("C", 0.5), ("D", 0.3)]
        assert select_top_k(pairs, 2) == ["A", "B"]

    def test_top_k_exceeds_list(self):
        pairs = [("A", 0.9), ("B", 0.7)]
        assert select_top_k(pairs, 5) == ["A", "B"]

    def test_bottom_k_basic(self):
        pairs = [("A", 0.9), ("B", 0.7), ("C", 0.5), ("D", 0.3)]
        assert select_bottom_k(pairs, 2) == ["C", "D"]

    def test_bottom_k_insufficient(self):
        pairs = [("A", 0.9)]
        assert select_bottom_k(pairs, 3) == []

    def test_ties_preserved(self):
        """With tied signal values, order is stable (insertion order)."""
        pairs = [("A", 0.5), ("B", 0.5), ("C", 0.5), ("D", 0.1)]
        top = select_top_k(pairs, 2)
        assert top == ["A", "B"]
        bottom = select_bottom_k(pairs, 2)
        assert bottom == ["C", "D"]


# ---------------------------------------------------------------------------
# Turnover formula (reuses eval_forward_returns.compute_turnover)
# ---------------------------------------------------------------------------

class TestTurnover:

    def test_identical_sets(self):
        assert compute_turnover(["A", "B", "C"], ["A", "B", "C"]) == 0.0

    def test_complete_replacement(self):
        """Completely different sets → turnover = 1.0."""
        assert compute_turnover(["A", "B"], ["C", "D"]) == pytest.approx(1.0)

    def test_one_swap(self):
        """One swap in set of 3 → turnover = 0.5 * 2 / 3 ≈ 0.3333."""
        turn = compute_turnover(["A", "B", "C"], ["A", "B", "D"])
        assert turn == pytest.approx(1.0 / 3.0)

    def test_empty_prev(self):
        """First rebalance from empty → turnover = 0.5 * 3/3 = 0.5."""
        turn = compute_turnover([], ["A", "B", "C"])
        assert turn == pytest.approx(0.5)

    def test_both_empty(self):
        assert compute_turnover([], []) == 0.0


# ---------------------------------------------------------------------------
# Net return formula
# ---------------------------------------------------------------------------

class TestNetReturn:

    def test_zero_turnover(self):
        """No turnover → net == gross."""
        assert net_return(0.05, 0.0, 30) == pytest.approx(0.05)

    def test_full_turnover(self):
        """100% turnover, 30 bps → cost = 2 * 1.0 * 30/10000 = 0.006."""
        nr = net_return(0.05, 1.0, 30)
        assert nr == pytest.approx(0.05 - 0.006)

    def test_hand_calc_long(self):
        """Hand-calc: gross=3%, turnover=0.3333, cost=30 bps.
        cost_haircut = 2 * 0.3333 * 30/10000 = 0.002
        net = 0.03 - 0.002 = 0.028
        """
        nr = net_return(0.03, 1.0 / 3.0, 30)
        expected = 0.03 - 2.0 * (1.0 / 3.0) * 30 / 10_000
        assert nr == pytest.approx(expected)

    def test_negative_gross(self):
        """Net return with negative gross still applies cost."""
        nr = net_return(-0.02, 0.5, 30)
        assert nr == pytest.approx(-0.02 - 2 * 0.5 * 30 / 10_000)

    def test_long_short_cost(self):
        """L/S net = long_net + short_net; each leg pays its own cost."""
        long_gross = 0.04
        short_gross_raw = 0.02  # short return (positive = shorts went up = loss)
        long_turn = 0.3
        short_turn = 0.4
        cost = 30

        long_net = net_return(long_gross, long_turn, cost)
        short_net = net_return(-short_gross_raw, short_turn, cost)
        ls_net = long_net + short_net

        expected_long_cost = 2 * long_turn * cost / 10_000
        expected_short_cost = 2 * short_turn * cost / 10_000
        expected = (long_gross - short_gross_raw) - expected_long_cost - expected_short_cost
        assert ls_net == pytest.approx(expected, abs=1e-10)


# ---------------------------------------------------------------------------
# _portfolio_gross
# ---------------------------------------------------------------------------

class TestPortfolioGross:

    def test_equal_weight(self):
        fwd = {"A": 0.10, "B": 0.04, "C": -0.02}
        ret, n = _portfolio_gross(["A", "B", "C"], fwd)
        assert n == 3
        assert ret == pytest.approx(0.04)

    def test_missing_tickers_excluded(self):
        fwd = {"A": 0.10, "C": -0.02}
        ret, n = _portfolio_gross(["A", "B", "C"], fwd)
        assert n == 2
        assert ret == pytest.approx(0.04)

    def test_empty_returns_none(self):
        ret, n = _portfolio_gross(["A", "B"], {})
        assert ret is None
        assert n == 0


# ---------------------------------------------------------------------------
# _cumulative
# ---------------------------------------------------------------------------

class TestCumulative:

    def test_basic(self):
        # (1+0.10)*(1+0.05) - 1 = 0.155
        assert _cumulative([0.10, 0.05]) == pytest.approx(0.155)

    def test_empty(self):
        assert _cumulative([]) is None

    def test_single(self):
        assert _cumulative([0.03]) == pytest.approx(0.03)

    def test_negative(self):
        # (1-0.10)*(1+0.20) - 1 = 0.08
        assert _cumulative([-0.10, 0.20]) == pytest.approx(0.08)


# ---------------------------------------------------------------------------
# _build_summary
# ---------------------------------------------------------------------------

class TestBuildSummary:

    def test_long_only_summary(self):
        s = _build_summary(
            signal="catalyst",
            horizon=20,
            top_k=5,
            cost_bps=30,
            anchor_mode="next_trading_day",
            long_short=False,
            n_dates=10,
            n_evaluated=8,
            n_skipped=2,
            long_gross_list=[0.03, 0.02],
            long_net_list=[0.028, 0.018],
            long_turn_list=[0.3, 0.2],
            short_gross_list=[],
            short_net_list=[],
            short_turn_list=[],
            ls_gross_list=[],
            ls_net_list=[],
            date_rows=[],
            regime_map={},
        )
        assert s["signal"] == "catalyst"
        assert s["n_evaluated"] == 8
        assert s["long"]["mean_gross"] == pytest.approx(0.025, abs=1e-4)
        assert "short" not in s
        assert "long_short" not in s

    def test_long_short_summary(self):
        s = _build_summary(
            signal="clinical",
            horizon=63,
            top_k=20,
            cost_bps=30,
            anchor_mode="next_trading_day",
            long_short=True,
            n_dates=5,
            n_evaluated=5,
            n_skipped=0,
            long_gross_list=[0.05],
            long_net_list=[0.048],
            long_turn_list=[0.5],
            short_gross_list=[-0.02],
            short_net_list=[-0.018],
            short_turn_list=[0.4],
            ls_gross_list=[0.07],
            ls_net_list=[0.066],
            date_rows=[],
            regime_map={},
        )
        assert "short" in s
        assert "long_short" in s
        assert s["long_short"]["mean_gross"] == pytest.approx(0.07)

    def test_regime_slices(self):
        rows = [
            DateRow(date="2026-01-01", trade_date="2026-01-02", regime="BULL",
                    long_gross=0.05, long_net=0.048, long_turnover=0.3),
            DateRow(date="2026-01-15", trade_date="2026-01-16", regime="BEAR",
                    long_gross=-0.02, long_net=-0.022, long_turnover=0.4),
            DateRow(date="2026-02-01", trade_date="2026-02-03", regime="BULL",
                    long_gross=0.03, long_net=0.028, long_turnover=0.2),
        ]
        regime_map = {
            "2026-01-01": "BULL",
            "2026-01-15": "BEAR",
            "2026-02-01": "BULL",
        }
        s = _build_summary(
            signal="catalyst", horizon=20, top_k=5, cost_bps=30,
            anchor_mode="next_trading_day", long_short=False,
            n_dates=3, n_evaluated=3, n_skipped=0,
            long_gross_list=[0.05, -0.02, 0.03],
            long_net_list=[0.048, -0.022, 0.028],
            long_turn_list=[0.3, 0.4, 0.2],
            short_gross_list=[], short_net_list=[], short_turn_list=[],
            ls_gross_list=[], ls_net_list=[],
            date_rows=rows, regime_map=regime_map,
        )
        assert "regime_slices" in s
        assert "BULL" in s["regime_slices"]
        assert "BEAR" in s["regime_slices"]
        assert s["regime_slices"]["BULL"]["n_dates"] == 2
        assert s["regime_slices"]["BEAR"]["n_dates"] == 1


# ---------------------------------------------------------------------------
# Integration: evaluate_signal_portfolio with synthetic data
# ---------------------------------------------------------------------------

def _write_price_csv(path: Path, data: Dict[str, Dict[str, float]]) -> None:
    """Write a minimal price_history.csv."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "ticker", "close"])
        for ticker, prices in data.items():
            for dt, close in sorted(prices.items()):
                w.writerow([dt, ticker, close])


import csv


def _write_snapshot(snap_dir: Path, rows: List[Dict[str, str]]) -> None:
    """Write a rankings.csv with metadata.json for a snapshot date."""
    snap_dir.mkdir(parents=True, exist_ok=True)

    # Compute fieldnames from all rows
    fieldnames = list(dict.fromkeys(
        k for r in rows for k in r.keys()
    ))

    with open(snap_dir / "rankings.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Write metadata.json
    meta = {"as_of_date": snap_dir.name}
    with open(snap_dir / "metadata.json", "w") as f:
        json.dump(meta, f)


class TestIntegrationLongOnly:

    def test_basic_long_portfolio(self, tmp_path):
        """Two dates, 4 tickers, top-2 portfolio."""
        snap_root = tmp_path / "snapshots"

        # Date 1
        rows1 = [
            _row("AAA", catalyst_decay_w="0.9", actionable_rank="1"),
            _row("BBB", catalyst_decay_w="0.7", actionable_rank="2"),
            _row("CCC", catalyst_decay_w="0.3", actionable_rank="3"),
            _row("DDD", catalyst_decay_w="0.1", actionable_rank="4"),
        ]
        _write_snapshot(snap_root / "2026-01-06", rows1)

        # Date 2
        rows2 = [
            _row("AAA", catalyst_decay_w="0.5", actionable_rank="3"),
            _row("BBB", catalyst_decay_w="0.9", actionable_rank="1"),
            _row("CCC", catalyst_decay_w="0.8", actionable_rank="2"),
            _row("DDD", catalyst_decay_w="0.2", actionable_rank="4"),
        ]
        _write_snapshot(snap_root / "2026-01-13", rows2)

        # Prices: 5 weeks of trading days for all 4 tickers
        prices: Dict[str, Dict[str, float]] = {}
        # Simple: all start at 100, AAA goes to 110, BBB 105, CCC 95, DDD 90
        trade_dates = [
            "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-10",
            "2026-01-13", "2026-01-14", "2026-01-15", "2026-01-16",
            "2026-01-20", "2026-01-21", "2026-01-22", "2026-01-23",
            "2026-01-27", "2026-01-28", "2026-01-29", "2026-01-30",
            "2026-02-03", "2026-02-04", "2026-02-05", "2026-02-06",
            "2026-02-10", "2026-02-11", "2026-02-12", "2026-02-13",
        ]
        for t in ["AAA", "BBB", "CCC", "DDD"]:
            prices[t] = {}
            for i, d in enumerate(trade_dates):
                if t == "AAA":
                    prices[t][d] = 100 + i * 0.5
                elif t == "BBB":
                    prices[t][d] = 100 + i * 0.25
                elif t == "CCC":
                    prices[t][d] = 100 - i * 0.3
                else:
                    prices[t][d] = 100 - i * 0.5

        price_csv = tmp_path / "prices.csv"
        _write_price_csv(price_csv, prices)

        date_rows, summary = evaluate_signal_portfolio(
            snapshot_root=snap_root,
            price_csv=price_csv,
            signal="catalyst",
            horizon=5,
            top_k=2,
            cost_bps=30,
            anchor_mode="next_trading_day",
            min_coverage=0.50,
        )

        # Both dates should be evaluated
        evaluated = [r for r in date_rows if not r.skipped]
        assert len(evaluated) == 2
        assert summary["n_evaluated"] == 2
        assert summary["long"]["mean_gross"] is not None
        assert summary["long"]["mean_net"] is not None
        assert summary["long"]["mean_turnover"] is not None

        # First date: turnover from empty → 0.5
        assert evaluated[0].long_turnover == pytest.approx(0.5)

        # Net < gross (cost haircut)
        for r in evaluated:
            if r.long_gross is not None and r.long_net is not None:
                assert r.long_net <= r.long_gross

    def test_skip_low_coverage(self, tmp_path):
        """Skip dates where price coverage is below threshold."""
        snap_root = tmp_path / "snapshots"
        rows = [
            _row("AAA", catalyst_decay_w="0.9"),
            _row("BBB", catalyst_decay_w="0.7"),
            _row("CCC", catalyst_decay_w="0.5"),
            _row("DDD", catalyst_decay_w="0.3"),
        ]
        _write_snapshot(snap_root / "2026-01-06", rows)

        # Only 1 ticker has prices → coverage = 0.25 < 0.50
        prices = {"AAA": {"2026-01-07": 100, "2026-01-14": 105}}
        price_csv = tmp_path / "prices.csv"
        _write_price_csv(price_csv, prices)

        date_rows, summary = evaluate_signal_portfolio(
            snapshot_root=snap_root,
            price_csv=price_csv,
            signal="catalyst",
            horizon=5,
            top_k=2,
            cost_bps=30,
            anchor_mode="next_trading_day",
            min_coverage=0.50,
        )

        assert summary["n_skipped"] == 1
        assert date_rows[0].skipped
        assert "LOW_COVERAGE" in date_rows[0].skip_reason

    def test_skip_insufficient_signal(self, tmp_path):
        """Skip if fewer tickers have signal than top_k."""
        snap_root = tmp_path / "snapshots"
        rows = [
            _row("AAA", catalyst_decay_w="0.9"),
            # BBB has no catalyst signal
            _row("BBB"),
        ]
        _write_snapshot(snap_root / "2026-01-06", rows)

        prices = {
            "AAA": {"2026-01-07": 100, "2026-01-14": 105},
            "BBB": {"2026-01-07": 100, "2026-01-14": 98},
        }
        price_csv = tmp_path / "prices.csv"
        _write_price_csv(price_csv, prices)

        date_rows, summary = evaluate_signal_portfolio(
            snapshot_root=snap_root,
            price_csv=price_csv,
            signal="catalyst",
            horizon=5,
            top_k=2,
            cost_bps=30,
            anchor_mode="next_trading_day",
        )

        assert summary["n_skipped"] == 1
        assert "INSUFFICIENT_SIGNAL" in date_rows[0].skip_reason


class TestIntegrationLongShort:

    def test_long_short_portfolio(self, tmp_path):
        """L/S portfolio: long top-2, short bottom-2."""
        snap_root = tmp_path / "snapshots"
        rows = [
            _row("AAA", catalyst_decay_w="0.9"),
            _row("BBB", catalyst_decay_w="0.7"),
            _row("CCC", catalyst_decay_w="0.3"),
            _row("DDD", catalyst_decay_w="0.1"),
        ]
        _write_snapshot(snap_root / "2026-01-06", rows)

        # AAA/BBB go up, CCC/DDD go down → L/S should profit
        trade_dates = [
            "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-10",
            "2026-01-13", "2026-01-14",
        ]
        prices: Dict[str, Dict[str, float]] = {
            "AAA": {d: 100 + i * 2 for i, d in enumerate(trade_dates)},
            "BBB": {d: 100 + i * 1 for i, d in enumerate(trade_dates)},
            "CCC": {d: 100 - i * 1 for i, d in enumerate(trade_dates)},
            "DDD": {d: 100 - i * 2 for i, d in enumerate(trade_dates)},
        }
        price_csv = tmp_path / "prices.csv"
        _write_price_csv(price_csv, prices)

        date_rows, summary = evaluate_signal_portfolio(
            snapshot_root=snap_root,
            price_csv=price_csv,
            signal="catalyst",
            horizon=5,
            top_k=2,
            cost_bps=30,
            anchor_mode="next_trading_day",
            long_short=True,
        )

        evaluated = [r for r in date_rows if not r.skipped]
        assert len(evaluated) == 1
        r = evaluated[0]

        # Long leg gross > 0 (AAA, BBB go up)
        assert r.long_gross is not None
        assert r.long_gross > 0

        # Short gross: CCC/DDD go down → their raw return is negative
        # but short_gross in DateRow is the raw portfolio return for the short tickers
        # (before negation). In summary, it's negated.
        assert r.short_gross is not None

        # L/S gross = long - short_raw > 0
        assert r.ls_gross is not None
        assert r.ls_gross > 0

        # Summary has long_short block
        assert "long_short" in summary
        assert summary["long_short"]["mean_gross"] is not None


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

class TestOutputWriters:

    def test_timeseries_csv(self, tmp_path):
        rows = [
            DateRow(date="2026-01-06", trade_date="2026-01-07",
                    long_gross=0.03, long_net=0.028, long_turnover=0.5,
                    long_n_held=5, n_signal=20, coverage=0.90),
        ]
        path = write_timeseries_csv(rows, tmp_path)
        assert path.exists()
        content = path.read_text()
        assert "2026-01-06" in content
        assert "long_gross" in content

    def test_summary_json(self, tmp_path):
        summary = {"signal": "catalyst", "long": {"mean_gross": 0.03}}
        path = write_summary_json(summary, tmp_path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["signal"] == "catalyst"


# ---------------------------------------------------------------------------
# Signal column registry
# ---------------------------------------------------------------------------

class TestSignalColumns:

    def test_all_signals_have_entries(self):
        expected = {"catalyst", "clinical", "alpha", "composite",
                    "actionable_rank", "alpha_incr"}
        assert set(SIGNAL_COLUMNS.keys()) == expected

    def test_unknown_signal_raises(self):
        with pytest.raises(ValueError, match="Unknown signal"):
            evaluate_signal_portfolio(
                snapshot_root=Path("/nonexistent"),
                price_csv=Path("/nonexistent"),
                signal="bogus",
            )
