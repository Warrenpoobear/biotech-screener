"""Tests for Tier-1 manager scoring."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.manager_scoring import (
    build_manager_leaderboard,
    compute_manager_returns,
)


def _make_panel(dates, tickers, returns_map):
    """
    Build a synthetic panel.

    returns_map: {ticker: {date: return}}
    """
    rows = []
    for dt in dates:
        for ticker in tickers:
            ret = returns_map.get(ticker, {}).get(dt, np.nan)
            rows.append({
                "rebalance_date": dt,
                "ticker": ticker,
                "fwd_21d": ret,
            })
    return pd.DataFrame(rows)


def _make_weekly_holdings(manager_tickers, dates):
    """
    Build a synthetic weekly holdings DataFrame.

    manager_tickers: {(cik, name): [ticker_list]}
    """
    rows = []
    for dt in dates:
        for (cik, name), tickers in manager_tickers.items():
            for t in sorted(tickers):
                rows.append({
                    "rebalance_date": dt,
                    "manager_cik": cik,
                    "manager_name": name,
                    "ticker": t,
                    "shares": 1000,
                    "value_kusd": 100,
                })
    return pd.DataFrame(rows)


class TestComputeManagerReturns:
    """Test manager return computation."""

    def test_equal_weight_returns(self):
        """Manager holds two tickers, equal weight."""
        dates = ["2025-01-03", "2025-01-10"]
        tickers = ["A", "B", "C"]
        returns_map = {
            "A": {"2025-01-03": 0.10, "2025-01-10": 0.05},
            "B": {"2025-01-03": 0.20, "2025-01-10": -0.05},
            "C": {"2025-01-03": -0.10, "2025-01-10": 0.00},
        }
        panel = _make_panel(dates, tickers, returns_map)

        holdings = _make_weekly_holdings(
            {("CIK1", "ManagerA"): ["A", "B"]},
            dates,
        )

        result = compute_manager_returns(holdings, panel, "fwd_21d")
        assert len(result) == 2

        # Week 1: (0.10 + 0.20) / 2 = 0.15
        w1 = result[result["rebalance_date"] == "2025-01-03"].iloc[0]
        assert abs(w1["portfolio_return"] - 0.15) < 1e-9

        # Week 2: (0.05 + -0.05) / 2 = 0.0
        w2 = result[result["rebalance_date"] == "2025-01-10"].iloc[0]
        assert abs(w2["portfolio_return"]) < 1e-9

    def test_universe_mean(self):
        """Universe mean is computed correctly."""
        dates = ["2025-01-03"]
        tickers = ["A", "B", "C"]
        returns_map = {
            "A": {"2025-01-03": 0.10},
            "B": {"2025-01-03": 0.20},
            "C": {"2025-01-03": 0.30},
        }
        panel = _make_panel(dates, tickers, returns_map)
        holdings = _make_weekly_holdings(
            {("CIK1", "M1"): ["A"]}, dates,
        )
        result = compute_manager_returns(holdings, panel, "fwd_21d")
        # Universe mean: (0.10 + 0.20 + 0.30) / 3 = 0.20
        assert abs(result.iloc[0]["universe_mean_return"] - 0.20) < 1e-9


class TestManagerLeaderboard:
    """Test leaderboard building."""

    def test_winner_ranks_above_loser(self):
        """Manager A (winners) ranks above Manager B (losers)."""
        dates = [f"2025-01-{d:02d}" for d in range(1, 11)]
        tickers = ["W1", "W2", "L1", "L2", "N1"]

        returns_map = {}
        for t in ["W1", "W2"]:
            returns_map[t] = {d: 0.05 for d in dates}  # Winners: +5% weekly
        for t in ["L1", "L2"]:
            returns_map[t] = {d: -0.05 for d in dates}  # Losers: -5% weekly
        returns_map["N1"] = {d: 0.0 for d in dates}

        panel = _make_panel(dates, tickers, returns_map)
        holdings = _make_weekly_holdings(
            {
                ("CIK_A", "Winner Manager"): ["W1", "W2"],
                ("CIK_B", "Loser Manager"): ["L1", "L2"],
            },
            dates,
        )

        mgr_returns = compute_manager_returns(holdings, panel, "fwd_21d")
        leaderboard = build_manager_leaderboard(mgr_returns, min_weeks=3)

        assert len(leaderboard) == 2
        assert leaderboard.iloc[0]["manager_name"] == "Winner Manager"
        assert leaderboard.iloc[1]["manager_name"] == "Loser Manager"
        assert leaderboard.iloc[0]["avg_return"] > 0
        assert leaderboard.iloc[1]["avg_return"] < 0

    def test_min_weeks_filter(self):
        """Managers with too few weeks are excluded."""
        dates = ["2025-01-03", "2025-01-10"]
        tickers = ["A"]
        returns_map = {"A": {d: 0.01 for d in dates}}
        panel = _make_panel(dates, tickers, returns_map)
        holdings = _make_weekly_holdings(
            {("CIK1", "ShortTrack"): ["A"]}, dates,
        )
        mgr_returns = compute_manager_returns(holdings, panel, "fwd_21d")
        leaderboard = build_manager_leaderboard(mgr_returns, min_weeks=5)
        assert len(leaderboard) == 0  # Only 2 weeks < 5

    def test_hit_rate(self):
        """Hit rate is fraction of positive-return weeks."""
        # 8 positive weeks, 2 negative
        dates = [f"2025-{i+1:04d}" for i in range(10)]
        tickers = ["A"]
        returns_map = {"A": {}}
        for i, d in enumerate(dates):
            returns_map["A"][d] = 0.02 if i < 8 else -0.01

        panel = _make_panel(dates, tickers, returns_map)
        holdings = _make_weekly_holdings(
            {("CIK1", "M1"): ["A"]}, dates,
        )
        mgr_returns = compute_manager_returns(holdings, panel, "fwd_21d")
        leaderboard = build_manager_leaderboard(mgr_returns, min_weeks=3)
        assert abs(leaderboard.iloc[0]["hit_rate"] - 0.8) < 1e-6

    def test_deterministic(self):
        """Same inputs → same output."""
        dates = [f"2025-{i+1:04d}" for i in range(10)]
        tickers = ["A", "B"]
        returns_map = {
            "A": {d: 0.01 for d in dates},
            "B": {d: -0.01 for d in dates},
        }
        panel = _make_panel(dates, tickers, returns_map)
        holdings = _make_weekly_holdings(
            {("CIK1", "M1"): ["A", "B"]}, dates,
        )
        r1 = compute_manager_returns(holdings, panel, "fwd_21d")
        r2 = compute_manager_returns(holdings, panel, "fwd_21d")
        pd.testing.assert_frame_equal(r1, r2)

    def test_empty_holdings(self):
        """Empty holdings returns empty leaderboard."""
        empty = pd.DataFrame(columns=[
            "rebalance_date", "manager_cik", "manager_name",
            "portfolio_return", "n_holdings", "n_with_returns",
            "universe_mean_return",
        ])
        lb = build_manager_leaderboard(empty)
        assert lb.empty
