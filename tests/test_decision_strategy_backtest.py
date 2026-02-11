#!/usr/bin/env python3
"""
Unit tests for run_decision_strategy_backtest.py

Tests portfolio construction, baseline building, evaluation, turnover,
and aggregation using synthetic data (no archive I/O).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from decision_engine import DecisionRuleset
from run_decision_ruleset_sweep import ArchiveData
from run_decision_strategy_backtest import (
    MultiHorizonReturns,
    SnapshotResult,
    _percentile,
    aggregate_results,
    build_composite_baseline,
    build_strategy_portfolio,
    build_universe_baseline,
    compute_portfolio_turnover,
    evaluate_portfolio,
)


# =============================================================================
# HELPERS: synthetic data builders
# =============================================================================

def _make_rec(
    severity: str = "",
    drawdown: Optional[float] = None,
    optionality_score: float = 0.70,
    catalyst_days: int = 30,
    catalyst_window: bool = True,
    tier1_count: int = 2,
) -> Dict[str, Any]:
    """Build a minimal rec dict for testing."""
    rec: Dict[str, Any] = {
        "severity": severity,
        "confidence_overall": 0.5,
        "fundamental_red_flag": False,
        "catalyst_decay": {
            "days_to_catalyst": catalyst_days,
            "in_optimal_window": catalyst_window,
        },
        "smart_money_signal": {},
        "coinvest": {"tier1_count": tier1_count},
        "defensive_features": {},
        "score_breakdown": {},
        "momentum_signal": {},
    }
    if drawdown is not None:
        rec["defensive_features"]["drawdown"] = drawdown
    return rec


def _make_archive_data(
    tickers_data: Dict[str, Dict[str, Any]],
    date_str: str = "2024-06-30",
) -> ArchiveData:
    """Build a minimal ArchiveData for testing.

    tickers_data: {ticker: {"rec": rec, "archetype": str, "opt": float|None,
                             "composite_rank": int}}
    """
    tickers = list(tickers_data.keys())
    recs = {t: d["rec"] for t, d in tickers_data.items()}
    archetypes = {t: d.get("archetype", "drug_developer") for t, d in tickers_data.items()}
    optionalities = {t: d.get("opt", 0.70) for t, d in tickers_data.items()}
    rows = []
    for t in tickers:
        row = {
            "ticker": t,
            "archetype": archetypes[t],
            "clinical_optionality_pct_dev": str(optionalities[t]) if optionalities[t] is not None else "",
        }
        rank = tickers_data[t].get("composite_rank")
        if rank is not None:
            row["composite_rank"] = str(rank)
        rows.append(row)

    return ArchiveData(
        date_str=date_str,
        rows=rows,
        recs=recs,
        archetypes=archetypes,
        optionalities=optionalities,
        tickers=tickers,
    )


def _make_multi_horizon_returns(
    raw_20: Dict[str, float],
    raw_60: Dict[str, float],
    regime: str = "RISK_ON",
) -> MultiHorizonReturns:
    """Build MultiHorizonReturns with resid=raw (beta=0)."""
    return MultiHorizonReturns(
        raw={20: raw_20, 60: raw_60},
        resid={20: dict(raw_20), 60: dict(raw_60)},
        xbi={20: 0.0, 60: 0.0},
        betas={t: 0.0 for t in set(list(raw_20) + list(raw_60))},
        regime=regime,
    )


# =============================================================================
# TEST: build_strategy_portfolio
# =============================================================================

class TestBuildStrategyPortfolio:
    def test_selects_eligible_dev_ab_only(self):
        """Only eligible drug_developer A/B tickers should be selected."""
        data = _make_archive_data({
            # A-tier: high opt + catalyst near
            "AATK": {"rec": _make_rec(catalyst_days=30), "opt": 0.80, "composite_rank": 1},
            # B-tier: high opt + catalyst far
            "BBTK": {"rec": _make_rec(catalyst_days=200), "opt": 0.70, "composite_rank": 2},
            # C-tier: low opt
            "CCTK": {"rec": _make_rec(catalyst_days=30), "opt": 0.20, "composite_rank": 3},
            # D-tier: ineligible (drawdown)
            "DDTK": {"rec": _make_rec(drawdown=-0.50), "opt": 0.80, "composite_rank": 4},
            # Commercial: excluded
            "CMTK": {"rec": _make_rec(), "archetype": "commercial_biotech",
                      "opt": 0.80, "composite_rank": 5},
        })

        result = build_strategy_portfolio(data, DecisionRuleset(), ["A", "B"], top_k=20)
        tickers = [p["ticker"] for p in result]

        assert "AATK" in tickers
        assert "BBTK" in tickers
        assert "CCTK" not in tickers  # C tier
        assert "DDTK" not in tickers  # ineligible
        assert "CMTK" not in tickers  # commercial

    def test_top_k_cap(self):
        """Should respect top-k limit."""
        tickers_data = {}
        for i in range(10):
            t = f"TK{i:02d}"
            tickers_data[t] = {
                "rec": _make_rec(catalyst_days=30),
                "opt": 0.80,
                "composite_rank": i + 1,
            }
        data = _make_archive_data(tickers_data)

        result = build_strategy_portfolio(data, DecisionRuleset(), ["A", "B"], top_k=5)
        assert len(result) == 5

    def test_weights_sum_to_100(self):
        """Weights should sum to approximately 100%."""
        tickers_data = {}
        for i in range(5):
            t = f"TK{i:02d}"
            tickers_data[t] = {
                "rec": _make_rec(catalyst_days=30),
                "opt": 0.80,
                "composite_rank": i + 1,
            }
        data = _make_archive_data(tickers_data)

        result = build_strategy_portfolio(data, DecisionRuleset(), ["A", "B"], top_k=20)
        total_weight = sum(p["weight_pct"] for p in result)
        assert abs(total_weight - 100.0) < 1.0  # within rounding

    def test_actionable_rank_assigned(self):
        """Each position should have an actionable_rank starting from 1."""
        data = _make_archive_data({
            "AA": {"rec": _make_rec(catalyst_days=30), "opt": 0.80, "composite_rank": 1},
            "BB": {"rec": _make_rec(catalyst_days=60), "opt": 0.70, "composite_rank": 2},
        })
        result = build_strategy_portfolio(data, DecisionRuleset(), ["A", "B"], top_k=20)
        ranks = [p["actionable_rank"] for p in result]
        assert ranks == list(range(1, len(result) + 1))

    def test_deterministic_ordering(self):
        """Same input should always produce same order."""
        data = _make_archive_data({
            "XX": {"rec": _make_rec(catalyst_days=30), "opt": 0.75, "composite_rank": 2},
            "YY": {"rec": _make_rec(catalyst_days=30), "opt": 0.80, "composite_rank": 1},
        })
        r1 = build_strategy_portfolio(data, DecisionRuleset(), ["A", "B"], top_k=20)
        r2 = build_strategy_portfolio(data, DecisionRuleset(), ["A", "B"], top_k=20)
        assert [p["ticker"] for p in r1] == [p["ticker"] for p in r2]


# =============================================================================
# TEST: build_composite_baseline
# =============================================================================

class TestBuildCompositeBaseline:
    def test_picks_top_k_dev_by_rank(self):
        """Should pick top-K drug developers by composite_rank."""
        data = _make_archive_data({
            "R1": {"rec": _make_rec(), "opt": 0.5, "composite_rank": 1},
            "R2": {"rec": _make_rec(), "opt": 0.5, "composite_rank": 2},
            "R3": {"rec": _make_rec(), "opt": 0.5, "composite_rank": 3},
            "R4": {"rec": _make_rec(), "opt": 0.5, "composite_rank": 10},
            "CM": {"rec": _make_rec(), "opt": 0.5, "composite_rank": 1,
                   "archetype": "commercial_biotech"},
        })

        result = build_composite_baseline(data, top_k=3)
        tickers = [p["ticker"] for p in result]
        assert tickers == ["R1", "R2", "R3"]
        assert "CM" not in tickers

    def test_equal_weight(self):
        """All positions should have equal weight."""
        data = _make_archive_data({
            "A": {"rec": _make_rec(), "opt": 0.5, "composite_rank": 1},
            "B": {"rec": _make_rec(), "opt": 0.5, "composite_rank": 2},
        })
        result = build_composite_baseline(data, top_k=5)
        assert len(result) == 2
        assert result[0]["weight_pct"] == result[1]["weight_pct"]
        assert abs(result[0]["weight_pct"] - 50.0) < 0.1


# =============================================================================
# TEST: evaluate_portfolio
# =============================================================================

class TestEvaluatePortfolio:
    def test_weighted_return(self):
        """Weighted return should match manual calculation."""
        positions = [
            {"ticker": "A", "weight_pct": 60.0},
            {"ticker": "B", "weight_pct": 40.0},
        ]
        returns = _make_multi_horizon_returns(
            raw_20={"A": 0.10, "B": -0.05},
            raw_60={"A": 0.20, "B": 0.10},
        )

        result = evaluate_portfolio(positions, returns, [20, 60])

        # 20d: (60/100)*10 + (40/100)*(-5) = 6 - 2 = 4%
        assert abs(result["weighted_ret_20d_pct"] - 4.0) < 0.01
        # 60d: (60/100)*20 + (40/100)*10 = 12 + 4 = 16%
        assert abs(result["weighted_ret_60d_pct"] - 16.0) < 0.01

    def test_hit_rate(self):
        """Hit rate should be fraction of positive returns."""
        positions = [
            {"ticker": "A", "weight_pct": 50.0},
            {"ticker": "B", "weight_pct": 50.0},
        ]
        returns = _make_multi_horizon_returns(
            raw_20={"A": 0.10, "B": -0.05},
            raw_60={"A": 0.20, "B": 0.10},
        )
        result = evaluate_portfolio(positions, returns, [20, 60])
        assert result["hit_rate_20d_pct"] == 50.0
        assert result["hit_rate_60d_pct"] == 100.0

    def test_left_tail(self):
        """Should compute P5, P10, max_loss."""
        positions = [
            {"ticker": "A", "weight_pct": 50.0},
            {"ticker": "B", "weight_pct": 50.0},
        ]
        returns = _make_multi_horizon_returns(
            raw_20={"A": -0.20, "B": 0.10},
            raw_60={"A": -0.20, "B": 0.10},
        )
        result = evaluate_portfolio(positions, returns, [20, 60])
        assert result["max_loss_20d_pct"] == -20.0
        assert result["p5_20d_pct"] is not None

    def test_missing_returns_renormalize(self):
        """If a ticker has no return data, weights should be renormalized."""
        positions = [
            {"ticker": "A", "weight_pct": 60.0},
            {"ticker": "B", "weight_pct": 40.0},
        ]
        # B has no return data
        returns = _make_multi_horizon_returns(
            raw_20={"A": 0.10},
            raw_60={"A": 0.20},
        )
        result = evaluate_portfolio(positions, returns, [20, 60])
        # Weighted return should be 100% A's return
        assert abs(result["weighted_ret_20d_pct"] - 10.0) < 0.01
        assert result["coverage_20d"] == 1


# =============================================================================
# TEST: compute_portfolio_turnover
# =============================================================================

class TestPortfolioTurnover:
    def test_identical_portfolios(self):
        """Turnover should be 0 for identical portfolios."""
        p = [{"ticker": "A", "weight_pct": 50.0}, {"ticker": "B", "weight_pct": 50.0}]
        result = compute_portfolio_turnover(p, p)
        assert result["position_turnover"] == 0.0
        assert result["weight_turnover_pct"] == 0.0

    def test_disjoint_portfolios(self):
        """Turnover should be 1.0 for completely different portfolios."""
        prev = [{"ticker": "A", "weight_pct": 50.0}, {"ticker": "B", "weight_pct": 50.0}]
        curr = [{"ticker": "C", "weight_pct": 50.0}, {"ticker": "D", "weight_pct": 50.0}]
        result = compute_portfolio_turnover(prev, curr)
        # sym_diff={A,B,C,D}=4, union={A,B,C,D}=4, turnover=4/4=1.0
        assert result["position_turnover"] == 1.0
        # weight_turnover: sum(|50-0|*4) / 2 = 200/2 = 100
        assert result["weight_turnover_pct"] == 100.0

    def test_partial_overlap(self):
        """Partial overlap should give intermediate turnover."""
        prev = [{"ticker": "A", "weight_pct": 50.0}, {"ticker": "B", "weight_pct": 50.0}]
        curr = [{"ticker": "A", "weight_pct": 50.0}, {"ticker": "C", "weight_pct": 50.0}]
        result = compute_portfolio_turnover(prev, curr)
        # sym_diff={B,C}=2, union={A,B,C}=3, turnover=2/3
        assert abs(result["position_turnover"] - 2/3) < 0.001
        # weight: |50-50|(A) + |0-50|(B) + |50-0|(C) = 0+50+50 = 100, /2 = 50
        assert result["weight_turnover_pct"] == 50.0

    def test_weight_change_only(self):
        """Same tickers but different weights should have 0 position turnover."""
        prev = [{"ticker": "A", "weight_pct": 70.0}, {"ticker": "B", "weight_pct": 30.0}]
        curr = [{"ticker": "A", "weight_pct": 50.0}, {"ticker": "B", "weight_pct": 50.0}]
        result = compute_portfolio_turnover(prev, curr)
        assert result["position_turnover"] == 0.0
        # weight: |50-70| + |50-30| = 20+20 = 40, /2 = 20
        assert result["weight_turnover_pct"] == 20.0


# =============================================================================
# TEST: aggregate_results
# =============================================================================

class TestAggregateResults:
    def _make_snapshot(
        self,
        date_str: str,
        strat_resid_60: float,
        baseline_resid_60: float,
        regime: str = "RISK_ON",
    ) -> SnapshotResult:
        """Build a minimal SnapshotResult for aggregation tests."""
        return SnapshotResult(
            date_str=date_str,
            regime=regime,
            strategy_eval={
                "n_positions": 10,
                "weighted_ret_20d_pct": strat_resid_60 * 0.5,
                "weighted_ret_60d_pct": strat_resid_60,
                "weighted_resid_20d_pct": strat_resid_60 * 0.5,
                "weighted_resid_60d_pct": strat_resid_60,
                "ew_ret_20d_pct": strat_resid_60 * 0.5,
                "ew_ret_60d_pct": strat_resid_60,
                "ew_resid_20d_pct": strat_resid_60 * 0.5,
                "ew_resid_60d_pct": strat_resid_60,
                "hit_rate_20d_pct": 60.0,
                "hit_rate_60d_pct": 65.0,
            },
            baseline_topk_eval={
                "n_positions": 10,
                "weighted_ret_20d_pct": baseline_resid_60 * 0.5,
                "weighted_ret_60d_pct": baseline_resid_60,
                "weighted_resid_20d_pct": baseline_resid_60 * 0.5,
                "weighted_resid_60d_pct": baseline_resid_60,
                "ew_ret_20d_pct": baseline_resid_60 * 0.5,
                "ew_ret_60d_pct": baseline_resid_60,
                "ew_resid_20d_pct": baseline_resid_60 * 0.5,
                "ew_resid_60d_pct": baseline_resid_60,
                "hit_rate_20d_pct": 50.0,
                "hit_rate_60d_pct": 55.0,
            },
            baseline_universe_eval={
                "n_positions": 50,
                "weighted_ret_20d_pct": baseline_resid_60 * 0.3,
                "weighted_ret_60d_pct": baseline_resid_60 * 0.5,
                "weighted_resid_20d_pct": baseline_resid_60 * 0.3,
                "weighted_resid_60d_pct": baseline_resid_60 * 0.5,
                "ew_ret_20d_pct": baseline_resid_60 * 0.3,
                "ew_ret_60d_pct": baseline_resid_60 * 0.5,
                "ew_resid_20d_pct": baseline_resid_60 * 0.3,
                "ew_resid_60d_pct": baseline_resid_60 * 0.5,
                "hit_rate_20d_pct": 48.0,
                "hit_rate_60d_pct": 50.0,
            },
            strategy_positions=[
                {"ticker": "AATK", "weight_pct": 30.0},
                {"ticker": "BBTK", "weight_pct": 70.0},
            ],
            baseline_topk_positions=[],
            baseline_universe_positions=[],
            turnover={"position_turnover": 0.3, "weight_turnover_pct": 15.0},
        )

    def test_n_snapshots(self):
        """Should count snapshots correctly."""
        results = [
            self._make_snapshot("2024-03-31", 5.0, 2.0),
            self._make_snapshot("2024-04-30", 3.0, 1.0),
        ]
        agg = aggregate_results(results)
        assert agg["n_snapshots"] == 2

    def test_mean_strategy_return(self):
        """Mean strategy return should be average of per-date values."""
        results = [
            self._make_snapshot("2024-03-31", 5.0, 2.0),
            self._make_snapshot("2024-04-30", 3.0, 1.0),
        ]
        agg = aggregate_results(results)
        # (5+3)/2 = 4.0
        assert abs(agg["strategy_weighted_resid_60d_pct_mean"] - 4.0) < 0.01

    def test_spread_vs_topk(self):
        """Spread should be strategy - baseline average."""
        results = [
            self._make_snapshot("2024-03-31", 5.0, 2.0),
            self._make_snapshot("2024-04-30", 3.0, 1.0),
        ]
        agg = aggregate_results(results)
        # Per-date spreads: (5-2)=3, (3-1)=2 -> mean=2.5
        assert abs(agg["spread_vs_topk_60d_mean"] - 2.5) < 0.01

    def test_regime_split(self):
        """Per-regime stats should partition snapshots."""
        results = [
            self._make_snapshot("2024-03-31", 5.0, 2.0, "RISK_ON"),
            self._make_snapshot("2024-04-30", -3.0, -1.0, "RISK_OFF"),
        ]
        agg = aggregate_results(results)
        per_regime = agg["per_regime"]
        assert "RISK_ON" in per_regime
        assert "RISK_OFF" in per_regime
        assert per_regime["RISK_ON"]["n_snapshots"] == 1
        assert per_regime["RISK_OFF"]["n_snapshots"] == 1

    def test_turnover_aggregation(self):
        """Mean turnover should average across snapshots."""
        results = [
            self._make_snapshot("2024-03-31", 5.0, 2.0),
            self._make_snapshot("2024-04-30", 3.0, 1.0),
        ]
        agg = aggregate_results(results)
        assert agg["mean_position_turnover"] == 0.3
        assert agg["mean_weight_turnover_pct"] == 15.0

    def test_top_tickers(self):
        """Should count ticker frequency across snapshots."""
        results = [
            self._make_snapshot("2024-03-31", 5.0, 2.0),
            self._make_snapshot("2024-04-30", 3.0, 1.0),
        ]
        agg = aggregate_results(results)
        # Both snapshots have AATK and BBTK
        top = {t["ticker"]: t["appearances"] for t in agg["top_tickers"]}
        assert top.get("AATK") == 2
        assert top.get("BBTK") == 2

    def test_empty_results(self):
        """Should return empty dict for no results."""
        assert aggregate_results([]) == {}


# =============================================================================
# TEST: _percentile
# =============================================================================

class TestPercentile:
    def test_p50_is_median(self):
        assert _percentile([1.0, 2.0, 3.0], 50) == 2.0

    def test_p0_is_min(self):
        assert _percentile([1.0, 2.0, 3.0], 0) == 1.0

    def test_p100_is_max(self):
        assert _percentile([1.0, 2.0, 3.0], 100) == 3.0

    def test_empty_returns_zero(self):
        assert _percentile([], 50) == 0.0


# =============================================================================
# TEST: build_universe_baseline
# =============================================================================

class TestBuildUniverseBaseline:
    def test_filters_to_eligible_dev(self):
        """Should include only eligible drug_developer tickers."""
        data = _make_archive_data({
            "OK": {"rec": _make_rec(), "opt": 0.5, "composite_rank": 1},
            "DD": {"rec": _make_rec(drawdown=-0.50), "opt": 0.5, "composite_rank": 2},
            "CM": {"rec": _make_rec(), "opt": 0.5, "composite_rank": 3,
                   "archetype": "commercial_biotech"},
        })
        result = build_universe_baseline(data, DecisionRuleset())
        tickers = [p["ticker"] for p in result]
        assert "OK" in tickers
        assert "DD" not in tickers
        assert "CM" not in tickers

    def test_equal_weight(self):
        """All eligible tickers should have equal weight."""
        data = _make_archive_data({
            "A": {"rec": _make_rec(), "opt": 0.5, "composite_rank": 1},
            "B": {"rec": _make_rec(), "opt": 0.5, "composite_rank": 2},
        })
        result = build_universe_baseline(data, DecisionRuleset())
        assert len(result) == 2
        assert result[0]["weight_pct"] == result[1]["weight_pct"]


# =============================================================================
# TEST: Panel cost fields
# =============================================================================

class TestPanelCostFields:
    """Verify panel COLUMNS include cost columns."""

    def test_panel_columns_include_cost_fields(self):
        """PANEL_COLUMNS contains adv_dollars, est_cost_bps, and net return columns."""
        from run_decision_strategy_backtest import PANEL_COLUMNS

        for col in ["adv_dollars", "est_cost_bps", "participation_pct",
                     "fwd_ret_20d_net", "fwd_ret_60d_net"]:
            assert col in PANEL_COLUMNS, f"Missing {col} in PANEL_COLUMNS"


# =============================================================================
# TEST: Cost wiring into portfolio construction
# =============================================================================

def _make_archive_data_with_market(
    tickers_data: Dict[str, Dict[str, Any]],
    date_str: str = "2024-06-30",
) -> ArchiveData:
    """Build ArchiveData with market_data for cost wiring tests."""
    tickers = list(tickers_data.keys())
    recs = {t: d["rec"] for t, d in tickers_data.items()}
    archetypes = {t: d.get("archetype", "drug_developer") for t, d in tickers_data.items()}
    optionalities = {t: d.get("opt", 0.70) for t, d in tickers_data.items()}
    market_data = {t: d.get("market_data", {}) for t, d in tickers_data.items()}
    rows = []
    for t in tickers:
        row = {
            "ticker": t,
            "archetype": archetypes[t],
            "clinical_optionality_pct_dev": str(optionalities[t]) if optionalities[t] is not None else "",
        }
        rank = tickers_data[t].get("composite_rank")
        if rank is not None:
            row["composite_rank"] = str(rank)
        rows.append(row)

    return ArchiveData(
        date_str=date_str,
        rows=rows,
        recs=recs,
        archetypes=archetypes,
        optionalities=optionalities,
        tickers=tickers,
        market_data=market_data,
    )


class TestCostWiringInPortfolio:
    """Cost haircut flows through build_strategy_portfolio when enabled."""

    def _two_ticker_archive(self):
        """Two identical tickers except ADV: CHEAP ($200M) vs COSTLY ($500K)."""
        base_rec = _make_rec(catalyst_days=30, catalyst_window=True)
        return _make_archive_data_with_market({
            "CHEAP": {
                "rec": {**base_rec},
                "archetype": "drug_developer",
                "opt": 0.75,
                "composite_rank": 1,
                "market_data": {"avg_volume": 2_000_000, "price": 100.0},  # ADV $200M
            },
            "COSTLY": {
                "rec": {**base_rec},
                "archetype": "drug_developer",
                "opt": 0.75,
                "composite_rank": 2,
                "market_data": {"avg_volume": 10_000, "price": 50.0},  # ADV $500K
            },
        })

    # Biotech-calibrated buckets: CHEAP (~230 bps RT) → 1.0x, COSTLY (~4036 bps RT) → floor
    _BIOTECH_BUCKETS = ((300, 1.0), (1000, 0.85), (2000, 0.70))

    def test_enabled_haircut_differentiates_weights(self):
        """With cost haircut enabled, high-cost ticker gets lower weight."""
        archive = self._two_ticker_archive()
        ruleset = DecisionRuleset(
            enable_cost_haircut=True,
            cost_impact_cap_bps=2000.0,
            cost_haircut_buckets=self._BIOTECH_BUCKETS,
        )
        result = build_strategy_portfolio(archive, ruleset, ["A", "B"], top_k=20)

        by_ticker = {p["ticker"]: p for p in result}
        assert "CHEAP" in by_ticker and "COSTLY" in by_ticker

        # COSTLY should have lower weight due to cost haircut
        assert by_ticker["COSTLY"]["weight_pct"] < by_ticker["CHEAP"]["weight_pct"], (
            f"COSTLY ({by_ticker['COSTLY']['weight_pct']}%) should be < "
            f"CHEAP ({by_ticker['CHEAP']['weight_pct']}%)"
        )

        # Cost fields should be populated
        assert by_ticker["COSTLY"]["cost_haircut_applied"] == "1"
        assert float(by_ticker["COSTLY"]["cost_mult"]) < 1.0

    def test_enabled_haircut_band_stepdown(self):
        """High-cost ticker gets band step-down when cost_mult <= 0.70."""
        archive = self._two_ticker_archive()
        ruleset = DecisionRuleset(
            enable_cost_haircut=True,
            cost_impact_cap_bps=2000.0,
            cost_haircut_buckets=self._BIOTECH_BUCKETS,
        )
        result = build_strategy_portfolio(archive, ruleset, ["A", "B"], top_k=20)
        by_ticker = {p["ticker"]: p for p in result}

        cheap_cost_mult = float(by_ticker["CHEAP"].get("cost_mult") or 1.0)
        costly_cost_mult = float(by_ticker["COSTLY"].get("cost_mult") or 1.0)

        # CHEAP (~230 bps RT < 300 threshold) → 1.0x, COSTLY (~4036 bps > 2000) → floor 0.55x
        assert cheap_cost_mult == 1.0, f"CHEAP should be 1.0x, got {cheap_cost_mult}"
        assert costly_cost_mult < cheap_cost_mult, (
            f"COSTLY cost_mult ({costly_cost_mult}) should be < CHEAP ({cheap_cost_mult})"
        )

    def test_disabled_haircut_identical_weights(self):
        """With cost haircut disabled, identical tickers get identical weights."""
        archive = self._two_ticker_archive()
        ruleset = DecisionRuleset(enable_cost_haircut=False)
        result = build_strategy_portfolio(archive, ruleset, ["A", "B"], top_k=20)

        by_ticker = {p["ticker"]: p for p in result}
        assert "CHEAP" in by_ticker and "COSTLY" in by_ticker

        # Weights should be identical (cost not applied)
        assert by_ticker["CHEAP"]["weight_pct"] == by_ticker["COSTLY"]["weight_pct"]

        # Cost fields should show no haircut
        assert by_ticker["COSTLY"].get("cost_haircut_applied") in ("", "0", None)

    def test_no_market_data_graceful(self):
        """Tickers without market_data get est_cost_bps=None, no haircut."""
        base_rec = _make_rec(catalyst_days=30, catalyst_window=True)
        archive = _make_archive_data_with_market({
            "NODATA": {
                "rec": {**base_rec},
                "archetype": "drug_developer",
                "opt": 0.75,
                "composite_rank": 1,
                # No market_data key
            },
        })
        ruleset = DecisionRuleset(enable_cost_haircut=True, cost_impact_cap_bps=2000.0)
        result = build_strategy_portfolio(archive, ruleset, ["A", "B"], top_k=20)
        assert len(result) == 1
        assert result[0]["est_cost_bps"] is None
