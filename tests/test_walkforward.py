#!/usr/bin/env python3
"""
Tests for walk-forward panel export, forward max-drawdown, and report metrics.
"""
from __future__ import annotations

import csv
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.returns_provider import CSVReturnsProvider
from decision_engine import DecisionRuleset
from run_decision_ruleset_sweep import ArchiveData
from run_decision_strategy_backtest import (
    PANEL_COLUMNS,
    build_all_dev_decisions,
)

# Lazy import — scripts/ not always on path
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


# =============================================================================
# HELPERS
# =============================================================================

def _write_price_csv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    """Write a tiny price CSV for testing. rows = [(date, ticker, close), ...]"""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "ticker", "close"])
        for row in rows:
            writer.writerow(row)


def _make_provider(tmp_path: Path, rows: list[tuple[str, str, str]]) -> CSVReturnsProvider:
    csv_path = tmp_path / "prices.csv"
    _write_price_csv(csv_path, rows)
    return CSVReturnsProvider(csv_path, price_col="close")


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


# =============================================================================
# TEST: Forward Max Drawdown
# =============================================================================

class TestForwardMaxDrawdown:
    """Tests for CSVReturnsProvider.get_forward_max_drawdown()."""

    def test_monotonic_up(self, tmp_path):
        """Monotonically increasing prices → dd = 0.0."""
        rows = [
            ("2024-01-01", "AAA", "100"),
            ("2024-01-02", "AAA", "101"),
            ("2024-01-03", "AAA", "102"),
            ("2024-01-04", "AAA", "103"),
            ("2024-01-05", "AAA", "104"),
            ("2024-01-08", "AAA", "105"),
            ("2024-01-09", "AAA", "106"),
            ("2024-01-10", "AAA", "107"),
            ("2024-01-11", "AAA", "108"),
            ("2024-01-12", "AAA", "109"),
            ("2024-01-15", "AAA", "110"),
        ]
        prov = _make_provider(tmp_path, rows)
        dd, reason = prov.get_forward_max_drawdown("AAA", "2024-01-01", 10)
        assert dd == 0.0
        assert reason == ""

    def test_peak_then_trough(self, tmp_path):
        """Known drawdown: peak at 100, trough at 80 → -20%."""
        rows = [
            ("2024-01-01", "BBB", "50"),   # as_of anchor (excluded)
            ("2024-01-02", "BBB", "100"),
            ("2024-01-03", "BBB", "95"),
            ("2024-01-04", "BBB", "90"),
            ("2024-01-05", "BBB", "80"),
            ("2024-01-08", "BBB", "85"),
            ("2024-01-09", "BBB", "90"),
            ("2024-01-10", "BBB", "95"),
            ("2024-01-11", "BBB", "100"),
            ("2024-01-12", "BBB", "105"),
            ("2024-01-15", "BBB", "120"),
        ]
        prov = _make_provider(tmp_path, rows)
        dd, reason = prov.get_forward_max_drawdown("BBB", "2024-01-01", 60)
        assert dd == pytest.approx(-0.20, abs=0.001)
        assert reason == ""

    def test_too_few_bars(self, tmp_path):
        """Fewer bars than MIN_BARS_FORWARD → None + 'series_too_short'."""
        rows = [
            ("2024-01-01", "CCC", "100"),
            ("2024-01-02", "CCC", "101"),
            ("2024-01-03", "CCC", "102"),
        ]
        prov = _make_provider(tmp_path, rows)
        dd, reason = prov.get_forward_max_drawdown("CCC", "2024-01-01", 60)
        assert dd is None
        assert reason == "series_too_short"

    def test_unknown_ticker(self, tmp_path):
        """Unknown ticker → None + 'no_price_series'."""
        rows = [("2024-01-01", "AAA", "100")]
        prov = _make_provider(tmp_path, rows)
        dd, reason = prov.get_forward_max_drawdown("ZZZ", "2024-01-01", 20)
        assert dd is None
        assert reason == "no_price_series"

    def test_all_equal(self, tmp_path):
        """All equal prices → dd = 0.0."""
        rows = [(f"2024-01-{d:02d}", "DDD", "100") for d in range(1, 16)]
        prov = _make_provider(tmp_path, rows)
        dd, reason = prov.get_forward_max_drawdown("DDD", "2024-01-01", 60)
        assert dd == 0.0
        assert reason == ""

    def test_horizon_respected(self, tmp_path):
        """Bars beyond horizon are excluded from DD calculation."""
        # 15 bars after as_of. Horizon=10 means only first 10 bars used.
        # First 10 bars are monotonic up, bar 11-15 crash.
        rows = [("2024-01-01", "EEE", "50")]  # as_of anchor
        for i in range(1, 16):
            price = 100 + i if i <= 10 else 50  # crash after bar 10
            rows.append((f"2024-01-{i+1:02d}", "EEE", str(price)))
        prov = _make_provider(tmp_path, rows)
        dd, reason = prov.get_forward_max_drawdown("EEE", "2024-01-01", 10)
        assert dd == 0.0  # crash happens outside horizon
        assert reason == ""


# =============================================================================
# TEST: Panel Schema
# =============================================================================

class TestPanelSchema:
    """Tests for panel columns and build_all_dev_decisions output."""

    def test_panel_columns_frozen(self):
        """PANEL_COLUMNS constant matches expected list."""
        expected = [
            "as_of_date", "ticker", "tier", "band", "eligible", "weight",
            "ineligible_reasons", "first_failed_gate",
            "tier_reason", "risk_flags", "catalyst_mode", "catalyst_strength", "mom_state",
            "optionality", "catalyst_days_raw", "ruleset_id",
            "drawdown_abs", "drawdown_xbi", "drawdown_rel_xbi",
            "dd_abs_margin", "dd_rel_margin", "rescued_by_rel", "dd_rel_margin_rescued",
            "optionality_margin_a", "actionable_catalyst",
            "fwd_ret_20d", "fwd_ret_60d", "fwd_max_dd_20d", "fwd_max_dd_60d",
            "fwd_dd_missing_reason",
            "adv_dollars", "est_cost_bps", "participation_pct",
            "fwd_ret_20d_net", "fwd_ret_60d_net",
            "catalyst_tilt_mult", "catalyst_tilt_applied",
            "mom_state_tilt_mult", "mom_state_tilt_applied",
        ]
        assert PANEL_COLUMNS == expected

    def test_panel_schema_has_net_columns(self):
        """PANEL_COLUMNS includes cost and net-return columns."""
        for col in ["adv_dollars", "est_cost_bps", "participation_pct",
                     "fwd_ret_20d_net", "fwd_ret_60d_net"]:
            assert col in PANEL_COLUMNS, f"Missing {col} in PANEL_COLUMNS"

    def test_panel_columns_has_drawdown_fields(self):
        """PANEL_COLUMNS includes regime-aware drawdown fields."""
        for col in ["drawdown_abs", "drawdown_xbi", "drawdown_rel_xbi"]:
            assert col in PANEL_COLUMNS, f"Missing {col} in PANEL_COLUMNS"

    def test_build_all_dev_decisions_keys(self):
        """build_all_dev_decisions returns expected keys for each ticker."""
        archive = _make_archive_data({
            "TICK1": {"rec": _make_rec(), "archetype": "drug_developer", "opt": 0.70, "composite_rank": 1},
            "TICK2": {"rec": _make_rec(), "archetype": "drug_developer", "opt": 0.30, "composite_rank": 2},
            "COMM1": {"rec": _make_rec(), "archetype": "commercial_biotech", "opt": None, "composite_rank": 3},
        })
        results = build_all_dev_decisions(archive, DecisionRuleset())
        # Should only include drug_developer tickers
        assert len(results) == 2
        tickers = {r["ticker"] for r in results}
        assert tickers == {"TICK1", "TICK2"}
        # Check keys
        expected_keys = {
            "ticker", "tier_dev", "band", "eligible",
            "ineligible_reasons", "first_failed_gate",
            "tier_reason",
            "risk_flags", "catalyst_mode", "catalyst_strength", "catalyst_days", "mom_state",
            "optionality",
            "drawdown_abs", "drawdown_xbi", "drawdown_rel_xbi",
            "dd_abs_margin", "dd_rel_margin", "rescued_by_rel",
            "dd_rel_margin_rescued",
            "optionality_margin_a", "actionable_catalyst",
            "catalyst_tilt_mult", "catalyst_tilt_applied",
            "mom_state_tilt_mult", "mom_state_tilt_applied",
        }
        for r in results:
            assert set(r.keys()) == expected_keys

    def test_build_all_dev_decisions_includes_ineligible(self):
        """Ineligible tickers (e.g. deep drawdown) are included in output."""
        archive = _make_archive_data({
            "GOOD": {"rec": _make_rec(drawdown=-0.10), "opt": 0.70, "composite_rank": 1},
            "BAD": {"rec": _make_rec(drawdown=-0.50), "opt": 0.70, "composite_rank": 2},
        })
        results = build_all_dev_decisions(archive, DecisionRuleset())
        assert len(results) == 2
        by_ticker = {r["ticker"]: r for r in results}
        assert by_ticker["GOOD"]["eligible"] == "1"
        assert by_ticker["BAD"]["eligible"] == "0"


# =============================================================================
# TEST: Tier Separation Metrics
# =============================================================================

class TestTierSeparationMetrics:
    """Tests for tier separation computation."""

    def test_known_values(self):
        """Synthetic panel with known returns produces correct metrics."""
        from run_walkforward_report import compute_tier_separation

        panel = [
            {"tier": "A", "fwd_ret_60d": 10.0, "fwd_max_dd_60d": -5.0},
            {"tier": "A", "fwd_ret_60d": 20.0, "fwd_max_dd_60d": -10.0},
            {"tier": "B", "fwd_ret_60d": -5.0, "fwd_max_dd_60d": -15.0},
            {"tier": "B", "fwd_ret_60d": 5.0, "fwd_max_dd_60d": -8.0},
            {"tier": "C", "fwd_ret_60d": -10.0, "fwd_max_dd_60d": -20.0},
            {"tier": "D", "fwd_ret_60d": "", "fwd_max_dd_60d": ""},
        ]
        result = compute_tier_separation(panel)
        by_tier = {r["tier"]: r for r in result}

        # Tier A: mean=(10+20)/2=15, median=15, hit=100%, mean_dd=(-5+-10)/2=-7.5
        assert by_tier["A"]["mean_ret"] == 15.0
        assert by_tier["A"]["hit_rate"] == 100.0
        assert by_tier["A"]["mean_max_dd"] == -7.5

        # Tier B: mean=(-5+5)/2=0, hit=50%
        assert by_tier["B"]["mean_ret"] == 0.0
        assert by_tier["B"]["hit_rate"] == 50.0

        # Tier C: mean=-10, hit=0%
        assert by_tier["C"]["mean_ret"] == -10.0
        assert by_tier["C"]["hit_rate"] == 0.0

        # Tier D: no valid returns
        assert by_tier["D"]["mean_ret"] is None

    def test_empty_panel(self):
        """Empty panel produces None for all metrics."""
        from run_walkforward_report import compute_tier_separation

        result = compute_tier_separation([])
        for entry in result:
            assert entry["count"] == 0
            assert entry["mean_ret"] is None


class TestCostSummary:
    """Tests for compute_cost_summary."""

    def test_cost_summary_filters_portfolio_only(self):
        """Only rows with weight > 0 contribute to cost stats."""
        from run_walkforward_report import compute_cost_summary

        panel = [
            # Portfolio position
            {"weight": 5.0, "est_cost_bps": 20.0, "participation_pct": 1.5, "adv_dollars": 5_000_000},
            {"weight": 3.0, "est_cost_bps": 40.0, "participation_pct": 2.0, "adv_dollars": 2_000_000},
            # Non-portfolio (weight=0): should be excluded
            {"weight": 0.0, "est_cost_bps": 0.0, "participation_pct": 0.0, "adv_dollars": 100_000},
            {"weight": 0, "est_cost_bps": 0, "participation_pct": 0, "adv_dollars": 50_000},
        ]
        result = compute_cost_summary(panel)
        assert result["n_portfolio_rows"] == 2
        assert result["median_cost_bps"] is not None
        assert result["low_adv_count"] == 0

    def test_cost_summary_empty_panel(self):
        """Empty panel returns None metrics."""
        from run_walkforward_report import compute_cost_summary

        result = compute_cost_summary([])
        assert result["n_portfolio_rows"] == 0
        assert result["median_cost_bps"] is None
