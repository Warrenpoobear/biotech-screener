#!/usr/bin/env python3
"""
Tests for scripts/calibrate_ruleset_from_panel.py

Tests re-tier logic, candidate evaluation, scoring, and constraint checking.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Add paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from calibrate_ruleset_from_panel import (
    check_neighbor_stability,
    check_panel_data_quality,
    compute_ridge_summary,
    evaluate_candidate,
    read_panel,
    retier_row,
    run_2d_sweep,
    score_candidate,
)


# =============================================================================
# HELPERS
# =============================================================================

def _row(
    eligible: str = "1",
    optionality: float | None = 0.60,
    catalyst_mode: str = "specific_days",
    catalyst_days_raw: float | None = 30,
    fwd_ret_60d: float | None = 5.0,
    fwd_max_dd_60d: float | None = -10.0,
    tier: str = "A",
    as_of_date: str = "2025-06-30",
    ticker: str = "AAA",
    **kwargs,
) -> Dict[str, Any]:
    """Build a synthetic panel row."""
    return {
        "as_of_date": as_of_date,
        "ticker": ticker,
        "tier": tier,
        "band": "M",
        "eligible": eligible,
        "weight": 5.0,
        "tier_reason": "",
        "risk_flags": "",
        "catalyst_mode": catalyst_mode,
        "mom_state": "neutral",
        "optionality": optionality,
        "catalyst_days_raw": catalyst_days_raw,
        "ruleset_id": "test",
        "fwd_ret_20d": None,
        "fwd_ret_60d": fwd_ret_60d,
        "fwd_max_dd_20d": None,
        "fwd_max_dd_60d": fwd_max_dd_60d,
        "fwd_dd_missing_reason": "",
        **kwargs,
    }


# =============================================================================
# TEST: retier_row
# =============================================================================

class TestRetierRow:
    """Tests for the lightweight re-tier function."""

    def test_high_opt_near_catalyst_is_A(self):
        row = _row(optionality=0.70, catalyst_mode="specific_days", catalyst_days_raw=30)
        assert retier_row(row, a_floor=0.55, b_floor=0.30, catalyst_near_days=90) == "A"

    def test_high_opt_far_catalyst_is_B(self):
        """Far catalyst (days=200 > mid_days=180) + high opt → B."""
        row = _row(optionality=0.70, catalyst_mode="specific_days", catalyst_days_raw=200)
        assert retier_row(row, a_floor=0.55, b_floor=0.30, catalyst_near_days=90, catalyst_mid_days=180) == "B"

    def test_high_opt_mid_catalyst_is_A(self):
        """Mid catalyst (90 < 120 ≤ 180) + high opt → A (actionable)."""
        row = _row(optionality=0.70, catalyst_mode="specific_days", catalyst_days_raw=120)
        assert retier_row(row, a_floor=0.55, b_floor=0.30, catalyst_near_days=90, catalyst_mid_days=180) == "A"

    def test_mod_opt_near_catalyst_is_B(self):
        row = _row(optionality=0.40, catalyst_mode="specific_days", catalyst_days_raw=30)
        assert retier_row(row, a_floor=0.55, b_floor=0.30, catalyst_near_days=90) == "B"

    def test_low_opt_is_C(self):
        row = _row(optionality=0.20, catalyst_mode="specific_days", catalyst_days_raw=30)
        assert retier_row(row, a_floor=0.55, b_floor=0.30, catalyst_near_days=90) == "C"

    def test_ineligible_is_D(self):
        row = _row(eligible="0", optionality=0.90)
        assert retier_row(row, a_floor=0.55, b_floor=0.30, catalyst_near_days=90) == "D"

    def test_no_optionality_is_C(self):
        row = _row(optionality=None, catalyst_mode="specific_days", catalyst_days_raw=30)
        assert retier_row(row, a_floor=0.55, b_floor=0.30, catalyst_near_days=90) == "C"

    def test_blended_window_counts_as_near(self):
        row = _row(optionality=0.70, catalyst_mode="blended_window", catalyst_days_raw=None)
        assert retier_row(row, a_floor=0.55, b_floor=0.30, catalyst_near_days=90) == "A"

    def test_no_catalyst_data_high_opt_is_B(self):
        row = _row(optionality=0.70, catalyst_mode="no_upcoming", catalyst_days_raw=None)
        assert retier_row(row, a_floor=0.55, b_floor=0.30, catalyst_near_days=90) == "B"

    def test_no_catalyst_data_mod_opt_is_C(self):
        row = _row(optionality=0.40, catalyst_mode="no_upcoming", catalyst_days_raw=None)
        assert retier_row(row, a_floor=0.55, b_floor=0.30, catalyst_near_days=90) == "C"

    def test_lowering_a_floor_promotes_to_A(self):
        """Ticker at opt=0.52 is C under a_floor=0.55, but A under a_floor=0.50."""
        row = _row(optionality=0.52, catalyst_mode="specific_days", catalyst_days_raw=30)
        assert retier_row(row, a_floor=0.55, b_floor=0.30, catalyst_near_days=90) == "B"
        assert retier_row(row, a_floor=0.50, b_floor=0.30, catalyst_near_days=90) == "A"

    def test_raising_a_floor_demotes_from_A(self):
        """Ticker at opt=0.58 is A under a_floor=0.55, but B under a_floor=0.60."""
        row = _row(optionality=0.58, catalyst_mode="specific_days", catalyst_days_raw=30)
        assert retier_row(row, a_floor=0.55, b_floor=0.30, catalyst_near_days=90) == "A"
        assert retier_row(row, a_floor=0.60, b_floor=0.30, catalyst_near_days=90) == "B"

    def test_missing_catalyst_mode_no_data(self):
        """catalyst_mode='missing' (no data) → no catalyst path."""
        row = _row(optionality=0.70, catalyst_mode="missing", catalyst_days_raw=None)
        # No catalyst data → high_opt → B
        assert retier_row(row, a_floor=0.55, b_floor=0.30, catalyst_near_days=90) == "B"


# =============================================================================
# TEST: evaluate_candidate
# =============================================================================

class TestEvaluateCandidate:
    """Tests for candidate evaluation logic."""

    def test_tier_distribution(self):
        """Verify tier counts match re-tiering."""
        rows = [
            _row(ticker="A1", optionality=0.70, catalyst_mode="specific_days", catalyst_days_raw=30, fwd_ret_60d=10.0),
            _row(ticker="B1", optionality=0.40, catalyst_mode="specific_days", catalyst_days_raw=30, fwd_ret_60d=5.0),
            _row(ticker="C1", optionality=0.20, catalyst_mode="specific_days", catalyst_days_raw=30, fwd_ret_60d=-5.0),
            _row(ticker="D1", eligible="0", fwd_ret_60d=-20.0),
        ]
        result = evaluate_candidate(rows, a_floor=0.55, b_floor=0.30, catalyst_near_days=90)
        assert result["tier_distribution"]["A"] == 25.0  # 1 of 4
        assert result["tier_distribution"]["B"] == 25.0  # 1 of 4
        assert result["tier_distribution"]["C"] == 25.0  # 1 of 4
        assert result["tier_distribution"]["D"] == 25.0  # 1 of 4
        # Eligible-only distribution (3 eligible rows: 1 A, 1 B, 1 C)
        elig = result["eligible_tier_distribution"]
        assert elig["A"] == pytest.approx(33.3, abs=0.1)
        assert elig["B"] == pytest.approx(33.3, abs=0.1)
        assert elig["C"] == pytest.approx(33.3, abs=0.1)

    def test_separation_positive_when_ab_better(self):
        """AB outperforming CD → positive separation."""
        rows = [
            _row(ticker="A1", optionality=0.70, catalyst_mode="specific_days", catalyst_days_raw=30, fwd_ret_60d=20.0),
            _row(ticker="A2", optionality=0.60, catalyst_mode="specific_days", catalyst_days_raw=30, fwd_ret_60d=10.0),
            _row(ticker="C1", optionality=0.20, catalyst_mode="specific_days", catalyst_days_raw=30, fwd_ret_60d=-5.0),
            _row(ticker="C2", optionality=0.10, catalyst_mode="specific_days", catalyst_days_raw=30, fwd_ret_60d=-10.0),
        ]
        result = evaluate_candidate(rows, a_floor=0.55, b_floor=0.30, catalyst_near_days=90)
        assert result["separation_60d"] is not None
        assert result["separation_60d"] > 0

    def test_mean_a_count(self):
        """Mean A-count computed across dates."""
        rows = [
            _row(ticker="A1", optionality=0.70, catalyst_mode="specific_days", catalyst_days_raw=30, as_of_date="2025-06-30"),
            _row(ticker="A2", optionality=0.80, catalyst_mode="specific_days", catalyst_days_raw=30, as_of_date="2025-06-30"),
            _row(ticker="A3", optionality=0.70, catalyst_mode="specific_days", catalyst_days_raw=30, as_of_date="2025-07-31"),
        ]
        result = evaluate_candidate(rows, a_floor=0.55, b_floor=0.30, catalyst_near_days=90)
        # Date 1: 2 A's, Date 2: 1 A → mean = 1.5
        assert result["mean_a_count_per_date"] == 1.5


# =============================================================================
# TEST: score_candidate
# =============================================================================

class TestScoreCandidate:
    """Tests for scoring and constraint checking."""

    def test_passes_when_all_constraints_met(self):
        candidate = {
            "tier_distribution": {"A": 10.0, "B": 30.0, "C": 40.0, "D": 20.0},
            "eligible_tier_distribution": {"A": 12.5, "B": 37.5, "C": 50.0},
            "separation_60d": 5.0,
            "ab_median_dd_60d": -8.0,
            "mean_turnover_pct": 30.0,
        }
        score, violations = score_candidate(candidate, min_a_pct=3.0, max_turnover=50.0)
        assert len(violations) == 0
        assert score > 0

    def test_fails_when_a_pct_too_low(self):
        candidate = {
            "tier_distribution": {"A": 1.0, "B": 30.0, "C": 49.0, "D": 20.0},
            "eligible_tier_distribution": {"A": 1.25, "B": 37.5, "C": 61.25},
            "separation_60d": 5.0,
            "ab_median_dd_60d": -8.0,
            "mean_turnover_pct": 30.0,
        }
        score, violations = score_candidate(candidate, min_a_pct=3.0, max_turnover=50.0)
        assert any("A-tier" in v for v in violations)

    def test_fails_when_turnover_too_high(self):
        candidate = {
            "tier_distribution": {"A": 10.0, "B": 30.0, "C": 40.0, "D": 20.0},
            "eligible_tier_distribution": {"A": 12.5, "B": 37.5, "C": 50.0},
            "separation_60d": 5.0,
            "ab_median_dd_60d": -8.0,
            "mean_turnover_pct": 60.0,
        }
        score, violations = score_candidate(candidate, min_a_pct=3.0, max_turnover=50.0)
        assert any("turnover" in v for v in violations)

    def test_fails_when_separation_negative(self):
        candidate = {
            "tier_distribution": {"A": 10.0, "B": 30.0, "C": 40.0, "D": 20.0},
            "eligible_tier_distribution": {"A": 12.5, "B": 37.5, "C": 50.0},
            "separation_60d": -2.0,
            "ab_median_dd_60d": -8.0,
            "mean_turnover_pct": 30.0,
        }
        score, violations = score_candidate(candidate, min_a_pct=3.0, max_turnover=50.0)
        assert any("separation" in v for v in violations)

    def test_higher_separation_means_higher_score(self):
        base = {
            "tier_distribution": {"A": 10.0, "B": 30.0, "C": 40.0, "D": 20.0},
            "eligible_tier_distribution": {"A": 12.5, "B": 37.5, "C": 50.0},
            "ab_median_dd_60d": -8.0,
            "mean_turnover_pct": 30.0,
        }
        c1 = {**base, "separation_60d": 5.0}
        c2 = {**base, "separation_60d": 10.0}
        s1, _ = score_candidate(c1, min_a_pct=0, max_turnover=100)
        s2, _ = score_candidate(c2, min_a_pct=0, max_turnover=100)
        assert s2 > s1

    def test_worse_drawdown_lowers_score(self):
        base = {
            "tier_distribution": {"A": 10.0, "B": 30.0, "C": 40.0, "D": 20.0},
            "eligible_tier_distribution": {"A": 12.5, "B": 37.5, "C": 50.0},
            "separation_60d": 10.0,
            "mean_turnover_pct": 30.0,
        }
        c1 = {**base, "ab_median_dd_60d": -5.0}
        c2 = {**base, "ab_median_dd_60d": -20.0}
        s1, _ = score_candidate(c1, min_a_pct=0, max_turnover=100)
        s2, _ = score_candidate(c2, min_a_pct=0, max_turnover=100)
        assert s1 > s2  # better (less negative) DD → higher score


# =============================================================================
# TEST: Panel reading round-trip
# =============================================================================

class TestPanelRoundTrip:
    """Test that read_panel correctly parses numeric columns."""

    def test_parses_numeric_columns(self, tmp_path):
        csv_path = tmp_path / "panel.csv"
        with open(csv_path, "w", newline="") as f:
            f.write(
                "as_of_date,ticker,tier,band,eligible,weight,"
                "tier_reason,risk_flags,catalyst_mode,mom_state,"
                "optionality,catalyst_days_raw,ruleset_id,"
                "fwd_ret_20d,fwd_ret_60d,fwd_max_dd_20d,fwd_max_dd_60d,"
                "fwd_dd_missing_reason\n"
            )
            f.write(
                "2025-06-30,AAA,A,M,1,5.0,"
                "high_opt+catalyst_near,,specific_days,neutral,"
                "0.70,30,abc123,"
                "2.5,8.3,-3.2,-5.1,\n"
            )
            f.write(
                "2025-06-30,BBB,C,S,1,0,"
                "low_opt,,no_upcoming,headwind,"
                ",,abc123,"
                ",,,,no_price_series\n"
            )

        rows = read_panel(csv_path)
        assert len(rows) == 2

        # AAA: all numeric fields parsed
        assert rows[0]["optionality"] == pytest.approx(0.70)
        assert rows[0]["catalyst_days_raw"] == pytest.approx(30.0)
        assert rows[0]["fwd_ret_60d"] == pytest.approx(8.3)
        assert rows[0]["fwd_max_dd_60d"] == pytest.approx(-5.1)

        # BBB: empty numerics → None
        assert rows[1]["optionality"] is None
        assert rows[1]["catalyst_days_raw"] is None
        assert rows[1]["fwd_ret_60d"] is None


# =============================================================================
# TEST: Determinism
# =============================================================================

class TestDeterminism:
    """Verify same inputs produce same outputs."""

    def test_same_rows_same_result(self):
        rows = [
            _row(ticker="X", optionality=0.60, catalyst_mode="specific_days", catalyst_days_raw=30, fwd_ret_60d=5.0),
            _row(ticker="Y", optionality=0.20, catalyst_mode="no_upcoming", catalyst_days_raw=None, fwd_ret_60d=-3.0),
        ]
        r1 = evaluate_candidate(rows, a_floor=0.55, b_floor=0.30, catalyst_near_days=90)
        r2 = evaluate_candidate(rows, a_floor=0.55, b_floor=0.30, catalyst_near_days=90)
        assert r1 == r2


# =============================================================================
# TEST: Data-quality gate
# =============================================================================

class TestDataQualityGate:
    """Tests for the catalyst coverage pre-check."""

    def test_ok_when_catalyst_coverage_sufficient(self):
        """Mostly specific_days → passes."""
        rows = [
            _row(ticker="A1", catalyst_mode="specific_days"),
            _row(ticker="A2", catalyst_mode="specific_days"),
            _row(ticker="A3", catalyst_mode="no_upcoming"),
            _row(ticker="A4", catalyst_mode="specific_days"),
        ]
        ok, diag = check_panel_data_quality(rows, max_missing_catalyst_pct=50.0)
        assert ok
        assert diag["reason"] == "ok"
        assert diag["catalyst_missing_pct"] == 0.0

    def test_fail_when_catalyst_mostly_missing(self):
        """79% missing catalyst → fails at 50% threshold."""
        rows = []
        # 4 eligible with catalyst data
        for i in range(4):
            rows.append(_row(ticker=f"G{i}", catalyst_mode="specific_days"))
        # 15 eligible with catalyst missing
        for i in range(15):
            rows.append(_row(ticker=f"M{i}", catalyst_mode="missing"))
        ok, diag = check_panel_data_quality(rows, max_missing_catalyst_pct=50.0)
        assert not ok
        assert diag["reason"] == "insufficient_catalyst_coverage"
        assert diag["catalyst_missing_pct"] > 50.0

    def test_ineligible_rows_excluded_from_catalyst_check(self):
        """Ineligible rows should not count toward catalyst coverage."""
        rows = [
            _row(ticker="E1", catalyst_mode="specific_days"),  # eligible
            _row(ticker="E2", catalyst_mode="specific_days"),  # eligible
            # Many ineligible rows with missing catalyst
            _row(ticker="D1", eligible="0", catalyst_mode="missing"),
            _row(ticker="D2", eligible="0", catalyst_mode="missing"),
            _row(ticker="D3", eligible="0", catalyst_mode="missing"),
            _row(ticker="D4", eligible="0", catalyst_mode="missing"),
        ]
        ok, diag = check_panel_data_quality(rows, max_missing_catalyst_pct=50.0)
        assert ok  # Only eligible rows count → 0% missing among eligible

    def test_fail_on_empty_panel(self):
        ok, diag = check_panel_data_quality([], max_missing_catalyst_pct=50.0)
        assert not ok
        assert diag["reason"] == "empty_panel"

    def test_fail_on_no_eligible_rows(self):
        rows = [
            _row(ticker="D1", eligible="0", catalyst_mode="missing"),
            _row(ticker="D2", eligible="0", catalyst_mode="missing"),
        ]
        ok, diag = check_panel_data_quality(rows, max_missing_catalyst_pct=50.0)
        assert not ok
        assert diag["reason"] == "no_eligible_rows"

    def test_exactly_at_threshold_passes(self):
        """50% missing at 50% threshold → passes (not strictly greater)."""
        rows = [
            _row(ticker="G1", catalyst_mode="specific_days"),
            _row(ticker="M1", catalyst_mode="missing"),
        ]
        ok, diag = check_panel_data_quality(rows, max_missing_catalyst_pct=50.0)
        assert ok  # 50.0% == threshold, NOT > threshold

    def test_empty_catalyst_mode_treated_as_missing(self):
        """Empty string catalyst_mode counts as missing."""
        rows = [
            _row(ticker="E1", catalyst_mode=""),
            _row(ticker="E2", catalyst_mode="specific_days"),
        ]
        ok, diag = check_panel_data_quality(rows, max_missing_catalyst_pct=40.0)
        assert not ok  # 50% > 40%


# =============================================================================
# TEST: Date filtering
# =============================================================================

class TestDateFiltering:
    """Tests for panel row date filtering (integration-level)."""

    def test_date_min_filters_rows(self):
        rows = [
            _row(ticker="A1", as_of_date="2024-06-30", catalyst_mode="specific_days"),
            _row(ticker="A2", as_of_date="2025-03-31", catalyst_mode="specific_days"),
            _row(ticker="A3", as_of_date="2025-06-30", catalyst_mode="specific_days"),
        ]
        filtered = [r for r in rows if r["as_of_date"] >= "2025-01-01"]
        assert len(filtered) == 2
        assert all(r["as_of_date"] >= "2025-01-01" for r in filtered)

    def test_date_max_filters_rows(self):
        rows = [
            _row(ticker="A1", as_of_date="2025-03-31", catalyst_mode="specific_days"),
            _row(ticker="A2", as_of_date="2025-06-30", catalyst_mode="specific_days"),
            _row(ticker="A3", as_of_date="2025-12-31", catalyst_mode="specific_days"),
        ]
        filtered = [r for r in rows if r["as_of_date"] <= "2025-06-30"]
        assert len(filtered) == 2

    def test_date_range_filters_both(self):
        rows = [
            _row(ticker="A1", as_of_date="2024-06-30"),
            _row(ticker="A2", as_of_date="2025-03-31"),
            _row(ticker="A3", as_of_date="2025-06-30"),
            _row(ticker="A4", as_of_date="2026-01-31"),
        ]
        filtered = [r for r in rows if r["as_of_date"] >= "2025-01-01" and r["as_of_date"] <= "2025-12-31"]
        assert len(filtered) == 2


# =============================================================================
# TEST: 2D Sweep
# =============================================================================

class Test2DSweep:
    """Tests for the 2D (a_floor × catalyst_near_days) sweep."""

    def _make_rows(self):
        """Build a panel with varied catalyst timing and optionality."""
        rows = []
        # A-tier candidates: high opt, near catalyst
        for i in range(5):
            rows.append(_row(
                ticker=f"AN{i}", optionality=0.70,
                catalyst_mode="specific_days", catalyst_days_raw=20,
                fwd_ret_60d=15.0, as_of_date="2025-06-30",
            ))
        # B-tier: high opt, far catalyst (depends on catalyst_near threshold)
        for i in range(5):
            rows.append(_row(
                ticker=f"BF{i}", optionality=0.65,
                catalyst_mode="specific_days", catalyst_days_raw=50,
                fwd_ret_60d=8.0, as_of_date="2025-06-30",
            ))
        # C-tier: low opt
        for i in range(5):
            rows.append(_row(
                ticker=f"CL{i}", optionality=0.20,
                catalyst_mode="specific_days", catalyst_days_raw=20,
                fwd_ret_60d=-5.0, as_of_date="2025-06-30",
            ))
        # D-tier: ineligible
        for i in range(5):
            rows.append(_row(
                ticker=f"DI{i}", eligible="0",
                catalyst_mode="missing", fwd_ret_60d=-10.0,
                as_of_date="2025-06-30",
            ))
        return rows

    def test_grid_size(self):
        """2D sweep produces correct number of candidates."""
        rows = self._make_rows()
        a_floors = [0.50, 0.55, 0.60]
        cat_nears = [30, 60, 90]
        candidates = run_2d_sweep(rows, a_floors, cat_nears, b_floor=0.30,
                                  min_a_pct=3.0, max_turnover=50.0)
        assert len(candidates) == 9  # 3 × 3

    def test_each_candidate_has_catalyst_near_days(self):
        """Every candidate dict includes the catalyst_near_days it was evaluated with."""
        rows = self._make_rows()
        candidates = run_2d_sweep(rows, [0.55], [30, 90], b_floor=0.30,
                                  min_a_pct=0, max_turnover=100)
        assert all("catalyst_near_days" in c for c in candidates)
        assert candidates[0]["catalyst_near_days"] == 30
        assert candidates[1]["catalyst_near_days"] == 90

    def test_wider_catalyst_window_promotes_more_to_A(self):
        """With catalyst_days_raw=50, a wider catalyst_near window includes more A-tier."""
        rows = self._make_rows()
        # catalyst_near=30 → only the 5 AN* tickers (days=20) qualify as near
        r30 = run_2d_sweep(rows, [0.55], [30], b_floor=0.30, min_a_pct=0, max_turnover=100)
        # catalyst_near=60 → both AN* (days=20) and BF* (days=50) qualify as near
        r60 = run_2d_sweep(rows, [0.55], [60], b_floor=0.30, min_a_pct=0, max_turnover=100)
        a30 = r30[0]["eligible_tier_distribution"]["A"]
        a60 = r60[0]["eligible_tier_distribution"]["A"]
        assert a60 >= a30

    def test_determinism(self):
        """Same inputs produce identical results."""
        rows = self._make_rows()
        r1 = run_2d_sweep(rows, [0.50, 0.55], [30, 90], b_floor=0.30,
                          min_a_pct=3.0, max_turnover=50.0)
        r2 = run_2d_sweep(rows, [0.50, 0.55], [30, 90], b_floor=0.30,
                          min_a_pct=3.0, max_turnover=50.0)
        assert r1 == r2


class TestRidgeSummary:
    """Tests for ridge summary computation."""

    def test_one_entry_per_catalyst_near(self):
        """Ridge has one entry per unique sweep_val value."""
        candidates = [
            {"a_floor": 0.50, "sweep_val": 30, "score": 1.0, "passed": True,
             "separation_60d": 2.0, "eligible_tier_distribution": {"A": 10.0},
             "ab_median_dd_60d": -5.0, "mean_a_count_per_date": 3.0},
            {"a_floor": 0.55, "sweep_val": 30, "score": 0.5, "passed": True,
             "separation_60d": 1.0, "eligible_tier_distribution": {"A": 8.0},
             "ab_median_dd_60d": -6.0, "mean_a_count_per_date": 2.0},
            {"a_floor": 0.50, "sweep_val": 90, "score": 0.8, "passed": True,
             "separation_60d": 1.5, "eligible_tier_distribution": {"A": 7.0},
             "ab_median_dd_60d": -4.0, "mean_a_count_per_date": 2.5},
        ]
        ridge = compute_ridge_summary(candidates)
        assert len(ridge) == 2  # sweep_val 30 and 90

    def test_picks_best_a_floor_per_catalyst_near(self):
        """Ridge picks highest-score a_floor for each sweep_val."""
        candidates = [
            {"a_floor": 0.50, "sweep_val": 30, "score": 1.0, "passed": True,
             "separation_60d": 2.0, "eligible_tier_distribution": {"A": 10.0},
             "ab_median_dd_60d": -5.0, "mean_a_count_per_date": 3.0},
            {"a_floor": 0.55, "sweep_val": 30, "score": 2.0, "passed": True,
             "separation_60d": 3.0, "eligible_tier_distribution": {"A": 12.0},
             "ab_median_dd_60d": -4.0, "mean_a_count_per_date": 4.0},
        ]
        ridge = compute_ridge_summary(candidates)
        assert ridge[0]["best_a_floor"] == 0.55  # higher score


class TestNeighborStability:
    """Tests for neighbor stability check."""

    def test_finds_immediate_neighbors(self):
        """Checks ±1 step in each dimension, excluding self."""
        a_floors = [0.40, 0.50, 0.60]
        cat_nears = [30, 60, 90]
        best = {"a_floor": 0.50, "sweep_val": 60}
        candidates = [
            {"a_floor": af, "sweep_val": cn, "score": 0, "passed": True, "separation_60d": 0}
            for af in a_floors for cn in cat_nears
        ]
        neighbors = check_neighbor_stability(best, candidates, a_floors, cat_nears)
        # Center (0.50, 60) excluded. Neighbors are all (±1, ±1) combos.
        # That's 3×3 - 1 = 8 neighbors
        assert len(neighbors) == 8

    def test_corner_has_fewer_neighbors(self):
        """Corner candidate (first a_floor, first cat_near) has fewer neighbors."""
        a_floors = [0.40, 0.50, 0.60]
        cat_nears = [30, 60, 90]
        best = {"a_floor": 0.40, "sweep_val": 30}
        candidates = [
            {"a_floor": af, "sweep_val": cn, "score": 0, "passed": True, "separation_60d": 0}
            for af in a_floors for cn in cat_nears
        ]
        neighbors = check_neighbor_stability(best, candidates, a_floors, cat_nears)
        # (0.40, 30) → neighbors are (0.40,60), (0.50,30), (0.50,60) = 3
        assert len(neighbors) == 3

    def test_excludes_self(self):
        """Best candidate itself is not in the neighbors list."""
        a_floors = [0.50]
        cat_nears = [60]
        best = {"a_floor": 0.50, "sweep_val": 60}
        candidates = [{"a_floor": 0.50, "sweep_val": 60, "score": 1, "passed": True, "separation_60d": 1}]
        neighbors = check_neighbor_stability(best, candidates, a_floors, cat_nears)
        assert len(neighbors) == 0


class TestSweepCatalystMidDays:
    """Tests for the --sweep-catalyst-mid-days dimension."""

    def _make_rows(self):
        """Panel rows with catalyst_days_raw at varied distances."""
        rows = []
        # Near catalyst (days=50, always within near_days=120)
        for i in range(5):
            rows.append(_row(
                ticker=f"NR{i}", optionality=0.70,
                catalyst_mode="specific_days", catalyst_days_raw=50,
                fwd_ret_60d=15.0, as_of_date="2025-06-30",
            ))
        # Boundary catalyst (days=170, mid at 180 → mid; mid at 150 → far)
        for i in range(5):
            rows.append(_row(
                ticker=f"BD{i}", optionality=0.70,
                catalyst_mode="specific_days", catalyst_days_raw=170,
                fwd_ret_60d=8.0, as_of_date="2025-06-30",
            ))
        # Low opt (always C)
        for i in range(5):
            rows.append(_row(
                ticker=f"CL{i}", optionality=0.20,
                catalyst_mode="specific_days", catalyst_days_raw=50,
                fwd_ret_60d=-5.0, as_of_date="2025-06-30",
            ))
        return rows

    def test_mid_sweep_grid_size(self):
        """2D sweep with sweep_dim='cat_mid' produces correct grid."""
        rows = self._make_rows()
        a_floors = [0.55, 0.60]
        mid_values = [150, 180]
        candidates = run_2d_sweep(
            rows, a_floors, mid_values, b_floor=0.30,
            min_a_pct=0, max_turnover=100,
            sweep_dim="cat_mid", base_near_days=120, base_mid_days=180,
        )
        assert len(candidates) == 4  # 2 × 2
        # All should have the correct catalyst_near_days (held fixed)
        assert all(c["catalyst_near_days"] == 120 for c in candidates)
        # sweep_val should match the mid values
        assert {c["sweep_val"] for c in candidates} == {150, 180}

    def test_mid_days_boundary_changes_tier(self):
        """Rows with catalyst_days_raw=170: mid at mid_days=180 (A), far at mid_days=150 (B)."""
        rows = self._make_rows()
        # mid_days=180: 170 <= 180 → mid → actionable → A (for high opt)
        r180 = run_2d_sweep(
            rows, [0.60], [180], b_floor=0.30,
            min_a_pct=0, max_turnover=100,
            sweep_dim="cat_mid", base_near_days=120,
        )
        # mid_days=150: 170 > 150 → far → NOT actionable → B (for high opt)
        r150 = run_2d_sweep(
            rows, [0.60], [150], b_floor=0.30,
            min_a_pct=0, max_turnover=100,
            sweep_dim="cat_mid", base_near_days=120,
        )
        a_at_180 = r180[0]["eligible_tier_distribution"]["A"]
        a_at_150 = r150[0]["eligible_tier_distribution"]["A"]
        # Mid=180 should promote more to A than mid=150
        assert a_at_180 > a_at_150
