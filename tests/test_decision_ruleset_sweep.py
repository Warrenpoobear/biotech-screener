"""Tests for run_decision_ruleset_sweep.py"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from decision_engine import DecisionRuleset
from run_decision_ruleset_sweep import (
    GRID_AXES,
    compute_turnover,
    evaluate_snapshot,
    generate_grid,
    run_holdout_validation,
    score_ruleset,
    split_dates,
    ArchiveData,
    ForwardReturnData,
)


# =============================================================================
# GRID TESTS
# =============================================================================


class TestGenerateGrid:
    def test_grid_validity(self):
        """All combos have tier_a > tier_b; grid size > 0."""
        grid = generate_grid()
        assert len(grid) > 0
        for rs in grid:
            assert rs.tier_a_optionality_floor > rs.tier_b_optionality_floor, (
                f"Invalid combo: a={rs.tier_a_optionality_floor} "
                f"<= b={rs.tier_b_optionality_floor}"
            )

    def test_grid_contains_default(self):
        """Default params (0.60, 0.30, 90, 2) must be in the grid."""
        grid = generate_grid()
        default = DecisionRuleset()
        ids = {rs.ruleset_id for rs in grid}
        assert default.ruleset_id in ids, (
            f"Default ruleset {default.ruleset_id} not in grid of {len(grid)}"
        )

    def test_grid_ids_unique(self):
        """Each ruleset in the grid has a unique id."""
        grid = generate_grid()
        ids = [rs.ruleset_id for rs in grid]
        assert len(ids) == len(set(ids))

    def test_grid_expected_size(self):
        """Grid size matches expected count from GRID_AXES."""
        grid = generate_grid()
        # 3 a_floors x 3 b_floors x 3 catalyst x 2 sponsor = 54
        # minus invalid (b >= a): 3 per a_floor=0.55(all3), 2 per a_floor=0.60(0.30,0.35->skip 0.35 is not >=0.60 actually)
        # a=0.55: b can be 0.25 only (0.30>=0.55? no, 0.30<0.55, 0.35<0.55) -> all 3 valid
        # a=0.60: b can be 0.25, 0.30, 0.35 (all < 0.60) -> 3 valid
        # a=0.65: b can be 0.25, 0.30, 0.35 (all < 0.65) -> 3 valid
        # So 9 valid param combos * 3 catalyst * 2 sponsor = 54 total
        # Wait — check: b=0.35 < a=0.55? Yes (0.35 < 0.55). All valid.
        # Total = 9 * 3 * 2 = 54
        assert len(grid) == 54


# =============================================================================
# SCORE TESTS
# =============================================================================


class TestScoreRuleset:
    def test_score_deterministic(self):
        """Same inputs produce same composite score."""
        per_date = {
            "2024-01-31": {
                "A": {"n": 10, "mean_pct": 5.0, "resid_mean_pct": 3.0,
                       "hit_pct": 60.0, "resid_hit_pct": 55.0},
                "C": {"n": 30, "mean_pct": 1.0, "resid_mean_pct": -1.0,
                       "hit_pct": 45.0, "resid_hit_pct": 40.0},
            },
            "2024-04-30": {
                "A": {"n": 8, "mean_pct": 4.0, "resid_mean_pct": 2.5,
                       "hit_pct": 55.0, "resid_hit_pct": 50.0},
                "C": {"n": 25, "mean_pct": 0.5, "resid_mean_pct": -0.5,
                       "hit_pct": 40.0, "resid_hit_pct": 38.0},
            },
        }
        turnover = 0.15

        s1 = score_ruleset(per_date, turnover)
        s2 = score_ruleset(per_date, turnover)
        assert s1["composite_score"] == s2["composite_score"]
        assert s1["delta_ac_resid_60d"] == s2["delta_ac_resid_60d"]

    def test_score_positive_delta(self):
        """When A outperforms C, delta should be positive."""
        per_date = {
            "2024-01-31": {
                "A": {"n": 15, "mean_pct": 10.0, "resid_mean_pct": 8.0,
                       "hit_pct": 70.0, "resid_hit_pct": 65.0},
                "C": {"n": 40, "mean_pct": 2.0, "resid_mean_pct": -2.0,
                       "hit_pct": 45.0, "resid_hit_pct": 40.0},
            },
        }
        scores = score_ruleset(per_date, turnover=0.0)
        assert scores["delta_ac_resid_60d"] > 0


# =============================================================================
# TURNOVER TESTS
# =============================================================================


class TestComputeTurnover:
    def test_no_change(self):
        """Identical tier assignments → 0 turnover."""
        history = [
            {"A": "A", "B": "B", "C": "C"},
            {"A": "A", "B": "B", "C": "C"},
        ]
        assert compute_turnover(history) == 0.0

    def test_full_change(self):
        """All tiers changed → 1.0 turnover."""
        history = [
            {"A": "A", "B": "B"},
            {"A": "C", "B": "D"},
        ]
        assert compute_turnover(history) == 1.0

    def test_single_snapshot(self):
        """Single snapshot → 0 turnover."""
        assert compute_turnover([{"A": "A"}]) == 0.0

    def test_partial_overlap(self):
        """Only common tickers count."""
        history = [
            {"A": "A", "B": "B", "C": "C"},
            {"A": "A", "B": "D"},  # C absent, B changed
        ]
        # common: A (same), B (changed) → 1/2 = 0.5
        assert compute_turnover(history) == 0.5


# =============================================================================
# HOLDOUT TESTS
# =============================================================================


class TestSplitDates:
    def test_split_basic(self):
        dates = ["2024-01-31", "2024-06-30", "2025-01-31", "2025-06-30"]
        early, late = split_dates(dates, "2024-12-31")
        assert early == ["2024-01-31", "2024-06-30"]
        assert late == ["2025-01-31", "2025-06-30"]

    def test_split_boundary_inclusive(self):
        """Cutoff date is included in early."""
        dates = ["2024-12-31", "2025-01-31"]
        early, late = split_dates(dates, "2024-12-31")
        assert early == ["2024-12-31"]
        assert late == ["2025-01-31"]

    def test_split_all_early(self):
        dates = ["2024-01-31", "2024-06-30"]
        early, late = split_dates(dates, "2025-12-31")
        assert early == dates
        assert late == []


class TestHoldoutValidation:
    def _make_results(self):
        """Build minimal results with per_date stats for two rulesets."""
        per_date_early = {
            "2024-01-31": {
                "A": {"n": 10, "mean_pct": 8.0, "resid_mean_pct": 6.0,
                       "median_resid_pct": 5.5, "winsor_resid_pct": 5.8,
                       "hit_pct": 65.0, "resid_hit_pct": 60.0},
                "C": {"n": 30, "mean_pct": 2.0, "resid_mean_pct": -1.0,
                       "median_resid_pct": -1.5, "winsor_resid_pct": -0.8,
                       "hit_pct": 40.0, "resid_hit_pct": 35.0},
            },
        }
        per_date_late = {
            "2025-01-31": {
                "A": {"n": 8, "mean_pct": 6.0, "resid_mean_pct": 4.0,
                       "median_resid_pct": 3.5, "winsor_resid_pct": 3.8,
                       "hit_pct": 60.0, "resid_hit_pct": 55.0},
                "C": {"n": 25, "mean_pct": 1.0, "resid_mean_pct": -0.5,
                       "median_resid_pct": -1.0, "winsor_resid_pct": -0.3,
                       "hit_pct": 42.0, "resid_hit_pct": 38.0},
            },
        }
        results = [
            {
                "ruleset_id": "good_one",
                "tier_a_floor": 0.55, "tier_b_floor": 0.30,
                "catalyst_near_days": 90, "sponsor_threshold": 2,
                "per_date": {**per_date_early, **per_date_late},
            },
            {
                "ruleset_id": "bad_one",
                "tier_a_floor": 0.65, "tier_b_floor": 0.30,
                "catalyst_near_days": 90, "sponsor_threshold": 2,
                "per_date": {
                    "2024-01-31": {
                        "A": {"n": 3, "mean_pct": 1.0, "resid_mean_pct": -2.0,
                               "median_resid_pct": -2.0, "winsor_resid_pct": -2.0,
                               "hit_pct": 33.0, "resid_hit_pct": 33.0},
                        "C": {"n": 40, "mean_pct": 3.0, "resid_mean_pct": 1.0,
                               "median_resid_pct": 0.5, "winsor_resid_pct": 0.8,
                               "hit_pct": 50.0, "resid_hit_pct": 45.0},
                    },
                    "2025-01-31": {
                        "A": {"n": 2, "mean_pct": -1.0, "resid_mean_pct": -3.0,
                               "median_resid_pct": -3.0, "winsor_resid_pct": -3.0,
                               "hit_pct": 0.0, "resid_hit_pct": 0.0},
                        "C": {"n": 35, "mean_pct": 2.0, "resid_mean_pct": 0.5,
                               "median_resid_pct": 0.0, "winsor_resid_pct": 0.3,
                               "hit_pct": 48.0, "resid_hit_pct": 42.0},
                    },
                },
            },
        ]
        tier_histories = {
            "good_one": [
                ("2024-01-31", {"T1": "A", "T2": "C", "T3": "C"}),
                ("2025-01-31", {"T1": "A", "T2": "C", "T3": "C"}),
            ],
            "bad_one": [
                ("2024-01-31", {"T1": "A", "T2": "C"}),
                ("2025-01-31", {"T1": "C", "T2": "C"}),
            ],
        }
        return results, tier_histories

    def test_holdout_identifies_passing(self):
        results, histories = self._make_results()
        holdout = run_holdout_validation(
            results, histories, cutoff="2024-12-31", top_k=2
        )
        assert holdout["n_early_dates"] == 1
        assert holdout["n_late_dates"] == 1
        # good_one should pass (positive late delta, sufficient A obs)
        passing_ids = [c["ruleset_id"] for c in holdout["candidates"]]
        assert "good_one" in passing_ids

    def test_holdout_rejects_negative_delta(self):
        results, histories = self._make_results()
        holdout = run_holdout_validation(
            results, histories, cutoff="2024-12-31", top_k=2
        )
        # bad_one has negative late delta
        for v in holdout["validated"]:
            if v["ruleset_id"] == "bad_one":
                assert not v["gate_positive_delta"]
                assert not v["passes_holdout"]
