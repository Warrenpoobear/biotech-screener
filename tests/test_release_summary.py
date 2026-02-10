#!/usr/bin/env python3
"""
Tests for scripts/generate_release_summary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from generate_release_summary import generate_release_summary


# =============================================================================
# FIXTURES
# =============================================================================

def _make_calibration(
    baseline_a_floor: float = 0.55,
    candidate_a_floor: float = 0.52,
    separation: float = 5.0,
    score: float = 3.5,
    passed: bool = True,
) -> Dict[str, Any]:
    """Build a minimal calibration report dict."""
    return {
        "config": {
            "panel_path": "artifacts/walkforward_panel.csv",
            "panel_rows": 2000,
            "a_floor_values": [0.45, 0.50, 0.52, 0.55, 0.60],
            "baseline_a_floor": baseline_a_floor,
            "min_a_pct": 3.0,
            "max_turnover": 50.0,
        },
        "baseline": {
            "a_floor": baseline_a_floor,
            "tier_distribution": {"A": 8.0, "B": 30.0, "C": 45.0, "D": 17.0},
            "tier_metrics": {
                "A": {"count": 160, "mean_ret_60d": 6.0, "median_ret_60d": 5.0, "hit_rate_60d": 55.0, "mean_max_dd_60d": -12.0, "median_max_dd_60d": -10.0},
                "B": {"count": 600, "mean_ret_60d": 3.0, "median_ret_60d": 2.5, "hit_rate_60d": 52.0, "mean_max_dd_60d": -15.0, "median_max_dd_60d": -13.0},
                "C": {"count": 900, "mean_ret_60d": -1.0, "median_ret_60d": -2.0, "hit_rate_60d": 45.0, "mean_max_dd_60d": -18.0, "median_max_dd_60d": -16.0},
                "D": {"count": 340, "mean_ret_60d": -5.0, "median_ret_60d": -6.0, "hit_rate_60d": 35.0, "mean_max_dd_60d": -25.0, "median_max_dd_60d": -22.0},
            },
            "ab_median_ret_60d": 3.0,
            "cd_median_ret_60d": -3.0,
            "separation_60d": 6.0,
            "ab_median_dd_60d": -12.0,
            "mean_a_count_per_date": 7.0,
            "mean_top25_overlap_pct": 65.0,
            "mean_turnover_pct": 35.0,
        },
        "candidates": [
            {
                "a_floor": candidate_a_floor,
                "tier_distribution": {"A": 12.0, "B": 28.0, "C": 42.0, "D": 18.0},
                "tier_metrics": {
                    "A": {"count": 240, "mean_ret_60d": 7.0, "median_ret_60d": 6.0, "hit_rate_60d": 58.0, "mean_max_dd_60d": -11.0, "median_max_dd_60d": -9.0},
                    "B": {"count": 560, "mean_ret_60d": 2.5, "median_ret_60d": 2.0, "hit_rate_60d": 51.0, "mean_max_dd_60d": -14.0, "median_max_dd_60d": -12.0},
                    "C": {"count": 840, "mean_ret_60d": -1.5, "median_ret_60d": -2.5, "hit_rate_60d": 44.0, "mean_max_dd_60d": -19.0, "median_max_dd_60d": -17.0},
                    "D": {"count": 360, "mean_ret_60d": -4.0, "median_ret_60d": -5.0, "hit_rate_60d": 36.0, "mean_max_dd_60d": -24.0, "median_max_dd_60d": -21.0},
                },
                "ab_median_ret_60d": 3.5,
                "cd_median_ret_60d": -3.5,
                "separation_60d": separation,
                "ab_median_dd_60d": -11.0,
                "mean_a_count_per_date": 10.0,
                "mean_top25_overlap_pct": 60.0,
                "mean_turnover_pct": 38.0,
                "score": score,
                "passed": passed,
                "violations": [],
            },
        ],
        "best": {
            "a_floor": candidate_a_floor,
            "score": score,
            "passed": passed,
            "separation_60d": separation,
            "tradeoffs": ["Lower stability: overlap 60.0% vs baseline 65.0%"],
        },
    }


def _make_walkforward() -> Dict[str, Any]:
    return {
        "coverage": {
            "total_rows": 2000,
            "with_fwd_returns": 1800,
            "with_fwd_returns_pct": 90.0,
            "with_fwd_max_dd": 1750,
            "with_fwd_max_dd_pct": 87.5,
        },
        "stability": {
            "mean_top_n_overlap": 62.0,
            "mean_turnover": 38.0,
            "n_transitions": 21,
        },
    }


# =============================================================================
# TESTS
# =============================================================================

class TestReleaseSummary:
    """Tests for release summary generation."""

    def test_contains_header(self):
        cal = _make_calibration()
        md = generate_release_summary(cal, None, None, None)
        assert "# Ruleset Release Summary" in md

    def test_contains_parameter_changes(self):
        cal = _make_calibration(baseline_a_floor=0.55, candidate_a_floor=0.52)
        md = generate_release_summary(cal, None, None, None)
        assert "tier_a_optionality_floor" in md
        assert "0.55" in md
        assert "0.52" in md

    def test_no_change_when_baseline_equals_candidate(self):
        cal = _make_calibration(baseline_a_floor=0.55, candidate_a_floor=0.55)
        md = generate_release_summary(cal, None, None, None)
        assert "No change recommended" in md

    def test_contains_objective_improvement_table(self):
        cal = _make_calibration()
        md = generate_release_summary(cal, None, None, None)
        assert "AB-CD separation" in md
        assert "AB median max-DD" in md

    def test_contains_tier_distribution(self):
        cal = _make_calibration()
        md = generate_release_summary(cal, None, None, None)
        assert "Tier Distribution" in md
        assert "| A |" in md
        assert "| D |" in md

    def test_contains_per_tier_detail(self):
        cal = _make_calibration()
        md = generate_release_summary(cal, None, None, None)
        assert "Per-Tier 60d Return Detail" in md
        assert "Hit Rate" in md
        assert "Mean Max-DD" in md

    def test_contains_tradeoffs(self):
        cal = _make_calibration()
        md = generate_release_summary(cal, None, None, None)
        assert "Tradeoffs" in md
        assert "Lower stability" in md

    def test_contains_scoring(self):
        cal = _make_calibration(score=3.5)
        md = generate_release_summary(cal, None, None, None)
        assert "3.5" in md
        assert "PASS" in md

    def test_contains_qa_checklist(self):
        cal = _make_calibration()
        md = generate_release_summary(cal, None, None, None)
        assert "QA Checklist" in md
        assert "bump_ruleset.py" in md
        assert "promote_ruleset.py" in md

    def test_includes_walkforward_coverage(self):
        cal = _make_calibration()
        wf = _make_walkforward()
        md = generate_release_summary(cal, wf, None, None)
        assert "Walkforward Panel Coverage" in md
        assert "2000" in md
        assert "90.0%" in md

    def test_handles_missing_walkforward(self):
        cal = _make_calibration()
        md = generate_release_summary(cal, None, None, None)
        # Should not crash, and should not contain walkforward section
        assert "Walkforward Panel Coverage" not in md

    def test_includes_active_ruleset_id(self):
        cal = _make_calibration()
        active = {"id": "d3cdf5c8", "file": "v2_phase2_default.json"}
        md = generate_release_summary(cal, None, active, None)
        assert "d3cdf5c8" in md
        assert "v2_phase2_default.json" in md

    def test_fail_status_shows_violations(self):
        cal = _make_calibration(passed=False)
        cal["candidates"][0]["passed"] = False
        cal["candidates"][0]["violations"] = ["A-tier 1.0% < min 3.0%"]
        md = generate_release_summary(cal, None, None, None)
        assert "FAIL" in md
