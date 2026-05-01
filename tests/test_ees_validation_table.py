"""Tests for scripts/research/ees_validation_table.

Pins the column schema and the universe set so downstream consumers
(plotting, verdict review on 2026-05-22) don't break silently if the
table layout changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.research.ees_validation_table import _residualize_linear, _row_for, _spearman, _t_stat  # noqa: E402

EXPECTED_COLUMNS = [
    "snap_date",
    "universe",
    "n_total",
    "n_universe",
    "n_quarantined",
    "n_clean",
    "n_resolved",
    "n_ic",
    "priced_move_cov",
    "short_interest_cov",
    "market_cap_cov",
    "close_price_cov",
    "ees_score_cov",
    "median_pmv",
    "p90_pmv",
    "median_abs_realized_5d",
    "mean_expectation_error",
    "top_third_excess_5d",
    "bottom_third_excess_5d",
    "spread_5d",
    "spearman_ees_vs_ex5d",
    "tstat_ees",
    "spearman_pmv_vs_ex5d",
    "tstat_pmv",
    "spearman_ees_resid_vs_ex5d",
    "tstat_ees_resid",
    "directional_hit_rate",
    "brier_directional",
]


def test_column_schema_pinned():
    """If this fails, downstream verdict-review tooling will need updating."""
    rankings = []
    panel = {}
    row = _row_for("2026-04-14", "A", rankings, panel)
    assert list(row.keys()) == EXPECTED_COLUMNS


def test_empty_input_returns_nones_for_metrics():
    row = _row_for("2026-04-14", "A", rankings=[], panel={})
    assert row["n_total"] == 0
    assert row["n_ic"] == 0
    assert row["spearman_ees_vs_ex5d"] is None
    assert row["spread_5d"] is None
    assert row["directional_hit_rate"] is None


def test_spearman_basic():
    # Perfect monotonic
    assert abs(_spearman([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) - 1.0) < 1e-9
    assert abs(_spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) + 1.0) < 1e-9


def test_spearman_tied_constant_x_returns_none():
    """When xs are constant, rank correlation is mathematically undefined.

    Regression test for the production-outage scenario where
    institutional_summary_delta.json is absent and inst_delta_z silently
    falls back to all zeros — naive competitive-ranking Spearman would
    return a meaningless number against any variable y.

    See `incomplete_production_run_fallback_2026_05_01` memo.
    """
    assert _spearman([0, 0, 0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6, 7, 8]) is None
    assert _spearman([1.0] * 10, list(range(10))) is None


def test_spearman_tied_constant_y_returns_none():
    """Symmetric case: ys constant, xs varying."""
    assert _spearman([1, 2, 3, 4, 5, 6, 7, 8], [0, 0, 0, 0, 0, 0, 0, 0]) is None
    assert _spearman(list(range(10)), [0.5] * 10) is None


def test_spearman_both_constant_returns_none():
    """Degenerate case: both inputs constant."""
    assert _spearman([0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]) is None
    assert _spearman([7.0] * 8, [3.5] * 8) is None


def test_spearman_normal_inputs_unaffected():
    """Tied-constant guard must NOT affect non-degenerate inputs."""
    # Perfect positive
    assert abs(_spearman([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]) - 1.0) < 1e-9
    # Perfect negative
    assert abs(_spearman([1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]) + 1.0) < 1e-9
    # Imperfect positive (moderate correlation)
    rho = _spearman([1, 2, 3, 4, 5, 6, 7, 8], [2, 1, 4, 3, 6, 5, 8, 7])
    assert rho is not None and 0.7 < rho < 1.0


def test_residualize_removes_linear_component():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [2.0, 4.0, 6.0, 8.0, 10.0]  # ys = 2*xs
    resid = _residualize_linear(ys, xs)
    # Residuals should all be ~0
    assert all(abs(r) < 1e-9 for r in resid)


def test_t_stat_handles_edge_cases():
    assert _t_stat(None, 100) is None
    assert _t_stat(0.5, 2) is None  # n too small
    assert _t_stat(1.0, 100) is None  # |rho| >= 1


def test_row_with_synthetic_resolved_data():
    rankings = [
        {
            "ticker": "AAA",
            "ees_v3_score": "1.5",
            "priced_move_pct": "30",
            "short_interest_pct": "5",
            "market_cap_mm": "1000",
            "close_price": "10",
            "ees_eligible": "True",
            "next_catalyst_date": "2026-04-16",
            "expectation_error_score": "0.2",
        },
        {
            "ticker": "BBB",
            "ees_v3_score": "-0.5",
            "priced_move_pct": "60",
            "short_interest_pct": "10",
            "market_cap_mm": "500",
            "close_price": "5",
            "ees_eligible": "True",
            "next_catalyst_date": "2026-04-17",
            "expectation_error_score": "-0.1",
        },
        {
            "ticker": "CCC",
            "ees_v3_score": "0.5",
            "priced_move_pct": "45",
            "short_interest_pct": "8",
            "market_cap_mm": "2000",
            "close_price": "20",
            "ees_eligible": "True",
            "next_catalyst_date": "2026-04-18",
            "expectation_error_score": "0.05",
        },
    ]
    panel = {
        ("2026-04-14", "AAA"): {
            "snap_date": "2026-04-14",
            "ticker": "AAA",
            "actual_abs_move_5d": "0.05",
            "excess_return_5d": "0.02",
            "forward_complete": "true",
        },
        ("2026-04-14", "BBB"): {
            "snap_date": "2026-04-14",
            "ticker": "BBB",
            "actual_abs_move_5d": "0.10",
            "excess_return_5d": "-0.03",
            "forward_complete": "true",
        },
        ("2026-04-14", "CCC"): {
            "snap_date": "2026-04-14",
            "ticker": "CCC",
            "actual_abs_move_5d": "0.04",
            "excess_return_5d": "0.01",
            "forward_complete": "true",
        },
    }
    row = _row_for("2026-04-14", "A", rankings, panel)
    assert row["n_total"] == 3
    assert row["n_universe"] == 3
    assert row["n_quarantined"] == 0
    assert row["n_clean"] == 3
    assert row["n_resolved"] == 3
    # 100% coverage on all 4 inputs
    assert row["priced_move_cov"] == 1.0
    assert row["close_price_cov"] == 1.0
    # Spearman should be defined (n=3 < 5 → returns None)
    assert row["spearman_ees_vs_ex5d"] is None
    # n_ic=3 < 5 → no IC, no spread, no hit rate
    assert row["n_ic"] == 3
    assert row["spread_5d"] is None
