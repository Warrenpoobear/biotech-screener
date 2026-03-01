"""Tests for scripts/research/run_alpha_experiment.py.

Covers:
- OLS correctness (multi_ols)
- Exposure neutralization reduces beta tilt
- Determinism (same inputs → same outputs)
- Production sort key is used (rerank_faithful)
- Comparison report generation
- Edge cases (missing exposures, single row, etc.)
"""
from __future__ import annotations

import copy
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.research.run_alpha_experiment import (
    EXPOSURE_MAP,
    MCAP_ORDINAL,
    DateMetrics,
    ExperimentResult,
    _extract_anchor,
    _extract_exposure,
    generate_comparison_report,
    multi_ols,
    neutralize_exposures,
    rerank_faithful,
    spearman_ic,
    _round_opt,
)
from decision_engine import DecisionRuleset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_ruleset():
    """Load the default production ruleset."""
    rs_paths = sorted(
        (PROJECT_ROOT / "production_data" / "decision_rulesets").glob("*.json"),
    )
    if rs_paths:
        return DecisionRuleset.from_json(str(rs_paths[-1]))
    # Fallback: try latest snapshot
    snap_root = PROJECT_ROOT / "data" / "snapshots"
    dates = sorted([
        p.name for p in snap_root.iterdir()
        if p.is_dir() and len(p.name) == 10 and p.name[4] == "-"
        and (p / "decision_ruleset.json").exists()
    ])
    assert dates, "No ruleset found"
    return DecisionRuleset.from_json(str(snap_root / dates[-1] / "decision_ruleset.json"))


def _make_row(
    ticker: str,
    eligible: str = "1",
    alpha_cohort_pct: str = "0.5",
    composite_rank: str = "10",
    beta: str = "1.0",
    drawdown: str = "-0.10",
    rsi: str = "50.0",
    vol_60d: str = "0.80",
    mcap_bucket: str = "small",
    archetype: str = "drug_developer",
    tier_dev: str = "B",
    actionable_rank: str = "",
    clinical_optionality_pct_dev: str = "0.5",
) -> Dict[str, str]:
    """Create a minimal rankings row for testing."""
    return {
        "ticker": ticker,
        "eligible": eligible,
        "alpha_cohort_pct": alpha_cohort_pct,
        "composite_rank": composite_rank,
        "de_beta_xbi_60d": beta,
        "de_drawdown": drawdown,
        "de_rsi_14d": rsi,
        "de_vol_60d": vol_60d,
        "de_drawdown_rel_xbi": drawdown,
        "market_cap_bucket": mcap_bucket,
        "archetype": archetype,
        "tier_dev": tier_dev,
        "actionable_rank": actionable_rank,
        "clinical_optionality_pct_dev": clinical_optionality_pct_dev,
        "catalyst_days": "90",
        "catalyst_mode": "specific_days",
        "catalyst_event_type": "",
        "catalyst_source": "",
        "mom_state": "neutral",
        "sponsor_tier1_count": "0",
        "coinvest_score_z": "0",
        "clinical_score_z_tier": "0",
        "inst_delta_z": "0",
        "stage_bucket": "mid",
        "missingness_penalty": "0",
        "missing_components": "",
        "alpha_cohort_raw": "0.5",
        "commercial_quality_pct": "",
    }


def _make_test_rows(n: int = 20) -> List[Dict[str, str]]:
    """Create n test rows with varying beta and signal."""
    rows = []
    for i in range(n):
        # Create a clear beta-signal correlation:
        # Higher beta → higher alpha_cohort_pct (beta-tilted signal)
        beta = 0.5 + i * 0.1  # 0.5 to 2.4
        pct = 0.1 + i * 0.04  # 0.1 to 0.86
        rows.append(_make_row(
            ticker=f"T{i:03d}",
            alpha_cohort_pct=f"{pct:.4f}",
            composite_rank=str(n - i),
            beta=f"{beta:.4f}",
            drawdown=f"{-0.05 - i * 0.01:.4f}",
            rsi=f"{40 + i * 1.5:.1f}",
            vol_60d=f"{0.50 + i * 0.05:.4f}",
            mcap_bucket="small" if i < 10 else "mid",
            actionable_rank=str(i + 1),
        ))
    return rows


# ---------------------------------------------------------------------------
# multi_ols tests
# ---------------------------------------------------------------------------

class TestMultiOLS:
    def test_simple_linear(self):
        """OLS recovers y = 2 + 3*x."""
        x = [float(i) for i in range(20)]
        y = [2.0 + 3.0 * xi for xi in x]
        beta, r2 = multi_ols([x], y)
        assert len(beta) == 2
        assert abs(beta[0] - 2.0) < 1e-6
        assert abs(beta[1] - 3.0) < 1e-6
        assert abs(r2 - 1.0) < 1e-6

    def test_two_regressors(self):
        """OLS with two independent regressors."""
        n = 30
        x1 = [float(i) for i in range(n)]
        x2 = [float(i % 5) for i in range(n)]  # independent of x1
        y = [1.0 + 2.0 * x1[i] + 0.5 * x2[i] for i in range(n)]
        beta, r2 = multi_ols([x1, x2], y)
        assert abs(beta[0] - 1.0) < 1e-4
        assert abs(beta[1] - 2.0) < 1e-4
        assert abs(beta[2] - 0.5) < 1e-4
        assert abs(r2 - 1.0) < 1e-4

    def test_too_few_observations(self):
        """Returns zeros when n < p + 2."""
        x = [1.0, 2.0]
        y = [3.0, 4.0]
        beta, r2 = multi_ols([x], y)
        assert beta == [0.0, 0.0]
        assert r2 == 0.0

    def test_noisy_regression(self):
        """OLS on noisy data has 0 < R² < 1."""
        import random
        random.seed(42)
        n = 50
        x = [float(i) for i in range(n)]
        y = [2.0 + 3.0 * xi + random.gauss(0, 5) for xi in x]
        beta, r2 = multi_ols([x], y)
        assert 0.0 < r2 < 1.0
        assert abs(beta[1] - 3.0) < 2.0  # approximate recovery

    def test_constant_y(self):
        """Constant y → R² = 0."""
        x = [float(i) for i in range(10)]
        y = [5.0] * 10
        beta, r2 = multi_ols([x], y)
        assert r2 == 0.0


# ---------------------------------------------------------------------------
# Exposure extraction tests
# ---------------------------------------------------------------------------

class TestExposureExtraction:
    def test_beta_extraction(self):
        row = _make_row("TEST", beta="1.5")
        assert _extract_exposure(row, "beta") == 1.5

    def test_drawdown_extraction(self):
        row = _make_row("TEST", drawdown="-0.25")
        assert _extract_exposure(row, "drawdown") == -0.25

    def test_vol_extraction(self):
        row = _make_row("TEST", vol_60d="1.25")
        assert _extract_exposure(row, "vol") == 1.25

    def test_rsi_extraction(self):
        row = _make_row("TEST", rsi="65.5")
        assert _extract_exposure(row, "rsi") == 65.5

    def test_drawdown_rel_xbi_extraction(self):
        row = _make_row("TEST", drawdown="-0.15")
        assert _extract_exposure(row, "drawdown_rel_xbi") == -0.15

    def test_mcap_ordinal(self):
        for bucket, expected in MCAP_ORDINAL.items():
            row = _make_row("TEST", mcap_bucket=bucket)
            assert _extract_exposure(row, "mcap") == expected

    def test_unknown_exposure(self):
        row = _make_row("TEST")
        assert _extract_exposure(row, "nonexistent") is None

    def test_missing_value(self):
        row = _make_row("TEST", beta="")
        assert _extract_exposure(row, "beta") is None


# ---------------------------------------------------------------------------
# Anchor extraction tests
# ---------------------------------------------------------------------------

class TestAnchorExtraction:
    def test_alpha_cohort_mode(self, default_ruleset):
        from dataclasses import replace as dc_replace
        rs = dc_replace(default_ruleset, sort_anchor="alpha_cohort")
        row = _make_row("TEST", alpha_cohort_pct="0.75")
        assert _extract_anchor(row, rs) == 0.75

    def test_composite_rank_mode(self, default_ruleset):
        from dataclasses import replace as dc_replace
        rs = dc_replace(default_ruleset, sort_anchor="composite_rank")
        row = _make_row("TEST", composite_rank="10")
        # Negated: lower rank = better = higher signal
        assert _extract_anchor(row, rs) == -10.0


# ---------------------------------------------------------------------------
# Neutralization tests
# ---------------------------------------------------------------------------

class TestNeutralization:
    def test_reduces_beta_tilt(self, default_ruleset):
        """After neutralization, top-K mean beta should be closer to universe mean."""
        rows = _make_test_rows(20)

        # Baseline: top-10 mean beta (signal correlates with beta)
        baseline_top = rows[:10]
        baseline_mean_beta = statistics.mean(
            float(r["de_beta_xbi_60d"]) for r in baseline_top
        )
        universe_mean_beta = statistics.mean(
            float(r["de_beta_xbi_60d"]) for r in rows
        )

        # Neutralize
        neutralized, r2, coeffs = neutralize_exposures(
            rows, ["beta"], default_ruleset,
        )

        # Re-rank with neutralized signal
        from dataclasses import replace as dc_replace
        neutral_rs = dc_replace(default_ruleset, sort_anchor="alpha_cohort")
        neutralized = rerank_faithful(neutralized, neutral_rs)

        # Get new top-10 by actionable_rank
        ranked = sorted(
            [r for r in neutralized if r.get("eligible") == "1"],
            key=lambda r: int(r.get("actionable_rank") or 9999),
        )
        neutral_top = ranked[:10]
        neutral_mean_beta = statistics.mean(
            float(r["de_beta_xbi_60d"]) for r in neutral_top
        )

        # After neutralization, top-K beta should be closer to universe mean
        baseline_tilt = abs(baseline_mean_beta - universe_mean_beta)
        neutral_tilt = abs(neutral_mean_beta - universe_mean_beta)
        assert neutral_tilt < baseline_tilt, (
            f"Neutralization should reduce beta tilt: "
            f"baseline={baseline_tilt:.4f} vs neutral={neutral_tilt:.4f}"
        )

    def test_r2_positive(self, default_ruleset):
        """OLS R² should be positive when signal correlates with exposure."""
        rows = _make_test_rows(20)
        _, r2, _ = neutralize_exposures(rows, ["beta"], default_ruleset)
        assert r2 is not None
        assert r2 > 0.0

    def test_deterministic(self, default_ruleset):
        """Same inputs → same outputs."""
        rows = _make_test_rows(20)
        r1, r2_a, c1 = neutralize_exposures(
            copy.deepcopy(rows), ["beta", "drawdown"], default_ruleset,
        )
        r2, r2_b, c2 = neutralize_exposures(
            copy.deepcopy(rows), ["beta", "drawdown"], default_ruleset,
        )

        # Coefficients must match exactly
        assert c1 == c2
        assert r2_a == r2_b

        # Neutralized alpha_cohort_pct must match
        for a, b in zip(r1, r2):
            assert a.get("alpha_cohort_pct") == b.get("alpha_cohort_pct")

    def test_ineligible_rows_unchanged(self, default_ruleset):
        """Ineligible rows should not have their alpha_cohort_pct modified."""
        rows = _make_test_rows(10)
        rows[0]["eligible"] = "0"
        rows[0]["alpha_cohort_pct"] = "0.999"
        original_pct = rows[0]["alpha_cohort_pct"]

        neutralized, _, _ = neutralize_exposures(
            rows, ["beta"], default_ruleset,
        )
        assert neutralized[0]["alpha_cohort_pct"] == original_pct

    def test_too_few_rows_returns_unmodified(self, default_ruleset):
        """With too few valid rows, neutralization is skipped."""
        rows = [_make_row("T001", alpha_cohort_pct="0.5", beta="1.0")]
        original_pct = rows[0]["alpha_cohort_pct"]
        result, r2, coeffs = neutralize_exposures(
            rows, ["beta", "drawdown", "vol"], default_ruleset,
        )
        assert r2 is None
        assert result[0]["alpha_cohort_pct"] == original_pct

    def test_missing_exposure_skips_row(self, default_ruleset):
        """Rows with missing exposure values are skipped in OLS."""
        rows = _make_test_rows(10)
        rows[0]["de_beta_xbi_60d"] = ""  # Missing beta

        neutralized, r2, _ = neutralize_exposures(
            rows, ["beta"], default_ruleset,
        )
        # Row 0 should be skipped but others processed
        assert r2 is not None

    def test_multiple_exposures(self, default_ruleset):
        """Neutralization works with multiple exposure factors."""
        rows = _make_test_rows(20)
        _, r2, coeffs = neutralize_exposures(
            rows, ["beta", "drawdown", "vol", "mcap"], default_ruleset,
        )
        assert r2 is not None
        # coefficients: intercept + 4 exposures = 5
        assert len(coeffs) == 5

    def test_residual_percentiles_span_unit_interval(self, default_ruleset):
        """Neutralized alpha_cohort_pct values should span [0, 1]."""
        # Use rows with imperfect signal-beta correlation to get real spread
        import random
        random.seed(99)
        rows = _make_test_rows(30)
        for r in rows:
            # Add noise to break perfect correlation
            orig = float(r["alpha_cohort_pct"])
            r["alpha_cohort_pct"] = f"{orig + random.gauss(0, 0.1):.6f}"

        neutralized, _, _ = neutralize_exposures(
            rows, ["beta"], default_ruleset,
        )
        pcts = [
            float(r["alpha_cohort_pct"])
            for r in neutralized
            if r.get("eligible") == "1"
        ]
        assert min(pcts) == pytest.approx(0.0, abs=0.05)
        assert max(pcts) == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------
# Reranking tests
# ---------------------------------------------------------------------------

class TestRerankFaithful:
    def test_uses_production_sort_key(self, default_ruleset):
        """rerank_faithful should produce valid actionable_rank 1..N."""
        rows = _make_test_rows(15)
        reranked = rerank_faithful(rows, default_ruleset)

        eligible = [r for r in reranked if r.get("eligible") == "1"]
        ranks = [int(r["actionable_rank"]) for r in eligible]
        assert ranks == list(range(1, len(eligible) + 1))

    def test_ineligible_rows_get_empty_rank(self, default_ruleset):
        """Ineligible rows should have empty actionable_rank."""
        rows = _make_test_rows(10)
        rows[0]["eligible"] = "0"
        reranked = rerank_faithful(rows, default_ruleset)

        # Find the ineligible row by ticker (sort may reorder)
        ineligible = [r for r in reranked if r.get("ticker") == "T000"]
        assert len(ineligible) == 1
        assert ineligible[0]["actionable_rank"] == ""

        # All ineligible rows must have empty rank
        for r in reranked:
            if r.get("eligible") != "1":
                assert r["actionable_rank"] == ""

    def test_deterministic_rerank(self, default_ruleset):
        """Same inputs → same ranking."""
        rows = _make_test_rows(15)
        r1 = rerank_faithful(copy.deepcopy(rows), default_ruleset)
        r2 = rerank_faithful(copy.deepcopy(rows), default_ruleset)

        t1 = [r["ticker"] for r in r1]
        t2 = [r["ticker"] for r in r2]
        assert t1 == t2


# ---------------------------------------------------------------------------
# Spearman IC tests
# ---------------------------------------------------------------------------

class TestSpearmanIC:
    def test_perfect_correlation(self):
        assert spearman_ic([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) == pytest.approx(1.0)

    def test_perfect_anticorrelation(self):
        assert spearman_ic([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) == pytest.approx(-1.0)

    def test_too_few_returns_none(self):
        assert spearman_ic([1, 2], [3, 4]) is None

    def test_constant_signal_returns_none(self):
        assert spearman_ic([5, 5, 5, 5], [1, 2, 3, 4]) is None


# ---------------------------------------------------------------------------
# Comparison report tests
# ---------------------------------------------------------------------------

class TestComparisonReport:
    def test_baseline_only_report(self):
        """Report generation with baseline only (no neutralization)."""
        baseline = ExperimentResult(
            name="test_baseline", mode="baseline",
            horizons=[5, 20], top_k=20, exposure_names=[],
            ruleset_id="test_rs",
        )
        baseline.mean_ic = {5: 0.05, 20: 0.10}
        baseline.mean_gross = {5: 0.01, 20: 0.03}
        baseline.n_dates = 3

        report = generate_comparison_report(baseline, None, [])
        assert "# Alpha Experiment: test" in report
        assert "Baseline IC" not in report  # no comparison columns
        assert "IC" in report
        assert "Baseline-only run" in report

    def test_comparison_report_has_all_sections(self):
        """Comparison report includes IC, returns, exposure shift, recommendation."""
        baseline = ExperimentResult(
            name="test_baseline", mode="baseline",
            horizons=[20], top_k=20, exposure_names=["beta"],
            ruleset_id="test_rs",
        )
        baseline.mean_ic = {20: 0.10}
        baseline.mean_gross = {20: 0.02}
        baseline.mean_exposure_topk = {"beta": 1.5}
        baseline.n_dates = 5
        baseline.date_metrics = [
            DateMetrics(date="2026-01-15", ics={20: 0.10}, gross_returns={20: 0.02},
                        exposure_means_topk={"beta": 1.5}),
        ]

        neutralized = ExperimentResult(
            name="test_neutralized", mode="neutralized",
            horizons=[20], top_k=20, exposure_names=["beta"],
            ruleset_id="test_rs",
        )
        neutralized.mean_ic = {20: 0.15}
        neutralized.mean_gross = {20: 0.025}
        neutralized.mean_exposure_topk = {"beta": 1.2}
        neutralized.mean_ols_r2 = 0.25
        neutralized.n_dates = 5
        neutralized.date_metrics = [
            DateMetrics(date="2026-01-15", ics={20: 0.15}, gross_returns={20: 0.025},
                        exposure_means_topk={"beta": 1.2}, ols_r2=0.25, ols_coefficients=[0.1, -0.05]),
        ]

        report = generate_comparison_report(baseline, neutralized, ["beta"])

        assert "## Information Coefficient" in report
        assert "Baseline IC" in report
        assert "Neutralized IC" in report
        assert "## Top-K Gross Returns" in report
        assert "## Top-K Exposure Shift" in report
        assert "## OLS Diagnostics" in report
        assert "## Recommendation" in report

    def test_promote_recommendation(self):
        """Large IC improvement → PROMOTE."""
        baseline = ExperimentResult(
            name="test_baseline", mode="baseline",
            horizons=[20], top_k=20, exposure_names=["beta"],
            ruleset_id="test_rs",
        )
        baseline.mean_ic = {20: 0.05}
        baseline.n_dates = 5
        baseline.date_metrics = []

        neutralized = ExperimentResult(
            name="test_neutralized", mode="neutralized",
            horizons=[20], top_k=20, exposure_names=["beta"],
            ruleset_id="test_rs",
        )
        neutralized.mean_ic = {20: 0.10}
        neutralized.n_dates = 5
        neutralized.date_metrics = []

        report = generate_comparison_report(baseline, neutralized, ["beta"])
        assert "PROMOTE" in report

    def test_reject_recommendation(self):
        """Large IC decrease → REJECT."""
        baseline = ExperimentResult(
            name="test_baseline", mode="baseline",
            horizons=[20], top_k=20, exposure_names=["beta"],
            ruleset_id="test_rs",
        )
        baseline.mean_ic = {20: 0.10}
        baseline.n_dates = 5
        baseline.date_metrics = []

        neutralized = ExperimentResult(
            name="test_neutralized", mode="neutralized",
            horizons=[20], top_k=20, exposure_names=["beta"],
            ruleset_id="test_rs",
        )
        neutralized.mean_ic = {20: 0.05}
        neutralized.n_dates = 5
        neutralized.date_metrics = []

        report = generate_comparison_report(baseline, neutralized, ["beta"])
        assert "REJECT" in report


# ---------------------------------------------------------------------------
# ExperimentResult aggregation tests
# ---------------------------------------------------------------------------

class TestExperimentResult:
    def test_aggregate_ics(self):
        result = ExperimentResult(
            name="test", mode="baseline",
            horizons=[5, 20], top_k=20, exposure_names=[],
        )
        result.date_metrics = [
            DateMetrics(date="2026-01-15", ics={5: 0.10, 20: 0.20}),
            DateMetrics(date="2026-01-16", ics={5: 0.20, 20: 0.30}),
        ]
        result.aggregate()
        assert result.mean_ic[5] == pytest.approx(0.15)
        assert result.mean_ic[20] == pytest.approx(0.25)

    def test_aggregate_skips(self):
        result = ExperimentResult(
            name="test", mode="baseline",
            horizons=[20], top_k=20, exposure_names=[],
        )
        result.date_metrics = [
            DateMetrics(date="2026-01-15", ics={20: 0.10}),
            DateMetrics(date="2026-01-16", skipped=True, skip_reason="TEST"),
        ]
        result.aggregate()
        assert result.n_dates == 1
        assert result.n_skipped == 1
        assert result.mean_ic[20] == pytest.approx(0.10)

    def test_to_dict(self):
        result = ExperimentResult(
            name="test", mode="baseline",
            horizons=[20], top_k=20, exposure_names=["beta"],
            ruleset_id="abc123",
        )
        result.mean_ic = {20: 0.1234}
        result.mean_gross = {20: 0.0567}
        result.date_metrics = []
        d = result.to_dict()
        assert d["name"] == "test"
        assert d["ruleset_id"] == "abc123"
        assert d["mean_ic"]["20"] == 0.1234


# ---------------------------------------------------------------------------
# round_opt test
# ---------------------------------------------------------------------------

class TestRoundOpt:
    def test_none(self):
        assert _round_opt(None, 4) is None

    def test_value(self):
        assert _round_opt(0.12345, 3) == 0.123
