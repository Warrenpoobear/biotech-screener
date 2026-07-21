"""
Tests for tools/run_stressed_optionality_shadow_test.py

Classification: STRESSED_OPTIONALITY_CONFIRMATION_SHADOW_TEST_NO_MODEL_CHANGE
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tools.run_stressed_optionality_shadow_test as module
from tools.run_stressed_optionality_shadow_test import (
    CLASSIFICATION,
    EES_SUPPRESS_THRESHOLD,
    EXPECTED_PASS_THROUGH,
    EXPECTED_SUPPRESSED,
    EXTREME_STRESS_FI_Z_THRESHOLD,
    MOMENTUM_CONFIRM_THRESHOLD,
    apply_shadow_rule,
    build_shadow_basket,
    run_shadow_test,
)

# Integration test: exercises the production stressed-optionality shadow test
# against production_data/price_history.csv and artifacts/surveillance/pit_backtest_5d_ytd_2026.csv,
# neither committed / present in a clean / CI checkout. Skip cleanly there;
# runs where the data exists. (CI_RED_2026 #521)
pytestmark = pytest.mark.skipif(
    not (module.PRICE_HISTORY_CSV.exists() and module.BACKTEST_CSV.exists()),
    reason="requires production_data/price_history.csv + artifacts/surveillance/pit_backtest_5d_ytd_2026.csv (absent in clean/CI checkout)",
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def results():
    return run_shadow_test(write_output=False)


@pytest.fixture(scope="module")
def model():
    return module.load_ranker_v2_model()


def _stats():
    """Synthetic cohort stats for unit tests (mean=1.0 for coinvest, 40.0 for financial)."""
    return {"ci_mean": 1.0, "ci_std": 0.5, "fi_mean": 40.0, "fi_std": 20.0}


def _row(**kwargs):
    defaults = {
        "ticker": "TEST",
        "coinvest_score_z": "1.0",
        "financial_score": "40.0",
        "ees_v3_score": "",
        "momentum_score": "70.0",
        "clinical_score": "50.0",
        "actionable_rank": "5",
    }
    defaults.update(kwargs)
    return defaults


# ---------------------------------------------------------------------------
# TestGovernance
# ---------------------------------------------------------------------------


class TestGovernance:
    def test_classification(self, results):
        assert results["classification"] == CLASSIFICATION

    def test_schema(self, results):
        assert results["schema"] == "stressed_optionality_shadow_test_v1"

    def test_governance_flags(self, results):
        gov = results["governance"]
        for flag in [
            "model_change",
            "ranker_change",
            "selector_change",
            "sizing_change",
            "regime_change",
            "production_wiring",
            "canonical_snapshots_modified",
            "cron",
        ]:
            assert gov[flag] is False, f"governance.{flag} should be False"

    def test_rule_parameters_match_constants(self, results):
        rp = results["rule_parameters"]
        assert rp["ees_suppress_threshold"] == EES_SUPPRESS_THRESHOLD
        assert rp["extreme_stress_fi_z_threshold"] == EXTREME_STRESS_FI_Z_THRESHOLD
        assert rp["momentum_confirm_threshold"] == MOMENTUM_CONFIRM_THRESHOLD

    def test_write_false_no_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(module, "OUTPUT_JSON", tmp_path / "nope.json")
        monkeypatch.setattr(module, "OUTPUT_MD", tmp_path / "nope.md")
        monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path / "nodir")
        run_shadow_test(write_output=False)
        assert not (tmp_path / "nope.json").exists()


# ---------------------------------------------------------------------------
# TestVerdict
# ---------------------------------------------------------------------------


class TestVerdict:
    def test_verdict_is_improvement(self, results):
        assert results["verdict"] == "SHADOW_GUARDRAIL_IMPROVES_PHASE3_WITHOUT_WINNER_DEGRADATION"

    def test_phase3_shadow_ret_greater_than_actual(self, results):
        p3 = results["phase3"]
        actual = p3["mean_actual_basket_ret"]
        shadow = p3["mean_shadow_basket_ret"]
        assert actual is not None and shadow is not None
        assert shadow > actual, f"Phase 3 shadow ret ({shadow:.4f}) should be > actual ({actual:.4f})"

    def test_phase3_improvement_positive(self, results):
        imp = results["phase3"]["mean_basket_improvement"]
        assert imp is not None
        assert imp > 0, f"Phase 3 improvement should be positive; got {imp:.4f}"

    def test_non_phase3_not_degraded(self, results):
        """Shadow rule should not hurt non-Phase-3 periods."""
        imp = results["non_phase3"]["mean_basket_improvement"]
        assert imp is not None
        assert imp >= 0, f"Non-Phase-3 improvement should be >= 0 (not degraded); got {imp:.4f}"


# ---------------------------------------------------------------------------
# TestPassThroughPreservation
# ---------------------------------------------------------------------------


class TestPassThroughPreservation:
    def test_no_phase3_violations(self, results):
        assert results["phase3"]["n_pass_through_violations"] == 0

    def test_no_non_phase3_violations(self, results):
        assert results["non_phase3"]["n_pass_through_violations"] == 0

    def test_tngx_not_suppressed_phase3(self, results):
        t = results["target_phase3_summary"].get("TNGX", {})
        assert t.get("n_suppressed", 0) == 0, f"TNGX should not be suppressed; got {t}"

    def test_alks_not_suppressed_phase3(self, results):
        t = results["target_phase3_summary"].get("ALKS", {})
        assert t.get("n_suppressed", 0) == 0, f"ALKS should not be suppressed; got {t}"

    def test_syre_not_suppressed_phase3(self, results):
        t = results["target_phase3_summary"].get("SYRE", {})
        assert t.get("n_suppressed", 0) == 0, f"SYRE should not be suppressed; got {t}"


# ---------------------------------------------------------------------------
# TestLoserSuppression
# ---------------------------------------------------------------------------


class TestLoserSuppression:
    def test_drug_fully_suppressed_phase3(self, results):
        """DRUG: extreme financial stress + low momentum → suppressed every appearance."""
        t = results["target_phase3_summary"].get("DRUG", {})
        assert t.get("n_suppressed", 0) == t.get(
            "n_appearances", -1
        ), f"DRUG should be suppressed every appearance; got {t['n_suppressed']}/{t['n_appearances']}"

    def test_celc_partially_suppressed_phase3(self, results):
        """CELC: EES-flagged on most dates; partially suppressed is correct."""
        t = results["target_phase3_summary"].get("CELC", {})
        assert t.get("n_suppressed", 0) > 0, "CELC should be suppressed on at least one date"

    def test_abvx_partially_suppressed_phase3(self, results):
        t = results["target_phase3_summary"].get("ABVX", {})
        assert t.get("n_suppressed", 0) > 0, "ABVX should be suppressed on at least one date"

    def test_phase3_dates_have_suppressions(self, results):
        p3 = results["phase3"]
        assert p3["n_dates_with_suppression"] > 0

    def test_16_phase3_dates(self, results):
        assert results["phase3"]["n_dates"] == 16


# ---------------------------------------------------------------------------
# TestApplyShadowRule
# ---------------------------------------------------------------------------


class TestApplyShadowRule:
    def test_coinvest_primary_always_eligible(self, model):
        """If coinvest drives rank, rule never fires — even with terrible EES."""
        stats = _stats()
        # ci_z = (2.5 - 1.0) / 0.5 = 3.0 → ci_contrib = 0.02 * 3 = 0.06
        # fi_z = (40 - 40) / 20 = 0   → fi_contrib = 0
        # ci dominates → ELIGIBLE regardless
        row = _row(coinvest_score_z="2.5", financial_score="40.0", ees_v3_score="-2.0", momentum_score="10.0")
        sr = apply_shadow_rule(row, stats, model)
        assert sr["shadow_status"] == "ELIGIBLE"
        assert not sr["is_financial_primary"]

    def test_path1_ees_flagged_suppression(self, model):
        """Negative EES + financial primary → SUPPRESSED_EES_FLAGGED."""
        stats = _stats()
        # fi_z = (10 - 40) / 20 = -1.5 → fi_contrib = -0.053 * -1.5 = +0.0797
        # ci_z = (1.0 - 1.0) / 0.5 = 0 → ci_contrib = 0
        # fi dominates + EES negative → Path 1
        row = _row(coinvest_score_z="1.0", financial_score="10.0", ees_v3_score="-1.0", momentum_score="70.0")
        sr = apply_shadow_rule(row, stats, model)
        assert sr["shadow_status"] == "SUPPRESSED"
        assert sr["suppression_type"] == "EES_FLAGGED"
        assert sr["rule_path"] == "path1_ees"

    def test_path1_ees_just_above_threshold_not_suppressed(self, model):
        """EES just above threshold → should NOT trigger Path 1."""
        stats = _stats()
        row = _row(coinvest_score_z="1.0", financial_score="10.0", ees_v3_score="-0.70", momentum_score="40.0")
        sr = apply_shadow_rule(row, stats, model)
        # EES = -0.70 > EES_SUPPRESS_THRESHOLD (-0.75) → Path 1 doesn't fire
        assert sr["suppression_type"] != "EES_FLAGGED"

    def test_path2_extreme_stress_low_momentum(self, model):
        """Extreme fi_z + EES neutral + low momentum → EXTREME_STRESS_UNCONFIRMED."""
        stats = _stats()
        # fi_z = (5 - 40) / 20 = -1.75 (below threshold -1.0)
        # ees = 0.1 (positive, passes ees_ok)
        # momentum = 40 < 60 (fails momentum_ok)
        # → suppressed
        row = _row(coinvest_score_z="1.0", financial_score="5.0", ees_v3_score="0.1", momentum_score="40.0")
        sr = apply_shadow_rule(row, stats, model)
        assert sr["shadow_status"] == "SUPPRESSED"
        assert sr["suppression_type"] == "EXTREME_STRESS_UNCONFIRMED"
        assert sr["rule_path"] == "path2_extreme_stress"

    def test_path2_extreme_stress_confirmed_passes(self, model):
        """Extreme fi_z + positive EES + high momentum → ELIGIBLE (SYRE-like)."""
        stats = _stats()
        # fi_z = (5 - 40) / 20 = -1.75 (below threshold -1.0)
        # ees = 1.4 > 0 (passes)
        # momentum = 90 >= 60 (passes)
        # → eligible
        row = _row(coinvest_score_z="1.0", financial_score="5.0", ees_v3_score="1.4", momentum_score="90.0")
        sr = apply_shadow_rule(row, stats, model)
        assert sr["shadow_status"] == "ELIGIBLE"

    def test_path2_extreme_stress_negative_ees_suppressed(self, model):
        """Extreme fi_z + negative EES (but not triggering Path 1) → suppressed."""
        stats = _stats()
        # fi_z = -1.75 (extreme)
        # ees = -0.3 (negative but > -0.75 so Path 1 doesn't fire)
        # ees_ok = False (ees present and <= 0) → NOT (ees_ok AND momentum_ok) → suppressed
        row = _row(coinvest_score_z="1.0", financial_score="5.0", ees_v3_score="-0.3", momentum_score="80.0")
        sr = apply_shadow_rule(row, stats, model)
        assert sr["shadow_status"] == "SUPPRESSED"
        assert sr["suppression_type"] == "EXTREME_STRESS_UNCONFIRMED"

    def test_path2_no_ees_data_high_momentum_eligible(self, model):
        """Extreme fi_z + missing EES + high momentum → ELIGIBLE (benefit of doubt)."""
        stats = _stats()
        row = _row(coinvest_score_z="1.0", financial_score="5.0", ees_v3_score="", momentum_score="75.0")
        sr = apply_shadow_rule(row, stats, model)
        assert sr["shadow_status"] == "ELIGIBLE"

    def test_path2_no_ees_data_low_momentum_suppressed(self, model):
        """Extreme fi_z + missing EES + low momentum → SUPPRESSED (stress unconfirmed)."""
        stats = _stats()
        row = _row(coinvest_score_z="1.0", financial_score="5.0", ees_v3_score="", momentum_score="40.0")
        sr = apply_shadow_rule(row, stats, model)
        assert sr["shadow_status"] == "SUPPRESSED"

    def test_moderate_stress_no_trigger(self, model):
        """Moderate fi_z (above extreme threshold) and EES not negative → ELIGIBLE."""
        stats = _stats()
        # fi_z = (25 - 40) / 20 = -0.75 (above -1.0 threshold)
        # EES = 0.0 (neutral, not < -0.75)
        row = _row(coinvest_score_z="1.0", financial_score="25.0", ees_v3_score="0.0", momentum_score="50.0")
        sr = apply_shadow_rule(row, stats, model)
        assert sr["shadow_status"] == "ELIGIBLE"

    def test_result_always_has_status(self, model):
        stats = _stats()
        row = _row()
        sr = apply_shadow_rule(row, stats, model)
        assert "shadow_status" in sr
        assert sr["shadow_status"] in ("ELIGIBLE", "SUPPRESSED")

    def test_result_has_contribution_fields(self, model):
        stats = _stats()
        row = _row()
        sr = apply_shadow_rule(row, stats, model)
        assert "ci_z" in sr
        assert "fi_z" in sr
        assert "ci_contrib" in sr
        assert "fi_contrib" in sr
        assert "is_financial_primary" in sr


# ---------------------------------------------------------------------------
# TestBuildShadowBasket
# ---------------------------------------------------------------------------


class TestBuildShadowBasket:
    def test_no_suppressions_returns_top30(self):
        rows = [{"ticker": f"T{i:02d}", "actionable_rank": str(i + 1)} for i in range(40)]
        shadow_by_ticker = {f"T{i:02d}": {"shadow_status": "ELIGIBLE"} for i in range(40)}
        basket = build_shadow_basket(rows, shadow_by_ticker, n=30)
        assert basket == [f"T{i:02d}" for i in range(30)]
        assert len(basket) == 30

    def test_suppressed_rank1_replaced_by_rank31(self):
        rows = [{"ticker": f"T{i:02d}", "actionable_rank": str(i + 1)} for i in range(35)]
        shadow = {f"T{i:02d}": {"shadow_status": "ELIGIBLE"} for i in range(35)}
        shadow["T00"] = {"shadow_status": "SUPPRESSED"}  # rank 1 suppressed
        basket = build_shadow_basket(rows, shadow, n=30)
        assert len(basket) == 30
        assert "T00" not in basket
        assert "T30" in basket  # rank 31 fills in

    def test_basket_always_eligible(self):
        rows = [{"ticker": f"T{i:02d}", "actionable_rank": str(i + 1)} for i in range(40)]
        shadow = {f"T{i:02d}": {"shadow_status": "SUPPRESSED" if i < 5 else "ELIGIBLE"} for i in range(40)}
        basket = build_shadow_basket(rows, shadow, n=30)
        for t in basket:
            assert shadow.get(t, {}).get("shadow_status") == "ELIGIBLE"

    def test_unknown_status_treated_as_eligible(self):
        """Tickers not in shadow_by_ticker default to ELIGIBLE."""
        rows = [{"ticker": f"T{i:02d}", "actionable_rank": str(i + 1)} for i in range(40)]
        shadow = {}  # no entries — all default to ELIGIBLE
        basket = build_shadow_basket(rows, shadow, n=30)
        assert len(basket) == 30


# ---------------------------------------------------------------------------
# TestWindowCoverage
# ---------------------------------------------------------------------------


class TestWindowCoverage:
    def test_16_phase3_dates(self, results):
        assert results["window"]["n_phase3_dates"] == 16

    def test_ytd_total_correct(self, results):
        w = results["window"]
        assert w["n_ytd_dates"] == w["n_phase3_dates"] + w["n_non_phase3_dates"]

    def test_non_phase3_dates_exist(self, results):
        assert results["window"]["n_non_phase3_dates"] > 0

    def test_json_serializable(self, results):
        summary = {k: v for k, v in results.items() if k != "detail"}
        json.dumps(summary, default=str)
