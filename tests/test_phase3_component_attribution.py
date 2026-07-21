"""
Tests for tools/analyze_phase3_component_attribution.py

Classification: PHASE3_COMPONENT_ATTRIBUTION_DIAGNOSTIC_NO_MODEL_CHANGE
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tools.analyze_phase3_component_attribution as module
from tools.analyze_phase3_component_attribution import (
    CLASSIFICATION,
    LOSERS,
    PHASE3_DATES,
    TARGET_NAMES,
    WINNERS,
    aggregate_ticker_profile,
    bear_sensitivity_section,
    classify_failure_mode,
    compute_5d_return,
    compute_contributions,
    counterfactual_rank,
    get_fwd_date,
    run_attribution,
)

# Integration test: exercises the production attribution tool against
# production_data/price_history.csv, which is not committed and is absent in a
# clean / CI checkout. Skip cleanly there; runs where the data exists. (CI_RED_2026 #521)
pytestmark = pytest.mark.skipif(
    not module.PRICE_HISTORY_CSV.exists(),
    reason="requires production_data/price_history.csv (absent in clean/CI checkout)",
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def attribution_results():
    return run_attribution(write_output=False)


@pytest.fixture(scope="module")
def profiles(attribution_results):
    return {p["ticker"]: p for p in attribution_results["per_ticker_attribution"]}


@pytest.fixture(scope="module")
def model():
    return module.load_ranker_v2_model()


@pytest.fixture(scope="module")
def prices():
    return module.load_price_history()


@pytest.fixture(scope="module")
def trading_dates(prices):
    return module.get_trading_dates(prices)


# ---------------------------------------------------------------------------
# TestGovernance
# ---------------------------------------------------------------------------


class TestGovernance:
    def test_classification(self, attribution_results):
        assert attribution_results["classification"] == CLASSIFICATION

    def test_schema(self, attribution_results):
        assert attribution_results["schema"] == "phase3_component_attribution_v1"

    def test_governance_flags_all_false(self, attribution_results):
        gov = attribution_results["governance"]
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

    def test_write_false_creates_no_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(module, "OUTPUT_JSON", tmp_path / "nope.json")
        monkeypatch.setattr(module, "OUTPUT_MD", tmp_path / "nope.md")
        monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path / "nodir")
        run_attribution(write_output=False)
        assert not (tmp_path / "nope.json").exists()
        assert not (tmp_path / "nodir").exists()


# ---------------------------------------------------------------------------
# TestTargetCoverage
# ---------------------------------------------------------------------------


class TestTargetCoverage:
    def test_all_8_targets_present(self, profiles):
        for ticker in TARGET_NAMES:
            assert ticker in profiles, f"{ticker} missing from attribution output"

    def test_losers_all_have_appearances(self, profiles):
        for ticker in LOSERS:
            assert profiles[ticker].get("n_appearances", 0) > 0, f"{ticker} has no appearances in Phase 3"

    def test_winners_all_have_appearances(self, profiles):
        for ticker in WINNERS:
            assert profiles[ticker].get("n_appearances", 0) > 0, f"{ticker} has no appearances in Phase 3"

    def test_window_covers_16_dates(self, attribution_results):
        assert attribution_results["window"]["n_dates"] == 16

    def test_losers_in_window(self, attribution_results):
        assert attribution_results["window"]["losers"] == LOSERS

    def test_winners_in_window(self, attribution_results):
        assert attribution_results["window"]["winners"] == WINNERS

    def test_celc_in_all_16_dates(self, profiles):
        """CELC appears in all 16 Phase 3 dates based on ranking data."""
        assert profiles["CELC"]["n_appearances"] == 16

    def test_tngx_in_all_16_dates(self, profiles):
        assert profiles["TNGX"]["n_appearances"] == 16


# ---------------------------------------------------------------------------
# TestLoserVsWinnerComparison
# ---------------------------------------------------------------------------


class TestLoserVsWinnerComparison:
    def test_losers_negative_mean_return(self, profiles):
        for ticker in LOSERS:
            p = profiles[ticker]
            ret = p.get("mean_5d_ret")
            assert ret is not None, f"{ticker} has no mean_5d_ret"
            assert ret < 0, f"{ticker} loser has positive mean_5d_ret={ret:.4f}"

    def test_winners_better_than_losers_aggregate(self, attribution_results):
        comp = attribution_results["loser_vs_winner_comparison"]
        lret = comp["loser_mean_5d_ret"]
        wret = comp["winner_mean_5d_ret"]
        assert lret is not None and wret is not None
        assert wret > lret, f"Winner mean ret ({wret:.4f}) not > loser mean ret ({lret:.4f})"

    def test_tngx_only_name_with_positive_fi_z(self, profiles):
        """TNGX is financially healthy (above cohort mean); all others below."""
        tngx_fi_z = profiles["TNGX"]["mean_fi_z"]
        assert tngx_fi_z is not None
        assert tngx_fi_z > 0, f"TNGX fi_z={tngx_fi_z:.3f} should be positive"

    def test_tngx_highest_ci_z_among_targets(self, profiles):
        all_ci = {t: profiles[t]["mean_ci_z"] for t in TARGET_NAMES if profiles[t].get("mean_ci_z") is not None}
        max_ticker = max(all_ci, key=all_ci.get)
        assert max_ticker == "TNGX", f"Expected TNGX to have highest ci_z; got {max_ticker}={all_ci[max_ticker]:.3f}"

    def test_syre_lowest_fi_z(self, profiles):
        """SYRE is the most financially stressed target — largest negative fi_z."""
        all_fi = {t: profiles[t]["mean_fi_z"] for t in TARGET_NAMES if profiles[t].get("mean_fi_z") is not None}
        min_ticker = min(all_fi, key=all_fi.get)
        assert min_ticker == "SYRE", f"Expected SYRE to have lowest fi_z; got {min_ticker}={all_fi[min_ticker]:.3f}"

    def test_drug_fi_z_more_negative_than_minus_one(self, profiles):
        """DRUG's financial stress (fi_z ≈ −1.4) should be well below −1.0."""
        drug_fi_z = profiles["DRUG"]["mean_fi_z"]
        assert drug_fi_z is not None
        assert drug_fi_z < -1.0, f"DRUG fi_z={drug_fi_z:.3f} expected < −1.0"


# ---------------------------------------------------------------------------
# TestPrimaryDriverClassification
# ---------------------------------------------------------------------------


class TestPrimaryDriverClassification:
    def test_drug_primary_driver_financial(self, profiles):
        """DRUG: dominated by financial_stress (fi_z ≈ −1.4, fi_contrib >> ci_contrib)."""
        assert profiles["DRUG"]["primary_driver"] == "financial_stress"

    def test_tngx_primary_driver_coinvest(self, profiles):
        """TNGX: ci_z ≈ +1.7 (highest), fi_z near zero → coinvest_signal dominant."""
        assert profiles["TNGX"]["primary_driver"] == "coinvest_signal"

    def test_financial_contribution_dominates_for_most_losers(self, profiles):
        """At least 3 of 5 losers should have financial_stress as primary driver."""
        n_fi = sum(1 for t in LOSERS if profiles[t].get("primary_driver") == "financial_stress")
        assert n_fi >= 3, f"Only {n_fi} losers have financial_stress as primary driver"


# ---------------------------------------------------------------------------
# TestCounterfactualRank
# ---------------------------------------------------------------------------


class TestCounterfactualRank:
    def test_drug_significantly_demoted_without_financial(self, profiles):
        """Zeroing DRUG's financial contribution should raise its rank by > 10 positions."""
        actual_rank = profiles["DRUG"].get("mean_actionable_rank")
        cf = profiles["DRUG"].get("mean_cf_rank_zero_fi")
        assert actual_rank is not None and cf is not None
        rank_increase = cf - actual_rank
        assert rank_increase > 10, (
            f"DRUG rank only moved {rank_increase:.1f} positions when fi zeroed "
            f"(actual={actual_rank:.0f}, cf={cf:.0f}); expected > 10"
        )

    def test_tngx_stays_in_top30_without_financial(self, profiles):
        """TNGX's rank is driven by coinvest; zeroing financial should keep it in top-30."""
        cf = profiles["TNGX"].get("mean_cf_rank_zero_fi")
        assert cf is not None
        assert cf <= 30, f"TNGX CF rank with fi=0 is {cf:.0f}; should still be in top-30"

    def test_rank_drop_positive_for_drug(self, profiles):
        """DRUG rank drop if primary zeroed should be positive (demoted)."""
        rd = profiles["DRUG"].get("rank_drop_if_primary_zeroed")
        assert rd is not None
        assert rd > 0, f"DRUG rank_drop_if_primary_zeroed={rd:.1f}; expected > 0"

    def test_counterfactual_rank_function_unit(self, model):
        """Zeroing both features leaves only the bias — score ≈ sigmoid(bias)."""
        base_score = 1.0 / (1.0 + math.exp(-model.bias))
        # Cohort of 60 equal-score names → rank 1 for target with same score
        equal_scores = [("T" + str(i), base_score) for i in range(60)]
        # With zero_ci and zero_fi, target score = sigmoid(bias) = same as all others
        # Rank = 1 (all others tied, none strictly higher)
        cf = counterfactual_rank(
            "DRUG",
            1.0,
            30.0,
            {"ci_mean": 1.0, "ci_std": 1.0, "fi_mean": 30.0, "fi_std": 1.0},
            equal_scores,
            model,
            zero_ci=True,
            zero_fi=True,
        )
        assert cf == 1  # no one scores strictly higher

    def test_zero_fi_lowers_score_for_stressed_name(self, model):
        """For a financially stressed name (fi_z < 0), zeroing fi should lower score."""
        stats = {"ci_mean": 1.0, "ci_std": 0.5, "fi_mean": 40.0, "fi_std": 20.0}
        # fi_raw=5 → fi_z=(5-40)/20 = -1.75 → fi_contrib = -0.053 * (-1.75) = +0.093 (positive)
        # Zeroing fi removes this boost → lower score
        c_actual = compute_contributions(1.0, 5.0, stats, model)
        c_zero_fi = compute_contributions(1.0, 40.0, stats, model)  # fi at mean → fi_z=0
        assert (
            c_actual["final_score"] > c_zero_fi["final_score"]
        ), f"Expected actual ({c_actual['final_score']:.4f}) > zero_fi ({c_zero_fi['final_score']:.4f})"


# ---------------------------------------------------------------------------
# TestFailureModeClassification
# ---------------------------------------------------------------------------


class TestFailureModeClassification:
    def test_drug_financing_under_penalized(self, profiles):
        assert profiles["DRUG"]["failure_mode"] == "FINANCING_UNDER_PENALIZED"

    def test_abvx_ees_veto_failed(self, profiles):
        assert profiles["ABVX"]["failure_mode"] == "EES_VETO_FAILED"

    def test_celc_ees_veto_failed(self, profiles):
        assert profiles["CELC"]["failure_mode"] == "EES_VETO_FAILED"

    def test_winners_are_winner_offset(self, profiles):
        for ticker in WINNERS:
            assert (
                profiles[ticker]["failure_mode"] == "WINNER_OFFSET"
            ), f"{ticker} expected WINNER_OFFSET, got {profiles[ticker]['failure_mode']}"

    def test_failure_mode_summary_has_drug_and_abvx(self, attribution_results):
        fm = attribution_results["failure_mode_summary"]
        found_drug = any("DRUG" in v for v in fm.values())
        found_abvx = any("ABVX" in v for v in fm.values())
        assert found_drug, "DRUG not in any failure_mode_summary entry"
        assert found_abvx, "ABVX not in any failure_mode_summary entry"

    def test_classify_failure_mode_winner(self, profiles):
        """Winners return WINNER_OFFSET regardless of signals."""
        winner_profile = profiles["TNGX"]
        mode, evidence = classify_failure_mode(winner_profile)
        assert mode == "WINNER_OFFSET"

    def test_classify_failure_mode_financing_low_fi(self):
        """Profile with very negative fi_z and negative return → FINANCING_UNDER_PENALIZED."""
        synthetic = {
            "ticker": "X",
            "role": "loser",
            "n_appearances": 10,
            "mean_ci_z": -0.1,
            "mean_fi_z": -1.5,
            "mean_ci_contrib": -0.002,
            "mean_fi_contrib": 0.080,  # fi dominant, positive (promotes stressed name)
            "mean_ees_v3_score": 0.1,  # near-zero EES, no veto signal
            "mean_5d_ret": -0.12,
            "mean_actionable_rank": 10.0,
            "rank_drop_if_primary_zeroed": 22.0,
        }
        mode, _ = classify_failure_mode(synthetic)
        assert mode == "FINANCING_UNDER_PENALIZED"

    def test_classify_failure_mode_ees_veto(self):
        """Profile with strong negative ees_v3 and negative return → EES_VETO_FAILED."""
        synthetic = {
            "ticker": "Y",
            "role": "loser",
            "n_appearances": 10,
            "mean_ci_z": 0.5,
            "mean_fi_z": -0.5,
            "mean_ci_contrib": 0.010,
            "mean_fi_contrib": 0.027,
            "mean_ees_v3_score": -1.2,  # strong negative EES
            "mean_5d_ret": -0.08,
            "mean_actionable_rank": 15.0,
            "rank_drop_if_primary_zeroed": 5.0,
        }
        mode, _ = classify_failure_mode(synthetic)
        assert mode == "EES_VETO_FAILED"


# ---------------------------------------------------------------------------
# TestBearSensitivity
# ---------------------------------------------------------------------------


class TestBearSensitivity:
    def test_bear_section_regime_invariant(self, attribution_results):
        assert attribution_results["bear_sensitivity"]["ranker_v2_regime_invariant"] is True

    def test_bear_section_no_rank_change(self, attribution_results):
        assert attribution_results["bear_sensitivity"]["bear_rank_change"] == "NONE"

    def test_bear_section_evidence_references_replay(self, attribution_results):
        evidence = attribution_results["bear_sensitivity"]["evidence"]
        assert "PHASE3_CORRECTED_REGIME_RANKING_REPLAY" in evidence

    def test_bear_section_function_standalone(self):
        section = bear_sensitivity_section()
        assert section["ranker_v2_regime_invariant"] is True
        assert "coinvest_score_z" in section["features_precomputed_before_regime"]
        assert "financial_score" in section["features_precomputed_before_regime"]


# ---------------------------------------------------------------------------
# TestStructuralFinding
# ---------------------------------------------------------------------------


class TestStructuralFinding:
    def test_dominant_feature_is_financial(self, attribution_results):
        sf = attribution_results["structural_finding"]
        assert sf["dominant_feature"] == "financial_score"

    def test_financial_weight_negative(self, attribution_results):
        assert attribution_results["structural_finding"]["weight"] < 0

    def test_ees_v3_not_in_ranker_v2(self, attribution_results):
        assert attribution_results["structural_finding"]["ees_v3_not_in_ranker_v2"] is True

    def test_losers_more_financially_stressed_than_tngx(self, attribution_results):
        sf = attribution_results["structural_finding"]
        lfi = sf.get("mean_fi_z_losers")
        # TNGX (winner) has fi_z > 0; losers have fi_z < 0 on average
        assert lfi is not None
        assert lfi < 0, f"Expected loser mean_fi_z < 0, got {lfi:.3f}"


# ---------------------------------------------------------------------------
# TestContributionComputation
# ---------------------------------------------------------------------------


class TestContributionComputation:
    def test_negative_fi_z_gives_positive_fi_contrib(self, model):
        """Negative fi_z × negative weight → positive contribution."""
        stats = {"ci_mean": 1.0, "ci_std": 0.5, "fi_mean": 40.0, "fi_std": 20.0}
        result = compute_contributions(1.0, 10.0, stats, model)
        assert result["fi_z"] < 0, "fi_z should be negative for fi_raw < fi_mean"
        assert result["fi_contrib"] > 0, "fi_contrib should be positive"

    def test_positive_ci_z_gives_positive_ci_contrib(self, model):
        stats = {"ci_mean": 1.0, "ci_std": 0.5, "fi_mean": 40.0, "fi_std": 20.0}
        result = compute_contributions(2.0, 40.0, stats, model)
        assert result["ci_z"] > 0
        assert result["ci_contrib"] > 0

    def test_final_score_is_probability(self, model):
        stats = {"ci_mean": 1.0, "ci_std": 0.5, "fi_mean": 40.0, "fi_std": 20.0}
        result = compute_contributions(1.0, 30.0, stats, model)
        assert 0.0 < result["final_score"] < 1.0

    def test_final_score_monotone_in_ci(self, model):
        """Increasing coinvest_score_z increases final_score."""
        stats = {"ci_mean": 1.0, "ci_std": 0.5, "fi_mean": 40.0, "fi_std": 20.0}
        low = compute_contributions(0.5, 30.0, stats, model)["final_score"]
        high = compute_contributions(2.5, 30.0, stats, model)["final_score"]
        assert high > low

    def test_final_score_monotone_decreasing_in_fi(self, model):
        """Increasing financial_score DECREASES final_score (negative weight)."""
        stats = {"ci_mean": 1.0, "ci_std": 0.5, "fi_mean": 40.0, "fi_std": 20.0}
        low_fi = compute_contributions(1.0, 10.0, stats, model)["final_score"]
        high_fi = compute_contributions(1.0, 70.0, stats, model)["final_score"]
        assert low_fi > high_fi, "More financial stress should give higher score"

    def test_contributions_sum_to_linear_minus_bias(self, model):
        """ci_contrib + fi_contrib + bias = linear (before sigmoid)."""
        stats = {"ci_mean": 1.0, "ci_std": 0.5, "fi_mean": 40.0, "fi_std": 20.0}
        result = compute_contributions(1.5, 25.0, stats, model)
        expected_linear = result["ci_contrib"] + result["fi_contrib"] + model.bias
        assert abs(expected_linear - result["linear"]) < 1e-10


# ---------------------------------------------------------------------------
# TestDataHelpers
# ---------------------------------------------------------------------------


class TestDataHelpers:
    def test_fwd_date_may18_is_may26(self, trading_dates):
        fwd = get_fwd_date("2026-05-18", trading_dates)
        assert fwd == "2026-05-26"

    def test_fwd_date_none_for_nontrading(self, trading_dates):
        fwd = get_fwd_date("2026-04-03", trading_dates, n=5)
        assert fwd is None

    def test_compute_5d_return_none_without_prices(self, prices):
        ret = compute_5d_return("XXXX", "2026-05-18", "2026-05-26", prices)
        assert ret is None

    def test_compute_5d_return_xbi_may18(self, prices):
        ret = compute_5d_return("XBI", "2026-05-18", "2026-05-26", prices)
        assert ret is not None
        assert isinstance(ret, float)

    def test_json_serializable(self, attribution_results):
        json.dumps(attribution_results, default=str)
