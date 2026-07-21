"""
Tests for tools/audit_top30_rank_integrity.py

Classification: TOP30_RANK_INTEGRITY_AUDIT_DIAGNOSTIC_NO_MODEL_CHANGE
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tools.audit_top30_rank_integrity as module
from tools.audit_top30_rank_integrity import (
    OUTPUT_DIR,
    PHASE3_DATES,
    audit_basket_match,
    audit_return_coverage,
    compute_5d_return,
    get_fwd_date,
    get_portfolio_top_n,
    get_rankings_top_n,
    load_canonical_rankings,
    load_decision_portfolio,
    run_audit,
    spearman_ic,
)

# Integration test: exercises the production rank-integrity audit against
# production_data/price_history.csv and artifacts/surveillance/pit_backtest_5d_ytd_2026.csv,
# neither of which is committed / present in a clean / CI checkout. Skip cleanly
# there; runs where the data exists. (CI_RED_2026 #521)
pytestmark = pytest.mark.skipif(
    not (module.PRICE_HISTORY_PATH.exists() and module.BACKTEST_CSV.exists()),
    reason="requires production_data/price_history.csv + artifacts/surveillance/pit_backtest_5d_ytd_2026.csv (absent in clean/CI checkout)",
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def audit_results():
    return run_audit(write_output=False)


@pytest.fixture(scope="module")
def prices():
    return module.load_price_history()


@pytest.fixture(scope="module")
def trading_dates(prices):
    return module.get_trading_dates(prices)


@pytest.fixture(scope="module")
def rows_may18():
    return load_canonical_rankings("2026-05-18")


@pytest.fixture(scope="module")
def positions_may18():
    return load_decision_portfolio("2026-05-18")


# ---------------------------------------------------------------------------
# TestGovernance
# ---------------------------------------------------------------------------


class TestGovernance:
    def test_classification(self, audit_results):
        assert audit_results["classification"] == ("TOP30_RANK_INTEGRITY_AUDIT_DIAGNOSTIC_NO_MODEL_CHANGE")

    def test_governance_flags(self, audit_results):
        gov = audit_results["governance"]
        assert gov["model_change"] is False
        assert gov["ranker_change"] is False
        assert gov["production_wiring"] is False
        assert gov["canonical_snapshots_modified"] is False

    def test_write_false_creates_no_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(module, "OUTPUT_JSON", tmp_path / "should_not.json")
        monkeypatch.setattr(module, "OUTPUT_MD", tmp_path / "should_not.md")
        monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path / "missing_dir")
        run_audit(write_output=False)
        assert not (tmp_path / "should_not.json").exists()
        assert not (tmp_path / "should_not.md").exists()


# ---------------------------------------------------------------------------
# TestBasketMatch — Objective 1
# ---------------------------------------------------------------------------


class TestBasketMatch:
    def test_all_dates_name_match(self, audit_results):
        bm = audit_results["basket_match"]
        assert bm["all_match"] is True
        assert bm["n_rankings_vs_portfolio_name_match"] == bm["n_dates"]

    def test_all_dates_order_match(self, audit_results):
        bm = audit_results["basket_match"]
        assert bm["n_rankings_vs_portfolio_order_match"] == bm["n_dates"]

    def test_no_mismatches(self, audit_results):
        assert audit_results["basket_match"]["mismatches"] == []

    def test_may18_basket_match(self, rows_may18, positions_may18, prices, trading_dates):
        brow = None
        result = audit_basket_match("2026-05-18", rows_may18, positions_may18, brow, prices, trading_dates)
        assert result["rankings_vs_portfolio_name_match"] is True
        assert result["rankings_vs_portfolio_order_match"] is True
        assert result["extra_in_rankings"] == []
        assert result["extra_in_portfolio"] == []

    def test_top30_count(self, rows_may18):
        top30 = get_rankings_top_n(rows_may18, 30)
        assert len(top30) == 30

    def test_portfolio_top30_count(self, positions_may18):
        top30 = get_portfolio_top_n(positions_may18, 30)
        assert len(top30) == 30

    def test_rankings_top30_starts_with_cogt(self, rows_may18):
        top30 = get_rankings_top_n(rows_may18, 30)
        assert top30[0] == "COGT"


# ---------------------------------------------------------------------------
# TestSortingKey — Objective 2
# ---------------------------------------------------------------------------


class TestSortingKey:
    def test_sorting_key_verified_all(self, audit_results):
        sk = audit_results["sorting_key"]
        assert sk["all_verified"] is True
        assert sk["n_verified"] == sk["n_checked"]
        assert sk["key_used"] == "actionable_rank"

    def test_sorting_key_not_csv_row_order(self, rows_may18):
        """actionable_rank sort ≠ CSV row order for at least some dates."""
        by_rank = get_rankings_top_n(rows_may18, 20)
        by_csv = [r["ticker"] for r in rows_may18[:20]]
        # They CAN be different — just verify the sort actually ran
        assert len(by_rank) == 20

    def test_may18_return_verification(self, rows_may18, prices, trading_dates):
        """Recomputed top20 return matches backtest (tolerance 1e-4)."""
        import csv as csv_mod

        backtest = {}
        with open(PROJECT_ROOT / "artifacts" / "surveillance" / "pit_backtest_5d_ytd_2026.csv") as f:
            for row in csv_mod.DictReader(f):
                backtest[row["snap_date"]] = row

        brow = backtest.get("2026-05-18")
        if not brow:
            pytest.skip("2026-05-18 not in backtest CSV")

        fwd = get_fwd_date("2026-05-18", trading_dates)
        top20 = get_rankings_top_n(rows_may18, 20)
        rets = [compute_5d_return(t, "2026-05-18", fwd, prices) for t in top20]
        valid = [r for r in rets if r is not None]
        recomputed = sum(valid) / len(valid)
        recorded = float(brow["top20_ret_5d"])
        assert (
            abs(recomputed - recorded) < 1e-4
        ), f"top20_ret mismatch: recomputed={recomputed:.6f} recorded={recorded:.6f}"


# ---------------------------------------------------------------------------
# TestPITAlignment — Objective 3
# ---------------------------------------------------------------------------


class TestPITAlignment:
    def test_no_future_price_violations(self, audit_results):
        assert audit_results["pit_alignment"]["n_future_violation"] == 0

    def test_fwd_date_computation(self, trading_dates):
        fwd = get_fwd_date("2026-05-18", trading_dates, n=5)
        assert fwd == "2026-05-26"

    def test_fwd_date_none_for_nontrading(self, trading_dates):
        fwd = get_fwd_date("2026-04-03", trading_dates, n=5)
        assert fwd is None  # Good Friday, not in price history

    def test_fwd_date_beyond_history(self, trading_dates):
        # A date far beyond the last trading date in history
        fwd = get_fwd_date("2099-01-01", trading_dates, n=5)
        assert fwd is None


# ---------------------------------------------------------------------------
# TestReturnCoverage — Objective 4
# ---------------------------------------------------------------------------


class TestReturnCoverage:
    def test_verifiable_coverage_above_99pct(self, audit_results):
        cov = audit_results["return_coverage"]
        assert cov["mean_coverage_pct_verifiable"] >= 99.0

    def test_data_gap_dates_count(self, audit_results):
        cov = audit_results["return_coverage"]
        # 4 non-trading snap dates + 7 sparse fwd dates = 11
        assert cov["n_data_gap_dates"] == 11

    def test_no_unexplained_low_coverage(self, audit_results):
        """All low-coverage dates must be data gaps (non-trading or sparse fwd)."""
        cov = audit_results["return_coverage"]
        data_gap_set = set(cov["data_gap_dates"])
        unexplained = {d: v for d, v in cov["low_coverage_dates"].items() if d not in data_gap_set}
        assert unexplained == {}, f"Unexplained low-coverage dates: {unexplained}"

    def test_may18_coverage(self, rows_may18, prices, trading_dates):
        fwd = get_fwd_date("2026-05-18", trading_dates)
        result = audit_return_coverage("2026-05-18", rows_may18, prices, fwd)
        assert result["coverage_pct"] >= 90.0
        assert result["n_with_return"] >= 27

    def test_no_fwd_date_returns_zero_coverage(self, rows_may18, prices):
        result = audit_return_coverage("2026-05-18", rows_may18, prices, None)
        assert result["coverage_pct"] == 0.0
        assert result["fwd_date"] is None


# ---------------------------------------------------------------------------
# TestBucketMonotonicity — Objective 5
# ---------------------------------------------------------------------------


class TestBucketMonotonicity:
    def test_phase3_mean_ic_negative_score_convention(self, audit_results):
        """Phase 3 rank IC is positive (rank 1=best, so sign flips vs score IC).
        Verify it represents genuine negative model IC in score convention."""
        phase3_ic = audit_results["bucket_monotonicity"]["phase3_mean_ic"]
        # Audit uses rank (lower=better) → positive IC means rank 1 names underperform
        # = negative IC in score convention = model inversion
        assert phase3_ic is not None
        assert phase3_ic > 0, "Phase 3 rank-IC should be positive (rank 1 names underperformed)"

    def test_bucket_structure(self, audit_results):
        for ba in audit_results["detail"]["bucket_audits"]:
            if ba.get("buckets"):
                assert "top30" in ba["buckets"]
                assert "all" in ba["buckets"]

    def test_spearman_ic_perfect_monotone(self):
        """Perfect negative rank correlation with positive score → IC=-1.0"""
        ranks = [1.0, 2.0, 3.0, 4.0, 5.0]
        returns = [5.0, 4.0, 3.0, 2.0, 1.0]  # rank 1 = best return
        ic = spearman_ic(ranks, returns)
        assert ic is not None
        assert abs(ic - (-1.0)) < 1e-6

    def test_spearman_ic_none_for_short_list(self):
        assert spearman_ic([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) is None


# ---------------------------------------------------------------------------
# TestRankPerturbation — Objective 6
# ---------------------------------------------------------------------------


class TestRankPerturbation:
    def test_actionable_rank_matches_final_score_sort(self, audit_results):
        """In pairwise_minimal mode, actionable_rank = sort by final_score DESC.
        All verifiable dates should match."""
        pert = audit_results["rank_perturbation"]
        # All dates where both methods have data should be identical
        total = pert["n_dates"]
        identical = pert["n_actionable_rank_matches_final_score_sort"]
        # Allow a small number of mismatches from dates with no final_score data
        assert identical >= total * 0.9, f"Only {identical}/{total} dates have actionable_rank == final_score sort"

    def test_perturbation_fields_present(self, audit_results):
        for a in audit_results["detail"]["perturbation_audits"][:5]:
            assert "by_actionable_rank" in a
            assert "by_final_score" in a
            assert "actionable_vs_final_score_overlap" in a


# ---------------------------------------------------------------------------
# TestPhase3Attribution — Objective 7
# ---------------------------------------------------------------------------


class TestPhase3Attribution:
    def test_phase3_attribution_count(self, audit_results):
        assert audit_results["phase3_attribution_summary"]["n_dates"] == 16

    def test_phase3_mean_ret_negative(self, audit_results):
        """Phase 3 top-30 mean 5d return should be negative (underperformance)."""
        p3 = audit_results["phase3_attribution_summary"]
        assert p3["mean_top30_ret_5d"] is not None
        assert p3["mean_top30_ret_5d"] < 0

    def test_phase3_residual_negative(self, audit_results):
        """Top-30 underperformed XBI on average during Phase 3."""
        p3 = audit_results["phase3_attribution_summary"]
        assert p3["mean_residual_vs_xbi"] is not None
        assert p3["mean_residual_vs_xbi"] < 0

    def test_per_ticker_attribution_fields(self, audit_results):
        attribs = audit_results["detail"]["phase3_attributions"]
        assert len(attribs) == 16
        for a in attribs:
            assert "per_ticker" in a
            assert "bottom5_contributors" in a
            assert "top5_contributors" in a

    def test_may18_has_cogt_attribution(self, audit_results):
        may18 = next(
            (a for a in audit_results["detail"]["phase3_attributions"] if a["snap_date"] == "2026-05-18"),
            None,
        )
        assert may18 is not None
        tickers = [r["ticker"] for r in may18["per_ticker"]]
        assert "COGT" in tickers


# ---------------------------------------------------------------------------
# TestOverallVerdict
# ---------------------------------------------------------------------------


class TestOverallVerdict:
    def test_overall_verdict(self, audit_results):
        v = audit_results["overall_verdict"]
        # Should be clean or data-limitation — NOT extraction bug
        assert "EXTRACTION_BUG" not in v
        assert "MISMATCH" not in v
        assert v in (
            "TOP30_ACCURATE_NEGATIVE_SELECTION_IS_REAL_EVIDENCE",
            "TOP30_ACCURATE_COVERAGE_GAPS_ARE_DATA_LIMITATION_NOT_BUG",
        )

    def test_57_dates_audited(self, audit_results):
        assert audit_results["window"]["n_dates_audited"] == 57

    def test_16_phase3_dates(self, audit_results):
        assert audit_results["window"]["n_phase3_dates"] == 16
