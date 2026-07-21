"""
Tests for tools/phase3_corrected_regime_replay.py

Classification: PHASE3_CORRECTED_REGIME_RANKING_REPLAY_DIAGNOSTIC_NO_MODEL_CHANGE
Governance:
    - canonical snapshots not modified
    - no model/ranker/selector/sizing/production changes
    - output only to artifacts/autopsy/phase3_corrected_regime_replay/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tools.phase3_corrected_regime_replay as module
from tools.phase3_corrected_regime_replay import (
    PHASE3_DATES,
    PHASE3_RECONSTRUCTED,
    SNAPSHOTS_DIR,
    compare_rankings,
    compute_corrected_ranker_v2_top30,
    get_original_top30,
    load_canonical_rankings,
    load_ranker_v2_model,
    run_replay,
)

# Integration test: exercises the production regime-replay against
# artifacts/surveillance/pit_backtest_5d_ytd_2026.csv, which is not committed
# and is absent in a clean / CI checkout. Skip cleanly there; runs where the
# data exists. (CI_RED_2026 #521)
pytestmark = pytest.mark.skipif(
    not module.BACKTEST_CSV.exists(),
    reason="requires artifacts/surveillance/pit_backtest_5d_ytd_2026.csv (absent in clean/CI checkout)",
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model():
    return load_ranker_v2_model()


@pytest.fixture(scope="module")
def canonical_rows_may18():
    return load_canonical_rankings("2026-05-18")


@pytest.fixture(scope="module")
def replay_results():
    """Full replay with write_output=False — used by multiple tests."""
    return run_replay(write_output=False)


def _make_minimal_rows(n: int = 70) -> list:
    """Build synthetic rows that pass filter_cohort (eligible=1, actionable_rank set)."""
    rows = []
    for i in range(n):
        rank = i + 1
        rows.append(
            {
                "ticker": f"T{i:03d}",
                "actionable_rank": str(rank),
                "eligible": "1",
                "coinvest_score_z": str(round(1.0 - i * 0.02, 4)),
                "financial_score": str(round(5.0 - i * 0.05, 4)),
                "ranker_v2_score": "",
                "final_score": "",
            }
        )
    return rows


# ---------------------------------------------------------------------------
# TestGovernance
# ---------------------------------------------------------------------------


class TestGovernance:
    def test_output_json_governance_flags(self, replay_results):
        gov = replay_results["governance"]
        assert gov["bypassed_freeze"] is False
        assert gov["canonical_snapshots_modified"] is False
        assert gov["model_change"] is False
        assert gov["ranker_change"] is False
        assert gov["selector_change"] is False
        assert gov["sizing_change"] is False
        assert gov["production_wiring"] is False

    def test_classification_field(self, replay_results):
        assert replay_results["classification"] == ("PHASE3_CORRECTED_REGIME_RANKING_REPLAY_DIAGNOSTIC_NO_MODEL_CHANGE")

    def test_schema_field(self, replay_results):
        assert replay_results["schema"] == "phase3_corrected_regime_replay_v1"

    def test_write_false_creates_no_files(self, tmp_path, monkeypatch):
        """run_replay(write_output=False) must not write to the output directory."""
        sentinel = tmp_path / "should_not_exist.json"
        monkeypatch.setattr(module, "OUTPUT_JSON", sentinel)
        monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path / "nonexistent_subdir")
        run_replay(write_output=False)
        assert not sentinel.exists()
        assert not (tmp_path / "nonexistent_subdir").exists()

    def test_canonical_snapshots_untouched(self, canonical_rows_may18):
        """Canonical snapshot file must not be modified by the replay."""
        path = SNAPSHOTS_DIR / "2026-05-18" / "rankings.csv"
        mtime_before = path.stat().st_mtime
        run_replay(write_output=False)
        mtime_after = path.stat().st_mtime
        assert mtime_before == mtime_after


# ---------------------------------------------------------------------------
# TestPhase3Dates
# ---------------------------------------------------------------------------


class TestPhase3Dates:
    def test_16_phase3_dates(self):
        assert len(PHASE3_DATES) == 16

    def test_dates_in_reconstructed(self):
        for d in PHASE3_DATES:
            assert d in PHASE3_RECONSTRUCTED, f"{d} missing from PHASE3_RECONSTRUCTED"

    def test_all_reconstructed_regime_bear(self):
        for d, info in PHASE3_RECONSTRUCTED.items():
            assert info["regime"] == "BEAR", f"{d} not BEAR: {info['regime']}"

    def test_all_xbi_underperforming(self):
        for d, info in PHASE3_RECONSTRUCTED.items():
            assert info["xbi_vs_spy_30d"] < -4.5, f"{d} xbi_vs_spy_30d={info['xbi_vs_spy_30d']} not BEAR-range"


# ---------------------------------------------------------------------------
# TestRankerV2Invariance — core result
# ---------------------------------------------------------------------------


class TestRankerV2Invariance:
    def test_all_16_dates_identical(self, replay_results):
        rv2 = replay_results["ranker_v2_comparison"]
        assert rv2["all_identical"] is True
        assert rv2["dates_identical"] == 16
        assert rv2["dates_checked"] == 16

    def test_verdict_is_identical(self, replay_results):
        assert replay_results["ranker_v2_comparison"]["verdict"] == "CORRECTED_REGIME_IDENTICAL_TOP30"

    def test_may18_top30_identical(self, canonical_rows_may18, model):
        orig = get_original_top30(canonical_rows_may18)
        corrected, _ = compute_corrected_ranker_v2_top30(canonical_rows_may18, model)
        assert set(orig) == set(corrected), "May 18 top-30 mismatch"
        assert orig == corrected, "May 18 rank order changed"

    def test_per_date_identical_flag(self, replay_results):
        for c in replay_results["ranking_comparison"]:
            assert c["identical"] is True, f"{c['snap_date']} not identical"
            assert c["overlap_count"] == 30, f"{c['snap_date']} overlap={c['overlap_count']}"
            assert c["rank_changes"] == [], f"{c['snap_date']} has rank changes"

    def test_synthetic_rows_idempotent(self, model):
        """Scoring the same rows twice returns identical results (regime-invariant)."""
        rows = _make_minimal_rows(70)
        top30_first, _ = compute_corrected_ranker_v2_top30(rows, model)
        top30_second, _ = compute_corrected_ranker_v2_top30(rows, model)
        assert top30_first == top30_second

    def test_compare_rankings_identity(self):
        top30 = [f"T{i:03d}" for i in range(30)]
        result = compare_rankings("2026-05-18", top30, top30)
        assert result["identical"] is True
        assert result["overlap_count"] == 30
        assert result["rank_changes"] == []


# ---------------------------------------------------------------------------
# TestArchitecturalFinding
# ---------------------------------------------------------------------------


class TestArchitecturalFinding:
    def test_features_regime_independent(self, replay_results):
        af = replay_results["architectural_finding"]
        assert af["features_regime_independent"] is True

    def test_production_sort_key_documented(self, replay_results):
        af = replay_results["architectural_finding"]
        assert "ranker_v2_score" in af["production_sort_key"]
        assert "final_score" in af["production_sort_key"]

    def test_features_list(self, replay_results):
        features = replay_results["architectural_finding"]["ranker_v2_features"]
        assert "coinvest_score_z" in features
        assert "financial_score" in features

    def test_ranker_mode_pairwise_minimal(self, replay_results):
        assert replay_results["architectural_finding"]["ranker_mode"] == "pairwise_minimal"


# ---------------------------------------------------------------------------
# TestBacktestPerformance
# ---------------------------------------------------------------------------


class TestBacktestPerformance:
    def test_performance_populated(self, replay_results):
        perf = replay_results["phase3_backtest_performance"]
        assert perf["n_dates"] == 16
        assert perf["mean_ic_5d"] is not None
        assert perf["mean_top20_xs_5d"] is not None

    def test_phase3_mean_ic_negative(self, replay_results):
        """Phase 3 was a BEAR period — mean IC should be negative."""
        perf = replay_results["phase3_backtest_performance"]
        assert perf["mean_ic_5d"] < 0, f"Expected negative Phase 3 IC, got {perf['mean_ic_5d']}"

    def test_per_date_backtest_present(self, replay_results):
        for c in replay_results["ranking_comparison"]:
            bp = c["backtest_performance"]
            assert "ic_5d" in bp, f"Missing ic_5d for {c['snap_date']}"
            assert "top20_xs_5d" in bp


# ---------------------------------------------------------------------------
# TestOutputStructure
# ---------------------------------------------------------------------------


class TestOutputStructure:
    def test_window_fields(self, replay_results):
        w = replay_results["window"]
        assert w["start_date"] == "2026-05-18"
        assert w["end_date"] == "2026-06-09"
        assert w["n_dates"] == 16

    def test_ranking_comparison_length(self, replay_results):
        assert len(replay_results["ranking_comparison"]) == 16

    def test_per_date_has_regime_fields(self, replay_results):
        for c in replay_results["ranking_comparison"]:
            assert c["actual_regime"] == "UNKNOWN"
            assert c["corrected_regime"] == "BEAR"
            assert c["reconstructed_vix"] is not None
            assert c["reconstructed_xbi_vs_spy_30d"] is not None

    def test_interpretation_present(self, replay_results):
        interp = replay_results["interpretation"]
        assert len(interp) > 50
        assert "IDENTICAL" in interp.upper() or "identical" in interp.lower()

    def test_json_serializable(self, replay_results):
        # Should not raise
        json.dumps(replay_results, default=str)


# ---------------------------------------------------------------------------
# TestModelLoading
# ---------------------------------------------------------------------------


class TestModelLoading:
    def test_model_trained(self, model):
        assert model.trained is True

    def test_model_weights_match_production(self, model):
        assert len(model.weights) == 2
        assert abs(model.weights[0] - 0.02) < 1e-6
        assert abs(model.weights[1] - (-0.05332037006884376)) < 1e-10

    def test_production_config_feature_set(self):
        assert module.PRODUCTION_RANKER_V2_CONFIG.feature_set == "minimal_v2"
