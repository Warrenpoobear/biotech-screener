"""Tests for scripts/replay_diff.py — replay-bundle regression harness."""
from __future__ import annotations

import csv
import io
import json
import math
import os
import sys
import tarfile
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# ── Ensure project root importable ────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from scripts.replay_diff import (
    BAD_CATALYST_MODES,
    GOOD_CATALYST_MODES,
    REQUIRED_COLUMNS,
    DiffResult,
    DiffThresholds,
    HealthVerdict,
    _is_eligible,
    _jaccard_overlap,
    _safe_float,
    compute_diff,
    evaluate_health,
    format_json_report,
    format_markdown_report,
    load_rankings,
    main,
)


# ---------------------------------------------------------------------------
# Synthetic DataFrame builder (pattern: test_phase2_health_gate.py:33-64)
# ---------------------------------------------------------------------------


def _make_rankings(
    n: int = 50,
    tickers: list[str] | None = None,
    archetypes: list[str] | None = None,
    tiers: list[str] | None = None,
    eligibles: list[str] | None = None,
    catalyst_modes: list[str] | None = None,
    scores: list[float] | None = None,
    score_rank_pcts: list[float] | None = None,
    weights: list[float] | None = None,
    extra_cols: dict | None = None,
) -> pd.DataFrame:
    """Build a synthetic rankings DataFrame with string dtype."""
    if tickers is None:
        tickers = [f"TK{i:03d}" for i in range(1, n + 1)]
    n = len(tickers)

    data = {
        "ticker": tickers,
        "archetype": archetypes or ["drug_developer"] * n,
        "eligible": eligibles or ["1"] * n,
        "tier_dev": tiers or ["B"] * n,
        "composite_rank": [str(i) for i in range(1, n + 1)],
        "actionable_rank": [str(i) for i in range(1, n + 1)],
        "composite_score": [str(s) for s in (scores or [50.0] * n)],
        "score_rank_pct": [
            str(s) for s in (score_rank_pcts or [round(1 - i / n, 4) for i in range(n)])
        ],
        "catalyst_mode": catalyst_modes or ["specific_days"] * n,
        "target_weight_pct": [str(w) for w in (weights or [5.0] * n)],
        "size_band": ["M"] * n,
    }
    if extra_cols:
        data.update(extra_cols)

    df = pd.DataFrame(data)
    # Ensure all columns are string type to match real CSV loading
    for col in df.columns:
        df[col] = df[col].astype(str)
    return df


def _write_rankings_csv(df: pd.DataFrame, path: Path) -> Path:
    csv_path = path / "rankings.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _make_tarball(df: pd.DataFrame, tar_path: Path) -> Path:
    """Create a .tar.gz with a rankings.csv inside."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    csv_bytes = buf.getvalue().encode("utf-8")

    with tarfile.open(tar_path, "w:gz") as tar:
        info = tarfile.TarInfo(name="2026-02-07/rankings.csv")
        info.size = len(csv_bytes)
        tar.addfile(info, io.BytesIO(csv_bytes))

    return tar_path


# ===========================================================================
# TestLoadRankings
# ===========================================================================


class TestLoadRankings:
    def test_load_from_directory(self, tmp_path):
        df = _make_rankings(n=10)
        _write_rankings_csv(df, tmp_path)

        loaded = load_rankings(str(tmp_path))
        assert len(loaded) == 10
        assert "ticker" in loaded.columns

    def test_load_from_tarball(self, tmp_path):
        df = _make_rankings(n=15)
        tar_path = _make_tarball(df, tmp_path / "test.tar.gz")

        loaded = load_rankings(str(tar_path))
        assert len(loaded) == 15
        assert set(REQUIRED_COLUMNS).issubset(set(loaded.columns))

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_rankings(str(tmp_path / "nonexistent.tar.gz"))

    def test_missing_csv_in_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="rankings.csv not found"):
            load_rankings(str(tmp_path))

    def test_empty_csv_raises(self, tmp_path):
        csv_path = tmp_path / "rankings.csv"
        csv_path.write_text("ticker,eligible,tier_dev,composite_rank,actionable_rank,archetype\n")
        with pytest.raises(ValueError, match="empty"):
            load_rankings(str(tmp_path))

    def test_missing_required_column_raises(self, tmp_path):
        df = _make_rankings(n=5)
        df = df.drop(columns=["tier_dev"])
        _write_rankings_csv(df, tmp_path)
        with pytest.raises(ValueError, match="required columns"):
            load_rankings(str(tmp_path))

    def test_deduplicates_tickers(self, tmp_path):
        df = _make_rankings(tickers=["AAA", "BBB", "AAA", "CCC"])
        _write_rankings_csv(df, tmp_path)
        loaded = load_rankings(str(tmp_path))
        assert len(loaded) == 3

    def test_ambiguous_tarball_raises(self, tmp_path):
        """Multiple rankings.csv in archive → fail fast."""
        df = _make_rankings(n=5)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        csv_bytes = buf.getvalue().encode("utf-8")

        tar_path = tmp_path / "ambiguous.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            info1 = tarfile.TarInfo(name="snap_a/rankings.csv")
            info1.size = len(csv_bytes)
            tar.addfile(info1, io.BytesIO(csv_bytes))
            info2 = tarfile.TarInfo(name="snap_b/rankings.csv")
            info2.size = len(csv_bytes)
            tar.addfile(info2, io.BytesIO(csv_bytes))

        with pytest.raises(ValueError, match="Ambiguous"):
            load_rankings(str(tar_path))

    def test_path_traversal_member_skipped(self, tmp_path):
        """Members with '..' in name are skipped during tarball search."""
        df = _make_rankings(n=5)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        csv_bytes = buf.getvalue().encode("utf-8")

        tar_path = tmp_path / "traversal.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            # Bad member with path traversal — should be skipped
            bad_info = tarfile.TarInfo(name="../etc/rankings.csv")
            bad_info.size = len(csv_bytes)
            tar.addfile(bad_info, io.BytesIO(csv_bytes))
            # Good member
            good_info = tarfile.TarInfo(name="2026-02-07/rankings.csv")
            good_info.size = len(csv_bytes)
            tar.addfile(good_info, io.BytesIO(csv_bytes))

        loaded = load_rankings(str(tar_path))
        assert len(loaded) == 5

    def test_rank_fallback_sets_attrs(self, tmp_path):
        """When actionable_rank is missing, attrs track the fallback."""
        df = _make_rankings(n=5)
        df = df.drop(columns=["actionable_rank"])
        _write_rankings_csv(df, tmp_path)

        loaded = load_rankings(str(tmp_path))
        assert loaded.attrs["has_actionable_rank"] is False
        assert loaded.attrs["rank_column_used"] == "composite_rank"
        assert "actionable_rank" in loaded.columns  # synthesized

    def test_rank_native_sets_attrs(self, tmp_path):
        """When actionable_rank exists natively, attrs reflect that."""
        df = _make_rankings(n=5)
        _write_rankings_csv(df, tmp_path)

        loaded = load_rankings(str(tmp_path))
        assert loaded.attrs["has_actionable_rank"] is True
        assert loaded.attrs["rank_column_used"] == "actionable_rank"


# ===========================================================================
# TestComputeDiff
# ===========================================================================


class TestComputeDiff:
    def test_identical_inputs_perfect_overlap(self):
        df = _make_rankings(n=30)
        result = compute_diff(df.copy(), df.copy())

        assert result.top20_overlap_pct == 100.0
        assert result.top60_overlap_pct == 100.0
        assert result.common_tickers == 30
        assert result.eligibility_change_count == 0
        assert result.tier_migration_count == 0
        assert result.top20_entrants == []
        assert result.top20_exits == []
        assert result.catalyst_mode_change_count == 0

    def test_identical_spearman_is_one(self):
        df = _make_rankings(n=30)
        result = compute_diff(df.copy(), df.copy())
        assert result.rank_spearman_rho is not None
        assert result.rank_spearman_rho == 1.0

    def test_shuffled_ranks_detects_churn(self):
        base = _make_rankings(n=30)
        cand = base.copy()
        # Reverse the ranking order
        cand["actionable_rank"] = [str(31 - i) for i in range(1, 31)]

        result = compute_diff(base, cand)
        assert result.rank_spearman_rho is not None
        assert result.rank_spearman_rho < 0.0  # anti-correlated
        assert result.top20_overlap_pct < 100.0

    def test_eligibility_flip_counted(self):
        base = _make_rankings(n=20, eligibles=["1"] * 20)
        cand = base.copy()
        # Flip 3 tickers from eligible to ineligible
        cand.loc[0, "eligible"] = "0"
        cand.loc[1, "eligible"] = "0"
        cand.loc[2, "eligible"] = "0"

        result = compute_diff(base, cand)
        assert result.eligibility_change_count == 3
        assert len(result.eligibility_lost) == 3
        assert len(result.eligibility_gained) == 0

    def test_eligibility_gained(self):
        base = _make_rankings(n=10, eligibles=["0"] * 3 + ["1"] * 7)
        cand = base.copy()
        # Flip first 3 from ineligible to eligible
        for i in range(3):
            cand.loc[i, "eligible"] = "1"

        result = compute_diff(base, cand)
        assert result.eligibility_change_count == 3
        assert len(result.eligibility_gained) == 3

    def test_tier_migration_matrix(self):
        tiers_base = ["A", "A", "B", "B", "C", "C", "D", "D"]
        tiers_cand = ["B", "A", "A", "B", "D", "C", "C", "D"]
        base = _make_rankings(n=8, tiers=tiers_base)
        cand = base.copy()
        for i, t in enumerate(tiers_cand):
            cand.loc[i, "tier_dev"] = t

        result = compute_diff(base, cand)
        # A->B, B->A, C->D, D->C = 4 migrations
        assert result.tier_migration_count == 4
        assert "A->B" in result.tier_migration_matrix
        assert "B->A" in result.tier_migration_matrix
        assert result.tier_migration_matrix["A->B"] == 1
        assert result.tier_migration_matrix["B->A"] == 1

    def test_score_rmse_computation(self):
        base = _make_rankings(n=10, scores=[50.0] * 10)
        cand = base.copy()
        # Offset all scores by 3.0
        cand["composite_score"] = ["53.0"] * 10

        result = compute_diff(base, cand)
        assert result.composite_score_rmse is not None
        assert abs(result.composite_score_rmse - 3.0) < 0.01

    def test_score_rank_pct_mae(self):
        base = _make_rankings(n=10)
        cand = base.copy()
        # Offset score_rank_pct by 0.05
        cand["score_rank_pct"] = [
            str(float(base.loc[i, "score_rank_pct"]) + 0.05) for i in range(10)
        ]

        result = compute_diff(base, cand)
        assert result.score_rank_pct_mae is not None
        assert abs(result.score_rank_pct_mae - 0.05) < 0.001

    def test_catalyst_mode_transitions(self):
        modes_base = ["specific_days", "no_upcoming", "missing", "blended_window", "far_window"]
        modes_cand = ["no_upcoming", "specific_days", "blended_window", "missing", "far_window"]
        base = _make_rankings(n=5, catalyst_modes=modes_base)
        cand = base.copy()
        for i, m in enumerate(modes_cand):
            cand.loc[i, "catalyst_mode"] = m

        result = compute_diff(base, cand)
        assert result.catalyst_mode_change_count == 4  # last one unchanged
        assert len(result.catalyst_bad_to_good) == 2   # no_upcoming->specific_days, missing->blended_window
        assert len(result.catalyst_good_to_bad) == 2   # specific_days->no_upcoming, blended_window->missing

    def test_weight_l1_computation(self):
        base = _make_rankings(n=20, weights=[5.0] * 20)
        cand = base.copy()
        # Change weight for first 5 tickers by 2pp each
        for i in range(5):
            cand.loc[i, "target_weight_pct"] = "7.0"

        result = compute_diff(base, cand)
        assert result.weight_l1_top20 is not None
        assert result.weight_l1_top20 == 10.0  # 5 * 2.0

    def test_missing_optional_column_graceful(self):
        base = _make_rankings(n=10)
        cand = base.copy()
        cand = cand.drop(columns=["composite_score"])

        result = compute_diff(base, cand)
        assert result.composite_score_rmse is None
        assert "composite_score" in result.schema_drift["baseline_only"]

    def test_disjoint_universes(self):
        base = _make_rankings(tickers=["AAA", "BBB", "CCC"])
        cand = _make_rankings(tickers=["XXX", "YYY", "ZZZ"])

        result = compute_diff(base, cand)
        assert result.common_tickers == 0
        assert result.top20_overlap_pct == 0.0
        assert result.rank_spearman_rho is None
        assert result.eligibility_change_count == 0

    def test_commercial_section(self):
        base = _make_rankings(
            tickers=["TK1", "TK2", "TK3", "CM1", "CM2"],
            archetypes=["drug_developer"] * 3 + ["commercial_biotech"] * 2,
            extra_cols={
                "tier_commercial": ["", "", "", "A", "B"],
            },
        )
        cand = base.copy()
        cand.loc[3, "tier_commercial"] = "B"  # CM1: A -> B

        result = compute_diff(base, cand)
        assert result.commercial_top20_overlap_pct is not None
        assert result.commercial_tier_migration_count == 1
        assert result.commercial_tier_migration_matrix == {"A->B": 1}

    def test_schema_drift_detection(self):
        base = _make_rankings(n=5, extra_cols={"extra_base_col": ["x"] * 5})
        cand = _make_rankings(n=5, extra_cols={"extra_cand_col": ["y"] * 5})

        result = compute_diff(base, cand)
        assert "extra_base_col" in result.schema_drift["baseline_only"]
        assert "extra_cand_col" in result.schema_drift["candidate_only"]

    def test_biggest_movers_sorted_by_abs_delta(self):
        base = _make_rankings(n=20)
        cand = base.copy()
        # Move TK001 from rank 1 to 20 (biggest move)
        cand.loc[0, "actionable_rank"] = "20"
        # Move TK002 from rank 2 to 10
        cand.loc[1, "actionable_rank"] = "10"

        result = compute_diff(base, cand)
        assert len(result.biggest_rank_movers) > 0
        # First mover should be the one with largest |delta|
        first = result.biggest_rank_movers[0]
        assert first[0] == "TK001"

    def test_rank_column_info_native(self):
        df = _make_rankings(n=10)
        df.attrs["has_actionable_rank"] = True
        df.attrs["rank_column_used"] = "actionable_rank"
        result = compute_diff(df.copy(), df.copy())
        rci = result.rank_column_info
        assert rci["baseline_has_actionable_rank"] is True
        assert rci["candidate_has_actionable_rank"] is True

    def test_rank_column_info_fallback(self):
        df = _make_rankings(n=10)
        df.attrs["has_actionable_rank"] = False
        df.attrs["rank_column_used"] = "composite_rank"
        result = compute_diff(df.copy(), df.copy())
        rci = result.rank_column_info
        assert rci["baseline_has_actionable_rank"] is False
        assert rci["baseline_rank_column_used"] == "composite_rank"


# ===========================================================================
# TestEvaluateHealth
# ===========================================================================


def _ok_result() -> DiffResult:
    """A DiffResult that should pass all default thresholds."""
    return DiffResult(
        baseline_source="test_base",
        candidate_source="test_cand",
        baseline_n_rows=100,
        candidate_n_rows=100,
        common_tickers=100,
        schema_drift={"baseline_only": [], "candidate_only": []},
        rank_column_info={
            "baseline_has_actionable_rank": True,
            "candidate_has_actionable_rank": True,
            "baseline_rank_column_used": "actionable_rank",
            "candidate_rank_column_used": "actionable_rank",
        },
        top20_overlap_pct=95.0,
        top60_overlap_pct=95.0,
        top100_overlap_pct=95.0,
        top20_entrants=[],
        top20_exits=[],
        top60_entrants=[],
        top60_exits=[],
        rank_spearman_rho=0.98,
        mean_abs_rank_delta_top60=2.0,
        max_abs_rank_delta_top60=5,
        biggest_rank_movers=[],
        composite_score_rmse=1.0,
        score_rank_pct_mae=0.02,
        eligibility_change_count=0,
        eligibility_gained=[],
        eligibility_lost=[],
        tier_migration_count=0,
        tier_migration_matrix={},
        tier_dist_baseline={"A": 5, "B": 20},
        tier_dist_candidate={"A": 5, "B": 20},
        catalyst_mode_change_count=0,
        catalyst_bad_to_good=[],
        catalyst_good_to_bad=[],
        weight_l1_top20=5.0,
        commercial_top20_overlap_pct=None,
        commercial_tier_migration_count=None,
        commercial_tier_migration_matrix=None,
    )


class TestEvaluateHealth:
    def test_all_ok(self):
        result = _ok_result()
        verdict = evaluate_health(result, DiffThresholds())
        assert verdict.status == "OK"
        assert verdict.exit_code == 0
        assert verdict.fail_reasons == []
        assert verdict.warn_reasons == []

    def test_warn_on_low_top20_overlap(self):
        result = _ok_result()
        result.top20_overlap_pct = 75.0
        verdict = evaluate_health(result, DiffThresholds())
        assert verdict.status == "WARN"
        assert verdict.exit_code == 2
        assert any("top20_overlap" in r for r in verdict.warn_reasons)

    def test_fail_on_very_low_top20_overlap(self):
        result = _ok_result()
        result.top20_overlap_pct = 40.0
        verdict = evaluate_health(result, DiffThresholds())
        assert verdict.status == "FAIL"
        assert verdict.exit_code == 1
        assert any("top20_overlap" in r for r in verdict.fail_reasons)

    def test_fail_on_low_spearman(self):
        result = _ok_result()
        result.rank_spearman_rho = 0.70
        verdict = evaluate_health(result, DiffThresholds())
        assert verdict.status == "FAIL"
        assert any("spearman" in r for r in verdict.fail_reasons)

    def test_warn_on_eligibility_changes(self):
        result = _ok_result()
        result.eligibility_change_count = 8
        verdict = evaluate_health(result, DiffThresholds())
        assert verdict.status == "WARN"
        assert any("eligibility" in r for r in verdict.warn_reasons)

    def test_fail_on_many_eligibility_changes(self):
        result = _ok_result()
        result.eligibility_change_count = 25
        verdict = evaluate_health(result, DiffThresholds())
        assert verdict.status == "FAIL"

    def test_multiple_warn_reasons(self):
        result = _ok_result()
        result.top20_overlap_pct = 75.0
        result.tier_migration_count = 15
        result.catalyst_mode_change_count = 20
        verdict = evaluate_health(result, DiffThresholds())
        assert verdict.status == "WARN"
        assert len(verdict.warn_reasons) >= 3

    def test_custom_thresholds(self):
        result = _ok_result()
        result.top20_overlap_pct = 92.0  # Would be OK with defaults

        # Custom strict thresholds
        strict = DiffThresholds(warn_top20_overlap_pct=95.0)
        verdict = evaluate_health(result, strict)
        assert verdict.status == "WARN"
        assert any("top20_overlap" in r for r in verdict.warn_reasons)

    def test_none_spearman_skipped(self):
        result = _ok_result()
        result.rank_spearman_rho = None
        verdict = evaluate_health(result, DiffThresholds())
        # Should not fail/warn on spearman if None
        assert not any("spearman" in r for r in verdict.fail_reasons)
        assert not any("spearman" in r for r in verdict.warn_reasons)

    def test_none_score_metrics_skipped(self):
        result = _ok_result()
        result.composite_score_rmse = None
        result.score_rank_pct_mae = None
        verdict = evaluate_health(result, DiffThresholds())
        assert verdict.status == "OK"

    def test_no_common_universe_is_fail(self):
        result = _ok_result()
        result.common_tickers = 0
        verdict = evaluate_health(result, DiffThresholds())
        assert verdict.status == "FAIL"
        assert verdict.exit_code == 1
        assert any("no_common_universe" in r for r in verdict.fail_reasons)


# ===========================================================================
# TestFormatReports
# ===========================================================================


class TestFormatReports:
    def test_json_report_schema(self):
        result = _ok_result()
        verdict = evaluate_health(result, DiffThresholds())
        report = format_json_report(result, verdict, DiffThresholds())

        assert "version" in report
        assert "timestamp" in report
        assert "thresholds_id" in report
        assert "verdict" in report
        assert "metrics" in report
        assert "thresholds" in report
        assert report["verdict"]["status"] == "OK"

    def test_markdown_has_sections(self):
        result = _ok_result()
        verdict = evaluate_health(result, DiffThresholds())
        md = format_markdown_report(result, verdict)

        assert "# Replay Diff Report" in md
        assert "## Overlap (Jaccard)" in md
        assert "## Rank Stability" in md
        assert "## Score Stability" in md
        assert "## Tier Migration (dev)" in md
        assert "## Eligibility Changes" in md
        assert "## Catalyst Mode Transitions" in md
        assert "## Weight Drift" in md

    def test_markdown_missing_column_shows_placeholder(self):
        result = _ok_result()
        result.composite_score_rmse = None
        verdict = evaluate_health(result, DiffThresholds())
        md = format_markdown_report(result, verdict)

        assert "(column missing)" in md

    def test_json_roundtrip(self, tmp_path):
        result = _ok_result()
        verdict = evaluate_health(result, DiffThresholds())
        report = format_json_report(result, verdict, DiffThresholds())

        json_path = tmp_path / "report.json"
        with open(json_path, "w") as fh:
            json.dump(report, fh)

        with open(json_path, "r") as fh:
            loaded = json.load(fh)

        assert loaded["verdict"]["status"] == report["verdict"]["status"]
        assert loaded["thresholds_id"] == report["thresholds_id"]
        assert loaded["metrics"]["common_tickers"] == 100

    def test_markdown_fail_shows_reasons(self):
        result = _ok_result()
        result.top20_overlap_pct = 40.0
        verdict = evaluate_health(result, DiffThresholds())
        md = format_markdown_report(result, verdict)

        assert "## FAIL Reasons" in md
        assert "**FAIL**" in md

    def test_markdown_schema_drift_section(self):
        result = _ok_result()
        result.schema_drift = {
            "baseline_only": ["old_col"],
            "candidate_only": ["new_col"],
        }
        verdict = evaluate_health(result, DiffThresholds())
        md = format_markdown_report(result, verdict)

        assert "## Schema Drift" in md
        assert "old_col" in md

    def test_markdown_rank_fallback_warning(self):
        result = _ok_result()
        result.rank_column_info = {
            "baseline_has_actionable_rank": False,
            "candidate_has_actionable_rank": True,
            "baseline_rank_column_used": "composite_rank",
            "candidate_rank_column_used": "actionable_rank",
        }
        verdict = evaluate_health(result, DiffThresholds())
        md = format_markdown_report(result, verdict)
        assert "Rank column fallback" in md
        assert "composite_rank" in md

    def test_json_has_rank_column_info(self):
        result = _ok_result()
        verdict = evaluate_health(result, DiffThresholds())
        report = format_json_report(result, verdict, DiffThresholds())
        rci = report["metrics"]["rank_column_info"]
        assert rci["baseline_has_actionable_rank"] is True


# ===========================================================================
# TestDiffThresholds
# ===========================================================================


class TestDiffThresholds:
    def test_frozen(self):
        t = DiffThresholds()
        with pytest.raises(AttributeError):
            t.fail_top20_overlap_pct = 99.0

    def test_thresholds_id_deterministic(self):
        t1 = DiffThresholds()
        t2 = DiffThresholds()
        assert t1.thresholds_id == t2.thresholds_id
        assert len(t1.thresholds_id) == 8

    def test_different_params_different_id(self):
        t1 = DiffThresholds()
        t2 = DiffThresholds(fail_top20_overlap_pct=99.0)
        assert t1.thresholds_id != t2.thresholds_id

    def test_from_json_roundtrip(self, tmp_path):
        original = DiffThresholds(
            fail_top20_overlap_pct=45.0,
            warn_tier_migration_count=8,
        )
        path = str(tmp_path / "thresholds.json")
        original.to_json(path)

        loaded = DiffThresholds.from_json(path)
        assert loaded.fail_top20_overlap_pct == 45.0
        assert loaded.warn_tier_migration_count == 8
        assert loaded.thresholds_id == original.thresholds_id


# ===========================================================================
# TestCLI
# ===========================================================================


class TestCLI:
    def test_main_ok(self, tmp_path):
        df = _make_rankings(n=30)
        base_dir = tmp_path / "base"
        base_dir.mkdir()
        _write_rankings_csv(df, base_dir)

        cand_dir = tmp_path / "cand"
        cand_dir.mkdir()
        _write_rankings_csv(df, cand_dir)

        out_dir = tmp_path / "output"
        rc = main([
            "--baseline", str(base_dir),
            "--candidate", str(cand_dir),
            "--output-dir", str(out_dir),
        ])
        assert rc == 0
        assert (out_dir / "diff_report.json").exists()
        assert (out_dir / "diff_report.md").exists()

    def test_main_strict_warn_exits_1(self, tmp_path):
        base = _make_rankings(n=30)
        cand = base.copy()
        # Reverse ranks to trigger WARN
        cand["actionable_rank"] = [str(31 - i) for i in range(1, 31)]

        base_dir = tmp_path / "base"
        base_dir.mkdir()
        _write_rankings_csv(base, base_dir)

        cand_dir = tmp_path / "cand"
        cand_dir.mkdir()
        _write_rankings_csv(cand, cand_dir)

        out_dir = tmp_path / "output"
        rc = main([
            "--baseline", str(base_dir),
            "--candidate", str(cand_dir),
            "--output-dir", str(out_dir),
            "--strict",
        ])
        # Should be WARN or FAIL (reversed ranks), --strict upgrades WARN→1
        assert rc == 1

    def test_main_with_tarball(self, tmp_path):
        df = _make_rankings(n=20)
        tar_path = _make_tarball(df, tmp_path / "base.tar.gz")

        cand_dir = tmp_path / "cand"
        cand_dir.mkdir()
        _write_rankings_csv(df, cand_dir)

        out_dir = tmp_path / "output"
        rc = main([
            "--baseline", str(tar_path),
            "--candidate", str(cand_dir),
            "--output-dir", str(out_dir),
        ])
        assert rc == 0


# ===========================================================================
# TestHelpers
# ===========================================================================


class TestHelpers:
    def test_jaccard_overlap_identical(self):
        assert _jaccard_overlap({"a", "b"}, {"a", "b"}) == 100.0

    def test_jaccard_overlap_disjoint(self):
        assert _jaccard_overlap({"a"}, {"b"}) == 0.0

    def test_jaccard_overlap_empty(self):
        assert _jaccard_overlap(set(), set()) == 100.0

    def test_jaccard_overlap_partial(self):
        # {a,b} & {b,c} = {b}; union = {a,b,c} = 3; 1/3 = 33.3%
        assert _jaccard_overlap({"a", "b"}, {"b", "c"}) == 33.3

    def test_safe_float_valid(self):
        assert _safe_float("3.14") == 3.14

    def test_safe_float_empty(self):
        assert _safe_float("") == 0.0

    def test_safe_float_nan(self):
        assert _safe_float(float("nan")) == 0.0

    def test_is_eligible_truthy(self):
        assert _is_eligible("1") is True
        assert _is_eligible("1.0") is True
        assert _is_eligible("True") is True
        assert _is_eligible("true") is True

    def test_is_eligible_falsy(self):
        assert _is_eligible("0") is False
        assert _is_eligible("") is False
        assert _is_eligible(None) is False
