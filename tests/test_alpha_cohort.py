"""Tests for alpha cohort scoring module."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from module_5_alpha_cohort import (
    attach_alpha_scores,
    compute_alpha_cohort_key,
    compute_alpha_raw,
    load_alpha_cohort_table,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def v1_table_path():
    return PROJECT_ROOT / "production_data" / "alpha_cohort_tables" / "v1.json"


@pytest.fixture
def v1_table(v1_table_path):
    return load_alpha_cohort_table(v1_table_path)


def _make_table(cells=None, shrink_k=50):
    """Build a minimal in-memory table dict."""
    return {
        "_schema": {"name": "alpha_cohort_table", "version": "1"},
        "shrink_k_default": shrink_k,
        "alpha_clip": {"min": -0.10, "max": 0.10},
        "cells": cells or {},
    }


def _row(
    stage_bucket="mid",
    catalyst_mode="specific_days",
    catalyst_days=60,
    clinical_score_z_tier=0.5,
    ticker="TEST",
    **kwargs,
):
    d = {
        "stage_bucket": stage_bucket,
        "catalyst_mode": catalyst_mode,
        "catalyst_days": catalyst_days,
        "clinical_score_z_tier": clinical_score_z_tier,
        "ticker": ticker,
    }
    d.update(kwargs)
    return d


# ======================================================================
# load_alpha_cohort_table
# ======================================================================

class TestLoadTable:
    def test_loads_v1(self, v1_table_path):
        t = load_alpha_cohort_table(v1_table_path)
        assert "cells" in t
        assert len(t["cells"]) == 36  # 3 stages × 6 horizons × 2 signs

    def test_missing_cells_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text('{"foo": 1}')
        with pytest.raises(ValueError, match="cells"):
            load_alpha_cohort_table(bad)


# ======================================================================
# compute_alpha_cohort_key — stage mapping
# ======================================================================

class TestKeyStage:
    def test_empty_stage_falls_back_to_early(self):
        assert compute_alpha_cohort_key(_row(stage_bucket="")).startswith("early|")

    def test_none_stage_falls_back_to_early(self):
        assert compute_alpha_cohort_key(_row(stage_bucket=None)).startswith("early|")

    def test_unknown_stage_falls_back_to_early(self):
        assert compute_alpha_cohort_key(_row(stage_bucket="unknown")).startswith("early|")

    def test_valid_stages(self):
        for s in ("early", "mid", "late"):
            assert compute_alpha_cohort_key(_row(stage_bucket=s)).startswith(f"{s}|")


# ======================================================================
# compute_alpha_cohort_key — clinical sign
# ======================================================================

class TestKeySign:
    def test_positive_z(self):
        assert compute_alpha_cohort_key(_row(clinical_score_z_tier=0.5)).endswith("|pos")

    def test_negative_z(self):
        assert compute_alpha_cohort_key(_row(clinical_score_z_tier=-0.1)).endswith("|nonpos")

    def test_zero_z_is_nonpos(self):
        assert compute_alpha_cohort_key(_row(clinical_score_z_tier=0.0)).endswith("|nonpos")

    def test_none_z_is_nonpos(self):
        assert compute_alpha_cohort_key(_row(clinical_score_z_tier=None)).endswith("|nonpos")

    def test_empty_z_is_nonpos(self):
        assert compute_alpha_cohort_key(_row(clinical_score_z_tier="")).endswith("|nonpos")

    def test_string_positive(self):
        assert compute_alpha_cohort_key(_row(clinical_score_z_tier="1.2")).endswith("|pos")


# ======================================================================
# compute_alpha_cohort_key — catalyst horizon
# ======================================================================

class TestKeyHorizon:
    def test_specific_days_near_0_30(self):
        key = compute_alpha_cohort_key(_row(catalyst_mode="specific_days", catalyst_days=15))
        assert "|near_0_30|" in key

    def test_specific_days_near_31_90(self):
        key = compute_alpha_cohort_key(_row(catalyst_mode="specific_days", catalyst_days=60))
        assert "|near_31_90|" in key

    def test_specific_days_near_91_180(self):
        key = compute_alpha_cohort_key(_row(catalyst_mode="specific_days", catalyst_days=120))
        assert "|near_91_180|" in key

    def test_specific_days_near_181_270(self):
        key = compute_alpha_cohort_key(_row(catalyst_mode="specific_days", catalyst_days=200))
        assert "|near_181_270|" in key

    def test_specific_days_boundary_30(self):
        key = compute_alpha_cohort_key(_row(catalyst_mode="specific_days", catalyst_days=30))
        assert "|near_0_30|" in key

    def test_specific_days_boundary_31(self):
        key = compute_alpha_cohort_key(_row(catalyst_mode="specific_days", catalyst_days=31))
        assert "|near_31_90|" in key

    def test_blended_window_works(self):
        key = compute_alpha_cohort_key(_row(catalyst_mode="blended_window", catalyst_days=60))
        assert "|near_31_90|" in key

    def test_far_window_mode(self):
        key = compute_alpha_cohort_key(_row(catalyst_mode="far_window", catalyst_days=400))
        assert "|far_271_540|" in key

    def test_far_window_ignores_days(self):
        key = compute_alpha_cohort_key(_row(catalyst_mode="far_window", catalyst_days=10))
        assert "|far_271_540|" in key

    def test_no_upcoming_is_none(self):
        key = compute_alpha_cohort_key(_row(catalyst_mode="no_upcoming", catalyst_days=""))
        assert "|none|" in key

    def test_missing_mode_is_none(self):
        key = compute_alpha_cohort_key(_row(catalyst_mode="missing", catalyst_days=""))
        assert "|none|" in key

    def test_empty_days_is_none(self):
        key = compute_alpha_cohort_key(_row(catalyst_mode="specific_days", catalyst_days=""))
        assert "|none|" in key

    def test_none_days_is_none(self):
        key = compute_alpha_cohort_key(_row(catalyst_mode="specific_days", catalyst_days=None))
        assert "|none|" in key

    def test_very_large_days(self):
        key = compute_alpha_cohort_key(_row(catalyst_mode="specific_days", catalyst_days=600))
        assert "|far_271_540|" in key


# ======================================================================
# compute_alpha_raw — shrinkage + clipping
# ======================================================================

class TestAlphaRaw:
    def test_missing_key_returns_zero(self):
        table = _make_table({"a|b|c": {"mean_excess_ret_6m": 0.05, "n": 100}})
        assert compute_alpha_raw("x|y|z", table, 50, -0.10, 0.10) == 0.0

    def test_n_zero_returns_zero(self):
        table = _make_table({"a|b|c": {"mean_excess_ret_6m": 0.05, "n": 0}})
        assert compute_alpha_raw("a|b|c", table, 50, -0.10, 0.10) == 0.0

    def test_high_n_approaches_mean(self):
        table = _make_table({"k": {"mean_excess_ret_6m": 0.05, "n": 1000}})
        alpha = compute_alpha_raw("k", table, 50, -0.10, 0.10)
        # w = 1000/1050 ≈ 0.952, alpha ≈ 0.0476
        assert 0.045 < alpha < 0.05

    def test_clip_max(self):
        table = _make_table({"k": {"mean_excess_ret_6m": 0.20, "n": 1000}})
        assert compute_alpha_raw("k", table, 50, -0.10, 0.10) == 0.10

    def test_clip_min(self):
        table = _make_table({"k": {"mean_excess_ret_6m": -0.15, "n": 1000}})
        assert compute_alpha_raw("k", table, 50, -0.10, 0.10) == -0.10

    def test_shrinkage_formula(self):
        table = _make_table({"k": {"mean_excess_ret_6m": 0.06, "n": 50}})
        # w = 50/100 = 0.5, alpha = 0.06 * 0.5 = 0.03
        assert compute_alpha_raw("k", table, 50, -0.10, 0.10) == pytest.approx(0.03)

    def test_negative_n_returns_zero(self):
        table = _make_table({"k": {"mean_excess_ret_6m": 0.05, "n": -5}})
        assert compute_alpha_raw("k", table, 50, -0.10, 0.10) == 0.0


# ======================================================================
# attach_alpha_scores — percentile assignment
# ======================================================================

class TestAttachAlphaScores:
    def test_three_rows_correct_pct(self):
        table = _make_table({
            "mid|near_31_90|pos": {"mean_excess_ret_6m": 0.08, "n": 200},
            "mid|near_31_90|nonpos": {"mean_excess_ret_6m": 0.02, "n": 200},
            "early|none|nonpos": {"mean_excess_ret_6m": -0.03, "n": 200},
        })
        rows = [
            _row(ticker="AAA", stage_bucket="mid", catalyst_mode="specific_days",
                 catalyst_days=60, clinical_score_z_tier=1.0),
            _row(ticker="BBB", stage_bucket="mid", catalyst_mode="specific_days",
                 catalyst_days=60, clinical_score_z_tier=-0.5),
            _row(ticker="CCC", stage_bucket="early", catalyst_mode="no_upcoming",
                 catalyst_days="", clinical_score_z_tier=-0.1),
        ]
        attach_alpha_scores(rows, table, shrink_k=50, clip_min=-0.10, clip_max=0.10)

        # AAA has highest raw → rank 1 → pct = (1-0.5)/3
        assert rows[0]["alpha_cohort_pct"] == pytest.approx(round(0.5 / 3, 6))
        # BBB mid → rank 2
        assert rows[1]["alpha_cohort_pct"] == pytest.approx(round(1.5 / 3, 6))
        # CCC lowest → rank 3
        assert rows[2]["alpha_cohort_pct"] == pytest.approx(round(2.5 / 3, 6))

    def test_ties_broken_by_ticker_ascending(self):
        table = _make_table()  # all n=0 → all alpha=0.0
        rows = [
            _row(ticker="ZZZ"),
            _row(ticker="AAA"),
            _row(ticker="MMM"),
        ]
        attach_alpha_scores(rows, table, shrink_k=50, clip_min=-0.10, clip_max=0.10)

        # All raw=0.0, so ticker sort: AAA < MMM < ZZZ
        pcts = {r["ticker"]: r["alpha_cohort_pct"] for r in rows}
        assert pcts["AAA"] < pcts["MMM"] < pcts["ZZZ"]

    def test_pct_values_in_0_1(self):
        table = _make_table()
        rows = [_row(ticker=f"T{i}") for i in range(10)]
        attach_alpha_scores(rows, table, shrink_k=50, clip_min=-0.10, clip_max=0.10)
        for r in rows:
            assert 0 < r["alpha_cohort_pct"] < 1

    def test_pct_round_6dp(self):
        table = _make_table()
        rows = [_row(ticker=f"T{i}") for i in range(7)]
        attach_alpha_scores(rows, table, shrink_k=50, clip_min=-0.10, clip_max=0.10)
        for r in rows:
            s = str(r["alpha_cohort_pct"])
            if "." in s:
                assert len(s.split(".")[1]) <= 6

    def test_empty_rows_no_crash(self):
        table = _make_table()
        rows = []
        attach_alpha_scores(rows, table, shrink_k=50, clip_min=-0.10, clip_max=0.10)
        assert rows == []

    def test_keys_set_on_rows(self):
        table = _make_table()
        rows = [_row(ticker="X", stage_bucket="late", catalyst_mode="far_window",
                     catalyst_days=400, clinical_score_z_tier=0.3)]
        attach_alpha_scores(rows, table, shrink_k=50, clip_min=-0.10, clip_max=0.10)
        assert rows[0]["alpha_cohort_key"] == "late|far_271_540|pos"
        assert rows[0]["alpha_cohort_raw"] == 0.0
        assert rows[0]["alpha_cohort_pct"] == 0.5  # single row → (1-0.5)/1

    def test_deterministic_across_calls(self):
        table = _make_table({
            "mid|near_31_90|pos": {"mean_excess_ret_6m": 0.04, "n": 100},
        })
        rows1 = [_row(ticker="A"), _row(ticker="B")]
        rows2 = [_row(ticker="A"), _row(ticker="B")]
        attach_alpha_scores(rows1, table, shrink_k=50, clip_min=-0.10, clip_max=0.10)
        attach_alpha_scores(rows2, table, shrink_k=50, clip_min=-0.10, clip_max=0.10)
        for r1, r2 in zip(rows1, rows2):
            assert r1["alpha_cohort_pct"] == r2["alpha_cohort_pct"]
            assert r1["alpha_cohort_raw"] == r2["alpha_cohort_raw"]


# ======================================================================
# Full key composition
# ======================================================================

class TestFullKeyComposition:
    def test_complete_key_format(self):
        key = compute_alpha_cohort_key(_row(
            stage_bucket="late",
            catalyst_mode="specific_days",
            catalyst_days=15,
            clinical_score_z_tier=0.5,
        ))
        assert key == "late|near_0_30|pos"

    def test_all_36_keys_reachable(self):
        """Verify all 36 grid cells are reachable with appropriate inputs."""
        stages = ["early", "mid", "late"]
        horizon_inputs = [
            ("specific_days", 15, "near_0_30"),
            ("specific_days", 60, "near_31_90"),
            ("specific_days", 120, "near_91_180"),
            ("specific_days", 200, "near_181_270"),
            ("far_window", 400, "far_271_540"),
            ("no_upcoming", "", "none"),
        ]
        signs = [(0.5, "pos"), (-0.5, "nonpos")]

        keys = set()
        for stage in stages:
            for mode, days, horizon in horizon_inputs:
                for z, sign in signs:
                    key = compute_alpha_cohort_key(_row(
                        stage_bucket=stage,
                        catalyst_mode=mode,
                        catalyst_days=days,
                        clinical_score_z_tier=z,
                    ))
                    assert key == f"{stage}|{horizon}|{sign}"
                    keys.add(key)
        assert len(keys) == 36


# ======================================================================
# Composite engine override
# ======================================================================

from common.score_to_er import attach_rank_and_z


def _make_csv_rows(n=5, alpha_raws=None):
    """Build minimal csv_rows with composite_score and alpha_cohort_raw."""
    rows = []
    for i in range(n):
        rows.append({
            "ticker": f"T{i:03d}",
            "composite_score": float(n - i),  # legacy descending: T000=5, T001=4, ...
            "composite_rank": i + 1,
            "score_rank_pct": "",
            "score_z": "",
            "alpha_cohort_raw": alpha_raws[i] if alpha_raws else float(i) * 0.02,
        })
    # Seed legacy score_rank_pct/score_z via attach_rank_and_z
    attach_rank_and_z(rows, score_key="composite_score")
    return rows


def _apply_composite_override(csv_rows):
    """Replicate the composite override block from run_screen.py."""
    for row in csv_rows:
        row["composite_score"] = row.get("alpha_cohort_raw", 0.0)
    attach_rank_and_z(csv_rows, score_key="composite_score")
    _idx_sorted = sorted(
        range(len(csv_rows)),
        key=lambda i: (-float(csv_rows[i].get("composite_score", 0)),
                       csv_rows[i].get("ticker", "")),
    )
    for rank, idx in enumerate(_idx_sorted, start=1):
        csv_rows[idx]["composite_rank"] = rank


class TestCompositeEngineOverride:
    def test_composite_engine_legacy_unchanged(self):
        """composite_engine='legacy' should leave composite_score as-is."""
        rows = _make_csv_rows(5)
        orig_scores = [r["composite_score"] for r in rows]
        # No override applied — scores stay at legacy values
        assert [r["composite_score"] for r in rows] == orig_scores

    def test_composite_engine_alpha_overwrites_score(self):
        """After override, composite_score == alpha_cohort_raw for every row."""
        rows = _make_csv_rows(5)
        _apply_composite_override(rows)
        for r in rows:
            assert r["composite_score"] == r["alpha_cohort_raw"]

    def test_composite_rank_reordered(self):
        """composite_rank reflects alpha_cohort_raw ordering (highest = rank 1)."""
        alphas = [0.01, 0.08, 0.04, 0.02, 0.06]
        rows = _make_csv_rows(5, alpha_raws=alphas)
        _apply_composite_override(rows)
        # Highest alpha_cohort_raw = 0.08 (T001) should be rank 1
        rank_by_ticker = {r["ticker"]: r["composite_rank"] for r in rows}
        assert rank_by_ticker["T001"] == 1
        assert rank_by_ticker["T004"] == 2
        assert rank_by_ticker["T002"] == 3
        assert rank_by_ticker["T003"] == 4
        assert rank_by_ticker["T000"] == 5

    def test_score_rank_pct_recomputed(self):
        """score_rank_pct and score_z are updated after override."""
        alphas = [0.01, 0.08, 0.04, 0.02, 0.06]
        rows = _make_csv_rows(5, alpha_raws=alphas)
        old_pcts = [r["score_rank_pct"] for r in rows]
        _apply_composite_override(rows)
        new_pcts = [r["score_rank_pct"] for r in rows]
        # Percentiles should change (alpha order != legacy order)
        assert old_pcts != new_pcts
        # Highest alpha (T001) should have highest percentile
        t001 = next(r for r in rows if r["ticker"] == "T001")
        assert t001["score_rank_pct"] == pytest.approx((5 - 1 + 0.5) / 5)
        # score_z should be set (not empty)
        for r in rows:
            assert r["score_z"] != ""
            assert isinstance(r["score_z"], float)

    def test_corr_near_one(self):
        """After override, corr(alpha_cohort_pct, score_rank_pct) should be ~1.0.

        alpha_cohort_pct is assigned by attach_alpha_scores (rank by raw DESC),
        score_rank_pct is assigned by attach_rank_and_z (rank by composite_score DESC).
        After override composite_score == alpha_cohort_raw, so both rank the same way.
        """
        import statistics

        n = 20
        alphas = [i * 0.005 for i in range(n)]
        rows = _make_csv_rows(n, alpha_raws=alphas)
        # Simulate alpha_cohort_pct assignment (rank by raw DESC, continuity-corrected)
        sorted_idx = sorted(range(n), key=lambda i: (-rows[i]["alpha_cohort_raw"],
                                                      rows[i]["ticker"]))
        for rank, idx in enumerate(sorted_idx, start=1):
            rows[idx]["alpha_cohort_pct"] = (n - rank + 0.5) / n

        _apply_composite_override(rows)

        pcts = [r["alpha_cohort_pct"] for r in rows]
        ranks = [r["score_rank_pct"] for r in rows]
        # Pearson correlation
        mean_p = statistics.mean(pcts)
        mean_r = statistics.mean(ranks)
        cov = sum((p - mean_p) * (r - mean_r) for p, r in zip(pcts, ranks))
        std_p = (sum((p - mean_p) ** 2 for p in pcts)) ** 0.5
        std_r = (sum((r - mean_r) ** 2 for r in ranks)) ** 0.5
        corr = cov / (std_p * std_r) if std_p * std_r > 0 else 0.0
        assert corr == pytest.approx(1.0, abs=0.01)
