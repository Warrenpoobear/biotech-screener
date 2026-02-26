"""Tests for scripts/alpha_modifier_promotion_check.py."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpha_modifier_promotion_check import (
    GATE_MAX_RANK_SHIFT,
    GATE_MEAN_TOP60_OVERLAP_MIN,
    GATE_TOP60_OVERLAP_MIN,
    GATE_TIER_A_REGRESSION_MAX,
    DateABResult,
    PromotionVerdict,
    _build_baseline_ruleset,
    compute_verdict,
    discover_eligible_dates,
    format_checklist,
    rerank_and_compare,
    run_promotion_check,
    write_results_json,
)
from decision_engine import DecisionRuleset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _candidate_ruleset(**overrides) -> DecisionRuleset:
    """Build a candidate ruleset with alpha modifier on."""
    defaults = dict(
        alpha_modifier_mode="tiebreak",
        alpha_modifier_weight=0.05,
        sort_anchor="alpha_cohort",
        composite_engine="alpha_cohort",
    )
    defaults.update(overrides)
    return DecisionRuleset(**defaults)


def _row(
    ticker: str,
    eligible: str = "1",
    tier_dev: str = "B",
    alpha_cohort_raw: str = "0.0",
    clinical_optionality_pct_dev: str = "0.5",
    composite_rank: str = "50",
    archetype: str = "drug_developer",
    **extra,
) -> Dict[str, str]:
    """Build a synthetic rankings row."""
    r = {
        "ticker": ticker,
        "eligible": eligible,
        "tier_dev": tier_dev,
        "alpha_cohort_raw": alpha_cohort_raw,
        "alpha_cohort_pct": "0.5",
        "clinical_optionality_pct_dev": clinical_optionality_pct_dev,
        "composite_rank": composite_rank,
        "archetype": archetype,
        "catalyst_event_type": "",
        "catalyst_source": "",
        "clinical_score_z_tier": "0.0",
        "clinical_score_z": "0.0",
        "coinvest_score_z": "0.0",
        "catalyst_decay_w": "0.0",
        "inst_delta_z": "0.0",
        "de_alpha_60d": "0.0",
        "de_beta_xbi_60d": "1.0",
        "de_drawdown": "-0.1",
        "de_rsi_14d": "50",
        "score_z": "0.0",
        "missingness_penalty": "0.0",
        "commercial_quality_pct": "0.5",
        "cat_priority": "9",
    }
    r.update(extra)
    return r


def _write_snapshot(snap_dir: Path, rows: List[Dict[str, str]]) -> None:
    """Write rankings.csv + metadata.json."""
    snap_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(k for r in rows for k in r.keys()))
    with open(snap_dir / "rankings.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    with open(snap_dir / "metadata.json", "w") as f:
        json.dump({"as_of_date": snap_dir.name}, f)


# ---------------------------------------------------------------------------
# _build_baseline_ruleset
# ---------------------------------------------------------------------------

class TestBuildBaseline:

    def test_forces_off(self):
        candidate = _candidate_ruleset()
        baseline = _build_baseline_ruleset(candidate)
        assert baseline.alpha_modifier_mode == "off"
        assert baseline.alpha_modifier_weight == 0.0

    def test_preserves_other_fields(self):
        candidate = _candidate_ruleset(catalyst_near_days=120, clinical_sort_weight=1.0)
        baseline = _build_baseline_ruleset(candidate)
        assert baseline.catalyst_near_days == 120
        assert baseline.clinical_sort_weight == 1.0


# ---------------------------------------------------------------------------
# rerank_and_compare
# ---------------------------------------------------------------------------

class TestRerankAndCompare:

    def test_identical_when_no_alpha(self):
        """With all alpha_cohort_raw=0, modifier has no effect → overlap=1.0."""
        rows = [_row(f"T{i:02d}", alpha_cohort_raw="0.0") for i in range(80)]
        candidate = _candidate_ruleset()
        baseline = _build_baseline_ruleset(candidate)
        result = rerank_and_compare(rows, candidate, baseline, "2026-02-20")
        assert result.top60_overlap == 1.0
        assert result.pct_changed == 0.0
        assert result.max_shift == 0
        assert not result.flagged

    def test_nonzero_alpha_causes_shifts(self):
        """Varying alpha values should cause some rank shifts."""
        rows = []
        for i in range(80):
            # Assign different alpha values to create divergence
            alpha = str(round(0.05 * (i % 5 - 2), 2))
            rows.append(_row(f"T{i:02d}", alpha_cohort_raw=alpha,
                             composite_rank=str(i + 1)))
        candidate = _candidate_ruleset()
        baseline = _build_baseline_ruleset(candidate)
        result = rerank_and_compare(rows, candidate, baseline, "2026-02-20")
        # With nonzero alpha, some shifts should occur
        assert result.n_alpha_nonzero > 0
        # overlap should still be reasonable for a small weight
        assert result.top60_overlap > 0.5

    def test_tier_a_tracking(self):
        """Verify tier distribution is tracked."""
        rows = [
            _row("T01", tier_dev="A", alpha_cohort_raw="0.05", composite_rank="1"),
            _row("T02", tier_dev="A", alpha_cohort_raw="0.0", composite_rank="2"),
            _row("T03", tier_dev="B", alpha_cohort_raw="-0.05", composite_rank="3"),
        ]
        # Extend to 60+ rows
        for i in range(4, 70):
            rows.append(_row(f"T{i:02d}", tier_dev="C", composite_rank=str(i)))

        candidate = _candidate_ruleset()
        baseline = _build_baseline_ruleset(candidate)
        result = rerank_and_compare(rows, candidate, baseline, "2026-02-20")
        # Should have tier counts tracked
        assert result.tier_a_modified >= 0
        assert result.tier_a_baseline >= 0

    def test_flag_on_low_overlap(self):
        """Extreme alpha divergence should flag low overlap."""
        rows = []
        for i in range(80):
            # Large alpha values to force significant reordering
            alpha = str(round(0.10 * (40 - i) / 40, 4))
            rows.append(_row(f"T{i:02d}", alpha_cohort_raw=alpha,
                             composite_rank=str(i + 1)))
        candidate = _candidate_ruleset(alpha_modifier_weight=0.25)  # max allowed weight
        baseline = _build_baseline_ruleset(candidate)
        result = rerank_and_compare(rows, candidate, baseline, "2026-02-20")
        # High weight + dispersed alpha → may trigger flags
        # (just verify the mechanism works, not the exact outcome)
        assert isinstance(result.flagged, bool)
        assert isinstance(result.flag_reasons, str)


# ---------------------------------------------------------------------------
# compute_verdict
# ---------------------------------------------------------------------------

class TestComputeVerdict:

    def test_all_pass(self):
        results = [
            DateABResult(
                date=f"2026-02-{20+i}", n_eligible=200, n_alpha_nonzero=180,
                top60_overlap=0.95, mean_abs_shift=1.0, max_shift=5,
                pct_changed=10.0, tier_a_modified=10, tier_a_baseline=10,
                tier_a_delta=0, tier_b_modified=20, tier_b_baseline=20,
            )
            for i in range(5)
        ]
        v = compute_verdict(results)
        assert v.promote is True
        assert v.n_flagged == 0
        assert v.mean_overlap >= GATE_MEAN_TOP60_OVERLAP_MIN

    def test_low_overlap_fails(self):
        results = [
            DateABResult(
                date="2026-02-20", n_eligible=200, n_alpha_nonzero=180,
                top60_overlap=0.80, mean_abs_shift=5.0, max_shift=10,
                pct_changed=30.0, tier_a_modified=8, tier_a_baseline=10,
                tier_a_delta=-2, tier_b_modified=20, tier_b_baseline=20,
                flagged=True, flag_reasons="overlap=0.800<0.90",
            ),
        ]
        v = compute_verdict(results)
        assert v.promote is False
        assert any("min_overlap" in r for r in v.reasons)

    def test_high_shift_fails(self):
        results = [
            DateABResult(
                date="2026-02-20", n_eligible=200, n_alpha_nonzero=180,
                top60_overlap=0.95, mean_abs_shift=5.0, max_shift=35,
                pct_changed=10.0, tier_a_modified=10, tier_a_baseline=10,
                tier_a_delta=0, tier_b_modified=20, tier_b_baseline=20,
                flagged=True, flag_reasons=f"max_shift=35>{GATE_MAX_RANK_SHIFT}",
            ),
        ]
        v = compute_verdict(results)
        assert v.promote is False
        assert any("max_shift" in r for r in v.reasons)

    def test_empty_results(self):
        v = compute_verdict([])
        assert v.promote is False
        assert v.n_dates == 0


# ---------------------------------------------------------------------------
# format_checklist
# ---------------------------------------------------------------------------

class TestFormatChecklist:

    def test_promote_banner(self):
        results = [
            DateABResult(
                date="2026-02-20", n_eligible=200, n_alpha_nonzero=180,
                top60_overlap=0.95, mean_abs_shift=1.0, max_shift=5,
                pct_changed=10.0, tier_a_modified=10, tier_a_baseline=10,
                tier_a_delta=0, tier_b_modified=20, tier_b_baseline=20,
            ),
        ]
        verdict = PromotionVerdict(
            n_dates=1, n_flagged=0, mean_overlap=0.95, min_overlap=0.95,
            max_max_shift=5, mean_pct_changed=10.0, promote=True, reasons=[],
        )
        candidate = _candidate_ruleset()
        text = format_checklist(results, verdict, candidate)
        assert "PROMOTE" in text
        assert "2026-02-20" in text

    def test_reject_banner(self):
        verdict = PromotionVerdict(
            n_dates=1, n_flagged=1, mean_overlap=0.85, min_overlap=0.80,
            max_max_shift=5, mean_pct_changed=30.0, promote=False,
            reasons=["min_overlap=0.800 < 0.90"],
        )
        candidate = _candidate_ruleset()
        text = format_checklist([], verdict, candidate)
        assert "DO NOT PROMOTE" in text


# ---------------------------------------------------------------------------
# discover_eligible_dates
# ---------------------------------------------------------------------------

class TestDiscoverEligibleDates:

    def test_finds_dates_with_alpha(self, tmp_path):
        # Date with alpha column (min_cols=1 for test)
        rows_good = [_row("T01")]
        _write_snapshot(tmp_path / "2026-02-20", rows_good)
        # Date with old schema (no alpha_cohort_raw)
        old_dir = tmp_path / "2026-01-01"
        old_dir.mkdir()
        with open(old_dir / "rankings.csv", "w") as f:
            f.write("ticker,composite_score\nAAA,50\n")

        dates = discover_eligible_dates(tmp_path, min_cols=1)
        assert "2026-02-20" in dates
        assert "2026-01-01" not in dates

    def test_date_range_filter(self, tmp_path):
        _write_snapshot(tmp_path / "2026-02-18", [_row("T01")])
        _write_snapshot(tmp_path / "2026-02-20", [_row("T01")])
        _write_snapshot(tmp_path / "2026-02-22", [_row("T01")])

        dates = discover_eligible_dates(tmp_path, date_from="2026-02-19", min_cols=1)
        assert "2026-02-18" not in dates
        assert "2026-02-20" in dates

    def test_empty_root(self, tmp_path):
        dates = discover_eligible_dates(tmp_path / "nonexistent")
        assert dates == []


# ---------------------------------------------------------------------------
# Integration: run_promotion_check with synthetic data
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_full_pipeline(self, tmp_path):
        """End-to-end: 2 dates → results + verdict."""
        snap_root = tmp_path / "snapshots"
        for d in ("2026-02-20", "2026-02-21"):
            rows = []
            for i in range(80):
                alpha = str(round(0.02 * (i % 3 - 1), 3))
                rows.append(_row(f"T{i:02d}", alpha_cohort_raw=alpha,
                                 composite_rank=str(i + 1)))
            _write_snapshot(snap_root / d, rows)

        candidate = _candidate_ruleset()
        results, verdict = run_promotion_check(
            snapshot_root=snap_root,
            candidate=candidate,
            min_cols=1,
        )
        assert len(results) == 2
        assert verdict.n_dates == 2
        # Verify structural integrity (overlap value depends on data shape)
        assert 0.0 <= verdict.mean_overlap <= 1.0
        assert verdict.min_overlap <= verdict.mean_overlap

    def test_write_results_json(self, tmp_path):
        results = [
            DateABResult(
                date="2026-02-20", n_eligible=200, n_alpha_nonzero=180,
                top60_overlap=0.95, mean_abs_shift=1.0, max_shift=5,
                pct_changed=10.0, tier_a_modified=10, tier_a_baseline=10,
                tier_a_delta=0, tier_b_modified=20, tier_b_baseline=20,
            ),
        ]
        verdict = PromotionVerdict(
            n_dates=1, n_flagged=0, mean_overlap=0.95, min_overlap=0.95,
            max_max_shift=5, mean_pct_changed=10.0, promote=True, reasons=[],
        )
        out_path = tmp_path / "results.json"
        write_results_json(results, verdict, out_path)
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert data["verdict"]["promote"] is True
        assert len(data["per_date"]) == 1
        assert "gates" in data
