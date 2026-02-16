"""Tests for scripts/compare_rulesets_replay.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from compare_rulesets_replay import compare, format_report, rank_portfolio
from decision_engine import DecisionRuleset

PROD_BASELINE = Path(__file__).resolve().parent.parent / "production_data" / "decision_rulesets" / "v1.3.2_candidate.json"
PROD_CANDIDATE = Path(__file__).resolve().parent.parent / "production_data" / "decision_rulesets" / "v1.3.3_missing_sort_only_candidate.json"


def _make_rankings(n=30, n_missing=0):
    """Build a synthetic rankings DataFrame matching snapshot schema."""
    tickers = [f"T{i:03d}" for i in range(n)]
    data = {
        "ticker": tickers,
        "archetype": ["drug_developer"] * n,
        "eligible": [1] * n,
        "tier_dev": ["A"] * min(15, n) + ["B"] * max(0, n - 15),
        "composite_rank": list(range(1, n + 1)),
        "composite_score": [50.0 - i * 0.5 for i in range(n)],
        "clinical_optionality_pct_dev": [0.8 - i * 0.01 for i in range(n)],
        "catalyst_mode": ["specific_days"] * n,
        "catalyst_days": [30 + i for i in range(n)],
        "catalyst_strength": ["NEAR"] * n,
        "catalyst_event_type": ["DATA_READOUT"] * n,
        "catalyst_source": ["CTGOV_CALENDAR"] * n,
        "cat_priority": [2] * n,
        "risk_flags": [""] * n,
        "size_band": ["L"] * n,
        "target_weight_pct": [5.0] * n,
        "actionable_rank": list(range(1, n + 1)),
        "decision_engine_ruleset_id": ["96f655ee"] * n,
        "decision_engine_version": ["v1.3.0"] * n,
        "missing_components": [""] * n,
        "missingness_penalty": [0] * n,
        "mom_state": ["neutral"] * n,
        "sponsor_tier1_count": [3] * n,
        "tier_reason": ["high_opt+catalyst_near"] * n,
        "size_reasons": ["tier_a_dev"] * n,
    }
    # Inject missingness for the last n_missing tickers
    for i in range(n - n_missing, n):
        data["missing_components"][i] = "sponsor"
        data["missingness_penalty"][i] = 1
    return pd.DataFrame(data)


class TestRankPortfolio:
    def test_returns_top_k(self):
        df = _make_rankings(30)
        rs = DecisionRuleset.from_json(str(PROD_BASELINE))
        result = rank_portfolio(df, rs)
        assert len(result) <= 30
        assert "_rank" in result.columns

    def test_filters_eligible_dev(self):
        df = _make_rankings(10)
        df.loc[0, "eligible"] = 0  # ineligible
        df.loc[1, "archetype"] = "commercial_biotech"  # not dev
        rs = DecisionRuleset.from_json(str(PROD_BASELINE))
        result = rank_portfolio(df, rs)
        assert "T000" not in result["ticker"].values
        assert "T001" not in result["ticker"].values


class TestCompare:
    def test_identical_rulesets_full_overlap(self):
        df = _make_rankings(30)
        rs = DecisionRuleset.from_json(str(PROD_BASELINE))
        result = compare(df, rs, rs)
        assert result["top20_overlap"] == 100.0
        assert result["top60_overlap"] == 100.0
        assert result["entrants_20"] == []
        assert result["exits_20"] == []

    def test_sort_penalty_demotes_missing(self):
        """Candidate with sort penalty should push missing-data tickers down."""
        df = _make_rankings(25, n_missing=3)
        baseline = DecisionRuleset.from_json(str(PROD_BASELINE))
        candidate = DecisionRuleset.from_json(str(PROD_CANDIDATE))

        base_sorted = rank_portfolio(df, baseline)
        cand_sorted = rank_portfolio(df, candidate)

        # Missing tickers should have higher (worse) rank in candidate
        missing_tickers = {"T022", "T023", "T024"}
        base_ranks = dict(zip(base_sorted["ticker"], base_sorted["_rank"]))
        cand_ranks = dict(zip(cand_sorted["ticker"], cand_sorted["_rank"]))

        for t in missing_tickers:
            if t in base_ranks and t in cand_ranks:
                assert cand_ranks[t] >= base_ranks[t], (
                    f"{t}: candidate rank {cand_ranks[t]} should be >= baseline {base_ranks[t]}"
                )


class TestFormatReport:
    def test_report_has_sections(self):
        result = {
            "baseline_id": "aaa",
            "candidate_id": "bbb",
            "top20_overlap": 90.0,
            "top60_overlap": 95.0,
            "entrants_20": ["X1"],
            "exits_20": ["X2"],
            "entrants_60": [],
            "exits_60": [],
            "base_top20_missing": 0,
            "cand_top20_missing": 0,
            "base_top60_missing": 1,
            "cand_top60_missing": 0,
            "mean_rank_churn_top60": 1.5,
            "max_rank_churn_top60": 3,
            "base_tier_dist_20": {"A": 15, "B": 5},
            "cand_tier_dist_20": {"A": 16, "B": 4},
            "base_eligible_count": 60,
            "cand_eligible_count": 60,
        }
        report = format_report(result, "2026-02-16")
        assert "# Ruleset Comparison" in report
        assert "## Overlap" in report
        assert "## Top-20 Changes" in report
        assert "## Top-60 Changes" in report
        assert "## Missingness in Portfolio" in report
        assert "## Tier Distribution" in report
        assert "90.0%" in report


class TestCLI:
    def test_writes_report(self, tmp_path):
        """Full CLI round-trip writes ruleset_compare.md."""
        from compare_rulesets_replay import main

        # Write a fake rankings.csv
        df = _make_rankings(25, n_missing=2)
        snap = tmp_path / "2026-02-16"
        snap.mkdir()
        df.to_csv(snap / "rankings.csv", index=False)

        rc = main([
            "--as-of-date", "2026-02-16",
            "--baseline-ruleset", str(PROD_BASELINE),
            "--candidate-ruleset", str(PROD_CANDIDATE),
            "--snapshot-dir", str(snap),
        ])
        assert rc == 0
        report_path = snap / "ruleset_compare.md"
        assert report_path.exists()
        content = report_path.read_text()
        assert "Ruleset Comparison" in content
