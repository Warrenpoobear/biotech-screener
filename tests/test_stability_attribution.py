#!/usr/bin/env python3
"""
Tests for Stability Attribution Panel

Covers:
- Monthly delta computation
- Aggregate attribution
- Instability diagnosis
"""

import pytest
from datetime import date
from pathlib import Path
from typing import Dict, Any, List
from statistics import mean

# Import module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.stability_attribution import (
    compute_monthly_deltas,
    compute_stability_attribution,
    diagnose_instability,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def snapshot_pair():
    """Pair of consecutive snapshots."""
    return (
        {
            "as_of_date": "2026-01-15",
            "ranked_securities": [
                {"ticker": "ACME", "composite_rank": 1, "composite_score": 85,
                 "clinical_dev_normalized": 80, "financial_normalized": 85,
                 "catalyst_normalized": 90, "uncertainty_penalty": 0.05, "severity": "none"},
                {"ticker": "BETA", "composite_rank": 2, "composite_score": 75,
                 "clinical_dev_normalized": 70, "financial_normalized": 80,
                 "catalyst_normalized": 75, "uncertainty_penalty": 0.10, "severity": "none"},
                {"ticker": "GAMA", "composite_rank": 3, "composite_score": 65,
                 "clinical_dev_normalized": 60, "financial_normalized": 70,
                 "catalyst_normalized": 65, "uncertainty_penalty": 0.15, "severity": "sev1"},
            ],
        },
        {
            "as_of_date": "2026-02-15",
            "ranked_securities": [
                {"ticker": "ACME", "composite_rank": 1, "composite_score": 90,  # Increased
                 "clinical_dev_normalized": 85, "financial_normalized": 88,
                 "catalyst_normalized": 97, "uncertainty_penalty": 0.03, "severity": "none"},
                {"ticker": "BETA", "composite_rank": 3, "composite_score": 70,  # Dropped
                 "clinical_dev_normalized": 65, "financial_normalized": 75,
                 "catalyst_normalized": 70, "uncertainty_penalty": 0.12, "severity": "sev1"},  # Severity changed
                {"ticker": "GAMA", "composite_rank": 2, "composite_score": 72,  # Rose
                 "clinical_dev_normalized": 68, "financial_normalized": 76,
                 "catalyst_normalized": 72, "uncertainty_penalty": 0.10, "severity": "none"},  # Severity improved
            ],
        },
    )


@pytest.fixture
def sample_snapshots(snapshot_pair):
    """Multiple snapshots for comprehensive testing."""
    snap1, snap2 = snapshot_pair
    snap3 = {
        "as_of_date": "2026-03-15",
        "ranked_securities": [
            {"ticker": "ACME", "composite_rank": 2, "composite_score": 82,
             "clinical_dev_normalized": 78, "financial_normalized": 82,
             "catalyst_normalized": 86, "uncertainty_penalty": 0.08, "severity": "none"},
            {"ticker": "BETA", "composite_rank": 1, "composite_score": 88,
             "clinical_dev_normalized": 82, "financial_normalized": 90,
             "catalyst_normalized": 92, "uncertainty_penalty": 0.05, "severity": "none"},
            {"ticker": "GAMA", "composite_rank": 3, "composite_score": 68,
             "clinical_dev_normalized": 62, "financial_normalized": 72,
             "catalyst_normalized": 70, "uncertainty_penalty": 0.12, "severity": "sev1"},
        ],
    }
    return [snap1, snap2, snap3]


@pytest.fixture
def rank_stability_results():
    """Sample rank stability results for diagnosis."""
    return {
        "rank_corr_mean": 0.4,
        "churn_mean": 0.3,
    }


# ============================================================================
# MONTHLY DELTAS
# ============================================================================

class TestMonthlyDeltas:
    """Tests for compute_monthly_deltas function."""

    def test_basic_delta_computation(self, snapshot_pair):
        """Computes basic deltas between snapshots."""
        snap_prev, snap_curr = snapshot_pair
        result = compute_monthly_deltas(snap_prev, snap_curr)

        assert result["date_prev"] == "2026-01-15"
        assert result["date_curr"] == "2026-02-15"
        assert "ticker_deltas" in result
        assert "top_movers" in result
        assert "aggregate_attribution" in result

    def test_ticker_deltas_computed(self, snapshot_pair):
        """Computes deltas for each ticker."""
        snap_prev, snap_curr = snapshot_pair
        result = compute_monthly_deltas(snap_prev, snap_curr)

        assert len(result["ticker_deltas"]) == 3  # All 3 tickers common

        for delta in result["ticker_deltas"]:
            assert "ticker" in delta
            assert "Δcomposite" in delta
            assert "Δclinical_norm" in delta
            assert "Δfinancial_norm" in delta
            assert "Δcatalyst_norm" in delta

    def test_delta_values_correct(self, snapshot_pair):
        """Delta values are mathematically correct."""
        snap_prev, snap_curr = snapshot_pair
        result = compute_monthly_deltas(snap_prev, snap_curr)

        acme_delta = next(d for d in result["ticker_deltas"] if d["ticker"] == "ACME")

        # ACME: 90 - 85 = 5
        assert acme_delta["Δcomposite"] == pytest.approx(5.0, rel=0.01)

    def test_rank_changes_tracked(self, snapshot_pair):
        """Rank changes are tracked."""
        snap_prev, snap_curr = snapshot_pair
        result = compute_monthly_deltas(snap_prev, snap_curr)

        beta_delta = next(d for d in result["ticker_deltas"] if d["ticker"] == "BETA")

        # BETA dropped from rank 2 to 3
        assert beta_delta["rank_prev"] == 2
        assert beta_delta["rank_curr"] == 3
        assert beta_delta["Δrank"] == 1

    def test_severity_changes_tracked(self, snapshot_pair):
        """Severity changes are tracked."""
        snap_prev, snap_curr = snapshot_pair
        result = compute_monthly_deltas(snap_prev, snap_curr)

        beta_delta = next(d for d in result["ticker_deltas"] if d["ticker"] == "BETA")
        gama_delta = next(d for d in result["ticker_deltas"] if d["ticker"] == "GAMA")

        # BETA: none -> sev1
        assert beta_delta["severity_prev"] == "none"
        assert beta_delta["severity_curr"] == "sev1"
        assert beta_delta["severity_changed"] is True

        # GAMA: sev1 -> none
        assert gama_delta["severity_prev"] == "sev1"
        assert gama_delta["severity_curr"] == "none"
        assert gama_delta["severity_changed"] is True

    def test_top_movers_sorted(self, snapshot_pair):
        """Top movers are sorted by absolute delta."""
        snap_prev, snap_curr = snapshot_pair
        result = compute_monthly_deltas(snap_prev, snap_curr)

        top_movers = result["top_movers"]
        for i in range(1, len(top_movers)):
            assert abs(top_movers[i-1]["Δcomposite"]) >= abs(top_movers[i]["Δcomposite"])

    def test_aggregate_attribution(self, snapshot_pair):
        """Aggregate attribution percentages sum reasonably."""
        snap_prev, snap_curr = snapshot_pair
        result = compute_monthly_deltas(snap_prev, snap_curr)

        attr = result["aggregate_attribution"]
        assert "clinical_pct" in attr
        assert "financial_pct" in attr
        assert "catalyst_pct" in attr

    def test_new_tickers_skipped(self):
        """New tickers (not in prev) are skipped."""
        snap_prev = {
            "as_of_date": "2026-01-15",
            "ranked_securities": [
                {"ticker": "ACME", "composite_score": 85},
            ],
        }
        snap_curr = {
            "as_of_date": "2026-02-15",
            "ranked_securities": [
                {"ticker": "ACME", "composite_score": 90},
                {"ticker": "NEWT", "composite_score": 70},  # New ticker
            ],
        }

        result = compute_monthly_deltas(snap_prev, snap_curr)

        # Only ACME should have delta
        assert result["n_common_tickers"] == 1
        assert result["ticker_deltas"][0]["ticker"] == "ACME"


# ============================================================================
# STABILITY ATTRIBUTION
# ============================================================================

class TestStabilityAttribution:
    """Tests for compute_stability_attribution function."""

    def test_basic_attribution(self, sample_snapshots):
        """Computes attribution across all snapshots."""
        result = compute_stability_attribution(sample_snapshots)

        assert "monthly_panels" in result
        assert "summary" in result
        assert "global_top_movers" in result

    def test_monthly_panels_count(self, sample_snapshots):
        """Creates correct number of monthly panels."""
        result = compute_stability_attribution(sample_snapshots)

        # 3 snapshots -> 2 consecutive pairs -> 2 panels
        assert len(result["monthly_panels"]) == 2

    def test_summary_computed(self, sample_snapshots):
        """Summary statistics are computed."""
        result = compute_stability_attribution(sample_snapshots)

        summary = result["summary"]
        assert "n_periods" in summary
        assert "mean_attribution" in summary
        assert "dominant_driver" in summary
        assert "interpretation" in summary

    def test_dominant_driver_identified(self, sample_snapshots):
        """Dominant driver is identified."""
        result = compute_stability_attribution(sample_snapshots)

        driver = result["summary"]["dominant_driver"]
        assert driver in ["clinical", "financial", "catalyst", "uncertainty", "severity", None]

    def test_global_top_movers_limited(self, sample_snapshots):
        """Global top movers are limited to top 20."""
        result = compute_stability_attribution(sample_snapshots)

        assert len(result["global_top_movers"]) <= 20

    def test_insufficient_snapshots(self):
        """Returns error for single snapshot."""
        single = [{"as_of_date": "2026-01-15", "ranked_securities": []}]
        result = compute_stability_attribution(single)

        assert "error" in result

    def test_empty_snapshots(self):
        """Handles empty snapshot list."""
        result = compute_stability_attribution([])

        assert "error" in result


# ============================================================================
# INSTABILITY DIAGNOSIS
# ============================================================================

class TestInstabilityDiagnosis:
    """Tests for diagnose_instability function."""

    def test_basic_diagnosis(self, sample_snapshots, rank_stability_results):
        """Produces diagnosis output."""
        attribution = compute_stability_attribution(sample_snapshots)
        diagnosis = diagnose_instability(attribution, rank_stability_results)

        assert "primary_cause" in diagnosis
        assert "confidence" in diagnosis
        assert "evidence" in diagnosis
        assert "recommendations" in diagnosis

    def test_uncertainty_driven(self):
        """Diagnoses uncertainty-driven instability."""
        attribution = {
            "summary": {
                "mean_attribution": {
                    "clinical_pct": 20,
                    "financial_pct": 15,
                    "catalyst_pct": 10,
                    "uncertainty_pct": 40,  # High
                    "severity_pct": 15,
                },
            },
        }
        rank_stability = {"rank_corr_mean": 0.3, "churn_mean": 0.5}

        diagnosis = diagnose_instability(attribution, rank_stability)

        assert diagnosis["primary_cause"] == "missingness_penalty"
        # Evidence mentions "Uncertainty penalty" which relates to missingness
        assert any("uncertainty" in e.lower() for e in diagnosis["evidence"])

    def test_severity_driven(self):
        """Diagnoses severity-driven instability."""
        attribution = {
            "summary": {
                "mean_attribution": {
                    "clinical_pct": 20,
                    "financial_pct": 15,
                    "catalyst_pct": 10,
                    "uncertainty_pct": 10,
                    "severity_pct": 45,  # High
                },
            },
        }
        rank_stability = {"rank_corr_mean": 0.3, "churn_mean": 0.5}

        diagnosis = diagnose_instability(attribution, rank_stability)

        assert diagnosis["primary_cause"] == "severity_gates"

    def test_real_signal_diagnosis(self):
        """Diagnoses real signal as primary cause."""
        attribution = {
            "summary": {
                "mean_attribution": {
                    "clinical_pct": 40,
                    "financial_pct": 35,
                    "catalyst_pct": 20,
                    "uncertainty_pct": 3,
                    "severity_pct": 2,
                },
            },
        }
        rank_stability = {"rank_corr_mean": 0.5, "churn_mean": 0.3}  # Stable

        diagnosis = diagnose_instability(attribution, rank_stability)

        assert diagnosis["primary_cause"] == "real_signal"
        assert diagnosis["confidence"] == "medium"

    def test_noisy_signal_diagnosis(self):
        """Diagnoses noisy signal when sub-scores dominate but stability is low."""
        attribution = {
            "summary": {
                "mean_attribution": {
                    "clinical_pct": 40,
                    "financial_pct": 35,
                    "catalyst_pct": 20,
                    "uncertainty_pct": 3,
                    "severity_pct": 2,
                },
            },
        }
        rank_stability = {"rank_corr_mean": 0.2, "churn_mean": 0.7}  # Unstable

        diagnosis = diagnose_instability(attribution, rank_stability)

        assert diagnosis["primary_cause"] == "real_signal_or_noise"
        assert diagnosis["confidence"] == "low"

    def test_unknown_diagnosis(self):
        """Returns unknown when no clear pattern."""
        attribution = {
            "summary": {
                "mean_attribution": {},
            },
        }
        rank_stability = {"rank_corr_mean": None, "churn_mean": None}

        diagnosis = diagnose_instability(attribution, rank_stability)

        assert diagnosis["primary_cause"] == "unknown"


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_missing_score_fields(self):
        """Handles snapshots with missing score fields."""
        snap_prev = {
            "as_of_date": "2026-01-15",
            "ranked_securities": [
                {"ticker": "ACME", "composite_score": 85},  # Missing sub-scores
            ],
        }
        snap_curr = {
            "as_of_date": "2026-02-15",
            "ranked_securities": [
                {"ticker": "ACME", "composite_score": 90},
            ],
        }

        result = compute_monthly_deltas(snap_prev, snap_curr)

        # Should compute composite delta even without sub-scores
        assert result["ticker_deltas"][0]["Δcomposite"] == pytest.approx(5.0, rel=0.01)

    def test_string_scores(self):
        """Handles scores stored as strings."""
        snap_prev = {
            "as_of_date": "2026-01-15",
            "ranked_securities": [
                {"ticker": "ACME", "composite_score": "85.5"},
            ],
        }
        snap_curr = {
            "as_of_date": "2026-02-15",
            "ranked_securities": [
                {"ticker": "ACME", "composite_score": "90.5"},
            ],
        }

        result = compute_monthly_deltas(snap_prev, snap_curr)

        assert result["ticker_deltas"][0]["Δcomposite"] == pytest.approx(5.0, rel=0.01)

    def test_none_scores(self):
        """Handles None scores gracefully."""
        snap_prev = {
            "as_of_date": "2026-01-15",
            "ranked_securities": [
                {"ticker": "ACME", "composite_score": None},
            ],
        }
        snap_curr = {
            "as_of_date": "2026-02-15",
            "ranked_securities": [
                {"ticker": "ACME", "composite_score": 90},
            ],
        }

        result = compute_monthly_deltas(snap_prev, snap_curr)

        # None should be treated as 0
        assert result["ticker_deltas"][0]["Δcomposite"] == pytest.approx(90.0, rel=0.01)


# ============================================================================
# DETERMINISM
# ============================================================================

class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_monthly_deltas_deterministic(self, snapshot_pair):
        """Monthly delta computation is deterministic."""
        snap_prev, snap_curr = snapshot_pair

        result1 = compute_monthly_deltas(snap_prev, snap_curr)
        result2 = compute_monthly_deltas(snap_prev, snap_curr)

        assert result1["ticker_deltas"] == result2["ticker_deltas"]
        assert result1["aggregate_attribution"] == result2["aggregate_attribution"]

    def test_stability_attribution_deterministic(self, sample_snapshots):
        """Stability attribution is deterministic."""
        result1 = compute_stability_attribution(sample_snapshots)
        result2 = compute_stability_attribution(sample_snapshots)

        assert result1["summary"] == result2["summary"]

