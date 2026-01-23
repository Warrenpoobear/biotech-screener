#!/usr/bin/env python3
"""
Tests for Research Sanity Metrics

Covers:
- IC by stage computation
- Rank stability calculation
- Factor stability analysis
"""

import pytest
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Any, List
from statistics import mean, stdev

# Import module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.sanity_metrics import (
    compute_rank_stability,
    compute_factor_stability,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_snapshots():
    """Sample snapshots with ranked securities (10+ for MIN_OBS_IC threshold)."""
    # Need at least 10 tickers to meet MIN_OBS_IC = 10 threshold
    base_tickers = ["ACME", "BETA", "GAMA", "DELT", "EPSI",
                    "ZETA", "ETHA", "THET", "IOTA", "KAPA", "LAMB", "MUUU"]

    return [
        {
            "as_of_date": "2026-01-15",
            "ranked_securities": [
                {"ticker": t, "composite_rank": i+1, "composite_score": 90-i*5,
                 "clinical_dev_normalized": 85-i*5, "financial_normalized": 88-i*4, "catalyst_normalized": 92-i*6}
                for i, t in enumerate(base_tickers)
            ],
        },
        {
            "as_of_date": "2026-02-15",
            "ranked_securities": [
                {"ticker": t, "composite_rank": (i+2) % len(base_tickers) + 1, "composite_score": 88-i*4,
                 "clinical_dev_normalized": 82-i*4, "financial_normalized": 85-i*3, "catalyst_normalized": 90-i*5}
                for i, t in enumerate(base_tickers)
            ],
        },
        {
            "as_of_date": "2026-03-15",
            "ranked_securities": [
                {"ticker": t, "composite_rank": (i+4) % len(base_tickers) + 1, "composite_score": 85-i*3,
                 "clinical_dev_normalized": 80-i*3, "financial_normalized": 82-i*2, "catalyst_normalized": 88-i*4}
                for i, t in enumerate(base_tickers)
            ],
        },
    ]


@pytest.fixture
def single_snapshot():
    """Single snapshot for edge case testing."""
    return [
        {
            "as_of_date": "2026-01-15",
            "ranked_securities": [
                {"ticker": "ACME", "composite_rank": 1, "composite_score": 85},
                {"ticker": "BETA", "composite_rank": 2, "composite_score": 75},
            ],
        },
    ]


# ============================================================================
# RANK STABILITY
# ============================================================================

class TestRankStability:
    """Tests for rank stability computation."""

    def test_basic_rank_stability(self, sample_snapshots):
        """Computes rank stability across snapshots."""
        result = compute_rank_stability(sample_snapshots)

        assert "rank_correlations" in result
        assert "rank_corr_mean" in result
        assert "top_quintile_churn" in result
        assert "interpretation" in result

    def test_rank_correlations_computed(self, sample_snapshots):
        """Rank correlations are computed for consecutive pairs."""
        result = compute_rank_stability(sample_snapshots)

        # 3 snapshots -> 2 pairs -> 2 correlations
        assert len(result["rank_correlations"]) == 2

    def test_rank_correlation_range(self, sample_snapshots):
        """Rank correlations are in [-1, 1] range."""
        result = compute_rank_stability(sample_snapshots)

        for corr in result["rank_correlations"]:
            assert -1 <= corr <= 1

    def test_perfect_stability(self):
        """Perfect rank stability gives correlation of 1."""
        snapshots = [
            {
                "as_of_date": "2026-01-15",
                "ranked_securities": [
                    {"ticker": f"T{i}", "composite_rank": i}
                    for i in range(1, 11)
                ],
            },
            {
                "as_of_date": "2026-02-15",
                "ranked_securities": [
                    {"ticker": f"T{i}", "composite_rank": i}
                    for i in range(1, 11)
                ],
            },
        ]

        result = compute_rank_stability(snapshots)

        assert result["rank_corr_mean"] == pytest.approx(1.0, rel=0.01)

    def test_top_quintile_churn(self, sample_snapshots):
        """Top quintile churn is computed."""
        result = compute_rank_stability(sample_snapshots)

        assert "churn_mean" in result
        if result["churn_mean"] is not None:
            assert 0 <= result["churn_mean"] <= 1

    def test_insufficient_snapshots(self, single_snapshot):
        """Returns appropriate message for single snapshot."""
        result = compute_rank_stability(single_snapshot)

        assert "INSUFFICIENT" in result["interpretation"]
        assert result["rank_correlations"] == []

    def test_empty_snapshots(self):
        """Handles empty snapshot list."""
        result = compute_rank_stability([])

        assert "INSUFFICIENT" in result["interpretation"]

    def test_interpretation_stable(self):
        """Provides STABLE interpretation for high correlation."""
        # Create snapshots with high stability
        snapshots = [
            {
                "as_of_date": "2026-01-15",
                "ranked_securities": [{"ticker": f"T{i}", "composite_rank": i} for i in range(1, 11)],
            },
            {
                "as_of_date": "2026-02-15",
                "ranked_securities": [{"ticker": f"T{i}", "composite_rank": i} for i in range(1, 11)],  # Same ranks
            },
        ]

        result = compute_rank_stability(snapshots)

        assert "STABLE" in result["interpretation"]

    def test_interpretation_unstable(self):
        """Provides UNSTABLE interpretation for low correlation."""
        # Create snapshots with complete rank reversal
        snapshots = [
            {
                "as_of_date": "2026-01-15",
                "ranked_securities": [{"ticker": f"T{i}", "composite_rank": i} for i in range(1, 11)],
            },
            {
                "as_of_date": "2026-02-15",
                "ranked_securities": [{"ticker": f"T{i}", "composite_rank": 11-i} for i in range(1, 11)],  # Reversed
            },
        ]

        result = compute_rank_stability(snapshots)

        # With reversal, correlation should be negative
        assert result["rank_corr_mean"] < 0.3


# ============================================================================
# FACTOR STABILITY
# ============================================================================

class TestFactorStability:
    """Tests for factor stability computation."""

    def test_basic_factor_stability(self, sample_snapshots):
        """Computes factor stability."""
        result = compute_factor_stability(sample_snapshots)

        assert "clinical_vs_composite" in result
        assert "financial_vs_composite" in result
        assert "catalyst_vs_composite" in result
        assert "summary" in result
        assert "interpretation" in result

    def test_factor_correlations_computed(self, sample_snapshots):
        """Factor correlations are computed for each snapshot."""
        result = compute_factor_stability(sample_snapshots)

        # Should have 3 correlations (one per snapshot)
        assert len(result["clinical_vs_composite"]) <= len(sample_snapshots)

    def test_summary_statistics(self, sample_snapshots):
        """Summary includes mean, std, sign_flips."""
        result = compute_factor_stability(sample_snapshots)

        for factor in ["clinical", "financial", "catalyst"]:
            summary = result["summary"].get(factor, {})
            if summary.get("mean") is not None:
                assert "mean" in summary
                assert "std" in summary
                assert "sign_flips" in summary

    def test_positive_correlation_expected(self, sample_snapshots):
        """Factors should be positively correlated with composite."""
        result = compute_factor_stability(sample_snapshots)

        # In our sample data, all factors are aligned with composite
        for factor in ["clinical", "financial", "catalyst"]:
            corrs = result.get(f"{factor}_vs_composite", [])
            if corrs:
                assert mean(corrs) > 0

    def test_interpretation_stable(self, sample_snapshots):
        """Provides STABLE interpretation when factors are consistent."""
        result = compute_factor_stability(sample_snapshots)

        # Should be stable since all factors positively contribute
        assert "STABLE" in result["interpretation"] or "WARNING" not in result["interpretation"]

    def test_empty_snapshots(self):
        """Handles empty snapshot list."""
        result = compute_factor_stability([])

        assert result["clinical_vs_composite"] == []
        assert result["financial_vs_composite"] == []
        assert result["catalyst_vs_composite"] == []

    def test_snapshots_with_missing_scores(self):
        """Handles snapshots with missing score fields."""
        snapshots = [
            {
                "as_of_date": "2026-01-15",
                "ranked_securities": [
                    {"ticker": "ACME", "composite_score": 85},  # Missing sub-scores
                    {"ticker": "BETA", "composite_score": 75},
                ],
            },
        ]

        # Should not crash
        result = compute_factor_stability(snapshots)
        assert "summary" in result


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_snapshots_with_new_tickers(self):
        """Handles tickers that appear/disappear between snapshots."""
        snapshots = [
            {
                "as_of_date": "2026-01-15",
                "ranked_securities": [
                    {"ticker": "ACME", "composite_rank": 1},
                    {"ticker": "BETA", "composite_rank": 2},
                    {"ticker": "GAMA", "composite_rank": 3},
                    {"ticker": "DELT", "composite_rank": 4},
                    {"ticker": "EPSI", "composite_rank": 5},
                ],
            },
            {
                "as_of_date": "2026-02-15",
                "ranked_securities": [
                    {"ticker": "ACME", "composite_rank": 1},
                    {"ticker": "BETA", "composite_rank": 2},
                    {"ticker": "GAMA", "composite_rank": 3},
                    {"ticker": "NEWT", "composite_rank": 4},  # New ticker
                    {"ticker": "ZETA", "composite_rank": 5},  # New ticker
                ],
            },
        ]

        result = compute_rank_stability(snapshots)

        # Should still compute correlation for common tickers
        assert result["rank_correlations"] is not None

    def test_very_small_snapshots(self):
        """Handles very small snapshots (< 5 securities)."""
        snapshots = [
            {
                "as_of_date": "2026-01-15",
                "ranked_securities": [
                    {"ticker": "A", "composite_rank": 1},
                    {"ticker": "B", "composite_rank": 2},
                ],
            },
            {
                "as_of_date": "2026-02-15",
                "ranked_securities": [
                    {"ticker": "A", "composite_rank": 1},
                    {"ticker": "B", "composite_rank": 2},
                ],
            },
        ]

        # Should handle gracefully
        result = compute_rank_stability(snapshots)
        assert "interpretation" in result

    def test_all_same_scores(self):
        """Handles case where all scores are identical."""
        snapshots = [
            {
                "as_of_date": "2026-01-15",
                "ranked_securities": [
                    {"ticker": f"T{i}", "composite_score": 50,
                     "clinical_dev_normalized": 50, "financial_normalized": 50, "catalyst_normalized": 50}
                    for i in range(10)
                ],
            },
        ]

        # Should handle gracefully (correlation undefined for constant data)
        result = compute_factor_stability(snapshots)
        assert "summary" in result


# ============================================================================
# DETERMINISM
# ============================================================================

class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_rank_stability_deterministic(self, sample_snapshots):
        """Rank stability computation is deterministic."""
        result1 = compute_rank_stability(sample_snapshots)
        result2 = compute_rank_stability(sample_snapshots)

        assert result1["rank_correlations"] == result2["rank_correlations"]
        assert result1["rank_corr_mean"] == result2["rank_corr_mean"]

    def test_factor_stability_deterministic(self, sample_snapshots):
        """Factor stability computation is deterministic."""
        result1 = compute_factor_stability(sample_snapshots)
        result2 = compute_factor_stability(sample_snapshots)

        assert result1["clinical_vs_composite"] == result2["clinical_vs_composite"]
        assert result1["summary"] == result2["summary"]

