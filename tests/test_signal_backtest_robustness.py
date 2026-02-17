"""Tests for scripts/backtest_signal_robustness.py helpers."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from backtest_signal_robustness import (
    ALPHA_CLIP_MAX,
    ALPHA_CLIP_MIN,
    DEFAULT_SHRINK_K,
    _avg_ranks,
    _extract_catalyst,
    _extract_clinical,
    build_rolling_alpha_table,
    compute_spread,
    score_alpha_oos,
    spearman_rank_corr,
)


# ---------------------------------------------------------------------------
# Spearman rank correlation
# ---------------------------------------------------------------------------

class TestSpearmanRankCorr:
    def test_rank_corr_perfect(self):
        ic = spearman_rank_corr([1, 2, 3, 4], [10, 20, 30, 40])
        assert abs(ic - 1.0) < 1e-9

    def test_rank_corr_constant(self):
        ic = spearman_rank_corr([5, 5, 5], [1, 2, 3])
        assert ic == 0.0

    def test_rank_corr_inverse(self):
        ic = spearman_rank_corr([4, 3, 2, 1], [10, 20, 30, 40])
        assert abs(ic - (-1.0)) < 1e-9

    def test_rank_corr_too_few(self):
        assert spearman_rank_corr([1, 2], [3, 4]) == 0.0
        assert spearman_rank_corr([], []) == 0.0

    def test_rank_corr_ties(self):
        # With ties: [1,1,3] vs [10,20,30] — still positive
        ic = spearman_rank_corr([1, 1, 3], [10, 20, 30])
        assert ic > 0.0


# ---------------------------------------------------------------------------
# avg_ranks
# ---------------------------------------------------------------------------

class TestAvgRanks:
    def test_no_ties(self):
        assert _avg_ranks([10, 20, 30]) == [1.0, 2.0, 3.0]

    def test_all_tied(self):
        assert _avg_ranks([5, 5, 5]) == [2.0, 2.0, 2.0]

    def test_partial_ties(self):
        ranks = _avg_ranks([10, 20, 20, 30])
        assert ranks[0] == 1.0
        assert ranks[1] == 2.5
        assert ranks[2] == 2.5
        assert ranks[3] == 4.0


# ---------------------------------------------------------------------------
# Catalyst signal extraction
# ---------------------------------------------------------------------------

class TestCatalystSignal:
    def test_specific_days(self):
        row = {"catalyst_mode": "specific_days", "catalyst_days": "60"}
        sig = _extract_catalyst(row)
        assert sig is not None
        assert abs(sig - 1.0 / 61.0) < 1e-9

    def test_no_upcoming(self):
        row = {"catalyst_mode": "no_upcoming", "catalyst_days": ""}
        assert _extract_catalyst(row) == 0.0

    def test_missing_days(self):
        row = {"catalyst_mode": "specific_days", "catalyst_days": ""}
        assert _extract_catalyst(row) == 0.0

    def test_blended_window(self):
        row = {"catalyst_mode": "blended_window", "catalyst_days": "30"}
        sig = _extract_catalyst(row)
        assert sig is not None
        assert abs(sig - 1.0 / 31.0) < 1e-9

    def test_zero_days(self):
        row = {"catalyst_mode": "specific_days", "catalyst_days": "0"}
        sig = _extract_catalyst(row)
        assert sig is not None
        assert abs(sig - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Clinical signal extraction
# ---------------------------------------------------------------------------

class TestClinicalSignal:
    def test_dd_with_value(self):
        row = {"archetype": "drug_developer", "clinical_score_z_tier": "1.5"}
        assert _extract_clinical(row) == 1.5

    def test_non_dd_returns_none(self):
        row = {"archetype": "commercial_biotech", "clinical_score_z_tier": "1.5"}
        assert _extract_clinical(row) is None

    def test_empty_value(self):
        row = {"archetype": "drug_developer", "clinical_score_z_tier": ""}
        assert _extract_clinical(row) is None


# ---------------------------------------------------------------------------
# Spread computation
# ---------------------------------------------------------------------------

class TestSpread:
    def test_spread_positive_toy(self):
        # 25 items: top 5 have high signal + positive excess, bottom 5 negative
        signals = list(range(25, 0, -1))  # 25 down to 1
        excess = [0.10] * 5 + [0.01] * 15 + [-0.10] * 5
        spread = compute_spread(
            [float(s) for s in signals],
            [float(e) for e in excess],
        )
        assert spread > 0.0

    def test_spread_too_few_rows(self):
        assert compute_spread([1.0, 2.0, 3.0], [0.1, 0.2, 0.3]) == 0.0

    def test_spread_exact_quintile(self):
        # 10 items, quintile = 2
        signals = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        excess = [0.20, 0.15, 0.05, 0.05, 0.0, 0.0, -0.05, -0.05, -0.15, -0.20]
        # Too few per quintile (10//5=2, < 5) → 0.0
        assert compute_spread(signals, excess) == 0.0

    def test_spread_min_quintile_met(self):
        # 25 items, quintile = 5 (meets default min_per_quintile=5)
        n = 25
        signals = [float(i) for i in range(n, 0, -1)]
        excess = [0.10 if i < 5 else (-0.10 if i >= 20 else 0.0) for i in range(n)]
        spread = compute_spread(signals, excess, min_per_quintile=5)
        assert spread > 0.0


# ---------------------------------------------------------------------------
# Alpha OOS: shrinkage + clipping
# ---------------------------------------------------------------------------

class TestAlphaShrinkage:
    def _make_train_entry(self, rows, excess):
        """Helper to build a train cache entry."""
        return {"rows": rows, "excess": excess}

    def test_alpha_shrinkage_clips(self):
        """Shrinkage moves mean toward 0; clip bounds [-0.10, 0.10] apply."""
        # One cell with extreme positive mean and high n
        rows = []
        excess = {}
        for i in range(200):
            tk = f"T{i:03d}"
            rows.append({
                "ticker": tk,
                "stage_bucket": "mid",
                "catalyst_mode": "specific_days",
                "catalyst_days": "50",
                "clinical_score_z_tier": "1.0",
            })
            excess[tk] = 0.50  # extreme 50% excess return

        train = [self._make_train_entry(rows, excess)]
        table = build_rolling_alpha_table(train)

        # Cell mean = 0.50, n=200, shrink_k=50 → w=200/250=0.8, raw=0.40
        # Clip to 0.10
        test_row = [{"ticker": "TEST", "stage_bucket": "mid",
                     "catalyst_mode": "specific_days", "catalyst_days": "50",
                     "clinical_score_z_tier": "1.0"}]
        scores = score_alpha_oos(test_row, table)
        assert scores["TEST"] == ALPHA_CLIP_MAX  # clipped to 0.10

    def test_alpha_shrinkage_small_n(self):
        """Small n → heavy shrinkage toward 0."""
        rows = [{"ticker": "A", "stage_bucket": "early",
                 "catalyst_mode": "no_upcoming", "catalyst_days": "",
                 "clinical_score_z_tier": "0.0"}]
        excess = {"A": 0.10}
        train = [self._make_train_entry(rows, excess)]
        table = build_rolling_alpha_table(train)

        # Cell mean = 0.10, n=1, shrink_k=50 → w=1/51≈0.0196, raw≈0.00196
        scores = score_alpha_oos(rows, table)
        assert abs(scores["A"]) < 0.003  # heavily shrunk

    def test_alpha_unseen_key_returns_zero(self):
        """Unseen cohort key → alpha=0.0."""
        # Train has mid|near_31_90|pos, eval has early|none|nonpos
        train_rows = [{"ticker": "X", "stage_bucket": "mid",
                       "catalyst_mode": "specific_days", "catalyst_days": "50",
                       "clinical_score_z_tier": "1.0"}]
        train = [self._make_train_entry(train_rows, {"X": 0.05})]
        table = build_rolling_alpha_table(train)

        eval_rows = [{"ticker": "Y", "stage_bucket": "early",
                      "catalyst_mode": "no_upcoming", "catalyst_days": "",
                      "clinical_score_z_tier": "-1.0"}]
        scores = score_alpha_oos(eval_rows, table)
        assert scores["Y"] == 0.0

    def test_alpha_negative_clips(self):
        """Negative excess → shrunk negative alpha, clipped at -0.10."""
        rows = []
        excess = {}
        for i in range(200):
            tk = f"N{i:03d}"
            rows.append({
                "ticker": tk,
                "stage_bucket": "late",
                "catalyst_mode": "specific_days",
                "catalyst_days": "10",
                "clinical_score_z_tier": "-0.5",
            })
            excess[tk] = -0.50

        train = [self._make_train_entry(rows, excess)]
        table = build_rolling_alpha_table(train)

        test_row = [{"ticker": "TEST", "stage_bucket": "late",
                     "catalyst_mode": "specific_days", "catalyst_days": "10",
                     "clinical_score_z_tier": "-0.5"}]
        scores = score_alpha_oos(test_row, table)
        assert scores["TEST"] == ALPHA_CLIP_MIN  # clipped to -0.10


class TestAlphaOOSRequiresTrain:
    def test_insufficient_train_produces_empty_table(self):
        """With zero train entries, all scores should be 0.0."""
        table = build_rolling_alpha_table([])
        assert table["cells"] == {}

        test_rows = [{"ticker": "X", "stage_bucket": "mid",
                      "catalyst_mode": "specific_days", "catalyst_days": "50",
                      "clinical_score_z_tier": "1.0"}]
        scores = score_alpha_oos(test_rows, table)
        assert scores["X"] == 0.0
