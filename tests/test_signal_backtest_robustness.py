"""Tests for scripts/backtest_signal_robustness.py helpers."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import math
import pytest

from backtest_signal_robustness import (
    ALPHA_CLIP_MAX,
    ALPHA_CLIP_MIN,
    DEFAULT_SHRINK_K,
    _avg_ranks,
    _build_summary,
    _extract_catalyst,
    _extract_clinical,
    build_rolling_alpha_table,
    build_weighted_alpha_table,
    compute_cell_diagnostics,
    compute_double_sort_spread,
    compute_spread,
    parse_train_mode,
    residualize_ranks,
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


# ---------------------------------------------------------------------------
# Residualize ranks
# ---------------------------------------------------------------------------

class TestResidualizeRanks:
    def test_residualize_identical(self):
        """If x == z (perfectly correlated), residuals should have zero
        correlation with z."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        z = list(x)  # identical
        resid = residualize_ranks(x, z)
        # Residuals should be all zero (or near-zero)
        assert all(abs(r) < 1e-9 for r in resid)
        # Correlation with z should be zero
        ic = spearman_rank_corr(resid, z)
        assert abs(ic) < 1e-6

    def test_residualize_orthogonal(self):
        """If x and z are independent/orthogonal, residuals ≈ x ranks."""
        # x ascending, z a shuffled pattern uncorrelated with x
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        z = [5.0, 1.0, 8.0, 3.0, 10.0, 2.0, 7.0, 4.0, 9.0, 6.0]
        resid = residualize_ranks(x, z)
        # Residuals should still correlate with the original x
        ic_resid_x = spearman_rank_corr(resid, x)
        assert ic_resid_x > 0.5  # strong residual correlation preserved

    def test_residualize_too_few(self):
        """n < 3 → returns x ranks unchanged."""
        x = [3.0, 1.0]
        z = [10.0, 20.0]
        resid = residualize_ranks(x, z)
        ranks = _avg_ranks(x)
        assert resid == ranks


# ---------------------------------------------------------------------------
# Double-sort spread
# ---------------------------------------------------------------------------

class TestDoubleSortSpread:
    def test_double_sort_positive(self):
        """Toy data where alpha separates within each catalyst tercile
        → spread should be positive."""
        n = 60
        # sort1 (catalyst): 3 groups of 20
        sort1 = [float(i) for i in range(n)]
        # sort2 (alpha): within each group, top half has positive excess
        sort2 = []
        excess = []
        for g in range(3):
            for i in range(20):
                sort2.append(float(20 - i))  # descending within group
                # Top 10 in each group get +0.10, bottom 10 get -0.10
                excess.append(0.10 if i < 10 else -0.10)
        spread = compute_double_sort_spread(sort1, sort2, excess, n_groups=3, min_per_group=10)
        assert spread > 0.0
        # Expected: each tercile spread = 0.10 - (-0.10) = 0.20, avg = 0.20
        assert abs(spread - 0.20) < 1e-9

    def test_double_sort_too_few(self):
        """Small n → returns 0.0."""
        sort1 = [1.0, 2.0, 3.0]
        sort2 = [3.0, 2.0, 1.0]
        excess = [0.1, 0.0, -0.1]
        spread = compute_double_sort_spread(sort1, sort2, excess, n_groups=3, min_per_group=10)
        assert spread == 0.0


# ---------------------------------------------------------------------------
# Cell diagnostics
# ---------------------------------------------------------------------------

def _make_entry(date, rows, excess):
    return {"date": date, "rows": rows, "excess": excess}


def _make_rows_excess(tickers, excess_vals, stage="mid", cat_mode="specific_days",
                      cat_days="50", cz="1.0"):
    """Build matching rows + excess dict for a list of tickers."""
    rows = [
        {"ticker": tk, "stage_bucket": stage, "catalyst_mode": cat_mode,
         "catalyst_days": cat_days, "clinical_score_z_tier": cz}
        for tk in tickers
    ]
    excess = dict(zip(tickers, excess_vals))
    return rows, excess


class TestCellDiagnostics:
    def test_cell_diagnostics_basic(self):
        """Pre/post means and delta computed correctly; sorted ascending."""
        tickers = ["A", "B"]
        # Pre period: both tickers have excess 0.10
        rows1, exc1 = _make_rows_excess(tickers, [0.10, 0.10])
        # Post period: both tickers have excess -0.05
        rows2, exc2 = _make_rows_excess(tickers, [-0.05, -0.05])
        cache = [
            _make_entry("2024-06-01", rows1, exc1),
            _make_entry("2025-06-01", rows2, exc2),
        ]
        diags = compute_cell_diagnostics(cache, cutoff="2025-01-01")
        assert len(diags) == 1  # all rows map to the same cell key
        d = diags[0]
        assert abs(d["mean_excess_pre"] - 0.10) < 1e-9
        assert abs(d["mean_excess_post"] - (-0.05)) < 1e-9
        assert abs(d["delta"] - (-0.15)) < 1e-9
        assert d["n_pre"] == 2
        assert d["n_post"] == 2

    def test_cell_diagnostics_one_period_only(self):
        """Cell with only pre data → post NaN, delta NaN, sorted to end."""
        tickers = ["A"]
        rows1, exc1 = _make_rows_excess(tickers, [0.05])
        # Second entry also pre (before cutoff)
        rows2, exc2 = _make_rows_excess(tickers, [0.03], stage="early")
        cache = [
            _make_entry("2024-03-01", rows1, exc1),
            _make_entry("2024-06-01", rows2, exc2),
        ]
        diags = compute_cell_diagnostics(cache, cutoff="2025-01-01")
        # Two cells: mid|...|pos and early|...|nonpos — both pre-only
        for d in diags:
            assert math.isnan(d["delta"])
            assert d["n_post"] == 0

    def test_cell_diagnostics_empty_cache(self):
        """Empty input → empty output."""
        assert compute_cell_diagnostics([]) == []

    def test_cell_diagnostics_all_same_period(self):
        """All dates before cutoff → post means are NaN."""
        tickers = ["X"]
        rows, exc = _make_rows_excess(tickers, [0.02])
        cache = [_make_entry("2024-01-01", rows, exc)]
        diags = compute_cell_diagnostics(cache, cutoff="2025-01-01")
        assert len(diags) == 1
        assert math.isnan(diags[0]["mean_excess_post"])
        assert math.isnan(diags[0]["delta"])


# ---------------------------------------------------------------------------
# Regime label propagation
# ---------------------------------------------------------------------------

class TestRegime:
    def test_regime_label_propagation(self):
        """ic_rows with regime → summary has regime stats."""
        ic_rows = [
            {"date": "2024-06-01", "regime": "BULL",
             "ic_clinical": 0.10, "ic_catalyst": 0.05, "ic_alpha": 0.08,
             "ic_alpha_incr": 0.03},
            {"date": "2025-06-01", "regime": "BEAR",
             "ic_clinical": -0.05, "ic_catalyst": -0.02, "ic_alpha": -0.04,
             "ic_alpha_incr": -0.01},
        ]
        spread_rows = [
            {"date": "2024-06-01", "regime": "BULL",
             "spread_clinical": 0.02, "spread_catalyst": 0.01, "spread_alpha": 0.03,
             "spread_alpha_double": 0.01},
            {"date": "2025-06-01", "regime": "BEAR",
             "spread_clinical": -0.01, "spread_catalyst": -0.005, "spread_alpha": -0.02,
             "spread_alpha_double": -0.005},
        ]
        summary = _build_summary(ic_rows, spread_rows, horizon=126, alpha_skipped=0)
        assert "regime" in summary
        assert "BULL" in summary["regime"]
        assert "BEAR" in summary["regime"]
        bull = summary["regime"]["BULL"]
        assert bull["clinical"]["mean_ic"] == 0.10
        assert bull["clinical"]["n"] == 1

    def test_regime_empty_when_not_used(self):
        """No regime labels → no regime key in summary."""
        ic_rows = [
            {"date": "2024-06-01", "regime": "",
             "ic_clinical": 0.10, "ic_catalyst": 0.05, "ic_alpha": 0.08,
             "ic_alpha_incr": 0.03},
        ]
        spread_rows = [
            {"date": "2024-06-01", "regime": "",
             "spread_clinical": 0.02, "spread_catalyst": 0.01, "spread_alpha": 0.03,
             "spread_alpha_double": 0.01},
        ]
        summary = _build_summary(ic_rows, spread_rows, horizon=126, alpha_skipped=0)
        assert "regime" not in summary


# ---------------------------------------------------------------------------
# Training mode parsing
# ---------------------------------------------------------------------------

class TestParseTrainMode:
    def test_parse_expanding(self):
        assert parse_train_mode("expanding") == ("expanding", None)

    def test_parse_trailing(self):
        assert parse_train_mode("trailing-6") == ("trailing", 6)

    def test_parse_decay(self):
        assert parse_train_mode("decay-3") == ("decay", 3)

    def test_parse_invalid(self):
        with pytest.raises(ValueError):
            parse_train_mode("bogus")
        with pytest.raises(ValueError):
            parse_train_mode("trailing-abc")
        with pytest.raises(ValueError):
            parse_train_mode("decay-0")


# ---------------------------------------------------------------------------
# Weighted alpha table
# ---------------------------------------------------------------------------

class TestWeightedAlphaTable:
    def _make_train_entry(self, rows, excess):
        return {"rows": rows, "excess": excess}

    def test_weighted_uniform_matches_unweighted(self):
        """All weights=1.0 matches build_rolling_alpha_table()."""
        rows = [
            {"ticker": "A", "stage_bucket": "mid", "catalyst_mode": "specific_days",
             "catalyst_days": "50", "clinical_score_z_tier": "1.0"},
            {"ticker": "B", "stage_bucket": "mid", "catalyst_mode": "specific_days",
             "catalyst_days": "50", "clinical_score_z_tier": "1.0"},
        ]
        exc1 = {"A": 0.10, "B": 0.20}
        exc2 = {"A": 0.05, "B": 0.15}
        cache = [
            self._make_train_entry(rows, exc1),
            self._make_train_entry(rows, exc2),
        ]
        rolling = build_rolling_alpha_table(cache, shrink_k=50)
        weighted = build_weighted_alpha_table(cache, [1.0, 1.0], shrink_k=50)
        # Cell means and n should match
        for key in rolling["cells"]:
            assert abs(rolling["cells"][key]["mean_excess_ret_6m"]
                       - weighted["cells"][key]["mean_excess_ret_6m"]) < 1e-6
            assert abs(rolling["cells"][key]["n"]
                       - weighted["cells"][key]["n"]) < 1e-6

    def test_weighted_decay_weights(self):
        """Known weights → correct weighted mean + effective n."""
        rows = [
            {"ticker": "A", "stage_bucket": "mid", "catalyst_mode": "specific_days",
             "catalyst_days": "50", "clinical_score_z_tier": "1.0"},
        ]
        exc1 = {"A": 0.10}  # weight 0.5
        exc2 = {"A": 0.20}  # weight 1.0
        cache = [
            self._make_train_entry(rows, exc1),
            self._make_train_entry(rows, exc2),
        ]
        table = build_weighted_alpha_table(cache, [0.5, 1.0], shrink_k=50)
        # Weighted mean = (0.5*0.10 + 1.0*0.20) / (0.5+1.0) = 0.25/1.5 = 0.1667
        cell = list(table["cells"].values())[0]
        assert abs(cell["mean_excess_ret_6m"] - (0.25 / 1.5)) < 1e-5
        # Effective n = 0.5 + 1.0 = 1.5
        assert abs(cell["n"] - 1.5) < 1e-9

    def test_weighted_zero_weight(self):
        """Weight=0.0 contributes nothing."""
        rows = [
            {"ticker": "A", "stage_bucket": "mid", "catalyst_mode": "specific_days",
             "catalyst_days": "50", "clinical_score_z_tier": "1.0"},
        ]
        exc1 = {"A": 0.10}  # weight 0.0 → ignored
        exc2 = {"A": 0.20}  # weight 1.0
        cache = [
            self._make_train_entry(rows, exc1),
            self._make_train_entry(rows, exc2),
        ]
        table = build_weighted_alpha_table(cache, [0.0, 1.0], shrink_k=50)
        cell = list(table["cells"].values())[0]
        # Only exc2 contributes: mean = 0.20, n = 1.0
        assert abs(cell["mean_excess_ret_6m"] - 0.20) < 1e-6
        assert abs(cell["n"] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Clinical coverage gate
# ---------------------------------------------------------------------------

class TestClinicalCoverageGate:
    def test_missing_z_tier_excluded_from_pairing(self):
        """Rows with empty clinical_score_z_tier are excluded by _extract_clinical."""
        # DD with valid z_tier → included
        row_ok = {"archetype": "drug_developer", "clinical_score_z_tier": "1.5"}
        assert _extract_clinical(row_ok) == 1.5

        # DD with empty z_tier (e.g., missing clinical_score in 2026 archives) → None
        row_blank = {"archetype": "drug_developer", "clinical_score_z_tier": ""}
        assert _extract_clinical(row_blank) is None

        # DD with z_tier = "0.0" (backfill fallback) → included (0.0 is valid)
        row_zero = {"archetype": "drug_developer", "clinical_score_z_tier": "0.0"}
        assert _extract_clinical(row_zero) == 0.0

    def test_clinical_ic_nan_when_all_z_tier_missing(self):
        """If no rows have parseable z_tier, IC should be NaN (< 3 pairs)."""
        # Simulate a 2026-01 archive: all DDs have z_tier="" after failed backfill
        rows_blank = [
            {"archetype": "drug_developer", "clinical_score_z_tier": "",
             "ticker": f"T{i}"} for i in range(50)
        ]
        clin_pairs = []
        excess = {f"T{i}": 0.01 * i for i in range(50)}
        for row in rows_blank:
            sig = _extract_clinical(row)
            tk = row["ticker"]
            if sig is not None and tk in excess:
                clin_pairs.append((sig, excess[tk]))
        # No pairs should form
        assert len(clin_pairs) == 0
        # IC computation would return NaN (< 3 pairs)
        ic = spearman_rank_corr(
            [p[0] for p in clin_pairs], [p[1] for p in clin_pairs]
        ) if len(clin_pairs) >= 3 else float("nan")
        assert math.isnan(ic)


# ---------------------------------------------------------------------------
# Cutoff-based subperiod split
# ---------------------------------------------------------------------------

class TestCutoffSubperiod:
    def test_subperiod_uses_cutoff(self):
        """Subperiod keys are pre_cutoff/post_cutoff, split at cutoff date."""
        ic_rows = [
            {"date": "2024-06-01", "regime": "",
             "ic_clinical": 0.10, "ic_catalyst": 0.05, "ic_alpha": 0.08,
             "ic_alpha_incr": 0.03},
            {"date": "2024-12-01", "regime": "",
             "ic_clinical": 0.12, "ic_catalyst": 0.06, "ic_alpha": 0.09,
             "ic_alpha_incr": 0.04},
            {"date": "2025-03-01", "regime": "",
             "ic_clinical": -0.02, "ic_catalyst": -0.01, "ic_alpha": -0.03,
             "ic_alpha_incr": -0.02},
        ]
        spread_rows = [
            {"date": "2024-06-01", "regime": "",
             "spread_clinical": 0.02, "spread_catalyst": 0.01, "spread_alpha": 0.03,
             "spread_alpha_double": 0.01},
            {"date": "2024-12-01", "regime": "",
             "spread_clinical": 0.03, "spread_catalyst": 0.015, "spread_alpha": 0.04,
             "spread_alpha_double": 0.02},
            {"date": "2025-03-01", "regime": "",
             "spread_clinical": -0.01, "spread_catalyst": -0.005, "spread_alpha": -0.02,
             "spread_alpha_double": -0.005},
        ]
        summary = _build_summary(ic_rows, spread_rows, horizon=126, alpha_skipped=0,
                                 cutoff="2025-01-01")
        assert "subperiod" in summary
        assert "pre_cutoff" in summary["subperiod"]
        assert "post_cutoff" in summary["subperiod"]
        # Pre-cutoff has 2 dates (2024-06-01, 2024-12-01)
        assert summary["subperiod"]["pre_cutoff"]["clinical"]["n"] == 2
        # Post-cutoff has 1 date (2025-03-01)
        assert summary["subperiod"]["post_cutoff"]["clinical"]["n"] == 1
        assert summary["cutoff"] == "2025-01-01"

    def test_pos_share_thresholds_in_summary(self):
        """min/max pos_share thresholds appear in summary output."""
        ic_rows = [
            {"date": "2024-06-01", "regime": "",
             "ic_clinical": 0.10, "ic_catalyst": 0.05, "ic_alpha": 0.08,
             "ic_alpha_incr": 0.03},
        ]
        spread_rows = [
            {"date": "2024-06-01", "regime": "",
             "spread_clinical": 0.02, "spread_catalyst": 0.01, "spread_alpha": 0.03,
             "spread_alpha_double": 0.01},
        ]
        summary = _build_summary(
            ic_rows, spread_rows, horizon=126, alpha_skipped=0,
            min_clinical_pos_share=0.10, max_clinical_pos_share=0.90,
        )
        assert summary["min_clinical_pos_share"] == 0.10
        assert summary["max_clinical_pos_share"] == 0.90

    def test_pos_share_filter_independent_of_coverage(self):
        """pos_share filter activates even when min_clinical_coverage=0.0.

        When max_clinical_pos_share < 1.0, dates with homogeneous sign
        distribution should be excluded from training regardless of
        the coverage threshold.
        """
        # Build two mock date_cache entries:
        # date A: pos_share=0.50 (healthy)
        # date B: pos_share=0.99 (homogeneous — should be excluded)
        # With min_clinical_coverage=0.0 but max_clinical_pos_share=0.95,
        # date B should be filtered out.
        train_dates = [
            {"clinical_coverage": 1.0, "clinical_pos_share": 0.50},
            {"clinical_coverage": 1.0, "clinical_pos_share": 0.99},
        ]

        min_cov = 0.0  # off
        min_ps = 0.05
        max_ps = 0.95

        # The decoupled filter condition:
        if (min_cov > 0.0) or (min_ps > 0.0) or (max_ps < 1.0):
            filtered = [
                d for d in train_dates
                if d["clinical_coverage"] >= min_cov
                and min_ps <= d["clinical_pos_share"] <= max_ps
            ]
        else:
            filtered = train_dates

        # Only date A survives
        assert len(filtered) == 1
        assert filtered[0]["clinical_pos_share"] == 0.50

    def test_cutoff_recorded_in_summary(self):
        """Cutoff date appears in summary.json output."""
        ic_rows = [
            {"date": "2024-06-01", "regime": "",
             "ic_clinical": 0.10, "ic_catalyst": 0.05, "ic_alpha": 0.08,
             "ic_alpha_incr": 0.03},
        ]
        spread_rows = [
            {"date": "2024-06-01", "regime": "",
             "spread_clinical": 0.02, "spread_catalyst": 0.01, "spread_alpha": 0.03,
             "spread_alpha_double": 0.01},
        ]
        summary = _build_summary(ic_rows, spread_rows, horizon=126, alpha_skipped=0,
                                 cutoff="2024-10-01")
        assert summary["cutoff"] == "2024-10-01"


# ---------------------------------------------------------------------------
# Forward-return coverage diagnostics
# ---------------------------------------------------------------------------

class TestFwdReturnDiagnostics:
    def test_coverage_metrics_basic(self):
        """Coverage = n_fwd / n_price_rows (priceable denominator);
        skip_reason="" when above threshold."""
        n_price_rows = 190
        n_fwd = 180
        min_fwd_coverage = 0.5
        coverage = n_fwd / n_price_rows if n_price_rows > 0 else 0.0

        skip_reason = ""
        if n_fwd == 0:
            skip_reason = "NO_FWD_RET"
        elif min_fwd_coverage > 0.0 and coverage < min_fwd_coverage:
            skip_reason = "LOW_COVERAGE"

        assert abs(coverage - 180 / 190) < 1e-9
        assert skip_reason == ""

    def test_skip_reason_no_fwd(self):
        """n_fwd=0 → skip_reason='NO_FWD_RET', included=False."""
        n_price_rows = 190
        n_fwd = 0
        min_fwd_coverage = 0.0
        coverage = n_fwd / n_price_rows if n_price_rows > 0 else 0.0

        skip_reason = ""
        if n_fwd == 0:
            skip_reason = "NO_FWD_RET"
        elif min_fwd_coverage > 0.0 and coverage < min_fwd_coverage:
            skip_reason = "LOW_COVERAGE"

        included = skip_reason == ""
        assert skip_reason == "NO_FWD_RET"
        assert included is False
        assert coverage == 0.0

    def test_skip_reason_low_coverage(self):
        """n_fwd > 0 but coverage < min_fwd_coverage → 'LOW_COVERAGE'."""
        n_price_rows = 190
        n_fwd = 50
        min_fwd_coverage = 0.5
        coverage = n_fwd / n_price_rows if n_price_rows > 0 else 0.0

        skip_reason = ""
        if n_fwd == 0:
            skip_reason = "NO_FWD_RET"
        elif min_fwd_coverage > 0.0 and coverage < min_fwd_coverage:
            skip_reason = "LOW_COVERAGE"

        included = skip_reason == ""
        assert skip_reason == "LOW_COVERAGE"
        assert included is False
        assert abs(coverage - 50 / 190) < 1e-9

    def test_diagnostics_in_summary(self):
        """_build_summary() with date_diagnostics produces fwd_return_diagnostics block."""
        date_diagnostics = [
            {"date": "2024-01-31", "n_rows": 300, "n_price_rows": 280,
             "n_fwd_rets": 260, "fwd_ret_coverage": 0.9286,
             "skip_reason": "", "included": True},
            {"date": "2024-02-28", "n_rows": 300, "n_price_rows": 280,
             "n_fwd_rets": 270, "fwd_ret_coverage": 0.9643,
             "skip_reason": "", "included": True},
            {"date": "2024-03-31", "n_rows": 300, "n_price_rows": 280,
             "n_fwd_rets": 0, "fwd_ret_coverage": 0.0,
             "skip_reason": "NO_FWD_RET", "included": False},
            {"date": "2024-04-30", "n_rows": 0, "n_price_rows": 0,
             "n_fwd_rets": 0, "fwd_ret_coverage": 0.0,
             "skip_reason": "EMPTY_RANKINGS", "included": False},
            {"date": "2024-05-31", "n_rows": 300, "n_price_rows": 280,
             "n_fwd_rets": 30, "fwd_ret_coverage": 0.1071,
             "skip_reason": "LOW_COVERAGE", "included": False},
        ]
        ic_rows = [
            {"date": "2024-01-31", "regime": "",
             "ic_clinical": 0.10, "ic_catalyst": 0.05, "ic_alpha": 0.08,
             "ic_alpha_incr": 0.03},
            {"date": "2024-02-28", "regime": "",
             "ic_clinical": 0.12, "ic_catalyst": 0.06, "ic_alpha": 0.09,
             "ic_alpha_incr": 0.04},
        ]
        spread_rows = [
            {"date": "2024-01-31", "regime": "",
             "spread_clinical": 0.02, "spread_catalyst": 0.01, "spread_alpha": 0.03,
             "spread_alpha_double": 0.01},
            {"date": "2024-02-28", "regime": "",
             "spread_clinical": 0.03, "spread_catalyst": 0.015, "spread_alpha": 0.04,
             "spread_alpha_double": 0.02},
        ]
        summary = _build_summary(
            ic_rows, spread_rows, horizon=126, alpha_skipped=0,
            date_diagnostics=date_diagnostics, min_fwd_coverage=0.5,
            n_price_universe=350,
        )
        assert "fwd_return_diagnostics" in summary
        diag = summary["fwd_return_diagnostics"]
        assert diag["n_archives_total"] == 5
        assert diag["n_dates_included"] == 2
        assert diag["n_dates_skipped_no_fwd"] == 1
        assert diag["n_dates_skipped_low_coverage"] == 1
        assert diag["n_dates_skipped_empty_rankings"] == 1
        assert diag["min_fwd_coverage_threshold"] == 0.5
        assert diag["n_price_universe"] == 350
        # Coverage stats from included dates only (260/280, 270/280)
        assert diag["coverage_stats"]["min"] == 0.9286
        assert diag["coverage_stats"]["max"] == 0.9643
        assert diag["coverage_stats"]["median"] == 0.9465  # (0.9286+0.9643)/2
        # Bottom 5 included dates (sorted by coverage ascending)
        assert len(diag["bottom_5_included"]) == 2
        assert diag["bottom_5_included"][0]["date"] == "2024-01-31"
        # Top 5 skipped LOW_COVERAGE
        assert len(diag["top_5_skipped_low_coverage"]) == 1
        assert diag["top_5_skipped_low_coverage"][0]["date"] == "2024-05-31"

    def test_diagnostics_empty_included(self):
        """When no dates are included, coverage_stats should be empty."""
        date_diagnostics = [
            {"date": "2024-01-31", "n_rows": 300, "n_price_rows": 280,
             "n_fwd_rets": 0, "fwd_ret_coverage": 0.0,
             "skip_reason": "NO_FWD_RET", "included": False},
        ]
        # Empty ic/spread means we get early return, so test _build_summary directly
        # with non-empty ic_rows but all dates skipped in diagnostics
        ic_rows = [
            {"date": "2024-01-31", "regime": "",
             "ic_clinical": 0.10, "ic_catalyst": 0.05, "ic_alpha": 0.08,
             "ic_alpha_incr": 0.03},
        ]
        spread_rows = [
            {"date": "2024-01-31", "regime": "",
             "spread_clinical": 0.02, "spread_catalyst": 0.01, "spread_alpha": 0.03,
             "spread_alpha_double": 0.01},
        ]
        summary = _build_summary(
            ic_rows, spread_rows, horizon=126, alpha_skipped=0,
            date_diagnostics=date_diagnostics, min_fwd_coverage=0.0,
        )
        diag = summary["fwd_return_diagnostics"]
        assert diag["coverage_stats"] == {}
        assert diag["bottom_5_included"] == []
        assert diag["n_dates_included"] == 0
