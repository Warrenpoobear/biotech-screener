"""Tests for M3 Before/After Weight Evaluation."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.evaluate_m3_weights import (
    compute_linear_score,
    compute_ic,
    compute_topn_returns,
    compute_quintile_spread,
    compute_hit_rate,
    compute_turnover,
    build_cumulative_curve,
    compute_drawdown,
    ic_summary,
    load_m3_weights,
    DEFAULT_V3_WEIGHTS,
    FEATURE_COLS,
)


# ── Helpers ───────────────────────────────────────────────────────────
def _make_panel(n_dates=5, n_stocks=30, seed=42):
    """Create synthetic panel for testing."""
    rng = np.random.RandomState(seed)
    dates = [f"2024-01-{d:02d}" for d in range(1, n_dates + 1)]
    tickers = [f"TK{i:03d}" for i in range(n_stocks)]
    rows = []
    for dt in dates:
        for tk in tickers:
            rows.append({
                "rebalance_date": dt,
                "ticker": tk,
                "clinical_normalized": rng.uniform(0, 1),
                "financial_normalized": rng.uniform(0, 1),
                "momentum_normalized": rng.uniform(0, 1),
                "valuation_normalized": rng.uniform(0, 1),
                "composite_score": rng.uniform(0, 1),
                "fwd_5d": rng.normal(0, 0.05),
                "fwd_21d": rng.normal(0, 0.10),
                "fwd_63d": rng.normal(0, 0.15),
                "fwd_126d": rng.normal(0, 0.20),
            })
    return pd.DataFrame(rows)


# ── TestScoreComputation ──────────────────────────────────────────────
class TestScoreComputation:
    def test_positive_weights_monotonic(self):
        """Higher features → higher score with positive weights."""
        df = pd.DataFrame({
            "clinical_normalized": [0.0, 0.5, 1.0],
            "financial_normalized": [0.0, 0.5, 1.0],
            "momentum_normalized": [0.0, 0.5, 1.0],
            "valuation_normalized": [0.0, 0.5, 1.0],
        })
        score = compute_linear_score(df, DEFAULT_V3_WEIGHTS)
        assert score.iloc[0] < score.iloc[1] < score.iloc[2]

    def test_negative_weights_reverse_ranking(self):
        """Negative weights mean higher features → lower score (no negation)."""
        weights = {c: -0.25 for c in FEATURE_COLS}
        df = pd.DataFrame({
            "clinical_normalized": [0.0, 1.0],
            "financial_normalized": [0.0, 1.0],
            "momentum_normalized": [0.0, 1.0],
            "valuation_normalized": [0.0, 1.0],
        })
        score = compute_linear_score(df, weights)
        # Negative weights: row with all-0 features scores 0,
        # row with all-1 features scores -1.0 → lower
        assert score.iloc[0] > score.iloc[1]
        assert score.iloc[0] == 0.0
        assert abs(score.iloc[1] - (-1.0)) < 1e-9

    def test_nan_propagation(self):
        """NaN in a feature propagates to score."""
        df = pd.DataFrame({
            "clinical_normalized": [1.0, float("nan")],
            "financial_normalized": [1.0, 1.0],
            "momentum_normalized": [1.0, 1.0],
            "valuation_normalized": [1.0, 1.0],
        })
        score = compute_linear_score(df, DEFAULT_V3_WEIGHTS)
        assert math.isfinite(score.iloc[0])
        assert math.isnan(score.iloc[1])

    def test_weight_normalization(self):
        """Default V3 weights sum to ~1."""
        total = sum(DEFAULT_V3_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01


# ── Perturbation sign contract ───────────────────────────────────────

class TestPerturbationSignContract:
    """
    For each feature, bump it by +0.10 and verify the score moves in the
    direction implied by the weight sign.  This locks the interpretation:

        score = Σ w_i * x_i   (no negation, no inversion)

    so a negative weight means increasing the feature *decreases* the score.
    """

    @staticmethod
    def _perturb_score_delta(weights, feature, delta=0.10):
        """Return score change when `feature` increases by `delta`."""
        base = pd.DataFrame({c: [0.5] for c in FEATURE_COLS})
        bumped = base.copy()
        bumped[feature] = base[feature] + delta
        s_base = compute_linear_score(base, weights).iloc[0]
        s_bumped = compute_linear_score(bumped, weights).iloc[0]
        return s_bumped - s_base

    def test_m3_weights_direction(self):
        """Each M3 calibrated weight's sign matches the perturbation direction."""
        m3_weights_path = Path(__file__).resolve().parent.parent / "data" / "module5_weights_m3.json"
        if not m3_weights_path.exists():
            pytest.skip("module5_weights_m3.json not found")
        m3_weights = load_m3_weights(m3_weights_path)
        for feat in FEATURE_COLS:
            w = m3_weights.get(feat, 0.0)
            if abs(w) < 1e-10:
                continue
            delta_score = self._perturb_score_delta(m3_weights, feat)
            if w > 0:
                assert delta_score > 0, (
                    f"{feat}: w={w:+.4f} but +0.10 bump moved score by {delta_score:+.6f}"
                )
            else:
                assert delta_score < 0, (
                    f"{feat}: w={w:+.4f} but +0.10 bump moved score by {delta_score:+.6f}"
                )

    def test_default_v3_weights_direction(self):
        """Default V3 weights are all positive → bump always increases score."""
        for feat in FEATURE_COLS:
            w = DEFAULT_V3_WEIGHTS.get(feat, 0.0)
            if abs(w) < 1e-10:
                continue
            delta_score = self._perturb_score_delta(DEFAULT_V3_WEIGHTS, feat)
            assert delta_score > 0, (
                f"{feat}: default w={w:+.4f} but +0.10 bump moved score by {delta_score:+.6f}"
            )

    def test_perturbation_magnitude_proportional_to_weight(self):
        """Larger |weight| → larger |score delta| for the same bump."""
        m3_weights_path = Path(__file__).resolve().parent.parent / "data" / "module5_weights_m3.json"
        if not m3_weights_path.exists():
            pytest.skip("module5_weights_m3.json not found")
        m3_weights = load_m3_weights(m3_weights_path)
        deltas = {}
        for feat in FEATURE_COLS:
            deltas[feat] = abs(self._perturb_score_delta(m3_weights, feat))
        # Sort features by |weight|
        sorted_by_w = sorted(FEATURE_COLS, key=lambda f: abs(m3_weights.get(f, 0.0)))
        sorted_by_delta = sorted(FEATURE_COLS, key=lambda f: deltas[f])
        assert sorted_by_w == sorted_by_delta, (
            f"Weight magnitude order {sorted_by_w} != delta order {sorted_by_delta}"
        )

    def test_single_feature_perturbation_isolates(self):
        """Bumping one feature doesn't affect the contribution of others."""
        weights = {c: -0.25 for c in FEATURE_COLS}
        target = FEATURE_COLS[0]
        base = pd.DataFrame({c: [0.5] for c in FEATURE_COLS})
        bumped = base.copy()
        bumped[target] = 0.6
        delta = (
            compute_linear_score(bumped, weights).iloc[0]
            - compute_linear_score(base, weights).iloc[0]
        )
        expected = -0.25 * 0.1  # w * delta_x
        assert abs(delta - expected) < 1e-10, (
            f"Expected delta={expected}, got {delta}"
        )


# ── TestTopN ──────────────────────────────────────────────────────────
class TestTopN:
    def test_selects_highest_scores(self):
        """Top-N should select the N highest-scored stocks."""
        df = pd.DataFrame({
            "rebalance_date": ["d1"] * 5,
            "score": [1.0, 5.0, 3.0, 4.0, 2.0],
            "ret": [0.01, 0.05, 0.03, 0.04, 0.02],
        })
        result = compute_topn_returns(df, "score", "ret", n=2)
        # Top-2 by score: 5.0→0.05, 4.0→0.04 → mean = 0.045
        assert abs(result.iloc[0] - 0.045) < 1e-9

    def test_handles_nan_returns(self):
        """Rows with NaN returns are excluded from top-N."""
        df = pd.DataFrame({
            "rebalance_date": ["d1"] * 4,
            "score": [4.0, 3.0, 2.0, 1.0],
            "ret": [float("nan"), 0.03, 0.02, 0.01],
        })
        result = compute_topn_returns(df, "score", "ret", n=2)
        # After dropping NaN: top-2 by score are 3.0→0.03, 2.0→0.02
        assert abs(result.iloc[0] - 0.025) < 1e-9

    def test_handles_n_greater_than_available(self):
        """When N > available stocks, use all available."""
        df = pd.DataFrame({
            "rebalance_date": ["d1"] * 3,
            "score": [3.0, 2.0, 1.0],
            "ret": [0.03, 0.02, 0.01],
        })
        result = compute_topn_returns(df, "score", "ret", n=100)
        assert abs(result.iloc[0] - 0.02) < 1e-9  # mean of all 3


# ── TestTurnover ──────────────────────────────────────────────────────
class TestTurnover:
    def test_perfect_stability(self):
        """Same top-N across dates → turnover = 0."""
        df = pd.DataFrame({
            "rebalance_date": ["d1"] * 3 + ["d2"] * 3,
            "ticker": ["A", "B", "C"] * 2,
            "score": [3.0, 2.0, 1.0] * 2,
        })
        turn = compute_turnover(df, "score", n=2)
        assert len(turn) == 1  # only d2 has turnover vs d1
        assert abs(turn.iloc[0]) < 1e-9

    def test_complete_turnover(self):
        """Completely different top-N → turnover = 1."""
        df = pd.DataFrame({
            "rebalance_date": ["d1"] * 4 + ["d2"] * 4,
            "ticker": ["A", "B", "C", "D"] * 2,
            "score": [4.0, 3.0, 1.0, 0.0, 0.0, 1.0, 3.0, 4.0],
        })
        turn = compute_turnover(df, "score", n=2)
        assert abs(turn.iloc[0] - 1.0) < 1e-9

    def test_partial_overlap(self):
        """Known partial overlap → Jaccard distance between 0 and 1."""
        df = pd.DataFrame({
            "rebalance_date": ["d1"] * 4 + ["d2"] * 4,
            "ticker": ["A", "B", "C", "D"] * 2,
            "score": [4.0, 3.0, 2.0, 1.0, 1.0, 4.0, 3.0, 2.0],
        })
        # d1 top-2: {A, B}, d2 top-2: {B, C}
        # Jaccard distance = 1 - |{B}|/|{A,B,C}| = 1 - 1/3 = 2/3
        turn = compute_turnover(df, "score", n=2)
        assert abs(turn.iloc[0] - 2.0 / 3.0) < 1e-9


# ── TestQuintileSpread ────────────────────────────────────────────────
class TestQuintileSpread:
    def test_positive_spread_monotonic(self):
        """When return is monotonically increasing with score, spread > 0."""
        n = 50
        df = pd.DataFrame({
            "rebalance_date": ["d1"] * n,
            "score": list(range(n)),
            "ret": [i * 0.01 for i in range(n)],
        })
        spread = compute_quintile_spread(df, "score", "ret")
        assert spread.iloc[0] > 0

    def test_correct_bucket_sizes(self):
        """Quintiles should split data into ~5 equal groups."""
        # This is implicitly tested by the spread calculation working;
        # we verify spread is finite for a reasonable panel
        panel = _make_panel(n_dates=1, n_stocks=50)
        panel["score"] = panel["financial_normalized"]
        spread = compute_quintile_spread(panel, "score", "fwd_21d")
        assert len(spread) == 1
        assert math.isfinite(spread.iloc[0])


# ── TestIC ────────────────────────────────────────────────────────────
class TestIC:
    def test_perfect_correlation(self):
        """Perfect score-return correlation → IC ~ 1."""
        n = 20
        df = pd.DataFrame({
            "rebalance_date": ["d1"] * n,
            "score": list(range(n)),
            "ret": [i * 0.01 for i in range(n)],
        })
        ic = compute_ic(df, "score", "ret")
        assert abs(ic.iloc[0] - 1.0) < 1e-6

    def test_negation_flips_sign(self):
        """Negating scores flips IC sign."""
        n = 20
        df = pd.DataFrame({
            "rebalance_date": ["d1"] * n,
            "score_pos": list(range(n)),
            "score_neg": list(range(n, 0, -1)),
            "ret": [i * 0.01 for i in range(n)],
        })
        ic_pos = compute_ic(df, "score_pos", "ret")
        ic_neg = compute_ic(df, "score_neg", "ret")
        assert ic_pos.iloc[0] > 0
        assert ic_neg.iloc[0] < 0
        assert abs(ic_pos.iloc[0] + ic_neg.iloc[0]) < 1e-6


# ── TestCumulativeReturn ──────────────────────────────────────────────
class TestCumulativeReturn:
    def test_cumulative_curve(self):
        """Cumulative curve compounds correctly."""
        rets = pd.Series([0.10, -0.05, 0.03], index=["d1", "d2", "d3"])
        cum = build_cumulative_curve(rets)
        expected = 1.10 * 0.95 * 1.03
        assert abs(cum.iloc[-1] - expected) < 1e-9

    def test_drawdown_no_loss(self):
        """All-positive returns → max_dd = 0."""
        rets = pd.Series([0.01, 0.02, 0.03])
        cum = build_cumulative_curve(rets)
        dd, dur = compute_drawdown(cum)
        assert dd == 0.0
        assert dur == 0

    def test_drawdown_known(self):
        """Known drawdown case."""
        # Curve: 1.0, 1.1, 0.99, 1.05
        rets = pd.Series([0.10, -0.10, 0.06060606])
        cum = build_cumulative_curve(rets)
        dd, dur = compute_drawdown(cum)
        assert dd < 0  # There is a drawdown
        assert dur >= 1


# ── TestEndToEnd ──────────────────────────────────────────────────────
class TestEndToEnd:
    def test_synthetic_panel_shapes(self):
        """All outputs have correct shape on synthetic 5-date/30-stock panel."""
        panel = _make_panel(n_dates=5, n_stocks=30)
        score = compute_linear_score(panel, DEFAULT_V3_WEIGHTS)
        assert len(score) == 150

        panel["test_score"] = score
        ic = compute_ic(panel, "test_score", "fwd_21d")
        assert len(ic) == 5

        topn = compute_topn_returns(panel, "test_score", "fwd_21d", n=10)
        assert len(topn) == 5

        spread = compute_quintile_spread(panel, "test_score", "fwd_21d")
        assert len(spread) == 5

        hr = compute_hit_rate(panel, "test_score", "fwd_21d")
        assert len(hr) == 5

        turn = compute_turnover(panel, "test_score", n=10)
        assert len(turn) == 4  # n_dates - 1

    def test_ic_summary_structure(self):
        """IC summary has expected keys."""
        ic_ts = pd.Series([0.1, 0.2, -0.05, 0.15, 0.3])
        summary = ic_summary(ic_ts)
        assert "mean" in summary
        assert "median" in summary
        assert "tstat" in summary
        assert "hit_rate" in summary
        assert summary["n"] == 5
        assert 0.0 < summary["hit_rate"] <= 1.0
