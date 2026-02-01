"""
Tests for M3 weight calibration (backtest/calibrate_m3_weights.py).

Uses synthetic panels only — no production_data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backtest.calibrate_m3_weights import (
    apply_sign_priors,
    calibrate,
    normalize_weights,
    pooled_ridge,
    write_weights_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_panel(
    n_dates: int = 30,
    n_stocks: int = 200,
    betas: dict | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Synthetic panel with known linear relationships.

    betas: {feature_name: true_beta}
    Returns DataFrame with rebalance_date, ticker, features, and fwd_21d.
    """
    if betas is None:
        betas = {
            "valuation_normalized": -0.5,
            "momentum_normalized": -0.3,
            "financial_normalized": -0.2,
        }
    np.random.seed(seed)
    rows = []
    for d in range(n_dates):
        dt = f"2024-{d+1:04d}"
        features = {
            name: np.random.randn(n_stocks) for name in sorted(betas.keys())
        }
        ret = np.zeros(n_stocks)
        for name, beta in betas.items():
            ret += beta * features[name]
        ret += np.random.randn(n_stocks) * 0.3  # noise

        for i in range(n_stocks):
            row = {"rebalance_date": dt, "ticker": f"T{i:03d}"}
            for name in sorted(betas.keys()):
                row[name] = features[name][i]
            row["fwd_21d"] = ret[i]
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ridge regression
# ---------------------------------------------------------------------------


class TestPooledRidge:
    def test_recovers_negative_signs(self):
        """Ridge should learn negative weights when features negatively predict returns."""
        panel = _make_panel(
            betas={
                "valuation_normalized": -0.5,
                "momentum_normalized": -0.3,
                "financial_normalized": -0.2,
            }
        )
        weights, meta = pooled_ridge(
            panel, "fwd_21d",
            ["financial_normalized", "momentum_normalized", "valuation_normalized"],
            ridge_lambda=1.0,
            min_stocks_per_week=50,
        )
        assert weights["valuation_normalized"] < 0
        assert weights["momentum_normalized"] < 0
        assert weights["financial_normalized"] < 0
        assert meta["n_usable_dates"] == 30

    def test_deterministic(self):
        """Same input → same output."""
        panel = _make_panel(seed=123)
        w1, _ = pooled_ridge(panel, "fwd_21d",
                             ["financial_normalized", "momentum_normalized", "valuation_normalized"],
                             ridge_lambda=5.0)
        w2, _ = pooled_ridge(panel, "fwd_21d",
                             ["financial_normalized", "momentum_normalized", "valuation_normalized"],
                             ridge_lambda=5.0)
        assert w1 == w2

    def test_higher_lambda_shrinks(self):
        """Higher lambda → smaller absolute weights."""
        panel = _make_panel()
        w_low, _ = pooled_ridge(panel, "fwd_21d",
                                ["valuation_normalized", "momentum_normalized"],
                                ridge_lambda=1.0)
        w_high, _ = pooled_ridge(panel, "fwd_21d",
                                 ["valuation_normalized", "momentum_normalized"],
                                 ridge_lambda=100.0)
        for feat in w_low:
            assert abs(w_high[feat]) <= abs(w_low[feat]) + 1e-10


# ---------------------------------------------------------------------------
# Sign priors
# ---------------------------------------------------------------------------


class TestSignPriors:
    def test_zeroes_violating_weights(self):
        weights = {"valuation_normalized": 0.1, "momentum_normalized": -0.3}
        priors = {"valuation_normalized": -1, "momentum_normalized": -1}
        adj, violations = apply_sign_priors(weights, priors)
        assert adj["valuation_normalized"] == 0.0
        assert adj["momentum_normalized"] == -0.3
        assert "valuation_normalized" in violations

    def test_none_prior_allows_any_sign(self):
        weights = {"clinical_normalized": 0.2}
        priors = {"clinical_normalized": None}
        adj, violations = apply_sign_priors(weights, priors)
        assert adj["clinical_normalized"] == 0.2
        assert not violations

    def test_disable_negative(self):
        weights = {"a": -0.5, "b": 0.3}
        adj, violations = apply_sign_priors(weights, {}, disable_negative=True)
        assert adj["a"] == 0.0
        assert adj["b"] == 0.3


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_l1_sums_to_one(self):
        weights = {"a": -0.5, "b": 0.3, "c": 0.1}
        normed = normalize_weights(weights)
        l1 = sum(abs(v) for v in normed.values())
        assert abs(l1 - 1.0) < 1e-6

    def test_cap_applied(self):
        weights = {"a": -10.0, "b": 0.001}
        normed = normalize_weights(weights, max_abs_weight=0.60)
        assert all(abs(v) <= 0.60 + 1e-8 for v in normed.values())

    def test_preserves_signs(self):
        weights = {"a": -0.5, "b": 0.3}
        normed = normalize_weights(weights)
        assert normed["a"] < 0
        assert normed["b"] > 0

    def test_all_zero(self):
        weights = {"a": 0.0, "b": 0.0}
        normed = normalize_weights(weights)
        assert all(v == 0.0 for v in normed.values())


# ---------------------------------------------------------------------------
# Full calibrate() pipeline
# ---------------------------------------------------------------------------


class TestCalibrate:
    def test_negative_weights_learned(self):
        """Full pipeline learns negative weights for negatively-predictive features."""
        panel = _make_panel(
            betas={
                "valuation_normalized": -0.5,
                "momentum_normalized": -0.3,
                "financial_normalized": -0.1,
            }
        )
        final, meta = calibrate(
            panel,
            return_col="fwd_21d",
            feature_cols=["valuation_normalized", "momentum_normalized", "financial_normalized"],
            method="ridge",
            ridge_lambda=1.0,
            min_stocks_per_week=50,
        )
        assert final["valuation_normalized"] < 0
        assert final["momentum_normalized"] < 0
        assert final["financial_normalized"] < 0
        # L1 = 1
        l1 = sum(abs(v) for v in final.values())
        assert abs(l1 - 1.0) < 1e-6

    def test_constant_feature_dropped(self):
        """Features that are constant across weeks get dropped."""
        panel = _make_panel(
            betas={"valuation_normalized": -0.5, "momentum_normalized": -0.3}
        )
        # Add a constant feature
        panel["catalyst_normalized"] = 50.0
        final, meta = calibrate(
            panel,
            return_col="fwd_21d",
            feature_cols=["valuation_normalized", "momentum_normalized", "catalyst_normalized"],
            method="ridge",
            ridge_lambda=1.0,
            min_stocks_per_week=50,
        )
        assert final["catalyst_normalized"] == 0.0
        assert "catalyst_normalized" in meta["dropped_features"]

    def test_deterministic_output(self):
        """Same panel → same weights on repeat runs."""
        panel = _make_panel(seed=99)
        w1, _ = calibrate(panel, "fwd_21d",
                          ["valuation_normalized", "momentum_normalized"],
                          method="ridge", ridge_lambda=5.0, min_stocks_per_week=50)
        w2, _ = calibrate(panel, "fwd_21d",
                          ["valuation_normalized", "momentum_normalized"],
                          method="ridge", ridge_lambda=5.0, min_stocks_per_week=50)
        assert w1 == w2

    def test_fmb_method(self):
        """FMB method should also produce negative weights for negative-beta features."""
        panel = _make_panel(
            n_dates=40,
            betas={"valuation_normalized": -0.5, "momentum_normalized": -0.3},
        )
        final, meta = calibrate(
            panel,
            return_col="fwd_21d",
            feature_cols=["valuation_normalized", "momentum_normalized"],
            method="fmb",
            min_stocks_per_week=50,
        )
        assert final["valuation_normalized"] < 0
        assert final["momentum_normalized"] < 0

    def test_disable_negative_mode(self):
        """--m3-disable-negative clamps all negative weights to 0."""
        panel = _make_panel(
            betas={"valuation_normalized": -0.5, "momentum_normalized": -0.3},
        )
        final, meta = calibrate(
            panel,
            return_col="fwd_21d",
            feature_cols=["valuation_normalized", "momentum_normalized"],
            method="ridge",
            ridge_lambda=1.0,
            min_stocks_per_week=50,
            disable_negative=True,
        )
        assert all(v >= 0 for v in final.values())


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


class TestWriteWeightsJSON:
    def test_write_and_read(self, tmp_path):
        path = tmp_path / "weights.json"
        weights = {"valuation_normalized": -0.4, "momentum_normalized": -0.3}
        meta = {"method": "ridge", "return_col": "fwd_21d"}
        write_weights_json(path, weights, meta)

        data = json.loads(path.read_text())
        assert data["feature_weights"]["valuation_normalized"] == -0.4
        assert data["method"] == "ridge"

    def test_sorted_keys(self, tmp_path):
        path = tmp_path / "weights.json"
        write_weights_json(
            path,
            {"z_feat": 0.1, "a_feat": -0.2},
            {"return_col": "fwd_21d"},
        )
        text = path.read_text()
        # "a_feat" should appear before "z_feat" in sorted JSON
        assert text.index("a_feat") < text.index("z_feat")
