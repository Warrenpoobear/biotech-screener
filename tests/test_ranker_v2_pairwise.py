"""Tests for ranker_v2_pairwise.py — Spec 051."""

import json
import math

import pytest

from ranker_v2_pairwise import (
    ALL_BLOCKS,
    FEATURES_MINIMAL,
    FeatureSpec,
    PairwiseLogisticModel,
    PointwiseLogisticModel,
    RankerV2Config,
    _encode_feature,
    _sf,
    _sigmoid,
    _spearman,
    compute_recency_weight,
    config_id,
    extract_features,
    filter_cohort,
    generate_pairs,
    get_feature_specs,
    model_from_dict,
    model_to_dict,
    score_snapshot,
    train_and_evaluate,
    train_pairwise_logistic,
    train_pointwise_logistic,
    zscore_cohort_features,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_row(ticker="AAAA", rank=1, eligible=1, **overrides):
    """Build a minimal research panel row."""
    base = {
        "ticker": ticker,
        "actionable_rank": rank,
        "eligible": eligible,
        "catalyst_in_window": "1",
        "catalyst_days": 60,
        # Institutional
        "coinvest_score_z": 0.5,
        "inst_delta_z": 0.3,
        "coinvest_conviction": 0.6,
        "coinvest_filing_age_days": 45,
        "sponsor_tier1_count": 2,
        "inst_delta_net": 1,
        # Clinical
        "clinical_score_v2_z": 0.4,
        "clinical_quality_composite": 0.5,
        "endpoint_strength_score": 0.6,
        "design_quality_score": 0.5,
        "binary_quality_score": 0.7,
        "aact_execution_score": 0.4,
        "execution_momentum": 0.3,
        "catalyst_decay_w": 0.8,
        "cat_priority": 2,
        "catalyst_type_tier": "T1",
        "catalyst_family": "REGULATORY",
        # Options
        "ovf_composite": 0.5,
        "ovf11_score": 0.4,
        "cheap_vol_score": 0.3,
        "opt_rr_25d": -0.01,
        "opt_event_premium": 0.2,
        "opt_term_slope": 0.1,
        "opt_iv_regime": "NORMAL",
        # Risk
        "financial_score": 0.7,
        "severity": "NONE",
        "runway_bucket": "adequate",
        "competitive_intensity_z": 0.2,
        "de_vol_60d": 0.5,
        "de_beta_xbi_60d": 1.0,
        "de_drawdown": -0.15,
        # Forward returns
        "fwd_ret_63d": 0.05,
        "fwd_excess_xbi_63d": 0.02,
        "fwd_ret_20d": 0.02,
        "regime_63d": "neutral",
    }
    base.update(overrides)
    return base


def _make_cohort(n=20, spread_returns=True):
    """Build a cohort with varying signals and returns."""
    rows = []
    for i in range(n):
        f = i / max(n - 1, 1)
        ret = -0.10 + 0.20 * f if spread_returns else 0.05
        rows.append(
            _make_row(
                ticker=f"TK{i:03d}",
                rank=i + 1,
                coinvest_score_z=round(-1.0 + 2.0 * f, 4),
                inst_delta_z=round(-0.5 + 1.0 * f, 4),
                clinical_score_v2_z=round(-1.0 + 2.0 * f, 4),
                financial_score=round(0.2 + 0.6 * f, 4),
                catalyst_decay_w=round(0.3 + 0.5 * f, 4),
                binary_quality_score=round(0.2 + 0.6 * f, 4),
                fwd_ret_63d=round(ret, 4),
                fwd_excess_xbi_63d=round(ret - 0.01, 4),
            )
        )
    return rows


def _make_snapshots(n_dates=20, n_per_date=30):
    """Build synthetic snapshots for training tests."""
    snapshots = {}
    for d in range(n_dates):
        date = f"2022-{(d % 12) + 1:02d}-28"
        if d >= 12:
            date = f"2023-{(d % 12) + 1:02d}-28"
        rows = _make_cohort(n_per_date)
        # Add some noise to returns
        import random

        rng = random.Random(42 + d)
        for row in rows:
            noise = rng.gauss(0, 0.03)
            base_ret = float(row["fwd_ret_63d"])
            row["fwd_ret_63d"] = round(base_ret + noise, 4)
            row["fwd_excess_xbi_63d"] = round(base_ret + noise - 0.01, 4)
        snapshots[date] = rows
    return snapshots


# ---------------------------------------------------------------------------
# _sf / _sigmoid
# ---------------------------------------------------------------------------


class TestSafeFloat:
    def test_valid(self):
        assert _sf(1.5) == 1.5
        assert _sf("2.3") == 2.3
        assert _sf(0) == 0.0

    def test_none(self):
        assert math.isnan(_sf(None))

    def test_empty(self):
        assert math.isnan(_sf(""))

    def test_nan(self):
        assert math.isnan(_sf(float("nan")))

    def test_custom_default(self):
        assert _sf(None, 0.0) == 0.0


class TestSigmoid:
    def test_zero(self):
        assert _sigmoid(0.0) == pytest.approx(0.5)

    def test_large_positive(self):
        assert _sigmoid(100.0) == pytest.approx(1.0, abs=1e-6)

    def test_large_negative(self):
        assert _sigmoid(-100.0) == pytest.approx(0.0, abs=1e-6)

    def test_symmetry(self):
        assert _sigmoid(1.0) + _sigmoid(-1.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


class TestFeatureExtraction:
    def test_numeric_feature(self):
        row = {"coinvest_score_z": 1.5}
        spec = FeatureSpec("coinvest_score_z")
        assert _encode_feature(row, spec) == 1.5

    def test_numeric_inverted(self):
        row = {"cat_priority": 2}
        spec = FeatureSpec("cat_priority", higher_is_better=False)
        assert _encode_feature(row, spec) == -2.0

    def test_categorical(self):
        row = {"catalyst_family": "REGULATORY"}
        spec = FeatureSpec(
            "catalyst_family",
            categorical=True,
            value_map=(("REGULATORY", 1.0), ("CLINICAL", 0.6), ("", 0.0)),
        )
        assert _encode_feature(row, spec) == 1.0

    def test_categorical_inverted(self):
        row = {"severity": "SEV2"}
        spec = FeatureSpec(
            "severity",
            higher_is_better=False,
            categorical=True,
            value_map=(("NONE", 0.0), ("SEV2", 0.67)),
        )
        assert _encode_feature(row, spec) == pytest.approx(1.0 - 0.67)

    def test_missing_numeric(self):
        row = {}
        spec = FeatureSpec("coinvest_score_z")
        assert math.isnan(_encode_feature(row, spec))

    def test_missing_categorical_with_default(self):
        row = {"opt_iv_regime": ""}
        spec = FeatureSpec(
            "opt_iv_regime",
            categorical=True,
            value_map=(("LOW", 0.8), ("NORMAL", 0.5), ("", 0.5)),
        )
        assert _encode_feature(row, spec) == 0.5

    def test_extract_features_vector(self):
        row = _make_row()
        specs = list(FEATURES_MINIMAL)
        vec = extract_features(row, specs)
        assert len(vec) == len(FEATURES_MINIMAL)
        assert all(isinstance(v, float) for v in vec)


class TestGetFeatureSpecs:
    def test_minimal(self):
        config = RankerV2Config(feature_set="minimal")
        specs = get_feature_specs(config)
        assert len(specs) == len(FEATURES_MINIMAL)

    def test_expanded(self):
        config = RankerV2Config(feature_set="expanded")
        specs = get_feature_specs(config)
        # Expanded = all numeric (non-categorical) signals from all blocks
        total = sum(1 for b in ALL_BLOCKS.values() for s in b if not s.categorical)
        assert len(specs) == total

    def test_ablation_drop(self):
        config = RankerV2Config(feature_set="ablation_drop_options")
        specs = get_feature_specs(config)
        names = {s.name for s in specs}
        assert "ovf_composite" not in names
        assert "coinvest_score_z" in names


# ---------------------------------------------------------------------------
# Z-scoring
# ---------------------------------------------------------------------------


class TestZscoring:
    def test_basic_zscore(self):
        rows = _make_cohort(10)
        specs = [FeatureSpec("coinvest_score_z")]
        result = zscore_cohort_features(rows, specs)
        assert len(result) == 10
        # Mean should be ~0
        mean = sum(r[0] for r in result) / 10
        assert abs(mean) < 0.01

    def test_missing_imputed_to_zero(self):
        rows = [_make_row(coinvest_score_z=1.0), _make_row(coinvest_score_z=None)]
        specs = [FeatureSpec("coinvest_score_z")]
        result = zscore_cohort_features(rows, specs)
        assert result[1][0] == 0.0  # missing imputed to mean (0 in z-space)

    def test_clamped_to_3(self):
        rows = [_make_row(coinvest_score_z=v) for v in [0, 0, 0, 0, 0, 0, 0, 0, 0, 100]]
        specs = [FeatureSpec("coinvest_score_z")]
        result = zscore_cohort_features(rows, specs)
        assert result[-1][0] <= 3.0


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------


class TestPairGeneration:
    def test_basic_pairs(self):
        returns = [0.10, 0.05, -0.02, 0.08]
        pairs = generate_pairs(returns, max_pairs=100)
        assert len(pairs) == 6  # C(4,2) = 6

    def test_labels_correct(self):
        returns = [0.10, -0.05]
        pairs = generate_pairs(returns, max_pairs=100)
        assert len(pairs) == 1
        assert pairs[0].label == 1.0  # 0.10 > -0.05 → i outranks j

    def test_ties_excluded(self):
        returns = [0.10, 0.10]
        pairs = generate_pairs(returns, max_pairs=100)
        assert len(pairs) == 0

    def test_sampling(self):
        returns = [i * 0.01 for i in range(50)]
        pairs = generate_pairs(returns, max_pairs=100)
        assert len(pairs) <= 100

    def test_nan_skipped(self):
        returns = [0.10, float("nan"), -0.05]
        pairs = generate_pairs(returns, max_pairs=100)
        assert len(pairs) == 1  # only (0, 2) valid

    def test_too_few(self):
        pairs = generate_pairs([0.10], max_pairs=100)
        assert len(pairs) == 0

    def test_weight_passed(self):
        returns = [0.10, -0.05]
        pairs = generate_pairs(returns, max_pairs=100, sample_weight=0.5)
        assert pairs[0].weight == 0.5


# ---------------------------------------------------------------------------
# Recency weighting
# ---------------------------------------------------------------------------


class TestRecencyWeight:
    def test_same_date(self):
        assert compute_recency_weight("2024-06-30", "2024-06-30") == 1.0

    def test_24_month_halflife(self):
        w = compute_recency_weight("2022-06-30", "2024-06-30", halflife_months=24)
        assert w == pytest.approx(0.5, abs=0.01)

    def test_very_old(self):
        w = compute_recency_weight("2015-01-01", "2024-06-30", halflife_months=24)
        assert w >= 0.01  # floor

    def test_future_date(self):
        w = compute_recency_weight("2025-01-01", "2024-06-30")
        assert w == 1.0  # future → no decay


# ---------------------------------------------------------------------------
# Cohort filtering
# ---------------------------------------------------------------------------


class TestCohortFilter:
    def test_basic_filter(self):
        rows = _make_cohort(100)
        config = RankerV2Config(cohort_top_n=30)
        cohort = filter_cohort(rows, config)
        assert len(cohort) == 30

    def test_eligible_filter(self):
        rows = [_make_row(rank=1, eligible=0), _make_row(rank=2, eligible=1)]
        config = RankerV2Config(cohort_top_n=60)
        cohort = filter_cohort(rows, config)
        assert len(cohort) == 1

    def test_catalyst_window(self):
        rows = [
            _make_row(rank=1, catalyst_in_window="1"),
            _make_row(rank=2, catalyst_in_window="0", catalyst_days=200),
        ]
        config = RankerV2Config(cohort_top_n=60, require_catalyst_window=True)
        cohort = filter_cohort(rows, config)
        assert len(cohort) == 1


# ---------------------------------------------------------------------------
# Pairwise logistic model
# ---------------------------------------------------------------------------


class TestPairwiseLogistic:
    def test_untrained_predicts_half(self):
        model = PairwiseLogisticModel(weights=[], n_features=0)
        assert model.predict_pair([], []) == 0.5

    def test_training_converges(self):
        # Linear separable: higher feature → higher return
        n = 20
        features = [[i / n] for i in range(n)]
        returns = [i / n for i in range(n)]
        pairs = generate_pairs(returns, max_pairs=200)

        model = train_pairwise_logistic(features, pairs, n_features=1, lr=0.05, n_epochs=100)
        assert model.trained
        assert model.train_accuracy > 0.7  # should learn the ordering
        assert model.weights[0] > 0  # positive weight for positive signal

    def test_score_name_ordering(self):
        """Higher features should get higher scores."""
        model = PairwiseLogisticModel(weights=[1.0], bias=0.0, n_features=1)
        features = [[0.0], [0.5], [1.0]]
        scores = [model.score_name(features[i], features, i) for i in range(3)]
        assert scores[2] > scores[1] > scores[0]


# ---------------------------------------------------------------------------
# Pointwise logistic model
# ---------------------------------------------------------------------------


class TestPointwiseLogistic:
    def test_untrained(self):
        model = PointwiseLogisticModel(weights=[], n_features=0)
        assert model.predict([]) == 0.5

    def test_training_basic(self):
        features = [[i / 10] for i in range(10)]
        labels = [1.0 if i >= 5 else 0.0 for i in range(10)]
        weights = [1.0] * 10

        model = train_pointwise_logistic(features, labels, weights, n_features=1, lr=0.1, n_epochs=100)
        assert model.trained
        assert model.weights[0] > 0


# ---------------------------------------------------------------------------
# Spearman IC
# ---------------------------------------------------------------------------


class TestSpearman:
    def test_perfect_positive(self):
        x = [1, 2, 3, 4, 5]
        y = [10, 20, 30, 40, 50]
        assert _spearman(x, y) == pytest.approx(1.0, abs=0.01)

    def test_perfect_negative(self):
        x = [1, 2, 3, 4, 5]
        y = [50, 40, 30, 20, 10]
        assert _spearman(x, y) == pytest.approx(-1.0, abs=0.01)

    def test_too_few(self):
        assert math.isnan(_spearman([1, 2], [3, 4]))


# ---------------------------------------------------------------------------
# End-to-end train_and_evaluate
# ---------------------------------------------------------------------------


class TestTrainAndEvaluate:
    pytestmark = pytest.mark.slow

    def test_pairwise_e2e(self):
        snapshots = _make_snapshots(n_dates=18, n_per_date=20)
        config = RankerV2Config(
            model_variant="pairwise_logistic",
            feature_set="minimal",
            cohort_top_n=20,
            min_train_dates=12,
            n_epochs=50,
            portfolio_top_n=10,
        )
        result = train_and_evaluate(snapshots, config)
        assert len(result.oos_results) > 0
        assert result.pairwise_model is not None
        assert result.pairwise_model.trained

    def test_pointwise_e2e(self):
        snapshots = _make_snapshots(n_dates=18, n_per_date=20)
        config = RankerV2Config(
            model_variant="pointwise_logistic",
            feature_set="minimal",
            cohort_top_n=20,
            min_train_dates=12,
            n_epochs=50,
            portfolio_top_n=10,
        )
        result = train_and_evaluate(snapshots, config)
        assert len(result.oos_results) > 0
        assert result.pointwise_model is not None

    def test_baseline_e2e(self):
        snapshots = _make_snapshots(n_dates=18, n_per_date=20)
        config = RankerV2Config(
            model_variant="baseline_bounded",
            cohort_top_n=20,
            min_train_dates=12,
            portfolio_top_n=10,
        )
        result = train_and_evaluate(snapshots, config)
        assert len(result.oos_results) > 0

    def test_too_few_dates(self):
        snapshots = _make_snapshots(n_dates=5, n_per_date=20)
        config = RankerV2Config(min_train_dates=12)
        result = train_and_evaluate(snapshots, config)
        assert len(result.oos_results) == 0


# ---------------------------------------------------------------------------
# Model serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_round_trip(self):
        model = PairwiseLogisticModel(
            weights=[0.5, -0.3, 0.1],
            bias=0.02,
            n_features=3,
            feature_names=["a", "b", "c"],
            trained=True,
            train_loss=0.5,
            train_accuracy=0.7,
        )
        d = model_to_dict(model)
        restored = model_from_dict(d)
        assert restored.weights == model.weights
        assert restored.bias == model.bias
        assert restored.n_features == model.n_features
        assert restored.feature_names == model.feature_names
        assert restored.trained == model.trained

    def test_json_serializable(self):
        model = PairwiseLogisticModel(weights=[1.0], bias=0.0, n_features=1)
        d = model_to_dict(model)
        s = json.dumps(d)
        assert isinstance(s, str)


# ---------------------------------------------------------------------------
# Config ID
# ---------------------------------------------------------------------------


class TestConfigId:
    def test_deterministic(self):
        c1 = RankerV2Config()
        c2 = RankerV2Config()
        assert config_id(c1) == config_id(c2)

    def test_changes_with_config(self):
        c1 = RankerV2Config(cohort_top_n=30)
        c2 = RankerV2Config(cohort_top_n=60)
        assert config_id(c1) != config_id(c2)


# ---------------------------------------------------------------------------
# score_snapshot
# ---------------------------------------------------------------------------


class TestScoreSnapshot:
    def test_basic(self):
        model = PairwiseLogisticModel(
            weights=[1.0] * 5,
            bias=0.0,
            n_features=5,
            trained=True,
        )
        rows = _make_cohort(20)
        config = RankerV2Config(feature_set="minimal", cohort_top_n=20)
        results = score_snapshot(rows, model, config)
        assert len(results) == 20
        assert all("ranker_v2_score" in r for r in results)
        # Should have ranks assigned
        ranks = [r["ranker_v2_rank"] for r in results if r["ranker_v2_rank"] is not None]
        assert sorted(ranks) == list(range(1, 21))

    def test_untrained_model(self):
        model = PairwiseLogisticModel()
        rows = _make_cohort(5)
        results = score_snapshot(rows, model)
        assert all(r["ranker_v2_score"] is None for r in results)
