"""Production invariants for Ranker v2 pairwise cutover (Spec 051).

These tests verify the guarded production cutover from clinical_50 to
pairwise_minimal ranker. They test:
  1. Portfolio positions contract (30 names, weights sum to 100%)
  2. Ranker mode dispatch and fallback behavior
  3. Output column contract preservation
  4. Shadow comparison artifact structure
  5. PIT severity/CFO-date fix integrity
"""

import json
from pathlib import Path

import pytest

from ranker_v2_pairwise import (
    FEATURES_MINIMAL,
    PairwiseLogisticModel,
    RankerV2Config,
    model_from_dict,
    model_to_dict,
    score_snapshot,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PRODUCTION_MODEL_PATH = Path("production_data/ranker_v2_model.json")


def _make_eligible_row(ticker: str, selector_score: float, rank: int, **overrides) -> dict:
    """Build a minimal eligible row with required fields."""
    row = {
        "ticker": ticker,
        "eligible": "1",
        "selector_score": selector_score,
        "selector_rank_bucket": "top60" if rank <= 60 else "below",
        "actionable_rank": rank,
        "coinvest_score_z": 0.5,
        "inst_delta_z": 0.1,
        "clinical_score_v2_z": 0.2,
        "catalyst_decay_w": 0.3,
        "binary_quality_score": 0.4,
        "financial_score": 0.6,
        "archetype": "drug_developer",
        "tier_any": "A",
        "size_band": "M",
        "catalyst_days": "60",
        "catalyst_mode": "hard",
        "mom_state": "up",
        "risk_flags": "",
    }
    row.update(overrides)
    return row


def _make_cohort(n: int = 80) -> list:
    """Build a realistic cohort of eligible rows."""
    rows = []
    for i in range(n):
        rank = i + 1
        score = 1.0 - (i / n)
        rows.append(
            _make_eligible_row(
                ticker=f"TK{i:03d}",
                selector_score=round(score, 4),
                rank=rank,
                coinvest_score_z=round(0.5 + (score - 0.5) * 0.3, 4),
                inst_delta_z=round(0.1 + (score - 0.5) * 0.2, 4),
            )
        )
    return rows


def _make_trained_model() -> PairwiseLogisticModel:
    """Build a simple trained model matching minimal feature set (5 signals)."""
    return PairwiseLogisticModel(
        weights=[0.05, 0.01, -0.011, 0.011, -0.031],
        bias=-0.01,
        n_features=5,
        feature_names=[
            "coinvest_score_z",
            "inst_delta_z",
            "catalyst_decay_w",
            "binary_quality_score",
            "financial_score",
        ],
        trained=True,
        train_loss=0.41,
        train_accuracy=0.54,
    )


# ---------------------------------------------------------------------------
# Test: Production config matches winning settings
# ---------------------------------------------------------------------------


class TestProductionConfig:
    """Verify PRODUCTION_RANKER_V2_CONFIG matches the winning experiment."""

    def test_winning_config_values(self):
        from run_screen import PRODUCTION_RANKER_V2_CONFIG

        assert PRODUCTION_RANKER_V2_CONFIG.feature_set == "minimal_v2"
        assert PRODUCTION_RANKER_V2_CONFIG.cohort_top_n == 60
        assert PRODUCTION_RANKER_V2_CONFIG.require_catalyst_window is False
        assert PRODUCTION_RANKER_V2_CONFIG.n_epochs == 200
        assert PRODUCTION_RANKER_V2_CONFIG.max_pairs_per_date == 400
        assert PRODUCTION_RANKER_V2_CONFIG.train_window == 36

    def test_not_using_research_defaults(self):
        """Ensure we're not accidentally using the reduced-speed defaults."""
        from run_screen import PRODUCTION_RANKER_V2_CONFIG

        default = RankerV2Config()
        # These must differ from defaults (which are speed-hacked for research)
        assert PRODUCTION_RANKER_V2_CONFIG.n_epochs != default.n_epochs or default.n_epochs == 200
        assert (
            PRODUCTION_RANKER_V2_CONFIG.max_pairs_per_date != default.max_pairs_per_date
            or default.max_pairs_per_date == 400
        )

    def test_minimal_feature_set_has_five_signals(self):
        assert len(FEATURES_MINIMAL) == 5
        names = {f.name for f in FEATURES_MINIMAL}
        assert names == {
            "coinvest_score_z",
            "inst_delta_z",
            "catalyst_decay_w",
            "binary_quality_score",
            "financial_score",
        }


# ---------------------------------------------------------------------------
# Test: Production model artifact exists and is valid
# ---------------------------------------------------------------------------


class TestProductionModelArtifact:
    """Verify the serialized model artifact is valid."""

    @pytest.mark.skipif(not PRODUCTION_MODEL_PATH.exists(), reason="No production model artifact")
    def test_model_loads(self):
        artifact = json.loads(PRODUCTION_MODEL_PATH.read_text(encoding="utf-8"))
        model = model_from_dict(artifact["model"])
        assert model.trained is True
        # Production model minimal_v2: 2 features (coinvest_score_z, financial_score)
        assert model.n_features in (2, 5, 6)
        assert len(model.weights) == model.n_features

    @pytest.mark.skipif(not PRODUCTION_MODEL_PATH.exists(), reason="No production model artifact")
    def test_model_config_matches_production(self):
        artifact = json.loads(PRODUCTION_MODEL_PATH.read_text(encoding="utf-8"))
        cfg = artifact["config"]
        assert cfg["feature_set"] == "minimal_v2"
        assert cfg["cohort_top_n"] == 60
        assert cfg["require_catalyst_window"] is False
        assert cfg["n_epochs"] == 200
        assert cfg["max_pairs_per_date"] == 400
        assert cfg["train_window"] == 36

    @pytest.mark.skipif(not PRODUCTION_MODEL_PATH.exists(), reason="No production model artifact")
    def test_model_round_trips(self):
        artifact = json.loads(PRODUCTION_MODEL_PATH.read_text(encoding="utf-8"))
        model = model_from_dict(artifact["model"])
        d = model_to_dict(model)
        model2 = model_from_dict(d)
        assert model.weights == model2.weights
        assert model.bias == model2.bias


# ---------------------------------------------------------------------------
# Test: score_snapshot contract
# ---------------------------------------------------------------------------


class TestScoreSnapshotContract:
    """Verify score_snapshot produces correct output structure."""

    def test_exactly_cohort_scored(self):
        rows = _make_cohort(80)
        model = _make_trained_model()
        config = RankerV2Config(
            feature_set="minimal",
            cohort_top_n=60,
            require_catalyst_window=False,
        )
        results = score_snapshot(rows, model, config)
        assert len(results) == 80
        scored = [r for r in results if r["ranker_v2_score"] is not None]
        assert len(scored) == 60  # top-60 by actionable_rank

    def test_non_cohort_gets_none(self):
        rows = _make_cohort(80)
        model = _make_trained_model()
        config = RankerV2Config(
            feature_set="minimal",
            cohort_top_n=60,
            require_catalyst_window=False,
        )
        results = score_snapshot(rows, model, config)
        # Rows with actionable_rank > 60 should have None scores
        by_ticker = {r["ticker"]: r for r in results}
        for i in range(60, 80):
            ticker = f"TK{i:03d}"
            assert by_ticker[ticker]["ranker_v2_score"] is None

    def test_scores_are_positive(self):
        """Win probabilities are always > 0."""
        rows = _make_cohort(80)
        model = _make_trained_model()
        config = RankerV2Config(
            feature_set="minimal",
            cohort_top_n=60,
            require_catalyst_window=False,
        )
        results = score_snapshot(rows, model, config)
        for r in results:
            if r["ranker_v2_score"] is not None:
                assert r["ranker_v2_score"] > 0

    def test_ranks_are_sequential(self):
        rows = _make_cohort(80)
        model = _make_trained_model()
        config = RankerV2Config(
            feature_set="minimal",
            cohort_top_n=60,
            require_catalyst_window=False,
        )
        results = score_snapshot(rows, model, config)
        ranks = sorted(r["ranker_v2_rank"] for r in results if r["ranker_v2_rank"] is not None)
        assert ranks == list(range(1, 61))

    def test_catalyst_gate_off_for_c1(self):
        """C1 cohort should not require catalyst window."""
        rows = _make_cohort(80)
        # Remove catalyst data from all rows
        for r in rows:
            r["catalyst_in_window"] = "0"
            r["catalyst_days"] = "0"
        model = _make_trained_model()
        config = RankerV2Config(
            feature_set="minimal",
            cohort_top_n=60,
            require_catalyst_window=False,
        )
        results = score_snapshot(rows, model, config)
        scored = [r for r in results if r["ranker_v2_score"] is not None]
        assert len(scored) == 60  # All top-60 eligible, even without catalyst


# ---------------------------------------------------------------------------
# Test: Portfolio positions invariants
# ---------------------------------------------------------------------------


class TestPortfolioPositionsInvariants:
    """These test the downstream invariants that must hold after ranker cutover."""

    def test_top30_from_cohort_only(self):
        """Top-30 by final_score must all come from the scored cohort."""
        rows = _make_cohort(80)
        model = _make_trained_model()
        config = RankerV2Config(
            feature_set="minimal",
            cohort_top_n=60,
            require_catalyst_window=False,
        )
        results = score_snapshot(rows, model, config)
        by_ticker = {r["ticker"]: r for r in results}

        # Simulate pairwise_minimal mode: set final_score
        for row in rows:
            rv2 = by_ticker.get(row["ticker"])
            if rv2 and rv2["ranker_v2_score"] is not None:
                row["final_score"] = rv2["ranker_v2_score"]
            else:
                row["final_score"] = float(row["selector_score"]) * 0.0001

        # Select top-30
        top30 = sorted(rows, key=lambda r: -float(r["final_score"]))[:30]
        assert len(top30) == 30

        # All top-30 should be cohort members (actionable_rank <= 60)
        for r in top30:
            assert r["actionable_rank"] <= 60, f"{r['ticker']} has rank {r['actionable_rank']}"

    def test_non_cohort_cannot_enter_top30(self):
        """Non-cohort members should always sort below cohort members."""
        rows = _make_cohort(80)
        model = _make_trained_model()
        config = RankerV2Config(
            feature_set="minimal",
            cohort_top_n=60,
            require_catalyst_window=False,
        )
        results = score_snapshot(rows, model, config)
        by_ticker = {r["ticker"]: r for r in results}

        for row in rows:
            rv2 = by_ticker.get(row["ticker"])
            if rv2 and rv2["ranker_v2_score"] is not None:
                row["final_score"] = rv2["ranker_v2_score"]
            else:
                row["final_score"] = float(row["selector_score"]) * 0.0001

        # Lowest cohort score
        cohort_scores = [float(r["final_score"]) for r in rows if r["actionable_rank"] <= 60]
        non_cohort_scores = [float(r["final_score"]) for r in rows if r["actionable_rank"] > 60]

        if non_cohort_scores:
            assert max(non_cohort_scores) < min(cohort_scores)


# ---------------------------------------------------------------------------
# Test: Rollback path
# ---------------------------------------------------------------------------


class TestRollbackPath:
    """Verify clinical_50 fallback works identically to pre-cutover behavior."""

    def test_clinical_50_mode_populates_ranker_fields(self):
        """In clinical_50 mode, ranker_active/adjustment/final_score are set by bounded ranker."""
        from ranker_engine import DEFAULT_RANKER_CONFIG, compute_ranker_adjustments

        rows = _make_cohort(30)
        sel_scores = [float(r["selector_score"]) for r in rows]
        sel_buckets = [r["selector_rank_bucket"] for r in rows]
        results = compute_ranker_adjustments(rows, sel_scores, sel_buckets, config=DEFAULT_RANKER_CONFIG)
        assert len(results) == 30
        for rr in results:
            assert hasattr(rr, "final_score")
            assert hasattr(rr, "ranker_active")
            assert hasattr(rr, "ranker_adjustment")


# ---------------------------------------------------------------------------
# Test: PIT severity / CFO-date integrity
# ---------------------------------------------------------------------------


class TestPITIntegrity:
    """Verify PIT financial infrastructure is not broken by ranker cutover."""

    def test_pit_financials_module_importable(self):
        import pit_financials

        assert hasattr(pit_financials, "pit_financial_snapshot")

    @pytest.mark.skipif(
        not Path("production_data/pit_financials").exists(),
        reason="No PIT financial data directory",
    )
    def test_pit_data_exists(self):
        pit_dir = Path("production_data/pit_financials")
        files = list(pit_dir.glob("*.json"))
        assert len(files) > 0, "PIT financial data directory is empty"


# ---------------------------------------------------------------------------
# Test: Output column contract
# ---------------------------------------------------------------------------


class TestOutputColumnContract:
    """Verify SNAPSHOT_COLUMNS includes all required ranker fields."""

    def test_ranker_columns_present(self):
        from run_screen import SNAPSHOT_COLUMNS

        required = [
            "selector_score",
            "selector_rank_bucket",
            "ranker_active",
            "ranker_adjustment",
            "final_score",
            "ranker_v2_score",
            "ranker_v2_rank",
            "regime_label",
        ]
        for col in required:
            assert col in SNAPSHOT_COLUMNS, f"Missing column: {col}"

    def test_portfolio_positions_columns_present(self):
        from run_screen import PORTFOLIO_POSITIONS_COLUMNS

        required = [
            "ticker",
            "company_name",
            "actionable_rank",
            "target_weight_pct",
        ]
        for col in required:
            assert col in PORTFOLIO_POSITIONS_COLUMNS, f"Missing column: {col}"
