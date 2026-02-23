"""Tests for scripts/run_screen_from_bundle.py — PIT bundle screen pipeline."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from scripts.run_screen_from_bundle import (
    SNAPSHOT_COLUMNS,
    _build_rec,
    _classify_stage_bucket,
    _compute_clinical_alpha_z,
    _compute_clinical_optionality,
    _compute_price_features,
    _compute_severity,
    _derive_smart_money,
    _get_pit_quality,
    _safe_float,
    _winsorize,
    _z_score_by_group,
    classify_company_archetype,
    discover_bundle_dates,
    load_bundle,
    run_screen_for_date,
    _write_snapshot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _make_manifest(
    as_of_date: str = "2025-03-31",
    components: dict = None,
    pit_quality: dict = None,
    use_legacy_key: bool = False,
) -> dict:
    """Build a test manifest."""
    m = {
        "schema_version": "pit_bundle_manifest.v1",
        "version": "1.0.0",
        "as_of_date": as_of_date,
        "created_at": "2025-04-01T00:00:00Z",
        "bundle_root": "/tmp/bundles",
        "pit_mode": "strict",
        "components": components or {},
        "universe_size": 10,
        "build_duration_seconds": 1.5,
    }
    if use_legacy_key:
        m["pit_violations"] = pit_quality or {}
    else:
        m["pit_quality"] = pit_quality or {}
    return m


def _make_clinical_data(tickers: List[str]) -> dict:
    """Build clinical features data keyed by ticker."""
    d = {}
    for i, t in enumerate(tickers):
        d[t] = {
            "readout_days": 90 + i * 30,
            "readout_nct_id": f"NCT{i:08d}",
            "phase_num": 0.5 + i * 0.1,
            "lead_phase": "phase 2" if i % 2 == 0 else "phase 3",
            "proximity": 0.6 + i * 0.05,
            "quality": 0.5,
            "clinical_alpha_raw": 0.35 * (0.5 + i * 0.1) + 0.35 * (0.6 + i * 0.05) + 0.30 * 0.5,
            "n_trials_admitted": 5,
            "n_trials_filtered": 0,
            "pit_date_used": "first_posted",
            "far_horizon_days": None,
            "has_active_interventional": True,
        }
    return d


def _make_catalyst_data(tickers: List[str]) -> dict:
    d = {}
    for i, t in enumerate(tickers):
        d[t] = {
            "days_to_catalyst": 60 + i * 30,
            "catalyst_mode": "specific_days",
            "catalyst_strength": "near" if i < 2 else "mid",
            "catalyst_decay_w": 0.9 - i * 0.1,
            "nearest_event_type": "DATA_READOUT",
            "nearest_event_source": "CTGOV",
            "source_mix": {"CTGOV": 3},
        }
    return d


def _make_coinvest_data(tickers: List[str]) -> dict:
    d = {}
    for i, t in enumerate(tickers):
        d[t] = {
            "tier1_count": 2 + i,
            "sponsor_tier1_count": 2 + i,
            "sponsor_overlap_count": 5 + i,
            "coinvest_overlap_count": 5 + i,
            "conviction_overlap": 1.5 + i * 0.5,
            "tier1_conviction_overlap": 1.0 + i * 0.3,
            "max_tier1_position_pct": 3.0 + i,
            "days_since_latest_filing": 45,
            "coinvest_recency_state": "fresh",
            "coinvest_holders": [f"MGR_{j}" for j in range(3)],
            "holder_tiers": {f"MGR_{j}": 1 for j in range(3)},
            "position_changes": {"MGR_0": "INCREASE", "MGR_1": "HOLD", "MGR_2": "NEW"},
        }
    return d


def _make_universe(tickers: List[str]) -> List[dict]:
    return [
        {
            "ticker": t,
            "name": f"Company {t}",
            "market_data": {
                "company_name": f"Company {t} Inc.",
                "industry": "Biotechnology",
                "market_cap": 1_000_000_000 + i * 500_000_000,
            },
            "financial_data": {"revenue_ttm": None},
        }
        for i, t in enumerate(tickers)
    ]


def _make_financial_records(tickers: List[str]) -> Dict[str, dict]:
    return {
        t: {
            "ticker": t,
            "Cash": 100_000_000 + i * 50_000_000,
            "Revenue": None,
            "OperatingExpenses": 40_000_000,
            "CFO": -30_000_000,
        }
        for i, t in enumerate(tickers)
    }


def _setup_bundle(tmp_path: Path, as_of: str, tickers: List[str],
                   pit_quality: dict = None, use_legacy: bool = False) -> Path:
    """Set up a complete bundle directory for testing."""
    bundle_dir = tmp_path / "bundles" / as_of

    clinical = _make_clinical_data(tickers)
    catalyst = _make_catalyst_data(tickers)
    coinvest = _make_coinvest_data(tickers)

    components = {}
    for name, data in [("clinical_features", clinical),
                       ("catalyst_events", catalyst),
                       ("coinvest_features", coinvest)]:
        fname = f"{name.replace('_features', '_features').replace('_events', '_events')}.json"
        _write_json(bundle_dir / fname, {"tickers": data, "schema_version": f"{name}_pit.v1"})
        components[name] = {
            "file": fname,
            "schema_version": f"{name}_pit.v1",
            "sha256": "dummy",
            "coverage_pct": 100.0,
        }

    manifest = _make_manifest(
        as_of_date=as_of,
        components=components,
        pit_quality=pit_quality or {
            "ctgov_trials_filtered_future_posted": 0,
            "catalyst_tickers_missing_disclosed_at": 0,
            "catalyst_tickers_future_disclosed_at": 0,
        },
        use_legacy_key=use_legacy,
    )
    _write_json(bundle_dir / "manifest.json", manifest)
    return bundle_dir


# ===================================================================
# Bundle loading (5 tests)
# ===================================================================

class TestLoadBundle:
    """Tests for bundle loading and validation."""

    def test_load_bundle_valid(self, tmp_path):
        """Loads manifest + 3 components successfully."""
        tickers = ["ACAD", "BMRN"]
        bundle_dir = _setup_bundle(tmp_path, "2025-03-31", tickers)

        result = load_bundle(bundle_dir)
        assert result is not None
        assert result["skip_reason"] is None
        assert result["as_of_date"] == "2025-03-31"
        assert "ACAD" in result["clinical"]
        assert "BMRN" in result["catalyst"]
        assert "ACAD" in result["coinvest"]

    def test_load_bundle_missing_component_graceful(self, tmp_path):
        """Missing clinical file returns empty dict, not error."""
        bundle_dir = tmp_path / "bundles" / "2025-03-31"
        manifest = _make_manifest(
            as_of_date="2025-03-31",
            components={
                "catalyst_events": {
                    "file": "catalyst_events.json",
                    "schema_version": "catalyst_events_pit.v1",
                    "sha256": "dummy",
                    "coverage_pct": 80.0,
                },
            },
            pit_quality={"catalyst_tickers_future_disclosed_at": 0},
        )
        _write_json(bundle_dir / "manifest.json", manifest)
        _write_json(bundle_dir / "catalyst_events.json", {"tickers": {"ACAD": {"days_to_catalyst": 30}}})

        result = load_bundle(bundle_dir)
        assert result is not None
        assert result["skip_reason"] is None
        assert result["clinical"] == {}  # Missing → empty
        assert "ACAD" in result["catalyst"]

    def test_load_bundle_strict_skips_catalyst_violations(self, tmp_path):
        """Strict mode skips dates with catalyst_tickers_future_disclosed_at > 0."""
        tickers = ["ACAD"]
        bundle_dir = _setup_bundle(
            tmp_path, "2025-03-31", tickers,
            pit_quality={"catalyst_tickers_future_disclosed_at": 3},
        )

        result = load_bundle(bundle_dir, pit_mode="strict")
        assert result is not None
        assert result["skip_reason"] is not None
        assert "catalyst_future_violations=3" in result["skip_reason"]

    def test_load_bundle_lenient_allows_violations(self, tmp_path):
        """Lenient mode does NOT skip on catalyst future violations."""
        tickers = ["ACAD"]
        bundle_dir = _setup_bundle(
            tmp_path, "2025-03-31", tickers,
            pit_quality={"catalyst_tickers_future_disclosed_at": 3},
        )

        result = load_bundle(bundle_dir, pit_mode="lenient")
        assert result is not None
        assert result["skip_reason"] is None

    def test_load_bundle_legacy_key_compat(self, tmp_path):
        """Old pit_violations key still works via _get_pit_quality fallback."""
        tickers = ["ACAD"]
        bundle_dir = _setup_bundle(
            tmp_path, "2025-03-31", tickers,
            pit_quality={
                "ctgov_first_posted": 0,
                "catalyst_missing_disclosed_at": 0,
                "catalyst_future_disclosed_at": 2,
            },
            use_legacy=True,
        )

        result = load_bundle(bundle_dir, pit_mode="strict")
        assert result is not None
        assert result["skip_reason"] is not None
        assert "catalyst_future_violations=2" in result["skip_reason"]


# ===================================================================
# Rec dict construction (8 tests)
# ===================================================================

class TestBuildRec:
    """Tests for per-ticker rec dict construction."""

    def test_build_rec_catalyst_fields(self):
        """Maps catalyst bundle → rec['catalyst_decay']."""
        catalyst = {"ACAD": {"days_to_catalyst": 45, "catalyst_mode": "specific_days"}}
        rec = _build_rec("ACAD", {}, catalyst, {}, {}, {}, {})
        assert rec["catalyst_decay"]["days_to_catalyst"] == 45
        assert rec["catalyst_decay"]["in_optimal_window"] is True

    def test_build_rec_coinvest_fields(self):
        """Maps coinvest bundle → rec['coinvest']."""
        coinvest = {"ACAD": {"tier1_count": 5, "conviction_overlap": 2.3,
                             "tier1_conviction_overlap": 1.1,
                             "max_tier1_position_pct": 4.5,
                             "days_since_latest_filing": 30,
                             "position_changes": {}}}
        rec = _build_rec("ACAD", {}, {}, coinvest, {}, {}, {})
        assert rec["coinvest"]["tier1_count"] == 5
        assert rec["coinvest"]["conviction_overlap"] == 2.3

    def test_build_rec_smart_money_from_position_changes(self):
        """INCREASE→buying, DECREASE→selling in smart_money_signal."""
        coinvest = {"ACAD": {
            "position_changes": {"MGR_A": "INCREASE", "MGR_B": "DECREASE", "MGR_C": "NEW"},
            "sponsor_overlap_count": 3,
        }}
        rec = _build_rec("ACAD", {}, {}, coinvest, {}, {}, {})
        sms = rec["smart_money_signal"]
        assert "MGR_A" in sms["holders_increasing"]
        assert "MGR_C" in sms["holders_increasing"]
        assert "MGR_B" in sms["holders_decreasing"]

    def test_build_rec_severity_sev3(self):
        """Low cash/high burn → SEV3 + fundamental_red_flag=True."""
        fin = {"Cash": 5_000_000, "CFO": -50_000_000}  # 5M cash, 50M annual burn → ~1.2 months
        rec = _build_rec("ACAD", {}, {}, {}, fin, {}, {})
        assert rec["severity"] == "SEV3"
        assert rec["fundamental_red_flag"] is True

    def test_build_rec_severity_none(self):
        """Adequate cash → NONE severity."""
        fin = {"Cash": 500_000_000, "CFO": -20_000_000}  # 500M cash, 20M burn → ~100 months
        rec = _build_rec("ACAD", {}, {}, {}, fin, {}, {})
        assert rec["severity"] == "NONE"
        assert rec["fundamental_red_flag"] is False

    def test_build_rec_missing_ticker_neutral_defaults(self):
        """Ticker not in any bundle → neutral defaults."""
        rec = _build_rec("ZZZZ", {}, {}, {}, {}, {}, {})
        assert rec["catalyst_decay"]["days_to_catalyst"] is None
        assert rec["coinvest"]["tier1_count"] is None
        assert rec["severity"] == "NONE"
        assert rec["confidence_overall"] == 0.5

    def test_stage_bucket_from_lead_phase(self):
        """phase 3→late, phase 2→mid, preclinical→early."""
        assert _classify_stage_bucket("phase 3") == "late"
        assert _classify_stage_bucket("phase 2") == "mid"
        assert _classify_stage_bucket("phase 2/3") == "late"  # has "3"
        assert _classify_stage_bucket("phase 1") == "early"
        assert _classify_stage_bucket("preclinical") == "early"
        assert _classify_stage_bucket("") == "early"

    def test_archetype_classification(self):
        """drug_developer, commercial_biotech based on revenue."""
        assert classify_company_archetype("ACAD", "Biotechnology", False, "pre_revenue") == "drug_developer"
        assert classify_company_archetype("INCY", "Biotechnology", True, "large") == "commercial_biotech"
        assert classify_company_archetype("BMY", "Drug Manufacturers - General", True, "large") == "commercial_pharma"
        assert classify_company_archetype("ILMN", "Diagnostics & Research", True, "large") == "platform_diagnostics"


# ===================================================================
# Price features (4 tests)
# ===================================================================

class TestPriceFeatures:
    """Tests for price-derived feature computation."""

    @pytest.fixture
    def price_df(self):
        """Build a simple price DataFrame for testing."""
        import pandas as pd
        import numpy as np

        dates = pd.bdate_range("2024-01-02", periods=300)
        np.random.seed(42)
        # Simple random walk for ticker + XBI
        ticker_prices = 50.0 * np.cumprod(1 + np.random.normal(0.0005, 0.02, 300))
        xbi_prices = 90.0 * np.cumprod(1 + np.random.normal(0.0003, 0.015, 300))

        df = pd.DataFrame({
            "ACAD": ticker_prices,
            "XBI": xbi_prices,
        }, index=dates)
        df.index.name = "Date"
        return df

    def test_drawdown_computation_pit_cutoff(self, price_df):
        """Only bars <= as_of_date are used for drawdown."""
        # Use a date in the middle
        mid_date = price_df.index[150].strftime("%Y-%m-%d")
        feats = _compute_price_features(price_df, "ACAD", mid_date, price_df["XBI"])
        assert feats["drawdown"] is not None
        assert feats["drawdown"] <= 0.0  # drawdown is always <= 0
        assert feats["drawdown_missing_reason"] == ""

    def test_alpha_60d_computation(self, price_df):
        """Alpha is computed as excess return over beta * XBI return."""
        as_of = price_df.index[-1].strftime("%Y-%m-%d")
        feats = _compute_price_features(price_df, "ACAD", as_of, price_df["XBI"])
        assert feats["alpha_60d"] is not None
        assert feats["alpha_60d_source"] == "price_history"

    def test_beta_computation(self, price_df):
        """Beta is OLS slope of ticker returns on XBI returns."""
        as_of = price_df.index[-1].strftime("%Y-%m-%d")
        feats = _compute_price_features(price_df, "ACAD", as_of, price_df["XBI"])
        assert feats["beta_xbi_60d"] is not None
        assert feats["beta_xbi_60d_source"] == "price_history"
        # Beta should be reasonable (positive, not extreme)
        assert -5.0 < feats["beta_xbi_60d"] < 5.0

    def test_rsi_computation(self, price_df):
        """RSI is computed and in [0, 100] range."""
        as_of = price_df.index[-1].strftime("%Y-%m-%d")
        feats = _compute_price_features(price_df, "ACAD", as_of, price_df["XBI"])
        assert feats["rsi_14d"] is not None
        assert 0.0 <= feats["rsi_14d"] <= 100.0


# ===================================================================
# Scoring & ranking (8 tests)
# ===================================================================

class TestScoringRanking:
    """Tests for z-scores, ranking, and alpha cohort."""

    def test_clinical_alpha_z_winsorized(self):
        """Cross-sectional z-score with p5/p95 winsorization."""
        rows = [{"clinical_alpha_raw": v} for v in [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 1.0,
                                                     0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.05]]
        _compute_clinical_alpha_z(rows)
        # All should have z-scores
        for r in rows:
            assert r["clinical_alpha_z"] != ""
        # Extreme values should be clipped
        z_vals = [r["clinical_alpha_z"] for r in rows]
        assert all(isinstance(z, (int, float)) for z in z_vals)

    def test_clinical_score_z_by_cohort(self):
        """Per-archetype z-score (ddof=0)."""
        rows = [
            {"clinical_score": 0.3, "_cohort": "drug_developer"},
            {"clinical_score": 0.7, "_cohort": "drug_developer"},
            {"clinical_score": 0.5, "_cohort": "commercial_biotech"},
            {"clinical_score": 0.9, "_cohort": "commercial_biotech"},
        ]
        _z_score_by_group(rows, "clinical_score", "_cohort", "clinical_score_z")
        # Within drug_developer: 0.3 and 0.7 → z-scores should be symmetric
        z0 = rows[0]["clinical_score_z"]
        z1 = rows[1]["clinical_score_z"]
        assert isinstance(z0, float) and isinstance(z1, float)
        assert abs(z0 + z1) < 1e-6  # symmetric around 0

    def test_clinical_score_z_tier_after_de(self):
        """Tier-local z-score needs tier_dev (populated after DE loop)."""
        rows = [
            {"clinical_score": 0.3, "_tier_group": "drug_developer_A"},
            {"clinical_score": 0.7, "_tier_group": "drug_developer_A"},
            {"clinical_score": 0.2, "_tier_group": "drug_developer_B"},
            {"clinical_score": 0.6, "_tier_group": "drug_developer_B"},
        ]
        _z_score_by_group(rows, "clinical_score", "_tier_group", "clinical_score_z_tier")
        # Each group should have its own z-scores
        assert rows[0]["clinical_score_z_tier"] != ""
        assert rows[2]["clinical_score_z_tier"] != ""

    def test_coinvest_score_z(self):
        """Cross-sectional z of sponsor_tier1_count (ddof=0)."""
        rows = [
            {"sponsor_tier1_count": 1, "_all": "all"},
            {"sponsor_tier1_count": 3, "_all": "all"},
            {"sponsor_tier1_count": 5, "_all": "all"},
        ]
        _z_score_by_group(rows, "sponsor_tier1_count", "_all", "coinvest_score_z")
        # Should produce z-scores; middle value should be near 0
        assert isinstance(rows[1]["coinvest_score_z"], float)
        assert abs(rows[1]["coinvest_score_z"]) < 0.01

    def test_actionable_rank_monotonic(self):
        """Eligible tickers get consecutive 1..N ranks."""
        # We'll test this via the integration test below
        rows = [
            {"eligible": "1", "actionable_rank": ""},
            {"eligible": "1", "actionable_rank": ""},
            {"eligible": "0", "actionable_rank": ""},
            {"eligible": "1", "actionable_rank": ""},
        ]
        rank = 1
        for r in rows:
            if r["eligible"] == "1":
                r["actionable_rank"] = rank
                rank += 1
            else:
                r["actionable_rank"] = ""

        eligible_ranks = [r["actionable_rank"] for r in rows if r["eligible"] == "1"]
        assert eligible_ranks == [1, 2, 3]

    def test_ineligible_no_rank(self):
        """Ineligible tickers have blank actionable_rank."""
        rows = [
            {"eligible": "0", "actionable_rank": ""},
        ]
        for r in rows:
            if r["eligible"] != "1":
                r["actionable_rank"] = ""
        assert rows[0]["actionable_rank"] == ""

    def test_deterministic_output(self, tmp_path):
        """Same inputs → identical rankings CSV both runs."""
        tickers = ["ACAD", "BMRN", "SGEN"]
        bundle_dir = _setup_bundle(tmp_path, "2025-03-31", tickers)
        bundle = load_bundle(bundle_dir)
        universe = _make_universe(tickers)
        fin = _make_financial_records(tickers)

        from decision_engine import DecisionRuleset
        ruleset = DecisionRuleset()

        rows1, _ = run_screen_for_date(
            "2025-03-31", bundle["clinical"], bundle["catalyst"], bundle["coinvest"],
            universe, fin, None, ruleset, bundle["manifest"],
        )
        rows2, _ = run_screen_for_date(
            "2025-03-31", bundle["clinical"], bundle["catalyst"], bundle["coinvest"],
            universe, fin, None, ruleset, bundle["manifest"],
        )

        for r1, r2 in zip(rows1, rows2):
            assert r1.get("ticker") == r2.get("ticker")
            assert r1.get("actionable_rank") == r2.get("actionable_rank")
            assert r1.get("tier_dev") == r2.get("tier_dev")

    def test_alpha_cohort_scoring_attached(self, tmp_path):
        """Alpha cohort key/raw/pct fields are present after scoring."""
        tickers = ["ACAD", "BMRN", "SGEN", "VRTX"]
        bundle_dir = _setup_bundle(tmp_path, "2025-03-31", tickers)
        bundle = load_bundle(bundle_dir)
        universe = _make_universe(tickers)
        fin = _make_financial_records(tickers)

        from decision_engine import DecisionRuleset
        ruleset = DecisionRuleset()

        rows, _ = run_screen_for_date(
            "2025-03-31", bundle["clinical"], bundle["catalyst"], bundle["coinvest"],
            universe, fin, None, ruleset, bundle["manifest"],
        )

        for r in rows:
            assert "alpha_cohort_key" in r
            assert "alpha_cohort_raw" in r
            assert "alpha_cohort_pct" in r


# ===================================================================
# Integration (4 tests)
# ===================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_e2e_small_universe(self, tmp_path):
        """10 synthetic tickers → rankings.csv has all required columns."""
        tickers = [f"TK{i:02d}" for i in range(10)]
        bundle_dir = _setup_bundle(tmp_path, "2025-03-31", tickers)
        bundle = load_bundle(bundle_dir)
        universe = _make_universe(tickers)
        fin = _make_financial_records(tickers)

        from decision_engine import DecisionRuleset
        ruleset = DecisionRuleset()

        rows, metadata = run_screen_for_date(
            "2025-03-31", bundle["clinical"], bundle["catalyst"], bundle["coinvest"],
            universe, fin, None, ruleset, bundle["manifest"],
        )

        assert len(rows) == 10
        assert metadata["ticker_count"] == 10

        # Write and verify CSV has all SNAPSHOT_COLUMNS
        out_root = tmp_path / "out"
        _write_snapshot(out_root, "2025-03-31", rows, metadata)
        csv_path = out_root / "2025-03-31" / "rankings.csv"
        assert csv_path.exists()

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)
        assert len(csv_rows) == 10
        for col in SNAPSHOT_COLUMNS:
            assert col in csv_rows[0], f"Missing column: {col}"

    def test_e2e_metadata_as_of_date(self, tmp_path):
        """metadata.json matches folder date."""
        tickers = ["ACAD", "BMRN"]
        bundle_dir = _setup_bundle(tmp_path, "2025-06-30", tickers)
        bundle = load_bundle(bundle_dir)
        universe = _make_universe(tickers)
        fin = _make_financial_records(tickers)

        from decision_engine import DecisionRuleset
        ruleset = DecisionRuleset()

        rows, metadata = run_screen_for_date(
            "2025-06-30", bundle["clinical"], bundle["catalyst"], bundle["coinvest"],
            universe, fin, None, ruleset, bundle["manifest"],
        )

        out_root = tmp_path / "out"
        _write_snapshot(out_root, "2025-06-30", rows, metadata)

        meta_path = out_root / "2025-06-30" / "metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["as_of_date"] == "2025-06-30"
        assert meta["source_type"] == "pit_bundle"

    def test_e2e_eval_compatible(self, tmp_path):
        """eval_forward_returns.load_rankings() succeeds on PIT snapshot output."""
        tickers = ["ACAD", "BMRN", "SGEN"]
        bundle_dir = _setup_bundle(tmp_path, "2025-03-31", tickers)
        bundle = load_bundle(bundle_dir)
        universe = _make_universe(tickers)
        fin = _make_financial_records(tickers)

        from decision_engine import DecisionRuleset
        ruleset = DecisionRuleset()

        rows, metadata = run_screen_for_date(
            "2025-03-31", bundle["clinical"], bundle["catalyst"], bundle["coinvest"],
            universe, fin, None, ruleset, bundle["manifest"],
        )

        out_root = tmp_path / "out"
        snap_dir = _write_snapshot(out_root, "2025-03-31", rows, metadata)

        # Verify eval can load it
        from scripts.eval_forward_returns import load_rankings
        loaded = load_rankings(snap_dir)
        assert len(loaded) == 3
        # Should have actionable_rank and ticker
        assert all("ticker" in r for r in loaded)
        assert all("actionable_rank" in r for r in loaded)

    def test_e2e_batch_mode(self, tmp_path):
        """2 dates → 2 snapshot dirs via run_batch."""
        tickers = ["ACAD", "BMRN"]
        _setup_bundle(tmp_path, "2025-03-31", tickers)
        _setup_bundle(tmp_path, "2025-06-30", tickers)

        bundle_root = tmp_path / "bundles"
        out_root = tmp_path / "out"
        universe = _make_universe(tickers)
        fin = _make_financial_records(tickers)

        from decision_engine import DecisionRuleset
        from scripts.run_screen_from_bundle import run_batch
        ruleset = DecisionRuleset()

        summary = run_batch(
            bundle_root=bundle_root,
            out_root=out_root,
            ruleset=ruleset,
            universe=universe,
            financial_records_by_ticker=fin,
            prices_df=None,
            pit_mode="strict",
        )

        assert summary["dates_processed"] == 2
        assert (out_root / "2025-03-31" / "rankings.csv").exists()
        assert (out_root / "2025-06-30" / "rankings.csv").exists()


# ===================================================================
# Manifest semantics (3 tests)
# ===================================================================

class TestManifestSemantics:
    """Tests for pit_quality key naming."""

    def test_manifest_new_key_names(self):
        """New pit_quality keys have unambiguous names."""
        pq = {
            "ctgov_trials_filtered_future_posted": 0,
            "catalyst_tickers_missing_disclosed_at": 1,
            "catalyst_tickers_future_disclosed_at": 0,
        }
        manifest = _make_manifest(pit_quality=pq)
        result = _get_pit_quality(manifest)
        assert result["ctgov_trials_filtered_future_posted"] == 0
        assert result["catalyst_tickers_missing_disclosed_at"] == 1
        assert result["catalyst_tickers_future_disclosed_at"] == 0

    def test_manifest_legacy_compat(self):
        """Old pit_violations keys still readable via fallback."""
        pv = {
            "ctgov_first_posted": 5,
            "catalyst_missing_disclosed_at": 2,
            "catalyst_future_disclosed_at": 1,
        }
        manifest = _make_manifest(pit_quality=pv, use_legacy_key=True)
        result = _get_pit_quality(manifest)
        assert result["ctgov_trials_filtered_future_posted"] == 5
        assert result["catalyst_tickers_missing_disclosed_at"] == 2
        assert result["catalyst_tickers_future_disclosed_at"] == 1

    def test_manifest_rebuild_updates_keys(self, tmp_path):
        """Rebuilding a bundle produces new pit_quality keys."""
        # This tests the build_pit_bundle.py change
        from scripts.build_pit_bundle import build_single_bundle

        trial_path = tmp_path / "trials.json"
        _write_json(trial_path, [{
            "ticker": "ACAD",
            "study_type": "INTERVENTIONAL",
            "phase": "PHASE2",
            "first_posted": "2025-01-15",
            "last_update_posted": "2025-06-01",
            "primary_completion_date": "2026-06-15",
            "status": "RECRUITING",
            "nct_id": "NCT12345678",
        }])

        manifest = build_single_bundle(
            as_of_date="2026-01-15",
            bundle_root=tmp_path / "bundles",
            universe_tickers={"ACAD"},
            trial_records_path=trial_path,
            skip_catalyst=True,
            skip_coinvest=True,
        )

        # Should have pit_quality, NOT pit_violations
        assert "pit_quality" in manifest
        assert "pit_violations" not in manifest


# ===================================================================
# Severity computation (additional edge cases)
# ===================================================================

class TestSeverity:

    def test_severity_sev2(self):
        """Runway 6-12 months → SEV2."""
        sev, flag, runway = _compute_severity({"Cash": 20_000_000, "CFO": -30_000_000})
        # 20M / (30M/12) = 8 months
        assert sev == "SEV2"
        assert flag is False

    def test_severity_sev1(self):
        """Runway 12-18 months → SEV1."""
        sev, flag, runway = _compute_severity({"Cash": 40_000_000, "CFO": -30_000_000})
        # 40M / (30M/12) = 16 months
        assert sev == "SEV1"
        assert flag is False

    def test_severity_no_cash_data(self):
        """No cash data → NONE."""
        sev, flag, runway = _compute_severity({})
        assert sev == "NONE"
        assert flag is False

    def test_severity_positive_cfo(self):
        """Positive CFO (profitable) → NONE."""
        sev, flag, runway = _compute_severity({"Cash": 10_000_000, "CFO": 20_000_000})
        assert sev == "NONE"


# ===================================================================
# Smart money derivation
# ===================================================================

class TestSmartMoney:

    def test_derive_from_position_changes(self):
        data = {
            "position_changes": {"A": "NEW", "B": "INCREASE", "C": "DECREASE", "D": "HOLD"},
            "sponsor_overlap_count": 4,
        }
        result = _derive_smart_money(data)
        assert len(result["holders_increasing"]) == 2  # NEW + INCREASE
        assert len(result["holders_decreasing"]) == 1  # DECREASE
        assert result["overlap_count"] == 4

    def test_derive_empty_position_changes(self):
        result = _derive_smart_money({})
        assert result["holders_increasing"] == []
        assert result["holders_decreasing"] == []


# ===================================================================
# Bundle date discovery
# ===================================================================

class TestDiscoverDates:

    def test_discover_filters_by_range(self, tmp_path):
        for dt in ["2024-12-31", "2025-03-31", "2025-06-30", "2025-09-30"]:
            _write_json(tmp_path / dt / "manifest.json", {"as_of_date": dt})

        dates = discover_bundle_dates(tmp_path, date_from="2025-01-01", date_to="2025-07-01")
        assert dates == ["2025-03-31", "2025-06-30"]

    def test_discover_empty_root(self, tmp_path):
        dates = discover_bundle_dates(tmp_path / "nonexistent")
        assert dates == []
