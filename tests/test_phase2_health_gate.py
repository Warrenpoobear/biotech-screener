"""Tests for the Phase-2 health gate (compute_health_gate / generate_health_json)."""
from __future__ import annotations

import pandas as pd
import pytest
from pathlib import Path

from run_phase2_snapshot_delta import (
    CatalystCoverage,
    DeltaResult,
    HealthResult,
    SingleResult,
    SnapshotData,
    compute_health_gate,
    generate_health_json,
    generate_report,
    FAIL_A_COUNT_MIN,
    FAIL_CATALYST_COVERAGE_MIN,
    FAIL_OPTIONALITY_COVERAGE_MIN,
    PHASE2_A_FLOOR,
    PHASE2_PINNED_RULESET_ID,
    WARN_A_COUNT_LOW,
    WARN_CATALYST_DROP_PP,
    WARN_TURNOVER_PCT,
    WARN_WEIGHT_L1_PCT,
)


# ---------------------------------------------------------------------------
# Synthetic data builders (same helpers as test_phase2_snapshot_delta.py)
# ---------------------------------------------------------------------------

def _make_rankings_df(
    tickers: list[str],
    archetypes: list[str] | None = None,
    tiers: list[str] | None = None,
    catalyst_modes: list[str] | None = None,
    catalyst_days: list | None = None,
    actionable_ranks: list | None = None,
    risk_flags: list[str] | None = None,
    weights: list | None = None,
    ruleset_id: str = PHASE2_PINNED_RULESET_ID,
    size_bands: list[str] | None = None,
    missing_components: list[str] | None = None,
) -> pd.DataFrame:
    n = len(tickers)
    data = {
        "ticker": tickers,
        "archetype": archetypes or ["drug_developer"] * n,
        "tier_dev": tiers or ["B"] * n,
        "catalyst_mode": catalyst_modes or ["specific_days"] * n,
        "catalyst_days": catalyst_days or list(range(10, 10 + n)),
        "actionable_rank": actionable_ranks or list(range(1, n + 1)),
        "risk_flags": risk_flags or [""] * n,
        "target_weight_pct": weights or [5.0] * n,
        "decision_engine_ruleset_id": [ruleset_id] * n,
        "composite_rank": list(range(1, n + 1)),
        "composite_score": [50.0] * n,
        "clinical_optionality_pct_dev": [0.6] * n,
        "size_band": size_bands or ["M"] * n,
        "decision_engine_version": ["v1.2.0"] * n,
        "missing_components": missing_components or [""] * n,
    }
    return pd.DataFrame(data)


def _make_portfolio_df(
    tickers: list[str],
    tiers: list[str] | None = None,
    weights: list | None = None,
    size_bands: list[str] | None = None,
    risk_flags: list[str] | None = None,
    actionable_ranks: list | None = None,
) -> pd.DataFrame:
    n = len(tickers)
    data = {
        "ticker": tickers,
        "tier_dev": tiers or ["B"] * n,
        "target_weight_pct": weights or [5.0] * n,
        "size_band": size_bands or ["M"] * n,
        "risk_flags": risk_flags or [""] * n,
        "actionable_rank": actionable_ranks or list(range(1, n + 1)),
        "archetype": ["drug_developer"] * n,
        "catalyst_mode": ["specific_days"] * n,
        "catalyst_days": list(range(1, n + 1)),
        "decision_engine_ruleset_id": [PHASE2_PINNED_RULESET_ID] * n,
        "decision_engine_version": ["v1.2.0"] * n,
    }
    return pd.DataFrame(data)


def _make_snapshot(
    date: str,
    rankings: pd.DataFrame,
    portfolio: pd.DataFrame,
    ruleset_id: str = PHASE2_PINNED_RULESET_ID,
    has_native: bool = True,
) -> SnapshotData:
    return SnapshotData(
        date=date,
        path=Path(f"/fake/{date}"),
        rankings=rankings,
        portfolio=portfolio,
        metadata={"as_of_date": date},
        ruleset_id=ruleset_id,
        has_native_portfolio=has_native,
    )


def _clean_snapshot(n_a=10, n_b=10, catalyst_pct=80):
    """Build a clean snapshot that should pass all gates."""
    n_total = n_a + n_b + 5  # 5 extra C/D
    tickers = [f"T{i:03d}" for i in range(n_total)]
    tiers = ["A"] * n_a + ["B"] * n_b + ["C"] * 3 + ["D"] * 2
    # catalyst coverage: first catalyst_pct% are specific_days
    n_specific = int(n_total * catalyst_pct / 100)
    cat_modes = ["specific_days"] * n_specific + ["no_upcoming"] * (n_total - n_specific)
    rankings = _make_rankings_df(tickers, tiers=tiers, catalyst_modes=cat_modes)
    portfolio = _make_portfolio_df(tickers[:n_a + n_b], tiers=tiers[:n_a + n_b])
    return _make_snapshot("2026-02-09", rankings, portfolio)


# ---------------------------------------------------------------------------
# 1. FAIL: ruleset mismatch
# ---------------------------------------------------------------------------

class TestFailRulesetMismatch:
    def test_non_pinned_ruleset(self):
        snap = _clean_snapshot()
        snap = SnapshotData(
            date=snap.date, path=snap.path, rankings=snap.rankings,
            portfolio=snap.portfolio, metadata=snap.metadata,
            ruleset_id="bad_id", has_native_portfolio=True,
        )
        from run_phase2_snapshot_delta import compute_single_snapshot_summary
        result = compute_single_snapshot_summary(snap)
        health = compute_health_gate(snap, None, result)
        assert health.status == "FAIL"
        assert "ruleset_mismatch" in health.reasons


# ---------------------------------------------------------------------------
# 2. FAIL/WARN: no A-tier (split by optionality coverage)
# ---------------------------------------------------------------------------

class TestNoATierSplit:
    def test_zero_a_low_optionality_coverage(self):
        """Zero A-tier + low optionality coverage → FAIL optionality_broken."""
        # All optionality values are blank/NaN → coverage = 0%
        rankings = _make_rankings_df(
            ["T1", "T2", "T3"],
            tiers=["C", "C", "D"],
            catalyst_modes=["specific_days", "specific_days", "specific_days"],
        )
        rankings["clinical_optionality_pct_dev"] = ""  # blank = no data
        portfolio = _make_portfolio_df(["T1", "T2"])
        snap = _make_snapshot("2026-02-09", rankings, portfolio)
        from run_phase2_snapshot_delta import compute_single_snapshot_summary
        result = compute_single_snapshot_summary(snap)
        health = compute_health_gate(snap, None, result)
        assert health.status == "FAIL"
        assert "optionality_broken" in health.reasons
        assert "no_a_tier" not in health.reasons
        # Metrics should include diagnostic
        assert "dev_optionality_coverage_pct" in health.metrics
        assert health.metrics["dev_optionality_coverage_pct"] == 0.0

    def test_zero_a_good_optionality_coverage(self):
        """Zero A-tier + good optionality coverage → WARN no_a_tier_regime."""
        # Full optionality data (all 0.6), zero A-tier because no near catalyst
        rankings = _make_rankings_df(
            ["T1", "T2", "T3"],
            tiers=["C", "C", "D"],
            catalyst_modes=["specific_days", "specific_days", "specific_days"],
        )
        # clinical_optionality_pct_dev defaults to 0.6 in _make_rankings_df
        portfolio = _make_portfolio_df(["T1", "T2"])
        snap = _make_snapshot("2026-02-09", rankings, portfolio)
        from run_phase2_snapshot_delta import compute_single_snapshot_summary
        result = compute_single_snapshot_summary(snap)
        health = compute_health_gate(snap, None, result)
        assert health.status == "WARN"
        assert "no_a_tier_regime" in health.reasons
        assert "optionality_broken" not in health.reasons
        # Metrics should include diagnostic
        assert health.metrics["dev_optionality_coverage_pct"] == 100.0
        assert health.metrics["dev_above_a_floor_count"] == 3  # all 0.6 >= 0.55


# ---------------------------------------------------------------------------
# 3. FAIL: catalyst broken
# ---------------------------------------------------------------------------

class TestFailCatalystBroken:
    def test_coverage_below_fail_threshold(self):
        # 10% coverage: 1 of 10 is specific_days
        tickers = [f"T{i}" for i in range(10)]
        cat_modes = ["specific_days"] + ["no_upcoming"] * 9
        rankings = _make_rankings_df(
            tickers,
            tiers=["A"] * 5 + ["B"] * 5,
            catalyst_modes=cat_modes,
        )
        portfolio = _make_portfolio_df(tickers[:5], tiers=["A"] * 5)
        snap = _make_snapshot("2026-02-09", rankings, portfolio)
        from run_phase2_snapshot_delta import compute_single_snapshot_summary
        result = compute_single_snapshot_summary(snap)
        health = compute_health_gate(snap, None, result)
        assert health.status == "FAIL"
        assert "catalyst_broken" in health.reasons


# ---------------------------------------------------------------------------
# 4. FAIL: empty portfolio
# ---------------------------------------------------------------------------

class TestFailEmptyPortfolio:
    def test_no_names(self):
        rankings = _make_rankings_df(
            ["T1", "T2"],
            tiers=["A", "B"],
            catalyst_modes=["specific_days", "specific_days"],
        )
        portfolio = _make_portfolio_df([])  # empty
        snap = _make_snapshot("2026-02-09", rankings, portfolio)
        from run_phase2_snapshot_delta import compute_single_snapshot_summary
        result = compute_single_snapshot_summary(snap)
        health = compute_health_gate(snap, None, result)
        assert health.status == "FAIL"
        assert "no_portfolio" in health.reasons


# ---------------------------------------------------------------------------
# 5. WARN: high turnover
# ---------------------------------------------------------------------------

class TestWarnHighTurnover:
    def test_100pct_turnover(self):
        from run_phase2_snapshot_delta import compute_delta
        cur_snap = _clean_snapshot()
        # Build a prior with completely different tickers
        n = len(cur_snap.portfolio)
        pri_tickers = [f"P{i:03d}" for i in range(n)]
        pri_portfolio = _make_portfolio_df(pri_tickers)
        # Rankings must contain all tickers
        all_tickers = cur_snap.rankings["ticker"].tolist() + pri_tickers
        all_tiers = cur_snap.rankings["tier_dev"].tolist() + ["B"] * n
        all_cat = cur_snap.rankings["catalyst_mode"].tolist() + ["specific_days"] * n
        pri_rankings = _make_rankings_df(all_tickers, tiers=all_tiers, catalyst_modes=all_cat)
        pri_snap = _make_snapshot("2026-01-20", pri_rankings, pri_portfolio)

        result = compute_delta(cur_snap, pri_snap)
        health = compute_health_gate(cur_snap, pri_snap, result)
        assert health.status == "WARN"
        assert "high_turnover" in health.reasons


# ---------------------------------------------------------------------------
# 6. WARN: catalyst drop
# ---------------------------------------------------------------------------

class TestWarnCatalystDrop:
    def test_large_coverage_drop(self):
        from run_phase2_snapshot_delta import compute_delta
        # Current: 50% coverage (5 of 10 specific_days)
        cur_tickers = [f"T{i}" for i in range(10)]
        cur_rankings = _make_rankings_df(
            cur_tickers,
            tiers=["A"] * 5 + ["B"] * 5,
            catalyst_modes=["specific_days"] * 5 + ["no_upcoming"] * 5,
        )
        cur_portfolio = _make_portfolio_df(cur_tickers[:10], tiers=["A"] * 5 + ["B"] * 5)
        cur_snap = _make_snapshot("2026-02-09", cur_rankings, cur_portfolio)

        # Prior: 80% coverage (8 of 10 specific_days) — drop of 30pp > 25pp threshold
        pri_rankings = _make_rankings_df(
            cur_tickers,
            tiers=["A"] * 5 + ["B"] * 5,
            catalyst_modes=["specific_days"] * 8 + ["no_upcoming"] * 2,
        )
        pri_portfolio = _make_portfolio_df(cur_tickers[:10], tiers=["A"] * 5 + ["B"] * 5)
        pri_snap = _make_snapshot("2026-01-20", pri_rankings, pri_portfolio)

        result = compute_delta(cur_snap, pri_snap)
        health = compute_health_gate(cur_snap, pri_snap, result)
        assert health.status == "WARN"
        assert "catalyst_drop" in health.reasons


# ---------------------------------------------------------------------------
# 7. WARN: low A-tier (above 0 but below 5)
# ---------------------------------------------------------------------------

class TestWarnLowATier:
    def test_a_count_between_thresholds(self):
        # 1 A-tier name: above FAIL_A_COUNT_MIN (1) but below WARN_A_COUNT_LOW (2)
        tickers = [f"T{i}" for i in range(10)]
        tiers = ["A"] * 1 + ["B"] * 6 + ["C"] * 3
        rankings = _make_rankings_df(
            tickers,
            tiers=tiers,
            catalyst_modes=["specific_days"] * 10,
        )
        portfolio = _make_portfolio_df(tickers[:7], tiers=tiers[:7])
        snap = _make_snapshot("2026-02-09", rankings, portfolio)
        from run_phase2_snapshot_delta import compute_single_snapshot_summary
        result = compute_single_snapshot_summary(snap)
        health = compute_health_gate(snap, None, result)
        assert health.status == "WARN"
        assert "low_a_tier" in health.reasons


# ---------------------------------------------------------------------------
# 8. OK: clean snapshot
# ---------------------------------------------------------------------------

class TestOkClean:
    def test_everything_within_bounds(self):
        snap = _clean_snapshot()
        from run_phase2_snapshot_delta import compute_single_snapshot_summary
        result = compute_single_snapshot_summary(snap)
        health = compute_health_gate(snap, None, result)
        assert health.status == "OK"
        assert health.reasons == []


# ---------------------------------------------------------------------------
# 9. FAIL trumps WARN
# ---------------------------------------------------------------------------

class TestFailTrumpsWarn:
    def test_fail_and_warn_both_present(self):
        """When both FAIL and WARN conditions are met, status should be FAIL."""
        from run_phase2_snapshot_delta import compute_delta

        # FAIL: ruleset mismatch + low A (WARN condition: low_a_tier with 1 A < 2)
        cur_tickers = [f"T{i}" for i in range(10)]
        cur_tiers = ["A"] * 1 + ["B"] * 6 + ["C"] * 3
        cur_rankings = _make_rankings_df(
            cur_tickers,
            tiers=cur_tiers,
            catalyst_modes=["specific_days"] * 10,
        )
        cur_portfolio = _make_portfolio_df(cur_tickers[:7], tiers=cur_tiers[:7])
        cur_snap = _make_snapshot("2026-02-09", cur_rankings, cur_portfolio, ruleset_id="wrong_id")

        # Make prior identical for delta mode
        pri_snap = _make_snapshot("2026-01-20", cur_rankings, cur_portfolio)

        result = compute_delta(cur_snap, pri_snap)
        health = compute_health_gate(cur_snap, pri_snap, result)
        assert health.status == "FAIL"
        assert "ruleset_mismatch" in health.reasons
        # WARN reasons should NOT appear when FAIL is present
        assert "low_a_tier" not in health.reasons


# ---------------------------------------------------------------------------
# 10. Health JSON structure
# ---------------------------------------------------------------------------

class TestHealthJsonStructure:
    def test_keys_and_types(self):
        snap = _clean_snapshot()
        from run_phase2_snapshot_delta import compute_single_snapshot_summary
        result = compute_single_snapshot_summary(snap)
        health = compute_health_gate(snap, None, result)
        hj = generate_health_json(health)

        assert isinstance(hj, dict)
        assert hj["status"] in ("OK", "WARN", "FAIL")
        assert isinstance(hj["reasons"], list)
        assert isinstance(hj["metrics"], dict)
        assert isinstance(hj["thresholds"], dict)

        # Thresholds should contain all gate constants
        assert "fail_catalyst_coverage_min" in hj["thresholds"]
        assert "fail_a_count_min" in hj["thresholds"]
        assert "fail_optionality_coverage_min" in hj["thresholds"]
        assert "warn_a_count_low" in hj["thresholds"]
        assert "warn_turnover_pct" in hj["thresholds"]
        assert "warn_weight_l1_pct" in hj["thresholds"]
        assert "warn_catalyst_drop_pp" in hj["thresholds"]

        # Thresholds ID should be present
        assert "thresholds_id" in hj
        assert isinstance(hj["thresholds_id"], str)
        assert len(hj["thresholds_id"]) == 8

        # Metrics should have core fields
        assert "a_count" in hj["metrics"]
        assert "catalyst_coverage_pct" in hj["metrics"]
        assert "portfolio_size" in hj["metrics"]
        assert "ruleset_id" in hj["metrics"]
        assert "thresholds_id" in hj["metrics"]

    def test_report_includes_health_headline(self):
        snap = _clean_snapshot()
        from run_phase2_snapshot_delta import compute_single_snapshot_summary
        result = compute_single_snapshot_summary(snap)
        health = compute_health_gate(snap, None, result)
        report = generate_report(snap, None, result, health=health)
        assert "HEALTH: OK" in report

    def test_report_without_health(self):
        """Report generated without health param should not contain HEALTH line."""
        snap = _clean_snapshot()
        from run_phase2_snapshot_delta import compute_single_snapshot_summary
        result = compute_single_snapshot_summary(snap)
        report = generate_report(snap, None, result)
        assert "HEALTH:" not in report


# ---------------------------------------------------------------------------
# 11. Pinned thresholds ID
# ---------------------------------------------------------------------------

class TestPinnedThresholds:
    def test_default_thresholds_id_pinned(self):
        """Default thresholds must produce the pinned ID."""
        from run_phase2_snapshot_delta import DEFAULT_HEALTH_THRESHOLDS
        assert DEFAULT_HEALTH_THRESHOLDS.thresholds_id == "0dae2ff0"

    def test_production_json_matches_defaults(self):
        """Production v1.json must round-trip to the same thresholds as defaults."""
        from run_phase2_snapshot_delta import (
            Phase2HealthThresholds,
            DEFAULT_HEALTH_THRESHOLDS,
        )
        prod_path = (
            Path(__file__).resolve().parent.parent
            / "production_data" / "phase2_health_thresholds" / "v1.json"
        )
        assert prod_path.exists(), f"Pinned JSON not found: {prod_path}"
        loaded = Phase2HealthThresholds.from_json(str(prod_path))
        assert loaded.thresholds_id == DEFAULT_HEALTH_THRESHOLDS.thresholds_id
        assert loaded == DEFAULT_HEALTH_THRESHOLDS

    def test_to_json_round_trip(self, tmp_path):
        """to_json → from_json should produce identical thresholds."""
        from run_phase2_snapshot_delta import (
            Phase2HealthThresholds,
            DEFAULT_HEALTH_THRESHOLDS,
        )
        path = tmp_path / "test_thresholds.json"
        DEFAULT_HEALTH_THRESHOLDS.to_json(str(path))
        loaded = Phase2HealthThresholds.from_json(str(path))
        assert loaded == DEFAULT_HEALTH_THRESHOLDS
        assert loaded.thresholds_id == DEFAULT_HEALTH_THRESHOLDS.thresholds_id

    def test_custom_thresholds_change_id(self):
        """Changing any field should produce a different thresholds_id."""
        from run_phase2_snapshot_delta import (
            Phase2HealthThresholds,
            DEFAULT_HEALTH_THRESHOLDS,
        )
        custom = Phase2HealthThresholds(warn_a_count_low=10)
        assert custom.thresholds_id != DEFAULT_HEALTH_THRESHOLDS.thresholds_id

    def test_compute_health_gate_uses_custom_thresholds(self):
        """Custom thresholds should override defaults in compute_health_gate."""
        from run_phase2_snapshot_delta import (
            Phase2HealthThresholds,
            compute_single_snapshot_summary,
        )
        # 4 A-tier names: passes default (warn < 3) but fails custom (warn < 5)
        tickers = [f"T{i}" for i in range(10)]
        tiers = ["A"] * 4 + ["B"] * 3 + ["C"] * 3
        rankings = _make_rankings_df(
            tickers, tiers=tiers,
            catalyst_modes=["specific_days"] * 10,
        )
        portfolio = _make_portfolio_df(tickers[:7], tiers=tiers[:7])
        snap = _make_snapshot("2026-02-09", rankings, portfolio)
        result = compute_single_snapshot_summary(snap)

        # Default thresholds (warn_a_count_low=2) → OK (4 >= 2)
        health_default = compute_health_gate(snap, None, result)
        assert health_default.status == "OK"

        # Custom thresholds (warn_a_count_low=5) → WARN (4 < 5)
        custom_th = Phase2HealthThresholds(warn_a_count_low=5)
        health_custom = compute_health_gate(snap, None, result, thresholds=custom_th)
        assert health_custom.status == "WARN"
        assert "low_a_tier" in health_custom.reasons


# ---------------------------------------------------------------------------
# 12. Missingness guardrails
# ---------------------------------------------------------------------------

class TestMissingnessGuardrails:
    """Tests for missingness coverage thresholds in health gate."""

    def test_portfolio_missing_data_warns(self):
        """portfolio_missing_count > 0 → WARN portfolio_missing_data."""
        from run_phase2_snapshot_delta import compute_single_snapshot_summary
        # Use enough tickers so 1 missing sponsor stays above all FAIL thresholds
        tickers = [f"T{i:03d}" for i in range(50)]
        rankings = _make_rankings_df(
            tickers,
            tiers=["A"] * 25 + ["B"] * 25,
            catalyst_modes=["specific_days"] * 50,
            missing_components=["sponsor"] + [""] * 49,  # T000 has missing sponsor (in portfolio)
        )
        portfolio = _make_portfolio_df(tickers[:25], tiers=["A"] * 25)
        snap = _make_snapshot("2026-02-09", rankings, portfolio)
        result = compute_single_snapshot_summary(snap)
        health = compute_health_gate(snap, None, result)
        assert health.status == "WARN"
        assert "portfolio_missing_data" in health.reasons
        assert health.metrics["portfolio_missing_count"] == 1

    def test_portfolio_missing_outside_portfolio_ok(self):
        """Missing data only in non-portfolio tickers → OK (portfolio_missing_count=0)."""
        from run_phase2_snapshot_delta import compute_single_snapshot_summary
        tickers = [f"T{i}" for i in range(10)]
        # T5 (non-portfolio) has missing sponsor — should not trigger WARN
        mc = [""] * 5 + ["sponsor"] + [""] * 4
        rankings = _make_rankings_df(
            tickers,
            tiers=["A"] * 5 + ["B"] * 5,
            catalyst_modes=["specific_days"] * 10,
            missing_components=mc,
        )
        portfolio = _make_portfolio_df(tickers[:5], tiers=["A"] * 5)
        snap = _make_snapshot("2026-02-09", rankings, portfolio)
        result = compute_single_snapshot_summary(snap)
        health = compute_health_gate(snap, None, result)
        assert health.status == "OK"
        assert health.metrics["portfolio_missing_count"] == 0

    def test_drawdown_coverage_fail(self):
        """coverage_drawdown_pct < 95 → FAIL drawdown_coverage_low."""
        from run_phase2_snapshot_delta import compute_single_snapshot_summary
        tickers = [f"T{i}" for i in range(20)]
        # 2/20 dev tickers have missing drawdown → 90% coverage < 95% FAIL
        mc = ["drawdown"] * 2 + [""] * 18
        rankings = _make_rankings_df(
            tickers,
            tiers=["A"] * 10 + ["B"] * 10,
            catalyst_modes=["specific_days"] * 20,
            missing_components=mc,
        )
        portfolio = _make_portfolio_df(tickers[:10], tiers=["A"] * 10)
        snap = _make_snapshot("2026-02-09", rankings, portfolio)
        result = compute_single_snapshot_summary(snap)
        health = compute_health_gate(snap, None, result)
        assert health.status == "FAIL"
        assert "drawdown_coverage_low" in health.reasons
        assert health.metrics["coverage_drawdown_pct"] == 90.0

    def test_drawdown_coverage_warn(self):
        """coverage_drawdown_pct < 99 but >= 95 → WARN drawdown_coverage_low."""
        from run_phase2_snapshot_delta import (
            Phase2HealthThresholds,
            compute_single_snapshot_summary,
        )
        # 1/50 missing → 98% coverage: below WARN(99%) but above FAIL(95%)
        tickers = [f"T{i:03d}" for i in range(50)]
        mc = ["drawdown"] + [""] * 49
        rankings = _make_rankings_df(
            tickers,
            tiers=["A"] * 25 + ["B"] * 25,
            catalyst_modes=["specific_days"] * 50,
            missing_components=mc,
        )
        portfolio = _make_portfolio_df(tickers[:25], tiers=["A"] * 25)
        snap = _make_snapshot("2026-02-09", rankings, portfolio)
        result = compute_single_snapshot_summary(snap)
        health = compute_health_gate(snap, None, result)
        assert health.status == "WARN"
        assert "drawdown_coverage_low" in health.reasons

    def test_sponsor_coverage_warn(self):
        """coverage_sponsor_pct < 90 → WARN sponsor_coverage_low."""
        from run_phase2_snapshot_delta import compute_single_snapshot_summary
        tickers = [f"T{i}" for i in range(10)]
        # 2/10 missing sponsor → 80% < 90%
        mc = ["sponsor"] * 2 + [""] * 8
        rankings = _make_rankings_df(
            tickers,
            tiers=["A"] * 5 + ["B"] * 5,
            catalyst_modes=["specific_days"] * 10,
            missing_components=mc,
        )
        portfolio = _make_portfolio_df(tickers[:5], tiers=["A"] * 5)
        snap = _make_snapshot("2026-02-09", rankings, portfolio)
        result = compute_single_snapshot_summary(snap)
        health = compute_health_gate(snap, None, result)
        assert health.status == "WARN"
        assert "sponsor_coverage_low" in health.reasons
        assert health.metrics["coverage_sponsor_pct"] == 80.0

    def test_catalyst_comp_coverage_warn(self):
        """coverage_catalyst_pct < 85 → WARN catalyst_coverage_low."""
        from run_phase2_snapshot_delta import compute_single_snapshot_summary
        tickers = [f"T{i}" for i in range(10)]
        # 2/10 missing catalyst component → 80% < 85%
        mc = ["catalyst"] * 2 + [""] * 8
        rankings = _make_rankings_df(
            tickers,
            tiers=["A"] * 5 + ["B"] * 5,
            catalyst_modes=["specific_days"] * 10,
            missing_components=mc,
        )
        portfolio = _make_portfolio_df(tickers[:5], tiers=["A"] * 5)
        snap = _make_snapshot("2026-02-09", rankings, portfolio)
        result = compute_single_snapshot_summary(snap)
        health = compute_health_gate(snap, None, result)
        assert health.status == "WARN"
        assert "catalyst_coverage_low" in health.reasons
        assert health.metrics["coverage_catalyst_pct"] == 80.0

    def test_no_missing_components_column_skips_guardrails(self):
        """Old snapshots without missing_components column → no missingness checks."""
        from run_phase2_snapshot_delta import compute_single_snapshot_summary
        snap = _clean_snapshot()
        # Remove the missing_components column to simulate old snapshot
        snap.rankings.drop(columns=["missing_components"], inplace=True)
        result = compute_single_snapshot_summary(snap)
        health = compute_health_gate(snap, None, result)
        assert health.status == "OK"
        assert "portfolio_missing_data" not in health.reasons
        assert "coverage_drawdown_pct" not in health.metrics

    def test_clean_snapshot_still_ok(self):
        """Clean snapshot with missing_components column (all empty) → OK."""
        from run_phase2_snapshot_delta import compute_single_snapshot_summary
        snap = _clean_snapshot()
        result = compute_single_snapshot_summary(snap)
        health = compute_health_gate(snap, None, result)
        assert health.status == "OK"
        assert health.metrics.get("coverage_drawdown_pct") == 100.0
        assert health.metrics.get("coverage_sponsor_pct") == 100.0
        assert health.metrics.get("coverage_catalyst_pct") == 100.0
        assert health.metrics.get("portfolio_missing_count") == 0

    def test_health_json_includes_new_thresholds(self):
        """generate_health_json should include missingness thresholds."""
        from run_phase2_snapshot_delta import compute_single_snapshot_summary
        snap = _clean_snapshot()
        result = compute_single_snapshot_summary(snap)
        health = compute_health_gate(snap, None, result)
        hj = generate_health_json(health)
        assert "fail_drawdown_coverage_min" in hj["thresholds"]
        assert "warn_drawdown_coverage_min" in hj["thresholds"]
        assert "warn_sponsor_coverage_min" in hj["thresholds"]
        assert "warn_catalyst_comp_coverage_min" in hj["thresholds"]
