"""Tests for run_phase2_snapshot_delta.py — all synthetic data, no disk I/O."""
from __future__ import annotations

import pandas as pd
import pytest
from pathlib import Path

from run_phase2_snapshot_delta import (
    CatalystCoverage,
    DeltaResult,
    SingleResult,
    SnapshotData,
    _catalyst_coverage,
    _count_risk_flags,
    _parse_risk_flags,
    _reconstruct_portfolio,
    _size_band_counts,
    _tier_counts,
    _tier_in_portfolio,
    compute_delta,
    compute_single_snapshot_summary,
    generate_delta_csv,
    generate_details_json,
    generate_report,
    load_snapshot,
    RECON_TOP_K,
    WARN_A_COUNT_MIN,
    WARN_CATALYST_COVERAGE_MIN,
    WARN_NAME_TURNOVER_PCT,
)


# ---------------------------------------------------------------------------
# Synthetic data builders
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
    ruleset_id: str = "eb833c56",
    size_bands: list[str] | None = None,
) -> pd.DataFrame:
    """Build a synthetic rankings DataFrame with decision engine columns."""
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
        "decision_engine_ruleset_id": ["eb833c56"] * n,
        "decision_engine_version": ["v1.2.0"] * n,
    }
    return pd.DataFrame(data)


def _make_snapshot(
    date: str,
    rankings: pd.DataFrame,
    portfolio: pd.DataFrame,
    ruleset_id: str = "eb833c56",
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


# ---------------------------------------------------------------------------
# Test 1: Turnover math
# ---------------------------------------------------------------------------

class TestTurnoverMath:
    def test_entrants_exits(self):
        """Verify entrants/exits/L1-delta computation."""
        cur_portfolio = _make_portfolio_df(
            ["AAA", "BBB", "CCC"],
            weights=[40.0, 30.0, 30.0],
        )
        pri_portfolio = _make_portfolio_df(
            ["AAA", "BBB", "DDD"],
            weights=[50.0, 25.0, 25.0],
        )
        cur_rankings = _make_rankings_df(["AAA", "BBB", "CCC", "DDD", "EEE"])
        pri_rankings = _make_rankings_df(["AAA", "BBB", "CCC", "DDD", "EEE"])

        cur = _make_snapshot("2026-02-09", cur_rankings, cur_portfolio)
        pri = _make_snapshot("2026-01-20", pri_rankings, pri_portfolio)

        delta = compute_delta(cur, pri)
        assert delta.entrants == ["CCC"]
        assert delta.exits == ["DDD"]
        # turnover: (1+1) / (2*3) * 100 = 33.3%
        assert abs(delta.name_turnover_pct - 33.3) < 0.1
        # L1: |40-50| + |30-25| + |30-0| + |0-25| = 10+5+30+25 = 70
        assert abs(delta.weight_l1_delta - 70.0) < 0.1

    def test_identical_portfolios(self):
        """Identical portfolios => zero turnover."""
        port = _make_portfolio_df(["X", "Y"], weights=[50.0, 50.0])
        rank = _make_rankings_df(["X", "Y"])
        cur = _make_snapshot("2026-02-09", rank, port)
        pri = _make_snapshot("2026-01-20", rank, port)

        delta = compute_delta(cur, pri)
        assert delta.entrants == []
        assert delta.exits == []
        assert delta.name_turnover_pct == 0.0
        assert delta.weight_l1_delta == 0.0

    def test_complete_turnover(self):
        """Completely different portfolios => 100% turnover."""
        cur = _make_portfolio_df(["A", "B"], weights=[50.0, 50.0])
        pri = _make_portfolio_df(["C", "D"], weights=[50.0, 50.0])
        rank = _make_rankings_df(["A", "B", "C", "D"])
        cs = _make_snapshot("2026-02-09", rank, cur)
        ps = _make_snapshot("2026-01-20", rank, pri)

        delta = compute_delta(cs, ps)
        assert delta.name_turnover_pct == 100.0


# ---------------------------------------------------------------------------
# Test 2: Tier drift
# ---------------------------------------------------------------------------

class TestTierDrift:
    def test_tier_counts(self):
        rankings = _make_rankings_df(
            ["A1", "A2", "B1", "C1", "C2", "C3", "D1", "COMM"],
            archetypes=["drug_developer"] * 7 + ["commercial_biotech"],
            tiers=["A", "A", "B", "C", "C", "C", "D", "B"],
        )
        counts = _tier_counts(rankings)
        assert counts == {"A": 2, "B": 1, "C": 3, "D": 1}

    def test_tier_in_portfolio(self):
        port = _make_portfolio_df(
            ["X", "Y", "Z"],
            tiers=["A", "A", "B"],
        )
        counts = _tier_in_portfolio(port)
        assert counts == {"A": 2, "B": 1, "C": 0, "D": 0}


# ---------------------------------------------------------------------------
# Test 3: Catalyst coverage
# ---------------------------------------------------------------------------

class TestCatalystCoverage:
    def test_coverage_pct(self):
        rankings = _make_rankings_df(
            ["A", "B", "C", "D", "E"],
            catalyst_modes=["specific_days", "specific_days", "no_upcoming", "missing", "specific_days"],
        )
        cc = _catalyst_coverage(rankings)
        assert cc.n_dev == 5
        assert cc.n_specific == 3
        assert abs(cc.pct - 60.0) < 0.1
        assert cc.mode_dist["specific_days"] == 3
        assert cc.mode_dist["no_upcoming"] == 1
        assert cc.mode_dist["missing"] == 1

    def test_empty_universe(self):
        rankings = _make_rankings_df(
            ["COMM"],
            archetypes=["commercial_biotech"],
        )
        cc = _catalyst_coverage(rankings)
        assert cc.n_dev == 0
        assert cc.pct == 0.0


# ---------------------------------------------------------------------------
# Test 4: Risk flag counting
# ---------------------------------------------------------------------------

class TestRiskFlags:
    def test_parse_risk_flags(self):
        assert _parse_risk_flags("high_vol|deep_drawdown") == ["high_vol", "deep_drawdown"]
        assert _parse_risk_flags("") == []
        assert _parse_risk_flags(None) == []

    def test_count_risk_flags(self):
        port = _make_portfolio_df(
            ["A", "B", "C"],
            risk_flags=["high_vol|deep_drawdown", "high_vol", ""],
        )
        counts = _count_risk_flags(port)
        assert counts["high_vol"] == 2
        assert counts["deep_drawdown"] == 1
        assert counts["high_beta"] == 0
        assert counts["overbought_rsi"] == 0
        assert counts["low_confidence"] == 0


# ---------------------------------------------------------------------------
# Test 5: Guardrail warnings
# ---------------------------------------------------------------------------

class TestGuardrails:
    def test_turnover_warning_fires(self):
        """Name turnover > threshold triggers warning."""
        cur = _make_portfolio_df(["A", "B", "C", "D"])
        pri = _make_portfolio_df(["E", "F", "G", "H"])
        rank = _make_rankings_df(["A", "B", "C", "D", "E", "F", "G", "H"])
        cs = _make_snapshot("2026-02-09", rank, cur)
        ps = _make_snapshot("2026-01-20", rank, pri)

        delta = compute_delta(cs, ps)
        assert delta.name_turnover_pct == 100.0
        turnover_warnings = [w for w in delta.warnings if "turnover" in w.lower()]
        assert len(turnover_warnings) == 1

    def test_ruleset_change_warning(self):
        """Different ruleset IDs trigger warning."""
        port = _make_portfolio_df(["A"])
        rank = _make_rankings_df(["A"])
        cs = _make_snapshot("2026-02-09", rank, port, ruleset_id="d3cdf5c8")
        ps = _make_snapshot("2026-01-20", rank, port, ruleset_id="d4f1f8a8")

        delta = compute_delta(cs, ps)
        ruleset_warnings = [w for w in delta.warnings if "Ruleset changed" in w]
        assert len(ruleset_warnings) == 1

    def test_low_a_tier_warning(self):
        """Zero A-tier count triggers warning."""
        rank = _make_rankings_df(["A", "B"], tiers=["C", "D"])
        port = _make_portfolio_df(["A"])
        snap = _make_snapshot("2026-02-09", rank, port)

        result = compute_single_snapshot_summary(snap)
        a_warnings = [w for w in result.warnings if "A-tier count" in w]
        assert len(a_warnings) == 1

    def test_low_catalyst_coverage_warning(self):
        """Catalyst coverage below threshold triggers warning."""
        rank = _make_rankings_df(
            ["A", "B", "C", "D"],
            catalyst_modes=["specific_days", "no_upcoming", "missing", "missing"],
        )
        port = _make_portfolio_df(["A"])
        snap = _make_snapshot("2026-02-09", rank, port)

        result = compute_single_snapshot_summary(snap)
        cat_warnings = [w for w in result.warnings if "Catalyst" in w]
        assert len(cat_warnings) == 1

    def test_no_warnings_clean(self):
        """Clean snapshot produces no warnings."""
        # 2 A-tier, 3 specific_days out of 4 dev = 75% > 50% threshold
        rank = _make_rankings_df(
            ["A", "B", "C", "D"],
            tiers=["A", "A", "B", "C"],
            catalyst_modes=["specific_days", "specific_days", "specific_days", "no_upcoming"],
        )
        port = _make_portfolio_df(["A", "B"])
        snap = _make_snapshot("2026-02-09", rank, port)

        result = compute_single_snapshot_summary(snap)
        assert result.warnings == []


# ---------------------------------------------------------------------------
# Test 6: Single-snapshot mode
# ---------------------------------------------------------------------------

class TestSingleSnapshot:
    def test_produces_single_result(self):
        rank = _make_rankings_df(["A", "B", "C"], tiers=["A", "B", "C"])
        port = _make_portfolio_df(["A", "B"])
        snap = _make_snapshot("2026-02-09", rank, port)

        result = compute_single_snapshot_summary(snap)
        assert isinstance(result, SingleResult)
        assert result.tier_counts["A"] == 1
        assert result.tier_counts["B"] == 1
        assert result.tier_counts["C"] == 1

    def test_report_no_prior(self):
        rank = _make_rankings_df(["A", "B"], tiers=["A", "B"])
        port = _make_portfolio_df(["A", "B"])
        snap = _make_snapshot("2026-02-09", rank, port)
        result = compute_single_snapshot_summary(snap)

        report = generate_report(snap, None, result)
        assert "Prior:     (none)" in report
        assert "PHASE-2 RUN DELTA REPORT" in report
        assert "no prior for comparison" in report


# ---------------------------------------------------------------------------
# Test 7: Portfolio reconstruction
# ---------------------------------------------------------------------------

class TestPortfolioReconstruction:
    def test_reconstruct_filters_tier_ab(self):
        """Only A and B tiers should be in reconstructed portfolio."""
        rank = _make_rankings_df(
            ["A1", "B1", "C1", "D1"],
            tiers=["A", "B", "C", "D"],
            actionable_ranks=[1, 2, 3, 4],
        )
        port = _reconstruct_portfolio(rank)
        assert set(port["ticker"].tolist()) == {"A1", "B1"}

    def test_reconstruct_respects_top_k(self):
        """Reconstruction should take at most RECON_TOP_K names."""
        tickers = [f"T{i:03d}" for i in range(30)]
        rank = _make_rankings_df(
            tickers,
            tiers=["A"] * 15 + ["B"] * 15,
            actionable_ranks=list(range(1, 31)),
        )
        port = _reconstruct_portfolio(rank)
        assert len(port) == RECON_TOP_K

    def test_reconstruct_sorts_by_actionable_rank(self):
        """Reconstructed portfolio should be sorted by actionable_rank."""
        rank = _make_rankings_df(
            ["Z", "A", "M"],
            tiers=["A", "B", "A"],
            actionable_ranks=[3, 1, 2],
        )
        port = _reconstruct_portfolio(rank)
        assert port["ticker"].tolist() == ["A", "M", "Z"]

    def test_reconstruct_excludes_non_dev(self):
        """Non drug_developer archetypes should be excluded."""
        rank = _make_rankings_df(
            ["DEV", "COMM"],
            tiers=["A", "A"],
            archetypes=["drug_developer", "commercial_biotech"],
            actionable_ranks=[1, 2],
        )
        port = _reconstruct_portfolio(rank)
        assert port["ticker"].tolist() == ["DEV"]


# ---------------------------------------------------------------------------
# Test 8: Missing columns
# ---------------------------------------------------------------------------

class TestMissingColumns:
    def test_load_snapshot_no_tier_dev(self, tmp_path):
        """Snapshot without tier_dev column should return None."""
        snap_dir = tmp_path / "2026-01-28"
        snap_dir.mkdir()
        # Write a rankings.csv without tier_dev
        rankings = pd.DataFrame({
            "ticker": ["A", "B"],
            "composite_rank": [1, 2],
            "composite_score": [50, 40],
            "archetype": ["drug_developer", "drug_developer"],
        })
        rankings.to_csv(snap_dir / "rankings.csv", index=False)

        result = load_snapshot(snap_dir)
        assert result is None


# ---------------------------------------------------------------------------
# Test: Report and JSON generation
# ---------------------------------------------------------------------------

class TestOutputGeneration:
    def test_report_contains_sections(self):
        rank = _make_rankings_df(["A", "B", "C"], tiers=["A", "B", "C"])
        cur_port = _make_portfolio_df(["A", "B"], tiers=["A", "B"])
        pri_port = _make_portfolio_df(["B", "C"], tiers=["B", "C"])

        cur = _make_snapshot("2026-02-09", rank, cur_port)
        pri = _make_snapshot("2026-01-20", rank, pri_port, ruleset_id="d4f1f8a8")

        delta = compute_delta(cur, pri)
        report = generate_report(cur, pri, delta)

        assert "1. PORTFOLIO TURNOVER" in report
        assert "2. TIER DISTRIBUTION" in report
        assert "3. SIZING" in report
        assert "4. CATALYST COVERAGE" in report
        assert "5. RISK SANITY" in report
        assert "6. GUARDRAILS" in report

    def test_json_structure(self):
        rank = _make_rankings_df(["A", "B"])
        port = _make_portfolio_df(["A", "B"])
        cur = _make_snapshot("2026-02-09", rank, port)
        pri = _make_snapshot("2026-01-20", rank, port)

        delta = compute_delta(cur, pri)
        details = generate_details_json(cur, pri, delta)

        assert "generated_at" in details
        assert details["current"]["date"] == "2026-02-09"
        assert details["prior"]["date"] == "2026-01-20"
        assert "portfolio_turnover" in details
        assert "tier_distribution" in details
        assert "guardrails" in details
        assert "passed" in details["guardrails"]
        assert "warnings" in details["guardrails"]

    def test_json_single_mode(self):
        rank = _make_rankings_df(["A"], tiers=["A"])
        port = _make_portfolio_df(["A"])
        snap = _make_snapshot("2026-02-09", rank, port)
        result = compute_single_snapshot_summary(snap)
        details = generate_details_json(snap, None, result)

        assert details["prior"] is None
        assert "tier_distribution" in details

    def test_delta_csv(self, tmp_path):
        cur_port = _make_portfolio_df(["A", "B"], weights=[60.0, 40.0])
        pri_port = _make_portfolio_df(["B", "C"], weights=[50.0, 50.0])
        rank = _make_rankings_df(["A", "B", "C"])
        cur = _make_snapshot("2026-02-09", rank, cur_port)
        pri = _make_snapshot("2026-01-20", rank, pri_port)

        csv_path = tmp_path / "delta.csv"
        generate_delta_csv(cur, pri, csv_path)

        df = pd.read_csv(csv_path)
        assert len(df) == 3  # union of {A,B} and {B,C}
        assert set(df["ticker"].tolist()) == {"A", "B", "C"}

        row_a = df[df["ticker"] == "A"].iloc[0]
        assert row_a["in_current"] == 1
        assert row_a["in_prior"] == 0
        assert row_a["weight_current"] == 60.0
        assert row_a["weight_prior"] == 0.0

    def test_delta_csv_single_mode(self, tmp_path):
        """CSV in single-snapshot mode (no prior)."""
        port = _make_portfolio_df(["A", "B"], weights=[60.0, 40.0])
        rank = _make_rankings_df(["A", "B"])
        cur = _make_snapshot("2026-02-09", rank, port)

        csv_path = tmp_path / "delta.csv"
        generate_delta_csv(cur, None, csv_path)

        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert all(df["in_current"] == 1)
        assert all(df["in_prior"] == 0)


# ---------------------------------------------------------------------------
# Test: Size band counts
# ---------------------------------------------------------------------------

class TestSizeBands:
    def test_size_band_counting(self):
        port = _make_portfolio_df(
            ["A", "B", "C", "D"],
            size_bands=["L", "L", "M", "XS"],
        )
        counts = _size_band_counts(port)
        assert counts == {"L": 2, "M": 1, "S": 0, "XS": 1}
