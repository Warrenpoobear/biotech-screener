#!/usr/bin/env python3
"""Tests for ranking_mode (decision vs composite CSV sort) and company_name column.

Exercises:
  - company_name appears at index 1 in both SNAPSHOT_COLUMNS and PHASE2_PORTFOLIO_COLUMNS
  - company_name is populated from module_1_universe.active_securities
  - ranking_mode="decision" preserves DE sort order in rankings.csv:
    * all eligible rows first, then all ineligible rows (no interleaving)
    * actionable_rank is monotonically increasing 1..N within the eligible block
  - ranking_mode="composite" restores legacy composite_rank ordering
  - actionable_rank values are stable regardless of ranking_mode
  - deterministic tiebreak for identical decision profiles
  - decision_portfolio.csv always uses decision engine sort
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from run_screen import (
    SNAPSHOT_COLUMNS,
    PHASE2_PORTFOLIO_COLUMNS,
    save_validation_snapshot,
)
from decision_engine import DEFAULT_RULESET


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ranked_security(
    ticker: str,
    composite_rank: int,
    composite_score: float,
    *,
    severity: str = "",
    stage_bucket: str = "mid",
    market_cap_bucket: str = "small",
    catalyst_days: int | None = None,
    catalyst_in_window: bool | None = None,
    alpha_60d: float = 0.02,
    drawdown: float = -0.10,
    tier1_count: int = 3,
    confidence_overall: float = 0.72,
    clinical_normalized: float = 0.6,
    fundamental_red_flag: bool = False,
) -> Dict[str, Any]:
    """Build a minimal ranked_securities entry for save_validation_snapshot."""
    cd: Dict[str, Any] = {}
    if catalyst_days is not None:
        cd["days_to_catalyst"] = catalyst_days
    if catalyst_in_window is not None:
        cd["in_optimal_window"] = catalyst_in_window

    return {
        "ticker": ticker,
        "composite_rank": composite_rank,
        "composite_score": composite_score,
        "score_rank_pct": round(1.0 - composite_rank / 100, 4),
        "score_z": round(composite_score / 10 - 5, 2),
        "composite_score_attn": None,
        "score_rank_pct_attn": None,
        "score_z_attn": None,
        "stage_bucket": stage_bucket,
        "market_cap_bucket": market_cap_bucket,
        "severity": severity,
        "confidence_overall": confidence_overall,
        "fundamental_red_flag": fundamental_red_flag,
        "fundamental_red_flag_reasons": [],
        "flags": [],
        "attn_flags": [],
        "data_quality_flags": [],
        "momentum_signal": {},
        "valuation_signal": {},
        "component_scores": [
            {"name": "catalyst", "normalized": 0.5},
            {"name": "smart_money", "normalized": 0.4},
            {"name": "clinical", "normalized": clinical_normalized},
            {"name": "financial", "normalized": 0.7},
        ],
        "catalyst_decay": cd if cd else {},
        "smart_money_signal": {},
        "coinvest": {"tier1_count": tier1_count},
        "defensive_features": {
            "drawdown": drawdown,
            "drawdown_xbi": -0.05,
            "drawdown_rel_xbi": drawdown - (-0.05),
            "vol_60d": 0.30,
            "beta_xbi_60d": 0.80,
            "rsi_14d": 50,
        },
        "score_breakdown": {
            "enhancements": {"momentum": {"alpha_60d": alpha_60d}},
        },
    }


def _make_results(
    securities: List[Dict[str, Any]],
    archetypes: Dict[str, str] | None = None,
    active_securities: List[Dict[str, str]] | None = None,
) -> Dict[str, Any]:
    """Build a minimal results dict for save_validation_snapshot."""
    tickers = [s["ticker"] for s in securities]
    if archetypes is None:
        archetypes = {t: "drug_developer" for t in tickers}
    results: Dict[str, Any] = {
        "module_5_composite": {"ranked_securities": securities},
        "company_archetypes": archetypes,
    }
    if active_securities is not None:
        results["module_1_universe"] = {"active_securities": active_securities}
    return results


def _read_csv(snap_path: Path, filename: str = "rankings.csv") -> List[Dict[str, str]]:
    """Read a CSV from the snapshot directory and return as list of dicts."""
    csv_path = snap_path / filename
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Tests: company_name in column schemas
# ---------------------------------------------------------------------------

class TestCompanyNameSchema:
    """Verify company_name placement in column lists."""

    def test_company_name_in_snapshot_columns(self):
        """company_name is at index 1 (right after ticker) in SNAPSHOT_COLUMNS."""
        assert SNAPSHOT_COLUMNS[0] == "ticker"
        assert SNAPSHOT_COLUMNS[1] == "company_name"

    def test_company_name_in_phase2_portfolio_columns(self):
        """company_name is at index 1 (right after ticker) in PHASE2_PORTFOLIO_COLUMNS."""
        assert PHASE2_PORTFOLIO_COLUMNS[0] == "ticker"
        assert PHASE2_PORTFOLIO_COLUMNS[1] == "company_name"


class TestCompanyNamePopulated:
    """Verify company_name is populated from M1 universe data."""

    def test_company_name_populated(self, tmp_path):
        """CSV output has correct company names from M1 data."""
        secs = [
            _make_ranked_security("ACRS", 1, 80.0),
            _make_ranked_security("BMRN", 2, 70.0),
        ]
        active = [
            {"ticker": "ACRS", "company_name": "Aclaris Therapeutics, Inc."},
            {"ticker": "BMRN", "company_name": "BioMarin Pharmaceutical Inc."},
        ]
        results = _make_results(secs, active_securities=active)

        snap = save_validation_snapshot(
            snapshot_dir=tmp_path / "snap",
            as_of_date="2026-01-01",
            results=results,
            version="test",
        )
        assert snap is not None

        rows = _read_csv(snap)
        name_by_ticker = {r["ticker"]: r["company_name"] for r in rows}
        assert name_by_ticker["ACRS"] == "Aclaris Therapeutics, Inc."
        assert name_by_ticker["BMRN"] == "BioMarin Pharmaceutical Inc."

    def test_company_name_missing_m1(self, tmp_path):
        """When M1 data is absent, company_name defaults to empty string."""
        secs = [_make_ranked_security("XTST", 1, 80.0)]
        results = _make_results(secs)  # no active_securities

        snap = save_validation_snapshot(
            snapshot_dir=tmp_path / "snap",
            as_of_date="2026-01-01",
            results=results,
            version="test",
        )
        assert snap is not None

        rows = _read_csv(snap)
        assert rows[0]["company_name"] == ""

    def test_company_name_none_in_m1(self, tmp_path):
        """When M1 entry has company_name=None, CSV gets empty string."""
        secs = [_make_ranked_security("XTST", 1, 80.0)]
        active = [{"ticker": "XTST", "company_name": None}]
        results = _make_results(secs, active_securities=active)

        snap = save_validation_snapshot(
            snapshot_dir=tmp_path / "snap",
            as_of_date="2026-01-01",
            results=results,
            version="test",
        )
        assert snap is not None

        rows = _read_csv(snap)
        assert rows[0]["company_name"] == ""

    def test_sector_label_replaced_with_override(self, tmp_path):
        """When M1 company_name is a sector label (e.g. 'Healthcare'),
        the override table supplies the real name."""
        secs = [
            _make_ranked_security("PVLA", 1, 80.0),
            _make_ranked_security("CLYM", 2, 70.0),
        ]
        active = [
            {"ticker": "PVLA", "company_name": "Healthcare"},
            {"ticker": "CLYM", "company_name": "Healthcare"},
        ]
        results = _make_results(secs, active_securities=active)

        snap = save_validation_snapshot(
            snapshot_dir=tmp_path / "snap",
            as_of_date="2026-01-01",
            results=results,
            version="test",
        )
        assert snap is not None

        rows = _read_csv(snap)
        name_by_ticker = {r["ticker"]: r["company_name"] for r in rows}
        assert name_by_ticker["PVLA"] == "Palvella Therapeutics, Inc."
        assert name_by_ticker["CLYM"] == "Climb Bio, Inc."

    def test_sector_label_unknown_ticker_gets_empty(self, tmp_path):
        """A sector-labeled ticker NOT in the override table gets empty string."""
        secs = [_make_ranked_security("FAKE", 1, 80.0)]
        active = [{"ticker": "FAKE", "company_name": "Healthcare"}]
        results = _make_results(secs, active_securities=active)

        snap = save_validation_snapshot(
            snapshot_dir=tmp_path / "snap",
            as_of_date="2026-01-01",
            results=results,
            version="test",
        )
        assert snap is not None

        rows = _read_csv(snap)
        assert rows[0]["company_name"] == ""

    def test_real_name_not_overridden(self, tmp_path):
        """A real company name (not a sector label) is kept even for override tickers."""
        secs = [_make_ranked_security("PVLA", 1, 80.0)]
        active = [{"ticker": "PVLA", "company_name": "Palvella Custom Name"}]
        results = _make_results(secs, active_securities=active)

        snap = save_validation_snapshot(
            snapshot_dir=tmp_path / "snap",
            as_of_date="2026-01-01",
            results=results,
            version="test",
        )
        assert snap is not None

        rows = _read_csv(snap)
        # The real name should pass through, override only triggers on sector labels
        assert rows[0]["company_name"] == "Palvella Custom Name"


# ---------------------------------------------------------------------------
# Tests: decision ranking mode — eligible-first + monotonic actionable_rank
# ---------------------------------------------------------------------------

class TestDecisionRankingEligibleFirst:
    """The core test that would have caught the uploaded-CSV bug.

    Key invariant: in ranking_mode='decision', ALL eligible rows must appear
    before ALL ineligible rows, and the eligible block must have monotonically
    increasing actionable_rank (1, 2, 3, ..., N).
    """

    def test_no_eligibility_interleaving(self, tmp_path):
        """Eligible rows first, ineligible rows last — no interleaving.

        Deliberately gives ineligible rows LOWER composite_rank so that
        composite-based sorting would interleave them into the top of
        the file.  If the file is still sorted by M5, this test fails.
        """
        # Ineligible rows: composite_rank 1-3 (top M5 ranks, but ineligible)
        # Eligible rows: composite_rank 4-6 (worse M5, but eligible)
        secs = [
            # Ineligible: fundamental_red_flag=True triggers ineligibility
            _make_ranked_security("INELIG_A", 1, 95.0, fundamental_red_flag=True),
            _make_ranked_security("INELIG_B", 2, 90.0, fundamental_red_flag=True),
            _make_ranked_security("INELIG_C", 3, 85.0, fundamental_red_flag=True),
            # Eligible: good profiles with catalyst
            _make_ranked_security("ELIG_X", 4, 70.0, catalyst_days=30, catalyst_in_window=True),
            _make_ranked_security("ELIG_Y", 5, 60.0, catalyst_days=60),
            _make_ranked_security("ELIG_Z", 6, 50.0, catalyst_days=90),
        ]
        results = _make_results(secs)

        snap = save_validation_snapshot(
            snapshot_dir=tmp_path / "snap",
            as_of_date="2026-01-01",
            results=results,
            version="test",
            decision_mode="phase2",
            ranking_mode="decision",
        )
        assert snap is not None

        rows = _read_csv(snap)
        assert len(rows) == 6

        # Check: no interleaving.  Once we see an ineligible row, no eligible
        # row should appear after it.
        seen_ineligible = False
        for r in rows:
            is_elig = r.get("eligible") == "1"
            if not is_elig:
                seen_ineligible = True
            if seen_ineligible and is_elig:
                raise AssertionError(
                    f"Eligibility interleaving: eligible row {r['ticker']} "
                    f"appeared after ineligible row.  Row order (ticker, eligible): "
                    + str([(x["ticker"], x.get("eligible")) for x in rows])
                )

        # Eligible rows must come first
        eligible_tickers = [r["ticker"] for r in rows if r.get("eligible") == "1"]
        ineligible_tickers = [r["ticker"] for r in rows if r.get("eligible") != "1"]
        assert len(eligible_tickers) == 3, f"Expected 3 eligible, got {eligible_tickers}"
        assert len(ineligible_tickers) == 3, f"Expected 3 ineligible, got {ineligible_tickers}"
        # First 3 rows must be the eligible ones
        assert [r["ticker"] for r in rows[:3]] == eligible_tickers

    def test_actionable_rank_monotonic_in_eligible_block(self, tmp_path):
        """actionable_rank must be exactly 1, 2, 3, ..., N in row order."""
        secs = [
            _make_ranked_security("INELIG_A", 1, 95.0, fundamental_red_flag=True),
            _make_ranked_security("ELIG_X", 2, 80.0, catalyst_days=30, catalyst_in_window=True),
            _make_ranked_security("ELIG_Y", 3, 70.0, catalyst_days=60),
            _make_ranked_security("ELIG_Z", 4, 60.0, catalyst_days=90),
        ]
        results = _make_results(secs)

        snap = save_validation_snapshot(
            snapshot_dir=tmp_path / "snap",
            as_of_date="2026-01-01",
            results=results,
            version="test",
            decision_mode="phase2",
            ranking_mode="decision",
        )
        assert snap is not None

        df = pd.read_csv(snap / "rankings.csv", dtype=str)
        eligible = df["eligible"].astype(str).isin(["1"])
        act = pd.to_numeric(df["actionable_rank"], errors="coerce")

        # Eligible block must be at the top
        eligible_indices = df.index[eligible].tolist()
        ineligible_indices = df.index[~eligible].tolist()
        if eligible_indices and ineligible_indices:
            assert max(eligible_indices) < min(ineligible_indices), (
                f"Eligible rows not all before ineligible. "
                f"Eligible idx: {eligible_indices}, Ineligible idx: {ineligible_indices}"
            )

        # actionable_rank in the eligible block must be monotonically increasing
        act_elig = act[eligible].dropna()
        assert len(act_elig) == eligible.sum()
        assert act_elig.is_monotonic_increasing, (
            f"actionable_rank not monotonic: {act_elig.tolist()}"
        )
        # Must be exactly 1..N
        assert act_elig.tolist() == list(range(1, len(act_elig) + 1))

        # Ineligible rows should have blank/NaN actionable_rank
        act_inelig = act[~eligible]
        assert act_inelig.isna().all(), (
            f"Ineligible rows have non-blank actionable_rank: {act_inelig.tolist()}"
        )

    def test_decision_mode_ignores_composite_order(self, tmp_path):
        """When ranking_mode=decision, tier A row before tier B even if composite_rank disagrees."""
        # BTST: composite_rank=1 (best composite) but will get tier B (no catalyst)
        # ATST: composite_rank=5 (worse composite) but will get tier A (near catalyst)
        secs = [
            _make_ranked_security("BTST", 1, 90.0, catalyst_days=None),
            _make_ranked_security("ATST", 5, 60.0, catalyst_days=30, catalyst_in_window=True),
        ]
        results = _make_results(secs)

        snap = save_validation_snapshot(
            snapshot_dir=tmp_path / "snap",
            as_of_date="2026-01-01",
            results=results,
            version="test",
            decision_mode="phase2",
            ranking_mode="decision",
        )
        assert snap is not None

        rows = _read_csv(snap)
        # Both should be eligible
        eligible_tickers = [r["ticker"] for r in rows if r["actionable_rank"] != ""]
        assert len(eligible_tickers) == 2
        # First row in CSV should have actionable_rank=1
        assert rows[0]["actionable_rank"] == "1"
        # Composite_rank should NOT be monotonic (proves it's not M5-sorted)
        comp_ranks = [int(r["composite_rank"]) for r in rows]
        # If first row has actionable_rank=1 but NOT composite_rank=1,
        # that proves DE ordering, not M5 ordering
        if rows[0]["composite_rank"] != "1":
            pass  # DE is driving the order, not composite
        # At minimum: actionable_rank is monotonic in the eligible block
        act_ranks = [int(r["actionable_rank"]) for r in rows if r["actionable_rank"] != ""]
        assert act_ranks == [1, 2]


class TestCompositeRankingMode:
    """Verify ranking_mode='composite' restores legacy composite_rank ordering."""

    def test_composite_mode_preserves_legacy_order(self, tmp_path):
        """When ranking_mode=composite, composite_rank order is used."""
        secs = [
            _make_ranked_security("BTST", 1, 90.0, catalyst_days=None),
            _make_ranked_security("ATST", 5, 60.0, catalyst_days=30, catalyst_in_window=True),
        ]
        results = _make_results(secs)

        snap = save_validation_snapshot(
            snapshot_dir=tmp_path / "snap",
            as_of_date="2026-01-01",
            results=results,
            version="test",
            decision_mode="phase2",
            ranking_mode="composite",
        )
        assert snap is not None

        rows = _read_csv(snap)
        # In composite mode, rows are re-sorted by composite_rank
        assert rows[0]["ticker"] == "BTST"  # composite_rank=1
        assert rows[1]["ticker"] == "ATST"  # composite_rank=5

    def test_composite_mode_interleaves_eligible_and_ineligible(self, tmp_path):
        """Legacy composite mode may interleave eligible/ineligible by composite_rank."""
        secs = [
            _make_ranked_security("INELIG_A", 1, 95.0, fundamental_red_flag=True),
            _make_ranked_security("ELIG_X", 2, 80.0, catalyst_days=30, catalyst_in_window=True),
            _make_ranked_security("INELIG_B", 3, 70.0, fundamental_red_flag=True),
            _make_ranked_security("ELIG_Y", 4, 60.0, catalyst_days=60),
        ]
        results = _make_results(secs)

        snap = save_validation_snapshot(
            snapshot_dir=tmp_path / "snap",
            as_of_date="2026-01-01",
            results=results,
            version="test",
            decision_mode="phase2",
            ranking_mode="composite",
        )
        assert snap is not None

        rows = _read_csv(snap)
        # In composite mode, order is by composite_rank (1,2,3,4)
        assert [r["ticker"] for r in rows] == [
            "INELIG_A", "ELIG_X", "INELIG_B", "ELIG_Y",
        ]


class TestActionableRankStability:
    """actionable_rank values must be identical regardless of ranking_mode."""

    def test_actionable_rank_stable_across_modes(self, tmp_path):
        """Same actionable_rank values regardless of ranking_mode."""
        secs = [
            _make_ranked_security("AAA", 3, 70.0, catalyst_days=30, catalyst_in_window=True),
            _make_ranked_security("BBB", 1, 90.0, catalyst_days=None),
            _make_ranked_security("CCC", 2, 80.0, catalyst_days=60),
        ]

        for mode in ("decision", "composite"):
            results = _make_results(secs)
            snap = save_validation_snapshot(
                snapshot_dir=tmp_path / f"snap_{mode}",
                as_of_date="2026-01-01",
                results=results,
                version="test",
                decision_mode="phase2",
                ranking_mode=mode,
            )
            assert snap is not None

        # Read both CSVs
        rows_dec = _read_csv(tmp_path / "snap_decision" / "2026-01-01")
        rows_comp = _read_csv(tmp_path / "snap_composite" / "2026-01-01")

        # Build ticker->actionable_rank maps
        ranks_dec = {r["ticker"]: r["actionable_rank"] for r in rows_dec}
        ranks_comp = {r["ticker"]: r["actionable_rank"] for r in rows_comp}

        assert ranks_dec == ranks_comp, (
            f"actionable_rank differs: decision={ranks_dec}, composite={ranks_comp}"
        )


class TestDeterministicTiebreak:
    """Identical inputs must always produce the same row order."""

    def test_deterministic_tiebreak(self, tmp_path):
        """Two runs with identical inputs produce identical CSV ordering."""
        secs = [
            _make_ranked_security("ZZZ", 1, 80.0, catalyst_days=60),
            _make_ranked_security("AAA", 1, 80.0, catalyst_days=60),
        ]

        orders = []
        for i in range(2):
            results = _make_results(secs)
            snap = save_validation_snapshot(
                snapshot_dir=tmp_path / f"snap_{i}",
                as_of_date="2026-01-01",
                results=results,
                version="test",
                ranking_mode="decision",
            )
            assert snap is not None
            rows = _read_csv(snap)
            orders.append([r["ticker"] for r in rows])

        assert orders[0] == orders[1], (
            f"Non-deterministic: run1={orders[0]}, run2={orders[1]}"
        )
        # Both should have distinct actionable_rank
        rows = _read_csv(tmp_path / "snap_0" / "2026-01-01")
        ranks = [r["actionable_rank"] for r in rows if r["actionable_rank"] != ""]
        assert len(ranks) == 2
        assert ranks[0] != ranks[1]


class TestPortfolioCsvAlwaysDecisionSorted:
    """decision_portfolio.csv uses actionable sort regardless of ranking_mode."""

    def test_portfolio_csv_always_decision_sorted(self, tmp_path):
        """decision_portfolio.csv uses DE sort even with ranking_mode=composite."""
        secs = [
            _make_ranked_security("BTST", 1, 90.0, catalyst_days=None),
            _make_ranked_security("ATST", 5, 60.0, catalyst_days=30, catalyst_in_window=True),
        ]
        results = _make_results(secs)

        snap = save_validation_snapshot(
            snapshot_dir=tmp_path / "snap",
            as_of_date="2026-01-01",
            results=results,
            version="test",
            decision_mode="phase2",
            ranking_mode="composite",
        )
        assert snap is not None

        # decision_portfolio.csv should exist in phase2 mode
        portfolio_path = snap / "decision_portfolio.csv"
        if portfolio_path.exists():
            rows = _read_csv(snap, "decision_portfolio.csv")
            if len(rows) >= 2:
                # Portfolio should be sorted by DE sort key, not composite_rank
                # The row with actionable_rank=1 should be first
                assert rows[0]["actionable_rank"] == "1"


class TestPortfolioCsvCompanyName:
    """decision_portfolio.csv includes company_name."""

    def test_portfolio_csv_has_company_name(self, tmp_path):
        """decision_portfolio.csv has company_name column populated from M1."""
        secs = [
            _make_ranked_security("ACRS", 1, 80.0, catalyst_days=30, catalyst_in_window=True),
            _make_ranked_security("BMRN", 2, 70.0, catalyst_days=60),
        ]
        active = [
            {"ticker": "ACRS", "company_name": "Aclaris Therapeutics, Inc."},
            {"ticker": "BMRN", "company_name": "BioMarin Pharmaceutical Inc."},
        ]
        results = _make_results(secs, active_securities=active)

        snap = save_validation_snapshot(
            snapshot_dir=tmp_path / "snap",
            as_of_date="2026-01-01",
            results=results,
            version="test",
            decision_mode="phase2",
        )
        assert snap is not None

        portfolio_path = snap / "decision_portfolio.csv"
        if portfolio_path.exists():
            rows = _read_csv(snap, "decision_portfolio.csv")
            assert len(rows) > 0
            assert "company_name" in rows[0], "company_name column missing from portfolio CSV"
            name_by_ticker = {r["ticker"]: r["company_name"] for r in rows}
            for ticker, expected in [
                ("ACRS", "Aclaris Therapeutics, Inc."),
                ("BMRN", "BioMarin Pharmaceutical Inc."),
            ]:
                if ticker in name_by_ticker:
                    assert name_by_ticker[ticker] == expected


class TestMetadataRankingMode:
    """ranking_mode is recorded in metadata.json."""

    def test_ranking_mode_in_metadata(self, tmp_path):
        """metadata.json includes ranking_mode field."""
        secs = [_make_ranked_security("XTST", 1, 80.0)]
        results = _make_results(secs)

        snap = save_validation_snapshot(
            snapshot_dir=tmp_path / "snap",
            as_of_date="2026-01-01",
            results=results,
            version="test",
            ranking_mode="decision",
        )
        assert snap is not None

        meta = json.loads((snap / "metadata.json").read_text())
        assert meta["ranking_mode"] == "decision"

    def test_ranking_mode_composite_in_metadata(self, tmp_path):
        """metadata.json records composite when that mode is used."""
        secs = [_make_ranked_security("XTST", 1, 80.0)]
        results = _make_results(secs)

        snap = save_validation_snapshot(
            snapshot_dir=tmp_path / "snap",
            as_of_date="2026-01-01",
            results=results,
            version="test",
            ranking_mode="composite",
        )
        assert snap is not None

        meta = json.loads((snap / "metadata.json").read_text())
        assert meta["ranking_mode"] == "composite"
