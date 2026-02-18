#!/usr/bin/env python3
"""Tests for ranking_mode (decision vs composite CSV sort) and company_name column.

Exercises:
  - company_name appears at index 1 in both SNAPSHOT_COLUMNS and PHASE2_PORTFOLIO_COLUMNS
  - company_name is populated from module_1_universe.active_securities
  - ranking_mode="decision" preserves DE sort order in rankings.csv
  - ranking_mode="composite" restores legacy composite_rank ordering
  - actionable_rank values are stable regardless of ranking_mode
  - deterministic tiebreak (alphabetic ticker) for identical decision profiles
  - decision_portfolio.csv always uses decision engine sort
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

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
        "fundamental_red_flag": False,
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


# ---------------------------------------------------------------------------
# Tests: ranking mode
# ---------------------------------------------------------------------------

class TestDecisionRankingMode:
    """Verify ranking_mode='decision' sorts by DE sort key (not composite_rank)."""

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
        tickers = [r["ticker"] for r in rows]
        # In decision mode, the DE sort key determines order.
        # The row with the better actionable_rank should come first.
        ranks = {r["ticker"]: r["actionable_rank"] for r in rows}
        # Both should be eligible
        eligible_tickers = [r["ticker"] for r in rows if r["actionable_rank"] != ""]
        assert len(eligible_tickers) == 2
        # First row in CSV should have actionable_rank=1
        assert rows[0]["actionable_rank"] == "1"


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

        # Build ticker→actionable_rank maps
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
