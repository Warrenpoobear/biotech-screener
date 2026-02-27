"""Tests for the financials_missing eligibility gate in the decision engine.

When survivability coverage flags include BOTH 'missing_cash' AND
'missing_burn_data', the ticker should be flagged 'financials_missing'
instead of 'fundamental_red_flag' (which would be a false positive).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from decision_engine import DecisionRuleset, compute_decision_fields


def _rec(
    ticker: str = "TEST",
    fundamental_red_flag: bool = False,
    survivability_coverage: list | None = None,
) -> Dict[str, Any]:
    """Build a minimal synthetic rec for financials_missing gate testing."""
    rec: Dict[str, Any] = {
        "ticker": ticker,
        "severity": "NONE",
        "confidence_overall": 0.72,
        "fundamental_red_flag": fundamental_red_flag,
        "fundamental_red_flag_reasons": [],
        "flags": [],
        "attn_flags": [],
        "catalyst_decay": None,
        "smart_money_signal": {},
        "coinvest": {},
        "defensive_features": {"drawdown": -0.10, "vol_60d": 0.50},
        "score_breakdown": {},
        "momentum_signal": {},
    }
    if survivability_coverage is not None:
        rec["survivability_signal"] = {"coverage": survivability_coverage}
    return rec


class TestFinancialsMissingGate:
    """Gate 0: financials_missing fires when survivability data is absent."""

    def test_financials_missing_gate_fires(self):
        """Both missing_cash + missing_burn_data → financials_missing."""
        rec = _rec(
            ticker="AZN",
            survivability_coverage=["missing_cash", "missing_burn_data",
                                    "missing_near_term_debt"],
        )
        result = compute_decision_fields(
            rec, archetype="commercial_pharma", optionality_pct_dev=0.50,
        )
        assert result["eligible"] == "0"
        assert "financials_missing" in result["ineligible_reasons"]
        assert "fundamental_red_flag" not in result["ineligible_reasons"]

    def test_financials_missing_skips_red_flag(self):
        """Even if fundamental_red_flag=True, financials_missing takes priority."""
        rec = _rec(
            ticker="ARGX",
            fundamental_red_flag=True,
            survivability_coverage=["missing_cash", "missing_burn_data"],
        )
        result = compute_decision_fields(
            rec, archetype="drug_developer", optionality_pct_dev=0.70,
        )
        assert result["eligible"] == "0"
        assert "financials_missing" in result["ineligible_reasons"]
        assert "fundamental_red_flag" not in result["ineligible_reasons"]

    def test_partial_missing_still_checks_red_flag(self):
        """Only missing_cash (not missing_burn_data) → red flag check runs."""
        rec = _rec(
            ticker="BIIB",
            fundamental_red_flag=True,
            survivability_coverage=["missing_cash"],
        )
        result = compute_decision_fields(
            rec, archetype="commercial_biotech", optionality_pct_dev=0.60,
        )
        assert result["eligible"] == "0"
        assert "fundamental_red_flag" in result["ineligible_reasons"]
        assert "financials_missing" not in result["ineligible_reasons"]

    def test_full_financial_data_unchanged(self):
        """No missing coverage → red flag check runs as before."""
        rec = _rec(
            ticker="SNDX",
            fundamental_red_flag=True,
            survivability_coverage=["burn_approximated"],
        )
        result = compute_decision_fields(
            rec, archetype="drug_developer", optionality_pct_dev=0.65,
        )
        assert result["eligible"] == "0"
        assert "fundamental_red_flag" in result["ineligible_reasons"]
        assert "financials_missing" not in result["ineligible_reasons"]

    def test_no_survivability_signal_runs_red_flag(self):
        """Missing survivability_signal key entirely → red flag check runs."""
        rec = _rec(
            ticker="BBIO",
            fundamental_red_flag=True,
        )
        # No survivability_signal key at all
        result = compute_decision_fields(
            rec, archetype="drug_developer", optionality_pct_dev=0.65,
        )
        assert "fundamental_red_flag" in result["ineligible_reasons"]
        assert "financials_missing" not in result["ineligible_reasons"]
