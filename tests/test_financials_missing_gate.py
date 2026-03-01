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
    surv_metrics: dict | None = None,
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
        surv_signal: Dict[str, Any] = {"coverage": survivability_coverage}
        if surv_metrics is not None:
            surv_signal["metrics"] = surv_metrics
        else:
            surv_signal["metrics"] = {"cash_total": 0.0, "burn_ttm": 0.0}
        rec["survivability_signal"] = surv_signal
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

    def test_cash_present_via_sti_not_financials_missing(self):
        """Coverage flags say missing but cash_total > 0 → NOT financials_missing.

        Regression test for GILD-like commercial pharma where cash arrives via
        MarketableSecurities (not cash_and_equivalents) and positive OCF means
        zero burn (not missing burn data).
        """
        rec = _rec(
            ticker="GILD",
            fundamental_red_flag=False,
            survivability_coverage=["missing_cash", "missing_burn_data",
                                    "missing_interest_expense"],
            surv_metrics={"cash_total": 68_000_000.0, "burn_ttm": 0.0},
        )
        result = compute_decision_fields(
            rec, archetype="commercial_pharma", optionality_pct_dev=0.50,
        )
        assert result["eligible"] == "1"
        assert "financials_missing" not in result["ineligible_reasons"]

    def test_cash_present_large_amount_eligible(self):
        """ARWR-like: $715M cash, coverage flags present, should be eligible."""
        rec = _rec(
            ticker="ARWR",
            fundamental_red_flag=False,
            survivability_coverage=["missing_cash", "missing_burn_data"],
            surv_metrics={"cash_total": 714_967_000.0, "burn_ttm": 0.0},
        )
        result = compute_decision_fields(
            rec, archetype="commercial_biotech", optionality_pct_dev=0.60,
        )
        assert result["eligible"] == "1"
        assert "financials_missing" not in result["ineligible_reasons"]

    def test_zero_cash_still_financials_missing(self):
        """True data absence: cash_total=0 with both flags → financials_missing."""
        rec = _rec(
            ticker="NODATA",
            fundamental_red_flag=False,
            survivability_coverage=["missing_cash", "missing_burn_data"],
            surv_metrics={"cash_total": 0.0, "burn_ttm": 0.0},
        )
        result = compute_decision_fields(
            rec, archetype="drug_developer", optionality_pct_dev=0.50,
        )
        assert result["eligible"] == "0"
        assert "financials_missing" in result["ineligible_reasons"]


class TestFinancialsMissingBypass:
    """Mega-cap bypass for the financials_missing gate."""

    def test_bypass_mega_cap(self):
        """cash_total=0, both flags, market_cap=10B, threshold=5B → eligible + financials_partial."""
        rec = _rec(
            ticker="AZN",
            fundamental_red_flag=False,
            survivability_coverage=["missing_cash", "missing_burn_data"],
            surv_metrics={"cash_total": 0.0, "burn_ttm": 0.0},
        )
        rs = DecisionRuleset(financials_missing_bypass_market_cap=5_000_000_000.0)
        result = compute_decision_fields(
            rec, archetype="commercial_pharma", optionality_pct_dev=0.50,
            ruleset=rs, market_cap=10_000_000_000.0,
        )
        assert result["eligible"] == "1"
        assert "financials_missing" not in result["ineligible_reasons"]
        assert "financials_partial" in result["risk_flags"]

    def test_bypass_disabled_by_default(self):
        """Default ruleset (threshold=0) → financials_missing still fires."""
        rec = _rec(
            ticker="BIIB",
            fundamental_red_flag=False,
            survivability_coverage=["missing_cash", "missing_burn_data"],
            surv_metrics={"cash_total": 0.0, "burn_ttm": 0.0},
        )
        result = compute_decision_fields(
            rec, archetype="commercial_biotech", optionality_pct_dev=0.50,
            market_cap=50_000_000_000.0,
        )
        assert result["eligible"] == "0"
        assert "financials_missing" in result["ineligible_reasons"]

    def test_bypass_below_threshold(self):
        """market_cap=3B, threshold=5B → financials_missing still fires."""
        rec = _rec(
            ticker="SMALL",
            fundamental_red_flag=False,
            survivability_coverage=["missing_cash", "missing_burn_data"],
            surv_metrics={"cash_total": 0.0, "burn_ttm": 0.0},
        )
        rs = DecisionRuleset(financials_missing_bypass_market_cap=5_000_000_000.0)
        result = compute_decision_fields(
            rec, archetype="drug_developer", optionality_pct_dev=0.50,
            ruleset=rs, market_cap=3_000_000_000.0,
        )
        assert result["eligible"] == "0"
        assert "financials_missing" in result["ineligible_reasons"]

    def test_bypass_no_market_cap(self):
        """market_cap=None, threshold=5B → financials_missing still fires."""
        rec = _rec(
            ticker="NOMC",
            fundamental_red_flag=False,
            survivability_coverage=["missing_cash", "missing_burn_data"],
            surv_metrics={"cash_total": 0.0, "burn_ttm": 0.0},
        )
        rs = DecisionRuleset(financials_missing_bypass_market_cap=5_000_000_000.0)
        result = compute_decision_fields(
            rec, archetype="drug_developer", optionality_pct_dev=0.50,
            ruleset=rs, market_cap=None,
        )
        assert result["eligible"] == "0"
        assert "financials_missing" in result["ineligible_reasons"]
