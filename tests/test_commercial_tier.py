"""
Tests for commercial tier promotion (tier_first mode).

Covers:
  - _compute_tier_commercial logic (quality × catalyst → tier)
  - tier_any derivation
  - Sort key behavior in dev_first vs tier_first mode
  - Portfolio filter behavior in both modes
  - Sizing with effective tier in tier_first mode
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from decision_engine import (
    DecisionRuleset,
    _compute_tier_commercial,
    compute_actionable_sort_key,
    compute_decision_fields,
    compute_target_weights,
)


# =============================================================================
# Helper: build synthetic rec (mirrors test_decision_engine_contract._rec)
# =============================================================================

def _rec(
    ticker: str = "TEST",
    severity: str = "NONE",
    fundamental_red_flag: bool = False,
    catalyst_days: int | None = None,
    catalyst_in_window: bool | None = None,
    drawdown: float | None = -0.15,
    alpha_60d: float | None = 0.02,
    tier1_count: int | None = 3,
) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "ticker": ticker,
        "severity": severity,
        "fundamental_red_flag": fundamental_red_flag,
        "fundamental_red_flag_reasons": [],
        "flags": [],
        "attn_flags": [],
        "confidence_overall": 0.72,
    }
    cd: Dict[str, Any] = {}
    if catalyst_days is not None:
        cd["days_to_catalyst"] = catalyst_days
    if catalyst_in_window is not None:
        cd["in_optimal_window"] = catalyst_in_window
    rec["catalyst_decay"] = cd if cd else None
    rec["smart_money_signal"] = {"overlap_count": 2}
    rec["coinvest"] = {"tier1_count": tier1_count} if tier1_count is not None else {}
    rec["defensive_features"] = {
        "vol_60d": 0.65,
        "beta_xbi_60d": 1.1,
        "drawdown": drawdown,
        "rsi_14d": 48.0,
    }
    if alpha_60d is not None:
        rec["score_breakdown"] = {"enhancements": {"momentum": {"alpha_60d": alpha_60d}}}
    else:
        rec["score_breakdown"] = {}
    rec["momentum_signal"] = {}
    return rec


# Default ruleset for reference
DEV_FIRST_RS = DecisionRuleset()  # tiering_priority_mode="dev_first"
TIER_FIRST_RS = DecisionRuleset(tiering_priority_mode="tier_first")


# =============================================================================
# 1–7: _compute_tier_commercial unit tests
# =============================================================================

class TestComputeTierCommercial:
    """Unit tests for _compute_tier_commercial."""

    def test_high_quality_near_catalyst(self):
        """quality_pct=0.90 + near catalyst → A."""
        tier, reason = _compute_tier_commercial(
            archetype="commercial_biotech",
            eligible=True,
            quality_pct=0.90,
            catalyst_in_window="1",
            catalyst_days=45,
            ruleset=DEV_FIRST_RS,
        )
        assert tier == "A"
        assert "high_quality" in reason
        assert "catalyst" in reason

    def test_high_quality_far_catalyst(self):
        """quality_pct=0.90 + far catalyst → B."""
        tier, reason = _compute_tier_commercial(
            archetype="commercial_biotech",
            eligible=True,
            quality_pct=0.90,
            catalyst_in_window="0",
            catalyst_days=250,
            ruleset=DEV_FIRST_RS,
        )
        assert tier == "B"
        assert "high_quality" in reason

    def test_mod_quality_near_catalyst(self):
        """quality_pct=0.70 + near catalyst → B."""
        tier, reason = _compute_tier_commercial(
            archetype="commercial_biotech",
            eligible=True,
            quality_pct=0.70,
            catalyst_in_window="1",
            catalyst_days=45,
            ruleset=DEV_FIRST_RS,
        )
        assert tier == "B"
        assert "mod_quality" in reason

    def test_low_quality(self):
        """quality_pct=0.40 → C."""
        tier, reason = _compute_tier_commercial(
            archetype="commercial_pharma",
            eligible=True,
            quality_pct=0.40,
            catalyst_in_window="1",
            catalyst_days=45,
            ruleset=DEV_FIRST_RS,
        )
        assert tier == "C"
        assert "low_quality" in reason

    def test_no_quality_data(self):
        """quality_pct=None → C, no_quality_data."""
        tier, reason = _compute_tier_commercial(
            archetype="commercial_biotech",
            eligible=True,
            quality_pct=None,
            catalyst_in_window="",
            catalyst_days="",
            ruleset=DEV_FIRST_RS,
        )
        assert tier == "C"
        assert reason == "no_quality_data"

    def test_ineligible(self):
        """eligible=False → D, ineligible."""
        tier, reason = _compute_tier_commercial(
            archetype="commercial_biotech",
            eligible=False,
            quality_pct=0.95,
            catalyst_in_window="1",
            catalyst_days=30,
            ruleset=DEV_FIRST_RS,
        )
        assert tier == "D"
        assert reason == "ineligible"

    def test_skips_dev(self):
        """drug_developer → ("", "")."""
        tier, reason = _compute_tier_commercial(
            archetype="drug_developer",
            eligible=True,
            quality_pct=0.95,
            catalyst_in_window="1",
            catalyst_days=30,
            ruleset=DEV_FIRST_RS,
        )
        assert tier == ""
        assert reason == ""


# =============================================================================
# 8–9: tier_any derivation via compute_decision_fields
# =============================================================================

class TestTierAny:
    """Verify tier_any = tier_dev for dev, tier_commercial for commercial."""

    def test_dev_archetype_tier_any_equals_tier_dev(self):
        """For drug_developer, tier_any should equal tier_dev."""
        rec = _rec(catalyst_days=45, catalyst_in_window=True)
        result = compute_decision_fields(rec, "drug_developer", 0.75)
        assert result["tier_any"] == result["tier_dev"]
        assert result["tier_commercial"] == ""

    def test_commercial_archetype_tier_any_equals_tier_commercial(self):
        """For commercial_biotech, tier_any should equal tier_commercial."""
        rec = _rec(catalyst_days=45, catalyst_in_window=True)
        result = compute_decision_fields(
            rec, "commercial_biotech", None,
            commercial_quality_pct=0.90,
        )
        assert result["tier_any"] == result["tier_commercial"]
        assert result["tier_dev"] == ""
        assert result["tier_any"] == "A"


# =============================================================================
# 10–12: Sort key behavior
# =============================================================================

class TestSortKeyModes:
    """Sort key tests for dev_first vs tier_first mode."""

    def _make_fields(self, tier_dev="", tier_commercial="", eligible="1"):
        """Build minimal decision_fields dict for sort key testing."""
        tier_any = tier_dev or tier_commercial
        return {
            "eligible": eligible,
            "tier_dev": tier_dev,
            "tier_commercial": tier_commercial,
            "tier_any": tier_any,
            "tier_any_reason": "",
            "catalyst_mode": "specific_days",
            "catalyst_days": 45,
            "catalyst_strength": "near",
            "catalyst_decay_w": 1.0,
            "sponsor_tier1_count": 3,
            "mom_state": "neutral",
            "missingness_penalty": 0,
        }

    def test_dev_first_dev_b_before_commercial_a(self):
        """In dev_first mode, dev B sorts before commercial A."""
        dev_b = self._make_fields(tier_dev="B")
        comm_a = self._make_fields(tier_commercial="A")

        key_dev = compute_actionable_sort_key(
            dev_b, "drug_developer", 0.70, 10, "DEV_B", ruleset=DEV_FIRST_RS,
        )
        key_comm = compute_actionable_sort_key(
            comm_a, "commercial_biotech", None, 5, "COMM_A", ruleset=DEV_FIRST_RS,
        )
        assert key_dev < key_comm, "dev B should sort before commercial A in dev_first"

    def test_tier_first_commercial_a_beats_dev_b(self):
        """In tier_first mode, commercial A sorts before dev B."""
        dev_b = self._make_fields(tier_dev="B")
        comm_a = self._make_fields(tier_commercial="A")

        key_dev = compute_actionable_sort_key(
            dev_b, "drug_developer", 0.70, 10, "DEV_B", ruleset=TIER_FIRST_RS,
        )
        key_comm = compute_actionable_sort_key(
            comm_a, "commercial_biotech", None, 5, "COMM_A", ruleset=TIER_FIRST_RS,
        )
        assert key_comm < key_dev, "commercial A should sort before dev B in tier_first"

    def test_tier_first_same_tier_dev_preferred(self):
        """In tier_first mode, within same tier (A), dev sorts first."""
        dev_a = self._make_fields(tier_dev="A")
        comm_a = self._make_fields(tier_commercial="A")

        key_dev = compute_actionable_sort_key(
            dev_a, "drug_developer", 0.80, 10, "DEV_A", ruleset=TIER_FIRST_RS,
        )
        key_comm = compute_actionable_sort_key(
            comm_a, "commercial_biotech", None, 5, "COMM_A", ruleset=TIER_FIRST_RS,
        )
        assert key_dev < key_comm, "within same tier, dev should sort before commercial"


# =============================================================================
# 13–14: Portfolio filter behavior
# =============================================================================

class TestPortfolioFilter:
    """Verify portfolio filter includes/excludes commercial names by mode."""

    def _make_eligible_rows(self):
        """Build a mix of eligible dev + commercial rows."""
        dev_a = {
            "ticker": "DEV_A", "eligible": "1", "archetype": "drug_developer",
            "tier_dev": "A", "tier_commercial": "", "tier_any": "A",
        }
        dev_b = {
            "ticker": "DEV_B", "eligible": "1", "archetype": "drug_developer",
            "tier_dev": "B", "tier_commercial": "", "tier_any": "B",
        }
        comm_a = {
            "ticker": "COMM_A", "eligible": "1", "archetype": "commercial_biotech",
            "tier_dev": "", "tier_commercial": "A", "tier_any": "A",
        }
        comm_c = {
            "ticker": "COMM_C", "eligible": "1", "archetype": "commercial_biotech",
            "tier_dev": "", "tier_commercial": "C", "tier_any": "C",
        }
        return [dev_a, dev_b, comm_a, comm_c]

    def test_tier_first_commercial_a_enters_portfolio(self):
        """In tier_first mode, commercial A enters portfolio."""
        tier_filter = ["A", "B"]
        rows = self._make_eligible_rows()

        # tier_first filter
        portfolio = [r for r in rows if r.get("tier_any") in tier_filter]
        tickers = {r["ticker"] for r in portfolio}
        assert "COMM_A" in tickers
        assert "COMM_C" not in tickers
        assert "DEV_A" in tickers
        assert "DEV_B" in tickers

    def test_dev_first_commercial_a_excluded(self):
        """In dev_first mode, commercial A excluded from portfolio."""
        tier_filter = ["A", "B"]
        rows = self._make_eligible_rows()

        # dev_first filter
        portfolio = [
            r for r in rows
            if r.get("archetype") == "drug_developer"
            and r.get("tier_dev") in tier_filter
        ]
        tickers = {r["ticker"] for r in portfolio}
        assert "COMM_A" not in tickers
        assert "DEV_A" in tickers
        assert "DEV_B" in tickers


# =============================================================================
# 15: Sizing does not give commercial names dev optionality boost
# =============================================================================

class TestSizingTierFirst:
    """Verify commercial names do NOT get dev optionality sizing boost.

    The tier_a_dev sizing check requires optionality_pct_dev, which is None
    for commercial names.  Even in tier_first mode, commercial A-tier names
    should not receive the tier_a_dev band boost (commercial quality != dev
    optionality).
    """

    def test_commercial_a_no_dev_sizing_boost(self):
        """Commercial A-tier does not get tier_a_dev sizing boost (no optionality data)."""
        rs = DecisionRuleset(tiering_priority_mode="tier_first")
        rec = _rec(catalyst_days=45, catalyst_in_window=True, tier1_count=3)
        result = compute_decision_fields(
            rec, "commercial_biotech", None,
            ruleset=rs, commercial_quality_pct=0.90,
        )
        assert result["tier_commercial"] == "A"
        assert result["tier_any"] == "A"
        # No tier_a_dev boost — commercial quality is not optionality
        assert "tier_a_dev" not in result["size_reasons"]

    def test_dev_a_still_gets_sizing_boost(self):
        """Dev A-tier still gets tier_a_dev sizing boost (unchanged)."""
        rs = DecisionRuleset(tiering_priority_mode="tier_first")
        rec = _rec(catalyst_days=45, catalyst_in_window=True, tier1_count=3)
        result = compute_decision_fields(
            rec, "drug_developer", 0.75,
            ruleset=rs,
        )
        assert result["tier_dev"] == "A"
        assert "tier_a_dev" in result["size_reasons"]


# =============================================================================
# Validation tests
# =============================================================================

class TestPlatformArchetypes:
    """Platform archetypes (diagnostics, devices, services) get commercial tiers."""

    @pytest.mark.parametrize("arch", [
        "platform_diagnostics", "platform_devices", "platform_services",
    ])
    def test_platform_gets_commercial_tier(self, arch):
        """platform_* with quality data → non-empty tier."""
        tier, reason = _compute_tier_commercial(
            archetype=arch,
            eligible=True,
            quality_pct=0.90,
            catalyst_in_window="1",
            catalyst_days=45,
            ruleset=DEV_FIRST_RS,
        )
        assert tier == "A"
        assert "high_quality" in reason

    @pytest.mark.parametrize("arch", [
        "platform_diagnostics", "platform_devices", "platform_services",
    ])
    def test_platform_ineligible_gets_D(self, arch):
        """Ineligible platform → D."""
        tier, reason = _compute_tier_commercial(
            archetype=arch,
            eligible=False,
            quality_pct=0.90,
            catalyst_in_window="1",
            catalyst_days=45,
            ruleset=DEV_FIRST_RS,
        )
        assert tier == "D"
        assert reason == "ineligible"

    def test_platform_tier_any_populated(self):
        """platform_diagnostics through compute_decision_fields → tier_any set."""
        rec = _rec(ticker="GH", catalyst_days=60, catalyst_in_window=True)
        d = compute_decision_fields(
            rec, "platform_diagnostics", optionality_pct_dev=None,
            ruleset=DEV_FIRST_RS, commercial_quality_pct=0.75,
        )
        assert d["tier_dev"] == ""
        assert d["tier_commercial"] != ""
        assert d["tier_any"] == d["tier_commercial"]

    def test_empty_archetype_no_tier(self):
        """Empty archetype → no tier assigned."""
        tier, reason = _compute_tier_commercial(
            archetype="",
            eligible=True,
            quality_pct=0.90,
            catalyst_in_window="1",
            catalyst_days=45,
            ruleset=DEV_FIRST_RS,
        )
        assert tier == ""
        assert reason == ""


class TestValidation:
    """Validate tiering_priority_mode validation."""

    def test_invalid_tiering_mode_rejected(self):
        with pytest.raises(ValueError, match="tiering_priority_mode"):
            DecisionRuleset(tiering_priority_mode="bogus")

    def test_dev_first_default(self):
        rs = DecisionRuleset()
        assert rs.tiering_priority_mode == "dev_first"
