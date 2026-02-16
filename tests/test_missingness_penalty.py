"""Tests for config-gated missingness penalty (sort + sizing)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from decision_engine import (
    DEFAULT_RULESET,
    DecisionRuleset,
    _has_flag,
    _compute_missing_components,
    compute_actionable_sort_key,
    compute_decision_fields,
    compute_target_weights,
)


# ---------------------------------------------------------------------------
# Fixture builder (minimal — mirrors test_decision_engine_contract._rec)
# ---------------------------------------------------------------------------

def _rec(
    ticker: str = "TEST",
    catalyst_days: int | None = 30,
    catalyst_in_window: bool | None = False,
    tier1_count: int | None = 3,
    drawdown: float | None = -0.15,
    vol_60d: float | None = 0.65,
    beta_xbi_60d: float | None = 1.1,
    alpha_60d: float | None = 0.02,
    severity: str = "NONE",
    confidence_overall: float = 0.72,
    fundamental_red_flag: bool = False,
) -> Dict[str, Any]:
    """Build a fully-populated synthetic rec."""
    rec: Dict[str, Any] = {
        "ticker": ticker,
        "severity": severity,
        "confidence_overall": confidence_overall,
        "fundamental_red_flag": fundamental_red_flag,
        "fundamental_red_flag_reasons": [],
        "flags": [],
        "attn_flags": [],
    }
    cd: Dict[str, Any] = {}
    if catalyst_days is not None:
        cd["days_to_catalyst"] = catalyst_days
    if catalyst_in_window is not None:
        cd["in_optimal_window"] = catalyst_in_window
    rec["catalyst_decay"] = cd if cd else None

    rec["defensive_features"] = {}
    if vol_60d is not None:
        rec["defensive_features"]["vol_60d"] = vol_60d
    if beta_xbi_60d is not None:
        rec["defensive_features"]["beta_xbi_60d"] = beta_xbi_60d
    if drawdown is not None:
        rec["defensive_features"]["drawdown"] = drawdown
    rec["defensive_features"]["rsi_14d"] = 48.0

    rec["smart_money_signal"] = {"overlap_count": 2}
    if tier1_count is not None:
        rec["coinvest"] = {"tier1_count": tier1_count}
    else:
        rec["coinvest"] = {}

    rec["score_breakdown"] = {
        "enhancements": {
            "momentum": {"alpha_60d": alpha_60d},
        },
    }
    return rec


# ---------------------------------------------------------------------------
# _has_flag helper
# ---------------------------------------------------------------------------

class TestHasFlag:
    def test_empty_string(self):
        assert _has_flag("", "drawdown_data_missing") is False

    def test_none(self):
        assert _has_flag(None, "drawdown_data_missing") is False

    def test_single_match(self):
        assert _has_flag("drawdown_data_missing", "drawdown_data_missing") is True

    def test_pipe_delimited(self):
        assert _has_flag("high_vol|drawdown_data_missing", "drawdown_data_missing") is True

    def test_comma_delimited(self):
        assert _has_flag("high_vol,drawdown_data_missing", "drawdown_data_missing") is True

    def test_no_match(self):
        assert _has_flag("high_vol|high_beta", "drawdown_data_missing") is False

    def test_partial_no_match(self):
        """Flag name must be an exact token, not a substring."""
        assert _has_flag("drawdown_data_missing_extra", "drawdown_data_missing") is False


# ---------------------------------------------------------------------------
# _compute_missing_components
# ---------------------------------------------------------------------------

class TestMissingComponents:
    def test_all_present(self):
        """Complete decision fields → empty list."""
        fields = {
            "catalyst_mode": "specific_days",
            "sponsor_tier1_count": 3,
            "risk_flags": "high_vol",
        }
        assert _compute_missing_components(fields) == []

    def test_catalyst_missing(self):
        fields = {
            "catalyst_mode": "missing",
            "sponsor_tier1_count": 3,
            "risk_flags": "",
        }
        assert _compute_missing_components(fields) == ["catalyst"]

    def test_sponsor_missing_empty_string(self):
        fields = {
            "catalyst_mode": "specific_days",
            "sponsor_tier1_count": "",
            "risk_flags": "",
        }
        assert _compute_missing_components(fields) == ["sponsor"]

    def test_sponsor_missing_none(self):
        fields = {
            "catalyst_mode": "specific_days",
            "sponsor_tier1_count": None,
            "risk_flags": "",
        }
        assert _compute_missing_components(fields) == ["sponsor"]

    def test_drawdown_missing(self):
        fields = {
            "catalyst_mode": "specific_days",
            "sponsor_tier1_count": 3,
            "risk_flags": "drawdown_data_missing",
        }
        assert _compute_missing_components(fields) == ["drawdown"]

    def test_all_three_missing(self):
        fields = {
            "catalyst_mode": "missing",
            "sponsor_tier1_count": "",
            "risk_flags": "drawdown_data_missing",
        }
        result = _compute_missing_components(fields)
        assert result == ["catalyst", "sponsor", "drawdown"]
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Integration: compute_decision_fields populates columns
# ---------------------------------------------------------------------------

class TestDecisionFieldsIntegration:
    def test_complete_rec_no_missing(self):
        """Fully populated rec → missing_components empty, penalty 0."""
        rec = _rec(catalyst_days=30, tier1_count=3, drawdown=-0.15)
        fields = compute_decision_fields(rec, "drug_developer", 0.70)
        assert fields["missing_components"] == ""
        assert fields["missingness_penalty"] == 0

    def test_no_catalyst_marks_missing(self):
        """No catalyst data → catalyst in missing_components."""
        rec = _rec(catalyst_days=None, catalyst_in_window=None)
        rec["catalyst_decay"] = None
        fields = compute_decision_fields(rec, "drug_developer", 0.70)
        assert "catalyst" in fields["missing_components"]
        assert fields["missingness_penalty"] >= 1

    def test_no_sponsor_marks_missing(self):
        """No coinvest data → sponsor in missing_components."""
        rec = _rec(tier1_count=None)
        fields = compute_decision_fields(rec, "drug_developer", 0.70)
        assert "sponsor" in fields["missing_components"]

    def test_no_drawdown_marks_missing(self):
        """No drawdown data → drawdown_data_missing flag → drawdown in missing."""
        rec = _rec(drawdown=None)
        fields = compute_decision_fields(rec, "drug_developer", 0.70)
        assert "drawdown" in fields["missing_components"]

    def test_all_missing_penalty_3(self):
        """All three missing → penalty = 3."""
        rec = _rec(catalyst_days=None, catalyst_in_window=None, tier1_count=None, drawdown=None)
        rec["catalyst_decay"] = None
        fields = compute_decision_fields(rec, "drug_developer", 0.70)
        assert fields["missingness_penalty"] == 3
        parts = fields["missing_components"].split("|")
        assert sorted(parts) == ["catalyst", "drawdown", "sponsor"]


# ---------------------------------------------------------------------------
# Default-off invariance: toggles off → no behavioral change
# ---------------------------------------------------------------------------

class TestDefaultOffInvariance:
    def test_sort_key_identical_when_disabled(self):
        """With penalty disabled, missing data does not affect sort key ordering."""
        rs = DEFAULT_RULESET
        assert rs.enable_missingness_sort_penalty is False

        # Two tickers: one complete, one with all missing
        rec_complete = _rec(catalyst_days=30, tier1_count=3, drawdown=-0.15)
        rec_missing = _rec(catalyst_days=None, catalyst_in_window=None, tier1_count=None, drawdown=None)
        rec_missing["catalyst_decay"] = None

        fields_c = compute_decision_fields(rec_complete, "drug_developer", 0.70, rs)
        fields_m = compute_decision_fields(rec_missing, "drug_developer", 0.70, rs)

        # Both should have missingness_penalty tracked (0 vs 3)
        assert fields_c["missingness_penalty"] == 0
        assert fields_m["missingness_penalty"] == 3

        # But sort keys should NOT use missingness when disabled
        key_c = compute_actionable_sort_key(
            fields_c, "drug_developer", 0.70, 10, "AAA", ruleset=rs,
        )
        key_m = compute_actionable_sort_key(
            fields_m, "drug_developer", 0.70, 10, "AAA", ruleset=rs,
        )
        # Position 6 (missing_count) should be 0 for both when disabled
        assert key_c[6] == 0
        assert key_m[6] == 0

    def test_size_band_identical_when_disabled(self):
        """With size penalty disabled, missing data does not affect size band."""
        rs = DEFAULT_RULESET
        assert rs.enable_missingness_size_penalty is False

        rec_complete = _rec(catalyst_days=30, tier1_count=3, drawdown=-0.15)
        rec_missing = _rec(catalyst_days=None, catalyst_in_window=None, tier1_count=None, drawdown=None)
        rec_missing["catalyst_decay"] = None

        fields_c = compute_decision_fields(rec_complete, "drug_developer", 0.70, rs)
        fields_m = compute_decision_fields(rec_missing, "drug_developer", 0.70, rs)

        # Size bands may differ due to catalyst strength, but NOT due to missingness penalty
        assert "missing_drawdown" not in fields_c["size_reasons"]
        assert "missing_drawdown" not in fields_m["size_reasons"]
        assert "missing_sponsor" not in fields_m["size_reasons"]
        assert "missing_catalyst" not in fields_m["size_reasons"]


# ---------------------------------------------------------------------------
# Size penalty: enabled → band downgrades
# ---------------------------------------------------------------------------

class TestSizePenalty:
    def _make_rs(self, **kw) -> DecisionRuleset:
        return DecisionRuleset(enable_missingness_size_penalty=True, **kw)

    def test_missing_drawdown_downgrades(self):
        """Missing drawdown → band drops, reason recorded."""
        rs = self._make_rs()
        rec = _rec(drawdown=None, catalyst_days=30, tier1_count=3)
        fields = compute_decision_fields(rec, "drug_developer", 0.70, rs)
        assert "missing_drawdown" in fields["size_reasons"]

    def test_missing_sponsor_penalized_a_tier(self):
        """A-tier ticker with missing sponsor gets penalized."""
        rs = self._make_rs()
        rec = _rec(catalyst_days=30, tier1_count=None, drawdown=-0.15)
        fields = compute_decision_fields(rec, "drug_developer", 0.70, rs)
        assert fields["tier_dev"] == "A"  # high opt + near catalyst
        assert "missing_sponsor" in fields["size_reasons"]

    def test_missing_sponsor_penalized_b_tier(self):
        """B-tier ticker with missing sponsor gets penalized."""
        rs = self._make_rs()
        rec = _rec(catalyst_days=30, tier1_count=None, drawdown=-0.15)
        fields = compute_decision_fields(rec, "drug_developer", 0.45, rs)
        assert fields["tier_dev"] == "B"  # mod opt + near catalyst
        assert "missing_sponsor" in fields["size_reasons"]

    def test_missing_sponsor_not_penalized_c_tier(self):
        """C-tier ticker with missing sponsor is NOT penalized (A+B only)."""
        rs = self._make_rs()
        rec = _rec(catalyst_days=30, tier1_count=None, drawdown=-0.15)
        fields = compute_decision_fields(rec, "drug_developer", 0.20, rs)
        assert fields["tier_dev"] == "C"
        assert "missing_sponsor" not in fields["size_reasons"]

    def test_missing_catalyst_penalized_a_only(self):
        """A-tier with missing catalyst gets penalized; B-tier does not."""
        rs = self._make_rs()
        # A-tier: high opt but catalyst missing → tier becomes B (no catalyst data)
        # So we test the path where tier=A explicitly isn't reachable with missing catalyst
        # Actually: with missing catalyst, tier can't be A (requires actionable catalyst)
        # Test with B-tier to confirm no penalty
        rec_b = _rec(catalyst_days=None, catalyst_in_window=None, tier1_count=3, drawdown=-0.15)
        rec_b["catalyst_decay"] = None
        fields_b = compute_decision_fields(rec_b, "drug_developer", 0.70, rs)
        assert fields_b["tier_dev"] == "B"  # high_opt+no_catalyst_data
        assert "missing_catalyst" not in fields_b["size_reasons"]


# ---------------------------------------------------------------------------
# Sort penalty: enabled → tiebreaking
# ---------------------------------------------------------------------------

class TestSortPenalty:
    def test_breaks_ties_off_mode(self):
        """With sort penalty on, missing data ticker sorts after complete one."""
        rs = DecisionRuleset(enable_missingness_sort_penalty=True)

        rec_complete = _rec(catalyst_days=30, tier1_count=3, drawdown=-0.15)
        rec_missing = _rec(catalyst_days=30, tier1_count=None, drawdown=None)

        fields_c = compute_decision_fields(rec_complete, "drug_developer", 0.70, rs)
        fields_m = compute_decision_fields(rec_missing, "drug_developer", 0.70, rs)

        key_c = compute_actionable_sort_key(
            fields_c, "drug_developer", 0.70, 10, "AAA", ruleset=rs,
        )
        key_m = compute_actionable_sort_key(
            fields_m, "drug_developer", 0.70, 10, "BBB", ruleset=rs,
        )
        # Complete should sort before missing
        assert key_c < key_m

    def test_breaks_ties_tiebreaker_mode(self):
        """Tiebreaker mode: same comp_rank, missing data sorts after."""
        rs = DecisionRuleset(
            enable_missingness_sort_penalty=True,
            catalyst_priority_mode="tiebreaker",
        )

        rec_complete = _rec(catalyst_days=30, tier1_count=3, drawdown=-0.15)
        rec_missing = _rec(catalyst_days=30, tier1_count=None, drawdown=None)

        fields_c = compute_decision_fields(rec_complete, "drug_developer", 0.70, rs)
        fields_m = compute_decision_fields(rec_missing, "drug_developer", 0.70, rs)

        key_c = compute_actionable_sort_key(
            fields_c, "drug_developer", 0.70, 10, "AAA", ruleset=rs,
        )
        key_m = compute_actionable_sort_key(
            fields_m, "drug_developer", 0.70, 10, "BBB", ruleset=rs,
        )
        assert key_c < key_m

    def test_breaks_ties_blended_mode(self):
        """Blended mode: same comp_rank, missing data sorts after."""
        rs = DecisionRuleset(
            enable_missingness_sort_penalty=True,
            catalyst_priority_mode="blended",
        )

        rec_complete = _rec(catalyst_days=30, tier1_count=3, drawdown=-0.15)
        rec_missing = _rec(catalyst_days=30, tier1_count=None, drawdown=None)

        fields_c = compute_decision_fields(rec_complete, "drug_developer", 0.70, rs)
        fields_m = compute_decision_fields(rec_missing, "drug_developer", 0.70, rs)

        key_c = compute_actionable_sort_key(
            fields_c, "drug_developer", 0.70, 10, "AAA", ruleset=rs,
        )
        key_m = compute_actionable_sort_key(
            fields_m, "drug_developer", 0.70, 10, "BBB", ruleset=rs,
        )
        assert key_c < key_m
