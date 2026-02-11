"""Tests for gate margin diagnostics — compute_gate_margins, compute_tier_margins.

These functions are panel-only analytics that compute how far each ticker is
from each gate threshold (margin), whether it was rescued by the relative gate,
and what minimal change would flip eligibility (counterfactual).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from decision_engine import (
    DECISION_COLUMNS,
    DecisionRuleset,
    compute_gate_margins,
    compute_tier_margins,
)
from run_decision_strategy_backtest import PANEL_COLUMNS

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from run_walkforward_report import compute_gate_pressure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rec(
    drawdown: Optional[float] = None,
    drawdown_rel_xbi: Optional[float] = None,
    fundamental_red_flag: bool = False,
    severity: str = "",
    adv_fail: bool = False,
) -> Dict[str, Any]:
    """Build a minimal rec dict for gate margin testing."""
    rec: Dict[str, Any] = {
        "fundamental_red_flag": fundamental_red_flag,
        "severity": severity,
        "defensive_features": {},
        "attn_flags": [],
        "flags": [],
    }
    if drawdown is not None:
        rec["defensive_features"]["drawdown"] = drawdown
    if drawdown_rel_xbi is not None:
        rec["defensive_features"]["drawdown_rel_xbi"] = drawdown_rel_xbi
    if adv_fail:
        rec["attn_flags"] = ["adv_fail"]
    return rec


DEFAULT_RULESET = DecisionRuleset()


# ===========================================================================
# TestGateMarginsMath — 6 tests
# ===========================================================================
class TestGateMarginsMath:
    """Test the arithmetic of dd_abs_margin and dd_rel_margin."""

    def test_passing_abs_margin_positive(self):
        rec = _rec(drawdown=-0.15)
        result = compute_gate_margins(rec, DEFAULT_RULESET)
        # margin = dd - gate = -0.15 - (-0.40) = +0.25
        assert result["dd_abs_margin"] is not None
        assert abs(result["dd_abs_margin"] - 0.25) < 1e-5

    def test_failing_abs_margin_negative(self):
        rec = _rec(drawdown=-0.50)
        result = compute_gate_margins(rec, DEFAULT_RULESET)
        # margin = -0.50 - (-0.40) = -0.10
        assert result["dd_abs_margin"] is not None
        assert abs(result["dd_abs_margin"] - (-0.10)) < 1e-5

    def test_boundary_abs_margin_zero(self):
        rec = _rec(drawdown=-0.40)
        result = compute_gate_margins(rec, DEFAULT_RULESET)
        # margin = -0.40 - (-0.40) = 0.0
        assert result["dd_abs_margin"] is not None
        assert abs(result["dd_abs_margin"]) < 1e-5

    def test_none_dd_abs_margin_is_none(self):
        rec = _rec(drawdown=None)
        result = compute_gate_margins(rec, DEFAULT_RULESET)
        assert result["dd_abs_margin"] is None

    def test_rel_margin_positive(self):
        rec = _rec(drawdown=-0.15, drawdown_rel_xbi=-0.10)
        result = compute_gate_margins(rec, DEFAULT_RULESET)
        # margin = -0.10 - (-0.20) = +0.10
        assert result["dd_rel_margin"] is not None
        assert abs(result["dd_rel_margin"] - 0.10) < 1e-5

    def test_rel_margin_none_when_missing(self):
        rec = _rec(drawdown=-0.15, drawdown_rel_xbi=None)
        result = compute_gate_margins(rec, DEFAULT_RULESET)
        assert result["dd_rel_margin"] is None


# ===========================================================================
# TestRescuedByRel — 4 tests
# ===========================================================================
class TestRescuedByRel:
    """Test the rescued_by_rel logic."""

    def test_rescued_abs_fails_rel_passes_require_both(self):
        """Abs breaches (<-0.40) but rel passes (>=-0.20) with require_both=True."""
        rs = DecisionRuleset(drawdown_gate_require_both=True)
        rec = _rec(drawdown=-0.50, drawdown_rel_xbi=-0.10)
        result = compute_gate_margins(rec, rs)
        assert result["rescued_by_rel"] is True

    def test_not_rescued_both_fail(self):
        """Both abs and rel breach — not rescued."""
        rs = DecisionRuleset(drawdown_gate_require_both=True)
        rec = _rec(drawdown=-0.50, drawdown_rel_xbi=-0.30)
        result = compute_gate_margins(rec, rs)
        assert result["rescued_by_rel"] is False

    def test_not_rescued_abs_passes(self):
        """Abs passes — not rescued (no need for rescue)."""
        rs = DecisionRuleset(drawdown_gate_require_both=True)
        rec = _rec(drawdown=-0.15, drawdown_rel_xbi=-0.05)
        result = compute_gate_margins(rec, rs)
        assert result["rescued_by_rel"] is False

    def test_not_rescued_require_either(self):
        """Abs fails, rel passes, but mode=require_either — not rescued."""
        rs = DecisionRuleset(drawdown_gate_require_both=False)
        rec = _rec(drawdown=-0.50, drawdown_rel_xbi=-0.10)
        result = compute_gate_margins(rec, rs)
        assert result["rescued_by_rel"] is False


# ===========================================================================
# TestTierMargins — 3 tests
# ===========================================================================
class TestTierMargins:
    """Test compute_tier_margins arithmetic."""

    def test_above_a_floor(self):
        result = compute_tier_margins(0.75, "near", DEFAULT_RULESET)
        # margin = 0.75 - 0.60 = +0.15
        assert result["optionality_margin_a"] is not None
        assert abs(result["optionality_margin_a"] - 0.15) < 1e-5
        assert result["actionable_catalyst"] is True

    def test_below_a_floor(self):
        result = compute_tier_margins(0.40, "far", DEFAULT_RULESET)
        # margin = 0.40 - 0.60 = -0.20
        assert result["optionality_margin_a"] is not None
        assert abs(result["optionality_margin_a"] - (-0.20)) < 1e-5
        assert result["actionable_catalyst"] is False

    def test_none_optionality(self):
        result = compute_tier_margins(None, "near", DEFAULT_RULESET)
        assert result["optionality_margin_a"] is None
        assert result["optionality_margin_b"] is None
        # Actionable depends on strength alone
        assert result["actionable_catalyst"] is True


# ===========================================================================
# TestCounterfactual — 3 tests
# ===========================================================================
class TestCounterfactual:
    """Test counterfactual field in compute_gate_margins."""

    def test_eligible_empty(self):
        """Eligible ticker → counterfactual is empty."""
        rec = _rec(drawdown=-0.15, drawdown_rel_xbi=-0.05)
        result = compute_gate_margins(rec, DEFAULT_RULESET)
        assert result["counterfactual"] == {}

    def test_ineligible_red_flag(self):
        """Red flag → counterfactual suggests removing it."""
        rec = _rec(fundamental_red_flag=True)
        result = compute_gate_margins(rec, DEFAULT_RULESET)
        assert "fundamental_red_flag" in result["counterfactual"]
        assert result["counterfactual"]["fundamental_red_flag"] is False

    def test_ineligible_drawdown_require_both_minimal(self):
        """Both drawdowns breach with require_both → counterfactual flips ONE (smaller delta)."""
        rs = DecisionRuleset(drawdown_gate_require_both=True)
        # dd=-0.50: abs_margin = -0.50 - (-0.40) = -0.10
        # dd_rel=-0.30: rel_margin = -0.30 - (-0.20) = -0.10
        # Equal deltas → abs wins (<=)
        rec = _rec(drawdown=-0.50, drawdown_rel_xbi=-0.30)
        result = compute_gate_margins(rec, rs)
        cf = result["counterfactual"]
        assert "drawdown" in cf
        assert cf["drawdown"] == rs.drawdown_gate
        # Minimal: only ONE side, not both
        assert "drawdown_rel_xbi" not in cf

    def test_ineligible_drawdown_require_both_picks_smaller_delta(self):
        """require_both: counterfactual picks the side with smaller |margin|."""
        rs = DecisionRuleset(drawdown_gate_require_both=True)
        # dd=-0.45: abs_margin = -0.45 - (-0.40) = -0.05
        # dd_rel=-0.35: rel_margin = -0.35 - (-0.20) = -0.15
        # abs delta (0.05) < rel delta (0.15) → flip abs
        rec = _rec(drawdown=-0.45, drawdown_rel_xbi=-0.35)
        result = compute_gate_margins(rec, rs)
        cf = result["counterfactual"]
        assert "drawdown" in cf
        assert "drawdown_rel_xbi" not in cf

    def test_ineligible_drawdown_require_both_picks_rel_when_closer(self):
        """require_both: when rel delta is smaller, counterfactual picks rel."""
        rs = DecisionRuleset(drawdown_gate_require_both=True)
        # dd=-0.70: abs_margin = -0.70 - (-0.40) = -0.30
        # dd_rel=-0.22: rel_margin = -0.22 - (-0.20) = -0.02
        # rel delta (0.02) < abs delta (0.30) → flip rel
        rec = _rec(drawdown=-0.70, drawdown_rel_xbi=-0.22)
        result = compute_gate_margins(rec, rs)
        cf = result["counterfactual"]
        assert "drawdown_rel_xbi" in cf
        assert "drawdown" not in cf


# ===========================================================================
# TestEffectiveFirstFail — 5 tests
# ===========================================================================
class TestEffectiveFirstFail:
    """Test first_failed_effective_gate uses combined gate order."""

    def test_rescued_ticker_effective_gate_empty(self):
        """Rescued ticker: abs breaches but combined passes → effective gate is empty."""
        rs = DecisionRuleset(drawdown_gate_require_both=True)
        rec = _rec(drawdown=-0.50, drawdown_rel_xbi=-0.10)
        result = compute_gate_margins(rec, rs)
        assert result["rescued_by_rel"] is True
        assert result["first_failed_effective_gate"] == ""
        assert result["counterfactual"] == {}

    def test_both_breach_effective_gate_is_deep_drawdown(self):
        """Both abs and rel breach → effective gate is deep_drawdown."""
        rs = DecisionRuleset(drawdown_gate_require_both=True)
        rec = _rec(drawdown=-0.50, drawdown_rel_xbi=-0.30)
        result = compute_gate_margins(rec, rs)
        assert result["first_failed_effective_gate"] == "deep_drawdown"

    def test_red_flag_precedes_drawdown(self):
        """Red flag fires before drawdown in effective gate order."""
        rs = DecisionRuleset(drawdown_gate_require_both=True)
        rec = _rec(drawdown=-0.50, drawdown_rel_xbi=-0.30, fundamental_red_flag=True)
        result = compute_gate_margins(rec, rs)
        assert result["first_failed_effective_gate"] == "fundamental_red_flag"

    def test_clean_eligible_both_gates_empty(self):
        """Clean eligible ticker: both first_failed fields are empty."""
        rec = _rec(drawdown=-0.15, drawdown_rel_xbi=-0.05)
        result = compute_gate_margins(rec, DEFAULT_RULESET)
        assert result["first_failed_gate"] == ""
        assert result["first_failed_effective_gate"] == ""

    def test_require_either_one_breach_effective_deep_drawdown(self):
        """require_either: single abs breach → effective gate is deep_drawdown."""
        rs = DecisionRuleset(drawdown_gate_require_both=False)
        rec = _rec(drawdown=-0.50, drawdown_rel_xbi=-0.10)
        result = compute_gate_margins(rec, rs)
        # OR mode: abs breach alone triggers combined gate failure
        assert result["first_failed_effective_gate"] == "deep_drawdown"


# ===========================================================================
# TestCounterfactualRequireEither — 2 tests
# ===========================================================================
class TestCounterfactualRequireEither:
    """Test minimal counterfactual under OR/require_either semantics."""

    def test_require_either_only_abs_breaching(self):
        """require_either: only abs breaches → counterfactual flips only abs."""
        rs = DecisionRuleset(drawdown_gate_require_both=False)
        rec = _rec(drawdown=-0.50, drawdown_rel_xbi=-0.10)
        result = compute_gate_margins(rec, rs)
        cf = result["counterfactual"]
        assert "drawdown" in cf
        assert cf["drawdown"] == rs.drawdown_gate
        # Rel didn't breach, so shouldn't appear
        assert "drawdown_rel_xbi" not in cf

    def test_require_either_both_breaching(self):
        """require_either: both breach → counterfactual includes both."""
        rs = DecisionRuleset(drawdown_gate_require_both=False)
        rec = _rec(drawdown=-0.50, drawdown_rel_xbi=-0.30)
        result = compute_gate_margins(rec, rs)
        cf = result["counterfactual"]
        assert "drawdown" in cf
        assert "drawdown_rel_xbi" in cf


# ===========================================================================
# TestPanelSchema — 2 tests
# ===========================================================================
class TestPanelSchema:
    """Verify margin columns are panel-only (NOT in DECISION_COLUMNS)."""

    MARGIN_COLS = [
        "dd_abs_margin", "dd_rel_margin", "rescued_by_rel",
        "optionality_margin_a", "actionable_catalyst",
    ]

    def test_margin_columns_in_panel_columns(self):
        for col in self.MARGIN_COLS:
            assert col in PANEL_COLUMNS, f"{col} missing from PANEL_COLUMNS"

    def test_margin_columns_not_in_decision_columns(self):
        for col in self.MARGIN_COLS:
            assert col not in DECISION_COLUMNS, f"{col} should NOT be in DECISION_COLUMNS"


# ===========================================================================
# TestGatePressure — walkforward compute_gate_pressure
# ===========================================================================
class TestGatePressure:
    """Test compute_gate_pressure from walkforward report."""

    def test_known_pressure_values(self):
        """Verify pressure percentages with known data."""
        rows = [
            {"dd_abs_margin": "0.03", "dd_rel_margin": "0.10",
             "optionality_margin_a": "0.02", "rescued_by_rel": "0", "eligible": "1"},
            {"dd_abs_margin": "-0.04", "dd_rel_margin": "-0.03",
             "optionality_margin_a": "0.20", "rescued_by_rel": "1", "eligible": "1"},
            {"dd_abs_margin": "0.25", "dd_rel_margin": "0.15",
             "optionality_margin_a": "-0.01", "rescued_by_rel": "0", "eligible": "1"},
            {"dd_abs_margin": "-0.15", "dd_rel_margin": "-0.20",
             "optionality_margin_a": "-0.10", "rescued_by_rel": "0", "eligible": "0"},
        ]
        result = compute_gate_pressure(rows)
        # dd_abs: 2/4 near (0.03, -0.04) = 50%
        assert result["dd_abs_near_gate_pct"] == 50.0
        # dd_rel: 1/4 near (-0.03) = 25%
        assert result["dd_rel_near_gate_pct"] == 25.0
        # opt_a: 2/4 near (0.02, -0.01) = 50%
        assert result["optionality_near_a_floor_pct"] == 50.0
        # rescued: 1/3 eligible = 33.3%
        assert abs(result["rescued_share_pct"] - 33.3) < 0.1
        assert result["rescued_count"] == 1
        assert result["eligible_count"] == 3

    def test_empty_panel(self):
        """Empty panel → None for pressure, 0 for rescued_share."""
        result = compute_gate_pressure([])
        assert result["dd_abs_near_gate_pct"] is None
        assert result["rescued_share_pct"] == 0.0
        assert result["rescued_count"] == 0

    def test_missing_margin_columns(self):
        """Rows without margin columns → None."""
        rows = [{"ticker": "AAA", "eligible": "1"}]
        result = compute_gate_pressure(rows)
        assert result["dd_abs_near_gate_pct"] is None
        assert result["dd_rel_near_gate_pct"] is None
        assert result["optionality_near_a_floor_pct"] is None
