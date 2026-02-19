#!/usr/bin/env python3
"""Tests for cost-aware sizing (L3-only, behind enable_cost_haircut flag).

Invariants:
1. Eligibility unchanged with/without cost haircut (same gates)
2. No cost data → no change (cost_mult==1.0, band unchanged)
3. High cost → haircut applied (weight reduced; optional band step-down)
4. Very high cost floors at min multiplier
5. Disabled flag → no change even with cost data
6. Band step-down only fires at heavy haircut (cost_mult <= 0.70)
7. Weight haircut propagates through compute_target_weights
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from decision_engine import (
    DecisionRuleset,
    _compute_cost_mult,
    _compute_size_band,
    compute_decision_fields,
    compute_target_weights,
    DECISION_COLUMNS,
)


# ---------------------------------------------------------------------------
# Rulesets
# ---------------------------------------------------------------------------
DISABLED_RS = DecisionRuleset()  # enable_cost_haircut=False (default)
ENABLED_RS = DecisionRuleset(enable_cost_haircut=True)


def _minimal_rec(drawdown=-0.10, severity="", flags=None):
    """Build a minimal rec dict that passes eligibility."""
    return {
        "defensive_features": {
            "drawdown": drawdown,
            "drawdown_xbi": -0.05,
            "drawdown_rel_xbi": drawdown - (-0.05),
            "vol_60d": "0.30",
            "beta_xbi_60d": "0.80",
            "rsi_14d": "50",
        },
        "fundamental_red_flag": False,
        "severity": severity,
        "confidence_overall": 0.5,
        "smart_money_signal": {},
        "coinvest": {},
        "catalyst_decay": {},
        "score_breakdown": {"enhancements": {}},
        "momentum_signal": {},
        "data_quality_flags": flags or [],
    }


# ============================================================================
# 1. _compute_cost_mult unit tests
# ============================================================================
class TestComputeCostMult:

    def test_disabled_returns_1(self):
        """When enable_cost_haircut=False, always returns 1.0."""
        mult, bucket = _compute_cost_mult(200.0, DISABLED_RS)
        assert mult == 1.0
        assert bucket == ""

    def test_none_cost_returns_1(self):
        """When est_cost_bps is None, always returns 1.0."""
        mult, bucket = _compute_cost_mult(None, ENABLED_RS)
        assert mult == 1.0
        assert bucket == ""

    def test_low_cost_no_haircut(self):
        """Cost <= 400 bps → multiplier 1.0."""
        mult, bucket = _compute_cost_mult(200.0, ENABLED_RS)
        assert mult == 1.0
        assert bucket == "<=400bps"

    def test_boundary_400(self):
        """Cost exactly at 400 bps → first bucket (1.0)."""
        mult, _ = _compute_cost_mult(400.0, ENABLED_RS)
        assert mult == 1.0

    def test_moderate_cost_haircut(self):
        """Cost 400-1000 bps → 0.85 multiplier."""
        mult, bucket = _compute_cost_mult(700.0, ENABLED_RS)
        assert mult == 0.85
        assert bucket == "<=1000bps"

    def test_high_cost_haircut(self):
        """Cost 1000-2000 bps → 0.70 multiplier."""
        mult, bucket = _compute_cost_mult(1500.0, ENABLED_RS)
        assert mult == 0.70
        assert bucket == "<=2000bps"

    def test_very_high_cost_floor(self):
        """Cost > 2000 bps → floor multiplier 0.55."""
        mult, bucket = _compute_cost_mult(3000.0, ENABLED_RS)
        assert mult == 0.55
        assert bucket == ">2000bps"

    def test_zero_cost(self):
        """Cost = 0 bps → first bucket (1.0)."""
        mult, _ = _compute_cost_mult(0.0, ENABLED_RS)
        assert mult == 1.0


# ============================================================================
# 2. Eligibility unchanged with cost haircut
# ============================================================================
class TestEligibilityUnchanged:

    def test_eligibility_same_with_and_without_cost(self):
        """Cost haircut must NOT affect eligibility gates."""
        rec = _minimal_rec()
        fields_no_cost = compute_decision_fields(rec, "drug_developer", 0.80, ENABLED_RS)
        fields_with_cost = compute_decision_fields(
            rec, "drug_developer", 0.80, ENABLED_RS, est_cost_bps=200.0
        )
        assert fields_no_cost["eligible"] == fields_with_cost["eligible"]
        assert fields_no_cost["ineligible_reasons"] == fields_with_cost["ineligible_reasons"]

    def test_ineligible_rec_unaffected_by_cost(self):
        """Ineligible ticker stays ineligible regardless of cost data."""
        rec = _minimal_rec(drawdown=-0.50)  # breaches hard gate
        fields = compute_decision_fields(
            rec, "drug_developer", 0.80, ENABLED_RS, est_cost_bps=10.0
        )
        assert fields["eligible"] == "0"


# ============================================================================
# 3. Band step-down at heavy haircut
# ============================================================================
class TestBandStepDown:

    def test_no_step_down_at_085(self):
        """cost_mult=0.85 should NOT trigger band step-down."""
        band, reasons = _compute_size_band(
            eligible=True, effective_tier="B", optionality=0.50,
            overlays={}, ruleset=ENABLED_RS,
            cost_mult=0.85, cost_bucket="<=1000bps",
        )
        assert "cost_haircut_<=1000bps" not in reasons

    def test_step_down_at_070(self):
        """cost_mult=0.70 triggers band step-down."""
        band, reasons = _compute_size_band(
            eligible=True, effective_tier="B", optionality=0.50,
            overlays={}, ruleset=ENABLED_RS,
            cost_mult=0.70, cost_bucket="<=2000bps",
        )
        assert "cost_haircut_<=2000bps" in reasons
        # Baseline without cost is M (idx=2); step-down → S (idx=1)
        assert band == "S"

    def test_step_down_at_floor(self):
        """cost_mult=0.55 (floor) also triggers band step-down."""
        band, reasons = _compute_size_band(
            eligible=True, effective_tier="B", optionality=0.50,
            overlays={}, ruleset=ENABLED_RS,
            cost_mult=0.55, cost_bucket=">2000bps",
        )
        assert "cost_haircut_>2000bps" in reasons

    def test_band_clamps_at_xs(self):
        """Band step-down can't go below XS."""
        # Headwind pushes to S, cost pushes again — should clamp at XS
        band, reasons = _compute_size_band(
            eligible=True, effective_tier="C", optionality=0.10,
            overlays={"mom_state": "headwind", "runway_bucket": "critical"},
            ruleset=ENABLED_RS,
            cost_mult=0.55, cost_bucket=">2000bps",
        )
        assert band == "XS"


# ============================================================================
# 4. Weight haircut through compute_target_weights
# ============================================================================
class TestWeightHaircut:

    def test_cost_mult_reduces_weight(self):
        """Row with cost_mult < 1.0 gets lower weight than row without."""
        rows = [
            {"size_band": "M", "cost_mult": 1.0},
            {"size_band": "M", "cost_mult": 0.70},
        ]
        compute_target_weights(rows, ENABLED_RS)
        # Both are M band, but second has 0.70 multiplier
        assert rows[0]["target_weight_pct"] > rows[1]["target_weight_pct"]

    def test_missing_cost_mult_defaults_to_1(self):
        """Rows without cost_mult field get multiplier 1.0 (backward compat)."""
        rows_with = [{"size_band": "M", "cost_mult": 1.0}]
        rows_without = [{"size_band": "M"}]
        compute_target_weights(rows_with, ENABLED_RS)
        compute_target_weights(rows_without, ENABLED_RS)
        assert rows_with[0]["target_weight_pct"] == rows_without[0]["target_weight_pct"]


# ============================================================================
# 5. Output fields present in DECISION_COLUMNS
# ============================================================================
class TestOutputSchema:

    def test_cost_columns_in_decision_columns(self):
        """DECISION_COLUMNS must include cost audit fields."""
        assert "cost_mult" in DECISION_COLUMNS
        assert "cost_bucket" in DECISION_COLUMNS
        assert "cost_haircut_applied" in DECISION_COLUMNS

    def test_compute_decision_fields_emits_cost_fields(self):
        """compute_decision_fields always emits cost fields."""
        rec = _minimal_rec()
        fields = compute_decision_fields(rec, "drug_developer", 0.50, DISABLED_RS)
        assert "cost_mult" in fields
        assert "cost_bucket" in fields
        assert "cost_haircut_applied" in fields

    def test_disabled_emits_no_haircut(self):
        """When disabled, cost_haircut_applied is '0' even with cost data."""
        rec = _minimal_rec()
        fields = compute_decision_fields(
            rec, "drug_developer", 0.50, DISABLED_RS, est_cost_bps=200.0
        )
        assert fields["cost_mult"] == 1.0
        assert fields["cost_bucket"] == ""
        assert fields["cost_haircut_applied"] == "0"

    def test_enabled_with_high_cost(self):
        """When enabled + high cost, fields reflect the haircut."""
        rec = _minimal_rec()
        fields = compute_decision_fields(
            rec, "drug_developer", 0.50, ENABLED_RS, est_cost_bps=1500.0
        )
        assert fields["cost_mult"] == 0.70
        assert fields["cost_bucket"] == "<=2000bps"
        assert fields["cost_haircut_applied"] == "1"
