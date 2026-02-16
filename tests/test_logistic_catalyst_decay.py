"""Tests for logistic catalyst decay and smooth gating."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from decision_engine import (
    DecisionRuleset,
    DEFAULT_RULESET,
    _logistic_decay,
    compute_decision_fields,
)


# =============================================================================
# _logistic_decay unit tests
# =============================================================================

def test_logistic_decay_monotone():
    """Decay weight should decrease as days increase."""
    w5 = _logistic_decay(5, 150, 30)
    w50 = _logistic_decay(50, 150, 30)
    w150 = _logistic_decay(150, 150, 30)
    w300 = _logistic_decay(300, 150, 30)
    assert w5 > w50 > w150 > w300


def test_logistic_midpoint_half():
    """At the midpoint, decay should be exactly 0.5."""
    w = _logistic_decay(150, 150, 30)
    assert abs(w - 0.5) < 0.001


def test_logistic_none_returns_zero():
    assert _logistic_decay(None, 150, 30) == 0.0


def test_logistic_negative_returns_zero():
    assert _logistic_decay(-10, 150, 30) == 0.0


def test_logistic_zero_days():
    """Day 0 should give a high weight (near 1.0)."""
    w = _logistic_decay(0, 150, 30)
    assert w > 0.99


def test_logistic_large_days():
    """Very far days should give near-zero weight."""
    w = _logistic_decay(1000, 150, 30)
    assert w < 0.01


# =============================================================================
# Integration: hard mode unchanged
# =============================================================================

def _base_rec(**overrides):
    """Build a minimal rec dict with sane defaults."""
    rec = {
        "ticker": "TEST",
        "composite_score": 50.0,
        "severity": "NONE",
        "fundamental_red_flag": False,
        "confidence_overall": 0.65,
        "defensive_features": {
            "vol_60d": 0.80,
            "beta_xbi_60d": 1.20,
            "drawdown": -0.15,
            "rsi_14d": 50.0,
        },
        "smart_money_signal": {
            "tier1_holders": ["FundA", "FundB"],
            "holders_increasing": ["FundA"],
            "holders_decreasing": [],
            "overlap_count": 2,
        },
        "coinvest": {"tier1_count": 3},
        "catalyst_decay": {
            "days_to_catalyst": 45,
            "in_optimal_window": True,
        },
        "score_breakdown": {
            "enhancements": {
                "momentum": {"alpha_60d": 0.08},
            }
        },
    }
    rec.update(overrides)
    return rec


def test_hard_mode_default_unchanged():
    """DEFAULT_RULESET (hard mode) should produce same tier as before."""
    rec = _base_rec()
    result = compute_decision_fields(rec, "drug_developer", 0.75)
    # 45 days < 90 = near, optionality 0.75 >= 0.60 → A tier
    assert result["tier_dev"] == "A"
    assert result["catalyst_strength"] == "near"
    assert DEFAULT_RULESET.catalyst_time_decay_mode == "hard"


def test_hard_mode_decay_w_values():
    """Hard mode should emit discrete decay_w values."""
    # near → 1.0
    rec_near = _base_rec(catalyst_decay={"days_to_catalyst": 45, "in_optimal_window": True})
    r = compute_decision_fields(rec_near, "drug_developer", 0.75)
    assert r["catalyst_decay_w"] == 1.0

    # mid → 0.7
    rec_mid = _base_rec(catalyst_decay={"days_to_catalyst": 120, "in_optimal_window": True})
    r = compute_decision_fields(rec_mid, "drug_developer", 0.75)
    assert r["catalyst_decay_w"] == 0.7

    # far → 0.3
    rec_far = _base_rec(catalyst_decay={"days_to_catalyst": 250, "in_optimal_window": True})
    r = compute_decision_fields(rec_far, "drug_developer", 0.75)
    assert r["catalyst_decay_w"] == 0.3

    # missing → 0.0
    rec_miss = _base_rec(catalyst_decay={})
    r = compute_decision_fields(rec_miss, "drug_developer", 0.75)
    assert r["catalyst_decay_w"] == 0.0


# =============================================================================
# Integration: logistic mode smooth boundary
# =============================================================================

def test_logistic_mode_smooth_boundary():
    """In hard mode day 89 → near, day 91 → mid. In logistic mode both should be 'near'
    because both are well above the 0.5 threshold (midpoint=150)."""
    rs_hard = DecisionRuleset(catalyst_near_days=90, catalyst_mid_days=180)
    rs_logistic = DecisionRuleset(
        catalyst_time_decay_mode="logistic",
        catalyst_logistic_midpoint_days=150,
        catalyst_logistic_scale_days=30.0,
    )

    rec_89 = _base_rec(catalyst_decay={"days_to_catalyst": 89, "in_optimal_window": True})
    rec_91 = _base_rec(catalyst_decay={"days_to_catalyst": 91, "in_optimal_window": True})

    # Hard mode: cliff at 90
    h89 = compute_decision_fields(rec_89, "drug_developer", 0.75, ruleset=rs_hard)
    h91 = compute_decision_fields(rec_91, "drug_developer", 0.75, ruleset=rs_hard)
    assert h89["catalyst_strength"] == "near"
    assert h91["catalyst_strength"] == "mid"

    # Logistic mode: both are near (decay_w > 0.5 for both 89 and 91)
    l89 = compute_decision_fields(rec_89, "drug_developer", 0.75, ruleset=rs_logistic)
    l91 = compute_decision_fields(rec_91, "drug_developer", 0.75, ruleset=rs_logistic)
    assert l89["catalyst_strength"] == "near"
    assert l91["catalyst_strength"] == "near"
    # Same tier
    assert l89["tier_dev"] == l91["tier_dev"] == "A"


def test_logistic_far_boundary():
    """Day 200 with midpoint=150, scale=30 → decay_w < 0.2 → 'far'."""
    rs = DecisionRuleset(
        catalyst_time_decay_mode="logistic",
        catalyst_logistic_midpoint_days=150,
        catalyst_logistic_scale_days=30.0,
    )
    rec = _base_rec(catalyst_decay={"days_to_catalyst": 200, "in_optimal_window": True})
    result = compute_decision_fields(rec, "drug_developer", 0.75, ruleset=rs)
    assert result["catalyst_strength"] == "far"
    assert result["catalyst_decay_w"] < 0.2


def test_logistic_mid_boundary():
    """Day 160 with midpoint=150, scale=30 → decay_w between 0.2 and 0.5 → 'mid'."""
    rs = DecisionRuleset(
        catalyst_time_decay_mode="logistic",
        catalyst_logistic_midpoint_days=150,
        catalyst_logistic_scale_days=30.0,
    )
    rec = _base_rec(catalyst_decay={"days_to_catalyst": 160, "in_optimal_window": True})
    result = compute_decision_fields(rec, "drug_developer", 0.75, ruleset=rs)
    # decay_w at 160: 1/(1+exp((160-150)/30)) ≈ 0.42
    assert result["catalyst_strength"] == "mid"
    assert 0.2 <= result["catalyst_decay_w"] < 0.5


def test_logistic_catalyst_tilt_continuous():
    """With catalyst tilt + logistic mode, tilt should be continuous."""
    rs = DecisionRuleset(
        catalyst_time_decay_mode="logistic",
        catalyst_logistic_midpoint_days=150,
        catalyst_logistic_scale_days=30.0,
        enable_catalyst_tilt=True,
    )
    rec_close = _base_rec(catalyst_decay={"days_to_catalyst": 30, "in_optimal_window": True})
    rec_far = _base_rec(catalyst_decay={"days_to_catalyst": 250, "in_optimal_window": True})

    r_close = compute_decision_fields(rec_close, "drug_developer", 0.75, ruleset=rs)
    r_far = compute_decision_fields(rec_far, "drug_developer", 0.75, ruleset=rs)

    # Close should have higher tilt than far
    assert r_close["catalyst_tilt_mult"] > r_far["catalyst_tilt_mult"]
    # Verify range: 0.90 to 1.10
    assert 0.90 <= r_close["catalyst_tilt_mult"] <= 1.10
    assert 0.90 <= r_far["catalyst_tilt_mult"] <= 1.10


def test_logistic_blended_window():
    """Blended window mode should always be decay_w=1.0 in logistic mode."""
    rs = DecisionRuleset(catalyst_time_decay_mode="logistic")
    rec = _base_rec(catalyst_decay={"days_to_catalyst": 0, "in_optimal_window": True})
    result = compute_decision_fields(rec, "drug_developer", 0.75, ruleset=rs)
    assert result["catalyst_decay_w"] == 1.0
    assert result["catalyst_strength"] == "near"


# =============================================================================
# Validation
# =============================================================================

def test_invalid_decay_mode():
    with pytest.raises(ValueError, match="catalyst_time_decay_mode"):
        DecisionRuleset(catalyst_time_decay_mode="invalid")


def test_invalid_scale_zero():
    with pytest.raises(ValueError, match="catalyst_logistic_scale_days"):
        DecisionRuleset(catalyst_logistic_scale_days=0)


def test_invalid_scale_negative():
    with pytest.raises(ValueError, match="catalyst_logistic_scale_days"):
        DecisionRuleset(catalyst_logistic_scale_days=-5)
