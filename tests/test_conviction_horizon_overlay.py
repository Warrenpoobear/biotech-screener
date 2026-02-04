from decimal import Decimal

import pytest

from module_5_scoring_v3 import (
    _apply_conviction_horizon_overlay,
    _conviction_horizon_weight_boost,
)


COHORT = {"mean": 0.0, "std": 1.0}


def test_days_none_no_boost():
    """days_to_cat is None → overlay never applies"""
    boost, diag = _conviction_horizon_weight_boost(
        weight_base=Decimal("0.0700"),
        weight_effective=Decimal("0.0300"),
        conviction_overlap=20.0,
        tier1_count=10,
        days_to_catalyst=None,
        cohort_conviction_stats=COHORT,
    )
    assert boost == Decimal("0")
    assert diag.get("skip_reason") == "no_catalyst_date"


@pytest.mark.parametrize("days, expect_positive", [
    (90, False),   # At boundary - blocked by days check
    (91, True),    # Just inside window
    (150, True),   # Peak of horizon curve
    (239, True),   # Just before horizon reaches 0
    (240, False),  # Horizon = 0 at this point
    (241, False),  # Outside window
])
def test_window_edges(days, expect_positive):
    """Test window boundaries: 91-239 valid, 90/240/241 blocked"""
    boost, diag = _conviction_horizon_weight_boost(
        weight_base=Decimal("0.0700"),
        weight_effective=Decimal("0.0300"),
        conviction_overlap=20.0,
        tier1_count=10,
        days_to_catalyst=days,
        cohort_conviction_stats=COHORT,
    )
    assert (boost > Decimal("0")) is expect_positive


def test_thesis_gate_hard_blocks_overlay():
    """thesis_gate_triggered=True → overlay never applies"""
    new_w, diag = _apply_conviction_horizon_overlay(
        weight_base=Decimal("0.0700"),
        weight_effective=Decimal("0.0300"),
        confidence_low=True,
        thesis_gate_blocked=True,
        conviction_overlap=20.0,
        tier1_count=10,
        days_to_catalyst=150,
        cohort_conviction_stats=COHORT,
    )
    assert new_w == Decimal("0.0300")
    assert diag and diag.get("skip_reason") == "thesis_gate_blocked"


def test_cap_total_at_90pct_base():
    """cap: weight_after <= 0.90 * weight_base always"""
    base = Decimal("0.0700")
    eff = Decimal("0.0625")  # already ~89.3% of base
    boost, diag = _conviction_horizon_weight_boost(
        weight_base=base,
        weight_effective=eff,
        conviction_overlap=20.0,
        tier1_count=10,
        days_to_catalyst=150,
        cohort_conviction_stats=COHORT,
    )
    assert eff + boost <= base * Decimal("0.90")


def test_confidence_not_low_skips():
    """If confidence is not low, overlay skips"""
    new_w, diag = _apply_conviction_horizon_overlay(
        weight_base=Decimal("0.0700"),
        weight_effective=Decimal("0.0500"),
        confidence_low=False,  # Not low
        thesis_gate_blocked=False,
        conviction_overlap=20.0,
        tier1_count=10,
        days_to_catalyst=150,
        cohort_conviction_stats=COHORT,
    )
    assert new_w == Decimal("0.0500")
    assert diag.get("skip_reason") == "confidence_not_low"


def test_insufficient_conviction_skips():
    """If tier1 < 6 and z-score < 1, overlay skips"""
    boost, diag = _conviction_horizon_weight_boost(
        weight_base=Decimal("0.0700"),
        weight_effective=Decimal("0.0300"),
        conviction_overlap=0.5,  # Low z-score
        tier1_count=3,           # Low tier1
        days_to_catalyst=150,
        cohort_conviction_stats=COHORT,
    )
    assert boost == Decimal("0")
    assert diag.get("skip_reason") == "insufficient_conviction"
