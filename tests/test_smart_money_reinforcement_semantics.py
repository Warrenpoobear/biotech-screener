"""
Tests for smart money reinforcement semantics.

Verifies the invariants:
- reinforcement_applied == (catalyst_mult != 1 or momentum_mult != 1)
- If applied=True => flags contain reinforcement strength flag
- If applied=False => no reinforcement flags (except thesis gate block)
"""
from decimal import Decimal

from module_5_scoring_v3 import ScoringMode, compute_smart_money_reinforcement


def _is_applied(diag: dict) -> bool:
    return bool(diag.get("reinforcement_applied") is True)


def test_no_catalyst_returns_not_applied_and_no_reinforcement_flags():
    """When days_to_catalyst is None, reinforcement should not apply."""
    cat, mom, flags, diag = compute_smart_money_reinforcement(
        stage_bucket="poc",
        tier1_overlap=5,
        smart_money_confidence=Decimal("0.9"),
        smart_money_trend="stable",
        mode=ScoringMode.BAKER_STYLE,
        conviction_overlap=5.0,
        days_to_catalyst=None,
        thesis_gate_triggered=False,
    )
    assert cat == Decimal("1.00") and mom == Decimal("1.00")
    assert _is_applied(diag) is False
    assert diag.get("skip_reason") == "no_catalyst_data"
    assert not any(f.startswith("sm_reinforcement_") for f in flags)


def test_thesis_gate_block_emits_block_flag_but_not_applied():
    """Thesis gate block emits audit flag but reinforcement_applied=False."""
    cat, mom, flags, diag = compute_smart_money_reinforcement(
        stage_bucket="poc",
        tier1_overlap=5,
        smart_money_confidence=Decimal("0.9"),
        smart_money_trend="stable",
        mode=ScoringMode.BAKER_STYLE,
        conviction_overlap=8.0,
        days_to_catalyst=60,
        thesis_gate_triggered=True,
    )
    assert _is_applied(diag) is False
    assert "sm_reinforcement_blocked_by_thesis_gate" in flags
    # Block is an audit flag; ensure no reinforcement strength flags
    assert not any(
        f in ("sm_reinforcement_weak", "sm_reinforcement_moderate", "sm_reinforcement_strong")
        for f in flags
    )


def test_positive_reinforcement_sets_applied_true_and_emits_strength_flag():
    """When conviction and catalyst align, reinforcement should apply."""
    cat, mom, flags, diag = compute_smart_money_reinforcement(
        stage_bucket="poc",
        tier1_overlap=5,
        smart_money_confidence=Decimal("0.9"),
        smart_money_trend="stable",
        mode=ScoringMode.BAKER_STYLE,
        conviction_overlap=10.0,
        days_to_catalyst=90,  # Peak timing
        thesis_gate_triggered=False,
    )
    assert _is_applied(diag) is True
    assert (cat != Decimal("1.00")) or (mom != Decimal("1.00"))
    assert any(
        f in ("sm_reinforcement_weak", "sm_reinforcement_moderate", "sm_reinforcement_strong")
        for f in flags
    )


def test_negative_trend_always_applies_penalty():
    """Negative smart money trend always applies penalty multiplier."""
    cat, mom, flags, diag = compute_smart_money_reinforcement(
        stage_bucket="poc",
        tier1_overlap=5,
        smart_money_confidence=Decimal("0.9"),
        smart_money_trend="decreasing",
        mode=ScoringMode.BAKER_STYLE,
        conviction_overlap=10.0,
        days_to_catalyst=90,
        thesis_gate_triggered=False,
    )
    assert _is_applied(diag) is True
    assert "sm_reinforcement_negative" in flags


def test_reinforce_zero_branch_not_applied_when_timing_underflows():
    """Very large days_to_catalyst underflows timing to 0 => not applied."""
    # exp(-((10000 - 90)/90)^2) underflows to 0.0 => reinforce_zero
    cat, mom, flags, diag = compute_smart_money_reinforcement(
        stage_bucket="poc",
        tier1_overlap=5,
        smart_money_confidence=Decimal("0.9"),
        smart_money_trend="stable",
        mode=ScoringMode.BAKER_STYLE,
        conviction_overlap=10.0,
        days_to_catalyst=10_000,
        thesis_gate_triggered=False,
    )
    assert cat == Decimal("1.00") and mom == Decimal("1.00")
    assert _is_applied(diag) is False
    assert diag.get("skip_reason") == "reinforce_zero"
    assert not any(f.startswith("sm_reinforcement_") for f in flags)


def test_non_baker_mode_skips_reinforcement():
    """Non-BAKER_STYLE modes should skip reinforcement entirely."""
    from module_5_scoring_v3 import ScoringMode

    cat, mom, flags, diag = compute_smart_money_reinforcement(
        stage_bucket="poc",
        tier1_overlap=5,
        smart_money_confidence=Decimal("0.9"),
        smart_money_trend="stable",
        mode=ScoringMode.ENHANCED,  # Not BAKER_STYLE
        conviction_overlap=10.0,
        days_to_catalyst=90,
        thesis_gate_triggered=False,
    )
    assert cat == Decimal("1.00") and mom == Decimal("1.00")
    assert _is_applied(diag) is False
    assert diag.get("skip_reason") == "not_baker_style"


def test_inactive_stage_skips_reinforcement():
    """Commercial stage should skip reinforcement (not in active_stages)."""
    cat, mom, flags, diag = compute_smart_money_reinforcement(
        stage_bucket="commercial",
        tier1_overlap=5,
        smart_money_confidence=Decimal("0.9"),
        smart_money_trend="stable",
        mode=ScoringMode.BAKER_STYLE,
        conviction_overlap=10.0,
        days_to_catalyst=90,
        thesis_gate_triggered=False,
    )
    assert cat == Decimal("1.00") and mom == Decimal("1.00")
    assert _is_applied(diag) is False
    assert "stage_commercial_not_active" in diag.get("skip_reason", "")
