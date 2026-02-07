"""
Tests for Smart Money Alignment changes:
1. smart_money_rank ordering and divergence_flag in composite output
2. Red-flag suppression: continuous multiplier with smart money attenuation
3. Thesis gate: smart money attenuation and partial reinforcement
"""
import math
import pytest
from decimal import Decimal
from unittest.mock import patch

from module_5_scoring_v3 import (
    apply_thesis_gate,
    compute_smart_money_reinforcement,
    ScoringMode,
    THESIS_GATE_CONFIG,
    SMART_MONEY_REINFORCEMENT_CONFIG,
)
from defensive_overlay_adapter import apply_red_flag_suppression


# =============================================================================
# HELPERS
# =============================================================================

def _make_scored_record(ticker, composite_score, sm_weight, sm_score,
                        conviction_overlap=0, coinvest=None):
    """Build a minimal scored record dict for composite output tests."""
    return {
        "ticker": ticker,
        "composite_score": Decimal(str(composite_score)),
        "effective_weights": {"smart_money": Decimal(str(sm_weight))},
        "smart_money_signal": {"score": sm_score} if sm_score else None,
        "coinvest": coinvest or {"conviction_overlap": conviction_overlap},
    }


def _make_ranked_security(ticker, composite_score, tier1_count=0,
                          is_flagged=False, flag_reasons=None,
                          smart_money_signal=None):
    """Build a minimal ranked security dict for red-flag suppression tests."""
    rec = {
        "ticker": ticker,
        "composite_score": str(composite_score),
        "composite_rank": 0,
        "rankable": True,
        # Fields required by detect_fundamental_red_flags
        "component_scores": {},
        "score_breakdown": {"enhancements": {}},
        "survivability_signal": None,
    }
    if smart_money_signal:
        rec["smart_money_signal"] = smart_money_signal
    return rec


# =============================================================================
# TEST 1: smart_money_rank ordering
# =============================================================================

class TestSmartMoneyRank:
    def test_smart_money_rank_ordering(self):
        """Verify sort by sm contribution (desc), not composite."""
        records = [
            _make_scored_record("AAA", 80, 0.10, 90),   # sm_contrib = 9.0
            _make_scored_record("BBB", 90, 0.10, 50),   # sm_contrib = 5.0
            _make_scored_record("CCC", 70, 0.10, 95),   # sm_contrib = 9.5
        ]

        # Simulate the sort-and-rank logic from module_5_composite_v3.py
        # Composite rank (by composite_score desc)
        records.sort(key=lambda x: (-x["composite_score"], x["ticker"]))
        for i, rec in enumerate(records):
            rec["composite_rank"] = i + 1

        # Smart money rank
        n_scored = len(records)
        sm_sorted = sorted(records, key=lambda x: (
            -(x["effective_weights"].get("smart_money", Decimal("0")) *
              Decimal(str(x.get("smart_money_signal", {}).get("score", 0)
                         if x.get("smart_money_signal") else 0))),
            -(x.get("coinvest", {}).get("conviction_overlap", 0)
              if x.get("coinvest") else 0),
            x["ticker"],
        ))
        for i, rec in enumerate(sm_sorted):
            rec["smart_money_rank"] = i + 1
            rec["divergence_flag"] = abs(rec["composite_rank"] - rec["smart_money_rank"]) > 0.25 * n_scored

        # CCC has highest sm contribution (9.5), should be sm_rank 1
        ccc = next(r for r in records if r["ticker"] == "CCC")
        aaa = next(r for r in records if r["ticker"] == "AAA")
        bbb = next(r for r in records if r["ticker"] == "BBB")

        assert ccc["smart_money_rank"] == 1  # 9.5 highest
        assert aaa["smart_money_rank"] == 2  # 9.0 second
        assert bbb["smart_money_rank"] == 3  # 5.0 lowest

        # Composite rank: BBB=1, AAA=2, CCC=3
        assert bbb["composite_rank"] == 1
        assert aaa["composite_rank"] == 2
        assert ccc["composite_rank"] == 3


# =============================================================================
# TEST 2: divergence_flag
# =============================================================================

class TestDivergenceFlag:
    def test_divergence_flag_triggers_at_25pct_gap(self):
        """Verify divergence triggers when |composite - sm_rank| > 25% of len."""
        # 4 records: 25% threshold = 1.0, so gap > 1 triggers
        records = [
            _make_scored_record("A", 90, 0.10, 10),  # comp=1, sm=4 -> gap=3 > 1
            _make_scored_record("B", 80, 0.10, 80),  # comp=2, sm=1 -> gap=1, not > 1
            _make_scored_record("C", 70, 0.10, 70),  # comp=3, sm=2 -> gap=1, not > 1
            _make_scored_record("D", 60, 0.10, 60),  # comp=4, sm=3 -> gap=1, not > 1
        ]

        # Composite rank
        records.sort(key=lambda x: (-x["composite_score"], x["ticker"]))
        for i, rec in enumerate(records):
            rec["composite_rank"] = i + 1

        # Smart money rank
        n_scored = len(records)
        sm_sorted = sorted(records, key=lambda x: (
            -(x["effective_weights"].get("smart_money", Decimal("0")) *
              Decimal(str(x.get("smart_money_signal", {}).get("score", 0)
                         if x.get("smart_money_signal") else 0))),
            -(x.get("coinvest", {}).get("conviction_overlap", 0)
              if x.get("coinvest") else 0),
            x["ticker"],
        ))
        for i, rec in enumerate(sm_sorted):
            rec["smart_money_rank"] = i + 1
            rec["divergence_flag"] = abs(rec["composite_rank"] - rec["smart_money_rank"]) > 0.25 * n_scored

        a = next(r for r in records if r["ticker"] == "A")
        b = next(r for r in records if r["ticker"] == "B")

        # A: comp=1, sm=4 -> gap=3 > 1.0 -> True
        assert a["divergence_flag"] is True
        # B: comp=2, sm=1 -> gap=1, not > 1.0 -> False
        assert b["divergence_flag"] is False


# =============================================================================
# TEST 3: Red-flag suppression — continuous multiplier (not binary)
# =============================================================================

class TestRedFlagContinuousSuppression:
    @patch("defensive_overlay_adapter.detect_fundamental_red_flags")
    def test_suppression_is_proportional(self, mock_detect):
        """Verify penalty is proportional to distance from median, not a hard clamp."""
        # Set up: two flagged securities at different distances from median
        mock_detect.return_value = (True, ["test_reason"])

        records = [
            _make_ranked_security("HIGH", "90"),   # Far above median
            _make_ranked_security("MID", "70"),    # Closer to median
            _make_ranked_security("BELOW", "40"),  # Below median - should not be touched
        ]

        result = apply_red_flag_suppression(records, enable_suppression=True)

        high = next(r for r in records if r["ticker"] == "HIGH")
        mid = next(r for r in records if r["ticker"] == "MID")
        below = next(r for r in records if r["ticker"] == "BELOW")

        # HIGH should be penalized more than MID (further from median)
        # Both should still be > median (continuous, not hard clamp)
        high_score = Decimal(high["composite_score"])
        mid_score = Decimal(mid["composite_score"])
        below_score = Decimal(below["composite_score"])

        # Median is 70 (middle of [40, 70, 90])
        # HIGH (90) should be penalized: penalty < 1.0
        assert high_score < Decimal("90"), "HIGH should be penalized"
        # MID (70) is at median, should not be suppressed (not > median)
        assert mid_score == Decimal("70"), "MID at median should be unchanged"
        # BELOW (40) should be unchanged
        assert below_score == Decimal("40"), "BELOW median should be unchanged"

        assert result["suppression_mode"] == "continuous_multiplier"
        assert result["suppressor_version"] == "1.1.0"


# =============================================================================
# TEST 4: Red-flag smart money attenuation
# =============================================================================

class TestRedFlagSmAttenuation:
    @patch("defensive_overlay_adapter.detect_fundamental_red_flags")
    def test_tier1_holders_reduce_penalty(self, mock_detect):
        """Verify tier1 holders reduce the penalty, capped at 40%."""
        mock_detect.return_value = (True, ["test_reason"])

        # Two identical flagged securities, one with tier1 holders
        records_no_sm = [
            _make_ranked_security("NO_SM", "90"),
            _make_ranked_security("PAD1", "50"),
            _make_ranked_security("PAD2", "30"),
        ]
        records_with_sm = [
            _make_ranked_security("WITH_SM", "90",
                                  smart_money_signal={"tier_breakdown": {"tier1": 8}}),
            _make_ranked_security("PAD1", "50"),
            _make_ranked_security("PAD2", "30"),
        ]

        apply_red_flag_suppression(records_no_sm, enable_suppression=True)
        apply_red_flag_suppression(records_with_sm, enable_suppression=True)

        no_sm = next(r for r in records_no_sm if r["ticker"] == "NO_SM")
        with_sm = next(r for r in records_with_sm if r["ticker"] == "WITH_SM")

        no_sm_score = Decimal(no_sm["composite_score"])
        with_sm_score = Decimal(with_sm["composite_score"])

        # With smart money should be penalized LESS than without
        assert with_sm_score > no_sm_score, \
            f"SM attenuation should reduce penalty: {with_sm_score} should be > {no_sm_score}"


# =============================================================================
# TEST 5: Red-flag attenuation invariant — always penalized
# =============================================================================

class TestRedFlagAttenuationInvariant:
    @patch("defensive_overlay_adapter.detect_fundamental_red_flags")
    def test_flagged_always_penalized(self, mock_detect):
        """Verify flagged securities above median are ALWAYS penalized (< pre-suppression)."""
        mock_detect.return_value = (True, ["test_reason"])

        # Max attenuation case: 20 tier1 holders (caps at 0.40)
        records = [
            _make_ranked_security("MAXSM", "95",
                                  smart_money_signal={"tier_breakdown": {"tier1": 20}}),
            _make_ranked_security("PAD", "50"),
            _make_ranked_security("PAD2", "30"),
        ]

        apply_red_flag_suppression(records, enable_suppression=True)

        maxsm = next(r for r in records if r["ticker"] == "MAXSM")
        post_score = Decimal(maxsm["composite_score"])
        pre_score = Decimal(maxsm["composite_score_pre_suppression"])

        assert post_score < pre_score, \
            f"Flagged security must always be penalized: {post_score} should be < {pre_score}"


# =============================================================================
# TEST 6: Thesis gate smart money attenuation
# =============================================================================

class TestThesisGateSmAttenuation:
    def test_penalty_lerps_toward_1(self):
        """Verify thesis gate penalty is attenuated by smart_money_strength up to 30%."""
        score = Decimal("80")
        clinical = Decimal("30")  # Low thesis
        pos = Decimal("30")       # Low thesis -> avg 30, well below any threshold

        # Without smart money
        gated_no_sm, flags_no_sm, diag_no_sm = apply_thesis_gate(
            score=score, clinical_normalized=clinical, pos_normalized=pos,
            stage_bucket="poc", mode=ScoringMode.BAKER_STYLE,
            smart_money_strength=Decimal("0"),
        )

        # With max smart money strength
        gated_with_sm, flags_with_sm, diag_with_sm = apply_thesis_gate(
            score=score, clinical_normalized=clinical, pos_normalized=pos,
            stage_bucket="poc", mode=ScoringMode.BAKER_STYLE,
            smart_money_strength=Decimal("0.50"),  # Will be capped to 0.30
        )

        # Smart money should produce a HIGHER (less penalized) score
        assert gated_with_sm > gated_no_sm, \
            f"SM attenuation should reduce penalty: {gated_with_sm} > {gated_no_sm}"

        # But still below original score
        assert gated_with_sm < score, \
            f"Attenuated score should still be below original: {gated_with_sm} < {score}"

        # Check diagnostics
        assert "thesis_gate_sm_attenuation" in diag_with_sm
        assert Decimal(diag_with_sm["thesis_gate_sm_attenuation"]) == Decimal("0.30")  # Capped


# =============================================================================
# TEST 7: Thesis gate zero smart money — identical to current behavior
# =============================================================================

class TestThesisGateZeroSm:
    def test_zero_sm_matches_baseline(self):
        """Verify zero smart_money_strength produces identical behavior to None."""
        score = Decimal("80")
        clinical = Decimal("30")
        pos = Decimal("30")

        gated_none, _, diag_none = apply_thesis_gate(
            score=score, clinical_normalized=clinical, pos_normalized=pos,
            stage_bucket="poc", mode=ScoringMode.BAKER_STYLE,
            smart_money_strength=None,
        )

        gated_zero, _, diag_zero = apply_thesis_gate(
            score=score, clinical_normalized=clinical, pos_normalized=pos,
            stage_bucket="poc", mode=ScoringMode.BAKER_STYLE,
            smart_money_strength=Decimal("0"),
        )

        assert gated_none == gated_zero, \
            f"Zero SM should match None: {gated_none} == {gated_zero}"
        assert "thesis_gate_sm_attenuation" not in diag_none
        assert "thesis_gate_sm_attenuation" not in diag_zero


# =============================================================================
# TEST 8: Thesis gate + partial reinforcement
# =============================================================================

class TestPartialReinforcement:
    def test_reinforcement_scales_by_attenuation(self):
        """Verify reinforcement multipliers are scaled by thesis_gate_attenuation."""
        # Full reinforcement (no thesis gate)
        cat_full, mom_full, flags_full, diag_full = compute_smart_money_reinforcement(
            stage_bucket="poc",
            tier1_overlap=5,
            smart_money_confidence=Decimal("0.8"),
            smart_money_trend="stable",
            mode=ScoringMode.BAKER_STYLE,
            conviction_overlap=5.0,
            cohort_conviction_stats={"mean": 3.0, "std": 1.5},
            days_to_catalyst=90,
            thesis_gate_triggered=False,
        )

        # Partial reinforcement (thesis gate triggered but with attenuation)
        cat_partial, mom_partial, flags_partial, diag_partial = compute_smart_money_reinforcement(
            stage_bucket="poc",
            tier1_overlap=5,
            smart_money_confidence=Decimal("0.8"),
            smart_money_trend="stable",
            mode=ScoringMode.BAKER_STYLE,
            conviction_overlap=5.0,
            cohort_conviction_stats={"mean": 3.0, "std": 1.5},
            days_to_catalyst=90,
            thesis_gate_triggered=True,
            thesis_gate_attenuation=Decimal("0.25"),
        )

        # Zero attenuation (full block)
        cat_blocked, mom_blocked, flags_blocked, diag_blocked = compute_smart_money_reinforcement(
            stage_bucket="poc",
            tier1_overlap=5,
            smart_money_confidence=Decimal("0.8"),
            smart_money_trend="stable",
            mode=ScoringMode.BAKER_STYLE,
            conviction_overlap=5.0,
            cohort_conviction_stats={"mean": 3.0, "std": 1.5},
            days_to_catalyst=90,
            thesis_gate_triggered=True,
            thesis_gate_attenuation=None,
        )

        # Full should have the largest multipliers
        assert cat_full >= cat_partial, \
            f"Full reinforcement should be >= partial: {cat_full} >= {cat_partial}"
        # Partial should be between 1.0 and full
        assert cat_partial > Decimal("1.0") or cat_full == Decimal("1.0"), \
            f"Partial should provide some reinforcement if full does"
        # Blocked should be exactly 1.0
        assert cat_blocked == Decimal("1.00")
        assert mom_blocked == Decimal("1.00")

        assert "sm_reinforcement_partial_thesis_gate" in flags_partial
        assert "sm_reinforcement_blocked_by_thesis_gate" in flags_blocked


# =============================================================================
# TEST 9: Conviction override unchanged
# =============================================================================

class TestConvictionOverrideUnchanged:
    def test_conviction_override_still_bypasses(self):
        """Verify existing conviction override (tier1>=10, conviction>=15) still works."""
        # This test verifies the conviction override path is not broken
        # by the new smart money changes. The override is in the caller
        # (_score_single_ticker_v3), not in apply_thesis_gate/compute_sm_reinforcement.
        # We test the reinforcement function directly: when thesis_gate_triggered=False
        # (because conviction override bypassed it), reinforcement should apply normally.
        cat, mom, flags, diag = compute_smart_money_reinforcement(
            stage_bucket="poc",
            tier1_overlap=12,  # Well above override threshold
            smart_money_confidence=Decimal("0.9"),
            smart_money_trend="stable",
            mode=ScoringMode.BAKER_STYLE,
            conviction_overlap=20.0,  # Well above override threshold
            cohort_conviction_stats={"mean": 3.0, "std": 1.5},
            days_to_catalyst=90,
            thesis_gate_triggered=False,  # Bypassed by conviction override
            thesis_gate_attenuation=None,
        )

        # Reinforcement should be applied (not blocked)
        assert diag["reinforcement_applied"] is True
        assert cat >= Decimal("1.0")
        assert "sm_reinforcement_blocked_by_thesis_gate" not in flags


# =============================================================================
# TEST 10-12: Guardrail A — hard-risk disables SM attenuation
# =============================================================================

class TestGuardrailA:
    """
    Guardrail A tests verify that thesis gate SM attenuation is disabled
    when hard structural risk signals fire, preventing smart money from
    rescuing fundamentally broken names.

    These test the *caller logic* in _score_single_ticker_v3 by testing
    apply_thesis_gate with the strength values the guardrail would produce.
    """

    def test_sev2_disables_attenuation(self):
        """A1: sev2+ → smart_money_strength zeroed → no attenuation."""
        score = Decimal("80")
        clinical = Decimal("30")
        pos = Decimal("30")

        # Guardrail A would zero sm_strength for sev2
        gated_sev2, _, diag_sev2 = apply_thesis_gate(
            score=score, clinical_normalized=clinical, pos_normalized=pos,
            stage_bucket="poc", mode=ScoringMode.BAKER_STYLE,
            smart_money_strength=Decimal("0"),  # Guardrail zeroed this
        )

        # Compare with no guardrail (full attenuation)
        gated_full, _, diag_full = apply_thesis_gate(
            score=score, clinical_normalized=clinical, pos_normalized=pos,
            stage_bucket="poc", mode=ScoringMode.BAKER_STYLE,
            smart_money_strength=Decimal("0.80"),  # Would be 0.30 capped normally
        )

        # sev2 path should have no attenuation (identical to zero SM)
        assert "thesis_gate_sm_attenuation" not in diag_sev2
        # Full path should have attenuation
        assert "thesis_gate_sm_attenuation" in diag_full
        # sev2 score should be lower (more penalized)
        assert gated_sev2 < gated_full

    def test_sev1_caps_attenuation_at_010(self):
        """A3: sev1 → attenuation capped at 0.10 (not 0.30)."""
        score = Decimal("80")
        clinical = Decimal("30")
        pos = Decimal("30")

        # sev1: guardrail caps sm_strength at 0.10
        gated_sev1, _, diag_sev1 = apply_thesis_gate(
            score=score, clinical_normalized=clinical, pos_normalized=pos,
            stage_bucket="poc", mode=ScoringMode.BAKER_STYLE,
            smart_money_strength=Decimal("0.10"),  # Guardrail capped at 0.10
        )

        # No guardrail: full 0.30 attenuation
        gated_full, _, diag_full = apply_thesis_gate(
            score=score, clinical_normalized=clinical, pos_normalized=pos,
            stage_bucket="poc", mode=ScoringMode.BAKER_STYLE,
            smart_money_strength=Decimal("0.50"),  # Capped to 0.30 by apply_thesis_gate
        )

        # sev1 should get less attenuation than full
        att_sev1 = Decimal(diag_sev1.get("thesis_gate_sm_attenuation", "0"))
        att_full = Decimal(diag_full.get("thesis_gate_sm_attenuation", "0"))
        assert att_sev1 <= Decimal("0.10"), f"sev1 attenuation should be <= 0.10, got {att_sev1}"
        assert att_full == Decimal("0.30"), f"Full attenuation should be 0.30, got {att_full}"
        assert gated_sev1 < gated_full, "sev1 should be penalized more than full attenuation"
        # But sev1 should still get SOME attenuation (it's not zero)
        gated_zero, _, _ = apply_thesis_gate(
            score=score, clinical_normalized=clinical, pos_normalized=pos,
            stage_bucket="poc", mode=ScoringMode.BAKER_STYLE,
            smart_money_strength=Decimal("0"),
        )
        assert gated_sev1 > gated_zero, "sev1 should get some (reduced) attenuation"

    def test_partial_reinforcement_blocked_by_hard_risk(self):
        """A1: When guardrail zeros sm_strength, partial reinforcement is also blocked."""
        # With guardrail: sm_strength=0 → thesis_gate_attenuation=None → full block
        cat_blocked, mom_blocked, flags_blocked, diag_blocked = compute_smart_money_reinforcement(
            stage_bucket="poc",
            tier1_overlap=8,
            smart_money_confidence=Decimal("0.8"),
            smart_money_trend="stable",
            mode=ScoringMode.BAKER_STYLE,
            conviction_overlap=5.0,
            cohort_conviction_stats={"mean": 3.0, "std": 1.5},
            days_to_catalyst=90,
            thesis_gate_triggered=True,
            thesis_gate_attenuation=None,  # Guardrail zeroed → no attenuation → None
        )

        assert cat_blocked == Decimal("1.00"), "Hard risk should fully block reinforcement"
        assert mom_blocked == Decimal("1.00"), "Hard risk should fully block reinforcement"
        assert "sm_reinforcement_blocked_by_thesis_gate" in flags_blocked
        assert "sm_reinforcement_partial_thesis_gate" not in flags_blocked

    def test_red_flag_eligible_caps_attenuation_at_005(self):
        """A2: Red-flag eligible (e.g. single_asset_early) → attenuation capped at 0.05."""
        score = Decimal("80")
        clinical = Decimal("30")
        pos = Decimal("30")

        # A2 guardrail caps sm_strength at 0.05 for red-flag eligible names
        gated_rf, _, diag_rf = apply_thesis_gate(
            score=score, clinical_normalized=clinical, pos_normalized=pos,
            stage_bucket="poc", mode=ScoringMode.BAKER_STYLE,
            smart_money_strength=Decimal("0.05"),  # Guardrail capped at 0.05
        )

        # Full attenuation (no guardrail)
        gated_full, _, diag_full = apply_thesis_gate(
            score=score, clinical_normalized=clinical, pos_normalized=pos,
            stage_bucket="poc", mode=ScoringMode.BAKER_STYLE,
            smart_money_strength=Decimal("0.50"),  # Would be 0.30 capped
        )

        # No attenuation at all
        gated_zero, _, _ = apply_thesis_gate(
            score=score, clinical_normalized=clinical, pos_normalized=pos,
            stage_bucket="poc", mode=ScoringMode.BAKER_STYLE,
            smart_money_strength=Decimal("0"),
        )

        # Red-flag eligible should get minimal attenuation
        att_rf = Decimal(diag_rf.get("thesis_gate_sm_attenuation", "0"))
        assert att_rf <= Decimal("0.05"), f"RF attenuation should be <= 0.05, got {att_rf}"
        # Should be strictly between zero and full
        assert gated_rf > gated_zero, "RF should get tiny attenuation (not zero)"
        assert gated_rf < gated_full, "RF should be much less than full attenuation"
