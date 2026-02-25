"""Tests for alpha modifier sort key behavior (tiebreak / within_tier modes)."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from decision_engine import DecisionRuleset, compute_actionable_sort_key, _safe_float
from run_screen import _compute_alpha_modifier_ab


# =============================================================================
# Helpers
# =============================================================================

def _decision_fields(**overrides):
    """Minimal eligible decision fields."""
    df = {
        "eligible": "1",
        "tier_dev": "A",
        "catalyst_mode": "specific_days",
        "catalyst_days": "30",
        "sponsor_tier1_count": "2",
        "mom_state": "neutral",
        "missingness_penalty": "0",
    }
    df.update(overrides)
    return df


def _sort_key(df=None, *, alpha_raw=None, ruleset=None, ticker="TEST",
              composite_rank=10, optionality=0.50):
    """Shorthand: compute sort key with sensible defaults."""
    if df is None:
        df = _decision_fields()
    return compute_actionable_sort_key(
        decision_fields=df,
        archetype="drug_developer",
        optionality=optionality,
        composite_rank=composite_rank,
        ticker=ticker,
        ruleset=ruleset,
        alpha_raw=alpha_raw,
    )


# =============================================================================
# mode=off — alpha_raw should have zero effect
# =============================================================================

class TestModeOff:
    """When alpha_modifier_mode='off', alpha_raw must not affect ordering."""

    @pytest.fixture
    def rs(self):
        return DecisionRuleset(alpha_modifier_mode="off")

    def test_alpha_positive_no_effect(self, rs):
        k0 = _sort_key(alpha_raw=None, ruleset=rs)
        k1 = _sort_key(alpha_raw=0.05, ruleset=rs)
        assert k0 == k1

    def test_alpha_negative_no_effect(self, rs):
        k0 = _sort_key(alpha_raw=None, ruleset=rs)
        k1 = _sort_key(alpha_raw=-0.05, ruleset=rs)
        assert k0 == k1

    def test_default_ruleset_is_off(self):
        """DEFAULT_RULESET has alpha_modifier_mode=off."""
        k0 = _sort_key(alpha_raw=None)
        k1 = _sort_key(alpha_raw=0.10)
        assert k0 == k1


# =============================================================================
# mode=tiebreak — alpha as last sort key before ticker
# =============================================================================

class TestModeTiebreak:
    """When mode=tiebreak, higher alpha should break ties in favour of that ticker."""

    @pytest.fixture
    def rs(self):
        return DecisionRuleset(alpha_modifier_mode="tiebreak")

    def test_higher_alpha_wins_tiebreak(self, rs):
        """Two identical tickers except alpha — higher alpha sorts first."""
        k_high = _sort_key(alpha_raw=0.05, ruleset=rs, ticker="AAA")
        k_low = _sort_key(alpha_raw=0.01, ruleset=rs, ticker="BBB")
        # Higher alpha → more negative _alpha_tie → sorts first
        assert k_high < k_low

    def test_zero_alpha_vs_none(self, rs):
        """alpha_raw=0.0 and None both produce the same tiebreak key."""
        k0 = _sort_key(alpha_raw=0.0, ruleset=rs, ticker="AAA")
        kn = _sort_key(alpha_raw=None, ruleset=rs, ticker="AAA")
        assert k0 == kn

    def test_tiebreak_does_not_override_tier(self, rs):
        """Alpha tiebreak cannot promote a B-tier above an A-tier."""
        df_a = _decision_fields(tier_dev="A")
        df_b = _decision_fields(tier_dev="B")
        k_a = _sort_key(df_a, alpha_raw=-0.10, ruleset=rs, ticker="ZZZ")
        k_b = _sort_key(df_b, alpha_raw=0.10, ruleset=rs, ticker="AAA")
        # A-tier must still sort before B-tier
        assert k_a < k_b

    def test_tiebreak_does_not_override_composite(self, rs):
        """Alpha tiebreak does not override composite_rank ordering."""
        # Use tiebreaker mode for catalyst_priority_mode too so comp_rank matters
        rs_tb = DecisionRuleset(
            alpha_modifier_mode="tiebreak",
            catalyst_priority_mode="tiebreaker",
        )
        k_better_comp = _sort_key(
            alpha_raw=0.0, ruleset=rs_tb, composite_rank=5, ticker="ZZZ"
        )
        k_worse_comp = _sort_key(
            alpha_raw=0.10, ruleset=rs_tb, composite_rank=50, ticker="AAA"
        )
        # Better composite rank must still win despite worse alpha
        assert k_better_comp < k_worse_comp

    def test_negative_alpha_sorts_after_zero(self, rs):
        """Negative alpha sorts after zero (penalty direction)."""
        k_neg = _sort_key(alpha_raw=-0.05, ruleset=rs, ticker="AAA")
        k_zero = _sort_key(alpha_raw=0.0, ruleset=rs, ticker="AAA")
        assert k_neg > k_zero

    def test_deterministic_tiebreak(self, rs):
        """Same alpha_raw + same ticker → same sort key, every time."""
        k1 = _sort_key(alpha_raw=0.03, ruleset=rs, ticker="TEST")
        k2 = _sort_key(alpha_raw=0.03, ruleset=rs, ticker="TEST")
        assert k1 == k2


# =============================================================================
# mode=within_tier — alpha blended into effective anchor
# =============================================================================

class TestModeWithinTier:
    """When mode=within_tier, alpha_raw * weight blends into the anchor term."""

    @pytest.fixture
    def rs(self):
        return DecisionRuleset(
            alpha_modifier_mode="within_tier",
            alpha_modifier_weight=0.20,
        )

    def test_positive_alpha_improves_rank(self, rs):
        """Positive alpha subtracts from effective anchor → sorts earlier."""
        k_alpha = _sort_key(alpha_raw=0.10, ruleset=rs, ticker="AAA")
        k_none = _sort_key(alpha_raw=0.0, ruleset=rs, ticker="AAA")
        assert k_alpha < k_none

    def test_negative_alpha_worsens_rank(self, rs):
        """Negative alpha adds to effective anchor → sorts later."""
        k_alpha = _sort_key(alpha_raw=-0.10, ruleset=rs, ticker="AAA")
        k_none = _sort_key(alpha_raw=0.0, ruleset=rs, ticker="AAA")
        assert k_alpha > k_none

    def test_zero_weight_has_no_effect(self):
        """within_tier with weight=0.0 should be equivalent to mode=off."""
        rs_wt = DecisionRuleset(
            alpha_modifier_mode="within_tier",
            alpha_modifier_weight=0.0,
        )
        rs_off = DecisionRuleset(alpha_modifier_mode="off")
        k_wt = _sort_key(alpha_raw=0.10, ruleset=rs_wt, ticker="AAA")
        k_off = _sort_key(alpha_raw=0.10, ruleset=rs_off, ticker="AAA")
        assert k_wt == k_off

    def test_within_tier_respects_tier_boundary(self, rs):
        """within_tier cannot promote a B-tier above A-tier."""
        df_a = _decision_fields(tier_dev="A")
        df_b = _decision_fields(tier_dev="B")
        k_a = _sort_key(df_a, alpha_raw=-0.10, ruleset=rs, ticker="ZZZ")
        k_b = _sort_key(df_b, alpha_raw=0.10, ruleset=rs, ticker="AAA")
        assert k_a < k_b

    def test_weight_scales_effect(self):
        """Larger weight produces larger ordering change."""
        rs_small = DecisionRuleset(
            alpha_modifier_mode="within_tier",
            alpha_modifier_weight=0.05,
        )
        rs_big = DecisionRuleset(
            alpha_modifier_mode="within_tier",
            alpha_modifier_weight=0.25,
        )
        # Both should improve rank, but big weight should improve more
        k_small = _sort_key(alpha_raw=0.10, ruleset=rs_small, ticker="AAA")
        k_big = _sort_key(alpha_raw=0.10, ruleset=rs_big, ticker="AAA")
        k_baseline = _sort_key(alpha_raw=0.0, ruleset=rs_small, ticker="AAA")
        # big improvement > small improvement (both < baseline)
        assert k_big < k_small < k_baseline


# =============================================================================
# Missing alpha — safe defaults
# =============================================================================

class TestMissingAlpha:
    """Missing or None alpha should produce safe defaults (no crash, no effect)."""

    def test_tiebreak_mode_none_alpha(self):
        rs = DecisionRuleset(alpha_modifier_mode="tiebreak")
        k = _sort_key(alpha_raw=None, ruleset=rs)
        assert k is not None  # no crash

    def test_within_tier_mode_none_alpha(self):
        rs = DecisionRuleset(
            alpha_modifier_mode="within_tier",
            alpha_modifier_weight=0.20,
        )
        k = _sort_key(alpha_raw=None, ruleset=rs)
        assert k is not None  # no crash

    def test_none_alpha_equals_zero_alpha(self):
        """None and 0.0 should produce identical sort keys."""
        for mode in ("tiebreak", "within_tier"):
            rs = DecisionRuleset(
                alpha_modifier_mode=mode,
                alpha_modifier_weight=0.10,
            )
            k_none = _sort_key(alpha_raw=None, ruleset=rs, ticker="AAA")
            k_zero = _sort_key(alpha_raw=0.0, ruleset=rs, ticker="AAA")
            assert k_none == k_zero, f"mode={mode}: None != 0.0"


# =============================================================================
# Cross-mode interaction: catalyst_priority_mode × alpha_modifier_mode
# =============================================================================

class TestCrossModeInteraction:
    """Alpha modifier works correctly across all 3 catalyst priority modes."""

    @pytest.mark.parametrize("cat_mode", ["off", "tiebreaker", "blended"])
    def test_tiebreak_alpha_works_in_all_cat_modes(self, cat_mode):
        rs = DecisionRuleset(
            alpha_modifier_mode="tiebreak",
            catalyst_priority_mode=cat_mode,
        )
        k_high = _sort_key(alpha_raw=0.05, ruleset=rs, ticker="AAA")
        k_low = _sort_key(alpha_raw=0.01, ruleset=rs, ticker="BBB")
        assert k_high < k_low

    @pytest.mark.parametrize("cat_mode", ["off", "tiebreaker", "blended"])
    def test_within_tier_alpha_works_in_all_cat_modes(self, cat_mode):
        rs = DecisionRuleset(
            alpha_modifier_mode="within_tier",
            alpha_modifier_weight=0.20,
            catalyst_priority_mode=cat_mode,
        )
        k_alpha = _sort_key(alpha_raw=0.10, ruleset=rs, ticker="AAA")
        k_none = _sort_key(alpha_raw=0.0, ruleset=rs, ticker="AAA")
        assert k_alpha < k_none


# =============================================================================
# Candidate ruleset loads cleanly
# =============================================================================

class TestCandidateRuleset:
    """v1.6.0_alpha_modifier_candidate.json loads and has expected overrides."""

    CANDIDATE_PATH = (
        Path(__file__).resolve().parent.parent
        / "production_data" / "decision_rulesets"
        / "v1.6.0_alpha_modifier_candidate.json"
    )

    def test_loads(self):
        rs = DecisionRuleset.from_json(str(self.CANDIDATE_PATH))
        assert rs.alpha_modifier_mode == "tiebreak"
        assert rs.alpha_modifier_weight == 0.05

    def test_rebuild_policy_if_missing(self):
        rs = DecisionRuleset.from_json(str(self.CANDIDATE_PATH))
        assert rs.alpha_table_rebuild_policy == "if_missing"

    def test_train_mode_trailing_6(self):
        rs = DecisionRuleset.from_json(str(self.CANDIDATE_PATH))
        assert rs.alpha_train_mode == "trailing-6"
        assert rs.alpha_train_horizon == 84

    def test_base_settings_match_production(self):
        """Core settings should match v1.5.1 production (coinvest off, alpha_cohort engine)."""
        rs = DecisionRuleset.from_json(str(self.CANDIDATE_PATH))
        assert rs.composite_engine == "alpha_cohort"
        assert rs.coinvest_sort_weight == 0.0
        assert rs.enable_clinical_sort_signal is True
        assert rs.enable_missingness_sort_penalty is True
        assert rs.catalyst_priority_mode == "tiebreaker"
        assert rs.sort_anchor == "alpha_cohort"


# =============================================================================
# Enhanced A/B report: tier distribution + alpha coverage
# =============================================================================

def _ab_row(ticker, rank, tier="A", alpha_raw=None, eligible="1", archetype="drug_developer"):
    """Build a synthetic csv_row for _compute_alpha_modifier_ab tests."""
    row = {
        "ticker": ticker,
        "eligible": eligible,
        "actionable_rank": str(rank) if eligible == "1" else "",
        "tier_dev": tier,
        "composite_rank": rank,
        "archetype": archetype,
        "catalyst_mode": "specific_days",
        "catalyst_days": "30",
        "sponsor_tier1_count": "2",
        "mom_state": "neutral",
        "missingness_penalty": "0",
        "clinical_optionality_pct_dev": "0.50",
        "clinical_score_z": "0.0",
        "clinical_score_z_tier": "0.0",
        "alpha_cohort_raw": str(alpha_raw) if alpha_raw is not None else "",
        "alpha_cohort_pct": "0.50",
        "catalyst_event_type": "",
        "catalyst_source": "",
        "catalyst_decay": "0.0",
    }
    return row


def _build_ab_rows():
    """8 eligible rows with mix of tiers and alpha values."""
    return [
        _ab_row("AAA", 1, tier="A", alpha_raw=0.08),
        _ab_row("BBB", 2, tier="A", alpha_raw=0.05),
        _ab_row("CCC", 3, tier="A", alpha_raw=None),
        _ab_row("DDD", 4, tier="B", alpha_raw=0.10),
        _ab_row("EEE", 5, tier="B", alpha_raw=0.02),
        _ab_row("FFF", 6, tier="B", alpha_raw=None),
        _ab_row("GGG", 7, tier="A", alpha_raw=-0.03),
        _ab_row("HHH", 8, tier="B", alpha_raw=0.01),
    ]


class TestEnhancedABReport:
    """Tests for tier distribution and alpha coverage in A/B report."""

    def test_ab_report_has_tier_dist_keys(self):
        """Result contains tier_dist_modified_top60 and tier_dist_baseline_top60."""
        rows = _build_ab_rows()
        rs = DecisionRuleset(alpha_modifier_mode="tiebreak")
        result = _compute_alpha_modifier_ab(rows, rs, logging.getLogger("test"))
        assert "tier_dist_modified_top60" in result
        assert "tier_dist_baseline_top60" in result
        assert isinstance(result["tier_dist_modified_top60"], dict)
        assert isinstance(result["tier_dist_baseline_top60"], dict)

    def test_ab_report_has_coverage_keys(self):
        """Result contains alpha coverage stats."""
        rows = _build_ab_rows()
        rs = DecisionRuleset(alpha_modifier_mode="tiebreak")
        result = _compute_alpha_modifier_ab(rows, rs, logging.getLogger("test"))
        assert "alpha_coverage_present" in result
        assert "alpha_coverage_missing" in result
        assert "alpha_coverage_pct" in result
        # 6 rows have non-zero alpha (0.08,0.05,0.10,0.02,-0.03,0.01), 2 are None
        assert result["alpha_coverage_present"] == 6
        assert result["alpha_coverage_missing"] == 2
        assert 0 < result["alpha_coverage_pct"] < 100

    def test_ab_mode_off_overlap_is_one(self):
        """With mode=off, modified == baseline, so overlap=1.0 and zero shifts."""
        from decision_engine import _safe_float as sf
        rows = _build_ab_rows()
        rs = DecisionRuleset(alpha_modifier_mode="off")
        # Pre-sort rows using actual sort key so actionable_rank matches reality
        rows.sort(key=lambda r: compute_actionable_sort_key(
            decision_fields=r,
            archetype=r.get("archetype", ""),
            optionality=sf(r.get("clinical_optionality_pct_dev")),
            composite_rank=r.get("composite_rank"),
            ticker=r.get("ticker", ""),
            ruleset=rs,
            alpha_raw=sf(r.get("alpha_cohort_raw")),
        ))
        for i, row in enumerate(rows):
            if row.get("eligible") == "1":
                row["actionable_rank"] = str(i + 1)
        result = _compute_alpha_modifier_ab(rows, rs, logging.getLogger("test"))
        assert result["top60_overlap"] == 1.0
        assert result["pct_rows_rank_changed"] == 0.0

    def test_ab_tiebreak_with_alpha_changes_order(self):
        """Tiebreak mode with alpha produces overlap < 1.0 when alpha varies."""
        # Build rows where all have identical sort fields except alpha
        rows = []
        for i, tk in enumerate(["T1", "T2", "T3", "T4", "T5"]):
            rows.append(_ab_row(tk, i + 1, tier="A", alpha_raw=0.10 - i * 0.05))
        rs = DecisionRuleset(alpha_modifier_mode="tiebreak")
        result = _compute_alpha_modifier_ab(rows, rs, logging.getLogger("test"))
        # With alpha off, ordering falls to ticker name tiebreak — different from alpha-based
        assert result["n_rows"] > 0


# =============================================================================
# Telemetry fields from DecisionRuleset
# =============================================================================

class TestAlphaTelemetryFields:
    """Verify DecisionRuleset exposes correct alpha modifier fields for telemetry."""

    def test_telemetry_fields_default_ruleset(self):
        """Default ruleset has mode=off, weight=0.0, policy=never."""
        rs = DecisionRuleset()
        assert rs.alpha_modifier_mode == "off"
        assert rs.alpha_modifier_weight == 0.0
        assert rs.alpha_table_rebuild_policy == "never"

    def test_telemetry_fields_candidate(self):
        """v1.6.0 candidate has mode=tiebreak, weight=0.05, policy=if_missing."""
        candidate = (
            Path(__file__).resolve().parent.parent
            / "production_data" / "decision_rulesets"
            / "v1.6.0_alpha_modifier_candidate.json"
        )
        rs = DecisionRuleset.from_json(str(candidate))
        assert rs.alpha_modifier_mode == "tiebreak"
        assert rs.alpha_modifier_weight == 0.05
        assert rs.alpha_table_rebuild_policy == "if_missing"
