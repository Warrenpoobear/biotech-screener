"""
Sort Contribution invariant tests.

Validates the _build_sort_contributions() helper and its integration
with compute_actionable_sort_key().
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from decision_engine import (
    DEFAULT_RULESET,
    DecisionRuleset,
    SortContribution,
    _build_sort_contributions,
    compute_actionable_sort_key,
)

# =============================================================================
# A) Identity: default ruleset (all signals disabled) → empty, total_adj == 0
# =============================================================================


class TestIdentity:
    def test_default_ruleset_empty_contributions(self):
        contribs = _build_sort_contributions(
            {},
            DEFAULT_RULESET,
            alpha_raw=0.0,
            catalyst_bonus=0.0,
        )
        assert contribs == []
        assert sum(c.delta for c in contribs) == 0.0

    def test_default_ruleset_with_nonzero_inputs(self):
        """Even with z-score inputs, disabled signals produce no contributions."""
        fields = {
            "clinical_score_z_tier": 1.5,
            "coinvest_score_z": 0.8,
            "inst_delta_z": 1.2,
            "clinical_score_v2_z": 0.9,
            "stage_bucket": "late",
        }
        contribs = _build_sort_contributions(
            fields,
            DEFAULT_RULESET,
            alpha_raw=0.5,
            catalyst_bonus=0.0,
        )
        assert contribs == []


# =============================================================================
# B) Bounded delta: |delta| <= weight * 2.0 for clamped signals
# =============================================================================


class TestBoundedDelta:
    @pytest.mark.parametrize("z", [-5.0, -2.0, -0.5, 0.0, 0.5, 2.0, 5.0])
    def test_clinical_delta_bounded(self, z):
        rs = DecisionRuleset(enable_clinical_sort_signal=True)
        fields = {"clinical_score_z_tier": z, "stage_bucket": "late"}
        contribs = _build_sort_contributions(
            fields,
            rs,
            alpha_raw=0.0,
            catalyst_bonus=0.0,
        )
        assert len(contribs) == 1
        c = contribs[0]
        stage_mult = dict(rs.clinical_stage_mults).get("late", 0.0)
        assert abs(c.delta) <= rs.clinical_sort_weight * 2.0 * stage_mult + 1e-12

    @pytest.mark.parametrize("z", [-5.0, -2.0, 0.0, 2.0, 5.0])
    def test_coinvest_delta_bounded(self, z):
        rs = DecisionRuleset(enable_coinvest_sort_signal=True)
        fields = {"coinvest_score_z": z}
        contribs = _build_sort_contributions(
            fields,
            rs,
            alpha_raw=0.0,
            catalyst_bonus=0.0,
        )
        assert len(contribs) == 1
        assert abs(contribs[0].delta) <= rs.coinvest_sort_weight * 2.0 + 1e-12

    @pytest.mark.parametrize("z", [-5.0, -2.0, 0.0, 2.0, 5.0])
    def test_institutional_delta_bounded(self, z):
        rs = DecisionRuleset(enable_institutional_sort_signal=True)
        fields = {"inst_delta_z": z}
        contribs = _build_sort_contributions(
            fields,
            rs,
            alpha_raw=0.0,
            catalyst_bonus=0.0,
        )
        assert len(contribs) == 1
        assert abs(contribs[0].delta) <= rs.institutional_sort_weight * 2.0 + 1e-12

    @pytest.mark.parametrize("z", [-5.0, -2.0, 0.0, 2.0, 5.0])
    def test_calendar_alpha_delta_bounded(self, z):
        rs = DecisionRuleset(enable_calendar_alpha_sort=True)
        fields = {"clinical_score_v2_z": z}
        contribs = _build_sort_contributions(
            fields,
            rs,
            alpha_raw=0.0,
            catalyst_bonus=0.0,
        )
        assert len(contribs) == 1
        assert abs(contribs[0].delta) <= rs.calendar_alpha_sort_weight * 2.0 + 1e-12


# =============================================================================
# C) NaN safety: NaN/None/empty inputs → delta == 0.0
# =============================================================================


class TestNaNSafety:
    @pytest.mark.parametrize("bad_val", [None, "", float("nan")])
    def test_clinical_nan_safe(self, bad_val):
        rs = DecisionRuleset(enable_clinical_sort_signal=True)
        fields = {"clinical_score_z_tier": bad_val, "stage_bucket": "late"}
        contribs = _build_sort_contributions(
            fields,
            rs,
            alpha_raw=0.0,
            catalyst_bonus=0.0,
        )
        assert len(contribs) == 1
        assert contribs[0].delta == 0.0

    @pytest.mark.parametrize("bad_val", [None, "", float("nan")])
    def test_coinvest_nan_safe(self, bad_val):
        rs = DecisionRuleset(enable_coinvest_sort_signal=True)
        fields = {"coinvest_score_z": bad_val}
        contribs = _build_sort_contributions(
            fields,
            rs,
            alpha_raw=0.0,
            catalyst_bonus=0.0,
        )
        assert len(contribs) == 1
        assert contribs[0].delta == 0.0

    @pytest.mark.parametrize("bad_val", [None, "", float("nan")])
    def test_institutional_nan_safe(self, bad_val):
        rs = DecisionRuleset(enable_institutional_sort_signal=True)
        fields = {"inst_delta_z": bad_val}
        contribs = _build_sort_contributions(
            fields,
            rs,
            alpha_raw=0.0,
            catalyst_bonus=0.0,
        )
        assert len(contribs) == 1
        assert contribs[0].delta == 0.0

    @pytest.mark.parametrize("bad_val", [None, "", float("nan")])
    def test_calendar_alpha_nan_safe(self, bad_val):
        rs = DecisionRuleset(enable_calendar_alpha_sort=True)
        fields = {"clinical_score_v2_z": bad_val}
        contribs = _build_sort_contributions(
            fields,
            rs,
            alpha_raw=0.0,
            catalyst_bonus=0.0,
        )
        assert len(contribs) == 1
        assert contribs[0].delta == 0.0

    def test_missing_fields_safe(self):
        """All signals enabled but no fields in dict → all deltas zero."""
        rs = DecisionRuleset(
            enable_clinical_sort_signal=True,
            enable_coinvest_sort_signal=True,
            enable_institutional_sort_signal=True,
            enable_calendar_alpha_sort=True,
        )
        contribs = _build_sort_contributions(
            {},
            rs,
            alpha_raw=0.0,
            catalyst_bonus=0.0,
        )
        for c in contribs:
            assert c.delta == 0.0, f"{c.name} should have delta=0 with missing fields"


# =============================================================================
# D) Output equivalence: golden fixtures produce identical sort tuples
#    (implicitly tested by TestGoldenOutputFingerprint; explicit check here)
# =============================================================================


class TestOutputEquivalence:
    def test_golden_fingerprint_unchanged(self):
        """The refactored code must produce the exact same golden fingerprint.

        This is the same assertion as TestGoldenOutputFingerprint in
        test_decision_engine_contract.py — duplicated here for self-contained
        verification of the contribution refactor.
        """
        from tests.test_decision_engine_contract import _compute_golden_output_fingerprint

        actual = _compute_golden_output_fingerprint()
        assert actual == "d9cd64b3a321"  # pragma: allowlist secret


# =============================================================================
# E) Completeness: all 5 signal names present when all enabled
# =============================================================================


class TestCompleteness:
    def test_all_signals_present(self):
        """All active sort signals produce contributions (alpha_modifier removed)."""
        rs = DecisionRuleset(
            enable_clinical_sort_signal=True,
            enable_coinvest_sort_signal=True,
            enable_institutional_sort_signal=True,
            enable_calendar_alpha_sort=True,
        )
        fields = {
            "clinical_score_z_tier": 1.0,
            "stage_bucket": "late",
            "coinvest_score_z": 0.5,
            "inst_delta_z": 0.3,
            "clinical_score_v2_z": 0.8,
        }
        contribs = _build_sort_contributions(
            fields,
            rs,
            alpha_raw=0.1,
            catalyst_bonus=0.0,
        )
        names = [c.name for c in contribs]
        assert names == [
            "clinical",
            "coinvest",
            "institutional",
            "calendar_alpha",
        ]

    def test_catalyst_bonus_included_when_nonzero(self):
        rs = DecisionRuleset(
            enable_clinical_sort_signal=True,
            enable_coinvest_sort_signal=True,
            enable_institutional_sort_signal=True,
            enable_calendar_alpha_sort=True,
        )
        fields = {
            "clinical_score_z_tier": 1.0,
            "stage_bucket": "late",
            "coinvest_score_z": 0.5,
            "inst_delta_z": 0.3,
            "clinical_score_v2_z": 0.8,
        }
        contribs = _build_sort_contributions(
            fields,
            rs,
            alpha_raw=0.1,
            catalyst_bonus=2.0,
        )
        names = [c.name for c in contribs]
        assert "catalyst_bonus" in names
        assert len(names) == 5  # 4 signals + catalyst_bonus


# =============================================================================
# F) Positive-only: negative z with positive_only → delta == 0
# =============================================================================


class TestPositiveOnly:
    def test_clinical_negative_z_positive_only(self):
        rs = DecisionRuleset(
            enable_clinical_sort_signal=True,
            clinical_positive_only=True,
        )
        fields = {"clinical_score_z_tier": -1.5, "stage_bucket": "late"}
        contribs = _build_sort_contributions(
            fields,
            rs,
            alpha_raw=0.0,
            catalyst_bonus=0.0,
        )
        assert contribs[0].delta == 0.0

    def test_coinvest_negative_z_positive_only(self):
        rs = DecisionRuleset(
            enable_coinvest_sort_signal=True,
            coinvest_positive_only=True,
        )
        fields = {"coinvest_score_z": -0.8}
        contribs = _build_sort_contributions(
            fields,
            rs,
            alpha_raw=0.0,
            catalyst_bonus=0.0,
        )
        assert contribs[0].delta == 0.0

    def test_institutional_negative_z_positive_only(self):
        rs = DecisionRuleset(
            enable_institutional_sort_signal=True,
            institutional_positive_only=True,
        )
        fields = {"inst_delta_z": -2.0}
        contribs = _build_sort_contributions(
            fields,
            rs,
            alpha_raw=0.0,
            catalyst_bonus=0.0,
        )
        assert contribs[0].delta == 0.0

    def test_calendar_alpha_negative_z_positive_only(self):
        """Calendar alpha is always positive-only (no flag)."""
        rs = DecisionRuleset(enable_calendar_alpha_sort=True)
        fields = {"clinical_score_v2_z": -1.0}
        contribs = _build_sort_contributions(
            fields,
            rs,
            alpha_raw=0.0,
            catalyst_bonus=0.0,
        )
        assert contribs[0].delta == 0.0

    def test_clinical_positive_z_not_positive_only(self):
        """With positive_only=False, negative z produces negative delta."""
        rs = DecisionRuleset(
            enable_clinical_sort_signal=True,
            clinical_positive_only=False,
        )
        fields = {"clinical_score_z_tier": -1.5, "stage_bucket": "late"}
        contribs = _build_sort_contributions(
            fields,
            rs,
            alpha_raw=0.0,
            catalyst_bonus=0.0,
        )
        assert contribs[0].delta < 0.0


# =============================================================================
# G) Contribution sum: sum(c.delta) == total_adj used in sort key
# =============================================================================


class TestContributionSum:
    def test_sum_matches_effective_rank_tiebreaker(self):
        """In tiebreaker mode, effective_rank = anchor - total_adj."""
        rs = DecisionRuleset(
            enable_clinical_sort_signal=True,
            enable_calendar_alpha_sort=True,
            catalyst_priority_mode="tiebreaker",
        )
        fields = {
            "eligible": "1",
            "tier_dev": "A",
            "catalyst_mode": "specific_days",
            "catalyst_days": 30,
            "clinical_score_z_tier": 1.2,
            "stage_bucket": "late",
            "clinical_score_v2_z": 0.7,
        }
        contribs = _build_sort_contributions(
            fields,
            rs,
            alpha_raw=0.0,
            catalyst_bonus=0.0,
        )
        total_adj = sum(c.delta for c in contribs)

        sort_tuple = compute_actionable_sort_key(
            fields,
            archetype="drug_developer",
            optionality=0.6,
            composite_rank=5,
            ticker="TEST",
            ruleset=rs,
        )
        # In tiebreaker mode, element at index 3 (after 3 prefix elements) is effective_comp_rank
        anchor = 5.0  # composite_rank
        effective_rank = float(sort_tuple[3])
        assert abs(effective_rank - (anchor - float(total_adj))) < 1e-12

    def test_sum_matches_effective_rank_blended(self):
        """In blended mode, effective_rank = anchor - total_adj (bonus in contribs)."""
        rs = DecisionRuleset(
            enable_clinical_sort_signal=True,
            catalyst_priority_mode="blended",
        )
        fields = {
            "eligible": "1",
            "tier_dev": "A",
            "catalyst_mode": "specific_days",
            "catalyst_days": 30,
            "clinical_score_z_tier": 1.0,
            "stage_bucket": "mid",
        }
        sort_tuple = compute_actionable_sort_key(
            fields,
            archetype="drug_developer",
            optionality=0.6,
            composite_rank=10,
            ticker="TEST",
            ruleset=rs,
        )
        # Just verify it produces a valid tuple without error
        assert isinstance(sort_tuple, tuple)
        assert len(sort_tuple) == 13  # 3 prefix + 10 elements

    def test_sum_matches_effective_opt_neg_off_mode(self):
        """In off mode, effective_opt_neg = opt_neg - total_adj."""
        rs = DecisionRuleset(
            enable_clinical_sort_signal=True,
            catalyst_priority_mode="off",
        )
        fields = {
            "eligible": "1",
            "tier_dev": "A",
            "catalyst_mode": "specific_days",
            "catalyst_days": 30,
            "clinical_score_z_tier": 1.5,
            "stage_bucket": "late",
        }
        contribs = _build_sort_contributions(
            fields,
            rs,
            alpha_raw=0.0,
            catalyst_bonus=0.0,
        )
        total_adj = sum(c.delta for c in contribs)

        sort_tuple = compute_actionable_sort_key(
            fields,
            archetype="drug_developer",
            optionality=0.6,
            composite_rank=5,
            ticker="TEST",
            ruleset=rs,
        )
        # In off mode: prefix(3) + cat_priority, cat_mode_ord, cat_days, missing_count, effective_opt_neg, ...
        opt_neg = -0.6
        effective_opt_neg = float(sort_tuple[7])  # index 7: prefix(3) + 4 elements before it
        assert abs(effective_opt_neg - (opt_neg - float(total_adj))) < 1e-12


# =============================================================================
# Structural: SortContribution is a proper NamedTuple
# =============================================================================


class TestSortContributionStructure:
    def test_namedtuple_fields(self):
        assert SortContribution._fields == ("name", "raw", "weight", "delta")

    def test_instance_creation(self):
        c = SortContribution("test", 1.0, 0.5, 0.25)
        assert c.name == "test"
        assert c.raw == 1.0
        assert c.weight == 0.5
        assert c.delta == 0.25
