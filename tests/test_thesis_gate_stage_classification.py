"""
Regression tests for thesis gate stage classification.

These tests encode the specific failure mode fixed in commit 069a897:
Late-stage companies were being gated on their *event* stage (poc/pivotal/regulatory)
instead of their *company* stage (early/mid/late), causing systematic over-penalization.

The canonical example is INSM: a Phase 3 company with a CT_PRIMARY_COMPLETION catalyst
was classified as "poc" (threshold 55) instead of "late" (threshold 45), triggering
the thesis gate when it shouldn't have.
"""
from decimal import Decimal

import pytest

from module_5_scoring_v3 import (
    _stage_bucket,
    _determine_stage_bucket_alpha,
    apply_thesis_gate,
    ScoringMode,
    THESIS_GATE_CONFIG,
)


class TestStageClassification:
    """Test that company stage and event stage are classified correctly."""

    def test_phase3_company_is_late_stage(self):
        """Phase 3 companies should be classified as 'late' for company stage."""
        assert _stage_bucket("phase 3") == "late"
        assert _stage_bucket("Phase 3") == "late"
        assert _stage_bucket("PHASE 3") == "late"
        assert _stage_bucket("phase_3") == "late"

    def test_approved_company_is_late_stage(self):
        """Approved/commercial companies should be classified as 'late'."""
        assert _stage_bucket("approved") == "late"
        assert _stage_bucket("Approved") == "late"

    def test_phase2_company_is_mid_stage(self):
        """Phase 2 companies should be classified as 'mid'."""
        assert _stage_bucket("phase 2") == "mid"
        assert _stage_bucket("Phase 2") == "mid"

    def test_phase1_company_is_early_stage(self):
        """Phase 1 companies should be classified as 'early'."""
        assert _stage_bucket("phase 1") == "early"
        assert _stage_bucket("preclinical") == "early"

    def test_ct_primary_completion_maps_to_poc_event_stage(self):
        """CT_PRIMARY_COMPLETION catalyst should map to 'poc' event stage."""
        # This is the event stage - used for timing/decay, NOT thesis gate threshold
        assert _determine_stage_bucket_alpha(
            lead_phase="phase 3",
            cat_event_type="CT_PRIMARY_COMPLETION"
        ) == "poc"

    def test_ct_results_posted_maps_to_poc_event_stage(self):
        """CT_RESULTS_POSTED catalyst should map to 'poc' event stage."""
        assert _determine_stage_bucket_alpha(
            lead_phase="phase 3",
            cat_event_type="CT_RESULTS_POSTED"
        ) == "poc"

    def test_fda_pdufa_maps_to_regulatory_event_stage(self):
        """FDA_PDUFA_DATE catalyst should map to 'regulatory' event stage."""
        assert _determine_stage_bucket_alpha(
            lead_phase="phase 3",
            cat_event_type="FDA_PDUFA_DATE"
        ) == "regulatory"


class TestThesisGateThresholds:
    """Verify thesis gate uses correct thresholds for different stages."""

    def test_late_stage_threshold_is_45(self):
        """Late stage companies should use threshold 45."""
        assert THESIS_GATE_CONFIG["thresholds_by_stage"]["late"] == Decimal("45")

    def test_poc_stage_threshold_is_55(self):
        """POC event stage has threshold 55 (but shouldn't be used for late companies)."""
        assert THESIS_GATE_CONFIG["thresholds_by_stage"]["poc"] == Decimal("55")

    def test_threshold_difference_is_significant(self):
        """The 10-point threshold difference is significant enough to cause misclassification."""
        late_threshold = THESIS_GATE_CONFIG["thresholds_by_stage"]["late"]
        poc_threshold = THESIS_GATE_CONFIG["thresholds_by_stage"]["poc"]
        assert poc_threshold - late_threshold == Decimal("10")


class TestThesisGateRegression:
    """
    Regression tests for the specific failure mode:
    Late-stage company + poc-type catalyst should NOT trigger thesis gate
    when thesis score is between late threshold (45) and poc threshold (55).
    """

    @pytest.mark.parametrize("thesis_score,should_trigger", [
        (Decimal("44"), True),   # Below late threshold (45) - should trigger
        (Decimal("45"), False),  # At late threshold - should pass
        (Decimal("46"), False),  # Above late threshold - should pass (INSM case)
        (Decimal("54"), False),  # Below poc threshold but above late - should pass
        (Decimal("55"), False),  # At poc threshold - should pass
        (Decimal("56"), False),  # Above poc threshold - should pass
    ])
    def test_late_stage_uses_late_threshold_not_event_threshold(
        self, thesis_score, should_trigger
    ):
        """
        Late-stage company should use 'late' threshold (45), not 'poc' threshold (55).

        This is the canonical regression test for the INSM bug:
        - INSM: Phase 3 company (late stage)
        - Catalyst: CT_PRIMARY_COMPLETION (poc event stage)
        - Thesis score: 46.65

        Before fix: Used "poc" threshold (55) → triggered (46.65 < 55)
        After fix:  Uses "late" threshold (45) → passed (46.65 >= 45)
        """
        # Simulate clinical_normalized + pos_normalized that averages to thesis_score
        clinical_norm = thesis_score
        pos_norm = thesis_score

        # Call thesis gate with "late" stage (the display/company stage)
        # NOT with "poc" (the event stage)
        gated_score, flags, diagnostics = apply_thesis_gate(
            score=Decimal("50"),
            clinical_normalized=clinical_norm,
            pos_normalized=pos_norm,
            stage_bucket="late",  # Company stage, NOT event stage
            mode=ScoringMode.BAKER_STYLE,
        )

        assert diagnostics["thesis_gate_triggered"] == should_trigger
        assert diagnostics["thesis_threshold"] == str(Decimal("45"))  # Late threshold

    def test_insm_exact_case(self):
        """
        Exact reproduction of INSM bug case.

        INSM had:
        - lead_phase: "phase 3" → company stage = "late"
        - cat_event_type: CT_PRIMARY_COMPLETION → event stage = "poc"
        - thesis_score: 46.65

        The bug was using event stage "poc" (threshold 55) instead of
        company stage "late" (threshold 45).
        """
        # INSM's actual thesis score
        clinical_norm = Decimal("46.65")
        pos_norm = Decimal("46.65")

        # With the fix: use company stage "late"
        gated_score, flags, diagnostics = apply_thesis_gate(
            score=Decimal("50"),
            clinical_normalized=clinical_norm,
            pos_normalized=pos_norm,
            stage_bucket="late",
            mode=ScoringMode.BAKER_STYLE,
        )

        # Should NOT trigger with late threshold (45)
        assert diagnostics["thesis_gate_triggered"] is False
        assert "thesis_gate_passed" in flags

        # Verify the threshold used is 45 (late), not 55 (poc)
        assert diagnostics["thesis_threshold"] == str(Decimal("45"))

    def test_wrong_stage_would_have_triggered(self):
        """
        Verify that using event stage "poc" would have triggered the gate.

        This confirms the bug existed and the fix is meaningful.
        """
        clinical_norm = Decimal("46.65")
        pos_norm = Decimal("46.65")

        # With the BUG: incorrectly using event stage "poc"
        gated_score, flags, diagnostics = apply_thesis_gate(
            score=Decimal("50"),
            clinical_normalized=clinical_norm,
            pos_normalized=pos_norm,
            stage_bucket="poc",  # WRONG: this was the bug
            mode=ScoringMode.BAKER_STYLE,
        )

        # WOULD trigger with poc threshold (55)
        assert diagnostics["thesis_gate_triggered"] is True
        assert diagnostics["thesis_threshold"] == str(Decimal("55"))
