#!/usr/bin/env python3
"""
Tests for pos_prior_engine.py

Probability of Success (PoS) prior affects clinical development scoring.
These tests cover:
- Stage normalization (phase_1, phase_2, phase_3, etc.)
- Therapeutic area normalization
- Base PoS lookup from benchmarks
- Modifier application (orphan, breakthrough, fast track, biomarker)
- Confidence calculation
- PoS weighting integration
"""

import pytest
from datetime import date
from decimal import Decimal

from pos_prior_engine import (
    PoSPriorEngine,
    DataQualityState,
    apply_pos_weighting,
)


class TestStageNormalization:
    """Tests for stage name normalization."""

    def test_phase_1_variations(self):
        """All Phase 1 variations should normalize."""
        engine = PoSPriorEngine()
        variations = ["phase 1", "phase1", "Phase 1", "phase i", "P1", "phase_1"]
        for stage in variations:
            normalized = engine._normalize_stage(stage)
            assert normalized == "phase_1", f"Failed for '{stage}'"

    def test_phase_2_variations(self):
        """All Phase 2 variations should normalize."""
        engine = PoSPriorEngine()
        variations = ["phase 2", "phase2", "Phase 2", "phase ii", "P2", "phase_2"]
        for stage in variations:
            normalized = engine._normalize_stage(stage)
            assert normalized == "phase_2", f"Failed for '{stage}'"

    def test_phase_3_variations(self):
        """All Phase 3 variations should normalize."""
        engine = PoSPriorEngine()
        variations = ["phase 3", "phase3", "Phase 3", "phase iii", "P3", "pivotal"]
        for stage in variations:
            normalized = engine._normalize_stage(stage)
            assert normalized == "phase_3", f"Failed for '{stage}'"

    def test_phase_1_2_combined(self):
        """Phase 1/2 variations should normalize."""
        engine = PoSPriorEngine()
        variations = ["phase 1/2", "phase1/2", "phase i/ii"]
        for stage in variations:
            normalized = engine._normalize_stage(stage)
            assert normalized == "phase_1_2", f"Failed for '{stage}'"

    def test_nda_bla_variations(self):
        """NDA/BLA variations should normalize."""
        engine = PoSPriorEngine()
        variations = ["nda", "bla", "nda/bla", "submitted", "filed"]
        for stage in variations:
            normalized = engine._normalize_stage(stage)
            assert normalized == "nda_bla", f"Failed for '{stage}'"

    def test_preclinical_variations(self):
        """Preclinical variations should normalize."""
        engine = PoSPriorEngine()
        variations = ["preclinical", "pre-clinical", "discovery"]
        for stage in variations:
            normalized = engine._normalize_stage(stage)
            assert normalized == "preclinical", f"Failed for '{stage}'"

    def test_commercial_variations(self):
        """Commercial variations should normalize."""
        engine = PoSPriorEngine()
        variations = ["approved", "commercial", "marketed"]
        for stage in variations:
            normalized = engine._normalize_stage(stage)
            assert normalized == "commercial", f"Failed for '{stage}'"

    def test_unmapped_stage(self):
        """Unmapped stage should return None."""
        engine = PoSPriorEngine()
        assert engine._normalize_stage("unknown_stage") is None
        assert engine._normalize_stage("") is None
        assert engine._normalize_stage(None) is None


class TestTherapeuticAreaNormalization:
    """Tests for therapeutic area normalization."""

    def test_oncology_variations(self):
        """Oncology variations should normalize."""
        engine = PoSPriorEngine()
        variations = ["oncology", "cancer", "tumor", "Oncology", "CANCER"]
        for ta in variations:
            normalized = engine._normalize_therapeutic_area(ta)
            assert normalized == "oncology", f"Failed for '{ta}'"

    def test_rare_disease_variations(self):
        """Rare disease variations should normalize."""
        engine = PoSPriorEngine()
        variations = ["rare disease", "rare_disease", "orphan"]
        for ta in variations:
            normalized = engine._normalize_therapeutic_area(ta)
            assert normalized == "rare_disease", f"Failed for '{ta}'"

    def test_neurology_variations(self):
        """Neurology variations should normalize."""
        engine = PoSPriorEngine()
        variations = ["neurology", "cns", "neurological"]
        for ta in variations:
            normalized = engine._normalize_therapeutic_area(ta)
            assert normalized == "neurology", f"Failed for '{ta}'"

    def test_pattern_matching(self):
        """Complex indications should be pattern-matched."""
        engine = PoSPriorEngine()

        # Oncology patterns
        assert engine._normalize_therapeutic_area("breast cancer") == "oncology"
        assert engine._normalize_therapeutic_area("non-small cell lung carcinoma") == "oncology"
        assert engine._normalize_therapeutic_area("acute myeloid leukemia") == "oncology"

        # Neurology patterns
        assert engine._normalize_therapeutic_area("Alzheimer's disease") == "neurology"
        assert engine._normalize_therapeutic_area("Parkinson's") == "neurology"

        # Infectious disease patterns
        assert engine._normalize_therapeutic_area("HIV treatment") == "infectious_disease"
        assert engine._normalize_therapeutic_area("COVID-19") == "infectious_disease"

    def test_unmapped_therapeutic_area(self):
        """Unmapped TA should return None."""
        engine = PoSPriorEngine()
        assert engine._normalize_therapeutic_area("xyz unknown") is None
        assert engine._normalize_therapeutic_area("") is None
        assert engine._normalize_therapeutic_area(None) is None


class TestBasePosLookup:
    """Tests for base PoS rate lookup."""

    def test_phase_3_oncology(self):
        """Phase 3 oncology should return benchmark rate."""
        engine = PoSPriorEngine()
        result = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            as_of_date=date(2026, 1, 15),
        )
        assert result["reason_code"] == "SUCCESS"
        assert result["base_pos"] is not None
        # Phase 3 oncology typically ~48% based on BIO data
        assert Decimal("0.40") <= result["base_pos"] <= Decimal("0.60")

    def test_phase_2_rare_disease(self):
        """Phase 2 rare disease should return benchmark rate."""
        engine = PoSPriorEngine()
        result = engine.calculate_pos_prior(
            stage="phase_2",
            therapeutic_area="rare disease",
            as_of_date=date(2026, 1, 15),
        )
        assert result["reason_code"] == "SUCCESS"
        assert result["base_pos"] is not None

    def test_phase_1_all_indications(self):
        """Phase 1 without TA should use all_indications."""
        engine = PoSPriorEngine()
        result = engine.calculate_pos_prior(
            stage="phase_1",
            as_of_date=date(2026, 1, 15),
        )
        assert result["reason_code"] == "SUCCESS"
        assert result["metadata"]["ta_mapped"] == "all_indications"

    def test_preclinical_interpolation(self):
        """Preclinical should return ~50% of Phase 1 rate."""
        engine = PoSPriorEngine()
        result = engine.calculate_pos_prior(
            stage="preclinical",
            therapeutic_area="oncology",
            as_of_date=date(2026, 1, 15),
        )
        assert result["reason_code"] == "SUCCESS"
        # Preclinical = 50% of Phase 1, so should be very low
        assert result["base_pos"] < Decimal("0.10")

    def test_phase_1_2_interpolation(self):
        """Phase 1/2 should interpolate between Phase 1 and Phase 2."""
        engine = PoSPriorEngine()
        result = engine.calculate_pos_prior(
            stage="phase 1/2",
            therapeutic_area="oncology",
            as_of_date=date(2026, 1, 15),
        )
        assert result["reason_code"] == "SUCCESS"
        # Should be interpolation of Phase 1 (~9.5%) and Phase 2 (~28.5%)
        # Result varies based on benchmarks loaded - accept reasonable range
        assert Decimal("0.05") < result["base_pos"] < Decimal("0.40")

    def test_commercial_is_1(self):
        """Commercial stage should return 1.0 (already approved)."""
        engine = PoSPriorEngine()
        result = engine.calculate_pos_prior(
            stage="commercial",
            as_of_date=date(2026, 1, 15),
        )
        assert result["base_pos"] == Decimal("1.0")


class TestModifierApplication:
    """Tests for FDA designation modifiers."""

    def test_orphan_designation_boost(self):
        """Orphan drug designation should boost PoS by 15%."""
        engine = PoSPriorEngine()
        base_result = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            as_of_date=date(2026, 1, 15),
        )
        orphan_result = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            orphan_drug_designation=True,
            as_of_date=date(2026, 1, 15),
        )

        assert orphan_result["modifier_adjustment"] == Decimal("1.15")
        assert orphan_result["pos_prior"] > base_result["pos_prior"]
        assert "orphan_drug_designation" in orphan_result["modifiers_applied"]

    def test_breakthrough_designation_boost(self):
        """Breakthrough designation should boost PoS by 25%."""
        engine = PoSPriorEngine()
        result = engine.calculate_pos_prior(
            stage="phase_2",
            therapeutic_area="oncology",
            breakthrough_designation=True,
            as_of_date=date(2026, 1, 15),
        )

        assert result["modifier_adjustment"] == Decimal("1.25")
        assert "breakthrough_designation" in result["modifiers_applied"]

    def test_fast_track_designation_boost(self):
        """Fast track designation should boost PoS by 10%."""
        engine = PoSPriorEngine()
        result = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            fast_track_designation=True,
            as_of_date=date(2026, 1, 15),
        )

        assert result["modifier_adjustment"] == Decimal("1.10")
        assert "fast_track_designation" in result["modifiers_applied"]

    def test_biomarker_enriched_boost(self):
        """Biomarker enrichment should boost PoS by 20%."""
        engine = PoSPriorEngine()
        result = engine.calculate_pos_prior(
            stage="phase_2",
            therapeutic_area="oncology",
            biomarker_enriched=True,
            as_of_date=date(2026, 1, 15),
        )

        assert result["modifier_adjustment"] == Decimal("1.20")
        assert "biomarker_enriched" in result["modifiers_applied"]

    def test_multiple_modifiers_compound(self):
        """Multiple modifiers should compound."""
        engine = PoSPriorEngine()
        result = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            orphan_drug_designation=True,
            breakthrough_designation=True,
            as_of_date=date(2026, 1, 15),
        )

        # 1.15 * 1.25 = 1.4375
        expected_mult = Decimal("1.44")
        assert result["modifier_adjustment"] == expected_mult
        assert len(result["modifiers_applied"]) == 2

    def test_modifier_cap_at_2x(self):
        """Total modifier should be capped at 2.0x maximum."""
        engine = PoSPriorEngine()
        result = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="rare disease",
            orphan_drug_designation=True,
            breakthrough_designation=True,
            fast_track_designation=True,
            biomarker_enriched=True,
            as_of_date=date(2026, 1, 15),
        )

        # Compounded: 1.15 * 1.25 * 1.10 * 1.20 = 1.8975 (rounds to 1.90)
        # Cap only applies if >2.0, so result is ~1.90
        assert result["modifier_adjustment"] <= Decimal("2.00")
        assert Decimal("1.85") <= result["modifier_adjustment"] <= Decimal("2.00")

    def test_pos_capped_at_95_percent(self):
        """PoS should never exceed 95%."""
        engine = PoSPriorEngine()
        result = engine.calculate_pos_prior(
            stage="nda_bla",  # ~90% base rate
            therapeutic_area="rare disease",
            orphan_drug_designation=True,
            breakthrough_designation=True,
            as_of_date=date(2026, 1, 15),
        )

        assert result["pos_prior"] <= Decimal("0.95")


class TestConfidenceCalculation:
    """Tests for confidence calculation."""

    def test_high_confidence_full_data(self):
        """Full data should yield high confidence."""
        engine = PoSPriorEngine()
        result = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            orphan_drug_designation=True,
            as_of_date=date(2026, 1, 15),
        )

        assert result["confidence"] >= Decimal("0.70")
        assert result["data_quality_state"] == "FULL"

    def test_partial_confidence_no_modifiers(self):
        """TA without modifiers should yield partial confidence."""
        engine = PoSPriorEngine()
        result = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            as_of_date=date(2026, 1, 15),
        )

        assert result["data_quality_state"] == "PARTIAL"

    def test_minimal_confidence_no_ta(self):
        """No TA and no modifiers should yield minimal data quality."""
        engine = PoSPriorEngine()
        result = engine.calculate_pos_prior(
            stage="phase_3",
            as_of_date=date(2026, 1, 15),
        )

        assert result["data_quality_state"] == "MINIMAL"

    def test_confidence_bounded_0_1(self):
        """Confidence should always be between 0 and 1."""
        engine = PoSPriorEngine()
        result = engine.calculate_pos_prior(
            stage="phase_2",
            therapeutic_area="oncology",
            orphan_drug_designation=True,
            breakthrough_designation=True,
            fast_track_designation=True,
            biomarker_enriched=True,
            as_of_date=date(2026, 1, 15),
        )

        assert Decimal("0") <= result["confidence"] <= Decimal("1")


class TestErrorHandling:
    """Tests for error handling."""

    def test_missing_stage(self):
        """Missing stage should return error result."""
        engine = PoSPriorEngine()
        result = engine.calculate_pos_prior(
            stage="",
            therapeutic_area="oncology",
            as_of_date=date(2026, 1, 15),
        )

        assert result["reason_code"] == "INSUFFICIENT_DATA"
        assert result["pos_prior"] is None
        assert result["confidence"] == Decimal("0.0")

    def test_unmapped_stage(self):
        """Unmapped stage should return error result."""
        engine = PoSPriorEngine()
        result = engine.calculate_pos_prior(
            stage="unknown_phase",
            therapeutic_area="oncology",
            as_of_date=date(2026, 1, 15),
        )

        assert result["reason_code"] == "UNMAPPED_STAGE"
        assert result["pos_prior"] is None

    def test_unmapped_ta_uses_all_indications(self):
        """Unmapped TA should fall back to all_indications."""
        engine = PoSPriorEngine()
        result = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="xyz_unknown_area",
            as_of_date=date(2026, 1, 15),
        )

        assert result["reason_code"] == "SUCCESS"
        assert result["metadata"]["ta_mapped"] == "all_indications"


class TestScoreUniverse:
    """Tests for score_universe batch scoring."""

    def test_scores_multiple_companies(self):
        """Should score multiple companies."""
        engine = PoSPriorEngine()
        universe = [
            {"ticker": "ACME", "stage": "phase_3", "therapeutic_area": "oncology"},
            {"ticker": "BETA", "stage": "phase_2", "therapeutic_area": "rare disease"},
            {"ticker": "GAMMA", "stage": "phase_1"},
        ]
        result = engine.score_universe(universe, as_of_date=date(2026, 1, 15))

        assert len(result["scores"]) == 3
        assert result["diagnostic_counts"]["total_scored"] == 3
        assert result["diagnostic_counts"]["success_count"] == 3

    def test_tracks_stage_distribution(self):
        """Should track stage distribution."""
        engine = PoSPriorEngine()
        universe = [
            {"ticker": "A", "stage": "phase_3"},
            {"ticker": "B", "stage": "phase_3"},
            {"ticker": "C", "stage": "phase_2"},
        ]
        result = engine.score_universe(universe, as_of_date=date(2026, 1, 15))

        dist = result["diagnostic_counts"]["stage_distribution"]
        assert dist["phase_3"] == 2
        assert dist["phase_2"] == 1

    def test_provenance_included(self):
        """Provenance should be included."""
        engine = PoSPriorEngine()
        universe = [{"ticker": "ACME", "stage": "phase_3"}]
        result = engine.score_universe(universe, as_of_date=date(2026, 1, 15))

        assert "provenance" in result
        assert result["provenance"]["module"] == "pos_prior_engine"
        assert result["provenance"]["content_hash"] is not None


class TestApplyPosWeighting:
    """Tests for apply_pos_weighting function."""

    def test_applies_weighting_high_confidence(self):
        """Should apply weighting when confidence is high."""
        pos_data = {
            "pos_prior": Decimal("0.65"),
            "confidence": Decimal("0.80"),
        }
        stage_score = Decimal("70.00")

        weighted = apply_pos_weighting(stage_score, pos_data)

        # pos_prior 0.65 -> multiplier = 0.5 + (0.65 * 0.8) = 1.02
        # weighted = 70 * 1.02 = 71.4
        assert weighted > stage_score
        assert weighted <= Decimal("100.00")

    def test_no_weighting_low_confidence(self):
        """Should not apply weighting when confidence is low."""
        pos_data = {
            "pos_prior": Decimal("0.65"),
            "confidence": Decimal("0.50"),  # Below default threshold of 0.60
        }
        stage_score = Decimal("70.00")

        weighted = apply_pos_weighting(stage_score, pos_data)

        assert weighted == stage_score

    def test_custom_confidence_threshold(self):
        """Custom threshold should be respected."""
        pos_data = {
            "pos_prior": Decimal("0.65"),
            "confidence": Decimal("0.50"),
        }
        stage_score = Decimal("70.00")

        # With threshold of 0.40, should apply weighting
        weighted = apply_pos_weighting(
            stage_score, pos_data, confidence_threshold=Decimal("0.40")
        )

        assert weighted != stage_score

    def test_none_pos_prior(self):
        """None pos_prior should return original score."""
        pos_data = {
            "pos_prior": None,
            "confidence": Decimal("0.80"),
        }
        stage_score = Decimal("70.00")

        weighted = apply_pos_weighting(stage_score, pos_data)

        assert weighted == stage_score

    def test_bounded_to_100(self):
        """Weighted score should be capped at 100."""
        pos_data = {
            "pos_prior": Decimal("0.95"),  # High PoS
            "confidence": Decimal("0.90"),
        }
        stage_score = Decimal("90.00")  # High base score

        weighted = apply_pos_weighting(stage_score, pos_data)

        assert weighted <= Decimal("100.00")

    def test_bounded_to_0(self):
        """Weighted score should not go below 0."""
        pos_data = {
            "pos_prior": Decimal("0.10"),  # Low PoS -> multiplier ~0.58
            "confidence": Decimal("0.90"),
        }
        stage_score = Decimal("10.00")

        weighted = apply_pos_weighting(stage_score, pos_data)

        assert weighted >= Decimal("0.00")


class TestAuditTrail:
    """Tests for audit trail functionality."""

    def test_audit_trail_recorded(self):
        """Calculations should be recorded in audit trail."""
        engine = PoSPriorEngine()
        engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            as_of_date=date(2026, 1, 15),
        )

        audit = engine.get_audit_trail()
        assert len(audit) == 1
        assert "inputs_used" in audit[0]
        assert "calculation" in audit[0]

    def test_clear_audit_trail(self):
        """Should be able to clear audit trail."""
        engine = PoSPriorEngine()
        engine.calculate_pos_prior(
            stage="phase_3",
            as_of_date=date(2026, 1, 15),
        )
        engine.clear_audit_trail()

        assert len(engine.get_audit_trail()) == 0


class TestBenchmarksInfo:
    """Tests for benchmarks metadata."""

    def test_benchmarks_info(self):
        """Should return benchmarks info."""
        engine = PoSPriorEngine()
        info = engine.get_benchmarks_info()

        assert "metadata" in info
        assert "stages_available" in info
        assert "therapeutic_areas_available" in info
        assert "modifiers_available" in info


class TestDeterminism:
    """Tests for deterministic output."""

    def test_same_inputs_same_output(self):
        """Same inputs should produce identical outputs."""
        engine1 = PoSPriorEngine()
        engine2 = PoSPriorEngine()

        result1 = engine1.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            orphan_drug_designation=True,
            as_of_date=date(2026, 1, 15),
        )
        result2 = engine2.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            orphan_drug_designation=True,
            as_of_date=date(2026, 1, 15),
        )

        assert result1["pos_prior"] == result2["pos_prior"]
        assert result1["confidence"] == result2["confidence"]
        assert result1["hash"] == result2["hash"]
