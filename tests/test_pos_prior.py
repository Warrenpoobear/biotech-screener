#!/usr/bin/env python3
"""
tests/test_pos_prior.py

Comprehensive tests for the Probability of Success Prior Engine.

Tests cover:
- Base rate lookup (stage × therapeutic area matrix)
- Modifier application and capping
- Confidence scoring
- Determinism
- Therapeutic area normalization
- Stage normalization
- Integration with composite scoring

Author: Wake Robin Capital Management
"""

import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

# Add parent to path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pos_prior_engine import (
    PoSPriorEngine,
    DataQualityState,
    apply_pos_weighting,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def engine() -> PoSPriorEngine:
    """Fresh engine instance for each test."""
    return PoSPriorEngine()


@pytest.fixture
def as_of_date() -> date:
    """Standard as_of_date for deterministic tests."""
    return date(2026, 1, 15)


# ============================================================================
# BASE RATE LOOKUP TESTS
# ============================================================================

class TestBaseRateLookup:
    """Tests for base PoS rate lookup."""

    def test_phase3_oncology_baseline(self, engine, as_of_date):
        """Phase 3 oncology should return correct base rate."""
        result = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "SUCCESS"
        assert result["base_pos"] == Decimal("0.439")  # From benchmarks file
        assert result["pos_prior"] == result["base_pos"]  # No modifiers

    def test_phase3_rare_disease_higher_than_oncology(self, engine, as_of_date):
        """Rare disease should have higher PoS than oncology."""
        oncology = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            as_of_date=as_of_date,
        )
        rare = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="rare disease",
            as_of_date=as_of_date,
        )

        assert rare["pos_prior"] > oncology["pos_prior"]

    def test_phase2_lower_than_phase3(self, engine, as_of_date):
        """Phase 2 PoS should be lower than Phase 3 for same indication."""
        phase2 = engine.calculate_pos_prior(
            stage="phase_2",
            therapeutic_area="oncology",
            as_of_date=as_of_date,
        )
        phase3 = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            as_of_date=as_of_date,
        )

        assert phase2["pos_prior"] < phase3["pos_prior"]

    def test_phase1_lowest(self, engine, as_of_date):
        """Phase 1 should have lowest PoS."""
        phase1 = engine.calculate_pos_prior(
            stage="phase_1",
            therapeutic_area="oncology",
            as_of_date=as_of_date,
        )
        phase2 = engine.calculate_pos_prior(
            stage="phase_2",
            therapeutic_area="oncology",
            as_of_date=as_of_date,
        )

        assert phase1["pos_prior"] < phase2["pos_prior"]

    def test_unknown_ta_falls_back_to_all_indications(self, engine, as_of_date):
        """Unknown therapeutic area should fall back to all_indications."""
        result = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="completely_unknown_area",
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "SUCCESS"
        assert result["metadata"]["ta_mapped"] == "all_indications"

    def test_no_ta_uses_all_indications(self, engine, as_of_date):
        """No therapeutic area provided should use all_indications."""
        result = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area=None,
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "SUCCESS"
        assert result["metadata"]["ta_mapped"] == "all_indications"


# ============================================================================
# MODIFIER TESTS
# ============================================================================

class TestModifiers:
    """Tests for PoS modifier application."""

    def test_orphan_designation_increases_pos(self, engine, as_of_date):
        """Orphan drug designation should increase PoS by 15%."""
        baseline = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            as_of_date=as_of_date,
        )
        with_orphan = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            orphan_drug_designation=True,
            as_of_date=as_of_date,
        )

        expected = baseline["base_pos"] * Decimal("1.15")
        assert with_orphan["pos_prior"] == expected.quantize(Decimal("0.001"))
        assert with_orphan["modifier_adjustment"] == Decimal("1.15")

    def test_breakthrough_designation_increases_pos(self, engine, as_of_date):
        """Breakthrough designation should increase PoS by 25%."""
        baseline = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            as_of_date=as_of_date,
        )
        with_breakthrough = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            breakthrough_designation=True,
            as_of_date=as_of_date,
        )

        expected = baseline["base_pos"] * Decimal("1.25")
        assert with_breakthrough["pos_prior"] == expected.quantize(Decimal("0.001"))

    def test_fast_track_increases_pos(self, engine, as_of_date):
        """Fast track designation should increase PoS by 10%."""
        baseline = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            as_of_date=as_of_date,
        )
        with_fast_track = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            fast_track_designation=True,
            as_of_date=as_of_date,
        )

        expected = baseline["base_pos"] * Decimal("1.10")
        assert with_fast_track["pos_prior"] == expected.quantize(Decimal("0.001"))

    def test_biomarker_enriched_increases_pos(self, engine, as_of_date):
        """Biomarker enriched trial should increase PoS by 20%."""
        baseline = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            as_of_date=as_of_date,
        )
        with_biomarker = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            biomarker_enriched=True,
            as_of_date=as_of_date,
        )

        expected = baseline["base_pos"] * Decimal("1.20")
        assert with_biomarker["pos_prior"] == expected.quantize(Decimal("0.001"))

    def test_multiple_modifiers_stack(self, engine, as_of_date):
        """Multiple modifiers should multiply together."""
        result = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            orphan_drug_designation=True,
            breakthrough_designation=True,
            as_of_date=as_of_date,
        )

        expected_multiplier = Decimal("1.15") * Decimal("1.25")  # 1.4375
        assert result["modifier_adjustment"] == expected_multiplier.quantize(Decimal("0.01"))

    def test_modifier_cap_at_2x(self, engine, as_of_date):
        """Total modifier adjustment should be capped at 2.0x."""
        result = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="rare disease",
            orphan_drug_designation=True,
            breakthrough_designation=True,
            fast_track_designation=True,
            biomarker_enriched=True,
            as_of_date=as_of_date,
        )

        # 1.15 * 1.25 * 1.10 * 1.20 = 1.8975, capped at 2.0
        assert result["modifier_adjustment"] <= Decimal("2.0")

    def test_pos_capped_at_95_percent(self, engine, as_of_date):
        """PoS should never exceed 0.95 (95%)."""
        # Use high base rate + all modifiers to try to exceed cap
        result = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="rare disease",  # High base rate
            orphan_drug_designation=True,
            breakthrough_designation=True,
            fast_track_designation=True,
            biomarker_enriched=True,
            as_of_date=as_of_date,
        )

        assert result["pos_prior"] <= Decimal("0.95")

    def test_modifiers_applied_list(self, engine, as_of_date):
        """Applied modifiers should be tracked in result."""
        result = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            orphan_drug_designation=True,
            breakthrough_designation=True,
            as_of_date=as_of_date,
        )

        assert "orphan_drug_designation" in result["modifiers_applied"]
        assert "breakthrough_designation" in result["modifiers_applied"]
        assert len(result["modifiers_applied"]) == 2


# ============================================================================
# STAGE NORMALIZATION TESTS
# ============================================================================

class TestStageNormalization:
    """Tests for stage name normalization."""

    @pytest.mark.parametrize("stage_input,expected_normalized", [
        ("phase_3", "phase_3"),
        ("Phase 3", "phase_3"),
        ("phase3", "phase_3"),
        ("PHASE III", "phase_3"),
        ("P3", "phase_3"),
        ("pivotal", "phase_3"),
        ("phase 2", "phase_2"),
        ("Phase II", "phase_2"),
        ("phase_1", "phase_1"),
        ("Phase I", "phase_1"),
        ("phase 1/2", "phase_1_2"),
        ("Phase I/II", "phase_1_2"),
        ("nda", "nda_bla"),
        ("BLA", "nda_bla"),
        ("submitted", "nda_bla"),
        ("commercial", "commercial"),
        ("approved", "commercial"),
    ])
    def test_stage_normalization(self, engine, as_of_date, stage_input, expected_normalized):
        """Various stage formats should normalize correctly."""
        result = engine.calculate_pos_prior(
            stage=stage_input,
            therapeutic_area="oncology",
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "SUCCESS"
        assert result["metadata"]["stage_mapped"] == expected_normalized

    def test_invalid_stage_returns_error(self, engine, as_of_date):
        """Invalid stage should return error."""
        result = engine.calculate_pos_prior(
            stage="completely_invalid_stage",
            therapeutic_area="oncology",
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "UNMAPPED_STAGE"
        assert result["pos_prior"] is None

    def test_empty_stage_returns_error(self, engine, as_of_date):
        """Empty stage should return error."""
        result = engine.calculate_pos_prior(
            stage="",
            therapeutic_area="oncology",
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "INSUFFICIENT_DATA"


# ============================================================================
# THERAPEUTIC AREA NORMALIZATION TESTS
# ============================================================================

class TestTherapeuticAreaNormalization:
    """Tests for therapeutic area normalization."""

    @pytest.mark.parametrize("ta_input,expected_normalized", [
        ("oncology", "oncology"),
        ("cancer", "oncology"),
        ("tumor", "oncology"),
        ("rare_disease", "rare_disease"),
        ("rare disease", "rare_disease"),
        ("orphan", "rare_disease"),
        ("neurology", "neurology"),
        ("cns", "neurology"),
        ("cardiovascular", "cardiovascular"),
        ("cardio", "cardiovascular"),
        ("immunology", "immunology"),
        ("autoimmune", "immunology"),
    ])
    def test_ta_normalization(self, engine, as_of_date, ta_input, expected_normalized):
        """Various TA formats should normalize correctly."""
        result = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area=ta_input,
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "SUCCESS"
        assert result["metadata"]["ta_mapped"] == expected_normalized

    def test_pattern_matching_for_complex_ta(self, engine, as_of_date):
        """Complex therapeutic area descriptions should be mapped."""
        result = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="solid tumor breast cancer HER2+",
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "SUCCESS"
        assert result["metadata"]["ta_mapped"] == "oncology"


# ============================================================================
# INTERMEDIATE STAGE TESTS
# ============================================================================

class TestIntermediateStages:
    """Tests for intermediate stage handling."""

    def test_phase_1_2_interpolates(self, engine, as_of_date):
        """Phase 1/2 should interpolate between Phase 1 and Phase 2."""
        p1 = engine.calculate_pos_prior(
            stage="phase_1",
            therapeutic_area="oncology",
            as_of_date=as_of_date,
        )
        p2 = engine.calculate_pos_prior(
            stage="phase_2",
            therapeutic_area="oncology",
            as_of_date=as_of_date,
        )
        p1_2 = engine.calculate_pos_prior(
            stage="phase_1_2",
            therapeutic_area="oncology",
            as_of_date=as_of_date,
        )

        # Should be between Phase 1 and Phase 2
        assert p1["pos_prior"] < p1_2["pos_prior"] < p2["pos_prior"]

    def test_phase_2_3_interpolates(self, engine, as_of_date):
        """Phase 2/3 should interpolate between Phase 2 and Phase 3."""
        p2 = engine.calculate_pos_prior(
            stage="phase_2",
            therapeutic_area="oncology",
            as_of_date=as_of_date,
        )
        p3 = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            as_of_date=as_of_date,
        )
        p2_3 = engine.calculate_pos_prior(
            stage="phase_2_3",
            therapeutic_area="oncology",
            as_of_date=as_of_date,
        )

        # Should be between Phase 2 and Phase 3
        assert p2["pos_prior"] < p2_3["pos_prior"] < p3["pos_prior"]

    def test_preclinical_is_half_phase1(self, engine, as_of_date):
        """Preclinical should be approximately 50% of Phase 1."""
        p1 = engine.calculate_pos_prior(
            stage="phase_1",
            therapeutic_area="oncology",
            as_of_date=as_of_date,
        )
        preclinical = engine.calculate_pos_prior(
            stage="preclinical",
            therapeutic_area="oncology",
            as_of_date=as_of_date,
        )

        expected = (p1["base_pos"] * Decimal("0.5")).quantize(Decimal("0.001"))
        assert preclinical["pos_prior"] == expected

    def test_commercial_is_100_percent(self, engine, as_of_date):
        """Commercial stage should have 100% PoS (already approved)."""
        result = engine.calculate_pos_prior(
            stage="commercial",
            therapeutic_area="oncology",
            as_of_date=as_of_date,
        )

        # Commercial base is 1.0, capped at 0.95
        assert result["pos_prior"] == Decimal("0.95")


# ============================================================================
# CONFIDENCE SCORING TESTS
# ============================================================================

class TestConfidenceScoring:
    """Tests for confidence scoring."""

    def test_high_confidence_with_known_ta(self, engine, as_of_date):
        """Known therapeutic area with good sample size should have high confidence."""
        result = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            as_of_date=as_of_date,
        )

        assert result["confidence"] >= Decimal("0.80")

    def test_lower_confidence_unknown_ta(self, engine, as_of_date):
        """Unknown therapeutic area should have lower confidence."""
        known = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            as_of_date=as_of_date,
        )
        unknown = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="completely_unknown_area_xyz",
            as_of_date=as_of_date,
        )

        assert unknown["confidence"] < known["confidence"]

    def test_many_modifiers_reduces_confidence_slightly(self, engine, as_of_date):
        """Many modifiers (>2) should slightly reduce confidence."""
        few_mods = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            orphan_drug_designation=True,
            as_of_date=as_of_date,
        )
        many_mods = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            orphan_drug_designation=True,
            breakthrough_designation=True,
            fast_track_designation=True,
            biomarker_enriched=True,
            as_of_date=as_of_date,
        )

        # Many modifiers should have slightly lower confidence
        assert many_mods["confidence"] <= few_mods["confidence"]

    def test_error_result_zero_confidence(self, engine, as_of_date):
        """Error results should have zero confidence."""
        result = engine.calculate_pos_prior(
            stage="",
            therapeutic_area="oncology",
            as_of_date=as_of_date,
        )

        assert result["confidence"] == Decimal("0.0")


# ============================================================================
# DETERMINISM TESTS
# ============================================================================

class TestDeterminism:
    """Tests ensuring identical inputs produce identical outputs."""

    def test_same_inputs_same_hash(self, engine, as_of_date):
        """Same inputs always produce same hash."""
        result1 = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            orphan_drug_designation=True,
            as_of_date=as_of_date,
        )

        engine2 = PoSPriorEngine()
        result2 = engine2.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            orphan_drug_designation=True,
            as_of_date=as_of_date,
        )

        assert result1["hash"] == result2["hash"]

    def test_same_inputs_same_pos(self, engine, as_of_date):
        """Same inputs always produce same PoS."""
        result1 = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="rare disease",
            breakthrough_designation=True,
            as_of_date=as_of_date,
        )

        engine2 = PoSPriorEngine()
        result2 = engine2.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="rare disease",
            breakthrough_designation=True,
            as_of_date=as_of_date,
        )

        assert result1["pos_prior"] == result2["pos_prior"]
        assert result1["confidence"] == result2["confidence"]

    def test_different_inputs_different_hash(self, engine, as_of_date):
        """Different inputs produce different hashes."""
        result1 = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            as_of_date=as_of_date,
        )
        result2 = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="rare disease",
            as_of_date=as_of_date,
        )

        assert result1["hash"] != result2["hash"]


# ============================================================================
# DATA QUALITY STATE TESTS
# ============================================================================

class TestDataQualityState:
    """Tests for data quality classification."""

    def test_full_quality_with_ta_and_modifiers(self, engine, as_of_date):
        """TA + modifiers = FULL quality."""
        result = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            orphan_drug_designation=True,
            as_of_date=as_of_date,
        )

        assert result["data_quality_state"] == "FULL"

    def test_partial_quality_ta_only(self, engine, as_of_date):
        """TA without modifiers = PARTIAL quality."""
        result = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            as_of_date=as_of_date,
        )

        assert result["data_quality_state"] == "PARTIAL"

    def test_partial_quality_modifiers_only(self, engine, as_of_date):
        """Modifiers without TA = PARTIAL quality."""
        result = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area=None,
            orphan_drug_designation=True,
            as_of_date=as_of_date,
        )

        assert result["data_quality_state"] == "PARTIAL"

    def test_minimal_quality_stage_only(self, engine, as_of_date):
        """Stage only (no TA, no modifiers) = MINIMAL quality."""
        result = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area=None,
            as_of_date=as_of_date,
        )

        assert result["data_quality_state"] == "MINIMAL"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Tests for integration with composite scoring."""

    def test_apply_pos_weighting_multiplies_score(self, engine, as_of_date):
        """PoS weighting should multiply stage score."""
        pos_result = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="rare disease",
            as_of_date=as_of_date,
        )

        base_score = Decimal("65.00")
        weighted = apply_pos_weighting(base_score, pos_result)

        # PoS ~0.65 -> multiplier ~0.5 + 0.65*0.8 = ~1.02
        # Weighted should be close to or slightly higher than base
        assert weighted > Decimal("0")
        assert weighted <= Decimal("100")

    def test_apply_pos_weighting_low_pos_reduces_score(self, engine, as_of_date):
        """Low PoS should reduce the score via multiplier."""
        pos_result = engine.calculate_pos_prior(
            stage="phase_1",
            therapeutic_area="neurology",  # Low PoS
            as_of_date=as_of_date,
        )

        base_score = Decimal("65.00")
        weighted = apply_pos_weighting(base_score, pos_result)

        # PoS ~0.08 -> multiplier ~0.5 + 0.08*0.8 = ~0.564
        # Should reduce the score
        assert weighted < base_score

    def test_apply_pos_weighting_high_pos_increases_score(self, engine, as_of_date):
        """High PoS should increase the score via multiplier."""
        pos_result = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="rare disease",
            orphan_drug_designation=True,
            breakthrough_designation=True,
            as_of_date=as_of_date,
        )

        base_score = Decimal("50.00")
        weighted = apply_pos_weighting(base_score, pos_result)

        # High PoS -> multiplier > 1.0
        assert weighted > base_score

    def test_apply_pos_weighting_below_confidence_no_change(self, engine, as_of_date):
        """Below confidence threshold should not apply weighting."""
        # Create result with low confidence
        pos_result = {
            "pos_prior": Decimal("0.80"),
            "confidence": Decimal("0.50"),  # Below default 0.60 threshold
        }

        base_score = Decimal("65.00")
        weighted = apply_pos_weighting(base_score, pos_result)

        assert weighted == base_score

    def test_apply_pos_weighting_none_pos_no_change(self, engine, as_of_date):
        """None PoS should not apply weighting."""
        pos_result = {
            "pos_prior": None,
            "confidence": Decimal("0.80"),
        }

        base_score = Decimal("65.00")
        weighted = apply_pos_weighting(base_score, pos_result)

        assert weighted == base_score

    def test_apply_pos_weighting_bounded_0_100(self, engine, as_of_date):
        """Weighted score should be bounded to 0-100."""
        pos_result = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="rare disease",
            orphan_drug_designation=True,
            breakthrough_designation=True,
            as_of_date=as_of_date,
        )

        # Very high base score
        base_score = Decimal("95.00")
        weighted = apply_pos_weighting(base_score, pos_result)

        assert Decimal("0") <= weighted <= Decimal("100")


# ============================================================================
# UNIVERSE SCORING TESTS
# ============================================================================

class TestUniverseScoring:
    """Tests for batch scoring of universe."""

    def test_score_universe_returns_all_tickers(self, engine, as_of_date):
        """Universe scoring returns result for each ticker."""
        universe = [
            {"ticker": "ONCO", "stage": "phase_3", "therapeutic_area": "oncology"},
            {"ticker": "RARE", "stage": "phase_3", "therapeutic_area": "rare disease"},
            {"ticker": "NEURO", "stage": "phase_2", "therapeutic_area": "neurology"},
        ]

        result = engine.score_universe(universe, as_of_date)

        assert len(result["scores"]) == 3
        tickers = [s["ticker"] for s in result["scores"]]
        assert "ONCO" in tickers
        assert "RARE" in tickers
        assert "NEURO" in tickers

    def test_score_universe_tracks_distributions(self, engine, as_of_date):
        """Universe scoring tracks stage and quality distributions."""
        universe = [
            {"ticker": "A", "stage": "phase_3", "therapeutic_area": "oncology"},
            {"ticker": "B", "stage": "phase_2", "therapeutic_area": "rare disease",
             "orphan_drug_designation": True},
        ]

        result = engine.score_universe(universe, as_of_date)

        assert "stage_distribution" in result["diagnostic_counts"]
        assert "data_quality_distribution" in result["diagnostic_counts"]

    def test_score_universe_deterministic_hash(self, engine, as_of_date):
        """Universe scoring produces deterministic content hash."""
        universe = [
            {"ticker": "TEST", "stage": "phase_3", "therapeutic_area": "oncology"},
        ]

        result1 = engine.score_universe(universe, as_of_date)

        engine2 = PoSPriorEngine()
        result2 = engine2.score_universe(universe, as_of_date)

        assert result1["provenance"]["content_hash"] == result2["provenance"]["content_hash"]

    def test_score_universe_handles_base_stage_field(self, engine, as_of_date):
        """Universe scoring should handle 'base_stage' as alias for 'stage'."""
        universe = [
            {"ticker": "ALT", "base_stage": "phase_3", "indication": "oncology"},
        ]

        result = engine.score_universe(universe, as_of_date)

        assert result["scores"][0]["reason_code"] == "SUCCESS"


# ============================================================================
# AUDIT TRAIL TESTS
# ============================================================================

class TestAuditTrail:
    """Tests for audit trail functionality."""

    def test_audit_trail_recorded(self, engine, as_of_date):
        """Each calculation is recorded in audit trail."""
        engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            as_of_date=as_of_date,
        )

        trail = engine.get_audit_trail()
        assert len(trail) == 1

    def test_audit_trail_deterministic_timestamp(self, engine, as_of_date):
        """Audit trail uses deterministic timestamp from as_of_date."""
        engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            as_of_date=as_of_date,
        )

        trail = engine.get_audit_trail()
        assert trail[0]["timestamp"] == "2026-01-15T00:00:00Z"

    def test_audit_trail_can_be_cleared(self, engine, as_of_date):
        """Audit trail can be cleared."""
        engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            as_of_date=as_of_date,
        )

        engine.clear_audit_trail()
        trail = engine.get_audit_trail()

        assert len(trail) == 0


# ============================================================================
# BENCHMARKS INFO TESTS
# ============================================================================

class TestBenchmarksInfo:
    """Tests for benchmarks information retrieval."""

    def test_get_benchmarks_info_returns_metadata(self, engine):
        """get_benchmarks_info should return metadata."""
        info = engine.get_benchmarks_info()

        assert "metadata" in info
        assert "stages_available" in info
        assert "therapeutic_areas_available" in info
        assert "modifiers_available" in info

    def test_benchmarks_has_expected_stages(self, engine):
        """Benchmarks should have expected stages."""
        info = engine.get_benchmarks_info()

        assert "phase_1" in info["stages_available"]
        assert "phase_2" in info["stages_available"]
        assert "phase_3" in info["stages_available"]

    def test_modifiers_available(self, engine):
        """All expected modifiers should be available."""
        info = engine.get_benchmarks_info()

        expected_modifiers = [
            "orphan_drug_designation",
            "breakthrough_designation",
            "fast_track_designation",
            "biomarker_enriched",
        ]

        for mod in expected_modifiers:
            assert mod in info["modifiers_available"]


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_pos_always_bounded_0_to_095(self, engine, as_of_date):
        """PoS should always be between 0 and 0.95."""
        # Test with high modifier case
        result = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="rare disease",
            orphan_drug_designation=True,
            breakthrough_designation=True,
            fast_track_designation=True,
            biomarker_enriched=True,
            as_of_date=as_of_date,
        )

        assert Decimal("0") <= result["pos_prior"] <= Decimal("0.95")

    def test_confidence_always_bounded_0_to_1(self, engine, as_of_date):
        """Confidence should always be between 0 and 1."""
        result = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            as_of_date=as_of_date,
        )

        assert Decimal("0") <= result["confidence"] <= Decimal("1")

    def test_whitespace_in_stage_handled(self, engine, as_of_date):
        """Whitespace in stage should be handled."""
        result = engine.calculate_pos_prior(
            stage="  phase 3  ",
            therapeutic_area="oncology",
            as_of_date=as_of_date,
        )

        assert result["reason_code"] == "SUCCESS"

    def test_case_insensitive_inputs(self, engine, as_of_date):
        """Inputs should be case insensitive."""
        upper = engine.calculate_pos_prior(
            stage="PHASE 3",
            therapeutic_area="ONCOLOGY",
            as_of_date=as_of_date,
        )
        lower = engine.calculate_pos_prior(
            stage="phase 3",
            therapeutic_area="oncology",
            as_of_date=as_of_date,
        )

        assert upper["pos_prior"] == lower["pos_prior"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
