#!/usr/bin/env python3
"""
Unit tests for pos_engine.py - Probability of Success Engine.

Tests:
1. Stage normalization and scoring
2. Indication normalization (word boundary matching)
3. LOA probability lookups
4. Data quality assessment
5. Score clamping and adjustments
6. Universe scoring with content hash determinism
7. Audit trail generation

Run: pytest tests/test_pos_engine.py -v
"""

import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pos_engine import (
    ProbabilityOfSuccessEngine,
    DataQualityState,
)


# ============================================================================
# ENGINE INITIALIZATION TESTS
# ============================================================================

class TestEngineInitialization:
    """Tests for PoS engine initialization."""

    def test_engine_creates_with_defaults(self):
        """Engine should initialize with default benchmarks."""
        engine = ProbabilityOfSuccessEngine()

        assert engine.VERSION == "1.2.0"
        assert len(engine.benchmarks) > 0
        assert engine.audit_trail == []

    def test_engine_loads_benchmarks(self):
        """Engine should load benchmark data for multiple phases."""
        engine = ProbabilityOfSuccessEngine()

        benchmarks = engine.get_benchmarks_info()
        assert "phase_1" in benchmarks["stages_available"]
        assert "phase_2" in benchmarks["stages_available"]
        assert "phase_3" in benchmarks["stages_available"]

    def test_engine_fallback_benchmarks(self, tmp_path):
        """Engine should use fallback when benchmarks file missing."""
        engine = ProbabilityOfSuccessEngine(
            benchmarks_path=str(tmp_path / "nonexistent.json")
        )

        # Should still work with fallback benchmarks
        result = engine.calculate_pos_score("phase_3", as_of_date=date(2026, 1, 15))
        assert result["pos_score"] >= Decimal("0")
        assert "FALLBACK" in engine.benchmarks_metadata.get("source", "")


# ============================================================================
# STAGE NORMALIZATION TESTS
# ============================================================================

class TestStageNormalization:
    """Tests for stage name normalization."""

    @pytest.fixture
    def engine(self):
        return ProbabilityOfSuccessEngine()

    def test_normalize_phase_formats(self, engine):
        """Various phase formats should normalize correctly."""
        # _normalize_stage now returns (stage, was_defaulted) tuple
        # Standard formats
        assert engine._normalize_stage("phase_1") == ("phase_1", False)
        assert engine._normalize_stage("phase_2") == ("phase_2", False)
        assert engine._normalize_stage("phase_3") == ("phase_3", False)

        # Space-separated
        assert engine._normalize_stage("Phase 1") == ("phase_1", False)
        assert engine._normalize_stage("Phase 2") == ("phase_2", False)
        assert engine._normalize_stage("Phase 3") == ("phase_3", False)

        # Roman numerals
        assert engine._normalize_stage("Phase I") == ("phase_1", False)
        assert engine._normalize_stage("Phase II") == ("phase_2", False)
        assert engine._normalize_stage("Phase III") == ("phase_3", False)

    def test_normalize_combined_phases(self, engine):
        """Combined phase formats should normalize correctly."""
        assert engine._normalize_stage("phase 1/2") == ("phase_1_2", False)
        assert engine._normalize_stage("Phase I/II") == ("phase_1_2", False)
        assert engine._normalize_stage("phase 2/3") == ("phase_2_3", False)
        assert engine._normalize_stage("Phase II/III") == ("phase_2_3", False)

    def test_normalize_special_stages(self, engine):
        """Special stage names should normalize correctly."""
        assert engine._normalize_stage("preclinical") == ("preclinical", False)
        assert engine._normalize_stage("pre-clinical") == ("preclinical", False)
        assert engine._normalize_stage("discovery") == ("preclinical", False)

        assert engine._normalize_stage("nda") == ("nda_bla", False)
        assert engine._normalize_stage("bla") == ("nda_bla", False)
        assert engine._normalize_stage("submitted") == ("nda_bla", False)

        assert engine._normalize_stage("approved") == ("commercial", False)
        assert engine._normalize_stage("commercial") == ("commercial", False)
        assert engine._normalize_stage("marketed") == ("commercial", False)

    def test_normalize_unknown_defaults_to_phase_2(self, engine):
        """Unknown stage should default to phase_2 with was_defaulted=True."""
        assert engine._normalize_stage("unknown_stage") == ("phase_2", True)
        assert engine._normalize_stage("") == ("phase_2", True)
        assert engine._normalize_stage(None) == ("phase_2", True)


# ============================================================================
# STAGE SCORE TESTS
# ============================================================================

class TestStageScoring:
    """Tests for stage score calculation."""

    @pytest.fixture
    def engine(self):
        return ProbabilityOfSuccessEngine()

    def test_stage_scores_are_ordered(self, engine):
        """Later stages should have higher scores."""
        as_of = date(2026, 1, 15)

        preclinical = engine.calculate_pos_score("preclinical", as_of_date=as_of)
        phase_1 = engine.calculate_pos_score("phase_1", as_of_date=as_of)
        phase_2 = engine.calculate_pos_score("phase_2", as_of_date=as_of)
        phase_3 = engine.calculate_pos_score("phase_3", as_of_date=as_of)
        nda = engine.calculate_pos_score("nda_bla", as_of_date=as_of)
        commercial = engine.calculate_pos_score("commercial", as_of_date=as_of)

        assert preclinical["stage_score"] < phase_1["stage_score"]
        assert phase_1["stage_score"] < phase_2["stage_score"]
        assert phase_2["stage_score"] < phase_3["stage_score"]
        assert phase_3["stage_score"] < nda["stage_score"]
        assert nda["stage_score"] < commercial["stage_score"]

    def test_stage_score_values(self, engine):
        """Stage scores should match expected values."""
        as_of = date(2026, 1, 15)

        assert engine.calculate_pos_score("preclinical", as_of_date=as_of)["stage_score"] == Decimal("10")
        assert engine.calculate_pos_score("phase_1", as_of_date=as_of)["stage_score"] == Decimal("20")
        assert engine.calculate_pos_score("phase_2", as_of_date=as_of)["stage_score"] == Decimal("40")
        assert engine.calculate_pos_score("phase_3", as_of_date=as_of)["stage_score"] == Decimal("65")
        assert engine.calculate_pos_score("nda_bla", as_of_date=as_of)["stage_score"] == Decimal("80")
        assert engine.calculate_pos_score("commercial", as_of_date=as_of)["stage_score"] == Decimal("90")


# ============================================================================
# INDICATION NORMALIZATION TESTS
# ============================================================================

class TestIndicationNormalization:
    """Tests for indication normalization with word-boundary matching."""

    @pytest.fixture
    def engine(self):
        return ProbabilityOfSuccessEngine()

    def test_oncology_matching(self, engine):
        """Oncology-related terms should match correctly."""
        assert engine._normalize_indication("oncology") == "oncology"
        assert engine._normalize_indication("Breast Cancer") == "oncology"
        assert engine._normalize_indication("solid tumor treatment") == "oncology"
        assert engine._normalize_indication("Non-Small Cell Lung Carcinoma") == "oncology"
        assert engine._normalize_indication("Multiple Myeloma") == "oncology"
        assert engine._normalize_indication("Acute Lymphoblastic Leukemia") == "oncology"

    def test_rare_disease_matching(self, engine):
        """Rare disease terms should match correctly."""
        assert engine._normalize_indication("rare disease") == "rare_disease"
        assert engine._normalize_indication("Rare Genetic Disorder") == "rare_disease"
        assert engine._normalize_indication("Orphan Drug Indication") == "rare_disease"
        assert engine._normalize_indication("ultra-rare condition") == "rare_disease"

    def test_word_boundary_prevents_false_positives(self, engine):
        """Word boundary matching should prevent false positives."""
        # "dose" contains "os" but should NOT match oncology patterns
        result = engine._normalize_indication("dose escalation study")
        assert result != "oncology"

        # "share" contains "are" but should NOT match "rare"
        result = engine._normalize_indication("market share analysis")
        assert result != "rare_disease"

        # "caring" contains "car" but should NOT match cardiovascular
        result = engine._normalize_indication("patient caring protocol")
        assert result != "cardiovascular"

    def test_neurology_matching(self, engine):
        """Neurology-related terms should match correctly."""
        assert engine._normalize_indication("neurology") == "neurology"
        assert engine._normalize_indication("Alzheimer's Disease") == "neurology"
        assert engine._normalize_indication("Parkinson's Disease") == "neurology"
        assert engine._normalize_indication("ALS Treatment") == "neurology"
        assert engine._normalize_indication("Multiple Sclerosis") == "neurology"

    def test_other_categories(self, engine):
        """Other indication categories should match correctly."""
        assert engine._normalize_indication("cardiovascular disease") == "cardiovascular"
        assert engine._normalize_indication("Rheumatoid Arthritis") == "immunology"
        assert engine._normalize_indication("Type 2 Diabetes") == "metabolic"
        assert engine._normalize_indication("COPD Treatment") == "respiratory"
        assert engine._normalize_indication("Atopic Dermatitis") == "dermatology"
        assert engine._normalize_indication("Macular Degeneration") == "ophthalmology"
        assert engine._normalize_indication("HIV Prevention") == "infectious_disease"

    def test_unknown_indication_returns_all_indications(self, engine):
        """Unknown indications should return 'all_indications'."""
        assert engine._normalize_indication("unknown condition xyz") == "all_indications"
        assert engine._normalize_indication("general study") == "all_indications"

    def test_none_indication_returns_none(self, engine):
        """None indication should return None."""
        assert engine._normalize_indication(None) is None
        assert engine._normalize_indication("") is None


# ============================================================================
# POS SCORE DIFFERENTIATION TESTS
# ============================================================================

class TestPosDifferentiation:
    """Tests for PoS score differentiation by indication."""

    @pytest.fixture
    def engine(self):
        return ProbabilityOfSuccessEngine()

    def test_oncology_vs_rare_disease_phase_3(self, engine):
        """Rare disease should have higher PoS than oncology at same stage."""
        as_of = date(2026, 1, 15)

        oncology = engine.calculate_pos_score(
            "phase_3", indication="oncology", as_of_date=as_of
        )
        rare = engine.calculate_pos_score(
            "phase_3", indication="rare disease", as_of_date=as_of
        )

        # Rare disease typically has higher approval rates
        assert rare["pos_score"] > oncology["pos_score"]

    def test_indication_affects_pos_not_stage_score(self, engine):
        """Indication should affect PoS score but not stage score."""
        as_of = date(2026, 1, 15)

        oncology = engine.calculate_pos_score(
            "phase_3", indication="oncology", as_of_date=as_of
        )
        rare = engine.calculate_pos_score(
            "phase_3", indication="rare disease", as_of_date=as_of
        )

        # Stage scores should be equal (same stage)
        assert oncology["stage_score"] == rare["stage_score"]
        # PoS scores should differ (different indications)
        assert oncology["pos_score"] != rare["pos_score"]


# ============================================================================
# DATA QUALITY ASSESSMENT TESTS
# ============================================================================

class TestDataQualityAssessment:
    """Tests for data quality state assessment."""

    @pytest.fixture
    def engine(self):
        return ProbabilityOfSuccessEngine()

    def test_full_data_quality(self, engine):
        """All fields present should return FULL quality."""
        result = engine.calculate_pos_score(
            "phase_3",
            indication="oncology",
            trial_design_quality=Decimal("1.1"),
            competitive_intensity=Decimal("0.9"),
            as_of_date=date(2026, 1, 15)
        )
        assert result["data_quality_state"] == "FULL"

    def test_partial_data_quality(self, engine):
        """Some optional fields missing should return PARTIAL."""
        result = engine.calculate_pos_score(
            "phase_3",
            indication="oncology",
            as_of_date=date(2026, 1, 15)
        )
        # Missing trial_design_quality and competitive_intensity
        assert result["data_quality_state"] == "PARTIAL"

    def test_missing_fields_tracked(self, engine):
        """Missing fields should be tracked in result."""
        result = engine.calculate_pos_score(
            "phase_3",
            as_of_date=date(2026, 1, 15)
        )

        assert "trial_design_quality" in result["missing_fields"]
        assert "competitive_intensity" in result["missing_fields"]


# ============================================================================
# ADJUSTMENT CLAMPING TESTS
# ============================================================================

class TestAdjustmentClamping:
    """Tests for adjustment value clamping."""

    @pytest.fixture
    def engine(self):
        return ProbabilityOfSuccessEngine()

    def test_trial_design_quality_clamped_high(self, engine):
        """Trial design quality above 1.30 should be clamped."""
        result = engine.calculate_pos_score(
            "phase_3",
            trial_design_quality=Decimal("2.0"),
            as_of_date=date(2026, 1, 15)
        )

        audit = result["audit_entry"]
        assert audit["calculation"]["adjustments_applied"]["trial_design_quality"] == "1.30"

    def test_trial_design_quality_clamped_low(self, engine):
        """Trial design quality below 0.70 should be clamped."""
        result = engine.calculate_pos_score(
            "phase_3",
            trial_design_quality=Decimal("0.5"),
            as_of_date=date(2026, 1, 15)
        )

        audit = result["audit_entry"]
        assert audit["calculation"]["adjustments_applied"]["trial_design_quality"] == "0.70"

    def test_competitive_intensity_clamped(self, engine):
        """Competitive intensity should be clamped to 0.70-1.00."""
        result = engine.calculate_pos_score(
            "phase_3",
            competitive_intensity=Decimal("0.5"),
            as_of_date=date(2026, 1, 15)
        )

        audit = result["audit_entry"]
        assert audit["calculation"]["adjustments_applied"]["competitive_intensity"] == "0.70"


# ============================================================================
# SCORE BOUNDING TESTS
# ============================================================================

class TestScoreBounding:
    """Tests for score value bounding."""

    @pytest.fixture
    def engine(self):
        return ProbabilityOfSuccessEngine()

    def test_pos_score_bounded_0_100(self, engine):
        """PoS score should always be between 0 and 100."""
        as_of = date(2026, 1, 15)

        # Test various combinations
        for stage in ["preclinical", "phase_1", "phase_2", "phase_3", "nda_bla", "commercial"]:
            for indication in ["oncology", "rare disease", "neurology", None]:
                result = engine.calculate_pos_score(
                    stage, indication=indication, as_of_date=as_of
                )
                assert Decimal("0") <= result["pos_score"] <= Decimal("100")

    def test_stage_score_bounded_0_100(self, engine):
        """Stage score should always be between 0 and 100."""
        as_of = date(2026, 1, 15)

        for stage in ["preclinical", "phase_1", "phase_2", "phase_3", "nda_bla", "commercial"]:
            result = engine.calculate_pos_score(stage, as_of_date=as_of)
            assert Decimal("0") <= result["stage_score"] <= Decimal("100")


# ============================================================================
# UNIVERSE SCORING TESTS
# ============================================================================

class TestUniverseScoring:
    """Tests for scoring an entire universe."""

    @pytest.fixture
    def engine(self):
        return ProbabilityOfSuccessEngine()

    def test_score_universe_returns_all_tickers(self, engine, pos_universe, as_of_date):
        """Universe scoring should return scores for all tickers."""
        result = engine.score_universe(pos_universe, as_of_date)

        assert len(result["scores"]) == len(pos_universe)
        tickers_scored = {s["ticker"] for s in result["scores"]}
        tickers_input = {u["ticker"] for u in pos_universe}
        assert tickers_scored == tickers_input

    def test_score_universe_tracks_indication_coverage(self, engine, pos_universe, as_of_date):
        """Universe scoring should track indication coverage."""
        result = engine.score_universe(pos_universe, as_of_date)

        diag = result["diagnostic_counts"]
        assert "indication_coverage" in diag
        assert "indication_coverage_pct" in diag
        # 4 of 5 tickers have indications
        assert diag["indication_coverage"] == 4

    def test_score_universe_deterministic_hash(self, engine, pos_universe, as_of_date):
        """Two runs with same input should produce same content hash."""
        result1 = engine.score_universe(pos_universe, as_of_date)
        result2 = engine.score_universe(pos_universe, as_of_date)

        assert result1["provenance"]["content_hash"] == result2["provenance"]["content_hash"]

    def test_score_universe_includes_provenance(self, engine, pos_universe, as_of_date):
        """Universe result should include provenance metadata."""
        result = engine.score_universe(pos_universe, as_of_date)

        assert "provenance" in result
        assert result["provenance"]["module"] == "pos_engine"
        assert result["provenance"]["module_version"] == engine.VERSION
        assert result["provenance"]["pit_cutoff"] == as_of_date.isoformat()


# ============================================================================
# AUDIT TRAIL TESTS
# ============================================================================

class TestAuditTrail:
    """Tests for audit trail functionality."""

    @pytest.fixture
    def engine(self):
        return ProbabilityOfSuccessEngine()

    def test_audit_trail_populated(self, engine):
        """Calculating scores should add entries to audit trail."""
        engine.calculate_pos_score("phase_3", as_of_date=date(2026, 1, 15))
        engine.calculate_pos_score("phase_2", as_of_date=date(2026, 1, 15))

        trail = engine.get_audit_trail()
        assert len(trail) == 2

    def test_audit_trail_contains_deterministic_timestamp(self, engine):
        """Audit entry should have deterministic timestamp from as_of_date."""
        result = engine.calculate_pos_score("phase_3", as_of_date=date(2026, 1, 15))

        audit = result["audit_entry"]
        assert audit["timestamp"] == "2026-01-15T00:00:00Z"

    def test_audit_trail_contains_inputs_hash(self, engine):
        """Audit entry should contain deterministic inputs hash."""
        result = engine.calculate_pos_score(
            "phase_3", indication="oncology", as_of_date=date(2026, 1, 15)
        )

        audit = result["audit_entry"]
        assert "inputs_hash" in audit
        assert len(audit["inputs_hash"]) == 16  # 16-char hex

    def test_clear_audit_trail(self, engine):
        """Clearing audit trail should remove all entries."""
        engine.calculate_pos_score("phase_3", as_of_date=date(2026, 1, 15))
        assert len(engine.get_audit_trail()) == 1

        engine.clear_audit_trail()
        assert len(engine.get_audit_trail()) == 0


# ============================================================================
# LOA PROBABILITY TESTS
# ============================================================================

class TestLoaProbability:
    """Tests for Likelihood of Approval probability."""

    @pytest.fixture
    def engine(self):
        return ProbabilityOfSuccessEngine()

    def test_loa_probability_bounded_0_1(self, engine):
        """LOA probability should always be between 0 and 1."""
        as_of = date(2026, 1, 15)

        for stage in ["phase_1", "phase_2", "phase_3", "nda_bla"]:
            result = engine.calculate_pos_score(stage, as_of_date=as_of)
            assert Decimal("0") <= result["loa_probability"] <= Decimal("1")

    def test_loa_increases_with_stage(self, engine):
        """LOA probability should generally increase with later stages."""
        as_of = date(2026, 1, 15)

        phase_1 = engine.calculate_pos_score("phase_1", as_of_date=as_of)
        phase_2 = engine.calculate_pos_score("phase_2", as_of_date=as_of)
        phase_3 = engine.calculate_pos_score("phase_3", as_of_date=as_of)
        nda = engine.calculate_pos_score("nda_bla", as_of_date=as_of)

        assert phase_1["loa_probability"] < phase_2["loa_probability"]
        assert phase_2["loa_probability"] < phase_3["loa_probability"]
        assert phase_3["loa_probability"] < nda["loa_probability"]

    def test_loa_provenance_tracked(self, engine):
        """LOA provenance should indicate data source."""
        result = engine.calculate_pos_score(
            "phase_3", indication="oncology", as_of_date=date(2026, 1, 15)
        )

        assert "loa_provenance" in result
        assert "BIO" in result["loa_provenance"] or "fallback" in result["loa_provenance"]


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def engine(self):
        return ProbabilityOfSuccessEngine()

    def test_empty_stage_string(self, engine):
        """Empty stage should default to phase_2."""
        result = engine.calculate_pos_score("", as_of_date=date(2026, 1, 15))
        assert result["stage_normalized"] == "phase_2"

    def test_whitespace_handling(self, engine):
        """Whitespace in inputs should be handled."""
        result = engine.calculate_pos_score(
            "  Phase 3  ",
            indication="  oncology  ",
            as_of_date=date(2026, 1, 15)
        )
        assert result["stage_normalized"] == "phase_3"
        assert result["indication_normalized"] == "oncology"

    def test_commercial_stage_full_loa(self, engine):
        """Commercial stage should have LOA = 1.0 (already approved)."""
        result = engine.calculate_pos_score("commercial", as_of_date=date(2026, 1, 15))
        assert result["loa_probability"] == Decimal("1.0")
        assert result["loa_provenance"] == "commercial_approved"


# ============================================================================
# CONFIDENCE GATING TESTS
# ============================================================================

class TestConfidenceGating:
    """Tests for PoS confidence gating to prevent silent contribution from unknown stages."""

    @pytest.fixture
    def engine(self):
        return ProbabilityOfSuccessEngine()

    def test_defaulted_stage_has_low_confidence(self, engine):
        """Unknown/empty stage should have low confidence (below 0.40 gating threshold)."""
        as_of = date(2026, 1, 15)

        # Empty string
        result = engine.calculate_pos_score("", as_of_date=as_of)
        assert result["stage_was_defaulted"] is True
        assert result["pos_confidence"] == Decimal("0.30")
        assert result["confidence_reason"] == "stage_defaulted"

        # Unknown stage
        result = engine.calculate_pos_score("unknown_xyz", as_of_date=as_of)
        assert result["stage_was_defaulted"] is True
        assert result["pos_confidence"] == Decimal("0.30")

    def test_known_stage_has_higher_confidence(self, engine):
        """Known stage should have higher confidence."""
        as_of = date(2026, 1, 15)

        result = engine.calculate_pos_score("phase_3", as_of_date=as_of)
        assert result["stage_was_defaulted"] is False
        # Without indication, confidence is medium
        assert result["pos_confidence"] == Decimal("0.55")
        assert result["confidence_reason"] == "stage_known_indication_unknown"

    def test_known_stage_and_indication_has_high_confidence(self, engine):
        """Known stage + known indication should have high confidence."""
        as_of = date(2026, 1, 15)

        result = engine.calculate_pos_score("phase_3", indication="oncology", as_of_date=as_of)
        assert result["stage_was_defaulted"] is False
        assert result["pos_confidence"] == Decimal("0.70")
        assert result["confidence_reason"] == "stage_and_indication_known"

    def test_confidence_below_gating_threshold_flagged(self, engine):
        """Universe scoring should flag tickers below gating threshold."""
        as_of = date(2026, 1, 15)
        universe = [
            {"ticker": "KNOWN", "base_stage": "phase_3", "indication": "oncology"},
            {"ticker": "UNKNOWN", "base_stage": ""},
        ]

        result = engine.score_universe(universe, as_of)

        # Check flags
        known_score = next(s for s in result["scores"] if s["ticker"] == "KNOWN")
        unknown_score = next(s for s in result["scores"] if s["ticker"] == "UNKNOWN")

        assert "below_confidence_gate" not in known_score["flags"]
        assert "below_confidence_gate" in unknown_score["flags"]
        assert "stage_defaulted" in unknown_score["flags"]

    def test_effective_coverage_tracked(self, engine):
        """Universe scoring should track effective coverage (above gating threshold)."""
        as_of = date(2026, 1, 15)
        universe = [
            {"ticker": "HIGH_CONF", "base_stage": "phase_3", "indication": "oncology"},
            {"ticker": "MED_CONF", "base_stage": "phase_2"},
            {"ticker": "LOW_CONF_1", "base_stage": "unknown"},
            {"ticker": "LOW_CONF_2", "base_stage": ""},
        ]

        result = engine.score_universe(universe, as_of)
        diag = result["diagnostic_counts"]

        # 2 are above 0.40 threshold (HIGH_CONF=0.70, MED_CONF=0.55)
        # 2 are below (LOW_CONF_1=0.30, LOW_CONF_2=0.30)
        assert diag["effective_coverage"] == 2
        assert diag["effective_coverage_pct"] == "50.0%"
        assert diag["stage_defaulted_count"] == 2
        assert diag["confidence_distribution"]["high"] == 1
        assert diag["confidence_distribution"]["medium"] == 1
        assert diag["confidence_distribution"]["low"] == 2


# ============================================================================
# DETERMINISM AND PIT DISCIPLINE TESTS
# ============================================================================

class TestDeterminismRequirements:
    """Tests for determinism and PIT discipline enforcement."""

    @pytest.fixture
    def engine(self):
        return ProbabilityOfSuccessEngine()

    def test_as_of_date_required_raises_on_none(self, engine):
        """as_of_date=None should raise ValueError (no silent defaults)."""
        with pytest.raises(ValueError) as exc_info:
            engine.calculate_pos_score("phase_3", as_of_date=None)

        assert "as_of_date is required" in str(exc_info.value)
        assert "deterministic" in str(exc_info.value).lower()

    def test_same_inputs_produce_identical_outputs(self, engine):
        """Same inputs must produce byte-identical outputs (determinism test)."""
        as_of = date(2026, 1, 15)

        result1 = engine.calculate_pos_score(
            "phase_3", indication="oncology", as_of_date=as_of
        )
        result2 = engine.calculate_pos_score(
            "phase_3", indication="oncology", as_of_date=as_of
        )

        # Compare key fields for determinism
        assert result1["pos_score"] == result2["pos_score"]
        assert result1["stage_score"] == result2["stage_score"]
        assert result1["loa_probability"] == result2["loa_probability"]
        assert result1["audit_entry"]["inputs_hash"] == result2["audit_entry"]["inputs_hash"]


# ============================================================================
# MISSING STAGE DETECTION TESTS
# ============================================================================

class TestMissingStageDetection:
    """Tests for proper detection of missing base_stage in score_universe()."""

    @pytest.fixture
    def engine(self):
        return ProbabilityOfSuccessEngine()

    def test_missing_base_stage_flagged_correctly(self, engine):
        """Company with no base_stage should have stage_was_defaulted=True."""
        as_of = date(2026, 1, 15)

        # Company with no base_stage key at all
        universe = [{"ticker": "NOSTAGE", "indication": "oncology"}]
        result = engine.score_universe(universe, as_of)

        score = result["scores"][0]
        assert score["stage_was_defaulted"] is True
        assert score["pos_confidence"] == Decimal("0.30")  # Low confidence
        assert "stage_defaulted" in score["flags"]
        assert "below_confidence_gate" in score["flags"]

    def test_none_base_stage_flagged_correctly(self, engine):
        """Company with base_stage=None should have stage_was_defaulted=True."""
        as_of = date(2026, 1, 15)

        universe = [{"ticker": "NULLSTAGE", "base_stage": None, "indication": "oncology"}]
        result = engine.score_universe(universe, as_of)

        score = result["scores"][0]
        assert score["stage_was_defaulted"] is True
        assert score["pos_confidence"] == Decimal("0.30")

    def test_provided_stage_not_marked_defaulted(self, engine):
        """Company with valid base_stage should NOT have stage_was_defaulted."""
        as_of = date(2026, 1, 15)

        universe = [{"ticker": "HASSTAGE", "base_stage": "phase_3", "indication": "oncology"}]
        result = engine.score_universe(universe, as_of)

        score = result["scores"][0]
        assert score["stage_was_defaulted"] is False
        assert score["pos_confidence"] == Decimal("0.70")  # High confidence
        assert "stage_defaulted" not in score["flags"]


# ============================================================================
# INDICATION PARSE FALLBACK TESTS
# ============================================================================

class TestIndicationParseFallback:
    """Tests for indication parse fallback detection."""

    @pytest.fixture
    def engine(self):
        return ProbabilityOfSuccessEngine()

    def test_unknown_indication_flags_parse_fallback(self, engine):
        """Unknown indication should flag indication_parse_fallback."""
        result = engine.calculate_pos_score(
            "phase_3",
            indication="some unknown therapy xyz",
            as_of_date=date(2026, 1, 15)
        )

        assert result["indication_normalized"] == "all_indications"
        assert result["inputs_used"].get("indication_parse_fallback") is True

    def test_valid_indication_no_fallback_flag(self, engine):
        """Valid indication should NOT have parse_fallback flag."""
        result = engine.calculate_pos_score(
            "phase_3",
            indication="oncology",
            as_of_date=date(2026, 1, 15)
        )

        assert result["indication_normalized"] == "oncology"
        assert "indication_parse_fallback" not in result["inputs_used"]

    def test_none_indication_no_fallback_flag(self, engine):
        """None indication should NOT have parse_fallback flag (expected behavior)."""
        result = engine.calculate_pos_score(
            "phase_3",
            indication=None,
            as_of_date=date(2026, 1, 15)
        )

        assert result["indication_normalized"] is None
        assert "indication_parse_fallback" not in result["inputs_used"]


# ============================================================================
# STRICT MODE BENCHMARK TESTS
# ============================================================================

class TestStrictModeBenchmarks:
    """Tests for strict mode benchmark loading."""

    def test_strict_mode_raises_on_missing_file(self, tmp_path):
        """strict=True should raise FileNotFoundError when benchmarks file missing."""
        with pytest.raises(FileNotFoundError) as exc_info:
            ProbabilityOfSuccessEngine(
                benchmarks_path=str(tmp_path / "nonexistent.json"),
                strict=True
            )

        assert "Benchmark file not found" in str(exc_info.value)
        assert "strict=False" in str(exc_info.value)

    def test_strict_mode_raises_on_invalid_json(self, tmp_path):
        """strict=True should raise ValueError when benchmark file is invalid JSON."""
        # Create invalid JSON file
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("{ not valid json }")

        with pytest.raises(ValueError) as exc_info:
            ProbabilityOfSuccessEngine(
                benchmarks_path=str(invalid_file),
                strict=True
            )

        assert "Invalid JSON" in str(exc_info.value)

    def test_non_strict_mode_falls_back_silently(self, tmp_path):
        """strict=False should use fallback benchmarks when file missing."""
        engine = ProbabilityOfSuccessEngine(
            benchmarks_path=str(tmp_path / "nonexistent.json"),
            strict=False  # Default behavior
        )

        # Should still work
        result = engine.calculate_pos_score("phase_3", as_of_date=date(2026, 1, 15))
        assert result["pos_score"] >= Decimal("0")

        # Should indicate fallback in metadata
        assert engine.benchmarks_metadata.get("source") == "FALLBACK_HARDCODED"
        assert "warning" in engine.benchmarks_metadata

    def test_fallback_benchmarks_labeled_in_provenance(self, tmp_path):
        """Fallback benchmarks should be clearly labeled in provenance."""
        engine = ProbabilityOfSuccessEngine(
            benchmarks_path=str(tmp_path / "nonexistent.json"),
            strict=False
        )

        result = engine.calculate_pos_score("phase_3", as_of_date=date(2026, 1, 15))

        # Audit entry should indicate fallback source
        assert result["audit_entry"]["benchmarks_source"] == "FALLBACK_HARDCODED"

    def test_valid_benchmarks_loads_without_error(self):
        """Default benchmarks path should load without error in strict mode."""
        # This tests that the real benchmark file exists and is valid
        try:
            engine = ProbabilityOfSuccessEngine(strict=True)
            # Should have loaded real benchmarks
            assert engine.benchmarks_metadata.get("source") != "FALLBACK_HARDCODED"
        except FileNotFoundError:
            # If running in environment without benchmark file, skip
            pytest.skip("Benchmark file not available in test environment")
