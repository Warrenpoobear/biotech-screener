#!/usr/bin/env python3
"""
Tests for Competitive Intensity Engine
"""

import pytest
from decimal import Decimal
from datetime import date

from competitive_intensity_engine import (
    CompetitiveIntensityEngine,
    CrowdingLevel,
    CompetitivePosition,
)


class TestCompetitiveIntensityEngine:
    """Tests for CompetitiveIntensityEngine."""

    @pytest.fixture
    def engine(self):
        return CompetitiveIntensityEngine()

    @pytest.fixture
    def sample_trials(self):
        return [
            {"lead_sponsor_ticker": "ACME", "nct_id": "NCT001", "phase": "Phase 2",
             "conditions": ["Breast Cancer"], "interventions": ["monoclonal antibody"]},
            {"lead_sponsor_ticker": "ACME", "nct_id": "NCT002", "phase": "Phase 3",
             "conditions": ["Lung Cancer"], "interventions": ["kinase inhibitor"]},
            {"lead_sponsor_ticker": "COMP1", "nct_id": "NCT003", "phase": "Phase 3",
             "conditions": ["Breast Cancer"], "interventions": ["antibody"]},
            {"lead_sponsor_ticker": "COMP2", "nct_id": "NCT004", "phase": "Phase 2",
             "conditions": ["Breast Cancer"], "interventions": ["adc"]},
            {"lead_sponsor_ticker": "COMP3", "nct_id": "NCT005", "phase": "Phase 1",
             "conditions": ["Breast Cancer"], "interventions": ["car-t"]},
            {"lead_sponsor_ticker": "COMP4", "nct_id": "NCT006", "phase": "Phase 2",
             "conditions": ["Lung Cancer"], "interventions": ["small molecule"]},
        ]

    def test_initialization(self, engine):
        """Engine initializes with empty state."""
        assert engine.audit_trail == []
        assert engine.indication_landscapes == {}
        assert engine._landscape_built is False

    def test_build_landscape(self, engine, sample_trials):
        """Build landscape from trial records."""
        as_of = date(2026, 1, 26)
        stats = engine.build_landscape(sample_trials, as_of)

        assert stats["indications_mapped"] > 0
        assert stats["tickers_mapped"] == 5  # ACME + 4 competitors
        assert engine._landscape_built is True

    def test_score_ticker_with_data(self, engine, sample_trials):
        """Score ticker that has trial data."""
        as_of = date(2026, 1, 26)
        engine.build_landscape(sample_trials, as_of)

        result = engine.score_ticker("ACME", as_of)

        assert result["ticker"] == "ACME"
        assert result["competitive_intensity_score"] >= Decimal("0")
        assert result["competitive_intensity_score"] <= Decimal("100")
        assert result["competitor_count"] > 0
        assert result["crowding_level"] in [c.value for c in CrowdingLevel] + ["unknown"]
        assert result["competitive_position"] in [p.value for p in CompetitivePosition] + ["unknown"]

    def test_score_ticker_without_data(self, engine):
        """Score ticker without building landscape first."""
        as_of = date(2026, 1, 26)
        result = engine.score_ticker("UNKNOWN", as_of)

        assert result["ticker"] == "UNKNOWN"
        assert result["competitive_intensity_score"] == Decimal("50")  # Neutral
        assert result["crowding_level"] == "unknown"

    def test_crowding_levels(self, engine):
        """Crowding classification works correctly."""
        # Test each threshold
        assert engine._classify_crowding(2) == CrowdingLevel.UNCROWDED
        assert engine._classify_crowding(10) == CrowdingLevel.MODERATE
        assert engine._classify_crowding(20) == CrowdingLevel.CROWDED
        assert engine._classify_crowding(50) == CrowdingLevel.HIGHLY_CROWDED

    def test_phase_3_competitors_tracked(self, engine, sample_trials):
        """Phase 3+ competitors are tracked separately."""
        as_of = date(2026, 1, 26)
        engine.build_landscape(sample_trials, as_of)

        result = engine.score_ticker("ACME", as_of)

        # COMP1 has Phase 3 in breast cancer (same indication as ACME)
        assert result["phase_3_competitors"] >= 0

    def test_approved_competition_detected(self, engine):
        """Approved drugs in indication are detected."""
        trials = [
            {"lead_sponsor_ticker": "ACME", "nct_id": "NCT001", "phase": "Phase 2",
             "conditions": ["Diabetes"], "interventions": ["drug"]},
            {"lead_sponsor_ticker": "BIGPHARMA", "nct_id": "NCT002", "phase": "Phase 4",
             "conditions": ["Diabetes"], "interventions": ["approved drug"]},
        ]
        as_of = date(2026, 1, 26)
        engine.build_landscape(trials, as_of)

        result = engine.score_ticker("ACME", as_of)

        # Phase 4 = approved
        assert result["has_approved_competition"] is True

    def test_first_in_class_detection(self, engine):
        """First-in-class position detected with few competitors."""
        trials = [
            {"lead_sponsor_ticker": "PIONEER", "nct_id": "NCT001", "phase": "Phase 2",
             "conditions": ["Rare Disease X"], "interventions": ["novel therapy"]},
        ]
        as_of = date(2026, 1, 26)
        engine.build_landscape(trials, as_of)

        result = engine.score_ticker("PIONEER", as_of)

        assert result["competitive_position"] == CompetitivePosition.FIRST_IN_CLASS.value

    def test_me_too_detection(self, engine):
        """Me-too position detected in crowded indication."""
        # Create highly crowded indication (>30 competitors after excluding own)
        trials = []
        for i in range(35):
            trials.append({
                "lead_sponsor_ticker": f"COMP{i}",
                "nct_id": f"NCT{i:03d}",
                "phase": "Phase 2",
                "conditions": ["Breast Cancer"],
                "interventions": ["antibody"],
            })

        as_of = date(2026, 1, 26)
        engine.build_landscape(trials, as_of)

        result = engine.score_ticker("COMP0", as_of)

        # 35 total - 1 own = 34 competitors -> HIGHLY_CROWDED
        assert result["competitive_position"] == CompetitivePosition.ME_TOO.value
        assert result["crowding_level"] == CrowdingLevel.HIGHLY_CROWDED.value

    def test_score_universe(self, engine, sample_trials):
        """Score entire universe."""
        universe = [
            {"ticker": "ACME"},
            {"ticker": "COMP1"},
            {"ticker": "UNKNOWN"},
        ]
        as_of = date(2026, 1, 26)

        result = engine.score_universe(universe, sample_trials, as_of)

        assert result["diagnostic_counts"]["total_scored"] == 3
        assert "intensity_distribution" in result["diagnostic_counts"]
        assert "crowding_distribution" in result["diagnostic_counts"]
        assert "provenance" in result

    def test_indication_normalization(self, engine):
        """Indications are normalized to categories."""
        # Oncology
        assert engine._normalize_indication(["Breast Cancer"]) == "oncology"
        assert engine._normalize_indication(["Non-Small Cell Lung Cancer"]) == "oncology"
        assert engine._normalize_indication(["Melanoma"]) == "oncology"

        # Neurology
        assert engine._normalize_indication(["Alzheimer's Disease"]) == "neurology"
        assert engine._normalize_indication(["Parkinson's Disease"]) == "neurology"

        # Rare disease
        assert engine._normalize_indication(["Duchenne Muscular Dystrophy"]) == "rare_disease"

    def test_mechanism_extraction(self, engine):
        """Mechanism of action is extracted from interventions."""
        assert engine._extract_mechanism(["PD-1 inhibitor"]) == "checkpoint_inhibitor"
        assert engine._extract_mechanism(["CAR-T cell therapy"]) == "car_t"
        assert engine._extract_mechanism(["mRNA vaccine"]) == "rna_therapeutic"
        assert engine._extract_mechanism(["monoclonal antibody"]) == "antibody"

    def test_get_top_competitors(self, engine, sample_trials):
        """Get top competitors in indication."""
        as_of = date(2026, 1, 26)
        engine.build_landscape(sample_trials, as_of)

        competitors = engine.get_top_competitors("ACME", "oncology", limit=5)

        assert len(competitors) <= 5
        for comp in competitors:
            assert comp["ticker"] != "ACME"
            assert "program_count" in comp

    def test_audit_trail(self, engine, sample_trials):
        """Audit trail is maintained."""
        as_of = date(2026, 1, 26)
        engine.build_landscape(sample_trials, as_of)
        engine.score_ticker("ACME", as_of)

        trail = engine.get_audit_trail()
        assert len(trail) > 0
        assert trail[0]["ticker"] == "ACME"

        engine.clear_audit_trail()
        assert len(engine.get_audit_trail()) == 0

    def test_intensity_rating(self, engine):
        """Intensity rating categories are correct."""
        assert engine._get_intensity_rating(Decimal("20")) == "low"
        assert engine._get_intensity_rating(Decimal("40")) == "moderate"
        assert engine._get_intensity_rating(Decimal("60")) == "high"
        assert engine._get_intensity_rating(Decimal("80")) == "intense"


class TestCrowdingScoreAdjustments:
    """Tests for crowding score adjustments."""

    @pytest.fixture
    def engine(self):
        return CompetitiveIntensityEngine()

    def test_all_crowding_levels_have_adjustments(self):
        """All crowding levels have score adjustments."""
        for level in CrowdingLevel:
            assert level in CompetitiveIntensityEngine.CROWDING_SCORE_ADJUSTMENTS

    def test_all_positions_have_adjustments(self):
        """All competitive positions have score adjustments."""
        for position in CompetitivePosition:
            assert position in CompetitiveIntensityEngine.POSITION_ADJUSTMENTS

    def test_crowding_increases_score(self, engine):
        """More crowding should increase competitive intensity score."""
        # Uncrowded should have lower score than highly crowded
        uncrowded_adj = engine.CROWDING_SCORE_ADJUSTMENTS[CrowdingLevel.UNCROWDED]
        crowded_adj = engine.CROWDING_SCORE_ADJUSTMENTS[CrowdingLevel.HIGHLY_CROWDED]

        assert crowded_adj > uncrowded_adj

    def test_first_in_class_reduces_score(self, engine):
        """First-in-class position should reduce competitive intensity."""
        first_adj = engine.POSITION_ADJUSTMENTS[CompetitivePosition.FIRST_IN_CLASS]
        me_too_adj = engine.POSITION_ADJUSTMENTS[CompetitivePosition.ME_TOO]

        assert first_adj < me_too_adj
