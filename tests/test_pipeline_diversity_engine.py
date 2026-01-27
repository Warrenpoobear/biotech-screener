#!/usr/bin/env python3
"""
Tests for Pipeline Diversity Engine
"""

import pytest
from decimal import Decimal
from datetime import date

from pipeline_diversity_engine import (
    PipelineDiversityEngine,
    PipelineRiskProfile,
    PlatformType,
)


class TestPipelineDiversityEngine:
    """Tests for PipelineDiversityEngine."""

    @pytest.fixture
    def engine(self):
        return PipelineDiversityEngine()

    def test_initialization(self, engine):
        """Engine initializes with empty state."""
        assert engine.audit_trail == []

    def test_single_asset_risk_profile(self, engine):
        """Single program company gets single_asset risk profile."""
        clinical_data = {"n_trials_unique": 1, "lead_phase": "phase_2"}
        as_of = date(2026, 1, 26)

        result = engine.score_pipeline(
            ticker="SINGLE",
            trial_records=[],
            as_of_date=as_of,
            clinical_score_data=clinical_data,
        )

        assert result["risk_profile"] == PipelineRiskProfile.SINGLE_ASSET.value
        assert result["diversity_score"] < Decimal("50")  # Below neutral

    def test_diversified_risk_profile(self, engine):
        """Large pipeline gets diversified risk profile."""
        clinical_data = {"n_trials_unique": 50, "lead_phase": "commercial"}
        as_of = date(2026, 1, 26)

        result = engine.score_pipeline(
            ticker="DIVERSE",
            trial_records=[],
            as_of_date=as_of,
            clinical_score_data=clinical_data,
        )

        assert result["risk_profile"] in [
            PipelineRiskProfile.DIVERSIFIED.value,
            PipelineRiskProfile.BROAD_PORTFOLIO.value,
        ]
        assert result["diversity_score"] > Decimal("50")  # Above neutral

    def test_phase_diversity_bonus(self, engine):
        """Programs across multiple phases get diversity bonus."""
        # Create mock trial records with multiple phases
        trial_records = [
            {"ticker": "MULTI", "nct_id": "NCT001", "phase": "Phase 1", "conditions": ["Cancer"], "interventions": []},
            {"ticker": "MULTI", "nct_id": "NCT002", "phase": "Phase 2", "conditions": ["Cancer"], "interventions": []},
            {"ticker": "MULTI", "nct_id": "NCT003", "phase": "Phase 3", "conditions": ["Cancer"], "interventions": []},
        ]
        as_of = date(2026, 1, 26)

        result = engine.score_pipeline(
            ticker="MULTI",
            trial_records=trial_records,
            as_of_date=as_of,
        )

        assert result["phase_diversity_count"] >= 3
        # Score should reflect phase diversity bonus

    def test_indication_diversity_bonus(self, engine):
        """Programs across multiple indications get diversity bonus."""
        trial_records = [
            {"ticker": "MULTI", "nct_id": "NCT001", "phase": "Phase 2", "conditions": ["Cancer"], "interventions": []},
            {"ticker": "MULTI", "nct_id": "NCT002", "phase": "Phase 2", "conditions": ["Diabetes"], "interventions": []},
            {"ticker": "MULTI", "nct_id": "NCT003", "phase": "Phase 2", "conditions": ["Alzheimer"], "interventions": []},
        ]
        as_of = date(2026, 1, 26)

        result = engine.score_pipeline(
            ticker="MULTI",
            trial_records=trial_records,
            as_of_date=as_of,
        )

        assert result["indication_diversity_count"] >= 2

    def test_platform_validation(self, engine):
        """Platform with 3+ programs gets validation bonus."""
        trial_records = [
            {"ticker": "PLAT", "nct_id": "NCT001", "phase": "Phase 1", "conditions": ["Cancer"], "interventions": ["mRNA vaccine"]},
            {"ticker": "PLAT", "nct_id": "NCT002", "phase": "Phase 2", "conditions": ["Cancer"], "interventions": ["mRNA therapeutic"]},
            {"ticker": "PLAT", "nct_id": "NCT003", "phase": "Phase 3", "conditions": ["Infectious"], "interventions": ["mRNA vaccine"]},
        ]
        as_of = date(2026, 1, 26)

        result = engine.score_pipeline(
            ticker="PLAT",
            trial_records=trial_records,
            as_of_date=as_of,
        )

        assert result["platform_validated"] is True

    def test_empty_pipeline(self, engine):
        """Empty pipeline gets low score."""
        as_of = date(2026, 1, 26)

        result = engine.score_pipeline(
            ticker="EMPTY",
            trial_records=[],
            as_of_date=as_of,
        )

        assert result["program_count"] == 0
        assert result["diversity_score"] < Decimal("40")  # Below average

    def test_score_universe(self, engine):
        """Score entire universe."""
        universe = [
            {"ticker": "SMALL", "clinical_data": {"n_trials_unique": 2, "lead_phase": "phase_1"}},
            {"ticker": "LARGE", "clinical_data": {"n_trials_unique": 100, "lead_phase": "commercial"}},
        ]
        as_of = date(2026, 1, 26)

        result = engine.score_universe(universe, [], as_of)

        assert result["diagnostic_counts"]["total_scored"] == 2
        assert "risk_distribution" in result["diagnostic_counts"]
        assert "provenance" in result

    def test_weighted_program_count(self, engine):
        """Later-stage programs weighted higher."""
        # Phase 3 program worth more than Phase 1
        trial_records_p1 = [
            {"ticker": "P1", "nct_id": "NCT001", "phase": "Phase 1", "conditions": ["Cancer"], "interventions": []},
        ]
        trial_records_p3 = [
            {"ticker": "P3", "nct_id": "NCT001", "phase": "Phase 3", "conditions": ["Cancer"], "interventions": []},
        ]
        as_of = date(2026, 1, 26)

        result_p1 = engine.score_pipeline("P1", trial_records_p1, as_of)
        result_p3 = engine.score_pipeline("P3", trial_records_p3, as_of)

        assert result_p3["weighted_program_count"] > result_p1["weighted_program_count"]

    def test_audit_trail(self, engine):
        """Audit trail is maintained."""
        clinical_data = {"n_trials_unique": 10, "lead_phase": "phase_2"}
        as_of = date(2026, 1, 26)

        engine.score_pipeline("TEST", [], as_of, clinical_data)

        trail = engine.get_audit_trail()
        assert len(trail) > 0
        assert trail[0]["ticker"] == "TEST"

        engine.clear_audit_trail()
        assert len(engine.get_audit_trail()) == 0


class TestRiskProfileClassification:
    """Tests for risk profile classification."""

    @pytest.fixture
    def engine(self):
        return PipelineDiversityEngine()

    def test_all_risk_profiles_have_adjustments(self):
        """All risk profiles have score adjustments."""
        for profile in PipelineRiskProfile:
            assert profile in PipelineDiversityEngine.RISK_PROFILE_SCORE_ADJUSTMENTS

    def test_risk_profile_ordering(self, engine):
        """Risk profiles should give progressively better scores."""
        as_of = date(2026, 1, 26)

        # Single asset (1 trial)
        single = engine.score_pipeline("S", [], as_of, {"n_trials_unique": 1, "lead_phase": "phase_2"})

        # Focused (5 trials)
        focused = engine.score_pipeline("F", [], as_of, {"n_trials_unique": 8, "lead_phase": "phase_3"})

        # Broad (100 trials)
        broad = engine.score_pipeline("B", [], as_of, {"n_trials_unique": 200, "lead_phase": "commercial"})

        assert single["diversity_score"] < focused["diversity_score"]
        assert focused["diversity_score"] < broad["diversity_score"]


class TestPlatformInference:
    """Tests for platform type inference."""

    @pytest.fixture
    def engine(self):
        return PipelineDiversityEngine()

    def test_infer_mrna_platform(self, engine):
        """mRNA interventions detected as RNA platform."""
        trial_records = [
            {"ticker": "RNA", "nct_id": "NCT001", "phase": "Phase 2",
             "conditions": ["Cancer"], "interventions": ["mRNA-1234 vaccine"]},
        ]
        as_of = date(2026, 1, 26)

        result = engine.score_pipeline("RNA", trial_records, as_of)
        assert result["dominant_platform"] == PlatformType.RNA.value

    def test_infer_car_t_platform(self, engine):
        """CAR-T interventions detected correctly."""
        trial_records = [
            {"ticker": "CART", "nct_id": "NCT001", "phase": "Phase 1",
             "conditions": ["Leukemia"], "interventions": ["CAR-T cell therapy"]},
        ]
        as_of = date(2026, 1, 26)

        result = engine.score_pipeline("CART", trial_records, as_of)
        assert result["dominant_platform"] == PlatformType.CAR_T.value

    def test_infer_antibody_platform(self, engine):
        """Antibody interventions detected correctly."""
        trial_records = [
            {"ticker": "MAB", "nct_id": "NCT001", "phase": "Phase 3",
             "conditions": ["Autoimmune"], "interventions": ["Anti-CD20 monoclonal antibody"]},
        ]
        as_of = date(2026, 1, 26)

        result = engine.score_pipeline("MAB", trial_records, as_of)
        assert result["dominant_platform"] == PlatformType.ANTIBODY.value
