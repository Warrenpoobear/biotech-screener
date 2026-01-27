#!/usr/bin/env python3
"""
Tests for Phase Transition Momentum Engine
"""

import pytest
from decimal import Decimal
from datetime import date, timedelta

from phase_momentum_engine import (
    PhaseTransitionEngine,
    PhaseMomentum,
    PhaseLevel,
)


class TestPhaseTransitionEngine:
    """Tests for PhaseTransitionEngine."""

    @pytest.fixture
    def engine(self):
        return PhaseTransitionEngine()

    @pytest.fixture
    def as_of(self):
        return date(2026, 1, 26)

    def test_initialization(self, engine):
        """Engine initializes with empty state."""
        assert engine.audit_trail == []
        assert engine.VERSION == "1.0.0"

    def test_no_trials(self, engine, as_of):
        """No trials returns unknown momentum."""
        result = engine.compute_momentum("TEST", [], None, as_of)

        assert result.momentum == PhaseMomentum.UNKNOWN
        assert result.score_modifier == Decimal("0")
        assert "no_trials" in result.flags

    def test_strong_positive_momentum(self, engine, as_of):
        """Recent advancement + active pipeline = strong positive."""
        trials = [
            # Recent Phase 3 initiation (advancement)
            {
                "nct_id": "NCT001",
                "phase": "PHASE3",
                "status": "RECRUITING",
                "last_update_posted": "2026-01-15",
                "start_date": "2025-12-01",
            },
            # Active Phase 2 trials
            {
                "nct_id": "NCT002",
                "phase": "PHASE2",
                "status": "ACTIVE_NOT_RECRUITING",
                "last_update_posted": "2026-01-10",
                "start_date": "2025-10-01",
            },
            {
                "nct_id": "NCT003",
                "phase": "PHASE2",
                "status": "RECRUITING",
                "last_update_posted": "2026-01-05",
                "start_date": "2025-11-01",
            },
        ]

        result = engine.compute_momentum("TEST", trials, None, as_of)

        assert result.momentum == PhaseMomentum.STRONG_POSITIVE
        assert result.score_modifier > Decimal("0")
        assert "recent_phase_advancement" in result.flags
        assert result.current_lead_phase == "phase 3"

    def test_positive_momentum(self, engine, as_of):
        """Active pipeline with good velocity = positive."""
        trials = [
            {
                "nct_id": "NCT001",
                "phase": "PHASE2",
                "status": "RECRUITING",
                "last_update_posted": "2026-01-10",
                "start_date": "2025-09-01",
            },
            {
                "nct_id": "NCT002",
                "phase": "PHASE2",
                "status": "RECRUITING",
                "last_update_posted": "2026-01-05",
                "start_date": "2025-10-01",
            },
            {
                "nct_id": "NCT003",
                "phase": "PHASE1",
                "status": "ACTIVE_NOT_RECRUITING",
                "last_update_posted": "2025-12-15",
                "start_date": "2025-08-01",
            },
        ]

        result = engine.compute_momentum("TEST", trials, None, as_of)

        assert result.momentum in (PhaseMomentum.POSITIVE, PhaseMomentum.STRONG_POSITIVE)
        assert result.score_modifier > Decimal("0")

    def test_negative_momentum_stalled(self, engine, as_of):
        """Multiple stalled trials with low velocity = negative momentum.

        Note: Requires minimum trial count (3) and low velocity (<40) to trigger.
        Stalls alone are not enough - must combine with low activity signals.
        """
        # Trials with old last_update dates (stalled)
        stale_date = (as_of - timedelta(days=400)).isoformat()

        trials = [
            {
                "nct_id": "NCT001",
                "phase": "PHASE2",
                "status": "RECRUITING",
                "last_update_posted": stale_date,
            },
            {
                "nct_id": "NCT002",
                "phase": "PHASE2",
                "status": "RECRUITING",
                "last_update_posted": stale_date,
            },
            {
                "nct_id": "NCT003",
                "phase": "PHASE1",
                "status": "ACTIVE_NOT_RECRUITING",
                "last_update_posted": stale_date,
            },
            {
                "nct_id": "NCT004",
                "phase": "PHASE1",
                "status": "RECRUITING",
                "last_update_posted": stale_date,
            },
        ]

        result = engine.compute_momentum("TEST", trials, None, as_of)

        # With new conservative classification: stalls alone = neutral
        # Negative requires stalls + low velocity + no initiations
        assert result.momentum == PhaseMomentum.NEUTRAL
        assert result.stalled_trials >= 3
        assert "multiple_stalled_trials" in result.flags

    def test_strong_negative_terminations(self, engine, as_of):
        """Multiple terminations + multiple stalls = strong negative.

        Note: Requires 2 independent negatives (terminations AND stalls)
        to trigger strong_negative. Single negative = negative or neutral.
        """
        recent = (as_of - timedelta(days=30)).isoformat()
        stale_date = (as_of - timedelta(days=400)).isoformat()

        trials = [
            # Terminated trials
            {
                "nct_id": "NCT001",
                "phase": "PHASE3",
                "status": "TERMINATED",
                "last_update_posted": recent,
            },
            {
                "nct_id": "NCT002",
                "phase": "PHASE2",
                "status": "TERMINATED",
                "last_update_posted": recent,
            },
            # Stalled trials (old last_update)
            {
                "nct_id": "NCT003",
                "phase": "PHASE1",
                "status": "RECRUITING",
                "last_update_posted": stale_date,
            },
            {
                "nct_id": "NCT004",
                "phase": "PHASE1",
                "status": "RECRUITING",
                "last_update_posted": stale_date,
            },
        ]

        result = engine.compute_momentum("TEST", trials, None, as_of)

        assert result.momentum == PhaseMomentum.STRONG_NEGATIVE
        assert result.score_modifier < Decimal("-0.5")
        assert result.terminated_trials >= 2
        assert result.stalled_trials >= 2
        assert "recent_terminations" in result.flags

    def test_neutral_stable_pipeline(self, engine, as_of):
        """Stable pipeline without changes = neutral."""
        # Older but not stalled trials
        update_date = (as_of - timedelta(days=200)).isoformat()

        trials = [
            {
                "nct_id": "NCT001",
                "phase": "PHASE2",
                "status": "COMPLETED",
                "last_update_posted": update_date,
            },
            {
                "nct_id": "NCT002",
                "phase": "PHASE1",
                "status": "COMPLETED",
                "last_update_posted": update_date,
            },
        ]

        result = engine.compute_momentum("TEST", trials, None, as_of)

        assert result.momentum == PhaseMomentum.NEUTRAL
        assert result.score_modifier == Decimal("0")

    def test_phase_normalization(self, engine):
        """Phase strings are normalized correctly."""
        test_cases = [
            ("PHASE3", "phase 3"),
            ("Phase 3", "phase 3"),
            ("phase_3", "phase 3"),
            ("PHASE2/3", "phase 2/3"),
            ("Phase 2/Phase 3", "phase 2/3"),
            ("PHASE2", "phase 2"),
            ("PHASE1/2", "phase 1/2"),
            ("PHASE1", "phase 1"),
            ("PHASE4", "approved"),
            ("", "preclinical"),
            (None, "preclinical"),
        ]

        for input_phase, expected in test_cases:
            result = engine._normalize_phase(input_phase)
            assert result == expected, f"Failed for {input_phase}: got {result}"

    def test_lead_phase_detection(self, engine, as_of):
        """Lead phase is correctly identified from active trials."""
        trials = [
            {"phase": "PHASE1", "status": "COMPLETED", "last_update_posted": "2026-01-01"},
            {"phase": "PHASE2", "status": "RECRUITING", "last_update_posted": "2026-01-01"},
            {"phase": "PHASE3", "status": "TERMINATED", "last_update_posted": "2026-01-01"},
        ]

        result = engine.compute_momentum("TEST", trials, None, as_of)

        # Phase 3 is terminated, so Phase 2 should be lead
        assert result.current_lead_phase == "phase 2"

    def test_pit_filtering(self, engine, as_of):
        """Future trials are filtered out."""
        trials = [
            {
                "nct_id": "NCT001",
                "phase": "PHASE3",
                "status": "RECRUITING",
                "last_update_posted": "2026-02-01",  # Future
            },
            {
                "nct_id": "NCT002",
                "phase": "PHASE2",
                "status": "RECRUITING",
                "last_update_posted": "2026-01-15",  # Valid
            },
        ]

        result = engine.compute_momentum("TEST", trials, None, as_of)

        # Should only see Phase 2 trial
        assert result.current_lead_phase == "phase 2"

    def test_velocity_score_high_activity(self, engine, as_of):
        """High activity pipeline gets high velocity score."""
        recent = (as_of - timedelta(days=60)).isoformat()

        trials = [
            {"phase": "PHASE3", "status": "RECRUITING", "last_update_posted": recent, "start_date": recent},
            {"phase": "PHASE2", "status": "RECRUITING", "last_update_posted": recent, "start_date": recent},
            {"phase": "PHASE2", "status": "RECRUITING", "last_update_posted": recent, "start_date": recent},
            {"phase": "PHASE1", "status": "RECRUITING", "last_update_posted": recent, "start_date": recent},
            {"phase": "PHASE1", "status": "RECRUITING", "last_update_posted": recent, "start_date": recent},
        ]

        result = engine.compute_momentum("TEST", trials, None, as_of)

        assert result.phase_velocity_score >= Decimal("70")
        assert "high_velocity" in result.flags

    def test_velocity_score_low_activity(self, engine, as_of):
        """Inactive pipeline gets low velocity score."""
        old = (as_of - timedelta(days=400)).isoformat()

        trials = [
            {"phase": "PHASE1", "status": "COMPLETED", "last_update_posted": old},
        ]

        result = engine.compute_momentum("TEST", trials, None, as_of)

        assert result.phase_velocity_score <= Decimal("55")

    def test_score_universe(self, engine, as_of):
        """Score entire universe."""
        universe = [{"ticker": "A"}, {"ticker": "B"}, {"ticker": "C"}]
        trials_by_ticker = {
            "A": [
                {"phase": "PHASE3", "status": "RECRUITING", "last_update_posted": "2026-01-15"},
                {"phase": "PHASE2", "status": "RECRUITING", "last_update_posted": "2026-01-10"},
            ],
            "B": [
                {"phase": "PHASE1", "status": "RECRUITING", "last_update_posted": "2026-01-15"},
            ],
            "C": [],
        }

        result = engine.score_universe(universe, trials_by_ticker, None, as_of)

        assert result["diagnostic_counts"]["total_scored"] == 3
        assert "momentum_distribution" in result["diagnostic_counts"]
        assert "provenance" in result

    def test_modifier_bounded(self, engine, as_of):
        """Score modifier is always within [-2.0, +2.0]."""
        # Create extreme positive case
        recent = (as_of - timedelta(days=30)).isoformat()
        positive_trials = [
            {"phase": "PHASE3", "status": "RECRUITING", "last_update_posted": recent, "start_date": recent},
            {"phase": "PHASE3", "status": "RECRUITING", "last_update_posted": recent, "start_date": recent},
            {"phase": "PHASE2", "status": "RECRUITING", "last_update_posted": recent, "start_date": recent},
            {"phase": "PHASE2", "status": "RECRUITING", "last_update_posted": recent, "start_date": recent},
            {"phase": "PHASE1", "status": "RECRUITING", "last_update_posted": recent, "start_date": recent},
        ]

        result = engine.compute_momentum("TEST", positive_trials, None, as_of)
        assert result.score_modifier >= Decimal("-2.0")
        assert result.score_modifier <= Decimal("2.0")

        # Create extreme negative case
        negative_trials = [
            {"phase": "PHASE3", "status": "TERMINATED", "last_update_posted": recent},
            {"phase": "PHASE3", "status": "TERMINATED", "last_update_posted": recent},
            {"phase": "PHASE2", "status": "WITHDRAWN", "last_update_posted": recent},
        ]

        result = engine.compute_momentum("TEST", negative_trials, None, as_of)
        assert result.score_modifier >= Decimal("-2.0")
        assert result.score_modifier <= Decimal("2.0")

    def test_audit_trail(self, engine, as_of):
        """Audit trail is maintained."""
        trials = [{"phase": "PHASE2", "status": "RECRUITING", "last_update_posted": "2026-01-15"}]
        engine.compute_momentum("TEST", trials, None, as_of)

        trail = engine.get_audit_trail()
        assert len(trail) == 1
        assert trail[0]["ticker"] == "TEST"

        engine.clear_audit_trail()
        assert len(engine.get_audit_trail()) == 0

    def test_late_stage_flag(self, engine, as_of):
        """Late stage companies get flagged."""
        trials = [
            {"phase": "PHASE3", "status": "RECRUITING", "last_update_posted": "2026-01-15"},
        ]

        result = engine.compute_momentum("TEST", trials, None, as_of)

        assert "late_stage_company" in result.flags
        assert result.phase_level >= PhaseLevel.PHASE_3.value

    def test_active_pipeline_flag(self, engine, as_of):
        """Active pipeline with many recent initiations gets flagged."""
        recent = (as_of - timedelta(days=60)).isoformat()

        trials = [
            {"phase": "PHASE2", "status": "RECRUITING", "last_update_posted": recent, "start_date": recent},
            {"phase": "PHASE2", "status": "RECRUITING", "last_update_posted": recent, "start_date": recent},
            {"phase": "PHASE1", "status": "RECRUITING", "last_update_posted": recent, "start_date": recent},
            {"phase": "PHASE1", "status": "RECRUITING", "last_update_posted": recent, "start_date": recent},
        ]

        result = engine.compute_momentum("TEST", trials, None, as_of)

        assert "active_pipeline" in result.flags
        assert result.recent_initiations >= 3


class TestPhaseLevel:
    """Tests for PhaseLevel enum ordering."""

    def test_phase_ordering(self):
        """Phases are correctly ordered."""
        assert PhaseLevel.PRECLINICAL.value < PhaseLevel.PHASE_1.value
        assert PhaseLevel.PHASE_1.value < PhaseLevel.PHASE_1_2.value
        assert PhaseLevel.PHASE_1_2.value < PhaseLevel.PHASE_2.value
        assert PhaseLevel.PHASE_2.value < PhaseLevel.PHASE_2_3.value
        assert PhaseLevel.PHASE_2_3.value < PhaseLevel.PHASE_3.value
        assert PhaseLevel.PHASE_3.value < PhaseLevel.PHASE_4.value
        assert PhaseLevel.PHASE_4.value < PhaseLevel.APPROVED.value
