#!/usr/bin/env python3
"""
Regression tests for Phase Momentum Interaction Rules.

These tests lock in the CTGov artifact fix and governance rules:
- Rule A: Skip penalty for commercial-stage companies
- Rule B: Double jeopardy for development-stage stagnation
- Rule C: Confidence ramp instead of hard threshold
- Commercial boost cap: max +0.5 for approved/marketed companies
"""

import pytest
from decimal import Decimal

from src.modules.ic_enhancements import compute_interaction_terms


class TestRuleA_CommercialStageSkip:
    """Rule A: Skip penalty for approved/commercial companies."""

    @pytest.mark.parametrize("lead_phase", [
        "approved",
        "marketed",
        "phase 4",
        "commercial",
        "APPROVED",  # Case insensitive
        "Marketed",
    ])
    def test_strong_negative_skipped_for_commercial(self, lead_phase):
        """strong_negative + commercial stage → penalty skipped."""
        result = compute_interaction_terms(
            clinical_normalized=Decimal('80'),
            financial_data={'runway_months': 24},
            catalyst_normalized=Decimal('70'),
            stage_bucket='approved',
            phase_momentum='strong_negative',
            phase_momentum_confidence=Decimal('1.0'),
            phase_momentum_lead_phase=lead_phase,
        )

        assert result.phase_momentum_interaction == Decimal('0')
        assert 'phase_momentum_penalty_skipped_commercial' in result.interaction_flags
        assert 'phase_momentum_penalty' not in result.interaction_flags

    @pytest.mark.parametrize("lead_phase", [
        "phase 1",
        "phase 2",
        "phase 3",
        "phase 1/2",
        "phase 2/3",
        "preclinical",
    ])
    def test_strong_negative_applied_for_development(self, lead_phase):
        """strong_negative + development stage → penalty applied."""
        result = compute_interaction_terms(
            clinical_normalized=Decimal('80'),
            financial_data={'runway_months': 24},
            catalyst_normalized=Decimal('70'),
            stage_bucket='phase_2',
            phase_momentum='strong_negative',
            phase_momentum_confidence=Decimal('1.0'),
            phase_momentum_lead_phase=lead_phase,
        )

        assert result.phase_momentum_interaction < Decimal('0')
        assert 'phase_momentum_penalty' in result.interaction_flags
        assert 'phase_momentum_penalty_skipped_commercial' not in result.interaction_flags


class TestRuleB_DoubleJeopardy:
    """Rule B: Double jeopardy for development-stage stagnation."""

    def test_double_jeopardy_triggers_with_high_burn_risk(self):
        """Double jeopardy when: strong_neg + high burn + no catalyst."""
        result = compute_interaction_terms(
            clinical_normalized=Decimal('80'),
            financial_data={'runway_months': 24},
            catalyst_normalized=Decimal('30'),
            stage_bucket='phase_2',
            cash_burn_risk='high',
            phase_momentum='strong_negative',
            phase_momentum_confidence=Decimal('1.0'),
            phase_momentum_lead_phase='phase 2',
            days_to_nearest_catalyst=None,
        )

        assert 'phase_momentum_double_jeopardy' in result.interaction_flags
        assert result.phase_momentum_interaction == Decimal('-2.5')  # Clamped

    def test_double_jeopardy_triggers_with_short_runway(self):
        """Double jeopardy when: strong_neg + runway < 12 + no catalyst."""
        result = compute_interaction_terms(
            clinical_normalized=Decimal('80'),
            financial_data={'runway_months': 8},  # Short runway
            catalyst_normalized=Decimal('30'),
            stage_bucket='phase_2',
            cash_burn_risk='moderate',  # Not high, but runway is short
            phase_momentum='strong_negative',
            phase_momentum_confidence=Decimal('1.0'),
            phase_momentum_lead_phase='phase 2',
            days_to_nearest_catalyst=None,
        )

        assert 'phase_momentum_double_jeopardy' in result.interaction_flags

    def test_no_double_jeopardy_with_near_catalyst(self):
        """No double jeopardy when catalyst within 90 days."""
        result = compute_interaction_terms(
            clinical_normalized=Decimal('80'),
            financial_data={'runway_months': 8},
            catalyst_normalized=Decimal('70'),
            stage_bucket='phase_2',
            cash_burn_risk='high',
            phase_momentum='strong_negative',
            phase_momentum_confidence=Decimal('1.0'),
            phase_momentum_lead_phase='phase 2',
            days_to_nearest_catalyst=45,  # Near catalyst
        )

        assert 'phase_momentum_double_jeopardy' not in result.interaction_flags
        assert result.phase_momentum_interaction == Decimal('-1.5')

    def test_no_double_jeopardy_for_commercial(self):
        """Double jeopardy never applies to commercial stage."""
        result = compute_interaction_terms(
            clinical_normalized=Decimal('80'),
            financial_data={'runway_months': 6},  # Very short
            catalyst_normalized=Decimal('30'),
            stage_bucket='approved',
            cash_burn_risk='critical',
            phase_momentum='strong_negative',
            phase_momentum_confidence=Decimal('1.0'),
            phase_momentum_lead_phase='approved',
            days_to_nearest_catalyst=None,
        )

        assert 'phase_momentum_double_jeopardy' not in result.interaction_flags
        assert 'phase_momentum_penalty_skipped_commercial' in result.interaction_flags


class TestRuleC_ConfidenceRamp:
    """Rule C: Confidence ramp instead of hard threshold."""

    def test_zero_effect_at_confidence_0_5(self):
        """No effect when confidence = 0.5 (multiplier = 0)."""
        result = compute_interaction_terms(
            clinical_normalized=Decimal('80'),
            financial_data={'runway_months': 24},
            catalyst_normalized=Decimal('70'),
            stage_bucket='phase_2',
            phase_momentum='strong_negative',
            phase_momentum_confidence=Decimal('0.5'),
            phase_momentum_lead_phase='phase 2',
        )

        assert result.phase_momentum_interaction == Decimal('0')

    def test_half_effect_at_confidence_0_75(self):
        """Half effect when confidence = 0.75 (multiplier = 0.5)."""
        result = compute_interaction_terms(
            clinical_normalized=Decimal('80'),
            financial_data={'runway_months': 24},
            catalyst_normalized=Decimal('70'),
            stage_bucket='phase_2',
            phase_momentum='strong_negative',
            phase_momentum_confidence=Decimal('0.75'),
            phase_momentum_lead_phase='phase 2',
        )

        expected = Decimal('-1.5') * Decimal('0.5')
        assert result.phase_momentum_interaction == expected

    def test_full_effect_at_confidence_1_0(self):
        """Full effect when confidence = 1.0 (multiplier = 1.0)."""
        result = compute_interaction_terms(
            clinical_normalized=Decimal('80'),
            financial_data={'runway_months': 24},
            catalyst_normalized=Decimal('70'),
            stage_bucket='phase_2',
            phase_momentum='strong_negative',
            phase_momentum_confidence=Decimal('1.0'),
            phase_momentum_lead_phase='phase 2',
        )

        assert result.phase_momentum_interaction == Decimal('-1.5')


class TestCommercialBoostCap:
    """Commercial boost cap: max +0.5 for approved/marketed companies."""

    def test_strong_positive_capped_for_commercial(self):
        """strong_positive + commercial → boost capped at +0.5."""
        result = compute_interaction_terms(
            clinical_normalized=Decimal('80'),
            financial_data={'runway_months': 24},
            catalyst_normalized=Decimal('70'),
            stage_bucket='approved',
            phase_momentum='strong_positive',
            phase_momentum_confidence=Decimal('1.0'),
            phase_momentum_lead_phase='approved',
        )

        assert result.phase_momentum_interaction == Decimal('0.5')
        assert 'phase_momentum_boost_capped_commercial' in result.interaction_flags

    def test_strong_positive_full_for_development(self):
        """strong_positive + development → full +1.5 boost."""
        result = compute_interaction_terms(
            clinical_normalized=Decimal('80'),
            financial_data={'runway_months': 24},
            catalyst_normalized=Decimal('70'),
            stage_bucket='phase_2',
            phase_momentum='strong_positive',
            phase_momentum_confidence=Decimal('1.0'),
            phase_momentum_lead_phase='phase 2',
        )

        assert result.phase_momentum_interaction == Decimal('1.5')
        assert 'phase_momentum_boost' in result.interaction_flags

    def test_positive_capped_for_commercial(self):
        """positive + commercial → boost capped at +0.25."""
        result = compute_interaction_terms(
            clinical_normalized=Decimal('80'),
            financial_data={'runway_months': 24},
            catalyst_normalized=Decimal('70'),
            stage_bucket='approved',
            phase_momentum='positive',
            phase_momentum_confidence=Decimal('1.0'),
            phase_momentum_lead_phase='marketed',
        )

        assert result.phase_momentum_interaction == Decimal('0.25')
        assert 'phase_momentum_positive_capped_commercial' in result.interaction_flags
