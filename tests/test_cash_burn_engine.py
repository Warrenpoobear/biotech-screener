#!/usr/bin/env python3
"""
Tests for Cash Burn Trajectory Engine
"""

import pytest
from decimal import Decimal
from datetime import date

from cash_burn_engine import (
    CashBurnEngine,
    BurnTrajectory,
    BurnRiskLevel,
)


class TestCashBurnEngine:
    """Tests for CashBurnEngine."""

    @pytest.fixture
    def engine(self):
        return CashBurnEngine()

    def test_initialization(self, engine):
        """Engine initializes with empty state."""
        assert engine.audit_trail == []
        assert engine.VERSION == "1.0.0"

    def test_decelerating_burn_good_runway(self, engine):
        """Decelerating burn with good runway gets positive modifier."""
        financial = {
            "burn_rate_current": 20,  # $20M/quarter
            "burn_rate_prior": 30,    # $30M/quarter (was higher)
            "runway_months": 36,
        }
        clinical = {"lead_phase": "phase_2"}
        as_of = date(2026, 1, 26)

        result = engine.compute_trajectory("TEST", financial, clinical, as_of)

        assert result.trajectory == BurnTrajectory.DECELERATING
        assert result.risk_level == BurnRiskLevel.LOW
        assert result.score_modifier > Decimal("0")
        assert "burn_decelerating" in result.flags

    def test_accelerating_burn_short_runway(self, engine):
        """Accelerating burn with short runway gets large penalty."""
        financial = {
            "burn_rate_current": 40,  # $40M/quarter
            "burn_rate_prior": 25,    # $25M/quarter (was lower)
            "runway_months": 8,       # Short runway
        }
        clinical = {"lead_phase": "phase_2"}
        as_of = date(2026, 1, 26)

        result = engine.compute_trajectory("TEST", financial, clinical, as_of)

        assert result.trajectory == BurnTrajectory.ACCELERATING
        assert result.risk_level == BurnRiskLevel.HIGH
        assert result.score_modifier < Decimal("-1.0")
        assert "burn_accelerating" in result.flags
        assert "burn_risk_high" in result.flags

    def test_accelerating_burn_phase3_justified(self, engine):
        """Accelerating burn with Phase 3 program is justified."""
        financial = {
            "burn_rate_current": 50,
            "burn_rate_prior": 30,
            "runway_months": 24,
        }
        clinical = {"lead_phase": "phase_3"}
        as_of = date(2026, 1, 26)

        result = engine.compute_trajectory("TEST", financial, clinical, as_of)

        assert result.trajectory == BurnTrajectory.ACCELERATING_JUSTIFIED
        assert result.risk_level == BurnRiskLevel.LOW
        assert result.score_modifier > Decimal("-1.0")  # Small penalty
        assert "burn_accelerating_phase3_justified" in result.flags

    def test_accelerating_critical_runway(self, engine):
        """Accelerating burn with <6mo runway is critical."""
        financial = {
            "burn_rate_current": 45,
            "burn_rate_prior": 30,
            "runway_months": 4,  # Very short
        }
        clinical = {"lead_phase": "phase_2"}
        as_of = date(2026, 1, 26)

        result = engine.compute_trajectory("TEST", financial, clinical, as_of)

        assert result.trajectory == BurnTrajectory.ACCELERATING
        assert result.risk_level == BurnRiskLevel.CRITICAL
        # Modifier is scaled by confidence (0.7), so -2.0 * 0.7 = -1.40
        assert result.score_modifier <= Decimal("-1.0")
        assert "burn_risk_critical" in result.flags

    def test_stable_burn(self, engine):
        """Stable burn (within ±15%) gets neutral modifier."""
        financial = {
            "burn_rate_current": 32,
            "burn_rate_prior": 30,  # ~7% increase = stable
            "runway_months": 24,
        }
        clinical = {}
        as_of = date(2026, 1, 26)

        result = engine.compute_trajectory("TEST", financial, clinical, as_of)

        assert result.trajectory == BurnTrajectory.STABLE
        assert result.score_modifier == Decimal("0")

    def test_insufficient_data(self, engine):
        """Missing burn data returns unknown with no modifier."""
        financial = {"runway_months": 24}
        clinical = {}
        as_of = date(2026, 1, 26)

        result = engine.compute_trajectory("TEST", financial, clinical, as_of)

        assert result.trajectory == BurnTrajectory.UNKNOWN
        assert result.risk_level == BurnRiskLevel.UNKNOWN
        assert result.score_modifier == Decimal("0")
        assert "insufficient_burn_data" in result.flags

    def test_quarterly_data_pit_filtering(self, engine):
        """Quarterly data is filtered by as_of_date."""
        financial = {
            "quarterly_burn": [
                {"period_end": "2025-09-30", "operating_cash_flow": -30},
                {"period_end": "2025-12-31", "operating_cash_flow": -35},
                {"period_end": "2026-03-31", "operating_cash_flow": -40},  # Future
            ],
            "runway_months": 24,
        }
        clinical = {}
        as_of = date(2026, 1, 26)

        result = engine.compute_trajectory("TEST", financial, clinical, as_of)

        # Should only use Q3 and Q4 2025 (Q1 2026 is in future)
        assert result.current_quarterly_burn == Decimal("35")  # Dec 2025
        assert result.prior_quarterly_burn == Decimal("30")    # Sep 2025

    def test_change_pct_calculation(self, engine):
        """Burn change percentage is calculated correctly."""
        financial = {
            "burn_rate_current": 40,
            "burn_rate_prior": 25,
            "runway_months": 24,
        }
        clinical = {}
        as_of = date(2026, 1, 26)

        result = engine.compute_trajectory("TEST", financial, clinical, as_of)

        # (40 - 25) / 25 = 0.60 = 60% increase
        assert result.burn_change_pct == Decimal("0.6000")

    def test_phase3_detection_variants(self, engine):
        """Various Phase 3 indicators are detected."""
        as_of = date(2026, 1, 26)
        financial = {"burn_rate_current": 50, "burn_rate_prior": 30, "runway_months": 24}

        # Test different phase 3 indicators
        for clinical in [
            {"lead_phase": "phase_3"},
            {"lead_phase": "Phase 3"},
            {"lead_phase": "phase_3_pivotal"},
            {"phase_3_trials": 2},
            {"lead_phase": "approved"},
        ]:
            result = engine.compute_trajectory("TEST", financial, clinical, as_of)
            assert result.has_late_stage_catalyst is True, f"Failed for {clinical}"

    def test_score_universe(self, engine):
        """Score entire universe."""
        universe = [{"ticker": "A"}, {"ticker": "B"}, {"ticker": "C"}]
        financial = {
            "A": {"burn_rate_current": 20, "burn_rate_prior": 30, "runway_months": 36},
            "B": {"burn_rate_current": 40, "burn_rate_prior": 25, "runway_months": 8},
            "C": {},
        }
        clinical = {
            "A": {"lead_phase": "phase_2"},
            "B": {"lead_phase": "phase_2"},
            "C": {},
        }
        as_of = date(2026, 1, 26)

        result = engine.score_universe(universe, financial, clinical, as_of)

        assert result["diagnostic_counts"]["total_scored"] == 3
        assert "trajectory_distribution" in result["diagnostic_counts"]
        assert "provenance" in result

    def test_modifier_bounded(self, engine):
        """Score modifier is always within [-2.0, +2.0]."""
        # Test extreme cases
        for burn_current, burn_prior, runway in [
            (100, 10, 3),   # Extreme acceleration + critical runway
            (5, 50, 60),    # Extreme deceleration + great runway
        ]:
            financial = {
                "burn_rate_current": burn_current,
                "burn_rate_prior": burn_prior,
                "runway_months": runway,
            }
            result = engine.compute_trajectory("TEST", financial, {}, date(2026, 1, 26))

            assert result.score_modifier >= Decimal("-2.0")
            assert result.score_modifier <= Decimal("2.0")

    def test_audit_trail(self, engine):
        """Audit trail is maintained."""
        financial = {"burn_rate_current": 30, "burn_rate_prior": 30, "runway_months": 24}
        engine.compute_trajectory("TEST", financial, {}, date(2026, 1, 26))

        trail = engine.get_audit_trail()
        assert len(trail) == 1
        assert trail[0]["ticker"] == "TEST"

        engine.clear_audit_trail()
        assert len(engine.get_audit_trail()) == 0


class TestBurnTrajectoryThresholds:
    """Tests for trajectory classification thresholds."""

    @pytest.fixture
    def engine(self):
        return CashBurnEngine()

    def test_threshold_boundaries(self, engine):
        """Test exact threshold boundaries."""
        as_of = date(2026, 1, 26)
        base_financial = {"runway_months": 24}

        # Exactly at deceleration threshold (-15%)
        financial = {**base_financial, "burn_rate_current": 85, "burn_rate_prior": 100}
        result = engine.compute_trajectory("TEST", financial, {}, as_of)
        assert result.trajectory == BurnTrajectory.DECELERATING

        # Just above deceleration threshold (-14%)
        financial = {**base_financial, "burn_rate_current": 86, "burn_rate_prior": 100}
        result = engine.compute_trajectory("TEST", financial, {}, as_of)
        assert result.trajectory == BurnTrajectory.STABLE

        # Just below acceleration threshold (+14%)
        financial = {**base_financial, "burn_rate_current": 114, "burn_rate_prior": 100}
        result = engine.compute_trajectory("TEST", financial, {}, as_of)
        assert result.trajectory == BurnTrajectory.STABLE

        # Exactly at acceleration threshold (+15%)
        financial = {**base_financial, "burn_rate_current": 115, "burn_rate_prior": 100}
        result = engine.compute_trajectory("TEST", financial, {}, as_of)
        assert result.trajectory == BurnTrajectory.ACCELERATING

    def test_runway_risk_boundaries(self, engine):
        """Test runway risk level boundaries."""
        as_of = date(2026, 1, 26)
        accel_financial = {"burn_rate_current": 50, "burn_rate_prior": 30}

        # Comfortable runway (24+ months) = LOW risk for justified
        financial = {**accel_financial, "runway_months": 24}
        result = engine.compute_trajectory("TEST", financial, {"lead_phase": "phase_3"}, as_of)
        assert result.risk_level == BurnRiskLevel.LOW

        # Short runway (6-12 months) = HIGH risk
        financial = {**accel_financial, "runway_months": 10}
        result = engine.compute_trajectory("TEST", financial, {}, as_of)
        assert result.risk_level == BurnRiskLevel.HIGH

        # Very short runway (<6 months) = CRITICAL
        financial = {**accel_financial, "runway_months": 5}
        result = engine.compute_trajectory("TEST", financial, {}, as_of)
        assert result.risk_level == BurnRiskLevel.CRITICAL
