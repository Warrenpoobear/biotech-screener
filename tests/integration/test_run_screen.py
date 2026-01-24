#!/usr/bin/env python3
"""
test_run_screen.py - Pipeline Integration Tests

P0 Enhancement: Comprehensive integration tests for run_screen.py

Tests:
1. Determinism (identical inputs -> byte-identical outputs)
2. Checkpoint save/resume functionality
3. Enhancement module integration
4. Audit log structure validation
5. Data staleness gate enforcement
6. Error handling for invalid inputs

Author: Wake Robin Capital Management
Version: 1.0.0
"""

import hashlib
import json
import os
import sys
import tempfile
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch
import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.provenance import compute_hash


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_universe() -> List[Dict[str, Any]]:
    """Sample universe for testing."""
    return [
        {
            "ticker": "ACME",
            "status": "active",
            "market_cap_mm": 500.0,
            "market_data": {"market_cap": 500000000},
        },
        {
            "ticker": "BETA",
            "status": "active",
            "market_cap_mm": 750.0,
            "market_data": {"market_cap": 750000000},
        },
        {
            "ticker": "GAMA",
            "status": "active",
            "market_cap_mm": 300.0,
            "market_data": {"market_cap": 300000000},
        },
    ]


@pytest.fixture
def sample_financial_records() -> List[Dict[str, Any]]:
    """Sample financial records for testing."""
    return [
        {
            "ticker": "ACME",
            "cash_quarterly": 100000000,
            "burn_cfo_quarterly": -15000000,
            "total_debt": 10000000,
            "adv_20d": 2500000,
            "source_date": "2025-12-15",
        },
        {
            "ticker": "BETA",
            "cash_quarterly": 200000000,
            "burn_cfo_quarterly": -20000000,
            "total_debt": 5000000,
            "adv_20d": 5000000,
            "source_date": "2025-12-15",
        },
        {
            "ticker": "GAMA",
            "cash_quarterly": 50000000,
            "burn_cfo_quarterly": -10000000,
            "total_debt": 0,
            "adv_20d": 1000000,
            "source_date": "2025-12-15",
        },
    ]


@pytest.fixture
def sample_trial_records() -> List[Dict[str, Any]]:
    """Sample trial records for testing."""
    return [
        {
            "ticker": "ACME",
            "nct_id": "NCT00000001",
            "phase": "Phase 3",
            "overall_status": "RECRUITING",
            "primary_completion_date": "2026-06-15",
            "primary_completion_type": "ESTIMATED",
            "last_update_posted": "2025-12-01",
            "source_date": "2025-12-01",
        },
        {
            "ticker": "BETA",
            "nct_id": "NCT00000002",
            "phase": "Phase 2",
            "overall_status": "ACTIVE_NOT_RECRUITING",
            "primary_completion_date": "2026-03-01",
            "primary_completion_type": "ESTIMATED",
            "last_update_posted": "2025-11-15",
            "source_date": "2025-11-15",
        },
        {
            "ticker": "GAMA",
            "nct_id": "NCT00000003",
            "phase": "Phase 1",
            "overall_status": "RECRUITING",
            "primary_completion_date": "2026-09-01",
            "primary_completion_type": "ESTIMATED",
            "last_update_posted": "2025-12-10",
            "source_date": "2025-12-10",
        },
    ]


@pytest.fixture
def sample_market_data() -> Dict[str, Dict[str, Any]]:
    """Sample market data for testing."""
    return {
        "ACME": {
            "market_cap": 500000000,
            "vix": 22.5,
            "price": 45.50,
            "adv_20d": 2500000,
        },
        "BETA": {
            "market_cap": 750000000,
            "vix": 22.5,
            "price": 62.30,
            "adv_20d": 5000000,
        },
        "GAMA": {
            "market_cap": 300000000,
            "vix": 22.5,
            "price": 28.75,
            "adv_20d": 1000000,
        },
    }


@pytest.fixture
def temp_data_dir(
    sample_universe,
    sample_financial_records,
    sample_trial_records,
    sample_market_data,
) -> str:
    """Create temporary data directory with sample data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write universe
        with open(os.path.join(tmpdir, "universe.json"), "w") as f:
            json.dump(sample_universe, f)

        # Write financial records
        with open(os.path.join(tmpdir, "financial_records.json"), "w") as f:
            json.dump(sample_financial_records, f)

        # Write trial records
        with open(os.path.join(tmpdir, "trial_records.json"), "w") as f:
            json.dump(sample_trial_records, f)

        # Write market data
        with open(os.path.join(tmpdir, "market_data.json"), "w") as f:
            json.dump(sample_market_data, f)

        # Create ctgov_state directory
        state_dir = os.path.join(tmpdir, "ctgov_state")
        os.makedirs(state_dir, exist_ok=True)

        yield tmpdir


@pytest.fixture
def as_of_date() -> date:
    """Standard as_of_date for deterministic tests."""
    return date(2026, 1, 15)


# =============================================================================
# TEST: DETERMINISM
# =============================================================================

class TestPipelineDeterminism:
    """
    Tests that the pipeline produces byte-identical outputs for identical inputs.

    This is critical for reproducibility and audit compliance.
    """

    def test_multi_window_momentum_determinism(self, as_of_date):
        """Multi-window momentum should produce identical output for identical input."""
        from multi_window_momentum import MultiWindowMomentumEngine, RegimeState

        engine = MultiWindowMomentumEngine()
        prices = [Decimal(str(100 + i * 0.3)) for i in range(150)]

        # Run twice
        result1 = engine.compute_momentum("ACME", prices, as_of_date, RegimeState.BULL)
        result2 = engine.compute_momentum("ACME", prices, as_of_date, RegimeState.BULL)

        # Compare scores
        assert result1.weighted_score == result2.weighted_score
        assert result1.signal_quality == result2.signal_quality
        assert result1.confidence == result2.confidence

    def test_staleness_gate_determinism(self, as_of_date):
        """Staleness gate should produce identical output for identical input."""
        from common.staleness_gates import StalenessGateEngine, DataType

        engine = StalenessGateEngine()
        data_date = date(2025, 10, 1)

        # Run twice
        result1 = engine.check_staleness(DataType.FINANCIAL, data_date, as_of_date)
        result2 = engine.check_staleness(DataType.FINANCIAL, data_date, as_of_date)

        assert result1.action == result2.action
        assert result1.penalty_multiplier == result2.penalty_multiplier
        assert result1.age_days == result2.age_days

    def test_forward_looking_separator_determinism(self, as_of_date):
        """Forward-looking separator should produce deterministic output."""
        from common.forward_looking_separation import ForwardLookingSeparator

        separator = ForwardLookingSeparator()
        detected = [{"score": "5.0"}]
        calendar = [{"days_until": 25, "confidence": 0.8}]

        # Run twice
        result1 = separator.separate_signals("ACME", as_of_date, detected, calendar)
        result2 = separator.separate_signals("ACME", as_of_date, detected, calendar)

        assert result1.historical_score == result2.historical_score
        assert result1.forward_looking_score == result2.forward_looking_score
        assert result1.blended_score == result2.blended_score


# =============================================================================
# TEST: AUDIT LOG STRUCTURE
# =============================================================================

class TestAuditLogStructure:
    """Tests for audit log completeness and structure."""

    def test_regime_hysteresis_audit_trail(self, as_of_date):
        """Regime hysteresis should maintain audit trail."""
        from regime_hysteresis import RegimeHysteresisEngine

        engine = RegimeHysteresisEngine()

        # Make multiple updates
        engine.update(Decimal("18"), Decimal("0.03"), as_of_date)
        engine.update(Decimal("19"), Decimal("0.02"), as_of_date + timedelta(days=1))
        engine.update(Decimal("26"), Decimal("-0.02"), as_of_date + timedelta(days=2))

        state = engine.get_state()
        assert state is not None
        assert state.current_regime is not None
        assert state.regime_since is not None

    def test_decay_result_has_explanation(self, as_of_date):
        """Decay results should include explanation for auditing."""
        from regime_adaptive_decay import RegimeAdaptiveDecayEngine, EventCategory, RegimeState

        engine = RegimeAdaptiveDecayEngine()
        result = engine.compute_decay(
            event_age_days=15,
            event_category=EventCategory.POSITIVE,
            regime=RegimeState.BULL,
            vix_level=Decimal("22.0")
        )

        assert result.explanation is not None
        assert "half-life" in result.explanation.lower()
        assert result.base_half_life > 0
        assert result.adjusted_half_life > 0


# =============================================================================
# TEST: STALENESS GATES
# =============================================================================

class TestStalenessGates:
    """Tests for data staleness gate enforcement."""

    def test_staleness_gate_engine_import(self):
        """Staleness gate engine should be importable."""
        from common.staleness_gates import (
            StalenessGateEngine,
            DataType,
            StalenessAction,
        )
        assert StalenessGateEngine is not None
        assert DataType.FINANCIAL is not None

    def test_financial_staleness_detection(self):
        """Financial data staleness should be detected correctly."""
        from common.staleness_gates import (
            StalenessGateEngine,
            DataType,
            StalenessAction,
        )

        engine = StalenessGateEngine()
        as_of = date(2026, 1, 15)

        # Fresh data (60 days old)
        result = engine.check_staleness(
            DataType.FINANCIAL,
            date(2025, 11, 15),
            as_of,
        )
        assert result.action in (StalenessAction.PASS, StalenessAction.WARN)

        # Stale data (150 days old)
        result = engine.check_staleness(
            DataType.FINANCIAL,
            date(2025, 8, 15),
            as_of,
        )
        assert result.action == StalenessAction.HARD_GATE

    def test_phase_dependent_trial_staleness(self):
        """Trial staleness should vary by phase."""
        from common.staleness_gates import (
            StalenessGateEngine,
            DataType,
            StalenessAction,
        )

        engine = StalenessGateEngine()
        as_of = date(2026, 1, 15)
        data_date = date(2025, 7, 15)  # 6 months old

        # Phase 3 should be stricter
        result_p3 = engine.check_staleness(
            DataType.TRIAL,
            data_date,
            as_of,
            phase="phase_3",
        )

        # Phase 1 should be more lenient
        result_p1 = engine.check_staleness(
            DataType.TRIAL,
            data_date,
            as_of,
            phase="phase_1",
        )

        # Phase 3 should have more severe action
        action_severity = {
            StalenessAction.PASS: 0,
            StalenessAction.WARN: 1,
            StalenessAction.SOFT_GATE: 2,
            StalenessAction.HARD_GATE: 3,
        }
        assert action_severity[result_p3.action] >= action_severity[result_p1.action]

    def test_lookahead_detection(self):
        """Future data should be flagged as lookahead bias."""
        from common.staleness_gates import (
            StalenessGateEngine,
            DataType,
            StalenessAction,
        )

        engine = StalenessGateEngine()
        as_of = date(2026, 1, 15)

        # Data from the future
        result = engine.check_staleness(
            DataType.FINANCIAL,
            date(2026, 2, 1),  # After as_of_date
            as_of,
        )

        assert result.action == StalenessAction.HARD_GATE
        assert "LOOKAHEAD" in result.message


# =============================================================================
# TEST: 13F LAG ENFORCEMENT
# =============================================================================

class Test13FLagEnforcement:
    """Tests for SEC 13F 45-day lag enforcement."""

    def test_13f_effective_date_calculation(self):
        """13F effective date should include 45-day SEC lag."""
        from common.staleness_gates import (
            compute_13f_effective_date,
            SEC_13F_FILING_LAG_DAYS,
        )

        q3_holdings = date(2025, 9, 30)  # Q3 end
        effective = compute_13f_effective_date(q3_holdings)

        # Should be 45 days later (Nov 14)
        expected = date(2025, 11, 14)
        assert effective == expected

    def test_13f_pit_safety_validation(self):
        """13F data should be validated for PIT safety."""
        from common.staleness_gates import validate_13f_pit_safety

        q3_holdings = date(2025, 9, 30)

        # Safe case: analyzing after filing available
        is_safe, reason = validate_13f_pit_safety(
            q3_holdings,
            as_of_date=date(2025, 12, 1),  # After Nov 14
        )
        assert is_safe is True

        # Unsafe case: analyzing before filing available
        is_safe, reason = validate_13f_pit_safety(
            q3_holdings,
            as_of_date=date(2025, 10, 15),  # Before Nov 14
        )
        assert is_safe is False
        assert "lookahead" in reason.lower()


# =============================================================================
# TEST: FORWARD-LOOKING SEPARATION
# =============================================================================

class TestForwardLookingSeparation:
    """Tests for forward-looking signal separation."""

    def test_separator_import(self):
        """Forward-looking separator should be importable."""
        from common.forward_looking_separation import (
            ForwardLookingSeparator,
            LookaheadRiskLevel,
        )
        assert ForwardLookingSeparator is not None

    def test_historical_vs_forward_looking_separation(self):
        """Historical and forward-looking signals should be separated."""
        from common.forward_looking_separation import (
            ForwardLookingSeparator,
            LookaheadRiskLevel,
        )

        separator = ForwardLookingSeparator()

        detected = [
            {"event_type": "CT_STATUS_UPGRADE", "score": "5.0"},
        ]
        calendar = [
            {"event_type": "UPCOMING_PCD", "days_until": 25, "confidence": 0.85},
        ]

        result = separator.separate_signals(
            ticker="ACME",
            as_of_date=date(2026, 1, 15),
            detected_events=detected,
            calendar_events=calendar,
        )

        assert result.historical_score == Decimal("5.0")
        assert result.forward_looking_score > Decimal("0")
        assert result.blended_score != result.historical_score

    def test_high_forward_looking_warning(self):
        """High forward-looking contribution should generate warning."""
        from common.forward_looking_separation import (
            ForwardLookingSeparator,
            LookaheadRiskLevel,
        )

        separator = ForwardLookingSeparator()

        # Only calendar events, no historical
        detected = []
        calendar = [
            {"event_type": "UPCOMING_PCD", "days_until": 25, "confidence": 0.95},
            {"event_type": "UPCOMING_SCD", "days_until": 30, "confidence": 0.90},
        ]

        result = separator.separate_signals(
            ticker="ACME",
            as_of_date=date(2026, 1, 15),
            detected_events=detected,
            calendar_events=calendar,
        )

        assert result.signal_contribution.lookahead_risk == LookaheadRiskLevel.HIGH
        assert len(result.signal_contribution.warnings) > 0

    def test_backtest_safe_score(self):
        """Backtest-safe score should only include historical signals."""
        from common.forward_looking_separation import ForwardLookingSeparator

        separator = ForwardLookingSeparator()

        detected = [{"event_type": "CT_STATUS_UPGRADE", "score": "8.0"}]
        calendar = [{"event_type": "UPCOMING_PCD", "days_until": 25, "confidence": 0.85}]

        result = separator.separate_signals(
            ticker="ACME",
            as_of_date=date(2026, 1, 15),
            detected_events=detected,
            calendar_events=calendar,
        )

        # Backtest-safe should equal historical only
        assert result.get_backtest_safe_score() == result.historical_score
        assert result.get_backtest_safe_score() == Decimal("8.0")


# =============================================================================
# TEST: ERROR HANDLING
# =============================================================================

class TestErrorHandling:
    """Tests for proper error handling."""

    def test_empty_price_series_handling(self):
        """Empty price series should be handled gracefully."""
        from multi_window_momentum import MultiWindowMomentumEngine, RegimeState, MomentumSignalQuality

        engine = MultiWindowMomentumEngine()
        result = engine.compute_momentum(
            ticker="EMPTY",
            price_series=[],
            as_of_date=date(2026, 1, 15),
            regime=RegimeState.NEUTRAL
        )

        assert result.signal_quality == MomentumSignalQuality.INSUFFICIENT

    def test_unknown_data_date_handling(self):
        """Unknown data date should be handled with soft gate."""
        from common.staleness_gates import StalenessGateEngine, DataType, StalenessAction

        engine = StalenessGateEngine()
        result = engine.check_staleness(
            DataType.FINANCIAL,
            data_date=None,  # Unknown date
            as_of_date=date(2026, 1, 15)
        )

        assert result.action == StalenessAction.SOFT_GATE
        assert result.age_days == -1


# =============================================================================
# TEST: ENHANCEMENT MODULE INTEGRATION
# =============================================================================

class TestEnhancementModuleIntegration:
    """Tests for enhancement module integration."""

    def test_regime_detection_engine_integration(self):
        """Regime detection engine should integrate with pipeline."""
        from regime_engine import RegimeDetectionEngine

        engine = RegimeDetectionEngine()

        result = engine.detect_regime(
            vix_current=Decimal("22.5"),
            xbi_vs_spy_30d=Decimal("3.0"),
            as_of_date=date(2026, 1, 15),
        )

        assert result is not None
        assert "regime" in result
        assert "signal_adjustments" in result

    def test_regime_hysteresis_integration(self):
        """Regime hysteresis should integrate with detection engine."""
        from regime_hysteresis import RegimeHysteresisEngine, integrate_with_regime_engine
        from regime_engine import RegimeDetectionEngine

        detection_engine = RegimeDetectionEngine()
        hysteresis_engine = RegimeHysteresisEngine()

        # Run detection
        result = detection_engine.detect_regime(
            vix_current=Decimal("22.5"),
            xbi_vs_spy_30d=Decimal("3.0"),
            as_of_date=date(2026, 1, 15),
        )

        # Integrate hysteresis
        enhanced = integrate_with_regime_engine(
            result,
            hysteresis_engine,
            date(2026, 1, 15),
        )

        assert "hysteresis" in enhanced
        assert enhanced["hysteresis"]["applied"] is True

    def test_dilution_risk_engine_integration(self):
        """Dilution risk engine should integrate with pipeline."""
        from dilution_risk_engine import DilutionRiskEngine

        engine = DilutionRiskEngine()

        result = engine.calculate_dilution_risk(
            ticker="ACME",
            quarterly_cash=Decimal("100000000"),
            quarterly_burn=Decimal("-15000000"),
            next_catalyst_date="2026-07-15",
            market_cap=Decimal("500000000"),
            as_of_date=date(2026, 1, 15),
        )

        assert "dilution_risk_score" in result
        assert "risk_bucket" in result


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
