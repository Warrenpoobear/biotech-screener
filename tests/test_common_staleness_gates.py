#!/usr/bin/env python3
"""
Unit tests for common/staleness_gates.py

Tests phase-dependent data staleness enforcement:
- Staleness thresholds by data type and phase
- Action determination (PASS, WARN, SOFT_GATE, HARD_GATE)
- Penalty multiplier calculation
- Pipeline health checks
- 13F PIT safety validation
- Future date (lookahead bias) detection
"""

import pytest
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.staleness_gates import (
    StalenessGateEngine,
    StalenessAction,
    DataType,
    StalenessThreshold,
    StalenessCheckResult,
    DEFAULT_STALENESS_THRESHOLDS,
    SEC_13F_FILING_LAG_DAYS,
    compute_13f_effective_date,
    validate_13f_pit_safety,
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def engine():
    """Standard staleness gate engine."""
    return StalenessGateEngine()


@pytest.fixture
def strict_engine():
    """Engine with strict mode enabled."""
    return StalenessGateEngine(strict_mode=True)


@pytest.fixture
def as_of_date():
    """Standard analysis date."""
    return date(2026, 1, 15)


# ============================================================================
# ENGINE INITIALIZATION TESTS
# ============================================================================

class TestEngineInitialization:
    """Tests for StalenessGateEngine initialization."""

    def test_default_thresholds(self, engine):
        """Should use default thresholds."""
        assert len(engine.thresholds) == len(DEFAULT_STALENESS_THRESHOLDS)

    def test_custom_thresholds(self):
        """Should accept custom thresholds."""
        custom = [
            StalenessThreshold(DataType.FINANCIAL, None, 10, 20, 30, Decimal("0.5")),
        ]
        engine = StalenessGateEngine(thresholds=custom)
        assert len(engine.thresholds) == 1

    def test_strict_mode_flag(self):
        """Should respect strict mode flag."""
        normal = StalenessGateEngine(strict_mode=False)
        strict = StalenessGateEngine(strict_mode=True)
        assert normal.strict_mode is False
        assert strict.strict_mode is True


# ============================================================================
# STALENESS CHECK TESTS
# ============================================================================

class TestStalenessCheck:
    """Tests for check_staleness method."""

    def test_fresh_data_passes(self, engine, as_of_date):
        """Fresh data should pass."""
        data_date = as_of_date - timedelta(days=10)
        result = engine.check_staleness(
            DataType.FINANCIAL, data_date, as_of_date
        )
        assert result.action == StalenessAction.PASS
        assert result.penalty_multiplier == Decimal("1.0")

    def test_warning_threshold(self, engine, as_of_date):
        """Data past warning threshold should warn."""
        # Financial warning is 60 days
        data_date = as_of_date - timedelta(days=70)
        result = engine.check_staleness(
            DataType.FINANCIAL, data_date, as_of_date
        )
        assert result.action == StalenessAction.WARN
        assert result.penalty_multiplier == Decimal("1.0")

    def test_soft_gate_threshold(self, engine, as_of_date):
        """Data past soft gate should apply penalty."""
        # Financial soft gate is 90 days
        data_date = as_of_date - timedelta(days=100)
        result = engine.check_staleness(
            DataType.FINANCIAL, data_date, as_of_date
        )
        assert result.action == StalenessAction.SOFT_GATE
        assert result.penalty_multiplier < Decimal("1.0")

    def test_hard_gate_threshold(self, engine, as_of_date):
        """Data past hard gate should trigger exclusion."""
        # Financial hard gate is 120 days
        data_date = as_of_date - timedelta(days=150)
        result = engine.check_staleness(
            DataType.FINANCIAL, data_date, as_of_date
        )
        assert result.action == StalenessAction.HARD_GATE
        assert result.penalty_multiplier == Decimal("0")

    def test_unknown_date_soft_gates(self, engine, as_of_date):
        """Unknown data date should soft gate."""
        result = engine.check_staleness(
            DataType.FINANCIAL, None, as_of_date
        )
        assert result.action == StalenessAction.SOFT_GATE
        assert result.age_days == -1

    def test_future_date_hard_gates(self, engine, as_of_date):
        """Future data should hard gate (lookahead bias)."""
        future_date = as_of_date + timedelta(days=10)
        result = engine.check_staleness(
            DataType.FINANCIAL, future_date, as_of_date
        )
        assert result.action == StalenessAction.HARD_GATE
        assert "LOOKAHEAD BIAS" in result.message


# ============================================================================
# PHASE-DEPENDENT THRESHOLD TESTS
# ============================================================================

class TestPhaseDependentThresholds:
    """Tests for phase-dependent staleness thresholds."""

    def test_phase_3_stricter_than_phase_1(self, engine, as_of_date):
        """Phase 3 trials should have stricter thresholds."""
        # 150 days old trial data
        data_date = as_of_date - timedelta(days=150)

        # Phase 3: hard gate at 180 days
        p3_result = engine.check_staleness(
            DataType.TRIAL, data_date, as_of_date, phase="phase_3"
        )

        # Phase 1: hard gate at 545 days
        p1_result = engine.check_staleness(
            DataType.TRIAL, data_date, as_of_date, phase="phase_1"
        )

        # Same data should be more serious for Phase 3
        assert p3_result.action in (StalenessAction.SOFT_GATE, StalenessAction.WARN)
        assert p1_result.action == StalenessAction.PASS

    def test_unknown_phase_uses_default(self, engine, as_of_date):
        """Unknown phase should use default thresholds."""
        data_date = as_of_date - timedelta(days=200)

        result = engine.check_staleness(
            DataType.TRIAL, data_date, as_of_date, phase=None
        )

        # Should get some result (default thresholds)
        assert result.action in StalenessAction


# ============================================================================
# STRICT MODE TESTS
# ============================================================================

class TestStrictMode:
    """Tests for strict mode behavior."""

    def test_soft_gate_becomes_hard_gate(self, strict_engine, as_of_date):
        """In strict mode, soft gates become hard gates."""
        # Financial soft gate is 90 days
        data_date = as_of_date - timedelta(days=100)

        result = strict_engine.check_staleness(
            DataType.FINANCIAL, data_date, as_of_date
        )

        assert result.action == StalenessAction.HARD_GATE


# ============================================================================
# PIPELINE STALENESS CHECK TESTS
# ============================================================================

class TestPipelineStalenessCheck:
    """Tests for check_pipeline_staleness method."""

    def test_all_fresh(self, engine, as_of_date):
        """All fresh data should pass."""
        results = engine.check_pipeline_staleness(
            as_of_date=as_of_date,
            financial_date=as_of_date - timedelta(days=30),
            trial_date=as_of_date - timedelta(days=60),
            market_date=as_of_date - timedelta(days=1),
        )

        assert all(r.action == StalenessAction.PASS for r in results.values())

    def test_returns_dict_by_type(self, engine, as_of_date):
        """Should return results keyed by data type."""
        results = engine.check_pipeline_staleness(
            as_of_date=as_of_date,
            financial_date=as_of_date - timedelta(days=30),
        )

        assert "financial" in results
        assert isinstance(results["financial"], StalenessCheckResult)

    def test_only_checks_provided_dates(self, engine, as_of_date):
        """Should only check data types with provided dates."""
        results = engine.check_pipeline_staleness(
            as_of_date=as_of_date,
            financial_date=as_of_date - timedelta(days=30),
            # trial_date not provided
        )

        assert "financial" in results
        assert "trial" not in results


# ============================================================================
# PIPELINE HEALTH TESTS
# ============================================================================

class TestPipelineHealth:
    """Tests for get_pipeline_health method."""

    def test_all_ok(self, engine, as_of_date):
        """All fresh data should be OK status."""
        results = engine.check_pipeline_staleness(
            as_of_date=as_of_date,
            financial_date=as_of_date - timedelta(days=30),
            market_date=as_of_date - timedelta(days=1),
        )

        status, errors, warnings = engine.get_pipeline_health(results)

        assert status == "OK"
        assert len(errors) == 0
        assert len(warnings) == 0

    def test_degraded_status(self, engine, as_of_date):
        """Soft gates should result in DEGRADED status."""
        results = engine.check_pipeline_staleness(
            as_of_date=as_of_date,
            financial_date=as_of_date - timedelta(days=100),  # Soft gate
        )

        status, errors, warnings = engine.get_pipeline_health(results)

        assert status == "DEGRADED"
        assert len(warnings) > 0

    def test_fail_status(self, engine, as_of_date):
        """Hard gates should result in FAIL status."""
        results = engine.check_pipeline_staleness(
            as_of_date=as_of_date,
            financial_date=as_of_date - timedelta(days=150),  # Hard gate
        )

        status, errors, warnings = engine.get_pipeline_health(results)

        assert status == "FAIL"
        assert len(errors) > 0


# ============================================================================
# 13F PIT SAFETY TESTS
# ============================================================================

class TestCompute13fEffectiveDate:
    """Tests for compute_13f_effective_date function."""

    def test_default_lag(self):
        """Should apply default 45-day lag."""
        held_date = date(2025, 9, 30)  # Q3 holdings
        effective = compute_13f_effective_date(held_date)

        expected = held_date + timedelta(days=SEC_13F_FILING_LAG_DAYS)
        assert effective == expected

    def test_actual_filing_date(self):
        """Should use actual filing date if provided."""
        held_date = date(2025, 9, 30)
        filing_date = date(2025, 11, 10)

        effective = compute_13f_effective_date(held_date, filing_date)

        assert effective == filing_date


class TestValidate13fPitSafety:
    """Tests for validate_13f_pit_safety function."""

    def test_safe_usage(self):
        """Holdings should be safe after effective date."""
        held_date = date(2025, 9, 30)
        as_of = date(2025, 12, 1)  # Well after Nov 14 effective date

        is_safe, reason = validate_13f_pit_safety(held_date, as_of)

        assert is_safe is True
        assert "PIT-safe" in reason

    def test_unsafe_usage_lookahead(self):
        """Holdings should be unsafe before effective date."""
        held_date = date(2025, 9, 30)
        as_of = date(2025, 10, 15)  # Before Nov 14 effective date

        is_safe, reason = validate_13f_pit_safety(held_date, as_of)

        assert is_safe is False
        assert "lookahead bias" in reason.lower()

    def test_with_actual_filing_date(self):
        """Should use actual filing date for safety check."""
        held_date = date(2025, 9, 30)
        filing_date = date(2025, 11, 5)  # Early filer
        as_of = date(2025, 11, 10)

        is_safe, reason = validate_13f_pit_safety(held_date, as_of, filing_date)

        assert is_safe is True


# ============================================================================
# STALENESS CHECK RESULT TESTS
# ============================================================================

class TestStalenessCheckResult:
    """Tests for StalenessCheckResult dataclass."""

    def test_to_dict(self, engine, as_of_date):
        """Should serialize to dict correctly."""
        data_date = as_of_date - timedelta(days=30)
        result = engine.check_staleness(
            DataType.FINANCIAL, data_date, as_of_date
        )

        d = result.to_dict()

        assert d["data_type"] == "financial"
        assert d["data_date"] == data_date.isoformat()
        assert d["as_of_date"] == as_of_date.isoformat()
        assert d["action"] == "PASS"
        assert isinstance(d["penalty_multiplier"], str)


# ============================================================================
# DATA TYPE COVERAGE TESTS
# ============================================================================

class TestDataTypeCoverage:
    """Tests ensuring all data types have thresholds."""

    def test_financial_thresholds_exist(self, engine, as_of_date):
        """Financial data type should have thresholds."""
        result = engine.check_staleness(
            DataType.FINANCIAL, as_of_date - timedelta(days=10), as_of_date
        )
        assert result is not None

    def test_trial_thresholds_exist(self, engine, as_of_date):
        """Trial data type should have thresholds."""
        result = engine.check_staleness(
            DataType.TRIAL, as_of_date - timedelta(days=10), as_of_date
        )
        assert result is not None

    def test_market_thresholds_exist(self, engine, as_of_date):
        """Market data type should have thresholds."""
        result = engine.check_staleness(
            DataType.MARKET, as_of_date - timedelta(days=1), as_of_date
        )
        assert result is not None

    def test_short_interest_thresholds_exist(self, engine, as_of_date):
        """Short interest data type should have thresholds."""
        result = engine.check_staleness(
            DataType.SHORT_INTEREST, as_of_date - timedelta(days=10), as_of_date
        )
        assert result is not None

    def test_holdings_thresholds_exist(self, engine, as_of_date):
        """13F holdings data type should have thresholds."""
        result = engine.check_staleness(
            DataType.HOLDINGS_13F, as_of_date - timedelta(days=30), as_of_date
        )
        assert result is not None


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Edge case tests for staleness gates."""

    def test_same_day_data(self, engine, as_of_date):
        """Same day data should pass."""
        result = engine.check_staleness(
            DataType.FINANCIAL, as_of_date, as_of_date
        )
        assert result.action == StalenessAction.PASS
        assert result.age_days == 0

    def test_one_day_old_data(self, engine, as_of_date):
        """One day old data should pass."""
        result = engine.check_staleness(
            DataType.FINANCIAL, as_of_date - timedelta(days=1), as_of_date
        )
        assert result.action == StalenessAction.PASS
        assert result.age_days == 1

    def test_market_data_very_fresh_requirement(self, engine, as_of_date):
        """Market data has very fresh requirements."""
        # 6 days old should soft gate (threshold is 5 days)
        result = engine.check_staleness(
            DataType.MARKET, as_of_date - timedelta(days=6), as_of_date
        )
        assert result.action == StalenessAction.SOFT_GATE

    def test_unknown_data_type_uses_fallback(self, engine, as_of_date):
        """Unknown data type/phase should use fallback threshold."""
        # Remove all thresholds to test fallback
        empty_engine = StalenessGateEngine(thresholds=[])
        result = empty_engine.check_staleness(
            DataType.FINANCIAL, as_of_date - timedelta(days=100), as_of_date
        )
        # Should still return a result using fallback
        assert result is not None
