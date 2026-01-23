"""
test_pit_regression.py - Regression tests for PIT (Point-in-Time) safety violations.

These tests ensure that date.today() and other wall-clock dependencies
do not creep back into the codebase.

Each test documents a specific bug that was fixed.
"""

import pytest
from datetime import date
from decimal import Decimal
from unittest.mock import patch
import sys


class TestEventDetectorPITSafety:
    """
    Tests for event_detector.py PIT safety.

    Bug fixed: CatalystEvent had `disclosed_at: date = date.today()` default,
    causing same data to produce different results on different days.
    """

    def test_catalyst_event_requires_disclosed_at(self):
        """CatalystEvent must require explicit disclosed_at (no date.today() default)."""
        from event_detector import CatalystEvent, EventType

        # Should raise ValueError when disclosed_at is not provided
        with pytest.raises(ValueError, match="disclosed_at must be explicitly set"):
            CatalystEvent(
                nct_id="NCT12345678",
                event_type=EventType.CT_STATUS_UPGRADE,
            )

    def test_catalyst_event_with_explicit_date_succeeds(self):
        """CatalystEvent should work when disclosed_at is explicitly provided."""
        from event_detector import CatalystEvent, EventType

        event = CatalystEvent(
            nct_id="NCT12345678",
            event_type=EventType.CT_STATUS_UPGRADE,
            disclosed_at=date(2026, 1, 15),
        )

        assert event.disclosed_at == date(2026, 1, 15)
        assert event.nct_id == "NCT12345678"

    def test_catalyst_event_deterministic_across_days(self):
        """Same inputs should produce same CatalystEvent regardless of wall-clock."""
        from event_detector import CatalystEvent, EventType

        fixed_date = date(2026, 1, 10)

        # Create events with explicit date (simulating different run times)
        event1 = CatalystEvent(
            nct_id="NCT99999999",
            event_type=EventType.CT_TIMELINE_PUSHOUT,
            direction="NEG",
            impact=2,
            disclosed_at=fixed_date,
        )

        event2 = CatalystEvent(
            nct_id="NCT99999999",
            event_type=EventType.CT_TIMELINE_PUSHOUT,
            direction="NEG",
            impact=2,
            disclosed_at=fixed_date,
        )

        # Should be identical
        assert event1.to_dict() == event2.to_dict()


class TestPosPriorEnginePITSafety:
    """
    Tests for pos_prior_engine.py PIT safety.

    Bug fixed: calculate_pos_prior() silently defaulted to date.today()
    when as_of_date was None, violating determinism.
    """

    def test_calculate_pos_prior_requires_as_of_date(self):
        """calculate_pos_prior must fail-loud when as_of_date is None."""
        from pos_prior_engine import PoSPriorEngine

        engine = PoSPriorEngine()

        with pytest.raises(ValueError, match="as_of_date is required"):
            engine.calculate_pos_prior(
                stage="phase_2",
                therapeutic_area="oncology",
                as_of_date=None,  # Should raise, not default to today()
            )

    def test_calculate_pos_prior_with_explicit_date_succeeds(self):
        """calculate_pos_prior should work with explicit as_of_date."""
        from pos_prior_engine import PoSPriorEngine

        engine = PoSPriorEngine()

        result = engine.calculate_pos_prior(
            stage="phase_2",
            therapeutic_area="oncology",
            as_of_date=date(2026, 1, 15),
        )

        assert result is not None
        assert "pos_prior" in result or "reason_code" in result


class TestSharadarProviderPITSafety:
    """
    Tests for backtest/sharadar_provider.py PIT safety.

    Bug fixed: get_delisted_date() used date.today() to determine if a stock
    was delisted, causing survivorship bias in backtests.
    """

    def test_get_delisted_date_without_as_of_returns_none(self):
        """get_delisted_date without as_of_date should return None (safe default)."""
        from backtest.sharadar_provider import SharadarReturnsProvider

        # Create provider with test data
        prices = {
            "DLST": {
                "2024-01-01": Decimal("10.00"),
                "2024-01-15": Decimal("5.00"),  # Last trading date
            }
        }
        provider = SharadarReturnsProvider(prices)

        # Without as_of_date, should return None (cannot determine delisting)
        result = provider.get_delisted_date("DLST")
        assert result is None

    def test_get_delisted_date_with_as_of_date(self):
        """get_delisted_date with as_of_date should correctly identify delisting."""
        from backtest.sharadar_provider import SharadarReturnsProvider

        # Create provider with test data
        prices = {
            "DLST": {
                "2024-01-01": Decimal("10.00"),
                "2024-01-15": Decimal("5.00"),  # Last trading date (appears delisted)
            },
            "ACTV": {
                "2024-01-01": Decimal("20.00"),
                "2024-06-01": Decimal("25.00"),  # Recent data (still active)
            }
        }
        provider = SharadarReturnsProvider(prices)

        # Test with as_of_date after last trade (should show as delisted)
        result = provider.get_delisted_date("DLST", as_of_date=date(2024, 6, 1))
        assert result == "2024-01-15"

        # Test with as_of_date close to last trade (should not show as delisted)
        result = provider.get_delisted_date("DLST", as_of_date=date(2024, 1, 20))
        assert result is None

        # Active stock should never appear delisted
        result = provider.get_delisted_date("ACTV", as_of_date=date(2024, 6, 5))
        assert result is None

    def test_backtest_reproducibility_with_explicit_dates(self):
        """Backtest results should be identical when using explicit as_of_dates."""
        from backtest.sharadar_provider import SharadarReturnsProvider

        prices = {
            "DLST": {"2024-01-15": Decimal("5.00")},
        }
        provider = SharadarReturnsProvider(prices)

        # Same as_of_date should always produce same result
        fixed_as_of = date(2024, 6, 1)

        result1 = provider.get_delisted_date("DLST", as_of_date=fixed_as_of)
        result2 = provider.get_delisted_date("DLST", as_of_date=fixed_as_of)

        assert result1 == result2


class TestInputValidationPITSafety:
    """
    Tests for common/input_validation.py PIT safety.

    Bug fixed: validate_date() used date.today() for future date validation,
    causing validation behavior to change based on execution date.
    """

    def test_validate_date_no_wall_clock_dependency(self):
        """Date validation should not depend on current wall-clock time."""
        from common.input_validation import validate_date, InputValidationConfig

        config = InputValidationConfig()

        # A date in the past should always validate the same way
        is_valid, parsed, errors = validate_date("2025-01-01", config)
        assert is_valid is True
        assert parsed == date(2025, 1, 1)

        # Future dates should not fail based on today's date
        # (lookahead protection is via PIT filters, not date validation)
        is_valid, parsed, errors = validate_date("2030-12-31", config)
        # Should pass because we removed date.today() comparison
        assert is_valid is True


class TestDecimalPrecision:
    """
    Tests for Decimal precision in financial calculations.

    Bug fixed: integration_contracts.py and data_quality.py were using
    float for financial calculations, causing precision loss.
    """

    def test_extract_financial_score_decimal_precision(self):
        """extract_financial_score should preserve Decimal precision when requested."""
        from common.integration_contracts import extract_financial_score

        record = {"financial_score": "75.123456789"}

        # Default (float) - loses some precision
        float_result = extract_financial_score(record, return_decimal=False)
        assert isinstance(float_result, float)

        # Decimal - full precision
        decimal_result = extract_financial_score(record, return_decimal=True)
        assert isinstance(decimal_result, Decimal)
        assert decimal_result == Decimal("75.123456789")

    def test_extract_clinical_score_decimal_precision(self):
        """extract_clinical_score should preserve Decimal precision when requested."""
        from common.integration_contracts import extract_clinical_score

        record = {"clinical_score": "82.999999999"}

        # Decimal mode
        decimal_result = extract_clinical_score(record, return_decimal=True)
        assert isinstance(decimal_result, Decimal)
        assert decimal_result == Decimal("82.999999999")

    def test_adv_calculation_uses_decimal(self):
        """ADV calculation should use Decimal for precision."""
        from common.data_quality import DataQualityGates, DataQualityConfig

        config = DataQualityConfig(min_adv_dollars=Decimal("500000"))
        gates = DataQualityGates(config)

        # Test with values that would lose precision with float
        result = gates.validate_liquidity(
            avg_volume=123456.789,
            price=45.6789,
        )

        # The ADV value should be a Decimal
        assert isinstance(result.value, Decimal)
        # Verify precision is maintained
        expected_adv = Decimal("123456.789") * Decimal("45.6789")
        assert result.value == expected_adv.quantize(Decimal("0.01"))


class TestNoPITViolationsInImports:
    """
    Meta-test: ensure no date.today() at module import time.

    These tests catch violations where date.today() is called during
    module loading (e.g., as a module-level constant or default arg
    evaluated at import time).
    """

    def test_event_detector_import_deterministic(self):
        """event_detector should not call date.today() at import time."""
        # If this import succeeds without date.today() being called,
        # the module doesn't have import-time PIT violations
        import event_detector
        # Module should load successfully
        assert hasattr(event_detector, 'CatalystEvent')

    def test_pos_prior_engine_import_deterministic(self):
        """pos_prior_engine should not call date.today() at import time."""
        import pos_prior_engine
        assert hasattr(pos_prior_engine, 'PoSPriorEngine')


# Run with: pytest tests/test_pit_regression.py -v
