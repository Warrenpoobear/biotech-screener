"""
common/data_quality.py - Data Quality Gates and Validation

Provides validation gates for data quality control across the screening pipeline.

Quality Gates:
- Staleness: Financial data age limits
- Liquidity: Minimum ADV thresholds
- Price: Penny stock filters
- Enrollment: Minimum trial participants
- Coverage: Required field presence

Usage:
    from common.data_quality import DataQualityGates, validate_financial_staleness

    gates = DataQualityGates()
    result = gates.validate_ticker_data(ticker_data)
    if not result.passed:
        logger.warning(f"Quality gates failed: {result.failures}")
"""

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from common.date_utils import normalize_date

# Type alias for date-like inputs (ISO string or date object)
DateLike = Union[str, "date"]

logger = logging.getLogger(__name__)


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    passed: bool
    gate_name: str
    message: str
    value: Optional[Any] = None
    threshold: Optional[Any] = None


@dataclass
class ValidationResult:
    """Aggregated validation results for a ticker."""
    ticker: str
    passed: bool
    gate_results: List[QualityGateResult] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)

    @property
    def failures(self) -> List[QualityGateResult]:
        return [r for r in self.gate_results if not r.passed]

    @property
    def warnings(self) -> List[str]:
        return [r.message for r in self.gate_results if not r.passed]


@dataclass
class DataQualityConfig:
    """Configuration for data quality gates."""
    # Staleness thresholds (days)
    max_financial_age_days: int = 90
    max_market_data_age_days: int = 7
    max_trial_data_age_days: int = 30

    # Liquidity thresholds - Use Decimal for financial precision
    min_adv_dollars: Decimal = field(default_factory=lambda: Decimal("500000"))
    min_price: Decimal = field(default_factory=lambda: Decimal("5.0"))

    # Trial quality thresholds
    min_enrollment: int = 10  # Minimum trial participants

    # Coverage thresholds (0-1)
    min_financial_coverage: float = 0.5  # At least 50% of financial fields
    min_market_coverage: float = 0.8  # At least 80% of market fields

    # Required fields
    required_financial_fields: Set[str] = field(default_factory=lambda: {
        "Cash", "NetIncome"
    })
    required_market_fields: Set[str] = field(default_factory=lambda: {
        "price", "market_cap"
    })


class DataQualityGates:
    """
    Data quality validation gates for the screening pipeline.

    Validates:
    - Financial data staleness
    - Liquidity requirements
    - Price thresholds
    - Data coverage
    """

    def __init__(self, config: Optional[DataQualityConfig] = None) -> None:
        self.config = config or DataQualityConfig()

    def validate_financial_staleness(
        self,
        data_date: Optional[DateLike],
        as_of_date: DateLike,
    ) -> QualityGateResult:
        """
        Validate financial data is not too stale.

        Args:
            data_date: Date of the financial data
            as_of_date: Current analysis date

        Returns:
            QualityGateResult with pass/fail status
        """
        if data_date is None:
            return QualityGateResult(
                passed=False,
                gate_name="financial_staleness",
                message="Missing financial data date",
                value=None,
                threshold=self.config.max_financial_age_days,
            )

        try:
            data_dt = normalize_date(data_date)
            as_of_dt = normalize_date(as_of_date)
        except (ValueError, TypeError) as e:
            return QualityGateResult(
                passed=False,
                gate_name="financial_staleness",
                message=f"Invalid date format: {e}",
                value=str(data_date),
                threshold=self.config.max_financial_age_days,
            )

        age_days = (as_of_dt - data_dt).days

        if age_days > self.config.max_financial_age_days:
            return QualityGateResult(
                passed=False,
                gate_name="financial_staleness",
                message=f"Financial data is {age_days} days old (max: {self.config.max_financial_age_days})",
                value=age_days,
                threshold=self.config.max_financial_age_days,
            )

        return QualityGateResult(
            passed=True,
            gate_name="financial_staleness",
            message=f"Financial data is {age_days} days old",
            value=age_days,
            threshold=self.config.max_financial_age_days,
        )

    def validate_liquidity(
        self,
        avg_volume: Optional[float],
        price: Optional[float],
    ) -> QualityGateResult:
        """
        Validate minimum liquidity requirements.

        Args:
            avg_volume: Average daily trading volume (shares)
            price: Current stock price

        Returns:
            QualityGateResult with pass/fail status

        Note:
            Uses Decimal arithmetic internally for precision in ADV calculation.
        """
        if avg_volume is None or price is None:
            return QualityGateResult(
                passed=False,
                gate_name="liquidity",
                message="Missing volume or price data",
                value=None,
                threshold=self.config.min_adv_dollars,
            )

        if avg_volume <= 0 or price <= 0:
            return QualityGateResult(
                passed=False,
                gate_name="liquidity",
                message="Invalid volume or price (zero or negative)",
                value=Decimal("0"),
                threshold=self.config.min_adv_dollars,
            )

        # PRECISION: Use Decimal arithmetic for financial calculations
        # Converting via string avoids float representation errors
        vol_decimal = Decimal(str(avg_volume))
        price_decimal = Decimal(str(price))
        adv_dollars = (vol_decimal * price_decimal).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        min_threshold = self.config.min_adv_dollars
        if not isinstance(min_threshold, Decimal):
            min_threshold = Decimal(str(min_threshold))

        if adv_dollars < min_threshold:
            return QualityGateResult(
                passed=False,
                gate_name="liquidity",
                message=f"ADV ${adv_dollars:,.0f} below minimum ${min_threshold:,.0f}",
                value=adv_dollars,
                threshold=min_threshold,
            )

        return QualityGateResult(
            passed=True,
            gate_name="liquidity",
            message=f"ADV ${adv_dollars:,.0f}",
            value=adv_dollars,
            threshold=min_threshold,
        )

    def validate_price(self, price: Optional[float]) -> QualityGateResult:
        """
        Validate minimum price (penny stock filter).

        Args:
            price: Current stock price

        Returns:
            QualityGateResult with pass/fail status
        """
        if price is None:
            return QualityGateResult(
                passed=False,
                gate_name="price",
                message="Missing price data",
                value=None,
                threshold=self.config.min_price,
            )

        # Use Decimal for comparison
        price_decimal = Decimal(str(price))
        min_price = self.config.min_price
        if not isinstance(min_price, Decimal):
            min_price = Decimal(str(min_price))

        if price_decimal < min_price:
            return QualityGateResult(
                passed=False,
                gate_name="price",
                message=f"Price ${price_decimal:.2f} below minimum ${min_price:.2f}",
                value=price_decimal,
                threshold=min_price,
            )

        return QualityGateResult(
            passed=True,
            gate_name="price",
            message=f"Price ${price_decimal:.2f}",
            value=price_decimal,
            threshold=min_price,
        )

    def validate_enrollment(self, enrollment: Optional[int]) -> QualityGateResult:
        """
        Validate minimum trial enrollment.

        Args:
            enrollment: Number of trial participants

        Returns:
            QualityGateResult with pass/fail status
        """
        if enrollment is None:
            return QualityGateResult(
                passed=True,  # Missing enrollment is not a failure
                gate_name="enrollment",
                message="Enrollment not specified",
                value=None,
                threshold=self.config.min_enrollment,
            )

        if enrollment < self.config.min_enrollment:
            return QualityGateResult(
                passed=False,
                gate_name="enrollment",
                message=f"Enrollment {enrollment} below minimum {self.config.min_enrollment}",
                value=enrollment,
                threshold=self.config.min_enrollment,
            )

        return QualityGateResult(
            passed=True,
            gate_name="enrollment",
            message=f"Enrollment {enrollment}",
            value=enrollment,
            threshold=self.config.min_enrollment,
        )

    def validate_financial_coverage(
        self,
        financial_data: Dict[str, Any],
    ) -> QualityGateResult:
        """
        Validate required financial fields are present.

        Args:
            financial_data: Dict of financial metrics

        Returns:
            QualityGateResult with pass/fail status
        """
        if not financial_data:
            return QualityGateResult(
                passed=False,
                gate_name="financial_coverage",
                message="No financial data",
                value=0,
                threshold=len(self.config.required_financial_fields),
            )

        present = sum(
            1 for f in self.config.required_financial_fields
            if financial_data.get(f) is not None
        )
        required = len(self.config.required_financial_fields)
        coverage = present / required if required > 0 else 0

        if coverage < self.config.min_financial_coverage:
            missing = [
                f for f in self.config.required_financial_fields
                if financial_data.get(f) is None
            ]
            return QualityGateResult(
                passed=False,
                gate_name="financial_coverage",
                message=f"Missing financial fields: {missing}",
                value=coverage,
                threshold=self.config.min_financial_coverage,
            )

        return QualityGateResult(
            passed=True,
            gate_name="financial_coverage",
            message=f"Financial coverage {coverage:.0%}",
            value=coverage,
            threshold=self.config.min_financial_coverage,
        )

    def validate_ticker_data(
        self,
        ticker: str,
        financial_data: Optional[Dict[str, Any]] = None,
        market_data: Optional[Dict[str, Any]] = None,
        trial_data: Optional[Dict[str, Any]] = None,
        as_of_date: Optional[DateLike] = None,
    ) -> ValidationResult:
        """
        Run all quality gates for a ticker.

        Args:
            ticker: Stock ticker symbol
            financial_data: Financial metrics dict
            market_data: Market data dict
            trial_data: Trial/clinical data dict
            as_of_date: Analysis date for staleness checks

        Returns:
            ValidationResult with all gate results
        """
        results = []
        flags = []

        # Financial data gates
        if financial_data:
            # Staleness check
            if as_of_date and financial_data.get("data_date"):
                staleness = self.validate_financial_staleness(
                    financial_data["data_date"],
                    as_of_date,
                )
                results.append(staleness)
                if not staleness.passed:
                    flags.append("stale_financial_data")

            # Coverage check
            coverage = self.validate_financial_coverage(financial_data)
            results.append(coverage)
            if not coverage.passed:
                flags.append("incomplete_financial_data")

        # Market data gates
        if market_data:
            # Price check
            price = market_data.get("price") or market_data.get("current")
            price_result = self.validate_price(price)
            results.append(price_result)
            if not price_result.passed:
                flags.append("penny_stock")

            # Liquidity check
            avg_volume = market_data.get("avg_volume") or market_data.get("average_30d")
            liquidity_result = self.validate_liquidity(avg_volume, price)
            results.append(liquidity_result)
            if not liquidity_result.passed:
                flags.append("low_liquidity")

        # Trial data gates
        if trial_data:
            enrollment = trial_data.get("enrollment") or trial_data.get("enrollment_count")
            enrollment_result = self.validate_enrollment(enrollment)
            results.append(enrollment_result)
            if not enrollment_result.passed:
                flags.append("low_enrollment")

        # Determine overall pass/fail
        passed = all(r.passed for r in results)

        return ValidationResult(
            ticker=ticker,
            passed=passed,
            gate_results=results,
            flags=flags,
        )


def validate_financial_staleness(
    data_date: Optional[DateLike],
    as_of_date: DateLike,
    max_age_days: int = 90,
) -> bool:
    """
    Convenience function to check if financial data is stale.

    Args:
        data_date: Date of the financial data
        as_of_date: Current analysis date
        max_age_days: Maximum allowed age in days

    Returns:
        True if data is fresh (not stale), False if stale
    """
    gates = DataQualityGates(DataQualityConfig(max_financial_age_days=max_age_days))
    result = gates.validate_financial_staleness(data_date, as_of_date)
    return result.passed


def validate_liquidity(
    avg_volume: float,
    price: float,
    min_adv_dollars: Union[float, Decimal] = Decimal("500000"),
) -> bool:
    """
    Convenience function to check if liquidity meets minimum.

    Args:
        avg_volume: Average daily volume (shares)
        price: Stock price
        min_adv_dollars: Minimum ADV in dollars (Decimal preferred)

    Returns:
        True if liquidity is sufficient, False otherwise
    """
    # Convert to Decimal if needed
    if not isinstance(min_adv_dollars, Decimal):
        min_adv_dollars = Decimal(str(min_adv_dollars))
    gates = DataQualityGates(DataQualityConfig(min_adv_dollars=min_adv_dollars))
    result = gates.validate_liquidity(avg_volume, price)
    return result.passed


# ============================================================================
# CIRCUIT BREAKER
# ============================================================================

class CircuitBreakerError(Exception):
    """Raised when circuit breaker trips due to excessive failures."""
    pass


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    # Thresholds
    failure_threshold: float = 0.50  # Trip if > 50% of records fail
    warning_threshold: float = 0.20  # Warn if > 20% fail
    min_records_for_check: int = 10  # Don't check if fewer than 10 records

    # Behavior
    strict_mode: bool = False  # If True, raise exception on trip; else warn


@dataclass
class CircuitBreakerResult:
    """Result of circuit breaker check."""
    tripped: bool
    warning: bool
    failure_rate: float
    total_records: int
    failed_records: int
    message: str


def check_circuit_breaker(
    total_records: int,
    failed_records: int,
    config: Optional[CircuitBreakerConfig] = None,
    context: str = "data validation",
) -> CircuitBreakerResult:
    """
    Check if circuit breaker should trip due to excessive failures.

    Prevents pipeline from continuing when data quality is too poor.

    Args:
        total_records: Total number of records processed
        failed_records: Number of records that failed validation
        config: Circuit breaker configuration
        context: Description for error message

    Returns:
        CircuitBreakerResult with trip status and details

    Raises:
        CircuitBreakerError: If strict_mode is True and breaker trips
    """
    config = config or CircuitBreakerConfig()

    # Don't check if too few records
    if total_records < config.min_records_for_check:
        return CircuitBreakerResult(
            tripped=False,
            warning=False,
            failure_rate=0.0,
            total_records=total_records,
            failed_records=failed_records,
            message=f"Skipped circuit breaker check: only {total_records} records"
        )

    # Calculate failure rate
    failure_rate = failed_records / total_records if total_records > 0 else 0.0

    # Check thresholds
    tripped = failure_rate > config.failure_threshold
    warning = failure_rate > config.warning_threshold

    # Build message
    if tripped:
        message = (
            f"Circuit breaker TRIPPED: {failure_rate:.1%} of {total_records} records "
            f"failed {context} (threshold: {config.failure_threshold:.1%})"
        )
        logger.error(message)

        if config.strict_mode:
            raise CircuitBreakerError(message)
    elif warning:
        message = (
            f"Circuit breaker WARNING: {failure_rate:.1%} of {total_records} records "
            f"failed {context} (threshold: {config.warning_threshold:.1%})"
        )
        logger.warning(message)
    else:
        message = (
            f"Circuit breaker OK: {failure_rate:.1%} failure rate "
            f"({failed_records}/{total_records})"
        )

    return CircuitBreakerResult(
        tripped=tripped,
        warning=warning,
        failure_rate=failure_rate,
        total_records=total_records,
        failed_records=failed_records,
        message=message
    )


def validate_batch_with_circuit_breaker(
    records: List[Dict[str, Any]],
    validator_func: Callable[[Dict[str, Any]], Tuple[bool, List[str]]],
    config: Optional[CircuitBreakerConfig] = None,
    context: str = "batch validation",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], CircuitBreakerResult]:
    """
    Validate a batch of records with circuit breaker protection.

    Args:
        records: List of records to validate
        validator_func: Function that takes a record and returns (valid: bool, errors: List[str])
        config: Circuit breaker configuration
        context: Description for error messages

    Returns:
        Tuple of (valid_records, invalid_records, circuit_breaker_result)
    """
    valid_records = []
    invalid_records = []

    for record in records:
        is_valid, errors = validator_func(record)
        if is_valid:
            valid_records.append(record)
        else:
            invalid_records.append({
                "record": record,
                "errors": errors,
            })

    # Check circuit breaker
    cb_result = check_circuit_breaker(
        total_records=len(records),
        failed_records=len(invalid_records),
        config=config,
        context=context,
    )

    return valid_records, invalid_records, cb_result
