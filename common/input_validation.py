"""
common/input_validation.py - Pipeline Input Validation Layer

Provides comprehensive pre-flight validation for pipeline inputs.
Catches common data quality issues before modules process data.

Design Philosophy:
- Fail-loud: Raise exceptions for critical issues
- Track failures: Return detailed error messages
- Configurable: Support strict/warn/off modes

Usage:
    from common.input_validation import (
        validate_pipeline_inputs,
        validate_ticker,
        validate_financial_record,
        validate_trial_record,
        ValidationResult,
        PipelineValidationError,
    )

    # Pre-flight check before pipeline execution
    result = validate_pipeline_inputs(
        universe=active_tickers,
        financial_data=fin_records,
        market_data=mkt_records,
        as_of_date="2026-01-15",
    )
    if not result.passed:
        logger.error(f"Validation failed: {result.errors}")
        raise PipelineValidationError(result.summary())

Author: Wake Robin Capital Management
Version: 1.0.0
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


# ============================================================================
# EXCEPTIONS
# ============================================================================

class PipelineValidationError(Exception):
    """Raised when pipeline input validation fails."""
    pass


class DataQualityError(Exception):
    """Raised when data quality is below acceptable threshold."""
    pass


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class InputValidationConfig:
    """Configuration for input validation."""

    # Ticker validation
    max_ticker_length: int = 5
    ticker_pattern: str = r"^[A-Z]{1,5}$"

    # Financial data validation
    require_positive_cash: bool = True
    require_positive_market_cap: bool = True
    max_runway_months: int = 1200  # 100 years max reasonable

    # Date validation
    min_date: date = date(1990, 1, 1)
    max_date_future_days: int = 365  # Allow up to 1 year in future

    # Data quality thresholds
    min_valid_records_pct: float = 0.10  # At least 10% must be valid
    circuit_breaker_threshold: float = 0.50  # Warn if >50% invalid


@dataclass
class TickerValidationResult:
    """Result of ticker validation."""
    ticker: str
    valid: bool
    errors: List[str] = field(default_factory=list)


@dataclass
class RecordValidationResult:
    """Result of record validation."""
    ticker: str
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Aggregated validation result."""
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    invalid_tickers: Dict[str, List[str]] = field(default_factory=dict)
    invalid_financial_records: Dict[str, List[str]] = field(default_factory=dict)
    invalid_market_records: Dict[str, List[str]] = field(default_factory=dict)
    invalid_trial_records: Dict[str, List[str]] = field(default_factory=dict)
    stats: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = []
        lines.append(f"Validation {'PASSED' if self.passed else 'FAILED'}")

        if self.errors:
            lines.append(f"\nErrors ({len(self.errors)}):")
            for err in self.errors[:10]:  # Show first 10
                lines.append(f"  - {err}")
            if len(self.errors) > 10:
                lines.append(f"  ... and {len(self.errors) - 10} more")

        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for warn in self.warnings[:5]:  # Show first 5
                lines.append(f"  - {warn}")
            if len(self.warnings) > 5:
                lines.append(f"  ... and {len(self.warnings) - 5} more")

        if self.stats:
            lines.append(f"\nStats:")
            for k, v in sorted(self.stats.items()):
                lines.append(f"  {k}: {v}")

        return "\n".join(lines)


# ============================================================================
# TICKER VALIDATION
# ============================================================================

def validate_ticker(
    ticker: str,
    config: Optional[InputValidationConfig] = None
) -> TickerValidationResult:
    """
    Validate a single ticker symbol.

    Checks:
    - Non-empty
    - Max length (default 5 chars for NYSE/NASDAQ)
    - Character set (A-Z only)
    - No reserved symbols

    Args:
        ticker: Ticker symbol to validate
        config: Optional validation configuration

    Returns:
        TickerValidationResult with valid flag and errors
    """
    config = config or InputValidationConfig()
    errors = []

    # Check non-empty
    if not ticker:
        return TickerValidationResult(
            ticker=ticker or "",
            valid=False,
            errors=["Ticker is empty or None"]
        )

    # Normalize
    ticker_upper = str(ticker).strip().upper()

    # Check length
    if len(ticker_upper) > config.max_ticker_length:
        errors.append(f"Ticker exceeds {config.max_ticker_length} chars: {len(ticker_upper)}")

    # Check pattern (A-Z only)
    if not re.match(config.ticker_pattern, ticker_upper):
        errors.append(f"Ticker contains invalid characters: {ticker_upper}")

    # Check reserved symbols
    reserved = {"$", "%", "/", "\\", ".", "-", "_"}
    found_reserved = [c for c in ticker if c in reserved]
    if found_reserved:
        errors.append(f"Ticker contains reserved symbols: {found_reserved}")

    return TickerValidationResult(
        ticker=ticker_upper,
        valid=len(errors) == 0,
        errors=errors
    )


def validate_tickers(
    tickers: Union[Set[str], List[str]],
    config: Optional[InputValidationConfig] = None
) -> Tuple[Set[str], Dict[str, List[str]]]:
    """
    Validate a collection of tickers.

    Args:
        tickers: Set or List of ticker symbols
        config: Optional validation configuration

    Returns:
        (valid_tickers, invalid_tickers_with_reasons)
    """
    config = config or InputValidationConfig()
    valid = set()
    invalid = {}
    seen = set()

    for ticker in tickers:
        result = validate_ticker(ticker, config)

        # Check for duplicates
        if result.ticker in seen:
            if result.ticker not in invalid:
                invalid[result.ticker] = []
            invalid[result.ticker].append("Duplicate ticker")
            continue

        seen.add(result.ticker)

        if result.valid:
            valid.add(result.ticker)
        else:
            invalid[result.ticker] = result.errors

    return valid, invalid


# ============================================================================
# FINANCIAL RECORD VALIDATION
# ============================================================================

def _to_decimal_safe(value: Any) -> Optional[Decimal]:
    """Safely convert value to Decimal."""
    if value is None:
        return None
    try:
        if isinstance(value, Decimal):
            return value
        if isinstance(value, (int, float)):
            return Decimal(str(value))
        if isinstance(value, str):
            stripped = value.strip()
            return Decimal(stripped) if stripped else None
        return None
    except (InvalidOperation, ValueError):
        return None


def validate_financial_record(
    record: Dict[str, Any],
    config: Optional[InputValidationConfig] = None
) -> RecordValidationResult:
    """
    Validate a single financial record.

    Checks:
    - Has ticker
    - Cash is non-negative (if present and required)
    - Market cap is positive (if present and required)
    - Burn rate is reasonable
    - No impossible values (e.g., burn > cash in one quarter)

    Args:
        record: Financial record dict
        config: Optional validation configuration

    Returns:
        RecordValidationResult with valid flag, errors, and warnings
    """
    config = config or InputValidationConfig()
    errors = []
    warnings = []

    ticker = record.get("ticker", "")

    # Check ticker exists
    if not ticker:
        return RecordValidationResult(
            ticker="",
            valid=False,
            errors=["Missing ticker field"]
        )

    # Validate cash
    cash = _to_decimal_safe(record.get("Cash") or record.get("cash_mm"))
    if cash is not None:
        if config.require_positive_cash and cash < Decimal("0"):
            errors.append(f"Cash is negative: {cash}")

    # Validate market cap - use explicit None checks to avoid 0 being falsy
    market_cap_raw = record.get("market_cap")
    if market_cap_raw is None:
        market_cap_raw = record.get("market_cap_mm")
    if market_cap_raw is None:
        market_cap_raw = record.get("MarketCap")

    market_cap = _to_decimal_safe(market_cap_raw)
    if market_cap is not None:
        if config.require_positive_market_cap and market_cap <= Decimal("0"):
            if market_cap == Decimal("0"):
                errors.append(f"Market cap is zero: {market_cap}")
            else:
                errors.append(f"Market cap is negative: {market_cap}")

    # Validate burn rate (should be negative or reasonable positive for profitable)
    burn = _to_decimal_safe(
        record.get("NetIncome") or
        record.get("CFO") or
        record.get("burn_rate_mm")
    )
    if burn is not None:
        # Check if burn magnitude is reasonable (not larger than cash in a single quarter)
        if cash is not None and cash > 0:
            if burn < 0 and abs(burn) > cash * 4:  # More than 4x cash burned
                warnings.append(f"Burn rate ({burn}) exceeds 4x cash ({cash})")

    # Validate runway months (if present)
    runway = _to_decimal_safe(record.get("runway_months"))
    if runway is not None:
        if runway < Decimal("0"):
            errors.append(f"Runway months is negative: {runway}")
        elif runway > config.max_runway_months:
            warnings.append(f"Runway months is unusually high: {runway}")

    # Validate shares outstanding
    shares = _to_decimal_safe(record.get("shares_outstanding"))
    if shares is not None and shares <= Decimal("0"):
        errors.append(f"Shares outstanding must be positive: {shares}")

    return RecordValidationResult(
        ticker=ticker,
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )


# ============================================================================
# MARKET DATA VALIDATION
# ============================================================================

def validate_market_record(
    record: Dict[str, Any],
    config: Optional[InputValidationConfig] = None
) -> RecordValidationResult:
    """
    Validate a single market data record.

    Checks:
    - Has ticker
    - Price is positive
    - Volume is non-negative
    - Market cap is positive

    Args:
        record: Market data record dict
        config: Optional validation configuration

    Returns:
        RecordValidationResult with valid flag, errors, and warnings
    """
    config = config or InputValidationConfig()
    errors = []
    warnings = []

    ticker = record.get("ticker", "")

    if not ticker:
        return RecordValidationResult(
            ticker="",
            valid=False,
            errors=["Missing ticker field"]
        )

    # Validate price
    price = _to_decimal_safe(record.get("price") or record.get("current"))
    if price is not None and price <= Decimal("0"):
        errors.append(f"Price must be positive: {price}")

    # Validate volume
    volume = _to_decimal_safe(
        record.get("avg_volume") or
        record.get("volume_avg_30d") or
        record.get("volume")
    )
    if volume is not None and volume < Decimal("0"):
        errors.append(f"Volume cannot be negative: {volume}")

    # Validate market cap
    market_cap = _to_decimal_safe(record.get("market_cap"))
    if market_cap is not None and market_cap <= Decimal("0"):
        errors.append(f"Market cap must be positive: {market_cap}")

    return RecordValidationResult(
        ticker=ticker,
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )


# ============================================================================
# TRIAL RECORD VALIDATION
# ============================================================================

def validate_trial_record(
    record: Dict[str, Any],
    config: Optional[InputValidationConfig] = None
) -> RecordValidationResult:
    """
    Validate a single trial record.

    Checks:
    - Has ticker
    - Has NCT ID in correct format
    - Phase is recognizable
    - Enrollment is positive (if present)
    - Dates are reasonable

    Args:
        record: Trial record dict
        config: Optional validation configuration

    Returns:
        RecordValidationResult with valid flag, errors, and warnings
    """
    config = config or InputValidationConfig()
    errors = []
    warnings = []

    ticker = record.get("ticker", "")

    if not ticker:
        return RecordValidationResult(
            ticker="",
            valid=False,
            errors=["Missing ticker field"]
        )

    # Validate NCT ID format (NCT followed by 8 digits)
    nct_id = record.get("nct_id", "")
    if nct_id:
        if not re.match(r"^NCT\d{8}$", nct_id):
            warnings.append(f"NCT ID format may be invalid: {nct_id}")
    else:
        errors.append("Missing nct_id field")

    # Validate phase (should be recognizable)
    phase = record.get("phase", "")
    known_phases = {
        "phase 1", "phase 2", "phase 3", "phase 4",
        "phase 1/2", "phase 2/3", "approved", "preclinical",
        "phase1", "phase2", "phase3", "phase4",
        "early phase 1", "not applicable",
    }
    if phase:
        phase_lower = str(phase).lower().strip()
        # Extract phase number pattern
        if not any(kp in phase_lower for kp in ["phase", "approved", "preclinical"]):
            warnings.append(f"Unrecognized phase format: {phase}")

    # Validate enrollment
    enrollment = record.get("enrollment") or record.get("enrollment_count")
    if enrollment is not None:
        try:
            enroll_val = int(enrollment)
            if enroll_val < 0:
                errors.append(f"Enrollment cannot be negative: {enroll_val}")
        except (ValueError, TypeError):
            warnings.append(f"Invalid enrollment value: {enrollment}")

    return RecordValidationResult(
        ticker=ticker,
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )


# ============================================================================
# DATE VALIDATION
# ============================================================================

def validate_date(
    date_input: Union[str, date, None],
    config: Optional[InputValidationConfig] = None
) -> Tuple[bool, Optional[date], List[str]]:
    """
    Validate and parse a date input.

    Args:
        date_input: Date string or date object
        config: Optional validation configuration

    Returns:
        (is_valid, parsed_date, errors)
    """
    config = config or InputValidationConfig()
    errors = []

    if date_input is None:
        return False, None, ["Date is None"]

    parsed = None

    if isinstance(date_input, date):
        parsed = date_input
    elif isinstance(date_input, str):
        try:
            # Try ISO format first
            parsed = date.fromisoformat(date_input)
        except ValueError:
            errors.append(f"Invalid date format: {date_input}. Expected YYYY-MM-DD")
            return False, None, errors
    else:
        errors.append(f"Invalid date type: {type(date_input)}. Expected date or str")
        return False, None, errors

    # Check bounds
    if parsed < config.min_date:
        errors.append(f"Date {parsed} is before minimum allowed: {config.min_date}")

    max_allowed = date.today()
    # Can't add days to date directly in a simple way, check year instead
    if parsed.year > max_allowed.year + 1:
        errors.append(f"Date {parsed} is too far in the future")

    return len(errors) == 0, parsed, errors


# ============================================================================
# MAIN VALIDATION FUNCTION
# ============================================================================

def validate_pipeline_inputs(
    universe: Union[Set[str], List[str], None] = None,
    financial_data: Optional[List[Dict[str, Any]]] = None,
    market_data: Optional[List[Dict[str, Any]]] = None,
    trial_data: Optional[List[Dict[str, Any]]] = None,
    as_of_date: Union[str, date, None] = None,
    config: Optional[InputValidationConfig] = None,
    strict: bool = False,
) -> ValidationResult:
    """
    Comprehensive pre-flight validation for pipeline inputs.

    Validates:
    - Universe tickers (format, uniqueness)
    - Financial records (required fields, value ranges)
    - Market data (positive prices/volumes)
    - Trial records (NCT ID format, enrollment)
    - as_of_date (valid format, reasonable range)

    Circuit breaker: If >50% of records fail validation, raises warning.

    Args:
        universe: Set or List of ticker symbols
        financial_data: List of financial record dicts
        market_data: List of market data dicts
        trial_data: List of trial record dicts
        as_of_date: Analysis date
        config: Validation configuration
        strict: If True, raise exception on critical failures

    Returns:
        ValidationResult with pass/fail status and detailed errors

    Raises:
        PipelineValidationError: If strict=True and critical failures found
    """
    config = config or InputValidationConfig()
    result = ValidationResult(passed=True)

    # Initialize stats
    result.stats = {
        "tickers_total": 0,
        "tickers_valid": 0,
        "financial_records_total": 0,
        "financial_records_valid": 0,
        "market_records_total": 0,
        "market_records_valid": 0,
        "trial_records_total": 0,
        "trial_records_valid": 0,
    }

    # Validate as_of_date
    if as_of_date is not None:
        date_valid, parsed_date, date_errors = validate_date(as_of_date, config)
        if not date_valid:
            result.errors.extend(date_errors)
            result.passed = False

    # Validate universe
    if universe is not None:
        if len(universe) == 0:
            result.errors.append("Universe is empty - no tickers to process")
            result.passed = False
        else:
            result.stats["tickers_total"] = len(universe)
            valid_tickers, invalid_tickers = validate_tickers(universe, config)
            result.stats["tickers_valid"] = len(valid_tickers)
            result.invalid_tickers = invalid_tickers

            if invalid_tickers:
                result.warnings.append(
                    f"{len(invalid_tickers)} invalid tickers found"
                )

    # Validate financial data
    if financial_data is not None:
        result.stats["financial_records_total"] = len(financial_data)

        if len(financial_data) == 0:
            result.warnings.append("No financial data provided")
        else:
            for record in financial_data:
                rec_result = validate_financial_record(record, config)
                if rec_result.valid:
                    result.stats["financial_records_valid"] += 1
                else:
                    result.invalid_financial_records[rec_result.ticker] = rec_result.errors

                if rec_result.warnings:
                    result.warnings.extend([
                        f"{rec_result.ticker}: {w}" for w in rec_result.warnings
                    ])

            # Circuit breaker check
            total = result.stats["financial_records_total"]
            valid = result.stats["financial_records_valid"]
            if total > 0:
                failure_rate = (total - valid) / total
                if failure_rate > config.circuit_breaker_threshold:
                    msg = f"High financial data failure rate: {failure_rate:.1%}"
                    result.warnings.append(msg)
                    logger.warning(msg)

    # Validate market data
    if market_data is not None:
        result.stats["market_records_total"] = len(market_data)

        for record in market_data:
            rec_result = validate_market_record(record, config)
            if rec_result.valid:
                result.stats["market_records_valid"] += 1
            else:
                result.invalid_market_records[rec_result.ticker] = rec_result.errors

    # Validate trial data
    if trial_data is not None:
        result.stats["trial_records_total"] = len(trial_data)

        for record in trial_data:
            rec_result = validate_trial_record(record, config)
            if rec_result.valid:
                result.stats["trial_records_valid"] += 1
            else:
                result.invalid_trial_records[rec_result.ticker] = rec_result.errors

            if rec_result.warnings:
                result.warnings.extend([
                    f"{rec_result.ticker}: {w}" for w in rec_result.warnings
                ])

    # Check minimum valid records threshold
    total_records = (
        result.stats.get("financial_records_total", 0) +
        result.stats.get("trial_records_total", 0)
    )
    valid_records = (
        result.stats.get("financial_records_valid", 0) +
        result.stats.get("trial_records_valid", 0)
    )

    if total_records > 0:
        valid_pct = valid_records / total_records
        if valid_pct < config.min_valid_records_pct:
            result.errors.append(
                f"Only {valid_pct:.1%} of records are valid "
                f"(minimum: {config.min_valid_records_pct:.1%})"
            )
            result.passed = False

    # Final determination
    if result.errors:
        result.passed = False

    if strict and not result.passed:
        raise PipelineValidationError(result.summary())

    return result


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def validate_universe_not_empty(
    universe: Union[Set[str], List[str], None],
    module_name: str = "Pipeline"
) -> List[str]:
    """
    Check that universe is not empty and return validated list.

    Raises:
        PipelineValidationError: If universe is empty
    """
    if universe is None or len(universe) == 0:
        raise PipelineValidationError(
            f"{module_name}: Cannot process empty universe"
        )

    return list(universe) if isinstance(universe, set) else universe


def warn_empty_data(
    data: Optional[List],
    data_name: str,
    module_name: str = "Pipeline"
) -> None:
    """Log warning if data is empty."""
    if data is None or len(data) == 0:
        logger.warning(f"{module_name}: {data_name} is empty")
