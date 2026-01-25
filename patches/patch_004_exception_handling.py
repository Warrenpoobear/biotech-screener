"""
PATCH 004: Proper Exception Handling
=====================================

HIGH FIX: Silent `except: pass` statements hide errors and make debugging
impossible. This patch provides proper exception handling patterns.

Files affected:
- defensive_overlay_adapter.py:47,89,126
- extend_universe_yfinance.py:153
- real_short_interest_feed.py:119
- module_5_composite_v3.py:1064,1070

Usage:
    from patches.patch_004_exception_handling import (
        safe_decimal_convert,
        safe_divide,
        log_exception,
        ExceptionAccumulator,
    )
"""

import logging
import traceback
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class ExceptionRecord:
    """Record of a caught exception."""
    location: str
    exception_type: str
    message: str
    context: Dict[str, Any]
    traceback_str: Optional[str] = None


class ExceptionAccumulator:
    """
    Accumulates exceptions instead of silently swallowing them.

    Use this when you want to continue processing despite errors,
    but still track all errors for debugging.

    Usage:
        accumulator = ExceptionAccumulator("defensive_overlay")

        for record in records:
            try:
                score = calculate_score(record)
            except Exception as e:
                accumulator.add(e, context={"ticker": record["ticker"]})
                score = default_score

        # At end of batch, log summary
        accumulator.log_summary()

        # Optionally raise if too many errors
        if accumulator.error_rate > 0.5:
            raise ExcessiveErrorsError(accumulator.get_report())
    """

    def __init__(
        self,
        context: str,
        max_individual_logs: int = 10,
        log_level: int = logging.WARNING,
    ):
        """
        Args:
            context: Name of the processing context (e.g., "defensive_overlay")
            max_individual_logs: Max number of individual exceptions to log
            log_level: Logging level for individual exceptions
        """
        self.context = context
        self.max_individual_logs = max_individual_logs
        self.log_level = log_level
        self.exceptions: List[ExceptionRecord] = []
        self.total_processed = 0

    def add(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None,
        include_traceback: bool = False,
    ) -> None:
        """
        Record an exception.

        Args:
            exception: The caught exception
            context: Additional context (e.g., ticker, field name)
            include_traceback: Whether to store full traceback
        """
        record = ExceptionRecord(
            location=self.context,
            exception_type=type(exception).__name__,
            message=str(exception),
            context=context or {},
            traceback_str=traceback.format_exc() if include_traceback else None,
        )
        self.exceptions.append(record)

        # Log individual exceptions up to limit
        if len(self.exceptions) <= self.max_individual_logs:
            ctx_str = ", ".join(f"{k}={v}" for k, v in (context or {}).items())
            logger.log(
                self.log_level,
                f"{self.context}: {record.exception_type}: {record.message}"
                f"{f' [{ctx_str}]' if ctx_str else ''}"
            )

    def increment_processed(self, count: int = 1) -> None:
        """Increment the total processed count."""
        self.total_processed += count

    @property
    def error_count(self) -> int:
        return len(self.exceptions)

    @property
    def error_rate(self) -> float:
        if self.total_processed == 0:
            return 0.0
        return self.error_count / self.total_processed

    def log_summary(self) -> None:
        """Log a summary of all accumulated exceptions."""
        if not self.exceptions:
            logger.info(f"{self.context}: {self.total_processed} processed, 0 errors")
            return

        # Group by exception type
        by_type: Dict[str, int] = {}
        for exc in self.exceptions:
            by_type[exc.exception_type] = by_type.get(exc.exception_type, 0) + 1

        type_summary = ", ".join(f"{t}:{c}" for t, c in sorted(by_type.items()))

        if self.error_rate > 0.1:
            log_func = logger.error
        elif self.error_rate > 0.01:
            log_func = logger.warning
        else:
            log_func = logger.info

        log_func(
            f"{self.context}: {self.total_processed} processed, "
            f"{self.error_count} errors ({self.error_rate:.1%}): {type_summary}"
        )

    def get_report(self) -> Dict[str, Any]:
        """Get a full report of all exceptions."""
        return {
            "context": self.context,
            "total_processed": self.total_processed,
            "error_count": self.error_count,
            "error_rate": self.error_rate,
            "exceptions": [
                {
                    "type": e.exception_type,
                    "message": e.message,
                    "context": e.context,
                }
                for e in self.exceptions
            ],
        }


def safe_decimal_convert(
    value: Any,
    default: Decimal = Decimal("0"),
    context: Optional[str] = None,
) -> Decimal:
    """
    Safely convert a value to Decimal.

    REPLACES:
        try:
            result = Decimal(str(value))
        except Exception:
            pass  # Silent failure!

    WITH:
        result = safe_decimal_convert(value, default=Decimal("0"), context="field_name")

    Args:
        value: Value to convert
        default: Default if conversion fails
        context: Description for logging

    Returns:
        Decimal value or default
    """
    if value is None:
        return default

    try:
        # Handle various input types
        if isinstance(value, Decimal):
            return value
        if isinstance(value, (int, float)):
            # Avoid float precision issues by going through string
            return Decimal(str(value))
        if isinstance(value, str):
            return Decimal(value)

        # Unknown type - try string conversion
        return Decimal(str(value))

    except (InvalidOperation, ValueError, TypeError) as e:
        if context:
            logger.debug(
                f"Failed to convert {context}={value!r} to Decimal: {e}. "
                f"Using default={default}"
            )
        return default


def safe_divide(
    numerator: Union[Decimal, float, int],
    denominator: Union[Decimal, float, int],
    default: Decimal = Decimal("0"),
    context: Optional[str] = None,
) -> Decimal:
    """
    Safely divide with Decimal arithmetic.

    Handles:
    - Division by zero
    - Invalid types
    - None values

    Args:
        numerator: Dividend
        denominator: Divisor
        default: Default if division fails
        context: Description for logging

    Returns:
        Result of division or default
    """
    try:
        num = safe_decimal_convert(numerator, context=f"{context}_numerator" if context else None)
        den = safe_decimal_convert(denominator, context=f"{context}_denominator" if context else None)

        if den == Decimal("0"):
            if context:
                logger.debug(f"Division by zero in {context}, using default={default}")
            return default

        return num / den

    except Exception as e:
        if context:
            logger.debug(f"Division failed in {context}: {e}, using default={default}")
        return default


def log_exception(
    func: Callable[..., T]
) -> Callable[..., T]:
    """
    Decorator that logs exceptions before re-raising.

    Use this instead of bare `try/except` to ensure exceptions are logged.

    Usage:
        @log_exception
        def calculate_score(record):
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(
                f"Exception in {func.__name__}: {type(e).__name__}: {e}",
                exc_info=True,
            )
            raise

    return wrapper


def with_fallback(
    default: T,
    log_errors: bool = True,
    context: Optional[str] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that returns a default value on exception.

    Use this instead of `try/except: pass` when you have a sensible default.

    Usage:
        @with_fallback(default=Decimal("0"), context="correlation_calc")
        def calculate_correlation(prices):
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    ctx = context or func.__name__
                    logger.warning(
                        f"{ctx}: {type(e).__name__}: {e}. Using default={default}"
                    )
                return default

        return wrapper

    return decorator


# =============================================================================
# FIXED defensive_overlay_adapter.sanitize_corr
# =============================================================================

def sanitize_corr_fixed(
    defensive_features: Dict[str, str]
) -> tuple[Optional[Decimal], List[str]]:
    """
    Fixed version of sanitize_corr that logs errors instead of swallowing them.

    REPLACES (defensive_overlay_adapter.py:45-49):
        try:
            corr = Decimal(str(corr_s))
        except Exception:
            flags.append("def_corr_parse_fail")
            return None, flags

    WITH proper error tracking.
    """
    flags: List[str] = []
    PLACEHOLDER_CORR = Decimal("0.50")

    # Try both field names
    corr_s = defensive_features.get("corr_xbi") or defensive_features.get("corr_xbi_120d")

    if not corr_s:
        flags.append("def_corr_missing")
        return None, flags

    try:
        corr = Decimal(str(corr_s))
    except (InvalidOperation, ValueError, TypeError) as e:
        # Log the actual error instead of just swallowing it
        logger.debug(
            f"Failed to parse correlation value '{corr_s}': "
            f"{type(e).__name__}: {e}"
        )
        flags.append(f"def_corr_parse_fail:{type(e).__name__}")
        return None, flags

    # Check for NaN/Inf
    if not corr.is_finite():
        flags.append("def_corr_not_finite")
        return None, flags

    # Check for placeholder
    if corr == PLACEHOLDER_CORR:
        flags.append("def_corr_placeholder_0.50")
        return None, flags

    # Validate range
    if corr < Decimal("-1") or corr > Decimal("1"):
        logger.warning(f"Correlation out of range: {corr}")
        flags.append("def_corr_out_of_range")
        return None, flags

    return corr, flags


# =============================================================================
# EXAMPLE: Fixed pattern for batch processing
# =============================================================================

def process_batch_with_error_tracking(
    records: List[Dict[str, Any]],
    processor: Callable[[Dict[str, Any]], Dict[str, Any]],
    error_threshold: float = 0.5,
) -> tuple[List[Dict[str, Any]], ExceptionAccumulator]:
    """
    Process a batch of records with proper error tracking.

    Instead of:
        for record in records:
            try:
                result = process(record)
            except:
                pass  # Silent failure!

    Use:
        results, errors = process_batch_with_error_tracking(
            records,
            processor=process_single_record,
            error_threshold=0.5,
        )

    Args:
        records: List of records to process
        processor: Function to process each record
        error_threshold: Raise if error rate exceeds this

    Returns:
        (successful_results, error_accumulator)

    Raises:
        ExcessiveErrorsError: If error rate exceeds threshold
    """
    results = []
    accumulator = ExceptionAccumulator("batch_processing")

    for record in records:
        accumulator.increment_processed()

        try:
            result = processor(record)
            results.append(result)
        except Exception as e:
            accumulator.add(
                e,
                context={"ticker": record.get("ticker", "unknown")},
            )
            # Optionally add a placeholder result
            # results.append({"ticker": record.get("ticker"), "error": str(e)})

    accumulator.log_summary()

    if accumulator.error_rate > error_threshold:
        raise ExcessiveErrorsError(
            f"Error rate {accumulator.error_rate:.1%} exceeds threshold {error_threshold:.1%}",
            accumulator.get_report(),
        )

    return results, accumulator


class ExcessiveErrorsError(Exception):
    """Raised when too many records fail processing."""

    def __init__(self, message: str, report: Dict[str, Any]):
        self.message = message
        self.report = report
        super().__init__(message)


if __name__ == "__main__":
    print("=" * 70)
    print("PATCH 004: Proper Exception Handling")
    print("=" * 70)
    print()
    print("This patch replaces silent exception swallowing with proper handling.")
    print()
    print("PROBLEM with original implementation:")
    print("  - `except Exception: pass` swallows all errors")
    print("  - No visibility into what's failing")
    print("  - Production issues are invisible until they cascade")
    print()
    print("SOLUTION (this patch):")
    print("  - ExceptionAccumulator: Track errors without stopping")
    print("  - safe_decimal_convert: Convert with logging")
    print("  - safe_divide: Handle division errors")
    print("  - @log_exception: Log before re-raising")
    print("  - @with_fallback: Use default with logging")
    print()
    print("Usage:")
    print("  from patches.patch_004_exception_handling import (")
    print("      ExceptionAccumulator,")
    print("      safe_decimal_convert,")
    print("      safe_divide,")
    print("  )")
    print()
    print("  # Track errors across batch")
    print("  accumulator = ExceptionAccumulator('my_process')")
    print("  for record in records:")
    print("      try:")
    print("          process(record)")
    print("      except Exception as e:")
    print("          accumulator.add(e, context={'ticker': record['ticker']})")
    print("  accumulator.log_summary()")
    print()
