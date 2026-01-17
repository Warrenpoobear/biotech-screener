"""
Robustness utilities for biotech screener pipeline.

Provides:
- Data staleness validation
- Cross-module consistency checks
- Retry logic with exponential backoff
- Memory guards for large datasets
- Structured logging with correlation IDs
- Graceful degradation helpers
"""
from __future__ import annotations

import functools
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, Union

from common.date_utils import to_date_object

__all__ = [
    # Data staleness
    "DataFreshnessConfig",
    "DataFreshnessResult",
    "validate_data_freshness",
    "validate_record_freshness",
    # Cross-module consistency
    "ConsistencyReport",
    "validate_ticker_coverage",
    "validate_module_handoff",
    # Retry logic
    "RetryConfig",
    "retry_with_backoff",
    "RetryExhaustedError",
    # Memory guards
    "MemoryGuardConfig",
    "chunk_universe",
    "estimate_memory_usage",
    # Structured logging
    "CorrelationContext",
    "get_correlation_id",
    "set_correlation_id",
    "with_correlation_id",
    # Graceful degradation
    "DegradationReport",
    "GracefulDegradationConfig",
    "compute_with_degradation",
]

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STALENESS VALIDATION
# ============================================================================

@dataclass
class DataFreshnessConfig:
    """Configuration for data freshness validation."""
    max_financial_age_days: int = 120  # ~1 quarter + buffer
    max_trial_age_days: int = 30       # Clinical trial data should be recent
    max_market_data_age_days: int = 5  # Market data should be very recent
    warn_threshold_pct: float = 0.75   # Warn at 75% of max age
    strict_mode: bool = False          # Raise exception on stale data


@dataclass
class DataFreshnessResult:
    """Result of data freshness validation."""
    is_fresh: bool
    is_warning: bool
    age_days: int
    max_age_days: int
    data_type: str
    record_id: Optional[str] = None
    message: str = ""

    @property
    def freshness_pct(self) -> float:
        """Percentage of max age consumed (0-100+)."""
        if self.max_age_days == 0:
            return 100.0
        return (self.age_days / self.max_age_days) * 100


def validate_data_freshness(
    record_date: Union[str, date],
    as_of_date: Union[str, date],
    data_type: str = "generic",
    config: Optional[DataFreshnessConfig] = None,
    record_id: Optional[str] = None,
) -> DataFreshnessResult:
    """
    Validate that data is fresh enough relative to as_of_date.

    Args:
        record_date: Date the data was recorded/fetched
        as_of_date: Reference date for the analysis
        data_type: Type of data ("financial", "trial", "market", "generic")
        config: Freshness configuration
        record_id: Optional identifier for the record (for logging)

    Returns:
        DataFreshnessResult with freshness status
    """
    config = config or DataFreshnessConfig()

    # Convert to date objects
    try:
        record_dt = to_date_object(record_date) if isinstance(record_date, str) else record_date
        as_of_dt = to_date_object(as_of_date) if isinstance(as_of_date, str) else as_of_date
    except (ValueError, TypeError):
        record_dt = None
        as_of_dt = None

    if record_dt is None or as_of_dt is None:
        return DataFreshnessResult(
            is_fresh=False,
            is_warning=True,
            age_days=-1,
            max_age_days=0,
            data_type=data_type,
            record_id=record_id,
            message="Invalid date format",
        )

    # Calculate age
    age_days = (as_of_dt - record_dt).days

    # Determine max age based on data type
    max_age_map = {
        "financial": config.max_financial_age_days,
        "trial": config.max_trial_age_days,
        "market": config.max_market_data_age_days,
        "generic": config.max_financial_age_days,
    }
    max_age_days = max_age_map.get(data_type, config.max_financial_age_days)

    # Check freshness
    is_fresh = age_days <= max_age_days
    warn_threshold = int(max_age_days * config.warn_threshold_pct)
    is_warning = age_days > warn_threshold

    message = ""
    if not is_fresh:
        message = f"{data_type} data is stale: {age_days} days old (max: {max_age_days})"
        if config.strict_mode:
            raise ValueError(message)
        logger.warning(message)
    elif is_warning:
        message = f"{data_type} data approaching staleness: {age_days}/{max_age_days} days"
        logger.info(message)

    return DataFreshnessResult(
        is_fresh=is_fresh,
        is_warning=is_warning,
        age_days=age_days,
        max_age_days=max_age_days,
        data_type=data_type,
        record_id=record_id,
        message=message,
    )


def validate_record_freshness(
    records: List[Dict[str, Any]],
    as_of_date: Union[str, date],
    date_field: str = "source_date",
    data_type: str = "generic",
    config: Optional[DataFreshnessConfig] = None,
) -> Dict[str, Any]:
    """
    Validate freshness of a batch of records.

    Returns:
        Dict with fresh_count, stale_count, warning_count, stale_records
    """
    config = config or DataFreshnessConfig()
    results = {
        "total": len(records),
        "fresh_count": 0,
        "stale_count": 0,
        "warning_count": 0,
        "stale_records": [],
        "avg_age_days": 0,
    }

    if not records:
        return results

    total_age = 0
    for record in records:
        record_date = record.get(date_field)
        if record_date is None:
            results["stale_count"] += 1
            continue

        ticker = record.get("ticker", record.get("nct_id", "unknown"))
        result = validate_data_freshness(
            record_date, as_of_date, data_type, config, record_id=ticker
        )

        total_age += max(0, result.age_days)

        if not result.is_fresh:
            results["stale_count"] += 1
            results["stale_records"].append(ticker)
        elif result.is_warning:
            results["warning_count"] += 1
            results["fresh_count"] += 1
        else:
            results["fresh_count"] += 1

    results["avg_age_days"] = total_age / len(records) if records else 0
    return results


# ============================================================================
# CROSS-MODULE CONSISTENCY CHECKS
# ============================================================================

@dataclass
class ConsistencyReport:
    """Report of cross-module consistency validation."""
    is_consistent: bool
    missing_in_m2: Set[str] = field(default_factory=set)
    missing_in_m3: Set[str] = field(default_factory=set)
    missing_in_m4: Set[str] = field(default_factory=set)
    orphan_in_m2: Set[str] = field(default_factory=set)
    orphan_in_m3: Set[str] = field(default_factory=set)
    orphan_in_m4: Set[str] = field(default_factory=set)
    coverage_m2_pct: float = 0.0
    coverage_m3_pct: float = 0.0
    coverage_m4_pct: float = 0.0
    warnings: List[str] = field(default_factory=list)


def validate_ticker_coverage(
    universe_tickers: Set[str],
    m2_tickers: Set[str],
    m3_tickers: Set[str],
    m4_tickers: Set[str],
    min_coverage_pct: float = 0.80,
) -> ConsistencyReport:
    """
    Validate that tickers are consistently covered across modules.

    Args:
        universe_tickers: Tickers from Module 1 universe
        m2_tickers: Tickers with Module 2 (financial) scores
        m3_tickers: Tickers with Module 3 (catalyst) data
        m4_tickers: Tickers with Module 4 (clinical) scores
        min_coverage_pct: Minimum acceptable coverage (0-1)

    Returns:
        ConsistencyReport with coverage analysis
    """
    if not universe_tickers:
        return ConsistencyReport(
            is_consistent=True,
            warnings=["Empty universe - no consistency check performed"],
        )

    # Calculate coverage
    n_universe = len(universe_tickers)
    coverage_m2 = len(m2_tickers & universe_tickers) / n_universe if n_universe else 0
    coverage_m3 = len(m3_tickers & universe_tickers) / n_universe if n_universe else 0
    coverage_m4 = len(m4_tickers & universe_tickers) / n_universe if n_universe else 0

    # Find missing and orphan tickers
    missing_m2 = universe_tickers - m2_tickers
    missing_m3 = universe_tickers - m3_tickers
    missing_m4 = universe_tickers - m4_tickers

    orphan_m2 = m2_tickers - universe_tickers
    orphan_m3 = m3_tickers - universe_tickers
    orphan_m4 = m4_tickers - universe_tickers

    # Build warnings
    warnings = []
    if coverage_m2 < min_coverage_pct:
        warnings.append(f"Module 2 coverage below threshold: {coverage_m2:.1%} < {min_coverage_pct:.1%}")
    if coverage_m3 < min_coverage_pct:
        warnings.append(f"Module 3 coverage below threshold: {coverage_m3:.1%} < {min_coverage_pct:.1%}")
    if coverage_m4 < min_coverage_pct:
        warnings.append(f"Module 4 coverage below threshold: {coverage_m4:.1%} < {min_coverage_pct:.1%}")
    if orphan_m2:
        warnings.append(f"Module 2 has {len(orphan_m2)} orphan tickers not in universe")
    if orphan_m3:
        warnings.append(f"Module 3 has {len(orphan_m3)} orphan tickers not in universe")
    if orphan_m4:
        warnings.append(f"Module 4 has {len(orphan_m4)} orphan tickers not in universe")

    is_consistent = (
        coverage_m2 >= min_coverage_pct and
        coverage_m3 >= min_coverage_pct and
        coverage_m4 >= min_coverage_pct and
        not orphan_m2 and not orphan_m3 and not orphan_m4
    )

    return ConsistencyReport(
        is_consistent=is_consistent,
        missing_in_m2=missing_m2,
        missing_in_m3=missing_m3,
        missing_in_m4=missing_m4,
        orphan_in_m2=orphan_m2,
        orphan_in_m3=orphan_m3,
        orphan_in_m4=orphan_m4,
        coverage_m2_pct=coverage_m2 * 100,
        coverage_m3_pct=coverage_m3 * 100,
        coverage_m4_pct=coverage_m4 * 100,
        warnings=warnings,
    )


def validate_module_handoff(
    source_module: str,
    target_module: str,
    source_output: Dict[str, Any],
    expected_fields: List[str],
) -> Dict[str, Any]:
    """
    Validate that module output has expected fields for handoff.

    Returns:
        Dict with is_valid, missing_fields, extra_fields
    """
    # Flatten nested fields for checking
    def get_all_keys(d: Dict, prefix: str = "") -> Set[str]:
        keys = set()
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            keys.add(full_key)
            if isinstance(v, dict):
                keys.update(get_all_keys(v, full_key))
        return keys

    actual_fields = get_all_keys(source_output)
    expected_set = set(expected_fields)

    missing = expected_set - actual_fields
    # Only report truly missing top-level expected fields
    missing_top_level = {f for f in missing if "." not in f and f not in source_output}

    return {
        "is_valid": len(missing_top_level) == 0,
        "source_module": source_module,
        "target_module": target_module,
        "missing_fields": list(missing_top_level),
        "field_count": len(actual_fields),
    }


# ============================================================================
# RETRY LOGIC WITH EXPONENTIAL BACKOFF
# ============================================================================

class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted."""
    pass


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    exponential_base: float = 2.0
    retryable_exceptions: tuple = (IOError, ConnectionError, TimeoutError)


T = TypeVar("T")


def retry_with_backoff(
    config: Optional[RetryConfig] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retrying functions with exponential backoff.

    Usage:
        @retry_with_backoff(RetryConfig(max_attempts=3))
        def load_data(path: str) -> Dict:
            ...
    """
    config = config or RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None

            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e
                    if attempt < config.max_attempts - 1:
                        delay = min(
                            config.base_delay_seconds * (config.exponential_base ** attempt),
                            config.max_delay_seconds,
                        )
                        logger.warning(
                            f"Attempt {attempt + 1}/{config.max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {config.max_attempts} attempts exhausted for {func.__name__}: {e}"
                        )

            raise RetryExhaustedError(
                f"Failed after {config.max_attempts} attempts: {last_exception}"
            ) from last_exception

        return wrapper
    return decorator


# ============================================================================
# MEMORY GUARDS
# ============================================================================

@dataclass
class MemoryGuardConfig:
    """Configuration for memory guards."""
    max_universe_size: int = 1000
    chunk_size: int = 200
    warn_threshold: int = 500
    estimated_bytes_per_ticker: int = 50000  # ~50KB per ticker with full data


def chunk_universe(
    tickers: List[str],
    config: Optional[MemoryGuardConfig] = None,
) -> List[List[str]]:
    """
    Split large universe into manageable chunks.

    Args:
        tickers: List of tickers to process
        config: Memory guard configuration

    Returns:
        List of ticker chunks
    """
    config = config or MemoryGuardConfig()

    if len(tickers) <= config.chunk_size:
        return [tickers]

    if len(tickers) > config.warn_threshold:
        logger.warning(
            f"Large universe detected: {len(tickers)} tickers. "
            f"Processing in {len(tickers) // config.chunk_size + 1} chunks."
        )

    chunks = []
    for i in range(0, len(tickers), config.chunk_size):
        chunks.append(tickers[i : i + config.chunk_size])

    return chunks


def estimate_memory_usage(
    ticker_count: int,
    config: Optional[MemoryGuardConfig] = None,
) -> Dict[str, Any]:
    """
    Estimate memory usage for processing a universe.

    Returns:
        Dict with estimated_mb, recommended_chunks, warning
    """
    config = config or MemoryGuardConfig()

    estimated_bytes = ticker_count * config.estimated_bytes_per_ticker
    estimated_mb = estimated_bytes / (1024 * 1024)

    recommended_chunks = max(1, ticker_count // config.chunk_size)

    warning = None
    if ticker_count > config.max_universe_size:
        warning = f"Universe size {ticker_count} exceeds recommended max {config.max_universe_size}"
    elif ticker_count > config.warn_threshold:
        warning = f"Large universe: consider chunking for better performance"

    return {
        "ticker_count": ticker_count,
        "estimated_mb": round(estimated_mb, 2),
        "recommended_chunks": recommended_chunks,
        "warning": warning,
    }


# ============================================================================
# STRUCTURED LOGGING WITH CORRELATION IDS
# ============================================================================

# Thread-local storage for correlation ID
import threading
_correlation_context = threading.local()


@dataclass
class CorrelationContext:
    """Context for correlated logging across operations."""
    correlation_id: str
    run_id: Optional[str] = None
    as_of_date: Optional[str] = None
    module: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def get_correlation_id() -> str:
    """Get current correlation ID or generate a new one."""
    if not hasattr(_correlation_context, "id"):
        _correlation_context.id = str(uuid.uuid4())[:8]
    return _correlation_context.id


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID for the current thread."""
    _correlation_context.id = correlation_id


def with_correlation_id(correlation_id: Optional[str] = None) -> Callable:
    """
    Decorator to set correlation ID for a function execution.

    Usage:
        @with_correlation_id("abc123")
        def process_module():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            cid = correlation_id or str(uuid.uuid4())[:8]
            old_id = getattr(_correlation_context, "id", None)
            set_correlation_id(cid)
            try:
                return func(*args, **kwargs)
            finally:
                if old_id:
                    set_correlation_id(old_id)
        return wrapper
    return decorator


class CorrelatedLogger:
    """Logger wrapper that includes correlation ID in all messages."""

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)

    def _format_message(self, msg: str) -> str:
        cid = get_correlation_id()
        return f"[{cid}] {msg}"

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.debug(self._format_message(msg), *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.info(self._format_message(msg), *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.warning(self._format_message(msg), *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.error(self._format_message(msg), *args, **kwargs)


# ============================================================================
# GRACEFUL DEGRADATION
# ============================================================================

@dataclass
class GracefulDegradationConfig:
    """Configuration for graceful degradation."""
    allow_missing_financial: bool = True
    allow_missing_catalyst: bool = True
    allow_missing_clinical: bool = False  # Clinical is critical
    min_components_required: int = 2
    apply_missing_penalty: bool = True
    missing_penalty_pct: Decimal = Decimal("0.15")


@dataclass
class DegradationReport:
    """Report of degradation applied during scoring."""
    was_degraded: bool
    missing_components: List[str] = field(default_factory=list)
    penalties_applied: Dict[str, str] = field(default_factory=dict)
    confidence_adjustment: Decimal = Decimal("1.0")
    warnings: List[str] = field(default_factory=list)


def compute_with_degradation(
    components: Dict[str, Optional[Any]],
    config: Optional[GracefulDegradationConfig] = None,
) -> DegradationReport:
    """
    Determine if graceful degradation is needed and compute adjustments.

    Args:
        components: Dict mapping component name to value (None if missing)
        config: Degradation configuration

    Returns:
        DegradationReport with degradation details
    """
    config = config or GracefulDegradationConfig()

    missing = [k for k, v in components.items() if v is None]
    available_count = len(components) - len(missing)

    report = DegradationReport(
        was_degraded=len(missing) > 0,
        missing_components=missing,
    )

    if not missing:
        return report

    # Check if degradation is allowed
    warnings = []
    for component in missing:
        if component == "financial" and not config.allow_missing_financial:
            warnings.append(f"Missing critical component: {component}")
        elif component == "catalyst" and not config.allow_missing_catalyst:
            warnings.append(f"Missing critical component: {component}")
        elif component == "clinical" and not config.allow_missing_clinical:
            warnings.append(f"Missing critical component: {component}")

    if available_count < config.min_components_required:
        warnings.append(
            f"Insufficient components: {available_count} < {config.min_components_required} required"
        )

    report.warnings = warnings

    # Apply penalties
    if config.apply_missing_penalty and missing:
        penalty_per_missing = config.missing_penalty_pct
        total_penalty = penalty_per_missing * len(missing)
        report.confidence_adjustment = Decimal("1.0") - min(total_penalty, Decimal("0.50"))
        report.penalties_applied = {
            comp: str(penalty_per_missing) for comp in missing
        }

    return report
