"""
Extended robustness utilities for biotech screener pipeline.

Provides advanced robustness features beyond the base module:
- Timeout protection for long-running operations
- Stateful circuit breaker (OPEN/HALF_OPEN/CLOSED)
- Operation throttling and rate limiting
- Data integrity verification with checksums
- Idempotency tracking for recoverable operations
- Module health checks and self-diagnostics
- Watchdog for detecting stuck operations
"""
from __future__ import annotations

import functools
import hashlib
import json
import logging
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Set, TypeVar, Union

__all__ = [
    # Timeout protection
    "TimeoutError",
    "TimeoutConfig",
    "with_timeout",
    "run_with_timeout",
    # Stateful circuit breaker
    "CircuitState",
    "StatefulCircuitBreaker",
    "StatefulCircuitBreakerConfig",
    # Throttling
    "ThrottleConfig",
    "Throttler",
    "with_throttle",
    # Data integrity
    "IntegrityConfig",
    "compute_data_hash",
    "verify_data_integrity",
    "IntegrityResult",
    # Idempotency
    "IdempotencyKey",
    "IdempotencyStore",
    "idempotent",
    # Health checks
    "HealthStatus",
    "HealthCheckResult",
    "ModuleHealthChecker",
    # Watchdog
    "WatchdogConfig",
    "OperationWatchdog",
    "WatchdogTimeoutError",
]

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================================
# TIMEOUT PROTECTION
# ============================================================================

class TimeoutError(Exception):
    """Raised when an operation exceeds its timeout."""

    def __init__(self, message: str, operation: str = "", elapsed_seconds: float = 0):
        super().__init__(message)
        self.operation = operation
        self.elapsed_seconds = elapsed_seconds


@dataclass
class TimeoutConfig:
    """Configuration for timeout protection."""
    timeout_seconds: float = 30.0
    raise_on_timeout: bool = True
    default_value: Any = None
    cleanup_func: Optional[Callable] = None


def _timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Operation timed out")


def with_timeout(config: Optional[TimeoutConfig] = None) -> Callable:
    """
    Decorator to add timeout protection to a function.

    Uses threading-based timeout (works on all platforms).

    Usage:
        @with_timeout(TimeoutConfig(timeout_seconds=10))
        def slow_operation():
            ...
    """
    config = config or TimeoutConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            result = [config.default_value]
            exception = [None]
            completed = threading.Event()

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
                finally:
                    completed.set()

            thread = threading.Thread(target=target, daemon=True)
            start_time = time.monotonic()
            thread.start()

            # Wait for completion or timeout
            completed.wait(timeout=config.timeout_seconds)
            elapsed = time.monotonic() - start_time

            if not completed.is_set():
                # Timeout occurred
                if config.cleanup_func:
                    try:
                        config.cleanup_func()
                    except Exception as cleanup_err:
                        logger.warning(f"Cleanup failed after timeout: {cleanup_err}")

                if config.raise_on_timeout:
                    raise TimeoutError(
                        f"Operation '{func.__name__}' timed out after {config.timeout_seconds}s",
                        operation=func.__name__,
                        elapsed_seconds=elapsed,
                    )

                logger.warning(
                    f"Operation '{func.__name__}' timed out after {config.timeout_seconds}s, "
                    f"returning default value"
                )
                return config.default_value

            # Check if exception occurred in thread
            if exception[0] is not None:
                raise exception[0]

            return result[0]

        return wrapper
    return decorator


def run_with_timeout(
    func: Callable[..., T],
    timeout_seconds: float,
    *args: Any,
    **kwargs: Any,
) -> T:
    """
    Run a function with a timeout.

    Args:
        func: Function to run
        timeout_seconds: Maximum execution time
        *args, **kwargs: Arguments to pass to function

    Returns:
        Function result

    Raises:
        TimeoutError: If operation exceeds timeout
    """
    config = TimeoutConfig(timeout_seconds=timeout_seconds)
    wrapped = with_timeout(config)(func)
    return wrapped(*args, **kwargs)


# ============================================================================
# STATEFUL CIRCUIT BREAKER
# ============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation, tracking failures
    OPEN = "open"          # Failing fast, not executing
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class StatefulCircuitBreakerConfig:
    """Configuration for stateful circuit breaker."""
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 3          # Successes in half-open to close
    timeout_seconds: float = 60.0       # Time to wait before half-open
    half_open_max_calls: int = 3        # Max calls allowed in half-open

    # Failure tracking window
    window_size_seconds: float = 120.0  # Time window for counting failures


class StatefulCircuitBreaker:
    """
    Stateful circuit breaker with OPEN/HALF_OPEN/CLOSED pattern.

    State transitions:
    - CLOSED → OPEN: When failures exceed threshold
    - OPEN → HALF_OPEN: After timeout expires
    - HALF_OPEN → CLOSED: After success threshold met
    - HALF_OPEN → OPEN: On any failure

    Usage:
        breaker = StatefulCircuitBreaker("external_api")

        @breaker.protect
        def call_external_api():
            ...

        # Or manually
        if breaker.allow_request():
            try:
                result = call_external_api()
                breaker.record_success()
            except Exception:
                breaker.record_failure()
    """

    def __init__(
        self,
        name: str,
        config: Optional[StatefulCircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or StatefulCircuitBreakerConfig()

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._opened_at: Optional[float] = None
        self._half_open_calls = 0

        # Sliding window for failure tracking
        self._failure_times: deque = deque()

        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        with self._lock:
            self._update_state()
            return self._state

    def _update_state(self) -> None:
        """Update state based on time and thresholds."""
        now = time.monotonic()

        # Clean old failures from window
        cutoff = now - self.config.window_size_seconds
        while self._failure_times and self._failure_times[0] < cutoff:
            self._failure_times.popleft()

        # Check for state transitions
        if self._state == CircuitState.OPEN:
            # Check if timeout expired
            if self._opened_at and (now - self._opened_at) >= self.config.timeout_seconds:
                logger.info(f"Circuit breaker '{self.name}': OPEN → HALF_OPEN")
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                self._success_count = 0

    def allow_request(self) -> bool:
        """
        Check if a request should be allowed through.

        Returns:
            True if request should proceed, False if circuit is open
        """
        with self._lock:
            self._update_state()

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                return False

            # HALF_OPEN: Allow limited requests
            if self._half_open_calls < self.config.half_open_max_calls:
                self._half_open_calls += 1
                return True

            return False

    def record_success(self) -> None:
        """Record a successful operation."""
        with self._lock:
            self._update_state()  # Check for OPEN → HALF_OPEN transition
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    logger.info(f"Circuit breaker '{self.name}': HALF_OPEN → CLOSED")
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._failure_times.clear()

    def record_failure(self, error: Optional[Exception] = None) -> None:
        """Record a failed operation."""
        with self._lock:
            self._update_state()  # Check for OPEN → HALF_OPEN transition
            now = time.monotonic()
            self._failure_times.append(now)
            self._failure_count = len(self._failure_times)
            self._last_failure_time = now

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open reopens the circuit
                logger.warning(
                    f"Circuit breaker '{self.name}': HALF_OPEN → OPEN (failure: {error})"
                )
                self._state = CircuitState.OPEN
                self._opened_at = now

            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    logger.warning(
                        f"Circuit breaker '{self.name}': CLOSED → OPEN "
                        f"({self._failure_count} failures)"
                    )
                    self._state = CircuitState.OPEN
                    self._opened_at = now

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED state."""
        with self._lock:
            logger.info(f"Circuit breaker '{self.name}': Manual reset to CLOSED")
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._failure_times.clear()
            self._opened_at = None

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            self._update_state()
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure_time": self._last_failure_time,
                "opened_at": self._opened_at,
            }

    def protect(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator to protect a function with this circuit breaker.

        Usage:
            @breaker.protect
            def my_function():
                ...
        """
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            if not self.allow_request():
                raise RuntimeError(
                    f"Circuit breaker '{self.name}' is OPEN - request rejected"
                )

            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure(e)
                raise

        return wrapper


# ============================================================================
# OPERATION THROTTLING
# ============================================================================

@dataclass
class ThrottleConfig:
    """Configuration for operation throttling."""
    max_calls: int = 10            # Maximum calls allowed
    period_seconds: float = 1.0    # Time period for the limit
    burst_allowance: int = 0       # Extra calls allowed in burst
    wait_on_limit: bool = True     # If True, wait; if False, raise error


class Throttler:
    """
    Rate limiter using token bucket algorithm.

    Usage:
        throttler = Throttler("api_calls", ThrottleConfig(max_calls=10, period_seconds=1.0))

        @throttler.throttle
        def call_api():
            ...

        # Or manually
        throttler.acquire()  # Blocks until allowed
        call_api()
    """

    def __init__(self, name: str, config: Optional[ThrottleConfig] = None):
        self.name = name
        self.config = config or ThrottleConfig()

        self._tokens = float(self.config.max_calls + self.config.burst_allowance)
        self._max_tokens = float(self.config.max_calls + self.config.burst_allowance)
        self._last_update = time.monotonic()
        self._lock = threading.Lock()

    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._last_update = now

        # Add tokens based on elapsed time
        tokens_to_add = elapsed * (self.config.max_calls / self.config.period_seconds)
        self._tokens = min(self._max_tokens, self._tokens + tokens_to_add)

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire permission to proceed.

        Args:
            timeout: Maximum time to wait (None = use config.wait_on_limit)

        Returns:
            True if acquired, False if timeout/rejected

        Raises:
            RuntimeError: If wait_on_limit is False and no tokens available
        """
        deadline = time.monotonic() + (timeout or 60.0) if self.config.wait_on_limit else None

        while True:
            with self._lock:
                self._refill_tokens()

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True

            if not self.config.wait_on_limit:
                raise RuntimeError(
                    f"Throttler '{self.name}' rate limit exceeded "
                    f"({self.config.max_calls} calls per {self.config.period_seconds}s)"
                )

            if deadline and time.monotonic() >= deadline:
                return False

            # Wait a bit before retrying
            time.sleep(0.01)

    def throttle(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to throttle function calls."""
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            self.acquire()
            return func(*args, **kwargs)
        return wrapper

    def get_stats(self) -> Dict[str, Any]:
        """Get throttler statistics."""
        with self._lock:
            self._refill_tokens()
            return {
                "name": self.name,
                "available_tokens": self._tokens,
                "max_tokens": self._max_tokens,
                "max_calls_per_period": self.config.max_calls,
                "period_seconds": self.config.period_seconds,
            }


def with_throttle(
    max_calls: int = 10,
    period_seconds: float = 1.0,
) -> Callable:
    """
    Simple throttle decorator.

    Usage:
        @with_throttle(max_calls=5, period_seconds=1.0)
        def rate_limited_function():
            ...
    """
    throttler = Throttler("inline", ThrottleConfig(
        max_calls=max_calls,
        period_seconds=period_seconds,
    ))
    return throttler.throttle


# ============================================================================
# DATA INTEGRITY VERIFICATION
# ============================================================================

@dataclass
class IntegrityConfig:
    """Configuration for data integrity verification."""
    hash_algorithm: str = "sha256"
    include_metadata: bool = True
    strict_mode: bool = False  # Raise on mismatch


@dataclass
class IntegrityResult:
    """Result of integrity verification."""
    is_valid: bool
    expected_hash: Optional[str]
    actual_hash: str
    message: str
    record_count: int = 0


def _stable_json_dumps(obj: Any) -> str:
    """Deterministic JSON serialization for hashing."""
    def default_serializer(o):
        if isinstance(o, date):
            return o.isoformat()
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, Decimal):
            return str(o)
        if isinstance(o, set):
            return sorted(list(o))
        if hasattr(o, "__dict__"):
            return o.__dict__
        raise TypeError(f"Cannot serialize {type(o)}")

    return json.dumps(obj, sort_keys=True, default=default_serializer, separators=(",", ":"))


def compute_data_hash(
    data: Any,
    config: Optional[IntegrityConfig] = None,
) -> str:
    """
    Compute a deterministic hash of data.

    Args:
        data: Data to hash (must be JSON-serializable)
        config: Integrity configuration

    Returns:
        Hash string in format "algorithm:hexdigest"
    """
    config = config or IntegrityConfig()

    serialized = _stable_json_dumps(data)

    if config.hash_algorithm == "sha256":
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    elif config.hash_algorithm == "sha512":
        digest = hashlib.sha512(serialized.encode("utf-8")).hexdigest()
    elif config.hash_algorithm == "md5":
        digest = hashlib.md5(serialized.encode("utf-8")).hexdigest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {config.hash_algorithm}")

    return f"{config.hash_algorithm}:{digest}"


def verify_data_integrity(
    data: Any,
    expected_hash: Optional[str],
    config: Optional[IntegrityConfig] = None,
) -> IntegrityResult:
    """
    Verify data integrity against an expected hash.

    Args:
        data: Data to verify
        expected_hash: Expected hash (None to just compute hash)
        config: Integrity configuration

    Returns:
        IntegrityResult with verification status

    Raises:
        ValueError: If strict_mode is True and hash doesn't match
    """
    config = config or IntegrityConfig()

    actual_hash = compute_data_hash(data, config)

    # Count records if data is a list
    record_count = len(data) if isinstance(data, (list, dict)) else 1

    if expected_hash is None:
        return IntegrityResult(
            is_valid=True,
            expected_hash=None,
            actual_hash=actual_hash,
            message="Hash computed (no expected hash provided)",
            record_count=record_count,
        )

    is_valid = actual_hash == expected_hash

    if not is_valid:
        message = f"Hash mismatch: expected {expected_hash}, got {actual_hash}"
        logger.warning(message)

        if config.strict_mode:
            raise ValueError(message)
    else:
        message = "Data integrity verified"

    return IntegrityResult(
        is_valid=is_valid,
        expected_hash=expected_hash,
        actual_hash=actual_hash,
        message=message,
        record_count=record_count,
    )


# ============================================================================
# IDEMPOTENCY TRACKING
# ============================================================================

@dataclass
class IdempotencyKey:
    """Key for idempotent operation tracking."""
    operation: str
    key: str
    timestamp: float = field(default_factory=time.time)
    result_hash: Optional[str] = None
    status: str = "pending"  # pending, completed, failed


class IdempotencyStore:
    """
    In-memory store for tracking idempotent operations.

    Ensures operations with the same key are not repeated.

    Usage:
        store = IdempotencyStore()

        key = store.create_key("process_ticker", "AAPL_2025-01-15")
        if store.is_completed(key):
            return store.get_result(key)

        result = process_ticker("AAPL", "2025-01-15")
        store.mark_completed(key, result)
    """

    def __init__(self, max_entries: int = 10000, ttl_seconds: float = 3600):
        self._entries: Dict[str, IdempotencyKey] = {}
        self._results: Dict[str, Any] = {}
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()

    def _cleanup_old_entries(self) -> None:
        """Remove expired entries."""
        now = time.time()
        expired = [
            k for k, v in self._entries.items()
            if (now - v.timestamp) > self._ttl_seconds
        ]
        for k in expired:
            del self._entries[k]
            self._results.pop(k, None)

    def create_key(self, operation: str, key: str) -> str:
        """Create a unique idempotency key."""
        return f"{operation}:{key}"

    def register(self, operation: str, key: str) -> Tuple[str, bool]:
        """
        Register an operation for idempotency tracking.

        Returns:
            Tuple of (idempotency_key, already_exists)
        """
        idem_key = self.create_key(operation, key)

        with self._lock:
            self._cleanup_old_entries()

            if idem_key in self._entries:
                return idem_key, True

            # Enforce max entries
            if len(self._entries) >= self._max_entries:
                # Remove oldest entry
                oldest = min(self._entries.items(), key=lambda x: x[1].timestamp)
                del self._entries[oldest[0]]
                self._results.pop(oldest[0], None)

            self._entries[idem_key] = IdempotencyKey(
                operation=operation,
                key=key,
            )
            return idem_key, False

    def is_completed(self, idem_key: str) -> bool:
        """Check if operation is already completed."""
        with self._lock:
            entry = self._entries.get(idem_key)
            return entry is not None and entry.status == "completed"

    def get_result(self, idem_key: str) -> Optional[Any]:
        """Get the cached result of a completed operation."""
        with self._lock:
            return self._results.get(idem_key)

    def mark_completed(self, idem_key: str, result: Any) -> None:
        """Mark an operation as completed with its result."""
        with self._lock:
            if idem_key in self._entries:
                self._entries[idem_key].status = "completed"
                self._entries[idem_key].result_hash = compute_data_hash(result)
                self._results[idem_key] = result

    def mark_failed(self, idem_key: str, error: str) -> None:
        """Mark an operation as failed."""
        with self._lock:
            if idem_key in self._entries:
                self._entries[idem_key].status = "failed"

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        with self._lock:
            return {
                "total_entries": len(self._entries),
                "completed": sum(1 for e in self._entries.values() if e.status == "completed"),
                "pending": sum(1 for e in self._entries.values() if e.status == "pending"),
                "failed": sum(1 for e in self._entries.values() if e.status == "failed"),
            }


def idempotent(store: IdempotencyStore, key_func: Callable[..., str]) -> Callable:
    """
    Decorator for idempotent function execution.

    Usage:
        store = IdempotencyStore()

        @idempotent(store, key_func=lambda ticker, date: f"{ticker}_{date}")
        def process_ticker(ticker: str, date: str):
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            key = key_func(*args, **kwargs)
            idem_key, exists = store.register(func.__name__, key)

            if exists and store.is_completed(idem_key):
                logger.debug(f"Idempotent hit for {func.__name__}:{key}")
                return store.get_result(idem_key)

            try:
                result = func(*args, **kwargs)
                store.mark_completed(idem_key, result)
                return result
            except Exception as e:
                store.mark_failed(idem_key, str(e))
                raise

        return wrapper
    return decorator


# ============================================================================
# MODULE HEALTH CHECKS
# ============================================================================

class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class ModuleHealthChecker:
    """
    Health checker for pipeline modules.

    Usage:
        checker = ModuleHealthChecker("module_2_financial")

        # Register checks
        checker.add_check("data_freshness", check_data_freshness)
        checker.add_check("connection", check_db_connection)

        # Run all checks
        results = checker.run_all_checks()
        overall = checker.get_overall_status()
    """

    def __init__(self, module_name: str):
        self.module_name = module_name
        self._checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        self._last_results: Dict[str, HealthCheckResult] = {}
        self._lock = threading.Lock()

    def add_check(
        self,
        name: str,
        check_func: Callable[[], HealthCheckResult],
    ) -> None:
        """Add a health check."""
        self._checks[name] = check_func

    def run_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check."""
        if name not in self._checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Check '{name}' not found",
            )

        start = time.monotonic()
        try:
            result = self._checks[name]()
            result.latency_ms = (time.monotonic() - start) * 1000

            with self._lock:
                self._last_results[name] = result

            return result
        except Exception as e:
            result = HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed with exception: {e}",
                latency_ms=(time.monotonic() - start) * 1000,
            )

            with self._lock:
                self._last_results[name] = result

            return result

    def run_all_checks(self) -> List[HealthCheckResult]:
        """Run all registered health checks."""
        return [self.run_check(name) for name in self._checks]

    def get_overall_status(self) -> HealthStatus:
        """Get overall health status based on all checks."""
        with self._lock:
            if not self._last_results:
                return HealthStatus.UNKNOWN

            statuses = [r.status for r in self._last_results.values()]

            if any(s == HealthStatus.UNHEALTHY for s in statuses):
                return HealthStatus.UNHEALTHY
            if any(s == HealthStatus.DEGRADED for s in statuses):
                return HealthStatus.DEGRADED
            if all(s == HealthStatus.HEALTHY for s in statuses):
                return HealthStatus.HEALTHY

            return HealthStatus.UNKNOWN

    def get_summary(self) -> Dict[str, Any]:
        """Get health check summary."""
        with self._lock:
            return {
                "module": self.module_name,
                "overall_status": self.get_overall_status().value,
                "checks": {
                    name: {
                        "status": result.status.value,
                        "message": result.message,
                        "latency_ms": result.latency_ms,
                    }
                    for name, result in self._last_results.items()
                },
                "timestamp": time.time(),
            }


# ============================================================================
# WATCHDOG FOR STUCK OPERATIONS
# ============================================================================

class WatchdogTimeoutError(Exception):
    """Raised when watchdog detects a stuck operation."""
    pass


@dataclass
class WatchdogConfig:
    """Configuration for operation watchdog."""
    timeout_seconds: float = 300.0  # 5 minutes default
    check_interval_seconds: float = 10.0
    on_timeout: str = "warn"  # "warn", "raise", or "callback"
    callback: Optional[Callable[[str], None]] = None


class OperationWatchdog:
    """
    Watchdog for detecting stuck/hung operations.

    Usage:
        watchdog = OperationWatchdog()

        # Start monitoring
        watchdog.start_operation("process_universe")

        # Heartbeat periodically to signal progress
        for ticker in tickers:
            process(ticker)
            watchdog.heartbeat("process_universe")

        # Complete when done
        watchdog.complete_operation("process_universe")
    """

    def __init__(self, config: Optional[WatchdogConfig] = None):
        self.config = config or WatchdogConfig()
        self._operations: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start_operation(self, operation_id: str, timeout_override: Optional[float] = None) -> None:
        """Start monitoring an operation."""
        with self._lock:
            self._operations[operation_id] = {
                "started_at": time.monotonic(),
                "last_heartbeat": time.monotonic(),
                "timeout": timeout_override or self.config.timeout_seconds,
                "heartbeat_count": 0,
            }
            logger.debug(f"Watchdog: Started monitoring '{operation_id}'")

    def heartbeat(self, operation_id: str) -> None:
        """Send heartbeat for an operation."""
        with self._lock:
            if operation_id in self._operations:
                self._operations[operation_id]["last_heartbeat"] = time.monotonic()
                self._operations[operation_id]["heartbeat_count"] += 1

    def complete_operation(self, operation_id: str) -> None:
        """Mark an operation as completed."""
        with self._lock:
            if operation_id in self._operations:
                elapsed = time.monotonic() - self._operations[operation_id]["started_at"]
                heartbeats = self._operations[operation_id]["heartbeat_count"]
                del self._operations[operation_id]
                logger.debug(
                    f"Watchdog: Operation '{operation_id}' completed "
                    f"(elapsed: {elapsed:.1f}s, heartbeats: {heartbeats})"
                )

    def check_operations(self) -> List[str]:
        """
        Check for stuck operations.

        Returns:
            List of operation IDs that have timed out
        """
        now = time.monotonic()
        timed_out = []

        with self._lock:
            for op_id, info in list(self._operations.items()):
                elapsed = now - info["last_heartbeat"]
                if elapsed > info["timeout"]:
                    timed_out.append(op_id)

                    message = (
                        f"Watchdog: Operation '{op_id}' appears stuck "
                        f"(no heartbeat for {elapsed:.1f}s, timeout: {info['timeout']}s)"
                    )

                    if self.config.on_timeout == "warn":
                        logger.warning(message)
                    elif self.config.on_timeout == "raise":
                        raise WatchdogTimeoutError(message)
                    elif self.config.on_timeout == "callback" and self.config.callback:
                        self.config.callback(op_id)

        return timed_out

    def start_monitoring(self) -> None:
        """Start background monitoring thread."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return

        self._stop_event.clear()

        def monitor_loop():
            while not self._stop_event.is_set():
                self.check_operations()
                self._stop_event.wait(self.config.check_interval_seconds)

        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Watchdog: Background monitoring started")

    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
            logger.info("Watchdog: Background monitoring stopped")

    def get_stats(self) -> Dict[str, Any]:
        """Get watchdog statistics."""
        with self._lock:
            now = time.monotonic()
            return {
                "active_operations": len(self._operations),
                "operations": {
                    op_id: {
                        "elapsed_seconds": now - info["started_at"],
                        "since_heartbeat_seconds": now - info["last_heartbeat"],
                        "heartbeat_count": info["heartbeat_count"],
                    }
                    for op_id, info in self._operations.items()
                },
                "monitoring_active": self._monitor_thread is not None and self._monitor_thread.is_alive(),
            }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_robust_pipeline_context(
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a context with all robustness utilities initialized.

    Returns:
        Dict with circuit_breaker, throttler, idempotency_store, watchdog
    """
    import uuid

    cid = correlation_id or str(uuid.uuid4())[:8]

    return {
        "correlation_id": cid,
        "circuit_breaker": StatefulCircuitBreaker(f"pipeline_{cid}"),
        "throttler": Throttler(f"pipeline_{cid}"),
        "idempotency_store": IdempotencyStore(),
        "watchdog": OperationWatchdog(),
        "health_checker": ModuleHealthChecker(f"pipeline_{cid}"),
    }
