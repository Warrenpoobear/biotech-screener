"""
Tests for extended robustness utilities.

Tests cover:
- Timeout protection
- Stateful circuit breaker
- Operation throttling
- Data integrity verification
- Idempotency tracking
- Module health checks
- Operation watchdog
"""
import threading
import time
from datetime import date
from decimal import Decimal
from typing import Tuple

import pytest

from common.robustness_extended import (
    # Timeout
    TimeoutConfig,
    TimeoutError,
    with_timeout,
    run_with_timeout,
    # Circuit breaker
    CircuitState,
    StatefulCircuitBreaker,
    StatefulCircuitBreakerConfig,
    # Throttling
    ThrottleConfig,
    Throttler,
    with_throttle,
    # Integrity
    IntegrityConfig,
    compute_data_hash,
    verify_data_integrity,
    IntegrityResult,
    # Idempotency
    IdempotencyStore,
    idempotent,
    # Health checks
    HealthStatus,
    HealthCheckResult,
    ModuleHealthChecker,
    # Watchdog
    WatchdogConfig,
    OperationWatchdog,
    WatchdogTimeoutError,
    # Utilities
    create_robust_pipeline_context,
)


# ============================================================================
# TIMEOUT PROTECTION TESTS
# ============================================================================

class TestTimeoutProtection:
    """Tests for timeout protection utilities."""

    def test_timeout_success(self):
        """Function completes within timeout."""
        @with_timeout(TimeoutConfig(timeout_seconds=1.0))
        def quick_operation():
            return "success"

        result = quick_operation()
        assert result == "success"

    def test_timeout_exceeded_raises(self):
        """Function exceeding timeout raises TimeoutError."""
        @with_timeout(TimeoutConfig(timeout_seconds=0.1))
        def slow_operation():
            time.sleep(1.0)
            return "never reached"

        with pytest.raises(TimeoutError) as exc_info:
            slow_operation()

        assert "timed out" in str(exc_info.value)
        assert exc_info.value.operation == "slow_operation"

    def test_timeout_returns_default(self):
        """Timeout returns default value when raise_on_timeout is False."""
        @with_timeout(TimeoutConfig(
            timeout_seconds=0.1,
            raise_on_timeout=False,
            default_value="fallback",
        ))
        def slow_operation():
            time.sleep(1.0)
            return "never reached"

        result = slow_operation()
        assert result == "fallback"

    def test_run_with_timeout_helper(self):
        """Helper function works correctly."""
        def add(a, b):
            return a + b

        result = run_with_timeout(add, 1.0, 2, 3)
        assert result == 5

    def test_timeout_propagates_exception(self):
        """Exceptions from function are propagated."""
        @with_timeout(TimeoutConfig(timeout_seconds=1.0))
        def error_operation():
            raise ValueError("intentional error")

        with pytest.raises(ValueError, match="intentional error"):
            error_operation()


# ============================================================================
# STATEFUL CIRCUIT BREAKER TESTS
# ============================================================================

class TestStatefulCircuitBreaker:
    """Tests for stateful circuit breaker."""

    def test_initial_state_closed(self):
        """Circuit starts in CLOSED state."""
        breaker = StatefulCircuitBreaker("test")
        assert breaker.state == CircuitState.CLOSED

    def test_opens_after_failures(self):
        """Circuit opens after failure threshold."""
        config = StatefulCircuitBreakerConfig(failure_threshold=3)
        breaker = StatefulCircuitBreaker("test", config)

        # Record failures
        for _ in range(3):
            breaker.record_failure()

        assert breaker.state == CircuitState.OPEN

    def test_open_rejects_requests(self):
        """Open circuit rejects requests."""
        config = StatefulCircuitBreakerConfig(failure_threshold=1)
        breaker = StatefulCircuitBreaker("test", config)

        breaker.record_failure()  # Opens circuit

        assert not breaker.allow_request()

    def test_half_open_after_timeout(self):
        """Circuit transitions to HALF_OPEN after timeout."""
        config = StatefulCircuitBreakerConfig(
            failure_threshold=1,
            timeout_seconds=0.1,
        )
        breaker = StatefulCircuitBreaker("test", config)

        breaker.record_failure()  # Opens circuit
        assert breaker.state == CircuitState.OPEN

        time.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN

    def test_half_open_closes_on_success(self):
        """HALF_OPEN closes after success threshold."""
        config = StatefulCircuitBreakerConfig(
            failure_threshold=1,
            success_threshold=2,
            timeout_seconds=0.1,
        )
        breaker = StatefulCircuitBreaker("test", config)

        breaker.record_failure()  # CLOSED -> OPEN
        time.sleep(0.15)  # OPEN -> HALF_OPEN

        breaker.record_success()
        assert breaker.state == CircuitState.HALF_OPEN

        breaker.record_success()  # HALF_OPEN -> CLOSED
        assert breaker.state == CircuitState.CLOSED

    def test_half_open_reopens_on_failure(self):
        """HALF_OPEN reopens on any failure."""
        config = StatefulCircuitBreakerConfig(
            failure_threshold=1,
            timeout_seconds=0.1,
        )
        breaker = StatefulCircuitBreaker("test", config)

        breaker.record_failure()  # CLOSED -> OPEN
        time.sleep(0.15)  # OPEN -> HALF_OPEN

        breaker.record_failure()  # HALF_OPEN -> OPEN
        assert breaker.state == CircuitState.OPEN

    def test_protect_decorator(self):
        """Protect decorator works correctly."""
        config = StatefulCircuitBreakerConfig(failure_threshold=2)
        breaker = StatefulCircuitBreaker("test", config)

        call_count = 0

        @breaker.protect
        def may_fail(should_fail: bool):
            nonlocal call_count
            call_count += 1
            if should_fail:
                raise RuntimeError("failure")
            return "success"

        # Two failures should open circuit
        with pytest.raises(RuntimeError):
            may_fail(True)
        with pytest.raises(RuntimeError):
            may_fail(True)

        # Circuit is now open
        with pytest.raises(RuntimeError, match="OPEN"):
            may_fail(False)

        # Call count should be 2 (third call was rejected)
        assert call_count == 2

    def test_reset(self):
        """Manual reset works."""
        config = StatefulCircuitBreakerConfig(failure_threshold=1)
        breaker = StatefulCircuitBreaker("test", config)

        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        breaker.reset()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.allow_request()

    def test_get_stats(self):
        """Stats are returned correctly."""
        breaker = StatefulCircuitBreaker("test")
        breaker.record_failure()
        breaker.record_failure()

        stats = breaker.get_stats()
        assert stats["name"] == "test"
        assert stats["state"] == "closed"  # Still under threshold
        assert stats["failure_count"] == 2


# ============================================================================
# THROTTLING TESTS
# ============================================================================

class TestThrottler:
    """Tests for operation throttling."""

    def test_throttle_allows_within_limit(self):
        """Requests within limit are allowed."""
        throttler = Throttler("test", ThrottleConfig(
            max_calls=5,
            period_seconds=1.0,
        ))

        for _ in range(5):
            assert throttler.acquire()

    def test_throttle_blocks_over_limit(self):
        """Requests over limit are blocked/rejected."""
        throttler = Throttler("test", ThrottleConfig(
            max_calls=2,
            period_seconds=10.0,  # Long period so tokens don't refill
            wait_on_limit=False,
        ))

        assert throttler.acquire()
        assert throttler.acquire()

        with pytest.raises(RuntimeError, match="rate limit exceeded"):
            throttler.acquire()

    def test_throttle_refills_over_time(self):
        """Tokens refill over time."""
        throttler = Throttler("test", ThrottleConfig(
            max_calls=10,
            period_seconds=0.1,
        ))

        # Exhaust tokens
        for _ in range(10):
            throttler.acquire()

        # Wait for refill
        time.sleep(0.15)

        # Should have tokens again
        assert throttler.acquire()

    def test_throttle_decorator(self):
        """Throttle decorator works."""
        call_count = 0

        @with_throttle(max_calls=5, period_seconds=1.0)
        def rate_limited():
            nonlocal call_count
            call_count += 1
            return call_count

        for _ in range(5):
            rate_limited()

        assert call_count == 5

    def test_get_stats(self):
        """Stats are returned correctly."""
        throttler = Throttler("test", ThrottleConfig(max_calls=10))
        throttler.acquire()
        throttler.acquire()

        stats = throttler.get_stats()
        assert stats["name"] == "test"
        assert stats["max_calls_per_period"] == 10
        assert stats["available_tokens"] < 10


# ============================================================================
# DATA INTEGRITY TESTS
# ============================================================================

class TestDataIntegrity:
    """Tests for data integrity verification."""

    def test_compute_hash_deterministic(self):
        """Same data produces same hash."""
        data = {"ticker": "AAPL", "score": 85.5}

        hash1 = compute_data_hash(data)
        hash2 = compute_data_hash(data)

        assert hash1 == hash2
        assert hash1.startswith("sha256:")

    def test_compute_hash_order_independent(self):
        """Dict key order doesn't affect hash."""
        data1 = {"a": 1, "b": 2, "c": 3}
        data2 = {"c": 3, "a": 1, "b": 2}

        assert compute_data_hash(data1) == compute_data_hash(data2)

    def test_hash_handles_special_types(self):
        """Hash handles date, Decimal, etc."""
        data = {
            "date": date(2025, 1, 15),
            "amount": Decimal("123.45"),
            "items": {"nested": True},
        }

        hash_result = compute_data_hash(data)
        assert hash_result.startswith("sha256:")

    def test_verify_integrity_success(self):
        """Verification succeeds with correct hash."""
        data = {"ticker": "AAPL", "score": 85}
        expected = compute_data_hash(data)

        result = verify_data_integrity(data, expected)

        assert result.is_valid
        assert result.expected_hash == expected
        assert result.actual_hash == expected

    def test_verify_integrity_failure(self):
        """Verification fails with wrong hash."""
        data = {"ticker": "AAPL", "score": 85}

        result = verify_data_integrity(data, "sha256:wrong")

        assert not result.is_valid
        assert "mismatch" in result.message.lower()

    def test_verify_integrity_strict_mode(self):
        """Strict mode raises on mismatch."""
        data = {"ticker": "AAPL"}
        config = IntegrityConfig(strict_mode=True)

        with pytest.raises(ValueError, match="mismatch"):
            verify_data_integrity(data, "sha256:wrong", config)

    def test_different_algorithms(self):
        """Different algorithms produce different hashes."""
        data = {"test": "data"}

        sha256 = compute_data_hash(data, IntegrityConfig(hash_algorithm="sha256"))
        sha512 = compute_data_hash(data, IntegrityConfig(hash_algorithm="sha512"))
        md5 = compute_data_hash(data, IntegrityConfig(hash_algorithm="md5"))

        assert sha256 != sha512 != md5
        assert sha256.startswith("sha256:")
        assert sha512.startswith("sha512:")
        assert md5.startswith("md5:")


# ============================================================================
# IDEMPOTENCY TESTS
# ============================================================================

class TestIdempotency:
    """Tests for idempotency tracking."""

    def test_idempotency_store_basic(self):
        """Basic store operations work."""
        store = IdempotencyStore()

        key, exists = store.register("op", "key1")
        assert not exists

        key2, exists2 = store.register("op", "key1")
        assert exists2
        assert key == key2

    def test_idempotency_completion(self):
        """Completed operations are tracked."""
        store = IdempotencyStore()

        key, _ = store.register("op", "key1")
        assert not store.is_completed(key)

        store.mark_completed(key, {"result": 42})
        assert store.is_completed(key)
        assert store.get_result(key) == {"result": 42}

    def test_idempotent_decorator(self):
        """Idempotent decorator prevents duplicate execution."""
        store = IdempotencyStore()
        call_count = 0

        @idempotent(store, key_func=lambda x: str(x))
        def expensive_operation(value: int) -> int:
            nonlocal call_count
            call_count += 1
            return value * 2

        # First call executes
        result1 = expensive_operation(5)
        assert result1 == 10
        assert call_count == 1

        # Second call returns cached result
        result2 = expensive_operation(5)
        assert result2 == 10
        assert call_count == 1  # Not incremented

        # Different key executes
        result3 = expensive_operation(7)
        assert result3 == 14
        assert call_count == 2

    def test_idempotency_ttl(self):
        """Old entries are cleaned up."""
        store = IdempotencyStore(ttl_seconds=0.1)

        key, _ = store.register("op", "key1")
        store.mark_completed(key, "result")

        time.sleep(0.15)

        # Entry should be cleaned up on next register
        key2, exists = store.register("op", "key1")
        assert not exists  # Expired, so treated as new

    def test_idempotency_max_entries(self):
        """Max entries limit is enforced."""
        store = IdempotencyStore(max_entries=3)

        for i in range(5):
            store.register("op", f"key{i}")

        stats = store.get_stats()
        assert stats["total_entries"] <= 3


# ============================================================================
# HEALTH CHECK TESTS
# ============================================================================

class TestHealthChecks:
    """Tests for module health checks."""

    def test_health_checker_basic(self):
        """Basic health check works."""
        checker = ModuleHealthChecker("test_module")

        def healthy_check() -> HealthCheckResult:
            return HealthCheckResult(
                name="db_connection",
                status=HealthStatus.HEALTHY,
                message="Connected",
            )

        checker.add_check("db", healthy_check)
        results = checker.run_all_checks()

        assert len(results) == 1
        assert results[0].status == HealthStatus.HEALTHY

    def test_overall_status_healthy(self):
        """Overall status is HEALTHY when all checks pass."""
        checker = ModuleHealthChecker("test")

        checker.add_check("check1", lambda: HealthCheckResult(
            name="check1", status=HealthStatus.HEALTHY, message="OK"
        ))
        checker.add_check("check2", lambda: HealthCheckResult(
            name="check2", status=HealthStatus.HEALTHY, message="OK"
        ))

        checker.run_all_checks()
        assert checker.get_overall_status() == HealthStatus.HEALTHY

    def test_overall_status_degraded(self):
        """Overall status is DEGRADED when any check is degraded."""
        checker = ModuleHealthChecker("test")

        checker.add_check("check1", lambda: HealthCheckResult(
            name="check1", status=HealthStatus.HEALTHY, message="OK"
        ))
        checker.add_check("check2", lambda: HealthCheckResult(
            name="check2", status=HealthStatus.DEGRADED, message="Slow"
        ))

        checker.run_all_checks()
        assert checker.get_overall_status() == HealthStatus.DEGRADED

    def test_overall_status_unhealthy(self):
        """Overall status is UNHEALTHY when any check fails."""
        checker = ModuleHealthChecker("test")

        checker.add_check("check1", lambda: HealthCheckResult(
            name="check1", status=HealthStatus.HEALTHY, message="OK"
        ))
        checker.add_check("check2", lambda: HealthCheckResult(
            name="check2", status=HealthStatus.UNHEALTHY, message="Down"
        ))

        checker.run_all_checks()
        assert checker.get_overall_status() == HealthStatus.UNHEALTHY

    def test_exception_handling(self):
        """Exceptions in checks are handled."""
        checker = ModuleHealthChecker("test")

        def failing_check():
            raise RuntimeError("Check failed")

        checker.add_check("failing", failing_check)
        results = checker.run_all_checks()

        assert len(results) == 1
        assert results[0].status == HealthStatus.UNHEALTHY

    def test_latency_tracking(self):
        """Check latency is tracked."""
        checker = ModuleHealthChecker("test")

        def slow_check():
            time.sleep(0.05)
            return HealthCheckResult(
                name="slow",
                status=HealthStatus.HEALTHY,
                message="OK",
            )

        checker.add_check("slow", slow_check)
        results = checker.run_all_checks()

        assert results[0].latency_ms >= 50


# ============================================================================
# WATCHDOG TESTS
# ============================================================================

class TestWatchdog:
    """Tests for operation watchdog."""

    def test_watchdog_basic(self):
        """Basic watchdog operations work."""
        watchdog = OperationWatchdog()

        watchdog.start_operation("test_op")
        watchdog.heartbeat("test_op")
        watchdog.complete_operation("test_op")

        stats = watchdog.get_stats()
        assert stats["active_operations"] == 0

    def test_watchdog_detects_stuck(self):
        """Watchdog detects stuck operations."""
        config = WatchdogConfig(timeout_seconds=0.1, on_timeout="warn")
        watchdog = OperationWatchdog(config)

        watchdog.start_operation("stuck_op")
        time.sleep(0.15)

        timed_out = watchdog.check_operations()
        assert "stuck_op" in timed_out

    def test_watchdog_heartbeat_resets_timeout(self):
        """Heartbeats reset the timeout."""
        config = WatchdogConfig(timeout_seconds=0.2)
        watchdog = OperationWatchdog(config)

        watchdog.start_operation("op")
        time.sleep(0.1)
        watchdog.heartbeat("op")
        time.sleep(0.1)

        # Should not be timed out because heartbeat reset timer
        timed_out = watchdog.check_operations()
        assert "op" not in timed_out

    def test_watchdog_raise_mode(self):
        """Watchdog raises exception on timeout in raise mode."""
        config = WatchdogConfig(timeout_seconds=0.1, on_timeout="raise")
        watchdog = OperationWatchdog(config)

        watchdog.start_operation("op")
        time.sleep(0.15)

        with pytest.raises(WatchdogTimeoutError):
            watchdog.check_operations()

    def test_watchdog_callback_mode(self):
        """Watchdog calls callback on timeout."""
        callback_called = []

        def timeout_callback(op_id: str):
            callback_called.append(op_id)

        config = WatchdogConfig(
            timeout_seconds=0.1,
            on_timeout="callback",
            callback=timeout_callback,
        )
        watchdog = OperationWatchdog(config)

        watchdog.start_operation("op1")
        time.sleep(0.15)
        watchdog.check_operations()

        assert "op1" in callback_called

    def test_watchdog_background_monitoring(self):
        """Background monitoring works."""
        callback_called = []

        config = WatchdogConfig(
            timeout_seconds=0.1,
            check_interval_seconds=0.05,
            on_timeout="callback",
            callback=lambda op: callback_called.append(op),
        )
        watchdog = OperationWatchdog(config)

        watchdog.start_monitoring()
        watchdog.start_operation("bg_op")

        time.sleep(0.2)  # Wait for background check

        watchdog.stop_monitoring()

        assert "bg_op" in callback_called

    def test_watchdog_stats(self):
        """Stats are returned correctly."""
        watchdog = OperationWatchdog()

        watchdog.start_operation("op1")
        watchdog.start_operation("op2")
        watchdog.heartbeat("op1")

        stats = watchdog.get_stats()
        assert stats["active_operations"] == 2
        assert "op1" in stats["operations"]
        assert stats["operations"]["op1"]["heartbeat_count"] == 1


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestRobustnessIntegration:
    """Integration tests for robustness utilities."""

    def test_create_pipeline_context(self):
        """Pipeline context creation works."""
        ctx = create_robust_pipeline_context()

        assert "correlation_id" in ctx
        assert ctx["circuit_breaker"] is not None
        assert ctx["throttler"] is not None
        assert ctx["idempotency_store"] is not None
        assert ctx["watchdog"] is not None
        assert ctx["health_checker"] is not None

    def test_combined_robustness_scenario(self):
        """Combined use of multiple robustness features."""
        ctx = create_robust_pipeline_context("test-123")

        # Setup health checker
        ctx["health_checker"].add_check("test", lambda: HealthCheckResult(
            name="test",
            status=HealthStatus.HEALTHY,
            message="OK",
        ))

        # Start watchdog
        ctx["watchdog"].start_operation("combined_test")

        # Use circuit breaker
        assert ctx["circuit_breaker"].allow_request()
        ctx["circuit_breaker"].record_success()

        # Use throttler
        assert ctx["throttler"].acquire()

        # Use idempotency
        key, _ = ctx["idempotency_store"].register("test", "key1")
        ctx["idempotency_store"].mark_completed(key, "result")

        # Heartbeat and complete
        ctx["watchdog"].heartbeat("combined_test")
        ctx["watchdog"].complete_operation("combined_test")

        # Check health
        ctx["health_checker"].run_all_checks()
        assert ctx["health_checker"].get_overall_status() == HealthStatus.HEALTHY

    def test_thread_safety(self):
        """Multiple threads can use robustness utilities safely."""
        breaker = StatefulCircuitBreaker("thread_test")
        store = IdempotencyStore()
        errors = []

        def worker(worker_id: int):
            try:
                for i in range(10):
                    if breaker.allow_request():
                        # Simulate work
                        key, _ = store.register("op", f"{worker_id}_{i}")
                        if not store.is_completed(key):
                            store.mark_completed(key, {"worker": worker_id, "i": i})
                        breaker.record_success()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert breaker.state == CircuitState.CLOSED

    def test_data_integrity_in_pipeline(self):
        """Data integrity checks work in pipeline context."""
        # Simulate module output
        module_output = {
            "scores": [
                {"ticker": "AAPL", "score": Decimal("85.5")},
                {"ticker": "GOOG", "score": Decimal("72.3")},
            ],
            "as_of_date": date(2025, 1, 15),
            "module": "test",
        }

        # Compute hash for later verification
        original_hash = compute_data_hash(module_output)

        # Simulate passing through pipeline
        received_output = module_output.copy()

        # Verify integrity
        result = verify_data_integrity(received_output, original_hash)
        assert result.is_valid

        # Tamper with data
        received_output["scores"][0]["score"] = Decimal("99.9")

        # Integrity check should fail
        result = verify_data_integrity(received_output, original_hash)
        assert not result.is_valid
