#!/usr/bin/env python3
"""
Tests for common/logging_config.py

Production logging configuration with security features.
Tests cover:
- Run ID generation and management (generate_run_id, set_run_id, get_run_id)
- RunIdFilter (adds run_id to log records)
- SanitizingFilter (redacts sensitive data)
- StructuredFormatter (JSON output)
- LogContext context manager
"""

import json
import logging
import pytest
from unittest.mock import MagicMock, patch

from common.logging_config import (
    # Run ID management
    generate_run_id,
    set_run_id,
    get_run_id,
    run_id_context,
    LogContext,
    # Filters
    RunIdFilter,
    SanitizingFilter,
    # Formatter
    StructuredFormatter,
    # Setup functions
    setup_logging,
    # Constants
    DEFAULT_LOG_FORMAT,
    DEFAULT_MAX_BYTES,
    DEFAULT_BACKUP_COUNT,
    SENSITIVE_PATTERNS,
)


class TestGenerateRunId:
    """Tests for generate_run_id function."""

    def test_returns_string(self):
        """Should return a string."""
        run_id = generate_run_id()
        assert isinstance(run_id, str)

    def test_returns_8_characters(self):
        """Should return an 8-character ID."""
        run_id = generate_run_id()
        assert len(run_id) == 8

    def test_returns_unique_ids(self):
        """Should return unique IDs on each call."""
        ids = [generate_run_id() for _ in range(100)]
        assert len(set(ids)) == 100  # All unique

    def test_returns_hex_characters(self):
        """Should return valid hex characters."""
        run_id = generate_run_id()
        # UUID chars are 0-9 and a-f
        valid_chars = set("0123456789abcdef-")
        assert all(c in valid_chars for c in run_id)


class TestSetRunId:
    """Tests for set_run_id function."""

    def teardown_method(self):
        """Reset run_id after each test."""
        run_id_context.set("")

    def test_sets_provided_id(self):
        """Should set the provided run ID."""
        result = set_run_id("test-1234")
        assert result == "test-1234"
        assert get_run_id() == "test-1234"

    def test_generates_id_if_none(self):
        """Should generate an ID if None provided."""
        result = set_run_id(None)
        assert isinstance(result, str)
        assert len(result) == 8
        assert get_run_id() == result

    def test_returns_set_id(self):
        """Should return the ID that was set."""
        result = set_run_id("my-run-id")
        assert result == "my-run-id"


class TestGetRunId:
    """Tests for get_run_id function."""

    def teardown_method(self):
        """Reset run_id after each test."""
        run_id_context.set("")

    def test_returns_empty_if_not_set(self):
        """Should return empty string if not set."""
        run_id_context.set("")
        assert get_run_id() == ""

    def test_returns_set_id(self):
        """Should return the set run ID."""
        set_run_id("test-id")
        assert get_run_id() == "test-id"


class TestLogContext:
    """Tests for LogContext context manager."""

    def teardown_method(self):
        """Reset run_id after each test."""
        run_id_context.set("")

    def test_sets_run_id_on_enter(self):
        """Should set run ID when entering context."""
        with LogContext("ctx-test") as run_id:
            assert run_id == "ctx-test"
            assert get_run_id() == "ctx-test"

    def test_generates_run_id_if_not_provided(self):
        """Should generate run ID if not provided."""
        with LogContext() as run_id:
            assert isinstance(run_id, str)
            assert len(run_id) == 8
            assert get_run_id() == run_id

    def test_clears_run_id_on_exit(self):
        """Should clear run ID when exiting context."""
        with LogContext("test-id"):
            pass
        assert get_run_id() == ""

    def test_clears_run_id_on_exception(self):
        """Should clear run ID even if exception occurs."""
        try:
            with LogContext("test-id"):
                raise ValueError("test error")
        except ValueError:
            pass
        assert get_run_id() == ""


class TestRunIdFilter:
    """Tests for RunIdFilter class."""

    def teardown_method(self):
        """Reset run_id after each test."""
        run_id_context.set("")

    def test_adds_run_id_to_record(self):
        """Should add run_id attribute to log record."""
        set_run_id("filter-test")
        filter_obj = RunIdFilter()

        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="test message", args=(), exc_info=None
        )

        filter_obj.filter(record)
        assert hasattr(record, "run_id")
        assert record.run_id == "filter-test"

    def test_uses_default_if_not_set(self):
        """Should use 'no-run-id' if run ID not set."""
        run_id_context.set("")
        filter_obj = RunIdFilter()

        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="test message", args=(), exc_info=None
        )

        filter_obj.filter(record)
        assert record.run_id == "no-run-id"

    def test_always_returns_true(self):
        """Filter should always return True (allow record)."""
        filter_obj = RunIdFilter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="test message", args=(), exc_info=None
        )
        assert filter_obj.filter(record) is True


class TestSanitizingFilter:
    """Tests for SanitizingFilter class."""

    def test_sanitizes_api_key(self):
        """Should sanitize api_key in message."""
        filter_obj = SanitizingFilter()

        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="api_key=secret123", args=(), exc_info=None
        )

        filter_obj.filter(record)
        assert "secret123" not in record.msg
        assert "[REDACTED]" in record.msg

    def test_sanitizes_password(self):
        """Should sanitize password in message."""
        filter_obj = SanitizingFilter()

        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="password: mypass123", args=(), exc_info=None
        )

        filter_obj.filter(record)
        assert "mypass123" not in record.msg
        assert "[REDACTED]" in record.msg

    def test_sanitizes_token(self):
        """Should sanitize token in message."""
        filter_obj = SanitizingFilter()

        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="token='abc123xyz'", args=(), exc_info=None
        )

        filter_obj.filter(record)
        assert "abc123xyz" not in record.msg
        assert "[REDACTED]" in record.msg

    def test_preserves_safe_messages(self):
        """Should not modify messages without sensitive data."""
        filter_obj = SanitizingFilter()

        original_msg = "Processing ticker AMGN with score 85.5"
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg=original_msg, args=(), exc_info=None
        )

        filter_obj.filter(record)
        assert record.msg == original_msg

    def test_case_insensitive(self):
        """Should sanitize regardless of case."""
        filter_obj = SanitizingFilter()

        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="API_KEY=SECRET", args=(), exc_info=None
        )

        filter_obj.filter(record)
        assert "SECRET" not in record.msg
        assert "[REDACTED]" in record.msg

    def test_custom_patterns(self):
        """Should use custom patterns if provided."""
        filter_obj = SanitizingFilter(patterns=frozenset({"custom_secret"}))

        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="custom_secret=myvalue", args=(), exc_info=None
        )

        filter_obj.filter(record)
        assert "myvalue" not in record.msg
        assert "[REDACTED]" in record.msg

    def test_always_returns_true(self):
        """Filter should always return True (allow record)."""
        filter_obj = SanitizingFilter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="test", args=(), exc_info=None
        )
        assert filter_obj.filter(record) is True

    def test_is_sensitive_key(self):
        """Should correctly identify sensitive keys."""
        filter_obj = SanitizingFilter()

        assert filter_obj._is_sensitive_key("api_key") is True
        assert filter_obj._is_sensitive_key("password") is True
        assert filter_obj._is_sensitive_key("user_token") is True
        assert filter_obj._is_sensitive_key("ticker") is False
        assert filter_obj._is_sensitive_key("score") is False


class TestStructuredFormatter:
    """Tests for StructuredFormatter class."""

    def test_outputs_valid_json(self):
        """Should output valid JSON."""
        formatter = StructuredFormatter()

        record = logging.LogRecord(
            name="test.logger", level=logging.INFO, pathname="",
            lineno=0, msg="test message", args=(), exc_info=None
        )
        record.run_id = "test-run"

        output = formatter.format(record)
        parsed = json.loads(output)

        assert isinstance(parsed, dict)

    def test_includes_timestamp(self):
        """Should include timestamp by default."""
        formatter = StructuredFormatter(include_timestamp=True)

        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="test", args=(), exc_info=None
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert "timestamp" in parsed
        assert "Z" in parsed["timestamp"]  # UTC indicator

    def test_excludes_timestamp_when_disabled(self):
        """Should exclude timestamp when disabled."""
        formatter = StructuredFormatter(include_timestamp=False)

        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="test", args=(), exc_info=None
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert "timestamp" not in parsed

    def test_includes_level(self):
        """Should include log level by default."""
        formatter = StructuredFormatter(include_level=True)

        record = logging.LogRecord(
            name="test", level=logging.WARNING, pathname="",
            lineno=0, msg="test", args=(), exc_info=None
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["level"] == "WARNING"

    def test_includes_logger_name(self):
        """Should include logger name by default."""
        formatter = StructuredFormatter(include_name=True)

        record = logging.LogRecord(
            name="my.logger.name", level=logging.INFO, pathname="",
            lineno=0, msg="test", args=(), exc_info=None
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["logger"] == "my.logger.name"

    def test_includes_run_id(self):
        """Should include run_id by default."""
        formatter = StructuredFormatter(include_run_id=True)

        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="test", args=(), exc_info=None
        )
        record.run_id = "struct-test"

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["run_id"] == "struct-test"

    def test_includes_message(self):
        """Should always include message."""
        formatter = StructuredFormatter()

        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="Hello %s", args=("World",), exc_info=None
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["message"] == "Hello World"

    def test_includes_extra_fields(self):
        """Should include extra fields if provided."""
        formatter = StructuredFormatter(
            extra_fields={"service": "screener", "version": "1.0"}
        )

        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="test", args=(), exc_info=None
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["service"] == "screener"
        assert parsed["version"] == "1.0"


class TestSetupLogging:
    """Tests for setup_logging function."""

    def teardown_method(self):
        """Reset root logger after each test."""
        root = logging.getLogger()
        root.handlers = []
        root.setLevel(logging.WARNING)

    def test_returns_logger(self):
        """Should return a logger instance."""
        logger = setup_logging(enable_console=False)
        assert isinstance(logger, logging.Logger)

    def test_sets_log_level(self):
        """Should set the specified log level."""
        logger = setup_logging(log_level=logging.DEBUG, enable_console=False)
        assert logger.level == logging.DEBUG

    def test_clears_existing_handlers(self):
        """Should clear existing handlers."""
        # Note: pytest adds its own handlers, so we check that our custom
        # handlers are cleared, not that there are exactly 0 handlers
        root = logging.getLogger()
        initial_count = len(root.handlers)

        # Add two custom handlers
        root.addHandler(logging.StreamHandler())
        root.addHandler(logging.StreamHandler())

        assert len(root.handlers) == initial_count + 2

        # After setup_logging with no console, our handlers should be cleared
        # (setup_logging clears all handlers then adds its own)
        setup_logging(enable_console=False)
        # With no console and no file, there should be no handlers
        assert len(root.handlers) == 0

    def test_adds_console_handler_when_enabled(self):
        """Should add console handler when enabled."""
        setup_logging(enable_console=True)
        root = logging.getLogger()

        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], logging.StreamHandler)

    def test_no_console_handler_when_disabled(self):
        """Should not add console handler when disabled."""
        setup_logging(enable_console=False)
        root = logging.getLogger()

        assert len(root.handlers) == 0


class TestConstants:
    """Tests for module constants."""

    def test_default_log_format_is_string(self):
        """Default log format should be a string."""
        assert isinstance(DEFAULT_LOG_FORMAT, str)
        assert "%(asctime)s" in DEFAULT_LOG_FORMAT

    def test_default_max_bytes(self):
        """Default max bytes should be 10 MB."""
        assert DEFAULT_MAX_BYTES == 10 * 1024 * 1024

    def test_default_backup_count(self):
        """Default backup count should be 5."""
        assert DEFAULT_BACKUP_COUNT == 5

    def test_sensitive_patterns_is_frozenset(self):
        """Sensitive patterns should be a frozenset."""
        assert isinstance(SENSITIVE_PATTERNS, frozenset)

    def test_sensitive_patterns_contains_common_secrets(self):
        """Sensitive patterns should contain common secret keywords."""
        assert "api_key" in SENSITIVE_PATTERNS
        assert "password" in SENSITIVE_PATTERNS
        assert "token" in SENSITIVE_PATTERNS
        assert "secret" in SENSITIVE_PATTERNS
