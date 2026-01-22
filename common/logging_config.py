"""
Production logging configuration with security features.

Provides:
- Log rotation (by size and time)
- Secure log handling
- Sanitization of sensitive data
- Structured logging support
- Correlation ID tracking

Version: 1.0.0
"""

import json
import logging
import os
import sys
from contextvars import ContextVar
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional, Union
import uuid

# Context variable for run correlation ID
run_id_context: ContextVar[str] = ContextVar("run_id", default="")

# Default configuration
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_FORMAT_WITH_RUN_ID = "%(asctime)s - [%(run_id)s] - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_BACKUP_COUNT = 5

# Patterns that should be redacted from logs
SENSITIVE_PATTERNS = frozenset({
    "api_key", "apikey", "api-key",
    "password", "passwd", "pwd",
    "secret", "token", "credential",
    "ssn", "social_security",
    "account_number", "account_num",
    "cusip", "isin",
    "bearer", "authorization",
})


class RunIdFilter(logging.Filter):
    """
    Filter that adds run_id to log records.

    Enables tracking of log entries across a single pipeline run.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = run_id_context.get() or "no-run-id"
        return True


class SanitizingFilter(logging.Filter):
    """
    Filter that sanitizes sensitive data from log messages.

    Prevents accidental logging of API keys, passwords, etc.
    """

    def __init__(self, patterns: Optional[frozenset] = None):
        super().__init__()
        self.patterns = patterns or SENSITIVE_PATTERNS

    def filter(self, record: logging.LogRecord) -> bool:
        # Sanitize the message
        if hasattr(record, "msg") and isinstance(record.msg, str):
            record.msg = self._sanitize(record.msg)

        # Sanitize args if present
        if hasattr(record, "args") and record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: self._sanitize(str(v)) if self._is_sensitive_key(k) else v
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    self._sanitize(str(arg)) if isinstance(arg, str) else arg
                    for arg in record.args
                )

        return True

    def _sanitize(self, text: str) -> str:
        """Sanitize sensitive patterns from text."""
        text_lower = text.lower()
        for pattern in self.patterns:
            if pattern in text_lower:
                # Find and redact the value after the pattern
                # This handles cases like "api_key=abc123" or "api_key: abc123"
                import re
                regex = rf"({pattern})\s*[=:]\s*['\"]?([^'\"\s,}}]+)['\"]?"
                text = re.sub(regex, r"\1=[REDACTED]", text, flags=re.IGNORECASE)
        return text

    def _is_sensitive_key(self, key: str) -> bool:
        """Check if a key name appears sensitive."""
        key_lower = key.lower()
        return any(pattern in key_lower for pattern in self.patterns)


class StructuredFormatter(logging.Formatter):
    """
    Formatter that outputs JSON-structured log entries.

    Useful for log aggregation systems like ELK, Splunk, etc.
    """

    def __init__(
        self,
        include_timestamp: bool = True,
        include_level: bool = True,
        include_name: bool = True,
        include_run_id: bool = True,
        extra_fields: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_name = include_name
        self.include_run_id = include_run_id
        self.extra_fields = extra_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {}

        if self.include_timestamp:
            log_data["timestamp"] = datetime.utcnow().isoformat() + "Z"

        if self.include_level:
            log_data["level"] = record.levelname

        if self.include_name:
            log_data["logger"] = record.name

        if self.include_run_id and hasattr(record, "run_id"):
            log_data["run_id"] = record.run_id

        log_data["message"] = record.getMessage()

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        log_data.update(self.extra_fields)

        return json.dumps(log_data, default=str)


def setup_logging(
    log_file: Optional[Union[str, Path]] = None,
    log_level: int = logging.INFO,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    enable_console: bool = True,
    enable_sanitization: bool = True,
    enable_run_id: bool = True,
    structured_output: bool = False,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """
    Configure production logging with rotation and security features.

    Args:
        log_file: Path to log file (None = console only)
        log_level: Logging level (default INFO)
        max_bytes: Maximum log file size before rotation (default 10MB)
        backup_count: Number of backup files to keep (default 5)
        enable_console: Enable console output (default True)
        enable_sanitization: Enable sensitive data sanitization (default True)
        enable_run_id: Enable run ID tracking (default True)
        structured_output: Use JSON structured output (default False)
        log_format: Custom log format string

    Returns:
        Configured root logger

    Example:
        logger = setup_logging(
            log_file="logs/pipeline.log",
            log_level=logging.INFO,
            enable_run_id=True,
        )
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers = []

    # Determine format
    if log_format is None:
        log_format = DEFAULT_LOG_FORMAT_WITH_RUN_ID if enable_run_id else DEFAULT_LOG_FORMAT

    # Create formatter
    if structured_output:
        formatter = StructuredFormatter(include_run_id=enable_run_id)
    else:
        formatter = logging.Formatter(log_format, datefmt=DEFAULT_DATE_FORMAT)

    # Add filters
    filters = []
    if enable_sanitization:
        filters.append(SanitizingFilter())
    if enable_run_id:
        filters.append(RunIdFilter())

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(formatter)
        for f in filters:
            console_handler.addFilter(f)
        root_logger.addHandler(console_handler)

    # File handler with rotation
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Set secure permissions on log directory
        if os.name == "posix":
            try:
                os.chmod(log_path.parent, 0o700)
            except OSError:
                pass  # May fail if not owner

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        for f in filters:
            file_handler.addFilter(f)
        root_logger.addHandler(file_handler)

        # Set secure permissions on log file
        if os.name == "posix":
            try:
                os.chmod(log_path, 0o600)
            except OSError:
                pass

    return root_logger


def setup_timed_rotating_logging(
    log_file: Union[str, Path],
    when: str = "midnight",
    interval: int = 1,
    backup_count: int = 30,
    **kwargs,
) -> logging.Logger:
    """
    Configure logging with time-based rotation.

    Args:
        log_file: Path to log file
        when: Rotation timing ("midnight", "H", "D", "W0"-"W6")
        interval: Rotation interval
        backup_count: Number of backup files to keep
        **kwargs: Additional arguments passed to setup_logging

    Returns:
        Configured root logger
    """
    # First set up basic logging
    logger = setup_logging(log_file=None, **kwargs)

    # Add timed rotating handler
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = TimedRotatingFileHandler(
        log_path,
        when=when,
        interval=interval,
        backupCount=backup_count,
        encoding="utf-8",
    )

    # Copy format from console handler
    if logger.handlers:
        file_handler.setFormatter(logger.handlers[0].formatter)
        for f in logger.handlers[0].filters:
            file_handler.addFilter(f)

    logger.addHandler(file_handler)

    return logger


def generate_run_id() -> str:
    """
    Generate a unique run ID for log correlation.

    Returns:
        Short unique ID string
    """
    return str(uuid.uuid4())[:8]


def set_run_id(run_id: Optional[str] = None) -> str:
    """
    Set the run ID for the current context.

    Args:
        run_id: Run ID to set (generates new if None)

    Returns:
        The run ID that was set
    """
    if run_id is None:
        run_id = generate_run_id()
    run_id_context.set(run_id)
    return run_id


def get_run_id() -> str:
    """
    Get the current run ID.

    Returns:
        Current run ID or empty string if not set
    """
    return run_id_context.get()


class LogContext:
    """
    Context manager for setting run ID.

    Example:
        with LogContext() as run_id:
            logger.info("Starting pipeline")  # Includes run_id
    """

    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id
        self._token = None

    def __enter__(self) -> str:
        self.run_id = set_run_id(self.run_id)
        return self.run_id

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        run_id_context.set("")


__all__ = [
    # Setup functions
    "setup_logging",
    "setup_timed_rotating_logging",
    # Run ID management
    "generate_run_id",
    "set_run_id",
    "get_run_id",
    "LogContext",
    "run_id_context",
    # Filters and formatters
    "RunIdFilter",
    "SanitizingFilter",
    "StructuredFormatter",
    # Constants
    "DEFAULT_LOG_FORMAT",
    "DEFAULT_MAX_BYTES",
    "DEFAULT_BACKUP_COUNT",
    "SENSITIVE_PATTERNS",
]
