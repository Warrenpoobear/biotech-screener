"""
Tests for production hardening utilities.

These tests verify the security and reliability features including:
- Path traversal protection
- Operation timeouts
- File integrity verification
- Secure file operations
- Input validation
- Logging sanitization
"""

import json
import os
import tempfile
import time
from datetime import date
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest

from common.production_hardening import (
    # Exceptions
    FileSizeError,
    IntegrityError,
    OperationTimeoutError,
    PathTraversalError,
    SecurityError,
    SymlinkError,
    DateParseError,
    FileOperationError,
    FileReadError,
    FileWriteError,
    DirectoryError,
    # Path validation
    safe_join_path,
    validate_checkpoint_path,
    validate_path_within_base,
    # Timeouts
    operation_timeout,
    with_timeout,
    # Integrity
    compute_content_hash,
    compute_file_hash,
    load_with_integrity_check,
    save_with_integrity,
    verify_integrity,
    # File operations
    safe_mkdir,
    safe_read_json,
    safe_write_json,
    validate_file_size,
    # File I/O with error handling
    safe_mkdir_with_error,
    safe_file_write,
    safe_file_read,
    safe_json_write,
    safe_json_read,
    # Logging
    sanitize_for_logging,
    ModuleLogger,
    get_module_logger,
    # Validation
    validate_date_format,
    validate_numeric_bounds,
    validate_ticker_format,
    # Safe date parsing
    safe_parse_date,
    safe_parse_date_or_none,
    # Safe dict access
    safe_get_nested,
    safe_get_decimal,
    safe_get_int,
    safe_get_str,
    # Decimal safety
    DECIMAL_BOUNDS,
    clamp_decimal,
    safe_decimal_divide,
    validate_decimal_field,
    # Resources
    check_available_memory_mb,
    # Constants
    MAX_JSON_FILE_SIZE_MB,
)


# =============================================================================
# Path Traversal Protection Tests
# =============================================================================


class TestPathTraversalProtection:
    """Tests for path traversal protection utilities."""

    def test_validate_path_within_base_valid(self, tmp_path):
        """Valid paths within base directory should pass."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        file_path = subdir / "test.json"
        file_path.touch()

        result = validate_path_within_base(file_path, tmp_path)
        assert result.exists()

    def test_validate_path_within_base_traversal_attack(self, tmp_path):
        """Path traversal attempts should be rejected."""
        # Attempt to escape base directory
        malicious_path = tmp_path / ".." / ".." / "etc" / "passwd"

        with pytest.raises(PathTraversalError):
            validate_path_within_base(malicious_path, tmp_path)

    def test_validate_path_within_base_absolute_escape(self, tmp_path):
        """Absolute paths outside base should be rejected."""
        with pytest.raises(PathTraversalError):
            validate_path_within_base(Path("/etc/passwd"), tmp_path)

    def test_validate_path_within_base_symlink_rejected(self, tmp_path):
        """Symlinks should be rejected by default."""
        target = tmp_path / "target.txt"
        target.write_text("content")
        link = tmp_path / "link.txt"
        link.symlink_to(target)

        with pytest.raises(SymlinkError):
            validate_path_within_base(link, tmp_path, allow_symlinks=False)

    def test_validate_path_within_base_symlink_allowed(self, tmp_path):
        """Symlinks should pass when explicitly allowed."""
        target = tmp_path / "target.txt"
        target.write_text("content")
        link = tmp_path / "link.txt"
        link.symlink_to(target)

        result = validate_path_within_base(link, tmp_path, allow_symlinks=True)
        assert result.exists()

    def test_safe_join_path_valid(self, tmp_path):
        """Valid path components should join safely."""
        result = safe_join_path(tmp_path, "subdir", "file.json")
        expected = tmp_path / "subdir" / "file.json"
        assert result == expected.resolve()

    def test_safe_join_path_traversal_in_component(self, tmp_path):
        """Path components with .. should be rejected."""
        with pytest.raises(PathTraversalError):
            safe_join_path(tmp_path, "..", "file.json")

    def test_safe_join_path_absolute_component(self, tmp_path):
        """Absolute path components should be rejected."""
        with pytest.raises(PathTraversalError):
            safe_join_path(tmp_path, "/etc", "passwd")

    def test_validate_checkpoint_path_valid(self, tmp_path):
        """Valid checkpoint paths should pass."""
        result = validate_checkpoint_path(tmp_path, "module_1", "2026-01-15")
        assert result.name == "module_1_2026-01-15.json"

    def test_validate_checkpoint_path_invalid_module_name(self, tmp_path):
        """Invalid module names should be rejected."""
        with pytest.raises(PathTraversalError):
            validate_checkpoint_path(tmp_path, "../etc", "2026-01-15")

        with pytest.raises(PathTraversalError):
            validate_checkpoint_path(tmp_path, "module;drop table", "2026-01-15")

    def test_validate_checkpoint_path_invalid_date_format(self, tmp_path):
        """Invalid date formats should be rejected."""
        with pytest.raises(ValueError):
            validate_checkpoint_path(tmp_path, "module_1", "2026/01/15")

        with pytest.raises(ValueError):
            validate_checkpoint_path(tmp_path, "module_1", "../../../etc")


# =============================================================================
# Operation Timeout Tests
# =============================================================================


class TestOperationTimeout:
    """Tests for operation timeout utilities."""

    @pytest.mark.skipif(os.name != "posix", reason="SIGALRM not available on Windows")
    def test_operation_timeout_completes_in_time(self):
        """Operations completing in time should succeed."""
        with operation_timeout(5, "Quick operation"):
            time.sleep(0.1)  # Should complete quickly

    @pytest.mark.skipif(os.name != "posix", reason="SIGALRM not available on Windows")
    def test_operation_timeout_exceeds_limit(self):
        """Operations exceeding timeout should raise error."""
        with pytest.raises(OperationTimeoutError) as exc_info:
            with operation_timeout(1, "Slow operation"):
                time.sleep(5)  # Should timeout

        assert "Slow operation" in str(exc_info.value)
        assert "1s timeout" in str(exc_info.value)

    @pytest.mark.skipif(os.name != "posix", reason="SIGALRM not available on Windows")
    def test_with_timeout_decorator(self):
        """Timeout decorator should work on functions."""

        @with_timeout(1, "Test function")
        def slow_function():
            time.sleep(5)

        with pytest.raises(OperationTimeoutError):
            slow_function()


# =============================================================================
# File Integrity Verification Tests
# =============================================================================


class TestFileIntegrity:
    """Tests for file integrity verification utilities."""

    def test_compute_content_hash_deterministic(self):
        """Same content should produce same hash."""
        content = '{"key": "value"}'
        hash1 = compute_content_hash(content)
        hash2 = compute_content_hash(content)
        assert hash1 == hash2

    def test_compute_content_hash_different_content(self):
        """Different content should produce different hashes."""
        hash1 = compute_content_hash("content1")
        hash2 = compute_content_hash("content2")
        assert hash1 != hash2

    def test_compute_file_hash(self, tmp_path):
        """File hash should be computed correctly."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("test content")

        hash_value = compute_file_hash(file_path)
        assert hash_value.startswith("sha256:")
        assert len(hash_value) > 10

    def test_save_with_integrity(self, tmp_path):
        """Save should embed integrity metadata."""
        file_path = tmp_path / "test.json"
        data = {"key": "value", "number": 42}

        metadata = save_with_integrity(file_path, data, "2026-01-15")

        assert metadata.content_hash.startswith("sha256:")
        assert metadata.created_at == "2026-01-15"

        # Verify file contains integrity metadata
        with open(file_path) as f:
            saved_data = json.load(f)
        assert "_integrity" in saved_data

    def test_verify_integrity_valid(self, tmp_path):
        """Valid files should pass integrity check."""
        file_path = tmp_path / "test.json"
        data = {"key": "value"}
        save_with_integrity(file_path, data, "2026-01-15")

        assert verify_integrity(file_path) is True

    def test_verify_integrity_corrupted(self, tmp_path):
        """Corrupted files should fail integrity check."""
        file_path = tmp_path / "test.json"
        data = {"key": "value"}
        save_with_integrity(file_path, data, "2026-01-15")

        # Corrupt the file
        with open(file_path, "r") as f:
            content = json.load(f)
        content["key"] = "tampered"
        with open(file_path, "w") as f:
            json.dump(content, f)

        with pytest.raises(IntegrityError):
            verify_integrity(file_path)

    def test_load_with_integrity_check(self, tmp_path):
        """Load should verify integrity and return data without metadata."""
        file_path = tmp_path / "test.json"
        data = {"key": "value", "number": 42}
        save_with_integrity(file_path, data, "2026-01-15")

        loaded = load_with_integrity_check(file_path)
        assert loaded == data
        assert "_integrity" not in loaded


# =============================================================================
# Secure File Operations Tests
# =============================================================================


class TestSecureFileOperations:
    """Tests for secure file operation utilities."""

    def test_validate_file_size_valid(self, tmp_path):
        """Files within size limit should pass."""
        file_path = tmp_path / "small.json"
        file_path.write_text('{"key": "value"}')

        size = validate_file_size(file_path, max_size_mb=1)
        assert size > 0

    def test_validate_file_size_exceeds_limit(self, tmp_path):
        """Files exceeding size limit should raise error."""
        file_path = tmp_path / "large.json"
        # Create a file larger than 1KB
        file_path.write_text("x" * 2000)

        with pytest.raises(FileSizeError):
            validate_file_size(file_path, max_size_mb=0.001)  # 1KB limit

    def test_validate_file_size_not_found(self, tmp_path):
        """Missing files should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            validate_file_size(tmp_path / "nonexistent.json")

    def test_safe_mkdir_creates_directory(self, tmp_path):
        """safe_mkdir should create directory with correct permissions."""
        new_dir = tmp_path / "secure_dir"
        result = safe_mkdir(new_dir, mode=0o700)

        assert result.exists()
        assert result.is_dir()
        # Check permissions (on Unix)
        if os.name == "posix":
            assert (result.stat().st_mode & 0o777) == 0o700

    def test_safe_mkdir_nested(self, tmp_path):
        """safe_mkdir should create nested directories."""
        nested = tmp_path / "a" / "b" / "c"
        result = safe_mkdir(nested)

        assert result.exists()
        assert result.is_dir()

    def test_safe_write_json_atomic(self, tmp_path):
        """safe_write_json should write atomically."""
        file_path = tmp_path / "output.json"
        data = {"key": "value", "number": 42}

        safe_write_json(file_path, data)

        assert file_path.exists()
        with open(file_path) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_safe_write_json_permissions(self, tmp_path):
        """safe_write_json should set secure permissions."""
        file_path = tmp_path / "secure.json"
        safe_write_json(file_path, {"key": "value"}, mode=0o600)

        if os.name == "posix":
            assert (file_path.stat().st_mode & 0o777) == 0o600

    def test_safe_read_json_valid(self, tmp_path):
        """safe_read_json should read valid JSON files."""
        file_path = tmp_path / "test.json"
        data = {"key": "value", "list": [1, 2, 3]}
        with open(file_path, "w") as f:
            json.dump(data, f)

        loaded = safe_read_json(file_path)
        assert loaded == data

    def test_safe_read_json_size_limit(self, tmp_path):
        """safe_read_json should enforce size limits."""
        file_path = tmp_path / "large.json"
        file_path.write_text('{"data": "' + "x" * 10000 + '"}')

        with pytest.raises(FileSizeError):
            safe_read_json(file_path, max_size_mb=0.001)

    def test_safe_read_json_symlink_rejected(self, tmp_path):
        """safe_read_json should reject symlinks by default."""
        target = tmp_path / "target.json"
        target.write_text('{"key": "value"}')
        link = tmp_path / "link.json"
        link.symlink_to(target)

        with pytest.raises(SymlinkError):
            safe_read_json(link, allow_symlinks=False)


# =============================================================================
# Logging Sanitization Tests
# =============================================================================


class TestLoggingSanitization:
    """Tests for logging sanitization utilities."""

    def test_sanitize_for_logging_simple_values(self):
        """Simple values should pass through."""
        assert sanitize_for_logging(42) == 42
        assert sanitize_for_logging("hello") == "hello"
        assert sanitize_for_logging(True) is True
        assert sanitize_for_logging(None) is None

    def test_sanitize_for_logging_truncates_long_strings(self):
        """Long strings should be truncated."""
        long_string = "x" * 500
        result = sanitize_for_logging(long_string, max_string_length=100)
        assert len(result) < 150
        assert "truncated" in result

    def test_sanitize_for_logging_truncates_long_lists(self):
        """Long lists should be summarized."""
        long_list = list(range(100))
        result = sanitize_for_logging(long_list, max_list_items=5)
        assert result == "[100 items]"

    def test_sanitize_for_logging_redacts_sensitive_keys(self):
        """Sensitive keys should be redacted."""
        data = {
            "api_key": "secret123",
            "password": "hunter2",
            "username": "admin",
            "count": 42,
        }
        result = sanitize_for_logging(data)

        assert result["api_key"] == "[REDACTED]"
        assert result["password"] == "[REDACTED]"
        assert result["username"] != "[REDACTED]"  # Not a sensitive key

    def test_sanitize_for_logging_redacts_sensitive_patterns_in_values(self):
        """Sensitive patterns in values should be redacted."""
        data = "my api_key is abc123"
        result = sanitize_for_logging(data)
        assert result == "[REDACTED]"

    def test_sanitize_for_logging_nested_dicts(self):
        """Nested dicts should be sanitized recursively when no whitelist."""
        data = {
            "outer": {
                "api_key": "secret",
                "data": {"count": 5},
            }
        }
        # Pass empty set for safe_keys to enable full recursive sanitization
        result = sanitize_for_logging(data, safe_keys=set())
        assert result["outer"]["api_key"] == "[REDACTED]"


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestInputValidation:
    """Tests for input validation utilities."""

    def test_validate_date_format_valid(self):
        """Valid date formats should pass."""
        assert validate_date_format("2026-01-15") == "2026-01-15"
        assert validate_date_format("2000-12-31") == "2000-12-31"

    def test_validate_date_format_invalid(self):
        """Invalid date formats should raise ValueError."""
        with pytest.raises(ValueError):
            validate_date_format("01-15-2026")  # Wrong format

        with pytest.raises(ValueError):
            validate_date_format("2026/01/15")  # Wrong separator

        with pytest.raises(ValueError):
            validate_date_format("2026-13-01")  # Invalid month

        with pytest.raises(ValueError):
            validate_date_format("")  # Empty

    def test_validate_numeric_bounds_valid(self):
        """Values within bounds should pass."""
        result = validate_numeric_bounds(50, 0, 100, "score")
        assert result == Decimal("50")

        result = validate_numeric_bounds("75.5", 0, 100, "score")
        assert result == Decimal("75.5")

    def test_validate_numeric_bounds_out_of_range(self):
        """Values outside bounds should raise ValueError."""
        with pytest.raises(ValueError):
            validate_numeric_bounds(150, 0, 100, "score")

        with pytest.raises(ValueError):
            validate_numeric_bounds(-10, 0, 100, "score")

    def test_validate_numeric_bounds_none_handling(self):
        """None handling should work correctly."""
        result = validate_numeric_bounds(None, 0, 100, "score", allow_none=True)
        assert result is None

        with pytest.raises(ValueError):
            validate_numeric_bounds(None, 0, 100, "score", allow_none=False)

    def test_validate_numeric_bounds_special_values(self):
        """NaN and Inf should be rejected."""
        with pytest.raises(ValueError):
            validate_numeric_bounds(float("nan"), 0, 100, "score")

        with pytest.raises(ValueError):
            validate_numeric_bounds(float("inf"), 0, 100, "score")

    def test_validate_ticker_format_valid(self):
        """Valid tickers should pass."""
        assert validate_ticker_format("AAPL") == "AAPL"
        assert validate_ticker_format("a") == "A"  # Uppercase
        assert validate_ticker_format("msft") == "MSFT"

    def test_validate_ticker_format_invalid(self):
        """Invalid tickers should raise ValueError."""
        with pytest.raises(ValueError):
            validate_ticker_format("TOOLONG")  # > 5 chars

        with pytest.raises(ValueError):
            validate_ticker_format("AB123")  # Contains numbers

        with pytest.raises(ValueError):
            validate_ticker_format("")  # Empty

        with pytest.raises(ValueError):
            validate_ticker_format(None)  # None


# =============================================================================
# Resource Management Tests
# =============================================================================


class TestResourceManagement:
    """Tests for resource management utilities."""

    def test_check_available_memory_mb(self):
        """Memory check should return a value."""
        result = check_available_memory_mb()
        # Should return positive number or -1 if psutil not available
        assert result == -1 or result > 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestHardeningIntegration:
    """Integration tests for hardening utilities."""

    def test_full_checkpoint_workflow(self, tmp_path):
        """Full checkpoint save/load workflow with integrity."""
        checkpoint_dir = tmp_path / "checkpoints"
        module_name = "module_1"
        as_of_date = "2026-01-15"
        data = {
            "scores": [
                {"ticker": "AAPL", "score": "85.5"},
                {"ticker": "GOOGL", "score": "72.3"},
            ],
            "count": 2,
        }

        # Save checkpoint
        filepath = validate_checkpoint_path(checkpoint_dir, module_name, as_of_date)
        safe_mkdir(checkpoint_dir)

        checkpoint_data = {
            "module": module_name,
            "as_of_date": as_of_date,
            "data": data,
        }
        data_json = json.dumps(data, sort_keys=True)
        checkpoint_data["_content_hash"] = compute_content_hash(data_json)

        safe_write_json(filepath, checkpoint_data)

        # Load and verify checkpoint
        loaded = safe_read_json(filepath, base_dir=checkpoint_dir)
        assert loaded["module"] == module_name
        assert loaded["data"] == data

        # Verify integrity
        loaded_data_json = json.dumps(loaded["data"], sort_keys=True)
        computed_hash = compute_content_hash(loaded_data_json)
        assert computed_hash == loaded["_content_hash"]

    def test_secure_data_pipeline(self, tmp_path):
        """Test secure data loading and processing workflow."""
        # Create test data files
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        universe = [
            {"ticker": "AAPL", "status": "active"},
            {"ticker": "GOOGL", "status": "active"},
        ]
        with open(data_dir / "universe.json", "w") as f:
            json.dump(universe, f)

        # Load with security checks
        loaded = safe_read_json(
            data_dir / "universe.json",
            max_size_mb=10,
            base_dir=data_dir,
        )

        assert len(loaded) == 2
        assert loaded[0]["ticker"] == "AAPL"


# =============================================================================
# Safe Date Parsing Tests
# =============================================================================


class TestSafeDateParsing:
    """Tests for safe date parsing utilities."""

    def test_safe_parse_date_valid_iso(self):
        """Valid ISO date strings should parse correctly."""
        result = safe_parse_date("2026-01-15")
        assert result == date(2026, 1, 15)

    def test_safe_parse_date_with_whitespace(self):
        """Whitespace around date should be trimmed."""
        result = safe_parse_date("  2026-01-15  ")
        assert result == date(2026, 1, 15)

    def test_safe_parse_date_none_with_default(self):
        """None with default should return default."""
        default = date(2026, 1, 1)
        result = safe_parse_date(None, default=default)
        assert result == default

    def test_safe_parse_date_none_without_default(self):
        """None without default should raise DateParseError."""
        with pytest.raises(DateParseError) as exc_info:
            safe_parse_date(None, field_name="test_date")
        assert "test_date" in str(exc_info.value)

    def test_safe_parse_date_empty_string(self):
        """Empty string without default should raise DateParseError."""
        with pytest.raises(DateParseError):
            safe_parse_date("")

    def test_safe_parse_date_invalid_format(self):
        """Invalid date format should raise DateParseError."""
        with pytest.raises(DateParseError) as exc_info:
            safe_parse_date("not-a-date")
        assert "invalid date format" in str(exc_info.value).lower()

    def test_safe_parse_date_wrong_type(self):
        """Non-string type should raise DateParseError."""
        with pytest.raises(DateParseError):
            safe_parse_date(12345)

    def test_safe_parse_date_or_none_valid(self):
        """Valid date should parse correctly."""
        result = safe_parse_date_or_none("2026-01-15")
        assert result == date(2026, 1, 15)

    def test_safe_parse_date_or_none_invalid(self):
        """Invalid date should return None."""
        result = safe_parse_date_or_none("not-a-date")
        assert result is None

    def test_safe_parse_date_or_none_none_input(self):
        """None input should return None."""
        result = safe_parse_date_or_none(None)
        assert result is None


# =============================================================================
# Safe Nested Dict Access Tests
# =============================================================================


class TestSafeNestedDictAccess:
    """Tests for safe nested dict access utilities."""

    def test_safe_get_nested_simple(self):
        """Simple nested access should work."""
        data = {"a": {"b": {"c": 1}}}
        assert safe_get_nested(data, "a", "b", "c") == 1

    def test_safe_get_nested_missing_key(self):
        """Missing key should return default."""
        data = {"a": {"b": 1}}
        assert safe_get_nested(data, "a", "x", default=0) == 0

    def test_safe_get_nested_none_data(self):
        """None data should return default."""
        assert safe_get_nested(None, "a", "b", default="default") == "default"

    def test_safe_get_nested_non_dict_intermediate(self):
        """Non-dict intermediate should return default."""
        data = {"a": "not_a_dict"}
        assert safe_get_nested(data, "a", "b", default=0) == 0

    def test_safe_get_decimal_valid(self):
        """Valid decimal should parse correctly."""
        data = {"score": "85.50"}
        result = safe_get_decimal(data, "score")
        assert result == Decimal("85.50")

    def test_safe_get_decimal_integer(self):
        """Integer should convert to decimal."""
        data = {"score": 85}
        result = safe_get_decimal(data, "score")
        assert result == Decimal("85")

    def test_safe_get_decimal_with_bounds(self):
        """Bounds should be enforced."""
        data = {"score": "150"}
        result = safe_get_decimal(data, "score", max_val=Decimal("100"))
        assert result == Decimal("100")

        data = {"score": "-10"}
        result = safe_get_decimal(data, "score", min_val=Decimal("0"))
        assert result == Decimal("0")

    def test_safe_get_decimal_invalid(self):
        """Invalid value should return default."""
        data = {"score": "not_a_number"}
        result = safe_get_decimal(data, "score", default=Decimal("0"))
        assert result == Decimal("0")

    def test_safe_get_decimal_none_data(self):
        """None data should return default."""
        result = safe_get_decimal(None, "score", default=Decimal("50"))
        assert result == Decimal("50")

    def test_safe_get_int_valid(self):
        """Valid int should parse correctly."""
        data = {"count": "42"}
        result = safe_get_int(data, "count")
        assert result == 42

    def test_safe_get_int_with_bounds(self):
        """Bounds should be enforced."""
        data = {"count": 150}
        result = safe_get_int(data, "count", max_val=100)
        assert result == 100

    def test_safe_get_int_invalid(self):
        """Invalid value should return default."""
        data = {"count": "not_a_number"}
        result = safe_get_int(data, "count", default=0)
        assert result == 0

    def test_safe_get_str_valid(self):
        """Valid string should return correctly."""
        data = {"name": "test"}
        result = safe_get_str(data, "name")
        assert result == "test"

    def test_safe_get_str_truncates(self):
        """Long string should be truncated."""
        data = {"name": "a" * 100}
        result = safe_get_str(data, "name", max_length=10)
        assert result == "a" * 10

    def test_safe_get_str_converts_non_string(self):
        """Non-string should be converted."""
        data = {"value": 123}
        result = safe_get_str(data, "value")
        assert result == "123"


# =============================================================================
# Module Logger Tests
# =============================================================================


class TestModuleLogger:
    """Tests for module logging utilities."""

    def test_module_logger_creation(self):
        """Module logger should be created correctly."""
        logger = get_module_logger("test_module", run_id="abc123", as_of_date="2026-01-15")
        assert logger.module_name == "test_module"
        assert logger.run_id == "abc123"
        assert logger.as_of_date == "2026-01-15"

    def test_module_logger_format_context(self):
        """Context formatting should include all parts."""
        logger = ModuleLogger("test_module", run_id="abc123def", as_of_date="2026-01-15")
        context = logger._format_context()
        assert "[test_module]" in context
        assert "[run=abc123de]" in context  # First 8 chars of run_id
        assert "[date=2026-01-15]" in context

    def test_module_logger_timer(self):
        """Timer should track duration."""
        logger = get_module_logger("test_module")
        logger.start_timer("test_operation")
        time.sleep(0.1)
        duration = logger.end_timer("test_operation")
        assert duration >= 0.1

    def test_module_logger_timer_not_found(self):
        """Missing timer should return 0."""
        logger = get_module_logger("test_module")
        duration = logger.end_timer("nonexistent")
        assert duration == 0.0


# =============================================================================
# Decimal Safety Tests
# =============================================================================


class TestDecimalSafety:
    """Tests for decimal safety utilities."""

    def test_clamp_decimal_within_bounds(self):
        """Value within bounds should pass through."""
        result = clamp_decimal(Decimal("50"), Decimal("0"), Decimal("100"))
        assert result == Decimal("50")

    def test_clamp_decimal_above_max(self):
        """Value above max should be clamped."""
        result = clamp_decimal(Decimal("150"), Decimal("0"), Decimal("100"))
        assert result == Decimal("100")

    def test_clamp_decimal_below_min(self):
        """Value below min should be clamped."""
        result = clamp_decimal(Decimal("-10"), Decimal("0"), Decimal("100"))
        assert result == Decimal("0")

    def test_clamp_decimal_none(self):
        """None should return default."""
        result = clamp_decimal(None, Decimal("0"), Decimal("100"), default=Decimal("50"))
        assert result == Decimal("50")

    def test_clamp_decimal_non_finite(self):
        """Non-finite values should return default."""
        result = clamp_decimal(
            Decimal("inf"), Decimal("0"), Decimal("100"), default=Decimal("0")
        )
        assert result == Decimal("0")

    def test_safe_decimal_divide_normal(self):
        """Normal division should work."""
        result = safe_decimal_divide(Decimal("10"), Decimal("2"))
        assert result == Decimal("5")

    def test_safe_decimal_divide_by_zero(self):
        """Division by zero should return default."""
        result = safe_decimal_divide(Decimal("10"), Decimal("0"), default=Decimal("0"))
        assert result == Decimal("0")

    def test_safe_decimal_divide_none_inputs(self):
        """None inputs should return default."""
        result = safe_decimal_divide(None, Decimal("2"), default=Decimal("0"))
        assert result == Decimal("0")

        result = safe_decimal_divide(Decimal("10"), None, default=Decimal("0"))
        assert result == Decimal("0")

    def test_safe_decimal_divide_with_max(self):
        """Result exceeding max should be clamped."""
        result = safe_decimal_divide(
            Decimal("1000000"),
            Decimal("0.001"),
            max_result=Decimal("1000000")
        )
        assert result == Decimal("1000000")

    def test_validate_decimal_field_valid(self):
        """Valid decimal should pass validation."""
        result = validate_decimal_field("85.5", "score", bounds_type="score")
        assert result == Decimal("85.5")

    def test_validate_decimal_field_out_of_bounds(self):
        """Out of bounds should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            validate_decimal_field("150", "score", bounds_type="score")
        assert "above maximum" in str(exc_info.value)

    def test_validate_decimal_field_none_allowed(self):
        """None should be allowed when specified."""
        result = validate_decimal_field(None, "score", allow_none=True)
        assert result is None

    def test_validate_decimal_field_none_not_allowed(self):
        """None should raise when not allowed."""
        with pytest.raises(ValueError) as exc_info:
            validate_decimal_field(None, "score", allow_none=False)
        assert "required" in str(exc_info.value)


# =============================================================================
# File I/O Error Handling Tests
# =============================================================================


class TestFileIOErrorHandling:
    """Tests for file I/O error handling utilities."""

    def test_safe_mkdir_with_error_creates_dir(self, tmp_path):
        """safe_mkdir_with_error should create directory."""
        new_dir = tmp_path / "new_dir"
        result = safe_mkdir_with_error(new_dir)
        assert result.exists()
        assert result.is_dir()

    def test_safe_file_write_creates_file(self, tmp_path):
        """safe_file_write should create file atomically."""
        file_path = tmp_path / "test.txt"
        result = safe_file_write(file_path, "test content")
        assert result.exists()
        assert result.read_text() == "test content"

    def test_safe_file_read_reads_content(self, tmp_path):
        """safe_file_read should read file content."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("test content")
        content = safe_file_read(file_path)
        assert content == "test content"

    def test_safe_file_read_file_not_found(self, tmp_path):
        """safe_file_read should raise FileReadError for missing file."""
        with pytest.raises(FileReadError) as exc_info:
            safe_file_read(tmp_path / "nonexistent.txt")
        assert "not found" in str(exc_info.value).lower()

    def test_safe_json_write_and_read(self, tmp_path):
        """safe_json_write and safe_json_read should work together."""
        file_path = tmp_path / "data.json"
        data = {"key": "value", "number": 42}

        safe_json_write(file_path, data)
        loaded = safe_json_read(file_path)

        assert loaded == data

    def test_safe_json_read_invalid_json(self, tmp_path):
        """safe_json_read should raise FileReadError for invalid JSON."""
        file_path = tmp_path / "invalid.json"
        file_path.write_text("not valid json {")

        with pytest.raises(FileReadError) as exc_info:
            safe_json_read(file_path)
        assert "Invalid JSON" in str(exc_info.value)
