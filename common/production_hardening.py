"""
Production hardening utilities for biotech-screener.

This module provides security and reliability utilities for production deployments:
- Path traversal protection
- Operation timeouts
- File integrity verification
- Secure file operations
- Input size limits
- Logging sanitization

Version: 1.0.0
"""

import hashlib
import json
import logging
import os
import signal
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, Union

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Maximum file sizes (bytes)
MAX_JSON_FILE_SIZE_MB = 100
MAX_CONFIG_FILE_SIZE_MB = 10
MAX_CHECKPOINT_FILE_SIZE_MB = 50

# Default operation timeouts (seconds)
DEFAULT_FILE_READ_TIMEOUT = 60
DEFAULT_MODULE_EXECUTION_TIMEOUT = 600  # 10 minutes
DEFAULT_PIPELINE_TIMEOUT = 3600  # 1 hour

# Logging sanitization
MAX_LOG_LIST_ITEMS = 10
MAX_LOG_STRING_LENGTH = 200
SAFE_LOG_KEYS = frozenset({
    'count', 'total', 'valid_count', 'invalid_count', 'events_detected',
    'tickers_analyzed', 'schema_version', 'score_version', 'as_of_date',
    'elapsed_ms', 'module_name', 'status', 'severity'
})

# Sensitive patterns that should never appear in logs
SENSITIVE_PATTERNS = frozenset({
    'api_key', 'password', 'secret', 'token', 'credential',
    'ssn', 'account_number', 'cusip'
})


# =============================================================================
# Exceptions
# =============================================================================

class SecurityError(Exception):
    """Base exception for security-related errors."""
    pass


class PathTraversalError(SecurityError):
    """Raised when path traversal attack is detected."""
    pass


class FileSizeError(SecurityError):
    """Raised when file exceeds size limits."""
    pass


class IntegrityError(SecurityError):
    """Raised when data integrity check fails."""
    pass


class OperationTimeoutError(Exception):
    """Raised when an operation exceeds its timeout."""
    pass


class SymlinkError(SecurityError):
    """Raised when unexpected symlink is encountered."""
    pass


# =============================================================================
# Path Traversal Protection
# =============================================================================

def validate_path_within_base(
    path: Union[str, Path],
    base_dir: Union[str, Path],
    allow_symlinks: bool = False
) -> Path:
    """
    Validate that a path is within the expected base directory.

    Prevents path traversal attacks (e.g., ../../../etc/passwd).

    Args:
        path: The path to validate
        base_dir: The base directory that path must be within
        allow_symlinks: Whether to allow symbolic links (default False)

    Returns:
        Resolved Path object

    Raises:
        PathTraversalError: If path escapes base_dir
        SymlinkError: If path is a symlink and allow_symlinks is False
    """
    path = Path(path)
    base_dir = Path(base_dir).resolve()

    # Resolve the path (handles .., ., etc.)
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError) as e:
        raise PathTraversalError(f"Cannot resolve path: {path}") from e

    # Check for symlinks
    if not allow_symlinks and path.exists() and path.is_symlink():
        raise SymlinkError(f"Symbolic links not allowed: {path}")

    # Verify path is within base_dir
    try:
        resolved.relative_to(base_dir)
    except ValueError:
        raise PathTraversalError(
            f"Path traversal detected: {path} escapes base directory {base_dir}"
        )

    return resolved


def safe_join_path(
    base_dir: Union[str, Path],
    *parts: str,
    allow_symlinks: bool = False
) -> Path:
    """
    Safely join path components with traversal protection.

    Args:
        base_dir: Base directory
        *parts: Path components to join
        allow_symlinks: Whether to allow symbolic links

    Returns:
        Safe joined path

    Raises:
        PathTraversalError: If resulting path escapes base_dir
    """
    base_dir = Path(base_dir).resolve()

    # Validate each part doesn't contain traversal
    for part in parts:
        if '..' in part or part.startswith('/'):
            raise PathTraversalError(f"Invalid path component: {part}")

    # Join and validate
    joined = base_dir.joinpath(*parts)
    return validate_path_within_base(joined, base_dir, allow_symlinks)


def validate_checkpoint_path(
    checkpoint_dir: Union[str, Path],
    module_name: str,
    as_of_date: str
) -> Path:
    """
    Validate and construct a checkpoint file path.

    Args:
        checkpoint_dir: Directory for checkpoints
        module_name: Name of the module
        as_of_date: Date string in YYYY-MM-DD format

    Returns:
        Validated checkpoint file path

    Raises:
        PathTraversalError: If path components are invalid
        ValueError: If date format is invalid
    """
    # Validate module_name (alphanumeric, underscore, hyphen only)
    if not module_name or not all(c.isalnum() or c in '_-' for c in module_name):
        raise PathTraversalError(f"Invalid module name: {module_name}")

    # Validate as_of_date format (YYYY-MM-DD)
    import re
    if not re.match(r'^\d{4}-\d{2}-\d{2}$', as_of_date):
        raise ValueError(f"Invalid date format: {as_of_date}")

    # Construct safe filename
    filename = f"{module_name}_{as_of_date}.json"

    return safe_join_path(checkpoint_dir, filename)


# =============================================================================
# Operation Timeouts
# =============================================================================

T = TypeVar('T')


class TimeoutHandler:
    """Handler for operation timeouts using SIGALRM."""

    _active_timeout: bool = False

    @classmethod
    def is_active(cls) -> bool:
        """Check if a timeout is currently active."""
        return cls._active_timeout


@contextmanager
def operation_timeout(
    seconds: int,
    description: str = "Operation"
):
    """
    Context manager for operation timeout.

    Uses SIGALRM on Unix systems. On Windows, this is a no-op (logs warning).

    Args:
        seconds: Maximum seconds for operation
        description: Description for error messages

    Yields:
        None

    Raises:
        OperationTimeoutError: If operation exceeds timeout

    Example:
        with operation_timeout(30, "File processing"):
            process_large_file()
    """
    if os.name != 'posix':
        logger.warning(f"operation_timeout not supported on {os.name}, proceeding without timeout")
        yield
        return

    def handler(signum, frame):
        raise OperationTimeoutError(
            f"{description} exceeded {seconds}s timeout"
        )

    # Store original handler
    original_handler = signal.signal(signal.SIGALRM, handler)
    TimeoutHandler._active_timeout = True

    try:
        signal.alarm(seconds)
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)
        TimeoutHandler._active_timeout = False


def with_timeout(
    seconds: int,
    description: str = "Operation"
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for adding timeout to functions.

    Args:
        seconds: Maximum seconds for operation
        description: Description for error messages

    Returns:
        Decorated function

    Example:
        @with_timeout(60, "Data loading")
        def load_data(path):
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            with operation_timeout(seconds, description):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# File Integrity Verification
# =============================================================================

def compute_file_hash(filepath: Union[str, Path], algorithm: str = 'sha256') -> str:
    """
    Compute cryptographic hash of a file.

    Args:
        filepath: Path to file
        algorithm: Hash algorithm (default sha256)

    Returns:
        Hash string in format "algorithm:hexdigest"
    """
    filepath = Path(filepath)
    hasher = hashlib.new(algorithm)

    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)

    return f"{algorithm}:{hasher.hexdigest()}"


def compute_content_hash(content: Union[str, bytes], algorithm: str = 'sha256') -> str:
    """
    Compute cryptographic hash of content.

    Args:
        content: String or bytes content
        algorithm: Hash algorithm

    Returns:
        Hash string
    """
    hasher = hashlib.new(algorithm)

    if isinstance(content, str):
        content = content.encode('utf-8')

    hasher.update(content)
    return f"{algorithm}:{hasher.hexdigest()}"


@dataclass
class IntegrityMetadata:
    """Metadata for integrity verification."""
    content_hash: str
    file_size: int
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            '_integrity': {
                'content_hash': self.content_hash,
                'file_size': self.file_size,
                'created_at': self.created_at
            }
        }


def save_with_integrity(
    filepath: Union[str, Path],
    data: Dict[str, Any],
    as_of_date: str
) -> IntegrityMetadata:
    """
    Save JSON data with integrity metadata embedded.

    Args:
        filepath: Output file path
        data: Data to save
        as_of_date: Date string for metadata

    Returns:
        IntegrityMetadata object
    """
    filepath = Path(filepath)

    # Serialize without integrity first to compute hash
    content = canonical_json_dumps(data)
    content_hash = compute_content_hash(content)

    # Create metadata
    metadata = IntegrityMetadata(
        content_hash=content_hash,
        file_size=len(content.encode('utf-8')),
        created_at=as_of_date
    )

    # Add integrity metadata to data
    data_with_integrity = {**data, **metadata.to_dict()}

    # Write atomically
    safe_write_json(filepath, data_with_integrity)

    return metadata


def verify_integrity(
    filepath: Union[str, Path],
    expected_hash: Optional[str] = None
) -> bool:
    """
    Verify integrity of a JSON file.

    Args:
        filepath: Path to file
        expected_hash: Expected hash (if None, uses embedded hash)

    Returns:
        True if integrity verified

    Raises:
        IntegrityError: If verification fails
    """
    filepath = Path(filepath)

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    data = json.loads(content)

    # Get expected hash
    if expected_hash is None:
        integrity = data.get('_integrity', {})
        expected_hash = integrity.get('content_hash')

        if not expected_hash:
            raise IntegrityError(f"No integrity metadata in {filepath}")

        # Remove integrity block and recompute
        data_without_integrity = {k: v for k, v in data.items() if k != '_integrity'}
        content_to_hash = canonical_json_dumps(data_without_integrity)
    else:
        content_to_hash = content

    # Compute actual hash
    actual_hash = compute_content_hash(content_to_hash)

    if actual_hash != expected_hash:
        raise IntegrityError(
            f"Integrity check failed for {filepath}: "
            f"expected {expected_hash}, got {actual_hash}"
        )

    return True


def load_with_integrity_check(
    filepath: Union[str, Path],
    verify: bool = True
) -> Dict[str, Any]:
    """
    Load JSON data and optionally verify integrity.

    Args:
        filepath: Path to file
        verify: Whether to verify integrity

    Returns:
        Loaded data (without integrity metadata)

    Raises:
        IntegrityError: If verification fails
    """
    filepath = Path(filepath)

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if verify and '_integrity' in data:
        verify_integrity(filepath)

    # Return data without integrity metadata
    return {k: v for k, v in data.items() if k != '_integrity'}


# =============================================================================
# Secure File Operations
# =============================================================================

def validate_file_size(
    filepath: Union[str, Path],
    max_size_mb: float = MAX_JSON_FILE_SIZE_MB
) -> int:
    """
    Validate that file size is within limits.

    Args:
        filepath: Path to file
        max_size_mb: Maximum size in megabytes

    Returns:
        File size in bytes

    Raises:
        FileSizeError: If file exceeds limit
        FileNotFoundError: If file doesn't exist
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    file_size = filepath.stat().st_size
    max_bytes = int(max_size_mb * 1024 * 1024)

    if file_size > max_bytes:
        raise FileSizeError(
            f"File too large: {filepath} is {file_size / 1024 / 1024:.1f}MB, "
            f"max is {max_size_mb}MB"
        )

    return file_size


def safe_mkdir(
    path: Union[str, Path],
    mode: int = 0o700
) -> Path:
    """
    Create directory with secure permissions.

    Args:
        path: Directory path
        mode: Permission mode (default 0o700 - owner only)

    Returns:
        Path object
    """
    path = Path(path)
    old_umask = os.umask(0o077)
    try:
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(mode)
    finally:
        os.umask(old_umask)
    return path


def safe_write_json(
    filepath: Union[str, Path],
    data: Any,
    mode: int = 0o600
) -> None:
    """
    Write JSON data atomically with secure permissions.

    Uses atomic write (temp file + rename) to prevent partial writes.

    Args:
        filepath: Output file path
        data: Data to serialize as JSON
        mode: File permission mode (default 0o600 - owner read/write only)
    """
    filepath = Path(filepath)

    # Ensure parent directory exists with secure permissions
    safe_mkdir(filepath.parent, mode=0o700)

    # Write to temporary file first
    fd, tmp_path = tempfile.mkstemp(
        dir=filepath.parent,
        prefix='.tmp_',
        suffix='.json'
    )

    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, sort_keys=True, default=json_serializer)

        # Set permissions before rename
        os.chmod(tmp_path, mode)

        # Atomic rename
        Path(tmp_path).replace(filepath)

    except Exception:
        # Clean up temp file on error
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def safe_read_json(
    filepath: Union[str, Path],
    max_size_mb: float = MAX_JSON_FILE_SIZE_MB,
    allow_symlinks: bool = False,
    base_dir: Optional[Union[str, Path]] = None
) -> Any:
    """
    Safely read JSON file with security checks.

    Args:
        filepath: Path to file
        max_size_mb: Maximum file size in MB
        allow_symlinks: Whether to allow symlinks
        base_dir: If provided, validate path is within this directory

    Returns:
        Parsed JSON data

    Raises:
        FileSizeError: If file too large
        PathTraversalError: If path escapes base_dir
        SymlinkError: If symlink and not allowed
    """
    filepath = Path(filepath)

    # Path traversal check
    if base_dir:
        filepath = validate_path_within_base(filepath, base_dir, allow_symlinks)

    # Symlink check
    if not allow_symlinks and filepath.is_symlink():
        raise SymlinkError(f"Symbolic links not allowed: {filepath}")

    # Size check
    validate_file_size(filepath, max_size_mb)

    # Read with encoding validation
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except UnicodeDecodeError as e:
        raise ValueError(f"File is not valid UTF-8: {filepath}") from e


# =============================================================================
# Logging Sanitization
# =============================================================================

def sanitize_for_logging(
    data: Any,
    max_list_items: int = MAX_LOG_LIST_ITEMS,
    max_string_length: int = MAX_LOG_STRING_LENGTH,
    safe_keys: Optional[Set[str]] = None
) -> Any:
    """
    Sanitize data for safe logging.

    Removes sensitive information and truncates large structures.

    Args:
        data: Data to sanitize
        max_list_items: Maximum list items to include
        max_string_length: Maximum string length
        safe_keys: Set of keys that are safe to log (optional whitelist)

    Returns:
        Sanitized data safe for logging
    """
    if safe_keys is None:
        safe_keys = SAFE_LOG_KEYS

    if data is None:
        return None

    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            # Skip keys that look sensitive
            key_lower = key.lower()
            if any(pattern in key_lower for pattern in SENSITIVE_PATTERNS):
                result[key] = "[REDACTED]"
            elif safe_keys and key not in safe_keys:
                # If whitelist provided and key not in it, summarize
                result[key] = _summarize_value(value)
            else:
                result[key] = sanitize_for_logging(
                    value, max_list_items, max_string_length, safe_keys
                )
        return result

    if isinstance(data, (list, tuple)):
        if len(data) > max_list_items:
            return f"[{len(data)} items]"
        return [sanitize_for_logging(item, max_list_items, max_string_length, safe_keys)
                for item in data]

    if isinstance(data, str):
        # Check for sensitive patterns
        if any(pattern in data.lower() for pattern in SENSITIVE_PATTERNS):
            return "[REDACTED]"
        if len(data) > max_string_length:
            return f"{data[:max_string_length]}... (truncated)"
        return data

    if isinstance(data, (int, float, bool, Decimal)):
        return data

    # For other types, convert to string and truncate
    str_repr = str(data)
    if len(str_repr) > max_string_length:
        return f"{str_repr[:max_string_length]}... (truncated)"
    return str_repr


def _summarize_value(value: Any) -> str:
    """Create a summary of a value for logging."""
    if value is None:
        return "null"
    if isinstance(value, dict):
        return f"{{dict with {len(value)} keys}}"
    if isinstance(value, (list, tuple)):
        return f"[{len(value)} items]"
    if isinstance(value, str):
        return f"string({len(value)} chars)"
    if isinstance(value, (int, float, Decimal)):
        return f"number"
    return f"{type(value).__name__}"


class SanitizedLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that automatically sanitizes logged data.

    Example:
        logger = SanitizedLoggerAdapter(logging.getLogger(__name__))
        logger.info("Results: %s", large_dict)  # Automatically sanitized
    """

    def process(self, msg, kwargs):
        # Sanitize any args
        if 'args' in kwargs:
            kwargs['args'] = tuple(
                sanitize_for_logging(arg) if isinstance(arg, (dict, list)) else arg
                for arg in kwargs['args']
            )
        return msg, kwargs


# =============================================================================
# JSON Serialization Helpers
# =============================================================================

def json_serializer(obj: Any) -> Any:
    """
    JSON serializer for objects not serializable by default.

    Handles: date, Decimal, Path, sets, bytes
    """
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(list(obj))
    if isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def canonical_json_dumps(data: Any) -> str:
    """
    Serialize data to canonical JSON (sorted keys, deterministic).

    Args:
        data: Data to serialize

    Returns:
        JSON string with sorted keys
    """
    return json.dumps(
        data,
        sort_keys=True,
        separators=(',', ':'),
        default=json_serializer
    )


# =============================================================================
# Input Validation Helpers
# =============================================================================

class NumericBounds(Enum):
    """Standard numeric bounds for validation."""
    MARKET_CAP_MIN = Decimal("0")
    MARKET_CAP_MAX = Decimal("1000000000000")  # 1 trillion
    SCORE_MIN = Decimal("0")
    SCORE_MAX = Decimal("100")
    PERCENTAGE_MIN = Decimal("-100")
    PERCENTAGE_MAX = Decimal("100")
    RUNWAY_MIN = Decimal("0")
    RUNWAY_MAX = Decimal("1200")  # 100 years in months


def validate_numeric_bounds(
    value: Union[int, float, Decimal, str],
    min_val: Union[int, float, Decimal],
    max_val: Union[int, float, Decimal],
    field_name: str,
    allow_none: bool = False
) -> Optional[Decimal]:
    """
    Validate that a numeric value is within bounds.

    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        field_name: Name for error messages
        allow_none: Whether None is allowed

    Returns:
        Decimal value if valid

    Raises:
        ValueError: If value is out of bounds or invalid
    """
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{field_name} is required but was None")

    # Convert to Decimal
    try:
        if isinstance(value, str):
            dec_value = Decimal(value)
        elif isinstance(value, float):
            dec_value = Decimal(str(value))
        elif isinstance(value, Decimal):
            dec_value = value
        else:
            dec_value = Decimal(str(value))
    except Exception as e:
        raise ValueError(f"{field_name} cannot be converted to Decimal: {value}") from e

    # Check for special values
    if not dec_value.is_finite():
        raise ValueError(f"{field_name} must be finite, got: {value}")

    # Check bounds
    min_dec = Decimal(str(min_val))
    max_dec = Decimal(str(max_val))

    if dec_value < min_dec or dec_value > max_dec:
        raise ValueError(
            f"{field_name} out of bounds: {value} not in [{min_val}, {max_val}]"
        )

    return dec_value


def validate_date_format(date_str: str, field_name: str = "date") -> str:
    """
    Validate date string format (YYYY-MM-DD).

    Args:
        date_str: Date string to validate
        field_name: Name for error messages

    Returns:
        Validated date string

    Raises:
        ValueError: If format is invalid
    """
    import re

    if not date_str:
        raise ValueError(f"{field_name} is required")

    if not isinstance(date_str, str):
        raise ValueError(f"{field_name} must be a string, got {type(date_str).__name__}")

    if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        raise ValueError(f"{field_name} must be YYYY-MM-DD format, got: {date_str}")

    # Validate it's a real date
    try:
        year, month, day = map(int, date_str.split('-'))
        date(year, month, day)
    except ValueError as e:
        raise ValueError(f"{field_name} is not a valid date: {date_str}") from e

    return date_str


def validate_ticker_format(ticker: str) -> str:
    """
    Validate ticker symbol format.

    Args:
        ticker: Ticker to validate

    Returns:
        Uppercase ticker

    Raises:
        ValueError: If format is invalid
    """
    if not ticker:
        raise ValueError("Ticker is required")

    if not isinstance(ticker, str):
        raise ValueError(f"Ticker must be a string, got {type(ticker).__name__}")

    ticker = ticker.upper().strip()

    if len(ticker) > 5:
        raise ValueError(f"Ticker too long (max 5 chars): {ticker}")

    if not ticker.isalpha():
        raise ValueError(f"Ticker must be alphabetic: {ticker}")

    return ticker


# =============================================================================
# Resource Management
# =============================================================================

def check_available_memory_mb() -> float:
    """
    Check available system memory in MB.

    Returns:
        Available memory in MB, or -1 if cannot determine
    """
    try:
        import psutil
        return psutil.virtual_memory().available / 1024 / 1024
    except ImportError:
        # psutil not available
        return -1
    except Exception:
        return -1


def require_minimum_memory(min_mb: float = 500) -> None:
    """
    Require minimum available memory before proceeding.

    Args:
        min_mb: Minimum required memory in MB

    Raises:
        RuntimeError: If insufficient memory
    """
    available = check_available_memory_mb()

    if available < 0:
        logger.warning("Cannot determine available memory, proceeding anyway")
        return

    if available < min_mb:
        raise RuntimeError(
            f"Insufficient memory: {available:.0f}MB available, "
            f"{min_mb:.0f}MB required"
        )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Exceptions
    'SecurityError',
    'PathTraversalError',
    'FileSizeError',
    'IntegrityError',
    'OperationTimeoutError',
    'SymlinkError',

    # Path validation
    'validate_path_within_base',
    'safe_join_path',
    'validate_checkpoint_path',

    # Timeouts
    'operation_timeout',
    'with_timeout',
    'TimeoutHandler',

    # Integrity
    'compute_file_hash',
    'compute_content_hash',
    'IntegrityMetadata',
    'save_with_integrity',
    'verify_integrity',
    'load_with_integrity_check',

    # File operations
    'validate_file_size',
    'safe_mkdir',
    'safe_write_json',
    'safe_read_json',

    # Logging
    'sanitize_for_logging',
    'SanitizedLoggerAdapter',

    # JSON
    'json_serializer',
    'canonical_json_dumps',

    # Validation
    'NumericBounds',
    'validate_numeric_bounds',
    'validate_date_format',
    'validate_ticker_format',

    # Resources
    'check_available_memory_mb',
    'require_minimum_memory',

    # Constants
    'MAX_JSON_FILE_SIZE_MB',
    'MAX_CONFIG_FILE_SIZE_MB',
    'MAX_CHECKPOINT_FILE_SIZE_MB',
    'DEFAULT_FILE_READ_TIMEOUT',
    'DEFAULT_MODULE_EXECUTION_TIMEOUT',
    'DEFAULT_PIPELINE_TIMEOUT',
]
