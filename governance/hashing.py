"""
Cryptographic Hashing for Governance

Provides SHA256-based hashing for:
- File contents
- Raw bytes
- Canonical JSON objects

All hashes are returned as lowercase hex strings.
"""

import hashlib
from pathlib import Path
from typing import Any, Union

from governance.canonical_json import canonical_dumps


def hash_bytes(data: bytes) -> str:
    """
    Compute SHA256 hash of raw bytes.

    Args:
        data: Bytes to hash

    Returns:
        Lowercase hex digest (64 characters)
    """
    return hashlib.sha256(data).hexdigest()


def hash_file(path: Union[str, Path]) -> str:
    """
    Compute SHA256 hash of file contents.

    Args:
        path: Path to file

    Returns:
        Lowercase hex digest (64 characters)

    Raises:
        FileNotFoundError: If file does not exist
        PermissionError: If file cannot be read
    """
    path = Path(path)
    return hash_bytes(path.read_bytes())


def hash_canonical_json(obj: Any) -> str:
    """
    Compute SHA256 hash of canonical JSON representation.

    Args:
        obj: Object to serialize and hash

    Returns:
        Lowercase hex digest (64 characters)

    Raises:
        ValueError: If obj contains NaN or Inf
        TypeError: If obj contains non-serializable types
    """
    canonical = canonical_dumps(obj, indent=None)
    return hash_bytes(canonical.encode('utf-8'))


def hash_canonical_json_short(obj: Any, length: int = 16) -> str:
    """
    Compute truncated SHA256 hash of canonical JSON representation.

    Args:
        obj: Object to serialize and hash
        length: Number of hex characters to return (max 64)

    Returns:
        Truncated lowercase hex digest
    """
    full_hash = hash_canonical_json(obj)
    return full_hash[:length]


def verify_file_hash(path: Union[str, Path], expected_hash: str) -> bool:
    """
    Verify that a file's hash matches expected value.

    Args:
        path: Path to file
        expected_hash: Expected hex digest (full or prefix)

    Returns:
        True if hash matches (prefix match allowed)
    """
    actual = hash_file(path)
    # Allow prefix matching for truncated hashes
    return actual.startswith(expected_hash.lower())


def compute_input_hashes(paths: list) -> list:
    """
    Compute hashes for a list of input files.

    Args:
        paths: List of file paths (str or Path)

    Returns:
        Sorted list of {"path": basename, "sha256": hash}

    Raises:
        FileNotFoundError: If any file does not exist
    """
    results = []
    for path in paths:
        path = Path(path)
        results.append({
            "path": path.name,
            "sha256": hash_file(path),
        })
    # Sort by path for determinism
    return sorted(results, key=lambda x: x["path"])
