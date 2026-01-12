"""
Canonical JSON serialization for deterministic outputs.

Ensures byte-identical JSON across platforms and Python versions.

This module provides simple wrappers for canonical JSON serialization
that guarantee deterministic output for identical input data structures.

CCFT Compliance Features:
- Minimal separators (no whitespace) for machine outputs
- Sorted keys for deterministic key order
- ASCII-safe encoding for platform independence
- Pretty-printed version for human readability

Usage:
    from utils.json_canonical import to_canonical_json, to_canonical_json_pretty

    # For machine outputs (hashing, comparison):
    output = to_canonical_json(data)

    # For audit logs (human review):
    audit_log = to_canonical_json_pretty(data)

Author: Wake Robin Capital Management
Version: 1.0.0
"""

import json
from decimal import Decimal
from typing import Any


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Decimal):
            # Convert Decimal to string to preserve precision
            return str(obj)
        return super().default(obj)


def to_canonical_json(data: Any) -> str:
    """
    Serialize to canonical JSON format for deterministic hashing.

    Features:
    - Minimal separators (no whitespace)
    - Sorted keys
    - Consistent formatting
    - Handles Decimal types

    Returns byte-identical strings for identical data.

    Args:
        data: Any JSON-serializable data structure

    Returns:
        Canonical JSON string
    """
    return json.dumps(
        data,
        separators=(',', ':'),  # No whitespace
        sort_keys=True,         # Deterministic key order
        ensure_ascii=True,      # Platform-independent encoding
        cls=DecimalEncoder,     # Handle Decimal types
    )


def to_canonical_json_pretty(data: Any) -> str:
    """
    Pretty-printed canonical JSON for human readability (audit logs).

    Features:
    - 2-space indentation
    - Sorted keys
    - Consistent formatting
    - Handles Decimal types

    Args:
        data: Any JSON-serializable data structure

    Returns:
        Pretty-printed canonical JSON string
    """
    return json.dumps(
        data,
        indent=2,
        separators=(',', ': '),
        sort_keys=True,
        ensure_ascii=True,
        cls=DecimalEncoder,
    )


def hash_canonical_json(data: Any) -> str:
    """
    Generate SHA-256 hash of canonical JSON representation.

    Useful for verifying data integrity and determinism.

    Args:
        data: Any JSON-serializable data structure

    Returns:
        16-character hex hash prefix
    """
    import hashlib
    canonical = to_canonical_json(data)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# Example usage
if __name__ == "__main__":
    test_data = {
        "ticker": "XYZ",
        "score": Decimal("84.32"),
        "nested": {"b": 2, "a": 1},
        "items": [3, 1, 2],
    }

    print("Canonical JSON (compact):")
    print(to_canonical_json(test_data))

    print("\nCanonical JSON (pretty):")
    print(to_canonical_json_pretty(test_data))

    print(f"\nHash: {hash_canonical_json(test_data)}")
