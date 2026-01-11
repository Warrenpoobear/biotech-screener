"""
Canonical JSON Serialization

Produces byte-identical JSON output for identical input data structures.

Rules:
1. All dict keys sorted recursively
2. Floats converted to stable string representation (no scientific notation for normal ranges)
3. NaN and Inf are forbidden (raise ValueError)
4. Lists are NOT reordered (caller must sort semantic lists)
5. Output ends with trailing newline
6. No whitespace except for indent
7. ASCII-safe output (no unicode escapes needed for standard chars)
"""

import json
import math
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, IO, Optional


class CanonicalJSONEncoder(json.JSONEncoder):
    """
    JSON encoder that produces deterministic, canonical output.

    - Sorts dict keys
    - Formats floats with fixed precision (no scientific notation)
    - Rejects NaN/Inf
    - Handles Decimal
    """

    # Maximum decimal places for float formatting
    FLOAT_PRECISION = 10

    def encode(self, o: Any) -> str:
        """Override to ensure top-level sorting."""
        return super().encode(self._canonicalize(o))

    def _canonicalize(self, obj: Any) -> Any:
        """Recursively canonicalize data structures."""
        if isinstance(obj, dict):
            # Sort keys and recurse
            return {k: self._canonicalize(v) for k, v in sorted(obj.items())}
        elif isinstance(obj, list):
            # Do NOT sort lists - caller must handle semantic ordering
            return [self._canonicalize(item) for item in obj]
        elif isinstance(obj, tuple):
            # Convert tuples to lists
            return [self._canonicalize(item) for item in obj]
        elif isinstance(obj, float):
            return self._format_float(obj)
        elif isinstance(obj, Decimal):
            return self._format_decimal(obj)
        elif isinstance(obj, (int, str, bool, type(None))):
            return obj
        else:
            # Let default encoder handle or raise
            return obj

    def _format_float(self, value: float) -> Any:
        """Format float with stable representation."""
        # Reject NaN and Inf
        if math.isnan(value):
            raise ValueError("NaN values are not allowed in canonical JSON")
        if math.isinf(value):
            raise ValueError("Infinity values are not allowed in canonical JSON")

        # For zero, return int 0
        if value == 0.0:
            return 0

        # Check if it's actually an integer
        if value == int(value) and abs(value) < 2**53:
            return int(value)

        # Format with fixed precision, strip trailing zeros
        formatted = f"{value:.{self.FLOAT_PRECISION}f}"

        # Strip trailing zeros but keep at least one decimal place
        if '.' in formatted:
            formatted = formatted.rstrip('0')
            if formatted.endswith('.'):
                formatted += '0'

        # Return as float for JSON encoding
        return float(formatted)

    def _format_decimal(self, value: Decimal) -> Any:
        """Format Decimal with stable representation."""
        # Check for special values
        if value.is_nan():
            raise ValueError("NaN values are not allowed in canonical JSON")
        if value.is_infinite():
            raise ValueError("Infinity values are not allowed in canonical JSON")

        # Convert to float for formatting
        float_val = float(value)
        return self._format_float(float_val)

    def default(self, o: Any) -> Any:
        """Handle types not natively supported by JSON."""
        if isinstance(o, Decimal):
            return self._format_decimal(o)
        elif hasattr(o, '__dict__'):
            # Convert objects to dicts
            return self._canonicalize(o.__dict__)
        else:
            raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def canonical_dumps(
    obj: Any,
    indent: Optional[int] = 2,
    ensure_ascii: bool = False,
) -> str:
    """
    Serialize object to canonical JSON string.

    Args:
        obj: Object to serialize
        indent: Indentation level (None for compact)
        ensure_ascii: If True, escape non-ASCII characters

    Returns:
        Canonical JSON string with trailing newline

    Raises:
        ValueError: If obj contains NaN or Inf
        TypeError: If obj contains non-serializable types
    """
    result = json.dumps(
        obj,
        cls=CanonicalJSONEncoder,
        indent=indent,
        sort_keys=True,  # Belt and suspenders with encoder
        ensure_ascii=ensure_ascii,
        separators=(',', ': ') if indent else (',', ':'),
    )
    return result + '\n'


def canonical_dump(
    obj: Any,
    fp: IO[str],
    indent: Optional[int] = 2,
    ensure_ascii: bool = False,
) -> None:
    """
    Serialize object to canonical JSON and write to file.

    Args:
        obj: Object to serialize
        fp: File-like object to write to
        indent: Indentation level (None for compact)
        ensure_ascii: If True, escape non-ASCII characters
    """
    fp.write(canonical_dumps(obj, indent=indent, ensure_ascii=ensure_ascii))


def validate_canonical_json(json_str: str) -> bool:
    """
    Validate that a JSON string is in canonical form.

    Args:
        json_str: JSON string to validate

    Returns:
        True if canonical, False otherwise
    """
    try:
        obj = json.loads(json_str.rstrip('\n'))
        canonical = canonical_dumps(obj)
        return json_str == canonical
    except (json.JSONDecodeError, ValueError):
        return False
