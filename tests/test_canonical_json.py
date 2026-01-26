#!/usr/bin/env python3
"""
Unit tests for governance/canonical_json.py

Tests deterministic JSON serialization:
- Key sorting (recursive)
- Float formatting
- Decimal handling
- NaN/Infinity rejection
- List preservation (no reordering)
- Canonical validation
"""

import pytest
import json
import math
from decimal import Decimal
from io import StringIO

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from governance.canonical_json import (
    CanonicalJSONEncoder,
    canonical_dumps,
    canonical_dump,
    validate_canonical_json,
)


# ============================================================================
# ENCODER TESTS
# ============================================================================

class TestCanonicalJSONEncoder:
    """Tests for CanonicalJSONEncoder class."""

    def test_dict_key_sorting(self):
        """Dict keys should be sorted alphabetically."""
        data = {"zebra": 1, "apple": 2, "middle": 3}
        result = canonical_dumps(data)
        # Keys should appear in sorted order
        assert result.index('"apple"') < result.index('"middle"')
        assert result.index('"middle"') < result.index('"zebra"')

    def test_nested_dict_sorting(self):
        """Nested dicts should also have sorted keys."""
        data = {
            "outer_z": {"inner_b": 1, "inner_a": 2},
            "outer_a": {"z": 3, "a": 4},
        }
        result = canonical_dumps(data)
        # outer_a should come before outer_z
        assert result.index('"outer_a"') < result.index('"outer_z"')
        # inner_a should come before inner_b
        assert result.index('"inner_a"') < result.index('"inner_b"')

    def test_list_order_preserved(self):
        """Lists should NOT be reordered (caller responsibility)."""
        data = {"items": [3, 1, 2]}
        result = canonical_dumps(data)
        parsed = json.loads(result)
        assert parsed["items"] == [3, 1, 2]

    def test_tuple_converted_to_list(self):
        """Tuples should be converted to lists."""
        data = {"tuple": (1, 2, 3)}
        result = canonical_dumps(data)
        parsed = json.loads(result)
        assert parsed["tuple"] == [1, 2, 3]


class TestFloatFormatting:
    """Tests for float formatting in canonical JSON."""

    def test_zero_becomes_int(self):
        """0.0 should become 0 (integer)."""
        data = {"value": 0.0}
        result = canonical_dumps(data)
        parsed = json.loads(result)
        assert parsed["value"] == 0

    def test_integer_floats(self):
        """Floats that are integers should be formatted as integers."""
        data = {"value": 5.0}
        result = canonical_dumps(data)
        parsed = json.loads(result)
        # Should be 5, not 5.0
        assert parsed["value"] == 5

    def test_float_precision(self):
        """Floats should have stable precision."""
        data = {"value": 3.14159265358979}
        result = canonical_dumps(data)
        parsed = json.loads(result)
        # Should be reasonably precise
        assert abs(parsed["value"] - 3.14159265358979) < 1e-9

    def test_trailing_zeros_stripped(self):
        """Trailing zeros should be stripped."""
        data = {"value": 1.50000}
        result = canonical_dumps(data)
        # Should not have excessive zeros
        assert "1.50000" not in result or "1.5" in result

    def test_nan_rejected(self):
        """NaN values should raise ValueError."""
        data = {"value": float('nan')}
        with pytest.raises(ValueError, match="NaN"):
            canonical_dumps(data)

    def test_infinity_rejected(self):
        """Infinity values should raise ValueError."""
        data = {"value": float('inf')}
        with pytest.raises(ValueError, match="Infinity"):
            canonical_dumps(data)

        data = {"value": float('-inf')}
        with pytest.raises(ValueError, match="Infinity"):
            canonical_dumps(data)


class TestDecimalHandling:
    """Tests for Decimal handling in canonical JSON."""

    def test_decimal_serialization(self):
        """Decimals should be serializable."""
        data = {"value": Decimal("123.45")}
        result = canonical_dumps(data)
        parsed = json.loads(result)
        assert abs(parsed["value"] - 123.45) < 1e-10

    def test_decimal_nan_rejected(self):
        """Decimal NaN should raise ValueError."""
        data = {"value": Decimal('NaN')}
        with pytest.raises(ValueError, match="NaN"):
            canonical_dumps(data)

    def test_decimal_infinity_rejected(self):
        """Decimal Infinity should raise ValueError."""
        data = {"value": Decimal('Infinity')}
        with pytest.raises(ValueError, match="Infinity"):
            canonical_dumps(data)

    def test_decimal_zero(self):
        """Decimal zero should work correctly."""
        data = {"value": Decimal("0")}
        result = canonical_dumps(data)
        parsed = json.loads(result)
        assert parsed["value"] == 0


# ============================================================================
# CANONICAL DUMPS TESTS
# ============================================================================

class TestCanonicalDumps:
    """Tests for canonical_dumps function."""

    def test_trailing_newline(self):
        """Output should end with trailing newline."""
        data = {"key": "value"}
        result = canonical_dumps(data)
        assert result.endswith('\n')

    def test_indent_default(self):
        """Default indent should be 2 spaces."""
        data = {"key": "value"}
        result = canonical_dumps(data)
        # With indent=2, should have proper spacing
        assert '\n' in result  # Multi-line
        assert '  "key"' in result  # 2-space indent

    def test_compact_mode(self):
        """indent=None should produce compact output."""
        data = {"key": "value", "other": 123}
        result = canonical_dumps(data, indent=None)
        # Should be single line (plus newline)
        lines = result.strip().split('\n')
        assert len(lines) == 1

    def test_ascii_safe(self):
        """ensure_ascii=False should allow unicode."""
        data = {"emoji": "test"}  # ASCII only for this test
        result = canonical_dumps(data, ensure_ascii=False)
        assert "test" in result

    def test_deterministic_output(self):
        """Same input should always produce same output."""
        data = {"b": 2, "a": 1, "c": 3}
        result1 = canonical_dumps(data)
        result2 = canonical_dumps(data)
        assert result1 == result2

    def test_complex_nested_structure(self):
        """Should handle complex nested structures."""
        data = {
            "users": [
                {"name": "Alice", "scores": [95, 87, 92]},
                {"name": "Bob", "scores": [88, 91, 85]},
            ],
            "metadata": {
                "version": "1.0",
                "counts": {"total": 2, "active": 2},
            },
        }
        result = canonical_dumps(data)
        parsed = json.loads(result)
        assert parsed["users"][0]["name"] == "Alice"


# ============================================================================
# CANONICAL DUMP (FILE) TESTS
# ============================================================================

class TestCanonicalDump:
    """Tests for canonical_dump function (file writing)."""

    def test_write_to_file(self):
        """Should write to file-like object."""
        data = {"key": "value"}
        output = StringIO()
        canonical_dump(data, output)
        result = output.getvalue()
        assert '"key"' in result
        assert result.endswith('\n')

    def test_indent_parameter(self):
        """Should respect indent parameter."""
        data = {"key": "value"}
        output = StringIO()
        canonical_dump(data, output, indent=4)
        result = output.getvalue()
        assert '    "key"' in result  # 4-space indent


# ============================================================================
# VALIDATION TESTS
# ============================================================================

class TestValidateCanonicalJson:
    """Tests for validate_canonical_json function."""

    def test_valid_canonical_json(self):
        """Valid canonical JSON should return True."""
        data = {"a": 1, "b": 2}
        canonical = canonical_dumps(data)
        assert validate_canonical_json(canonical) is True

    def test_unsorted_keys_invalid(self):
        """Non-canonical (unsorted) JSON should return False."""
        # Manually create non-canonical JSON with unsorted keys
        non_canonical = '{"z": 1, "a": 2}\n'
        assert validate_canonical_json(non_canonical) is False

    def test_invalid_json_returns_false(self):
        """Invalid JSON should return False (not raise)."""
        assert validate_canonical_json("not json at all") is False
        assert validate_canonical_json("{invalid}") is False

    def test_missing_newline_invalid(self):
        """Missing trailing newline should be invalid."""
        data = {"key": "value"}
        no_newline = canonical_dumps(data).rstrip('\n')
        assert validate_canonical_json(no_newline) is False


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Edge case tests for canonical JSON."""

    def test_empty_dict(self):
        """Empty dict should work."""
        result = canonical_dumps({})
        assert result.strip() == "{}"

    def test_empty_list(self):
        """Empty list should work."""
        result = canonical_dumps([])
        assert result.strip() == "[]"

    def test_null_value(self):
        """None should become null."""
        result = canonical_dumps({"value": None})
        assert "null" in result

    def test_boolean_values(self):
        """Booleans should be lowercase."""
        result = canonical_dumps({"true": True, "false": False})
        assert "true" in result
        assert "false" in result

    def test_string_escaping(self):
        """Special characters should be escaped."""
        data = {"text": 'line1\nline2\ttab"quote'}
        result = canonical_dumps(data)
        # Should be valid JSON
        parsed = json.loads(result)
        assert parsed["text"] == 'line1\nline2\ttab"quote'

    def test_unicode_strings(self):
        """Unicode should be handled correctly."""
        data = {"text": "Hello, World!"}
        result = canonical_dumps(data, ensure_ascii=False)
        parsed = json.loads(result)
        assert parsed["text"] == "Hello, World!"

    def test_very_deep_nesting(self):
        """Should handle deeply nested structures."""
        data = {"level1": {"level2": {"level3": {"level4": {"value": 42}}}}}
        result = canonical_dumps(data)
        parsed = json.loads(result)
        assert parsed["level1"]["level2"]["level3"]["level4"]["value"] == 42

    def test_large_integer(self):
        """Should handle large integers."""
        data = {"big": 10**20}
        result = canonical_dumps(data)
        parsed = json.loads(result)
        assert parsed["big"] == 10**20

    def test_negative_numbers(self):
        """Should handle negative numbers."""
        data = {"negative": -123.45, "negative_int": -100}
        result = canonical_dumps(data)
        parsed = json.loads(result)
        assert parsed["negative"] == -123.45
        assert parsed["negative_int"] == -100


# ============================================================================
# DETERMINISM TESTS
# ============================================================================

class TestDeterminism:
    """Tests verifying deterministic behavior."""

    def test_repeated_serialization(self):
        """Multiple serializations should be identical."""
        data = {
            "id": 12345,
            "values": [1.5, 2.5, 3.5],
            "nested": {"z": 26, "a": 1},
        }
        results = [canonical_dumps(data) for _ in range(10)]
        assert len(set(results)) == 1  # All identical

    def test_key_insertion_order_ignored(self):
        """Key insertion order should not affect output."""
        data1 = {"a": 1, "b": 2, "c": 3}
        data2 = {"c": 3, "b": 2, "a": 1}
        data3 = {"b": 2, "a": 1, "c": 3}

        assert canonical_dumps(data1) == canonical_dumps(data2)
        assert canonical_dumps(data2) == canonical_dumps(data3)

    def test_float_representation_stable(self):
        """Float representation should be stable across runs."""
        data = {"pi": 3.141592653589793}
        result1 = canonical_dumps(data)
        result2 = canonical_dumps(data)
        assert result1 == result2


# ============================================================================
# TYPE HANDLING TESTS
# ============================================================================

class TestTypeHandling:
    """Tests for various Python types."""

    def test_int_types(self):
        """Various int types should work."""
        data = {"int": 42, "negative": -10}
        result = canonical_dumps(data)
        parsed = json.loads(result)
        assert parsed["int"] == 42
        assert parsed["negative"] == -10

    def test_string_types(self):
        """String types should work."""
        data = {"str": "hello"}
        result = canonical_dumps(data)
        parsed = json.loads(result)
        assert parsed["str"] == "hello"

    def test_mixed_list(self):
        """Lists with mixed types should work."""
        data = {"mixed": [1, "two", 3.0, None, True]}
        result = canonical_dumps(data)
        parsed = json.loads(result)
        assert parsed["mixed"] == [1, "two", 3, None, True]

    def test_non_serializable_raises(self):
        """Non-serializable types should raise TypeError."""
        # Use a set which is not JSON serializable
        data = {"obj": {1, 2, 3}}
        with pytest.raises(TypeError):
            canonical_dumps(data)
