#!/usr/bin/env python3
"""
Tests for common/provenance.py

Tests provenance tracking and deterministic hashing.
Covers:
- Hash computation determinism
- Type serialization (Decimal, date, frozenset)
- Excluded fields handling
- Provenance record creation
"""

import pytest
from datetime import date, datetime
from decimal import Decimal

from common.provenance import (
    compute_hash,
    create_provenance,
    _json_serializer,
    HASH_EXCLUDED_FIELDS,
)


class TestJsonSerializer:
    """Tests for _json_serializer function."""

    def test_serialize_decimal(self):
        """Decimal should serialize to string."""
        result = _json_serializer(Decimal("123.45"))
        assert result == "123.45"
        assert isinstance(result, str)

    def test_serialize_decimal_high_precision(self):
        """High precision Decimal should preserve all digits."""
        result = _json_serializer(Decimal("123.456789012345"))
        assert result == "123.456789012345"

    def test_serialize_decimal_integer(self):
        """Integer Decimal should serialize without decimal point."""
        result = _json_serializer(Decimal("100"))
        assert result == "100"

    def test_serialize_date(self):
        """Date should serialize to ISO format."""
        result = _json_serializer(date(2026, 1, 15))
        assert result == "2026-01-15"

    def test_serialize_datetime(self):
        """Datetime should serialize to ISO format."""
        result = _json_serializer(datetime(2026, 1, 15, 10, 30, 0))
        assert result == "2026-01-15T10:30:00"

    def test_serialize_frozenset(self):
        """Frozenset should serialize to sorted list."""
        result = _json_serializer(frozenset(["c", "a", "b"]))
        assert result == ["a", "b", "c"]

    def test_serialize_frozenset_numbers(self):
        """Frozenset of numbers should serialize to sorted list."""
        result = _json_serializer(frozenset([3, 1, 2]))
        assert result == [1, 2, 3]

    def test_serialize_empty_frozenset(self):
        """Empty frozenset should serialize to empty list."""
        result = _json_serializer(frozenset())
        assert result == []

    def test_serialize_unsupported_type_raises(self):
        """Unsupported type should raise TypeError."""
        with pytest.raises(TypeError, match="not JSON serializable"):
            _json_serializer(object())

    def test_serialize_list_raises(self):
        """List is already serializable, should raise if passed directly."""
        with pytest.raises(TypeError):
            _json_serializer([1, 2, 3])


class TestComputeHash:
    """Tests for compute_hash function."""

    def test_hash_simple_dict(self):
        """Simple dict should hash deterministically."""
        data = {"key": "value", "number": 42}
        result = compute_hash(data)
        assert result.startswith("sha256:")
        assert len(result) == 71  # "sha256:" + 64 hex chars

    def test_hash_determinism(self):
        """Same data should produce same hash."""
        data = {"ticker": "ACME", "score": 75.5}
        hash1 = compute_hash(data)
        hash2 = compute_hash(data)
        assert hash1 == hash2

    def test_hash_key_order_independence(self):
        """Key order should not affect hash (sorted keys)."""
        data1 = {"a": 1, "b": 2, "c": 3}
        data2 = {"c": 3, "a": 1, "b": 2}
        assert compute_hash(data1) == compute_hash(data2)

    def test_hash_excludes_timestamp_fields(self):
        """Fields in HASH_EXCLUDED_FIELDS should not affect hash."""
        data_with_timestamp = {"ticker": "ACME", "timestamp": "2026-01-15T10:00:00"}
        data_without_timestamp = {"ticker": "ACME"}
        assert compute_hash(data_with_timestamp) == compute_hash(data_without_timestamp)

    def test_hash_excludes_loaded_at(self):
        """loaded_at field should be excluded from hash."""
        data1 = {"ticker": "ACME", "loaded_at": "2026-01-15T10:00:00"}
        data2 = {"ticker": "ACME", "loaded_at": "2026-01-15T12:00:00"}
        data3 = {"ticker": "ACME"}
        assert compute_hash(data1) == compute_hash(data2) == compute_hash(data3)

    def test_hash_excludes_generated_at(self):
        """generated_at field should be excluded from hash."""
        data1 = {"result": 100, "generated_at": "now"}
        data2 = {"result": 100}
        assert compute_hash(data1) == compute_hash(data2)

    def test_hash_excludes_runtime_ms(self):
        """runtime_ms field should be excluded from hash."""
        data1 = {"result": 100, "runtime_ms": 150}
        data2 = {"result": 100, "runtime_ms": 200}
        assert compute_hash(data1) == compute_hash(data2)

    def test_hash_with_decimal(self):
        """Decimal values should hash correctly."""
        data = {"score": Decimal("75.123")}
        result = compute_hash(data)
        assert result.startswith("sha256:")

    def test_hash_with_date(self):
        """Date values should hash correctly."""
        data = {"as_of_date": date(2026, 1, 15)}
        result = compute_hash(data)
        assert result.startswith("sha256:")

    def test_hash_with_frozenset(self):
        """Frozenset values should hash correctly."""
        data = {"tickers": frozenset(["ACME", "BETA"])}
        result = compute_hash(data)
        assert result.startswith("sha256:")

    def test_hash_nested_dict(self):
        """Nested dicts should hash correctly."""
        data = {
            "outer": {
                "inner": {"value": 42},
                "timestamp": "excluded"  # Should be excluded at any level
            }
        }
        result = compute_hash(data)
        assert result.startswith("sha256:")

    def test_hash_nested_excludes_timestamp(self):
        """Excluded fields in nested dicts should be excluded."""
        data1 = {"outer": {"value": 1, "timestamp": "a"}}
        data2 = {"outer": {"value": 1, "timestamp": "b"}}
        assert compute_hash(data1) == compute_hash(data2)

    def test_hash_different_values_different_hash(self):
        """Different data should produce different hashes."""
        data1 = {"value": 1}
        data2 = {"value": 2}
        assert compute_hash(data1) != compute_hash(data2)

    def test_hash_non_dict(self):
        """Non-dict data should hash correctly."""
        data = [1, 2, 3]
        result = compute_hash(data)
        assert result.startswith("sha256:")

    def test_hash_string(self):
        """String should hash correctly."""
        result = compute_hash("test string")
        assert result.startswith("sha256:")

    def test_hash_empty_dict(self):
        """Empty dict should hash correctly."""
        result = compute_hash({})
        assert result.startswith("sha256:")


class TestHashExcludedFields:
    """Tests for HASH_EXCLUDED_FIELDS constant."""

    def test_excluded_fields_contains_loaded_at(self):
        assert "loaded_at" in HASH_EXCLUDED_FIELDS

    def test_excluded_fields_contains_generated_at(self):
        assert "generated_at" in HASH_EXCLUDED_FIELDS

    def test_excluded_fields_contains_timestamp(self):
        assert "timestamp" in HASH_EXCLUDED_FIELDS

    def test_excluded_fields_contains_runtime_ms(self):
        assert "runtime_ms" in HASH_EXCLUDED_FIELDS

    def test_excluded_fields_is_frozenset(self):
        """Excluded fields should be immutable."""
        assert isinstance(HASH_EXCLUDED_FIELDS, frozenset)


class TestCreateProvenance:
    """Tests for create_provenance function."""

    def test_create_provenance_basic(self):
        """Basic provenance creation."""
        inputs = {"ticker": "ACME", "score": 75}
        result = create_provenance(
            ruleset_version="v1.0.0",
            inputs=inputs,
            pit_cutoff="2026-01-14",
        )

        assert result["ruleset_version"] == "v1.0.0"
        assert result["pit_cutoff"] == "2026-01-14"
        assert "inputs_hash" in result
        assert result["inputs_hash"].startswith("sha256:")

    def test_create_provenance_hash_determinism(self):
        """Provenance hash should be deterministic."""
        inputs = {"ticker": "ACME", "value": 100}

        prov1 = create_provenance("v1", inputs, "2026-01-14")
        prov2 = create_provenance("v1", inputs, "2026-01-14")

        assert prov1["inputs_hash"] == prov2["inputs_hash"]

    def test_create_provenance_different_inputs_different_hash(self):
        """Different inputs should produce different hash."""
        prov1 = create_provenance("v1", {"value": 1}, "2026-01-14")
        prov2 = create_provenance("v1", {"value": 2}, "2026-01-14")

        assert prov1["inputs_hash"] != prov2["inputs_hash"]

    def test_create_provenance_with_decimal_inputs(self):
        """Provenance should handle Decimal inputs."""
        inputs = {"score": Decimal("85.50")}
        result = create_provenance("v1", inputs, "2026-01-14")

        assert "inputs_hash" in result

    def test_create_provenance_with_date_inputs(self):
        """Provenance should handle date inputs."""
        inputs = {"as_of_date": date(2026, 1, 15)}
        result = create_provenance("v1", inputs, "2026-01-14")

        assert "inputs_hash" in result

    def test_create_provenance_contains_required_fields(self):
        """Provenance should contain all required fields."""
        result = create_provenance("v1", {}, "2026-01-14")

        assert "ruleset_version" in result
        assert "inputs_hash" in result
        assert "pit_cutoff" in result


class TestDeterminism:
    """Tests for overall deterministic behavior."""

    def test_hash_same_data_multiple_times(self):
        """Hashing same data multiple times should be deterministic."""
        data = {
            "ticker": "ACME",
            "scores": [Decimal("75.5"), Decimal("80.2")],
            "date": date(2026, 1, 15),
        }

        hashes = [compute_hash(data) for _ in range(100)]
        assert len(set(hashes)) == 1  # All hashes should be identical

    def test_provenance_reproducibility(self):
        """Creating provenance multiple times should be reproducible."""
        inputs = {"complex": {"nested": Decimal("123.45")}}

        provs = [create_provenance("v1", inputs, "2026-01-14") for _ in range(10)]
        hashes = [p["inputs_hash"] for p in provs]

        assert len(set(hashes)) == 1


class TestSpecialCases:
    """Tests for special/edge cases."""

    def test_hash_with_unicode(self):
        """Unicode characters should hash correctly."""
        data = {"name": "日本語テスト", "emoji": "🧬"}
        result = compute_hash(data)
        assert result.startswith("sha256:")

    def test_hash_with_negative_numbers(self):
        """Negative numbers should hash correctly."""
        data = {"value": -100, "decimal": Decimal("-50.5")}
        result = compute_hash(data)
        assert result.startswith("sha256:")

    def test_hash_with_zero(self):
        """Zero values should hash correctly."""
        data = {"value": 0, "decimal": Decimal("0")}
        result = compute_hash(data)
        assert result.startswith("sha256:")

    def test_hash_with_none(self):
        """None values should hash correctly."""
        data = {"value": None}
        result = compute_hash(data)
        assert result.startswith("sha256:")

    def test_hash_with_boolean(self):
        """Boolean values should hash correctly."""
        data = {"flag": True, "other": False}
        result = compute_hash(data)
        assert result.startswith("sha256:")
