#!/usr/bin/env python3
"""
Tests for governance/mapping_loader.py

Adapter mapping loader for field mappings between source and canonical schemas.
Tests cover:
- get_mapping_path (path resolution)
- load_mapping (loading and hashing)
- compute_mapping_hash (deterministic hashing)
- save_mapping (saving with canonical JSON)
- validate_source_schema (schema validation)
- apply_field_mapping (field transformation)
- get_mapping_metadata (metadata extraction)
- MappingLoadError and SchemaMismatchError
"""

import json
import pytest
import tempfile
from pathlib import Path

from governance.mapping_loader import (
    get_mapping_path,
    load_mapping,
    compute_mapping_hash,
    save_mapping,
    validate_source_schema,
    apply_field_mapping,
    get_mapping_metadata,
    MappingLoadError,
    SchemaMismatchError,
    DEFAULT_ADAPTERS_DIR,
)


class TestGetMappingPath:
    """Tests for get_mapping_path function."""

    def test_returns_path(self):
        """Should return a Path object."""
        result = get_mapping_path("test_source")
        assert isinstance(result, Path)

    def test_default_version(self):
        """Should use v1 as default version."""
        result = get_mapping_path("test_source")
        assert "mapping_v1.json" in str(result)

    def test_custom_version(self):
        """Should use specified version."""
        result = get_mapping_path("test_source", mapping_version="v2")
        assert "mapping_v2.json" in str(result)

    def test_includes_source_name(self):
        """Should include source name in path."""
        result = get_mapping_path("my_source")
        assert "my_source" in str(result)

    def test_custom_adapters_dir(self):
        """Should use custom adapters directory."""
        result = get_mapping_path("test", adapters_dir="/custom/path")
        assert str(result).startswith("/custom/path")


class TestComputeMappingHash:
    """Tests for compute_mapping_hash function."""

    def test_returns_string(self):
        """Should return a string."""
        result = compute_mapping_hash({"key": "value"})
        assert isinstance(result, str)

    def test_default_length(self):
        """Should return 16 characters by default."""
        result = compute_mapping_hash({"key": "value"})
        assert len(result) == 16

    def test_custom_length(self):
        """Should use custom length."""
        result = compute_mapping_hash({"key": "value"}, length=8)
        assert len(result) == 8

    def test_deterministic(self):
        """Same input should produce same hash."""
        mapping = {"a": 1, "b": 2}
        hash1 = compute_mapping_hash(mapping)
        hash2 = compute_mapping_hash(mapping)
        assert hash1 == hash2

    def test_key_order_independent(self):
        """Hash should be same regardless of key order."""
        mapping1 = {"a": 1, "b": 2}
        mapping2 = {"b": 2, "a": 1}
        assert compute_mapping_hash(mapping1) == compute_mapping_hash(mapping2)

    def test_different_mappings_different_hash(self):
        """Different mappings should produce different hashes."""
        hash1 = compute_mapping_hash({"a": 1})
        hash2 = compute_mapping_hash({"a": 2})
        assert hash1 != hash2


class TestLoadMapping:
    """Tests for load_mapping function."""

    @pytest.fixture
    def temp_adapters(self):
        """Create temporary adapters directory with test mapping."""
        with tempfile.TemporaryDirectory() as temp_dir:
            adapters_dir = Path(temp_dir)
            source_dir = adapters_dir / "test_source"
            source_dir.mkdir()

            mapping = {
                "required_fields": ["ticker", "date"],
                "field_mappings": {
                    "sym": "ticker",
                    "dt": "date",
                },
            }

            mapping_file = source_dir / "mapping_v1.json"
            with open(mapping_file, "w") as f:
                json.dump(mapping, f)

            yield adapters_dir

    def test_loads_mapping(self, temp_adapters):
        """Should load mapping from file."""
        mapping, hash_value = load_mapping("test_source", adapters_dir=temp_adapters)

        assert "required_fields" in mapping
        assert "field_mappings" in mapping
        assert mapping["field_mappings"]["sym"] == "ticker"

    def test_returns_hash(self, temp_adapters):
        """Should return mapping hash."""
        mapping, hash_value = load_mapping("test_source", adapters_dir=temp_adapters)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 16

    def test_raises_on_missing_file(self, temp_adapters):
        """Should raise MappingLoadError for missing file."""
        with pytest.raises(MappingLoadError) as exc_info:
            load_mapping("nonexistent_source", adapters_dir=temp_adapters)
        assert "not found" in str(exc_info.value)

    def test_raises_on_invalid_json(self, temp_adapters):
        """Should raise MappingLoadError for invalid JSON."""
        source_dir = temp_adapters / "bad_source"
        source_dir.mkdir()
        mapping_file = source_dir / "mapping_v1.json"
        with open(mapping_file, "w") as f:
            f.write("not valid json {")

        with pytest.raises(MappingLoadError) as exc_info:
            load_mapping("bad_source", adapters_dir=temp_adapters)
        assert "Invalid JSON" in str(exc_info.value)

    def test_raises_on_non_dict(self, temp_adapters):
        """Should raise MappingLoadError if mapping is not a dict."""
        source_dir = temp_adapters / "array_source"
        source_dir.mkdir()
        mapping_file = source_dir / "mapping_v1.json"
        with open(mapping_file, "w") as f:
            json.dump(["list", "not", "dict"], f)

        with pytest.raises(MappingLoadError) as exc_info:
            load_mapping("array_source", adapters_dir=temp_adapters)
        assert "must be a JSON object" in str(exc_info.value)


class TestSaveMapping:
    """Tests for save_mapping function."""

    @pytest.fixture
    def temp_adapters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_saves_mapping(self, temp_adapters):
        """Should save mapping to file."""
        mapping = {"field_mappings": {"a": "b"}}

        path, hash_value = save_mapping(
            mapping, "new_source", adapters_dir=temp_adapters
        )

        assert path.exists()
        with open(path) as f:
            saved = json.load(f)
        assert saved["field_mappings"]["a"] == "b"

    def test_creates_source_dir(self, temp_adapters):
        """Should create source directory."""
        mapping = {"field_mappings": {}}

        path, _ = save_mapping(
            mapping, "new_source", adapters_dir=temp_adapters
        )

        assert (temp_adapters / "new_source").exists()

    def test_returns_path_and_hash(self, temp_adapters):
        """Should return path and hash."""
        mapping = {"field_mappings": {}}

        path, hash_value = save_mapping(
            mapping, "test", adapters_dir=temp_adapters
        )

        assert isinstance(path, Path)
        assert isinstance(hash_value, str)


class TestValidateSourceSchema:
    """Tests for validate_source_schema function."""

    def test_empty_required_fields(self):
        """Should pass if no required fields."""
        mapping = {"required_fields": []}
        is_valid, missing = validate_source_schema(
            {"any": "field"}, mapping, "test"
        )
        assert is_valid is True
        assert missing == []

    def test_all_fields_present(self):
        """Should pass if all required fields present."""
        mapping = {"required_fields": ["ticker", "date"]}
        data = {"ticker": "AMGN", "date": "2024-01-01", "extra": "ok"}

        is_valid, missing = validate_source_schema(data, mapping, "test")
        assert is_valid is True
        assert missing == []

    def test_raises_on_missing_fields(self):
        """Should raise SchemaMismatchError for missing fields."""
        mapping = {"required_fields": ["ticker", "date", "score"]}
        data = {"ticker": "AMGN"}  # Missing date and score

        with pytest.raises(SchemaMismatchError) as exc_info:
            validate_source_schema(data, mapping, "test_source")

        assert "date" in exc_info.value.missing_fields
        assert "score" in exc_info.value.missing_fields
        assert exc_info.value.source == "test_source"

    def test_validates_list_of_dicts(self):
        """Should validate first item in list of dicts."""
        mapping = {"required_fields": ["ticker"]}
        data = [
            {"ticker": "AMGN"},
            {"ticker": "GILD"},
        ]

        is_valid, missing = validate_source_schema(data, mapping, "test")
        assert is_valid is True

    def test_empty_list_passes(self):
        """Should pass for empty list."""
        mapping = {"required_fields": ["ticker"]}
        data = []

        is_valid, missing = validate_source_schema(data, mapping, "test")
        assert is_valid is True

    def test_raises_on_non_dict_sample(self):
        """Should raise if sample is not a dict."""
        mapping = {"required_fields": ["ticker"]}
        data = ["string", "list"]

        with pytest.raises(SchemaMismatchError):
            validate_source_schema(data, mapping, "test")


class TestApplyFieldMapping:
    """Tests for apply_field_mapping function."""

    def test_renames_fields(self):
        """Should rename fields according to mapping."""
        mapping = {
            "field_mappings": {
                "sym": "ticker",
                "dt": "date",
            }
        }
        record = {"sym": "AMGN", "dt": "2024-01-01"}

        result = apply_field_mapping(record, mapping)

        assert result["ticker"] == "AMGN"
        assert result["date"] == "2024-01-01"
        assert "sym" not in result
        assert "dt" not in result

    def test_preserves_unmapped_fields(self):
        """Should preserve fields not in mapping."""
        mapping = {"field_mappings": {"sym": "ticker"}}
        record = {"sym": "AMGN", "extra_field": "preserved"}

        result = apply_field_mapping(record, mapping)

        assert result["ticker"] == "AMGN"
        assert result["extra_field"] == "preserved"

    def test_empty_mapping(self):
        """Should preserve all fields with empty mapping."""
        mapping = {"field_mappings": {}}
        record = {"a": 1, "b": 2}

        result = apply_field_mapping(record, mapping)

        assert result == {"a": 1, "b": 2}

    def test_skips_missing_source_fields(self):
        """Should skip mappings for missing source fields."""
        mapping = {
            "field_mappings": {
                "sym": "ticker",
                "missing": "target",
            }
        }
        record = {"sym": "AMGN"}

        result = apply_field_mapping(record, mapping)

        assert result["ticker"] == "AMGN"
        assert "target" not in result


class TestGetMappingMetadata:
    """Tests for get_mapping_metadata function."""

    @pytest.fixture
    def temp_adapters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            adapters_dir = Path(temp_dir)
            source_dir = adapters_dir / "test_source"
            source_dir.mkdir()

            mapping = {
                "required_fields": ["ticker", "date"],
                "field_mappings": {"a": "b", "c": "d"},
            }

            mapping_file = source_dir / "mapping_v1.json"
            with open(mapping_file, "w") as f:
                json.dump(mapping, f)

            yield adapters_dir

    def test_returns_metadata(self, temp_adapters):
        """Should return metadata dict."""
        result = get_mapping_metadata("test_source", adapters_dir=temp_adapters)

        assert result["source_name"] == "test_source"
        assert result["mapping_version"] == "v1"
        assert result["exists"] is True
        assert "mapping_hash" in result
        assert result["required_fields"] == ["ticker", "date"]
        assert result["field_count"] == 2

    def test_reports_missing_file(self, temp_adapters):
        """Should report when file doesn't exist."""
        result = get_mapping_metadata("nonexistent", adapters_dir=temp_adapters)

        assert result["exists"] is False
        assert "mapping_hash" not in result


class TestSchemaMismatchError:
    """Tests for SchemaMismatchError exception."""

    def test_stores_missing_fields(self):
        """Should store missing fields."""
        error = SchemaMismatchError(["field1", "field2"], "test_source")
        assert error.missing_fields == ["field1", "field2"]

    def test_stores_source(self):
        """Should store source name."""
        error = SchemaMismatchError(["field1"], "my_source")
        assert error.source == "my_source"

    def test_message_includes_fields(self):
        """Error message should include missing fields."""
        error = SchemaMismatchError(["field1", "field2"], "test")
        assert "field1" in str(error)
        assert "field2" in str(error)
        assert "SCHEMA_MISMATCH" in str(error)


class TestConstants:
    """Tests for module constants."""

    def test_default_adapters_dir(self):
        """Default adapters dir should be 'adapters'."""
        assert DEFAULT_ADAPTERS_DIR == "adapters"
