#!/usr/bin/env python3
"""
Tests for Schema Registry

Covers:
- Schema version validation
- Score version validation
- Version constants
- Output schema lookups
"""

import pytest
from pathlib import Path

# Import module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from governance.schema_registry import (
    PIPELINE_VERSION,
    SCHEMA_VERSION,
    OUTPUT_SCHEMA_VERSIONS,
    DEFAULT_SCORE_VERSION,
    SUPPORTED_SCORE_VERSIONS,
    validate_schema_version,
    validate_score_version,
    get_schema_info,
    get_output_schema_version,
)


# ============================================================================
# VERSION CONSTANTS
# ============================================================================

class TestVersionConstants:
    """Tests for version constants."""

    def test_pipeline_version_format(self):
        """Pipeline version is in X.Y.Z format."""
        parts = PIPELINE_VERSION.split('.')
        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)

    def test_schema_version_format(self):
        """Schema version is in X.Y.Z format."""
        parts = SCHEMA_VERSION.split('.')
        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)

    def test_default_score_version_supported(self):
        """Default score version is in supported list."""
        assert DEFAULT_SCORE_VERSION in SUPPORTED_SCORE_VERSIONS

    def test_output_schema_versions_not_empty(self):
        """Output schema versions dictionary is not empty."""
        assert len(OUTPUT_SCHEMA_VERSIONS) > 0

    def test_output_schema_versions_format(self):
        """All output schema versions are in X.Y.Z format."""
        for output_type, version in OUTPUT_SCHEMA_VERSIONS.items():
            parts = version.split('.')
            assert len(parts) == 3, f"Invalid version for {output_type}: {version}"


# ============================================================================
# VALIDATE SCHEMA VERSION
# ============================================================================

class TestValidateSchemaVersion:
    """Tests for schema version validation."""

    def test_exact_match_valid(self):
        """Exact version match is valid."""
        valid, msg = validate_schema_version("1.0.0", "1.0.0")
        assert valid is True
        assert "match" in msg.lower()

    def test_compatible_minor_version(self):
        """Higher minor version is compatible (non-strict)."""
        valid, msg = validate_schema_version("1.1.0", "1.0.0", strict=False)
        assert valid is True
        assert "compatible" in msg.lower()

    def test_compatible_patch_version(self):
        """Higher patch version is compatible (non-strict)."""
        valid, msg = validate_schema_version("1.0.5", "1.0.0", strict=False)
        assert valid is True

    def test_strict_rejects_different_versions(self):
        """Strict mode rejects different versions."""
        valid, msg = validate_schema_version("1.1.0", "1.0.0", strict=True)
        assert valid is False
        assert "mismatch" in msg.lower()

    def test_incompatible_major_version(self):
        """Different major version is incompatible."""
        valid, msg = validate_schema_version("2.0.0", "1.0.0", strict=False)
        assert valid is False
        assert "incompatible" in msg.lower()

    def test_lower_minor_version_invalid(self):
        """Lower minor version is invalid."""
        valid, msg = validate_schema_version("1.0.0", "1.1.0", strict=False)
        assert valid is False
        assert "too old" in msg.lower()

    def test_invalid_format_rejected(self):
        """Invalid version format is rejected."""
        valid, msg = validate_schema_version("invalid", "1.0.0")
        assert valid is False
        assert "invalid" in msg.lower()

    def test_partial_version_rejected(self):
        """Partial version (X.Y) is rejected."""
        valid, msg = validate_schema_version("1.0", "1.0.0")
        assert valid is False

    def test_uses_default_when_expected_not_provided(self):
        """Uses SCHEMA_VERSION when expected_version not provided."""
        valid, msg = validate_schema_version(SCHEMA_VERSION)
        assert valid is True

    def test_none_version_invalid(self):
        """None version is invalid."""
        valid, msg = validate_schema_version(None, "1.0.0")
        assert valid is False


# ============================================================================
# VALIDATE SCORE VERSION
# ============================================================================

class TestValidateScoreVersion:
    """Tests for score version validation."""

    def test_v1_supported(self):
        """v1 is a supported score version."""
        valid, msg = validate_score_version("v1")
        assert valid is True
        assert "supported" in msg.lower()

    def test_unsupported_version_rejected(self):
        """Unsupported version is rejected."""
        valid, msg = validate_score_version("v999")
        assert valid is False
        assert "unsupported" in msg.lower()

    def test_empty_version_rejected(self):
        """Empty version is rejected."""
        valid, msg = validate_score_version("")
        assert valid is False

    def test_error_message_lists_supported(self):
        """Error message lists supported versions."""
        valid, msg = validate_score_version("invalid")
        assert valid is False
        assert "v1" in msg  # Should mention supported versions


# ============================================================================
# GET SCHEMA INFO
# ============================================================================

class TestGetSchemaInfo:
    """Tests for get_schema_info function."""

    def test_returns_dict(self):
        """Returns a dictionary."""
        info = get_schema_info()
        assert isinstance(info, dict)

    def test_includes_pipeline_version(self):
        """Includes pipeline_version."""
        info = get_schema_info()
        assert info["pipeline_version"] == PIPELINE_VERSION

    def test_includes_schema_version(self):
        """Includes schema_version."""
        info = get_schema_info()
        assert info["schema_version"] == SCHEMA_VERSION

    def test_includes_default_score_version(self):
        """Includes default_score_version."""
        info = get_schema_info()
        assert info["default_score_version"] == DEFAULT_SCORE_VERSION

    def test_includes_supported_score_versions(self):
        """Includes supported_score_versions."""
        info = get_schema_info()
        assert info["supported_score_versions"] == SUPPORTED_SCORE_VERSIONS

    def test_includes_output_schema_versions(self):
        """Includes output_schema_versions."""
        info = get_schema_info()
        assert info["output_schema_versions"] == OUTPUT_SCHEMA_VERSIONS


# ============================================================================
# GET OUTPUT SCHEMA VERSION
# ============================================================================

class TestGetOutputSchemaVersion:
    """Tests for get_output_schema_version function."""

    def test_known_output_type(self):
        """Returns version for known output type."""
        for output_type in OUTPUT_SCHEMA_VERSIONS:
            version = get_output_schema_version(output_type)
            assert version == OUTPUT_SCHEMA_VERSIONS[output_type]

    def test_unknown_output_type_raises(self):
        """Raises KeyError for unknown output type."""
        with pytest.raises(KeyError, match="Unknown output type"):
            get_output_schema_version("unknown_type")

    def test_error_lists_known_types(self):
        """Error message lists known types."""
        try:
            get_output_schema_version("unknown_type")
        except KeyError as e:
            error_msg = str(e)
            # At least one known type should be in the error
            assert any(t in error_msg for t in OUTPUT_SCHEMA_VERSIONS.keys())


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_version_with_leading_zeros(self):
        """Version with leading zeros (e.g., 1.01.0) works."""
        # This is technically valid semver-ish
        valid, msg = validate_schema_version("1.01.0", "1.1.0")
        # 01 should parse as 1
        assert valid is True

    def test_version_with_high_numbers(self):
        """High version numbers work."""
        valid, msg = validate_schema_version("999.999.999", "999.999.999")
        assert valid is True

    def test_empty_string_version_invalid(self):
        """Empty string version is invalid."""
        valid, msg = validate_schema_version("", "1.0.0")
        assert valid is False

    def test_whitespace_version_invalid(self):
        """Whitespace version is invalid."""
        valid, msg = validate_schema_version("  ", "1.0.0")
        assert valid is False

    def test_version_with_extra_parts_invalid(self):
        """Version with extra parts (X.Y.Z.W) is invalid."""
        valid, msg = validate_schema_version("1.0.0.0", "1.0.0")
        assert valid is False


# ============================================================================
# DETERMINISM
# ============================================================================

class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_validate_schema_version_deterministic(self):
        """Schema validation is deterministic."""
        for _ in range(10):
            valid, msg = validate_schema_version("1.1.0", "1.0.0", strict=False)
            assert valid is True

    def test_validate_score_version_deterministic(self):
        """Score validation is deterministic."""
        for _ in range(10):
            valid, msg = validate_score_version("v1")
            assert valid is True

    def test_get_schema_info_deterministic(self):
        """Schema info is deterministic."""
        info1 = get_schema_info()
        info2 = get_schema_info()
        assert info1 == info2

