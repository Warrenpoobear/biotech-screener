#!/usr/bin/env python3
"""
Unit tests for governance/run_id.py

Tests deterministic run ID generation:
- Canonical computation from inputs
- Determinism verification
- Input ordering independence
- ID validation
- Metadata parsing
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from governance.run_id import (
    compute_run_id,
    validate_run_id,
    parse_run_id_components,
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def sample_inputs():
    """Standard inputs for computing run ID."""
    return {
        "as_of_date": "2026-01-15",
        "score_version": "v1",
        "parameters_hash": "abc123def4567890",
        "input_hashes": [
            {"path": "universe.json", "sha256": "aaa111"},
            {"path": "financial.json", "sha256": "bbb222"},
            {"path": "clinical.json", "sha256": "ccc333"},
        ],
        "pipeline_version": "1.0.0",
    }


# ============================================================================
# COMPUTE RUN ID TESTS
# ============================================================================

class TestComputeRunId:
    """Tests for compute_run_id function."""

    def test_basic_computation(self, sample_inputs):
        """Should compute a valid run ID."""
        run_id = compute_run_id(**sample_inputs)

        assert isinstance(run_id, str)
        assert len(run_id) == 16  # Default length
        assert validate_run_id(run_id)

    def test_deterministic(self, sample_inputs):
        """Same inputs should produce same run ID."""
        run_id1 = compute_run_id(**sample_inputs)
        run_id2 = compute_run_id(**sample_inputs)

        assert run_id1 == run_id2

    def test_different_inputs_different_id(self, sample_inputs):
        """Different inputs should produce different run ID."""
        run_id1 = compute_run_id(**sample_inputs)

        modified = sample_inputs.copy()
        modified["as_of_date"] = "2026-01-16"
        run_id2 = compute_run_id(**modified)

        assert run_id1 != run_id2

    def test_input_hash_order_independent(self, sample_inputs):
        """Run ID should be independent of input hash order."""
        run_id1 = compute_run_id(**sample_inputs)

        # Reverse the input_hashes order
        modified = sample_inputs.copy()
        modified["input_hashes"] = list(reversed(sample_inputs["input_hashes"]))
        run_id2 = compute_run_id(**modified)

        assert run_id1 == run_id2

    def test_custom_length(self, sample_inputs):
        """Should respect custom length parameter."""
        run_id_8 = compute_run_id(**sample_inputs, length=8)
        run_id_32 = compute_run_id(**sample_inputs, length=32)

        assert len(run_id_8) == 8
        assert len(run_id_32) == 32
        # Shorter should be prefix of longer
        assert run_id_32.startswith(run_id_8)

    def test_with_mapping_hashes(self, sample_inputs):
        """Should handle optional mapping hashes."""
        mapping_hashes = [
            {"name": "indication_mapping", "sha256": "map111"},
            {"name": "pos_benchmarks", "sha256": "map222"},
        ]

        run_id1 = compute_run_id(**sample_inputs)
        run_id2 = compute_run_id(**sample_inputs, mapping_hashes=mapping_hashes)

        # With mappings should be different
        assert run_id1 != run_id2

    def test_mapping_hash_order_independent(self, sample_inputs):
        """Run ID should be independent of mapping hash order."""
        mapping1 = [
            {"name": "a_mapping", "sha256": "aaa"},
            {"name": "z_mapping", "sha256": "zzz"},
        ]
        mapping2 = [
            {"name": "z_mapping", "sha256": "zzz"},
            {"name": "a_mapping", "sha256": "aaa"},
        ]

        run_id1 = compute_run_id(**sample_inputs, mapping_hashes=mapping1)
        run_id2 = compute_run_id(**sample_inputs, mapping_hashes=mapping2)

        assert run_id1 == run_id2

    def test_empty_mapping_hashes(self, sample_inputs):
        """Empty mapping hashes should work."""
        run_id1 = compute_run_id(**sample_inputs)
        run_id2 = compute_run_id(**sample_inputs, mapping_hashes=[])

        assert run_id1 == run_id2  # Both should be equivalent


# ============================================================================
# VALIDATE RUN ID TESTS
# ============================================================================

class TestValidateRunId:
    """Tests for validate_run_id function."""

    def test_valid_hex_string(self):
        """Valid hex string should pass."""
        assert validate_run_id("abc123def4567890") is True
        assert validate_run_id("ABCDEF0123456789") is True

    def test_invalid_type(self):
        """Non-string should fail."""
        assert validate_run_id(12345) is False
        assert validate_run_id(None) is False
        assert validate_run_id(["list"]) is False

    def test_too_short(self):
        """Too short should fail."""
        assert validate_run_id("abc") is False
        assert validate_run_id("1234567") is False  # 7 chars

    def test_too_long(self):
        """Too long should fail."""
        long_id = "a" * 65  # Over 64 chars
        assert validate_run_id(long_id) is False

    def test_valid_lengths(self):
        """Valid lengths should pass."""
        assert validate_run_id("12345678") is True  # 8 chars (min)
        assert validate_run_id("a" * 64) is True  # 64 chars (max)

    def test_non_hex_chars(self):
        """Non-hex characters should fail."""
        assert validate_run_id("ghijklmn12345678") is False  # 'g' is not hex
        assert validate_run_id("abc_xyz_12345678") is False  # underscore not hex


# ============================================================================
# PARSE RUN ID COMPONENTS TESTS
# ============================================================================

class TestParseRunIdComponents:
    """Tests for parse_run_id_components function."""

    def test_parse_valid_id(self):
        """Should parse valid run ID."""
        run_id = "abc123def4567890"
        result = parse_run_id_components(run_id)

        assert result["run_id"] == run_id
        assert result["length"] == 16
        assert result["format"] == "sha256_truncated"
        assert result["valid"] is True

    def test_parse_invalid_id(self):
        """Should indicate invalid IDs."""
        result = parse_run_id_components("invalid_id!")

        assert result["valid"] is False


# ============================================================================
# DETERMINISM TESTS
# ============================================================================

class TestDeterminism:
    """Tests verifying deterministic behavior."""

    def test_repeated_computation(self, sample_inputs):
        """Multiple computations should be identical."""
        run_ids = [compute_run_id(**sample_inputs) for _ in range(100)]

        assert len(set(run_ids)) == 1  # All identical

    def test_all_inputs_affect_id(self, sample_inputs):
        """Each input field should affect the run ID."""
        baseline = compute_run_id(**sample_inputs)

        # Change as_of_date
        modified = sample_inputs.copy()
        modified["as_of_date"] = "2026-01-14"
        assert compute_run_id(**modified) != baseline

        # Change score_version
        modified = sample_inputs.copy()
        modified["score_version"] = "v2"
        assert compute_run_id(**modified) != baseline

        # Change parameters_hash
        modified = sample_inputs.copy()
        modified["parameters_hash"] = "different_hash"
        assert compute_run_id(**modified) != baseline

        # Change pipeline_version
        modified = sample_inputs.copy()
        modified["pipeline_version"] = "2.0.0"
        assert compute_run_id(**modified) != baseline

        # Change an input hash
        modified = sample_inputs.copy()
        modified["input_hashes"] = [
            {"path": "universe.json", "sha256": "changed"},
            {"path": "financial.json", "sha256": "bbb222"},
            {"path": "clinical.json", "sha256": "ccc333"},
        ]
        assert compute_run_id(**modified) != baseline


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Edge case tests for run ID generation."""

    def test_empty_input_hashes(self, sample_inputs):
        """Should handle empty input hashes."""
        modified = sample_inputs.copy()
        modified["input_hashes"] = []

        run_id = compute_run_id(**modified)
        assert validate_run_id(run_id)

    def test_single_input_hash(self, sample_inputs):
        """Should handle single input hash."""
        modified = sample_inputs.copy()
        modified["input_hashes"] = [{"path": "only.json", "sha256": "only111"}]

        run_id = compute_run_id(**modified)
        assert validate_run_id(run_id)

    def test_special_characters_in_version(self, sample_inputs):
        """Should handle special characters in version strings."""
        modified = sample_inputs.copy()
        modified["score_version"] = "v1.2.3-beta+build123"
        modified["pipeline_version"] = "1.0.0-rc.1"

        run_id = compute_run_id(**modified)
        assert validate_run_id(run_id)
