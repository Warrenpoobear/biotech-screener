#!/usr/bin/env python3
"""
Unit tests for governance/hashing.py

Tests cryptographic hashing functionality:
- Bytes hashing
- File hashing
- Canonical JSON hashing
- Hash verification
- Input hash computation
"""

import pytest
import json
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from governance.hashing import (
    hash_bytes,
    hash_file,
    hash_canonical_json,
    hash_canonical_json_short,
    verify_file_hash,
    compute_input_hashes,
)


# ============================================================================
# BYTES HASHING TESTS
# ============================================================================

class TestHashBytes:
    """Tests for hash_bytes function."""

    def test_basic_hashing(self):
        """Basic bytes hashing should work."""
        data = b"Hello, World!"
        result = hash_bytes(data)
        assert isinstance(result, str)
        assert len(result) == 64  # SHA256 hex digest

    def test_hash_is_lowercase_hex(self):
        """Hash should be lowercase hexadecimal."""
        result = hash_bytes(b"test")
        assert result.islower() or result.isdigit()
        assert all(c in "0123456789abcdef" for c in result)

    def test_empty_bytes(self):
        """Empty bytes should produce valid hash."""
        result = hash_bytes(b"")
        assert len(result) == 64
        # Known SHA256 of empty string
        assert result == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_deterministic(self):
        """Same input should always produce same hash."""
        data = b"deterministic test"
        hash1 = hash_bytes(data)
        hash2 = hash_bytes(data)
        assert hash1 == hash2

    def test_different_inputs_different_hashes(self):
        """Different inputs should produce different hashes."""
        hash1 = hash_bytes(b"input1")
        hash2 = hash_bytes(b"input2")
        assert hash1 != hash2

    def test_binary_data(self):
        """Should handle arbitrary binary data."""
        data = bytes(range(256))  # All byte values
        result = hash_bytes(data)
        assert len(result) == 64


# ============================================================================
# FILE HASHING TESTS
# ============================================================================

class TestHashFile:
    """Tests for hash_file function."""

    def test_hash_existing_file(self, tmp_path):
        """Should hash existing file contents."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"file content")

        result = hash_file(test_file)
        assert len(result) == 64

    def test_hash_matches_content_hash(self, tmp_path):
        """File hash should match hash of file contents."""
        content = b"matching content test"
        test_file = tmp_path / "match.txt"
        test_file.write_bytes(content)

        file_hash = hash_file(test_file)
        content_hash = hash_bytes(content)
        assert file_hash == content_hash

    def test_nonexistent_file_raises(self):
        """Nonexistent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            hash_file("/nonexistent/path/file.txt")

    def test_accepts_string_path(self, tmp_path):
        """Should accept string path in addition to Path."""
        test_file = tmp_path / "string_path.txt"
        test_file.write_bytes(b"content")

        result = hash_file(str(test_file))
        assert len(result) == 64

    def test_binary_file(self, tmp_path):
        """Should handle binary files."""
        test_file = tmp_path / "binary.bin"
        test_file.write_bytes(bytes(range(256)))

        result = hash_file(test_file)
        assert len(result) == 64

    def test_empty_file(self, tmp_path):
        """Should handle empty files."""
        test_file = tmp_path / "empty.txt"
        test_file.write_bytes(b"")

        result = hash_file(test_file)
        # Known SHA256 of empty content
        assert result == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


# ============================================================================
# CANONICAL JSON HASHING TESTS
# ============================================================================

class TestHashCanonicalJson:
    """Tests for hash_canonical_json function."""

    def test_basic_object(self):
        """Should hash JSON object."""
        obj = {"key": "value"}
        result = hash_canonical_json(obj)
        assert len(result) == 64

    def test_deterministic_regardless_of_key_order(self):
        """Hash should be same regardless of key insertion order."""
        obj1 = {"a": 1, "b": 2, "c": 3}
        obj2 = {"c": 3, "b": 2, "a": 1}

        hash1 = hash_canonical_json(obj1)
        hash2 = hash_canonical_json(obj2)
        assert hash1 == hash2

    def test_nested_objects(self):
        """Should handle nested objects."""
        obj = {
            "outer": {
                "inner": {
                    "value": 42
                }
            }
        }
        result = hash_canonical_json(obj)
        assert len(result) == 64

    def test_arrays(self):
        """Should handle arrays."""
        obj = {"items": [1, 2, 3]}
        result = hash_canonical_json(obj)
        assert len(result) == 64

    def test_array_order_matters(self):
        """Array order should affect hash (not sorted)."""
        obj1 = {"items": [1, 2, 3]}
        obj2 = {"items": [3, 2, 1]}

        hash1 = hash_canonical_json(obj1)
        hash2 = hash_canonical_json(obj2)
        assert hash1 != hash2

    def test_nan_raises_error(self):
        """NaN values should raise ValueError."""
        obj = {"value": float('nan')}
        with pytest.raises(ValueError):
            hash_canonical_json(obj)

    def test_infinity_raises_error(self):
        """Infinity values should raise ValueError."""
        obj = {"value": float('inf')}
        with pytest.raises(ValueError):
            hash_canonical_json(obj)


class TestHashCanonicalJsonShort:
    """Tests for hash_canonical_json_short function."""

    def test_default_length(self):
        """Default length should be 16 characters."""
        obj = {"key": "value"}
        result = hash_canonical_json_short(obj)
        assert len(result) == 16

    def test_custom_length(self):
        """Should respect custom length parameter."""
        obj = {"key": "value"}

        assert len(hash_canonical_json_short(obj, length=8)) == 8
        assert len(hash_canonical_json_short(obj, length=32)) == 32

    def test_is_prefix_of_full_hash(self):
        """Short hash should be prefix of full hash."""
        obj = {"key": "value"}
        full_hash = hash_canonical_json(obj)
        short_hash = hash_canonical_json_short(obj, length=16)

        assert full_hash.startswith(short_hash)

    def test_deterministic(self):
        """Should be deterministic."""
        obj = {"test": "data"}
        hash1 = hash_canonical_json_short(obj)
        hash2 = hash_canonical_json_short(obj)
        assert hash1 == hash2


# ============================================================================
# HASH VERIFICATION TESTS
# ============================================================================

class TestVerifyFileHash:
    """Tests for verify_file_hash function."""

    def test_correct_hash_verifies(self, tmp_path):
        """Correct hash should verify."""
        content = b"verify this content"
        test_file = tmp_path / "verify.txt"
        test_file.write_bytes(content)

        expected_hash = hash_bytes(content)
        assert verify_file_hash(test_file, expected_hash) is True

    def test_incorrect_hash_fails(self, tmp_path):
        """Incorrect hash should fail verification."""
        test_file = tmp_path / "verify.txt"
        test_file.write_bytes(b"content")

        wrong_hash = "0" * 64
        assert verify_file_hash(test_file, wrong_hash) is False

    def test_prefix_matching(self, tmp_path):
        """Should support prefix matching for truncated hashes."""
        content = b"prefix test"
        test_file = tmp_path / "prefix.txt"
        test_file.write_bytes(content)

        full_hash = hash_bytes(content)
        prefix = full_hash[:16]

        assert verify_file_hash(test_file, prefix) is True

    def test_case_insensitive(self, tmp_path):
        """Hash comparison should be case-insensitive."""
        content = b"case test"
        test_file = tmp_path / "case.txt"
        test_file.write_bytes(content)

        expected_hash = hash_bytes(content).upper()
        # verify_file_hash lowercases the expected hash
        assert verify_file_hash(test_file, expected_hash) is True

    def test_nonexistent_file(self):
        """Nonexistent file should raise error."""
        with pytest.raises(FileNotFoundError):
            verify_file_hash("/nonexistent/file.txt", "abc123")


# ============================================================================
# INPUT HASHES TESTS
# ============================================================================

class TestComputeInputHashes:
    """Tests for compute_input_hashes function."""

    def test_single_file(self, tmp_path):
        """Should hash single file."""
        test_file = tmp_path / "single.txt"
        test_file.write_bytes(b"content")

        results = compute_input_hashes([test_file])
        assert len(results) == 1
        assert results[0]["path"] == "single.txt"
        assert len(results[0]["sha256"]) == 64

    def test_multiple_files(self, tmp_path):
        """Should hash multiple files."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_bytes(b"content1")
        file2.write_bytes(b"content2")

        results = compute_input_hashes([file1, file2])
        assert len(results) == 2

    def test_sorted_by_path(self, tmp_path):
        """Results should be sorted by path for determinism."""
        file_z = tmp_path / "z_file.txt"
        file_a = tmp_path / "a_file.txt"
        file_m = tmp_path / "m_file.txt"

        for f in [file_z, file_a, file_m]:
            f.write_bytes(b"content")

        # Pass in unsorted order
        results = compute_input_hashes([file_z, file_a, file_m])

        paths = [r["path"] for r in results]
        assert paths == sorted(paths)

    def test_returns_basename(self, tmp_path):
        """Should return basename, not full path."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        test_file = subdir / "nested.txt"
        test_file.write_bytes(b"content")

        results = compute_input_hashes([test_file])
        assert results[0]["path"] == "nested.txt"  # Not full path

    def test_nonexistent_file_raises(self, tmp_path):
        """Nonexistent file should raise error."""
        existing = tmp_path / "exists.txt"
        existing.write_bytes(b"content")
        nonexistent = tmp_path / "missing.txt"

        with pytest.raises(FileNotFoundError):
            compute_input_hashes([existing, nonexistent])

    def test_empty_list(self):
        """Empty list should return empty list."""
        results = compute_input_hashes([])
        assert results == []


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Edge case tests for hashing functions."""

    def test_unicode_file_content(self, tmp_path):
        """Should handle unicode in files."""
        test_file = tmp_path / "unicode.txt"
        test_file.write_text("Hello, World!", encoding='utf-8')

        result = hash_file(test_file)
        assert len(result) == 64

    def test_large_file(self, tmp_path):
        """Should handle large files."""
        test_file = tmp_path / "large.bin"
        # 1MB of data
        test_file.write_bytes(b"x" * (1024 * 1024))

        result = hash_file(test_file)
        assert len(result) == 64

    def test_special_characters_in_filename(self, tmp_path):
        """Should handle special characters in filename."""
        test_file = tmp_path / "file-with_special.chars.txt"
        test_file.write_bytes(b"content")

        result = hash_file(test_file)
        assert len(result) == 64

    def test_json_with_special_strings(self):
        """Should handle JSON with special string characters."""
        obj = {
            "newlines": "line1\nline2",
            "tabs": "col1\tcol2",
            "quotes": '"quoted"',
        }
        result = hash_canonical_json(obj)
        assert len(result) == 64


# ============================================================================
# DETERMINISM TESTS
# ============================================================================

class TestDeterminism:
    """Tests verifying deterministic behavior across runs."""

    def test_bytes_hash_deterministic(self):
        """Bytes hashing should be deterministic."""
        data = b"determinism test data"
        hashes = [hash_bytes(data) for _ in range(100)]
        assert len(set(hashes)) == 1

    def test_json_hash_deterministic(self):
        """JSON hashing should be deterministic."""
        obj = {"complex": {"nested": [1, 2, 3]}, "data": "test"}
        hashes = [hash_canonical_json(obj) for _ in range(100)]
        assert len(set(hashes)) == 1

    def test_file_hash_deterministic(self, tmp_path):
        """File hashing should be deterministic."""
        test_file = tmp_path / "deterministic.txt"
        test_file.write_bytes(b"deterministic file content")

        hashes = [hash_file(test_file) for _ in range(100)]
        assert len(set(hashes)) == 1


# ============================================================================
# KNOWN VALUE TESTS
# ============================================================================

class TestKnownValues:
    """Tests against known SHA256 values."""

    def test_known_string_hash(self):
        """Test against known SHA256 hash."""
        # SHA256("test") = known value
        result = hash_bytes(b"test")
        assert result == "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"

    def test_known_empty_hash(self):
        """Test empty string hash."""
        result = hash_bytes(b"")
        assert result == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
