#!/usr/bin/env python3
"""
Unit tests for governance/params_loader.py

Tests parameter loading and validation:
- Path resolution
- JSON loading and validation
- Security features (symlink check, file size)
- Schema validation
- Hash computation
- Parameter saving
"""

import pytest
import json
import tempfile
from pathlib import Path
from decimal import Decimal
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from governance.params_loader import (
    get_params_path,
    load_params,
    save_params,
    compute_parameters_hash,
    validate_params_structure,
    get_params_metadata,
    load_and_validate_params,
    ParamsLoadError,
    ParamsValidationError,
    DEFAULT_PARAMS_DIR,
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def sample_params():
    """Sample parameters for testing."""
    return {
        "score_version": "v1",
        "weights": {
            "clinical": "0.40",
            "financial": "0.35",
            "catalyst": "0.25",
        },
        "thresholds": {
            "min_market_cap": 50,
            "max_positions": 60,
        },
    }


@pytest.fixture
def params_dir(tmp_path):
    """Create temporary params directory."""
    params_dir = tmp_path / "params_archive"
    params_dir.mkdir()
    return params_dir


# ============================================================================
# PATH RESOLUTION TESTS
# ============================================================================

class TestGetParamsPath:
    """Tests for get_params_path function."""

    def test_default_directory(self):
        """Should use default directory when not specified."""
        path = get_params_path("v1")
        assert path.name == "v1.json"

    def test_custom_directory(self, params_dir):
        """Should use custom directory when specified."""
        path = get_params_path("v1", params_dir)
        assert path == params_dir / "v1.json"

    def test_version_in_filename(self, params_dir):
        """Version should be in filename."""
        path = get_params_path("v2.1", params_dir)
        assert path.name == "v2.1.json"


# ============================================================================
# LOAD PARAMS TESTS
# ============================================================================

class TestLoadParams:
    """Tests for load_params function."""

    def test_load_valid_params(self, params_dir, sample_params):
        """Should load valid params file."""
        params_path = params_dir / "v1.json"
        with open(params_path, "w") as f:
            json.dump(sample_params, f)

        params, params_hash = load_params("v1", params_dir, validate_schema=False)

        assert params["score_version"] == "v1"
        assert params_hash is not None
        assert len(params_hash) == 16  # Default hash length

    def test_missing_file_raises(self, params_dir):
        """Should raise error for missing file."""
        with pytest.raises(ParamsLoadError, match="not found"):
            load_params("nonexistent", params_dir)

    def test_invalid_json_raises(self, params_dir):
        """Should raise error for invalid JSON."""
        params_path = params_dir / "v1.json"
        params_path.write_text("not valid json {")

        with pytest.raises(ParamsLoadError, match="Invalid JSON"):
            load_params("v1", params_dir, validate_schema=False)

    def test_non_dict_raises(self, params_dir):
        """Should raise error if params is not a dict."""
        params_path = params_dir / "v1.json"
        with open(params_path, "w") as f:
            json.dump(["list", "not", "dict"], f)

        with pytest.raises(ParamsLoadError, match="must be a JSON object"):
            load_params("v1", params_dir, validate_schema=False)


# ============================================================================
# HASH COMPUTATION TESTS
# ============================================================================

class TestComputeParametersHash:
    """Tests for compute_parameters_hash function."""

    def test_deterministic_hash(self, sample_params):
        """Same params should produce same hash."""
        hash1 = compute_parameters_hash(sample_params)
        hash2 = compute_parameters_hash(sample_params)
        assert hash1 == hash2

    def test_different_params_different_hash(self, sample_params):
        """Different params should produce different hash."""
        hash1 = compute_parameters_hash(sample_params)
        modified_params = sample_params.copy()
        modified_params["extra"] = "value"
        hash2 = compute_parameters_hash(modified_params)
        assert hash1 != hash2

    def test_custom_length(self, sample_params):
        """Should respect custom hash length."""
        hash8 = compute_parameters_hash(sample_params, length=8)
        hash32 = compute_parameters_hash(sample_params, length=32)
        assert len(hash8) == 8
        assert len(hash32) == 32

    def test_key_order_independent(self):
        """Hash should be independent of key insertion order."""
        params1 = {"a": 1, "b": 2, "c": 3}
        params2 = {"c": 3, "b": 2, "a": 1}
        assert compute_parameters_hash(params1) == compute_parameters_hash(params2)


# ============================================================================
# SAVE PARAMS TESTS
# ============================================================================

class TestSaveParams:
    """Tests for save_params function."""

    def test_save_and_reload(self, params_dir, sample_params):
        """Saved params should be loadable."""
        path, hash1 = save_params(sample_params, "v1", params_dir, validate_schema=False)

        assert path.exists()

        params, hash2 = load_params("v1", params_dir, validate_schema=False)
        assert params == sample_params
        assert hash1 == hash2

    def test_creates_directory(self, tmp_path, sample_params):
        """Should create directory if it doesn't exist."""
        new_dir = tmp_path / "new_params"

        path, _ = save_params(sample_params, "v1", new_dir, validate_schema=False)

        assert new_dir.exists()
        assert path.exists()

    def test_overwrites_existing(self, params_dir, sample_params):
        """Should overwrite existing file."""
        path1, hash1 = save_params(sample_params, "v1", params_dir, validate_schema=False)

        modified = sample_params.copy()
        modified["new_key"] = "new_value"
        path2, hash2 = save_params(modified, "v1", params_dir, validate_schema=False)

        assert path1 == path2
        assert hash1 != hash2


# ============================================================================
# VALIDATE PARAMS STRUCTURE TESTS
# ============================================================================

class TestValidateParamsStructure:
    """Tests for validate_params_structure function."""

    def test_valid_structure(self, sample_params):
        """Valid params should pass."""
        valid, msg = validate_params_structure(sample_params)
        assert valid is True

    def test_non_dict_fails(self):
        """Non-dict should fail."""
        valid, msg = validate_params_structure(["list"])
        assert valid is False
        assert "must be dict" in msg

    def test_missing_required_keys(self, sample_params):
        """Missing required keys should fail."""
        valid, msg = validate_params_structure(
            sample_params,
            required_keys=["score_version", "missing_key"]
        )
        assert valid is False
        assert "missing_key" in msg

    def test_all_required_keys_present(self, sample_params):
        """All required keys present should pass."""
        valid, msg = validate_params_structure(
            sample_params,
            required_keys=["score_version", "weights"]
        )
        assert valid is True


# ============================================================================
# GET PARAMS METADATA TESTS
# ============================================================================

class TestGetParamsMetadata:
    """Tests for get_params_metadata function."""

    def test_metadata_existing_file(self, params_dir, sample_params):
        """Should return metadata for existing file."""
        save_params(sample_params, "v1", params_dir, validate_schema=False)

        metadata = get_params_metadata("v1", params_dir)

        assert metadata["exists"] is True
        assert metadata["score_version"] == "v1"
        assert "parameters_hash" in metadata
        assert "keys" in metadata

    def test_metadata_nonexistent_file(self, params_dir):
        """Should indicate file doesn't exist."""
        metadata = get_params_metadata("nonexistent", params_dir)

        assert metadata["exists"] is False
        assert "parameters_hash" not in metadata


# ============================================================================
# SECURITY TESTS
# ============================================================================

class TestSecurityFeatures:
    """Tests for security features."""

    def test_symlink_rejected(self, params_dir, tmp_path, sample_params):
        """Symlinks should be rejected."""
        # Create actual file
        actual_file = tmp_path / "actual.json"
        with open(actual_file, "w") as f:
            json.dump(sample_params, f)

        # Create symlink
        symlink = params_dir / "v1.json"
        symlink.symlink_to(actual_file)

        with pytest.raises(ParamsLoadError, match="symbolic link"):
            load_params("v1", params_dir, validate_schema=False)


# ============================================================================
# DETERMINISM TESTS
# ============================================================================

class TestDeterminism:
    """Tests verifying deterministic behavior."""

    def test_repeated_load_same_hash(self, params_dir, sample_params):
        """Multiple loads should produce same hash."""
        save_params(sample_params, "v1", params_dir, validate_schema=False)

        hashes = [load_params("v1", params_dir, validate_schema=False)[1] for _ in range(5)]

        assert len(set(hashes)) == 1  # All hashes identical

    def test_save_produces_canonical_json(self, params_dir, sample_params):
        """Saved JSON should be canonical (sorted keys)."""
        save_params(sample_params, "v1", params_dir, validate_schema=False)

        with open(params_dir / "v1.json", "r") as f:
            content = f.read()

        # Should end with newline (canonical format)
        assert content.endswith("\n")

        # Keys should be sorted
        parsed = json.loads(content)
        assert list(parsed.keys()) == sorted(parsed.keys())
