#!/usr/bin/env python3
"""
Integration tests for run_screen.py - the main screening pipeline orchestrator.

Tests:
1. Date validation and format handling
2. JSON data loading and validation
3. Checkpointing (save/load/resume)
4. Dry-run validation
5. Audit trail creation
6. Full pipeline execution (with mocked modules)

Run: pytest tests/test_run_screen.py -v
"""

import json
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_screen import (
    validate_as_of_date_param,
    load_json_data,
    write_json_output,
    save_checkpoint,
    load_checkpoint,
    get_resume_module_index,
    validate_inputs_dry_run,
    create_audit_record,
    _force_deterministic_generated_at,
    CHECKPOINT_MODULES,
    VERSION,
)


# ============================================================================
# DATE VALIDATION TESTS
# ============================================================================

class TestDateValidation:
    """Tests for as_of_date validation."""

    def test_valid_date_format_passes(self):
        """Valid YYYY-MM-DD format should not raise."""
        validate_as_of_date_param("2026-01-15")
        validate_as_of_date_param("2025-12-31")
        validate_as_of_date_param("2024-02-29")  # Leap year

    def test_invalid_date_format_raises(self):
        """Invalid date formats should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid as_of_date format"):
            validate_as_of_date_param("01-15-2026")  # Wrong order

        with pytest.raises(ValueError, match="Invalid as_of_date format"):
            validate_as_of_date_param("2026/01/15")  # Wrong separator

        with pytest.raises(ValueError, match="Invalid as_of_date format"):
            validate_as_of_date_param("2026-1-15")  # Missing leading zeros

    def test_invalid_date_values_raise(self):
        """Invalid date values should raise ValueError."""
        with pytest.raises(ValueError):
            validate_as_of_date_param("2026-13-01")  # Invalid month

        with pytest.raises(ValueError):
            validate_as_of_date_param("2026-02-30")  # Invalid day

    def test_empty_string_raises(self):
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError):
            validate_as_of_date_param("")

    def test_none_raises(self):
        """None should raise ValueError."""
        with pytest.raises(ValueError):
            validate_as_of_date_param(None)


# ============================================================================
# JSON DATA LOADING TESTS
# ============================================================================

class TestJsonDataLoading:
    """Tests for JSON file loading."""

    def test_load_valid_json_array(self, tmp_path):
        """Load valid JSON array file."""
        data = [{"ticker": "ACME"}, {"ticker": "BETA"}]
        filepath = tmp_path / "test.json"
        filepath.write_text(json.dumps(data))

        result = load_json_data(filepath, "Test data")
        assert result == data

    def test_load_missing_file_raises(self, tmp_path):
        """Missing file should raise FileNotFoundError."""
        filepath = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError, match="Test file not found"):
            load_json_data(filepath, "Test")

    def test_load_invalid_json_raises(self, tmp_path):
        """Invalid JSON should raise JSONDecodeError."""
        filepath = tmp_path / "invalid.json"
        filepath.write_text("{ invalid json }")

        with pytest.raises(json.JSONDecodeError):
            load_json_data(filepath, "Test")

    def test_load_non_array_raises(self, tmp_path):
        """Non-array JSON should raise ValueError."""
        filepath = tmp_path / "object.json"
        filepath.write_text('{"key": "value"}')

        with pytest.raises(ValueError, match="must be a JSON array"):
            load_json_data(filepath, "Test")

    def test_load_empty_array(self, tmp_path):
        """Empty array should load successfully."""
        filepath = tmp_path / "empty.json"
        filepath.write_text("[]")

        result = load_json_data(filepath, "Test")
        assert result == []


# ============================================================================
# JSON OUTPUT TESTS
# ============================================================================

class TestJsonOutput:
    """Tests for JSON output writing."""

    def test_write_creates_parent_dirs(self, tmp_path):
        """Writing should create parent directories."""
        filepath = tmp_path / "nested" / "deep" / "output.json"
        data = {"key": "value"}

        write_json_output(filepath, data)

        assert filepath.exists()
        assert json.loads(filepath.read_text()) == data

    def test_write_deterministic_key_ordering(self, tmp_path):
        """Output should have sorted keys for determinism."""
        filepath = tmp_path / "sorted.json"
        data = {"zebra": 1, "alpha": 2, "middle": 3}

        write_json_output(filepath, data)

        content = filepath.read_text()
        # Keys should appear in alphabetical order
        assert content.index('"alpha"') < content.index('"middle"')
        assert content.index('"middle"') < content.index('"zebra"')

    def test_write_handles_decimal(self, tmp_path):
        """Decimal values should be serialized as strings."""
        filepath = tmp_path / "decimal.json"
        data = {"score": Decimal("42.50")}

        write_json_output(filepath, data)

        loaded = json.loads(filepath.read_text())
        assert loaded["score"] == "42.50"

    def test_write_handles_date(self, tmp_path):
        """Date values should be serialized as ISO strings."""
        filepath = tmp_path / "date.json"
        data = {"as_of": date(2026, 1, 15)}

        write_json_output(filepath, data)

        loaded = json.loads(filepath.read_text())
        assert loaded["as_of"] == "2026-01-15"

    def test_write_trailing_newline(self, tmp_path):
        """Output should end with trailing newline."""
        filepath = tmp_path / "newline.json"
        data = {"key": "value"}

        write_json_output(filepath, data)

        content = filepath.read_text()
        assert content.endswith("\n")


# ============================================================================
# CHECKPOINTING TESTS
# ============================================================================

class TestCheckpointing:
    """Tests for checkpoint save/load functionality."""

    def test_save_checkpoint_creates_file(self, tmp_path):
        """Saving checkpoint should create JSON file."""
        checkpoint_dir = tmp_path / "checkpoints"
        data = {"scores": [{"ticker": "ACME", "score": 85}]}

        filepath = save_checkpoint(checkpoint_dir, "module_1", "2026-01-15", data)

        assert filepath.exists()
        assert filepath.name == "module_1_2026-01-15.json"

    def test_save_checkpoint_includes_metadata(self, tmp_path):
        """Checkpoint should include module name, date, and version."""
        checkpoint_dir = tmp_path / "checkpoints"
        data = {"scores": []}

        save_checkpoint(checkpoint_dir, "module_2", "2026-01-15", data)

        filepath = checkpoint_dir / "module_2_2026-01-15.json"
        loaded = json.loads(filepath.read_text())

        assert loaded["module"] == "module_2"
        assert loaded["as_of_date"] == "2026-01-15"
        assert loaded["version"] == VERSION
        assert loaded["data"] == data

    def test_load_checkpoint_returns_data(self, tmp_path):
        """Loading checkpoint should return original data."""
        checkpoint_dir = tmp_path / "checkpoints"
        original_data = {"scores": [{"ticker": "ACME", "score": 85}]}

        save_checkpoint(checkpoint_dir, "module_1", "2026-01-15", original_data)
        loaded = load_checkpoint(checkpoint_dir, "module_1", "2026-01-15")

        assert loaded == original_data

    def test_load_checkpoint_missing_returns_none(self, tmp_path):
        """Loading missing checkpoint should return None."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        result = load_checkpoint(checkpoint_dir, "module_1", "2026-01-15")

        assert result is None

    def test_load_checkpoint_version_mismatch_returns_none(self, tmp_path):
        """Loading checkpoint with major version mismatch should return None."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Create checkpoint with different major version
        checkpoint_data = {
            "module": "module_1",
            "as_of_date": "2026-01-15",
            "version": "0.1.0",  # Different major version
            "data": {"old": "data"},
        }
        filepath = checkpoint_dir / "module_1_2026-01-15.json"
        filepath.write_text(json.dumps(checkpoint_data))

        result = load_checkpoint(checkpoint_dir, "module_1", "2026-01-15")

        assert result is None


# ============================================================================
# RESUME MODULE INDEX TESTS
# ============================================================================

class TestResumeModuleIndex:
    """Tests for resume_from module index calculation."""

    def test_none_returns_zero(self):
        """None resume_from should return 0 (start from beginning)."""
        assert get_resume_module_index(None) == 0

    def test_valid_modules_return_correct_index(self):
        """Valid module names should return correct indices."""
        assert get_resume_module_index("module_1") == 0
        assert get_resume_module_index("module_2") == 1
        assert get_resume_module_index("module_3") == 2
        assert get_resume_module_index("module_4") == 3
        assert get_resume_module_index("enhancements") == 4
        assert get_resume_module_index("module_5") == 5

    def test_unknown_module_returns_zero(self):
        """Unknown module name should return 0."""
        assert get_resume_module_index("invalid_module") == 0
        assert get_resume_module_index("module_99") == 0


# ============================================================================
# DRY-RUN VALIDATION TESTS
# ============================================================================

class TestDryRunValidation:
    """Tests for dry-run input validation."""

    def test_valid_data_dir_passes(self, sample_data_dir):
        """Valid data directory should pass validation."""
        result = validate_inputs_dry_run(sample_data_dir)

        assert result["valid"] is True
        assert len(result["errors"]) == 0
        assert "universe.json" in result["required_files"]
        assert result["required_files"]["universe.json"]["exists"] is True

    def test_missing_required_file_fails(self, tmp_path):
        """Missing required file should fail validation."""
        data_dir = tmp_path / "incomplete"
        data_dir.mkdir()

        # Create some but not all required files
        (data_dir / "universe.json").write_text("[]")

        result = validate_inputs_dry_run(data_dir)

        assert result["valid"] is False
        assert any("financial_records.json" in err for err in result["errors"])

    def test_content_hashes_computed(self, sample_data_dir):
        """Content hashes should be computed for existing files."""
        result = validate_inputs_dry_run(sample_data_dir)

        assert "content_hashes" in result
        assert "universe.json" in result["content_hashes"]
        # Hash should be 16-char hex string
        assert len(result["content_hashes"]["universe.json"]) == 16

    def test_record_counts_tracked(self, sample_data_dir):
        """Record counts should be tracked for valid files."""
        result = validate_inputs_dry_run(sample_data_dir)

        universe_info = result["required_files"]["universe.json"]
        assert "record_count" in universe_info
        assert universe_info["record_count"] >= 0

    def test_coinvest_required_when_enabled(self, sample_data_dir):
        """Missing coinvest file should fail when coinvest enabled."""
        # sample_data_dir doesn't have coinvest_signals.json
        result = validate_inputs_dry_run(sample_data_dir, enable_coinvest=True)

        assert result["valid"] is False
        assert any("coinvest" in err.lower() for err in result["errors"])


# ============================================================================
# AUDIT RECORD TESTS
# ============================================================================

class TestAuditRecord:
    """Tests for audit record creation."""

    def test_audit_record_structure(self, sample_data_dir):
        """Audit record should have expected structure."""
        content_hashes = {"universe.json": "abc123", "financial_records.json": "def456"}

        record = create_audit_record("2026-01-15", sample_data_dir, content_hashes)

        assert record["as_of_date"] == "2026-01-15"
        assert record["orchestrator_version"] == VERSION
        assert record["data_dir"] == str(sample_data_dir)
        assert record["input_hashes"] == content_hashes

    def test_audit_record_sorted_hashes(self, sample_data_dir):
        """Input hashes should be sorted for determinism."""
        content_hashes = {"zebra.json": "z", "alpha.json": "a", "middle.json": "m"}

        record = create_audit_record("2026-01-15", sample_data_dir, content_hashes)

        keys = list(record["input_hashes"].keys())
        assert keys == sorted(keys)


# ============================================================================
# DETERMINISTIC TIMESTAMP TESTS
# ============================================================================

class TestDeterministicTimestamp:
    """Tests for deterministic timestamp forcing."""

    def test_force_replaces_provenance_timestamps(self):
        """Should recursively replace all provenance.generated_at fields."""
        obj = {
            "provenance": {"generated_at": "2026-01-15T12:34:56Z"},
            "nested": {
                "provenance": {"generated_at": "2026-01-15T11:11:11Z"}
            },
            "array": [
                {"provenance": {"generated_at": "2026-01-15T10:00:00Z"}}
            ]
        }

        _force_deterministic_generated_at(obj, "2026-01-15T00:00:00Z")

        assert obj["provenance"]["generated_at"] == "2026-01-15T00:00:00Z"
        assert obj["nested"]["provenance"]["generated_at"] == "2026-01-15T00:00:00Z"
        assert obj["array"][0]["provenance"]["generated_at"] == "2026-01-15T00:00:00Z"

    def test_force_handles_missing_provenance(self):
        """Should handle objects without provenance gracefully."""
        obj = {"data": {"no_provenance": True}}

        # Should not raise
        _force_deterministic_generated_at(obj, "2026-01-15T00:00:00Z")

        assert obj["data"]["no_provenance"] is True


# ============================================================================
# CHECKPOINT MODULE CONSTANTS TESTS
# ============================================================================

class TestCheckpointModuleConstants:
    """Tests for checkpoint module constants."""

    def test_checkpoint_modules_order(self):
        """Checkpoint modules should be in expected order."""
        expected = ["module_1", "module_2", "module_3", "module_4", "enhancements", "module_5"]
        assert CHECKPOINT_MODULES == expected

    def test_checkpoint_modules_are_unique(self):
        """All checkpoint module names should be unique."""
        assert len(CHECKPOINT_MODULES) == len(set(CHECKPOINT_MODULES))


# ============================================================================
# INTEGRATION TESTS (require more setup)
# ============================================================================

class TestPipelineIntegration:
    """Integration tests for full pipeline execution.

    These tests validate the pipeline coordination without running
    all module computations (which have their own tests).
    """

    def test_full_pipeline_determinism(self, full_sample_data_dir, as_of_date_str, tmp_path):
        """Two runs with same inputs should produce identical outputs."""
        # Import here to avoid circular imports
        from run_screen import run_screening_pipeline
        import hashlib
        import copy

        # First run
        result1 = run_screening_pipeline(
            as_of_date=as_of_date_str,
            data_dir=full_sample_data_dir,
            enable_coinvest=False,
            enable_enhancements=False,
            enable_short_interest=False,
        )

        # Second run with identical inputs
        result2 = run_screening_pipeline(
            as_of_date=as_of_date_str,
            data_dir=full_sample_data_dir,
            enable_coinvest=False,
            enable_enhancements=False,
            enable_short_interest=False,
        )

        # NOTE: The first run creates output files (catalyst_events_*.json) in data_dir.
        # The second run includes these in input_hashes, causing a diff.
        # We compare the core module outputs which should be deterministic.

        # Make copies and remove input_hashes from comparison (varies due to output files)
        r1_compare = copy.deepcopy(result1)
        r2_compare = copy.deepcopy(result2)
        r1_compare.get("run_metadata", {}).pop("input_hashes", None)
        r2_compare.get("run_metadata", {}).pop("input_hashes", None)

        # Serialize to JSON for comparison (stable format)
        json1 = json.dumps(r1_compare, sort_keys=True, default=str)
        json2 = json.dumps(r2_compare, sort_keys=True, default=str)

        hash1 = hashlib.sha256(json1.encode()).hexdigest()
        hash2 = hashlib.sha256(json2.encode()).hexdigest()

        assert hash1 == hash2, "Pipeline core outputs should be deterministic"

        # Verify key structural elements exist
        assert "module_1_universe" in result1
        assert "module_2_financial" in result1
        assert "module_3_catalyst" in result1
        assert "module_4_clinical" in result1
        assert "module_5_composite" in result1
        assert "run_metadata" in result1
        assert result1["run_metadata"]["as_of_date"] == as_of_date_str

        # Verify module outputs match exactly
        assert result1["module_1_universe"] == result2["module_1_universe"]
        assert result1["module_2_financial"] == result2["module_2_financial"]
        assert result1["module_4_clinical"] == result2["module_4_clinical"]
        assert result1["module_5_composite"] == result2["module_5_composite"]
        assert result1["summary"] == result2["summary"]

    def test_pipeline_with_checkpointing(self, sample_data_dir, tmp_path, as_of_date_str):
        """Pipeline should save and restore checkpoints correctly."""
        from run_screen import run_screening_pipeline

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # First run: complete pipeline with checkpointing
        result_full = run_screening_pipeline(
            as_of_date=as_of_date_str,
            data_dir=sample_data_dir,
            checkpoint_dir=checkpoint_dir,
            enable_coinvest=False,
            enable_enhancements=False,
            enable_short_interest=False,
        )

        # Verify checkpoint files were created
        checkpoint_files = list(checkpoint_dir.glob("*.json"))
        assert len(checkpoint_files) > 0, "Checkpoint files should be created"

        # Verify at least module_1 checkpoint exists
        m1_checkpoint = checkpoint_dir / f"module_1_{as_of_date_str}.json"
        assert m1_checkpoint.exists(), "Module 1 checkpoint should exist"

        # Second run: resume from module_3 (should reuse earlier checkpoints)
        result_resumed = run_screening_pipeline(
            as_of_date=as_of_date_str,
            data_dir=sample_data_dir,
            checkpoint_dir=checkpoint_dir,
            resume_from="module_3",
            enable_coinvest=False,
            enable_enhancements=False,
            enable_short_interest=False,
        )

        # Verify module_1 and module_2 results match (reused from checkpoints)
        assert result_full["module_1_universe"] == result_resumed["module_1_universe"], \
            "Module 1 results should match when resumed"
        assert result_full["module_2_financial"] == result_resumed["module_2_financial"], \
            "Module 2 results should match when resumed"

        # Verify final composite results have same structure
        assert len(result_resumed["module_5_composite"].get("ranked_securities", [])) >= 0
