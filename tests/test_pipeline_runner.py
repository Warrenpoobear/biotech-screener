#!/usr/bin/env python3
"""
Tests for Governed Pipeline Runner

Covers:
- Pipeline initialization
- Input validation
- Stage logging
- Output writing
- Error handling
"""

import pytest
import json
import tempfile
from datetime import date
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock

# Import module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from governance.pipeline_runner import (
    GovernedPipeline,
    PipelineError,
)
from governance.audit_log import AuditStage, AuditStatus, AuditErrorCode


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary output directory."""
    return tmp_path / "output"


@pytest.fixture
def tmp_input_files(tmp_path):
    """Create temporary input files."""
    files = []
    for name, content in [
        ("universe.json", '{"tickers": ["ACME", "BETA"]}'),
        ("financial.json", '{"records": []}'),
    ]:
        path = tmp_path / name
        path.write_text(content)
        files.append(path)
    return files


@pytest.fixture
def tmp_params_dir(tmp_path):
    """Create temporary params directory with v1 params."""
    params_dir = tmp_path / "params"
    params_dir.mkdir()

    params_file = params_dir / "v1.json"
    params_file.write_text(json.dumps({
        "financial_weight": 0.25,
        "clinical_weight": 0.40,
        "catalyst_weight": 0.15,
        "score_weights": {"clinical": 0.4, "financial": 0.3},
        "thresholds": {"min_market_cap": 50},
    }))

    return params_dir


@pytest.fixture
def basic_pipeline(tmp_input_files, tmp_output_dir, tmp_params_dir):
    """Create a basic pipeline instance."""
    return GovernedPipeline(
        as_of_date="2026-01-15",
        score_version="v1",
        output_dir=tmp_output_dir,
        input_files=tmp_input_files,
        params_dir=tmp_params_dir,
    )


# ============================================================================
# PIPELINE INITIALIZATION
# ============================================================================

class TestPipelineInitialization:
    """Tests for pipeline initialization."""

    def test_creates_instance(self, tmp_input_files, tmp_output_dir, tmp_params_dir):
        """Creates pipeline instance."""
        pipeline = GovernedPipeline(
            as_of_date="2026-01-15",
            score_version="v1",
            output_dir=tmp_output_dir,
            input_files=tmp_input_files,
            params_dir=tmp_params_dir,
        )

        assert pipeline.as_of_date == "2026-01-15"
        assert pipeline.score_version == "v1"

    def test_initialize_loads_params(self, basic_pipeline):
        """Initialize loads parameters."""
        basic_pipeline.initialize()

        assert basic_pipeline.params != {}
        assert basic_pipeline.parameters_hash != ""

    def test_initialize_computes_run_id(self, basic_pipeline):
        """Initialize computes run_id."""
        basic_pipeline.initialize()

        assert basic_pipeline.run_id != ""
        assert len(basic_pipeline.run_id) > 10

    def test_initialize_creates_output_dir(self, basic_pipeline, tmp_output_dir):
        """Initialize creates output directory."""
        assert not tmp_output_dir.exists()

        basic_pipeline.initialize()

        assert tmp_output_dir.exists()

    def test_initialize_computes_input_hashes(self, basic_pipeline, tmp_input_files):
        """Initialize computes input hashes."""
        basic_pipeline.initialize()

        assert len(basic_pipeline.input_hashes) == len(tmp_input_files)
        for h in basic_pipeline.input_hashes:
            assert "path" in h
            assert "sha256" in h

    def test_dry_run_uses_shadow_dir(self, tmp_input_files, tmp_output_dir, tmp_params_dir):
        """Dry run uses shadow subdirectory."""
        pipeline = GovernedPipeline(
            as_of_date="2026-01-15",
            score_version="v1",
            output_dir=tmp_output_dir,
            input_files=tmp_input_files,
            params_dir=tmp_params_dir,
            dry_run=True,
        )

        pipeline.initialize()

        assert "shadow" in str(pipeline.output_dir)


# ============================================================================
# INPUT VALIDATION
# ============================================================================

class TestInputValidation:
    """Tests for input validation."""

    def test_missing_input_file_raises(self, tmp_output_dir, tmp_params_dir):
        """Missing input file raises PipelineError."""
        pipeline = GovernedPipeline(
            as_of_date="2026-01-15",
            score_version="v1",
            output_dir=tmp_output_dir,
            input_files=[Path("/nonexistent/file.json")],
            params_dir=tmp_params_dir,
        )

        with pytest.raises(PipelineError) as exc_info:
            pipeline.initialize()

        assert exc_info.value.error_code == AuditErrorCode.MISSING_INPUT

    def test_invalid_score_version_raises(self, tmp_input_files, tmp_output_dir, tmp_params_dir):
        """Invalid score version raises PipelineError."""
        pipeline = GovernedPipeline(
            as_of_date="2026-01-15",
            score_version="v999",  # Invalid
            output_dir=tmp_output_dir,
            input_files=tmp_input_files,
            params_dir=tmp_params_dir,
        )

        with pytest.raises(PipelineError) as exc_info:
            pipeline.initialize()

        assert exc_info.value.error_code == AuditErrorCode.PARAMS_MISSING

    def test_validate_input_schema_with_missing_fields(self, basic_pipeline):
        """Schema validation catches missing fields."""
        basic_pipeline.initialize()

        # Set up mapping with required fields
        basic_pipeline.mapping = {
            "source_schemas": {
                "market_data": {
                    "required_fields": ["ticker", "price", "volume"]
                }
            }
        }

        # Data missing required fields
        data = [{"ticker": "ACME"}]  # Missing price, volume

        with pytest.raises(PipelineError) as exc_info:
            basic_pipeline.validate_input_schema(data, "market_data")

        assert exc_info.value.error_code == AuditErrorCode.SCHEMA_MISMATCH

    def test_validate_input_schema_passes_with_all_fields(self, basic_pipeline):
        """Schema validation passes when all fields present."""
        basic_pipeline.initialize()

        basic_pipeline.mapping = {
            "source_schemas": {
                "market_data": {
                    "required_fields": ["ticker", "price"]
                }
            }
        }

        data = [{"ticker": "ACME", "price": 50.0, "extra": "ok"}]

        # Should not raise
        basic_pipeline.validate_input_schema(data, "market_data")

    def test_validate_input_schema_skips_unknown_source(self, basic_pipeline):
        """Schema validation skips unknown source types."""
        basic_pipeline.initialize()
        basic_pipeline.mapping = {"source_schemas": {}}

        # Should not raise even with empty data
        basic_pipeline.validate_input_schema({}, "unknown_source")


# ============================================================================
# OUTPUT WRITING
# ============================================================================

class TestOutputWriting:
    """Tests for output writing."""

    def test_write_output_creates_file(self, basic_pipeline):
        """Write output creates file."""
        basic_pipeline.initialize()

        result = basic_pipeline.write_output(
            data={"scores": [{"ticker": "ACME", "score": 85}]},
            filename="scores.json",
        )

        assert Path(result["path"]).exists()
        assert "sha256" in result

    def test_write_output_includes_governance(self, basic_pipeline):
        """Written output includes governance metadata."""
        basic_pipeline.initialize()

        result = basic_pipeline.write_output(
            data={"scores": []},
            filename="scores.json",
        )

        with open(result["path"]) as f:
            loaded = json.load(f)

        assert "_governance" in loaded
        assert loaded["_governance"]["run_id"] == basic_pipeline.run_id

    def test_write_output_tracks_outputs(self, basic_pipeline):
        """Written outputs are tracked."""
        basic_pipeline.initialize()

        assert len(basic_pipeline.outputs) == 0

        basic_pipeline.write_output({"data": 1}, "output1.json")
        assert len(basic_pipeline.outputs) == 1

        basic_pipeline.write_output({"data": 2}, "output2.json")
        assert len(basic_pipeline.outputs) == 2

    def test_write_output_with_custom_schema_version(self, basic_pipeline):
        """Write output accepts custom schema version."""
        basic_pipeline.initialize()

        result = basic_pipeline.write_output(
            data={"scores": []},
            filename="scores.json",
            schema_version="2.0.0",
        )

        with open(result["path"]) as f:
            loaded = json.load(f)

        assert loaded["_governance"]["schema_version"] == "2.0.0"


# ============================================================================
# STAGE LOGGING
# ============================================================================

class TestStageLogging:
    """Tests for stage logging."""

    def test_log_stage_success(self, basic_pipeline):
        """Log successful stage."""
        basic_pipeline.initialize()

        # Should not raise
        basic_pipeline.log_stage(
            stage=AuditStage.LOAD,
            inputs=None,
            outputs=None,
        )

    def test_log_failure(self, basic_pipeline):
        """Log failed stage."""
        basic_pipeline.initialize()

        # Should not raise
        basic_pipeline.log_failure(
            stage=AuditStage.SCORE,
            error_code=AuditErrorCode.VALIDATION_ERROR,
            error_message="Test error",
        )


# ============================================================================
# FINALIZATION
# ============================================================================

class TestFinalization:
    """Tests for pipeline finalization."""

    def test_finalize_creates_audit_log(self, basic_pipeline):
        """Finalize creates audit log file."""
        basic_pipeline.initialize()
        basic_pipeline.write_output({"data": "test"}, "test.json")

        audit_path = basic_pipeline.finalize(success=True)

        assert Path(audit_path).exists()

    def test_finalize_requires_init(self, tmp_input_files, tmp_output_dir, tmp_params_dir):
        """Finalize requires initialization."""
        pipeline = GovernedPipeline(
            as_of_date="2026-01-15",
            score_version="v1",
            output_dir=tmp_output_dir,
            input_files=tmp_input_files,
            params_dir=tmp_params_dir,
        )

        with pytest.raises(RuntimeError):
            pipeline.finalize()

    def test_finalize_tracks_all_outputs(self, basic_pipeline):
        """Finalize includes all outputs in audit."""
        basic_pipeline.initialize()
        basic_pipeline.write_output({"data": 1}, "output1.json")
        basic_pipeline.write_output({"data": 2}, "output2.json")

        audit_path = basic_pipeline.finalize(success=True)

        # Read audit log and verify outputs tracked
        with open(audit_path) as f:
            lines = f.readlines()

        # Should have multiple log entries
        assert len(lines) > 1


# ============================================================================
# RUN METADATA
# ============================================================================

class TestRunMetadata:
    """Tests for run metadata."""

    def test_get_run_metadata(self, basic_pipeline):
        """Get run metadata returns expected fields."""
        basic_pipeline.initialize()

        metadata = basic_pipeline.get_run_metadata()

        assert metadata["as_of_date"] == "2026-01-15"
        assert metadata["score_version"] == "v1"
        assert metadata["run_id"] == basic_pipeline.run_id
        assert metadata["parameters_hash"] == basic_pipeline.parameters_hash
        assert "pipeline_version" in metadata
        assert "schema_version" in metadata
        assert "environment" in metadata

    def test_get_run_metadata_includes_environment(self, basic_pipeline):
        """Run metadata includes environment fingerprint."""
        basic_pipeline.initialize()

        metadata = basic_pipeline.get_run_metadata()

        assert "platform" in metadata["environment"]
        assert "python_version" in metadata["environment"]


# ============================================================================
# PIPELINE ERROR
# ============================================================================

class TestPipelineError:
    """Tests for PipelineError class."""

    def test_error_has_code(self):
        """Error has error code."""
        error = PipelineError(AuditErrorCode.MISSING_INPUT, "Test message")
        assert error.error_code == AuditErrorCode.MISSING_INPUT

    def test_error_has_message(self):
        """Error has message."""
        error = PipelineError(AuditErrorCode.MISSING_INPUT, "Test message")
        assert str(error) == "Test message"

    def test_error_is_exception(self):
        """PipelineError is an Exception."""
        error = PipelineError(AuditErrorCode.MISSING_INPUT, "Test")
        assert isinstance(error, Exception)


# ============================================================================
# DETERMINISM
# ============================================================================

class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_run_id_deterministic(self, tmp_input_files, tmp_params_dir):
        """Same inputs produce same run_id."""
        run_ids = []

        for i in range(3):
            output_dir = Path(tempfile.mkdtemp())
            pipeline = GovernedPipeline(
                as_of_date="2026-01-15",
                score_version="v1",
                output_dir=output_dir,
                input_files=tmp_input_files,
                params_dir=tmp_params_dir,
            )
            pipeline.initialize()
            run_ids.append(pipeline.run_id)

        assert len(set(run_ids)) == 1

    def test_input_hashes_deterministic(self, tmp_input_files, tmp_output_dir, tmp_params_dir):
        """Input hashes are deterministic."""
        pipeline1 = GovernedPipeline(
            as_of_date="2026-01-15",
            score_version="v1",
            output_dir=tmp_output_dir / "run1",
            input_files=tmp_input_files,
            params_dir=tmp_params_dir,
        )
        pipeline1.initialize()

        pipeline2 = GovernedPipeline(
            as_of_date="2026-01-15",
            score_version="v1",
            output_dir=tmp_output_dir / "run2",
            input_files=tmp_input_files,
            params_dir=tmp_params_dir,
        )
        pipeline2.initialize()

        hashes1 = {h["path"]: h["sha256"] for h in pipeline1.input_hashes}
        hashes2 = {h["path"]: h["sha256"] for h in pipeline2.input_hashes}

        assert hashes1 == hashes2


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_empty_input_files(self, tmp_output_dir, tmp_params_dir):
        """Pipeline with no input files."""
        pipeline = GovernedPipeline(
            as_of_date="2026-01-15",
            score_version="v1",
            output_dir=tmp_output_dir,
            input_files=[],
            params_dir=tmp_params_dir,
        )

        pipeline.initialize()
        assert pipeline.input_hashes == []

    def test_special_characters_in_path(self, tmp_path, tmp_input_files, tmp_params_dir):
        """Handles special characters in output path."""
        output_dir = tmp_path / "output with spaces"

        pipeline = GovernedPipeline(
            as_of_date="2026-01-15",
            score_version="v1",
            output_dir=output_dir,
            input_files=tmp_input_files,
            params_dir=tmp_params_dir,
        )

        pipeline.initialize()
        assert output_dir.exists()

    def test_string_paths_accepted(self, tmp_input_files, tmp_output_dir, tmp_params_dir):
        """String paths are accepted."""
        pipeline = GovernedPipeline(
            as_of_date="2026-01-15",
            score_version="v1",
            output_dir=str(tmp_output_dir),
            input_files=[str(f) for f in tmp_input_files],
            params_dir=str(tmp_params_dir),
        )

        pipeline.initialize()
        assert pipeline.run_id != ""

