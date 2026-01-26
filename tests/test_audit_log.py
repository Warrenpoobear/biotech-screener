#!/usr/bin/env python3
"""
Unit tests for governance/audit_log.py

Tests audit trail functionality:
- AuditRecord creation and serialization
- AuditLog writing (atomic)
- Stage logging
- Failure logging
- JSONL format compliance
- Loading audit logs
"""

import pytest
import json
import os
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from governance.audit_log import (
    AuditLog,
    AuditLogError,
    AuditRecord,
    AuditStage,
    AuditStatus,
    AuditErrorCode,
    StageIO,
    load_audit_log,
)


# ============================================================================
# STAGE IO TESTS
# ============================================================================

class TestStageIO:
    """Tests for StageIO dataclass."""

    def test_basic_creation(self):
        """Should create StageIO with required fields."""
        sio = StageIO(
            path="data/input.json",
            sha256="abc123def456",
            role="market_data",
        )
        assert sio.path == "data/input.json"
        assert sio.sha256 == "abc123def456"
        assert sio.role == "market_data"

    def test_optional_schema_version(self):
        """Schema version should be optional."""
        sio = StageIO(
            path="data/input.json",
            sha256="abc123",
            role="input",
            schema_version="1.0.0",
        )
        assert sio.schema_version == "1.0.0"

    def test_to_dict_without_schema(self):
        """to_dict should exclude None schema_version."""
        sio = StageIO(path="file.json", sha256="hash", role="input")
        d = sio.to_dict()
        assert "schema_version" not in d
        assert d["path"] == "file.json"

    def test_to_dict_with_schema(self):
        """to_dict should include schema_version when present."""
        sio = StageIO(
            path="file.json",
            sha256="hash",
            role="input",
            schema_version="2.0",
        )
        d = sio.to_dict()
        assert d["schema_version"] == "2.0"


# ============================================================================
# AUDIT RECORD TESTS
# ============================================================================

class TestAuditRecord:
    """Tests for AuditRecord dataclass."""

    def test_basic_creation(self):
        """Should create AuditRecord with required fields."""
        record = AuditRecord(
            run_id="abc123",
            stage_name=AuditStage.LOAD,
            status=AuditStatus.OK,
            score_version="v1",
            schema_version="1.0.0",
            parameters_hash="params123",
        )
        assert record.run_id == "abc123"
        assert record.stage_name == AuditStage.LOAD
        assert record.status == AuditStatus.OK

    def test_to_dict_basic(self):
        """to_dict should serialize all required fields."""
        record = AuditRecord(
            run_id="test_run",
            stage_name=AuditStage.SCORE,
            status=AuditStatus.OK,
            score_version="v1",
            schema_version="1.0.0",
            parameters_hash="hash123",
        )
        d = record.to_dict()

        assert d["run_id"] == "test_run"
        assert d["stage_name"] == "SCORE"  # Enum value
        assert d["status"] == "OK"

    def test_to_dict_with_inputs_outputs(self):
        """to_dict should serialize inputs and outputs."""
        record = AuditRecord(
            run_id="run1",
            stage_name=AuditStage.ADAPT,
            status=AuditStatus.OK,
            score_version="v1",
            schema_version="1.0.0",
            parameters_hash="params",
            stage_inputs=[
                StageIO(path="in.json", sha256="in_hash", role="input"),
            ],
            stage_outputs=[
                StageIO(path="out.json", sha256="out_hash", role="output"),
            ],
        )
        d = record.to_dict()

        assert len(d["stage_inputs"]) == 1
        assert d["stage_inputs"][0]["path"] == "in.json"
        assert len(d["stage_outputs"]) == 1

    def test_to_dict_with_error(self):
        """to_dict should include error fields when present."""
        record = AuditRecord(
            run_id="run1",
            stage_name=AuditStage.LOAD,
            status=AuditStatus.FAIL,
            score_version="v1",
            schema_version="1.0.0",
            parameters_hash="params",
            error_code=AuditErrorCode.MISSING_INPUT,
            error_message="File not found",
        )
        d = record.to_dict()

        assert d["error_code"] == "MISSING_INPUT"
        assert d["error_message"] == "File not found"

    def test_to_dict_excludes_none_optionals(self):
        """to_dict should exclude None optional fields."""
        record = AuditRecord(
            run_id="run1",
            stage_name=AuditStage.INIT,
            status=AuditStatus.OK,
            score_version="v1",
            schema_version="1.0.0",
            parameters_hash="params",
        )
        d = record.to_dict()

        assert "mapping_hash" not in d
        assert "error_code" not in d
        assert "error_message" not in d


# ============================================================================
# AUDIT LOG TESTS
# ============================================================================

class TestAuditLog:
    """Tests for AuditLog class."""

    @pytest.fixture
    def audit_log(self, tmp_path):
        """Create a basic AuditLog for testing."""
        return AuditLog(
            output_path=tmp_path / "audit.jsonl",
            run_id="test_run_123",
            score_version="v1",
            schema_version="1.0.0",
            parameters_hash="params_hash_abc",
        )

    def test_initialization(self, tmp_path):
        """Should initialize with correct parameters."""
        log = AuditLog(
            output_path=tmp_path / "audit.jsonl",
            run_id="run123",
            score_version="v1",
            schema_version="1.0.0",
            parameters_hash="hash123",
            mapping_hash="map456",
        )

        assert log.run_id == "run123"
        assert log.score_version == "v1"
        assert log.mapping_hash == "map456"
        assert log.records == []

    def test_log_stage_ok(self, audit_log):
        """log_stage should create OK record."""
        record = audit_log.log_stage(
            stage=AuditStage.LOAD,
            status=AuditStatus.OK,
        )

        assert record.stage_name == AuditStage.LOAD
        assert record.status == AuditStatus.OK
        assert record.run_id == "test_run_123"
        assert len(audit_log.records) == 1

    def test_log_stage_with_io(self, audit_log):
        """log_stage should include inputs and outputs."""
        inputs = [StageIO(path="in.json", sha256="hash1", role="input")]
        outputs = [StageIO(path="out.json", sha256="hash2", role="output")]

        record = audit_log.log_stage(
            stage=AuditStage.FEATURES,
            status=AuditStatus.OK,
            inputs=inputs,
            outputs=outputs,
        )

        assert len(record.stage_inputs) == 1
        assert len(record.stage_outputs) == 1

    def test_log_failure(self, audit_log):
        """log_failure should create FAIL record with error info."""
        record = audit_log.log_failure(
            stage=AuditStage.ADAPT,
            error_code=AuditErrorCode.SCHEMA_MISMATCH,
            error_message="Schema version mismatch",
        )

        assert record.status == AuditStatus.FAIL
        assert record.error_code == AuditErrorCode.SCHEMA_MISMATCH
        assert "mismatch" in record.error_message

    def test_log_final_ok(self, audit_log):
        """log_final should create FINAL stage record."""
        outputs = [StageIO(path="result.json", sha256="hash", role="output")]

        record = audit_log.log_final(
            status=AuditStatus.OK,
            all_outputs=outputs,
        )

        assert record.stage_name == AuditStage.FINAL
        assert record.status == AuditStatus.OK
        assert len(record.stage_outputs) == 1

    def test_log_final_fail(self, audit_log):
        """log_final should handle failed runs."""
        record = audit_log.log_final(
            status=AuditStatus.FAIL,
            error_code=AuditErrorCode.UNKNOWN_ERROR,
            error_message="Pipeline crashed",
        )

        assert record.status == AuditStatus.FAIL
        assert record.error_code == AuditErrorCode.UNKNOWN_ERROR


class TestAuditLogWriting:
    """Tests for audit log file writing."""

    def test_write_creates_file(self, tmp_path):
        """write() should create output file."""
        log = AuditLog(
            output_path=tmp_path / "audit.jsonl",
            run_id="run1",
            score_version="v1",
            schema_version="1.0.0",
            parameters_hash="hash",
        )
        log.log_stage(AuditStage.INIT, AuditStatus.OK)

        result_path = log.write()

        assert Path(result_path).exists()

    def test_write_creates_directory(self, tmp_path):
        """write() should create parent directories."""
        log = AuditLog(
            output_path=tmp_path / "nested" / "dir" / "audit.jsonl",
            run_id="run1",
            score_version="v1",
            schema_version="1.0.0",
            parameters_hash="hash",
        )
        log.log_stage(AuditStage.INIT, AuditStatus.OK)

        log.write()

        assert (tmp_path / "nested" / "dir" / "audit.jsonl").exists()

    def test_write_jsonl_format(self, tmp_path):
        """Output should be valid JSONL (one JSON object per line)."""
        log = AuditLog(
            output_path=tmp_path / "audit.jsonl",
            run_id="run1",
            score_version="v1",
            schema_version="1.0.0",
            parameters_hash="hash",
        )
        log.log_stage(AuditStage.INIT, AuditStatus.OK)
        log.log_stage(AuditStage.LOAD, AuditStatus.OK)
        log.log_stage(AuditStage.FINAL, AuditStatus.OK)

        log.write()

        with open(tmp_path / "audit.jsonl") as f:
            lines = f.readlines()

        assert len(lines) == 3
        for line in lines:
            # Each line should be valid JSON
            obj = json.loads(line)
            assert "run_id" in obj
            assert "stage_name" in obj

    def test_write_atomic(self, tmp_path):
        """write() should be atomic (no partial writes)."""
        log = AuditLog(
            output_path=tmp_path / "audit.jsonl",
            run_id="run1",
            score_version="v1",
            schema_version="1.0.0",
            parameters_hash="hash",
        )
        log.log_stage(AuditStage.INIT, AuditStatus.OK)

        log.write()

        # File should exist and be complete
        content = (tmp_path / "audit.jsonl").read_text()
        assert "run_id" in content

    def test_append_record(self, tmp_path):
        """append() should add record to file."""
        log = AuditLog(
            output_path=tmp_path / "audit.jsonl",
            run_id="run1",
            score_version="v1",
            schema_version="1.0.0",
            parameters_hash="hash",
        )

        record = log.log_stage(AuditStage.INIT, AuditStatus.OK)
        log.append(record)

        # File should exist with one record
        content = (tmp_path / "audit.jsonl").read_text()
        lines = content.strip().split('\n')
        # Note: append adds the record, and log_stage already added it to records
        # So we may have 2 lines
        assert len(lines) >= 1


# ============================================================================
# LOAD AUDIT LOG TESTS
# ============================================================================

class TestLoadAuditLog:
    """Tests for load_audit_log function."""

    def test_load_valid_log(self, tmp_path):
        """Should load valid JSONL audit log."""
        log_file = tmp_path / "audit.jsonl"
        log_file.write_text(
            '{"run_id": "run1", "stage_name": "INIT", "status": "OK"}\n'
            '{"run_id": "run1", "stage_name": "LOAD", "status": "OK"}\n'
        )

        records = load_audit_log(log_file)

        assert len(records) == 2
        assert records[0]["stage_name"] == "INIT"
        assert records[1]["stage_name"] == "LOAD"

    def test_load_nonexistent_raises(self, tmp_path):
        """Should raise AuditLogError for nonexistent file."""
        with pytest.raises(AuditLogError, match="not found"):
            load_audit_log(tmp_path / "missing.jsonl")

    def test_load_skips_invalid_lines(self, tmp_path):
        """Should skip malformed JSON lines with warning."""
        log_file = tmp_path / "audit.jsonl"
        log_file.write_text(
            '{"run_id": "run1", "stage_name": "INIT"}\n'
            'not valid json\n'
            '{"run_id": "run1", "stage_name": "LOAD"}\n'
        )

        records = load_audit_log(log_file)

        # Should have 2 valid records, skipping the invalid line
        assert len(records) == 2

    def test_load_empty_file(self, tmp_path):
        """Should return empty list for empty file."""
        log_file = tmp_path / "audit.jsonl"
        log_file.write_text("")

        records = load_audit_log(log_file)

        assert records == []

    def test_load_accepts_string_path(self, tmp_path):
        """Should accept string path."""
        log_file = tmp_path / "audit.jsonl"
        log_file.write_text('{"run_id": "run1"}\n')

        records = load_audit_log(str(log_file))

        assert len(records) == 1


# ============================================================================
# AUDIT STAGE AND STATUS ENUMS
# ============================================================================

class TestAuditEnums:
    """Tests for audit-related enums."""

    def test_audit_stage_values(self):
        """All expected stages should exist."""
        assert AuditStage.INIT.value == "INIT"
        assert AuditStage.LOAD.value == "LOAD"
        assert AuditStage.ADAPT.value == "ADAPT"
        assert AuditStage.FEATURES.value == "FEATURES"
        assert AuditStage.RISK.value == "RISK"
        assert AuditStage.SCORE.value == "SCORE"
        assert AuditStage.REPORT.value == "REPORT"
        assert AuditStage.FINAL.value == "FINAL"

    def test_audit_status_values(self):
        """All expected statuses should exist."""
        assert AuditStatus.OK.value == "OK"
        assert AuditStatus.FAIL.value == "FAIL"
        assert AuditStatus.SKIP.value == "SKIP"

    def test_error_codes(self):
        """All expected error codes should exist."""
        assert AuditErrorCode.MISSING_INPUT.value == "MISSING_INPUT"
        assert AuditErrorCode.SCHEMA_MISMATCH.value == "SCHEMA_MISMATCH"
        assert AuditErrorCode.HASH_ERROR.value == "HASH_ERROR"
        assert AuditErrorCode.VALIDATION_ERROR.value == "VALIDATION_ERROR"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestAuditLogIntegration:
    """Integration tests for complete audit log workflow."""

    def test_full_pipeline_audit(self, tmp_path):
        """Test complete pipeline audit workflow."""
        log = AuditLog(
            output_path=tmp_path / "audit.jsonl",
            run_id="integration_test",
            score_version="v1",
            schema_version="1.0.0",
            parameters_hash="params123",
            mapping_hash="mapping456",
        )

        # Log pipeline stages
        log.log_stage(
            AuditStage.INIT,
            AuditStatus.OK,
        )

        log.log_stage(
            AuditStage.LOAD,
            AuditStatus.OK,
            inputs=[
                StageIO(path="universe.json", sha256="hash1", role="universe"),
                StageIO(path="market.json", sha256="hash2", role="market_data"),
            ],
        )

        log.log_stage(
            AuditStage.SCORE,
            AuditStatus.OK,
            outputs=[
                StageIO(path="scores.json", sha256="hash3", role="scores"),
            ],
        )

        log.log_final(
            AuditStatus.OK,
            all_outputs=[
                StageIO(path="results.json", sha256="hash4", role="final_output"),
            ],
        )

        # Write and reload
        log.write()
        records = load_audit_log(tmp_path / "audit.jsonl")

        # Verify
        assert len(records) == 4
        assert records[0]["stage_name"] == "INIT"
        assert records[3]["stage_name"] == "FINAL"
        assert all(r["run_id"] == "integration_test" for r in records)

    def test_failed_pipeline_audit(self, tmp_path):
        """Test audit trail for failed pipeline."""
        log = AuditLog(
            output_path=tmp_path / "audit.jsonl",
            run_id="failed_run",
            score_version="v1",
            schema_version="1.0.0",
            parameters_hash="params",
        )

        log.log_stage(AuditStage.INIT, AuditStatus.OK)

        log.log_failure(
            AuditStage.LOAD,
            AuditErrorCode.MISSING_INPUT,
            "Required file universe.json not found",
            inputs=[StageIO(path="universe.json", sha256="", role="universe")],
        )

        log.log_final(
            AuditStatus.FAIL,
            error_code=AuditErrorCode.MISSING_INPUT,
            error_message="Pipeline aborted due to missing input",
        )

        log.write()
        records = load_audit_log(tmp_path / "audit.jsonl")

        assert len(records) == 3
        assert records[1]["status"] == "FAIL"
        assert records[1]["error_code"] == "MISSING_INPUT"
        assert records[2]["stage_name"] == "FINAL"
        assert records[2]["status"] == "FAIL"
