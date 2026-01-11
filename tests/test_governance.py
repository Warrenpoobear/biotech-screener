"""
Tests for Governance Module

Tests:
1. Canonical JSON determinism
2. Hashing consistency
3. Run ID stability
4. Audit log structure
5. Params loading
6. Mapping validation
"""

import json
import math
import tempfile
from decimal import Decimal
from pathlib import Path

import pytest

from governance.canonical_json import (
    canonical_dumps,
    canonical_dump,
    validate_canonical_json,
    CanonicalJSONEncoder,
)
from governance.hashing import (
    hash_bytes,
    hash_file,
    hash_canonical_json,
    hash_canonical_json_short,
    compute_input_hashes,
)
from governance.run_id import compute_run_id, validate_run_id
from governance.audit_log import (
    AuditLog,
    AuditStage,
    AuditStatus,
    AuditErrorCode,
    StageIO,
    load_audit_log,
)
from governance.params_loader import (
    load_params,
    compute_parameters_hash,
    save_params,
    ParamsLoadError,
)
from governance.mapping_loader import (
    load_mapping,
    compute_mapping_hash,
    save_mapping,
    validate_source_schema,
    MappingLoadError,
    SchemaMismatchError,
)


# =============================================================================
# CANONICAL JSON TESTS
# =============================================================================

class TestCanonicalJSON:
    """Tests for canonical JSON serialization."""

    def test_sorted_keys(self):
        """Dict keys are sorted."""
        data = {"z": 1, "a": 2, "m": 3}
        result = canonical_dumps(data)
        # Keys should appear in order: a, m, z
        assert result.index('"a"') < result.index('"m"') < result.index('"z"')

    def test_nested_sorted_keys(self):
        """Nested dict keys are sorted."""
        data = {"outer": {"z": 1, "a": 2}}
        result = canonical_dumps(data)
        assert result.index('"a"') < result.index('"z"')

    def test_float_formatting(self):
        """Floats are formatted consistently."""
        # Integer-valued floats become ints
        assert '"value": 5' in canonical_dumps({"value": 5.0})

        # Regular floats keep decimal
        result = canonical_dumps({"value": 3.14159})
        assert "3.14159" in result

    def test_nan_rejected(self):
        """NaN values raise ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            canonical_dumps({"value": float('nan')})

    def test_inf_rejected(self):
        """Infinity values raise ValueError."""
        with pytest.raises(ValueError, match="Infinity"):
            canonical_dumps({"value": float('inf')})

        with pytest.raises(ValueError, match="Infinity"):
            canonical_dumps({"value": float('-inf')})

    def test_trailing_newline(self):
        """Output ends with newline."""
        result = canonical_dumps({"key": "value"})
        assert result.endswith('\n')

    def test_decimal_handling(self):
        """Decimal values are handled."""
        data = {"value": Decimal("123.456")}
        result = canonical_dumps(data)
        assert "123.456" in result

    def test_determinism(self):
        """Same input produces same output."""
        data = {"b": [1, 2, 3], "a": {"z": 1, "y": 2}}

        results = [canonical_dumps(data) for _ in range(10)]
        assert len(set(results)) == 1

    def test_list_order_preserved(self):
        """List order is preserved (not sorted)."""
        data = {"items": [3, 1, 2]}
        result = canonical_dumps(data)
        # Items should appear in original order
        assert result.index("3") < result.index("1") < result.index("2")


# =============================================================================
# HASHING TESTS
# =============================================================================

class TestHashing:
    """Tests for hashing functions."""

    def test_hash_bytes_consistent(self):
        """Same bytes produce same hash."""
        data = b"test data"
        hashes = [hash_bytes(data) for _ in range(5)]
        assert len(set(hashes)) == 1

    def test_hash_bytes_different_input(self):
        """Different bytes produce different hash."""
        hash1 = hash_bytes(b"data1")
        hash2 = hash_bytes(b"data2")
        assert hash1 != hash2

    def test_hash_file(self, tmp_path):
        """File hashing works correctly."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("test content")

        hash1 = hash_file(file_path)
        hash2 = hash_file(file_path)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex

    def test_hash_canonical_json(self):
        """JSON hashing is deterministic."""
        data = {"b": 2, "a": 1}
        hashes = [hash_canonical_json(data) for _ in range(5)]
        assert len(set(hashes)) == 1

    def test_hash_canonical_json_short(self):
        """Short hash is truncated correctly."""
        data = {"key": "value"}
        short = hash_canonical_json_short(data, length=16)
        full = hash_canonical_json(data)

        assert len(short) == 16
        assert full.startswith(short)

    def test_compute_input_hashes(self, tmp_path):
        """Input hashes are computed and sorted."""
        # Create test files
        (tmp_path / "b.json").write_text('{"key": "b"}')
        (tmp_path / "a.json").write_text('{"key": "a"}')

        hashes = compute_input_hashes([
            tmp_path / "b.json",
            tmp_path / "a.json",
        ])

        # Should be sorted by path
        assert hashes[0]["path"] == "a.json"
        assert hashes[1]["path"] == "b.json"


# =============================================================================
# RUN ID TESTS
# =============================================================================

class TestRunId:
    """Tests for run ID generation."""

    def test_run_id_deterministic(self):
        """Same inputs produce same run_id."""
        input_hashes = [{"path": "data.json", "sha256": "abc123"}]

        ids = [
            compute_run_id(
                as_of_date="2024-01-15",
                score_version="v1",
                parameters_hash="params123",
                input_hashes=input_hashes,
                pipeline_version="1.0.0",
            )
            for _ in range(5)
        ]

        assert len(set(ids)) == 1

    def test_run_id_changes_with_date(self):
        """Different date produces different run_id."""
        input_hashes = [{"path": "data.json", "sha256": "abc123"}]

        id1 = compute_run_id(
            as_of_date="2024-01-15",
            score_version="v1",
            parameters_hash="params123",
            input_hashes=input_hashes,
            pipeline_version="1.0.0",
        )

        id2 = compute_run_id(
            as_of_date="2024-01-16",
            score_version="v1",
            parameters_hash="params123",
            input_hashes=input_hashes,
            pipeline_version="1.0.0",
        )

        assert id1 != id2

    def test_run_id_changes_with_params(self):
        """Different params hash produces different run_id."""
        input_hashes = [{"path": "data.json", "sha256": "abc123"}]

        id1 = compute_run_id(
            as_of_date="2024-01-15",
            score_version="v1",
            parameters_hash="params123",
            input_hashes=input_hashes,
            pipeline_version="1.0.0",
        )

        id2 = compute_run_id(
            as_of_date="2024-01-15",
            score_version="v1",
            parameters_hash="params456",
            input_hashes=input_hashes,
            pipeline_version="1.0.0",
        )

        assert id1 != id2

    def test_run_id_length(self):
        """Run ID has correct length."""
        run_id = compute_run_id(
            as_of_date="2024-01-15",
            score_version="v1",
            parameters_hash="params123",
            input_hashes=[],
            pipeline_version="1.0.0",
            length=16,
        )
        assert len(run_id) == 16

    def test_validate_run_id(self):
        """Run ID validation works."""
        assert validate_run_id("abc123def456") is True
        assert validate_run_id("not-hex!") is False
        assert validate_run_id("") is False


# =============================================================================
# AUDIT LOG TESTS
# =============================================================================

class TestAuditLog:
    """Tests for audit log."""

    def test_audit_log_write(self, tmp_path):
        """Audit log writes correctly."""
        audit_path = tmp_path / "audit.jsonl"

        log = AuditLog(
            output_path=audit_path,
            run_id="test123",
            score_version="v1",
            schema_version="1.0.0",
            parameters_hash="params456",
        )

        log.log_stage(
            stage=AuditStage.INIT,
            status=AuditStatus.OK,
        )

        log.log_stage(
            stage=AuditStage.SCORE,
            status=AuditStatus.OK,
            outputs=[StageIO(path="output.json", sha256="hash123", role="output")],
        )

        log.write()

        # Read back
        records = load_audit_log(audit_path)
        assert len(records) == 2
        assert records[0]["stage_name"] == "INIT"
        assert records[1]["stage_name"] == "SCORE"

    def test_audit_log_failure(self, tmp_path):
        """Audit log records failures."""
        audit_path = tmp_path / "audit.jsonl"

        log = AuditLog(
            output_path=audit_path,
            run_id="test123",
            score_version="v1",
            schema_version="1.0.0",
            parameters_hash="params456",
        )

        log.log_failure(
            stage=AuditStage.LOAD,
            error_code=AuditErrorCode.MISSING_INPUT,
            error_message="File not found: data.json",
        )

        log.write()

        records = load_audit_log(audit_path)
        assert records[0]["status"] == "FAIL"
        assert records[0]["error_code"] == "MISSING_INPUT"

    def test_audit_record_deterministic(self, tmp_path):
        """Audit records are deterministic."""
        def write_log():
            audit_path = tmp_path / f"audit_{id(write_log)}.jsonl"
            log = AuditLog(
                output_path=audit_path,
                run_id="test123",
                score_version="v1",
                schema_version="1.0.0",
                parameters_hash="params456",
            )
            log.log_stage(AuditStage.INIT, AuditStatus.OK)
            log.write()
            return audit_path.read_text()

        # Multiple writes should be identical
        contents = [write_log() for _ in range(3)]
        assert len(set(contents)) == 1


# =============================================================================
# PARAMS LOADER TESTS
# =============================================================================

class TestParamsLoader:
    """Tests for params loading."""

    def test_load_params(self, tmp_path):
        """Params load correctly."""
        params_dir = tmp_path / "params"
        params_dir.mkdir()

        params = {"threshold": 100, "enabled": True}
        params_file = params_dir / "v1.json"
        params_file.write_text(json.dumps(params))

        loaded, hash_val = load_params("v1", params_dir)

        assert loaded == params
        assert len(hash_val) == 16

    def test_load_params_missing(self, tmp_path):
        """Missing params file raises error."""
        with pytest.raises(ParamsLoadError, match="not found"):
            load_params("v99", tmp_path)

    def test_params_hash_deterministic(self):
        """Same params produce same hash."""
        params = {"b": 2, "a": 1}
        hashes = [compute_parameters_hash(params) for _ in range(5)]
        assert len(set(hashes)) == 1

    def test_params_hash_changes(self):
        """Different params produce different hash."""
        hash1 = compute_parameters_hash({"value": 1})
        hash2 = compute_parameters_hash({"value": 2})
        assert hash1 != hash2


# =============================================================================
# MAPPING LOADER TESTS
# =============================================================================

class TestMappingLoader:
    """Tests for mapping loading."""

    def test_load_mapping(self, tmp_path):
        """Mapping loads correctly."""
        adapters_dir = tmp_path / "adapters"
        source_dir = adapters_dir / "test_source"
        source_dir.mkdir(parents=True)

        mapping = {
            "required_fields": ["ticker"],
            "field_mappings": {"old_name": "new_name"},
        }
        mapping_file = source_dir / "mapping_v1.json"
        mapping_file.write_text(json.dumps(mapping))

        loaded, hash_val = load_mapping("test_source", "v1", adapters_dir)

        assert loaded == mapping
        assert len(hash_val) == 16

    def test_load_mapping_missing(self, tmp_path):
        """Missing mapping file raises error."""
        with pytest.raises(MappingLoadError, match="not found"):
            load_mapping("nonexistent", "v1", tmp_path)

    def test_validate_source_schema_pass(self):
        """Schema validation passes for valid data."""
        data = [{"ticker": "AAPL", "price": 150}]
        mapping = {"required_fields": ["ticker"]}

        valid, missing = validate_source_schema(data, mapping, "test")
        assert valid is True
        assert missing == []

    def test_validate_source_schema_fail(self):
        """Schema validation fails for missing fields."""
        data = [{"price": 150}]
        mapping = {"required_fields": ["ticker"]}

        with pytest.raises(SchemaMismatchError) as exc:
            validate_source_schema(data, mapping, "test")

        assert "ticker" in exc.value.missing_fields


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestGovernanceIntegration:
    """Integration tests for governance module."""

    def test_end_to_end_determinism(self, tmp_path):
        """Full pipeline produces identical outputs."""
        # Create input files
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        market_data = [{"ticker": "AAPL", "price": 150}]
        (data_dir / "market.json").write_text(json.dumps(market_data))

        # Create params
        params_dir = tmp_path / "params"
        params_dir.mkdir()
        params = {"threshold": 100}
        (params_dir / "v1.json").write_text(json.dumps(params))

        # Compute run_id twice
        input_hashes = compute_input_hashes([data_dir / "market.json"])
        params_loaded, params_hash = load_params("v1", params_dir)

        run_ids = [
            compute_run_id(
                as_of_date="2024-01-15",
                score_version="v1",
                parameters_hash=params_hash,
                input_hashes=input_hashes,
                pipeline_version="1.0.0",
            )
            for _ in range(3)
        ]

        assert len(set(run_ids)) == 1

    def test_param_change_changes_output(self, tmp_path):
        """Parameter change produces different run_id."""
        input_hashes = [{"path": "data.json", "sha256": "abc123"}]

        # Original params
        params1 = {"threshold": 100}
        hash1 = compute_parameters_hash(params1)
        run_id1 = compute_run_id(
            as_of_date="2024-01-15",
            score_version="v1",
            parameters_hash=hash1,
            input_hashes=input_hashes,
            pipeline_version="1.0.0",
        )

        # Changed params
        params2 = {"threshold": 200}
        hash2 = compute_parameters_hash(params2)
        run_id2 = compute_run_id(
            as_of_date="2024-01-15",
            score_version="v1",
            parameters_hash=hash2,
            input_hashes=input_hashes,
            pipeline_version="1.0.0",
        )

        assert hash1 != hash2
        assert run_id1 != run_id2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
