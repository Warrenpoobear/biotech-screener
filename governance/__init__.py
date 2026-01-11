"""
Governance Module - Audit, Lineage, and Deterministic Output

Provides:
- Canonical JSON serialization for byte-identical outputs
- SHA256 hashing for files and JSON objects
- Deterministic run_id generation
- JSONL audit log writing
- Schema version registry
- Parameters archive loading
- Adapter mapping validation

All operations are deterministic: same inputs produce identical outputs.
No timestamps, no UUIDs, no network calls.
"""

from governance.hashing import (
    hash_file,
    hash_bytes,
    hash_canonical_json,
    hash_canonical_json_short,
    compute_input_hashes,
)
from governance.canonical_json import (
    canonical_dumps,
    canonical_dump,
    validate_canonical_json,
)
from governance.run_id import compute_run_id, validate_run_id
from governance.audit_log import (
    AuditLog,
    AuditRecord,
    AuditStage,
    AuditStatus,
    AuditErrorCode,
    StageIO,
    load_audit_log,
)
from governance.schema_registry import (
    SCHEMA_VERSION,
    PIPELINE_VERSION,
    DEFAULT_SCORE_VERSION,
    SUPPORTED_SCORE_VERSIONS,
    validate_schema_version,
    validate_score_version,
    get_schema_info,
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
from governance.output_writer import (
    inject_governance_metadata,
    write_canonical_output,
    build_input_lineage,
    get_environment_fingerprint,
)

__all__ = [
    # Hashing
    "hash_file",
    "hash_bytes",
    "hash_canonical_json",
    "hash_canonical_json_short",
    "compute_input_hashes",
    # Canonical JSON
    "canonical_dumps",
    "canonical_dump",
    "validate_canonical_json",
    # Run ID
    "compute_run_id",
    "validate_run_id",
    # Audit
    "AuditLog",
    "AuditRecord",
    "AuditStage",
    "AuditStatus",
    "AuditErrorCode",
    "StageIO",
    "load_audit_log",
    # Schema
    "SCHEMA_VERSION",
    "PIPELINE_VERSION",
    "DEFAULT_SCORE_VERSION",
    "SUPPORTED_SCORE_VERSIONS",
    "validate_schema_version",
    "validate_score_version",
    "get_schema_info",
    # Params
    "load_params",
    "compute_parameters_hash",
    "save_params",
    "ParamsLoadError",
    # Mapping
    "load_mapping",
    "compute_mapping_hash",
    "save_mapping",
    "validate_source_schema",
    "MappingLoadError",
    "SchemaMismatchError",
    # Output
    "inject_governance_metadata",
    "write_canonical_output",
    "build_input_lineage",
    "get_environment_fingerprint",
]
