"""
Governed Pipeline Runner

Orchestrates pipeline execution with:
- Deterministic run_id generation
- Input validation and hashing
- Stage-by-stage audit logging
- Canonical output writing
- Fail-closed error handling
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable

from governance.hashing import hash_file, compute_input_hashes
from governance.canonical_json import canonical_dumps
from governance.run_id import compute_run_id
from governance.audit_log import (
    AuditLog,
    AuditStage,
    AuditStatus,
    AuditErrorCode,
    StageIO,
)
from governance.params_loader import load_params, ParamsLoadError
from governance.mapping_loader import (
    load_mapping,
    validate_source_schema,
    MappingLoadError,
    SchemaMismatchError,
)
from governance.schema_registry import (
    PIPELINE_VERSION,
    SCHEMA_VERSION,
    validate_score_version,
)
from governance.output_writer import (
    write_canonical_output,
    build_input_lineage,
    get_environment_fingerprint,
)


class PipelineError(Exception):
    """Pipeline execution error."""

    def __init__(self, error_code: AuditErrorCode, message: str):
        self.error_code = error_code
        super().__init__(message)


class GovernedPipeline:
    """
    Pipeline runner with full governance and audit trail.
    """

    def __init__(
        self,
        as_of_date: str,
        score_version: str,
        output_dir: Union[str, Path],
        input_files: List[Union[str, Path]],
        params_dir: Optional[Union[str, Path]] = None,
        adapters_dir: Optional[Union[str, Path]] = None,
        dry_run: bool = False,
    ):
        """
        Initialize governed pipeline.

        Args:
            as_of_date: Analysis date (YYYY-MM-DD)
            score_version: Score version (e.g., "v1")
            output_dir: Directory for outputs
            input_files: List of input file paths
            params_dir: Optional params archive directory
            adapters_dir: Optional adapters directory
            dry_run: If True, write to shadow dir
        """
        self.as_of_date = as_of_date
        self.score_version = score_version
        self.output_dir = Path(output_dir)
        self.input_files = [Path(f) for f in input_files]
        self.params_dir = params_dir
        self.adapters_dir = adapters_dir
        self.dry_run = dry_run

        # Will be populated during init
        self.params: Dict[str, Any] = {}
        self.parameters_hash: str = ""
        self.mapping: Dict[str, Any] = {}
        self.mapping_hash: str = ""
        self.input_hashes: List[Dict[str, str]] = []
        self.run_id: str = ""
        self.audit_log: Optional[AuditLog] = None

        # Track outputs
        self.outputs: List[Dict[str, str]] = []

    def initialize(self) -> None:
        """
        Initialize pipeline: load params, compute hashes, create audit log.

        Raises:
            PipelineError: If initialization fails
        """
        # Validate score version
        valid, msg = validate_score_version(self.score_version)
        if not valid:
            raise PipelineError(AuditErrorCode.PARAMS_MISSING, msg)

        # Load parameters
        try:
            self.params, self.parameters_hash = load_params(
                self.score_version,
                self.params_dir,
            )
        except ParamsLoadError as e:
            raise PipelineError(AuditErrorCode.PARAMS_MISSING, str(e))

        # Load mapping (optional - may not exist yet)
        try:
            self.mapping, self.mapping_hash = load_mapping(
                "screener",
                "v1",
                self.adapters_dir,
            )
        except MappingLoadError:
            # Mapping is optional for now
            self.mapping = {}
            self.mapping_hash = ""

        # Compute input hashes
        for path in self.input_files:
            if not path.exists():
                raise PipelineError(
                    AuditErrorCode.MISSING_INPUT,
                    f"Input file not found: {path}"
                )

        self.input_hashes = compute_input_hashes(self.input_files)

        # Compute run_id
        mapping_hashes = []
        if self.mapping_hash:
            mapping_hashes.append({"name": "screener_v1", "sha256": self.mapping_hash})

        self.run_id = compute_run_id(
            as_of_date=self.as_of_date,
            score_version=self.score_version,
            parameters_hash=self.parameters_hash,
            input_hashes=self.input_hashes,
            pipeline_version=PIPELINE_VERSION,
            mapping_hashes=mapping_hashes,
        )

        # Create output directory
        if self.dry_run:
            self.output_dir = self.output_dir / "shadow"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize audit log
        audit_path = self.output_dir / "audit.jsonl"
        self.audit_log = AuditLog(
            output_path=audit_path,
            run_id=self.run_id,
            score_version=self.score_version,
            schema_version=SCHEMA_VERSION,
            parameters_hash=self.parameters_hash,
            mapping_hash=self.mapping_hash or None,
        )

        # Log init stage
        init_inputs = [
            StageIO(path=h["path"], sha256=h["sha256"], role="input")
            for h in self.input_hashes
        ]
        self.audit_log.log_stage(
            stage=AuditStage.INIT,
            status=AuditStatus.OK,
            inputs=init_inputs,
        )

    def validate_input_schema(
        self,
        data: Any,
        source_type: str,
    ) -> None:
        """
        Validate input data against mapping schema.

        Args:
            data: Input data
            source_type: Source type (e.g., "market_data")

        Raises:
            PipelineError: If schema validation fails
        """
        if not self.mapping:
            return  # No mapping to validate against

        source_schemas = self.mapping.get("source_schemas", {})
        if source_type not in source_schemas:
            return  # No schema defined for this source

        schema = source_schemas[source_type]
        required = schema.get("required_fields", [])

        if not required:
            return

        # Check first record
        sample = data[0] if isinstance(data, list) and data else data
        if not isinstance(sample, dict):
            raise PipelineError(
                AuditErrorCode.SCHEMA_MISMATCH,
                f"Expected dict for {source_type}, got {type(sample).__name__}"
            )

        missing = [f for f in required if f not in sample]
        if missing:
            raise PipelineError(
                AuditErrorCode.SCHEMA_MISMATCH,
                f"Source '{source_type}' missing required fields: {missing}"
            )

    def log_stage(
        self,
        stage: AuditStage,
        inputs: Optional[List[StageIO]] = None,
        outputs: Optional[List[StageIO]] = None,
    ) -> None:
        """Log successful stage completion."""
        if self.audit_log:
            self.audit_log.log_stage(
                stage=stage,
                status=AuditStatus.OK,
                inputs=inputs,
                outputs=outputs,
            )

    def log_failure(
        self,
        stage: AuditStage,
        error_code: AuditErrorCode,
        error_message: str,
    ) -> None:
        """Log stage failure."""
        if self.audit_log:
            self.audit_log.log_failure(
                stage=stage,
                error_code=error_code,
                error_message=error_message,
            )

    def write_output(
        self,
        data: Dict[str, Any],
        filename: str,
        schema_version: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Write output with governance metadata.

        Args:
            data: Output data
            filename: Output filename
            schema_version: Optional schema version override

        Returns:
            Dict with path and sha256
        """
        output_path = self.output_dir / filename

        input_lineage = build_input_lineage(
            self.input_files,
            self.as_of_date,
            schema_version,
        )

        result = write_canonical_output(
            data=data,
            output_path=output_path,
            run_id=self.run_id,
            score_version=self.score_version,
            parameters_hash=self.parameters_hash,
            input_lineage=input_lineage,
            schema_version=schema_version,
        )

        self.outputs.append(result)
        return result

    def finalize(self, success: bool = True) -> str:
        """
        Finalize pipeline run.

        Args:
            success: Whether run was successful

        Returns:
            Path to audit log
        """
        if not self.audit_log:
            raise RuntimeError("Pipeline not initialized")

        output_ios = [
            StageIO(path=Path(o["path"]).name, sha256=o["sha256"], role="output")
            for o in self.outputs
        ]

        status = AuditStatus.OK if success else AuditStatus.FAIL
        self.audit_log.log_final(status=status, all_outputs=output_ios)

        return self.audit_log.write()

    def get_run_metadata(self) -> Dict[str, Any]:
        """Get run metadata for output."""
        return {
            "as_of_date": self.as_of_date,
            "dry_run": self.dry_run,
            "environment": get_environment_fingerprint(),
            "input_hashes": self.input_hashes,
            "mapping_hash": self.mapping_hash,
            "parameters_hash": self.parameters_hash,
            "pipeline_version": PIPELINE_VERSION,
            "run_id": self.run_id,
            "schema_version": SCHEMA_VERSION,
            "score_version": self.score_version,
        }
