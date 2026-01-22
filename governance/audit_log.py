"""
Audit Log Writer

Writes machine-readable audit records in JSONL format.

Each record captures:
- Stage name and status
- Input/output lineage with hashes
- Version information
- Error details (on failure)

No timestamps - deterministic records only.
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Any, Union

from governance.canonical_json import canonical_dumps

logger = logging.getLogger(__name__)


class AuditLogError(Exception):
    """Raised when audit log operations fail."""
    pass


class AuditStage(str, Enum):
    """Pipeline stages for audit tracking."""
    INIT = "INIT"
    LOAD = "LOAD"
    ADAPT = "ADAPT"
    FEATURES = "FEATURES"
    RISK = "RISK"
    SCORE = "SCORE"
    REPORT = "REPORT"
    FINAL = "FINAL"


class AuditStatus(str, Enum):
    """Audit record status."""
    OK = "OK"
    FAIL = "FAIL"
    SKIP = "SKIP"


class AuditErrorCode(str, Enum):
    """Standard error codes for audit failures."""
    MISSING_INPUT = "MISSING_INPUT"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    HASH_ERROR = "HASH_ERROR"
    PARAMS_MISSING = "PARAMS_MISSING"
    MAPPING_MISSING = "MAPPING_MISSING"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


@dataclass
class StageIO:
    """Input or output artifact reference."""
    path: str
    sha256: str
    role: str  # e.g., "market_data", "output", "parameters"
    schema_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, excluding None values."""
        d = {"path": self.path, "sha256": self.sha256, "role": self.role}
        if self.schema_version:
            d["schema_version"] = self.schema_version
        return d


@dataclass
class AuditRecord:
    """Single audit log record."""
    run_id: str
    stage_name: AuditStage
    status: AuditStatus
    score_version: str
    schema_version: str
    parameters_hash: str
    stage_inputs: List[StageIO] = field(default_factory=list)
    stage_outputs: List[StageIO] = field(default_factory=list)
    mapping_hash: Optional[str] = None
    error_code: Optional[AuditErrorCode] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        d = {
            "run_id": self.run_id,
            "stage_name": self.stage_name.value if isinstance(self.stage_name, AuditStage) else self.stage_name,
            "status": self.status.value if isinstance(self.status, AuditStatus) else self.status,
            "score_version": self.score_version,
            "schema_version": self.schema_version,
            "parameters_hash": self.parameters_hash,
            "stage_inputs": [sio.to_dict() if isinstance(sio, StageIO) else sio for sio in self.stage_inputs],
            "stage_outputs": [sio.to_dict() if isinstance(sio, StageIO) else sio for sio in self.stage_outputs],
        }

        if self.mapping_hash:
            d["mapping_hash"] = self.mapping_hash

        if self.error_code:
            d["error_code"] = self.error_code.value if isinstance(self.error_code, AuditErrorCode) else self.error_code

        if self.error_message:
            d["error_message"] = self.error_message

        return d


class AuditLog:
    """
    Audit log writer for pipeline runs.

    Writes JSONL records with deterministic formatting.
    """

    def __init__(
        self,
        output_path: Union[str, Path],
        run_id: str,
        score_version: str,
        schema_version: str,
        parameters_hash: str,
        mapping_hash: Optional[str] = None,
    ):
        """
        Initialize audit log.

        Args:
            output_path: Path to audit.jsonl file
            run_id: Deterministic run identifier
            score_version: Score version (e.g., "v1")
            schema_version: Schema version (e.g., "1.0.0")
            parameters_hash: Hash of parameters
            mapping_hash: Optional hash of adapter mappings
        """
        self.output_path = Path(output_path)
        self.run_id = run_id
        self.score_version = score_version
        self.schema_version = schema_version
        self.parameters_hash = parameters_hash
        self.mapping_hash = mapping_hash
        self.records: List[AuditRecord] = []

    def _serialize_record(self, record: AuditRecord) -> str:
        """Serialize record to canonical JSON line."""
        # Use compact canonical JSON (no indent)
        return canonical_dumps(record.to_dict(), indent=None).rstrip('\n')

    def log_stage(
        self,
        stage: AuditStage,
        status: AuditStatus,
        inputs: Optional[List[StageIO]] = None,
        outputs: Optional[List[StageIO]] = None,
        error_code: Optional[AuditErrorCode] = None,
        error_message: Optional[str] = None,
    ) -> AuditRecord:
        """
        Log a pipeline stage.

        Args:
            stage: Pipeline stage
            status: OK, FAIL, or SKIP
            inputs: List of input artifacts
            outputs: List of output artifacts
            error_code: Error code if status is FAIL
            error_message: Error message if status is FAIL

        Returns:
            The created AuditRecord
        """
        record = AuditRecord(
            run_id=self.run_id,
            stage_name=stage,
            status=status,
            score_version=self.score_version,
            schema_version=self.schema_version,
            parameters_hash=self.parameters_hash,
            stage_inputs=inputs or [],
            stage_outputs=outputs or [],
            mapping_hash=self.mapping_hash,
            error_code=error_code,
            error_message=error_message,
        )
        self.records.append(record)
        return record

    def log_failure(
        self,
        stage: AuditStage,
        error_code: AuditErrorCode,
        error_message: str,
        inputs: Optional[List[StageIO]] = None,
    ) -> AuditRecord:
        """
        Log a stage failure.

        Args:
            stage: Pipeline stage that failed
            error_code: Standard error code
            error_message: Human-readable error message
            inputs: Inputs that were attempted

        Returns:
            The created AuditRecord
        """
        return self.log_stage(
            stage=stage,
            status=AuditStatus.FAIL,
            inputs=inputs,
            error_code=error_code,
            error_message=error_message,
        )

    def log_final(
        self,
        status: AuditStatus,
        all_outputs: Optional[List[StageIO]] = None,
        error_code: Optional[AuditErrorCode] = None,
        error_message: Optional[str] = None,
    ) -> AuditRecord:
        """
        Log final run summary.

        Args:
            status: Overall run status
            all_outputs: All output artifacts from the run
            error_code: Error code if failed
            error_message: Error message if failed

        Returns:
            The created AuditRecord
        """
        return self.log_stage(
            stage=AuditStage.FINAL,
            status=status,
            outputs=all_outputs,
            error_code=error_code,
            error_message=error_message,
        )

    def write(self) -> str:
        """
        Write all records to audit log file atomically.

        Uses atomic write pattern (temp file + rename) to prevent partial writes.

        Returns:
            Path to written file

        Raises:
            AuditLogError: If write fails
        """
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            logger.error(f"Permission denied creating audit log directory: {self.output_path.parent}")
            raise AuditLogError(f"Cannot create audit log directory: {e}") from e
        except OSError as e:
            logger.error(f"Failed to create audit log directory: {e}")
            raise AuditLogError(f"Cannot create audit log directory: {e}") from e

        # Write to temp file first for atomic operation
        fd = None
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=self.output_path.parent,
                prefix='.audit_tmp_',
                suffix='.jsonl'
            )
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                fd = None  # fd is now owned by the file object
                for record in self.records:
                    f.write(self._serialize_record(record) + '\n')

            # Atomic rename
            Path(tmp_path).replace(self.output_path)
            logger.debug(f"Audit log written: {self.output_path}")
            return str(self.output_path)

        except PermissionError as e:
            logger.error(f"Permission denied writing audit log: {self.output_path}")
            raise AuditLogError(f"Cannot write audit log: {e}") from e
        except OSError as e:
            if "No space left" in str(e) or getattr(e, 'errno', 0) == 28:
                logger.error(f"Disk full writing audit log: {self.output_path}")
                raise AuditLogError("Disk full - cannot write audit log") from e
            logger.error(f"Failed to write audit log: {e}")
            raise AuditLogError(f"Cannot write audit log: {e}") from e
        finally:
            # Clean up fd if still open
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            # Clean up temp file on failure
            if tmp_path and Path(tmp_path).exists():
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def append(self, record: AuditRecord) -> None:
        """
        Append a single record to the log file.

        Args:
            record: Record to append

        Raises:
            AuditLogError: If append fails
        """
        self.records.append(record)

        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            logger.error(f"Permission denied creating audit log directory: {self.output_path.parent}")
            raise AuditLogError(f"Cannot create audit log directory: {e}") from e
        except OSError as e:
            logger.error(f"Failed to create audit log directory: {e}")
            raise AuditLogError(f"Cannot create audit log directory: {e}") from e

        try:
            with open(self.output_path, 'a', encoding='utf-8') as f:
                f.write(self._serialize_record(record) + '\n')
        except PermissionError as e:
            logger.error(f"Permission denied appending to audit log: {self.output_path}")
            raise AuditLogError(f"Cannot append to audit log: {e}") from e
        except OSError as e:
            if "No space left" in str(e) or getattr(e, 'errno', 0) == 28:
                logger.error(f"Disk full appending to audit log: {self.output_path}")
                raise AuditLogError("Disk full - cannot append to audit log") from e
            logger.error(f"Failed to append to audit log: {e}")
            raise AuditLogError(f"Cannot append to audit log: {e}") from e


def load_audit_log(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load audit log from JSONL file.

    Args:
        path: Path to audit.jsonl

    Returns:
        List of audit record dicts

    Raises:
        AuditLogError: If file cannot be read or parsed
    """
    path = Path(path)
    records = []

    if not path.exists():
        raise AuditLogError(f"Audit log not found: {path}")

    try:
        with open(path, 'r', encoding='utf-8') as f:
            line_num = 0
            for line in f:
                line_num += 1
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed JSON at line {line_num}: {e}")
                        continue
    except PermissionError as e:
        logger.error(f"Permission denied reading audit log: {path}")
        raise AuditLogError(f"Cannot read audit log: {e}") from e
    except UnicodeDecodeError as e:
        logger.error(f"Audit log is not valid UTF-8: {path}")
        raise AuditLogError(f"Audit log encoding error: {e}") from e
    except OSError as e:
        logger.error(f"Failed to read audit log: {e}")
        raise AuditLogError(f"Cannot read audit log: {e}") from e

    return records
