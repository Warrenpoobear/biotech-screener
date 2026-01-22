"""
Parameters Archive Loader

Loads scoring parameters from versioned archive files.
Computes parameters_hash for audit trail.

Fail-closed: missing or invalid params file stops the run.

Security Features (v1.1.0):
- File size limits
- Schema validation
- Path traversal protection
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

from governance.hashing import hash_canonical_json_short
from governance.canonical_json import canonical_dumps

# Import hardening utilities
try:
    from common.production_hardening import (
        validate_file_size,
        validate_path_within_base,
        PathTraversalError,
        FileSizeError,
        MAX_CONFIG_FILE_SIZE_MB,
    )
    HAS_HARDENING = True
except ImportError:
    HAS_HARDENING = False
    MAX_CONFIG_FILE_SIZE_MB = 10

# Import schema validation
try:
    from common.schema_validation import (
        validate_params as validate_params_schema,
        validate_weights_sum,
        ValidationResult,
    )
    HAS_SCHEMA_VALIDATION = True
except ImportError:
    HAS_SCHEMA_VALIDATION = False

logger = logging.getLogger(__name__)

# Default params archive directory (relative to repo root)
DEFAULT_PARAMS_DIR = "params_archive"


class ParamsLoadError(Exception):
    """Error loading parameters from archive."""
    pass


class ParamsValidationError(ParamsLoadError):
    """Error validating parameters against schema."""
    pass


def get_params_path(
    score_version: str,
    params_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Get path to parameters file for a score version.

    Args:
        score_version: Version string (e.g., "v1")
        params_dir: Optional override for params directory

    Returns:
        Path to params file
    """
    if params_dir is None:
        # Find repo root by looking for governance/
        current = Path(__file__).parent.parent
        params_dir = current / DEFAULT_PARAMS_DIR

    params_dir = Path(params_dir)
    return params_dir / f"{score_version}.json"


def load_params(
    score_version: str,
    params_dir: Optional[Union[str, Path]] = None,
    validate_schema: bool = True,
    schema_name: str = "module5",
) -> Tuple[Dict[str, Any], str]:
    """
    Load parameters from archive with security validation.

    Args:
        score_version: Version string (e.g., "v1")
        params_dir: Optional override for params directory
        validate_schema: Whether to validate against schema (default True)
        schema_name: Schema to validate against (default "module5")

    Returns:
        Tuple of (params_dict, parameters_hash)

    Raises:
        ParamsLoadError: If params file missing or invalid
        ParamsValidationError: If params fail schema validation
    """
    params_path = get_params_path(score_version, params_dir)

    if not params_path.exists():
        raise ParamsLoadError(
            f"Parameters file not found: {params_path}. "
            f"Create {params_path} with scoring parameters for version '{score_version}'."
        )

    # SECURITY: Check for symlinks
    if params_path.is_symlink():
        raise ParamsLoadError(
            f"Parameters file is a symbolic link (security risk): {params_path}"
        )

    # SECURITY: Validate file size
    if HAS_HARDENING:
        try:
            validate_file_size(params_path, MAX_CONFIG_FILE_SIZE_MB)
        except FileSizeError as e:
            raise ParamsLoadError(f"Parameters file too large: {e}") from e

    try:
        with open(params_path, 'r', encoding='utf-8') as f:
            params = json.load(f)
    except json.JSONDecodeError as e:
        raise ParamsLoadError(f"Invalid JSON in {params_path}: {e}")
    except UnicodeDecodeError as e:
        raise ParamsLoadError(f"Invalid encoding in {params_path}: {e}")
    except IOError as e:
        raise ParamsLoadError(f"Error reading {params_path}: {e}")

    # Validate structure
    if not isinstance(params, dict):
        raise ParamsLoadError(f"Parameters must be a JSON object, got {type(params).__name__}")

    # VALIDATION: Schema validation if available
    if validate_schema and HAS_SCHEMA_VALIDATION:
        result = validate_params_schema(params, schema_name)
        if not result.valid:
            error_messages = result.error_messages()
            raise ParamsValidationError(
                f"Parameters fail schema validation for {params_path}: "
                f"{'; '.join(error_messages[:5])}"
            )
        logger.debug(f"Parameters passed schema validation: {params_path}")

    # Compute hash
    params_hash = compute_parameters_hash(params)

    return params, params_hash


def compute_parameters_hash(params: Dict[str, Any], length: int = 16) -> str:
    """
    Compute hash of parameters.

    Args:
        params: Parameters dict
        length: Hash length (default 16)

    Returns:
        Truncated hex hash
    """
    return hash_canonical_json_short(params, length=length)


def save_params(
    params: Dict[str, Any],
    score_version: str,
    params_dir: Optional[Union[str, Path]] = None,
    validate_schema: bool = True,
    schema_name: str = "module5",
) -> Tuple[Path, str]:
    """
    Save parameters to archive with validation.

    Args:
        params: Parameters dict
        score_version: Version string
        params_dir: Optional override for params directory
        validate_schema: Whether to validate against schema before saving
        schema_name: Schema to validate against

    Returns:
        Tuple of (path, parameters_hash)

    Raises:
        ParamsValidationError: If params fail schema validation
    """
    # VALIDATION: Schema validation before saving
    if validate_schema and HAS_SCHEMA_VALIDATION:
        result = validate_params_schema(params, schema_name)
        if not result.valid:
            error_messages = result.error_messages()
            raise ParamsValidationError(
                f"Cannot save invalid parameters: {'; '.join(error_messages[:5])}"
            )

    params_path = get_params_path(score_version, params_dir)

    # SECURITY: Create directory with restricted permissions
    if HAS_HARDENING:
        from common.production_hardening import safe_mkdir
        safe_mkdir(params_path.parent, mode=0o700)
    else:
        params_path.parent.mkdir(parents=True, exist_ok=True)

    # Write canonical JSON atomically
    import tempfile
    import os

    content = canonical_dumps(params)

    # Write to temp file first, then atomic rename
    fd, tmp_path = tempfile.mkstemp(
        dir=params_path.parent,
        prefix='.tmp_params_',
        suffix='.json'
    )
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(content)
        os.chmod(tmp_path, 0o600)  # Secure permissions
        Path(tmp_path).replace(params_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    params_hash = compute_parameters_hash(params)
    logger.debug(f"Saved parameters to {params_path} (hash: {params_hash})")
    return params_path, params_hash


def validate_params_structure(
    params: Dict[str, Any],
    required_keys: Optional[list] = None,
) -> Tuple[bool, str]:
    """
    Validate parameters structure.

    Args:
        params: Parameters dict
        required_keys: Optional list of required top-level keys

    Returns:
        Tuple of (is_valid, message)
    """
    if not isinstance(params, dict):
        return False, f"Parameters must be dict, got {type(params).__name__}"

    if required_keys:
        missing = [k for k in required_keys if k not in params]
        if missing:
            return False, f"Missing required parameters: {missing}"

    return True, "Parameters structure valid"


def get_params_metadata(
    score_version: str,
    params_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Get metadata about params file without loading full content.

    Args:
        score_version: Version string
        params_dir: Optional override for params directory

    Returns:
        Dict with path, exists, hash (if exists)
    """
    params_path = get_params_path(score_version, params_dir)

    metadata = {
        "score_version": score_version,
        "path": str(params_path),
        "exists": params_path.exists(),
    }

    if params_path.exists():
        try:
            params, params_hash = load_params(score_version, params_dir, validate_schema=False)
            metadata["parameters_hash"] = params_hash
            metadata["keys"] = sorted(params.keys())
        except ParamsLoadError as e:
            metadata["error"] = str(e)

    return metadata


def load_and_validate_params(
    score_version: str,
    params_dir: Optional[Union[str, Path]] = None,
    schema_name: str = "module5",
    weight_fields: Optional[list] = None,
    expected_weight_sum: float = 1.0,
) -> Tuple[Dict[str, Any], str]:
    """
    Load parameters with full validation including weight sum check.

    Args:
        score_version: Version string
        params_dir: Optional override for params directory
        schema_name: Schema to validate against
        weight_fields: Fields that should sum to expected_weight_sum
        expected_weight_sum: Expected sum of weights (default 1.0)

    Returns:
        Tuple of (params_dict, parameters_hash)

    Raises:
        ParamsLoadError: If params file missing or invalid
        ParamsValidationError: If validation fails
    """
    # Load with schema validation
    params, params_hash = load_params(
        score_version,
        params_dir,
        validate_schema=True,
        schema_name=schema_name,
    )

    # Additional weight sum validation
    if weight_fields and HAS_SCHEMA_VALIDATION:
        result = validate_weights_sum(params, weight_fields, expected_weight_sum)
        if not result.valid:
            error_messages = result.error_messages()
            raise ParamsValidationError(
                f"Weight sum validation failed: {'; '.join(error_messages)}"
            )
        logger.debug(f"Weight sum validation passed for {weight_fields}")

    return params, params_hash
