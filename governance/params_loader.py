"""
Parameters Archive Loader

Loads scoring parameters from versioned archive files.
Computes parameters_hash for audit trail.

Fail-closed: missing or invalid params file stops the run.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

from governance.hashing import hash_canonical_json_short
from governance.canonical_json import canonical_dumps


# Default params archive directory (relative to repo root)
DEFAULT_PARAMS_DIR = "params_archive"


class ParamsLoadError(Exception):
    """Error loading parameters from archive."""
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
) -> Tuple[Dict[str, Any], str]:
    """
    Load parameters from archive.

    Args:
        score_version: Version string (e.g., "v1")
        params_dir: Optional override for params directory

    Returns:
        Tuple of (params_dict, parameters_hash)

    Raises:
        ParamsLoadError: If params file missing or invalid
    """
    params_path = get_params_path(score_version, params_dir)

    if not params_path.exists():
        raise ParamsLoadError(
            f"Parameters file not found: {params_path}. "
            f"Create {params_path} with scoring parameters for version '{score_version}'."
        )

    try:
        with open(params_path, 'r', encoding='utf-8') as f:
            params = json.load(f)
    except json.JSONDecodeError as e:
        raise ParamsLoadError(f"Invalid JSON in {params_path}: {e}")
    except Exception as e:
        raise ParamsLoadError(f"Error reading {params_path}: {e}")

    # Validate structure
    if not isinstance(params, dict):
        raise ParamsLoadError(f"Parameters must be a JSON object, got {type(params).__name__}")

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
) -> Tuple[Path, str]:
    """
    Save parameters to archive.

    Args:
        params: Parameters dict
        score_version: Version string
        params_dir: Optional override for params directory

    Returns:
        Tuple of (path, parameters_hash)
    """
    params_path = get_params_path(score_version, params_dir)
    params_path.parent.mkdir(parents=True, exist_ok=True)

    # Write canonical JSON
    with open(params_path, 'w', encoding='utf-8') as f:
        f.write(canonical_dumps(params))

    params_hash = compute_parameters_hash(params)
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
            params, params_hash = load_params(score_version, params_dir)
            metadata["parameters_hash"] = params_hash
            metadata["keys"] = sorted(params.keys())
        except ParamsLoadError as e:
            metadata["error"] = str(e)

    return metadata
