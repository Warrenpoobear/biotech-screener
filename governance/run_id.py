"""
Deterministic Run ID Generation

Computes a stable run_id based on:
- as_of_date
- score_version
- parameters_hash
- input_hashes (sorted)
- mapping_hashes (sorted)
- pipeline_version

Same inputs always produce the same run_id.
No timestamps, no UUIDs, no randomness.
"""

from typing import List, Dict, Optional

from governance.hashing import hash_canonical_json_short


def compute_run_id(
    as_of_date: str,
    score_version: str,
    parameters_hash: str,
    input_hashes: List[Dict[str, str]],
    pipeline_version: str,
    mapping_hashes: Optional[List[Dict[str, str]]] = None,
    length: int = 16,
) -> str:
    """
    Compute deterministic run ID.

    Args:
        as_of_date: Analysis date (YYYY-MM-DD format)
        score_version: Version of scoring parameters (e.g., "v1")
        parameters_hash: SHA256 hash of parameters (truncated)
        input_hashes: List of {"path": name, "sha256": hash}
        pipeline_version: Version of pipeline code
        mapping_hashes: Optional list of {"name": name, "sha256": hash}
        length: Length of returned hash (default 16)

    Returns:
        Truncated hex digest (run_id)
    """
    # Ensure sorted input hashes
    sorted_input_hashes = sorted(
        [{"path": h["path"], "sha256": h["sha256"]} for h in input_hashes],
        key=lambda x: x["path"]
    )

    # Ensure sorted mapping hashes
    sorted_mapping_hashes = []
    if mapping_hashes:
        sorted_mapping_hashes = sorted(
            [{"name": h["name"], "sha256": h["sha256"]} for h in mapping_hashes],
            key=lambda x: x["name"]
        )

    # Build canonical run identity
    run_identity = {
        "as_of": as_of_date,
        "input_hashes": sorted_input_hashes,
        "mapping_hashes": sorted_mapping_hashes,
        "parameters_hash": parameters_hash,
        "pipeline_version": pipeline_version,
        "score_version": score_version,
    }

    return hash_canonical_json_short(run_identity, length=length)


def validate_run_id(run_id: str) -> bool:
    """
    Validate run_id format.

    Args:
        run_id: Run ID to validate

    Returns:
        True if valid hex string of expected length
    """
    if not isinstance(run_id, str):
        return False

    # Check length (typically 16 hex chars)
    if not (8 <= len(run_id) <= 64):
        return False

    # Check hex format
    try:
        int(run_id, 16)
        return True
    except ValueError:
        return False


def parse_run_id_components(run_id: str) -> Dict[str, str]:
    """
    Parse run_id to extract components (for display only).

    Note: Run IDs are hashes, so this just returns metadata about the ID itself.

    Args:
        run_id: Run ID string

    Returns:
        Dict with id, length, format info
    """
    return {
        "run_id": run_id,
        "length": len(run_id),
        "format": "sha256_truncated",
        "valid": validate_run_id(run_id),
    }
