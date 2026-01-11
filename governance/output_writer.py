"""
Canonical Output Writer with Governance Metadata

Wraps output writing to ensure:
1. All outputs include governance metadata
2. All outputs use canonical JSON serialization
3. Output hashes are computed and returned
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from governance.canonical_json import canonical_dumps
from governance.hashing import hash_bytes, hash_file
from governance.schema_registry import PIPELINE_VERSION, SCHEMA_VERSION


def inject_governance_metadata(
    data: Dict[str, Any],
    run_id: str,
    score_version: str,
    parameters_hash: str,
    input_lineage: List[Dict[str, str]],
    schema_version: Optional[str] = None,
    output_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Inject governance metadata into output data.

    Args:
        data: Output data dict
        run_id: Deterministic run identifier
        score_version: Score version (e.g., "v1")
        parameters_hash: Hash of parameters
        input_lineage: List of input file metadata
        schema_version: Optional schema version override
        output_type: Optional output type for schema lookup

    Returns:
        Data dict with governance metadata injected
    """
    # Build metadata block
    governance = {
        "generation_metadata": {
            "pipeline_version": PIPELINE_VERSION,
            "tool_name": "biotech-screener",
        },
        "input_lineage": sorted(input_lineage, key=lambda x: x.get("path", "")),
        "parameters_hash": parameters_hash,
        "run_id": run_id,
        "schema_version": schema_version or SCHEMA_VERSION,
        "score_version": score_version,
    }

    # Create new dict with governance first (for readability)
    result = {"_governance": governance}
    result.update(data)

    return result


def write_canonical_output(
    data: Dict[str, Any],
    output_path: Union[str, Path],
    run_id: str,
    score_version: str,
    parameters_hash: str,
    input_lineage: List[Dict[str, str]],
    schema_version: Optional[str] = None,
) -> Dict[str, str]:
    """
    Write output with governance metadata and canonical formatting.

    Args:
        data: Output data dict
        output_path: Path to write output
        run_id: Deterministic run identifier
        score_version: Score version
        parameters_hash: Hash of parameters
        input_lineage: List of input file metadata
        schema_version: Optional schema version override

    Returns:
        Dict with path and sha256 hash of written file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Inject metadata
    enriched = inject_governance_metadata(
        data=data,
        run_id=run_id,
        score_version=score_version,
        parameters_hash=parameters_hash,
        input_lineage=input_lineage,
        schema_version=schema_version,
    )

    # Serialize canonically
    content = canonical_dumps(enriched)

    # Write
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    # Compute hash
    output_hash = hash_bytes(content.encode('utf-8'))

    return {
        "path": str(output_path),
        "sha256": output_hash,
    }


def get_environment_fingerprint() -> Dict[str, str]:
    """
    Get environment fingerprint for documentation (not hashing).

    Returns:
        Dict with python version, platform info
    """
    import platform

    return {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "python_version": platform.python_version(),
    }


def build_input_lineage(
    input_files: List[Union[str, Path]],
    as_of_date: str,
    schema_version: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Build input lineage list from file paths.

    Args:
        input_files: List of input file paths
        as_of_date: Analysis date
        schema_version: Optional schema version

    Returns:
        List of lineage records
    """
    lineage = []

    for path in input_files:
        path = Path(path)
        if path.exists():
            record = {
                "as_of_date": as_of_date,
                "path": path.name,
                "sha256": hash_file(path),
            }
            if schema_version:
                record["schema_version"] = schema_version
            lineage.append(record)

    # Sort for determinism
    return sorted(lineage, key=lambda x: x["path"])
