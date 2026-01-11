"""
Adapter Mapping Loader

Loads and validates field mappings between source schemas and canonical schemas.
Computes mapping_hash for audit trail.

Fail-closed: missing required fields trigger SCHEMA_MISMATCH.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Set

from governance.hashing import hash_canonical_json_short
from governance.canonical_json import canonical_dumps


# Default adapters directory (relative to repo root)
DEFAULT_ADAPTERS_DIR = "adapters"


class MappingLoadError(Exception):
    """Error loading mapping from file."""
    pass


class SchemaMismatchError(Exception):
    """Source data does not match required schema."""

    def __init__(self, missing_fields: List[str], source: str):
        self.missing_fields = missing_fields
        self.source = source
        super().__init__(
            f"SCHEMA_MISMATCH: Source '{source}' missing required fields: {missing_fields}"
        )


def get_mapping_path(
    source_name: str,
    mapping_version: str = "v1",
    adapters_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Get path to mapping file.

    Args:
        source_name: Source adapter name (e.g., "screener", "holdings")
        mapping_version: Mapping version (e.g., "v1")
        adapters_dir: Optional override for adapters directory

    Returns:
        Path to mapping JSON file
    """
    if adapters_dir is None:
        current = Path(__file__).parent.parent
        adapters_dir = current / DEFAULT_ADAPTERS_DIR

    adapters_dir = Path(adapters_dir)
    return adapters_dir / source_name / f"mapping_{mapping_version}.json"


def load_mapping(
    source_name: str,
    mapping_version: str = "v1",
    adapters_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Dict[str, Any], str]:
    """
    Load adapter mapping from file.

    Args:
        source_name: Source adapter name
        mapping_version: Mapping version
        adapters_dir: Optional override for adapters directory

    Returns:
        Tuple of (mapping_dict, mapping_hash)

    Raises:
        MappingLoadError: If mapping file missing or invalid
    """
    mapping_path = get_mapping_path(source_name, mapping_version, adapters_dir)

    if not mapping_path.exists():
        raise MappingLoadError(
            f"Mapping file not found: {mapping_path}. "
            f"Create mapping file for source '{source_name}' version '{mapping_version}'."
        )

    try:
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
    except json.JSONDecodeError as e:
        raise MappingLoadError(f"Invalid JSON in {mapping_path}: {e}")
    except Exception as e:
        raise MappingLoadError(f"Error reading {mapping_path}: {e}")

    # Validate structure
    if not isinstance(mapping, dict):
        raise MappingLoadError(f"Mapping must be a JSON object, got {type(mapping).__name__}")

    # Compute hash
    mapping_hash = compute_mapping_hash(mapping)

    return mapping, mapping_hash


def compute_mapping_hash(mapping: Dict[str, Any], length: int = 16) -> str:
    """
    Compute hash of mapping.

    Args:
        mapping: Mapping dict
        length: Hash length (default 16)

    Returns:
        Truncated hex hash
    """
    return hash_canonical_json_short(mapping, length=length)


def save_mapping(
    mapping: Dict[str, Any],
    source_name: str,
    mapping_version: str = "v1",
    adapters_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Path, str]:
    """
    Save mapping to file.

    Args:
        mapping: Mapping dict
        source_name: Source adapter name
        mapping_version: Mapping version
        adapters_dir: Optional override for adapters directory

    Returns:
        Tuple of (path, mapping_hash)
    """
    mapping_path = get_mapping_path(source_name, mapping_version, adapters_dir)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)

    with open(mapping_path, 'w', encoding='utf-8') as f:
        f.write(canonical_dumps(mapping))

    mapping_hash = compute_mapping_hash(mapping)
    return mapping_path, mapping_hash


def validate_source_schema(
    data: Union[Dict[str, Any], List[Dict[str, Any]]],
    mapping: Dict[str, Any],
    source_name: str,
) -> Tuple[bool, List[str]]:
    """
    Validate that source data has required fields from mapping.

    Args:
        data: Source data (dict or list of dicts)
        mapping: Mapping dict with "required_fields" key
        source_name: Source name for error messages

    Returns:
        Tuple of (is_valid, missing_fields)

    Raises:
        SchemaMismatchError: If required fields are missing (fail-closed)
    """
    required_fields = mapping.get("required_fields", [])
    if not required_fields:
        return True, []

    # Get first record to check fields
    if isinstance(data, list):
        if not data:
            # Empty data - can't validate, but pass
            return True, []
        sample = data[0]
    else:
        sample = data

    if not isinstance(sample, dict):
        raise SchemaMismatchError(required_fields, source_name)

    available_fields = set(sample.keys())
    missing = [f for f in required_fields if f not in available_fields]

    if missing:
        raise SchemaMismatchError(missing, source_name)

    return True, []


def apply_field_mapping(
    record: Dict[str, Any],
    mapping: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Apply field mapping to transform source record to canonical format.

    Args:
        record: Source record
        mapping: Mapping dict with "field_mappings" key

    Returns:
        Transformed record with canonical field names
    """
    field_mappings = mapping.get("field_mappings", {})

    result = {}
    for source_field, target_field in field_mappings.items():
        if source_field in record:
            result[target_field] = record[source_field]

    # Copy unmapped fields as-is
    mapped_sources = set(field_mappings.keys())
    for key, value in record.items():
        if key not in mapped_sources:
            result[key] = value

    return result


def get_mapping_metadata(
    source_name: str,
    mapping_version: str = "v1",
    adapters_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Get metadata about mapping file.

    Args:
        source_name: Source adapter name
        mapping_version: Mapping version
        adapters_dir: Optional override for adapters directory

    Returns:
        Dict with path, exists, hash, required_fields (if exists)
    """
    mapping_path = get_mapping_path(source_name, mapping_version, adapters_dir)

    metadata = {
        "source_name": source_name,
        "mapping_version": mapping_version,
        "path": str(mapping_path),
        "exists": mapping_path.exists(),
    }

    if mapping_path.exists():
        try:
            mapping, mapping_hash = load_mapping(source_name, mapping_version, adapters_dir)
            metadata["mapping_hash"] = mapping_hash
            metadata["required_fields"] = mapping.get("required_fields", [])
            metadata["field_count"] = len(mapping.get("field_mappings", {}))
        except MappingLoadError as e:
            metadata["error"] = str(e)

    return metadata
