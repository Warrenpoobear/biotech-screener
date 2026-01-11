"""
Schema Version Registry

Centralized version constants for:
- Pipeline version
- Schema versions for each output type
- Score versions

Provides validation helpers for schema compatibility.
"""

from typing import Dict, Optional, Tuple

# =============================================================================
# VERSION CONSTANTS
# =============================================================================

# Pipeline version - bump on significant orchestration changes
PIPELINE_VERSION = "2.0.0"

# Schema version - bump when output structure changes
SCHEMA_VERSION = "1.0.0"

# Output-specific schema versions
OUTPUT_SCHEMA_VERSIONS = {
    "screening_results": "1.0.0",
    "institutional_signals": "1.0.0",
    "risk_gates": "1.0.0",
    "liquidity_scores": "1.0.0",
    "audit_log": "1.0.0",
}

# Default score version (can be overridden via CLI)
DEFAULT_SCORE_VERSION = "v1"

# Supported score versions
SUPPORTED_SCORE_VERSIONS = ["v1"]


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def validate_schema_version(
    schema_version: str,
    expected_version: Optional[str] = None,
    strict: bool = False,
) -> Tuple[bool, str]:
    """
    Validate schema version compatibility.

    Args:
        schema_version: Version to validate
        expected_version: Expected version (if None, uses SCHEMA_VERSION)
        strict: If True, require exact match; if False, allow compatible versions

    Returns:
        Tuple of (is_valid, message)
    """
    if expected_version is None:
        expected_version = SCHEMA_VERSION

    if schema_version == expected_version:
        return True, "Schema version match"

    # Parse versions
    try:
        actual_parts = [int(x) for x in schema_version.split('.')]
        expected_parts = [int(x) for x in expected_version.split('.')]
    except (ValueError, AttributeError):
        return False, f"Invalid version format: {schema_version}"

    if len(actual_parts) != 3 or len(expected_parts) != 3:
        return False, f"Version must be in X.Y.Z format: {schema_version}"

    if strict:
        return False, f"Schema version mismatch: expected {expected_version}, got {schema_version}"

    # Non-strict: allow compatible minor/patch versions
    # Major version must match
    if actual_parts[0] != expected_parts[0]:
        return False, f"Incompatible major version: expected {expected_parts[0]}.x.x, got {schema_version}"

    # Minor version should not be lower
    if actual_parts[1] < expected_parts[1]:
        return False, f"Schema version too old: expected >= {expected_version}, got {schema_version}"

    return True, f"Schema version compatible: {schema_version} >= {expected_version}"


def validate_score_version(score_version: str) -> Tuple[bool, str]:
    """
    Validate score version is supported.

    Args:
        score_version: Score version to validate (e.g., "v1")

    Returns:
        Tuple of (is_valid, message)
    """
    if score_version in SUPPORTED_SCORE_VERSIONS:
        return True, f"Score version {score_version} is supported"

    return False, f"Unsupported score version: {score_version}. Supported: {SUPPORTED_SCORE_VERSIONS}"


def get_schema_info() -> Dict[str, str]:
    """
    Get all schema/version information.

    Returns:
        Dict with all version constants
    """
    return {
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "default_score_version": DEFAULT_SCORE_VERSION,
        "supported_score_versions": SUPPORTED_SCORE_VERSIONS,
        "output_schema_versions": OUTPUT_SCHEMA_VERSIONS,
    }


def get_output_schema_version(output_type: str) -> str:
    """
    Get schema version for a specific output type.

    Args:
        output_type: Type of output (e.g., "screening_results")

    Returns:
        Schema version string

    Raises:
        KeyError: If output_type is not recognized
    """
    if output_type not in OUTPUT_SCHEMA_VERSIONS:
        raise KeyError(f"Unknown output type: {output_type}. Known types: {list(OUTPUT_SCHEMA_VERSIONS.keys())}")
    return OUTPUT_SCHEMA_VERSIONS[output_type]
