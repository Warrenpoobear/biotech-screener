"""
common/provenance.py - Provenance tracking for deterministic outputs.

Creates immutable provenance records that track:
- Ruleset version
- Input hash (content-based deduplication)
- Timestamp and cutoff dates
- Environment info
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _compute_input_hash(inputs: Any) -> str:
    """
    Compute deterministic SHA-256 hash of inputs.
    
    Uses JSON serialization with sorted keys for stability.
    """
    # Serialize with sorted keys for determinism
    serialized = json.dumps(inputs, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()[:16]


def create_provenance(
    ruleset_version: str,
    inputs: Dict[str, Any],
    pit_cutoff: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create provenance record for module output.
    
    Args:
        ruleset_version: Semantic version of the scoring ruleset
        inputs: Dict of input data (will be hashed)
        pit_cutoff: Point-in-time cutoff date (ISO format)
        extra: Additional provenance fields
    
    Returns:
        Provenance dict suitable for JSON serialization
    """
    provenance = {
        "ruleset_version": ruleset_version,
        "input_hash": _compute_input_hash(inputs),
        "pit_cutoff": pit_cutoff,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
    }
    
    if extra:
        provenance.update(extra)
    
    return provenance


def verify_provenance(
    output: Dict[str, Any],
    expected_version: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Verify provenance record integrity.
    
    Returns:
        {"valid": bool, "issues": [str], "provenance": {...}}
    """
    issues = []
    provenance = output.get("provenance", {})
    
    if not provenance:
        issues.append("missing_provenance")
        return {"valid": False, "issues": issues, "provenance": None}
    
    # Check required fields
    required = ["ruleset_version", "input_hash", "pit_cutoff", "generated_at"]
    for field in required:
        if field not in provenance:
            issues.append(f"missing_field_{field}")
    
    # Version check
    if expected_version and provenance.get("ruleset_version") != expected_version:
        issues.append(f"version_mismatch_expected_{expected_version}_got_{provenance.get('ruleset_version')}")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "provenance": provenance,
    }
