"""
Provenance tracking for deterministic outputs.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict

# Fields excluded from hash computation (non-deterministic)
HASH_EXCLUDED_FIELDS = frozenset([
    "loaded_at",
    "generated_at",
    "timestamp",
    "runtime_ms",
])


def _json_serializer(obj: Any) -> Any:
    """JSON serializer for types not serializable by default."""
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, frozenset):
        return sorted(list(obj))
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def compute_hash(data: Any) -> str:
    """
    Compute deterministic SHA-256 hash of data.
    Excludes fields in HASH_EXCLUDED_FIELDS.
    """
    def clean_dict(d: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(d, dict):
            return d
        return {
            k: clean_dict(v) if isinstance(v, dict) else v
            for k, v in sorted(d.items())
            if k not in HASH_EXCLUDED_FIELDS
        }
    
    cleaned = clean_dict(data) if isinstance(data, dict) else data
    json_str = json.dumps(cleaned, sort_keys=True, default=_json_serializer)
    return f"sha256:{hashlib.sha256(json_str.encode()).hexdigest()}"


def create_provenance(
    ruleset_version: str,
    inputs: Dict[str, Any],
    pit_cutoff: str,
) -> Dict[str, Any]:
    """
    Create provenance record for a module output.
    
    Args:
        ruleset_version: Version of the ruleset used
        inputs: Input data for hash computation
        pit_cutoff: PIT cutoff date used
    
    Returns:
        Dict with ruleset_version, inputs_hash, pit_cutoff
    """
    return {
        "ruleset_version": ruleset_version,
        "inputs_hash": compute_hash(inputs),
        "pit_cutoff": pit_cutoff,
    }
