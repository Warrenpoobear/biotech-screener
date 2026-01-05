"""
Deterministic hash utilities for Wake Robin Biotech Alpha System.

Uses SHA-256 for all hashing to ensure:
    - Cross-run reproducibility (Python's hash() is randomized)
    - Cross-platform consistency
    - Audit trail for exact input states
"""

import hashlib
import json
from datetime import date
from decimal import Decimal
from typing import Any, Union


def stable_json_dumps(obj: Any) -> str:
    """
    Convert object to JSON string with deterministic ordering.
    
    Handles:
        - dict sorting by keys
        - date serialization
        - Decimal serialization
        - Enum serialization
    
    Args:
        obj: Object to serialize
    
    Returns:
        Deterministic JSON string
    """
    def default_serializer(o: Any) -> Any:
        if isinstance(o, date):
            return o.isoformat()
        if isinstance(o, Decimal):
            return str(o)
        if hasattr(o, "value"):  # Enum
            return o.value
        if hasattr(o, "to_dict"):  # TrialRow and similar
            return o.to_dict()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
    
    return json.dumps(obj, sort_keys=True, default=default_serializer)


def compute_hash(data: Any) -> str:
    """
    Compute SHA-256 hash of data.
    
    Args:
        data: Any JSON-serializable object
    
    Returns:
        Hash string in format "sha256:abc123..."
    """
    if isinstance(data, str):
        json_str = data
    else:
        json_str = stable_json_dumps(data)
    
    hash_bytes = hashlib.sha256(json_str.encode("utf-8")).hexdigest()
    return f"sha256:{hash_bytes}"


def compute_trial_facts_hash(trials_by_ticker: dict[str, list]) -> str:
    """
    Compute deterministic hash of trial facts for provenance tracking.
    
    The hash captures the exact state of trial data used to generate
    a snapshot, enabling:
        - Reproducibility verification
        - Input drift detection
        - Audit trail
    
    Args:
        trials_by_ticker: Dict mapping ticker to list of TrialRow (or dicts)
    
    Returns:
        Hash string in format "sha256:abc123..."
    """
    # Build canonical representation
    canonical: dict[str, list[dict]] = {}
    
    for ticker in sorted(trials_by_ticker.keys()):
        trials = trials_by_ticker[ticker]
        # Convert to dicts if needed and sort by nct_id
        trial_dicts = []
        for t in trials:
            if hasattr(t, "to_dict"):
                trial_dicts.append(t.to_dict())
            elif isinstance(t, dict):
                trial_dicts.append(t)
            else:
                raise TypeError(f"Unknown trial type: {type(t)}")
        
        # Sort by nct_id for determinism
        trial_dicts.sort(key=lambda x: x.get("nct_id", ""))
        canonical[ticker] = trial_dicts
    
    return compute_hash(canonical)


def compute_snapshot_id(
    as_of_date: date,
    pit_cutoff: date,
    input_hashes: dict[str, str],
    provider_metadata: dict[str, Any],
) -> str:
    """
    Compute unique snapshot ID from all inputs.
    
    Same inputs -> identical snapshot_id -> identical snapshot content.
    
    Args:
        as_of_date: Snapshot date
        pit_cutoff: PIT cutoff date
        input_hashes: Dict of input component hashes
        provider_metadata: Provider configuration and state
    
    Returns:
        Hash string serving as snapshot ID
    """
    canonical = {
        "as_of_date": as_of_date.isoformat(),
        "pit_cutoff": pit_cutoff.isoformat(),
        "input_hashes": input_hashes,
        "provider_metadata": provider_metadata,
    }
    
    return compute_hash(canonical)
