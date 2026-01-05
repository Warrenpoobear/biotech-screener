"""
Run Manifest Logging

Production-grade audit trail for backtest reproducibility.
Every backtest run creates a manifest entry with:
- Timestamp
- Module versions
- Config hash
- Data hashes
- Results hash
- Parameters

Supports:
- JSONL append-only log
- Query by run_id, date range
- Reproducibility verification

Usage:
    manifest = RunManifest("output/manifests")
    
    # Log a run
    entry = manifest.log_run(
        run_id="prod_v1_20240101",
        config=config,
        data_hashes={"prices": "sha256:...", "trials": "sha256:..."},
        results=results,
        metadata={"analyst": "system"}
    )
    
    # Query runs
    runs = manifest.query_runs(after="2024-01-01")
    
    # Verify reproducibility
    is_valid = manifest.verify_run("prod_v1_20240101", results)
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

# Manifest version for schema evolution
MANIFEST_VERSION = "1.0.0"


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal types."""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        return super().default(obj)


def compute_content_hash(data: Any) -> str:
    """
    Compute deterministic SHA256 hash of data.
    
    Handles nested dicts, lists, Decimals.
    """
    json_str = json.dumps(
        data,
        sort_keys=True,
        cls=DecimalEncoder,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(json_str.encode()).hexdigest()}"


# Fields to exclude from results hash (non-deterministic or metadata-only)
RESULTS_HASH_EXCLUDE_KEYS = {
    "timestamp",
    "run_id", 
    "generated_at",
    "file_path",
    "cache_hits",
    "cache_misses",
    "wall_time_seconds",
    "retrieval_date",
}


def canonicalize_for_hash(data: Any, exclude_keys: Optional[set] = None) -> Any:
    """
    Recursively remove non-deterministic fields before hashing.
    
    Args:
        data: Data structure to canonicalize
        exclude_keys: Keys to exclude (uses RESULTS_HASH_EXCLUDE_KEYS if None)
    
    Returns:
        Canonicalized copy safe for deterministic hashing
    """
    if exclude_keys is None:
        exclude_keys = RESULTS_HASH_EXCLUDE_KEYS
    
    if isinstance(data, dict):
        return {
            k: canonicalize_for_hash(v, exclude_keys)
            for k, v in sorted(data.items())
            if k not in exclude_keys
        }
    elif isinstance(data, list):
        return [canonicalize_for_hash(item, exclude_keys) for item in data]
    elif isinstance(data, Decimal):
        return str(data)
    else:
        return data


def compute_results_hash(results: Dict[str, Any]) -> str:
    """
    Compute hash of results excluding non-deterministic fields.
    
    This ensures reproducibility verification works correctly:
    same inputs → same results_hash, regardless of when run.
    """
    canonical = canonicalize_for_hash(results)
    return compute_content_hash(canonical)


class ManifestEntry:
    """Single run manifest entry."""
    
    def __init__(
        self,
        run_id: str,
        timestamp: str,
        config_hash: str,
        data_hashes: Dict[str, str],
        results_hash: str,
        module_versions: Dict[str, str],
        parameters: Dict[str, Any],
        metadata: Dict[str, Any],
    ):
        self.run_id = run_id
        self.timestamp = timestamp
        self.config_hash = config_hash
        self.data_hashes = data_hashes
        self.results_hash = results_hash
        self.module_versions = module_versions
        self.parameters = parameters
        self.metadata = metadata
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_version": MANIFEST_VERSION,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "config_hash": self.config_hash,
            "data_hashes": self.data_hashes,
            "results_hash": self.results_hash,
            "module_versions": self.module_versions,
            "parameters": self.parameters,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ManifestEntry":
        return cls(
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            config_hash=data["config_hash"],
            data_hashes=data["data_hashes"],
            results_hash=data["results_hash"],
            module_versions=data["module_versions"],
            parameters=data["parameters"],
            metadata=data.get("metadata", {}),
        )


class RunManifest:
    """
    Append-only run manifest log.
    
    Stores JSONL entries for each backtest run.
    Enables reproducibility verification and audit.
    """
    
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.output_dir / "run_manifest.jsonl"
    
    def log_run(
        self,
        run_id: str,
        config: Dict[str, Any],
        data_hashes: Dict[str, str],
        results: Dict[str, Any],
        module_versions: Optional[Dict[str, str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ManifestEntry:
        """
        Log a backtest run to the manifest.
        
        Args:
            run_id: Unique run identifier
            config: Configuration dict (weights, horizons, etc.)
            data_hashes: Dict of data source → hash
            results: Backtest results dict
            module_versions: Dict of module → version
            parameters: Run parameters
            metadata: Additional metadata (analyst, notes, etc.)
        
        Returns:
            ManifestEntry that was logged
        """
        # Compute hashes
        config_hash = compute_content_hash(config)
        results_hash = compute_results_hash(results)  # Excludes non-deterministic fields
        
        # Extract module versions from results if not provided
        if module_versions is None:
            provenance = results.get("provenance", {})
            module_versions = {
                "metrics": provenance.get("metrics_version", "unknown"),
                "composite": provenance.get("module_ruleset_versions", ["unknown"])[0]
                if provenance.get("module_ruleset_versions") else "unknown",
            }
        
        # Default parameters
        if parameters is None:
            parameters = {
                "horizons": results.get("horizons", []),
                "n_dates": len(results.get("period_metrics", {})),
            }
        
        # Create entry
        entry = ManifestEntry(
            run_id=run_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            config_hash=config_hash,
            data_hashes=data_hashes,
            results_hash=results_hash,
            module_versions=module_versions,
            parameters=parameters,
            metadata=metadata or {},
        )
        
        # Append to JSONL
        with open(self.manifest_file, "a") as f:
            f.write(json.dumps(entry.to_dict(), cls=DecimalEncoder) + "\n")
        
        return entry
    
    def get_run(self, run_id: str) -> Optional[ManifestEntry]:
        """Get a specific run by ID."""
        for entry in self._read_entries():
            if entry.run_id == run_id:
                return entry
        return None
    
    def query_runs(
        self,
        after: Optional[str] = None,
        before: Optional[str] = None,
        config_hash: Optional[str] = None,
    ) -> List[ManifestEntry]:
        """
        Query runs by criteria.
        
        Args:
            after: Return runs after this timestamp (ISO format)
            before: Return runs before this timestamp (ISO format)
            config_hash: Filter by config hash
        
        Returns:
            List of matching ManifestEntry objects
        """
        results = []
        
        for entry in self._read_entries():
            # Filter by timestamp
            if after and entry.timestamp < after:
                continue
            if before and entry.timestamp > before:
                continue
            
            # Filter by config hash
            if config_hash and entry.config_hash != config_hash:
                continue
            
            results.append(entry)
        
        return results
    
    def verify_run(
        self,
        run_id: str,
        results: Dict[str, Any],
    ) -> bool:
        """
        Verify that results match the logged manifest entry.
        
        Args:
            run_id: Run ID to verify
            results: Results to verify against manifest
        
        Returns:
            True if results hash matches manifest
        """
        entry = self.get_run(run_id)
        if entry is None:
            return False
        
        current_hash = compute_results_hash(results)
        return current_hash == entry.results_hash
    
    def _read_entries(self) -> List[ManifestEntry]:
        """Read all manifest entries."""
        if not self.manifest_file.exists():
            return []
        
        entries = []
        with open(self.manifest_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        entries.append(ManifestEntry.from_dict(data))
                    except json.JSONDecodeError:
                        continue
        
        return entries
    
    def get_latest_run(self) -> Optional[ManifestEntry]:
        """Get the most recent run."""
        entries = self._read_entries()
        if not entries:
            return None
        return max(entries, key=lambda e: e.timestamp)
    
    def export_summary(self) -> Dict[str, Any]:
        """Export summary statistics."""
        entries = self._read_entries()
        
        if not entries:
            return {"total_runs": 0}
        
        # Aggregate stats
        config_hashes = set(e.config_hash for e in entries)
        module_versions = set()
        for e in entries:
            for v in e.module_versions.values():
                module_versions.add(v)
        
        return {
            "total_runs": len(entries),
            "unique_configs": len(config_hashes),
            "date_range": {
                "first": min(e.timestamp for e in entries),
                "last": max(e.timestamp for e in entries),
            },
            "module_versions_seen": sorted(module_versions),
        }


def create_data_hashes(
    prices_file: Optional[str] = None,
    trials_data: Optional[List[Dict]] = None,
    snapshots: Optional[List[Dict]] = None,
) -> Dict[str, str]:
    """
    Helper to create data hashes for manifest logging.
    
    Args:
        prices_file: Path to prices CSV
        trials_data: Trial records
        snapshots: Module 5 snapshots
    
    Returns:
        Dict of data source → hash
    """
    hashes = {}
    
    if prices_file:
        with open(prices_file, "rb") as f:
            hashes["prices"] = f"sha256:{hashlib.sha256(f.read()).hexdigest()}"
    
    if trials_data:
        hashes["trials"] = compute_content_hash(trials_data)
    
    if snapshots:
        hashes["snapshots"] = compute_content_hash(snapshots)
    
    return hashes


# Convenience function for backtest integration
def log_backtest_run(
    output_dir: str,
    run_id: str,
    config: Dict[str, Any],
    results: Dict[str, Any],
    prices_file: Optional[str] = None,
    snapshots: Optional[List[Dict]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ManifestEntry:
    """
    Convenience function to log a backtest run.
    
    Args:
        output_dir: Directory for manifest files
        run_id: Unique run identifier
        config: Configuration dict
        results: Backtest results
        prices_file: Optional path to prices file
        snapshots: Optional list of snapshots
        metadata: Optional metadata
    
    Returns:
        ManifestEntry
    """
    manifest = RunManifest(output_dir)
    
    data_hashes = create_data_hashes(
        prices_file=prices_file,
        snapshots=snapshots,
    )
    
    return manifest.log_run(
        run_id=run_id,
        config=config,
        data_hashes=data_hashes,
        results=results,
        metadata=metadata,
    )
