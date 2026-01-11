"""
Snapshot Store Utilities for 13F Holdings History

Provides:
- Quarter date arithmetic (PIT-safe)
- Snapshot loading and listing
- Manifest management
- Schema validation

All operations are deterministic.
"""

import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

# Schema versions
SNAPSHOT_SCHEMA_VERSION = "13f_holdings_snapshot_v1"
MANIFEST_SCHEMA_VERSION = "13f_manifest_v1"

# Quarter end months and days
QUARTER_ENDS = [
    (3, 31),   # Q1
    (6, 30),   # Q2
    (9, 30),   # Q3
    (12, 31),  # Q4
]


class SnapshotError(Exception):
    """Error in snapshot operations."""
    pass


class SchemaValidationError(SnapshotError):
    """Schema validation failed."""
    pass


# =============================================================================
# QUARTER DATE ARITHMETIC
# =============================================================================

def get_quarter_end_for_date(d: Union[date, str]) -> date:
    """
    Get the quarter end date for a given date.

    Args:
        d: Date or ISO date string

    Returns:
        Quarter end date (Mar 31, Jun 30, Sep 30, or Dec 31)
    """
    if isinstance(d, str):
        d = date.fromisoformat(d)

    year = d.year
    month = d.month

    for q_month, q_day in QUARTER_ENDS:
        if month <= q_month:
            return date(year, q_month, q_day)

    # Should not reach here, but handle edge case
    return date(year, 12, 31)


def get_prior_quarter(quarter_end: Union[date, str]) -> date:
    """
    Get the prior quarter end date.

    Args:
        quarter_end: Quarter end date or ISO string

    Returns:
        Prior quarter end date

    Examples:
        2025-12-31 -> 2025-09-30
        2025-03-31 -> 2024-12-31
    """
    if isinstance(quarter_end, str):
        quarter_end = date.fromisoformat(quarter_end)

    year = quarter_end.year
    month = quarter_end.month

    # Find current quarter index
    for i, (q_month, q_day) in enumerate(QUARTER_ENDS):
        if month == q_month:
            if i == 0:
                # Q1 -> prior year Q4
                return date(year - 1, 12, 31)
            else:
                # Go to prior quarter
                prior_month, prior_day = QUARTER_ENDS[i - 1]
                return date(year, prior_month, prior_day)

    raise ValueError(f"Invalid quarter end date: {quarter_end}")


def get_quarter_sequence(
    end_quarter: Union[date, str],
    num_quarters: int,
) -> List[date]:
    """
    Get a sequence of quarter end dates going backward.

    Args:
        end_quarter: Most recent quarter end
        num_quarters: Number of quarters to include

    Returns:
        List of quarter end dates, newest first
    """
    if isinstance(end_quarter, str):
        end_quarter = date.fromisoformat(end_quarter)

    quarters = []
    current = end_quarter

    for _ in range(num_quarters):
        quarters.append(current)
        current = get_prior_quarter(current)

    return quarters


def is_valid_quarter_end(d: Union[date, str]) -> bool:
    """Check if date is a valid quarter end."""
    if isinstance(d, str):
        try:
            d = date.fromisoformat(d)
        except ValueError:
            return False

    for q_month, q_day in QUARTER_ENDS:
        if d.month == q_month and d.day == q_day:
            return True

    return False


# =============================================================================
# SNAPSHOT LOADING
# =============================================================================

def _get_snapshot_filename(quarter_end: Union[date, str]) -> str:
    """Get the filename for a quarter snapshot."""
    if isinstance(quarter_end, date):
        quarter_end = quarter_end.isoformat()
    return f"holdings_{quarter_end}.json"


def list_quarters(out_dir: Union[str, Path]) -> List[date]:
    """
    List all quarter end dates present in the holdings history directory.

    Args:
        out_dir: Holdings history directory

    Returns:
        Sorted list of quarter end dates (newest first)
    """
    out_dir = Path(out_dir)

    if not out_dir.exists():
        return []

    pattern = re.compile(r'^holdings_(\d{4}-\d{2}-\d{2})\.json$')
    quarters = []

    for path in out_dir.iterdir():
        if path.is_file():
            match = pattern.match(path.name)
            if match:
                try:
                    d = date.fromisoformat(match.group(1))
                    if is_valid_quarter_end(d):
                        quarters.append(d)
                except ValueError:
                    continue

    # Sort newest first
    return sorted(quarters, reverse=True)


def load_snapshot(
    quarter_end: Union[date, str],
    out_dir: Union[str, Path],
) -> Dict[str, Any]:
    """
    Load a quarter snapshot from disk.

    Args:
        quarter_end: Quarter end date or ISO string
        out_dir: Holdings history directory

    Returns:
        Snapshot dict

    Raises:
        SnapshotError: If snapshot not found or invalid
    """
    out_dir = Path(out_dir)

    if isinstance(quarter_end, date):
        quarter_end_str = quarter_end.isoformat()
    else:
        quarter_end_str = quarter_end

    filename = _get_snapshot_filename(quarter_end_str)
    filepath = out_dir / filename

    if not filepath.exists():
        raise SnapshotError(f"Snapshot not found: {filepath}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            snapshot = json.load(f)
    except json.JSONDecodeError as e:
        raise SnapshotError(f"Invalid JSON in {filepath}: {e}")

    return snapshot


def load_manifest(out_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Load the manifest file from holdings history directory.

    Args:
        out_dir: Holdings history directory

    Returns:
        Manifest dict

    Raises:
        SnapshotError: If manifest not found or invalid
    """
    out_dir = Path(out_dir)
    filepath = out_dir / "manifest.json"

    if not filepath.exists():
        raise SnapshotError(f"Manifest not found: {filepath}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
    except json.JSONDecodeError as e:
        raise SnapshotError(f"Invalid JSON in {filepath}: {e}")

    return manifest


# =============================================================================
# SCHEMA VALIDATION
# =============================================================================

def validate_snapshot_schema(snapshot: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a snapshot against the expected schema.

    Args:
        snapshot: Snapshot dict to validate

    Returns:
        Tuple of (is_valid, list of error messages)

    Raises:
        SchemaValidationError: If validation fails (when raise_on_error=True)
    """
    errors = []

    # Check _schema section
    if "_schema" not in snapshot:
        errors.append("Missing '_schema' section")
    else:
        schema = snapshot["_schema"]
        if schema.get("version") != SNAPSHOT_SCHEMA_VERSION:
            errors.append(
                f"Schema version mismatch: expected {SNAPSHOT_SCHEMA_VERSION}, "
                f"got {schema.get('version')}"
            )
        if "quarter_end" not in schema:
            errors.append("Missing 'quarter_end' in _schema")

    # Check required top-level keys
    required_keys = ["tickers", "managers", "stats"]
    for key in required_keys:
        if key not in snapshot:
            errors.append(f"Missing required key: '{key}'")

    # Validate tickers structure
    if "tickers" in snapshot:
        tickers = snapshot["tickers"]
        if not isinstance(tickers, dict):
            errors.append("'tickers' must be a dict")
        else:
            # Spot check first ticker
            for ticker, data in list(tickers.items())[:1]:
                if "holdings" not in data:
                    errors.append(f"Ticker '{ticker}' missing 'holdings'")
                elif "current" not in data["holdings"]:
                    errors.append(f"Ticker '{ticker}' missing 'holdings.current'")

    # Validate stats
    if "stats" in snapshot:
        stats = snapshot["stats"]
        required_stats = ["tickers_count", "managers_count"]
        for stat in required_stats:
            if stat not in stats:
                errors.append(f"Missing stat: '{stat}'")

    return len(errors) == 0, errors


def validate_manifest_schema(manifest: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a manifest against the expected schema.

    Args:
        manifest: Manifest dict to validate

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    # Check _schema section
    if "_schema" not in manifest:
        errors.append("Missing '_schema' section")
    else:
        schema = manifest["_schema"]
        if schema.get("version") != MANIFEST_SCHEMA_VERSION:
            errors.append(
                f"Schema version mismatch: expected {MANIFEST_SCHEMA_VERSION}, "
                f"got {schema.get('version')}"
            )

    # Check required keys
    required_keys = ["run_id", "params", "quarters", "input_hashes"]
    for key in required_keys:
        if key not in manifest:
            errors.append(f"Missing required key: '{key}'")

    # Validate quarters array
    if "quarters" in manifest:
        quarters = manifest["quarters"]
        if not isinstance(quarters, list):
            errors.append("'quarters' must be a list")
        else:
            for i, q in enumerate(quarters):
                required_q_keys = ["quarter_end", "filename", "sha256"]
                for key in required_q_keys:
                    if key not in q:
                        errors.append(f"Quarter {i} missing '{key}'")

    return len(errors) == 0, errors


# =============================================================================
# SNAPSHOT WRITING
# =============================================================================

def write_snapshot(
    snapshot: Dict[str, Any],
    quarter_end: Union[date, str],
    out_dir: Union[str, Path],
) -> Tuple[Path, str]:
    """
    Write a snapshot to disk with canonical JSON formatting.

    Args:
        snapshot: Snapshot data
        quarter_end: Quarter end date
        out_dir: Output directory

    Returns:
        Tuple of (filepath, sha256_hash)
    """
    from governance.canonical_json import canonical_dumps
    from governance.hashing import hash_bytes

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(quarter_end, date):
        quarter_end_str = quarter_end.isoformat()
    else:
        quarter_end_str = quarter_end

    filename = _get_snapshot_filename(quarter_end_str)
    filepath = out_dir / filename

    # Serialize canonically
    content = canonical_dumps(snapshot)
    content_bytes = content.encode('utf-8')

    # Compute hash
    file_hash = hash_bytes(content_bytes)

    # Write
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    return filepath, file_hash


def write_manifest(
    manifest: Dict[str, Any],
    out_dir: Union[str, Path],
) -> Tuple[Path, str]:
    """
    Write manifest to disk with canonical JSON formatting.

    Args:
        manifest: Manifest data
        out_dir: Output directory

    Returns:
        Tuple of (filepath, sha256_hash)
    """
    from governance.canonical_json import canonical_dumps
    from governance.hashing import hash_bytes

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filepath = out_dir / "manifest.json"

    # Serialize canonically
    content = canonical_dumps(manifest)
    content_bytes = content.encode('utf-8')

    # Compute hash
    file_hash = hash_bytes(content_bytes)

    # Write
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    return filepath, file_hash
