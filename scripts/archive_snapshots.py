#!/usr/bin/env python3
"""
archive_snapshots.py - Daily snapshot archiving for time-decay analysis

Maintains a rolling archive of CT.gov state snapshots to enable multi-window
time-decay scoring. Keeps snapshots for configurable retention period.

Usage:
    python scripts/archive_snapshots.py --state-dir production_data/ctgov_state
    python scripts/archive_snapshots.py --state-dir production_data/ctgov_state --retention-days 120

Design:
- DETERMINISTIC: No datetime.now() in core logic
- PIT-SAFE: All operations based on explicit dates
- IDEMPOTENT: Safe to run multiple times per day
"""

import argparse
import json
import logging
import os
import shutil
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_RETENTION_DAYS = 90  # Keep 90 days of history for full time-decay support
ARCHIVE_SUBDIR = "archive"
MANIFEST_FILE = "archive_manifest.json"


# =============================================================================
# ARCHIVE MANAGEMENT
# =============================================================================

def list_snapshots(state_dir: Path) -> List[date]:
    """List all snapshot dates in the state directory."""
    snapshots = []
    for f in state_dir.glob("state_*.jsonl"):
        try:
            date_str = f.stem.replace("state_", "")
            snapshot_date = date.fromisoformat(date_str)
            snapshots.append(snapshot_date)
        except ValueError:
            continue
    return sorted(snapshots)


def archive_snapshot(
    state_dir: Path,
    snapshot_date: date,
    archive_dir: Path,
) -> bool:
    """
    Archive a snapshot to the archive directory.

    Returns True if archived, False if already exists.
    """
    source = state_dir / f"state_{snapshot_date.isoformat()}.jsonl"
    dest = archive_dir / f"state_{snapshot_date.isoformat()}.jsonl"

    if not source.exists():
        logger.warning(f"Source snapshot not found: {source}")
        return False

    if dest.exists():
        # Already archived
        return False

    shutil.copy2(source, dest)
    logger.info(f"Archived: {source.name} -> {archive_dir.name}/")
    return True


def cleanup_old_snapshots(
    archive_dir: Path,
    retention_days: int,
    as_of_date: date,
) -> int:
    """
    Remove snapshots older than retention period.

    Returns count of removed files.
    """
    cutoff_date = as_of_date - timedelta(days=retention_days)
    removed = 0

    for f in archive_dir.glob("state_*.jsonl"):
        try:
            date_str = f.stem.replace("state_", "")
            snapshot_date = date.fromisoformat(date_str)

            if snapshot_date < cutoff_date:
                f.unlink()
                logger.info(f"Removed old snapshot: {f.name} (older than {retention_days} days)")
                removed += 1
        except ValueError:
            continue

    return removed


def update_manifest(
    archive_dir: Path,
    as_of_date: date,
) -> Dict[str, Any]:
    """
    Update the archive manifest with current state.
    """
    snapshots = []
    for f in sorted(archive_dir.glob("state_*.jsonl")):
        try:
            date_str = f.stem.replace("state_", "")
            snapshot_date = date.fromisoformat(date_str)
            stat = f.stat()
            snapshots.append({
                "date": snapshot_date.isoformat(),
                "file": f.name,
                "size_bytes": stat.st_size,
            })
        except (ValueError, OSError):
            continue

    # Compute coverage stats
    if snapshots:
        oldest = snapshots[0]["date"]
        newest = snapshots[-1]["date"]
        coverage_days = (date.fromisoformat(newest) - date.fromisoformat(oldest)).days + 1
        gaps = coverage_days - len(snapshots)
    else:
        oldest = None
        newest = None
        coverage_days = 0
        gaps = 0

    manifest = {
        "last_updated": as_of_date.isoformat(),
        "snapshot_count": len(snapshots),
        "oldest_snapshot": oldest,
        "newest_snapshot": newest,
        "coverage_days": coverage_days,
        "missing_days": gaps,
        "snapshots": snapshots,
    }

    manifest_path = archive_dir / MANIFEST_FILE
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def get_archive_status(archive_dir: Path) -> Dict[str, Any]:
    """Get current archive status for reporting."""
    manifest_path = archive_dir / MANIFEST_FILE

    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)

    return {"snapshot_count": 0, "coverage_days": 0}


# =============================================================================
# MAIN ARCHIVE ROUTINE
# =============================================================================

def run_archive(
    state_dir: str,
    retention_days: int = DEFAULT_RETENTION_DAYS,
    as_of_date: Optional[date] = None,
) -> Dict[str, Any]:
    """
    Main archive routine.

    1. Create archive directory if needed
    2. Archive any new snapshots
    3. Cleanup old snapshots beyond retention
    4. Update manifest

    Args:
        state_dir: Path to ctgov_state directory
        retention_days: Days to retain snapshots
        as_of_date: Reference date (defaults to today)

    Returns:
        Summary dict with archive stats
    """
    state_path = Path(state_dir)
    archive_path = state_path / ARCHIVE_SUBDIR

    if as_of_date is None:
        as_of_date = date.today()

    # Create archive directory
    archive_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Archive directory: {archive_path}")
    logger.info(f"Retention period: {retention_days} days")
    logger.info(f"As-of date: {as_of_date}")

    # List current snapshots
    current_snapshots = list_snapshots(state_path)
    logger.info(f"Found {len(current_snapshots)} snapshots in state directory")

    # Archive new snapshots
    archived_count = 0
    for snapshot_date in current_snapshots:
        if archive_snapshot(state_path, snapshot_date, archive_path):
            archived_count += 1

    # Cleanup old snapshots
    removed_count = cleanup_old_snapshots(archive_path, retention_days, as_of_date)

    # Update manifest
    manifest = update_manifest(archive_path, as_of_date)

    # Report
    logger.info("=" * 60)
    logger.info("ARCHIVE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Archived today:    {archived_count}")
    logger.info(f"  Removed (expired): {removed_count}")
    logger.info(f"  Total in archive:  {manifest['snapshot_count']}")
    logger.info(f"  Coverage period:   {manifest['coverage_days']} days")
    logger.info(f"  Missing days:      {manifest['missing_days']}")
    if manifest['oldest_snapshot']:
        logger.info(f"  Date range:        {manifest['oldest_snapshot']} to {manifest['newest_snapshot']}")

    # Time-decay readiness
    logger.info("=" * 60)
    logger.info("TIME-DECAY WINDOW READINESS")
    logger.info("=" * 60)

    windows = [
        ("7d", 7),
        ("30d", 30),
        ("90d", 90),
    ]

    for name, days in windows:
        if manifest['coverage_days'] >= days:
            status = "✓ READY"
        else:
            remaining = days - manifest['coverage_days']
            status = f"✗ Need {remaining} more days"
        logger.info(f"  {name} window: {status}")

    return {
        "archived": archived_count,
        "removed": removed_count,
        "total": manifest['snapshot_count'],
        "coverage_days": manifest['coverage_days'],
        "oldest": manifest['oldest_snapshot'],
        "newest": manifest['newest_snapshot'],
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Archive CT.gov snapshots for time-decay analysis"
    )
    parser.add_argument(
        "--state-dir",
        required=True,
        help="Path to ctgov_state directory"
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=DEFAULT_RETENTION_DAYS,
        help=f"Days to retain snapshots (default: {DEFAULT_RETENTION_DAYS})"
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="Reference date (YYYY-MM-DD, default: today)"
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Only show current archive status, don't archive"
    )

    args = parser.parse_args()

    as_of = date.fromisoformat(args.as_of_date) if args.as_of_date else None

    if args.status_only:
        archive_path = Path(args.state_dir) / ARCHIVE_SUBDIR
        status = get_archive_status(archive_path)
        print(json.dumps(status, indent=2))
        return

    run_archive(
        state_dir=args.state_dir,
        retention_days=args.retention_days,
        as_of_date=as_of,
    )


if __name__ == "__main__":
    main()
