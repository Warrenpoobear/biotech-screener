#!/usr/bin/env python3
"""
archive_daily_snapshot.py - Daily CT.gov Snapshot Archiver

This script is designed to run as a cron job to build historical CT.gov data.
Since CT.gov API only provides current state (not historical), we must build
our own historical archive by saving daily snapshots.

Usage:
    # Run manually
    python archive_daily_snapshot.py --as-of-date 2026-01-19 --state-dir production_data/ctgov_state

    # Run via cron (recommended: 2 AM daily)
    # Add to crontab -e:
    # 0 2 * * * cd /path/to/biotech-screener && python archive_daily_snapshot.py >> logs/snapshot.log 2>&1

Features:
- Fetches fresh trial data from CT.gov (if fetch script exists)
- Creates a dated snapshot in JSONL format
- Maintains manifest.json with snapshot metadata
- Validates snapshot integrity (record counts, hash)
- Supports dry-run mode for testing

PIT Safety:
- as_of_date is REQUIRED (no defaults to datetime.now())
- Snapshots are immutable once created
- All timestamps from source data

Determinism:
- Sorted records by (ticker, nct_id)
- Canonical JSON serialization
- SHA256 hash for verification
"""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_STATE_DIR = "production_data/ctgov_state"
DEFAULT_TRIAL_RECORDS = "production_data/trial_records.json"
DEFAULT_UNIVERSE = "production_data/universe.json"
MANIFEST_FILE = "manifest.json"

# CT.gov fetch script (if exists)
FETCH_SCRIPT = "wake_robin_data_pipeline/ctgov_collector.py"


# ============================================================================
# SNAPSHOT ARCHIVER
# ============================================================================

class SnapshotArchiver:
    """
    Archives daily CT.gov snapshots for historical analysis.

    This builds the historical data that CT.gov API doesn't provide.
    """

    def __init__(
        self,
        state_dir: Path,
        trial_records_path: Path,
        universe_path: Optional[Path] = None,
    ):
        self.state_dir = Path(state_dir)
        self.trial_records_path = Path(trial_records_path)
        self.universe_path = Path(universe_path) if universe_path else None

        # Ensure state directory exists
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Load or create manifest
        self.manifest_path = self.state_dir / MANIFEST_FILE
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        """Load or create the snapshot manifest."""
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)
        return {
            "version": "1.0.0",
            "created_at": None,  # Will be set on first snapshot
            "snapshots": [],
        }

    def _save_manifest(self) -> None:
        """Save the manifest with sorted keys."""
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2, sort_keys=True)

    def fetch_fresh_data(self, as_of_date: date) -> bool:
        """
        Fetch fresh CT.gov data if fetch script exists.

        Returns True if data was fetched successfully.
        """
        fetch_script = Path(FETCH_SCRIPT)
        if not fetch_script.exists():
            logger.warning(f"Fetch script not found: {fetch_script}")
            logger.warning("Using existing trial_records.json (may be stale)")
            return False

        logger.info(f"Fetching fresh CT.gov data...")
        try:
            result = subprocess.run(
                [sys.executable, str(fetch_script), "--as-of-date", as_of_date.isoformat()],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )
            if result.returncode != 0:
                logger.error(f"Fetch failed: {result.stderr}")
                return False
            logger.info("Fresh data fetched successfully")
            return True
        except subprocess.TimeoutExpired:
            logger.error("Fetch script timed out after 10 minutes")
            return False
        except Exception as e:
            logger.error(f"Fetch failed: {e}")
            return False

    def load_trial_records(self, as_of_date: date) -> List[Dict[str, Any]]:
        """Load trial records from JSON file."""
        if not self.trial_records_path.exists():
            raise FileNotFoundError(f"Trial records not found: {self.trial_records_path}")

        with open(self.trial_records_path) as f:
            records = json.load(f)

        logger.info(f"Loaded {len(records)} trial records")
        return records

    def load_active_tickers(self) -> set:
        """Load active tickers from universe file."""
        if not self.universe_path or not self.universe_path.exists():
            logger.warning("Universe file not specified or not found - including all tickers")
            return set()

        with open(self.universe_path) as f:
            universe = json.load(f)

        tickers = {s['ticker'] for s in universe if s.get('ticker') != '_XBI_BENCHMARK_'}
        logger.info(f"Loaded {len(tickers)} active tickers from universe")
        return tickers

    def create_snapshot(
        self,
        as_of_date: date,
        trial_records: List[Dict[str, Any]],
        active_tickers: Optional[set] = None,
    ) -> Dict[str, Any]:
        """
        Create a snapshot from trial records.

        Returns snapshot metadata.
        """
        # Filter to active tickers if specified
        if active_tickers:
            trial_records = [r for r in trial_records if r.get('ticker') in active_tickers]
            logger.info(f"Filtered to {len(trial_records)} records for active tickers")

        # Sort records for determinism
        trial_records.sort(key=lambda r: (r.get('ticker', ''), r.get('nct_id', '')))

        # Create snapshot file
        snapshot_file = self.state_dir / f"state_{as_of_date.isoformat()}.jsonl"

        if snapshot_file.exists():
            logger.warning(f"Snapshot already exists: {snapshot_file}")
            logger.warning("Skipping to prevent overwriting historical data")
            return self._get_existing_snapshot_metadata(as_of_date)

        # Write JSONL format (one record per line)
        with open(snapshot_file, 'w') as f:
            for record in trial_records:
                f.write(json.dumps(record, sort_keys=True) + '\n')

        # Compute file metadata
        file_size = snapshot_file.stat().st_size
        file_hash = self._compute_file_hash(snapshot_file)

        # Count unique tickers
        tickers = set(r.get('ticker') for r in trial_records)

        metadata = {
            "snapshot_date": as_of_date.isoformat(),
            "file_name": snapshot_file.name,
            "record_count": len(trial_records),
            "ticker_count": len(tickers),
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "sha256": file_hash,
        }

        logger.info(f"Created snapshot: {snapshot_file.name}")
        logger.info(f"  Records: {metadata['record_count']}")
        logger.info(f"  Tickers: {metadata['ticker_count']}")
        logger.info(f"  Size: {metadata['file_size_mb']} MB")
        logger.info(f"  Hash: {file_hash[:16]}...")

        return metadata

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _get_existing_snapshot_metadata(self, as_of_date: date) -> Dict[str, Any]:
        """Get metadata for an existing snapshot."""
        for snapshot in self.manifest.get('snapshots', []):
            if snapshot.get('snapshot_date') == as_of_date.isoformat():
                return snapshot
        return {"snapshot_date": as_of_date.isoformat(), "status": "already_exists"}

    def update_manifest(self, metadata: Dict[str, Any]) -> None:
        """Update manifest with new snapshot metadata."""
        # Check if snapshot already in manifest
        existing_dates = {s.get('snapshot_date') for s in self.manifest.get('snapshots', [])}
        if metadata.get('snapshot_date') in existing_dates:
            logger.info("Snapshot already in manifest - skipping update")
            return

        # Add new snapshot
        self.manifest['snapshots'].append(metadata)

        # Sort snapshots by date (oldest first)
        self.manifest['snapshots'].sort(key=lambda s: s.get('snapshot_date', ''))

        # Update created_at if first snapshot
        if self.manifest.get('created_at') is None:
            self.manifest['created_at'] = metadata['snapshot_date']

        # Save manifest
        self._save_manifest()
        logger.info("Manifest updated")

    def archive_snapshot(
        self,
        as_of_date: date,
        fetch_fresh: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Main entry point: Archive a daily snapshot.

        Args:
            as_of_date: Date for the snapshot (REQUIRED)
            fetch_fresh: Whether to fetch fresh data first
            dry_run: If True, don't actually create files

        Returns:
            Snapshot metadata
        """
        logger.info(f"{'[DRY RUN] ' if dry_run else ''}Archiving snapshot for {as_of_date}")

        # Optionally fetch fresh data
        if fetch_fresh:
            self.fetch_fresh_data(as_of_date)

        # Load trial records
        trial_records = self.load_trial_records(as_of_date)

        # Load active tickers
        active_tickers = self.load_active_tickers()

        if dry_run:
            logger.info("[DRY RUN] Would create snapshot with:")
            logger.info(f"  Records: {len(trial_records)}")
            logger.info(f"  Active tickers: {len(active_tickers) if active_tickers else 'all'}")
            return {"status": "dry_run", "snapshot_date": as_of_date.isoformat()}

        # Create snapshot
        metadata = self.create_snapshot(as_of_date, trial_records, active_tickers)

        # Update manifest
        self.update_manifest(metadata)

        return metadata

    def get_snapshot_history(self) -> List[Dict[str, Any]]:
        """Get list of all archived snapshots."""
        return self.manifest.get('snapshots', [])

    def get_snapshot_coverage(self, start_date: date, end_date: date) -> Dict[str, Any]:
        """
        Check snapshot coverage for a date range.

        Returns coverage statistics.
        """
        snapshots = self.get_snapshot_history()
        snapshot_dates = {s.get('snapshot_date') for s in snapshots}

        total_days = (end_date - start_date).days + 1
        covered_days = 0
        missing_dates = []

        current = start_date
        while current <= end_date:
            if current.isoformat() in snapshot_dates:
                covered_days += 1
            else:
                missing_dates.append(current.isoformat())
            current += timedelta(days=1)

        return {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "total_days": total_days,
            "covered_days": covered_days,
            "coverage_percent": round(100 * covered_days / total_days, 1),
            "missing_dates": missing_dates[:10],  # First 10 missing
            "missing_count": len(missing_dates),
        }


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Command-line interface for snapshot archiver."""
    parser = argparse.ArgumentParser(
        description='Archive daily CT.gov snapshots for historical analysis',
        epilog='''
Examples:
    # Archive snapshot for today
    python archive_daily_snapshot.py --as-of-date 2026-01-19

    # Archive with fresh data fetch
    python archive_daily_snapshot.py --as-of-date 2026-01-19 --fetch-fresh

    # Check coverage for last 90 days
    python archive_daily_snapshot.py --check-coverage --days 90

    # Dry run (no files created)
    python archive_daily_snapshot.py --as-of-date 2026-01-19 --dry-run

Cron setup (recommended: 2 AM daily):
    0 2 * * * cd /path/to/biotech-screener && python archive_daily_snapshot.py --as-of-date $(date +\\%Y-\\%m-\\%d) --fetch-fresh >> logs/snapshot.log 2>&1
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--as-of-date',
        type=str,
        help='Snapshot date (YYYY-MM-DD) - REQUIRED for archive'
    )
    parser.add_argument(
        '--state-dir',
        type=str,
        default=DEFAULT_STATE_DIR,
        help=f'State directory for snapshots (default: {DEFAULT_STATE_DIR})'
    )
    parser.add_argument(
        '--trial-records',
        type=str,
        default=DEFAULT_TRIAL_RECORDS,
        help=f'Path to trial_records.json (default: {DEFAULT_TRIAL_RECORDS})'
    )
    parser.add_argument(
        '--universe',
        type=str,
        default=DEFAULT_UNIVERSE,
        help=f'Path to universe.json (default: {DEFAULT_UNIVERSE})'
    )
    parser.add_argument(
        '--fetch-fresh',
        action='store_true',
        help='Fetch fresh CT.gov data before archiving'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run - show what would be done without creating files'
    )
    parser.add_argument(
        '--check-coverage',
        action='store_true',
        help='Check snapshot coverage instead of archiving'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=90,
        help='Number of days to check coverage (default: 90)'
    )
    parser.add_argument(
        '--list-snapshots',
        action='store_true',
        help='List all archived snapshots'
    )

    args = parser.parse_args()

    # Initialize archiver
    archiver = SnapshotArchiver(
        state_dir=Path(args.state_dir),
        trial_records_path=Path(args.trial_records),
        universe_path=Path(args.universe) if args.universe else None,
    )

    # Handle different modes
    if args.list_snapshots:
        snapshots = archiver.get_snapshot_history()
        print(f"\nArchived Snapshots ({len(snapshots)} total):")
        print("-" * 60)
        for s in snapshots[-20:]:  # Last 20
            print(f"  {s.get('snapshot_date')}: "
                  f"{s.get('record_count', '?')} records, "
                  f"{s.get('ticker_count', '?')} tickers, "
                  f"{s.get('file_size_mb', '?')} MB")
        if len(snapshots) > 20:
            print(f"  ... ({len(snapshots) - 20} more)")
        return

    if args.check_coverage:
        if not args.as_of_date:
            # Default to today for coverage check
            end_date = date.today()
        else:
            end_date = date.fromisoformat(args.as_of_date)

        start_date = end_date - timedelta(days=args.days)
        coverage = archiver.get_snapshot_coverage(start_date, end_date)

        print(f"\nSnapshot Coverage Report:")
        print("-" * 60)
        print(f"  Date range: {coverage['start_date']} to {coverage['end_date']}")
        print(f"  Total days: {coverage['total_days']}")
        print(f"  Covered days: {coverage['covered_days']}")
        print(f"  Coverage: {coverage['coverage_percent']}%")

        if coverage['missing_dates']:
            print(f"\n  Missing dates ({coverage['missing_count']} total):")
            for d in coverage['missing_dates']:
                print(f"    - {d}")
            if coverage['missing_count'] > 10:
                print(f"    ... ({coverage['missing_count'] - 10} more)")
        return

    # Archive mode requires as_of_date
    if not args.as_of_date:
        parser.error("--as-of-date is required for archiving (PIT safety)")

    as_of_date = date.fromisoformat(args.as_of_date)

    # Archive snapshot
    result = archiver.archive_snapshot(
        as_of_date=as_of_date,
        fetch_fresh=args.fetch_fresh,
        dry_run=args.dry_run,
    )

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Archive Complete:")
    print("-" * 60)
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
