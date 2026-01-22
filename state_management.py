#!/usr/bin/env python3
"""
state_management.py - Trial State Snapshot Management

Manages trial state snapshots in JSONL format with sorted keys.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import cached_property
from pathlib import Path
from typing import Optional
import json
import hashlib
import logging

from ctgov_adapter import CanonicalTrialRecord

logger = logging.getLogger(__name__)


# ============================================================================
# STATE SNAPSHOT
# ============================================================================

@dataclass
class StateSnapshot:
    """
    Snapshot of all trial states at a given date
    
    Records stored sorted by (ticker, nct_id) for stable diffs
    """
    snapshot_date: date
    records: list[CanonicalTrialRecord]
    
    def __post_init__(self):
        """Ensure records are sorted"""
        self.records = sorted(self.records, key=lambda r: (r.ticker, r.nct_id))
    
    @property
    def key_count(self) -> int:
        return len(self.records)
    
    @property
    def ticker_count(self) -> int:
        return len(set(r.ticker for r in self.records))
    
    def get_record(self, ticker: str, nct_id: str) -> Optional[CanonicalTrialRecord]:
        """Binary search for record (since sorted)"""
        key = (ticker, nct_id)
        left, right = 0, len(self.records) - 1

        while left <= right:
            mid = (left + right) // 2
            mid_key = (self.records[mid].ticker, self.records[mid].nct_id)

            if mid_key == key:
                return self.records[mid]
            elif mid_key < key:
                left = mid + 1
            else:
                right = mid - 1

        return None

    @cached_property
    def records_by_nct_id(self) -> dict:
        """
        Lazily build and cache NCT ID -> record lookup dict.

        This is useful for cross-snapshot comparisons where ticker
        association may change but NCT ID is stable.

        Returns:
            Dict mapping nct_id -> CanonicalTrialRecord
        """
        return {record.nct_id: record for record in self.records}

    def get_record_by_nct_id(self, nct_id: str) -> Optional[CanonicalTrialRecord]:
        """
        O(1) lookup by NCT ID using cached dict.

        Args:
            nct_id: ClinicalTrials.gov NCT identifier

        Returns:
            CanonicalTrialRecord or None if not found
        """
        return self.records_by_nct_id.get(nct_id)

    def to_dict(self) -> dict:
        """Serialize for manifest"""
        return {
            'snapshot_date': self.snapshot_date.isoformat(),
            'ticker_count': self.ticker_count,
            'record_count': self.key_count,
            'first_key': f"{self.records[0].ticker}|{self.records[0].nct_id}" if self.records else None,
            'last_key': f"{self.records[-1].ticker}|{self.records[-1].nct_id}" if self.records else None
        }


# ============================================================================
# STATE STORE
# ============================================================================

class StateStore:
    """
    Manages trial state snapshots in JSONL format
    
    - Single JSONL file per snapshot date
    - Records stored sorted by (ticker, nct_id)
    - Fast binary search for lookups
    """
    
    def __init__(self, state_dir: Path):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.state_dir / "manifest.json"
    
    def save_snapshot(self, snapshot: StateSnapshot) -> Path:
        """Save snapshot as JSONL"""
        filepath = self._get_snapshot_path(snapshot.snapshot_date)
        
        # Write JSONL (one line per record, pre-sorted)
        with open(filepath, 'w') as f:
            for record in snapshot.records:
                f.write(json.dumps(record.to_dict(), sort_keys=True) + '\n')
        
        # Update manifest
        self._update_manifest(snapshot, filepath)
        
        logger.info(f"Saved snapshot: {filepath} ({snapshot.key_count} records)")
        return filepath
    
    def load_snapshot(self, snapshot_date: date) -> Optional[StateSnapshot]:
        """Load snapshot from JSONL"""
        filepath = self._get_snapshot_path(snapshot_date)
        
        if not filepath.exists():
            logger.warning(f"Snapshot not found: {filepath}")
            return None
        
        records = []
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    record = CanonicalTrialRecord.from_dict(data)
                    records.append(record)
                except Exception as e:
                    logger.error(f"Failed to parse line {line_num}: {e}")
        
        snapshot = StateSnapshot(snapshot_date=snapshot_date, records=records)
        logger.info(f"Loaded snapshot: {filepath} ({snapshot.key_count} records)")
        return snapshot
    
    def list_snapshots(self) -> list[date]:
        """List all available snapshot dates"""
        snapshots = []
        for filepath in sorted(self.state_dir.glob("state_*.jsonl")):
            try:
                date_str = filepath.stem.replace("state_", "")
                snapshot_date = date.fromisoformat(date_str)
                snapshots.append(snapshot_date)
            except ValueError:
                logger.warning(f"Invalid snapshot filename: {filepath.name}")
        
        return sorted(snapshots)
    
    def get_most_recent_snapshot(self) -> Optional[StateSnapshot]:
        """Get the most recent snapshot"""
        snapshots = self.list_snapshots()
        if not snapshots:
            return None
        return self.load_snapshot(snapshots[-1])

    def get_prior_snapshot(self, before_date: date) -> Optional[StateSnapshot]:
        """
        Get the most recent snapshot strictly before the given date.

        Used for delta comparison when running for a date that already has a snapshot.

        Args:
            before_date: The reference date - returns most recent snapshot < this date

        Returns:
            StateSnapshot if one exists before the date, None otherwise
        """
        snapshots = self.list_snapshots()
        prior_snapshots = [d for d in snapshots if d < before_date]
        if not prior_snapshots:
            return None
        return self.load_snapshot(prior_snapshots[-1])

    def get_snapshot_for_window(
        self,
        as_of_date: date,
        lookback_days: int,
    ) -> Optional[StateSnapshot]:
        """
        Get the snapshot closest to (as_of_date - lookback_days).

        For time-decay scoring, we want to compare against a snapshot
        from approximately `lookback_days` ago. This method finds the
        closest available snapshot to that target date.

        Args:
            as_of_date: Reference date
            lookback_days: Days to look back

        Returns:
            StateSnapshot closest to target date, None if no snapshots exist
        """
        target_date = as_of_date - timedelta(days=lookback_days)
        snapshots = self.list_snapshots()

        if not snapshots:
            return None

        # Filter to snapshots on or before target
        valid_snapshots = [d for d in snapshots if d <= target_date]

        if not valid_snapshots:
            # If no snapshots exist at/before target, use earliest available
            # but only if it's before as_of_date (PIT safety)
            valid_snapshots = [d for d in snapshots if d < as_of_date]
            if not valid_snapshots:
                return None

        # Use most recent snapshot at/before target
        best_date = valid_snapshots[-1]
        return self.load_snapshot(best_date)

    def get_snapshots_for_time_decay(
        self,
        as_of_date: date,
        lookback_windows: list[int],
    ) -> dict[int, Optional[StateSnapshot]]:
        """
        Get snapshots for multiple time-decay windows.

        Efficiently loads snapshots needed for multi-window comparison.

        Args:
            as_of_date: Reference date
            lookback_windows: List of lookback days (e.g., [7, 30, 90])

        Returns:
            Dict mapping lookback_days to StateSnapshot (or None)
        """
        results = {}
        for lookback_days in lookback_windows:
            results[lookback_days] = self.get_snapshot_for_window(
                as_of_date, lookback_days
            )
            if results[lookback_days]:
                logger.info(
                    f"Time-decay window {lookback_days}d: using snapshot from "
                    f"{results[lookback_days].snapshot_date}"
                )
            else:
                logger.warning(
                    f"Time-decay window {lookback_days}d: no snapshot available"
                )
        return results
    
    def _get_snapshot_path(self, snapshot_date: date) -> Path:
        """Get filepath for snapshot date"""
        return self.state_dir / f"state_{snapshot_date.isoformat()}.jsonl"
    
    def _update_manifest(self, snapshot: StateSnapshot, filepath: Path):
        """Update manifest.json with snapshot metadata"""
        # Load existing manifest
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                manifest = json.load(f)
        else:
            manifest = {"snapshots": []}
        
        # Compute file hash
        file_hash = self._compute_file_hash(filepath)
        
        # Add/update snapshot entry
        snapshot_entry = {
            **snapshot.to_dict(),
            'file_size_mb': round(filepath.stat().st_size / 1024 / 1024, 2),
            'sha256': file_hash
        }
        
        # Remove existing entry for this date
        manifest["snapshots"] = [
            s for s in manifest["snapshots"] 
            if s['snapshot_date'] != snapshot.snapshot_date.isoformat()
        ]
        
        # Add new entry
        manifest["snapshots"].append(snapshot_entry)
        
        # Sort by date
        manifest["snapshots"].sort(key=lambda s: s['snapshot_date'])
        
        # Write manifest
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
    
    @staticmethod
    def _compute_file_hash(filepath: Path) -> str:
        """Compute SHA256 hash of file"""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()


if __name__ == "__main__":
    print("State Management loaded successfully")
