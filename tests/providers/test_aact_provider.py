"""
Unit tests for AACT Clinical Trials Provider.

Tests cover:
    1. Snapshot selection with PIT constraints
    2. Canonical TrialRow serialization and hashing
    3. Diff counts (PCD pushes, status flips)
"""

import csv
import tempfile
from datetime import date
from pathlib import Path
from unittest import TestCase

from src.common.hash_utils import compute_trial_facts_hash
from src.providers.aact_provider import AACTClinicalTrialsProvider, load_trial_mapping
from src.providers.protocols import Phase, PCDType, TrialRow, TrialStatus


class TestSnapshotSelection(TestCase):
    """Test snapshot selection with PIT constraints."""
    
    def setUp(self):
        """Create temp directory with mock snapshots."""
        self.temp_dir = tempfile.mkdtemp()
        self.snapshots_root = Path(self.temp_dir) / "aact_snapshots"
        self.snapshots_root.mkdir()
        
        # Create snapshot folders for 2024-01-15 and 2024-01-29
        for snapshot_date in ["2024-01-15", "2024-01-29"]:
            folder = self.snapshots_root / snapshot_date
            folder.mkdir()
            
            # Create minimal studies.csv
            with open(folder / "studies.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["nct_id", "phase", "overall_status", "primary_completion_date"])
                writer.writerow(["NCT00000001", "Phase 2", "Recruiting", "2024-06-01"])
    
    def test_strict_pit_excludes_same_day(self):
        """With strict PIT, pit_cutoff=2024-01-29 should select 2024-01-15."""
        provider = AACTClinicalTrialsProvider(
            snapshots_root=self.snapshots_root,
            strict_pit=True,  # < pit_cutoff
        )
        
        # pit_cutoff = 2024-01-29 -> must select snapshot < 2024-01-29
        selected = provider.select_snapshot(pit_cutoff=date(2024, 1, 29))
        
        # Should select 2024-01-15 (not 2024-01-29)
        self.assertEqual(selected, date(2024, 1, 15))
    
    def test_non_strict_pit_includes_same_day(self):
        """With non-strict PIT, pit_cutoff=2024-01-29 can select 2024-01-29."""
        provider = AACTClinicalTrialsProvider(
            snapshots_root=self.snapshots_root,
            strict_pit=False,  # <= pit_cutoff
        )
        
        selected = provider.select_snapshot(pit_cutoff=date(2024, 1, 29))
        
        # Should select 2024-01-29 (included)
        self.assertEqual(selected, date(2024, 1, 29))
    
    def test_selects_latest_valid_snapshot(self):
        """Should select the latest snapshot that satisfies constraint."""
        provider = AACTClinicalTrialsProvider(
            snapshots_root=self.snapshots_root,
            strict_pit=True,
        )
        
        # pit_cutoff = 2024-01-30 -> can select up to < 2024-01-30
        selected = provider.select_snapshot(pit_cutoff=date(2024, 1, 30))
        
        # Should select 2024-01-29 (latest before 2024-01-30)
        self.assertEqual(selected, date(2024, 1, 29))
    
    def test_returns_none_when_no_valid_snapshot(self):
        """Should return None if no snapshot satisfies constraint."""
        provider = AACTClinicalTrialsProvider(
            snapshots_root=self.snapshots_root,
            strict_pit=True,
        )
        
        # pit_cutoff = 2024-01-15 -> must select snapshot < 2024-01-15
        selected = provider.select_snapshot(pit_cutoff=date(2024, 1, 15))
        
        # No snapshot before 2024-01-15
        self.assertIsNone(selected)


class TestTrialRowSerialization(TestCase):
    """Test canonical TrialRow serialization and hashing."""
    
    def test_order_independent_hashing(self):
        """Same trials in different order should produce same hash."""
        trial_a = TrialRow(
            nct_id="NCT00000001",
            phase=Phase.P2,
            overall_status=TrialStatus.RECRUITING,
            primary_completion_date=date(2024, 6, 1),
            primary_completion_date_type=PCDType.ANTICIPATED,
            last_update_posted_date=date(2024, 1, 15),
            lead_sponsor="Acme Inc",
        )
        
        trial_b = TrialRow(
            nct_id="NCT00000002",
            phase=Phase.P3,
            overall_status=TrialStatus.ACTIVE,
            primary_completion_date=date(2024, 9, 1),
            primary_completion_date_type=PCDType.ANTICIPATED,
            last_update_posted_date=date(2024, 1, 10),
            lead_sponsor="BioTech Co",
        )
        
        # Order 1: A, B
        trials_by_ticker_1 = {
            "ACME": [trial_a],
            "BTCH": [trial_b],
        }
        
        # Order 2: B, A (different dict insertion order)
        trials_by_ticker_2 = {
            "BTCH": [trial_b],
            "ACME": [trial_a],
        }
        
        hash_1 = compute_trial_facts_hash(trials_by_ticker_1)
        hash_2 = compute_trial_facts_hash(trials_by_ticker_2)
        
        # Hashes should be identical
        self.assertEqual(hash_1, hash_2)
    
    def test_deterministic_sorting(self):
        """Trials within a ticker should be sorted by nct_id."""
        trial_a = TrialRow(
            nct_id="NCT00000003",
            phase=Phase.P2,
            overall_status=TrialStatus.RECRUITING,
            primary_completion_date=None,
            primary_completion_date_type=PCDType.UNKNOWN,
            last_update_posted_date=None,
            lead_sponsor="Test",
        )
        
        trial_b = TrialRow(
            nct_id="NCT00000001",
            phase=Phase.P1,
            overall_status=TrialStatus.COMPLETED,
            primary_completion_date=None,
            primary_completion_date_type=PCDType.UNKNOWN,
            last_update_posted_date=None,
            lead_sponsor="Test",
        )
        
        # Insert in "wrong" order
        trials_1 = {"TEST": [trial_a, trial_b]}
        trials_2 = {"TEST": [trial_b, trial_a]}
        
        hash_1 = compute_trial_facts_hash(trials_1)
        hash_2 = compute_trial_facts_hash(trials_2)
        
        # Should produce same hash due to sorting
        self.assertEqual(hash_1, hash_2)
    
    def test_hash_changes_with_data_change(self):
        """Hash should change when trial data changes."""
        trial_original = TrialRow(
            nct_id="NCT00000001",
            phase=Phase.P2,
            overall_status=TrialStatus.RECRUITING,
            primary_completion_date=date(2024, 6, 1),
            primary_completion_date_type=PCDType.ANTICIPATED,
            last_update_posted_date=date(2024, 1, 15),
            lead_sponsor="Acme Inc",
        )
        
        trial_modified = TrialRow(
            nct_id="NCT00000001",
            phase=Phase.P2,
            overall_status=TrialStatus.RECRUITING,
            primary_completion_date=date(2024, 9, 1),  # PCD changed
            primary_completion_date_type=PCDType.ANTICIPATED,
            last_update_posted_date=date(2024, 1, 15),
            lead_sponsor="Acme Inc",
        )
        
        hash_original = compute_trial_facts_hash({"TEST": [trial_original]})
        hash_modified = compute_trial_facts_hash({"TEST": [trial_modified]})
        
        # Hashes should differ
        self.assertNotEqual(hash_original, hash_modified)


class TestDiffCounts(TestCase):
    """Test PCD push and status flip counting."""
    
    def setUp(self):
        """Create temp directory with snapshots for diff testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.snapshots_root = Path(self.temp_dir) / "aact_snapshots"
        self.snapshots_root.mkdir()
    
    def _create_snapshot(self, snapshot_date: str, trials: list[dict]):
        """Helper to create a snapshot with given trials."""
        folder = self.snapshots_root / snapshot_date
        folder.mkdir(exist_ok=True)
        
        # Ensure all required columns are present
        required_columns = [
            "nct_id", "phase", "overall_status", "primary_completion_date",
            "primary_completion_date_type", "last_update_posted_date", "study_type"
        ]
        
        with open(folder / "studies.csv", "w", newline="") as f:
            if trials:
                # Add study_type default if missing
                for trial in trials:
                    if "study_type" not in trial:
                        trial["study_type"] = "Interventional"
                
                writer = csv.DictWriter(f, fieldnames=required_columns)
                writer.writeheader()
                writer.writerows(trials)
            else:
                writer = csv.writer(f)
                writer.writerow(required_columns)
        
        # Empty sponsors file with required columns
        with open(folder / "sponsors.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["nct_id", "name", "lead_or_collaborator"])
    
    def test_pcd_push_counted(self):
        """PCD moving to later date should count as push."""
        # Snapshot 1: PCD = 2024-06-01
        self._create_snapshot("2024-01-01", [{
            "nct_id": "NCT00000001",
            "phase": "Phase 2",
            "overall_status": "Recruiting",
            "primary_completion_date": "2024-06-01",
            "primary_completion_date_type": "Anticipated",
            "last_update_posted_date": "2024-01-01",
        }])
        
        # Snapshot 2: PCD pushed to 2024-09-01
        self._create_snapshot("2024-02-01", [{
            "nct_id": "NCT00000001",
            "phase": "Phase 2",
            "overall_status": "Recruiting",
            "primary_completion_date": "2024-09-01",  # Pushed 3 months
            "primary_completion_date_type": "Anticipated",
            "last_update_posted_date": "2024-02-01",
        }])
        
        provider = AACTClinicalTrialsProvider(
            snapshots_root=self.snapshots_root,
            strict_pit=False,
        )
        
        snapshots = [
            provider.load_snapshot(date(2024, 1, 1)),
            provider.load_snapshot(date(2024, 2, 1)),
        ]
        
        pcd_pushes, status_flips = provider._compute_diffs(
            "NCT00000001",
            snapshots,
            lookback_days=365,
            as_of_date=date(2024, 3, 1),
        )
        
        self.assertEqual(pcd_pushes, 1)
        self.assertEqual(status_flips, 0)
    
    def test_status_flip_active_to_suspended(self):
        """Status change Active -> Suspended should count as flip."""
        self._create_snapshot("2024-01-01", [{
            "nct_id": "NCT00000001",
            "phase": "Phase 2",
            "overall_status": "Active, not recruiting",
            "primary_completion_date": "2024-06-01",
            "primary_completion_date_type": "Anticipated",
            "last_update_posted_date": "2024-01-01",
        }])
        
        self._create_snapshot("2024-02-01", [{
            "nct_id": "NCT00000001",
            "phase": "Phase 2",
            "overall_status": "Suspended",  # Status changed
            "primary_completion_date": "2024-06-01",
            "primary_completion_date_type": "Anticipated",
            "last_update_posted_date": "2024-02-01",
        }])
        
        provider = AACTClinicalTrialsProvider(
            snapshots_root=self.snapshots_root,
            strict_pit=False,
        )
        
        snapshots = [
            provider.load_snapshot(date(2024, 1, 1)),
            provider.load_snapshot(date(2024, 2, 1)),
        ]
        
        pcd_pushes, status_flips = provider._compute_diffs(
            "NCT00000001",
            snapshots,
            lookback_days=365,
            as_of_date=date(2024, 3, 1),
        )
        
        self.assertEqual(pcd_pushes, 0)
        self.assertEqual(status_flips, 1)
    
    def test_multiple_changes_across_snapshots(self):
        """Multiple changes across multiple snapshots should accumulate."""
        # Snapshot 1: Active, PCD = 2024-06-01
        self._create_snapshot("2024-01-01", [{
            "nct_id": "NCT00000001",
            "phase": "Phase 2",
            "overall_status": "Recruiting",
            "primary_completion_date": "2024-06-01",
            "primary_completion_date_type": "Anticipated",
            "last_update_posted_date": "2024-01-01",
        }])
        
        # Snapshot 2: Suspended, PCD pushed
        self._create_snapshot("2024-02-01", [{
            "nct_id": "NCT00000001",
            "phase": "Phase 2",
            "overall_status": "Suspended",
            "primary_completion_date": "2024-09-01",
            "primary_completion_date_type": "Anticipated",
            "last_update_posted_date": "2024-02-01",
        }])
        
        # Snapshot 3: Back to Recruiting, PCD pushed again
        self._create_snapshot("2024-03-01", [{
            "nct_id": "NCT00000001",
            "phase": "Phase 2",
            "overall_status": "Recruiting",
            "primary_completion_date": "2024-12-01",
            "primary_completion_date_type": "Anticipated",
            "last_update_posted_date": "2024-03-01",
        }])
        
        provider = AACTClinicalTrialsProvider(
            snapshots_root=self.snapshots_root,
            strict_pit=False,
        )
        
        snapshots = [
            provider.load_snapshot(date(2024, 1, 1)),
            provider.load_snapshot(date(2024, 2, 1)),
            provider.load_snapshot(date(2024, 3, 1)),
        ]
        
        pcd_pushes, status_flips = provider._compute_diffs(
            "NCT00000001",
            snapshots,
            lookback_days=365,
            as_of_date=date(2024, 4, 1),
        )
        
        # 2 PCD pushes: 06-01 -> 09-01, 09-01 -> 12-01
        self.assertEqual(pcd_pushes, 2)
        # 2 status flips: Recruiting -> Suspended, Suspended -> Recruiting
        self.assertEqual(status_flips, 2)


class TestTrialMapping(TestCase):
    """Test trial mapping loading."""
    
    def test_load_trial_mapping(self):
        """Should load ticker -> NCT ID mapping from CSV."""
        temp_dir = tempfile.mkdtemp()
        mapping_file = Path(temp_dir) / "trial_mapping.csv"
        
        with open(mapping_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ticker", "nct_id", "effective_start", "effective_end", "source"])
            writer.writerow(["MRNA", "NCT04470427", "2020-07-01", "", "manual"])
            writer.writerow(["MRNA", "NCT04860297", "2021-01-01", "", "manual"])
            writer.writerow(["BNTX", "NCT04368728", "2020-04-29", "", "manual"])
        
        mapping = load_trial_mapping(mapping_file)
        
        self.assertEqual(len(mapping), 2)  # 2 tickers
        self.assertEqual(len(mapping["MRNA"]), 2)
        self.assertEqual(len(mapping["BNTX"]), 1)
        self.assertIn("NCT04470427", mapping["MRNA"])
        self.assertIn("NCT04860297", mapping["MRNA"])


if __name__ == "__main__":
    import unittest
    unittest.main()
