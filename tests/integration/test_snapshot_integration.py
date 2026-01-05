"""
Integration tests for snapshot generation and comparison.

Tests cover:
    4. Snapshot contains coverage + provenance
    5. A/B compare script handles missing trials gracefully
"""

import csv
import json
import tempfile
from datetime import date
from pathlib import Path
from unittest import TestCase

from src.snapshot_generator import SnapshotConfig, generate_snapshot, save_snapshot


class TestSnapshotIntegration(TestCase):
    """Test complete snapshot generation with AACT provider."""
    
    def setUp(self):
        """Create temp directory with all required data."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_path = Path(self.temp_dir)
        
        # Create universe
        universe_dir = self.base_path / "universe"
        universe_dir.mkdir()
        self.universe_file = universe_dir / "universe.csv"
        with open(self.universe_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ticker"])
            writer.writerow(["MRNA"])
            writer.writerow(["BNTX"])
            writer.writerow(["NOTRIAL"])  # Ticker without trials
        
        # Create AACT snapshots
        self.aact_dir = self.base_path / "aact_snapshots"
        self.aact_dir.mkdir()
        
        snapshot_dir = self.aact_dir / "2024-01-15"
        snapshot_dir.mkdir()
        
        with open(snapshot_dir / "studies.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "nct_id", "phase", "overall_status", 
                "primary_completion_date", "primary_completion_date_type",
                "last_update_posted_date", "study_type"
            ])
            writer.writerow([
                "NCT04470427", "Phase 2", "Recruiting",
                "2024-06-01", "Anticipated", "2024-01-10", "Interventional"
            ])
            writer.writerow([
                "NCT04860297", "Phase 3", "Active, not recruiting",
                "2024-09-01", "Anticipated", "2024-01-12", "Interventional"
            ])
        
        with open(snapshot_dir / "sponsors.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["nct_id", "name", "lead_or_collaborator"])
            writer.writerow(["NCT04470427", "Moderna", "LEAD"])
            writer.writerow(["NCT04860297", "Moderna", "LEAD"])
        
        # Create trial mapping
        self.trial_mapping_file = self.base_path / "trial_mapping.csv"
        with open(self.trial_mapping_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ticker", "nct_id", "effective_start", "effective_end", "source"])
            writer.writerow(["MRNA", "NCT04470427", "2020-07-01", "", "manual"])
            writer.writerow(["MRNA", "NCT04860297", "2021-01-01", "", "manual"])
            # BNTX and NOTRIAL have no mappings
        
        # Output directories
        self.output_dir = self.base_path / "snapshots"
        self.output_dir.mkdir()
    
    def test_snapshot_contains_coverage_blocks(self):
        """Snapshot should include coverage.catalyst and coverage.clinical_dev."""
        config = SnapshotConfig(
            as_of_date=date(2024, 1, 31),
            pit_lag_days=1,
            universe_file=self.universe_file,
            output_dir=self.output_dir / "aact",
            clinical_provider="aact",
            aact_snapshots_dir=self.aact_dir,
            trial_mapping_file=self.trial_mapping_file,
        )
        
        snapshot = generate_snapshot(config)
        
        # Check coverage blocks exist
        self.assertIn("coverage", snapshot.to_dict())
        self.assertIn("catalyst", snapshot.coverage)
        self.assertIn("clinical_dev", snapshot.coverage)
        
        # Check coverage values
        self.assertEqual(snapshot.coverage["catalyst"]["tickers_total"], 3)
        self.assertEqual(snapshot.coverage["catalyst"]["tickers_with_trials"], 1)  # Only MRNA
        self.assertAlmostEqual(
            snapshot.coverage["catalyst"]["coverage_rate"], 
            1/3, 
            places=4
        )
    
    def test_snapshot_contains_provenance(self):
        """Snapshot should include provenance.providers.clinical with snapshot_date_used."""
        config = SnapshotConfig(
            as_of_date=date(2024, 1, 31),
            pit_lag_days=1,
            universe_file=self.universe_file,
            output_dir=self.output_dir / "aact",
            clinical_provider="aact",
            aact_snapshots_dir=self.aact_dir,
            trial_mapping_file=self.trial_mapping_file,
        )
        
        snapshot = generate_snapshot(config)
        
        # Check provenance structure
        self.assertIn("provenance", snapshot.to_dict())
        self.assertIn("providers", snapshot.provenance)
        self.assertIn("clinical", snapshot.provenance["providers"])
        
        clinical_prov = snapshot.provenance["providers"]["clinical"]
        self.assertEqual(clinical_prov["name"], "aact")
        self.assertEqual(clinical_prov["snapshot_date_used"], "2024-01-15")
        self.assertIn("snapshots_root", clinical_prov)
    
    def test_snapshot_contains_input_hashes(self):
        """Snapshot should include input_hashes.trial_facts."""
        config = SnapshotConfig(
            as_of_date=date(2024, 1, 31),
            pit_lag_days=1,
            universe_file=self.universe_file,
            output_dir=self.output_dir / "aact",
            clinical_provider="aact",
            aact_snapshots_dir=self.aact_dir,
            trial_mapping_file=self.trial_mapping_file,
        )
        
        snapshot = generate_snapshot(config)
        
        # Check input_hashes
        self.assertIn("input_hashes", snapshot.to_dict())
        self.assertIn("trial_facts", snapshot.input_hashes)
        self.assertIn("universe", snapshot.input_hashes)
        
        # Hash should be sha256 format
        self.assertTrue(snapshot.input_hashes["trial_facts"].startswith("sha256:"))
    
    def test_snapshot_determinism(self):
        """Same inputs should produce identical snapshot_id."""
        config = SnapshotConfig(
            as_of_date=date(2024, 1, 31),
            pit_lag_days=1,
            universe_file=self.universe_file,
            output_dir=self.output_dir / "aact",
            clinical_provider="aact",
            aact_snapshots_dir=self.aact_dir,
            trial_mapping_file=self.trial_mapping_file,
        )
        
        snapshot1 = generate_snapshot(config)
        snapshot2 = generate_snapshot(config)
        
        # Snapshot IDs should match (ignoring generated_at timestamp)
        self.assertEqual(snapshot1.snapshot_id, snapshot2.snapshot_id)
        self.assertEqual(snapshot1.input_hashes, snapshot2.input_hashes)


class TestCompareSnapshots(TestCase):
    """Test A/B snapshot comparison."""
    
    def setUp(self):
        """Create temp directory with stub and AACT snapshots."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_path = Path(self.temp_dir)
        
        # Create universe
        universe_dir = self.base_path / "universe"
        universe_dir.mkdir()
        self.universe_file = universe_dir / "universe.csv"
        with open(self.universe_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ticker"])
            writer.writerow(["MRNA"])
            writer.writerow(["BNTX"])
        
        # Create AACT data with all required columns
        self.aact_dir = self.base_path / "aact_snapshots"
        self.aact_dir.mkdir()
        
        snapshot_dir = self.aact_dir / "2024-01-15"
        snapshot_dir.mkdir()
        
        with open(snapshot_dir / "studies.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "nct_id", "phase", "overall_status", 
                "primary_completion_date", "primary_completion_date_type",
                "last_update_posted_date", "study_type"
            ])
            writer.writerow([
                "NCT04470427", "Phase 2", "Recruiting", 
                "2024-06-01", "Anticipated", "2024-01-10", "Interventional"
            ])
        
        with open(snapshot_dir / "sponsors.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["nct_id", "name", "lead_or_collaborator"])
            writer.writerow(["NCT04470427", "Moderna", "LEAD"])
        
        # Create trial mapping (only MRNA)
        self.trial_mapping_file = self.base_path / "trial_mapping.csv"
        with open(self.trial_mapping_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ticker", "nct_id", "effective_start", "effective_end", "source"])
            writer.writerow(["MRNA", "NCT04470427", "2020-07-01", "", "manual"])
    
    def test_compare_stub_vs_aact(self):
        """Compare script should handle stub (coverage 0) vs AACT (coverage >0)."""
        from sec_13f.scripts.compare_snapshots import compare_snapshots
        
        # Generate stub snapshot
        stub_config = SnapshotConfig(
            as_of_date=date(2024, 1, 31),
            pit_lag_days=1,
            universe_file=self.universe_file,
            output_dir=self.base_path / "snapshots" / "stub",
            clinical_provider="stub",
        )
        stub_snapshot = generate_snapshot(stub_config)
        stub_file = save_snapshot(stub_snapshot, stub_config.output_dir)
        
        # Generate AACT snapshot
        aact_config = SnapshotConfig(
            as_of_date=date(2024, 1, 31),
            pit_lag_days=1,
            universe_file=self.universe_file,
            output_dir=self.base_path / "snapshots" / "aact",
            clinical_provider="aact",
            aact_snapshots_dir=self.aact_dir,
            trial_mapping_file=self.trial_mapping_file,
        )
        aact_snapshot = generate_snapshot(aact_config)
        aact_file = save_snapshot(aact_snapshot, aact_config.output_dir)
        
        # Compare
        result = compare_snapshots(stub_file, aact_file)
        
        # Stub should have 0 coverage, AACT should have >0
        self.assertEqual(result.baseline_coverage.get("catalyst", 0), 0)
        self.assertGreater(result.compare_coverage.get("catalyst", 0), 0)
        
        # Coverage delta should be positive
        self.assertGreater(result.coverage_delta.get("catalyst", 0), 0)
        
        # MRNA should have gained trials
        self.assertIn("MRNA", result.tickers_gained_trials)
        
        # Total trials: stub=0, aact=1
        self.assertEqual(result.total_trials_baseline, 0)
        self.assertEqual(result.total_trials_compare, 1)
    
    def test_compare_produces_sensible_deltas(self):
        """Comparison should correctly identify trial count changes."""
        from sec_13f.scripts.compare_snapshots import compare_snapshots
        
        # Generate stub snapshot
        stub_config = SnapshotConfig(
            as_of_date=date(2024, 1, 31),
            pit_lag_days=1,
            universe_file=self.universe_file,
            output_dir=self.base_path / "snapshots" / "stub",
            clinical_provider="stub",
        )
        stub_snapshot = generate_snapshot(stub_config)
        stub_file = save_snapshot(stub_snapshot, stub_config.output_dir)
        
        # Generate AACT snapshot
        aact_config = SnapshotConfig(
            as_of_date=date(2024, 1, 31),
            pit_lag_days=1,
            universe_file=self.universe_file,
            output_dir=self.base_path / "snapshots" / "aact",
            clinical_provider="aact",
            aact_snapshots_dir=self.aact_dir,
            trial_mapping_file=self.trial_mapping_file,
        )
        aact_snapshot = generate_snapshot(aact_config)
        aact_file = save_snapshot(aact_snapshot, aact_config.output_dir)
        
        # Compare
        result = compare_snapshots(stub_file, aact_file)
        
        # Check per-ticker changes
        self.assertIn("MRNA", result.trial_count_changes)
        mrna_change = result.trial_count_changes["MRNA"]
        self.assertEqual(mrna_change["baseline"], 0)
        self.assertEqual(mrna_change["compare"], 1)
        self.assertEqual(mrna_change["delta"], 1)
        
        # BNTX should not be in changes (0 -> 0)
        self.assertNotIn("BNTX", result.trial_count_changes)


if __name__ == "__main__":
    import unittest
    unittest.main()
