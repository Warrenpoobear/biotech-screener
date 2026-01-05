"""
Tests for AACT provider schema validation and data integrity.
"""

import csv
import os
import tempfile
from datetime import date
from pathlib import Path

import pytest

from src.providers.aact_provider import AACTClinicalTrialsProvider


class TestSchemaValidation:
    """Test that missing required columns fail loudly."""
    
    def test_missing_studies_column_raises_error(self):
        """Missing required column in studies.csv should raise ValueError."""
        temp_dir = tempfile.mkdtemp()
        snapshots_root = Path(temp_dir) / "aact_snapshots"
        snapshots_root.mkdir()
        
        snapshot_dir = snapshots_root / "2024-01-15"
        snapshot_dir.mkdir()
        
        # Create studies.csv missing 'last_update_posted_date'
        with open(snapshot_dir / "studies.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["nct_id", "phase", "overall_status", 
                            "primary_completion_date", "primary_completion_date_type",
                            "study_type"])  # Missing last_update_posted_date!
            writer.writerow(["NCT00000001", "Phase 2", "Recruiting", 
                            "2024-06-01", "Anticipated", "Interventional"])
        
        # Create valid sponsors.csv
        with open(snapshot_dir / "sponsors.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["nct_id", "name", "lead_or_collaborator"])
            writer.writerow(["NCT00000001", "Test Sponsor", "LEAD"])
        
        provider = AACTClinicalTrialsProvider(snapshots_root=snapshots_root)
        
        with pytest.raises(ValueError) as exc_info:
            provider.load_snapshot(date(2024, 1, 15))
        
        assert "last_update_posted_date" in str(exc_info.value)
        assert "Missing required columns" in str(exc_info.value)
    
    def test_missing_sponsors_column_raises_error(self):
        """Missing required column in sponsors.csv should raise ValueError."""
        temp_dir = tempfile.mkdtemp()
        snapshots_root = Path(temp_dir) / "aact_snapshots"
        snapshots_root.mkdir()
        
        snapshot_dir = snapshots_root / "2024-01-15"
        snapshot_dir.mkdir()
        
        # Create valid studies.csv
        with open(snapshot_dir / "studies.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["nct_id", "phase", "overall_status", 
                            "primary_completion_date", "primary_completion_date_type",
                            "last_update_posted_date", "study_type"])
            writer.writerow(["NCT00000001", "Phase 2", "Recruiting", 
                            "2024-06-01", "Anticipated", "2024-01-10", "Interventional"])
        
        # Create sponsors.csv missing 'lead_or_collaborator'
        with open(snapshot_dir / "sponsors.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["nct_id", "name"])  # Missing lead_or_collaborator!
            writer.writerow(["NCT00000001", "Test Sponsor"])
        
        provider = AACTClinicalTrialsProvider(snapshots_root=snapshots_root)
        
        with pytest.raises(ValueError) as exc_info:
            provider.load_snapshot(date(2024, 1, 15))
        
        assert "lead_or_collaborator" in str(exc_info.value)


class TestDuplicateKeyHandling:
    """Test that duplicate NCT IDs are handled deterministically."""
    
    def test_duplicate_nct_id_keeps_latest_update(self):
        """Duplicate NCT IDs should keep row with latest last_update_posted_date."""
        temp_dir = tempfile.mkdtemp()
        snapshots_root = Path(temp_dir) / "aact_snapshots"
        snapshots_root.mkdir()
        
        snapshot_dir = snapshots_root / "2024-01-15"
        snapshot_dir.mkdir()
        
        # Create studies.csv with duplicate NCT ID
        with open(snapshot_dir / "studies.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["nct_id", "phase", "overall_status", 
                            "primary_completion_date", "primary_completion_date_type",
                            "last_update_posted_date", "study_type"])
            # First row with older date
            writer.writerow(["NCT00000001", "Phase 1", "Recruiting", 
                            "2024-06-01", "Anticipated", "2024-01-05", "Interventional"])
            # Second row with newer date - should be kept
            writer.writerow(["NCT00000001", "Phase 2", "Recruiting", 
                            "2024-06-01", "Anticipated", "2024-01-10", "Interventional"])
        
        # Create valid sponsors.csv
        with open(snapshot_dir / "sponsors.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["nct_id", "name", "lead_or_collaborator"])
            writer.writerow(["NCT00000001", "Test Sponsor", "LEAD"])
        
        provider = AACTClinicalTrialsProvider(snapshots_root=snapshots_root)
        snapshot = provider.load_snapshot(date(2024, 1, 15))
        
        # Should have only one study
        assert len(snapshot.studies) == 1
        # Should be the one with Phase 2 (newer last_update_posted_date)
        assert snapshot.studies["NCT00000001"]["phase"] == "Phase 2"
        assert snapshot.studies["NCT00000001"]["last_update_posted_date"] == "2024-01-10"


class TestFlagBehavior:
    """Test that flags are emitted correctly for missing data."""
    
    def test_missing_pcd_emits_flag(self):
        """Missing primary_completion_date should emit pcd_missing flag."""
        temp_dir = tempfile.mkdtemp()
        snapshots_root = Path(temp_dir) / "aact_snapshots"
        snapshots_root.mkdir()
        
        snapshot_dir = snapshots_root / "2024-01-15"
        snapshot_dir.mkdir()
        
        with open(snapshot_dir / "studies.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["nct_id", "phase", "overall_status", 
                            "primary_completion_date", "primary_completion_date_type",
                            "last_update_posted_date", "study_type"])
            # Row with missing PCD
            writer.writerow(["NCT00000001", "Phase 2", "Recruiting", 
                            "", "", "2024-01-10", "Interventional"])
        
        with open(snapshot_dir / "sponsors.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["nct_id", "name", "lead_or_collaborator"])
            writer.writerow(["NCT00000001", "Test Sponsor", "LEAD"])
        
        provider = AACTClinicalTrialsProvider(
            snapshots_root=snapshots_root,
            compute_diffs=False
        )
        result = provider.get_trials_as_of(
            as_of_date=date(2024, 1, 31),
            pit_cutoff=date(2024, 1, 30),
            tickers=["TEST"],
            trial_mapping={"TEST": ["NCT00000001"]}
        )
        
        trial = result.trials_by_ticker["TEST"][0]
        assert "pcd_missing" in trial.flags
        assert "pcd_type_missing" in trial.flags
    
    def test_diffs_disabled_emits_flag(self):
        """When compute_diffs=False, trials should have diffs_disabled flag."""
        temp_dir = tempfile.mkdtemp()
        snapshots_root = Path(temp_dir) / "aact_snapshots"
        snapshots_root.mkdir()
        
        snapshot_dir = snapshots_root / "2024-01-15"
        snapshot_dir.mkdir()
        
        with open(snapshot_dir / "studies.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["nct_id", "phase", "overall_status", 
                            "primary_completion_date", "primary_completion_date_type",
                            "last_update_posted_date", "study_type"])
            writer.writerow(["NCT00000001", "Phase 2", "Recruiting", 
                            "2024-06-01", "Anticipated", "2024-01-10", "Interventional"])
        
        with open(snapshot_dir / "sponsors.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["nct_id", "name", "lead_or_collaborator"])
            writer.writerow(["NCT00000001", "Test Sponsor", "LEAD"])
        
        provider = AACTClinicalTrialsProvider(
            snapshots_root=snapshots_root,
            compute_diffs=False  # Explicitly disabled
        )
        result = provider.get_trials_as_of(
            as_of_date=date(2024, 1, 31),
            pit_cutoff=date(2024, 1, 30),
            tickers=["TEST"],
            trial_mapping={"TEST": ["NCT00000001"]}
        )
        
        trial = result.trials_by_ticker["TEST"][0]
        assert "diffs_disabled" in trial.flags
        assert result.compute_diffs_enabled is False
        assert result.compute_diffs_available is False
    
    def test_diffs_unavailable_insufficient_snapshots_emits_flag(self):
        """When compute_diffs=True but only 1 snapshot, emit insufficient_snapshots flag."""
        temp_dir = tempfile.mkdtemp()
        snapshots_root = Path(temp_dir) / "aact_snapshots"
        snapshots_root.mkdir()
        
        # Only one snapshot
        snapshot_dir = snapshots_root / "2024-01-15"
        snapshot_dir.mkdir()
        
        with open(snapshot_dir / "studies.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["nct_id", "phase", "overall_status", 
                            "primary_completion_date", "primary_completion_date_type",
                            "last_update_posted_date", "study_type"])
            writer.writerow(["NCT00000001", "Phase 2", "Recruiting", 
                            "2024-06-01", "Anticipated", "2024-01-10", "Interventional"])
        
        with open(snapshot_dir / "sponsors.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["nct_id", "name", "lead_or_collaborator"])
            writer.writerow(["NCT00000001", "Test Sponsor", "LEAD"])
        
        provider = AACTClinicalTrialsProvider(
            snapshots_root=snapshots_root,
            compute_diffs=True  # Enabled but insufficient snapshots
        )
        result = provider.get_trials_as_of(
            as_of_date=date(2024, 1, 31),
            pit_cutoff=date(2024, 1, 30),
            tickers=["TEST"],
            trial_mapping={"TEST": ["NCT00000001"]}
        )
        
        trial = result.trials_by_ticker["TEST"][0]
        assert "diffs_unavailable_insufficient_snapshots" in trial.flags
        assert result.compute_diffs_enabled is True
        assert result.compute_diffs_available is False
        assert result.snapshots_available_count == 1
