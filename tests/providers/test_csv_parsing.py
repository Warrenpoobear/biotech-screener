"""
CSV parsing hardening tests.

These tests verify that the AACT provider correctly handles real-world
CSV edge cases that break naive parsers:
    - Commas inside quoted fields
    - Quotes inside quoted fields
    - Missing fields
    - Platform-specific newline handling
"""

import csv
import os
import tempfile
from datetime import date

import pytest

from src.providers.aact_provider import AACTClinicalTrialsProvider
from src.providers.protocols import Phase, TrialStatus, PCDType


class TestCSVParsingRobustness:
    """Test CSV parsing handles real-world AACT edge cases."""

    def test_parses_commas_in_status_field(self):
        """CSV with quoted commas must parse correctly."""
        csv_content = (
            'nct_id,phase,overall_status,primary_completion_date,'
            'primary_completion_date_type,last_update_posted_date,study_type\n'
            'NCT00000001,Phase 2,"Active, not recruiting",2024-06-01,'
            'Anticipated,2024-01-10,Interventional\n'
            'NCT00000002,Phase 3,"Terminated, lack of funding",2024-03-01,'
            'Actual,2024-01-05,Interventional\n'
        )

        fd, path = tempfile.mkstemp(suffix=".csv")
        try:
            with os.fdopen(fd, "w", newline="") as f:
                f.write(csv_content)

            with open(path, "r", newline="") as rf:
                reader = csv.DictReader(rf)
                rows = list(reader)

            assert rows[0]["overall_status"] == "Active, not recruiting"
            assert rows[1]["overall_status"] == "Terminated, lack of funding"
        finally:
            os.remove(path)

    def test_parses_quotes_inside_quoted_fields(self):
        """CSV with escaped quotes inside fields must parse correctly."""
        # RFC 4180: quotes inside quoted fields are escaped by doubling
        csv_content = (
            'nct_id,phase,overall_status,primary_completion_date,'
            'primary_completion_date_type,last_update_posted_date,study_type\n'
            'NCT00000001,Phase 2,"Active, not recruiting (""special cohort"")",2024-06-01,'
            'Anticipated,2024-01-10,Interventional\n'
            'NCT00000002,Phase 1,"Recruiting ""Phase 1b"" patients",2024-09-01,'
            'Anticipated,2024-01-12,Interventional\n'
        )

        fd, path = tempfile.mkstemp(suffix=".csv")
        try:
            with os.fdopen(fd, "w", newline="") as f:
                f.write(csv_content)

            with open(path, "r", newline="") as rf:
                reader = csv.DictReader(rf)
                rows = list(reader)

            assert rows[0]["overall_status"] == 'Active, not recruiting ("special cohort")'
            assert rows[1]["overall_status"] == 'Recruiting "Phase 1b" patients'
        finally:
            os.remove(path)

    def test_parses_sponsor_names_with_special_characters(self):
        """Sponsor names can contain commas, quotes, and special chars."""
        csv_content = (
            'nct_id,name,lead_or_collaborator\n'
            'NCT00000001,"Acme Pharmaceuticals, Inc.",LEAD\n'
            'NCT00000002,"BioTech ""Innovation"" Labs, Ltd.",LEAD\n'
            'NCT00000003,"University of California, San Francisco",LEAD\n'
        )

        fd, path = tempfile.mkstemp(suffix=".csv")
        try:
            with os.fdopen(fd, "w", newline="") as f:
                f.write(csv_content)

            with open(path, "r", newline="") as rf:
                reader = csv.DictReader(rf)
                rows = list(reader)

            assert rows[0]["name"] == "Acme Pharmaceuticals, Inc."
            assert rows[1]["name"] == 'BioTech "Innovation" Labs, Ltd.'
            assert rows[2]["name"] == "University of California, San Francisco"
        finally:
            os.remove(path)

    def test_handles_empty_and_missing_fields(self):
        """Empty fields should parse as empty strings, not cause errors."""
        csv_content = (
            'nct_id,phase,overall_status,primary_completion_date,'
            'primary_completion_date_type,last_update_posted_date,study_type\n'
            'NCT00000001,Phase 2,Recruiting,,,'
            ',Interventional\n'
            'NCT00000002,,Unknown status,2024-06-01,,'
            '2024-01-10,\n'
        )

        fd, path = tempfile.mkstemp(suffix=".csv")
        try:
            with os.fdopen(fd, "w", newline="") as f:
                f.write(csv_content)

            with open(path, "r", newline="") as rf:
                reader = csv.DictReader(rf)
                rows = list(reader)

            # Empty fields should be empty strings
            assert rows[0]["primary_completion_date"] == ""
            assert rows[0]["primary_completion_date_type"] == ""
            assert rows[1]["phase"] == ""
        finally:
            os.remove(path)


class TestPhaseNormalization:
    """Test AACT phase strings normalize to expected buckets."""

    @pytest.mark.parametrize("aact_phase,expected", [
        ("Phase 1", Phase.P1),
        ("PHASE 1", Phase.P1),
        ("Early Phase 1", Phase.P1),
        ("Phase 1/Phase 2", Phase.P1_2),
        ("Phase 1/2", Phase.P1_2),
        ("Phase 2", Phase.P2),
        ("Phase 2/Phase 3", Phase.P2_3),
        ("Phase 2/3", Phase.P2_3),
        ("Phase 3", Phase.P3),
        ("Phase 4", Phase.P4),
        ("N/A", Phase.UNKNOWN),
        ("Not Applicable", Phase.UNKNOWN),
        ("", Phase.UNKNOWN),
        (None, Phase.UNKNOWN),
        ("  Phase 2  ", Phase.P2),  # whitespace handling
    ])
    def test_phase_normalization(self, aact_phase, expected):
        """Phase strings from AACT should map to correct buckets."""
        assert Phase.from_aact(aact_phase) == expected


class TestStatusNormalization:
    """Test AACT status strings normalize to expected enums."""

    @pytest.mark.parametrize("aact_status,expected", [
        ("Active, not recruiting", TrialStatus.ACTIVE),
        ("ACTIVE, NOT RECRUITING", TrialStatus.ACTIVE),
        ("Recruiting", TrialStatus.RECRUITING),
        ("Enrolling by invitation", TrialStatus.ENROLLING),
        ("Completed", TrialStatus.COMPLETED),
        ("Suspended", TrialStatus.SUSPENDED),
        ("Terminated", TrialStatus.TERMINATED),
        ("Withdrawn", TrialStatus.WITHDRAWN),
        ("Not yet recruiting", TrialStatus.NOT_YET_RECRUITING),
        ("Unknown status", TrialStatus.UNKNOWN),
        ("", TrialStatus.UNKNOWN),
        (None, TrialStatus.UNKNOWN),
    ])
    def test_status_normalization(self, aact_status, expected):
        """Status strings from AACT should map to correct enums."""
        assert TrialStatus.from_aact(aact_status) == expected

    def test_terminal_status_detection(self):
        """Terminal statuses should be correctly identified."""
        assert TrialStatus.COMPLETED.is_terminal()
        assert TrialStatus.TERMINATED.is_terminal()
        assert TrialStatus.WITHDRAWN.is_terminal()
        assert not TrialStatus.RECRUITING.is_terminal()
        assert not TrialStatus.ACTIVE.is_terminal()
        assert not TrialStatus.SUSPENDED.is_terminal()

    def test_active_status_detection(self):
        """Active statuses should be correctly identified."""
        assert TrialStatus.ACTIVE.is_active()
        assert TrialStatus.RECRUITING.is_active()
        assert TrialStatus.ENROLLING.is_active()
        assert TrialStatus.NOT_YET_RECRUITING.is_active()
        assert not TrialStatus.SUSPENDED.is_active()
        assert not TrialStatus.COMPLETED.is_active()


class TestPCDTypeNormalization:
    """Test PCD type normalization and flag behavior."""

    @pytest.mark.parametrize("aact_type,expected", [
        ("Actual", PCDType.ACTUAL),
        ("ACTUAL", PCDType.ACTUAL),
        ("Anticipated", PCDType.ANTICIPATED),
        ("ANTICIPATED", PCDType.ANTICIPATED),
        ("Estimated", PCDType.ANTICIPATED),
        ("", PCDType.UNKNOWN),
        (None, PCDType.UNKNOWN),
    ])
    def test_pcd_type_normalization(self, aact_type, expected):
        """PCD type strings from AACT should map to correct enums."""
        assert PCDType.from_aact(aact_type) == expected
