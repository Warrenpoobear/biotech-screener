"""
Tests for Snapshot Utilities

Tests:
1. Quarter date arithmetic
2. Snapshot schema validation
3. Snapshot loading and writing
4. Manifest validation
5. Determinism verification
"""

import json
import tempfile
from datetime import date
from pathlib import Path

import pytest

from src.history.snapshots import (
    # Quarter arithmetic
    get_quarter_end_for_date,
    get_prior_quarter,
    get_quarter_sequence,
    is_valid_quarter_end,
    # Loading
    list_quarters,
    load_snapshot,
    load_manifest,
    # Writing
    write_snapshot,
    write_manifest,
    # Validation
    validate_snapshot_schema,
    validate_manifest_schema,
    # Errors
    SnapshotError,
    SchemaValidationError,
    # Constants
    SNAPSHOT_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
)


# =============================================================================
# QUARTER ARITHMETIC TESTS
# =============================================================================

class TestQuarterArithmetic:
    """Tests for quarter date calculations."""

    def test_get_quarter_end_q1(self):
        """Q1 dates map to March 31."""
        assert get_quarter_end_for_date(date(2025, 1, 15)) == date(2025, 3, 31)
        assert get_quarter_end_for_date(date(2025, 2, 28)) == date(2025, 3, 31)
        assert get_quarter_end_for_date(date(2025, 3, 31)) == date(2025, 3, 31)

    def test_get_quarter_end_q2(self):
        """Q2 dates map to June 30."""
        assert get_quarter_end_for_date(date(2025, 4, 1)) == date(2025, 6, 30)
        assert get_quarter_end_for_date(date(2025, 5, 15)) == date(2025, 6, 30)
        assert get_quarter_end_for_date(date(2025, 6, 30)) == date(2025, 6, 30)

    def test_get_quarter_end_q3(self):
        """Q3 dates map to September 30."""
        assert get_quarter_end_for_date(date(2025, 7, 1)) == date(2025, 9, 30)
        assert get_quarter_end_for_date(date(2025, 8, 15)) == date(2025, 9, 30)
        assert get_quarter_end_for_date(date(2025, 9, 30)) == date(2025, 9, 30)

    def test_get_quarter_end_q4(self):
        """Q4 dates map to December 31."""
        assert get_quarter_end_for_date(date(2025, 10, 1)) == date(2025, 12, 31)
        assert get_quarter_end_for_date(date(2025, 11, 15)) == date(2025, 12, 31)
        assert get_quarter_end_for_date(date(2025, 12, 31)) == date(2025, 12, 31)

    def test_get_quarter_end_string_input(self):
        """Accepts ISO date strings."""
        assert get_quarter_end_for_date("2025-01-15") == date(2025, 3, 31)
        assert get_quarter_end_for_date("2025-07-15") == date(2025, 9, 30)

    def test_get_prior_quarter_q4_to_q3(self):
        """Q4 goes back to Q3."""
        assert get_prior_quarter(date(2025, 12, 31)) == date(2025, 9, 30)

    def test_get_prior_quarter_q3_to_q2(self):
        """Q3 goes back to Q2."""
        assert get_prior_quarter(date(2025, 9, 30)) == date(2025, 6, 30)

    def test_get_prior_quarter_q2_to_q1(self):
        """Q2 goes back to Q1."""
        assert get_prior_quarter(date(2025, 6, 30)) == date(2025, 3, 31)

    def test_get_prior_quarter_q1_to_prior_year_q4(self):
        """Q1 goes back to prior year Q4."""
        assert get_prior_quarter(date(2025, 3, 31)) == date(2024, 12, 31)

    def test_get_prior_quarter_string_input(self):
        """Accepts ISO date strings."""
        assert get_prior_quarter("2025-12-31") == date(2025, 9, 30)
        assert get_prior_quarter("2025-03-31") == date(2024, 12, 31)

    def test_get_prior_quarter_invalid_date(self):
        """Raises error for non-quarter-end dates."""
        with pytest.raises(ValueError, match="Invalid quarter end"):
            get_prior_quarter(date(2025, 1, 15))

    def test_get_quarter_sequence(self):
        """Get sequence of quarter ends going backward."""
        quarters = get_quarter_sequence(date(2025, 12, 31), 4)

        assert len(quarters) == 4
        assert quarters[0] == date(2025, 12, 31)
        assert quarters[1] == date(2025, 9, 30)
        assert quarters[2] == date(2025, 6, 30)
        assert quarters[3] == date(2025, 3, 31)

    def test_get_quarter_sequence_crosses_year(self):
        """Quarter sequence crossing year boundary."""
        quarters = get_quarter_sequence(date(2025, 3, 31), 5)

        assert quarters[0] == date(2025, 3, 31)
        assert quarters[1] == date(2024, 12, 31)
        assert quarters[2] == date(2024, 9, 30)
        assert quarters[3] == date(2024, 6, 30)
        assert quarters[4] == date(2024, 3, 31)

    def test_is_valid_quarter_end_valid(self):
        """Valid quarter ends return True."""
        assert is_valid_quarter_end(date(2025, 3, 31)) is True
        assert is_valid_quarter_end(date(2025, 6, 30)) is True
        assert is_valid_quarter_end(date(2025, 9, 30)) is True
        assert is_valid_quarter_end(date(2025, 12, 31)) is True
        assert is_valid_quarter_end("2025-09-30") is True

    def test_is_valid_quarter_end_invalid(self):
        """Invalid dates return False."""
        assert is_valid_quarter_end(date(2025, 1, 15)) is False
        assert is_valid_quarter_end(date(2025, 3, 30)) is False
        assert is_valid_quarter_end(date(2025, 6, 29)) is False
        assert is_valid_quarter_end("invalid-date") is False


# =============================================================================
# SNAPSHOT SCHEMA VALIDATION TESTS
# =============================================================================

class TestSnapshotSchemaValidation:
    """Tests for snapshot schema validation."""

    def test_valid_snapshot(self):
        """Valid snapshot passes validation."""
        snapshot = {
            "_schema": {
                "version": SNAPSHOT_SCHEMA_VERSION,
                "quarter_end": "2025-09-30",
            },
            "tickers": {
                "AAPL": {
                    "market_cap_usd": 1000000000,
                    "holdings": {
                        "current": {
                            "0001234567": {
                                "quarter_end": "2025-09-30",
                                "state": "KNOWN",
                                "shares": 1000,
                                "value_kusd": 500,
                                "put_call": "",
                            }
                        },
                        "prior": {},
                    },
                }
            },
            "managers": {
                "0001234567": {
                    "name": "Test Manager",
                    "aum_b": 10.0,
                    "style": "concentrated",
                }
            },
            "stats": {
                "tickers_count": 1,
                "managers_count": 1,
            },
        }

        is_valid, errors = validate_snapshot_schema(snapshot)
        assert is_valid is True
        assert errors == []

    def test_missing_schema_section(self):
        """Missing _schema fails validation."""
        snapshot = {
            "tickers": {},
            "managers": {},
            "stats": {"tickers_count": 0, "managers_count": 0},
        }

        is_valid, errors = validate_snapshot_schema(snapshot)
        assert is_valid is False
        assert "Missing '_schema' section" in errors

    def test_wrong_schema_version(self):
        """Wrong schema version fails validation."""
        snapshot = {
            "_schema": {"version": "wrong_version"},
            "tickers": {},
            "managers": {},
            "stats": {"tickers_count": 0, "managers_count": 0},
        }

        is_valid, errors = validate_snapshot_schema(snapshot)
        assert is_valid is False
        assert any("Schema version mismatch" in e for e in errors)

    def test_missing_required_keys(self):
        """Missing required keys fail validation."""
        snapshot = {
            "_schema": {"version": SNAPSHOT_SCHEMA_VERSION, "quarter_end": "2025-09-30"},
            # Missing: tickers, managers, stats
        }

        is_valid, errors = validate_snapshot_schema(snapshot)
        assert is_valid is False
        assert any("tickers" in e for e in errors)
        assert any("managers" in e for e in errors)
        assert any("stats" in e for e in errors)


# =============================================================================
# MANIFEST SCHEMA VALIDATION TESTS
# =============================================================================

class TestManifestSchemaValidation:
    """Tests for manifest schema validation."""

    def test_valid_manifest(self):
        """Valid manifest passes validation."""
        manifest = {
            "_schema": {"version": MANIFEST_SCHEMA_VERSION},
            "run_id": "abc123def456",
            "params": {
                "quarter_end": "2025-12-31",
                "quarters": 12,
            },
            "quarters": [
                {
                    "quarter_end": "2025-12-31",
                    "filename": "holdings_2025-12-31.json",
                    "sha256": "abc123...",
                }
            ],
            "input_hashes": [
                {"path": "manager_registry.json", "sha256": "..."},
            ],
        }

        is_valid, errors = validate_manifest_schema(manifest)
        assert is_valid is True
        assert errors == []

    def test_missing_manifest_keys(self):
        """Missing required manifest keys fail validation."""
        manifest = {
            "_schema": {"version": MANIFEST_SCHEMA_VERSION},
            # Missing: run_id, params, quarters, input_hashes
        }

        is_valid, errors = validate_manifest_schema(manifest)
        assert is_valid is False
        assert any("run_id" in e for e in errors)


# =============================================================================
# SNAPSHOT READ/WRITE TESTS
# =============================================================================

class TestSnapshotReadWrite:
    """Tests for snapshot read/write operations."""

    def test_write_and_read_snapshot(self, tmp_path):
        """Snapshot round-trips correctly."""
        snapshot = {
            "_schema": {
                "version": SNAPSHOT_SCHEMA_VERSION,
                "quarter_end": "2025-09-30",
            },
            "tickers": {"TEST": {"market_cap_usd": 1000}},
            "managers": {},
            "stats": {"tickers_count": 1, "managers_count": 0},
        }

        quarter_end = date(2025, 9, 30)

        # Write
        filepath, file_hash = write_snapshot(snapshot, quarter_end, tmp_path)

        assert filepath.exists()
        assert filepath.name == "holdings_2025-09-30.json"
        assert len(file_hash) == 64

        # Read
        loaded = load_snapshot(quarter_end, tmp_path)

        assert loaded["_schema"]["version"] == SNAPSHOT_SCHEMA_VERSION
        assert loaded["tickers"]["TEST"]["market_cap_usd"] == 1000

    def test_write_snapshot_canonical_format(self, tmp_path):
        """Snapshots are written in canonical JSON format."""
        # Dict with keys in non-alphabetical order
        snapshot = {
            "_schema": {"version": SNAPSHOT_SCHEMA_VERSION, "quarter_end": "2025-09-30"},
            "z_key": "last",
            "a_key": "first",
            "tickers": {},
            "managers": {},
            "stats": {"tickers_count": 0, "managers_count": 0},
        }

        filepath, _ = write_snapshot(snapshot, date(2025, 9, 30), tmp_path)

        content = filepath.read_text()

        # Keys should be sorted (a_key before z_key)
        assert content.index('"a_key"') < content.index('"z_key"')

        # Trailing newline
        assert content.endswith('\n')

    def test_write_snapshot_deterministic(self, tmp_path):
        """Same snapshot produces identical output."""
        snapshot = {
            "_schema": {"version": SNAPSHOT_SCHEMA_VERSION, "quarter_end": "2025-09-30"},
            "tickers": {"B": {"val": 2}, "A": {"val": 1}},
            "managers": {},
            "stats": {"tickers_count": 2, "managers_count": 0},
        }

        # Write twice to different dirs
        dir1 = tmp_path / "run1"
        dir2 = tmp_path / "run2"
        dir1.mkdir()
        dir2.mkdir()

        _, hash1 = write_snapshot(snapshot, date(2025, 9, 30), dir1)
        _, hash2 = write_snapshot(snapshot, date(2025, 9, 30), dir2)

        assert hash1 == hash2

    def test_load_snapshot_not_found(self, tmp_path):
        """Loading missing snapshot raises SnapshotError."""
        with pytest.raises(SnapshotError, match="not found"):
            load_snapshot(date(2025, 9, 30), tmp_path)

    def test_list_quarters(self, tmp_path):
        """list_quarters returns sorted quarter dates."""
        # Create some snapshot files
        quarters = [
            date(2024, 9, 30),
            date(2024, 12, 31),
            date(2025, 3, 31),
        ]

        for q in quarters:
            filepath = tmp_path / f"holdings_{q.isoformat()}.json"
            filepath.write_text('{}')

        # Also add a non-snapshot file
        (tmp_path / "manifest.json").write_text('{}')
        (tmp_path / "other.txt").write_text('')

        result = list_quarters(tmp_path)

        assert len(result) == 3
        # Newest first
        assert result[0] == date(2025, 3, 31)
        assert result[1] == date(2024, 12, 31)
        assert result[2] == date(2024, 9, 30)


# =============================================================================
# MANIFEST READ/WRITE TESTS
# =============================================================================

class TestManifestReadWrite:
    """Tests for manifest read/write operations."""

    def test_write_and_read_manifest(self, tmp_path):
        """Manifest round-trips correctly."""
        manifest = {
            "_schema": {"version": MANIFEST_SCHEMA_VERSION},
            "run_id": "test123",
            "params": {"quarters": 4},
            "quarters": [],
            "input_hashes": [],
        }

        filepath, file_hash = write_manifest(manifest, tmp_path)

        assert filepath.exists()
        assert filepath.name == "manifest.json"

        loaded = load_manifest(tmp_path)
        assert loaded["run_id"] == "test123"

    def test_manifest_deterministic(self, tmp_path):
        """Same manifest produces identical output."""
        manifest = {
            "_schema": {"version": MANIFEST_SCHEMA_VERSION},
            "run_id": "abc",
            "params": {"b": 2, "a": 1},
            "quarters": [],
            "input_hashes": [],
        }

        dir1 = tmp_path / "run1"
        dir2 = tmp_path / "run2"
        dir1.mkdir()
        dir2.mkdir()

        _, hash1 = write_manifest(manifest, dir1)
        _, hash2 = write_manifest(manifest, dir2)

        assert hash1 == hash2


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestSnapshotIntegration:
    """Integration tests for snapshot system."""

    def test_full_history_workflow(self, tmp_path):
        """Full workflow: write multiple quarters, list, load."""
        quarters_data = [
            (date(2025, 9, 30), {"tickers": {"A": {"v": 1}}}),
            (date(2025, 6, 30), {"tickers": {"B": {"v": 2}}}),
            (date(2025, 3, 31), {"tickers": {"C": {"v": 3}}}),
        ]

        # Write snapshots
        hashes = []
        for q_end, data in quarters_data:
            snapshot = {
                "_schema": {"version": SNAPSHOT_SCHEMA_VERSION, "quarter_end": q_end.isoformat()},
                "tickers": data["tickers"],
                "managers": {},
                "stats": {"tickers_count": 1, "managers_count": 0},
            }
            _, h = write_snapshot(snapshot, q_end, tmp_path)
            hashes.append(h)

        # Write manifest
        manifest = {
            "_schema": {"version": MANIFEST_SCHEMA_VERSION},
            "run_id": "integration_test",
            "params": {"quarters": 3},
            "quarters": [
                {"quarter_end": q_end.isoformat(), "sha256": h}
                for (q_end, _), h in zip(quarters_data, hashes)
            ],
            "input_hashes": [],
        }
        write_manifest(manifest, tmp_path)

        # List quarters
        available = list_quarters(tmp_path)
        assert len(available) == 3
        assert available[0] == date(2025, 9, 30)

        # Load each quarter
        for q_end, expected_data in quarters_data:
            loaded = load_snapshot(q_end, tmp_path)
            assert loaded["tickers"] == expected_data["tickers"]

        # Load manifest
        loaded_manifest = load_manifest(tmp_path)
        assert loaded_manifest["run_id"] == "integration_test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
