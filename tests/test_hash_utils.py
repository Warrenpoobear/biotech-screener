#!/usr/bin/env python3
"""
Tests for Hash Utilities

Covers:
- Stable JSON serialization
- SHA256 hash computation
- Trial facts hashing
- Snapshot ID computation
- Determinism guarantees
"""

import pytest
from datetime import date
from decimal import Decimal
from pathlib import Path
from enum import Enum
from typing import Dict, Any

# Import module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.hash_utils import (
    stable_json_dumps,
    compute_hash,
    compute_trial_facts_hash,
    compute_snapshot_id,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_dict():
    """Simple dictionary for testing."""
    return {"b": 2, "a": 1, "c": 3}


@pytest.fixture
def nested_dict():
    """Nested dictionary for testing."""
    return {
        "outer": {
            "z": 26,
            "a": 1,
        },
        "list": [3, 1, 2],
        "value": "test",
    }


@pytest.fixture
def dict_with_special_types():
    """Dictionary with special types (date, Decimal, Enum)."""
    class TestEnum(Enum):
        VALUE = "test_value"

    return {
        "date": date(2026, 1, 15),
        "decimal": Decimal("123.45"),
        "enum": TestEnum.VALUE,
        "string": "normal",
        "int": 42,
    }


@pytest.fixture
def sample_trials_by_ticker():
    """Sample trial data organized by ticker."""
    return {
        "ACME": [
            {"nct_id": "NCT00000002", "phase": "Phase 3", "status": "recruiting"},
            {"nct_id": "NCT00000001", "phase": "Phase 2", "status": "completed"},
        ],
        "BETA": [
            {"nct_id": "NCT00000003", "phase": "Phase 1", "status": "recruiting"},
        ],
    }


# ============================================================================
# STABLE JSON DUMPS
# ============================================================================

class TestStableJsonDumps:
    """Tests for stable_json_dumps function."""

    def test_sorts_dict_keys(self, sample_dict):
        """Dictionary keys are sorted."""
        result = stable_json_dumps(sample_dict)

        # Keys should appear in sorted order
        assert result.index('"a"') < result.index('"b"')
        assert result.index('"b"') < result.index('"c"')

    def test_nested_dict_keys_sorted(self, nested_dict):
        """Nested dictionary keys are also sorted."""
        result = stable_json_dumps(nested_dict)

        # Outer keys sorted
        assert result.index('"list"') < result.index('"outer"')

        # Inner keys sorted
        assert result.index('"a"') < result.index('"z"')

    def test_serializes_date(self):
        """Dates are serialized to ISO format."""
        data = {"date": date(2026, 1, 15)}
        result = stable_json_dumps(data)

        assert '"2026-01-15"' in result

    def test_serializes_decimal(self):
        """Decimals are serialized as strings."""
        data = {"amount": Decimal("123.456789")}
        result = stable_json_dumps(data)

        assert '"123.456789"' in result

    def test_serializes_enum(self):
        """Enums are serialized by value."""
        class Color(Enum):
            RED = "red"
            BLUE = "blue"

        data = {"color": Color.RED}
        result = stable_json_dumps(data)

        assert '"red"' in result

    def test_serializes_basic_types(self):
        """Basic types are serialized correctly."""
        data = {
            "string": "hello",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "null": None,
            "list": [1, 2, 3],
        }
        result = stable_json_dumps(data)

        assert '"hello"' in result
        assert "42" in result
        assert "3.14" in result
        assert "true" in result
        assert "null" in result

    def test_raises_for_unsupported_types(self):
        """Raises TypeError for unsupported types."""
        class CustomClass:
            pass

        data = {"custom": CustomClass()}

        with pytest.raises(TypeError, match="not JSON serializable"):
            stable_json_dumps(data)

    def test_deterministic_output(self, sample_dict):
        """Same input always produces same output."""
        result1 = stable_json_dumps(sample_dict)
        result2 = stable_json_dumps(sample_dict)

        assert result1 == result2

    def test_empty_dict(self):
        """Empty dict produces valid JSON."""
        result = stable_json_dumps({})
        assert result == "{}"

    def test_empty_list(self):
        """Empty list produces valid JSON."""
        result = stable_json_dumps([])
        assert result == "[]"


# ============================================================================
# COMPUTE HASH
# ============================================================================

class TestComputeHash:
    """Tests for compute_hash function."""

    def test_returns_sha256_format(self, sample_dict):
        """Hash is returned in 'sha256:...' format."""
        result = compute_hash(sample_dict)

        assert result.startswith("sha256:")
        # SHA256 hex is 64 characters
        assert len(result) == len("sha256:") + 64

    def test_hash_is_deterministic(self, sample_dict):
        """Same input produces same hash."""
        hash1 = compute_hash(sample_dict)
        hash2 = compute_hash(sample_dict)

        assert hash1 == hash2

    def test_different_data_different_hash(self):
        """Different data produces different hashes."""
        hash1 = compute_hash({"a": 1})
        hash2 = compute_hash({"a": 2})

        assert hash1 != hash2

    def test_key_order_independent(self):
        """Hash is independent of key order."""
        hash1 = compute_hash({"a": 1, "b": 2})
        hash2 = compute_hash({"b": 2, "a": 1})

        assert hash1 == hash2

    def test_hash_string_directly(self):
        """Can hash a string directly."""
        result = compute_hash("test string")

        assert result.startswith("sha256:")

    def test_hash_special_types(self, dict_with_special_types):
        """Can hash dictionary with special types."""
        result = compute_hash(dict_with_special_types)

        assert result.startswith("sha256:")

    def test_hash_empty_dict(self):
        """Hash of empty dict is deterministic."""
        hash1 = compute_hash({})
        hash2 = compute_hash({})

        assert hash1 == hash2

    def test_hash_nested_structure(self, nested_dict):
        """Hash works with nested structures."""
        result = compute_hash(nested_dict)

        assert result.startswith("sha256:")


# ============================================================================
# TRIAL FACTS HASH
# ============================================================================

class TestTrialFactsHash:
    """Tests for compute_trial_facts_hash function."""

    def test_basic_trial_hash(self, sample_trials_by_ticker):
        """Compute hash of trial facts."""
        result = compute_trial_facts_hash(sample_trials_by_ticker)

        assert result.startswith("sha256:")

    def test_trial_hash_deterministic(self, sample_trials_by_ticker):
        """Same trials produce same hash."""
        hash1 = compute_trial_facts_hash(sample_trials_by_ticker)
        hash2 = compute_trial_facts_hash(sample_trials_by_ticker)

        assert hash1 == hash2

    def test_trial_hash_sorts_by_ticker(self):
        """Ticker order doesn't affect hash."""
        trials1 = {
            "ACME": [{"nct_id": "NCT001"}],
            "BETA": [{"nct_id": "NCT002"}],
        }
        trials2 = {
            "BETA": [{"nct_id": "NCT002"}],
            "ACME": [{"nct_id": "NCT001"}],
        }

        hash1 = compute_trial_facts_hash(trials1)
        hash2 = compute_trial_facts_hash(trials2)

        assert hash1 == hash2

    def test_trial_hash_sorts_by_nct_id(self):
        """Trial order within ticker doesn't affect hash."""
        trials1 = {
            "ACME": [
                {"nct_id": "NCT002"},
                {"nct_id": "NCT001"},
            ],
        }
        trials2 = {
            "ACME": [
                {"nct_id": "NCT001"},
                {"nct_id": "NCT002"},
            ],
        }

        hash1 = compute_trial_facts_hash(trials1)
        hash2 = compute_trial_facts_hash(trials2)

        assert hash1 == hash2

    def test_trial_hash_empty(self):
        """Hash of empty trials is deterministic."""
        hash1 = compute_trial_facts_hash({})
        hash2 = compute_trial_facts_hash({})

        assert hash1 == hash2

    def test_trial_hash_different_data(self, sample_trials_by_ticker):
        """Different trials produce different hash."""
        hash1 = compute_trial_facts_hash(sample_trials_by_ticker)

        modified = sample_trials_by_ticker.copy()
        modified["ACME"][0]["status"] = "completed"

        hash2 = compute_trial_facts_hash(modified)

        assert hash1 != hash2

    def test_trial_hash_with_to_dict_objects(self):
        """Hash works with objects that have to_dict method."""
        class MockTrial:
            def __init__(self, nct_id: str, phase: str):
                self.nct_id = nct_id
                self.phase = phase

            def to_dict(self) -> Dict[str, Any]:
                return {"nct_id": self.nct_id, "phase": self.phase}

        trials = {
            "ACME": [
                MockTrial("NCT001", "Phase 1"),
                MockTrial("NCT002", "Phase 2"),
            ],
        }

        result = compute_trial_facts_hash(trials)

        assert result.startswith("sha256:")


# ============================================================================
# SNAPSHOT ID
# ============================================================================

class TestSnapshotId:
    """Tests for compute_snapshot_id function."""

    def test_basic_snapshot_id(self):
        """Compute snapshot ID from inputs."""
        result = compute_snapshot_id(
            as_of_date=date(2026, 1, 15),
            pit_cutoff=date(2026, 1, 14),
            input_hashes={"universe": "sha256:abc123"},
            provider_metadata={"provider": "test"},
        )

        assert result.startswith("sha256:")

    def test_snapshot_id_deterministic(self):
        """Same inputs produce same snapshot ID."""
        args = {
            "as_of_date": date(2026, 1, 15),
            "pit_cutoff": date(2026, 1, 14),
            "input_hashes": {"universe": "sha256:abc123"},
            "provider_metadata": {"provider": "test"},
        }

        id1 = compute_snapshot_id(**args)
        id2 = compute_snapshot_id(**args)

        assert id1 == id2

    def test_snapshot_id_changes_with_date(self):
        """Different as_of_date produces different ID."""
        base_args = {
            "pit_cutoff": date(2026, 1, 14),
            "input_hashes": {"universe": "sha256:abc123"},
            "provider_metadata": {"provider": "test"},
        }

        id1 = compute_snapshot_id(as_of_date=date(2026, 1, 15), **base_args)
        id2 = compute_snapshot_id(as_of_date=date(2026, 1, 16), **base_args)

        assert id1 != id2

    def test_snapshot_id_changes_with_input_hashes(self):
        """Different input hashes produce different ID."""
        base_args = {
            "as_of_date": date(2026, 1, 15),
            "pit_cutoff": date(2026, 1, 14),
            "provider_metadata": {"provider": "test"},
        }

        id1 = compute_snapshot_id(input_hashes={"universe": "sha256:abc"}, **base_args)
        id2 = compute_snapshot_id(input_hashes={"universe": "sha256:xyz"}, **base_args)

        assert id1 != id2

    def test_snapshot_id_changes_with_provider_metadata(self):
        """Different provider metadata produces different ID."""
        base_args = {
            "as_of_date": date(2026, 1, 15),
            "pit_cutoff": date(2026, 1, 14),
            "input_hashes": {"universe": "sha256:abc123"},
        }

        id1 = compute_snapshot_id(provider_metadata={"version": "1.0"}, **base_args)
        id2 = compute_snapshot_id(provider_metadata={"version": "2.0"}, **base_args)

        assert id1 != id2


# ============================================================================
# DETERMINISM
# ============================================================================

class TestDeterminism:
    """Cross-cutting determinism tests."""

    def test_hash_same_data_multiple_calls(self):
        """Multiple hash calls on same data are identical."""
        data = {
            "nested": {"z": 1, "a": 2},
            "list": [1, 2, 3],
            "date": date(2026, 1, 15),
            "decimal": Decimal("99.99"),
        }

        hashes = [compute_hash(data) for _ in range(10)]

        assert len(set(hashes)) == 1

    def test_json_dumps_same_data_multiple_calls(self):
        """Multiple stable_json_dumps calls produce identical output."""
        data = {"b": 2, "a": 1}

        outputs = [stable_json_dumps(data) for _ in range(10)]

        assert len(set(outputs)) == 1

    def test_trial_hash_multiple_calls(self, sample_trials_by_ticker):
        """Multiple trial hash calls are identical."""
        hashes = [
            compute_trial_facts_hash(sample_trials_by_ticker)
            for _ in range(10)
        ]

        assert len(set(hashes)) == 1


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_hash_deeply_nested_structure(self):
        """Hash works with deeply nested structures."""
        data = {"level1": {"level2": {"level3": {"level4": "value"}}}}
        result = compute_hash(data)

        assert result.startswith("sha256:")

    def test_hash_large_list(self):
        """Hash works with large lists."""
        data = {"values": list(range(10000))}
        result = compute_hash(data)

        assert result.startswith("sha256:")

    def test_hash_unicode_strings(self):
        """Hash works with unicode strings."""
        data = {
            "japanese": "日本語",
            "emoji": "🧬",
            "french": "café",
        }
        result = compute_hash(data)

        assert result.startswith("sha256:")

    def test_hash_special_decimal_values(self):
        """Hash works with special Decimal values."""
        data = {
            "zero": Decimal("0"),
            "negative": Decimal("-123.45"),
            "large": Decimal("9999999999999.99"),
            "tiny": Decimal("0.000000001"),
        }
        result = compute_hash(data)

        assert result.startswith("sha256:")

    def test_stable_json_preserves_list_order(self):
        """stable_json_dumps preserves list order."""
        data = {"list": [3, 1, 2]}
        result = stable_json_dumps(data)

        # List order should be preserved
        assert "[3, 1, 2]" in result or "[3,1,2]" in result.replace(" ", "")
