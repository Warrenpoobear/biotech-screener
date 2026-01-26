#!/usr/bin/env python3
"""
Tests for common/run_manifest.py

Run manifest logging for backtest reproducibility.
Tests cover:
- compute_content_hash (deterministic SHA256 hashing)
- canonicalize_for_hash (removes non-deterministic fields)
- compute_results_hash (hash excluding timestamps, etc.)
- ManifestEntry (entry serialization/deserialization)
- RunManifest (JSONL manifest logging and querying)
- log_backtest_run (convenience function)
"""

import json
import pytest
import tempfile
from decimal import Decimal
from pathlib import Path

from common.run_manifest import (
    # Hashing functions
    compute_content_hash,
    canonicalize_for_hash,
    compute_results_hash,
    RESULTS_HASH_EXCLUDE_KEYS,
    # Classes
    ManifestEntry,
    RunManifest,
    DecimalEncoder,
    # Helpers
    create_data_hashes,
    log_backtest_run,
    # Constants
    MANIFEST_VERSION,
)


class TestComputeContentHash:
    """Tests for compute_content_hash function."""

    def test_returns_sha256_prefix(self):
        """Hash should start with sha256: prefix."""
        result = compute_content_hash({"key": "value"})
        assert result.startswith("sha256:")

    def test_returns_64_char_hex(self):
        """Hash should be 64 hex characters after prefix."""
        result = compute_content_hash({"key": "value"})
        hex_part = result.replace("sha256:", "")
        assert len(hex_part) == 64
        assert all(c in "0123456789abcdef" for c in hex_part)

    def test_deterministic(self):
        """Same input should produce same hash."""
        data = {"a": 1, "b": [2, 3], "c": {"nested": True}}
        hash1 = compute_content_hash(data)
        hash2 = compute_content_hash(data)
        assert hash1 == hash2

    def test_key_order_independent(self):
        """Hash should be same regardless of key order."""
        data1 = {"a": 1, "b": 2, "c": 3}
        data2 = {"c": 3, "a": 1, "b": 2}
        assert compute_content_hash(data1) == compute_content_hash(data2)

    def test_different_data_different_hash(self):
        """Different data should produce different hash."""
        hash1 = compute_content_hash({"a": 1})
        hash2 = compute_content_hash({"a": 2})
        assert hash1 != hash2

    def test_handles_nested_structures(self):
        """Should handle nested dicts and lists."""
        data = {
            "level1": {
                "level2": {
                    "list": [1, 2, {"deep": True}]
                }
            }
        }
        result = compute_content_hash(data)
        assert result.startswith("sha256:")

    def test_handles_decimals(self):
        """Should handle Decimal values."""
        data = {"score": Decimal("75.50")}
        result = compute_content_hash(data)
        assert result.startswith("sha256:")

    def test_handles_primitives(self):
        """Should handle primitive types."""
        assert compute_content_hash("string").startswith("sha256:")
        assert compute_content_hash(123).startswith("sha256:")
        assert compute_content_hash(45.67).startswith("sha256:")
        assert compute_content_hash(True).startswith("sha256:")
        assert compute_content_hash(None).startswith("sha256:")

    def test_handles_empty_structures(self):
        """Should handle empty dicts and lists."""
        assert compute_content_hash({}).startswith("sha256:")
        assert compute_content_hash([]).startswith("sha256:")


class TestCanonicalizeForHash:
    """Tests for canonicalize_for_hash function."""

    def test_removes_timestamp(self):
        """Should remove timestamp key."""
        data = {"timestamp": "2024-01-01", "score": 75}
        result = canonicalize_for_hash(data)
        assert "timestamp" not in result
        assert result["score"] == 75

    def test_removes_run_id(self):
        """Should remove run_id key."""
        data = {"run_id": "abc123", "score": 75}
        result = canonicalize_for_hash(data)
        assert "run_id" not in result

    def test_removes_all_exclude_keys(self):
        """Should remove all keys in RESULTS_HASH_EXCLUDE_KEYS."""
        data = {key: "value" for key in RESULTS_HASH_EXCLUDE_KEYS}
        data["keep_me"] = "preserved"
        result = canonicalize_for_hash(data)

        for key in RESULTS_HASH_EXCLUDE_KEYS:
            assert key not in result
        assert result["keep_me"] == "preserved"

    def test_recursive_removal(self):
        """Should remove excluded keys from nested structures."""
        data = {
            "outer": {
                "timestamp": "2024-01-01",
                "inner_score": 75,
            },
            "timestamp": "2024-01-02",
        }
        result = canonicalize_for_hash(data)

        assert "timestamp" not in result
        assert "timestamp" not in result["outer"]
        assert result["outer"]["inner_score"] == 75

    def test_handles_lists(self):
        """Should process items in lists."""
        data = [
            {"timestamp": "2024-01-01", "score": 75},
            {"timestamp": "2024-01-02", "score": 80},
        ]
        result = canonicalize_for_hash(data)

        assert isinstance(result, list)
        assert len(result) == 2
        assert "timestamp" not in result[0]
        assert "timestamp" not in result[1]

    def test_converts_decimal_to_string(self):
        """Should convert Decimal to string."""
        data = {"score": Decimal("75.50")}
        result = canonicalize_for_hash(data)
        assert result["score"] == "75.50"

    def test_custom_exclude_keys(self):
        """Should use custom exclude keys if provided."""
        data = {"custom_key": "value", "keep_key": "kept"}
        result = canonicalize_for_hash(data, exclude_keys={"custom_key"})

        assert "custom_key" not in result
        assert result["keep_key"] == "kept"

    def test_sorts_dict_keys(self):
        """Should sort dict keys for determinism."""
        data = {"z": 1, "a": 2, "m": 3}
        result = canonicalize_for_hash(data)
        keys = list(result.keys())
        assert keys == sorted(keys)


class TestComputeResultsHash:
    """Tests for compute_results_hash function."""

    def test_excludes_timestamps(self):
        """Results with different timestamps should have same hash."""
        results1 = {"score": 75, "timestamp": "2024-01-01T00:00:00Z"}
        results2 = {"score": 75, "timestamp": "2024-01-02T00:00:00Z"}

        assert compute_results_hash(results1) == compute_results_hash(results2)

    def test_excludes_run_id(self):
        """Results with different run_ids should have same hash."""
        results1 = {"score": 75, "run_id": "run1"}
        results2 = {"score": 75, "run_id": "run2"}

        assert compute_results_hash(results1) == compute_results_hash(results2)

    def test_different_scores_different_hash(self):
        """Results with different scores should have different hash."""
        results1 = {"score": 75}
        results2 = {"score": 80}

        assert compute_results_hash(results1) != compute_results_hash(results2)


class TestDecimalEncoder:
    """Tests for DecimalEncoder JSON encoder."""

    def test_encodes_decimal(self):
        """Should encode Decimal as string."""
        data = {"value": Decimal("123.45")}
        result = json.dumps(data, cls=DecimalEncoder)
        assert '"123.45"' in result

    def test_preserves_other_types(self):
        """Should preserve other JSON types."""
        data = {"int": 1, "float": 2.5, "str": "test", "list": [1, 2]}
        result = json.dumps(data, cls=DecimalEncoder)
        parsed = json.loads(result)
        assert parsed["int"] == 1
        assert parsed["float"] == 2.5
        assert parsed["str"] == "test"
        assert parsed["list"] == [1, 2]


class TestManifestEntry:
    """Tests for ManifestEntry class."""

    def test_to_dict(self):
        """Should serialize to dict."""
        entry = ManifestEntry(
            run_id="test-run",
            timestamp="2024-01-01T00:00:00Z",
            config_hash="sha256:abc123",
            data_hashes={"prices": "sha256:def456"},
            results_hash="sha256:ghi789",
            module_versions={"v1": "1.0"},
            parameters={"horizon": 90},
            metadata={"analyst": "system"},
        )

        d = entry.to_dict()

        assert d["manifest_version"] == MANIFEST_VERSION
        assert d["run_id"] == "test-run"
        assert d["timestamp"] == "2024-01-01T00:00:00Z"
        assert d["config_hash"] == "sha256:abc123"
        assert d["data_hashes"]["prices"] == "sha256:def456"
        assert d["results_hash"] == "sha256:ghi789"
        assert d["module_versions"]["v1"] == "1.0"
        assert d["parameters"]["horizon"] == 90
        assert d["metadata"]["analyst"] == "system"

    def test_from_dict(self):
        """Should deserialize from dict."""
        d = {
            "manifest_version": MANIFEST_VERSION,
            "run_id": "test-run",
            "timestamp": "2024-01-01T00:00:00Z",
            "config_hash": "sha256:abc123",
            "data_hashes": {"prices": "sha256:def456"},
            "results_hash": "sha256:ghi789",
            "module_versions": {"v1": "1.0"},
            "parameters": {"horizon": 90},
            "metadata": {"analyst": "system"},
        }

        entry = ManifestEntry.from_dict(d)

        assert entry.run_id == "test-run"
        assert entry.timestamp == "2024-01-01T00:00:00Z"
        assert entry.config_hash == "sha256:abc123"
        assert entry.data_hashes["prices"] == "sha256:def456"
        assert entry.results_hash == "sha256:ghi789"
        assert entry.module_versions["v1"] == "1.0"
        assert entry.parameters["horizon"] == 90
        assert entry.metadata["analyst"] == "system"

    def test_roundtrip(self):
        """Should roundtrip through to_dict/from_dict."""
        original = ManifestEntry(
            run_id="roundtrip-test",
            timestamp="2024-06-15T12:30:00Z",
            config_hash="sha256:roundtrip",
            data_hashes={"data1": "sha256:hash1", "data2": "sha256:hash2"},
            results_hash="sha256:results",
            module_versions={"module1": "v1", "module2": "v2"},
            parameters={"param1": 1, "param2": "two"},
            metadata={"note": "test roundtrip"},
        )

        d = original.to_dict()
        restored = ManifestEntry.from_dict(d)

        assert restored.run_id == original.run_id
        assert restored.timestamp == original.timestamp
        assert restored.config_hash == original.config_hash
        assert restored.data_hashes == original.data_hashes
        assert restored.results_hash == original.results_hash
        assert restored.module_versions == original.module_versions
        assert restored.parameters == original.parameters
        assert restored.metadata == original.metadata


class TestRunManifest:
    """Tests for RunManifest class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_creates_output_dir(self, temp_dir):
        """Should create output directory if missing."""
        manifest_dir = Path(temp_dir) / "new_dir" / "nested"
        RunManifest(manifest_dir)
        assert manifest_dir.exists()

    def test_log_run_creates_entry(self, temp_dir):
        """Should create manifest entry."""
        manifest = RunManifest(temp_dir)

        entry = manifest.log_run(
            run_id="test-run-1",
            config={"weight": 0.5},
            data_hashes={"prices": "sha256:prices123"},
            results={"score": 75},
        )

        assert entry.run_id == "test-run-1"
        assert entry.config_hash.startswith("sha256:")
        assert entry.results_hash.startswith("sha256:")

    def test_log_run_writes_to_file(self, temp_dir):
        """Should write entry to JSONL file."""
        manifest = RunManifest(temp_dir)

        manifest.log_run(
            run_id="test-run-1",
            config={"weight": 0.5},
            data_hashes={"prices": "sha256:prices123"},
            results={"score": 75},
        )

        manifest_file = Path(temp_dir) / "run_manifest.jsonl"
        assert manifest_file.exists()

        with open(manifest_file) as f:
            lines = f.readlines()
        assert len(lines) == 1

        entry_data = json.loads(lines[0])
        assert entry_data["run_id"] == "test-run-1"

    def test_log_run_appends(self, temp_dir):
        """Should append multiple entries."""
        manifest = RunManifest(temp_dir)

        manifest.log_run(
            run_id="run-1",
            config={},
            data_hashes={},
            results={"score": 1},
        )
        manifest.log_run(
            run_id="run-2",
            config={},
            data_hashes={},
            results={"score": 2},
        )

        manifest_file = Path(temp_dir) / "run_manifest.jsonl"
        with open(manifest_file) as f:
            lines = f.readlines()
        assert len(lines) == 2

    def test_get_run(self, temp_dir):
        """Should retrieve run by ID."""
        manifest = RunManifest(temp_dir)

        manifest.log_run(
            run_id="target-run",
            config={"target": True},
            data_hashes={},
            results={"score": 75},
        )
        manifest.log_run(
            run_id="other-run",
            config={"target": False},
            data_hashes={},
            results={"score": 80},
        )

        entry = manifest.get_run("target-run")
        assert entry is not None
        assert entry.run_id == "target-run"

    def test_get_run_not_found(self, temp_dir):
        """Should return None if run not found."""
        manifest = RunManifest(temp_dir)
        result = manifest.get_run("nonexistent")
        assert result is None

    def test_query_runs_no_filter(self, temp_dir):
        """Should return all runs with no filter."""
        manifest = RunManifest(temp_dir)

        for i in range(3):
            manifest.log_run(
                run_id=f"run-{i}",
                config={},
                data_hashes={},
                results={"score": i},
            )

        results = manifest.query_runs()
        assert len(results) == 3

    def test_query_runs_by_config_hash(self, temp_dir):
        """Should filter by config hash."""
        manifest = RunManifest(temp_dir)

        entry1 = manifest.log_run(
            run_id="run-1",
            config={"config": "A"},
            data_hashes={},
            results={},
        )
        manifest.log_run(
            run_id="run-2",
            config={"config": "B"},
            data_hashes={},
            results={},
        )

        results = manifest.query_runs(config_hash=entry1.config_hash)
        assert len(results) == 1
        assert results[0].run_id == "run-1"

    def test_verify_run_success(self, temp_dir):
        """Should return True when results match."""
        manifest = RunManifest(temp_dir)
        results = {"score": 75, "tickers": ["A", "B"]}

        manifest.log_run(
            run_id="verify-test",
            config={},
            data_hashes={},
            results=results,
        )

        assert manifest.verify_run("verify-test", results) is True

    def test_verify_run_failure(self, temp_dir):
        """Should return False when results don't match."""
        manifest = RunManifest(temp_dir)

        manifest.log_run(
            run_id="verify-test",
            config={},
            data_hashes={},
            results={"score": 75},
        )

        # Different results
        different_results = {"score": 80}
        assert manifest.verify_run("verify-test", different_results) is False

    def test_verify_run_not_found(self, temp_dir):
        """Should return False when run not found."""
        manifest = RunManifest(temp_dir)
        assert manifest.verify_run("nonexistent", {}) is False

    def test_get_latest_run(self, temp_dir):
        """Should return most recent run."""
        manifest = RunManifest(temp_dir)

        manifest.log_run(run_id="run-1", config={}, data_hashes={}, results={})
        manifest.log_run(run_id="run-2", config={}, data_hashes={}, results={})
        manifest.log_run(run_id="run-3", config={}, data_hashes={}, results={})

        latest = manifest.get_latest_run()
        assert latest is not None
        assert latest.run_id == "run-3"

    def test_get_latest_run_empty(self, temp_dir):
        """Should return None when no runs exist."""
        manifest = RunManifest(temp_dir)
        assert manifest.get_latest_run() is None

    def test_export_summary(self, temp_dir):
        """Should export summary statistics."""
        manifest = RunManifest(temp_dir)

        manifest.log_run(
            run_id="run-1",
            config={"config": "A"},
            data_hashes={},
            results={},
            module_versions={"module1": "v1"},
        )
        manifest.log_run(
            run_id="run-2",
            config={"config": "A"},
            data_hashes={},
            results={},
            module_versions={"module1": "v2"},
        )

        summary = manifest.export_summary()

        assert summary["total_runs"] == 2
        assert summary["unique_configs"] == 1
        assert "date_range" in summary
        assert "v1" in summary["module_versions_seen"]
        assert "v2" in summary["module_versions_seen"]

    def test_export_summary_empty(self, temp_dir):
        """Should handle empty manifest."""
        manifest = RunManifest(temp_dir)
        summary = manifest.export_summary()
        assert summary["total_runs"] == 0


class TestCreateDataHashes:
    """Tests for create_data_hashes helper."""

    @pytest.fixture
    def temp_file(self):
        """Create a temporary file for tests."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("ticker,date,close\nAMGN,2024-01-01,100.0\n")
            f.flush()
            yield f.name
        Path(f.name).unlink()

    def test_creates_prices_hash(self, temp_file):
        """Should create hash for prices file."""
        hashes = create_data_hashes(prices_file=temp_file)
        assert "prices" in hashes
        assert hashes["prices"].startswith("sha256:")

    def test_creates_trials_hash(self):
        """Should create hash for trials data."""
        trials = [{"nct_id": "NCT001"}, {"nct_id": "NCT002"}]
        hashes = create_data_hashes(trials_data=trials)
        assert "trials" in hashes
        assert hashes["trials"].startswith("sha256:")

    def test_creates_snapshots_hash(self):
        """Should create hash for snapshots."""
        snapshots = [{"date": "2024-01-01", "tickers": ["A", "B"]}]
        hashes = create_data_hashes(snapshots=snapshots)
        assert "snapshots" in hashes
        assert hashes["snapshots"].startswith("sha256:")

    def test_handles_all_none(self):
        """Should return empty dict when all inputs are None."""
        hashes = create_data_hashes()
        assert hashes == {}


class TestLogBacktestRun:
    """Tests for log_backtest_run convenience function."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_logs_run(self, temp_dir):
        """Should log a backtest run."""
        entry = log_backtest_run(
            output_dir=temp_dir,
            run_id="backtest-1",
            config={"horizon": 90},
            results={"ic_mean": 0.05},
        )

        assert entry.run_id == "backtest-1"
        assert entry.config_hash.startswith("sha256:")

    def test_creates_manifest_file(self, temp_dir):
        """Should create manifest file."""
        log_backtest_run(
            output_dir=temp_dir,
            run_id="backtest-1",
            config={},
            results={},
        )

        manifest_file = Path(temp_dir) / "run_manifest.jsonl"
        assert manifest_file.exists()

    def test_with_metadata(self, temp_dir):
        """Should include metadata."""
        entry = log_backtest_run(
            output_dir=temp_dir,
            run_id="backtest-1",
            config={},
            results={},
            metadata={"analyst": "test-user"},
        )

        assert entry.metadata["analyst"] == "test-user"


class TestConstants:
    """Tests for module constants."""

    def test_manifest_version(self):
        """Manifest version should be semantic version."""
        parts = MANIFEST_VERSION.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_results_hash_exclude_keys(self):
        """Exclude keys should contain timestamp and run_id."""
        assert "timestamp" in RESULTS_HASH_EXCLUDE_KEYS
        assert "run_id" in RESULTS_HASH_EXCLUDE_KEYS
        assert "generated_at" in RESULTS_HASH_EXCLUDE_KEYS
