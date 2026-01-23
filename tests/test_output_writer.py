#!/usr/bin/env python3
"""
Tests for Governance Output Writer

Covers:
- Governance metadata injection
- Canonical output writing
- Input lineage building
- Environment fingerprinting
"""

import pytest
import json
import tempfile
from datetime import date
from pathlib import Path
from typing import Dict, Any, List

# Import module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from governance.output_writer import (
    inject_governance_metadata,
    write_canonical_output,
    build_input_lineage,
    get_environment_fingerprint,
)
from governance.schema_registry import PIPELINE_VERSION, SCHEMA_VERSION


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_data():
    """Sample output data."""
    return {
        "scores": [
            {"ticker": "ACME", "score": 85},
            {"ticker": "BETA", "score": 72},
        ],
        "metadata": {
            "count": 2,
        },
    }


@pytest.fixture
def sample_lineage():
    """Sample input lineage."""
    return [
        {"path": "universe.json", "sha256": "abc123", "as_of_date": "2026-01-15"},
        {"path": "financial.json", "sha256": "def456", "as_of_date": "2026-01-15"},
    ]


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary output directory."""
    return tmp_path / "output"


@pytest.fixture
def tmp_input_files(tmp_path):
    """Create temporary input files."""
    files = []
    for name, content in [
        ("universe.json", '{"tickers": ["ACME", "BETA"]}'),
        ("financial.json", '{"records": []}'),
    ]:
        path = tmp_path / name
        path.write_text(content)
        files.append(path)
    return files


# ============================================================================
# INJECT GOVERNANCE METADATA
# ============================================================================

class TestInjectGovernanceMetadata:
    """Tests for governance metadata injection."""

    def test_basic_injection(self, sample_data, sample_lineage):
        """Injects _governance key into data."""
        result = inject_governance_metadata(
            data=sample_data,
            run_id="run_abc123",
            score_version="v1",
            parameters_hash="sha256:params123",
            input_lineage=sample_lineage,
        )

        assert "_governance" in result
        assert result["_governance"]["run_id"] == "run_abc123"
        assert result["_governance"]["score_version"] == "v1"
        assert result["_governance"]["parameters_hash"] == "sha256:params123"

    def test_preserves_original_data(self, sample_data, sample_lineage):
        """Original data fields are preserved."""
        result = inject_governance_metadata(
            data=sample_data,
            run_id="run_abc123",
            score_version="v1",
            parameters_hash="sha256:params123",
            input_lineage=sample_lineage,
        )

        assert "scores" in result
        assert result["scores"] == sample_data["scores"]
        assert "metadata" in result

    def test_includes_pipeline_version(self, sample_data, sample_lineage):
        """Includes pipeline version."""
        result = inject_governance_metadata(
            data=sample_data,
            run_id="run_abc123",
            score_version="v1",
            parameters_hash="sha256:params123",
            input_lineage=sample_lineage,
        )

        assert result["_governance"]["generation_metadata"]["pipeline_version"] == PIPELINE_VERSION

    def test_includes_tool_name(self, sample_data, sample_lineage):
        """Includes tool name."""
        result = inject_governance_metadata(
            data=sample_data,
            run_id="run_abc123",
            score_version="v1",
            parameters_hash="sha256:params123",
            input_lineage=sample_lineage,
        )

        assert result["_governance"]["generation_metadata"]["tool_name"] == "biotech-screener"

    def test_uses_default_schema_version(self, sample_data, sample_lineage):
        """Uses default schema version when not specified."""
        result = inject_governance_metadata(
            data=sample_data,
            run_id="run_abc123",
            score_version="v1",
            parameters_hash="sha256:params123",
            input_lineage=sample_lineage,
        )

        assert result["_governance"]["schema_version"] == SCHEMA_VERSION

    def test_uses_custom_schema_version(self, sample_data, sample_lineage):
        """Uses custom schema version when specified."""
        result = inject_governance_metadata(
            data=sample_data,
            run_id="run_abc123",
            score_version="v1",
            parameters_hash="sha256:params123",
            input_lineage=sample_lineage,
            schema_version="2.0.0",
        )

        assert result["_governance"]["schema_version"] == "2.0.0"

    def test_sorts_input_lineage(self, sample_data):
        """Input lineage is sorted by path."""
        lineage = [
            {"path": "z_file.json", "sha256": "hash1"},
            {"path": "a_file.json", "sha256": "hash2"},
            {"path": "m_file.json", "sha256": "hash3"},
        ]

        result = inject_governance_metadata(
            data=sample_data,
            run_id="run_abc123",
            score_version="v1",
            parameters_hash="sha256:params123",
            input_lineage=lineage,
        )

        paths = [item["path"] for item in result["_governance"]["input_lineage"]]
        assert paths == ["a_file.json", "m_file.json", "z_file.json"]

    def test_governance_first_in_output(self, sample_data, sample_lineage):
        """_governance key appears first in output."""
        result = inject_governance_metadata(
            data=sample_data,
            run_id="run_abc123",
            score_version="v1",
            parameters_hash="sha256:params123",
            input_lineage=sample_lineage,
        )

        keys = list(result.keys())
        assert keys[0] == "_governance"

    def test_empty_data(self, sample_lineage):
        """Works with empty data dict."""
        result = inject_governance_metadata(
            data={},
            run_id="run_abc123",
            score_version="v1",
            parameters_hash="sha256:params123",
            input_lineage=sample_lineage,
        )

        assert "_governance" in result
        assert len(result) == 1  # Only _governance

    def test_empty_lineage(self, sample_data):
        """Works with empty lineage."""
        result = inject_governance_metadata(
            data=sample_data,
            run_id="run_abc123",
            score_version="v1",
            parameters_hash="sha256:params123",
            input_lineage=[],
        )

        assert result["_governance"]["input_lineage"] == []


# ============================================================================
# WRITE CANONICAL OUTPUT
# ============================================================================

class TestWriteCanonicalOutput:
    """Tests for canonical output writing."""

    def test_writes_file(self, sample_data, sample_lineage, tmp_output_dir):
        """Writes output file."""
        output_path = tmp_output_dir / "output.json"

        result = write_canonical_output(
            data=sample_data,
            output_path=output_path,
            run_id="run_abc123",
            score_version="v1",
            parameters_hash="sha256:params123",
            input_lineage=sample_lineage,
        )

        assert output_path.exists()
        assert result["path"] == str(output_path)

    def test_returns_hash(self, sample_data, sample_lineage, tmp_output_dir):
        """Returns SHA256 hash of output."""
        output_path = tmp_output_dir / "output.json"

        result = write_canonical_output(
            data=sample_data,
            output_path=output_path,
            run_id="run_abc123",
            score_version="v1",
            parameters_hash="sha256:params123",
            input_lineage=sample_lineage,
        )

        assert "sha256" in result
        assert len(result["sha256"]) == 64  # SHA256 hex length

    def test_creates_parent_directories(self, sample_data, sample_lineage, tmp_path):
        """Creates parent directories if needed."""
        output_path = tmp_path / "deep" / "nested" / "dir" / "output.json"

        write_canonical_output(
            data=sample_data,
            output_path=output_path,
            run_id="run_abc123",
            score_version="v1",
            parameters_hash="sha256:params123",
            input_lineage=sample_lineage,
        )

        assert output_path.exists()

    def test_output_is_valid_json(self, sample_data, sample_lineage, tmp_output_dir):
        """Output is valid JSON."""
        output_path = tmp_output_dir / "output.json"

        write_canonical_output(
            data=sample_data,
            output_path=output_path,
            run_id="run_abc123",
            score_version="v1",
            parameters_hash="sha256:params123",
            input_lineage=sample_lineage,
        )

        with open(output_path) as f:
            loaded = json.load(f)

        assert "_governance" in loaded

    def test_output_includes_governance(self, sample_data, sample_lineage, tmp_output_dir):
        """Output includes governance metadata."""
        output_path = tmp_output_dir / "output.json"

        write_canonical_output(
            data=sample_data,
            output_path=output_path,
            run_id="run_abc123",
            score_version="v1",
            parameters_hash="sha256:params123",
            input_lineage=sample_lineage,
        )

        with open(output_path) as f:
            loaded = json.load(f)

        assert loaded["_governance"]["run_id"] == "run_abc123"

    def test_deterministic_hash(self, sample_data, sample_lineage, tmp_output_dir):
        """Same inputs produce same hash."""
        output_path1 = tmp_output_dir / "output1.json"
        output_path2 = tmp_output_dir / "output2.json"

        result1 = write_canonical_output(
            data=sample_data,
            output_path=output_path1,
            run_id="run_abc123",
            score_version="v1",
            parameters_hash="sha256:params123",
            input_lineage=sample_lineage,
        )

        result2 = write_canonical_output(
            data=sample_data,
            output_path=output_path2,
            run_id="run_abc123",
            score_version="v1",
            parameters_hash="sha256:params123",
            input_lineage=sample_lineage,
        )

        assert result1["sha256"] == result2["sha256"]

    def test_accepts_string_path(self, sample_data, sample_lineage, tmp_output_dir):
        """Accepts string path."""
        output_path = str(tmp_output_dir / "output.json")

        result = write_canonical_output(
            data=sample_data,
            output_path=output_path,
            run_id="run_abc123",
            score_version="v1",
            parameters_hash="sha256:params123",
            input_lineage=sample_lineage,
        )

        assert Path(output_path).exists()


# ============================================================================
# BUILD INPUT LINEAGE
# ============================================================================

class TestBuildInputLineage:
    """Tests for input lineage building."""

    def test_basic_lineage(self, tmp_input_files):
        """Builds lineage from existing files."""
        lineage = build_input_lineage(
            input_files=tmp_input_files,
            as_of_date="2026-01-15",
        )

        assert len(lineage) == 2
        assert all("path" in item for item in lineage)
        assert all("sha256" in item for item in lineage)
        assert all("as_of_date" in item for item in lineage)

    def test_includes_as_of_date(self, tmp_input_files):
        """Includes as_of_date in each record."""
        lineage = build_input_lineage(
            input_files=tmp_input_files,
            as_of_date="2026-01-15",
        )

        for item in lineage:
            assert item["as_of_date"] == "2026-01-15"

    def test_uses_filename_only(self, tmp_input_files):
        """Uses filename only, not full path."""
        lineage = build_input_lineage(
            input_files=tmp_input_files,
            as_of_date="2026-01-15",
        )

        for item in lineage:
            assert "/" not in item["path"]

    def test_includes_schema_version_when_provided(self, tmp_input_files):
        """Includes schema_version when provided."""
        lineage = build_input_lineage(
            input_files=tmp_input_files,
            as_of_date="2026-01-15",
            schema_version="1.0.0",
        )

        for item in lineage:
            assert item["schema_version"] == "1.0.0"

    def test_skips_missing_files(self, tmp_path, tmp_input_files):
        """Skips files that don't exist."""
        files = tmp_input_files + [tmp_path / "nonexistent.json"]

        lineage = build_input_lineage(
            input_files=files,
            as_of_date="2026-01-15",
        )

        assert len(lineage) == 2  # Only existing files

    def test_sorted_by_path(self, tmp_path):
        """Lineage is sorted by path."""
        files = []
        for name in ["z_file.json", "a_file.json", "m_file.json"]:
            path = tmp_path / name
            path.write_text("{}")
            files.append(path)

        lineage = build_input_lineage(
            input_files=files,
            as_of_date="2026-01-15",
        )

        paths = [item["path"] for item in lineage]
        assert paths == ["a_file.json", "m_file.json", "z_file.json"]

    def test_empty_input_list(self):
        """Handles empty input list."""
        lineage = build_input_lineage(
            input_files=[],
            as_of_date="2026-01-15",
        )

        assert lineage == []

    def test_accepts_string_paths(self, tmp_input_files):
        """Accepts string paths."""
        string_paths = [str(f) for f in tmp_input_files]

        lineage = build_input_lineage(
            input_files=string_paths,
            as_of_date="2026-01-15",
        )

        assert len(lineage) == 2

    def test_hash_is_deterministic(self, tmp_input_files):
        """Same file produces same hash."""
        lineage1 = build_input_lineage(
            input_files=tmp_input_files,
            as_of_date="2026-01-15",
        )

        lineage2 = build_input_lineage(
            input_files=tmp_input_files,
            as_of_date="2026-01-15",
        )

        hashes1 = {item["path"]: item["sha256"] for item in lineage1}
        hashes2 = {item["path"]: item["sha256"] for item in lineage2}

        assert hashes1 == hashes2


# ============================================================================
# ENVIRONMENT FINGERPRINT
# ============================================================================

class TestGetEnvironmentFingerprint:
    """Tests for environment fingerprinting."""

    def test_returns_dict(self):
        """Returns a dictionary."""
        fingerprint = get_environment_fingerprint()
        assert isinstance(fingerprint, dict)

    def test_includes_platform(self):
        """Includes platform info."""
        fingerprint = get_environment_fingerprint()
        assert "platform" in fingerprint
        assert isinstance(fingerprint["platform"], str)

    def test_includes_platform_release(self):
        """Includes platform release."""
        fingerprint = get_environment_fingerprint()
        assert "platform_release" in fingerprint

    def test_includes_python_version(self):
        """Includes Python version."""
        fingerprint = get_environment_fingerprint()
        assert "python_version" in fingerprint
        assert "." in fingerprint["python_version"]

    def test_deterministic(self):
        """Same call produces same fingerprint."""
        fp1 = get_environment_fingerprint()
        fp2 = get_environment_fingerprint()

        assert fp1 == fp2


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_special_characters_in_data(self, sample_lineage, tmp_output_dir):
        """Handles special characters in data."""
        data = {
            "unicode": "\u4e2d\u6587",  # Chinese characters
            "emoji": "\U0001F4C8",  # Chart emoji
            "quotes": 'He said "hello"',
        }

        output_path = tmp_output_dir / "output.json"

        result = write_canonical_output(
            data=data,
            output_path=output_path,
            run_id="run_abc123",
            score_version="v1",
            parameters_hash="sha256:params123",
            input_lineage=sample_lineage,
        )

        with open(output_path, encoding='utf-8') as f:
            loaded = json.load(f)

        assert loaded["unicode"] == "\u4e2d\u6587"

    def test_nested_data_structures(self, sample_lineage, tmp_output_dir):
        """Handles deeply nested data structures."""
        data = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": ["a", "b", "c"]
                    }
                }
            }
        }

        output_path = tmp_output_dir / "output.json"

        write_canonical_output(
            data=data,
            output_path=output_path,
            run_id="run_abc123",
            score_version="v1",
            parameters_hash="sha256:params123",
            input_lineage=sample_lineage,
        )

        with open(output_path) as f:
            loaded = json.load(f)

        assert loaded["level1"]["level2"]["level3"]["level4"] == ["a", "b", "c"]

    def test_large_data_structure(self, sample_lineage, tmp_output_dir):
        """Handles large data structures."""
        data = {
            "items": [{"id": i, "value": f"item_{i}"} for i in range(1000)]
        }

        output_path = tmp_output_dir / "output.json"

        result = write_canonical_output(
            data=data,
            output_path=output_path,
            run_id="run_abc123",
            score_version="v1",
            parameters_hash="sha256:params123",
            input_lineage=sample_lineage,
        )

        assert output_path.exists()
        assert "sha256" in result

