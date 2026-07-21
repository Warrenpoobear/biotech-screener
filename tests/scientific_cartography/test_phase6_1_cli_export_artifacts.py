"""Tests for Phase 6.1 CLI export-artifacts command."""

import json
import tempfile
from pathlib import Path

from scientific_cartography.cli import export_artifacts_command
from scientific_cartography.schemas.cluster_schema import CompetitiveClusterRecord
from scientific_cartography.schemas.landscape_feature_schema import LandscapeFeatureRecord
from scientific_cartography.schemas.program_schema import ProgramRecord


class Args:
    """Mock argparse.Namespace for CLI testing."""

    def __init__(
        self,
        as_of_date: str,
        artifact_dir: str,
        output_dir: str,
        created_at_utc: str = None,
    ):
        self.as_of_date = as_of_date
        self.artifact_dir = artifact_dir
        self.output_dir = output_dir
        self.created_at_utc = created_at_utc


class TestExportArtifactsCommand:
    """Test export-artifacts CLI command."""

    def _write_program_records(self, artifact_dir: Path, programs: list[ProgramRecord]) -> None:
        """Write programs to JSONL file."""
        programs_path = artifact_dir / "program_records.jsonl"
        with open(programs_path, "w") as f:
            for program in programs:
                f.write(json.dumps(program.to_dict()) + "\n")

    def _write_clusters(self, artifact_dir: Path, clusters: list[CompetitiveClusterRecord]) -> None:
        """Write clusters to JSONL file."""
        clusters_path = artifact_dir / "competitive_clusters.jsonl"
        with open(clusters_path, "w") as f:
            for cluster in clusters:
                f.write(json.dumps(cluster.to_dict()) + "\n")

    def _write_features(self, artifact_dir: Path, features: list[LandscapeFeatureRecord]) -> None:
        """Write features to JSONL file."""
        features_path = artifact_dir / "landscape_features.jsonl"
        with open(features_path, "w") as f:
            for feature in features:
                f.write(json.dumps(feature.to_dict()) + "\n")

    def test_writes_all_four_artifacts(self):
        """Should write all four export files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "artifacts"
            output_dir = Path(tmpdir) / "output"
            artifact_dir.mkdir()

            # Create minimal test data
            programs = [
                ProgramRecord(
                    program_id="P1",
                    asset_id="A1",
                    asset_name="Drug",
                    disease_id="DOID_001",
                    disease_name="Disease",
                    company_name="Company",
                    source_refs=["NCT001"],
                    as_of_date="2026-06-17",
                ),
            ]
            self._write_program_records(artifact_dir, programs)

            args = Args(
                as_of_date="2026-06-17",
                artifact_dir=str(artifact_dir),
                output_dir=str(output_dir),
            )

            result = export_artifacts_command(args)

            assert result == 0
            assert (output_dir / "map_index.json").exists()
            assert (output_dir / "disease_map_summary.json").exists()
            assert (output_dir / "disease_map_summary.md").exists()
            assert (output_dir / "artifact_manifest.json").exists()

    def test_fails_when_program_records_missing(self):
        """Should fail clearly when program_records.jsonl is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "artifacts"
            output_dir = Path(tmpdir) / "output"
            artifact_dir.mkdir()

            args = Args(
                as_of_date="2026-06-17",
                artifact_dir=str(artifact_dir),
                output_dir=str(output_dir),
            )

            result = export_artifacts_command(args)

            assert result == 1  # Failure

    def test_warns_when_clusters_missing(self):
        """Should warn and continue when competitive_clusters.jsonl is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "artifacts"
            output_dir = Path(tmpdir) / "output"
            artifact_dir.mkdir()

            programs = [
                ProgramRecord(
                    program_id="P1",
                    asset_id="A1",
                    asset_name="Drug",
                    disease_id="DOID_001",
                    disease_name="Disease",
                    company_name="Company",
                    source_refs=["NCT001"],
                    as_of_date="2026-06-17",
                ),
            ]
            self._write_program_records(artifact_dir, programs)

            args = Args(
                as_of_date="2026-06-17",
                artifact_dir=str(artifact_dir),
                output_dir=str(output_dir),
            )

            result = export_artifacts_command(args)

            assert result == 0
            assert (output_dir / "map_index.json").exists()

    def test_warns_when_features_missing(self):
        """Should warn and continue when landscape_features.jsonl is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "artifacts"
            output_dir = Path(tmpdir) / "output"
            artifact_dir.mkdir()

            programs = [
                ProgramRecord(
                    program_id="P1",
                    asset_id="A1",
                    asset_name="Drug",
                    disease_id="DOID_001",
                    disease_name="Disease",
                    company_name="Company",
                    source_refs=["NCT001"],
                    as_of_date="2026-06-17",
                ),
            ]
            self._write_program_records(artifact_dir, programs)

            args = Args(
                as_of_date="2026-06-17",
                artifact_dir=str(artifact_dir),
                output_dir=str(output_dir),
            )

            result = export_artifacts_command(args)

            assert result == 0
            assert (output_dir / "map_index.json").exists()

    def test_supports_deterministic_created_at_utc(self):
        """Should support deterministic --created-at-utc."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "artifacts"
            output_dir = Path(tmpdir) / "output"
            artifact_dir.mkdir()

            programs = [
                ProgramRecord(
                    program_id="P1",
                    asset_id="A1",
                    asset_name="Drug",
                    disease_id="DOID_001",
                    disease_name="Disease",
                    company_name="Company",
                    source_refs=["NCT001"],
                    as_of_date="2026-06-17",
                ),
            ]
            self._write_program_records(artifact_dir, programs)

            deterministic_timestamp = "2026-06-17T12:00:00Z"
            args = Args(
                as_of_date="2026-06-17",
                artifact_dir=str(artifact_dir),
                output_dir=str(output_dir),
                created_at_utc=deterministic_timestamp,
            )

            result = export_artifacts_command(args)

            assert result == 0
            manifest_path = output_dir / "artifact_manifest.json"
            with open(manifest_path) as f:
                manifest = json.load(f)
            assert manifest["created_at_utc"] == deterministic_timestamp

    def test_artifacts_contain_governance_flags(self):
        """Should write artifacts with governance flags."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "artifacts"
            output_dir = Path(tmpdir) / "output"
            artifact_dir.mkdir()

            programs = [
                ProgramRecord(
                    program_id="P1",
                    asset_id="A1",
                    asset_name="Drug",
                    disease_id="DOID_001",
                    disease_name="Disease",
                    company_name="Company",
                    source_refs=["NCT001"],
                    as_of_date="2026-06-17",
                ),
            ]
            self._write_program_records(artifact_dir, programs)

            args = Args(
                as_of_date="2026-06-17",
                artifact_dir=str(artifact_dir),
                output_dir=str(output_dir),
            )

            result = export_artifacts_command(args)

            assert result == 0

            # Check map_index has governance flags
            with open(output_dir / "map_index.json") as f:
                map_index = json.load(f)
            assert "governance" in map_index
            assert map_index["governance"]["production_wiring"] is False
            assert map_index["governance"]["ranker_change"] is False

            # Check manifest has governance flags
            with open(output_dir / "artifact_manifest.json") as f:
                manifest = json.load(f)
            assert "governance" in manifest
            assert manifest["governance"]["read_only_diagnostic"] is True
            assert manifest["governance"]["production_wiring"] is False

    def test_does_not_touch_production_rankings(self):
        """Should not modify production rankings.csv."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "artifacts"
            output_dir = Path(tmpdir) / "output"
            artifact_dir.mkdir()

            # Create a fake rankings.csv in a parent directory
            rankings_path = Path(tmpdir) / "rankings.csv"
            rankings_path.write_text("ticker,rank,score\nAAA,1,0.95\n")
            original_mtime = rankings_path.stat().st_mtime

            programs = [
                ProgramRecord(
                    program_id="P1",
                    asset_id="A1",
                    asset_name="Drug",
                    disease_id="DOID_001",
                    disease_name="Disease",
                    company_name="Company",
                    source_refs=["NCT001"],
                    as_of_date="2026-06-17",
                ),
            ]
            self._write_program_records(artifact_dir, programs)

            args = Args(
                as_of_date="2026-06-17",
                artifact_dir=str(artifact_dir),
                output_dir=str(output_dir),
            )

            export_artifacts_command(args)

            # Verify rankings.csv was not touched
            assert rankings_path.exists()
            assert rankings_path.stat().st_mtime == original_mtime

    def test_handles_all_artifacts_present(self):
        """Should work when all artifacts are present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "artifacts"
            output_dir = Path(tmpdir) / "output"
            artifact_dir.mkdir()

            programs = [
                ProgramRecord(
                    program_id="P1",
                    asset_id="A1",
                    asset_name="Drug",
                    disease_id="DOID_001",
                    disease_name="Disease",
                    company_name="Company",
                    source_refs=["NCT001"],
                    as_of_date="2026-06-17",
                ),
            ]
            clusters = [
                CompetitiveClusterRecord(
                    cluster_id="C1",
                    disease_id="DOID_001",
                    disease_name="Disease",
                    program_count=1,
                    as_of_date="2026-06-17",
                ),
            ]
            features = [
                LandscapeFeatureRecord(
                    feature_id="F1",
                    program_id="P1",
                    disease_id="DOID_001",
                    as_of_date="2026-06-17",
                ),
            ]

            self._write_program_records(artifact_dir, programs)
            self._write_clusters(artifact_dir, clusters)
            self._write_features(artifact_dir, features)

            args = Args(
                as_of_date="2026-06-17",
                artifact_dir=str(artifact_dir),
                output_dir=str(output_dir),
            )

            result = export_artifacts_command(args)

            assert result == 0
            # Verify all artifacts were written
            for artifact in [
                "map_index.json",
                "disease_map_summary.json",
                "disease_map_summary.md",
                "artifact_manifest.json",
            ]:
                assert (output_dir / artifact).exists()
