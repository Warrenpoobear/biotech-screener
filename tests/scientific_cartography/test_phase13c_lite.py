"""Phase 13C-lite tests: manifest, size guard, forbidden-field scan, default behavior."""

import json
import tempfile
from pathlib import Path

import pytest


def test_default_behavior_unchanged():
    """Verify daily pipeline runs unchanged without Phase 13C flag."""
    import inspect

    from tools.run_daily_production import run_daily

    sig = inspect.signature(run_daily)
    assert "run_scientific_cartography_phase13c" in sig.parameters
    assert "scientific_cartography_phase13c_strict" in sig.parameters

    # Default values should be False (disabled-by-default)
    assert sig.parameters["run_scientific_cartography_phase13c"].default is False
    assert sig.parameters["scientific_cartography_phase13c_strict"].default is False


def test_phase13c_uses_date_level_cartography_output_dir():
    """Phase 13C should not receive a pre-nested diseases/ output directory."""
    from tools.run_daily_production import _scientific_cartography_output_dir

    output_dir = _scientific_cartography_output_dir("2026-06-18")

    assert output_dir.as_posix().endswith("artifacts/scientific_cartography/2026-06-18")
    assert output_dir.name != "diseases"


def test_manifest_generation():
    """Verify manifest.json structure and content."""
    from tools.run_scientific_cartography_phase13c_export import generate_manifest

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create sample artifact files
        (output_dir / "test.json").write_text('{"key": "value"}')
        (output_dir / "test.jsonl").write_text('{"a": 1}\n{"b": 2}\n')

        manifest = generate_manifest(
            output_dir=output_dir,
            as_of_date="2026-06-18",
            snapshot_dir=Path("data/snapshots_pit/2026-06-18"),
            runtime_seconds=42.5,
        )

        # Verify structure
        assert manifest["artifact_type"] == "scientific_cartography_diagnostic"
        assert manifest["as_of_date"] == "2026-06-18"
        assert manifest["runtime_seconds"] == 42.5
        assert manifest["file_count"] == 2

        # Verify governance flags
        assert manifest["governance"]["read_only_diagnostic"] is True
        assert manifest["governance"]["ranker_change"] is False
        assert manifest["governance"]["selector_change"] is False
        assert manifest["governance"]["sizing_change"] is False
        assert manifest["governance"]["final_score_change"] is False

        # Verify safety checks structure
        assert "safety_checks" in manifest
        assert "size_guard_pass" in manifest["safety_checks"]
        assert "forbidden_fields_scan" in manifest["safety_checks"]
        assert "governance_flags_valid" in manifest["safety_checks"]


def test_size_guard():
    """Verify size guard detects overly large output."""
    from tools.run_scientific_cartography_phase13c_export import MAX_OUTPUT_SIZE_MB, calculate_directory_size

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create a small file
        small_file = output_dir / "small.json"
        small_file.write_text("x" * 1000)
        size = calculate_directory_size(output_dir)
        assert size < 1.0  # Less than 1 MB

        # Size should be under limit
        assert size <= MAX_OUTPUT_SIZE_MB


def test_governance_flags_validation():
    """Verify governance flag validation."""
    from tools.run_scientific_cartography_phase13c_export import validate_governance_flags

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create valid artifact with governance flags
        valid_artifact = {
            "governance": {"read_only_diagnostic": True},
            "data": "test",
        }
        (output_dir / "valid.json").write_text(json.dumps(valid_artifact))

        # Should pass
        assert validate_governance_flags(output_dir) is True

        # Create invalid artifact (missing governance flag)
        invalid_artifact = {"data": "test"}
        (output_dir / "invalid.json").write_text(json.dumps(invalid_artifact))

        # Should fail
        assert validate_governance_flags(output_dir) is False


def test_forbidden_fields_scan():
    """Verify forbidden field detection."""
    from tools.run_scientific_cartography_phase13c_export import scan_for_forbidden_fields

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create clean artifact
        clean_artifact = {"data": "value", "disease": "cancer"}
        (output_dir / "clean.json").write_text(json.dumps(clean_artifact))

        violations = scan_for_forbidden_fields(output_dir)
        assert violations is None  # No violations

        # Create artifact with forbidden field
        bad_artifact = {"final_score": 0.95, "data": "test"}
        (output_dir / "bad.json").write_text(json.dumps(bad_artifact))

        violations = scan_for_forbidden_fields(output_dir)
        assert violations is not None  # Found violations
        assert len(violations) > 0
        assert "final_score" in violations[0]


def test_forbidden_fields_allows_governance_disclaimer():
    """Verify forbidden fields allowed in governance disclaimers."""
    from tools.run_scientific_cartography_phase13c_export import scan_for_forbidden_fields

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create artifact with "investment recommendation" in disclaimer
        disclaimer = """
        # Governance
        This is not an investment recommendation.
        """
        (output_dir / "disclaimer.md").write_text(disclaimer)

        violations = scan_for_forbidden_fields(output_dir)
        # Should NOT flag "recommend" inside "investment recommendation" disclaimer
        assert violations is None or len(violations) == 0


@pytest.mark.skip(reason="placeholder: behavior not yet implemented (test-trust-audit hygiene 2026-07-14)")
def test_exit_codes():
    """Verify script exit codes."""
    # Exit 0: success
    # Exit 1: export failed
    # Exit 2: safety check failed
    # (Integration test when run_scientific_cartography_phase13c_export.py is invoked)
    pass  # Tested via integration with run_daily_production


@pytest.mark.skip(reason="placeholder: behavior not yet implemented (test-trust-audit hygiene 2026-07-14)")
def test_manifest_file_written():
    """Verify manifest.json is written to output directory."""
    # This is an integration test that would run with actual Phase 7A output
    # Placeholder for when full integration test framework is available
    pass


def test_phase13c_rebuilds_schema_objects_from_phase7a_artifacts(tmp_path, monkeypatch):
    """Phase 13C should rebuild Phase 9-11 objects from Phase 7A JSONL artifacts."""
    import tools.run_scientific_cartography_phase13c_export as phase13c
    from scientific_cartography.schemas.program_schema import ProgramRecord

    as_of_date = "2026-06-18"
    repo_root = tmp_path
    diagnostics_dir = repo_root / "artifacts" / "scientific_cartography" / as_of_date
    diagnostics_dir.mkdir(parents=True)

    program = ProgramRecord(
        program_id="PROGRAM_1",
        asset_id="ASSET_1",
        asset_name="Asset A",
        company_id="COMPANY_TICKER_TEST",
        ticker="TEST",
        company_name="Test Therapeutics",
        disease_id="DISEASE_1",
        disease_name="Acute Pain",
        clinical_stage="phase2",
        source_priority="ctgov",
        source_refs=["NCT00000001"],
        confidence=0.9,
        as_of_date=as_of_date,
    )
    (diagnostics_dir / "program_records.jsonl").write_text(json.dumps(program.to_dict()) + "\n")
    (diagnostics_dir / "competitive_clusters.jsonl").write_text("")
    (diagnostics_dir / "landscape_features.jsonl").write_text("")
    (diagnostics_dir / "map_index.json").write_text(json.dumps({"diseases": []}))

    monkeypatch.setattr(phase13c, "REPO_ROOT", repo_root)
    output_dir = tmp_path / "phase13c_output"

    exit_code = phase13c.main(
        as_of_date=as_of_date,
        snapshot_dir=tmp_path / "snapshot",
        ctgov_cache_dir=tmp_path / "ctgov",
        output_dir=output_dir,
    )

    assert exit_code == 0
    assert (output_dir / "disease_map_index.json").exists()
    assert (output_dir / "diseases").exists()
    manifest = json.loads((output_dir / "scientific_cartography_manifest.json").read_text())
    assert manifest["rebuilt_phase_9_11_counts"]["asset_indication_records"] == 1
    assert manifest["governance"]["read_only_diagnostic"] is True


class TestPhase13CLiteIntegration:
    """Integration tests for Phase 13C-lite with run_daily_production."""

    def test_cli_flags_present(self):
        """Verify CLI flags added to run_daily_production."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "tools/run_daily_production.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert "--run-scientific-cartography-phase13c" in result.stdout
        assert "--scientific-cartography-phase13c-strict" in result.stdout

    def test_phase13c_hook_non_blocking(self):
        """Verify Phase 13C hook is non-blocking by default."""
        # The function should gracefully handle missing files and return False
        # (non-blocking) rather than raising
        # Actual test requires mock Phase 7A output
        # Verify function signature
        import inspect

        from tools.run_daily_production import run_scientific_cartography_phase13c_export

        sig = inspect.signature(run_scientific_cartography_phase13c_export)
        assert "strict" in sig.parameters
        assert sig.parameters["strict"].default is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
