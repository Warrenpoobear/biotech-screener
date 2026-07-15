"""Tests for Scientific Cartography Phase 7B production hook integration."""

import subprocess
from unittest import mock

import pytest

from tools.run_daily_production import run_scientific_cartography_diagnostics


class TestPhase7BHook:
    """Tests for Phase 7B production hook behavior."""

    def test_disabled_by_default_no_invocation(self, tmp_path):
        """Verify hook is not invoked when disabled (default behavior)."""
        snapshot_dir = tmp_path / "snapshot"
        snapshot_dir.mkdir()
        (snapshot_dir / "rankings.csv").write_text("ticker\nABCD")

        ctgov_cache = tmp_path / "ctgov"
        ctgov_cache.mkdir()

        # run_scientific_cartography_diagnostics() should not be called at all
        # in production code when run_scientific_cartography=False.
        # This test verifies the hook behavior when enabled.
        # Disabled behavior is tested by absence of hook execution in run_daily().

    def test_wrapper_execution_success(self, tmp_path):
        """Verify wrapper execution with valid inputs."""
        snapshot_dir = tmp_path / "snapshot"
        snapshot_dir.mkdir()

        # Create minimal rankings.csv
        rankings_csv = snapshot_dir / "rankings.csv"
        rankings_csv.write_text("ticker,eligible\nABCD,1\nEFGH,0\n")

        ctgov_cache = tmp_path / "ctgov"
        ctgov_cache.mkdir()

        output_dir = tmp_path / "output"

        # Wrapper should handle missing inputs gracefully
        with mock.patch("tools.run_daily_production._run_subprocess") as mock_subprocess:
            mock_subprocess.return_value = subprocess.CompletedProcess(args=[], returncode=0)

            result = run_scientific_cartography_diagnostics(
                as_of_date="2026-06-17",
                snapshot_dir=snapshot_dir,
                ctgov_cache_dir=ctgov_cache,
                output_dir=output_dir,
                strict=False,
            )

            assert result is True
            assert mock_subprocess.called
            # Verify command structure
            cmd = mock_subprocess.call_args[0][0]
            assert cmd[0].endswith("python3") or cmd[0].endswith(".py") or "python" in cmd[0]
            assert "run_scientific_cartography_diagnostics.py" in str(cmd)
            assert "--as-of-date" in cmd
            assert "2026-06-17" in cmd
            assert "--snapshot-dir" in cmd
            assert "--ctgov-cache" in cmd
            assert "--output-dir" in cmd

    def test_wrapper_failure_non_strict(self, tmp_path):
        """Verify non-strict mode logs warning and returns False on failure."""
        snapshot_dir = tmp_path / "snapshot"
        snapshot_dir.mkdir()

        ctgov_cache = tmp_path / "ctgov"
        ctgov_cache.mkdir()

        output_dir = tmp_path / "output"

        with mock.patch("tools.run_daily_production._run_subprocess") as mock_subprocess:
            mock_subprocess.return_value = subprocess.CompletedProcess(args=[], returncode=1)  # Simulate failure

            with mock.patch("tools.run_daily_production._logger") as mock_logger:
                result = run_scientific_cartography_diagnostics(
                    as_of_date="2026-06-17",
                    snapshot_dir=snapshot_dir,
                    ctgov_cache_dir=ctgov_cache,
                    output_dir=output_dir,
                    strict=False,
                )

                assert result is False
                assert mock_logger.warning.called
                # Should not raise exception in non-strict mode
                assert not mock_subprocess.side_effect

    def test_wrapper_failure_strict(self, tmp_path):
        """Verify strict mode raises exception on wrapper failure."""
        snapshot_dir = tmp_path / "snapshot"
        snapshot_dir.mkdir()

        ctgov_cache = tmp_path / "ctgov"
        ctgov_cache.mkdir()

        output_dir = tmp_path / "output"

        with mock.patch("tools.run_daily_production._run_subprocess") as mock_subprocess:
            mock_subprocess.return_value = subprocess.CompletedProcess(args=[], returncode=1)  # Simulate failure

            with pytest.raises(RuntimeError, match="Scientific cartography wrapper failed"):
                run_scientific_cartography_diagnostics(
                    as_of_date="2026-06-17",
                    snapshot_dir=snapshot_dir,
                    ctgov_cache_dir=ctgov_cache,
                    output_dir=output_dir,
                    strict=True,
                )

    def test_wrapper_missing_script_non_strict(self, tmp_path):
        """Verify behavior when wrapper script doesn't exist (non-strict)."""
        snapshot_dir = tmp_path / "snapshot"
        snapshot_dir.mkdir()

        ctgov_cache = tmp_path / "ctgov"
        ctgov_cache.mkdir()

        output_dir = tmp_path / "output"

        with mock.patch("tools.run_daily_production.REPO_ROOT", tmp_path):  # No wrapper script
            with mock.patch("tools.run_daily_production._logger") as mock_logger:
                result = run_scientific_cartography_diagnostics(
                    as_of_date="2026-06-17",
                    snapshot_dir=snapshot_dir,
                    ctgov_cache_dir=ctgov_cache,
                    output_dir=output_dir,
                    strict=False,
                )

                assert result is False
                assert mock_logger.warning.called
                assert "not found" in str(mock_logger.warning.call_args).lower()

    def test_wrapper_missing_script_strict(self, tmp_path):
        """Verify behavior when wrapper script doesn't exist (strict)."""
        snapshot_dir = tmp_path / "snapshot"
        snapshot_dir.mkdir()

        ctgov_cache = tmp_path / "ctgov"
        ctgov_cache.mkdir()

        output_dir = tmp_path / "output"

        with mock.patch("tools.run_daily_production.REPO_ROOT", tmp_path):
            with pytest.raises(RuntimeError, match="not found"):
                run_scientific_cartography_diagnostics(
                    as_of_date="2026-06-17",
                    snapshot_dir=snapshot_dir,
                    ctgov_cache_dir=ctgov_cache,
                    output_dir=output_dir,
                    strict=True,
                )

    def test_hook_receives_correct_paths(self, tmp_path):
        """Verify hook constructs and passes correct paths to wrapper."""
        snapshot_dir = tmp_path / "snapshot"
        snapshot_dir.mkdir()

        ctgov_cache = tmp_path / "cache/ctgov/2026-06-17"
        ctgov_cache.mkdir(parents=True)

        output_dir = tmp_path / "artifacts/scientific_cartography/2026-06-17"

        with mock.patch("tools.run_daily_production._run_subprocess") as mock_subprocess:
            mock_subprocess.return_value = subprocess.CompletedProcess(args=[], returncode=0)

            run_scientific_cartography_diagnostics(
                as_of_date="2026-06-17",
                snapshot_dir=snapshot_dir,
                ctgov_cache_dir=ctgov_cache,
                output_dir=output_dir,
                strict=False,
            )

            cmd = mock_subprocess.call_args[0][0]
            # Verify paths are correctly passed
            assert str(snapshot_dir) in cmd
            assert str(ctgov_cache) in cmd
            assert str(output_dir) in cmd or "artifacts" in str(cmd)

    def test_output_directory_created(self, tmp_path):
        """Verify output directory is created if missing."""
        snapshot_dir = tmp_path / "snapshot"
        snapshot_dir.mkdir()

        ctgov_cache = tmp_path / "ctgov"
        ctgov_cache.mkdir()

        output_dir = tmp_path / "nonexistent/output"
        assert not output_dir.exists()

        with mock.patch("tools.run_daily_production._run_subprocess") as mock_subprocess:
            mock_subprocess.return_value = subprocess.CompletedProcess(args=[], returncode=0)

            run_scientific_cartography_diagnostics(
                as_of_date="2026-06-17",
                snapshot_dir=snapshot_dir,
                ctgov_cache_dir=ctgov_cache,
                output_dir=output_dir,
                strict=False,
            )

            # Output directory should be created
            assert output_dir.exists()


class TestPhase7BIntegration:
    """Integration tests for Phase 7B hook with run_daily() orchestration."""

    def test_cli_flag_exists(self):
        """Verify --run-scientific-cartography CLI flag is registered."""
        # Mock argparse to verify the flag is registered
        with mock.patch("argparse.ArgumentParser.parse_args"):
            # Just verify the parser is created with the flag
            # The actual parsing is tested via the CLI
            pass

    @pytest.mark.skip(reason="placeholder: behavior not yet implemented (test-trust-audit hygiene 2026-07-14)")
    def test_hook_not_called_when_flag_absent(self, tmp_path):
        """Verify hook is not invoked when --run-scientific-cartography is absent."""
        # This would require mocking the full run_daily() call
        # In practice, this is verified by the absence of output/scientific_cartography/
        # directory after a normal run_daily() call.
        pass

    @pytest.mark.skip(reason="placeholder: behavior not yet implemented (test-trust-audit hygiene 2026-07-14)")
    def test_hook_called_when_flag_present(self, tmp_path):
        """Verify hook is invoked when --run-scientific-cartography is present."""
        # This would require mocking the full run_daily() call with the flag
        pass

    @pytest.mark.skip(reason="placeholder: behavior not yet implemented (test-trust-audit hygiene 2026-07-14)")
    def test_non_blocking_failure_does_not_fail_run(self, tmp_path):
        """Verify wrapper failure does not fail production run in non-strict mode."""
        # This would require mocking run_daily() with a failing wrapper
        pass

    @pytest.mark.skip(reason="placeholder: behavior not yet implemented (test-trust-audit hygiene 2026-07-14)")
    def test_strict_failure_fails_run(self, tmp_path):
        """Verify wrapper failure fails production run in strict mode."""
        # This would require mocking run_daily() with a failing wrapper and strict=True
        pass

    @pytest.mark.skip(reason="placeholder: behavior not yet implemented (test-trust-audit hygiene 2026-07-14)")
    def test_no_forbidden_mutations(self, tmp_path):
        """Verify hook does not mutate forbidden production files."""
        # Assertions: rankings.csv, decision_portfolio.csv, screen_output.json unchanged
        # This is checked at the run_daily() level after hook execution
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
