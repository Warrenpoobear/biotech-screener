"""Tests for run_phase2_daily.py — the Phase-2 daily wrapper.

Covers:
    - exit-code passthrough (0 / 1 / 2)
    - missing phase2_health.json → UNKNOWN / exit 1
    - malformed JSON → exit 1
    - log path emitted on WARN / FAIL
    - --dry-run skips health JSON parsing
    - unexpected exit code → exit 1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from run_phase2_daily import (
    assert_outputs_exist,
    build_command,
    format_summary,
    main,
    read_health_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_health_json(snap_dir: Path, date: str, status: str,
                       reasons: list[str] | None = None) -> Path:
    """Write a minimal phase2_health.json + sibling files into snap_dir/date/."""
    d = snap_dir / date
    d.mkdir(parents=True, exist_ok=True)
    health = {
        "status": status,
        "reasons": reasons or [],
        "metrics": {},
        "thresholds_id": "test0000",
    }
    p = d / "phase2_health.json"
    p.write_text(json.dumps(health))
    # Sibling files expected by assert_outputs_exist (rankings.csv must be >= 100 bytes)
    csv_header = "ticker,composite_score,rank,tier\n"
    csv_rows = "".join(f"TICK{i:02d},0.{i:02d},{i},B\n" for i in range(10))
    (d / "rankings.csv").write_text(csv_header + csv_rows)
    (d / "metadata.json").write_text(json.dumps({"as_of_date": date}))
    return p


def _fake_subprocess_run(returncode: int):
    """Return a mock for subprocess.run that returns *returncode* and writes to the log file."""
    def _run(cmd, stdout=None, stderr=None):
        # Write something to the log so the file exists
        if stdout and hasattr(stdout, "write"):
            stdout.write(f"fake log output, rc={returncode}\n")
        result = MagicMock()
        result.returncode = returncode
        return result
    return _run


# ---------------------------------------------------------------------------
# read_health_json
# ---------------------------------------------------------------------------

class TestReadHealthJson:
    def test_ok(self, tmp_path: Path):
        _write_health_json(tmp_path, "2026-01-15", "OK")
        h = read_health_json(tmp_path, "2026-01-15")
        assert h is not None
        assert h["status"] == "OK"

    def test_missing_file(self, tmp_path: Path):
        assert read_health_json(tmp_path, "2026-01-15") is None

    def test_malformed_json(self, tmp_path: Path):
        d = tmp_path / "2026-01-15"
        d.mkdir()
        (d / "phase2_health.json").write_text("{bad json")
        assert read_health_json(tmp_path, "2026-01-15") is None


# ---------------------------------------------------------------------------
# format_summary
# ---------------------------------------------------------------------------

class TestFormatSummary:
    def test_ok_no_reasons(self, tmp_path: Path):
        s = format_summary("2026-01-15", "OK", [], tmp_path)
        assert "[PHASE2] 2026-01-15" in s
        assert "OK" in s
        assert "snapshot=" in s

    def test_warn_with_reasons(self, tmp_path: Path):
        s = format_summary("2026-01-15", "WARN", ["no_a_tier_regime"], tmp_path)
        assert "WARN" in s
        assert "no_a_tier_regime" in s

    def test_fail_multiple_reasons(self, tmp_path: Path):
        s = format_summary("2026-01-15", "FAIL",
                           ["optionality_broken", "catalyst_broken"], tmp_path)
        assert "FAIL" in s
        assert "optionality_broken" in s
        assert "catalyst_broken" in s


# ---------------------------------------------------------------------------
# build_command
# ---------------------------------------------------------------------------

class TestBuildCommand:
    def test_minimal(self, tmp_path: Path):
        import argparse
        args = argparse.Namespace(
            as_of_date="2026-01-15",
            data_dir=tmp_path / "data",
            snapshot_dir=tmp_path / "snaps",
            health_thresholds=None,
            dry_run=False,
        )
        cmd = build_command(args, [])
        assert "--decision-mode" in cmd
        assert "phase2" in cmd
        assert "--strict" in cmd
        assert "--as-of-date" in cmd

    def test_with_extras(self, tmp_path: Path):
        import argparse
        args = argparse.Namespace(
            as_of_date="2026-01-15",
            data_dir=tmp_path / "data",
            snapshot_dir=tmp_path / "snaps",
            health_thresholds=Path("/th.json"),
            dry_run=True,
        )
        cmd = build_command(args, ["--no-delta"])
        assert "--health-thresholds" in cmd
        assert "--dry-run" in cmd
        assert "--no-delta" in cmd


# ---------------------------------------------------------------------------
# main() — exit-code passthrough
# ---------------------------------------------------------------------------

class TestExitCodePassthrough:
    """Subprocess exit code flows through correctly."""

    @patch("run_phase2_daily.subprocess.run")
    def test_exit_0_ok(self, mock_run, tmp_path: Path):
        mock_run.side_effect = _fake_subprocess_run(0)
        snap = tmp_path / "snaps"
        _write_health_json(snap, "2026-01-15", "OK")
        log = tmp_path / "log.txt"

        rc = main([
            "--as-of-date", "2026-01-15",
            "--data-dir", str(tmp_path / "data"),
            "--snapshot-dir", str(snap),
            "--log-file", str(log),
        ])
        assert rc == 0

    @patch("run_phase2_daily.subprocess.run")
    def test_exit_1_fail(self, mock_run, tmp_path: Path):
        mock_run.side_effect = _fake_subprocess_run(1)
        snap = tmp_path / "snaps"
        _write_health_json(snap, "2026-01-15", "FAIL", ["optionality_broken"])
        log = tmp_path / "log.txt"

        rc = main([
            "--as-of-date", "2026-01-15",
            "--data-dir", str(tmp_path / "data"),
            "--snapshot-dir", str(snap),
            "--log-file", str(log),
        ])
        assert rc == 1

    @patch("run_phase2_daily.subprocess.run")
    def test_exit_2_warn(self, mock_run, tmp_path: Path):
        mock_run.side_effect = _fake_subprocess_run(2)
        snap = tmp_path / "snaps"
        _write_health_json(snap, "2026-01-15", "WARN", ["no_a_tier_regime"])
        log = tmp_path / "log.txt"

        rc = main([
            "--as-of-date", "2026-01-15",
            "--data-dir", str(tmp_path / "data"),
            "--snapshot-dir", str(snap),
            "--log-file", str(log),
        ])
        assert rc == 2


# ---------------------------------------------------------------------------
# main() — missing / malformed health JSON
# ---------------------------------------------------------------------------

class TestMissingHealthJson:
    @patch("run_phase2_daily.subprocess.run")
    def test_no_health_json_returns_1(self, mock_run, tmp_path: Path):
        """Pipeline exits 0 but no health JSON → UNKNOWN / exit 1."""
        mock_run.side_effect = _fake_subprocess_run(0)
        snap = tmp_path / "snaps"
        snap.mkdir()
        log = tmp_path / "log.txt"

        rc = main([
            "--as-of-date", "2026-01-15",
            "--data-dir", str(tmp_path / "data"),
            "--snapshot-dir", str(snap),
            "--log-file", str(log),
        ])
        assert rc == 1

    @patch("run_phase2_daily.subprocess.run")
    def test_malformed_json_returns_1(self, mock_run, tmp_path: Path):
        mock_run.side_effect = _fake_subprocess_run(0)
        snap = tmp_path / "snaps"
        d = snap / "2026-01-15"
        d.mkdir(parents=True)
        (d / "phase2_health.json").write_text("NOT JSON{{{{")
        log = tmp_path / "log.txt"

        rc = main([
            "--as-of-date", "2026-01-15",
            "--data-dir", str(tmp_path / "data"),
            "--snapshot-dir", str(snap),
            "--log-file", str(log),
        ])
        assert rc == 1


# ---------------------------------------------------------------------------
# main() — unexpected exit code
# ---------------------------------------------------------------------------

class TestUnexpectedExitCode:
    @patch("run_phase2_daily.subprocess.run")
    def test_exit_130_treated_as_fail(self, mock_run, tmp_path: Path):
        """Unexpected exit (e.g. SIGINT=130) with valid health JSON → exit 1."""
        mock_run.side_effect = _fake_subprocess_run(130)
        snap = tmp_path / "snaps"
        _write_health_json(snap, "2026-01-15", "OK")
        log = tmp_path / "log.txt"

        rc = main([
            "--as-of-date", "2026-01-15",
            "--data-dir", str(tmp_path / "data"),
            "--snapshot-dir", str(snap),
            "--log-file", str(log),
        ])
        assert rc == 1


# ---------------------------------------------------------------------------
# main() — --dry-run
# ---------------------------------------------------------------------------

class TestDryRun:
    @patch("run_phase2_daily.subprocess.run")
    def test_dry_run_passthrough(self, mock_run, tmp_path: Path):
        """--dry-run passes through exit code without reading health JSON."""
        mock_run.side_effect = _fake_subprocess_run(0)
        snap = tmp_path / "snaps"
        # Deliberately do NOT write health JSON
        log = tmp_path / "log.txt"

        rc = main([
            "--as-of-date", "2026-01-15",
            "--data-dir", str(tmp_path / "data"),
            "--snapshot-dir", str(snap),
            "--log-file", str(log),
            "--dry-run",
        ])
        assert rc == 0

    @patch("run_phase2_daily.subprocess.run")
    def test_dry_run_nonzero(self, mock_run, tmp_path: Path):
        """--dry-run with non-zero exit passes through."""
        mock_run.side_effect = _fake_subprocess_run(1)
        log = tmp_path / "log.txt"

        rc = main([
            "--as-of-date", "2026-01-15",
            "--data-dir", str(tmp_path / "data"),
            "--snapshot-dir", str(tmp_path / "snaps"),
            "--log-file", str(log),
            "--dry-run",
        ])
        assert rc == 1


# ---------------------------------------------------------------------------
# main() — log file created
# ---------------------------------------------------------------------------

class TestLogFile:
    @patch("run_phase2_daily.subprocess.run")
    def test_log_file_created(self, mock_run, tmp_path: Path):
        mock_run.side_effect = _fake_subprocess_run(0)
        snap = tmp_path / "snaps"
        _write_health_json(snap, "2026-01-15", "OK")
        log = tmp_path / "output" / "phase2_daily.log"

        main([
            "--as-of-date", "2026-01-15",
            "--data-dir", str(tmp_path / "data"),
            "--snapshot-dir", str(snap),
            "--log-file", str(log),
        ])
        assert log.exists()

    @patch("run_phase2_daily.subprocess.run")
    def test_stderr_on_fail(self, mock_run, tmp_path: Path, capsys):
        mock_run.side_effect = _fake_subprocess_run(1)
        snap = tmp_path / "snaps"
        _write_health_json(snap, "2026-01-15", "FAIL", ["optionality_broken"])
        log = tmp_path / "log.txt"

        main([
            "--as-of-date", "2026-01-15",
            "--data-dir", str(tmp_path / "data"),
            "--snapshot-dir", str(snap),
            "--log-file", str(log),
        ])
        err = capsys.readouterr().err
        assert "optionality_broken" in err
        assert str(log) in err


# ---------------------------------------------------------------------------
# assert_outputs_exist
# ---------------------------------------------------------------------------

class TestAssertOutputsExist:
    def test_all_present(self, tmp_path: Path):
        """All three files present and non-empty → no errors."""
        _write_health_json(tmp_path, "2026-01-15", "OK")
        errors = assert_outputs_exist(tmp_path, "2026-01-15")
        assert errors == []

    def test_missing_one_file(self, tmp_path: Path):
        """One critical file missing → one error."""
        _write_health_json(tmp_path, "2026-01-15", "OK")
        (tmp_path / "2026-01-15" / "rankings.csv").unlink()
        errors = assert_outputs_exist(tmp_path, "2026-01-15")
        assert len(errors) == 1
        assert "Missing" in errors[0]
        assert "rankings.csv" in errors[0]

    def test_empty_file(self, tmp_path: Path):
        """A file exists but is too small → truncation error."""
        _write_health_json(tmp_path, "2026-01-15", "OK")
        (tmp_path / "2026-01-15" / "rankings.csv").write_text("")
        errors = assert_outputs_exist(tmp_path, "2026-01-15")
        assert len(errors) == 1
        assert "Empty/truncated" in errors[0]

    def test_no_snapshot_dir(self, tmp_path: Path):
        """Snapshot dir doesn't exist → all three files missing."""
        errors = assert_outputs_exist(tmp_path, "2026-01-15")
        assert len(errors) == 3


# ---------------------------------------------------------------------------
# main() — output assertion integration
# ---------------------------------------------------------------------------

class TestOutputAssertion:
    @patch("run_phase2_daily.subprocess.run")
    def test_subprocess_ok_but_no_outputs(self, mock_run, tmp_path: Path):
        """Subprocess exits 0 but no output files → main returns 1."""
        mock_run.side_effect = _fake_subprocess_run(0)
        snap = tmp_path / "snaps"
        snap.mkdir()
        log = tmp_path / "log.txt"

        rc = main([
            "--as-of-date", "2026-01-15",
            "--data-dir", str(tmp_path / "data"),
            "--snapshot-dir", str(snap),
            "--log-file", str(log),
        ])
        assert rc == 1
