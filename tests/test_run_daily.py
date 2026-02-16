"""Tests for run_daily.py — cache-complete daily orchestrator."""
from __future__ import annotations

import subprocess
import sys
from io import StringIO
from pathlib import Path
from unittest import mock

import pytest

# Import the module under test
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_daily


# ── TestResolveDates ────────────────────────────────────────────────────


class TestResolveDates:
    def _ns(self, **kw):
        """Build a minimal Namespace for resolve_dates."""
        import argparse
        defaults = {"as_of_date": "2026-02-15", "from_date": None, "to_date": None}
        defaults.update(kw)
        return argparse.Namespace(**defaults)

    def test_single_date(self):
        dates = run_daily.resolve_dates(self._ns(as_of_date="2026-02-15"))
        assert dates == ["2026-02-15"]

    def test_range_skips_weekends(self):
        # 2026-02-13 Fri, 2026-02-14 Sat, 2026-02-15 Sun, 2026-02-16 Mon
        dates = run_daily.resolve_dates(
            self._ns(from_date="2026-02-13", to_date="2026-02-16")
        )
        assert dates == ["2026-02-13", "2026-02-16"]

    def test_range_same_day_business(self):
        # Wednesday
        dates = run_daily.resolve_dates(
            self._ns(from_date="2026-02-11", to_date="2026-02-11")
        )
        assert dates == ["2026-02-11"]

    def test_range_same_day_weekend(self):
        # Saturday → empty
        dates = run_daily.resolve_dates(
            self._ns(from_date="2026-02-14", to_date="2026-02-14")
        )
        assert dates == []

    def test_range_full_week(self):
        # Mon 2026-02-09 through Fri 2026-02-13
        dates = run_daily.resolve_dates(
            self._ns(from_date="2026-02-09", to_date="2026-02-13")
        )
        assert len(dates) == 5
        assert dates[0] == "2026-02-09"
        assert dates[-1] == "2026-02-13"


# ── TestBuildCommands ───────────────────────────────────────────────────


class TestBuildCommands:
    def _ns(self, **overrides):
        import argparse
        defaults = {
            "as_of_date": "2026-02-15",
            "sources": "fda_adcom,sec_8k",
            "data_dir": Path("production_data"),
            "snapshot_dir": Path("data/snapshots"),
            "health_thresholds": None,
            "dry_run": False,
            "pit_mode": "strict",
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_warm_cmd(self):
        cmd = run_daily.build_warm_cmd("2026-02-15", self._ns())
        assert cmd[1].endswith("warm_caches.py")
        assert "--as-of-date" in cmd
        assert "2026-02-15" in cmd
        assert "--sources" in cmd
        assert "fda_adcom,sec_8k" in cmd
        assert "--data-dir" in cmd

    def test_screen_cmd_basic(self):
        cmd = run_daily.build_screen_cmd("2026-02-15", self._ns())
        assert cmd[1].endswith("run_phase2_daily.py")
        assert "2026-02-15" in cmd
        assert "--pit-mode" not in cmd  # strict is default, omitted

    def test_screen_cmd_with_pit_mode(self):
        cmd = run_daily.build_screen_cmd("2026-02-15", self._ns(pit_mode="degrade"))
        assert "--pit-mode" in cmd
        assert "degrade" in cmd

    def test_screen_cmd_with_health_thresholds(self):
        cmd = run_daily.build_screen_cmd(
            "2026-02-15", self._ns(health_thresholds=Path("custom.json"))
        )
        assert "--health-thresholds" in cmd
        assert "custom.json" in cmd

    def test_screen_cmd_with_dry_run(self):
        cmd = run_daily.build_screen_cmd("2026-02-15", self._ns(dry_run=True))
        assert "--dry-run" in cmd

    def test_screen_cmd_with_extras(self):
        cmd = run_daily.build_screen_cmd(
            "2026-02-15", self._ns(), extra=["--some-flag", "val"]
        )
        assert cmd[-2:] == ["--some-flag", "val"]

    def test_rollup_cmd(self):
        cmd = run_daily.build_rollup_cmd(self._ns())
        assert cmd[1].endswith("rollup_shadow_metrics.py")
        assert "--snapshot-dir" in cmd


# ── TestRunStep ─────────────────────────────────────────────────────────


class TestRunStep:
    def test_dry_run_returns_zero(self, capsys):
        rc = run_daily.run_step(["echo", "hello"], "test-label", dry_run=True)
        assert rc == 0
        out = capsys.readouterr().out
        assert "[DAILY]" in out
        assert "echo hello" in out

    @mock.patch("run_daily.subprocess.run")
    def test_returns_exit_code(self, mock_run):
        mock_run.return_value = mock.Mock(returncode=2)
        rc = run_daily.run_step(["fake"], "test-label", dry_run=False)
        assert rc == 2
        mock_run.assert_called_once_with(["fake"])


# ── TestMain ────────────────────────────────────────────────────────────


class TestMain:
    @mock.patch("run_daily.subprocess.run")
    def test_single_date_runs_all_steps(self, mock_run):
        mock_run.return_value = mock.Mock(returncode=0)
        rc = run_daily.main(["--as-of-date", "2026-02-15", "--no-skip-existing"])
        assert rc == 0
        # warm + screen + rollup = 3 calls
        assert mock_run.call_count == 3
        cmds = [call.args[0] for call in mock_run.call_args_list]
        assert "warm_caches.py" in cmds[0][1]
        assert "run_phase2_daily.py" in cmds[1][1]
        assert "rollup_shadow_metrics.py" in cmds[2][1]

    @mock.patch("run_daily.subprocess.run")
    def test_no_warm_skips_warm(self, mock_run):
        mock_run.return_value = mock.Mock(returncode=0)
        rc = run_daily.main(["--as-of-date", "2026-02-15", "--no-warm", "--no-skip-existing"])
        assert rc == 0
        # screen + rollup = 2 calls
        assert mock_run.call_count == 2
        cmds = [call.args[0] for call in mock_run.call_args_list]
        assert all("warm_caches.py" not in c[1] for c in cmds)

    @mock.patch("run_daily.subprocess.run")
    def test_no_rollup_skips_rollup(self, mock_run):
        mock_run.return_value = mock.Mock(returncode=0)
        rc = run_daily.main(["--as-of-date", "2026-02-15", "--no-rollup", "--no-skip-existing"])
        assert rc == 0
        # warm + screen = 2 calls
        assert mock_run.call_count == 2
        cmds = [call.args[0] for call in mock_run.call_args_list]
        assert all("rollup_shadow_metrics.py" not in c[1] for c in cmds)

    @mock.patch("run_daily.subprocess.run")
    def test_fail_returns_nonzero(self, mock_run):
        # screen returns FAIL
        mock_run.side_effect = [
            mock.Mock(returncode=0),  # warm
            mock.Mock(returncode=1),  # screen FAIL
            mock.Mock(returncode=0),  # rollup
        ]
        rc = run_daily.main(["--as-of-date", "2026-02-15", "--no-skip-existing"])
        assert rc == 1

    @mock.patch("run_daily.subprocess.run")
    def test_stop_on_fail_halts_backfill(self, mock_run):
        # First date: warm OK, screen FAIL → stop, no second date
        mock_run.side_effect = [
            mock.Mock(returncode=0),  # warm date-1
            mock.Mock(returncode=1),  # screen date-1 FAIL
            mock.Mock(returncode=0),  # rollup (still runs)
        ]
        rc = run_daily.main([
            "--from-date", "2026-02-09",  # Mon
            "--to-date", "2026-02-10",    # Tue
            "--stop-on-fail",
            "--no-skip-existing",
        ])
        assert rc == 1
        # warm(1) + screen(1) + rollup = 3, NOT warm(2)+screen(2)+rollup=5
        assert mock_run.call_count == 3

    def test_dry_run_no_subprocess(self, capsys):
        rc = run_daily.main(["--as-of-date", "2026-02-15", "--dry-run"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "warm_caches.py" in out
        assert "run_phase2_daily.py" in out
        assert "rollup_shadow_metrics.py" in out


# ── TestSkipExisting ────────────────────────────────────────────────────


class TestSkipExisting:
    @mock.patch("run_daily.subprocess.run")
    def test_skips_date_with_existing_rankings(self, mock_run, tmp_path, capsys):
        # Create a fake existing snapshot
        snap = tmp_path / "2026-02-15"
        snap.mkdir()
        (snap / "rankings.csv").write_text("ticker\nAAPL\n")
        mock_run.return_value = mock.Mock(returncode=0)
        rc = run_daily.main([
            "--as-of-date", "2026-02-15",
            "--snapshot-dir", str(tmp_path),
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "snapshot exists" in out
        # Only rollup should run (warm+screen skipped)
        assert mock_run.call_count == 1
        assert "rollup_shadow_metrics.py" in mock_run.call_args_list[0].args[0][1]

    @mock.patch("run_daily.subprocess.run")
    def test_no_skip_existing_forces_rerun(self, mock_run, tmp_path):
        snap = tmp_path / "2026-02-15"
        snap.mkdir()
        (snap / "rankings.csv").write_text("ticker\nAAPL\n")
        mock_run.return_value = mock.Mock(returncode=0)
        rc = run_daily.main([
            "--as-of-date", "2026-02-15",
            "--snapshot-dir", str(tmp_path),
            "--no-skip-existing",
        ])
        assert rc == 0
        # warm + screen + rollup = 3
        assert mock_run.call_count == 3

    @mock.patch("run_daily.subprocess.run")
    def test_range_skips_only_completed_dates(self, mock_run, tmp_path, capsys):
        # Mon 2026-02-09 exists, Tue 2026-02-10 does not
        snap_09 = tmp_path / "2026-02-09"
        snap_09.mkdir()
        (snap_09 / "rankings.csv").write_text("ticker\nAAPL\n")
        mock_run.return_value = mock.Mock(returncode=0)
        rc = run_daily.main([
            "--from-date", "2026-02-09",
            "--to-date", "2026-02-10",
            "--snapshot-dir", str(tmp_path),
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "2026-02-09" in out and "snapshot exists" in out
        # warm(10) + screen(10) + rollup = 3 (only date-10 processed)
        assert mock_run.call_count == 3


# ── TestValidation ──────────────────────────────────────────────────────


class TestValidation:
    def test_range_missing_to_date(self):
        with pytest.raises(SystemExit):
            run_daily.main(["--from-date", "2026-02-09"])

    def test_range_inverted(self):
        with pytest.raises(SystemExit):
            run_daily.main(["--from-date", "2026-02-15", "--to-date", "2026-02-09"])


# ── TestDeltaSeedThreading ─────────────────────────────────────────────


class TestFindCache:
    def test_find_8k_cache_found(self, tmp_path):
        """Glob finds versioned cache file in cache dir."""
        cache_file = tmp_path / "8k_catalysts_2026-02-14_abcd1234.json"
        cache_file.write_text("[]")
        result = run_daily._find_8k_cache("2026-02-14", tmp_path)
        assert result == cache_file

    def test_find_8k_cache_missing(self, tmp_path):
        """No cache file → returns None."""
        result = run_daily._find_8k_cache("2026-02-14", tmp_path)
        assert result is None


class TestDeltaSeedThreading:
    def test_range_second_date_gets_seed(self, tmp_path, capsys, monkeypatch):
        """In range mode, 2nd date's warm cmd includes --seed-cache."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        # Plant a cache for the first date
        cache_file = cache_dir / "8k_catalysts_2026-02-12_abcd1234.json"
        cache_file.write_text("[]")

        monkeypatch.setattr("run_daily._DEFAULT_8K_CACHE_DIR", cache_dir)

        captured_cmds: list[list[str]] = []

        def fake_run_step(cmd, label, *, dry_run=False):
            captured_cmds.append(cmd)
            return 0

        monkeypatch.setattr("run_daily.run_step", fake_run_step)

        run_daily.main([
            "--from-date", "2026-02-12",
            "--to-date", "2026-02-13",
            "--no-rollup",
            "--no-skip-existing",
        ])

        warm_cmds = [c for c in captured_cmds if any("warm_caches.py" in s for s in c)]
        assert len(warm_cmds) == 2
        # First date: no --seed-cache
        assert "--seed-cache" not in warm_cmds[0]
        # Second date: has --seed-cache pointing to first date's cache
        assert "--seed-cache" in warm_cmds[1]
        seed_idx = warm_cmds[1].index("--seed-cache")
        assert "2026-02-12" in warm_cmds[1][seed_idx + 1]

    def test_single_date_no_seed(self, tmp_path, capsys, monkeypatch):
        """Single-date mode: no --seed-cache in warm cmd."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr("run_daily._DEFAULT_8K_CACHE_DIR", cache_dir)

        captured_cmds: list[list[str]] = []

        def fake_run_step(cmd, label, *, dry_run=False):
            captured_cmds.append(cmd)
            return 0

        monkeypatch.setattr("run_daily.run_step", fake_run_step)

        run_daily.main([
            "--as-of-date", "2026-02-15",
            "--no-rollup",
            "--no-skip-existing",
        ])

        warm_cmds = [c for c in captured_cmds if any("warm_caches.py" in s for s in c)]
        assert len(warm_cmds) == 1
        assert "--seed-cache" not in warm_cmds[0]
