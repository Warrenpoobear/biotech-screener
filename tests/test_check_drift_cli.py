"""Tests for tools/check_drift_cli.py — standalone drift check CLI."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CLI_PATH = REPO_ROOT / "tools" / "check_drift_cli.py"

HEADER = [
    "ticker", "actionable_rank", "tier_dev", "eligible",
    "composite_rank", "composite_score",
]


def _write_rankings(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def _make_tickers(n: int, start: int = 1) -> list[dict]:
    return [
        {
            "ticker": f"TICK{i}",
            "actionable_rank": str(i),
            "tier_dev": "A" if i <= n // 4 else "B",
            "eligible": "True",
            "composite_rank": str(i),
            "composite_score": str(round(1.0 - i / (n + 1), 4)),
        }
        for i in range(start, start + n)
    ]


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CLI_PATH), *args],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )


class TestDriftCLIExitCodes:
    def test_identical_csv_exits_0(self, tmp_path):
        """Identical rankings → exit 0 (OK)."""
        rows = _make_tickers(30)
        current = tmp_path / "current" / "rankings.csv"
        prior = tmp_path / "prior" / "rankings.csv"
        _write_rankings(current, rows)
        _write_rankings(prior, rows)

        result = _run_cli(
            "--current-csv", str(current),
            "--prior-csv", str(prior),
            "--output-dir", str(tmp_path / "out"),
        )
        assert result.returncode == 0
        assert "OK" in result.stdout

    def test_high_drift_exits_2(self, tmp_path):
        """Completely different universes → exit 2 (WARN)."""
        current = tmp_path / "current" / "rankings.csv"
        prior = tmp_path / "prior" / "rankings.csv"
        _write_rankings(current, _make_tickers(30, start=1))
        _write_rankings(prior, _make_tickers(30, start=100))

        result = _run_cli(
            "--current-csv", str(current),
            "--prior-csv", str(prior),
            "--output-dir", str(tmp_path / "out"),
        )
        assert result.returncode == 2
        assert "WARN" in result.stdout

    def test_missing_current_csv_exits_2(self, tmp_path):
        """Missing current CSV → exit 2 (not 1)."""
        prior = tmp_path / "prior" / "rankings.csv"
        _write_rankings(prior, _make_tickers(10))

        result = _run_cli(
            "--current-csv", str(tmp_path / "nonexistent.csv"),
            "--prior-csv", str(prior),
            "--output-dir", str(tmp_path / "out"),
        )
        assert result.returncode == 2

    def test_missing_prior_csv_exits_2(self, tmp_path):
        """Missing prior CSV → exit 2 (not 1)."""
        current = tmp_path / "current" / "rankings.csv"
        _write_rankings(current, _make_tickers(10))

        result = _run_cli(
            "--current-csv", str(current),
            "--prior-csv", str(tmp_path / "nonexistent.csv"),
            "--output-dir", str(tmp_path / "out"),
        )
        assert result.returncode == 2


class TestDriftCLIOutputFiles:
    def test_writes_json_and_md(self, tmp_path):
        """Both drift_report.json and drift_report.md created."""
        rows = _make_tickers(30)
        current = tmp_path / "current" / "rankings.csv"
        prior = tmp_path / "prior" / "rankings.csv"
        _write_rankings(current, rows)
        _write_rankings(prior, rows)
        out_dir = tmp_path / "out"

        _run_cli(
            "--current-csv", str(current),
            "--prior-csv", str(prior),
            "--output-dir", str(out_dir),
        )

        assert (out_dir / "drift_report.json").exists()
        assert (out_dir / "drift_report.md").exists()

    def test_json_schema(self, tmp_path):
        """drift_report.json has expected keys."""
        rows = _make_tickers(30)
        current = tmp_path / "current" / "rankings.csv"
        prior = tmp_path / "prior" / "rankings.csv"
        _write_rankings(current, rows)
        _write_rankings(prior, rows)
        out_dir = tmp_path / "out"

        _run_cli(
            "--current-csv", str(current),
            "--prior-csv", str(prior),
            "--output-dir", str(out_dir),
        )

        report = json.loads((out_dir / "drift_report.json").read_text())
        assert report["version"] == "1.0.0"
        assert "thresholds_id" in report
        assert "metrics" in report
        assert "status" in report
        assert "warn_reasons" in report
        assert "top20_overlap_pct" in report["metrics"]
        assert "rank_spearman_rho" in report["metrics"]

    def test_creates_output_dir(self, tmp_path):
        """Output dir is created if it doesn't exist."""
        rows = _make_tickers(10)
        current = tmp_path / "current" / "rankings.csv"
        prior = tmp_path / "prior" / "rankings.csv"
        _write_rankings(current, rows)
        _write_rankings(prior, rows)
        deep_out = tmp_path / "a" / "b" / "c"

        _run_cli(
            "--current-csv", str(current),
            "--prior-csv", str(prior),
            "--output-dir", str(deep_out),
        )

        assert (deep_out / "drift_report.json").exists()


class TestDriftCLICustomThresholds:
    def test_custom_thresholds_respected(self, tmp_path):
        """Tight thresholds trigger WARN on moderate drift."""
        rows_prior = _make_tickers(30)
        # Small perturbation: swap ranks 1 and 2
        rows_current = [dict(r) for r in rows_prior]
        rows_current[0]["actionable_rank"] = "2"
        rows_current[0]["composite_rank"] = "2"
        rows_current[1]["actionable_rank"] = "1"
        rows_current[1]["composite_rank"] = "1"

        current = tmp_path / "current" / "rankings.csv"
        prior = tmp_path / "prior" / "rankings.csv"
        _write_rankings(current, rows_current)
        _write_rankings(prior, rows_prior)

        # With defaults → should be OK
        result_default = _run_cli(
            "--current-csv", str(current),
            "--prior-csv", str(prior),
            "--output-dir", str(tmp_path / "out_default"),
        )
        assert result_default.returncode == 0

        # With impossibly tight thresholds → should be WARN
        tight = {
            "warn_top20_overlap_pct": 100.0,
            "warn_top60_overlap_pct": 100.0,
            "warn_rank_spearman_rho": 0.9999,
            "warn_mean_abs_rank_delta_top60": 0.01,
            "warn_tier_migration_count": 0,
            "warn_eligibility_change_count": 0,
        }
        th_path = tmp_path / "tight.json"
        with open(th_path, "w") as f:
            json.dump(tight, f)

        result_tight = _run_cli(
            "--current-csv", str(current),
            "--prior-csv", str(prior),
            "--output-dir", str(tmp_path / "out_tight"),
            "--thresholds", str(th_path),
        )
        assert result_tight.returncode == 2

    def test_thresholds_id_in_report(self, tmp_path):
        """Custom thresholds ID appears in the drift report."""
        rows = _make_tickers(10)
        current = tmp_path / "current" / "rankings.csv"
        prior = tmp_path / "prior" / "rankings.csv"
        _write_rankings(current, rows)
        _write_rankings(prior, rows)

        custom = {"warn_top20_overlap_pct": 50.0}
        th_path = tmp_path / "custom.json"
        with open(th_path, "w") as f:
            json.dump(custom, f)

        _run_cli(
            "--current-csv", str(current),
            "--prior-csv", str(prior),
            "--output-dir", str(tmp_path / "out"),
            "--thresholds", str(th_path),
        )

        report = json.loads((tmp_path / "out" / "drift_report.json").read_text())
        assert report["thresholds"]["warn_top20_overlap_pct"] == 50.0


class TestDriftCLIAnnotations:
    def test_warning_annotations_on_breach(self, tmp_path):
        """::warning:: lines emitted to stdout when thresholds breached."""
        current = tmp_path / "current" / "rankings.csv"
        prior = tmp_path / "prior" / "rankings.csv"
        _write_rankings(current, _make_tickers(30, start=1))
        _write_rankings(prior, _make_tickers(30, start=100))

        result = _run_cli(
            "--current-csv", str(current),
            "--prior-csv", str(prior),
            "--output-dir", str(tmp_path / "out"),
        )

        warning_lines = [l for l in result.stdout.splitlines() if l.startswith("::warning::")]
        assert len(warning_lines) >= 1
        # At least overlap should be called out
        assert any("top20_overlap_pct" in l for l in warning_lines)

    def test_no_annotations_when_ok(self, tmp_path):
        """No ::warning:: lines when all metrics are fine."""
        rows = _make_tickers(30)
        current = tmp_path / "current" / "rankings.csv"
        prior = tmp_path / "prior" / "rankings.csv"
        _write_rankings(current, rows)
        _write_rankings(prior, rows)

        result = _run_cli(
            "--current-csv", str(current),
            "--prior-csv", str(prior),
            "--output-dir", str(tmp_path / "out"),
        )

        warning_lines = [l for l in result.stdout.splitlines() if l.startswith("::warning::")]
        assert len(warning_lines) == 0
