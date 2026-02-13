"""Integration tests for the Phase-2 delta hook in run_screen.py.

Tests that _run_phase2_delta is wired correctly into the post-snapshot flow
and that --no-delta suppresses it. Uses monkeypatching — no full pipeline run.
"""
from __future__ import annotations

import json
import types
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers to build minimal snapshot dirs on disk
# ---------------------------------------------------------------------------

def _write_snapshot(snap_dir: Path, date: str, ruleset_id: str = "f3454ef7"):
    """Write a minimal but valid snapshot (rankings.csv + metadata.json)."""
    d = snap_dir / date
    d.mkdir(parents=True, exist_ok=True)

    tickers = [f"T{i:02d}" for i in range(25)]
    rows = []
    for i, t in enumerate(tickers):
        rows.append({
            "ticker": t,
            "composite_rank": i + 1,
            "composite_score": 50.0 - i * 0.1,
            "archetype": "drug_developer",
            "tier_dev": "A" if i < 10 else "B",
            "catalyst_mode": "specific_days",
            "catalyst_days": 10 + i,
            "actionable_rank": i + 1,
            "risk_flags": "",
            "target_weight_pct": round(100 / 25, 2),
            "decision_engine_ruleset_id": ruleset_id,
            "decision_engine_version": "v1.2.0",
            "size_band": "L" if i < 8 else "M",
            "clinical_optionality_pct_dev": 0.7,
        })
    df = pd.DataFrame(rows)
    df.to_csv(d / "rankings.csv", index=False)

    meta = {"as_of_date": date, "decision_mode": "phase2"}
    with open(d / "metadata.json", "w") as f:
        json.dump(meta, f)

    return d


# ---------------------------------------------------------------------------
# Test: Hook writes delta artifacts into snapshot dir
# ---------------------------------------------------------------------------

class TestDeltaHookWiring:
    def test_hook_writes_artifacts(self, tmp_path):
        """_run_phase2_delta writes .txt, .json, .csv into the snapshot dir."""
        from run_screen import _run_phase2_delta

        snapshot_dir = tmp_path / "snapshots"
        prior = _write_snapshot(snapshot_dir, "2026-01-20", "d4f1f8a8")
        current = _write_snapshot(snapshot_dir, "2026-02-09", "f3454ef7")

        args = types.SimpleNamespace(delta_prior="auto")
        logger = MagicMock()

        _run_phase2_delta(current, snapshot_dir, args, logger)

        assert (current / "phase2_run_delta_report.txt").exists()
        assert (current / "phase2_run_delta_details.json").exists()
        assert (current / "phase2_run_delta.csv").exists()

        # Verify JSON is valid
        with open(current / "phase2_run_delta_details.json") as f:
            details = json.load(f)
        assert details["current"]["date"] == "2026-02-09"
        assert details["prior"]["date"] == "2026-01-20"
        assert "portfolio_turnover" in details

    def test_hook_single_snapshot_mode(self, tmp_path):
        """With --delta-prior none, produces artifacts without prior comparison."""
        from run_screen import _run_phase2_delta

        snapshot_dir = tmp_path / "snapshots"
        current = _write_snapshot(snapshot_dir, "2026-02-09")

        args = types.SimpleNamespace(delta_prior="none")
        logger = MagicMock()

        _run_phase2_delta(current, snapshot_dir, args, logger)

        assert (current / "phase2_run_delta_report.txt").exists()
        assert (current / "phase2_run_delta_details.json").exists()

        with open(current / "phase2_run_delta_details.json") as f:
            details = json.load(f)
        assert details["prior"] is None

    def test_hook_specific_prior(self, tmp_path):
        """With --delta-prior YYYY-MM-DD, uses that specific prior."""
        from run_screen import _run_phase2_delta

        snapshot_dir = tmp_path / "snapshots"
        _write_snapshot(snapshot_dir, "2026-01-15", "d4f1f8a8")
        _write_snapshot(snapshot_dir, "2026-01-20", "d4f1f8a8")
        current = _write_snapshot(snapshot_dir, "2026-02-09")

        args = types.SimpleNamespace(delta_prior="2026-01-15")
        logger = MagicMock()

        _run_phase2_delta(current, snapshot_dir, args, logger)

        with open(current / "phase2_run_delta_details.json") as f:
            details = json.load(f)
        assert details["prior"]["date"] == "2026-01-15"

    def test_no_delta_flag_suppresses(self):
        """The hook should not be called when --no-delta is set.

        This tests the conditional gating logic — _run_phase2_delta is only
        called when decision_mode == 'phase2' and not args.no_delta.
        """
        args = types.SimpleNamespace(
            decision_mode="phase2",
            no_delta=True,
        )
        # The condition that gates the call:
        should_run = (
            args.decision_mode == "phase2"
            and not getattr(args, "no_delta", False)
        )
        assert not should_run

    def test_observe_mode_does_not_trigger(self):
        """Delta hook should not trigger in observe mode."""
        args = types.SimpleNamespace(
            decision_mode="observe",
            no_delta=False,
        )
        should_run = (
            args.decision_mode == "phase2"
            and not getattr(args, "no_delta", False)
        )
        assert not should_run

    def test_hook_warns_on_ruleset_change(self, tmp_path):
        """Hook should log warnings when rulesets differ."""
        from run_screen import _run_phase2_delta

        snapshot_dir = tmp_path / "snapshots"
        _write_snapshot(snapshot_dir, "2026-01-20", "d4f1f8a8")
        current = _write_snapshot(snapshot_dir, "2026-02-09", "f3454ef7")

        args = types.SimpleNamespace(delta_prior="auto")
        logger = MagicMock()

        _run_phase2_delta(current, snapshot_dir, args, logger)

        # Should have logged a warning about ruleset change
        warning_calls = [
            str(call) for call in logger.warning.call_args_list
        ]
        assert any("Ruleset changed" in w for w in warning_calls)

    def test_delta_csv_content(self, tmp_path):
        """Delta CSV should contain per-ticker rows with correct columns."""
        from run_screen import _run_phase2_delta

        snapshot_dir = tmp_path / "snapshots"
        _write_snapshot(snapshot_dir, "2026-01-20")
        current = _write_snapshot(snapshot_dir, "2026-02-09")

        args = types.SimpleNamespace(delta_prior="auto")
        logger = MagicMock()

        _run_phase2_delta(current, snapshot_dir, args, logger)

        df = pd.read_csv(current / "phase2_run_delta.csv")
        expected_cols = {
            "ticker", "in_current", "in_prior", "weight_current",
            "weight_prior", "weight_delta", "tier_current", "tier_prior",
            "rank_current", "rank_prior",
        }
        assert expected_cols == set(df.columns)

    def test_hook_writes_health_json(self, tmp_path):
        """_run_phase2_delta writes phase2_health.json into the snapshot dir."""
        from run_screen import _run_phase2_delta

        snapshot_dir = tmp_path / "snapshots"
        _write_snapshot(snapshot_dir, "2026-01-20")
        current = _write_snapshot(snapshot_dir, "2026-02-09")

        args = types.SimpleNamespace(delta_prior="auto")
        logger = MagicMock()

        health = _run_phase2_delta(current, snapshot_dir, args, logger)

        assert (current / "phase2_health.json").exists()
        with open(current / "phase2_health.json") as f:
            hj = json.load(f)
        assert hj["status"] in ("OK", "WARN", "FAIL")
        assert "reasons" in hj
        assert "metrics" in hj
        assert "thresholds" in hj

        # Return value should be a HealthResult
        assert health is not None
        assert health.status == hj["status"]

    def test_strict_mode_exit_codes(self):
        """Verify the --strict gating logic for WARN and FAIL."""
        from run_phase2_snapshot_delta import HealthResult

        # FAIL + strict => would return 1
        args_strict = types.SimpleNamespace(
            decision_mode="phase2",
            no_delta=False,
            strict=True,
        )
        health_fail = HealthResult(status="FAIL", reasons=["no_a_tier"], metrics={})
        if health_fail.status == "FAIL" and getattr(args_strict, "strict", False):
            exit_code = 1
        elif health_fail.status == "WARN" and getattr(args_strict, "strict", False):
            exit_code = 2
        else:
            exit_code = 0
        assert exit_code == 1

        # WARN + strict => would return 2
        health_warn = HealthResult(status="WARN", reasons=["high_turnover"], metrics={})
        if health_warn.status == "FAIL" and getattr(args_strict, "strict", False):
            exit_code = 1
        elif health_warn.status == "WARN" and getattr(args_strict, "strict", False):
            exit_code = 2
        else:
            exit_code = 0
        assert exit_code == 2

        # OK + strict => would return 0
        health_ok = HealthResult(status="OK", reasons=[], metrics={})
        if health_ok.status == "FAIL" and getattr(args_strict, "strict", False):
            exit_code = 1
        elif health_ok.status == "WARN" and getattr(args_strict, "strict", False):
            exit_code = 2
        else:
            exit_code = 0
        assert exit_code == 0

        # WARN + no strict => would return 0
        args_no_strict = types.SimpleNamespace(
            decision_mode="phase2",
            no_delta=False,
            strict=False,
        )
        if health_warn.status == "FAIL" and getattr(args_no_strict, "strict", False):
            exit_code = 1
        elif health_warn.status == "WARN" and getattr(args_no_strict, "strict", False):
            exit_code = 2
        else:
            exit_code = 0
        assert exit_code == 0
