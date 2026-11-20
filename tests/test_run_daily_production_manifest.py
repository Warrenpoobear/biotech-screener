"""Tests for tools/run_daily_production.build_run_manifest.

Regression: a corrupt/truncated metadata.json or phase2_health.json in the
snapshot dir must not crash manifest build — the manifest must still be
written with a FAIL record so ops/QA can see the state.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.run_daily_production import build_run_manifest  # noqa: E402


@dataclass
class _GateStub:
    name: str = "cache_health"  # must be in GATE_ALLOWLIST
    status: str = "PASS"
    detail: str = ""
    value: Any = None
    threshold: Any = None


@dataclass
class _ConfigStub:
    """Minimal GateConfig-shaped stand-in — build_run_manifest only serialises asdict(config)."""

    xbi_stale_days: int = 3


@pytest.fixture
def screen_proc():
    return subprocess.CompletedProcess(args=["screen"], returncode=0)


def test_manifest_tolerates_corrupt_metadata(tmp_path, screen_proc):
    """Malformed metadata.json must not crash manifest assembly."""
    snap = tmp_path / "2026-04-15"
    snap.mkdir()
    (snap / "metadata.json").write_text("{ truncated JSON", encoding="utf-8")
    (snap / "rankings.csv").write_text("ticker,actionable_rank\nAAA,1\n", encoding="utf-8")

    manifest = build_run_manifest(
        as_of_date="2026-04-15",
        gate_results=[_GateStub(status="PASS")],
        price_stats={"n_extended": 1, "xbi_last_date": "2026-04-15"},
        screen_proc=screen_proc,
        audit_proc=None,
        config=_ConfigStub(),
        snapshot_date_dir=snap,
    )
    assert manifest["overall_status"] == "PASS"
    # Ruleset fields fall back to empty strings (no crash, recoverable record)
    assert manifest["ruleset"]["ruleset_version"] == ""
    assert manifest["ruleset"]["ruleset_hash"] == ""


def test_manifest_tolerates_corrupt_health(tmp_path, screen_proc):
    snap = tmp_path / "2026-04-15"
    snap.mkdir()
    (snap / "metadata.json").write_text(
        '{"version": "v1.13.0", "clinical_sort_telemetry": {"ruleset_id": "2a3e79eb"}, '
        '"ticker_count": 341, "active_universe": 297}',
        encoding="utf-8",
    )
    # Health file corrupt — must not override the good ruleset_id from metadata
    (snap / "phase2_health.json").write_text("{ garbage", encoding="utf-8")

    manifest = build_run_manifest(
        as_of_date="2026-04-15",
        gate_results=[_GateStub()],
        price_stats={},
        screen_proc=screen_proc,
        audit_proc=None,
        config=_ConfigStub(),
        snapshot_date_dir=snap,
    )
    # metadata's ruleset_id is kept since health parse failed
    assert manifest["ruleset"]["ruleset_hash"] == "2a3e79eb"
    assert manifest["ruleset"]["ruleset_version"] == "v1.13.0"


def test_manifest_tolerates_non_utf8_metadata(tmp_path, screen_proc):
    """Bytes-level corruption must still not crash."""
    snap = tmp_path / "2026-04-15"
    snap.mkdir()
    (snap / "metadata.json").write_bytes(b"\xff\xfe\x00\x01not valid anything")

    manifest = build_run_manifest(
        as_of_date="2026-04-15",
        gate_results=[_GateStub()],
        price_stats={},
        screen_proc=screen_proc,
        audit_proc=None,
        config=_ConfigStub(),
        snapshot_date_dir=snap,
    )
    assert manifest["overall_status"] == "PASS"
    assert manifest["price_staleness_exclusions"] == []


def test_manifest_records_price_staleness_exclusions(tmp_path, screen_proc):
    snap = tmp_path / "2026-11-20"
    snap.mkdir()
    (snap / "metadata.json").write_text(
        json.dumps(
            {
                "price_staleness_exclusions": [
                    {
                        "ticker": "SGMO",
                        "last_close": "2026-07-01",
                        "gap_days": 35,
                        "threshold_days": 10,
                        "action": "excluded",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest = build_run_manifest(
        as_of_date="2026-11-20",
        gate_results=[_GateStub()],
        price_stats={},
        screen_proc=screen_proc,
        audit_proc=None,
        config=_ConfigStub(),
        snapshot_date_dir=snap,
    )
    assert manifest["price_staleness_exclusions"][0]["ticker"] == "SGMO"
