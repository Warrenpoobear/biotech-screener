"""Regression tests for run_screen audit fixes."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from run_production_screen import save_run_log
from run_screen import _attach_phase2_decision_ruleset_manifest, add_bootstrap_analysis, compute_data_hash
from run_screen_checkpoint import verify_against_prior_manifest
from tools.run_daily_production import GateConfig, _load_progress, _mark_step, build_run_manifest, promote_snapshot

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_bootstrap_analysis_does_not_reference_missing_regime_result(tmp_path):
    (tmp_path / "universe.json").write_text('[{"ticker":"AAA"}]\n', encoding="utf-8")
    (tmp_path / "price_history.csv").write_text("ticker,date,close\nAAA,2026-06-19,1\n", encoding="utf-8")
    results = {
        "module_5_composite": {
            "ranked_securities": [
                {"ticker": "AAA", "composite_score": "1.0"},
                {"ticker": "BBB", "composite_score": "2.0"},
            ]
        }
    }

    updated = add_bootstrap_analysis(results, "2026-06-19", tmp_path, n_bootstrap=4)

    assert "bootstrap_analysis" in updated
    assert "regime_result" not in updated


def test_bootstrap_data_hash_ignores_generated_json_outputs(tmp_path):
    (tmp_path / "universe.json").write_text('[{"ticker":"AAA"}]\n', encoding="utf-8")
    (tmp_path / "price_history.csv").write_text("ticker,date,close\nAAA,2026-06-19,1\n", encoding="utf-8")
    before = compute_data_hash(tmp_path)

    (tmp_path / "screen_output.json").write_text('{"generated":1}\n', encoding="utf-8")
    (tmp_path / "diagnostics_screen_2026-06-19.json").write_text('{"generated":2}\n', encoding="utf-8")
    after = compute_data_hash(tmp_path)

    assert after == before


def test_phase2_decision_ruleset_manifest_is_required_and_drift_checked(tmp_path):
    ruleset_path = tmp_path / "ruleset.json"
    ruleset_path.write_text('{"ruleset_id":"abc"}\n', encoding="utf-8")
    manifest = {
        "manifest_version": "v1",
        "as_of_date": "2026-06-19",
        "generated_at": "2026-06-19T00:00:00Z",
        "data_dir": str(tmp_path),
        "dependencies": [],
        "validation": {"all_required_present": True, "errors": [], "warnings": []},
    }
    results = {"run_metadata": {"inputs_manifest": manifest}}

    _attach_phase2_decision_ruleset_manifest(results, ruleset_path)

    dep = next(d for d in manifest["dependencies"] if d["key"] == "decision_ruleset")
    assert dep["required"] is True
    assert dep["sha256"] == hashlib.sha256(ruleset_path.read_bytes()).hexdigest()

    prior = json.loads(json.dumps(manifest))
    prior["dependencies"][0]["sha256"] = "0" * 64
    assert verify_against_prior_manifest(manifest, prior)


def test_run_screen_cli_exposes_no_coinvest_flag():
    proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "run_screen.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--no-coinvest" in proc.stdout


def test_legacy_production_run_log_uses_as_of_timestamp(tmp_path):
    log_path = save_run_log(
        production_dir=tmp_path,
        as_of_date="2026-06-19",
        success=True,
        results_path=tmp_path / "results.json",
    )

    payload = json.loads(log_path.read_text(encoding="utf-8"))
    assert payload["run_timestamp"] == "2026-06-19T00:00:00Z"


def test_daily_manifest_uses_as_of_timestamp():
    manifest = build_run_manifest(
        "2026-06-19",
        [],
        {},
        subprocess.CompletedProcess(args=[], returncode=0),
        None,
        GateConfig(),
        git_pre_run={"sha": "abc"},
    )

    assert manifest["generated_at"] == "2026-06-19T00:00:00Z"


def test_daily_progress_timestamps_are_as_of_based(tmp_path):
    snap_dir = tmp_path / "2026-06-19"

    assert _load_progress(snap_dir)["started_at"] == "2026-06-19T00:00:00Z"

    _mark_step(snap_dir, "screen")
    progress = json.loads((snap_dir / "_step_progress.json").read_text(encoding="utf-8"))
    assert progress["steps"]["screen"]["completed_at"] == "2026-06-19T00:00:00Z"
    assert progress["last_updated"] == "2026-06-19T00:00:00Z"


def test_snapshot_backup_name_does_not_use_wall_clock(tmp_path):
    final_snapshots_dir = tmp_path / "snapshots"
    final_date_dir = final_snapshots_dir / "2026-06-19"
    staging_date_dir = tmp_path / "staging" / "2026-06-19"
    final_date_dir.mkdir(parents=True)
    staging_date_dir.mkdir(parents=True)
    (final_date_dir / "rankings.csv").write_text("ticker,rank\nOLD,1\n", encoding="utf-8")
    (staging_date_dir / "rankings.csv").write_text("ticker,rank\nNEW,1\n", encoding="utf-8")

    promote_snapshot(staging_date_dir, final_snapshots_dir, "2026-06-19")

    backups = [p.name for p in final_snapshots_dir.iterdir() if "__pre_" in p.name]
    assert len(backups) == 1
    assert not re.search(r"__pre_\d{8}T\d{6}Z$", backups[0])
