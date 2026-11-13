"""Tests for regenerate_pit_v2_snapshots.py hardening.

Classification: PIT_V2_REGENERATION_FALSE_SUCCESS_HARDENING_NO_MODEL_CHANGE

These tests verify that:
- exit 0 without rankings.csv is classified as failed_false_success
- partial snapshot dirs are detected correctly
- --clean-partial removes only incomplete dirs
- complete dirs are not removed by clean_partial
- force_overwrite passes the flag through to the command
- allow_weekend passes the flag through
- manifest is written with per-date statuses
- nonzero exit when any date fails
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "research"
sys.path.insert(0, str(SCRIPTS_DIR.parent.parent))

from scripts.research.regenerate_pit_v2_snapshots import (
    _clean_partial_dir,
    _snapshot_is_partial,
    _write_manifest,
    run_one,
)


def _make_snap_dir(tmp_path: Path, date_str: str, complete: bool = True) -> Path:
    """Create a snapshot dir, optionally with rankings.csv."""
    d = tmp_path / date_str
    d.mkdir(parents=True)
    (d / "screen_output.json").write_text("{}")
    if complete:
        (d / "rankings.csv").write_text("ticker,rank\nCOGT,1\n")
    return d


# ---------------------------------------------------------------------------
# _snapshot_is_partial
# ---------------------------------------------------------------------------


def test_partial_dir_detected_when_no_rankings(tmp_path):
    _make_snap_dir(tmp_path, "2026-05-29", complete=False)
    assert _snapshot_is_partial("2026-05-29", tmp_path) is True


def test_partial_dir_not_flagged_when_rankings_present(tmp_path):
    _make_snap_dir(tmp_path, "2026-05-29", complete=True)
    assert _snapshot_is_partial("2026-05-29", tmp_path) is False


def test_partial_dir_not_flagged_when_dir_absent(tmp_path):
    assert _snapshot_is_partial("2026-05-29", tmp_path) is False


# ---------------------------------------------------------------------------
# _clean_partial_dir
# ---------------------------------------------------------------------------


def test_clean_partial_removes_incomplete_dir(tmp_path):
    _make_snap_dir(tmp_path, "2026-05-29", complete=False)
    removed = _clean_partial_dir("2026-05-29", tmp_path)
    assert removed is True
    assert not (tmp_path / "2026-05-29").exists()


def test_clean_partial_does_not_remove_complete_dir(tmp_path):
    _make_snap_dir(tmp_path, "2026-05-29", complete=True)
    removed = _clean_partial_dir("2026-05-29", tmp_path)
    assert removed is False
    assert (tmp_path / "2026-05-29" / "rankings.csv").exists()


def test_clean_partial_no_op_when_dir_absent(tmp_path):
    removed = _clean_partial_dir("2026-05-29", tmp_path)
    assert removed is False


# ---------------------------------------------------------------------------
# run_one — false success detection
# ---------------------------------------------------------------------------


def _mock_subprocess_zero_no_file(cmd, **kwargs):
    """Simulates run_screen.py: exits 0 but writes no rankings.csv."""
    result = MagicMock()
    result.returncode = 0
    result.stdout = "some output\n"
    result.stderr = "! Optional file missing: catalyst_events.json\n"
    return result


def test_exit_zero_without_rankings_is_failed_false_success(tmp_path):
    with patch(
        "scripts.research.regenerate_pit_v2_snapshots.subprocess.run", side_effect=_mock_subprocess_zero_no_file
    ):
        with patch(
            "scripts.research.regenerate_pit_v2_snapshots._resolve_data_dir", return_value=(tmp_path, "current")
        ):
            with patch(
                "scripts.research.regenerate_pit_v2_snapshots._resolve_institutional_source",
                return_value="contaminated",
            ):
                r = run_one("2026-05-29", out_dir=tmp_path, stage_pit_institutional=False)
    assert r["status"] == "failed_false_success"
    assert r["returncode"] == 0
    assert r["rankings_csv_exists"] is False
    assert "hint" in r


def _mock_subprocess_zero_with_file(out_dir: Path):
    """Simulates run_screen.py: exits 0 and writes rankings.csv."""

    def _run(cmd, **kwargs):
        date = None
        for i, arg in enumerate(cmd):
            if arg == "--as-of-date" and i + 1 < len(cmd):
                date = cmd[i + 1]
        if date:
            snap = out_dir / date
            snap.mkdir(parents=True, exist_ok=True)
            (snap / "rankings.csv").write_text("ticker,rank\nCOGT,1\n")
        result = MagicMock()
        result.returncode = 0
        result.stdout = "done\n"
        result.stderr = ""
        return result

    return _run


def test_exit_zero_with_rankings_is_ok(tmp_path):
    with patch(
        "scripts.research.regenerate_pit_v2_snapshots.subprocess.run",
        side_effect=_mock_subprocess_zero_with_file(tmp_path),
    ):
        with patch(
            "scripts.research.regenerate_pit_v2_snapshots._resolve_data_dir", return_value=(tmp_path, "current")
        ):
            with patch(
                "scripts.research.regenerate_pit_v2_snapshots._resolve_institutional_source",
                return_value="contaminated",
            ):
                r = run_one("2026-05-29", out_dir=tmp_path, stage_pit_institutional=False)
    assert r["status"] == "ok"
    assert r["rankings_csv_exists"] is True


def test_nonzero_returncode_is_error(tmp_path):
    def _run(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 1
        result.stdout = ""
        result.stderr = "fatal error\n"
        return result

    with patch("scripts.research.regenerate_pit_v2_snapshots.subprocess.run", side_effect=_run):
        with patch(
            "scripts.research.regenerate_pit_v2_snapshots._resolve_data_dir", return_value=(tmp_path, "current")
        ):
            with patch(
                "scripts.research.regenerate_pit_v2_snapshots._resolve_institutional_source",
                return_value="contaminated",
            ):
                r = run_one("2026-05-29", out_dir=tmp_path, stage_pit_institutional=False)
    assert r["status"] == "error"
    assert r["returncode"] == 1
    assert r["rankings_csv_exists"] is False


# ---------------------------------------------------------------------------
# run_one — flag passthrough
# ---------------------------------------------------------------------------


def _capture_cmd(tmp_path: Path):
    """Returns (mock_fn, captured_cmds list)."""
    captured = []

    def _run(cmd, **kwargs):
        captured.append(list(cmd))
        snap = None
        for i, arg in enumerate(cmd):
            if arg == "--as-of-date" and i + 1 < len(cmd):
                snap_dir_arg = None
                for j, a in enumerate(cmd):
                    if a == "--snapshot-dir" and j + 1 < len(cmd):
                        snap_dir_arg = Path(cmd[j + 1])
                date = cmd[i + 1]
                if snap_dir_arg:
                    d = snap_dir_arg / date
                    d.mkdir(parents=True, exist_ok=True)
                    (d / "rankings.csv").write_text("ticker,rank\nCOGT,1\n")
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    return _run, captured


def test_force_overwrite_passed_to_command(tmp_path):
    mock_run, captured = _capture_cmd(tmp_path)
    with patch("scripts.research.regenerate_pit_v2_snapshots.subprocess.run", side_effect=mock_run):
        with patch(
            "scripts.research.regenerate_pit_v2_snapshots._resolve_data_dir", return_value=(tmp_path, "current")
        ):
            with patch(
                "scripts.research.regenerate_pit_v2_snapshots._resolve_institutional_source",
                return_value="contaminated",
            ):
                run_one("2026-05-29", out_dir=tmp_path, stage_pit_institutional=False, force_overwrite=True)
    assert len(captured) == 1
    assert "--force-overwrite" in captured[0]


def test_force_overwrite_not_passed_when_false(tmp_path):
    mock_run, captured = _capture_cmd(tmp_path)
    with patch("scripts.research.regenerate_pit_v2_snapshots.subprocess.run", side_effect=mock_run):
        with patch(
            "scripts.research.regenerate_pit_v2_snapshots._resolve_data_dir", return_value=(tmp_path, "current")
        ):
            with patch(
                "scripts.research.regenerate_pit_v2_snapshots._resolve_institutional_source",
                return_value="contaminated",
            ):
                run_one("2026-05-29", out_dir=tmp_path, stage_pit_institutional=False, force_overwrite=False)
    assert "--force-overwrite" not in captured[0]


def test_allow_weekend_passed_to_command(tmp_path):
    mock_run, captured = _capture_cmd(tmp_path)
    with patch("scripts.research.regenerate_pit_v2_snapshots.subprocess.run", side_effect=mock_run):
        with patch(
            "scripts.research.regenerate_pit_v2_snapshots._resolve_data_dir", return_value=(tmp_path, "current")
        ):
            with patch(
                "scripts.research.regenerate_pit_v2_snapshots._resolve_institutional_source",
                return_value="contaminated",
            ):
                run_one("2026-02-28", out_dir=tmp_path, stage_pit_institutional=False, allow_weekend=True)
    assert "--allow-weekend" in captured[0]


# ---------------------------------------------------------------------------
# _write_manifest
# ---------------------------------------------------------------------------


def test_manifest_written_with_per_date_statuses(tmp_path):
    results = [
        {"date": "2026-04-30", "status": "ok", "rankings_csv_exists": True},
        {"date": "2026-05-29", "status": "failed_false_success", "rankings_csv_exists": False},
    ]
    path = _write_manifest(results, tmp_path)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["schema"] == "pit_v2_regen_manifest.v1"
    assert len(data["results"]) == 2
    statuses = {r["date"]: r["status"] for r in data["results"]}
    assert statuses["2026-04-30"] == "ok"
    assert statuses["2026-05-29"] == "failed_false_success"


def test_manifest_dir_created_if_absent(tmp_path):
    results = [{"date": "2026-06-26", "status": "ok", "rankings_csv_exists": True}]
    nested = tmp_path / "deep" / "nested" / "dir"
    path = _write_manifest(results, nested)
    assert path.exists()


# ---------------------------------------------------------------------------
# No production artifacts modified
# ---------------------------------------------------------------------------


def test_no_production_files_in_scope():
    """Confirm the regen script does not import or reference frozen production modules."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "regen",
        SCRIPTS_DIR / "regenerate_pit_v2_snapshots.py",
    )
    mod = importlib.util.module_from_spec(spec)
    src = (SCRIPTS_DIR / "regenerate_pit_v2_snapshots.py").read_text()
    forbidden = ["ranker_v2_pairwise", "selector_engine", "final_score", "composite_score"]
    for token in forbidden:
        assert token not in src, f"Production module reference found: {token}"


# ---------------------------------------------------------------------------
# 2026-11-13: point-in-time market data, side-output isolation, live-state guard
# ---------------------------------------------------------------------------

import scripts.research.regenerate_pit_v2_snapshots as regen  # noqa: E402


def _ok_run(tmp_path, date_str):
    def _fake(cmd, **kwargs):
        d = tmp_path / "out" / date_str
        d.mkdir(parents=True, exist_ok=True)
        (d / "rankings.csv").write_text("ticker,rank\nCOGT,1\n")
        return MagicMock(returncode=0, stdout="", stderr="")

    return _fake


def _fake_checkout(tmp_path, monkeypatch):
    """Point the script at a throwaway checkout with a minimal production_data."""
    root = tmp_path / "repo"
    prod = root / "production_data"
    prod.mkdir(parents=True)
    (prod / "universe.json").write_text("[]")
    (prod / "financial_records.json").write_text("[]")
    (prod / "market_data.json").write_text('[{"ticker": "AAA", "market_cap": 999}]')
    monkeypatch.setattr(regen, "PROJECT_ROOT", root)
    monkeypatch.setattr(regen, "PROD_DATA", prod)
    monkeypatch.setattr(regen, "STAGING_ROOT", root / "data" / "staging" / "pit_regen")
    monkeypatch.setattr(regen, "SNAPSHOTS_DIR", root / "data" / "snapshots")
    return root, prod


def test_pit_market_data_is_staged_and_used_instead_of_todays_file(tmp_path, monkeypatch):
    root, prod = _fake_checkout(tmp_path, monkeypatch)

    def _stage(date_str, base):
        staging = regen.STAGING_ROOT / date_str
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "market_data.json").write_text('[{"ticker": "AAA", "market_cap": 111}]')
        return staging, {"pit_shares_x_close": 1}

    monkeypatch.setattr(regen, "stage_pit_market_data", _stage)
    seen = {}

    def _fake(cmd, **kwargs):
        seen["data_dir"] = Path(cmd[cmd.index("--data-dir") + 1])
        return _ok_run(tmp_path, "2022-06-30")(cmd, **kwargs)

    with patch("subprocess.run", side_effect=_fake):
        r = regen.run_one("2022-06-30", out_dir=tmp_path / "out", pit_market_data=True)

    assert r["status"] == "ok"
    assert r["market_data_source"] == "pit_market_data"
    assert r["market_cap_sources"] == {"pit_shares_x_close": 1}
    assert seen["data_dir"] == regen.STAGING_ROOT / "2022-06-30"
    staged = json.loads((seen["data_dir"] / "market_data.json").read_text())
    assert staged[0]["market_cap"] == 111  # not today's 999
    assert (seen["data_dir"] / "universe.json").exists()  # everything else is shadowed in
    assert json.loads((prod / "market_data.json").read_text())[0]["market_cap"] == 999  # source untouched


def test_default_run_one_reports_current_market_data(tmp_path, monkeypatch):
    _fake_checkout(tmp_path, monkeypatch)
    with patch("subprocess.run", side_effect=_ok_run(tmp_path, "2022-06-30")):
        r = regen.run_one("2022-06-30", out_dir=tmp_path / "out")
    assert r["market_data_source"] == "current_market_data_json"


def test_pit_market_data_failure_fails_the_date_without_running(tmp_path, monkeypatch):
    _fake_checkout(tmp_path, monkeypatch)

    def _boom(date_str, base):
        raise FileNotFoundError("price_history.csv")

    monkeypatch.setattr(regen, "stage_pit_market_data", _boom)
    with patch("subprocess.run") as run:
        r = regen.run_one("2022-06-30", out_dir=tmp_path / "out", pit_market_data=True)
    assert r["status"] == "error"
    assert "point-in-time market data" in r["error_tail"][0]
    run.assert_not_called()


def test_side_outputs_are_sent_under_the_out_dir(tmp_path, monkeypatch):
    _fake_checkout(tmp_path, monkeypatch)
    seen = {}

    def _fake(cmd, **kwargs):
        seen["cmd"] = cmd
        return _ok_run(tmp_path, "2022-06-30")(cmd, **kwargs)

    with patch("subprocess.run", side_effect=_fake):
        regen.run_one("2022-06-30", out_dir=tmp_path / "out", isolate_side_outputs=True)
    side = Path(seen["cmd"][seen["cmd"].index("--output-dir") + 1])
    assert side == tmp_path / "out" / "_side_outputs" / "2022-06-30"
    assert side.is_dir()


def test_side_outputs_flag_absent_by_default(tmp_path, monkeypatch):
    _fake_checkout(tmp_path, monkeypatch)
    seen = {}

    def _fake(cmd, **kwargs):
        seen["cmd"] = cmd
        return _ok_run(tmp_path, "2022-06-30")(cmd, **kwargs)

    with patch("subprocess.run", side_effect=_fake):
        regen.run_one("2022-06-30", out_dir=tmp_path / "out")
    assert "--output-dir" not in seen["cmd"]


def test_live_state_guard_restores_changed_removes_new_and_recreates_deleted(tmp_path):
    root = tmp_path / "repo"
    (root / "data" / "state").mkdir(parents=True)
    streak = root / "data" / "state" / "blind_spot_streak.json"
    streak.write_text('{"AARD": 1}')
    carry = root / "data" / "state" / "hard_catalyst_carry.json"
    carry.write_text('{"AQST": {}}')
    log = root / "data" / "expression_decision_log.jsonl"
    log.write_text('{"as_of_date": "2026-11-12"}\n')

    with regen.LiveStateGuard(root) as guard:
        streak.write_text("{}")  # rewritten
        with open(log, "a") as f:  # appended
            f.write('{"as_of_date": "2020-01-31"}\n')
        carry.unlink()  # deleted
        (root / "data" / "state" / "new_state.json").write_text("{}")  # created

    assert streak.read_text() == '{"AARD": 1}'
    assert log.read_text() == '{"as_of_date": "2026-11-12"}\n'
    assert carry.read_text() == '{"AQST": {}}'
    assert not (root / "data" / "state" / "new_state.json").exists()
    assert sorted(guard.restored) == sorted(
        [
            "data/state/blind_spot_streak.json",
            "data/expression_decision_log.jsonl",
            "data/state/hard_catalyst_carry.json",
            "data/state/new_state.json",
        ]
    )


def test_live_state_guard_leaves_untouched_files_alone(tmp_path):
    root = tmp_path / "repo"
    (root / "data" / "state").mkdir(parents=True)
    f = root / "data" / "state" / "blind_spot_streak.json"
    f.write_text("{}")
    before = f.stat().st_mtime_ns
    with regen.LiveStateGuard(root) as guard:
        pass
    assert guard.restored == []
    assert f.stat().st_mtime_ns == before


def test_run_one_restores_live_state_written_by_the_screen(tmp_path, monkeypatch):
    root, _ = _fake_checkout(tmp_path, monkeypatch)
    (root / "data" / "state").mkdir(parents=True)
    streak = root / "data" / "state" / "blind_spot_streak.json"
    streak.write_text('{"AARD": 1}')

    def _fake(cmd, **kwargs):
        streak.write_text("{}")  # what a historical run_screen.py does
        return _ok_run(tmp_path, "2022-06-30")(cmd, **kwargs)

    with patch("subprocess.run", side_effect=_fake):
        r = regen.run_one("2022-06-30", out_dir=tmp_path / "out", protect_live_state=True)
    assert streak.read_text() == '{"AARD": 1}'
    assert r["live_state_restored"] == ["data/state/blind_spot_streak.json"]


def test_leftover_production_lock_file_is_not_a_run(tmp_path):
    root = tmp_path / "repo"
    (root / "logs").mkdir(parents=True)
    assert regen.production_lock_held(root) is None
    (root / "logs" / ".daily_production_py.lock").write_text("")  # flock-based; the file is never removed
    assert regen.production_lock_held(root) is None


def test_live_production_run_is_detected(tmp_path):
    import subprocess

    root = tmp_path / "repo"
    (root / "logs").mkdir(parents=True)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        (root / "logs" / ".daily_production.lock").write_text(str(child.pid))
        assert regen.production_lock_held(root) == "logs/.daily_production.lock"
    finally:
        child.kill()
        child.wait()


def test_archived_inputs_without_financial_records_are_not_used(tmp_path, monkeypatch):
    root, prod = _fake_checkout(tmp_path, monkeypatch)
    archived = root / "data" / "snapshots" / "2026-06-26" / "inputs"
    archived.mkdir(parents=True)
    (archived / "universe.json").write_text("[]")
    assert regen._resolve_data_dir("2026-06-26") == (prod, "current")
    (archived / "financial_records.json").write_text("[]")
    assert regen._resolve_data_dir("2026-06-26") == (archived, "archived")


def test_fresh_staging_removes_stale_links(tmp_path, monkeypatch):
    root, prod = _fake_checkout(tmp_path, monkeypatch)
    stale = regen.STAGING_ROOT / "2022-06-30"
    stale.mkdir(parents=True)
    (stale / "financial_records.json").write_text('["stale"]')
    monkeypatch.setattr(
        regen,
        "stage_pit_market_data",
        lambda d, b: ((regen.STAGING_ROOT / d).mkdir(parents=True, exist_ok=True), (regen.STAGING_ROOT / d, {}))[1],
    )
    with patch("subprocess.run", side_effect=_ok_run(tmp_path, "2022-06-30")):
        regen.run_one("2022-06-30", out_dir=tmp_path / "out", pit_market_data=True, fresh_staging=True)
    assert json.loads((stale / "financial_records.json").read_text()) == []  # relinked from production_data
