"""Tests for tools/herald_health_check.py."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch

from tools import herald_health_check as hc

REPO = Path(__file__).resolve().parent.parent


def test_herald_done_requires_both_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, "PR_DIR", tmp_path)
    monkeypatch.setattr(hc, "DEDUPED_DIR", tmp_path / "deduped")
    monkeypatch.setattr(hc, "CLASSIFIED_DIR", tmp_path / "classified")
    hc.DEDUPED_DIR.mkdir(parents=True)
    hc.CLASSIFIED_DIR.mkdir(parents=True)

    ds = "2026-06-24"
    assert hc.herald_done(ds) is False

    (hc.DEDUPED_DIR / f"deduped_{ds}.jsonl").write_text("{}\n", encoding="utf-8")
    assert hc.herald_done(ds) is False

    (hc.CLASSIFIED_DIR / f"classified_{ds}.jsonl").write_text("{}\n", encoding="utf-8")
    assert hc.herald_done(ds) is True


def test_run_check_healthy_when_done_and_fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, "PR_DIR", tmp_path)
    monkeypatch.setattr(hc, "DEDUPED_DIR", tmp_path / "deduped")
    monkeypatch.setattr(hc, "CLASSIFIED_DIR", tmp_path / "classified")
    monkeypatch.setattr(hc, "DIGEST_DIR", tmp_path / "digests")
    monkeypatch.setattr(hc, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(hc, "STATE_PATH", tmp_path / "fetch_state.json")
    hc.DEDUPED_DIR.mkdir(parents=True)
    hc.CLASSIFIED_DIR.mkdir(parents=True)
    hc.DIGEST_DIR.mkdir(parents=True)

    ds = date(2026, 6, 24)
    (hc.DEDUPED_DIR / "deduped_2026-06-24.jsonl").write_text("{}\n", encoding="utf-8")
    (hc.CLASSIFIED_DIR / "classified_2026-06-24.jsonl").write_text("{}\n", encoding="utf-8")
    (hc.DIGEST_DIR / "biotech_news_digest_2026-06-23_morning.json").write_text("{}", encoding="utf-8")
    hc.STATE_PATH.write_text("{}", encoding="utf-8")

    report = hc.run_check(ds)
    assert report["herald_done"] is True
    assert report["verdict"] in ("HEALTHY", "WARN")


def test_run_check_fail_when_dark(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, "PR_DIR", tmp_path)
    monkeypatch.setattr(hc, "DEDUPED_DIR", tmp_path / "deduped")
    monkeypatch.setattr(hc, "CLASSIFIED_DIR", tmp_path / "classified")
    monkeypatch.setattr(hc, "DIGEST_DIR", tmp_path / "digests")
    monkeypatch.setattr(hc, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(hc, "STATE_PATH", tmp_path / "fetch_state.json")
    hc.CLASSIFIED_DIR.mkdir(parents=True)

    (hc.CLASSIFIED_DIR / "classified_2026-04-09.jsonl").write_text("{}\n", encoding="utf-8")

    report = hc.run_check(date(2026, 6, 24))
    assert report["verdict"] == "FAIL"
    assert "STALE_SOURCE" in report["status_codes"]
    assert hc._exit_code(report["verdict"]) == 2


def test_fetch_imports_without_pythonpath():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "tools/fetch_company_press_releases.py", "--health-check"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "Source Health Check" in result.stdout


def test_classify_deduped_stem_maps_to_classified_date():
    stem = "deduped_2026-06-24"
    out_stem = stem.replace("deduped_", "classified_", 1)
    assert out_stem == "classified_2026-06-24"


def test_main_writes_json_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, "PR_DIR", tmp_path)
    monkeypatch.setattr(hc, "DEDUPED_DIR", tmp_path / "deduped")
    monkeypatch.setattr(hc, "CLASSIFIED_DIR", tmp_path / "classified")
    monkeypatch.setattr(hc, "DIGEST_DIR", tmp_path / "digests")
    monkeypatch.setattr(hc, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(hc, "STATE_PATH", tmp_path / "fetch_state.json")
    hc.CLASSIFIED_DIR.mkdir(parents=True)
    (hc.CLASSIFIED_DIR / "classified_2026-06-24.jsonl").write_text("{}\n", encoding="utf-8")
    hc.DEDUPED_DIR.mkdir(parents=True)
    (hc.DEDUPED_DIR / "deduped_2026-06-24.jsonl").write_text("{}\n", encoding="utf-8")

    with patch.object(sys, "argv", ["herald_health_check.py", "--as-of-date", "2026-06-24"]):
        hc.main()

    out = tmp_path / "out" / "health_check_2026-06-24.json"
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema"] == "herald_health_check.v1"


def test_herald_health_check_recover_dry_run(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, "PR_DIR", tmp_path)
    monkeypatch.setattr(hc, "DEDUPED_DIR", tmp_path / "deduped")
    monkeypatch.setattr(hc, "CLASSIFIED_DIR", tmp_path / "classified")
    monkeypatch.setattr(hc, "DIGEST_DIR", tmp_path / "digests")
    monkeypatch.setattr(hc, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(hc, "STATE_PATH", tmp_path / "fetch_state.json")
    hc.CLASSIFIED_DIR.mkdir(parents=True)
    (hc.CLASSIFIED_DIR / "classified_2026-04-09.jsonl").write_text("{}\n", encoding="utf-8")

    calls = []

    def _fake_recovery(as_of, *, dry_run=False, full=False, include_digest=False, pre_report=None):
        calls.append({"as_of": as_of, "dry_run": dry_run, "pre_report": pre_report})
        return 0

    monkeypatch.setattr("tools.herald_recovery.run_recovery", _fake_recovery)

    with patch.object(
        sys,
        "argv",
        ["herald_health_check.py", "--as-of-date", "2026-06-24", "--recover", "--dry-run-recover", "--no-write"],
    ):
        assert hc.main() == 0

    assert len(calls) == 1
    assert calls[0]["dry_run"] is True
    assert calls[0]["pre_report"]["verdict"] == "FAIL"


def test_herald_health_check_attaches_outcome_verdict(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, "PR_DIR", tmp_path)
    monkeypatch.setattr(hc, "DEDUPED_DIR", tmp_path / "deduped")
    monkeypatch.setattr(hc, "CLASSIFIED_DIR", tmp_path / "classified")
    monkeypatch.setattr(hc, "DIGEST_DIR", tmp_path / "digests")
    monkeypatch.setattr(hc, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(hc, "STATE_PATH", tmp_path / "fetch_state.json")
    hc.CLASSIFIED_DIR.mkdir(parents=True)
    (hc.CLASSIFIED_DIR / "classified_2026-06-24.jsonl").write_text("{}\n", encoding="utf-8")
    hc.DEDUPED_DIR.mkdir(parents=True)
    (hc.DEDUPED_DIR / "deduped_2026-06-24.jsonl").write_text("{}\n", encoding="utf-8")

    calls = []

    def _fake_attach(exec_id, was_correct, evidence, environment="prod"):
        calls.append((exec_id, was_correct, evidence))

    def _fake_log(*_a, **_k):
        return "exec-test-herald"

    monkeypatch.setattr("tools.record_skill_feedback.attach_outcome_verdict", _fake_attach)
    monkeypatch.setattr("tools.agent_skill_telemetry.log_agent_run", _fake_log)

    with patch.object(sys, "argv", ["herald_health_check.py", "--as-of-date", "2026-06-24", "--no-write"]):
        assert hc.main() == 1  # WARN exit code

    assert calls
    assert calls[0][0] == "exec-test-herald"
    assert calls[0][1] is False  # WARN verdict despite herald_done
