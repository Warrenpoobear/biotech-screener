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
