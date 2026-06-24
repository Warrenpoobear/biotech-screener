"""Tests for tools/herald_recovery.py."""

from __future__ import annotations

from unittest.mock import patch

from tools import herald_recovery as hr


def test_plan_recovery_minimal_when_classified_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(hr, "REPO", tmp_path)
    pr = tmp_path / "data" / "press_releases"
    (pr / "deduped").mkdir(parents=True)
    (pr / "classified").mkdir(parents=True)
    (pr / "releases_2026-06-24.jsonl").write_text("{}\n")
    (pr / "deduped" / "deduped_2026-06-24.jsonl").write_text("{}\n")

    steps = hr.plan_recovery_steps("2026-06-24")
    assert steps == ["classify"]


def test_plan_recovery_full_includes_fetch_dedupe_classify():
    steps = hr.plan_recovery_steps("2026-06-24", full=True)
    assert steps == ["fetch", "dedupe", "classify"]


def test_plan_recovery_from_report_when_not_done():
    report = {"herald_done": False, "as_of_date": "2026-06-24"}
    steps = hr.plan_recovery_steps("2026-06-24", pre_report=report)
    assert "classify" in steps
    assert steps[0] == "fetch"


def test_run_recovery_dry_run_prints_steps(capsys):
    with patch.object(hr, "plan_recovery_steps", return_value=["fetch", "classify"]):
        rc = hr.run_recovery("2026-06-24", dry_run=True, full=True)
    assert rc == 0
    out = capsys.readouterr().out
    assert "fetch" in out
    assert "DRY_RUN" in out


def test_run_step_dry_run():
    rc, detail = hr.run_step("fetch", "2026-06-24", dry_run=True)
    assert rc == 0
    assert "fetch_company_press_releases" in detail
