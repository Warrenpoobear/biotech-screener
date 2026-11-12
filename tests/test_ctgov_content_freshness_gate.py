"""Tests for the ctgov_content_freshness gate in run_daily_production.py.

source_freshness ages CT.gov by the *filename date* of the daily cache copy,
which warm_caches rewrites every day even when production_data/trial_records.json
is never refreshed. From 2026-08-07 to 2026-11-03 the newest trial update in the
file stayed at 2026-09-14 and diff-based catalyst detection saw zero events.
This gate ages the *content*: the newest last_update_posted on or before as_of.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from run_daily_production import check_ctgov_content_freshness  # noqa: E402


def _write(path: Path, dates: list[str]) -> Path:
    path.write_text(
        json.dumps([{"nct_id": f"NCT{i:08d}", "last_update_posted": d} for i, d in enumerate(dates)]),
        encoding="utf-8",
    )
    return path


def test_fresh_content_passes(tmp_path):
    p = _write(tmp_path / "trial_records.json", ["2026-10-01", "2026-11-02"])
    g = check_ctgov_content_freshness(p, "2026-11-03")
    assert g.name == "ctgov_content_freshness"
    assert g.status == "PASS"
    assert g.value["newest_last_update_posted"] == "2026-11-02"
    assert g.value["age_days"] == 1


def test_stale_content_fails(tmp_path):
    # The 2026-11-03 production state: newest update 2026-09-14, 50 days old.
    p = _write(tmp_path / "trial_records.json", ["2026-08-01", "2026-09-14"])
    g = check_ctgov_content_freshness(p, "2026-11-03")
    assert g.status == "FAIL"
    assert g.value["age_days"] == 50
    assert "2026-09-14" in g.detail


def test_boundary_is_inclusive(tmp_path):
    p = _write(tmp_path / "trial_records.json", ["2026-10-27"])
    assert check_ctgov_content_freshness(p, "2026-11-03", max_age_days=7).status == "PASS"
    assert check_ctgov_content_freshness(p, "2026-11-04", max_age_days=7).status == "FAIL"


def test_future_dates_ignored_for_pit(tmp_path):
    # A historical rerun must not treat updates after as_of as evidence of freshness.
    p = _write(tmp_path / "trial_records.json", ["2026-09-14", "2026-11-03"])
    g = check_ctgov_content_freshness(p, "2026-10-01")
    assert g.value["newest_last_update_posted"] == "2026-09-14"
    assert g.status == "FAIL"


def test_blank_and_malformed_dates_ignored(tmp_path):
    p = _write(tmp_path / "trial_records.json", ["", "not-a-date", "2026-11-01"])
    assert check_ctgov_content_freshness(p, "2026-11-03").status == "PASS"


def test_missing_file_fails(tmp_path):
    g = check_ctgov_content_freshness(tmp_path / "trial_records.json", "2026-11-03")
    assert g.status == "FAIL"


def test_no_dates_fails(tmp_path):
    p = _write(tmp_path / "trial_records.json", ["", ""])
    assert check_ctgov_content_freshness(p, "2026-11-03").status == "FAIL"
