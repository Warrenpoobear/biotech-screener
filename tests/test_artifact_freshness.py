"""Tests for tools/artifact_freshness.py."""

from __future__ import annotations

import os
import time
from datetime import date
from pathlib import Path

from tools.artifact_freshness import age_days, newest_artifact_freshness, newest_content_date_under, parse_dates_in_name


def test_parse_dates_in_name_extracts_iso_dates():
    assert parse_dates_in_name(Path("digest_2026-05-01_evening.json")) == [date(2026, 5, 1)]


def test_newest_content_date_under_prefers_latest_filename_date(tmp_path: Path):
    d = tmp_path / "artifacts"
    d.mkdir()
    (d / "old_2026-03-01.json").write_text("{}")
    (d / "new_2026-05-08.json").write_text("{}")
    latest, path = newest_content_date_under(d)
    assert latest == date(2026, 5, 8)
    assert path.name == "new_2026-05-08.json"


def test_newest_artifact_freshness_prefers_content_date_over_mtime(tmp_path: Path):
    art = tmp_path / "artifacts" / "ops_digest"
    art.mkdir(parents=True)
    stale = art / "2026-03-01_digest.json"
    stale.write_text("{}")
    # Fresh mtime but old embedded date
    os.utime(stale, (time.time(), time.time()))

    newest, sample, method = newest_artifact_freshness(tmp_path, ["artifacts/ops_digest/"])
    assert newest == date(2026, 3, 1)
    assert method == "content_date"
    assert sample == stale


def test_newest_artifact_freshness_falls_back_to_mtime(tmp_path: Path):
    art = tmp_path / "artifacts" / "event_analyst"
    art.mkdir(parents=True)
    f = art / "notes.txt"
    f.write_text("no date in name")
    os.utime(f, (time.time(), time.time()))

    newest, _, method = newest_artifact_freshness(tmp_path, ["artifacts/event_analyst/"])
    assert newest == date.today()
    assert method == "mtime"


def test_age_days():
    assert age_days(date(2026, 5, 8), date(2026, 5, 1)) == 7
