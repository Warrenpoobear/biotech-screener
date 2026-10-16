"""Tests for build_pit_financials.is_fresh (scheduled refresh of stale PIT records)."""

import importlib.util
import os
import time
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "build_pit_financials", Path(__file__).resolve().parents[1] / "tools" / "build_pit_financials.py"
)
bpf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bpf)

DAY = 86400


@pytest.fixture
def record(tmp_path):
    p = tmp_path / "ALNY.json"
    p.write_text("{}")
    return p


def _age(path, days, now):
    os.utime(path, (now - days * DAY, now - days * DAY))


def test_missing_file_is_never_fresh(tmp_path):
    assert not bpf.is_fresh(tmp_path / "NOPE.json", None)
    assert not bpf.is_fresh(tmp_path / "NOPE.json", 7)


def test_no_max_age_keeps_any_existing_file(record):
    now = time.time()
    _age(record, 400, now)
    assert bpf.is_fresh(record, None, now=now)


def test_recent_file_is_fresh(record):
    now = time.time()
    _age(record, 2, now)
    assert bpf.is_fresh(record, 7, now=now)


def test_old_file_is_stale(record):
    now = time.time()
    _age(record, 116, now)  # the 2026-06-22 -> 2026-10-16 staleness that motivated this
    assert not bpf.is_fresh(record, 7, now=now)
