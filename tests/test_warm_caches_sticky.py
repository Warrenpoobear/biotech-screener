"""Tests for sticky cache refresh — staging + validation logic in warm_caches.py."""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from warm_caches import (
    validate_cache_refresh,
    warm_sec_8k,
    warm_ctgov,
    _find_prior_sec8k_count,
    _find_prior_ctgov_count,
)


# ── validate_cache_refresh: SEC 8-K ─────────────────────────────────


def test_sec_8k_zero_rejected():
    accepted, reason = validate_cache_refresh("sec_8k", 0, 100)
    assert not accepted
    assert reason == "empty_refresh"


def test_sec_8k_positive_no_prior_accepted():
    accepted, reason = validate_cache_refresh("sec_8k", 50, None)
    assert accepted
    assert reason == ""


def test_sec_8k_collapse_rejected():
    # 20/100 = 0.20 < 0.30 threshold
    accepted, reason = validate_cache_refresh("sec_8k", 20, 100)
    assert not accepted
    assert "collapse" in reason
    assert "0.20" in reason


def test_sec_8k_moderate_drop_accepted():
    # 50/100 = 0.50 >= 0.30 threshold
    accepted, reason = validate_cache_refresh("sec_8k", 50, 100)
    assert accepted
    assert reason == ""


def test_sec_8k_growth_accepted():
    accepted, reason = validate_cache_refresh("sec_8k", 200, 100)
    assert accepted
    assert reason == ""


def test_sec_8k_prior_zero_accepted():
    # prior=0 → skip ratio check
    accepted, reason = validate_cache_refresh("sec_8k", 50, 0)
    assert accepted
    assert reason == ""


# ── validate_cache_refresh: CTGov ────────────────────────────────────


def test_ctgov_zero_rejected():
    accepted, reason = validate_cache_refresh("ctgov", 0, 1000)
    assert not accepted
    assert reason == "empty_refresh"


def test_ctgov_within_band_accepted():
    # 900/1000 = 0.90 — within [0.60, 1.50]
    accepted, reason = validate_cache_refresh("ctgov", 900, 1000)
    assert accepted
    assert reason == ""


def test_ctgov_spike_rejected():
    # 1600/1000 = 1.60 > 1.50
    accepted, reason = validate_cache_refresh("ctgov", 1600, 1000)
    assert not accepted
    assert "out_of_band" in reason
    assert "1.60" in reason


def test_ctgov_crash_rejected():
    # 500/1000 = 0.50 < 0.60
    accepted, reason = validate_cache_refresh("ctgov", 500, 1000)
    assert not accepted
    assert "out_of_band" in reason
    assert "0.50" in reason


def test_ctgov_no_prior_accepted():
    accepted, reason = validate_cache_refresh("ctgov", 1000, None)
    assert accepted
    assert reason == ""


def test_ctgov_prior_zero_accepted():
    accepted, reason = validate_cache_refresh("ctgov", 1000, 0)
    assert accepted
    assert reason == ""


# ── validate_cache_refresh: unknown source ───────────────────────────


def test_unknown_source_always_accepted():
    accepted, reason = validate_cache_refresh("fda_adcom", 0, 100)
    assert accepted
    assert reason == ""


# ── _find_prior helpers ──────────────────────────────────────────────


def test_find_prior_sec8k_count(tmp_path):
    events = [{"ticker": "A", "event_type": "X", "event_date": "2026-01-01"}] * 42
    (tmp_path / "8k_catalysts_2026-02-01_abcd1234.json").write_text(json.dumps(events))
    assert _find_prior_sec8k_count(tmp_path) == 42


def test_find_prior_sec8k_count_no_files(tmp_path):
    assert _find_prior_sec8k_count(tmp_path) is None


def test_find_prior_ctgov_count(tmp_path):
    records = [{"nct_id": f"NCT{i}"} for i in range(100)]
    (tmp_path / "trial_records_2026-02-01.json").write_text(json.dumps(records))
    assert _find_prior_ctgov_count(tmp_path) == 100


def test_find_prior_ctgov_count_no_files(tmp_path):
    assert _find_prior_ctgov_count(tmp_path) is None


# ── SEC 8-K staging tests (mocked collector) ─────────────────────────

FAKE_PATTERN_VERSION = "abcd1234"


def _make_events(n: int) -> list[dict]:
    return [
        {
            "ticker": f"T{i}",
            "event_type": "DATA_READOUT",
            "event_date": f"2026-03-{i+1:02d}",
            "disclosed_at": "2026-02-10",
        }
        for i in range(n)
    ]


def _write_universe(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "universe.json").write_text(
        json.dumps([{"ticker": "ACME", "cik": "12345"}])
    )


@patch(
    "wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector.PATTERN_VERSION",
    FAKE_PATTERN_VERSION,
)
@patch(
    "wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector.collect_8k_timing_events",
)
def test_sec_8k_staging_accepted(mock_collect, tmp_path):
    """Collector returns events → live file created, staging dir cleaned."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    data_dir = tmp_path / "data"
    _write_universe(data_dir)

    events = _make_events(50)
    mock_collect.return_value = events

    result = warm_sec_8k(date(2026, 2, 15), data_dir, cache_dir)
    assert result == 50

    # Live file should exist
    live_path = cache_dir / f"8k_catalysts_2026-02-15_{FAKE_PATTERN_VERSION}.json"
    assert live_path.exists()
    assert len(json.loads(live_path.read_text())) == 50

    # Staging dir should be cleaned up (no .staging_ dirs)
    staging_dirs = [p for p in cache_dir.iterdir() if p.name.startswith(".staging_")]
    assert staging_dirs == []


@patch(
    "wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector.PATTERN_VERSION",
    FAKE_PATTERN_VERSION,
)
@patch(
    "wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector.collect_8k_timing_events",
)
def test_sec_8k_staging_rejected_empty(mock_collect, tmp_path):
    """Collector returns [] → live file NOT created, returns attempted count (0)."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    data_dir = tmp_path / "data"
    _write_universe(data_dir)

    mock_collect.return_value = []

    result = warm_sec_8k(date(2026, 2, 15), data_dir, cache_dir)
    assert result == 0  # attempted 0 events

    # No live file
    live_path = cache_dir / f"8k_catalysts_2026-02-15_{FAKE_PATTERN_VERSION}.json"
    assert not live_path.exists()


@patch(
    "wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector.PATTERN_VERSION",
    FAKE_PATTERN_VERSION,
)
@patch(
    "wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector.collect_8k_timing_events",
)
def test_sec_8k_staging_rejected_collapse(mock_collect, tmp_path):
    """New count / prior < 0.30 → rejected, prior untouched, returns attempted count."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    data_dir = tmp_path / "data"
    _write_universe(data_dir)

    # Write a prior cache with 100 events
    prior_events = _make_events(100)
    prior_path = cache_dir / f"8k_catalysts_2026-02-14_{FAKE_PATTERN_VERSION}.json"
    prior_path.write_text(json.dumps(prior_events))

    # Collector returns only 10 events (ratio=0.10 < 0.30)
    mock_collect.return_value = _make_events(10)

    result = warm_sec_8k(date(2026, 2, 15), data_dir, cache_dir)
    assert result == 10  # attempted count preserved for sidecar fidelity

    # Prior still intact
    assert prior_path.exists()
    assert len(json.loads(prior_path.read_text())) == 100

    # No new file for 2026-02-15
    live_path = cache_dir / f"8k_catalysts_2026-02-15_{FAKE_PATTERN_VERSION}.json"
    assert not live_path.exists()

    # validate_cache_refresh sees the true reason
    accepted, reason = validate_cache_refresh("sec_8k", result, 100)
    assert not accepted
    assert "collapse" in reason


@patch(
    "wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector.PATTERN_VERSION",
    FAKE_PATTERN_VERSION,
)
def test_sec_8k_existing_cache_no_refetch(tmp_path):
    """Live cache exists → short-circuits, no collector call."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    data_dir = tmp_path / "data"
    _write_universe(data_dir)

    # Pre-populate live cache
    live_path = cache_dir / f"8k_catalysts_2026-02-15_{FAKE_PATTERN_VERSION}.json"
    events = _make_events(75)
    live_path.write_text(json.dumps(events))

    # No mock needed — should not reach the collector
    result = warm_sec_8k(date(2026, 2, 15), data_dir, cache_dir)
    assert result == 75


# ── CTGov staging tests ──────────────────────────────────────────────


def _write_trial_records(data_dir: Path, n: int, lup_date: str = "2026-02-01") -> None:
    """Write n trial records with given last_update_posted date."""
    data_dir.mkdir(parents=True, exist_ok=True)
    records = [
        {"nct_id": f"NCT{i:06d}", "last_update_posted": lup_date}
        for i in range(n)
    ]
    (data_dir / "trial_records.json").write_text(json.dumps(records))


def test_ctgov_staging_accepted(tmp_path):
    """Filtered count validates → live file created via atomic rename."""
    cache_dir = tmp_path / "cache"
    data_dir = tmp_path / "data"
    _write_trial_records(data_dir, 500)

    result = warm_ctgov(date(2026, 2, 15), data_dir, cache_dir)
    assert result == 500

    target = cache_dir / "trial_records_2026-02-15.json"
    assert target.exists()
    assert len(json.loads(target.read_text())) == 500

    # No temp files left
    tmp_files = [p for p in cache_dir.iterdir() if p.suffix == ".tmp"]
    assert tmp_files == []


def test_ctgov_staging_rejected_spike(tmp_path):
    """Count/prior > 1.50 → live untouched, returns attempted count."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "data"

    # Write a prior cache with 100 records
    prior_records = [{"nct_id": f"NCT{i:06d}", "last_update_posted": "2026-01-15"} for i in range(100)]
    (cache_dir / "trial_records_2026-02-14.json").write_text(json.dumps(prior_records))

    # Source has 200 records (all pass filter) → ratio = 2.0 > 1.50
    _write_trial_records(data_dir, 200, lup_date="2026-02-01")

    result = warm_ctgov(date(2026, 2, 15), data_dir, cache_dir)
    assert result == 200  # attempted count preserved

    # No new file for 2026-02-15
    assert not (cache_dir / "trial_records_2026-02-15.json").exists()
    # Prior still intact
    assert (cache_dir / "trial_records_2026-02-14.json").exists()

    # validate_cache_refresh sees the true reason
    accepted, reason = validate_cache_refresh("ctgov", result, 100)
    assert not accepted
    assert "out_of_band" in reason


def test_ctgov_staging_rejected_crash(tmp_path):
    """Count/prior < 0.60 → live untouched, returns attempted count."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "data"

    # Prior: 1000 records
    prior = [{"nct_id": f"NCT{i:06d}", "last_update_posted": "2026-01-15"} for i in range(1000)]
    (cache_dir / "trial_records_2026-02-14.json").write_text(json.dumps(prior))

    # Source: only 500 pass filter → ratio = 0.50 < 0.60
    records = [{"nct_id": f"NCT{i:06d}", "last_update_posted": "2026-02-01"} for i in range(500)]
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "trial_records.json").write_text(json.dumps(records))

    result = warm_ctgov(date(2026, 2, 15), data_dir, cache_dir)
    assert result == 500  # attempted count preserved
    assert not (cache_dir / "trial_records_2026-02-15.json").exists()


def test_ctgov_no_prior_fresh_accepted(tmp_path):
    """No prior cache, count>0 → writes normally."""
    cache_dir = tmp_path / "cache"
    data_dir = tmp_path / "data"
    _write_trial_records(data_dir, 800)

    result = warm_ctgov(date(2026, 2, 15), data_dir, cache_dir)
    assert result == 800

    target = cache_dir / "trial_records_2026-02-15.json"
    assert target.exists()


def test_ctgov_existing_cache_short_circuits(tmp_path):
    """Existing cache → returns count without re-filtering."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    records = [{"nct_id": "NCT000001", "last_update_posted": "2026-02-01"}] * 42
    (cache_dir / "trial_records_2026-02-15.json").write_text(json.dumps(records))

    # data_dir doesn't even need trial_records.json — should short-circuit
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    result = warm_ctgov(date(2026, 2, 15), data_dir, cache_dir)
    assert result == 42
