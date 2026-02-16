"""Tests for SEC 8-K delta warming in warm_caches.py."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from warm_caches import (
    warm_sec_8k_delta,
    _extract_pattern_version,
    _dedup_events,
    _DELTA_LOOKBACK_DAYS,
    _EXPIRE_WINDOW_DAYS,
)

# ── fixtures ──────────────────────────────────────────────────────────

FAKE_PATTERN_VERSION = "abcd1234"


def _make_event(ticker: str, event_type: str, event_date: str, disclosed_at: str) -> dict:
    return {
        "ticker": ticker,
        "event_type": event_type,
        "event_date": event_date,
        "event_date_end": None,
        "date_precision": "DAY",
        "event_name": f"8-K: test event for {ticker}",
        "confidence": "HIGH",
        "source": "SEC_8K_FILING",
        "disclosed_at": disclosed_at,
        "tags": ["sec_8k"],
    }


def _write_seed(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(events, f)


def _write_universe(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / "universe.json", "w") as f:
        json.dump([{"ticker": "ACME", "cik": "12345"}], f)


# ── unit tests for helpers ────────────────────────────────────────────


def test_extract_pattern_version_valid():
    p = Path("8k_catalysts_2026-02-14_abcd1234.json")
    assert _extract_pattern_version(p) == "abcd1234"


def test_extract_pattern_version_no_version():
    p = Path("8k_catalysts_2026-02-14.json")
    assert _extract_pattern_version(p) is None


def test_dedup_events():
    e1 = _make_event("ACME", "DATA_READOUT", "2026-03-01", "2026-02-10")
    e2 = _make_event("ACME", "DATA_READOUT", "2026-03-01", "2026-02-11")  # same key
    e3 = _make_event("ACME", "FDA_PDUFA_DATE", "2026-03-01", "2026-02-10")  # different type
    result = _dedup_events([e1, e2, e3])
    assert len(result) == 2
    # First occurrence wins
    assert result[0]["disclosed_at"] == "2026-02-10"
    assert result[1]["event_type"] == "FDA_PDUFA_DATE"


# ── delta warm integration tests (mocked collector) ───────────────────


@patch("warm_caches.warm_sec_8k")
def test_delta_missing_seed_falls_back(mock_full, tmp_path):
    """Nonexistent seed path → full warm."""
    mock_full.return_value = 100
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    data_dir = tmp_path / "data"
    _write_universe(data_dir)

    result = warm_sec_8k_delta(
        date(2026, 2, 15), data_dir, cache_dir,
        seed_cache_path=tmp_path / "nonexistent.json",
    )
    assert result == 100
    mock_full.assert_called_once()


@patch("warm_caches.warm_sec_8k")
def test_delta_pattern_mismatch_falls_back(mock_full, tmp_path):
    """Wrong PATTERN_VERSION in seed filename → full warm."""
    mock_full.return_value = 100
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    data_dir = tmp_path / "data"
    _write_universe(data_dir)

    # Seed with wrong version (deadbeef != current PATTERN_VERSION)
    seed = cache_dir / "8k_catalysts_2026-02-14_deadbeef.json"
    _write_seed(seed, [])

    # The function imports PATTERN_VERSION inside — patch at collector level
    # Use a version that does NOT match "deadbeef" in the seed filename
    with patch(
        "wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector.PATTERN_VERSION",
        FAKE_PATTERN_VERSION,  # abcd1234 != deadbeef → mismatch
    ):
        result = warm_sec_8k_delta(
            date(2026, 2, 15), data_dir, cache_dir, seed_cache_path=seed,
        )
    assert result == 100
    mock_full.assert_called_once()


def test_delta_merges_seed_and_new(tmp_path):
    """5 seed + 2 new → 7 merged (no overlap)."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    data_dir = tmp_path / "data"
    _write_universe(data_dir)

    as_of = date(2026, 2, 15)

    seed_events = [
        _make_event(f"T{i}", "DATA_READOUT", f"2026-03-0{i}", "2026-02-10")
        for i in range(1, 6)
    ]
    delta_events = [
        _make_event("NEW1", "DATA_READOUT", "2026-04-01", "2026-02-14"),
        _make_event("NEW2", "FDA_PDUFA_DATE", "2026-05-01", "2026-02-14"),
    ]

    # Write seed with matching version
    with patch(
        "wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector.PATTERN_VERSION",
        FAKE_PATTERN_VERSION,
    ):
        from wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector import (
            _versioned_cache_path,
        )
        seed_path = cache_dir / f"8k_catalysts_2026-02-14_{FAKE_PATTERN_VERSION}.json"
        _write_seed(seed_path, seed_events)

        with patch(
            "wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector.collect_8k_timing_events",
            return_value=delta_events,
        ) as mock_collect:
            result = warm_sec_8k_delta(as_of, data_dir, cache_dir, seed_path)

    assert result == 7
    mock_collect.assert_called_once()
    assert mock_collect.call_args.kwargs.get("lookback_days") == _DELTA_LOOKBACK_DAYS


def test_delta_dedup_by_key(tmp_path):
    """Overlapping (ticker, event_type, event_date) → kept once."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    data_dir = tmp_path / "data"
    _write_universe(data_dir)

    as_of = date(2026, 2, 15)
    shared_event = _make_event("ACME", "DATA_READOUT", "2026-03-15", "2026-02-10")
    seed_events = [shared_event]
    delta_events = [
        _make_event("ACME", "DATA_READOUT", "2026-03-15", "2026-02-14"),  # same key
    ]

    seed_path = cache_dir / f"8k_catalysts_2026-02-14_{FAKE_PATTERN_VERSION}.json"
    _write_seed(seed_path, seed_events)

    with patch(
        "wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector.PATTERN_VERSION",
        FAKE_PATTERN_VERSION,
    ), patch(
        "wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector.collect_8k_timing_events",
        return_value=delta_events,
    ):
        result = warm_sec_8k_delta(as_of, data_dir, cache_dir, seed_path)

    assert result == 1  # deduped to 1


def test_delta_expires_old_events(tmp_path):
    """Events with disclosed_at > 180d ago → dropped."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    data_dir = tmp_path / "data"
    _write_universe(data_dir)

    as_of = date(2026, 2, 15)
    # disclosed_at = 2025-08-01 → 198 days before as_of → expired
    old_event = _make_event("OLD", "DATA_READOUT", "2025-09-01", "2025-08-01")
    recent_event = _make_event("RECENT", "DATA_READOUT", "2026-03-01", "2026-02-10")
    seed_events = [old_event, recent_event]

    seed_path = cache_dir / f"8k_catalysts_2026-02-14_{FAKE_PATTERN_VERSION}.json"
    _write_seed(seed_path, seed_events)

    with patch(
        "wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector.PATTERN_VERSION",
        FAKE_PATTERN_VERSION,
    ), patch(
        "wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector.collect_8k_timing_events",
        return_value=[],
    ):
        result = warm_sec_8k_delta(as_of, data_dir, cache_dir, seed_path)

    assert result == 1  # only recent survives

    # Verify cache file contents
    from wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector import (
        _versioned_cache_path,
    )
    with patch(
        "wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector.PATTERN_VERSION",
        FAKE_PATTERN_VERSION,
    ):
        target = cache_dir / f"8k_catalysts_{as_of.isoformat()}_{FAKE_PATTERN_VERSION}.json"
    with open(target) as f:
        cached = json.load(f)
    assert len(cached) == 1
    assert cached[0]["ticker"] == "RECENT"


def test_delta_empty_seed(tmp_path):
    """Empty seed + delta events = just delta."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    data_dir = tmp_path / "data"
    _write_universe(data_dir)

    as_of = date(2026, 2, 15)
    delta_events = [
        _make_event("NEW1", "DATA_READOUT", "2026-04-01", "2026-02-14"),
    ]

    seed_path = cache_dir / f"8k_catalysts_2026-02-14_{FAKE_PATTERN_VERSION}.json"
    _write_seed(seed_path, [])

    with patch(
        "wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector.PATTERN_VERSION",
        FAKE_PATTERN_VERSION,
    ), patch(
        "wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector.collect_8k_timing_events",
        return_value=delta_events,
    ):
        result = warm_sec_8k_delta(as_of, data_dir, cache_dir, seed_path)

    assert result == 1


def test_delta_lookback_is_narrow(tmp_path):
    """Collector called with lookback_days=7."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    data_dir = tmp_path / "data"
    _write_universe(data_dir)

    as_of = date(2026, 2, 15)
    seed_path = cache_dir / f"8k_catalysts_2026-02-14_{FAKE_PATTERN_VERSION}.json"
    _write_seed(seed_path, [])

    with patch(
        "wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector.PATTERN_VERSION",
        FAKE_PATTERN_VERSION,
    ), patch(
        "wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector.collect_8k_timing_events",
        return_value=[],
    ) as mock_collect:
        warm_sec_8k_delta(as_of, data_dir, cache_dir, seed_path)

    assert mock_collect.call_args.kwargs["lookback_days"] == 7


def test_delta_cli_flag_wired(tmp_path):
    """--seed-cache CLI flag reaches the delta function."""
    seed_path = tmp_path / f"8k_catalysts_2026-02-14_{FAKE_PATTERN_VERSION}.json"
    _write_seed(seed_path, [_make_event("A", "X", "2026-03-01", "2026-02-10")])

    data_dir = tmp_path / "data"
    _write_universe(data_dir)

    with patch(
        "warm_caches.warm_sec_8k_delta", return_value=42,
    ) as mock_delta, patch(
        "warm_caches.warm_fda_adcom", return_value=0,
    ):
        from warm_caches import main as warm_main
        import warm_caches
        # We need to call main with argv
        warm_caches.main.__wrapped__ if hasattr(warm_caches.main, '__wrapped__') else None

        # Patch sys.argv via argparse
        rc = 0
        import argparse
        with patch("sys.argv", [
            "warm_caches.py",
            "--as-of-date", "2026-02-15",
            "--sources", "sec_8k",
            "--data-dir", str(data_dir),
            "--sec-cache-dir", str(tmp_path),
            "--seed-cache", str(seed_path),
        ]):
            from warm_caches import main
            rc = main()

    mock_delta.assert_called_once()
    call_kwargs = mock_delta.call_args
    assert call_kwargs[0][3] == seed_path  # seed_cache_path positional arg
