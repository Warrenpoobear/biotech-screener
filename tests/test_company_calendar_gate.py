"""Tests for scripts/eval_company_calendar_gate.py — 8 tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.eval_company_calendar_gate import (
    EXIT_FAIL,
    EXIT_PASS,
    EXIT_WARN,
    evaluate_gate,
    main,
    validate_stream,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_cache(
    tmp_path: Path,
    stream: str,
    events: list,
    stats: dict | None = None,
) -> Path:
    """Write a cache wrapper JSON file and return its path."""
    schema = "ir_events.v1" if stream == "ir" else "press_release_events.v1"
    prefix = "ir_events" if stream == "ir" else "press_release_events"
    cache_dir = tmp_path / ("ir" if stream == "ir" else "press")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{prefix}_2026-03-01.json"
    data = {
        "schema": schema,
        "as_of_date": "2026-03-01",
        "events": events,
        "stats": stats or {
            "fetched_pages": 0,
            "parsed_items": 0,
            "matched": len(events),
            "unmatched": 0,
            "unmatched_samples": [],
        },
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _make_event(ticker: str, event_date: str, source: str = "IR_EVENTS") -> dict:
    return {
        "ticker": ticker,
        "event_date": event_date,
        "source": source,
        "confidence": "MED",
        "title": f"{ticker} event on {event_date}",
    }


AS_OF = "2026-03-01"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPassFutureDatesOnly:
    def test_pass_future_dates_only(self, tmp_path: Path) -> None:
        ir_path = _make_cache(tmp_path, "ir", [
            _make_event("MRNA", "2026-04-15"),
            _make_event("BIIB", "2026-05-01"),
        ])
        pr_path = _make_cache(tmp_path, "pr", [
            _make_event("GILD", "2026-03-20", "PRESS_RELEASE"),
        ])
        out_dir = tmp_path / "out"
        rc = main([
            "--as-of-date", AS_OF,
            "--ir-cache", str(ir_path),
            "--pr-cache", str(pr_path),
            "--out-dir", str(out_dir),
        ])
        assert rc == EXIT_PASS


class TestFailPastDate:
    def test_fail_past_date(self, tmp_path: Path) -> None:
        ir_path = _make_cache(tmp_path, "ir", [
            _make_event("MRNA", "2026-04-15"),
            _make_event("BIIB", "2026-02-15"),  # past date
        ])
        pr_path = _make_cache(tmp_path, "pr", [
            _make_event("GILD", "2026-03-20", "PRESS_RELEASE"),
        ])
        out_dir = tmp_path / "out"
        rc = main([
            "--as-of-date", AS_OF,
            "--ir-cache", str(ir_path),
            "--pr-cache", str(pr_path),
            "--out-dir", str(out_dir),
        ])
        assert rc == EXIT_FAIL


class TestFailNonDayPrecision:
    def test_fail_non_day_precision(self, tmp_path: Path) -> None:
        ir_path = _make_cache(tmp_path, "ir", [
            _make_event("MRNA", "2026-03"),  # month only
        ])
        pr_path = _make_cache(tmp_path, "pr", [
            _make_event("GILD", "2026-03-20", "PRESS_RELEASE"),
        ])
        out_dir = tmp_path / "out"
        rc = main([
            "--as-of-date", AS_OF,
            "--ir-cache", str(ir_path),
            "--pr-cache", str(pr_path),
            "--out-dir", str(out_dir),
        ])
        assert rc == EXIT_FAIL


class TestWarnEmptyCaches:
    def test_warn_empty_caches(self, tmp_path: Path) -> None:
        ir_path = _make_cache(tmp_path, "ir", [])
        pr_path = _make_cache(tmp_path, "pr", [])
        out_dir = tmp_path / "out"
        rc = main([
            "--as-of-date", AS_OF,
            "--ir-cache", str(ir_path),
            "--pr-cache", str(pr_path),
            "--out-dir", str(out_dir),
        ])
        assert rc == EXIT_WARN


class TestWarnMissingCacheFile:
    def test_warn_missing_cache_file(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        rc = main([
            "--as-of-date", AS_OF,
            "--ir-cache", str(tmp_path / "nonexistent_ir.json"),
            "--pr-cache", str(tmp_path / "nonexistent_pr.json"),
            "--out-dir", str(out_dir),
        ])
        assert rc == EXIT_WARN


class TestMixedPassAndWarn:
    def test_mixed_pass_and_warn(self, tmp_path: Path) -> None:
        """IR has events, PR empty → WARN (below min)."""
        ir_path = _make_cache(tmp_path, "ir", [
            _make_event("MRNA", "2026-04-15"),
            _make_event("BIIB", "2026-05-01"),
        ])
        pr_path = _make_cache(tmp_path, "pr", [])  # empty
        out_dir = tmp_path / "out"
        rc = main([
            "--as-of-date", AS_OF,
            "--ir-cache", str(ir_path),
            "--pr-cache", str(pr_path),
            "--out-dir", str(out_dir),
        ])
        assert rc == EXIT_WARN


class TestUnmatchedSamplesBounded:
    def test_unmatched_samples_bounded(self, tmp_path: Path) -> None:
        """200 unmatched samples, cap=100 → output has 100."""
        big_unmatched = [{"ticker": f"T{i}", "raw": f"text_{i}"} for i in range(200)]
        ir_path = _make_cache(
            tmp_path, "ir",
            [_make_event("MRNA", "2026-04-15")],
            stats={
                "fetched_pages": 1, "parsed_items": 200,
                "matched": 1, "unmatched": 200,
                "unmatched_samples": big_unmatched,
            },
        )
        pr_path = _make_cache(tmp_path, "pr", [
            _make_event("GILD", "2026-03-20", "PRESS_RELEASE"),
        ])
        out_dir = tmp_path / "out"
        rc = main([
            "--as-of-date", AS_OF,
            "--ir-cache", str(ir_path),
            "--pr-cache", str(pr_path),
            "--out-dir", str(out_dir),
            "--max-unmatched-samples", "100",
        ])
        assert rc == EXIT_PASS

        unmatched = json.loads(
            (out_dir / "unmatched_samples.json").read_text(encoding="utf-8")
        )
        assert len(unmatched) == 100


class TestOutputFilesWritten:
    def test_output_files_written(self, tmp_path: Path) -> None:
        ir_path = _make_cache(tmp_path, "ir", [
            _make_event("MRNA", "2026-04-15"),
        ])
        pr_path = _make_cache(tmp_path, "pr", [
            _make_event("GILD", "2026-03-20", "PRESS_RELEASE"),
        ])
        out_dir = tmp_path / "out"
        rc = main([
            "--as-of-date", AS_OF,
            "--ir-cache", str(ir_path),
            "--pr-cache", str(pr_path),
            "--out-dir", str(out_dir),
        ])
        assert rc == EXIT_PASS

        # Verify all three output files exist
        assert (out_dir / "company_calendar_gate.json").exists()
        assert (out_dir / "company_calendar_gate.md").exists()
        assert (out_dir / "unmatched_samples.json").exists()

        # Verify JSON schema
        gate = json.loads(
            (out_dir / "company_calendar_gate.json").read_text(encoding="utf-8")
        )
        assert gate["schema"] == "company_calendar_gate.v1"
        assert gate["verdict"] == "PASS"
        assert gate["as_of_date"] == AS_OF

        # Verify MD content
        md = (out_dir / "company_calendar_gate.md").read_text(encoding="utf-8")
        assert "## Company Calendar Gate" in md
        assert "**PASS**" in md
