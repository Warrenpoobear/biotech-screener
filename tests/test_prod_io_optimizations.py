#!/usr/bin/env python3
"""Tests for production I/O optimizations in run_screen.py / run_daily_production.py.

Covers (all behavior-preserving / additive):
  - _read_price_history_rows: parse-once caching keyed by (path, mtime, size)
  - detect_no_material_input_change: logging-only no-op detector
  - _load_json_threadsafe: thread-safe JSON loader for parallel preload
  - compute_module_3_catalyst trial_records kwarg: skips redundant 15MB re-parse

These guard the production-infrastructure changes only; model/ranker/selector
behavior is covered by the golden-baseline regression suite.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MINI_PRICE_CSV = REPO_ROOT / "tests" / "fixtures" / "mini_data" / "price_history.csv"


# ---------------------------------------------------------------------------
# Price-history row cache
# ---------------------------------------------------------------------------
class TestPriceHistoryRowCache:
    def test_rows_match_raw_dictreader(self):
        """Cached rows are byte-identical to a direct csv.DictReader pass."""
        import run_screen

        with open(MINI_PRICE_CSV, "r", encoding="utf-8") as fh:
            expected = list(csv.DictReader(fh))
        got = run_screen._read_price_history_rows(MINI_PRICE_CSV)
        assert got == expected, "cached rows must equal raw DictReader output"

    def test_parse_once_then_reuse(self):
        """Second call returns the SAME cached object (no re-parse)."""
        import run_screen

        run_screen._PRICE_HISTORY_ROW_CACHE.clear()
        first = run_screen._read_price_history_rows(MINI_PRICE_CSV)
        second = run_screen._read_price_history_rows(MINI_PRICE_CSV)
        assert first is second, "repeat read should return the cached list object"

    def test_cache_invalidates_on_change(self, tmp_path):
        """A changed file (new mtime/size) is re-parsed, not served stale."""
        import run_screen

        run_screen._PRICE_HISTORY_ROW_CACHE.clear()
        p = tmp_path / "price_history.csv"
        p.write_text("date,ticker,close\n2026-01-02,AAA,10.0\n", encoding="utf-8")
        rows1 = run_screen._read_price_history_rows(p)
        assert len(rows1) == 1
        # Rewrite with an extra row; cache key (mtime/size) changes → re-parse.
        p.write_text("date,ticker,close\n2026-01-02,AAA,10.0\n2026-01-03,BBB,20.0\n", encoding="utf-8")
        rows2 = run_screen._read_price_history_rows(p)
        assert len(rows2) == 2, "changed file must be re-parsed"

    def test_cache_bounded_to_current_file(self, tmp_path):
        """Only the current file is retained (memory stays bounded)."""
        import run_screen

        run_screen._PRICE_HISTORY_ROW_CACHE.clear()
        a = tmp_path / "a.csv"
        b = tmp_path / "b.csv"
        a.write_text("date,ticker,close\n2026-01-02,AAA,10.0\n", encoding="utf-8")
        b.write_text("date,ticker,close\n2026-01-02,BBB,20.0\n", encoding="utf-8")
        run_screen._read_price_history_rows(a)
        run_screen._read_price_history_rows(b)
        assert len(run_screen._PRICE_HISTORY_ROW_CACHE) == 1


# ---------------------------------------------------------------------------
# No-op detector (logging-only)
# ---------------------------------------------------------------------------
class TestNoOpDetection:
    def _make_snapshot(self, root: Path, date_str: str, hashes: dict):
        d = root / date_str
        d.mkdir(parents=True, exist_ok=True)
        (d / "run_manifest.json").write_text(json.dumps({"hashes": hashes}), encoding="utf-8")
        return d

    def test_returns_none_without_prior(self, tmp_path):
        from tools.run_daily_production import detect_no_material_input_change

        out = detect_no_material_input_change(
            "2026-06-28",
            data_dir=tmp_path,
            price_csv=tmp_path / "nope.csv",
            ctgov_cache_dir=tmp_path / "ctgov",
            final_snapshots_dir=tmp_path / "snaps",
        )
        assert out is None

    def test_detects_unchanged_inputs(self, tmp_path, monkeypatch):
        import tools.run_daily_production as m

        snaps = tmp_path / "snaps"
        prior_hashes = {
            "model_hash": "M",
            "universe_hash": "U",
            "price_hash": "P",
            "clinical_hash": "C",
        }
        self._make_snapshot(snaps, "2026-06-27", prior_hashes)
        # Force current input hashes to equal the prior ones.
        monkeypatch.setattr(m, "_compute_provenance_hashes", lambda **kw: dict(prior_hashes, rankings_hash=""))
        out = m.detect_no_material_input_change(
            "2026-06-28",
            data_dir=tmp_path,
            price_csv=tmp_path / "p.csv",
            ctgov_cache_dir=None,
            final_snapshots_dir=snaps,
        )
        assert out is not None
        assert out["no_material_input_change"] is True
        assert out["changed_inputs"] == []
        assert out["compared_against"] == "2026-06-27"

    def test_detects_changed_inputs(self, tmp_path, monkeypatch):
        import tools.run_daily_production as m

        snaps = tmp_path / "snaps"
        self._make_snapshot(snaps, "2026-06-27", {"model_hash": "M", "price_hash": "P"})
        monkeypatch.setattr(
            m, "_compute_provenance_hashes", lambda **kw: {"model_hash": "M", "price_hash": "P2", "rankings_hash": ""}
        )
        out = m.detect_no_material_input_change(
            "2026-06-28",
            data_dir=tmp_path,
            price_csv=tmp_path / "p.csv",
            ctgov_cache_dir=None,
            final_snapshots_dir=snaps,
        )
        assert out["no_material_input_change"] is False
        assert "price_hash" in out["changed_inputs"]


# ---------------------------------------------------------------------------
# _load_json_threadsafe
# ---------------------------------------------------------------------------
class TestLoadJsonThreadsafe:
    def test_returns_same_data_as_json_load(self, tmp_path):
        """Produces identical output to a direct json.load call."""
        import run_screen

        p = tmp_path / "data.json"
        payload = [{"ticker": "AAA", "val": 1}, {"ticker": "BBB", "val": 2}]
        p.write_text(json.dumps(payload), encoding="utf-8")
        got = run_screen._load_json_threadsafe(p, "Test")
        assert got == payload

    def test_raises_on_missing_file(self, tmp_path):
        """FileNotFoundError for a path that does not exist."""
        import run_screen

        with pytest.raises(FileNotFoundError, match="Test file not found"):
            run_screen._load_json_threadsafe(tmp_path / "missing.json", "Test")

    def test_raises_on_non_list_json(self, tmp_path):
        """ValueError when the JSON root is not an array."""
        import run_screen

        p = tmp_path / "obj.json"
        p.write_text('{"key": "value"}', encoding="utf-8")
        with pytest.raises(ValueError, match="must be a JSON array"):
            run_screen._load_json_threadsafe(p, "Test")

    def test_raises_on_symlink(self, tmp_path):
        """SymlinkError for symlinked files (security check)."""
        import run_screen

        real = tmp_path / "real.json"
        real.write_text("[{}]", encoding="utf-8")
        link = tmp_path / "link.json"
        link.symlink_to(real)
        with pytest.raises(run_screen.SymlinkError):
            run_screen._load_json_threadsafe(link, "Test")

    def test_parallel_loads_return_correct_data(self, tmp_path):
        """ThreadPoolExecutor with 4 workers returns correct data for each file."""
        import concurrent.futures

        import run_screen

        files = {}
        for i in range(4):
            p = tmp_path / f"f{i}.json"
            payload = [{"idx": i, "ticker": f"T{i}"}]
            p.write_text(json.dumps(payload), encoding="utf-8")
            files[f"f{i}"] = (p, payload)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futs = {k: pool.submit(run_screen._load_json_threadsafe, p, k) for k, (p, _) in files.items()}
            results = {k: fut.result() for k, fut in futs.items()}

        for k, (_, expected) in files.items():
            assert results[k] == expected, f"parallel load mismatch for {k}"


# ---------------------------------------------------------------------------
# Module 3 trial_records kwarg (pre-loaded bypass)
# ---------------------------------------------------------------------------
class TestModule3TrialRecordsKwarg:
    def _minimal_record(self, nct_id: str, ticker: str) -> dict:
        """Minimal trial record that passes ctgov_adapter critical-field validation."""
        return {
            "nct_id": nct_id,
            "ticker": ticker,
            "status": "RECRUITING",
            "overall_status": "RECRUITING",
            "primary_completion_date": "2027-06-01",
            "last_update_posted": "2026-06-01",
            "phases": ["PHASE2"],
            "conditions": ["Cancer"],
            "interventions": [{"intervention_type": "DRUG", "intervention_name": "TestDrug"}],
            "sponsor": "Test Pharma",
        }

    def test_accepts_preloaded_trial_records(self, tmp_path):
        """compute_module_3_catalyst uses trial_records kwarg and does NOT open the path."""
        from datetime import date

        from module_3_catalyst import compute_module_3_catalyst

        records = [self._minimal_record("NCT00000001", "FAKE")]
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        # Ghost path does not exist — a file-open would raise FileNotFoundError.
        ghost_path = tmp_path / "nonexistent_trial_records.json"

        result = compute_module_3_catalyst(
            trial_records_path=ghost_path,
            trial_records=records,
            state_dir=state_dir,
            active_tickers={"FAKE"},
            as_of_date=date(2026, 6, 28),
            pit_mode="degrade",
        )
        assert "summaries" in result
        assert isinstance(result["summaries"], dict)

    def test_falls_back_to_file_when_not_provided(self, tmp_path):
        """Without trial_records kwarg, the function reads from trial_records_path."""
        from datetime import date

        from module_3_catalyst import compute_module_3_catalyst

        records = [self._minimal_record("NCT00000002", "FAKE2")]
        p = tmp_path / "trial_records.json"
        p.write_text(json.dumps(records), encoding="utf-8")
        state_dir = tmp_path / "state"
        state_dir.mkdir()

        result = compute_module_3_catalyst(
            trial_records_path=p,
            state_dir=state_dir,
            active_tickers={"FAKE2"},
            as_of_date=date(2026, 6, 28),
            pit_mode="degrade",
        )
        assert "summaries" in result


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
