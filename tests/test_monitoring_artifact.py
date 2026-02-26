"""Tests for daily monitoring artifact generation."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys_path_added = False


def _ensure_path():
    global sys_path_added
    if not sys_path_added:
        import sys

        root = str(Path(__file__).resolve().parent.parent)
        if root not in sys.path:
            sys.path.insert(0, root)
        sys_path_added = True


_ensure_path()

from scripts.build_monitoring_artifact import (
    SCHEMA,
    build_monitoring,
    _render_markdown,
    _eligible_rows,
    _ranked_tickers,
    _compute_overlap,
    _rank_correlation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


COLS = [
    "ticker", "actionable_rank", "eligible", "tier_dev",
    "market_cap_bucket", "stage_bucket", "archetype", "mom_state",
    "size_band", "has_catalyst_signal", "has_coinvest_signal",
    "composite_rank",
]


def _make_snapshot(
    snap_dir: Path,
    date: str,
    tickers: List[str],
    *,
    degraded: bool = False,
    extra_meta: Dict = None,
) -> Path:
    """Create a synthetic snapshot with rankings + metadata."""
    d = snap_dir / date
    d.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, t in enumerate(tickers):
        rows.append({
            "ticker": t,
            "actionable_rank": str(i + 1),
            "eligible": "1",
            "tier_dev": "A" if i < len(tickers) // 2 else "B",
            "market_cap_bucket": "mid" if i % 2 == 0 else "small",
            "stage_bucket": "late" if i < 3 else "early",
            "archetype": "drug_developer",
            "mom_state": "tailwind" if i % 3 == 0 else "headwind",
            "size_band": "L" if i < 4 else "M",
            "has_catalyst_signal": "TRUE" if i % 2 == 0 else "FALSE",
            "has_coinvest_signal": "TRUE" if i < 6 else "FALSE",
            "composite_rank": str(i + 1),
        })
    _write_csv(d / "rankings.csv", rows)

    meta = {
        "as_of_date": date,
        "version": "1.6.0",
        "decision_mode": "phase2",
        "clinical_sort_telemetry": {"ruleset_id": "test1234"},
        "cache_health_overall_status": "bad" if degraded else "ok",
        "cache_degraded_run": degraded,
        "cache_refresh_had_rejections": degraded,
        "cache_refresh_rejected_sources": ["sec_8k"] if degraded else [],
        "alpha_modifier_telemetry": {
            "alpha_table_source": "preexisting_local",
            "alpha_score_present": 190,
            "alpha_score_missing": 10,
        },
        "cache_refresh_summary": {
            "sec_8k": {"count": 220},
            "ctgov": {"count": 18000},
        },
    }
    if extra_meta:
        meta.update(extra_meta)
    _write_json(d / "metadata.json", meta)

    _write_json(d / "cache_health.json", {
        "schema": "cache_health.v1",
        "overall_status": "bad" if degraded else "ok",
        "degraded_run": degraded,
    })

    return d


# ---------------------------------------------------------------------------
# Tests: build_monitoring
# ---------------------------------------------------------------------------


class TestBuildMonitoring:
    def test_happy_path_not_degraded(self, tmp_path):
        snap_dir = tmp_path / "snapshots"
        prior_tickers = [f"T{i:02d}" for i in range(1, 11)]
        curr_tickers = [f"T{i:02d}" for i in range(1, 9)] + ["T20", "T21"]

        _make_snapshot(snap_dir, "2026-01-14", prior_tickers)
        _make_snapshot(snap_dir, "2026-01-15", curr_tickers)

        mon = build_monitoring("2026-01-15", snap_dir, top_k=5)

        assert mon["schema"] == SCHEMA
        assert mon["context"]["as_of_date"] == "2026-01-15"
        assert mon["context"]["cache_degraded_run"] is False

        # Coverage
        assert mon["coverage"]["n_universe"] == 10
        assert mon["coverage"]["n_eligible"] == 10
        assert "catalyst_signal_pct" in mon["coverage"]
        assert "coinvest_signal_pct" in mon["coverage"]

        # Stability should be computed
        stab = mon["stability"]
        assert stab["suppressed"] is False
        assert stab["prior_date"] == "2026-01-14"
        assert "top_20" in stab
        assert "top_5" in stab
        assert stab["top_5"]["overlap_count"] >= 0
        assert "rank_correlation" in stab
        assert "spearman" in stab["rank_correlation"]

        # Exposures
        assert "tier_distribution" in mon["exposures"]
        assert "market_cap_distribution" in mon["exposures"]

    def test_degraded_suppresses_churn(self, tmp_path):
        snap_dir = tmp_path / "snapshots"
        _make_snapshot(snap_dir, "2026-01-14", ["T01", "T02", "T03"])
        _make_snapshot(snap_dir, "2026-01-15", ["T01", "T02", "T03"], degraded=True)

        mon = build_monitoring("2026-01-15", snap_dir, top_k=3)

        assert mon["context"]["cache_degraded_run"] is True
        assert mon["stability"]["suppressed"] is True
        assert mon["stability"]["reason"] == "degraded_run"

    def test_no_prior_snapshot(self, tmp_path):
        snap_dir = tmp_path / "snapshots"
        _make_snapshot(snap_dir, "2026-01-15", ["T01", "T02", "T03"])

        mon = build_monitoring("2026-01-15", snap_dir, top_k=3)

        assert mon["stability"]["suppressed"] is True
        assert mon["stability"]["reason"] == "no_prior_snapshot"

    def test_missing_columns_graceful(self, tmp_path):
        snap_dir = tmp_path / "snapshots"
        d = snap_dir / "2026-01-15"
        d.mkdir(parents=True)

        # Rankings with minimal columns (no stage_bucket, market_cap_bucket)
        rows = [
            {"ticker": "T01", "actionable_rank": "1", "eligible": "1", "tier_dev": "A"},
            {"ticker": "T02", "actionable_rank": "2", "eligible": "1", "tier_dev": "B"},
        ]
        _write_csv(d / "rankings.csv", rows)
        _write_json(d / "metadata.json", {
            "as_of_date": "2026-01-15",
            "version": "1.6.0",
            "decision_mode": "phase2",
            "clinical_sort_telemetry": {"ruleset_id": "test1234"},
            "cache_health_overall_status": "ok",
            "cache_degraded_run": False,
        })

        mon = build_monitoring("2026-01-15", snap_dir, top_k=2)

        # Should still work; missing columns reported
        assert "tier_distribution" in mon["exposures"]
        missing = mon["exposures"]["missing_columns"]
        assert "market_cap_bucket" in missing
        assert "stage_bucket" in missing

    def test_output_is_deterministic(self, tmp_path):
        snap_dir = tmp_path / "snapshots"
        tickers = [f"T{i:02d}" for i in range(1, 6)]
        _make_snapshot(snap_dir, "2026-01-15", tickers)

        mon1 = build_monitoring("2026-01-15", snap_dir, top_k=3)
        mon2 = build_monitoring("2026-01-15", snap_dir, top_k=3)

        # Remove non-deterministic created_at_utc
        mon1["context"].pop("created_at_utc", None)
        mon2["context"].pop("created_at_utc", None)

        j1 = json.dumps(mon1, sort_keys=True, indent=2)
        j2 = json.dumps(mon2, sort_keys=True, indent=2)
        assert j1 == j2

    def test_missing_snapshot_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_monitoring("2099-01-01", tmp_path / "snapshots")

    def test_cache_refresh_counts_in_coverage(self, tmp_path):
        snap_dir = tmp_path / "snapshots"
        _make_snapshot(snap_dir, "2026-01-15", ["T01", "T02"])

        mon = build_monitoring("2026-01-15", snap_dir, top_k=2)
        assert mon["coverage"]["sec_8k_count"] == 220
        assert mon["coverage"]["ctgov_count"] == 18000


# ---------------------------------------------------------------------------
# Tests: markdown rendering
# ---------------------------------------------------------------------------


class TestRenderMarkdown:
    def test_markdown_contains_date(self, tmp_path):
        snap_dir = tmp_path / "snapshots"
        _make_snapshot(snap_dir, "2026-01-15", ["T01", "T02", "T03"])

        mon = build_monitoring("2026-01-15", snap_dir, top_k=3)
        md = _render_markdown(mon, top_k=3)

        assert "2026-01-15" in md
        assert "## Coverage" in md
        assert "## Stability" in md

    def test_degraded_banner(self, tmp_path):
        snap_dir = tmp_path / "snapshots"
        _make_snapshot(snap_dir, "2026-01-15", ["T01", "T02"], degraded=True)

        mon = build_monitoring("2026-01-15", snap_dir, top_k=2)
        md = _render_markdown(mon, top_k=2)

        assert "DEGRADED" in md

    def test_stability_with_prior(self, tmp_path):
        snap_dir = tmp_path / "snapshots"
        _make_snapshot(snap_dir, "2026-01-14", ["T01", "T02", "T03"])
        _make_snapshot(snap_dir, "2026-01-15", ["T01", "T02", "T04"])

        mon = build_monitoring("2026-01-15", snap_dir, top_k=3)
        md = _render_markdown(mon, top_k=3)

        assert "Spearman" in md
        assert "overlap" in md.lower()
        assert "Entrants" in md or "Exits" in md


# ---------------------------------------------------------------------------
# Tests: helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_eligible_rows(self):
        rows = [
            {"ticker": "A", "eligible": "1"},
            {"ticker": "B", "eligible": "0"},
            {"ticker": "C", "eligible": "TRUE"},
            {"ticker": "D", "eligible": ""},
        ]
        result = _eligible_rows(rows)
        assert len(result) == 2
        assert result[0]["ticker"] == "A"
        assert result[1]["ticker"] == "C"

    def test_ranked_tickers(self):
        rows = [
            {"ticker": "B", "actionable_rank": "2"},
            {"ticker": "A", "actionable_rank": "1"},
            {"ticker": "C", "actionable_rank": "3"},
        ]
        assert _ranked_tickers(rows) == ["A", "B", "C"]

    def test_compute_overlap(self):
        result = _compute_overlap(["A", "B", "C", "D"], ["A", "B", "E", "F"], k=3)
        assert result["top_k"] == 3
        assert result["overlap_count"] == 2
        assert result["overlap_pct"] == pytest.approx(66.7, abs=0.1)

    def test_rank_correlation_identical(self):
        ranks = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
        result = _rank_correlation(ranks, ranks)
        assert result["spearman"] == pytest.approx(1.0)
        assert result["n_common"] == 5

    def test_rank_correlation_reversed(self):
        curr = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
        prior = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}
        result = _rank_correlation(curr, prior)
        assert result["spearman"] == pytest.approx(-1.0)

    def test_rank_correlation_insufficient_data(self):
        curr = {"A": 1, "B": 2}
        prior = {"A": 2, "B": 1}
        result = _rank_correlation(curr, prior)
        assert result["spearman"] is None


# ---------------------------------------------------------------------------
# Tests: CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_valid_exits_0(self, tmp_path):
        snap_dir = tmp_path / "snapshots"
        _make_snapshot(snap_dir, "2026-01-15", ["T01", "T02", "T03"])

        from scripts.build_monitoring_artifact import main

        rc = main([
            "--as-of-date", "2026-01-15",
            "--snapshot-dir", str(snap_dir),
            "--out-json", str(tmp_path / "mon.json"),
            "--out-md", str(tmp_path / "mon.md"),
            "--top-k", "3",
        ])
        assert rc == 0
        assert (tmp_path / "mon.json").exists()
        assert (tmp_path / "mon.md").exists()

        data = json.loads((tmp_path / "mon.json").read_text())
        assert data["schema"] == SCHEMA

    def test_missing_snapshot_exits_1(self, tmp_path):
        from scripts.build_monitoring_artifact import main

        rc = main([
            "--as-of-date", "2099-01-01",
            "--snapshot-dir", str(tmp_path / "nope"),
            "--out-json", str(tmp_path / "mon.json"),
            "--out-md", str(tmp_path / "mon.md"),
        ])
        assert rc == 1
