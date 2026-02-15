"""Tests for catalyst shadow metrics telemetry sidecar."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from run_screen import (
    _find_prior_rankings,
    _compute_shadow_metrics,
    _GOOD_CATALYST_MODES,
    _BAD_CATALYST_MODES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_rankings(snap_dir: Path, date_str: str, rows: list[dict]) -> Path:
    """Write a minimal rankings.csv into snap_dir/date_str/."""
    d = snap_dir / date_str
    d.mkdir(parents=True, exist_ok=True)
    if not rows:
        return d
    csv_path = d / "rankings.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return d


def _make_dev_row(ticker, composite_rank=1, catalyst_mode="specific_days",
                  catalyst_days="90", tier_dev="B", **extra):
    """Build a minimal dev-archetype row dict."""
    row = {
        "ticker": ticker,
        "composite_rank": str(composite_rank),
        "archetype": "drug_developer",
        "catalyst_mode": catalyst_mode,
        "catalyst_days": str(catalyst_days) if catalyst_days is not None else "",
        "tier_dev": tier_dev,
    }
    row.update(extra)
    return row


def _make_comm_row(ticker, composite_rank=1, **extra):
    """Build a minimal commercial-archetype row dict."""
    row = {
        "ticker": ticker,
        "composite_rank": str(composite_rank),
        "archetype": "commercial_biotech",
        "catalyst_mode": "no_upcoming",
        "catalyst_days": "",
        "tier_dev": "",
    }
    row.update(extra)
    return row


# ===================================================================
# TestFindPriorRankings
# ===================================================================

class TestFindPriorRankings:
    """Tests for _find_prior_rankings()."""

    def test_finds_most_recent_prior(self, tmp_path):
        snap = tmp_path / "snapshots"
        _write_rankings(snap, "2026-02-10", [_make_dev_row("AAA")])
        _write_rankings(snap, "2026-02-12", [_make_dev_row("BBB")])
        _write_rankings(snap, "2026-02-14", [_make_dev_row("CCC")])

        result = _find_prior_rankings(snap, "2026-02-14")
        assert result is not None
        assert result["date"] == "2026-02-12"
        assert result["rows"][0]["ticker"] == "BBB"

    def test_returns_none_when_no_prior(self, tmp_path):
        snap = tmp_path / "snapshots"
        _write_rankings(snap, "2026-02-14", [_make_dev_row("AAA")])

        result = _find_prior_rankings(snap, "2026-02-14")
        assert result is None

    def test_skips_dirs_without_tier_dev(self, tmp_path):
        snap = tmp_path / "snapshots"
        # Write a prior without tier_dev column
        d = snap / "2026-02-12"
        d.mkdir(parents=True)
        csv_path = d / "rankings.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("ticker,composite_rank\n")
            f.write("OLD,1\n")
        # Write an older prior WITH tier_dev
        _write_rankings(snap, "2026-02-10", [_make_dev_row("GOOD")])
        # Current
        _write_rankings(snap, "2026-02-14", [_make_dev_row("CUR")])

        result = _find_prior_rankings(snap, "2026-02-14")
        assert result is not None
        assert result["date"] == "2026-02-10"

    def test_skips_dirs_without_rankings_csv(self, tmp_path):
        snap = tmp_path / "snapshots"
        # Dir exists but no rankings.csv
        (snap / "2026-02-12").mkdir(parents=True)
        _write_rankings(snap, "2026-02-10", [_make_dev_row("GOOD")])
        _write_rankings(snap, "2026-02-14", [_make_dev_row("CUR")])

        result = _find_prior_rankings(snap, "2026-02-14")
        assert result is not None
        assert result["date"] == "2026-02-10"

    def test_returns_none_snapshot_dir_missing(self, tmp_path):
        result = _find_prior_rankings(tmp_path / "nonexistent", "2026-02-14")
        assert result is None

    def test_returns_none_current_date_not_found(self, tmp_path):
        snap = tmp_path / "snapshots"
        _write_rankings(snap, "2026-02-10", [_make_dev_row("AAA")])

        result = _find_prior_rankings(snap, "2026-02-14")
        assert result is None


# ===================================================================
# TestComputeShadowMetrics
# ===================================================================

class TestComputeShadowMetrics:
    """Tests for _compute_shadow_metrics()."""

    def test_no_prior_transition_none_distribution_works(self, tmp_path):
        snap = tmp_path / "snapshots"
        _write_rankings(snap, "2026-02-14", [_make_dev_row("AAA", catalyst_days="60")])

        rows = [_make_dev_row("AAA", catalyst_days="60")]
        result = _compute_shadow_metrics(rows, "2026-02-14", snap, None)

        assert result["as_of_date"] == "2026-02-14"
        assert result["prior_date"] is None
        assert result["bad_to_good_count"] is None
        assert result["good_to_bad_count"] is None
        assert result["bad_to_good_in_top60"] is None
        assert result["bad_to_good_in_top100"] is None
        assert result["top60_overlap"] is None
        assert result["top100_overlap"] is None
        # Distribution should still compute (single value edge case)
        assert result["p10_catalyst_days"] == 60.0

    def test_bad_to_good_transition(self, tmp_path):
        snap = tmp_path / "snapshots"
        # Prior: ACAD and FOLD have bad modes
        prior_rows = [
            _make_dev_row("ACAD", catalyst_mode="no_upcoming"),
            _make_dev_row("FOLD", catalyst_mode="missing"),
            _make_dev_row("VRTX", catalyst_mode="specific_days"),
        ]
        _write_rankings(snap, "2026-02-12", prior_rows)
        _write_rankings(snap, "2026-02-14", [_make_dev_row("X")])  # placeholder

        # Current: ACAD and FOLD now have good modes
        cur_rows = [
            _make_dev_row("ACAD", catalyst_mode="specific_days"),
            _make_dev_row("FOLD", catalyst_mode="blended_window"),
            _make_dev_row("VRTX", catalyst_mode="specific_days"),
        ]
        result = _compute_shadow_metrics(cur_rows, "2026-02-14", snap, None)

        assert result["bad_to_good_count"] == 2
        assert result["bad_to_good_tickers"] == ["ACAD", "FOLD"]
        assert result["good_to_bad_count"] == 0

    def test_good_to_bad_transition(self, tmp_path):
        snap = tmp_path / "snapshots"
        prior_rows = [
            _make_dev_row("SGEN", catalyst_mode="specific_days"),
        ]
        _write_rankings(snap, "2026-02-12", prior_rows)
        _write_rankings(snap, "2026-02-14", [_make_dev_row("X")])

        cur_rows = [
            _make_dev_row("SGEN", catalyst_mode="no_upcoming"),
        ]
        result = _compute_shadow_metrics(cur_rows, "2026-02-14", snap, None)

        assert result["good_to_bad_count"] == 1
        assert result["good_to_bad_tickers"] == ["SGEN"]
        assert result["bad_to_good_count"] == 0

    def test_commercial_tickers_excluded_from_transitions(self, tmp_path):
        snap = tmp_path / "snapshots"
        prior_rows = [
            _make_comm_row("BIGCO", catalyst_mode="no_upcoming"),
        ]
        # Need tier_dev in header for prior detection
        prior_rows[0]["tier_dev"] = ""
        _write_rankings(snap, "2026-02-12", prior_rows)
        _write_rankings(snap, "2026-02-14", [_make_dev_row("X")])

        cur_rows = [
            _make_comm_row("BIGCO", catalyst_mode="specific_days"),
        ]
        cur_rows[0]["tier_dev"] = ""
        result = _compute_shadow_metrics(cur_rows, "2026-02-14", snap, None)

        assert result["bad_to_good_count"] == 0
        assert result["good_to_bad_count"] == 0

    def test_new_tickers_not_counted_in_transitions(self, tmp_path):
        snap = tmp_path / "snapshots"
        prior_rows = [_make_dev_row("OLD")]
        _write_rankings(snap, "2026-02-12", prior_rows)
        _write_rankings(snap, "2026-02-14", [_make_dev_row("X")])

        cur_rows = [
            _make_dev_row("OLD", catalyst_mode="specific_days"),
            _make_dev_row("NEW", catalyst_mode="specific_days"),
        ]
        result = _compute_shadow_metrics(cur_rows, "2026-02-14", snap, None)

        # NEW shouldn't be counted (not in prior)
        assert result["bad_to_good_count"] == 0
        assert result["good_to_bad_count"] == 0

    def test_distribution_metrics_known_data(self, tmp_path):
        snap = tmp_path / "snapshots"
        rows = [
            _make_dev_row("A", catalyst_days="10", catalyst_mode="specific_days"),
            _make_dev_row("B", catalyst_days="30", catalyst_mode="no_upcoming"),
            _make_dev_row("C", catalyst_days="50", catalyst_mode="specific_days"),
            _make_dev_row("D", catalyst_days="90", catalyst_mode="blended_window"),
            _make_dev_row("E", catalyst_days="150", catalyst_mode="specific_days"),
            _make_dev_row("F", catalyst_days="200", catalyst_mode="missing"),
        ]
        _write_rankings(snap, "2026-02-14", rows)

        result = _compute_shadow_metrics(rows, "2026-02-14", snap, None)

        # median of good-mode: A(10), C(50), D(90), E(150) → median=(50+90)/2=70
        assert result["median_catalyst_days_good"] == 70.0
        # p10/p50/p90 from all 6 values: 10,30,50,90,150,200
        assert result["p10_catalyst_days"] is not None
        assert result["p50_catalyst_days"] is not None
        assert result["p90_catalyst_days"] is not None
        assert result["p10_catalyst_days"] < result["p50_catalyst_days"]
        assert result["p50_catalyst_days"] < result["p90_catalyst_days"]

    def test_empty_catalyst_days_all_none(self, tmp_path):
        snap = tmp_path / "snapshots"
        rows = [_make_dev_row("A", catalyst_days="")]
        _write_rankings(snap, "2026-02-14", rows)

        result = _compute_shadow_metrics(rows, "2026-02-14", snap, None)

        assert result["median_catalyst_days_good"] is None
        assert result["p10_catalyst_days"] is None
        assert result["p50_catalyst_days"] is None
        assert result["p90_catalyst_days"] is None

    def test_a_tier_count_dev_only(self, tmp_path):
        snap = tmp_path / "snapshots"
        rows = [
            _make_dev_row("A1", tier_dev="A"),
            _make_dev_row("A2", tier_dev="A"),
            _make_dev_row("B1", tier_dev="B"),
            _make_comm_row("COMM"),
        ]
        rows[3]["tier_dev"] = "A"  # commercial A shouldn't count
        _write_rankings(snap, "2026-02-14", rows)

        result = _compute_shadow_metrics(rows, "2026-02-14", snap, None)
        assert result["A_tier_count"] == 2

    def test_jaccard_overlap_known_sets(self, tmp_path):
        snap = tmp_path / "snapshots"
        # Prior: tickers T01..T05 ranked 1..5
        prior_rows = [_make_dev_row(f"T{i:02d}", composite_rank=i) for i in range(1, 6)]
        _write_rankings(snap, "2026-02-12", prior_rows)

        # Current: T03..T07 ranked 1..5
        cur_rows = [_make_dev_row(f"T{i:02d}", composite_rank=i - 2) for i in range(3, 8)]
        _write_rankings(snap, "2026-02-14", cur_rows)

        result = _compute_shadow_metrics(cur_rows, "2026-02-14", snap, None)

        # Both sets have 5 elements, top-60 gets all 5 each
        # Intersection: T03,T04,T05 (3), Union: T01..T07 (7)
        assert result["top60_overlap"] == round(3 / 7, 4)

    def test_bad_to_good_in_top_n(self, tmp_path):
        snap = tmp_path / "snapshots"
        # Build 100 filler tickers ranked 1-100 so FOLD at rank=200 is outside top-100
        filler_prior = [
            _make_dev_row(f"F{i:03d}", composite_rank=i, catalyst_mode="specific_days")
            for i in range(1, 101)
        ]
        # Prior: ACAD(rank=10) bad, QRST(rank=80) bad, FOLD(rank=200) bad
        prior_rows = filler_prior + [
            _make_dev_row("ACAD", composite_rank=10, catalyst_mode="no_upcoming"),
            _make_dev_row("QRST", composite_rank=80, catalyst_mode="missing"),
            _make_dev_row("FOLD", composite_rank=200, catalyst_mode="missing"),
        ]
        _write_rankings(snap, "2026-02-12", prior_rows)
        _write_rankings(snap, "2026-02-14", [_make_dev_row("X")])

        # Current: same fillers + all three flip to good
        filler_cur = [
            _make_dev_row(f"F{i:03d}", composite_rank=i, catalyst_mode="specific_days")
            for i in range(1, 101)
        ]
        cur_rows = filler_cur + [
            _make_dev_row("ACAD", composite_rank=10, catalyst_mode="specific_days"),
            _make_dev_row("QRST", composite_rank=80, catalyst_mode="blended_window"),
            _make_dev_row("FOLD", composite_rank=200, catalyst_mode="blended_window"),
        ]
        result = _compute_shadow_metrics(cur_rows, "2026-02-14", snap, None)

        assert result["bad_to_good_count"] == 3
        assert result["bad_to_good_in_top60"] == 1   # ACAD (rank 10)
        assert result["bad_to_good_in_top100"] == 2   # ACAD + QRST (rank 80)

    def test_source_attribution_extraction(self, tmp_path):
        snap = tmp_path / "snapshots"
        rows = [_make_dev_row("A")]
        _write_rankings(snap, "2026-02-14", rows)

        source_mix = {
            "by_source": {
                "SEC_8K_FILING": 206,
                "CTGOV_CALENDAR": 1084,
                "FDA_CALENDAR": 8,
                "FDA_ADCOM_CALENDAR": 2,
                "FEDERAL_REGISTER": 1,
            }
        }
        result = _compute_shadow_metrics(rows, "2026-02-14", snap, source_mix)

        assert result["sec_8k_events"] == 206
        assert result["ctgov_events"] == 1084
        assert result["fda_events"] == 11  # 8 + 2 + 1


# ===================================================================
# TestIntegration
# ===================================================================

class TestIntegration:
    """Integration tests for shadow metrics in save_validation_snapshot."""

    def test_shadow_metrics_json_written(self, tmp_path):
        """Minimal save_validation_snapshot produces catalyst_shadow_metrics.json."""
        from run_screen import save_validation_snapshot, DEFAULT_RULESET

        snap_dir = tmp_path / "snapshots"
        results = {
            "module_5_composite": {
                "ranked_securities": [
                    {
                        "ticker": "AAPL",
                        "composite_rank": 1,
                        "composite_score": 0.8,
                        "score_rank_pct": 0.99,
                        "score_z": 1.5,
                        "stage_bucket": "late",
                        "market_cap_bucket": "large",
                        "severity": "low",
                        "momentum_signal": {},
                        "valuation_signal": {},
                        "component_scores": [
                            {"name": "catalyst", "normalized": 0.5},
                            {"name": "smart_money", "normalized": 0.4},
                            {"name": "clinical", "normalized": 0.6},
                            {"name": "financial", "normalized": 0.7},
                        ],
                    }
                ],
            },
            "company_archetypes": {"AAPL": "drug_developer"},
        }

        path = save_validation_snapshot(
            snapshot_dir=snap_dir,
            as_of_date="2026-02-14",
            results=results,
            version="test",
        )
        assert path is not None

        shadow_path = path / "catalyst_shadow_metrics.json"
        assert shadow_path.exists()

        data = json.loads(shadow_path.read_text())
        assert data["as_of_date"] == "2026-02-14"
        assert data["prior_date"] is None
        assert "A_tier_count" in data

    def test_computation_failure_doesnt_break_snapshot(self, tmp_path, monkeypatch):
        """If _compute_shadow_metrics raises, rankings.csv is still written."""
        from run_screen import save_validation_snapshot

        snap_dir = tmp_path / "snapshots"
        results = {
            "module_5_composite": {
                "ranked_securities": [
                    {
                        "ticker": "TST",
                        "composite_rank": 1,
                        "composite_score": 0.5,
                        "score_rank_pct": 0.5,
                        "score_z": 0.0,
                        "stage_bucket": "mid",
                        "market_cap_bucket": "small",
                        "severity": "medium",
                        "momentum_signal": {},
                        "valuation_signal": {},
                        "component_scores": [],
                    }
                ],
            },
            "company_archetypes": {"TST": "drug_developer"},
        }

        # Patch _compute_shadow_metrics to raise
        import run_screen
        monkeypatch.setattr(
            run_screen, "_compute_shadow_metrics",
            lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        path = save_validation_snapshot(
            snapshot_dir=snap_dir,
            as_of_date="2026-02-14",
            results=results,
            version="test",
        )
        assert path is not None
        # rankings.csv should still exist
        assert (path / "rankings.csv").exists()
        # shadow metrics file should NOT exist (write failed)
        assert not (path / "catalyst_shadow_metrics.json").exists()
