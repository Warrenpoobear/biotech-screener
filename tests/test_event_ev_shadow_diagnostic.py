"""Tests for tools/event_ev_shadow_diagnostic.py."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tools.event_ev_shadow_diagnostic import _miscal_label, _phase_bucket, run


class TestPhaseBucket:
    def test_phase3(self):
        assert _phase_bucket("3") == "phase3"
        assert _phase_bucket("3.0") == "phase3"

    def test_phase2(self):
        assert _phase_bucket("2") == "phase2"
        assert _phase_bucket("2.5") == "phase2"

    def test_early(self):
        assert _phase_bucket("1") == "early"
        assert _phase_bucket("") == "early"
        assert _phase_bucket("N/A") == "early"


class TestMiscalLabel:
    def test_overpriced(self):
        # 40% above base → OVERPRICED (threshold is >40%)
        assert _miscal_label(35.0 * 1.45, 35.0) == "OVERPRICED"

    def test_underpriced(self):
        # 30% below base → UNDERPRICED (threshold is <25%)
        assert _miscal_label(35.0 * 0.70, 35.0) == "UNDERPRICED"

    def test_in_range(self):
        assert _miscal_label(35.0, 35.0) == "IN_RANGE"
        assert _miscal_label(40.0, 35.0) == "IN_RANGE"  # ~14% over, within range


class TestRun:
    def _write_rankings(self, tmp_path: Path, rows: list[dict]) -> Path:
        snap = tmp_path / "2026-06-23"
        snap.mkdir()
        fieldnames = [
            "ticker",
            "catalyst_family",
            "lead_program_phase",
            "priced_move_pct",
            "implied_event_move",
            "catalyst_date_precision",
            "base_rate_gap_score",
            "ees_v2_score",
            "ees_v3_score",
        ]
        csv_path = snap / "rankings.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k, "") for k in fieldnames})
        return snap

    def test_outputs_created(self, tmp_path):
        snap = self._write_rankings(
            tmp_path,
            [
                {
                    "ticker": "COGT",
                    "catalyst_family": "CLINICAL",
                    "lead_program_phase": "3",
                    "priced_move_pct": "35.0",
                    "catalyst_date_precision": "DAY",
                },
                {
                    "ticker": "DNTH",
                    "catalyst_family": "REGULATORY",
                    "lead_program_phase": "3",
                    "priced_move_pct": "19.0",
                    "catalyst_date_precision": "WEEK",
                },
            ],
        )
        run(snap)
        assert (snap / "event_ev_shadow_diagnostic.json").exists()
        assert (snap / "event_ev_shadow_diagnostic.md").exists()

    def test_json_structure(self, tmp_path):
        snap = self._write_rankings(
            tmp_path,
            [
                {
                    "ticker": "COGT",
                    "catalyst_family": "CLINICAL",
                    "lead_program_phase": "3",
                    "priced_move_pct": "35.0",
                    "catalyst_date_precision": "DAY",
                },
            ],
        )
        run(snap)
        data = json.loads((snap / "event_ev_shadow_diagnostic.json").read_text())
        assert data["label"] == "EVENT_EV_SHADOW_DIAGNOSTIC_ONLY_NO_ALPHA_PROMOTION"
        assert data["governance"]["read_only"] is True
        assert data["governance"]["no_alpha_promotion"] is True
        assert "universe_summary" in data
        assert "cohort_summaries" in data

    def test_no_catalyst_skipped(self, tmp_path):
        snap = self._write_rankings(
            tmp_path,
            [
                {
                    "ticker": "COGT",
                    "catalyst_family": "NO_CATALYST",
                    "lead_program_phase": "3",
                    "priced_move_pct": "35.0",
                },
                {"ticker": "DNTH", "catalyst_family": "CLINICAL", "lead_program_phase": "3", "priced_move_pct": "35.0"},
            ],
        )
        run(snap)
        data = json.loads((snap / "event_ev_shadow_diagnostic.json").read_text())
        assert data["universe_summary"]["n_with_priced_move"] == 1

    def test_missing_priced_move_skipped(self, tmp_path):
        snap = self._write_rankings(
            tmp_path,
            [
                {"ticker": "COGT", "catalyst_family": "CLINICAL", "lead_program_phase": "3", "priced_move_pct": ""},
                {"ticker": "DNTH", "catalyst_family": "CLINICAL", "lead_program_phase": "3", "priced_move_pct": "35.0"},
            ],
        )
        run(snap)
        data = json.loads((snap / "event_ev_shadow_diagnostic.json").read_text())
        assert data["universe_summary"]["n_with_priced_move"] == 1

    def test_overpriced_flagged(self, tmp_path):
        # CLINICAL|phase3 base p50 = 35%; 60% implied is +71% above → OVERPRICED
        snap = self._write_rankings(
            tmp_path,
            [
                {"ticker": "XTST", "catalyst_family": "CLINICAL", "lead_program_phase": "3", "priced_move_pct": "60.0"},
            ],
        )
        run(snap)
        data = json.loads((snap / "event_ev_shadow_diagnostic.json").read_text())
        assert data["universe_summary"]["n_overpriced"] == 1
        assert len(data["overpriced_names"]) == 1
        assert data["overpriced_names"][0]["ticker"] == "XTST"
