#!/usr/bin/env python3
"""Regression tests for tools/weekly_health_packet.py.

Covers the two bugs fixed at initial commit:
  1. Fresh-start runs (prior_active=0) → turnover None, not 2000%
  2. "unknown" cache health does NOT trigger action item
  3. Trailing 4-week avg skips fresh-start snapshots
  4. Turnover spike label shows ">2.5x"
  5. Explicit cache WARN/FAIL does trigger action item
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from weekly_health_packet import (
    _action_items,
    _compute_turnover,
    _turnover_from_delta,
    build_health_packet,
    render_markdown,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_delta_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["ticker", "in_current", "in_prior", "weight_current", "weight_prior",
                  "weight_delta", "tier_current", "tier_prior", "rank_current", "rank_prior"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_portfolio_positions(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["ticker", "actionable_rank", "eligible", "target_weight_pct",
                  "tier_any", "size_band", "risk_flags"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _fresh_start_delta_rows(n: int = 20) -> List[Dict]:
    """All entries, no exits, no prior — fresh start."""
    return [{"ticker": f"T{i:03d}", "in_current": "1", "in_prior": "0",
             "weight_current": "5.0", "weight_prior": "0.0", "weight_delta": "5.0",
             "tier_current": "A", "tier_prior": "", "rank_current": str(i), "rank_prior": ""}
            for i in range(1, n + 1)]


def _normal_delta_rows(entries: int, exits: int, prior_active: int) -> List[Dict]:
    """Synthetic rows for a run with a prior portfolio."""
    rows = []
    idx = 1
    for _ in range(entries):
        rows.append({"ticker": f"IN{idx:03d}", "in_current": "1", "in_prior": "0",
                     "weight_current": "5.0", "weight_prior": "0.0", "weight_delta": "5.0",
                     "tier_current": "A", "tier_prior": "", "rank_current": str(idx), "rank_prior": ""})
        idx += 1
    for _ in range(exits):
        rows.append({"ticker": f"EX{idx:03d}", "in_current": "0", "in_prior": "1",
                     "weight_current": "0.0", "weight_prior": "5.0", "weight_delta": "-5.0",
                     "tier_current": "", "tier_prior": "A", "rank_current": "", "rank_prior": str(idx)})
        idx += 1
    unchanged = prior_active - exits
    for _ in range(unchanged):
        rows.append({"ticker": f"UK{idx:03d}", "in_current": "1", "in_prior": "1",
                     "weight_current": "5.0", "weight_prior": "5.0", "weight_delta": "0.0",
                     "tier_current": "A", "tier_prior": "A", "rank_current": str(idx), "rank_prior": str(idx)})
        idx += 1
    return rows


def _minimal_snapshot(snap_root: Path, as_of_date: str, *,
                       delta_rows: Optional[List[Dict]] = None,
                       cache_status: str = "ok") -> Path:
    """Write a minimal snapshot directory under snap_root/as_of_date."""
    snap = snap_root / as_of_date
    snap.mkdir(parents=True)

    if delta_rows is not None:
        _write_delta_csv(snap / "phase2_run_delta.csv", delta_rows)

    _write_portfolio_positions(snap / "portfolio_positions.csv", [
        {"ticker": f"T{i:03d}", "actionable_rank": str(i), "eligible": "1",
         "target_weight_pct": "5.0", "tier_any": "A", "size_band": "M", "risk_flags": ""}
        for i in range(1, 5)
    ])

    (snap / "run_manifest.json").write_text(json.dumps({
        "overall_status": "PASS",
        "git": {"commit_sha": "abc12345def"},
        "ruleset": {"ruleset_hash": "82982998", "ruleset_version": "1.8.3"},
        "gates": [{"name": "drift_monitoring", "status": "PASS"}],
    }))
    (snap / "drift_report.json").write_text(json.dumps({
        "status": "PASS",
        "metrics": {"top20_overlap_pct": 95.0, "top60_overlap_pct": 90.0, "rank_spearman_rho": 0.998},
        "warn_reasons": [],
    }))
    (snap / "ruleset_health.json").write_text(json.dumps({
        "status": "OK",
        "consecutive_warn_days": 0,
        "recommend_rollback": False,
        "days_since_promotion": 3,
        "today": {"top60_overlap_pct": 90.0, "max_rank_shift": 1.5},
    }))
    (snap / "cache_health.json").write_text(json.dumps({
        "overall_status": cache_status,
        "sec8k": {"count": 100, "status": "ok"},
        "ctgov": {"count": 500, "status": "ok"},
        "degraded_run": False,
    }))
    (snap / "decision_portfolio.json").write_text(json.dumps({"n_eligible": 195}))
    (snap / "health_exposure_metrics.json").write_text(json.dumps({
        "metrics": {"high_vol_or_beta_pct": 20.0, "headwind_pct": 5.0},
    }))
    return snap


# ---------------------------------------------------------------------------
# _turnover_from_delta
# ---------------------------------------------------------------------------

class TestTurnoverFromDelta:
    def test_fresh_start_returns_none(self):
        rows = _fresh_start_delta_rows(20)
        assert _turnover_from_delta(rows) is None

    def test_empty_delta_returns_none(self):
        assert _turnover_from_delta([]) is None

    def test_normal_turnover(self):
        # 2 entries, 2 exits, 20 prior → (4/40)*100 = 10%
        rows = _normal_delta_rows(entries=2, exits=2, prior_active=20)
        result = _turnover_from_delta(rows)
        assert result == pytest.approx(10.0)

    def test_zero_turnover(self):
        rows = _normal_delta_rows(entries=0, exits=0, prior_active=20)
        assert _turnover_from_delta(rows) == 0.0

    def test_full_turnover(self):
        # 20 entries, 20 exits, 20 prior → 100%
        rows = _normal_delta_rows(entries=20, exits=20, prior_active=20)
        assert _turnover_from_delta(rows) == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# _compute_turnover (fresh-start + trailing avg)
# ---------------------------------------------------------------------------

class TestComputeTurnover:
    def _write_snap(self, root: Path, date: str, rows: List[Dict]) -> None:
        p = root / date
        p.mkdir(parents=True)
        _write_delta_csv(p / "phase2_run_delta.csv", rows)

    def test_fresh_start_turnover_is_none(self, tmp_path, monkeypatch):
        import weekly_health_packet as whp
        snap_root = tmp_path / "snapshots"
        snap_root.mkdir()
        self._write_snap(snap_root, "2026-03-05", _fresh_start_delta_rows(20))
        monkeypatch.setattr(whp, "SNAPSHOTS_ROOT", snap_root)
        result = _compute_turnover(snap_root / "2026-03-05", "2026-03-05")
        assert result["turnover_pct"] is None
        assert result["fresh_start"] is True

    def test_trailing_avg_skips_fresh_start(self, tmp_path, monkeypatch):
        """Fresh-start prior snapshots should not contribute to trailing avg."""
        import weekly_health_packet as whp
        snap_root = tmp_path / "snapshots"
        snap_root.mkdir()
        # 4 fresh-start priors + 1 real (10%) prior
        self._write_snap(snap_root, "2026-03-01", _fresh_start_delta_rows())
        self._write_snap(snap_root, "2026-03-02", _fresh_start_delta_rows())
        self._write_snap(snap_root, "2026-03-03", _fresh_start_delta_rows())
        self._write_snap(snap_root, "2026-03-04", _normal_delta_rows(2, 2, 20))  # 10%
        self._write_snap(snap_root, "2026-03-05", _normal_delta_rows(1, 1, 20))  # 5%
        monkeypatch.setattr(whp, "SNAPSHOTS_ROOT", snap_root)
        result = _compute_turnover(snap_root / "2026-03-05", "2026-03-05")
        # Only 1 valid prior (10%) → trailing avg = 10%
        assert result["trailing_4w_avg_pct"] == pytest.approx(10.0)
        assert result["trailing_n"] == 1

    def test_trailing_avg_up_to_four(self, tmp_path, monkeypatch):
        import weekly_health_packet as whp
        snap_root = tmp_path / "snapshots"
        snap_root.mkdir()
        for i in range(1, 5):
            self._write_snap(snap_root, f"2026-03-0{i}", _normal_delta_rows(2, 2, 20))  # all 10%
        self._write_snap(snap_root, "2026-03-05", _normal_delta_rows(0, 0, 20))
        monkeypatch.setattr(whp, "SNAPSHOTS_ROOT", snap_root)
        result = _compute_turnover(snap_root / "2026-03-05", "2026-03-05")
        assert result["trailing_n"] == 4
        assert result["trailing_4w_avg_pct"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# _action_items — cache health
# ---------------------------------------------------------------------------

class TestActionItemsCacheHealth:
    def _run(self, cache_status: str):
        return _action_items(
            gates=[],
            drift={"status": "PASS", "top20_overlap_pct": 95, "top60_overlap_pct": 90,
                   "warn_reasons": []},
            ruleset_health={"consecutive_warn_days": 0, "recommend_rollback": False},
            turnover={"turnover_pct": 5.0, "trailing_4w_avg_pct": 5.0,
                      "fresh_start": False, "entries": 1, "exits": 1, "trailing_n": 4},
            cache={"overall_status": cache_status, "degraded_run": False},
        )

    def test_unknown_cache_no_action_item(self):
        """Old snapshots with unknown cache status must not create noise."""
        items = self._run("unknown")
        cache_items = [i for i in items if i["type"] == "cache_health"]
        assert cache_items == []

    def test_none_cache_no_action_item(self):
        items = self._run(None)
        assert not any(i["type"] == "cache_health" for i in items)

    def test_ok_cache_no_action_item(self):
        items = self._run("ok")
        assert not any(i["type"] == "cache_health" for i in items)

    def test_warn_cache_triggers_action(self):
        items = self._run("warn")
        assert any(i["type"] == "cache_health" for i in items)

    def test_fail_cache_triggers_action(self):
        items = self._run("fail")
        assert any(i["type"] == "cache_health" for i in items)


# ---------------------------------------------------------------------------
# _action_items — turnover spike label
# ---------------------------------------------------------------------------

class TestTurnoverSpikeLabel:
    def _spike_items(self, tc: float, t4: float):
        return _action_items(
            gates=[],
            drift={"status": "PASS", "top20_overlap_pct": 95, "top60_overlap_pct": 90,
                   "warn_reasons": []},
            ruleset_health={"consecutive_warn_days": 0, "recommend_rollback": False},
            turnover={"turnover_pct": tc, "trailing_4w_avg_pct": t4,
                      "fresh_start": False, "entries": 5, "exits": 5, "trailing_n": 4},
            cache={"overall_status": "ok", "degraded_run": False},
        )

    def test_spike_label_shows_2_5x(self):
        items = self._spike_items(tc=60.0, t4=20.0)  # 3x spike
        spike_items = [i for i in items if i["type"] == "turnover_spike"]
        assert len(spike_items) == 1
        assert "2.5x" in spike_items[0]["detail"]

    def test_no_spike_below_threshold(self):
        items = self._spike_items(tc=40.0, t4=20.0)  # 2x, below 2.5x
        assert not any(i["type"] == "turnover_spike" for i in items)

    def test_no_spike_on_fresh_start(self):
        items = _action_items(
            gates=[],
            drift={"status": "PASS", "top20_overlap_pct": 95, "top60_overlap_pct": 90,
                   "warn_reasons": []},
            ruleset_health={"consecutive_warn_days": 0, "recommend_rollback": False},
            turnover={"turnover_pct": None, "trailing_4w_avg_pct": 20.0,
                      "fresh_start": True, "entries": 20, "exits": 0, "trailing_n": 2},
            cache={"overall_status": "ok", "degraded_run": False},
        )
        assert not any(i["type"] == "turnover_spike" for i in items)


# ---------------------------------------------------------------------------
# render_markdown — fresh-start note in turnover section
# ---------------------------------------------------------------------------

class TestRenderMarkdown:
    def _make_packet(self, turnover_pct: Optional[float], fresh_start: bool) -> Dict:
        return {
            "schema": "health_packet.v1",
            "relaxed": False,
            "provenance": {
                "as_of_date": "2026-03-05",
                "generated_at": "2026-03-05T12:00:00+00:00",
                "git_sha": "abc12345",
                "ruleset_id": "82982998",
                "ruleset_version": "1.8.3",
                "snapshot_root": "/data/snapshots",
                "mode": "strict",
                "overall_status": "PASS",
            },
            "gates": {"pass_count": 24, "warn_count": 0, "fail_count": 0,
                      "warn_names": [], "fail_names": [], "fail_details": []},
            "preflight": None,
            "drift": {"status": "PASS", "top20_overlap_pct": 95.0, "top60_overlap_pct": 90.0,
                      "rank_spearman_rho": 0.998, "warn_reasons": []},
            "ruleset_health": {"status": "OK", "consecutive_warn_days": 0,
                               "recommend_rollback": False, "days_since_promotion": 3,
                               "top60_overlap_pct": 90.0, "max_rank_shift": 1.5},
            "cache": {"overall_status": "ok", "sec8k": {"count": 100, "status": "ok"},
                      "ctgov": {"count": 500, "status": "ok"}, "degraded_run": False},
            "turnover": {"turnover_pct": turnover_pct, "trailing_4w_avg_pct": 20.0,
                         "entries": 20, "exits": 0, "trailing_n": 2, "fresh_start": fresh_start},
            "portfolio": {"n_eligible": 195, "n_portfolio": 20, "weight_sum_pct": 100.0,
                          "top5_weight_pct": 30.0, "max_single_weight_pct": 6.0,
                          "high_vol_or_beta_pct": 20.0, "headwind_pct": 5.0, "top10": []},
            "action_items": [],
        }

    def test_fresh_start_shows_dash_and_note(self):
        md = render_markdown(self._make_packet(None, fresh_start=True))
        assert "fresh start" in md
        assert "—" in md  # em dash for None turnover

    def test_normal_turnover_shows_value(self):
        md = render_markdown(self._make_packet(10.0, fresh_start=False))
        assert "10.00%" in md
        assert "fresh start" not in md

    def test_relaxed_banner_present_when_relaxed(self):
        packet = self._make_packet(5.0, fresh_start=False)
        packet["relaxed"] = True
        packet["provenance"]["mode"] = "relaxed"
        md = render_markdown(packet)
        assert "RELAXED MODE" in md

    def test_no_relaxed_banner_in_strict(self):
        md = render_markdown(self._make_packet(5.0, fresh_start=False))
        assert "RELAXED MODE" not in md


# ---------------------------------------------------------------------------
# build_health_packet (integration — uses tmp snapshot dir)
# ---------------------------------------------------------------------------

class TestBuildHealthPacket:
    def test_fresh_start_snapshot(self, tmp_path, monkeypatch):
        import weekly_health_packet as whp
        snap_root = tmp_path / "snapshots"
        snap = _minimal_snapshot(snap_root, "2026-03-05",
                                 delta_rows=_fresh_start_delta_rows(20))
        monkeypatch.setattr(whp, "SNAPSHOTS_ROOT", snap_root)
        packet = build_health_packet("2026-03-05")
        assert packet["turnover"]["turnover_pct"] is None
        assert packet["turnover"]["fresh_start"] is True
        assert packet["schema"] == "health_packet.v1"

    def test_normal_snapshot(self, tmp_path, monkeypatch):
        import weekly_health_packet as whp
        snap_root = tmp_path / "snapshots"
        _minimal_snapshot(snap_root, "2026-03-05",
                          delta_rows=_normal_delta_rows(2, 2, 20))
        monkeypatch.setattr(whp, "SNAPSHOTS_ROOT", snap_root)
        packet = build_health_packet("2026-03-05")
        assert packet["turnover"]["turnover_pct"] == pytest.approx(10.0)
        assert packet["turnover"]["fresh_start"] is False

    def test_unknown_cache_no_action_item(self, tmp_path, monkeypatch):
        import weekly_health_packet as whp
        snap_root = tmp_path / "snapshots"
        _minimal_snapshot(snap_root, "2026-03-05",
                          delta_rows=_normal_delta_rows(0, 0, 20),
                          cache_status="unknown")
        monkeypatch.setattr(whp, "SNAPSHOTS_ROOT", snap_root)
        packet = build_health_packet("2026-03-05")
        cache_items = [i for i in packet["action_items"] if i["type"] == "cache_health"]
        assert cache_items == []

    def test_snapshot_not_found_raises(self, tmp_path, monkeypatch):
        import weekly_health_packet as whp
        monkeypatch.setattr(whp, "SNAPSHOTS_ROOT", tmp_path / "snapshots")
        with pytest.raises(FileNotFoundError):
            build_health_packet("2026-03-05")
