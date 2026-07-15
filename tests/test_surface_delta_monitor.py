"""Tests for tools/surface_delta_monitor.py."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.surface_delta_monitor import (
    IV_JUMP_REL_LARGE,
    IV_JUMP_REL_THRESHOLD,
    IV_JUMP_THRESHOLD,
    RR_FLIP_THRESHOLD,
    RR_FLIP_THRESHOLD_EXTREME,
    RR_MOVE_THRESHOLD_EXTREME,
    SKEW_SHIFT_THRESHOLD_EXTREME,
    _pick_thresholds,
    _sf,
    classify_iv_regime,
    compute_delta,
    format_briefing,
    load_diagnostics_csv,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _diag(
    atm_iv="0.50",
    rr_25d="0.05",
    skew="0.02",
    term_slope="0.05",
    has_data="1",
    use_for_judgment="YES",
    catalyst_days="30",
    catalyst_bucket="build_window",
    liquidity_state="active",
):
    """Build a minimal diagnostics row."""
    return {
        "opt_has_data": has_data,
        "opt_atm_iv": atm_iv,
        "opt_rr_25d": rr_25d,
        "opt_put_call_skew": skew,
        "opt_term_slope": term_slope,
        "opt_use_for_judgment": use_for_judgment,
        "opt_liquidity_state": liquidity_state,
        "catalyst_days": catalyst_days,
        "catalyst_bucket": catalyst_bucket,
    }


# ---------------------------------------------------------------------------
# _sf
# ---------------------------------------------------------------------------


class TestSafeFloat:
    def test_normal(self):
        assert _sf("1.23") == 1.23

    def test_empty(self):
        assert math.isnan(_sf(""))

    def test_none(self):
        assert math.isnan(_sf(None))

    def test_invalid(self):
        assert math.isnan(_sf("abc"))

    def test_numeric_passthrough(self):
        assert _sf(0.5) == 0.5


# ---------------------------------------------------------------------------
# classify_iv_regime
# ---------------------------------------------------------------------------


class TestClassifyIvRegime:
    def test_normal(self):
        assert classify_iv_regime(0.40) == "NORMAL"

    def test_elevated(self):
        assert classify_iv_regime(0.80) == "ELEVATED"

    def test_extreme(self):
        assert classify_iv_regime(3.00) == "EXTREME"

    def test_boundary_elevated(self):
        assert classify_iv_regime(0.60) == "ELEVATED"

    def test_boundary_extreme(self):
        assert classify_iv_regime(2.00) == "EXTREME"

    def test_nan(self):
        assert classify_iv_regime(float("nan")) == ""


# ---------------------------------------------------------------------------
# _pick_thresholds
# ---------------------------------------------------------------------------


class TestPickThresholds:
    def test_normal_regime(self):
        th = _pick_thresholds(0.40, 0.50)
        assert th["iv_jump"] == IV_JUMP_THRESHOLD
        assert th["rr_flip"] == RR_FLIP_THRESHOLD

    def test_elevated_regime(self):
        th = _pick_thresholds(0.80, 1.20)
        assert th["iv_jump"] == IV_JUMP_THRESHOLD  # still absolute
        assert th["rr_flip"] == RR_FLIP_THRESHOLD

    def test_extreme_regime(self):
        th = _pick_thresholds(3.00, 2.50)
        # Relative IV threshold: 30% of max(3.0, 2.5) = 0.9
        assert th["iv_jump"] == pytest.approx(IV_JUMP_REL_THRESHOLD * 3.00)
        assert th["iv_jump_large"] == pytest.approx(IV_JUMP_REL_LARGE * 3.00)
        assert th["rr_flip"] == RR_FLIP_THRESHOLD_EXTREME
        assert th["rr_move"] == RR_MOVE_THRESHOLD_EXTREME
        assert th["skew_shift"] == SKEW_SHIFT_THRESHOLD_EXTREME

    def test_one_extreme(self):
        """If either side is EXTREME, use EXTREME thresholds."""
        th = _pick_thresholds(0.50, 2.50)
        assert th["iv_jump"] == pytest.approx(IV_JUMP_REL_THRESHOLD * 2.50)

    def test_nan_uses_normal(self):
        th = _pick_thresholds(float("nan"), 0.50)
        assert th["iv_jump"] == IV_JUMP_THRESHOLD


# ---------------------------------------------------------------------------
# compute_delta — core logic
# ---------------------------------------------------------------------------


class TestComputeDelta:
    def test_no_change_returns_none(self):
        """Identical snapshots should produce no delta."""
        prior = _diag()
        current = _diag()
        result = compute_delta("TEST", prior, current)
        assert result is None

    def test_missing_data_prior(self):
        """Prior has no data → skip."""
        prior = _diag(has_data="0")
        current = _diag()
        assert compute_delta("TEST", prior, current) is None

    def test_missing_data_current(self):
        """Current has no data → skip."""
        prior = _diag()
        current = _diag(has_data="0")
        assert compute_delta("TEST", prior, current) is None

    def test_illiquid_both_sides_skipped(self):
        """Neither snapshot is judgment-grade → skip."""
        prior = _diag(use_for_judgment="NO")
        current = _diag(use_for_judgment="NO")
        assert compute_delta("TEST", prior, current) is None

    def test_one_side_judgment_ok(self):
        """One side judgment-grade is enough to proceed."""
        prior = _diag(atm_iv="0.30", use_for_judgment="YES")
        current = _diag(atm_iv="0.55", use_for_judgment="NO")
        result = compute_delta("TEST", prior, current)
        assert result is not None
        assert "iv_jump_large_up" in result["flags"]

    def test_iv_jump_watch(self):
        """IV change >= 10pp but < 20pp → watch."""
        prior = _diag(atm_iv="0.40")
        current = _diag(atm_iv="0.52")
        result = compute_delta("TEST", prior, current)
        assert result is not None
        assert "iv_jump_up" in result["flags"]
        assert result["severity"] == "watch"

    def test_iv_jump_large_alert(self):
        """IV change >= 20pp → alert."""
        prior = _diag(atm_iv="0.30")
        current = _diag(atm_iv="0.55")
        result = compute_delta("TEST", prior, current)
        assert result is not None
        assert "iv_jump_large_up" in result["flags"]
        assert result["severity"] == "alert"

    def test_iv_jump_down(self):
        prior = _diag(atm_iv="0.55")
        current = _diag(atm_iv="0.30")
        result = compute_delta("TEST", prior, current)
        assert result is not None
        assert "iv_jump_large_down" in result["flags"]

    def test_extreme_iv_normal_swing_suppressed(self):
        """In EXTREME regime, a 30pp swing (< 30% of 300%) should not flag."""
        prior = _diag(atm_iv="3.00")
        current = _diag(atm_iv="3.30")
        result = compute_delta("TEST", prior, current)
        assert result is None

    def test_extreme_iv_large_swing_flags(self):
        """In EXTREME regime, a >50% swing should flag as large."""
        prior = _diag(atm_iv="3.00")
        current = _diag(atm_iv="6.00")  # 100% increase, threshold is 50% of max=6.0=3.0
        result = compute_delta("TEST", prior, current)
        assert result is not None
        assert "iv_jump_large_up" in result["flags"]

    def test_regime_transition(self):
        prior = _diag(atm_iv="0.55")
        current = _diag(atm_iv="0.65")
        result = compute_delta("TEST", prior, current)
        assert result is not None
        assert "regime_normal_to_elevated" in result["flags"]

    def test_rr_flip_bearish(self):
        """RR flips from positive (call skew) to negative (put skew)."""
        prior = _diag(rr_25d="0.05")
        current = _diag(rr_25d="-0.05")
        result = compute_delta("TEST", prior, current)
        assert result is not None
        assert "rr_flipped_bearish" in result["flags"]
        assert result["severity"] == "alert"

    def test_rr_flip_bullish(self):
        prior = _diag(rr_25d="-0.05")
        current = _diag(rr_25d="0.05")
        result = compute_delta("TEST", prior, current)
        assert result is not None
        assert "rr_flipped_bullish" in result["flags"]

    def test_rr_move_no_flip(self):
        """Large RR move without sign change → watch."""
        prior = _diag(rr_25d="0.10")
        current = _diag(rr_25d="0.16")
        result = compute_delta("TEST", prior, current)
        assert result is not None
        assert "rr_move_bullish" in result["flags"]
        assert result["severity"] == "watch"

    def test_rr_extreme_suppressed(self):
        """EXTREME-regime RR jitter should not flag."""
        prior = _diag(atm_iv="3.00", rr_25d="0.10")
        current = _diag(atm_iv="3.00", rr_25d="-0.05")
        # Flip but only 0.15 magnitude, < 0.20 EXTREME threshold
        result = compute_delta("TEST", prior, current)
        # Should not flag rr_flip because below EXTREME threshold
        if result is not None:
            assert "rr_flipped_bearish" not in result["flags"]

    def test_skew_shift(self):
        prior = _diag(skew="0.02")
        current = _diag(skew="0.15")
        result = compute_delta("TEST", prior, current)
        assert result is not None
        assert "skew_shift_puts_bid" in result["flags"]

    def test_skew_shift_extreme_suppressed(self):
        """EXTREME-regime small skew change suppressed."""
        prior = _diag(atm_iv="3.00", skew="0.10")
        current = _diag(atm_iv="3.00", skew="0.30")
        # 0.20 change < 0.40 EXTREME threshold
        result = compute_delta("TEST", prior, current)
        if result is not None:
            assert not any("skew_shift" in f for f in result["flags"])

    def test_term_entered_backwardation(self):
        prior = _diag(term_slope="0.05")
        current = _diag(term_slope="-0.05")
        result = compute_delta("TEST", prior, current)
        assert result is not None
        assert "term_entered_backwardation" in result["flags"]

    def test_term_exited_backwardation(self):
        prior = _diag(term_slope="-0.10")
        current = _diag(term_slope="0.05")
        result = compute_delta("TEST", prior, current)
        assert result is not None
        assert "term_exited_backwardation" in result["flags"]

    def test_multiple_flags(self):
        """Multiple signals → highest severity wins."""
        prior = _diag(atm_iv="0.30", rr_25d="0.05", term_slope="0.10")
        current = _diag(atm_iv="0.55", rr_25d="-0.05", term_slope="-0.05")
        result = compute_delta("TEST", prior, current)
        assert result is not None
        assert result["severity"] == "alert"
        assert result["n_flags"] >= 3

    def test_catalyst_context_carried(self):
        prior = _diag(atm_iv="0.30", catalyst_days="15", catalyst_bucket="binary_now")
        current = _diag(atm_iv="0.55", catalyst_days="15", catalyst_bucket="binary_now")
        result = compute_delta("TEST", prior, current)
        assert result["catalyst_days"] == "15"
        assert result["catalyst_bucket"] == "binary_now"

    def test_empty_rr_skipped(self):
        """Missing RR on one side should not crash."""
        prior = _diag(rr_25d="")
        current = _diag(rr_25d="0.10", atm_iv="0.55")
        result = compute_delta("TEST", prior, current)
        # May or may not flag (IV change), but should not crash
        if result is not None:
            assert "rr_flipped_bearish" not in result["flags"]
            assert "rr_flipped_bullish" not in result["flags"]


# ---------------------------------------------------------------------------
# load_diagnostics_csv
# ---------------------------------------------------------------------------


class TestLoadDiagnosticsCsv:
    def test_loads_csv(self, tmp_path):
        csv_path = tmp_path / "options_diagnostics.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ticker", "opt_has_data", "opt_atm_iv", "opt_rr_25d"])
            w.writerow(["AAPL", "1", "0.25", "0.03"])
            w.writerow(["BIIB", "1", "0.80", "-0.05"])
        result = load_diagnostics_csv(csv_path)
        assert len(result) == 2
        assert "AAPL" in result
        assert result["BIIB"]["opt_atm_iv"] == "0.80"

    def test_missing_file(self, tmp_path):
        result = load_diagnostics_csv(tmp_path / "nonexistent.csv")
        assert result == {}

    def test_deduplicates_on_ticker(self, tmp_path):
        csv_path = tmp_path / "options_diagnostics.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ticker", "opt_has_data"])
            w.writerow(["AAPL", "1"])
            w.writerow(["aapl", "0"])  # duplicate, case-insensitive
        result = load_diagnostics_csv(csv_path)
        assert len(result) == 1
        assert "AAPL" in result


# ---------------------------------------------------------------------------
# format_briefing
# ---------------------------------------------------------------------------


class TestFormatBriefing:
    def test_no_deltas(self):
        text = format_briefing([], "2026-03-25", "2026-03-26", 100, 100, 95, False)
        assert "No significant surface shifts" in text

    def test_has_alerts(self):
        deltas = [
            {
                "ticker": "PVLA",
                "severity": "alert",
                "n_flags": 1,
                "flags": ["rr_flipped_bearish"],
                "catalyst_days": "5",
                "catalyst_bucket": "binary_now",
                "prior_rr_25d": 0.05,
                "current_rr_25d": -0.03,
                "rr_25d_change": -0.08,
            }
        ]
        text = format_briefing(deltas, "2026-03-25", "2026-03-26", 100, 100, 95, True)
        assert "ALERT" in text
        assert "PVLA" in text
        assert "flipped bearish" in text

    def test_watch_section(self):
        deltas = [
            {
                "ticker": "CELC",
                "severity": "watch",
                "n_flags": 1,
                "flags": ["iv_jump_up"],
                "catalyst_days": "10",
                "catalyst_bucket": "binary_now",
                "atm_iv_change": 0.15,
            }
        ]
        text = format_briefing(deltas, "2026-03-25", "2026-03-26", 100, 100, 95, False)
        assert "WATCH" in text
        assert "CELC" in text


# ---------------------------------------------------------------------------
# Integration: run() with snapshot data
# ---------------------------------------------------------------------------


class TestRunIntegration:
    """Test run() end-to-end using temp snapshot directories."""

    def _write_diag_csv(self, snap_dir: Path, rows: list):
        snap_dir.mkdir(parents=True, exist_ok=True)
        csv_path = snap_dir / "options_diagnostics.csv"
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    def test_snapshot_vs_snapshot(self, tmp_path, monkeypatch):
        from tools import surface_delta_monitor as mod

        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)

        snap_root = tmp_path / "data" / "snapshots"

        prior_rows = [
            {
                "ticker": "AAPL",
                "opt_has_data": "1",
                "opt_atm_iv": "0.30",
                "opt_rr_25d": "0.05",
                "opt_put_call_skew": "0.01",
                "opt_term_slope": "0.05",
                "opt_use_for_judgment": "YES",
                "opt_liquidity_state": "active",
                "catalyst_days": "30",
                "catalyst_bucket": "build_window",
            },
            {
                "ticker": "BIIB",
                "opt_has_data": "1",
                "opt_atm_iv": "0.45",
                "opt_rr_25d": "0.03",
                "opt_put_call_skew": "-0.02",
                "opt_term_slope": "0.08",
                "opt_use_for_judgment": "YES",
                "opt_liquidity_state": "active",
                "catalyst_days": "10",
                "catalyst_bucket": "binary_now",
            },
        ]
        current_rows = [
            {
                "ticker": "AAPL",
                "opt_has_data": "1",
                "opt_atm_iv": "0.55",
                "opt_rr_25d": "-0.08",
                "opt_put_call_skew": "0.15",
                "opt_term_slope": "-0.05",
                "opt_use_for_judgment": "YES",
                "opt_liquidity_state": "active",
                "catalyst_days": "30",
                "catalyst_bucket": "build_window",
            },
            {
                "ticker": "BIIB",
                "opt_has_data": "1",
                "opt_atm_iv": "0.46",
                "opt_rr_25d": "0.04",
                "opt_put_call_skew": "-0.01",
                "opt_term_slope": "0.07",
                "opt_use_for_judgment": "YES",
                "opt_liquidity_state": "active",
                "catalyst_days": "10",
                "catalyst_bucket": "binary_now",
            },
        ]

        self._write_diag_csv(snap_root / "2026-03-25", prior_rows)
        self._write_diag_csv(snap_root / "2026-03-26", current_rows)

        result = mod.run(
            as_of_date="2026-03-26",
            prior_date="2026-03-25",
            live=False,
            dry_run=False,
        )

        assert result["schema"] == "surface_delta.v1"
        assert result["n_compared"] == 2
        assert result["n_flagged"] >= 1  # AAPL should flag

        # Check artifacts written
        out_dir = snap_root / "2026-03-26"
        assert (out_dir / "surface_delta.json").exists()
        assert (out_dir / "surface_delta.csv").exists()
        assert (out_dir / "surface_delta.md").exists()

        # Verify JSON content
        with open(out_dir / "surface_delta.json") as f:
            j = json.load(f)
        assert j["prior_date"] == "2026-03-25"
        assert len(j["deltas"]) >= 1

        # AAPL should be alert (IV jump + RR flip + term flip)
        aapl_delta = next((d for d in j["deltas"] if d["ticker"] == "AAPL"), None)
        assert aapl_delta is not None
        assert aapl_delta["severity"] == "alert"
        assert "iv_jump_large_up" in aapl_delta["flags"]
        assert "rr_flipped_bearish" in aapl_delta["flags"]

        # BIIB should not flag (tiny changes)
        biib_delta = next((d for d in j["deltas"] if d["ticker"] == "BIIB"), None)
        assert biib_delta is None


# ---------------------------------------------------------------------------
# run() error contract + crash-fragment walk-back (2026-07-15 outage)
# ---------------------------------------------------------------------------


class TestRunErrorContract:
    """run() must raise RuntimeError, never SystemExit: it is called in-process
    by run_daily_production's non-blocking step, and SystemExit bypasses that
    step's `except Exception`, failing the whole pipeline."""

    @staticmethod
    def _write_diag(snap_dir, rows):
        snap_dir.mkdir(parents=True, exist_ok=True)
        path = snap_dir / "options_diagnostics.csv"
        fieldnames = sorted({k for r in rows for k in r})
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    _ROW = {
        "ticker": "AAPL",
        "opt_has_data": "1",
        "opt_atm_iv": "0.30",
        "opt_rr_25d": "0.05",
        "opt_put_call_skew": "0.01",
        "opt_term_slope": "0.05",
        "opt_use_for_judgment": "YES",
        "opt_liquidity_state": "liquid",
    }

    def test_no_prior_snapshot_raises_runtime_error(self, tmp_path, monkeypatch):
        from tools import surface_delta_monitor as mod

        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        (tmp_path / "data" / "snapshots").mkdir(parents=True)
        with pytest.raises(RuntimeError):
            mod.run(as_of_date="2026-07-15", live=False, dry_run=True)

    def test_explicit_prior_without_diagnostics_raises(self, tmp_path, monkeypatch):
        from tools import surface_delta_monitor as mod

        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        snap_root = tmp_path / "data" / "snapshots"
        (snap_root / "2026-07-14").mkdir(parents=True)  # fragment: no diagnostics
        with pytest.raises(RuntimeError):
            mod.run(as_of_date="2026-07-15", prior_date="2026-07-14", live=False, dry_run=True)

    def test_autofind_walks_past_fragment_days(self, tmp_path, monkeypatch):
        from tools import surface_delta_monitor as mod

        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        snap_root = tmp_path / "data" / "snapshots"
        # 07-10 has diagnostics; 07-13/07-14 are crash fragments without them
        self._write_diag(snap_root / "2026-07-10", [self._ROW])
        (snap_root / "2026-07-13").mkdir(parents=True)
        (snap_root / "2026-07-14").mkdir(parents=True)
        self._write_diag(snap_root / "2026-07-15", [dict(self._ROW, opt_atm_iv="0.31")])

        result = mod.run(as_of_date="2026-07-15", live=False, dry_run=True, json_only=True)
        assert result["prior_date"] == "2026-07-10"

    def test_find_prior_snapshots_newest_first(self, tmp_path):
        from tools import surface_delta_monitor as mod

        snap_root = tmp_path / "snaps"
        for d in ("2026-07-08", "2026-07-10", "2026-07-14", "state", "_staging"):
            (snap_root / d).mkdir(parents=True)
        found = mod.find_prior_snapshots(snap_root, "2026-07-15")
        assert [p.name for p in found] == ["2026-07-14", "2026-07-10", "2026-07-08"]
