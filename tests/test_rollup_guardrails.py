"""Regression guardrails on the catalyst shadow timeseries rollup.

Reads the live rollup CSV and asserts invariants that should hold
across the full business-day series (Jan 15+).
Catches accidental coverage cliffs, data regressions, or broken pipelines.

Usage:
    pytest tests/test_rollup_guardrails.py -v
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

ROLLUP_CSV = Path(__file__).resolve().parent.parent / "output" / "catalyst_shadow_timeseries.csv"

# ── Series boundary ─────────────────────────────────────────────────
# With PIT-filtered CTGov caches, CTG is nonzero across the full series.
# Churn checks use STEADY_STATE_START to skip prior-chain transition dates
# (Feb 02 + Feb 04 have inflated B→G from stale priors being replaced).
SERIES_START = "2026-01-15"
STEADY_STATE_START = "2026-02-05"  # after prior-chain artifacts settle

# Known one-off data pipeline events that cause transient churn spikes.
# These dates are excluded from steady-state guardrails because the anomaly
# is a data refresh, not a logic regression (verified: next day overlaps
# return to >0.93).
KNOWN_CACHE_REFRESH_DATES = {
    "2026-02-17",  # CTGov cache refresh (1082→1315 entries) + SEC 8-K cache empty (0 events)
}

# ── Thresholds (generous — these are guardrails, not tight bounds) ──
MAX_DAILY_B2G = 15          # steady-state B→G per day (Feb 14 SEC refresh was 8)
MAX_DAILY_G2B = 5           # coverage regressions should be near-zero
MIN_A_TIER = 25             # A-tier should stay above this across series
MIN_TOP60_OVERLAP = 0.70    # Jaccard; below this = major portfolio churn
MIN_TOP100_OVERLAP = 0.85
MIN_CTG_EVENTS = 500        # CTG must be present across entire series


def _load_rollup() -> list[dict]:
    if not ROLLUP_CSV.exists():
        pytest.skip(f"Rollup CSV not found: {ROLLUP_CSV}")
    with open(ROLLUP_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _steady_state_rows(rows: list[dict]) -> list[dict]:
    out = [
        r for r in rows
        if r["as_of_date"] >= STEADY_STATE_START
        and r["as_of_date"] not in KNOWN_CACHE_REFRESH_DATES
    ]
    if not out:
        pytest.skip("No steady-state rows in rollup (need Feb 05+)")
    return out


def _float_or_none(val: str) -> float | None:
    if val in ("", None):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _int_or_none(val: str) -> int | None:
    f = _float_or_none(val)
    return int(f) if f is not None else None


# ── Tests ───────────────────────────────────────────────────────────


class TestSteadyStateChurn:
    """Post-Feb-05 daily churn should be bounded."""

    def test_b2g_bounded(self):
        rows = _steady_state_rows(_load_rollup())
        for r in rows:
            b2g = _int_or_none(r["bad_to_good_count"])
            if b2g is not None:
                assert b2g <= MAX_DAILY_B2G, (
                    f"{r['as_of_date']}: B→G={b2g} exceeds {MAX_DAILY_B2G}"
                )

    def test_g2b_bounded(self):
        rows = _steady_state_rows(_load_rollup())
        for r in rows:
            g2b = _int_or_none(r["good_to_bad_count"])
            if g2b is not None:
                assert g2b <= MAX_DAILY_G2B, (
                    f"{r['as_of_date']}: G→B={g2b} exceeds {MAX_DAILY_G2B}"
                )


class TestCTGCoverage:
    """CTG events should be nonzero across the entire business-day series."""

    def test_ctg_nonzero_all_rows(self):
        rows = _load_rollup()
        for r in rows:
            ctg = _int_or_none(r["ctgov_events"])
            if ctg is not None:
                assert ctg >= MIN_CTG_EVENTS, (
                    f"{r['as_of_date']}: CTG={ctg} below {MIN_CTG_EVENTS}"
                )


class TestATierStability:
    """A-tier count should stay above threshold post-CTG."""

    def test_a_tier_above_minimum(self):
        rows = _steady_state_rows(_load_rollup())
        for r in rows:
            a = _int_or_none(r["A_tier_count"])
            if a is not None:
                assert a >= MIN_A_TIER, (
                    f"{r['as_of_date']}: A-tier={a} below {MIN_A_TIER}"
                )


class TestOverlapStability:
    """Top-60/100 overlap should stay in expected range."""

    def test_top60_overlap(self):
        rows = _steady_state_rows(_load_rollup())
        for r in rows:
            v = _float_or_none(r["top60_overlap"])
            if v is not None:
                assert v >= MIN_TOP60_OVERLAP, (
                    f"{r['as_of_date']}: top60_overlap={v:.2f} below {MIN_TOP60_OVERLAP}"
                )

    def test_top100_overlap(self):
        rows = _steady_state_rows(_load_rollup())
        for r in rows:
            v = _float_or_none(r["top100_overlap"])
            if v is not None:
                assert v >= MIN_TOP100_OVERLAP, (
                    f"{r['as_of_date']}: top100_overlap={v:.2f} below {MIN_TOP100_OVERLAP}"
                )


class TestMedianCatalystDays:
    """Median catalyst days should be reasonable in steady state."""

    def test_positive_and_bounded(self):
        rows = _steady_state_rows(_load_rollup())
        for r in rows:
            med = _float_or_none(r["median_catalyst_days_good"])
            if med is not None:
                assert 0 < med < 500, (
                    f"{r['as_of_date']}: median_catalyst_days={med} out of range"
                )
