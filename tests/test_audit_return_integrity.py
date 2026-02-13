"""Tests for scripts/audit_return_integrity.py."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.audit_return_integrity import (
    build_json_output,
    classify_panel_extremes,
    compute_tier_impact,
    detect_discontinuities,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _price_map(ticker: str, date_prices: List[tuple]) -> Dict[str, Dict[date, Decimal]]:
    """Build a prices dict for a single ticker: [(date_str, price_float), ...]."""
    return {
        ticker: {date.fromisoformat(d): Decimal(str(p)) for d, p in date_prices}
    }


def _merge_prices(*maps) -> Dict[str, Dict[date, Decimal]]:
    out: Dict[str, Dict[date, Decimal]] = {}
    for m in maps:
        out.update(m)
    return out


def _panel_row(
    ticker: str = "TEST",
    as_of_date: str = "2025-06-30",
    tier: str = "B",
    eligible: str = "1",
    fwd_ret_20d: str = "0.05",
    fwd_ret_60d: str = "0.10",
) -> Dict[str, str]:
    return {
        "ticker": ticker,
        "as_of_date": as_of_date,
        "tier": tier,
        "eligible": eligible,
        "fwd_ret_20d": fwd_ret_20d,
        "fwd_ret_60d": fwd_ret_60d,
    }


# ---------------------------------------------------------------------------
# Phase A — Discontinuity detection
# ---------------------------------------------------------------------------

class TestDetectDiscontinuities:

    def test_detect_reverse_split(self):
        """A 10x single-day jump is flagged as reverse_split."""
        prices = _price_map("DRUG", [
            ("2024-10-11", 1.10),
            ("2024-10-14", 1.08),
            ("2024-10-15", 38.49),  # ~35x jump
            ("2024-10-16", 37.50),
        ])
        flags = detect_discontinuities(prices)
        assert len(flags) == 1
        assert flags[0]["ticker"] == "DRUG"
        assert flags[0]["flag_type"] == "reverse_split"
        assert flags[0]["pct_change"] > 3.0

    def test_detect_forward_split(self):
        """An 80% single-day drop is flagged as forward_split."""
        prices = _price_map("SPLT", [
            ("2025-03-01", 100.0),
            ("2025-03-02", 100.0),
            ("2025-03-03", 20.0),   # -80% drop
            ("2025-03-04", 21.0),
        ])
        flags = detect_discontinuities(prices)
        assert len(flags) == 1
        assert flags[0]["ticker"] == "SPLT"
        assert flags[0]["flag_type"] == "forward_split"
        assert flags[0]["pct_change"] <= -0.75

    def test_no_false_positive_normal_volatility(self):
        """A ±50% day in a biotech should NOT be flagged (real vol)."""
        prices = _price_map("BIOT", [
            ("2025-01-10", 10.0),
            ("2025-01-13", 15.0),   # +50%
            ("2025-01-14", 7.5),    # -50%
            ("2025-01-15", 8.0),
        ])
        flags = detect_discontinuities(prices)
        assert len(flags) == 0


# ---------------------------------------------------------------------------
# Phase B — Panel extreme classification
# ---------------------------------------------------------------------------

class TestPanelExtremeClassification:

    def test_confirmed_split_vs_plausible(self):
        """A row whose ticker has a flagged date in its forward window → confirmed_split."""
        disc = [{
            "ticker": "DRUG",
            "date": "2024-10-15",
            "close_before": 1.08,
            "close_after": 38.49,
            "pct_change": 34.64,
            "flag_type": "reverse_split",
        }]
        rows = [
            # DRUG with as_of before the split → confirmed
            _panel_row("DRUG", "2024-09-30", "D", "0", "0.50", "39.87"),
            # SAFE ticker with normal returns → plausible
            _panel_row("SAFE", "2024-09-30", "B", "1", "0.05", "0.10"),
        ]
        # Need enough rows so 0.5% tail captures at least 1
        filler = [_panel_row(f"F{i}", "2024-06-30", "B", "1", "0.03", "0.08")
                   for i in range(200)]
        all_rows = rows + filler
        result = classify_panel_extremes(all_rows, disc)
        year_2024 = result.get("2024", [])
        drug_entries = [e for e in year_2024 if e["ticker"] == "DRUG"]
        assert any(e["classification"] == "confirmed_split" for e in drug_entries)

    def test_suspicious_jump_no_flagged_day(self):
        """Extreme return (>500%) with no flagged discontinuity → suspicious_jump."""
        disc = []  # no discontinuities at all
        rows = [_panel_row("WEIRD", "2025-06-30", "D", "0", "2.0", "6.0")]
        filler = [_panel_row(f"F{i}", "2025-06-30", "B", "1", "0.03", "0.08")
                   for i in range(200)]
        all_rows = rows + filler
        result = classify_panel_extremes(all_rows, disc)
        year_2025 = result.get("2025", [])
        weird_entries = [e for e in year_2025 if e["ticker"] == "WEIRD"]
        assert any(e["classification"] == "suspicious_jump" for e in weird_entries)


# ---------------------------------------------------------------------------
# Phase C — Tier impact
# ---------------------------------------------------------------------------

class TestTierImpact:

    def _build_panel(self, extra_rows=None):
        """Build a panel with controlled tier returns + optional outlier."""
        rows = []
        # 10 A-tier at +10%
        for i in range(10):
            rows.append(_panel_row(f"A{i}", tier="A", fwd_ret_60d="0.10"))
        # 10 B-tier at +8%
        for i in range(10):
            rows.append(_panel_row(f"B{i}", tier="B", fwd_ret_60d="0.08"))
        # 10 C-tier at +5%
        for i in range(10):
            rows.append(_panel_row(f"C{i}", tier="C", fwd_ret_60d="0.05"))
        # 10 D-tier at +3%
        for i in range(10):
            rows.append(_panel_row(f"D{i}", tier="D", eligible="1", fwd_ret_60d="0.03"))
        if extra_rows:
            rows.extend(extra_rows)
        return rows

    def test_tier_impact_winsorization(self):
        """Winsorized mean < raw mean when extreme outlier present."""
        outlier = _panel_row("DRUG", tier="D", eligible="1", fwd_ret_60d="39.87")
        rows = self._build_panel([outlier])
        impact = compute_tier_impact(rows, set())
        raw_d_mean = impact["raw"]["tiers"]["D"]["mean_60d"]
        win_d_mean = impact["winsorized_200"]["tiers"]["D"]["mean_60d"]
        assert win_d_mean < raw_d_mean

    def test_tier_impact_exclusion(self):
        """Excluding flagged ticker removes its contribution entirely."""
        outlier = _panel_row("DRUG", tier="D", eligible="1", fwd_ret_60d="39.87")
        rows = self._build_panel([outlier])
        impact = compute_tier_impact(rows, {"DRUG"})
        # excl_splits should have 10 D-tier rows (the outlier excluded)
        assert impact["excl_splits"]["tiers"]["D"]["count"] == 10
        assert impact["raw"]["tiers"]["D"]["count"] == 11

    def test_ineligible_excluded(self):
        """Ineligible rows are excluded from all treatments."""
        rows = [_panel_row(f"A{i}", tier="A", eligible="1", fwd_ret_60d="0.10") for i in range(5)]
        rows.append(_panel_row("BAD", tier="A", eligible="0", fwd_ret_60d="5.0"))
        impact = compute_tier_impact(rows, set())
        assert impact["raw"]["tiers"]["A"]["count"] == 5


# ---------------------------------------------------------------------------
# JSON output schema
# ---------------------------------------------------------------------------

class TestJsonOutput:

    def test_json_output_schema(self):
        """Output JSON has all required top-level keys."""
        output = build_json_output(
            discontinuities=[],
            flagged_tickers=[],
            panel_extremes={},
            tier_impact={},
            scan_date="2026-02-13",
        )
        for key in ("scan_date", "price_discontinuities", "flagged_tickers",
                     "panel_extremes", "tier_impact"):
            assert key in output

    def test_empty_panel_no_crash(self):
        """Empty panel produces valid results without errors."""
        prices = _price_map("TEST", [("2025-01-01", 10.0), ("2025-01-02", 11.0)])
        disc = detect_discontinuities(prices)
        extremes = classify_panel_extremes([], disc)
        impact = compute_tier_impact([], set())
        assert extremes == {}
        assert impact["raw"]["AB_mean_60d"] is None
