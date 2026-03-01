"""Tests for portfolio exposure guardrail in the Phase-2 health gate.

Verifies compute_exposure_metrics() and its integration into
compute_health_gate() via WARN/FAIL thresholds on portfolio-level
risk concentration metrics.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from run_phase2_snapshot_delta import (
    CatalystCoverage,
    Phase2HealthThresholds,
    SnapshotData,
    SingleResult,
    HealthResult,
    compute_exposure_metrics,
    compute_health_gate,
    _EXPOSURE_CHECKS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _portfolio_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build a portfolio DataFrame from row dicts."""
    return pd.DataFrame(rows)


def _make_rankings(n: int = 5) -> pd.DataFrame:
    """Minimal rankings DataFrame for SnapshotData."""
    return pd.DataFrame({
        "ticker": [f"T{i}" for i in range(n)],
        "composite_score": [50.0] * n,
        "eligible": ["1"] * n,
        "tier_any": ["A"] * n,
        "archetype": ["drug_developer"] * n,
    })


def _make_snapshot(
    portfolio_rows: List[Dict[str, Any]],
    ruleset_id: str = "test_rs",
) -> SnapshotData:
    """Build a minimal SnapshotData for health gate testing."""
    portfolio = _portfolio_df(portfolio_rows) if portfolio_rows else pd.DataFrame()
    rankings = _make_rankings(max(len(portfolio_rows), 5))
    return SnapshotData(
        date="2026-03-01",
        path=Path("/tmp/test_snapshot"),
        rankings=rankings,
        portfolio=portfolio,
        metadata={},
        ruleset_id=ruleset_id,
        has_native_portfolio=True,
    )


def _single_result(portfolio_rows: List[Dict[str, Any]]) -> SingleResult:
    """Build a minimal SingleResult with tier/catalyst counts."""
    n_a = sum(1 for r in portfolio_rows if r.get("tier_any") == "A")
    return SingleResult(
        tier_counts={"A": max(n_a, 3), "B": 5},
        tier_in_portfolio={"A": max(n_a, 3), "B": 2},
        size_band={"L": 3, "M": 2},
        catalyst_coverage=CatalystCoverage(
            n_dev=15, n_specific=10, pct=80.0, mode_dist={"specific_days": 10},
        ),
        top_catalysts=[],
        risk={},
        warnings=[],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExposureMetrics:
    """Tests for compute_exposure_metrics() and health gate integration."""

    def test_exposure_catalyst_7d_warn(self):
        """22% weight in catalyst<=7d triggers WARN (threshold 20%)."""
        rows = [
            {"ticker": "A", "target_weight_pct": 6.0, "catalyst_days": "5", "risk_flags": "", "mom_state": "neutral"},
            {"ticker": "B", "target_weight_pct": 5.0, "catalyst_days": "3", "risk_flags": "", "mom_state": "neutral"},
            {"ticker": "C", "target_weight_pct": 5.0, "catalyst_days": "2", "risk_flags": "", "mom_state": "neutral"},
            {"ticker": "D", "target_weight_pct": 6.0, "catalyst_days": "7", "risk_flags": "", "mom_state": "neutral"},
            {"ticker": "E", "target_weight_pct": 5.0, "catalyst_days": "90", "risk_flags": "", "mom_state": "neutral"},
            {"ticker": "F", "target_weight_pct": 5.0, "catalyst_days": "90", "risk_flags": "", "mom_state": "neutral"},
            {"ticker": "G", "target_weight_pct": 5.0, "catalyst_days": "90", "risk_flags": "", "mom_state": "neutral"},
            {"ticker": "H", "target_weight_pct": 5.0, "catalyst_days": "90", "risk_flags": "", "mom_state": "neutral"},
            {"ticker": "I", "target_weight_pct": 5.0, "catalyst_days": "90", "risk_flags": "", "mom_state": "neutral"},
            {"ticker": "J", "target_weight_pct": 5.0, "catalyst_days": "90", "risk_flags": "", "mom_state": "neutral"},
            {"ticker": "K", "target_weight_pct": 5.0, "catalyst_days": "90", "risk_flags": "", "mom_state": "neutral"},
            {"ticker": "L", "target_weight_pct": 5.0, "catalyst_days": "90", "risk_flags": "", "mom_state": "neutral"},
            {"ticker": "M", "target_weight_pct": 5.0, "catalyst_days": "90", "risk_flags": "", "mom_state": "neutral"},
            {"ticker": "N", "target_weight_pct": 5.0, "catalyst_days": "90", "risk_flags": "", "mom_state": "neutral"},
            {"ticker": "O", "target_weight_pct": 5.0, "catalyst_days": "90", "risk_flags": "", "mom_state": "neutral"},
            {"ticker": "P", "target_weight_pct": 5.0, "catalyst_days": "90", "risk_flags": "", "mom_state": "neutral"},
            {"ticker": "Q", "target_weight_pct": 5.0, "catalyst_days": "90", "risk_flags": "", "mom_state": "neutral"},
            {"ticker": "R", "target_weight_pct": 3.0, "catalyst_days": "90", "risk_flags": "", "mom_state": "neutral"},
        ]
        metrics = compute_exposure_metrics(_portfolio_df(rows))
        assert metrics["catalyst_le_7d_weight_pct"] == 22.0
        assert metrics["catalyst_le_7d_count"] == 4

        # Verify WARN via health gate — top5 = 6+6+5+5+5 = 27 < 40 threshold
        current = _make_snapshot(rows, ruleset_id="test_rs")
        th = Phase2HealthThresholds(
            warn_catalyst_le_7d_weight_pct=20.0,
            fail_catalyst_le_7d_weight_pct=35.0,
        )
        result = _single_result(rows)
        health = compute_health_gate(current, None, result, thresholds=th, expected_ruleset_id="test_rs")
        assert "catalyst_7d_weight_high" in health.reasons
        assert health.status == "WARN"

    def test_exposure_catalyst_7d_fail(self):
        """40% weight in catalyst<=7d triggers FAIL (threshold 35%)."""
        # Spread across many positions so top-5 concentration doesn't also fire FAIL
        base = {"risk_flags": "", "mom_state": "neutral"}
        rows = [
            {"ticker": "A", "target_weight_pct": 5.0, "catalyst_days": "2", **base},
            {"ticker": "B", "target_weight_pct": 5.0, "catalyst_days": "7", **base},
            {"ticker": "C", "target_weight_pct": 5.0, "catalyst_days": "3", **base},
            {"ticker": "D", "target_weight_pct": 5.0, "catalyst_days": "5", **base},
            {"ticker": "E", "target_weight_pct": 5.0, "catalyst_days": "1", **base},
            {"ticker": "F", "target_weight_pct": 5.0, "catalyst_days": "6", **base},
            {"ticker": "G", "target_weight_pct": 5.0, "catalyst_days": "4", **base},
            {"ticker": "H", "target_weight_pct": 5.0, "catalyst_days": "7", **base},
        ] + [
            {"ticker": f"Z{i}", "target_weight_pct": 5.0, "catalyst_days": "90", **base}
            for i in range(12)
        ]
        metrics = compute_exposure_metrics(_portfolio_df(rows))
        assert metrics["catalyst_le_7d_weight_pct"] == 40.0

        current = _make_snapshot(rows, ruleset_id="test_rs")
        th = Phase2HealthThresholds(
            warn_catalyst_le_7d_weight_pct=20.0,
            fail_catalyst_le_7d_weight_pct=35.0,
        )
        result = _single_result(rows)
        health = compute_health_gate(current, None, result, thresholds=th, expected_ruleset_id="test_rs")
        assert "catalyst_7d_weight_high" in health.reasons
        assert health.status == "FAIL"

    def test_exposure_high_vol_warn(self):
        """28% high_vol weight triggers WARN (threshold 25%)."""
        base = {"catalyst_days": "90", "mom_state": "neutral"}
        rows = [
            {"ticker": "A", "target_weight_pct": 5.0, "risk_flags": "high_vol", **base},
            {"ticker": "B", "target_weight_pct": 5.0, "risk_flags": "high_vol", **base},
            {"ticker": "C", "target_weight_pct": 5.0, "risk_flags": "high_vol", **base},
            {"ticker": "D", "target_weight_pct": 5.0, "risk_flags": "high_vol", **base},
            {"ticker": "E", "target_weight_pct": 4.0, "risk_flags": "high_vol", **base},
            {"ticker": "F", "target_weight_pct": 4.0, "risk_flags": "high_vol|overbought_rsi", **base},
        ] + [
            {"ticker": f"Z{i}", "target_weight_pct": 4.5, "risk_flags": "", **base}
            for i in range(16)
        ]
        metrics = compute_exposure_metrics(_portfolio_df(rows))
        assert metrics["high_vol_weight_pct"] == 28.0

        current = _make_snapshot(rows, ruleset_id="test_rs")
        result = _single_result(rows)
        health = compute_health_gate(current, None, result, expected_ruleset_id="test_rs")
        assert "high_vol_exposure" in health.reasons

    def test_exposure_stacked_risk_warn(self):
        """12% catalyst<=7d AND drawdown_rel_xbi triggers WARN (threshold 10%)."""
        base = {"mom_state": "neutral"}
        rows = [
            {"ticker": "A", "target_weight_pct": 5.0, "catalyst_days": "5", "risk_flags": "deep_drawdown_rel_xbi", **base},
            {"ticker": "B", "target_weight_pct": 4.0, "catalyst_days": "3", "risk_flags": "deep_drawdown_rel_xbi", **base},
            {"ticker": "C", "target_weight_pct": 3.0, "catalyst_days": "7", "risk_flags": "deep_drawdown_rel_xbi", **base},
        ] + [
            {"ticker": f"Z{i}", "target_weight_pct": 4.4, "risk_flags": "", "catalyst_days": "90", **base}
            for i in range(20)
        ]
        metrics = compute_exposure_metrics(_portfolio_df(rows))
        assert metrics["catalyst_le_7d_and_drawdown_weight_pct"] == 12.0

        current = _make_snapshot(rows, ruleset_id="test_rs")
        result = _single_result(rows)
        health = compute_health_gate(current, None, result, expected_ruleset_id="test_rs")
        assert "stacked_catalyst_drawdown" in health.reasons

    def test_exposure_headwind_pass(self):
        """20% headwind is below WARN threshold (30%) → PASS (no reason)."""
        base = {"catalyst_days": "90", "risk_flags": ""}
        rows = [
            {"ticker": "A", "target_weight_pct": 5.0, "mom_state": "headwind", **base},
            {"ticker": "B", "target_weight_pct": 5.0, "mom_state": "headwind", **base},
            {"ticker": "C", "target_weight_pct": 5.0, "mom_state": "headwind", **base},
            {"ticker": "D", "target_weight_pct": 5.0, "mom_state": "headwind", **base},
        ] + [
            {"ticker": f"Z{i}", "target_weight_pct": 4.0, "mom_state": "tailwind", **base}
            for i in range(20)
        ]
        metrics = compute_exposure_metrics(_portfolio_df(rows))
        assert metrics["headwind_weight_pct"] == 20.0

        current = _make_snapshot(rows, ruleset_id="test_rs")
        result = _single_result(rows)
        health = compute_health_gate(current, None, result, expected_ruleset_id="test_rs")
        assert "headwind_exposure" not in health.reasons

    def test_exposure_top5_warn(self):
        """42% top-5 weight triggers WARN (threshold 40%)."""
        # 6 positions, top-5 sum = 9+9+9+8+7 = 42
        rows = [
            {"ticker": "A", "target_weight_pct": 9.0, "catalyst_days": "90", "risk_flags": "", "mom_state": "neutral"},
            {"ticker": "B", "target_weight_pct": 9.0, "catalyst_days": "90", "risk_flags": "", "mom_state": "neutral"},
            {"ticker": "C", "target_weight_pct": 9.0, "catalyst_days": "90", "risk_flags": "", "mom_state": "neutral"},
            {"ticker": "D", "target_weight_pct": 8.0, "catalyst_days": "90", "risk_flags": "", "mom_state": "neutral"},
            {"ticker": "E", "target_weight_pct": 7.0, "catalyst_days": "90", "risk_flags": "", "mom_state": "neutral"},
            {"ticker": "F", "target_weight_pct": 58.0, "catalyst_days": "", "risk_flags": "", "mom_state": "neutral"},
        ]
        metrics = compute_exposure_metrics(_portfolio_df(rows))
        # top-5 sorted desc: 58, 9, 9, 9, 8 = 93
        assert metrics["top5_weight_pct"] == 93.0

        # Use tight thresholds for this test
        current = _make_snapshot(rows, ruleset_id="test_rs")
        th = Phase2HealthThresholds(warn_top5_weight_pct=40.0, fail_top5_weight_pct=55.0)
        result = _single_result(rows)
        health = compute_health_gate(current, None, result, thresholds=th, expected_ruleset_id="test_rs")
        assert "top5_concentration" in health.reasons

    def test_exposure_clean_portfolio(self):
        """Clean portfolio (no risk flags, distant catalysts) → no exposure reasons."""
        base = {"risk_flags": ""}
        rows = [
            {"ticker": f"T{i}", "target_weight_pct": 5.0,
             "catalyst_days": "90", "mom_state": "tailwind", **base}
            for i in range(20)
        ]
        metrics = compute_exposure_metrics(_portfolio_df(rows))
        assert metrics["catalyst_le_7d_weight_pct"] == 0.0
        assert metrics["high_vol_weight_pct"] == 0.0
        assert metrics["headwind_weight_pct"] == 0.0

        exposure_reasons = {r for _, _, _, r in _EXPOSURE_CHECKS}
        current = _make_snapshot(rows, ruleset_id="test_rs")
        result = _single_result(rows)
        health = compute_health_gate(current, None, result, expected_ruleset_id="test_rs")
        assert not exposure_reasons.intersection(health.reasons)

    def test_exposure_empty_portfolio(self):
        """Empty DataFrame → all zeros, no crash."""
        metrics = compute_exposure_metrics(pd.DataFrame())
        assert metrics["catalyst_le_7d_weight_pct"] == 0.0
        assert metrics["catalyst_le_7d_count"] == 0
        assert metrics["top5_weight_pct"] == 0.0
        assert metrics["max_single_weight_pct"] == 0.0

    def test_exposure_metrics_in_health_json(self):
        """Verify exposure dict appears in health.metrics when portfolio is non-empty."""
        rows = [
            {"ticker": "A", "target_weight_pct": 10.0, "catalyst_days": "30", "risk_flags": "", "mom_state": "neutral"},
            {"ticker": "B", "target_weight_pct": 90.0, "catalyst_days": "90", "risk_flags": "", "mom_state": "neutral"},
        ]
        current = _make_snapshot(rows, ruleset_id="test_rs")
        result = _single_result(rows)
        health = compute_health_gate(current, None, result, expected_ruleset_id="test_rs")
        assert "exposure" in health.metrics
        exp = health.metrics["exposure"]
        assert "catalyst_le_7d_weight_pct" in exp
        assert "top5_weight_pct" in exp
        assert "max_single_weight_pct" in exp
