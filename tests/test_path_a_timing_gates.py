"""Tests for Path A portfolio timing gates (Spec 106 A0)."""

from __future__ import annotations

import copy
import sys
from decimal import Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.path_a_timing_gates import (
    PathATimingGates,
    classify_path_a_zone,
    enforce_path_a_gates,
    gates_from_policy,
    measure_zone_weights,
    parse_catalyst_days,
    zone_map_from_rows,
)


def _row(ticker: str, rank: int, days: str, mode: str = "specific_days") -> dict:
    return {
        "ticker": ticker,
        "actionable_rank": str(rank),
        "eligible": "1",
        "catalyst_days": days,
        "catalyst_mode": mode,
    }


def _pos(ticker: str, rank: int, wt: float, days: str = "", mode: str = "specific_days") -> dict:
    return {
        "ticker": ticker,
        "weight_pct": wt,
        "target_dollars": wt * 1000,
        "actionable_rank": rank,
        "catalyst_days": days,
        "catalyst_mode": mode,
    }


class TestClassifyPathAZone:
    def test_t0_boundary(self):
        assert classify_path_a_zone(7, "specific_days") == "T0"
        assert classify_path_a_zone(8, "specific_days") == "T1"

    def test_t3_boundary(self):
        assert classify_path_a_zone(89, "specific_days") == "T2"
        assert classify_path_a_zone(90, "specific_days") == "T3"

    def test_blended_is_t4(self):
        assert classify_path_a_zone(0, "blended_window") == "T4"

    def test_missing_is_t4(self):
        assert classify_path_a_zone(None, "missing") == "T4"


def test_gates_from_policy_shadow_file():
    import json

    policy = json.loads((REPO / "production_data" / "portfolio_policy_path_a_shadow.json").read_text())
    gates = gates_from_policy(policy)
    assert gates.enabled is True
    assert gates.t0_max_weight_pct == Decimal("30.0")
    assert gates.t3_plus_min_weight_pct == Decimal("40.0")


def test_t0_cap_enforcement():
    gates = PathATimingGates(enabled=True)
    rankings = [
        _row("A", 1, "5"),
        _row("B", 2, "6"),
        _row("C", 3, "4"),
        _row("D", 4, "120"),
        _row("E", 5, "150"),
        _row("F", 6, "200"),
        _row("G", 7, "110"),
    ]
    zones = zone_map_from_rows(rankings)
    positions = [
        _pos("A", 1, 20.0, "5"),
        _pos("B", 2, 20.0, "6"),
        _pos("C", 3, 20.0, "4"),
        _pos("D", 4, 20.0, "120"),
        _pos("E", 5, 20.0, "150"),
    ]
    t0_before, _ = measure_zone_weights(positions, zones, t3_plus_zones=gates.t3_plus_zones)
    assert t0_before == Decimal("60.0")

    result = enforce_path_a_gates(
        positions,
        rankings,
        gates,
        account_usd=Decimal("100000"),
        target_n=5,
    )
    assert result.t0_weight_pct <= gates.t0_max_weight_pct
    assert len(result.swaps) >= 2
    assert len(result.positions) == 5


def test_t3_plus_floor_swap():
    gates = PathATimingGates(enabled=True, t0_max_weight_pct=Decimal("100"))
    rankings = [
        _row("NEAR1", 1, "10"),
        _row("NEAR2", 2, "15"),
        _row("FAR1", 3, "120"),
        _row("FAR2", 4, "150"),
    ]
    positions = [
        _pos("NEAR1", 1, 50.0, "10"),
        _pos("NEAR2", 2, 50.0, "15"),
    ]
    result = enforce_path_a_gates(
        positions,
        rankings,
        gates,
        account_usd=Decimal("100000"),
        target_n=2,
    )
    assert result.t3_plus_weight_pct >= gates.t3_plus_min_weight_pct or result.floor_relaxed


def test_deterministic_double_run():
    gates = PathATimingGates(enabled=True)
    rankings = [_row(f"T{i}", i, str(5 if i <= 3 else 120)) for i in range(1, 11)]
    positions = [_pos(f"T{i}", i, 10.0, "5" if i <= 3 else "120") for i in range(1, 11)]
    r1 = enforce_path_a_gates(positions, rankings, gates, account_usd=Decimal("500000"), target_n=10)
    r2 = enforce_path_a_gates(copy.deepcopy(positions), rankings, gates, account_usd=Decimal("500000"), target_n=10)
    t1 = [p["ticker"] for p in r1.positions]
    t2 = [p["ticker"] for p in r2.positions]
    assert t1 == t2
    assert r1.t0_weight_pct == r2.t0_weight_pct


def test_build_positions_overlay_disabled_by_default():
    from tools.live_shadow_portfolio import build_positions

    policy = {
        "construction_mode": "ew_top_n",
        "ew_top_n": 3,
        "account_usd": 100_000,
    }
    rankings = [_row("A", 1, "5"), _row("B", 2, "120"), _row("C", 3, "130")]
    out = build_positions(rankings, policy)
    assert "path_a" not in out.get("summary", {})


def test_build_positions_overlay_when_enabled():
    from tools.live_shadow_portfolio import build_positions

    policy = {
        "construction_mode": "ew_top_n",
        "ew_top_n": 4,
        "account_usd": 100_000,
        "_as_of_date": "2026-06-24",
        "path_a_timing_gates": {
            "enabled": True,
            "t0_imminent": {"max_days": 7, "max_weight_pct": 50.0},
            "t3_plus": {"min_days": 90, "min_weight_pct": 25.0, "include_zones": ["T3", "T4"]},
        },
    }
    rankings = [
        _row("T0A", 1, "3"),
        _row("T0B", 2, "5"),
        _row("FAR", 3, "120"),
        _row("FAR2", 4, "150"),
    ]
    out = build_positions(rankings, policy)
    assert out["summary"]["path_a"]["applied"] is True
    assert out["summary"]["path_a"]["t0_compliant"] is True
    assert "path_a_manifest" in out


def test_parse_catalyst_days():
    assert parse_catalyst_days("7") == 7
    assert parse_catalyst_days("") is None
    assert parse_catalyst_days(None) is None
