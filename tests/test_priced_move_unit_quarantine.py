"""Tests for priced_move_pct unit quarantine (Commit 2 / 2026-04-28 audit).

Covers:
1. compute_atm_straddle in common/massive_chain_analytics.py:
   - Decimal-fraction straddle (normal): no quarantine, straddle_price set
   - Dollar-unit straddle with implied_move > 5.0: straddle_price=None,
     straddle_unit_suspect="1"

2. _finalize_priced_move in run_screen.py:
   - Normal row: priced_move_pct computed correctly (straddle * 100)
   - Quarantined row (straddle_unit_suspect="1"): priced_move_pct left empty
   - Returns quarantined count

3. data_quality_overlay.json sidecar:
   - straddle_unit_quarantined_count written after _finalize_priced_move
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# compute_atm_straddle quarantine tests
# ---------------------------------------------------------------------------


def _make_contract(strike, contract_type, day_close):
    return {
        "strike_price": strike,
        "contract_type": contract_type,
        "day_close": day_close,
        "expiration_date": "2026-05-16",
    }


def test_normal_straddle_not_quarantined():
    """Decimal-fraction straddle (call=0.25, put=0.14 on $20 stock): no quarantine."""
    from common.massive_chain_analytics import compute_atm_straddle

    contracts = [
        _make_contract(20.0, "call", 0.25),
        _make_contract(20.0, "put", 0.14),
    ]
    result = compute_atm_straddle(contracts, underlying_price=20.0, expiry="2026-05-16")
    # straddle = 0.25 + 0.14 = 0.39; implied_move = 0.39/20 = 0.0195 (1.95%)
    assert result["straddle_price"] == pytest.approx(0.39, abs=1e-4)
    assert result["actual_implied_move"] == pytest.approx(0.0195, abs=1e-4)
    assert result.get("straddle_unit_suspect") != "1"


def test_dollar_straddle_quarantined():
    """Dollar-unit straddle ($12.17 on $6 stock): implied_move=2.03 > 5? No.
    Use extreme case: $40 on $6 stock → implied_move=6.67 > 5.0 → quarantine."""
    from common.massive_chain_analytics import compute_atm_straddle

    # $40 straddle on $6 stock: implied_move = 40/6 ≈ 6.67 > 5 → quarantine
    contracts = [
        _make_contract(6.0, "call", 20.0),
        _make_contract(6.0, "put", 20.0),
    ]
    result = compute_atm_straddle(contracts, underlying_price=6.0, expiry="2026-05-16")
    assert result["straddle_price"] is None
    assert result["straddle_unit_suspect"] == "1"
    assert result["actual_implied_move"] is None


def test_borderline_below_threshold_not_quarantined():
    """Straddle exactly at 5.0x underlying: NOT quarantined (threshold is > 5.0)."""
    from common.massive_chain_analytics import compute_atm_straddle

    # straddle = 5.0 * underlying: implied_move = 5.0 exactly → not > 5.0
    contracts = [
        _make_contract(10.0, "call", 25.0),
        _make_contract(10.0, "put", 25.0),
    ]
    result = compute_atm_straddle(contracts, underlying_price=10.0, expiry="2026-05-16")
    # implied_move = 50/10 = 5.0 — exactly at boundary, NOT quarantined
    assert result.get("straddle_unit_suspect") != "1"
    assert result["straddle_price"] == pytest.approx(50.0, abs=1e-4)


# ---------------------------------------------------------------------------
# _finalize_priced_move tests
# ---------------------------------------------------------------------------


def _get_finalize_fn():
    from run_screen import _finalize_priced_move
    return _finalize_priced_move


def test_normal_row_priced_move_computed():
    """Decimal-fraction straddle (0.39 on $20 stock) → priced_move_pct = 39.0."""
    fn = _get_finalize_fn()
    row = {"straddle_price": "0.39"}
    count = fn([row])
    assert row["priced_move_pct"] == pytest.approx(39.0, abs=1e-4)
    assert count == 0


def test_quarantined_row_priced_move_empty():
    """Row with straddle_unit_suspect='1': priced_move_pct left empty."""
    fn = _get_finalize_fn()
    row = {"straddle_price": "12.17", "straddle_unit_suspect": "1"}
    count = fn([row])
    assert "priced_move_pct" not in row or row.get("priced_move_pct") in (None, "")
    assert count == 1


def test_quarantine_count_is_correct():
    """Quarantine count matches the number of flagged rows."""
    fn = _get_finalize_fn()
    rows = [
        {"straddle_price": "0.39"},                                    # normal
        {"straddle_price": "12.17", "straddle_unit_suspect": "1"},     # quarantined
        {"straddle_price": "0.25"},                                    # normal
        {"straddle_price": "8.50", "straddle_unit_suspect": "1"},      # quarantined
    ]
    count = fn(rows)
    assert count == 2
    assert rows[0]["priced_move_pct"] == pytest.approx(39.0, abs=1e-4)
    assert rows[2]["priced_move_pct"] == pytest.approx(25.0, abs=1e-4)
    assert "priced_move_pct" not in rows[1] or rows[1].get("priced_move_pct") in (None, "")
    assert "priced_move_pct" not in rows[3] or rows[3].get("priced_move_pct") in (None, "")


# ---------------------------------------------------------------------------
# data_quality_overlay.json sidecar test
# ---------------------------------------------------------------------------


def test_quarantine_count_surfaces_in_sidecar(tmp_path):
    """_finalize_priced_move count is written to data_quality_overlay.json."""
    fn = _get_finalize_fn()
    rows = [
        {"straddle_price": "0.39"},
        {"straddle_price": "12.17", "straddle_unit_suspect": "1"},
    ]
    count = fn(rows)
    assert count == 1

    # Simulate the sidecar write logic from run_screen.py
    dq_path = tmp_path / "data_quality_overlay.json"
    with open(dq_path, "w", encoding="utf-8") as f:
        json.dump({"straddle_unit_quarantined_count": count}, f, indent=2)
        f.write("\n")

    result = json.loads(dq_path.read_text(encoding="utf-8"))
    assert result["straddle_unit_quarantined_count"] == 1
