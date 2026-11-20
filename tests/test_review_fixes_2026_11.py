"""Confirmed fixes from the 11 Nov 2026 review.

Coverage display, implausible priced-move values, and tickers renamed away
(INDV -> SUPN) must not stay eligible after the rename's effective date.
"""

from __future__ import annotations

from common.corporate_actions import load_actions
from module_1_universe import compute_module_1_universe
from production_validation import coverage_fraction, validate_screening_output
from run_screen import _finalize_priced_move


def test_coverage_fraction_never_exceeds_one():
    denom, pct = coverage_fraction(309, 297)
    assert denom == 309
    assert pct == 100.0
    assert coverage_fraction(240, 297) == (297, 240 / 297 * 100)


def test_validation_prints_module_population_not_ranked_count(capsys):
    result = {
        "module_5_composite": {"ranked_securities": [{"ticker": "A"}, {"ticker": "B"}]},
        "module_2_financial": {"diagnostic_counts": {"scored": 3, "missing": 0}},
        "module_3_catalyst": {"diagnostic_counts": {"tickers_with_events": 1}},
        "module_4_clinical": {"diagnostic_counts": {"scored": 2, "total_trials": 4}},
    }
    validate_screening_output(result, "2026-11-11", {})
    out = capsys.readouterr().out
    assert "3/3" in out
    assert "3/2" not in out


def test_priced_move_above_500_is_not_written():
    rows = [
        {"ticker": "JUNK", "straddle_price": "6.41"},
        {"ticker": "OK", "straddle_price": "0.39"},
    ]
    _finalize_priced_move(rows)
    assert "priced_move_pct" not in rows[0]
    assert rows[1]["priced_move_pct"] == 39.0


def test_indv_dropped_on_rename_effective_date_supn_stays():
    registry = load_actions()
    records = [
        {"ticker": "INDV", "company_name": "Indivior", "market_cap_mm": 4000},
        {"ticker": "SUPN", "company_name": "Supernus", "market_cap_mm": 4000},
    ]
    before = compute_module_1_universe(records, "2026-11-02", corporate_actions=registry)
    assert {s["ticker"] for s in before["active_securities"]} == {"INDV", "SUPN"}

    after = compute_module_1_universe(records, "2026-11-03", corporate_actions=registry)
    assert [s["ticker"] for s in after["active_securities"]] == ["SUPN"]
    indv = next(s for s in after["excluded_securities"] if s["ticker"] == "INDV")
    assert indv["reason"] == "excluded_delisted"
    assert indv["reason_detail"] == "ticker_retired"
