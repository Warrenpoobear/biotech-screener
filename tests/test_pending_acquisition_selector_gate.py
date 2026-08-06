"""Tests for the pending-acquisition selector/ranker veto (2026-08-05).

corporate_actions.json has asked for this gate since APGE was added — the entry
note reads "Gate from ranker/selector until acquisition closes". The request was
inert because "pending_acquisition" was not in ACTION_TYPES, so the registry
logged "Unknown action type ... skipping" and dropped it on every load. APGE was
consequently still ranked and held — actionable_rank 23, tier A, inside the
Top-30 portfolio — at $134.19 against AbbVie's $135.11 cash price: +0.69% of
remaining upside, i.e. closed-end deal arb being scored on momentum and catalyst
features. (Its row sits at position 6 in rankings.csv, but that is the report
ordering — target_weight_pct desc, then tier — not its rank.)

Design: a VETO, not an exclusion. An announced-but-unclosed deal still trades,
so is_dead() stays False and the ticker keeps its universe membership, its
rankings.csv row, and all its features/diagnostics. It loses only
actionable_rank and target_weight_pct, and gains a "pending_acquisition" entry
in ineligible_reasons — so the name remains observable rather than vanishing.
"""

from __future__ import annotations

import logging

import pytest

from common.corporate_actions import CorporateAction, CorporateActionRegistry, is_dead, is_pending_deal

AS_OF = "2026-08-05"


def _registry(*actions):
    reg = CorporateActionRegistry(actions=list(actions))
    reg._build_indices()
    return reg


def _pending(ticker="APGE", announced="2026-06-22", price=135.11):
    return CorporateAction(
        ticker=ticker,
        action="pending_acquisition",
        effective_date=announced,  # loader substitutes announced_date
        acquirer="AbbVie",
        deal_price=price,
        announced_date=announced,
    )


# --- the veto predicate -----------------------------------------------------
def test_pending_deal_detected():
    reg = _registry(_pending())
    assert is_pending_deal("APGE", AS_OF, reg) is True


def test_pending_deal_does_not_mark_dead():
    """Must stay in the universe — it still trades."""
    reg = _registry(_pending())
    assert is_dead("APGE", AS_OF, reg) is False


def test_veto_is_pit_gated():
    """No retroactive knowledge before the announcement."""
    reg = _registry(_pending(announced="2026-06-22"))
    assert is_pending_deal("APGE", "2026-06-21", reg) is False
    assert is_pending_deal("APGE", "2026-06-22", reg) is True


def test_closed_acquisition_is_not_a_pending_deal():
    """Once it closes it becomes `acquisition` and is_dead handles it instead."""
    reg = _registry(CorporateAction(ticker="NUVL", action="acquisition", effective_date="2026-07-15", deal_price=124.0))
    assert is_pending_deal("NUVL", AS_OF, reg) is False
    assert is_dead("NUVL", AS_OF, reg) is True


def test_unaffected_ticker_is_not_vetoed():
    reg = _registry(_pending())
    assert is_pending_deal("COGT", AS_OF, reg) is False


# --- the veto as applied to rows (mirrors the run_screen block) -------------
def _apply_veto(rows, registry, as_of=AS_OF):
    """Same logic as the run_screen veto block, exercised directly."""
    from run_screen import _is_eligible

    vetoed = []
    for row in rows:
        tkr = str(row.get("ticker", "") or "").upper()
        if not tkr or not _is_eligible(row):
            continue
        if is_pending_deal(tkr, as_of, registry):
            row["eligible"] = ""
            prior = str(row.get("ineligible_reasons", "") or "")
            row["ineligible_reasons"] = f"{prior}|pending_acquisition" if prior else "pending_acquisition"
            vetoed.append(tkr)
    return vetoed


def test_row_is_made_ineligible_with_reason():
    rows = [
        {"ticker": "APGE", "eligible": "1", "ineligible_reasons": ""},
        {"ticker": "COGT", "eligible": "1", "ineligible_reasons": ""},
    ]
    vetoed = _apply_veto(rows, _registry(_pending()))
    assert vetoed == ["APGE"]
    assert rows[0]["eligible"] == ""
    assert rows[0]["ineligible_reasons"] == "pending_acquisition"
    # untouched
    assert rows[1]["eligible"] == "1"
    assert rows[1]["ineligible_reasons"] == ""


def test_existing_reasons_are_preserved_not_overwritten():
    rows = [{"ticker": "APGE", "eligible": "1", "ineligible_reasons": "deep_drawdown"}]
    _apply_veto(rows, _registry(_pending()))
    assert rows[0]["ineligible_reasons"] == "deep_drawdown|pending_acquisition"


def test_already_ineligible_row_is_not_double_tagged():
    rows = [{"ticker": "APGE", "eligible": "", "ineligible_reasons": "adv_fail"}]
    vetoed = _apply_veto(rows, _registry(_pending()))
    assert vetoed == []
    assert rows[0]["ineligible_reasons"] == "adv_fail"


def test_row_keeps_its_other_fields():
    """Veto, not deletion — features and diagnostics survive."""
    rows = [{"ticker": "APGE", "eligible": "1", "ineligible_reasons": "", "final_score": 0.87, "archetype": "x"}]
    _apply_veto(rows, _registry(_pending()))
    assert rows[0]["final_score"] == 0.87
    assert rows[0]["archetype"] == "x"


def test_empty_registry_vetoes_nothing():
    rows = [{"ticker": "APGE", "eligible": "1", "ineligible_reasons": ""}]
    assert _apply_veto(rows, _registry()) == []
    assert rows[0]["eligible"] == "1"


def test_blank_ticker_is_skipped():
    rows = [{"ticker": "", "eligible": "1", "ineligible_reasons": ""}]
    assert _apply_veto(rows, _registry(_pending())) == []


def test_veto_is_case_insensitive_on_ticker():
    rows = [{"ticker": "apge", "eligible": "1", "ineligible_reasons": ""}]
    assert _apply_veto(rows, _registry(_pending())) == ["APGE"]
