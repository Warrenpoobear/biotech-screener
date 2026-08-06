"""Tests for pending_acquisition support + the 2026-08 corporate-action backfill.

Two defects, found 2026-08-05 while chasing five "yfinance fetch failures" that
turned out to be corporate actions:

1. ``pending_acquisition`` was not in ``ACTION_TYPES``, so the loader logged
   "Unknown action type 'pending_acquisition' for APGE — skipping" and dropped it.
   APGE's AbbVie deal was therefore invisible to the whole pipeline, even though
   the registry entry's own note says "Gate from ranker/selector until acquisition
   closes". APGE was ranked 6th, held, at $134.19 against a $135.11 deal price
   (+0.69% remaining).

   Naively adding the type would have CRASHED: APGE has ``effective_date: null``,
   and ``raw.get("effective_date", "")`` does not protect against an explicit null,
   so ``None`` reached ``a.effective_date > as_of`` and
   ``sorted(key=effective_date)`` — both TypeError against str. The loader now
   falls back to ``announced_date``, which is also the PIT-correct visibility date.

2. CPRX, ESPR and SGMO were missing from the registry entirely, so they were still
   scored (ranks 207 / 112 / 282) on closes 15-35 days stale, while CNTA and NUVL
   — which WERE registered — were correctly excluded.
"""

from __future__ import annotations

import json
import logging

import pytest

from common.corporate_actions import (
    ACTION_TYPES,
    DEAD_ACTION_TYPES,
    death_date,
    get_actions,
    is_dead,
    is_pending_deal,
    load_actions,
)

AS_OF = "2026-08-05"


@pytest.fixture(scope="module")
def registry():
    return load_actions()


# --- defect 1: pending_acquisition is recognised, and does not crash ---------
def test_pending_acquisition_is_a_known_action_type():
    assert "pending_acquisition" in ACTION_TYPES


def test_registry_loads_with_no_unknown_action_warnings(caplog):
    """The loader used to warn+skip on APGE every single load."""
    with caplog.at_level(logging.WARNING):
        load_actions()
    assert "Unknown action type" not in caplog.text


def test_null_effective_date_falls_back_to_announced_date(registry):
    """THE crash regression. APGE has effective_date: null."""
    acts = [a for a in registry.actions if a.ticker == "APGE"]
    assert acts, "APGE should now load"
    a = acts[0]
    assert a.action == "pending_acquisition"
    assert a.effective_date == "2026-06-22"  # announced_date, not None
    assert a.announced_date == "2026-06-22"


def test_get_actions_does_not_raise_on_pending_entries(registry):
    """None in effective_date would TypeError in the > comparison and the sort."""
    assert get_actions("APGE", registry, as_of=AS_OF)  # must not raise
    assert get_actions("APGE", registry) is not None


# --- pending is NOT dead ----------------------------------------------------
def test_pending_acquisition_is_not_a_dead_type():
    assert "pending_acquisition" not in DEAD_ACTION_TYPES


def test_apge_still_trades(registry):
    """An announced-but-unclosed deal still trades — must stay in the universe."""
    assert is_dead("APGE", AS_OF, registry) is False
    assert death_date("APGE", registry) is None
    assert is_pending_deal("APGE", AS_OF, registry) is True


def test_pending_deal_is_pit_gated_by_announcement(registry):
    """No retroactive knowledge: invisible before the 2026-06-22 announcement."""
    assert is_pending_deal("APGE", "2026-06-01", registry) is False
    assert is_pending_deal("APGE", "2026-06-22", registry) is True


# --- defect 2: the backfill --------------------------------------------------
@pytest.mark.parametrize(
    "ticker,eff",
    [
        ("CNTA", "2026-06-24"),  # Eli Lilly       (already registered)
        ("NUVL", "2026-07-15"),  # GSK             (already registered)
        ("CPRX", "2026-07-15"),  # Angelini Pharma (backfilled)
        ("ESPR", "2026-07-13"),  # ARCHIMED        (backfilled)
        ("SGMO", "2026-06-24"),  # Ch.11 / SGMOQ   (backfilled)
    ],
)
def test_all_five_are_registered_and_dead(registry, ticker, eff):
    assert is_dead(ticker, AS_OF, registry) is True, f"{ticker} still scored"
    assert death_date(registry=registry, ticker=ticker) == eff


@pytest.mark.parametrize(
    "ticker,eff",
    [("CPRX", "2026-07-15"), ("ESPR", "2026-07-13"), ("SGMO", "2026-06-24")],
)
def test_backfilled_tickers_are_pit_safe(registry, ticker, eff):
    """Must not read as dead the day before the event."""
    day_before = (
        f"{eff[:8]}{int(eff[8:]) - 1:02d}" if int(eff[8:]) > 1 else None
    )  # crude prev-day, fine for these dates
    if day_before:
        assert is_dead(ticker, day_before, registry) is False, f"{ticker} dead too early"
    assert is_dead(ticker, eff, registry) is True


def test_deal_prices_recorded(registry):
    """Deal price is what the frozen series should carry to."""
    expected = {"CPRX": 31.50, "ESPR": 3.16, "CNTA": 38.0, "NUVL": 124.0}
    for tkr, px in expected.items():
        acts = [a for a in registry.actions if a.ticker == tkr]
        assert acts[0].deal_price == px, tkr


def test_sgmo_has_no_ticker_change_redirect(registry):
    """Deliberate: a SGMO->SGMOQ ticker_change would make resolve_ticker() point
    fetches at a bankrupt sub-$0.10 OTCID listing."""
    assert not get_actions("SGMO", registry, as_of=AS_OF, action_type="ticker_change")


# --- registry file integrity ------------------------------------------------
def test_registry_json_is_wellformed_and_every_action_type_valid():
    from common.corporate_actions import _DEFAULT_PATH

    raw = json.loads(_DEFAULT_PATH.read_text(encoding="utf-8"))
    assert raw["_schema"] == "corporate_actions.v1"
    bad = [a for a in raw["actions"] if a.get("action") not in ACTION_TYPES]
    assert not bad, f"unknown action types would be silently skipped: {bad}"


def test_no_duplicate_ticker_action_pairs():
    from common.corporate_actions import _DEFAULT_PATH

    raw = json.loads(_DEFAULT_PATH.read_text(encoding="utf-8"))
    seen = [(a.get("ticker"), a.get("action"), a.get("effective_date")) for a in raw["actions"]]
    dupes = {s for s in seen if seen.count(s) > 1}
    assert not dupes, f"duplicate entries: {dupes}"
