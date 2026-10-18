"""Regression tests for the 2026-10-18 corporate-actions / universe corrections.

An audit of the 18 universe tickers whose price feed had stopped found:

  * BHVN was retired as "excluded_acquired" on the strength of a registry entry
    that actually describes Cerevel (CERE) — acquired by AbbVie 2024-10-15.
    Biohaven still trades; it was wrongly dropped from the universe.
  * GLPG renamed to LKFT (Lakefront Biotherapeutics) on 2026-05-08. Nothing
    recorded the rename, so the company silently fell out of the universe.
  * CRNX, ATAI, TBPH closed acquisitions with no registry record; APGE's deal
    closed but was still recorded as pending.
  * DAWN and FOLD carried placeholder records ("Unknown" acquirer, guessed
    dates inferred from flat prices).

These tests pin the corrected facts and the universe <-> registry invariant
that would have caught the drift.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.corporate_actions import get_actions, is_dead, load_actions, resolve_ticker  # noqa: E402
from common.types import is_retired_status  # noqa: E402

AS_OF = "2026-10-18"


@pytest.fixture(scope="module")
def registry():
    return load_actions(ROOT / "production_data" / "corporate_actions.json")


@pytest.fixture(scope="module")
def universe():
    entries = json.loads((ROOT / "production_data" / "universe.json").read_text())
    return {e["ticker"]: e for e in entries if isinstance(e, dict) and e.get("ticker")}


def _death(ticker, registry):
    deaths = [a for a in get_actions(ticker, registry) if a.action in ("acquisition", "delisted", "bankruptcy")]
    assert deaths, f"{ticker}: no death record in registry"
    return deaths[0]


# --- BHVN / CERE misattribution -------------------------------------------
def test_bhvn_is_alive(registry):
    assert not is_dead("BHVN", AS_OF, registry)


def test_cerevel_acquisition_recorded_under_cere(registry):
    a = _death("CERE", registry)
    assert a.acquirer == "AbbVie"
    assert a.effective_date == "2024-10-15"


def test_bhvn_active_in_universe(universe):
    assert universe["BHVN"]["status"] == "active"


# --- GLPG -> LKFT rename ---------------------------------------------------
def test_glpg_renamed_to_lkft(registry):
    assert resolve_ticker("GLPG", "2026-06-01", registry) == "LKFT"
    assert resolve_ticker("GLPG", "2026-05-07", registry) == "GLPG"


def test_lkft_active_in_universe(universe):
    assert universe["LKFT"]["status"] == "active"
    assert universe["LKFT"]["cik"] == "0001421876"


# --- closed deals ------------------------------------------------------------
@pytest.mark.parametrize(
    "ticker, acquirer, effective, deal_price",
    [
        ("CRNX", "Vertex Pharmaceuticals", "2026-09-01", 85.0),
        ("APGE", "AbbVie", "2026-09-03", 135.11),
        ("ATAI", "Eli Lilly", "2026-09-11", 6.75),
        ("TBPH", "Zymeworks", "2026-09-23", 17.0),
        ("DAWN", "Servier", "2026-04-23", 21.5),
        ("FOLD", "BioMarin", "2026-04-27", 14.5),
    ],
)
def test_closed_acquisition(registry, universe, ticker, acquirer, effective, deal_price):
    a = _death(ticker, registry)
    assert a.acquirer == acquirer
    assert a.effective_date == effective
    assert a.deal_price == pytest.approx(deal_price)
    assert is_retired_status(universe[ticker]["status"])


def test_apge_no_longer_pending(registry):
    assert not any(a.action == "pending_acquisition" for a in get_actions("APGE", registry))


# --- the invariant that would have caught all of the above -------------------
def test_no_live_universe_ticker_is_dead_in_registry(registry, universe):
    """A ticker the pipeline still fetches and scores must not be dead in the registry."""
    bad = [t for t, e in universe.items() if not is_retired_status(e.get("status")) and is_dead(t, AS_OF, registry)]
    assert bad == []


def test_every_retired_universe_ticker_is_explained_by_registry(registry, universe):
    """A retired ticker must be dead in the registry, or renamed to a ticker that is in the universe."""
    unexplained = []
    for t, e in universe.items():
        if not is_retired_status(e.get("status")):
            continue
        if is_dead(t, AS_OF, registry):
            continue
        successor = resolve_ticker(t, AS_OF, registry)
        if successor != t and successor in universe:
            continue
        unexplained.append(t)
    assert unexplained == []
