"""CUSIP-first ticker resolution for coinvest (2026-11-01).

13F holdings in the PIT cache carry a ticker filled by the cache warmer's live
lookup. That lookup is unreliable in two ways:

1. It leaves universe names blank (DMRA, VOR, PBLS, ENGN, STRO, IONS, NUVB...),
   and the builder only consulted cusip_static_map.json for blank tickers, so
   names missing from the map got no coinvest signal at all.
2. It assigns the wrong company's ticker (Organon -> ORIC, IQVIA -> IRWD,
   Integer -> INSM, Fox -> FOLD, ...), and the builder trusted it.

The builder now resolves CUSIP-first: the static map wins over the cached
ticker, CUSIPs are case-normalised, and debt CUSIPs never resolve to equity.

Mappings are verified against OpenFIGI (US-listed equity, universe ticker,
issuer name agrees) or by matching the 13F implied price to the ticker's
period-end close — never by issuer-name matching alone (see VTRS/VKTX,
69e49c031).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.build_coinvest_features_from_13f import _resolve_ticker

MAP_PATH = ROOT / "production_data" / "cusip_static_map.json"

# Universe names that resolved to nothing (blank cached ticker, no map entry).
VERIFIED = {
    "929033207": "VOR",  # OpenFIGI
    "698955101": "PBLS",  # OpenFIGI
    "337185102": "TRAX",  # OpenFIGI
    "24477V105": "DFTX",  # OpenFIGI
    "462222100": "IONS",  # OpenFIGI
    "67080N101": "NUVB",  # OpenFIGI
    "36322Q206": "DMRA",  # 13F implied $26.01 == DMRA 2026-06-30 close
    "K08588103": "ASND",  # 13F implied $266.72 == ASND 2026-06-30 close
}

# Non-universe issuers the cache warmer tagged with a universe ticker.
MISTAGGED = {
    "68622V106": ("ORIC", "OGN"),  # Organon
    "46266C105": ("IRWD", "IQV"),  # IQVIA
    "45826H109": ("INSM", "ITGR"),  # Integer Holdings
    "35137L105": ("FOLD", "FOXA"),  # Fox Corp
}


@pytest.fixture(scope="module")
def cusip_map():
    return json.loads(MAP_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("cusip,ticker", sorted(VERIFIED.items()))
def test_verified_cusip_mapped(cusip_map, cusip, ticker):
    assert cusip_map.get(cusip) == ticker


@pytest.mark.parametrize("cusip,ticker", sorted(VERIFIED.items()))
def test_blank_ticker_holding_resolves_via_map(cusip_map, cusip, ticker):
    assert _resolve_ticker({"cusip": cusip, "ticker": ""}, cusip_map) == ticker


@pytest.mark.parametrize("cusip,wrong_right", sorted(MISTAGGED.items()))
def test_static_map_overrides_wrong_cached_ticker(cusip_map, cusip, wrong_right):
    wrong, right = wrong_right
    assert _resolve_ticker({"cusip": cusip, "ticker": wrong}, cusip_map) == right


def test_cached_ticker_used_when_cusip_unmapped():
    assert _resolve_ticker({"cusip": "999999999", "ticker": "abcd"}, {}) == "ABCD"


def test_lowercase_cusip_resolves():
    assert _resolve_ticker({"cusip": "29337e102", "ticker": ""}, {"29337E102": "ELVN"}) == "ELVN"


def test_debt_cusip_never_resolves_to_equity():
    # Immunocore convertible note cached with ticker IMCR.
    assert _resolve_ticker({"cusip": "45258DAB1", "ticker": "IMCR"}, {}) == ""


def test_newamsterdam_warrants_not_mapped_to_shares(cusip_map):
    # N62509117 prices at $21.37 vs NAMS $33.89 on 2026-06-30: warrants.
    assert cusip_map.get("N62509117") != "NAMS"


def test_map_keys_are_well_formed_cusips(cusip_map):
    for cusip in cusip_map:
        assert re.fullmatch(r"[0-9A-Z]{8}[0-9]", cusip), cusip


def test_no_debt_cusips_in_map(cusip_map):
    bad = [c for c in cusip_map if re.fullmatch(r"[0-9A-Z]{6}[A-Z]{2}[0-9]", c)]
    assert bad == []
