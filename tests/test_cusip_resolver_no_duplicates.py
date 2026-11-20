"""KNOWN_CUSIP_MAPPINGS: no duplicate keys, and the four corrected CUSIPs.

A Python dict literal silently keeps the last value for a repeated key. Until
2026-11-12 four CUSIPs appeared twice with different tickers, and the two
copies of the table (root and sec_13f/) disagreed on one of them.

Issuer evidence is from 13F filings in the repo's own cache:
  74587V107  PUMA BIOTECHNOLOGY INC   (was QTTB, then PTCT)
  04621X108  ASSURANT INC             (was ASTH, then ATXS)
  09061G101  BIOMARIN PHARMACEUTICAL  (was BIIB, then BMRN)
"""

import ast
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLES = [REPO_ROOT / "cusip_resolver.py", REPO_ROOT / "sec_13f" / "cusip_resolver.py"]


def _literal_items(path: Path):
    """(key, value) pairs of the KNOWN_CUSIP_MAPPINGS literal, duplicates included."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if any(isinstance(t, ast.Name) and t.id == "KNOWN_CUSIP_MAPPINGS" for t in targets):
            assert isinstance(node.value, ast.Dict)
            return [(k.value, v.value) for k, v in zip(node.value.keys, node.value.values)]
    raise AssertionError(f"KNOWN_CUSIP_MAPPINGS not found in {path}")


@pytest.mark.parametrize("path", TABLES, ids=["root", "sec_13f"])
def test_no_duplicate_cusip_keys(path):
    counts = Counter(k for k, _ in _literal_items(path))
    assert {k: n for k, n in counts.items() if n > 1} == {}


@pytest.mark.parametrize("path", TABLES, ids=["root", "sec_13f"])
def test_corrected_cusips(path):
    table = dict(_literal_items(path))
    assert table["74587V107"] == "PBYI"  # Puma Biotechnology
    assert table["04621X108"] == "AIZ"  # Assurant
    assert table["09061G101"] == "BMRN"  # BioMarin
    # Never seen in a filing and its issuer prefix is Puma's, so neither old guess is kept.
    assert "74587V305" not in table


@pytest.mark.parametrize("path", TABLES, ids=["root", "sec_13f"])
def test_no_ticker_borrows_another_issuers_prefix(path):
    """The first six CUSIP characters identify the issuer."""
    table = dict(_literal_items(path))
    # 74587V is Puma Biotechnology and 04621X is Assurant (13F issuer names).
    assert {k: v for k, v in table.items() if k.startswith("74587V") and v != "PBYI"} == {}
    assert {k: v for k, v in table.items() if k.startswith("04621X") and v != "AIZ"} == {}


def test_both_copies_agree_on_shared_keys():
    root, sec = dict(_literal_items(TABLES[0])), dict(_literal_items(TABLES[1]))
    assert {k: (root[k], sec[k]) for k in root.keys() & sec.keys() if root[k] != sec[k]} == {}


def test_live_table_resolves_the_corrected_cusips():
    from cusip_resolver import KNOWN_CUSIP_MAPPINGS

    assert KNOWN_CUSIP_MAPPINGS["74587V107"] == "PBYI"
    assert KNOWN_CUSIP_MAPPINGS["09061G101"] == "BMRN"
