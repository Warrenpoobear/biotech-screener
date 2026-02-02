#!/usr/bin/env python3
"""
Tests for normalize_financial_field_alias — the central remap that handles
both 'financial_data' (canonical) and 'financials' (legacy) keys in
universe records.

Requirement: both input shapes must produce identical downstream extracted
values for Module 2.
"""

import copy
import sys
from pathlib import Path

import pytest

# Ensure parent is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.data_integration_contracts import normalize_financial_field_alias


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_FIN_DICT = {
    "cash": 150_000_000,
    "debt": 20_000_000,
    "revenue_ttm": 0,
    "source_date": "2025-12-15",
}


def _make_record(key: str, fin=None):
    """Build a minimal universe record with financial data under *key*."""
    rec = {"ticker": "ACME"}
    if fin is not None:
        rec[key] = fin
    return rec


# ---------------------------------------------------------------------------
# Core contract tests
# ---------------------------------------------------------------------------


class TestNormalizeFinancialFieldAlias:
    """Both input shapes must yield identical financial_data after normalization."""

    def test_canonical_key_unchanged(self):
        """If 'financial_data' exists it must be kept as-is."""
        rec = _make_record("financial_data", SAMPLE_FIN_DICT)
        normalize_financial_field_alias(rec)
        assert rec["financial_data"] is SAMPLE_FIN_DICT

    def test_legacy_key_remapped(self):
        """If only 'financials' exists it must be copied to 'financial_data'."""
        rec = _make_record("financials", SAMPLE_FIN_DICT)
        normalize_financial_field_alias(rec)
        assert rec["financial_data"] is SAMPLE_FIN_DICT

    def test_canonical_takes_precedence(self):
        """If both keys exist 'financial_data' wins (no overwrite)."""
        canonical = {"cash": 999}
        rec = {"ticker": "ACME", "financial_data": canonical, "financials": SAMPLE_FIN_DICT}
        normalize_financial_field_alias(rec)
        assert rec["financial_data"] is canonical

    def test_neither_key_present(self):
        """If neither key exists the record passes through unchanged."""
        rec = {"ticker": "ACME", "market_data": {"price": 10}}
        before = copy.deepcopy(rec)
        normalize_financial_field_alias(rec)
        assert rec == before
        assert "financial_data" not in rec

    def test_none_canonical_falls_through(self):
        """If 'financial_data' is explicitly None, legacy key is used."""
        rec = {"ticker": "ACME", "financial_data": None, "financials": SAMPLE_FIN_DICT}
        normalize_financial_field_alias(rec)
        assert rec["financial_data"] is SAMPLE_FIN_DICT

    def test_none_legacy_ignored(self):
        """If 'financials' is None it should NOT overwrite missing canonical."""
        rec = {"ticker": "ACME", "financials": None}
        normalize_financial_field_alias(rec)
        assert rec.get("financial_data") is None

    def test_mutates_in_place(self):
        """Function must mutate the dict in-place (zero-copy)."""
        rec = _make_record("financials", SAMPLE_FIN_DICT)
        returned = normalize_financial_field_alias(rec)
        assert returned is rec

    def test_pit_fields_untouched(self):
        """PIT-sensitive fields inside the nested dict must not be modified."""
        fin = {"cash": 100, "source_date": "2025-06-30", "as_of_date": "2025-06-30"}
        rec = _make_record("financials", fin)
        normalize_financial_field_alias(rec)
        assert rec["financial_data"]["source_date"] == "2025-06-30"
        assert rec["financial_data"]["as_of_date"] == "2025-06-30"


class TestBothShapesProduceIdenticalValues:
    """Key acceptance criterion: identical downstream values regardless of input key."""

    def test_extracted_values_identical(self):
        """Module 2 field extraction must produce same results for both shapes."""
        fin = {
            "cash": 200_000_000,
            "debt": 50_000_000,
            "revenue_ttm": 5_000_000,
            "net_debt": -150_000_000,
            "source_date": "2025-11-01",
        }

        rec_canonical = {"ticker": "BIO1", "financial_data": copy.deepcopy(fin)}
        rec_legacy = {"ticker": "BIO1", "financials": copy.deepcopy(fin)}

        normalize_financial_field_alias(rec_canonical)
        normalize_financial_field_alias(rec_legacy)

        assert rec_canonical["financial_data"] == rec_legacy["financial_data"]
        # Verify all nested fields survived
        for key in fin:
            assert rec_canonical["financial_data"][key] == fin[key]
            assert rec_legacy["financial_data"][key] == fin[key]
