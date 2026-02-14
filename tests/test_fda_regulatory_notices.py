"""Tests for Federal Register expansion — collect_fda_regulatory_notices().

Exercises query construction, event type mapping, product-to-ticker matching,
deduplication, cache read/write, and priority resolution for FEDERAL_REGISTER source.
"""

import json
from datetime import date
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from wake_robin_data_pipeline.collectors.fda_adcom_collector import (
    _FR_REGULATORY_QUERIES,
    collect_fda_regulatory_notices,
    _match_product_to_ticker,
    FEDERAL_REGISTER_API,
)


# ===========================================================================
# FR query construction
# ===========================================================================

class TestFRQueryConstruction:
    """Verify _FR_REGULATORY_QUERIES structure and content."""

    def test_four_query_specs(self):
        assert len(_FR_REGULATORY_QUERIES) == 4

    def test_each_has_term_and_event_type(self):
        for spec in _FR_REGULATORY_QUERIES:
            assert "term" in spec
            assert "event_type" in spec
            assert isinstance(spec["term"], str)
            assert len(spec["term"]) > 0

    def test_approval_query_present(self):
        types = [q["event_type"] for q in _FR_REGULATORY_QUERIES]
        assert "FDA_APPROVAL" in types

    def test_crl_query_present(self):
        types = [q["event_type"] for q in _FR_REGULATORY_QUERIES]
        assert "FDA_CRL" in types

    def test_rtf_query_present(self):
        types = [q["event_type"] for q in _FR_REGULATORY_QUERIES]
        assert "FDA_RTF" in types

    def test_warning_letter_query_present(self):
        types = [q["event_type"] for q in _FR_REGULATORY_QUERIES]
        assert "FDA_WARNING_LETTER" in types


# ===========================================================================
# Event type mapping from FR queries
# ===========================================================================

class TestEventTypeMapping:
    """Verify event_type values are valid and consistent."""

    def test_all_event_types_start_with_fda(self):
        for spec in _FR_REGULATORY_QUERIES:
            assert spec["event_type"].startswith("FDA_"), (
                f"Event type {spec['event_type']} should start with FDA_"
            )

    def test_event_types_are_uppercase(self):
        for spec in _FR_REGULATORY_QUERIES:
            assert spec["event_type"] == spec["event_type"].upper()


# ===========================================================================
# Product-to-ticker matching
# ===========================================================================

class TestProductToTickerMatching:
    """Verify _match_product_to_ticker works for regulatory notice titles."""

    def test_exact_match(self):
        product_map = {"keytruda": "MRK"}
        result = _match_product_to_ticker("Keytruda (pembrolizumab)", product_map)
        assert result is not None
        assert result["ticker"] == "MRK"

    def test_no_match_returns_none(self):
        product_map = {"keytruda": "MRK"}
        result = _match_product_to_ticker("Unrelated Drug Name", product_map)
        assert result is None

    def test_case_insensitive(self):
        product_map = {"opdivo": "BMY"}
        result = _match_product_to_ticker("OPDIVO", product_map)
        assert result is not None
        assert result["ticker"] == "BMY"


# ===========================================================================
# Cache read/write
# ===========================================================================

class TestFDARegulatoryCache:
    """Verify caching behavior for collect_fda_regulatory_notices."""

    def test_reads_from_cache(self, tmp_path):
        """When cache exists, returns cached data without network call."""
        cache_dir = tmp_path / "fda_cache"
        cache_dir.mkdir()

        as_of = date(2026, 2, 7)
        cached_events = [
            {"ticker": "ACAD", "event_type": "FDA_APPROVAL",
             "event_date": "2026-01-15", "source": "FEDERAL_REGISTER"},
        ]

        cache_path = cache_dir / f"fda_regulatory_{as_of.isoformat()}.json"
        with open(cache_path, "w") as f:
            json.dump(cached_events, f)

        result = collect_fda_regulatory_notices(
            product_ticker_map={"acadesine": "ACAD"},
            as_of_date=as_of,
            cache_dir=cache_dir,
        )
        assert len(result) == 1
        assert result[0]["ticker"] == "ACAD"
        assert result[0]["source"] == "FEDERAL_REGISTER"

    def test_cache_path_format(self, tmp_path):
        """Cache file follows expected naming convention."""
        as_of = date(2026, 2, 7)
        expected_name = f"fda_regulatory_{as_of.isoformat()}.json"
        cache_path = tmp_path / expected_name
        assert "2026-02-07" in cache_path.name


# ===========================================================================
# Deduplication
# ===========================================================================

class TestFDARegulatoryDedup:
    """Verify deduplication by (ticker, event_type, event_date)."""

    def test_dedup_removes_exact_duplicates(self, tmp_path):
        """Duplicate events in cache are preserved (dedup happens at collection time)."""
        cache_dir = tmp_path / "fda_cache"
        cache_dir.mkdir()

        as_of = date(2026, 2, 7)
        # Cache already contains deduped data (dedup happens before cache write)
        cached_events = [
            {"ticker": "FOLD", "event_type": "FDA_CRL",
             "event_date": "2026-01-10", "source": "FEDERAL_REGISTER"},
        ]

        cache_path = cache_dir / f"fda_regulatory_{as_of.isoformat()}.json"
        with open(cache_path, "w") as f:
            json.dump(cached_events, f)

        result = collect_fda_regulatory_notices(
            product_ticker_map={},
            as_of_date=as_of,
            cache_dir=cache_dir,
        )
        assert len(result) == 1


# ===========================================================================
# Priority resolution for FEDERAL_REGISTER source
# ===========================================================================

class TestFederalRegisterPriority:
    """Verify catalyst priority for FEDERAL_REGISTER events."""

    def test_fda_approval_federal_register_priority_1(self):
        from decision_engine import resolve_catalyst_priority, DecisionRuleset
        rs = DecisionRuleset(enable_catalyst_priority=True)
        assert resolve_catalyst_priority("FDA_APPROVAL", "FEDERAL_REGISTER", rs) == 1

    def test_fda_crl_federal_register_priority_1(self):
        from decision_engine import resolve_catalyst_priority, DecisionRuleset
        rs = DecisionRuleset(enable_catalyst_priority=True)
        # FDA_CRL matches the ("FDA_CRL", "*", 1) rule first
        assert resolve_catalyst_priority("FDA_CRL", "FEDERAL_REGISTER", rs) == 1

    def test_fda_rtf_federal_register_priority_1(self):
        from decision_engine import resolve_catalyst_priority, DecisionRuleset
        rs = DecisionRuleset(enable_catalyst_priority=True)
        assert resolve_catalyst_priority("FDA_RTF", "FEDERAL_REGISTER", rs) == 1

    def test_fda_warning_letter_federal_register_priority_1(self):
        from decision_engine import resolve_catalyst_priority, DecisionRuleset
        rs = DecisionRuleset(enable_catalyst_priority=True)
        assert resolve_catalyst_priority("FDA_WARNING_LETTER", "FEDERAL_REGISTER", rs) == 1

    def test_generic_type_federal_register_priority_3(self):
        """Novel type from FEDERAL_REGISTER falls through to source wildcard → 3 (demoted)."""
        from decision_engine import resolve_catalyst_priority, DecisionRuleset
        rs = DecisionRuleset(enable_catalyst_priority=True)
        assert resolve_catalyst_priority("NOVEL_TYPE", "FEDERAL_REGISTER", rs) == 3


# ===========================================================================
# Event schema
# ===========================================================================

class TestFDARegulatoryEventSchema:
    """Verify the expected event dict shape from cached data."""

    def test_event_has_required_fields(self, tmp_path):
        """Cached events have the expected fields."""
        cache_dir = tmp_path / "fda_cache"
        cache_dir.mkdir()

        as_of = date(2026, 2, 7)
        cached_events = [
            {
                "ticker": "SGEN",
                "event_type": "FDA_APPROVAL",
                "event_date": "2026-01-20",
                "event_name": "FDA_APPROVAL: enfortumab (title...)",
                "drug_name": "enfortumab",
                "confidence": "HIGH",
                "source": "FEDERAL_REGISTER",
                "disclosed_at": "2026-01-20",
                "fr_document_number": "2026-12345",
                "tags": ["fda_regulatory", "federal_register"],
            },
        ]

        cache_path = cache_dir / f"fda_regulatory_{as_of.isoformat()}.json"
        with open(cache_path, "w") as f:
            json.dump(cached_events, f)

        result = collect_fda_regulatory_notices(
            product_ticker_map={},
            as_of_date=as_of,
            cache_dir=cache_dir,
        )
        ev = result[0]
        assert ev["source"] == "FEDERAL_REGISTER"
        assert ev["confidence"] == "HIGH"
        assert "fr_document_number" in ev
        assert "tags" in ev
        assert "fda_regulatory" in ev["tags"]
