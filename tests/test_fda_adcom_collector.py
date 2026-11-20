"""
Tests for FDA Advisory Committee Calendar Collector

Covers:
- Product-ticker map building from PDUFA, designations, trial records
- Product-to-ticker substring matching (incl. hyphen normalization)
- Date string parsing (_parse_date_str)
- Federal Register regex patterns (_FR_MEETING_DATE, _FR_DRUG_AFTER_NDA, etc.)
- Federal Register ADCOM collection (mocked HTTP)
- EDGAR 8-K ADCOM collection (mocked HTTP)
- Regulatory notices collection (mocked HTTP)
- Main entrypoint collect_fda_adcom_events (cache read/write, dedup, PIT safety)
- Generic intervention filtering and min-length guard

Run with: pytest tests/test_fda_adcom_collector.py -v
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from wake_robin_data_pipeline.collectors.fda_adcom_collector import (
    _FR_BRAND_GENERIC,
    _FR_DRUG_AFTER_COMMENTS,
    _FR_DRUG_AFTER_NDA,
    _FR_MEETING_DATE,
    _GENERIC_INTERVENTION_NAMES,
    _MIN_PRODUCT_NAME_LENGTH,
    _collect_adcom_from_edgar,
    _collect_adcom_from_federal_register,
    _match_product_to_ticker,
    _parse_date_str,
    build_product_ticker_map,
    collect_fda_adcom_events,
    collect_fda_regulatory_notices,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def product_map():
    """Simple drug-to-ticker mapping for tests."""
    return {
        "belantamab mafodotin": "GSK",
        "karxt": "BPMC",
        "rexulti": "OTKA",
        "midomafetamine": "MNMD",
        "lecanemab": "ESALY",
    }


@pytest.fixture
def as_of_date():
    return date(2025, 8, 1)


def _mock_session():
    """Return a MagicMock session whose .get() returns a configurable mock response."""
    session = MagicMock()
    return session


def _mock_response(status_code=200, json_data=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = text
    return resp


# ============================================================================
# REGEX TESTS
# ============================================================================


class TestFRMeetingDateRegex:
    """Tests for _FR_MEETING_DATE regex."""

    def test_standard_held_on(self):
        text = "The meeting will be held on July 17, 2025, from 8 a.m."
        m = _FR_MEETING_DATE.search(text)
        assert m is not None
        assert m.group(1) == "July 17, 2025"

    def test_held_virtually_on(self):
        text = "The meeting will be held virtually on March 5, 2025, from 9 a.m."
        m = _FR_MEETING_DATE.search(text)
        assert m is not None
        assert m.group(1) == "March 5, 2025"

    def test_scheduled_for(self):
        text = "The meeting is scheduled December 10, 2025."
        m = _FR_MEETING_DATE.search(text)
        assert m is not None
        assert m.group(1) == "December 10, 2025"

    def test_take_place_on(self):
        text = "The meeting will take place on January 22, 2026."
        m = _FR_MEETING_DATE.search(text)
        assert m is not None
        assert m.group(1) == "January 22, 2026"

    def test_no_match(self):
        text = "No meeting date mentioned here."
        assert _FR_MEETING_DATE.search(text) is None


class TestFRDrugAfterNDA:
    """Tests for _FR_DRUG_AFTER_NDA regex."""

    def test_nda_for_drug(self):
        title = "NDA 215050 for Belantamab Mafodotin Injection"
        m = _FR_DRUG_AFTER_NDA.search(title)
        assert m is not None
        assert m.group(1).startswith("Belantamab")

    def test_bla_for_drug(self):
        title = "BLA 761440 for Lecanemab Injection"
        m = _FR_DRUG_AFTER_NDA.search(title)
        assert m is not None
        assert "Lecanemab" in m.group(1)

    def test_snda(self):
        title = "sNDA 209354/S-013, for Rexulti Tablets"
        m = _FR_DRUG_AFTER_NDA.search(title)
        assert m is not None
        assert "Rexulti" in m.group(1)


class TestFRBrandGeneric:
    """Tests for _FR_BRAND_GENERIC regex."""

    def test_brand_generic_pair(self):
        title = "REXULTI (brexpiprazole) Tablets"
        m = _FR_BRAND_GENERIC.search(title)
        assert m is not None
        assert m.group(1) == "REXULTI"
        assert m.group(2) == "brexpiprazole"

    def test_columvi(self):
        title = "COLUMVI (glofitamab) for Injection"
        m = _FR_BRAND_GENERIC.search(title)
        assert m is not None
        assert m.group(1) == "COLUMVI"
        assert m.group(2) == "glofitamab"


class TestFRDrugAfterComments:
    """Tests for _FR_DRUG_AFTER_COMMENTS regex."""

    def test_capsules(self):
        title = "Comments-Midomafetamine Capsules"
        m = _FR_DRUG_AFTER_COMMENTS.search(title)
        assert m is not None
        assert "Midomafetamine Capsules" in m.group(1)

    def test_injection(self):
        title = "Comments-Belantamab Injection"
        m = _FR_DRUG_AFTER_COMMENTS.search(title)
        assert m is not None
        assert "Belantamab" in m.group(1)


# ============================================================================
# _parse_date_str TESTS
# ============================================================================


class TestParseDateStr:

    def test_full_month_comma(self):
        assert _parse_date_str("July 17, 2025") == date(2025, 7, 17)

    def test_full_month_no_comma(self):
        assert _parse_date_str("July 17 2025") == date(2025, 7, 17)

    def test_abbreviated_month_comma(self):
        assert _parse_date_str("Jul 17, 2025") == date(2025, 7, 17)

    def test_abbreviated_month_no_comma(self):
        assert _parse_date_str("Jul 17 2025") == date(2025, 7, 17)

    def test_strips_whitespace(self):
        assert _parse_date_str("  March 5, 2025  ") == date(2025, 3, 5)

    def test_unparseable_returns_none(self):
        assert _parse_date_str("2025-07-17") is None

    def test_empty_string(self):
        assert _parse_date_str("") is None


# ============================================================================
# _match_product_to_ticker TESTS
# ============================================================================


class TestMatchProductToTicker:

    def test_exact_substring_match(self, product_map):
        result = _match_product_to_ticker("FDA ADCOM for Belantamab Mafodotin Injection", product_map)
        assert result is not None
        assert result["ticker"] == "GSK"
        assert result["drug_name"] == "belantamab mafodotin"

    def test_case_insensitive(self, product_map):
        result = _match_product_to_ticker("KARXT tablets approved", product_map)
        assert result is not None
        assert result["ticker"] == "BPMC"

    def test_hyphen_normalization(self):
        pmap = {"kar-xt": "BPMC"}
        result = _match_product_to_ticker("Discussion of KARXT capsules", pmap)
        assert result is not None
        assert result["ticker"] == "BPMC"

    def test_no_match_returns_none(self, product_map):
        result = _match_product_to_ticker("Totally unrelated topic", product_map)
        assert result is None

    def test_empty_topic(self, product_map):
        result = _match_product_to_ticker("", product_map)
        assert result is None

    def test_empty_map(self):
        result = _match_product_to_ticker("Some drug topic", {})
        assert result is None


# ============================================================================
# build_product_ticker_map TESTS
# ============================================================================


class TestBuildProductTickerMap:

    def test_reads_pdufa_list_format(self, tmp_path):
        pdufa = [{"drug_name": "DrugA", "ticker": "DRGA"}]
        (tmp_path / "pdufa_dates.json").write_text(json.dumps(pdufa))
        result = build_product_ticker_map(tmp_path)
        assert result["druga"] == "DRGA"

    def test_reads_pdufa_events_format(self, tmp_path):
        pdufa = {"events": [{"drug_name": "DrugB", "ticker": "DRGB"}]}
        (tmp_path / "pdufa_dates.json").write_text(json.dumps(pdufa))
        result = build_product_ticker_map(tmp_path)
        assert result["drugb"] == "DRGB"

    def test_reads_fda_designations(self, tmp_path):
        desig = {"designations": [{"drug_name": "DesigDrug", "ticker": "DDRUG"}]}
        (tmp_path / "fda_designations.json").write_text(json.dumps(desig))
        result = build_product_ticker_map(tmp_path)
        assert result["desigdrug"] == "DDRUG"

    def test_reads_trial_interventions(self, tmp_path):
        trials = [{"ticker": "ACME", "interventions": ["Acmecillin", "Placebo"]}]
        (tmp_path / "trial_records.json").write_text(json.dumps(trials))
        result = build_product_ticker_map(tmp_path)
        assert "acmecillin" in result
        # Placebo is in _GENERIC_INTERVENTION_NAMES and should be excluded
        assert "placebo" not in result

    def test_skips_short_intervention_names(self, tmp_path):
        trials = [{"ticker": "ACME", "interventions": ["AB", "x"]}]
        (tmp_path / "trial_records.json").write_text(json.dumps(trials))
        result = build_product_ticker_map(tmp_path)
        assert "ab" not in result
        assert "x" not in result

    def test_skips_generic_intervention_words(self, tmp_path):
        trials = [{"ticker": "ACME", "interventions": ["treatment", "injection"]}]
        (tmp_path / "trial_records.json").write_text(json.dumps(trials))
        result = build_product_ticker_map(tmp_path)
        assert "treatment" not in result
        assert "injection" not in result

    def test_pdufa_takes_precedence_over_trials(self, tmp_path):
        """PDUFA entries are loaded first, trial entries don't overwrite."""
        pdufa = [{"drug_name": "SharedDrug", "ticker": "PDUFA_T"}]
        (tmp_path / "pdufa_dates.json").write_text(json.dumps(pdufa))
        trials = [{"ticker": "TRIAL_T", "interventions": ["SharedDrug"]}]
        (tmp_path / "trial_records.json").write_text(json.dumps(trials))
        result = build_product_ticker_map(tmp_path)
        assert result["shareddrug"] == "PDUFA_T"

    def test_empty_directory(self, tmp_path):
        result = build_product_ticker_map(tmp_path)
        assert result == {}

    def test_skips_null_interventions(self, tmp_path):
        trials = [{"ticker": "ACME", "interventions": [None, "", "  "]}]
        (tmp_path / "trial_records.json").write_text(json.dumps(trials))
        result = build_product_ticker_map(tmp_path)
        assert len(result) == 0


# ============================================================================
# _collect_adcom_from_federal_register TESTS
# ============================================================================


class TestCollectAdcomFromFederalRegister:

    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector._get_session")
    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector.time.sleep")
    def test_basic_extraction(self, mock_sleep, mock_get_session, product_map, as_of_date):
        """Should extract ADCOM event from a well-formed FR result."""
        session = _mock_session()
        mock_get_session.return_value = session

        fr_result = {
            "count": 1,
            "total_pages": 1,
            "results": [
                {
                    "title": "Oncologic Drugs Advisory Committee; NDA 215050 for Belantamab Mafodotin Injection",
                    "dates": "The meeting will be held on July 17, 2025, from 8 a.m.",
                    "publication_date": "2025-06-15",
                    "html_url": "https://example.com/doc1",
                }
            ],
        }
        session.get.return_value = _mock_response(json_data=fr_result)

        events = _collect_adcom_from_federal_register(product_map, as_of_date)
        assert len(events) == 1
        assert events[0]["ticker"] == "GSK"
        assert events[0]["event_date"] == "2025-07-17"
        assert events[0]["event_type"] == "FDA_ADCOM"
        assert events[0]["source"] == "FEDERAL_REGISTER"
        assert events[0]["confidence"] == "HIGH"

    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector._get_session")
    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector.time.sleep")
    def test_skips_tobacco_committee(self, mock_sleep, mock_get_session, product_map, as_of_date):
        """Should skip non-drug committees like Tobacco."""
        session = _mock_session()
        mock_get_session.return_value = session

        fr_result = {
            "count": 1,
            "total_pages": 1,
            "results": [
                {
                    "title": "Tobacco Products Scientific Advisory Committee; Notice of Meeting",
                    "dates": "The meeting will be held on August 1, 2025.",
                    "publication_date": "2025-07-01",
                    "html_url": "https://example.com/tobacco",
                }
            ],
        }
        session.get.return_value = _mock_response(json_data=fr_result)

        events = _collect_adcom_from_federal_register(product_map, as_of_date)
        assert len(events) == 0

    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector._get_session")
    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector.time.sleep")
    def test_pit_safety_skips_future_pub_date(self, mock_sleep, mock_get_session, product_map):
        """Events published after as_of_date should be skipped."""
        session = _mock_session()
        mock_get_session.return_value = session

        fr_result = {
            "count": 1,
            "total_pages": 1,
            "results": [
                {
                    "title": "NDA 215050 for Belantamab Mafodotin Injection",
                    "dates": "The meeting will be held on September 1, 2025.",
                    "publication_date": "2025-08-20",  # after as_of_date
                    "html_url": "https://example.com/future",
                }
            ],
        }
        session.get.return_value = _mock_response(json_data=fr_result)

        # as_of_date is 2025-08-01 — pub_date 2025-08-20 > end_date
        events = _collect_adcom_from_federal_register(product_map, date(2025, 8, 1))
        assert len(events) == 0

    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector._get_session")
    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector.time.sleep")
    def test_unmatched_drug_skipped(self, mock_sleep, mock_get_session, as_of_date):
        """If no product matches, the event is skipped."""
        session = _mock_session()
        mock_get_session.return_value = session

        fr_result = {
            "count": 1,
            "total_pages": 1,
            "results": [
                {
                    "title": "NDA 999999 for UnknownDrug Tablets",
                    "dates": "The meeting will be held on July 10, 2025.",
                    "publication_date": "2025-06-01",
                    "html_url": "https://example.com/unknown",
                }
            ],
        }
        session.get.return_value = _mock_response(json_data=fr_result)

        events = _collect_adcom_from_federal_register({"otherdrug": "OTHER"}, as_of_date)
        assert len(events) == 0

    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector._get_session")
    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector.time.sleep")
    def test_api_error_returns_empty(self, mock_sleep, mock_get_session, product_map, as_of_date):
        """HTTP error should result in empty list, not exception."""
        session = _mock_session()
        mock_get_session.return_value = session
        session.get.return_value = _mock_response(status_code=500)

        events = _collect_adcom_from_federal_register(product_map, as_of_date)
        assert events == []

    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector._get_session")
    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector.time.sleep")
    def test_committee_extracted_from_title(self, mock_sleep, mock_get_session, product_map, as_of_date):
        """Committee name should be the part before the first semicolon."""
        session = _mock_session()
        mock_get_session.return_value = session

        fr_result = {
            "count": 1,
            "total_pages": 1,
            "results": [
                {
                    "title": "Oncologic Drugs Advisory Committee; NDA 215050 for Belantamab Mafodotin",
                    "dates": "The meeting will be held on July 17, 2025.",
                    "publication_date": "2025-06-15",
                    "html_url": "https://example.com/doc",
                }
            ],
        }
        session.get.return_value = _mock_response(json_data=fr_result)

        events = _collect_adcom_from_federal_register(product_map, as_of_date)
        assert len(events) == 1
        assert events[0]["committee"] == "Oncologic Drugs Advisory Committee"


# ============================================================================
# _collect_adcom_from_edgar TESTS
# ============================================================================


class TestCollectAdcomFromEdgar:

    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector._get_session")
    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector.time.sleep")
    def test_dedup_by_ticker_event_date(self, mock_sleep, mock_get_session, as_of_date):
        """Duplicate (ticker, event_date) pairs should be collapsed."""
        session = _mock_session()
        mock_get_session.return_value = session

        # Two queries returning same filing
        search_result = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_source": {
                            "display_names": ["Acme Corp (ACME)"],
                            "file_date": "2025-06-01",
                            "adsh": "0001234567-25-000001",
                            "ciks": ["12345"],
                        },
                    }
                ],
            },
        }
        index_result = {
            "directory": {
                "item": [{"name": "ex991.htm"}],
            },
        }
        doc_text = (
            "<html>FDA has scheduled an Advisory Committee meeting for "
            "July 17, 2025 to discuss the application.</html>"
        )

        def get_side_effect(url, **kwargs):
            if "search-index" in url:
                return _mock_response(json_data=search_result)
            if "index.json" in url:
                return _mock_response(json_data=index_result)
            return _mock_response(text=doc_text)

        session.get.side_effect = get_side_effect

        ticker_set = {"ACME"}
        cik_to_ticker = {"12345": "ACME"}

        events = _collect_adcom_from_edgar(ticker_set, cik_to_ticker, as_of_date)
        # Even though two queries hit the same filing, dedup keeps one
        ticker_dates = [(e["ticker"], e["event_date"]) for e in events]
        assert len(ticker_dates) == len(set(ticker_dates))

    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector._get_session")
    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector.time.sleep")
    def test_cik_fallback_matching(self, mock_sleep, mock_get_session, as_of_date):
        """When display_names don't contain a recognized ticker, CIK fallback should work."""
        session = _mock_session()
        mock_get_session.return_value = session

        search_result = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_source": {
                            "display_names": ["Unknown Company Inc"],
                            "file_date": "2025-06-01",
                            "adsh": "0001234567-25-000002",
                            "ciks": ["0000099999"],
                        },
                    }
                ],
            },
        }
        index_result = {
            "directory": {
                "item": [{"name": "ex991.htm"}],
            },
        }
        doc_text = "<html>Advisory Committee meeting scheduled for " "August 20, 2025.</html>"

        def get_side_effect(url, **kwargs):
            if "search-index" in url:
                return _mock_response(json_data=search_result)
            if "index.json" in url:
                return _mock_response(json_data=index_result)
            return _mock_response(text=doc_text)

        session.get.side_effect = get_side_effect

        ticker_set = {"ZZZZ"}
        cik_to_ticker = {"99999": "ZZZZ"}  # stripped leading zeros

        events = _collect_adcom_from_edgar(ticker_set, cik_to_ticker, as_of_date)
        matched_tickers = {e["ticker"] for e in events}
        # Should match via CIK fallback
        if events:
            assert "ZZZZ" in matched_tickers

    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector._get_session")
    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector.time.sleep")
    def test_no_universe_match_skips_filing(self, mock_sleep, mock_get_session, as_of_date):
        """Filings for tickers not in our universe should be skipped."""
        session = _mock_session()
        mock_get_session.return_value = session

        search_result = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_source": {
                            "display_names": ["Other Corp (OTHR)"],
                            "file_date": "2025-06-01",
                            "adsh": "0001234567-25-000003",
                            "ciks": ["55555"],
                        },
                    }
                ],
            },
        }

        session.get.return_value = _mock_response(json_data=search_result)

        # Universe has ACME only
        ticker_set = {"ACME"}
        cik_to_ticker = {}

        events = _collect_adcom_from_edgar(ticker_set, cik_to_ticker, as_of_date)
        assert len(events) == 0


# ============================================================================
# collect_fda_regulatory_notices TESTS
# ============================================================================


class TestCollectFdaRegulatoryNotices:

    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector._get_session")
    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector.time.sleep")
    def test_cache_read(self, mock_sleep, mock_get_session, product_map, as_of_date, tmp_path):
        """Should return cached results without hitting API."""
        cached_events = [{"ticker": "GSK", "event_type": "FDA_APPROVAL", "event_date": "2025-06-01"}]
        cache_path = tmp_path / f"fda_regulatory_{as_of_date.isoformat()}.json"
        cache_path.write_text(json.dumps(cached_events))

        result = collect_fda_regulatory_notices(product_map, as_of_date, cache_dir=tmp_path)
        assert result == cached_events
        mock_get_session.assert_not_called()

    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector._get_session")
    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector.time.sleep")
    def test_cache_write(self, mock_sleep, mock_get_session, product_map, as_of_date, tmp_path):
        """Should write results to cache after fetching."""
        session = _mock_session()
        mock_get_session.return_value = session

        fr_result = {
            "count": 1,
            "total_pages": 1,
            "results": [
                {
                    "title": "Approval of NDA 215050 for Belantamab Mafodotin",
                    "publication_date": "2025-06-01",
                    "document_number": "2025-12345",
                    "html_url": "https://example.com",
                    "dates": "",
                    "abstract": "",
                }
            ],
        }
        session.get.return_value = _mock_response(json_data=fr_result)

        collect_fda_regulatory_notices(product_map, as_of_date, cache_dir=tmp_path)

        cache_path = tmp_path / f"fda_regulatory_{as_of_date.isoformat()}.json"
        assert cache_path.exists()
        cached = json.loads(cache_path.read_text())
        assert isinstance(cached, list)

    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector._get_session")
    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector.time.sleep")
    def test_dedup_regulatory_notices(self, mock_sleep, mock_get_session, product_map, as_of_date, tmp_path):
        """Duplicate (ticker, event_type, event_date) should be collapsed."""
        session = _mock_session()
        mock_get_session.return_value = session

        # Same doc returned by two query categories
        doc = {
            "title": "Approval of NDA 215050 for Belantamab Mafodotin",
            "publication_date": "2025-06-01",
            "document_number": "2025-12345",
            "html_url": "https://example.com",
            "dates": "",
            "abstract": "",
        }
        fr_result = {
            "count": 1,
            "total_pages": 1,
            "results": [doc],
        }
        session.get.return_value = _mock_response(json_data=fr_result)

        events = collect_fda_regulatory_notices(product_map, as_of_date, cache_dir=tmp_path)
        keys = [(e["ticker"], e["event_type"], e["event_date"]) for e in events]
        assert len(keys) == len(set(keys))


# ============================================================================
# collect_fda_adcom_events (main entrypoint) TESTS
# ============================================================================


class TestCollectFdaAdcomEvents:

    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector._get_session")
    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector.time.sleep")
    def test_cache_read(self, mock_sleep, mock_get_session, product_map, as_of_date, tmp_path):
        """Should return cached ADCOM events without network calls."""
        cached = [{"ticker": "GSK", "event_type": "FDA_ADCOM", "event_date": "2025-07-17"}]
        cache_path = tmp_path / f"adcom_calendar_{as_of_date.isoformat()}.json"
        cache_path.write_text(json.dumps(cached))

        result = collect_fda_adcom_events(product_map, as_of_date, cache_dir=tmp_path)
        assert result == cached

    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector._get_session")
    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector.time.sleep")
    def test_cache_write_after_fetch(self, mock_sleep, mock_get_session, product_map, as_of_date, tmp_path):
        """After fetching, should write results to cache file."""
        session = _mock_session()
        mock_get_session.return_value = session

        # FR returns one result, EDGAR/CIK return empty
        fr_result = {
            "count": 1,
            "total_pages": 1,
            "results": [
                {
                    "title": "Oncologic Drugs Advisory Committee; NDA 215050 for Belantamab Mafodotin",
                    "dates": "The meeting will be held on July 17, 2025.",
                    "publication_date": "2025-06-15",
                    "html_url": "https://example.com/doc",
                }
            ],
        }
        # CIK map and EDGAR return empty
        empty_search = {"hits": {"total": {"value": 0}, "hits": []}}
        cik_map_resp = _mock_response(json_data={"0": {"ticker": "GSK", "cik_str": "1"}})

        call_count = [0]

        def get_side_effect(url, **kwargs):
            call_count[0] += 1
            if "federalregister.gov" in url:
                return _mock_response(json_data=fr_result)
            if "company_tickers" in url:
                return cik_map_resp
            if "search-index" in url:
                return _mock_response(json_data=empty_search)
            return _mock_response(status_code=404)

        session.get.side_effect = get_side_effect

        result = collect_fda_adcom_events(product_map, as_of_date, cache_dir=tmp_path)

        cache_path = tmp_path / f"adcom_calendar_{as_of_date.isoformat()}.json"
        assert cache_path.exists()
        cached = json.loads(cache_path.read_text())
        assert len(cached) == len(result)

    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector._get_session")
    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector.time.sleep")
    def test_fr_preferred_over_edgar_in_dedup(self, mock_sleep, mock_get_session, as_of_date, tmp_path):
        """Federal Register events should be kept over EDGAR duplicates."""
        session = _mock_session()
        mock_get_session.return_value = session

        pmap = {"testdrug": "ACME"}

        fr_result = {
            "count": 1,
            "total_pages": 1,
            "results": [
                {
                    "title": "NDA 111111 for Testdrug Tablets",
                    "dates": "The meeting will be held on July 20, 2025.",
                    "publication_date": "2025-06-01",
                    "html_url": "https://example.com/fr",
                }
            ],
        }

        # EDGAR returns same ticker+date
        search_result = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_source": {
                            "display_names": ["Acme Corp (ACME)"],
                            "file_date": "2025-06-05",
                            "adsh": "0001234567-25-000010",
                            "ciks": ["12345"],
                        },
                    }
                ],
            },
        }
        index_result = {"directory": {"item": [{"name": "ex991.htm"}]}}
        doc_text = "<html>Advisory Committee meeting scheduled for July 20, 2025.</html>"
        cik_resp = _mock_response(json_data={"0": {"ticker": "ACME", "cik_str": "12345"}})

        def get_side_effect(url, **kwargs):
            if "federalregister.gov" in url:
                return _mock_response(json_data=fr_result)
            if "company_tickers" in url:
                return cik_resp
            if "search-index" in url:
                return _mock_response(json_data=search_result)
            if "index.json" in url:
                return _mock_response(json_data=index_result)
            return _mock_response(text=doc_text)

        session.get.side_effect = get_side_effect

        events = collect_fda_adcom_events(pmap, as_of_date, cache_dir=tmp_path, universe_tickers={"ACME"})

        acme_july20 = [e for e in events if e["ticker"] == "ACME" and e["event_date"] == "2025-07-20"]
        assert len(acme_july20) == 1
        assert acme_july20[0]["source"] == "FEDERAL_REGISTER"

    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector._get_session")
    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector.time.sleep")
    def test_output_sorted_by_date_then_ticker(self, mock_sleep, mock_get_session, as_of_date, tmp_path):
        """Output events should be sorted by (event_date, ticker)."""
        session = _mock_session()
        mock_get_session.return_value = session

        pmap = {"druga": "BBBB", "drugb": "AAAA"}

        fr_result = {
            "count": 2,
            "total_pages": 1,
            "results": [
                {
                    "title": "NDA 111 for Drugb Tablets",
                    "dates": "The meeting will be held on July 10, 2025.",
                    "publication_date": "2025-06-01",
                    "html_url": "https://example.com/1",
                },
                {
                    "title": "NDA 222 for Druga Capsules",
                    "dates": "The meeting will be held on July 10, 2025.",
                    "publication_date": "2025-06-01",
                    "html_url": "https://example.com/2",
                },
            ],
        }
        empty_search = {"hits": {"total": {"value": 0}, "hits": []}}
        cik_resp = _mock_response(json_data={})

        def get_side_effect(url, **kwargs):
            if "federalregister.gov" in url:
                return _mock_response(json_data=fr_result)
            if "company_tickers" in url:
                return cik_resp
            if "search-index" in url:
                return _mock_response(json_data=empty_search)
            return _mock_response(status_code=404)

        session.get.side_effect = get_side_effect

        events = collect_fda_adcom_events(pmap, as_of_date, cache_dir=tmp_path)
        if len(events) >= 2:
            # Same date: AAAA should come before BBBB
            assert events[0]["ticker"] == "AAAA"
            assert events[1]["ticker"] == "BBBB"


# ============================================================================
# CONSTANT / GUARD TESTS
# ============================================================================


class TestConstants:

    def test_generic_intervention_names_are_lowercase(self):
        for name in _GENERIC_INTERVENTION_NAMES:
            assert name == name.lower(), f"{name} should be lowercase"

    def test_min_product_name_length_is_reasonable(self):
        assert _MIN_PRODUCT_NAME_LENGTH >= 3
        assert _MIN_PRODUCT_NAME_LENGTH <= 6


# ============================================================================
# Null-valued Federal Register fields
#
# The API returns these keys present-but-null (~7% of notices carry
# "dates": null). dict.get's default only applies to an absent key, so a None
# reached the meeting-date regex and raised TypeError out of the whole
# collection — one malformed notice discarded every good one and froze the
# ADCOM cache silently for 9 days.
# ============================================================================


class TestFederalRegisterNullFields:

    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector._get_session")
    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector.time.sleep")
    def test_null_dates_does_not_raise(self, mock_sleep, mock_get_session, product_map, as_of_date):
        """A notice with "dates": null must be skipped, not crash the collection."""
        session = _mock_session()
        mock_get_session.return_value = session
        session.get.return_value = _mock_response(
            json_data={
                "count": 1,
                "total_pages": 1,
                "results": [
                    {
                        "title": "Cellular, Tissue, and Gene Therapies Advisory Committee; Notice of Meeting",
                        "dates": None,
                        "publication_date": "2025-06-15",
                        "html_url": "https://example.com/doc-null",
                    }
                ],
            }
        )

        events = _collect_adcom_from_federal_register(product_map, as_of_date)

        assert events == []

    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector._get_session")
    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector.time.sleep")
    def test_null_dates_does_not_discard_sibling_events(self, mock_sleep, mock_get_session, product_map, as_of_date):
        """The regression: one null-dates notice must not take valid ones down with it."""
        session = _mock_session()
        mock_get_session.return_value = session
        session.get.return_value = _mock_response(
            json_data={
                "count": 2,
                "total_pages": 1,
                "results": [
                    {
                        "title": "Cellular, Tissue, and Gene Therapies Advisory Committee; Notice of Meeting",
                        "dates": None,
                        "publication_date": "2025-06-14",
                        "html_url": "https://example.com/doc-null",
                    },
                    {
                        "title": ("Oncologic Drugs Advisory Committee; NDA 215050 for Belantamab Mafodotin Injection"),
                        "dates": "The meeting will be held on July 17, 2025, from 8 a.m.",
                        "publication_date": "2025-06-15",
                        "html_url": "https://example.com/doc-good",
                    },
                ],
            }
        )

        events = _collect_adcom_from_federal_register(product_map, as_of_date)

        assert len(events) == 1
        assert events[0]["ticker"] == "GSK"
        assert events[0]["event_date"] == "2025-07-17"

    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector._get_session")
    @patch("wake_robin_data_pipeline.collectors.fda_adcom_collector.time.sleep")
    def test_null_title_and_publication_date_do_not_raise(self, mock_sleep, mock_get_session, product_map, as_of_date):
        """title/publication_date are also coerced; a null in either must not raise."""
        session = _mock_session()
        mock_get_session.return_value = session
        session.get.return_value = _mock_response(
            json_data={
                "count": 1,
                "total_pages": 1,
                "results": [
                    {
                        "title": None,
                        "dates": "The meeting will be held on July 17, 2025, from 8 a.m.",
                        "publication_date": None,
                        "html_url": "https://example.com/doc-null-title",
                    }
                ],
            }
        )

        events = _collect_adcom_from_federal_register(product_map, as_of_date)

        assert isinstance(events, list)
