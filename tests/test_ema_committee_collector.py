"""
Tests for EMA committee agenda + meeting highlights collectors.

All tests use inline HTML fixtures + monkeypatched _fetch — zero network calls.
"""

import io
import json
import sys
import zipfile
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_MOD = "wake_robin_data_pipeline.collectors.ema_committee_collector"


# ---------------------------------------------------------------------------
# Inline HTML fixtures
# ---------------------------------------------------------------------------

# Realistic EMA committee landing page structure with div.bcl-file blocks.
# Anchor text is "View"; the descriptive title and "First published" date
# live in the enclosing div.bcl-file container.
COMMITTEE_LANDING_HTML = """
<html>
<head><title>CHMP | European Medicines Agency</title></head>
<body>
<h1>Committee for Medicinal Products for Human Use (CHMP)</h1>
<div class="field--items">
  <div class="bcl-file">
    <span>Agenda of the CHMP meeting 8 - 11 December 2025</span>
    <span>First published:05/12/2025</span>
    <div class="file-language-links">
      <a href="/en/documents/agenda/agenda-chmp-meeting-8-11-december-2025_en.pdf">View</a>
    </div>
  </div>
  <div class="bcl-file">
    <span>Annex to agenda of the CHMP meeting 8 - 11 December 2025</span>
    <span>First published:05/12/2025</span>
    <div class="file-language-links">
      <a href="/en/documents/annex/annex-to-agenda-chmp-meeting-8-11-december-2025_en.xlsx">View</a>
    </div>
  </div>
  <div class="bcl-file">
    <span>Meeting highlights from the CHMP 8 - 11 December 2025</span>
    <div class="file-language-links">
      <a href="/en/news/meeting-highlights-committee-medicinal-products-human-use-chmp-8-11-december-2025">View</a>
    </div>
  </div>
  <div class="bcl-file">
    <span>Meeting highlights from the CHMP 10 - 13 November 2025</span>
    <div class="file-language-links">
      <a href="/en/news/meeting-highlights-committee-medicinal-products-human-use-chmp-10-13-november-2025">View</a>
    </div>
  </div>
  <div class="bcl-file">
    <span>Agenda of the CHMP meeting 27 - 30 January 2026</span>
    <span>First published:24/01/2026</span>
    <div class="file-language-links">
      <a href="/en/documents/agenda/agenda-chmp-meeting-27-30-january-2026_en.pdf">View</a>
    </div>
  </div>
</div>
</body>
</html>
"""

# Legacy-style HTML where the anchor text itself contains the title (fallback path).
COMMITTEE_LANDING_HTML_LEGACY = """
<html>
<head><title>CHMP | European Medicines Agency</title></head>
<body>
<h1>Committee for Medicinal Products for Human Use (CHMP)</h1>
<div class="field--items">
  <a href="/en/documents/agenda/agenda-chmp-meeting-8-11-december-2025">
    Agenda of the CHMP meeting 8 - 11 December 2025
  </a>
  <a href="/en/news/meeting-highlights-committee-medicinal-products-human-use-chmp-8-11-december-2025">
    Meeting highlights from the CHMP 8 - 11 December 2025
  </a>
</div>
</body>
</html>
"""

AGENDA_HTML = """
<html>
<head><title>Agenda of the CHMP meeting 27 - 30 January 2026</title></head>
<body>
<h1>Agenda of the CHMP meeting 27 - 30 January 2026</h1>
<ul>
  <li>Marketing Authorisation Application for Aficamten (company: Cytokinetics)</li>
  <li>Type II Variation for Keytruda (pembrolizumab) — new indication</li>
  <li>Referral procedure for Unmatched Drug XYZ</li>
  <li>Administrative matters (no product)</li>
</ul>
</body>
</html>
"""

HIGHLIGHTS_HTML = """
<html>
<head>
  <title>Meeting highlights from the CHMP 8 - 11 December 2025</title>
  <meta name="dcterms.issued" content="2025-12-11" />
</head>
<body>
<h1>Meeting highlights from the CHMP 8 - 11 December 2025</h1>
<h2>Medicines recommended for approval</h2>
<ul>
  <li>Aficamten — for obstructive hypertrophic cardiomyopathy</li>
  <li>Patisiran — for hereditary transthyretin amyloidosis</li>
</ul>
<h2>Negative opinions</h2>
<p>No negative opinions were adopted at this meeting.</p>
<h2>Withdrawn applications</h2>
<ul>
  <li>Keytruda — withdrawn by marketing authorisation holder</li>
</ul>
<h2>Safety referrals</h2>
<ul>
  <li>Unmapped Safety Drug — referral started</li>
</ul>
</body>
</html>
"""

# Product-ticker map for testing
PRODUCT_MAP = {
    "aficamten": "CYTK",
    "keytruda": "MRK",
    "pembrolizumab": "MRK",
    "patisiran": "ALNY",
}


# ===========================================================================
# Discovery tests
# ===========================================================================


class TestDiscovery:
    """Tests for _discover_links."""

    def test_discovers_agenda_links_from_bcl_file(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _discover_links

        links = _discover_links(COMMITTEE_LANDING_HTML, "https://www.ema.europa.eu")
        assert len(links["agenda_docs"]) == 3  # 2 agenda + 1 annex
        urls = [d["url"] for d in links["agenda_docs"]]
        assert any("agenda-chmp-meeting-8-11-december-2025" in u for u in urls)
        assert any("annex" in u for u in urls)
        assert any("27-30-january-2026" in u for u in urls)

    def test_discovers_highlights_links_from_bcl_file(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _discover_links

        links = _discover_links(COMMITTEE_LANDING_HTML, "https://www.ema.europa.eu")
        assert len(links["highlights"]) == 2
        urls = [d["url"] for d in links["highlights"]]
        assert any("8-11-december-2025" in u for u in urls)
        assert any("10-13-november-2025" in u for u in urls)

    def test_resolves_relative_urls(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _discover_links

        links = _discover_links(COMMITTEE_LANDING_HTML, "https://www.ema.europa.eu")
        for doc in links["agenda_docs"] + links["highlights"]:
            assert doc["url"].startswith("https://"), f"URL not absolute: {doc['url']}"

    def test_title_from_bcl_file_block(self):
        """Title comes from the div.bcl-file text, not the anchor text."""
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _discover_links

        links = _discover_links(COMMITTEE_LANDING_HTML, "https://www.ema.europa.eu")
        titles = [d["title"] for d in links["agenda_docs"]]
        assert any("Agenda of the CHMP meeting 8 - 11 December 2025" in t for t in titles)
        # Anchor text "View" should NOT be the title
        assert not any(t == "View" for t in titles)

    def test_disclosed_at_from_first_published(self):
        """disclosed_at extracted from 'First published:DD/MM/YYYY' in block."""
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _discover_links

        links = _discover_links(COMMITTEE_LANDING_HTML, "https://www.ema.europa.eu")
        # The December 2025 agenda has "First published:05/12/2025"
        dec_agenda = [d for d in links["agenda_docs"]
                      if "8-11-december-2025" in d["url"] and "annex" not in d["url"]]
        assert len(dec_agenda) == 1
        assert dec_agenda[0]["disclosed_at"] == "2025-12-05"

    def test_legacy_anchor_text_fallback(self):
        """Falls back to anchor-text matching when no div.bcl-file blocks."""
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _discover_links

        links = _discover_links(COMMITTEE_LANDING_HTML_LEGACY, "https://www.ema.europa.eu")
        assert len(links["agenda_docs"]) == 1
        assert "agenda-chmp-meeting-8-11-december-2025" in links["agenda_docs"][0]["url"]
        assert len(links["highlights"]) == 1


# ===========================================================================
# Date parsing tests
# ===========================================================================


class TestParseMeetingDates:
    """Tests for _parse_meeting_dates."""

    def test_range_format(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _parse_meeting_dates

        result = _parse_meeting_dates("8 - 11 December 2025")
        assert result == ("2025-12-08", "2025-12-11")

    def test_dash_variants(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _parse_meeting_dates

        # En-dash
        result = _parse_meeting_dates("8\u201311 December 2025")
        assert result == ("2025-12-08", "2025-12-11")

    def test_single_day_meeting(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _parse_meeting_dates

        result = _parse_meeting_dates("15 January 2026")
        assert result == ("2026-01-15", "2026-01-15")

    def test_embedded_in_title(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _parse_meeting_dates

        result = _parse_meeting_dates(
            "Meeting highlights from the CHMP 8 - 11 December 2025"
        )
        assert result == ("2025-12-08", "2025-12-11")

    def test_no_date_returns_none(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _parse_meeting_dates

        assert _parse_meeting_dates("No date here") is None
        assert _parse_meeting_dates("") is None


# ===========================================================================
# Medicine matching tests
# ===========================================================================


class TestMedicineMatching:
    """Tests for _match_medicine_to_ticker."""

    def test_exact_match(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _match_medicine_to_ticker

        assert _match_medicine_to_ticker("Aficamten", None, PRODUCT_MAP) == "CYTK"

    def test_case_insensitive(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _match_medicine_to_ticker

        assert _match_medicine_to_ticker("AFICAMTEN", None, PRODUCT_MAP) == "CYTK"

    def test_substance_match(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _match_medicine_to_ticker

        assert _match_medicine_to_ticker("Unknown Brand", "pembrolizumab", PRODUCT_MAP) == "MRK"

    def test_substring_match(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _match_medicine_to_ticker

        assert _match_medicine_to_ticker(
            "Marketing Authorisation Application for Aficamten (company: Cytokinetics)",
            None, PRODUCT_MAP
        ) == "CYTK"

    def test_no_match_returns_none(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _match_medicine_to_ticker

        assert _match_medicine_to_ticker("Unknown Drug XYZ", None, PRODUCT_MAP) is None


# ===========================================================================
# Pharma token stripping + hardened matching
# ===========================================================================


class TestStripPharmaTokens:
    """Tests for _strip_pharma_tokens."""

    def test_strips_salt_form(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _strip_pharma_tokens

        assert _strip_pharma_tokens("aficamten hydrochloride") == "aficamten"

    def test_strips_multiple_salt_tokens(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _strip_pharma_tokens

        assert _strip_pharma_tokens("pembrolizumab sodium dihydrate") == "pembrolizumab"

    def test_strips_dosage(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _strip_pharma_tokens

        assert _strip_pharma_tokens("aficamten 10mg tablets") == "aficamten"

    def test_strips_parenthetical(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _strip_pharma_tokens

        assert _strip_pharma_tokens("doxorubicin (liposomal)") == "doxorubicin"

    def test_strips_formulation_suffix(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _strip_pharma_tokens

        result = _strip_pharma_tokens("aficamten film-coated tablets")
        assert result == "aficamten"

    def test_preserves_hyphenated_drug_name(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _strip_pharma_tokens

        assert _strip_pharma_tokens("axs-05") == "axs-05"

    def test_noop_on_clean_name(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _strip_pharma_tokens

        assert _strip_pharma_tokens("aficamten") == "aficamten"


class TestHardenedMatching:
    """Tests for salt-form / dosage-stripped matching fallback."""

    def test_salt_form_medicine_name_matches(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _match_medicine_to_ticker

        assert _match_medicine_to_ticker("Aficamten hydrochloride", None, PRODUCT_MAP) == "CYTK"

    def test_salt_form_active_substance_matches(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _match_medicine_to_ticker

        assert _match_medicine_to_ticker("Unknown Brand", "pembrolizumab sodium", PRODUCT_MAP) == "MRK"

    def test_dosage_token_stripped_match(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _match_medicine_to_ticker

        assert _match_medicine_to_ticker("Keytruda 25mg solution", None, PRODUCT_MAP) == "MRK"

    def test_parenthetical_form_stripped(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _match_medicine_to_ticker

        assert _match_medicine_to_ticker("Patisiran (liposomal)", None, PRODUCT_MAP) == "ALNY"

    def test_salt_form_on_map_side(self):
        """Map key contains salt form — stripped match should still work."""
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _match_medicine_to_ticker

        salt_map = {"forodesine hydrochloride": "BCRX"}
        assert _match_medicine_to_ticker("Forodesine", None, salt_map) == "BCRX"

    def test_still_returns_none_for_truly_unknown(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _match_medicine_to_ticker

        assert _match_medicine_to_ticker(
            "Completely Unknown Drug hydrochloride", None, PRODUCT_MAP
        ) is None


class TestUnmatchedTop:
    """Tests for unmatched_top frequency tracking in stats."""

    def test_unmatched_top_in_stats(self, tmp_path):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import (
            collect_ema_committee_events,
        )

        # Build XLSX with unmatched items that repeat
        xlsx_bytes = _build_test_xlsx([
            _make_row("MAA", "UnknownDrugA", "sub_a", "CompanyA"),
            _make_row("MAA", "UnknownDrugA", "sub_a2", "CompanyA"),
            _make_row("MAA", "UnknownDrugB", "sub_b", "CompanyB"),
            _make_row("MAA", "Aficamten", "aficamten", "Cytokinetics"),
        ])

        def mock_fetch(url):
            if "committees/committee-medicinal-products" in url:
                return LANDING_WITH_EVENT_PAGES_HTML
            if "/en/events/" in url:
                return EVENT_PAGE_HTML
            return "<html></html>"

        with patch(f"{_MOD}._fetch", side_effect=mock_fetch), \
             patch(f"{_MOD}._fetch_xlsx_bytes", return_value=xlsx_bytes), \
             patch(f"{_MOD}._fetch_xml", side_effect=Exception("no RSS in tests")):
            collect_ema_committee_events(
                as_of_date=date(2026, 2, 28),
                cache_dir=tmp_path,
                product_ticker_map=PRODUCT_MAP,
                committees=("CHMP",),
            )

        cache_path = tmp_path / "ema_committee_events_2026-02-28.json"
        payload = json.loads(cache_path.read_text())
        top = payload["stats"]["unmatched_top"]
        assert isinstance(top, list)
        # UnknownDrugA appears twice, UnknownDrugB once
        names = [entry["name"] for entry in top]
        assert "UnknownDrugA" in names
        assert "UnknownDrugB" in names
        # Most frequent first
        assert top[0]["count"] >= top[-1]["count"]


# ===========================================================================
# Outcome code classification
# ===========================================================================


class TestOutcomeClassification:
    """Tests for _classify_outcome_code."""

    def test_positive_opinion(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _classify_outcome_code

        assert _classify_outcome_code("Medicines recommended for approval") == "positive_opinion"

    def test_withdrawn(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _classify_outcome_code

        assert _classify_outcome_code("Withdrawn applications") == "withdrawn"

    def test_safety_signal(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _classify_outcome_code

        assert _classify_outcome_code("Safety referrals") == "referral_started"

    def test_unknown_defaults_to_other(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _classify_outcome_code

        assert _classify_outcome_code("Miscellaneous items") == "other"


# ===========================================================================
# Agenda parsing
# ===========================================================================


class TestAgendaParsing:
    """Tests for _parse_agenda_page."""

    def test_extracts_matched_items(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _parse_agenda_page

        events, unmatched = _parse_agenda_page(
            AGENDA_HTML, "https://example.com/agenda", "CHMP",
            "2026-01-27", "2026-01-30", PRODUCT_MAP,
        )
        tickers = [e["ticker"] for e in events]
        assert "CYTK" in tickers
        assert "MRK" in tickers

    def test_unmatched_items_collected(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _parse_agenda_page

        events, unmatched = _parse_agenda_page(
            AGENDA_HTML, "https://example.com/agenda", "CHMP",
            "2026-01-27", "2026-01-30", PRODUCT_MAP,
        )
        # "Unmatched Drug XYZ" and admin items should be unmatched
        assert len(unmatched) > 0

    def test_event_schema_fields(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _parse_agenda_page

        events, _ = _parse_agenda_page(
            AGENDA_HTML, "https://example.com/agenda", "CHMP",
            "2026-01-27", "2026-01-30", PRODUCT_MAP,
        )
        if events:
            ev = events[0]
            assert ev["schema"] == "ema_committee_event.v1"
            assert ev["jurisdiction"] == "EU"
            assert ev["committee"] == "CHMP"
            assert ev["meeting_start"] == "2026-01-27"
            assert ev["meeting_end"] == "2026-01-30"
            assert ev["event_date"] == "2026-01-30"
            assert ev["confidence"] == "MED"
            assert ev["item_type"] == "agenda_item"
            assert ev["id"].startswith("EMA_CHMP_AGENDA_2026-01-30_")


# ===========================================================================
# Meeting highlights parsing
# ===========================================================================


class TestMeetingHighlightsParsing:
    """Tests for _parse_meeting_highlights."""

    def test_extracts_outcomes_with_correct_codes(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _parse_meeting_highlights

        outcomes, unmatched = _parse_meeting_highlights(
            HIGHLIGHTS_HTML, "https://example.com/highlights", "CHMP", PRODUCT_MAP,
        )
        outcome_map = {o["ticker"]: o for o in outcomes}

        # Aficamten and Patisiran should be positive_opinion
        assert "CYTK" in outcome_map
        assert outcome_map["CYTK"]["outcome_code"] == "positive_opinion"
        assert "ALNY" in outcome_map
        assert outcome_map["ALNY"]["outcome_code"] == "positive_opinion"

        # Keytruda should be withdrawn
        assert "MRK" in outcome_map
        assert outcome_map["MRK"]["outcome_code"] == "withdrawn"

    def test_dates_extracted_from_title(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _parse_meeting_highlights

        outcomes, _ = _parse_meeting_highlights(
            HIGHLIGHTS_HTML, "https://example.com/highlights", "CHMP", PRODUCT_MAP,
        )
        if outcomes:
            assert outcomes[0]["meeting_start"] == "2025-12-08"
            assert outcomes[0]["meeting_end"] == "2025-12-11"

    def test_disclosed_at_from_meta(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _parse_meeting_highlights

        outcomes, _ = _parse_meeting_highlights(
            HIGHLIGHTS_HTML, "https://example.com/highlights", "CHMP", PRODUCT_MAP,
        )
        if outcomes:
            assert outcomes[0]["disclosed_at"] == "2025-12-11"

    def test_outcome_schema_fields(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _parse_meeting_highlights

        outcomes, _ = _parse_meeting_highlights(
            HIGHLIGHTS_HTML, "https://example.com/highlights", "CHMP", PRODUCT_MAP,
        )
        if outcomes:
            ev = outcomes[0]
            assert ev["schema"] == "ema_meeting_outcome.v1"
            assert ev["confidence"] == "HIGH"
            assert ev["source"] == "EMA_MEETING_HIGHLIGHTS"
            assert ev["jurisdiction"] == "EU"
            assert ev["id"].startswith("EMA_CHMP_OUTCOME_2025-12-11_")


# ===========================================================================
# ID stability
# ===========================================================================


class TestIdStability:
    """IDs are deterministic and stable across calls."""

    def test_stable_outcome_ids(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _parse_meeting_highlights

        outcomes1, _ = _parse_meeting_highlights(
            HIGHLIGHTS_HTML, "https://example.com/highlights", "CHMP", PRODUCT_MAP,
        )
        outcomes2, _ = _parse_meeting_highlights(
            HIGHLIGHTS_HTML, "https://example.com/highlights", "CHMP", PRODUCT_MAP,
        )
        ids1 = [o["id"] for o in outcomes1]
        ids2 = [o["id"] for o in outcomes2]
        assert ids1 == ids2

    def test_unique_ids(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _parse_meeting_highlights

        outcomes, _ = _parse_meeting_highlights(
            HIGHLIGHTS_HTML, "https://example.com/highlights", "CHMP", PRODUCT_MAP,
        )
        ids = [o["id"] for o in outcomes]
        assert len(ids) == len(set(ids)), "Duplicate IDs found"


# ===========================================================================
# Cache write and reload
# ===========================================================================


class TestCacheWriteReload:
    """Tests for cache determinism."""

    def test_cache_write_and_reload(self, tmp_path):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import (
            collect_ema_meeting_outcomes,
        )

        def mock_fetch(url):
            if "committee" in url or "chmp" in url.lower():
                return COMMITTEE_LANDING_HTML
            if "meeting-highlights" in url:
                return HIGHLIGHTS_HTML
            return "<html></html>"

        with patch("wake_robin_data_pipeline.collectors.ema_committee_collector._fetch",
                    side_effect=mock_fetch):
            result1 = collect_ema_meeting_outcomes(
                as_of_date=date(2025, 12, 31),
                cache_dir=tmp_path,
                product_ticker_map=PRODUCT_MAP,
                committees=("CHMP",),
            )

        # Second call should hit cache
        result2 = collect_ema_meeting_outcomes(
            as_of_date=date(2025, 12, 31),
            cache_dir=tmp_path,
            product_ticker_map=PRODUCT_MAP,
            committees=("CHMP",),
        )

        assert len(result1) == len(result2)
        assert [e["id"] for e in result1] == [e["id"] for e in result2]

        # Verify cache file is valid JSON
        cache_path = tmp_path / "ema_meeting_outcomes_2025-12-31.json"
        assert cache_path.exists()
        payload = json.loads(cache_path.read_text())
        assert payload["schema"] == "ema_meeting_outcomes.v1"
        assert payload["collector_version"] == "ema_committee_collector.v1"

    def test_cache_deterministic_ordering(self, tmp_path):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _write_cache

        payload = {
            "schema": "test.v1",
            "events": [{"z": 1}, {"a": 2}],
        }
        cache_path = tmp_path / "test.json"
        _write_cache(cache_path, payload)

        raw = cache_path.read_text()
        reloaded = json.loads(raw)
        # sort_keys=True ensures key ordering
        assert list(reloaded.keys()) == sorted(reloaded.keys())


# ===========================================================================
# Stats / unmatched bounded
# ===========================================================================


class TestStatsBounded:
    """Stats tracking and unmatched bounding."""

    def test_unmatched_bounded(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _parse_meeting_highlights

        # Build HTML with many unmatched items
        items = "".join(f"<li>Unknown Drug {i}</li>" for i in range(200))
        html = f"""
        <html><head><title>CHMP 8 - 11 December 2025</title></head>
        <body>
        <h2>Medicines recommended for approval</h2>
        <ul>{items}</ul>
        </body></html>
        """
        _, unmatched = _parse_meeting_highlights(html, "https://example.com", "CHMP", {})
        assert len(unmatched) <= 100  # bounded by _MAX_UNMATCHED_LOG


# ===========================================================================
# Outcomes precision filter tests
# ===========================================================================


# HTML with mixed decision + non-decision sections
HIGHLIGHTS_MIXED_HTML = """
<html>
<head>
  <title>Meeting highlights from the CHMP 8 - 11 December 2025</title>
  <meta name="dcterms.issued" content="2025-12-11" />
</head>
<body>
<h1>Meeting highlights from the CHMP 8 - 11 December 2025</h1>

<h2>Medicines recommended for approval</h2>
<ul>
  <li>Aficamten — for obstructive hypertrophic cardiomyopathy</li>
</ul>

<h2>Negative opinions</h2>
<ul>
  <li>Keytruda — rejected indication</li>
</ul>

<h2>Withdrawn applications</h2>
<ul>
  <li>Patisiran — withdrawn by MAH</li>
</ul>

<h2>Periodic safety update reports (PSURs)</h2>
<ul>
  <li>Aficamten — PSUR assessment</li>
  <li>Keytruda — PSUR assessment</li>
</ul>

<h2>Administrative and procedural matters</h2>
<ul>
  <li>Aficamten — updated SmPC wording</li>
</ul>

<h2>Other topics of interest</h2>
<ul>
  <li>Keytruda — paediatric investigation plan</li>
</ul>

<h2>Safety referrals</h2>
<ul>
  <li>Keytruda — safety referral started</li>
</ul>
</body>
</html>
"""


class TestOutcomePrecisionFilter:
    """Outcomes precision: only decision-grade sections emit events."""

    def test_decision_sections_produce_events(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _parse_meeting_highlights

        outcomes, _ = _parse_meeting_highlights(
            HIGHLIGHTS_MIXED_HTML, "https://example.com/highlights", "CHMP", PRODUCT_MAP,
        )
        codes = {o["outcome_code"] for o in outcomes}
        # Should contain the three decision codes
        assert "positive_opinion" in codes
        assert "negative_opinion" in codes
        assert "withdrawn" in codes
        assert "referral_started" in codes

    def test_other_sections_produce_zero_events(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _parse_meeting_highlights

        outcomes, _ = _parse_meeting_highlights(
            HIGHLIGHTS_MIXED_HTML, "https://example.com/highlights", "CHMP", PRODUCT_MAP,
        )
        codes = [o["outcome_code"] for o in outcomes]
        assert "other" not in codes

    def test_psur_section_filtered(self):
        """PSUR section items should not produce outcome events."""
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _parse_meeting_highlights

        html = """
        <html><head><title>CHMP 8 - 11 December 2025</title></head>
        <body>
        <h2>Periodic safety update reports (PSURs)</h2>
        <ul>
          <li>Aficamten — PSUR assessment</li>
          <li>Keytruda — PSUR assessment</li>
        </ul>
        </body></html>
        """
        outcomes, _ = _parse_meeting_highlights(html, "https://example.com", "CHMP", PRODUCT_MAP)
        assert len(outcomes) == 0

    def test_administrative_section_filtered(self):
        """Administrative items should not produce outcome events."""
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _parse_meeting_highlights

        html = """
        <html><head><title>CHMP 8 - 11 December 2025</title></head>
        <body>
        <h2>Administrative and procedural matters</h2>
        <ul>
          <li>Aficamten — updated SmPC wording</li>
          <li>Keytruda — variation assessment</li>
        </ul>
        <h2>Other topics of interest</h2>
        <ul>
          <li>Patisiran — paediatric investigation plan</li>
        </ul>
        </body></html>
        """
        outcomes, _ = _parse_meeting_highlights(html, "https://example.com", "CHMP", PRODUCT_MAP)
        assert len(outcomes) == 0

    def test_mixed_page_correct_count(self):
        """Mixed page: only decision-grade items counted."""
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _parse_meeting_highlights

        outcomes, _ = _parse_meeting_highlights(
            HIGHLIGHTS_MIXED_HTML, "https://example.com/highlights", "CHMP", PRODUCT_MAP,
        )
        # positive_opinion: CYTK, negative_opinion: MRK, withdrawn: ALNY,
        # referral_started: MRK (Keytruda)
        assert len(outcomes) == 4

    def test_unmatched_only_from_decision_sections(self):
        """Unmatched list should only contain items from decision-grade sections."""
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _parse_meeting_highlights

        html = """
        <html><head><title>CHMP 8 - 11 December 2025</title></head>
        <body>
        <h2>Medicines recommended for approval</h2>
        <ul>
          <li>SomeUnknownDrug — for rare disease</li>
        </ul>
        <h2>Other topics of interest</h2>
        <ul>
          <li>AnotherUnknownDrug — paediatric plan</li>
        </ul>
        </body></html>
        """
        outcomes, unmatched = _parse_meeting_highlights(html, "https://example.com", "CHMP", {})
        assert len(outcomes) == 0
        # Only "SomeUnknownDrug" from the decision section should be in unmatched
        assert len(unmatched) == 1
        assert "SomeUnknownDrug" in unmatched[0]


# ===========================================================================
# Agenda catalyst relevance tagging
# ===========================================================================


class TestMakeTags:
    """Tests for _make_tags helper."""

    def test_maa_procedure_tag(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _make_tags

        tags = _make_tags("MAA", "CHMP")
        assert "procedure:maa" in tags
        assert "committee:chmp" in tags

    def test_variation_procedure_tag(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _make_tags

        tags = _make_tags("VARIATION", "CHMP")
        assert "procedure:variation" in tags
        assert "committee:chmp" in tags

    def test_prac_committee_tag(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _make_tags

        tags = _make_tags("REFERRAL", "PRAC")
        assert "procedure:referral" in tags
        assert "committee:prac" in tags

    def test_none_procedure(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _make_tags

        tags = _make_tags(None, "CHMP")
        assert len(tags) == 1
        assert tags[0] == "committee:chmp"

    def test_all_procedure_types(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _make_tags

        for proc, expected in [("MAA", "maa"), ("VARIATION", "variation"),
                               ("REFERRAL", "referral"), ("PSUR", "psur"),
                               ("SIGNAL", "signal"), ("OTHER", "other")]:
            tags = _make_tags(proc, "CHMP")
            assert f"procedure:{expected}" in tags


class TestTagsOnEmittedEvents:
    """Tags field present on all emitted events."""

    def test_xlsx_parsed_events_have_tags(self, tmp_path):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import (
            collect_ema_committee_events,
        )

        xlsx_bytes = _build_test_xlsx([
            _make_row("Marketing authorisation application", "Aficamten", "aficamten", "Cytokinetics"),
            _make_row("Variation type II", "Keytruda", "pembrolizumab", "Merck"),
        ])

        def mock_fetch(url):
            if "committees/committee-medicinal-products" in url:
                return LANDING_WITH_EVENT_PAGES_HTML
            if "/en/events/" in url:
                return EVENT_PAGE_HTML
            return "<html></html>"

        with patch(f"{_MOD}._fetch", side_effect=mock_fetch), \
             patch(f"{_MOD}._fetch_xlsx_bytes", return_value=xlsx_bytes), \
             patch(f"{_MOD}._fetch_xml", side_effect=Exception("no RSS in tests")):
            events = collect_ema_committee_events(
                as_of_date=date(2026, 2, 28),
                cache_dir=tmp_path,
                product_ticker_map=PRODUCT_MAP,
                committees=("CHMP",),
            )

        product_events = [e for e in events if e["ticker"]]
        assert len(product_events) == 2
        for ev in product_events:
            assert "tags" in ev
            assert "committee:chmp" in ev["tags"]
            assert any(t.startswith("procedure:") for t in ev["tags"])

        # MAA item should have procedure:maa
        maa_ev = [e for e in product_events if e["ticker"] == "CYTK"][0]
        assert "procedure:maa" in maa_ev["tags"]

        # Variation item should have procedure:variation
        var_ev = [e for e in product_events if e["ticker"] == "MRK"][0]
        assert "procedure:variation" in var_ev["tags"]

    def test_fallback_meeting_events_have_tags(self, tmp_path):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import (
            collect_ema_committee_events,
        )

        def mock_fetch(url):
            return COMMITTEE_LANDING_HTML

        with patch(f"{_MOD}._fetch", side_effect=mock_fetch), \
             patch(f"{_MOD}._fetch_xml", side_effect=Exception("no RSS in tests")):
            events = collect_ema_committee_events(
                as_of_date=date(2026, 1, 31),
                cache_dir=tmp_path,
                product_ticker_map=PRODUCT_MAP,
                committees=("CHMP",),
            )

        for ev in events:
            assert "tags" in ev
            assert "committee:chmp" in ev["tags"]


# ===========================================================================
# Module3 conversion test
# ===========================================================================


class TestModule3Conversion:
    """Test convert_ema_committee_to_v2."""

    def test_agenda_event_conversion(self):
        from module_3_catalyst import convert_ema_committee_to_v2

        event = {
            "schema": "ema_committee_event.v1",
            "id": "EMA_CHMP_AGENDA_2025-12-11_abc1234567",
            "ticker": "CYTK",
            "event_date": "2025-12-11",
            "medicine_name": "Aficamten",
            "title": "CHMP agenda: MAA — Aficamten",
            "confidence": "MED",
            "source": "EMA_CHMP_AGENDA",
            "disclosed_at": "2025-12-01",
        }
        v2 = convert_ema_committee_to_v2(event, date(2025, 12, 1))
        assert v2 is not None
        assert v2.ticker == "CYTK"
        assert v2.event_type.value == "EMA_COMMITTEE_AGENDA"
        assert v2.confidence.value == "MED"
        assert v2.source == "EMA_CHMP_AGENDA"
        assert v2.disclosed_at == "2025-12-01"

    def test_outcome_event_conversion(self):
        from module_3_catalyst import convert_ema_committee_to_v2

        event = {
            "schema": "ema_meeting_outcome.v1",
            "id": "EMA_CHMP_OUTCOME_2025-12-11_positive_abc123",
            "ticker": "CYTK",
            "event_date": "2025-12-11",
            "medicine_name": "Aficamten",
            "title": "CHMP: positive opinion — Aficamten",
            "confidence": "HIGH",
            "source": "EMA_MEETING_HIGHLIGHTS",
            "disclosed_at": "2025-12-11",
        }
        v2 = convert_ema_committee_to_v2(event, date(2025, 12, 11))
        assert v2 is not None
        assert v2.event_type.value == "EMA_COMMITTEE_OUTCOME"
        assert v2.confidence.value == "HIGH"

    def test_missing_ticker_returns_none(self):
        from module_3_catalyst import convert_ema_committee_to_v2

        event = {"schema": "ema_committee_event.v1", "event_date": "2025-12-11"}
        assert convert_ema_committee_to_v2(event, date(2025, 12, 11)) is None


class TestModule3TagBoosting:
    """Test tag-aware severity/field_changed boosting in convert_ema_committee_to_v2."""

    def test_maa_tag_boosts_severity(self):
        from module_3_catalyst import convert_ema_committee_to_v2

        event = {
            "schema": "ema_committee_event.v1",
            "id": "EMA_CHMP_AGENDA_2026-02-26_abc",
            "ticker": "CYTK",
            "event_date": "2026-02-26",
            "medicine_name": "Aficamten",
            "title": "CHMP agenda: MAA — Aficamten",
            "confidence": "MED",
            "source": "EMA_CHMP_AGENDA",
            "disclosed_at": "2026-02-20",
            "tags": ["procedure:maa", "committee:chmp"],
        }
        v2 = convert_ema_committee_to_v2(event, date(2026, 2, 20))
        assert v2 is not None
        assert v2.event_severity.value == "CRITICAL_POSITIVE"
        assert v2.field_changed == "ema_maa"
        assert "procedure:maa" in v2.tags
        assert "committee:chmp" in v2.tags

    def test_variation_tag_keeps_default_severity(self):
        from module_3_catalyst import convert_ema_committee_to_v2

        event = {
            "schema": "ema_committee_event.v1",
            "id": "EMA_CHMP_AGENDA_2026-02-26_def",
            "ticker": "MRK",
            "event_date": "2026-02-26",
            "medicine_name": "Keytruda",
            "title": "CHMP agenda: VARIATION — Keytruda",
            "confidence": "MED",
            "source": "EMA_CHMP_AGENDA",
            "disclosed_at": "2026-02-20",
            "tags": ["procedure:variation", "committee:chmp"],
        }
        v2 = convert_ema_committee_to_v2(event, date(2026, 2, 20))
        assert v2 is not None
        assert v2.event_severity.value == "POSITIVE"
        assert v2.field_changed == "ema_committee"

    def test_no_tags_keeps_default(self):
        from module_3_catalyst import convert_ema_committee_to_v2

        event = {
            "schema": "ema_committee_event.v1",
            "id": "EMA_CHMP_AGENDA_2026-02-26_ghi",
            "ticker": "CYTK",
            "event_date": "2026-02-26",
            "medicine_name": "Aficamten",
            "title": "CHMP agenda: MAA — Aficamten",
            "confidence": "MED",
            "source": "EMA_CHMP_AGENDA",
            "disclosed_at": "2026-02-20",
        }
        v2 = convert_ema_committee_to_v2(event, date(2026, 2, 20))
        assert v2 is not None
        assert v2.event_severity.value == "POSITIVE"
        assert v2.field_changed == "ema_committee"
        assert v2.tags == ()

    def test_outcome_with_maa_tag_not_boosted(self):
        """Outcomes already have CRITICAL_POSITIVE; MAA tag should not change behavior."""
        from module_3_catalyst import convert_ema_committee_to_v2

        event = {
            "schema": "ema_meeting_outcome.v1",
            "id": "EMA_CHMP_OUTCOME_2026-02-26_abc",
            "ticker": "CYTK",
            "event_date": "2026-02-26",
            "medicine_name": "Aficamten",
            "title": "CHMP: positive opinion — Aficamten",
            "confidence": "HIGH",
            "source": "EMA_MEETING_HIGHLIGHTS",
            "disclosed_at": "2026-02-26",
            "tags": ["procedure:maa", "committee:chmp"],
        }
        v2 = convert_ema_committee_to_v2(event, date(2026, 2, 26))
        assert v2 is not None
        assert v2.event_severity.value == "CRITICAL_POSITIVE"
        assert v2.field_changed == "ema_committee"  # outcomes keep generic field_changed

    def test_tags_pass_through(self):
        from module_3_catalyst import convert_ema_committee_to_v2

        event = {
            "schema": "ema_committee_event.v1",
            "id": "EMA_CHMP_AGENDA_2026-02-26_jkl",
            "ticker": "MRK",
            "event_date": "2026-02-26",
            "medicine_name": "Keytruda",
            "confidence": "MED",
            "source": "EMA_CHMP_AGENDA",
            "disclosed_at": "2026-02-20",
            "tags": ["procedure:variation", "committee:chmp"],
        }
        v2 = convert_ema_committee_to_v2(event, date(2026, 2, 20))
        assert v2.tags == ("procedure:variation", "committee:chmp")


# ===========================================================================
# Event ledger loader
# ===========================================================================


class TestEventLedgerEmaLoader:
    """Test _load_ema_events from event_ledger.py."""

    def test_loads_from_cache(self, tmp_path):
        from event_ledger import _load_ema_events

        payload = {
            "schema": "ema_committee_events.v1",
            "events": [
                {
                    "id": "EMA_CHMP_AGENDA_2026-01-30_abc",
                    "ticker": "CYTK",
                    "event_date": "2026-01-30",
                    "disclosed_at": "2026-01-20",
                    "title": "CHMP agenda: MAA — Aficamten",
                    "confidence": "MED",
                },
            ],
        }
        cache_path = tmp_path / "ema_committee_events_2026-01-30.json"
        cache_path.write_text(json.dumps(payload))

        entries = _load_ema_events(date(2026, 1, 30), tmp_path)
        assert len(entries) == 1
        assert entries[0].ticker == "CYTK"
        assert entries[0].source == "EMA_AGENDA"
        assert entries[0].confidence == "MED"

    def test_loads_both_caches(self, tmp_path):
        from event_ledger import _load_ema_events

        events_payload = {
            "schema": "ema_committee_events.v1",
            "events": [{"id": "a1", "ticker": "CYTK", "event_date": "2026-01-30",
                         "disclosed_at": "2026-01-20", "title": "A", "confidence": "MED"}],
        }
        outcomes_payload = {
            "schema": "ema_meeting_outcomes.v1",
            "events": [{"id": "o1", "ticker": "MRK", "event_date": "2025-12-11",
                         "disclosed_at": "2025-12-11", "title": "O", "confidence": "HIGH"}],
        }
        (tmp_path / "ema_committee_events_2026-01-30.json").write_text(json.dumps(events_payload))
        (tmp_path / "ema_meeting_outcomes_2026-01-30.json").write_text(json.dumps(outcomes_payload))

        entries = _load_ema_events(date(2026, 1, 30), tmp_path)
        assert len(entries) == 2
        tickers = {e.ticker for e in entries}
        assert tickers == {"CYTK", "MRK"}

    def test_empty_when_no_cache(self, tmp_path):
        from event_ledger import _load_ema_events

        entries = _load_ema_events(date(2026, 1, 30), tmp_path)
        assert entries == []

    def test_tags_propagated(self, tmp_path):
        """Tags from collector events are propagated into LedgerEntry.tags."""
        from event_ledger import _load_ema_events

        payload = {
            "schema": "ema_committee_events.v1",
            "events": [
                {
                    "id": "EMA_CHMP_AGENDA_2026-01-30_abc",
                    "ticker": "CYTK",
                    "event_date": "2026-01-30",
                    "disclosed_at": "2026-01-20",
                    "title": "CHMP agenda: MAA — Aficamten",
                    "confidence": "MED",
                    "tags": ["procedure:maa", "committee:chmp"],
                },
            ],
        }
        cache_path = tmp_path / "ema_committee_events_2026-01-30.json"
        cache_path.write_text(json.dumps(payload))

        entries = _load_ema_events(date(2026, 1, 30), tmp_path)
        assert len(entries) == 1
        assert entries[0].tags == ("procedure:maa", "committee:chmp")

    def test_tags_default_empty(self, tmp_path):
        """Events without tags field get empty tuple."""
        from event_ledger import _load_ema_events

        payload = {
            "schema": "ema_committee_events.v1",
            "events": [
                {
                    "id": "EMA_CHMP_AGENDA_2026-01-30_abc",
                    "ticker": "CYTK",
                    "event_date": "2026-01-30",
                    "disclosed_at": "2026-01-20",
                    "title": "CHMP agenda item",
                    "confidence": "MED",
                },
            ],
        }
        cache_path = tmp_path / "ema_committee_events_2026-01-30.json"
        cache_path.write_text(json.dumps(payload))

        entries = _load_ema_events(date(2026, 1, 30), tmp_path)
        assert len(entries) == 1
        assert entries[0].tags == ()


# ===========================================================================
# run_screen.py cache health — EMA counts wired through
# ===========================================================================


class TestRunScreenCacheHealthEma:
    """Verify run_screen passes EMA counts to compute_cache_health."""

    def test_cache_health_includes_ema_agenda(self):
        """compute_cache_health includes ema_agenda section when count provided."""
        from cache_health import compute_cache_health

        result = compute_cache_health(
            sec8k_count=100,
            ctgov_count=500,
            ema_agenda_count=15,
            ema_outcomes_count=8,
        )
        assert "ema_agenda" in result
        assert result["ema_agenda"]["count"] == 15
        assert result["ema_agenda"]["status"] == "ok"
        assert "ema_outcomes" in result
        assert result["ema_outcomes"]["count"] == 8
        assert result["ema_outcomes"]["status"] == "ok"

    def test_cache_health_ema_ratio_warning(self):
        """EMA ratio outside band produces warning status."""
        from cache_health import compute_cache_health

        result = compute_cache_health(
            sec8k_count=100,
            ctgov_count=500,
            ema_agenda_count=5,
            prior_ema_agenda_count=50,  # ratio = 0.10 < 0.30
            ema_outcomes_count=8,
        )
        assert result["ema_agenda"]["status"] == "warning"
        assert result["overall_status"] == "warning"
        assert result["degraded_run"] is True

    def test_cache_health_ema_thresholds_in_output(self):
        """EMA thresholds appear in the thresholds dict."""
        from cache_health import compute_cache_health

        result = compute_cache_health(
            sec8k_count=100,
            ctgov_count=500,
            ema_agenda_count=10,
            ema_outcomes_count=5,
        )
        assert "ema_agenda_bad_ratio_low" in result["thresholds"]
        assert "ema_agenda_bad_ratio_high" in result["thresholds"]
        assert "ema_outcomes_bad_ratio_low" in result["thresholds"]
        assert "ema_outcomes_bad_ratio_high" in result["thresholds"]

    def test_cache_health_ema_ratios_in_comparisons(self):
        """EMA ratios appear in comparisons dict when priors provided."""
        from cache_health import compute_cache_health

        result = compute_cache_health(
            sec8k_count=100,
            ctgov_count=500,
            ema_agenda_count=15,
            prior_ema_agenda_count=10,  # ratio 1.5
            ema_outcomes_count=8,
            prior_ema_outcomes_count=8,  # ratio 1.0
        )
        assert result["comparisons"]["ema_agenda_ratio_vs_prior"] == 1.5
        assert result["comparisons"]["ema_outcomes_ratio_vs_prior"] == 1.0

    def test_cache_health_omits_ema_when_none(self):
        """No ema_agenda/ema_outcomes in result when counts are None."""
        from cache_health import compute_cache_health

        result = compute_cache_health(
            sec8k_count=100,
            ctgov_count=500,
        )
        assert "ema_agenda" not in result
        assert "ema_outcomes" not in result


# ===========================================================================
# End-to-end: EMA events flow through module_3 with tags preserved
# ===========================================================================


class TestEmaEndToEndTagFlow:
    """Verify tags survive the full collector → module_3 → CatalystEventV2 path."""

    def test_tags_preserved_through_convert(self):
        """Tags from collector event dict survive convert_ema_committee_to_v2."""
        from module_3_catalyst import convert_ema_committee_to_v2

        event = {
            "schema": "ema_committee_event.v1",
            "ticker": "CYTK",
            "event_date": "2026-03-15",
            "disclosed_at": "2026-02-20",
            "title": "CHMP agenda: MAA — Aficamten",
            "medicine_name": "Aficamten",
            "procedure_type": "MAA",
            "source": "EMA_CHMP_AGENDA",
            "confidence": "MED",
            "tags": ["procedure:maa", "committee:chmp"],
        }
        v2 = convert_ema_committee_to_v2(event, date(2026, 2, 28))
        assert v2 is not None
        assert v2.tags == ("procedure:maa", "committee:chmp")
        # MAA tag should boost severity
        from module_3_schema import EventSeverity
        assert v2.event_severity == EventSeverity.CRITICAL_POSITIVE
        assert v2.field_changed == "ema_maa"

    def test_outcome_tags_preserved_no_boost(self):
        """Outcome events preserve tags but don't get MAA boosting."""
        from module_3_catalyst import convert_ema_committee_to_v2
        from module_3_schema import EventSeverity

        event = {
            "schema": "ema_meeting_outcome.v1",
            "ticker": "MRK",
            "event_date": "2025-12-11",
            "disclosed_at": "2025-12-11",
            "title": "CHMP: positive_opinion — Keytruda",
            "outcome_code": "positive_opinion",
            "source": "EMA_MEETING_HIGHLIGHTS",
            "confidence": "HIGH",
            "tags": ["committee:chmp"],
        }
        v2 = convert_ema_committee_to_v2(event, date(2026, 2, 28))
        assert v2 is not None
        assert v2.tags == ("committee:chmp",)
        # Outcomes keep default severity, not boosted
        assert v2.event_severity == EventSeverity.CRITICAL_POSITIVE
        assert v2.field_changed == "ema_committee"

    def test_no_tags_field_defaults_empty(self):
        """Events without 'tags' key get empty tuple in CatalystEventV2."""
        from module_3_catalyst import convert_ema_committee_to_v2

        event = {
            "schema": "ema_committee_event.v1",
            "ticker": "CYTK",
            "event_date": "2026-03-15",
            "disclosed_at": "2026-02-20",
            "title": "CHMP agenda item",
            "source": "EMA_CHMP_AGENDA",
            "confidence": "MED",
        }
        v2 = convert_ema_committee_to_v2(event, date(2026, 2, 28))
        assert v2 is not None
        assert v2.tags == ()

    def test_source_key_for_source_mix(self):
        """EMA v2 events use the collector's source key (not generic 'EMA_COMMITTEE')."""
        from module_3_catalyst import convert_ema_committee_to_v2

        event = {
            "schema": "ema_committee_event.v1",
            "ticker": "CYTK",
            "event_date": "2026-03-15",
            "disclosed_at": "2026-02-20",
            "title": "CHMP agenda: MAA — Aficamten",
            "source": "EMA_CHMP_AGENDA",
            "confidence": "MED",
            "tags": ["procedure:maa", "committee:chmp"],
        }
        v2 = convert_ema_committee_to_v2(event, date(2026, 2, 28))
        # source passed through from collector → used as by_source key
        assert v2.source == "EMA_CHMP_AGENDA"

    def test_ema_source_mix_aggregation(self):
        """Verify source_mix keys that run_screen uses to extract EMA counts."""
        # Simulate what module_3 does: count events by v2.source
        sources = [
            "EMA_CHMP_AGENDA", "EMA_CHMP_AGENDA", "EMA_PRAC_AGENDA",
            "EMA_MEETING_HIGHLIGHTS",
        ]
        by_source = {}
        for s in sources:
            by_source[s] = by_source.get(s, 0) + 1

        # run_screen aggregation logic
        ema_agenda_count = sum(
            v for k, v in by_source.items()
            if k.startswith("EMA_") and k.endswith("_AGENDA")
        )
        ema_outcomes_count = by_source.get("EMA_MEETING_HIGHLIGHTS", 0)

        assert ema_agenda_count == 3  # 2 CHMP + 1 PRAC
        assert ema_outcomes_count == 1


# ===========================================================================
# XLSX builder fixture + Annex XLSX parsing tests
# ===========================================================================


def _build_test_xlsx(
    rows: list[list[str]],
    *,
    header_row_num: int = 24,
    include_table: bool = True,
    headers: list[str] | None = None,
) -> bytes:
    """Build a minimal valid XLSX in-memory using zipfile + XML.

    Args:
        rows: List of rows, each a list of 15 string cell values.
              Columns map to A-O (indices 0-14).
        header_row_num: 1-based row number for the header row.
        include_table: Whether to include xl/tables/table1.xml.
        headers: Header labels. Defaults to EMA Annex layout.

    Returns:
        Raw bytes of a valid XLSX file.
    """
    if headers is None:
        headers = [
            "No", "Procedure No", "EMEA No", "Rapporteur/Co-rapporteur",
            "Process type",  # E = col 4
            "Invented name",  # F = col 5
            "Active substance",  # G = col 6
            "Customer",  # H = col 7
            "Scope", "Background", "Action", "Timetable",
            "Outcome", "Notes", "Status",
        ]

    # Build shared strings
    all_strings: list[str] = []
    string_index: dict[str, int] = {}

    def _add_string(s: str) -> int:
        if s not in string_index:
            string_index[s] = len(all_strings)
            all_strings.append(s)
        return string_index[s]

    # Register header strings
    for h in headers:
        _add_string(h)

    # Register data strings
    for row in rows:
        for cell in row:
            if cell:
                _add_string(cell)

    # Build sharedStrings.xml
    ss_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    ss_root = ET.Element("sst", xmlns=ss_ns, count=str(len(all_strings)),
                         uniqueCount=str(len(all_strings)))
    for s in all_strings:
        si = ET.SubElement(ss_root, "si")
        t = ET.SubElement(si, "t")
        t.text = s
    ss_xml = ET.tostring(ss_root, encoding="unicode", xml_declaration=True)

    # Build sheet1.xml
    ws_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"

    def _col_letter(idx: int) -> str:
        result = ""
        idx += 1
        while idx > 0:
            idx, rem = divmod(idx - 1, 26)
            result = chr(65 + rem) + result
        return result

    ws_root = ET.Element("worksheet", xmlns=ws_ns)
    sheet_data = ET.SubElement(ws_root, "sheetData")

    # Header row
    h_row = ET.SubElement(sheet_data, "row", r=str(header_row_num))
    for ci, h in enumerate(headers):
        ref = f"{_col_letter(ci)}{header_row_num}"
        c = ET.SubElement(h_row, "c", r=ref, t="s")
        v = ET.SubElement(c, "v")
        v.text = str(_add_string(h))

    # Data rows
    data_start = header_row_num + 1
    for ri, row in enumerate(rows):
        row_num = data_start + ri
        r_el = ET.SubElement(sheet_data, "row", r=str(row_num))
        for ci, cell in enumerate(row):
            if not cell:
                continue
            ref = f"{_col_letter(ci)}{row_num}"
            c = ET.SubElement(r_el, "c", r=ref, t="s")
            v = ET.SubElement(c, "v")
            v.text = str(_add_string(cell))

    ws_xml = ET.tostring(ws_root, encoding="unicode", xml_declaration=True)

    # Build table1.xml (optional)
    last_col = _col_letter(len(headers) - 1)
    last_row = data_start + len(rows) - 1 if rows else data_start
    tbl_ref = f"A{header_row_num}:{last_col}{last_row}"
    tbl_xml = None
    if include_table:
        tbl_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
        tbl_root = ET.Element("table", xmlns=tbl_ns, ref=tbl_ref, id="1", name="Table1")
        tbl_xml = ET.tostring(tbl_root, encoding="unicode", xml_declaration=True)

    # Assemble ZIP
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>
</Types>""")
        zf.writestr("_rels/.rels", """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>""")
        zf.writestr("xl/workbook.xml", """<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"
    xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/></sheets>
</workbook>""")
        zf.writestr("xl/_rels/workbook.xml.rels", """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>
</Relationships>""")
        zf.writestr("xl/sharedStrings.xml", ss_xml)
        zf.writestr("xl/worksheets/sheet1.xml", ws_xml)
        if tbl_xml:
            zf.writestr("xl/tables/table1.xml", tbl_xml)

    return buf.getvalue()


# 15-element row helper (A-O columns)
def _make_row(
    process_type: str = "",
    invented_name: str = "",
    active_substance: str = "",
    customer: str = "",
) -> list[str]:
    """Build a 15-element row with values in the correct column positions."""
    row = [""] * 15
    row[4] = process_type     # E
    row[5] = invented_name    # F
    row[6] = active_substance  # G
    row[7] = customer         # H
    return row


# Event page HTML fixture — mimics a real EMA event page with documents
EVENT_PAGE_HTML = """
<html>
<head><title>CHMP meeting 23 - 26 February 2026</title></head>
<body>
<h1>Committee for Medicinal Products for Human Use (CHMP) meeting 23 - 26 February 2026</h1>
<div class="field--items">
  <div class="bcl-file">
    <span>Agenda of CHMP meeting 23 - 26 February 2026</span>
    <span>First published:20/02/2026</span>
    <div class="file-language-links">
      <a href="/en/documents/agenda/agenda-chmp-meeting-23-26-february-2026_en.pdf">View</a>
    </div>
  </div>
  <div class="bcl-file">
    <span>Annex to agenda of CHMP meeting 23 - 26 February 2026</span>
    <span>First published:20/02/2026</span>
    <div class="file-language-links">
      <a href="/en/documents/annex/annex-to-agenda-chmp-meeting-23-26-february-2026_en.xlsx">View</a>
    </div>
  </div>
</div>
</body>
</html>
"""

# Committee landing page with event page links
LANDING_WITH_EVENT_PAGES_HTML = """
<html>
<head><title>CHMP | European Medicines Agency</title></head>
<body>
<h1>Committee for Medicinal Products for Human Use (CHMP)</h1>
<div class="field--items">
  <h3>Upcoming meetings</h3>
  <ul>
    <li><a href="/en/events/committee-medicinal-products-human-use-chmp-23-26-february-2026">
      CHMP meeting 23 - 26 February 2026
    </a></li>
    <li><a href="/en/events/committee-medicinal-products-human-use-chmp-24-27-march-2026">
      CHMP meeting 24 - 27 March 2026
    </a></li>
  </ul>
</div>
</body>
</html>
"""


class TestParseAnnexXlsx:
    """Tests for _parse_annex_xlsx."""

    def test_basic_extraction(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _parse_annex_xlsx

        rows = [
            _make_row("Variation type II", "Keytruda", "pembrolizumab", "Merck"),
            _make_row("MAA", "Aficamten", "aficamten", "Cytokinetics"),
        ]
        xlsx = _build_test_xlsx(rows)
        items = _parse_annex_xlsx(xlsx)
        assert len(items) == 2
        assert items[0]["medicine_name"] == "Keytruda"
        assert items[0]["active_substance"] == "pembrolizumab"
        assert items[0]["process_type"] == "Variation type II"
        assert items[0]["sponsor"] == "Merck"
        assert items[1]["medicine_name"] == "Aficamten"

    def test_empty_rows_skipped(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _parse_annex_xlsx

        rows = [
            _make_row("Variation type II", "Keytruda", "pembrolizumab", "Merck"),
            _make_row("", "", "", ""),  # empty invented_name → skipped
            _make_row("MAA", "Aficamten", "aficamten", "Cytokinetics"),
        ]
        xlsx = _build_test_xlsx(rows)
        items = _parse_annex_xlsx(xlsx)
        assert len(items) == 2

    def test_multiline_medicine_name(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _parse_annex_xlsx

        rows = [
            _make_row("Variation type II", "Keytruda\nKeytiva", "pembrolizumab\npembrolizumab+x", "Merck"),
        ]
        xlsx = _build_test_xlsx(rows)
        items = _parse_annex_xlsx(xlsx)
        assert len(items) == 1
        assert items[0]["medicine_name"] == "Keytruda"
        assert items[0]["active_substance"] == "pembrolizumab"

    def test_fallback_header_detection_without_table(self):
        """When table1.xml is missing, header row is detected by scanning for 'Invented name'."""
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _parse_annex_xlsx

        rows = [
            _make_row("Variation type II", "Keytruda", "pembrolizumab", "Merck"),
        ]
        xlsx = _build_test_xlsx(rows, include_table=False)
        items = _parse_annex_xlsx(xlsx)
        assert len(items) == 1
        assert items[0]["medicine_name"] == "Keytruda"

    def test_invalid_zip_returns_empty(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _parse_annex_xlsx

        items = _parse_annex_xlsx(b"not a zip file")
        assert items == []

    def test_many_rows(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _parse_annex_xlsx

        rows = [_make_row("Variation", f"Drug{i}", f"substance{i}", f"Co{i}") for i in range(150)]
        xlsx = _build_test_xlsx(rows)
        items = _parse_annex_xlsx(xlsx)
        assert len(items) == 150


class TestDiscoverMeetingEventPages:
    """Tests for _discover_meeting_event_pages."""

    def test_finds_chmp_events(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _discover_meeting_event_pages

        pages = _discover_meeting_event_pages(
            LANDING_WITH_EVENT_PAGES_HTML, "https://www.ema.europa.eu", "CHMP",
        )
        assert len(pages) == 2
        assert pages[0]["meeting_start"] == "2026-02-23"
        assert pages[0]["meeting_end"] == "2026-02-26"
        assert pages[0]["committee"] == "CHMP"
        assert "chmp-23-26-february-2026" in pages[0]["url"]

    def test_skips_non_committee_events(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _discover_meeting_event_pages

        html = """<html><body>
        <a href="/en/events/committee-medicinal-products-human-use-chmp-23-26-february-2026">
          CHMP meeting 23 - 26 February 2026
        </a>
        <a href="/en/events/some-other-event-2026">Other event</a>
        </body></html>"""
        pages = _discover_meeting_event_pages(html, "https://www.ema.europa.eu", "CHMP")
        assert len(pages) == 1

    def test_prac_events(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _discover_meeting_event_pages

        html = """<html><body>
        <a href="/en/events/pharmacovigilance-risk-assessment-committee-prac-3-6-february-2026">
          PRAC meeting 3 - 6 February 2026
        </a>
        </body></html>"""
        pages = _discover_meeting_event_pages(html, "https://www.ema.europa.eu", "PRAC")
        assert len(pages) == 1
        assert pages[0]["committee"] == "PRAC"


class TestDiscoverEventPageDocuments:
    """Tests for _discover_event_page_documents."""

    def test_finds_agenda_and_annex(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _discover_event_page_documents

        docs = _discover_event_page_documents(EVENT_PAGE_HTML, "https://www.ema.europa.eu")
        assert len(docs["agenda"]) == 1
        assert len(docs["annex"]) == 1
        assert docs["agenda"][0]["url"].endswith(".pdf")
        assert docs["annex"][0]["url"].endswith(".xlsx")

    def test_extracts_first_published(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _discover_event_page_documents

        docs = _discover_event_page_documents(EVENT_PAGE_HTML, "https://www.ema.europa.eu")
        assert docs["annex"][0]["first_published"] == "2026-02-20"
        assert docs["agenda"][0]["first_published"] == "2026-02-20"

    def test_empty_page(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _discover_event_page_documents

        docs = _discover_event_page_documents("<html><body></body></html>", "https://www.ema.europa.eu")
        assert docs["agenda"] == []
        assert docs["annex"] == []


class TestCollectEmaWithEventPages:
    """Integration tests for collect_ema_committee_events with event page path."""

    def _build_xlsx_for_test(self):
        return _build_test_xlsx([
            _make_row("Variation type II", "Keytruda", "pembrolizumab", "Merck"),
            _make_row("MAA", "Aficamten", "aficamten", "Cytokinetics"),
            _make_row("Variation type II", "Unmapped Drug", "unknown_sub", "UnknownCo"),
        ])

    def test_product_level_events_from_annex(self, tmp_path):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import (
            collect_ema_committee_events,
        )

        xlsx_bytes = self._build_xlsx_for_test()

        def mock_fetch(url):
            if "committees/committee-medicinal-products" in url:
                return LANDING_WITH_EVENT_PAGES_HTML
            if "/en/events/" in url:
                return EVENT_PAGE_HTML
            return "<html></html>"

        def mock_fetch_xlsx(url):
            return xlsx_bytes

        with patch(f"{_MOD}._fetch", side_effect=mock_fetch), \
             patch(f"{_MOD}._fetch_xlsx_bytes", side_effect=mock_fetch_xlsx), \
             patch(f"{_MOD}._fetch_xml", side_effect=Exception("no RSS in tests")):
            events = collect_ema_committee_events(
                as_of_date=date(2026, 2, 28),
                cache_dir=tmp_path,
                product_ticker_map=PRODUCT_MAP,
                committees=("CHMP",),
            )

        # Should have product-level events for matched tickers
        tickers = [e["ticker"] for e in events if e["ticker"]]
        assert "MRK" in tickers
        assert "CYTK" in tickers
        # Should have item_type=agenda_item (not meeting)
        assert all(e["item_type"] == "agenda_item" for e in events if e["ticker"])

    def test_stats_correct(self, tmp_path):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import (
            collect_ema_committee_events,
        )

        xlsx_bytes = self._build_xlsx_for_test()

        def mock_fetch(url):
            if "committees/committee-medicinal-products" in url:
                return LANDING_WITH_EVENT_PAGES_HTML
            if "/en/events/" in url:
                return EVENT_PAGE_HTML
            return "<html></html>"

        def mock_fetch_xlsx(url):
            return xlsx_bytes

        with patch(f"{_MOD}._fetch", side_effect=mock_fetch), \
             patch(f"{_MOD}._fetch_xlsx_bytes", side_effect=mock_fetch_xlsx), \
             patch(f"{_MOD}._fetch_xml", side_effect=Exception("no RSS in tests")):
            collect_ema_committee_events(
                as_of_date=date(2026, 2, 28),
                cache_dir=tmp_path,
                product_ticker_map=PRODUCT_MAP,
                committees=("CHMP",),
            )

        cache_path = tmp_path / "ema_committee_events_2026-02-28.json"
        payload = json.loads(cache_path.read_text())
        stats = payload["stats"]
        assert stats["items_parsed"] == 3
        assert stats["matched"] == 2
        assert stats["unmatched"] == 1
        assert stats["meeting_pages_found"] >= 1

    def test_deterministic_cache(self, tmp_path):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import (
            collect_ema_committee_events,
        )

        xlsx_bytes = self._build_xlsx_for_test()

        def mock_fetch(url):
            if "committees/committee-medicinal-products" in url:
                return LANDING_WITH_EVENT_PAGES_HTML
            if "/en/events/" in url:
                return EVENT_PAGE_HTML
            return "<html></html>"

        def mock_fetch_xlsx(url):
            return xlsx_bytes

        with patch(f"{_MOD}._fetch", side_effect=mock_fetch), \
             patch(f"{_MOD}._fetch_xlsx_bytes", side_effect=mock_fetch_xlsx), \
             patch(f"{_MOD}._fetch_xml", side_effect=Exception("no RSS in tests")):
            result1 = collect_ema_committee_events(
                as_of_date=date(2026, 2, 28),
                cache_dir=tmp_path,
                product_ticker_map=PRODUCT_MAP,
                committees=("CHMP",),
            )

        # Second call hits cache
        result2 = collect_ema_committee_events(
            as_of_date=date(2026, 2, 28),
            cache_dir=tmp_path,
            product_ticker_map=PRODUCT_MAP,
            committees=("CHMP",),
        )
        assert [e["id"] for e in result1] == [e["id"] for e in result2]

    def test_fallback_to_landing_page_docs(self, tmp_path):
        """When no event pages found, falls back to landing page doc discovery."""
        from wake_robin_data_pipeline.collectors.ema_committee_collector import (
            collect_ema_committee_events,
        )

        def mock_fetch(url):
            # Return landing page with docs but NO event page links
            return COMMITTEE_LANDING_HTML

        with patch(f"{_MOD}._fetch", side_effect=mock_fetch), \
             patch(f"{_MOD}._fetch_xml", side_effect=Exception("no RSS in tests")):
            events = collect_ema_committee_events(
                as_of_date=date(2026, 1, 31),
                cache_dir=tmp_path,
                product_ticker_map=PRODUCT_MAP,
                committees=("CHMP",),
            )

        # Should fall back to meeting-level events
        meeting_events = [e for e in events if e["item_type"] == "meeting"]
        assert len(meeting_events) > 0

    def test_xlsx_parse_error_continues(self, tmp_path):
        """XLSX parse error is logged and collector continues."""
        from wake_robin_data_pipeline.collectors.ema_committee_collector import (
            collect_ema_committee_events,
        )

        def mock_fetch(url):
            if "committees/committee-medicinal-products" in url:
                return LANDING_WITH_EVENT_PAGES_HTML
            if "/en/events/" in url:
                return EVENT_PAGE_HTML
            return "<html></html>"

        def mock_fetch_xlsx(url):
            return b"not a valid xlsx"

        with patch(f"{_MOD}._fetch", side_effect=mock_fetch), \
             patch(f"{_MOD}._fetch_xlsx_bytes", side_effect=mock_fetch_xlsx), \
             patch(f"{_MOD}._fetch_xml", side_effect=Exception("no RSS in tests")):
            events = collect_ema_committee_events(
                as_of_date=date(2026, 2, 28),
                cache_dir=tmp_path,
                product_ticker_map=PRODUCT_MAP,
                committees=("CHMP",),
            )

        # Should still produce events (meeting-level placeholders)
        assert len(events) >= 1


# ===========================================================================
# RSS feed fixtures
# ===========================================================================

RSS_AGENDAS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>EMA Agendas and minutes</title>
<item>
  <title>Agenda of the CHMP meeting 23-26 February 2026</title>
  <link>https://www.ema.europa.eu/en/documents/agenda/agenda-chmp-meeting-23-26-february-2026_en.pdf</link>
  <pubDate>Thu, 20 Feb 2026 00:00:00 +0100</pubDate>
</item>
<item>
  <title>Agenda of the PRAC meeting 3-6 February 2026</title>
  <link>https://www.ema.europa.eu/en/documents/agenda/agenda-prac-meeting-3-6-february-2026_en.pdf</link>
  <pubDate>Fri, 31 Jan 2026 00:00:00 +0100</pubDate>
</item>
<item>
  <title>Minutes of the CAT meeting January 2026</title>
  <link>https://www.ema.europa.eu/en/documents/minutes/cat-january-2026.pdf</link>
  <pubDate>Mon, 05 Jan 2026 00:00:00 +0100</pubDate>
</item>
</channel></rss>"""

RSS_EVENTS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>EMA Events</title>
<item>
  <title>CHMP meeting 23-26 February 2026</title>
  <link>https://www.ema.europa.eu/en/events/committee-medicinal-products-human-use-chmp-23-26-february-2026</link>
  <pubDate>Fri, 01 Jan 2026 00:00:00 +0100</pubDate>
</item>
<item>
  <title>CHMP meeting 24-27 March 2026</title>
  <link>https://www.ema.europa.eu/en/events/committee-medicinal-products-human-use-chmp-24-27-march-2026</link>
  <pubDate>Fri, 01 Jan 2026 00:00:00 +0100</pubDate>
</item>
<item>
  <title>Scientific workshop on AI in drug development</title>
  <link>https://www.ema.europa.eu/en/events/ai-workshop-2026</link>
  <pubDate>Mon, 01 Dec 2025 00:00:00 +0100</pubDate>
</item>
</channel></rss>"""


# ===========================================================================
# RSS helper unit tests
# ===========================================================================


class TestFetchXml:
    """Tests for _fetch_xml."""

    def test_successful_fetch(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _fetch_xml

        with patch("wake_robin_data_pipeline.collectors.ema_committee_collector.requests.get") as mock_get, \
             patch("wake_robin_data_pipeline.collectors.ema_committee_collector.time.sleep"):
            mock_get.return_value.text = RSS_AGENDAS_XML
            mock_get.return_value.raise_for_status = lambda: None
            result = _fetch_xml("https://example.com/feed.xml")

        assert "CHMP" in result
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args
        assert "application/rss+xml" in call_kwargs.kwargs.get("headers", {}).get("Accept", "")

    def test_network_error_raises(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _fetch_xml

        with patch("wake_robin_data_pipeline.collectors.ema_committee_collector.requests.get",
                    side_effect=Exception("Connection failed")), \
             patch("wake_robin_data_pipeline.collectors.ema_committee_collector.time.sleep"):
            with pytest.raises(Exception, match="Connection failed"):
                _fetch_xml("https://example.com/feed.xml")


class TestParseRssAgendaItems:
    """Tests for _parse_rss_agenda_items."""

    def test_finds_chmp_items(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _parse_rss_agenda_items

        items = _parse_rss_agenda_items(RSS_AGENDAS_XML, "CHMP")
        assert len(items) == 1
        assert items[0]["meeting_start"] == "2026-02-23"
        assert items[0]["meeting_end"] == "2026-02-26"
        assert "agenda-chmp" in items[0]["agenda_url"]
        assert items[0]["pub_date"] == "Thu, 20 Feb 2026 00:00:00 +0100"

    def test_finds_prac_items(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _parse_rss_agenda_items

        items = _parse_rss_agenda_items(RSS_AGENDAS_XML, "PRAC")
        assert len(items) == 1
        assert items[0]["meeting_start"] == "2026-02-03"
        assert items[0]["meeting_end"] == "2026-02-06"

    def test_skips_non_committee(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _parse_rss_agenda_items

        items = _parse_rss_agenda_items(RSS_AGENDAS_XML, "CHMP")
        # CAT minutes and PRAC agenda should not appear in CHMP results
        titles = [i["title"] for i in items]
        assert not any("CAT" in t for t in titles)
        assert not any("PRAC" in t for t in titles)

    def test_empty_feed(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _parse_rss_agenda_items

        xml = '<?xml version="1.0"?><rss><channel></channel></rss>'
        items = _parse_rss_agenda_items(xml, "CHMP")
        assert items == []

    def test_invalid_xml_returns_empty(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _parse_rss_agenda_items

        items = _parse_rss_agenda_items("not valid xml at all", "CHMP")
        assert items == []


class TestConstructAnnexUrl:
    """Tests for _construct_annex_url."""

    def test_happy_path(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _construct_annex_url

        url = "https://www.ema.europa.eu/en/documents/agenda/agenda-chmp-meeting-23-26-february-2026_en.pdf"
        result = _construct_annex_url(url)
        assert result == "https://www.ema.europa.eu/en/documents/agenda/annex-agenda-chmp-meeting-23-26-february-2026_en.xlsx"

    def test_already_annex_url_returns_none(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _construct_annex_url

        url = "https://www.ema.europa.eu/en/documents/annex/annex-agenda-chmp-meeting-23-26-february-2026_en.xlsx"
        result = _construct_annex_url(url)
        # no "/agenda-" in URL (has "/annex-agenda-" but no bare "/agenda-")
        # Actually "/annex-agenda-" contains "/agenda-" via replace... let me check
        # The URL has "/annex/" not "/agenda/" so the replace changes /agenda- prefix part
        # But this URL does contain "/agenda-" (in "annex-agenda-"). Let's test:
        # It DOES contain "/agenda-" so replace will fire. But .pdf check fails.
        assert result is None  # not .pdf

    def test_non_matching_url_returns_none(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _construct_annex_url

        assert _construct_annex_url("https://example.com/report.docx") is None
        assert _construct_annex_url("https://example.com/agenda.docx") is None
        assert _construct_annex_url("") is None


class TestParseRssEventItems:
    """Tests for _parse_rss_event_items."""

    def test_finds_chmp_events(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _parse_rss_event_items

        items = _parse_rss_event_items(RSS_EVENTS_XML, "CHMP")
        assert len(items) == 2
        assert items[0]["meeting_start"] == "2026-02-23"
        assert items[0]["meeting_end"] == "2026-02-26"
        assert "/en/events/" in items[0]["url"]

    def test_skips_non_committee(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _parse_rss_event_items

        items = _parse_rss_event_items(RSS_EVENTS_XML, "CHMP")
        # The AI workshop should not match CHMP slugs
        titles = [i["title"] for i in items]
        assert not any("workshop" in t.lower() for t in titles)

    def test_parses_dates_from_title(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _parse_rss_event_items

        items = _parse_rss_event_items(RSS_EVENTS_XML, "CHMP")
        march_items = [i for i in items if i["meeting_end"] == "2026-03-27"]
        assert len(march_items) == 1
        assert march_items[0]["meeting_start"] == "2026-03-24"


class TestParseRfc2822Date:
    """Tests for _parse_rfc2822_date."""

    def test_valid_date(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _parse_rfc2822_date

        assert _parse_rfc2822_date("Thu, 20 Feb 2026 00:00:00 +0100") == "2026-02-20"

    def test_invalid_date_returns_none(self):
        from wake_robin_data_pipeline.collectors.ema_committee_collector import _parse_rfc2822_date

        assert _parse_rfc2822_date("not a date") is None
        assert _parse_rfc2822_date("") is None


# ===========================================================================
# RSS integration tests
# ===========================================================================


class TestCollectRssIntegration:
    """Integration tests for three-tier RSS discovery."""

    def _build_xlsx(self):
        return _build_test_xlsx([
            _make_row("MAA", "Aficamten", "aficamten", "Cytokinetics"),
            _make_row("Variation type II", "Keytruda", "pembrolizumab", "Merck"),
            _make_row("MAA", "UnmappedDrug", "unknown_sub", "UnknownCo"),
        ])

    def test_tier1_rss_agenda_success(self, tmp_path):
        """Tier 1: RSS agenda → annex XLSX → product events with tickers."""
        from wake_robin_data_pipeline.collectors.ema_committee_collector import (
            collect_ema_committee_events,
        )

        xlsx_bytes = self._build_xlsx()

        with patch(f"{_MOD}._fetch_xml", return_value=RSS_AGENDAS_XML), \
             patch(f"{_MOD}._fetch_xlsx_bytes", return_value=xlsx_bytes), \
             patch(f"{_MOD}._fetch") as mock_fetch:
            # _fetch for landing page (tier 3) — returns page with NO event links
            mock_fetch.return_value = "<html><body></body></html>"
            events = collect_ema_committee_events(
                as_of_date=date(2026, 2, 28),
                cache_dir=tmp_path,
                product_ticker_map=PRODUCT_MAP,
                committees=("CHMP",),
            )

        tickers = [e["ticker"] for e in events if e["ticker"]]
        assert "CYTK" in tickers
        assert "MRK" in tickers
        assert all(e["item_type"] == "agenda_item" for e in events if e["ticker"])

        # Check stats
        cache_path = tmp_path / "ema_committee_events_2026-02-28.json"
        payload = json.loads(cache_path.read_text())
        stats = payload["stats"]
        assert stats["rss_agenda_items"] >= 1
        assert stats["discovery_tier"].get("CHMP|2026-02-26") == "rss_agendas"

    def test_tier2_when_tier1_fails(self, tmp_path):
        """Tier 2: RSS events feed → event page → annex XLSX when tier 1 fails."""
        from wake_robin_data_pipeline.collectors.ema_committee_collector import (
            collect_ema_committee_events,
        )

        xlsx_bytes = self._build_xlsx()

        # RSS agendas returns no matching items (empty feed)
        empty_rss = '<?xml version="1.0"?><rss><channel></channel></rss>'

        def mock_fetch(url):
            if "/en/events/" in url:
                return EVENT_PAGE_HTML
            # Landing page — no event page links
            return "<html><body></body></html>"

        with patch(f"{_MOD}._fetch_xml") as mock_xml, \
             patch(f"{_MOD}._fetch_xlsx_bytes", return_value=xlsx_bytes), \
             patch(f"{_MOD}._fetch", side_effect=mock_fetch):
            # First call (_RSS_AGENDAS_URL) returns empty, second (_RSS_EVENTS_URL) returns events
            mock_xml.side_effect = [empty_rss, RSS_EVENTS_XML]
            events = collect_ema_committee_events(
                as_of_date=date(2026, 2, 28),
                cache_dir=tmp_path,
                product_ticker_map=PRODUCT_MAP,
                committees=("CHMP",),
            )

        tickers = [e["ticker"] for e in events if e["ticker"]]
        assert "CYTK" in tickers
        assert "MRK" in tickers

        cache_path = tmp_path / "ema_committee_events_2026-02-28.json"
        payload = json.loads(cache_path.read_text())
        stats = payload["stats"]
        assert stats["rss_event_items"] >= 1
        assert stats["discovery_tier"].get("CHMP|2026-02-26") == "rss_events"

    def test_tier3_fallback_when_rss_empty(self, tmp_path):
        """Tier 3: both RSS feeds empty → landing page fallback."""
        from wake_robin_data_pipeline.collectors.ema_committee_collector import (
            collect_ema_committee_events,
        )

        def mock_fetch(url):
            # Return landing page with docs but NO event page links
            return COMMITTEE_LANDING_HTML

        with patch(f"{_MOD}._fetch_xml", side_effect=Exception("no RSS")), \
             patch(f"{_MOD}._fetch", side_effect=mock_fetch):
            events = collect_ema_committee_events(
                as_of_date=date(2026, 1, 31),
                cache_dir=tmp_path,
                product_ticker_map=PRODUCT_MAP,
                committees=("CHMP",),
            )

        # Should fall back to meeting-level events
        meeting_events = [e for e in events if e["item_type"] == "meeting"]
        assert len(meeting_events) > 0

    def test_dedup_across_tiers(self, tmp_path):
        """Same meeting in tier 1 and tier 2: only processed once."""
        from wake_robin_data_pipeline.collectors.ema_committee_collector import (
            collect_ema_committee_events,
        )

        xlsx_bytes = self._build_xlsx()

        def mock_fetch(url):
            if "/en/events/" in url:
                return EVENT_PAGE_HTML
            return "<html><body></body></html>"

        with patch(f"{_MOD}._fetch_xml") as mock_xml, \
             patch(f"{_MOD}._fetch_xlsx_bytes", return_value=xlsx_bytes), \
             patch(f"{_MOD}._fetch", side_effect=mock_fetch):
            # Both RSS feeds return the same Feb 2026 meeting
            mock_xml.side_effect = [RSS_AGENDAS_XML, RSS_EVENTS_XML]
            events = collect_ema_committee_events(
                as_of_date=date(2026, 2, 28),
                cache_dir=tmp_path,
                product_ticker_map=PRODUCT_MAP,
                committees=("CHMP",),
            )

        # Count product-level events for the Feb meeting
        feb_events = [e for e in events if e["meeting_end"] == "2026-02-26" and e["ticker"]]
        # Should be 2 (CYTK + MRK) not 4 (duplicated)
        assert len(feb_events) == 2

    def test_stats_fields_populated(self, tmp_path):
        """Stats include rss_agenda_items, rss_event_items, discovery_tier."""
        from wake_robin_data_pipeline.collectors.ema_committee_collector import (
            collect_ema_committee_events,
        )

        xlsx_bytes = self._build_xlsx()

        with patch(f"{_MOD}._fetch_xml", return_value=RSS_AGENDAS_XML), \
             patch(f"{_MOD}._fetch_xlsx_bytes", return_value=xlsx_bytes), \
             patch(f"{_MOD}._fetch", return_value="<html><body></body></html>"):
            collect_ema_committee_events(
                as_of_date=date(2026, 2, 28),
                cache_dir=tmp_path,
                product_ticker_map=PRODUCT_MAP,
                committees=("CHMP",),
            )

        cache_path = tmp_path / "ema_committee_events_2026-02-28.json"
        payload = json.loads(cache_path.read_text())
        stats = payload["stats"]
        assert "rss_agenda_items" in stats
        assert "rss_event_items" in stats
        assert "discovery_tier" in stats
        assert isinstance(stats["discovery_tier"], dict)

    def test_determinism_two_runs(self, tmp_path):
        """Two runs with same inputs produce identical output."""
        from wake_robin_data_pipeline.collectors.ema_committee_collector import (
            collect_ema_committee_events,
        )

        xlsx_bytes = self._build_xlsx()

        def run():
            d = tmp_path / "run"
            d.mkdir(exist_ok=True)
            with patch(f"{_MOD}._fetch_xml", return_value=RSS_AGENDAS_XML), \
                 patch(f"{_MOD}._fetch_xlsx_bytes", return_value=xlsx_bytes), \
                 patch(f"{_MOD}._fetch", return_value="<html><body></body></html>"):
                return collect_ema_committee_events(
                    as_of_date=date(2026, 2, 28),
                    cache_dir=d,
                    product_ticker_map=PRODUCT_MAP,
                    committees=("CHMP",),
                )

        # Delete cache between runs to force re-collection
        result1 = run()
        (tmp_path / "run" / "ema_committee_events_2026-02-28.json").unlink()
        result2 = run()

        ids1 = [e["id"] for e in result1]
        ids2 = [e["id"] for e in result2]
        assert ids1 == ids2
