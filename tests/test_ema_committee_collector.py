"""
Tests for EMA committee agenda + meeting highlights collectors.

All tests use inline HTML fixtures + monkeypatched _fetch — zero network calls.
"""

import json
import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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
