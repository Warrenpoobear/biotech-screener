"""Tests for press release collector (offline, deterministic)."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from wake_robin_data_pipeline.collectors.press_release_collector import (
    collect_press_release_events,
    normalize_company,
    extract_future_dates_from_text,
    _build_universe_map,
    _classify_event_kind,
    _parse_rss_items,
    _parse_pub_date,
    _parse_month_day_infer_year,
    _stable_id,
    _has_future_intent_near,
    _is_historical_context,
)

AS_OF = date(2026, 3, 1)

# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------

UNIVERSE = [
    {
        "ticker": "ACME",
        "name": "Acme Therapeutics Inc",
        "pr_rss_url": "https://ir.acme.com/rss/news",
        "market_data": {"company_name": "Acme Therapeutics, Inc."},
    },
    {
        "ticker": "BETA",
        "name": "Beta Biotech",
        "market_data": {"company_name": "Beta Biotech Ltd"},
        # no pr_rss_url — should be skipped
    },
]

# RSS fixture with 2 items:
# 1) "will present on March 15, 2026" => included (future + intent)
# 2) "presented on February 10, 2026" => excluded (historical)
RSS_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Acme Therapeutics Press Releases</title>
    <item>
      <title>Acme Therapeutics will present Phase 3 data at ASCO on March 15, 2026</title>
      <link>https://ir.acme.com/news/release1</link>
      <pubDate>Mon, 24 Feb 2026 12:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Acme Therapeutics presented Phase 2 data on February 10, 2026 at JPM</title>
      <link>https://ir.acme.com/news/release2</link>
      <pubDate>Tue, 10 Feb 2026 09:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Acme Therapeutics reports Q2 2026 results expected in Q2 2026</title>
      <link>https://ir.acme.com/news/release3</link>
      <pubDate>Wed, 25 Feb 2026 14:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""


# ---------------------------------------------------------------------------
# Tests: normalize_company
# ---------------------------------------------------------------------------

class TestNormalizeCompany:
    def test_basic(self):
        assert normalize_company("Acme Therapeutics, Inc.") == "acme"

    def test_strips_suffixes(self):
        assert normalize_company("Beta Biotech Ltd") == "beta"


# ---------------------------------------------------------------------------
# Tests: _parse_rss_items
# ---------------------------------------------------------------------------

class TestParseRssItems:
    def test_rss_items(self):
        items = _parse_rss_items(RSS_XML)
        assert len(items) == 3
        assert items[0]["title"].startswith("Acme Therapeutics will present")
        assert items[0]["link"] == "https://ir.acme.com/news/release1"

    def test_empty_xml(self):
        items = _parse_rss_items("<rss><channel></channel></rss>")
        assert items == []

    def test_invalid_xml(self):
        items = _parse_rss_items("not xml at all")
        assert items == []


# ---------------------------------------------------------------------------
# Tests: _parse_pub_date
# ---------------------------------------------------------------------------

class TestParsePubDate:
    def test_rfc2822(self):
        d = _parse_pub_date("Mon, 24 Feb 2026 12:00:00 GMT")
        assert d == date(2026, 2, 24)

    def test_iso(self):
        d = _parse_pub_date("2026-02-24T12:00:00Z")
        assert d == date(2026, 2, 24)

    def test_empty(self):
        assert _parse_pub_date("") is None

    def test_bad(self):
        assert _parse_pub_date("not a date") is None


# ---------------------------------------------------------------------------
# Tests: extract_future_dates_from_text
# ---------------------------------------------------------------------------

class TestExtractFutureDates:
    def test_future_with_intent(self):
        text = "Acme Therapeutics will present Phase 3 data at ASCO on March 15, 2026"
        results = extract_future_dates_from_text(text, date(2026, 2, 24), AS_OF)
        assert len(results) == 1
        assert results[0]["event_date"] == "2026-03-15"
        assert "will present" in results[0]["matched_phrases"]

    def test_historical_excluded(self):
        text = "Acme Therapeutics presented data on February 10, 2026 at JPM"
        results = extract_future_dates_from_text(text, date(2026, 2, 10), AS_OF)
        assert len(results) == 0

    def test_quarter_only_excluded(self):
        """Q2 2026 without day precision should NOT be emitted."""
        text = "Results expected in Q2 2026"
        results = extract_future_dates_from_text(text, date(2026, 2, 25), AS_OF)
        assert len(results) == 0

    def test_past_date_excluded(self):
        text = "We will present data on January 15, 2026"
        results = extract_future_dates_from_text(text, date(2026, 1, 10), AS_OF)
        # January 15, 2026 is before AS_OF (March 1, 2026)
        assert len(results) == 0

    def test_month_day_no_year_infer(self):
        """Month/day without year should infer year from disclosed_at."""
        text = "Acme Therapeutics will present data on April 20"
        results = extract_future_dates_from_text(text, date(2026, 3, 1), AS_OF)
        assert len(results) == 1
        assert results[0]["event_date"] == "2026-04-20"

    def test_month_day_no_year_roll_forward(self):
        """If inferred date would be past, roll to next year."""
        text = "Acme Therapeutics will present data on January 10"
        # disclosed_at is March 2026 => January 10 would be in the past,
        # so should roll to January 10, 2027
        results = extract_future_dates_from_text(text, date(2026, 3, 1), AS_OF)
        assert len(results) == 1
        assert results[0]["event_date"] == "2027-01-10"

    def test_multiple_dates_in_text(self):
        text = (
            "Acme will present at ASCO on June 5, 2026 "
            "and will host an investor day on July 10, 2026"
        )
        results = extract_future_dates_from_text(text, date(2026, 3, 1), AS_OF)
        assert len(results) == 2
        dates = {r["event_date"] for r in results}
        assert "2026-06-05" in dates
        assert "2026-07-10" in dates

    def test_no_intent_phrase_excluded(self):
        """Date without a future-intent phrase should not be emitted."""
        text = "The company was founded on March 15, 2026 in Delaware"
        results = extract_future_dates_from_text(text, date(2026, 2, 1), AS_OF)
        assert len(results) == 0

    def test_abbreviated_month(self):
        text = "Acme Therapeutics will present data on Apr 20, 2026"
        results = extract_future_dates_from_text(text, date(2026, 3, 1), AS_OF)
        assert len(results) == 1
        assert results[0]["event_date"] == "2026-04-20"


# ---------------------------------------------------------------------------
# Tests: _has_future_intent_near / _is_historical_context
# ---------------------------------------------------------------------------

class TestIntentDetection:
    def test_future_intent(self):
        text = "We will present new data on the target date"
        is_future, phrases = _has_future_intent_near(text, 30)
        assert is_future
        assert "will present" in phrases

    def test_no_intent(self):
        text = "The company was established in the area"
        is_future, phrases = _has_future_intent_near(text, 10)
        assert not is_future
        assert phrases == []

    def test_historical(self):
        text = "Data was presented on December 1 at the conference"
        assert _is_historical_context(text, 20)

    def test_not_historical(self):
        text = "We will present data at the conference"
        assert not _is_historical_context(text, 5)


# ---------------------------------------------------------------------------
# Tests: _classify_event_kind
# ---------------------------------------------------------------------------

class TestClassifyEventKind:
    def test_data_readout(self):
        assert _classify_event_kind("Phase 3 top-line data readout") == "data_readout"

    def test_conference(self):
        assert _classify_event_kind("Oral presentation at ASCO") == "conference_presentation"

    def test_abstract(self):
        assert _classify_event_kind("Abstract accepted for poster") == "abstract_accepted"

    def test_earnings(self):
        assert _classify_event_kind("Q1 2026 Earnings Call") == "earnings_date"

    def test_guidance(self):
        assert _classify_event_kind("2026 Financial Guidance Update") == "guidance"

    def test_other(self):
        assert _classify_event_kind("Board of directors meeting") == "other"


# ---------------------------------------------------------------------------
# Tests: _stable_id
# ---------------------------------------------------------------------------

class TestStableId:
    def test_deterministic(self):
        id1 = _stable_id("ACME", "2026-03-15", "Title", "http://x.com")
        id2 = _stable_id("ACME", "2026-03-15", "Title", "http://x.com")
        assert id1 == id2
        assert len(id1) == 10

    def test_different_inputs(self):
        id1 = _stable_id("ACME", "2026-03-15", "Title1", "http://x.com")
        id2 = _stable_id("ACME", "2026-03-15", "Title2", "http://x.com")
        assert id1 != id2


# ---------------------------------------------------------------------------
# Tests: _parse_month_day_infer_year
# ---------------------------------------------------------------------------

class TestParseMonthDayInferYear:
    def test_future_date(self):
        d = _parse_month_day_infer_year("April", "20", date(2026, 3, 1))
        assert d == date(2026, 4, 20)

    def test_past_date_rolls_forward(self):
        d = _parse_month_day_infer_year("January", "10", date(2026, 3, 1))
        assert d == date(2027, 1, 10)

    def test_abbreviated_month(self):
        d = _parse_month_day_infer_year("Apr", "20", date(2026, 3, 1))
        assert d == date(2026, 4, 20)

    def test_invalid(self):
        d = _parse_month_day_infer_year("NotAMonth", "99", date(2026, 3, 1))
        assert d is None


# ---------------------------------------------------------------------------
# Tests: collect_press_release_events (integration, no network)
# ---------------------------------------------------------------------------

class TestCollectPressReleaseEvents:
    def test_cache_only_no_cache(self, tmp_path):
        events = collect_press_release_events(
            universe=UNIVERSE,
            as_of_date=AS_OF,
            cache_dir=tmp_path,
            cache_only=True,
        )
        assert events == []

    def test_extracts_future_only(self, tmp_path):
        """Only future-dated events with intent phrases should be emitted."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = RSS_XML

        with patch("wake_robin_data_pipeline.collectors.press_release_collector.time.sleep"):
            with patch(
                "wake_robin_data_pipeline.collectors.press_release_collector.requests",
            ) as mock_req_mod:
                mock_req_mod.get.return_value = mock_resp
                events = collect_press_release_events(
                    universe=UNIVERSE,
                    as_of_date=AS_OF,
                    cache_dir=tmp_path,
                )

        # Only the "will present on March 15, 2026" should be extracted
        assert len(events) == 1
        assert events[0]["event_date"] == "2026-03-15"
        assert events[0]["schema"] == "press_release_event.v1"
        assert events[0]["source"] == "PRESS_RELEASE"
        assert events[0]["id"].startswith("PR_ACME_")

    def test_cache_round_trip(self, tmp_path):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = RSS_XML

        with patch("wake_robin_data_pipeline.collectors.press_release_collector.time.sleep"):
            with patch(
                "wake_robin_data_pipeline.collectors.press_release_collector.requests",
            ) as mock_req_mod:
                mock_req_mod.get.return_value = mock_resp
                events_live = collect_press_release_events(
                    universe=UNIVERSE,
                    as_of_date=AS_OF,
                    cache_dir=tmp_path,
                )

        cache_path = tmp_path / f"press_release_events_{AS_OF.isoformat()}.json"
        assert cache_path.exists()

        wrapper = json.loads(cache_path.read_text())
        assert wrapper["schema"] == "press_release_events.v1"
        assert wrapper["as_of_date"] == AS_OF.isoformat()
        assert wrapper["collector_version"] == "press_release_collector.v1"
        assert "events" in wrapper
        assert "stats" in wrapper

        events_cached = collect_press_release_events(
            universe=UNIVERSE,
            as_of_date=AS_OF,
            cache_dir=tmp_path,
            cache_only=True,
        )
        assert len(events_cached) == len(events_live)

    def test_stable_ordering(self, tmp_path):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = RSS_XML

        with patch("wake_robin_data_pipeline.collectors.press_release_collector.time.sleep"):
            with patch(
                "wake_robin_data_pipeline.collectors.press_release_collector.requests",
            ) as mock_req_mod:
                mock_req_mod.get.return_value = mock_resp
                events = collect_press_release_events(
                    universe=UNIVERSE,
                    as_of_date=AS_OF,
                    cache_dir=tmp_path,
                )

        dates = [e["event_date"] for e in events]
        assert dates == sorted(dates)

    def test_no_rss_url_returns_empty(self, tmp_path):
        universe_no_rss = [
            {"ticker": "BETA", "name": "Beta Biotech", "market_data": {}},
        ]
        with patch("wake_robin_data_pipeline.collectors.press_release_collector.time.sleep"):
            with patch(
                "wake_robin_data_pipeline.collectors.press_release_collector.requests",
            ) as mock_req_mod:
                events = collect_press_release_events(
                    universe=universe_no_rss,
                    as_of_date=AS_OF,
                    cache_dir=tmp_path,
                )
        assert events == []

    def test_network_failure_returns_empty(self, tmp_path):
        with patch("wake_robin_data_pipeline.collectors.press_release_collector.time.sleep"):
            with patch(
                "wake_robin_data_pipeline.collectors.press_release_collector.requests",
            ) as mock_req_mod:
                mock_req_mod.get.side_effect = ConnectionError("timeout")
                events = collect_press_release_events(
                    universe=UNIVERSE,
                    as_of_date=AS_OF,
                    cache_dir=tmp_path,
                )
        assert events == []

    def test_event_kind_classification(self, tmp_path):
        """The first RSS item mentions 'Phase 3 data' -> data_readout (matched first)."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = RSS_XML

        with patch("wake_robin_data_pipeline.collectors.press_release_collector.time.sleep"):
            with patch(
                "wake_robin_data_pipeline.collectors.press_release_collector.requests",
            ) as mock_req_mod:
                mock_req_mod.get.return_value = mock_resp
                events = collect_press_release_events(
                    universe=UNIVERSE,
                    as_of_date=AS_OF,
                    cache_dir=tmp_path,
                )

        assert len(events) >= 1
        # "Phase 3 data" matches before "will present" in keyword priority
        assert events[0]["event_kind"] == "data_readout"

    def test_disclosed_at_from_pub_date(self, tmp_path):
        """disclosed_at should come from pubDate."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = RSS_XML

        with patch("wake_robin_data_pipeline.collectors.press_release_collector.time.sleep"):
            with patch(
                "wake_robin_data_pipeline.collectors.press_release_collector.requests",
            ) as mock_req_mod:
                mock_req_mod.get.return_value = mock_resp
                events = collect_press_release_events(
                    universe=UNIVERSE,
                    as_of_date=AS_OF,
                    cache_dir=tmp_path,
                )

        assert len(events) >= 1
        assert events[0]["disclosed_at"] == "2026-02-24"

    def test_stats_telemetry(self, tmp_path):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = RSS_XML

        with patch("wake_robin_data_pipeline.collectors.press_release_collector.time.sleep"):
            with patch(
                "wake_robin_data_pipeline.collectors.press_release_collector.requests",
            ) as mock_req_mod:
                mock_req_mod.get.return_value = mock_resp
                collect_press_release_events(
                    universe=UNIVERSE,
                    as_of_date=AS_OF,
                    cache_dir=tmp_path,
                )

        cache_path = tmp_path / f"press_release_events_{AS_OF.isoformat()}.json"
        wrapper = json.loads(cache_path.read_text())
        stats = wrapper["stats"]
        assert stats["fetched_pages"] == 1
        assert stats["parsed_items"] == 3  # 3 RSS items
        assert stats["matched"] >= 1
