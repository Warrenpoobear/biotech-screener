"""Tests for IR events collector (offline, deterministic)."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from wake_robin_data_pipeline.collectors.ir_events_collector import (
    collect_ir_events,
    normalize_company,
    _build_universe_map,
    _classify_event_kind,
    _extract_events_from_html,
    _stable_id,
)

AS_OF = date(2026, 3, 1)

# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------

UNIVERSE = [
    {
        "ticker": "ACME",
        "name": "Acme Therapeutics Inc",
        "ir_url": "https://ir.acme.com/events",
        "market_data": {"company_name": "Acme Therapeutics, Inc."},
    },
    {
        "ticker": "BETA",
        "name": "Beta Biotech",
        "market_data": {"company_name": "Beta Biotech Ltd"},
        # no ir_url — should be skipped
    },
]

# HTML fixture with 3 events:
# 1) Future event (2026-03-15) — should be included
# 2) Past event (2026-02-10) — should be excluded
# 3) Non-date text — should be excluded
IR_HTML = """\
<html><body>
<div class="events-list">
  <div class="event-item">
    <time datetime="2026-03-15">March 15, 2026</time>
    <span class="event-title">Q1 2026 Earnings Conference Call</span>
  </div>
  <div class="event-item">
    <time datetime="2026-02-10">February 10, 2026</time>
    <span class="event-title">Investor Day Recap Webcast</span>
  </div>
  <div class="event-item">
    <span class="event-title">Upcoming conferences to be announced</span>
  </div>
  <div class="event-item">
    <time datetime="2026-04-20">April 20, 2026</time>
    <span class="event-title">Annual Shareholders Meeting Webcast</span>
  </div>
</div>
</body></html>
"""


# ---------------------------------------------------------------------------
# Tests: normalize_company
# ---------------------------------------------------------------------------

class TestNormalizeCompany:
    def test_basic(self):
        assert normalize_company("Acme Therapeutics, Inc.") == "acme"

    def test_strips_suffixes(self):
        assert normalize_company("Beta Biotech Ltd") == "beta"

    def test_collapses_whitespace(self):
        assert normalize_company("  Gamma   Pharma  Corp  ") == "gamma pharma"

    def test_strips_punctuation(self):
        assert normalize_company("A.B.C. Inc.") == "a b c"

    def test_preserves_hyphen(self):
        assert normalize_company("Xeno-Bio Inc") == "xeno-bio"


# ---------------------------------------------------------------------------
# Tests: _build_universe_map
# ---------------------------------------------------------------------------

class TestBuildUniverseMap:
    def test_maps_company_name(self):
        exact, ir_urls = _build_universe_map(UNIVERSE)
        assert "acme" in exact
        assert exact["acme"] == "ACME"

    def test_ir_url_only_for_entries_with_url(self):
        _, ir_urls = _build_universe_map(UNIVERSE)
        assert "ACME" in ir_urls
        assert "BETA" not in ir_urls

    def test_empty_universe(self):
        exact, ir_urls = _build_universe_map([])
        assert exact == {}
        assert ir_urls == {}


# ---------------------------------------------------------------------------
# Tests: _classify_event_kind
# ---------------------------------------------------------------------------

class TestClassifyEventKind:
    def test_earnings(self):
        assert _classify_event_kind("Q1 2026 Earnings Conference Call") == "earnings"

    def test_investor_day(self):
        assert _classify_event_kind("Investor Day Recap Webcast") == "investor_day"

    def test_conference(self):
        assert _classify_event_kind("ASCO Symposium Presentation") == "conference"

    def test_webcast(self):
        assert _classify_event_kind("Corporate Webcast Update") == "webcast"

    def test_other(self):
        assert _classify_event_kind("Board meeting") == "other"


# ---------------------------------------------------------------------------
# Tests: _stable_id
# ---------------------------------------------------------------------------

class TestStableId:
    def test_deterministic(self):
        id1 = _stable_id("ACME", "2026-03-15", "Earnings", "http://x.com")
        id2 = _stable_id("ACME", "2026-03-15", "Earnings", "http://x.com")
        assert id1 == id2
        assert len(id1) == 10

    def test_different_inputs(self):
        id1 = _stable_id("ACME", "2026-03-15", "Earnings", "http://x.com")
        id2 = _stable_id("ACME", "2026-03-16", "Earnings", "http://x.com")
        assert id1 != id2


# ---------------------------------------------------------------------------
# Tests: _extract_events_from_html
# ---------------------------------------------------------------------------

class TestExtractEventsFromHtml:
    def test_extracts_future_events_only(self):
        events = _extract_events_from_html(
            IR_HTML, "ACME", "https://ir.acme.com/events", AS_OF
        )
        dates = [e["event_date"] for e in events]
        # 2026-03-15 and 2026-04-20 are future; 2026-02-10 is past
        assert "2026-03-15" in dates
        assert "2026-04-20" in dates
        assert "2026-02-10" not in dates

    def test_event_schema(self):
        events = _extract_events_from_html(
            IR_HTML, "ACME", "https://ir.acme.com/events", AS_OF
        )
        assert len(events) >= 2
        for e in events:
            assert e["schema"] == "ir_event.v1"
            assert e["ticker"] == "ACME"
            assert e["source"] == "IR_EVENTS"
            assert e["confidence"] == "MED"
            assert e["id"].startswith("IR_ACME_")

    def test_no_duplicate_ids(self):
        events = _extract_events_from_html(
            IR_HTML, "ACME", "https://ir.acme.com/events", AS_OF
        )
        ids = [e["id"] for e in events]
        assert len(ids) == len(set(ids))

    def test_empty_html(self):
        events = _extract_events_from_html(
            "<html><body>Nothing here</body></html>",
            "ACME", "https://ir.acme.com/events", AS_OF,
        )
        assert events == []


# ---------------------------------------------------------------------------
# Tests: collect_ir_events (integration, no network)
# ---------------------------------------------------------------------------

class TestCollectIrEvents:
    def test_cache_only_no_cache(self, tmp_path):
        events = collect_ir_events(
            universe=UNIVERSE,
            as_of_date=AS_OF,
            cache_dir=tmp_path,
            cache_only=True,
        )
        assert events == []

    def test_cache_round_trip(self, tmp_path):
        """Write cache, then load from cache_only mode."""
        # First call: simulate a live fetch by patching requests
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = IR_HTML

        with patch("wake_robin_data_pipeline.collectors.ir_events_collector.time.sleep"):
            with patch(
                "wake_robin_data_pipeline.collectors.ir_events_collector.requests",
            ) as mock_req_mod:
                mock_req_mod.get.return_value = mock_resp
                events_live = collect_ir_events(
                    universe=UNIVERSE,
                    as_of_date=AS_OF,
                    cache_dir=tmp_path,
                )

        assert len(events_live) >= 2

        # Verify cache file exists
        cache_path = tmp_path / f"ir_events_{AS_OF.isoformat()}.json"
        assert cache_path.exists()

        # Verify cache wrapper schema
        wrapper = json.loads(cache_path.read_text())
        assert wrapper["schema"] == "ir_events.v1"
        assert wrapper["as_of_date"] == AS_OF.isoformat()
        assert wrapper["collector_version"] == "ir_events_collector.v1"
        assert "events" in wrapper
        assert "stats" in wrapper

        # Load from cache
        events_cached = collect_ir_events(
            universe=UNIVERSE,
            as_of_date=AS_OF,
            cache_dir=tmp_path,
            cache_only=True,
        )
        assert len(events_cached) == len(events_live)
        assert events_cached[0]["id"] == events_live[0]["id"]

    def test_stable_ordering(self, tmp_path):
        """Events should be sorted by (event_date, ticker, source, id)."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = IR_HTML

        with patch("wake_robin_data_pipeline.collectors.ir_events_collector.time.sleep"):
            with patch(
                "wake_robin_data_pipeline.collectors.ir_events_collector.requests",
            ) as mock_req_mod:
                mock_req_mod.get.return_value = mock_resp
                events = collect_ir_events(
                    universe=UNIVERSE,
                    as_of_date=AS_OF,
                    cache_dir=tmp_path,
                )

        dates = [e["event_date"] for e in events]
        assert dates == sorted(dates)

    def test_no_ir_url_returns_empty(self, tmp_path):
        """Universe with no ir_url entries should return empty list."""
        universe_no_url = [
            {"ticker": "BETA", "name": "Beta Biotech", "market_data": {}},
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = IR_HTML

        with patch("wake_robin_data_pipeline.collectors.ir_events_collector.time.sleep"):
            with patch(
                "wake_robin_data_pipeline.collectors.ir_events_collector.requests",
            ) as mock_req_mod:
                mock_req_mod.get.return_value = mock_resp
                events = collect_ir_events(
                    universe=universe_no_url,
                    as_of_date=AS_OF,
                    cache_dir=tmp_path,
                )
        assert events == []

    def test_network_failure_returns_empty(self, tmp_path):
        """Network failure should not crash, just return empty."""
        with patch("wake_robin_data_pipeline.collectors.ir_events_collector.time.sleep"):
            with patch(
                "wake_robin_data_pipeline.collectors.ir_events_collector.requests",
            ) as mock_req_mod:
                mock_req_mod.get.side_effect = ConnectionError("timeout")
                events = collect_ir_events(
                    universe=UNIVERSE,
                    as_of_date=AS_OF,
                    cache_dir=tmp_path,
                )
        # Should return empty but not crash
        assert events == []

    def test_stats_telemetry(self, tmp_path):
        """Cache wrapper should include stats telemetry."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = IR_HTML

        with patch("wake_robin_data_pipeline.collectors.ir_events_collector.time.sleep"):
            with patch(
                "wake_robin_data_pipeline.collectors.ir_events_collector.requests",
            ) as mock_req_mod:
                mock_req_mod.get.return_value = mock_resp
                collect_ir_events(
                    universe=UNIVERSE,
                    as_of_date=AS_OF,
                    cache_dir=tmp_path,
                )

        cache_path = tmp_path / f"ir_events_{AS_OF.isoformat()}.json"
        wrapper = json.loads(cache_path.read_text())
        stats = wrapper["stats"]
        assert stats["fetched_pages"] == 1  # Only ACME has ir_url
        assert stats["matched"] >= 2
        assert isinstance(stats["unmatched_samples"], list)
