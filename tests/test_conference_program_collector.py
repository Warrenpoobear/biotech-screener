"""Tests for conference program + abstract collector (offline, deterministic)."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from wake_robin_data_pipeline.collectors.conference_program_collector import (
    collect_conference_sessions,
    collect_conference_abstracts,
    collect_conference_derived_events,
    derive_events_from_records,
    normalize_name,
    _stable_hash10,
    _extract_entities,
    _map_to_ticker,
    _write_cache,
    _read_cache_if_exists,
)

AS_OF = date(2026, 5, 20)

# ---------------------------------------------------------------------------
# Inline HTML fixtures
# ---------------------------------------------------------------------------

PROGRAM_HTML = """\
<html><body>
<div class="session" data-type="Late-Breaking Abstract Session" data-track="Lung Cancer">
  <span class="session-title">Late-Breaking Abstract Session: Lung Cancer NCT01234567</span>
  <span class="session-time" data-start="2026-06-01T09:00:00" data-end="2026-06-01T10:30:00"></span>
  <span class="session-location">Hall B</span>
</div>
<div class="session" data-type="Poster Session" data-track="Breast Cancer">
  <span class="session-title">Poster Session: Breast Cancer Highlights</span>
  <span class="session-time" data-start="2026-06-02T14:00:00" data-end="2026-06-02T16:00:00"></span>
  <span class="session-location">Exhibit Hall</span>
</div>
</body></html>
"""

ABSTRACTS_HTML = """\
<html><body>
<div class="abstract" data-code="LBA9001" data-type="oral">
  <span class="abstract-title">Pembrolizumab + Chemo in Advanced NSCLC: NCT01234567</span>
  <span class="abstract-presenter">Smith, John</span>
  <span class="abstract-affiliation">Memorial Sloan Kettering</span>
  <span class="abstract-sponsor">Merck Sharp &amp; Dohme</span>
</div>
<div class="abstract" data-code="5002" data-type="poster">
  <span class="abstract-title">Novel ABX-101 in Solid Tumors: Phase 2 Results</span>
  <span class="abstract-presenter">Doe, Jane</span>
  <span class="abstract-affiliation">Dana-Farber</span>
  <span class="abstract-sponsor">AcmeBio Inc</span>
</div>
</body></html>
"""

# Mapping dicts
PRODUCT_MAP = {
    "pembrolizumab": "MRK",
    "abx-101": "ACME",
}
COMPANY_MAP = {
    "merck sharp dohme": "MRK",  # normalized: lowercase, strip punct, collapse ws
    "acmebio inc": "ACME",
}
NCT_MAP = {
    "NCT01234567": "MRK",
}


def _mock_fetch(url: str) -> str:
    if "program" in url:
        return PROGRAM_HTML
    if "abstract" in url:
        return ABSTRACTS_HTML
    return ""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCacheWriteAndReload:
    """Test deterministic cache write and reload."""

    def test_cache_write_and_reload_deterministic(self, tmp_path):
        sessions = [
            {"id": "S1", "title": "Session A", "start_dt_utc": "2026-06-01T14:00:00Z"},
            {"id": "S2", "title": "Session B", "start_dt_utc": "2026-06-01T09:00:00Z"},
        ]
        cache_path = tmp_path / "sessions_2026-05-20.json"
        payload = {
            "schema": "conference_sessions.v1",
            "as_of_date": "2026-05-20",
            "records": sessions,
        }
        _write_cache(cache_path, payload)

        # Reload
        loaded = _read_cache_if_exists(cache_path)
        assert loaded is not None
        assert len(loaded) == 2

        # Verify sort_keys determinism: JSON keys should be sorted
        raw = cache_path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        keys = list(parsed.keys())
        assert keys == sorted(keys)


class TestIdStability:
    """Test that same payload produces same IDs."""

    def test_id_stability(self, tmp_path):
        with patch(
            "wake_robin_data_pipeline.collectors.conference_program_collector._fetch",
            side_effect=_mock_fetch,
        ):
            s1 = collect_conference_sessions(
                conference_slug="asco", edition_year=2026,
                as_of_date=AS_OF, cache_dir=tmp_path, fetch_live=True,
            )
            # Clear cache and re-collect
            for f in tmp_path.rglob("sessions_*.json"):
                f.unlink()
            s2 = collect_conference_sessions(
                conference_slug="asco", edition_year=2026,
                as_of_date=AS_OF, cache_dir=tmp_path, fetch_live=True,
            )
        assert len(s1) == len(s2)
        for a, b in zip(s1, s2):
            assert a["id"] == b["id"], f"ID mismatch: {a['id']} vs {b['id']}"


class TestEntityExtraction:
    """Test NCT ID regex extraction."""

    def test_entity_extraction_nct(self):
        text = "Trial NCT01234567 and NCT99887766 showed promising results"
        entities = _extract_entities(text)
        assert "NCT01234567" in entities["nct_ids"]
        assert "NCT99887766" in entities["nct_ids"]
        assert len(entities["nct_ids"]) == 2

    def test_entity_extraction_no_nct(self):
        text = "No clinical trial IDs in this text"
        entities = _extract_entities(text)
        assert len(entities["nct_ids"]) == 0


class TestMappingPrecedence:
    """Test NCT > drug > company mapping precedence."""

    def test_mapping_precedence_nct_over_drug_over_company(self):
        entities = {
            "companies": ["AcmeBio Inc"],
            "drugs": ["pembrolizumab"],
            "nct_ids": ["NCT01234567"],
        }
        ticker, method, val = _map_to_ticker(entities, PRODUCT_MAP, COMPANY_MAP, NCT_MAP)
        assert ticker == "MRK"
        assert method == "nct"

    def test_mapping_drug_when_no_nct(self):
        entities = {
            "companies": ["AcmeBio Inc"],
            "drugs": ["abx-101"],
            "nct_ids": [],
        }
        ticker, method, val = _map_to_ticker(entities, PRODUCT_MAP, COMPANY_MAP, None)
        assert ticker == "ACME"
        assert method == "drug"

    def test_mapping_company_when_no_drug(self):
        entities = {
            "companies": ["Merck Sharp & Dohme"],
            "drugs": [],
            "nct_ids": [],
        }
        ticker, method, val = _map_to_ticker(entities, PRODUCT_MAP, COMPANY_MAP, None)
        assert ticker == "MRK"
        assert method == "company"

    def test_mapping_none_when_unmatched(self):
        entities = {
            "companies": ["Unknown Corp"],
            "drugs": ["mystery-drug"],
            "nct_ids": [],
        }
        ticker, method, val = _map_to_ticker(entities, PRODUCT_MAP, COMPANY_MAP, None)
        assert ticker is None
        assert method is None


class TestUnmatchedTelemetry:
    """Test that unmatched items are logged and bounded."""

    def test_unmatched_logged_and_bounded(self):
        # Create many abstracts with unmatchable entities
        abstracts = []
        for i in range(150):
            abstracts.append({
                "schema": "conference_abstract.v1",
                "id": f"ABS_{i}",
                "title": f"Unknown Drug-{i} Phase 2 Results",
                "abstract_code": f"P{i:04d}",
                "presentation_type": "poster",
                "sponsor_company": f"UnknownCo-{i}",
                "entities": {"companies": [], "drugs": [f"unknown-drug-{i}"], "nct_ids": []},
                "url": "",
                "disclosed_at": None,
                "session_id": None,
            })

        events, stats = derive_events_from_records(
            sessions=[], abstracts=abstracts,
            product_ticker_map={}, company_ticker_map={},
            nct_ticker_map=None, conference="ASCO",
            edition_year=2026, as_of_date=AS_OF,
        )
        assert len(events) == 0
        assert stats["unmatched_items"] <= 100


class TestDerivedEventsSortingAndConfidence:
    """Test derived events are sorted and have correct confidence."""

    def test_derived_events_sorted_and_confidence(self, tmp_path):
        with patch(
            "wake_robin_data_pipeline.collectors.conference_program_collector._fetch",
            side_effect=_mock_fetch,
        ):
            events = collect_conference_derived_events(
                conference_slug="asco", edition_year=2026,
                as_of_date=AS_OF, cache_dir=tmp_path,
                product_ticker_map=PRODUCT_MAP,
                company_ticker_map=COMPANY_MAP,
                nct_ticker_map=NCT_MAP,
            )

        assert len(events) >= 1

        # Check sorted by (event_date, ticker, event_type, id)
        sort_keys = [(e["event_date"], e["ticker"], e["event_type"], e["id"]) for e in events]
        assert sort_keys == sorted(sort_keys)

        # Check late-breaker confidence
        late_breakers = [e for e in events if e["event_type"] == "CONF_LATE_BREAKER"]
        for lb in late_breakers:
            assert lb["confidence"] == "HIGH"

        # Check presentation confidence
        presentations = [e for e in events if e["event_type"] == "CONF_PRESENTATION"]
        for p in presentations:
            assert p["confidence"] == "MED"


class TestCacheOnlyMode:
    """Test that fetch_live=False reads from cache only."""

    def test_disable_live_uses_cache_only(self, tmp_path):
        # No cache exists, fetch_live=False → empty
        sessions = collect_conference_sessions(
            conference_slug="asco", edition_year=2026,
            as_of_date=AS_OF, cache_dir=tmp_path, fetch_live=False,
        )
        assert sessions == []

        # Write cache, then read with fetch_live=False
        with patch(
            "wake_robin_data_pipeline.collectors.conference_program_collector._fetch",
            side_effect=_mock_fetch,
        ):
            sessions_live = collect_conference_sessions(
                conference_slug="asco", edition_year=2026,
                as_of_date=AS_OF, cache_dir=tmp_path, fetch_live=True,
            )
        assert len(sessions_live) > 0

        # Now read from cache without network
        sessions_cached = collect_conference_sessions(
            conference_slug="asco", edition_year=2026,
            as_of_date=AS_OF, cache_dir=tmp_path, fetch_live=False,
        )
        assert len(sessions_cached) == len(sessions_live)


class TestNormalizeName:
    """Test name normalization."""

    def test_normalize_strips_punctuation(self):
        assert normalize_name("Merck Sharp & Dohme") == "merck sharp dohme"

    def test_normalize_collapses_whitespace(self):
        assert normalize_name("  AcmeBio   Inc  ") == "acmebio inc"


class TestModule3ConferenceIngest:
    """Test Module 3 conference event conversion."""

    def test_convert_conference_event_to_v2(self):
        from module_3_catalyst import convert_conference_event_to_v2
        from module_3_schema import EventType, ConfidenceLevel

        event = {
            "id": "CONF_ASCO_2026_CONF_LATE_BREAKER_2026-06-01_abc123",
            "ticker": "MRK",
            "conference": "ASCO",
            "edition_year": 2026,
            "event_type": "CONF_LATE_BREAKER",
            "event_date": "2026-06-01",
            "disclosed_at": "2026-05-10",
            "title": "Late-Breaking Lung Cancer Data",
            "confidence": "HIGH",
            "source": "CONF_ABSTRACTS",
        }
        v2 = convert_conference_event_to_v2(event, AS_OF)
        assert v2 is not None
        assert v2.ticker == "MRK"
        assert v2.event_type == EventType.CONFERENCE_LATE_BREAKER
        assert v2.confidence == ConfidenceLevel.HIGH
        assert v2.event_date == "2026-06-01"
        assert v2.field_changed == "conference_calendar"
        assert "conference:asco" in v2.tags

    def test_convert_presentation_event(self):
        from module_3_catalyst import convert_conference_event_to_v2
        from module_3_schema import EventType, ConfidenceLevel

        event = {
            "id": "CONF_ASCO_2026_CONF_PRESENTATION_2026-06-02_def456",
            "ticker": "ACME",
            "conference": "ASCO",
            "event_type": "CONF_PRESENTATION",
            "event_date": "2026-06-02",
            "confidence": "MED",
            "source": "CONF_ABSTRACTS",
            "title": "Phase 2 poster",
        }
        v2 = convert_conference_event_to_v2(event, AS_OF)
        assert v2 is not None
        assert v2.event_type == EventType.CONFERENCE_PRESENTATION
        assert v2.confidence == ConfidenceLevel.MED

    def test_convert_returns_none_for_missing_ticker(self):
        from module_3_catalyst import convert_conference_event_to_v2

        event = {"event_type": "CONF_PRESENTATION", "event_date": "2026-06-02"}
        v2 = convert_conference_event_to_v2(event, AS_OF)
        assert v2 is None

    def test_convert_returns_none_for_unknown_type(self):
        from module_3_catalyst import convert_conference_event_to_v2

        event = {"ticker": "MRK", "event_type": "UNKNOWN_TYPE", "event_date": "2026-06-02"}
        v2 = convert_conference_event_to_v2(event, AS_OF)
        assert v2 is None
