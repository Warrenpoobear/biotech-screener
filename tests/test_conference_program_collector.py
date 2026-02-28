"""Tests for conference program + abstract collector (offline, deterministic)."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from wake_robin_data_pipeline.collectors.conference_program_collector import (
    ALL_CONFERENCE_SLUGS,
    CONFERENCE_ADAPTERS,
    EsmoAdapter,
    AacrAdapter,
    AshAdapter,
    SitcAdapter,
    SabcsAdapter,
    AanAdapter,
    AcrAdapter,
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


# ---------------------------------------------------------------------------
# Inline HTML fixtures for new adapters
# ---------------------------------------------------------------------------

# --- ESMO ---
ESMO_PROGRAM_HTML = """\
<html><body>
<div class="session-item" data-type="Late-Breaking Abstract" data-track="Thoracic">
  <span class="session-item-title">Late-Breaking Results in NSCLC NCT05678901</span>
  <span class="session-item-time" data-start="2026-09-15T08:00:00" data-end="2026-09-15T09:30:00"></span>
  <span class="session-item-location">Madrid Hall A</span>
</div>
<div class="session-item" data-type="Poster Discussion" data-track="GI">
  <span class="session-item-title">Poster Discussion: GI Cancers</span>
  <span class="session-item-time" data-start="2026-09-16T14:00:00" data-end="2026-09-16T15:30:00"></span>
  <span class="session-item-location">Poster Area 3</span>
</div>
</body></html>
"""

ESMO_ABSTRACTS_HTML = """\
<html><body>
<div class="abstract-item" data-code="LBA1" data-type="oral">
  <span class="abstract-item-title">Osimertinib Combo in EGFR+ NSCLC NCT05678901</span>
  <span class="abstract-item-presenter">Garcia, Maria</span>
  <span class="abstract-item-affiliation">Vall d'Hebron</span>
  <span class="abstract-item-sponsor">AstraZeneca</span>
</div>
<div class="abstract-item" data-code="450P" data-type="poster">
  <span class="abstract-item-title">Novel BiTE in MSI-H CRC: Phase 1b</span>
  <span class="abstract-item-presenter">Müller, Hans</span>
  <span class="abstract-item-affiliation">Charité Berlin</span>
</div>
</body></html>
"""

# --- AACR ---
AACR_PROGRAM_HTML = """\
<html><body>
<div class="program-session" data-type="Plenary Session" data-track="Immunology">
  <span class="program-session-title">Plenary: Advances in Cancer Immunology</span>
  <span class="program-session-time" data-start="2026-04-12T08:30:00" data-end="2026-04-12T10:00:00"></span>
  <span class="program-session-location">Convention Center Hall A</span>
</div>
<div class="program-session" data-type="Oral Presentations" data-track="Drug Discovery">
  <span class="program-session-title">Oral Session: Novel Drug Targets</span>
  <span class="program-session-time" data-start="2026-04-13T11:00:00" data-end="2026-04-13T12:30:00"></span>
  <span class="program-session-location">Room 210</span>
</div>
</body></html>
"""

AACR_ABSTRACTS_HTML = """\
<html><body>
<div class="abstract-entry" data-code="LB001" data-type="oral">
  <span class="abstract-entry-title">CAR-T Targeting CLDN18.2 in Gastric Cancer NCT06123456</span>
  <span class="abstract-entry-presenter">Chen, Wei</span>
  <span class="abstract-entry-affiliation">MD Anderson</span>
  <span class="abstract-entry-sponsor">Innovent Biologics</span>
</div>
<div class="abstract-entry" data-code="3456" data-type="poster">
  <span class="abstract-entry-title">KRAS G12C Inhibitor Combinations in PDAC</span>
  <span class="abstract-entry-presenter">Patel, Amit</span>
  <span class="abstract-entry-affiliation">Memorial Sloan Kettering</span>
</div>
</body></html>
"""

# --- ASH (Confex) ---
ASH_PROGRAM_HTML = """\
<html><body>
<div class="item" data-type="Late-Breaking Abstract Session">
  <span class="title">LBA Session: Novel Therapies in AML</span>
  <span class="time" data-start="2026-12-07T08:00:00" data-end="2026-12-07T09:30:00"></span>
  <span class="location">San Diego Convention Center Hall H</span>
</div>
<div class="item" data-type="Oral Abstract Session">
  <span class="title">Oral: CAR-T in DLBCL NCT07777777</span>
  <span class="time" data-start="2026-12-08T10:00:00" data-end="2026-12-08T11:30:00"></span>
  <span class="location">Room 6A</span>
</div>
</body></html>
"""

ASH_ABSTRACTS_HTML = """\
<html><body>
<div class="abstract" data-number="LBA-1" data-type="oral">
  <span class="title">Blinatumomab Frontline in Ph-Neg ALL NCT07777777</span>
  <span class="author">Thompson, David</span>
  <span class="affiliation">Fred Hutch</span>
</div>
<div class="abstract" data-number="812" data-type="poster">
  <span class="title">Factor VIII Gene Therapy in Hemophilia A</span>
  <span class="author">Kim, Soo-Yeon</span>
  <span class="affiliation">Seoul National University</span>
</div>
</body></html>
"""

# --- SITC (Confex) ---
SITC_PROGRAM_HTML = """\
<html><body>
<div class="item" data-type="Symposium">
  <span class="title">Symposium: Checkpoint Combinations in Melanoma</span>
  <span class="time" data-start="2026-11-04T09:00:00" data-end="2026-11-04T10:30:00"></span>
  <span class="location">Houston GRB Room 350</span>
</div>
<div class="item" data-type="Poster Session">
  <span class="title">Poster: Bispecific Antibodies in Solid Tumors</span>
  <span class="time" data-start="2026-11-05T12:00:00" data-end="2026-11-05T14:00:00"></span>
  <span class="location">Exhibit Hall</span>
</div>
</body></html>
"""

SITC_ABSTRACTS_HTML = """\
<html><body>
<div class="abstract" data-number="O01" data-type="oral">
  <span class="title">TIL Therapy in Advanced Melanoma NCT08888888</span>
  <span class="author">Rosenberg, Steven</span>
  <span class="affiliation">NCI</span>
</div>
<div class="abstract" data-number="P205" data-type="poster">
  <span class="title">Anti-LAG3 + Anti-PD1 in MSI-H Tumors</span>
  <span class="author">Wang, Li</span>
  <span class="affiliation">Peking University</span>
</div>
</body></html>
"""

# --- SABCS ---
SABCS_PROGRAM_HTML = """\
<html><body>
<div class="program-item" data-type="Late-Breaking Clinical Trial" data-track="Triple Negative">
  <span class="program-item-title">LBCT: ADC in Triple-Negative Breast Cancer</span>
  <span class="program-item-time" data-start="2026-12-10T08:00:00" data-end="2026-12-10T09:00:00"></span>
  <span class="program-item-location">Lila Cockrell Theatre</span>
</div>
<div class="program-item" data-type="Poster Spotlight" data-track="HR+">
  <span class="program-item-title">Poster Spotlight: CDK4/6 Inhibitor Sequencing</span>
  <span class="program-item-time" data-start="2026-12-11T10:00:00" data-end="2026-12-11T11:00:00"></span>
  <span class="program-item-location">Stars at Night Ballroom</span>
</div>
</body></html>
"""

SABCS_ABSTRACTS_HTML = """\
<html><body>
<div class="abstract-card" data-code="LB01" data-type="oral">
  <span class="abstract-card-title">Trastuzumab Deruxtecan in HER2-Low BC NCT09999999</span>
  <span class="abstract-card-presenter">Modi, Shanu</span>
  <span class="abstract-card-affiliation">MSKCC</span>
  <span class="abstract-card-sponsor">Daiichi Sankyo</span>
</div>
<div class="abstract-card" data-code="P3-05-08" data-type="poster">
  <span class="abstract-card-title">AI Pathology for TNBC Subtyping</span>
  <span class="abstract-card-presenter">Jones, Rebecca</span>
  <span class="abstract-card-affiliation">UCSF</span>
</div>
</body></html>
"""

# --- AAN ---
AAN_PROGRAM_HTML = """\
<html><body>
<div class="itinerary-item" data-type="Plenary Session" data-track="Neurodegeneration">
  <span class="itinerary-item-title">Plenary: Anti-Amyloid Therapies Update</span>
  <span class="itinerary-item-time" data-start="2026-04-06T08:00:00" data-end="2026-04-06T09:30:00"></span>
  <span class="itinerary-item-location">Ballroom A</span>
</div>
<div class="itinerary-item" data-type="Oral Presentation" data-track="Epilepsy">
  <span class="itinerary-item-title">Oral: Novel ASM Mechanisms NCT10101010</span>
  <span class="itinerary-item-time" data-start="2026-04-07T14:00:00" data-end="2026-04-07T15:30:00"></span>
  <span class="itinerary-item-location">Room 302</span>
</div>
</body></html>
"""

AAN_ABSTRACTS_HTML = """\
<html><body>
<div class="submission-item" data-code="LBN001" data-type="oral">
  <span class="submission-item-title">Lecanemab 18-Month Data in Early AD NCT10101010</span>
  <span class="submission-item-presenter">van Dyck, Christopher</span>
  <span class="submission-item-affiliation">Yale</span>
  <span class="submission-item-sponsor">Eisai</span>
</div>
<div class="submission-item" data-code="P4.123" data-type="poster">
  <span class="submission-item-title">Gene Therapy in SMA Type 1: Long-Term Follow-Up</span>
  <span class="submission-item-presenter">Mercuri, Eugenio</span>
  <span class="submission-item-affiliation">Catholic University Rome</span>
</div>
</body></html>
"""

# --- ACR ---
ACR_PROGRAM_HTML = """\
<html><body>
<div class="session-entry" data-type="Late-Breaking Abstract Session" data-track="RA">
  <span class="session-entry-title">LBA Session: JAK Inhibitor Safety Update</span>
  <span class="session-entry-time" data-start="2026-11-16T08:00:00" data-end="2026-11-16T09:30:00"></span>
  <span class="session-entry-location">Washington Convention Center Hall D</span>
</div>
<div class="session-entry" data-type="Oral Abstract Session" data-track="Lupus">
  <span class="session-entry-title">Oral: Novel Targets in SLE</span>
  <span class="session-entry-time" data-start="2026-11-17T10:00:00" data-end="2026-11-17T11:30:00"></span>
  <span class="session-entry-location">Room 145</span>
</div>
</body></html>
"""

ACR_ABSTRACTS_HTML = """\
<html><body>
<article class="abstract-entry" data-code="LB0001" data-type="oral">
  <span class="abstract-entry-title">Deucravacitinib in PsA: 52-Week Results NCT11223344</span>
  <span class="abstract-entry-presenter">Mease, Philip</span>
  <span class="abstract-entry-affiliation">Swedish Medical Center</span>
  <span class="abstract-entry-sponsor">Bristol Myers Squibb</span>
</article>
<article class="abstract-entry" data-code="2145" data-type="poster">
  <span class="abstract-entry-title">Anifrolumab in Active Lupus Nephritis</span>
  <span class="abstract-entry-presenter">Furie, Richard</span>
  <span class="abstract-entry-affiliation">Northwell Health</span>
</article>
</body></html>
"""


# ---------------------------------------------------------------------------
# New adapter test classes
# ---------------------------------------------------------------------------


class TestAllConferenceSlugsConstant:
    """Verify ALL_CONFERENCE_SLUGS and registry are in sync."""

    def test_all_slugs_registered(self):
        for slug in ALL_CONFERENCE_SLUGS:
            assert slug in CONFERENCE_ADAPTERS, f"Slug '{slug}' missing from CONFERENCE_ADAPTERS"

    def test_registry_contains_only_known_slugs(self):
        for slug in CONFERENCE_ADAPTERS:
            assert slug in ALL_CONFERENCE_SLUGS, f"Registry slug '{slug}' not in ALL_CONFERENCE_SLUGS"


class TestEsmoAdapter:
    """ESMO adapter tests."""

    def test_esmo_sessions(self):
        adapter = EsmoAdapter()
        sessions = adapter.parse_sessions(ESMO_PROGRAM_HTML, 2026)
        assert len(sessions) == 2
        types = {s["session_type"] for s in sessions}
        assert "lbct" in types  # Late-Breaking Abstract

    def test_esmo_abstracts(self):
        adapter = EsmoAdapter()
        abstracts = adapter.parse_abstracts(ESMO_ABSTRACTS_HTML, 2026)
        assert len(abstracts) == 2
        codes = {a["abstract_code"] for a in abstracts}
        assert "LBA1" in codes
        assert "450P" in codes
        lba = [a for a in abstracts if a["abstract_code"] == "LBA1"][0]
        assert lba["presentation_type"] == "lbct"

    def test_esmo_endpoints(self):
        adapter = EsmoAdapter()
        eps = adapter.discover_endpoints(2026)
        assert "2026" in eps["program"]
        assert "2026" in eps["abstracts"]


class TestAacrAdapter:
    """AACR adapter tests."""

    def test_aacr_sessions(self):
        adapter = AacrAdapter()
        sessions = adapter.parse_sessions(AACR_PROGRAM_HTML, 2026)
        assert len(sessions) == 2
        types = {s["session_type"] for s in sessions}
        assert "plenary" in types
        assert "oral" in types

    def test_aacr_abstracts(self):
        adapter = AacrAdapter()
        abstracts = adapter.parse_abstracts(AACR_ABSTRACTS_HTML, 2026)
        assert len(abstracts) == 2
        lb = [a for a in abstracts if a["abstract_code"] == "LB001"][0]
        assert lb["presentation_type"] == "lbct"

    def test_aacr_endpoints(self):
        adapter = AacrAdapter()
        eps = adapter.discover_endpoints(2026)
        assert "2026" in eps["program"]
        assert "aacr.org" in eps["program"]


class TestAshAdapter:
    """ASH adapter tests (Confex mixin)."""

    def test_ash_sessions(self):
        adapter = AshAdapter()
        sessions = adapter.parse_sessions(ASH_PROGRAM_HTML, 2026)
        assert len(sessions) == 2
        types = {s["session_type"] for s in sessions}
        assert "lbct" in types
        assert "oral" in types

    def test_ash_abstracts(self):
        adapter = AshAdapter()
        abstracts = adapter.parse_abstracts(ASH_ABSTRACTS_HTML, 2026)
        assert len(abstracts) == 2
        lba = [a for a in abstracts if a["abstract_code"] == "LBA-1"][0]
        assert lba["presentation_type"] == "lbct"
        poster = [a for a in abstracts if a["abstract_code"] == "812"][0]
        assert poster["presentation_type"] == "poster"

    def test_ash_endpoints(self):
        adapter = AshAdapter()
        eps = adapter.discover_endpoints(2026)
        assert "confex.com" in eps["program"]
        assert "2026" in eps["abstracts"]


class TestSitcAdapter:
    """SITC adapter tests (Confex mixin)."""

    def test_sitc_sessions(self):
        adapter = SitcAdapter()
        sessions = adapter.parse_sessions(SITC_PROGRAM_HTML, 2026)
        assert len(sessions) == 2
        types = {s["session_type"] for s in sessions}
        assert "symposium" in types
        assert "poster" in types

    def test_sitc_abstracts(self):
        adapter = SitcAdapter()
        abstracts = adapter.parse_abstracts(SITC_ABSTRACTS_HTML, 2026)
        assert len(abstracts) == 2
        oral = [a for a in abstracts if a["abstract_code"] == "O01"][0]
        assert oral["presentation_type"] == "oral"

    def test_sitc_endpoints(self):
        adapter = SitcAdapter()
        eps = adapter.discover_endpoints(2026)
        assert "confex.com" in eps["program"]
        assert "sitcancer" in eps["abstracts"]


class TestSabcsAdapter:
    """SABCS adapter tests."""

    def test_sabcs_sessions(self):
        adapter = SabcsAdapter()
        sessions = adapter.parse_sessions(SABCS_PROGRAM_HTML, 2026)
        assert len(sessions) == 2
        types = {s["session_type"] for s in sessions}
        assert "lbct" in types  # Late-Breaking Clinical Trial

    def test_sabcs_abstracts(self):
        adapter = SabcsAdapter()
        abstracts = adapter.parse_abstracts(SABCS_ABSTRACTS_HTML, 2026)
        assert len(abstracts) == 2
        lb = [a for a in abstracts if a["abstract_code"] == "LB01"][0]
        assert lb["presentation_type"] == "lbct"

    def test_sabcs_endpoints(self):
        adapter = SabcsAdapter()
        eps = adapter.discover_endpoints(2026)
        assert "2026" in eps["program"]
        assert "sabcs.org" in eps["program"]


class TestAanAdapter:
    """AAN adapter tests."""

    def test_aan_sessions(self):
        adapter = AanAdapter()
        sessions = adapter.parse_sessions(AAN_PROGRAM_HTML, 2026)
        assert len(sessions) == 2
        types = {s["session_type"] for s in sessions}
        assert "plenary" in types
        assert "oral" in types

    def test_aan_abstracts(self):
        adapter = AanAdapter()
        abstracts = adapter.parse_abstracts(AAN_ABSTRACTS_HTML, 2026)
        assert len(abstracts) == 2
        lb = [a for a in abstracts if a["abstract_code"] == "LBN001"][0]
        assert lb["presentation_type"] == "lbct"  # LB prefix

    def test_aan_endpoints(self):
        adapter = AanAdapter()
        eps = adapter.discover_endpoints(2026)
        assert "2026" in eps["program"]
        assert "mirasmart.com" in eps["program"]


class TestAcrAdapter:
    """ACR adapter tests."""

    def test_acr_sessions(self):
        adapter = AcrAdapter()
        sessions = adapter.parse_sessions(ACR_PROGRAM_HTML, 2026)
        assert len(sessions) == 2
        types = {s["session_type"] for s in sessions}
        assert "lbct" in types
        assert "oral" in types

    def test_acr_abstracts(self):
        adapter = AcrAdapter()
        abstracts = adapter.parse_abstracts(ACR_ABSTRACTS_HTML, 2026)
        assert len(abstracts) == 2
        lb = [a for a in abstracts if a["abstract_code"] == "LB0001"][0]
        assert lb["presentation_type"] == "lbct"

    def test_acr_endpoints(self):
        adapter = AcrAdapter()
        eps = adapter.discover_endpoints(2026)
        assert "2026" in eps["program"]
        assert "acrabstracts.org" in eps["program"]


class TestMultiConferenceWarming:
    """Integration test: warm multiple conferences via mock fetch."""

    def test_multi_conference_warming(self, tmp_path):
        # Map slug → fixture pairs
        fixture_map = {
            "esmo": (ESMO_PROGRAM_HTML, ESMO_ABSTRACTS_HTML),
            "aacr": (AACR_PROGRAM_HTML, AACR_ABSTRACTS_HTML),
            "ash": (ASH_PROGRAM_HTML, ASH_ABSTRACTS_HTML),
        }

        def _multi_fetch(url: str) -> str:
            for slug, (prog, abst) in fixture_map.items():
                if slug in url.lower():
                    if "abstract" in url.lower() or "ALLABSTRACTS" in url:
                        return abst
                    return prog
            return ""

        with patch(
            "wake_robin_data_pipeline.collectors.conference_program_collector._fetch",
            side_effect=_multi_fetch,
        ):
            total_events = 0
            for slug in ("esmo", "aacr", "ash"):
                events = collect_conference_derived_events(
                    conference_slug=slug,
                    edition_year=2026,
                    as_of_date=AS_OF,
                    cache_dir=tmp_path,
                    product_ticker_map={},
                    company_ticker_map={},
                )
                total_events += len(events)

                # Verify separate cache dirs
                conf_dir = tmp_path / slug
                assert conf_dir.exists()
                assert (conf_dir / f"sessions_{AS_OF.isoformat()}.json").exists()
                assert (conf_dir / f"abstracts_{AS_OF.isoformat()}.json").exists()
                assert (conf_dir / f"derived_events_{AS_OF.isoformat()}.json").exists()

        # With empty ticker maps we get 0 derived events, but parsing ran fine
        # The point is that 3 separate cache dirs were created successfully
        assert (tmp_path / "esmo").exists()
        assert (tmp_path / "aacr").exists()
        assert (tmp_path / "ash").exists()
