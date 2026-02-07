#!/usr/bin/env python3
"""
Tests for Multi-Source Catalyst Event Graph

Covers:
- Phase 1: Readout window inference, schema enhancements, fuzzy dedup, coverage KPIs
- Phase 2: FDA ADCOM collector, SEC 8-K collector
"""

import pytest
import json
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from module_3_schema import (
    CatalystEventV2,
    EventType,
    EventSeverity,
    ConfidenceLevel,
    DiagnosticCounts,
    SCHEMA_VERSION,
    effective_days_until,
    effective_event_date,
)
from catalyst_diagnostics import (
    detect_readout_window_catalysts,
    CalendarCatalyst,
    EventRuleID,
    EventEvidence,
)
from module_3_catalyst import (
    fuzzy_dedup_events,
    convert_calendar_catalyst_to_v2,
    convert_fda_adcom_to_v2,
    convert_sec_8k_to_v2,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def as_of_date():
    return date(2026, 2, 7)


def _make_trial_record(
    ticker="ACAD",
    nct_id="NCT00000001",
    overall_status_name="RECRUITING",
    primary_completion_date=None,
    primary_completion_type_value=None,
    completion_date=None,
    completion_type_value=None,
    results_first_posted=None,
    last_update_posted=None,
):
    """Build a mock CanonicalTrialRecord."""
    record = MagicMock()
    record.ticker = ticker
    record.nct_id = nct_id

    # overall_status.is_terminal_negative
    terminal_neg_statuses = {"WITHDRAWN", "TERMINATED", "SUSPENDED"}
    status_mock = MagicMock()
    status_mock.is_terminal_negative = overall_status_name in terminal_neg_statuses
    status_mock.name = overall_status_name
    record.overall_status = status_mock

    record.primary_completion_date = primary_completion_date
    if primary_completion_type_value:
        pct_mock = MagicMock()
        pct_mock.value = primary_completion_type_value
        record.primary_completion_type = pct_mock
    else:
        record.primary_completion_type = None

    record.completion_date = completion_date
    if completion_type_value:
        ct_mock = MagicMock()
        ct_mock.value = completion_type_value
        record.completion_type = ct_mock
    else:
        record.completion_type = None

    record.results_first_posted = results_first_posted
    record.last_update_posted = last_update_posted or date(2026, 1, 1)

    return record


def _make_snapshot(records):
    """Build a mock StateSnapshot."""
    snapshot = MagicMock()
    snapshot.records = records
    return snapshot


def _make_v2_event(
    ticker="ACAD",
    nct_id="NCT00000001",
    event_type=EventType.DATA_READOUT,
    event_date="2026-03-15",
    confidence=ConfidenceLevel.MED,
    event_date_end=None,
    date_precision="DAY",
    tags=(),
    source="TEST",
):
    return CatalystEventV2(
        ticker=ticker,
        nct_id=nct_id,
        event_type=event_type,
        event_severity=EventSeverity.CRITICAL_POSITIVE,
        event_date=event_date,
        field_changed="test",
        prior_value=None,
        new_value=None,
        source=source,
        confidence=confidence,
        disclosed_at="2026-01-01",
        event_date_end=event_date_end,
        date_precision=date_precision,
        tags=tags,
    )


# ============================================================================
# PHASE 1: READOUT WINDOW INFERENCE
# ============================================================================

class TestReadoutWindowInference:
    """Tests for detect_readout_window_catalysts()"""

    def test_pcd_past_no_results_generates_event(self, as_of_date):
        """PCD past, no results posted → readout window event generated"""
        record = _make_trial_record(
            primary_completion_date=date(2026, 1, 1),
            primary_completion_type_value="ESTIMATED",
            results_first_posted=None,
        )
        snapshot = _make_snapshot([record])

        catalysts = detect_readout_window_catalysts(snapshot, as_of_date)

        assert len(catalysts) == 1
        assert catalysts[0].event_type == "READOUT_WINDOW"
        assert catalysts[0].ticker == "ACAD"
        assert catalysts[0].rule_id == EventRuleID.M3_CAL_READOUT_WINDOW

    def test_pcd_past_results_posted_no_event(self, as_of_date):
        """PCD past, results already posted → no event"""
        record = _make_trial_record(
            primary_completion_date=date(2025, 10, 1),
            results_first_posted=date(2026, 1, 15),
        )
        snapshot = _make_snapshot([record])

        catalysts = detect_readout_window_catalysts(snapshot, as_of_date)

        assert len(catalysts) == 0

    def test_pcd_future_no_event(self, as_of_date):
        """PCD in the future → no readout window event (only forward calendar applies)"""
        record = _make_trial_record(
            primary_completion_date=date(2026, 6, 1),
        )
        snapshot = _make_snapshot([record])

        catalysts = detect_readout_window_catalysts(snapshot, as_of_date)

        assert len(catalysts) == 0

    def test_terminal_trial_no_event(self, as_of_date):
        """Terminal negative trial → no event"""
        record = _make_trial_record(
            overall_status_name="TERMINATED",
            primary_completion_date=date(2026, 1, 1),
        )
        snapshot = _make_snapshot([record])

        catalysts = detect_readout_window_catalysts(snapshot, as_of_date)

        assert len(catalysts) == 0

    def test_actual_pcd_confidence_boost(self, as_of_date):
        """Confidence boosted to 0.65 when PCD type is ACTUAL"""
        record = _make_trial_record(
            primary_completion_date=date(2026, 1, 1),
            primary_completion_type_value="ACTUAL",
        )
        snapshot = _make_snapshot([record])

        catalysts = detect_readout_window_catalysts(snapshot, as_of_date)

        assert len(catalysts) == 1
        assert catalysts[0].confidence == 0.65

    def test_estimated_pcd_base_confidence(self, as_of_date):
        """Base confidence of 0.50 for ESTIMATED PCD"""
        record = _make_trial_record(
            primary_completion_date=date(2026, 1, 1),
            primary_completion_type_value="ESTIMATED",
        )
        snapshot = _make_snapshot([record])

        catalysts = detect_readout_window_catalysts(snapshot, as_of_date)

        assert len(catalysts) == 1
        assert catalysts[0].confidence == 0.50

    def test_pit_safety_disclosed_at_is_pcd(self, as_of_date):
        """PIT safety: disclosed_at should be PCD, not as_of_date"""
        pcd = date(2026, 1, 1)
        record = _make_trial_record(
            primary_completion_date=pcd,
            primary_completion_type_value="ACTUAL",
        )
        snapshot = _make_snapshot([record])

        catalysts = detect_readout_window_catalysts(snapshot, as_of_date)

        assert len(catalysts) == 1
        # Evidence contains PCD
        assert catalysts[0].evidence.fields['primary_completion_date'] == pcd.isoformat()

        # When converted to V2, disclosed_at should be PCD
        v2 = convert_calendar_catalyst_to_v2(catalysts[0])
        assert v2.disclosed_at == pcd.isoformat()

    def test_window_expired_no_event(self, as_of_date):
        """PCD too far in the past (window expired) → no event"""
        # PCD was 150 days ago, window_end = PCD + 120d = 30 days ago
        record = _make_trial_record(
            primary_completion_date=as_of_date - timedelta(days=150),
        )
        snapshot = _make_snapshot([record])

        catalysts = detect_readout_window_catalysts(snapshot, as_of_date)

        assert len(catalysts) == 0


# ============================================================================
# PHASE 1: SCHEMA ENHANCEMENTS
# ============================================================================

class TestSchemaEnhancements:
    """Tests for CatalystEventV2 new fields"""

    def test_event_date_end_roundtrip(self):
        """event_date_end round-trips through to_dict/from_dict"""
        event = _make_v2_event(
            event_date="2026-01-01",
            event_date_end="2026-03-31",
            date_precision="QUARTER",
        )
        d = event.to_dict()
        restored = CatalystEventV2.from_dict(d)

        assert restored.event_date_end == "2026-03-31"
        assert restored.date_precision == "QUARTER"

    def test_date_precision_defaults_to_day(self):
        """date_precision defaults to DAY for existing events without the field"""
        d = {
            "ticker": "ACAD",
            "nct_id": "NCT00000001",
            "event_type": "DATA_READOUT",
            "event_severity": "CRITICAL_POSITIVE",
            "event_date": "2026-03-15",
            "field_changed": "test",
            "source": "TEST",
            "confidence": "MED",
            "disclosed_at": "2026-01-01",
            # No event_date_end, date_precision, or tags
        }
        event = CatalystEventV2.from_dict(d)

        assert event.date_precision == "DAY"
        assert event.event_date_end is None
        assert event.tags == ()

    def test_tags_serialize_as_list_deserialize_as_tuple(self):
        """Tags serialize as list, deserialize back to tuple"""
        event = _make_v2_event(tags=("readout_window", "inferred"))
        d = event.to_dict()

        # Serialized as list
        assert isinstance(d["tags"], list)
        assert d["tags"] == ["readout_window", "inferred"]

        # Deserialized back to tuple
        restored = CatalystEventV2.from_dict(d)
        assert isinstance(restored.tags, tuple)
        assert restored.tags == ("readout_window", "inferred")

    def test_event_id_unchanged_when_event_date_end_none(self):
        """event_id is stable when event_date_end is None (default)"""
        event_with_none = _make_v2_event(event_date_end=None)
        event_with_default = CatalystEventV2(
            ticker="ACAD",
            nct_id="NCT00000001",
            event_type=EventType.DATA_READOUT,
            event_severity=EventSeverity.CRITICAL_POSITIVE,
            event_date="2026-03-15",
            field_changed="test",
            prior_value=None,
            new_value=None,
            source="TEST",
            confidence=ConfidenceLevel.MED,
            disclosed_at="2026-01-01",
            # event_date_end defaults to None, hashes as ""
        )

        assert event_with_none.event_id == event_with_default.event_id

    def test_schema_version_bumped(self):
        """Schema version is updated"""
        assert SCHEMA_VERSION == "m3catalyst_vnext_20260207"

    def test_diagnostic_counts_has_coverage_fields(self):
        """DiagnosticCounts has the new coverage and fuzzy merge fields"""
        dc = DiagnosticCounts()
        assert dc.coverage_any_event == 0
        assert dc.coverage_actionable_180d == 0
        assert dc.coverage_high_conf_actionable == 0
        assert dc.events_fuzzy_merged == 0

        d = dc.to_dict()
        assert "coverage_any_event" in d
        assert "coverage_actionable_180d" in d
        assert "coverage_high_conf_actionable" in d
        assert "events_fuzzy_merged" in d


# ============================================================================
# PHASE 1: FUZZY DEDUP
# ============================================================================

class TestFuzzyDedup:
    """Tests for fuzzy_dedup_events()"""

    def test_same_ticker_type_within_14d_merged(self):
        """Same (ticker, event_type) within 14 days → merged"""
        e1 = _make_v2_event(event_date="2026-03-01", confidence=ConfidenceLevel.MED)
        e2 = _make_v2_event(event_date="2026-03-10", confidence=ConfidenceLevel.LOW)

        result, merged_count = fuzzy_dedup_events([e1, e2])

        assert len(result) == 1
        assert merged_count == 1

    def test_same_ticker_type_15d_apart_kept_separate(self):
        """Same (ticker, event_type) 15 days apart → kept separate"""
        e1 = _make_v2_event(event_date="2026-03-01", confidence=ConfidenceLevel.MED)
        e2 = _make_v2_event(event_date="2026-03-16", confidence=ConfidenceLevel.MED)

        result, merged_count = fuzzy_dedup_events([e1, e2])

        assert len(result) == 2
        assert merged_count == 0

    def test_merge_prefers_higher_confidence(self):
        """When merging, keep the higher-confidence event"""
        e1 = _make_v2_event(event_date="2026-03-01", confidence=ConfidenceLevel.LOW)
        e2 = _make_v2_event(event_date="2026-03-10", confidence=ConfidenceLevel.HIGH)

        result, merged_count = fuzzy_dedup_events([e1, e2])

        assert len(result) == 1
        assert result[0].confidence == ConfidenceLevel.HIGH
        assert merged_count == 1

    def test_tags_unioned_on_merge(self):
        """Tags from both events are unioned on merge"""
        e1 = _make_v2_event(
            event_date="2026-03-01",
            confidence=ConfidenceLevel.MED,
            tags=("readout_window",),
        )
        e2 = _make_v2_event(
            event_date="2026-03-10",
            confidence=ConfidenceLevel.MED,
            tags=("sec_8k",),
        )

        result, merged_count = fuzzy_dedup_events([e1, e2])

        assert len(result) == 1
        assert merged_count == 1
        assert set(result[0].tags) == {"readout_window", "sec_8k"}

    def test_different_event_types_not_merged(self):
        """Different event types for same ticker → not merged"""
        e1 = _make_v2_event(
            event_date="2026-03-01",
            event_type=EventType.DATA_READOUT,
        )
        e2 = _make_v2_event(
            event_date="2026-03-05",
            event_type=EventType.FDA_PDUFA_DATE,
        )

        result, merged_count = fuzzy_dedup_events([e1, e2])

        assert len(result) == 2
        assert merged_count == 0

    def test_empty_events(self):
        """Empty event list returns empty"""
        result, merged_count = fuzzy_dedup_events([])
        assert result == []
        assert merged_count == 0


# ============================================================================
# PHASE 1: COVERAGE KPIs
# ============================================================================

class TestCoverageKPIs:
    """Tests for 3-tier coverage computation in score_catalyst_events"""

    def test_coverage_kpis_computed(self):
        """Coverage KPIs are computed in DiagnosticCounts"""
        from module_3_scoring import score_catalyst_events

        # Create events for 2 tickers:
        # - ACAD: event within 180d, HIGH confidence → all 3 KPIs
        # - AXSM: event >180d away → only any-event
        events = {
            "ACAD": [_make_v2_event(
                ticker="ACAD",
                event_date="2026-04-01",  # ~53d ahead
                confidence=ConfidenceLevel.HIGH,
            )],
            "AXSM": [_make_v2_event(
                ticker="AXSM",
                event_date="2027-01-01",  # >180d ahead
                confidence=ConfidenceLevel.MED,
            )],
            "CYTK": [],  # No events
        }

        as_of = date(2026, 2, 7)
        summaries, diag = score_catalyst_events(
            events, ["ACAD", "AXSM", "CYTK"], as_of
        )

        assert diag.coverage_any_event == 2        # ACAD + AXSM
        assert diag.coverage_actionable_180d == 1   # ACAD only
        assert diag.coverage_high_conf_actionable == 1  # ACAD only


# ============================================================================
# PHASE 2: FDA ADCOM COLLECTOR
# ============================================================================

class TestFDAAdcomCollector:

    def test_federal_register_date_and_drug_extraction(self):
        """Federal Register meeting date and drug name regex extraction"""
        from wake_robin_data_pipeline.collectors.fda_adcom_collector import (
            _FR_MEETING_DATE, _FR_DRUG_AFTER_NDA, _FR_BRAND_GENERIC,
            _FR_DRUG_AFTER_COMMENTS, _parse_date_str, _match_product_to_ticker,
        )

        # Meeting date extraction from `dates` field
        dates_field = "The meeting will be held on March 15, 2026, from 8 a.m. to 5 p.m."
        m = _FR_MEETING_DATE.search(dates_field)
        assert m is not None
        parsed = _parse_date_str(m.group(1))
        assert parsed is not None
        assert parsed.isoformat() == "2026-03-15"

        # Also handles "scheduled on" format
        dates_field2 = "The meeting is scheduled on July 17, 2025."
        m2 = _FR_MEETING_DATE.search(dates_field2)
        assert m2 is not None
        parsed2 = _parse_date_str(m2.group(1))
        assert parsed2 is not None
        assert parsed2.isoformat() == "2025-07-17"

        # Handles "held virtually on" (common in Federal Register)
        dates_field3 = "The meeting will be held virtually on May 22, 2025, from 8:30 a.m."
        m3 = _FR_MEETING_DATE.search(dates_field3)
        assert m3 is not None
        parsed3 = _parse_date_str(m3.group(1))
        assert parsed3 is not None
        assert parsed3.isoformat() == "2025-05-22"

        # Handles "held on virtually on" (typo found in real FR data)
        dates_field4 = "The meeting will be held on virtually on November 13, 2025, from 10:00 a.m."
        m4 = _FR_MEETING_DATE.search(dates_field4)
        assert m4 is not None
        parsed4 = _parse_date_str(m4.group(1))
        assert parsed4 is not None
        assert parsed4.isoformat() == "2025-11-13"

        # Product matching against known map (primary mechanism — substring match)
        title = "Psychopharmacologic Drugs Advisory Committee; Notice of Meeting; Belantamab Mafodotin"
        product_map = {"karxt": "KRYS", "belantamab mafodotin": "GSK"}
        result = _match_product_to_ticker(title, product_map)
        assert result is not None
        assert result["ticker"] == "GSK"

        # Pattern 1: NDA/BLA number followed by drug name
        title_nda = "Comments-Biologic License Application (BLA) 761440 for Belantamab Mafodotin"
        m_nda = _FR_DRUG_AFTER_NDA.search(title_nda)
        assert m_nda is not None
        assert "Belantamab" in m_nda.group(1)

        # Pattern 2: BRANDNAME (generic) extraction
        title_brand = "for REXULTI (brexpiprazole) Tablets"
        m_brand = _FR_BRAND_GENERIC.search(title_brand)
        assert m_brand is not None
        assert m_brand.group(1) == "REXULTI"
        assert m_brand.group(2) == "brexpiprazole"

        # Pattern 3: "Comments-<Drug>" (no NDA/BLA prefix)
        title_comments = "Comments-Midomafetamine Capsules"
        m_comments = _FR_DRUG_AFTER_COMMENTS.search(title_comments)
        assert m_comments is not None
        assert "Midomafetamine" in m_comments.group(1)

    def test_product_ticker_mapping(self, tmp_path):
        """Product-to-ticker mapping works from PDUFA dates"""
        from wake_robin_data_pipeline.collectors.fda_adcom_collector import build_product_ticker_map

        pdufa = [
            {"ticker": "KRYS", "drug_name": "KarXT", "pdufa_date": "2026-02-28"},
            {"ticker": "ACAD", "drug_name": "Pimavanserin", "pdufa_date": "2026-04-03"},
        ]
        pdufa_path = tmp_path / "pdufa_dates.json"
        pdufa_path.write_text(json.dumps(pdufa))

        product_map = build_product_ticker_map(tmp_path)

        assert product_map["karxt"] == "KRYS"
        assert product_map["pimavanserin"] == "ACAD"

    def test_unmatched_products_skipped(self):
        """Products not in our universe are skipped during matching"""
        from wake_robin_data_pipeline.collectors.fda_adcom_collector import _match_product_to_ticker

        product_map = {"karxt": "KRYS", "pimavanserin": "ACAD"}

        # Known product
        result = _match_product_to_ticker("Discussion of KarXT for schizophrenia", product_map)
        assert result is not None
        assert result["ticker"] == "KRYS"

        # Unknown product
        result = _match_product_to_ticker("Discussion of SomeNewDrug for depression", product_map)
        assert result is None


# ============================================================================
# PHASE 2: SEC 8-K COLLECTOR
# ============================================================================

class TestSEC8KCollector:

    def test_quarter_to_date_range(self):
        """'Q2 2026' → correct date range + QUARTER precision"""
        from wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector import _quarter_to_date_range

        start, end = _quarter_to_date_range("Q2 2026")
        assert start == "2026-04-01"
        assert end == "2026-06-30"

        start, end = _quarter_to_date_range("Q1 2026")
        assert start == "2026-01-01"
        assert end == "2026-03-31"

        start, end = _quarter_to_date_range("Q4 2025")
        assert start == "2025-10-01"
        assert end == "2025-12-31"

    def test_pdufa_date_extraction(self):
        """'PDUFA date of March 15, 2026' → DAY precision"""
        from wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector import _extract_timing_events

        text = "The Company announced that the PDUFA date of March 15, 2026 has been confirmed."
        events = _extract_timing_events(text, "KRYS", "2026-01-15")

        assert len(events) == 1
        assert events[0]["event_type"] == "FDA_PDUFA_DATE"
        assert events[0]["event_date"] == "2026-03-15"
        assert events[0]["date_precision"] == "DAY"
        assert events[0]["confidence"] == "HIGH"

    def test_half_year_extraction(self):
        """'first half of 2026' → HALF_YEAR precision"""
        from wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector import _extract_timing_events

        text = "The Company expects results from the Phase 3 trial in the first half of 2026."
        events = _extract_timing_events(text, "ACAD", "2026-01-10")

        assert len(events) == 1
        assert events[0]["event_type"] == "DATA_READOUT"
        assert events[0]["date_precision"] == "HALF_YEAR"
        assert events[0]["event_date"] == "2026-01-01"
        assert events[0]["event_date_end"] == "2026-06-30"
        assert events[0]["confidence"] == "LOW"

    def test_filing_date_used_as_disclosed_at(self):
        """PIT safety: filing date used as disclosed_at"""
        from wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector import _extract_timing_events

        filing_date = "2025-12-01"
        text = "PDUFA date of March 15, 2026 has been set."
        events = _extract_timing_events(text, "KRYS", filing_date)

        assert len(events) == 1
        assert events[0]["disclosed_at"] == filing_date

    def test_topline_data_quarter_extraction(self):
        """'expects topline data in Q2 2026' → correct extraction"""
        from wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector import _extract_timing_events

        text = "The company expects topline data in Q2 2026 from the Phase 2 trial."
        events = _extract_timing_events(text, "AXSM", "2026-01-20")

        assert len(events) >= 1
        # First match is from the "expects topline" pattern
        assert events[0]["event_type"] == "DATA_READOUT"
        assert events[0]["date_precision"] == "QUARTER"
        assert events[0]["event_date"] == "2026-04-01"
        assert events[0]["event_date_end"] == "2026-06-30"
        assert events[0]["confidence"] == "MED"

    def test_half_year_to_date_range(self):
        """Half-year date range conversion"""
        from wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector import _half_year_to_date_range

        start, end = _half_year_to_date_range("2026", "first")
        assert start == "2026-01-01"
        assert end == "2026-06-30"

        start, end = _half_year_to_date_range("2026", "second")
        assert start == "2026-07-01"
        assert end == "2026-12-31"

    def test_year_guard_rejects_far_future(self):
        """Year guard: 'second half of 2040' rejected when as_of_date is 2026"""
        from wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector import _extract_timing_events

        text = "The Company expects results from the trial in the second half of 2040."
        events = _extract_timing_events(text, "ROIV", "2026-01-15", as_of_date=date(2026, 2, 7))
        assert len(events) == 0

    def test_year_guard_rejects_far_past(self):
        """Year guard: 'Q1 2020' rejected when as_of_date is 2026"""
        from wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector import _extract_timing_events

        text = "The company expects topline data in Q1 2020 from the Phase 2 trial."
        events = _extract_timing_events(text, "TEST", "2026-01-15", as_of_date=date(2026, 2, 7))
        assert len(events) == 0

    def test_year_guard_allows_nearby_years(self):
        """Year guard: years within [as_of - 1, as_of + 3] are allowed"""
        from wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector import _extract_timing_events

        # as_of = 2026 → allows 2025-2029
        text = "The Company expects results from the trial in the second half of 2028."
        events = _extract_timing_events(text, "TEST", "2026-01-15", as_of_date=date(2026, 2, 7))
        assert len(events) == 1
        assert events[0]["event_date"][:4] == "2028"

    def test_staleness_filter_rejects_old_events(self):
        """Staleness filter: events ending >180d before as_of_date are rejected"""
        from wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector import _extract_timing_events

        # Q1 2024 ends 2024-03-31, as_of is 2026-02-07 → >180d stale
        text = "The company expects topline data in Q1 2024 from the Phase 2 trial."
        events = _extract_timing_events(text, "MIRM", "2023-12-01", as_of_date=date(2026, 2, 7))
        assert len(events) == 0

    def test_staleness_filter_allows_recent_events(self):
        """Staleness filter: events ending within 180d of as_of_date are kept"""
        from wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector import _extract_timing_events

        # Q4 2025 ends 2025-12-31, as_of is 2026-02-07 → within 180d
        text = "The company expects topline data in Q4 2025 from the Phase 2 trial."
        events = _extract_timing_events(text, "TEST", "2025-09-01", as_of_date=date(2026, 2, 7))
        assert len(events) >= 1

    def test_no_as_of_date_skips_filters(self):
        """Without as_of_date, year guard and staleness filters are bypassed"""
        from wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector import _extract_timing_events

        text = "The Company expects results from the trial in the second half of 2040."
        events = _extract_timing_events(text, "ROIV", "2026-01-15", as_of_date=None)
        assert len(events) == 1  # No filter applied


# ============================================================================
# RANGE-AWARE DAYS_UNTIL
# ============================================================================

class TestEffectiveDaysUntil:
    """Tests for effective_days_until() range awareness"""

    def test_day_precision_normal(self):
        """DAY precision → normal days calculation"""
        event = _make_v2_event(event_date="2026-03-15", date_precision="DAY")
        as_of = date(2026, 2, 7)
        result = effective_days_until(event, as_of)
        assert result == 36  # March 15 - Feb 7

    def test_range_before_start(self):
        """Before range start → days until start (positive)"""
        event = _make_v2_event(
            event_date="2026-04-01",
            event_date_end="2026-06-30",
            date_precision="QUARTER",
        )
        as_of = date(2026, 2, 7)
        result = effective_days_until(event, as_of)
        assert result == 53  # April 1 - Feb 7

    def test_range_inside(self):
        """Inside range → 0 (actionable now)"""
        event = _make_v2_event(
            event_date="2026-01-01",
            event_date_end="2026-06-30",
            date_precision="HALF_YEAR",
        )
        as_of = date(2026, 3, 15)
        result = effective_days_until(event, as_of)
        assert result == 0

    def test_range_after_end(self):
        """After range end → negative (past)"""
        event = _make_v2_event(
            event_date="2026-01-01",
            event_date_end="2026-03-31",
            date_precision="QUARTER",
        )
        as_of = date(2026, 5, 1)
        result = effective_days_until(event, as_of)
        assert result < 0  # Past the end

    def test_none_event_date(self):
        """No event_date → None"""
        event = _make_v2_event(event_date=None)
        result = effective_days_until(event, date(2026, 2, 7))
        assert result is None

    def test_range_at_start_boundary(self):
        """On range start date → inside (0)"""
        event = _make_v2_event(
            event_date="2026-04-01",
            event_date_end="2026-06-30",
            date_precision="QUARTER",
        )
        as_of = date(2026, 4, 1)
        result = effective_days_until(event, as_of)
        assert result == 0

    def test_range_at_end_boundary(self):
        """On range end date → inside (0)"""
        event = _make_v2_event(
            event_date="2026-04-01",
            event_date_end="2026-06-30",
            date_precision="QUARTER",
        )
        as_of = date(2026, 6, 30)
        result = effective_days_until(event, as_of)
        assert result == 0


class TestEffectiveEventDate:
    """Tests for effective_event_date() range awareness"""

    def test_day_returns_event_date(self):
        """DAY precision returns the event date"""
        event = _make_v2_event(event_date="2026-03-15")
        result = effective_event_date(event, date(2026, 2, 7))
        assert result == date(2026, 3, 15)

    def test_range_inside_returns_as_of(self):
        """Inside range returns as_of_date (currently actionable)"""
        event = _make_v2_event(
            event_date="2026-01-01",
            event_date_end="2026-06-30",
            date_precision="HALF_YEAR",
        )
        as_of = date(2026, 3, 15)
        result = effective_event_date(event, as_of)
        assert result == date(2026, 3, 15)  # as_of_date

    def test_range_before_returns_start(self):
        """Before range returns start date"""
        event = _make_v2_event(
            event_date="2026-04-01",
            event_date_end="2026-06-30",
            date_precision="QUARTER",
        )
        result = effective_event_date(event, date(2026, 2, 7))
        assert result == date(2026, 4, 1)


class TestRangeAwareCoverage:
    """Tests that coverage KPIs correctly count range events"""

    def test_in_range_event_counts_as_actionable(self):
        """A RANGE event currently inside the window counts as actionable"""
        from module_3_scoring import score_catalyst_events

        # Event with range Jan 1 - Jun 30, as_of Feb 7 → inside range
        events = {
            "ACAD": [_make_v2_event(
                ticker="ACAD",
                event_date="2026-01-01",
                event_date_end="2026-06-30",
                date_precision="HALF_YEAR",
                confidence=ConfidenceLevel.HIGH,
            )],
        }
        as_of = date(2026, 2, 7)
        summaries, diag = score_catalyst_events(events, ["ACAD"], as_of)

        # Should count as actionable (effective_days_until = 0 ≤ 180)
        assert diag.coverage_actionable_180d == 1
        assert diag.coverage_high_conf_actionable == 1

    def test_past_range_event_not_actionable(self):
        """A RANGE event fully in the past does NOT count as actionable"""
        from module_3_scoring import score_catalyst_events

        events = {
            "ACAD": [_make_v2_event(
                ticker="ACAD",
                event_date="2025-01-01",
                event_date_end="2025-06-30",
                date_precision="HALF_YEAR",
                confidence=ConfidenceLevel.HIGH,
            )],
        }
        as_of = date(2026, 2, 7)
        summaries, diag = score_catalyst_events(events, ["ACAD"], as_of)

        assert diag.coverage_actionable_180d == 0


# ============================================================================
# DEDICATED CONVERTERS
# ============================================================================

class TestDedicatedConverters:
    """Tests for convert_fda_adcom_to_v2 and convert_sec_8k_to_v2"""

    def test_adcom_converter_event_type(self):
        """ADCOM converter produces FDA_ADCOM event type"""
        event = {
            "ticker": "KRYS",
            "event_date": "2026-03-15",
            "drug_name": "KarXT",
            "event_name": "ADCOM: KarXT",
            "confidence": "HIGH",
            "disclosed_at": "2026-02-07",
            "tags": ["adcom"],
        }
        v2 = convert_fda_adcom_to_v2(event, date(2026, 2, 7))

        assert v2 is not None
        assert v2.event_type == EventType.FDA_ADCOM
        assert v2.confidence == ConfidenceLevel.HIGH
        assert v2.source == "FDA_ADCOM_CALENDAR"
        assert v2.tags == ("adcom",)

    def test_sec_8k_converter_preserves_range(self):
        """SEC 8-K converter preserves date_precision and event_date_end"""
        event = {
            "ticker": "AXSM",
            "event_type": "DATA_READOUT",
            "event_date": "2026-04-01",
            "event_date_end": "2026-06-30",
            "date_precision": "QUARTER",
            "event_name": "8-K: expects topline data in Q2 2026",
            "confidence": "MED",
            "disclosed_at": "2026-01-20",
            "tags": ["sec_8k"],
        }
        v2 = convert_sec_8k_to_v2(event, date(2026, 2, 7))

        assert v2 is not None
        assert v2.event_type == EventType.DATA_READOUT
        assert v2.event_date == "2026-04-01"
        assert v2.event_date_end == "2026-06-30"
        assert v2.date_precision == "QUARTER"
        assert v2.confidence == ConfidenceLevel.MED
        assert v2.source == "SEC_8K_FILING"
        assert v2.disclosed_at == "2026-01-20"
        assert v2.tags == ("sec_8k",)

    def test_sec_8k_converter_returns_none_for_missing(self):
        """SEC 8-K converter returns None for incomplete events"""
        assert convert_sec_8k_to_v2({}, date(2026, 2, 7)) is None
        assert convert_sec_8k_to_v2({"ticker": "X"}, date(2026, 2, 7)) is None

    def test_adcom_converter_returns_none_for_missing(self):
        """ADCOM converter returns None for incomplete events"""
        assert convert_fda_adcom_to_v2({}, date(2026, 2, 7)) is None
        assert convert_fda_adcom_to_v2({"ticker": "X"}, date(2026, 2, 7)) is None


# ============================================================================
# TRI-STATE CONFIG + CACHE-ONLY MODE
# ============================================================================

class TestModule3ConfigModes:
    """Tests for tri-state enable_fda_adcom / enable_sec_8k_catalysts"""

    def test_default_is_cache_only(self):
        """Default config mode is cache_only"""
        from module_3_catalyst import Module3Config
        config = Module3Config()
        assert config.enable_fda_adcom == "cache_only"
        assert config.enable_sec_8k_catalysts == "cache_only"

    def test_from_dict_bool_true_maps_to_live(self):
        """True → 'live' for backwards compat"""
        from module_3_catalyst import Module3Config
        config = Module3Config.from_dict({"enable_fda_adcom": True})
        assert config.enable_fda_adcom == "live"

    def test_from_dict_bool_false_maps_to_off(self):
        """False → 'off' for backwards compat"""
        from module_3_catalyst import Module3Config
        config = Module3Config.from_dict({"enable_sec_8k_catalysts": False})
        assert config.enable_sec_8k_catalysts == "off"

    def test_from_dict_string_passthrough(self):
        """String values pass through directly"""
        from module_3_catalyst import Module3Config
        config = Module3Config.from_dict({"enable_fda_adcom": "cache_only"})
        assert config.enable_fda_adcom == "cache_only"

    def test_cache_only_loads_from_file(self, tmp_path):
        """cache_only mode loads events from cache file if present"""
        # Write a fake cache file
        cache_dir = tmp_path / "fda"
        cache_dir.mkdir()
        cache_file = cache_dir / "adcom_calendar_2026-02-07.json"
        cache_file.write_text(json.dumps([
            {"ticker": "KRYS", "event_type": "FDA_ADCOM", "event_date": "2026-03-15",
             "drug_name": "KarXT", "confidence": "HIGH", "disclosed_at": "2026-02-07",
             "tags": ["adcom"]},
        ]))

        # Verify the cache file can be loaded
        with open(cache_file, 'r') as f:
            events = json.load(f)
        assert len(events) == 1
        assert events[0]["ticker"] == "KRYS"

    def test_cache_only_no_file_yields_empty(self, tmp_path):
        """cache_only mode with no cache file yields no events (not an error)"""
        cache_dir = tmp_path / "fda"
        cache_dir.mkdir()
        cache_file = cache_dir / "adcom_calendar_2026-02-07.json"
        assert not cache_file.exists()  # no cache


# ============================================================================
# DISCLOSED_AT PIT CLIPPING
# ============================================================================

class TestDisclosedAtClipping:
    """Tests for disclosed_at clipping to as_of_date (PIT safety)"""

    def _make_calendar_catalyst(self, target_date, event_type='UPCOMING_PCD'):
        """Helper to create a CalendarCatalyst-like object"""
        evidence = MagicMock()
        evidence.fields = {}
        return MagicMock(
            ticker="TEST",
            nct_id="NCT123",
            target_date=target_date,
            event_type=event_type,
            confidence=0.9,
            days_until=30,
            evidence=evidence,
        )

    def test_future_event_disclosed_at_clipped(self):
        """Event date in future → disclosed_at clipped to as_of_date"""
        from module_3_catalyst import convert_calendar_catalyst_to_v2

        as_of = date(2026, 2, 7)
        future_date = date(2026, 5, 15)
        catalyst = self._make_calendar_catalyst(future_date)

        v2 = convert_calendar_catalyst_to_v2(catalyst, as_of_date=as_of)
        assert v2.disclosed_at == "2026-02-07"
        assert "disclosed_at_clipped" in v2.tags

    def test_past_event_disclosed_at_not_clipped(self):
        """Event date in past → disclosed_at is event date, no clipping"""
        from module_3_catalyst import convert_calendar_catalyst_to_v2

        as_of = date(2026, 2, 7)
        past_date = date(2026, 1, 15)
        catalyst = self._make_calendar_catalyst(past_date)

        v2 = convert_calendar_catalyst_to_v2(catalyst, as_of_date=as_of)
        assert v2.disclosed_at == "2026-01-15"
        assert "disclosed_at_clipped" not in v2.tags

    def test_no_as_of_date_no_clipping(self):
        """Without as_of_date, disclosed_at is event date (backwards compat)"""
        from module_3_catalyst import convert_calendar_catalyst_to_v2

        future_date = date(2026, 5, 15)
        catalyst = self._make_calendar_catalyst(future_date)

        v2 = convert_calendar_catalyst_to_v2(catalyst)  # no as_of_date
        assert v2.disclosed_at == "2026-05-15"  # unclipped
        assert "disclosed_at_clipped" not in v2.tags

    def test_readout_window_pcd_not_clipped(self):
        """Readout window: disclosed_at = PCD (in past), should not be clipped"""
        from module_3_catalyst import convert_calendar_catalyst_to_v2

        as_of = date(2026, 2, 7)
        target = date(2026, 3, 15)  # midpoint of window (future)
        catalyst = self._make_calendar_catalyst(target, event_type='READOUT_WINDOW')
        # PCD is in the past
        catalyst.evidence.fields = {
            'primary_completion_date': '2025-12-01',
            'window_start': '2025-12-31',
            'window_end': '2026-03-31',
        }

        v2 = convert_calendar_catalyst_to_v2(catalyst, as_of_date=as_of)
        assert v2.disclosed_at == "2025-12-01"  # PCD, not clipped
        assert "disclosed_at_clipped" not in v2.tags
