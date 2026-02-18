#!/usr/bin/env python3
"""
Tests for event_ledger.py - Unified PIT Event Ledger

Tests cover:
- LedgerEntry normalization from each source type
- Deterministic event_id generation
- PIT filtering (disclosed_at gate)
- Nearest catalyst query (window-aware)
- Leakage audit reporting
- Strict/degrade CTGov modes
"""

import json
import pytest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from event_ledger import (
    LedgerEntry,
    LedgerConfig,
    build_event_ledger,
    compute_event_id,
    query_nearest_catalyst,
    leakage_audit,
    write_ledger_jsonl,
    _load_ctgov_events,
    _load_sec_8k_events,
    _load_pdufa_events,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def tmp_cache(tmp_path):
    """Create a temporary cache structure with sample data."""
    ctgov_dir = tmp_path / "cache" / "ctgov"
    ctgov_dir.mkdir(parents=True)
    sec_dir = tmp_path / "cache" / "sec" / "8k_catalysts"
    sec_dir.mkdir(parents=True)
    fda_dir = tmp_path / "cache" / "fda"
    fda_dir.mkdir(parents=True)
    data_dir = tmp_path / "production_data"
    data_dir.mkdir(parents=True)

    return {
        "root": tmp_path,
        "ctgov": ctgov_dir,
        "sec": sec_dir,
        "fda": fda_dir,
        "data": data_dir,
    }


def _make_config(tmp_cache):
    return LedgerConfig(
        ctgov_cache_dir=tmp_cache["ctgov"],
        sec_cache_dir=tmp_cache["sec"],
        fda_cache_dir=tmp_cache["fda"],
        data_dir=tmp_cache["data"],
        strict_ctgov=False,
    )


# =============================================================================
# TestLedgerBuilder
# =============================================================================

class TestLedgerBuilder:
    """Tests for building ledger from cache sources."""

    def test_sec_8k_normalization(self, tmp_cache):
        """SEC 8-K cache event -> LedgerEntry with correct source, event_id,
        disclosed_at, date_precision."""
        events = [
            {
                "ticker": "ACAD",
                "event_type": "DATA_READOUT",
                "event_date": "2026-06-15",
                "event_date_end": None,
                "date_precision": "DAY",
                "event_name": "8-K: Phase 3 data readout",
                "confidence": "HIGH",
                "source": "SEC_8K_FILING",
                "disclosed_at": "2026-02-10",
                "tags": ["sec_8k"],
            }
        ]
        cache_path = tmp_cache["sec"] / "8k_catalysts_2026-02-17.json"
        cache_path.write_text(json.dumps(events))

        entries = _load_sec_8k_events(date(2026, 2, 17), tmp_cache["sec"])

        assert len(entries) == 1
        e = entries[0]
        assert e.source == "SEC_8K"
        assert e.ticker == "ACAD"
        assert e.event_type == "DATA_READOUT"
        assert e.event_date == "2026-06-15"
        assert e.date_precision == "DAY"
        assert e.disclosed_at == "2026-02-10"
        assert e.confidence == "HIGH"
        assert len(e.event_id) == 16  # SHA-1 truncated

    def test_ctgov_pcd_extraction(self, tmp_cache):
        """Trial record with PCD -> CLINICAL_PCD entry, disclosed_at=last_update_posted."""
        records = [
            {
                "nct_id": "NCT12345678",
                "ticker": "MRNA",
                "study_type": "Interventional",
                "phase": "Phase 3",
                "overall_status": "Recruiting",
                "primary_completion_date": "2026-09-01",
                "completion_date": "2027-01-15",
                "last_update_posted": "2026-02-05",
            }
        ]
        pit_path = tmp_cache["ctgov"] / "trial_records_2026-02-17.json"
        pit_path.write_text(json.dumps(records))

        entries = _load_ctgov_events(
            date(2026, 2, 17), tmp_cache["ctgov"], strict=False, data_dir=tmp_cache["data"],
        )

        # Should produce both PCD and CD entries
        pcd_entries = [e for e in entries if e.event_type == "CLINICAL_PCD"]
        cd_entries = [e for e in entries if e.event_type == "CLINICAL_CD"]

        assert len(pcd_entries) == 1
        assert len(cd_entries) == 1

        pcd = pcd_entries[0]
        assert pcd.source == "CTGOV"
        assert pcd.source_uid == "NCT12345678"
        assert pcd.disclosed_at == "2026-02-05"
        assert pcd.event_date == "2026-09-01"
        assert pcd.confidence == "HIGH"

    def test_pdufa_missing_disclosed_at(self, tmp_cache):
        """PDUFA entry without curated_disclosed_at -> confidence=LOW, tagged."""
        pdufa = [
            {
                "ticker": "ACAD",
                "drug_name": "Pimavanserin",
                "indication": "MDD",
                "pdufa_date": "2026-04-03",
                "submission_type": "sNDA",
                "confidence": "confirmed",
                "source": "company_guidance",
                "curated_disclosed_at": None,
            }
        ]
        pdufa_path = tmp_cache["data"] / "pdufa_dates.json"
        pdufa_path.write_text(json.dumps(pdufa))

        entries = _load_pdufa_events(tmp_cache["data"])

        assert len(entries) == 1
        e = entries[0]
        assert e.confidence == "LOW"
        assert "missing_disclosed_at" in e.tags
        assert e.source == "PDUFA_MANUAL"

    def test_pit_filter_excludes_future(self, tmp_cache):
        """Entry with disclosed_at > as_of_date excluded from ledger output."""
        events = [
            {
                "ticker": "ACAD",
                "event_type": "DATA_READOUT",
                "event_date": "2026-06-15",
                "event_date_end": None,
                "date_precision": "DAY",
                "event_name": "future filing",
                "confidence": "HIGH",
                "source": "SEC_8K_FILING",
                "disclosed_at": "2026-03-01",  # after as_of_date of 2026-02-17
                "tags": [],
            }
        ]
        cache_path = tmp_cache["sec"] / "8k_catalysts_2026-02-17.json"
        cache_path.write_text(json.dumps(events))

        # Empty PDUFA, empty trial_records
        (tmp_cache["data"] / "pdufa_dates.json").write_text("[]")
        (tmp_cache["data"] / "trial_records.json").write_text("[]")

        config = _make_config(tmp_cache)
        ledger = build_event_ledger(date(2026, 2, 17), config)

        # The event disclosed on 2026-03-01 should be excluded
        assert len(ledger) == 0

    def test_deterministic_event_id(self):
        """Same inputs -> same event_id; different value -> different event_id."""
        base_args = dict(
            source="SEC_8K", source_uid="ACAD_DATA_READOUT_2026-02-10",
            ticker="ACAD", event_type="DATA_READOUT",
            field_changed="event_date", value="2026-06-15",
            event_date="2026-06-15", event_date_end=None,
            disclosed_at="2026-02-10", extractor_version="abc123",
        )

        id1 = compute_event_id(**base_args)
        id2 = compute_event_id(**base_args)
        assert id1 == id2

        # Different value -> different id
        modified = {**base_args, "value": "2026-07-01"}
        id3 = compute_event_id(**modified)
        assert id3 != id1


# =============================================================================
# TestQueryNearestCatalyst
# =============================================================================

class TestQueryNearestCatalyst:
    """Tests for window-aware nearest catalyst query."""

    def _make_entry(self, ticker, event_date, event_date_end=None,
                    date_precision="DAY", confidence="HIGH",
                    disclosed_at="2026-01-01", event_type="DATA_READOUT"):
        eid = compute_event_id(
            "TEST", f"{ticker}_test", ticker, event_type,
            "event_date", event_date or "", event_date, event_date_end,
            disclosed_at, "test",
        )
        return LedgerEntry(
            event_id=eid, ticker=ticker, event_type=event_type,
            event_date=event_date, event_date_end=event_date_end,
            date_precision=date_precision, disclosed_at=disclosed_at,
            source="TEST", source_uid=f"{ticker}_test", confidence=confidence,
            extractor_version="test", event_name="test event",
            field_changed="event_date", value=event_date or "",
        )

    def test_day_precision_nearest(self):
        """DAY event 30d away wins over DAY event 60d away."""
        as_of = date(2026, 3, 1)
        e30 = self._make_entry("ACAD", "2026-03-31")  # 30 days
        e60 = self._make_entry("ACAD", "2026-04-30")  # 60 days

        result = query_nearest_catalyst([e30, e60], "ACAD", as_of)

        assert result is not None
        assert result["days_to_catalyst"] == 30
        assert result["event_date"] == "2026-03-31"

    def test_quarter_window_still_upcoming(self):
        """QUARTER event (Q2=Apr-Jun) queried in May: event_date_end=Jun 30 -> still upcoming."""
        as_of = date(2026, 5, 15)
        # Q2 event: start=Apr 1, end=Jun 30
        e = self._make_entry(
            "MRNA", "2026-04-01", event_date_end="2026-06-30",
            date_precision="QUARTER",
        )

        result = query_nearest_catalyst([e], "MRNA", as_of)

        assert result is not None
        # Days to end of Q2 from May 15
        assert result["days_to_catalyst"] == (date(2026, 6, 30) - as_of).days

    def test_quarter_window_stale_exclusion(self):
        """Filing in July referencing Q2 (event_date_end=Jun 30): excluded."""
        as_of = date(2026, 7, 15)
        e = self._make_entry(
            "MRNA", "2026-04-01", event_date_end="2026-06-30",
            date_precision="QUARTER",
        )

        result = query_nearest_catalyst([e], "MRNA", as_of)

        assert result is None  # Window has passed

    def test_low_confidence_excluded(self):
        """LOW confidence event skipped even if nearest."""
        as_of = date(2026, 3, 1)
        e_low = self._make_entry("ACAD", "2026-03-15", confidence="LOW")
        e_high = self._make_entry("ACAD", "2026-04-15", confidence="HIGH")

        result = query_nearest_catalyst([e_low, e_high], "ACAD", as_of)

        assert result is not None
        # Should skip the LOW confidence event and return the HIGH one
        assert result["event_date"] == "2026-04-15"
        assert result["days_to_catalyst"] == 45


# =============================================================================
# TestLeakageAudit
# =============================================================================

class TestLeakageAudit:
    """Tests for PIT leakage audit reporting."""

    def _make_entry(self, ticker, disclosed_at, event_date="2026-06-01",
                    source="TEST", confidence="HIGH", tags=()):
        eid = compute_event_id(
            source, f"{ticker}_test", ticker, "DATA_READOUT",
            "event_date", event_date, event_date, None,
            disclosed_at, "test",
        )
        return LedgerEntry(
            event_id=eid, ticker=ticker, event_type="DATA_READOUT",
            event_date=event_date, event_date_end=None,
            date_precision="DAY", disclosed_at=disclosed_at,
            source=source, source_uid=f"{ticker}_test",
            confidence=confidence, extractor_version="test",
            event_name="test", field_changed="event_date",
            value=event_date, tags=tags,
        )

    def test_audit_counts(self):
        """Correct n_included, n_excluded_future_disclosed, n_excluded_stale_window."""
        as_of = date(2026, 3, 1)

        # 2 entries: one included, one future-disclosed
        e_ok = self._make_entry("ACAD", "2026-02-15")
        e_future = self._make_entry("MRNA", "2026-04-01")

        all_raw = [e_ok, e_future]
        included = [e for e in all_raw if e.disclosed_at <= as_of.isoformat()]

        result = leakage_audit(included, as_of, all_raw_entries=all_raw)

        assert result["n_total_raw"] == 2
        assert result["n_included"] == 1
        assert result["n_excluded_future_disclosed"] == 1

    def test_pdufa_missing_flagged(self):
        """PDUFA with null disclosed_at counted in n_pdufa_missing_disclosed_at."""
        as_of = date(2026, 3, 1)
        e = self._make_entry(
            "ACAD", "2026-02-15",
            source="PDUFA_MANUAL",
            tags=("missing_disclosed_at",),
        )

        result = leakage_audit([e], as_of)

        assert result["n_pdufa_missing_disclosed_at"] == 1

    def test_excluded_by_source_breakdown(self):
        """Per-source exclusion counts are accurate."""
        as_of = date(2026, 3, 1)

        e_sec = self._make_entry("ACAD", "2026-04-01", source="SEC_8K")
        e_fda = self._make_entry("MRNA", "2026-05-01", source="FDA_ADCOM")
        e_ok = self._make_entry("VRTX", "2026-02-01", source="SEC_8K")

        all_raw = [e_sec, e_fda, e_ok]
        included = [e for e in all_raw if e.disclosed_at <= as_of.isoformat()]

        result = leakage_audit(included, as_of, all_raw_entries=all_raw)

        assert result["excluded_by_source"]["SEC_8K"] == 1
        assert result["excluded_by_source"]["FDA_ADCOM"] == 1
        assert result["n_excluded_future_disclosed"] == 2


# =============================================================================
# TestStrictCtgovMode
# =============================================================================

class TestStrictCtgovMode:
    """Tests for CTGov strict vs degrade modes."""

    def test_strict_mode_raises_on_missing_cache(self, tmp_cache):
        """strict=True + missing PIT cache -> FileNotFoundError."""
        # No cache file created
        with pytest.raises(FileNotFoundError, match="CTGov PIT cache missing"):
            _load_ctgov_events(
                date(2026, 2, 17),
                tmp_cache["ctgov"],
                strict=True,
                data_dir=tmp_cache["data"],
            )

    def test_degrade_mode_falls_back(self, tmp_cache):
        """strict=False + missing PIT cache -> uses fallback with inline PIT filter."""
        # Create unfiltered trial_records.json with one future record
        records = [
            {
                "nct_id": "NCT11111111",
                "ticker": "ACAD",
                "study_type": "Interventional",
                "phase": "Phase 2",
                "overall_status": "Recruiting",
                "primary_completion_date": "2026-09-01",
                "last_update_posted": "2026-02-10",  # before as_of
            },
            {
                "nct_id": "NCT22222222",
                "ticker": "MRNA",
                "study_type": "Interventional",
                "phase": "Phase 3",
                "overall_status": "Active",
                "primary_completion_date": "2026-12-01",
                "last_update_posted": "2026-03-01",  # AFTER as_of
            },
        ]
        fallback_path = tmp_cache["data"] / "trial_records.json"
        fallback_path.write_text(json.dumps(records))

        entries = _load_ctgov_events(
            date(2026, 2, 17),
            tmp_cache["ctgov"],
            strict=False,
            data_dir=tmp_cache["data"],
        )

        # Only the first record should pass PIT filter
        tickers = [e.ticker for e in entries]
        assert "ACAD" in tickers
        assert "MRNA" not in tickers


# =============================================================================
# TestWriteLedger
# =============================================================================

class TestWriteLedger:
    """Tests for JSONL ledger output."""

    def test_write_and_read_round_trip(self, tmp_path):
        """write_ledger_jsonl produces valid JSONL that can be parsed back."""
        eid = compute_event_id(
            "TEST", "uid", "ACAD", "DATA_READOUT",
            "event_date", "2026-06-01", "2026-06-01", None,
            "2026-02-01", "v1",
        )
        entry = LedgerEntry(
            event_id=eid, ticker="ACAD", event_type="DATA_READOUT",
            event_date="2026-06-01", event_date_end=None,
            date_precision="DAY", disclosed_at="2026-02-01",
            source="TEST", source_uid="uid", confidence="HIGH",
            extractor_version="v1", event_name="test",
            field_changed="event_date", value="2026-06-01",
        )

        out_path = tmp_path / "ledger" / "test.jsonl"
        write_ledger_jsonl([entry], out_path)

        assert out_path.exists()
        lines = out_path.read_text().strip().split("\n")
        assert len(lines) == 1

        parsed = json.loads(lines[0])
        assert parsed["ticker"] == "ACAD"
        assert parsed["event_id"] == eid
        assert parsed["source"] == "TEST"
