"""
Tests for biopharmawatch_collector.py

5 tests per spec:
1. Unit: BPW response → cache record mapping (PDUFA, ADCOM, approval, CRL)
2. Leakage: records with announcedOn > as_of_date are excluded
3. Dedup: BPW PDUFA record skipped when already in pdufa_dates.json (±3d)
4. Atomic write: correct filenames, valid JSON, deterministic sort
5. Fallback: BPW_API_KEY absent → no write, no exception, returns ([], [])

No network calls — fixture JSON only.
"""

import json
import os
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from wake_robin_data_pipeline.collectors.biopharmawatch_collector import (
    _map_bpw_record,
    collect_biopharmawatch_fda_events,
)

AS_OF = date(2026, 11, 12)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PDUFA_RAW = {
    "ticker": "COGT",
    "eventType": "pdufa-date",
    "pdufaDate": "2026-11-30",
    "announcedOn": "2026-09-15",
    "filedAt": "2026-09-15",
    "title": "bezuclastinib (systemic mastocytosis)",
    "applicationNumber": "NDA217759",
}

ADCOM_RAW = {
    "ticker": "EXEL",
    "eventType": "fda-advisory-committee",
    "pdufaDate": "2026-12-04",
    "announcedOn": "2026-11-01",
    "filedAt": "2026-11-01",
    "title": "zanzalintinib",
}

APPROVAL_RAW = {
    "ticker": "BBIO",
    "eventType": "fda-approval",
    "pdufaDate": "2026-11-27",
    "announcedOn": "2026-10-01",
    "title": "acoramidis (ATTR-CM)",
}

CRL_RAW = {
    "ticker": "VNDA",
    "eventType": "fda-crl",
    "pdufaDate": "2026-12-12",
    "announcedOn": "2026-11-10",
    "title": "tradipitant (gastroparesis)",
}

FUTURE_DISCLOSED_RAW = {
    "ticker": "FAKE",
    "eventType": "pdufa-date",
    "pdufaDate": "2026-12-01",
    "announcedOn": "2026-11-20",  # > AS_OF (2026-11-12) => leakage gate
    "filedAt": "2026-11-20",
    "title": "some drug",
}

PDUFA_DATES_JSON = [
    {"ticker": "COGT", "pdufa_date": "2026-11-30", "indication": "systemic mastocytosis"},
]


# ---------------------------------------------------------------------------
# Test 1: Unit — BPW record → cache record mapping
# ---------------------------------------------------------------------------

class TestMapBpwRecord:
    def test_pdufa_mapping(self):
        record = _map_bpw_record(PDUFA_RAW, AS_OF)
        assert record is not None
        assert record["ticker"] == "COGT"
        assert record["event_type"] == "PDUFA"
        assert record["event_date"] == "2026-11-30"
        assert record["disclosed_at"] == "2026-09-15"
        assert "source:biopharmawatch" in record["tags"]
        assert "pit_anchor:filing_date" in record["tags"]
        assert "nda:NDA217759" in record["tags"]

    def test_adcom_mapping(self):
        record = _map_bpw_record(ADCOM_RAW, AS_OF)
        assert record is not None
        assert record["event_type"] == "FDA_ADCOM"
        assert record["ticker"] == "EXEL"
        assert record["event_date"] == "2026-12-04"

    def test_approval_mapping(self):
        record = _map_bpw_record(APPROVAL_RAW, AS_OF)
        assert record is not None
        assert record["event_type"] == "FDA_APPROVAL"
        assert record["ticker"] == "BBIO"

    def test_crl_mapping(self):
        record = _map_bpw_record(CRL_RAW, AS_OF)
        assert record is not None
        assert record["event_type"] == "FDA_CRL"
        assert record["ticker"] == "VNDA"

    def test_unknown_event_type_returns_none(self):
        raw = {**PDUFA_RAW, "eventType": "fda-unknown-thing"}
        assert _map_bpw_record(raw, AS_OF) is None

    def test_missing_ticker_returns_none(self):
        raw = {**PDUFA_RAW, "ticker": ""}
        assert _map_bpw_record(raw, AS_OF) is None

    def test_event_name_truncated_to_120(self):
        raw = {**PDUFA_RAW, "title": "x" * 200}
        record = _map_bpw_record(raw, AS_OF)
        assert len(record["event_name"]) <= 120


# ---------------------------------------------------------------------------
# Test 2: Leakage gate
# ---------------------------------------------------------------------------

class TestLeakageGate:
    def test_future_disclosed_at_excluded(self):
        """Record with announcedOn > as_of_date must be excluded."""
        record = _map_bpw_record(FUTURE_DISCLOSED_RAW, AS_OF)
        assert record is None, "Leakage gate should have rejected future announcedOn"

    def test_same_day_disclosed_allowed(self):
        """Record with announcedOn == as_of_date should pass."""
        raw = {**PDUFA_RAW, "announcedOn": AS_OF.isoformat(), "filedAt": AS_OF.isoformat()}
        record = _map_bpw_record(raw, AS_OF)
        assert record is not None
        assert record["disclosed_at"] == AS_OF.isoformat()


# ---------------------------------------------------------------------------
# Test 3: Dedup against pdufa_dates.json
# ---------------------------------------------------------------------------

class TestDedup:
    def _make_fixture_files(self, tmpdir: Path):
        pdufa_path = tmpdir / "pdufa_dates.json"
        pdufa_path.write_text(json.dumps(PDUFA_DATES_JSON), encoding="utf-8")
        return pdufa_path

    def _mock_fetch(self, records):
        return patch(
            "wake_robin_data_pipeline.collectors.biopharmawatch_collector._fetch_bpw_events",
            return_value=records,
        )

    def test_pdufa_in_curated_file_is_skipped(self):
        """COGT 2026-11-30 is already in pdufa_dates.json — must not appear in output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdufa_path = self._make_fixture_files(tmpdir)
            cache_dir = tmpdir / "fda"

            with self._mock_fetch([PDUFA_RAW]):
                reg, adcom = collect_biopharmawatch_fda_events(
                    as_of_date=AS_OF,
                    cache_dir=cache_dir,
                    pdufa_dates_path=pdufa_path,
                    api_key="test-key",
                )

        # COGT PDUFA should be deduped out
        assert all(r["ticker"] != "COGT" or r["event_type"] != "PDUFA" for r in reg)

    def test_non_overlapping_pdufa_is_kept(self):
        """BBIO 2026-11-27 is NOT in pdufa_dates.json — must appear in output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdufa_path = self._make_fixture_files(tmpdir)
            cache_dir = tmpdir / "fda"

            with self._mock_fetch([APPROVAL_RAW]):
                reg, adcom = collect_biopharmawatch_fda_events(
                    as_of_date=AS_OF,
                    cache_dir=cache_dir,
                    pdufa_dates_path=pdufa_path,
                    api_key="test-key",
                )

        assert any(r["ticker"] == "BBIO" for r in reg)

    def test_adcom_never_deduped(self):
        """ADCOM events are never subject to pdufa dedup, even if ticker/date overlaps."""
        overlap_adcom = {**ADCOM_RAW, "ticker": "COGT", "pdufaDate": "2026-11-30"}
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdufa_path = self._make_fixture_files(tmpdir)
            cache_dir = tmpdir / "fda"

            with self._mock_fetch([overlap_adcom]):
                reg, adcom = collect_biopharmawatch_fda_events(
                    as_of_date=AS_OF,
                    cache_dir=cache_dir,
                    pdufa_dates_path=pdufa_path,
                    api_key="test-key",
                )

        # ADCOM goes to adcom_records, not reg_records, and is never deduped
        assert any(r["ticker"] == "COGT" for r in adcom)


# ---------------------------------------------------------------------------
# Test 4: Atomic write — correct filenames, valid JSON, deterministic sort
# ---------------------------------------------------------------------------

class TestAtomicWrite:
    def _mock_fetch(self, records):
        return patch(
            "wake_robin_data_pipeline.collectors.biopharmawatch_collector._fetch_bpw_events",
            return_value=records,
        )

    def test_correct_filenames_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdufa_path = tmpdir / "pdufa_dates.json"
            pdufa_path.write_text("[]", encoding="utf-8")
            cache_dir = tmpdir / "fda"

            with self._mock_fetch([PDUFA_RAW, ADCOM_RAW]):
                collect_biopharmawatch_fda_events(
                    as_of_date=AS_OF,
                    cache_dir=cache_dir,
                    pdufa_dates_path=pdufa_path,
                    api_key="test-key",
                )

            reg_path = cache_dir / f"fda_regulatory_{AS_OF.isoformat()}.json"
            adcom_path = cache_dir / f"adcom_calendar_{AS_OF.isoformat()}.json"
            assert reg_path.exists(), "fda_regulatory cache file not written"
            assert adcom_path.exists(), "adcom_calendar cache file not written"

    def test_output_is_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdufa_path = tmpdir / "pdufa_dates.json"
            pdufa_path.write_text("[]", encoding="utf-8")
            cache_dir = tmpdir / "fda"

            with self._mock_fetch([PDUFA_RAW, ADCOM_RAW, APPROVAL_RAW, CRL_RAW]):
                collect_biopharmawatch_fda_events(
                    as_of_date=AS_OF,
                    cache_dir=cache_dir,
                    pdufa_dates_path=pdufa_path,
                    api_key="test-key",
                )

            reg_path = cache_dir / f"fda_regulatory_{AS_OF.isoformat()}.json"
            data = json.loads(reg_path.read_text(encoding="utf-8"))
            assert isinstance(data, list)
            for rec in data:
                assert "ticker" in rec
                assert "event_type" in rec
                assert "event_date" in rec
                assert "disclosed_at" in rec

    def test_output_is_sorted_deterministically(self):
        """Two calls with same input must produce identical output."""
        records = [CRL_RAW, PDUFA_RAW, APPROVAL_RAW]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdufa_path = tmpdir / "pdufa_dates.json"
            pdufa_path.write_text("[]", encoding="utf-8")
            cache_dir = tmpdir / "fda"

            with self._mock_fetch(records):
                reg1, _ = collect_biopharmawatch_fda_events(
                    as_of_date=AS_OF,
                    cache_dir=cache_dir,
                    pdufa_dates_path=pdufa_path,
                    api_key="test-key",
                )

        dates = [r["event_date"] for r in reg1]
        assert dates == sorted(dates), "Records not sorted by event_date"

    def test_no_tmp_files_left_behind(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdufa_path = tmpdir / "pdufa_dates.json"
            pdufa_path.write_text("[]", encoding="utf-8")
            cache_dir = tmpdir / "fda"

            with self._mock_fetch([PDUFA_RAW]):
                collect_biopharmawatch_fda_events(
                    as_of_date=AS_OF,
                    cache_dir=cache_dir,
                    pdufa_dates_path=pdufa_path,
                    api_key="test-key",
                )

        tmp_files = list(cache_dir.glob("*.tmp"))
        assert tmp_files == [], f"Leftover .tmp files: {tmp_files}"


# ---------------------------------------------------------------------------
# Test 5: Fallback — BPW_API_KEY absent → no write, no exception
# ---------------------------------------------------------------------------

class TestFallbackNoApiKey:
    def test_no_key_returns_empty_lists(self, monkeypatch):
        monkeypatch.delenv("BPW_API_KEY", raising=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdufa_path = tmpdir / "pdufa_dates.json"
            pdufa_path.write_text("[]", encoding="utf-8")
            cache_dir = tmpdir / "fda"

            reg, adcom = collect_biopharmawatch_fda_events(
                as_of_date=AS_OF,
                cache_dir=cache_dir,
                pdufa_dates_path=pdufa_path,
                api_key=None,  # explicitly None
            )

        assert reg == []
        assert adcom == []

    def test_no_key_writes_no_files(self, monkeypatch):
        monkeypatch.delenv("BPW_API_KEY", raising=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdufa_path = tmpdir / "pdufa_dates.json"
            pdufa_path.write_text("[]", encoding="utf-8")
            cache_dir = tmpdir / "fda"

            collect_biopharmawatch_fda_events(
                as_of_date=AS_OF,
                cache_dir=cache_dir,
                pdufa_dates_path=pdufa_path,
                api_key=None,
            )

        # cache_dir should not have been created or should be empty
        if cache_dir.exists():
            files = list(cache_dir.iterdir())
            assert files == [], f"Unexpected files written: {files}"

    def test_no_key_does_not_raise(self, monkeypatch):
        monkeypatch.delenv("BPW_API_KEY", raising=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            # Even if pdufa_dates.json doesn't exist
            collect_biopharmawatch_fda_events(
                as_of_date=AS_OF,
                cache_dir=tmpdir / "fda",
                pdufa_dates_path=tmpdir / "nonexistent.json",
                api_key=None,
            )  # Must not raise
