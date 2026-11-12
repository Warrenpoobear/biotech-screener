"""
Tests for pdufa_bio_collector.py

5 test classes per spec:
1. Unit: pdufa.bio response → cache record mapping (PDUFA, ADCOM, day precision filter)
2. Leakage: records with updated_at > as_of_date excluded
3. Dedup: BPW PDUFA record skipped when already in pdufa_dates.json (±3d)
4. Atomic write: correct filenames, valid JSON, deterministic sort, no .tmp leftover
5. Cache-exists: when both cache files exist (e.g. BPW ran), returns early without fetch

No network calls — fixture JSON only.
"""

import json
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from wake_robin_data_pipeline.collectors.pdufa_bio_collector import (
    _map_record,
    collect_pdufa_bio_events,
)

AS_OF = date(2026, 11, 12)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PDUFA_RAW = {
    "ticker": "COGT",
    "date": "2026-11-30",
    "date_precision": "day",
    "name": "bezuclastinib (systemic mastocytosis)",
    "type": "PDUFA",
    "updated_at": "2026-09-15T00:00:00Z",
    "nct_id": "NCT04784468",
}

ADCOM_RAW = {
    "ticker": "EXEL",
    "date": "2026-12-04",
    "date_precision": "day",
    "name": "zanzalintinib",
    "type": "ADCOMM",
    "updated_at": "2026-11-01T00:00:00Z",
}

NON_DAY_PRECISION_RAW = {
    "ticker": "FAKE",
    "date": None,
    "date_precision": "month",
    "name": "some drug",
    "type": "PDUFA",
    "updated_at": "2026-10-01T00:00:00Z",
}

FUTURE_UPDATED_RAW = {
    "ticker": "FUTK",
    "date": "2026-12-01",
    "date_precision": "day",
    "name": "future drug",
    "type": "PDUFA",
    "updated_at": "2026-11-20T00:00:00Z",  # > AS_OF (2026-11-12)
}

PDUFA_DATES_JSON = [
    {"ticker": "COGT", "pdufa_date": "2026-11-30", "indication": "systemic mastocytosis"},
]


def _mock_fetch(pdufa_records, adcom_records):
    """Patch _fetch_endpoint to return fixture records by endpoint path."""
    def side_effect(session, endpoint, params):
        if endpoint == "/pdufa":
            return pdufa_records
        if endpoint == "/adcomm":
            return adcom_records
        return []
    return patch(
        "wake_robin_data_pipeline.collectors.pdufa_bio_collector._fetch_endpoint",
        side_effect=side_effect,
    )


# ---------------------------------------------------------------------------
# Test 1: Unit — pdufa.bio record → cache record mapping
# ---------------------------------------------------------------------------

class TestMapRecord:
    def test_pdufa_mapping(self):
        record = _map_record(PDUFA_RAW, "PDUFA", AS_OF)
        assert record is not None
        assert record["ticker"] == "COGT"
        assert record["event_type"] == "PDUFA"
        assert record["event_date"] == "2026-11-30"
        assert record["disclosed_at"] == "2026-09-15"
        assert "source:pdufa_bio" in record["tags"]
        assert "pit_anchor:updated_at" in record["tags"]
        assert "nct:NCT04784468" in record["tags"]

    def test_adcom_mapping(self):
        record = _map_record(ADCOM_RAW, "FDA_ADCOM", AS_OF)
        assert record is not None
        assert record["event_type"] == "FDA_ADCOM"
        assert record["ticker"] == "EXEL"
        assert record["event_date"] == "2026-12-04"
        assert record["disclosed_at"] == "2026-11-01"

    def test_non_day_precision_returns_none(self):
        """Records with date_precision != 'day' must be skipped."""
        record = _map_record(NON_DAY_PRECISION_RAW, "PDUFA", AS_OF)
        assert record is None

    def test_missing_ticker_returns_none(self):
        raw = {**PDUFA_RAW, "ticker": ""}
        assert _map_record(raw, "PDUFA", AS_OF) is None

    def test_null_date_returns_none(self):
        raw = {**PDUFA_RAW, "date": None, "date_precision": "day"}
        assert _map_record(raw, "PDUFA", AS_OF) is None

    def test_missing_updated_at_uses_as_of_date(self):
        raw = {**PDUFA_RAW, "updated_at": None}
        record = _map_record(raw, "PDUFA", AS_OF)
        assert record is not None
        assert record["disclosed_at"] == AS_OF.isoformat()
        assert "pit_anchor:as_of_date" in record["tags"]

    def test_event_name_truncated_to_120(self):
        raw = {**PDUFA_RAW, "name": "x" * 200}
        record = _map_record(raw, "PDUFA", AS_OF)
        assert len(record["event_name"]) <= 120


# ---------------------------------------------------------------------------
# Test 2: Leakage gate
# ---------------------------------------------------------------------------

class TestLeakageGate:
    def test_future_updated_at_excluded(self):
        """Record with updated_at > as_of_date must be excluded."""
        record = _map_record(FUTURE_UPDATED_RAW, "PDUFA", AS_OF)
        assert record is None, "Leakage gate should reject future updated_at"

    def test_same_day_updated_at_allowed(self):
        raw = {**PDUFA_RAW, "updated_at": AS_OF.isoformat() + "T00:00:00Z"}
        record = _map_record(raw, "PDUFA", AS_OF)
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

    def test_pdufa_in_curated_file_is_skipped(self):
        """COGT 2026-11-30 is in pdufa_dates.json — must not appear in output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdufa_path = self._make_fixture_files(tmpdir)
            cache_dir = tmpdir / "fda"

            with _mock_fetch([PDUFA_RAW], []):
                reg, adcom = collect_pdufa_bio_events(
                    as_of_date=AS_OF,
                    cache_dir=cache_dir,
                    pdufa_dates_path=pdufa_path,
                )

            assert all(r["ticker"] != "COGT" or r["event_type"] != "PDUFA" for r in reg)

    def test_non_overlapping_pdufa_is_kept(self):
        """EXEL 2026-12-04 is not in pdufa_dates.json — must appear."""
        different_pdufa = {**PDUFA_RAW, "ticker": "EXEL", "date": "2026-12-04"}
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdufa_path = self._make_fixture_files(tmpdir)
            cache_dir = tmpdir / "fda"

            with _mock_fetch([different_pdufa], []):
                reg, _ = collect_pdufa_bio_events(
                    as_of_date=AS_OF,
                    cache_dir=cache_dir,
                    pdufa_dates_path=pdufa_path,
                )

            assert any(r["ticker"] == "EXEL" for r in reg)

    def test_adcom_never_deduped(self):
        """ADCOM records are never subject to pdufa dedup."""
        overlap_adcom = {**ADCOM_RAW, "ticker": "COGT", "date": "2026-11-30"}
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdufa_path = self._make_fixture_files(tmpdir)
            cache_dir = tmpdir / "fda"

            with _mock_fetch([], [overlap_adcom]):
                _, adcom = collect_pdufa_bio_events(
                    as_of_date=AS_OF,
                    cache_dir=cache_dir,
                    pdufa_dates_path=pdufa_path,
                )

            assert any(r["ticker"] == "COGT" for r in adcom)


# ---------------------------------------------------------------------------
# Test 4: Atomic write
# ---------------------------------------------------------------------------

class TestAtomicWrite:
    def test_correct_filenames_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdufa_path = tmpdir / "pdufa_dates.json"
            pdufa_path.write_text("[]", encoding="utf-8")
            cache_dir = tmpdir / "fda"

            with _mock_fetch([PDUFA_RAW], [ADCOM_RAW]):
                collect_pdufa_bio_events(
                    as_of_date=AS_OF,
                    cache_dir=cache_dir,
                    pdufa_dates_path=pdufa_path,
                )

            assert (cache_dir / f"fda_regulatory_{AS_OF.isoformat()}.json").exists()
            assert (cache_dir / f"adcom_calendar_{AS_OF.isoformat()}.json").exists()

    def test_output_is_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdufa_path = tmpdir / "pdufa_dates.json"
            pdufa_path.write_text("[]", encoding="utf-8")
            cache_dir = tmpdir / "fda"

            with _mock_fetch([PDUFA_RAW], [ADCOM_RAW]):
                collect_pdufa_bio_events(
                    as_of_date=AS_OF,
                    cache_dir=cache_dir,
                    pdufa_dates_path=pdufa_path,
                )

            reg_path = cache_dir / f"fda_regulatory_{AS_OF.isoformat()}.json"
            data = json.loads(reg_path.read_text(encoding="utf-8"))
            assert isinstance(data, list)
            for rec in data:
                assert "ticker" in rec
                assert "event_type" in rec
                assert "event_date" in rec
                assert "disclosed_at" in rec

    def test_output_is_sorted_by_event_date(self):
        raw2 = {**PDUFA_RAW, "ticker": "BBIO", "date": "2026-11-27"}
        raw3 = {**PDUFA_RAW, "ticker": "EXEL", "date": "2026-12-03"}
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdufa_path = tmpdir / "pdufa_dates.json"
            pdufa_path.write_text("[]", encoding="utf-8")
            cache_dir = tmpdir / "fda"

            with _mock_fetch([raw3, raw2], []):
                reg, _ = collect_pdufa_bio_events(
                    as_of_date=AS_OF,
                    cache_dir=cache_dir,
                    pdufa_dates_path=pdufa_path,
                )

            dates = [r["event_date"] for r in reg]
            assert dates == sorted(dates)

    def test_no_tmp_files_left_behind(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdufa_path = tmpdir / "pdufa_dates.json"
            pdufa_path.write_text("[]", encoding="utf-8")
            cache_dir = tmpdir / "fda"

            with _mock_fetch([PDUFA_RAW], []):
                collect_pdufa_bio_events(
                    as_of_date=AS_OF,
                    cache_dir=cache_dir,
                    pdufa_dates_path=pdufa_path,
                )

            assert list(cache_dir.glob("*.tmp")) == []


# ---------------------------------------------------------------------------
# Test 5: Cache-exists short-circuit (BPW already ran)
# ---------------------------------------------------------------------------

class TestCacheExistsShortCircuit:
    def test_returns_early_when_both_files_exist(self):
        """If fda_regulatory and adcom_calendar both exist, no fetch should occur."""
        existing_reg = [{"ticker": "COGT", "event_type": "PDUFA", "event_date": "2026-11-30",
                         "disclosed_at": "2026-09-15", "event_name": "existing", "tags": []}]
        existing_adcom = [{"ticker": "EXEL", "event_type": "FDA_ADCOM", "event_date": "2026-12-04",
                           "disclosed_at": "2026-11-01", "event_name": "existing adcom", "tags": []}]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdufa_path = tmpdir / "pdufa_dates.json"
            pdufa_path.write_text("[]", encoding="utf-8")
            cache_dir = tmpdir / "fda"
            cache_dir.mkdir()

            reg_path = cache_dir / f"fda_regulatory_{AS_OF.isoformat()}.json"
            adcom_path = cache_dir / f"adcom_calendar_{AS_OF.isoformat()}.json"
            reg_path.write_text(json.dumps(existing_reg), encoding="utf-8")
            adcom_path.write_text(json.dumps(existing_adcom), encoding="utf-8")

            fetch_called = []

            def mock_fetch(session, endpoint, params):
                fetch_called.append(endpoint)
                return []

            with patch(
                "wake_robin_data_pipeline.collectors.pdufa_bio_collector._fetch_endpoint",
                side_effect=mock_fetch,
            ):
                reg, adcom = collect_pdufa_bio_events(
                    as_of_date=AS_OF,
                    cache_dir=cache_dir,
                    pdufa_dates_path=pdufa_path,
                )

            assert fetch_called == [], "Should not have fetched when cache files exist"
            assert reg == existing_reg
            assert adcom == existing_adcom
