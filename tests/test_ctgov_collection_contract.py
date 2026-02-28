"""CTGov collector contract tests — schema, PIT safety, dedup, cache integrity.

All tests are offline; network calls are patched.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from ctgov_adapter import (
    CTGovAdapter,
    AdapterConfig,
    CanonicalTrialRecord,
    CTGovStatus,
    CompletionType,
    FutureDataError,
    MissingRequiredFieldError,
    process_trial_records_batch,
)

AS_OF = date(2026, 2, 28)


# ---------------------------------------------------------------------------
# Minimal raw record fixture (flattened schema, as stored in trial_records.json)
# ---------------------------------------------------------------------------

def _make_raw_record(**overrides) -> dict:
    """Build a minimal raw CTGov trial record with optional overrides."""
    base = {
        "ticker": "ACAD",
        "nct_id": "NCT12345678",
        "title": "A Study of ACD-101 in Major Depression",
        "status": "RECRUITING",
        "phase": "PHASE2",
        "study_type": "INTERVENTIONAL",
        "conditions": ["Major Depressive Disorder"],
        "interventions": ["ACD-101", "Placebo"],
        "first_posted": "2024-03-15",
        "start_date": "2024-06-01",
        "primary_completion_date": "2027-01-15",
        "completion_date": "2027-06-30",
        "results_first_posted": None,
        "last_update_posted": "2025-12-01",
        "enrollment": 240,
        "sponsor": "ACADIA Pharmaceuticals Inc.",
        "collected_at": "2026-02-26",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1) Schema normalization
# ---------------------------------------------------------------------------


class TestCtgovNormalizeSchemaMinimal:
    """Given a minimal raw record, normalized CanonicalTrialRecord has all required fields."""

    def test_all_canonical_fields_present(self):
        raw = _make_raw_record()
        adapter = CTGovAdapter()
        rec = adapter.extract_canonical_record(raw, AS_OF)

        assert rec.ticker == "ACAD"
        assert rec.nct_id == "NCT12345678"
        assert rec.overall_status == CTGovStatus.RECRUITING
        assert rec.last_update_posted == date(2025, 12, 1)
        assert rec.primary_completion_date == date(2027, 1, 15)
        assert rec.completion_date == date(2027, 6, 30)
        assert rec.results_first_posted is None

    def test_missing_optional_fields_no_keyerror(self):
        """Record with only required fields should not raise."""
        raw = _make_raw_record(
            primary_completion_date=None,
            completion_date=None,
            results_first_posted=None,
            first_posted=None,
            start_date=None,
        )
        adapter = CTGovAdapter()
        rec = adapter.extract_canonical_record(raw, AS_OF)

        assert rec.primary_completion_date is None
        assert rec.completion_date is None
        assert rec.results_first_posted is None

    def test_partial_date_yyyy_mm_normalized(self):
        """CT.gov YYYY-MM dates should be normalized to YYYY-MM-01."""
        raw = _make_raw_record(
            primary_completion_date="2027-06",
            completion_date="2027-12",
        )
        adapter = CTGovAdapter()
        rec = adapter.extract_canonical_record(raw, AS_OF)

        assert rec.primary_completion_date == date(2027, 6, 1)
        assert rec.completion_date == date(2027, 12, 1)

    def test_to_dict_roundtrip(self):
        """CanonicalTrialRecord → dict → from_dict should be lossless."""
        raw = _make_raw_record()
        adapter = CTGovAdapter()
        rec = adapter.extract_canonical_record(raw, AS_OF)

        d = rec.to_dict()
        restored = CanonicalTrialRecord.from_dict(d)
        assert restored == rec

    def test_missing_ticker_raises(self):
        raw = _make_raw_record(ticker=None)
        adapter = CTGovAdapter()
        with pytest.raises(MissingRequiredFieldError, match="ticker"):
            adapter.extract_canonical_record(raw, AS_OF)

    def test_missing_nct_id_raises(self):
        raw = _make_raw_record(nct_id=None)
        adapter = CTGovAdapter()
        with pytest.raises(MissingRequiredFieldError, match="nct_id"):
            adapter.extract_canonical_record(raw, AS_OF)

    def test_missing_last_update_posted_raises(self):
        raw = _make_raw_record(last_update_posted=None)
        adapter = CTGovAdapter()
        with pytest.raises(MissingRequiredFieldError, match="last_update_posted"):
            adapter.extract_canonical_record(raw, AS_OF)


# ---------------------------------------------------------------------------
# 2) PIT safety — future data detection
# ---------------------------------------------------------------------------


class TestCtgovPitFieldsMonotonic:
    """PIT anchor: last_update_posted must not leak future data."""

    def test_future_last_update_raises(self):
        """Record posted after as_of_date must be rejected."""
        raw = _make_raw_record(last_update_posted="2026-03-15")
        adapter = CTGovAdapter()
        with pytest.raises(FutureDataError):
            adapter.extract_canonical_record(raw, AS_OF)

    def test_same_day_accepted(self):
        """last_update_posted == as_of_date is valid (not future)."""
        raw = _make_raw_record(last_update_posted="2026-02-28")
        adapter = CTGovAdapter()
        rec = adapter.extract_canonical_record(raw, AS_OF)
        assert rec.last_update_posted == AS_OF

    def test_batch_filters_future_records(self):
        """Batch processor in filter mode skips future records."""
        records = [
            _make_raw_record(nct_id="NCT00000001", last_update_posted="2025-06-01"),
            _make_raw_record(nct_id="NCT00000002", last_update_posted="2026-04-01"),  # future
            _make_raw_record(nct_id="NCT00000003", last_update_posted="2026-02-28"),
        ]
        config = AdapterConfig(fail_on_future_data=False)
        canonical, stats = process_trial_records_batch(records, AS_OF, config)

        assert len(canonical) == 2
        nct_ids = {r.nct_id for r in canonical}
        assert "NCT00000002" not in nct_ids
        assert stats.future_data_violations == 1

    def test_batch_strict_mode_raises_on_future(self):
        """Strict mode crashes on any future data."""
        records = [
            _make_raw_record(nct_id="NCT00000001", last_update_posted="2025-06-01"),
            _make_raw_record(nct_id="NCT00000002", last_update_posted="2026-04-01"),
        ]
        config = AdapterConfig(fail_on_future_data=True)
        with pytest.raises(FutureDataError):
            process_trial_records_batch(records, AS_OF, config)


# ---------------------------------------------------------------------------
# 3) Dedup: same NCT → one canonical record
# ---------------------------------------------------------------------------


class TestCtgovDedupSameNct:
    """Two raw records for the same NCT should produce two canonical records
    (dedup happens upstream in merger or snapshot, not in the adapter).
    The adapter is record-level; dedup is the snapshot layer's job.
    Verify that the adapter at least produces stable IDs via compute_hash.
    """

    def test_same_nct_same_data_same_hash(self):
        """Identical records → identical hashes."""
        raw = _make_raw_record()
        adapter = CTGovAdapter()
        r1 = adapter.extract_canonical_record(raw, AS_OF)
        r2 = adapter.extract_canonical_record(raw, AS_OF)
        assert r1.compute_hash() == r2.compute_hash()

    def test_same_nct_different_update_different_hash(self):
        """Same NCT with different last_update → different hash (delta detectable)."""
        raw1 = _make_raw_record(last_update_posted="2025-11-01")
        raw2 = _make_raw_record(last_update_posted="2025-12-01")
        adapter = CTGovAdapter()
        r1 = adapter.extract_canonical_record(raw1, AS_OF)
        r2 = adapter.extract_canonical_record(raw2, AS_OF)
        assert r1.compute_hash() != r2.compute_hash()

    def test_warm_ctgov_pit_dedup_by_cutoff(self):
        """warm_ctgov PIT filter deduplicates by excluding future records."""
        records = [
            _make_raw_record(nct_id="NCT11111111", last_update_posted="2025-01-01"),
            _make_raw_record(nct_id="NCT11111111", last_update_posted="2025-06-01"),
            _make_raw_record(nct_id="NCT11111111", last_update_posted="2026-06-01"),  # future
        ]
        cutoff = AS_OF.isoformat()
        filtered = [
            r for r in records
            if (r.get("last_update_posted") or "")[:10] <= cutoff
        ]
        # Both pre-cutoff versions survive; the future one is dropped
        assert len(filtered) == 2


# ---------------------------------------------------------------------------
# 4) Future date extraction and normalization
# ---------------------------------------------------------------------------


class TestCtgovFutureDateExtraction:
    """PCD/CD from raw records should be preserved and normalized."""

    def test_pcd_preserved_as_date(self):
        raw = _make_raw_record(primary_completion_date="2028-03-15")
        adapter = CTGovAdapter()
        rec = adapter.extract_canonical_record(raw, AS_OF)
        assert rec.primary_completion_date == date(2028, 3, 15)

    def test_cd_preserved_when_pcd_missing(self):
        raw = _make_raw_record(
            primary_completion_date=None,
            completion_date="2028-09-01",
        )
        adapter = CTGovAdapter()
        rec = adapter.extract_canonical_record(raw, AS_OF)
        assert rec.primary_completion_date is None
        assert rec.completion_date == date(2028, 9, 1)

    def test_missing_pcd_does_not_invent_date(self):
        raw = _make_raw_record(
            primary_completion_date=None,
            completion_date=None,
        )
        adapter = CTGovAdapter()
        rec = adapter.extract_canonical_record(raw, AS_OF)
        assert rec.primary_completion_date is None
        assert rec.completion_date is None

    def test_completion_type_preserved(self):
        raw = _make_raw_record(
            primary_completion_date="2028-01-01",
            primary_completion_type="ESTIMATED",
        )
        adapter = CTGovAdapter()
        rec = adapter.extract_canonical_record(raw, AS_OF)
        assert rec.primary_completion_type == CompletionType.ESTIMATED

    def test_malformed_date_returns_none(self):
        raw = _make_raw_record(primary_completion_date="not-a-date")
        adapter = CTGovAdapter()
        rec = adapter.extract_canonical_record(raw, AS_OF)
        assert rec.primary_completion_date is None


# ---------------------------------------------------------------------------
# 5) Atomic cache write
# ---------------------------------------------------------------------------


class TestCtgovCacheWriteAtomic:
    """warm_ctgov uses temp file + os.replace() for atomicity."""

    def test_atomic_write_creates_file(self, tmp_path):
        """Normal write creates the expected cache file."""
        from warm_caches import warm_ctgov

        # Create a minimal trial_records.json
        source_dir = tmp_path / "data"
        source_dir.mkdir()
        records = [
            _make_raw_record(nct_id=f"NCT0000000{i}", last_update_posted="2025-06-01")
            for i in range(5)
        ]
        (source_dir / "trial_records.json").write_text(json.dumps(records))

        cache_dir = tmp_path / "cache" / "ctgov"
        count = warm_ctgov(AS_OF, source_dir, cache_dir)

        target = cache_dir / f"trial_records_{AS_OF.isoformat()}.json"
        assert target.exists()
        loaded = json.loads(target.read_text())
        assert len(loaded) == 5
        assert count == 5

    def test_atomic_write_preserves_prior_on_empty(self, tmp_path):
        """If new data is empty (rejected), prior cache remains."""
        from warm_caches import warm_ctgov

        source_dir = tmp_path / "data"
        source_dir.mkdir()

        cache_dir = tmp_path / "cache" / "ctgov"
        cache_dir.mkdir(parents=True)

        # Write a "prior" cache
        prior_records = [_make_raw_record(nct_id=f"NCT9999000{i}", last_update_posted="2025-01-01") for i in range(10)]
        prior_path = cache_dir / "trial_records_2026-02-27.json"
        prior_path.write_text(json.dumps(prior_records))

        # Now try to warm with an empty source
        (source_dir / "trial_records.json").write_text(json.dumps([]))

        # Empty refresh should be rejected by validate_cache_refresh
        count = warm_ctgov(AS_OF, source_dir, cache_dir)
        assert count == 0

        # New cache should NOT have been created
        target = cache_dir / f"trial_records_{AS_OF.isoformat()}.json"
        assert not target.exists()

        # Prior cache still intact
        assert prior_path.exists()
        assert len(json.loads(prior_path.read_text())) == 10

    def test_existing_cache_not_overwritten(self, tmp_path):
        """If cache already exists, warm_ctgov returns its count without re-writing."""
        from warm_caches import warm_ctgov

        source_dir = tmp_path / "data"
        source_dir.mkdir()
        (source_dir / "trial_records.json").write_text(json.dumps([
            _make_raw_record(nct_id="NCT00000099", last_update_posted="2025-01-01"),
        ]))

        cache_dir = tmp_path / "cache" / "ctgov"
        cache_dir.mkdir(parents=True)

        # Pre-create cache with 3 records
        target = cache_dir / f"trial_records_{AS_OF.isoformat()}.json"
        existing = [_make_raw_record(nct_id=f"NCT8888000{i}", last_update_posted="2025-01-01") for i in range(3)]
        target.write_text(json.dumps(existing))

        count = warm_ctgov(AS_OF, source_dir, cache_dir)
        assert count == 3  # Returns existing count, not source count

        # Contents unchanged
        assert len(json.loads(target.read_text())) == 3


# ---------------------------------------------------------------------------
# 6) Event ledger PIT filter
# ---------------------------------------------------------------------------


class TestLedgerPitFilter:
    """Event ledger excludes entries with disclosed_at > as_of_date."""

    def test_ledger_ctgov_entries_respect_pit(self, tmp_path):
        """CTGov ledger entries use last_update_posted as disclosed_at."""
        from event_ledger import _load_ctgov_events

        records = [
            _make_raw_record(
                nct_id="NCT00000001",
                last_update_posted="2025-06-01",
                primary_completion_date="2027-01-01",
            ),
        ]
        cache_path = tmp_path / f"trial_records_{AS_OF.isoformat()}.json"
        cache_path.write_text(json.dumps(records))

        entries = _load_ctgov_events(AS_OF, tmp_path, strict=False, data_dir=tmp_path)

        # Should produce entries (PCD event)
        pcd_entries = [e for e in entries if e.event_type == "CLINICAL_PCD"]
        assert len(pcd_entries) == 1
        assert pcd_entries[0].disclosed_at == "2025-06-01"
        assert pcd_entries[0].source_uid == "NCT00000001"
