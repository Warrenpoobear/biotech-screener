"""Tests for tools/warm_13f_cache.py — PIT-safe 13F warm cache builder."""
from __future__ import annotations

import json
import threading
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import under test
# ---------------------------------------------------------------------------
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from sec_13f.edgar_13f import SEC13FFetcher, Filing13F, Holding
from tools.warm_13f_cache import (
    SCHEMA_VERSION,
    CACHE_TYPE,
    FILINGS_LOOKBACK_N,
    PITFilingSelection,
    RateLimiter,
    build_index,
    check_13f_cache_health,
    select_pit_filing,
    validate_sec_13f_index_schema,
    warm_one_manager,
    warm_13f_cache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_filing(
    cik: str = "0001234567",
    accession: str = "0001234567260000XX",
    filing_date: date = date(2026, 2, 14),
    report_date: date = date(2025, 12, 31),
    form_type: str = "13F-HR",
    filer_name: str = "Test Fund",
) -> Filing13F:
    return Filing13F(
        cik=cik,
        accession_number=accession,
        filing_date=filing_date,
        report_date=report_date,
        form_type=form_type,
        primary_doc_url=f"https://sec.gov/{accession}/primary.htm",
        filer_name=filer_name,
    )


def _make_holding(
    cusip: str = "037833100",
    ticker: str = "AAPL",
    issuer: str = "APPLE INC",
    shares: int = 100000,
    value: int = 15000,
) -> Holding:
    return Holding(
        cusip=cusip,
        issuer_name=issuer,
        class_title="COM",
        shares=shares,
        value=value,
        shares_type="SH",
        investment_discretion="SOLE",
        ticker=ticker,
    )


def _make_valid_index(**overrides) -> dict:
    """Build a minimal valid v1 index dict. Override any field."""
    base = {
        "schema_version": SCHEMA_VERSION,
        "cache_type": CACHE_TYPE,
        "as_of_date": "2026-02-19",
        "created_at": "2026-02-19T15:30:00Z",
        "elite_only": True,
        "total_managers": 2,
        "managers_with_filing": 2,
        "coverage_pct": 100.0,
        "selection_policy": {
            "pit_rule": "filing_date<=as_of; latest report_date; prefer 13F-HR/A; latest filing_date",
            "recent_filings_lookback_n": FILINGS_LOOKBACK_N,
        },
        "managers": [
            {
                "manager_cik": "0001234567",
                "manager_name": "Fund A",
                "selected": True,
                "period_of_report": "2025-12-31",
                "filed_at": "2026-02-14",
                "form_type": "13F-HR",
                "accession": "A1",
                "holdings_count": 50,
                "manager_json_path": "managers/0001234567.json",
                "raw_xml_path": "raw/0001234567/A1.xml",
                "rejection_reason": "",
            },
            {
                "manager_cik": "0009876543",
                "manager_name": "Fund B",
                "selected": True,
                "period_of_report": "2025-12-31",
                "filed_at": "2026-02-14",
                "form_type": "13F-HR",
                "accession": "B1",
                "holdings_count": 30,
                "manager_json_path": "managers/0009876543.json",
                "raw_xml_path": "raw/0009876543/B1.xml",
                "rejection_reason": "",
            },
        ],
    }
    base.update(overrides)
    return base


# ===================================================================
# PIT selection tests (pure logic, no mocks)
# ===================================================================

class TestSelectPITFiling:
    """Pure PIT selection tests — construct Filing13F directly."""

    def test_single_filing_before_cutoff(self):
        f = _make_filing(filing_date=date(2026, 2, 14))
        sel = select_pit_filing([f], date(2026, 2, 19))
        assert sel.filing is f
        assert sel.period_of_report == "2025-12-31"
        assert sel.rejection_reason is None

    def test_single_filing_after_cutoff_rejected(self):
        f = _make_filing(filing_date=date(2026, 2, 20))
        sel = select_pit_filing([f], date(2026, 2, 19))
        assert sel.filing is None
        assert "no_filings_before" in sel.rejection_reason

    def test_single_filing_on_cutoff_included(self):
        f = _make_filing(filing_date=date(2026, 2, 19))
        sel = select_pit_filing([f], date(2026, 2, 19))
        assert sel.filing is f

    def test_latest_report_date_wins(self):
        old = _make_filing(
            filing_date=date(2025, 11, 15),
            report_date=date(2025, 9, 30),
            accession="A001",
        )
        new = _make_filing(
            filing_date=date(2026, 2, 14),
            report_date=date(2025, 12, 31),
            accession="A002",
        )
        sel = select_pit_filing([old, new], date(2026, 2, 19))
        assert sel.filing is new
        assert sel.period_of_report == "2025-12-31"

    def test_amendment_preferred_over_original(self):
        original = _make_filing(
            filing_date=date(2026, 2, 14),
            report_date=date(2025, 12, 31),
            form_type="13F-HR",
            accession="A_orig",
        )
        amendment = _make_filing(
            filing_date=date(2026, 2, 16),
            report_date=date(2025, 12, 31),
            form_type="13F-HR/A",
            accession="A_amend",
        )
        sel = select_pit_filing([original, amendment], date(2026, 2, 19))
        assert sel.filing is amendment
        assert sel.form_type == "13F-HR/A"

    def test_latest_amendment_wins(self):
        amend1 = _make_filing(
            filing_date=date(2026, 2, 15),
            report_date=date(2025, 12, 31),
            form_type="13F-HR/A",
            accession="A1",
        )
        amend2 = _make_filing(
            filing_date=date(2026, 2, 17),
            report_date=date(2025, 12, 31),
            form_type="13F-HR/A",
            accession="A2",
        )
        sel = select_pit_filing([amend1, amend2], date(2026, 2, 19))
        assert sel.filing is amend2

    def test_future_amendment_excluded(self):
        original = _make_filing(
            filing_date=date(2026, 2, 14),
            report_date=date(2025, 12, 31),
            form_type="13F-HR",
            accession="A_orig",
        )
        future_amend = _make_filing(
            filing_date=date(2026, 2, 25),
            report_date=date(2025, 12, 31),
            form_type="13F-HR/A",
            accession="A_future",
        )
        sel = select_pit_filing([original, future_amend], date(2026, 2, 19))
        assert sel.filing is original
        assert sel.form_type == "13F-HR"

    def test_empty_filings_list(self):
        sel = select_pit_filing([], date(2026, 2, 19))
        assert sel.filing is None
        assert sel.rejection_reason == "no_filings"

    def test_multiple_report_dates_picks_latest_eligible(self):
        q3 = _make_filing(
            filing_date=date(2025, 11, 14),
            report_date=date(2025, 9, 30),
            accession="Q3",
        )
        q4 = _make_filing(
            filing_date=date(2026, 2, 14),
            report_date=date(2025, 12, 31),
            accession="Q4",
        )
        # Q1 filed after cutoff
        q1 = _make_filing(
            filing_date=date(2026, 5, 15),
            report_date=date(2026, 3, 31),
            accession="Q1",
        )
        sel = select_pit_filing([q3, q4, q1], date(2026, 2, 19))
        assert sel.filing is q4

    def test_cik_preserved_in_selection(self):
        f = _make_filing(cik="1166559", filing_date=date(2026, 2, 14))
        sel = select_pit_filing([f], date(2026, 2, 19))
        assert sel.manager_cik == "1166559"


# ===================================================================
# Index building tests (pure logic)
# ===================================================================

class TestBuildIndex:

    def test_v1_schema_fields(self):
        results = [
            {"manager_cik": "0001234567", "manager_name": "Fund A", "status": "ok",
             "period_of_report": "2025-12-31", "filed_at": "2026-02-14",
             "form_type": "13F-HR", "accession": "A1", "holdings_count": 50,
             "manager_json_path": "managers/0001234567.json",
             "raw_xml_path": "raw/0001234567/A1.xml"},
            {"manager_cik": "0009876543", "manager_name": "Fund B", "status": "no_filing",
             "rejection_reason": "no_filings"},
        ]
        idx = build_index(date(2026, 2, 19), results, total_managers=2, elite_only=True)

        # v1 required top-level fields
        assert idx["schema_version"] == SCHEMA_VERSION
        assert idx["cache_type"] == CACHE_TYPE
        assert idx["as_of_date"] == "2026-02-19"
        assert idx["created_at"].endswith("Z")
        assert idx["elite_only"] is True
        assert idx["total_managers"] == 2
        assert idx["managers_with_filing"] == 1
        assert idx["coverage_pct"] == 50.0
        assert idx["selection_policy"]["recent_filings_lookback_n"] == FILINGS_LOOKBACK_N
        assert len(idx["managers"]) == 2

    def test_selected_manager_has_paths(self):
        results = [
            {"manager_cik": "0001234567", "manager_name": "Fund A", "status": "ok",
             "period_of_report": "2025-12-31", "filed_at": "2026-02-14",
             "form_type": "13F-HR", "accession": "A1", "holdings_count": 50,
             "manager_json_path": "managers/0001234567.json",
             "raw_xml_path": "raw/0001234567/A1.xml"},
        ]
        idx = build_index(date(2026, 2, 19), results, total_managers=1, elite_only=True)
        mgr = idx["managers"][0]

        assert mgr["selected"] is True
        assert mgr["manager_json_path"] == "managers/0001234567.json"
        assert mgr["raw_xml_path"] == "raw/0001234567/A1.xml"
        assert mgr["rejection_reason"] == ""

    def test_unselected_manager_has_rejection_reason(self):
        results = [
            {"manager_cik": "0009876543", "manager_name": "Fund B", "status": "no_filing",
             "rejection_reason": "no_filings"},
        ]
        idx = build_index(date(2026, 2, 19), results, total_managers=1, elite_only=True)
        mgr = idx["managers"][0]

        assert mgr["selected"] is False
        assert mgr["rejection_reason"] == "no_filings"
        assert "manager_json_path" not in mgr

    def test_coverage_100_percent(self):
        results = [
            {"manager_cik": f"CIK{i:07d}", "manager_name": f"Fund {i}", "status": "ok",
             "period_of_report": "2025-12-31", "filed_at": "2026-02-14",
             "form_type": "13F-HR", "accession": f"A{i}", "holdings_count": 10,
             "manager_json_path": f"managers/CIK{i:07d}.json",
             "raw_xml_path": f"raw/CIK{i:07d}/A{i}.xml"}
            for i in range(5)
        ]
        idx = build_index(date(2026, 2, 19), results, total_managers=5, elite_only=False)
        assert idx["coverage_pct"] == 100.0

    def test_empty_results(self):
        idx = build_index(date(2026, 2, 19), [], total_managers=0, elite_only=True)
        assert idx["coverage_pct"] == 0.0
        assert idx["managers_with_filing"] == 0

    def test_validates_against_own_schema(self):
        """Index produced by build_index must pass the schema validator."""
        results = [
            {"manager_cik": "0001234567", "manager_name": "Fund A", "status": "ok",
             "period_of_report": "2025-12-31", "filed_at": "2026-02-14",
             "form_type": "13F-HR", "accession": "A1", "holdings_count": 50,
             "manager_json_path": "managers/0001234567.json",
             "raw_xml_path": "raw/0001234567/A1.xml"},
            {"manager_cik": "0009876543", "manager_name": "Fund B", "status": "no_filing",
             "rejection_reason": "no_filings"},
        ]
        idx = build_index(date(2026, 2, 19), results, total_managers=2, elite_only=True)
        ok, detail = validate_sec_13f_index_schema(idx)
        assert ok, f"build_index output failed validation: {detail}"


# ===================================================================
# Schema validation tests (pure logic)
# ===================================================================

class TestValidateSchema:

    def test_valid_index_passes(self):
        ok, detail = validate_sec_13f_index_schema(_make_valid_index())
        assert ok, detail

    def test_missing_top_level_field(self):
        idx = _make_valid_index()
        del idx["schema_version"]
        ok, detail = validate_sec_13f_index_schema(idx)
        assert not ok
        assert "missing top-level" in detail

    def test_wrong_schema_version(self):
        idx = _make_valid_index(schema_version="sec_13f_pit_index.v99")
        ok, detail = validate_sec_13f_index_schema(idx)
        assert not ok
        assert "schema_version mismatch" in detail

    def test_wrong_cache_type(self):
        idx = _make_valid_index(cache_type="something_else")
        ok, detail = validate_sec_13f_index_schema(idx)
        assert not ok
        assert "cache_type mismatch" in detail

    def test_as_of_date_mismatch(self):
        idx = _make_valid_index(as_of_date="2026-02-19")
        ok, detail = validate_sec_13f_index_schema(idx, expected_as_of_date="2026-02-20")
        assert not ok
        assert "as_of_date mismatch" in detail

    def test_created_at_must_end_with_z(self):
        idx = _make_valid_index(created_at="2026-02-19T15:30:00+00:00")
        ok, detail = validate_sec_13f_index_schema(idx)
        assert not ok
        assert "created_at" in detail

    def test_managers_with_filing_exceeds_total(self):
        idx = _make_valid_index(total_managers=1, managers_with_filing=5, coverage_pct=500.0)
        ok, detail = validate_sec_13f_index_schema(idx)
        assert not ok
        assert "managers_with_filing" in detail

    def test_coverage_pct_inconsistent(self):
        # 2 managers, 2 with filing → should be 100.0, not 50.0
        idx = _make_valid_index(coverage_pct=50.0)
        ok, detail = validate_sec_13f_index_schema(idx)
        assert not ok
        assert "coverage_pct inconsistent" in detail

    def test_selected_manager_missing_required_fields(self):
        idx = _make_valid_index()
        # Remove a required field from a selected manager
        del idx["managers"][0]["holdings_count"]
        ok, detail = validate_sec_13f_index_schema(idx)
        assert not ok
        assert "missing" in detail
        assert "holdings_count" in detail

    def test_unselected_manager_empty_rejection_reason(self):
        idx = _make_valid_index()
        idx["managers"][1] = {
            "manager_cik": "0009876543",
            "manager_name": "Fund B",
            "selected": False,
            "rejection_reason": "",  # empty → invalid
        }
        idx["managers_with_filing"] = 1
        idx["coverage_pct"] = 50.0
        ok, detail = validate_sec_13f_index_schema(idx)
        assert not ok
        assert "rejection_reason is empty" in detail

    def test_managers_not_sorted_by_cik(self):
        idx = _make_valid_index()
        # Reverse order so CIKs are descending
        idx["managers"] = list(reversed(idx["managers"]))
        ok, detail = validate_sec_13f_index_schema(idx)
        assert not ok
        assert "not sorted" in detail

    def test_determinism_sorted_ciks_pass(self):
        """Managers sorted by CIK must pass."""
        idx = _make_valid_index()
        ciks = [m["manager_cik"] for m in idx["managers"]]
        assert ciks == sorted(ciks)
        ok, _ = validate_sec_13f_index_schema(idx)
        assert ok


# ===================================================================
# Rate limiter tests
# ===================================================================

class TestRateLimiter:

    def test_sequential_spacing(self):
        rl = RateLimiter(rate=100.0)  # 100/s = 10ms interval
        t0 = time.monotonic()
        for _ in range(5):
            rl.acquire()
        elapsed = time.monotonic() - t0
        # 5 acquires at 100/s → ~40ms minimum (4 intervals)
        assert elapsed >= 0.035  # small margin

    def test_thread_safety(self):
        rl = RateLimiter(rate=50.0)  # 20ms interval
        timestamps: list[float] = []
        lock = threading.Lock()

        def worker():
            rl.acquire()
            with lock:
                timestamps.append(time.monotonic())

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(timestamps) == 4
        # All timestamps should be spaced by at least ~18ms (small margin)
        timestamps.sort()
        for i in range(1, len(timestamps)):
            assert timestamps[i] - timestamps[i - 1] >= 0.015


# ===================================================================
# warm_one_manager integration tests (mock EDGAR)
# ===================================================================

class TestWarmOneManager:

    def _make_mock_fetcher(self, filings: list[Filing13F], holdings: list[Holding]):
        fetcher = MagicMock(spec=SEC13FFetcher)
        fetcher.get_recent_filings.return_value = filings
        fetcher.parse_holdings.return_value = holdings
        fetcher.cache_dir = None
        return fetcher

    def test_writes_manager_json(self, tmp_path):
        filing = _make_filing(cik="1166559")
        holdings = [_make_holding()]
        fetcher = self._make_mock_fetcher([filing], holdings)
        rl = RateLimiter(rate=1000.0)

        manager = {"cik": "1166559", "name": "Baker Bros"}
        result = warm_one_manager(manager, date(2026, 2, 19), tmp_path, fetcher, rl)

        assert result["status"] == "ok"
        assert result["holdings_count"] == 1
        assert result["manager_json_path"] == "managers/0001166559.json"

        # Check JSON file written
        cik_padded = "0001166559"
        mgr_path = tmp_path / "managers" / f"{cik_padded}.json"
        assert mgr_path.exists()

        data = json.loads(mgr_path.read_text())
        assert data["manager_cik"] == cik_padded
        assert data["as_of_date"] == "2026-02-19"
        assert len(data["holdings"]) == 1
        assert data["holdings"][0]["ticker"] == "AAPL"
        assert data["holdings"][0]["value_usd_thousands"] == 15000

    def test_no_filing_returns_no_filing_status(self, tmp_path):
        fetcher = self._make_mock_fetcher([], [])
        rl = RateLimiter(rate=1000.0)

        manager = {"cik": "9999999", "name": "Empty Fund"}
        result = warm_one_manager(manager, date(2026, 2, 19), tmp_path, fetcher, rl)

        assert result["status"] == "no_filing"
        assert "no_filings" in result["rejection_reason"]

    def test_fetcher_error_returns_error_status(self, tmp_path):
        fetcher = MagicMock(spec=SEC13FFetcher)
        fetcher.get_recent_filings.side_effect = RuntimeError("network error")
        fetcher.cache_dir = None
        rl = RateLimiter(rate=1000.0)

        manager = {"cik": "1111111", "name": "Broken Fund"}
        result = warm_one_manager(manager, date(2026, 2, 19), tmp_path, fetcher, rl)

        assert result["status"] == "error"
        assert "network error" in result["error"]

    def test_raw_xml_copied_when_cache_exists(self, tmp_path):
        filing = _make_filing(cik="1166559", accession="0001166559260001")
        holdings = [_make_holding()]

        # Set up fetcher with a real cache_dir containing a cached XML
        fetcher_cache = tmp_path / "fetcher_cache"
        fetcher_cache.mkdir()
        xml_content = "<xml>test</xml>"
        xml_file = fetcher_cache / f"{filing.filing_id}_infotable.xml"
        xml_file.write_text(xml_content)

        fetcher = MagicMock(spec=SEC13FFetcher)
        fetcher.get_recent_filings.return_value = [filing]
        fetcher.parse_holdings.return_value = holdings
        fetcher.cache_dir = fetcher_cache

        rl = RateLimiter(rate=1000.0)
        out = tmp_path / "out"

        manager = {"cik": "1166559", "name": "Baker Bros"}
        result = warm_one_manager(manager, date(2026, 2, 19), out, fetcher, rl)

        assert result["status"] == "ok"
        assert result["raw_xml_path"] == f"raw/0001166559/{filing.accession_number}.xml"
        raw_path = out / "raw" / "0001166559" / f"{filing.accession_number}.xml"
        assert raw_path.exists()
        assert raw_path.read_text() == xml_content

    def test_manager_json_schema_complete(self, tmp_path):
        filing = _make_filing(cik="1166559")
        h1 = _make_holding(cusip="037833100", ticker="AAPL", shares=100000, value=15000)
        h2 = _make_holding(cusip="594918104", ticker="MSFT", shares=50000, value=20000)
        h2.put_call = "CALL"

        fetcher = self._make_mock_fetcher([filing], [h1, h2])
        rl = RateLimiter(rate=1000.0)

        manager = {"cik": "1166559", "name": "Baker Bros"}
        warm_one_manager(manager, date(2026, 2, 19), tmp_path, fetcher, rl)

        data = json.loads((tmp_path / "managers" / "0001166559.json").read_text())
        assert data["period_of_report"] == "2025-12-31"
        assert data["filed_at"] == "2026-02-14"
        assert data["form_type"] == "13F-HR"
        assert len(data["holdings"]) == 2

        # Check put_call propagation
        msft = [h for h in data["holdings"] if h["ticker"] == "MSFT"][0]
        assert msft["put_call"] == "CALL"


# ===================================================================
# Gate function tests
# ===================================================================

class TestCheck13FCacheHealth:

    def _write_valid_index(self, cache_dir: Path, as_of: str, **overrides):
        date_dir = cache_dir / as_of
        date_dir.mkdir(parents=True, exist_ok=True)
        idx = _make_valid_index(as_of_date=as_of, **overrides)
        (date_dir / "index.json").write_text(json.dumps(idx))
        return idx

    def test_pass_when_coverage_adequate(self, tmp_path):
        self._write_valid_index(tmp_path, "2026-02-19")
        result = check_13f_cache_health(tmp_path, "2026-02-19")
        assert result["status"] == "PASS"
        assert "100.0%" in result["detail"]

    def test_warn_when_coverage_below_threshold(self, tmp_path):
        self._write_valid_index(
            tmp_path, "2026-02-19",
            managers_with_filing=1, coverage_pct=50.0,
            managers=[
                _make_valid_index()["managers"][0],
                {
                    "manager_cik": "0009876543",
                    "manager_name": "Fund B",
                    "selected": False,
                    "rejection_reason": "no_filings",
                },
            ],
        )
        result = check_13f_cache_health(tmp_path, "2026-02-19")
        assert result["status"] == "WARN"
        assert "below" in result["detail"]

    def test_warn_when_index_missing(self, tmp_path):
        result = check_13f_cache_health(tmp_path, "2026-02-19")
        assert result["status"] == "WARN"
        assert "No 13F cache index" in result["detail"]

    def test_warn_when_index_malformed(self, tmp_path):
        date_dir = tmp_path / "2026-02-19"
        date_dir.mkdir(parents=True)
        (date_dir / "index.json").write_text("not valid json{{{")

        result = check_13f_cache_health(tmp_path, "2026-02-19")
        assert result["status"] == "WARN"
        assert "Cannot read" in result["detail"]

    def test_warn_when_schema_version_wrong(self, tmp_path):
        self._write_valid_index(
            tmp_path, "2026-02-19",
            schema_version="sec_13f_pit_index.v99",
        )
        result = check_13f_cache_health(tmp_path, "2026-02-19")
        assert result["status"] == "WARN"
        assert "schema invalid" in result["detail"]

    def test_warn_when_as_of_date_mismatch(self, tmp_path):
        # Write index with wrong as_of_date
        date_dir = tmp_path / "2026-02-19"
        date_dir.mkdir(parents=True)
        idx = _make_valid_index(as_of_date="2026-02-18")
        (date_dir / "index.json").write_text(json.dumps(idx))

        result = check_13f_cache_health(tmp_path, "2026-02-19")
        assert result["status"] == "WARN"
        assert "schema invalid" in result["detail"]

    def test_custom_threshold(self, tmp_path):
        self._write_valid_index(tmp_path, "2026-02-19")

        # With default 80% threshold → PASS (index has 100% coverage)
        result = check_13f_cache_health(tmp_path, "2026-02-19", warn_coverage_pct=80.0)
        assert result["status"] == "PASS"
