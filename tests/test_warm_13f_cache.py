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
    PITFilingSelection,
    RateLimiter,
    build_index,
    check_13f_cache_health,
    select_pit_filing,
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

    def test_basic_index_schema(self):
        results = [
            {"manager_cik": "0001234567", "manager_name": "Fund A", "status": "ok",
             "period_of_report": "2025-12-31", "filed_at": "2026-02-14",
             "form_type": "13F-HR", "accession": "A1", "holdings_count": 50},
            {"manager_cik": "0009876543", "manager_name": "Fund B", "status": "no_filing",
             "rejection_reason": "no_filings"},
        ]
        idx = build_index(date(2026, 2, 19), results, total_managers=2, elite_only=True)

        assert idx["as_of_date"] == "2026-02-19"
        assert idx["elite_only"] is True
        assert idx["total_managers"] == 2
        assert idx["managers_with_filing"] == 1
        assert idx["managers_no_filing"] == 1
        assert idx["managers_error"] == 0
        assert idx["coverage_pct"] == 50.0
        assert len(idx["managers"]) == 2

    def test_coverage_100_percent(self):
        results = [
            {"manager_cik": f"CIK{i}", "manager_name": f"Fund {i}", "status": "ok",
             "period_of_report": "2025-12-31", "filed_at": "2026-02-14",
             "form_type": "13F-HR", "accession": f"A{i}", "holdings_count": 10}
            for i in range(5)
        ]
        idx = build_index(date(2026, 2, 19), results, total_managers=5, elite_only=False)
        assert idx["coverage_pct"] == 100.0

    def test_empty_results(self):
        idx = build_index(date(2026, 2, 19), [], total_managers=0, elite_only=True)
        assert idx["coverage_pct"] == 0.0
        assert idx["managers_with_filing"] == 0


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

    def test_pass_when_coverage_adequate(self, tmp_path):
        cache_dir = tmp_path / "sec_13f" / "PIT"
        date_dir = cache_dir / "2026-02-19"
        date_dir.mkdir(parents=True)

        index = {
            "coverage_pct": 90.0,
            "managers_with_filing": 18,
            "total_managers": 20,
        }
        (date_dir / "index.json").write_text(json.dumps(index))

        result = check_13f_cache_health(cache_dir, "2026-02-19")
        assert result["status"] == "PASS"
        assert "90.0%" in result["detail"]

    def test_warn_when_coverage_below_threshold(self, tmp_path):
        cache_dir = tmp_path / "sec_13f" / "PIT"
        date_dir = cache_dir / "2026-02-19"
        date_dir.mkdir(parents=True)

        index = {
            "coverage_pct": 60.0,
            "managers_with_filing": 12,
            "total_managers": 20,
        }
        (date_dir / "index.json").write_text(json.dumps(index))

        result = check_13f_cache_health(cache_dir, "2026-02-19")
        assert result["status"] == "WARN"
        assert "below" in result["detail"]

    def test_warn_when_index_missing(self, tmp_path):
        cache_dir = tmp_path / "sec_13f" / "PIT"
        result = check_13f_cache_health(cache_dir, "2026-02-19")
        assert result["status"] == "WARN"
        assert "No 13F cache index" in result["detail"]

    def test_warn_when_index_malformed(self, tmp_path):
        cache_dir = tmp_path / "sec_13f" / "PIT"
        date_dir = cache_dir / "2026-02-19"
        date_dir.mkdir(parents=True)
        (date_dir / "index.json").write_text("not valid json{{{")

        result = check_13f_cache_health(cache_dir, "2026-02-19")
        assert result["status"] == "WARN"
        assert "Cannot read" in result["detail"]

    def test_custom_threshold(self, tmp_path):
        cache_dir = tmp_path / "sec_13f" / "PIT"
        date_dir = cache_dir / "2026-02-19"
        date_dir.mkdir(parents=True)

        index = {
            "coverage_pct": 85.0,
            "managers_with_filing": 17,
            "total_managers": 20,
        }
        (date_dir / "index.json").write_text(json.dumps(index))

        # With default 80% threshold → PASS
        result = check_13f_cache_health(cache_dir, "2026-02-19", warn_coverage_pct=80.0)
        assert result["status"] == "PASS"

        # With 90% threshold → WARN
        result = check_13f_cache_health(cache_dir, "2026-02-19", warn_coverage_pct=90.0)
        assert result["status"] == "WARN"
