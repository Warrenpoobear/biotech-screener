"""Tests for tools/backfill_13f_history.py — PIT-safe 13F history backfill."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from sec_13f.edgar_13f import SEC13FFetcher, Filing13F, Holding
from tools.warm_13f_cache import (
    SCHEMA_VERSION,
    CACHE_TYPE,
    validate_sec_13f_index_schema,
)
from tools.backfill_13f_history import (
    TOOL_VERSION,
    DEFAULT_LOOKBACK_FILINGS,
    generate_quarter_end_dates,
    _warm_one_manager_backfill,
    _build_backfill_index,
    _is_date_complete,
    _find_missing_managers,
    _backfill_one_date,
    _build_coverage_report,
    _load_managers,
    backfill_13f_history,
)
from tools.warm_13f_cache import RateLimiter


# ---------------------------------------------------------------------------
# Helpers (mirrors test_warm_13f_cache.py patterns)
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


def _make_mock_fetcher(
    filings: list[Filing13F] | None = None,
    holdings: list[Holding] | None = None,
) -> MagicMock:
    fetcher = MagicMock(spec=SEC13FFetcher)
    fetcher.get_recent_filings.return_value = filings or []
    fetcher.parse_holdings.return_value = holdings or []
    fetcher.cache_dir = None
    return fetcher


MANAGER_A = {"cik": "1234567", "name": "Fund A"}
MANAGER_B = {"cik": "9876543", "name": "Fund B"}
MANAGER_C = {"cik": "5555555", "name": "Fund C"}


# ===================================================================
# Quarter-end date generation
# ===================================================================

class TestGenerateQuarterEndDates:

    def test_single_year_all_four_quarters(self):
        dates = generate_quarter_end_dates(date(2024, 1, 1), date(2024, 12, 31))
        assert dates == [
            date(2024, 3, 31),
            date(2024, 6, 30),
            date(2024, 9, 30),
            date(2024, 12, 31),
        ]

    def test_cross_year_boundary(self):
        dates = generate_quarter_end_dates(date(2024, 10, 1), date(2025, 6, 30))
        assert dates == [
            date(2024, 12, 31),
            date(2025, 3, 31),
            date(2025, 6, 30),
        ]

    def test_exact_boundary_inclusive(self):
        # date_from is exactly a quarter-end
        dates = generate_quarter_end_dates(date(2024, 3, 31), date(2024, 3, 31))
        assert dates == [date(2024, 3, 31)]

    def test_mid_quarter_start(self):
        # Start after Q1 end, before Q2 end
        dates = generate_quarter_end_dates(date(2024, 4, 15), date(2024, 7, 15))
        assert dates == [date(2024, 6, 30)]

    def test_empty_range(self):
        dates = generate_quarter_end_dates(date(2024, 12, 31), date(2024, 1, 1))
        assert dates == []

    def test_no_quarter_ends_in_range(self):
        # Range entirely within a quarter
        dates = generate_quarter_end_dates(date(2024, 1, 1), date(2024, 1, 31))
        assert dates == []

    def test_multi_year_count(self):
        dates = generate_quarter_end_dates(date(2020, 1, 1), date(2025, 12, 31))
        # 6 years * 4 quarters = 24
        assert len(dates) == 24
        assert dates[0] == date(2020, 3, 31)
        assert dates[-1] == date(2025, 12, 31)

    def test_ascending_order(self):
        dates = generate_quarter_end_dates(date(2022, 1, 1), date(2024, 12, 31))
        for i in range(1, len(dates)):
            assert dates[i] > dates[i - 1]


# ===================================================================
# Resume: skip valid existing index
# ===================================================================

class TestResumeSkipsValidExistingIndex:

    def _write_valid_index(self, out_root: Path, as_of: str) -> None:
        date_dir = out_root / as_of
        date_dir.mkdir(parents=True)
        index = {
            "schema_version": SCHEMA_VERSION,
            "cache_type": CACHE_TYPE,
            "as_of_date": as_of,
            "created_at": "2026-01-01T00:00:00Z",
            "elite_only": True,
            "total_managers": 1,
            "managers_with_filing": 1,
            "coverage_pct": 100.0,
            "selection_policy": {
                "pit_rule": "filing_date<=as_of; latest report_date; prefer 13F-HR/A; latest filing_date",
                "recent_filings_lookback_n": 16,
            },
            "managers": [
                {
                    "manager_cik": "0001234567",
                    "manager_name": "Fund A",
                    "selected": True,
                    "period_of_report": "2025-12-31",
                    "filed_at": "2026-01-15",
                    "form_type": "13F-HR",
                    "accession": "A1",
                    "holdings_count": 10,
                    "manager_json_path": "managers/0001234567.json",
                    "raw_xml_path": "",
                    "rejection_reason": "",
                },
            ],
        }
        (date_dir / "index.json").write_text(json.dumps(index))

    def test_complete_date_skipped(self, tmp_path):
        self._write_valid_index(tmp_path, "2025-12-31")
        assert _is_date_complete(tmp_path, date(2025, 12, 31)) is True

    def test_missing_index_not_skipped(self, tmp_path):
        assert _is_date_complete(tmp_path, date(2025, 12, 31)) is False

    def test_corrupt_json_not_skipped(self, tmp_path):
        date_dir = tmp_path / "2025-12-31"
        date_dir.mkdir(parents=True)
        (date_dir / "index.json").write_text("{bad json")
        assert _is_date_complete(tmp_path, date(2025, 12, 31)) is False

    def test_invalid_schema_not_skipped(self, tmp_path):
        date_dir = tmp_path / "2025-12-31"
        date_dir.mkdir(parents=True)
        (date_dir / "index.json").write_text(json.dumps({
            "schema_version": "wrong_version",
        }))
        assert _is_date_complete(tmp_path, date(2025, 12, 31)) is False


# ===================================================================
# Resume: repair partial cache
# ===================================================================

class TestResumeRepairsPartialCache:

    def _write_manager_json(self, date_dir: Path, cik_padded: str) -> None:
        managers_dir = date_dir / "managers"
        managers_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "manager_cik": cik_padded,
            "manager_name": "Fund",
            "as_of_date": "2025-12-31",
            "period_of_report": "2025-09-30",
            "filed_at": "2025-11-15",
            "form_type": "13F-HR",
            "accession": "ACC1",
            "holdings": [{"cusip": "037833100", "ticker": "AAPL",
                          "issuer": "APPLE", "shares": 100, "value_usd_thousands": 10,
                          "put_call": ""}],
        }
        (managers_dir / f"{cik_padded}.json").write_text(json.dumps(data))

    def test_partial_cache_only_fetches_missing(self, tmp_path):
        date_dir = tmp_path / "2025-12-31"
        # Manager A is cached
        self._write_manager_json(date_dir, "0001234567")

        missing, existing = _find_missing_managers(
            date_dir, [MANAGER_A, MANAGER_B]
        )

        assert len(existing) == 1
        assert existing[0]["manager_cik"] == "0001234567"
        assert existing[0]["status"] == "ok"
        assert len(missing) == 1
        assert missing[0]["cik"] == "9876543"

    def test_corrupt_manager_json_refetched(self, tmp_path):
        date_dir = tmp_path / "2025-12-31"
        managers_dir = date_dir / "managers"
        managers_dir.mkdir(parents=True, exist_ok=True)
        # Write corrupt JSON
        (managers_dir / "0001234567.json").write_text("{corrupt")

        missing, existing = _find_missing_managers(
            date_dir, [MANAGER_A]
        )
        assert len(missing) == 1
        assert len(existing) == 0


# ===================================================================
# PIT selection via backfill worker
# ===================================================================

class TestPITSelectionViaBackfill:

    def test_prefers_latest_report_date(self, tmp_path):
        old = _make_filing(
            filing_date=date(2025, 8, 14), report_date=date(2025, 6, 30),
            accession="Q2",
        )
        new = _make_filing(
            filing_date=date(2025, 11, 14), report_date=date(2025, 9, 30),
            accession="Q3",
        )
        fetcher = _make_mock_fetcher([old, new], [_make_holding()])
        rl = RateLimiter(rate=1000.0)

        result = _warm_one_manager_backfill(
            MANAGER_A, date(2025, 12, 31), tmp_path, fetcher, rl,
            lookback_filings=16,
        )
        assert result["status"] == "ok"
        assert result["period_of_report"] == "2025-09-30"

    def test_prefers_amendment(self, tmp_path):
        original = _make_filing(
            filing_date=date(2025, 11, 14), report_date=date(2025, 9, 30),
            form_type="13F-HR", accession="ORIG",
        )
        amendment = _make_filing(
            filing_date=date(2025, 11, 20), report_date=date(2025, 9, 30),
            form_type="13F-HR/A", accession="AMEND",
        )
        fetcher = _make_mock_fetcher([original, amendment], [_make_holding()])
        rl = RateLimiter(rate=1000.0)

        result = _warm_one_manager_backfill(
            MANAGER_A, date(2025, 12, 31), tmp_path, fetcher, rl,
            lookback_filings=16,
        )
        assert result["form_type"] == "13F-HR/A"
        assert result["accession"] == "AMEND"

    def test_excludes_future_filings(self, tmp_path):
        past = _make_filing(
            filing_date=date(2025, 8, 14), report_date=date(2025, 6, 30),
            accession="PAST",
        )
        future = _make_filing(
            filing_date=date(2026, 2, 14), report_date=date(2025, 12, 31),
            accession="FUTURE",
        )
        fetcher = _make_mock_fetcher([past, future], [_make_holding()])
        rl = RateLimiter(rate=1000.0)

        result = _warm_one_manager_backfill(
            MANAGER_A, date(2025, 12, 31), tmp_path, fetcher, rl,
            lookback_filings=16,
        )
        assert result["status"] == "ok"
        assert result["period_of_report"] == "2025-06-30"


# ===================================================================
# Index schema has backfill block
# ===================================================================

class TestIndexSchemaHasBackfillBlock:

    def _make_ok_result(self, cik_padded: str = "0001234567") -> Dict[str, Any]:
        return {
            "manager_cik": cik_padded,
            "manager_name": "Fund A",
            "status": "ok",
            "period_of_report": "2025-12-31",
            "filed_at": "2026-02-14",
            "form_type": "13F-HR",
            "accession": "A1",
            "holdings_count": 50,
            "manager_json_path": f"managers/{cik_padded}.json",
            "raw_xml_path": "",
        }

    def test_backfill_key_present(self):
        index = _build_backfill_index(
            date(2025, 12, 31),
            [self._make_ok_result()],
            total_managers=1,
            elite_only=False,
            lookback_filings=16,
            cadence="quarterly",
            manager_set="coinvest",
        )
        assert "backfill" in index
        bf = index["backfill"]
        assert bf["tool_version"] == TOOL_VERSION
        assert bf["cadence"] == "quarterly"
        assert bf["manager_set"] == "coinvest"
        assert bf["lookback_filings"] == 16
        assert bf["generated_at"].endswith("Z")

    def test_index_passes_v1_validation(self):
        index = _build_backfill_index(
            date(2025, 12, 31),
            [self._make_ok_result()],
            total_managers=1,
            elite_only=False,
            lookback_filings=16,
            cadence="quarterly",
            manager_set="coinvest",
        )
        ok, detail = validate_sec_13f_index_schema(
            index, expected_as_of_date="2025-12-31"
        )
        assert ok, f"backfill index failed v1 validation: {detail}"

    def test_selection_policy_has_correct_lookback(self):
        index = _build_backfill_index(
            date(2025, 12, 31),
            [self._make_ok_result()],
            total_managers=1,
            elite_only=True,
            lookback_filings=16,
            cadence="quarterly",
            manager_set="elite",
        )
        assert index["selection_policy"]["recent_filings_lookback_n"] == 16


# ===================================================================
# Backfill integration (mock fetcher, real filesystem)
# ===================================================================

class TestBackfillIntegration:

    def _run_backfill(
        self, tmp_path, managers, filings, holdings, **kwargs
    ) -> Dict[str, Any]:
        fetcher = _make_mock_fetcher(filings, holdings)

        with patch(
            "tools.backfill_13f_history.SEC13FFetcher", return_value=fetcher
        ), patch(
            "tools.backfill_13f_history._load_managers", return_value=managers
        ):
            return backfill_13f_history(
                date_from=kwargs.pop("date_from", date(2025, 9, 30)),
                date_to=kwargs.pop("date_to", date(2025, 12, 31)),
                out_root=tmp_path,
                manager_set="coinvest",
                max_workers=1,
                rate_limit=1000.0,
                **kwargs,
            )

    def test_single_date_e2e(self, tmp_path):
        filing = _make_filing(
            cik="1234567",
            filing_date=date(2025, 8, 14),
            report_date=date(2025, 6, 30),
        )
        summary = self._run_backfill(
            tmp_path,
            managers=[MANAGER_A],
            filings=[filing],
            holdings=[_make_holding()],
            date_from=date(2025, 9, 30),
            date_to=date(2025, 9, 30),
        )

        assert summary["total_dates"] == 1
        assert len(summary["per_date"]) == 1
        assert summary["per_date"][0]["coverage_pct"] == 100.0

        # Verify on-disk index
        index_path = tmp_path / "2025-09-30" / "index.json"
        assert index_path.exists()
        index = json.loads(index_path.read_text())
        assert index["schema_version"] == SCHEMA_VERSION
        assert index["as_of_date"] == "2025-09-30"
        assert "backfill" in index

        # Verify manager JSON
        mgr_path = tmp_path / "2025-09-30" / "managers" / "0001234567.json"
        assert mgr_path.exists()

    def test_multiple_dates(self, tmp_path):
        filing = _make_filing(
            cik="1234567",
            filing_date=date(2025, 8, 14),
            report_date=date(2025, 6, 30),
        )
        summary = self._run_backfill(
            tmp_path,
            managers=[MANAGER_A],
            filings=[filing],
            holdings=[_make_holding()],
            date_from=date(2025, 6, 30),
            date_to=date(2025, 12, 31),
        )
        # Q2 (Jun 30) + Q3 (Sep 30) + Q4 (Dec 31) = 3 dates
        assert summary["total_dates"] == 3
        assert len(summary["per_date"]) == 3

    def test_max_dates_limits(self, tmp_path):
        filing = _make_filing(
            cik="1234567",
            filing_date=date(2025, 8, 14),
            report_date=date(2025, 6, 30),
        )
        summary = self._run_backfill(
            tmp_path,
            managers=[MANAGER_A],
            filings=[filing],
            holdings=[_make_holding()],
            date_from=date(2024, 1, 1),
            date_to=date(2025, 12, 31),
            max_dates=2,
        )
        assert summary["total_dates"] == 2
        assert len(summary["per_date"]) == 2

    def test_max_managers_limits(self, tmp_path):
        filing = _make_filing(
            cik="1234567",
            filing_date=date(2025, 11, 14),
            report_date=date(2025, 9, 30),
        )
        summary = self._run_backfill(
            tmp_path,
            managers=[MANAGER_A, MANAGER_B, MANAGER_C],
            filings=[filing],
            holdings=[_make_holding()],
            date_from=date(2025, 12, 31),
            date_to=date(2025, 12, 31),
            max_managers=1,
        )
        # Only 1 manager processed
        pd = summary["per_date"][0]
        assert pd["total_managers"] == 1


# ===================================================================
# Manager set loading
# ===================================================================

class TestManagerSet:

    @patch("tools.backfill_13f_history.get_all_managers")
    def test_coinvest_calls_get_all_managers(self, mock_all):
        mock_all.return_value = [MANAGER_A]
        result = _load_managers("coinvest")
        mock_all.assert_called_once()
        assert result == [MANAGER_A]

    @patch("tools.backfill_13f_history.get_elite_managers")
    def test_elite_calls_get_elite_managers(self, mock_elite):
        mock_elite.return_value = [MANAGER_A]
        result = _load_managers("elite")
        mock_elite.assert_called_once()
        assert result == [MANAGER_A]

    def test_unknown_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unknown manager_set"):
            _load_managers("unknown_set")


# ===================================================================
# Coverage report
# ===================================================================

class TestCoverageReport:

    def test_empty_per_date(self):
        rpt = _build_coverage_report([], total_managers=10)
        assert rpt["total_dates"] == 0

    def test_all_100_percent(self):
        per_date = [
            {"coverage_pct": 100.0},
            {"coverage_pct": 100.0},
        ]
        rpt = _build_coverage_report(per_date, total_managers=5)
        assert rpt["avg_coverage_pct"] == 100.0
        assert rpt["min_coverage_pct"] == 100.0
        assert rpt["dates_below_80pct"] == 0
        assert rpt["dates_at_100pct"] == 2

    def test_mixed_coverage(self):
        per_date = [
            {"coverage_pct": 100.0},
            {"coverage_pct": 60.0},
            {"coverage_pct": 80.0},
        ]
        rpt = _build_coverage_report(per_date, total_managers=10)
        assert rpt["avg_coverage_pct"] == 80.0
        assert rpt["min_coverage_pct"] == 60.0
        assert rpt["max_coverage_pct"] == 100.0
        assert rpt["dates_below_80pct"] == 1
        assert rpt["dates_at_100pct"] == 1


# ===================================================================
# Resume integration: skip + repair
# ===================================================================

class TestResumeIntegration:

    def test_resume_skips_complete_dates(self, tmp_path):
        # Pre-populate a complete date
        date_dir = tmp_path / "2025-09-30"
        date_dir.mkdir(parents=True)
        index = {
            "schema_version": SCHEMA_VERSION,
            "cache_type": CACHE_TYPE,
            "as_of_date": "2025-09-30",
            "created_at": "2025-10-01T00:00:00Z",
            "elite_only": False,
            "total_managers": 1,
            "managers_with_filing": 1,
            "coverage_pct": 100.0,
            "selection_policy": {
                "pit_rule": "filing_date<=as_of; latest report_date; prefer 13F-HR/A; latest filing_date",
                "recent_filings_lookback_n": 16,
            },
            "managers": [
                {
                    "manager_cik": "0001234567",
                    "manager_name": "Fund A",
                    "selected": True,
                    "period_of_report": "2025-06-30",
                    "filed_at": "2025-08-14",
                    "form_type": "13F-HR",
                    "accession": "A1",
                    "holdings_count": 10,
                    "manager_json_path": "managers/0001234567.json",
                    "raw_xml_path": "",
                    "rejection_reason": "",
                },
            ],
        }
        (date_dir / "index.json").write_text(json.dumps(index))

        filing = _make_filing(
            cik="1234567",
            filing_date=date(2025, 11, 14),
            report_date=date(2025, 9, 30),
        )
        fetcher = _make_mock_fetcher([filing], [_make_holding()])

        with patch(
            "tools.backfill_13f_history.SEC13FFetcher", return_value=fetcher
        ), patch(
            "tools.backfill_13f_history._load_managers",
            return_value=[MANAGER_A],
        ):
            summary = backfill_13f_history(
                date_from=date(2025, 9, 30),
                date_to=date(2025, 12, 31),
                out_root=tmp_path,
                resume=True,
                max_workers=1,
                rate_limit=1000.0,
            )

        # Q3 should be skipped, Q4 fetched
        assert summary["dates_skipped"] == 1
        assert summary["total_dates"] == 2
        # Q3 was skipped (fetched=0)
        q3 = summary["per_date"][0]
        assert q3["as_of_date"] == "2025-09-30"
        assert q3.get("skipped") is True
        assert q3["fetched"] == 0
