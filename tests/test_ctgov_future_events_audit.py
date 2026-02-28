"""Tests for CTGov future events audit script (offline, deterministic)."""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest

# Add project root to path for script imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from scripts.audit_ctgov_future_events import (
    audit_future_events,
    format_markdown,
    _parse_date_safe,
)

AS_OF = date(2026, 2, 28)


def _rec(nct_id: str, ticker: str = "ACAD", **kw) -> dict:
    """Minimal trial record for audit tests."""
    base = {
        "nct_id": nct_id,
        "ticker": ticker,
        "status": "RECRUITING",
        "phase": "PHASE2",
        "study_type": "INTERVENTIONAL",
        "last_update_posted": "2025-12-01",
        "primary_completion_date": None,
        "completion_date": None,
        "results_first_posted": None,
    }
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# 1) Counts future dates correctly
# ---------------------------------------------------------------------------


class TestAuditCountsFutureDates:
    """Verify audit correctly categorizes dates relative to as_of_date."""

    def test_future_pcd_counted(self):
        records = [
            _rec("NCT001", primary_completion_date="2027-06-15"),
            _rec("NCT002", primary_completion_date="2025-01-01"),  # past
            _rec("NCT003", primary_completion_date="2026-03-01"),  # future (1 day after as_of)
        ]
        audit = audit_future_events(records, AS_OF)
        assert audit["metrics"]["future_primary_completion_date"] == 2
        assert audit["metrics"]["pct_future_primary_completion_date"] == pytest.approx(66.67, abs=0.01)

    def test_future_cd_counted(self):
        records = [
            _rec("NCT001", completion_date="2027-12-01"),
            _rec("NCT002", completion_date="2024-06-01"),  # past
        ]
        audit = audit_future_events(records, AS_OF)
        assert audit["metrics"]["future_completion_date"] == 1

    def test_horizon_buckets(self):
        records = [
            _rec("NCT001", primary_completion_date="2026-03-15"),  # 15 days → 0-30d
            _rec("NCT002", primary_completion_date="2026-05-01"),  # 62 days → 31-90d
            _rec("NCT003", primary_completion_date="2026-08-01"),  # 154 days → 91-180d
            _rec("NCT004", primary_completion_date="2027-01-01"),  # 307 days → 181-365d
            _rec("NCT005", primary_completion_date="2028-01-01"),  # >365d
        ]
        audit = audit_future_events(records, AS_OF)
        buckets = audit["metrics"]["pcd_horizon_buckets"]
        assert buckets.get("0-30d", 0) == 1
        assert buckets.get("31-90d", 0) == 1
        assert buckets.get("91-180d", 0) == 1
        assert buckets.get("181-365d", 0) == 1
        assert buckets.get(">365d", 0) == 1

    def test_empty_records(self):
        audit = audit_future_events([], AS_OF)
        assert audit["metrics"]["total_trials"] == 0
        assert audit["metrics"]["future_primary_completion_date"] == 0

    def test_no_future_dates(self):
        records = [
            _rec("NCT001", primary_completion_date="2025-01-01"),
            _rec("NCT002", primary_completion_date="2024-06-01"),
        ]
        audit = audit_future_events(records, AS_OF)
        assert audit["metrics"]["future_primary_completion_date"] == 0
        assert audit["metrics"]["pct_future_primary_completion_date"] == 0.0


# ---------------------------------------------------------------------------
# 2) Flags malformed dates
# ---------------------------------------------------------------------------


class TestAuditFlagsMalformed:
    """Detect and flag unparseable date strings."""

    def test_malformed_pcd_flagged(self):
        records = [
            _rec("NCT001", primary_completion_date="not-a-date"),
            _rec("NCT002", primary_completion_date="2027-13-01"),  # month 13
            _rec("NCT003", primary_completion_date="2027-06-15"),  # valid
        ]
        audit = audit_future_events(records, AS_OF)
        assert audit["flag_summary"]["malformed_count"] == 2

    def test_malformed_samples_bounded(self):
        """Malformed flag list is bounded to 50."""
        records = [
            _rec(f"NCT{i:08d}", primary_completion_date="bad-date")
            for i in range(100)
        ]
        audit = audit_future_events(records, AS_OF)
        assert len(audit["flags"]["malformed_dates"]) <= 50
        assert audit["flag_summary"]["malformed_count"] == 100

    def test_none_date_not_flagged(self):
        """None/missing dates are not malformed — they're just absent."""
        records = [_rec("NCT001", primary_completion_date=None)]
        audit = audit_future_events(records, AS_OF)
        assert audit["flag_summary"]["malformed_count"] == 0


# ---------------------------------------------------------------------------
# 3) Top-ticker breakdown
# ---------------------------------------------------------------------------


class TestAuditTopTickers:
    """Verify top-ticker-by-future-PCD ranking."""

    def test_top_tickers_ordered_by_count(self):
        records = [
            _rec("NCT001", ticker="BIIB", primary_completion_date="2027-01-01"),
            _rec("NCT002", ticker="BIIB", primary_completion_date="2027-06-01"),
            _rec("NCT003", ticker="BIIB", primary_completion_date="2028-01-01"),
            _rec("NCT004", ticker="ACAD", primary_completion_date="2027-03-01"),
        ]
        audit = audit_future_events(records, AS_OF, top_n=5)
        top = audit["top_tickers_by_future_pcd"]
        assert len(top) == 2
        assert top[0]["ticker"] == "BIIB"
        assert top[0]["future_pcd_count"] == 3
        assert top[1]["ticker"] == "ACAD"

    def test_earliest_pcd_tracked(self):
        records = [
            _rec("NCT001", ticker="ACAD", primary_completion_date="2028-01-01"),
            _rec("NCT002", ticker="ACAD", primary_completion_date="2027-03-01"),
        ]
        audit = audit_future_events(records, AS_OF)
        top = audit["top_tickers_by_future_pcd"]
        assert top[0]["earliest_future_pcd"] == "2027-03-01"


# ---------------------------------------------------------------------------
# 4) Deterministic output
# ---------------------------------------------------------------------------


class TestAuditDeterministicOutput:
    """Same inputs → identical JSON (sort_keys)."""

    def test_deterministic_json(self):
        records = [
            _rec("NCT001", ticker="BIIB", primary_completion_date="2027-01-01"),
            _rec("NCT002", ticker="ACAD", primary_completion_date="2027-06-01"),
        ]
        a1 = audit_future_events(records, AS_OF)
        a2 = audit_future_events(records, AS_OF)

        j1 = json.dumps(a1, indent=2, sort_keys=True)
        j2 = json.dumps(a2, indent=2, sort_keys=True)
        assert j1 == j2


# ---------------------------------------------------------------------------
# 5) PCD-after-CD flag
# ---------------------------------------------------------------------------


class TestAuditPcdAfterCd:
    """Flag trials where PCD is >30 days after CD (inconsistent)."""

    def test_pcd_after_cd_flagged(self):
        records = [
            _rec("NCT001",
                 primary_completion_date="2027-06-01",
                 completion_date="2027-03-01"),  # PCD 92 days after CD
        ]
        audit = audit_future_events(records, AS_OF)
        assert audit["flag_summary"]["pcd_after_cd_count"] == 1

    def test_small_gap_not_flagged(self):
        records = [
            _rec("NCT001",
                 primary_completion_date="2027-06-15",
                 completion_date="2027-06-01"),  # 14 days, within tolerance
        ]
        audit = audit_future_events(records, AS_OF)
        assert audit["flag_summary"]["pcd_after_cd_count"] == 0


# ---------------------------------------------------------------------------
# 6) Markdown format
# ---------------------------------------------------------------------------


class TestMarkdownFormat:
    """Verify markdown rendering doesn't crash and contains key sections."""

    def test_markdown_contains_sections(self):
        records = [
            _rec("NCT001", primary_completion_date="2027-06-01"),
        ]
        audit = audit_future_events(records, AS_OF)
        md = format_markdown(audit)
        assert "# CTGov Future Events Audit" in md
        assert "## Key Metrics" in md
        assert "## PCD Horizon Buckets" in md
        assert "## Flags" in md
        assert "## Top Tickers" in md


# ---------------------------------------------------------------------------
# 7) Date parser edge cases
# ---------------------------------------------------------------------------


class TestParseDateSafe:
    """Verify the safe date parser handles edge cases."""

    def test_yyyy_mm_dd(self):
        assert _parse_date_safe("2027-06-15") == date(2027, 6, 15)

    def test_yyyy_mm(self):
        assert _parse_date_safe("2027-06") == date(2027, 6, 1)

    def test_none(self):
        assert _parse_date_safe(None) is None

    def test_empty_string(self):
        assert _parse_date_safe("") is None

    def test_garbage(self):
        assert _parse_date_safe("not-a-date") is None

    def test_integer_input(self):
        assert _parse_date_safe(12345) is None
