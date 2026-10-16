"""Tests for collect_short_interest.py — short interest provenance and null handling.

Covers:
- report_date derives from Yahoo's dateShortInterest in UTC (not local time)
- report_date is None when vintage is unknown, never today's date
- genuine zero values are preserved rather than coerced to None
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from collect_short_interest import get_short_interest_data  # noqa: E402

# 2026-07-31 00:00 UTC — a real FINRA settlement date (last business day of July).
# In US/Eastern this instant is 2026-07-30 20:00, which is what a naive
# fromtimestamp() would render.
EPOCH_2026_07_31_UTC = 1785456000


def _info(**overrides):
    base = {
        "shortPercentOfFloat": 0.1523,
        "shortRatio": 7.21,
        "sharesShort": 49768882,
        "sharesShortPriorMonth": 48000000,
        "heldPercentInstitutions": 0.65,
        "averageVolume": 12345678,
        "dateShortInterest": EPOCH_2026_07_31_UTC,
    }
    base.update(overrides)
    return base


def _patched(info):
    ticker_obj = MagicMock()
    ticker_obj.info = info
    return patch("collect_short_interest.yf.Ticker", return_value=ticker_obj)


class TestReportDateTimezone:
    """dateShortInterest is midnight UTC; rendering it locally shifts the date back a day."""

    def test_report_date_is_utc_settlement_date(self):
        with _patched(_info()):
            rec = get_short_interest_data("MRNA")

        assert rec["report_date"] == "2026-07-31"

    def test_report_date_is_not_the_local_time_rendering(self):
        """Regression: the naive conversion produced 2026-07-30, not a settlement date."""
        naive = datetime.fromtimestamp(EPOCH_2026_07_31_UTC).strftime("%Y-%m-%d")
        utc = datetime.fromtimestamp(EPOCH_2026_07_31_UTC, tz=timezone.utc).strftime("%Y-%m-%d")

        with _patched(_info()):
            rec = get_short_interest_data("MRNA")

        assert rec["report_date"] == utc
        # Only meaningful where local time is behind UTC, which is where this bug bit.
        if naive != utc:
            assert rec["report_date"] != naive


class TestReportDateUnknownVintage:
    """Unknown vintage must stay unknown — never today's date."""

    def test_missing_date_yields_none_not_today(self):
        with _patched(_info(dateShortInterest=None)):
            rec = get_short_interest_data("MRNA")

        assert rec["report_date"] is None
        assert rec["report_date"] != date.today().isoformat()

    def test_unparseable_date_yields_none(self):
        with _patched(_info(dateShortInterest="not-an-epoch")):
            rec = get_short_interest_data("MRNA")

        assert rec["report_date"] is None

    def test_data_still_returned_when_vintage_unknown(self):
        """A missing date must not discard the short interest values themselves."""
        with _patched(_info(dateShortInterest=None)):
            rec = get_short_interest_data("MRNA")

        assert rec["short_interest_pct"] == 15.2
        assert rec["report_date"] is None


class TestZeroValuesPreserved:
    """A genuine 0 is falsy; it must not be recorded as missing."""

    def test_zero_short_interest_pct_preserved(self):
        with _patched(_info(shortPercentOfFloat=0.0)):
            rec = get_short_interest_data("MRNA")

        assert rec["short_interest_pct"] == 0.0
        assert rec["short_interest_pct"] is not None

    def test_zero_days_to_cover_preserved(self):
        with _patched(_info(shortRatio=0.0)):
            rec = get_short_interest_data("MRNA")

        assert rec["days_to_cover"] == 0.0

    def test_zero_institutional_pct_preserved(self):
        with _patched(_info(heldPercentInstitutions=0.0)):
            rec = get_short_interest_data("MRNA")

        assert rec["institutional_long_pct"] == 0.0

    @pytest.mark.parametrize("field", ["short_interest_pct", "days_to_cover", "institutional_long_pct"])
    def test_none_inputs_still_yield_none(self, field):
        """Absent values must remain None — the fix must not invent zeros."""
        with _patched(_info(shortPercentOfFloat=None, shortRatio=None, heldPercentInstitutions=None, sharesShort=123)):
            rec = get_short_interest_data("MRNA")

        assert rec[field] is None


class TestSkipCondition:
    def test_returns_none_when_no_short_data_at_all(self):
        with _patched(_info(shortPercentOfFloat=None, sharesShort=None)):
            assert get_short_interest_data("MRNA") is None
