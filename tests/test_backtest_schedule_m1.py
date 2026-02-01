"""Tests for weekly rebalance schedule generation."""
from datetime import date

import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.panel_builder_m1 import generate_rebalance_dates


class TestGenerateRebalanceDates:
    """Weekly date generation is deterministic and correct."""

    def test_basic_friday_schedule(self):
        """4-week span produces expected Fridays."""
        dates = generate_rebalance_dates(
            start=date(2025, 1, 3),  # Friday
            end=date(2025, 1, 31),
            weekday="FRI",
        )
        expected = [
            date(2025, 1, 3),
            date(2025, 1, 10),
            date(2025, 1, 17),
            date(2025, 1, 24),
            date(2025, 1, 31),
        ]
        assert dates == expected

    def test_start_mid_week(self):
        """Start on a Wednesday — first Friday should still be included."""
        dates = generate_rebalance_dates(
            start=date(2025, 1, 1),  # Wednesday
            end=date(2025, 1, 17),
            weekday="FRI",
        )
        assert dates[0] == date(2025, 1, 3)
        assert len(dates) == 3

    def test_monday_schedule(self):
        """Non-Friday weekday works."""
        dates = generate_rebalance_dates(
            start=date(2025, 1, 6),  # Monday
            end=date(2025, 1, 20),
            weekday="MON",
        )
        expected = [date(2025, 1, 6), date(2025, 1, 13), date(2025, 1, 20)]
        assert dates == expected

    def test_holiday_roll(self):
        """When a Friday is not a trading day, roll to previous trading day."""
        # Simulate: Jan 10 is not a trading day
        trading_days = {
            "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08",
            "2025-01-09",  # Thursday before missing Friday
            # "2025-01-10" — missing (holiday)
            "2025-01-13", "2025-01-14", "2025-01-15", "2025-01-16",
            "2025-01-17",
        }
        dates = generate_rebalance_dates(
            start=date(2025, 1, 3),
            end=date(2025, 1, 17),
            weekday="FRI",
            trading_days=trading_days,
        )
        assert date(2025, 1, 3) in dates
        # Jan 10 should roll to Jan 9 (Thursday)
        assert date(2025, 1, 9) in dates
        assert date(2025, 1, 10) not in dates
        assert date(2025, 1, 17) in dates

    def test_deterministic(self):
        """Same inputs always produce same output."""
        kwargs = dict(
            start=date(2024, 2, 2),
            end=date(2026, 2, 1),
            weekday="FRI",
        )
        a = generate_rebalance_dates(**kwargs)
        b = generate_rebalance_dates(**kwargs)
        assert a == b

    def test_two_year_count(self):
        """~104 weeks in 2 years."""
        dates = generate_rebalance_dates(
            start=date(2024, 2, 2),
            end=date(2026, 2, 1),
            weekday="FRI",
        )
        assert 100 <= len(dates) <= 106

    def test_all_dates_in_range(self):
        """No date outside [start, end]."""
        start = date(2025, 3, 1)
        end = date(2025, 3, 31)
        dates = generate_rebalance_dates(start, end, weekday="FRI")
        for d in dates:
            assert start <= d <= end

    def test_invalid_weekday_raises(self):
        with pytest.raises(ValueError):
            generate_rebalance_dates(date(2025, 1, 1), date(2025, 1, 31), weekday="XYZ")
