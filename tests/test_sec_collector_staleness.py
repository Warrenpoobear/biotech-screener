#!/usr/bin/env python3
"""
Unit test for SEC collector staleness filtering (PIT-anchored).

Verifies extract_latest_metric filters stale data based on as_of_dt.
"""

from datetime import datetime
from decimal import Decimal

import pytest

from wake_robin_data_pipeline.collectors.sec_collector import extract_latest_metric


class TestStalenessAnchoring:
    """Tests for PIT-safe staleness filtering."""

    def test_keeps_fresh_drops_stale(self):
        """Fresh data kept, stale data (>365d) dropped."""
        # Fake SEC facts with two data points for same metric
        facts = {
            "facts": {
                "us-gaap": {
                    "Assets": {
                        "units": {
                            "USD": [
                                {"end": "2025-09-30", "val": 90000000000},  # Fresh
                                {"end": "2024-01-01", "val": 80000000000},  # Stale
                            ]
                        }
                    }
                }
            }
        }
        as_of = datetime(2026, 1, 29)

        val, end_date = extract_latest_metric(
            facts, "Assets", max_age_days=365, as_of_dt=as_of
        )

        # Should return the fresh 2025-09-30 value
        assert val == 90000000000
        assert end_date == "2025-09-30"

    def test_returns_none_if_only_stale(self):
        """Returns (None, None) if all data is stale."""
        facts = {
            "facts": {
                "us-gaap": {
                    "Assets": {
                        "units": {
                            "USD": [
                                {"end": "2024-01-01", "val": 80000000000},  # >365d stale
                            ]
                        }
                    }
                }
            }
        }
        as_of = datetime(2026, 1, 29)

        val, end_date = extract_latest_metric(
            facts, "Assets", max_age_days=365, as_of_dt=as_of
        )

        assert val is None
        assert end_date is None

    def test_no_filter_without_as_of_dt(self):
        """Without as_of_dt, no staleness filtering applied."""
        facts = {
            "facts": {
                "us-gaap": {
                    "Assets": {
                        "units": {
                            "USD": [
                                {"end": "2020-01-01", "val": 50000000000},  # Very old
                            ]
                        }
                    }
                }
            }
        }

        # No as_of_dt = no filtering
        val, end_date = extract_latest_metric(
            facts, "Assets", max_age_days=365, as_of_dt=None
        )

        # Should return the data (no filter applied)
        assert val == 50000000000
        assert end_date == "2020-01-01"
