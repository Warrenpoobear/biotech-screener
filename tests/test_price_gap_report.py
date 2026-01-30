#!/usr/bin/env python3
"""Unit test for price gap report."""

import json
import tempfile
from pathlib import Path

from wake_robin_data_pipeline.price_gap_report import build_report


class TestPriceGapReport:
    """Tests for gap report core logic."""

    def test_coverage_thresholds_and_missing(self):
        """Test 120-row ticker passes, 50-row fails, missing flagged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create price CSV: GOOD has 120 rows, SHORT has 50 rows
            price_path = Path(tmpdir) / "prices.csv"
            with open(price_path, "w") as f:
                f.write("date,ticker,close\n")
                for i in range(120):
                    f.write(f"2025-{(i // 28) + 1:02d}-{(i % 28) + 1:02d},GOOD,100.0\n")
                for i in range(50):
                    f.write(f"2025-{(i // 28) + 1:02d}-{(i % 28) + 1:02d},SHORT,50.0\n")

            # Create universe with GOOD, SHORT, and MISSING
            universe_path = Path(tmpdir) / "universe.json"
            with open(universe_path, "w") as f:
                json.dump({"active_securities": [
                    {"ticker": "GOOD"}, {"ticker": "SHORT"}, {"ticker": "MISSING"}
                ]}, f)

            report = build_report(
                as_of="2025-12-31",
                price_file=str(price_path),
                universe_file=str(universe_path),
            )

            # Verify summary
            assert report["summary"]["universe_tickers"] == 3
            assert report["summary"]["present_in_prices"] == 2
            assert report["summary"]["missing"] == 1
            assert report["summary"]["ok_120"] == 1
            assert report["summary"]["blocking_120"] == 1

            # Verify by_ticker
            assert report["by_ticker"]["GOOD"]["ok_120"] is True
            assert report["by_ticker"]["GOOD"]["blocking_reason"] is None
            assert report["by_ticker"]["SHORT"]["ok_120"] is False
            assert report["by_ticker"]["SHORT"]["blocking_reason"] == "insufficient_rows_120"
            assert report["by_ticker"]["MISSING"]["blocking_reason"] == "missing_ticker"

            # Verify blocking_tickers order (missing first, then blocking)
            assert "MISSING" in report["blocking_tickers"]
            assert "SHORT" in report["blocking_tickers"]
            assert "GOOD" not in report["blocking_tickers"]
