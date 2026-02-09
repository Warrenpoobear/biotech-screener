"""Tests for scripts/audit_drawdown_coverage.py."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.audit_drawdown_coverage import (
    audit_drawdown_coverage,
    format_audit_markdown,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_rankings(path: Path, rows: list[dict]) -> None:
    """Write a minimal rankings.csv."""
    cols = [
        "ticker", "archetype", "de_drawdown", "de_drawdown_missing_reason",
        "eligible", "tier_dev", "composite_score", "composite_rank",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            out = {c: row.get(c, "") for c in cols}
            writer.writerow(out)


def _write_price_csv(path: Path, rows: list[dict]) -> None:
    """Write a minimal price_history.csv."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ticker", "date", "close"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _dev_row(ticker: str, drawdown: str = "-0.20", reason: str = "") -> dict:
    return {
        "ticker": ticker,
        "archetype": "drug_developer",
        "de_drawdown": drawdown,
        "de_drawdown_missing_reason": reason,
        "eligible": "1",
        "tier_dev": "B",
        "composite_score": "50",
        "composite_rank": "100",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAuditDrawdownCoverage:

    def test_basic_counts(self):
        """Correct present / missing counts with reason column."""
        with tempfile.TemporaryDirectory() as tmp:
            rank_path = Path(tmp) / "rankings.csv"
            rows = [_dev_row(f"T{i}", drawdown="-0.10", reason="") for i in range(8)]
            rows += [_dev_row(f"M{i}", drawdown="", reason="no_price_series") for i in range(2)]
            _write_rankings(rank_path, rows)

            result = audit_drawdown_coverage(rank_path, None, snapshot_date="2026-01-01")

        assert result["total_dev_tickers"] == 10
        assert result["drawdown_present"] == 8
        assert result["drawdown_missing"] == 2
        assert result["coverage_pct"] == 80.0
        assert result["missing_reason_counts"]["no_price_series"] == 2

    def test_series_too_short_reason(self):
        """series_too_short reason is counted correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            rank_path = Path(tmp) / "rankings.csv"
            rows = [_dev_row("A", drawdown="-0.10", reason="")]
            rows += [_dev_row("B", drawdown="", reason="series_too_short")]
            rows += [_dev_row("C", drawdown="", reason="no_price_series")]
            _write_rankings(rank_path, rows)

            result = audit_drawdown_coverage(rank_path, None)

        assert result["missing_reason_counts"] == {
            "series_too_short": 1,
            "no_price_series": 1,
        }
        assert "B" in result["missing_tickers_by_reason"]["series_too_short"]
        assert "C" in result["missing_tickers_by_reason"]["no_price_series"]

    def test_price_history_stats(self):
        """Price history stats are computed when CSV is available."""
        with tempfile.TemporaryDirectory() as tmp:
            rank_path = Path(tmp) / "rankings.csv"
            rows = [_dev_row("A", drawdown="-0.10")]
            _write_rankings(rank_path, rows)

            price_path = Path(tmp) / "price_history.csv"
            prices = [
                {"ticker": "A", "date": "2025-01-01", "close": "100"},
                {"ticker": "A", "date": "2025-01-02", "close": "101"},
                {"ticker": "B", "date": "2025-01-01", "close": "50"},
            ]
            _write_price_csv(price_path, prices)

            result = audit_drawdown_coverage(rank_path, price_path)

        ph = result["price_history_stats"]
        assert ph["available"] is True
        assert ph["total_tickers"] == 2
        assert ph["date_range"] == ["2025-01-01", "2025-01-02"]

    def test_all_present(self):
        """100% coverage returns empty missing_reason_counts."""
        with tempfile.TemporaryDirectory() as tmp:
            rank_path = Path(tmp) / "rankings.csv"
            rows = [_dev_row(f"T{i}", drawdown="-0.10", reason="") for i in range(5)]
            _write_rankings(rank_path, rows)

            result = audit_drawdown_coverage(rank_path, None)

        assert result["coverage_pct"] == 100.0
        assert result["missing_reason_counts"] == {}

    def test_markdown_output(self):
        """Markdown output contains expected headings."""
        with tempfile.TemporaryDirectory() as tmp:
            rank_path = Path(tmp) / "rankings.csv"
            rows = [_dev_row("A", drawdown="-0.10")]
            rows += [_dev_row("B", drawdown="", reason="no_price_series")]
            _write_rankings(rank_path, rows)

            result = audit_drawdown_coverage(rank_path, None)

        md = format_audit_markdown(result)
        assert "# Drawdown Coverage Audit" in md
        assert "## Missing Reasons" in md
        assert "no_price_series" in md

    def test_excludes_non_dev(self):
        """Non drug_developer rows are excluded from the audit."""
        with tempfile.TemporaryDirectory() as tmp:
            rank_path = Path(tmp) / "rankings.csv"
            rows = [_dev_row("A", drawdown="-0.10")]
            rows += [{
                "ticker": "COMM",
                "archetype": "commercial_biotech",
                "de_drawdown": "",
                "de_drawdown_missing_reason": "no_price_series",
                "eligible": "1",
            }]
            _write_rankings(rank_path, rows)

            result = audit_drawdown_coverage(rank_path, None)

        assert result["total_dev_tickers"] == 1
        assert result["coverage_pct"] == 100.0
