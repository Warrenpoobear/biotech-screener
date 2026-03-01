"""Tests for wake_robin_data_pipeline.company_calendar_sources."""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from typing import Any, Dict, List

from wake_robin_data_pipeline.company_calendar_sources import (
    load_company_calendar_sources,
    validate_company_calendar_sources,
    merge_universe_with_company_calendar_sources,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_sources_file(tmp_path: Path, sources: Dict[str, Any]) -> Path:
    """Write a company_calendar_sources.json and return its path."""
    data = {
        "_schema": "company_calendar_sources.v1",
        "_updated_at": "2026-02-28",
        "sources": sources,
    }
    p = tmp_path / "company_calendar_sources.json"
    p.write_text(json.dumps(data, indent=2))
    return p


def _universe_row(ticker: str, ir_url: str = "", pr_rss_url: str = "") -> Dict[str, Any]:
    """Minimal universe row for testing."""
    row: Dict[str, Any] = {"ticker": ticker, "company_name": f"{ticker} Inc"}
    if ir_url:
        row["ir_url"] = ir_url
    if pr_rss_url:
        row["pr_rss_url"] = pr_rss_url
    return row


# ---------------------------------------------------------------------------
# load_company_calendar_sources
# ---------------------------------------------------------------------------

class TestLoadCompanyCalendarSources:
    def test_missing_file_returns_empty(self, tmp_path: Path):
        result = load_company_calendar_sources(tmp_path / "nonexistent.json")
        assert result == {}

    def test_valid_file_loads(self, tmp_path: Path):
        p = _make_sources_file(tmp_path, {
            "VRTX": {"ir_url": "https://investors.vrtx.com/events"},
            "REGN": {"ir_url": "https://investor.regeneron.com/events", "pr_rss_url": "https://newsroom.regeneron.com/rss.xml"},
        })
        result = load_company_calendar_sources(p)
        assert len(result) == 2
        assert result["VRTX"]["ir_url"] == "https://investors.vrtx.com/events"
        assert "pr_rss_url" not in result["VRTX"]
        assert result["REGN"]["pr_rss_url"] == "https://newsroom.regeneron.com/rss.xml"

    def test_normalizes_tickers_to_uppercase(self, tmp_path: Path):
        p = _make_sources_file(tmp_path, {
            "vrtx": {"ir_url": "https://example.com/vrtx"},
            " Regn ": {"ir_url": "https://example.com/regn"},
        })
        result = load_company_calendar_sources(p)
        assert "VRTX" in result
        assert "REGN" in result
        assert len(result) == 2

    def test_malformed_json_raises(self, tmp_path: Path):
        p = tmp_path / "company_calendar_sources.json"
        p.write_text("{bad json")
        with pytest.raises(json.JSONDecodeError):
            load_company_calendar_sources(p)

    def test_non_dict_sources_returns_empty(self, tmp_path: Path):
        p = tmp_path / "company_calendar_sources.json"
        p.write_text(json.dumps({"_schema": "company_calendar_sources.v1", "sources": []}))
        result = load_company_calendar_sources(p)
        assert result == {}

    def test_filters_non_string_values(self, tmp_path: Path):
        p = _make_sources_file(tmp_path, {
            "VRTX": {"ir_url": "https://example.com", "bad_field": 123},
        })
        result = load_company_calendar_sources(p)
        assert "ir_url" in result["VRTX"]
        assert "bad_field" not in result["VRTX"]


# ---------------------------------------------------------------------------
# validate_company_calendar_sources
# ---------------------------------------------------------------------------

class TestValidateCompanyCalendarSources:
    def test_valid_sources_no_warnings(self):
        sources = {
            "VRTX": {"ir_url": "https://investors.vrtx.com/events", "pr_rss_url": "https://investors.vrtx.com/rss/news.xml"},
        }
        warnings = validate_company_calendar_sources(sources)
        assert warnings == []

    def test_warns_non_http_url(self):
        sources = {"VRTX": {"ir_url": "ftp://bad.com/events"}}
        warnings = validate_company_calendar_sources(sources)
        assert len(warnings) == 1
        assert "http(s)" in warnings[0]

    def test_warns_rss_url_without_hint(self):
        sources = {"VRTX": {"pr_rss_url": "https://example.com/news"}}
        warnings = validate_company_calendar_sources(sources)
        assert len(warnings) == 1
        assert "rss/feed/.xml" in warnings[0]

    def test_rss_url_with_hint_ok(self):
        sources = {"VRTX": {"pr_rss_url": "https://example.com/rss/news"}}
        warnings = validate_company_calendar_sources(sources)
        assert warnings == []


# ---------------------------------------------------------------------------
# merge_universe_with_company_calendar_sources
# ---------------------------------------------------------------------------

class TestMergeUniverse:
    def test_adds_missing_urls(self):
        universe = [_universe_row("VRTX"), _universe_row("REGN")]
        sources = {
            "VRTX": {"ir_url": "https://vrtx.com/ir", "pr_rss_url": "https://vrtx.com/rss.xml"},
            "REGN": {"ir_url": "https://regn.com/ir"},
        }
        merged, stats = merge_universe_with_company_calendar_sources(universe, sources)
        assert merged[0]["ir_url"] == "https://vrtx.com/ir"
        assert merged[0]["pr_rss_url"] == "https://vrtx.com/rss.xml"
        assert merged[1]["ir_url"] == "https://regn.com/ir"
        assert "pr_rss_url" not in merged[1]
        assert stats["ir_url_added"] == 2
        assert stats["pr_rss_url_added"] == 1

    def test_never_overrides_existing_urls(self):
        universe = [_universe_row("VRTX", ir_url="https://original.com/ir", pr_rss_url="https://original.com/rss")]
        sources = {
            "VRTX": {"ir_url": "https://new.com/ir", "pr_rss_url": "https://new.com/rss"},
        }
        merged, stats = merge_universe_with_company_calendar_sources(universe, sources)
        assert merged[0]["ir_url"] == "https://original.com/ir"
        assert merged[0]["pr_rss_url"] == "https://original.com/rss"
        assert stats["ir_url_added"] == 0
        assert stats["pr_rss_url_added"] == 0

    def test_does_not_mutate_input(self):
        universe = [_universe_row("VRTX")]
        sources = {"VRTX": {"ir_url": "https://vrtx.com/ir"}}
        merged, _ = merge_universe_with_company_calendar_sources(universe, sources)
        assert "ir_url" not in universe[0]
        assert merged[0]["ir_url"] == "https://vrtx.com/ir"

    def test_stats_correct(self):
        universe = [
            _universe_row("VRTX", ir_url="https://existing.com/ir"),
            _universe_row("REGN"),
            _universe_row("GILD"),
        ]
        sources = {
            "VRTX": {"ir_url": "https://vrtx.com/ir"},
            "REGN": {"ir_url": "https://regn.com/ir", "pr_rss_url": "https://regn.com/rss.xml"},
            "MISSING": {"ir_url": "https://missing.com/ir"},
        }
        merged, stats = merge_universe_with_company_calendar_sources(universe, sources)

        assert stats["tickers_total"] == 3
        assert stats["ir_url_before"] == 1
        assert stats["ir_url_added"] == 1  # REGN only (VRTX already has, GILD not in sources)
        assert stats["ir_url_after"] == 2
        assert stats["pr_rss_url_before"] == 0
        assert stats["pr_rss_url_added"] == 1  # REGN
        assert stats["pr_rss_url_after"] == 1
        assert stats["unmapped_sources_count"] == 1
        assert stats["unmapped_sources"] == ["MISSING"]

    def test_empty_sources_no_change(self):
        universe = [_universe_row("VRTX")]
        merged, stats = merge_universe_with_company_calendar_sources(universe, {})
        assert len(merged) == 1
        assert "ir_url" not in merged[0]
        assert stats["ir_url_added"] == 0

    def test_unmapped_capped_at_50(self):
        universe = [_universe_row("VRTX")]
        # 60 sources not in universe
        sources = {f"FAKE{i:03d}": {"ir_url": f"https://fake{i}.com"} for i in range(60)}
        _, stats = merge_universe_with_company_calendar_sources(universe, sources)
        assert stats["unmapped_sources_count"] == 60
        assert len(stats["unmapped_sources"]) == 50

    def test_preserves_order(self):
        universe = [_universe_row("GILD"), _universe_row("VRTX"), _universe_row("REGN")]
        sources = {
            "VRTX": {"ir_url": "https://vrtx.com/ir"},
            "REGN": {"ir_url": "https://regn.com/ir"},
        }
        merged, _ = merge_universe_with_company_calendar_sources(universe, sources)
        assert [r["ticker"] for r in merged] == ["GILD", "VRTX", "REGN"]


# ---------------------------------------------------------------------------
# Integration: load + validate + merge
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_production_file_loads_and_validates(self):
        """Smoke test: production company_calendar_sources.json loads cleanly."""
        prod_path = Path(__file__).resolve().parent.parent / "production_data" / "company_calendar_sources.json"
        if not prod_path.exists():
            pytest.skip("production file not present")

        sources = load_company_calendar_sources(prod_path)
        assert len(sources) >= 10  # We have 15 tickers

        warnings = validate_company_calendar_sources(sources)
        assert len(warnings) == 0, f"Validation warnings: {warnings}"

        # All tickers should be uppercase
        for tk in sources:
            assert tk == tk.upper(), f"Ticker not uppercase: {tk}"

        # All ir_url values should be https
        for tk, urls in sources.items():
            if "ir_url" in urls:
                assert urls["ir_url"].startswith("https://"), f"{tk}.ir_url not https"

    def test_full_merge_pipeline(self, tmp_path: Path):
        """Load from file, validate, merge — full pipeline."""
        p = _make_sources_file(tmp_path, {
            "VRTX": {"ir_url": "https://vrtx.com/ir", "pr_rss_url": "https://vrtx.com/rss.xml"},
            "REGN": {"ir_url": "https://regn.com/ir"},
        })
        sources = load_company_calendar_sources(p)
        warnings = validate_company_calendar_sources(sources)
        assert warnings == []

        universe = [_universe_row("VRTX"), _universe_row("REGN"), _universe_row("GILD")]
        merged, stats = merge_universe_with_company_calendar_sources(universe, sources)

        assert len(merged) == 3
        assert merged[0]["ir_url"] == "https://vrtx.com/ir"
        assert merged[0]["pr_rss_url"] == "https://vrtx.com/rss.xml"
        assert merged[1]["ir_url"] == "https://regn.com/ir"
        assert "ir_url" not in merged[2]  # GILD not in sources
        assert stats["ir_url_added"] == 2
        assert stats["pr_rss_url_added"] == 1
