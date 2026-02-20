"""Tests for institutional_summary.build_institutional_summary()."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from institutional_summary import build_institutional_summary, _pad_cik


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_cache(tmp_path, as_of_date, managers_data):
    """Write synthetic 13F PIT cache matching real schema.

    managers_data: list of (cik_padded, manager_name, holdings_list)
      where holdings_list: list of {cusip, ticker, issuer, shares, value_usd_thousands, put_call}
    """
    date_dir = tmp_path / as_of_date
    date_dir.mkdir(parents=True, exist_ok=True)
    (date_dir / "managers").mkdir(exist_ok=True)

    mgr_entries = []
    for cik, name, holdings in managers_data:
        mgr_entries.append({
            "manager_cik": cik,
            "manager_name": name,
            "selected": True,
            "period_of_report": "2025-12-31",
            "filed_at": "2026-02-14",
            "form_type": "13F-HR",
            "accession": "0000000000-00-000000",
            "holdings_count": len(holdings),
            "manager_json_path": f"managers/{cik}.json",
            "raw_xml_path": "",
            "rejection_reason": "",
        })
        mgr_json = {
            "manager_cik": cik,
            "manager_name": name,
            "as_of_date": as_of_date,
            "period_of_report": "2025-12-31",
            "filed_at": "2026-02-14",
            "form_type": "13F-HR",
            "accession": "0000000000-00-000000",
            "holdings": holdings,
        }
        with open(date_dir / "managers" / f"{cik}.json", "w") as f:
            json.dump(mgr_json, f)

    index = {
        "schema_version": "sec_13f_pit_index.v1",
        "cache_type": "sec_13f_pit",
        "as_of_date": as_of_date,
        "created_at": "2026-02-19T12:00:00Z",
        "elite_only": True,
        "total_managers": len(managers_data),
        "managers_with_filing": len(managers_data),
        "coverage_pct": 100.0,
        "selection_policy": {"pit_rule": "test", "recent_filings_lookback_n": 8},
        "managers": sorted(mgr_entries, key=lambda m: m["manager_cik"]),
    }
    with open(date_dir / "index.json", "w") as f:
        json.dump(index, f)

    return tmp_path


def _holding(ticker, shares=1000, value=100, cusip="000000000", issuer="Test Inc"):
    return {
        "cusip": cusip,
        "ticker": ticker,
        "issuer": issuer,
        "shares": shares,
        "value_usd_thousands": value,
        "put_call": "",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSchemaContract:
    """Output has all required top-level and per-ticker keys."""

    def test_required_top_level_keys(self, tmp_path):
        _write_cache(tmp_path, "2026-01-01", [
            ("0000000001", "Mgr A", [_holding("AAAA", shares=500)]),
        ])
        result = build_institutional_summary("2026-01-01", {"AAAA", "BBBB"}, cache_base_dir=tmp_path)
        assert result is not None
        expected_keys = {
            "version", "as_of_date", "cache_as_of_date", "cache_schema_version",
            "elite_managers_total", "elite_managers_with_filing",
            "tickers_with_signal", "tickers_in_universe", "signal_coverage_pct",
            "tickers",
        }
        assert set(result.keys()) == expected_keys

    def test_required_per_ticker_keys(self, tmp_path):
        _write_cache(tmp_path, "2026-01-01", [
            ("0000000001", "Mgr A", [_holding("AAAA")]),
        ])
        result = build_institutional_summary("2026-01-01", {"AAAA"}, cache_base_dir=tmp_path)
        per_ticker_keys = {
            "elite_holders_count", "elite_holder_names",
            "elite_total_shares", "elite_total_value_usd_thousands",
            "inst_score_raw", "inst_score_z",
        }
        assert set(result["tickers"]["AAAA"].keys()) == per_ticker_keys


class TestDeterminism:
    """Same inputs produce byte-identical JSON output."""

    def test_byte_identical_output(self, tmp_path):
        _write_cache(tmp_path, "2026-01-01", [
            ("0000000002", "Mgr B", [_holding("BBBB"), _holding("AAAA")]),
            ("0000000001", "Mgr A", [_holding("AAAA"), _holding("CCCC")]),
        ])
        uni = {"AAAA", "BBBB", "CCCC"}
        r1 = build_institutional_summary("2026-01-01", uni, cache_base_dir=tmp_path)
        r2 = build_institutional_summary("2026-01-01", uni, cache_base_dir=tmp_path)
        j1 = json.dumps(r1, indent=2, sort_keys=False)
        j2 = json.dumps(r2, indent=2, sort_keys=False)
        assert j1 == j2

    def test_ticker_keys_sorted(self, tmp_path):
        _write_cache(tmp_path, "2026-01-01", [
            ("0000000001", "Mgr A", [_holding("ZZZZ"), _holding("AAAA")]),
        ])
        result = build_institutional_summary("2026-01-01", {"ZZZZ", "AAAA", "MMMM"}, cache_base_dir=tmp_path)
        assert list(result["tickers"].keys()) == ["AAAA", "MMMM", "ZZZZ"]


class TestZScore:
    """Z-score computation with ddof=0."""

    def test_known_z_values(self, tmp_path):
        # 3 tickers: A held by 2, B held by 1, C held by 0
        _write_cache(tmp_path, "2026-01-01", [
            ("0000000001", "Mgr A", [_holding("AAAA"), _holding("BBBB")]),
            ("0000000002", "Mgr B", [_holding("AAAA")]),
        ])
        result = build_institutional_summary("2026-01-01", {"AAAA", "BBBB", "CCCC"}, cache_base_dir=tmp_path)
        # raw scores: A=2, B=1, C=0  → mean=1.0, std=sqrt(2/3)
        mean = 1.0
        std = math.sqrt(2.0 / 3.0)
        assert result["tickers"]["AAAA"]["inst_score_z"] == pytest.approx((2 - mean) / std, abs=1e-3)
        assert result["tickers"]["BBBB"]["inst_score_z"] == pytest.approx((1 - mean) / std, abs=1e-3)
        assert result["tickers"]["CCCC"]["inst_score_z"] == pytest.approx((0 - mean) / std, abs=1e-3)

    def test_all_same_scores_z_is_zero(self, tmp_path):
        # All tickers held by 1 manager → std=0 → z=0
        _write_cache(tmp_path, "2026-01-01", [
            ("0000000001", "Mgr A", [_holding("AAAA"), _holding("BBBB")]),
        ])
        result = build_institutional_summary("2026-01-01", {"AAAA", "BBBB"}, cache_base_dir=tmp_path)
        assert result["tickers"]["AAAA"]["inst_score_z"] == 0.0
        assert result["tickers"]["BBBB"]["inst_score_z"] == 0.0


class TestCoverage:
    """Tickers not held get elite_holders_count=0."""

    def test_unheld_tickers_included(self, tmp_path):
        _write_cache(tmp_path, "2026-01-01", [
            ("0000000001", "Mgr A", [_holding("AAAA")]),
        ])
        result = build_institutional_summary("2026-01-01", {"AAAA", "XXXX"}, cache_base_dir=tmp_path)
        assert result["tickers"]["XXXX"]["elite_holders_count"] == 0
        assert result["tickers"]["XXXX"]["elite_holder_names"] == []
        assert result["tickers"]["XXXX"]["elite_total_shares"] == 0
        assert result["tickers_with_signal"] == 1
        assert result["signal_coverage_pct"] == 50.0


class TestTop10Names:
    """If a ticker is held by >10 managers, only top 10 by shares appear."""

    def test_top_10_cap(self, tmp_path):
        managers = []
        for i in range(12):
            cik = f"{i + 1:010d}"
            managers.append((cik, f"Manager {i:02d}", [_holding("AAAA", shares=1000 * (12 - i))]))
        _write_cache(tmp_path, "2026-01-01", managers)
        result = build_institutional_summary("2026-01-01", {"AAAA"}, cache_base_dir=tmp_path)
        assert result["tickers"]["AAAA"]["elite_holders_count"] == 12
        assert len(result["tickers"]["AAAA"]["elite_holder_names"]) == 10


class TestGracefulDegradation:
    """Missing or malformed cache returns None."""

    def test_missing_cache_returns_none(self, tmp_path):
        result = build_institutional_summary("2099-01-01", {"AAAA"}, cache_base_dir=tmp_path)
        assert result is None

    def test_malformed_index_returns_none(self, tmp_path):
        date_dir = tmp_path / "2026-01-01"
        date_dir.mkdir(parents=True)
        (date_dir / "index.json").write_text("NOT JSON")
        result = build_institutional_summary("2026-01-01", {"AAAA"}, cache_base_dir=tmp_path)
        assert result is None

    def test_wrong_schema_returns_none(self, tmp_path):
        date_dir = tmp_path / "2026-01-01"
        date_dir.mkdir(parents=True)
        with open(date_dir / "index.json", "w") as f:
            json.dump({"schema_version": "wrong"}, f)
        result = build_institutional_summary("2026-01-01", {"AAAA"}, cache_base_dir=tmp_path)
        assert result is None


class TestEmptyUniverse:
    """Empty universe returns summary with 0 tickers."""

    def test_empty_universe(self, tmp_path):
        _write_cache(tmp_path, "2026-01-01", [
            ("0000000001", "Mgr A", [_holding("AAAA")]),
        ])
        result = build_institutional_summary("2026-01-01", set(), cache_base_dir=tmp_path)
        assert result is not None
        assert result["tickers_in_universe"] == 0
        assert result["tickers_with_signal"] == 0
        assert result["signal_coverage_pct"] == 0.0
        assert result["tickers"] == {}


class TestTickerMatching:
    """Only universe tickers included; non-universe holdings ignored."""

    def test_non_universe_excluded(self, tmp_path):
        _write_cache(tmp_path, "2026-01-01", [
            ("0000000001", "Mgr A", [_holding("AAAA"), _holding("XXXX")]),
        ])
        result = build_institutional_summary("2026-01-01", {"AAAA"}, cache_base_dir=tmp_path)
        assert "XXXX" not in result["tickers"]
        assert "AAAA" in result["tickers"]

    def test_empty_ticker_holdings_skipped(self, tmp_path):
        """Holdings with ticker="" are not matched."""
        _write_cache(tmp_path, "2026-01-01", [
            ("0000000001", "Mgr A", [_holding(""), _holding("AAAA")]),
        ])
        result = build_institutional_summary("2026-01-01", {"AAAA", ""}, cache_base_dir=tmp_path)
        # Empty string ticker should not appear (filtered by `if r.get("ticker")`)
        assert result["tickers"]["AAAA"]["elite_holders_count"] == 1


class TestSharesAggregation:
    """Multiple holdings lines for same ticker by same manager are summed."""

    def test_multi_line_same_ticker(self, tmp_path):
        holdings = [
            _holding("AAAA", shares=500, value=50),
            _holding("AAAA", shares=300, value=30),  # e.g. common + call
        ]
        _write_cache(tmp_path, "2026-01-01", [
            ("0000000001", "Mgr A", holdings),
        ])
        result = build_institutional_summary("2026-01-01", {"AAAA"}, cache_base_dir=tmp_path)
        assert result["tickers"]["AAAA"]["elite_total_shares"] == 800
        assert result["tickers"]["AAAA"]["elite_total_value_usd_thousands"] == 80
        assert result["tickers"]["AAAA"]["elite_holders_count"] == 1  # still 1 manager


class TestPadCik:
    """CIK padding helper."""

    def test_pad_short(self):
        assert _pad_cik("1263508") == "0001263508"

    def test_pad_already_padded(self):
        assert _pad_cik("0001263508") == "0001263508"

    def test_pad_leading_zeros(self):
        assert _pad_cik("0000001") == "0000000001"
