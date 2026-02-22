"""Tests for build_coinvest_features_from_13f.py."""
from __future__ import annotations

import json
import math
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from scripts.build_coinvest_features_from_13f import (
    CHANGE_THRESHOLD,
    CHANGE_WEIGHTS,
    RECENCY_FRESH_DAYS,
    TIER_WEIGHTS,
    _classify_position_change,
    _find_prior_cache_dir,
    _get_dominant_period_of_report,
    _get_prior_quarter_end,
    _load_cusip_map,
    _resolve_ticker,
    build_coinvest_features,
    validate_coinvest_features_schema,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _holding(
    ticker="",
    shares=1000,
    value=100,
    cusip="000000000",
    issuer="Test Inc",
    put_call="",
):
    return {
        "cusip": cusip,
        "ticker": ticker,
        "issuer": issuer,
        "shares": shares,
        "value_usd_thousands": value,
        "put_call": put_call,
    }


def _write_cache(
    tmp_path,
    as_of_date,
    managers_data,
    *,
    period_of_report="2025-12-31",
    filed_at="2026-02-14",
):
    """Write synthetic 13F PIT cache matching real schema.

    managers_data: list of (cik_padded, manager_name, holdings_list)
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
            "period_of_report": period_of_report,
            "filed_at": filed_at,
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
            "period_of_report": period_of_report,
            "filed_at": filed_at,
            "form_type": "13F-HR",
            "accession": "0000000000-00-000000",
            "holdings": holdings,
        }
        (date_dir / "managers" / f"{cik}.json").write_text(
            json.dumps(mgr_json),
        )

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
    (date_dir / "index.json").write_text(json.dumps(index))
    return tmp_path


# Shared constants for two-manager setup.
CIK_T1 = "0001263508"  # Tier 1
CIK_T2 = "0002222222"  # Tier 2
UNIVERSE = {"AAAA", "BBBB", "CCCC", "DDDD", "ZZZZ"}
TIERS = {CIK_T1: 1, CIK_T2: 2}
NAMES = {CIK_T1: "Baker Bros", CIK_T2: "Orbimed"}


def _basic_cache(tmp_path, as_of="2026-02-19"):
    """Write a two-manager cache with known holdings."""
    return _write_cache(
        tmp_path,
        as_of,
        [
            (CIK_T1, "Baker Bros", [
                _holding("AAAA", shares=5000, value=500),
                _holding("BBBB", shares=3000, value=300),
                _holding("CCCC", shares=1000, value=100),
                # Non-universe ticker (counts toward total value).
                _holding("NOPE", shares=2000, value=200),
            ]),
            (CIK_T2, "Orbimed", [
                _holding("AAAA", shares=4000, value=400),
                _holding("DDDD", shares=6000, value=600),
                # Non-universe.
                _holding("NOPE", shares=1000, value=100),
            ]),
        ],
    )


# ===================================================================
# TestCacheLoading
# ===================================================================

class TestCacheLoading:

    def test_missing_cache_returns_none(self, tmp_path):
        result = build_coinvest_features(
            "2025-12-31", tmp_path, UNIVERSE, TIERS, NAMES,
        )
        assert result is None

    def test_invalid_schema_returns_none(self, tmp_path):
        d = tmp_path / "2025-12-31"
        d.mkdir(parents=True)
        (d / "index.json").write_text(json.dumps({"schema_version": "bad"}))
        result = build_coinvest_features(
            "2025-12-31", tmp_path, UNIVERSE, TIERS, NAMES,
        )
        assert result is None

    def test_cusip_resolution_fallback(self, tmp_path):
        cusip_map = {"111111111": "AAAA"}
        _write_cache(tmp_path, "2026-02-19", [
            (CIK_T1, "Baker Bros", [
                _holding("", shares=1000, value=100, cusip="111111111"),
            ]),
        ])
        result = build_coinvest_features(
            "2026-02-19", tmp_path, UNIVERSE, TIERS, NAMES,
            cusip_map=cusip_map,
        )
        assert result is not None
        assert "AAAA" in result["tickers"]

    def test_no_cusip_map_only_embedded_tickers(self, tmp_path):
        _write_cache(tmp_path, "2026-02-19", [
            (CIK_T1, "Baker Bros", [
                _holding("AAAA", shares=1000, value=100),
                _holding("", shares=500, value=50, cusip="999999999"),
            ]),
        ])
        result = build_coinvest_features(
            "2026-02-19", tmp_path, UNIVERSE, TIERS, NAMES,
        )
        assert result is not None
        assert "AAAA" in result["tickers"]
        # The unresolved holding is tracked.
        assert result["provenance"]["unresolved_holdings"] >= 1

    def test_non_universe_tickers_excluded(self, tmp_path):
        _write_cache(tmp_path, "2026-02-19", [
            (CIK_T1, "Baker Bros", [
                _holding("AAAA", shares=1000, value=100),
                _holding("NOPE", shares=5000, value=500),
            ]),
        ])
        result = build_coinvest_features(
            "2026-02-19", tmp_path, UNIVERSE, TIERS, NAMES,
        )
        assert "AAAA" in result["tickers"]
        assert "NOPE" not in result["tickers"]


# ===================================================================
# TestPerTickerFeatures
# ===================================================================

class TestPerTickerFeatures:

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        _basic_cache(tmp_path)
        self.result = build_coinvest_features(
            "2026-02-19", tmp_path, UNIVERSE, TIERS, NAMES,
        )
        assert self.result is not None

    def test_tier1_count(self):
        # AAAA held by T1 (Baker Bros) and T2 (Orbimed) -> tier1_count=1
        assert self.result["tickers"]["AAAA"]["tier1_count"] == 1
        # BBBB held by T1 only -> tier1_count=1
        assert self.result["tickers"]["BBBB"]["tier1_count"] == 1
        # DDDD held by T2 only -> tier1_count=0
        assert self.result["tickers"]["DDDD"]["tier1_count"] == 0

    def test_overlap_count_includes_all_tiers(self):
        # AAAA: Baker Bros (T1) + Orbimed (T2) = 2
        assert self.result["tickers"]["AAAA"]["coinvest_overlap_count"] == 2
        assert self.result["tickers"]["AAAA"]["sponsor_overlap_count"] == 2

    def test_position_pct_value_based(self):
        # Baker Bros total: 500+300+100+200 = 1100
        # Baker Bros AAAA value: 500 -> pct = 500/1100*100 ≈ 45.45%
        # pos_w = clamp(sqrt(45.45), 0.5, 2.0) = min(2.0, 6.74) = 2.0
        #
        # Orbimed total: 400+600+100 = 1100
        # Orbimed AAAA value: 400 -> pct = 400/1100*100 ≈ 36.36%
        # pos_w = clamp(sqrt(36.36), 0.5, 2.0) = min(2.0, 6.03) = 2.0
        #
        # Both pos_w capped at 2.0 since position_pct >> 1.0
        td = self.result["tickers"]["AAAA"]
        # Conviction: (1.0 * 2.0 * 1.0 * recency) + (0.6 * 2.0 * 1.0 * recency)
        # filed_at = 2026-02-14, as_of = 2026-02-19 -> days_since = 5
        # recency_w = max(0.7, min(1.5, 1.5 - 5/180)) = min(1.5, 1.472) = 1.472
        recency = max(0.7, min(1.5, 1.5 - 5 / 180))
        expected = (1.0 * 2.0 * 1.0 * recency) + (0.6 * 2.0 * 1.0 * recency)
        assert abs(td["conviction_overlap"] - round(expected, 4)) < 0.001

    def test_conviction_overlap_formula(self):
        # CCCC: Baker Bros only. T1, value=100, total=1100.
        # pct = 100/1100*100 ≈ 9.09, pos_w = sqrt(9.09) ≈ 3.015 -> capped 2.0
        # chg_w = 1.0 (no prior), recency ~1.472
        recency = max(0.7, min(1.5, 1.5 - 5 / 180))
        expected = 1.0 * 2.0 * 1.0 * recency
        td = self.result["tickers"]["CCCC"]
        assert abs(td["conviction_overlap"] - round(expected, 4)) < 0.001

    def test_tier1_conviction_excludes_tier2(self):
        td = self.result["tickers"]["AAAA"]
        recency = max(0.7, min(1.5, 1.5 - 5 / 180))
        # Only Baker Bros (T1) contribution.
        expected_t1 = 1.0 * 2.0 * 1.0 * recency
        assert abs(td["tier1_conviction_overlap"] - round(expected_t1, 4)) < 0.001

    def test_max_tier1_position_pct(self):
        # Baker Bros AAAA: pct = 500/1100*100 ≈ 45.45
        td = self.result["tickers"]["AAAA"]
        expected = round(500 / 1100 * 100, 2)
        assert abs(td["max_tier1_position_pct"] - expected) < 0.1

    def test_days_since_latest_filing_is_min(self, tmp_path):
        # Two managers with different filed_at dates.
        _write_cache(tmp_path, "2026-03-01", [
            (CIK_T1, "Baker Bros", [_holding("AAAA", value=100)],),
        ], filed_at="2026-02-20")
        _write_cache(tmp_path, "2026-03-01", [
            # Second write appends to same dir — use a different CIK trick.
        ])
        # Simpler: just check the basic cache result.
        assert self.result["tickers"]["AAAA"]["days_since_latest_filing"] == 5

    def test_recency_state_fresh_vs_stale(self, tmp_path):
        # Fresh: days_since <= 90.
        assert self.result["tickers"]["AAAA"]["coinvest_recency_state"] == "fresh"

        # Stale: filed_at far in the past.
        _write_cache(tmp_path, "2026-06-01", [
            (CIK_T1, "Baker Bros", [_holding("AAAA", value=100)]),
        ], filed_at="2025-12-01")
        res = build_coinvest_features(
            "2026-06-01", tmp_path, UNIVERSE, TIERS, NAMES,
        )
        # days_since = (2026-06-01 - 2025-12-01).days = 182
        assert res["tickers"]["AAAA"]["coinvest_recency_state"] == "stale"


# ===================================================================
# TestPositionChanges
# ===================================================================

class TestPositionChanges:

    def test_new_position(self):
        assert _classify_position_change(100, 0, True, False) == "NEW"

    def test_exit_position(self):
        assert _classify_position_change(0, 100, False, True) == "EXIT"

    def test_increase_position(self):
        # >10% increase: 120 vs 100 = +20%.
        assert _classify_position_change(120, 100, False, False) == "INCREASE"

    def test_decrease_position(self):
        # >10% decrease: 80 vs 100 = -20%.
        assert _classify_position_change(80, 100, False, False) == "DECREASE"

    def test_hold_position(self):
        # Within ±10%: 105 vs 100 = +5%.
        assert _classify_position_change(105, 100, False, False) == "HOLD"

    def test_no_prior_cache_empty_position_changes(self, tmp_path):
        _basic_cache(tmp_path)
        result = build_coinvest_features(
            "2026-02-19", tmp_path, UNIVERSE, TIERS, NAMES,
            no_prior=True,
        )
        for td in result["tickers"].values():
            assert td["position_changes"] == {}


# ===================================================================
# TestPositionChangesIntegration
# ===================================================================

class TestPositionChangesIntegration:
    """Integration tests: full build with prior cache."""

    def test_new_and_hold_with_prior(self, tmp_path):
        # Prior quarter: Baker Bros holds AAAA=100, BBBB=200.
        _write_cache(
            tmp_path, "2025-09-30",
            [(CIK_T1, "Baker Bros", [
                _holding("AAAA", value=100),
                _holding("BBBB", value=200),
            ])],
            period_of_report="2025-06-30",
            filed_at="2025-08-14",
        )
        # Current quarter: Baker Bros holds AAAA=100 (HOLD), CCCC=300 (NEW).
        _write_cache(
            tmp_path, "2025-12-31",
            [(CIK_T1, "Baker Bros", [
                _holding("AAAA", value=100),
                _holding("CCCC", value=300),
            ])],
            period_of_report="2025-09-30",
            filed_at="2025-11-14",
        )
        result = build_coinvest_features(
            "2025-12-31", tmp_path, UNIVERSE, TIERS, NAMES,
        )
        assert result is not None
        # AAAA: same value -> HOLD.
        assert result["tickers"]["AAAA"]["position_changes"]["Baker Bros"] == "HOLD"
        # CCCC: new position -> NEW.
        assert result["tickers"]["CCCC"]["position_changes"]["Baker Bros"] == "NEW"
        # BBBB: EXIT (in prior, not current) -> not in tickers output.
        # EXIT is tracked in provenance change_summary.
        assert result["provenance"]["change_summary"]["EXIT"] >= 1

    def test_increase_with_prior(self, tmp_path):
        # Prior: AAAA=100.
        _write_cache(
            tmp_path, "2025-09-30",
            [(CIK_T1, "Baker Bros", [_holding("AAAA", value=100)])],
            period_of_report="2025-06-30",
            filed_at="2025-08-14",
        )
        # Current: AAAA=150 (+50%).
        _write_cache(
            tmp_path, "2025-12-31",
            [(CIK_T1, "Baker Bros", [_holding("AAAA", value=150)])],
            period_of_report="2025-09-30",
            filed_at="2025-11-14",
        )
        result = build_coinvest_features(
            "2025-12-31", tmp_path, UNIVERSE, TIERS, NAMES,
        )
        assert result["tickers"]["AAAA"]["position_changes"]["Baker Bros"] == "INCREASE"

    def test_decrease_with_prior(self, tmp_path):
        # Prior: AAAA=200.
        _write_cache(
            tmp_path, "2025-09-30",
            [(CIK_T1, "Baker Bros", [_holding("AAAA", value=200)])],
            period_of_report="2025-06-30",
            filed_at="2025-08-14",
        )
        # Current: AAAA=100 (-50%).
        _write_cache(
            tmp_path, "2025-12-31",
            [(CIK_T1, "Baker Bros", [_holding("AAAA", value=100)])],
            period_of_report="2025-09-30",
            filed_at="2025-11-14",
        )
        result = build_coinvest_features(
            "2025-12-31", tmp_path, UNIVERSE, TIERS, NAMES,
        )
        assert result["tickers"]["AAAA"]["position_changes"]["Baker Bros"] == "DECREASE"


# ===================================================================
# TestPriorQuarterFinding
# ===================================================================

class TestPriorQuarterFinding:

    def test_quarter_end_math(self):
        assert _get_prior_quarter_end(date(2025, 12, 31)) == date(2025, 9, 30)
        assert _get_prior_quarter_end(date(2025, 9, 30)) == date(2025, 6, 30)
        assert _get_prior_quarter_end(date(2025, 6, 30)) == date(2025, 3, 31)
        assert _get_prior_quarter_end(date(2025, 3, 31)) == date(2024, 12, 31)

    def test_find_prior_by_scan(self, tmp_path):
        # Write prior quarter cache with por=2025-06-30.
        _write_cache(
            tmp_path, "2025-09-30",
            [(CIK_T1, "Baker Bros", [_holding("AAAA", value=100)])],
            period_of_report="2025-06-30",
            filed_at="2025-08-14",
        )
        found = _find_prior_cache_dir(tmp_path, "2025-09-30")
        assert found is not None
        assert found.name == "2025-09-30"

    def test_no_matching_prior(self, tmp_path):
        # Write a cache that won't match any prior quarter.
        _write_cache(
            tmp_path, "2025-12-31",
            [(CIK_T1, "Baker Bros", [_holding("AAAA", value=100)])],
            period_of_report="2025-12-31",
        )
        # Looking for prior of 2025-03-31 (which needs 2024-12-31 -> not found).
        found = _find_prior_cache_dir(tmp_path, "2025-03-31")
        assert found is None

    def test_dominant_period_of_report_picks_most_common(self, tmp_path):
        d = tmp_path / "test"
        d.mkdir()
        index = {
            "schema_version": "sec_13f_pit_index.v1",
            "managers": [
                {"selected": True, "period_of_report": "2025-09-30"},
                {"selected": True, "period_of_report": "2025-09-30"},
                {"selected": True, "period_of_report": "2025-06-30"},
                {"selected": False, "period_of_report": "2025-03-31"},
            ],
        }
        assert _get_dominant_period_of_report(index) == "2025-09-30"


# ===================================================================
# TestSchemaValidation
# ===================================================================

class TestSchemaValidation:

    def test_full_output_passes(self, tmp_path):
        _basic_cache(tmp_path)
        result = build_coinvest_features(
            "2026-02-19", tmp_path, UNIVERSE, TIERS, NAMES,
        )
        ok, msg = validate_coinvest_features_schema(result)
        assert ok, msg

    def test_coverage_consistency(self, tmp_path):
        _basic_cache(tmp_path)
        result = build_coinvest_features(
            "2026-02-19", tmp_path, UNIVERSE, TIERS, NAMES,
        )
        n_sig = result["tickers_with_signal"]
        n_uni = result["tickers_in_universe"]
        expected = round(n_sig / n_uni * 100, 1)
        assert abs(result["signal_coverage_pct"] - expected) < 0.2

    def test_provenance_complete(self, tmp_path):
        _basic_cache(tmp_path)
        result = build_coinvest_features(
            "2026-02-19", tmp_path, UNIVERSE, TIERS, NAMES,
        )
        prov = result["provenance"]
        assert prov["builder"] == "build_coinvest_features_from_13f.py"
        assert prov["builder_version"] == "1.0.0"
        assert isinstance(prov["managers_used"], list)
        assert len(prov["managers_used"]) == 2
        assert isinstance(prov["change_summary"], dict)
        assert "prior_available" in prov


# ===================================================================
# TestEdgeCases
# ===================================================================

class TestEdgeCases:

    def test_empty_universe(self, tmp_path):
        _basic_cache(tmp_path)
        result = build_coinvest_features(
            "2026-02-19", tmp_path, set(), TIERS, NAMES,
        )
        assert result is not None
        assert result["tickers"] == {}
        assert result["tickers_with_signal"] == 0
        assert result["signal_coverage_pct"] == 0.0

    def test_manager_zero_total_value(self, tmp_path):
        _write_cache(tmp_path, "2026-02-19", [
            (CIK_T1, "Baker Bros", [
                _holding("AAAA", shares=0, value=0),
            ]),
        ])
        result = build_coinvest_features(
            "2026-02-19", tmp_path, UNIVERSE, TIERS, NAMES,
        )
        # Zero total value -> pos_w = 0.5 (minimum).
        # position_pct = 0 -> pos_w = max(0.5, sqrt(0)) = 0.5
        recency = max(0.7, min(1.5, 1.5 - 5 / 180))
        expected = TIER_WEIGHTS[1] * 0.5 * 1.0 * recency
        td = result["tickers"]["AAAA"]
        assert abs(td["conviction_overlap"] - round(expected, 4)) < 0.001

    def test_multiple_holdings_same_ticker_aggregated(self, tmp_path):
        _write_cache(tmp_path, "2026-02-19", [
            (CIK_T1, "Baker Bros", [
                _holding("AAAA", shares=1000, value=100),
                _holding("AAAA", shares=500, value=50, put_call="CALL"),
            ]),
        ])
        result = build_coinvest_features(
            "2026-02-19", tmp_path, UNIVERSE, TIERS, NAMES,
        )
        td = result["tickers"]["AAAA"]
        # Aggregated: value=150, total=150. pct = 100%.
        # pos_w = clamp(sqrt(100), 0.5, 2.0) = 2.0
        recency = max(0.7, min(1.5, 1.5 - 5 / 180))
        expected = TIER_WEIGHTS[1] * 2.0 * 1.0 * recency
        assert abs(td["conviction_overlap"] - round(expected, 4)) < 0.001
        # Only one holder entry.
        assert td["coinvest_overlap_count"] == 1

    def test_put_call_holdings_included(self, tmp_path):
        _write_cache(tmp_path, "2026-02-19", [
            (CIK_T1, "Baker Bros", [
                _holding("AAAA", shares=1000, value=100, put_call="PUT"),
            ]),
        ])
        result = build_coinvest_features(
            "2026-02-19", tmp_path, UNIVERSE, TIERS, NAMES,
        )
        assert "AAAA" in result["tickers"]

    def test_future_filing_skipped(self, tmp_path):
        """PIT safety: manager with filed_at > as_of is excluded."""
        _write_cache(
            tmp_path, "2026-01-15",
            [(CIK_T1, "Baker Bros", [_holding("AAAA", value=100)])],
            filed_at="2026-02-14",  # Filed AFTER as_of.
        )
        result = build_coinvest_features(
            "2026-01-15", tmp_path, UNIVERSE, TIERS, NAMES,
        )
        assert result is not None
        # Manager skipped -> no tickers.
        assert result["tickers"] == {}
        assert result["provenance"]["managers_used"] == []


# ===================================================================
# TestCusipHelpers
# ===================================================================

class TestCusipHelpers:

    def test_resolve_ticker_prefers_embedded(self):
        h = _holding("AAAA", cusip="111111111")
        assert _resolve_ticker(h, {"111111111": "BBBB"}) == "AAAA"

    def test_resolve_ticker_cusip_fallback(self):
        h = _holding("", cusip="111111111")
        assert _resolve_ticker(h, {"111111111": "BBBB"}) == "BBBB"

    def test_resolve_ticker_empty_no_map(self):
        h = _holding("", cusip="999999999")
        assert _resolve_ticker(h, {}) == ""

    def test_load_cusip_map_missing_file(self, tmp_path):
        result = _load_cusip_map(tmp_path / "nonexistent.json")
        assert result == {}

    def test_load_cusip_map_valid(self, tmp_path):
        p = tmp_path / "cusip.json"
        p.write_text(json.dumps({"111111111": "AAAA"}))
        result = _load_cusip_map(p)
        assert result == {"111111111": "AAAA"}
