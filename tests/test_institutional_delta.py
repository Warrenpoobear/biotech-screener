"""Tests for institutional delta sidecar + gate + sort signal."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from institutional_summary import (
    DELTA_SCHEMA_VERSION,
    SCHEMA_VERSION,
    build_institutional_summary,
    compute_institutional_delta,
    _find_prior_institutional_summary,
    validate_institutional_summary_schema_v1,
    validate_institutional_summary_delta_schema_v1,
)
from run_daily_production import check_institutional_delta


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_summary(tickers: dict, as_of_date: str = "2026-02-20") -> dict:
    """Build a minimal institutional_summary dict for testing."""
    return {
        "schema_version": "institutional_summary.v1",
        "version": "1.0.0",
        "created_at": "2026-02-20T12:00:00Z",
        "as_of_date": as_of_date,
        "cache_as_of_date": as_of_date,
        "cache_schema_version": "sec_13f_pit_index.v1",
        "elite_managers_total": 29,
        "elite_managers_with_filing": 29,
        "tickers_with_signal": sum(1 for v in tickers.values() if v.get("elite_holders_count", 0) > 0),
        "tickers_in_universe": len(tickers),
        "signal_coverage_pct": 0.0,
        "tickers": tickers,
    }


def _ticker_data(
    holders_count: int = 0,
    holder_shares: dict | None = None,
    total_shares: int = 0,
    total_value: int = 0,
) -> dict:
    """Build per-ticker data block."""
    return {
        "elite_holders_count": holders_count,
        "elite_holder_names": sorted(list(holder_shares.keys()))[:10] if holder_shares else [],
        "elite_holder_shares": holder_shares if holder_shares is not None else {},
        "elite_total_shares": total_shares,
        "elite_total_value_usd_thousands": total_value,
        "inst_score_raw": holders_count,
        "inst_score_z": 0.0,
    }


# ---------------------------------------------------------------------------
# TestComputeDelta
# ---------------------------------------------------------------------------

class TestComputeDelta:
    """compute_institutional_delta() logic."""

    def test_new_and_exit_counts(self):
        current = _make_summary({
            "AAAA": _ticker_data(2, {"FundA": 100, "FundB": 200}, 300, 30),
        })
        prior = _make_summary({
            "AAAA": _ticker_data(2, {"FundA": 100, "FundC": 150}, 250, 25),
        }, as_of_date="2026-02-19")
        delta = compute_institutional_delta(current, prior)
        assert delta is not None
        tk = delta["tickers"]["AAAA"]
        assert tk["elite_new_count"] == 1       # FundB is new
        assert tk["elite_exit_count"] == 1      # FundC exited
        assert tk["net_elite_holders_delta"] == 0

    def test_add_and_trim_counts(self):
        current = _make_summary({
            "AAAA": _ticker_data(2, {"FundA": 200, "FundB": 50}, 250, 25),
        })
        prior = _make_summary({
            "AAAA": _ticker_data(2, {"FundA": 100, "FundB": 100}, 200, 20),
        }, as_of_date="2026-02-19")
        delta = compute_institutional_delta(current, prior)
        tk = delta["tickers"]["AAAA"]
        assert tk["elite_add_count"] == 1    # FundA increased
        assert tk["elite_trim_count"] == 1   # FundB decreased

    def test_shares_and_value_deltas(self):
        current = _make_summary({
            "AAAA": _ticker_data(1, {"FundA": 300}, 300, 30),
        })
        prior = _make_summary({
            "AAAA": _ticker_data(1, {"FundA": 200}, 200, 20),
        }, as_of_date="2026-02-19")
        delta = compute_institutional_delta(current, prior)
        tk = delta["tickers"]["AAAA"]
        assert tk["shares_delta_total"] == 100
        assert tk["value_delta_total"] == 10

    def test_ticker_not_in_prior_all_new(self):
        current = _make_summary({
            "AAAA": _ticker_data(2, {"FundA": 100, "FundB": 200}, 300, 30),
        })
        prior = _make_summary({}, as_of_date="2026-02-19")
        delta = compute_institutional_delta(current, prior)
        tk = delta["tickers"]["AAAA"]
        assert tk["elite_new_count"] == 2
        assert tk["elite_exit_count"] == 0
        assert tk["net_elite_holders_delta"] == 2
        assert tk["shares_delta_total"] == 300

    def test_both_empty_all_zeros(self):
        current = _make_summary({
            "AAAA": _ticker_data(0, {}, 0, 0),
        })
        prior = _make_summary({
            "AAAA": _ticker_data(0, {}, 0, 0),
        }, as_of_date="2026-02-19")
        delta = compute_institutional_delta(current, prior)
        tk = delta["tickers"]["AAAA"]
        assert tk["elite_new_count"] == 0
        assert tk["elite_exit_count"] == 0
        assert tk["elite_add_count"] == 0
        assert tk["elite_trim_count"] == 0
        assert tk["net_elite_holders_delta"] == 0

    def test_deterministic_output(self):
        tickers = {
            "BBBB": _ticker_data(1, {"FundA": 100}, 100, 10),
            "AAAA": _ticker_data(1, {"FundB": 200}, 200, 20),
        }
        current = _make_summary(tickers)
        prior = _make_summary(tickers, as_of_date="2026-02-19")
        d1 = compute_institutional_delta(current, prior)
        d2 = compute_institutional_delta(current, prior)
        d1.pop("created_at"); d2.pop("created_at")
        assert json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True)

    def test_v1_prior_returns_none(self):
        """Prior without elite_holder_shares → None."""
        current = _make_summary({
            "AAAA": _ticker_data(1, {"FundA": 100}, 100, 10),
        })
        # Build a v1-style prior (no elite_holder_shares)
        prior = _make_summary({
            "AAAA": {
                "elite_holders_count": 1,
                "elite_holder_names": ["FundA"],
                "elite_total_shares": 100,
                "elite_total_value_usd_thousands": 10,
                "inst_score_raw": 1,
                "inst_score_z": 0.0,
            },
        }, as_of_date="2026-02-19")
        delta = compute_institutional_delta(current, prior)
        assert delta is None

    def test_schema_version_correct(self):
        current = _make_summary({"AAAA": _ticker_data(1, {"FundA": 100}, 100, 10)})
        prior = _make_summary({"AAAA": _ticker_data(1, {"FundA": 100}, 100, 10)}, as_of_date="2026-02-19")
        delta = compute_institutional_delta(current, prior)
        assert delta["schema_version"] == DELTA_SCHEMA_VERSION

    def test_pit_provenance_in_delta(self):
        """Delta output includes PIT provenance fields."""
        current = _make_summary(
            {"AAAA": _ticker_data(1, {"FundA": 100}, 100, 10)},
            as_of_date="2026-02-20",
        )
        current["cache_as_of_date"] = "2026-02-20"
        prior = _make_summary(
            {"AAAA": _ticker_data(1, {"FundA": 100}, 100, 10)},
            as_of_date="2026-02-19",
        )
        prior["cache_as_of_date"] = "2026-02-19"
        delta = compute_institutional_delta(current, prior)
        assert delta is not None
        assert delta["current_cache_as_of_date"] == "2026-02-20"
        assert delta["prior_cache_as_of_date"] == "2026-02-19"

    def test_top_level_counts(self):
        current = _make_summary({
            "AAAA": _ticker_data(1, {"FundA": 100}, 100, 10),
            "BBBB": _ticker_data(0, {}, 0, 0),
        })
        prior = _make_summary({
            "AAAA": _ticker_data(1, {"FundA": 100}, 100, 10),
            "CCCC": _ticker_data(0, {}, 0, 0),
        }, as_of_date="2026-02-19")
        delta = compute_institutional_delta(current, prior)
        assert delta["tickers_in_current"] == 2
        assert delta["tickers_in_prior"] == 2
        assert delta["tickers_common"] == 1  # only AAAA


# ---------------------------------------------------------------------------
# TestFindPrior
# ---------------------------------------------------------------------------

class TestFindPrior:
    """_find_prior_institutional_summary() logic."""

    def _write_summary(self, snap_dir, date_str, *, with_shares=True):
        """Write a summary sidecar into snap_dir/date_str/."""
        date_dir = snap_dir / date_str
        date_dir.mkdir(parents=True, exist_ok=True)
        tickers = {
            "AAAA": _ticker_data(1, {"FundA": 100}, 100, 10) if with_shares else {
                "elite_holders_count": 1,
                "elite_holder_names": ["FundA"],
                "elite_total_shares": 100,
                "elite_total_value_usd_thousands": 10,
                "inst_score_raw": 1,
                "inst_score_z": 0.0,
            },
        }
        data = _make_summary(tickers, as_of_date=date_str)
        with open(date_dir / "institutional_summary.json", "w") as f:
            json.dump(data, f)

    def test_finds_most_recent(self, tmp_path):
        self._write_summary(tmp_path, "2026-02-17")
        self._write_summary(tmp_path, "2026-02-18")
        result = _find_prior_institutional_summary(tmp_path, "2026-02-20")
        assert result is not None
        assert result["as_of_date"] == "2026-02-18"

    def test_skips_v1_without_shares(self, tmp_path):
        self._write_summary(tmp_path, "2026-02-18", with_shares=False)
        self._write_summary(tmp_path, "2026-02-17", with_shares=True)
        result = _find_prior_institutional_summary(tmp_path, "2026-02-20")
        assert result is not None
        assert result["as_of_date"] == "2026-02-17"

    def test_returns_none_when_none_exist(self, tmp_path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        result = _find_prior_institutional_summary(tmp_path, "2026-02-20")
        assert result is None

    def test_returns_none_for_empty_dir(self, tmp_path):
        result = _find_prior_institutional_summary(tmp_path / "nonexistent", "2026-02-20")
        assert result is None

    def test_ignores_current_date(self, tmp_path):
        self._write_summary(tmp_path, "2026-02-20")
        result = _find_prior_institutional_summary(tmp_path, "2026-02-20")
        assert result is None

    def test_max_candidates_limit(self, tmp_path):
        # Write 15 dates, only Feb-03 has shares
        for i in range(1, 16):
            d = f"2026-02-{i:02d}"
            has = (i == 3)  # only Feb-03 has shares
            self._write_summary(tmp_path, d, with_shares=has)
        # max_candidates=5 should only look at 5 most recent before 2026-02-20
        result = _find_prior_institutional_summary(tmp_path, "2026-02-20", max_candidates=5)
        # Feb 15,14,13,12,11 — none have shares
        assert result is None
        # With max_candidates=15, finds Feb-03
        result2 = _find_prior_institutional_summary(tmp_path, "2026-02-20", max_candidates=15)
        assert result2 is not None
        assert result2["as_of_date"] == "2026-02-03"

    def test_skips_pit_mismatch(self, tmp_path):
        """Prior with cache_as_of_date != snapshot date → skipped."""
        date_dir = tmp_path / "2026-02-15"
        date_dir.mkdir(parents=True, exist_ok=True)
        tickers = {"AAAA": _ticker_data(1, {"FundA": 100}, 100, 10)}
        data = _make_summary(tickers, as_of_date="2026-02-15")
        # Inject mismatched cache_as_of_date
        data["cache_as_of_date"] = "2026-01-15"
        with open(date_dir / "institutional_summary.json", "w") as f:
            json.dump(data, f)
        result = _find_prior_institutional_summary(tmp_path, "2026-02-20")
        assert result is None

    def test_accepts_matching_pit(self, tmp_path):
        """Prior with cache_as_of_date matching snapshot date → returned."""
        date_dir = tmp_path / "2026-02-15"
        date_dir.mkdir(parents=True, exist_ok=True)
        tickers = {"AAAA": _ticker_data(1, {"FundA": 100}, 100, 10)}
        data = _make_summary(tickers, as_of_date="2026-02-15")
        data["cache_as_of_date"] = "2026-02-15"
        with open(date_dir / "institutional_summary.json", "w") as f:
            json.dump(data, f)
        result = _find_prior_institutional_summary(tmp_path, "2026-02-20")
        assert result is not None
        assert result["as_of_date"] == "2026-02-15"


# ---------------------------------------------------------------------------
# TestDeltaGate
# ---------------------------------------------------------------------------

class TestDeltaGate:
    """check_institutional_delta() gate function."""

    def test_pass_when_valid_delta_exists(self, tmp_path):
        snap_date = tmp_path / "2026-02-20"
        snap_date.mkdir(parents=True)
        with open(snap_date / "institutional_summary_delta.json", "w") as f:
            json.dump(_valid_delta(), f)
        result = check_institutional_delta(snap_date, tmp_path, "2026-02-20")
        assert result.status == "PASS"
        assert result.name == "institutional_delta"

    def test_pass_cold_start(self, tmp_path):
        """No delta + no prior → PASS (cold-start)."""
        snap_date = tmp_path / "2026-02-20"
        snap_date.mkdir(parents=True)
        result = check_institutional_delta(snap_date, tmp_path, "2026-02-20")
        assert result.status == "PASS"
        assert "cold-start" in result.detail

    def test_warn_when_prior_exists_but_no_delta(self, tmp_path):
        """Prior with elite_holder_shares exists, but delta not written → WARN."""
        # Write a prior with shares
        prior_dir = tmp_path / "2026-02-19"
        prior_dir.mkdir(parents=True)
        prior_data = _make_summary({
            "AAAA": _ticker_data(1, {"FundA": 100}, 100, 10),
        }, as_of_date="2026-02-19")
        with open(prior_dir / "institutional_summary.json", "w") as f:
            json.dump(prior_data, f)
        # Current dir exists but no delta file
        snap_date = tmp_path / "2026-02-20"
        snap_date.mkdir(parents=True)
        result = check_institutional_delta(snap_date, tmp_path, "2026-02-20")
        assert result.status == "WARN"
        assert "Delta expected" in result.detail


# ---------------------------------------------------------------------------
# TestGateTightening — invariant checks in check_institutional_summary
# ---------------------------------------------------------------------------

class TestGateTightening:
    """Gate routes through the pure validator — WARN on any invariant violation."""

    def test_zero_universe_warns(self, tmp_path):
        """tickers_in_universe=0 with mismatched dict → validator catches size mismatch."""
        from run_daily_production import check_institutional_summary
        d = _valid_summary()
        d["tickers_in_universe"] = 0
        # Keep tickers dict non-empty → size mismatch with tickers_in_universe=0
        tmp_path.mkdir(parents=True, exist_ok=True)
        with open(tmp_path / "institutional_summary.json", "w") as f:
            json.dump(d, f)
        result = check_institutional_summary(tmp_path, warn_coverage_pct=0.0)
        assert result.status == "WARN"
        assert "schema invalid" in result.detail

    def test_signal_exceeds_universe_warns(self, tmp_path):
        from run_daily_production import check_institutional_summary
        d = _valid_summary()
        d["tickers_with_signal"] = 5  # > 2 in universe
        tmp_path.mkdir(parents=True, exist_ok=True)
        with open(tmp_path / "institutional_summary.json", "w") as f:
            json.dump(d, f)
        result = check_institutional_summary(tmp_path, warn_coverage_pct=0.0)
        assert result.status == "WARN"
        assert "schema invalid" in result.detail

    def test_managers_inconsistency_warns(self, tmp_path):
        from run_daily_production import check_institutional_summary
        d = _valid_summary()
        d["elite_managers_with_filing"] = 30  # > total=29
        tmp_path.mkdir(parents=True, exist_ok=True)
        with open(tmp_path / "institutional_summary.json", "w") as f:
            json.dump(d, f)
        result = check_institutional_summary(tmp_path, warn_coverage_pct=0.0)
        assert result.status == "WARN"
        assert "schema invalid" in result.detail

    def test_coverage_mismatch_warns(self, tmp_path):
        from run_daily_production import check_institutional_summary
        d = _valid_summary()
        d["signal_coverage_pct"] = 99.0  # actual: 1/2 = 50.0
        tmp_path.mkdir(parents=True, exist_ok=True)
        with open(tmp_path / "institutional_summary.json", "w") as f:
            json.dump(d, f)
        result = check_institutional_summary(tmp_path, warn_coverage_pct=0.0)
        assert result.status == "WARN"
        assert "schema invalid" in result.detail

    def test_consistent_data_passes(self, tmp_path):
        from run_daily_production import check_institutional_summary
        d = _valid_summary()
        tmp_path.mkdir(parents=True, exist_ok=True)
        with open(tmp_path / "institutional_summary.json", "w") as f:
            json.dump(d, f)
        result = check_institutional_summary(tmp_path, warn_coverage_pct=0.0)
        assert result.status == "PASS"


# ---------------------------------------------------------------------------
# TestValidateSummarySchema — pure validator for institutional_summary.v1
# ---------------------------------------------------------------------------

def _valid_summary() -> dict:
    """Minimal valid institutional_summary.v1 dict."""
    return {
        "schema_version": SCHEMA_VERSION,
        "version": "1.0.0",
        "created_at": "2026-02-20T12:00:00Z",
        "as_of_date": "2026-02-20",
        "cache_as_of_date": "2026-02-20",
        "cache_schema_version": "sec_13f_pit_index.v1",
        "elite_managers_total": 29,
        "elite_managers_with_filing": 29,
        "tickers_with_signal": 1,
        "tickers_in_universe": 2,
        "signal_coverage_pct": 50.0,
        "tickers": {
            "AAAA": {
                "elite_holders_count": 1,
                "elite_holder_names": ["FundA"],
                "elite_holder_shares": {"FundA": 100},
                "elite_total_shares": 100,
                "elite_total_value_usd_thousands": 10,
                "inst_score_raw": 1,
                "inst_score_z": 1.0,
            },
            "BBBB": {
                "elite_holders_count": 0,
                "elite_holder_names": [],
                "elite_holder_shares": {},
                "elite_total_shares": 0,
                "elite_total_value_usd_thousands": 0,
                "inst_score_raw": 0,
                "inst_score_z": -1.0,
            },
        },
    }


class TestValidateSummarySchema:
    """validate_institutional_summary_schema_v1() invariants."""

    def test_valid_passes(self):
        ok, detail = validate_institutional_summary_schema_v1(_valid_summary())
        assert ok, detail

    def test_missing_top_level_field(self):
        d = _valid_summary()
        del d["as_of_date"]
        ok, detail = validate_institutional_summary_schema_v1(d)
        assert not ok
        assert "missing top-level" in detail

    def test_wrong_schema_version(self):
        d = _valid_summary()
        d["schema_version"] = "wrong.v99"
        ok, detail = validate_institutional_summary_schema_v1(d)
        assert not ok
        assert "schema_version mismatch" in detail

    def test_created_at_must_end_z(self):
        d = _valid_summary()
        d["created_at"] = "2026-02-20T12:00:00+00:00"
        ok, detail = validate_institutional_summary_schema_v1(d)
        assert not ok
        assert "created_at must end with 'Z'" in detail

    def test_as_of_date_format(self):
        d = _valid_summary()
        d["as_of_date"] = "20260220"
        ok, detail = validate_institutional_summary_schema_v1(d)
        assert not ok
        assert "as_of_date not ISO date" in detail

    def test_as_of_date_mismatch(self):
        d = _valid_summary()
        ok, detail = validate_institutional_summary_schema_v1(d, expected_as_of_date="2026-02-21")
        assert not ok
        assert "as_of_date mismatch" in detail

    def test_managers_filing_exceeds_total(self):
        d = _valid_summary()
        d["elite_managers_with_filing"] = 30
        ok, detail = validate_institutional_summary_schema_v1(d)
        assert not ok
        assert "elite_managers_with_filing (30) > elite_managers_total (29)" in detail

    def test_tickers_with_exceeds_universe(self):
        d = _valid_summary()
        d["tickers_with_signal"] = 3
        ok, detail = validate_institutional_summary_schema_v1(d)
        assert not ok
        assert "tickers_with_signal (3) > tickers_in_universe (2)" in detail

    def test_coverage_out_of_range(self):
        d = _valid_summary()
        d["signal_coverage_pct"] = 150.0
        ok, detail = validate_institutional_summary_schema_v1(d)
        assert not ok
        assert "signal_coverage_pct must be in [0, 100]" in detail

    def test_coverage_inconsistency(self):
        d = _valid_summary()
        d["signal_coverage_pct"] = 99.0  # actual: 1/2 = 50.0
        ok, detail = validate_institutional_summary_schema_v1(d)
        assert not ok
        assert "signal_coverage_pct inconsistent" in detail

    def test_tickers_dict_size_mismatch(self):
        d = _valid_summary()
        d["tickers_in_universe"] = 5  # but only 2 in dict
        d["tickers_with_signal"] = 1
        d["signal_coverage_pct"] = 20.0
        ok, detail = validate_institutional_summary_schema_v1(d)
        assert not ok
        assert "tickers dict size" in detail

    def test_per_ticker_missing_field(self):
        d = _valid_summary()
        del d["tickers"]["AAAA"]["elite_holder_shares"]
        ok, detail = validate_institutional_summary_schema_v1(d)
        assert not ok
        assert "missing fields" in detail

    def test_holders_count_mismatch(self):
        d = _valid_summary()
        d["tickers"]["AAAA"]["elite_holders_count"] = 5  # but only 1 in elite_holder_shares
        ok, detail = validate_institutional_summary_schema_v1(d)
        assert not ok
        assert "elite_holders_count" in detail

    def test_tickers_keys_not_sorted(self):
        d = _valid_summary()
        # Force non-sorted order by rebuilding dict
        d["tickers"] = {"BBBB": d["tickers"]["BBBB"], "AAAA": d["tickers"]["AAAA"]}
        ok, detail = validate_institutional_summary_schema_v1(d)
        assert not ok
        assert "not sorted" in detail

    def test_tickers_with_signal_consistency(self):
        d = _valid_summary()
        d["tickers_with_signal"] = 0  # but AAAA has holders_count=1
        d["signal_coverage_pct"] = 0.0
        ok, detail = validate_institutional_summary_schema_v1(d)
        assert not ok
        assert "tickers_with_signal (0) != actual count" in detail

    def test_build_output_passes_validator(self, tmp_path):
        """Output of build_institutional_summary() passes its own validator."""
        from tests.test_institutional_summary import _write_cache, _holding
        _write_cache(tmp_path, "2026-01-01", [
            ("0000000001", "Mgr A", [_holding("AAAA", shares=500)]),
            ("0000000002", "Mgr B", [_holding("AAAA", shares=300), _holding("BBBB", shares=100)]),
        ])
        result = build_institutional_summary("2026-01-01", {"AAAA", "BBBB", "CCCC"}, cache_base_dir=tmp_path)
        assert result is not None
        ok, detail = validate_institutional_summary_schema_v1(result)
        assert ok, detail


# ---------------------------------------------------------------------------
# TestValidateDeltaSchema — pure validator for institutional_summary_delta.v1
# ---------------------------------------------------------------------------

def _valid_delta() -> dict:
    """Minimal valid institutional_summary_delta.v1 dict."""
    return {
        "schema_version": DELTA_SCHEMA_VERSION,
        "version": "1.0.0",
        "created_at": "2026-02-20T12:00:00Z",
        "as_of_date": "2026-02-20",
        "prior_date": "2026-02-19",
        "tickers_in_current": 2,
        "tickers_in_prior": 2,
        "tickers_common": 2,
        "tickers": {
            "AAAA": {
                "elite_new_count": 1,
                "elite_exit_count": 0,
                "elite_add_count": 1,
                "elite_trim_count": 0,
                "net_elite_holders_delta": 1,
                "shares_delta_total": 500,
                "value_delta_total": 50,
            },
            "BBBB": {
                "elite_new_count": 0,
                "elite_exit_count": 0,
                "elite_add_count": 0,
                "elite_trim_count": 0,
                "net_elite_holders_delta": 0,
                "shares_delta_total": 0,
                "value_delta_total": 0,
            },
        },
    }


class TestValidateDeltaSchema:
    """validate_institutional_summary_delta_schema_v1() invariants."""

    def test_valid_passes(self):
        ok, detail = validate_institutional_summary_delta_schema_v1(_valid_delta())
        assert ok, detail

    def test_missing_top_level_field(self):
        d = _valid_delta()
        del d["prior_date"]
        ok, detail = validate_institutional_summary_delta_schema_v1(d)
        assert not ok
        assert "missing top-level" in detail

    def test_wrong_schema_version(self):
        d = _valid_delta()
        d["schema_version"] = "wrong.v99"
        ok, detail = validate_institutional_summary_delta_schema_v1(d)
        assert not ok
        assert "schema_version mismatch" in detail

    def test_created_at_must_end_z(self):
        d = _valid_delta()
        d["created_at"] = "2026-02-20T12:00:00+00:00"
        ok, detail = validate_institutional_summary_delta_schema_v1(d)
        assert not ok
        assert "created_at must end with 'Z'" in detail

    def test_prior_date_format(self):
        d = _valid_delta()
        d["prior_date"] = "bad"
        ok, detail = validate_institutional_summary_delta_schema_v1(d)
        assert not ok
        assert "prior_date not ISO date" in detail

    def test_prior_date_not_before_as_of(self):
        d = _valid_delta()
        d["prior_date"] = "2026-02-20"  # same as as_of_date
        ok, detail = validate_institutional_summary_delta_schema_v1(d)
        assert not ok
        assert "prior_date" in detail and "must be before" in detail

    def test_as_of_date_mismatch(self):
        d = _valid_delta()
        ok, detail = validate_institutional_summary_delta_schema_v1(d, expected_as_of_date="2026-02-21")
        assert not ok
        assert "as_of_date mismatch" in detail

    def test_tickers_common_exceeds_min(self):
        d = _valid_delta()
        d["tickers_common"] = 5  # > min(2, 2) = 2
        ok, detail = validate_institutional_summary_delta_schema_v1(d)
        assert not ok
        assert "tickers_common (5)" in detail

    def test_tickers_dict_size_mismatch(self):
        d = _valid_delta()
        d["tickers_in_current"] = 5  # but only 2 in dict
        ok, detail = validate_institutional_summary_delta_schema_v1(d)
        assert not ok
        assert "tickers dict size" in detail

    def test_per_ticker_missing_field(self):
        d = _valid_delta()
        del d["tickers"]["AAAA"]["elite_new_count"]
        ok, detail = validate_institutional_summary_delta_schema_v1(d)
        assert not ok
        assert "missing fields" in detail

    def test_per_ticker_non_int(self):
        d = _valid_delta()
        d["tickers"]["AAAA"]["elite_new_count"] = 1.5
        ok, detail = validate_institutional_summary_delta_schema_v1(d)
        assert not ok
        assert "must be int" in detail

    def test_net_delta_consistency(self):
        d = _valid_delta()
        d["tickers"]["AAAA"]["net_elite_holders_delta"] = 999
        ok, detail = validate_institutional_summary_delta_schema_v1(d)
        assert not ok
        assert "net_elite_holders_delta" in detail

    def test_negative_count_rejected(self):
        d = _valid_delta()
        d["tickers"]["AAAA"]["elite_new_count"] = -1
        d["tickers"]["AAAA"]["net_elite_holders_delta"] = -1
        ok, detail = validate_institutional_summary_delta_schema_v1(d)
        assert not ok
        assert "must be >= 0" in detail

    def test_compute_output_passes_validator(self):
        """Output of compute_institutional_delta() passes its own validator."""
        current = _make_summary({
            "AAAA": _ticker_data(2, {"FundA": 200, "FundB": 100}, 300, 30),
            "BBBB": _ticker_data(0, {}, 0, 0),
        }, as_of_date="2026-02-20")
        prior = _make_summary({
            "AAAA": _ticker_data(1, {"FundA": 100}, 100, 10),
            "BBBB": _ticker_data(0, {}, 0, 0),
        }, as_of_date="2026-02-19")
        delta = compute_institutional_delta(current, prior)
        assert delta is not None
        ok, detail = validate_institutional_summary_delta_schema_v1(delta)
        assert ok, detail


# ---------------------------------------------------------------------------
# TestDeltaGateWithValidator — delta gate uses validator
# ---------------------------------------------------------------------------

class TestDeltaGateWithValidator:
    """check_institutional_delta() uses validator when delta file exists."""

    def test_pass_valid_delta(self, tmp_path):
        snap_date = tmp_path / "2026-02-20"
        snap_date.mkdir(parents=True)
        with open(snap_date / "institutional_summary_delta.json", "w") as f:
            json.dump(_valid_delta(), f)
        result = check_institutional_delta(snap_date, tmp_path, "2026-02-20")
        assert result.status == "PASS"
        assert "valid" in result.detail

    def test_warn_invalid_delta(self, tmp_path):
        snap_date = tmp_path / "2026-02-20"
        snap_date.mkdir(parents=True)
        bad_delta = _valid_delta()
        bad_delta["schema_version"] = "wrong"
        with open(snap_date / "institutional_summary_delta.json", "w") as f:
            json.dump(bad_delta, f)
        result = check_institutional_delta(snap_date, tmp_path, "2026-02-20")
        assert result.status == "WARN"
        assert "schema invalid" in result.detail

    def test_warn_malformed_json(self, tmp_path):
        snap_date = tmp_path / "2026-02-20"
        snap_date.mkdir(parents=True)
        (snap_date / "institutional_summary_delta.json").write_text("NOT JSON")
        result = check_institutional_delta(snap_date, tmp_path, "2026-02-20")
        assert result.status == "WARN"
        assert "Cannot read" in result.detail


# ---------------------------------------------------------------------------
# TestSortSignal — institutional delta sort tilt in decision engine
# ---------------------------------------------------------------------------

from decision_engine import DecisionRuleset, compute_actionable_sort_key


def _sort_key(inst_delta_z=0.0, **overrides):
    """Helper: compute sort key with overridable decision fields."""
    defaults = {
        "eligible": "1",
        "tier_dev": "A",
        "archetype": "drug_developer",
        "clinical_optionality_pct_dev": "0.80",
        "composite_rank": "10",
        "catalyst_days": "90",
        "catalyst_mode": "specific_days",
        "catalyst_event_type": "",
        "catalyst_source": "",
        "mom_state": "neutral",
        "sponsor_tier1_count": "1",
        "inst_delta_z": str(inst_delta_z),
    }
    defaults.update(overrides)
    return compute_actionable_sort_key(
        decision_fields=defaults,
        archetype=defaults.get("archetype", "drug_developer"),
        optionality=float(defaults.get("clinical_optionality_pct_dev", "0.80")),
        composite_rank=defaults.get("composite_rank"),
        ticker=defaults.get("ticker", "TEST"),
        catalyst_event_type=defaults.get("catalyst_event_type", ""),
        catalyst_source=defaults.get("catalyst_source", ""),
        ruleset=overrides.pop("_ruleset", None),
    )


class TestSortSignal:
    """Institutional delta sort tilt in compute_actionable_sort_key()."""

    def test_default_disabled(self):
        rs = DecisionRuleset()
        assert rs.enable_institutional_sort_signal is False

    def test_zero_effect_when_disabled(self):
        """Disabled: positive inst_delta_z has no effect."""
        rs = DecisionRuleset()
        k1 = _sort_key(inst_delta_z=0.0, _ruleset=rs)
        k2 = _sort_key(inst_delta_z=2.0, _ruleset=rs)
        assert k1 == k2

    def test_positive_boost_when_enabled(self):
        """Enabled + positive z → lower sort key (better rank)."""
        rs = DecisionRuleset(
            enable_institutional_sort_signal=True,
            institutional_sort_weight=0.3,
            institutional_positive_only=True,
            catalyst_priority_mode="tiebreaker",
        )
        k_neutral = _sort_key(inst_delta_z=0.0, _ruleset=rs)
        k_positive = _sort_key(inst_delta_z=1.5, _ruleset=rs)
        # Positive z → lower effective_comp_rank → sorts first
        assert k_positive < k_neutral

    def test_positive_only_clamps_negative(self):
        """Positive-only: negative z clamped to 0, no penalty."""
        rs = DecisionRuleset(
            enable_institutional_sort_signal=True,
            institutional_sort_weight=0.3,
            institutional_positive_only=True,
            catalyst_priority_mode="tiebreaker",
        )
        k_neutral = _sort_key(inst_delta_z=0.0, _ruleset=rs)
        k_negative = _sort_key(inst_delta_z=-2.0, _ruleset=rs)
        assert k_negative == k_neutral

    def test_negative_penalty_when_not_positive_only(self):
        """positive_only=False + negative z → higher sort key (worse rank)."""
        rs = DecisionRuleset(
            enable_institutional_sort_signal=True,
            institutional_sort_weight=0.3,
            institutional_positive_only=False,
            catalyst_priority_mode="tiebreaker",
        )
        k_neutral = _sort_key(inst_delta_z=0.0, _ruleset=rs)
        k_negative = _sort_key(inst_delta_z=-1.0, _ruleset=rs)
        assert k_negative > k_neutral

    def test_clamping_at_2(self):
        """Z clamped at ±2.0 — z=5.0 and z=2.0 produce same key."""
        rs = DecisionRuleset(
            enable_institutional_sort_signal=True,
            institutional_sort_weight=0.3,
            institutional_positive_only=True,
            catalyst_priority_mode="tiebreaker",
        )
        k_at_2 = _sort_key(inst_delta_z=2.0, _ruleset=rs)
        k_at_5 = _sort_key(inst_delta_z=5.0, _ruleset=rs)
        assert k_at_2 == k_at_5

    def test_additive_with_clinical_and_coinvest(self):
        """All three signals combine additively."""
        rs = DecisionRuleset(
            enable_institutional_sort_signal=True,
            institutional_sort_weight=0.3,
            enable_clinical_sort_signal=True,
            clinical_sort_weight=1.0,
            enable_coinvest_sort_signal=True,
            coinvest_sort_weight=0.5,
            catalyst_priority_mode="tiebreaker",
        )
        # All signals positive → better than none
        k_none = _sort_key(inst_delta_z=0.0, clinical_score_z_tier="0.0",
                          coinvest_score_z="0.0", stage_bucket="mid", _ruleset=rs)
        k_all = _sort_key(inst_delta_z=1.0, clinical_score_z_tier="1.0",
                         coinvest_score_z="1.0", stage_bucket="mid", _ruleset=rs)
        assert k_all < k_none

    def test_graceful_missing_data(self):
        """Missing inst_delta_z → neutral (no crash)."""
        rs = DecisionRuleset(
            enable_institutional_sort_signal=True,
            institutional_sort_weight=0.3,
            catalyst_priority_mode="tiebreaker",
        )
        fields = {
            "eligible": "1",
            "tier_dev": "A",
            "archetype": "drug_developer",
            "clinical_optionality_pct_dev": "0.80",
            "composite_rank": "10",
            "catalyst_days": "90",
            "catalyst_mode": "specific_days",
            "mom_state": "neutral",
            "sponsor_tier1_count": "1",
            # inst_delta_z intentionally missing
        }
        k = compute_actionable_sort_key(
            decision_fields=fields,
            archetype="drug_developer",
            optionality=0.80,
            composite_rank="10",
            ticker="TEST",
            catalyst_event_type="",
            catalyst_source="",
            ruleset=rs,
        )
        # Should not crash, and the key should be same as z=0
        k_zero = _sort_key(inst_delta_z=0.0, _ruleset=rs)
        assert k == k_zero
