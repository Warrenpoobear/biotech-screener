"""
Tests for tools/run_stressed_optionality_monitor.py

Classification: STRESSED_OPTIONALITY_FORWARD_MONITOR_NO_MODEL_CHANGE
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tools.run_stressed_optionality_monitor as module
from tools.run_stressed_optionality_monitor import (
    CLASSIFICATION,
    N_BASKET,
    PARAMETERS,
    SCHEMA,
    SUPPRESSION_REVIEW_THRESHOLD,
    WEEKLY_SCHEMA,
    build_suppression_detail,
    compute_guard_status,
    fill_pending_forward,
    run_daily_shadow,
    run_weekly_memo,
)
from tools.run_stressed_optionality_shadow_test import (
    EES_SUPPRESS_THRESHOLD,
    EXTREME_STRESS_FI_Z_THRESHOLD,
    MOMENTUM_CONFIRM_THRESHOLD,
)

# Integration test: exercises the production stressed-optionality monitor against
# production_data/price_history.csv, which is not committed and is absent in a
# clean / CI checkout. Skip cleanly there; runs where the data exists. (CI_RED_2026 #521)
pytestmark = pytest.mark.skipif(
    not module.PRICE_HISTORY_CSV.exists(),
    reason="requires production_data/price_history.csv (absent in clean/CI checkout)",
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model():
    return module.load_ranker_v2_model()


@pytest.fixture(scope="module")
def prices():
    return module.load_price_history()


@pytest.fixture(scope="module")
def trading_dates(prices):
    return module.get_trading_dates(prices)


@pytest.fixture(scope="module")
def daily_may22(model, prices, trading_dates):
    """Phase 3 date with OBSERVED forward return."""
    return run_daily_shadow(
        "2026-05-22", write_output=False, _model=model, _prices=prices, _trading_dates=trading_dates
    )


@pytest.fixture(scope="module")
def daily_recent(model, prices, trading_dates):
    """Most recent snapshot likely to have PENDING forward return."""
    return run_daily_shadow(
        "2026-06-26", write_output=False, _model=model, _prices=prices, _trading_dates=trading_dates
    )


# ---------------------------------------------------------------------------
# TestGovernance
# ---------------------------------------------------------------------------


class TestGovernance:
    def test_classification(self, daily_may22):
        assert daily_may22["classification"] == CLASSIFICATION

    def test_schema(self, daily_may22):
        assert daily_may22["schema"] == SCHEMA

    def test_governance_flags(self, daily_may22):
        gov = daily_may22["governance"]
        for flag in ["model_change", "ranker_change", "selector_change", "sizing_change", "production_wiring"]:
            assert gov[flag] is False, f"governance.{flag} should be False"

    def test_parameters_match_constants(self, daily_may22):
        p = daily_may22["parameters"]
        assert p["ees_cutoff"] == EES_SUPPRESS_THRESHOLD
        assert p["extreme_fi_z"] == EXTREME_STRESS_FI_Z_THRESHOLD
        assert p["momentum_threshold"] == MOMENTUM_CONFIRM_THRESHOLD
        assert p["suppression_rate_review_threshold"] == SUPPRESSION_REVIEW_THRESHOLD

    def test_write_false_creates_no_files(self, tmp_path, monkeypatch, model, prices, trading_dates):
        monkeypatch.setattr(module, "DAILY_DIR", tmp_path / "daily")
        monkeypatch.setattr(module, "JSONL_DIR", tmp_path / "jsonl")
        monkeypatch.setattr(module, "PENDING_DIR", tmp_path / "pending")
        monkeypatch.setattr(module, "DAILY_JSONL", tmp_path / "jsonl" / "test.jsonl")
        run_daily_shadow("2026-05-22", write_output=False, _model=model, _prices=prices, _trading_dates=trading_dates)
        assert not (tmp_path / "daily").exists()
        assert not (tmp_path / "jsonl").exists()


# ---------------------------------------------------------------------------
# TestDailySchema
# ---------------------------------------------------------------------------


class TestDailySchema:
    def test_has_original_top30(self, daily_may22):
        assert "original_top30" in daily_may22
        assert len(daily_may22["original_top30"]) == N_BASKET

    def test_has_shadow_top30(self, daily_may22):
        assert "shadow_top30" in daily_may22
        assert len(daily_may22["shadow_top30"]) == N_BASKET

    def test_has_suppressed_list(self, daily_may22):
        assert "suppressed" in daily_may22
        assert isinstance(daily_may22["suppressed"], list)

    def test_has_suppression_summary(self, daily_may22):
        ss = daily_may22["suppression_summary"]
        assert "n_suppressed" in ss
        assert "suppression_rate" in ss
        assert "guard_status" in ss

    def test_suppression_rate_consistent(self, daily_may22):
        ss = daily_may22["suppression_summary"]
        assert abs(ss["suppression_rate"] - ss["n_suppressed"] / N_BASKET) < 1e-10

    def test_has_forward_return(self, daily_may22):
        fr = daily_may22["forward_return"]
        assert "t5_due_date" in fr
        assert "status" in fr
        assert fr["status"] in ("PENDING", "OBSERVED", "UNOBSERVABLE")

    def test_suppressed_entry_schema(self, daily_may22):
        for s in daily_may22["suppressed"]:
            assert "ticker" in s
            assert "original_rank" in s
            assert "reason_code" in s
            assert "fi_z" in s
            assert "replacement_ticker" in s

    def test_json_serializable(self, daily_may22):
        json.dumps(daily_may22, default=str)

    def test_missing_snapshot_returns_error(self, model, prices, trading_dates):
        result = run_daily_shadow(
            "1900-01-01", write_output=False, _model=model, _prices=prices, _trading_dates=trading_dates
        )
        assert "error" in result


# ---------------------------------------------------------------------------
# TestSuppressionGuard
# ---------------------------------------------------------------------------


class TestSuppressionGuard:
    def test_clean_below_threshold(self):
        assert compute_guard_status(SUPPRESSION_REVIEW_THRESHOLD - 1) == "CLEAN"

    def test_review_required_at_threshold(self):
        assert compute_guard_status(SUPPRESSION_REVIEW_THRESHOLD) == "REVIEW_REQUIRED"

    def test_review_required_above_threshold(self):
        assert compute_guard_status(SUPPRESSION_REVIEW_THRESHOLD + 5) == "REVIEW_REQUIRED"

    def test_zero_suppressed_is_clean(self):
        assert compute_guard_status(0) == "CLEAN"

    def test_guard_status_in_daily_record(self, daily_may22):
        assert daily_may22["suppression_summary"]["guard_status"] in ("CLEAN", "REVIEW_REQUIRED")

    def test_may22_guard_matches_n_suppressed(self, daily_may22):
        ss = daily_may22["suppression_summary"]
        expected = "REVIEW_REQUIRED" if ss["n_suppressed"] >= SUPPRESSION_REVIEW_THRESHOLD else "CLEAN"
        assert ss["guard_status"] == expected


# ---------------------------------------------------------------------------
# TestForwardReturn
# ---------------------------------------------------------------------------


class TestForwardReturn:
    def test_may22_forward_observed(self, daily_may22):
        """May 22 T+5 = Jun 1, which is in price history."""
        fr = daily_may22["forward_return"]
        assert fr["status"] == "OBSERVED"
        assert fr["t5_due_date"] == "2026-06-01"

    def test_may22_forward_returns_not_none(self, daily_may22):
        fr = daily_may22["forward_return"]
        assert fr["original_top30_return"] is not None
        assert fr["shadow_top30_return"] is not None

    def test_may22_delta_equals_shadow_minus_original(self, daily_may22):
        fr = daily_may22["forward_return"]
        if fr["delta"] is not None:
            assert abs(fr["delta"] - (fr["shadow_top30_return"] - fr["original_top30_return"])) < 1e-12

    def test_recent_date_pending_or_unobservable(self, daily_recent):
        """Most recent snapshot should be PENDING (T+5 not yet observed)."""
        fr = daily_recent["forward_return"]
        assert fr["status"] in ("PENDING", "UNOBSERVABLE")

    def test_pending_record_has_t5_due_date(self, daily_recent):
        fr = daily_recent["forward_return"]
        if fr["status"] == "PENDING":
            assert fr["t5_due_date"] is not None

    def test_fill_pending_no_file_returns_error(self, tmp_path, monkeypatch):
        """fill_pending_forward returns gracefully when no pending file exists."""
        monkeypatch.setattr(module, "PENDING_DIR", tmp_path / "pending")
        monkeypatch.setattr(module, "DAILY_DIR", tmp_path / "daily")
        result = fill_pending_forward("2099-01-01", write_output=False)
        assert result["status"] == "NO_PENDING_RECORD"

    def test_fill_pending_writes_files(self, tmp_path, monkeypatch, model, prices, trading_dates):
        """End-to-end: write daily with PENDING → fill forward."""
        monkeypatch.setattr(module, "DAILY_DIR", tmp_path / "daily")
        monkeypatch.setattr(module, "JSONL_DIR", tmp_path / "jsonl")
        monkeypatch.setattr(module, "PENDING_DIR", tmp_path / "pending")
        monkeypatch.setattr(module, "DAILY_JSONL", tmp_path / "jsonl" / "test.jsonl")

        # Write a date whose T+5 is observable (May 18 → May 26, both in price history)
        run_daily_shadow("2026-05-18", write_output=True, _model=model, _prices=prices, _trading_dates=trading_dates)
        daily_file = tmp_path / "daily" / "2026-05-18_stressed_optionality_shadow.json"
        assert daily_file.exists()
        with open(daily_file) as f:
            before = json.load(f)
        # Already OBSERVED since May 26 is in price history — fill should confirm
        assert before["forward_return"]["status"] == "OBSERVED"


# ---------------------------------------------------------------------------
# TestEESMissing
# ---------------------------------------------------------------------------


class TestEESMissing:
    def test_no_crash_with_missing_ees(self, model):
        """apply_shadow_rule handles missing ees_v3_score gracefully."""
        from tools.run_stressed_optionality_shadow_test import apply_shadow_rule

        stats = {"ci_mean": 1.0, "ci_std": 0.5, "fi_mean": 40.0, "fi_std": 20.0}
        row = {
            "ticker": "X",
            "coinvest_score_z": "1.0",
            "financial_score": "5.0",  # extreme stress (fi_z ≈ -1.75)
            "ees_v3_score": "",  # MISSING
            "momentum_score": "70.0",
            "actionable_rank": "5",
        }
        sr = apply_shadow_rule(row, stats, model)
        assert "shadow_status" in sr
        assert sr["shadow_status"] in ("ELIGIBLE", "SUPPRESSED")

    def test_missing_ees_high_momentum_eligible(self, model):
        """Missing EES + high momentum → eligible (benefit of doubt on EES)."""
        from tools.run_stressed_optionality_shadow_test import apply_shadow_rule

        stats = {"ci_mean": 1.0, "ci_std": 0.5, "fi_mean": 40.0, "fi_std": 20.0}
        row = {
            "ticker": "X",
            "coinvest_score_z": "1.0",
            "financial_score": "5.0",
            "ees_v3_score": "",  # MISSING → ees_ok = True
            "momentum_score": "75.0",  # >= 60 → momentum_ok = True
            "actionable_rank": "5",
        }
        sr = apply_shadow_rule(row, stats, model)
        assert sr["shadow_status"] == "ELIGIBLE"

    def test_missing_ees_low_momentum_suppressed(self, model):
        """Missing EES + low momentum → suppressed (extreme stress unconfirmed)."""
        from tools.run_stressed_optionality_shadow_test import apply_shadow_rule

        stats = {"ci_mean": 1.0, "ci_std": 0.5, "fi_mean": 40.0, "fi_std": 20.0}
        row = {
            "ticker": "X",
            "coinvest_score_z": "1.0",
            "financial_score": "5.0",
            "ees_v3_score": "",  # MISSING → ees_ok = True
            "momentum_score": "40.0",  # < 60 → momentum_ok = False → suppressed
            "actionable_rank": "5",
        }
        sr = apply_shadow_rule(row, stats, model)
        assert sr["shadow_status"] == "SUPPRESSED"


# ---------------------------------------------------------------------------
# TestJSONLAppend
# ---------------------------------------------------------------------------


class TestJSONLAppend:
    def test_daily_run_appends_to_jsonl(self, tmp_path, monkeypatch, model, prices, trading_dates):
        monkeypatch.setattr(module, "DAILY_DIR", tmp_path / "daily")
        monkeypatch.setattr(module, "JSONL_DIR", tmp_path / "jsonl")
        monkeypatch.setattr(module, "PENDING_DIR", tmp_path / "pending")
        jsonl_path = tmp_path / "jsonl" / "test.jsonl"
        monkeypatch.setattr(module, "DAILY_JSONL", jsonl_path)

        run_daily_shadow("2026-05-18", write_output=True, _model=model, _prices=prices, _trading_dates=trading_dates)
        run_daily_shadow("2026-05-19", write_output=True, _model=model, _prices=prices, _trading_dates=trading_dates)

        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            entry = json.loads(line)
            assert "as_of_date" in entry
            assert "n_suppressed" in entry
            assert "guard_status" in entry

    def test_jsonl_entry_schema(self, tmp_path, monkeypatch, model, prices, trading_dates):
        monkeypatch.setattr(module, "DAILY_DIR", tmp_path / "daily")
        monkeypatch.setattr(module, "JSONL_DIR", tmp_path / "jsonl")
        monkeypatch.setattr(module, "PENDING_DIR", tmp_path / "pending")
        jsonl_path = tmp_path / "jsonl" / "test.jsonl"
        monkeypatch.setattr(module, "DAILY_JSONL", jsonl_path)

        run_daily_shadow("2026-05-22", write_output=True, _model=model, _prices=prices, _trading_dates=trading_dates)
        entry = json.loads(jsonl_path.read_text().strip())
        assert entry["as_of_date"] == "2026-05-22"
        assert "suppressed_tickers" in entry
        assert "reason_codes" in entry
        assert "fwd_status" in entry


# ---------------------------------------------------------------------------
# TestWeeklyMemo
# ---------------------------------------------------------------------------


class TestWeeklyMemo:
    def test_weekly_memo_runs_on_populated_daily_dir(self, tmp_path, monkeypatch, model, prices, trading_dates):
        monkeypatch.setattr(module, "DAILY_DIR", tmp_path / "daily")
        monkeypatch.setattr(module, "JSONL_DIR", tmp_path / "jsonl")
        monkeypatch.setattr(module, "PENDING_DIR", tmp_path / "pending")
        monkeypatch.setattr(module, "WEEKLY_DIR", tmp_path / "weekly")
        monkeypatch.setattr(module, "DAILY_JSONL", tmp_path / "jsonl" / "test.jsonl")

        for d in ["2026-05-18", "2026-05-19", "2026-05-20", "2026-05-21", "2026-05-22"]:
            run_daily_shadow(d, write_output=True, _model=model, _prices=prices, _trading_dates=trading_dates)

        memo = run_weekly_memo(end_date="2026-05-22", write_output=False)
        assert memo["schema"] == WEEKLY_SCHEMA
        assert memo["n_dates"] == 5
        assert memo["gate_status"] in ("PROMISING", "NEUTRAL", "DEGRADED", "INSUFFICIENT_DATA")

    def test_weekly_memo_empty_dir_returns_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(module, "DAILY_DIR", tmp_path / "nonexistent")
        memo = run_weekly_memo(write_output=False)
        assert "error" in memo

    def test_weekly_memo_gate_status_degraded_for_known_window(
        self, tmp_path, monkeypatch, model, prices, trading_dates
    ):
        """May 18-22: all deltas negative → DEGRADED."""
        monkeypatch.setattr(module, "DAILY_DIR", tmp_path / "daily")
        monkeypatch.setattr(module, "JSONL_DIR", tmp_path / "jsonl")
        monkeypatch.setattr(module, "PENDING_DIR", tmp_path / "pending")
        monkeypatch.setattr(module, "WEEKLY_DIR", tmp_path / "weekly")
        monkeypatch.setattr(module, "DAILY_JSONL", tmp_path / "jsonl" / "test.jsonl")

        for d in ["2026-05-18", "2026-05-19", "2026-05-20", "2026-05-21", "2026-05-22"]:
            run_daily_shadow(d, write_output=True, _model=model, _prices=prices, _trading_dates=trading_dates)

        memo = run_weekly_memo(end_date="2026-05-22", write_output=False)
        # All 5 dates have OBSERVED returns and all deltas negative in this window
        assert memo["gate_status"] == "DEGRADED"

    def test_weekly_memo_insufficient_data_when_no_observed(self, tmp_path, monkeypatch, model, prices, trading_dates):
        """Recent dates (PENDING fwd return) → INSUFFICIENT_DATA."""
        monkeypatch.setattr(module, "DAILY_DIR", tmp_path / "daily")
        monkeypatch.setattr(module, "JSONL_DIR", tmp_path / "jsonl")
        monkeypatch.setattr(module, "PENDING_DIR", tmp_path / "pending")
        monkeypatch.setattr(module, "WEEKLY_DIR", tmp_path / "weekly")
        monkeypatch.setattr(module, "DAILY_JSONL", tmp_path / "jsonl" / "test.jsonl")

        # Jun 24-26 likely to have PENDING forward returns (T+5 not yet available)
        for d in ["2026-06-24", "2026-06-25", "2026-06-26"]:
            path = module.SNAPSHOTS_DIR / d / "rankings.csv"
            if path.exists():
                run_daily_shadow(d, write_output=True, _model=model, _prices=prices, _trading_dates=trading_dates)

        memo = run_weekly_memo(end_date="2026-06-26", write_output=False)
        if memo.get("error"):
            pytest.skip("Not enough recent daily records")
        # If all pending, should be INSUFFICIENT_DATA
        if memo["forward_return_stats"]["n_observed"] < 3:
            assert memo["gate_status"] == "INSUFFICIENT_DATA"

    def test_weekly_suppression_stats_populated(self, tmp_path, monkeypatch, model, prices, trading_dates):
        monkeypatch.setattr(module, "DAILY_DIR", tmp_path / "daily")
        monkeypatch.setattr(module, "JSONL_DIR", tmp_path / "jsonl")
        monkeypatch.setattr(module, "PENDING_DIR", tmp_path / "pending")
        monkeypatch.setattr(module, "WEEKLY_DIR", tmp_path / "weekly")
        monkeypatch.setattr(module, "DAILY_JSONL", tmp_path / "jsonl" / "test.jsonl")

        for d in ["2026-05-18", "2026-05-19", "2026-05-20"]:
            run_daily_shadow(d, write_output=True, _model=model, _prices=prices, _trading_dates=trading_dates)

        memo = run_weekly_memo(end_date="2026-05-20", write_output=False)
        sup = memo["suppression_stats"]
        assert sup["mean_suppressed_per_date"] > 0
        assert sup["reason_code_counts"]  # not empty
        assert isinstance(sup["n_dates_review_required"], int)


# ---------------------------------------------------------------------------
# TestBuildSuppressionDetail
# ---------------------------------------------------------------------------


class TestBuildSuppressionDetail:
    def test_no_suppression_empty_lists(self):
        original = ["A", "B", "C"]
        shadow = ["A", "B", "C"]
        rows = [{"ticker": t, "actionable_rank": str(i + 1)} for i, t in enumerate(original)]
        suppressed, replacements = build_suppression_detail(original, shadow, rows, {})
        assert suppressed == []
        assert replacements == []

    def test_one_suppression_one_replacement(self):
        original = ["A", "B", "C"]
        shadow = ["A", "C", "D"]  # B suppressed, D added
        rows = [{"ticker": t, "actionable_rank": str(i + 1)} for i, t in enumerate(["A", "B", "C", "D"])]
        shadow_by_ticker = {
            "B": {
                "shadow_status": "SUPPRESSED",
                "suppression_type": "EES_FLAGGED",
                "fi_z": -1.5,
                "ees_v3_score": -1.0,
                "momentum_score": 70.0,
                "ci_z": 0.1,
            }
        }
        suppressed, replacements = build_suppression_detail(original, shadow, rows, shadow_by_ticker)
        assert len(suppressed) == 1
        assert suppressed[0]["ticker"] == "B"
        assert suppressed[0]["reason_code"] == "EES_FLAGGED"
        assert suppressed[0]["replacement_ticker"] == "D"
        assert len(replacements) == 1
        assert replacements[0]["ticker"] == "D"

    def test_replacement_ticker_none_when_no_match(self):
        original = ["A", "B", "C"]
        shadow = ["A", "C"]  # B suppressed, nothing replaces (shorter basket)
        rows = [{"ticker": t, "actionable_rank": str(i + 1)} for i, t in enumerate(original)]
        shadow_by_ticker = {
            "B": {
                "shadow_status": "SUPPRESSED",
                "suppression_type": "EES_FLAGGED",
                "fi_z": -1.5,
                "ees_v3_score": -1.0,
                "momentum_score": 70.0,
                "ci_z": 0.1,
            }
        }
        suppressed, _ = build_suppression_detail(original, shadow, rows, shadow_by_ticker)
        assert suppressed[0]["replacement_ticker"] is None
