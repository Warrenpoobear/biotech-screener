"""Tests for tools/ruleset_health_monitor.py — post-promotion health check."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.ruleset_health_monitor import (
    HealthThresholds,
    _count_consecutive_warns,
    _find_active_receipt,
    _load_history,
    evaluate_health,
    run_health_check,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_receipt(
    new_active_id: str = "abc12345",
    mean_top60_overlap: float = 95.0,
    max_rank_shift: float = 4.0,
    mean_turnover: float = 0.12,
    created_at_utc: str = "2026-02-25T12:00:00Z",
    action: str = "promote",
) -> dict:
    return {
        "schema": "promote_receipt.v1",
        "created_at_utc": created_at_utc,
        "action": action,
        "new_active_id": new_active_id,
        "old_active_id": "old12345",
        "gate": {
            "verdict": "PASS",
            "mean_top60_overlap": mean_top60_overlap,
            "max_rank_shift": max_rank_shift,
            "mean_turnover": mean_turnover,
        },
    }


def _make_drift_report(
    top60_overlap_pct: float = 93.0,
    mean_abs_rank_delta_top60: float = 5.0,
    current_date: str = "2026-02-28",
) -> dict:
    return {
        "version": "1.0.0",
        "current_date": current_date,
        "prior_date": "2026-02-27",
        "metrics": {
            "top60_overlap_pct": top60_overlap_pct,
            "mean_abs_rank_delta_top60": mean_abs_rank_delta_top60,
            "top20_overlap_pct": 90.0,
            "rank_spearman_rho": 0.95,
        },
        "status": "PASS",
    }


def _write_receipt_file(tmp_path: Path, receipt: dict, filename: str = "promotion_2026-02-25_abc12345.json"):
    receipts_dir = tmp_path / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    path = receipts_dir / filename
    path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipts_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOKWithinBaseline:
    def test_ok_when_within_baseline(self, tmp_path):
        """Status is OK when today's metrics are within baseline thresholds."""
        receipt = _make_receipt(mean_top60_overlap=95.0, max_rank_shift=4.0)
        # today: overlap=93 (floor=85), rank_shift=5 (ceiling=12) → OK
        drift = _make_drift_report(top60_overlap_pct=93.0, mean_abs_rank_delta_top60=5.0)
        history = tmp_path / "history.jsonl"

        result = evaluate_health(drift, receipt, history)
        assert result["status"] == "OK"
        assert result["recommend_rollback"] is False
        assert result["consecutive_warn_days"] == 0


class TestWarnOnOverlapDegradation:
    def test_warn_on_overlap_degradation(self, tmp_path):
        """WARN when overlap drops below baseline - delta."""
        receipt = _make_receipt(mean_top60_overlap=95.0)
        # floor = 95 - 10 = 85; today=80 → WARN
        drift = _make_drift_report(top60_overlap_pct=80.0)
        history = tmp_path / "history.jsonl"

        result = evaluate_health(drift, receipt, history)
        assert result["status"] == "WARN"
        assert "top60_overlap" in result["detail"]


class TestWarnOnRankShiftSpike:
    def test_warn_on_rank_shift_spike(self, tmp_path):
        """WARN when rank shift exceeds baseline * factor."""
        receipt = _make_receipt(max_rank_shift=4.0)
        # ceiling = 4.0 * 3.0 = 12.0; today=15 → WARN
        drift = _make_drift_report(mean_abs_rank_delta_top60=15.0)
        history = tmp_path / "history.jsonl"

        result = evaluate_health(drift, receipt, history)
        assert result["status"] == "WARN"
        assert "rank_shift" in result["detail"]


class TestRecommendRollback:
    def test_recommend_rollback_after_n_consecutive_warns(self, tmp_path):
        """Recommend rollback after K consecutive WARN days."""
        receipt = _make_receipt(mean_top60_overlap=95.0)
        history = tmp_path / "history.jsonl"

        # Pre-populate 2 prior WARN days
        prior_entries = [
            {"date": "2026-02-26", "active_ruleset_id": "abc12345", "status": "WARN",
             "top60_overlap_pct": 80.0, "max_rank_shift": 15.0,
             "consecutive_warn_days": 1, "recommend_rollback": False},
            {"date": "2026-02-27", "active_ruleset_id": "abc12345", "status": "WARN",
             "top60_overlap_pct": 79.0, "max_rank_shift": 16.0,
             "consecutive_warn_days": 2, "recommend_rollback": False},
        ]
        with open(history, "w") as f:
            for entry in prior_entries:
                f.write(json.dumps(entry) + "\n")

        # Today: also WARN → 3 consecutive → recommend rollback
        drift = _make_drift_report(top60_overlap_pct=78.0)
        th = HealthThresholds(consecutive_warn_days_for_rollback=3)

        result = evaluate_health(drift, receipt, history, th)
        assert result["status"] == "WARN"
        assert result["consecutive_warn_days"] == 3
        assert result["recommend_rollback"] is True


class TestNoReceiptGraceful:
    def test_no_receipt_returns_pass(self, tmp_path):
        """Cold start: no receipt → PASS."""
        drift = _make_drift_report()
        history = tmp_path / "history.jsonl"

        result = evaluate_health(drift, None, history)
        assert result["status"] == "PASS"
        assert "No promotion receipt" in result["detail"]


class TestHistoryAppended:
    def test_history_appended(self, tmp_path):
        """JSONL history grows with each evaluation."""
        receipt = _make_receipt()
        history = tmp_path / "history.jsonl"
        drift = _make_drift_report()

        evaluate_health(drift, receipt, history)
        lines = history.read_text().strip().splitlines()
        assert len(lines) == 1

        drift2 = _make_drift_report(current_date="2026-03-01")
        evaluate_health(drift2, receipt, history)
        lines = history.read_text().strip().splitlines()
        assert len(lines) == 2


class TestFirstDayAfterPromotion:
    def test_first_day_after_promotion(self, tmp_path):
        """First day after promotion — no false alarm when within baseline."""
        receipt = _make_receipt(
            mean_top60_overlap=95.0, max_rank_shift=4.0,
            created_at_utc="2026-02-27T12:00:00Z",
        )
        drift = _make_drift_report(
            top60_overlap_pct=94.0, mean_abs_rank_delta_top60=3.0,
            current_date="2026-02-28",
        )
        history = tmp_path / "history.jsonl"

        result = evaluate_health(drift, receipt, history)
        assert result["status"] == "OK"
        assert result["days_since_promotion"] == 1
        assert result["consecutive_warn_days"] == 0


class TestStatusClearsAfterGoodDay:
    def test_status_clears_after_good_day(self, tmp_path):
        """Consecutive WARN counter resets after a good day."""
        receipt = _make_receipt(mean_top60_overlap=95.0)
        history = tmp_path / "history.jsonl"

        # Pre-populate 2 consecutive WARNs
        prior_entries = [
            {"date": "2026-02-26", "active_ruleset_id": "abc12345", "status": "WARN",
             "consecutive_warn_days": 1, "recommend_rollback": False},
            {"date": "2026-02-27", "active_ruleset_id": "abc12345", "status": "WARN",
             "consecutive_warn_days": 2, "recommend_rollback": False},
        ]
        with open(history, "w") as f:
            for entry in prior_entries:
                f.write(json.dumps(entry) + "\n")

        # Today: good day → resets to 0
        drift = _make_drift_report(top60_overlap_pct=92.0, mean_abs_rank_delta_top60=3.0)
        result = evaluate_health(drift, receipt, history)
        assert result["status"] == "OK"
        assert result["consecutive_warn_days"] == 0
        assert result["recommend_rollback"] is False


class TestMissingDriftReport:
    def test_missing_drift_report_graceful(self, tmp_path):
        """Missing drift report → PASS with detail."""
        receipt = _make_receipt()
        history = tmp_path / "history.jsonl"

        result = evaluate_health(None, receipt, history)
        assert result["status"] == "PASS"
        assert "No drift report" in result["detail"]


class TestGateIntegration:
    def test_gate_result_wrapper(self, tmp_path):
        """check_ruleset_health returns a correct GateResult."""
        # Set up receipts dir with a receipt
        receipts_dir = tmp_path / "receipts"
        receipts_dir.mkdir()
        receipt = _make_receipt()
        (receipts_dir / "promotion_2026-02-25_abc12345.json").write_text(
            json.dumps(receipt), encoding="utf-8"
        )

        # Set up staging dir with drift report
        staging = tmp_path / "staging"
        staging.mkdir()
        drift = _make_drift_report()
        (staging / "drift_report.json").write_text(
            json.dumps(drift), encoding="utf-8"
        )

        history = tmp_path / "history.jsonl"

        result = run_health_check(
            drift_report_path=staging / "drift_report.json",
            receipts_dir=receipts_dir,
            history_path=history,
            output_dir=staging,
            active_ruleset_id="abc12345",
        )

        assert result["schema"] == "ruleset_health.v1"
        assert result["active_ruleset_id"] == "abc12345"
        assert result["status"] in ("OK", "WARN")

        # Sidecar written
        sidecar = staging / "ruleset_health.json"
        assert sidecar.exists()
        sidecar_data = json.loads(sidecar.read_text())
        assert sidecar_data["schema"] == "ruleset_health.v1"
