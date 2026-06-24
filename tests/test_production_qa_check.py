"""Tests for tools/production_qa_check.py.

Covers regressions for:
  - check_tracebacks must only scan today's per-date log (not cron.log's
    accumulated history).
  - check_gates/check_readiness must not crash on malformed JSON.
  - check_snapshot/check_schema_drift must gracefully report read errors.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


@pytest.fixture
def qa_mod(tmp_path, monkeypatch):
    """Re-import production_qa_check with SNAPSHOTS_DIR/LOGS_DIR/ARTIFACTS_DIR rerouted to tmp_path."""
    import tools.production_qa_check as mod

    mod = importlib.reload(mod)
    monkeypatch.setattr(mod, "SNAPSHOTS_DIR", tmp_path / "snapshots")
    monkeypatch.setattr(mod, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(mod, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(mod, "OUTPUT_DIR", tmp_path / "artifacts" / "production_qa")
    (tmp_path / "snapshots").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "artifacts").mkdir()
    return mod


# ---------------------------------------------------------------------------
# check_tracebacks
# ---------------------------------------------------------------------------


def test_check_tracebacks_scans_today_log_only(qa_mod, tmp_path):
    """cron.log contains historical tracebacks that must NOT affect today's QA."""
    # Stale trace in shared cron.log — prior day's issue, already resolved
    (tmp_path / "logs" / "cron.log").write_text(
        "[2026-04-10] Traceback (most recent call last):\n  File ...\nValueError\n",
        encoding="utf-8",
    )
    # Today's log is clean
    (tmp_path / "logs" / "daily_production_2026-04-15.log").write_text(
        "RESULT: PASS — snapshot promoted\n",
        encoding="utf-8",
    )

    result = qa_mod.check_tracebacks("2026-04-15")
    assert result["status"] == "PASS", result


def test_check_tracebacks_detects_today_tracebacks(qa_mod, tmp_path):
    (tmp_path / "logs" / "daily_production_2026-04-15.log").write_text(
        "oops\nTraceback (most recent call last):\n  File ...\nKeyError\n",
        encoding="utf-8",
    )
    result = qa_mod.check_tracebacks("2026-04-15")
    assert result["status"] == "FAIL"
    assert "1 traceback" in result["detail"].lower()


def test_check_tracebacks_passes_when_no_log(qa_mod):
    """A missing per-date log isn't FAIL; QA gate below will flag it."""
    result = qa_mod.check_tracebacks("2026-04-15")
    assert result["status"] == "PASS"


# ---------------------------------------------------------------------------
# check_gates — corrupt JSON must not crash
# ---------------------------------------------------------------------------


def test_check_gates_tolerates_malformed_manifest(qa_mod, tmp_path):
    snap = tmp_path / "snapshots" / "2026-04-15"
    snap.mkdir(parents=True)
    (snap / "run_manifest.json").write_text("{ not valid json", encoding="utf-8")

    result = qa_mod.check_gates("2026-04-15")
    assert result["status"] == "FAIL"
    assert "corrupt" in result["detail"].lower()


def test_check_gates_reads_valid_manifest(qa_mod, tmp_path):
    snap = tmp_path / "snapshots" / "2026-04-15"
    snap.mkdir(parents=True)
    (snap / "run_manifest.json").write_text(
        json.dumps(
            {
                "gates": [
                    {"name": "a", "status": "PASS"},
                    {"name": "b", "status": "WARN"},
                ]
            }
        ),
        encoding="utf-8",
    )
    result = qa_mod.check_gates("2026-04-15")
    assert result["status"] == "PASS"
    assert "1 WARN" in result["detail"]


def test_check_gates_reports_fails(qa_mod, tmp_path):
    snap = tmp_path / "snapshots" / "2026-04-15"
    snap.mkdir(parents=True)
    (snap / "run_manifest.json").write_text(
        json.dumps({"gates": [{"name": "turnover", "status": "FAIL"}]}),
        encoding="utf-8",
    )
    result = qa_mod.check_gates("2026-04-15")
    assert result["status"] == "FAIL"
    assert "turnover" in result["detail"]


# ---------------------------------------------------------------------------
# check_readiness — corrupt scorecard must not crash
# ---------------------------------------------------------------------------


def test_check_readiness_tolerates_corrupt_scorecard(qa_mod, tmp_path):
    scorecard_dir = tmp_path / "artifacts" / "readiness"
    scorecard_dir.mkdir(parents=True)
    (scorecard_dir / "scorecard_2026-04-15.json").write_text("not json at all", encoding="utf-8")

    result = qa_mod.check_readiness("2026-04-15")
    assert result["status"] == "FAIL"
    assert "corrupt" in result["detail"].lower()


# ---------------------------------------------------------------------------
# check_feature_coverage (Fix #4) — guard expectation-layer fields in rankings.csv
# ---------------------------------------------------------------------------


def _write_rankings(tmp_path, ds, columns, rows):
    snap = tmp_path / "snapshots" / ds
    snap.mkdir(parents=True, exist_ok=True)
    import csv as _csv

    with open(snap / "rankings.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_feature_coverage_passes_when_all_required_fields_above_threshold(qa_mod, tmp_path):
    """Day-1 baseline: 4 required fields well above floor, insider non-required & missing."""
    ds = "2026-04-24"
    cols = ["ticker", "short_interest_pct", "close_price", "market_cap_mm", "priced_move_pct"]
    rows = [
        {
            "ticker": f"T{i}",
            "short_interest_pct": "5.0",
            "close_price": "10.0",
            "market_cap_mm": "200.0",
            "priced_move_pct": "15.0",
        }
        for i in range(10)
    ]
    _write_rankings(tmp_path, ds, cols, rows)

    result = qa_mod.check_feature_coverage(ds)
    assert result["status"] == "PASS", result
    # Missing non-required insider must carry the full (tracked_nonblocking) suffix
    # and the legend. Older assertions accepted the bare 'MISSING' substring,
    # which still matched after the 2026-04-24 format update — stale.
    assert "insider_net_buy_value_90d*=MISSING (tracked_nonblocking)" in result["detail"]
    assert "* = tracked nonblocking field" in result["detail"]
    assert "close_price=100.0%" in result["detail"]


def test_feature_coverage_fails_when_required_field_below_threshold(qa_mod, tmp_path):
    """Required field below floor must fail and surface the shortfall."""
    ds = "2026-04-24"
    cols = ["ticker", "short_interest_pct", "close_price", "market_cap_mm", "priced_move_pct"]
    # priced_move_pct at 50% — below the 80% floor
    rows = []
    for i in range(10):
        r = {
            "ticker": f"T{i}",
            "short_interest_pct": "5.0",
            "close_price": "10.0",
            "market_cap_mm": "200.0",
            "priced_move_pct": "15.0" if i < 5 else "",
        }
        rows.append(r)
    _write_rankings(tmp_path, ds, cols, rows)

    result = qa_mod.check_feature_coverage(ds)
    assert result["status"] == "FAIL"
    assert "priced_move_pct" in result["detail"]
    assert "50.0% < 80%" in result["detail"]


def test_feature_coverage_fails_when_required_column_missing(qa_mod, tmp_path):
    """Required column absent from rankings.csv must fail."""
    ds = "2026-04-24"
    cols = ["ticker", "close_price", "market_cap_mm", "priced_move_pct"]  # no short_interest_pct
    rows = [
        {"ticker": f"T{i}", "close_price": "10.0", "market_cap_mm": "200.0", "priced_move_pct": "15.0"}
        for i in range(10)
    ]
    _write_rankings(tmp_path, ds, cols, rows)

    result = qa_mod.check_feature_coverage(ds)
    assert result["status"] == "FAIL"
    assert "short_interest_pct" in result["detail"]
    assert "column missing" in result["detail"]


def test_feature_coverage_tolerates_missing_non_required_field(qa_mod, tmp_path):
    """insider_net_buy_value_90d is non-required; missing column must not fail."""
    ds = "2026-04-24"
    cols = ["ticker", "short_interest_pct", "close_price", "market_cap_mm", "priced_move_pct"]
    rows = [
        {
            "ticker": f"T{i}",
            "short_interest_pct": "5.0",
            "close_price": "10.0",
            "market_cap_mm": "200.0",
            "priced_move_pct": "15.0",
        }
        for i in range(10)
    ]
    _write_rankings(tmp_path, ds, cols, rows)

    result = qa_mod.check_feature_coverage(ds)
    assert result["status"] == "PASS"
    # Non-required marker (*), (tracked_nonblocking) suffix, and legend must
    # all appear when the column is missing. Old assertion only checked for
    # 'insider_net_buy_value_90d*' substring — silently passed even if the
    # suffix/legend were dropped.
    assert "insider_net_buy_value_90d*=MISSING (tracked_nonblocking)" in result["detail"]
    assert "* = tracked nonblocking field" in result["detail"]


def test_feature_coverage_insider_missing_reports_tracked_nonblocking_and_passes(qa_mod, tmp_path):
    """Explicit contract: insider column missing must (a) not fail the check and
    (b) surface the exact '(tracked_nonblocking)' marker plus legend so the gap
    stays visible in production_qa output without training readers to ignore it.
    """
    ds = "2026-04-24"
    cols = ["ticker", "short_interest_pct", "close_price", "market_cap_mm", "priced_move_pct"]
    rows = [
        {
            "ticker": f"T{i}",
            "short_interest_pct": "5.0",
            "close_price": "10.0",
            "market_cap_mm": "200.0",
            "priced_move_pct": "15.0",
        }
        for i in range(10)
    ]
    _write_rankings(tmp_path, ds, cols, rows)

    result = qa_mod.check_feature_coverage(ds)
    assert result["status"] == "PASS", f"insider missing must not fail: {result}"
    detail = result["detail"]
    # Exact format required by the user spec
    assert "insider_net_buy_value_90d*=MISSING (tracked_nonblocking)" in detail
    # Legend must accompany any non-required field appearing in the output
    assert "* = tracked nonblocking field" in detail
    # No | FAIL: section in the detail
    assert "| FAIL:" not in detail


def test_feature_coverage_missing_rankings_file_fails(qa_mod):
    result = qa_mod.check_feature_coverage("2026-04-24")
    assert result["status"] == "FAIL"
    assert "No rankings.csv" in result["detail"]


def test_feature_coverage_empty_value_markers_are_not_counted(qa_mod, tmp_path):
    """Empty, 'None', 'nan', 'NaN' must all count as missing for coverage."""
    ds = "2026-04-24"
    cols = ["ticker", "short_interest_pct", "close_price", "market_cap_mm", "priced_move_pct"]
    rows = [
        {
            "ticker": "A",
            "short_interest_pct": "",
            "close_price": "10.0",
            "market_cap_mm": "200",
            "priced_move_pct": "15",
        },
        {
            "ticker": "B",
            "short_interest_pct": "None",
            "close_price": "10.0",
            "market_cap_mm": "200",
            "priced_move_pct": "15",
        },
        {
            "ticker": "C",
            "short_interest_pct": "nan",
            "close_price": "10.0",
            "market_cap_mm": "200",
            "priced_move_pct": "15",
        },
        {
            "ticker": "D",
            "short_interest_pct": "NaN",
            "close_price": "10.0",
            "market_cap_mm": "200",
            "priced_move_pct": "15",
        },
        {
            "ticker": "E",
            "short_interest_pct": "5.0",
            "close_price": "10.0",
            "market_cap_mm": "200",
            "priced_move_pct": "15",
        },
    ]
    _write_rankings(tmp_path, ds, cols, rows)

    result = qa_mod.check_feature_coverage(ds)
    # short_interest_pct is 20% — below 90% floor → FAIL
    assert result["status"] == "FAIL"
    assert "short_interest_pct=20.0%" in result["detail"]


def test_main_attaches_outcome_verdict_on_green(qa_mod, monkeypatch):
    import sys
    from unittest.mock import patch

    verdicts: list[tuple] = []

    def _fake_run_qa(_ds):
        return {"verdict": "GREEN", "n_pass": 9, "n_fail": 0, "checks": []}

    def _fake_log(*_a, **_k):
        return "exec-qa-1"

    def _fake_attach(exec_id, was_correct, evidence, environment="prod"):
        verdicts.append((exec_id, was_correct, evidence))

    monkeypatch.setattr(qa_mod, "run_qa", _fake_run_qa)
    monkeypatch.setattr("tools.agent_skill_telemetry.log_agent_run", _fake_log)
    monkeypatch.setattr("tools.record_skill_feedback.attach_outcome_verdict", _fake_attach)

    with patch.object(sys, "argv", ["production_qa_check.py", "--as-of-date", "2026-06-24"]):
        qa_mod.main()
    assert verdicts == [("exec-qa-1", True, "verdict=GREEN n_fail=0")]
