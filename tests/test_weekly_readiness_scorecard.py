"""Tests for weekly readiness scorecard."""

import csv
import json

from tools.weekly_readiness_scorecard import (
    SCHEMA_VERSION,
    append_history,
    build_scorecard,
    check_bucket_drift,
    check_gap_risk,
    check_pre_trade_gate,
    check_shadow_excess,
    check_turnover,
    check_warn_streak,
    compute_verdict,
    format_scorecard_md,
    load_history,
    load_performance_rows,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_perf_csv(path, rows):
    """Write a minimal performance.csv."""
    fieldnames = [
        "schema_version",
        "date",
        "prior_date",
        "total_pnl",
        "pnl_pct",
        "xbi_return_pct",
        "excess_vs_xbi_pct",
        "n_held",
        "turnover",
        "gap_risk_high_count",
        "n_missing_price",
        "sleeve_binary_0_30_pnl",
        "sleeve_binary_31_90_pnl",
        "sleeve_binary_91_180_pnl",
        "sleeve_less_binary_pnl",
        "ruleset_id",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            full = {k: "" for k in fieldnames}
            full.update(row)
            w.writerow(full)


def _perf_row(date, excess, turnover=0.05, ruleset_id="7177a4ea"):
    return {
        "schema_version": "live_shadow_perf.v1",
        "date": date,
        "excess_vs_xbi_pct": str(excess) if excess is not None else "",
        "turnover": str(turnover),
        "n_missing_price": "0",
        "ruleset_id": ruleset_id,
    }


def _pre_trade(overall="PASS", can_trade=True, checks=None):
    return {
        "schema": "pre_trade_check.v1",
        "overall": overall,
        "can_trade": can_trade,
        "checks": checks or [],
    }


# ---------------------------------------------------------------------------
# 1. check_shadow_excess
# ---------------------------------------------------------------------------


class TestCheckShadowExcess:
    def test_pass(self):
        rows = [_perf_row(f"2026-03-0{i}", 0.5) for i in range(1, 6)]
        result = check_shadow_excess(rows, 4)
        assert result["status"] == "PASS"
        assert result["value"] > 0

    def test_warn(self):
        rows = [_perf_row(f"2026-03-0{i}", -0.7) for i in range(1, 6)]
        result = check_shadow_excess(rows, 4)
        assert result["status"] == "WARN"

    def test_fail(self):
        rows = [_perf_row(f"2026-03-0{i}", -3.0) for i in range(1, 6)]
        result = check_shadow_excess(rows, 4)
        assert result["status"] == "FAIL"

    def test_insufficient_data(self):
        rows = [_perf_row("2026-03-01", 0.5)]
        result = check_shadow_excess(rows, 4)
        assert result["status"] == "HOLD"

    def test_empty_excess_skipped(self):
        rows = [
            _perf_row("2026-03-01", None),
            _perf_row("2026-03-02", None),
        ]
        result = check_shadow_excess(rows, 4)
        assert result["status"] == "HOLD"


# ---------------------------------------------------------------------------
# 2. check_turnover
# ---------------------------------------------------------------------------


class TestCheckTurnover:
    def test_pass(self):
        rows = [_perf_row(f"2026-03-0{i}", 0.5, turnover=0.05) for i in range(1, 6)]
        result = check_turnover(rows)
        assert result["status"] == "PASS"

    def test_warn(self):
        rows = [_perf_row(f"2026-03-0{i}", 0.5, turnover=0.28) for i in range(1, 6)]
        result = check_turnover(rows)
        assert result["status"] == "WARN"

    def test_fail(self):
        rows = [_perf_row(f"2026-03-0{i}", 0.5, turnover=0.40) for i in range(1, 6)]
        result = check_turnover(rows)
        assert result["status"] == "FAIL"

    def test_insufficient(self):
        rows = [_perf_row("2026-03-01", 0.5)]
        result = check_turnover(rows)
        assert result["status"] == "HOLD"


# ---------------------------------------------------------------------------
# 3. check_bucket_drift
# ---------------------------------------------------------------------------


class TestCheckBucketDrift:
    def test_pass(self):
        pt = _pre_trade(
            checks=[
                {"name": "bucket_deviation", "status": "PASS", "value": 10.0, "detail": "All within limits"},
            ]
        )
        result = check_bucket_drift(pt, {})
        assert result["status"] == "PASS"

    def test_warn(self):
        pt = _pre_trade(
            checks=[
                {"name": "bucket_deviation", "status": "PASS", "value": 20.0, "detail": "Elevated"},
            ]
        )
        result = check_bucket_drift(pt, {})
        assert result["status"] == "WARN"

    def test_fail(self):
        pt = _pre_trade(
            checks=[
                {"name": "bucket_deviation", "status": "PASS", "value": 30.0, "detail": "High"},
            ]
        )
        result = check_bucket_drift(pt, {})
        assert result["status"] == "FAIL"

    def test_no_pre_trade_skips(self):
        """Missing pre_trade is PASS (snapshot-only run), not HOLD."""
        result = check_bucket_drift(None, {})
        assert result["status"] == "PASS"
        assert "skipped" in result["detail"]


# ---------------------------------------------------------------------------
# 4. check_warn_streak
# ---------------------------------------------------------------------------


class TestCheckWarnStreak:
    def test_all_clean(self):
        p2 = {"status": "PASS"}
        rs = {"status": "OK", "consecutive_warn_days": 0, "recommend_rollback": False}
        alerts = {"alert_count": 0, "alerts": []}
        result = check_warn_streak(p2, rs, alerts)
        assert result["status"] == "PASS"

    def test_single_warn(self):
        p2 = {"status": "WARN", "reasons": ["drawdown_coverage_low"]}
        rs = {"status": "OK", "consecutive_warn_days": 0, "recommend_rollback": False}
        alerts = {"alert_count": 0, "alerts": []}
        result = check_warn_streak(p2, rs, alerts)
        assert result["status"] == "PASS"  # 1 warn < threshold 2

    def test_two_warns(self):
        p2 = {"status": "WARN", "reasons": ["drawdown_coverage_low"]}
        rs = {"status": "WARN", "consecutive_warn_days": 1, "recommend_rollback": False}
        alerts = {"alert_count": 0, "alerts": []}
        result = check_warn_streak(p2, rs, alerts)
        assert result["status"] == "WARN"

    def test_rollback_recommended(self):
        p2 = {"status": "PASS"}
        rs = {"status": "WARN", "consecutive_warn_days": 5, "recommend_rollback": True}
        result = check_warn_streak(p2, rs, None)
        assert result["status"] == "FAIL"

    def test_phase2_fail(self):
        p2 = {"status": "FAIL"}
        result = check_warn_streak(p2, None, None)
        assert result["status"] == "FAIL"

    def test_alert_fail(self):
        alerts = {
            "alert_count": 1,
            "alerts": [{"type": "HARD_GATE_FAIL", "severity": "FAIL"}],
        }
        result = check_warn_streak({"status": "PASS"}, None, alerts)
        assert result["status"] == "FAIL"


# ---------------------------------------------------------------------------
# 5. check_gap_risk
# ---------------------------------------------------------------------------


class TestCheckGapRisk:
    def test_pass(self):
        pt = _pre_trade(
            checks=[
                {"name": "gap_risk_concentration", "status": "PASS", "value": 1.0, "detail": "Gap-risk HIGH = 1%"},
            ]
        )
        result = check_gap_risk(None, pt)
        assert result["status"] == "PASS"

    def test_warn(self):
        pt = _pre_trade(
            checks=[
                {"name": "gap_risk_concentration", "status": "PASS", "value": 7.0, "detail": "Gap-risk HIGH = 7%"},
            ]
        )
        result = check_gap_risk(None, pt)
        assert result["status"] == "WARN"

    def test_fail(self):
        pt = _pre_trade(
            checks=[
                {"name": "gap_risk_concentration", "status": "PASS", "value": 12.0, "detail": "Gap-risk HIGH = 12%"},
            ]
        )
        result = check_gap_risk(None, pt)
        assert result["status"] == "FAIL"

    def test_fallback_to_phase2(self):
        p2 = {"metrics": {"exposure": {"catalyst_le_7d_weight_pct": 3.0}}}
        result = check_gap_risk(p2, None)
        assert result["status"] == "PASS"
        assert result["value"] == 3.0


# ---------------------------------------------------------------------------
# 6. check_pre_trade_gate
# ---------------------------------------------------------------------------


class TestCheckPreTradeGate:
    def test_pass(self):
        result = check_pre_trade_gate(_pre_trade("PASS"))
        assert result["status"] == "PASS"

    def test_warn(self):
        pt = _pre_trade(
            "WARN",
            checks=[
                {"name": "missing_prices", "status": "WARN"},
            ],
        )
        result = check_pre_trade_gate(pt)
        assert result["status"] == "WARN"

    def test_fail(self):
        pt = _pre_trade(
            "FAIL",
            can_trade=False,
            checks=[
                {"name": "ruleset_active", "status": "FAIL"},
            ],
        )
        result = check_pre_trade_gate(pt)
        assert result["status"] == "FAIL"

    def test_no_data_skips(self):
        """Missing pre_trade is PASS (snapshot-only run), not HOLD."""
        result = check_pre_trade_gate(None)
        assert result["status"] == "PASS"
        assert "skipped" in result["detail"]


# ---------------------------------------------------------------------------
# 7. Verdict logic
# ---------------------------------------------------------------------------


class TestComputeVerdict:
    def test_all_pass(self):
        checks = [{"status": "PASS"}, {"status": "PASS"}]
        assert compute_verdict(checks) == "READY"

    def test_any_warn(self):
        checks = [{"status": "PASS"}, {"status": "WARN"}]
        assert compute_verdict(checks) == "REVIEW"

    def test_any_fail(self):
        checks = [{"status": "PASS"}, {"status": "FAIL"}]
        assert compute_verdict(checks) == "HOLD"

    def test_hold_from_hold(self):
        checks = [{"status": "PASS"}, {"status": "HOLD"}]
        assert compute_verdict(checks) == "HOLD"

    def test_fail_trumps_warn(self):
        checks = [{"status": "WARN"}, {"status": "FAIL"}]
        assert compute_verdict(checks) == "HOLD"


# ---------------------------------------------------------------------------
# 8. build_scorecard integration
# ---------------------------------------------------------------------------


class TestBuildScorecard:
    def test_cold_start_hold(self, tmp_path):
        """No artifacts at all → HOLD (insufficient data)."""
        snap = tmp_path / "snapshots"
        snap.mkdir()
        art = tmp_path / "artifacts"
        art.mkdir()
        policy = tmp_path / "policy.json"
        policy.write_text("{}")

        sc = build_scorecard("2026-03-10", snap, art, policy)
        assert sc["verdict"] == "HOLD"
        assert sc["schema"] == SCHEMA_VERSION

    def test_healthy_system_ready(self, tmp_path):
        """Full healthy artifacts → READY."""
        snap = tmp_path / "snapshots" / "2026-03-10"
        snap.mkdir(parents=True)
        art = tmp_path / "artifacts"
        art.mkdir()

        # Performance CSV
        rows = [_perf_row(f"2026-03-0{i}", 0.5, 0.05, "test") for i in range(1, 8)]
        _write_perf_csv(art / "performance.csv", rows)

        # Pre-trade
        tp = art / "trade_plan" / "2026-03-10"
        tp.mkdir(parents=True)
        (tp / "pre_trade.json").write_text(
            json.dumps(
                _pre_trade(
                    "PASS",
                    True,
                    [
                        {"name": "bucket_deviation", "status": "PASS", "value": 5.0, "detail": "ok"},
                        {"name": "gap_risk_concentration", "status": "PASS", "value": 1.0, "detail": "ok"},
                    ],
                )
            )
        )

        # Phase2 health
        (snap / "phase2_health.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "reasons": [],
                    "metrics": {"exposure": {"catalyst_le_7d_weight_pct": 2.0}},
                }
            )
        )

        # Ruleset health
        (snap / "ruleset_health.json").write_text(
            json.dumps(
                {
                    "status": "OK",
                    "consecutive_warn_days": 0,
                    "recommend_rollback": False,
                }
            )
        )

        # Alerts
        alerts_dir = art / "alerts"
        alerts_dir.mkdir()
        (alerts_dir / "2026-03-10.json").write_text(
            json.dumps(
                {
                    "alert_count": 0,
                    "alerts": [],
                }
            )
        )

        # Policy
        policy = tmp_path / "policy.json"
        policy.write_text(
            json.dumps(
                {
                    "bucket_targets": {
                        "binary_0_30": 0.10,
                        "binary_31_90": 0.25,
                        "binary_91_180": 0.55,
                        "less_binary": 0.10,
                    }
                }
            )
        )

        sc = build_scorecard("2026-03-10", snap.parent, art, policy, ruleset_id="test")
        assert sc["verdict"] == "READY"
        assert all(c["status"] == "PASS" for c in sc["checks"])

    def test_warn_system_review(self, tmp_path):
        """Performance OK but phase2 WARN → REVIEW."""
        snap = tmp_path / "snapshots" / "2026-03-10"
        snap.mkdir(parents=True)
        art = tmp_path / "artifacts"
        art.mkdir()

        rows = [_perf_row(f"2026-03-0{i}", 0.5, 0.05, "test") for i in range(1, 8)]
        _write_perf_csv(art / "performance.csv", rows)

        tp = art / "trade_plan" / "2026-03-10"
        tp.mkdir(parents=True)
        (tp / "pre_trade.json").write_text(
            json.dumps(
                _pre_trade(
                    "WARN",
                    True,
                    [
                        {"name": "bucket_deviation", "status": "PASS", "value": 5.0, "detail": "ok"},
                        {"name": "gap_risk_concentration", "status": "PASS", "value": 1.0, "detail": "ok"},
                        {"name": "missing_prices", "status": "WARN", "detail": "1 missing"},
                    ],
                )
            )
        )

        (snap / "phase2_health.json").write_text(
            json.dumps(
                {
                    "status": "WARN",
                    "reasons": ["drawdown_coverage_low"],
                    "metrics": {"exposure": {"catalyst_le_7d_weight_pct": 2.0}},
                }
            )
        )
        (snap / "ruleset_health.json").write_text(
            json.dumps(
                {
                    "status": "WARN",
                    "consecutive_warn_days": 1,
                    "recommend_rollback": False,
                }
            )
        )
        alerts_dir = art / "alerts"
        alerts_dir.mkdir()
        (alerts_dir / "2026-03-10.json").write_text(
            json.dumps(
                {
                    "alert_count": 0,
                    "alerts": [],
                }
            )
        )
        policy = tmp_path / "policy.json"
        policy.write_text("{}")

        sc = build_scorecard("2026-03-10", snap.parent, art, policy, ruleset_id="test")
        assert sc["verdict"] == "REVIEW"


# ---------------------------------------------------------------------------
# 9. Output formatting
# ---------------------------------------------------------------------------


class TestFormatScorecard:
    def test_md_contains_verdict(self):
        sc = {
            "as_of_date": "2026-03-10",
            "generated_at": "2026-03-10T12:00:00Z",
            "ruleset_id": "7177a4ea",
            "verdict": "READY",
            "checks": [
                {"name": "shadow_excess_vs_xbi", "status": "PASS", "value": 0.5, "detail": "ok"},
            ],
            "context": {
                "n_performance_rows_for_ruleset": 5,
                "latest_perf_date": "2026-03-10",
                "phase2_health_status": "PASS",
                "ruleset_health_status": "OK",
                "alert_count": 0,
                "pre_trade_overall": "PASS",
            },
        }
        md = format_scorecard_md(sc)
        assert "READY" in md
        assert "7177a4ea" in md
        assert "shadow_excess_vs_xbi" in md

    def test_hold_guidance(self):
        sc = {
            "as_of_date": "2026-03-10",
            "generated_at": "",
            "ruleset_id": "",
            "verdict": "HOLD",
            "checks": [
                {"name": "test", "status": "FAIL", "value": None, "detail": "bad"},
            ],
            "context": {},
        }
        md = format_scorecard_md(sc)
        assert "Do not trade" in md


# ---------------------------------------------------------------------------
# 10. History
# ---------------------------------------------------------------------------


class TestHistory:
    def test_append_and_load(self, tmp_path):
        hist = tmp_path / "history.jsonl"
        sc = {
            "as_of_date": "2026-03-10",
            "generated_at": "2026-03-10T12:00:00Z",
            "ruleset_id": "test",
            "verdict": "READY",
            "checks": [
                {"name": "a", "status": "PASS"},
                {"name": "b", "status": "WARN"},
            ],
        }
        append_history(hist, sc)
        append_history(hist, {**sc, "as_of_date": "2026-03-17", "verdict": "REVIEW"})

        entries = load_history(hist)
        assert len(entries) == 2
        assert entries[0]["verdict"] == "READY"
        assert entries[1]["verdict"] == "REVIEW"
        assert entries[0]["checks_summary"] == {"a": "PASS", "b": "WARN"}

    def test_load_nonexistent(self, tmp_path):
        assert load_history(tmp_path / "nope.jsonl") == []


# ---------------------------------------------------------------------------
# 11. load_performance_rows
# ---------------------------------------------------------------------------


class TestLoadPerformanceRows:
    def test_filters_by_ruleset(self, tmp_path):
        rows = [
            _perf_row("2026-03-01", 0.5, ruleset_id="aaa"),
            _perf_row("2026-03-02", 0.3, ruleset_id="bbb"),
            _perf_row("2026-03-03", 0.1, ruleset_id="aaa"),
        ]
        _write_perf_csv(tmp_path / "perf.csv", rows)
        loaded = load_performance_rows(tmp_path / "perf.csv", "aaa")
        assert len(loaded) == 2
        assert all(r["ruleset_id"] == "aaa" for r in loaded)

    def test_no_filter(self, tmp_path):
        rows = [
            _perf_row("2026-03-01", 0.5, ruleset_id="aaa"),
            _perf_row("2026-03-02", 0.3, ruleset_id="bbb"),
        ]
        _write_perf_csv(tmp_path / "perf.csv", rows)
        loaded = load_performance_rows(tmp_path / "perf.csv")
        assert len(loaded) == 2

    def test_missing_file(self, tmp_path):
        assert load_performance_rows(tmp_path / "nope.csv") == []
