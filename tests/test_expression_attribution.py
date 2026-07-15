"""Tests for Expression Attribution Engine (Spec 062, Phase 2).

Covers: logging, resolution, P&L estimation, kill switches, metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# Re-use Phase 1 test fixtures
from event_ev.data_contracts import (
    CatalystNode,
    CrowdBelief,
    ExpectationErrorScore,
    OutcomeProbabilities,
    ScenarioPayoffs,
    TimingEstimate,
)
from event_ev.expression_attribution import (
    _compute_pnl_estimate,
    _recommendation_to_attribution_record,
    _recommendation_to_decision_record,
    append_records_idempotent,
    compute_attribution_metrics,
    dedup_keep_last,
    evaluate_kill_switches,
    load_attribution_log,
    load_decision_log,
    log_decision,
    log_recommendation,
    resolve_attributions,
)
from event_ev.expression_layer import build_recommendation

# ============================================================================
# Fixtures
# ============================================================================


def _make_node(**overrides: Any) -> CatalystNode:
    defaults = {
        "ticker": "ACAD",
        "event_family": "CLINICAL",
        "event_type": "DATA_READOUT",
        "event_subtype": "TOPLINE",
        "expected_date": "2026-05-15",
        "date_range_start": "2026-05-15",
        "date_range_end": None,
        "date_precision": "MONTH",
        "date_confidence": 0.6,
        "source": "CTGOV",
        "source_uid": "NCT12345678",
        "disclosed_at": "2026-01-15",
        "phase": "3",
        "indication": "oncology",
    }
    defaults.update(overrides)
    return CatalystNode(**defaults)


def _make_rec(**kw: Any):
    """Build a full ExpressionRecommendation via build_recommendation."""
    defaults = {
        "node": _make_node(),
        "outcome": OutcomeProbabilities(
            node_id="abc123",
            as_of_date="2026-04-13",
            p_hit=0.55,
            p_miss=0.35,
            p_mixed=0.10,
            confidence=0.65,
            prior_source="v2_empirical",
        ),
        "crowd": CrowdBelief(
            node_id="abc123",
            as_of_date="2026-04-13",
            implied_p_hit=0.40,
            belief_direction="BEARISH",
            belief_intensity=0.5,
            priced_move_pct=25.0,
            mispricing_score=0.20,
        ),
        "payoff": ScenarioPayoffs(
            node_id="abc123",
            as_of_date="2026-04-13",
            upside_hit=40.0,
            downside_miss=-30.0,
            move_mixed=5.0,
            scenario_ev=8.0,
            asymmetry_ratio=1.33,
            downside_adjusted_ev=5.0,
            kelly_fraction=0.10,
            analog_count=35,
            analog_confidence="ok",
        ),
        "ees": ExpectationErrorScore(
            ticker="ACAD",
            as_of_date="2026-04-13",
            base_rate_gap_score=0.10,
            conditional_misprice_score=0.20,
            slippage_penalty_score=0.15,
            divergence_score=0.10,
            crowding_bias_score=0.05,
            timing_decay_risk_score=0.10,
            expectation_error_score=0.15,
            expectation_confidence=0.70,
            expectation_notes="",
            quality_overlay_score=-0.10,
            trap_overlay_score=-0.15,
            ees_v2_score=-0.12,
        ),
        "timing": TimingEstimate(
            node_id="abc123",
            as_of_date="2026-04-13",
            prob_on_time=0.60,
            prob_slip=0.30,
            prob_early=0.10,
            expected_delay_days=15.0,
            median_arrival_days=32.0,
            hazard_rate=0.03,
        ),
        "as_of_date": "2026-04-13",
        "opt_liquidity_state": "liquid",
        "opt_front_iv": 0.90,
        "opt_back_iv": 0.50,
        "opt_atm_iv": 0.85,
        "bid_ask_spread_pct": 0.02,
        "priced_move_pct": 25.0,
        "quote_fresh": True,
    }
    defaults.update(kw)
    return build_recommendation(**defaults)


# ============================================================================
# Record creation tests
# ============================================================================


class TestAttributionRecord:
    def test_schema_complete(self):
        rec = _make_rec()
        record = _recommendation_to_attribution_record(rec, timestamp="2026-04-13T10:00:00")
        assert record["ticker"] == "ACAD"
        assert record["timestamp"] == "2026-04-13T10:00:00"
        assert record["mispricing_type"] == rec.mispricing_type
        assert record["overlay_class"] == rec.overlay_class
        assert record["priced_move_pct"] == pytest.approx(25.0, abs=0.01)
        assert record["scenario_ev"] == pytest.approx(8.0, abs=0.01)
        assert record["opt_atm_iv"] == pytest.approx(0.85, abs=0.01)
        # Resolution fields are null at log time
        assert record["resolved_date"] is None
        assert record["realized_outcome"] is None
        assert record["pnl_estimate"] is None
        assert record["attribution_status"] == "pending"

    def test_json_serializable(self):
        rec = _make_rec()
        record = _recommendation_to_attribution_record(rec)
        s = json.dumps(record)
        assert isinstance(s, str)

    def test_governance_fields(self):
        rec = _make_rec()
        record = _recommendation_to_attribution_record(rec)
        assert record["governance_class"] == "overlay_only"
        assert "not_alpha" in record["policy_flags"]


class TestDecisionRecord:
    def test_tradeable_decision(self):
        rec = _make_rec()
        record = _recommendation_to_decision_record(rec)
        if rec.is_tradeable:
            assert record["decision"] == "tradeable"

    def test_rejected_decision(self):
        rec = _make_rec(opt_liquidity_state="illiquid")
        record = _recommendation_to_decision_record(rec)
        assert record["decision"] == "rejected"
        assert len(record["gate_failures"]) > 0

    def test_kill_switched_decision(self):
        rec = _make_rec()
        record = _recommendation_to_decision_record(rec, kill_switch_active=True, kill_switch_reason="sharpe < -0.50")
        assert record["decision"] == "kill_switched"
        assert record["kill_switch_active"] is True
        assert record["kill_switch_reason"] == "sharpe < -0.50"


# ============================================================================
# File I/O tests
# ============================================================================


class TestLogRecommendation:
    def test_append_creates_file(self, tmp_path: Path):
        log_path = tmp_path / "attr.jsonl"
        rec = _make_rec()
        result = log_recommendation(rec, log_path=log_path, timestamp="2026-04-13T10:00:00")
        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["ticker"] == "ACAD"
        assert result["ticker"] == "ACAD"

    def test_same_node_replaces_not_duplicates(self, tmp_path: Path):
        # Re-running the pipeline for the same node must not duplicate (#495)
        log_path = tmp_path / "attr.jsonl"
        rec = _make_rec()
        log_recommendation(rec, log_path=log_path, timestamp="2026-04-13T10:00:00")
        log_recommendation(rec, log_path=log_path, timestamp="2026-04-13T11:00:00")
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0])["timestamp"] == "2026-04-13T11:00:00"

    def test_distinct_nodes_append(self, tmp_path: Path):
        log_path = tmp_path / "attr.jsonl"
        log_recommendation(_make_rec(), log_path=log_path, timestamp="2026-04-13T10:00:00")
        log_recommendation(
            _make_rec(node=_make_node(ticker="SRPT")),
            log_path=log_path,
            timestamp="2026-04-13T10:00:00",
        )
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_no_overwrite(self, tmp_path: Path):
        log_path = tmp_path / "attr.jsonl"
        log_path.write_text('{"existing": true}\n')
        rec = _make_rec()
        log_recommendation(rec, log_path=log_path, timestamp="2026-04-13T10:00:00")
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"existing": True}


class TestLogDecision:
    def test_append_creates_file(self, tmp_path: Path):
        log_path = tmp_path / "decisions.jsonl"
        rec = _make_rec()
        result = log_decision(rec, log_path=log_path, timestamp="2026-04-13T10:00:00")
        assert log_path.exists()
        assert result["ticker"] == "ACAD"

    def test_records_rejection(self, tmp_path: Path):
        log_path = tmp_path / "decisions.jsonl"
        rec = _make_rec(opt_liquidity_state="illiquid")
        log_decision(rec, log_path=log_path)
        records = load_decision_log(log_path)
        assert len(records) == 1
        assert records[0]["decision"] == "rejected"
        assert len(records[0]["gate_failures"]) > 0

    def test_records_kill_switch(self, tmp_path: Path):
        log_path = tmp_path / "decisions.jsonl"
        rec = _make_rec()
        log_decision(
            rec,
            log_path=log_path,
            kill_switch_active=True,
            kill_switch_reason="aggregate win rate < 40%",
        )
        records = load_decision_log(log_path)
        assert records[0]["decision"] == "kill_switched"
        assert records[0]["kill_switch_active"] is True

    def test_records_tradeable(self, tmp_path: Path):
        log_path = tmp_path / "decisions.jsonl"
        rec = _make_rec()
        log_decision(rec, log_path=log_path)
        records = load_decision_log(log_path)
        assert len(records) == 1
        if rec.is_tradeable:
            assert records[0]["decision"] == "tradeable"


class TestIdempotentAppend:
    def _rec_dict(self, ticker: str, node_id: str, **extra: Any) -> dict:
        return {"ticker": ticker, "node_id": node_id, "timestamp": "2026-04-13T00:00:00", **extra}

    def test_rerun_batch_does_not_grow_file(self, tmp_path: Path):
        log_path = tmp_path / "log.jsonl"
        batch = [self._rec_dict("ACAD", "ACAD_2026-04-13"), self._rec_dict("SRPT", "SRPT_2026-04-13")]
        append_records_idempotent(log_path, batch)
        counts = append_records_idempotent(log_path, batch)
        assert counts == {"appended": 2, "replaced": 2, "kept_resolved": 0}
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_resolved_record_not_displaced_by_pending(self, tmp_path: Path):
        log_path = tmp_path / "log.jsonl"
        resolved = self._rec_dict("ACAD", "ACAD_2026-04-13", attribution_status="resolved", realized_outcome="HIT")
        append_records_idempotent(log_path, [resolved])
        pending = self._rec_dict("ACAD", "ACAD_2026-04-13", attribution_status="pending")
        counts = append_records_idempotent(log_path, [pending])
        assert counts == {"appended": 0, "replaced": 0, "kept_resolved": 1}
        records = [json.loads(line) for line in log_path.read_text().strip().split("\n")]
        assert len(records) == 1
        assert records[0]["attribution_status"] == "resolved"

    def test_keyless_existing_records_preserved(self, tmp_path: Path):
        log_path = tmp_path / "log.jsonl"
        log_path.write_text('{"existing": true}\nnot json at all\n')
        append_records_idempotent(log_path, [self._rec_dict("ACAD", "ACAD_2026-04-13")])
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 3
        assert json.loads(lines[0]) == {"existing": True}
        assert lines[1] == "not json at all"

    def test_empty_batch_is_noop(self, tmp_path: Path):
        log_path = tmp_path / "log.jsonl"
        counts = append_records_idempotent(log_path, [])
        assert counts == {"appended": 0, "replaced": 0, "kept_resolved": 0}
        assert not log_path.exists()

    def test_legacy_duplicates_compact_on_touch(self, tmp_path: Path):
        # Pre-#495 files hold the same key many times; writing that key
        # again collapses all copies to one.
        log_path = tmp_path / "log.jsonl"
        legacy = self._rec_dict("ACAD", "ACAD_2026-04-13")
        log_path.write_text("".join(json.dumps(legacy) + "\n" for _ in range(8)))
        append_records_idempotent(log_path, [self._rec_dict("ACAD", "ACAD_2026-04-13")])
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1


class TestDedupKeepLast:
    def test_keeps_last_per_key(self):
        records = [
            {"ticker": "ACAD", "node_id": "n1", "v": 1},
            {"ticker": "ACAD", "node_id": "n1", "v": 2},
            {"ticker": "SRPT", "node_id": "n2", "v": 3},
        ]
        out = dedup_keep_last(records)
        assert len(out) == 2
        assert out[0]["v"] == 2

    def test_resolved_wins_over_later_pending(self):
        records = [
            {"ticker": "ACAD", "node_id": "n1", "attribution_status": "resolved"},
            {"ticker": "ACAD", "node_id": "n1", "attribution_status": "pending"},
        ]
        out = dedup_keep_last(records)
        assert len(out) == 1
        assert out[0]["attribution_status"] == "resolved"

    def test_keyless_records_pass_through(self):
        records = [{"foo": 1}, {"ticker": "ACAD", "node_id": "n1"}, {"foo": 2}]
        assert len(dedup_keep_last(records)) == 3

    def test_loader_dedups_legacy_duplicates(self, tmp_path: Path):
        log_path = tmp_path / "attr.jsonl"
        rec = {"ticker": "ACAD", "node_id": "n1", "attribution_status": "pending"}
        log_path.write_text("".join(json.dumps(rec) + "\n" for _ in range(13)))
        assert len(load_attribution_log(log_path)) == 1
        assert len(load_decision_log(log_path)) == 1


class TestLoadLog:
    def test_empty_file(self, tmp_path: Path):
        log_path = tmp_path / "empty.jsonl"
        log_path.write_text("")
        assert load_attribution_log(log_path) == []

    def test_missing_file(self, tmp_path: Path):
        log_path = tmp_path / "nonexistent.jsonl"
        assert load_attribution_log(log_path) == []

    def test_malformed_line(self, tmp_path: Path):
        log_path = tmp_path / "bad.jsonl"
        log_path.write_text('{"ok": true}\nnot json\n{"also_ok": true}\n')
        records = load_attribution_log(log_path)
        assert len(records) == 2


# ============================================================================
# Resolution tests
# ============================================================================


class TestResolveAttributions:
    def test_resolve_pending(self, tmp_path: Path):
        log_path = tmp_path / "attr.jsonl"
        rec = _make_rec()
        log_recommendation(rec, log_path=log_path, timestamp="2026-04-13T10:00:00")

        resolutions = [
            {
                "ticker": "ACAD",
                "node_id": rec.node_id,
                "outcome": "HIT",
                "resolved_date": "2026-05-20",
                "price_t_minus_1": 20.0,
                "price_t_0": 30.0,
                "price_t_plus_5": 28.0,
            }
        ]

        count = resolve_attributions(log_path=log_path, resolutions=resolutions)
        assert count == 1

        records = load_attribution_log(log_path)
        assert records[0]["attribution_status"] == "resolved"
        assert records[0]["realized_outcome"] == "HIT"
        assert records[0]["realized_move_1d_pct"] == pytest.approx(50.0, abs=0.01)
        assert records[0]["realized_move_5d_pct"] == pytest.approx(40.0, abs=0.01)

    def test_no_double_resolve(self, tmp_path: Path):
        log_path = tmp_path / "attr.jsonl"
        rec = _make_rec()
        log_recommendation(rec, log_path=log_path)

        resolutions = [
            {
                "ticker": "ACAD",
                "node_id": rec.node_id,
                "outcome": "HIT",
                "resolved_date": "2026-05-20",
                "price_t_minus_1": 20.0,
                "price_t_0": 30.0,
                "price_t_plus_5": 28.0,
            }
        ]

        resolve_attributions(log_path=log_path, resolutions=resolutions)
        # Second pass: already resolved, should not re-resolve
        count = resolve_attributions(log_path=log_path, resolutions=resolutions)
        assert count == 0

    def test_no_resolutions(self, tmp_path: Path):
        log_path = tmp_path / "attr.jsonl"
        rec = _make_rec()
        log_recommendation(rec, log_path=log_path)
        count = resolve_attributions(log_path=log_path, resolutions=[])
        assert count == 0

    def test_unmatched_resolution(self, tmp_path: Path):
        log_path = tmp_path / "attr.jsonl"
        rec = _make_rec()
        log_recommendation(rec, log_path=log_path)

        resolutions = [
            {
                "ticker": "UNKNOWN",
                "node_id": "bad_id",
                "outcome": "HIT",
                "resolved_date": "2026-05-20",
                "price_t_minus_1": 20.0,
                "price_t_0": 30.0,
                "price_t_plus_5": 28.0,
            }
        ]
        count = resolve_attributions(log_path=log_path, resolutions=resolutions)
        assert count == 0
        records = load_attribution_log(log_path)
        assert records[0]["attribution_status"] == "pending"


# ============================================================================
# P&L estimation tests
# ============================================================================


class TestPnlEstimate:
    def test_directional(self):
        pnl = _compute_pnl_estimate("DIRECTIONAL_DEBIT", 25.0, 35.0)
        assert pnl == pytest.approx(10.0)

    def test_variance_debit(self):
        pnl = _compute_pnl_estimate("VARIANCE_DEBIT", 25.0, -35.0)
        assert pnl == pytest.approx(10.0)  # |realized| - priced

    def test_variance_underperform(self):
        pnl = _compute_pnl_estimate("VARIANCE_DEBIT", 25.0, 15.0)
        assert pnl == pytest.approx(-10.0)  # |15| - 25

    def test_credit(self):
        pnl = _compute_pnl_estimate("DEFINED_RISK_CREDIT", 25.0, 10.0)
        assert pnl == pytest.approx(15.0)  # priced - |realized|

    def test_calendar_returns_none(self):
        assert _compute_pnl_estimate("TIMING_CALENDAR", 25.0, 10.0) is None

    def test_manual_review_returns_none(self):
        assert _compute_pnl_estimate("MANUAL_REVIEW", 25.0, 10.0) is None

    def test_missing_inputs(self):
        assert _compute_pnl_estimate("DIRECTIONAL_DEBIT", None, 10.0) is None
        assert _compute_pnl_estimate("DIRECTIONAL_DEBIT", 25.0, None) is None


# ============================================================================
# Metrics tests
# ============================================================================


def _make_resolved_records(
    n: int = 25,
    win_rate: float = 0.60,
    mean_pnl: float = 2.0,
    mispricing_type: str = "DIRECTIONAL",
    confidence: float = 0.65,
) -> list:
    """Generate synthetic resolved attribution records."""
    records = []
    n_wins = int(n * win_rate)
    for i in range(n):
        pnl = abs(mean_pnl) if i < n_wins else -abs(mean_pnl) * 0.5
        records.append(
            {
                "ticker": f"T{i}",
                "node_id": f"n{i}",
                "mispricing_type": mispricing_type,
                "overlay_class": "DIRECTIONAL_DEBIT",
                "mispricing_confidence": confidence,
                "pnl_estimate": pnl,
                "attribution_status": "resolved",
            }
        )
    return records


class TestAttributionMetrics:
    def test_no_records(self):
        m = compute_attribution_metrics([])
        assert m["n_resolved"] == 0
        assert m["sufficient"] is False

    def test_insufficient(self):
        records = _make_resolved_records(n=10)
        m = compute_attribution_metrics(records)
        assert m["n_resolved"] == 10
        assert m["sufficient"] is False

    def test_sufficient(self):
        records = _make_resolved_records(n=25)
        m = compute_attribution_metrics(records)
        assert m["n_resolved"] == 25
        assert m["sufficient"] is True

    def test_win_rate(self):
        records = _make_resolved_records(n=25, win_rate=0.60)
        m = compute_attribution_metrics(records)
        assert m["aggregate"]["win_rate"] == pytest.approx(0.60, abs=0.01)

    def test_by_type(self):
        records = _make_resolved_records(n=20, mispricing_type="DIRECTIONAL")
        records += _make_resolved_records(n=10, mispricing_type="VARIANCE")
        m = compute_attribution_metrics(records)
        assert "DIRECTIONAL" in m["by_type"]
        assert "VARIANCE" in m["by_type"]

    def test_by_confidence(self):
        low = _make_resolved_records(n=10, confidence=0.40)
        high = _make_resolved_records(n=10, confidence=0.75)
        m = compute_attribution_metrics(low + high)
        assert "low" in m["by_confidence"]
        assert "high" in m["by_confidence"]

    def test_pending_excluded(self):
        records = _make_resolved_records(n=5)
        records.append({"attribution_status": "pending", "pnl_estimate": 100.0})
        m = compute_attribution_metrics(records)
        assert m["n_resolved"] == 5


# ============================================================================
# Kill switch tests
# ============================================================================


class TestKillSwitches:
    def test_insufficient_data(self):
        records = _make_resolved_records(n=10)
        ks = evaluate_kill_switches(records)
        assert ks["evaluation_status"] == "insufficient_data"
        assert ks["overlay_enabled"] is True

    def test_healthy_overlay(self):
        records = _make_resolved_records(n=25, win_rate=0.60, mean_pnl=2.0)
        ks = evaluate_kill_switches(records)
        assert ks["overlay_enabled"] is True
        assert ks["sizing_enabled"] is True
        assert ks["triggered_rules"] == []

    def test_kill_switch_win_rate(self):
        records = _make_resolved_records(n=25, win_rate=0.30)
        ks = evaluate_kill_switches(records)
        assert ks["overlay_enabled"] is False
        assert any("aggregate_win_rate" in r for r in ks["triggered_rules"])

    def test_kill_switch_type_win_rate(self):
        # 25 DIRECTIONAL records with 20% win rate
        records = _make_resolved_records(n=25, win_rate=0.20, mispricing_type="DIRECTIONAL")
        ks = evaluate_kill_switches(records)
        assert "DIRECTIONAL" in ks["disabled_types"]
        assert any("type_DIRECTIONAL" in r for r in ks["triggered_rules"])

    def test_kill_switch_sharpe(self):
        # All losses → negative sharpe
        records = []
        for i in range(25):
            records.append(
                {
                    "ticker": f"T{i}",
                    "node_id": f"n{i}",
                    "mispricing_type": "DIRECTIONAL",
                    "overlay_class": "DIRECTIONAL_DEBIT",
                    "mispricing_confidence": 0.65,
                    "pnl_estimate": -5.0 - i * 0.1,
                    "attribution_status": "resolved",
                }
            )
        ks = evaluate_kill_switches(records)
        assert ks["overlay_enabled"] is False
        assert any("sharpe" in r for r in ks["triggered_rules"])

    def test_kill_switch_confidence_monotonicity(self):
        # High confidence performs WORST
        low_conf = _make_resolved_records(n=10, win_rate=0.80, mean_pnl=3.0, confidence=0.40)
        high_conf = _make_resolved_records(n=15, win_rate=0.30, mean_pnl=-2.0, confidence=0.75)
        ks = evaluate_kill_switches(low_conf + high_conf)
        assert ks["sizing_enabled"] is False
        assert any("monotonicity" in r for r in ks["triggered_rules"])
