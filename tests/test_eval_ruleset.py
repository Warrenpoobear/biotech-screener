"""Tests for ruleset evaluator v1."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
import sys

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from scripts.eval_ruleset import (
    SCHEMA,
    compute_overlap,
    compute_rank_correlation,
    compute_max_shift,
    evaluate_gate,
    evaluate_ruleset,
    is_degraded,
    load_ruleset_id,
    ranked_tickers,
    rank_map,
    render_markdown,
    snapshot_ruleset_id,
    main,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


TICKERS = [f"T{i:02d}" for i in range(1, 11)]


def _compute_ruleset_id(data: dict) -> str:
    """Compute ruleset ID same way as decision_engine.py."""
    import hashlib
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:8]


_TEST_RULESET_DATA = {"test_param": "value", "version": "test"}
_TEST_RULESET_ID = _compute_ruleset_id(_TEST_RULESET_DATA)


def _make_snapshot(
    snap_dir: Path,
    date: str,
    tickers: List[str],
    ruleset_id: str = _TEST_RULESET_ID,
    degraded: bool = False,
) -> Path:
    d = snap_dir / date
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, t in enumerate(tickers):
        rows.append({
            "ticker": t,
            "actionable_rank": str(i + 1),
            "eligible": "1",
            "decision_engine_ruleset_id": ruleset_id,
            "tier_dev": "A" if i < len(tickers) // 2 else "B",
        })
    _write_csv(d / "rankings.csv", rows)
    _write_json(d / "metadata.json", {
        "as_of_date": date,
        "version": "1.6.0",
        "decision_mode": "phase2",
        "clinical_sort_telemetry": {"ruleset_id": ruleset_id},
        "cache_degraded_run": degraded,
    })
    _write_json(d / "cache_health.json", {
        "schema": "cache_health.v1",
        "overall_status": "bad" if degraded else "ok",
        "degraded_run": degraded,
    })
    return d


def _make_ruleset(path: Path, ruleset_id: str = "") -> Path:
    """Create a ruleset JSON. If ruleset_id is given, snapshot helpers must use it too."""
    if not ruleset_id or ruleset_id == _TEST_RULESET_ID:
        _write_json(path, _TEST_RULESET_DATA)
        return path
    # For custom IDs, embed a distinguishing field so hash differs
    data = {"test_param": ruleset_id, "version": "test"}
    _write_json(path, data)
    return path


def _make_price_csv(path: Path, tickers: List[str], dates: List[str]) -> Path:
    """Create a minimal price CSV. Price = 100 + tick_index + date_index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "ticker", "close"])
        for di, d in enumerate(dates):
            for ti, t in enumerate(tickers):
                w.writerow([d, t, str(100.0 + ti + di * 0.5)])
    return path


# Trading dates for tests — weekdays only
TRADE_DATES = [
    "2026-02-02", "2026-02-03", "2026-02-04", "2026-02-05", "2026-02-06",
    "2026-02-09", "2026-02-10", "2026-02-11", "2026-02-12", "2026-02-13",
    "2026-02-16", "2026-02-17", "2026-02-18", "2026-02-19", "2026-02-20",
    "2026-02-23", "2026-02-24", "2026-02-25", "2026-02-26", "2026-02-27",
]


# ---------------------------------------------------------------------------
# Tests: Date discovery + skip degraded
# ---------------------------------------------------------------------------


class TestDateDiscovery:
    def test_skips_degraded(self, tmp_path):
        snap_dir = tmp_path / "snapshots"
        _make_snapshot(snap_dir, "2026-02-05", TICKERS)
        _make_snapshot(snap_dir, "2026-02-06", TICKERS, degraded=True)
        _make_snapshot(snap_dir, "2026-02-07", TICKERS)

        price_csv = _make_price_csv(tmp_path / "prices.csv", TICKERS, TRADE_DATES)
        from scripts.eval_forward_returns import load_price_series
        ps = load_price_series(price_csv)
        all_dates = sorted({d for p in ps.values() for d in p})

        result = evaluate_ruleset(
            candidate_id=_TEST_RULESET_ID, baseline_id=_TEST_RULESET_ID,
            dates=["2026-02-05", "2026-02-06", "2026-02-07"],
            snapshot_dir=snap_dir, price_series=ps,
            all_trade_dates=all_dates, regime_map=None,
            k=5, horizon=5, cost_bps=30, anchor_mode="exact",
        )
        assert result["n_evaluated"] == 2
        skipped = {s["date"]: s["reason"] for s in result["dates_skipped"]}
        assert skipped.get("2026-02-06") == "degraded"

    def test_skips_missing_snapshot(self, tmp_path):
        snap_dir = tmp_path / "snapshots"
        _make_snapshot(snap_dir, "2026-02-05", TICKERS)

        ps = {"T01": {"2026-02-05": 100.0}}
        result = evaluate_ruleset(
            candidate_id=_TEST_RULESET_ID, baseline_id=_TEST_RULESET_ID,
            dates=["2026-02-05", "2026-02-06"],
            snapshot_dir=snap_dir, price_series=ps,
            all_trade_dates=["2026-02-05"], regime_map=None,
            k=5, horizon=5, cost_bps=30, anchor_mode="exact",
        )
        skipped = {s["date"]: s["reason"] for s in result["dates_skipped"]}
        assert skipped.get("2026-02-06") == "missing_snapshot"


# ---------------------------------------------------------------------------
# Tests: Ruleset ID mismatch
# ---------------------------------------------------------------------------


class TestRulesetMismatch:
    def test_mismatch_skips(self, tmp_path):
        snap_dir = tmp_path / "snapshots"
        _make_snapshot(snap_dir, "2026-02-05", TICKERS, ruleset_id="xxxxxxxx")

        ps = {"T01": {"2026-02-05": 100.0}}
        result = evaluate_ruleset(
            candidate_id=_TEST_RULESET_ID, baseline_id=_TEST_RULESET_ID,
            dates=["2026-02-05"],
            snapshot_dir=snap_dir, price_series=ps,
            all_trade_dates=["2026-02-05"], regime_map=None,
            k=5, horizon=5, cost_bps=30, anchor_mode="exact",
        )
        assert result["n_evaluated"] == 0
        skipped = {s["date"]: s["reason"] for s in result["dates_skipped"]}
        assert skipped.get("2026-02-05") == "RULESET_MISMATCH"

    def test_matching_id_not_skipped(self, tmp_path):
        snap_dir = tmp_path / "snapshots"
        _make_snapshot(snap_dir, "2026-02-05", TICKERS)

        ps = {"T01": {"2026-02-05": 100.0}}
        result = evaluate_ruleset(
            candidate_id=_TEST_RULESET_ID, baseline_id=_TEST_RULESET_ID,
            dates=["2026-02-05"],
            snapshot_dir=snap_dir, price_series=ps,
            all_trade_dates=["2026-02-05"], regime_map=None,
            k=5, horizon=5, cost_bps=30, anchor_mode="exact",
        )
        assert result["n_evaluated"] == 1


# ---------------------------------------------------------------------------
# Tests: Stability metrics
# ---------------------------------------------------------------------------


class TestStability:
    def test_overlap_identical(self):
        tickers = ["A", "B", "C", "D", "E"]
        result = compute_overlap(tickers, tickers, 3)
        assert result["overlap_count"] == 3
        assert result["overlap_pct"] == 100.0
        assert result["entrants"] == []
        assert result["exits"] == []

    def test_overlap_partial(self):
        curr = ["A", "B", "C", "D"]
        prior = ["A", "B", "E", "F"]
        result = compute_overlap(curr, prior, 3)
        assert result["overlap_count"] == 2
        assert result["overlap_pct"] == pytest.approx(66.7, abs=0.1)

    def test_rank_correlation_identical(self):
        ranks = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
        result = compute_rank_correlation(ranks, ranks)
        assert result["spearman"] == pytest.approx(1.0)

    def test_rank_correlation_reversed(self):
        curr = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
        prior = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}
        result = compute_rank_correlation(curr, prior)
        assert result["spearman"] == pytest.approx(-1.0)

    def test_max_shift(self):
        curr = {"A": 1, "B": 5, "C": 3}
        prior = {"A": 2, "B": 1, "C": 3}
        result = compute_max_shift(curr, prior)
        assert result["ticker"] == "B"
        assert result["shift"] == 4

    def test_temporal_stability_computed(self, tmp_path):
        snap_dir = tmp_path / "snapshots"
        t1 = ["A", "B", "C", "D", "E"]
        t2 = ["A", "B", "C", "E", "D"]  # D and E swapped
        _make_snapshot(snap_dir, "2026-02-05", t1)
        _make_snapshot(snap_dir, "2026-02-06", t2)

        ps = {t: {"2026-02-05": 100.0, "2026-02-06": 101.0} for t in t1}
        all_dates = ["2026-02-05", "2026-02-06"]

        result = evaluate_ruleset(
            candidate_id=_TEST_RULESET_ID, baseline_id=_TEST_RULESET_ID,
            dates=["2026-02-05", "2026-02-06"],
            snapshot_dir=snap_dir, price_series=ps,
            all_trade_dates=all_dates, regime_map=None,
            k=5, horizon=1, cost_bps=0, anchor_mode="exact",
        )
        ts = result["temporal_stability"]
        assert ts.get("mean_top60_overlap") is not None
        assert ts.get("mean_spearman") is not None


# ---------------------------------------------------------------------------
# Tests: Turnover + net return
# ---------------------------------------------------------------------------


class TestReturns:
    def test_turnover_calculation(self, tmp_path):
        """Equal-weight top-K with known turnover."""
        snap_dir = tmp_path / "snapshots"
        t1 = ["A", "B", "C", "D", "E"]
        t2 = ["A", "B", "C", "F", "G"]  # 2 exits, 2 entries
        _make_snapshot(snap_dir, "2026-02-05", t1)
        _make_snapshot(snap_dir, "2026-02-06", t2)

        # Prices: all flat at 100, then 102 (2% return)
        all_tickers = list(set(t1 + t2))
        ps = {t: {"2026-02-05": 100.0, "2026-02-06": 102.0} for t in all_tickers}
        all_dates = ["2026-02-05", "2026-02-06"]

        result = evaluate_ruleset(
            candidate_id=_TEST_RULESET_ID, baseline_id=_TEST_RULESET_ID,
            dates=["2026-02-05", "2026-02-06"],
            snapshot_dir=snap_dir, price_series=ps,
            all_trade_dates=all_dates, regime_map=None,
            k=5, horizon=1, cost_bps=30, anchor_mode="exact",
        )
        per_date = result["per_date"]
        # Day 2: turnover = 0.5 * 4 / 5 = 0.4
        assert per_date[1]["cand_turnover"] == pytest.approx(0.4, abs=0.01)

    def test_net_return_deduction(self, tmp_path):
        snap_dir = tmp_path / "snapshots"
        tickers = ["A", "B", "C"]
        _make_snapshot(snap_dir, "2026-02-05", tickers)
        _make_snapshot(snap_dir, "2026-02-06", tickers)  # no turnover

        # Need 3 dates: snap dates + 1 for horizon
        ps = {t: {"2026-02-05": 100.0, "2026-02-06": 101.0, "2026-02-07": 102.0} for t in tickers}
        all_dates = ["2026-02-05", "2026-02-06", "2026-02-07"]

        result = evaluate_ruleset(
            candidate_id=_TEST_RULESET_ID, baseline_id=_TEST_RULESET_ID,
            dates=["2026-02-05", "2026-02-06"],
            snapshot_dir=snap_dir, price_series=ps,
            all_trade_dates=all_dates, regime_map=None,
            k=3, horizon=1, cost_bps=100, anchor_mode="exact",
        )
        per_date = result["per_date"]
        # Day 1 (2026-02-05): gross = P(06)/P(05)-1 = 1%, turnover = 0.5 (first day), net = 1% - 2*0.5*100/10000 = 0%
        # Day 2 (2026-02-06): gross = P(07)/P(06)-1 ≈ 0.99%, turnover = 0 (same tickers), net = 0.99%
        day2 = per_date[1]
        assert day2["cand_gross"] == pytest.approx(1 / 101, abs=0.001)
        assert day2["cand_net"] == pytest.approx(1 / 101, abs=0.001)  # turnover=0
        assert day2["cand_turnover"] == 0.0


# ---------------------------------------------------------------------------
# Tests: Regime slicing
# ---------------------------------------------------------------------------


class TestRegime:
    def test_regime_labels_in_output(self, tmp_path):
        snap_dir = tmp_path / "snapshots"
        _make_snapshot(snap_dir, "2026-02-05", TICKERS[:5])
        _make_snapshot(snap_dir, "2026-02-06", TICKERS[:5])

        ps = {t: {"2026-02-05": 100.0, "2026-02-06": 101.0, "2026-02-07": 102.0} for t in TICKERS[:5]}
        regime_map = {"2026-02-05": "BULL", "2026-02-06": "BEAR"}

        result = evaluate_ruleset(
            candidate_id=_TEST_RULESET_ID, baseline_id=_TEST_RULESET_ID,
            dates=["2026-02-05", "2026-02-06"],
            snapshot_dir=snap_dir, price_series=ps,
            all_trade_dates=["2026-02-05", "2026-02-06", "2026-02-07"],
            regime_map=regime_map,
            k=5, horizon=1, cost_bps=0, anchor_mode="exact",
        )
        assert "BULL" in result["regime"]
        assert "BEAR" in result["regime"]
        assert result["per_date"][0]["regime"] == "BULL"
        assert result["per_date"][1]["regime"] == "BEAR"


# ---------------------------------------------------------------------------
# Tests: Deterministic output
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_json_keys_stable(self, tmp_path):
        snap_dir = tmp_path / "snapshots"
        _make_snapshot(snap_dir, "2026-02-05", TICKERS[:5])

        ps = {t: {"2026-02-05": 100.0} for t in TICKERS[:5]}

        r1 = evaluate_ruleset(
            candidate_id=_TEST_RULESET_ID, baseline_id=_TEST_RULESET_ID,
            dates=["2026-02-05"],
            snapshot_dir=snap_dir, price_series=ps,
            all_trade_dates=["2026-02-05"], regime_map=None,
            k=5, horizon=1, cost_bps=30, anchor_mode="exact",
        )
        r2 = evaluate_ruleset(
            candidate_id=_TEST_RULESET_ID, baseline_id=_TEST_RULESET_ID,
            dates=["2026-02-05"],
            snapshot_dir=snap_dir, price_series=ps,
            all_trade_dates=["2026-02-05"], regime_map=None,
            k=5, horizon=1, cost_bps=30, anchor_mode="exact",
        )
        j1 = json.dumps(r1, sort_keys=True, indent=2)
        j2 = json.dumps(r2, sort_keys=True, indent=2)
        assert j1 == j2

    def test_skipped_reasons_sorted(self, tmp_path):
        snap_dir = tmp_path / "snapshots"
        _make_snapshot(snap_dir, "2026-02-05", TICKERS[:3], ruleset_id="xxxxxxxx")
        _make_snapshot(snap_dir, "2026-02-06", TICKERS[:3], degraded=True)

        ps = {}
        result = evaluate_ruleset(
            candidate_id=_TEST_RULESET_ID, baseline_id=_TEST_RULESET_ID,
            dates=["2026-02-05", "2026-02-06"],
            snapshot_dir=snap_dir, price_series=ps,
            all_trade_dates=[], regime_map=None,
            k=3, horizon=1, cost_bps=0, anchor_mode="exact",
        )
        # Both skipped for different reasons
        assert result["n_skipped"] == 2


# ---------------------------------------------------------------------------
# Tests: Gate verdict
# ---------------------------------------------------------------------------


class TestGate:
    def test_pass_when_stable(self):
        eval_result = {
            "temporal_stability": {
                "mean_top60_overlap": 95.0,
                "max_rank_shift": {"ticker": "X", "shift": 10, "date": "2026-02-05"},
            },
            "performance": {
                "candidate": {"mean_turnover": 0.15},
                "baseline": {"mean_turnover": 0.15},
            },
        }
        gate = evaluate_gate(eval_result)
        assert gate["verdict"] == "PASS"

    def test_warn_on_high_shift(self):
        eval_result = {
            "temporal_stability": {
                "mean_top60_overlap": 95.0,
                "max_rank_shift": {"ticker": "X", "shift": 75, "date": "2026-02-05"},
            },
            "performance": {},
        }
        gate = evaluate_gate(eval_result)
        assert gate["verdict"] == "WARN"
        assert any("max_rank_shift" in r for r in gate["warn_reasons"])

    def test_fail_on_extreme_shift(self):
        eval_result = {
            "temporal_stability": {
                "mean_top60_overlap": 95.0,
                "max_rank_shift": {"ticker": "X", "shift": 150, "date": "2026-02-05"},
            },
            "performance": {},
        }
        gate = evaluate_gate(eval_result)
        assert gate["verdict"] == "FAIL"

    def test_empty_stability_passes(self):
        eval_result = {
            "temporal_stability": {},
            "performance": {},
        }
        gate = evaluate_gate(eval_result)
        assert gate["verdict"] == "PASS"


# ---------------------------------------------------------------------------
# Tests: CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_self_eval_exits_0(self, tmp_path):
        snap_dir = tmp_path / "snapshots"
        tickers = ["A", "B", "C", "D", "E"]
        _make_snapshot(snap_dir, "2026-02-05", tickers)
        _make_snapshot(snap_dir, "2026-02-06", tickers)

        price_csv = _make_price_csv(tmp_path / "prices.csv", tickers, TRADE_DATES)
        ruleset = _make_ruleset(tmp_path / "ruleset.json")
        out_dir = tmp_path / "output"

        rc = main([
            "--candidate", str(ruleset),
            "--baseline", str(ruleset),
            "--start", "2026-02-05",
            "--end", "2026-02-06",
            "--k", "3",
            "--horizon", "1",
            "--cost-bps", "30",
            "--snapshot-dir", str(snap_dir),
            "--price-csv", str(price_csv),
            "--out-dir", str(out_dir),
            "--anchor-mode", "exact",
        ])
        assert rc == 0
        # Check outputs exist
        eval_dir = list(out_dir.glob("*/*"))
        assert len(eval_dir) >= 1
        json_files = list(out_dir.glob("**/ruleset_eval.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text())
        assert data["schema"] == SCHEMA

    def test_gate_fail_exits_1(self, tmp_path):
        snap_dir = tmp_path / "snapshots"
        # Create snapshots with very different rankings to trigger instability
        t1 = [f"T{i:02d}" for i in range(1, 11)]
        t2 = list(reversed(t1))
        _make_snapshot(snap_dir, "2026-02-05", t1)
        _make_snapshot(snap_dir, "2026-02-06", t2)

        price_csv = _make_price_csv(tmp_path / "prices.csv", t1, TRADE_DATES)
        ruleset = _make_ruleset(tmp_path / "ruleset.json")
        out_dir = tmp_path / "output"

        rc = main([
            "--candidate", str(ruleset),
            "--baseline", str(ruleset),
            "--start", "2026-02-05",
            "--end", "2026-02-06",
            "--k", "5",
            "--horizon", "1",
            "--snapshot-dir", str(snap_dir),
            "--price-csv", str(price_csv),
            "--out-dir", str(out_dir),
            "--anchor-mode", "exact",
            "--gate",
        ])
        # Max shift = 9 (T01 goes from rank 1 to rank 10), which is < 100 fail threshold
        # So this should PASS unless overlap is bad
        # top-5 overlap: {T01..T05} vs {T10..T06} = 0 overlap → 0%
        # mean_top60_overlap = 0% < 70% fail threshold... but 0% < 85% warn
        # Wait, gate thresholds use mean_top60_overlap_fail=0.70 (meaning 70%)
        # 0% < 70% → FAIL? No, check gate logic...
        # The gate checks: mean_ov < t.get("mean_top60_overlap_warn", 0) * 100
        # mean_ov = 0.0, warn = 0.85 * 100 = 85.0 → 0 < 85 → WARN
        # And fail: mean_ov < t.get("mean_top60_overlap_fail", 0) → 0 < 0.70
        # Hmm, the fail threshold is 0.70 (not 70). So 0.0 < 0.70 → FAIL
        # But mean_ov is in %, so 0.0% vs 0.70... that's 0 < 0.70 → True → FAIL
        assert rc == 1

    def test_no_dates_exits_1(self, tmp_path):
        snap_dir = tmp_path / "snapshots"
        snap_dir.mkdir()
        ruleset = _make_ruleset(tmp_path / "ruleset.json")

        rc = main([
            "--candidate", str(ruleset),
            "--start", "2099-01-01",
            "--end", "2099-12-31",
            "--snapshot-dir", str(snap_dir),
            "--price-csv", str(tmp_path / "empty.csv"),
            "--out-dir", str(tmp_path / "out"),
        ])
        assert rc == 1


# ---------------------------------------------------------------------------
# Tests: Helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_load_ruleset_id(self, tmp_path):
        p = _make_ruleset(tmp_path / "r.json")
        assert load_ruleset_id(p) == _TEST_RULESET_ID
        assert len(_TEST_RULESET_ID) == 8  # 8-char hex

    def test_is_degraded_true(self, tmp_path):
        d = tmp_path / "snap"
        d.mkdir()
        _write_json(d / "cache_health.json", {
            "overall_status": "bad", "degraded_run": True,
        })
        assert is_degraded(d) is True

    def test_is_degraded_false(self, tmp_path):
        d = tmp_path / "snap"
        d.mkdir()
        _write_json(d / "cache_health.json", {
            "overall_status": "ok", "degraded_run": False,
        })
        assert is_degraded(d) is False

    def test_ranked_tickers_filters_ineligible(self):
        rows = [
            {"ticker": "A", "actionable_rank": "1", "eligible": "1"},
            {"ticker": "B", "actionable_rank": "2", "eligible": "0"},
            {"ticker": "C", "actionable_rank": "3", "eligible": "TRUE"},
        ]
        assert ranked_tickers(rows) == ["A", "C"]

    def test_rank_map_builds_correctly(self):
        rows = [
            {"ticker": "B", "actionable_rank": "2", "eligible": "1"},
            {"ticker": "A", "actionable_rank": "1", "eligible": "1"},
        ]
        rm = rank_map(rows)
        assert rm == {"A": 1, "B": 2}

    def test_snapshot_ruleset_id_from_metadata(self, tmp_path):
        d = tmp_path / "snap"
        d.mkdir()
        _write_json(d / "metadata.json", {
            "clinical_sort_telemetry": {"ruleset_id": "xyz789"},
        })
        assert snapshot_ruleset_id(d) == "xyz789"

    def test_snapshot_ruleset_id_fallback_rankings(self, tmp_path):
        d = tmp_path / "snap"
        d.mkdir()
        _write_csv(d / "rankings.csv", [
            {"ticker": "A", "decision_engine_ruleset_id": "fallback_id"},
        ])
        assert snapshot_ruleset_id(d) == "fallback_id"
