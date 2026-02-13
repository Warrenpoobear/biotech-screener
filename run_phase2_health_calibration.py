#!/usr/bin/env python3
"""Phase-2 Health Gate Calibration.

Replays Phase-2 deltas across the historical snapshot/archive chain and measures
health gate behavior. Answers:
  - How often would we have hit WARN / FAIL historically?
  - Which reason tags dominate?
  - What are the empirical distributions for key metrics?

Approach:
  1. Load each archive, recompute decision engine fields (same as sweep/backtest)
  2. Build SnapshotData objects (rankings DF + reconstructed portfolio)
  3. For consecutive pairs, compute delta + health gate
  4. Collect per-date metrics, compute quantiles, write outputs

Outputs:
  - output/phase2_health_calibration_summary.csv
  - output/phase2_health_calibration_details.json
  - output/phase2_health_threshold_recommendation.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.cost_model import CostSchedule, estimate_trade_cost
from decision_engine import (
    DecisionRuleset,
    compute_actionable_sort_key,
    compute_decision_fields,
    compute_target_weights,
)
from run_decision_ruleset_sweep import (
    ArchiveData,
    load_archive_data,
)
from run_phase2_snapshot_delta import (
    CatalystCoverage,
    DeltaResult,
    HealthResult,
    Phase2HealthThresholds,
    DEFAULT_HEALTH_THRESHOLDS,
    SingleResult,
    SnapshotData,
    compute_delta,
    compute_health_gate,
    compute_single_snapshot_summary,
    generate_health_json,
    PHASE2_PINNED_RULESET_ID,
    RECON_TIER_FILTER,
    RECON_TOP_K,
)

ARCHIVE_DIR = PROJECT_ROOT / "data" / "archives"
PHASE2_RULESET_PATH = (
    PROJECT_ROOT / "production_data" / "decision_rulesets" / "v1.3.0_candidate.json"
)
PHASE2_HEALTH_THRESHOLDS_PATH = (
    PROJECT_ROOT / "production_data" / "phase2_health_thresholds" / "v1.json"
)
OUTPUT_DIR = PROJECT_ROOT / "output"


# ---------------------------------------------------------------------------
# Archive → SnapshotData conversion
# ---------------------------------------------------------------------------

def _build_snapshot_from_archive(
    archive_data: ArchiveData,
    ruleset: DecisionRuleset,
    tier_filter: Tuple[str, ...] = RECON_TIER_FILTER,
    top_k: int = RECON_TOP_K,
) -> Optional[SnapshotData]:
    """Recompute decision fields for an archive and build a SnapshotData.

    This mirrors the logic in build_strategy_portfolio() from the strategy backtest
    but produces a full rankings DataFrame + portfolio DataFrame suitable for
    the delta/health gate machinery.
    """
    # --- Build full rankings DataFrame with decision columns ---
    rank_by_ticker: Dict[str, int] = {}
    for row in archive_data.rows:
        ticker = row.get("ticker", "").strip().upper()
        rank_str = row.get("composite_rank", "").strip()
        if ticker and rank_str:
            try:
                rank_by_ticker[ticker] = int(rank_str)
            except ValueError:
                pass

    # Cost-aware sizing: initialize cost schedule when enabled
    cost_schedule = None
    rep_weight = 0.0
    if ruleset.enable_cost_haircut:
        cost_schedule = CostSchedule(
            impact_cap_bps=ruleset.cost_impact_cap_bps,
        )
        rep_weight = 100.0 / max(top_k, 1)

    ranking_rows: List[Dict[str, Any]] = []
    for row in archive_data.rows:
        ticker = row.get("ticker", "").strip().upper()
        if not ticker:
            continue

        archetype = archive_data.archetypes.get(ticker, "")
        rec = archive_data.recs.get(ticker)
        opt = archive_data.optionalities.get(ticker)

        # Compute est_cost_bps from archive market data when cost haircut is on
        est_cost_bps = None
        if cost_schedule is not None:
            md = archive_data.market_data.get(ticker, {})
            adv = (md.get("avg_volume") or 0) * (md.get("price") or 0)
            if adv > 0:
                cost_est = estimate_trade_cost(rep_weight, adv, cost_schedule)
                est_cost_bps = cost_est.round_trip_bps

        # Compute decision fields (only meaningful for drug_developer)
        if archetype == "drug_developer" and rec is not None:
            fields = compute_decision_fields(
                rec, archetype, opt, ruleset=ruleset, est_cost_bps=est_cost_bps,
            )
        else:
            fields = {}

        composite_rank = rank_by_ticker.get(ticker)
        ranking_rows.append({
            "ticker": ticker,
            "archetype": archetype,
            "composite_rank": composite_rank or 999,
            "composite_score": row.get("composite_score", ""),
            "tier_dev": fields.get("tier_dev", ""),
            "eligible": fields.get("eligible", ""),
            "catalyst_mode": fields.get("catalyst_mode", ""),
            "catalyst_days": fields.get("catalyst_days", ""),
            "size_band": fields.get("size_band", ""),
            "risk_flags": fields.get("risk_flags", ""),
            "mom_state": fields.get("mom_state", ""),
            "tier_reason": fields.get("tier_reason", ""),
            "actionable_rank": "",
            "target_weight_pct": "",
            "decision_engine_ruleset_id": ruleset.ruleset_id,
            "decision_engine_version": "v1.2.0",
            "clinical_optionality_pct_dev": opt if opt is not None else "",
        })

    if not ranking_rows:
        return None

    # --- Build portfolio (filter → sort → top-K → weight) ---
    # Same logic as build_strategy_portfolio but we keep the full rankings DF
    candidates: List[Tuple[Any, Dict[str, Any]]] = []
    for rr in ranking_rows:
        if rr["archetype"] != "drug_developer":
            continue
        if rr["eligible"] != "1":
            continue
        tier = rr["tier_dev"]
        if tier not in tier_filter:
            continue

        ticker = rr["ticker"]
        rec = archive_data.recs.get(ticker)
        opt = archive_data.optionalities.get(ticker)
        if rec is None:
            continue

        # Recompute cost for this candidate (same schedule as above)
        cand_cost_bps = None
        if cost_schedule is not None:
            md = archive_data.market_data.get(ticker, {})
            adv = (md.get("avg_volume") or 0) * (md.get("price") or 0)
            if adv > 0:
                cost_est = estimate_trade_cost(rep_weight, adv, cost_schedule)
                cand_cost_bps = cost_est.round_trip_bps

        fields = compute_decision_fields(
            rec, rr["archetype"], opt, ruleset=ruleset, est_cost_bps=cand_cost_bps,
        )
        composite_rank = rank_by_ticker.get(ticker)
        sort_key = compute_actionable_sort_key(
            fields, rr["archetype"], opt, composite_rank, ticker,
            catalyst_event_type=rec.get("catalyst_event_type", ""),
            catalyst_source=rec.get("catalyst_source", ""),
            ruleset=ruleset,
        )
        candidates.append((sort_key, rr))

    candidates.sort(key=lambda x: x[0])
    selected = [c[1] for c in candidates[:top_k]]

    # Assign actionable ranks
    for i, pos in enumerate(selected, 1):
        pos["actionable_rank"] = i

    # Compute target weights
    weight_input = [
        {"ticker": p["ticker"], "size_band": p["size_band"]}
        for p in selected
    ]
    weighted = compute_target_weights(weight_input, ruleset=ruleset)
    for p, w in zip(selected, weighted):
        p["target_weight_pct"] = w.get("target_weight_pct", 0.0)

    # Backfill actionable_rank and target_weight into rankings rows
    selected_map = {p["ticker"]: p for p in selected}
    for rr in ranking_rows:
        sel = selected_map.get(rr["ticker"])
        if sel:
            rr["actionable_rank"] = sel["actionable_rank"]
            rr["target_weight_pct"] = sel["target_weight_pct"]

    # Build DataFrames
    rankings_df = pd.DataFrame(ranking_rows)
    portfolio_df = pd.DataFrame(selected) if selected else pd.DataFrame(
        columns=list(ranking_rows[0].keys())
    )

    return SnapshotData(
        date=archive_data.date_str,
        path=Path(f"/archive/{archive_data.date_str}"),
        rankings=rankings_df,
        portfolio=portfolio_df,
        metadata={"as_of_date": archive_data.date_str, "source": "archive"},
        ruleset_id=ruleset.ruleset_id,
        has_native_portfolio=False,
    )


# ---------------------------------------------------------------------------
# Quantile helpers
# ---------------------------------------------------------------------------

def _quantiles(values: List[float]) -> Dict[str, float]:
    """Compute P5, P25, P50, P75, P90, P95 for a list of values."""
    if not values:
        return {k: 0.0 for k in ["min", "p5", "p25", "p50", "p75", "p90", "p95", "max", "mean"]}
    s = sorted(values)
    n = len(s)

    def pct(p: float) -> float:
        idx = p / 100 * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return round(s[lo] * (1 - frac) + s[hi] * frac, 2)

    return {
        "min": round(s[0], 2),
        "p5": pct(5),
        "p25": pct(25),
        "p50": pct(50),
        "p75": pct(75),
        "p90": pct(90),
        "p95": pct(95),
        "max": round(s[-1], 2),
        "mean": round(mean(s), 2),
    }


# ---------------------------------------------------------------------------
# Main calibration
# ---------------------------------------------------------------------------

def run_calibration(
    archive_dir: Path,
    ruleset: DecisionRuleset,
    output_dir: Path,
    health_thresholds: Optional[Phase2HealthThresholds] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> None:
    """Run health gate calibration across archive chain."""
    ht = health_thresholds or DEFAULT_HEALTH_THRESHOLDS

    # Discover archives
    archive_files = sorted(archive_dir.glob("*.tar.gz"))
    if not archive_files:
        print("ERROR: No archives found", file=sys.stderr)
        sys.exit(1)

    dates = []
    for af in archive_files:
        date_str = af.stem.replace(".tar", "")
        if start_date and date_str < start_date:
            continue
        if end_date and date_str > end_date:
            continue
        dates.append((date_str, af))

    print(f"Archives: {len(dates)} found ({dates[0][0]} to {dates[-1][0]})")
    print(f"Ruleset: {ruleset.ruleset_id}")
    print(f"Health thresholds: {ht.thresholds_id}")

    # --- Phase 1: Load all archives and build SnapshotData objects ---
    snapshots: List[SnapshotData] = []
    for i, (date_str, tar_path) in enumerate(dates):
        print(f"  [{i+1}/{len(dates)}] Loading {date_str} ... ", end="", flush=True)
        try:
            archive_data = load_archive_data(tar_path, date_str)
            snap = _build_snapshot_from_archive(archive_data, ruleset)
            if snap is None:
                print("SKIP (no data)")
                continue
            n_port = len(snap.portfolio)
            print(f"OK ({n_port} portfolio names)")
            snapshots.append(snap)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

    if len(snapshots) < 2:
        print("ERROR: Need at least 2 loadable snapshots for calibration", file=sys.stderr)
        sys.exit(1)

    print(f"\nLoaded {len(snapshots)} snapshots, computing {len(snapshots)-1} deltas...")

    # --- Phase 2: Compute deltas + health for consecutive pairs ---
    per_date_records: List[Dict[str, Any]] = []

    for idx in range(len(snapshots)):
        current = snapshots[idx]

        if idx == 0:
            # First snapshot: single-snapshot mode
            result = compute_single_snapshot_summary(current)
            health = compute_health_gate(current, None, result, thresholds=ht)
            per_date_records.append({
                "date": current.date,
                "mode": "single",
                "status": health.status,
                "reasons": health.reasons,
                "metrics": health.metrics,
                "portfolio_size": len(current.portfolio),
                # Delta-specific (N/A for single)
                "name_turnover_pct": None,
                "weight_l1_delta": None,
                "catalyst_coverage_pct": health.metrics.get("catalyst_coverage_pct"),
                "a_count": health.metrics.get("a_count"),
                "catalyst_drop_pp": None,
                "n_entrants": None,
                "n_exits": None,
                "dev_optionality_coverage_pct": health.metrics.get("dev_optionality_coverage_pct"),
                "dev_above_a_floor_count": health.metrics.get("dev_above_a_floor_count"),
            })
        else:
            prior = snapshots[idx - 1]
            result = compute_delta(current, prior)
            health = compute_health_gate(current, prior, result, thresholds=ht)

            # Compute catalyst drop
            cat_drop = result.catalyst_coverage_prior.pct - result.catalyst_coverage_current.pct

            per_date_records.append({
                "date": current.date,
                "mode": "delta",
                "status": health.status,
                "reasons": health.reasons,
                "metrics": health.metrics,
                "portfolio_size": len(current.portfolio),
                "name_turnover_pct": result.name_turnover_pct,
                "weight_l1_delta": result.weight_l1_delta,
                "catalyst_coverage_pct": result.catalyst_coverage_current.pct,
                "a_count": result.tier_counts_current.get("A", 0),
                "catalyst_drop_pp": round(cat_drop, 1),
                "n_entrants": len(result.entrants),
                "n_exits": len(result.exits),
                "dev_optionality_coverage_pct": health.metrics.get("dev_optionality_coverage_pct"),
                "dev_above_a_floor_count": health.metrics.get("dev_above_a_floor_count"),
            })

    # --- Phase 3: Compute summary statistics ---
    delta_records = [r for r in per_date_records if r["mode"] == "delta"]
    all_records = per_date_records

    # Status counts
    status_counts = Counter(r["status"] for r in all_records)
    delta_status_counts = Counter(r["status"] for r in delta_records)

    # Reason frequency
    reason_counts = Counter()
    for r in all_records:
        for reason in r["reasons"]:
            reason_counts[reason] += 1

    # Metric distributions (delta mode only, where the metrics are meaningful)
    turnover_vals = [r["name_turnover_pct"] for r in delta_records if r["name_turnover_pct"] is not None]
    weight_l1_vals = [r["weight_l1_delta"] for r in delta_records if r["weight_l1_delta"] is not None]
    cat_cov_vals = [r["catalyst_coverage_pct"] for r in all_records if r["catalyst_coverage_pct"] is not None]
    a_count_vals = [r["a_count"] for r in all_records if r["a_count"] is not None]
    cat_drop_vals = [r["catalyst_drop_pp"] for r in delta_records if r["catalyst_drop_pp"] is not None]

    distributions = {
        "turnover_pct": _quantiles(turnover_vals),
        "weight_l1_pct": _quantiles(weight_l1_vals),
        "catalyst_coverage_pct": _quantiles(cat_cov_vals),
        "a_count": _quantiles([float(v) for v in a_count_vals]),
        "catalyst_drop_pp": _quantiles(cat_drop_vals),
    }

    # --- Phase 4: Write outputs ---
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4a. Summary CSV
    summary_csv_path = output_dir / "phase2_health_calibration_summary.csv"
    _write_summary_csv(
        summary_csv_path, status_counts, delta_status_counts,
        reason_counts, distributions, len(all_records), len(delta_records),
    )
    print(f"  -> {summary_csv_path}")

    # 4b. Details JSON
    details_json_path = output_dir / "phase2_health_calibration_details.json"
    details = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ruleset_id": ruleset.ruleset_id,
        "n_snapshots": len(all_records),
        "n_deltas": len(delta_records),
        "status_counts": dict(status_counts),
        "delta_status_counts": dict(delta_status_counts),
        "reason_frequency": dict(reason_counts.most_common()),
        "distributions": distributions,
        "thresholds_id": ht.thresholds_id,
        "thresholds_tested": {
            "fail_catalyst_coverage_min": ht.fail_catalyst_coverage_min,
            "fail_a_count_min": ht.fail_a_count_min,
            "fail_optionality_coverage_min": ht.fail_optionality_coverage_min,
            "warn_a_count_low": ht.warn_a_count_low,
            "warn_turnover_pct": ht.warn_turnover_pct,
            "warn_weight_l1_pct": ht.warn_weight_l1_pct,
            "warn_catalyst_drop_pp": ht.warn_catalyst_drop_pp,
        },
        "per_date": [
            {
                "date": r["date"],
                "mode": r["mode"],
                "status": r["status"],
                "reasons": r["reasons"],
                "portfolio_size": r["portfolio_size"],
                "name_turnover_pct": r["name_turnover_pct"],
                "weight_l1_delta": r["weight_l1_delta"],
                "catalyst_coverage_pct": r["catalyst_coverage_pct"],
                "a_count": r["a_count"],
                "catalyst_drop_pp": r["catalyst_drop_pp"],
                "n_entrants": r["n_entrants"],
                "n_exits": r["n_exits"],
                "dev_optionality_coverage_pct": r.get("dev_optionality_coverage_pct"),
                "dev_above_a_floor_count": r.get("dev_above_a_floor_count"),
            }
            for r in per_date_records
        ],
    }
    with open(details_json_path, "w") as f:
        json.dump(details, f, indent=2)
    print(f"  -> {details_json_path}")

    # 4c. Threshold recommendation report
    report_path = output_dir / "phase2_health_threshold_recommendation.txt"
    report = _generate_recommendation_report(
        status_counts, delta_status_counts, reason_counts,
        distributions, per_date_records, ruleset, ht,
    )
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  -> {report_path}")

    # Print summary to stdout
    print()
    print(report)


def _write_summary_csv(
    path: Path,
    status_counts: Counter,
    delta_status_counts: Counter,
    reason_counts: Counter,
    distributions: Dict[str, Dict[str, float]],
    n_total: int,
    n_deltas: int,
) -> None:
    """Write calibration summary CSV with three sections."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        # Section 1: Status frequency
        writer.writerow(["section", "key", "count", "pct_of_total"])
        for status in ["OK", "WARN", "FAIL"]:
            count = status_counts.get(status, 0)
            pct = round(count / n_total * 100, 1) if n_total else 0
            writer.writerow(["status_all", status, count, pct])
        for status in ["OK", "WARN", "FAIL"]:
            count = delta_status_counts.get(status, 0)
            pct = round(count / n_deltas * 100, 1) if n_deltas else 0
            writer.writerow(["status_delta", status, count, pct])

        # Section 2: Reason frequency
        writer.writerow([])
        writer.writerow(["section", "reason", "count", "pct_of_total"])
        for reason, count in reason_counts.most_common():
            pct = round(count / n_total * 100, 1) if n_total else 0
            writer.writerow(["reason", reason, count, pct])

        # Section 3: Distributions
        writer.writerow([])
        writer.writerow(["section", "metric", "min", "p5", "p25", "p50", "p75", "p90", "p95", "max", "mean"])
        for metric, dist in distributions.items():
            writer.writerow([
                "distribution", metric,
                dist["min"], dist["p5"], dist["p25"], dist["p50"],
                dist["p75"], dist["p90"], dist["p95"], dist["max"], dist["mean"],
            ])


def _generate_recommendation_report(
    status_counts: Counter,
    delta_status_counts: Counter,
    reason_counts: Counter,
    distributions: Dict[str, Dict[str, float]],
    per_date_records: List[Dict[str, Any]],
    ruleset: DecisionRuleset,
    ht: Phase2HealthThresholds,
) -> str:
    """Generate human-readable threshold recommendation report."""
    W = 72
    lines: List[str] = []
    n_total = sum(status_counts.values())
    n_deltas = sum(delta_status_counts.values())

    lines.append("=" * W)
    lines.append("PHASE-2 HEALTH GATE THRESHOLD CALIBRATION")
    lines.append("=" * W)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines.append(f"Generated: {now}")
    lines.append(f"Ruleset:   {ruleset.ruleset_id}")
    lines.append(f"Thresholds: {ht.thresholds_id}")
    lines.append(f"Snapshots: {n_total} total, {n_deltas} delta pairs")
    lines.append("")

    # --- Section 1: Status distribution ---
    lines.append("1. HEALTH STATUS DISTRIBUTION")
    lines.append("-" * W)
    lines.append(f"  {'Status':8s} {'All':>8s} {'%':>7s}   {'Delta':>8s} {'%':>7s}")
    for status in ["OK", "WARN", "FAIL"]:
        all_n = status_counts.get(status, 0)
        all_p = round(all_n / n_total * 100, 1) if n_total else 0
        delta_n = delta_status_counts.get(status, 0)
        delta_p = round(delta_n / n_deltas * 100, 1) if n_deltas else 0
        lines.append(f"  {status:8s} {all_n:8d} {all_p:6.1f}%   {delta_n:8d} {delta_p:6.1f}%")
    lines.append("")

    # --- Section 2: Reason breakdown ---
    lines.append("2. REASON TAG FREQUENCY")
    lines.append("-" * W)
    if reason_counts:
        lines.append(f"  {'Reason':25s} {'Count':>6s} {'%':>7s}")
        for reason, count in reason_counts.most_common():
            pct = round(count / n_total * 100, 1) if n_total else 0
            lines.append(f"  {reason:25s} {count:6d} {pct:6.1f}%")
    else:
        lines.append("  (no WARN/FAIL events)")
    lines.append("")

    # --- Section 3: Metric distributions ---
    lines.append("3. EMPIRICAL DISTRIBUTIONS")
    lines.append("-" * W)
    metric_labels = {
        "turnover_pct": "Name Turnover %",
        "weight_l1_pct": "Weight L1 %",
        "catalyst_coverage_pct": "Catalyst Coverage %",
        "a_count": "A-tier Count",
        "catalyst_drop_pp": "Catalyst Drop (pp)",
    }
    lines.append(f"  {'Metric':22s} {'P50':>7s} {'P90':>7s} {'P95':>7s} {'Max':>7s} {'Mean':>7s}")
    for key, label in metric_labels.items():
        d = distributions.get(key, {})
        lines.append(
            f"  {label:22s} {d.get('p50', 0):7.1f} {d.get('p90', 0):7.1f} "
            f"{d.get('p95', 0):7.1f} {d.get('max', 0):7.1f} {d.get('mean', 0):7.1f}"
        )
    lines.append("")

    # --- Section 4: Current thresholds vs data ---
    lines.append("4. CURRENT THRESHOLDS vs EMPIRICAL DATA")
    lines.append("-" * W)
    threshold_vs_data = [
        ("Turnover WARN", ht.warn_turnover_pct, distributions.get("turnover_pct", {}), "above"),
        ("Weight L1 WARN", ht.warn_weight_l1_pct, distributions.get("weight_l1_pct", {}), "above"),
        ("Catalyst FAIL", ht.fail_catalyst_coverage_min, distributions.get("catalyst_coverage_pct", {}), "below"),
        ("A-count FAIL", float(ht.fail_a_count_min), distributions.get("a_count", {}), "below"),
        ("A-count WARN", float(ht.warn_a_count_low), distributions.get("a_count", {}), "below"),
        ("Cat Drop WARN", ht.warn_catalyst_drop_pp, distributions.get("catalyst_drop_pp", {}), "above"),
    ]
    lines.append(f"  {'Gate':20s} {'Threshold':>10s} {'Direction':>10s} {'P50':>7s} {'P90':>7s} {'P95':>7s}")
    for label, threshold, dist, direction in threshold_vs_data:
        lines.append(
            f"  {label:20s} {threshold:10.1f} {'>' if direction == 'above' else '<':>10s} "
            f"{dist.get('p50', 0):7.1f} {dist.get('p90', 0):7.1f} {dist.get('p95', 0):7.1f}"
        )
    lines.append("")

    # --- Section 5: WARN/FAIL dates ---
    lines.append("5. WARN/FAIL EVENT LOG")
    lines.append("-" * W)
    events = [r for r in per_date_records if r["status"] != "OK"]
    if events:
        for r in events:
            reason_str = ", ".join(r["reasons"]) if r["reasons"] else ""
            lines.append(f"  {r['date']:12s} {r['status']:5s}  [{reason_str}]")
    else:
        lines.append("  (none — all dates passed)")
    lines.append("")

    # --- Section 6: Recommendations ---
    lines.append("6. THRESHOLD RECOMMENDATIONS")
    lines.append("-" * W)

    # Turnover
    turnover_d = distributions.get("turnover_pct", {})
    p90_turn = turnover_d.get("p90", 0)
    p95_turn = turnover_d.get("p95", 0)
    warn_rate_turn = sum(
        1 for r in per_date_records
        if r["name_turnover_pct"] is not None and r["name_turnover_pct"] > ht.warn_turnover_pct
    )
    lines.append(f"  Turnover WARN ({ht.warn_turnover_pct}%):")
    lines.append(f"    Fires {warn_rate_turn}/{n_deltas} delta dates ({round(warn_rate_turn/n_deltas*100, 1) if n_deltas else 0}%)")
    lines.append(f"    P90={p90_turn}%, P95={p95_turn}%")
    if p95_turn > 0 and ht.warn_turnover_pct < p90_turn:
        lines.append(f"    -> Consider raising to ~{round(p90_turn + 5, -1)}% to reduce noise (fires below P90)")
    elif warn_rate_turn == 0:
        lines.append(f"    -> Never fires; threshold may be too loose")
    else:
        lines.append(f"    -> Current threshold seems well-calibrated")
    lines.append("")

    # Weight L1
    weight_d = distributions.get("weight_l1_pct", {})
    p90_wt = weight_d.get("p90", 0)
    warn_rate_wt = sum(
        1 for r in per_date_records
        if r["weight_l1_delta"] is not None and r["weight_l1_delta"] > ht.warn_weight_l1_pct
    )
    lines.append(f"  Weight L1 WARN ({ht.warn_weight_l1_pct}%):")
    lines.append(f"    Fires {warn_rate_wt}/{n_deltas} delta dates ({round(warn_rate_wt/n_deltas*100, 1) if n_deltas else 0}%)")
    lines.append(f"    P90={p90_wt}%")
    if warn_rate_wt == 0:
        lines.append(f"    -> Never fires; may be too loose or data doesn't exhibit high churn")
    elif warn_rate_wt / n_deltas > 0.3:
        lines.append(f"    -> Fires >30% of dates; consider raising threshold to reduce alert fatigue")
    else:
        lines.append(f"    -> Current threshold seems well-calibrated")
    lines.append("")

    # Catalyst coverage
    cat_d = distributions.get("catalyst_coverage_pct", {})
    p5_cat = cat_d.get("p5", 0)
    fail_rate_cat = sum(
        1 for r in per_date_records
        if r["catalyst_coverage_pct"] is not None and r["catalyst_coverage_pct"] < ht.fail_catalyst_coverage_min
    )
    lines.append(f"  Catalyst FAIL (<{ht.fail_catalyst_coverage_min}%):")
    lines.append(f"    Fires {fail_rate_cat}/{n_total} dates ({round(fail_rate_cat/n_total*100, 1) if n_total else 0}%)")
    lines.append(f"    P5={p5_cat}%, Min={cat_d.get('min', 0)}%")
    if fail_rate_cat > 0:
        lines.append(f"    -> FAIL fires historically — check if those dates had genuine data issues")
    else:
        lines.append(f"    -> Never fires; FAIL threshold protects against broken data feed")
    lines.append("")

    # Catalyst drop
    drop_d = distributions.get("catalyst_drop_pp", {})
    p90_drop = drop_d.get("p90", 0)
    p95_drop = drop_d.get("p95", 0)
    warn_rate_drop = sum(
        1 for r in per_date_records
        if r["catalyst_drop_pp"] is not None and r["catalyst_drop_pp"] > ht.warn_catalyst_drop_pp
    )
    lines.append(f"  Catalyst Drop WARN (>{ht.warn_catalyst_drop_pp}pp):")
    lines.append(f"    Fires {warn_rate_drop}/{n_deltas} delta dates ({round(warn_rate_drop/n_deltas*100, 1) if n_deltas else 0}%)")
    lines.append(f"    P90={p90_drop}pp, P95={p95_drop}pp")
    if warn_rate_drop == 0:
        lines.append(f"    -> Never fires; catalyst coverage is stable across snapshots")
    elif warn_rate_drop / n_deltas > 0.2:
        lines.append(f"    -> Fires frequently; consider raising threshold")
    else:
        lines.append(f"    -> Current threshold seems well-calibrated")
    lines.append("")

    # A-tier count
    a_d = distributions.get("a_count", {})
    p5_a = a_d.get("p5", 0)
    warn_rate_a = sum(
        1 for r in per_date_records
        if r["a_count"] is not None
        and r["a_count"] >= ht.fail_a_count_min
        and r["a_count"] < ht.warn_a_count_low
    )
    fail_rate_a = sum(
        1 for r in per_date_records
        if r["a_count"] is not None and r["a_count"] < ht.fail_a_count_min
    )
    lines.append(f"  A-count FAIL (<{ht.fail_a_count_min}) / WARN (<{ht.warn_a_count_low}):")
    lines.append(f"    FAIL fires {fail_rate_a}/{n_total} dates")
    lines.append(f"    WARN fires {warn_rate_a}/{n_total} dates")
    lines.append(f"    P5={p5_a}, Min={a_d.get('min', 0)}")
    if fail_rate_a > 0:
        lines.append(f"    -> A-count FAIL fires historically; review those dates for data issues")
    lines.append("")

    # Overall assessment
    lines.append("=" * W)
    total_warn = status_counts.get("WARN", 0)
    total_fail = status_counts.get("FAIL", 0)
    warn_pct = round(total_warn / n_total * 100, 1) if n_total else 0
    fail_pct = round(total_fail / n_total * 100, 1) if n_total else 0
    lines.append(f"OVERALL: WARN={warn_pct}% ({total_warn}/{n_total}), "
                 f"FAIL={fail_pct}% ({total_fail}/{n_total}) of historical dates")

    if warn_pct > 30:
        lines.append("VERDICT: WARN rate >30% — thresholds are too tight, will cause alert fatigue")
    elif warn_pct > 15:
        lines.append("VERDICT: WARN rate 15-30% — consider tightening if these are genuine signals")
    elif total_warn + total_fail == 0:
        lines.append("VERDICT: Zero alerts — thresholds may be too loose, all historical dates pass")
    else:
        lines.append("VERDICT: Alert rate looks reasonable — WARN is rare and meaningful")
    lines.append("=" * W)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase-2 Health Gate Threshold Calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=ARCHIVE_DIR,
        help=f"Archive directory (default: {ARCHIVE_DIR})",
    )
    parser.add_argument(
        "--ruleset",
        type=Path,
        default=PHASE2_RULESET_PATH,
        help=f"DecisionRuleset JSON (default: {PHASE2_RULESET_PATH.name})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--health-thresholds",
        type=Path,
        default=None,
        help="Phase2HealthThresholds JSON override (default: pinned production or built-in)",
    )
    args = parser.parse_args()

    if not args.ruleset.exists():
        print(f"ERROR: Ruleset not found: {args.ruleset}", file=sys.stderr)
        sys.exit(1)

    ruleset = DecisionRuleset.from_json(str(args.ruleset))
    print(f"Loaded ruleset: {ruleset.ruleset_id}")

    # Load health thresholds
    if args.health_thresholds:
        if not args.health_thresholds.exists():
            print(f"ERROR: Health thresholds not found: {args.health_thresholds}", file=sys.stderr)
            sys.exit(1)
        health_thresholds = Phase2HealthThresholds.from_json(str(args.health_thresholds))
    elif PHASE2_HEALTH_THRESHOLDS_PATH.exists():
        health_thresholds = Phase2HealthThresholds.from_json(str(PHASE2_HEALTH_THRESHOLDS_PATH))
    else:
        health_thresholds = DEFAULT_HEALTH_THRESHOLDS
    print(f"Health thresholds: {health_thresholds.thresholds_id}")

    run_calibration(
        archive_dir=args.archive_dir,
        ruleset=ruleset,
        output_dir=args.output_dir,
        health_thresholds=health_thresholds,
        start_date=args.start,
        end_date=args.end,
    )


if __name__ == "__main__":
    main()
