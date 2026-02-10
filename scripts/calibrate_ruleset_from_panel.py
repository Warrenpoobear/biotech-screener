#!/usr/bin/env python3
"""
Ruleset Calibration from Walk-Forward Panel

Reads the walkforward_panel.csv, sweeps tier_a_optionality_floor across
candidate values, re-tiers each row from the panel's raw inputs, and produces:

  - artifacts/calibration_report.md     Human-readable summary
  - artifacts/calibration_report.json   Structured metrics for all candidates
  - artifacts/candidate_overrides.json  Best candidate override (bump_ruleset input)

Usage:
    python scripts/calibrate_ruleset_from_panel.py [options]

    --panel PATH         Panel CSV (default: artifacts/walkforward_panel.csv)
    --output-dir PATH    Output directory (default: artifacts/)
    --a-floor-values     Comma-separated tier_a_optionality_floor values to test
    --b-floor FLOAT      Held constant (default: 0.30)
    --catalyst-near INT  Held constant (default: 90)
    --min-a-pct FLOAT    Constraint: min % of rows in tier A (default: 3.0)
    --max-turnover FLOAT Constraint: max mean turnover % (default: 50.0)
    --sweep-catalyst-near-days LIST  Enable 2D sweep: comma-separated catalyst_near_days values
    --date-min DATE      Filter panel rows: as_of_date >= DATE (YYYY-MM-DD)
    --date-max DATE      Filter panel rows: as_of_date <= DATE (YYYY-MM-DD)
    --max-missing-catalyst-pct FLOAT  Data-quality gate: max catalyst_mode=missing % (default: 80)
    --suffix TAG         Suffix for output files (e.g., '2025' → calibration_report_2025.md)
    --dry-run            Print sweep config, exit
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from outcome_provenance import build_provenance
from decision_engine_codes import registry_fingerprint

DEFAULT_PANEL = PROJECT_ROOT / "artifacts" / "walkforward_panel.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts"

DEFAULT_A_FLOOR_VALUES = [0.40, 0.45, 0.50, 0.52, 0.55, 0.58, 0.60, 0.65]
DEFAULT_B_FLOOR = 0.30
DEFAULT_CATALYST_NEAR_DAYS = 90
DEFAULT_MAX_MISSING_CATALYST_PCT = 80.0
DEFAULT_CATALYST_NEAR_SWEEP = [30, 45, 60, 90, 120]


# =============================================================================
# PANEL READER
# =============================================================================

def read_panel(path: Path) -> List[Dict[str, Any]]:
    """Read the panel CSV and parse numeric columns."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        for raw in csv.DictReader(f):
            row: Dict[str, Any] = dict(raw)
            # Parse numeric fields
            for col in ("optionality", "catalyst_days_raw",
                        "fwd_ret_20d", "fwd_ret_60d",
                        "fwd_max_dd_20d", "fwd_max_dd_60d", "weight"):
                v = row.get(col, "")
                if v == "" or v is None:
                    row[col] = None
                else:
                    try:
                        row[col] = float(v)
                    except (ValueError, TypeError):
                        row[col] = None
            rows.append(row)
    return rows


# =============================================================================
# DATA-QUALITY GATE
# =============================================================================

def check_panel_data_quality(
    rows: List[Dict[str, Any]],
    max_missing_catalyst_pct: float,
) -> Tuple[bool, Dict[str, Any]]:
    """Pre-check panel data quality for calibration fitness.

    Returns (ok, diagnostics). When ok=False, calibration should not proceed
    because the panel lacks sufficient catalyst coverage to produce meaningful
    tier separation signals.
    """
    total = len(rows)
    if total == 0:
        return False, {"reason": "empty_panel", "total_rows": 0}

    # Eligible rows only (D-tier is ineligible, catalyst irrelevant for them)
    eligible = [r for r in rows if str(r.get("eligible", "0")) == "1"]
    n_eligible = len(eligible)
    if n_eligible == 0:
        return False, {
            "reason": "no_eligible_rows",
            "total_rows": total,
            "eligible_rows": 0,
        }

    # Catalyst coverage among eligible rows
    n_missing = sum(1 for r in eligible if r.get("catalyst_mode", "") in ("missing", ""))
    missing_pct = round(n_missing / n_eligible * 100, 1)

    diagnostics = {
        "total_rows": total,
        "eligible_rows": n_eligible,
        "eligible_with_catalyst_missing": n_missing,
        "catalyst_missing_pct": missing_pct,
        "threshold_pct": max_missing_catalyst_pct,
    }

    if missing_pct > max_missing_catalyst_pct:
        diagnostics["reason"] = "insufficient_catalyst_coverage"
        return False, diagnostics

    diagnostics["reason"] = "ok"
    return True, diagnostics


# =============================================================================
# RE-TIER LOGIC
# =============================================================================

def retier_row(
    row: Dict[str, Any],
    a_floor: float,
    b_floor: float,
    catalyst_near_days: int,
    catalyst_mid_days: int = 180,
) -> str:
    """Re-compute tier for a panel row under a candidate a_floor.

    Replicates _compute_tier_dev() from decision_engine.py using
    panel columns: eligible, optionality, catalyst_mode, catalyst_days_raw.

    catalyst_mid_days: upper bound of MID strength band (default 180).
    A-tier requires actionable catalyst (near or mid). FAR treated same
    as MISSING for tier gating.
    """
    if str(row.get("eligible", "0")) != "1":
        return "D"

    opt = row.get("optionality")
    if opt is None:
        return "C"

    catalyst_mode = row.get("catalyst_mode", "")
    cat_days = row.get("catalyst_days_raw")

    has_specific_days = catalyst_mode == "specific_days"
    has_blended_window = catalyst_mode == "blended_window"
    has_catalyst_data = has_specific_days or has_blended_window

    # Compute catalyst strength
    if has_blended_window:
        strength = "near"
    elif has_specific_days and cat_days is not None and cat_days > 0:
        if cat_days <= catalyst_near_days:
            strength = "near"
        elif cat_days <= catalyst_mid_days:
            strength = "mid"
        else:
            strength = "far"
    else:
        strength = "missing"

    is_actionable = strength in ("near", "mid")

    if has_catalyst_data:
        if opt >= a_floor and is_actionable:
            return "A"
        elif opt >= a_floor:
            return "B"
        elif opt >= b_floor and is_actionable:
            return "B"
        else:
            return "C"
    else:
        if opt >= a_floor:
            return "B"
        elif opt >= b_floor:
            return "C"
        else:
            return "C"


# =============================================================================
# CANDIDATE EVALUATION
# =============================================================================

def evaluate_candidate(
    rows: List[Dict[str, Any]],
    a_floor: float,
    b_floor: float,
    catalyst_near_days: int,
    catalyst_mid_days: int = 180,
) -> Dict[str, Any]:
    """Evaluate a candidate a_floor value by re-tiering all rows."""
    # Re-tier
    tiers: List[str] = []
    for row in rows:
        new_tier = retier_row(row, a_floor, b_floor, catalyst_near_days, catalyst_mid_days)
        tiers.append(new_tier)

    total = len(rows)

    # Tier distribution (all rows)
    tier_counts: Dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0}
    for t in tiers:
        tier_counts[t] = tier_counts.get(t, 0) + 1
    tier_pcts = {t: round(c / total * 100, 1) if total else 0 for t, c in tier_counts.items()}

    # Tier distribution among eligible rows only (D excluded — always ineligible)
    n_eligible = sum(1 for i, row in enumerate(rows) if str(row.get("eligible", "0")) == "1")
    eligible_tier_pcts = {
        t: round(tier_counts[t] / n_eligible * 100, 1) if n_eligible else 0
        for t in ("A", "B", "C")
    }

    # Per-tier return and drawdown metrics
    tier_metrics: Dict[str, Dict[str, Any]] = {}
    for tier in ["A", "B", "C", "D"]:
        rets_60d: List[float] = []
        dds_60d: List[float] = []
        for i, row in enumerate(rows):
            if tiers[i] != tier:
                continue
            ret = row.get("fwd_ret_60d")
            if ret is not None:
                rets_60d.append(ret)
            dd = row.get("fwd_max_dd_60d")
            if dd is not None:
                dds_60d.append(dd)

        entry: Dict[str, Any] = {"count": tier_counts[tier], "pct": tier_pcts[tier]}
        if rets_60d:
            entry["mean_ret_60d"] = round(mean(rets_60d), 2)
            entry["median_ret_60d"] = round(median(rets_60d), 2)
            entry["hit_rate_60d"] = round(sum(1 for r in rets_60d if r > 0) / len(rets_60d) * 100, 1)
        else:
            entry["mean_ret_60d"] = None
            entry["median_ret_60d"] = None
            entry["hit_rate_60d"] = None

        if dds_60d:
            entry["mean_max_dd_60d"] = round(mean(dds_60d), 2)
            entry["median_max_dd_60d"] = round(median(dds_60d), 2)
        else:
            entry["mean_max_dd_60d"] = None
            entry["median_max_dd_60d"] = None

        tier_metrics[tier] = entry

    # AB vs C separation (the objective)
    # D-tier (ineligible) is excluded — it's a categorically different
    # population that would confound the eligible-tier signal.
    ab_rets: List[float] = []
    cd_rets: List[float] = []
    ab_dds: List[float] = []
    cd_dds: List[float] = []
    for i, row in enumerate(rows):
        ret = row.get("fwd_ret_60d")
        dd = row.get("fwd_max_dd_60d")
        if tiers[i] in ("A", "B"):
            if ret is not None:
                ab_rets.append(ret)
            if dd is not None:
                ab_dds.append(dd)
        elif tiers[i] == "C":
            if ret is not None:
                cd_rets.append(ret)
            if dd is not None:
                cd_dds.append(dd)

    ab_median = median(ab_rets) if ab_rets else None
    cd_median = median(cd_rets) if cd_rets else None
    separation = (ab_median - cd_median) if ab_median is not None and cd_median is not None else None

    ab_dd = median(ab_dds) if ab_dds else None
    cd_dd = median(cd_dds) if cd_dds else None

    # Stability: top-25 overlap (Jaccard) between consecutive dates
    # under the new tiering (for tickers that would be in portfolio: eligible A/B)
    by_date: Dict[str, List[str]] = {}
    for i, row in enumerate(rows):
        if str(row.get("eligible", "0")) == "1" and tiers[i] in ("A", "B"):
            d = row.get("as_of_date", "")
            by_date.setdefault(d, []).append(row.get("ticker", ""))

    sorted_dates = sorted(by_date.keys())
    overlaps: List[float] = []
    for j in range(1, len(sorted_dates)):
        prev_set = set(by_date[sorted_dates[j - 1]][:25])
        curr_set = set(by_date[sorted_dates[j]][:25])
        union = prev_set | curr_set
        if union:
            overlaps.append(len(prev_set & curr_set) / len(union))

    mean_overlap = round(mean(overlaps) * 100, 1) if overlaps else None
    mean_turnover = round((1.0 - mean(overlaps)) * 100, 1) if overlaps else None

    # Mean A-count per date
    a_counts = [
        sum(1 for j, row in enumerate(rows) if row.get("as_of_date") == d and tiers[j] == "A")
        for d in sorted_dates
    ] if sorted_dates else []
    mean_a_count = round(mean(a_counts), 1) if a_counts else 0

    return {
        "a_floor": a_floor,
        "tier_distribution": tier_pcts,
        "eligible_tier_distribution": eligible_tier_pcts,
        "tier_metrics": tier_metrics,
        "ab_median_ret_60d": ab_median,
        "cd_median_ret_60d": cd_median,
        "separation_60d": round(separation, 2) if separation is not None else None,
        "ab_median_dd_60d": ab_dd,
        "cd_median_dd_60d": cd_dd,
        "mean_a_count_per_date": mean_a_count,
        "mean_top25_overlap_pct": mean_overlap,
        "mean_turnover_pct": mean_turnover,
    }


def score_candidate(
    candidate: Dict[str, Any],
    min_a_pct: float,
    max_turnover: float,
) -> Tuple[float, List[str]]:
    """Score a candidate and check constraints.

    Objective: maximize separation_60d, penalize AB drawdown.
    Returns (score, list_of_constraint_violations).
    """
    violations: List[str] = []

    # Constraint: minimum A-tier % (among eligible rows — D is always ineligible)
    elig_dist = candidate.get("eligible_tier_distribution", {})
    a_pct = elig_dist.get("A", candidate["tier_distribution"].get("A", 0))
    if a_pct < min_a_pct:
        violations.append(f"A-tier {a_pct}% of eligible < min {min_a_pct}%")

    # Constraint: max turnover
    turnover = candidate.get("mean_turnover_pct")
    if turnover is not None and turnover > max_turnover:
        violations.append(f"turnover {turnover}% > max {max_turnover}%")

    # Constraint: separation must be positive
    sep = candidate.get("separation_60d")
    if sep is None or sep <= 0:
        violations.append(f"separation {sep} <= 0")

    # Score = separation - drawdown penalty
    # Higher is better. Drawdown is negative, so we add it (making it a penalty).
    ab_dd = candidate.get("ab_median_dd_60d")
    dd_penalty = abs(ab_dd) * 0.3 if ab_dd is not None else 0  # 30% weight on DD
    score = (sep if sep is not None else -999) - dd_penalty

    return round(score, 4), violations


# =============================================================================
# 2D SWEEP HELPERS
# =============================================================================

def _compute_tradeoffs(
    best: Dict[str, Any],
    baseline: Dict[str, Any],
) -> List[str]:
    """Compute tradeoffs of best candidate vs baseline."""
    tradeoffs: List[str] = []
    b_dd = baseline.get("ab_median_dd_60d")
    c_dd = best.get("ab_median_dd_60d")
    if b_dd is not None and c_dd is not None and c_dd < b_dd:
        tradeoffs.append(
            f"Worse AB max-DD: {c_dd:.2f}% vs baseline {b_dd:.2f}% "
            f"(delta {c_dd - b_dd:+.2f}pp)"
        )
    b_overlap = baseline.get("mean_top25_overlap_pct")
    c_overlap = best.get("mean_top25_overlap_pct")
    if b_overlap is not None and c_overlap is not None and c_overlap < b_overlap - 2:
        tradeoffs.append(
            f"Lower stability: overlap {c_overlap:.1f}% vs baseline {b_overlap:.1f}%"
        )
    b_a_count = baseline.get("mean_a_count_per_date", 0)
    c_a_count = best.get("mean_a_count_per_date", 0)
    if c_a_count > b_a_count * 2:
        tradeoffs.append(
            f"A-tier inflation: {c_a_count:.1f} vs baseline {b_a_count:.1f} "
            f"(may reduce selectivity)"
        )
    return tradeoffs


def run_2d_sweep(
    rows: List[Dict[str, Any]],
    a_floor_values: List[float],
    sweep_values: List[int],
    b_floor: float,
    min_a_pct: float,
    max_turnover: float,
    sweep_dim: str = "cat_near",
    base_near_days: int = 90,
    base_mid_days: int = 180,
) -> List[Dict[str, Any]]:
    """Evaluate all (a_floor, sweep_dim) combinations.

    sweep_dim="cat_near": varies catalyst_near_days, holds catalyst_mid_days fixed.
    sweep_dim="cat_mid":  varies catalyst_mid_days, holds catalyst_near_days fixed.

    Returns a list of candidate dicts, each with a_floor, sweep_val,
    score, violations, passed, and all metrics from evaluate_candidate().
    """
    candidates: List[Dict[str, Any]] = []
    for sv in sweep_values:
        if sweep_dim == "cat_mid":
            cat_near, cat_mid = base_near_days, sv
        else:
            cat_near, cat_mid = sv, base_mid_days
        for a_floor in a_floor_values:
            result = evaluate_candidate(rows, a_floor, b_floor, cat_near, cat_mid)
            result["sweep_val"] = sv
            # Keep legacy key for backward compatibility
            result["catalyst_near_days"] = cat_near
            result["catalyst_mid_days"] = cat_mid
            score, violations = score_candidate(result, min_a_pct, max_turnover)
            result["score"] = score
            result["violations"] = violations
            result["passed"] = len(violations) == 0
            candidates.append(result)
    return candidates


def compute_ridge_summary(
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """For each sweep_val, find the best a_floor.

    Returns a list of dicts with: sweep_val, best_a_floor, score,
    separation_60d, passed, a_pct_elig.
    """
    by_sv: Dict[int, List[Dict[str, Any]]] = {}
    for c in candidates:
        sv = c["sweep_val"]
        by_sv.setdefault(sv, []).append(c)

    ridge: List[Dict[str, Any]] = []
    for sv in sorted(by_sv.keys()):
        group = by_sv[sv]
        passing = [c for c in group if c["passed"]]
        if passing:
            best = max(passing, key=lambda c: c["score"])
        else:
            best = max(group, key=lambda c: c["score"])

        elig_dist = best.get("eligible_tier_distribution", {})
        ridge.append({
            "sweep_val": sv,
            "best_a_floor": best["a_floor"],
            "score": best["score"],
            "separation_60d": best.get("separation_60d"),
            "passed": best.get("passed", False),
            "a_pct_elig": elig_dist.get("A", 0),
            "ab_median_dd_60d": best.get("ab_median_dd_60d"),
            "mean_a_count": best.get("mean_a_count_per_date", 0),
        })
    return ridge


def check_neighbor_stability(
    best: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    a_floor_values: List[float],
    sweep_values: List[int],
) -> List[Dict[str, Any]]:
    """Check all immediate neighbors (±1 step in each dimension) of the best candidate.

    Returns a list of neighbor dicts with their scores and pass/fail status.
    """
    best_af = best["a_floor"]
    best_sv = best["sweep_val"]

    # Find ±1 step indices
    af_idx = a_floor_values.index(best_af) if best_af in a_floor_values else -1
    sv_idx = sweep_values.index(best_sv) if best_sv in sweep_values else -1

    neighbor_afs = set()
    if af_idx > 0:
        neighbor_afs.add(a_floor_values[af_idx - 1])
    neighbor_afs.add(best_af)
    if af_idx < len(a_floor_values) - 1:
        neighbor_afs.add(a_floor_values[af_idx + 1])

    neighbor_svs = set()
    if sv_idx > 0:
        neighbor_svs.add(sweep_values[sv_idx - 1])
    neighbor_svs.add(best_sv)
    if sv_idx < len(sweep_values) - 1:
        neighbor_svs.add(sweep_values[sv_idx + 1])

    neighbors: List[Dict[str, Any]] = []
    for c in candidates:
        if c["a_floor"] in neighbor_afs and c["sweep_val"] in neighbor_svs:
            if c["a_floor"] == best_af and c["sweep_val"] == best_sv:
                continue  # skip self
            neighbors.append({
                "a_floor": c["a_floor"],
                "sweep_val": c["sweep_val"],
                "score": c["score"],
                "separation_60d": c.get("separation_60d"),
                "passed": c.get("passed", False),
            })
    return neighbors


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_calibration_report_md(
    candidates: List[Dict[str, Any]],
    best: Dict[str, Any],
    baseline: Dict[str, Any],
    config: Dict[str, Any],
) -> str:
    """Generate the markdown calibration report."""
    lines: List[str] = []
    lines.append("# Ruleset Calibration Report")
    lines.append("")
    lines.append(f"Generated: {config.get('generated', '')}")
    lines.append(
        f"Panel: {config.get('panel_path', '')} | "
        f"Rows: {config.get('panel_rows', 0)} | "
        f"Baseline a_floor: {config.get('baseline_a_floor', '')}"
    )
    lines.append("")

    # Sweep results table
    lines.append("## Sweep Results")
    lines.append("")
    lines.append(
        "| a_floor | A%(elig) | B%(elig) | C%(elig) | "
        "AB med 60d | CD med 60d | Sep | AB DD | Overlap% | Score | Pass |"
    )
    lines.append(
        "|---------|----------|----------|----------|"
        "------------|------------|-----|-------|----------|-------|------|"
    )
    for c in candidates:
        td = c.get("eligible_tier_distribution", c["tier_distribution"])
        sep = c.get("separation_60d")
        ab_dd = c.get("ab_median_dd_60d")
        ab_ret = c.get("ab_median_ret_60d")
        cd_ret = c.get("cd_median_ret_60d")
        overlap = c.get("mean_top25_overlap_pct")
        score = c.get("score")
        passed = c.get("passed", False)

        marker = "**" if c["a_floor"] == best["a_floor"] else ""

        def _fmt(v, fmt="+.2f"):
            return f"{v:{fmt}}" if v is not None else "n/a"

        line = (
            f"| {marker}{c['a_floor']:.2f}{marker} "
            f"| {td.get('A', 0):.1f} | {td.get('B', 0):.1f} "
            f"| {td.get('C', 0):.1f} "
            f"| {_fmt(ab_ret)} | {_fmt(cd_ret)} | {_fmt(sep)} "
            f"| {_fmt(ab_dd, '.2f')} | {_fmt(overlap, '.1f')} "
            f"| {_fmt(score, '.2f')} | {'Y' if passed else 'N'} |"
        )
        lines.append(line)
    lines.append("")

    # Best candidate
    lines.append("## Recommended Candidate")
    lines.append("")
    if best.get("passed"):
        lines.append(f"**`tier_a_optionality_floor = {best['a_floor']:.2f}`**")
        lines.append("")
        lines.append("### Baseline vs Candidate")
        lines.append("")
        lines.append("| Metric | Baseline | Candidate | Delta |")
        lines.append("|--------|----------|-----------|-------|")

        metrics = [
            ("AB median 60d ret %", "ab_median_ret_60d"),
            ("CD median 60d ret %", "cd_median_ret_60d"),
            ("Separation (AB-CD) %", "separation_60d"),
            ("AB median max-DD %", "ab_median_dd_60d"),
            ("Mean A-count/date", "mean_a_count_per_date"),
            ("Top-25 overlap %", "mean_top25_overlap_pct"),
            ("Mean turnover %", "mean_turnover_pct"),
        ]
        for label, key in metrics:
            bv = baseline.get(key)
            cv = best.get(key)
            if bv is not None and cv is not None:
                delta = cv - bv
                lines.append(
                    f"| {label} | {bv:.2f} | {cv:.2f} | {delta:+.2f} |"
                )
            else:
                bv_s = f"{bv:.2f}" if bv is not None else "n/a"
                cv_s = f"{cv:.2f}" if cv is not None else "n/a"
                lines.append(f"| {label} | {bv_s} | {cv_s} | - |")
        lines.append("")

        # Tradeoffs
        tradeoffs = best.get("tradeoffs", [])
        if tradeoffs:
            lines.append("### Tradeoffs")
            lines.append("")
            for t in tradeoffs:
                lines.append(f"- {t}")
            lines.append("")

        lines.append("### Why Chosen")
        lines.append("")
        lines.append(
            f"- Objective score: {best.get('score', 'n/a')} "
            f"(best among {sum(1 for c in candidates if c.get('passed'))} passing candidates)"
        )
        lines.append(f"- All constraints satisfied")
    else:
        lines.append("**No candidate passed all constraints.**")
        lines.append("")
        lines.append("Consider relaxing constraints or expanding the search space.")

    lines.append("")
    return "\n".join(lines) + "\n"


def generate_calibration_report_2d_md(
    candidates: List[Dict[str, Any]],
    best: Dict[str, Any],
    baseline: Dict[str, Any],
    ridge: List[Dict[str, Any]],
    neighbors: List[Dict[str, Any]],
    config: Dict[str, Any],
    sweep_label: str = "cat_near",
) -> str:
    """Generate the markdown calibration report for a 2D sweep."""
    sweep_param = "catalyst_mid_days" if sweep_label == "cat_mid" else "catalyst_near_days"
    lines: List[str] = []
    lines.append("# 2D Ruleset Calibration Report")
    lines.append("")
    lines.append(f"Generated: {config.get('generated', '')}")
    lines.append(
        f"Panel: {config.get('panel_path', '')} | "
        f"Rows: {config.get('panel_rows', 0)} | "
        f"Baseline: a_floor={config.get('baseline_a_floor', '?')}, "
        f"{sweep_label}={config.get('baseline_sweep_val', '?')}"
    )
    lines.append(
        f"Grid: {len(config.get('a_floor_values', []))} a_floor × "
        f"{len(config.get('sweep_values', []))} {sweep_label} = "
        f"{len(candidates)} combos"
    )
    lines.append("")

    # Top 10 candidates table
    sorted_cands = sorted(candidates, key=lambda c: c["score"], reverse=True)
    top_n = sorted_cands[:10]
    lines.append("## Top 10 Candidates")
    lines.append("")
    lines.append(
        f"| # | a_floor | {sweep_label} | A%(elig) | Sep | AB DD | Score | Pass |"
    )
    lines.append(
        "|---|---------|----------|----------|-----|-------|-------|------|"
    )

    def _fmt(v, fmt="+.2f"):
        return f"{v:{fmt}}" if v is not None else "n/a"

    for i, c in enumerate(top_n, 1):
        etd = c.get("eligible_tier_distribution", {})
        marker = " **" if c is best else ""
        marker_end = "**" if c is best else ""
        lines.append(
            f"| {i} | {marker}{c['a_floor']:.2f}{marker_end} "
            f"| {c['sweep_val']} "
            f"| {etd.get('A', 0):.1f} "
            f"| {_fmt(c.get('separation_60d'))} "
            f"| {_fmt(c.get('ab_median_dd_60d'), '.2f')} "
            f"| {_fmt(c.get('score'), '.2f')} "
            f"| {'Y' if c.get('passed') else 'N'} |"
        )
    lines.append("")

    # Ridge summary
    lines.append(f"## Ridge Summary (best a_floor per {sweep_label})")
    lines.append("")
    lines.append(
        f"| {sweep_label} | best a_floor | A%(elig) | A count | Sep | AB DD | Score | Pass |"
    )
    lines.append(
        "|----------|-------------|----------|---------|-----|-------|-------|------|"
    )
    for r in ridge:
        lines.append(
            f"| {r['sweep_val']} "
            f"| {r['best_a_floor']:.2f} "
            f"| {r['a_pct_elig']:.1f} "
            f"| {r['mean_a_count']:.1f} "
            f"| {_fmt(r['separation_60d'])} "
            f"| {_fmt(r['ab_median_dd_60d'], '.2f')} "
            f"| {_fmt(r['score'], '.2f')} "
            f"| {'Y' if r['passed'] else 'N'} |"
        )
    lines.append("")

    # Neighbor stability
    if neighbors:
        lines.append("## Neighbor Stability")
        lines.append("")
        lines.append(
            f"Best candidate: a_floor={best['a_floor']:.2f}, "
            f"{sweep_label}={best['sweep_val']}, "
            f"score={best['score']:.4f}"
        )
        lines.append("")
        n_passing = sum(1 for n in neighbors if n["passed"])
        lines.append(
            f"Neighbors (±1 step): {len(neighbors)} checked, "
            f"{n_passing} passing"
        )
        lines.append("")
        lines.append(f"| a_floor | {sweep_label} | Sep | Score | Pass |")
        lines.append("|---------|----------|-----|-------|------|")
        for n in sorted(neighbors, key=lambda x: x["score"], reverse=True):
            lines.append(
                f"| {n['a_floor']:.2f} "
                f"| {n['sweep_val']} "
                f"| {_fmt(n['separation_60d'])} "
                f"| {_fmt(n['score'], '.2f')} "
                f"| {'Y' if n['passed'] else 'N'} |"
            )
        lines.append("")

        # Stability verdict
        if n_passing == len(neighbors):
            lines.append("**Stability: STRONG** — all neighbors pass constraints.")
        elif n_passing >= len(neighbors) / 2:
            lines.append(f"**Stability: MODERATE** — {n_passing}/{len(neighbors)} neighbors pass.")
        else:
            lines.append(f"**Stability: WEAK** — only {n_passing}/{len(neighbors)} neighbors pass.")
        lines.append("")

    # Recommended candidate
    lines.append("## Recommended Candidate")
    lines.append("")
    if best.get("passed"):
        lines.append(
            f"**`tier_a_optionality_floor = {best['a_floor']:.2f}`, "
            f"`{sweep_param} = {best['sweep_val']}`**"
        )
        lines.append("")
        lines.append("### Baseline vs Candidate")
        lines.append("")
        lines.append("| Metric | Baseline | Candidate | Delta |")
        lines.append("|--------|----------|-----------|-------|")

        metrics = [
            ("AB median 60d ret %", "ab_median_ret_60d"),
            ("CD median 60d ret %", "cd_median_ret_60d"),
            ("Separation (AB-CD) %", "separation_60d"),
            ("AB median max-DD %", "ab_median_dd_60d"),
            ("Mean A-count/date", "mean_a_count_per_date"),
            ("Top-25 overlap %", "mean_top25_overlap_pct"),
            ("Mean turnover %", "mean_turnover_pct"),
        ]
        for label, key in metrics:
            bv = baseline.get(key)
            cv = best.get(key)
            if bv is not None and cv is not None:
                delta = cv - bv
                lines.append(f"| {label} | {bv:.2f} | {cv:.2f} | {delta:+.2f} |")
            else:
                bv_s = f"{bv:.2f}" if bv is not None else "n/a"
                cv_s = f"{cv:.2f}" if cv is not None else "n/a"
                lines.append(f"| {label} | {bv_s} | {cv_s} | - |")
        lines.append("")

        tradeoffs = best.get("tradeoffs", [])
        if tradeoffs:
            lines.append("### Tradeoffs")
            lines.append("")
            for t in tradeoffs:
                lines.append(f"- {t}")
            lines.append("")

        lines.append("### Why Chosen")
        lines.append("")
        n_passing = sum(1 for c in candidates if c.get("passed"))
        lines.append(
            f"- Objective score: {best.get('score', 'n/a')} "
            f"(best among {n_passing} passing candidates out of {len(candidates)} combos)"
        )
        lines.append("- All constraints satisfied")
    else:
        lines.append("**No candidate passed all constraints.**")
        lines.append("")
        lines.append("Consider relaxing constraints or expanding the search space.")

    lines.append("")
    return "\n".join(lines) + "\n"


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ruleset Calibration from Walk-Forward Panel"
    )
    parser.add_argument(
        "--panel", type=str, default=str(DEFAULT_PANEL),
        help="Panel CSV path (default: artifacts/walkforward_panel.csv)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory (default: artifacts/)",
    )
    parser.add_argument(
        "--a-floor-values", type=str, default=None,
        help="Comma-separated a_floor values (default: 0.40,0.45,...,0.65)",
    )
    parser.add_argument(
        "--b-floor", type=float, default=DEFAULT_B_FLOOR,
        help=f"tier_b_optionality_floor held constant (default: {DEFAULT_B_FLOOR})",
    )
    parser.add_argument(
        "--catalyst-near", type=int, default=DEFAULT_CATALYST_NEAR_DAYS,
        help=f"catalyst_near_days held constant for 1D sweep (default: {DEFAULT_CATALYST_NEAR_DAYS})",
    )
    parser.add_argument(
        "--catalyst-mid", type=int, default=180,
        help="catalyst_mid_days held constant when sweeping near (default: 180)",
    )
    parser.add_argument(
        "--sweep-catalyst-near-days", type=str, default=None,
        help="Enable 2D sweep: comma-separated catalyst_near_days values (e.g., '30,45,60,90,120')",
    )
    parser.add_argument(
        "--sweep-catalyst-mid-days", type=str, default=None,
        help="Enable 2D sweep: comma-separated catalyst_mid_days values (e.g., '150,180,210,270,365')",
    )
    parser.add_argument(
        "--min-a-pct", type=float, default=3.0,
        help="Constraint: min %% of rows in tier A (default: 3.0)",
    )
    parser.add_argument(
        "--max-turnover", type=float, default=50.0,
        help="Constraint: max mean turnover %% (default: 50.0)",
    )
    parser.add_argument(
        "--date-min", type=str, default=None,
        help="Filter panel rows: as_of_date >= DATE (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--date-max", type=str, default=None,
        help="Filter panel rows: as_of_date <= DATE (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--max-missing-catalyst-pct", type=float,
        default=DEFAULT_MAX_MISSING_CATALYST_PCT,
        help=f"Data-quality gate: max catalyst_mode=missing %% among eligible rows (default: {DEFAULT_MAX_MISSING_CATALYST_PCT})",
    )
    parser.add_argument(
        "--suffix", type=str, default=None,
        help="Suffix for output files (e.g., '2025' → calibration_report_2025.md)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print sweep config, exit",
    )
    args = parser.parse_args()

    # Parse a_floor sweep values
    if args.a_floor_values:
        a_floor_values = sorted(float(v.strip()) for v in args.a_floor_values.split(","))
    else:
        a_floor_values = DEFAULT_A_FLOOR_VALUES

    # Validate: b_floor must be strictly less than all a_floor values
    for v in a_floor_values:
        if v <= args.b_floor:
            print(f"ERROR: a_floor={v} <= b_floor={args.b_floor}", file=sys.stderr)
            return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "panel_path": args.panel,
        "a_floor_values": a_floor_values,
        "b_floor": args.b_floor,
        "catalyst_near_days": args.catalyst_near,
        "min_a_pct": args.min_a_pct,
        "max_turnover": args.max_turnover,
        "generated": datetime.now().isoformat(timespec="seconds"),
    }

    # Parse 2D catalyst sweep values
    if args.sweep_catalyst_near_days and args.sweep_catalyst_mid_days:
        print("ERROR: Cannot use both --sweep-catalyst-near-days and --sweep-catalyst-mid-days", file=sys.stderr)
        return 1

    sweep_label = "cat_near"
    sweep_values: Optional[List[int]] = None
    if args.sweep_catalyst_near_days:
        sweep_values = sorted(int(v.strip()) for v in args.sweep_catalyst_near_days.split(","))
    elif args.sweep_catalyst_mid_days:
        sweep_values = sorted(int(v.strip()) for v in args.sweep_catalyst_mid_days.split(","))
        sweep_label = "cat_mid"
    is_2d = sweep_values is not None

    print("Ruleset Calibration Configuration:")
    print(f"  Panel:           {args.panel}")
    print(f"  a_floor values:  {a_floor_values}")
    print(f"  b_floor:         {args.b_floor}")
    if is_2d:
        print(f"  {sweep_label}:   {sweep_values} (2D sweep)")
        print(f"  Grid size:       {len(a_floor_values)} × {len(sweep_values)} = {len(a_floor_values) * len(sweep_values)} combos")
    else:
        print(f"  catalyst_near:   {args.catalyst_near}")
    print(f"  Constraints:     A% >= {args.min_a_pct}, turnover <= {args.max_turnover}%")
    if args.date_min or args.date_max:
        print(f"  Date filter:     {args.date_min or '*'} to {args.date_max or '*'}")
    print(f"  Max missing catalyst: {args.max_missing_catalyst_pct}%")

    if args.dry_run:
        print("\n[DRY RUN] Exiting before computation.")
        return 0

    # Read panel
    panel_path = Path(args.panel)
    if not panel_path.exists():
        print(f"ERROR: Panel not found: {panel_path}", file=sys.stderr)
        return 1

    print(f"\nReading panel ...")
    rows = read_panel(panel_path)
    print(f"  {len(rows)} rows loaded")

    # Date filtering
    if args.date_min or args.date_max:
        before = len(rows)
        if args.date_min:
            rows = [r for r in rows if r.get("as_of_date", "") >= args.date_min]
        if args.date_max:
            rows = [r for r in rows if r.get("as_of_date", "") <= args.date_max]
        print(f"  Date filter: {before} → {len(rows)} rows")
        if not rows:
            print("ERROR: No rows remain after date filtering.", file=sys.stderr)
            return 1

    config["panel_rows"] = len(rows)
    config["date_min"] = args.date_min
    config["date_max"] = args.date_max

    # Extract baseline ruleset_id from panel (first row's ruleset_id)
    baseline_ruleset_id = ""
    for r in rows:
        rid = r.get("ruleset_id", "")
        if rid:
            baseline_ruleset_id = rid
            break
    config["baseline_ruleset_id"] = baseline_ruleset_id

    # Data-quality gate
    dq_ok, dq_diag = check_panel_data_quality(rows, args.max_missing_catalyst_pct)
    config["data_quality"] = dq_diag
    if not dq_ok:
        reason = dq_diag.get("reason", "unknown")
        print(f"\n{'=' * 60}")
        print(f"DATA-QUALITY GATE: FAIL")
        print(f"  Reason: {reason}")
        if reason == "insufficient_catalyst_coverage":
            print(f"  catalyst_mode=missing: {dq_diag['catalyst_missing_pct']}% "
                  f"of eligible rows (threshold: {dq_diag['threshold_pct']}%)")
            print(f"  Recommendation: filter to date range with well-formed catalyst data")
            print(f"  Example: --date-min 2025-01-01")
        elif reason == "no_eligible_rows":
            print(f"  No eligible rows in panel ({dq_diag['total_rows']} total)")
        elif reason == "empty_panel":
            print(f"  Panel is empty after filtering")
        print(f"{'=' * 60}")
        return 1
    else:
        print(f"  Data-quality gate: OK (catalyst_missing={dq_diag['catalyst_missing_pct']}%"
              f" <= {dq_diag['threshold_pct']}%)")

    # Identify baseline
    baseline_a_floor = 0.55  # current active
    baseline_cat_near = args.catalyst_near  # current active (90)
    baseline_cat_mid = args.catalyst_mid  # current active (180)
    config["baseline_a_floor"] = baseline_a_floor
    config["baseline_catalyst_near"] = baseline_cat_near
    config["baseline_catalyst_mid"] = baseline_cat_mid
    if is_2d:
        config["sweep_values"] = sweep_values
        config["sweep_label"] = sweep_label
        config["baseline_sweep_val"] = (
            baseline_cat_mid if sweep_label == "cat_mid" else baseline_cat_near
        )

    # File naming
    sfx = f"_{args.suffix}" if args.suffix else ""

    # =====================================================================
    # 2D SWEEP PATH
    # =====================================================================
    if is_2d:
        n_combos = len(a_floor_values) * len(sweep_values)
        print(f"\nRunning 2D sweep ({n_combos} combos, dim={sweep_label}) ...")
        candidates = run_2d_sweep(
            rows, a_floor_values, sweep_values,
            args.b_floor, args.min_a_pct, args.max_turnover,
            sweep_dim=sweep_label,
            base_near_days=args.catalyst_near,
            base_mid_days=args.catalyst_mid,
        )

        # Print grid summary
        n_passing = sum(1 for c in candidates if c["passed"])
        print(f"  {n_passing}/{n_combos} combos pass all constraints")

        # Evaluate baseline
        baseline_result = evaluate_candidate(
            rows, baseline_a_floor, args.b_floor, baseline_cat_near, baseline_cat_mid,
        )
        score, violations = score_candidate(baseline_result, args.min_a_pct, args.max_turnover)
        baseline_result["score"] = score
        baseline_result["violations"] = violations
        baseline_result["passed"] = len(violations) == 0
        baseline_result["catalyst_near_days"] = baseline_cat_near
        baseline_result["catalyst_mid_days"] = baseline_cat_mid
        baseline_result["sweep_val"] = (
            baseline_cat_mid if sweep_label == "cat_mid" else baseline_cat_near
        )

        # Pick best
        passing = [c for c in candidates if c["passed"]]
        if passing:
            best = max(passing, key=lambda c: c["score"])
        else:
            best = max(candidates, key=lambda c: c["score"])

        # Compute tradeoffs
        tradeoffs = _compute_tradeoffs(best, baseline_result)
        best["tradeoffs"] = tradeoffs

        # Ridge summary + neighbor stability
        ridge = compute_ridge_summary(candidates)
        neighbors = check_neighbor_stability(best, candidates, a_floor_values, sweep_values)

        # Print ridge
        print(f"\n  Ridge summary (best a_floor per {sweep_label}):")
        for r in ridge:
            sep_s = f"{r['separation_60d']:+.2f}" if r['separation_60d'] is not None else "n/a"
            print(f"    {sweep_label}={r['sweep_val']:3d}: "
                  f"a_floor={r['best_a_floor']:.2f} "
                  f"sep={sep_s} score={r['score']:.2f} "
                  f"{'PASS' if r['passed'] else 'FAIL'}")

        # Write candidate_overrides.json
        overrides_path = output_dir / f"candidate_overrides{sfx}.json"
        overrides: Dict[str, Any] = {}
        if best.get("passed"):
            if best["a_floor"] != baseline_a_floor:
                overrides["tier_a_optionality_floor"] = best["a_floor"]
            if sweep_label == "cat_mid":
                if best["catalyst_mid_days"] != baseline_cat_mid:
                    overrides["catalyst_mid_days"] = best["catalyst_mid_days"]
            else:
                if best["catalyst_near_days"] != baseline_cat_near:
                    overrides["catalyst_near_days"] = best["catalyst_near_days"]
        if overrides:
            with open(overrides_path, "w", encoding="utf-8") as f:
                json.dump(overrides, f, indent=2)
                f.write("\n")
            print(f"\nWrote candidate overrides: {overrides_path}")
            print(f"  Use: python scripts/bump_ruleset.py --from-json {overrides_path}")
        else:
            if best.get("passed"):
                print(f"\nBaseline is already optimal. No override generated.")
            else:
                print(f"\nWARNING: No candidate passed constraints. Writing best-effort override.")
                overrides = {"tier_a_optionality_floor": best["a_floor"]}
                if sweep_label == "cat_mid":
                    overrides["catalyst_mid_days"] = best["catalyst_mid_days"]
                else:
                    overrides["catalyst_near_days"] = best["catalyst_near_days"]
            with open(overrides_path, "w", encoding="utf-8") as f:
                json.dump(overrides, f, indent=2)
                f.write("\n")

        # Write JSON report
        json_path = output_dir / f"calibration_report{sfx}.json"
        provenance = build_provenance(
            panel_path=Path(args.panel),
            ruleset_id=baseline_ruleset_id,
            explanation_registry_fingerprint=registry_fingerprint(),
        )
        report_json = {
            "config": config,
            "provenance": provenance,
            "baseline": baseline_result,
            "candidates": candidates,
            "ridge": ridge,
            "neighbors": neighbors,
            "best": {
                "a_floor": best["a_floor"],
                "sweep_val": best["sweep_val"],
                "catalyst_near_days": best["catalyst_near_days"],
                "catalyst_mid_days": best["catalyst_mid_days"],
                "score": best["score"],
                "passed": best["passed"],
                "separation_60d": best["separation_60d"],
                "tradeoffs": tradeoffs,
            },
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_json, f, indent=2, default=str)
        print(f"Wrote calibration report JSON: {json_path}")

        # Write MD report
        md_content = generate_calibration_report_2d_md(
            candidates, best, baseline_result, ridge, neighbors, config,
            sweep_label=sweep_label,
        )
        md_path = output_dir / f"calibration_report{sfx}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"Wrote calibration report MD: {md_path}")

        # Print headline
        print("\n" + "=" * 60)
        if best.get("passed"):
            print(f"RECOMMENDATION: a_floor={best['a_floor']:.2f}, {sweep_label}={best['sweep_val']}")
            print(f"  Separation (AB-CD): {best['separation_60d']:+.2f}pp")
            print(f"  Score: {best['score']:.2f}")
            n_neighbor_pass = sum(1 for n in neighbors if n["passed"])
            print(f"  Neighbor stability: {n_neighbor_pass}/{len(neighbors)} pass")
            if tradeoffs:
                for t in tradeoffs:
                    print(f"  - {t}")
        else:
            print("NO CANDIDATE PASSED ALL CONSTRAINTS")
            print(f"  Best: a_floor={best['a_floor']:.2f}, {sweep_label}={best['sweep_val']}")
            print(f"  Score: {best['score']:.2f}")
            print(f"  Violations: {'; '.join(best['violations'])}")
        print("=" * 60)
        return 0

    # =====================================================================
    # 1D SWEEP PATH (existing)
    # =====================================================================
    config["catalyst_near_days"] = args.catalyst_near

    # Evaluate all candidates
    print(f"\nEvaluating {len(a_floor_values)} candidates ...")
    candidates: List[Dict[str, Any]] = []
    baseline_result: Optional[Dict[str, Any]] = None

    for a_floor in a_floor_values:
        result = evaluate_candidate(rows, a_floor, args.b_floor, args.catalyst_near)
        score, violations = score_candidate(result, args.min_a_pct, args.max_turnover)
        result["score"] = score
        result["violations"] = violations
        result["passed"] = len(violations) == 0

        candidates.append(result)

        if a_floor == baseline_a_floor:
            baseline_result = result

        sep_s = f"{result['separation_60d']:+.2f}" if result["separation_60d"] is not None else "n/a"
        etd = result["eligible_tier_distribution"]
        status = "PASS" if result["passed"] else f"FAIL({'; '.join(violations)})"
        print(f"  a_floor={a_floor:.2f}: A={etd['A']:.1f}%(elig) sep={sep_s} score={score:.2f} {status}")

    # If baseline wasn't in the sweep, evaluate it separately
    if baseline_result is None:
        baseline_result = evaluate_candidate(rows, baseline_a_floor, args.b_floor, args.catalyst_near)
        score, violations = score_candidate(baseline_result, args.min_a_pct, args.max_turnover)
        baseline_result["score"] = score
        baseline_result["violations"] = violations
        baseline_result["passed"] = len(violations) == 0

    # Pick best passing candidate
    passing = [c for c in candidates if c["passed"]]
    if passing:
        best = max(passing, key=lambda c: c["score"])
    else:
        best = max(candidates, key=lambda c: c["score"])

    # Compute tradeoffs vs baseline
    tradeoffs = _compute_tradeoffs(best, baseline_result)
    best["tradeoffs"] = tradeoffs

    # Write candidate_overrides.json
    overrides_path = output_dir / f"candidate_overrides{sfx}.json"
    if best.get("passed") and best["a_floor"] != baseline_a_floor:
        overrides = {"tier_a_optionality_floor": best["a_floor"]}
        with open(overrides_path, "w", encoding="utf-8") as f:
            json.dump(overrides, f, indent=2)
            f.write("\n")
        print(f"\nWrote candidate overrides: {overrides_path}")
        print(f"  Use: python scripts/bump_ruleset.py --from-json {overrides_path}")
    elif best["a_floor"] == baseline_a_floor:
        print(f"\nBaseline (a_floor={baseline_a_floor}) is already optimal. No override generated.")
        overrides = {}
        with open(overrides_path, "w", encoding="utf-8") as f:
            json.dump(overrides, f, indent=2)
            f.write("\n")
    else:
        print(f"\nWARNING: No candidate passed constraints. Writing best-effort override.")
        overrides = {"tier_a_optionality_floor": best["a_floor"]}
        with open(overrides_path, "w", encoding="utf-8") as f:
            json.dump(overrides, f, indent=2)
            f.write("\n")

    # Write calibration_report.json
    json_path = output_dir / f"calibration_report{sfx}.json"
    provenance = build_provenance(
        panel_path=Path(args.panel),
        ruleset_id=baseline_ruleset_id,
        explanation_registry_fingerprint=registry_fingerprint(),
    )
    report_json = {
        "config": config,
        "provenance": provenance,
        "baseline": baseline_result,
        "candidates": candidates,
        "best": {
            "a_floor": best["a_floor"],
            "score": best["score"],
            "passed": best["passed"],
            "separation_60d": best["separation_60d"],
            "tradeoffs": tradeoffs,
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2, default=str)
    print(f"Wrote calibration report JSON: {json_path}")

    # Write calibration_report.md
    md_content = generate_calibration_report_md(
        candidates, best, baseline_result, config
    )
    md_path = output_dir / f"calibration_report{sfx}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Wrote calibration report MD: {md_path}")

    # Print headline
    print("\n" + "=" * 60)
    if best.get("passed"):
        print(f"RECOMMENDATION: tier_a_optionality_floor = {best['a_floor']:.2f}")
        print(f"  Separation (AB-CD): {best['separation_60d']:+.2f}pp")
        print(f"  Score: {best['score']:.2f}")
        if tradeoffs:
            print(f"  Tradeoffs: {len(tradeoffs)}")
            for t in tradeoffs:
                print(f"    - {t}")
    else:
        print("NO CANDIDATE PASSED ALL CONSTRAINTS")
        print(f"  Best a_floor={best['a_floor']:.2f} score={best['score']:.2f}")
        print(f"  Violations: {'; '.join(best['violations'])}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
