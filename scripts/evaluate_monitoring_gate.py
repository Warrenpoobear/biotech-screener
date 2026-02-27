#!/usr/bin/env python3
"""Evaluate monitoring.json against thresholds and emit CI annotations.

Exit codes:
    0 = PASS or WARN (non-blocking)
    1 = FAIL (blocking)
    2 = usage/config error (bad JSON, missing required fields)

Usage:
    python3 scripts/evaluate_monitoring_gate.py \
        --monitoring data/snapshots/2026-02-26/monitoring.json \
        --thresholds production_data/monitoring_thresholds/v1.json \
        --as-of-date 2026-02-26
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _get_nested(d: Dict, *keys: str, default: Any = None) -> Any:
    """Traverse nested dict safely."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def evaluate(
    monitoring: Dict[str, Any],
    thresholds: Dict[str, Any],
) -> Tuple[str, List[str], List[str]]:
    """Evaluate monitoring against thresholds.

    Returns (verdict, fail_reasons, warn_reasons).
    verdict is "PASS", "WARN", "FAIL", or "DEGRADED_SKIP".
    """
    fail_reasons: List[str] = []
    warn_reasons: List[str] = []

    context = monitoring.get("context", {})
    coverage = monitoring.get("coverage", {})
    stability = monitoring.get("stability", {})

    # --- Degraded skip ---
    degraded = context.get("cache_degraded_run", False)
    degraded_cfg = thresholds.get("degraded_mode", {})

    # --- Coverage: eligible_pct ---
    n_universe = coverage.get("n_universe", 0)
    n_eligible = coverage.get("n_eligible", 0)
    if n_universe > 0:
        eligible_pct = 100 * n_eligible / n_universe
        cov_thresh = thresholds.get("coverage", {})
        fail_floor = cov_thresh.get("eligible_pct_fail")
        warn_floor = cov_thresh.get("eligible_pct_warn")
        if fail_floor is not None and eligible_pct < fail_floor:
            fail_reasons.append(
                f"eligible_pct={eligible_pct:.1f}% < fail={fail_floor}%"
            )
        elif warn_floor is not None and eligible_pct < warn_floor:
            warn_reasons.append(
                f"eligible_pct={eligible_pct:.1f}% < warn={warn_floor}%"
            )

    # --- Data counts ---
    dc_thresh = thresholds.get("data_counts", {})
    for source, field in [("sec_8k", "sec_8k_count"), ("ctgov", "ctgov_count")]:
        count = coverage.get(field)
        if count is None:
            continue
        fail_min_key = f"{source}_events_fail_min" if source == "sec_8k" else f"{source}_records_fail_min"
        warn_min_key = f"{source}_events_warn_min" if source == "sec_8k" else f"{source}_records_warn_min"
        fail_min = dc_thresh.get(fail_min_key)
        warn_min = dc_thresh.get(warn_min_key)
        if fail_min is not None and count < fail_min:
            fail_reasons.append(f"{field}={count} < fail_min={fail_min}")
        elif warn_min is not None and count < warn_min:
            warn_reasons.append(f"{field}={count} < warn_min={warn_min}")

    # --- Stability (skip if degraded or suppressed) ---
    skip_stability = (
        degraded and degraded_cfg.get("skip_stability_gates", True)
    ) or stability.get("suppressed", False)

    if not skip_stability:
        stab_thresh = thresholds.get("stability", {})

        # top-60 overlap
        top60_overlap = _get_nested(stability, "top_60", "overlap_pct")
        if top60_overlap is not None:
            fail_floor = stab_thresh.get("top60_overlap_fail")
            warn_floor = stab_thresh.get("top60_overlap_warn")
            if fail_floor is not None and top60_overlap < fail_floor:
                fail_reasons.append(
                    f"top60_overlap={top60_overlap:.1f}% < fail={fail_floor}%"
                )
            elif warn_floor is not None and top60_overlap < warn_floor:
                warn_reasons.append(
                    f"top60_overlap={top60_overlap:.1f}% < warn={warn_floor}%"
                )

        # Spearman rho
        spearman = _get_nested(stability, "rank_correlation", "spearman")
        if spearman is not None:
            fail_floor = stab_thresh.get("spearman_rho_fail")
            warn_floor = stab_thresh.get("spearman_rho_warn")
            if fail_floor is not None and spearman < fail_floor:
                fail_reasons.append(
                    f"spearman_rho={spearman:.4f} < fail={fail_floor}"
                )
            elif warn_floor is not None and spearman < warn_floor:
                warn_reasons.append(
                    f"spearman_rho={spearman:.4f} < warn={warn_floor}"
                )

        # Name turnover (higher is worse)
        turnover = stability.get("name_turnover_pct")
        if turnover is not None:
            fail_ceil = stab_thresh.get("name_turnover_pct_fail")
            warn_ceil = stab_thresh.get("name_turnover_pct_warn")
            if fail_ceil is not None and turnover > fail_ceil:
                fail_reasons.append(
                    f"name_turnover_pct={turnover:.1f}% > fail={fail_ceil}%"
                )
            elif warn_ceil is not None and turnover > warn_ceil:
                warn_reasons.append(
                    f"name_turnover_pct={turnover:.1f}% > warn={warn_ceil}%"
                )

    # --- Verdict ---
    if degraded and degraded_cfg.get("skip_stability_gates", True):
        if fail_reasons:
            verdict = "FAIL"
        elif warn_reasons:
            verdict = "WARN"
        else:
            verdict = "DEGRADED_SKIP"
    elif fail_reasons:
        verdict = "FAIL"
    elif warn_reasons:
        verdict = "WARN"
    else:
        verdict = "PASS"

    # Sort for determinism
    fail_reasons.sort()
    warn_reasons.sort()

    return verdict, fail_reasons, warn_reasons


def build_summary_table(
    monitoring: Dict[str, Any],
    thresholds: Dict[str, Any],
    verdict: str,
    fail_reasons: List[str],
    warn_reasons: List[str],
    as_of_date: str = "",
) -> str:
    """Build a Markdown summary table for GITHUB_STEP_SUMMARY."""
    context = monitoring.get("context", {})
    coverage = monitoring.get("coverage", {})
    stability = monitoring.get("stability", {})

    lines = ["## Monitoring Gate", ""]

    date_str = as_of_date or context.get("as_of_date", "?")
    degraded = context.get("cache_degraded_run", False)
    icon = {
        "PASS": ":white_check_mark:",
        "WARN": ":warning:",
        "FAIL": ":x:",
        "DEGRADED_SKIP": ":fast_forward:",
    }.get(verdict, "")

    lines.append(f"| Field | Value |")
    lines.append(f"|-------|-------|")
    lines.append(f"| as_of_date | `{date_str}` |")
    lines.append(f"| verdict | {icon} **{verdict}** |")
    lines.append(f"| degraded | `{degraded}` |")
    lines.append("")

    # Metrics table
    lines.append("| Metric | Value | Warn | Fail | Status |")
    lines.append("|--------|-------|------|------|--------|")

    stab_thresh = thresholds.get("stability", {})
    cov_thresh = thresholds.get("coverage", {})
    dc_thresh = thresholds.get("data_counts", {})

    suppressed = stability.get("suppressed", False)

    # Eligible %
    n_uni = coverage.get("n_universe", 0)
    n_elig = coverage.get("n_eligible", 0)
    elig_pct = round(100 * n_elig / n_uni, 1) if n_uni > 0 else 0.0
    elig_status = _metric_status(
        elig_pct,
        cov_thresh.get("eligible_pct_warn"),
        cov_thresh.get("eligible_pct_fail"),
        lower_is_worse=True,
    )
    lines.append(
        f"| eligible_pct | {elig_pct}% | "
        f"{cov_thresh.get('eligible_pct_warn', '-')}% | "
        f"{cov_thresh.get('eligible_pct_fail', '-')}% | "
        f"{elig_status} |"
    )

    # SEC 8-K count
    sec_count = coverage.get("sec_8k_count")
    if sec_count is not None:
        sec_status = _metric_status(
            sec_count,
            dc_thresh.get("sec_8k_events_warn_min"),
            dc_thresh.get("sec_8k_events_fail_min"),
            lower_is_worse=True,
        )
        lines.append(
            f"| sec_8k_count | {sec_count} | "
            f"{dc_thresh.get('sec_8k_events_warn_min', '-')} | "
            f"{dc_thresh.get('sec_8k_events_fail_min', '-')} | "
            f"{sec_status} |"
        )

    # CTGov count
    ctg_count = coverage.get("ctgov_count")
    if ctg_count is not None:
        ctg_status = _metric_status(
            ctg_count,
            dc_thresh.get("ctgov_records_warn_min"),
            dc_thresh.get("ctgov_records_fail_min"),
            lower_is_worse=True,
        )
        lines.append(
            f"| ctgov_count | {ctg_count} | "
            f"{dc_thresh.get('ctgov_records_warn_min', '-')} | "
            f"{dc_thresh.get('ctgov_records_fail_min', '-')} | "
            f"{ctg_status} |"
        )

    # Stability metrics (skip if suppressed/degraded)
    if suppressed:
        reason = stability.get("reason", "unknown")
        lines.append(f"| top60_overlap | *suppressed ({reason})* | - | - | - |")
        lines.append(f"| spearman_rho | *suppressed ({reason})* | - | - | - |")
        lines.append(f"| name_turnover_pct | *suppressed ({reason})* | - | - | - |")
    else:
        top60_overlap = _get_nested(stability, "top_60", "overlap_pct")
        if top60_overlap is not None:
            ov_status = _metric_status(
                top60_overlap,
                stab_thresh.get("top60_overlap_warn"),
                stab_thresh.get("top60_overlap_fail"),
                lower_is_worse=True,
            )
            lines.append(
                f"| top60_overlap | {top60_overlap}% | "
                f"{stab_thresh.get('top60_overlap_warn', '-')}% | "
                f"{stab_thresh.get('top60_overlap_fail', '-')}% | "
                f"{ov_status} |"
            )

        spearman = _get_nested(stability, "rank_correlation", "spearman")
        if spearman is not None:
            sp_status = _metric_status(
                spearman,
                stab_thresh.get("spearman_rho_warn"),
                stab_thresh.get("spearman_rho_fail"),
                lower_is_worse=True,
            )
            lines.append(
                f"| spearman_rho | {spearman} | "
                f"{stab_thresh.get('spearman_rho_warn', '-')} | "
                f"{stab_thresh.get('spearman_rho_fail', '-')} | "
                f"{sp_status} |"
            )

        turnover = stability.get("name_turnover_pct")
        if turnover is not None:
            tn_status = _metric_status(
                turnover,
                stab_thresh.get("name_turnover_pct_warn"),
                stab_thresh.get("name_turnover_pct_fail"),
                lower_is_worse=False,
            )
            lines.append(
                f"| name_turnover_pct | {turnover}% | "
                f"{stab_thresh.get('name_turnover_pct_warn', '-')}% | "
                f"{stab_thresh.get('name_turnover_pct_fail', '-')}% | "
                f"{tn_status} |"
            )

    lines.append("")

    # Reasons
    if fail_reasons:
        lines.append("### Fail reasons")
        for r in fail_reasons:
            lines.append(f"- {r}")
        lines.append("")
    if warn_reasons:
        lines.append("### Warn reasons")
        for r in warn_reasons:
            lines.append(f"- {r}")
        lines.append("")

    return "\n".join(lines)


def _metric_status(
    value: float,
    warn_threshold: Optional[float],
    fail_threshold: Optional[float],
    lower_is_worse: bool = True,
) -> str:
    """Return ok/warn/fail for a single metric."""
    if lower_is_worse:
        if fail_threshold is not None and value < fail_threshold:
            return "FAIL"
        if warn_threshold is not None and value < warn_threshold:
            return "WARN"
    else:
        if fail_threshold is not None and value > fail_threshold:
            return "FAIL"
        if warn_threshold is not None and value > warn_threshold:
            return "WARN"
    return "ok"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate monitoring gate.")
    parser.add_argument("--monitoring", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=Path("production_data/monitoring_thresholds/v1.json"),
    )
    parser.add_argument("--as-of-date", default="")
    parser.add_argument(
        "--emit-annotations",
        action="store_true",
        default=True,
    )
    parser.add_argument("--no-annotations", dest="emit_annotations", action="store_false")
    args = parser.parse_args(argv)

    # Load inputs
    try:
        monitoring = _read_json(args.monitoring)
    except Exception as e:
        print(f"ERROR: cannot read monitoring file: {e}", file=sys.stderr)
        return 2

    try:
        thresholds = _read_json(args.thresholds)
    except Exception as e:
        print(f"ERROR: cannot read thresholds file: {e}", file=sys.stderr)
        return 2

    # Validate schema
    if monitoring.get("schema") != "monitoring.v1":
        print(
            f"ERROR: unexpected monitoring schema: {monitoring.get('schema')}",
            file=sys.stderr,
        )
        return 2

    if thresholds.get("schema") != "monitoring_thresholds.v1":
        print(
            f"ERROR: unexpected thresholds schema: {thresholds.get('schema')}",
            file=sys.stderr,
        )
        return 2

    # Evaluate
    verdict, fail_reasons, warn_reasons = evaluate(monitoring, thresholds)

    date_str = args.as_of_date or monitoring.get("context", {}).get("as_of_date", "")

    # Emit annotations
    if args.emit_annotations:
        for reason in fail_reasons:
            print(f"::error::MONITORING FAIL: {reason}")
        for reason in warn_reasons:
            print(f"::warning::MONITORING WARN: {reason}")
        if verdict == "DEGRADED_SKIP":
            print("::notice::MONITORING: degraded run — stability gates skipped")

    # Summary
    summary = build_summary_table(
        monitoring, thresholds, verdict, fail_reasons, warn_reasons,
        as_of_date=date_str,
    )

    # Write to GITHUB_STEP_SUMMARY if available
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write("\n" + summary + "\n")

    # Console output
    print(f"Monitoring gate: {verdict} ({date_str})")
    if fail_reasons:
        print(f"  FAIL: {'; '.join(fail_reasons)}")
    if warn_reasons:
        print(f"  WARN: {'; '.join(warn_reasons)}")
    if not fail_reasons and not warn_reasons and verdict != "DEGRADED_SKIP":
        print("  All metrics within thresholds.")

    # Exit code
    if verdict == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
