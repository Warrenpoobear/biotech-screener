#!/usr/bin/env python3
"""Standalone drift check CLI — compare two rankings.csv files.

Thin wrapper around _compute_drift_metrics / _write_drift_report_md /
DriftThresholds from run_daily_production.py.

Exit codes:
  0  all metrics within thresholds (OK)
  2  at least one threshold breached (WARN)
  Never exits 1 — drift is advisory only.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

# Ensure tools/ and repo root are importable
_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOLS_DIR.parent
sys.path.insert(0, str(_TOOLS_DIR))
sys.path.insert(0, str(_REPO_ROOT))

from run_daily_production import (
    DriftThresholds,
    _compute_drift_metrics,
    _write_drift_report_md,
)


def run_drift_check(
    current_csv: Path,
    prior_csv: Path,
    output_dir: Path,
    thresholds: DriftThresholds,
) -> int:
    """Compare two rankings CSVs and emit drift report.

    Returns 0 (OK) or 2 (WARN).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = _compute_drift_metrics(current_csv, prior_csv)

    # Evaluate thresholds
    warn_reasons: list[str] = []
    if metrics["top20_overlap_pct"] < thresholds.warn_top20_overlap_pct:
        warn_reasons.append(
            f"warn_top20_overlap_pct: {metrics['top20_overlap_pct']:.1f}% < {thresholds.warn_top20_overlap_pct}%"
        )
    if metrics["top60_overlap_pct"] < thresholds.warn_top60_overlap_pct:
        warn_reasons.append(
            f"warn_top60_overlap_pct: {metrics['top60_overlap_pct']:.1f}% < {thresholds.warn_top60_overlap_pct}%"
        )
    if metrics["rank_spearman_rho"] is not None and metrics["rank_spearman_rho"] < thresholds.warn_rank_spearman_rho:
        warn_reasons.append(
            f"warn_rank_spearman_rho: {metrics['rank_spearman_rho']:.4f} < {thresholds.warn_rank_spearman_rho}"
        )
    if metrics["mean_abs_rank_delta_top60"] > thresholds.warn_mean_abs_rank_delta_top60:
        warn_reasons.append(
            f"warn_mean_abs_rank_delta_top60: {metrics['mean_abs_rank_delta_top60']:.1f} > {thresholds.warn_mean_abs_rank_delta_top60}"
        )
    if metrics["tier_migration_count"] > thresholds.warn_tier_migration_count:
        warn_reasons.append(
            f"warn_tier_migration_count: {metrics['tier_migration_count']} > {thresholds.warn_tier_migration_count}"
        )
    if metrics["eligibility_change_count"] > thresholds.warn_eligibility_change_count:
        warn_reasons.append(
            f"warn_eligibility_change_count: {metrics['eligibility_change_count']} > {thresholds.warn_eligibility_change_count}"
        )

    status = "WARN" if warn_reasons else "OK"

    # Build report
    report = {
        "version": "1.0.0",
        "thresholds_id": thresholds.thresholds_id,
        "thresholds": asdict(thresholds),
        "current_date": current_csv.parent.name,
        "prior_date": prior_csv.parent.name,
        "metrics": metrics,
        "warn_reasons": warn_reasons,
        "status": status,
    }

    # Write artifacts
    report_json_path = output_dir / "drift_report.json"
    with open(report_json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    report_md_path = output_dir / "drift_report.md"
    _write_drift_report_md(report, report_md_path)

    # Emit per-metric ::warning:: annotations for CI
    for reason in warn_reasons:
        print(f"::warning::{reason}")

    # Summary
    if warn_reasons:
        print(f"Drift check: WARN ({len(warn_reasons)} threshold(s) breached)")
        return 2
    else:
        rho = metrics.get("rank_spearman_rho")
        print(
            f"Drift check: OK — "
            f"top20={metrics['top20_overlap_pct']:.0f}%, "
            f"top60={metrics['top60_overlap_pct']:.0f}%, "
            f"rho={rho}"
        )
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two rankings.csv files for drift",
        epilog="Exit codes: 0=OK, 2=WARN (never 1)",
    )
    parser.add_argument(
        "--current-csv", type=Path, required=True,
        help="Path to current/candidate rankings.csv",
    )
    parser.add_argument(
        "--prior-csv", type=Path, required=True,
        help="Path to prior/baseline rankings.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory for drift_report.json + drift_report.md",
    )
    parser.add_argument(
        "--thresholds", type=Path, default=None,
        help="Path to drift thresholds JSON (default: built-in defaults)",
    )
    args = parser.parse_args()

    if not args.current_csv.exists():
        print(f"::error::Current CSV not found: {args.current_csv}")
        sys.exit(2)
    if not args.prior_csv.exists():
        print(f"::error::Prior CSV not found: {args.prior_csv}")
        sys.exit(2)

    thresholds = DriftThresholds()
    if args.thresholds:
        thresholds = DriftThresholds.from_json(args.thresholds)

    rc = run_drift_check(args.current_csv, args.prior_csv, args.output_dir, thresholds)
    sys.exit(rc)


if __name__ == "__main__":
    main()
