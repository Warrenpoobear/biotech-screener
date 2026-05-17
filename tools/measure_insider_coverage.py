#!/usr/bin/env python3
"""Spec 104: Measure insider_net_buy_value_90d coverage across production snapshots.

Standalone CLI that reads production rankings.csv snapshots and reports
coverage statistics for the insider diagnostic column. Emits per-snapshot
JSON artifacts and a summary markdown report with a stability verdict.

Semantics (CRITICAL — never collapse these categories):
    blank / "" / None / NaN = no coverage / not fetched
    0.0                     = fetched, no insider activity
    positive                = net insider buying
    negative                = net insider selling

Usage:
    python tools/measure_insider_coverage.py \
        --start-date 2026-05-01 --end-date 2026-05-14

    python tools/measure_insider_coverage.py \
        --start-date 2026-05-01 --end-date 2026-05-14 \
        --snapshot-dir data/snapshots
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
COLUMN = "insider_net_buy_value_90d"
DEFAULT_SNAPSHOT_DIR = REPO_ROOT / "data" / "snapshots"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "insider_diagnostics"

# Stability threshold: max-min spread of nonblank_pct across snapshots
STABILITY_THRESHOLD_PP = 5.0  # percentage points


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure insider_net_buy_value_90d coverage across production snapshots.",
    )
    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date (YYYY-MM-DD, inclusive)",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="End date (YYYY-MM-DD, inclusive)",
    )
    parser.add_argument(
        "--snapshot-dir",
        default=str(DEFAULT_SNAPSHOT_DIR),
        help="Path to snapshot root directory (default: data/snapshots)",
    )
    return parser.parse_args(argv)


def _date_range(start: str, end: str) -> List[str]:
    """Return list of YYYY-MM-DD strings from start to end inclusive."""
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    dates: List[str] = []
    current = start_dt
    while current <= end_dt:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates


def measure_snapshot(rankings_path: Path) -> Optional[Dict[str, Any]]:
    """Measure insider coverage for a single rankings.csv.

    Returns None if the file does not exist or lacks the target column.
    """
    if not rankings_path.exists():
        return None

    df = pd.read_csv(rankings_path)

    if COLUMN not in df.columns:
        return None

    series = df[COLUMN]
    total_rows = len(series)

    if total_rows == 0:
        return None

    # blank = NaN / None / empty string (after read_csv, empty strings become NaN,
    # but be defensive about mixed types)
    blank_mask = series.isna()
    # Also catch literal empty strings if the CSV wasn't parsed cleanly
    if series.dtype == object:
        blank_mask = blank_mask | (series.astype(str).str.strip() == "")

    # Convert to numeric for value comparisons (non-numeric -> NaN)
    numeric = pd.to_numeric(series, errors="coerce")
    nonblank = numeric.dropna()

    blank_count = int(blank_mask.sum())
    zero_count = int((nonblank == 0.0).sum())
    positive_count = int((nonblank > 0).sum())
    negative_count = int((nonblank < 0).sum())

    blank_pct = round(blank_count / total_rows * 100, 2)
    zero_pct = round(zero_count / total_rows * 100, 2)
    positive_pct = round(positive_count / total_rows * 100, 2)
    negative_pct = round(negative_count / total_rows * 100, 2)
    nonblank_pct = round(100 - blank_pct, 2)
    activity_pct = round((positive_count + negative_count) / total_rows * 100, 2)

    return {
        "total_rows": total_rows,
        "blank_count": blank_count,
        "blank_pct": blank_pct,
        "zero_count": zero_count,
        "zero_pct": zero_pct,
        "positive_count": positive_count,
        "positive_pct": positive_pct,
        "negative_count": negative_count,
        "negative_pct": negative_pct,
        "nonblank_pct": nonblank_pct,
        "activity_pct": activity_pct,
    }


def _write_per_snapshot_artifact(date_str: str, stats: Dict[str, Any]) -> Path:
    """Write per-snapshot JSON artifact."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_date = date_str.replace("-", "_")
    path = ARTIFACTS_DIR / f"coverage_{safe_date}.json"
    payload = {"date": date_str, "column": COLUMN, **stats}
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _generate_report(
    all_stats: List[Dict[str, Any]],
    start_date: str,
    end_date: str,
) -> str:
    """Generate the summary markdown report."""
    lines: List[str] = []
    lines.append("# Insider Diagnostic Stabilization Report")
    lines.append("")
    lines.append(f"**Column:** `{COLUMN}`")
    lines.append(f"**Date range:** {start_date} to {end_date}")
    lines.append(f"**Snapshots measured:** {len(all_stats)}")
    lines.append("")

    if not all_stats:
        lines.append("## Result")
        lines.append("")
        lines.append("No snapshots found in the specified date range.")
        lines.append("")
        lines.append("**Verdict:** NO DATA")
        return "\n".join(lines)

    # --- Per-snapshot table ---
    lines.append("## Per-Snapshot Coverage")
    lines.append("")
    lines.append("| Date | Rows | Blank% | Zero% | Positive% | Negative% | Nonblank% | Activity% |")
    lines.append("|------|------|--------|-------|-----------|-----------|-----------|-----------|")
    for s in all_stats:
        lines.append(
            f"| {s['date']} "
            f"| {s['total_rows']} "
            f"| {s['blank_pct']:.1f} "
            f"| {s['zero_pct']:.1f} "
            f"| {s['positive_pct']:.1f} "
            f"| {s['negative_pct']:.1f} "
            f"| {s['nonblank_pct']:.1f} "
            f"| {s['activity_pct']:.1f} |"
        )
    lines.append("")

    # --- Stability calculation ---
    nonblank_values = [s["nonblank_pct"] for s in all_stats]
    nb_min = min(nonblank_values)
    nb_max = max(nonblank_values)
    nb_spread = round(nb_max - nb_min, 2)
    spread_pass = nb_spread <= STABILITY_THRESHOLD_PP

    lines.append("## Stability Analysis")
    lines.append("")
    lines.append(f"- **Nonblank% range:** {nb_min:.1f}% to {nb_max:.1f}%")
    lines.append(f"- **Max-min spread:** {nb_spread:.1f} pp")
    lines.append(f"- **Spread threshold:** {STABILITY_THRESHOLD_PP:.1f} pp")
    lines.append(
        f"- **Spread check:** {'PASS' if spread_pass else 'FAIL'} "
        f"({nb_spread:.1f} pp {'<=' if spread_pass else '>'} {STABILITY_THRESHOLD_PP:.1f} pp)"
    )
    lines.append("")

    # --- Blank/zero ratio stability ---
    blank_pcts = [s["blank_pct"] for s in all_stats]
    zero_pcts = [s["zero_pct"] for s in all_stats]
    blank_spread = round(max(blank_pcts) - min(blank_pcts), 2)
    zero_spread = round(max(zero_pcts) - min(zero_pcts), 2)

    lines.append("### Blank/Zero Ratio Stability")
    lines.append("")
    lines.append(f"- **Blank% spread:** {blank_spread:.1f} pp")
    lines.append(f"- **Zero% spread:** {zero_spread:.1f} pp")
    lines.append("")

    # --- Verdict ---
    lines.append("## Verdict")
    lines.append("")

    # Determine end_date relative to the 2026-05-15 final snapshot date
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    final_snapshot_dt = datetime(2026, 5, 15)

    if end_dt < final_snapshot_dt:
        verdict = "MEASURED / pending 2026-05-15 final snapshot"
    elif spread_pass:
        verdict = "STABLE"
    else:
        verdict = "UNSTABLE"

    lines.append(f"**{verdict}**")
    lines.append("")

    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    snapshot_dir = Path(args.snapshot_dir)
    dates = _date_range(args.start_date, args.end_date)

    if not dates:
        print("ERROR: start-date must be <= end-date", file=sys.stderr)
        return 1

    print(f"Scanning {len(dates)} dates from {args.start_date} to {args.end_date}")
    print(f"Snapshot directory: {snapshot_dir}")
    print(f"Target column: {COLUMN}")
    print()

    all_stats: List[Dict[str, Any]] = []
    skipped = 0

    for date_str in dates:
        rankings_path = snapshot_dir / date_str / "rankings.csv"
        stats = measure_snapshot(rankings_path)

        if stats is None:
            skipped += 1
            continue

        stats_with_date = {"date": date_str, **stats}
        all_stats.append(stats_with_date)

        # Write per-snapshot artifact
        artifact_path = _write_per_snapshot_artifact(date_str, stats)
        print(
            f"  {date_str}: nonblank={stats['nonblank_pct']:.1f}% "
            f"activity={stats['activity_pct']:.1f}% "
            f"-> {artifact_path.name}"
        )

    print()
    print(f"Measured: {len(all_stats)} snapshots, skipped: {skipped} (missing/no column)")

    # Write summary report
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report = _generate_report(all_stats, args.start_date, args.end_date)
    safe_end = args.end_date.replace("-", "_")
    report_path = ARTIFACTS_DIR / f"stabilization_report_{safe_end}.md"
    report_path.write_text(report)
    print(f"Report: {report_path}")

    # Print summary to stdout
    if all_stats:
        nonblank_values = [s["nonblank_pct"] for s in all_stats]
        spread = round(max(nonblank_values) - min(nonblank_values), 2)
        print()
        print(f"Nonblank% spread: {spread:.1f} pp (threshold: {STABILITY_THRESHOLD_PP:.1f} pp)")
        print(f"Spread check: {'PASS' if spread <= STABILITY_THRESHOLD_PP else 'FAIL'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
