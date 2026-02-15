#!/usr/bin/env python3
"""Rollup catalyst shadow metrics into a time-series CSV for trending.

Scans ``data/snapshots/*/catalyst_shadow_metrics.json`` and writes a
single CSV with one row per snapshot date.

Usage:
    python scripts/rollup_shadow_metrics.py [--snapshot-dir DIR] [--output PATH]

Defaults:
    --snapshot-dir  data/snapshots
    --output        output/catalyst_shadow_timeseries.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

_DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

COLUMNS = [
    "as_of_date",
    "prior_date",
    "bad_to_good_count",
    "good_to_bad_count",
    "bad_to_good_in_top60",
    "bad_to_good_in_top100",
    "median_catalyst_days_good",
    "p10_catalyst_days",
    "p50_catalyst_days",
    "p90_catalyst_days",
    "A_tier_count",
    "top60_overlap",
    "top100_overlap",
    "sec_8k_events",
    "ctgov_events",
    "fda_events",
]


def collect_shadow_metrics(snapshot_dir: Path) -> list[dict]:
    """Read all catalyst_shadow_metrics.json files, return sorted rows."""
    rows: list[dict] = []
    if not snapshot_dir.exists():
        return rows

    for d in sorted(snapshot_dir.iterdir()):
        if not d.is_dir() or not _DATE_DIR_RE.match(d.name):
            continue
        metrics_path = d / "catalyst_shadow_metrics.json"
        if not metrics_path.exists():
            continue
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        row = {col: data.get(col) for col in COLUMNS}
        # Ensure as_of_date is always set (fallback to dir name)
        if not row.get("as_of_date"):
            row["as_of_date"] = d.name
        rows.append(row)

    rows.sort(key=lambda r: r.get("as_of_date", ""))
    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    """Write rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rollup catalyst shadow metrics into a time-series CSV.",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path("data/snapshots"),
        help="Root snapshot directory (default: data/snapshots)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/catalyst_shadow_timeseries.csv"),
        help="Output CSV path (default: output/catalyst_shadow_timeseries.csv)",
    )
    args = parser.parse_args(argv)

    rows = collect_shadow_metrics(args.snapshot_dir)
    if not rows:
        print("No catalyst_shadow_metrics.json files found.", file=sys.stderr)
        return 1

    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")

    # Print summary table
    print(f"\n{'date':<12} {'B→G':>4} {'t60':>4} {'t100':>4} {'G→B':>4} {'med_d':>6} "
          f"{'A_cnt':>5} {'t60%':>6} {'t100%':>6} {'8K':>5} {'CTG':>5} {'FDA':>4}")
    print("-" * 83)
    for r in rows:
        def _f(v, fmt=".0f"):
            return f"{v:{fmt}}" if v is not None else "-"

        def _pct(v):
            return f"{v * 100:.1f}" if v is not None else "-"

        print(
            f"{r['as_of_date']:<12} "
            f"{_f(r['bad_to_good_count']):>4} "
            f"{_f(r.get('bad_to_good_in_top60')):>4} "
            f"{_f(r.get('bad_to_good_in_top100')):>4} "
            f"{_f(r['good_to_bad_count']):>4} "
            f"{_f(r['median_catalyst_days_good']):>6} "
            f"{_f(r['A_tier_count']):>5} "
            f"{_pct(r['top60_overlap']):>6} "
            f"{_pct(r['top100_overlap']):>6} "
            f"{_f(r['sec_8k_events']):>5} "
            f"{_f(r['ctgov_events']):>5} "
            f"{_f(r['fda_events']):>4}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
