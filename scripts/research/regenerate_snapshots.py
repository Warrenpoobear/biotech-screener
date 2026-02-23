#!/usr/bin/env python3
"""Batch-regenerate historical snapshots using the current model.

RESEARCH ONLY — not for production use.

Inputs:  ruleset JSON, production_data/, existing snapshot dates
Outputs: data/snapshots_regenerated/{date}/rankings.csv per date

Runs the full run_screen.py pipeline (--decision-mode phase2) for each
historical date, producing fresh rankings with current hydration, decision
engine, overlays, and clinical alpha z-scores.

Usage:
    python scripts/research/regenerate_snapshots.py \
        --ruleset data/snapshots/2026-02-21/decision_ruleset.json

    python scripts/research/regenerate_snapshots.py \
        --ruleset data/snapshots/2026-02-21/decision_ruleset.json \
        --date-from 2026-02-01 --date-to 2026-02-10 --force
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from decision_engine import DecisionRuleset
from warm_caches import warm_ctgov


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_dates(snapshot_source: Path, date_from: str, date_to: str) -> list[str]:
    """Return sorted YYYY-MM-DD dirs that contain rankings.csv."""
    dates = sorted(
        p.name
        for p in snapshot_source.iterdir()
        if p.is_dir()
        and len(p.name) == 10
        and p.name[4] == "-"
        and (p / "rankings.csv").exists()
    )
    if date_from:
        dates = [d for d in dates if d >= date_from]
    if date_to:
        dates = [d for d in dates if d <= date_to]
    return dates


def preflight(args: argparse.Namespace) -> tuple[DecisionRuleset, list[str]]:
    """Validate inputs and return (ruleset, dates)."""
    # Ruleset
    ruleset = DecisionRuleset.from_json(str(args.ruleset))
    print(f"Ruleset: {args.ruleset.name}  ID={ruleset.ruleset_id}")

    # Production data
    data_dir = args.data_dir
    required = ["universe.json", "financial_records.json", "market_data.json", "price_history.csv"]
    missing = [f for f in required if not (data_dir / f).exists()]
    if missing:
        print(f"FATAL: missing in {data_dir}: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    # Dates
    dates = discover_dates(args.snapshot_source, args.date_from, args.date_to)
    if not dates:
        print("FATAL: no snapshot dates found in range", file=sys.stderr)
        sys.exit(1)
    print(f"Dates: {len(dates)}  ({dates[0]} → {dates[-1]})")

    return ruleset, dates


def warm_ctgov_caches(dates: list[str], data_dir: Path) -> None:
    """Pre-warm CTGov PIT caches for all dates (idempotent, ~1s each)."""
    print(f"\n--- Phase 1: Warming CTGov caches ({len(dates)} dates) ---")
    for snap_date in dates:
        as_of = date.fromisoformat(snap_date)
        n = warm_ctgov(as_of, data_dir)
        print(f"  {snap_date}: {n} trial records")


def count_rankings(csv_path: Path) -> tuple[int, int]:
    """Return (total_rows, eligible_count) from a rankings.csv."""
    total = 0
    eligible = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if row.get("eligible", "") in ("1", "True", "true"):
                eligible += 1
    return total, eligible


def run_one_date(
    snap_date: str,
    args: argparse.Namespace,
) -> bool:
    """Run run_screen.py for a single date. Returns True on success."""
    out_dir = args.out_dir / snap_date
    rankings = out_dir / "rankings.csv"

    if rankings.exists() and not args.force:
        return None  # skipped

    out_dir.mkdir(parents=True, exist_ok=True)
    output_json = out_dir / "screen_output.json"

    cmd = [
        sys.executable, str(PROJECT_ROOT / "run_screen.py"),
        "--as-of-date", snap_date,
        "--data-dir", str(args.data_dir),
        "--output", str(output_json),
        "--decision-mode", "phase2",
        "--ranking-mode", "decision",
        "--ruleset", str(args.ruleset),
        "--snapshot-dir", str(args.out_dir),
        "--prior-snapshot-dir", str(args.out_dir),
        "--no-delta",
        "--pit-mode", "degrade",
    ]

    t0 = time.monotonic()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        stderr_lines = result.stderr.strip().splitlines()
        tail = stderr_lines[-5:] if len(stderr_lines) > 5 else stderr_lines
        print(f"  {snap_date}: FAIL (exit {result.returncode}, {elapsed:.0f}s)")
        for line in tail:
            print(f"    | {line}")
        return False

    # Success — print summary
    if rankings.exists():
        total, elig = count_rankings(rankings)
        print(f"  {snap_date}: OK  ({total} tickers, {elig} eligible, {elapsed:.0f}s)")
    else:
        print(f"  {snap_date}: OK  (no rankings.csv found, {elapsed:.0f}s)")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-regenerate historical snapshots using current model.",
    )
    parser.add_argument(
        "--ruleset", type=Path, required=True,
        help="Path to target ruleset JSON",
    )
    parser.add_argument(
        "--date-from", type=str, default="2026-01-15",
        help="Earliest date to regenerate (default: 2026-01-15)",
    )
    parser.add_argument(
        "--date-to", type=str, default="2026-02-21",
        help="Latest date to regenerate (default: 2026-02-21)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=PROJECT_ROOT / "data" / "snapshots_regenerated",
        help="Output directory (default: data/snapshots_regenerated)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_ROOT / "production_data",
        help="Production data directory (default: production_data)",
    )
    parser.add_argument(
        "--snapshot-source", type=Path, default=PROJECT_ROOT / "data" / "snapshots",
        help="Source directory to discover which dates exist (default: data/snapshots)",
    )
    parser.add_argument(
        "--force", action="store_true", default=False,
        help="Re-run even if output already exists",
    )
    args = parser.parse_args()

    # Phase 0: Pre-flight
    print("=== Phase 0: Pre-flight ===")
    ruleset, dates = preflight(args)

    # Phase 1: Warm CTGov caches
    warm_ctgov_caches(dates, args.data_dir)

    # Phase 2: Sequential snapshot generation
    print(f"\n=== Phase 2: Generating snapshots ({len(dates)} dates) ===")
    ok = 0
    skip = 0
    fail = 0
    failed_dates: list[str] = []
    t_total = time.monotonic()

    for snap_date in dates:
        result = run_one_date(snap_date, args)
        if result is None:
            skip += 1
            print(f"  {snap_date}: SKIP (exists)")
        elif result:
            ok += 1
        else:
            fail += 1
            failed_dates.append(snap_date)

    elapsed_total = time.monotonic() - t_total

    # Phase 3: Summary
    print(f"\n=== Summary ===")
    print(f"  OK:   {ok}")
    print(f"  SKIP: {skip}")
    print(f"  FAIL: {fail}")
    print(f"  Wall: {elapsed_total:.0f}s ({elapsed_total / 60:.1f}m)")
    if failed_dates:
        print(f"  Failed dates: {', '.join(failed_dates)}")

    sys.exit(1 if fail > 0 else 0)


if __name__ == "__main__":
    main()
