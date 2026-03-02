#!/usr/bin/env python3
"""Generate PIT-lite monthly archives from 2020-01 through present.

Uses run_screen.py --pit-mode degrade + archive_snapshot.py to build
self-contained archives for each end-of-month date.

PIT caveat: fundamentals/trials/13F use current-state production_data.
Prices/timing are true as-of.  This is "PIT-lite".

Usage:
    python scripts/generate_monthly_archives.py
    python scripts/generate_monthly_archives.py --start 2020-01 --end 2023-12
    python scripts/generate_monthly_archives.py --dry-run
    python scripts/generate_monthly_archives.py --single 2020-01-31
"""
from __future__ import annotations

import argparse
import calendar
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def last_business_day(year: int, month: int) -> date:
    """Return the last weekday (Mon-Fri) of the given month."""
    last_day = calendar.monthrange(year, month)[1]
    d = date(year, month, last_day)
    while d.weekday() >= 5:  # Sat=5, Sun=6
        d -= timedelta(days=1)
    return d


def generate_date_grid(
    start_ym: str, end_ym: str,
) -> List[str]:
    """Generate end-of-month business day dates from YYYY-MM to YYYY-MM."""
    start_y, start_m = int(start_ym[:4]), int(start_ym[5:7])
    end_y, end_m = int(end_ym[:4]), int(end_ym[5:7])

    dates: List[str] = []
    y, m = start_y, start_m
    while (y, m) <= (end_y, end_m):
        d = last_business_day(y, m)
        dates.append(d.isoformat())
        m += 1
        if m > 12:
            m = 1
            y += 1
    return dates


def run_one_date(
    date_str: str,
    data_dir: Path,
    snapshot_dir: Path,
    archive_dir: Path,
    output_dir: Path,
    log_dir: Path,
    dry_run: bool = False,
) -> bool:
    """Run screen + archive for a single date. Returns True on success."""
    output_file = output_dir / f"results_{date_str}.json"
    log_file = log_dir / f"backfill_{date_str}.log"

    if dry_run:
        print(f"  DRY RUN: would run screen + archive for {date_str}")
        return True

    # Step 1: Run scoring pipeline
    cmd_screen = [
        sys.executable, str(PROJECT_ROOT / "run_screen.py"),
        "--as-of-date", date_str,
        "--data-dir", str(data_dir),
        "--pit-mode", "degrade",
        "--output", str(output_file),
        "--snapshot-dir", str(snapshot_dir),
        "--log-level", "WARNING",
    ]

    with open(log_file, "w") as lf:
        result = subprocess.run(
            cmd_screen, cwd=str(PROJECT_ROOT),
            stdout=lf, stderr=subprocess.STDOUT,
            timeout=600,
        )

    if result.returncode != 0:
        print(f"  FAIL: screen exit {result.returncode}")
        return False

    # Step 2: Archive
    cmd_archive = [
        sys.executable, str(PROJECT_ROOT / "archive_snapshot.py"),
        "--date", date_str,
        "--snapshot-dir", str(snapshot_dir),
        "--data-dir", str(data_dir),
        "--archive-dir", str(archive_dir),
    ]

    with open(log_file, "a") as lf:
        result = subprocess.run(
            cmd_archive, cwd=str(PROJECT_ROOT),
            stdout=lf, stderr=subprocess.STDOUT,
            timeout=120,
        )

    if result.returncode != 0:
        print(f"  FAIL: archive exit {result.returncode}")
        return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PIT-lite monthly archives",
    )
    parser.add_argument("--start", default="2020-01",
                        help="Start YYYY-MM (default: 2020-01)")
    parser.add_argument("--end", default="2026-02",
                        help="End YYYY-MM (default: 2026-02)")
    parser.add_argument("--single", type=str, default=None,
                        help="Run a single date (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if archive exists")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / "production_data"
    snapshot_dir = PROJECT_ROOT / "data" / "snapshots"
    archive_dir = PROJECT_ROOT / "data" / "archives"
    output_dir = PROJECT_ROOT / "backfill_results"
    log_dir = PROJECT_ROOT / "backfill_logs"

    for d in [output_dir, log_dir, archive_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if args.single:
        dates = [args.single]
    else:
        dates = generate_date_grid(args.start, args.end)

    # Filter out existing archives unless --force
    todo = []
    skipped = 0
    for d in dates:
        if not args.force and (archive_dir / f"{d}.tar.gz").exists():
            skipped += 1
        else:
            todo.append(d)

    print("=" * 70)
    print("MONTHLY ARCHIVE GENERATION (PIT-lite)")
    print("=" * 70)
    print(f"  Date range:  {dates[0]} → {dates[-1]}")
    print(f"  Total dates: {len(dates)}")
    print(f"  To generate: {len(todo)}")
    print(f"  Skipped:     {skipped} (already exist)")
    print(f"  Dry run:     {args.dry_run}")
    print("=" * 70)
    print()

    if not todo:
        print("Nothing to do.")
        return

    success = 0
    fail = 0
    fail_dates = []
    t_start = time.time()

    for i, d in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {d} ...", end=" ", flush=True)
        t0 = time.time()

        ok = run_one_date(
            d, data_dir, snapshot_dir, archive_dir,
            output_dir, log_dir, args.dry_run,
        )

        dt = time.time() - t0
        if ok:
            success += 1
            sz = (archive_dir / f"{d}.tar.gz").stat().st_size if not args.dry_run else 0
            print(f"OK ({dt:.0f}s, {sz/1024:.0f} KB)")
        else:
            fail += 1
            fail_dates.append(d)
            print(f"FAIL ({dt:.0f}s)")

    elapsed = time.time() - t_start
    print()
    print("=" * 70)
    print(f"DONE: {success} ok, {fail} failed, {skipped} skipped ({elapsed:.0f}s)")
    if fail_dates:
        print(f"  Failed: {fail_dates}")
    print("=" * 70)


if __name__ == "__main__":
    main()
