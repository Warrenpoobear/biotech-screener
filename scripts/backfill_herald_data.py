#!/usr/bin/env python3
"""Backfill Herald press release data for a date range.

Runs the full Herald pipeline for each date:
  1. Fetch company press releases (all IR sources)
  2. Deduplicate cross-source
  3. Classify (local keywords; optionally Grok API)
  4. Herald -> CRT intake (if snapshot exists for that date)

Produces:
  data/press_releases/releases_{DATE}.jsonl
  data/press_releases/deduped/deduped_{DATE}.jsonl
  data/press_releases/classified/classified_{DATE}.jsonl
  data/press_releases/health_{DATE}.json
  artifacts/herald_crt_intake/{DATE}_candidates.json  (if snapshot exists)

Usage:
    # Backfill last 7 days
    python scripts/backfill_herald_data.py --start 2026-04-03 --end 2026-04-10

    # Backfill with Grok classification (requires XAI_API_KEY)
    python scripts/backfill_herald_data.py --start 2026-04-01 --end 2026-04-10 --use-grok

    # Dry run (no writes)
    python scripts/backfill_herald_data.py --start 2026-04-01 --end 2026-04-10 --dry-run

    # Force re-process even if output exists
    python scripts/backfill_herald_data.py --start 2026-04-03 --end 2026-04-10 --force

    # Include CRT intake for dates with snapshots
    python scripts/backfill_herald_data.py --start 2026-04-03 --end 2026-04-10 --with-crt-intake
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# --- Paths ---
RELEASES_DIR = PROJECT_ROOT / "data" / "press_releases"
DEDUPED_DIR = RELEASES_DIR / "deduped"
CLASSIFIED_DIR = RELEASES_DIR / "classified"
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "snapshots"
CRT_INTAKE_DIR = PROJECT_ROOT / "artifacts" / "herald_crt_intake"

FETCH_SCRIPT = PROJECT_ROOT / "tools" / "fetch_company_press_releases.py"
DEDUPE_SCRIPT = PROJECT_ROOT / "tools" / "dedupe_press_releases.py"
CLASSIFY_SCRIPT = PROJECT_ROOT / "tools" / "classify_press_releases.py"
CRT_INTAKE_SCRIPT = PROJECT_ROOT / "tools" / "herald_crt_intake.py"


def _all_days(start: date, end: date):
    """Yield every calendar day in [start, end] inclusive."""
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _is_weekday(d: date) -> bool:
    """True if d is Monday-Friday."""
    return d.weekday() < 5


def _output_exists(d: date) -> Dict[str, bool]:
    """Check which Herald outputs exist for a date."""
    ds = d.isoformat()
    return {
        "releases": (RELEASES_DIR / f"releases_{ds}.jsonl").exists(),
        "deduped": (DEDUPED_DIR / f"deduped_{ds}.jsonl").exists(),
        "classified": (CLASSIFIED_DIR / f"classified_{ds}.jsonl").exists(),
        "health": (RELEASES_DIR / f"health_{ds}.json").exists(),
        "crt_intake": (CRT_INTAKE_DIR / f"{ds}_candidates.json").exists(),
        "snapshot": (SNAPSHOT_DIR / ds / "rankings.csv").exists(),
    }


def _run_step(
    label: str,
    cmd: List[str],
    timeout: int = 900,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run a subprocess step and return result dict."""
    if dry_run:
        logger.info("[DRY-RUN] %s: %s", label, " ".join(str(c) for c in cmd))
        return {"step": label, "status": "dry_run", "stdout": "", "stderr": ""}

    logger.info("%s: %s", label, " ".join(str(c) for c in cmd[-3:]))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT),
        )
        stdout_lines = (result.stdout or "").strip().split("\n")
        summary = stdout_lines[-1] if stdout_lines else "no output"
        if result.returncode != 0:
            logger.warning("%s failed (rc=%d): %s", label, result.returncode, summary)
            return {
                "step": label,
                "status": "failed",
                "returncode": result.returncode,
                "stdout": result.stdout or "",
                "stderr": result.stderr or "",
            }
        logger.info("%s -> %s", label, summary)
        return {
            "step": label,
            "status": "ok",
            "returncode": 0,
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
        }
    except subprocess.TimeoutExpired:
        logger.error("%s timed out after %ds", label, timeout)
        return {"step": label, "status": "timeout", "stdout": "", "stderr": ""}
    except Exception as e:
        logger.error("%s error: %s", label, e)
        return {"step": label, "status": "error", "stdout": "", "stderr": str(e)}


def backfill_one_date(
    d: date,
    *,
    use_grok: bool = False,
    with_crt_intake: bool = False,
    dry_run: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    """Run the full Herald pipeline for a single date.

    Returns a summary dict with status and step results.
    """
    ds = d.isoformat()
    existing = _output_exists(d)
    result: Dict[str, Any] = {
        "date": ds,
        "steps": [],
        "skipped": False,
        "releases_count": 0,
        "deduped_count": 0,
        "classified_count": 0,
        "crt_candidates": 0,
    }

    # Skip if fully backfilled and not forced
    if not force and existing["classified"]:
        logger.info("%s: classified output exists, skipping (use --force to override)", ds)
        result["skipped"] = True
        # Count existing records
        classified_path = CLASSIFIED_DIR / f"classified_{ds}.jsonl"
        if classified_path.exists():
            result["classified_count"] = sum(1 for line in classified_path.read_text().splitlines() if line.strip())
        return result

    # Step 1: Fetch press releases
    if force or not existing["releases"]:
        step = _run_step(
            f"fetch ({ds})",
            [sys.executable, str(FETCH_SCRIPT), "--as-of-date", ds],
            timeout=900,
            dry_run=dry_run,
        )
        result["steps"].append(step)
        if step["status"] not in ("ok", "dry_run"):
            result["error"] = f"fetch failed: {step['status']}"
            return result
    else:
        logger.info("%s: releases already exist, reusing", ds)

    releases_path = RELEASES_DIR / f"releases_{ds}.jsonl"
    if not dry_run and not releases_path.exists():
        logger.warning("%s: no releases file produced, skipping downstream", ds)
        result["releases_count"] = 0
        return result

    if not dry_run:
        result["releases_count"] = sum(
            1 for line in releases_path.read_text().splitlines() if line.strip()
        )
        if result["releases_count"] == 0:
            logger.info("%s: 0 releases fetched, skipping dedupe/classify", ds)
            return result

    # Step 2: Deduplicate
    if force or not existing["deduped"]:
        step = _run_step(
            f"dedupe ({ds})",
            [sys.executable, str(DEDUPE_SCRIPT), "--input", str(releases_path)],
            timeout=120,
            dry_run=dry_run,
        )
        result["steps"].append(step)
    else:
        logger.info("%s: deduped output exists, reusing", ds)

    # Determine classify input: prefer deduped, fall back to raw
    deduped_path = DEDUPED_DIR / f"deduped_{ds}.jsonl"
    classify_input = deduped_path if deduped_path.exists() else releases_path
    if not dry_run:
        result["deduped_count"] = (
            sum(1 for line in deduped_path.read_text().splitlines() if line.strip())
            if deduped_path.exists()
            else result["releases_count"]
        )

    # Step 3: Classify
    if force or not existing["classified"]:
        classify_cmd = [sys.executable, str(CLASSIFY_SCRIPT), "--input", str(classify_input)]
        if use_grok:
            classify_cmd.append("--use-grok")
        step = _run_step(
            f"classify ({ds})",
            classify_cmd,
            timeout=600,
            dry_run=dry_run,
        )
        result["steps"].append(step)
    else:
        logger.info("%s: classified output exists, reusing", ds)

    if not dry_run:
        classified_path = CLASSIFIED_DIR / f"classified_{ds}.jsonl"
        if classified_path.exists():
            result["classified_count"] = sum(
                1 for line in classified_path.read_text().splitlines() if line.strip()
            )

    # Step 4: CRT intake (optional, requires snapshot)
    if with_crt_intake and existing["snapshot"]:
        if force or not existing["crt_intake"]:
            step = _run_step(
                f"crt-intake ({ds})",
                [sys.executable, str(CRT_INTAKE_SCRIPT), "--as-of-date", ds],
                timeout=120,
                dry_run=dry_run,
            )
            result["steps"].append(step)

            if not dry_run:
                crt_path = CRT_INTAKE_DIR / f"{ds}_candidates.json"
                if crt_path.exists():
                    crt_data = json.loads(crt_path.read_text())
                    result["crt_candidates"] = crt_data.get("n_candidates", 0)
    elif with_crt_intake and not existing["snapshot"]:
        logger.info("%s: no snapshot for CRT intake, skipping", ds)

    return result


def backfill(
    start: date,
    end: date,
    *,
    use_grok: bool = False,
    with_crt_intake: bool = False,
    dry_run: bool = False,
    force: bool = False,
    weekdays_only: bool = True,
    sleep_seconds: float = 5.0,
) -> Dict[str, Any]:
    """Run the Herald backfill across a date range.

    Returns aggregate summary.
    """
    dates = [d for d in _all_days(start, end)]
    if weekdays_only:
        dates = [d for d in dates if _is_weekday(d)]

    summary = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "total_dates": len(dates),
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "total_releases": 0,
        "total_deduped": 0,
        "total_classified": 0,
        "total_crt_candidates": 0,
        "date_results": [],
    }

    for i, d in enumerate(dates):
        logger.info("--- [%d/%d] %s ---", i + 1, len(dates), d.isoformat())
        result = backfill_one_date(
            d,
            use_grok=use_grok,
            with_crt_intake=with_crt_intake,
            dry_run=dry_run,
            force=force,
        )
        summary["date_results"].append(result)

        if result.get("skipped"):
            summary["skipped"] += 1
        elif result.get("error"):
            summary["failed"] += 1
        else:
            summary["processed"] += 1

        summary["total_releases"] += result.get("releases_count", 0)
        summary["total_deduped"] += result.get("deduped_count", 0)
        summary["total_classified"] += result.get("classified_count", 0)
        summary["total_crt_candidates"] += result.get("crt_candidates", 0)

        # Rate limit between dates (skip after last)
        if i < len(dates) - 1 and not dry_run and sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    """Print a human-readable backfill summary."""
    print()
    print("=" * 60)
    print("HERALD DATA BACKFILL SUMMARY")
    print("=" * 60)
    print(f"  Date range:      {summary['start']} -> {summary['end']}")
    print(f"  Total dates:     {summary['total_dates']}")
    print(f"  Processed:       {summary['processed']}")
    print(f"  Skipped:         {summary['skipped']}")
    print(f"  Failed:          {summary['failed']}")
    print(f"  Total releases:  {summary['total_releases']}")
    print(f"  Total deduped:   {summary['total_deduped']}")
    print(f"  Total classified:{summary['total_classified']}")
    if summary["total_crt_candidates"] > 0:
        print(f"  CRT candidates:  {summary['total_crt_candidates']}")

    # Per-date detail
    print()
    print(f"  {'Date':<12} {'Status':<10} {'Releases':>8} {'Deduped':>8} {'Classified':>10} {'CRT':>5}")
    print(f"  {'-' * 55}")
    for dr in summary["date_results"]:
        status = "SKIP" if dr.get("skipped") else "FAIL" if dr.get("error") else "OK"
        print(
            f"  {dr['date']:<12} {status:<10} "
            f"{dr.get('releases_count', 0):>8} "
            f"{dr.get('deduped_count', 0):>8} "
            f"{dr.get('classified_count', 0):>10} "
            f"{dr.get('crt_candidates', 0):>5}"
        )

    if summary["failed"] > 0:
        print()
        print("  FAILURES:")
        for dr in summary["date_results"]:
            if dr.get("error"):
                print(f"    {dr['date']}: {dr['error']}")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Backfill Herald press release data")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--use-grok", action="store_true", help="Use xAI Grok API for classification")
    parser.add_argument("--with-crt-intake", action="store_true", help="Run CRT intake for dates with snapshots")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without writing")
    parser.add_argument("--force", action="store_true", help="Re-process even if output exists")
    parser.add_argument("--include-weekends", action="store_true", help="Include weekends (default: weekdays only)")
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=5.0,
        help="Sleep between dates for rate limiting (default: 5s)",
    )
    parser.add_argument("--save-report", action="store_true", help="Write JSON report to artifacts/")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    if start > end:
        print(f"ERROR: start ({start}) must be <= end ({end})")
        return 1

    days = (end - start).days + 1
    mode = "DRY-RUN" if args.dry_run else "LIVE"
    print(f"Herald Backfill [{mode}]: {start} -> {end} ({days} calendar days)")
    print(f"  Grok: {'ON' if args.use_grok else 'OFF (local keywords only)'}")
    print(f"  CRT intake: {'ON' if args.with_crt_intake else 'OFF'}")
    print(f"  Force: {'ON' if args.force else 'OFF'}")
    print(f"  Weekdays only: {'NO' if args.include_weekends else 'YES'}")
    print()

    summary = backfill(
        start=start,
        end=end,
        use_grok=args.use_grok,
        with_crt_intake=args.with_crt_intake,
        dry_run=args.dry_run,
        force=args.force,
        weekdays_only=not args.include_weekends,
        sleep_seconds=args.sleep_seconds,
    )

    print_summary(summary)

    # Save report artifact
    if args.save_report and not args.dry_run:
        report_dir = PROJECT_ROOT / "artifacts" / "herald_backfill"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"backfill_{start}_{end}.json"
        report = {
            "schema": "herald_backfill_report.v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            **summary,
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
            f.write("\n")
        print(f"\nReport -> {report_path}")

    return 1 if summary["failed"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
