#!/usr/bin/env python3
"""
backfill_ctgov_pit_history.py — Generate PIT-safe CT.gov caches for backtesting.

Reads the full trial_records.json and produces date-partitioned cache files
where each trial is PIT-filtered (study_first_posted OR last_update_posted
<= as_of_date).

Unlike the 13F backfill, this is purely offline: no network calls, just
slicing the existing trial dataset by PIT date.

Usage:
    python tools/backfill_ctgov_pit_history.py \
      --date-from 2024-01-01 --date-to 2026-02-20 \
      --cadence quarterly \
      --out-root cache/ctgov \
      --resume --report

Output per date:
    {out_root}/trial_records_{as_of_date}.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Repo root — same pattern as backfill_13f_history.py
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

TOOL_VERSION = "1.0.0"
SCHEMA_VERSION = "ctgov_pit_cache.v1"

# Reuse PIT date priority from the clinical features builder
PIT_DATE_PRIORITY = ["first_posted", "last_update_posted"]


# ---------------------------------------------------------------------------
# Date generation
# ---------------------------------------------------------------------------

def generate_quarter_end_dates(
    date_from: date,
    date_to: date,
) -> List[date]:
    """Return ascending list of quarter-end dates within [date_from, date_to]."""
    if date_from > date_to:
        return []
    quarter_ends = []
    qe_specs = [(3, 31), (6, 30), (9, 30), (12, 31)]
    year = date_from.year
    while year <= date_to.year:
        for month, day in qe_specs:
            d = date(year, month, day)
            if date_from <= d <= date_to:
                quarter_ends.append(d)
        year += 1
    return quarter_ends


def generate_month_end_dates(
    date_from: date,
    date_to: date,
) -> List[date]:
    """Return ascending list of month-end dates within [date_from, date_to]."""
    if date_from > date_to:
        return []
    dates = []
    year, month = date_from.year, date_from.month
    while True:
        # Last day of this month
        if month == 12:
            eom = date(year, 12, 31)
        else:
            eom = date(year, month + 1, 1) - timedelta(days=1)
        if eom > date_to:
            break
        if eom >= date_from:
            dates.append(eom)
        month += 1
        if month > 12:
            month = 1
            year += 1
    return dates


def generate_weekly_dates(
    date_from: date,
    date_to: date,
) -> List[date]:
    """Return ascending list of Friday dates within [date_from, date_to]."""
    if date_from > date_to:
        return []
    dates = []
    # Find first Friday >= date_from
    d = date_from
    while d.weekday() != 4:  # 4 = Friday
        d += timedelta(days=1)
    while d <= date_to:
        dates.append(d)
        d += timedelta(days=7)
    return dates


def generate_daily_dates(
    date_from: date,
    date_to: date,
) -> List[date]:
    """Return ascending list of weekday dates within [date_from, date_to]."""
    if date_from > date_to:
        return []
    dates = []
    d = date_from
    while d <= date_to:
        if d.weekday() < 5:  # Mon-Fri
            dates.append(d)
        d += timedelta(days=1)
    return dates


def generate_dates(
    date_from: date,
    date_to: date,
    cadence: str,
) -> List[date]:
    """Generate dates for the given cadence."""
    if cadence == "quarterly":
        return generate_quarter_end_dates(date_from, date_to)
    elif cadence == "monthly":
        return generate_month_end_dates(date_from, date_to)
    elif cadence == "weekly":
        return generate_weekly_dates(date_from, date_to)
    elif cadence == "daily":
        return generate_daily_dates(date_from, date_to)
    else:
        raise ValueError(f"Unknown cadence: {cadence!r}")


# ---------------------------------------------------------------------------
# PIT filtering
# ---------------------------------------------------------------------------

def _parse_trial_date(s) -> Optional[date]:
    """Parse YYYY-MM-DD or YYYY-MM string to date, or None."""
    if not s:
        return None
    raw = str(s).strip()[:10]
    if len(raw) == 7 and raw[4] == "-":
        raw = raw + "-01"
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def is_pit_admitted(trial: dict, as_of: date) -> Tuple[bool, str]:
    """Canonical PIT gate: trial admitted iff pit_date <= as_of_date.

    Returns (admitted, pit_date_field_used).
    """
    for field_name in PIT_DATE_PRIORITY:
        d = _parse_trial_date(trial.get(field_name))
        if d is not None:
            if d <= as_of:
                return True, field_name
            else:
                return False, field_name
    return False, ""


def filter_trials_pit(
    trials: List[dict],
    as_of: date,
) -> Tuple[List[dict], int, int]:
    """Filter trials by PIT admission. Returns (admitted, n_admitted, n_filtered)."""
    admitted = []
    n_filtered = 0
    for t in trials:
        ok, _ = is_pit_admitted(t, as_of)
        if ok:
            admitted.append(t)
        else:
            n_filtered += 1
    return admitted, len(admitted), n_filtered


def _stable_sort_trials(trials: List[dict]) -> List[dict]:
    """Sort trials deterministically for byte-stable output."""
    return sorted(trials, key=lambda t: (
        t.get("ticker", ""),
        t.get("nct_id", ""),
        t.get("brief_title", ""),
    ))


def _compute_input_hash(data: bytes) -> str:
    """SHA-256 of raw input bytes (first 8 hex chars)."""
    return hashlib.sha256(data).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _is_date_complete(out_root: Path, as_of: date) -> bool:
    """Return True if a valid cache file exists for this date."""
    cache_path = out_root / f"trial_records_{as_of.isoformat()}.json"
    if not cache_path.exists():
        return False
    try:
        with open(cache_path) as f:
            data = json.load(f)
        # Minimal invariants: is a list, has reasonable size
        if not isinstance(data, list):
            return False
        return True
    except (json.JSONDecodeError, OSError):
        return False


# ---------------------------------------------------------------------------
# Per-date builder
# ---------------------------------------------------------------------------

def build_pit_cache_for_date(
    as_of: date,
    all_trials: List[dict],
    out_root: Path,
    input_hash: str,
    pit_mode: str = "strict",
) -> Dict[str, Any]:
    """Build PIT-filtered trial cache for a single date.

    Returns summary dict with counts.
    """
    admitted, n_admitted, n_filtered = filter_trials_pit(all_trials, as_of)

    # Stable ordering
    admitted = _stable_sort_trials(admitted)

    # Write cache file
    out_root.mkdir(parents=True, exist_ok=True)
    cache_path = out_root / f"trial_records_{as_of.isoformat()}.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(admitted, f, indent=2)

    # Write sidecar metadata
    meta = {
        "schema_version": SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "as_of_date": as_of.isoformat(),
        "created_at": datetime.now(tz=timezone.utc)
            .isoformat().replace("+00:00", "Z"),
        "pit_mode": pit_mode,
        "pit_convention": "first_posted_or_last_update_lte_as_of",
        "input_hash": input_hash,
        "trials_total": n_admitted + n_filtered,
        "trials_admitted": n_admitted,
        "trials_filtered": n_filtered,
        "cache_file": cache_path.name,
    }
    meta_path = out_root / f"trial_records_{as_of.isoformat()}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "as_of_date": as_of.isoformat(),
        "trials_admitted": n_admitted,
        "trials_filtered": n_filtered,
        "trials_total": n_admitted + n_filtered,
        "cache_path": str(cache_path),
    }


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def backfill_ctgov_history(
    date_from: date,
    date_to: date,
    *,
    trial_records_path: Optional[Path] = None,
    cadence: str = "quarterly",
    out_root: Optional[Path] = None,
    resume: bool = False,
    report: bool = False,
    max_dates: Optional[int] = None,
    pit_mode: str = "strict",
) -> Dict[str, Any]:
    """Generate PIT-safe CT.gov caches for a date grid.

    Returns summary dict with per-date results.
    """
    if out_root is None:
        out_root = REPO_ROOT / "cache" / "ctgov"
    if trial_records_path is None:
        trial_records_path = REPO_ROOT / "production_data" / "trial_records.json"

    # Generate dates
    dates = generate_dates(date_from, date_to, cadence)
    if max_dates:
        dates = dates[:max_dates]

    # Load trial records
    raw_bytes = trial_records_path.read_bytes()
    input_hash = _compute_input_hash(raw_bytes)
    all_trials = json.loads(raw_bytes)
    if not isinstance(all_trials, list):
        raise ValueError(f"trial_records must be a list, got {type(all_trials)}")

    total_trials = len(all_trials)

    # Process dates
    per_date: List[Dict[str, Any]] = []
    skipped = 0

    for i, d in enumerate(dates):
        # Resume: skip if complete
        if resume and _is_date_complete(out_root, d):
            # Read existing for summary
            cache_path = out_root / f"trial_records_{d.isoformat()}.json"
            with open(cache_path) as f:
                existing = json.load(f)
            per_date.append({
                "as_of_date": d.isoformat(),
                "trials_admitted": len(existing),
                "trials_filtered": total_trials - len(existing),
                "trials_total": total_trials,
                "skipped": True,
            })
            skipped += 1
            continue

        result = build_pit_cache_for_date(
            as_of=d,
            all_trials=all_trials,
            out_root=out_root,
            input_hash=input_hash,
            pit_mode=pit_mode,
        )
        per_date.append(result)

    summary: Dict[str, Any] = {
        "tool_version": TOOL_VERSION,
        "date_from": date_from.isoformat(),
        "date_to": date_to.isoformat(),
        "cadence": cadence,
        "pit_mode": pit_mode,
        "input_hash": input_hash,
        "input_trial_count": total_trials,
        "total_dates": len(dates),
        "dates_skipped": skipped,
        "per_date": per_date,
    }

    if report and per_date:
        admitted_counts = [d["trials_admitted"] for d in per_date]
        summary["coverage_report"] = {
            "total_dates": len(per_date),
            "avg_admitted": round(sum(admitted_counts) / len(admitted_counts), 0),
            "min_admitted": min(admitted_counts),
            "max_admitted": max(admitted_counts),
        }

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate PIT-safe CT.gov caches for backtesting.",
    )
    parser.add_argument(
        "--date-from", required=True,
        help="Start date inclusive (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--date-to", required=True,
        help="End date inclusive (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--cadence", default="quarterly",
        choices=["quarterly", "monthly", "weekly", "daily"],
        help="Date cadence (default: quarterly)",
    )
    parser.add_argument(
        "--trial-records", default=None,
        help="Path to trial_records.json (default: production_data/trial_records.json)",
    )
    parser.add_argument(
        "--out-root", type=Path, default=None,
        help="Output root directory (default: cache/ctgov)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip dates with existing valid cache file",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Print aggregate coverage report at end",
    )
    parser.add_argument(
        "--max-dates", type=int, default=None,
        help="Limit number of dates to process (for testing)",
    )
    parser.add_argument(
        "--pit-mode", default="strict",
        choices=["strict", "degrade"],
        help="PIT mode (default: strict)",
    )
    args = parser.parse_args(argv)

    trial_path = None
    if args.trial_records:
        trial_path = Path(args.trial_records)
        if not trial_path.is_absolute():
            trial_path = REPO_ROOT / trial_path

    summary = backfill_ctgov_history(
        date_from=date.fromisoformat(args.date_from),
        date_to=date.fromisoformat(args.date_to),
        trial_records_path=trial_path,
        cadence=args.cadence,
        out_root=args.out_root,
        resume=args.resume,
        report=args.report,
        max_dates=args.max_dates,
        pit_mode=args.pit_mode,
    )

    total = summary["total_dates"]
    skipped = summary["dates_skipped"]
    print(f"\nDone: {total} dates processed ({skipped} skipped)")
    print(f"  Input: {summary['input_trial_count']} trials (hash={summary['input_hash']})")

    if "coverage_report" in summary:
        rpt = summary["coverage_report"]
        print(f"\nCoverage Report:")
        print(f"  Dates:         {rpt['total_dates']}")
        print(f"  Avg admitted:  {rpt['avg_admitted']:.0f}")
        print(f"  Min admitted:  {rpt['min_admitted']}")
        print(f"  Max admitted:  {rpt['max_admitted']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
