#!/usr/bin/env python3
"""Backfill Archive Catalysts — add catalyst signals from trial milestones.

Reads each archive's trial_records.json + existing decision_inputs.json and
adds catalyst signals for tickers that currently have NONE.  Never overwrites
an existing catalyst signal (additive only).

New source types (all PIT-safe: first_posted <= as_of, Phase 2/3 only):

  trial_cd     — completion_date fallback when primary_completion_date is
                 missing.  Active-status filter.  Event type: DATA_READOUT.

  trial_active — presence signal for active Phase 2/3 trials whose dates
                 fall outside the forward window.  Emits synthetic
                 days_to_catalyst = window_days + 1.  Event type: TRIAL_ONGOING.

Usage:
    python scripts/backfill_archive_catalysts.py [--archives-dir PATH] \
        [--dry-run] [--dates YYYY-MM-DD,...] [--window-days 365]
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tarfile
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_ARCHIVE_DIR = PROJECT_ROOT / "data" / "archives"
DEFAULT_WINDOW_DAYS = 365

# Status whitelist for trial_cd and trial_active sources
ACTIVE_STATUSES = frozenset({
    "RECRUITING",
    "ACTIVE_NOT_RECRUITING",
    "NOT_YET_RECRUITING",
    "ENROLLING_BY_INVITATION",
})

# Phase whitelist (matches enrich_archive_inputs.py)
PHASE_WHITELIST = frozenset({
    "PHASE2", "PHASE3", "PHASE 2", "PHASE 3",
    "PHASE2/PHASE3", "PHASE 2/PHASE 3",
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class BackfillResult:
    """Per-archive backfill summary."""
    date_str: str
    n_dev: int = 0
    n_existing: int = 0
    n_backfilled: int = 0
    n_still_missing: int = 0
    source_counts: Dict[str, int] = field(default_factory=dict)
    status: str = "ok"
    error: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_date(s: Optional[str]) -> Optional[date]:
    """Parse YYYY-MM-DD string to date, or None."""
    if not s:
        return None
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _is_phase_2_3(phase_str: str) -> bool:
    return str(phase_str).upper() in PHASE_WHITELIST


def _extract_archive(tar_path: Path) -> Tuple[Path, Path]:
    """Extract tar.gz to tempdir, return (tmp_dir, archive_root)."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="backfill_cat_"))
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(tmp_dir)

    date_str = tar_path.name.replace(".tar.gz", "")
    archive_root = tmp_dir / date_str
    if not archive_root.exists():
        subdirs = [d for d in tmp_dir.iterdir() if d.is_dir()]
        if subdirs:
            archive_root = subdirs[0]
    return tmp_dir, archive_root


def _load_json(path: Path) -> Any:
    """Load JSON file or return empty list."""
    if not path.exists():
        return []
    with open(path, "r") as f:
        return json.load(f)


def _load_dev_tickers(csv_path: Path) -> Set[str]:
    """Load drug_developer tickers from rankings.csv."""
    tickers: Set[str] = set()
    if not csv_path.exists():
        return tickers
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("archetype", "").strip() == "drug_developer":
                t = row.get("ticker", "").strip().upper()
                if t:
                    tickers.add(t)
    return tickers


# ---------------------------------------------------------------------------
# Core backfill logic
# ---------------------------------------------------------------------------
def compute_backfill_catalyst(
    ticker: str,
    trials: List[Dict[str, Any]],
    as_of: date,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> Optional[Dict[str, Any]]:
    """Compute a backfill catalyst signal for one ticker.

    Tries sources in priority order:
      1. trial_cd   — completion_date fallback (real date)
      2. trial_active — active trial presence (synthetic date)

    Returns dict with {days_to_catalyst, in_optimal_window, catalyst_source,
    catalyst_event_type} or None if no signal found.
    """
    best_cd_days: Optional[int] = None
    has_active_trial = False

    for trial in trials:
        if trial.get("ticker", "").upper() != ticker.upper():
            continue

        phase = str(trial.get("phase", "")).upper()
        if not _is_phase_2_3(phase):
            continue

        # PIT gate: trial must be publicly known at as_of
        fp = _parse_date(trial.get("first_posted"))
        if fp is None or fp > as_of:
            continue

        status = str(trial.get("status", "")).upper()
        if status not in ACTIVE_STATUSES:
            continue

        # --- Source 1: trial_cd (completion_date fallback) ---
        pcd = _parse_date(trial.get("primary_completion_date"))
        cd = _parse_date(trial.get("completion_date"))

        if pcd is None and cd is not None:
            cd_days = (cd - as_of).days
            if 0 < cd_days <= window_days:
                if best_cd_days is None or cd_days < best_cd_days:
                    best_cd_days = cd_days

        # --- Source 2: trial_active (presence signal) ---
        start = _parse_date(trial.get("start_date"))
        if start is not None and start <= as_of:
            has_active_trial = True

    # Priority: trial_cd > trial_active
    if best_cd_days is not None:
        return {
            "days_to_catalyst": best_cd_days,
            "in_optimal_window": 14 <= best_cd_days <= 60,
            "catalyst_source": "trial_cd",
            "catalyst_event_type": "DATA_READOUT",
        }

    if has_active_trial:
        return {
            "days_to_catalyst": window_days + 1,
            "in_optimal_window": False,
            "catalyst_source": "trial_active",
            "catalyst_event_type": "TRIAL_ONGOING",
        }

    return None


# ---------------------------------------------------------------------------
# Per-archive backfill
# ---------------------------------------------------------------------------
def backfill_archive(
    tar_path: Path,
    window_days: int = DEFAULT_WINDOW_DAYS,
    dry_run: bool = False,
) -> BackfillResult:
    """Backfill catalyst signals in one archive.

    Reads decision_inputs.json + trial_records.json + rankings.csv.
    For dev tickers without existing catalyst, computes backfill signal.
    Writes updated decision_inputs.json back into the archive (unless dry_run).
    """
    date_str = tar_path.name.replace(".tar.gz", "")
    result = BackfillResult(date_str=date_str)

    as_of = _parse_date(date_str)
    if as_of is None:
        result.status = "error"
        result.error = f"Cannot parse date from {tar_path.name}"
        return result

    tmp_dir = None
    try:
        tmp_dir, archive_root = _extract_archive(tar_path)
        inputs_dir = archive_root / "inputs"
        csv_path = archive_root / "rankings.csv"

        if not inputs_dir.exists():
            result.status = "error"
            result.error = "no inputs/ directory"
            return result

        # Load data
        dev_tickers = _load_dev_tickers(csv_path)
        result.n_dev = len(dev_tickers)
        if not dev_tickers:
            result.status = "skipped"
            result.error = "no dev tickers"
            return result

        trials: List[Dict[str, Any]] = _load_json(inputs_dir / "trial_records.json")
        di_path = inputs_dir / "decision_inputs.json"
        decision_inputs: Dict[str, Dict[str, Any]] = {}
        if di_path.exists():
            with open(di_path, "r") as f:
                decision_inputs = json.load(f)

        # Count existing catalyst
        for ticker in dev_tickers:
            entry = decision_inputs.get(ticker, {})
            if "days_to_catalyst" in entry:
                result.n_existing += 1

        # Backfill missing tickers
        source_counts: Dict[str, int] = {}
        modified = False

        for ticker in sorted(dev_tickers):
            entry = decision_inputs.get(ticker, {})
            if "days_to_catalyst" in entry:
                continue  # never overwrite existing

            signal = compute_backfill_catalyst(ticker, trials, as_of, window_days)
            if signal is not None:
                # Merge into existing entry (preserving other fields)
                if ticker not in decision_inputs:
                    decision_inputs[ticker] = {}
                decision_inputs[ticker].update(signal)
                result.n_backfilled += 1
                src = signal["catalyst_source"]
                source_counts[src] = source_counts.get(src, 0) + 1
                modified = True

        result.n_still_missing = result.n_dev - result.n_existing - result.n_backfilled
        result.source_counts = source_counts

        if dry_run:
            result.status = "dry_run"
            return result

        if not modified:
            return result

        # Write updated decision_inputs.json
        with open(di_path, "w") as f:
            json.dump(decision_inputs, f, indent=2, default=str)

        # Repack archive
        new_tar = tar_path.with_suffix(".tar.gz.tmp")
        with tarfile.open(new_tar, "w:gz") as tar:
            tar.add(str(archive_root), arcname=date_str)
        shutil.move(str(new_tar), str(tar_path))

    except Exception as e:
        result.status = "error"
        result.error = str(e)
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill catalyst signals from trial milestones into archives."
    )
    parser.add_argument(
        "--archives-dir", type=str, default=str(DEFAULT_ARCHIVE_DIR),
        help=f"Directory containing .tar.gz archives (default: {DEFAULT_ARCHIVE_DIR})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute coverage deltas without modifying archives",
    )
    parser.add_argument(
        "--dates", type=str, default=None,
        help="Comma-separated list of dates to process (default: all)",
    )
    parser.add_argument(
        "--window-days", type=int, default=DEFAULT_WINDOW_DAYS,
        help=f"Forward look-ahead window in days (default: {DEFAULT_WINDOW_DAYS})",
    )
    args = parser.parse_args()

    archive_dir = Path(args.archives_dir)
    if not archive_dir.exists():
        print(f"ERROR: Archive directory not found: {archive_dir}", file=sys.stderr)
        return 1

    archives = sorted(archive_dir.glob("*.tar.gz"))
    if args.dates:
        date_set = set(d.strip() for d in args.dates.split(","))
        archives = [a for a in archives if a.name.replace(".tar.gz", "") in date_set]

    if not archives:
        print("No archives to process.")
        return 0

    mode = "DRY RUN" if args.dry_run else "BACKFILL"
    print(f"Catalyst backfill ({mode}, window={args.window_days}d)")
    print(f"Processing {len(archives)} archives from {archive_dir}")
    print()

    results: List[BackfillResult] = []
    total_backfilled = 0
    total_dev = 0

    for tar_path in archives:
        print(f"  {tar_path.name} ...", end=" ", flush=True)
        r = backfill_archive(tar_path, window_days=args.window_days, dry_run=args.dry_run)
        results.append(r)

        if r.status in ("ok", "dry_run"):
            tag = "DRY" if r.status == "dry_run" else "OK"
            src_str = ", ".join(f"{k}={v}" for k, v in sorted(r.source_counts.items()))
            print(
                f"{tag} (dev={r.n_dev}, existing={r.n_existing}, "
                f"+{r.n_backfilled}, still_missing={r.n_still_missing}"
                f"{', ' + src_str if src_str else ''})"
            )
            total_backfilled += r.n_backfilled
            total_dev += r.n_dev
        elif r.status == "skipped":
            print(f"SKIP: {r.error}")
        else:
            print(f"ERROR: {r.error}")

    # Summary
    print()
    n_ok = sum(1 for r in results if r.status in ("ok", "dry_run"))
    n_skip = sum(1 for r in results if r.status == "skipped")
    n_err = sum(1 for r in results if r.status == "error")
    total_existing = sum(r.n_existing for r in results)
    total_still = sum(r.n_still_missing for r in results)

    print(f"Summary: {n_ok} processed, {n_skip} skipped, {n_err} errors")
    print(f"  Total dev tickers:  {total_dev}")
    print(f"  Already had signal: {total_existing}")
    print(f"  Backfilled:         {total_backfilled}")
    print(f"  Still missing:      {total_still}")

    # Source breakdown
    all_sources: Dict[str, int] = {}
    for r in results:
        for k, v in r.source_counts.items():
            all_sources[k] = all_sources.get(k, 0) + v
    if all_sources:
        print(f"  Sources: {', '.join(f'{k}={v}' for k, v in sorted(all_sources.items()))}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
