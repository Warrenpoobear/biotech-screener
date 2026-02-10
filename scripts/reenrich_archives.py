#!/usr/bin/env python3
"""Re-enrich archives with wider trial window and validate catalyst coverage.

Two-phase pipeline:
  1. enrich_archive() — updates decision_inputs.json with wider trial PCD window
  2. backfill_decision_columns.process_archive() — propagates catalyst data into rankings.csv

Then validates each archive against the calibration DQ gate.

CLI:
  python scripts/reenrich_archives.py \
    [--date-range 2024-01-01 2024-12-31] \
    [--trial-window 365] \
    [--ruleset production_data/decision_rulesets/v1.2.0_candidate.json] \
    [--dry-run] \
    [--validate-only]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from enrich_archive_inputs import (
    ARCHIVE_DIR,
    PRICE_CSV,
    TRIAL_PCD_WINDOW_DAYS,
    enrich_archive,
    load_price_history,
)
from backfill_decision_columns import process_archive
from decision_engine import DecisionRuleset, DEFAULT_RULESET
from scripts.audit_archive_catalyst import audit_single_archive


DEFAULT_DQ_THRESHOLD = 80.0


# ---------------------------------------------------------------------------
# Core orchestration
# ---------------------------------------------------------------------------

def reenrich_and_validate(
    archive_dir: Path = ARCHIVE_DIR,
    price_csv: Path = PRICE_CSV,
    date_range: Tuple[str, str] | None = None,
    trial_window_days: int = TRIAL_PCD_WINDOW_DAYS,
    ruleset: DecisionRuleset | None = None,
    dq_threshold: float = DEFAULT_DQ_THRESHOLD,
    dry_run: bool = False,
    validate_only: bool = False,
    catalyst_only: bool = False,
    output_archive_dir: Path | None = None,
) -> Dict[str, Any]:
    """Re-enrich archives and validate catalyst coverage.

    When catalyst_only=True, only update catalyst fields in the existing
    decision_inputs.json, preserving defensive features (drawdown, beta, etc.).

    When output_archive_dir is set, enriched archives are written there
    instead of modifying originals in-place.

    Returns summary with per-archive before/after metrics.
    """
    ruleset = ruleset or DEFAULT_RULESET
    archives = sorted(archive_dir.glob("*.tar.gz"))

    # Filter by date range if specified
    if date_range:
        d_min = datetime.strptime(date_range[0], "%Y-%m-%d").date()
        d_max = datetime.strptime(date_range[1], "%Y-%m-%d").date()
        filtered = []
        for a in archives:
            date_str = a.name.replace(".tar.gz", "")
            try:
                d = datetime.strptime(date_str, "%Y-%m-%d").date()
                if d_min <= d <= d_max:
                    filtered.append(a)
            except ValueError:
                continue
        archives = filtered

    summary: Dict[str, Any] = {
        "n_archives": len(archives),
        "trial_window_days": trial_window_days,
        "ruleset_id": ruleset.ruleset_id,
        "dq_threshold": dq_threshold,
        "dry_run": dry_run,
        "validate_only": validate_only,
        "catalyst_only": catalyst_only,
        "archives": [],
    }

    if not archives:
        print("No archives matched.")
        return summary

    # Load prices once (only needed for enrichment, not validate-only)
    all_prices = {}
    if not validate_only:
        print(f"Loading price history from {price_csv} ...")
        all_prices = load_price_history(price_csv)
        print(f"  {len(all_prices)} tickers loaded")
        print()

    # Ensure output directory exists
    if output_archive_dir is not None:
        output_archive_dir.mkdir(parents=True, exist_ok=True)

    for tar_path in archives:
        date_str = tar_path.name.replace(".tar.gz", "")
        entry: Dict[str, Any] = {"date": date_str}

        # When output_archive_dir is set, enriched archives go there
        out_tar = (output_archive_dir / tar_path.name) if output_archive_dir else None

        # Phase 0: Pre-audit (before changes)
        pre_audit = audit_single_archive(tar_path, dq_threshold=dq_threshold)
        entry["before"] = {
            "missing_pct": pre_audit.get("catalyst_missing_pct"),
            "passes_dq": pre_audit.get("passes_dq_gate"),
            "status": pre_audit.get("status"),
        }

        if validate_only:
            entry["after"] = entry["before"]
            entry["action"] = "validate_only"
            summary["archives"].append(entry)
            continue

        # Phase 1: Re-enrich (updates decision_inputs.json)
        print(f"  {date_str}: enrich ...", end=" ", flush=True)
        enrich_result = enrich_archive(
            tar_path, all_prices,
            dry_run=dry_run,
            trial_window_days=trial_window_days,
            catalyst_only=catalyst_only,
            output_tar_path=out_tar,
        )
        enrich_status = enrich_result.get("status", "error")
        cat_count = enrich_result.get("stats", {}).get("n_catalyst", 0)
        print(f"{enrich_status} (cat={cat_count})", end="", flush=True)

        if enrich_status == "error":
            entry["action"] = "enrich_error"
            entry["error"] = enrich_result.get("error", "unknown")
            print(f" ERROR: {entry['error']}")
            summary["archives"].append(entry)
            continue

        # Phase 2: Backfill decision columns (updates rankings.csv)
        # When output dir is set, backfill reads/writes the output copy
        bf_tar = out_tar or tar_path
        if not dry_run:
            print(" → backfill ...", end=" ", flush=True)
            bf_result = process_archive(
                bf_tar, dry_run=False, ruleset=ruleset,
            )
            bf_status = bf_result.get("status", "error")
            print(f"{bf_status}", end="", flush=True)

            if bf_status == "error":
                entry["action"] = "backfill_error"
                entry["error"] = bf_result.get("error", "unknown")
                print(f" ERROR: {entry['error']}")
                summary["archives"].append(entry)
                continue

        # Phase 3: Post-audit (after changes)
        if not dry_run:
            print(" → audit ...", end=" ", flush=True)
            post_audit = audit_single_archive(bf_tar, dq_threshold=dq_threshold)
            entry["after"] = {
                "missing_pct": post_audit.get("catalyst_missing_pct"),
                "passes_dq": post_audit.get("passes_dq_gate"),
                "status": post_audit.get("status"),
            }
            delta = "—"
            if (entry["before"].get("missing_pct") is not None
                    and entry["after"].get("missing_pct") is not None):
                d = entry["after"]["missing_pct"] - entry["before"]["missing_pct"]
                delta = f"{d:+.1f}pp"
            dq_tag = "PASS" if entry["after"].get("passes_dq") else "FAIL"
            print(f" {dq_tag} (Δ={delta})")
        else:
            entry["after"] = {"status": "dry_run"}
            entry["action"] = "dry_run"
            print(" (dry run)")

        summary["archives"].append(entry)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-enrich archives with wider trial window and validate."
    )
    parser.add_argument(
        "--archive-dir", type=str, default=str(ARCHIVE_DIR),
        help=f"Directory containing .tar.gz archives (default: {ARCHIVE_DIR})",
    )
    parser.add_argument(
        "--price-csv", type=str, default=str(PRICE_CSV),
        help=f"Path to price_history.csv (default: {PRICE_CSV})",
    )
    parser.add_argument(
        "--date-range", nargs=2, metavar=("START", "END"),
        help="Filter archives by date range (YYYY-MM-DD YYYY-MM-DD)",
    )
    parser.add_argument(
        "--trial-window", type=int, default=TRIAL_PCD_WINDOW_DAYS,
        help=f"Trial PCD look-ahead window in days (default: {TRIAL_PCD_WINDOW_DAYS})",
    )
    parser.add_argument(
        "--ruleset", type=str, default=None,
        help="Path to decision engine ruleset JSON (default: built-in defaults)",
    )
    parser.add_argument(
        "--dq-threshold", type=float, default=DEFAULT_DQ_THRESHOLD,
        help=f"DQ gate: max catalyst_missing_pct (default: {DEFAULT_DQ_THRESHOLD})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute but don't modify archives",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Audit without modifying archives (for CI)",
    )
    parser.add_argument(
        "--catalyst-only", action="store_true",
        help="Only update catalyst fields; preserve existing defensive features",
    )
    parser.add_argument(
        "--output-archive-dir", type=str, default=None,
        help="Write enriched archives to this directory instead of modifying originals",
    )
    args = parser.parse_args()

    ruleset = (
        DecisionRuleset.from_json(args.ruleset)
        if args.ruleset
        else DEFAULT_RULESET
    )

    archive_dir = Path(args.archive_dir)
    if not archive_dir.exists():
        print(f"ERROR: Archive directory not found: {archive_dir}", file=sys.stderr)
        return 1

    date_range = tuple(args.date_range) if args.date_range else None

    mode = "VALIDATE-ONLY" if args.validate_only else ("DRY RUN" if args.dry_run else "LIVE")
    cat_label = " catalyst-only" if args.catalyst_only else ""
    print(f"Re-enrich pipeline ({mode}{cat_label})")
    print(f"  Trial window: {args.trial_window}d")
    print(f"  Ruleset: {ruleset.ruleset_id}")
    print(f"  DQ threshold: {args.dq_threshold}%")
    if date_range:
        print(f"  Date range: {date_range[0]} to {date_range[1]}")

    out_dir = Path(args.output_archive_dir) if args.output_archive_dir else None
    if out_dir:
        print(f"  Output dir: {out_dir}")
    print()

    result = reenrich_and_validate(
        archive_dir=archive_dir,
        price_csv=Path(args.price_csv),
        date_range=date_range,
        trial_window_days=args.trial_window,
        ruleset=ruleset,
        dq_threshold=args.dq_threshold,
        dry_run=args.dry_run,
        validate_only=args.validate_only,
        catalyst_only=args.catalyst_only,
        output_archive_dir=out_dir,
    )

    # Summary
    entries = result.get("archives", [])
    before_pass = sum(1 for e in entries
                      if e.get("before", {}).get("passes_dq"))
    after_pass = sum(1 for e in entries
                     if e.get("after", {}).get("passes_dq"))

    print()
    print(f"Results: {len(entries)} archives processed")
    print(f"  DQ gate PASS: {before_pass} → {after_pass}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
