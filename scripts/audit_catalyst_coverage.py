#!/usr/bin/env python3
"""Audit Catalyst Coverage — show backfill impact per archive.

For each archive, computes existing vs backfill catalyst coverage and prints
a summary table.  Uses the same logic as backfill_archive_catalysts.py but
in read-only mode.

Usage:
    python scripts/audit_catalyst_coverage.py [--archives-dir PATH]
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tarfile
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backfill_archive_catalysts import (
    DEFAULT_ARCHIVE_DIR,
    DEFAULT_WINDOW_DAYS,
    _load_dev_tickers,
    _load_json,
    _parse_date,
    compute_backfill_catalyst,
)


def _extract_archive(tar_path: Path) -> Tuple[Path, Path]:
    """Extract tar.gz to tempdir, return (tmp_dir, archive_root)."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="audit_cov_"))
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(tmp_dir)
    date_str = tar_path.name.replace(".tar.gz", "")
    archive_root = tmp_dir / date_str
    if not archive_root.exists():
        subdirs = [d for d in tmp_dir.iterdir() if d.is_dir()]
        if subdirs:
            archive_root = subdirs[0]
    return tmp_dir, archive_root


def audit_single(
    tar_path: Path, window_days: int = DEFAULT_WINDOW_DAYS,
) -> Dict[str, Any]:
    """Audit one archive: existing + backfill coverage."""
    date_str = tar_path.name.replace(".tar.gz", "")
    result: Dict[str, Any] = {"date": date_str, "status": "ok"}

    as_of = _parse_date(date_str)
    if as_of is None:
        result["status"] = "error"
        result["error"] = f"Cannot parse date from {tar_path.name}"
        return result

    tmp_dir = None
    try:
        tmp_dir, archive_root = _extract_archive(tar_path)
        inputs_dir = archive_root / "inputs"
        csv_path = archive_root / "rankings.csv"

        dev_tickers = _load_dev_tickers(csv_path)
        result["n_dev"] = len(dev_tickers)
        if not dev_tickers:
            result["status"] = "skipped"
            result["error"] = "no dev tickers"
            return result

        trials = _load_json(inputs_dir / "trial_records.json")
        di_path = inputs_dir / "decision_inputs.json"
        decision_inputs: Dict[str, Dict[str, Any]] = {}
        if di_path.exists():
            with open(di_path, "r") as f:
                decision_inputs = json.load(f)

        # Count existing
        n_existing = 0
        for ticker in dev_tickers:
            entry = decision_inputs.get(ticker, {})
            if "days_to_catalyst" in entry:
                n_existing += 1

        # Count backfill candidates
        n_trial_cd = 0
        n_trial_active = 0
        for ticker in sorted(dev_tickers):
            entry = decision_inputs.get(ticker, {})
            if "days_to_catalyst" in entry:
                continue
            signal = compute_backfill_catalyst(ticker, trials, as_of, window_days)
            if signal is not None:
                src = signal["catalyst_source"]
                if src == "trial_cd":
                    n_trial_cd += 1
                elif src == "trial_active":
                    n_trial_active += 1

        n_backfill = n_trial_cd + n_trial_active
        n_combined = n_existing + n_backfill
        n_dev = len(dev_tickers)

        result["n_existing"] = n_existing
        result["existing_pct"] = round(n_existing / n_dev * 100, 1) if n_dev else 0.0
        result["n_backfill"] = n_backfill
        result["backfill_pct"] = round(n_backfill / n_dev * 100, 1) if n_dev else 0.0
        result["n_combined"] = n_combined
        result["combined_pct"] = round(n_combined / n_dev * 100, 1) if n_dev else 0.0
        result["delta_pp"] = round(result["combined_pct"] - result["existing_pct"], 1)
        result["trial_cd_n"] = n_trial_cd
        result["trial_active_n"] = n_trial_active

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit catalyst coverage (existing + backfill) per archive."
    )
    parser.add_argument(
        "--archives-dir", type=str, default=str(DEFAULT_ARCHIVE_DIR),
        help=f"Directory containing .tar.gz archives (default: {DEFAULT_ARCHIVE_DIR})",
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
    if not archives:
        print("No archives found.")
        return 0

    print(f"Auditing catalyst coverage for {len(archives)} archives (window={args.window_days}d)")
    print()

    # Header
    print(
        f"{'Date':<12} {'Dev':>4} {'Exist%':>7} {'Bkfill%':>8} {'Comb%':>7} "
        f"{'Delta':>6} {'CD':>4} {'Active':>6}"
    )
    print("-" * 60)

    results: List[Dict[str, Any]] = []
    for tar_path in archives:
        r = audit_single(tar_path, window_days=args.window_days)
        results.append(r)

        if r["status"] == "ok":
            print(
                f"{r['date']:<12} {r['n_dev']:>4} "
                f"{r['existing_pct']:>6.1f}% {r['backfill_pct']:>7.1f}% "
                f"{r['combined_pct']:>6.1f}% {r['delta_pp']:>+5.1f}pp "
                f"{r['trial_cd_n']:>4} {r['trial_active_n']:>6}"
            )
        else:
            print(f"{r['date']:<12} {r['status'].upper()}: {r.get('error', '')}")

    # Summary
    ok = [r for r in results if r["status"] == "ok"]
    if ok:
        avg_delta = sum(r["delta_pp"] for r in ok) / len(ok)
        avg_existing = sum(r["existing_pct"] for r in ok) / len(ok)
        avg_combined = sum(r["combined_pct"] for r in ok) / len(ok)
        print("-" * 60)
        print(
            f"{'Average':<12} {'':>4} {avg_existing:>6.1f}% {'':>8} "
            f"{avg_combined:>6.1f}% {avg_delta:>+5.1f}pp"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
