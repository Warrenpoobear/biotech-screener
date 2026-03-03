#!/usr/bin/env python3
"""Re-stamp historical snapshots with data_sources provenance.

Reconstructs ``data_sources`` from existing PIT cache structure so that
``snapshot_preflight.py --check-dated-sources`` passes without WARNs.

Usage
-----
    python tools/restamp_snapshot_sources.py \
        --snapshot-root /tmp/snapshots_reranked_clinical_off \
        --sec-13f-pit-base data/caches/sec_13f/PIT \
        --ctgov-cache-dir cache/ctgov \
        [--dry-run] [--force] [--date-from YYYY-MM-DD] [--date-to YYYY-MM-DD] [--verbose]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Inline the column mapping so this module stays standalone — mirrors
# snapshot_preflight._SOURCE_FAMILY_COLUMNS exactly.
_SOURCE_FAMILY_COLUMNS = {
    "sec_13f": {
        "coinvest_score_z",
        "sponsor_tier1_count",
        "inst_delta_z",
        "has_coinvest_signal",
        "has_inst_delta",
    },
    "ctgov": {
        "clinical_score",
        "clinical_score_z",
        "clinical_score_v2",
        "clinical_readout_days",
        "clinical_coverage_flag",
    },
    "sec_8k": {
        "catalyst_days",
        "catalyst_decay_w",
        "catalyst_source",
        "has_catalyst_signal",
    },
}

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def _discover_13f_dates(pit_base: Path) -> List[str]:
    """Return sorted list of YYYY-MM-DD date strings from 13F PIT dirs."""
    dates: list[str] = []
    if not pit_base.is_dir():
        return dates
    for child in pit_base.iterdir():
        if child.is_dir() and _DATE_RE.match(child.name):
            dates.append(child.name)
    dates.sort()
    return dates


def _discover_ctgov_dates(ctgov_dir: Path) -> List[str]:
    """Return sorted list of YYYY-MM-DD date strings from ctgov cache files."""
    dates: list[str] = []
    if not ctgov_dir.is_dir():
        return dates
    for child in ctgov_dir.iterdir():
        if child.name.startswith("trial_records_") and child.name.endswith(".json"):
            # trial_records_YYYY-MM-DD.json
            stem = child.stem  # trial_records_YYYY-MM-DD
            date_part = stem.replace("trial_records_", "", 1)
            if _DATE_RE.match(date_part):
                dates.append(date_part)
    dates.sort()
    return dates


def _find_latest_cache_date(
    cache_dates: List[str], snap_date: str
) -> Optional[str]:
    """Return the latest cache date ``<= snap_date``, or ``None``."""
    best: Optional[str] = None
    for d in cache_dates:
        if d <= snap_date:
            best = d
        else:
            break  # sorted, so all remaining are > snap_date
    return best


# ---------------------------------------------------------------------------
# Detection helper
# ---------------------------------------------------------------------------

def _detect_active_families(header_cols: set[str]) -> set[str]:
    """Return set of family names whose indicator columns appear in header."""
    active: set[str] = set()
    for family, cols in _SOURCE_FAMILY_COLUMNS.items():
        if cols & header_cols:
            active.add(family)
    return active


def _read_csv_header(rankings_csv: Path) -> set[str]:
    """Read only the header row from a CSV file."""
    with open(rankings_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
    return set(header) if header else set()


# ---------------------------------------------------------------------------
# Core reconstruction
# ---------------------------------------------------------------------------

def reconstruct_data_sources(
    snap_date: str,
    rankings_csv: Path,
    *,
    sec_13f_cache_dates: List[str],
    ctgov_cache_dates: List[str],
    sec_13f_pit_base: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build a ``data_sources`` dict for one snapshot date.

    Only stamps families whose indicator columns are present in the CSV
    header.  Families without a matching PIT cache ``<= snap_date`` are
    omitted (not fabricated).
    """
    header = _read_csv_header(rankings_csv)
    active = _detect_active_families(header)
    sources: Dict[str, Any] = {}

    # --- sec_13f ---
    if "sec_13f" in active:
        cache_date = _find_latest_cache_date(sec_13f_cache_dates, snap_date)
        if cache_date is not None:
            effective_date = cache_date
            # Try to read effective date from index.json
            if sec_13f_pit_base is not None:
                idx_path = sec_13f_pit_base / cache_date / "index.json"
                if idx_path.is_file():
                    with open(idx_path, "r", encoding="utf-8") as f:
                        idx = json.load(f)
                    effective_date = idx.get("as_of_date", cache_date)
            sources["sec_13f"] = {
                "effective_date": effective_date,
                "lag_days": 45,
            }

    # --- ctgov ---
    if "ctgov" in active:
        cache_date = _find_latest_cache_date(ctgov_cache_dates, snap_date)
        if cache_date is not None:
            sources["ctgov"] = {"effective_date": cache_date}

    # --- sec_8k ---
    if "sec_8k" in active:
        sources["sec_8k"] = {"effective_date": snap_date}

    return sources


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def restamp_batch(
    snapshot_root: Path,
    *,
    sec_13f_pit_base: Path,
    ctgov_cache_dir: Path,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    dry_run: bool = False,
    force: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Re-stamp ``data_sources`` into every snapshot's metadata.json.

    Returns summary dict with counts.
    """
    # Discover cache dates once
    sec_13f_dates = _discover_13f_dates(sec_13f_pit_base)
    ctgov_dates = _discover_ctgov_dates(ctgov_cache_dir)

    # Discover snapshot dates
    snap_dates: list[str] = []
    for child in sorted(snapshot_root.iterdir()):
        if child.is_dir() and _DATE_RE.match(child.name):
            snap_dates.append(child.name)

    # Apply date filters
    if date_from:
        snap_dates = [d for d in snap_dates if d >= date_from]
    if date_to:
        snap_dates = [d for d in snap_dates if d <= date_to]

    summary = {
        "n_total": len(snap_dates),
        "n_stamped": 0,
        "n_skipped_existing": 0,
        "n_skipped_no_rankings": 0,
        "families": {"sec_13f": 0, "ctgov": 0, "sec_8k": 0},
    }

    for snap_date in snap_dates:
        snap_dir = snapshot_root / snap_date
        meta_path = snap_dir / "metadata.json"
        rankings_path = snap_dir / "rankings.csv"

        if not rankings_path.is_file():
            summary["n_skipped_no_rankings"] += 1
            if verbose:
                print(f"  SKIP {snap_date}: no rankings.csv")
            continue

        if not meta_path.is_file():
            summary["n_skipped_no_rankings"] += 1
            if verbose:
                print(f"  SKIP {snap_date}: no metadata.json")
            continue

        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        if "data_sources" in metadata and not force:
            summary["n_skipped_existing"] += 1
            if verbose:
                print(f"  SKIP {snap_date}: data_sources already present")
            continue

        sources = reconstruct_data_sources(
            snap_date,
            rankings_path,
            sec_13f_cache_dates=sec_13f_dates,
            ctgov_cache_dates=ctgov_dates,
            sec_13f_pit_base=sec_13f_pit_base,
        )

        if verbose:
            families_str = ", ".join(sorted(sources.keys())) or "(none)"
            print(f"  {'DRY ' if dry_run else ''}STAMP {snap_date}: {families_str}")

        if not dry_run:
            metadata["data_sources"] = sources
            metadata["data_sources_restamped_at"] = (
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            )
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, sort_keys=True)
                f.write("\n")

        summary["n_stamped"] += 1
        for fam in sources:
            if fam in summary["families"]:
                summary["families"][fam] += 1

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Re-stamp historical snapshots with data_sources provenance."
    )
    p.add_argument(
        "--snapshot-root",
        type=Path,
        required=True,
        help="Root directory containing YYYY-MM-DD snapshot subdirs.",
    )
    p.add_argument(
        "--sec-13f-pit-base",
        type=Path,
        required=True,
        help="Path to sec_13f PIT cache (e.g. data/caches/sec_13f/PIT).",
    )
    p.add_argument(
        "--ctgov-cache-dir",
        type=Path,
        required=True,
        help="Path to ctgov cache dir (e.g. cache/ctgov).",
    )
    p.add_argument("--dry-run", action="store_true", help="Preview without writing.")
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing data_sources entries.",
    )
    p.add_argument("--date-from", help="Only process dates >= YYYY-MM-DD.")
    p.add_argument("--date-to", help="Only process dates <= YYYY-MM-DD.")
    p.add_argument("--verbose", action="store_true", help="Print per-snapshot detail.")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    summary = restamp_batch(
        args.snapshot_root,
        sec_13f_pit_base=args.sec_13f_pit_base,
        ctgov_cache_dir=args.ctgov_cache_dir,
        date_from=args.date_from,
        date_to=args.date_to,
        dry_run=args.dry_run,
        force=args.force,
        verbose=args.verbose,
    )
    mode = "DRY RUN" if args.dry_run else "DONE"
    print(f"\n{mode}: {summary['n_stamped']} stamped, "
          f"{summary['n_skipped_existing']} skipped (existing), "
          f"{summary['n_skipped_no_rankings']} skipped (no data)")
    print(f"  Families — sec_13f: {summary['families']['sec_13f']}, "
          f"ctgov: {summary['families']['ctgov']}, "
          f"sec_8k: {summary['families']['sec_8k']}")


if __name__ == "__main__":
    main()
