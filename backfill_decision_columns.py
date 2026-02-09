#!/usr/bin/env python3
"""
Backfill Decision Engine columns into historical archives.

Reads each .tar.gz archive, reconstructs minimal rec dicts from
rankings.csv + archive input JSONs, calls compute_decision_fields(),
and writes enriched rankings.csv back into the archive.

Fields available from archive inputs:
  - severity, confidence_overall, archetype  → rankings.csv
  - clinical_optionality_pct_dev             → rankings.csv (PIT-correct)
  - catalyst_decay (days_to_catalyst)        → catalyst_events_*.json
  - smart_money_signal (holders)             → holdings_detailed.json
  - defensive_features (vol_60d)             → universe.json

Fields NOT available (set to None / missing):
  - coinvest tier1_count                     → blank
  - drawdown, rsi_14d, beta_xbi_60d         → blank (only vol_60d available)
  - fundamental_red_flag                     → False (SEV3 gate still works)
  - momentum alpha_60d                       → not directly stored; set neutral

Net result: Tier differentiation (A/B/C/D) works because it depends on
optionality + catalyst — both available. Sizing/risk overlays are partially
degraded (missing drawdown, rsi, beta_xbi, tier1_count).
"""
from __future__ import annotations

import csv
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from decision_engine import DECISION_COLUMNS, compute_decision_fields


ARCHIVE_DIR = PROJECT_ROOT / "data" / "archives"


# =============================================================================
# ENRICHMENT: Build rec dicts from archive inputs
# =============================================================================

def _load_catalyst_map(inputs_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Build {ticker: {days_to_catalyst, in_optimal_window}} from catalyst_events JSON."""
    catalyst_map: Dict[str, Dict[str, Any]] = {}

    # Find catalyst_events_*.json (filename includes date)
    cat_files = list(inputs_dir.glob("catalyst_events_*.json"))
    if not cat_files:
        return catalyst_map

    with open(cat_files[0], "r") as f:
        data = json.load(f)

    for summary in data.get("summaries", []):
        ticker = summary.get("ticker")
        if not ticker:
            continue
        npd = summary.get("nearest_positive_days")
        if npd is not None and npd > 0:
            catalyst_map[ticker] = {
                "days_to_catalyst": int(npd),
                "in_optimal_window": 14 <= npd <= 60,
            }
        elif npd == 0:
            # Zero days = blended proximity mode (not "no catalyst")
            catalyst_map[ticker] = {
                "days_to_catalyst": 0,
                "in_optimal_window": True,
            }
        else:
            # None / absent = no catalyst data → leave out of map
            pass

    return catalyst_map


def _load_holdings_map(inputs_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Build {ticker: smart_money_signal-like dict} from holdings_detailed.json."""
    holdings_map: Dict[str, Dict[str, Any]] = {}

    holdings_path = inputs_dir / "holdings_detailed.json"
    if not holdings_path.exists():
        return holdings_map

    with open(holdings_path, "r") as f:
        data = json.load(f)

    tickers_data = data.get("tickers", {})
    for ticker, entry in tickers_data.items():
        holdings = entry.get("holdings", {})
        current = holdings.get("current", {})
        prior = holdings.get("prior", {})

        current_ids = set(current.keys())
        prior_ids = set(prior.keys())

        # Increasing = in current but not in prior (new holders)
        # Decreasing = in prior but not in current (dropped holders)
        # Overlap = in both
        increasing = list(current_ids - prior_ids)
        decreasing = list(prior_ids - current_ids)
        overlap = current_ids & prior_ids

        holders_map: Dict[str, Any] = {
            "tier1_holders": len(current_ids),
            "holders_increasing": increasing,
            "holders_decreasing": decreasing,
            "overlap_count": len(overlap),
        }
        holdings_map[ticker] = holders_map

    return holdings_map


def _load_defensive_map(inputs_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Build {ticker: {vol_60d, ...}} from universe.json defensive_features."""
    defensive_map: Dict[str, Dict[str, Any]] = {}

    universe_path = inputs_dir / "universe.json"
    if not universe_path.exists():
        return defensive_map

    with open(universe_path, "r") as f:
        data = json.load(f)

    for item in data:
        ticker = item.get("ticker")
        if not ticker:
            continue
        df = item.get("defensive_features")
        if df:
            defensive_map[ticker] = df

    return defensive_map


def build_rec(
    row: Dict[str, str],
    catalyst_map: Dict[str, Dict[str, Any]],
    holdings_map: Dict[str, Dict[str, Any]],
    defensive_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Reconstruct a minimal rec dict from CSV row + enrichment maps."""
    ticker = row.get("ticker", "").strip().upper()
    rec: Dict[str, Any] = {}

    # Severity and confidence from CSV
    rec["severity"] = row.get("severity", "").strip()
    conf_str = row.get("confidence_overall", "").strip()
    rec["confidence_overall"] = float(conf_str) if conf_str else None

    # fundamental_red_flag: not available in archives, set False
    # (SEV3 gate in eligibility still works via severity field)
    rec["fundamental_red_flag"] = False

    # Catalyst from archive input
    if ticker in catalyst_map:
        rec["catalyst_decay"] = catalyst_map[ticker]
    else:
        rec["catalyst_decay"] = {}

    # Holdings from archive input → smart_money_signal
    if ticker in holdings_map:
        rec["smart_money_signal"] = holdings_map[ticker]
    else:
        rec["smart_money_signal"] = {}

    # coinvest: not available in archive → leave empty so tier1_count = blank
    rec["coinvest"] = {}

    # Defensive features from universe.json
    if ticker in defensive_map:
        rec["defensive_features"] = defensive_map[ticker]
    else:
        rec["defensive_features"] = {}

    # Momentum: not directly available from archive inputs.
    # alpha_60d not stored. Set empty so mom_state = "neutral".
    rec["score_breakdown"] = {}
    rec["momentum_signal"] = {}

    return rec


# =============================================================================
# CORE: Process a single archive
# =============================================================================

def process_archive(tar_path: Path, dry_run: bool = False) -> Dict[str, Any]:
    """Backfill decision columns into one archive.

    Returns summary dict with counts and any errors.
    """
    date_str = tar_path.name.replace(".tar.gz", "")  # e.g. "2026-02-07"
    result: Dict[str, Any] = {"date": date_str, "status": "ok", "n_rows": 0, "n_enriched": 0}

    # Extract to temp dir
    tmp_dir = tempfile.mkdtemp(prefix=f"backfill_{date_str}_")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(tmp_dir)

        archive_root = Path(tmp_dir) / date_str
        if not archive_root.exists():
            # Try to find extracted dir
            subdirs = [d for d in Path(tmp_dir).iterdir() if d.is_dir()]
            if subdirs:
                archive_root = subdirs[0]
            else:
                result["status"] = "error"
                result["error"] = "no archive root dir found"
                return result

        csv_path = archive_root / "rankings.csv"
        inputs_dir = archive_root / "inputs"

        if not csv_path.exists():
            result["status"] = "error"
            result["error"] = "rankings.csv not found"
            return result

        # Load enrichment maps from input JSONs
        catalyst_map = _load_catalyst_map(inputs_dir) if inputs_dir.exists() else {}
        holdings_map = _load_holdings_map(inputs_dir) if inputs_dir.exists() else {}
        defensive_map = _load_defensive_map(inputs_dir) if inputs_dir.exists() else {}

        # Read existing CSV
        with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            existing_headers = list(reader.fieldnames or [])
            rows = list(reader)

        result["n_rows"] = len(rows)

        # Build output headers: existing + decision columns (avoiding duplicates)
        output_headers = list(existing_headers)
        for col in DECISION_COLUMNS:
            if col not in output_headers:
                output_headers.append(col)

        # Process each row
        enriched_rows: List[Dict[str, str]] = []
        tier_counts: Dict[str, int] = {}
        for row in rows:
            ticker = row.get("ticker", "").strip().upper()
            archetype = row.get("archetype", "").strip()

            # Get optionality from CSV (already PIT-correct)
            opt_str = row.get("clinical_optionality_pct_dev", "").strip()
            optionality: Optional[float] = None
            if opt_str:
                try:
                    optionality = float(opt_str)
                except ValueError:
                    pass

            # Build minimal rec dict
            rec = build_rec(row, catalyst_map, holdings_map, defensive_map)

            # Call decision engine
            fields = compute_decision_fields(rec, archetype, optionality)

            # Merge decision fields into row
            out_row = dict(row)
            for col in DECISION_COLUMNS:
                out_row[col] = str(fields.get(col, ""))

            enriched_rows.append(out_row)

            # Track tier distribution
            tier = fields.get("tier_dev", "")
            if tier:
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
                result["n_enriched"] = result.get("n_enriched", 0) + 1

        result["tier_counts"] = tier_counts

        if dry_run:
            result["status"] = "dry_run"
            return result

        # Write enriched CSV
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=output_headers)
            writer.writeheader()
            writer.writerows(enriched_rows)

        # Repack tar.gz
        new_tar_path = tar_path.with_suffix(".tar.gz.tmp")
        with tarfile.open(new_tar_path, "w:gz") as tar:
            tar.add(str(archive_root), arcname=date_str)

        # Replace original
        shutil.move(str(new_tar_path), str(tar_path))

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Backfill decision engine columns into archives")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and compute but don't modify archives")
    parser.add_argument("--archive", type=str, default=None,
                        help="Process a single archive date (e.g. 2026-02-07)")
    args = parser.parse_args()

    if not ARCHIVE_DIR.exists():
        print(f"ERROR: Archive directory not found: {ARCHIVE_DIR}")
        sys.exit(1)

    archives = sorted(ARCHIVE_DIR.glob("*.tar.gz"))
    if args.archive:
        archives = [a for a in archives if a.name.replace(".tar.gz", "") == args.archive]
        if not archives:
            print(f"ERROR: Archive not found for date: {args.archive}")
            sys.exit(1)

    print(f"Backfill Decision Engine v1 into {len(archives)} archives")
    if args.dry_run:
        print("  (dry run — no files will be modified)")
    print()

    results: List[Dict[str, Any]] = []
    for tar_path in archives:
        print(f"  Processing {tar_path.name} ...", end=" ", flush=True)
        r = process_archive(tar_path, dry_run=args.dry_run)
        results.append(r)

        tier_str = ", ".join(f"{k}={v}" for k, v in sorted(r.get("tier_counts", {}).items()))
        if r["status"] == "ok":
            print(f"OK ({r['n_rows']} rows, {r['n_enriched']} dev-tiered: {tier_str})")
        elif r["status"] == "dry_run":
            print(f"DRY ({r['n_rows']} rows, {r['n_enriched']} dev-tiered: {tier_str})")
        else:
            print(f"ERROR: {r.get('error', 'unknown')}")

    # Summary
    ok_count = sum(1 for r in results if r["status"] in ("ok", "dry_run"))
    err_count = sum(1 for r in results if r["status"] == "error")
    total_enriched = sum(r.get("n_enriched", 0) for r in results)
    print()
    print(f"Done: {ok_count}/{len(results)} archives processed, "
          f"{err_count} errors, {total_enriched} total dev-tiered rows")


if __name__ == "__main__":
    main()
