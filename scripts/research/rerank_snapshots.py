#!/usr/bin/env python3
"""Re-rank historical snapshots through a target ruleset.

RESEARCH ONLY — not for production use.

Inputs:  snapshot-root with rankings.csv per date, ruleset JSON
Outputs: data/snapshots_reranked/{date}/rankings.csv + metadata.json

Reads each snapshot's rankings.csv, backfills missing columns with safe
defaults, re-computes the sort key using the target ruleset, assigns fresh
actionable_rank 1..N, and writes results to a staging directory.

This allows forward-return evaluation on a consistent ranking model
across dates that were originally produced with different rulesets.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.ranking_utils import backfill_columns, safe_float as _safe_float
from decision_engine import DecisionRuleset, compute_actionable_sort_key


# ---------------------------------------------------------------------------
# Re-ranking
# ---------------------------------------------------------------------------

def rerank(rows: List[Dict[str, str]], ruleset: DecisionRuleset) -> List[Dict[str, str]]:
    """Re-sort rows using the target ruleset and assign fresh actionable_rank."""
    backfill_columns(rows)

    # Sort using production-identical logic
    rows.sort(key=lambda r: compute_actionable_sort_key(
        decision_fields=r,
        archetype=r.get("archetype", ""),
        optionality=_safe_float(r.get("clinical_optionality_pct_dev")),
        composite_rank=r.get("composite_rank"),
        ticker=r.get("ticker", ""),
        catalyst_event_type=r.get("catalyst_event_type", ""),
        catalyst_source=r.get("catalyst_source", ""),
        ruleset=ruleset,
        tiebreaker_pct=(
            _safe_float(r.get("alpha_cohort_pct"))
            if ruleset.sort_anchor == "alpha_cohort"
            else (_safe_float(r.get("commercial_quality_pct"))
                  if r.get("archetype", "").startswith("commercial_")
                  else _safe_float(r.get("clinical_optionality_pct_dev")))
        ),
    ))

    # Assign actionable_rank
    rank = 1
    for r in rows:
        if r.get("eligible") == "1":
            r["actionable_rank"] = str(rank)
            rank += 1
        else:
            r["actionable_rank"] = ""

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-rank historical snapshots through a target ruleset",
    )
    parser.add_argument(
        "--snapshot-root", type=Path,
        default=PROJECT_ROOT / "data" / "snapshots",
    )
    parser.add_argument(
        "--out-root", type=Path,
        default=PROJECT_ROOT / "data" / "snapshots_reranked",
    )
    parser.add_argument(
        "--ruleset", type=Path, default=None,
        help="Path to ruleset JSON. Default: latest snapshot's decision_ruleset.json",
    )
    parser.add_argument("--date-from", type=str, default=None)
    parser.add_argument("--date-to", type=str, default=None)
    parser.add_argument(
        "--min-cols", type=int, default=50,
        help="Skip snapshots with fewer than this many columns (default: 50)",
    )
    args = parser.parse_args()

    # Load ruleset
    if args.ruleset:
        ruleset = DecisionRuleset.from_json(str(args.ruleset))
    else:
        # Find latest snapshot with decision_ruleset.json
        dates = sorted([p.name for p in args.snapshot_root.iterdir()
                        if p.is_dir() and len(p.name) == 10 and p.name[4] == "-"
                        and (p / "decision_ruleset.json").exists()])
        if not dates:
            print("ERROR: No snapshot with decision_ruleset.json found")
            sys.exit(1)
        rs_path = args.snapshot_root / dates[-1] / "decision_ruleset.json"
        ruleset = DecisionRuleset.from_json(str(rs_path))
        print(f"Using ruleset from {rs_path} (ID: {ruleset.ruleset_id})")

    # Discover snapshots
    snap_dates = sorted([
        p.name for p in args.snapshot_root.iterdir()
        if p.is_dir() and len(p.name) == 10 and p.name[4] == "-"
        and (p / "rankings.csv").exists()
    ])

    if args.date_from:
        snap_dates = [d for d in snap_dates if d >= args.date_from]
    if args.date_to:
        snap_dates = [d for d in snap_dates if d <= args.date_to]

    print(f"Re-ranking {len(snap_dates)} snapshots → {args.out_root}")

    n_ok = 0
    n_skip = 0
    for snap_date in snap_dates:
        src = args.snapshot_root / snap_date
        csv_path = src / "rankings.csv"

        # Load rankings
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames
            rows = list(reader)

        if len(cols) < args.min_cols:
            print(f"  {snap_date}: SKIP ({len(cols)} cols < {args.min_cols})")
            n_skip += 1
            continue

        # Re-rank
        rows = rerank(rows, ruleset)

        # Write output
        out_dir = args.out_root / snap_date
        out_dir.mkdir(parents=True, exist_ok=True)

        # Use the full column set from the re-ranked rows
        out_cols = list(rows[0].keys()) if rows else cols
        with open(out_dir / "rankings.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=out_cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        # Copy metadata.json (or create minimal one)
        meta_src = src / "metadata.json"
        meta = {}
        if meta_src.exists():
            with open(meta_src, encoding="utf-8") as f:
                meta = json.load(f)
        meta["as_of_date"] = snap_date
        meta["reranked_with_ruleset"] = ruleset.ruleset_id
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)

        n_eligible = sum(1 for r in rows if r.get("eligible") == "1")
        print(f"  {snap_date}: OK ({len(rows)} rows, {n_eligible} eligible)")
        n_ok += 1

    print(f"\nDone: {n_ok} re-ranked, {n_skip} skipped")


if __name__ == "__main__":
    main()
