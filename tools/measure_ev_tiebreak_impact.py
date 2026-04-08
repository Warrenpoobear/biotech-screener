#!/usr/bin/env python3
"""Stage 1 Event EV Tiebreak Impact Measurement (Spec 062).

Reads a rankings.csv snapshot and measures the real-world impact of
enabling Stage 1 (tiebreaker) of the Event EV promotion ladder:

  1. How often do genuine ties occur in the sort key?
  2. How often does EV data change the ordering within ties?
  3. Does the reordering change top-30 membership?
  4. Which specific names are affected?

Produces a structured JSON artifact for daily trend tracking.

Usage:
    python tools/measure_ev_tiebreak_impact.py --as-of 2026-04-08
    python tools/measure_ev_tiebreak_impact.py  # latest snapshot
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from decision_engine import (
    DEFAULT_RULESET,
    DecisionRuleset,
    _safe_float,
    compute_actionable_sort_key,
)

logger = logging.getLogger("measure_ev_tiebreak")

SNAPSHOT_DIR = REPO_ROOT / "data" / "snapshots"


def _find_latest_snapshot() -> Optional[str]:
    if not SNAPSHOT_DIR.exists():
        return None
    dates = sorted(
        [d.name for d in SNAPSHOT_DIR.iterdir() if d.is_dir() and len(d.name) == 10],
        reverse=True,
    )
    return dates[0] if dates else None


def _sort_key_for_row(row: Dict[str, Any], ruleset: DecisionRuleset) -> Tuple:
    """Compute the sort key for a row using the given ruleset."""
    return compute_actionable_sort_key(
        decision_fields=row,
        archetype=row.get("archetype", ""),
        optionality=_safe_float(row.get("clinical_optionality_pct_dev")),
        composite_rank=row.get("composite_rank"),
        ticker=row.get("ticker", ""),
        catalyst_event_type=row.get("catalyst_event_type", ""),
        catalyst_source=row.get("catalyst_source", ""),
        ruleset=ruleset,
        tiebreaker_pct=(
            _safe_float(row.get("alpha_cohort_pct"))
            if ruleset.sort_anchor == "alpha_cohort"
            else (
                _safe_float(row.get("commercial_quality_pct"))
                if row.get("archetype", "").startswith("commercial_")
                else _safe_float(row.get("clinical_optionality_pct_dev"))
            )
        ),
        alpha_raw=_safe_float(row.get("alpha_cohort_raw")),
    )


def measure_impact(
    as_of_date: str,
    top_k: int = 30,
    base_ruleset_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Measure the impact of Stage 1 tiebreaker on a snapshot.

    Compares sort ordering with event_ev_stage="off" vs "tiebreaker".

    Returns:
        Dict with tie_groups, reordered names, top-K changes, etc.
    """
    rankings_path = SNAPSHOT_DIR / as_of_date / "rankings.csv"
    if not rankings_path.exists():
        return {"error": f"No rankings.csv at {rankings_path}", "as_of_date": as_of_date}

    # Load rows
    with open(rankings_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return {"error": "Empty rankings.csv", "as_of_date": as_of_date}

    # Load base ruleset (production active)
    if base_ruleset_path:
        base_rs = DecisionRuleset.from_json(base_ruleset_path)
    else:
        base_rs = DecisionRuleset.from_json(
            str(REPO_ROOT / "production_data" / "decision_rulesets" / "v1.13.0_a4_selector_ranker.json")
        )

    # Build candidate with tiebreaker on
    from dataclasses import fields as dc_fields

    d = {f.name: getattr(base_rs, f.name) for f in dc_fields(base_rs)}
    d["event_ev_stage"] = "tiebreaker"
    candidate_rs = DecisionRuleset(**d)

    # Compute sort keys under both rulesets
    # For the OFF ruleset, we need to ensure event_ev_stage is off
    d_off = dict(d)
    d_off["event_ev_stage"] = "off"
    off_rs = DecisionRuleset(**d_off)

    off_order = []
    on_order = []

    for row in rows:
        ticker = row.get("ticker", "")
        try:
            key_off = _sort_key_for_row(row, off_rs)
            key_on = _sort_key_for_row(row, candidate_rs)
        except Exception:
            continue
        off_order.append((ticker, key_off))
        on_order.append((ticker, key_on))

    off_sorted = sorted(off_order, key=lambda x: x[1])
    on_sorted = sorted(on_order, key=lambda x: x[1])

    off_ranking = [t for t, _ in off_sorted]
    on_ranking = [t for t, _ in on_sorted]

    # --- Measure ties in the OFF ordering ---
    # A "tie" = two adjacent names sharing the same sort key prefix
    # (everything except the ticker tiebreak at the end)
    tie_groups: List[List[str]] = []
    current_group: List[str] = []
    prev_key_prefix = None

    for ticker, key in off_sorted:
        # Strip ticker (last element) to get the prefix
        key_prefix = key[:-1]
        if key_prefix == prev_key_prefix:
            if not current_group:
                # Add the previous ticker too
                current_group.append(off_sorted[off_sorted.index((ticker, key)) - 1][0])
            current_group.append(ticker)
        else:
            if current_group:
                tie_groups.append(current_group)
            current_group = []
        prev_key_prefix = key_prefix

    if current_group:
        tie_groups.append(current_group)

    # Deduplicate tie groups (the adjacency method above can miss non-adjacent ties)
    # Re-do with groupby on key prefix
    from itertools import groupby

    tie_groups = []
    for _prefix, group_iter in groupby(off_sorted, key=lambda x: x[1][:-1]):
        group = [t for t, _ in group_iter]
        if len(group) > 1:
            tie_groups.append(group)

    n_ties = len(tie_groups)
    n_tied_names = sum(len(g) for g in tie_groups)

    # --- Measure reordering ---
    # Names that changed position between off and on
    reordered = []
    for i, (t_off, t_on) in enumerate(zip(off_ranking, on_ranking)):
        if t_off != t_on:
            reordered.append({
                "rank": i + 1,
                "off": t_off,
                "on": t_on,
            })

    # --- Top-K changes ---
    off_top_k = set(off_ranking[:top_k])
    on_top_k = set(on_ranking[:top_k])
    entered = on_top_k - off_top_k
    exited = off_top_k - on_top_k

    # --- EV data coverage in ties ---
    ev_in_ties = 0
    for group in tie_groups:
        for t in group:
            row_match = next((r for r in rows if r.get("ticker") == t), None)
            if row_match and row_match.get("event_ev_score", "") not in ("", None):
                ev_in_ties += 1

    result = {
        "as_of_date": as_of_date,
        "n_total": len(rows),
        "n_eligible": len(off_ranking),
        "top_k": top_k,
        "ties": {
            "n_tie_groups": n_ties,
            "n_tied_names": n_tied_names,
            "tie_groups": [g[:5] for g in tie_groups[:10]],  # truncate for readability
            "ev_coverage_in_ties": ev_in_ties,
        },
        "reordering": {
            "n_reordered": len(reordered),
            "reordered_ranks": reordered[:20],  # first 20
        },
        "top_k_changes": {
            "entered": sorted(entered),
            "exited": sorted(exited),
            "n_changed": len(entered),
        },
        "summary": (
            f"{n_ties} tie groups ({n_tied_names} names), "
            f"{len(reordered)} positions reordered, "
            f"{len(entered)} top-{top_k} changes"
        ),
    }

    return result


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Measure Stage 1 tiebreak impact")
    parser.add_argument("--as-of", default=None, help="Snapshot date (default: latest)")
    parser.add_argument("--top-k", type=int, default=30, help="Top-K cohort size")
    parser.add_argument("--base-ruleset", default=None, help="Path to base ruleset JSON")
    args = parser.parse_args()

    as_of = args.as_of or _find_latest_snapshot()
    if not as_of:
        print("No snapshot found and no --as-of provided")
        sys.exit(1)

    result = measure_impact(as_of, top_k=args.top_k, base_ruleset_path=args.base_ruleset)

    if "error" in result:
        print(f"Error: {result['error']}")
        sys.exit(1)

    print(json.dumps(result, indent=2))

    # Write artifact
    out_dir = REPO_ROOT / "artifacts" / "event_ev"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{as_of}_tiebreak_impact.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
