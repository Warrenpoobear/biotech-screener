#!/usr/bin/env python3
"""A/B rerank overlap comparison between two rulesets.

RESEARCH ONLY — not for production use.

Reranks raw snapshots with both baseline and candidate rulesets,
then computes per-date top-K overlap (Jaccard) and name churn.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from decision_engine import DecisionRuleset
from scripts.research.rerank_snapshots import rerank


def _top_k_tickers(rows: List[Dict[str, str]], k: int) -> List[str]:
    """Return tickers with actionable_rank 1..k, in rank order."""
    ranked = [(int(r["actionable_rank"]), r.get("ticker", ""))
              for r in rows if r.get("actionable_rank", "").isdigit()]
    ranked.sort()
    return [t for _, t in ranked[:k]]


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _pct_overlap(a: List[str], b: List[str], k: int) -> float:
    """% of top-k names that appear in both lists."""
    sa, sb = set(a[:k]), set(b[:k])
    if not sa:
        return 1.0
    return len(sa & sb) / k


def compare_date(
    csv_path: Path,
    baseline_rs: DecisionRuleset,
    candidate_rs: DecisionRuleset,
    top_ks: List[int],
    min_cols: int,
) -> Optional[Dict]:
    """Rerank one snapshot date with both rulesets and compute overlap."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        rows = list(reader)

    if len(cols) < min_cols:
        return None

    # Deep-copy rows for independent reranking
    rows_baseline = copy.deepcopy(rows)
    rows_candidate = copy.deepcopy(rows)

    rerank(rows_baseline, baseline_rs)
    rerank(rows_candidate, candidate_rs)

    result: Dict = {}
    for k in top_ks:
        base_top = _top_k_tickers(rows_baseline, k)
        cand_top = _top_k_tickers(rows_candidate, k)

        base_set = set(base_top)
        cand_set = set(cand_top)

        entering = cand_set - base_set
        leaving = base_set - cand_set

        result[k] = {
            "jaccard": _jaccard(base_set, cand_set),
            "pct_overlap": _pct_overlap(base_top, cand_top, k),
            "entering": sorted(entering),
            "leaving": sorted(leaving),
            "churn": len(entering),
        }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A/B rerank overlap between baseline and candidate rulesets",
    )
    parser.add_argument(
        "--snapshot-root", type=Path,
        default=PROJECT_ROOT / "data" / "snapshots",
    )
    parser.add_argument(
        "--baseline-ruleset", type=Path, required=True,
        help="Baseline DecisionRuleset JSON",
    )
    parser.add_argument(
        "--candidate-ruleset", type=Path, required=True,
        help="Candidate DecisionRuleset JSON",
    )
    parser.add_argument("--top-ks", type=str, default="20,60")
    parser.add_argument("--date-from", type=str, default=None)
    parser.add_argument("--date-to", type=str, default=None)
    parser.add_argument(
        "--min-cols", type=int, default=50,
        help="Skip snapshots with fewer than this many columns",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    top_ks = [int(x.strip()) for x in args.top_ks.split(",")]

    baseline_rs = DecisionRuleset.from_json(str(args.baseline_ruleset))
    candidate_rs = DecisionRuleset.from_json(str(args.candidate_ruleset))
    print(f"Baseline:  {baseline_rs.ruleset_id}")
    print(f"Candidate: {candidate_rs.ruleset_id}")

    # Discover snapshot dates (only clean YYYY-MM-DD dirs)
    snap_dates = sorted([
        p.name for p in args.snapshot_root.iterdir()
        if p.is_dir() and len(p.name) == 10 and p.name[4] == "-"
        and (p / "rankings.csv").exists()
    ])

    if args.date_from:
        snap_dates = [d for d in snap_dates if d >= args.date_from]
    if args.date_to:
        snap_dates = [d for d in snap_dates if d <= args.date_to]

    print(f"Comparing {len(snap_dates)} snapshot dates\n")

    # Header
    hdr_parts = [f"{'Date':>12s}"]
    for k in top_ks:
        hdr_parts.append(f"Top-{k} Jac")
        hdr_parts.append(f"Top-{k} Churn")
    print("  ".join(hdr_parts))
    print("  ".join(["-" * len(p) for p in hdr_parts]))

    all_results = {}
    n_ok = 0
    n_skip = 0

    for snap_date in snap_dates:
        csv_path = args.snapshot_root / snap_date / "rankings.csv"
        result = compare_date(csv_path, baseline_rs, candidate_rs, top_ks, args.min_cols)

        if result is None:
            n_skip += 1
            continue

        all_results[snap_date] = result
        n_ok += 1

        parts = [f"{snap_date:>12s}"]
        for k in top_ks:
            r = result[k]
            parts.append(f"  {r['jaccard']:.3f}   ")
            churn_str = f"  {r['churn']:3d}     "
            if r["churn"] > 0:
                entering = ",".join(r["entering"][:3])
                leaving = ",".join(r["leaving"][:3])
                churn_str = f"  {r['churn']:3d} +{entering} -{leaving}"
            parts.append(churn_str)
        print("  ".join(parts))

    # Summary
    print(f"\n{'='*70}")
    print(f"Dates: {n_ok} compared, {n_skip} skipped\n")

    for k in top_ks:
        jaccards = [all_results[d][k]["jaccard"] for d in all_results]
        churns = [all_results[d][k]["churn"] for d in all_results]
        zero_churn = sum(1 for c in churns if c == 0)

        if jaccards:
            import statistics
            print(f"Top-{k}:")
            print(f"  Jaccard:  mean={statistics.mean(jaccards):.4f}  "
                  f"min={min(jaccards):.4f}  median={statistics.median(jaccards):.4f}")
            print(f"  Churn:    mean={statistics.mean(churns):.2f}  "
                  f"max={max(churns)}  zero_churn_dates={zero_churn}/{len(churns)}")
            print()

    # Collect all unique name changes across dates
    for k in top_ks:
        all_entering = {}
        all_leaving = {}
        for d in all_results:
            for t in all_results[d][k]["entering"]:
                all_entering[t] = all_entering.get(t, 0) + 1
            for t in all_results[d][k]["leaving"]:
                all_leaving[t] = all_leaving.get(t, 0) + 1
        if all_entering:
            print(f"Top-{k} name changes (entering candidate, leaving baseline):")
            for t, c in sorted(all_entering.items(), key=lambda x: -x[1])[:10]:
                print(f"  +{t}: {c} dates")
            for t, c in sorted(all_leaving.items(), key=lambda x: -x[1])[:10]:
                print(f"  -{t}: {c} dates")
            print()

    # Write JSON output
    if args.out:
        out_data = {
            "baseline_ruleset_id": baseline_rs.ruleset_id,
            "candidate_ruleset_id": candidate_rs.ruleset_id,
            "top_ks": top_ks,
            "n_dates": n_ok,
            "n_skipped": n_skip,
            "summary": {},
            "by_date": all_results,
        }
        for k in top_ks:
            jaccards = [all_results[d][k]["jaccard"] for d in all_results]
            churns = [all_results[d][k]["churn"] for d in all_results]
            if jaccards:
                import statistics
                out_data["summary"][f"top_{k}"] = {
                    "mean_jaccard": round(statistics.mean(jaccards), 4),
                    "min_jaccard": round(min(jaccards), 4),
                    "median_jaccard": round(statistics.median(jaccards), 4),
                    "mean_churn": round(statistics.mean(churns), 2),
                    "max_churn": max(churns),
                    "zero_churn_dates": sum(1 for c in churns if c == 0),
                }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, default=str)
        print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
