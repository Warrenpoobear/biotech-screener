#!/usr/bin/env python3
"""Catalyst Coverage Decomposition — diagnostic script.

For each ticker in the ranked universe, classify WHY it has or lacks a
catalyst signal. Decomposes the `no_upcoming` bucket into deterministic
sub-buckets so we can target the biggest coverage hole.

Buckets:
  OK      — Has catalyst signal (specific_days or blended_window)
  A       — No trials in CT.gov snapshot at all
  B       — Trials exist but none are INTERVENTIONAL
  C       — All interventional trials are terminal (Withdrawn/Terminated/Suspended)
  D       — Active interventional trials but ALL PCDs in the past
  E       — Active trials, all forward PCDs > max_window (270d)
  F       — Active trials, mix of past + far (>270d) PCDs
  G       — Active trials with near PCD — should have catalyst (investigate)
  MISSING — catalyst_mode == "missing" (no catalyst data at all)

Usage:
    python3 scripts/diag_catalyst_coverage.py \
        --snapshot-dir data/snapshots/2026-02-16 \
        --trial-records cache/ctgov/trial_records_2026-02-16.json \
        [--output-dir data/diag] \
        [--max-window 270]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TERMINAL_NEGATIVE_STATUSES = frozenset({"WITHDRAWN", "TERMINATED", "SUSPENDED"})
DEFAULT_MAX_WINDOW = 270
BUCKET_LABELS = {
    "OK": "Has catalyst signal",
    "A": "No trials in snapshot",
    "B": "Trials but none interventional",
    "C": "All interventional trials terminal (W/T/S)",
    "D": "Active trials, all PCDs past",
    "E": "Active trials, all PCDs >max_window",
    "F": "Active trials, mix past + far PCDs",
    "G": "Active trials, near PCD exists (investigate)",
    "MISSING": "No catalyst data at all",
}


# ---------------------------------------------------------------------------
# Core decomposition
# ---------------------------------------------------------------------------
def _build_ticker_trial_info(
    trials: List[dict],
    tickers: set,
    as_of: date,
    max_window: int,
) -> Dict[str, dict]:
    """Build per-ticker trial summary with status-aware PCD classification."""
    info: Dict[str, dict] = {}
    for tk in tickers:
        info[tk] = {
            "n_trials": 0,
            "n_interventional": 0,
            "n_active_interventional": 0,
            "near_pcds_active": [],
            "past_pcds_active": [],
            "far_pcds_active": [],
            "min_near_days": None,
            "min_far_days": None,
        }

    for t in trials:
        tk = t.get("ticker", "")
        if tk not in info:
            continue
        rec = info[tk]
        rec["n_trials"] += 1

        stype = (t.get("study_type") or "").upper()
        status = (t.get("status") or "?").upper()
        is_terminal = status in TERMINAL_NEGATIVE_STATUSES

        if "INTERVENTIONAL" not in stype:
            continue
        rec["n_interventional"] += 1
        if not is_terminal:
            rec["n_active_interventional"] += 1

        pcd = t.get("primary_completion_date")
        if not pcd:
            continue
        try:
            pcd_date = date.fromisoformat(pcd[:10])
        except (ValueError, TypeError):
            continue

        days = (pcd_date - as_of).days
        if is_terminal:
            continue  # don't count terminal trials for PCD bucketing

        if days <= 0:
            rec["past_pcds_active"].append(days)
        elif days <= max_window:
            rec["near_pcds_active"].append(days)
            if rec["min_near_days"] is None or days < rec["min_near_days"]:
                rec["min_near_days"] = days
        else:
            rec["far_pcds_active"].append(days)
            if rec["min_far_days"] is None or days < rec["min_far_days"]:
                rec["min_far_days"] = days

    return info


def classify_ticker(
    tk: str,
    catalyst_mode: str,
    trial_info: dict,
) -> str:
    """Classify a ticker into a coverage bucket."""
    if catalyst_mode in ("specific_days", "blended_window"):
        return "OK"
    if catalyst_mode == "missing":
        return "MISSING"

    # catalyst_mode == "no_upcoming" — decompose why
    ti = trial_info
    if ti["n_trials"] == 0:
        return "A"
    if ti["n_interventional"] == 0:
        return "B"
    if ti["n_active_interventional"] == 0:
        return "C"
    if ti["near_pcds_active"]:
        return "G"  # should have catalyst — investigate
    if ti["past_pcds_active"] and ti["far_pcds_active"]:
        return "F"
    if ti["far_pcds_active"]:
        return "E"
    if ti["past_pcds_active"]:
        return "D"
    return "A"  # fallback: no PCD data at all


def decompose_snapshot(
    rankings_csv: Path,
    trial_records_json: Path,
    as_of: date,
    max_window: int = DEFAULT_MAX_WINDOW,
) -> Tuple[Dict[str, List[dict]], Dict[str, dict]]:
    """
    Decompose a snapshot into coverage buckets.

    Returns:
        (buckets, row_by_ticker)
        buckets: {bucket_code: [row_dicts]}
        row_by_ticker: {ticker: row_dict}
    """
    with open(rankings_csv) as f:
        rows = list(csv.DictReader(f))

    with open(trial_records_json) as f:
        trials = json.load(f)

    all_tickers = {r["ticker"] for r in rows}
    row_by_ticker = {r["ticker"]: r for r in rows}

    trial_info = _build_ticker_trial_info(trials, all_tickers, as_of, max_window)

    buckets: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        tk = r["ticker"]
        mode = r.get("catalyst_mode", "missing")
        bucket = classify_ticker(tk, mode, trial_info[tk])
        r["_bucket"] = bucket
        r["_trial_info"] = trial_info[tk]
        buckets[bucket].append(r)

    return dict(buckets), row_by_ticker


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _arch_summary(rows: List[dict]) -> str:
    c = Counter(r.get("archetype", "?") for r in rows)
    return ", ".join(f"{a}:{n}" for a, n in c.most_common())


def _tier_summary(rows: List[dict]) -> str:
    c = Counter()
    for r in rows:
        tier = r.get("tier_any") or r.get("tier_dev") or ""
        if tier:
            c[tier] += 1
    if not c:
        return "none"
    return ", ".join(f"{t}:{n}" for t, n in c.most_common())


def print_report(
    buckets: Dict[str, List[dict]],
    total: int,
    as_of: date,
    max_window: int,
    output_path: Optional[Path] = None,
) -> str:
    """Generate the markdown report."""
    lines: List[str] = []
    w = lines.append

    w(f"# Catalyst Coverage Decomposition — {as_of}")
    w(f"")
    w(f"Universe: {total} ranked tickers | Max catalyst window: {max_window}d")
    w("")

    # Summary table
    w("| Bucket | Label | N | % | Archetypes |")
    w("|--------|-------|---|---|------------|")
    for key in ["OK", "A", "B", "C", "D", "E", "F", "G", "MISSING"]:
        rows = buckets.get(key, [])
        n = len(rows)
        pct = n / total * 100 if total else 0
        label = BUCKET_LABELS[key]
        arch = _arch_summary(rows) if rows else ""
        w(f"| {key} | {label} | {n} | {pct:.1f}% | {arch} |")

    # No-catalyst subtotal
    no_cat = sum(len(buckets.get(k, [])) for k in "ABCDEFG")
    no_cat_pct = no_cat / total * 100 if total else 0
    w("")
    w(f"**No catalyst total: {no_cat} ({no_cat_pct:.1f}%)**")

    # Addressable gap: D+E+F have trials but dates outside window
    addressable = sum(len(buckets.get(k, [])) for k in "DEF")
    addr_pct = addressable / total * 100 if total else 0
    w(f"**Addressable gap (D+E+F): {addressable} ({addr_pct:.1f}%)** — "
      f"have active trials but no PCD within {max_window}d")

    # Detail per bucket (non-OK)
    for key in ["A", "B", "C", "D", "E", "F", "G", "MISSING"]:
        rows = buckets.get(key, [])
        if not rows:
            continue
        w("")
        w(f"## Bucket {key}: {BUCKET_LABELS[key]} ({len(rows)})")
        w(f"Archetypes: {_arch_summary(rows)}")
        w(f"Tiers: {_tier_summary(rows)}")

        # Top tickers by composite_rank
        sorted_rows = sorted(rows, key=lambda r: int(r.get("composite_rank", 999)))
        w("")
        w("| Rank | Ticker | Arch | Tier | Score | Detail |")
        w("|------|--------|------|------|-------|--------|")
        for r in sorted_rows[:15]:
            rank = r.get("composite_rank", "?")
            tk = r["ticker"]
            arch = r.get("archetype", "?")[:15]
            tier = r.get("tier_any") or r.get("tier_dev") or ""
            score = r.get("composite_score", "?")
            ti = r.get("_trial_info", {})

            if key in ("E", "F"):
                detail = f"min_far={ti.get('min_far_days', '?')}d"
            elif key == "D":
                n_past = len(ti.get("past_pcds_active", []))
                detail = f"{n_past} past PCDs"
            elif key == "G":
                detail = f"near={ti.get('min_near_days', '?')}d"
            else:
                detail = f"{ti.get('n_trials', 0)} trials"
            w(f"| {rank} | {tk} | {arch} | {tier} | {score} | {detail} |")
        if len(sorted_rows) > 15:
            w(f"| ... | +{len(sorted_rows)-15} more | | | | |")

    report = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report + "\n")
        print(f"Report written to {output_path}", file=sys.stderr)

    return report


def write_detail_csv(
    buckets: Dict[str, List[dict]],
    output_path: Path,
) -> None:
    """Write per-ticker CSV with bucket assignment."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ticker", "composite_rank", "composite_score", "archetype",
        "catalyst_mode", "catalyst_days", "tier_dev", "tier_commercial",
        "tier_any", "coverage_bucket", "n_trials", "n_active_interventional",
        "n_near_pcds", "n_past_pcds", "n_far_pcds", "min_near_days", "min_far_days",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        all_rows = []
        for bucket_code, rows in buckets.items():
            for r in rows:
                ti = r.get("_trial_info", {})
                all_rows.append({
                    "ticker": r["ticker"],
                    "composite_rank": r.get("composite_rank", ""),
                    "composite_score": r.get("composite_score", ""),
                    "archetype": r.get("archetype", ""),
                    "catalyst_mode": r.get("catalyst_mode", ""),
                    "catalyst_days": r.get("catalyst_days", ""),
                    "tier_dev": r.get("tier_dev", ""),
                    "tier_commercial": r.get("tier_commercial", ""),
                    "tier_any": r.get("tier_any", ""),
                    "coverage_bucket": bucket_code,
                    "n_trials": ti.get("n_trials", 0),
                    "n_active_interventional": ti.get("n_active_interventional", 0),
                    "n_near_pcds": len(ti.get("near_pcds_active", [])),
                    "n_past_pcds": len(ti.get("past_pcds_active", [])),
                    "n_far_pcds": len(ti.get("far_pcds_active", [])),
                    "min_near_days": ti.get("min_near_days", ""),
                    "min_far_days": ti.get("min_far_days", ""),
                })
        all_rows.sort(key=lambda r: int(r["composite_rank"]) if r["composite_rank"] else 999)
        writer.writerows(all_rows)
    print(f"Detail CSV written to {output_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_date(s: str) -> date:
    """Parse YYYY-MM-DD date string, or infer from snapshot dir name."""
    return date.fromisoformat(s)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Catalyst coverage decomposition diagnostic",
    )
    parser.add_argument(
        "--snapshot-dir", type=Path, required=True,
        help="Path to snapshot directory (contains rankings.csv)",
    )
    parser.add_argument(
        "--trial-records", type=Path, required=True,
        help="Path to trial_records JSON (from ctgov cache)",
    )
    parser.add_argument(
        "--as-of-date", type=str, default=None,
        help="Analysis date (YYYY-MM-DD). Default: infer from snapshot dir name.",
    )
    parser.add_argument(
        "--max-window", type=int, default=DEFAULT_MAX_WINDOW,
        help=f"Max catalyst detection window in days (default: {DEFAULT_MAX_WINDOW})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for report + CSV (default: print to stdout only)",
    )
    args = parser.parse_args()

    # Resolve paths
    rankings_csv = args.snapshot_dir / "rankings.csv"
    if not rankings_csv.exists():
        print(f"ERROR: {rankings_csv} not found", file=sys.stderr)
        sys.exit(1)
    if not args.trial_records.exists():
        print(f"ERROR: {args.trial_records} not found", file=sys.stderr)
        sys.exit(1)

    # Infer as_of_date
    if args.as_of_date:
        as_of = _parse_date(args.as_of_date)
    else:
        # Try to infer from directory name
        dir_name = args.snapshot_dir.name
        try:
            as_of = _parse_date(dir_name)
        except ValueError:
            print(
                "ERROR: Cannot infer as_of_date from directory name. "
                "Use --as-of-date YYYY-MM-DD.",
                file=sys.stderr,
            )
            sys.exit(1)

    # Run decomposition
    buckets, row_by_ticker = decompose_snapshot(
        rankings_csv, args.trial_records, as_of, args.max_window,
    )
    total = sum(len(v) for v in buckets.values())

    # Report
    report = print_report(buckets, total, as_of, args.max_window,
                          output_path=args.output_dir / "catalyst_coverage.md" if args.output_dir else None)

    if not args.output_dir:
        print(report)
    else:
        write_detail_csv(buckets, args.output_dir / "catalyst_coverage_detail.csv")

    # Summary to stdout
    ok_n = len(buckets.get("OK", []))
    no_cat = total - ok_n
    addr = sum(len(buckets.get(k, [])) for k in "DEF")
    print(f"\nCoverage: {ok_n}/{total} ({ok_n/total*100:.1f}%) have catalyst signal")
    print(f"No catalyst: {no_cat} ({no_cat/total*100:.1f}%)")
    print(f"Addressable gap (D+E+F): {addr} ({addr/total*100:.1f}%)")


if __name__ == "__main__":
    main()
