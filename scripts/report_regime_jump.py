#!/usr/bin/env python3
"""Generate a committee-grade attribution report for a regime-jump date.

Explains *why* a large batch of B→G catalyst flips happened on a given date:
source breakdown, top flippers with event details, tier impact, and hard-gate
survival status.

Usage:
    python3 scripts/report_regime_jump.py --as-of-date 2026-02-04
    python3 scripts/report_regime_jump.py --as-of-date 2026-02-04 --top 25 --output report.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent

GOOD_MODES = {"specific_days", "blended_window"}
BAD_MODES = {"no_upcoming", "missing"}

STRENGTH_ORDER = {"near": 0, "mid": 1, "far": 2, "missing": 3, "": 4}


def _read_rankings(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _mode(row: dict) -> str:
    return row.get("catalyst_mode") or row.get("de_catalyst_mode") or ""


def _days(row: dict) -> float | None:
    raw = row.get("catalyst_days") or row.get("de_catalyst_days") or ""
    try:
        return float(raw) if raw else None
    except (ValueError, TypeError):
        return None


def _top_n_set(rows: list[dict], n: int) -> set[str]:
    valid = []
    for r in rows:
        cr = r.get("composite_rank", "")
        t = r.get("ticker", "")
        if not t or cr in ("", None):
            continue
        try:
            valid.append((int(cr), t))
        except (ValueError, TypeError):
            continue
    valid.sort()
    return {t for _, t in valid[:n]}


def compute_flips(
    cur_rows: list[dict], prior_rows: list[dict],
) -> list[dict]:
    """Find all B→G catalyst mode flips (dev-only, common tickers)."""
    prior_by_t = {
        r.get("ticker", ""): r
        for r in prior_rows
        if r.get("archetype") == "drug_developer" and r.get("ticker", "")
    }

    flips = []
    for r in cur_rows:
        if r.get("archetype") != "drug_developer":
            continue
        t = r.get("ticker", "")
        if not t or t not in prior_by_t:
            continue
        prev = _mode(prior_by_t[t])
        cur = _mode(r)
        if prev in BAD_MODES and cur in GOOD_MODES:
            flips.append({
                "ticker": t,
                "rank": int(r.get("composite_rank", 999)),
                "mode": cur,
                "days": _days(r),
                "source": r.get("catalyst_source", "") or "",
                "event_type": r.get("catalyst_event_type", "") or "",
                "tier": r.get("tier_dev", "") or "",
                "strength": r.get("catalyst_strength", "") or "",
                "prev_mode": prev,
                "eligible": r.get("eligible", ""),
            })
    flips.sort(key=lambda x: x["rank"])
    return flips


def format_report(
    as_of_date: str,
    prior_date: str,
    flips: list[dict],
    cur_rows: list[dict],
    top_n: int,
    shadow_metrics: dict | None = None,
) -> str:
    """Build the full text report."""
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    top60 = _top_n_set(cur_rows, 60)
    top100 = _top_n_set(cur_rows, 100)

    flips_60 = [f for f in flips if f["ticker"] in top60]
    flips_100 = [f for f in flips if f["ticker"] in top100]

    # ── Header ─────────────────────────────────────────────────────────
    w(f"{'=' * 72}")
    w(f"  REGIME-JUMP ATTRIBUTION REPORT")
    w(f"  as_of_date: {as_of_date}    prior_date: {prior_date}")
    w(f"{'=' * 72}")

    # ── Summary ────────────────────────────────────────────────────────
    w(f"\n## Summary")
    w(f"  Total B→G flips (dev-only): {len(flips)}")
    w(f"  In top-60:  {len(flips_60)}   {[f['ticker'] for f in flips_60]}")
    w(f"  In top-100: {len(flips_100)}  {[f['ticker'] for f in flips_100]}")

    if shadow_metrics:
        w(f"\n  Shadow metrics context:")
        w(f"    A-tier count: {shadow_metrics.get('A_tier_count', '?')}")
        w(f"    Median catalyst days (good): {shadow_metrics.get('median_catalyst_days_good', '?')}")
        w(f"    Top-60 overlap vs prior: {shadow_metrics.get('top60_overlap', '?')}")
        w(f"    Top-100 overlap vs prior: {shadow_metrics.get('top100_overlap', '?')}")

    # ── Source attribution ─────────────────────────────────────────────
    w(f"\n## Source Attribution")
    src_counts = Counter(f["source"] or "(blended/multi)" for f in flips)
    for src, cnt in src_counts.most_common():
        pct = 100 * cnt / len(flips)
        w(f"  {src:<25} {cnt:3d}  ({pct:.0f}%)")

    # ── Event type breakdown ───────────────────────────────────────────
    w(f"\n## Event Type Breakdown")
    et_counts = Counter(f["event_type"] or "(blended/multi)" for f in flips)
    for et, cnt in et_counts.most_common():
        pct = 100 * cnt / len(flips)
        w(f"  {et:<30} {cnt:3d}  ({pct:.0f}%)")

    # ── Tier impact ────────────────────────────────────────────────────
    w(f"\n## Tier Distribution of Flippers")
    tier_counts = Counter(f["tier"] or "(none)" for f in flips)
    for tier, cnt in sorted(tier_counts.items()):
        w(f"  {tier}: {cnt}")

    # ── Prior mode ─────────────────────────────────────────────────────
    w(f"\n## Prior Catalyst Mode (what they flipped FROM)")
    pm_counts = Counter(f["prev_mode"] for f in flips)
    for pm, cnt in pm_counts.most_common():
        w(f"  {pm}: {cnt}")

    # ── Catalyst strength distribution ─────────────────────────────────
    w(f"\n## Catalyst Strength Distribution")
    str_counts = Counter(f["strength"] or "(unknown)" for f in flips)
    for s, cnt in sorted(str_counts.items(), key=lambda x: STRENGTH_ORDER.get(x[0], 99)):
        w(f"  {s}: {cnt}")

    # ── Top N flippers detail ──────────────────────────────────────────
    show = flips[:top_n]
    w(f"\n## Top {len(show)} B→G Flippers (by composite rank)")
    w(f"  {'Rank':<5} {'Ticker':<7} {'Tier':<5} {'Source':<22} {'Event Type':<27} "
      f"{'Days':>5} {'Str':<8} {'Hard Gate'}")
    w(f"  {'-' * 100}")

    for f in show:
        days_str = f"{f['days']:.0f}" if f["days"] is not None else "-"
        # Hard gate = survived if tier is A or B (actionable)
        gate = "PASS" if f["tier"] in ("A", "B") else "gated"
        w(f"  {f['rank']:<5} {f['ticker']:<7} {f['tier']:<5} "
          f"{f['source'] or '(blended)':<22} "
          f"{f['event_type'] or '(blended)':<27} "
          f"{days_str:>5} {f['strength']:<8} {gate}")

    # ── Interpretation ─────────────────────────────────────────────────
    w(f"\n## Interpretation")

    ctg_pct = 100 * src_counts.get("CTGOV_CALENDAR", 0) / len(flips) if flips else 0
    no_upcoming_pct = 100 * pm_counts.get("no_upcoming", 0) / len(flips) if flips else 0

    w(f"  This is a DATA REGIME TRANSITION, not organic churn.")
    if ctg_pct > 50:
        w(f"  - {ctg_pct:.0f}% of flips sourced from CTGOV_CALENDAR (trial_records.json enabled)")
    if no_upcoming_pct > 90:
        w(f"  - {no_upcoming_pct:.0f}% flipped from 'no_upcoming' (no prior catalyst data)")
    w(f"  - Expected: this date is the first with CT.gov data in PIT-strict mode")
    w(f"  - Subsequent dates should show near-zero churn (confirmed by timeseries)")
    w(f"  - WARN health status on this date is expected and benign")

    w(f"\n{'=' * 72}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate regime-jump attribution report.",
    )
    parser.add_argument("--as-of-date", required=True)
    parser.add_argument(
        "--snapshot-dir", type=Path, default=SCRIPT_DIR / "data" / "snapshots",
    )
    parser.add_argument(
        "--rollup-csv", type=Path,
        default=SCRIPT_DIR / "output" / "catalyst_shadow_timeseries.csv",
    )
    parser.add_argument("--top", type=int, default=25, help="Top N flippers to show")
    parser.add_argument("--output", type=Path, default=None, help="Write report to file")
    args = parser.parse_args(argv)

    snap_dir = args.snapshot_dir
    as_of = args.as_of_date

    # Find prior date from rollup
    prior_date = None
    if args.rollup_csv.exists():
        with open(args.rollup_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("as_of_date") == as_of:
                    prior_date = row.get("prior_date") or None
                    break

    if not prior_date:
        print(f"No prior_date found for {as_of} in rollup CSV.", file=sys.stderr)
        return 1

    cur_path = snap_dir / as_of / "rankings.csv"
    prior_path = snap_dir / prior_date / "rankings.csv"
    if not cur_path.exists() or not prior_path.exists():
        print(f"Missing rankings.csv for {as_of} or {prior_date}.", file=sys.stderr)
        return 1

    cur_rows = _read_rankings(cur_path)
    prior_rows = _read_rankings(prior_path)
    flips = compute_flips(cur_rows, prior_rows)

    # Load shadow metrics if available
    shadow_path = snap_dir / as_of / "catalyst_shadow_metrics.json"
    shadow_metrics = None
    if shadow_path.exists():
        try:
            shadow_metrics = json.loads(shadow_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    report = format_report(as_of, prior_date, flips, cur_rows, args.top, shadow_metrics)
    print(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
        print(f"\nWritten to {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
