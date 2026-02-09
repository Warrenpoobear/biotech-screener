#!/usr/bin/env python3
"""Audit drawdown data coverage.

Reads a snapshot directory and cross-references with price_history.csv to
produce a diagnostic report showing why drawdown data is missing for
specific tickers.

CLI:
  python scripts/audit_drawdown_coverage.py \
    --snapshot-dir data/snapshots/2026-02-07 \
    [--price-history production_data/price_history.csv] \
    [--output-dir /tmp/audit-test]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# ---------------------------------------------------------------------------
# Core audit logic
# ---------------------------------------------------------------------------

MAX_TICKERS_PER_BUCKET = 25


def audit_drawdown_coverage(
    rankings_path: Path,
    price_history_path: Path | None,
    snapshot_date: str | None = None,
) -> Dict[str, Any]:
    """Analyse drawdown coverage and return structured results.

    Parameters
    ----------
    rankings_path : Path
        Path to the ``rankings.csv`` inside a snapshot directory.
    price_history_path : Path | None
        Path to ``price_history.csv`` (optional — if absent, all missing
        tickers are attributed to ``no_price_series``).
    snapshot_date : str | None
        As-of date for the snapshot (``YYYY-MM-DD``). Derived from the
        parent directory name if not supplied.

    Returns
    -------
    dict
        Structured audit results suitable for JSON serialization.
    """
    rankings = pd.read_csv(rankings_path, dtype=str)

    # Restrict to drug_developer archetype
    dev = rankings[rankings["archetype"] == "drug_developer"].copy()
    total_dev = len(dev)

    # Drawdown present / missing
    dd_col = "de_drawdown"
    reason_col = "de_drawdown_missing_reason"

    dd_present_mask = dev[dd_col].apply(
        lambda v: not pd.isna(v) and str(v).strip() != ""
    ) if dd_col in dev.columns else pd.Series([False] * total_dev)
    dd_present = int(dd_present_mask.sum())
    dd_missing = total_dev - dd_present

    coverage_pct = round(dd_present / total_dev * 100, 1) if total_dev else 0.0

    # --- Missing-reason breakdown ---
    missing_reason_counts: Dict[str, int] = {}
    missing_tickers_by_reason: Dict[str, List[str]] = {}

    if reason_col in dev.columns:
        dev_missing = dev[~dd_present_mask]
        for _, row in dev_missing.iterrows():
            reason = str(row.get(reason_col, "")).strip()
            if reason == "":
                reason = "other"
            missing_reason_counts[reason] = missing_reason_counts.get(reason, 0) + 1
            missing_tickers_by_reason.setdefault(reason, []).append(
                str(row.get("ticker", ""))
            )
    else:
        # No reason column — fall back to cross-referencing price CSV
        missing_tickers = list(dev[~dd_present_mask]["ticker"])
        price_tickers: set[str] = set()
        if price_history_path and price_history_path.exists():
            with open(price_history_path, "r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for r in reader:
                    price_tickers.add((r.get("ticker") or "").upper())

        for t in missing_tickers:
            t_upper = str(t).strip().upper()
            if t_upper in price_tickers:
                reason = "series_too_short"
            else:
                reason = "no_price_series"
            missing_reason_counts[reason] = missing_reason_counts.get(reason, 0) + 1
            missing_tickers_by_reason.setdefault(reason, []).append(t_upper)

    # Cap ticker lists for readability
    for reason in missing_tickers_by_reason:
        missing_tickers_by_reason[reason] = sorted(
            missing_tickers_by_reason[reason]
        )[:MAX_TICKERS_PER_BUCKET]

    # --- Price history stats ---
    price_history_stats: Dict[str, Any] = {"available": False}
    if price_history_path and price_history_path.exists():
        ph_tickers: Dict[str, int] = {}
        date_min: str | None = None
        date_max: str | None = None
        with open(price_history_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                t = (r.get("ticker") or "").upper()
                d = r.get("date", "")
                if t:
                    ph_tickers[t] = ph_tickers.get(t, 0) + 1
                if d:
                    if date_min is None or d < date_min:
                        date_min = d
                    if date_max is None or d > date_max:
                        date_max = d

        bar_counts = list(ph_tickers.values())
        bar_counts.sort()
        median_bars = bar_counts[len(bar_counts) // 2] if bar_counts else 0

        price_history_stats = {
            "available": True,
            "total_tickers": len(ph_tickers),
            "date_range": [date_min, date_max],
            "median_bars_per_ticker": median_bars,
        }

    if snapshot_date is None:
        snapshot_date = rankings_path.parent.name

    return {
        "snapshot_date": snapshot_date,
        "total_dev_tickers": total_dev,
        "drawdown_present": dd_present,
        "drawdown_missing": dd_missing,
        "coverage_pct": coverage_pct,
        "missing_reason_counts": missing_reason_counts,
        "missing_tickers_by_reason": missing_tickers_by_reason,
        "price_history_stats": price_history_stats,
    }


def format_audit_markdown(result: Dict[str, Any]) -> str:
    """Render audit results as Markdown."""
    lines: List[str] = []
    lines.append("# Drawdown Coverage Audit")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Snapshot date | {result['snapshot_date']} |")
    lines.append(f"| Dev tickers | {result['total_dev_tickers']} |")
    lines.append(f"| Drawdown present | {result['drawdown_present']} |")
    lines.append(f"| Drawdown missing | {result['drawdown_missing']} |")
    lines.append(f"| Coverage % | {result['coverage_pct']}% |")
    lines.append("")

    mr = result.get("missing_reason_counts", {})
    if mr:
        lines.append("## Missing Reasons")
        lines.append("")
        lines.append("| Reason | Count |")
        lines.append("|---|---|")
        for reason, cnt in sorted(mr.items(), key=lambda x: -x[1]):
            lines.append(f"| {reason} | {cnt} |")
        lines.append("")

    mt = result.get("missing_tickers_by_reason", {})
    if mt:
        lines.append("## Missing Tickers (top 25 per reason)")
        lines.append("")
        for reason, tickers in sorted(mt.items()):
            lines.append(f"**{reason}** ({len(tickers)}): {', '.join(tickers)}")
            lines.append("")

    ph = result.get("price_history_stats", {})
    if ph.get("available"):
        lines.append("## Price History Stats")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Tickers in CSV | {ph['total_tickers']} |")
        dr = ph.get("date_range", [None, None])
        lines.append(f"| Date range | {dr[0]} to {dr[1]} |")
        lines.append(f"| Median bars/ticker | {ph['median_bars_per_ticker']} |")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit drawdown data coverage in a snapshot."
    )
    parser.add_argument(
        "--snapshot-dir", required=True,
        help="Path to the snapshot directory (e.g. data/snapshots/2026-02-07)",
    )
    parser.add_argument(
        "--price-history",
        default="production_data/price_history.csv",
        help="Path to price_history.csv (default: production_data/price_history.csv)",
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Directory for output files (default: output)",
    )
    args = parser.parse_args()

    snap_dir = Path(args.snapshot_dir)
    rankings_path = snap_dir / "rankings.csv"
    if not rankings_path.exists():
        print(f"ERROR: rankings.csv not found in {snap_dir}", file=sys.stderr)
        return 1

    price_path = Path(args.price_history)
    if not price_path.exists():
        print(f"WARNING: price_history.csv not found at {price_path}", file=sys.stderr)
        price_path = None

    result = audit_drawdown_coverage(rankings_path, price_path)
    md_text = format_audit_markdown(result)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "drawdown_coverage_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    md_path = out_dir / "drawdown_coverage_audit.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    print(f"Coverage: {result['coverage_pct']}% ({result['drawdown_present']}/{result['total_dev_tickers']})")
    if result["missing_reason_counts"]:
        for reason, cnt in sorted(result["missing_reason_counts"].items(), key=lambda x: -x[1]):
            print(f"  {reason}: {cnt}")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
