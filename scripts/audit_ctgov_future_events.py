#!/usr/bin/env python3
"""
audit_ctgov_future_events.py - Audit CTGov future-dated milestones

Quantifies and validates "future catalyst" yield from CTGov trial records:
  - How many trials have future PCD / CD / results_first_posted?
  - Are those dates plausible (not malformed, not absurdly far-future)?
  - Distribution by horizon bucket
  - Per-ticker coverage of future events
  - Suspicious flags (PCD after CD, malformed, missing disclosed_at)

Exit codes:
  0 — audit complete (this is audit, not a gate)
  2 — file missing or parse error
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Core audit logic (importable for testing)
# ---------------------------------------------------------------------------

_HORIZON_BUCKETS = [
    ("0-30d", 0, 30),
    ("31-90d", 31, 90),
    ("91-180d", 91, 180),
    ("181-365d", 181, 365),
    (">365d", 366, 999999),
]

_DATE_FIELDS = ("primary_completion_date", "completion_date", "results_first_posted")


def _parse_date_safe(raw: Any) -> Optional[date]:
    """Parse YYYY-MM-DD or YYYY-MM; return None on failure."""
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip()[:10]
    # YYYY-MM → YYYY-MM-01
    if len(s) == 7 and s[4] == "-":
        s = s + "-01"
    try:
        return date.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _bucket_name(days_ahead: int) -> str:
    for name, lo, hi in _HORIZON_BUCKETS:
        if lo <= days_ahead <= hi:
            return name
    return ">365d"


def audit_future_events(
    records: List[dict],
    as_of_date: date,
    horizon_days: int = 540,
    top_n: int = 30,
) -> dict:
    """Run the full audit. Returns a schema=ctgov_future_audit.v1 dict."""
    total = len(records)
    tickers_seen: set = set()
    future_pcd_count = 0
    future_cd_count = 0
    future_results_count = 0

    # Horizon buckets per field
    pcd_buckets: Counter = Counter()
    cd_buckets: Counter = Counter()

    # Per-ticker: earliest future PCD
    ticker_earliest_pcd: Dict[str, date] = {}
    ticker_future_pcd_count: Counter = Counter()

    # Suspicious flags
    malformed_dates: List[dict] = []
    far_future: List[dict] = []
    pcd_after_cd: List[dict] = []
    missing_disclosed_at: List[dict] = []

    for rec in records:
        ticker = rec.get("ticker", "")
        nct_id = rec.get("nct_id", "")
        if ticker:
            tickers_seen.add(ticker)

        pcd = _parse_date_safe(rec.get("primary_completion_date"))
        cd = _parse_date_safe(rec.get("completion_date"))
        rfp = _parse_date_safe(rec.get("results_first_posted"))
        lup = rec.get("last_update_posted")

        # Check malformed dates
        for field in _DATE_FIELDS:
            raw = rec.get(field)
            if raw and _parse_date_safe(raw) is None:
                malformed_dates.append({
                    "nct_id": nct_id, "ticker": ticker,
                    "field": field, "raw_value": str(raw)[:50],
                })

        # Future PCD
        if pcd and pcd > as_of_date:
            future_pcd_count += 1
            days_ahead = (pcd - as_of_date).days
            pcd_buckets[_bucket_name(days_ahead)] += 1
            ticker_future_pcd_count[ticker] += 1
            if ticker not in ticker_earliest_pcd or pcd < ticker_earliest_pcd[ticker]:
                ticker_earliest_pcd[ticker] = pcd

            # Far-future flag
            if days_ahead > horizon_days:
                far_future.append({
                    "nct_id": nct_id, "ticker": ticker,
                    "field": "primary_completion_date",
                    "date": pcd.isoformat(), "days_ahead": days_ahead,
                })

        # Future CD
        if cd and cd > as_of_date:
            future_cd_count += 1
            days_ahead = (cd - as_of_date).days
            cd_buckets[_bucket_name(days_ahead)] += 1

        # Future results
        if rfp and rfp > as_of_date:
            future_results_count += 1

        # PCD after CD by > 30 days
        if pcd and cd and pcd > cd and (pcd - cd).days > 30:
            pcd_after_cd.append({
                "nct_id": nct_id, "ticker": ticker,
                "pcd": pcd.isoformat(), "cd": cd.isoformat(),
                "gap_days": (pcd - cd).days,
            })

        # Missing disclosed_at on records with future dates
        if (pcd and pcd > as_of_date) or (cd and cd > as_of_date):
            if not lup:
                missing_disclosed_at.append({
                    "nct_id": nct_id, "ticker": ticker,
                })

    # Top tickers by future PCD count
    top_tickers = []
    for tk, cnt in ticker_future_pcd_count.most_common(top_n):
        earliest = ticker_earliest_pcd.get(tk)
        top_tickers.append({
            "ticker": tk,
            "future_pcd_count": cnt,
            "earliest_future_pcd": earliest.isoformat() if earliest else None,
        })

    # Status distribution
    status_dist = Counter((r.get("status") or "MISSING") for r in records)

    metrics = {
        "total_trials": total,
        "tickers_covered": len(tickers_seen),
        "future_primary_completion_date": future_pcd_count,
        "future_completion_date": future_cd_count,
        "future_results_first_posted": future_results_count,
        "pct_future_primary_completion_date": round(100.0 * future_pcd_count / max(total, 1), 2),
        "pct_future_completion_date": round(100.0 * future_cd_count / max(total, 1), 2),
        "pct_future_results_first_posted": round(100.0 * future_results_count / max(total, 1), 2),
        "pcd_horizon_buckets": dict(sorted(pcd_buckets.items())),
        "cd_horizon_buckets": dict(sorted(cd_buckets.items())),
        "status_distribution": dict(status_dist.most_common(15)),
    }

    flags = {
        "malformed_dates": malformed_dates[:50],
        "far_future_dates": far_future[:50],
        "pcd_after_cd": pcd_after_cd[:50],
        "missing_disclosed_at_with_future_dates": missing_disclosed_at[:50],
    }

    flag_summary = {
        "malformed_count": len(malformed_dates),
        "far_future_count": len(far_future),
        "pcd_after_cd_count": len(pcd_after_cd),
        "missing_disclosed_at_count": len(missing_disclosed_at),
    }

    return {
        "schema": "ctgov_future_audit.v1",
        "as_of_date": as_of_date.isoformat(),
        "horizon_days": horizon_days,
        "metrics": metrics,
        "top_tickers_by_future_pcd": top_tickers,
        "flags": flags,
        "flag_summary": flag_summary,
    }


def format_markdown(audit: dict, top_n: int = 30) -> str:
    """Render audit dict as a compact Markdown summary."""
    m = audit["metrics"]
    fs = audit["flag_summary"]
    lines = [
        f"# CTGov Future Events Audit — {audit['as_of_date']}",
        "",
        "## Key Metrics",
        f"- Total trials: **{m['total_trials']:,}**",
        f"- Tickers covered: **{m['tickers_covered']}**",
        f"- Future PCD: **{m['future_primary_completion_date']:,}** ({m['pct_future_primary_completion_date']:.1f}%)",
        f"- Future CD: **{m['future_completion_date']:,}** ({m['pct_future_completion_date']:.1f}%)",
        f"- Future results posted: **{m['future_results_first_posted']:,}** ({m['pct_future_results_first_posted']:.1f}%)",
        "",
        "## PCD Horizon Buckets",
    ]
    for bucket, count in sorted(m.get("pcd_horizon_buckets", {}).items()):
        lines.append(f"- {bucket}: {count}")

    lines += [
        "",
        "## Flags",
        f"- Malformed dates: {fs['malformed_count']}",
        f"- Far-future (>{audit['horizon_days']}d): {fs['far_future_count']}",
        f"- PCD after CD (>30d gap): {fs['pcd_after_cd_count']}",
        f"- Missing disclosed_at w/ future dates: {fs['missing_disclosed_at_count']}",
        "",
        "## Top Tickers by Future PCD Count",
        "| Ticker | Future PCD | Earliest |",
        "|--------|-----------|----------|",
    ]
    for t in audit.get("top_tickers_by_future_pcd", [])[:top_n]:
        lines.append(f"| {t['ticker']} | {t['future_pcd_count']} | {t['earliest_future_pcd'] or 'N/A'} |")

    lines += [
        "",
        "## Status Distribution",
    ]
    for status, count in sorted(m.get("status_distribution", {}).items(), key=lambda x: -x[1]):
        lines.append(f"- {status}: {count}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Audit CTGov future event extraction quality")
    parser.add_argument("--as-of-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--ctgov-cache", required=True, help="Path to trial_records_YYYY-MM-DD.json")
    parser.add_argument("--out-json", default=None, help="Output JSON path")
    parser.add_argument("--out-md", default=None, help="Output Markdown path")
    parser.add_argument("--horizon-days", type=int, default=540, help="Flag dates beyond N days (default: 540)")
    parser.add_argument("--top-n", type=int, default=30, help="Top-N tickers in report (default: 30)")
    args = parser.parse_args()

    # sys.path setup for project imports
    import sys as _sys
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in _sys.path:
        _sys.path.insert(0, project_root)

    cache_path = Path(args.ctgov_cache)
    if not cache_path.exists():
        print(f"ERROR: CTGov cache not found: {cache_path}", file=sys.stderr)
        sys.exit(2)

    try:
        records = json.loads(cache_path.read_text())
        if not isinstance(records, list):
            print(f"ERROR: Expected list, got {type(records).__name__}", file=sys.stderr)
            sys.exit(2)
    except (json.JSONDecodeError, OSError) as e:
        print(f"ERROR: Failed to read cache: {e}", file=sys.stderr)
        sys.exit(2)

    as_of = date.fromisoformat(args.as_of_date)
    audit = audit_future_events(records, as_of, args.horizon_days, args.top_n)

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
        print(f"JSON → {args.out_json}")

    md = format_markdown(audit, args.top_n)
    if args.out_md:
        Path(args.out_md).write_text(md)
        print(f"Markdown → {args.out_md}")
    else:
        print(md)

    # Print key metrics to stdout
    m = audit["metrics"]
    print(f"\n--- Summary ---")
    print(f"Total trials: {m['total_trials']:,}")
    print(f"Future PCD: {m['future_primary_completion_date']:,} ({m['pct_future_primary_completion_date']:.1f}%)")
    print(f"Future CD: {m['future_completion_date']:,} ({m['pct_future_completion_date']:.1f}%)")
    print(f"Flags: malformed={audit['flag_summary']['malformed_count']}, "
          f"far_future={audit['flag_summary']['far_future_count']}, "
          f"pcd>cd={audit['flag_summary']['pcd_after_cd_count']}")

    sys.exit(0)


if __name__ == "__main__":
    main()
