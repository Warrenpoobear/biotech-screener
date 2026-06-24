#!/usr/bin/env python3
"""Universe maintenance — detect delistings, ticker changes, and coverage gaps.

Checks the universe against price data freshness, rankings coverage, and
SEC EDGAR CIK validity to flag names that may need attention: stale prices
(possible delisting), missing from rankings (screening failure), names
with no recent CTgov trials (dormant programs), and tickers with no CIK.

Read-only — does not modify universe.json. Writes a maintenance report
for human review.

Output:
    artifacts/universe_maintenance/{date}_report.json
    artifacts/universe_maintenance/{date}_report.md

Usage:
    python tools/build_universe_maintenance.py --as-of-date 2026-03-27
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("universe_maintenance")

SCHEMA_VERSION = "universe_maintenance.v1"

# Staleness thresholds (trading days)
PRICE_STALE_WARN = 5
PRICE_STALE_ALERT = 15


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def load_universe(path: Path) -> List[Dict]:
    data = _load_json(path)
    return data if isinstance(data, list) else []


def check_price_freshness(
    price_csv: Path,
    universe_tickers: Set[str],
    as_of_date: str,
) -> Dict[str, Dict[str, Any]]:
    """Check how recent each ticker's price data is."""
    latest_dates: Dict[str, str] = {}
    if not price_csv.exists():
        return {}
    with open(price_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = row.get("ticker", "")
            d = row.get("date", "")
            if t in universe_tickers and d:
                if t not in latest_dates or d > latest_dates[t]:
                    latest_dates[t] = d

    results = {}
    for t in universe_tickers:
        latest = latest_dates.get(t)
        if latest is None:
            results[t] = {"status": "NO_PRICE_DATA", "latest_date": None}
        else:
            # Simple day count (approximate, ignores weekends)
            from datetime import date as dt_date

            try:
                latest_d = dt_date.fromisoformat(latest)
                as_of_d = dt_date.fromisoformat(as_of_date)
                gap = (as_of_d - latest_d).days
            except ValueError:
                gap = 999

            if gap >= PRICE_STALE_ALERT:
                results[t] = {"status": "STALE_ALERT", "latest_date": latest, "gap_days": gap}
            elif gap >= PRICE_STALE_WARN:
                results[t] = {"status": "STALE_WARN", "latest_date": latest, "gap_days": gap}
    return results


def check_rankings_coverage(
    snapshots_dir: Path,
    universe_tickers: Set[str],
    as_of_date: str,
) -> Set[str]:
    """Find universe tickers missing from latest rankings."""
    rankings_path = snapshots_dir / as_of_date / "rankings.csv"
    if not rankings_path.exists():
        return universe_tickers
    with open(rankings_path, encoding="utf-8") as f:
        ranked_tickers = {row["ticker"] for row in csv.DictReader(f) if row.get("ticker")}
    return universe_tickers - ranked_tickers


def check_trial_coverage(
    cache_dir: Path,
    universe_tickers: Set[str],
) -> Set[str]:
    """Find universe tickers with zero CTgov trials."""
    candidates = sorted(p for p in cache_dir.glob("trial_records_*.json") if not p.name.endswith(".meta.json"))
    if not candidates:
        return set()

    with open(candidates[-1], encoding="utf-8") as f:
        records = json.load(f)

    tickers_with_trials = {r.get("ticker", "") for r in records}
    return universe_tickers - tickers_with_trials


def check_cik_coverage(universe: List[Dict]) -> List[str]:
    """Find tickers missing CIK (needed for SEC EDGAR)."""
    return [u["ticker"] for u in universe if not u.get("cik") and u.get("ticker")]


def build_universe_maintenance(
    as_of_date: str,
    *,
    production_dir: Path = REPO_ROOT / "production_data",
    snapshots_dir: Path = REPO_ROOT / "data" / "snapshots",
    cache_dir: Path = REPO_ROOT / "cache" / "ctgov",
    price_csv: Path = REPO_ROOT / "production_data" / "price_history.csv",
) -> Dict[str, Any]:
    """Build universe maintenance report."""
    universe = load_universe(production_dir / "universe.json")
    if not universe:
        return {"error": "empty universe"}

    universe_tickers = {u["ticker"] for u in universe if u.get("ticker")}

    # Run all checks
    price_issues = check_price_freshness(price_csv, universe_tickers, as_of_date)
    missing_rankings = check_rankings_coverage(snapshots_dir, universe_tickers, as_of_date)
    no_trials = check_trial_coverage(cache_dir, universe_tickers)
    no_cik = check_cik_coverage(universe)

    # Status counts
    no_status = [u["ticker"] for u in universe if not u.get("status") and u.get("ticker")]

    # Build flags per ticker
    flags: Dict[str, List[str]] = defaultdict(list)
    for t, issue in price_issues.items():
        flags[t].append(f"price:{issue['status']}")
    for t in missing_rankings:
        flags[t].append("missing_from_rankings")
    for t in no_trials:
        flags[t].append("no_ctgov_trials")
    for t in no_cik:
        flags[t].append("no_cik")
    for t in no_status:
        flags[t].append("no_status_field")

    # Sort by number of flags
    flagged = sorted(flags.items(), key=lambda x: -len(x[1]))

    result = {
        "schema": SCHEMA_VERSION,
        "as_of_date": as_of_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "universe_size": len(universe),
        "n_active": sum(1 for u in universe if u.get("status") == "active"),
        "n_flagged": len(flagged),
        "summary": {
            "price_stale_alert": sum(1 for v in price_issues.values() if v["status"] == "STALE_ALERT"),
            "price_stale_warn": sum(1 for v in price_issues.values() if v["status"] == "STALE_WARN"),
            "no_price_data": sum(1 for v in price_issues.values() if v["status"] == "NO_PRICE_DATA"),
            "missing_from_rankings": len(missing_rankings),
            "no_ctgov_trials": len(no_trials),
            "no_cik": len(no_cik),
            "no_status": len(no_status),
        },
        "flagged_tickers": [{"ticker": t, "flags": f, "n_flags": len(f)} for t, f in flagged],
    }

    # Write
    out_dir = REPO_ROOT / "artifacts" / "universe_maintenance"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{as_of_date}_report.json"
    md_path = out_dir / f"{as_of_date}_report.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Wrote %s", json_path)

    md_path.write_text(format_report_md(result), encoding="utf-8")
    logger.info("Wrote %s", md_path)

    result["_json_path"] = str(json_path)
    return result


def format_report_md(d: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"# Universe Maintenance — {d['as_of_date']}")
    lines.append("")
    lines.append(f"Universe: {d['universe_size']} tickers ({d['n_active']} active) | Flagged: {d['n_flagged']}")
    lines.append("")

    s = d.get("summary", {})
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Price stale (ALERT, >={PRICE_STALE_ALERT}d): {s.get('price_stale_alert', 0)}")
    lines.append(f"- Price stale (WARN, >={PRICE_STALE_WARN}d): {s.get('price_stale_warn', 0)}")
    lines.append(f"- No price data: {s.get('no_price_data', 0)}")
    lines.append(f"- Missing from rankings: {s.get('missing_from_rankings', 0)}")
    lines.append(f"- No CTgov trials: {s.get('no_ctgov_trials', 0)}")
    lines.append(f"- No SEC CIK: {s.get('no_cik', 0)}")
    lines.append(f"- No status field: {s.get('no_status', 0)}")
    lines.append("")

    flagged = d.get("flagged_tickers", [])
    if flagged:
        # Show worst first
        lines.append("## Flagged Tickers (sorted by flag count)")
        lines.append("")
        lines.append("| Ticker | Flags | Issues |")
        lines.append("|--------|-------|--------|")
        for f in flagged[:40]:
            issues = ", ".join(f["flags"])
            lines.append(f"| {f['ticker']} | {f['n_flags']} | {issues} |")
        if len(flagged) > 40:
            lines.append(f"| ... | | +{len(flagged) - 40} more |")
        lines.append("")

    lines.append(f"*Generated: {d.get('generated_at', '')}*")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Universe maintenance report")
    parser.add_argument("--as-of-date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    args = parser.parse_args()
    started = time.perf_counter()

    result = build_universe_maintenance(args.as_of_date)
    if "error" in result:
        logger.error(result["error"])
        try:
            from tools.agent_skill_telemetry import log_agent_run

            log_agent_run(
                "build_universe_maintenance",
                f"Universe maintenance for {args.as_of_date}",
                inputs={"as_of_date": args.as_of_date},
                outputs={"error": result["error"]},
                success=False,
                error=result["error"],
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception:
            pass
        sys.exit(1)
    logger.info("Universe: %d tickers, %d flagged", result["universe_size"], result["n_flagged"])
    try:
        from tools.agent_skill_telemetry import log_agent_run
        from tools.record_skill_feedback import attach_outcome_verdict

        stale_alert = (result.get("summary") or {}).get("price_stale_alert", 0)
        exec_id = log_agent_run(
            "build_universe_maintenance",
            f"Universe maintenance for {args.as_of_date}",
            inputs={"as_of_date": args.as_of_date},
            outputs={"universe_size": result.get("universe_size"), "n_flagged": result.get("n_flagged")},
            success=True,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        if exec_id:
            attach_outcome_verdict(
                exec_id,
                was_correct=stale_alert == 0,
                evidence=f"price_stale_alert={stale_alert} n_flagged={result.get('n_flagged', 0)}",
            )
    except Exception:
        pass


if __name__ == "__main__":
    main()
