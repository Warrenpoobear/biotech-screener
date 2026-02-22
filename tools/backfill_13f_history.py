#!/usr/bin/env python3
"""
backfill_13f_history.py — Generate quarterly PIT-safe 13F caches for backtesting.

Reuses PIT selection logic and schema from warm_13f_cache.py but with extended
lookback and multi-date orchestration.  No existing files are modified.

Usage:
    python tools/backfill_13f_history.py \
      --date-from 2020-01-01 --date-to 2026-02-20 \
      --cadence quarterly --manager-set coinvest \
      --out-root data/caches/sec_13f/PIT \
      --max-workers 4 --rate-limit 8 --lookback-filings 16 \
      --resume --report

Output per date:
    {out_root}/{as_of_date}/
        index.json
        managers/{CIK}.json
        raw/{CIK}/{accession}.xml   (if fetcher has cached XML)
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Repo root — same pattern as warm_13f_cache.py
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sec_13f.edgar_13f import SEC13FFetcher, Filing13F, Holding
from elite_managers import get_elite_managers, get_all_managers
from tools.warm_13f_cache import (
    SCHEMA_VERSION,
    CACHE_TYPE,
    RateLimiter,
    select_pit_filing,
    validate_sec_13f_index_schema,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("backfill_13f_history")

TOOL_VERSION = "1.0.0"
DEFAULT_LOOKBACK_FILINGS = 16  # ~4-year lookback (vs 8 in daily)


# ---------------------------------------------------------------------------
# Quarter-end date generation
# ---------------------------------------------------------------------------

def generate_quarter_end_dates(
    date_from: date,
    date_to: date,
) -> List[date]:
    """Return ascending list of quarter-end dates within [date_from, date_to]."""
    if date_from > date_to:
        return []

    quarter_ends = []
    # Quarter-end months and days
    qe_specs = [(3, 31), (6, 30), (9, 30), (12, 31)]

    year = date_from.year
    while year <= date_to.year:
        for month, day in qe_specs:
            d = date(year, month, day)
            if date_from <= d <= date_to:
                quarter_ends.append(d)
        year += 1

    return quarter_ends


# ---------------------------------------------------------------------------
# Per-manager backfill (parameterised lookback)
# ---------------------------------------------------------------------------

def _warm_one_manager_backfill(
    manager: Dict[str, Any],
    as_of_date: date,
    out_dir: Path,
    fetcher: SEC13FFetcher,
    rate_limiter: RateLimiter,
    lookback_filings: int,
) -> Dict[str, Any]:
    """Fetch, PIT-select, parse, and write cache for one manager.

    Like warm_one_manager() but accepts configurable lookback_filings.
    """
    cik = manager["cik"]
    name = manager.get("name", "")
    cik_padded = cik.lstrip("0").zfill(10)

    result: Dict[str, Any] = {
        "manager_cik": cik_padded,
        "manager_name": name,
        "status": "error",
    }

    try:
        rate_limiter.acquire()
        filings = fetcher.get_recent_filings(cik, count=lookback_filings)

        selection = select_pit_filing(filings, as_of_date)

        if selection.filing is None:
            result["status"] = "no_filing"
            result["rejection_reason"] = selection.rejection_reason
            logger.debug(
                f"  {name} (CIK {cik}): {selection.rejection_reason}"
            )
            return result

        filing = selection.filing

        rate_limiter.acquire()
        holdings = fetcher.parse_holdings(filing, resolve_tickers=True)

        manager_data = {
            "manager_cik": cik_padded,
            "manager_name": name,
            "as_of_date": as_of_date.isoformat(),
            "period_of_report": selection.period_of_report,
            "filed_at": selection.filed_at,
            "form_type": selection.form_type,
            "accession": selection.accession,
            "holdings": [
                {
                    "cusip": h.cusip,
                    "ticker": h.ticker or "",
                    "issuer": h.issuer_name,
                    "shares": h.shares,
                    "value_usd_thousands": h.value,
                    "put_call": h.put_call or "",
                }
                for h in holdings
            ],
        }

        managers_dir = out_dir / "managers"
        managers_dir.mkdir(parents=True, exist_ok=True)
        manager_path = managers_dir / f"{cik_padded}.json"
        with open(manager_path, "w", encoding="utf-8") as f:
            json.dump(manager_data, f, indent=2)

        has_raw_xml = False
        if fetcher.cache_dir:
            raw_xml_src = fetcher.cache_dir / f"{filing.filing_id}_infotable.xml"
            if raw_xml_src.exists():
                raw_dir = out_dir / "raw" / cik_padded
                raw_dir.mkdir(parents=True, exist_ok=True)
                dest = raw_dir / f"{filing.accession_number}.xml"
                shutil.copy2(raw_xml_src, dest)
                has_raw_xml = True

        result["status"] = "ok"
        result["period_of_report"] = selection.period_of_report
        result["filed_at"] = selection.filed_at
        result["form_type"] = selection.form_type
        result["accession"] = selection.accession
        result["holdings_count"] = len(holdings)
        result["manager_json_path"] = f"managers/{cik_padded}.json"
        result["raw_xml_path"] = (
            f"raw/{cik_padded}/{filing.accession_number}.xml"
            if has_raw_xml
            else ""
        )

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        logger.error(f"  {name} (CIK {cik}): {e}")

    return result


# ---------------------------------------------------------------------------
# Index builder with backfill metadata
# ---------------------------------------------------------------------------

def _build_backfill_index(
    as_of_date: date,
    manager_results: List[Dict[str, Any]],
    total_managers: int,
    elite_only: bool,
    lookback_filings: int,
    cadence: str,
    manager_set: str,
) -> Dict[str, Any]:
    """Build index.json with v1 schema fields + backfill metadata block."""
    ok_results = [r for r in manager_results if r["status"] == "ok"]
    coverage_pct = (
        round(len(ok_results) / total_managers * 100, 1)
        if total_managers
        else 0.0
    )

    managers_list = []
    for r in manager_results:
        selected = r["status"] == "ok"
        entry: Dict[str, Any] = {
            "manager_cik": r["manager_cik"],
            "manager_name": r["manager_name"],
            "selected": selected,
        }
        if selected:
            entry["period_of_report"] = r.get("period_of_report", "")
            entry["filed_at"] = r.get("filed_at", "")
            entry["form_type"] = r.get("form_type", "")
            entry["accession"] = r.get("accession", "")
            entry["holdings_count"] = r.get("holdings_count", 0)
            entry["manager_json_path"] = r.get("manager_json_path", "")
            entry["raw_xml_path"] = r.get("raw_xml_path", "")
            entry["rejection_reason"] = ""
        else:
            entry["rejection_reason"] = (
                r.get("rejection_reason") or r.get("error") or "unknown"
            )
        managers_list.append(entry)

    return {
        "schema_version": SCHEMA_VERSION,
        "cache_type": CACHE_TYPE,
        "as_of_date": as_of_date.isoformat(),
        "created_at": datetime.now(tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "elite_only": elite_only,
        "total_managers": total_managers,
        "managers_with_filing": len(ok_results),
        "coverage_pct": coverage_pct,
        "selection_policy": {
            "pit_rule": (
                "filing_date<=as_of; latest report_date; "
                "prefer 13F-HR/A; latest filing_date"
            ),
            "recent_filings_lookback_n": lookback_filings,
        },
        "managers": managers_list,
        "backfill": {
            "tool_version": TOOL_VERSION,
            "cadence": cadence,
            "manager_set": manager_set,
            "lookback_filings": lookback_filings,
            "generated_at": datetime.now(tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        },
    }


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _is_date_complete(out_root: Path, as_of_date: date) -> bool:
    """Return True if a valid v1 index.json already exists for this date."""
    index_path = out_root / as_of_date.isoformat() / "index.json"
    if not index_path.exists():
        return False
    try:
        with open(index_path) as f:
            index = json.load(f)
        ok, _ = validate_sec_13f_index_schema(
            index, expected_as_of_date=as_of_date.isoformat()
        )
        return ok
    except (json.JSONDecodeError, OSError):
        return False


def _find_missing_managers(
    date_dir: Path,
    managers: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Scan managers/ dir; return (missing_managers, existing_results).

    existing_results are reconstructed result dicts for index assembly.
    """
    managers_dir = date_dir / "managers"
    existing_results: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []

    for mgr in managers:
        cik = mgr["cik"]
        cik_padded = cik.lstrip("0").zfill(10)
        mgr_path = managers_dir / f"{cik_padded}.json"

        if mgr_path.exists():
            try:
                with open(mgr_path) as f:
                    data = json.load(f)
                # Reconstruct a result dict from the cached manager JSON
                existing_results.append({
                    "manager_cik": cik_padded,
                    "manager_name": data.get("manager_name", mgr.get("name", "")),
                    "status": "ok",
                    "period_of_report": data.get("period_of_report", ""),
                    "filed_at": data.get("filed_at", ""),
                    "form_type": data.get("form_type", ""),
                    "accession": data.get("accession", ""),
                    "holdings_count": len(data.get("holdings", [])),
                    "manager_json_path": f"managers/{cik_padded}.json",
                    "raw_xml_path": "",  # cannot recover from manager JSON alone
                })
            except (json.JSONDecodeError, OSError):
                # Corrupt JSON — re-fetch
                missing.append(mgr)
        else:
            missing.append(mgr)

    return missing, existing_results


# ---------------------------------------------------------------------------
# Per-date orchestrator
# ---------------------------------------------------------------------------

def _backfill_one_date(
    as_of_date: date,
    managers: List[Dict[str, Any]],
    out_root: Path,
    fetcher: SEC13FFetcher,
    rate_limiter: RateLimiter,
    lookback_filings: int,
    max_workers: int,
    cadence: str,
    manager_set: str,
    resume: bool,
    max_managers: Optional[int] = None,
) -> Dict[str, Any]:
    """Warm cache for a single date, with resume support."""
    date_dir = out_root / as_of_date.isoformat()
    elite_only = manager_set == "elite"
    total_managers = len(managers)

    # Apply max_managers limit
    effective_managers = managers[:max_managers] if max_managers else managers
    effective_total = len(effective_managers)

    # Resume: find missing managers
    existing_results: List[Dict[str, Any]] = []
    to_fetch = effective_managers

    if resume and date_dir.exists():
        missing, existing_results = _find_missing_managers(
            date_dir, effective_managers
        )
        to_fetch = missing
        if not to_fetch:
            logger.info(
                f"  {as_of_date}: all {len(existing_results)} managers "
                f"cached, skipping"
            )

    date_dir.mkdir(parents=True, exist_ok=True)

    # Fetch missing managers (threaded)
    new_results: List[Dict[str, Any]] = []
    if to_fetch:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _warm_one_manager_backfill,
                    mgr,
                    as_of_date,
                    date_dir,
                    fetcher,
                    rate_limiter,
                    lookback_filings,
                ): mgr
                for mgr in to_fetch
            }
            for future in as_completed(futures):
                try:
                    result = future.result()
                    new_results.append(result)
                except Exception as e:
                    mgr = futures[future]
                    logger.error(
                        f"  {mgr.get('name', '?')}: unhandled {e}"
                    )
                    new_results.append({
                        "manager_cik": mgr.get("cik", "").lstrip("0").zfill(10),
                        "manager_name": mgr.get("name", ""),
                        "status": "error",
                        "error": str(e),
                    })

    # Merge existing + new, deduplicate by CIK (new wins)
    results_by_cik: Dict[str, Dict[str, Any]] = {}
    for r in existing_results:
        results_by_cik[r["manager_cik"]] = r
    for r in new_results:
        results_by_cik[r["manager_cik"]] = r

    all_results = sorted(results_by_cik.values(), key=lambda r: r["manager_cik"])

    # Build and write index
    index = _build_backfill_index(
        as_of_date,
        all_results,
        effective_total,
        elite_only,
        lookback_filings,
        cadence,
        manager_set,
    )
    index_path = date_dir / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    ok_count = index["managers_with_filing"]
    logger.info(
        f"  {as_of_date}: {ok_count}/{effective_total} managers, "
        f"coverage={index['coverage_pct']}% "
        f"(fetched={len(new_results)}, cached={len(existing_results)})"
    )

    return {
        "as_of_date": as_of_date.isoformat(),
        "managers_with_filing": ok_count,
        "total_managers": effective_total,
        "coverage_pct": index["coverage_pct"],
        "fetched": len(new_results),
        "cached": len(existing_results),
    }


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------

def _build_coverage_report(
    per_date: List[Dict[str, Any]],
    total_managers: int,
) -> Dict[str, Any]:
    """Aggregate coverage statistics across all dates."""
    if not per_date:
        return {
            "total_dates": 0,
            "total_managers_target": total_managers,
        }

    coverages = [d["coverage_pct"] for d in per_date]
    return {
        "total_dates": len(per_date),
        "total_managers_target": total_managers,
        "avg_coverage_pct": round(sum(coverages) / len(coverages), 1),
        "min_coverage_pct": min(coverages),
        "max_coverage_pct": max(coverages),
        "dates_below_80pct": sum(1 for c in coverages if c < 80.0),
        "dates_at_100pct": sum(1 for c in coverages if c >= 100.0),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _load_managers(manager_set: str) -> List[Dict[str, Any]]:
    """Load manager list based on manager_set parameter."""
    if manager_set == "coinvest":
        return get_all_managers()
    elif manager_set == "elite":
        return get_elite_managers()
    else:
        raise ValueError(
            f"Unknown manager_set: {manager_set!r}. "
            f"Must be 'coinvest' or 'elite'."
        )


def backfill_13f_history(
    date_from: date,
    date_to: date,
    *,
    cadence: str = "quarterly",
    manager_set: str = "coinvest",
    out_root: Optional[Path] = None,
    max_workers: int = 4,
    rate_limit: float = 8.0,
    lookback_filings: int = DEFAULT_LOOKBACK_FILINGS,
    resume: bool = False,
    report: bool = False,
    max_dates: Optional[int] = None,
    max_managers: Optional[int] = None,
    fetcher_cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generate quarterly PIT caches for backtesting.

    Returns summary dict with per-date results and optional coverage report.
    """
    if out_root is None:
        out_root = REPO_ROOT / "data" / "caches" / "sec_13f" / "PIT"

    # Generate dates
    dates = generate_quarter_end_dates(date_from, date_to)
    if max_dates:
        dates = dates[:max_dates]

    # Load managers
    managers = _load_managers(manager_set)
    total_managers = len(managers)

    logger.info(
        f"Backfill 13F: {len(dates)} dates, {total_managers} managers "
        f"({manager_set}), lookback={lookback_filings}, "
        f"workers={max_workers}, rate={rate_limit}/s"
    )

    # Shared resources
    cache_dir = fetcher_cache_dir or (REPO_ROOT / "data" / "13f_cache")
    fetcher = SEC13FFetcher(cache_dir=str(cache_dir))
    rate_limiter = RateLimiter(rate=rate_limit)

    # Process dates sequentially
    per_date: List[Dict[str, Any]] = []
    skipped = 0

    for i, d in enumerate(dates):
        logger.info(f"[{i + 1}/{len(dates)}] Processing {d.isoformat()}...")

        # Fast skip for resume: check complete index first
        if resume and _is_date_complete(out_root, d):
            logger.info(f"  {d}: complete index exists, skipping")
            # Read existing index for summary
            with open(out_root / d.isoformat() / "index.json") as f:
                existing_index = json.load(f)
            per_date.append({
                "as_of_date": d.isoformat(),
                "managers_with_filing": existing_index["managers_with_filing"],
                "total_managers": existing_index["total_managers"],
                "coverage_pct": existing_index["coverage_pct"],
                "fetched": 0,
                "cached": existing_index["total_managers"],
                "skipped": True,
            })
            skipped += 1
            continue

        result = _backfill_one_date(
            as_of_date=d,
            managers=managers,
            out_root=out_root,
            fetcher=fetcher,
            rate_limiter=rate_limiter,
            lookback_filings=lookback_filings,
            max_workers=max_workers,
            cadence=cadence,
            manager_set=manager_set,
            resume=resume,
            max_managers=max_managers,
        )
        per_date.append(result)

    summary: Dict[str, Any] = {
        "tool_version": TOOL_VERSION,
        "date_from": date_from.isoformat(),
        "date_to": date_to.isoformat(),
        "cadence": cadence,
        "manager_set": manager_set,
        "lookback_filings": lookback_filings,
        "total_dates": len(dates),
        "dates_skipped": skipped,
        "per_date": per_date,
    }

    if report:
        summary["coverage_report"] = _build_coverage_report(
            per_date, total_managers
        )

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate quarterly PIT-safe 13F caches for backtesting.",
    )
    parser.add_argument(
        "--date-from",
        required=True,
        help="Start date inclusive (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--date-to",
        required=True,
        help="End date inclusive (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--cadence",
        default="quarterly",
        choices=["quarterly"],
        help="Date cadence (default: quarterly)",
    )
    parser.add_argument(
        "--manager-set",
        default="coinvest",
        choices=["coinvest", "elite"],
        help="Manager set: coinvest (elite+conditional) or elite only",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output root directory (default: data/caches/sec_13f/PIT)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Thread pool size (default: 4)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=8.0,
        help="SEC EDGAR rate limit (req/s, default: 8.0)",
    )
    parser.add_argument(
        "--lookback-filings",
        type=int,
        default=DEFAULT_LOOKBACK_FILINGS,
        help=f"Filing lookback count (default: {DEFAULT_LOOKBACK_FILINGS})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip dates with valid index; repair partial caches",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print aggregate coverage report at end",
    )
    parser.add_argument(
        "--max-dates",
        type=int,
        default=None,
        help="Limit number of dates to process (for testing)",
    )
    parser.add_argument(
        "--max-managers",
        type=int,
        default=None,
        help="Limit managers per date (for testing)",
    )
    parser.add_argument(
        "--fetcher-cache-dir",
        type=Path,
        default=None,
        help="SEC13FFetcher XML cache directory",
    )

    args = parser.parse_args()

    summary = backfill_13f_history(
        date_from=date.fromisoformat(args.date_from),
        date_to=date.fromisoformat(args.date_to),
        cadence=args.cadence,
        manager_set=args.manager_set,
        out_root=args.out_root,
        max_workers=args.max_workers,
        rate_limit=args.rate_limit,
        lookback_filings=args.lookback_filings,
        resume=args.resume,
        report=args.report,
        max_dates=args.max_dates,
        max_managers=args.max_managers,
        fetcher_cache_dir=args.fetcher_cache_dir,
    )

    total = summary["total_dates"]
    skipped = summary["dates_skipped"]
    print(f"\nDone: {total} dates processed ({skipped} skipped)")

    if "coverage_report" in summary:
        rpt = summary["coverage_report"]
        print(f"\nCoverage Report:")
        print(f"  Dates:           {rpt['total_dates']}")
        print(f"  Target managers: {rpt['total_managers_target']}")
        if rpt["total_dates"] > 0:
            print(f"  Avg coverage:    {rpt['avg_coverage_pct']}%")
            print(f"  Min coverage:    {rpt['min_coverage_pct']}%")
            print(f"  Max coverage:    {rpt['max_coverage_pct']}%")
            print(f"  Below 80%:       {rpt['dates_below_80pct']}")
            print(f"  At 100%:         {rpt['dates_at_100pct']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
