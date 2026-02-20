#!/usr/bin/env python3
"""
warm_13f_cache.py — PIT-safe 13F warm cache builder.

Fetches 13F filings from SEC EDGAR for elite biotech managers, applies
point-in-time (PIT) filtering, and stores parsed results in a versioned,
date-partitioned cache with an index manifest.

Usage:
    python tools/warm_13f_cache.py --as-of-date 2026-02-19 --elite-only
    python tools/warm_13f_cache.py --as-of-date 2026-02-19 --max-managers 40

Output structure:
    data/caches/sec_13f/PIT/{as_of_date}/
        index.json                      # manifest
        managers/{CIK}.json             # parsed holdings per manager
        raw/{CIK}/{accession}.xml       # raw info table XML
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Repo root — same pattern as run_daily_production.py
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sec_13f.edgar_13f import SEC13FFetcher, Filing13F, Holding
from elite_managers import get_elite_managers, get_all_managers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("warm_13f_cache")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PITFilingSelection:
    """Result of PIT filing selection for one manager."""
    manager_cik: str
    filing: Optional[Filing13F]
    period_of_report: Optional[str]
    filed_at: Optional[str]
    form_type: Optional[str]
    accession: Optional[str]
    rejection_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Rate limiter — thread-safe token bucket
# ---------------------------------------------------------------------------

class RateLimiter:
    """Thread-safe token-bucket rate limiter.

    SEC EDGAR requests 10 req/s max; we use 8 as a safety margin.
    """

    def __init__(self, rate: float = 8.0):
        self._rate = rate
        self._interval = 1.0 / rate
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._last + self._interval - now
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()


# ---------------------------------------------------------------------------
# PIT selection — pure logic, no I/O
# ---------------------------------------------------------------------------

def select_pit_filing(
    filings: list[Filing13F],
    as_of_date: date,
) -> PITFilingSelection:
    """Select the PIT-correct filing for a manager.

    Algorithm:
      1. Filter filings where filing_date <= as_of_date
      2. Group by report_date, pick latest group
      3. Within group: prefer 13F-HR/A over 13F-HR
      4. If multiple amendments: pick latest filing_date
    """
    if not filings:
        return PITFilingSelection(
            manager_cik=filings[0].cik if filings else "",
            filing=None,
            period_of_report=None,
            filed_at=None,
            form_type=None,
            accession=None,
            rejection_reason="no_filings",
        )

    cik = filings[0].cik

    # Step 1: filter by as_of_date
    eligible = [f for f in filings if f.filing_date <= as_of_date]
    if not eligible:
        return PITFilingSelection(
            manager_cik=cik,
            filing=None,
            period_of_report=None,
            filed_at=None,
            form_type=None,
            accession=None,
            rejection_reason=f"no_filings_before_{as_of_date.isoformat()}",
        )

    # Step 2: group by report_date, pick latest
    latest_report_date = max(f.report_date for f in eligible)
    group = [f for f in eligible if f.report_date == latest_report_date]

    # Step 3: prefer amendments over originals
    amendments = [f for f in group if "/A" in f.form_type]
    candidates = amendments if amendments else group

    # Step 4: latest filing_date among candidates
    best = max(candidates, key=lambda f: f.filing_date)

    return PITFilingSelection(
        manager_cik=cik,
        filing=best,
        period_of_report=best.report_date.isoformat(),
        filed_at=best.filing_date.isoformat(),
        form_type=best.form_type,
        accession=best.accession_number,
    )


# ---------------------------------------------------------------------------
# Per-manager warm
# ---------------------------------------------------------------------------

def warm_one_manager(
    manager: Dict[str, Any],
    as_of_date: date,
    out_dir: Path,
    fetcher: SEC13FFetcher,
    rate_limiter: RateLimiter,
) -> Dict[str, Any]:
    """Fetch, PIT-select, parse, and write cache for one manager.

    Returns a result dict for index assembly.
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
        # Fetch recent filings (8 for ~2-year lookback)
        rate_limiter.acquire()
        filings = fetcher.get_recent_filings(cik, count=8)

        selection = select_pit_filing(filings, as_of_date)

        if selection.filing is None:
            result["status"] = "no_filing"
            result["rejection_reason"] = selection.rejection_reason
            logger.warning(f"  {name} (CIK {cik}): {selection.rejection_reason}")
            return result

        filing = selection.filing

        # Parse holdings
        rate_limiter.acquire()
        holdings = fetcher.parse_holdings(filing, resolve_tickers=True)

        # Build manager JSON
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

        # Write manager JSON
        managers_dir = out_dir / "managers"
        managers_dir.mkdir(parents=True, exist_ok=True)
        manager_path = managers_dir / f"{cik_padded}.json"
        with open(manager_path, "w", encoding="utf-8") as f:
            json.dump(manager_data, f, indent=2)

        # Copy raw XML if cached
        if fetcher.cache_dir:
            raw_xml_path = fetcher.cache_dir / f"{filing.filing_id}_infotable.xml"
            if raw_xml_path.exists():
                raw_dir = out_dir / "raw" / cik_padded
                raw_dir.mkdir(parents=True, exist_ok=True)
                dest = raw_dir / f"{filing.accession_number}.xml"
                shutil.copy2(raw_xml_path, dest)

        result["status"] = "ok"
        result["period_of_report"] = selection.period_of_report
        result["filed_at"] = selection.filed_at
        result["form_type"] = selection.form_type
        result["accession"] = selection.accession
        result["holdings_count"] = len(holdings)

        logger.info(
            f"  {name} (CIK {cik}): {len(holdings)} holdings, "
            f"report={selection.period_of_report}, filed={selection.filed_at}"
        )

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        logger.error(f"  {name} (CIK {cik}): {e}")

    return result


# ---------------------------------------------------------------------------
# Index builder — pure logic
# ---------------------------------------------------------------------------

def build_index(
    as_of_date: date,
    manager_results: list[Dict[str, Any]],
    total_managers: int,
    elite_only: bool,
) -> Dict[str, Any]:
    """Assemble the index.json manifest."""
    ok_results = [r for r in manager_results if r["status"] == "ok"]
    no_filing = [r for r in manager_results if r["status"] == "no_filing"]
    errors = [r for r in manager_results if r["status"] == "error"]

    coverage_pct = round(len(ok_results) / total_managers * 100, 1) if total_managers else 0.0

    return {
        "as_of_date": as_of_date.isoformat(),
        "created_at": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "elite_only": elite_only,
        "total_managers": total_managers,
        "managers_with_filing": len(ok_results),
        "managers_no_filing": len(no_filing),
        "managers_error": len(errors),
        "coverage_pct": coverage_pct,
        "managers": [
            {
                "cik": r["manager_cik"],
                "name": r["manager_name"],
                "status": r["status"],
                "period_of_report": r.get("period_of_report"),
                "filed_at": r.get("filed_at"),
                "form_type": r.get("form_type"),
                "accession": r.get("accession"),
                "holdings_count": r.get("holdings_count"),
                "rejection_reason": r.get("rejection_reason"),
                "error": r.get("error"),
            }
            for r in manager_results
        ],
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def warm_13f_cache(
    as_of_date: date,
    out_dir: Path,
    *,
    elite_only: bool = True,
    max_managers: Optional[int] = None,
    max_workers: int = 4,
    fetcher_cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Orchestrate 13F cache warming for all managers.

    Returns the index dict.
    """
    # Load manager list
    managers = get_elite_managers() if elite_only else get_all_managers()
    if max_managers:
        managers = managers[:max_managers]

    total = len(managers)
    logger.info(
        f"Warming 13F cache: {total} managers, as_of={as_of_date}, "
        f"elite_only={elite_only}, workers={max_workers}"
    )

    # Shared resources
    cache_dir = fetcher_cache_dir or (REPO_ROOT / "data" / "13f_cache")
    fetcher = SEC13FFetcher(cache_dir=str(cache_dir))
    rate_limiter = RateLimiter(rate=8.0)

    # Create output dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Warm each manager (threaded)
    results: list[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                warm_one_manager, mgr, as_of_date, out_dir, fetcher, rate_limiter,
            ): mgr
            for mgr in managers
        }

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                mgr = futures[future]
                logger.error(f"  {mgr.get('name', '?')}: unhandled {e}")
                results.append({
                    "manager_cik": mgr.get("cik", "").lstrip("0").zfill(10),
                    "manager_name": mgr.get("name", ""),
                    "status": "error",
                    "error": str(e),
                })

    # Sort results by CIK for determinism
    results.sort(key=lambda r: r["manager_cik"])

    # Build and write index
    index = build_index(as_of_date, results, total, elite_only)
    index_path = out_dir / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    ok_count = index["managers_with_filing"]
    logger.info(
        f"13F cache complete: {ok_count}/{total} managers with filings, "
        f"coverage={index['coverage_pct']}%"
    )

    return index


# ---------------------------------------------------------------------------
# Gate function (for run_daily_production.py)
# ---------------------------------------------------------------------------

def check_13f_cache_health(
    cache_dir: Path,
    as_of_date: str,
    *,
    warn_coverage_pct: float = 80.0,
) -> Dict[str, Any]:
    """Check 13F cache health. Returns dict with status/detail/value/threshold.

    WARN-only gate — never returns FAIL.
    """
    index_path = cache_dir / as_of_date / "index.json"

    if not index_path.exists():
        return {
            "status": "WARN",
            "detail": f"No 13F cache index found at {index_path}",
            "value": None,
            "threshold": {"warn_coverage_pct": warn_coverage_pct},
        }

    try:
        with open(index_path) as f:
            index = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return {
            "status": "WARN",
            "detail": f"Cannot read 13F cache index: {e}",
            "value": None,
            "threshold": {"warn_coverage_pct": warn_coverage_pct},
        }

    coverage = index.get("coverage_pct", 0.0)
    managers_ok = index.get("managers_with_filing", 0)
    total = index.get("total_managers", 0)

    detail_parts = [
        f"coverage={coverage:.1f}%",
        f"({managers_ok}/{total} managers)",
    ]

    if coverage < warn_coverage_pct:
        detail_parts.append(f"below {warn_coverage_pct:.0f}% threshold")
        return {
            "status": "WARN",
            "detail": ", ".join(detail_parts),
            "value": {"coverage_pct": coverage, "managers_ok": managers_ok, "total": total},
            "threshold": {"warn_coverage_pct": warn_coverage_pct},
        }

    return {
        "status": "PASS",
        "detail": ", ".join(detail_parts),
        "value": {"coverage_pct": coverage, "managers_ok": managers_ok, "total": total},
        "threshold": {"warn_coverage_pct": warn_coverage_pct},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build PIT-safe 13F cache from SEC EDGAR filings.",
    )
    parser.add_argument(
        "--as-of-date", required=True,
        help="PIT cutoff date (YYYY-MM-DD). Only filings filed on or before this date are included.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output directory (default: data/caches/sec_13f/PIT/{as_of_date})",
    )
    parser.add_argument(
        "--elite-only", action="store_true", default=False,
        help="Only warm Elite Core (Tier 1) managers (default: all managers)",
    )
    parser.add_argument(
        "--max-managers", type=int, default=None,
        help="Limit number of managers (for testing)",
    )
    parser.add_argument(
        "--max-workers", type=int, default=4,
        help="Thread pool size (default: 4)",
    )
    parser.add_argument(
        "--fetcher-cache-dir", type=Path, default=None,
        help="SEC13FFetcher XML cache directory (default: data/13f_cache)",
    )

    args = parser.parse_args()
    as_of = date.fromisoformat(args.as_of_date)
    out_dir = args.out or (REPO_ROOT / "data" / "caches" / "sec_13f" / "PIT" / as_of.isoformat())

    index = warm_13f_cache(
        as_of_date=as_of,
        out_dir=out_dir,
        elite_only=args.elite_only,
        max_managers=args.max_managers,
        max_workers=args.max_workers,
        fetcher_cache_dir=args.fetcher_cache_dir,
    )

    ok = index["managers_with_filing"]
    total = index["total_managers"]
    print(f"\nDone: {ok}/{total} managers cached → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
