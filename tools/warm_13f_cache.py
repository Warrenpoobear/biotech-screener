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
# Schema contract constants
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "sec_13f_pit_index.v1"
CACHE_TYPE = "sec_13f_pit"
FILINGS_LOOKBACK_N = 8  # ~2-year lookback for get_recent_filings

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
        # Fetch recent filings
        rate_limiter.acquire()
        filings = fetcher.get_recent_filings(cik, count=FILINGS_LOOKBACK_N)

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
            f"raw/{cik_padded}/{filing.accession_number}.xml" if has_raw_xml else ""
        )

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
    """Assemble the index.json manifest (schema: sec_13f_pit_index.v1)."""
    ok_results = [r for r in manager_results if r["status"] == "ok"]

    coverage_pct = round(len(ok_results) / total_managers * 100, 1) if total_managers else 0.0

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
            entry["rejection_reason"] = r.get("rejection_reason") or r.get("error") or "unknown"

        managers_list.append(entry)

    return {
        "schema_version": SCHEMA_VERSION,
        "cache_type": CACHE_TYPE,
        "as_of_date": as_of_date.isoformat(),
        "created_at": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "elite_only": elite_only,
        "total_managers": total_managers,
        "managers_with_filing": len(ok_results),
        "coverage_pct": coverage_pct,
        "selection_policy": {
            "pit_rule": "filing_date<=as_of; latest report_date; prefer 13F-HR/A; latest filing_date",
            "recent_filings_lookback_n": FILINGS_LOOKBACK_N,
        },
        "managers": managers_list,
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
# Schema validator — pure logic, no I/O
# ---------------------------------------------------------------------------

_REQUIRED_TOP_LEVEL = {
    "schema_version", "cache_type", "as_of_date", "created_at",
    "elite_only", "total_managers", "managers_with_filing",
    "coverage_pct", "managers",
}

_SELECTED_REQUIRED = {
    "period_of_report", "filed_at", "form_type", "accession",
    "holdings_count", "manager_json_path", "raw_xml_path",
}


def validate_sec_13f_index_schema(
    index: Dict[str, Any],
    *,
    expected_as_of_date: Optional[str] = None,
) -> tuple[bool, str]:
    """Validate index.json against the sec_13f_pit_index.v1 contract.

    Returns (ok, detail). Pure function — no I/O.
    """
    # --- Top-level required fields ---
    missing = _REQUIRED_TOP_LEVEL - set(index.keys())
    if missing:
        return False, f"missing top-level fields: {sorted(missing)}"

    # --- schema_version ---
    sv = index["schema_version"]
    if sv != SCHEMA_VERSION:
        return False, f"schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION!r}"

    # --- cache_type ---
    ct = index["cache_type"]
    if ct != CACHE_TYPE:
        return False, f"cache_type mismatch: got {ct!r}, expected {CACHE_TYPE!r}"

    # --- as_of_date format ---
    aod = index["as_of_date"]
    if not isinstance(aod, str) or len(aod) != 10:
        return False, f"as_of_date not ISO date: {aod!r}"

    if expected_as_of_date and aod != expected_as_of_date:
        return False, f"as_of_date mismatch: index={aod}, expected={expected_as_of_date}"

    # --- created_at format (must end with Z) ---
    ca = index.get("created_at", "")
    if not isinstance(ca, str) or not ca.endswith("Z"):
        return False, f"created_at must end with 'Z': {ca!r}"

    # --- elite_only ---
    if not isinstance(index["elite_only"], bool):
        return False, "elite_only must be boolean"

    # --- numeric fields ---
    tm = index["total_managers"]
    mwf = index["managers_with_filing"]
    cov = index["coverage_pct"]

    if not isinstance(tm, int) or tm < 0:
        return False, f"total_managers must be int >= 0: {tm!r}"
    if not isinstance(mwf, int) or mwf < 0:
        return False, f"managers_with_filing must be int >= 0: {mwf!r}"
    if mwf > tm:
        return False, f"managers_with_filing ({mwf}) > total_managers ({tm})"
    if not isinstance(cov, (int, float)) or cov < 0 or cov > 100:
        return False, f"coverage_pct must be in [0, 100]: {cov!r}"

    # --- coverage_pct consistency (allow ±0.2 rounding tolerance) ---
    if tm > 0:
        expected_cov = mwf / tm * 100
        if abs(cov - expected_cov) > 0.2:
            return False, (
                f"coverage_pct inconsistent: {cov} vs expected "
                f"{expected_cov:.1f} ({mwf}/{tm})"
            )

    # --- managers list ---
    managers = index["managers"]
    if not isinstance(managers, list):
        return False, "managers must be a list"

    for i, m in enumerate(managers):
        if not isinstance(m, dict):
            return False, f"managers[{i}] must be a dict"

        for key in ("manager_cik", "selected"):
            if key not in m:
                return False, f"managers[{i}] missing required field: {key}"

        if not isinstance(m["selected"], bool):
            return False, f"managers[{i}].selected must be boolean"

        if m["selected"]:
            missing_sel = _SELECTED_REQUIRED - set(m.keys())
            if missing_sel:
                return False, (
                    f"managers[{i}] (cik={m.get('manager_cik', '?')}) "
                    f"selected=true but missing: {sorted(missing_sel)}"
                )
        else:
            rr = m.get("rejection_reason", "")
            if not rr:
                return False, (
                    f"managers[{i}] (cik={m.get('manager_cik', '?')}) "
                    f"selected=false but rejection_reason is empty"
                )

    # --- determinism: managers sorted by manager_cik ---
    ciks = [m.get("manager_cik", "") for m in managers]
    if ciks != sorted(ciks):
        return False, "managers list not sorted by manager_cik"

    return True, "ok"


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

    # Schema validation
    ok, schema_detail = validate_sec_13f_index_schema(
        index, expected_as_of_date=as_of_date,
    )
    if not ok:
        return {
            "status": "WARN",
            "detail": f"schema invalid: {schema_detail}",
            "value": None,
            "threshold": {"warn_coverage_pct": warn_coverage_pct},
        }

    coverage = index["coverage_pct"]
    managers_ok = index["managers_with_filing"]
    total = index["total_managers"]

    detail_parts = [
        f"coverage={coverage:.1f}%",
        f"({managers_ok}/{total} managers)",
    ]

    value = {"coverage_pct": coverage, "managers_ok": managers_ok, "total": total}
    threshold = {"warn_coverage_pct": warn_coverage_pct}

    if coverage < warn_coverage_pct:
        detail_parts.append(f"below {warn_coverage_pct:.0f}% threshold")
        return {
            "status": "WARN",
            "detail": ", ".join(detail_parts),
            "value": value,
            "threshold": threshold,
        }

    return {
        "status": "PASS",
        "detail": ", ".join(detail_parts),
        "value": value,
        "threshold": threshold,
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
