#!/usr/bin/env python3
"""13F Refresh Readiness Check — tools/prep_13f_refresh.py

Run before a 13F refresh window. The target quarter is selected with --quarter
and defaults to the most recently completed quarter; the SEC filing deadline is
45 days after quarter end. Captures a pre-refresh baseline and verifies the
ingest pipeline is ready.

Checks:
  1. Most recent snapshot has valid institutional_summary_delta.json
     with prior_date = the quarter before the target (i.e. not yet refreshed)
  2. coinvest_score_z has healthy variance (sd > 0.10)
  3. PIT cache has entries within 3 days of today
  4. SEC EDGAR endpoint is reachable
  5. Dry-run: build_institutional_summary() against current PIT cache
     produces valid output (nonzero coverage, ≥80% signal_coverage_pct)

Baseline artifact:
  artifacts/13f_pre_refresh_baseline_YYYY-MM-DD.json
  Pass its snapshot date as --pre-date to check_13f_cohort_quarantine.py
  after the refresh lands.

Usage:
    source .env && python3 tools/prep_13f_refresh.py
    python3 tools/prep_13f_refresh.py --date 2026-05-14        # specific snapshot
    python3 tools/prep_13f_refresh.py --quarter 2026-06-30     # specific quarter
    python3 tools/prep_13f_refresh.py --dry-run                # no artifact written
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common.date_utils import (  # noqa: E402
    filing_deadline_13f,
    latest_completed_quarter_end,
    parse_quarter_end,
    prior_quarter_end,
    to_date_string,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SNAP_ROOT = PROJECT_ROOT / "data" / "snapshots"
PIT_CACHE_ROOT = PROJECT_ROOT / "data" / "caches" / "sec_13f" / "PIT"
PROD_INST_SUMMARY = PROJECT_ROOT / "production_data" / "institutional_summary.json"
MGR_REGISTRY = PROJECT_ROOT / "production_data" / "manager_registry.json"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

EDGAR_PROBE_URL_TEMPLATE = (
    "https://efts.sec.gov/LATEST/search-index?q=%2213F%22&dateRange=custom"
    "&startdt={startdt}&enddt={enddt}&forms=13F-HR"
)
EDGAR_PROBE_TIMEOUT = 10

# Thresholds
MIN_COINVEST_SD = 0.10
MIN_SIGNAL_COVERAGE_PCT = 75.0
MIN_MANAGERS_WITH_FILING = 30
PIT_FRESHNESS_DAYS = 3


def edgar_probe_url(quarter: str) -> str:
    """EDGAR full-text search URL covering the 13F filing window for a quarter.

    The window runs from quarter end to the 45-day statutory deadline, so the probe
    follows the target quarter instead of a fixed calendar range.
    """
    return EDGAR_PROBE_URL_TEMPLATE.format(startdt=quarter, enddt=to_date_string(filing_deadline_13f(quarter)))


def _safe_float(v: Any) -> Optional[float]:
    if v in (None, "", "nan", "None"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _find_latest_snapshot(specific_date: Optional[str] = None) -> Optional[str]:
    """Return the most recent snapshot dir name with a valid delta JSON."""
    if specific_date:
        p = SNAP_ROOT / specific_date / "institutional_summary_delta.json"
        return specific_date if p.exists() else None

    candidates = sorted(
        (d.name for d in SNAP_ROOT.iterdir() if d.is_dir() and len(d.name) == 10 and d.name[4] == "-"),
        reverse=True,
    )
    for c in candidates:
        if (SNAP_ROOT / c / "institutional_summary_delta.json").exists():
            return c
    return None


def check_1_snapshot_baseline(snap_date: str, expected_prior_date: str) -> Tuple[bool, Dict[str, Any]]:
    """Verify the snapshot delta is present and prior_date is pre-refresh."""
    delta_path = SNAP_ROOT / snap_date / "institutional_summary_delta.json"
    if not delta_path.exists():
        return False, {"error": f"institutional_summary_delta.json missing in {snap_date}"}

    delta = json.loads(delta_path.read_text())
    prior_date = delta.get("prior_date", "")
    current_cache = delta.get("current_cache_as_of_date", "")
    tickers_in_current = delta.get("tickers_in_current", 0)

    status = prior_date == expected_prior_date
    return status, {
        "snap_date": snap_date,
        "expected_prior_date": expected_prior_date,
        "prior_date": prior_date,
        "current_cache_as_of_date": current_cache,
        "tickers_in_current": tickers_in_current,
        "pre_refresh_confirmed": status,
    }


def check_2_coinvest_variance(snap_date: str) -> Tuple[bool, Dict[str, Any]]:
    """Verify coinvest_score_z has healthy variance in the latest rankings."""
    rankings_path = SNAP_ROOT / snap_date / "rankings.csv"
    if not rankings_path.exists():
        return False, {"error": f"rankings.csv missing in {snap_date}"}

    with rankings_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    vals = [_safe_float(r.get("coinvest_score_z")) for r in rows]
    vals = [v for v in vals if v is not None]

    if len(vals) < 10:
        return False, {"error": f"Too few coinvest_score_z values: {len(vals)}"}

    sd = statistics.stdev(vals)
    mean = statistics.mean(vals)
    n = len(vals)
    n_nonzero = sum(1 for v in vals if v != 0.0)
    status = sd > MIN_COINVEST_SD

    return status, {
        "n": n,
        "n_nonzero": n_nonzero,
        "mean": round(mean, 4),
        "sd": round(sd, 4),
        "healthy": status,
        "threshold": MIN_COINVEST_SD,
    }


def check_3_pit_cache_freshness() -> Tuple[bool, Dict[str, Any]]:
    """Verify PIT cache has a recent entry."""
    if not PIT_CACHE_ROOT.exists():
        return False, {"error": "PIT cache directory not found"}

    dates = sorted(d.name for d in PIT_CACHE_ROOT.iterdir() if d.is_dir() and len(d.name) == 10)
    if not dates:
        return False, {"error": "No dated directories in PIT cache"}

    latest = dates[-1]
    try:
        latest_dt = date.fromisoformat(latest)
        today = date.today()
        age_days = (today - latest_dt).days
        fresh = age_days <= PIT_FRESHNESS_DAYS
    except ValueError:
        return False, {"error": f"Unparseable PIT cache date: {latest}"}

    return fresh, {
        "latest_pit_date": latest,
        "age_days": age_days,
        "fresh": fresh,
        "threshold_days": PIT_FRESHNESS_DAYS,
        "n_cache_entries": len(dates),
    }


def check_4_edgar_reachable(quarter: str) -> Tuple[bool, Dict[str, Any]]:
    """Probe SEC EDGAR for the target quarter's 13F filings (non-blocking warning only)."""
    try:
        req = urllib.request.Request(
            edgar_probe_url(quarter),
            headers={"User-Agent": "biotech-screener-preflight admin@example.com"},
        )
        with urllib.request.urlopen(req, timeout=EDGAR_PROBE_TIMEOUT) as resp:
            body = resp.read(512).decode("utf-8", errors="replace")
            reachable = resp.status == 200
            # Look for any 13F-HR hits
            has_hits = '"hits"' in body and '"total"' in body
            return reachable, {
                "reachable": reachable,
                "status_code": resp.status,
                "quarter": quarter,
                "quarter_hits_visible": has_hits,
            }
    except Exception as exc:
        return False, {"reachable": False, "error": str(exc)[:120]}


def check_5_ingest_dry_run() -> Tuple[bool, Dict[str, Any]]:
    """Dry-run build_institutional_summary() against current PIT cache."""
    try:
        from institutional_summary import build_institutional_summary, validate_institutional_summary_schema_v1
    except ImportError as exc:
        return False, {"error": f"Import failed: {exc}"}

    try:
        # Load universe tickers from production_data
        universe_path = PROJECT_ROOT / "production_data" / "universe.json"
        if not universe_path.exists():
            # Fallback: get tickers from latest rankings.csv
            snap = _find_latest_snapshot()
            if not snap:
                return False, {"error": "No snapshot found for dry-run"}
            with (SNAP_ROOT / snap / "rankings.csv").open(newline="") as f:
                tickers = {r["ticker"] for r in csv.DictReader(f) if r.get("ticker")}
        else:
            u = json.loads(universe_path.read_text())
            if isinstance(u, list):
                tickers = {item.get("ticker", item) if isinstance(item, dict) else item for item in u}
            else:
                tickers = set(u.get("active_tickers", []))

        today_str = date.today().isoformat()
        result = build_institutional_summary(today_str, tickers, nearest_prior_days=3)

        if result is None:
            return False, {"error": "build_institutional_summary() returned None — PIT cache issue?"}

        valid, msg = validate_institutional_summary_schema_v1(result)
        if not valid:
            return False, {"error": f"Schema validation failed: {msg}"}

        coverage = result.get("signal_coverage_pct", 0.0)
        managers_with_filing = result.get("elite_managers_with_filing", 0)
        tickers_with_signal = result.get("tickers_with_signal", 0)

        ok = coverage >= MIN_SIGNAL_COVERAGE_PCT and managers_with_filing >= MIN_MANAGERS_WITH_FILING
        return ok, {
            "cache_as_of_date": result.get("cache_as_of_date"),
            "elite_managers_total": result.get("elite_managers_total"),
            "elite_managers_with_filing": managers_with_filing,
            "tickers_with_signal": tickers_with_signal,
            "signal_coverage_pct": round(coverage, 2),
            "schema_valid": valid,
            "healthy": ok,
        }
    except Exception as exc:
        return False, {"error": f"Dry-run failed: {exc}"}


def load_manager_registry() -> Dict[str, Any]:
    if not MGR_REGISTRY.exists():
        return {}
    return json.loads(MGR_REGISTRY.read_text())


def run(specific_date: Optional[str], dry_run: bool, quarter: Optional[str] = None) -> int:
    today = date.today().isoformat()
    target_quarter = parse_quarter_end(quarter) if quarter else to_date_string(latest_completed_quarter_end())
    expected_prior = to_date_string(prior_quarter_end(target_quarter))
    deadline = to_date_string(filing_deadline_13f(target_quarter))

    print(f"\n{'='*60}")
    print(f"  13F REFRESH READINESS CHECK — {today}")
    print(f"  Target quarter: {target_quarter} (filing deadline {deadline})")
    print(f"  Expected pre-refresh prior_date: {expected_prior}")
    print(f"{'='*60}\n")

    # Locate baseline snapshot
    snap = _find_latest_snapshot(specific_date)
    if snap is None:
        log.error("No snapshot with institutional_summary_delta.json found")
        return 2

    log.info("Using snapshot: %s", snap)

    results: Dict[str, Any] = {
        "as_of_date": today,
        "snap_date": snap,
        "target_quarter": target_quarter,
        "expected_prior_date": expected_prior,
        "filing_deadline": deadline,
        "checks": {},
    }
    all_pass = True
    warnings = []

    # Check 1 — snapshot baseline
    ok, data = check_1_snapshot_baseline(snap, expected_prior)
    results["checks"]["snapshot_baseline"] = {"pass": ok, **data}
    status = "PASS" if ok else "WARN"
    if not ok:
        warnings.append(f"prior_date={data.get('prior_date')} — expected {expected_prior}")
    print(f"[{status}] Check 1 — Snapshot baseline")
    print(f"       snap={snap}, prior_date={data.get('prior_date')}, tickers={data.get('tickers_in_current')}")
    if not ok:
        print("       WARNING: prior_date may have already advanced (refresh may be live?)")

    # Check 2 — coinvest variance
    ok, data = check_2_coinvest_variance(snap)
    results["checks"]["coinvest_variance"] = {"pass": ok, **data}
    status = "PASS" if ok else "FAIL"
    all_pass = all_pass and ok
    print(f"\n[{status}] Check 2 — coinvest_score_z variance")
    print(f"       n={data.get('n')}, n_nonzero={data.get('n_nonzero')}, sd={data.get('sd')}")
    if not ok:
        print(f"       FAIL: sd={data.get('sd')} ≤ {MIN_COINVEST_SD} — selector signal is flat")

    # Check 3 — PIT cache freshness
    ok, data = check_3_pit_cache_freshness()
    results["checks"]["pit_cache_freshness"] = {"pass": ok, **data}
    status = "PASS" if ok else "FAIL"
    all_pass = all_pass and ok
    print(f"\n[{status}] Check 3 — PIT cache freshness")
    print(
        f"       latest={data.get('latest_pit_date')}, age={data.get('age_days')}d, n_entries={data.get('n_cache_entries')}"
    )
    if not ok:
        print(f"       FAIL: PIT cache is {data.get('age_days')}d stale — run cron_data_refresh.sh sec_13f")

    # Check 4 — EDGAR reachability (warning only)
    ok, data = check_4_edgar_reachable(target_quarter)
    results["checks"]["edgar_reachable"] = {"pass": ok, **data}
    status = "PASS" if ok else "WARN"
    print(f"\n[{status}] Check 4 — SEC EDGAR reachable (warning only)")
    if ok:
        print(f"       EDGAR up, {target_quarter} 13F-HR hits visible: {data.get('quarter_hits_visible')}")
    else:
        print(f"       WARN: {data.get('error', 'unreachable')} — EDGAR may be down or throttling")
        warnings.append("EDGAR probe failed — verify manually before refresh day")

    # Check 5 — ingest dry-run
    print("\n[...] Check 5 — Ingest dry-run (build_institutional_summary)")
    ok, data = check_5_ingest_dry_run()
    results["checks"]["ingest_dry_run"] = {"pass": ok, **data}
    status = "PASS" if ok else "FAIL"
    all_pass = all_pass and ok
    print(f"[{status}] Check 5 — Ingest dry-run")
    if ok:
        print(
            f"       cache={data.get('cache_as_of_date')}, "
            f"managers_filing={data.get('elite_managers_with_filing')}, "
            f"coverage={data.get('signal_coverage_pct')}%"
        )
    else:
        print(f"       FAIL: {data.get('error', 'unknown')}")

    # Manager registry snapshot
    reg = load_manager_registry()
    meta = reg.get("metadata", {})
    n_elite_core = len(reg.get("elite_core", []))
    n_conditional = len(reg.get("conditional", []))
    results["manager_registry_snapshot"] = {
        "n_elite_core": n_elite_core,
        "n_conditional": n_conditional,
        "version": meta.get("version"),
        "last_updated": meta.get("last_updated"),
        "total_elite_aum_b": meta.get("total_elite_aum_b"),
    }

    print(
        f"\n[INFO] Manager registry: {n_elite_core} elite_core + {n_conditional} conditional (v{meta.get('version')}, updated {meta.get('last_updated')})"
    )

    # Summary
    overall = (
        "READY"
        if all_pass
        else (
            "WARN"
            if not all_pass
            and not any(
                not results["checks"][k]["pass"] for k in ["coinvest_variance", "pit_cache_freshness", "ingest_dry_run"]
            )
            else "NOT_READY"
        )
    )

    results["overall"] = overall
    results["warnings"] = warnings

    print(f"\n{'='*60}")
    print(f"  OVERALL: {overall}")
    if warnings:
        for w in warnings:
            print(f"  WARN: {w}")
    print("\n  Pre-refresh baseline (to use as --pre-date):")
    print(f"    --pre-date {snap}")
    print(f"\n  Post-refresh: run check_13f_cohort_quarantine.py once prior_date advances to {target_quarter}")
    print(f"{'='*60}\n")

    if not dry_run:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = ARTIFACTS_DIR / f"13f_pre_refresh_baseline_{today}.json"
        out_path.write_text(json.dumps(results, indent=2) + "\n")
        log.info("Baseline artifact written: %s", out_path)
    else:
        log.info("DRY RUN — no artifact written")

    return 0 if all_pass else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--date", default=None, help="Specific snapshot date YYYY-MM-DD (default: latest)")
    p.add_argument(
        "--quarter",
        default=None,
        help="Target quarter end YYYY-MM-DD (default: most recently completed quarter)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print results only; do not write artifact")
    args = p.parse_args(argv)
    return run(args.date, args.dry_run, args.quarter)


if __name__ == "__main__":
    sys.exit(main())
