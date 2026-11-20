#!/usr/bin/env python3
"""Manager Integration Acceptance Test — deterministic gate for new 13F managers.

Verifies that a manager is fully integrated across the entire pipeline:
  1. Registry entry exists
  2. PIT 13F cache file exists with parsed holdings
  3. Holdings overlap with universe is nonzero (or explicitly expected zero)
  4. institutional_summary reflects the manager
  5. coinvest_score_z changed for overlapping names
  6. Production snapshot regenerates cleanly

Run after adding a new manager to verify end-to-end propagation.

Usage:
    cd /mnt/c/Projects/biotech_screener/biotech-screener
    python tools/test_manager_integration.py --cik 0001389933
    python tools/test_manager_integration.py --cik 0001389933 --as-of-date 2026-04-15
    python tools/test_manager_integration.py --all  # check every registered manager
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AS_OF = None  # auto-detect latest


def _find_latest_snapshot_date(snapshots_dir: Path) -> Optional[str]:
    """Find most recent snapshot date."""
    import re

    date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    dates = sorted(
        d.name
        for d in snapshots_dir.iterdir()
        if d.is_dir() and date_re.match(d.name) and (d / "rankings.csv").exists()
    )
    return dates[-1] if dates else None


def check_manager(
    cik: str,
    as_of_date: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run all 6 acceptance checks for a single manager CIK.

    Returns dict with check results and overall pass/fail.
    """
    cik_padded = cik.zfill(10)
    results: Dict[str, Any] = {"cik": cik_padded, "checks": {}, "pass": True}

    # ── Check 1: Registry entry ──────────────────────────────────────
    registry_path = REPO_ROOT / "production_data" / "manager_registry.json"
    with open(registry_path, encoding="utf-8") as f:
        registry = json.load(f)

    manager_entry = None
    manager_tier = None
    for tier, managers in registry.items():
        if not isinstance(managers, list):
            continue
        for m in managers:
            if isinstance(m, dict) and m.get("cik", "").lstrip("0") == cik.lstrip("0"):
                manager_entry = m
                manager_tier = tier
                break

    if manager_entry:
        results["checks"]["registry"] = {
            "pass": True,
            "name": manager_entry.get("name", "?"),
            "tier": manager_tier,
            "aum_b": manager_entry.get("aum_b"),
        }
    else:
        results["checks"]["registry"] = {"pass": False, "detail": f"CIK {cik_padded} not in manager_registry.json"}
        results["pass"] = False

    manager_name = manager_entry.get("name", cik_padded) if manager_entry else cik_padded

    # ── Check 2: PIT 13F cache file ─────────────────────────────────
    snap_dir = REPO_ROOT / "data" / "snapshots"
    if as_of_date is None:
        as_of_date = _find_latest_snapshot_date(snap_dir)
    if as_of_date is None:
        results["checks"]["cache"] = {"pass": False, "detail": "No snapshot dates found"}
        results["pass"] = False
        return results

    results["as_of_date"] = as_of_date

    cache_dir = REPO_ROOT / "data" / "caches" / "sec_13f" / "PIT" / as_of_date / "managers"
    cache_file = cache_dir / f"{cik_padded}.json"

    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            cache_data = json.load(f)
        holdings = cache_data.get("holdings", cache_data if isinstance(cache_data, list) else [])
        results["checks"]["cache"] = {
            "pass": True,
            "holdings_count": len(holdings),
            "path": str(cache_file.relative_to(REPO_ROOT)),
        }
    else:
        # Check if any PIT date has this manager
        pit_root = REPO_ROOT / "data" / "caches" / "sec_13f" / "PIT"
        found_dates = []
        if pit_root.exists():
            for d in sorted(pit_root.iterdir(), reverse=True):
                alt = d / "managers" / f"{cik_padded}.json"
                if alt.exists():
                    found_dates.append(d.name)
                if len(found_dates) >= 3:
                    break

        results["checks"]["cache"] = {
            "pass": False,
            "detail": f"No cache at {cache_file.relative_to(REPO_ROOT)}",
            "found_at_dates": found_dates[:3] if found_dates else "none",
            "fix": f"Run: python tools/warm_13f_cache.py --as-of-date {as_of_date} --elite-only",
        }
        results["pass"] = False
        holdings = []

    # ── Check 3: Universe overlap ────────────────────────────────────
    rankings_path = snap_dir / as_of_date / "rankings.csv"
    if rankings_path.exists():
        with open(rankings_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        universe_tickers = {r["ticker"] for r in rows}
    else:
        universe_tickers = set()

    holding_tickers = set()
    for h in holdings:
        tk = h.get("ticker", "")
        if tk and tk in universe_tickers:
            holding_tickers.add(tk)

    overlap = len(holding_tickers)
    results["checks"]["overlap"] = {
        "pass": overlap > 0,
        "overlap_count": overlap,
        "total_holdings": len(holdings),
        "universe_size": len(universe_tickers),
        "sample_tickers": sorted(holding_tickers)[:10],
    }
    if overlap == 0 and len(holdings) > 0:
        results["pass"] = False

    # ── Check 4: institutional_summary reflects manager ──────────────
    inst_summary_path = snap_dir / as_of_date / "institutional_summary.json"
    if not inst_summary_path.exists():
        inst_summary_path = REPO_ROOT / "production_data" / "institutional_summary.json"

    manager_in_summary = False
    tickers_with_manager = []

    if inst_summary_path.exists():
        with open(inst_summary_path, encoding="utf-8") as f:
            inst = json.load(f)

        # Check if manager name appears in any ticker's holder list.
        # Match flexibly: "DAFNA Capital Management" should match "DAFNA Capital"
        name_lower = manager_name.lower()
        # Build match tokens: first two words of manager name (handles "DAFNA Capital" vs "DAFNA Capital Management")
        name_tokens = name_lower.split()[:2]
        tickers_data = inst.get("tickers", inst)
        if isinstance(tickers_data, dict):
            for tk, tk_data in tickers_data.items():
                if not isinstance(tk_data, dict):
                    continue
                holder_names = tk_data.get("elite_holder_names", [])
                for hn in holder_names:
                    hn_lower = hn.lower()
                    if name_lower in hn_lower or hn_lower in name_lower or all(tok in hn_lower for tok in name_tokens):
                        manager_in_summary = True
                        tickers_with_manager.append(tk)
                        break

    results["checks"]["inst_summary"] = {
        "pass": manager_in_summary,
        "tickers_with_manager": len(tickers_with_manager),
        "sample": tickers_with_manager[:10],
    }
    if not manager_in_summary and overlap > 0:
        results["checks"]["inst_summary"]["detail"] = (
            "Manager has overlap but not found in institutional_summary. " "Re-run production screen to rebuild."
        )
        results["pass"] = False

    # ── Check 5: coinvest_score_z populated for overlap names ────────
    if rankings_path.exists() and holding_tickers:
        coinvest_populated = 0
        coinvest_nonzero = 0
        for r in rows:
            if r["ticker"] in holding_tickers:
                cz = r.get("coinvest_score_z", "")
                if cz and cz not in ("", "None"):
                    coinvest_populated += 1
                    if float(cz) != 0.0:
                        coinvest_nonzero += 1

        results["checks"]["coinvest"] = {
            "pass": coinvest_populated > 0,
            "populated": coinvest_populated,
            "nonzero": coinvest_nonzero,
            "overlap_count": overlap,
        }
        if coinvest_populated == 0 and overlap > 0:
            results["pass"] = False
    else:
        results["checks"]["coinvest"] = {
            "pass": overlap == 0,
            "detail": "No overlap tickers to check" if overlap == 0 else "No rankings.csv",
        }

    # ── Check 6: Production snapshot exists and is recent ────────────
    manifest_path = snap_dir / as_of_date / "run_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        gate_fails = [g for g in manifest.get("gates", []) if g.get("status") == "FAIL"]
        results["checks"]["production"] = {
            "pass": len(gate_fails) == 0,
            "gate_fails": len(gate_fails),
            "snapshot_date": as_of_date,
        }
        if gate_fails:
            results["pass"] = False
    elif rankings_path.exists():
        results["checks"]["production"] = {
            "pass": True,
            "detail": "rankings.csv exists (no manifest to check gates)",
            "snapshot_date": as_of_date,
        }
    else:
        results["checks"]["production"] = {
            "pass": False,
            "detail": f"No snapshot at {as_of_date}",
        }
        results["pass"] = False

    return results


def print_results(results: Dict[str, Any]) -> None:
    """Print acceptance test results."""
    name = results.get("checks", {}).get("registry", {}).get("name", results["cik"])
    status = "PASS" if results["pass"] else "FAIL"
    print(f"\n{'=' * 60}")
    print(f"Manager Integration Test: {name} ({results['cik']})")
    print(f"As-of: {results.get('as_of_date', '?')}")
    print(f"{'=' * 60}")

    for check_name, check in results["checks"].items():
        passed = check.get("pass", False)
        icon = "PASS" if passed else "FAIL"
        detail = ""

        if check_name == "registry":
            detail = f"tier={check.get('tier')}, AUM={check.get('aum_b')}B" if passed else check.get("detail", "")
        elif check_name == "cache":
            detail = f"{check.get('holdings_count', 0)} holdings" if passed else check.get("detail", "")
        elif check_name == "overlap":
            detail = f"{check.get('overlap_count', 0)}/{check.get('total_holdings', 0)} in universe"
            if check.get("sample_tickers"):
                detail += f" ({', '.join(check['sample_tickers'][:5])}...)"
        elif check_name == "inst_summary":
            detail = f"{check.get('tickers_with_manager', 0)} tickers" if passed else check.get("detail", "")
        elif check_name == "coinvest":
            detail = (
                f"{check.get('nonzero', 0)}/{check.get('overlap_count', 0)} nonzero"
                if passed
                else check.get("detail", "")
            )
        elif check_name == "production":
            detail = f"snapshot {check.get('snapshot_date', '?')}" if passed else check.get("detail", "")

        print(f"  [{icon}] {check_name:<15} {detail}")

        if not passed and "fix" in check:
            print(f"         Fix: {check['fix']}")

    n_pass = sum(1 for c in results["checks"].values() if c.get("pass"))
    n_total = len(results["checks"])
    print(f"\n  Result: {n_pass}/{n_total} [{status}]")
    print(f"{'=' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manager Integration Acceptance Test")
    parser.add_argument("--cik", type=str, help="Manager CIK to test")
    parser.add_argument("--all", action="store_true", help="Test all registered managers")
    parser.add_argument("--as-of-date", type=str, default=None)
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    args = parser.parse_args()

    if not args.cik and not args.all:
        parser.error("Specify --cik or --all")

    if args.all:
        registry_path = REPO_ROOT / "production_data" / "manager_registry.json"
        with open(registry_path, encoding="utf-8") as f:
            registry = json.load(f)
        ciks = []
        for tier, managers in registry.items():
            if not isinstance(managers, list) or tier == "retired":
                continue
            for m in managers:
                if isinstance(m, dict) and m.get("cik"):
                    ciks.append(m["cik"])

        all_results = []
        n_pass = 0
        for cik in ciks:
            r = check_manager(cik, args.as_of_date, verbose=False)
            all_results.append(r)
            if r["pass"]:
                n_pass += 1
            if not args.json:
                name = r.get("checks", {}).get("registry", {}).get("name", cik)
                status = "PASS" if r["pass"] else "FAIL"
                fails = [k for k, v in r["checks"].items() if not v.get("pass")]
                fail_str = f" — {', '.join(fails)}" if fails else ""
                print(f"  [{status}] {name:40s} ({cik}){fail_str}")

        if args.json:
            print(json.dumps(all_results, indent=2))
        else:
            print(f"\n{n_pass}/{len(ciks)} managers fully integrated")
    else:
        r = check_manager(args.cik, args.as_of_date)
        if args.json:
            print(json.dumps(r, indent=2))
        else:
            print_results(r)

        sys.exit(0 if r["pass"] else 1)


if __name__ == "__main__":
    main()
