#!/usr/bin/env python3
"""Standalone PIT input validators for clinical and catalyst data.

Validates that no lookahead bias leaks into historical simulations:
- CTgov trials: no study_first_posted > as_of_date
- Catalyst events: all events have disclosed_at <= as_of_date

Exit codes:
  0 = OK  (no violations)
  2 = WARN (violations found)
  Never exits 1.

Usage:
    python3 scripts/validate_pit_inputs.py \
        --as-of-date 2025-06-30 \
        --ctgov-cache cache/ctgov/trial_records_2025-06-30.json \
        --catalyst-events data/bundles/PIT/2025-06-30/catalyst_events.json \
        --mode strict

    python3 scripts/validate_pit_inputs.py \
        --as-of-date 2025-06-30 \
        --ctgov-cache cache/ctgov/trial_records_2025-06-30.json \
        --out /tmp/pit_validation_2025-06-30.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

VERSION = "1.0.0"
SCHEMA_VERSION = "pit_validation.v1"

_PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_date(s) -> Optional[date]:
    """Parse YYYY-MM-DD or YYYY-MM string to date, or None."""
    if not s:
        return None
    raw = str(s).strip()[:10]
    if len(raw) == 7 and raw[4] == "-":
        raw = raw + "-01"
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file (first 16 chars)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# CTgov PIT validator
# ---------------------------------------------------------------------------

def validate_ctgov_pit(
    trials: List[dict],
    as_of_date: str,
    mode: str = "strict",
    max_examples: int = 20,
) -> Dict[str, Any]:
    """Validate PIT correctness of CTgov trial records.

    Checks:
    - strict: any trial with study_first_posted > as_of_date is a violation
    - strict: any trial with last_update_posted > as_of_date that still
      has potentially unsafe fields is flagged (informational)

    Returns validation result dict.
    """
    as_of = date.fromisoformat(as_of_date[:10])
    total = len(trials)

    first_posted_violations: List[dict] = []
    last_update_violations: List[dict] = []

    for t in trials:
        nct_id = t.get("nct_id", "?")
        ticker = t.get("ticker", "?")

        # Check study_first_posted > as_of_date
        fp = _parse_date(t.get("first_posted") or t.get("study_first_posted"))
        if fp is not None and fp > as_of:
            if len(first_posted_violations) < max_examples:
                first_posted_violations.append({
                    "nct_id": nct_id,
                    "ticker": ticker,
                    "first_posted": str(fp),
                    "as_of_date": as_of_date,
                })

        # Check last_update_posted > as_of_date (informational in strict)
        lup = _parse_date(t.get("last_update_posted"))
        if lup is not None and lup > as_of:
            if len(last_update_violations) < max_examples:
                last_update_violations.append({
                    "nct_id": nct_id,
                    "ticker": ticker,
                    "last_update_posted": str(lup),
                    "as_of_date": as_of_date,
                })

    n_fp = len(first_posted_violations)
    n_lup = len(last_update_violations)

    # In strict mode, first_posted violations are hard violations
    # last_update_posted > as_of is flagged but only a hard violation
    # if the trial was admitted by first_posted alone
    is_clean = (n_fp == 0)
    if mode == "strict":
        is_clean = (n_fp == 0)
    else:
        # degrade: only flag, don't count as violation
        is_clean = True

    return {
        "source": "ctgov",
        "total_records": total,
        "mode": mode,
        "clean": is_clean,
        "first_posted_violations": n_fp,
        "first_posted_examples": first_posted_violations,
        "last_update_future_count": n_lup,
        "last_update_future_examples": last_update_violations,
    }


# ---------------------------------------------------------------------------
# Catalyst PIT validator
# ---------------------------------------------------------------------------

def validate_catalyst_pit(
    events: Dict[str, Any],
    as_of_date: str,
    mode: str = "strict",
    max_examples: int = 20,
) -> Dict[str, Any]:
    """Validate PIT correctness of catalyst event features.

    Checks:
    - strict: every ticker's nearest event must have disclosed_at
    - strict: disclosed_at <= as_of_date

    Args:
        events: Full catalyst_events_pit.v1 dict (with "tickers" key),
                or a flat dict of {ticker: event_dict}.
    """
    as_of = date.fromisoformat(as_of_date[:10])

    # Handle both schema formats
    tickers_data = events.get("tickers", events)
    if not isinstance(tickers_data, dict):
        return {
            "source": "catalyst",
            "total_tickers": 0,
            "mode": mode,
            "clean": True,
            "missing_disclosed_at": 0,
            "future_disclosed_at": 0,
            "missing_examples": [],
            "future_examples": [],
        }

    total = len(tickers_data)
    missing_disclosed_at: List[dict] = []
    future_disclosed_at: List[dict] = []

    for ticker, td in sorted(tickers_data.items()):
        # Skip tickers with no catalyst
        if td.get("catalyst_mode") == "missing":
            continue

        da = td.get("nearest_disclosed_at", "")
        if not da:
            if len(missing_disclosed_at) < max_examples:
                missing_disclosed_at.append({
                    "ticker": ticker,
                    "catalyst_mode": td.get("catalyst_mode", ""),
                    "nearest_event_source": td.get("nearest_event_source", ""),
                })
            continue

        da_date = _parse_date(da)
        if da_date is not None and da_date > as_of:
            if len(future_disclosed_at) < max_examples:
                future_disclosed_at.append({
                    "ticker": ticker,
                    "disclosed_at": str(da_date),
                    "as_of_date": as_of_date,
                    "nearest_event_source": td.get("nearest_event_source", ""),
                })

    n_missing = len(missing_disclosed_at)
    n_future = len(future_disclosed_at)

    if mode == "strict":
        is_clean = (n_missing == 0 and n_future == 0)
    else:
        is_clean = (n_future == 0)

    return {
        "source": "catalyst",
        "total_tickers": total,
        "mode": mode,
        "clean": is_clean,
        "missing_disclosed_at": n_missing,
        "future_disclosed_at": n_future,
        "missing_examples": missing_disclosed_at,
        "future_examples": future_disclosed_at,
    }


# ---------------------------------------------------------------------------
# Combined report builder
# ---------------------------------------------------------------------------

def build_validation_report(
    as_of_date: str,
    mode: str,
    ctgov_result: Optional[Dict] = None,
    catalyst_result: Optional[Dict] = None,
    file_hashes: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Build a combined PIT validation report."""
    all_clean = True
    components = {}

    if ctgov_result is not None:
        components["ctgov"] = ctgov_result
        if not ctgov_result["clean"]:
            all_clean = False

    if catalyst_result is not None:
        components["catalyst"] = catalyst_result
        if not catalyst_result["clean"]:
            all_clean = False

    return {
        "schema_version": SCHEMA_VERSION,
        "version": VERSION,
        "as_of_date": as_of_date,
        "mode": mode,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "overall_clean": all_clean,
        "components": components,
        "file_hashes": file_hashes or {},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate PIT correctness of clinical and catalyst inputs",
    )
    parser.add_argument(
        "--as-of-date", required=True,
        help="As-of date (ISO format)",
    )
    parser.add_argument(
        "--ctgov-cache", default=None,
        help="Path to CTgov trial records cache JSON",
    )
    parser.add_argument(
        "--catalyst-events", default=None,
        help="Path to catalyst events PIT JSON",
    )
    parser.add_argument(
        "--mode", default="strict",
        choices=["strict", "degrade"],
        help="Validation mode (default: strict)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output JSON path (default: output/pit_validation_{date}.json)",
    )
    args = parser.parse_args(argv)

    sys.path.insert(0, str(_PROJECT_ROOT))

    as_of_date = args.as_of_date
    mode = args.mode

    ctgov_result = None
    catalyst_result = None
    file_hashes: Dict[str, str] = {}

    # Validate CTgov
    if args.ctgov_cache:
        ctgov_path = Path(args.ctgov_cache)
        if not ctgov_path.is_absolute():
            ctgov_path = _PROJECT_ROOT / ctgov_path

        if ctgov_path.exists():
            with open(ctgov_path) as f:
                trials = json.load(f)
            ctgov_result = validate_ctgov_pit(trials, as_of_date, mode)
            file_hashes["ctgov_cache"] = _sha256_file(ctgov_path)

            if not ctgov_result["clean"]:
                n_viol = ctgov_result["first_posted_violations"]
                print(
                    f"::warning::CTgov PIT: {n_viol} first_posted violations "
                    f"for as_of_date={as_of_date}",
                    file=sys.stderr,
                )
            print(f"CTgov: {ctgov_result['total_records']} records, "
                  f"{ctgov_result['first_posted_violations']} first_posted violations, "
                  f"{ctgov_result['last_update_future_count']} last_update future")
        else:
            print(f"::warning::CTgov cache not found: {ctgov_path}", file=sys.stderr)

    # Validate catalyst
    if args.catalyst_events:
        cat_path = Path(args.catalyst_events)
        if not cat_path.is_absolute():
            cat_path = _PROJECT_ROOT / cat_path

        if cat_path.exists():
            with open(cat_path) as f:
                events = json.load(f)
            catalyst_result = validate_catalyst_pit(events, as_of_date, mode)
            file_hashes["catalyst_events"] = _sha256_file(cat_path)

            if not catalyst_result["clean"]:
                n_miss = catalyst_result["missing_disclosed_at"]
                n_fut = catalyst_result["future_disclosed_at"]
                print(
                    f"::warning::Catalyst PIT: {n_miss} missing disclosed_at, "
                    f"{n_fut} future disclosed_at for as_of_date={as_of_date}",
                    file=sys.stderr,
                )
            print(f"Catalyst: {catalyst_result['total_tickers']} tickers, "
                  f"{catalyst_result['missing_disclosed_at']} missing disclosed_at, "
                  f"{catalyst_result['future_disclosed_at']} future disclosed_at")
        else:
            print(f"::warning::Catalyst events not found: {cat_path}", file=sys.stderr)

    # Build report
    report = build_validation_report(
        as_of_date, mode, ctgov_result, catalyst_result, file_hashes,
    )

    # Write output
    out_path = Path(args.out) if args.out else (
        _PROJECT_ROOT / "output" / f"pit_validation_{as_of_date}.json"
    )
    if not out_path.is_absolute():
        out_path = _PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport: {out_path}")
    print(f"Overall: {'CLEAN' if report['overall_clean'] else 'VIOLATIONS FOUND'}")

    if not report["overall_clean"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
