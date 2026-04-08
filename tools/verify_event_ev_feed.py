#!/usr/bin/env python3
"""Event EV Feed Health Verifier (Spec 062).

Checks that Event EV data is flowing correctly through the pipeline:
  1. Daily EV artifacts exist in artifacts/event_ev/
  2. rankings.csv has event_ev_* columns populated
  3. Expectation model input features are present in snapshots
  4. Artifact date matches snapshot date (no staleness)

Exit codes:
  0 = healthy
  1 = degraded (partial coverage or stale)
  2 = missing (no EV data at all)

Usage:
    python tools/verify_event_ev_feed.py --as-of 2026-04-08
    python tools/verify_event_ev_feed.py  # auto-detect latest
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("verify_ev_feed")

SNAPSHOT_DIR = REPO_ROOT / "data" / "snapshots"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "event_ev"

# Expectation model input features that should appear in rankings.csv
# (wired by Spec 060 market data export)
EXPECTATION_FEATURES = [
    "short_interest_pct",
    "close_price",
    "market_cap_mm",
]

# Event EV columns that should be populated
EV_COLUMNS = [
    "event_ev_score",
    "event_ev_score_z",
    "event_ev_bucket",
    "event_ev_analog_confidence",
]


def _find_latest_snapshot() -> Optional[str]:
    """Find the most recent snapshot date."""
    if not SNAPSHOT_DIR.exists():
        return None
    dates = sorted(
        [d.name for d in SNAPSHOT_DIR.iterdir() if d.is_dir() and len(d.name) == 10],
        reverse=True,
    )
    return dates[0] if dates else None


def verify_feed(as_of_date: str) -> Dict[str, Any]:
    """Run all feed health checks.

    Returns:
        Dict with status ("healthy" / "degraded" / "missing"), checks, and details.
    """
    result: Dict[str, Any] = {
        "as_of_date": as_of_date,
        "status": "missing",
        "checks": {},
    }

    # --- Check 1: Daily EV artifacts ---
    artifact_path = ARTIFACTS_DIR / f"{as_of_date}_event_ev_scores.json"
    artifact_exists = artifact_path.exists()
    artifact_count = 0
    artifact_tickers: List[str] = []

    if artifact_exists:
        try:
            with open(artifact_path, encoding="utf-8") as f:
                data = json.load(f)
            events = data if isinstance(data, list) else data.get("events", [])
            artifact_count = len(events)
            artifact_tickers = [
                ev.get("node", {}).get("ticker", "")
                for ev in events
                if ev.get("node", {}).get("ticker")
            ]
        except (json.JSONDecodeError, OSError):
            artifact_exists = False

    result["checks"]["artifact"] = {
        "exists": artifact_exists,
        "path": str(artifact_path),
        "n_events": artifact_count,
        "tickers": artifact_tickers[:10],  # first 10 for display
    }

    # --- Check 2: rankings.csv EV columns ---
    rankings_path = SNAPSHOT_DIR / as_of_date / "rankings.csv"
    rankings_exists = rankings_path.exists()
    ev_populated = 0
    ev_total = 0
    ev_column_present = False

    if rankings_exists:
        try:
            with open(rankings_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                ev_column_present = all(c in fieldnames for c in EV_COLUMNS)
                for row in reader:
                    ev_total += 1
                    score = row.get("event_ev_score", "")
                    if score and score != "":
                        ev_populated += 1
        except (OSError, csv.Error):
            rankings_exists = False

    coverage_pct = round(ev_populated / ev_total * 100, 1) if ev_total > 0 else 0.0
    result["checks"]["rankings"] = {
        "exists": rankings_exists,
        "ev_columns_present": ev_column_present,
        "ev_populated": ev_populated,
        "total_rows": ev_total,
        "coverage_pct": coverage_pct,
    }

    # --- Check 3: Expectation model features ---
    features_present: Dict[str, bool] = {}
    features_populated: Dict[str, int] = {}
    if rankings_exists:
        try:
            with open(rankings_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                for feat in EXPECTATION_FEATURES:
                    features_present[feat] = feat in fieldnames
                    features_populated[feat] = 0
                rows = list(reader)
            for row in rows:
                for feat in EXPECTATION_FEATURES:
                    val = row.get(feat, "")
                    if val and val != "" and val != "0":
                        features_populated[feat] = features_populated.get(feat, 0) + 1
        except (OSError, csv.Error):
            pass

    result["checks"]["expectation_features"] = {
        "present": features_present,
        "populated_counts": features_populated,
        "total_rows": ev_total,
    }

    # --- Check 4: Staleness ---
    # Count total daily artifacts to assess accumulation
    all_artifacts = sorted(ARTIFACTS_DIR.glob("*_event_ev_scores.json")) if ARTIFACTS_DIR.exists() else []
    n_daily_artifacts = len(all_artifacts)
    latest_artifact_date = ""
    if all_artifacts:
        try:
            latest_artifact_date = all_artifacts[-1].name[:10]
        except (IndexError, ValueError):
            pass

    stale = latest_artifact_date != as_of_date if latest_artifact_date else True
    result["checks"]["staleness"] = {
        "latest_artifact_date": latest_artifact_date,
        "snapshot_date": as_of_date,
        "is_stale": stale,
        "n_daily_artifacts": n_daily_artifacts,
    }

    # --- Overall status ---
    if not artifact_exists and ev_populated == 0:
        result["status"] = "missing"
    elif stale or coverage_pct < 10.0 or not ev_column_present:
        result["status"] = "degraded"
    else:
        result["status"] = "healthy"

    return result


def format_report(result: Dict[str, Any]) -> str:
    """Format verification result as human-readable text."""
    lines = [f"Event EV Feed Verification — {result['as_of_date']}"]
    lines.append(f"Status: {result['status'].upper()}")
    lines.append("")

    art = result["checks"]["artifact"]
    lines.append(f"Artifact: {'FOUND' if art['exists'] else 'MISSING'} ({art['n_events']} events)")

    rnk = result["checks"]["rankings"]
    lines.append(
        f"Rankings: {'OK' if rnk['ev_columns_present'] else 'MISSING COLUMNS'}, "
        f"{rnk['ev_populated']}/{rnk['total_rows']} rows with EV ({rnk['coverage_pct']}%)"
    )

    feat = result["checks"]["expectation_features"]
    for f_name, present in feat["present"].items():
        count = feat["populated_counts"].get(f_name, 0)
        lines.append(f"  {f_name}: {'present' if present else 'MISSING'} ({count}/{feat['total_rows']} populated)")

    stale = result["checks"]["staleness"]
    lines.append(
        f"Staleness: {'STALE' if stale['is_stale'] else 'FRESH'} "
        f"(latest artifact: {stale['latest_artifact_date'] or 'none'}, "
        f"{stale['n_daily_artifacts']} total artifacts)"
    )

    return "\n".join(lines)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Verify Event EV feed health")
    parser.add_argument("--as-of", default=None, help="Date to verify (default: latest snapshot)")
    args = parser.parse_args()

    as_of = args.as_of or _find_latest_snapshot()
    if not as_of:
        print("No snapshot found and no --as-of provided")
        sys.exit(2)

    result = verify_feed(as_of)
    print(format_report(result))

    # Write JSON artifact
    out_dir = REPO_ROOT / "artifacts" / "event_ev"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{as_of}_feed_health.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"\nWritten to {out_path}")

    # Exit code
    code = {"healthy": 0, "degraded": 1, "missing": 2}
    sys.exit(code.get(result["status"], 2))


if __name__ == "__main__":
    main()
