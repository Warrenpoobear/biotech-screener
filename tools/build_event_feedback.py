#!/usr/bin/env python3
"""Resolved event materializer — joins Herald, CRT, postmortem, and Event EV.

Produces a unified event feedback artifact with one row per adjudicated event.
CRT resolutions are the anchor: Herald records are matched TO resolutions,
not the other way around. Scores against event outcome, not price direction.

Read-only — produces evidence, never updates model priors.

Output:
    artifacts/event_feedback/{date}_resolved_events.jsonl  (one row per event)
    artifacts/event_feedback/{date}_summary.json
    artifacts/event_feedback/ledger.jsonl  (append)

Usage:
    python tools/build_event_feedback.py --as-of-date 2026-04-14
    python tools/build_event_feedback.py --as-of-date 2026-04-14 --backfill
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("event_feedback")

SCHEMA_VERSION = "event_feedback.v1"

RESOLUTION_DIR = REPO_ROOT / "data" / "snapshots" / "resolutions"
CLASSIFIED_DIR = REPO_ROOT / "data" / "press_releases" / "classified"
POSTMORTEM_DIR = REPO_ROOT / "artifacts" / "postmortem"
OVERRIDES_PATH = REPO_ROOT / "production_data" / "crt_manual_overrides.json"
OUTPUT_DIR = REPO_ROOT / "artifacts" / "event_feedback"

# CRT catalyst_type → event_family mapping
_CATALYST_TYPE_TO_FAMILY = {
    "PDUFA_ACTION": "REGULATORY",
    "NDA_BLA_FILING": "REGULATORY",
    "REGULATORY_DESIGNATION": "REGULATORY",
    "ADVISORY_COMMITTEE": "REGULATORY",
    "PHASE_3_READOUT": "CLINICAL",
    "PHASE_2_READOUT": "CLINICAL",
    "PHASE_1_DATA": "CLINICAL",
    "DATA_READOUT": "CLINICAL",
    "CORPORATE_UPDATE": "CORPORATE",
}

# Resolved outcomes only (skip DELAYED, NEEDS_REVIEW)
RESOLVED_OUTCOMES = frozenset({"HIT", "MISS", "MIXED"})

HERALD_MATCH_WINDOW_DAYS = 3


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def load_crt_resolutions(as_of_date: str) -> List[Dict]:
    """Load all CRT resolution records up to as_of_date, applying manual overrides."""
    records = []
    if not RESOLUTION_DIR.exists():
        return records

    overrides = {}
    override_data = _load_json(OVERRIDES_PATH) or _load_json(RESOLUTION_DIR / "manual_overrides.json")
    if isinstance(override_data, list):
        for ov in override_data:
            key = (ov.get("ticker", ""), ov.get("catalyst_date", ""))
            overrides[key] = ov
    elif isinstance(override_data, dict):
        for key_str, ov in override_data.items():
            if isinstance(ov, dict):
                key = (ov.get("ticker", ""), ov.get("catalyst_date", ""))
                overrides[key] = ov

    for month_dir in sorted(RESOLUTION_DIR.iterdir()):
        if not month_dir.is_dir():
            continue
        for f in sorted(month_dir.glob("*.json")):
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            if not rec.get("ticker") or not rec.get("catalyst_date"):
                continue

            # Apply manual override if exists
            key = (rec["ticker"], rec["catalyst_date"])
            if key in overrides:
                ov = overrides[key]
                if ov.get("outcome"):
                    rec["outcome"] = ov["outcome"]
                if ov.get("outcome_detail"):
                    rec["outcome_detail"] = ov["outcome_detail"]
                rec["_override_applied"] = True

            # Only include resolved outcomes
            if rec.get("outcome") not in RESOLVED_OUTCOMES:
                continue

            # Filter by as_of_date: resolution must have occurred by then
            res_date = rec.get("resolution_date") or rec.get("catalyst_date", "")
            if res_date > as_of_date:
                continue

            records.append(rec)

    return records


def load_classified_index(max_days: int = 90) -> Dict[str, List[Dict]]:
    """Load Herald classified records, indexed by ticker for fast lookup."""
    index: Dict[str, List[Dict]] = defaultdict(list)
    if not CLASSIFIED_DIR.exists():
        return index

    today = date.today()
    for f in sorted(CLASSIFIED_DIR.glob("classified_*.jsonl")):
        # Extract date from filename
        try:
            file_date = f.stem.replace("classified_", "")
            fd = date.fromisoformat(file_date)
        except ValueError:
            continue
        if (today - fd).days > max_days:
            continue

        try:
            for line in f.read_text(encoding="utf-8").strip().split("\n"):
                if not line.strip():
                    continue
                rec = json.loads(line)
                ticker = (rec.get("ticker") or "").upper()
                if ticker:
                    # Fallback: if published_at_utc is empty, use file date
                    if not rec.get("published_at_utc"):
                        rec["published_at_utc"] = file_date
                    index[ticker].append(rec)
        except (json.JSONDecodeError, OSError):
            continue

    return index


def load_postmortem_index(as_of_date: str) -> Dict[Tuple[str, str], Dict]:
    """Load postmortems indexed by (ticker, event_date)."""
    index: Dict[Tuple[str, str], Dict] = {}
    if not POSTMORTEM_DIR.exists():
        return index

    for date_dir in sorted(POSTMORTEM_DIR.iterdir()):
        if not date_dir.is_dir() or date_dir.name > as_of_date:
            continue
        for pm_file in date_dir.glob("*.json"):
            try:
                rec = json.loads(pm_file.read_text(encoding="utf-8"))
                if rec.get("schema") != "postmortem.v1":
                    continue
                key = (rec.get("ticker", ""), rec.get("event_date", ""))
                if key[0] and key[1]:
                    index[key] = rec
            except (json.JSONDecodeError, OSError):
                continue

    return index


def match_herald_to_resolution(
    resolution: Dict,
    classified_index: Dict[str, List[Dict]],
) -> Optional[Dict]:
    """Find the best Herald classified record matching a CRT resolution.

    Match by ticker + date within ±3 days. Keep highest-confidence match.
    """
    ticker = resolution.get("ticker", "")
    cat_date_str = resolution.get("catalyst_date", "")

    if not ticker or not cat_date_str:
        return None

    candidates = classified_index.get(ticker, [])
    if not candidates:
        return None

    try:
        cat_d = date.fromisoformat(cat_date_str)
    except ValueError:
        return None

    best = None
    best_conf = -1.0

    for rec in candidates:
        pub = (rec.get("published_at_utc") or "")[:10]
        if not pub:
            continue
        try:
            pub_d = date.fromisoformat(pub)
        except ValueError:
            continue

        if abs((pub_d - cat_d).days) <= HERALD_MATCH_WINDOW_DAYS:
            # Prefer catalyst-relevant categories but accept any match
            # since CRT resolution is the authority on relevance
            cat = rec.get("event_category", "")
            cat_bonus = 1.0 if cat in ("clinical", "regulatory", "safety") else 0.0

            conf = rec.get("confidence", 0.0) + cat_bonus
            if conf > best_conf:
                best = rec
                best_conf = conf

    return best


def _map_source_class(resolution: Dict, herald_match: Optional[Dict]) -> str:
    """Determine source class from resolution and herald data."""
    src = resolution.get("source_type", "")
    if src == "MANUAL":
        return "MANUAL_REVIEW"
    if src == "PRESS_RELEASE":
        return "OFFICIAL_COMPANY_IR"
    if src in ("SEC_8K", "SEC_FILING"):
        return "SEC_FILING"
    if herald_match:
        hsrc = herald_match.get("source_type", "")
        if hsrc == "company_ir":
            return "OFFICIAL_COMPANY_IR"
        if hsrc in ("globenewswire", "businesswire"):
            return "WIRE_SERVICE"
    return "UNKNOWN"


def build_resolved_event(
    resolution: Dict,
    herald_match: Optional[Dict],
    postmortem: Optional[Dict],
) -> Dict[str, Any]:
    """Build a single resolved event feedback record."""
    ticker = resolution["ticker"]
    catalyst_type = resolution.get("catalyst_type", "CORPORATE_UPDATE")
    event_family = _CATALYST_TYPE_TO_FAMILY.get(catalyst_type, "OTHER")

    # Snapshot date: T-1 from resolution
    snapshot_date = resolution.get("prediction_snapshot_date")

    # DEM rank at snapshot
    dem_rank = resolution.get("prediction_dem_rank")

    # Herald fields
    herald_event_id = None
    herald_confidence = None
    herald_outcome_guess = None
    herald_price_direction = None
    exogenous_flag = False

    if herald_match:
        herald_event_id = herald_match.get("event_id")
        herald_confidence = herald_match.get("confidence")
        herald_outcome_guess = herald_match.get("event_outcome_guess")
        herald_price_direction = herald_match.get("price_direction_guess")
        exogenous_flag = herald_match.get("exogenous_to_primary_catalyst", False)

    # Postmortem enrichment
    return_t1 = None
    return_t5 = None
    if postmortem:
        outcome_data = postmortem.get("outcome", {})
        return_t1 = outcome_data.get("return_t1")
        return_t5 = outcome_data.get("return_t5")

    # Label source
    if resolution.get("_override_applied"):
        label_source = "crt_manual"
    elif resolution.get("source_type") == "MANUAL":
        label_source = "crt_manual"
    else:
        label_source = "crt_auto"

    return {
        "schema": SCHEMA_VERSION,
        "ticker": ticker,
        "event_family": event_family,
        "event_type": catalyst_type,
        "event_date": resolution.get("catalyst_date", ""),
        "snapshot_date_t_minus_1": snapshot_date,
        "source_class": _map_source_class(resolution, herald_match),
        "source_url": resolution.get("source_id", ""),
        "headline": (
            resolution.get("catalyst_description", "") or (herald_match.get("headline", "") if herald_match else "")
        ),
        "confirmed": True,
        "actual_outcome": resolution.get("outcome", ""),
        "price_t_minus_1": resolution.get("price_t_minus_1"),
        "price_t_0": resolution.get("price_t_0"),
        "price_t_plus_5": resolution.get("price_t_plus_5"),
        "price_direction": resolution.get("price_direction"),
        "return_t1": return_t1,
        "return_t5": return_t5,
        "resolution_status": "RESOLVED",
        "resolution_date": resolution.get("resolution_date", ""),
        "label_source": label_source,
        "herald_event_id": herald_event_id,
        "herald_confidence": herald_confidence,
        "herald_outcome_guess": herald_outcome_guess,
        "herald_price_direction_guess": herald_price_direction,
        "dem_rank_at_snapshot": dem_rank,
        "exogenous_flag": exogenous_flag,
        "adjudication_method": "crt_resolution",
        "adjudication_timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_event_feedback(
    as_of_date: str,
    *,
    backfill: bool = False,
) -> Dict[str, Any]:
    """Build event feedback artifact from CRT resolutions."""
    logger.info("Building event feedback for %s (backfill=%s)", as_of_date, backfill)

    resolutions = load_crt_resolutions(as_of_date)
    logger.info("Loaded %d resolved CRT records", len(resolutions))

    if not resolutions:
        summary = {
            "schema": SCHEMA_VERSION,
            "as_of_date": as_of_date,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_resolved": 0,
            "status": "NO_DATA",
        }
        _write_outputs(as_of_date, [], summary)
        return summary

    classified_index = load_classified_index(max_days=120 if backfill else 90)
    logger.info(
        "Herald classified index: %d tickers",
        len(classified_index),
    )

    postmortem_index = load_postmortem_index(as_of_date)
    logger.info("Postmortem index: %d records", len(postmortem_index))

    # Check what we've already materialized (skip duplicates on incremental runs)
    existing_keys: Set[Tuple[str, str]] = set()
    if not backfill:
        existing_keys = _load_existing_keys()

    events = []
    stats = {
        "herald_matched": 0,
        "herald_unmatched": 0,
        "postmortem_matched": 0,
        "exogenous_flagged": 0,
        "skipped_existing": 0,
        "by_outcome": defaultdict(int),
        "by_family": defaultdict(int),
        "by_source_class": defaultdict(int),
    }

    for res in resolutions:
        key = (res["ticker"], res["catalyst_date"])

        if key in existing_keys and not backfill:
            stats["skipped_existing"] += 1
            continue

        herald_match = match_herald_to_resolution(res, classified_index)
        pm_key = (res["ticker"], res.get("catalyst_date", ""))
        postmortem = postmortem_index.get(pm_key)

        event = build_resolved_event(res, herald_match, postmortem)
        events.append(event)

        # Track stats
        if herald_match:
            stats["herald_matched"] += 1
        else:
            stats["herald_unmatched"] += 1
        if postmortem:
            stats["postmortem_matched"] += 1
        if event["exogenous_flag"]:
            stats["exogenous_flagged"] += 1
        stats["by_outcome"][event["actual_outcome"]] += 1
        stats["by_family"][event["event_family"]] += 1
        stats["by_source_class"][event["source_class"]] += 1

    summary = {
        "schema": SCHEMA_VERSION,
        "as_of_date": as_of_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_resolved": len(events),
        "n_skipped_existing": stats["skipped_existing"],
        "herald_match_rate": (round(stats["herald_matched"] / max(len(events), 1), 3)),
        "postmortem_match_rate": (round(stats["postmortem_matched"] / max(len(events), 1), 3)),
        "exogenous_count": stats["exogenous_flagged"],
        "by_outcome": dict(stats["by_outcome"]),
        "by_family": dict(stats["by_family"]),
        "by_source_class": dict(stats["by_source_class"]),
        "status": "OK",
    }

    _write_outputs(as_of_date, events, summary)
    return summary


def _load_existing_keys() -> Set[Tuple[str, str]]:
    """Load (ticker, event_date) keys from all existing resolved_events files."""
    keys: Set[Tuple[str, str]] = set()
    if not OUTPUT_DIR.exists():
        return keys
    for f in OUTPUT_DIR.glob("*_resolved_events.jsonl"):
        try:
            for line in f.read_text(encoding="utf-8").strip().split("\n"):
                if not line.strip():
                    continue
                rec = json.loads(line)
                keys.add((rec.get("ticker", ""), rec.get("event_date", "")))
        except (json.JSONDecodeError, OSError):
            continue
    return keys


def _write_outputs(
    as_of_date: str,
    events: List[Dict],
    summary: Dict,
) -> None:
    """Write JSONL events, JSON summary, and append to ledger."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # JSONL events
    events_path = OUTPUT_DIR / f"{as_of_date}_resolved_events.jsonl"
    with open(events_path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, default=str) + "\n")
    logger.info("Wrote %d events to %s", len(events), events_path)

    # JSON summary
    summary_path = OUTPUT_DIR / f"{as_of_date}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Wrote %s", summary_path)

    # Append to ledger
    ledger_path = OUTPUT_DIR / "ledger.jsonl"
    ledger_entry = {
        "date": as_of_date,
        "n_resolved": summary.get("n_resolved", 0),
        "herald_match_rate": summary.get("herald_match_rate", 0),
        "by_outcome": summary.get("by_outcome", {}),
        "generated_at": summary.get("generated_at", ""),
    }
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(ledger_entry, default=str) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Resolved event materializer")
    parser.add_argument(
        "--as-of-date",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Rebuild all events (ignore existing)",
    )
    args = parser.parse_args()
    started = time.perf_counter()

    result = build_event_feedback(args.as_of_date, backfill=args.backfill)
    if result.get("status") == "NO_DATA":
        logger.info("No resolved CRT events found")
    else:
        logger.info(
            "Event feedback: %d resolved, herald match %.0f%%",
            result["n_resolved"],
            result.get("herald_match_rate", 0) * 100,
        )
    try:
        from tools.agent_skill_telemetry import log_agent_run

        log_agent_run(
            "build_event_feedback",
            f"Event feedback for {args.as_of_date}",
            inputs={"as_of_date": args.as_of_date},
            outputs={"n_resolved": result.get("n_resolved"), "status": result.get("status")},
            success=True,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
    except Exception:
        pass


if __name__ == "__main__":
    main()
