#!/usr/bin/env python3
"""Catalyst delta builder — detect event changes between snapshots.

Compares catalyst fields between today's and prior day's rankings.csv
to surface material event changes: new catalysts, date shifts, source
family changes, event resolution, and ticker entry/exit.

Noise filter: only surfaces names that meet at least one:
  - A or B tier
  - catalyst_days <= 30
  - source family changed (hard <-> soft)
  - in shadow portfolio or trade plan

Read-only — does not affect rankings, scoring, or execution.

Output:
    artifacts/catalyst_delta/{date}_delta.json
    artifacts/catalyst_delta/{date}_delta.md

Usage:
    python tools/build_catalyst_delta.py --as-of-date 2026-03-27
    python tools/build_catalyst_delta.py --as-of-date 2026-03-27 --prior-date 2026-03-26
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("catalyst_delta")

SCHEMA_VERSION = "catalyst_delta.v1"

# Material change thresholds
DATE_SHIFT_MATERIAL = 3  # days
HARD_FAMILIES = {"REGULATORY", "CLINICAL"}
SOFT_FAMILIES = {"IR_EVENTS", "CONFERENCE"}

# Spec 088 Phase B v2 filter constants
V2_DAYS_THRESHOLD = 60
V2_SCHEMA = "catalyst_delta.v1.filtered"
V2_FILTER_DEFINITION = (
    "EXITED→drop_unless_held "
    "AND catalyst_days<=60 "
    "AND (family in {REGULATORY,CLINICAL} OR CTGOV-aware family-changing code)"
)

# Fields to compare per ticker
CATALYST_FIELDS = [
    "catalyst_days",
    "catalyst_family",
    "catalyst_event_type",
    "catalyst_source",
    "catalyst_mode",
    "catalyst_bucket",
    "is_hard_catalyst",
]


def _sf(val: Any) -> float:
    if val is None or val == "":
        return math.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return math.nan


def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _is_promoted_snapshot(name: str) -> bool:
    return len(name) == 10 and not name.startswith("_") and name != "state"


def _find_prior_snapshot(snapshots_dir: Path, current_date: str) -> Optional[str]:
    candidates = sorted(
        d.name
        for d in snapshots_dir.iterdir()
        if d.is_dir() and _is_promoted_snapshot(d.name) and d.name < current_date
    )
    return candidates[-1] if candidates else None


def _load_rankings(snap_dir: Path) -> Dict[str, Dict[str, str]]:
    path = snap_dir / "rankings.csv"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {row["ticker"]: row for row in csv.DictReader(f) if row.get("ticker")}


# ---------------------------------------------------------------------------
# Change classification
# ---------------------------------------------------------------------------
def classify_change(
    ticker: str,
    prior: Optional[Dict[str, str]],
    current: Optional[Dict[str, str]],
) -> Optional[Dict[str, Any]]:
    """Classify the catalyst change for one ticker.

    Returns a change dict with code and details, or None if no material change.
    """
    codes: List[str] = []
    details: Dict[str, Any] = {"ticker": ticker}

    if prior is None and current is not None:
        # New entrant
        codes.append("NEW_ENTRANT")
        details["catalyst_days"] = current.get("catalyst_days", "")
        details["catalyst_family"] = current.get("catalyst_family", "")
        details["catalyst_source"] = current.get("catalyst_source", "")
        details["tier"] = current.get("tier_dev", "")
        details["rank"] = current.get("actionable_rank", "")
    elif current is None and prior is not None:
        # Exited universe
        codes.append("EXITED")
        details["prior_catalyst_days"] = prior.get("catalyst_days", "")
        details["prior_family"] = prior.get("catalyst_family", "")
        details["tier"] = prior.get("tier_dev", "")
    elif prior is not None and current is not None:
        # Both exist — compare fields
        p_days = _sf(prior.get("catalyst_days", ""))
        c_days = _sf(current.get("catalyst_days", ""))
        p_family = prior.get("catalyst_family", "")
        c_family = current.get("catalyst_family", "")
        p_source = prior.get("catalyst_source", "")
        c_source = current.get("catalyst_source", "")
        p_type = prior.get("catalyst_event_type", "")
        c_type = current.get("catalyst_event_type", "")
        p_mode = prior.get("catalyst_mode", "")
        c_mode = current.get("catalyst_mode", "")
        p_hard = prior.get("is_hard_catalyst", "")
        c_hard = current.get("is_hard_catalyst", "")

        details["tier"] = current.get("tier_dev", "")
        details["rank"] = current.get("actionable_rank", "")
        details["catalyst_days"] = current.get("catalyst_days", "")
        details["catalyst_family"] = c_family

        # Date shift (beyond natural -1 day decay)
        if not math.isnan(p_days) and not math.isnan(c_days):
            # Expected: c_days = p_days - 1 (one trading day passed)
            # Material if |actual - expected| >= threshold
            expected = p_days - 1
            shift = c_days - expected
            if abs(shift) >= DATE_SHIFT_MATERIAL:
                if c_days > p_days:
                    codes.append("DATE_PUSHED_BACK")
                elif shift < -DATE_SHIFT_MATERIAL:
                    codes.append("DATE_PULLED_FORWARD")
                else:
                    codes.append("DATE_SHIFTED")
                details["prior_days"] = int(p_days)
                details["current_days"] = int(c_days)
                details["shift"] = int(shift)

        # Family changed
        if p_family and c_family and p_family != c_family:
            codes.append("FAMILY_CHANGED")
            details["prior_family"] = p_family
            details["current_family"] = c_family

        # Source changed
        if p_source and c_source and p_source != c_source:
            codes.append("SOURCE_CHANGED")
            details["prior_source"] = p_source
            details["current_source"] = c_source

        # Event type changed
        if p_type and c_type and p_type != c_type:
            codes.append("EVENT_TYPE_CHANGED")
            details["prior_event_type"] = p_type
            details["current_event_type"] = c_type

        # Hard/soft status changed
        if p_hard and c_hard and p_hard != c_hard:
            if c_hard == "1":
                codes.append("BECAME_HARD")
            else:
                codes.append("BECAME_SOFT")

        # Mode transition
        if p_mode and c_mode and p_mode != c_mode:
            if p_mode == "specific_days" and c_mode == "no_upcoming":
                codes.append("EVENT_RESOLVED")
            elif p_mode == "no_upcoming" and c_mode == "specific_days":
                codes.append("NEW_EVENT_APPEARED")
            elif p_mode == "specific_days" and c_mode == "far_window":
                codes.append("EVENT_BECAME_FAR")
            else:
                codes.append("MODE_CHANGED")
            details["prior_mode"] = p_mode
            details["current_mode"] = c_mode

    if not codes:
        return None

    details["codes"] = codes
    return details


# ---------------------------------------------------------------------------
# Noise filter
# ---------------------------------------------------------------------------
def passes_noise_filter(
    change: Dict[str, Any],
    position_tickers: Set[str],
    trade_plan_tickers: Set[str],
) -> bool:
    """Return True if the change should be surfaced."""
    ticker = change["ticker"]
    tier = change.get("tier", "")
    codes = change.get("codes", [])

    # A or B tier
    if tier in ("A", "B"):
        return True

    # Catalyst <= 30 days
    days = _sf(change.get("catalyst_days", ""))
    if not math.isnan(days) and days <= 30:
        return True

    # Family changed (hard <-> soft)
    if "FAMILY_CHANGED" in codes:
        return True

    # In shadow or trade plan
    if ticker in position_tickers or ticker in trade_plan_tickers:
        return True

    return False


def _passes_v2_filter(delta: Dict[str, Any], held_tickers: Set[str]) -> bool:
    """Return True if delta passes the Spec 088 v2 filter.

    Gate logic (AND):
    1. in_universe: EXITED rows dropped unless ticker in held_tickers
    2. catalyst_days <= V2_DAYS_THRESHOLD (NaN → excluded)
    3. HARD family OR family-changing codes (CTGOV-prefix-stripped)
    """
    codes = delta.get("codes", [])
    ticker = delta.get("ticker", "")

    # EXITED: pass only if held; bypass remaining checks for held exits
    if "EXITED" in codes:
        return ticker in held_tickers

    # catalyst_days <= 60 (NaN excluded)
    days = _sf(delta.get("catalyst_days", ""))
    if math.isnan(days) or days > V2_DAYS_THRESHOLD:
        return False

    # Hard family
    if delta.get("catalyst_family", "") in HARD_FAMILIES:
        return True

    # Family-changing codes (CTGOV-prefix-aware)
    normalized = {c[len("CTGOV_") :] if c.startswith("CTGOV_") else c for c in codes}
    if normalized & {"FAMILY_CHANGED", "BECAME_HARD", "BECAME_SOFT"}:
        return True

    return False


def apply_v2_filter(
    deltas: List[Dict[str, Any]],
    held_tickers: Set[str],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split deltas into (passed, suppressed) per v2 filter."""
    passed, suppressed = [], []
    for delta in deltas:
        (passed if _passes_v2_filter(delta, held_tickers) else suppressed).append(delta)
    return passed, suppressed


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
def build_catalyst_delta(
    as_of_date: str,
    *,
    prior_date: Optional[str] = None,
    snapshots_dir: Path = REPO_ROOT / "data" / "snapshots",
    artifacts_dir: Path = REPO_ROOT / "artifacts",
) -> Dict[str, Any]:
    """Build catalyst delta artifact.

    Parameters
    ----------
    as_of_date : str
        Current snapshot date.
    prior_date : str, optional
        Explicit prior date. Auto-detects if omitted.
    snapshots_dir : Path
        Snapshots root.
    artifacts_dir : Path
        Artifacts root.

    Returns
    -------
    dict with schema, deltas, and metadata.
    """
    snap_dir = snapshots_dir / as_of_date
    if not (snap_dir / "rankings.csv").exists():
        return {"error": f"rankings.csv not found for {as_of_date}"}

    # Find prior
    if prior_date is None:
        prior_date = _find_prior_snapshot(snapshots_dir, as_of_date)
    if prior_date is None:
        return {"error": "no prior snapshot found"}

    prior_dir = snapshots_dir / prior_date
    if not (prior_dir / "rankings.csv").exists():
        return {"error": f"rankings.csv not found for {prior_date}"}

    # Load rankings
    current_rankings = _load_rankings(snap_dir)
    prior_rankings = _load_rankings(prior_dir)

    # Load context
    position_tickers: Set[str] = set()
    pos_data = _load_json(artifacts_dir / "live_shadow" / "positions" / f"{as_of_date}.json")
    if pos_data:
        position_tickers = {p["ticker"] for p in pos_data.get("positions", []) if p.get("ticker")}

    trade_plan_tickers: Set[str] = set()
    tp_path = artifacts_dir / "live_shadow" / "trade_plan" / as_of_date / "trade_plan.csv"
    if tp_path.exists():
        with open(tp_path, encoding="utf-8") as f:
            trade_plan_tickers = {row["ticker"] for row in csv.DictReader(f) if row.get("ticker")}

    # Classify changes for all tickers
    all_tickers = set(current_rankings.keys()) | set(prior_rankings.keys())
    raw_changes = []
    for ticker in sorted(all_tickers):
        prior = prior_rankings.get(ticker)
        current = current_rankings.get(ticker)
        change = classify_change(ticker, prior, current)
        if change is not None:
            raw_changes.append(change)

    # Merge CTgov daily diff (trial-level transitions the snapshot comparison may miss)
    ctgov_diff_path = artifacts_dir / "ctgov_daily" / f"{as_of_date}_diff.json"
    ctgov_diff = _load_json(ctgov_diff_path)
    n_ctgov_merged = 0
    if ctgov_diff:
        snapshot_tickers = {c["ticker"] for c in raw_changes}
        for trial_change in ctgov_diff.get("changes", []):
            ticker = trial_change.get("ticker", "")
            if ticker in all_tickers and ticker not in snapshot_tickers:
                # Trial-level change not visible in snapshot comparison
                ctgov_codes = trial_change.get("codes", [])
                raw_changes.append(
                    {
                        "ticker": ticker,
                        "codes": [f"CTGOV_{c}" for c in ctgov_codes],
                        "tier": current_rankings.get(ticker, {}).get("tier_dev", ""),
                        "rank": current_rankings.get(ticker, {}).get("actionable_rank", ""),
                        "catalyst_days": current_rankings.get(ticker, {}).get("catalyst_days", ""),
                        "catalyst_family": current_rankings.get(ticker, {}).get("catalyst_family", ""),
                        "nct_id": trial_change.get("nct_id", ""),
                        "trial_detail": trial_change.get("title", "")[:60],
                    }
                )
                n_ctgov_merged += 1

    # Apply noise filter
    filtered = [c for c in raw_changes if passes_noise_filter(c, position_tickers, trade_plan_tickers)]

    # Sort by impact: A-tier first, then by catalyst_days ascending
    tier_order = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    filtered.sort(
        key=lambda c: (
            tier_order.get(c.get("tier", ""), 9),
            _sf(c.get("catalyst_days", "")) if not math.isnan(_sf(c.get("catalyst_days", ""))) else 9999,
        )
    )

    # Aggregate code counts
    from collections import Counter

    code_counts = Counter()
    for c in filtered:
        for code in c.get("codes", []):
            code_counts[code] += 1

    result = {
        "schema": SCHEMA_VERSION,
        "as_of_date": as_of_date,
        "prior_date": prior_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_raw_changes": len(raw_changes),
        "n_filtered": len(filtered),
        "n_noise_suppressed": len(raw_changes) - len(filtered),
        "code_counts": dict(code_counts.most_common()),
        "context": {
            "n_current_tickers": len(current_rankings),
            "n_prior_tickers": len(prior_rankings),
            "n_positions": len(position_tickers),
            "n_trade_plan": len(trade_plan_tickers),
            "n_ctgov_merged": n_ctgov_merged,
        },
        "deltas": filtered,
    }

    # Write artifacts
    delta_dir = artifacts_dir / "catalyst_delta"
    delta_dir.mkdir(parents=True, exist_ok=True)

    json_path = delta_dir / f"{as_of_date}_delta.json"
    md_path = delta_dir / f"{as_of_date}_delta.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Wrote %s", json_path)

    md_text = format_delta_md(result)
    md_path.write_text(md_text, encoding="utf-8")
    logger.info("Wrote %s", md_path)

    result["_json_path"] = str(json_path)
    result["_md_path"] = str(md_path)

    # --- Spec 088 Phase B: filtered companion artifact ---
    held_tickers = position_tickers | trade_plan_tickers
    v2_passed, _v2_suppressed = apply_v2_filter(filtered, held_tickers)

    filtered_result = {
        "schema": V2_SCHEMA,
        "as_of_date": as_of_date,
        "prior_date": prior_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_raw_changes": len(raw_changes),
        "n_filtered": len(v2_passed),
        "n_noise_suppressed": len(raw_changes) - len(filtered),
        "n_v2_filtered": len(v2_passed),
        "n_v2_suppressed": len(filtered) - len(v2_passed),
        "v2_filter_definition": V2_FILTER_DEFINITION,
        "code_counts": dict(Counter(code for c in v2_passed for code in c.get("codes", [])).most_common()),
        "context": result["context"],
        "deltas": v2_passed,
    }

    filtered_json_path = delta_dir / f"{as_of_date}_delta_filtered.json"
    filtered_md_path = delta_dir / f"{as_of_date}_delta_filtered.md"

    with open(filtered_json_path, "w", encoding="utf-8") as f:
        json.dump(filtered_result, f, indent=2, default=str)
    logger.info(
        "Wrote %s (v2: %d kept, %d suppressed)",
        filtered_json_path,
        len(v2_passed),
        len(filtered) - len(v2_passed),
    )

    filtered_md_path.write_text(format_delta_md(filtered_result), encoding="utf-8")
    logger.info("Wrote %s", filtered_md_path)

    result["_filtered_json_path"] = str(filtered_json_path)
    result["_filtered_md_path"] = str(filtered_md_path)

    return result


# ---------------------------------------------------------------------------
# Markdown formatter
# ---------------------------------------------------------------------------
def format_delta_md(d: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"# Catalyst Delta — {d['as_of_date']}")
    lines.append("")
    lines.append(
        f"Prior: {d['prior_date']} | Changes: {d['n_filtered']} surfaced, {d['n_noise_suppressed']} suppressed"
    )
    lines.append("")

    code_counts = d.get("code_counts", {})
    if code_counts:
        lines.append("## Change Codes")
        lines.append("")
        for code, count in code_counts.items():
            lines.append(f"- {code}: {count}")
        lines.append("")

    deltas = d.get("deltas", [])
    if deltas:
        lines.append("## Surfaced Changes")
        lines.append("")
        lines.append("| Ticker | Tier | Rank | Days | Family | Codes | Detail |")
        lines.append("|--------|------|------|------|--------|-------|--------|")
        for c in deltas:
            codes_str = ", ".join(c.get("codes", []))
            rank = c.get("rank", "?")
            days = c.get("catalyst_days", "?")
            tier = c.get("tier", "?")
            family = c.get("catalyst_family", c.get("prior_family", "?"))

            # Build detail string
            detail_parts = []
            if "shift" in c:
                detail_parts.append(f"shift {c['shift']:+d}d")
            if "prior_family" in c and "current_family" in c:
                detail_parts.append(f"{c['prior_family']}→{c['current_family']}")
            if "prior_source" in c and "current_source" in c:
                detail_parts.append(f"{c['prior_source']}→{c['current_source']}")
            if "prior_mode" in c and "current_mode" in c:
                detail_parts.append(f"{c['prior_mode']}→{c['current_mode']}")
            detail = "; ".join(detail_parts) if detail_parts else "-"

            lines.append(f"| {c['ticker']} | {tier} | {rank} | {days} | {family} | {codes_str} | {detail} |")
        lines.append("")
    else:
        lines.append("No material catalyst changes detected.")
        lines.append("")

    lines.append(f"*Generated: {d.get('generated_at', '')}*")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Catalyst delta builder")
    parser.add_argument("--as-of-date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument("--prior-date", default=None)
    parser.add_argument("--snapshots-dir", type=Path, default=REPO_ROOT / "data" / "snapshots")
    parser.add_argument("--artifacts-dir", type=Path, default=REPO_ROOT / "artifacts")
    args = parser.parse_args()
    started = time.perf_counter()

    result = build_catalyst_delta(
        args.as_of_date,
        prior_date=args.prior_date,
        snapshots_dir=args.snapshots_dir,
        artifacts_dir=args.artifacts_dir,
    )

    if "error" in result:
        logger.error(result["error"])
        try:
            from tools.agent_skill_telemetry import log_agent_run

            log_agent_run(
                "build_catalyst_delta",
                f"Catalyst delta for {args.as_of_date}",
                inputs={"as_of_date": args.as_of_date},
                outputs={"error": result["error"]},
                success=False,
                error=result["error"],
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception:
            pass
        sys.exit(1)

    logger.info(
        "Delta: %d surfaced, %d suppressed (%s → %s)",
        result["n_filtered"],
        result["n_noise_suppressed"],
        result["prior_date"],
        result["as_of_date"],
    )
    try:
        from tools.agent_skill_telemetry import log_agent_run

        log_agent_run(
            "build_catalyst_delta",
            f"Catalyst delta for {args.as_of_date}",
            inputs={"as_of_date": args.as_of_date, "prior_date": result.get("prior_date")},
            outputs={
                "n_filtered": result.get("n_filtered"),
                "n_noise_suppressed": result.get("n_noise_suppressed"),
            },
            success=True,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
    except Exception:
        pass


if __name__ == "__main__":
    main()
