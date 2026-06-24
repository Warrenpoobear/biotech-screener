#!/usr/bin/env python3
"""Options watch builder — post-packet and pre-open watchlist.

Builds a structured watchlist of names that warrant options surface review,
prioritized by review queue, trade plan, shadow positions, catalyst delta,
and A-tier proximity.  Read-only — does not affect rankings, scoring, or
execution.

Modes:
    post_packet — full Phase 2 watchlist (default, 5:40 PM ET context)
    pre_open    — narrow Phase 3 watchlist (stricter gates, fewer names)

Inputs (required):
    data/snapshots/{date}/rankings.csv

Inputs (optional, graceful degradation):
    data/snapshots/{date}/review_queue.csv
    artifacts/live_shadow/trade_plan/{date}/trade_plan.csv
    artifacts/live_shadow/positions/{date}.json
    artifacts/catalyst_delta/{date}_delta.json
    data/snapshots/{date}/coverage_quality.json
    data/snapshots/{date}/options_diagnostics_summary.json
    artifacts/options_watch/{prior_date}_watch.json

Output:
    artifacts/options_watch/{date}_watch.json
    artifacts/options_watch/{date}_watch.md
    (pre_open mode: {date}_premarket_watch.json / .md)

Usage:
    python tools/build_options_watch.py --as-of-date 2026-03-27
    python tools/build_options_watch.py --as-of-date 2026-03-27 --mode pre_open
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("options_watch")

SCHEMA_VERSION = "options_watch.v1"
WATCHLIST_MAX = 30

# ---------------------------------------------------------------------------
# Thresholds — all from Phase 2 spec (agents/options_watch/)
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "event_premium_slope": -0.10,
    "surface_move_high": 0.80,
    "surface_move_med": 0.60,
    "iv_ramp_high": 0.10,
    "iv_ramp_med": 0.05,
    "iv_falling": -0.05,
    "drift_risk_high_pctile": 0.85,
    "drift_risk_high_iv": 0.12,
    "drift_risk_med_pctile": 0.65,
    "drift_risk_med_iv": 0.06,
    "extreme_skew": 0.15,
    "priority_cap": 3,
    "watchlist_max": WATCHLIST_MAX,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sf(val: Any) -> float:
    """Safe float — returns NaN for empty/missing."""
    import math

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


def _load_csv_tickers(path: Path) -> Set[str]:
    """Load ticker column from a CSV, return set."""
    if not path.exists():
        return set()
    try:
        with open(path, encoding="utf-8") as f:
            return {row["ticker"] for row in csv.DictReader(f) if row.get("ticker")}
    except (KeyError, OSError):
        return set()


def _find_prior_watch(watch_dir: Path, current_date: str) -> Optional[Dict]:
    """Find the most recent watch JSON before current_date."""
    candidates = sorted(p for p in watch_dir.glob("*_watch.json") if p.stem.replace("_watch", "") < current_date)
    if not candidates:
        return None
    return _load_json(candidates[-1])


# ---------------------------------------------------------------------------
# Watchlist building
# ---------------------------------------------------------------------------
def _build_candidate_set(
    rankings: List[Dict[str, str]],
    review_queue_tickers: Set[str],
    trade_plan_tickers: Set[str],
    position_tickers: Set[str],
    catalyst_delta_tickers: Set[str],
    mode: str,
) -> List[Dict[str, Any]]:
    """Build prioritized candidate list from rankings + context sources.

    Returns list of dicts with ranking fields plus source_priority.
    Priority tiers (lower = higher priority):
        1 = review queue (hard-catalyst)
        2 = trade plan
        3 = shadow positions
        4 = catalyst delta
        5 = A-tier <= 30d
    """
    candidates = []
    for row in rankings:
        ticker = row.get("ticker", "")
        if not ticker:
            continue

        tier = row.get("tier_dev", "")
        cat_days = _sf(row.get("catalyst_days", ""))
        is_hard = row.get("is_hard_catalyst", "") == "1"

        # Determine source priority
        priority = None
        if ticker in review_queue_tickers:
            priority = 1
        elif ticker in trade_plan_tickers:
            priority = 2
        elif ticker in position_tickers:
            priority = 3
        elif ticker in catalyst_delta_tickers:
            priority = 4
        elif tier == "A" and not __import__("math").isnan(cat_days) and cat_days <= 30:
            priority = 5
        elif is_hard and not __import__("math").isnan(cat_days) and cat_days <= 14:
            priority = 5  # hard-catalyst near-term, any tier

        if priority is None:
            continue

        # Pre-open mode: stricter — only tiers 1-3 and hard-catalyst <= 14d
        if mode == "pre_open":
            if priority > 3 and not (is_hard and not __import__("math").isnan(cat_days) and cat_days <= 14):
                continue

        candidates.append(
            {
                "ticker": ticker,
                "source_priority": priority,
                "row": row,
            }
        )

    return candidates


def _check_eligibility(row: Dict[str, str]) -> bool:
    """Check options eligibility gates."""
    return row.get("opt_liquidity_state", "absent") == "liquid" and row.get("opt_use_for_judgment") == "YES"


def _compute_flags(row: Dict[str, str]) -> List[str]:
    """Compute alert flags for a name based on thresholds."""
    flags = []

    # EVENT_PREMIUM
    event_prem = row.get("opt_event_premium", "")
    term_slope = _sf(row.get("opt_term_slope", ""))
    if event_prem == "YES" or (
        not __import__("math").isnan(term_slope) and term_slope <= THRESHOLDS["event_premium_slope"]
    ):
        flags.append("EVENT_PREMIUM")

    # IV_RAMP
    iv_change_5d = _sf(row.get("atm_iv_change_5d", ""))
    if not __import__("math").isnan(iv_change_5d):
        if iv_change_5d >= THRESHOLDS["iv_ramp_high"]:
            flags.append("IV_RAMP_HIGH")
        elif iv_change_5d >= THRESHOLDS["iv_ramp_med"]:
            flags.append("IV_RAMP_MED")
        elif iv_change_5d <= THRESHOLDS["iv_falling"]:
            flags.append("IV_FALLING")

    # SURFACE_MOVE
    move_pctile = _sf(row.get("actual_implied_move_pctile", ""))
    if not __import__("math").isnan(move_pctile):
        if move_pctile >= THRESHOLDS["surface_move_high"]:
            flags.append("SURFACE_MOVE_HIGH")
        elif move_pctile >= THRESHOLDS["surface_move_med"]:
            flags.append("SURFACE_MOVE_MED")

    # DRIFT_RISK
    if not __import__("math").isnan(move_pctile) or not __import__("math").isnan(iv_change_5d):
        is_high = (
            not __import__("math").isnan(move_pctile) and move_pctile >= THRESHOLDS["drift_risk_high_pctile"]
        ) or (not __import__("math").isnan(iv_change_5d) and iv_change_5d >= THRESHOLDS["drift_risk_high_iv"])
        is_med = (not __import__("math").isnan(move_pctile) and move_pctile >= THRESHOLDS["drift_risk_med_pctile"]) or (
            not __import__("math").isnan(iv_change_5d) and iv_change_5d >= THRESHOLDS["drift_risk_med_iv"]
        )
        if is_high:
            flags.append("DRIFT_RISK_HIGH")
        elif is_med:
            flags.append("DRIFT_RISK_MED")

    # EXTREME_SKEW
    rr_25d = _sf(row.get("opt_rr_25d", ""))
    if not __import__("math").isnan(rr_25d) and abs(rr_25d) >= THRESHOLDS["extreme_skew"]:
        flags.append("EXTREME_SKEW")

    return flags


def _compute_priority_score(flags: List[str]) -> int:
    """Compute priority score from flags, capped at +3."""
    score = 0
    if "SURFACE_MOVE_HIGH" in flags:
        score += 2
    elif "SURFACE_MOVE_MED" in flags:
        score += 1
    if "IV_RAMP_HIGH" in flags:
        score += 2
    elif "IV_RAMP_MED" in flags:
        score += 1
    return min(score, THRESHOLDS["priority_cap"])


def _should_suppress(row: Dict[str, str], is_hard: bool, in_trade_plan: bool) -> Optional[str]:
    """Check suppression rules. Returns reason string or None."""
    regime = row.get("opt_iv_regime", "")
    if regime == "EXTREME" and not is_hard and not in_trade_plan:
        return "opt_iv_regime=EXTREME, not hard-catalyst or trade-plan"
    return None


def _build_why(row: Dict[str, str], flags: List[str], source_priority: int) -> str:
    """Build human-readable explanation."""
    import math

    parts = []
    cat_days = _sf(row.get("catalyst_days", ""))
    is_hard = row.get("is_hard_catalyst", "") == "1"

    source_names = {
        1: "review queue",
        2: "trade plan",
        3: "shadow position",
        4: "catalyst delta",
        5: "A-tier near-term",
    }
    parts.append(source_names.get(source_priority, "unknown source"))

    if is_hard and not math.isnan(cat_days):
        parts.append(f"hard catalyst {int(cat_days)}d")
    elif not math.isnan(cat_days):
        parts.append(f"{int(cat_days)}d to catalyst")

    signal_parts = []
    if "EVENT_PREMIUM" in flags:
        signal_parts.append("backwardation")
    if "IV_RAMP_HIGH" in flags:
        signal_parts.append("rising IV")
    elif "IV_FALLING" in flags:
        signal_parts.append("falling IV")
    if "SURFACE_MOVE_HIGH" in flags:
        signal_parts.append("high implied move")
    if "EXTREME_SKEW" in flags:
        signal_parts.append("extreme skew")
    if "DRIFT_RISK_HIGH" in flags:
        signal_parts.append("drift risk")

    if signal_parts:
        parts.append(" + ".join(signal_parts))

    return ", ".join(parts)


_STRONG_FLAGS = {"SURFACE_MOVE_HIGH", "IV_RAMP_HIGH", "EVENT_PREMIUM", "EXTREME_SKEW"}


def build_options_watch(
    as_of_date: str,
    *,
    mode: str = "post_packet",
    snapshots_dir: Path = REPO_ROOT / "data" / "snapshots",
    artifacts_dir: Path = REPO_ROOT / "artifacts",
) -> Dict[str, Any]:
    """Build options watch artifact.

    Parameters
    ----------
    as_of_date : str
        Snapshot date (YYYY-MM-DD).
    mode : str
        "post_packet" (Phase 2) or "pre_open" (Phase 3).
    snapshots_dir : Path
        Root directory for snapshots.
    artifacts_dir : Path
        Root directory for artifacts.

    Returns
    -------
    dict with schema, rows, suppressed, and metadata.
    """
    import math

    snap_dir = snapshots_dir / as_of_date

    # --- Load rankings (required) ---
    rankings_path = snap_dir / "rankings.csv"
    if not rankings_path.exists():
        return {"error": f"rankings.csv not found for {as_of_date}"}

    with open(rankings_path, encoding="utf-8") as f:
        rankings = list(csv.DictReader(f))

    # --- Load context sources (optional) ---
    review_queue_tickers = _load_csv_tickers(snap_dir / "review_queue.csv")

    trade_plan_dir = artifacts_dir / "live_shadow" / "trade_plan" / as_of_date
    trade_plan_tickers = _load_csv_tickers(trade_plan_dir / "trade_plan.csv")

    position_tickers: Set[str] = set()
    pos_path = artifacts_dir / "live_shadow" / "positions" / f"{as_of_date}.json"
    pos_data = _load_json(pos_path)
    if pos_data:
        position_tickers = {p["ticker"] for p in pos_data.get("positions", []) if p.get("ticker")}

    catalyst_delta_tickers: Set[str] = set()
    cd_path = artifacts_dir / "catalyst_delta" / f"{as_of_date}_delta.json"
    cd_data = _load_json(cd_path)
    if cd_data:
        catalyst_delta_tickers = {d["ticker"] for d in cd_data.get("deltas", []) if d.get("ticker")}

    # --- Build candidate set ---
    candidates = _build_candidate_set(
        rankings,
        review_queue_tickers,
        trade_plan_tickers,
        position_tickers,
        catalyst_delta_tickers,
        mode,
    )

    # --- Apply eligibility, flags, suppression ---
    rows = []
    suppressed = []
    n_eligible = 0

    for cand in candidates:
        row = cand["row"]
        ticker = cand["ticker"]
        source_priority = cand["source_priority"]
        is_hard = row.get("is_hard_catalyst", "") == "1"

        # Eligibility gate
        if not _check_eligibility(row):
            continue
        n_eligible += 1

        # Suppression check
        suppress_reason = _should_suppress(row, is_hard, ticker in trade_plan_tickers)
        if suppress_reason:
            suppressed.append({"ticker": ticker, "reason": suppress_reason})
            continue

        # Compute flags and priority
        flags = _compute_flags(row)

        # Pre-open mode: require at least one strong signal
        if mode == "pre_open":
            if not any(f in _STRONG_FLAGS for f in flags):
                continue

        priority_score = _compute_priority_score(flags)
        why = _build_why(row, flags, source_priority)

        cat_days = _sf(row.get("catalyst_days", ""))

        entry = {
            "ticker": ticker,
            "tier": row.get("tier_dev", ""),
            "actionable_rank": int(_sf(row.get("actionable_rank", "0"))) if row.get("actionable_rank") else None,
            "catalyst_days": int(cat_days) if not math.isnan(cat_days) else None,
            "catalyst_family": row.get("catalyst_family", ""),
            "is_hard_catalyst": 1 if is_hard else 0,
            "opt_atm_iv": round(_sf(row.get("opt_atm_iv", "")), 4) if row.get("opt_atm_iv") else None,
            "opt_term_slope": round(_sf(row.get("opt_term_slope", "")), 4) if row.get("opt_term_slope") else None,
            "opt_rr_25d": round(_sf(row.get("opt_rr_25d", "")), 4) if row.get("opt_rr_25d") else None,
            "actual_implied_move_pctile": (
                round(_sf(row.get("actual_implied_move_pctile", "")), 4)
                if row.get("actual_implied_move_pctile")
                else None
            ),
            "atm_iv_change_5d": round(_sf(row.get("atm_iv_change_5d", "")), 4) if row.get("atm_iv_change_5d") else None,
            "opt_iv_regime": row.get("opt_iv_regime", ""),
            "opt_event_premium": row.get("opt_event_premium", ""),
            "flags": flags,
            "priority_score": priority_score,
            "source_priority": source_priority,
            "why": why,
        }
        rows.append(entry)

    # Sort: priority_score desc, is_hard desc, actionable_rank asc
    rows.sort(
        key=lambda r: (
            -r["priority_score"],
            -r["is_hard_catalyst"],
            r["actionable_rank"] if r["actionable_rank"] is not None else 9999,
        )
    )

    # Cap at watchlist max
    if len(rows) > WATCHLIST_MAX:
        rows = rows[:WATCHLIST_MAX]

    n_flagged = sum(1 for r in rows if r["flags"])

    result = {
        "schema": SCHEMA_VERSION,
        "as_of_date": as_of_date,
        "mode": mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "watchlist_size": len(rows),
        "n_eligible": n_eligible,
        "n_flagged": n_flagged,
        "n_suppressed": len(suppressed),
        "sources": {
            "review_queue": len(review_queue_tickers),
            "trade_plan": len(trade_plan_tickers),
            "positions": len(position_tickers),
            "catalyst_delta": len(catalyst_delta_tickers),
        },
        "thresholds": THRESHOLDS,
        "rows": rows,
        "suppressed": suppressed,
    }

    # --- Write artifacts ---
    watch_dir = artifacts_dir / "options_watch"
    watch_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_premarket_watch" if mode == "pre_open" else "_watch"
    json_path = watch_dir / f"{as_of_date}{suffix}.json"
    md_path = watch_dir / f"{as_of_date}{suffix}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Wrote %s", json_path)

    md_text = format_watch_md(result)
    md_path.write_text(md_text, encoding="utf-8")
    logger.info("Wrote %s", md_path)

    result["_json_path"] = str(json_path)
    result["_md_path"] = str(md_path)
    return result


# ---------------------------------------------------------------------------
# Markdown formatter
# ---------------------------------------------------------------------------
def format_watch_md(d: Dict[str, Any]) -> str:
    """Format watch result as markdown."""
    lines = []
    mode_label = "Pre-Open" if d.get("mode") == "pre_open" else "Post-Packet"
    lines.append(f"# Options Watch ({mode_label}) — {d['as_of_date']}")
    lines.append("")
    lines.append(
        f"Watchlist: {d['watchlist_size']} names | "
        f"Eligible: {d['n_eligible']} | "
        f"Flagged: {d['n_flagged']} | "
        f"Suppressed: {d['n_suppressed']}"
    )
    lines.append("")
    lines.append(
        f"Sources: review_queue={d['sources']['review_queue']}, "
        f"trade_plan={d['sources']['trade_plan']}, "
        f"positions={d['sources']['positions']}, "
        f"catalyst_delta={d['sources']['catalyst_delta']}"
    )
    lines.append("")

    rows = d.get("rows", [])
    if rows:
        lines.append("## Watchlist")
        lines.append("")
        lines.append("| Ticker | Tier | Rank | Days | Family | Flags | Priority | Why |")
        lines.append("|--------|------|------|------|--------|-------|----------|-----|")
        for r in rows:
            flags_str = ", ".join(r.get("flags", [])) or "-"
            rank = r.get("actionable_rank", "?")
            days = r.get("catalyst_days", "?")
            lines.append(
                f"| {r['ticker']} | {r.get('tier', '')} | {rank} | {days} | "
                f"{r.get('catalyst_family', '')} | {flags_str} | {r['priority_score']} | {r.get('why', '')} |"
            )
        lines.append("")

    # Top flagged names detail
    flagged = [r for r in rows if r.get("flags")]
    if flagged:
        lines.append("## Top Flagged")
        lines.append("")
        for r in flagged[:10]:
            iv = r.get("opt_atm_iv")
            slope = r.get("opt_term_slope")
            rr = r.get("opt_rr_25d")
            move = r.get("actual_implied_move_pctile")
            iv5d = r.get("atm_iv_change_5d")
            parts = []
            if iv is not None:
                parts.append(f"IV={iv:.2f}")
            if slope is not None:
                parts.append(f"slope={slope:+.3f}")
            if rr is not None:
                parts.append(f"RR={rr:+.3f}")
            if move is not None:
                parts.append(f"move_pctile={move:.2f}")
            if iv5d is not None:
                parts.append(f"IV_5d={iv5d:+.3f}")
            detail = " | ".join(parts)
            lines.append(f"- **{r['ticker']}** [{r.get('opt_iv_regime', '')}]: {detail}")
        lines.append("")

    # Suppressed
    suppressed = d.get("suppressed", [])
    if suppressed:
        lines.append("## Suppressed")
        lines.append("")
        for s in suppressed:
            lines.append(f"- {s['ticker']}: {s['reason']}")
        lines.append("")

    lines.append(f"*Generated: {d.get('generated_at', '')}*")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Options watch builder — post-packet and pre-open watchlist")
    parser.add_argument("--as-of-date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"), help="Snapshot date")
    parser.add_argument("--mode", choices=["post_packet", "pre_open"], default="post_packet", help="Watch mode")
    parser.add_argument("--snapshots-dir", type=Path, default=REPO_ROOT / "data" / "snapshots")
    parser.add_argument("--artifacts-dir", type=Path, default=REPO_ROOT / "artifacts")
    args = parser.parse_args()
    started = time.perf_counter()

    result = build_options_watch(
        args.as_of_date,
        mode=args.mode,
        snapshots_dir=args.snapshots_dir,
        artifacts_dir=args.artifacts_dir,
    )

    if "error" in result:
        logger.error(result["error"])
        try:
            from tools.agent_skill_telemetry import log_agent_run

            log_agent_run(
                "build_options_watch",
                f"Options watch for {args.as_of_date}",
                inputs={"as_of_date": args.as_of_date, "mode": args.mode},
                outputs={"error": result["error"]},
                success=False,
                error=result["error"],
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception:
            pass
        sys.exit(1)

    logger.info(
        "Watch: %d names, %d flagged, %d suppressed (%s mode)",
        result["watchlist_size"],
        result["n_flagged"],
        result["n_suppressed"],
        result["mode"],
    )
    try:
        from tools.agent_skill_telemetry import log_agent_run
        from tools.record_skill_feedback import attach_outcome_verdict

        exec_id = log_agent_run(
            "build_options_watch",
            f"Options watch for {args.as_of_date}",
            inputs={"as_of_date": args.as_of_date, "mode": args.mode},
            outputs={
                "n_flagged": result.get("n_flagged"),
                "watchlist_size": result.get("watchlist_size"),
            },
            success=True,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        if exec_id:
            attach_outcome_verdict(
                exec_id,
                was_correct=True,
                evidence=f"mode={result.get('mode')} n_flagged={result.get('n_flagged', 0)}",
            )
    except Exception:
        pass


if __name__ == "__main__":
    main()
