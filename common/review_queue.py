"""Per-run review queue — converts shadow diagnostics into explicit actions.

Consumes market_model_disagreement, ts_flag, and Step-10 eligibility to
produce a short, actionable list of names that need human attention before
adding risk.

Actions (priority order):
    no_add_until_review — do not add/build position until human clears it
    size_haircut        — eligible but size at 50% until review
    manual_review_required — flag for IC meeting, no immediate constraint
    monitor_only        — informational, track but no action
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Blind-spot streak tracker
# ---------------------------------------------------------------------------

STATE_FILE = "blind_spot_streak.json"


def compute_blind_spot_streaks(
    csv_rows: List[dict],
    prev_rankings_path: Optional[Path],
    state_path: Path,
) -> Dict[str, int]:
    """Update blind-spot streak counter and return {ticker: days}.

    Loads prior streak state, checks which tickers were BLIND_SPOT in
    the previous snapshot, increments or resets counters.
    """
    # Load prior state
    state_file = state_path / STATE_FILE
    prior_streaks: Dict[str, int] = {}
    if state_file.exists():
        try:
            prior_streaks = json.loads(state_file.read_text())
        except Exception:
            pass

    # Current BLIND_SPOT tickers
    current_blind = {r.get("ticker", "") for r in csv_rows if r.get("ts_flag_type") == "BLIND_SPOT"}

    # Check if prior snapshot also had BLIND_SPOT for continuity
    prior_blind: set = set()
    if prev_rankings_path and prev_rankings_path.exists():
        try:
            with open(prev_rankings_path, newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    if row.get("ts_flag_type") in ("BLIND_SPOT", "BLIND_SPOT_UNCONFIRMED"):
                        prior_blind.add(row.get("ticker", ""))
        except Exception:
            pass

    # Update streaks
    new_streaks: Dict[str, int] = {}
    for ticker in current_blind:
        if ticker in prior_blind or ticker in prior_streaks:
            new_streaks[ticker] = prior_streaks.get(ticker, 0) + 1
        else:
            new_streaks[ticker] = 1

    # Write updated state
    state_path.mkdir(parents=True, exist_ok=True)
    try:
        with open(state_file, "w") as f:
            json.dump(new_streaks, f, indent=2)
            f.write("\n")
    except OSError as exc:
        logger.warning("Could not write blind_spot_streak.json: %s", exc)

    return new_streaks


# ---------------------------------------------------------------------------
# Action assignment
# ---------------------------------------------------------------------------


def assign_action(
    row: dict,
    options_data_fresh: bool,
    blind_spot_days: int = 0,
) -> tuple:
    """Assign review action to a single row. Returns (action, reason) or None.

    Rules applied in priority order — first match wins.
    """
    eligible = row.get("eligible") == "1"
    disagreement = row.get("market_model_disagreement", "")
    ts_type = row.get("ts_flag_type", "")
    tier = row.get("tier_any", "")
    oqc = row.get("options_quality_composite", "")
    has_reg = row.get("has_regulatory_upcoming_180d", "")

    try:
        cat_days = int(float(row.get("catalyst_days", "") or "9999"))
    except (ValueError, TypeError):
        cat_days = 9999

    try:
        score = float(row.get("composite_score", "") or "0")
    except (ValueError, TypeError):
        score = 0.0

    try:
        reg_days = float(row.get("regulatory_days", "") or "9999")
    except (ValueError, TypeError):
        reg_days = 9999

    oqc_nonzero = oqc not in ("", "0", "0.0")

    # Rule 1: high disagreement + near catalyst + fresh data
    if disagreement == "high" and eligible and cat_days <= 90 and options_data_fresh:
        return (
            "no_add_until_review",
            f"High model-market disagreement with near-term catalyst ({cat_days}d)",
        )

    # Rule 2: persistent blind spot + fresh data
    if ts_type == "BLIND_SPOT" and eligible and blind_spot_days >= 3 and options_data_fresh:
        return (
            "no_add_until_review",
            f"Persistent blind spot ({blind_spot_days}d) — market sees unsurfaced event",
        )

    # Rule 3: high disagreement + far catalyst
    if disagreement == "high" and eligible and cat_days > 90:
        return (
            "size_haircut",
            f"High model-market disagreement, catalyst distant ({cat_days}d)",
        )

    # Rule 4: MARKET_SEES_SOONER on ranked name
    if ts_type == "MARKET_SEES_SOONER" and tier in ("A", "B") and score >= 60:
        return (
            "size_haircut",
            "Market sees nearer event than model on ranked name",
        )

    # Rule 5: early blind spot (not yet persistent)
    if ts_type == "BLIND_SPOT" and blind_spot_days < 3:
        return (
            "manual_review_required",
            f"Blind spot flag ({blind_spot_days}d) — not yet persistent",
        )

    # Rule 6: MARKET_NOT_PRICING_EVENT on ranked near-term name
    if ts_type == "MARKET_NOT_PRICING_EVENT" and cat_days <= 45 and tier in ("A", "B"):
        return (
            "manual_review_required",
            "Near-term catalyst on ranked name but market not pricing it — verify date",
        )

    # Rule 7: Step-10 regulatory opportunity
    if has_reg == "1" and oqc_nonzero and tier in ("A", "B", "C") and 90 < reg_days <= 180:
        return (
            "manual_review_required",
            f"Step-10 eligible regulatory name (reg_days={int(reg_days)}) with OQC",
        )

    # Rule 8: any remaining flag
    if disagreement in ("high", "medium") or ts_type:
        return ("monitor_only", f"Flag present: disagreement={disagreement}, ts={ts_type}")

    return None


# ---------------------------------------------------------------------------
# Queue builder
# ---------------------------------------------------------------------------

ACTION_PRIORITY = {
    "no_add_until_review": 0,
    "size_haircut": 1,
    "manual_review_required": 2,
    "monitor_only": 3,
}


def build_review_queue(
    csv_rows: List[dict],
    options_data_fresh: bool,
    blind_spot_streaks: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """Build the review queue from enriched csv_rows."""
    streaks = blind_spot_streaks or {}
    queue: List[Dict[str, Any]] = []

    for row in csv_rows:
        ticker = row.get("ticker", "")
        bs_days = streaks.get(ticker, 0)
        result = assign_action(row, options_data_fresh, bs_days)
        if result is None:
            continue
        action, reason = result
        queue.append(
            {
                "ticker": ticker,
                "tier": row.get("tier_any", ""),
                "composite_score": row.get("composite_score", ""),
                "catalyst_days": row.get("catalyst_days", ""),
                "catalyst_family": row.get("catalyst_family", ""),
                "action": action,
                "action_reason": reason,
                "market_model_disagreement": row.get("market_model_disagreement", ""),
                "ts_flag_type": row.get("ts_flag_type", ""),
                "ts_blind_spot_days": bs_days,
                "options_quality_composite": row.get("options_quality_composite", ""),
                "regulatory_days": row.get("regulatory_days", ""),
            }
        )

    # Sort by action severity then composite score descending
    queue.sort(
        key=lambda r: (
            ACTION_PRIORITY.get(r["action"], 99),
            -float(r.get("composite_score") or 0),
        )
    )
    return queue


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

QUEUE_CSV_COLUMNS = [
    "ticker",
    "tier",
    "composite_score",
    "catalyst_days",
    "catalyst_family",
    "action",
    "action_reason",
    "market_model_disagreement",
    "ts_flag_type",
    "ts_blind_spot_days",
    "options_quality_composite",
    "regulatory_days",
    "as_of_date",
]


def write_review_queue_csv(
    queue: List[Dict[str, Any]],
    path: Path,
    as_of_date: str,
) -> None:
    """Write review_queue.csv."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=QUEUE_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in queue:
            row["as_of_date"] = as_of_date
            writer.writerow(row)


def format_review_queue_md(
    queue: List[Dict[str, Any]],
    as_of_date: str,
    options_fresh: bool,
) -> str:
    """Format the review queue as human-readable Markdown."""
    lines = [f"# Review Queue — {as_of_date}", ""]

    if not options_fresh:
        lines.append(
            "**OPTIONS DATA STALE** — high-disagreement and blind-spot "
            "actions may be suppressed. Verify before acting."
        )
        lines.append("")

    groups: dict[str, list] = {
        "no_add_until_review": [],
        "size_haircut": [],
        "manual_review_required": [],
        "monitor_only": [],
    }
    for row in queue:
        groups.setdefault(row["action"], []).append(row)

    headers = {
        "no_add_until_review": "No-Add Until Review",
        "size_haircut": "Size Haircut",
        "manual_review_required": "Manual Review Required",
        "monitor_only": "Monitor Only",
    }

    for action in ["no_add_until_review", "size_haircut", "manual_review_required", "monitor_only"]:
        rows = groups.get(action, [])
        label = headers.get(action, action)
        lines.append(f"## {label} ({len(rows)})")
        lines.append("")
        if not rows:
            lines.append("None.")
            lines.append("")
            continue
        lines.append("| Ticker | Tier | Score | CatDays | Reason |")
        lines.append("|--------|------|-------|---------|--------|")
        for r in rows:
            score = r.get("composite_score", "")
            if score:
                try:
                    score = f"{float(score):.2f}"
                except (ValueError, TypeError):
                    pass
            lines.append(
                f"| {r['ticker']} | {r.get('tier', '')} | {score} "
                f"| {r.get('catalyst_days', '')} | {r['action_reason']} |"
            )
        lines.append("")

    fresh_str = "FRESH" if options_fresh else "STALE"
    lines.append(f"_Options data: {fresh_str} as of {as_of_date}_")
    lines.append("")

    return "\n".join(lines)


def compute_queue_summary(queue: List[Dict[str, Any]]) -> Dict[str, int]:
    """Return action counts for coverage_quality.json."""
    counts: Dict[str, int] = {
        "no_add_until_review": 0,
        "size_haircut": 0,
        "manual_review_required": 0,
        "monitor_only": 0,
    }
    for row in queue:
        action = row.get("action", "")
        if action in counts:
            counts[action] += 1
    counts["total_flagged"] = sum(counts.values())
    return counts
