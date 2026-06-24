#!/usr/bin/env python3
"""Daily policy shadow compare — tier-weighted vs current baseline (Spec 035).

Runs after each shadow portfolio update. Computes what the tier-weighted
policy would hold, compares against actual positions, and writes a daily
comparison artifact. Accumulates history for rolling evaluation.

Read-only: does not modify positions, rankings, or execution.

Output:
    artifacts/policy_shadow/tier_weighted/{date}_comparison.json
    artifacts/policy_shadow/tier_weighted/{date}_comparison.md
    artifacts/policy_shadow/tier_weighted/history.jsonl (append)

Usage:
    python tools/build_policy_shadow_compare.py
    python tools/build_policy_shadow_compare.py --as-of-date 2026-03-28
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("policy_shadow")

SCHEMA_VERSION = "policy_shadow_compare.v1"

# Tier weights for Variant A
DEFAULT_TIER_WEIGHTS = {"A": 4.0, "B": 2.5, "C": 1.0, "D": 0.0}

# Headwind exit persistence threshold
DEFAULT_EXIT_PERSISTENCE = 3


def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _load_rankings(snap_dir: Path, date: str) -> Dict[str, Dict]:
    rpath = snap_dir / date / "rankings.csv"
    if not rpath.exists():
        return {}
    with open(rpath, encoding="utf-8") as f:
        return {r["ticker"]: r for r in csv.DictReader(f)}


def _load_prices(price_path: Path) -> Dict[str, Dict[str, float]]:
    prices: Dict[str, Dict[str, float]] = defaultdict(dict)
    if not price_path.exists():
        return prices
    with open(price_path) as f:
        for row in csv.DictReader(f):
            t = row.get("ticker", "")
            d = row.get("date", "")
            c = row.get("close", "")
            if t and d and c:
                try:
                    prices[t][d] = float(c)
                except ValueError:
                    pass
    return prices


def _get_price(prices: Dict[str, Dict[str, float]], ticker: str, date: str) -> Optional[float]:
    tp = prices.get(ticker, {})
    avail = sorted(d for d in tp if d <= date)
    return tp[avail[-1]] if avail else None


def compute_tiered_weights(
    positions: Dict[str, Dict],
    rankings: Dict[str, Dict],
    tier_weights: Dict[str, float],
) -> Dict[str, float]:
    """Compute normalized tier-weighted allocation."""
    raw = {}
    for ticker in positions:
        r = rankings.get(ticker, {})
        tier = r.get("tier_dev", "")
        w = tier_weights.get(tier, 0.0)
        if w > 0:
            raw[ticker] = w

    total = sum(raw.values())
    if total <= 0:
        return {}
    return {t: round(w / total * 100, 2) for t, w in raw.items()}


def compute_tiered_exit_weights(
    positions: Dict[str, Dict],
    rankings: Dict[str, Dict],
    tier_weights: Dict[str, float],
    history_path: Path,
    exit_persistence: int = DEFAULT_EXIT_PERSISTENCE,
) -> Dict[str, float]:
    """Compute tier-weighted allocation with headwind+drawdown exit."""
    # Load headwind streak from history
    headwind_streak: Dict[str, int] = defaultdict(int)
    if history_path.exists():
        with open(history_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    for ticker, streak in row.get("headwind_streaks", {}).items():
                        headwind_streak[ticker] = streak
                except json.JSONDecodeError:
                    continue

    # Update streaks with current day
    for ticker in positions:
        r = rankings.get(ticker, {})
        if r.get("mom_state") == "headwind" and "deep_drawdown" in r.get("risk_flags", ""):
            headwind_streak[ticker] += 1
        else:
            headwind_streak[ticker] = 0

    raw = {}
    excluded = []
    for ticker in positions:
        r = rankings.get(ticker, {})
        tier = r.get("tier_dev", "")
        w = tier_weights.get(tier, 0.0)

        if headwind_streak[ticker] >= exit_persistence:
            w = 0.0
            excluded.append(ticker)

        if w > 0:
            raw[ticker] = w

    total = sum(raw.values())
    if total <= 0:
        return {}
    weights = {t: round(w / total * 100, 2) for t, w in raw.items()}
    return weights, excluded, dict(headwind_streak)


def compute_daily_pnl(
    weights: Dict[str, float],
    prices: Dict[str, Dict[str, float]],
    date: str,
    prior_date: str,
) -> float:
    """Compute weighted portfolio return for one day."""
    total_ret = 0.0
    for ticker, w in weights.items():
        p0 = _get_price(prices, ticker, prior_date)
        p1 = _get_price(prices, ticker, date)
        if p0 and p1 and p0 > 0:
            total_ret += (w / 100.0) * ((p1 - p0) / p0)
    return round(total_ret * 100, 4)


def build_policy_shadow_compare(
    *,
    as_of_date: str,
    pos_dir: Path = REPO_ROOT / "artifacts" / "live_shadow" / "positions",
    snap_dir: Path = REPO_ROOT / "data" / "snapshots",
    price_path: Path = REPO_ROOT / "production_data" / "price_history.csv",
    output_dir: Path = REPO_ROOT / "artifacts" / "policy_shadow" / "tier_weighted",
    tier_weights: Dict[str, float] = None,
) -> Dict[str, Any]:
    """Build daily policy shadow comparison."""
    if tier_weights is None:
        tier_weights = DEFAULT_TIER_WEIGHTS

    # Load current positions
    pos_data = _load_json(pos_dir / f"{as_of_date}.json")
    if not pos_data:
        return {"error": f"no positions for {as_of_date}"}
    positions = {p["ticker"]: p for p in pos_data.get("positions", [])}

    # Load rankings
    rankings = _load_rankings(snap_dir, as_of_date)
    if not rankings:
        return {"error": f"no rankings for {as_of_date}"}

    # Load prices
    prices = _load_prices(price_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"

    # Compute weights under each policy
    w_current = {t: p.get("weight_pct", 3.0) for t, p in positions.items()}
    total_c = sum(w_current.values())
    if total_c > 0:
        w_current = {t: round(w / total_c * 100, 2) for t, w in w_current.items()}

    w_tiered = compute_tiered_weights(positions, rankings, tier_weights)
    w_exit_result = compute_tiered_exit_weights(positions, rankings, tier_weights, history_path)
    w_tiered_exit, excluded_tickers, headwind_streaks = w_exit_result

    # Find prior date for P&L
    pos_files = sorted(f.stem for f in pos_dir.glob("*.json"))
    prior_idx = pos_files.index(as_of_date) - 1 if as_of_date in pos_files else -1
    prior_date = pos_files[prior_idx] if prior_idx >= 0 else None

    pnl = {}
    if prior_date:
        pnl["current"] = compute_daily_pnl(w_current, prices, as_of_date, prior_date)
        pnl["tiered"] = compute_daily_pnl(w_tiered, prices, as_of_date, prior_date)
        pnl["tiered_exit"] = compute_daily_pnl(w_tiered_exit, prices, as_of_date, prior_date)

    # Position overlap
    current_set = set(w_current.keys())
    tiered_set = set(w_tiered.keys())
    exit_set = set(w_tiered_exit.keys())
    overlap_tiered = len(current_set & tiered_set) / max(len(current_set | tiered_set), 1)
    overlap_exit = len(current_set & exit_set) / max(len(current_set | exit_set), 1)

    # Tier distribution
    tier_dist = {"current": Counter(), "tiered": Counter(), "tiered_exit": Counter()}
    for t in w_current:
        tier_dist["current"][rankings.get(t, {}).get("tier_dev", "?")] += 1
    for t in w_tiered:
        tier_dist["tiered"][rankings.get(t, {}).get("tier_dev", "?")] += 1
    for t in w_tiered_exit:
        tier_dist["tiered_exit"][rankings.get(t, {}).get("tier_dev", "?")] += 1

    # Weight concentration
    def _top_n_weight(weights, n):
        return round(sum(sorted(weights.values(), reverse=True)[:n]), 1)

    result = {
        "schema": SCHEMA_VERSION,
        "as_of_date": as_of_date,
        "prior_date": prior_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tier_weights": tier_weights,
        "n_positions": {
            "current": len(w_current),
            "tiered": len(w_tiered),
            "tiered_exit": len(w_tiered_exit),
        },
        "daily_pnl_pct": pnl,
        "overlap": {
            "current_vs_tiered": round(overlap_tiered, 3),
            "current_vs_tiered_exit": round(overlap_exit, 3),
        },
        "excluded_by_exit": excluded_tickers,
        "tier_distribution": {p: dict(d) for p, d in tier_dist.items()},
        "concentration": {
            "current_top5": _top_n_weight(w_current, 5),
            "tiered_top5": _top_n_weight(w_tiered, 5),
            "tiered_exit_top5": _top_n_weight(w_tiered_exit, 5),
        },
        "weight_changes_top5": [],
    }

    # Top weight changes (biggest differences between current and tiered)
    weight_deltas = []
    for t in current_set | tiered_set:
        wc = w_current.get(t, 0)
        wt = w_tiered.get(t, 0)
        delta = wt - wc
        if abs(delta) > 0.1:
            weight_deltas.append(
                {
                    "ticker": t,
                    "current_wt": wc,
                    "tiered_wt": wt,
                    "delta": round(delta, 2),
                    "tier": rankings.get(t, {}).get("tier_dev", "?"),
                }
            )
    weight_deltas.sort(key=lambda x: -abs(x["delta"]))
    result["weight_changes_top5"] = weight_deltas[:10]

    # Write daily comparison
    json_path = output_dir / f"{as_of_date}_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    # Append to history (idempotent: skip if this date is already present).
    # Comparison artifact above is regenerated unconditionally; only the
    # append is gated, since duplicate same-date rows can mislead any
    # consumer that doesn't dedupe by date.
    if _date_in_history(history_path, as_of_date):
        logger.info("history already contains %s; skipping append", as_of_date)
    else:
        history_row = {
            "date": as_of_date,
            "n_current": len(w_current),
            "n_tiered": len(w_tiered),
            "n_exit": len(w_tiered_exit),
            "pnl_current": pnl.get("current"),
            "pnl_tiered": pnl.get("tiered"),
            "pnl_exit": pnl.get("tiered_exit"),
            "overlap": overlap_tiered,
            "excluded": excluded_tickers,
            "headwind_streaks": headwind_streaks,
        }
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(history_row, default=str) + "\n")

    # Markdown
    md_path = output_dir / f"{as_of_date}_comparison.md"
    md_path.write_text(_format_md(result), encoding="utf-8")

    logger.info("Wrote %s", json_path)
    logger.info(
        "P&L: current=%.2f%% tiered=%.2f%% exit=%.2f%% | overlap=%.1f%% | excluded=%s",
        pnl.get("current", 0),
        pnl.get("tiered", 0),
        pnl.get("tiered_exit", 0),
        overlap_tiered * 100,
        excluded_tickers or "none",
    )

    return result


def _format_md(d: Dict) -> str:
    lines = []
    lines.append(f"# Policy Shadow Compare — {d['as_of_date']}")
    lines.append("")
    tw = d["tier_weights"]
    lines.append(f"Weights: A={tw['A']} / B={tw['B']} / C={tw['C']} / D={tw['D']}")
    lines.append("")

    pnl = d.get("daily_pnl_pct", {})
    if pnl:
        lines.append("## Daily P&L")
        lines.append("")
        lines.append("| Policy | P&L |")
        lines.append("|--------|-----|")
        for p in ["current", "tiered", "tiered_exit"]:
            lines.append(f"| {p} | {pnl.get(p, 0):+.2f}% |")
        lines.append("")

    lines.append("## Positions")
    lines.append("")
    np = d.get("n_positions", {})
    lines.append("| Policy | N | Top 5 wt | Overlap |")
    lines.append("|--------|---|----------|---------|")
    conc = d.get("concentration", {})
    ovl = d.get("overlap", {})
    lines.append(f"| current | {np.get('current', 0)} | {conc.get('current_top5', 0)}% | — |")
    lines.append(
        f"| tiered | {np.get('tiered', 0)} | {conc.get('tiered_top5', 0)}% | {ovl.get('current_vs_tiered', 0)*100:.0f}% |"
    )
    lines.append(
        f"| tiered_exit | {np.get('tiered_exit', 0)} | {conc.get('tiered_exit_top5', 0)}% | {ovl.get('current_vs_tiered_exit', 0)*100:.0f}% |"
    )
    lines.append("")

    excluded = d.get("excluded_by_exit", [])
    if excluded:
        lines.append(f"**Excluded by headwind exit:** {', '.join(excluded)}")
        lines.append("")

    changes = d.get("weight_changes_top5", [])
    if changes:
        lines.append("## Biggest Weight Changes (current → tiered)")
        lines.append("")
        lines.append("| Ticker | Current | Tiered | Delta | Tier |")
        lines.append("|--------|---------|--------|-------|------|")
        for c in changes[:10]:
            lines.append(
                f"| {c['ticker']} | {c['current_wt']:.1f}% | {c['tiered_wt']:.1f}% | {c['delta']:+.1f}% | {c['tier']} |"
            )
        lines.append("")

    lines.append(f"*Generated: {d.get('generated_at', '')}*")
    lines.append("")
    return "\n".join(lines)


def _latest_history_date(history_path: Path) -> Optional[str]:
    if not history_path.exists():
        return None
    latest = None
    try:
        with open(history_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line).get("date")
                except json.JSONDecodeError:
                    continue
                if d and (latest is None or d > latest):
                    latest = d
    except OSError:
        return None
    return latest


def _date_in_history(history_path: Path, date: str) -> bool:
    if not history_path.exists():
        return False
    try:
        with open(history_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    if json.loads(line).get("date") == date:
                        return True
                except json.JSONDecodeError:
                    continue
    except OSError:
        return False
    return False


def _identify_caller() -> str:
    ppid = os.getppid()
    try:
        cmdline = Path(f"/proc/{ppid}/cmdline").read_text(errors="replace").replace("\0", " ").strip()
        return f"ppid={ppid} cmd={cmdline}" if cmdline else f"ppid={ppid}"
    except OSError:
        return f"ppid={ppid}"


def main():
    parser = argparse.ArgumentParser(description="Daily policy shadow compare (Spec 035)")
    parser.add_argument("--as-of-date", default=None)
    parser.add_argument(
        "--allow-backfill",
        action="store_true",
        help="Permit --as-of-date older than the latest history.jsonl entry. "
        "Default: refuse, to prevent the contamination pattern observed on 2026-05-06 (P0 #1).",
    )
    args = parser.parse_args()
    started = time.perf_counter()

    # Default to latest position date
    pos_dir = REPO_ROOT / "artifacts" / "live_shadow" / "positions"
    if args.as_of_date:
        date = args.as_of_date
    else:
        pos_files = sorted(f.stem for f in pos_dir.glob("*.json"))
        date = pos_files[-1] if pos_files else datetime.now(timezone.utc).strftime("%Y-%m-%d")

    history_path = REPO_ROOT / "artifacts" / "policy_shadow" / "tier_weighted" / "history.jsonl"
    latest = _latest_history_date(history_path)
    if latest and date < latest and not args.allow_backfill:
        logger.error(
            "Refusing to append: --as-of-date %s is older than latest history entry %s. "
            "This caused historical contamination on 2026-05-06 (P0 #1). "
            "Pass --allow-backfill to override. Caller: %s",
            date,
            latest,
            _identify_caller(),
        )
        sys.exit(2)

    result = build_policy_shadow_compare(as_of_date=date)
    if "error" in result:
        logger.error(result["error"])
        try:
            from tools.agent_skill_telemetry import log_agent_run

            log_agent_run(
                "build_policy_shadow_compare",
                f"Policy shadow compare for {date}",
                inputs={"as_of_date": date},
                outputs={"error": result["error"]},
                success=False,
                error=result["error"],
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception:
            pass
        sys.exit(1)
    try:
        from tools.agent_skill_telemetry import log_agent_run

        log_agent_run(
            "build_policy_shadow_compare",
            f"Policy shadow compare for {date}",
            inputs={"as_of_date": date},
            success=True,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
    except Exception:
        pass


if __name__ == "__main__":
    main()
