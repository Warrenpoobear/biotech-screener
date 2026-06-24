#!/usr/bin/env python3
"""Shadow portfolio performance monitor — daily triage briefing.

Reads shadow portfolio performance history, position data, and readiness
scorecard to produce a structured alert/briefing artifact. Flags drawdown
streaks, excess deterioration, sleeve blowups, single-name concentration,
and noteworthy wins.

Read-only — does not affect rankings, scoring, or execution.

Output:
    artifacts/shadow_monitor/{date}_monitor.json
    artifacts/shadow_monitor/{date}_monitor.md

Usage:
    python tools/build_shadow_monitor.py --as-of-date 2026-03-27
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
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("shadow_monitor")

SCHEMA_VERSION = "shadow_monitor.v1"

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
THRESHOLDS = {
    # Drawdown streak
    "drawdown_streak_warn": 3,  # consecutive losing days
    "drawdown_streak_alert": 5,
    # Single-day loss
    "single_day_loss_warn": -2.0,  # percent
    "single_day_loss_alert": -4.0,
    # Cumulative excess vs XBI
    "excess_warn": -3.0,  # percent
    "excess_alert": -6.0,
    # Sleeve concentration
    "sleeve_loss_pct_warn": 60.0,  # one sleeve is >60% of total loss
    # Single-name win threshold (for noteworthy wins)
    "single_name_win_pct": 15.0,  # >15% return
    # Max drawdown
    "max_dd_warn": 8.0,
    "max_dd_alert": 12.0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


def load_performance(perf_path: Path) -> List[Dict[str, Any]]:
    """Load performance.csv into list of dicts with parsed floats."""
    if not perf_path.exists():
        return []
    rows = []
    with open(perf_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed = {
                "date": row.get("date", ""),
                "prior_date": row.get("prior_date", ""),
                "pnl": _sf(row.get("total_pnl", "")),
                "pnl_pct": _sf(row.get("pnl_pct", "")),
                "xbi_return_pct": _sf(row.get("xbi_return_pct", "")),
                "excess_pct": _sf(row.get("excess_vs_xbi_pct", "")),
                "n_held": _sf(row.get("n_held", "")),
                "turnover": _sf(row.get("turnover", "")),
                "gap_risk_high_count": _sf(row.get("gap_risk_high_count", "")),
                "sleeve_0_30": _sf(row.get("sleeve_binary_0_30_pnl", "")),
                "sleeve_31_90": _sf(row.get("sleeve_binary_31_90_pnl", "")),
                "sleeve_91_180": _sf(row.get("sleeve_binary_91_180_pnl", "")),
                "sleeve_less_binary": _sf(row.get("sleeve_less_binary_pnl", "")),
                "ruleset_id": row.get("ruleset_id", ""),
            }
            rows.append(parsed)
    return rows


def compute_drawdown_streak(rows: List[Dict[str, Any]]) -> int:
    """Count consecutive losing days from the most recent row backward."""
    streak = 0
    for row in reversed(rows):
        pnl = row.get("pnl_pct", math.nan)
        if math.isnan(pnl) or pnl >= 0:
            break
        streak += 1
    return streak


def compute_cumulative(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute cumulative metrics from performance rows."""
    total_pnl = 0.0
    total_excess = 0.0
    max_cum_pnl = 0.0
    max_drawdown = 0.0
    cum_pnl = 0.0
    n_win = 0
    n_total = 0

    for row in rows:
        pnl_pct = row.get("pnl_pct", math.nan)
        excess = row.get("excess_pct", math.nan)

        if not math.isnan(pnl_pct):
            total_pnl += pnl_pct
            cum_pnl += pnl_pct
            max_cum_pnl = max(max_cum_pnl, cum_pnl)
            dd = max_cum_pnl - cum_pnl
            max_drawdown = max(max_drawdown, dd)
            n_total += 1
            if pnl_pct > 0:
                n_win += 1

        if not math.isnan(excess):
            total_excess += excess

    sleeve_totals = {"0_30": 0.0, "31_90": 0.0, "91_180": 0.0, "less_binary": 0.0}
    for row in rows:
        for key, field in [
            ("0_30", "sleeve_0_30"),
            ("31_90", "sleeve_31_90"),
            ("91_180", "sleeve_91_180"),
            ("less_binary", "sleeve_less_binary"),
        ]:
            v = row.get(field, math.nan)
            if not math.isnan(v):
                sleeve_totals[key] += v

    return {
        "total_pnl_pct": round(total_pnl, 4),
        "total_excess_pct": round(total_excess, 4),
        "max_drawdown_pct": round(max_drawdown, 4),
        "win_rate": round(n_win / n_total, 4) if n_total > 0 else 0.0,
        "n_periods": n_total,
        "sleeve_totals": {k: round(v, 2) for k, v in sleeve_totals.items()},
    }


def find_noteworthy_positions(
    positions_path: Path,
    price_csv: Path,
    as_of_date: str,
) -> List[Dict[str, Any]]:
    """Find positions with extreme returns (winners and losers)."""
    pos_data = _load_json(positions_path)
    if not pos_data:
        return []

    positions = pos_data.get("positions", [])
    if not positions:
        return []

    # Load latest prices
    prices: Dict[str, float] = {}
    if price_csv.exists():
        with open(price_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("date") == as_of_date or (
                    not prices.get(row.get("ticker")) and row.get("date", "") >= as_of_date[:8]
                ):
                    t = row.get("ticker", "")
                    c = _sf(row.get("close", ""))
                    if t and not math.isnan(c):
                        prices[t] = c

    noteworthy = []
    for p in positions:
        ticker = p.get("ticker", "")
        entry = _sf(p.get("entry_price", ""))
        current = prices.get(ticker, math.nan)

        if math.isnan(entry) or math.isnan(current) or entry <= 0:
            continue

        ret_pct = (current - entry) / entry * 100
        if abs(ret_pct) >= THRESHOLDS["single_name_win_pct"]:
            noteworthy.append(
                {
                    "ticker": ticker,
                    "entry_price": round(entry, 2),
                    "current_price": round(current, 2),
                    "return_pct": round(ret_pct, 1),
                    "bucket": p.get("bucket", ""),
                }
            )

    noteworthy.sort(key=lambda x: x["return_pct"])
    return noteworthy


# ---------------------------------------------------------------------------
# Alert classification
# ---------------------------------------------------------------------------
def classify_alerts(
    rows: List[Dict[str, Any]],
    cumulative: Dict[str, Any],
    scorecard: Optional[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Generate alerts from performance data."""
    alerts = []

    if not rows:
        return alerts

    latest = rows[-1]

    # Drawdown streak
    streak = compute_drawdown_streak(rows)
    if streak >= THRESHOLDS["drawdown_streak_alert"]:
        alerts.append({"level": "ALERT", "code": "DRAWDOWN_STREAK", "detail": f"{streak} consecutive losing days"})
    elif streak >= THRESHOLDS["drawdown_streak_warn"]:
        alerts.append({"level": "WARN", "code": "DRAWDOWN_STREAK", "detail": f"{streak} consecutive losing days"})

    # Single-day loss
    latest_pnl = latest.get("pnl_pct", math.nan)
    if not math.isnan(latest_pnl):
        if latest_pnl <= THRESHOLDS["single_day_loss_alert"]:
            alerts.append({"level": "ALERT", "code": "SINGLE_DAY_LOSS", "detail": f"{latest_pnl:.2f}%"})
        elif latest_pnl <= THRESHOLDS["single_day_loss_warn"]:
            alerts.append({"level": "WARN", "code": "SINGLE_DAY_LOSS", "detail": f"{latest_pnl:.2f}%"})

    # Cumulative excess
    excess = cumulative["total_excess_pct"]
    if excess <= THRESHOLDS["excess_alert"]:
        alerts.append({"level": "ALERT", "code": "EXCESS_DETERIORATION", "detail": f"{excess:.2f}% vs XBI"})
    elif excess <= THRESHOLDS["excess_warn"]:
        alerts.append({"level": "WARN", "code": "EXCESS_DETERIORATION", "detail": f"{excess:.2f}% vs XBI"})

    # Max drawdown
    dd = cumulative["max_drawdown_pct"]
    if dd >= THRESHOLDS["max_dd_alert"]:
        alerts.append({"level": "ALERT", "code": "MAX_DRAWDOWN", "detail": f"{dd:.2f}%"})
    elif dd >= THRESHOLDS["max_dd_warn"]:
        alerts.append({"level": "WARN", "code": "MAX_DRAWDOWN", "detail": f"{dd:.2f}%"})

    # Sleeve concentration
    sleeve_totals = cumulative.get("sleeve_totals", {})
    total_loss = sum(v for v in sleeve_totals.values() if v < 0)
    if total_loss < 0:
        for sleeve, pnl in sleeve_totals.items():
            if pnl < 0:
                pct_of_loss = abs(pnl) / abs(total_loss) * 100
                if pct_of_loss >= THRESHOLDS["sleeve_loss_pct_warn"]:
                    alerts.append(
                        {
                            "level": "WARN",
                            "code": "SLEEVE_CONCENTRATION",
                            "detail": f"{sleeve} is {pct_of_loss:.0f}% of total loss (${pnl:,.0f})",
                        }
                    )

    # Scorecard FAIL checks
    if scorecard:
        for check in scorecard.get("checks", []):
            if check.get("status") == "FAIL":
                alerts.append(
                    {
                        "level": "WARN",
                        "code": "SCORECARD_FAIL",
                        "detail": f"{check['name']}: {check.get('detail', '')}",
                    }
                )

    return alerts


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
def build_shadow_monitor(
    as_of_date: str,
    *,
    artifacts_dir: Path = REPO_ROOT / "artifacts",
    snapshots_dir: Path = REPO_ROOT / "data" / "snapshots",
    price_csv: Path = REPO_ROOT / "production_data" / "price_history.csv",
) -> Dict[str, Any]:
    """Build shadow monitor artifact."""
    perf_path = artifacts_dir / "live_shadow" / "performance.csv"
    rows = load_performance(perf_path)
    if not rows:
        return {"error": "no performance data"}

    # Filter to current ruleset rows up to as_of_date
    rows = [r for r in rows if r["date"] <= as_of_date]
    if not rows:
        return {"error": f"no performance data through {as_of_date}"}

    latest = rows[-1]
    cumulative = compute_cumulative(rows)

    # Scorecard
    scorecard_path = artifacts_dir / "readiness" / f"scorecard_{as_of_date}.json"
    scorecard = _load_json(scorecard_path)

    # Alerts
    alerts = classify_alerts(rows, cumulative, scorecard)

    # Noteworthy positions
    pos_path = artifacts_dir / "live_shadow" / "positions" / f"{as_of_date}.json"
    noteworthy = find_noteworthy_positions(pos_path, price_csv, as_of_date)

    # Recent trend (last 5 days)
    recent = rows[-5:] if len(rows) >= 5 else rows
    recent_summary = []
    for r in recent:
        pnl = r.get("pnl_pct", math.nan)
        excess = r.get("excess_pct", math.nan)
        recent_summary.append(
            {
                "date": r["date"],
                "pnl_pct": round(pnl, 2) if not math.isnan(pnl) else None,
                "excess_pct": round(excess, 2) if not math.isnan(excess) else None,
            }
        )

    # Attention level
    n_alerts = sum(1 for a in alerts if a["level"] == "ALERT")
    n_warns = sum(1 for a in alerts if a["level"] == "WARN")
    if n_alerts > 0:
        attention = "HIGH"
    elif n_warns > 0:
        attention = "MEDIUM"
    else:
        attention = "LOW"

    result = {
        "schema": SCHEMA_VERSION,
        "as_of_date": as_of_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "attention": attention,
        "latest": {
            "date": latest["date"],
            "pnl_pct": round(latest.get("pnl_pct", 0), 4),
            "excess_pct": (
                round(latest.get("excess_pct", 0), 4) if not math.isnan(latest.get("excess_pct", math.nan)) else None
            ),
            "n_held": int(latest.get("n_held", 0)) if not math.isnan(latest.get("n_held", math.nan)) else None,
        },
        "cumulative": cumulative,
        "drawdown_streak": compute_drawdown_streak(rows),
        "alerts": alerts,
        "noteworthy_positions": noteworthy,
        "recent_trend": recent_summary,
        "scorecard_verdict": scorecard.get("verdict") if scorecard else None,
        "thresholds": THRESHOLDS,
    }

    # Write artifacts
    out_dir = artifacts_dir / "shadow_monitor"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{as_of_date}_monitor.json"
    md_path = out_dir / f"{as_of_date}_monitor.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Wrote %s", json_path)

    md_text = format_monitor_md(result)
    md_path.write_text(md_text, encoding="utf-8")
    logger.info("Wrote %s", md_path)

    result["_json_path"] = str(json_path)
    result["_md_path"] = str(md_path)
    return result


# ---------------------------------------------------------------------------
# Markdown formatter
# ---------------------------------------------------------------------------
def format_monitor_md(d: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"# Shadow Monitor — {d['as_of_date']}")
    lines.append("")
    lines.append(f"**Attention: {d['attention']}**")
    lines.append("")

    # Latest
    latest = d.get("latest", {})
    cum = d.get("cumulative", {})
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Latest PnL | {latest.get('pnl_pct', '?')}% |")
    lines.append(f"| Latest excess | {latest.get('excess_pct', '?')}% |")
    lines.append(f"| Cumulative PnL | {cum.get('total_pnl_pct', '?')}% |")
    lines.append(f"| Cumulative excess | {cum.get('total_excess_pct', '?')}% |")
    lines.append(f"| Max drawdown | {cum.get('max_drawdown_pct', '?')}% |")
    lines.append(f"| Win rate | {cum.get('win_rate', 0):.0%} |")
    lines.append(f"| Periods | {cum.get('n_periods', 0)} |")
    lines.append(f"| Drawdown streak | {d.get('drawdown_streak', 0)} days |")
    lines.append(f"| Scorecard | {d.get('scorecard_verdict', '?')} |")
    lines.append("")

    # Alerts
    alerts = d.get("alerts", [])
    if alerts:
        lines.append("## Alerts")
        lines.append("")
        for a in alerts:
            lines.append(f"- **{a['level']}** [{a['code']}]: {a['detail']}")
        lines.append("")

    # Sleeve attribution
    sleeves = cum.get("sleeve_totals", {})
    if sleeves:
        lines.append("## Sleeve Attribution")
        lines.append("")
        lines.append("| Sleeve | Cumulative P&L |")
        lines.append("|--------|---------------|")
        for s, v in sleeves.items():
            lines.append(f"| {s} | ${v:,.0f} |")
        lines.append("")

    # Recent trend
    recent = d.get("recent_trend", [])
    if recent:
        lines.append("## Recent Trend")
        lines.append("")
        lines.append("| Date | PnL | Excess |")
        lines.append("|------|-----|--------|")
        for r in recent:
            pnl = f"{r['pnl_pct']}%" if r.get("pnl_pct") is not None else "?"
            excess = f"{r['excess_pct']}%" if r.get("excess_pct") is not None else "?"
            lines.append(f"| {r['date']} | {pnl} | {excess} |")
        lines.append("")

    # Noteworthy positions
    noteworthy = d.get("noteworthy_positions", [])
    if noteworthy:
        lines.append("## Noteworthy Positions")
        lines.append("")
        lines.append("| Ticker | Entry | Current | Return | Bucket |")
        lines.append("|--------|-------|---------|--------|--------|")
        for n in noteworthy:
            lines.append(
                f"| {n['ticker']} | ${n['entry_price']:.2f} | ${n['current_price']:.2f} | "
                f"{n['return_pct']:+.1f}% | {n['bucket']} |"
            )
        lines.append("")

    lines.append(f"*Generated: {d.get('generated_at', '')}*")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Shadow portfolio performance monitor")
    parser.add_argument("--as-of-date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument("--artifacts-dir", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument("--snapshots-dir", type=Path, default=REPO_ROOT / "data" / "snapshots")
    parser.add_argument("--price-csv", type=Path, default=REPO_ROOT / "production_data" / "price_history.csv")
    args = parser.parse_args()
    started = time.perf_counter()

    result = build_shadow_monitor(
        args.as_of_date,
        artifacts_dir=args.artifacts_dir,
        snapshots_dir=args.snapshots_dir,
        price_csv=args.price_csv,
    )

    if "error" in result:
        logger.error(result["error"])
        try:
            from tools.agent_skill_telemetry import log_agent_run

            log_agent_run(
                "build_shadow_monitor",
                f"Shadow monitor for {args.as_of_date}",
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
        "Monitor: %s attention, %d alerts (%s)", result["attention"], len(result["alerts"]), result["as_of_date"]
    )
    try:
        from tools.agent_skill_telemetry import log_agent_run
        from tools.record_skill_feedback import attach_outcome_verdict

        attention = result.get("attention", "UNKNOWN")
        exec_id = log_agent_run(
            "build_shadow_monitor",
            f"Shadow monitor for {args.as_of_date}",
            inputs={"as_of_date": args.as_of_date},
            outputs={"attention": attention, "n_alerts": len(result.get("alerts", []))},
            success=True,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        if exec_id:
            attach_outcome_verdict(
                exec_id,
                was_correct=attention in ("LOW", "MEDIUM"),
                evidence=f"attention={attention}",
            )
    except Exception:
        pass


if __name__ == "__main__":
    main()
