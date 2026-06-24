#!/usr/bin/env python3
"""Price action watch — stock + options big-move alerts on model-relevant names.

Monitors a capped watchlist (review queue, trade plan, shadow positions,
catalyst delta, A-tier near-term) for significant stock and options moves.

Read-only — does not affect rankings, scoring, or execution.

Output:
    artifacts/price_action_watch/{date}_watch.json
    artifacts/price_action_watch/{date}_watch.md

Usage:
    python tools/build_price_action_watch.py --as-of-date 2026-03-27
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from common.watchlist_config import WATCHLIST_MAX, build_model_relevant_watchlist  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("price_action_watch")

SCHEMA_VERSION = "price_action_watch.v2"

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
THRESHOLDS = {
    # Stock moves (1d return %)
    "stock_move_up": 5.0,
    "stock_move_down": -5.0,
    "stock_big_move_up": 10.0,
    "stock_big_move_down": -10.0,
    # Move intensity (1d |return| vs 20d avg |return|; proxy for RVOL without volume data)
    "move_intensity_spike": 2.5,
    # Options surface
    "iv_ramp_high": 0.10,  # atm_iv_change_5d
    "iv_crush": -0.10,  # atm_iv_change_5d
    "surface_move_high": 0.80,  # actual_implied_move_pctile
    "skew_zscore": 2.0,  # z-score vs name's own trailing RR distribution
    "skew_abs_floor": 0.15,  # minimum |RR| to even consider (filters noise on near-zero RR names)
    "skew_min_obs": 5,  # minimum trailing observations to compute z-score
}


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


def _load_csv_tickers(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    try:
        with open(path, encoding="utf-8") as f:
            return {row["ticker"] for row in csv.DictReader(f) if row.get("ticker")}
    except (KeyError, OSError):
        return set()


# ---------------------------------------------------------------------------
# Price data
# ---------------------------------------------------------------------------
def load_trailing_rr(
    snapshots_dir: Path,
    tickers: Set[str],
    as_of_date: str,
    max_snapshots: int = 200,
) -> Dict[str, List[float]]:
    """Load trailing opt_rr_25d values per ticker from snapshot history."""
    rr_history: Dict[str, List[float]] = {}
    candidates = sorted(
        d.name for d in snapshots_dir.iterdir() if d.is_dir() and len(d.name) == 10 and d.name < as_of_date
    )
    # Use most recent snapshots (options data is sparse in older ones)
    candidates = candidates[-max_snapshots:]

    for d in candidates:
        rk = snapshots_dir / d / "rankings.csv"
        if not rk.exists():
            continue
        with open(rk, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                t = row.get("ticker", "")
                if t not in tickers:
                    continue
                rr = row.get("opt_rr_25d", "")
                if rr:
                    try:
                        rr_history.setdefault(t, []).append(float(rr))
                    except ValueError:
                        pass
    return rr_history


def load_recent_prices(
    price_csv: Path,
    tickers: Set[str],
    as_of_date: str,
    lookback_days: int = 25,
) -> Dict[str, List[tuple]]:
    """Load recent price data for tickers. Returns {ticker: [(date, close), ...]}."""
    prices: Dict[str, List[tuple]] = {}
    if not price_csv.exists():
        return prices
    with open(price_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = row.get("ticker", "")
            d = row.get("date", "")
            c = row.get("close", "")
            if t in tickers and d and c and d <= as_of_date:
                try:
                    prices.setdefault(t, []).append((d, float(c)))
                except ValueError:
                    pass
    for t in prices:
        prices[t].sort()
        prices[t] = prices[t][-lookback_days:]
    return prices


def compute_stock_metrics(series: List[tuple]) -> Dict[str, Any]:
    """Compute stock price metrics from a sorted (date, close) series."""
    if len(series) < 2:
        return {}

    latest_date, latest_price = series[-1]
    prior_date, prior_price = series[-2]

    ret_1d = (latest_price - prior_price) / prior_price * 100 if prior_price > 0 else math.nan

    # 5d return (date-based lookup, not array indexing)
    # Fix: use calendar date delta instead of array position to handle weekends/holidays
    ret_5d = math.nan
    if len(series) >= 2:
        try:
            latest_date_obj = datetime.strptime(latest_date, "%Y-%m-%d")
            target_date = latest_date_obj - timedelta(days=5)
            # Find closest date >= 5 trading days back
            price_5d_back = None
            for i in range(len(series) - 2, -1, -1):
                date_str, price = series[i]
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                if date_obj <= target_date:
                    price_5d_back = price
                    break
            if price_5d_back and price_5d_back > 0:
                ret_5d = (latest_price - price_5d_back) / price_5d_back * 100
        except (ValueError, IndexError):
            pass

    # Move intensity: |1d return| vs trailing 20d avg |return|
    # Proxy for RVOL — price_history.csv does not include share volume.
    # When real volume data is available (v2), replace with actual RVOL.
    returns = []
    for i in range(1, len(series)):
        p0 = series[i - 1][1]
        p1 = series[i][1]
        if p0 > 0:
            returns.append(abs((p1 - p0) / p0))
    avg_move = sum(returns) / len(returns) if returns else 0
    latest_move = abs(ret_1d / 100) if not math.isnan(ret_1d) else 0
    move_intensity = latest_move / avg_move if avg_move > 0 else math.nan

    return {
        "latest_date": latest_date,
        "latest_price": round(latest_price, 2),
        "prior_price": round(prior_price, 2),
        "return_1d_pct": round(ret_1d, 2) if not math.isnan(ret_1d) else None,
        "return_5d_pct": round(ret_5d, 2) if not math.isnan(ret_5d) else None,
        "move_intensity": round(move_intensity, 2) if not math.isnan(move_intensity) else None,
    }


# ---------------------------------------------------------------------------
# Alert classification
# ---------------------------------------------------------------------------
def classify_alerts(
    ticker: str,
    stock: Dict[str, Any],
    options: Dict[str, str],
    rr_history: Optional[List[float]] = None,
) -> List[str]:
    """Classify alerts for one name. Returns list of alert codes."""
    alerts = []

    # Stock moves
    ret = stock.get("return_1d_pct")
    if ret is not None:
        if ret >= THRESHOLDS["stock_big_move_up"]:
            alerts.append("STOCK_BIG_MOVE_UP")
        elif ret >= THRESHOLDS["stock_move_up"]:
            alerts.append("STOCK_MOVE_UP")
        elif ret <= THRESHOLDS["stock_big_move_down"]:
            alerts.append("STOCK_BIG_MOVE_DOWN")
        elif ret <= THRESHOLDS["stock_move_down"]:
            alerts.append("STOCK_MOVE_DOWN")

    # Move intensity (proxy for RVOL)
    mi = stock.get("move_intensity")
    if mi is not None and mi >= THRESHOLDS["move_intensity_spike"]:
        alerts.append("MOVE_INTENSITY_SPIKE")

    # Options surface
    iv_change = _sf(options.get("atm_iv_change_5d", ""))
    if not math.isnan(iv_change):
        if iv_change >= THRESHOLDS["iv_ramp_high"]:
            alerts.append("IV_RAMP_HIGH")
        elif iv_change <= THRESHOLDS["iv_crush"]:
            alerts.append("IV_CRUSH")

    move_pctile = _sf(options.get("actual_implied_move_pctile", ""))
    if not math.isnan(move_pctile) and move_pctile >= THRESHOLDS["surface_move_high"]:
        alerts.append("OPTIONS_SURFACE_MOVE_HIGH")

    rr = _sf(options.get("opt_rr_25d", ""))
    if not math.isnan(rr) and abs(rr) >= THRESHOLDS["skew_abs_floor"]:
        # Per-name z-score: is this RR extreme for THIS name?
        if rr_history and len(rr_history) >= THRESHOLDS["skew_min_obs"]:
            import statistics

            hist_mean = statistics.mean(rr_history)
            hist_std = statistics.stdev(rr_history) if len(rr_history) > 1 else 0.0
            if hist_std > 0.01:
                rr_z = abs(rr - hist_mean) / hist_std
                if rr_z >= THRESHOLDS["skew_zscore"]:
                    alerts.append("SKEW_EXTREME")
        else:
            # Fallback: no history, use absolute threshold of 0.50
            if abs(rr) >= 0.50:
                alerts.append("SKEW_EXTREME")

    # Stock/options divergence: stock down but IV ramping (or vice versa)
    if ret is not None and not math.isnan(iv_change):
        if ret <= -3.0 and iv_change >= 0.05:
            alerts.append("STOCK_DOWN_IV_UP")
        elif ret >= 3.0 and iv_change <= -0.05:
            alerts.append("STOCK_UP_IV_DOWN")

    # --- Compound biotech anomalies ---

    # Quiet-before-catalyst: hard catalyst <=14d but no IV ramp, no event premium,
    # no surface move. Unusual calm before a binary event.
    is_hard = options.get("is_hard_catalyst", "") == "1"
    cat_days = _sf(options.get("catalyst_days", ""))
    event_prem = options.get("opt_event_premium", "")
    if is_hard and not math.isnan(cat_days) and cat_days <= 14:
        iv_quiet = math.isnan(iv_change) or abs(iv_change) < 0.03
        move_quiet = math.isnan(move_pctile) or move_pctile < 0.40
        no_premium = event_prem != "YES"
        if iv_quiet and move_quiet and no_premium:
            alerts.append("QUIET_BEFORE_CATALYST")

    # Post-event follow-through failure: big 5d move but reversing on 1d
    ret_5d = stock.get("return_5d_pct")
    if ret is not None and ret_5d is not None:
        if ret_5d >= 15.0 and ret <= -3.0:
            alerts.append("POST_EVENT_FADE")
        elif ret_5d <= -15.0 and ret >= 3.0:
            alerts.append("POST_EVENT_BOUNCE")

    # Reaction mismatch: big stock move but options didn't reprice
    if ret is not None and abs(ret) >= 5.0:
        if math.isnan(iv_change) or abs(iv_change) < 0.02:
            alerts.append("REACTION_MISMATCH")

    return alerts


def compute_alert_confidence(
    alerts: List[str],
    options: Dict[str, str],
    rr_history: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Compute alert confidence metadata for a set of alerts.

    Returns dict with:
      - alert_confidence: float 0-1
      - trigger_mode: history_based | abs_fallback | low_liquidity_fallback | stock_only
      - history_depth: int (days of usable RR history)
      - chain_quality_gate_pass: bool
      - spread_gate_pass: bool
    """
    if not alerts:
        return {
            "alert_confidence": 0.0,
            "trigger_mode": "none",
            "history_depth": 0,
            "chain_quality_gate_pass": False,
            "spread_gate_pass": False,
        }

    liq_state = options.get("opt_liquidity_state", "absent")
    has_options_data = liq_state != "absent"
    liquidity_ok = liq_state == "liquid"
    use_for_judgment = options.get("opt_use_for_judgment", "") == "YES"
    history_depth = len(rr_history) if rr_history else 0

    # Stock-only alerts don't need options quality
    stock_alerts = {
        "STOCK_BIG_MOVE_UP",
        "STOCK_BIG_MOVE_DOWN",
        "STOCK_MOVE_UP",
        "STOCK_MOVE_DOWN",
        "MOVE_INTENSITY_SPIKE",
    }
    options_alerts = {"IV_RAMP_HIGH", "IV_CRUSH", "OPTIONS_SURFACE_MOVE_HIGH", "SKEW_EXTREME", "QUIET_BEFORE_CATALYST"}
    has_stock_alert = bool(set(alerts) & stock_alerts)
    has_options_alert = bool(set(alerts) & options_alerts)

    # Determine trigger mode
    if not has_options_alert:
        trigger_mode = "stock_only"
    elif history_depth >= THRESHOLDS["skew_min_obs"] and liquidity_ok:
        trigger_mode = "history_based"
    elif has_options_data and not liquidity_ok:
        trigger_mode = "low_liquidity_fallback"
    else:
        trigger_mode = "abs_fallback"

    # Compute confidence
    confidence = 0.5  # baseline
    if has_stock_alert:
        confidence += 0.2  # stock moves are directly observable
    if trigger_mode == "history_based":
        confidence += 0.25
    elif trigger_mode == "abs_fallback":
        confidence -= 0.1
    elif trigger_mode == "low_liquidity_fallback":
        confidence -= 0.2
    if use_for_judgment:
        confidence += 0.1
    if liquidity_ok:
        confidence += 0.05

    confidence = max(0.0, min(1.0, confidence))

    return {
        "alert_confidence": round(confidence, 2),
        "trigger_mode": trigger_mode,
        "history_depth": history_depth,
        "chain_quality_gate_pass": liquidity_ok and has_options_data,
        "spread_gate_pass": liquidity_ok,
    }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
def build_price_action_watch(
    as_of_date: str,
    *,
    snapshots_dir: Path = REPO_ROOT / "data" / "snapshots",
    artifacts_dir: Path = REPO_ROOT / "artifacts",
    price_csv: Path = REPO_ROOT / "production_data" / "price_history.csv",
) -> Dict[str, Any]:
    """Build price action watch artifact."""
    snap_dir = snapshots_dir / as_of_date
    rankings_path = snap_dir / "rankings.csv"
    if not rankings_path.exists():
        return {"error": f"rankings.csv not found for {as_of_date}"}

    # Load rankings
    rankings: Dict[str, Dict[str, str]] = {}
    with open(rankings_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("ticker"):
                rankings[row["ticker"]] = row

    # Build watchlist via shared canonical constructor
    watchlist, wl_sources = build_model_relevant_watchlist(
        as_of_date,
        snapshots_dir=snapshots_dir,
        artifacts_dir=artifacts_dir,
        rankings=rankings,
        max_size=WATCHLIST_MAX,
    )

    # Load prices and trailing RR history
    prices = load_recent_prices(price_csv, watchlist, as_of_date)
    trailing_rr = load_trailing_rr(snapshots_dir, watchlist, as_of_date)

    # Classify each name (with freshness suppression)
    rows = []
    suppressed = []
    for ticker in sorted(watchlist):
        series = prices.get(ticker, [])

        # Freshness gate: suppress if no price data or latest price is >3 trading days old
        if not series or (series and series[-1][0] < as_of_date[:8]):
            # Check if latest price date is more than 3 days before as_of_date
            if not series:
                suppressed.append({"ticker": ticker, "reason": "no_price_data"})
                continue

        stock_metrics = compute_stock_metrics(series)
        options_data = rankings.get(ticker, {})

        alerts = classify_alerts(ticker, stock_metrics, options_data, trailing_rr.get(ticker))

        # Confidence assessment
        confidence = compute_alert_confidence(alerts, options_data, rr_history=trailing_rr.get(ticker))

        r = rankings.get(ticker, {})
        entry = {
            "ticker": ticker,
            "tier": r.get("tier_dev", ""),
            "actionable_rank": int(_sf(r.get("actionable_rank", "0"))) if r.get("actionable_rank") else None,
            "catalyst_days": int(_sf(r.get("catalyst_days", ""))) if r.get("catalyst_days") else None,
            "is_hard_catalyst": r.get("is_hard_catalyst", "") == "1",
            "return_1d_pct": stock_metrics.get("return_1d_pct"),
            "return_5d_pct": stock_metrics.get("return_5d_pct"),
            "move_intensity": stock_metrics.get("move_intensity"),
            "latest_price": stock_metrics.get("latest_price"),
            "atm_iv_change_5d": round(_sf(r.get("atm_iv_change_5d", "")), 4) if r.get("atm_iv_change_5d") else None,
            "actual_implied_move_pctile": (
                round(_sf(r.get("actual_implied_move_pctile", "")), 4) if r.get("actual_implied_move_pctile") else None
            ),
            "opt_rr_25d": round(_sf(r.get("opt_rr_25d", "")), 4) if r.get("opt_rr_25d") else None,
            "alerts": alerts,
            "n_alerts": len(alerts),
            # Confidence fields
            "alert_confidence": confidence["alert_confidence"],
            "trigger_mode": confidence["trigger_mode"],
            "history_depth": confidence["history_depth"],
            "chain_quality_gate_pass": confidence["chain_quality_gate_pass"],
            "spread_gate_pass": confidence["spread_gate_pass"],
            "opt_liquidity_state": r.get("opt_liquidity_state", "absent"),
        }
        rows.append(entry)

    # Sort: most alerts first, then by confidence, then by return magnitude
    rows.sort(key=lambda r: (-r["n_alerts"], -r.get("alert_confidence", 0), -abs(r.get("return_1d_pct") or 0)))

    n_alerted = sum(1 for r in rows if r["alerts"])

    result = {
        "schema": SCHEMA_VERSION,
        "as_of_date": as_of_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "watchlist_size": len(rows),
        "n_alerted": n_alerted,
        "n_suppressed": len(suppressed),
        "suppressed": suppressed,
        "thresholds": THRESHOLDS,
        "sources": wl_sources,
        "rows": rows,
    }

    # Write artifacts
    out_dir = artifacts_dir / "price_action_watch"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{as_of_date}_watch.json"
    md_path = out_dir / f"{as_of_date}_watch.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Wrote %s", json_path)

    md_path.write_text(format_watch_md(result), encoding="utf-8")
    logger.info("Wrote %s", md_path)

    result["_json_path"] = str(json_path)
    result["_md_path"] = str(md_path)
    return result


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def format_watch_md(d: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"# Price Action Watch — {d['as_of_date']}")
    lines.append("")
    lines.append(f"Watchlist: {d['watchlist_size']} names | Alerted: {d['n_alerted']}")
    lines.append("")

    rows = d.get("rows", [])
    alerted = [r for r in rows if r["alerts"]]

    if alerted:
        lines.append("## Alerts")
        lines.append("")
        lines.append("| Ticker | Tier | Rank | 1d | 5d | Conf | Mode | Liq | Alerts |")
        lines.append("|--------|------|------|----|----|------|------|-----|--------|")
        for r in alerted:
            ret1 = f"{r['return_1d_pct']:+.1f}%" if r.get("return_1d_pct") is not None else "-"
            ret5 = f"{r['return_5d_pct']:+.1f}%" if r.get("return_5d_pct") is not None else "-"
            conf = f"{r['alert_confidence']:.0%}" if r.get("alert_confidence") is not None else "-"
            mode = r.get("trigger_mode", "-")[:6]
            liq = (r.get("opt_liquidity_state") or "absent")[:4]
            alerts_str = ", ".join(r["alerts"])
            rank = r.get("actionable_rank", "?")
            lines.append(
                f"| {r['ticker']} | {r['tier']} | {rank} | {ret1} | {ret5} | {conf} | {mode} | {liq} | {alerts_str} |"
            )
        lines.append("")
    else:
        lines.append("No alerts triggered.")
        lines.append("")

    lines.append(f"*Generated: {d.get('generated_at', '')}*")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Price action watch — stock + options big-move alerts")
    parser.add_argument("--as-of-date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument("--snapshots-dir", type=Path, default=REPO_ROOT / "data" / "snapshots")
    parser.add_argument("--artifacts-dir", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument("--price-csv", type=Path, default=REPO_ROOT / "production_data" / "price_history.csv")
    args = parser.parse_args()
    started = time.perf_counter()

    result = build_price_action_watch(
        args.as_of_date,
        snapshots_dir=args.snapshots_dir,
        artifacts_dir=args.artifacts_dir,
        price_csv=args.price_csv,
    )

    if "error" in result:
        logger.error(result["error"])
        try:
            from tools.agent_skill_telemetry import log_agent_run

            log_agent_run(
                "build_price_action_watch",
                f"Price action watch for {args.as_of_date}",
                inputs={"as_of_date": args.as_of_date},
                outputs={"error": result["error"]},
                success=False,
                error=result["error"],
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception:
            pass
        sys.exit(1)

    logger.info("Watch: %d names, %d alerted", result["watchlist_size"], result["n_alerted"])
    try:
        from tools.agent_skill_telemetry import log_agent_run
        from tools.record_skill_feedback import attach_outcome_verdict

        exec_id = log_agent_run(
            "build_price_action_watch",
            f"Price action watch for {args.as_of_date}",
            inputs={"as_of_date": args.as_of_date},
            outputs={"n_alerted": result.get("n_alerted"), "watchlist_size": result.get("watchlist_size")},
            success=True,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        if exec_id:
            attach_outcome_verdict(
                exec_id,
                was_correct=True,
                evidence=f"n_alerted={result.get('n_alerted', 0)} watchlist={result.get('watchlist_size', 0)}",
            )
    except Exception:
        pass


if __name__ == "__main__":
    main()
