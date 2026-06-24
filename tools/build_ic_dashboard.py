#!/usr/bin/env python3
"""Rolling IC dashboard — per-signal health tracking.

Computes trailing rolling Spearman IC for each sort signal against
forward returns. Persists a JSONL time series for trend analysis.

Signals tracked:
  - score_rank_pct (composite)
  - inst_delta_z (institutional delta)

Removed (Spec 111 2026-06-12):
  - clinical_optionality_pct_dev (deprecated: negative IC, poor hit rate)

Read-only — does not affect rankings, scoring, or execution.

Output:
    artifacts/ic_dashboard/{date}_dashboard.json
    artifacts/ic_dashboard/{date}_dashboard.md
    artifacts/ic_dashboard/history.jsonl (append)

Usage:
    python tools/build_ic_dashboard.py --as-of-date 2026-03-27
    python tools/build_ic_dashboard.py --as-of-date 2026-03-27 --lookback 20
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
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("ic_dashboard")

SCHEMA_VERSION = "ic_dashboard.v1"

# Signals to track
SIGNALS = [
    ("score_rank_pct", False),  # (field, higher_is_better)
    ("inst_delta_z", True),
]

# Removed signals (closed-lane or deprecated)
# ("clinical_optionality_pct_dev", True),
#    REMOVED: Spec 111 2026-06-12. Signal backwards: mean_ic=-0.0338, hit_rate=23.68%
# ("clinical_score_v2_z", True),
#    CLOSED: rejected as selector/ranker 2026-04

DEFAULT_LOOKBACK = 12  # snapshots
DEFAULT_HORIZON = 20  # trading days

# Thresholds
IC_HEALTHY = 0.03
IC_WARN = 0.00
IC_ALERT = -0.03


def _sf(val: Any) -> float:
    if val is None or val == "":
        return math.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return math.nan


def _is_promoted(name: str) -> bool:
    return len(name) == 10 and not name.startswith("_") and name != "state"


def _find_prior_snapshots(snapshots_dir: Path, current_date: str, n: int) -> List[str]:
    """Find the n most recent promoted snapshots before current_date."""
    if not snapshots_dir.exists():
        return []
    candidates = sorted(
        d.name for d in snapshots_dir.iterdir() if d.is_dir() and _is_promoted(d.name) and d.name < current_date
    )
    return candidates[-n:] if len(candidates) >= n else candidates


def _load_signal_values(snap_dir: Path, signal_field: str) -> Dict[str, float]:
    """Load signal values for all tickers from rankings.csv."""
    path = snap_dir / "rankings.csv"
    if not path.exists():
        return {}
    values = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ticker = row.get("ticker", "")
            val = _sf(row.get(signal_field, ""))
            if ticker and not math.isnan(val):
                values[ticker] = val
    return values


def _load_prices(price_csv: Path, tickers: set, start_date: str) -> Dict[str, Dict[str, float]]:
    """Load prices for specified tickers from start_date onward."""
    prices: Dict[str, Dict[str, float]] = {}
    if not price_csv.exists():
        return prices
    with open(price_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = row.get("ticker", "")
            d = row.get("date", "")
            if t in tickers and d >= start_date:
                c = _sf(row.get("close", ""))
                if not math.isnan(c):
                    prices.setdefault(t, {})[d] = c
    return prices


def _get_forward_return(
    ticker: str,
    snap_date: str,
    prices: Dict[str, Dict[str, float]],
    horizon: int,
) -> float:
    """Compute forward return for a ticker from snap_date over horizon days."""
    tp = prices.get(ticker, {})
    if not tp:
        return math.nan

    # Find entry price on or just after snap_date
    sorted_dates = sorted(tp.keys())
    entry_dates = [d for d in sorted_dates if d >= snap_date]
    if not entry_dates:
        return math.nan
    entry_date = entry_dates[0]
    entry_price = tp[entry_date]

    # Find exit price ~horizon trading days later
    future_dates = [d for d in sorted_dates if d > entry_date]
    if len(future_dates) < horizon:
        return math.nan
    exit_date = future_dates[min(horizon - 1, len(future_dates) - 1)]
    exit_price = tp[exit_date]

    if entry_price <= 0:
        return math.nan
    return (exit_price - entry_price) / entry_price


def compute_ic(
    signal_values: Dict[str, float],
    forward_returns: Dict[str, float],
    higher_is_better: bool,
) -> Tuple[float, int]:
    """Compute Spearman IC between signal and forward returns.

    Returns (ic, n_obs).
    """
    from scipy import stats

    common = set(signal_values.keys()) & set(forward_returns.keys())
    common = {t for t in common if not math.isnan(forward_returns[t])}

    if len(common) < 10:
        return math.nan, len(common)

    tickers = sorted(common)
    sig = [signal_values[t] for t in tickers]
    ret = [forward_returns[t] for t in tickers]

    if not higher_is_better:
        sig = [-s for s in sig]

    ic, _ = stats.spearmanr(sig, ret)
    return ic, len(common)


def build_ic_dashboard(
    as_of_date: str,
    *,
    snapshots_dir: Path = REPO_ROOT / "data" / "snapshots",
    price_csv: Path = REPO_ROOT / "production_data" / "price_history.csv",
    artifacts_dir: Path = REPO_ROOT / "artifacts",
    lookback: int = DEFAULT_LOOKBACK,
    horizon: int = DEFAULT_HORIZON,
) -> Dict[str, Any]:
    """Build rolling IC dashboard."""
    prior_dates = _find_prior_snapshots(snapshots_dir, as_of_date, lookback)

    if len(prior_dates) < 3:
        return {"error": f"need >= 3 prior snapshots, found {len(prior_dates)}"}

    # Collect all tickers across snapshots for price loading
    all_tickers: set = set()
    for d in prior_dates:
        path = snapshots_dir / d / "rankings.csv"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("ticker"):
                        all_tickers.add(row["ticker"])

    # Load prices
    prices = _load_prices(price_csv, all_tickers, prior_dates[0])

    # Compute per-signal rolling IC
    signal_results = {}
    for signal_field, higher_is_better in SIGNALS:
        per_date_ics = []
        for snap_date in prior_dates:
            sig_vals = _load_signal_values(snapshots_dir / snap_date, signal_field)
            if len(sig_vals) < 10:
                continue

            # Compute forward returns for this date
            fwd = {}
            for ticker in sig_vals:
                ret = _get_forward_return(ticker, snap_date, prices, horizon)
                if not math.isnan(ret):
                    fwd[ticker] = ret

            ic, n_obs = compute_ic(sig_vals, fwd, higher_is_better)
            if not math.isnan(ic):
                per_date_ics.append({"date": snap_date, "ic": round(ic, 4), "n_obs": n_obs})

        if per_date_ics:
            ics = [d["ic"] for d in per_date_ics]
            mean_ic = sum(ics) / len(ics)
            hit_rate = sum(1 for ic in ics if ic > 0) / len(ics)

            # Health classification
            if mean_ic >= IC_HEALTHY:
                health = "HEALTHY"
            elif mean_ic >= IC_WARN:
                health = "WEAK"
            elif mean_ic >= IC_ALERT:
                health = "WARN"
            else:
                health = "ALERT"

            signal_results[signal_field] = {
                "mean_ic": round(mean_ic, 4),
                "hit_rate": round(hit_rate, 4),
                "n_dates": len(per_date_ics),
                "health": health,
                "per_date": per_date_ics,
                "latest_ic": per_date_ics[-1]["ic"] if per_date_ics else None,
            }
        else:
            signal_results[signal_field] = {
                "mean_ic": None,
                "hit_rate": None,
                "n_dates": 0,
                "health": "NO_DATA",
                "per_date": [],
                "latest_ic": None,
            }

    # Overall attention
    healths = [r["health"] for r in signal_results.values()]
    if "ALERT" in healths:
        attention = "HIGH"
    elif "WARN" in healths:
        attention = "MEDIUM"
    else:
        attention = "LOW"

    result = {
        "schema": SCHEMA_VERSION,
        "as_of_date": as_of_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lookback": lookback,
        "horizon": horizon,
        "n_prior_dates": len(prior_dates),
        "date_range": [prior_dates[0], prior_dates[-1]] if prior_dates else [],
        "attention": attention,
        "signals": signal_results,
        "thresholds": {
            "healthy": IC_HEALTHY,
            "warn": IC_WARN,
            "alert": IC_ALERT,
        },
    }

    # Write artifacts
    out_dir = artifacts_dir / "ic_dashboard"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{as_of_date}_dashboard.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Wrote %s", json_path)

    md_path = out_dir / f"{as_of_date}_dashboard.md"
    md_path.write_text(format_dashboard_md(result), encoding="utf-8")
    logger.info("Wrote %s", md_path)

    # Append to history (with dedup guard — skip if date already present)
    history_path = out_dir / "history.jsonl"
    existing_dates: set = set()
    if history_path.exists():
        with open(history_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        existing_dates.add(json.loads(line).get("date"))
                    except json.JSONDecodeError:
                        pass

    if as_of_date not in existing_dates:
        summary = {
            "date": as_of_date,
            "attention": attention,
        }
        for sig, data in signal_results.items():
            summary[f"{sig}_ic"] = data["mean_ic"]
            summary[f"{sig}_health"] = data["health"]
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary, default=str) + "\n")
    else:
        logger.info("IC dashboard history: %s already present, skipping append", as_of_date)

    result["_json_path"] = str(json_path)
    result["_md_path"] = str(md_path)
    return result


def format_dashboard_md(d: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"# IC Dashboard — {d['as_of_date']}")
    lines.append("")
    lines.append(f"**Attention: {d['attention']}** | Lookback: {d['lookback']} dates | Horizon: {d['horizon']}d")
    if d.get("date_range"):
        lines.append(f"Window: {d['date_range'][0]} → {d['date_range'][1]}")
    lines.append("")

    lines.append("## Signal Health")
    lines.append("")
    lines.append("| Signal | Mean IC | Hit Rate | Latest IC | Health | Dates |")
    lines.append("|--------|---------|----------|-----------|--------|-------|")

    for sig_name, data in d.get("signals", {}).items():
        mean_ic = f"{data['mean_ic']:+.4f}" if data["mean_ic"] is not None else "n/a"
        hit_rate = f"{data['hit_rate']:.0%}" if data["hit_rate"] is not None else "n/a"
        latest = f"{data['latest_ic']:+.4f}" if data["latest_ic"] is not None else "n/a"
        lines.append(f"| {sig_name} | {mean_ic} | {hit_rate} | {latest} | {data['health']} | {data['n_dates']} |")
    lines.append("")

    # Sparkline per signal
    for sig_name, data in d.get("signals", {}).items():
        per_date = data.get("per_date", [])
        if per_date:
            lines.append(f"### {sig_name}")
            lines.append("")
            lines.append("```")
            for pd in per_date:
                bar_len = int(max(0, min(20, (pd["ic"] + 0.2) * 50)))
                bar = "#" * bar_len
                lines.append(f"{pd['date']}: {pd['ic']:+.4f} {bar}")
            lines.append("```")
            lines.append("")

    lines.append(f"*Generated: {d.get('generated_at', '')}*")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Rolling IC dashboard")
    parser.add_argument("--as-of-date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--snapshots-dir", type=Path, default=REPO_ROOT / "data" / "snapshots")
    parser.add_argument("--price-csv", type=Path, default=REPO_ROOT / "production_data" / "price_history.csv")
    parser.add_argument("--artifacts-dir", type=Path, default=REPO_ROOT / "artifacts")
    args = parser.parse_args()
    started = time.perf_counter()

    result = build_ic_dashboard(
        args.as_of_date,
        snapshots_dir=args.snapshots_dir,
        price_csv=args.price_csv,
        artifacts_dir=args.artifacts_dir,
        lookback=args.lookback,
        horizon=args.horizon,
    )

    if "error" in result:
        logger.error(result["error"])
        try:
            from tools.agent_skill_telemetry import log_agent_run

            log_agent_run(
                "build_ic_dashboard",
                f"IC dashboard for {args.as_of_date}",
                inputs={"as_of_date": args.as_of_date},
                outputs={"error": result["error"]},
                success=False,
                error=result["error"],
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception:
            pass
        sys.exit(1)

    logger.info("Dashboard: %s attention (%s)", result["attention"], result["as_of_date"])
    try:
        from tools.agent_skill_telemetry import log_agent_run
        from tools.record_skill_feedback import attach_outcome_verdict

        attention = result.get("attention", "UNKNOWN")
        exec_id = log_agent_run(
            "build_ic_dashboard",
            f"IC dashboard for {args.as_of_date}",
            inputs={"as_of_date": args.as_of_date},
            outputs={"attention": attention},
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
