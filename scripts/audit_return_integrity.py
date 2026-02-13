#!/usr/bin/env python3
"""Audit forward-return integrity in walkforward panels.

Detects stock-split artifacts in price_history.csv and quantifies their
impact on tier-separation analysis.  Produces both human-readable (.md)
and machine-readable (.json) reports.

CLI:
  python scripts/audit_return_integrity.py \
    [--panel artifacts/walkforward_panel__baseline_full.csv] \
    [--prices production_data/price_history.csv] \
    [--output-dir artifacts/]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JUMP_THRESHOLD = 3.0     # +300% single-day → likely reverse split
DROP_THRESHOLD = -0.75   # -75% single-day → likely forward split / delisting
TAIL_PCT = 0.005         # top/bottom 0.5% of returns for tail extraction
WINSORIZE_CAP = 2.0      # ±200% cap for winsorized treatment
SUSPICIOUS_RET = 5.0     # >500% 60d return with no flagged day → suspicious

# ---------------------------------------------------------------------------
# Phase A — Price discontinuity scan
# ---------------------------------------------------------------------------

def detect_discontinuities(
    prices: Dict[str, Dict[date, Decimal]],
    jump_threshold: float = JUMP_THRESHOLD,
    drop_threshold: float = DROP_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Scan all tickers for single-day price jumps/drops.

    Parameters
    ----------
    prices : dict
        ticker → {date → Decimal} price map (from CSVReturnsProvider._prices).
    jump_threshold : float
        Minimum single-day return to flag as reverse split (default +300%).
    drop_threshold : float
        Maximum single-day return to flag as forward split (default -75%).

    Returns
    -------
    list of dict
        Each entry: ticker, date, close_before, close_after, pct_change, flag_type.
    """
    flags: List[Dict[str, Any]] = []
    for ticker, ticker_prices in prices.items():
        sorted_dates = sorted(ticker_prices.keys())
        for i in range(1, len(sorted_dates)):
            prev = float(ticker_prices[sorted_dates[i - 1]])
            curr = float(ticker_prices[sorted_dates[i]])
            if prev <= 0:
                continue
            pct = (curr / prev) - 1.0
            if pct >= jump_threshold:
                flags.append({
                    "ticker": ticker,
                    "date": sorted_dates[i].isoformat(),
                    "close_before": round(prev, 4),
                    "close_after": round(curr, 4),
                    "pct_change": round(pct, 4),
                    "flag_type": "reverse_split",
                })
            elif pct <= drop_threshold:
                flags.append({
                    "ticker": ticker,
                    "date": sorted_dates[i].isoformat(),
                    "close_before": round(prev, 4),
                    "close_after": round(curr, 4),
                    "pct_change": round(pct, 4),
                    "flag_type": "forward_split",
                })
    return flags


# ---------------------------------------------------------------------------
# Phase B — Panel tail extraction + classification
# ---------------------------------------------------------------------------

def _flagged_ticker_dates(discontinuities: List[Dict]) -> Dict[str, List[date]]:
    """Build ticker → [flagged dates] lookup from discontinuity list."""
    out: Dict[str, List[date]] = {}
    for d in discontinuities:
        out.setdefault(d["ticker"], []).append(date.fromisoformat(d["date"]))
    return out


def classify_panel_extremes(
    panel_rows: List[Dict[str, str]],
    discontinuities: List[Dict],
    tail_pct: float = TAIL_PCT,
    suspicious_ret: float = SUSPICIOUS_RET,
) -> Dict[str, Any]:
    """Extract return tails per year and cross-reference with split flags.

    Returns dict keyed by year (str), each containing lists of extreme rows
    with classification: confirmed_split | suspicious_jump | plausible.
    """
    flagged_dates = _flagged_ticker_dates(discontinuities)
    flagged_tickers = set(flagged_dates.keys())

    # Group rows by year
    by_year: Dict[str, List[Dict]] = {}
    for row in panel_rows:
        as_of = row.get("as_of_date", "")[:4]
        if not as_of:
            continue
        by_year.setdefault(as_of, []).append(row)

    result: Dict[str, Any] = {}
    for year, rows in sorted(by_year.items()):
        year_extremes: List[Dict[str, Any]] = []
        for col in ("fwd_ret_60d", "fwd_ret_20d"):
            vals: List[Tuple[int, float]] = []
            for idx, r in enumerate(rows):
                raw = r.get(col, "")
                if raw == "" or raw is None:
                    continue
                try:
                    vals.append((idx, float(raw)))
                except (ValueError, TypeError):
                    continue
            if not vals:
                continue
            vals.sort(key=lambda x: x[1])
            n_tail = max(1, int(len(vals) * tail_pct))
            bottom = vals[:n_tail]
            top = vals[-n_tail:]
            for idx, val in bottom + top:
                row = rows[idx]
                ticker = row.get("ticker", "")
                as_of_date = row.get("as_of_date", "")
                classification = _classify_row(
                    ticker, as_of_date, val, col, flagged_dates,
                    flagged_tickers, suspicious_ret,
                )
                year_extremes.append({
                    "ticker": ticker,
                    "as_of_date": as_of_date,
                    "return_col": col,
                    "return_value": round(val, 6),
                    "tier": row.get("tier", ""),
                    "eligible": row.get("eligible", ""),
                    "classification": classification,
                })
        result[year] = year_extremes
    return result


def _classify_row(
    ticker: str,
    as_of_date: str,
    ret_val: float,
    col: str,
    flagged_dates: Dict[str, List[date]],
    flagged_tickers: set,
    suspicious_ret: float,
) -> str:
    """Classify a single extreme return row."""
    if ticker in flagged_tickers:
        # Check if any flagged date falls within plausible forward window
        try:
            as_of = date.fromisoformat(as_of_date)
        except (ValueError, TypeError):
            return "confirmed_split"  # can't parse date → conservative
        horizon = 90 if "60d" in col else 40  # generous trading-day window
        for fd in flagged_dates[ticker]:
            if as_of < fd <= _add_calendar_days(as_of, horizon):
                return "confirmed_split"
        # Ticker has a split, but not in this forward window
        if abs(ret_val) > suspicious_ret:
            return "suspicious_jump"
        return "plausible"
    if abs(ret_val) > suspicious_ret:
        return "suspicious_jump"
    return "plausible"


def _add_calendar_days(d: date, n: int) -> date:
    """Add n calendar days to date d."""
    from datetime import timedelta
    return d + timedelta(days=n)


# ---------------------------------------------------------------------------
# Phase C — Tier impact quantification
# ---------------------------------------------------------------------------

def compute_tier_impact(
    panel_rows: List[Dict[str, str]],
    flagged_tickers: set,
    winsorize_cap: float = WINSORIZE_CAP,
) -> Dict[str, Any]:
    """Compute tier separation under raw / excl_splits / winsorized treatments.

    Uses only eligible rows (eligible == "1").
    """
    # Filter to eligible rows
    eligible = [r for r in panel_rows if r.get("eligible") == "1"]

    treatments = {}
    for label, filter_fn, transform_fn in [
        ("raw", lambda t, v: True, lambda v: v),
        ("excl_splits", lambda t, v: t not in flagged_tickers, lambda v: v),
        ("winsorized_200", lambda t, v: True,
         lambda v: max(-winsorize_cap, min(winsorize_cap, v))),
    ]:
        tier_stats: Dict[str, Dict[str, Any]] = {}
        for tier in ("A", "B", "C", "D"):
            vals = []
            for r in eligible:
                if r.get("tier") != tier:
                    continue
                ticker = r.get("ticker", "")
                raw = r.get("fwd_ret_60d", "")
                if raw == "" or raw is None:
                    continue
                try:
                    v = float(raw)
                except (ValueError, TypeError):
                    continue
                if not filter_fn(ticker, v):
                    continue
                vals.append(transform_fn(v))
            tier_stats[tier] = {
                "count": len(vals),
                "mean_60d": round(_safe_mean(vals), 6) if vals else None,
                "median_60d": round(_safe_median(vals), 6) if vals else None,
            }
        ab_mean = _weighted_mean(tier_stats, ("A", "B"))
        cd_mean = _weighted_mean(tier_stats, ("C", "D"))
        spread = round(ab_mean - cd_mean, 6) if ab_mean is not None and cd_mean is not None else None
        treatments[label] = {
            "tiers": tier_stats,
            "AB_mean_60d": ab_mean,
            "CD_mean_60d": cd_mean,
            "spread": spread,
        }
    return treatments


def _safe_mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _safe_median(vals: List[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def _weighted_mean(tier_stats: Dict, tiers: Tuple[str, ...]) -> Optional[float]:
    """Count-weighted mean across tiers."""
    total_n = 0
    total_sum = 0.0
    for t in tiers:
        n = tier_stats[t]["count"]
        m = tier_stats[t]["mean_60d"]
        if n > 0 and m is not None:
            total_n += n
            total_sum += n * m
    if total_n == 0:
        return None
    return round(total_sum / total_n, 6)


# ---------------------------------------------------------------------------
# Phase D — Remediation candidates
# ---------------------------------------------------------------------------

def build_flagged_tickers_manifest(
    discontinuities: List[Dict],
) -> List[Dict[str, Any]]:
    """Build per-ticker summary of flagged discontinuities."""
    by_ticker: Dict[str, List[Dict]] = {}
    for d in discontinuities:
        by_ticker.setdefault(d["ticker"], []).append(d)
    manifest = []
    for ticker in sorted(by_ticker):
        events = by_ticker[ticker]
        manifest.append({
            "ticker": ticker,
            "n_events": len(events),
            "events": events,
        })
    return manifest


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_report_markdown(
    discontinuities: List[Dict],
    panel_extremes: Dict[str, Any],
    tier_impact: Dict[str, Any],
    flagged_manifest: List[Dict],
    scan_date: str,
) -> str:
    """Build the human-readable markdown report."""
    lines: List[str] = []
    lines.append("# Forward Return Integrity Audit")
    lines.append(f"\nScan date: {scan_date}\n")

    # Section 1 — price discontinuities
    lines.append("## 1. Price Discontinuities Detected\n")
    if not discontinuities:
        lines.append("No single-day price discontinuities found.\n")
    else:
        lines.append(f"Found **{len(discontinuities)}** discontinuities "
                      f"across **{len(set(d['ticker'] for d in discontinuities))}** tickers.\n")
        lines.append("| Ticker | Date | Close Before | Close After | Pct Change | Type |")
        lines.append("|--------|------|-------------|------------|------------|------|")
        for d in sorted(discontinuities, key=lambda x: x["pct_change"], reverse=True):
            lines.append(
                f"| {d['ticker']} | {d['date']} | {d['close_before']:.2f} "
                f"| {d['close_after']:.2f} | {d['pct_change']:+.1%} | {d['flag_type']} |"
            )
        lines.append("")

    # Section 2 — panel tail extremes
    lines.append("## 2. Panel Tail Extremes (top/bottom 0.5%)\n")
    for year, extremes in sorted(panel_extremes.items()):
        if not extremes:
            continue
        lines.append(f"### {year}\n")
        lines.append("| Ticker | As-Of | Return Col | Value | Tier | Eligible | Classification |")
        lines.append("|--------|-------|-----------|-------|------|----------|----------------|")
        for e in sorted(extremes, key=lambda x: x["return_value"], reverse=True):
            lines.append(
                f"| {e['ticker']} | {e['as_of_date']} | {e['return_col']} "
                f"| {e['return_value']:+.2%} | {e['tier']} | {e['eligible']} "
                f"| {e['classification']} |"
            )
        lines.append("")

    # Section 3 — tier separation impact
    lines.append("## 3. Tier Separation Impact\n")
    for label, data in tier_impact.items():
        lines.append(f"### Treatment: {label}\n")
        lines.append("| Tier | N | Mean 60d | Median 60d |")
        lines.append("|------|---|----------|------------|")
        for tier in ("A", "B", "C", "D"):
            ts = data["tiers"][tier]
            mean_s = f"{ts['mean_60d']:+.2%}" if ts["mean_60d"] is not None else "N/A"
            med_s = f"{ts['median_60d']:+.2%}" if ts["median_60d"] is not None else "N/A"
            lines.append(f"| {tier} | {ts['count']} | {mean_s} | {med_s} |")
        ab_s = f"{data['AB_mean_60d']:+.2%}" if data["AB_mean_60d"] is not None else "N/A"
        cd_s = f"{data['CD_mean_60d']:+.2%}" if data["CD_mean_60d"] is not None else "N/A"
        sp_s = f"{data['spread']:+.4f}" if data["spread"] is not None else "N/A"
        lines.append(f"\n**AB mean**: {ab_s}  |  **CD mean**: {cd_s}  |  **Spread (AB−CD)**: {sp_s}\n")

    # Section 4 — remediation
    lines.append("## 4. Remediation Candidates\n")
    if not flagged_manifest:
        lines.append("No tickers require price remediation.\n")
    else:
        lines.append(f"**{len(flagged_manifest)}** tickers need split-adjusted prices:\n")
        for entry in flagged_manifest:
            events_str = "; ".join(
                f"{e['date']} ({e['flag_type']}, {e['pct_change']:+.1%})"
                for e in entry["events"]
            )
            lines.append(f"- **{entry['ticker']}** — {entry['n_events']} event(s): {events_str}")
        lines.append("")

    return "\n".join(lines)


def build_json_output(
    discontinuities: List[Dict],
    flagged_tickers: List[str],
    panel_extremes: Dict[str, Any],
    tier_impact: Dict[str, Any],
    scan_date: str,
) -> Dict[str, Any]:
    """Build the machine-readable JSON output."""
    return {
        "scan_date": scan_date,
        "price_discontinuities": discontinuities,
        "flagged_tickers": sorted(flagged_tickers),
        "panel_extremes": panel_extremes,
        "tier_impact": tier_impact,
    }


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_audit(
    panel_path: Path,
    prices_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Execute full audit pipeline and write artifacts."""
    scan_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Load prices via CSVReturnsProvider
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from backtest.returns_provider import CSVReturnsProvider
    provider = CSVReturnsProvider(prices_path, price_col="close")

    # Phase A
    discontinuities = detect_discontinuities(provider._prices)
    flagged_set = set(d["ticker"] for d in discontinuities)

    # Load panel
    panel_rows: List[Dict[str, str]] = []
    if panel_path.exists():
        with open(panel_path, "r", newline="", encoding="utf-8-sig") as f:
            panel_rows = list(csv.DictReader(f))

    # Phase B
    panel_extremes = classify_panel_extremes(panel_rows, discontinuities)

    # Phase C
    tier_impact = compute_tier_impact(panel_rows, flagged_set)

    # Phase D
    flagged_manifest = build_flagged_tickers_manifest(discontinuities)

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    md_report = format_report_markdown(
        discontinuities, panel_extremes, tier_impact, flagged_manifest, scan_date,
    )
    (output_dir / "return_integrity_audit.md").write_text(md_report)

    json_output = build_json_output(
        discontinuities, sorted(flagged_set), panel_extremes, tier_impact, scan_date,
    )
    (output_dir / "return_integrity_audit.json").write_text(
        json.dumps(json_output, indent=2, default=str)
    )

    # Also write flagged_tickers.json for downstream consumption
    (output_dir / "flagged_tickers.json").write_text(
        json.dumps(flagged_manifest, indent=2, default=str)
    )

    return json_output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Audit forward-return integrity")
    parser.add_argument(
        "--panel",
        default="artifacts/walkforward_panel__baseline_full.csv",
        help="Path to walkforward panel CSV",
    )
    parser.add_argument(
        "--prices",
        default="production_data/price_history.csv",
        help="Path to price_history.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/",
        help="Directory for output artifacts",
    )
    args = parser.parse_args()

    result = run_audit(
        panel_path=Path(args.panel),
        prices_path=Path(args.prices),
        output_dir=Path(args.output_dir),
    )

    n_disc = len(result["price_discontinuities"])
    n_flagged = len(result["flagged_tickers"])
    print(f"Audit complete: {n_disc} discontinuities, {n_flagged} flagged tickers")

    # Print tier impact summary
    for label in ("raw", "excl_splits", "winsorized_200"):
        data = result["tier_impact"].get(label, {})
        spread = data.get("spread")
        spread_s = f"{spread:+.4f}" if spread is not None else "N/A"
        print(f"  {label:20s} AB−CD spread: {spread_s}")


if __name__ == "__main__":
    main()
