#!/usr/bin/env python3
"""IC One-Pager Generator — assembles a single Markdown summary from snapshot artifacts.

Reads phase2_health.json, decision_portfolio.csv, rankings.csv,
phase2_run_delta_details.json, and catalyst_source_mix.json from a snapshot
directory and produces ``ic_onepager.md`` alongside them.

Usage:
    python scripts/make_ic_onepager.py --snapshot-dir data/snapshots/2026-02-16
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Optional[dict]:
    """Return parsed JSON or None if file missing / malformed."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _load_csv_rows(path: Path) -> Optional[List[dict]]:
    """Return list-of-dicts from CSV or None if missing."""
    try:
        with open(path, "r") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return None


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val, default: int = 0) -> int:
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default


def _fmt_pct(val, decimals: int = 1) -> str:
    return f"{_safe_float(val):.{decimals}f}%"


_NOT_AVAILABLE = "*Not available (file missing)*"


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _health_section(health_data: Optional[dict]) -> str:
    """## Health Gate"""
    if health_data is None:
        return f"## Health Gate\n{_NOT_AVAILABLE}\n"

    status = health_data.get("status", "UNKNOWN")
    metrics = health_data.get("metrics", {})

    a_count = metrics.get("a_count", "?")
    cat_cov = _fmt_pct(metrics.get("catalyst_coverage_pct", 0))
    turnover = _fmt_pct(metrics.get("name_turnover_pct", 0))
    weight_l1 = _fmt_pct(metrics.get("weight_l1_delta", 0))

    lines = [
        f"## Health Gate: {status}",
        f"A-tier: {a_count} | Catalyst coverage: {cat_cov}"
        f" | Turnover: {turnover} | Weight L1: {weight_l1}",
    ]

    reasons = health_data.get("reasons", [])
    if reasons:
        lines.append(f"Reasons: {', '.join(reasons)}")

    return "\n".join(lines) + "\n"


def _portfolio_section(
    portfolio_rows: Optional[List[dict]],
    rankings_rows: Optional[List[dict]],
) -> str:
    """## Portfolio table with explainability column."""
    if portfolio_rows is None:
        return f"## Portfolio\n{_NOT_AVAILABLE}\n"

    # Build lookup: ticker -> top_3_drivers first entry
    driver_map: Dict[str, str] = {}
    if rankings_rows:
        for row in rankings_rows:
            t3 = row.get("top_3_drivers", "")
            if t3:
                # e.g. "clinical_score:+22.0;financial_score:+17.4;..."
                first = t3.split(";")[0] if t3 else ""
                driver_map[row.get("ticker", "")] = first

    n = len(portfolio_rows)
    lines = [
        f"## Portfolio ({n} positions)",
        "| # | Ticker | Tier | Wt% | Cat Days | Strength | Mom | Risk | Top Driver |",
        "|---|--------|------|-----|----------|----------|-----|------|------------|",
    ]

    for row in portfolio_rows:
        rank = row.get("actionable_rank", "")
        ticker = row.get("ticker", "")
        tier = row.get("tier_dev", "")
        wt = _fmt_pct(row.get("target_weight_pct", 0))
        cat_days = row.get("catalyst_days", "")
        strength = row.get("catalyst_strength", row.get("catalyst_mode", ""))
        mom = row.get("mom_state", "")
        risk = row.get("risk_flags", "")
        driver = driver_map.get(ticker, "")
        lines.append(
            f"| {rank} | {ticker} | {tier} | {wt} | {cat_days} "
            f"| {strength} | {mom} | {risk} | {driver} |"
        )

    return "\n".join(lines) + "\n"


def _delta_section(delta_data: Optional[dict]) -> str:
    """## Delta vs Prior."""
    if delta_data is None:
        return f"## Delta vs Prior\n{_NOT_AVAILABLE}\n"

    prior = delta_data.get("prior", {})
    prior_date = prior.get("date", "?")
    pt = delta_data.get("portfolio_turnover", {})
    turnover = _fmt_pct(pt.get("name_turnover_pct", 0))
    weight_l1 = _fmt_pct(pt.get("weight_l1_delta", 0))
    entrants = pt.get("entrants", [])
    exits = pt.get("exits", [])

    lines = [
        f"## Delta vs Prior ({prior_date})",
        f"Turnover: {turnover} | Weight L1: {weight_l1}",
    ]

    ent_str = ", ".join(entrants) if entrants else "none"
    exit_str = ", ".join(exits) if exits else "none"
    lines.append(f"Entrants: {ent_str}  ")
    lines.append(f"Exits: {exit_str}")

    return "\n".join(lines) + "\n"


def _catalyst_section(
    delta_data: Optional[dict],
    source_mix: Optional[dict],
) -> str:
    """## Catalyst Coverage."""
    lines = ["## Catalyst Coverage"]

    has_content = False

    # Coverage from delta
    if delta_data:
        cat_cov = delta_data.get("catalyst_coverage", {}).get("current", {})
        n_dev = cat_cov.get("n_dev", 0)
        n_specific = cat_cov.get("n_specific", 0)
        pct = _fmt_pct(cat_cov.get("pct", 0))
        lines.append(f"Dev specific_days: {n_specific}/{n_dev} ({pct})")

        # Nearest catalysts
        top_cats = delta_data.get("top_catalysts", {}).get("current", [])
        if top_cats:
            nearest = top_cats[:5]
            parts = [
                f"{c['ticker']} ({c['catalyst_days']}d, {c.get('tier_dev', '?')})"
                for c in nearest
            ]
            lines.append(f"Nearest 5: {', '.join(parts)}")
        has_content = True

    # Source mix
    if source_mix:
        by_src = source_mix.get("by_source", {})
        total = source_mix.get("total_events", 0)
        src_parts = [f"{k}={v}" for k, v in sorted(by_src.items())]
        lines.append(f"Sources ({total} total): {', '.join(src_parts)}")
        has_content = True

    if not has_content:
        lines.append(_NOT_AVAILABLE)

    return "\n".join(lines) + "\n"


def _tier_section(
    rankings_rows: Optional[List[dict]],
    portfolio_rows: Optional[List[dict]],
) -> str:
    """## Tier Distribution table (universe + portfolio)."""
    if rankings_rows is None:
        return f"## Tier Distribution\n{_NOT_AVAILABLE}\n"

    # Count tiers in dev universe
    tier_order = ["A", "B", "C", "D"]
    uni_counts: Dict[str, int] = {t: 0 for t in tier_order}
    for row in rankings_rows:
        td = row.get("tier_dev", "")
        if td in uni_counts:
            uni_counts[td] += 1

    port_counts: Dict[str, int] = {t: 0 for t in tier_order}
    if portfolio_rows:
        for row in portfolio_rows:
            td = row.get("tier_dev", "")
            if td in port_counts:
                port_counts[td] += 1

    lines = [
        "## Tier Distribution",
        "| Tier | Universe | Portfolio |",
        "|------|----------|-----------|",
    ]
    for t in tier_order:
        lines.append(f"| {t} | {uni_counts[t]} | {port_counts[t]} |")

    return "\n".join(lines) + "\n"


def _exceptions_section(
    rankings_rows: Optional[List[dict]],
    portfolio_rows: Optional[List[dict]],
    health_data: Optional[dict],
) -> str:
    """## Exceptions — ineligible tickers, risk flags, health warnings."""
    lines = ["## Exceptions"]

    # Ineligible count + reasons
    if rankings_rows:
        ineligible = [
            r for r in rankings_rows if r.get("eligible", "1") == "0"
        ]
        if ineligible:
            reason_counts: Dict[str, int] = {}
            for r in ineligible:
                reason = r.get("ineligible_reasons", "unknown") or "unknown"
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            top_reasons = sorted(
                reason_counts.items(), key=lambda x: -x[1]
            )[:3]
            reason_str = ", ".join(f"{k} ({v})" for k, v in top_reasons)
            lines.append(f"Ineligible: {len(ineligible)} ({reason_str})")
        else:
            lines.append("Ineligible: 0")
    else:
        lines.append(f"Ineligible: {_NOT_AVAILABLE}")

    # Risk flags in portfolio
    if portfolio_rows:
        risk_count = 0
        risk_details: List[str] = []
        for r in portfolio_rows:
            rf = r.get("risk_flags", "")
            if rf and rf.strip():
                risk_count += 1
                risk_details.append(f"{r.get('ticker', '?')}: {rf}")
        if risk_count:
            lines.append(
                f"Risk flags in portfolio: {risk_count}"
                f" ({'; '.join(risk_details)})"
            )
        else:
            lines.append("Risk flags in portfolio: 0")
    else:
        lines.append("Risk flags in portfolio: N/A")

    # Health warnings
    if health_data:
        reasons = health_data.get("reasons", [])
        if reasons:
            lines.append(f"Health warnings: {', '.join(reasons)}")
        else:
            lines.append("Health warnings: none")
    else:
        lines.append("Health warnings: N/A")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main assembler
# ---------------------------------------------------------------------------

def generate_ic_onepager(snap_dir: Path) -> str:
    """Return IC one-pager markdown from a snapshot directory.

    Never raises — partial output is better than no output.
    """
    date_str = snap_dir.name

    health_data = _load_json(snap_dir / "phase2_health.json")
    delta_data = _load_json(snap_dir / "phase2_run_delta_details.json")
    source_mix = _load_json(snap_dir / "catalyst_source_mix.json")
    portfolio_rows = _load_csv_rows(snap_dir / "decision_portfolio.csv")
    rankings_rows = _load_csv_rows(snap_dir / "rankings.csv")

    sections = [
        f"# IC Run Summary — {date_str}",
        "",
        _health_section(health_data),
        _portfolio_section(portfolio_rows, rankings_rows),
        _delta_section(delta_data),
        _catalyst_section(delta_data, source_mix),
        _tier_section(rankings_rows, portfolio_rows),
        _exceptions_section(rankings_rows, portfolio_rows, health_data),
    ]

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate IC one-pager from a snapshot directory."
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        required=True,
        help="Path to snapshot directory (e.g. data/snapshots/2026-02-16)",
    )
    args = parser.parse_args()

    snap_dir = args.snapshot_dir.resolve()
    if not snap_dir.is_dir():
        print(f"ERROR: {snap_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    md = generate_ic_onepager(snap_dir)
    out_path = snap_dir / "ic_onepager.md"
    out_path.write_text(md, encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
