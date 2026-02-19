#!/usr/bin/env python3
"""Morningstar ablation: run pipeline with --morningstar-mode off vs on, compare.

Usage:
    python3 scripts/ablate_morningstar.py --as-of-date 2026-02-16
    python3 scripts/ablate_morningstar.py --as-of-date 2026-02-16 --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from decision_engine import DecisionRuleset
from scripts.compare_rulesets_replay import overlap_pct, rank_portfolio

TOP_K = 20
TOP_60 = 60

# ── helpers ────────────────────────────────────────────────────────────


def _build_screen_cmd(
    date_str: str,
    snapshot_dir: Path,
    data_dir: Path,
    ms_mode: str,
    extra: list[str],
) -> list[str]:
    """Build run_phase2_daily.py command with morningstar mode."""
    cmd = [
        sys.executable,
        str(PROJECT_DIR / "run_phase2_daily.py"),
        "--as-of-date", date_str,
        "--data-dir", str(data_dir),
        "--snapshot-dir", str(snapshot_dir),
        f"--morningstar-mode={ms_mode}",
    ]
    if extra:
        cmd.extend(extra)
    return cmd


def _load_rankings(snapshot_dir: Path, date_str: str) -> pd.DataFrame:
    path = snapshot_dir / date_str / "rankings.csv"
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(path)


# ── comparison ─────────────────────────────────────────────────────────


def compare_ablation(
    off_df: pd.DataFrame,
    on_df: pd.DataFrame,
    ruleset: DecisionRuleset,
) -> dict:
    """Compare OFF vs ON rankings using the same ruleset."""
    off_sorted = rank_portfolio(off_df, ruleset)
    on_sorted = rank_portfolio(on_df, ruleset)

    off_top20 = set(off_sorted.head(TOP_K)["ticker"])
    on_top20 = set(on_sorted.head(TOP_K)["ticker"])
    off_top60 = set(off_sorted.head(TOP_60)["ticker"])
    on_top60 = set(on_sorted.head(TOP_60)["ticker"])

    entrants_20 = sorted(on_top20 - off_top20)
    exits_20 = sorted(off_top20 - on_top20)
    entrants_60 = sorted(on_top60 - off_top60)
    exits_60 = sorted(off_top60 - on_top60)

    # Rank churn on common tickers in top-60
    common = off_top60 & on_top60
    rank_changes = []
    if common:
        off_ranks = dict(zip(off_sorted["ticker"], off_sorted["_rank"]))
        on_ranks = dict(zip(on_sorted["ticker"], on_sorted["_rank"]))
        for t in common:
            if t in off_ranks and t in on_ranks:
                rank_changes.append(abs(off_ranks[t] - on_ranks[t]))

    # Tier distribution (top-20)
    def _tier_col(df):
        if "tier_any" in df.columns and df["tier_any"].notna().any():
            return "tier_any"
        return "tier_dev"

    off_tiers = off_sorted.head(TOP_K)[_tier_col(off_sorted)].value_counts().to_dict() if not off_sorted.empty else {}
    on_tiers = on_sorted.head(TOP_K)[_tier_col(on_sorted)].value_counts().to_dict() if not on_sorted.empty else {}

    # Top rank movers (biggest |rank_change|)
    top_movers = []
    if rank_changes:
        off_ranks = dict(zip(off_sorted["ticker"], off_sorted["_rank"]))
        on_ranks = dict(zip(on_sorted["ticker"], on_sorted["_rank"]))
        for t in common:
            if t in off_ranks and t in on_ranks:
                delta = on_ranks[t] - off_ranks[t]
                top_movers.append((t, off_ranks[t], on_ranks[t], delta))
        top_movers.sort(key=lambda x: abs(x[3]), reverse=True)

    return {
        "top20_overlap": overlap_pct(off_top20, on_top20),
        "top60_overlap": overlap_pct(off_top60, on_top60),
        "entrants_20": entrants_20,
        "exits_20": exits_20,
        "entrants_60": entrants_60,
        "exits_60": exits_60,
        "mean_rank_churn_top60": round(sum(rank_changes) / len(rank_changes), 2) if rank_changes else 0.0,
        "max_rank_churn_top60": max(rank_changes) if rank_changes else 0,
        "off_tier_dist_20": off_tiers,
        "on_tier_dist_20": on_tiers,
        "off_eligible_count": len(off_sorted),
        "on_eligible_count": len(on_sorted),
        "off_total": len(off_df),
        "on_total": len(on_df),
        "top_movers": top_movers[:10],
    }


def format_report(result: dict, date_str: str, ruleset_id: str) -> str:
    """Format ablation comparison as markdown."""
    lines = [
        f"# Morningstar Ablation — {date_str}",
        "",
        f"Ruleset: `{ruleset_id}`",
        f"Comparison: `--morningstar-mode off` vs `--morningstar-mode on` (default)",
        "",
        "## Overlap",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Top-20 overlap | {result['top20_overlap']}% |",
        f"| Top-60 overlap | {result['top60_overlap']}% |",
        f"| Mean rank churn (top-60 common) | {result['mean_rank_churn_top60']} |",
        f"| Max rank churn (top-60 common) | {result['max_rank_churn_top60']} |",
        "",
        "## Top-20 Changes (ON vs OFF)",
        f"Entrants (ON only): {', '.join(result['entrants_20']) or 'none'}  ",
        f"Exits (OFF only): {', '.join(result['exits_20']) or 'none'}",
        "",
        "## Top-60 Changes (ON vs OFF)",
        f"Entrants (ON only): {', '.join(result['entrants_60']) or 'none'}  ",
        f"Exits (OFF only): {', '.join(result['exits_60']) or 'none'}",
        "",
        "## Tier Distribution (Top-20)",
        "| Tier | OFF | ON |",
        "|------|-----|-----|",
    ]
    all_tiers = sorted(set(list(result["off_tier_dist_20"]) + list(result["on_tier_dist_20"])))
    for t in all_tiers:
        lines.append(f"| {t} | {result['off_tier_dist_20'].get(t, 0)} | {result['on_tier_dist_20'].get(t, 0)} |")

    lines += [
        "",
        f"Eligible pool: OFF={result['off_eligible_count']}, ON={result['on_eligible_count']}",
        f"Total ranked: OFF={result['off_total']}, ON={result['on_total']}",
    ]

    movers = result.get("top_movers", [])
    if movers:
        lines += [
            "",
            "## Top Rank Movers (top-60 common)",
            "| Ticker | OFF rank | ON rank | Delta |",
            "|--------|----------|---------|-------|",
        ]
        for ticker, off_r, on_r, delta in movers[:10]:
            sign = "+" if delta > 0 else ""
            lines.append(f"| {ticker} | {off_r} | {on_r} | {sign}{delta} |")

    lines.append("")
    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Morningstar ablation: compare pipeline with Morningstar OFF vs ON",
    )
    p.add_argument("--as-of-date", required=True, help="Screen date (YYYY-MM-DD)")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_DIR / "production_data",
        help="Production data directory",
    )
    p.add_argument(
        "--snapshot-dir",
        type=Path,
        default=PROJECT_DIR / "data" / "snapshots",
        help="Base snapshot directory (subdirs created per mode)",
    )
    p.add_argument(
        "--ruleset",
        type=Path,
        default=None,
        help="Ruleset JSON for portfolio comparison (default: active pinned ruleset)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print commands only")
    return p.parse_known_args(argv)


def main(argv=None) -> int:
    args, extra = parse_args(argv)
    date_str = args.as_of_date

    # Subdirectories for each mode
    off_snap_dir = args.snapshot_dir / f"{date_str}__ms_off"
    on_snap_dir = args.snapshot_dir / f"{date_str}__ms_on"

    off_cmd = _build_screen_cmd(date_str, off_snap_dir, args.data_dir, "off", extra)
    on_cmd = _build_screen_cmd(date_str, on_snap_dir, args.data_dir, "on", extra)

    if args.dry_run:
        print(f"[ABLATION] OFF: {' '.join(off_cmd)}")
        print(f"[ABLATION] ON:  {' '.join(on_cmd)}")
        return 0

    # Run OFF
    print(f"[ABLATION] Running morningstar=off ...")
    rc_off = subprocess.run(off_cmd).returncode
    if rc_off not in (0, 2):
        print(f"[ABLATION] OFF run failed (exit {rc_off})", file=sys.stderr)
        return 1

    # Run ON
    print(f"[ABLATION] Running morningstar=on ...")
    rc_on = subprocess.run(on_cmd).returncode
    if rc_on not in (0, 2):
        print(f"[ABLATION] ON run failed (exit {rc_on})", file=sys.stderr)
        return 1

    # Load rankings
    off_df = _load_rankings(off_snap_dir, date_str)
    on_df = _load_rankings(on_snap_dir, date_str)

    # Load ruleset
    if args.ruleset:
        ruleset = DecisionRuleset.from_json(str(args.ruleset))
    else:
        # Use the pinned ruleset from run_screen
        from run_screen import PHASE2_DEFAULT_RULESET_PATH
        ruleset = DecisionRuleset.from_json(str(PHASE2_DEFAULT_RULESET_PATH))

    # Compare
    result = compare_ablation(off_df, on_df, ruleset)
    report = format_report(result, date_str, ruleset.ruleset_id)

    # Print
    print()
    print(report)

    # Write report
    report_path = args.snapshot_dir / f"{date_str}__morningstar_ablation.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(f"[ABLATION] Report written to {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
