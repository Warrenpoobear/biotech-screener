#!/usr/bin/env python3
"""
rank_depth_validation_summary.py — DEM rank-depth shadow validation card.

VALIDATION_INFRASTRUCTURE / RANK_DEPTH_SHADOW_TRACKING / NO_MODEL_CHANGE.

Reads the forward-validation ledgers (captures.jsonl + fills.jsonl) and reports
equal-weight forward performance for three rank-depth cohorts versus XBI:

    top30     ranks 1-30   — current selected model basket (primary)
    rank31_60 ranks 31-60  — shadow reserve bench (monitored only)
    top60     ranks 1-60   — depth-of-rank validation cohort

Writes artifacts/validation/rank_depth/RANK_DEPTH_VALIDATION.md.

This is annotation/measurement only. No ranker, selector, scoring, eligibility,
sizing, or trading behavior is changed. Top-30 remains the primary basket;
ranks 31-60 and Top-60 are shadow candidates and are never tradable by default.

Usage:
    python3 tools/rank_depth_validation_summary.py
    python3 tools/rank_depth_validation_summary.py --since 2026-06-01
"""

from __future__ import annotations

import argparse
import json
from datetime import date as ddate
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO_ROOT / "artifacts" / "forward_validation"
CAPTURES_LEDGER = ARTIFACTS / "captures.jsonl"
FILLS_LEDGER = ARTIFACTS / "fills.jsonl"
CANDIDATE_FILE = ARTIFACTS / "CANDIDATE.json"
OUT_DIR = REPO_ROOT / "artifacts" / "validation" / "rank_depth"
OUT_PATH = OUT_DIR / "RANK_DEPTH_VALIDATION.md"

COHORTS = ["top30", "rank31_60", "top60"]
COHORT_LABELS = {"top30": "Top-30", "rank31_60": "Ranks 31–60", "top60": "Top-60"}


def week_key(date_str: str) -> str:
    d = ddate.fromisoformat(date_str)
    year, week, _ = d.isocalendar()
    return f"{year}-W{week:02d}"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def cohort_horizon(fill: dict, cohort: str, horizon: str) -> dict:
    """Return the per-cohort per-horizon record (ew/xbi/xs/n_names) or {}."""
    return (fill.get("cohorts") or {}).get(cohort, {}).get(horizon, {}) or {}


def latest_completed_5d(captures: list[dict], fills: dict[str, dict], cohort: str) -> dict | None:
    """Most recent capture with a completed 5d window for this cohort."""
    for cap in sorted(captures, key=lambda c: c["date"], reverse=True):
        rec = cohort_horizon(fills.get(cap["date"], {}), cohort, "5d")
        if rec.get("xs_return") is not None:
            return {"date": cap["date"], **rec}
    return None


def weekly_stats(captures: list[dict], fills: dict[str, dict], cohort: str, since: str | None) -> dict:
    """Non-overlapping weekly 5d windows (earliest completed capture per ISO week)."""
    by_week: dict[str, dict] = {}
    for cap in sorted(captures, key=lambda c: c["date"]):
        date = cap["date"]
        if since and date < since:
            continue
        rec = cohort_horizon(fills.get(date, {}), cohort, "5d")
        if rec.get("xs_return") is None:
            continue
        wk = week_key(date)
        if wk not in by_week:
            by_week[wk] = {
                "date": date,
                "ew": rec.get("ew_return"),
                "xbi": rec.get("xbi_return"),
                "xs": rec.get("xs_return"),
            }

    rows = list(by_week.values())
    n = len(rows)
    if n == 0:
        return {"n": 0}

    xs_vals = [r["xs"] for r in rows]
    ew_vals = [r["ew"] for r in rows if r["ew"] is not None]
    xbi_vals = [r["xbi"] for r in rows if r["xbi"] is not None]
    mean_xs = sum(xs_vals) / n
    variance = sum((x - mean_xs) ** 2 for x in xs_vals) / max(n - 1, 1)
    std_xs = variance**0.5
    t_stat = mean_xs / (std_xs / n**0.5) if std_xs > 0 else 0.0
    hit_rate = sum(1 for x in xs_vals if x > 0) / n
    return {
        "n": n,
        "cum_ew": sum(ew_vals),
        "cum_xbi": sum(xbi_vals),
        "cum_xs": sum(xs_vals),
        "mean_xs": mean_xs,
        "t_stat": t_stat,
        "hit_rate": hit_rate,
    }


def _pct(v) -> str:
    return f"{v:+.2%}" if isinstance(v, (int, float)) else "PENDING"


def build_card(captures: list[dict], fills: dict[str, dict], candidate: dict, since: str | None) -> str:
    as_of = max((c["date"] for c in captures), default="—")
    lines = [
        "# DEM Rank-Depth Validation",
        "",
        f"**As of:** {as_of}  ",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%MZ')}  ",
        f"**Candidate model hash:** `{candidate.get('model_hash', 'unregistered')}`  ",
        f"**Ruleset hash:** `{candidate.get('ruleset_hash', 'unknown')}`  ",
        "**Classification:** `VALIDATION_INFRASTRUCTURE / RANK_DEPTH_SHADOW_TRACKING / NO_MODEL_CHANGE`",
        "",
        "## Latest completed 5d window",
        "",
        "| Cohort | N | EW return | XBI | XS |",
        "|---|---:|---:|---:|---:|",
    ]
    for cohort in COHORTS:
        rec = latest_completed_5d(captures, fills, cohort)
        if rec:
            lines.append(
                f"| {COHORT_LABELS[cohort]} | {rec.get('n_names', '—')} | "
                f"{_pct(rec.get('ew_return'))} | {_pct(rec.get('xbi_return'))} | {_pct(rec.get('xs_return'))} |"
            )
        else:
            lines.append(f"| {COHORT_LABELS[cohort]} | — | PENDING | PENDING | PENDING |")

    lines += [
        "",
        "## Cumulative forward validation (non-overlapping weekly 5d windows)",
        "",
        "| Cohort | Windows | Cum EW | Cum XBI | Cum XS | Mean weekly XS | t-stat | Hit rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cohort in COHORTS:
        s = weekly_stats(captures, fills, cohort, since)
        if s["n"] == 0:
            lines.append(f"| {COHORT_LABELS[cohort]} | 0 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |")
            continue
        lines.append(
            f"| {COHORT_LABELS[cohort]} | {s['n']} | {_pct(s['cum_ew'])} | {_pct(s['cum_xbi'])} | "
            f"{_pct(s['cum_xs'])} | {_pct(s['mean_xs'])} | {s['t_stat']:.2f} | {s['hit_rate']:.0%} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- **Top-30** remains the primary validation basket / selected model.",
        "- **Ranks 31–60** are shadow candidates only (reserve bench) — never tradable by default.",
        "- **Top-60** tests rank-depth robustness: does alpha persist beyond rank 30?",
        "- A real ranker slope looks like `Top-30 XS > Top-60 XS > Ranks 31–60 XS > 0`.",
        "- No selector, ranker, sizing, or trading behavior changed to produce this card.",
        "",
        "_Data source: `artifacts/forward_validation/captures.jsonl` + `fills.jsonl` "
        "(equal-weight, close, split-adjusted, XBI benchmark)._",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="DEM rank-depth shadow validation card")
    parser.add_argument("--since", help="Only include windows from YYYY-MM-DD onward")
    args = parser.parse_args()

    captures = load_jsonl(CAPTURES_LEDGER)
    if not captures:
        print("No captures found. Run run_forward_validation.py first.")
        return 1

    fills_list = load_jsonl(FILLS_LEDGER)
    fills: dict[str, dict] = {}
    for f in fills_list:
        fills[f["capture_date"]] = f  # last entry wins

    candidate = {}
    if CANDIDATE_FILE.exists():
        candidate = json.loads(CANDIDATE_FILE.read_text())

    card = build_card(captures, fills, candidate, args.since)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(card, encoding="utf-8")

    print(card)
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
