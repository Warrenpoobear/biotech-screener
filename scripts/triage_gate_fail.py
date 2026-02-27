#!/usr/bin/env python3
"""Triage a ruleset promotion gate FAIL/WARN.

Reads the eval JSON produced by eval_ruleset.py, identifies the worst date
and top movers, and produces a triage bundle with per-ticker explain files.

Usage:
    python3 scripts/triage_gate_fail.py \
        --eval-json /tmp/ruleset_gate_out/.../ruleset_eval.json \
        --snapshot-dir data/snapshots \
        --out-dir /tmp/ruleset_gate_out/.../triage \
        --ruleset production_data/decision_rulesets/v1.6.1_alpha_modifier_within_tier.json \
        --max-tickers 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from decision_engine import DecisionRuleset
from scripts.eval_ruleset import (
    _read_rankings,
    rank_map as _existing_rank_map,
    rerank_rows,
)
from scripts.explain_rank_shift import explain_ticker_shift

VERSION = "1.0.0"
SCHEMA = "triage.v1"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=2, default=str) + "\n"


# ---------------------------------------------------------------------------
# Worst-date selection
# ---------------------------------------------------------------------------


def select_worst_date(
    per_date: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Select the worst date by max_rank_shift, then lowest overlap, then latest.

    Skips entries without temporal metrics (typically the first evaluated date).
    """
    if not per_date:
        return None

    # Filter to dates that have at least one temporal metric
    candidates = [
        pd for pd in per_date
        if pd.get("temporal_max_shift") is not None
        or pd.get("temporal_overlap_60") is not None
    ]
    if not candidates:
        return per_date[-1]

    def _severity(pd: Dict[str, Any]):
        shift = pd.get("temporal_max_shift") or 0
        ov60 = pd.get("temporal_overlap_60")
        inv_overlap = (100.0 - ov60) if ov60 is not None else 0.0
        date = pd.get("date", "")
        return (shift, inv_overlap, date)

    candidates.sort(key=_severity, reverse=True)
    return candidates[0]


def find_prior_date(
    worst_date: str, dates_evaluated: List[str]
) -> Optional[str]:
    """Find the date immediately before *worst_date* in the evaluated list."""
    for i, d in enumerate(dates_evaluated):
        if d == worst_date and i > 0:
            return dates_evaluated[i - 1]
    return None


# ---------------------------------------------------------------------------
# Movers
# ---------------------------------------------------------------------------


def collect_top_movers(
    eval_data: Dict[str, Any],
    worst_date: str,
    max_tickers: int,
) -> List[Dict[str, Any]]:
    """Get top movers for *worst_date* from the eval's enriched top_movers."""
    movers = eval_data.get("temporal_stability", {}).get("top_movers", [])
    date_movers = [m for m in movers if m.get("date") == worst_date]
    date_movers.sort(key=lambda x: (-x.get("shift", 0), x.get("ticker", "")))
    return date_movers[:max_tickers]


def reconstruct_movers_from_snapshots(
    worst_date: str,
    prior_date: str,
    snapshot_dir: Path,
    max_tickers: int,
    ruleset: Optional[DecisionRuleset] = None,
) -> List[Dict[str, Any]]:
    """Reconstruct top movers by loading two snapshots and computing shifts.

    Used as a fallback when the eval JSON lacks enriched top_movers.
    If *ruleset* is provided, re-ranks both snapshots; otherwise uses
    existing actionable_rank columns.
    """
    cur_path = snapshot_dir / worst_date / "rankings.csv"
    pri_path = snapshot_dir / prior_date / "rankings.csv"
    if not cur_path.exists() or not pri_path.exists():
        return []

    cur_rows = _read_rankings(cur_path)
    pri_rows = _read_rankings(pri_path)
    if not cur_rows or not pri_rows:
        return []

    if ruleset:
        _, cur_ranks = rerank_rows(cur_rows, ruleset)
        _, pri_ranks = rerank_rows(pri_rows, ruleset)
    else:
        cur_ranks = _existing_rank_map(cur_rows)
        pri_ranks = _existing_rank_map(pri_rows)

    common = set(cur_ranks) & set(pri_ranks)
    if not common:
        return []

    entries = []
    for t in common:
        s = abs(cur_ranks[t] - pri_ranks[t])
        if s > 0:
            entries.append({
                "ticker": t,
                "shift": s,
                "from": pri_ranks[t],
                "to": cur_ranks[t],
                "date": worst_date,
            })
    entries.sort(key=lambda x: (-x["shift"], x["ticker"]))
    return entries[:max_tickers]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def triage_gate_fail(
    eval_json_path: Path,
    snapshot_dir: Path,
    out_dir: Path,
    max_tickers: int = 10,
    ruleset: Optional[DecisionRuleset] = None,
) -> Dict[str, Any]:
    """Produce a triage bundle from an eval JSON.

    Returns the triage summary dict and writes:
      - triage_summary.json
      - triage_summary.md
      - explain/<DATE>_<TICKER>.json  (one per top mover, if ruleset given)
    """
    eval_data = _read_json(eval_json_path)

    per_date = eval_data.get("per_date", [])
    dates_evaluated = eval_data.get("dates_evaluated", [])
    gate = eval_data.get("gate", {})
    config = eval_data.get("config", {})
    cand_info = eval_data.get("candidate", {})
    base_info = eval_data.get("baseline", {})

    # Worst date
    worst = select_worst_date(per_date)
    worst_date = worst["date"] if worst else None
    prior_date = (
        find_prior_date(worst_date, dates_evaluated) if worst_date else None
    )

    # Severity metric
    worst_metric_name = "temporal_max_shift"
    worst_metric_value = worst.get("temporal_max_shift") if worst else None
    if worst and worst_metric_value is None:
        worst_metric_name = "temporal_overlap_60"
        worst_metric_value = worst.get("temporal_overlap_60")

    notes: List[str] = []

    # Top movers for worst date
    movers = (
        collect_top_movers(eval_data, worst_date, max_tickers)
        if worst_date
        else []
    )

    # Fallback 1: use per-date shift info if enriched movers are absent
    if not movers and worst:
        t = worst.get("temporal_max_shift_ticker")
        if t:
            movers = [{
                "ticker": t,
                "shift": worst.get("temporal_max_shift", 0),
                "from": worst.get("temporal_max_shift_from"),
                "to": worst.get("temporal_max_shift_to"),
                "date": worst_date,
            }]

    # Fallback 2: reconstruct from snapshots when eval JSON lacks movers
    if not movers and worst_date and prior_date:
        movers = reconstruct_movers_from_snapshots(
            worst_date=worst_date,
            prior_date=prior_date,
            snapshot_dir=snapshot_dir,
            max_tickers=max_tickers,
            ruleset=ruleset,
        )
        if movers:
            notes.append(
                f"movers reconstructed from snapshots "
                f"({worst_date} vs {prior_date})"
            )

    # Fallback 3: aggregate max_rank_shift from temporal_stability
    if not movers:
        agg_ms = eval_data.get("temporal_stability", {}).get(
            "max_rank_shift", {}
        )
        if agg_ms.get("ticker"):
            movers = [{
                "ticker": agg_ms["ticker"],
                "shift": agg_ms.get("shift", 0),
                "from": agg_ms.get("from"),
                "to": agg_ms.get("to"),
                "date": agg_ms.get("date", worst_date),
            }]
            notes.append("movers from aggregate max_rank_shift")

    # Build per-ticker explains
    explain_dir = out_dir / "explain"
    explain_dir.mkdir(parents=True, exist_ok=True)

    explain_files: List[str] = []

    if ruleset and worst_date and prior_date:
        for mover in movers:
            ticker = mover.get("ticker", "")
            if not ticker:
                continue
            try:
                explain = explain_ticker_shift(
                    ticker=ticker,
                    current_date=worst_date,
                    prior_date=prior_date,
                    snapshot_dir=snapshot_dir,
                    ruleset=ruleset,
                )
                fname = f"explain_{worst_date}_{ticker}.json"
                (explain_dir / fname).write_text(
                    _stable_json(explain), encoding="utf-8"
                )
                explain_files.append(f"explain/{fname}")
            except Exception as exc:
                notes.append(f"explain failed for {ticker}: {exc}")
    elif not ruleset:
        notes.append("no ruleset provided — skipped per-ticker explains")
    elif not prior_date:
        notes.append(
            f"no prior date for {worst_date} — skipped per-ticker explains"
        )

    # Movers summary (stable subset of fields)
    movers_summary = []
    for m in movers:
        fr = m.get("from")
        to = m.get("to")
        movers_summary.append({
            "ticker": m.get("ticker", ""),
            "from_rank": fr,
            "to_rank": to,
            "delta_rank": (to - fr) if to is not None and fr is not None else None,
            "shift": m.get("shift", 0),
        })

    summary: Dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "candidate_id": cand_info.get("id", ""),
        "baseline_id": base_info.get("id", ""),
        "eval_mode": eval_data.get(
            "mode", config.get("eval_mode", "unknown")
        ),
        "verdict": gate.get("verdict", "UNKNOWN"),
        "date_range": {
            "start": config.get("start", ""),
            "end": config.get("end", ""),
        },
        "n_evaluated": eval_data.get("n_evaluated", 0),
        "worst_date": worst_date,
        "worst_metric_name": worst_metric_name,
        "worst_metric_value": worst_metric_value,
        "top_movers": movers_summary,
        "explain_files": sorted(explain_files),
        "notes": notes,
    }

    # Write outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "triage_summary.json").write_text(
        _stable_json(summary), encoding="utf-8"
    )
    (out_dir / "triage_summary.md").write_text(
        _build_markdown(summary), encoding="utf-8"
    )
    return summary


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _build_markdown(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Gate Fail Triage")
    lines.append("")
    lines.append(f"- **Verdict:** {summary['verdict']}")
    lines.append(f"- **Candidate:** `{summary['candidate_id']}`")
    lines.append(f"- **Baseline:** `{summary['baseline_id']}`")
    lines.append(f"- **Mode:** {summary['eval_mode']}")
    lines.append(f"- **Dates evaluated:** {summary['n_evaluated']}")
    lines.append("")

    lines.append("## Worst Date")
    lines.append("")
    wd = summary["worst_date"] or "none"
    lines.append(f"- **Date:** `{wd}`")
    lines.append(
        f"- **{summary['worst_metric_name']}:** "
        f"{summary['worst_metric_value']}"
    )
    lines.append("")

    movers = summary.get("top_movers", [])
    if movers:
        lines.append("## Top Movers")
        lines.append("")
        lines.append(
            "| Ticker | Prior Rank | Current Rank | Delta | Shift |"
        )
        lines.append("|--------|-----------|-------------|-------|-------|")
        for m in movers:
            lines.append(
                f"| {m['ticker']} "
                f"| {m.get('from_rank', '?')} "
                f"| {m.get('to_rank', '?')} "
                f"| {m.get('delta_rank', '?')} "
                f"| {m.get('shift', '?')} |"
            )
        lines.append("")

    explains = summary.get("explain_files", [])
    if explains:
        lines.append("## Explain Files")
        lines.append("")
        for f in explains:
            lines.append(f"- `{f}`")
        lines.append("")

    notes = summary.get("notes", [])
    if notes:
        lines.append("## Notes")
        lines.append("")
        for n in notes:
            lines.append(f"- {n}")
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Triage a ruleset gate FAIL/WARN"
    )
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument(
        "--snapshot-dir", type=Path, default=Path("data/snapshots")
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-tickers", type=int, default=10)
    parser.add_argument(
        "--ruleset",
        type=Path,
        help="Candidate ruleset JSON for per-ticker explains",
    )
    args = parser.parse_args(argv)

    if not args.eval_json.exists():
        print(
            f"Error: eval JSON not found: {args.eval_json}", file=sys.stderr
        )
        return 1

    ruleset = None
    if args.ruleset and args.ruleset.exists():
        ruleset = DecisionRuleset.from_json(str(args.ruleset))

    summary = triage_gate_fail(
        eval_json_path=args.eval_json,
        snapshot_dir=args.snapshot_dir,
        out_dir=args.out_dir,
        max_tickers=args.max_tickers,
        ruleset=ruleset,
    )

    print(
        f"Triage complete: worst_date={summary['worst_date']}, "
        f"movers={len(summary['top_movers'])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
