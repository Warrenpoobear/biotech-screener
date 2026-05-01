#!/usr/bin/env python3
"""Form 4 signed-delta reconciliation — Spec 065 §1 #5.

For each ticker present in two consecutive production snapshots, reconcile
the change in `insider_net_buy_value_90d` between snapshots against the
same change reconstructed from `data/form4/form4_panel.csv`.

Tolerance per Spec 065 §2:
  ≤ $1,000 absolute OR ≤ 0.5% relative, whichever is larger, per ticker.

A ticker is "reconciled" iff the snapshot-level delta and the panel-level
delta agree within tolerance. Any single failure marks the snapshot pair
as not reconciled (Spec 065 "no partial credit" applies at the gate level,
but per-ticker failures are reported here for diagnosis).

Diagnostic only. Not invoked by production.

Usage:
    python -m tools.check_form4_reconciliation
    python -m tools.check_form4_reconciliation --from 2026-04-30 --to 2026-05-01
    python -m tools.check_form4_reconciliation --window 5  # last 5 snapshot pairs
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SNAP_ROOT = PROJECT_ROOT / "data" / "snapshots"
PANEL_CSV = PROJECT_ROOT / "data" / "form4" / "form4_panel.csv"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

ABS_TOL = 1_000.0  # dollars
REL_TOL = 0.005  # 0.5%

INSIDER_FIELD = "insider_net_buy_value_90d"


def _safe_float(v) -> Optional[float]:
    if v in (None, "", "nan", "None", "null"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _load_snapshot_values(snap_date: str) -> Dict[str, float]:
    """Return {ticker: insider_net_buy_value_90d} for one snapshot. Skips
    rows where the field is blank ("" = no raw file, criterion #4 NA case).
    """
    p = SNAP_ROOT / snap_date / "rankings.csv"
    if not p.exists():
        return {}
    out: Dict[str, float] = {}
    with p.open(newline="") as f:
        for row in csv.DictReader(f):
            t = row.get("ticker")
            v = _safe_float(row.get(INSIDER_FIELD))
            if t and v is not None:
                out[t] = v
    return out


def _load_panel_values(panel_path: Path = PANEL_CSV) -> Dict[Tuple[str, str], float]:
    """Return {(ticker, as_of_date): insider_net_buy_value_90d} from the panel.
    Panel is event-keyed but each row carries an as_of_date — we use that.
    """
    if not panel_path.exists():
        return {}
    out: Dict[Tuple[str, str], float] = {}
    with panel_path.open(newline="") as f:
        for row in csv.DictReader(f):
            t = row.get("ticker")
            d = row.get("as_of_date")
            v = _safe_float(row.get(INSIDER_FIELD))
            if t and d and v is not None:
                out[(t, d)] = v
    return out


def _within_tol(delta_a: float, delta_b: float) -> bool:
    """Spec 065 §2 tolerance: |Δa - Δb| ≤ max($1000, 0.5% × max(|Δa|, |Δb|))."""
    diff = abs(delta_a - delta_b)
    rel_bound = REL_TOL * max(abs(delta_a), abs(delta_b))
    bound = max(ABS_TOL, rel_bound)
    return diff <= bound


def reconcile_pair(
    snap_from: str,
    snap_to: str,
    panel_lookup: Dict[Tuple[str, str], float],
) -> dict:
    """Reconcile insider_net_buy_value_90d delta between two snapshots
    against the panel's reconstruction. Returns a per-ticker report."""
    sv_from = _load_snapshot_values(snap_from)
    sv_to = _load_snapshot_values(snap_to)
    common = sorted(set(sv_from.keys()) & set(sv_to.keys()))

    matched = 0
    mismatched: List[dict] = []
    no_panel_data: List[str] = []

    for ticker in common:
        delta_snap = sv_to[ticker] - sv_from[ticker]
        pv_from = panel_lookup.get((ticker, snap_from))
        pv_to = panel_lookup.get((ticker, snap_to))
        if pv_from is None or pv_to is None:
            no_panel_data.append(ticker)
            continue
        delta_panel = pv_to - pv_from
        if _within_tol(delta_snap, delta_panel):
            matched += 1
        else:
            mismatched.append(
                {
                    "ticker": ticker,
                    "delta_snapshot": round(delta_snap, 2),
                    "delta_panel": round(delta_panel, 2),
                    "abs_diff": round(abs(delta_snap - delta_panel), 2),
                }
            )

    return {
        "snap_from": snap_from,
        "snap_to": snap_to,
        "n_common_tickers": len(common),
        "n_matched": matched,
        "n_mismatched": len(mismatched),
        "n_no_panel_data": len(no_panel_data),
        "mismatched": sorted(mismatched, key=lambda m: -m["abs_diff"])[:20],
        "no_panel_data_sample": sorted(no_panel_data)[:5],
    }


def _list_snapshots(since: Optional[str] = None) -> List[str]:
    out = []
    for d in sorted(SNAP_ROOT.iterdir()):
        if not d.is_dir() or not DATE_RE.match(d.name):
            continue
        if since and d.name < since:
            continue
        if (d / "rankings.csv").exists():
            out.append(d.name)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--from", dest="snap_from", help="Earlier snapshot date YYYY-MM-DD")
    parser.add_argument("--to", dest="snap_to", help="Later snapshot date YYYY-MM-DD")
    parser.add_argument(
        "--window", type=int, default=5, help="Number of most-recent consecutive pairs to reconcile (default 5)"
    )
    parser.add_argument("--panel", type=Path, default=PANEL_CSV)
    parser.add_argument("--json", action="store_true", help="Output JSON instead of human-readable")
    args = parser.parse_args(argv)

    log.info(f"Loading panel: {args.panel}")
    panel_lookup = _load_panel_values(args.panel)
    if not panel_lookup:
        log.error(f"Panel empty or missing: {args.panel}")
        return 2
    log.info(f"Panel rows: {len(panel_lookup)}")

    pairs: List[Tuple[str, str]] = []
    if args.snap_from and args.snap_to:
        pairs = [(args.snap_from, args.snap_to)]
    else:
        snaps = _list_snapshots()
        # Take last (window+1) snapshots and form (window) consecutive pairs
        recent = snaps[-(args.window + 1) :]
        pairs = list(zip(recent[:-1], recent[1:]))

    if not pairs:
        log.error("No snapshot pairs to reconcile.")
        return 2

    reports = []
    for s_from, s_to in pairs:
        log.info(f"Reconciling {s_from} → {s_to}")
        reports.append(reconcile_pair(s_from, s_to, panel_lookup))

    if args.json:
        print(json.dumps(reports, indent=2))
    else:
        print(
            f"\n{'snap_from':<12} {'snap_to':<12} {'common':>7} {'matched':>8} "
            f"{'mismatched':>11} {'no_panel':>9} {'verdict':<10}"
        )
        print("-" * 80)
        any_fail = False
        for r in reports:
            verdict = "PASS" if r["n_mismatched"] == 0 else "FAIL"
            if r["n_mismatched"] > 0:
                any_fail = True
            print(
                f"{r['snap_from']:<12} {r['snap_to']:<12} {r['n_common_tickers']:>7} "
                f"{r['n_matched']:>8} {r['n_mismatched']:>11} {r['n_no_panel_data']:>9} "
                f"{verdict:<10}"
            )
            if r["mismatched"]:
                print("  Top mismatches (Δsnap, Δpanel, |diff|):")
                for m in r["mismatched"][:5]:
                    print(
                        f"    {m['ticker']:<6}  snap={m['delta_snapshot']:>+12.2f}  "
                        f"panel={m['delta_panel']:>+12.2f}  diff={m['abs_diff']:>10.2f}"
                    )
            if r["no_panel_data_sample"]:
                print(f"  No-panel-data sample: {r['no_panel_data_sample']}")

        return 1 if any_fail else 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
