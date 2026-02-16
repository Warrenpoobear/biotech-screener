#!/usr/bin/env python3
"""
Ad-hoc diagnostic: which tickers drove bad_to_good_in_top60/top100, and what were their best events?

Usage:
  python3 scripts/diag_topn_flippers.py --as-of-date 2026-02-14 \
      --snapshot-dir data/snapshots \
      --rollup-csv output/catalyst_shadow_timeseries.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

GOOD_MODES = {"specific_days", "blended_window"}
BAD_MODES = {"no_upcoming", "missing"}

CONF_RANK = {"LOW": 0, "MED": 1, "HIGH": 2}
PREC_RANK = {"DAY": 4, "WEEK": 3, "MONTH": 2, "QUARTER": 1, "HALF_YEAR": 0, "YEAR": 0, "UNKNOWN": -1}


def _read_rollup_prior(rollup_csv: Path, as_of_date: str) -> Optional[str]:
    with rollup_csv.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("as_of_date") == as_of_date:
                pd = row.get("prior_date") or None
                return pd
    return None


def _read_rankings(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing rankings file: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _mode(row: Dict[str, str]) -> str:
    return row.get("catalyst_mode") or row.get("de_catalyst_mode") or ""


def _days(row: Dict[str, str]) -> Optional[float]:
    raw = row.get("catalyst_days") or row.get("de_catalyst_days") or ""
    if raw in ("", None):
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


def _top_n_set(rows: List[Dict[str, str]], n: int) -> set[str]:
    valid: List[Tuple[int, str]] = []
    for r in rows:
        cr = r.get("composite_rank", "")
        t = r.get("ticker", "")
        if not t or cr in ("", None):
            continue
        try:
            valid.append((int(cr), t))
        except (ValueError, TypeError):
            continue
    valid.sort()
    return {t for _, t in valid[:n]}


def _compute_bad_to_good(cur_rows: List[Dict[str, str]], prior_rows: List[Dict[str, str]]) -> List[str]:
    prior_mode_by_ticker = {
        r.get("ticker", ""): _mode(r)
        for r in prior_rows
        if r.get("archetype") == "drug_developer" and r.get("ticker", "")
    }

    flips: List[str] = []
    for r in cur_rows:
        if r.get("archetype") != "drug_developer":
            continue
        t = r.get("ticker", "")
        if not t or t not in prior_mode_by_ticker:
            continue  # new ticker, skip
        prev = prior_mode_by_ticker[t]
        cur = _mode(r)
        if prev in BAD_MODES and cur in GOOD_MODES:
            flips.append(t)
    return sorted(set(flips))


def _load_source_mix(snap_dir: Path, as_of_date: str) -> Optional[Dict[str, Any]]:
    p = snap_dir / as_of_date / "catalyst_source_mix.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _extract_8k_events_for_ticker(source_mix: Dict[str, Any], ticker: str) -> List[Dict[str, Any]]:
    """
    Best-effort. Supports a few plausible layouts:

    - source_mix["events_by_source"]["SEC_8K_FILING"] = [ {ticker, confidence, date_precision, ...}, ... ]
    - source_mix["by_source_details"]["SEC_8K_FILING"]["events"] = [...]
    - source_mix["SEC_8K_FILING"]["events"] = [...]
    """
    candidates: List[Dict[str, Any]] = []

    def add_from(obj: Any):
        if isinstance(obj, list):
            for e in obj:
                if isinstance(e, dict) and (e.get("ticker") == ticker or e.get("symbol") == ticker):
                    candidates.append(e)

    if isinstance(source_mix.get("events_by_source"), dict):
        add_from(source_mix["events_by_source"].get("SEC_8K_FILING"))
    if isinstance(source_mix.get("by_source_details"), dict):
        sec = source_mix["by_source_details"].get("SEC_8K_FILING")
        if isinstance(sec, dict):
            add_from(sec.get("events"))
    sec2 = source_mix.get("SEC_8K_FILING")
    if isinstance(sec2, dict):
        add_from(sec2.get("events"))

    return candidates


def _score_event(e: Dict[str, Any]) -> Tuple[int, int, float]:
    conf = str(e.get("confidence", "UNKNOWN")).upper()
    prec = str(e.get("date_precision", "UNKNOWN")).upper()
    days = e.get("catalyst_days")
    try:
        days_f = float(days) if days not in ("", None) else 1e9
    except (ValueError, TypeError):
        days_f = 1e9
    return (CONF_RANK.get(conf, -1), PREC_RANK.get(prec, -1), -days_f)  # higher is better; nearer is better


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of-date", required=True)
    ap.add_argument("--snapshot-dir", default="data/snapshots")
    ap.add_argument("--rollup-csv", default="output/catalyst_shadow_timeseries.csv")
    ap.add_argument("--show-top", type=int, default=5, help="How many best events to show per ticker (if available).")
    args = ap.parse_args()

    as_of = args.as_of_date
    snap_dir = Path(args.snapshot_dir)
    rollup = Path(args.rollup_csv)

    prior_date = _read_rollup_prior(rollup, as_of)
    if prior_date is None:
        print(f"No rollup row found for {as_of} in {rollup}")
        return 2
    if prior_date == "" or prior_date is None:
        print(f"{as_of}: prior_date is empty/None; no transitions to compute.")
        return 0

    cur_rankings = snap_dir / as_of / "rankings.csv"
    prior_rankings = snap_dir / prior_date / "rankings.csv"
    cur_rows = _read_rankings(cur_rankings)
    prior_rows = _read_rankings(prior_rankings)

    flips = _compute_bad_to_good(cur_rows, prior_rows)
    cur_top60 = _top_n_set(cur_rows, 60)
    cur_top100 = _top_n_set(cur_rows, 100)

    flips_60 = [t for t in flips if t in cur_top60]
    flips_100 = [t for t in flips if t in cur_top100]

    print(f"\nAs-of: {as_of}  Prior: {prior_date}")
    print(f"BAD→GOOD flips (dev-only, skipping new tickers): {len(flips)}")
    print(f"  in top-60:  {len(flips_60)}  {flips_60}")
    print(f"  in top-100: {len(flips_100)}  {flips_100}")

    if not flips_100:
        return 0

    # index current rows by ticker for quick lookup
    cur_by_t = {r.get("ticker", ""): r for r in cur_rows if r.get("ticker", "")}

    source_mix = _load_source_mix(snap_dir, as_of)

    print("\n=== Details for B→G tickers in top-100 ===")
    for t in flips_100:
        r = cur_by_t.get(t, {})
        rank = r.get("composite_rank", "")
        mode = _mode(r)
        days = _days(r)
        print(f"\n{t}  rank={rank}  mode={mode}  days={days}")

        if not source_mix:
            continue

        events = _extract_8k_events_for_ticker(source_mix, t)
        if not events:
            continue

        events_sorted = sorted(events, key=_score_event, reverse=True)[: args.show_top]
        print("  best SEC_8K_FILING events (best-effort):")
        for e in events_sorted:
            conf = e.get("confidence")
            prec = e.get("date_precision")
            edate = e.get("event_date") or e.get("disclosed_at") or e.get("date")
            etype = e.get("event_type") or e.get("type")
            cdays = e.get("catalyst_days")
            headline = e.get("headline") or e.get("summary") or e.get("title")
            print(f"    - {etype}  {edate}  conf={conf}  prec={prec}  days={cdays}  {headline or ''}".rstrip())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
