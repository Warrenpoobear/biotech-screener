#!/usr/bin/env python3
"""Selector-only shadow — does the ranker earn its place? (Spec 094)

Production takes the selector's top-60 cohort, reorders it with the pairwise
ranker and holds the top 30 by ``actionable_rank``. This tool records, for each
daily snapshot, that production Top-30 next to the Top-30 the selector alone
would have chosen (``selector_score`` descending), and later compares their
equal-weight forward returns.

Capture-only and read-only with respect to the model: it reads a promoted
``rankings.csv`` and writes its own ledger. It does not change rankings,
scoring, eligibility, sizing or any gate.

Evidence discipline:
  - A capture written on the snapshot date is ``LIVE``. Anything written later
    (``--backfill``) is ``REPLAY`` and is reported separately, never as
    evidence.
  - Captures are append-only. A date already in the ledger is never rewritten.
  - Statistics use non-overlapping windows only.

Output:
    artifacts/selector_only_shadow/captures.jsonl   append-only ledger
    artifacts/selector_only_shadow/summary.json
    artifacts/selector_only_shadow/summary.md

Usage:
    python3 tools/selector_only_shadow.py --as-of-date 2026-11-12 --capture
    python3 tools/selector_only_shadow.py --summary
    python3 tools/selector_only_shadow.py --backfill          # REPLAY rows only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SCHEMA_VERSION = "selector_only_shadow.v1"
TOP_N = 30
HORIZONS = (5, 20)
BENCHMARK = "XBI"
MIN_PRICED = 25  # a basket with fewer priced names is not scored
MIN_WINDOWS_FOR_READING = 20

SNAPSHOTS_DIR = REPO_ROOT / "data" / "snapshots"
OUT_DIR = REPO_ROOT / "artifacts" / "selector_only_shadow"
PROVENANCE_EXCLUSIONS = REPO_ROOT / "config" / "snapshot_provenance_exclusions.json"

LIVE = "LIVE"
REPLAY = "REPLAY"


def default_price_csv() -> Path:
    """Split-adjusted closes when present, else the raw history."""
    adjusted = REPO_ROOT / "production_data" / "price_history_split_adj.csv"
    return adjusted if adjusted.exists() else REPO_ROOT / "production_data" / "price_history.csv"


def _num(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


# ── Baskets ──────────────────────────────────────────────────────────


def build_baskets(rows: Iterable[Dict[str, str]], top_n: int = TOP_N) -> Dict[str, List[str]]:
    """Production Top-N and selector-only Top-N from one snapshot's rows.

    Both draw from the same pool: eligible rows that carry an
    ``actionable_rank``. Ties in ``selector_score`` break on ticker so the
    result is deterministic.
    """
    pool = []
    for row in rows:
        ticker = (row.get("ticker") or "").strip()
        rank = _num(row.get("actionable_rank"))
        if not ticker or _num(row.get("eligible")) != 1.0 or rank is None:
            continue
        pool.append((ticker, rank, _num(row.get("selector_score")), _num(row.get("ranker_v2_rank"))))

    production = [t for t, _, _, _ in sorted(pool, key=lambda x: (x[1], x[0]))[:top_n]]
    scored = [(t, s) for t, _, s, _ in pool if s is not None]
    selector_only = [t for t, _ in sorted(scored, key=lambda x: (-x[1], x[0]))[:top_n]]
    baskets = {"production": production, "selector_only": selector_only}
    # Pairwise-ranker order: the ranker's cohort by ranker_v2_rank, then everything else by
    # selector_score. Recorded so the ranker's counterfactual keeps accruing after production
    # moved to selector order (2026-11-13). Absent when the snapshot carries no ranker ranks.
    cohort = [(t, r) for t, _, _, r in pool if r is not None]
    if cohort:
        in_cohort = {t for t, _ in cohort}
        rest = [(t, s) for t, s in scored if t not in in_cohort]
        ordered = [t for t, _ in sorted(cohort, key=lambda x: (x[1], x[0]))]
        ordered += [t for t, _ in sorted(rest, key=lambda x: (-x[1], x[0]))]
        baskets["ranker"] = ordered[:top_n]
    return baskets


def _read_rankings(path: Path) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── Ledger ───────────────────────────────────────────────────────────


def load_captures(out_dir: Path = OUT_DIR) -> List[Dict[str, Any]]:
    path = Path(out_dir) / "captures.jsonl"
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return sorted(rows, key=lambda r: r["as_of_date"])


def capture(
    as_of_date: str,
    *,
    snapshots_dir: Path = SNAPSHOTS_DIR,
    out_dir: Path = OUT_DIR,
    today: Optional[str] = None,
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Record both baskets for ``as_of_date``. Never rewrites an existing date."""
    rankings = Path(snapshots_dir) / as_of_date / "rankings.csv"
    if not rankings.exists():
        return {"status": "skipped", "reason": f"no rankings.csv for {as_of_date}"}

    if any(r["as_of_date"] == as_of_date for r in load_captures(out_dir)):
        return {"status": "exists", "as_of_date": as_of_date}

    baskets = build_baskets(_read_rankings(rankings))
    if len(baskets["production"]) < TOP_N or len(baskets["selector_only"]) < TOP_N:
        return {
            "status": "skipped",
            "reason": f"fewer than {TOP_N} names "
            f"(production={len(baskets['production'])}, selector_only={len(baskets['selector_only'])})",
        }

    today = today or date.today().isoformat()
    if mode is None:
        mode = LIVE if today == as_of_date else REPLAY
    overlap = len(set(baskets["production"]) & set(baskets["selector_only"]))
    row = {
        "schema": SCHEMA_VERSION,
        "as_of_date": as_of_date,
        "capture_mode": mode,
        "captured_on": today,
        "rankings_sha256": _sha256(rankings),
        "top_n": TOP_N,
        "overlap": overlap,
        "production": baskets["production"],
        "selector_only": baskets["selector_only"],
    }
    if len(baskets.get("ranker", [])) == TOP_N:
        row["ranker"] = baskets["ranker"]
        row["ranker_overlap_with_selector_only"] = len(set(baskets["ranker"]) & set(baskets["selector_only"]))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / "captures.jsonl", "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    return {"status": "captured", "as_of_date": as_of_date, "capture_mode": mode, "overlap": overlap}


def _excluded_dates(path: Path = PROVENANCE_EXCLUSIONS) -> set:
    try:
        return {r["date"] for r in json.loads(Path(path).read_text(encoding="utf-8")).get("snapshots", [])}
    except Exception:
        return set()


def backfill(
    *,
    snapshots_dir: Path = SNAPSHOTS_DIR,
    out_dir: Path = OUT_DIR,
    start_date: str = "2026-05-13",
    exclusions_path: Path = PROVENANCE_EXCLUSIONS,
    today: Optional[str] = None,
) -> Dict[str, int]:
    """Write REPLAY rows for existing snapshots that were not regenerated."""
    excluded = _excluded_dates(exclusions_path)
    counts = {"captured": 0, "exists": 0, "skipped": 0}
    for day in sorted(p.name for p in Path(snapshots_dir).iterdir() if p.is_dir()):
        if len(day) != 10 or day < start_date or day in excluded:
            continue
        result = capture(day, snapshots_dir=snapshots_dir, out_dir=out_dir, today=today, mode=REPLAY)
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    return counts


# ── Forward returns ──────────────────────────────────────────────────


def load_prices(price_csv: Path, tickers: set, start_date: str) -> Dict[str, Dict[str, float]]:
    prices: Dict[str, Dict[str, float]] = {}
    with open(price_csv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ticker, day = row.get("ticker", ""), row.get("date", "")
            if ticker in tickers and day >= start_date:
                close = _num(row.get("close"))
                if close is not None and close > 0:
                    prices.setdefault(ticker, {})[day] = close
    return prices


def _forward_return(series: Dict[str, float], calendar: List[str], as_of: str, horizon: int) -> Optional[float]:
    """Close-to-close return from ``as_of`` to ``horizon`` trading days later."""
    if as_of not in calendar:
        return None
    idx = calendar.index(as_of)
    if idx + horizon >= len(calendar):
        return None
    entry, exit_ = series.get(calendar[idx]), series.get(calendar[idx + horizon])
    if entry is None or exit_ is None:
        return None
    return exit_ / entry - 1.0


def _basket_return(
    tickers: List[str], prices: Dict[str, Dict[str, float]], calendar: List[str], as_of: str, horizon: int
) -> Tuple[Optional[float], int]:
    rets = [r for r in (_forward_return(prices.get(t, {}), calendar, as_of, horizon) for t in tickers) if r is not None]
    if len(rets) < MIN_PRICED:
        return None, len(rets)
    return sum(rets) / len(rets), len(rets)


def _non_overlapping(windows: List[Dict[str, Any]], calendar: List[str], horizon: int) -> List[Dict[str, Any]]:
    """Greedy pick of windows whose start dates are at least ``horizon`` trading days apart."""
    picked: List[Dict[str, Any]] = []
    last_idx: Optional[int] = None
    for w in windows:
        idx = calendar.index(w["as_of_date"])
        if last_idx is None or idx - last_idx >= horizon:
            picked.append(w)
            last_idx = idx
    return picked


def _stats(diffs: List[float]) -> Dict[str, Any]:
    n = len(diffs)
    if n == 0:
        return {"n": 0, "mean_diff_pct": None, "t_stat": None, "ranker_win_rate": None}
    mean = sum(diffs) / n
    t_stat = None
    if n >= 3:
        sd = statistics.stdev(diffs)
        if sd > 0:
            t_stat = round(mean / (sd / math.sqrt(n)), 2)
    return {
        "n": n,
        "mean_diff_pct": round(mean * 100, 3),
        "t_stat": t_stat,
        "ranker_win_rate": round(sum(1 for d in diffs if d > 0) / n, 3),
    }


def summarize(
    *,
    out_dir: Path = OUT_DIR,
    price_csv: Optional[Path] = None,
    horizons: Tuple[int, ...] = HORIZONS,
    write: bool = True,
) -> Dict[str, Any]:
    """Compare matured forward returns. ``diff`` = production minus selector-only."""
    price_csv = Path(price_csv) if price_csv is not None else default_price_csv()
    captures = load_captures(out_dir)
    result: Dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "price_source": price_csv.name,
        "n_captures": {m: sum(1 for c in captures if c["capture_mode"] == m) for m in (LIVE, REPLAY)},
        "min_windows_for_reading": MIN_WINDOWS_FOR_READING,
        "by_mode": {},
    }
    if not captures:
        result["status"] = "no_captures"
        return _write_summary(result, out_dir) if write else result

    tickers = {BENCHMARK}
    for c in captures:
        tickers.update(c["production"])
        tickers.update(c["selector_only"])
        tickers.update(c.get("ranker", []))
    prices = load_prices(price_csv, tickers, captures[0]["as_of_date"])
    calendar = sorted(prices.get(BENCHMARK, {}))

    for mode in (LIVE, REPLAY):
        mode_out: Dict[str, Any] = {}
        for h in horizons:
            windows = []
            ranker_windows = []
            for c in captures:
                if c["capture_mode"] != mode:
                    continue
                prod, n_prod = _basket_return(c["production"], prices, calendar, c["as_of_date"], h)
                sel, n_sel = _basket_return(c["selector_only"], prices, calendar, c["as_of_date"], h)
                bench = _forward_return(prices.get(BENCHMARK, {}), calendar, c["as_of_date"], h)
                if prod is None or sel is None:
                    continue
                windows.append(
                    {
                        "as_of_date": c["as_of_date"],
                        "overlap": c["overlap"],
                        "production_pct": round(prod * 100, 3),
                        "selector_only_pct": round(sel * 100, 3),
                        "benchmark_pct": round(bench * 100, 3) if bench is not None else None,
                        "diff_pct": round((prod - sel) * 100, 3),
                        "n_priced": [n_prod, n_sel],
                        "_diff": prod - sel,
                    }
                )
                if c.get("ranker"):
                    rk, n_rk = _basket_return(c["ranker"], prices, calendar, c["as_of_date"], h)
                    if rk is not None:
                        ranker_windows.append(
                            {
                                "as_of_date": c["as_of_date"],
                                "ranker_pct": round(rk * 100, 3),
                                "selector_only_pct": round(sel * 100, 3),
                                "diff_pct": round((rk - sel) * 100, 3),
                                "n_priced": [n_rk, n_sel],
                                "_diff": rk - sel,
                            }
                        )
            independent = _non_overlapping(windows, calendar, h)
            stats = _stats([w["_diff"] for w in independent])
            stats["n_matured_captures"] = len(windows)
            stats["reading"] = (
                "insufficient_windows" if stats["n"] < MIN_WINDOWS_FOR_READING else "descriptive_only_see_t_stat"
            )
            for w in windows:
                w.pop("_diff", None)
            mode_out[f"h{h}d"] = {"stats": stats, "windows": independent}
            # Ranker order minus selector-only. This is the comparison that stays informative
            # once production itself is selector-only and the first diff is zero by construction.
            rk_independent = _non_overlapping(ranker_windows, calendar, h)
            rk_stats = _stats([w["_diff"] for w in rk_independent])
            rk_stats["n_matured_captures"] = len(ranker_windows)
            rk_stats["reading"] = (
                "insufficient_windows" if rk_stats["n"] < MIN_WINDOWS_FOR_READING else "descriptive_only_see_t_stat"
            )
            for w in ranker_windows:
                w.pop("_diff", None)
            mode_out[f"h{h}d"]["ranker_minus_selector_only"] = {"stats": rk_stats, "windows": rk_independent}
        result["by_mode"][mode] = mode_out

    result["status"] = "ok"
    return _write_summary(result, out_dir) if write else result


def _write_summary(result: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    (out_dir / "summary.md").write_text(format_summary_md(result), encoding="utf-8", newline="\n")
    return result


def format_summary_md(result: Dict[str, Any]) -> str:
    lines = [
        "# Selector-only shadow — production Top-30 vs selector-only Top-30",
        "",
        "Diff = production (selector + ranker) minus selector-only, equal-weight forward return.",
        "A positive diff means the ranker helped. Statistics use non-overlapping windows only.",
        "",
        f"Captures: {result['n_captures'].get(LIVE, 0)} live, {result['n_captures'].get(REPLAY, 0)} replay. "
        f"Price source: `{result['price_source']}`.",
        "",
        f"**Only LIVE captures are evidence.** No reading is offered below "
        f"{result['min_windows_for_reading']} non-overlapping windows. REPLAY rows were written after the "
        "snapshot date and are shown for context.",
        "",
    ]
    for mode in (LIVE, REPLAY):
        block = result.get("by_mode", {}).get(mode)
        if not block:
            continue
        lines += [
            f"## {mode}",
            "",
            "| Horizon | Windows | Mean diff | t | Ranker win rate | Reading |",
            "|---|---|---|---|---|---|",
        ]
        for key, data in block.items():
            s = data["stats"]
            mean = f"{s['mean_diff_pct']:+.2f} pts" if s["mean_diff_pct"] is not None else "n/a"
            t_stat = f"{s['t_stat']:+.2f}" if s["t_stat"] is not None else "n/a"
            win = f"{s['ranker_win_rate']:.0%}" if s["ranker_win_rate"] is not None else "n/a"
            lines.append(f"| {key[1:]} | {s['n']} | {mean} | {t_stat} | {win} | {s['reading']} |")
        lines.append("")
        for key, data in block.items():
            if not data["windows"]:
                continue
            lines += [
                f"### {mode} — {key[1:]} windows",
                "",
                "| Date | Overlap | Production | Selector-only | XBI | Diff |",
                "|---|---|---|---|---|---|",
            ]
            for w in data["windows"]:
                bench = f"{w['benchmark_pct']:+.2f}%" if w["benchmark_pct"] is not None else "n/a"
                lines.append(
                    f"| {w['as_of_date']} | {w['overlap']}/{TOP_N} | {w['production_pct']:+.2f}% | "
                    f"{w['selector_only_pct']:+.2f}% | {bench} | {w['diff_pct']:+.2f} pts |"
                )
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--as-of-date")
    parser.add_argument("--capture", action="store_true", help="Record today's baskets (needs --as-of-date)")
    parser.add_argument("--summary", action="store_true", help="Recompute the forward-return comparison")
    parser.add_argument("--backfill", action="store_true", help="Write REPLAY rows for existing live snapshots")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--snapshots-dir", type=Path, default=SNAPSHOTS_DIR)
    parser.add_argument("--price-csv", type=Path, default=None)
    args = parser.parse_args()

    if args.capture:
        if not args.as_of_date:
            parser.error("--capture needs --as-of-date")
        print(json.dumps(capture(args.as_of_date, snapshots_dir=args.snapshots_dir, out_dir=args.out_dir)))
    if args.backfill:
        print(json.dumps(backfill(snapshots_dir=args.snapshots_dir, out_dir=args.out_dir)))
    if args.summary or args.capture or args.backfill:
        result = summarize(out_dir=args.out_dir, price_csv=args.price_csv)
        print(format_summary_md(result))
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
