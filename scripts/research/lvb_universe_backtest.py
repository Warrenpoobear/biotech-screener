"""LVB (low vol-beta) sleeve backtest: full eligible universe vs ex-Top-30 pool.

Research only (Tier 2). Changes no ranking, gate, sizing or account.

LVB = the 30 eligible names with the lowest risk score, equal weight, where the
score is the mean percentile rank of vol_60d and beta_120d to XBI. The sleeve is
built by the shadow tool's own builder (tools/low_risk_sleeve_shadow.build_sleeve),
so the backtest and the forward shadows share one definition:

  - ex_top30: pool excludes the production Top-30 (shadow variant `vol_beta`)
  - full:     pool includes the Top-30 (shadow variant `vol_beta_full`)
  - top30:    production Top-30 (actionable_rank 1..30), for reference

Monthly, from each month's last snapshot close to the next month's last snapshot close
(extra mid-month snapshots are skipped: month_end_dates), net
of 50 bp x one-way turnover (turnover measured against drifted weights). XBI is
gross. Closes are split-adjusted and dated on or before each date.

Limits: in-sample, survivor-only history, pseudo-PIT snapshots (see
docs/MODEL_DOCUMENTATION.md, 2026-11-17 research record). Not evidence for a
production change.

Usage:
  python scripts/research/lvb_universe_backtest.py \
      --snapshot-dir data/snapshots_pit_v9 --out artifacts/research/lvb_universe_backtest.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.low_risk_sleeve_shadow import _read_rankings, build_sleeve, load_prices, production_top  # noqa: E402

MEASURES = ["vol_60d", "beta_120d"]
VARIANTS = ("ex_top30", "full", "top30")
COST_ONE_WAY = 0.005
MIN_PRICED = 25


def _close_on_or_before(series: Dict[str, float], day: str):
    days = [d for d in series if d <= day]
    return series[max(days)] if days else None


def month_end_dates(dates: List[str]) -> List[str]:
    """The last snapshot date in each calendar month (plus the final date), sorted.

    Snapshot sets can hold extra mid-month dates (v9: 2026-02-27 beside 2026-02-28, and
    2026-04-16). Using every snapshot as a rebalance date makes 1-day and half-month
    periods that are then annualised as months.
    """
    ds = sorted(set(dates))
    return [d for i, d in enumerate(ds) if i + 1 == len(ds) or ds[i + 1][:7] != d[:7]]


def span_years(rows: List[dict]) -> float:
    """Calendar years from the first period's start (d0) to the last period's end (d1)."""
    return (date.fromisoformat(rows[-1]["d1"]) - date.fromisoformat(rows[0]["d0"])).days / 365.25


def _is_rebalance(d0: str, every: int, phase: int) -> bool:
    """Reconstitute when the snapshot month is on the schedule (every=1: monthly)."""
    return every == 1 or int(d0[5:7]) % every == phase % every


def run_variant(variant: str, dates: List[str], snap_dir: Path, prices, every: int = 1, phase: int = 0) -> List[dict]:
    """Monthly return rows. Between rebalances the book is held at drifted weights (no trades, no cost)."""
    out, prev_w = [], {}
    xbi = prices["XBI"]
    for d0, d1 in zip(dates, dates[1:]):
        rows = _read_rankings(snap_dir / d0 / "rankings.csv")
        top = production_top(rows)
        rebalance = not prev_w or _is_rebalance(d0, every, phase)
        if rebalance:
            if variant == "top30":
                names = top
            else:
                names = build_sleeve(rows, prices, d0, measures=MEASURES, exclude_production=(variant == "ex_top30"))[
                    "sleeve"
                ]
            if len(names) < 30:
                continue
        else:
            names = sorted(prev_w)
        rets = {}
        for t in names:
            a = _close_on_or_before(prices.get(t, {}), d0)
            z = _close_on_or_before(prices.get(t, {}), d1)
            if a and z:
                rets[t] = z / a - 1
        if len(rets) < MIN_PRICED:
            continue
        if rebalance:
            w = {t: 1 / len(rets) for t in rets}
            turnover = sum(abs(w.get(t, 0) - prev_w.get(t, 0)) for t in set(w) | set(prev_w)) / 2
        else:
            held = sum(prev_w[t] for t in rets)
            w = {t: prev_w[t] / held for t in rets}
            turnover = 0.0
        gross = sum(w[t] * rets[t] for t in rets)
        grown = sum(w[t] * (1 + rets[t]) for t in rets)
        prev_w = {t: w[t] * (1 + rets[t]) / grown for t in rets}
        out.append(
            {
                "d0": d0,
                "d1": d1,
                "rebalanced": rebalance,
                "net": gross - COST_ONE_WAY * turnover,
                "xbi": _close_on_or_before(xbi, d1) / _close_on_or_before(xbi, d0) - 1,
                "turnover": turnover,
                "n_priced": len(rets),
                "overlap_top30": len(set(top) & set(rets)),
                "names": sorted(rets),
            }
        )
    return out


def stats(rs: List[float], xs: List[float], years: Optional[float] = None) -> dict:
    """Annual return compounds over ``years`` when given (actual span), else treats each period as a month."""
    n = len(rs)
    m = sum(rs) / n
    sd = math.sqrt(sum((r - m) ** 2 for r in rs) / (n - 1))
    cum = peak = 1.0
    mdd = 0.0
    for r in rs:
        cum *= 1 + r
        peak = max(peak, cum)
        mdd = min(mdd, cum / peak - 1)
    mx = sum(xs) / n
    vx = sum((x - mx) ** 2 for x in xs)
    beta = sum((x - mx) * (r - m) for x, r in zip(xs, rs)) / vx
    alpha = m - beta * mx
    s2 = sum((r - alpha - beta * x) ** 2 for r, x in zip(rs, xs)) / (n - 2)
    se_a = math.sqrt(s2 * (1 / n + mx * mx / vx))
    ex = [r - x for r, x in zip(rs, xs)]
    me = sum(ex) / n
    sde = math.sqrt(sum((e - me) ** 2 for e in ex) / (n - 1))
    rel = rpk = 1.0
    worst = 0.0
    for r, x in zip(rs, xs):
        rel *= (1 + r) / (1 + x)
        rpk = max(rpk, rel)
        worst = min(worst, rel / rpk - 1)
    return {
        "months": n,
        "ann_return": cum ** (1 / (years if years else n / 12)) - 1,
        "ann_vol": sd * math.sqrt(12),
        "sharpe": m / sd * math.sqrt(12),
        "max_drawdown": mdd,
        "beta_xbi": beta,
        "alpha_xbi_pt_month": alpha * 100,
        "alpha_xbi_t": alpha / se_a if se_a > 1e-12 else None,
        "excess_xbi_pt_month": me * 100,
        "excess_xbi_t": me / (sde / math.sqrt(n)) if sde > 1e-12 else None,
        "worst_stretch_vs_xbi": worst,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot-dir", default="data/snapshots_pit_v9")
    ap.add_argument("--prices", default="production_data/price_history_split_adj.csv")
    ap.add_argument(
        "--rebalance-every", type=int, default=1, choices=[1, 3, 6, 12], help="months between reconstitutions"
    )
    ap.add_argument("--phase", type=int, default=0, help="rebalance when snapshot month %% every == phase")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    snap_dir = (REPO / args.snapshot_dir).resolve()
    dates = month_end_dates(
        [p.name for p in snap_dir.iterdir() if p.is_dir() and p.name[:2] == "20" and (p / "rankings.csv").exists()]
    )
    prices = load_prices(REPO / args.prices)
    monthly = {v: run_variant(v, dates, snap_dir, prices, args.rebalance_every, args.phase) for v in VARIANTS}

    common = sorted(set.intersection(*[{r["d0"] for r in monthly[v]} for v in VARIANTS]))
    keep = {v: [r for r in monthly[v] if r["d0"] in set(common)] for v in VARIANTS}
    summary = {}
    for v in VARIANTS:
        rs = keep[v]
        s = stats([r["net"] for r in rs], [r["xbi"] for r in rs], years=span_years(rs))
        s["avg_turnover"] = sum(r["turnover"] for r in rs) / len(rs)
        s["avg_overlap_top30"] = sum(r["overlap_top30"] for r in rs) / len(rs)
        s["first"], s["last"] = rs[0]["d0"], rs[-1]["d1"]
        summary[v] = s
    xs = [r["xbi"] for r in keep["full"]]
    summary["XBI"] = stats(xs, xs, years=span_years(keep["full"]))

    diff = [a["net"] - b["net"] for a, b in zip(keep["full"], keep["ex_top30"])]
    md = sum(diff) / len(diff)
    sdd = math.sqrt(sum((x - md) ** 2 for x in diff) / (len(diff) - 1))
    summary["full_minus_ex_top30"] = {"pt_month": md * 100, "t": md / (sdd / math.sqrt(len(diff)))}

    yearly: Dict[str, Dict[str, float]] = {}
    for v in VARIANTS:
        for r in keep[v]:
            y = yearly.setdefault(r["d1"][:4], {})
            y[v] = y.get(v, 1.0) * (1 + r["net"])
    for r in keep["full"]:
        y = yearly[r["d1"][:4]]
        y["XBI"] = y.get("XBI", 1.0) * (1 + r["xbi"])
    summary["calendar_year_return_pct"] = {
        yr: {k: round((g - 1) * 100, 1) for k, g in d.items()} for yr, d in sorted(yearly.items())
    }

    result = {
        "schema": "lvb_universe_backtest.v2",
        "method": {
            "measures": MEASURES,
            "cost_one_way": COST_ONE_WAY,
            "snapshot_dir": args.snapshot_dir,
            "prices": args.prices,
            "rebalance_every_months": args.rebalance_every,
            "phase": args.phase,
            "calendar": "month_end",
            "annualisation": "actual_span",
        },
        "summary": summary,
        "monthly": keep,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
