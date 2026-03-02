#!/usr/bin/env python3
"""EB 126d regression diagnosis.

RESEARCH ONLY — not for production.

1. Movers analysis: who enters/exits bottom-20 under EB vs E?
   What are their 126d realized returns?
2. Correlation: optionality_pct vs clinical_alpha_z cross-sectionally
3. Residualized clinical tilt: clinical_alpha_resid = zscore(clinical_alpha_z - beta*optionality_pct)
   Ablate at same clinical sort weight; compare 63/84/126d spread

Output: markdown summary printed to stdout.
"""
from __future__ import annotations

import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

E_ROOT   = PROJECT_ROOT / "data" / "snapshots_reranked"
EB_ROOT  = PROJECT_ROOT / "data" / "snapshots_reranked_eb_combined"
PRICE_CSV = PROJECT_ROOT / "production_data" / "price_history.csv"

K_TOP = 20
K_BOT = 20
HORIZONS = [63, 84, 126]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_float(v, default=float("nan")) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_rankings(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def eligible_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [r for r in rows if r.get("eligible") == "1"]


def sorted_by_rank(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    def _key(r):
        v = r.get("actionable_rank", "")
        try:
            return int(v)
        except (TypeError, ValueError):
            return 99999
    return sorted(rows, key=_key)


def z_score(vals: List[float]) -> List[float]:
    if len(vals) < 2:
        return [0.0] * len(vals)
    m = statistics.mean(vals)
    s = statistics.stdev(vals)
    if s == 0:
        return [0.0] * len(vals)
    return [(v - m) / s for v in vals]


def ols_beta(x: List[float], y: List[float]) -> float:
    """Simple OLS slope of y ~ x."""
    n = len(x)
    if n < 3:
        return 0.0
    mx, my = statistics.mean(x), statistics.mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = sum((xi - mx) ** 2 for xi in x)
    return num / den if den != 0 else 0.0


def pearson(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = statistics.mean(x), statistics.mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if sx == 0 or sy == 0:
        return float("nan")
    return num / (sx * sy)


# ---------------------------------------------------------------------------
# Price index
# ---------------------------------------------------------------------------

def build_price_index() -> Dict[str, Dict[str, float]]:
    """Returns {ticker: {date: close_price}}."""
    idx: Dict[str, Dict[str, float]] = defaultdict(dict)
    with open(PRICE_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = row["ticker"]
            d = row["date"]
            try:
                idx[t][d] = float(row["close"])
            except (ValueError, KeyError):
                pass
    return idx


def trading_dates_sorted(price_idx: Dict[str, Dict[str, float]]) -> List[str]:
    dates = set()
    for td in price_idx.values():
        dates.update(td.keys())
    return sorted(dates)


def forward_return(
    ticker: str,
    snap_date: str,
    horizon: int,
    price_idx: Dict[str, Dict[str, float]],
    trade_dates: List[str],
) -> Optional[float]:
    """Return pct return from snap_date+1 to snap_date+horizon trading days."""
    try:
        idx0 = trade_dates.index(snap_date)
    except ValueError:
        # find nearest date
        for i, d in enumerate(trade_dates):
            if d >= snap_date:
                idx0 = i - 1
                break
        else:
            return None
    if idx0 < 0:
        return None
    entry_idx = idx0 + 1
    exit_idx  = idx0 + horizon
    if exit_idx >= len(trade_dates):
        return None
    entry_date = trade_dates[entry_idx]
    exit_date  = trade_dates[exit_idx]
    p0 = price_idx.get(ticker, {}).get(entry_date)
    p1 = price_idx.get(ticker, {}).get(exit_date)
    if p0 and p1 and p0 > 0:
        return (p1 - p0) / p0
    return None


# ---------------------------------------------------------------------------
# Snapshot discovery
# ---------------------------------------------------------------------------

def snapshot_dates(root: Path) -> List[str]:
    return sorted([
        p.name for p in root.iterdir()
        if p.is_dir() and len(p.name) == 10 and (p / "rankings.csv").exists()
    ])


# ---------------------------------------------------------------------------
# Part 1: Movers analysis
# ---------------------------------------------------------------------------

def movers_analysis(
    price_idx: Dict[str, Dict[str, float]],
    trade_dates: List[str],
    cutoff_date: str,
) -> dict:
    """
    For each date <= cutoff_date (126d available), compare E vs EB:
    - top-K and bottom-K overlap
    - who enters bottom-K in EB that wasn't there in E?
    - their 126d returns
    """
    e_dates = set(snapshot_dates(E_ROOT))
    eb_dates = set(snapshot_dates(EB_ROOT))
    common = sorted(e_dates & eb_dates)
    common = [d for d in common if d <= cutoff_date]

    # Per-horizon: spread contribution tracking
    # spread = mean(top-K returns) - mean(bottom-K returns)
    spread_e:  Dict[int, List[float]] = {h: [] for h in HORIZONS}
    spread_eb: Dict[int, List[float]] = {h: [] for h in HORIZONS}

    # Movers: tickers that enter EB bottom-K but not E bottom-K
    # Collect (ticker, date, delta_rank, return_126d, optionality_pct, clinical_alpha_z)
    movers_into_bottom: List[dict] = []
    corr_data: List[Tuple[float, float]] = []  # (optionality_pct, clinical_alpha_z)

    n_dates = 0
    for d in common:
        e_rows = sorted_by_rank(eligible_rows(load_rankings(E_ROOT / d / "rankings.csv")))
        eb_rows = sorted_by_rank(eligible_rows(load_rankings(EB_ROOT / d / "rankings.csv")))
        if len(e_rows) < K_BOT + K_TOP or len(eb_rows) < K_BOT + K_TOP:
            continue

        e_tickers = [r["ticker"] for r in e_rows]
        eb_tickers = [r["ticker"] for r in eb_rows]
        n = max(len(e_tickers), len(eb_tickers))

        e_bottom = set(e_tickers[-K_BOT:])
        eb_bottom = set(eb_tickers[-K_BOT:])
        entered_bottom = eb_bottom - e_bottom  # new arrivals in EB bottom

        # Collect correlation data: all eligible rows with both signals
        e_by_ticker = {r["ticker"]: r for r in e_rows}
        for r in e_rows:
            opt = safe_float(r.get("clinical_optionality_pct_dev"))
            caz = safe_float(r.get("clinical_alpha_z"))
            if not (math.isnan(opt) or math.isnan(caz)):
                corr_data.append((opt, caz))

        # Movers into bottom
        for tk in entered_bottom:
            # find rank in E and EB
            e_rank = e_tickers.index(tk) + 1 if tk in e_tickers else None
            eb_rank = eb_tickers.index(tk) + 1 if tk in eb_tickers else None
            if e_rank is None or eb_rank is None:
                continue
            delta_rank = eb_rank - e_rank  # positive = moved down (worse)
            ret126 = forward_return(tk, d, 126, price_idx, trade_dates)
            row_data = e_by_ticker.get(tk, {})
            movers_into_bottom.append({
                "ticker": tk,
                "date": d,
                "e_rank": e_rank,
                "eb_rank": eb_rank,
                "delta_rank": delta_rank,
                "return_126d": ret126,
                "optionality_pct": safe_float(row_data.get("clinical_optionality_pct_dev")),
                "clinical_alpha_z": safe_float(row_data.get("clinical_alpha_z")),
            })

        # Spread computation per horizon
        for h in HORIZONS:
            e_top_rets = [
                forward_return(tk, d, h, price_idx, trade_dates)
                for tk in e_tickers[:K_TOP]
            ]
            e_bot_rets = [
                forward_return(tk, d, h, price_idx, trade_dates)
                for tk in e_tickers[-K_BOT:]
            ]
            eb_top_rets = [
                forward_return(tk, d, h, price_idx, trade_dates)
                for tk in eb_tickers[:K_TOP]
            ]
            eb_bot_rets = [
                forward_return(tk, d, h, price_idx, trade_dates)
                for tk in eb_tickers[-K_BOT:]
            ]
            e_top  = [r for r in e_top_rets if r is not None]
            e_bot  = [r for r in e_bot_rets if r is not None]
            eb_top = [r for r in eb_top_rets if r is not None]
            eb_bot = [r for r in eb_bot_rets if r is not None]
            if e_top and e_bot:
                spread_e[h].append(statistics.mean(e_top) - statistics.mean(e_bot))
            if eb_top and eb_bot:
                spread_eb[h].append(statistics.mean(eb_top) - statistics.mean(eb_bot))

        n_dates += 1

    return {
        "n_dates": n_dates,
        "common_dates": common,
        "spread_e": spread_e,
        "spread_eb": spread_eb,
        "movers_into_bottom": movers_into_bottom,
        "corr_data": corr_data,
    }


# ---------------------------------------------------------------------------
# Part 2: Residualized clinical tilt ablation
# ---------------------------------------------------------------------------

def residualized_ablation(
    price_idx: Dict[str, Dict[str, float]],
    trade_dates: List[str],
    cutoff_date: str,
) -> dict:
    """
    For each date: compute clinical_alpha_resid = zscore(caz - beta*opt) per date.
    Then re-sort using that residual as the clinical tilt (same positive-only clamp at w=0.3).
    Compare spread vs E baseline.
    """
    e_dates = set(snapshot_dates(E_ROOT))
    common = sorted(d for d in e_dates if d <= cutoff_date)

    CLINICAL_W = 0.3  # same as calendar_alpha_sort_weight in the active ruleset

    spread_e:     Dict[int, List[float]] = {h: [] for h in HORIZONS}
    spread_resid: Dict[int, List[float]] = {h: [] for h in HORIZONS}

    for d in common:
        e_rows = eligible_rows(load_rankings(E_ROOT / d / "rankings.csv"))
        if len(e_rows) < K_BOT + K_TOP:
            continue

        # Collect opt and caz
        tickers = [r["ticker"] for r in e_rows]
        opts = [safe_float(r.get("clinical_optionality_pct_dev")) for r in e_rows]
        cazs = [safe_float(r.get("clinical_alpha_z")) for r in e_rows]

        # OLS: caz ~ opt on rows with both valid
        valid = [(o, c) for o, c in zip(opts, cazs)
                 if not (math.isnan(o) or math.isnan(c))]
        if len(valid) < 5:
            continue
        vx, vy = zip(*valid)
        beta = ols_beta(list(vx), list(vy))

        # Compute residual for each row
        resid_vals = []
        for o, c in zip(opts, cazs):
            if math.isnan(o) or math.isnan(c):
                resid_vals.append(float("nan"))
            else:
                resid_vals.append(c - beta * o)

        # Z-score the residuals (cross-sectional, exclude nan)
        valid_resid = [v for v in resid_vals if not math.isnan(v)]
        if len(valid_resid) < 5:
            continue
        mr = statistics.mean(valid_resid)
        sr = statistics.stdev(valid_resid) if len(valid_resid) > 1 else 1.0
        if sr == 0:
            continue
        resid_z = [(v - mr) / sr if not math.isnan(v) else float("nan") for v in resid_vals]

        # Compute the effective sort rank for E (using optionality_pct) and for resid variant
        # E: anchor = optionality_pct (lower rank = better, subtract calendar alpha adj)
        # resid: same, but the calendar alpha tilt uses resid_z instead of clinical_score_v2_z
        # Simplified sort key: use (composite_rank, -CLINICAL_W * max(tilt, 0))
        composite_ranks = [safe_float(r.get("composite_rank"), default=9999) for r in e_rows]
        cal_v2_z = [safe_float(r.get("clinical_score_v2_z")) for r in e_rows]

        # E sort key: composite_rank - CLINICAL_W * clamp(cv2_z, 0, 2.0)
        # (lower is better)
        def clamp(v, lo, hi):
            if math.isnan(v):
                return 0.0
            return max(lo, min(hi, v))

        e_keys = [
            cr - CLINICAL_W * clamp(cv2, 0.0, 2.0)
            for cr, cv2 in zip(composite_ranks, cal_v2_z)
        ]
        resid_keys = [
            cr - CLINICAL_W * clamp(rz, 0.0, 2.0)
            for cr, rz in zip(composite_ranks, resid_z)
        ]

        e_order    = sorted(range(len(e_rows)), key=lambda i: e_keys[i])
        resid_order = sorted(range(len(e_rows)), key=lambda i: resid_keys[i])

        e_tickers_sorted     = [tickers[i] for i in e_order]
        resid_tickers_sorted = [tickers[i] for i in resid_order]

        for h in HORIZONS:
            e_top_rets = [forward_return(tk, d, h, price_idx, trade_dates)
                          for tk in e_tickers_sorted[:K_TOP]]
            e_bot_rets = [forward_return(tk, d, h, price_idx, trade_dates)
                          for tk in e_tickers_sorted[-K_BOT:]]
            r_top_rets = [forward_return(tk, d, h, price_idx, trade_dates)
                          for tk in resid_tickers_sorted[:K_TOP]]
            r_bot_rets = [forward_return(tk, d, h, price_idx, trade_dates)
                          for tk in resid_tickers_sorted[-K_BOT:]]

            e_t = [r for r in e_top_rets if r is not None]
            e_b = [r for r in e_bot_rets if r is not None]
            r_t = [r for r in r_top_rets if r is not None]
            r_b = [r for r in r_bot_rets if r is not None]

            if e_t and e_b:
                spread_e[h].append(statistics.mean(e_t) - statistics.mean(e_b))
            if r_t and r_b:
                spread_resid[h].append(statistics.mean(r_t) - statistics.mean(r_b))

    return {"spread_e": spread_e, "spread_resid": spread_resid}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def fmt_pct(v: Optional[float]) -> str:
    if v is None or math.isnan(v):
        return "n/a"
    return f"{v * 100:+.2f}%"


def main() -> None:
    print("Loading price index...")
    price_idx = build_price_index()
    trade_dates = trading_dates_sorted(price_idx)

    # Cutoff: latest snapshot with 126d forward data
    # 126 trading days before last price date
    last_price_date = trade_dates[-1]
    cutoff_idx = trade_dates.index(last_price_date) - 126
    if cutoff_idx < 0:
        print("Not enough price history for 126d analysis.")
        return
    cutoff_date = trade_dates[cutoff_idx]
    print(f"Price range: {trade_dates[0]} → {last_price_date}")
    print(f"126d cutoff: {cutoff_date}\n")

    # -----------------------------------------------------------------------
    # Part 1: Movers
    # -----------------------------------------------------------------------
    print("Running movers analysis (E vs EB)...")
    r1 = movers_analysis(price_idx, trade_dates, cutoff_date)

    # Spread summary
    print(f"\n## Part 1: E vs EB — Spread & Movers")
    print(f"\nDates evaluated: {r1['n_dates']}")
    print(f"\n### Top-{K_TOP}/Bottom-{K_BOT} Spread (annualized ~ multiply by 252/h)")
    print(f"{'Horizon':>8} | {'E spread':>10} | {'EB spread':>10} | {'Delta':>10}")
    print("-" * 46)
    for h in HORIZONS:
        se = r1["spread_e"][h]
        seb = r1["spread_eb"][h]
        me = statistics.mean(se) if se else float("nan")
        meb = statistics.mean(seb) if seb else float("nan")
        delta = meb - me if not (math.isnan(me) or math.isnan(meb)) else float("nan")
        print(f"{h:>8}d | {fmt_pct(me):>10} | {fmt_pct(meb):>10} | {fmt_pct(delta):>10}")

    # Correlation
    corr_data = r1["corr_data"]
    if len(corr_data) >= 5:
        opts, cazs = zip(*corr_data)
        r_oc = pearson(list(opts), list(cazs))
        print(f"\n### Correlation: optionality_pct vs clinical_alpha_z")
        print(f"Pooled (n={len(corr_data):,} rows): r = {r_oc:.3f}")
    else:
        print("\nInsufficient data for correlation.")

    # Top movers into EB bottom-K
    movers = r1["movers_into_bottom"]
    # Sort by delta_rank descending (largest downshift first)
    movers_sorted = sorted(movers, key=lambda m: m["delta_rank"], reverse=True)

    print(f"\n### Top Entrants into EB Bottom-{K_BOT} (worst to best realized 126d return)")
    # Aggregate by ticker: average return and median rank shift
    by_ticker: Dict[str, List[dict]] = defaultdict(list)
    for m in movers_sorted:
        by_ticker[m["ticker"]].append(m)

    # For each ticker: mean return_126d, count appearances
    ticker_summary = []
    for tk, entries in by_ticker.items():
        rets = [e["return_126d"] for e in entries if e["return_126d"] is not None]
        mean_ret = statistics.mean(rets) if rets else float("nan")
        mean_delta = statistics.mean(e["delta_rank"] for e in entries)
        mean_opt = statistics.mean(
            e["optionality_pct"] for e in entries
            if not math.isnan(e["optionality_pct"])
        ) if any(not math.isnan(e["optionality_pct"]) for e in entries) else float("nan")
        mean_caz = statistics.mean(
            e["clinical_alpha_z"] for e in entries
            if not math.isnan(e["clinical_alpha_z"])
        ) if any(not math.isnan(e["clinical_alpha_z"]) for e in entries) else float("nan")
        ticker_summary.append({
            "ticker": tk,
            "n_dates": len(entries),
            "mean_delta_rank": mean_delta,
            "mean_ret_126d": mean_ret,
            "mean_opt_pct": mean_opt,
            "mean_caz": mean_caz,
        })

    # Sort by mean 126d return ascending (worst first)
    ticker_summary.sort(key=lambda x: x["mean_ret_126d"]
                        if not math.isnan(x["mean_ret_126d"]) else float("inf"))

    print(f"\n{'Ticker':>6} | {'N':>3} | {'ΔRank':>6} | {'Ret126d':>8} | {'OptPct':>7} | {'CAZ':>6}")
    print("-" * 52)
    for ts in ticker_summary[:15]:
        print(
            f"{ts['ticker']:>6} | {ts['n_dates']:>3} | "
            f"{ts['mean_delta_rank']:>+6.0f} | "
            f"{fmt_pct(ts['mean_ret_126d']):>8} | "
            f"{ts['mean_opt_pct']:>7.3f} | "
            f"{ts['mean_caz']:>6.2f}"
        )

    # Characterize: entrants vs non-entrants
    all_movers_rets = [m["return_126d"] for m in movers if m["return_126d"] is not None]
    if all_movers_rets:
        print(f"\nAll EB-bottom entrants (n={len(all_movers_rets)}): "
              f"mean 126d = {fmt_pct(statistics.mean(all_movers_rets))}, "
              f"median = {fmt_pct(statistics.median(all_movers_rets))}")

    # -----------------------------------------------------------------------
    # Part 2: Residualized clinical tilt
    # -----------------------------------------------------------------------
    print("\n\n## Part 2: Residualized Clinical Tilt Ablation")
    print("Computing clinical_alpha_resid = zscore(caz - beta*opt) per date...")
    r2 = residualized_ablation(price_idx, trade_dates, cutoff_date)

    print(f"\n### Spread comparison: E vs E+Resid (clinical_alpha_z residualized on optionality_pct)")
    print(f"\n{'Horizon':>8} | {'E spread':>10} | {'Resid spread':>13} | {'Delta':>10}")
    print("-" * 50)
    for h in HORIZONS:
        se = r2["spread_e"][h]
        sr = r2["spread_resid"][h]
        me = statistics.mean(se) if se else float("nan")
        mr = statistics.mean(sr) if sr else float("nan")
        delta = mr - me if not (math.isnan(me) or math.isnan(mr)) else float("nan")
        print(f"{h:>8}d | {fmt_pct(me):>10} | {fmt_pct(mr):>13} | {fmt_pct(delta):>10}")

    print("""
### Interpretation
- If resid spread ≈ E spread at 126d → residualization successfully decouples the clinical tilt
  from optionality correlation → worth productizing as a clinical sort signal
- If resid spread still regresses → the correlation is not the root cause; the issue is
  purely that clinical_alpha_z pushes overexposed names into top-K regardless of residualization
""")


if __name__ == "__main__":
    main()
