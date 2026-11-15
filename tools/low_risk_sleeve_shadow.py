#!/usr/bin/env python3
"""Low-risk sleeve shadow — forward record for a low-volatility biotech sleeve.

Research question (backtest 2026-11-15, 84 months, in-sample): an equal-weight
basket of the 30 lowest-risk eligible names had beta ~0.43 to XBI, CAPM alpha
+0.69 pt/month (t = 1.6) and half the drawdown, and its excess returns were
nearly uncorrelated with the Top-30's. This tool builds that out-of-sample
record. It is not a production signal.

Each day it reads the promoted ``rankings.csv``, takes the eligible names that
are NOT in the production Top-30 (``actionable_rank`` 1..30), scores them on
three point-in-time risk measures from the split-adjusted price history, and
records the 30 lowest-risk names next to the production Top-30. Later it
compares equal-weight forward returns against XBI and the Top-30.

Capture-only and read-only with respect to the model: it reads a promoted
``rankings.csv`` and the price history, and writes its own ledger. It does not
change rankings, scoring, eligibility, sizing, any gate, or any account.

Risk measures (each uses only closes dated on or before the snapshot date):
  vol_60d       annualised standard deviation of daily log returns, last 60 days
  max_gain_21d  largest one-day simple return, last 21 days ("lottery" measure)
  beta_120d     beta of daily log returns to XBI, last 120 days
The sleeve score is the mean of the available percentile ranks (lower risk =
higher score). A name needs ``vol_60d`` to be scored.

Evidence discipline (same as tools/selector_only_shadow.py):
  - A capture written on the snapshot date is ``LIVE``; anything written later
    is ``REPLAY`` and is reported separately, never as evidence.
  - Captures are append-only. A date already in the ledger is never rewritten.
  - A capture is refused when the newest XBI close is more than
    ``MAX_PRICE_LAG_DAYS`` calendar days before the snapshot date.
  - Statistics use non-overlapping windows only.
  - The method is fixed. Changing it means a new ``SCHEMA_VERSION`` and a new
    ledger, not a rewrite of this one.

Output:
    artifacts/low_risk_sleeve_shadow/captures.jsonl   append-only ledger
    artifacts/low_risk_sleeve_shadow/summary.json
    artifacts/low_risk_sleeve_shadow/summary.md

Usage:
    python3 tools/low_risk_sleeve_shadow.py --as-of-date 2026-11-16 --capture
    python3 tools/low_risk_sleeve_shadow.py --summary
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SCHEMA_VERSION = "low_risk_sleeve_shadow.v1"
SLEEVE_N = 30
PRODUCTION_N = 30
HORIZONS = (5, 21)
BENCHMARK = "XBI"
MIN_PRICED = 25  # a basket with fewer priced names is not scored
MIN_WINDOWS_FOR_READING = 20
MAX_PRICE_LAG_DAYS = 5
VOL_WINDOW, MIN_VOL_RETURNS = 60, 40
GAIN_WINDOW, MIN_GAIN_RETURNS = 21, 15
BETA_WINDOW, MIN_BETA_RETURNS = 120, 80
PRICE_LOOKBACK_DAYS = 200  # calendar days of history loaded for a capture

SNAPSHOTS_DIR = REPO_ROOT / "data" / "snapshots"
OUT_DIR = REPO_ROOT / "artifacts" / "low_risk_sleeve_shadow"

LIVE = "LIVE"
REPLAY = "REPLAY"

METHOD = {
    "pool": "eligible names in rankings.csv, excluding production Top-30 (actionable_rank 1..30)",
    "measures": ["vol_60d", "max_gain_21d", "beta_120d"],
    "score": "mean of available percentile ranks, lower risk = higher score; vol_60d required",
    "basket": f"top {SLEEVE_N} by score, equal weight; ties break on ticker",
    "price_source": "production_data/price_history_split_adj.csv, closes dated <= as_of_date",
}

PRE_REGISTRATION = [
    "Pre-registered 2026-11-15, before the first capture.",
    "- The sleeve is judged on risk-adjusted terms, not on excess return over XBI: by design it has a beta near "
    "0.4 and will trail XBI in strong rallies.",
    "- Primary reading: CAPM alpha of the sleeve against XBI on non-overlapping 21-trading-day windows, "
    "and whether the 50/50 Top-30 + sleeve mix has a higher Sharpe ratio than the Top-30 alone.",
    "- Secondary: the same on non-overlapping 5-day windows; correlation of sleeve and Top-30 excess returns.",
    f"- No reading below {MIN_WINDOWS_FOR_READING} non-overlapping LIVE windows at a horizon "
    "(about 5 months at 5 days, about 20 months at 21 days).",
    "- Backtest expectation for comparison (in-sample, 84 months): alpha +0.69 pt/month, beta 0.43, "
    "correlation with Top-30 excess 0.14.",
    "- Nothing here changes the model or the account. Using the sleeve with real money is a separate governed decision.",
]


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


def _read_rankings(path: Path) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_prices(
    price_csv: Path, tickers: Optional[set] = None, start_date: str = "", end_date: str = "9999-12-31"
) -> Dict[str, Dict[str, float]]:
    """``{ticker: {date: close}}`` for positive closes in ``[start_date, end_date]``."""
    prices: Dict[str, Dict[str, float]] = {}
    with open(price_csv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ticker, day = row.get("ticker", ""), row.get("date", "")
            if tickers is not None and ticker not in tickers:
                continue
            if not (start_date <= day <= end_date):
                continue
            close = _num(row.get("close"))
            if close is not None and close > 0:
                prices.setdefault(ticker, {})[day] = close
    return prices


# ── Risk measures ────────────────────────────────────────────────────


def _closes_upto(series: Dict[str, float], as_of: str, n: int) -> List[Tuple[str, float]]:
    """The last ``n + 1`` (date, close) pairs dated on or before ``as_of``."""
    pts = sorted((d, c) for d, c in series.items() if d <= as_of)
    return pts[-(n + 1) :]


def risk_measures(series: Dict[str, float], xbi: Dict[str, float], as_of: str) -> Dict[str, Optional[float]]:
    """Point-in-time risk measures for one ticker. Uses no close dated after ``as_of``."""
    out: Dict[str, Optional[float]] = {"vol_60d": None, "max_gain_21d": None, "beta_120d": None}

    pts = _closes_upto(series, as_of, VOL_WINDOW)
    logs = [math.log(b[1] / a[1]) for a, b in zip(pts, pts[1:])]
    if len(logs) >= MIN_VOL_RETURNS:
        out["vol_60d"] = statistics.pstdev(logs) * math.sqrt(252)

    pts = _closes_upto(series, as_of, GAIN_WINDOW)
    simple = [b[1] / a[1] - 1 for a, b in zip(pts, pts[1:])]
    if len(simple) >= MIN_GAIN_RETURNS:
        out["max_gain_21d"] = max(simple)

    pts = _closes_upto(series, as_of, BETA_WINDOW)
    pairs = []
    for a, b in zip(pts, pts[1:]):
        xa, xb = xbi.get(a[0]), xbi.get(b[0])
        if xa and xb:
            pairs.append((math.log(b[1] / a[1]), math.log(xb / xa)))
    if len(pairs) >= MIN_BETA_RETURNS:
        ys, xs = [p[0] for p in pairs], [p[1] for p in pairs]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        var = sum((x - mx) ** 2 for x in xs)
        if var > 0:
            out["beta_120d"] = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var
    return out


def _pct_ranks(values: Dict[str, float]) -> Dict[str, float]:
    """Average-rank percentile in (0, 1]; higher value = higher percentile."""
    items = sorted(values.items(), key=lambda kv: kv[1])
    n = len(items)
    out: Dict[str, float] = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and items[j + 1][1] == items[i][1]:
            j += 1
        pct = ((i + j) / 2 + 1) / n
        for k in range(i, j + 1):
            out[items[k][0]] = pct
        i = j + 1
    return out


def production_top(rows: Iterable[Dict[str, str]], top_n: int = PRODUCTION_N) -> List[str]:
    pool = []
    for row in rows:
        ticker = (row.get("ticker") or "").strip()
        rank = _num(row.get("actionable_rank"))
        if ticker and _num(row.get("eligible")) == 1.0 and rank is not None:
            pool.append((rank, ticker))
    return [t for _, t in sorted(pool)[:top_n]]


def build_sleeve(rows: List[Dict[str, str]], prices: Dict[str, Dict[str, float]], as_of: str) -> Dict[str, Any]:
    """Production Top-30 and the low-risk sleeve (disjoint from it) for one snapshot."""
    production = production_top(rows)
    held = set(production)
    xbi = prices.get(BENCHMARK, {})
    measures: Dict[str, Dict[str, Optional[float]]] = {}
    for row in rows:
        ticker = (row.get("ticker") or "").strip()
        if not ticker or ticker in held or _num(row.get("eligible")) != 1.0:
            continue
        m = risk_measures(prices.get(ticker, {}), xbi, as_of)
        if m["vol_60d"] is not None:
            measures[ticker] = m

    # Lower risk = higher percentile: rank the negated measure.
    ranks = {k: _pct_ranks({t: -m[k] for t, m in measures.items() if m[k] is not None}) for k in METHOD["measures"]}
    scores = {}
    for t in measures:
        got = [ranks[k][t] for k in METHOD["measures"] if t in ranks[k]]
        scores[t] = sum(got) / len(got)
    sleeve = [t for t, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:SLEEVE_N]]
    price_asof = max((d for d in xbi if d <= as_of), default=None)
    return {
        "production": production,
        "sleeve": sleeve,
        "measures": {t: dict(measures[t], score=round(scores[t], 6)) for t in sleeve},
        "n_scored": len(scores),
        "price_asof": price_asof,
    }


# ── Ledger ───────────────────────────────────────────────────────────


def load_captures(out_dir: Path = OUT_DIR) -> List[Dict[str, Any]]:
    path = Path(out_dir) / "captures.jsonl"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return sorted(rows, key=lambda r: r["as_of_date"])


def _round_measures(m: Dict[str, Any]) -> Dict[str, Any]:
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items()}


def capture(
    as_of_date: str,
    *,
    snapshots_dir: Path = SNAPSHOTS_DIR,
    out_dir: Path = OUT_DIR,
    price_csv: Optional[Path] = None,
    today: Optional[str] = None,
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Record the sleeve and the production Top-30 for ``as_of_date``. Never rewrites a date."""
    rankings = Path(snapshots_dir) / as_of_date / "rankings.csv"
    if not rankings.exists():
        return {"status": "skipped", "reason": f"no rankings.csv for {as_of_date}"}
    if any(r["as_of_date"] == as_of_date for r in load_captures(out_dir)):
        return {"status": "exists", "as_of_date": as_of_date}

    rows = _read_rankings(rankings)
    price_csv = Path(price_csv) if price_csv is not None else default_price_csv()
    start = (date.fromisoformat(as_of_date) - timedelta(days=PRICE_LOOKBACK_DAYS)).isoformat()
    tickers = {(r.get("ticker") or "").strip() for r in rows} | {BENCHMARK}
    prices = load_prices(price_csv, tickers, start, as_of_date)
    result = build_sleeve(rows, prices, as_of_date)

    price_asof = result["price_asof"]
    if (
        price_asof is None
        or (date.fromisoformat(as_of_date) - date.fromisoformat(price_asof)).days > MAX_PRICE_LAG_DAYS
    ):
        return {"status": "skipped", "reason": f"stale prices: newest {BENCHMARK} close {price_asof} for {as_of_date}"}
    if len(result["production"]) < PRODUCTION_N or len(result["sleeve"]) < SLEEVE_N:
        return {
            "status": "skipped",
            "reason": f"too few names (production={len(result['production'])}, sleeve={len(result['sleeve'])})",
        }

    today = today or date.today().isoformat()
    if mode is None:
        mode = LIVE if today == as_of_date else REPLAY
    row = {
        "schema": SCHEMA_VERSION,
        "as_of_date": as_of_date,
        "capture_mode": mode,
        "captured_on": today,
        "rankings_sha256": _sha256(rankings),
        "price_source": price_csv.name,
        "price_asof": price_asof,
        "method": METHOD,
        "n_scored": result["n_scored"],
        "production": result["production"],
        "sleeve": result["sleeve"],
        "sleeve_measures": {t: _round_measures(m) for t, m in result["measures"].items()},
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / "captures.jsonl", "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    return {"status": "captured", "as_of_date": as_of_date, "capture_mode": mode, "price_asof": price_asof}


# ── Forward returns ──────────────────────────────────────────────────


def _forward_return(series: Dict[str, float], calendar: List[str], as_of: str, horizon: int) -> Optional[float]:
    if as_of not in calendar:
        return None
    idx = calendar.index(as_of)
    if idx + horizon >= len(calendar):
        return None
    entry, exit_ = series.get(calendar[idx]), series.get(calendar[idx + horizon])
    if entry is None or exit_ is None:
        return None
    return exit_ / entry - 1.0


def _basket_return(tickers, prices, calendar, as_of, horizon) -> Tuple[Optional[float], int]:
    rets = [r for r in (_forward_return(prices.get(t, {}), calendar, as_of, horizon) for t in tickers) if r is not None]
    if len(rets) < MIN_PRICED:
        return None, len(rets)
    return sum(rets) / len(rets), len(rets)


def _non_overlapping(windows: List[Dict[str, Any]], calendar: List[str], horizon: int) -> List[Dict[str, Any]]:
    picked: List[Dict[str, Any]] = []
    last_idx: Optional[int] = None
    for w in windows:
        idx = calendar.index(w["as_of_date"])
        if last_idx is None or idx - last_idx >= horizon:
            picked.append(w)
            last_idx = idx
    return picked


def _series_stats(xs: List[float]) -> Dict[str, Any]:
    n = len(xs)
    out: Dict[str, Any] = {"n": n, "mean_pct": None, "t_stat": None}
    if n:
        mean = sum(xs) / n
        out["mean_pct"] = round(mean * 100, 3)
        if n >= 3:
            sd = statistics.stdev(xs)
            if sd > 0:
                out["t_stat"] = round(mean / (sd / math.sqrt(n)), 2)
    out["reading"] = "insufficient_windows" if n < MIN_WINDOWS_FOR_READING else "descriptive_only_see_t_stat"
    return out


def _capm(y: List[float], x: List[float]) -> Dict[str, Any]:
    """OLS of sleeve return on XBI return: alpha per window, its t, and beta."""
    n = len(y)
    if n < 5:
        return {"n": n, "alpha_pct": None, "alpha_t": None, "beta": None}
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    if sxx <= 0:
        return {"n": n, "alpha_pct": None, "alpha_t": None, "beta": None}
    beta = sum((a - mx) * (b - my) for a, b in zip(x, y)) / sxx
    alpha = my - beta * mx
    resid = [b - alpha - beta * a for a, b in zip(x, y)]
    s2 = sum(r * r for r in resid) / (n - 2)
    se = math.sqrt(s2 * (1 / n + mx * mx / sxx)) if s2 > 0 else 0.0
    return {
        "n": n,
        "alpha_pct": round(alpha * 100, 3),
        "alpha_t": round(alpha / se, 2) if se > 0 else None,
        "beta": round(beta, 3),
    }


def _sharpe(xs: List[float]) -> Optional[float]:
    if len(xs) < 3:
        return None
    sd = statistics.stdev(xs)
    return round(sum(xs) / len(xs) / sd, 3) if sd > 0 else None


def summarize(
    *,
    out_dir: Path = OUT_DIR,
    price_csv: Optional[Path] = None,
    horizons: Tuple[int, ...] = HORIZONS,
    write: bool = True,
) -> Dict[str, Any]:
    price_csv = Path(price_csv) if price_csv is not None else default_price_csv()
    captures = load_captures(out_dir)
    result: Dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "price_source": price_csv.name,
        "method": METHOD,
        "pre_registration": PRE_REGISTRATION,
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
        tickers.update(c["sleeve"])
    prices = load_prices(price_csv, tickers, captures[0]["as_of_date"])
    calendar = sorted(prices.get(BENCHMARK, {}))

    for mode in (LIVE, REPLAY):
        mode_out: Dict[str, Any] = {}
        for h in horizons:
            windows = []
            for c in captures:
                if c["capture_mode"] != mode:
                    continue
                d = c["as_of_date"]
                sl, n_sl = _basket_return(c["sleeve"], prices, calendar, d, h)
                pr, n_pr = _basket_return(c["production"], prices, calendar, d, h)
                bx = _forward_return(prices.get(BENCHMARK, {}), calendar, d, h)
                if sl is None or pr is None or bx is None:
                    continue
                windows.append(
                    {
                        "as_of_date": d,
                        "sleeve_pct": round(sl * 100, 3),
                        "production_pct": round(pr * 100, 3),
                        "combo_pct": round((sl + pr) / 2 * 100, 3),
                        "xbi_pct": round(bx * 100, 3),
                        "n_priced": [n_sl, n_pr],
                        "_r": (sl, pr, bx),
                    }
                )
            ind = _non_overlapping(windows, calendar, h)
            sl_r = [w["_r"][0] for w in ind]
            pr_r = [w["_r"][1] for w in ind]
            bx_r = [w["_r"][2] for w in ind]
            combo = [(a + b) / 2 for a, b in zip(sl_r, pr_r)]
            ex_sl = [a - b for a, b in zip(sl_r, bx_r)]
            ex_pr = [a - b for a, b in zip(pr_r, bx_r)]
            corr = None
            if len(ex_sl) >= 3 and statistics.pstdev(ex_sl) > 0 and statistics.pstdev(ex_pr) > 0:
                corr = round(statistics.correlation(ex_sl, ex_pr), 3)
            stats = {
                "sleeve_minus_xbi": _series_stats(ex_sl),
                "production_minus_xbi": _series_stats(ex_pr),
                "combo_minus_xbi": _series_stats([a - b for a, b in zip(combo, bx_r)]),
                "sleeve_minus_production": _series_stats([a - b for a, b in zip(sl_r, pr_r)]),
                "sleeve_capm": _capm(sl_r, bx_r),
                "sharpe_per_window": {
                    "sleeve": _sharpe(sl_r),
                    "production": _sharpe(pr_r),
                    "combo": _sharpe(combo),
                    "xbi": _sharpe(bx_r),
                },
                "corr_sleeve_vs_production_excess": corr,
                "n_matured_captures": len(windows),
            }
            for w in windows:
                w.pop("_r", None)
            mode_out[f"h{h}d"] = {"stats": stats, "windows": ind}
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


def _fmt(v: Any, spec: str, suffix: str = "") -> str:
    return "n/a" if v is None else f"{v:{spec}}{suffix}"


def format_summary_md(result: Dict[str, Any]) -> str:
    lines = [
        "# Low-risk sleeve shadow — forward record",
        "",
        "Sleeve = the 30 lowest-risk eligible names outside the production Top-30 (60-day volatility, "
        "largest one-day gain over 21 days, 120-day beta to XBI), equal weight. Compared with XBI and the "
        "production Top-30 on equal-weight forward returns, gross of costs. Non-overlapping windows only.",
        "",
        *PRE_REGISTRATION,
        "",
        f"Captures: {result['n_captures'].get(LIVE, 0)} live, {result['n_captures'].get(REPLAY, 0)} replay. "
        f"Price source: `{result['price_source']}`. **Only LIVE captures are evidence.**",
        "",
    ]
    for mode in (LIVE, REPLAY):
        block = result.get("by_mode", {}).get(mode)
        if not block:
            continue
        lines += [
            f"## {mode}",
            "",
            "| Horizon | Windows | Sleeve − XBI | t | Sleeve alpha | t | Beta | Combo − XBI | Top-30 − XBI | "
            "Sharpe sleeve / combo / Top-30 / XBI | Corr | Reading |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for key, data in block.items():
            s = data["stats"]
            a, sh = s["sleeve_capm"], s["sharpe_per_window"]
            sharpe = " / ".join(_fmt(sh[k], ".2f") for k in ("sleeve", "combo", "production", "xbi"))
            lines.append(
                f"| {key[1:]} | {s['sleeve_minus_xbi']['n']} | {_fmt(s['sleeve_minus_xbi']['mean_pct'], '+.2f', ' pts')} | "
                f"{_fmt(s['sleeve_minus_xbi']['t_stat'], '+.2f')} | {_fmt(a['alpha_pct'], '+.2f', ' pts')} | "
                f"{_fmt(a['alpha_t'], '+.2f')} | {_fmt(a['beta'], '.2f')} | "
                f"{_fmt(s['combo_minus_xbi']['mean_pct'], '+.2f', ' pts')} | "
                f"{_fmt(s['production_minus_xbi']['mean_pct'], '+.2f', ' pts')} | {sharpe} | "
                f"{_fmt(s['corr_sleeve_vs_production_excess'], '+.2f')} | {s['sleeve_minus_xbi']['reading']} |"
            )
        lines.append("")
        for key, data in block.items():
            if not data["windows"]:
                continue
            lines += [
                f"### {mode} — {key[1:]} windows",
                "",
                "| Date | Sleeve | Top-30 | 50/50 | XBI |",
                "|---|---|---|---|---|",
            ]
            for w in data["windows"]:
                lines.append(
                    f"| {w['as_of_date']} | {w['sleeve_pct']:+.2f}% | {w['production_pct']:+.2f}% | "
                    f"{w['combo_pct']:+.2f}% | {w['xbi_pct']:+.2f}% |"
                )
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--as-of-date")
    parser.add_argument("--capture", action="store_true", help="Record the sleeve for --as-of-date")
    parser.add_argument("--summary", action="store_true", help="Recompute the forward-return comparison")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--snapshots-dir", type=Path, default=SNAPSHOTS_DIR)
    parser.add_argument("--price-csv", type=Path, default=None)
    args = parser.parse_args()

    if args.capture:
        if not args.as_of_date:
            parser.error("--capture needs --as-of-date")
        res = capture(args.as_of_date, snapshots_dir=args.snapshots_dir, out_dir=args.out_dir, price_csv=args.price_csv)
        print(json.dumps(res))
    if args.summary or args.capture:
        print(format_summary_md(summarize(out_dir=args.out_dir, price_csv=args.price_csv)))
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
