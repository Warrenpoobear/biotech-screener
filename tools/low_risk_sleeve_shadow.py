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

LVB (low vol-beta): the variants ``lvb`` (outside the Top-30) and ``lvb_full`` (full
eligible universe, standalone) score on ``vol_60d`` and ``beta_120d`` only. They were
registered as ``vol_beta`` / ``vol_beta_full``; those names still work as aliases, and
the ledgers keep their original schema strings and directories.

Exclusion: a name whose 20-day annualised volatility is below ``PINNED_VOL_20D``
(10%) is left out and listed in ``excluded_pinned``. Such a price is pinned near a
pending cash-deal price (e.g. TECH, OPCH, PCRX on 2026-11-12) or the feed is stale;
it would otherwise rank as the "safest" name while carrying deal-break risk.

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
PINNED_WINDOW = 20  # trading days
PINNED_VOL_20D = 0.10  # annualised; below this the price is pinned (pending cash deal) or stale
GAIN_WINDOW, MIN_GAIN_RETURNS = 21, 15
BETA_WINDOW, MIN_BETA_RETURNS = 120, 80
PRICE_LOOKBACK_DAYS = 200  # calendar days of history loaded for a capture

SNAPSHOTS_DIR = REPO_ROOT / "data" / "snapshots"
OUT_DIR = REPO_ROOT / "artifacts" / "low_risk_sleeve_shadow"

LIVE = "LIVE"
REPLAY = "REPLAY"

METHOD = {
    "pool": "eligible names in rankings.csv, excluding production Top-30 (actionable_rank 1..30)",
    "exclusion": f"20-day annualised volatility below {PINNED_VOL_20D:.0%} (price pinned near a deal price, or stale)",
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

# Second shadow, LVB (added 2026-11-16 as `vol_beta`): same pool, exclusion and basket, scored on vol_60d and
# beta_120d only. It has its own schema and ledger so the base record is never rewritten.
METHOD_LVB = dict(METHOD, measures=["vol_60d", "beta_120d"])

PRE_REGISTRATION_LVB = [
    "Pre-registered 2026-11-16, before the first capture. Variant `vol_beta`: the base sleeve without the "
    "max_gain_21d measure.",
    "- Why: in the 2026-11-16 hold-out sensitivity test (77 months, net 50 bps, in-sample) max_gain_21d was the "
    "weak measure (alone: alpha +0.28 pt/month, t 0.65). vol+beta: alpha +1.16 (t 2.95), beta 0.37, Sharpe 1.27, "
    "max drawdown -12.6%; base: alpha +0.81 (t 2.07), beta 0.40, Sharpe 0.99, max drawdown -15.4%. The variant "
    "was chosen after seeing that table, so its backtest edge over the base is not evidence.",
    "- Same readings and thresholds as the base sleeve: CAPM alpha on non-overlapping 21-day windows, the 50/50 "
    "Top-30 + sleeve Sharpe, and the paired difference against the base sleeve; no reading below "
    f"{MIN_WINDOWS_FOR_READING} non-overlapping LIVE windows.",
    "- Nothing here changes the model or the account.",
]

# Third and fourth readings (added 2026-11-17): the production Top-20 (first 20 names of the captured,
# rank-ordered production Top-30) and a 50/50 mix of that Top-20 with the sleeve. No new capture: both
# are derived from fields every capture already records. Only captures dated after the registration
# date count, so nothing seen before registration is evidence.
TOP20_N = 20
MIN_PRICED_TOP20 = 17  # same ~85% coverage floor as MIN_PRICED for 30 names
TOP20_REGISTERED = "2026-11-17"

PRE_REGISTRATION_TOP20 = [
    f"Pre-registered {TOP20_REGISTERED}. Readings `top20` (production Top-20 = actionable_rank 1..20) and "
    "`combo20` (50/50 Top-20 + sleeve), derived from the existing captures. Only captures dated after "
    f"{TOP20_REGISTERED} count.",
    "- Why: on the v9 rebuild (82-84 months, in-sample, $1M per-trade costs) Top-20 minus Top-30 was "
    "+0.40 pt/month (t 1.53) and the 50/50 Top-20 + sleeve had Sharpe 1.09 vs 0.97 for 50/50 Top-30, "
    "CAPM alpha +1.16 pt/month (t 3.33), beta 0.67. Top-20 was chosen after a size sweep (5..100) and its "
    "edge came from ranks 11-20 alone (ranks 1-10 were no better than 21-30), so the backtest is not evidence.",
    "- Primary readings: Top-20 minus Top-30 (paired) on non-overlapping 21-trading-day windows; 50/50 Top-20 "
    "Sharpe against 50/50 Top-30 and Top-30, and its CAPM alpha against XBI.",
    "- Secondary: the same on non-overlapping 5-day windows; 50/50 Top-20 minus 50/50 Top-30 (paired).",
    f"- No reading below {MIN_WINDOWS_FOR_READING} non-overlapping LIVE windows at a horizon. "
    "Gross of costs, like the other readings here.",
    "- Nothing here changes the model, the portfolio size or the account. Adopting either is a governed "
    "model change with an out-of-sample clock reset.",
]

# Fifth reading (added 2026-11-17): production ranks 1-10 against ranks 11-30, paired, at longer horizons.
# Same source (the captured rank-ordered Top-30) and the same registration cutoff as the Top-20 readings.
RANK_SPLIT_HORIZONS = (21, 63, 126, 252)  # ~1, 3, 6, 12 months
MIN_PRICED_TOP10 = 8

PRE_REGISTRATION_RANK_SPLIT = [
    f"Pre-registered {TOP20_REGISTERED}. Reading `rank_split`: production ranks 1-10 minus ranks 11-30, "
    "equal weight, paired, from the existing captures. Only captures dated after "
    f"{TOP20_REGISTERED} count.",
    "- Hypothesis: the most crowded top picks (ranks 1-10: median 9 tier-1 managers, highest 12-month "
    "run-up) trail ranks 11-30, and the gap grows with holding period.",
    "- Why: on the v9 rebuild (in-sample, gross) ranks 1-10 minus ranks 11-30 was -0.4 / -1.9 / -3.5 / "
    "-9.6% at 1 / 3 / 6 / 12 months (12-month Newey-West t -1.3; negative in both halves and 4 of 6 start "
    "years, but the 1-6 month sign flips between halves). Crowding variants of the selector did not beat "
    "the current model, so this is watched, not acted on.",
    "- Primary reading: the 252- and 126-trading-day differences on non-overlapping windows. Secondary: "
    "63 and 21 days. The 252-day reading needs about 20 years of LIVE captures to reach "
    f"{MIN_WINDOWS_FOR_READING} windows, so read its sign and size at each annual review instead, alongside "
    "the shorter horizons, and treat it as descriptive.",
    "- Nothing here changes the model or the account.",
]

# Sixth reading, LVB standalone (added 2026-11-17 as `vol_beta_full`): the LVB sleeve drawn from the full eligible
# universe (production Top-30 names allowed) as a standalone strategy. Own schema and ledger.
METHOD_LVB_FULL = dict(
    METHOD_LVB, pool="eligible names in rankings.csv, including the production Top-30 (standalone sleeve)"
)

PRE_REGISTRATION_LVB_FULL = [
    "Pre-registered 2026-11-17, before the first capture. Variant `vol_beta_full`: the vol_beta sleeve "
    "(vol_60d + beta_120d, 30 names, equal weight, same pinned-price exclusion) drawn from the full eligible "
    "universe, so production Top-30 names may be held. Tracked as a candidate standalone strategy.",
    "- Why: on the v9 rebuild (82 months, in-sample, survivor-only, 50 bp per unit of one-way turnover) the "
    "full-universe sleeve returned 22.0%/yr, Sharpe 1.35, max drawdown -12.2%, beta 0.35, CAPM alpha "
    "+1.40 pt/month (t 3.42); 2024-26 alpha +1.74 (t 2.83). It overlapped the Top-30 by 2.5 names on average "
    "and differed from the ex-Top-30 sleeve by +0.01 pt/month (t 0.09). The measures and windows were chosen "
    "on the same data, so the backtest is not evidence.",
    "- Judged on risk-adjusted terms, not on excess return over XBI: by design it trails XBI in strong "
    "rallies (worst 12 months vs XBI -48% in the 2020 rally).",
    "- Primary reading: CAPM alpha against XBI on non-overlapping 21-trading-day windows, and Sharpe against "
    "XBI and the production Top-30. Secondary: the same at 5 days; the paired difference against the "
    "`vol_beta` (ex-Top-30) sleeve.",
    "- The `combo` reading here is a 50/50 mix that can hold a name in both halves; the proposal's mix uses "
    "the `vol_beta` ledger instead.",
    f"- No reading below {MIN_WINDOWS_FOR_READING} non-overlapping LIVE windows at a horizon. Gross of costs.",
    "- Nothing here changes the model or the account. Running the sleeve alone with real money is a separate "
    "governed decision with its own benchmark and mandate.",
]

VARIANTS: Dict[str, Dict[str, Any]] = {
    "base": {"schema": SCHEMA_VERSION, "method": METHOD, "out_dir": OUT_DIR, "pre_registration": PRE_REGISTRATION},
    "lvb": {
        "schema": "low_risk_sleeve_shadow.vol_beta.v1",
        "method": METHOD_LVB,
        "out_dir": REPO_ROOT / "artifacts" / "low_risk_sleeve_shadow_vol_beta",
        "pre_registration": PRE_REGISTRATION_LVB,
    },
    "lvb_full": {
        "schema": "low_risk_sleeve_shadow.vol_beta_full.v1",
        "method": METHOD_LVB_FULL,
        "out_dir": REPO_ROOT / "artifacts" / "low_risk_sleeve_shadow_vol_beta_full",
        "pre_registration": PRE_REGISTRATION_LVB_FULL,
        "exclude_production": False,
    },
}


# Names the LVB variants were registered under (2026-11-16/17). Kept so older commands still work;
# they resolve to the same spec, schema and ledger.
VARIANT_ALIASES: Dict[str, str] = {"vol_beta": "lvb", "vol_beta_full": "lvb_full"}


def resolve_variant(variant: str) -> str:
    """Canonical variant name (``vol_beta`` -> ``lvb``); unknown names raise KeyError."""
    name = VARIANT_ALIASES.get(variant, variant)
    if name not in VARIANTS:
        raise KeyError(variant)
    return name


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
    out: Dict[str, Optional[float]] = {"vol_60d": None, "vol_20d": None, "max_gain_21d": None, "beta_120d": None}

    pts = _closes_upto(series, as_of, VOL_WINDOW)
    logs = [math.log(b[1] / a[1]) for a, b in zip(pts, pts[1:])]
    if len(logs) >= MIN_VOL_RETURNS:
        out["vol_60d"] = statistics.pstdev(logs) * math.sqrt(252)
    if len(logs) >= PINNED_WINDOW:
        out["vol_20d"] = statistics.pstdev(logs[-PINNED_WINDOW:]) * math.sqrt(252)

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


def build_sleeve(
    rows: List[Dict[str, str]],
    prices: Dict[str, Dict[str, float]],
    as_of: str,
    measures: Optional[List[str]] = None,
    exclude_production: bool = True,
) -> Dict[str, Any]:
    """Production Top-30 and the low-risk sleeve for one snapshot.

    The sleeve is disjoint from the Top-30 unless ``exclude_production`` is False (standalone sleeve).
    """
    keys = list(measures) if measures is not None else METHOD["measures"]
    production = production_top(rows)
    held = set(production) if exclude_production else set()
    xbi = prices.get(BENCHMARK, {})
    risk: Dict[str, Dict[str, Optional[float]]] = {}
    pinned: List[str] = []
    for row in rows:
        ticker = (row.get("ticker") or "").strip()
        if not ticker or ticker in held or _num(row.get("eligible")) != 1.0:
            continue
        m = risk_measures(prices.get(ticker, {}), xbi, as_of)
        if m["vol_60d"] is None:
            continue
        # A price pinned near a pending cash-deal price (or a stale feed) reads as "safe";
        # it is not a low-risk holding, so it is excluded and recorded.
        if m["vol_20d"] is not None and m["vol_20d"] < PINNED_VOL_20D:
            pinned.append(ticker)
            continue
        risk[ticker] = m

    # Lower risk = higher percentile: rank the negated measure.
    ranks = {k: _pct_ranks({t: -m[k] for t, m in risk.items() if m.get(k) is not None}) for k in keys}
    scores = {}
    for t in risk:
        got = [ranks[k][t] for k in keys if t in ranks[k]]
        scores[t] = sum(got) / len(got)
    sleeve = [t for t, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:SLEEVE_N]]
    price_asof = max((d for d in xbi if d <= as_of), default=None)
    return {
        "production": production,
        "sleeve": sleeve,
        "measures": {t: dict(risk[t], score=round(scores[t], 6)) for t in sleeve},
        "n_scored": len(scores),
        "excluded_pinned": sorted(pinned),
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
    out_dir: Optional[Path] = None,
    price_csv: Optional[Path] = None,
    today: Optional[str] = None,
    mode: Optional[str] = None,
    variant: str = "base",
) -> Dict[str, Any]:
    """Record the sleeve and the production Top-30 for ``as_of_date``. Never rewrites a date."""
    variant = resolve_variant(variant)
    spec = VARIANTS[variant]
    out_dir = Path(out_dir) if out_dir is not None else spec["out_dir"]
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
    result = build_sleeve(
        rows,
        prices,
        as_of_date,
        measures=spec["method"]["measures"],
        exclude_production=spec.get("exclude_production", True),
    )

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
        "schema": spec["schema"],
        "as_of_date": as_of_date,
        "capture_mode": mode,
        "captured_on": today,
        "rankings_sha256": _sha256(rankings),
        "price_source": price_csv.name,
        "price_asof": price_asof,
        "method": spec["method"],
        "n_scored": result["n_scored"],
        "excluded_pinned": result["excluded_pinned"],
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


def _top20_block(captures, mode, prices, calendar, h) -> Dict[str, Any]:
    """Top-20 and 50/50 Top-20 + sleeve readings, from captures dated after TOP20_REGISTERED only."""
    windows = []
    for c in captures:
        d = c["as_of_date"]
        if c["capture_mode"] != mode or d <= TOP20_REGISTERED:
            continue
        top20 = c["production"][:TOP20_N]
        rets = [r for r in (_forward_return(prices.get(t, {}), calendar, d, h) for t in top20) if r is not None]
        t20 = sum(rets) / len(rets) if len(rets) >= MIN_PRICED_TOP20 else None
        sl, _ = _basket_return(c["sleeve"], prices, calendar, d, h)
        pr, _ = _basket_return(c["production"], prices, calendar, d, h)
        bx = _forward_return(prices.get(BENCHMARK, {}), calendar, d, h)
        if None in (t20, sl, pr, bx):
            continue
        windows.append(
            {
                "as_of_date": d,
                "top20_pct": round(t20 * 100, 3),
                "production_pct": round(pr * 100, 3),
                "sleeve_pct": round(sl * 100, 3),
                "combo20_pct": round((t20 + sl) / 2 * 100, 3),
                "combo_pct": round((pr + sl) / 2 * 100, 3),
                "xbi_pct": round(bx * 100, 3),
                "n_priced_top20": len(rets),
                "_r": (t20, pr, sl, bx),
            }
        )
    ind = _non_overlapping(windows, calendar, h)
    t20 = [w["_r"][0] for w in ind]
    pr = [w["_r"][1] for w in ind]
    sl = [w["_r"][2] for w in ind]
    bx = [w["_r"][3] for w in ind]
    c20 = [(a + b) / 2 for a, b in zip(t20, sl)]
    c30 = [(a + b) / 2 for a, b in zip(pr, sl)]
    stats = {
        "top20_minus_production": _series_stats([a - b for a, b in zip(t20, pr)]),
        "top20_minus_xbi": _series_stats([a - b for a, b in zip(t20, bx)]),
        "combo20_minus_xbi": _series_stats([a - b for a, b in zip(c20, bx)]),
        "combo20_minus_combo": _series_stats([a - b for a, b in zip(c20, c30)]),
        "combo20_capm": _capm(c20, bx),
        "sharpe_per_window": {
            "top20": _sharpe(t20),
            "production": _sharpe(pr),
            "combo20": _sharpe(c20),
            "combo": _sharpe(c30),
            "xbi": _sharpe(bx),
        },
        "registered": TOP20_REGISTERED,
        "n_matured_captures": len(windows),
    }
    for w in windows:
        w.pop("_r", None)
    return {"stats": stats, "windows": ind}


def _rank_split_block(captures, mode, prices, calendar, h) -> Dict[str, Any]:
    """Ranks 1-10 minus ranks 11-30 of the captured production order, captures after TOP20_REGISTERED only."""
    windows = []
    for c in captures:
        d = c["as_of_date"]
        if c["capture_mode"] != mode or d <= TOP20_REGISTERED:
            continue
        legs = []
        for names, floor in ((c["production"][:10], MIN_PRICED_TOP10), (c["production"][10:30], MIN_PRICED_TOP20)):
            rets = [r for r in (_forward_return(prices.get(t, {}), calendar, d, h) for t in names) if r is not None]
            legs.append(sum(rets) / len(rets) if len(rets) >= floor else None)
        bx = _forward_return(prices.get(BENCHMARK, {}), calendar, d, h)
        if None in legs or bx is None:
            continue
        top10, rest = legs
        windows.append(
            {
                "as_of_date": d,
                "top10_pct": round(top10 * 100, 3),
                "ranks11_30_pct": round(rest * 100, 3),
                "top10_minus_ranks11_30_pct": round((top10 - rest) * 100, 3),
                "xbi_pct": round(bx * 100, 3),
                "_r": (top10, rest, bx),
            }
        )
    ind = _non_overlapping(windows, calendar, h)
    t10 = [w["_r"][0] for w in ind]
    rest = [w["_r"][1] for w in ind]
    bx = [w["_r"][2] for w in ind]
    stats = {
        "top10_minus_ranks11_30": _series_stats([a - b for a, b in zip(t10, rest)]),
        "top10_minus_xbi": _series_stats([a - b for a, b in zip(t10, bx)]),
        "ranks11_30_minus_xbi": _series_stats([a - b for a, b in zip(rest, bx)]),
        "registered": TOP20_REGISTERED,
        "n_matured_captures": len(windows),
    }
    for w in windows:
        w.pop("_r", None)
    return {"stats": stats, "windows": ind}


def summarize(
    *,
    out_dir: Optional[Path] = None,
    price_csv: Optional[Path] = None,
    horizons: Tuple[int, ...] = HORIZONS,
    write: bool = True,
    variant: str = "base",
) -> Dict[str, Any]:
    variant = resolve_variant(variant)
    spec = VARIANTS[variant]
    out_dir = Path(out_dir) if out_dir is not None else spec["out_dir"]
    price_csv = Path(price_csv) if price_csv is not None else default_price_csv()
    captures = load_captures(out_dir)
    result: Dict[str, Any] = {
        "schema": spec["schema"],
        "variant": variant,
        "price_source": price_csv.name,
        "method": spec["method"],
        "pre_registration": spec["pre_registration"],
        "top20_pre_registration": PRE_REGISTRATION_TOP20,
        "rank_split_pre_registration": PRE_REGISTRATION_RANK_SPLIT,
        "rank_split": {},
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
            mode_out[f"h{h}d"] = {
                "stats": stats,
                "windows": ind,
                "top20": _top20_block(captures, mode, prices, calendar, h),
            }
        result["by_mode"][mode] = mode_out
        result["rank_split"][mode] = {
            f"h{h}d": _rank_split_block(captures, mode, prices, calendar, h) for h in RANK_SPLIT_HORIZONS
        }
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


_LVB_LABEL = {"lvb": ", LVB", "lvb_full": ", LVB standalone"}


def format_summary_md(result: Dict[str, Any]) -> str:
    lines = [
        "# Low-risk sleeve shadow — forward record"
        + (
            ""
            if result.get("variant", "base") == "base"
            else f" (variant `{result['variant']}`{_LVB_LABEL.get(result['variant'], '')})"
        ),
        "",
        f"Sleeve = the {SLEEVE_N} lowest-risk names; pool: {result.get('method', METHOD)['pool']}; measures: "
        f"{', '.join(result.get('method', METHOD)['measures'])}; equal weight; names whose 20-day "
        "volatility is below 10% (price pinned near a deal price, or stale) are excluded. Compared with XBI "
        "and the production Top-30 on equal-weight forward returns, gross of costs. Non-overlapping windows only.",
        "",
        *result.get("pre_registration", PRE_REGISTRATION),
        "",
        *result.get("top20_pre_registration", []),
        "",
        *result.get("rank_split_pre_registration", []),
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
        lines += [
            f"### {mode} — Top-20 and 50/50 Top-20 + sleeve (captures after {TOP20_REGISTERED} only)",
            "",
            "| Horizon | Windows | Top-20 − Top-30 | t | Top-20 − XBI | 50/50 Top-20 − XBI | 50/50 Top-20 − 50/50 Top-30 | t | "
            "50/50 Top-20 alpha | t | Beta | Sharpe Top-20 / Top-30 / 50/50 Top-20 / 50/50 Top-30 / XBI | Reading |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for key, data in block.items():
            t = data.get("top20")
            if not t:
                continue
            s = t["stats"]
            a, sh = s["combo20_capm"], s["sharpe_per_window"]
            sharpe = " / ".join(_fmt(sh[k], ".2f") for k in ("top20", "production", "combo20", "combo", "xbi"))
            lines.append(
                f"| {key[1:]} | {s['top20_minus_production']['n']} | "
                f"{_fmt(s['top20_minus_production']['mean_pct'], '+.2f', ' pts')} | "
                f"{_fmt(s['top20_minus_production']['t_stat'], '+.2f')} | "
                f"{_fmt(s['top20_minus_xbi']['mean_pct'], '+.2f', ' pts')} | "
                f"{_fmt(s['combo20_minus_xbi']['mean_pct'], '+.2f', ' pts')} | "
                f"{_fmt(s['combo20_minus_combo']['mean_pct'], '+.2f', ' pts')} | "
                f"{_fmt(s['combo20_minus_combo']['t_stat'], '+.2f')} | {_fmt(a['alpha_pct'], '+.2f', ' pts')} | "
                f"{_fmt(a['alpha_t'], '+.2f')} | {_fmt(a['beta'], '.2f')} | {sharpe} | "
                f"{s['top20_minus_production']['reading']} |"
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
    for mode in (LIVE, REPLAY):
        block = result.get("rank_split", {}).get(mode)
        if not block:
            continue
        lines += [
            f"## {mode} — Ranks 1-10 vs 11-30 (captures after {TOP20_REGISTERED} only)",
            "",
            "| Horizon | Windows | Ranks 1-10 − 11-30 | t | Ranks 1-10 − XBI | Ranks 11-30 − XBI | Reading |",
            "|---|---|---|---|---|---|---|",
        ]
        for key, data in block.items():
            s = data["stats"]
            d = s["top10_minus_ranks11_30"]
            lines.append(
                f"| {key[1:]} | {d['n']} | {_fmt(d['mean_pct'], '+.2f', ' pts')} | {_fmt(d['t_stat'], '+.2f')} | "
                f"{_fmt(s['top10_minus_xbi']['mean_pct'], '+.2f', ' pts')} | "
                f"{_fmt(s['ranks11_30_minus_xbi']['mean_pct'], '+.2f', ' pts')} | {d['reading']} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--as-of-date")
    parser.add_argument("--capture", action="store_true", help="Record the sleeve for --as-of-date")
    parser.add_argument("--summary", action="store_true", help="Recompute the forward-return comparison")
    parser.add_argument("--variant", choices=sorted(VARIANTS) + sorted(VARIANT_ALIASES), default="base")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--snapshots-dir", type=Path, default=SNAPSHOTS_DIR)
    parser.add_argument("--price-csv", type=Path, default=None)
    args = parser.parse_args()

    if args.capture:
        if not args.as_of_date:
            parser.error("--capture needs --as-of-date")
        res = capture(
            args.as_of_date,
            snapshots_dir=args.snapshots_dir,
            out_dir=args.out_dir,
            price_csv=args.price_csv,
            variant=args.variant,
        )
        print(json.dumps(res))
    if args.summary or args.capture:
        print(format_summary_md(summarize(out_dir=args.out_dir, price_csv=args.price_csv, variant=args.variant)))
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
