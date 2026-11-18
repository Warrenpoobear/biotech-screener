"""50/50 Top-30 + LVB mix from the LVB backtest outputs.

Research only (Tier 2). Combines monthly net returns already written by
lvb_universe_backtest.py (Top-30 half) and, optionally, lvb_survivorship_check.py
(survivorship-corrected LVB half), rebalanced to 50/50 every month. Reports the
mix, both halves, XBI and the S&P 500 (SPY total return from the --etf-prices file).

Usage:
  python scripts/research/lvb_5050_mix.py --backtest <grid.json> \
      [--survivorship <lvb_surv.json>] --etf-prices <closeadj.csv> [--start 2020-03-31]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lvb_universe_backtest import stats  # noqa: E402

from tools.low_risk_sleeve_shadow import load_prices  # noqa: E402


def _rows(rows, start):
    return {r["d0"]: r for r in rows if r["d0"] >= start}


def _alpha(y, x):
    X = np.column_stack([np.ones(len(y)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    se = math.sqrt((e @ e / (len(y) - 2)) * np.linalg.inv(X.T @ X)[0, 0])
    return round(b[0] * 100, 2), round(b[0] / se, 2)


def _rolling12(series, bench):
    rel = []
    for i in range(len(series) - 11):
        a = np.prod([1 + v for v in series[i : i + 12]])
        b = np.prod([1 + v for v in bench[i : i + 12]])
        rel.append(a / b - 1)
    return min(rel), sum(r < 0 for r in rel) / len(rel)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backtest", required=True, help="lvb_universe_backtest.py output")
    ap.add_argument("--survivorship", default=None, help="lvb_survivorship_check.py output (optional)")
    ap.add_argument("--etf-prices", required=True, help="ticker,date,close CSV with SPY and XLV")
    ap.add_argument("--start", default="2020-03-31")
    args = ap.parse_args()

    bt = json.load(open(args.backtest, encoding="utf-8"))["monthly"]
    top = _rows(bt["top30"], args.start)
    lvb = _rows(bt["full"], args.start)
    label = "LVB (survivors)"
    if args.survivorship:
        lvb = _rows(json.load(open(args.survivorship, encoding="utf-8"))["monthly"]["delisted:full"], args.start)
        label = "LVB (survivorship-corrected)"
    dates = sorted(set(top) & set(lvb))
    etf = load_prices(Path(args.etf_prices))

    def ret(t, a, b):
        s = etf[t]
        return s[max(k for k in s if k <= b)] / s[max(k for k in s if k <= a)] - 1

    xbi = [top[d]["xbi"] for d in dates]
    spy = np.array([ret("SPY", d, top[d]["d1"]) for d in dates])
    xlv = np.array([ret("XLV", d, top[d]["d1"]) for d in dates])
    T = [top[d]["net"] for d in dates]
    L = [lvb[d]["net"] for d in dates]
    M = [0.5 * a + 0.5 * b for a, b in zip(T, L)]

    out = {"months": len(dates), "first": dates[0], "last": top[dates[-1]]["d1"], "portfolios": {}}
    for name, s in (("Top-30", T), (label, L), ("50/50", M), ("S&P 500 (SPY)", list(spy))):
        x = stats(s, xbi)
        worst12, behind = _rolling12(s, xbi)
        out["portfolios"][name] = {
            "ann_return": round(x["ann_return"], 4),
            "ann_vol": round(x["ann_vol"], 4),
            "sharpe": round(x["sharpe"], 2),
            "max_drawdown": round(x["max_drawdown"], 4),
            "beta_xbi": round(x["beta_xbi"], 2),
            "alpha_xbi": [round(x["alpha_xbi_pt_month"], 2), round(x["alpha_xbi_t"], 2)],
            "excess_xbi": [round(x["excess_xbi_pt_month"], 2), round(x["excess_xbi_t"], 2)],
            "worst_12m_vs_xbi": round(worst12, 4),
            "share_12m_behind_xbi": round(behind, 2),
            "alpha_spy": _alpha(np.array(s), spy),
            "alpha_xlv": _alpha(np.array(s), xlv),
        }
    d = np.array(M) - np.array(T)
    out["mix_minus_top30"] = [round(d.mean() * 100, 2), round(d.mean() / (d.std(ddof=1) / math.sqrt(len(d))), 2)]
    out["corr_excess_halves"] = round(float(np.corrcoef(np.array(T) - xbi, np.array(L) - xbi)[0, 1]), 2)
    years = {}
    for dd, a, b, c, x in zip(dates, T, L, M, xbi):
        y = years.setdefault(top[dd]["d1"][:4], [1.0, 1.0, 1.0, 1.0])
        for i, v in enumerate((a, b, c, x)):
            y[i] *= 1 + v
    out["calendar_years_pct"] = {
        y: dict(zip(("Top-30", label, "50/50", "XBI"), [round((g - 1) * 100, 1) for g in v]))
        for y, v in sorted(years.items())
    }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
