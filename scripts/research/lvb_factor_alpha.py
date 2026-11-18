"""Alpha of the LVB backtest returns against ETF benchmarks (XBI, SPY, XLV, IWM).

Research only (Tier 2). Reads the monthly returns written by
scripts/research/lvb_universe_backtest.py and regresses each LVB variant on ETF
total returns over the same windows: XBI, SPY, XLV alone; SPY + XLV;
SPY + XLV + IWM + XBI; and SPY + XLV for windows starting 2024 or later.

ETF prices come from yfinance adjusted closes (live fetch, research copy only;
never used in tests or production). Results therefore depend on the fetch date.

Usage:
  python scripts/research/lvb_factor_alpha.py \
      --backtest artifacts/research/lvb_universe_backtest.json
"""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np
import yfinance as yf

ETFS = ["SPY", "XLV", "IWM", "XBI"]
MODELS = [
    ("XBI", ["XBI"]),
    ("SPY", ["SPY"]),
    ("XLV", ["XLV"]),
    ("SPY+XLV", ["SPY", "XLV"]),
    ("SPY+XLV+IWM+XBI", ["SPY", "XLV", "IWM", "XBI"]),
]
LATE_START = "2023-12-29"


def _ols(y, factors):
    X = np.column_stack([np.ones(len(y))] + factors)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    s2 = e @ e / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return b, se, r2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backtest", required=True)
    ap.add_argument("--variants", default="ex_top30,full")
    ap.add_argument(
        "--etf-prices",
        default=None,
        help="ticker,date,close CSV (e.g. Sharadar closeadj from lvb_sharadar_prices.py); default: yfinance",
    )
    args = ap.parse_args()

    monthly = json.load(open(args.backtest, encoding="utf-8"))["monthly"]
    first = min(r["d0"] for rows in monthly.values() for r in rows)
    last = max(r["d1"] for rows in monthly.values() for r in rows)

    if args.etf_prices:
        series = {t: {} for t in ETFS}
        with open(args.etf_prices, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row["ticker"] in series:
                    series[row["ticker"]][row["date"]] = float(row["close"])

        def ret(t, a, b):
            s = series[t]
            pa = s[max(d for d in s if d <= a)]
            pb = s[max(d for d in s if d <= b)]
            return pb / pa - 1

    else:
        end = np.datetime64(last) + np.timedelta64(5, "D")
        px = yf.download(ETFS, start=first[:8] + "01", end=str(end), auto_adjust=True, progress=False)["Close"]

        def ret(t, a, b):
            s = px[t]
            return float(s[:b].iloc[-1] / s[:a].iloc[-1] - 1)

    out = {}
    for v in args.variants.split(","):
        rows = monthly[v]
        f = {t: np.array([ret(t, r["d0"], r["d1"]) for r in rows]) for t in ETFS}
        y = np.array([r["net"] for r in rows])
        res = {}
        for name, keys in MODELS:
            b, se, r2 = _ols(y, [f[k] for k in keys])
            res[name] = {
                "alpha_pt_month": round(b[0] * 100, 2),
                "t": round(b[0] / se[0], 2),
                "betas": dict(zip(keys, [round(x, 2) for x in b[1:]])),
                "r2": round(r2, 2),
            }
        late = [i for i, r in enumerate(rows) if r["d0"] >= LATE_START]
        b, se, _ = _ols(y[late], [f["SPY"][late], f["XLV"][late]])
        res["SPY+XLV_2024_on"] = {
            "alpha_pt_month": round(b[0] * 100, 2),
            "t": round(b[0] / se[0], 2),
            "months": len(late),
        }
        out[v] = res
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
