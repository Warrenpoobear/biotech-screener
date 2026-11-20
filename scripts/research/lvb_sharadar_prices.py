"""Fetch Sharadar dividend-adjusted closes (closeadj) for the LVB backtest.

Research only (Tier 2). Writes a local CSV with columns ticker,date,close where
`close` is Sharadar `closeadj` (split AND dividend adjusted, i.e. total return).
The production price_history_split_adj.csv is split-adjusted only, so it leaves
out dividends; LVB holds dividend payers, so this is the fair comparison.

Stocks come from SEP, ETFs (XBI, SPY, XLV, IWM) from SFP. Tickers = every name
in the snapshot set's rankings.csv files. Needs SHARADAR_API_KEY (environment or
the repo's .env). The output is licensed vendor data: keep it out of git.

Usage:
  python scripts/research/lvb_sharadar_prices.py \
      --snapshot-dir data/snapshots_pit_v9 --out <local path>/sharadar_closeadj.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from common.sharadar_client import API_KEY_ENV_VAR, SharadarClient  # noqa: E402

ETFS = ["XBI", "SPY", "XLV", "IWM"]


def _load_key() -> None:
    if os.environ.get(API_KEY_ENV_VAR):
        return
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition("=")
            if k.strip() == API_KEY_ENV_VAR and v.strip():
                os.environ[API_KEY_ENV_VAR] = v.strip().strip('"').strip("'")
                return


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot-dir", default="data/snapshots_pit_v9")
    ap.add_argument("--start", default="2019-06-01")
    ap.add_argument("--end", default="2026-12-31")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    snap = (REPO / args.snapshot_dir).resolve()
    tickers = set()
    for f in snap.glob("20*/rankings.csv"):
        with open(f, encoding="utf-8", newline="") as fh:
            tickers.update((r.get("ticker") or "").strip().upper() for r in csv.DictReader(fh))
    tickers.discard("")
    stocks = sorted(tickers - set(ETFS))

    _load_key()
    client = SharadarClient()
    rng = (args.start, args.end)
    fields = ["ticker", "date", "closeadj"]
    rows = client.fetch("SEP", tickers=stocks, fields=fields, date_range=rng)
    rows += client.fetch("SFP", tickers=ETFS, fields=fields, date_range=rng)

    out_rows = sorted(
        (r["ticker"], str(r["date"])[:10], float(r["closeadj"])) for r in rows if r.get("closeadj") not in (None, "")
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        w.writerow(["ticker", "date", "close"])
        for t, d, c in out_rows:
            w.writerow([t, d, f"{c:.6f}"])
    got = {t for t, _, _ in out_rows}
    print(f"tickers requested {len(stocks) + len(ETFS)}, returned {len(got)}, rows {len(out_rows)}")
    missing = sorted((set(stocks) | set(ETFS)) - got)
    print(f"missing ({len(missing)}): {' '.join(missing[:60])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
