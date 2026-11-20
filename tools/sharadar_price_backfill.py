#!/usr/bin/env python3
"""One-time cutover: upsert Sharadar history into price_history.csv.

Re-bases the raw price series on Sharadar official closes from --start (default
2020-01-01) through --through for every ticker in price_history.csv (plus XBI),
and sets market_data.json market_cap/enterprise_value from Sharadar DAILY.
Rows before --start are untouched, so the first-price dates that
ipo_dates.json derives from price history do not move.

Default is a DRY RUN against a temporary copy: it reports how many closes would
be revised and by how much, and writes nothing to production_data. Use
--apply to write (atomically). Afterwards the daily run regenerates
price_history_split_adj.csv and reports the retroactive change.

    python3 tools/sharadar_price_backfill.py                      # dry run
    python3 tools/sharadar_price_backfill.py --apply              # cutover
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.sharadar_prices import last_dates, refresh_prices_from_sharadar, update_market_caps_from_daily  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "production_data")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--through", default=date.today().isoformat())
    p.add_argument("--apply", action="store_true", help="write to --data-dir (default: dry run on a temp copy)")
    args = p.parse_args(argv)

    src_prices = args.data_dir / "price_history.csv"
    src_md = args.data_dir / "market_data.json"
    if args.apply:
        prices, md, tmp = src_prices, src_md, None
    else:
        tmp = Path(tempfile.mkdtemp(prefix="sharadar_backfill_"))
        prices, md = tmp / "price_history.csv", tmp / "market_data.json"
        shutil.copy(src_prices, prices)
        if src_md.exists():
            shutil.copy(src_md, md)

    have = last_dates(prices)
    tickers = sorted(t for t, d in have.items() if d >= args.start and not t.startswith("_"))
    if "XBI" not in tickers:
        tickers.append("XBI")
    print(f"{'APPLY' if args.apply else 'DRY RUN'}: {len(tickers)} tickers, {args.start} .. {args.through}")

    # A window as long as the whole range turns the "trailing window" into a backfill.
    span = (date.fromisoformat(args.through) - date.fromisoformat(args.start)).days
    stats = refresh_prices_from_sharadar(prices, args.through, tickers, lookback_days=span, backfill_start=args.start)
    print(json.dumps({k: v for k, v in stats.items() if k not in ("revised_tickers", "appended_tickers")}, indent=1))
    print(
        f"tickers revised: {len(stats.get('revised_tickers') or [])}; appended: {len(stats.get('appended_tickers') or [])}"
    )
    if md.exists():
        mc = update_market_caps_from_daily(md, args.through)
        print("market caps:", json.dumps(mc))
    if tmp is not None:
        print(f"(dry run output left in {tmp})")
    return 0 if stats.get("status") in ("OK", "PARTIAL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
