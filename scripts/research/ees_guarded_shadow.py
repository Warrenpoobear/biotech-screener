"""
EES-Guarded Top-30 Shadow Basket
VALIDATION_INFRASTRUCTURE / FAILURE_MODE_SHADOW_GUARDS / NO_MODEL_CHANGE

Logs two baskets each run (raw Top-30 and EES-guarded Top-30 with 31-60 fill).
A separate fill-returns pass computes forward deltas once prices are available.

Usage:
    # Log today's baskets
    python3 scripts/research/ees_guarded_shadow.py --snap-date YYYY-MM-DD

    # Fill forward returns for all logged baskets (run after prices update)
    python3 scripts/research/ees_guarded_shadow.py --fill-returns

Promotion gate (from post-mortem):
    20+ completed windows, positive raw-vs-guarded delta,
    no major degradation in normal/rally weeks,
    helps >=60% of drawdown/failure windows.
"""

import argparse
import glob
import json
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
OUTDIR = REPO / "artifacts/failure_mode_guards"
OUTDIR.mkdir(parents=True, exist_ok=True)

BASKET_LOG = OUTDIR / "ees_guarded_baskets.jsonl"
RETURNS_LOG = OUTDIR / "ees_guarded_returns.jsonl"
PRICE_FILE = REPO / "production_data/price_history_split_adj.csv"


def find_latest_snap(target: str) -> str:
    dirs = sorted(glob.glob(str(REPO / "data/snapshots/????-??-??")))
    eligible = [Path(d).name for d in dirs if len(Path(d).name) == 10 and Path(d).name <= target]
    return eligible[-1] if eligible else None


def load_snap(snap_date: str) -> pd.DataFrame:
    path = REPO / f"data/snapshots/{snap_date}/rankings.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df["ticker"] = df["ticker"].astype(str)
    return df


def build_ees_guarded_basket(df_snap: pd.DataFrame) -> dict:
    """
    Build EES-guarded Top-30:
      - Start with Top-30 names
      - Remove ees_v3_gate == False (EES-fail)
      - Fill gaps from ranks 31-60 (skipping EES-fail names)
    Returns dict with raw_top30, ees_excluded, guarded_top30, fill_names.
    """
    has_ees = "ees_v3_gate" in df_snap.columns

    top30 = df_snap[df_snap["actionable_rank"] <= 30].sort_values("actionable_rank")
    bench = df_snap[(df_snap["actionable_rank"] > 30) & (df_snap["actionable_rank"] <= 60)].sort_values(
        "actionable_rank"
    )

    raw_top30 = top30["ticker"].tolist()

    if not has_ees:
        return {
            "raw_top30": raw_top30,
            "ees_excluded": [],
            "guarded_top30": raw_top30,
            "fill_names": [],
            "has_ees_data": False,
        }

    def is_fail(v):
        return v is False or str(v).lower() in ("false", "0")

    excluded = [
        {
            "ticker": r["ticker"],
            "rank": int(r["actionable_rank"]),
            "ees_gate": str(r["ees_v3_gate"]),
            "catalyst_bucket": r.get("catalyst_bucket", "?"),
        }
        for _, r in top30.iterrows()
        if is_fail(r["ees_v3_gate"])
    ]
    excluded_set = {e["ticker"] for e in excluded}

    kept = [t for t in raw_top30 if t not in excluded_set]
    fills = []
    need = 30 - len(kept)

    for _, br in bench.iterrows():
        if need <= 0:
            break
        bt = br["ticker"]
        if bt in excluded_set:
            continue
        if is_fail(br["ees_v3_gate"]):
            continue
        kept.append(bt)
        fills.append(
            {
                "ticker": bt,
                "rank": int(br["actionable_rank"]),
                "ees_gate": str(br["ees_v3_gate"]),
                "catalyst_bucket": br.get("catalyst_bucket", "?"),
            }
        )
        need -= 1

    return {
        "raw_top30": raw_top30,
        "ees_excluded": excluded,
        "guarded_top30": kept,
        "fill_names": fills,
        "has_ees_data": True,
    }


def log_baskets(snap_date: str) -> None:
    actual_snap = find_latest_snap(snap_date)
    if not actual_snap:
        print(f"ERROR: No snapshot found at or before {snap_date}")
        return

    print(f"EES-Guarded Shadow — logging baskets for {snap_date} (snap: {actual_snap})")

    df_snap = load_snap(actual_snap)
    if df_snap.empty:
        print("ERROR: Could not load snapshot")
        return

    baskets = build_ees_guarded_basket(df_snap)
    regime = str(df_snap["regime_label"].iloc[0]) if "regime_label" in df_snap.columns else "UNKNOWN"
    # Log forward return window: snap_date to snap_date + 5 business days
    fwd_date = _add_business_days(snap_date, 5)

    record = {
        "snap_date": snap_date,
        "actual_snap": actual_snap,
        "fwd_date": fwd_date,
        "regime": regime,
        "n_excluded": len(baskets["ees_excluded"]),
        "n_filled": len(baskets["fill_names"]),
        "has_ees_data": baskets["has_ees_data"],
        "raw_top30": baskets["raw_top30"],
        "guarded_top30": baskets["guarded_top30"],
        "ees_excluded": baskets["ees_excluded"],
        "fill_names": baskets["fill_names"],
        "returns_filled": False,
        "classification": "VALIDATION_INFRASTRUCTURE/NO_MODEL_CHANGE",
    }

    # Check for duplicates before writing
    existing_dates = set()
    if BASKET_LOG.exists():
        with open(BASKET_LOG) as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_dates.add(json.loads(line).get("snap_date"))

    if snap_date in existing_dates:
        print(f"  Basket for {snap_date} already logged — skipping.")
        return

    with open(BASKET_LOG, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")

    print(f"  raw_top30: {len(baskets['raw_top30'])} names")
    print(f"  ees_excluded: {len(baskets['ees_excluded'])} names — {[e['ticker'] for e in baskets['ees_excluded']]}")
    print(f"  fill_names: {[e['ticker'] for e in baskets['fill_names']]}")
    print(f"  guarded_top30: {len(baskets['guarded_top30'])} names")
    print(f"  Forward window: {snap_date} → {fwd_date}")
    print(f"  Logged to {BASKET_LOG}")


def fill_returns() -> None:
    """Compute forward returns for all logged baskets with returns_filled=False."""
    if not BASKET_LOG.exists():
        print("No basket log found.")
        return

    print("Filling forward returns for logged EES-guarded baskets...")
    prices_raw = pd.read_csv(PRICE_FILE, parse_dates=["date"])
    prices = prices_raw.pivot(index="date", columns="ticker", values="close").sort_index()
    xbi = prices["XBI"].dropna() if "XBI" in prices.columns else None

    records = []
    with open(BASKET_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    updated = []
    results = []

    for rec in records:
        if rec.get("returns_filled"):
            updated.append(rec)
            continue

        snap = rec["snap_date"]
        fwd = rec["fwd_date"]

        # Check if fwd date is available in prices
        avail_dates = prices.index[prices.index >= pd.Timestamp(fwd)]
        if avail_dates.empty:
            print(f"  {snap}: fwd date {fwd} not yet in prices — skipping")
            updated.append(rec)
            continue

        def basket_xs(tickers):
            rets = []
            for t in tickers:
                if t not in prices.columns:
                    continue
                p = prices[t].dropna()
                s = p.loc[:snap].iloc[-1] if not p.loc[:snap].empty else None
                fv = p.loc[:fwd].iloc[-1] if not p.loc[:fwd].empty else None
                if s and fv and s > 0:
                    rets.append((fv - s) / s)
            return np.mean(rets) if rets else None

        xbi_s = xbi.loc[:snap].iloc[-1] if xbi is not None and not xbi.loc[:snap].empty else None
        xbi_fv = xbi.loc[:fwd].iloc[-1] if xbi is not None and not xbi.loc[:fwd].empty else None
        xbi_r = (xbi_fv - xbi_s) / xbi_s if xbi_s and xbi_fv else None

        raw_r = basket_xs(rec["raw_top30"])
        grd_r = basket_xs(rec["guarded_top30"])

        if raw_r is None or grd_r is None or xbi_r is None:
            updated.append(rec)
            continue

        raw_xs = raw_r - xbi_r
        grd_xs = grd_r - xbi_r
        delta = grd_xs - raw_xs

        rec["raw_port_ret"] = round(raw_r * 100, 3)
        rec["guarded_port_ret"] = round(grd_r * 100, 3)
        rec["xbi_ret"] = round(xbi_r * 100, 3)
        rec["raw_xs"] = round(raw_xs * 100, 3)
        rec["guarded_xs"] = round(grd_xs * 100, 3)
        rec["delta_pp"] = round(delta * 100, 3)
        rec["returns_filled"] = True
        updated.append(rec)

        results.append(
            {
                "snap_date": snap,
                "fwd_date": fwd,
                "raw_xs": rec["raw_xs"],
                "guarded_xs": rec["guarded_xs"],
                "delta_pp": rec["delta_pp"],
                "n_excluded": rec["n_excluded"],
            }
        )
        print(
            f"  {snap}: raw_xs={rec['raw_xs']:>+6.2f}pp  "
            f"guarded_xs={rec['guarded_xs']:>+6.2f}pp  "
            f"delta={rec['delta_pp']:>+5.2f}pp  "
            f"(excl={rec['n_excluded']})"
        )

    # Rewrite log
    with open(BASKET_LOG, "w") as f:
        for rec in updated:
            f.write(json.dumps(rec, default=str) + "\n")

    # Append results to returns log
    with open(RETURNS_LOG, "a") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    # Summary
    if results:
        filled_df = pd.read_json(RETURNS_LOG, lines=True)
        print(f"\nEES-GUARDED SHADOW SUMMARY ({len(filled_df)} completed windows):")
        print(f"  avg raw_xs:     {filled_df['raw_xs'].mean():>+6.2f}pp")
        print(f"  avg guarded_xs: {filled_df['guarded_xs'].mean():>+6.2f}pp")
        print(f"  avg delta:      {filled_df['delta_pp'].mean():>+6.2f}pp")
        print(f"  pct delta>0:    {(filled_df['delta_pp'] > 0).mean()*100:.0f}%")
        print(f"\n  Promotion gate: 20 windows, +delta, >=60% positive — currently {len(filled_df)}/20 windows")


def _add_business_days(date_str: str, n: int) -> str:
    d = date.fromisoformat(date_str)
    count = 0
    while count < n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            count += 1
    return str(d)


def main():
    parser = argparse.ArgumentParser(description="EES-guarded Top-30 shadow basket")
    parser.add_argument("--snap-date", default=str(date.today()))
    parser.add_argument("--fill-returns", action="store_true", help="Fill forward returns for all logged baskets")
    args = parser.parse_args()

    if args.fill_returns:
        fill_returns()
    else:
        log_baskets(args.snap_date)


if __name__ == "__main__":
    main()
