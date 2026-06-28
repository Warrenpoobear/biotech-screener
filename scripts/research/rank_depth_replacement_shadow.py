"""
Rank-Depth Replacement Shadow
VALIDATION_INFRASTRUCTURE / FAILURE_MODE_SHADOW_GUARDS / NO_MODEL_CHANGE

Logs five baskets each run and computes forward XS once prices are available:
  top30           — current model selection
  rank31_60       — bench cohort
  top60_ew        — combined
  ees_guarded     — EES-excluded top30 with 31-60 fill
  ro_guarded      — repeat-offender-excluded top30 with 31-60 fill

Key question: is the bench (31-60) advantage failure-specific or persistent?
If bench consistently beats top30, the rank cutoff / ranker ordering needs review.
If bench beats top30 only during stress, it is a risk-replacement bench.

Usage:
    python3 scripts/research/rank_depth_replacement_shadow.py [--snap-date YYYY-MM-DD]
    python3 scripts/research/rank_depth_replacement_shadow.py --fill-returns
    python3 scripts/research/rank_depth_replacement_shadow.py --summary
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

BASKET_LOG = OUTDIR / "rank_depth_baskets.jsonl"
RETURNS_LOG = OUTDIR / "rank_depth_returns.jsonl"
PRICE_FILE = REPO / "production_data/price_history_split_adj.csv"

# Import repeat-offender data if available
OFFENDER_DIR = OUTDIR


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


def build_baskets(df_snap: pd.DataFrame, offender_set: set = None) -> dict:
    top30 = df_snap[df_snap["actionable_rank"] <= 30].sort_values("actionable_rank")
    bench = df_snap[(df_snap["actionable_rank"] > 30) & (df_snap["actionable_rank"] <= 60)].sort_values(
        "actionable_rank"
    )

    t30_list = top30["ticker"].tolist()
    b_list = bench["ticker"].tolist()
    has_ees = "ees_v3_gate" in df_snap.columns

    def is_ees_fail(v):
        return v is False or str(v).lower() in ("false", "0")

    def fill_from_bench(excluded_set, bench_df, excl_ees=False):
        kept = [t for t in t30_list if t not in excluded_set]
        need = 30 - len(kept)
        for _, br in bench_df.iterrows():
            if need <= 0:
                break
            bt = br["ticker"]
            if bt in excluded_set:
                continue
            if excl_ees and has_ees and is_ees_fail(br.get("ees_v3_gate")):
                continue
            kept.append(bt)
            need -= 1
        return kept

    # EES-guarded
    ees_excluded = set()
    if has_ees:
        ees_excluded = {r["ticker"] for _, r in top30.iterrows() if is_ees_fail(r["ees_v3_gate"])}
    ees_guarded = fill_from_bench(ees_excluded, bench, excl_ees=True)

    # Repeat-offender-guarded
    ro_excluded = offender_set or set()
    ro_guarded = fill_from_bench(ro_excluded, bench)

    return {
        "top30": t30_list,
        "rank31_60": b_list,
        "top60_ew": t30_list + b_list,
        "ees_guarded": ees_guarded,
        "ro_guarded": ro_guarded,
        "ees_excluded_n": len(ees_excluded),
        "ro_excluded_n": len(ro_excluded),
    }


def log_baskets(snap_date: str) -> None:
    actual_snap = find_latest_snap(snap_date)
    if not actual_snap:
        print(f"ERROR: No snapshot found at or before {snap_date}")
        return

    print(f"Rank-Depth Replacement Shadow — {snap_date} (snap: {actual_snap})")

    df_snap = load_snap(actual_snap)
    if df_snap.empty:
        print("ERROR: Could not load snapshot")
        return

    # Load any repeat-offender flags from today
    offender_set = set()
    ro_file = OFFENDER_DIR / f"repeat_offender_{snap_date}.json"
    if ro_file.exists():
        with open(ro_file) as f:
            ro_data = json.load(f)
        offender_set = {x["ticker"] for x in ro_data.get("flagged", [])}
    if offender_set:
        print(f"  Loaded {len(offender_set)} repeat-offender flags: {sorted(offender_set)}")

    baskets = build_baskets(df_snap, offender_set)
    regime = str(df_snap["regime_label"].iloc[0]) if "regime_label" in df_snap.columns else "UNKNOWN"
    fwd_date = _add_business_days(snap_date, 5)

    # Dedup check
    existing = set()
    if BASKET_LOG.exists():
        with open(BASKET_LOG) as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.add(json.loads(line).get("snap_date"))
    if snap_date in existing:
        print(f"  Already logged for {snap_date} — skipping.")
        return

    record = {
        "snap_date": snap_date,
        "actual_snap": actual_snap,
        "fwd_date": fwd_date,
        "regime": regime,
        "ees_excluded_n": baskets["ees_excluded_n"],
        "ro_excluded_n": baskets["ro_excluded_n"],
        "baskets": {k: v for k, v in baskets.items() if k.endswith("_n") is False and isinstance(v, list)},
        "returns_filled": False,
        "classification": "VALIDATION_INFRASTRUCTURE/NO_MODEL_CHANGE",
    }

    with open(BASKET_LOG, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")

    for bname, tlist in record["baskets"].items():
        print(f"  {bname:<16}: {len(tlist)} names")
    print(f"  Forward window: {snap_date} → {fwd_date}")
    print(f"  Logged to {BASKET_LOG}")


def fill_returns() -> None:
    if not BASKET_LOG.exists():
        print("No basket log found.")
        return

    print("Filling forward returns for rank-depth baskets...")
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

        avail = prices.index[prices.index >= pd.Timestamp(fwd)]
        if avail.empty:
            updated.append(rec)
            continue

        def bxs(tickers):
            rets = []
            for t in tickers:
                if t not in prices.columns:
                    continue
                p = prices[t].dropna()
                s = p.loc[:snap].iloc[-1] if not p.loc[:snap].empty else None
                fv = p.loc[:fwd].iloc[-1] if not p.loc[:fwd].empty else None
                if s and fv and s > 0:
                    rets.append((fv - s) / s)
            return np.mean(rets) * 100 if rets else None

        xbi_s = xbi.loc[:snap].iloc[-1] if xbi is not None and not xbi.loc[:snap].empty else None
        xbi_fv = xbi.loc[:fwd].iloc[-1] if xbi is not None and not xbi.loc[:fwd].empty else None
        xbi_r = round((xbi_fv - xbi_s) / xbi_s * 100, 3) if xbi_s and xbi_fv else None

        basket_rets = {}
        for bname, tlist in rec["baskets"].items():
            r = bxs(tlist)
            basket_rets[bname] = round(r - xbi_r, 3) if r is not None and xbi_r is not None else None

        if any(v is None for v in basket_rets.values()):
            updated.append(rec)
            continue

        rec["xbi_ret"] = xbi_r
        rec["basket_xs"] = basket_rets
        rec["returns_filled"] = True
        updated.append(rec)

        result = {"snap_date": snap, "fwd_date": fwd, "xbi_ret": xbi_r}
        result.update({f"xs_{k}": v for k, v in basket_rets.items()})
        results.append(result)

        print(
            f"  {snap}: top30={basket_rets.get('top30', 0):>+5.2f}pp  "
            f"bench={basket_rets.get('rank31_60', 0):>+5.2f}pp  "
            f"ees_guard={basket_rets.get('ees_guarded', 0):>+5.2f}pp  "
            f"ro_guard={basket_rets.get('ro_guarded', 0):>+5.2f}pp"
        )

    with open(BASKET_LOG, "w") as f:
        for rec in updated:
            f.write(json.dumps(rec, default=str) + "\n")

    with open(RETURNS_LOG, "a") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    if results:
        _print_summary()


def _print_summary() -> None:
    if not RETURNS_LOG.exists():
        print("No returns log yet.")
        return

    df = pd.read_json(RETURNS_LOG, lines=True)
    if df.empty:
        return

    print(f"\nRANK-DEPTH REPLACEMENT SUMMARY ({len(df)} completed windows):")
    xs_cols = [c for c in df.columns if c.startswith("xs_")]
    for c in xs_cols:
        bname = c[3:]
        avg = df[c].mean()
        pct_pos = (df[c] > 0).mean() * 100
        print(f"  {bname:<18}: avg_xs={avg:>+5.2f}pp  pct_positive={pct_pos:.0f}%")

    # Key question answer
    if "xs_top30" in df.columns and "xs_rank31_60" in df.columns:
        bench_delta = df["xs_rank31_60"].mean() - df["xs_top30"].mean()
        print(f"\n  Bench vs Top-30 avg delta: {bench_delta:>+5.2f}pp")
        print(
            f"  {'→ PERSISTENT bench advantage — review rank cutoff' if bench_delta > 0.5 else '→ Bench advantage appears failure-specific (expected)'}"
        )
    print(f"\n  n_windows={len(df)} — promotion gate threshold: 20 windows")


def _add_business_days(date_str: str, n: int) -> str:
    d = date.fromisoformat(date_str)
    count = 0
    while count < n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            count += 1
    return str(d)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snap-date", default=str(date.today()))
    parser.add_argument("--fill-returns", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    if args.summary:
        _print_summary()
    elif args.fill_returns:
        fill_returns()
    else:
        log_baskets(args.snap_date)


if __name__ == "__main__":
    main()
