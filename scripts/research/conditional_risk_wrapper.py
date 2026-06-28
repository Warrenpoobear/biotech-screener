"""
Conditional Risk Wrapper Shadow
VALIDATION_INFRASTRUCTURE / STRESS_WRAPPER_SHADOW / NO_MODEL_CHANGE / NO_SELECTOR_CHANGE

Computes risk_wrapper_active flag from four activation conditions.
Logs five comparison baskets and fills forward returns for completed windows.

Activation conditions (any one triggers):
  (A) rolling_4w_xs <= -5.00pp     — 4-window rolling drawdown
  (B) repeat_offender_count >= 2   — ≥2 names flagged as serial bottom-5
  (C) ees_false_top30_count >= 5   — ≥5 EES-False names in Top-30
  (D) build_window + less_binary >= 55%  — elevated event-premium exposure

Baskets (always built; comparison is meaningful only when wrapper active):
  raw_top30    — model selection unchanged
  ees_guarded  — exclude EES-False from top30, fill from ranks 31-60
  ro_guarded   — exclude flagged repeat offenders, fill from ranks 31-60
  risk_guarded — exclude (EES-False OR flagged repeat offenders), fill from 31-60
  rank31_60    — ranks 31-60 EW (stress replacement bench)

Usage:
    python3 scripts/research/conditional_risk_wrapper.py [--snap-date YYYY-MM-DD]
    python3 scripts/research/conditional_risk_wrapper.py --fill-returns
    python3 scripts/research/conditional_risk_wrapper.py --summary
    python3 scripts/research/conditional_risk_wrapper.py --backfill
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

CAPTURES_LOG = OUTDIR / "stress_wrapper_captures.jsonl"
RETURNS_LOG = OUTDIR / "stress_wrapper_returns.jsonl"
PRICE_FILE = REPO / "production_data/price_history_split_adj.csv"

THRESHOLDS = {
    "rolling_4w_xs_pp": -5.0,
    "repeat_offender_count": 2,
    "ees_false_count": 5,
    "build_less_binary_pct": 55.0,
}


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


def _rolling_xs_from_validation(as_of: str) -> float | None:
    """Compute rolling 4w XS directly from validation JSON when trigger file absent."""
    vpath = REPO / "artifacts/autopsy/top30_ytd_validation/2026_ytd_top30_validation.json"
    if not vpath.exists():
        return None
    data = json.loads(vpath.read_text())
    rows = data.get("all_snapshots") or data.get("weekly_snapshots", [])
    df = pd.DataFrame(rows)
    if df.empty or "xs_5d" not in df.columns:
        return None
    df["snap_date"] = df["snap_date"].astype(str)
    df = df[df["snap_date"] <= as_of].sort_values("snap_date")
    if len(df) < 4:
        return None
    last4 = df.tail(4)["xs_5d"].sum() * 100
    return round(last4, 3)


def _repeat_offender_count_from_seed(snap_date: str, df_snap: pd.DataFrame) -> int:
    """Count flagged repeat offenders inline when repeat_offender JSON absent."""
    seed = OUTDIR / "repeat_offender_ledger.jsonl"
    src_csv = REPO / "artifacts/autopsy/ytd_top30_failure_postmortem_2026/name_attribution.csv"
    frames = []
    if src_csv.exists():
        frames.append(pd.read_csv(src_csv))
    if seed.exists():
        rows = []
        with open(seed) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return 0
    hist = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["window", "ticker"], keep="last")
    if hist.empty or "contrib_to_xs" not in hist.columns:
        return 0

    top30 = df_snap[df_snap["actionable_rank"] <= 30] if not df_snap.empty else pd.DataFrame()
    if top30.empty:
        return 0

    all_windows = sorted(hist["window"].unique(), reverse=True)[:12]
    count = 0
    for _, row in top30.iterrows():
        ticker = row["ticker"]
        bad = []
        for w in all_windows:
            wdf = hist[hist["window"] == w].dropna(subset=["contrib_to_xs"])
            if wdf.empty:
                continue
            if ticker in wdf.nsmallest(5, "contrib_to_xs")["ticker"].tolist():
                bad.append(w)
        if len(bad) >= 2:
            count += 1
    return count


def check_conditions(snap_date: str, df_snap: pd.DataFrame) -> dict:
    """Evaluate all four activation conditions. Returns full conditions dict."""
    conditions = {}

    # Condition A: rolling 4w XS
    trigger_path = OUTDIR / f"drawdown_trigger_{snap_date}.json"
    if trigger_path.exists():
        with open(trigger_path) as f:
            td = json.load(f)
        rolling_xs = td.get("rolling_xs_pp")
    else:
        rolling_xs = _rolling_xs_from_validation(snap_date)
    conditions["rolling_4w_xs_pp"] = rolling_xs
    conditions["cond_A"] = rolling_xs is not None and rolling_xs <= THRESHOLDS["rolling_4w_xs_pp"]

    # Condition B: repeat-offender count
    ro_path = OUTDIR / f"repeat_offender_{snap_date}.json"
    if ro_path.exists():
        with open(ro_path) as f:
            ro_data = json.load(f)
        ro_count = ro_data.get("n_flagged", 0)
    else:
        ro_count = _repeat_offender_count_from_seed(snap_date, df_snap)
    conditions["repeat_offender_count"] = ro_count
    conditions["cond_B"] = ro_count >= THRESHOLDS["repeat_offender_count"]

    # Condition C: EES-False count in Top-30
    top30 = df_snap[df_snap["actionable_rank"] <= 30] if not df_snap.empty else pd.DataFrame()
    ees_false_count = 0
    if not top30.empty and "ees_v3_gate" in top30.columns:
        ees_false_count = int(top30["ees_v3_gate"].eq(False).sum())
    conditions["ees_false_count"] = ees_false_count
    conditions["cond_C"] = ees_false_count >= THRESHOLDS["ees_false_count"]

    # Condition D: build_window + less_binary pct
    bl_pct = 0.0
    if not top30.empty and "catalyst_bucket" in top30.columns:
        vc = top30["catalyst_bucket"].value_counts(normalize=True) * 100
        bl_pct = vc.get("build_window", 0.0) + vc.get("less_binary", 0.0)
    conditions["build_less_binary_pct"] = round(float(bl_pct), 1)
    conditions["cond_D"] = bl_pct >= THRESHOLDS["build_less_binary_pct"]

    active_triggers = [k for k in ("cond_A", "cond_B", "cond_C", "cond_D") if conditions[k]]
    conditions["risk_wrapper_active"] = len(active_triggers) > 0
    conditions["active_triggers"] = active_triggers
    return conditions


def _is_ees_fail(v) -> bool:
    return v is False or str(v).lower() in ("false", "0")


def build_baskets(df_snap: pd.DataFrame, ro_flagged: set) -> dict:
    """Build five comparison baskets from current snapshot."""
    top30 = df_snap[df_snap["actionable_rank"] <= 30].sort_values("actionable_rank")
    bench = df_snap[(df_snap["actionable_rank"] > 30) & (df_snap["actionable_rank"] <= 60)].sort_values(
        "actionable_rank"
    )

    t30_list = top30["ticker"].tolist()
    b_list = bench["ticker"].tolist()
    has_ees = "ees_v3_gate" in df_snap.columns

    def fill_from_bench(excluded: set) -> list:
        kept = [t for t in t30_list if t not in excluded]
        need = 30 - len(kept)
        for _, br in bench.iterrows():
            if need <= 0:
                break
            bt = br["ticker"]
            if bt not in excluded:
                kept.append(bt)
                need -= 1
        return kept

    ees_excluded = set()
    if has_ees:
        ees_excluded = {r["ticker"] for _, r in top30.iterrows() if _is_ees_fail(r["ees_v3_gate"])}

    combined = ees_excluded | ro_flagged

    return {
        "raw_top30": t30_list,
        "ees_guarded": fill_from_bench(ees_excluded),
        "ro_guarded": fill_from_bench(ro_flagged),
        "risk_guarded": fill_from_bench(combined),
        "rank31_60": b_list,
        "n_ees_excluded": len(ees_excluded),
        "n_ro_excluded": len(ro_flagged),
        "n_combined_excl": len(combined),
    }


def log_capture(snap_date: str, verbose: bool = True) -> None:
    actual_snap = find_latest_snap(snap_date)
    if not actual_snap:
        print(f"ERROR: No snapshot found at or before {snap_date}")
        return

    if verbose:
        print(f"Conditional Risk Wrapper — {snap_date} (snap: {actual_snap})")

    df_snap = load_snap(actual_snap)
    if df_snap.empty:
        print("ERROR: Could not load snapshot")
        return

    # Dedup
    existing = set()
    if CAPTURES_LOG.exists():
        with open(CAPTURES_LOG) as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.add(json.loads(line).get("snap_date"))
    if snap_date in existing:
        if verbose:
            print(f"  Already logged for {snap_date} — skipping.")
        return

    conditions = check_conditions(snap_date, df_snap)
    active = conditions["risk_wrapper_active"]
    triggers = conditions["active_triggers"]

    if verbose:
        status = "ACTIVE" if active else "inactive"
        print(f"  risk_wrapper: {status}  triggers={triggers}")
        if active:
            for t in triggers:
                print(f"    [{t}] fired")

    ro_flagged = set()
    ro_path = OUTDIR / f"repeat_offender_{snap_date}.json"
    if ro_path.exists():
        with open(ro_path) as f:
            rd = json.load(f)
        ro_flagged = {x["ticker"] for x in rd.get("flagged", [])}

    baskets = build_baskets(df_snap, ro_flagged)
    fwd_date = _add_business_days(snap_date, 5)

    record = {
        "snap_date": snap_date,
        "actual_snap": actual_snap,
        "fwd_date": fwd_date,
        "risk_wrapper_active": active,
        "active_triggers": triggers,
        "conditions": {k: v for k, v in conditions.items() if k not in ("active_triggers", "risk_wrapper_active")},
        "n_ees_excluded": baskets["n_ees_excluded"],
        "n_ro_excluded": baskets["n_ro_excluded"],
        "n_combined_excl": baskets["n_combined_excl"],
        "baskets": {k: v for k, v in baskets.items() if isinstance(v, list)},
        "returns_filled": False,
        "classification": "VALIDATION_INFRASTRUCTURE/STRESS_WRAPPER_SHADOW/NO_MODEL_CHANGE",
    }

    with open(CAPTURES_LOG, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")

    if verbose:
        for bname, tlist in record["baskets"].items():
            print(f"  {bname:<18}: {len(tlist)} names")
        print(f"  → {fwd_date}  logged to {CAPTURES_LOG.name}")


def fill_returns() -> None:
    if not CAPTURES_LOG.exists():
        print("No captures log found.")
        return

    print("Filling forward returns for stress wrapper baskets...")
    prices_raw = pd.read_csv(PRICE_FILE, parse_dates=["date"])
    prices = prices_raw.pivot(index="date", columns="ticker", values="close").sort_index()
    xbi = prices["XBI"].dropna() if "XBI" in prices.columns else None

    records = []
    with open(CAPTURES_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    updated = []
    results = []
    new_count = 0

    for rec in records:
        if rec.get("returns_filled"):
            updated.append(rec)
            continue

        snap = rec["snap_date"]
        fwd = rec["fwd_date"]

        if prices.index[prices.index >= pd.Timestamp(fwd)].empty:
            updated.append(rec)
            continue

        def bxs(tickers):
            rets = []
            for t in tickers:
                if t not in prices.columns:
                    continue
                p = prices[t].dropna()
                sv = p.loc[:snap]
                fv = p.loc[:fwd]
                s = sv.iloc[-1] if not sv.empty else None
                v = fv.iloc[-1] if not fv.empty else None
                if s and v and s > 0:
                    rets.append((v - s) / s)
            return np.mean(rets) * 100 if rets else None

        xbi_s = xbi.loc[:snap].iloc[-1] if xbi is not None and not xbi.loc[:snap].empty else None
        xbi_fv = xbi.loc[:fwd].iloc[-1] if xbi is not None and not xbi.loc[:fwd].empty else None
        xbi_r = round((xbi_fv - xbi_s) / xbi_s * 100, 3) if xbi_s and xbi_fv else None

        basket_xs = {}
        for bname, tlist in rec["baskets"].items():
            r = bxs(tlist)
            basket_xs[bname] = round(r - xbi_r, 3) if r is not None and xbi_r is not None else None

        if any(v is None for v in basket_xs.values()):
            updated.append(rec)
            continue

        rec["xbi_ret"] = xbi_r
        rec["basket_xs"] = basket_xs
        rec["returns_filled"] = True
        updated.append(rec)
        new_count += 1

        result = {
            "snap_date": snap,
            "fwd_date": fwd,
            "xbi_ret": xbi_r,
            "wrapper_active": rec["risk_wrapper_active"],
        }
        result.update({f"xs_{k}": v for k, v in basket_xs.items()})
        results.append(result)

        if new_count <= 20 or new_count % 20 == 0:
            flag = "ACTIVE " if rec["risk_wrapper_active"] else "passive"
            print(
                f"  {snap} [{flag}]: raw={basket_xs.get('raw_top30', 0):>+5.2f}pp  "
                f"risk_g={basket_xs.get('risk_guarded', 0):>+5.2f}pp  "
                f"bench={basket_xs.get('rank31_60', 0):>+5.2f}pp"
            )

    with open(CAPTURES_LOG, "w") as f:
        for rec in updated:
            f.write(json.dumps(rec, default=str) + "\n")

    with open(RETURNS_LOG, "a") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    print(f"  Filled {new_count} new windows.")
    if results:
        _print_summary()


def _print_summary() -> None:
    if not RETURNS_LOG.exists():
        print("No returns log yet.")
        return

    df = pd.read_json(RETURNS_LOG, lines=True)
    if df.empty:
        return

    xs_cols = [c for c in df.columns if c.startswith("xs_")]
    active_df = df[df["wrapper_active"]]
    passive_df = df[~df["wrapper_active"]]

    print(f"\nSTRESS WRAPPER RETURNS SUMMARY ({len(df)} completed windows):")
    print(f"  WRAPPER ACTIVE: {len(active_df)}  |  WRAPPER PASSIVE: {len(passive_df)}")

    for label, sub in [("ALL WINDOWS", df), ("ACTIVE ONLY", active_df), ("PASSIVE ONLY", passive_df)]:
        if sub.empty:
            continue
        print(f"\n  {label} (n={len(sub)}):")
        for c in xs_cols:
            bname = c[3:]
            avg = sub[c].mean()
            pct = (sub[c] > 0).mean() * 100
            print(f"    {bname:<20}: avg_xs={avg:>+5.2f}pp  pct_pos={pct:.0f}%")

    if "xs_raw_top30" in df.columns and "xs_risk_guarded" in df.columns and not active_df.empty:
        delta = active_df["xs_risk_guarded"].mean() - active_df["xs_raw_top30"].mean()
        print(f"\n  Key Q — risk_guarded vs raw in ACTIVE windows: {delta:>+.2f}pp avg delta")
    print("\n  Promotion gate: 20 active windows with positive risk_guarded delta")


def _backfill() -> None:
    """Backfill all snapshot dates into the captures log, then fill returns."""
    snap_dirs = sorted(glob.glob(str(REPO / "data/snapshots/????-??-??")))
    all_dates = [Path(d).name for d in snap_dirs if len(Path(d).name) == 10]
    print(f"Backfilling {len(all_dates)} snapshot dates...")

    existing = set()
    if CAPTURES_LOG.exists():
        with open(CAPTURES_LOG) as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.add(json.loads(line).get("snap_date"))

    new_count = 0
    for snap_date in all_dates:
        if snap_date in existing:
            continue
        log_capture(snap_date, verbose=False)
        new_count += 1
        if new_count % 20 == 0:
            print(f"  {new_count} / {len(all_dates) - len(existing)} captured...")

    print(f"Captured {new_count} new dates. Filling returns...")
    fill_returns()


def _add_business_days(date_str: str, n: int) -> str:
    d = date.fromisoformat(date_str)
    count = 0
    while count < n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            count += 1
    return str(d)


def main():
    parser = argparse.ArgumentParser(description="Conditional risk wrapper shadow")
    parser.add_argument("--snap-date", default=str(date.today()))
    parser.add_argument("--fill-returns", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--backfill", action="store_true")
    args = parser.parse_args()

    if args.summary:
        _print_summary()
    elif args.fill_returns:
        fill_returns()
    elif args.backfill:
        _backfill()
    else:
        log_capture(args.snap_date)


if __name__ == "__main__":
    main()
