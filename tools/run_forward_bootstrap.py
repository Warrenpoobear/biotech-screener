#!/usr/bin/env python3
"""
run_forward_bootstrap.py — Backfilled current-model bootstrap baseline for SM-20260629-001

For each weekly snapshot window, compare the model's Top-30 5d forward excess
to a null distribution of 2,000 random same-universe EW baskets.

⚠️  BACKFILL NOTICE: The 2026 TRUTH_CARDs were generated 2026-06-28 by replaying
    the current frozen model (hash a9983a67c6954813) on historical snapshots.
    They are NOT live forward selections. These results establish a baseline for
    the current model, not out-of-sample investability proof.
    SM-20260629-001 resolution requires post-mandate forward windows only
    (2026-06-29 onward).

Classification: BACKFILLED_FORWARD_BOOTSTRAP_BASELINE / CURRENT_MODEL_REPLAY
              / BOOTSTRAP_VALIDATION / NO_MODEL_CHANGE
Mandate: SM-20260629-001 / ICD-20260629-001
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
TRUTH_CARD_DIR = REPO_ROOT / "artifacts" / "forward_validation"
SNAPSHOT_DIR = REPO_ROOT / "data" / "snapshots"
PRICE_FILE = REPO_ROOT / "production_data" / "price_history_split_adj.csv"
REGIME_MONITOR = REPO_ROOT / "artifacts" / "forward_validation" / "dem_regime_monitor_2026-06-28.json"
OUTPUT_DIR = REPO_ROOT / "artifacts" / "forward_bootstrap" / "SM-20260629-001"

N_BOOTSTRAP = 2000
BASKET_SIZE = 30
COST_BPS = 25  # conservative round-trip estimate; applied to model only


def load_regime_windows():
    with open(REGIME_MONITOR) as f:
        data = json.load(f)
    return data["windows"]


def parse_truth_card_tickers(snap_date: str):
    tc_path = TRUTH_CARD_DIR / snap_date / "TRUTH_CARD.md"
    if not tc_path.exists():
        return None
    content = tc_path.read_text()
    tickers = re.findall(r"\|\s*\d+\s*\|\s*([A-Z]{2,6})\s*\|", content)
    return tickers[:BASKET_SIZE]


def load_eligible_universe(snap_date: str, ees_excl: bool = False):
    csv_path = SNAPSHOT_DIR / snap_date / "rankings.csv"
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path, usecols=["ticker", "eligible", "ees_eligible"])
    mask = df["eligible"] == 1
    if ees_excl:
        mask = mask & df["ees_eligible"]
    return set(df.loc[mask, "ticker"].tolist())


def get_ees_flagged_in_top30(snap_date: str, top30: list):
    csv_path = SNAPSHOT_DIR / snap_date / "rankings.csv"
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path, usecols=["ticker", "ees_eligible"])
    ees_map = dict(zip(df["ticker"], df["ees_eligible"]))
    return {t for t in top30 if ees_map.get(t) is False}


def build_price_index(prices_df: pd.DataFrame):
    """Return {date: {ticker: close}} nested dict."""
    idx: dict[str, dict[str, float]] = {}
    for row in prices_df.itertuples(index=False):
        d = row.date
        if d not in idx:
            idx[d] = {}
        idx[d][row.ticker] = row.close
    return idx


def nearest_trading_date(requested: str, price_idx: dict, direction: str = "prev"):
    """Return the nearest trading date in price_idx relative to requested.

    direction='prev' looks back (for snap_date); 'next' looks forward (for end_date).
    Returns None if no date found within 5 calendar days.
    """
    if requested in price_idx and price_idx[requested]:
        return requested
    from datetime import date, timedelta

    dt = date.fromisoformat(requested)
    delta = timedelta(days=-1) if direction == "prev" else timedelta(days=1)
    for _ in range(7):
        dt += delta
        d = dt.isoformat()
        if d in price_idx and price_idx[d]:
            return d
    return None


def bootstrap_window(
    snap_date: str,
    end_date: str,
    model_xs: float,
    xbi_ret: float,
    eligible: set,
    price_idx: dict,
    n_bootstrap: int,
    cost_bps: int,
    rng: np.random.Generator,
):
    """Bootstrap one window against an eligible universe.

    Returns None if the priceable pool is too small.
    Returns dict with percentile stats and net-of-cost percentile.
    """
    # Use nearest trading day if snap_date/end_date falls on weekend/holiday
    effective_snap = nearest_trading_date(snap_date, price_idx, "prev")
    effective_end = nearest_trading_date(end_date, price_idx, "next")
    snap_prices = price_idx.get(effective_snap, {}) if effective_snap else {}
    end_prices = price_idx.get(effective_end, {}) if effective_end else {}

    pool_tickers = []
    pool_returns = []
    for t in eligible:
        p0 = snap_prices.get(t)
        p1 = end_prices.get(t)
        if p0 and p1 and p0 > 0:
            pool_tickers.append(t)
            pool_returns.append((p1 - p0) / p0 * 100)

    if len(pool_tickers) < BASKET_SIZE:
        return None

    pool_ret_arr = np.array(pool_returns)

    null_xs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(len(pool_tickers), size=BASKET_SIZE, replace=False)
        null_xs[i] = np.mean(pool_ret_arr[idx]) - xbi_ret

    percentile = float(np.mean(null_xs <= model_xs) * 100)
    model_xs_net = model_xs - cost_bps / 100
    net_percentile = float(np.mean(null_xs <= model_xs_net) * 100)

    return {
        "priceable_universe_n": len(pool_tickers),
        "null_mean_xs": float(np.mean(null_xs)),
        "null_std_xs": float(np.std(null_xs)),
        "null_p25": float(np.percentile(null_xs, 25)),
        "null_p50": float(np.percentile(null_xs, 50)),
        "null_p75": float(np.percentile(null_xs, 75)),
        "percentile": percentile,
        "net_percentile": net_percentile,
        "p_value": float(np.mean(null_xs >= model_xs)),
    }


def run_all_windows(args):
    print("Loading regime windows...")
    windows = load_regime_windows()
    print(f"  {len(windows)} windows")

    print("Loading prices (may take a moment)...")
    prices_df = pd.read_csv(PRICE_FILE, usecols=["ticker", "date", "close"])
    price_idx = build_price_index(prices_df)
    print(f"  {len(price_idx)} date entries, {prices_df['ticker'].nunique()} tickers")

    rng = np.random.default_rng(args.seed)
    results = []
    skipped = []

    for w in windows:
        snap_date = w["snap_date"]
        end_date = w.get("end_5d")
        model_xs = w.get("xs_5d_pct")
        xbi_ret = w.get("xbi_5d_pct")
        regime = w.get("regime", "UNKNOWN")

        if end_date is None or model_xs is None or xbi_ret is None:
            skipped.append({"snap_date": snap_date, "reason": "missing_return_data"})
            continue

        top30 = parse_truth_card_tickers(snap_date)
        if not top30 or len(top30) < BASKET_SIZE:
            skipped.append({"snap_date": snap_date, "reason": "truth_card_unavailable"})
            continue

        eligible_base = load_eligible_universe(snap_date, ees_excl=False)
        eligible_ees = load_eligible_universe(snap_date, ees_excl=True)
        ees_flagged = get_ees_flagged_in_top30(snap_date, top30)

        b_base = bootstrap_window(
            snap_date,
            end_date,
            model_xs,
            xbi_ret,
            eligible_base,
            price_idx,
            args.n_bootstrap,
            args.cost_bps,
            rng,
        )
        if b_base is None:
            skipped.append({"snap_date": snap_date, "reason": "priceable_pool_too_small"})
            continue

        b_ees = bootstrap_window(
            snap_date,
            end_date,
            model_xs,
            xbi_ret,
            eligible_ees,
            price_idx,
            args.n_bootstrap,
            args.cost_bps,
            rng,
        )

        rec = {
            "snap_date": snap_date,
            "end_5d": end_date,
            "regime": regime,
            "model_xs_5d": model_xs,
            "model_xs_5d_net": round(model_xs - args.cost_bps / 100, 4),
            "xbi_5d": xbi_ret,
            "cost_bps": args.cost_bps,
            "baseline": {
                "priceable_universe_n": b_base["priceable_universe_n"],
                "null_mean_xs": round(b_base["null_mean_xs"], 4),
                "null_std_xs": round(b_base["null_std_xs"], 4),
                "null_p25": round(b_base["null_p25"], 4),
                "null_p50": round(b_base["null_p50"], 4),
                "null_p75": round(b_base["null_p75"], 4),
                "percentile": round(b_base["percentile"], 2),
                "net_percentile": round(b_base["net_percentile"], 2),
                "p_value": round(b_base["p_value"], 4),
            },
            "ees_sensitivity": (
                {
                    "priceable_universe_n": b_ees["priceable_universe_n"],
                    "percentile": round(b_ees["percentile"], 2),
                    "net_percentile": round(b_ees["net_percentile"], 2),
                    "null_mean_xs": round(b_ees["null_mean_xs"], 4),
                    "p_value": round(b_ees["p_value"], 4),
                }
                if b_ees is not None
                else None
            ),
            "ees_flagged_in_top30": sorted(ees_flagged),
            "n_ees_flagged": len(ees_flagged),
        }
        results.append(rec)

        status = "✓" if b_base["percentile"] >= 50 else "✗"
        print(f"  {snap_date} [{regime:9s}] xs={model_xs:+5.2f}%  " f"pct={b_base['percentile']:5.1f}  {status}")

    return results, skipped, windows


def compute_summary(results, n_total_windows, args):
    n = len(results)
    if n == 0:
        return {}

    pcts = [r["baseline"]["percentile"] for r in results]
    net_pcts = [r["baseline"]["net_percentile"] for r in results]
    xs = [r["model_xs_5d"] for r in results]

    rally = [r for r in results if r["regime"] == "RALLY"]
    non_rally = [r for r in results if r["regime"] == "NON_RALLY"]

    def regime_block(subset):
        if not subset:
            return None
        p = [r["baseline"]["percentile"] for r in subset]
        x = [r["model_xs_5d"] for r in subset]
        return {
            "n": len(subset),
            "median_percentile": round(float(np.median(p)), 2),
            "mean_percentile": round(float(np.mean(p)), 2),
            "pct_above_50": round(float(np.mean([v >= 50 for v in p])) * 100, 1),
            "pct_above_75": round(float(np.mean([v >= 75 for v in p])) * 100, 1),
            "mean_model_xs": round(float(np.mean(x)), 4),
        }

    median_pct = float(np.median(pcts))
    pct_above_50 = float(np.mean([v >= 50 for v in pcts]))

    # SM-20260629-001 success gate: median ≥75 AND >50% windows above 50th pct
    # AND rally + non_rally both >50 median percentile
    rally_ok = rally and float(np.median([r["baseline"]["percentile"] for r in rally])) > 50
    non_rally_ok = non_rally and float(np.median([r["baseline"]["percentile"] for r in non_rally])) > 50

    gate_success = median_pct >= 75 and pct_above_50 >= 0.5 and non_rally_ok
    gate_failure = median_pct <= 50 and pct_above_50 < 0.5

    # Mean net xs
    net_xs = [r["model_xs_5d_net"] for r in results]

    return {
        "n_windows": n,
        "n_windows_total": n_total_windows,
        "n_bootstrap_per_window": args.n_bootstrap,
        "cost_bps": args.cost_bps,
        "seed": args.seed,
        # Baseline
        "median_percentile": round(median_pct, 2),
        "mean_percentile": round(float(np.mean(pcts)), 2),
        "pct_above_50": round(pct_above_50 * 100, 1),
        "pct_above_75": round(float(np.mean([v >= 75 for v in pcts])) * 100, 1),
        "mean_model_xs": round(float(np.mean(xs)), 4),
        "median_model_xs": round(float(np.median(xs)), 4),
        # Net-of-cost
        "median_net_percentile": round(float(np.median(net_pcts)), 2),
        "mean_net_xs": round(float(np.mean(net_xs)), 4),
        # Regime breakdown
        "rally": regime_block(rally),
        "non_rally": regime_block(non_rally),
        # Gate evaluation
        "mandate_gate": {
            "success": gate_success,
            "failure": gate_failure,
            "non_rally_ok": non_rally_ok,
            "rally_ok": rally_ok,
            "verdict": ("SUCCESS" if gate_success else "FAILURE" if gate_failure else "INCONCLUSIVE"),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Forward bootstrap for SM-20260629-001")
    parser.add_argument(
        "--n-bootstrap", type=int, default=N_BOOTSTRAP, help=f"Random baskets per window (default {N_BOOTSTRAP})"
    )
    parser.add_argument(
        "--cost-bps", type=int, default=COST_BPS, help=f"Round-trip cost in bps for net percentile (default {COST_BPS})"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    print("\n=== Backfilled Current-Model Bootstrap Baseline — SM-20260629-001 ===")
    print("⚠️  BACKFILL: TRUTH_CARDs generated 2026-06-28 by replaying current model on historical snapshots.")
    print("   These are NOT live forward selections. Use as baseline only.")
    print(f"n_bootstrap={args.n_bootstrap}, cost_bps={args.cost_bps}, seed={args.seed}")
    print(f"Output: {args.output_dir}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    results, skipped, windows = run_all_windows(args)

    summary = compute_summary(results, len(windows), args)

    # Window-level CSV
    rows = []
    for r in results:
        b = r["baseline"]
        e = r.get("ees_sensitivity") or {}
        rows.append(
            {
                "snap_date": r["snap_date"],
                "end_5d": r["end_5d"],
                "regime": r["regime"],
                "model_xs_5d": r["model_xs_5d"],
                "model_xs_5d_net": r["model_xs_5d_net"],
                "xbi_5d": r["xbi_5d"],
                "priceable_n": b["priceable_universe_n"],
                "null_mean_xs": b["null_mean_xs"],
                "null_p50": b["null_p50"],
                "percentile": b["percentile"],
                "net_percentile": b["net_percentile"],
                "p_value": b["p_value"],
                "ees_percentile": e.get("percentile"),
                "ees_null_mean": e.get("null_mean_xs"),
                "n_ees_flagged": r["n_ees_flagged"],
            }
        )
    csv_path = args.output_dir / "forward_bootstrap_windows.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # JSON summary
    output = {
        "mandate_id": "SM-20260629-001",
        "dol_row": "ICD-20260629-001",
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"),
        "classification": (
            "BACKFILLED_FORWARD_BOOTSTRAP_BASELINE / CURRENT_MODEL_REPLAY" " / BOOTSTRAP_VALIDATION / NO_MODEL_CHANGE"
        ),
        "backfilled_current_model_replay": True,
        "truth_cards_generated_at": "2026-06-28",
        "model_hash": "a9983a67...",  # truncated; full hash in TRUTH_CARD model_hash field
        "forward_evidence_status": "BASELINE_ONLY",
        "forward_window_start": "2026-06-29",
        "resolution_requires": (
            "post-mandate live forward windows only (2026-06-29 onward); "
            "backfilled windows do not count toward 20-window success gate"
        ),
        "summary": summary,
        "windows": results,
        "skipped": skipped,
    }
    json_path = args.output_dir / "forward_bootstrap_summary.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print table
    s = summary
    g = s.get("mandate_gate", {})
    print("\n" + "=" * 70)
    print("BACKFILLED BASELINE — SM-20260629-001 (current-model replay, NOT forward proof)")
    print("=" * 70)
    print(f"Windows processed:       {s['n_windows']} / {s['n_windows_total']}")
    print(f"Skipped:                 {len(skipped)}")
    print()
    print(f"Median percentile:       {s['median_percentile']:.1f}")
    print(f"Mean percentile:         {s['mean_percentile']:.1f}")
    print(f"% windows >= 50th pct:   {s['pct_above_50']:.1f}%")
    print(f"% windows >= 75th pct:   {s['pct_above_75']:.1f}%")
    print(f"Mean model xs (gross):   {s['mean_model_xs']:+.3f}%")
    print(f"Mean model xs (net):     {s['mean_net_xs']:+.3f}%")
    print(f"Median net percentile:   {s['median_net_percentile']:.1f}")
    print()
    if s.get("rally"):
        r = s["rally"]
        print(
            f"RALLY     ({r['n']:3d} windows):  median_pct={r['median_percentile']:.1f}  "
            f"pct>50={r['pct_above_50']:.0f}%  xs={r['mean_model_xs']:+.3f}%"
        )
    if s.get("non_rally"):
        nr = s["non_rally"]
        print(
            f"NON_RALLY ({nr['n']:3d} windows):  median_pct={nr['median_percentile']:.1f}  "
            f"pct>50={nr['pct_above_50']:.0f}%  xs={nr['mean_model_xs']:+.3f}%"
        )
    print()
    print(f"Baseline verdict:        {g.get('verdict', 'N/A')} (backfilled; does not resolve mandate)")
    print(f"  success gate (median>=75 & >50% above 50):  {g.get('success')}")
    print(f"  non-rally gate (non_rally median pct>50):   {g.get('non_rally_ok')}")
    print(f"  failure gate  (median<=50 & <50% above 50): {g.get('failure')}")
    print()
    print("⚠️  SM-20260629-001 remains OPEN. Resolution requires post-mandate")
    print("   forward windows (2026-06-29 onward). 20-window gate starts from today.")
    print("=" * 70)
    print(f"\nCSV:  {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
