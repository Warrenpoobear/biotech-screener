#!/usr/bin/env python3
"""Shared-regime comparator IC check.

Replicates build_ic_dashboard.py methodology exactly for a flagged signal
and its comparator over the IDENTICAL date window. Diagnoses whether the
flagged signal's failure is shared-regime (comparator also broken) or
signal-specific (comparator healthy, low IC trajectory correlation).

Usage:
    python scripts/shared_regime_check.py \\
        --as-of-date 2026-05-20 \\
        --dashboard artifacts/ic_dashboard/2026-05-20_dashboard.json

Output:
    artifacts/ic_dashboard/<date>_shared_regime_check.json
    <FLAG1>_<FLAG2>_SHARED_REGIME_CHECK_<date>.md at repo root
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SCHEMA_VERSION = "shared_regime_check.v1"
DEFAULT_HORIZON = 20
MIN_N_OBS = 10

# Signal pairs to check: (flagged_signal, comparator_signal, higher_is_better_flag, higher_is_better_comp)
# higher_is_better from build_ic_dashboard.py SIGNALS list:
#   score_rank_pct:  higher_is_better=True   (percentile of composite_score;
#                    was negated until 2026-11-12, which inverted its IC)
#   inst_delta_z:    higher_is_better=True
#   clinical_optionality_pct_dev: higher_is_better=True
#   coinvest_score_z: higher_is_better=True   (same lane as inst_delta_z)

SIGNAL_HIB = {
    "score_rank_pct": True,
    "clinical_optionality_pct_dev": True,
    "inst_delta_z": True,
    "coinvest_score_z": True,
}

# Comparator pairs: (flagged, comparator, lane_description)
COMPARATOR_PAIRS = [
    ("inst_delta_z", "coinvest_score_z", "institutional-flow"),
    ("score_rank_pct", "clinical_optionality_pct_dev", "composite-vs-clinical"),
]


def _sf(val: Any) -> float:
    if val is None or val == "":
        return math.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return math.nan


def _is_promoted(name: str) -> bool:
    return len(name) == 10 and not name.startswith("_") and name != "state"


def _find_prior_snapshots(snapshots_dir: Path, current_date: str, n: int) -> List[str]:
    candidates = sorted(
        d.name for d in snapshots_dir.iterdir() if d.is_dir() and _is_promoted(d.name) and d.name < current_date
    )
    return candidates[-n:] if len(candidates) >= n else candidates


def _load_signal_values(snap_dir: Path, signal_field: str) -> Dict[str, float]:
    path = snap_dir / "rankings.csv"
    if not path.exists():
        return {}
    values = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ticker = row.get("ticker", "")
            val = _sf(row.get(signal_field, ""))
            if ticker and not math.isnan(val):
                values[ticker] = val
    return values


def _load_prices(price_csv: Path, tickers: set, start_date: str) -> Dict[str, Dict[str, float]]:
    prices: Dict[str, Dict[str, float]] = {}
    if not price_csv.exists():
        return prices
    with open(price_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = row.get("ticker", "")
            d = row.get("date", "")
            if t in tickers and d >= start_date:
                c = _sf(row.get("close", ""))
                if not math.isnan(c):
                    prices.setdefault(t, {})[d] = c
    return prices


def _get_forward_return(
    ticker: str,
    snap_date: str,
    prices: Dict[str, Dict[str, float]],
    horizon: int,
) -> float:
    tp = prices.get(ticker, {})
    if not tp:
        return math.nan
    sorted_dates = sorted(tp.keys())
    entry_dates = [d for d in sorted_dates if d >= snap_date]
    if not entry_dates:
        return math.nan
    entry_date = entry_dates[0]
    entry_price = tp[entry_date]
    future_dates = [d for d in sorted_dates if d > entry_date]
    if len(future_dates) < horizon:
        return math.nan
    exit_date = future_dates[min(horizon - 1, len(future_dates) - 1)]
    exit_price = tp[exit_date]
    if entry_price <= 0:
        return math.nan
    return (exit_price - entry_price) / entry_price


def compute_ic(
    signal_values: Dict[str, float],
    forward_returns: Dict[str, float],
    higher_is_better: bool,
) -> Tuple[float, int]:
    from scipy import stats

    common = set(signal_values.keys()) & set(forward_returns.keys())
    common = {t for t in common if not math.isnan(forward_returns[t])}

    if len(common) < MIN_N_OBS:
        return math.nan, len(common)

    tickers = sorted(common)
    sig = [signal_values[t] for t in tickers]
    ret = [forward_returns[t] for t in tickers]

    if not higher_is_better:
        sig = [-s for s in sig]

    ic, _ = stats.spearmanr(sig, ret)
    return ic, len(common)


def run_probe(
    snapshots_dir: Path,
    price_csv: Path,
    flag_signal: str,
    comp_signal: str,
    per_date_windows: List[Dict[str, Any]],
    horizon: int,
    flag_hib: bool,
    comp_hib: bool,
) -> Dict[str, Any]:
    """Compute IC for flagged + comparator signals over identical window dates.

    Returns dict with per_date IC arrays for both signals and cross-signal statistics.
    """
    # Collect all tickers across the date window for price loading
    all_tickers: set = set()
    for pd_entry in per_date_windows:
        snap_date = pd_entry["date"]
        snap_path = snapshots_dir / snap_date
        for field in [flag_signal, comp_signal]:
            sig_vals = _load_signal_values(snap_path, field)
            all_tickers.update(sig_vals.keys())

    # Load prices
    window_start = per_date_windows[0]["date"]
    prices = _load_prices(price_csv, all_tickers, window_start)

    # Compute per-date IC for both signals
    flag_ics: List[Dict[str, Any]] = []
    comp_ics: List[Dict[str, Any]] = []

    for pd_entry in per_date_windows:
        snap_date = pd_entry["date"]
        snap_path = snapshots_dir / snap_date

        # Flagged signal
        sig_vals_f = _load_signal_values(snap_path, flag_signal)
        if len(sig_vals_f) >= MIN_N_OBS:
            fwd_f = {}
            for ticker in sig_vals_f:
                ret = _get_forward_return(ticker, snap_date, prices, horizon)
                if not math.isnan(ret):
                    fwd_f[ticker] = ret
            ic_f, n_f = compute_ic(sig_vals_f, fwd_f, flag_hib)
            if not math.isnan(ic_f):
                flag_ics.append({"date": snap_date, "ic": round(ic_f, 4), "n_obs": n_f})

        # Comparator signal
        sig_vals_c = _load_signal_values(snap_path, comp_signal)
        if len(sig_vals_c) >= MIN_N_OBS:
            fwd_c = {}
            for ticker in sig_vals_c:
                ret = _get_forward_return(ticker, snap_date, prices, horizon)
                if not math.isnan(ret):
                    fwd_c[ticker] = ret
            ic_c, n_c = compute_ic(sig_vals_c, fwd_c, comp_hib)
            if not math.isnan(ic_c):
                comp_ics.append({"date": snap_date, "ic": round(ic_c, 4), "n_obs": n_c})

    # Align by common dates
    flag_by_date = {d["date"]: d for d in flag_ics}
    comp_by_date = {d["date"]: d for d in comp_ics}
    common_dates = sorted(set(flag_by_date.keys()) & set(comp_by_date.keys()))

    aligned = []
    flag_ic_vals = []
    comp_ic_vals = []
    for dt in common_dates:
        f_ic = flag_by_date[dt]["ic"]
        c_ic = comp_by_date[dt]["ic"]
        aligned.append(
            {
                "date": dt,
                f"{flag_signal}_ic": f_ic,
                f"{comp_signal}_ic": c_ic,
            }
        )
        flag_ic_vals.append(f_ic)
        comp_ic_vals.append(c_ic)

    # Summary stats
    flag_mean = sum(flag_ic_vals) / len(flag_ic_vals) if flag_ic_vals else math.nan
    comp_mean = sum(comp_ic_vals) / len(comp_ic_vals) if comp_ic_vals else math.nan

    # Cross-signal Spearman and Pearson
    rho, rho_p = stats.spearmanr(flag_ic_vals, comp_ic_vals)
    r_pearson = stats.pearsonr(flag_ic_vals, comp_ic_vals)

    result = {
        "flagged_signal": flag_signal,
        "comparator_signal": comp_signal,
        "lane": COMPARATOR_PAIRS[0][2] if flag_signal == "inst_delta_z" else "composite-vs-clinical",
        "horizon": horizon,
        "n_dates_flagged": len(flag_ics),
        "n_dates_comparator": len(comp_ics),
        "n_dates_aligned": len(common_dates),
        "window_start": per_date_windows[0]["date"],
        "window_end": per_date_windows[-1]["date"],
        f"{flag_signal}_mean_ic": round(flag_mean, 4),
        f"{comp_signal}_mean_ic": round(comp_mean, 4),
        "spearman_rho": round(rho, 4),
        "spearman_p": round(float(rho_p), 6),
        "pearson_r": round(float(r_pearson[0]), 4),
        "pearson_p": round(float(r_pearson[1]), 6),
        "aligned_ics": aligned,
        "interpretation": None,  # filled by caller
    }

    return result


def interpret_result(r: Dict[str, Any]) -> str:
    flag_mean = r.get(f"{r['flagged_signal']}_mean_ic", 0)
    comp_mean = r.get(f"{r['comparator_signal']}_mean_ic", 0)
    rho = r.get("spearman_rho", 0)

    flag_negative = flag_mean < -0.01
    comp_positive = comp_mean > 0.01

    if flag_negative and comp_positive and rho < 0:
        return "signal_specific_failure"
    elif flag_negative and comp_mean < -0.01 and rho > 0.5:
        return "shared_regime_failure"
    elif flag_negative and comp_mean < -0.01 and abs(rho) < 0.3:
        return "independent_failures"
    elif flag_negative and comp_positive:
        return "signal_specific_failure"  # lane healthy, signal broken (low rho not needed)
    else:
        return "ambiguous"


def main():
    parser = argparse.ArgumentParser(description="Shared-regime comparator IC check")
    parser.add_argument("--as-of-date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument(
        "--dashboard",
        type=Path,
        required=True,
        help="Path to dashboard JSON (e.g. artifacts/ic_dashboard/2026-05-20_dashboard.json)",
    )
    parser.add_argument("--snapshots-dir", type=Path, default=REPO_ROOT / "data" / "snapshots")
    from tools.build_ic_dashboard import default_price_csv

    parser.add_argument(
        "--price-csv",
        type=Path,
        default=default_price_csv(),
        help="Default: production_data/price_history_split_adj.csv, else price_history.csv",
    )
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    args = parser.parse_args()

    as_of_date = args.as_of_date

    # Load dashboard
    if not args.dashboard.exists():
        print(f"ERROR: Dashboard not found: {args.dashboard}")
        sys.exit(1)

    dash = json.loads(args.dashboard.read_text(encoding="utf-8"))
    signals = dash.get("signals", {})

    # Identify flagged signals (ALERT or WARN)
    flagged = []
    for sig_name, sig_data in signals.items():
        health = sig_data.get("health", "")
        if health in ("ALERT", "WARN"):
            flagged.append(sig_name)
            print(f"  FLAGGED: {sig_name} (health={health}, mean_ic={sig_data.get('mean_ic', 'N/A')})")

    if not flagged:
        print("No signals at WARN/ALERT this week.")
        return

    # Determine comparator pairs based on what's flagged
    pairs_to_run = []
    for flag_name in flagged:
        for pair in COMPARATOR_PAIRS:
            if pair[0] == flag_name:
                pairs_to_run.append(pair)
                break

    if not pairs_to_run:
        print(f"No comparator defined for flagged signals: {flagged}")
        print("Need to extend COMPARATOR_PAIRS in the script.")
        sys.exit(1)

    horizon = args.horizon if args.horizon else dash.get("horizon", DEFAULT_HORIZON)

    results = []
    for flag_signal, comp_signal, lane in pairs_to_run:
        print(f"\n{'='*60}")
        print(f"PROBE: {flag_signal} (flagged) vs {comp_signal} (comparator)")
        print(f"Lane: {lane}")
        print(f"{'='*60}")

        # Extract window from dashboard per_date arrays
        sig_data = signals.get(flag_signal, {})
        per_date = sig_data.get("per_date", [])
        if not per_date:
            print(f"  ERROR: No per_date data for {flag_signal}")
            continue

        window_start = per_date[0]["date"]
        window_end = per_date[-1]["date"]
        print(f"  Window: {window_start} -> {window_end} ({len(per_date)} dates)")

        # Determine higher_is_better
        flag_hib = SIGNAL_HIB.get(flag_signal, True)
        comp_hib = SIGNAL_HIB.get(comp_signal, True)
        print(f"  higher_is_better: {flag_signal}={flag_hib}, {comp_signal}={comp_hib}")

        # Run probe
        result = run_probe(
            snapshots_dir=args.snapshots_dir,
            price_csv=args.price_csv,
            flag_signal=flag_signal,
            comp_signal=comp_signal,
            per_date_windows=per_date,
            horizon=horizon,
            flag_hib=flag_hib,
            comp_hib=comp_hib,
        )

        # Fidelity check: compare flagged signal mean_ic to dashboard value
        dash_mean = sig_data.get("mean_ic")
        my_mean = result.get(f"{flag_signal}_mean_ic")
        if dash_mean is not None and my_mean is not None:
            delta = abs(my_mean - dash_mean)
            if delta > 0.020:
                print(f"  ❌ FIDELITY FAIL: dashboard mean_ic={dash_mean}, probe mean_ic={my_mean}, delta={delta:.4f}")
                print("     Methodology mismatch — DO NOT TRUST comparator finding.")
                # Still record, but flag the fidelity issue
                result["fidelity_ok"] = False
                result["fidelity_delta"] = round(delta, 4)
            else:
                print(f"  ✅ FIDELITY OK: dashboard mean_ic={dash_mean}, probe mean_ic={my_mean}, delta={delta:.4f}")
                result["fidelity_ok"] = True
                result["fidelity_delta"] = round(delta, 4)
        else:
            result["fidelity_ok"] = None
            result["fidelity_delta"] = None

        # Interpret
        interp = interpret_result(result)
        result["interpretation"] = interp
        print(f"  Interpretation: {interp}")
        print(f"  {flag_signal}_mean_ic: {result[f'{flag_signal}_mean_ic']}")
        print(f"  {comp_signal}_mean_ic: {result[f'{comp_signal}_mean_ic']}")
        print(f"  Spearman ρ: {result['spearman_rho']} (p={result['spearman_p']})")
        print(f"  Pearson r: {result['pearson_r']} (p={result['pearson_p']})")
        print(f"  Aligned dates: {result['n_dates_aligned']}")

        results.append(result)

    # Save artifact JSON
    artifact_dir = REPO_ROOT / "artifacts" / "ic_dashboard"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    json_path = artifact_dir / f"{as_of_date}_shared_regime_check.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema": SCHEMA_VERSION,
                "as_of_date": as_of_date,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "dashboard_source": str(args.dashboard.name),
                "results": results,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nWrote artifact: {json_path}")

    # Write markdown summary at repo root
    md_lines = []
    md_lines.append(f"# Weekly Signal Regime Sweep — {as_of_date}")
    md_lines.append("")
    md_lines.append(
        f"**Dashboard:** {args.dashboard.name} | **Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    md_lines.append("")
    md_lines.append("## Summary")
    md_lines.append("")
    n_actionable = len(
        [
            r
            for r in results
            if r.get("fidelity_ok", False)
            and r.get("interpretation", "") in ("signal_specific_failure", "shared_regime_failure")
        ]
    )
    n_ambiguous = len([r for r in results if r.get("interpretation") == "ambiguous"])
    n_fidelity_fail = len([r for r in results if not r.get("fidelity_ok", True)])
    md_lines.append(
        f"**Fleet verdict:** {n_actionable} actionable findings, {n_ambiguous} ambiguous, {n_fidelity_fail} fidelity failures"
    )
    md_lines.append("")

    for r in results:
        md_lines.append(f"### {r['flagged_signal']} vs {r['comparator_signal']}")
        md_lines.append("")
        md_lines.append(f"**Lane:** {r['lane']}")
        md_lines.append(f"**Window:** {r['window_start']} → {r['window_end']} ({r['n_dates_aligned']} aligned dates)")
        md_lines.append(f"**Horizon:** {r['horizon']}d")
        md_lines.append("")

        # Interpretation badge
        interp = r.get("interpretation", "unknown")
        fidelity = r.get("fidelity_ok", False)
        if fidelity:
            if interp == "signal_specific_failure":
                badge = "❗ SIGNAL-SPECIFIC FAILURE"
            elif interp == "shared_regime_failure":
                badge = "⚠️ SHARED-REGIME FAILURE"
            elif interp == "independent_failures":
                badge = "🔶 INDEPENDENT FAILURES"
            else:
                badge = "❓ AMBIGUOUS"
        else:
            badge = "❌ FIDELITY FAIL — DO NOT USE"
        md_lines.append(f"**Verdict:** {badge}")
        md_lines.append("")

        f_sig = r["flagged_signal"]
        c_sig = r["comparator_signal"]
        f_mean_key = f"{f_sig}_mean_ic"
        c_mean_key = f"{c_sig}_mean_ic"
        f_mean = r.get(f_mean_key, 0)
        c_mean = r.get(c_mean_key, 0)

        md_lines.append("| Metric | Flagged | Comparator |")
        md_lines.append("|--------|---------|------------|")
        md_lines.append(f"| Mean IC | {f_mean:+.4f} | {c_mean:+.4f} |")
        md_lines.append(f"| N dates | {r['n_dates_flagged']} | {r['n_dates_comparator']} |")
        md_lines.append(f"| Cross-signal Spearman ρ | {r['spearman_rho']:+.4f} (p={r['spearman_p']}) | |")
        md_lines.append(f"| Cross-signal Pearson r | {r['pearson_r']:+.4f} (p={r['pearson_p']}) | |")
        if r.get("fidelity_delta") is not None:
            md_lines.append(f"| Dashboard fidelity delta | {r['fidelity_delta']:.4f} | |")
        md_lines.append("")

        # Interpretation detail
        md_lines.append("#### Interpretation")
        md_lines.append("")
        if interp == "signal_specific_failure":
            md_lines.append(f"- {f_sig} mean_ic is negative ({f_mean:+.4f})")
            md_lines.append(f"- {c_sig} mean_ic is positive ({c_mean:+.4f}) — the lane is healthy")
            low_or_mod = "low" if abs(r["spearman_rho"]) < 0.3 else "moderate"
            md_lines.append(f"- Cross-signal ρ={r['spearman_rho']:+.4f} indicates {low_or_mod} correlation")
            md_lines.append(
                "- **Conclusion: Signal-specific degradation.** The comparator in the same lane is performing well."
            )
            md_lines.append("- Governance path: signal-health review, possible weight reduction or zero-out.")
        elif interp == "shared_regime_failure":
            md_lines.append(f"- Both {f_sig} ({f_mean:+.4f}) and {c_sig} ({c_mean:+.4f}) are negative")
            md_lines.append(f"- Cross-signal ρ={r['spearman_rho']:+.4f} > 0.5 indicates shared regime")
            md_lines.append("- **Conclusion: Shared-regime failure.** The entire lane is degraded.")
            md_lines.append(
                "- Governance path: lane-level investigation; weight reduction for the block, not individual signals."
            )
        elif interp == "independent_failures":
            md_lines.append(f"- Both signals are negative but cross-signal ρ ≈ 0 ({r['spearman_rho']:+.4f})")
            md_lines.append("- **Conclusion: Independent failures.** Both signals broken for different reasons.")
            md_lines.append("- Investigate separately.")
        else:
            md_lines.append(f"- Mean_ICs: flagged={f_mean:+.4f}, comparator={c_mean:+.4f}, ρ={r['spearman_rho']:+.4f}")
            md_lines.append("- **Conclusion: Ambiguous.** Insufficient signal to classify.")

        md_lines.append("")
        md_lines.append("#### Aligned IC Trajectories (first 10)")

        aligned = r.get("aligned_ics", [])
        md_lines.append("")
        md_lines.append("| Date | Flagged IC | Comparator IC |")
        md_lines.append("|------|-----------|---------------|")
        for entry in aligned[:10]:
            f_ic_key = f"{f_sig}_ic"
            c_ic_key = f"{c_sig}_ic"
            md_lines.append(f"| {entry['date']} | {entry.get(f_ic_key, 0):+.4f} | {entry.get(c_ic_key, 0):+.4f} |")
        if len(aligned) > 10:
            md_lines.append(f"| … | ({len(aligned) - 10} more dates) | |")
        md_lines.append("")

    # All signals report
    md_lines.append("## All Signals Status")
    md_lines.append("")
    md_lines.append("| Signal | Mean IC | Hit Rate | Health | N Dates |")
    md_lines.append("|--------|---------|----------|--------|---------|")
    for sig_name, sig_data in sorted(signals.items()):
        mn = f"{sig_data['mean_ic']:+.4f}" if sig_data["mean_ic"] is not None else "n/a"
        hr = f"{sig_data['hit_rate']:.1%}" if sig_data["hit_rate"] is not None else "n/a"
        hl = sig_data.get("health", "n/a")
        nd = sig_data.get("n_dates", "n/a")
        md_lines.append(f"| {sig_name} | {mn} | {hr} | {hl} | {nd} |")
    md_lines.append("")

    md_lines.append("## What This Does NOT Prove")
    md_lines.append("")
    md_lines.append(
        "- Does not recommend specific weight changes. Governance review required before any selector weight modification."
    )
    md_lines.append(
        "- Does not rule out structural regime shift in the broader market (sector-wide factor degradation)."
    )
    md_lines.append(
        "- Does not validate the comparator signal as investable — only assesses whether both degrade together."
    )
    md_lines.append(
        "- The probe n_dates (with settled 20d forward returns) may be smaller than the dashboard n_dates. This is expected."
    )
    md_lines.append("")
    md_lines.append("## Provenance")
    md_lines.append("")
    md_lines.append(f"- **Methodology:** `tools/build_ic_dashboard.py` (Spearman ρ, horizon={horizon}, min_n=10)")
    md_lines.append("- **Script:** `scripts/shared_regime_check.py`")
    md_lines.append(f"- **Dashboard:** `{args.dashboard.name}`")
    md_lines.append(f"- **Price source:** `production_data/{Path(args.price_csv).name}`")
    md_lines.append("- **Generated by:** Hermes Agent (signal-shared-regime-check skill)")
    md_lines.append(f"- **Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    md_path = REPO_ROOT / f"WEEKLY_SIGNAL_REGIME_SWEEP_{as_of_date}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote memo: {md_path}")


if __name__ == "__main__":
    main()
