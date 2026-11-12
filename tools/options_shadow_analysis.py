#!/usr/bin/env python3
"""Options shadow layer analysis — IC test + autopsy cross-reference.

Three analyses in one pass:
  1. RV IC from price history — full historical test (properly powered)
  2. Market-implied cross-section — single snapshot correlations (growing over time)
  3. Autopsy cross-reference — options profile of top-30 during failure windows

Classification:
    OPTIONS_IC_SHADOW / NO_MODEL_CHANGE / NO_RANKER_CHANGE /
    NO_SELECTOR_CHANGE / NO_SIZING_CHANGE / NO_TRADING_CHANGE

Usage:
    python3 tools/options_shadow_analysis.py
    python3 tools/options_shadow_analysis.py --snap-date 2026-06-26 --output-dir artifacts/options_shadow_analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CLASSIFICATION = (
    "OPTIONS_IC_SHADOW/NO_MODEL_CHANGE/NO_RANKER_CHANGE/" "NO_SELECTOR_CHANGE/NO_SIZING_CHANGE/NO_TRADING_CHANGE"
)

FAILURE_WINDOWS = [
    ("2026-03-16", "2026-03-23", "HARD_FAIL"),
    ("2026-04-20", "2026-04-27", "HARD_FAIL"),
    ("2026-05-04", "2026-05-11", "SOFT_FAIL"),
    ("2026-05-19", "2026-05-26", "SOFT_FAIL"),
    ("2026-05-26", "2026-06-02", "HARD_FAIL"),
    ("2026-06-01", "2026-06-08", "SOFT_FAIL"),
    ("2026-06-08", "2026-06-15", "SOFT_FAIL"),
]

SNAP_DIR = REPO / "data" / "snapshots"
PRICE_CSV = REPO / "production_data" / "price_history_split_adj.csv"


# ── helpers ──────────────────────────────────────────────────────────────────


def _ic(x: pd.Series, y: pd.Series) -> float | None:
    """Spearman rank IC, requiring >= 10 paired obs."""
    paired = pd.concat([x, y], axis=1).dropna()
    if len(paired) < 10:
        return None
    return float(stats.spearmanr(paired.iloc[:, 0], paired.iloc[:, 1]).statistic)


def _ic_series_summary(ic_vals: list[float]) -> dict:
    arr = np.array(ic_vals)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    t = float(mean / (std / np.sqrt(len(arr)))) if std > 0 else 0.0
    hit = float((arr > 0).mean())
    return {
        "n_obs": len(arr),
        "mean_ic": round(mean, 4),
        "std_ic": round(std, 4),
        "t_stat": round(t, 3),
        "hit_rate": round(hit, 3),
        "ir": round(mean / std, 3) if std > 0 else None,
    }


# ── Analysis 1: RV IC from price history ─────────────────────────────────────


def rv_ic_from_price_history(price_df: pd.DataFrame, forward_days: list[int]) -> dict:
    """
    Compute rolling RV_20d and RV_60d for each date, then IC vs N-day forward returns.
    Uses the full price history — properly powered.
    """
    wide = price_df.pivot(index="date", columns="ticker", values="close").sort_index()
    log_ret = np.log(wide / wide.shift(1))

    results = {}
    for fwd in forward_days:
        fwd_ret = wide.shift(-fwd) / wide - 1  # N-day forward return

        ic_rv20, ic_rv60 = [], []
        dates = wide.index[120:-fwd]  # need 120d lookback + fwd buffer

        for dt in dates:
            # RV_20d and RV_60d annualized
            rv20 = log_ret.loc[:dt].tail(20).std() * np.sqrt(252)
            rv60 = log_ret.loc[:dt].tail(60).std() * np.sqrt(252)
            fwd_x = fwd_ret.loc[dt]

            ic20 = _ic(rv20, fwd_x)
            ic60 = _ic(rv60, fwd_x)
            if ic20 is not None:
                ic_rv20.append(ic20)
            if ic60 is not None:
                ic_rv60.append(ic60)

        results[f"rv20d_fwd{fwd}d"] = _ic_series_summary(ic_rv20) if ic_rv20 else None
        results[f"rv60d_fwd{fwd}d"] = _ic_series_summary(ic_rv60) if ic_rv60 else None

    return results


# ── Analysis 2: Market-implied cross-section (single snapshot) ────────────────


def market_implied_cross_section(enrich_df: pd.DataFrame, price_df: pd.DataFrame, snap_date: str) -> dict:
    """
    Single cross-section IC: options features vs available forward returns.
    With only one snapshot, returns Spearman rho (not a time-series IC).
    """
    wide = price_df.pivot(index="date", columns="ticker", values="close")
    snap_dt = pd.Timestamp(snap_date)

    results = {}
    for fwd in [5, 10, 20]:
        target_dt = snap_dt + pd.offsets.BDay(fwd)
        if target_dt.strftime("%Y-%m-%d") > wide.index.max():
            results[f"fwd{fwd}d"] = {"status": "insufficient_forward_data", "snap_date": snap_date}
            continue

        # price_df dates are "YYYY-MM-DD" strings, so the pivot index is string-typed:
        # look up by the same string key used in the membership test. Indexing with
        # the Timestamp raised KeyError on every run that had forward data.
        snap_key = snap_dt.strftime("%Y-%m-%d")
        target_key = target_dt.strftime("%Y-%m-%d")
        base = wide.loc[snap_key] if snap_key in wide.index else None
        fwd_close = wide.loc[target_key] if target_key in wide.index else None
        if base is None or fwd_close is None:
            results[f"fwd{fwd}d"] = {"status": "price_not_found", "snap_date": snap_date}
            continue

        fwd_ret = (fwd_close / base - 1).rename("fwd_ret")
        merged = enrich_df.set_index("ticker").join(fwd_ret, how="inner")

        fwd_obs = {}
        for feat in [
            "observed_iv_rank_tw",
            "variance_risk_premium_20d",
            "idiosyncratic_vol_60d",
            "rv_20d",
            "rv_60d",
            "event_premium_ratio",
            "iv_over_rv_20d",
        ]:
            if feat in merged.columns:
                rho = _ic(merged[feat], merged["fwd_ret"])
                fwd_obs[feat] = round(rho, 4) if rho is not None else None

        results[f"fwd{fwd}d"] = {
            "n_pairs": int(merged["fwd_ret"].notna().sum()),
            "snap_date": snap_date,
            "correlations": fwd_obs,
            "note": "single_cross_section — accumulate snapshots for IC time series",
        }

    return results


# ── Analysis 3: Autopsy cross-reference ──────────────────────────────────────


def autopsy_options_crossref(enrich_df: pd.DataFrame) -> list[dict]:
    """
    For each failure window, load top-30 names from the snapshot rankings.csv
    and check their current options profile.
    """
    enrich_idx = enrich_df.set_index("ticker")
    option_cols = [
        "options_chain_status",
        "observed_iv_regime",
        "observed_iv_rank_tw",
        "event_premium_flag",
        "vrp_flag",
        "options_flags",
        "rv_60d",
        "idiosyncratic_vol_60d",
        "atm_straddle_move_pct",
    ]

    rows = []
    for snap_date, fwd_date, severity in FAILURE_WINDOWS:
        rank_csv = SNAP_DIR / snap_date / "rankings.csv"
        if not rank_csv.exists():
            rows.append({"window": f"{snap_date}→{fwd_date}", "severity": severity, "status": "rankings_missing"})
            continue

        try:
            rdf = pd.read_csv(rank_csv)
        except Exception as e:
            rows.append({"window": f"{snap_date}→{fwd_date}", "status": f"read_error: {e}"})
            continue

        # Find the rank column
        rank_col = next((c for c in ["actionable_rank", "rank", "final_rank"] if c in rdf.columns), None)
        tick_col = next((c for c in ["ticker", "symbol"] if c in rdf.columns), None)
        if rank_col is None or tick_col is None:
            rows.append({"window": f"{snap_date}→{fwd_date}", "status": "no_rank_or_ticker_col"})
            continue

        top30 = rdf.nsmallest(30, rank_col)[tick_col].tolist()

        # Pull forward return from price history (if available)
        window_row: dict = {
            "window": f"{snap_date}→{fwd_date}",
            "severity": severity,
            "n_top30": len(top30),
        }

        # Summarize current options profile for these names
        matched = [t for t in top30 if t in enrich_idx.index]
        window_row["n_matched_current_options"] = len(matched)

        if matched:
            sub = enrich_idx.loc[matched, [c for c in option_cols if c in enrich_idx.columns]]
            flags_series = sub.get("options_flags", pd.Series([], dtype=str))
            all_flags: list[str] = []
            for f in flags_series.dropna():
                all_flags.extend(str(f).split("|"))

            from collections import Counter

            flag_counts = Counter(all_flags)

            window_row["flag_summary"] = dict(flag_counts.most_common(10))
            window_row["n_extreme_iv"] = int((sub.get("observed_iv_regime", pd.Series()) == "EXTREME").sum())
            window_row["n_event_premium_high"] = int(
                (sub.get("event_premium_flag", pd.Series()) == "EVENT_PREMIUM_HIGH").sum()
            )
            window_row["n_high_vrp"] = int((sub.get("vrp_flag", pd.Series()) == "HIGH_VRP").sum())
            window_row["median_iv_rank_tw"] = (
                round(float(sub.get("observed_iv_rank_tw", pd.Series(dtype=float)).dropna().median()), 3)
                if not sub.get("observed_iv_rank_tw", pd.Series(dtype=float)).dropna().empty
                else None
            )
            window_row["median_rv60d"] = (
                round(float(sub.get("rv_60d", pd.Series(dtype=float)).dropna().median()), 3)
                if not sub.get("rv_60d", pd.Series(dtype=float)).dropna().empty
                else None
            )

            # Names with no options coverage in current snapshot
            no_cov = (
                [
                    t
                    for t in matched
                    if enrich_idx.loc[t, "options_chain_status"] in ("NO_LISTED_OPTIONS", "FETCH_FAILED")
                ]
                if "options_chain_status" in enrich_idx.columns
                else []
            )
            window_row["no_options_coverage_now"] = no_cov

        rows.append(window_row)

    return rows


# ── Report builder ────────────────────────────────────────────────────────────


def build_report(rv_ic: dict, xs: dict, autopsy: list[dict], snap_date: str) -> str:
    lines = [
        "# Options Shadow Layer Analysis",
        "",
        f"**Snap date:** {snap_date}  |  **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"**Classification:** `{CLASSIFICATION}`",
        "",
        "> Shadow diagnostics only. No options-derived feature affects final_score, actionable_rank,",
        "> eligibility, sizing, or trading.",
        "",
        "---",
        "",
        "## 1. RV IC from Price History (properly powered)",
        "",
        "Spearman IC of annualized realized volatility vs N-day forward return.",
        f"Full price history used (~{len(rv_ic)} feature-horizon pairs).",
        "",
        "| Feature | Horizon | N obs | Mean IC | Std | t-stat | Hit Rate | IR |",
        "|---------|---------|-------|---------|-----|--------|----------|----|",
    ]
    for key, v in sorted(rv_ic.items()):
        if v is None:
            continue
        feat, hor = key.replace("_fwd", " → fwd").split(" → ")
        lines.append(
            f"| {feat} | {hor} | {v['n_obs']} | {v['mean_ic']:+.4f} | {v['std_ic']:.4f} "
            f"| {v['t_stat']:+.3f} | {v['hit_rate']:.1%} | {v['ir'] or 'N/A'} |"
        )

    lines += [
        "",
        "*Positive IC = higher RV → better forward return. Negative IC = higher RV → worse forward return.*",
        "",
        "---",
        "",
        f"## 2. Market-Implied Cross-Section (single snapshot — {snap_date})",
        "",
        "Spearman ρ of options features vs N-day forward returns. Single data point — accumulate snapshots for IC time series.",
        "",
    ]
    for hor, v in xs.items():
        if "status" in v:
            lines.append(f"**{hor}:** {v['status']} (snap: {v.get('snap_date', '')})")
            continue
        lines.append(f"**{hor}** ({v['n_pairs']} pairs):")
        lines.append("")
        lines.append("| Feature | Spearman ρ |")
        lines.append("|---------|------------|")
        for feat, rho in (v.get("correlations") or {}).items():
            lines.append(f"| {feat} | {rho:+.4f} |" if rho is not None else f"| {feat} | N/A |")
        lines.append("")

    lines += [
        "---",
        "",
        "## 3. Autopsy Cross-Reference",
        "",
        "Current options profile of top-30 names during each failure window.",
        "",
        "| Window | Severity | Matched | Extreme IV | Event Prem High | High VRP | Med IV Rank | Med RV60d |",
        "|--------|----------|---------|------------|-----------------|----------|-------------|-----------|",
    ]
    for r in autopsy:
        if "status" in r:
            lines.append(f"| {r.get('window', '?')} | — | — | — | — | — | — | {r['status']} |")
            continue
        lines.append(
            f"| {r['window']} | {r['severity']} | {r['n_matched_current_options']} | "
            f"{r.get('n_extreme_iv', '—')} | {r.get('n_event_premium_high', '—')} | "
            f"{r.get('n_high_vrp', '—')} | {r.get('median_iv_rank_tw', '—')} | {r.get('median_rv60d', '—')} |"
        )
    lines.append("")
    lines.append("### Flag Frequency Across Failure Window Names (current snapshot)")
    lines.append("")
    from collections import Counter

    all_flags: Counter = Counter()
    for r in autopsy:
        if "flag_summary" in r:
            all_flags.update(r["flag_summary"])
    if all_flags:
        lines.append("| Flag | Count |")
        lines.append("|------|-------|")
        for flag, cnt in all_flags.most_common(15):
            lines.append(f"| {flag} | {cnt} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"*Generated by `tools/options_shadow_analysis.py` — {CLASSIFICATION}*")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────


def run(snap_date: str, output_dir: Path, verbose: bool, fast_mode: bool = False) -> dict:
    """Run the options shadow analysis.

    fast_mode=True skips the expensive RV IC computation (static between daily runs).
    Use fast_mode for the production pipeline step; run full mode weekly or on-demand.
    """
    if verbose:
        print("Loading price history...", flush=True)
    price_df = pd.read_csv(PRICE_CSV)
    price_df["date"] = pd.to_datetime(price_df["date"]).dt.strftime("%Y-%m-%d")

    # Load enrichment for the given snap date
    enrich_date = snap_date.replace("-", "_")
    enrich_path = REPO / "artifacts" / "options_enrichment" / enrich_date / "options_enriched_features.csv"
    if not enrich_path.exists():
        # Fall back to latest available
        all_enrich = sorted((REPO / "artifacts" / "options_enrichment").glob("*/options_enriched_features.csv"))
        if not all_enrich:
            raise FileNotFoundError("No options enrichment CSV found — run enrich_options_features.py first")
        enrich_path = all_enrich[-1]
        if verbose:
            print(f"Using fallback enrichment: {enrich_path}", flush=True)

    enrich_df = pd.read_csv(enrich_path)
    if verbose:
        print(f"Enrichment: {len(enrich_df)} tickers from {enrich_path.parent.name}", flush=True)

    # 1. RV IC from price history — skipped in fast_mode (static, run weekly/on-demand)
    rv_ic: dict = {}
    if not fast_mode:
        if verbose:
            print("Computing RV IC from price history (this takes ~30s)...", flush=True)
        wide = price_df.pivot(index="date", columns="ticker", values="close").sort_index()
        log_ret = np.log(wide / wide.shift(1))
        for fwd in [5, 20]:
            fwd_ret = wide.shift(-fwd) / wide - 1
            ic_rv20, ic_rv60 = [], []
            dates = wide.index[120:-fwd]
            step = 5  # sample every 5 days to keep it fast
            for dt in dates[::step]:
                rv20 = log_ret.loc[:dt].tail(20).std() * np.sqrt(252)
                rv60 = log_ret.loc[:dt].tail(60).std() * np.sqrt(252)
                fwd_x = fwd_ret.loc[dt]
                ic20 = _ic(rv20, fwd_x)
                ic60 = _ic(rv60, fwd_x)
                if ic20 is not None:
                    ic_rv20.append(ic20)
                if ic60 is not None:
                    ic_rv60.append(ic60)
            rv_ic[f"rv20d_fwd{fwd}d"] = _ic_series_summary(ic_rv20) if ic_rv20 else None
            rv_ic[f"rv60d_fwd{fwd}d"] = _ic_series_summary(ic_rv60) if ic_rv60 else None
            if verbose:
                for k in [f"rv20d_fwd{fwd}d", f"rv60d_fwd{fwd}d"]:
                    v = rv_ic[k]
                    if v:
                        print(f"  {k}: mean_ic={v['mean_ic']:+.4f} t={v['t_stat']:+.3f} n={v['n_obs']}", flush=True)

    # 2. Market-implied cross-section
    if verbose:
        print("Market-implied cross-section...", flush=True)
    xs = market_implied_cross_section(enrich_df, price_df, snap_date)
    for hor, v in xs.items():
        if verbose:
            if "status" in v:
                print(f"  {hor}: {v['status']}", flush=True)
            else:
                print(f"  {hor}: {v['n_pairs']} pairs, correlations: {v.get('correlations', {})}", flush=True)

    # 3. Autopsy cross-reference
    if verbose:
        print("Autopsy cross-reference...", flush=True)
    autopsy = autopsy_options_crossref(enrich_df)

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(rv_ic, xs, autopsy, snap_date)
    (output_dir / "OPTIONS_SHADOW_ANALYSIS.md").write_text(report)

    summary = {
        "as_of_date": snap_date,
        "generated_at": datetime.now().isoformat(),
        "classification": CLASSIFICATION,
        "rv_ic": rv_ic,
        "market_implied_xs": xs,
        "autopsy_crossref": autopsy,
    }
    (output_dir / "options_shadow_analysis.json").write_text(json.dumps(summary, indent=2, default=str))

    if verbose:
        print(f"\nWritten to {output_dir}/", flush=True)
        print("  OPTIONS_SHADOW_ANALYSIS.md")
        print("  options_shadow_analysis.json")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Options shadow layer analysis")
    parser.add_argument("--snap-date", default=date.today().isoformat())
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--fast", action="store_true", help="Skip RV IC (static); run market-implied XS + autopsy only")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    out = (
        Path(args.output_dir)
        if args.output_dir
        else (REPO / "artifacts" / "options_shadow_analysis" / args.snap_date.replace("-", "_"))
    )
    run(args.snap_date, out, verbose=not args.quiet, fast_mode=args.fast)


if __name__ == "__main__":
    main()
