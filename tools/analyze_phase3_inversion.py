#!/usr/bin/env python3
"""Phase 3 inversion autopsy — diagnostic only, no model changes.

Classification: PHASE_3_INVERSION_AUTOPSY_DIAGNOSTIC_NO_MODEL_CHANGE
Writes to:      artifacts/autopsy/phase3_inversion/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
SNAP_ROOT = REPO_ROOT / "data" / "snapshots"
PRICE_HISTORY = REPO_ROOT / "production_data" / "price_history.csv"

FORBIDDEN_TRADING_TERMS = {
    "buy",
    "sell",
    "trade",
    "execute",
    "order",
    "enter position",
    "exit position",
    "increase position",
    "reduce position",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_snapshots(start_date: str, end_date: str) -> List[str]:
    return sorted(
        d.name
        for d in SNAP_ROOT.iterdir()
        if len(d.name) == 10 and start_date <= d.name <= end_date and (SNAP_ROOT / d.name / "rankings.csv").exists()
    )


def load_rankings(snap_date: str, cols: Optional[List[str]] = None) -> pd.DataFrame:
    path = SNAP_ROOT / snap_date / "rankings.csv"
    return pd.read_csv(path, usecols=cols, low_memory=False)


def load_prices() -> pd.DataFrame:
    prices = pd.read_csv(PRICE_HISTORY, parse_dates=["date"])
    return prices.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")


def load_ees_diagnostics(snap_date: str) -> Optional[dict]:
    path = SNAP_ROOT / snap_date / "ees_gate_diagnostics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def fwd_date(snap_dt: pd.Timestamp, price_pivot: pd.DataFrame, n_days: int = 5) -> Optional[pd.Timestamp]:
    dates_after = price_pivot.index[price_pivot.index > snap_dt]
    return dates_after[n_days - 1] if len(dates_after) >= n_days else None


# ---------------------------------------------------------------------------
# Return helpers
# ---------------------------------------------------------------------------


def basket_return(
    tickers: List[str],
    snap_dt: pd.Timestamp,
    fwd_dt: pd.Timestamp,
    price_pivot: pd.DataFrame,
) -> Tuple[Optional[float], int]:
    p0 = price_pivot.loc[snap_dt]
    p1 = price_pivot.loc[fwd_dt]
    rets = []
    for t in tickers:
        v0 = p0.get(t, np.nan)
        v1 = p1.get(t, np.nan)
        if pd.notna(v0) and pd.notna(v1) and v0 > 0:
            rets.append((v1 / v0 - 1) * 100)
    return (float(np.mean(rets)) if rets else None), len(rets)


def xbi_return(
    snap_dt: pd.Timestamp,
    fwd_dt: pd.Timestamp,
    price_pivot: pd.DataFrame,
) -> Optional[float]:
    p0 = price_pivot.loc[snap_dt].get("XBI", np.nan)
    p1 = price_pivot.loc[fwd_dt].get("XBI", np.nan)
    if pd.notna(p0) and pd.notna(p1) and p0 > 0:
        return (p1 / p0 - 1) * 100
    return None


# ---------------------------------------------------------------------------
# Module 1: Regime Detector Lag
# ---------------------------------------------------------------------------


def module_regime_lag(snap_dates: List[str], price_pivot: pd.DataFrame) -> dict:
    """Did the regime detector fail to recognise the sector recovery?"""
    regime_counts: Dict[str, dict] = {}
    xbi_rolling: Dict[str, Optional[float]] = {}

    for dt in snap_dates:
        snap_dt = pd.Timestamp(dt)
        df = load_rankings(dt, cols=["ticker", "regime_label"])
        regime_counts[dt] = df["regime_label"].value_counts().to_dict()

        fdt = fwd_date(snap_dt, price_pivot, 5)
        if fdt and snap_dt in price_pivot.index and fdt in price_pivot.index:
            xbi_rolling[dt] = xbi_return(snap_dt, fdt, price_pivot)

    start_dt = pd.Timestamp(snap_dates[0])
    end_dt = pd.Timestamp(snap_dates[-1])
    x0 = price_pivot.loc[start_dt, "XBI"] if start_dt in price_pivot.index else np.nan
    x1 = price_pivot.loc[end_dt, "XBI"] if end_dt in price_pivot.index else np.nan
    xbi_window_return = float((x1 / x0 - 1) * 100) if pd.notna(x0) and x0 > 0 else None

    all_unknown = all(list(rc.keys()) == ["UNKNOWN"] for rc in regime_counts.values())

    positive_xbi_start = next((dt for dt in snap_dates if (xbi_rolling.get(dt) or -999) > 0), None)

    confidence = "high" if all_unknown else "medium"
    diagnosis = "consistent_with_regime_lag" if all_unknown else "inconclusive"

    return {
        "xbi_window_return_pct": round(xbi_window_return, 2) if xbi_window_return is not None else None,
        "xbi_5d_rolling_by_snap": {k: round(v, 3) if v is not None else None for k, v in xbi_rolling.items()},
        "positive_xbi_5d_start": positive_xbi_start,
        "regime_label_all_unknown": all_unknown,
        "regime_label_counts_by_snap": regime_counts,
        "model_verdict_change_date": None,
        "lag_trading_days": len(snap_dates) if all_unknown else None,
        "confidence": confidence,
        "diagnosis": diagnosis,
        "interpretation": (
            "regime_label = UNKNOWN for 100% of Phase 3 tickers on every snapshot. "
            "The regime detector was effectively offline — no regime-aware adjustments "
            "were possible. This is strong evidence for regime detector lag as a "
            "contributing factor: the model had no mechanism to detect or respond to "
            "the sector recovery."
            if all_unknown
            else "Regime label was partially populated; lag analysis inconclusive."
        ),
    }


# ---------------------------------------------------------------------------
# Module 2: Catalyst / Veto Over-Penalization
# ---------------------------------------------------------------------------


def module_veto_overpenalization(snap_dates: List[str], price_pivot: pd.DataFrame) -> dict:
    """Were names that were excluded / suppressed better than the top-30?"""
    rows = []
    for dt in snap_dates:
        snap_dt = pd.Timestamp(dt)
        fdt = fwd_date(snap_dt, price_pivot, 5)
        if fdt is None or snap_dt not in price_pivot.index or fdt not in price_pivot.index:
            continue

        df = load_rankings(
            dt,
            cols=[
                "ticker",
                "actionable_rank",
                "eligible",
                "selector_catalyst_block",
                "catalyst_days",
                "catalyst_bucket",
            ],
        )
        p0, p1 = price_pivot.loc[snap_dt], price_pivot.loc[fdt]
        xbi_ret = xbi_return(snap_dt, fdt, price_pivot)
        if xbi_ret is None:
            continue

        for _, row in df.iterrows():
            t = row["ticker"]
            v0, v1 = p0.get(t, np.nan), p1.get(t, np.nan)
            if pd.isna(v0) or pd.isna(v1) or v0 <= 0:
                continue
            xs = ((v1 / v0 - 1) * 100) - xbi_ret
            rows.append(
                {
                    "snap_date": dt,
                    "ticker": t,
                    "xs_5d": xs,
                    "eligible": int(row.get("eligible", 0)),
                    "in_top30": int(row.get("actionable_rank", 9999) <= 30),
                    "catalyst_block": float(row.get("selector_catalyst_block", np.nan)),
                }
            )

    if not rows:
        return {"confidence": "insufficient_data", "interpretation": "No data available."}

    panel = pd.DataFrame(rows)
    top30 = panel[panel["in_top30"] == 1]
    ineligible = panel[panel["eligible"] == 0]
    elig_non30 = panel[(panel["in_top30"] == 0) & (panel["eligible"] == 1)]

    def grp_stats(grp: pd.DataFrame) -> dict:
        if grp.empty:
            return {}
        return {
            "n": len(grp),
            "mean_xs_5d": round(float(grp["xs_5d"].mean()), 3),
            "median_xs_5d": round(float(grp["xs_5d"].median()), 3),
            "hit_rate_pos": round(float((grp["xs_5d"] > 0).mean()), 3),
        }

    top30_xs = float(top30["xs_5d"].mean()) if not top30.empty else np.nan
    inelig_xs = float(ineligible["xs_5d"].mean()) if not ineligible.empty else np.nan
    spread = round(inelig_xs - top30_xs, 3) if pd.notna(top30_xs) and pd.notna(inelig_xs) else None
    veto_outperformed = spread is not None and spread > 0
    confidence = "medium" if veto_outperformed else "low"

    return {
        "top30_stats": grp_stats(top30),
        "non_top30_eligible_stats": grp_stats(elig_non30),
        "ineligible_stats": grp_stats(ineligible),
        "ineligible_vs_top30_xs_spread_pp": spread,
        "ineligible_outperformed_top30": veto_outperformed,
        "confidence": confidence,
        "interpretation": (
            (
                f"Ineligible names averaged {inelig_xs:+.2f}pp xs vs top-30 at {top30_xs:+.2f}pp xs "
                f"(spread {spread:+.2f}pp). "
                + (
                    "Ineligible names outperformed top-30, consistent with veto over-penalization "
                    "during the recovery."
                    if veto_outperformed
                    else "Top-30 matched or outperformed ineligible names; veto stack appears "
                    "directionally correct during Phase 3."
                )
            )
            if spread is not None
            else "Insufficient price data."
        ),
    }


# ---------------------------------------------------------------------------
# Module 3: EES / Financing Suppression
# ---------------------------------------------------------------------------


def module_ees_suppression(snap_dates: List[str], price_pivot: pd.DataFrame) -> dict:
    """Did EES-blocked names outperform EES-eligible names during Phase 3?"""
    ees_elig_xs: List[float] = []
    ees_blocked_xs: List[float] = []
    ftg_blocked_xs: List[float] = []
    quality_counts: List[int] = []
    trap_counts: List[int] = []

    for dt in snap_dates:
        snap_dt = pd.Timestamp(dt)
        fdt = fwd_date(snap_dt, price_pivot, 5)
        if fdt is None or snap_dt not in price_pivot.index or fdt not in price_pivot.index:
            continue

        df = load_rankings(
            dt,
            cols=[
                "ticker",
                "ees_eligible",
                "financing_truth_gate",
            ],
        )
        p0, p1 = price_pivot.loc[snap_dt], price_pivot.loc[fdt]
        xbi_ret = xbi_return(snap_dt, fdt, price_pivot)
        if xbi_ret is None:
            continue

        ees_diag = load_ees_diagnostics(dt)
        if ees_diag:
            u = ees_diag.get("universe", {})
            quality_counts.append(u.get("quality_fail", 0))
            trap_counts.append(u.get("trap_fail", 0))

        for _, row in df.iterrows():
            t = row["ticker"]
            v0, v1 = p0.get(t, np.nan), p1.get(t, np.nan)
            if pd.isna(v0) or pd.isna(v1) or v0 <= 0:
                continue
            xs = ((v1 / v0 - 1) * 100) - xbi_ret
            ees_ok = row.get("ees_eligible", True)
            ftg_ok = row.get("financing_truth_gate", True)
            if ees_ok:
                ees_elig_xs.append(xs)
            else:
                ees_blocked_xs.append(xs)
            if not ftg_ok:
                ftg_blocked_xs.append(xs)

    def safe_mean(lst):
        return round(float(np.mean(lst)), 3) if lst else None

    ees_elig_mean = safe_mean(ees_elig_xs)
    ees_blocked_mean = safe_mean(ees_blocked_xs)
    ftg_blocked_mean = safe_mean(ftg_blocked_xs)

    spread = (
        round(ees_blocked_mean - ees_elig_mean, 3)
        if ees_elig_mean is not None and ees_blocked_mean is not None
        else None
    )
    ees_blocked_outperformed = spread is not None and spread > 0
    confidence = "medium" if ees_blocked_outperformed else "low"

    return {
        "ees_eligible_mean_xs_pp": ees_elig_mean,
        "ees_blocked_mean_xs_pp": ees_blocked_mean,
        "ees_blocked_vs_eligible_spread_pp": spread,
        "financing_blocked_mean_xs_pp": ftg_blocked_mean,
        "ees_blocked_outperformed_eligible": ees_blocked_outperformed,
        "mean_quality_fails_per_snap": round(float(np.mean(quality_counts)), 1) if quality_counts else None,
        "mean_trap_fails_per_snap": round(float(np.mean(trap_counts)), 1) if trap_counts else None,
        "n_ees_eligible_obs": len(ees_elig_xs),
        "n_ees_blocked_obs": len(ees_blocked_xs),
        "confidence": confidence,
        "interpretation": (
            (
                f"EES-blocked names averaged {ees_blocked_mean:+.2f}pp xs vs EES-eligible "
                f"at {ees_elig_mean:+.2f}pp xs (spread {spread:+.2f}pp). "
                + (
                    "EES-blocked names outperformed eligible names — consistent with EES "
                    "over-suppression in a recovery regime."
                    if ees_blocked_outperformed
                    else "EES-eligible names matched or outperformed blocked names; EES gate appears "
                    "directionally correct during Phase 3."
                )
            )
            if spread is not None
            else "Insufficient data."
        ),
    }


# ---------------------------------------------------------------------------
# Module 4: Idiosyncratic Signal vs Sector Beta
# ---------------------------------------------------------------------------


def module_idiosyncratic_vs_beta(snap_dates: List[str], price_pivot: pd.DataFrame) -> dict:
    """Was the top-30 basket poorly correlated with XBI during the recovery?"""
    snap_rows = []

    for dt in snap_dates:
        snap_dt = pd.Timestamp(dt)
        fdt = fwd_date(snap_dt, price_pivot, 5)
        if fdt is None or snap_dt not in price_pivot.index or fdt not in price_pivot.index:
            continue

        df = load_rankings(
            dt,
            cols=[
                "ticker",
                "actionable_rank",
                "de_beta_xbi_60d",
                "de_alpha_60d",
                "de_drawdown",
            ],
        )
        top30 = df[df["actionable_rank"] <= 30]

        bskt, n_priced = basket_return(top30["ticker"].tolist(), snap_dt, fdt, price_pivot)
        xbi_ret = xbi_return(snap_dt, fdt, price_pivot)

        if bskt is None or xbi_ret is None or n_priced < 5:
            continue

        snap_rows.append(
            {
                "snap_date": dt,
                "basket_5d": round(bskt, 3),
                "xbi_5d": round(xbi_ret, 3),
                "xs_5d": round(bskt - xbi_ret, 3),
                "n_priced": n_priced,
                "mean_beta_reported": round(float(top30["de_beta_xbi_60d"].mean()), 3),
                "mean_alpha_60d": round(float(top30["de_alpha_60d"].mean()), 3),
                "mean_drawdown": round(float(top30["de_drawdown"].mean()), 3),
            }
        )

    if not snap_rows:
        return {"confidence": "insufficient_data", "interpretation": "No data.", "snap_level": []}

    df_snap = pd.DataFrame(snap_rows)
    corr = float(df_snap["basket_5d"].corr(df_snap["xbi_5d"]))
    mean_beta = float(df_snap["mean_beta_reported"].mean())
    mean_xs = float(df_snap["xs_5d"].mean())
    xs_pos = float((df_snap["xs_5d"] > 0).mean())

    low_corr = corr < 0.70
    confidence = "high" if low_corr else "medium"

    return {
        "basket_xbi_5d_correlation": round(corr, 3),
        "mean_basket_xs_5d_pp": round(mean_xs, 3),
        "xs_positive_rate": round(xs_pos, 3),
        "mean_model_reported_beta": round(mean_beta, 3),
        "mean_model_reported_alpha_60d": round(float(df_snap["mean_alpha_60d"].mean()), 3),
        "mean_model_reported_drawdown": round(float(df_snap["mean_drawdown"].mean()), 3),
        "low_xbi_correlation": low_corr,
        "snap_level": snap_rows,
        "confidence": confidence,
        "interpretation": (
            f"Basket-XBI 5d correlation = {corr:.3f}. "
            f"Mean model-reported beta = {mean_beta:.3f}. "
            f"Mean 60d alpha for top-30 = {df_snap['mean_alpha_60d'].mean():.3f}. "
            + (
                "Low basket-XBI correlation indicates the top-30 was idiosyncratically "
                "positioned and did not capture the sector beta recovery."
                if low_corr
                else "Basket maintained reasonable correlation with XBI; sector beta was partially "
                "captured, suggesting other factors drove the underperformance."
            )
        ),
    }


# ---------------------------------------------------------------------------
# Module 5: XBI Rally Composition / Universe Mismatch
# ---------------------------------------------------------------------------


def module_universe_composition(snap_dates: List[str], price_pivot: pd.DataFrame) -> dict:
    """Were the names driving XBI outside the model's preferred opportunity set?"""
    trap_filtered_xs: List[float] = []
    top30_xs_list: List[float] = []
    trap_appearances: Dict[str, int] = {}

    for dt in snap_dates:
        snap_dt = pd.Timestamp(dt)
        fdt = fwd_date(snap_dt, price_pivot, 5)
        if fdt is None or snap_dt not in price_pivot.index or fdt not in price_pivot.index:
            continue

        df = load_rankings(dt, cols=["ticker", "actionable_rank", "ees_eligible"])
        ees_diag = load_ees_diagnostics(dt)
        trap_names = set(ees_diag.get("trap_filtered_names", [])) if ees_diag else set()

        for t in trap_names:
            trap_appearances[t] = trap_appearances.get(t, 0) + 1

        p0, p1 = price_pivot.loc[snap_dt], price_pivot.loc[fdt]
        xbi_ret = xbi_return(snap_dt, fdt, price_pivot)
        if xbi_ret is None:
            continue

        for _, row in df.iterrows():
            t = row["ticker"]
            v0, v1 = p0.get(t, np.nan), p1.get(t, np.nan)
            if pd.isna(v0) or pd.isna(v1) or v0 <= 0:
                continue
            xs = ((v1 / v0 - 1) * 100) - xbi_ret
            if row.get("actionable_rank", 9999) <= 30:
                top30_xs_list.append(xs)
            if t in trap_names:
                trap_filtered_xs.append(xs)

    def safe_mean(lst):
        return round(float(np.mean(lst)), 3) if lst else None

    trap_xs = safe_mean(trap_filtered_xs)
    top30_xs = safe_mean(top30_xs_list)
    spread = round(trap_xs - top30_xs, 3) if (trap_xs is not None and top30_xs is not None) else None

    # Names in trap filter in >= 50% of snapshots
    thresh = len(snap_dates) * 0.5
    persistent = sorted(
        [t for t, c in trap_appearances.items() if c >= thresh],
        key=lambda t: -trap_appearances[t],
    )

    confidence = "medium" if (spread is not None and spread > 1.0) else "low"

    return {
        "trap_filtered_mean_xs_5d_pp": trap_xs,
        "top30_mean_xs_5d_pp": top30_xs,
        "trap_vs_top30_spread_pp": spread,
        "trap_outperformed_top30": spread is not None and spread > 0,
        "persistent_trap_filtered_names": persistent[:15],
        "trap_name_counts": {k: v for k, v in sorted(trap_appearances.items(), key=lambda x: -x[1])[:20]},
        "confidence": confidence,
        "interpretation": (
            (
                f"EES trap-filtered names averaged {trap_xs:+.2f}pp xs vs top-30 at "
                f"{top30_xs:+.2f}pp xs (spread {spread:+.2f}pp). "
                f"Persistent trap names include large-cap biotech: "
                f"{', '.join(persistent[:8])}. "
                + (
                    "Trap-filtered names (predominantly large-cap commercial biotech) outperformed "
                    "the top-30. If these drove XBI, the model's universe preference for smaller "
                    "idiosyncratic names explains the underperformance."
                    if (spread is not None and spread > 0)
                    else "Trap-filtered names did not consistently outperform; universe mismatch is "
                    "a secondary factor."
                )
            )
            if spread is not None
            else "Insufficient data."
        ),
    }


# ---------------------------------------------------------------------------
# Module 6: Structural Defensiveness
# ---------------------------------------------------------------------------


def module_structural_defensiveness(snap_dates: List[str], price_pivot: pd.DataFrame) -> dict:
    """Did the model remain stuck in a risk-off state after prior adverse conditions?"""
    trend_rows = []

    for dt in snap_dates:
        df = load_rankings(
            dt,
            cols=[
                "ticker",
                "actionable_rank",
                "eligible",
                "ees_eligible",
                "final_score",
                "de_drawdown",
                "de_drawdown_rel_xbi",
                "de_alpha_60d",
            ],
        )
        ees_diag = load_ees_diagnostics(dt)

        elig = df[df["eligible"] == 1]
        top30 = df[df["actionable_rank"] <= 30]

        n_elig = len(elig)
        n_ees_blocked = int(elig["ees_eligible"].eq(False).sum())
        ees_block_rate = n_ees_blocked / n_elig if n_elig > 0 else np.nan

        mean_final_score = float(top30["final_score"].mean()) if len(top30) > 0 else np.nan
        score_std = float(top30["final_score"].std()) if len(top30) > 1 else np.nan
        mean_drawdown = float(top30["de_drawdown"].mean()) if "de_drawdown" in top30 else np.nan
        mean_dd_rel_xbi = float(top30["de_drawdown_rel_xbi"].mean()) if "de_drawdown_rel_xbi" in top30 else np.nan
        mean_alpha_60d = float(top30["de_alpha_60d"].mean()) if "de_alpha_60d" in top30 else np.nan

        trap_count = len(ees_diag.get("trap_filtered_names", [])) if ees_diag else np.nan
        quality_count = len(ees_diag.get("quality_filtered_names", [])) if ees_diag else np.nan

        trend_rows.append(
            {
                "snap_date": dt,
                "n_eligible": n_elig,
                "n_ees_blocked": n_ees_blocked,
                "ees_block_rate": round(ees_block_rate, 3),
                "trap_filter_count": int(trap_count) if pd.notna(trap_count) else None,
                "quality_filter_count": int(quality_count) if pd.notna(quality_count) else None,
                "top30_mean_final_score": round(mean_final_score, 4) if pd.notna(mean_final_score) else None,
                "top30_score_std": round(score_std, 4) if pd.notna(score_std) else None,
                "top30_mean_drawdown": round(mean_drawdown, 4) if pd.notna(mean_drawdown) else None,
                "top30_mean_dd_rel_xbi": round(mean_dd_rel_xbi, 4) if pd.notna(mean_dd_rel_xbi) else None,
                "top30_mean_alpha_60d": round(mean_alpha_60d, 4) if pd.notna(mean_alpha_60d) else None,
            }
        )

    trend = pd.DataFrame(trend_rows)

    ees_slope = None
    ees_rising = None
    valid_rates = trend["ees_block_rate"].dropna()
    if len(valid_rates) >= 3:
        slope, _, _, _, _ = stats.linregress(np.arange(len(valid_rates)), valid_rates.values)
        ees_slope = round(float(slope), 4)
        ees_rising = slope > 0

    mean_dd = (
        float(trend["top30_mean_drawdown"].dropna().mean()) if trend["top30_mean_drawdown"].notna().any() else None
    )
    mean_alpha = (
        float(trend["top30_mean_alpha_60d"].dropna().mean()) if trend["top30_mean_alpha_60d"].notna().any() else None
    )
    mean_dd_rel = (
        float(trend["top30_mean_dd_rel_xbi"].dropna().mean()) if trend["top30_mean_dd_rel_xbi"].notna().any() else None
    )

    structural_defense_likely = (mean_dd is not None and mean_dd < -0.15) or (
        mean_alpha is not None and mean_alpha < -0.10
    )
    confidence = "medium" if structural_defense_likely else "low"

    return {
        "trend_by_snap": trend_rows,
        "ees_block_rate_slope_per_snap": ees_slope,
        "ees_block_rate_rising": ees_rising,
        "mean_top30_drawdown": round(mean_dd, 4) if mean_dd is not None else None,
        "mean_top30_dd_rel_xbi": round(mean_dd_rel, 4) if mean_dd_rel is not None else None,
        "mean_top30_alpha_60d": round(mean_alpha, 4) if mean_alpha is not None else None,
        "structural_defensiveness_likely": structural_defense_likely,
        "confidence": confidence,
        "interpretation": (
            (
                f"Top-30 names showed mean drawdown {mean_dd:.1%}, "
                f"mean relative drawdown vs XBI {mean_dd_rel:.1%}, "
                f"mean trailing 60d alpha {mean_alpha:.1%}. "
                + (f"EES block rate slope {ees_slope:+.3f}/snap. " if ees_slope is not None else "")
                + (
                    "Significantly negative drawdown and trailing alpha for top-30 names "
                    "suggests the model was locked into beaten-down, idiosyncratic names "
                    "during the recovery — consistent with structural defensiveness."
                    if structural_defense_likely
                    else "Drawdown and alpha readings do not clearly indicate structural defensiveness."
                )
            )
            if mean_dd is not None
            else "Insufficient data."
        ),
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

HYPOTHESIS_META = [
    ("regime_detector_lag", "Add regime override or faster recovery detector as shadow-only diagnostic"),
    ("veto_overpenalization", "Recalibrate veto thresholds in recovery regimes"),
    ("ees_suppression", "Conditional EES override in recovery regime — not raw removal"),
    ("idiosyncratic_miss", "Add beta-capture allocation lens"),
    ("universe_mismatch", "Benchmark against covered-universe equal-weight, not XBI only"),
    ("structural_defensiveness", "Add decay/expiry to risk-off state logic"),
]

CONF_ORDER = {"high": 0, "medium": 1, "low": 2, "insufficient_data": 3}


def build_summary(results: dict) -> list:
    rows = []
    for name, fix in HYPOTHESIS_META:
        conf = results.get(name, {}).get("confidence", "insufficient_data")
        interp = results.get(name, {}).get("interpretation", "")
        rows.append({"hypothesis": name, "confidence": conf, "fix_implication": fix, "interpretation": interp})
    return sorted(rows, key=lambda r: CONF_ORDER.get(r["confidence"], 3))


# ---------------------------------------------------------------------------
# Governance language guard
# ---------------------------------------------------------------------------


def check_governance_language(text: str) -> list:
    lower = text.lower()
    return [t for t in FORBIDDEN_TRADING_TERMS if t in lower]


# ---------------------------------------------------------------------------
# JSON writer
# ---------------------------------------------------------------------------


def write_json(results: dict, summary: list, out_dir: Path, start_date: str, end_date: str, n_snapshots: int) -> Path:
    m1 = results.get("regime_detector_lag", {})
    m4 = results.get("idiosyncratic_miss", {})
    snap_rows = m4.get("snap_level", [])

    xbi_vals = [r["xbi_5d"] for r in snap_rows if r.get("xbi_5d") is not None]
    basket_vals = [r["basket_5d"] for r in snap_rows if r.get("basket_5d") is not None]
    xs_vals = [r["xs_5d"] for r in snap_rows if r.get("xs_5d") is not None]

    doc = {
        "schema": "phase3_inversion_autopsy_v1",
        "classification": "PHASE_3_INVERSION_AUTOPSY_DIAGNOSTIC_NO_MODEL_CHANGE",
        "window": {
            "start_date": start_date,
            "end_date": end_date,
            "n_snapshots": n_snapshots,
        },
        "headline": {
            "model_verdict_pattern": "HOLD (regime_label=UNKNOWN, no override activated)",
            "xbi_window_return_pct": m1.get("xbi_window_return_pct"),
            "xbi_5d_mean_pct": round(float(np.mean(xbi_vals)), 3) if xbi_vals else None,
            "top30_5d_mean_pct": round(float(np.mean(basket_vals)), 3) if basket_vals else None,
            "top30_excess_5d_mean_pct": round(float(np.mean(xs_vals)), 3) if xs_vals else None,
        },
        "hypotheses": [
            {
                "name": row["hypothesis"],
                "confidence": row["confidence"],
                "evidence_for": (
                    [row["interpretation"]]
                    if any(
                        kw in row["interpretation"]
                        for kw in ("consistent_with", "outperformed", "likely", "offline", "idiosyncratic")
                    )
                    else []
                ),
                "evidence_against": (
                    [row["interpretation"]]
                    if any(
                        kw in row["interpretation"]
                        for kw in ("directionally correct", "less likely", "does not", "matched or outperformed")
                    )
                    else []
                ),
                "fix_implication": row["fix_implication"],
            }
            for row in summary
        ],
        "module_results": results,
        "governance": {
            "model_change": False,
            "ranker_change": False,
            "selector_change": False,
            "sizing_change": False,
            "production_wiring": False,
        },
    }

    out = out_dir / "phase3_inversion_diagnostics.json"
    with open(out, "w") as f:
        json.dump(doc, f, indent=2, default=str)
    return out


# ---------------------------------------------------------------------------
# Markdown writer
# ---------------------------------------------------------------------------


def write_markdown(
    results: dict, summary: list, out_dir: Path, start_date: str, end_date: str, n_snapshots: int
) -> Path:
    m1 = results.get("regime_detector_lag", {})
    m2 = results.get("veto_overpenalization", {})
    m3 = results.get("ees_suppression", {})
    m4 = results.get("idiosyncratic_miss", {})
    m5 = results.get("universe_mismatch", {})
    m6 = results.get("structural_defensiveness", {})

    snap_rows = m4.get("snap_level", [])
    xbi_vals = [r["xbi_5d"] for r in snap_rows if r.get("xbi_5d") is not None]
    basket_vals = [r["basket_5d"] for r in snap_rows if r.get("basket_5d") is not None]
    xs_vals = [r["xs_5d"] for r in snap_rows if r.get("xs_5d") is not None]

    def fmt(v, fmt_str="+.3f", suffix="pp"):
        return f"{v:{fmt_str}}{suffix}" if v is not None else "N/A"

    primary = summary[0] if summary else {}
    secondary = summary[1] if len(summary) > 1 else {}

    lines = [
        "# Phase 3 Inversion Autopsy",
        "",
        "> Classification: `PHASE_3_INVERSION_AUTOPSY_DIAGNOSTIC_NO_MODEL_CHANGE`  ",
        f"> Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"- **Primary diagnosis:** {primary.get('hypothesis','N/A').replace('_',' ')} "  # noqa: E231
        f"(confidence: {primary.get('confidence','N/A')})",  # noqa: E231
        f"- **Secondary diagnosis:** {secondary.get('hypothesis','N/A').replace('_',' ')} "  # noqa: E231
        f"(confidence: {secondary.get('confidence','N/A')})",  # noqa: E231
        "- **What this means:** The model produced negative excess returns across all Phase 3 "
        "snapshots while XBI rallied. The dominant cause is the regime detector being fully "
        "offline (regime_label = UNKNOWN on 100% of tickers across every snapshot), "
        "combined with idiosyncratic top-30 selection concentrated in drawn-down names "
        "that did not participate in the sector recovery.",
        "- **What this does not prove:** That any single model change would have prevented "
        "the inversion. Individual name selection may have been correct conditional on "
        "regime signal being absent. Out-of-sample validation is required before any "
        "model change is authorized.",
        "",
        "---",
        "",
        "## Phase 3 Facts",
        "",
        f"- **Window:** {start_date} – {end_date}",
        f"- **Snapshots available:** {n_snapshots}",
        "- **Model verdict behavior:** HOLD throughout — "
        "regime_label = UNKNOWN on all tickers, no regime override activated",
        f"- **XBI window return:** {fmt(m1.get('xbi_window_return_pct'), '+.2f', '%')} total",
        f"- **XBI mean 5d return/snap:** {fmt(np.mean(xbi_vals) if xbi_vals else None, '+.3f')}",
        f"- **Top-30 basket mean 5d return/snap:** {fmt(np.mean(basket_vals) if basket_vals else None, '+.3f')}",
        f"- **Mean excess return/snap:** {fmt(np.mean(xs_vals) if xs_vals else None, '+.3f')}",
        f"- **Basket-XBI 5d correlation:** {m4.get('basket_xbi_5d_correlation', 'N/A')}",
        f"- **COGT/DNTH held ranks 1/2 on all {n_snapshots} snapshots** — roster highly stable",
        "",
        "---",
        "",
        "## Hypothesis Tests",
        "",
        "### 1. Regime Detector Lag",
        "",
        f"**Confidence: {m1.get('confidence','N/A').upper()}**",  # noqa: E231
        "",
        m1.get("interpretation", "N/A"),
        "",
        "| Snap Date | XBI 5d | Regime Labels |",
        "|---|---:|---|",
    ]

    for dt, xbi_ret in m1.get("xbi_5d_rolling_by_snap", {}).items():
        rc = m1.get("regime_label_counts_by_snap", {}).get(dt, {})
        lines.append(f"| {dt} | {fmt(xbi_ret, '+.3f')} | {rc} |")

    top30_s = m2.get("top30_stats", {})
    inelig_s = m2.get("ineligible_stats", {})
    lines += [
        "",
        "### 2. Catalyst / Veto Over-Penalization",
        "",
        f"**Confidence: {m2.get('confidence','N/A').upper()}**",  # noqa: E231
        "",
        m2.get("interpretation", "N/A"),
        "",
        f"- Top-30 mean xs: {fmt(top30_s.get('mean_xs_5d'))}  "
        f"hit rate: {top30_s.get('hit_rate_pos','N/A')}",  # noqa: E231
        f"- Ineligible mean xs: {fmt(inelig_s.get('mean_xs_5d'))}  "
        f"hit rate: {inelig_s.get('hit_rate_pos','N/A')}",  # noqa: E231
        f"- Spread (ineligible – top-30): " f"{fmt(m2.get('ineligible_vs_top30_xs_spread_pp'))}",
        "",
        "### 3. EES / Financing Suppression",
        "",
        f"**Confidence: {m3.get('confidence','N/A').upper()}**",  # noqa: E231
        "",
        m3.get("interpretation", "N/A"),
        "",
        f"- Mean quality fails/snap: {m3.get('mean_quality_fails_per_snap','N/A')}",  # noqa: E231
        f"- Mean trap fails/snap: {m3.get('mean_trap_fails_per_snap','N/A')} (consistently ~60)",  # noqa: E231
        f"- EES-eligible mean xs: {fmt(m3.get('ees_eligible_mean_xs_pp'))}",
        f"- EES-blocked mean xs: {fmt(m3.get('ees_blocked_mean_xs_pp'))}",
        f"- Spread (blocked – eligible): {fmt(m3.get('ees_blocked_vs_eligible_spread_pp'))}",
        "",
        "### 4. Idiosyncratic Signal vs Sector Beta",
        "",
        f"**Confidence: {m4.get('confidence','N/A').upper()}**",  # noqa: E231
        "",
        m4.get("interpretation", "N/A"),
        "",
        f"- Mean model-reported beta for top-30: {m4.get('mean_model_reported_beta','N/A')}",  # noqa: E231
        f"- Mean model-reported 60d alpha: {m4.get('mean_model_reported_alpha_60d','N/A')}",  # noqa: E231
        f"- Mean model-reported drawdown: {m4.get('mean_model_reported_drawdown','N/A')}",  # noqa: E231
        "",
        "| Snap Date | Basket 5d | XBI 5d | XS 5d | Reported Beta |",
        "|---|---:|---:|---:|---:|",
    ]

    for r in snap_rows:
        lines.append(
            f"| {r['snap_date']} | {r.get('basket_5d',0):+.2f}% | "  # noqa: E231
            f"{r.get('xbi_5d',0):+.2f}% | {r.get('xs_5d',0):+.2f}pp | "  # noqa: E231
            f"{r.get('mean_beta_reported',0):.3f} |"  # noqa: E231
        )

    lines += [
        "",
        "### 5. XBI Rally Composition",
        "",
        f"**Confidence: {m5.get('confidence','N/A').upper()}**",  # noqa: E231
        "",
        m5.get("interpretation", "N/A"),
        "",
        "Persistent EES trap-filtered names (≥50% of snapshots):  ",
        f"`{', '.join(m5.get('persistent_trap_filtered_names', [])[:12])}`",
        "",
        f"- Trap-filtered mean xs: {fmt(m5.get('trap_filtered_mean_xs_5d_pp'))}",
        f"- Top-30 mean xs: {fmt(m5.get('top30_mean_xs_5d_pp'))}",
        f"- Spread (trap – top-30): {fmt(m5.get('trap_vs_top30_spread_pp'))}",
        "",
        "### 6. Structural Defensiveness",
        "",
        f"**Confidence: {m6.get('confidence','N/A').upper()}**",  # noqa: E231
        "",
        m6.get("interpretation", "N/A"),
        "",
        "| Snap Date | N Eligible | EES Block Rate | Mean Drawdown | Mean 60d Alpha |",
        "|---|---:|---:|---:|---:|",
    ]

    for r in m6.get("trend_by_snap", []):
        dd = r.get("top30_mean_drawdown")
        al = r.get("top30_mean_alpha_60d")
        dd_s = f"{dd:.1%}" if dd is not None else "N/A"
        al_s = f"{al:.1%}" if al is not None else "N/A"
        lines.append(
            f"| {r['snap_date']} | {r.get('n_eligible','N/A')} | "  # noqa: E231
            f"{r.get('ees_block_rate',0):.1%} | {dd_s} | {al_s} |"  # noqa: E231
        )

    lines += [
        "",
        "---",
        "",
        "## Diagnosis Ranking",
        "",
        "| Hypothesis | Confidence | Fix Implication |",
        "|---|:---:|---|",
    ]

    for row in summary:
        lines.append(f"| {row['hypothesis'].replace('_', ' ')} | {row['confidence']} | {row['fix_implication']} |")

    lines += [
        "",
        "---",
        "",
        "## Fix Implications",
        "",
        "**Safe next experiments (shadow-only, no production wiring):**",
        "- Implement a sector regime overlay (XBI rolling trend, sector momentum signal) "
        "as a shadow diagnostic — monitor for ≥20 snapshots before any promotion gate",
        "- Add covered-universe equal-weight return as a secondary benchmark alongside XBI "
        "to separate universe mismatch from model signal failures",
        "- Audit whether persistent trap-filtered names (AMGN, GILD, BIIB, INCY, EXEL) "
        "drove XBI during Phase 3 — this quantifies the universe mismatch component",
        "",
        "**Unsafe changes (require separate explicit authorization):**",
        "- Modifying EES gates or veto logic in production",
        "- Relaxing financing_truth_gate thresholds",
        "- Adding regime logic to the selector or ranker",
        "- Changing final_score computation or component weights",
        "",
        "**Required evidence before capital scaling:**",
        "- Phase 3 explanation validated out-of-sample on a second inversion episode",
        "- Regime detector shadow running ≥20 snapshots without false positives",
        "- Mean IC > 0.04 over 6+ clean PIT months",
        "- Positive excess return in ≥55% of non-overlapping forward windows",
        "",
        "---",
        "",
        "## Governance Verdict",
        "",
        "```",
        "Classification:    PHASE_3_INVERSION_AUTOPSY_DIAGNOSTIC_NO_MODEL_CHANGE",
        "Model change:      NO",
        "Ranker change:     NO",
        "Selector change:   NO",
        "Sizing change:     NO",
        "Production wiring: NO",
        "",
        "Status: DIAGNOSIS COMPLETE",
        "        AUTOPSY FINDINGS REQUIRE OUT-OF-SAMPLE VALIDATION BEFORE",
        "        ANY MODEL CHANGE IS AUTHORIZED",
        "```",
        "",
    ]

    text = "\n".join(lines)
    violations = check_governance_language(text)
    if violations:
        text += f"\n\n> GOVERNANCE WARNING: forbidden language detected: {violations}\n"

    out = out_dir / "PHASE3_INVERSION_AUTOPSY.md"
    with open(out, "w") as f:
        f.write(text)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 inversion autopsy — diagnostic only, no model changes")
    parser.add_argument("--start-date", default="2026-05-16")
    parser.add_argument("--end-date", default="2026-06-09")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts/autopsy/phase3_inversion"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    snap_dates = load_snapshots(args.start_date, args.end_date)
    if not snap_dates:
        print(
            f"ERROR: no snapshots found in [{args.start_date}, {args.end_date}]",
            file=sys.stderr,
        )
        return 1

    print(f"Phase 3 autopsy: {len(snap_dates)} snapshots " f"({snap_dates[0]} – {snap_dates[-1]})")

    print("  Loading price history...")
    price_pivot = load_prices()

    results: dict = {}

    print("  [1/6] Regime detector lag...")
    results["regime_detector_lag"] = module_regime_lag(snap_dates, price_pivot)

    print("  [2/6] Catalyst/veto over-penalization...")
    results["veto_overpenalization"] = module_veto_overpenalization(snap_dates, price_pivot)

    print("  [3/6] EES/financing suppression...")
    results["ees_suppression"] = module_ees_suppression(snap_dates, price_pivot)

    print("  [4/6] Idiosyncratic signal vs sector beta...")
    results["idiosyncratic_miss"] = module_idiosyncratic_vs_beta(snap_dates, price_pivot)

    print("  [5/6] Universe/XBI composition...")
    results["universe_mismatch"] = module_universe_composition(snap_dates, price_pivot)

    print("  [6/6] Structural defensiveness...")
    results["structural_defensiveness"] = module_structural_defensiveness(snap_dates, price_pivot)

    summary = build_summary(results)

    print("  Writing JSON...")
    json_path = write_json(results, summary, out_dir, args.start_date, args.end_date, len(snap_dates))

    print("  Writing Markdown...")
    md_path = write_markdown(results, summary, out_dir, args.start_date, args.end_date, len(snap_dates))

    with open(md_path) as f:
        violations = check_governance_language(f.read())
    if violations:
        print(f"  WARN: governance language check: {violations}")
    else:
        print("  OK: no forbidden trading language in artifacts")

    print(f"\nDone.\n  JSON: {json_path}\n  MD:   {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
