#!/usr/bin/env python3
"""Options feature enrichment — shadow diagnostics layer.

Builds a richer options-derived feature set for the biotech model using:
  - Universe options snapshot (TT market metrics: ATM IV, IV rank, term structure)
  - Split-adjusted price history (realized volatility, beta, idiosyncratic vol)
  - Latest rankings (catalyst dates, spot price, market cap, existing straddle data)
  - Options diagnostics sidecar (DTE, nearest expiry)

Implements feasible layers given available data:
  Layer 1  — Coverage normalization (VALID / LOW_LIQ / NO_CHAIN / FETCH_FAILED)
  Layer 2  — Realized volatility (10d/20d/60d/120d, XBI beta, idiosyncratic vol)
  Layer 3  — IV vs realized volatility comparison (VRP, IV/RV ratio, flags)
  Layer 4  — Event pricing (straddle move, event premium IV, jump proxy)
  Layer 5  — Term structure diagnostics (front/back, slope, backwardation/contango)
  Layer 6  — Options liquidity quality score (0-5)
  Layer 7  — Synthetic proxies for NO_LISTED_OPTIONS tickers (clearly labeled)
  Layer 8  — Cross-sectional features (zscore, percentile within universe)
  Layer 9  — Enriched flags (human-readable)

Hard constraints (governance):
  OPTIONS_ENRICHMENT / EXPECTATION_LAYER_SHADOW / NO_MODEL_CHANGE /
  NO_RANKER_CHANGE / NO_SELECTOR_CHANGE / NO_SIZING_CHANGE / NO_TRADING_CHANGE

  - No observed option field populated for NO_LISTED_OPTIONS tickers.
  - All synthetic fields marked NOT_MARKET_IMPLIED.
  - No negative IVs used.
  - No model/ranker/selector/sizing/trading effect.

Usage:
  python3 tools/enrich_options_features.py
  python3 tools/enrich_options_features.py --date 2026-06-28 --write-shadow-only
  python3 tools/enrich_options_features.py --no-synthetic
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

ENRICHMENT_VERSION = "1.1.0"
# ATM straddle approximation constant: straddle_pct ≈ STRADDLE_K * iv * sqrt(T)
STRADDLE_K = math.sqrt(2 / math.pi)  # ≈ 0.7979
# Tenor assumed for TT implied_volatility_index (approx 30-day constant maturity)
TT_IV_TENOR_DAYS = 30
SELL_ONLY_TICKERS = {"ABVX"}

# Coverage status constants
VALID_OPTIONS = "VALID_OPTIONS"
LOW_LIQUIDITY_CHAIN = "LOW_LIQUIDITY_CHAIN"
NO_LISTED_OPTIONS = "NO_LISTED_OPTIONS"
FETCH_FAILED = "FETCH_FAILED"

# Options data quality scores (0-5)
_BASE_QUALITY = {
    NO_LISTED_OPTIONS: 0,
    FETCH_FAILED: 0,
    LOW_LIQUIDITY_CHAIN: 1,
    VALID_OPTIONS: 3,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sf(v: Any) -> float | None:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _derive_coverage(o: dict) -> str:
    """Derive coverage status from legacy snapshot (no opt_coverage_status field)."""
    cov = o.get("opt_coverage_status")
    if cov:
        return cov
    atm = _sf(o.get("opt_atm_iv"))
    liq = o.get("opt_liquidity_state", "absent")
    if atm is None:
        return NO_LISTED_OPTIONS
    return LOW_LIQUIDITY_CHAIN if liq == "absent" else VALID_OPTIONS


def _quality_score(cov: str, liq: str, has_event_expiry: bool) -> int:
    """Options data quality 0-5."""
    base = _BASE_QUALITY.get(cov, 0)
    if base == 0:
        return 0
    if liq == "liquid":
        base = min(base + 1, 4)
    if has_event_expiry:
        base = min(base + 1, 5)
    return base


# ---------------------------------------------------------------------------
# Layer 2 — Realized volatility
# ---------------------------------------------------------------------------


def compute_realized_vols(price_df: pd.DataFrame, as_of: str, windows: list[int]) -> pd.DataFrame:
    """Compute annualized realized vol for multiple windows from daily close prices."""
    price_df = price_df[price_df["date"] <= as_of].copy()
    price_df["date"] = pd.to_datetime(price_df["date"])
    price_df = price_df.sort_values(["ticker", "date"])
    price_df["log_ret"] = price_df.groupby("ticker")["close"].transform(lambda x: np.log(x / x.shift(1)))
    max_window = max(windows)
    # Keep only the last max_window+1 rows per ticker
    price_df = price_df.groupby("ticker").tail(max_window + 1)

    results = {}
    for ticker, grp in price_df.groupby("ticker"):
        rets = grp["log_ret"].dropna()
        row: dict = {"ticker": ticker}
        for w in windows:
            tail = rets.tail(w)
            rv = float(tail.std() * math.sqrt(252)) if len(tail) >= max(5, w // 2) else None
            row[f"rv_{w}d"] = rv
        results[ticker] = row

    return pd.DataFrame(list(results.values()))


def compute_xbi_features(price_df: pd.DataFrame, as_of: str, window: int = 60) -> pd.DataFrame:
    """Compute realized beta to XBI and idiosyncratic vol."""
    price_df = price_df[price_df["date"] <= as_of].copy()
    price_df["date"] = pd.to_datetime(price_df["date"])
    price_df = price_df.sort_values(["ticker", "date"])
    price_df["log_ret"] = price_df.groupby("ticker")["close"].transform(lambda x: np.log(x / x.shift(1)))

    xbi = price_df[price_df["ticker"] == "XBI"][["date", "log_ret"]].tail(window + 1)
    xbi = xbi.dropna().tail(window).set_index("date")["log_ret"]

    rows = []
    for ticker, grp in price_df.groupby("ticker"):
        if ticker == "XBI":
            continue
        rets = grp.dropna(subset=["log_ret"]).tail(window).set_index("date")["log_ret"]
        aligned = rets.align(xbi, join="inner")
        t_rets, x_rets = aligned[0], aligned[1]
        if len(t_rets) < 20:
            rows.append({"ticker": ticker, "realized_beta_xbi_60d": None, "idiosyncratic_vol_60d": None})
            continue
        slope, _, _, _, _ = stats.linregress(x_rets.values, t_rets.values)
        resid = t_rets.values - slope * x_rets.values
        idio = float(resid.std() * math.sqrt(252))
        rows.append(
            {
                "ticker": ticker,
                "realized_beta_xbi_60d": round(float(slope), 4),
                "idiosyncratic_vol_60d": round(idio, 4),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Layer 4/5 — Event pricing and term structure
# ---------------------------------------------------------------------------


def event_pricing_features(
    atm_iv: float | None,
    front_iv: float | None,
    back_iv: float | None,
    slope: float | None,
    dte: int | None,
    spot: float | None,
    existing_straddle_pct: float | None,
) -> dict:
    """Compute event pricing features from available IV inputs."""
    out: dict = {}

    # Straddle move approximation (BS ATM straddle ≈ STRADDLE_K * iv * sqrt(T))
    tenor = (dte / 365.0) if dte and dte > 0 else (TT_IV_TENOR_DAYS / 365.0)
    if atm_iv is not None and atm_iv > 0:
        approx_straddle_pct = round(STRADDLE_K * atm_iv * math.sqrt(tenor), 4)
        out["atm_straddle_move_pct_approx"] = approx_straddle_pct
    else:
        out["atm_straddle_move_pct_approx"] = None

    # Prefer existing model-computed straddle if available
    out["atm_straddle_move_pct_observed"] = existing_straddle_pct

    # Use observed if available, else approximation
    best_straddle = existing_straddle_pct if existing_straddle_pct is not None else out["atm_straddle_move_pct_approx"]
    out["atm_straddle_move_pct"] = best_straddle

    # Event premium from term structure
    if front_iv is not None and back_iv is not None and back_iv > 0:
        ep_pp = round(front_iv - back_iv, 4)
        ep_ratio = round(front_iv / back_iv, 4)
        out["event_premium_iv_pp"] = ep_pp
        out["event_premium_ratio"] = ep_ratio
        out["event_premium_flag"] = (
            "EVENT_PREMIUM_HIGH"
            if ep_pp > 0.20
            else ("EVENT_PREMIUM_MODERATE" if ep_pp > 0.05 else "NO_EVENT_PREMIUM_OBSERVED")
        )
    else:
        out["event_premium_iv_pp"] = None
        out["event_premium_ratio"] = None
        out["event_premium_flag"] = "NO_EVENT_EXPIRY" if front_iv is None else "NO_EVENT_PREMIUM_OBSERVED"

    # Jump diagnostics
    if front_iv is not None and back_iv is not None:
        jump_var = max(front_iv**2 - back_iv**2, 0.0)
        out["jump_premium_proxy"] = round(front_iv - back_iv, 4)
        out["jump_variance_proxy"] = round(jump_var, 6)
        out["jump_move_proxy_pct"] = round(math.sqrt(jump_var * tenor), 4) if jump_var > 0 else 0.0
    else:
        out["jump_premium_proxy"] = None
        out["jump_variance_proxy"] = None
        out["jump_move_proxy_pct"] = None

    # Term structure interpretation
    if slope is not None:
        if slope < -0.20:
            term_flag = "STEEP_BACKWARDATION"
        elif slope < -0.05:
            term_flag = "MILD_BACKWARDATION"
        elif slope > 0.50:
            term_flag = "STEEP_CONTANGO"
        elif slope > 0.10:
            term_flag = "CONTANGO"
        else:
            term_flag = "FLAT"
    else:
        term_flag = "NO_TERM_DATA"
    out["term_structure_flag"] = term_flag

    return out


# ---------------------------------------------------------------------------
# Layer 3 — IV vs realized volatility
# ---------------------------------------------------------------------------


def iv_rv_features(atm_iv: float | None, rv_20d: float | None, rv_60d: float | None) -> dict:
    out: dict = {}
    out["iv_minus_rv_20d"] = round(atm_iv - rv_20d, 4) if (atm_iv and rv_20d) else None
    out["iv_over_rv_20d"] = round(atm_iv / rv_20d, 4) if (atm_iv and rv_20d and rv_20d > 0) else None
    out["variance_risk_premium_20d"] = round(atm_iv**2 - rv_20d**2, 4) if (atm_iv and rv_20d) else None
    out["iv_minus_rv_60d"] = round(atm_iv - rv_60d, 4) if (atm_iv and rv_60d) else None
    out["iv_over_rv_60d"] = round(atm_iv / rv_60d, 4) if (atm_iv and rv_60d and rv_60d > 0) else None

    # VRP flag
    vrp = out["variance_risk_premium_20d"]
    if vrp is None:
        out["vrp_flag"] = "NO_DATA"
    elif vrp > 0.10:
        out["vrp_flag"] = "HIGH_VRP"
    elif vrp < -0.02:
        out["vrp_flag"] = "IV_BELOW_REALIZED"
    else:
        out["vrp_flag"] = "LOW_VRP"

    return out


# ---------------------------------------------------------------------------
# Layer 7 — Synthetic proxies
# ---------------------------------------------------------------------------


def synthetic_proxy(rv_60d: float | None, days_to_catalyst: float | None, rv_20d: float | None) -> dict:
    """Realized-vol proxy for tickers without listed options."""
    out: dict = {
        "synthetic_move_pct": None,
        "synthetic_move_source": "INSUFFICIENT_DATA",
        "synthetic_model": None,
        "synthetic_confidence": 0.0,
        "synthetic_warning": "NOT_MARKET_IMPLIED",
    }
    rv = rv_60d if rv_60d is not None else rv_20d
    if rv is not None and rv > 0 and days_to_catalyst is not None and days_to_catalyst > 0:
        t = float(days_to_catalyst) / 252.0
        move = round(rv * math.sqrt(t), 4)
        out["synthetic_move_pct"] = move
        out["synthetic_move_source"] = "REALIZED_VOL_PROXY"
        out["synthetic_model"] = f"rv_{'60d' if rv_60d else '20d'} * sqrt(days_to_catalyst/252)"
        out["synthetic_confidence"] = 0.3  # low — not market-implied
    return out


# ---------------------------------------------------------------------------
# Main enrichment pipeline
# ---------------------------------------------------------------------------


def build_enriched_features(
    opt_snap: dict,
    rankings_df: pd.DataFrame,
    rv_df: pd.DataFrame,
    xbi_df: pd.DataFrame,
    opt_diag_df: pd.DataFrame | None,
    as_of_date: str,
    include_synthetic: bool,
) -> pd.DataFrame:
    opt_tickers = opt_snap["tickers"]
    tickers = list(opt_tickers.keys())

    # Index lookups
    rk_idx = rankings_df.set_index("ticker") if "ticker" in rankings_df.columns else pd.DataFrame()
    rv_idx = rv_df.set_index("ticker") if not rv_df.empty else pd.DataFrame()
    xbi_idx = xbi_df.set_index("ticker") if not xbi_df.empty else pd.DataFrame()
    diag_idx = opt_diag_df.set_index("ticker") if opt_diag_df is not None and not opt_diag_df.empty else pd.DataFrame()

    rows = []
    for ticker in tickers:
        o = opt_tickers[ticker]
        row: dict = {"ticker": ticker, "as_of_date": as_of_date}

        # --- Identity fields from rankings ---
        rk = rk_idx.loc[ticker] if ticker in rk_idx.index else pd.Series(dtype=object)
        row["company_name"] = rk.get("company_name") if not rk.empty else None
        row["spot_price"] = _sf(rk.get("close_price")) if not rk.empty else None
        row["market_cap_mm"] = _sf(rk.get("market_cap_mm")) if not rk.empty else None
        row["actionable_rank"] = _sf(rk.get("actionable_rank")) if not rk.empty else None
        row["catalyst_days"] = _sf(rk.get("catalyst_days")) if not rk.empty else None
        row["catalyst_bucket"] = rk.get("catalyst_bucket") if not rk.empty else None
        row["catalyst_date"] = rk.get("next_catalyst_date") if not rk.empty else None

        # --- Coverage ---
        cov = _derive_coverage(o)
        row["has_options"] = 1 if cov in (VALID_OPTIONS, LOW_LIQUIDITY_CHAIN) else 0
        row["options_chain_status"] = cov

        # --- Observed IV fields (null for NO_CHAIN / FETCH_FAILED) ---
        atm_iv = _sf(o.get("opt_atm_iv")) if row["has_options"] else None
        front_iv = _sf(o.get("opt_front_iv")) if row["has_options"] else None
        back_iv = _sf(o.get("opt_back_iv")) if row["has_options"] else None
        slope = _sf(o.get("opt_term_slope")) if row["has_options"] else None
        iv_rk_tw = _sf(o.get("opt_iv_rank_tw")) if row["has_options"] else None
        iv_rk_tos = _sf(o.get("opt_iv_rank_tos")) if row["has_options"] else None
        iv_pct = _sf(o.get("opt_iv_percentile")) if row["has_options"] else None
        iv_5d_chg = _sf(o.get("opt_iv_5d_change")) if row["has_options"] else None
        liq_state = o.get("opt_liquidity_state") if row["has_options"] else None
        liq_rank = _sf(o.get("opt_liquidity_rank")) if row["has_options"] else None
        usable = o.get("opt_use_for_judgment", "NO")

        row["observed_atm_iv"] = atm_iv
        row["observed_front_iv"] = front_iv
        row["observed_back_iv"] = back_iv
        row["observed_term_slope"] = slope
        row["observed_iv_rank_tw"] = iv_rk_tw
        row["observed_iv_rank_tos"] = iv_rk_tos
        row["observed_iv_percentile"] = iv_pct
        row["observed_iv_5d_change"] = iv_5d_chg
        row["observed_liquidity_state"] = liq_state
        row["observed_liquidity_rank"] = liq_rank
        row["observed_iv_regime"] = o.get("opt_iv_regime") if row["has_options"] else None
        row["opt_use_for_judgment"] = usable

        # --- DTE from options diagnostics sidecar ---
        diag = diag_idx.loc[ticker] if ticker in diag_idx.index else pd.Series(dtype=object)
        dte = int(diag.get("opt_dte")) if not diag.empty and pd.notna(diag.get("opt_dte")) else None
        nearest_expiry = diag.get("opt_nearest_expiry") if not diag.empty else None
        row["nearest_expiry"] = nearest_expiry
        row["dte"] = dte

        # --- Event expiry availability ---
        # Approximation: event_expiry_available if catalyst_days fits in nearest expiry window
        cat_days = row["catalyst_days"]
        has_event_expiry = False
        if cat_days is not None and dte is not None:
            # Check if catalyst falls within ~7 days of the nearest expiry
            has_event_expiry = abs(float(cat_days) - float(dte)) <= 7 or float(cat_days) < float(dte)
        row["event_expiry_available"] = has_event_expiry

        # --- Quality score ---
        row["options_data_quality"] = _quality_score(cov, liq_state or "absent", has_event_expiry)

        # --- Layer 2: Realized vol ---
        rv = rv_idx.loc[ticker] if ticker in rv_idx.index else pd.Series(dtype=object)
        rv_10d = _sf(rv.get("rv_10d")) if not rv.empty else None
        rv_20d = _sf(rv.get("rv_20d")) if not rv.empty else None
        rv_60d = _sf(rv.get("rv_60d")) if not rv.empty else None
        rv_120d = _sf(rv.get("rv_120d")) if not rv.empty else None
        row["rv_10d"] = rv_10d
        row["rv_20d"] = rv_20d
        row["rv_60d"] = rv_60d
        row["rv_120d"] = rv_120d

        xbi = xbi_idx.loc[ticker] if ticker in xbi_idx.index else pd.Series(dtype=object)
        row["realized_beta_xbi_60d"] = _sf(xbi.get("realized_beta_xbi_60d")) if not xbi.empty else None
        row["idiosyncratic_vol_60d"] = _sf(xbi.get("idiosyncratic_vol_60d")) if not xbi.empty else None

        # --- Layer 3: IV vs RV ---
        existing_straddle_pct = (
            _sf(rk.get("priced_move_pct")) / 100.0 if not rk.empty and _sf(rk.get("priced_move_pct")) else None
        )
        ivr = iv_rv_features(atm_iv, rv_20d, rv_60d)
        row.update(ivr)

        # --- Layer 4/5: Event pricing and term structure ---
        ep = event_pricing_features(atm_iv, front_iv, back_iv, slope, dte, row["spot_price"], existing_straddle_pct)
        row.update(ep)

        # Existing model straddle (for reference)
        row["model_straddle_price"] = _sf(rk.get("straddle_price")) if not rk.empty else None
        row["model_priced_move_pct"] = _sf(rk.get("priced_move_pct")) if not rk.empty else None

        # --- Layer 6: Liquidity quality ---
        liq_map = {"liquid": 4, "thin": 2, "absent": 0}
        row["chain_liquidity_score"] = liq_map.get(liq_state or "absent", 0) if row["has_options"] else 0

        # --- Layer 7: Synthetic proxy (only for no-options names) ---
        syn_fields = {
            "synthetic_move_pct": None,
            "synthetic_move_source": None,
            "synthetic_model": None,
            "synthetic_confidence": None,
            "synthetic_warning": None,
        }
        if not row["has_options"] and include_synthetic:
            syn_fields = synthetic_proxy(rv_60d, cat_days, rv_20d)
        row.update(syn_fields)

        # --- Sell-only flag ---
        row["is_sell_only"] = 1 if ticker in SELL_ONLY_TICKERS else 0

        rows.append(row)

    df = pd.DataFrame(rows)

    # --- Layer 8: Cross-sectional features ---
    # event_premium_zscore within universe (names with valid event_premium_iv_pp)
    ep_col = df["event_premium_iv_pp"].dropna()
    if len(ep_col) >= 5:
        df["event_premium_zscore"] = ((df["event_premium_iv_pp"] - ep_col.mean()) / ep_col.std()).round(3)
    else:
        df["event_premium_zscore"] = None

    # iv_cross_section_percentile — rank of ATM IV within universe
    atm_valid = df["observed_atm_iv"].dropna()
    if len(atm_valid) >= 5:
        df["iv_cross_section_pctile"] = df["observed_atm_iv"].rank(pct=True).round(3)
    else:
        df["iv_cross_section_pctile"] = None

    # vrp cross-section percentile
    vrp_valid = df["variance_risk_premium_20d"].dropna()
    if len(vrp_valid) >= 5:
        df["vrp_cross_section_pctile"] = df["variance_risk_premium_20d"].rank(pct=True).round(3)
    else:
        df["vrp_cross_section_pctile"] = None

    # --- Layer 9: Flags ---
    flags_list = []
    for _, row in df.iterrows():
        flags = []
        cov = row.get("options_chain_status", "")
        if cov == NO_LISTED_OPTIONS:
            flags.append("NO_LISTED_OPTIONS")
        elif cov == FETCH_FAILED:
            flags.append("FETCH_FAILED")
        elif cov == LOW_LIQUIDITY_CHAIN:
            flags.append("LOW_LIQUIDITY_CHAIN")

        if row.get("observed_iv_regime") == "EXTREME":
            flags.append("EXTREME_IV")
        elif row.get("observed_iv_regime") == "ELEVATED":
            flags.append("ELEVATED_IV")

        ep_flag = row.get("event_premium_flag", "")
        if ep_flag == "EVENT_PREMIUM_HIGH":
            flags.append("EVENT_PREMIUM_HIGH")

        ts_flag = row.get("term_structure_flag", "")
        if "BACKWARDATION" in str(ts_flag):
            flags.append(ts_flag)
        elif "CONTANGO" in str(ts_flag):
            flags.append(ts_flag)

        vrp_flag = row.get("vrp_flag", "")
        if vrp_flag == "HIGH_VRP":
            flags.append("HIGH_VRP")
        elif vrp_flag == "IV_BELOW_REALIZED":
            flags.append("IV_BELOW_REALIZED")

        # IV momentum — 5d change in ATM IV (applies to any ticker with IV data,
        # including LOW_LIQUIDITY_CHAIN; meaningful even without full liquidity)
        iv_5d = _sf(row.get("observed_iv_5d_change"))
        if iv_5d is not None:
            if iv_5d >= 0.05:
                flags.append("IV_RISING")
            elif iv_5d <= -0.05:
                flags.append("IV_FALLING")

        # IV rank context for LOW_LIQ names — IVRk still reflects historical
        # positioning even when chain liquidity is absent
        iv_rk = _sf(row.get("observed_iv_rank_tw"))
        if iv_rk is not None and cov == LOW_LIQUIDITY_CHAIN:
            if iv_rk >= 0.70:
                flags.append("IV_RANK_HIGH")
            elif iv_rk >= 0.30:
                flags.append("IV_RANK_MODERATE")

        # High idiosyncratic vol — useful for no-chain names where realized vol
        # is the only signal available (proxy for name-specific binary risk)
        idio = _sf(row.get("idiosyncratic_vol_60d"))
        if idio is not None and idio >= 0.50 and cov == NO_LISTED_OPTIONS:
            flags.append("HIGH_IDIO_VOL")

        if row.get("is_sell_only") and iv_rk is not None:
            if iv_rk > 0.60:
                flags.append("SELL_ONLY_IV_PEAK")

        if row.get("synthetic_warning") == "NOT_MARKET_IMPLIED":
            flags.append("SYNTHETIC_ONLY_NOT_MARKET_IMPLIED")

        flags_list.append("|".join(flags) if flags else "CLEAN")

    df["options_flags"] = flags_list
    df["options_enrichment_version"] = ENRICHMENT_VERSION
    df["enrichment_classification"] = "OPTIONS_ENRICHMENT/EXPECTATION_LAYER_SHADOW/NO_MODEL_CHANGE"

    return df


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def build_report(df: pd.DataFrame, as_of_date: str, meta: dict) -> str:
    n = len(df)
    valid = (df["options_chain_status"] == VALID_OPTIONS).sum()
    low_liq = (df["options_chain_status"] == LOW_LIQUIDITY_CHAIN).sum()
    no_chain = (df["options_chain_status"] == NO_LISTED_OPTIONS).sum()
    failed = (df["options_chain_status"] == FETCH_FAILED).sum()
    usable = (df["opt_use_for_judgment"] == "YES").sum()
    event_prem_high = df["options_flags"].str.contains("EVENT_PREMIUM_HIGH", na=False).sum()
    extreme = df["options_flags"].str.contains("EXTREME_IV", na=False).sum()
    high_vrp = df["options_flags"].str.contains("HIGH_VRP", na=False).sum()
    iv_below_rv = df["options_flags"].str.contains("IV_BELOW_REALIZED", na=False).sum()

    # Top-30 specific
    top30 = df[df["actionable_rank"].notna() & (df["actionable_rank"] <= 30)].sort_values("actionable_rank")
    t30_valid = (top30["options_chain_status"] == VALID_OPTIONS).sum()
    t30_low = (top30["options_chain_status"] == LOW_LIQUIDITY_CHAIN).sum()
    t30_no = (top30["options_chain_status"] == NO_LISTED_OPTIONS).sum()

    ep_names = top30[top30["options_flags"].str.contains("EVENT_PREMIUM_HIGH", na=False)]["ticker"].tolist()
    extreme_names = top30[top30["options_flags"].str.contains("EXTREME_IV", na=False)]["ticker"].tolist()
    bw_names = top30[top30["options_flags"].str.contains("BACKWARDATION", na=False)]["ticker"].tolist()
    high_vrp_names = top30[top30["options_flags"].str.contains("HIGH_VRP", na=False)]["ticker"].tolist()
    iv_below_names = top30[top30["options_flags"].str.contains("IV_BELOW_REALIZED", na=False)]["ticker"].tolist()
    sell_timing = top30[top30["options_flags"].str.contains("SELL_ONLY_IV_PEAK", na=False)]["ticker"].tolist()
    synthetic_top30 = top30[top30["options_flags"].str.contains("SYNTHETIC_ONLY", na=False)]["ticker"].tolist()

    lines = [
        f"# Options Enrichment Report — {as_of_date}",
        "",
        "## Classification",
        "",
        "```",
        "OPTIONS_ENRICHMENT / EXPECTATION_LAYER_SHADOW / NO_MODEL_CHANGE /",
        "NO_RANKER_CHANGE / NO_SELECTOR_CHANGE / NO_SIZING_CHANGE / NO_TRADING_CHANGE",
        "```",
        "",
        "## Executive Verdict",
        "",
        f"Enriched options features computed for **{n} universe tickers**. "
        f"Coverage: {valid} valid, {low_liq} low-liquidity, {no_chain} no listed options, "
        f"{failed} fetch failures. "
        f"{event_prem_high} names show EVENT_PREMIUM_HIGH; {extreme} EXTREME IV; "
        f"{high_vrp} HIGH_VRP; {iv_below_rv} IV_BELOW_REALIZED. "
        f"All outputs are shadow-diagnostic only.",
        "",
        "## Coverage Summary",
        "",
        f"| Status | Universe ({n}) | Top-30 |",
        "|--------|---------------|--------|",
        f"| VALID_OPTIONS | {valid} | {t30_valid} |",
        f"| LOW_LIQUIDITY_CHAIN | {low_liq} | {t30_low} |",
        f"| NO_LISTED_OPTIONS | {no_chain} | {t30_no} |",
        f"| FETCH_FAILED | {failed} | 0 |",
        f"| Usable for judgment | {usable} | {(top30['opt_use_for_judgment'] == 'YES').sum()} |",
        "",
        "## Event-Pricing Diagnostics (Top-30)",
        "",
        f"**EVENT_PREMIUM_HIGH ({len(ep_names)}):** {', '.join(ep_names) or 'none'}",
        f"**EXTREME_IV ({len(extreme_names)}):** {', '.join(extreme_names) or 'none'}",
        f"**STEEP_BACKWARDATION ({len(bw_names)}):** {', '.join(bw_names) or 'none'}",
        "",
        "## IV/RV and Variance Risk Premium (Top-30)",
        "",
        f"**HIGH_VRP ({len(high_vrp_names)}):** {', '.join(high_vrp_names) or 'none'}",
        f"**IV_BELOW_REALIZED ({len(iv_below_names)}):** {', '.join(iv_below_names) or 'none'}",
        "",
    ]

    # VRP table for top-30 with data
    vrp_rows = top30[top30["iv_minus_rv_20d"].notna()][
        [
            "ticker",
            "observed_atm_iv",
            "rv_20d",
            "iv_minus_rv_20d",
            "iv_over_rv_20d",
            "variance_risk_premium_20d",
            "vrp_flag",
        ]
    ].sort_values("iv_over_rv_20d", ascending=False)
    if not vrp_rows.empty:
        lines.append("| Ticker | ATM_IV | RV_20d | IV-RV | IV/RV | VRP_20d | Flag |")
        lines.append("|--------|--------|--------|-------|-------|---------|------|")
        for _, r in vrp_rows.iterrows():
            lines.append(
                f"| {r['ticker']} | {r['observed_atm_iv']*100:.0f}% | {r['rv_20d']*100:.0f}% | "
                f"{r['iv_minus_rv_20d']:+.3f} | {r['iv_over_rv_20d']:.2f}x | "
                f"{r['variance_risk_premium_20d']:.4f} | {r['vrp_flag']} |"
            )
        lines.append("")

    lines += [
        "## Sell Timing Flags",
        "",
        f"**SELL_ONLY_IV_PEAK:** {', '.join(sell_timing) or 'none'}",
        "",
        "## Synthetic Proxy Coverage (Top-30)",
        "",
        f"**SYNTHETIC_ONLY_NOT_MARKET_IMPLIED:** {', '.join(synthetic_top30) or 'none'}",
        "  (Realized-vol proxy; not market-implied pricing.)",
        "",
        "## Implemented Layers",
        "",
        "| Layer | Feature | Source | Status |",
        "|-------|---------|--------|--------|",
        "| 1 | Coverage normalization (VALID/LOW_LIQ/NO_CHAIN/FAILED) | TT snapshot | ✓ |",
        "| 2 | Realized vol 10/20/60/120d, XBI beta, idiosyncratic vol | Price history | ✓ |",
        "| 3 | IV−RV, IV/RV, variance risk premium, VRP flag | TT + price | ✓ |",
        "| 4 | Event premium IV pp/ratio/zscore, jump proxy | TT term structure | ✓ |",
        "| 5 | Straddle move pct (observed + BS approximation) | Rankings + TT | ✓ |",
        "| 6 | Term structure flag (backwardation/contango/flat) | TT | ✓ |",
        "| 7 | Liquidity quality score 0-5 | TT | ✓ |",
        "| 8 | Synthetic move proxy for no-options tickers | Price history | ✓ |",
        "| 9 | Cross-sectional zscore/percentile | Universe | ✓ |",
        "| — | 25-delta skew / risk reversal / butterfly | Chains | ✗ insufficient data |",
        "| — | Put/call volume & OI ratio | Chains | ✗ no per-contract data |",
        "| — | Dealer GEX / pinning proxy | Chains + Greeks | ✗ no per-contract data |",
        "| — | Full IV surface (30/60/90d tenor breakdown) | Chains | ✗ single tenor from TT |",
        "",
        "## Known Limitations",
        "",
        "- **Single IV tenor**: TT `implied_volatility_index` is approximately 30-day constant-maturity. "
        "  Separate 30/60/90d ATM IVs require per-expiry chain data.",
        "- **No per-contract Greeks**: 25-delta skew, risk reversal, butterfly, and GEX require individual "
        "  contract bid/ask/OI data not currently fetched.",
        "- **Straddle approximation**: Uses BS ATM approximation (0.7979 × IV × √T) where model-computed "
        "  `priced_move_pct` from rankings.csv is unavailable.",
        "- **HV from TT**: `opt_hv_30d/60d/90d` fields return null from tastytrade for most biotech names; "
        "  RV computed from price history instead.",
        "- **Parkinson/Garman-Klass vol**: Requires OHLC; only daily close is available.",
        "",
        "## Governance Conclusion",
        "",
        "No model, ranker, selector, sizing, or trading changes. Options features are enriched "
        "for shadow diagnostics, action-card context, and future validation only. "
        "Observed option fields are null for NO_LISTED_OPTIONS tickers. "
        "Synthetic proxies are labeled NOT_MARKET_IMPLIED throughout.",
        "",
        f"*Generated: {datetime.now(timezone.utc).isoformat()}*",
        f"*Enrichment version: {ENRICHMENT_VERSION}*",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run(
    as_of_date: str,
    output_dir: Path,
    write_shadow_only: bool,
    include_synthetic: bool,
    verbose: bool,
) -> dict:
    data_dir = REPO_ROOT / "production_data"
    snap_dir = REPO_ROOT / "data" / "snapshots"

    # Load the options snapshot FOR as_of_date. "latest" is only valid when it
    # is labelled with as_of_date; a catch-up run for a past date must not read
    # a newer chain (look-ahead). Falls back to the dated file, else refuses.
    opt_snap = None
    snap_path = None
    for cand in (data_dir / "options_snapshot_latest.json", data_dir / f"options_snapshot_{as_of_date}.json"):
        if not cand.exists():
            continue
        data = json.loads(cand.read_text())
        if data.get("metadata", {}).get("as_of_date") == as_of_date:
            opt_snap, snap_path = data, cand
            break
    if opt_snap is None:
        raise FileNotFoundError(
            f"No options snapshot labelled {as_of_date} (latest/dated); refusing to enrich from another date"
        )
    if verbose:
        print(f"Options snapshot: {snap_path.name}  ({len(opt_snap['tickers'])} tickers)")

    # Load the newest rankings ON OR BEFORE as_of_date (date-named dirs only).
    # Production runs this before today's screen, so this is normally the prior
    # day's rankings; a catch-up run must never see a later date's rankings.
    import re as _re

    _date_dir = _re.compile(r"^\d{4}-\d{2}-\d{2}$")
    rk_paths = sorted(
        p for p in snap_dir.glob("*/rankings.csv") if _date_dir.match(p.parent.name) and p.parent.name <= as_of_date
    )
    if not rk_paths:
        raise FileNotFoundError(f"No rankings snapshot on or before {as_of_date}")
    rk_path = rk_paths[-1]
    rankings_df = pd.read_csv(rk_path)
    snap_date = rk_path.parent.name
    if verbose:
        print(f"Rankings: {snap_date}  ({len(rankings_df)} rows)")

    # Load options diagnostics sidecar
    diag_path = rk_path.parent / "options_diagnostics.csv"
    opt_diag_df = pd.read_csv(diag_path) if diag_path.exists() else None
    if verbose and opt_diag_df is not None:
        print(f"Options diagnostics: {len(opt_diag_df)} rows")

    # Load price history for RV computation
    price_path = data_dir / "price_history_split_adj.csv"
    price_df = pd.read_csv(price_path)
    if verbose:
        print(f"Price history: {len(price_df)} rows  ({price_df['ticker'].nunique()} tickers)")

    # Compute realized vols
    if verbose:
        print("Computing realized volatilities...")
    rv_df = compute_realized_vols(price_df, as_of_date, windows=[10, 20, 60, 120])
    xbi_df = compute_xbi_features(price_df, as_of_date, window=60)

    # Build enriched features
    if verbose:
        print("Building enriched features...")
    df = build_enriched_features(opt_snap, rankings_df, rv_df, xbi_df, opt_diag_df, as_of_date, include_synthetic)

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    # Main enriched features
    df.to_csv(output_dir / "options_enriched_features.csv", index=False)

    # Coverage summary
    cov_summary = (
        df.groupby("options_chain_status")
        .agg(count=("ticker", "count"), usable=("opt_use_for_judgment", lambda x: (x == "YES").sum()))
        .reset_index()
    )
    cov_summary.to_csv(output_dir / "options_coverage_summary.csv", index=False)

    # Surface diagnostics
    surf_cols = [
        "ticker",
        "as_of_date",
        "observed_atm_iv",
        "observed_front_iv",
        "observed_back_iv",
        "observed_term_slope",
        "term_structure_flag",
        "observed_iv_rank_tw",
        "observed_iv_rank_tos",
        "observed_iv_percentile",
        "observed_iv_5d_change",
        "observed_iv_regime",
        "nearest_expiry",
        "dte",
    ]
    df[[c for c in surf_cols if c in df.columns]].to_csv(output_dir / "options_surface_diagnostics.csv", index=False)

    # Event pricing
    event_cols = [
        "ticker",
        "as_of_date",
        "catalyst_date",
        "catalyst_days",
        "catalyst_bucket",
        "event_expiry_available",
        "atm_straddle_move_pct",
        "atm_straddle_move_pct_approx",
        "model_priced_move_pct",
        "model_straddle_price",
        "event_premium_iv_pp",
        "event_premium_ratio",
        "event_premium_zscore",
        "event_premium_flag",
        "jump_premium_proxy",
        "jump_move_proxy_pct",
    ]
    df[[c for c in event_cols if c in df.columns]].to_csv(output_dir / "options_event_pricing.csv", index=False)

    # Liquidity quality
    liq_cols = [
        "ticker",
        "as_of_date",
        "options_chain_status",
        "options_data_quality",
        "chain_liquidity_score",
        "observed_liquidity_state",
        "observed_liquidity_rank",
        "opt_use_for_judgment",
        "event_expiry_available",
    ]
    df[[c for c in liq_cols if c in df.columns]].to_csv(output_dir / "options_liquidity_quality.csv", index=False)

    # IV/RV comparison
    ivr_cols = [
        "ticker",
        "as_of_date",
        "observed_atm_iv",
        "rv_10d",
        "rv_20d",
        "rv_60d",
        "rv_120d",
        "realized_beta_xbi_60d",
        "idiosyncratic_vol_60d",
        "iv_minus_rv_20d",
        "iv_over_rv_20d",
        "variance_risk_premium_20d",
        "iv_minus_rv_60d",
        "iv_over_rv_60d",
        "vrp_flag",
        "iv_cross_section_pctile",
        "vrp_cross_section_pctile",
    ]
    df[[c for c in ivr_cols if c in df.columns]].to_csv(output_dir / "options_iv_rv_diagnostics.csv", index=False)

    # Synthetic proxy
    syn_cols = [
        "ticker",
        "as_of_date",
        "options_chain_status",
        "rv_60d",
        "rv_20d",
        "catalyst_days",
        "catalyst_bucket",
        "synthetic_move_pct",
        "synthetic_move_source",
        "synthetic_model",
        "synthetic_confidence",
        "synthetic_warning",
    ]
    syn_df = df[df["options_chain_status"].isin([NO_LISTED_OPTIONS, FETCH_FAILED])]
    syn_df[[c for c in syn_cols if c in df.columns]].to_csv(output_dir / "options_synthetic_proxy.csv", index=False)

    # Markdown report
    meta = {
        "as_of_date": as_of_date,
        "options_snapshot": snap_path.name,
        "rankings_date": snap_date,
        "enrichment_version": ENRICHMENT_VERSION,
    }
    report_md = build_report(df, as_of_date, meta)
    (output_dir / "OPTIONS_ENRICHMENT_REPORT.md").write_text(report_md)

    # JSON summary
    summary = {
        "as_of_date": as_of_date,
        "enrichment_version": ENRICHMENT_VERSION,
        "classification": "OPTIONS_ENRICHMENT/EXPECTATION_LAYER_SHADOW/NO_MODEL_CHANGE",
        "universe_count": len(df),
        "coverage": {
            "valid_options": int((df["options_chain_status"] == VALID_OPTIONS).sum()),
            "low_liquidity": int((df["options_chain_status"] == LOW_LIQUIDITY_CHAIN).sum()),
            "no_listed_options": int((df["options_chain_status"] == NO_LISTED_OPTIONS).sum()),
            "fetch_failed": int((df["options_chain_status"] == FETCH_FAILED).sum()),
            "usable_for_judgment": int((df["opt_use_for_judgment"] == "YES").sum()),
        },
        "flags": {
            "extreme_iv": int(df["options_flags"].str.contains("EXTREME_IV", na=False).sum()),
            "event_premium_high": int(df["options_flags"].str.contains("EVENT_PREMIUM_HIGH", na=False).sum()),
            "steep_backwardation": int(df["options_flags"].str.contains("BACKWARDATION", na=False).sum()),
            "high_vrp": int(df["options_flags"].str.contains("HIGH_VRP", na=False).sum()),
            "iv_below_realized": int(df["options_flags"].str.contains("IV_BELOW_REALIZED", na=False).sum()),
            "iv_rising": int(df["options_flags"].str.contains("IV_RISING", na=False).sum()),
            "iv_falling": int(df["options_flags"].str.contains("IV_FALLING", na=False).sum()),
            "iv_rank_high": int(df["options_flags"].str.contains("IV_RANK_HIGH", na=False).sum()),
            "iv_rank_moderate": int(df["options_flags"].str.contains("IV_RANK_MODERATE", na=False).sum()),
            "high_idio_vol": int(df["options_flags"].str.contains("HIGH_IDIO_VOL", na=False).sum()),
            "sell_only_iv_peak": int(df["options_flags"].str.contains("SELL_ONLY_IV_PEAK", na=False).sum()),
            "synthetic_only": int(df["options_flags"].str.contains("SYNTHETIC_ONLY", na=False).sum()),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "write_shadow_only": write_shadow_only,
    }
    (output_dir / "options_enrichment_summary.json").write_text(json.dumps(summary, indent=2))

    if verbose:
        print(f"\nWritten to: {output_dir}/")
        for f in sorted(output_dir.iterdir()):
            print(f"  {f.name}  ({f.stat().st_size / 1024:.1f} KB)")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Options feature enrichment — shadow layer")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "artifacts" / "options_enrichment" / date.today().strftime("%Y_%m_%d")),
    )
    parser.add_argument("--write-shadow-only", action="store_true", default=True)
    parser.add_argument("--no-synthetic", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run(
        as_of_date=args.date,
        output_dir=Path(args.output_dir),
        write_shadow_only=args.write_shadow_only,
        include_synthetic=not args.no_synthetic,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
