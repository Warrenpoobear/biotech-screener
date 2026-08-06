#!/usr/bin/env python3
"""Run the decision-engine screen from PIT bundles.

Constructs rec dicts from PIT bundle data (clinical, catalyst, coinvest),
enriches with price-derived features and financial records, runs the
decision engine, and outputs rankings.csv + metadata.json per date.

Data flow:
    PIT Bundle (3 JSONs) ─┐
    universe.json ─────────┼─> build rec dicts ─> decision engine ─> sort ─> rankings.csv
    financial_records.json ─┤
    price_history.csv ──────┘
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Constants (replicated from run_screen.py for PIT-safe isolation)
# ---------------------------------------------------------------------------

INDUSTRY_TO_ARCHETYPE = {
    "Biotechnology": "drug_developer",
    "Drug Manufacturers - Specialty & Generic": "commercial_pharma",
    "Drug Manufacturers - General": "commercial_pharma",
    "Diagnostics & Research": "platform_diagnostics",
    "Medical Devices": "platform_devices",
    "Medical Instruments & Supplies": "platform_devices",
    "Health Information Services": "platform_services",
    "Medical Care Facilities": "platform_services",
}

# Price-feature computation constants (mirrored from run_screen.py)
DRAWDOWN_WINDOW = 252
MIN_BARS_FOR_ESTIMATE = 126
BETA_WINDOW = 60
MIN_OVERLAP_BARS = 30
RSI_PERIOD = 14
MIN_BARS_RSI = 15
XBI_STALE_THRESHOLD = 3

VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Phase A — Load & validate
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_pit_quality(manifest: dict) -> dict:
    """Read pit_quality with fallback to legacy pit_violations key."""
    pq = manifest.get("pit_quality")
    if pq is not None:
        return pq
    # Legacy manifests
    pv = manifest.get("pit_violations", {})
    if not pv:
        return {}
    return {
        "ctgov_trials_filtered_future_posted": pv.get("ctgov_first_posted", 0),
        "catalyst_tickers_missing_disclosed_at": pv.get("catalyst_missing_disclosed_at", 0),
        "catalyst_tickers_future_disclosed_at": pv.get("catalyst_future_disclosed_at", 0),
    }


def load_bundle(bundle_dir: Path, pit_mode: str = "strict") -> Optional[dict]:
    """Load manifest + component files from a single-date bundle.

    Returns dict with keys: manifest, clinical, catalyst, coinvest, as_of_date,
    skip_reason (None if OK).
    In strict mode, skips dates with catalyst future PIT violations.
    """
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        return None

    manifest = _load_json(manifest_path)
    as_of_date = manifest.get("as_of_date", bundle_dir.name)

    pq = _get_pit_quality(manifest)
    if pit_mode == "strict" and pq.get("catalyst_tickers_future_disclosed_at", 0) > 0:
        return {
            "manifest": manifest,
            "clinical": {},
            "catalyst": {},
            "coinvest": {},
            "as_of_date": as_of_date,
            "skip_reason": f"PIT:catalyst_future_violations={pq['catalyst_tickers_future_disclosed_at']}",
        }

    components = manifest.get("components", {})

    def _load_component(name: str) -> dict:
        info = components.get(name)
        if info is None:
            return {}
        fpath = bundle_dir / info["file"]
        if not fpath.exists():
            return {}
        data = _load_json(fpath)
        return data.get("tickers", data.get("data", data))

    clinical = _load_component("clinical_features")
    catalyst = _load_component("catalyst_events")
    coinvest = _load_component("coinvest_features")

    return {
        "manifest": manifest,
        "clinical": clinical,
        "catalyst": catalyst,
        "coinvest": coinvest,
        "as_of_date": as_of_date,
        "skip_reason": None,
    }


def discover_bundle_dates(
    bundle_root: Path,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> List[str]:
    """Find all YYYY-MM-DD subdirectories with manifest.json."""
    dates = []
    if not bundle_root.exists():
        return dates
    for p in sorted(bundle_root.iterdir()):
        if not p.is_dir():
            continue
        name = p.name
        if len(name) != 10 or name[4] != "-" or name[7] != "-":
            continue
        if not (p / "manifest.json").exists():
            continue
        if date_from and name < date_from:
            continue
        if date_to and name > date_to:
            continue
        dates.append(name)
    return dates


# ---------------------------------------------------------------------------
# Phase B — Build per-ticker rec dicts
# ---------------------------------------------------------------------------


def classify_company_archetype(
    ticker: str,
    industry: str,
    has_revenue: bool,
    revenue_scale_bucket: str,
) -> str:
    """Classify company archetype from industry + revenue signals."""
    archetype = INDUSTRY_TO_ARCHETYPE.get(industry, "drug_developer")
    if archetype == "drug_developer" and has_revenue and revenue_scale_bucket in ("medium", "large"):
        archetype = "commercial_biotech"
    return archetype


def _classify_stage_bucket(lead_phase: str) -> str:
    """Map lead clinical phase to stage bucket."""
    lp = (lead_phase or "").lower().strip()
    if "3" in lp:
        return "late"
    if "2" in lp:
        return "mid"
    return "early"


def _compute_severity(fin_rec: dict) -> Tuple[str, bool, float]:
    """Compute severity, fundamental_red_flag, and effective_runway_months.

    Returns (severity_str, red_flag_bool, runway_months).
    """
    cash = fin_rec.get("Cash") or fin_rec.get("TotalLiquidity")
    if cash is None:
        return "NONE", False, 999.0

    try:
        cash = float(cash)
    except (ValueError, TypeError):
        return "NONE", False, 999.0

    # Compute quarterly burn rate from operating expenses and revenue
    opex = fin_rec.get("OperatingExpenses")
    revenue = fin_rec.get("Revenue")
    cfo = fin_rec.get("CFO")

    quarterly_burn = None
    if cfo is not None:
        try:
            # CFO is TTM; quarterly = CFO / 4; negative CFO = burning cash
            cfo_val = float(cfo)
            if cfo_val < 0:
                quarterly_burn = abs(cfo_val) / 4.0
        except (ValueError, TypeError):
            pass

    if quarterly_burn is None and opex is not None:
        try:
            opex_val = float(opex)
            rev_val = float(revenue) if revenue is not None else 0.0
            # TTM net burn: opex - revenue; quarterly = / 4
            net_burn = opex_val - rev_val
            if net_burn > 0:
                quarterly_burn = net_burn / 4.0
        except (ValueError, TypeError):
            pass

    if quarterly_burn is None or quarterly_burn <= 0:
        return "NONE", False, 999.0

    monthly_burn = quarterly_burn / 3.0
    runway_months = cash / monthly_burn if monthly_burn > 0 else 999.0

    if runway_months < 6:
        return "SEV3", True, runway_months
    elif runway_months < 12:
        return "SEV2", False, runway_months
    elif runway_months < 18:
        return "SEV1", False, runway_months
    else:
        return "NONE", False, runway_months


def _derive_smart_money(coinvest_data: dict) -> dict:
    """Derive smart_money_signal dict from coinvest position_changes."""
    pc = coinvest_data.get("position_changes", {})
    holders_increasing = []
    holders_decreasing = []
    for mgr, change in pc.items():
        if change in ("NEW", "INCREASE"):
            holders_increasing.append(mgr)
        elif change == "DECREASE":
            holders_decreasing.append(mgr)

    return {
        "overlap_count": coinvest_data.get("sponsor_overlap_count", coinvest_data.get("coinvest_overlap_count", 0)),
        "holders_increasing": holders_increasing,
        "holders_decreasing": holders_decreasing,
        "tier_breakdown": coinvest_data.get("holder_tiers", {}),
    }


def _load_price_history(price_csv: Path) -> Optional["pd.DataFrame"]:
    """Load price_history.csv into a wide DataFrame (Date index, ticker columns).

    Handles both wide format (Date, ACAD, BMRN, ...) and long format
    (date, ticker, close, ...) used in production_data/price_history.csv.
    """
    try:
        import pandas as pd

        df = pd.read_csv(price_csv)
        cols_lower = [c.lower() for c in df.columns]
        if "ticker" in cols_lower and "close" in cols_lower:
            # Long format → pivot to wide
            date_col = "date" if "date" in cols_lower else "Date"
            "ticker" if "ticker" in cols_lower else "Ticker"
            "close" if "close" in cols_lower else "Close"
            # Use actual column names (case-sensitive)
            actual_cols = {c.lower(): c for c in df.columns}
            dc = actual_cols.get("date", "Date")
            tc = actual_cols.get("ticker", "Ticker")
            cc = actual_cols.get("close", "Close")
            df[dc] = pd.to_datetime(df[dc])
            wide = df.pivot(index=dc, columns=tc, values=cc).sort_index()
            wide.index.name = "Date"
            return wide
        else:
            # Already wide format
            date_col = "Date" if "Date" in df.columns else "date"
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col).sort_index()
            df.index.name = "Date"
            return df
    except Exception as e:
        print(f"  WARN: failed to load price history: {e}", file=sys.stderr)
        return None


def _compute_price_features(
    prices_df: "pd.DataFrame",
    ticker: str,
    as_of_date: str,
    xbi_series: Optional["pd.Series"] = None,
) -> dict:
    """Compute PIT-safe price features for a single ticker.

    Returns dict with: drawdown, drawdown_xbi, drawdown_rel_xbi,
    beta_xbi_60d, alpha_60d, rsi_14d, vol_60d, and source/missing_reason fields.
    """
    import numpy as np
    import pandas as pd

    result = {
        "drawdown": None,
        "drawdown_xbi": None,
        "drawdown_rel_xbi": None,
        "beta_xbi_60d": None,
        "alpha_60d": None,
        "rsi_14d": None,
        "vol_60d": None,
        "beta_xbi_60d_source": "",
        "beta_xbi_60d_missing_reason": "",
        "alpha_60d_source": "",
        "alpha_60d_missing_reason": "",
        "drawdown_missing_reason": "",
    }

    cutoff = pd.Timestamp(as_of_date)
    if ticker not in prices_df.columns:
        result["drawdown_missing_reason"] = "no_price_series"
        result["beta_xbi_60d_missing_reason"] = "no_price_series"
        result["alpha_60d_missing_reason"] = "no_price_series"
        return result

    series = prices_df[ticker].loc[:cutoff].dropna()
    if len(series) < MIN_BARS_FOR_ESTIMATE:
        result["drawdown_missing_reason"] = "series_too_short"
        result["beta_xbi_60d_missing_reason"] = "series_too_short"
        result["alpha_60d_missing_reason"] = "series_too_short"
        return result

    closes = series.values

    # --- Drawdown ---
    window_closes = closes[-DRAWDOWN_WINDOW:] if len(closes) >= DRAWDOWN_WINDOW else closes
    peak = float(np.max(window_closes))
    current = float(closes[-1])
    if peak > 0:
        result["drawdown"] = round((current / peak) - 1.0, 6)

    # XBI drawdown
    if xbi_series is not None:
        xbi_pit = xbi_series.loc[:cutoff].dropna()
        if len(xbi_pit) >= MIN_BARS_FOR_ESTIMATE:
            xbi_closes = xbi_pit.values
            xbi_window = xbi_closes[-DRAWDOWN_WINDOW:] if len(xbi_closes) >= DRAWDOWN_WINDOW else xbi_closes
            xbi_peak = float(np.max(xbi_window))
            xbi_current = float(xbi_closes[-1])
            if xbi_peak > 0:
                result["drawdown_xbi"] = round((xbi_current / xbi_peak) - 1.0, 6)
                if result["drawdown"] is not None:
                    result["drawdown_rel_xbi"] = round(result["drawdown"] - result["drawdown_xbi"], 6)

    # --- RSI ---
    if len(closes) >= MIN_BARS_RSI:
        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains = [max(d, 0) for d in deltas]
        losses = [max(-d, 0) for d in deltas]
        avg_gain = sum(gains[:RSI_PERIOD]) / RSI_PERIOD
        avg_loss = sum(losses[:RSI_PERIOD]) / RSI_PERIOD
        for i in range(RSI_PERIOD, len(deltas)):
            avg_gain = (avg_gain * (RSI_PERIOD - 1) + gains[i]) / RSI_PERIOD
            avg_loss = (avg_loss * (RSI_PERIOD - 1) + losses[i]) / RSI_PERIOD
        if avg_loss == 0:
            result["rsi_14d"] = 100.0
        else:
            rs = avg_gain / avg_loss
            result["rsi_14d"] = round(100.0 - (100.0 / (1.0 + rs)), 4)

    # --- Volatility (60d) ---
    if len(closes) >= 61:
        log_rets = [math.log(closes[i] / closes[i - 1]) for i in range(max(1, len(closes) - 60), len(closes))]
        if len(log_rets) >= 2:
            mu = sum(log_rets) / len(log_rets)
            var = sum((r - mu) ** 2 for r in log_rets) / (len(log_rets) - 1)
            result["vol_60d"] = round(math.sqrt(var) * math.sqrt(252), 4)

    # --- Beta + Alpha ---
    if xbi_series is None:
        result["beta_xbi_60d_missing_reason"] = "no_xbi_series"
        result["alpha_60d_missing_reason"] = "no_xbi_series"
        return result

    xbi_pit = xbi_series.loc[:cutoff].dropna()

    # XBI staleness check
    if len(xbi_pit) > 0 and len(series) > 0:
        xbi_last = xbi_pit.index[-1]
        ticker_last = series.index[-1]
        # Count trading days gap
        gap_days = (ticker_last - xbi_last).days
        if gap_days > XBI_STALE_THRESHOLD:
            result["beta_xbi_60d_missing_reason"] = "xbi_stale"
            result["alpha_60d_missing_reason"] = "xbi_stale"
            return result

    # Align on common dates
    common_idx = series.index.intersection(xbi_pit.index)
    if len(common_idx) < MIN_OVERLAP_BARS + 1:
        result["beta_xbi_60d_missing_reason"] = "insufficient_overlap"
        result["alpha_60d_missing_reason"] = "insufficient_overlap"
        return result

    # Take last BETA_WINDOW+1 aligned bars
    aligned_idx = common_idx[-(BETA_WINDOW + 1) :]
    t_vals = series.loc[aligned_idx].values
    x_vals = xbi_pit.loc[aligned_idx].values

    # Daily returns
    n_rets = len(t_vals) - 1
    if n_rets < 2:
        result["beta_xbi_60d_missing_reason"] = "insufficient_overlap"
        result["alpha_60d_missing_reason"] = "insufficient_overlap"
        return result

    t_rets = [(t_vals[i] / t_vals[i - 1]) - 1.0 for i in range(1, len(t_vals))]
    x_rets = [(x_vals[i] / x_vals[i - 1]) - 1.0 for i in range(1, len(x_vals))]

    t_mean = sum(t_rets) / n_rets
    x_mean = sum(x_rets) / n_rets
    cov_tx = sum((t_rets[i] - t_mean) * (x_rets[i] - x_mean) for i in range(n_rets)) / n_rets
    var_x = sum((x_rets[i] - x_mean) ** 2 for i in range(n_rets)) / n_rets

    if var_x < 1e-14:
        result["beta_xbi_60d_missing_reason"] = "zero_var_benchmark"
        result["alpha_60d_missing_reason"] = "beta_missing:zero_var_benchmark"
        return result

    beta = round(cov_tx / var_x, 4)
    result["beta_xbi_60d"] = beta
    result["beta_xbi_60d_source"] = "price_history"

    # Alpha: cumulative excess return
    t_cum = (t_vals[-1] / t_vals[0]) - 1.0
    x_cum = (x_vals[-1] / x_vals[0]) - 1.0
    alpha = round(t_cum - beta * x_cum, 6)
    result["alpha_60d"] = alpha
    result["alpha_60d_source"] = "price_history"

    return result


def _build_rec(
    ticker: str,
    clinical: dict,
    catalyst: dict,
    coinvest_data: dict,
    fin_rec: dict,
    univ_entry: dict,
    price_feats: dict,
) -> dict:
    """Assemble one rec dict satisfying compute_decision_fields() contract."""
    # Catalyst decay (top-level)
    cat = catalyst.get(ticker, {})
    days_to_cat = cat.get("days_to_catalyst")
    cat_mode = cat.get("catalyst_mode", "missing")
    in_optimal_window = cat_mode == "specific_days" and days_to_cat is not None and days_to_cat > 0

    # Coinvest fields
    coinv = coinvest_data.get(ticker, {})

    # Smart money signal
    smart_money = _derive_smart_money(coinv)

    # Clinical data
    clin = clinical.get(ticker, {})
    clinical_score = clin.get("clinical_alpha_raw", 0.0)

    # Severity from financial records
    severity, red_flag, runway_months = _compute_severity(fin_rec)

    # Defensive features from price computation
    df = {
        "drawdown": price_feats.get("drawdown"),
        "drawdown_xbi": price_feats.get("drawdown_xbi"),
        "drawdown_rel_xbi": price_feats.get("drawdown_rel_xbi"),
        "vol_60d": price_feats.get("vol_60d"),
        "beta_xbi_60d": price_feats.get("beta_xbi_60d"),
        "rsi_14d": price_feats.get("rsi_14d"),
    }

    # Alpha for momentum signal
    alpha_60d = price_feats.get("alpha_60d")

    rec = {
        "ticker": ticker,
        "catalyst_decay": {
            "days_to_catalyst": days_to_cat,
            "in_optimal_window": in_optimal_window,
        },
        "coinvest": {
            "tier1_count": coinv.get("tier1_count", coinv.get("sponsor_tier1_count")),
            "conviction_overlap": coinv.get("conviction_overlap", 0.0),
            "tier1_conviction_overlap": coinv.get("tier1_conviction_overlap", 0.0),
            "max_tier1_position_pct": coinv.get("max_tier1_position_pct", 0.0),
            "days_since_latest_filing": coinv.get("days_since_latest_filing"),
        },
        "smart_money_signal": smart_money,
        "defensive_features": df,
        "score_breakdown": {
            "enhancements": {
                "momentum": {
                    "alpha_60d": alpha_60d,
                },
            },
        },
        "momentum_signal": {"alpha_60d": alpha_60d},
        "fundamental_red_flag": red_flag,
        "fundamental_red_flag_reasons": ["cash_runway_lt_6m"] if red_flag else [],
        "severity": severity,
        "confidence_overall": 0.5,
        "survivability_signal": {"effective_runway_months": runway_months},
        # Module score placeholders (not available from bundles)
        "momentum_score": 0.0,
        "catalyst_score": 0.0,
        "smart_money_score": 0.0,
        "valuation_score": 0.0,
        "clinical_score": clinical_score,
        "financial_score": 0.0,
        # Composite placeholders
        "composite_score": 0.0,
        "composite_rank": None,
        "score_rank_pct": None,
        "score_z": None,
        # Extras for snapshot
        "catalyst_event_type": cat.get("nearest_event_type", ""),
        "catalyst_source": cat.get("nearest_event_source", ""),
        # Price source tracking
        "de_beta_xbi_60d_source": price_feats.get("beta_xbi_60d_source", ""),
        "de_beta_xbi_60d_missing_reason": price_feats.get("beta_xbi_60d_missing_reason", ""),
        "de_alpha_60d_source": price_feats.get("alpha_60d_source", ""),
        "de_alpha_60d_missing_reason": price_feats.get("alpha_60d_missing_reason", ""),
        "de_drawdown_missing_reason": price_feats.get("drawdown_missing_reason", ""),
    }

    return rec


# ---------------------------------------------------------------------------
# Phase C — Score & rank
# ---------------------------------------------------------------------------


def _safe_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        f = float(v)
        if f != f:  # NaN
            return None
        return f
    except (ValueError, TypeError):
        return None


def _z_score_by_group(rows: List[Dict[str, Any]], value_col: str, group_col: str, out_col: str) -> None:
    """Compute per-group z-scores (ddof=0) and write to out_col."""
    groups: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        g = r.get(group_col, "") or "unknown"
        groups.setdefault(g, []).append(i)

    for g, indices in groups.items():
        vals = []
        for i in indices:
            v = _safe_float(rows[i].get(value_col))
            if v is not None:
                vals.append((i, v))

        if len(vals) < 2:
            for i in indices:
                rows[i][out_col] = ""
            continue

        raw = [v for _, v in vals]
        mu = statistics.mean(raw)
        var = sum((x - mu) ** 2 for x in raw) / len(raw)
        std = var**0.5

        for i in indices:
            v = _safe_float(rows[i].get(value_col))
            if v is not None and std > 0:
                rows[i][out_col] = round((v - mu) / std, 6)
            else:
                rows[i][out_col] = ""


def _z_score_sparse(
    rows: List[Dict[str, Any]],
    value_col: str,
    out_col: str,
    flag_col: str = "",
    sparse_mode: str = "legacy",
) -> None:
    """Cross-sectional z-score (ddof=0) with sparse signal handling.

    When sparse_mode="exclude_missing" and flag_col is provided, only rows
    where row[flag_col] == "True" contribute to mean/std.  Rows without signal
    get z=0 (neutral).  In "legacy" mode, all rows contribute (missing → 0).
    """
    pairs: List[Tuple[int, float, bool]] = []
    for i, r in enumerate(rows):
        v = _safe_float(r.get(value_col))
        has_signal = (r.get(flag_col, "") == "True") if flag_col else True
        if v is not None:
            pairs.append((i, v, has_signal))
        else:
            pairs.append((i, 0.0, False))

    if not pairs:
        return

    if sparse_mode == "exclude_missing":
        real = [v for _, v, has in pairs if has]
    else:
        real = [v for _, v, _ in pairs]

    if real:
        mu = sum(real) / len(real)
        var = sum((x - mu) ** 2 for x in real) / len(real)
        std = var**0.5
    else:
        mu, std = 0.0, 0.0

    for idx, val, has in pairs:
        if sparse_mode == "exclude_missing" and not has:
            rows[idx][out_col] = 0.0
        elif std > 0:
            rows[idx][out_col] = round((val - mu) / std, 4)
        else:
            rows[idx][out_col] = 0.0


def _winsorize(values: List[float], lower_pct: float = 5.0, upper_pct: float = 95.0) -> Tuple[float, float]:
    """Return (p_lower, p_upper) cutoff values for winsorization."""
    if not values:
        return 0.0, 1.0
    s = sorted(values)
    n = len(s)
    lo_idx = max(0, int(n * lower_pct / 100.0))
    hi_idx = min(n - 1, int(n * upper_pct / 100.0))
    return s[lo_idx], s[hi_idx]


def _compute_clinical_alpha_z(rows: List[Dict[str, Any]]) -> None:
    """Cross-sectional z-score of clinical_alpha_raw, winsorized p5/p95."""
    raws = []
    for r in rows:
        v = _safe_float(r.get("clinical_alpha_raw"))
        if v is not None:
            raws.append(v)

    if len(raws) < 2:
        for r in rows:
            r["clinical_alpha_z"] = ""
        return

    lo, hi = _winsorize(raws, 5.0, 95.0)
    clipped = [max(lo, min(hi, v)) for v in raws]
    mu = sum(clipped) / len(clipped)
    var = sum((x - mu) ** 2 for x in clipped) / len(clipped)
    std = var**0.5

    for r in rows:
        v = _safe_float(r.get("clinical_alpha_raw"))
        if v is not None and std > 0:
            clamped = max(lo, min(hi, v))
            r["clinical_alpha_z"] = round((clamped - mu) / std, 6)
        else:
            r["clinical_alpha_z"] = ""


def _compute_clinical_optionality(rows: List[Dict[str, Any]]) -> None:
    """Inverted percentile of clinical_score within drug_developer cohort (DESC)."""
    dev_indices = []
    for i, r in enumerate(rows):
        if r.get("archetype") == "drug_developer":
            dev_indices.append(i)

    if not dev_indices:
        for r in rows:
            r["clinical_optionality_pct_dev"] = ""
            r["has_clinical_optionality_dev"] = ""
            r["clinical_rank_pct_dev"] = ""
        return

    # Sort dev indices by clinical_score DESC, ticker ASC for ties
    dev_with_score = []
    for i in dev_indices:
        v = _safe_float(rows[i].get("clinical_score"))
        dev_with_score.append((i, v if v is not None else -999.0, rows[i].get("ticker", "")))

    dev_with_score.sort(key=lambda x: (-x[1], x[2]))

    n = len(dev_with_score)
    for rank_idx, (i, score, _) in enumerate(dev_with_score, start=1):
        pct = round((rank_idx - 0.5) / n, 6) if n > 0 else 0.5
        rows[i]["clinical_optionality_pct_dev"] = pct
        rows[i]["has_clinical_optionality_dev"] = "1"
        rows[i]["clinical_rank_pct_dev"] = round(1.0 - pct, 6)

    # Non-dev tickers get empty
    dev_set = set(dev_indices)
    for i, r in enumerate(rows):
        if i not in dev_set:
            r["clinical_optionality_pct_dev"] = ""
            r["has_clinical_optionality_dev"] = ""
            r["clinical_rank_pct_dev"] = ""


def _cohort_for_z(archetype: str) -> str:
    """Map archetype to z-score cohort."""
    if archetype.startswith("drug_developer"):
        return "drug_developer"
    if archetype.startswith("commercial_"):
        return archetype
    return "other"


def _tier_group_for_z(archetype: str, tier_dev: str, tier_commercial: str) -> str:
    """Map archetype + tier to tier-local z-score group."""
    if archetype.startswith("drug_developer"):
        return f"{archetype}_{tier_dev}"
    if archetype.startswith("commercial_"):
        tier = tier_commercial or tier_dev
        return f"{archetype}_{tier}"
    return f"{archetype}_"


def _resolve_alpha_table_for_bundle(
    as_of_date: str,
    ruleset: Any,
) -> Tuple[Path, str]:
    """Resolve alpha cohort table path for a PIT bundle date.

    Delegates to the shared resolver in ``common.alpha_table``, with
    ``allow_rebuild=False`` (batch bundle runs should not build OOS
    tables in-process — use prebuilt per-date tables or nearest earlier).
    """
    from common.alpha_table import resolve_alpha_table_path

    _logger = logging.getLogger("run_screen_from_bundle.alpha")
    return resolve_alpha_table_path(
        ruleset,
        as_of_date,
        _PROJECT_ROOT,
        _logger,
        allow_rebuild=False,
    )


def run_screen_for_date(
    as_of_date: str,
    clinical: dict,
    catalyst: dict,
    coinvest: dict,
    universe: List[dict],
    financial_records_by_ticker: Dict[str, dict],
    prices_df: Optional[Any],
    ruleset: Any,
    manifest: dict,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run the full screen pipeline for a single date.

    Returns (csv_rows, metadata).
    """
    # Lazy imports for decision engine
    sys.path.insert(0, str(_PROJECT_ROOT))
    # Prepare XBI series

    from decision_engine import (
        SORT_CONTRIB_KEYS,
        compute_actionable_sort_key,
        compute_decision_fields,
        compute_sort_contribs,
    )
    from module_5_alpha_cohort import attach_alpha_scores, load_alpha_cohort_table

    xbi_series = None
    if prices_df is not None and "XBI" in prices_df.columns:
        xbi_series = prices_df["XBI"]

    # Build rec dicts
    rows: List[Dict[str, Any]] = []
    skip_reasons_by_ticker: Dict[str, str] = {}

    for entry in universe:
        ticker = entry.get("ticker", "")
        if not ticker:
            continue

        # Price features
        if prices_df is not None:
            price_feats = _compute_price_features(prices_df, ticker, as_of_date, xbi_series)
        else:
            price_feats = {
                "drawdown": None,
                "drawdown_xbi": None,
                "drawdown_rel_xbi": None,
                "beta_xbi_60d": None,
                "alpha_60d": None,
                "rsi_14d": None,
                "vol_60d": None,
                "beta_xbi_60d_source": "",
                "beta_xbi_60d_missing_reason": "no_price_data",
                "alpha_60d_source": "",
                "alpha_60d_missing_reason": "no_price_data",
                "drawdown_missing_reason": "no_price_data",
            }

        fin_rec = financial_records_by_ticker.get(ticker, {})

        rec = _build_rec(
            ticker=ticker,
            clinical=clinical,
            catalyst=catalyst,
            coinvest_data=coinvest,
            fin_rec=fin_rec,
            univ_entry=entry,
            price_feats=price_feats,
        )

        # Archetype
        md = entry.get("market_data", {})
        industry = md.get("industry", entry.get("sector", "Biotechnology"))
        fd = entry.get("financial_data", {})
        revenue = fd.get("revenue_ttm") or fin_rec.get("Revenue")
        has_revenue = revenue is not None and float(revenue or 0) > 0
        # Revenue scale bucket
        try:
            rev_val = float(revenue or 0)
            if rev_val <= 0:
                rev_bucket = "pre_revenue"
            elif rev_val < 500_000_000:
                rev_bucket = "small"
            elif rev_val < 2_000_000_000:
                rev_bucket = "medium"
            else:
                rev_bucket = "large"
        except (ValueError, TypeError):
            rev_bucket = "pre_revenue"

        archetype = classify_company_archetype(ticker, industry, has_revenue, rev_bucket)

        # Clinical data
        clin = clinical.get(ticker, {})

        # Stage bucket
        stage_bucket = _classify_stage_bucket(clin.get("lead_phase", ""))

        # Market cap bucket
        mc = md.get("market_cap")
        if mc is not None:
            try:
                mc_val = float(mc)
                if mc_val >= 10_000_000_000:
                    mc_bucket = "large"
                elif mc_val >= 2_000_000_000:
                    mc_bucket = "mid"
                elif mc_val >= 300_000_000:
                    mc_bucket = "small"
                else:
                    mc_bucket = "micro"
            except (ValueError, TypeError):
                mc_bucket = ""
        else:
            mc_bucket = ""

        row = {
            "ticker": ticker,
            "company_name": md.get("company_name", entry.get("name", ticker)),
            "archetype": archetype,
            "stage_bucket": stage_bucket,
            "market_cap_bucket": mc_bucket,
            "severity": rec["severity"],
            "confidence_overall": rec["confidence_overall"],
            "clinical_score": rec["clinical_score"],
            "clinical_alpha_raw": clin.get("clinical_alpha_raw"),
            "clinical_readout_days": clin.get("readout_days", ""),
            "clinical_coverage_flag": "1" if clin else "",
            # Module scores (placeholders)
            "momentum_score": "",
            "catalyst_score": "",
            "smart_money_score": "",
            "valuation_score": "",
            "financial_score": "",
            # Composite placeholders
            "composite_rank": "",
            "composite_score": "",
            "score_rank_pct": "",
            "score_z": "",
            "composite_score_attn": "",
            "score_rank_pct_attn": "",
            "score_z_attn": "",
            # Catalyst source/type
            "catalyst_source": rec.get("catalyst_source", ""),
            "catalyst_event_type": rec.get("catalyst_event_type", ""),
            "catalyst_reason_detail": "",
            # Price tracking columns
            "de_alpha_60d": price_feats.get("alpha_60d", ""),
            "de_alpha_60d_source": price_feats.get("alpha_60d_source", ""),
            "de_alpha_60d_missing_reason": price_feats.get("alpha_60d_missing_reason", ""),
            "de_tier1_count": coinvest.get(ticker, {}).get("tier1_count", ""),
            "de_beta_xbi_60d": price_feats.get("beta_xbi_60d", ""),
            "de_beta_xbi_60d_source": price_feats.get("beta_xbi_60d_source", ""),
            "de_beta_xbi_60d_missing_reason": price_feats.get("beta_xbi_60d_missing_reason", ""),
            "de_drawdown": price_feats.get("drawdown", ""),
            "de_drawdown_missing_reason": price_feats.get("drawdown_missing_reason", ""),
            "de_rsi_14d": price_feats.get("rsi_14d", ""),
            "de_drawdown_xbi": price_feats.get("drawdown_xbi", ""),
            "de_drawdown_rel_xbi": price_feats.get("drawdown_rel_xbi", ""),
            "de_catalyst_days": catalyst.get(ticker, {}).get("days_to_catalyst", ""),
            "de_catalyst_in_window": "",
            "de_catalyst_mode": catalyst.get(ticker, {}).get("catalyst_mode", ""),
            "returns_source": "pit_bundle",
            # Cost placeholder
            "est_cost_bps": "",
            "top_3_drivers": "",
            "cat_priority": "",
            # Signal presence flags
            "has_coinvest_signal": str(
                bool(coinvest.get(ticker, {}).get("tier1_count") is not None and ticker in coinvest)
            ),
            "has_inst_delta": "False",  # bundles don't carry institutional delta
            "has_catalyst_signal": str(catalyst.get(ticker, {}).get("catalyst_mode", "missing") != "missing"),
            # Institutional delta (not available from bundles)
            "inst_delta_z": "0",
            "inst_delta_net": "0",
            "inst_delta_new": "0",
            "inst_delta_exit": "0",
            # Coinvest tag
            "coinvest_tag": "",
            "coinvest_recency_state": coinvest.get(ticker, {}).get("coinvest_recency_state", ""),
            # Commercial quality
            "commercial_quality": "",
            "commercial_quality_pct": "",
            "has_commercial_quality": "",
            "target_weight_pct": "",
            # Internal reference to rec + archetype
            "_rec": rec,
            "_archetype": archetype,
        }
        rows.append(row)

    if not rows:
        return [], {"as_of_date": as_of_date, "source_type": "pit_bundle", "ticker_count": 0}

    # --- Clinical alpha z-score (cross-sectional, winsorized) ---
    _compute_clinical_alpha_z(rows)

    # --- Clinical optionality percentile (dev cohort) ---
    _compute_clinical_optionality(rows)

    # --- Decision engine loop ---
    for r in rows:
        rec = r["_rec"]
        archetype = r["_archetype"]
        opt = _safe_float(r.get("clinical_optionality_pct_dev"))
        cqp = _safe_float(r.get("commercial_quality_pct"))

        de_fields = compute_decision_fields(
            rec=rec,
            archetype=archetype,
            optionality_pct_dev=opt,
            ruleset=ruleset,
            est_cost_bps=_safe_float(r.get("est_cost_bps")),
            commercial_quality_pct=cqp,
        )

        # Merge DE fields into row
        for k, v in de_fields.items():
            r[k] = v

    # --- Clinical score z by cohort (per-archetype, ddof=0) ---
    for r in rows:
        r["_cohort"] = _cohort_for_z(r.get("archetype", ""))
    _z_score_by_group(rows, "clinical_score", "_cohort", "clinical_score_z")
    for r in rows:
        r.pop("_cohort", None)

    # --- Clinical score z tier (per-tier within archetype, AFTER DE loop) ---
    for r in rows:
        r["_tier_group"] = _tier_group_for_z(
            r.get("archetype", ""),
            r.get("tier_dev", ""),
            r.get("tier_commercial", ""),
        )
    _z_score_by_group(rows, "clinical_score", "_tier_group", "clinical_score_z_tier")
    for r in rows:
        r.pop("_tier_group", None)

    # --- Coinvest score z (cross-sectional z of sponsor_tier1_count, ddof=0) ---
    _sparse_mode = getattr(ruleset, "sparse_signal_mode", "legacy")
    _coinvest_z_scored = _z_score_sparse(
        rows,
        "sponsor_tier1_count",
        "coinvest_score_z",
        flag_col="has_coinvest_signal",
        sparse_mode=_sparse_mode,
    )

    # --- Alpha cohort scoring ---
    # Resolve per-date OOS table when available (PIT-safe); fall back to static.
    _alpha_table_source = "none"
    _alpha_table_path_resolved = Path()
    try:
        _alpha_table_path_resolved, _alpha_table_source = _resolve_alpha_table_for_bundle(
            as_of_date,
            ruleset,
        )
        table = load_alpha_cohort_table(_alpha_table_path_resolved)
        attach_alpha_scores(
            rows,
            table,
            shrink_k=ruleset.alpha_cohort_shrink_k,
            clip_min=ruleset.alpha_cohort_clip_min,
            clip_max=ruleset.alpha_cohort_clip_max,
        )
    except Exception as e:
        print(f"  WARN: alpha cohort scoring failed: {e}", file=sys.stderr)
        for r in rows:
            r.setdefault("alpha_cohort_key", "")
            r.setdefault("alpha_cohort_raw", "")
            r.setdefault("alpha_cohort_pct", "")

    # Warn loudly if static fallback was used for a date that should have per-date tables
    from common.alpha_table import EARLIEST_DAILY_TABLE_DATE

    if "static" in _alpha_table_source and as_of_date >= EARLIEST_DAILY_TABLE_DATE:
        print(
            f"  WARNING: alpha table static fallback for {as_of_date} " f"(source={_alpha_table_source}) — PIT-unsafe!",
            file=sys.stderr,
        )

    # --- Composite rank override for alpha_cohort engine ---
    if ruleset.composite_engine == "alpha_cohort":
        # Sort by alpha_cohort_raw DESC, ticker ASC → assign composite_rank
        rows_sorted = sorted(
            rows, key=lambda r: (-(_safe_float(r.get("alpha_cohort_raw")) or 0.0), r.get("ticker", ""))
        )
        n = len(rows_sorted)
        for i, r in enumerate(rows_sorted, start=1):
            r["composite_rank"] = i
            r["composite_score"] = r.get("alpha_cohort_raw", 0.0)
            r["score_rank_pct"] = round((i - 0.5) / n, 6) if n > 0 else 0.5

    # --- Sort: compute actionable sort key ---
    rows.sort(
        key=lambda r: compute_actionable_sort_key(
            decision_fields=r,
            archetype=r.get("archetype", ""),
            optionality=_safe_float(r.get("clinical_optionality_pct_dev")),
            composite_rank=r.get("composite_rank"),
            ticker=r.get("ticker", ""),
            catalyst_event_type=r.get("catalyst_event_type", ""),
            catalyst_source=r.get("catalyst_source", ""),
            ruleset=ruleset,
            tiebreaker_pct=(
                _safe_float(r.get("alpha_cohort_pct"))
                if ruleset.sort_anchor == "alpha_cohort"
                else (
                    _safe_float(r.get("commercial_quality_pct"))
                    if r.get("archetype", "").startswith("commercial_")
                    else _safe_float(r.get("clinical_optionality_pct_dev"))
                )
            ),
        )
    )

    # Populate sort contribution diagnostics (same inputs as sort key above)
    for r in rows:
        total_adj, cmap = compute_sort_contribs(
            decision_fields=r,
            archetype=r.get("archetype", ""),
            ruleset=ruleset,
            tiebreaker_pct=(
                _safe_float(r.get("alpha_cohort_pct"))
                if ruleset.sort_anchor == "alpha_cohort"
                else (
                    _safe_float(r.get("commercial_quality_pct"))
                    if r.get("archetype", "").startswith("commercial_")
                    else _safe_float(r.get("clinical_optionality_pct_dev"))
                )
            ),
            alpha_raw=_safe_float(r.get("alpha_cohort_raw")),
            catalyst_event_type=r.get("catalyst_event_type", ""),
            catalyst_source=r.get("catalyst_source", ""),
        )
        r["de_sort_total_adj"] = round(total_adj, 6)
        for k in SORT_CONTRIB_KEYS:
            r[f"de_sort_contrib_{k}"] = round(cmap[k], 6)

    # Assign actionable_rank
    rank = 1
    for r in rows:
        if r.get("eligible") == "1":
            r["actionable_rank"] = rank
            rank += 1
        else:
            r["actionable_rank"] = ""

    # Clean internal keys
    for r in rows:
        r.pop("_rec", None)
        r.pop("_archetype", None)

    # Build metadata
    components = manifest.get("components", {})
    pq = _get_pit_quality(manifest)
    metadata = {
        "as_of_date": as_of_date,
        "source_type": "pit_bundle",
        "ruleset_id": ruleset.ruleset_id,
        "decision_engine_version": rows[0].get("decision_engine_version", "") if rows else "",
        "ticker_count": len(rows),
        "eligible_count": sum(1 for r in rows if r.get("eligible") == "1"),
        "component_coverage": {name: info.get("coverage_pct", 0.0) for name, info in components.items()},
        "pit_quality": pq,
        "bundle_manifest_schema": manifest.get("schema_version", ""),
    }

    # Alpha table audit metadata
    from common.alpha_table import alpha_table_metadata

    metadata.update(
        alpha_table_metadata(
            as_of_date,
            _alpha_table_path_resolved,
            _alpha_table_source,
        )
    )

    return rows, metadata


# ---------------------------------------------------------------------------
# Phase D — Output + batch
# ---------------------------------------------------------------------------

# Columns to write — matches SNAPSHOT_COLUMNS from run_screen.py
SNAPSHOT_COLUMNS = [
    "ticker",
    "company_name",
    "actionable_rank",
    "target_weight_pct",
    "tier_any",
    "tier_any_reason",
    "tier_dev",
    "tier_reason",
    "tier_commercial",
    "alpha_cohort_pct",
    "commercial_quality_pct",
    "has_commercial_quality",
    "clinical_optionality_pct_dev",
    "has_clinical_optionality_dev",
    "clinical_rank_pct_dev",
    "catalyst_days",
    "catalyst_in_window",
    "catalyst_mode",
    "cat_priority",
    "mom_state",
    "risk_flags",
    "size_band",
    "size_reasons",
    "top_3_drivers",
    "catalyst_reason_detail",
    "decision_engine_version",
    "decision_engine_ruleset_id",
    "eligible",
    "ineligible_reasons",
    "alpha_cohort_key",
    "alpha_cohort_raw",
    "commercial_quality",
    "clinical_alpha_z",
    "clinical_readout_days",
    "clinical_coverage_flag",
    "clinical_score_z",
    "clinical_score_z_tier",
    "sponsor_tier1_count",
    "sponsor_overlap_count",
    "sponsor_net_buying",
    "coinvest_score_z",
    "coinvest_tag",
    "coinvest_conviction",
    "coinvest_tier1_conviction",
    "coinvest_max_position_pct",
    "coinvest_filing_age_days",
    "coinvest_recency_state",
    "inst_delta_z",
    "inst_delta_net",
    "inst_delta_new",
    "inst_delta_exit",
    "has_coinvest_signal",
    "has_inst_delta",
    "has_catalyst_signal",
    "catalyst_strength",
    "catalyst_decay_w",
    "runway_bucket",
    "cost_bucket",
    "est_cost_bps",
    "cost_mult",
    "cost_haircut_applied",
    "dd_rel_margin_rescued",
    "catalyst_tilt_mult",
    "catalyst_tilt_applied",
    "mom_state_tilt_mult",
    "mom_state_tilt_applied",
    "de_catalyst_days",
    "de_catalyst_in_window",
    "de_catalyst_mode",
    "de_alpha_60d",
    "de_alpha_60d_source",
    "de_alpha_60d_missing_reason",
    "de_tier1_count",
    "de_beta_xbi_60d",
    "de_beta_xbi_60d_source",
    "de_beta_xbi_60d_missing_reason",
    "de_drawdown",
    "de_drawdown_missing_reason",
    "de_rsi_14d",
    "de_drawdown_xbi",
    "de_drawdown_rel_xbi",
    "stage_bucket",
    "market_cap_bucket",
    "severity",
    "archetype",
    "returns_source",
    "catalyst_source",
    "catalyst_event_type",
    "missing_components",
    "missingness_penalty",
    "confidence_overall",
    "momentum_score",
    "catalyst_score",
    "smart_money_score",
    "valuation_score",
    "clinical_score",
    "financial_score",
    "composite_rank",
    "composite_score",
    "score_rank_pct",
    "score_z",
    "composite_score_attn",
    "score_rank_pct_attn",
    "score_z_attn",
    # --- Sort contribution diagnostics ---
    "de_sort_total_adj",
    "de_sort_contrib_clinical",
    "de_sort_contrib_coinvest",
    "de_sort_contrib_institutional",
    "de_sort_contrib_calendar_alpha",
    "de_sort_contrib_alpha_cohort_tb",
    "de_sort_contrib_catalyst_bonus",
]


def _write_snapshot(
    out_dir: Path,
    as_of_date: str,
    rows: List[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> Path:
    """Write rankings.csv + metadata.json to out_dir/YYYY-MM-DD/."""
    snap_dir = out_dir / as_of_date
    snap_dir.mkdir(parents=True, exist_ok=True)

    # Rankings CSV
    csv_path = snap_dir / "rankings.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SNAPSHOT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            # Ensure all values are strings for CSV
            clean = {}
            for col in SNAPSHOT_COLUMNS:
                v = r.get(col, "")
                if v is None:
                    clean[col] = ""
                else:
                    clean[col] = str(v) if not isinstance(v, str) else v
            writer.writerow(clean)

    # Metadata JSON
    meta_path = snap_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    return snap_dir


def run_batch(
    bundle_root: Path,
    out_root: Path,
    ruleset: Any,
    universe: List[dict],
    financial_records_by_ticker: Dict[str, dict],
    prices_df: Optional[Any],
    pit_mode: str = "strict",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Dict[str, Any]:
    """Run screen for all bundle dates. Returns summary dict."""
    dates = discover_bundle_dates(bundle_root, date_from, date_to)
    if not dates:
        print("No bundle dates found.", file=sys.stderr)
        return {"dates_processed": 0, "dates_skipped": 0, "skip_reasons": {}}

    processed = 0
    skipped = 0
    skip_reasons: Dict[str, str] = {}

    for dt in dates:
        bundle_dir = bundle_root / dt
        bundle = load_bundle(bundle_dir, pit_mode=pit_mode)
        if bundle is None:
            skip_reasons[dt] = "no_manifest"
            skipped += 1
            continue
        if bundle["skip_reason"]:
            skip_reasons[dt] = bundle["skip_reason"]
            skipped += 1
            print(f"  SKIP {dt}: {bundle['skip_reason']}", file=sys.stderr)
            continue

        print(f"  Processing {dt}...", file=sys.stderr)
        t0 = time.monotonic()

        rows, metadata = run_screen_for_date(
            as_of_date=dt,
            clinical=bundle["clinical"],
            catalyst=bundle["catalyst"],
            coinvest=bundle["coinvest"],
            universe=universe,
            financial_records_by_ticker=financial_records_by_ticker,
            prices_df=prices_df,
            ruleset=ruleset,
            manifest=bundle["manifest"],
        )

        if rows:
            snap_dir = _write_snapshot(out_root, dt, rows, metadata)
            elapsed = time.monotonic() - t0
            n_elig = sum(1 for r in rows if r.get("eligible") == "1")
            print(f"    {len(rows)} tickers, {n_elig} eligible, {elapsed:.1f}s → {snap_dir}", file=sys.stderr)
            processed += 1
        else:
            skip_reasons[dt] = "no_tickers"
            skipped += 1

    summary = {
        "dates_processed": processed,
        "dates_skipped": skipped,
        "skip_reasons": skip_reasons,
        "total_dates": len(dates),
    }
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run decision-engine screen from PIT bundles",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help="Single date bundle directory",
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=None,
        help="Batch mode: root containing date subdirs",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=_PROJECT_ROOT / "data" / "snapshots_pit",
        help="Output root for snapshots",
    )
    parser.add_argument(
        "--ruleset",
        type=Path,
        default=None,
        help="Path to decision_ruleset.json",
    )
    parser.add_argument(
        "--universe",
        type=Path,
        default=_PROJECT_ROOT / "production_data" / "universe.json",
    )
    parser.add_argument(
        "--financial-records",
        type=Path,
        default=_PROJECT_ROOT / "production_data" / "financial_records.json",
    )
    parser.add_argument(
        "--price-csv",
        type=Path,
        default=_PROJECT_ROOT / "production_data" / "price_history.csv",
    )
    parser.add_argument(
        "--pit-mode",
        choices=["strict", "lenient"],
        default="strict",
    )
    parser.add_argument("--date-from", type=str, default=None)
    parser.add_argument("--date-to", type=str, default=None)
    parser.add_argument(
        "--alpha-table-policy",
        choices=["never", "if_missing", "daily"],
        default="if_missing",
        help="Alpha table rebuild policy (default: if_missing for PIT safety)",
    )

    args = parser.parse_args()

    if not args.bundle_dir and not args.bundle_root:
        parser.error("Either --bundle-dir or --bundle-root is required")

    # Add project root to sys.path
    sys.path.insert(0, str(_PROJECT_ROOT))

    from common.types import is_retired_status
    from decision_engine import DecisionRuleset

    # Load ruleset
    if args.ruleset:
        ruleset = DecisionRuleset.from_json(str(args.ruleset))
    else:
        # Try default path
        default_rs = _PROJECT_ROOT / "production_data" / "decision_ruleset.json"
        if default_rs.exists():
            ruleset = DecisionRuleset.from_json(str(default_rs))
        else:
            ruleset = DecisionRuleset()
    # Override alpha table policy from CLI (default: if_missing for PIT safety)
    import dataclasses

    ruleset = dataclasses.replace(
        ruleset,
        alpha_table_rebuild_policy=args.alpha_table_policy,
    )
    print(f"Ruleset: {ruleset.ruleset_id}", file=sys.stderr)
    print(f"Alpha table policy: {ruleset.alpha_table_rebuild_policy}", file=sys.stderr)

    # Load shared data
    print("Loading universe...", file=sys.stderr)
    universe = _load_json(args.universe)
    # Exclude delisted tickers — keep entries in universe.json for history,
    # but never pass them into the screening pipeline.
    _delisted = [e.get("ticker") for e in universe if isinstance(e, dict) and is_retired_status(e.get("status"))]
    if _delisted:
        universe = [e for e in universe if not (isinstance(e, dict) and is_retired_status(e.get("status")))]
        print(
            f"  Delisted filter: excluded {len(_delisted)} tickers (status=delisted): {sorted(_delisted)}",
            file=sys.stderr,
        )

    print("Loading financial records...", file=sys.stderr)
    fin_records = _load_json(args.financial_records)
    financial_records_by_ticker = {r["ticker"]: r for r in fin_records if "ticker" in r}

    print("Loading price history...", file=sys.stderr)
    prices_df = None
    if args.price_csv.exists():
        prices_df = _load_price_history(args.price_csv)
        if prices_df is not None:
            print(f"  {len(prices_df.columns)} tickers, {len(prices_df)} bars", file=sys.stderr)

    # Single date mode
    if args.bundle_dir:
        bundle = load_bundle(args.bundle_dir, pit_mode=args.pit_mode)
        if bundle is None:
            print("ERROR: No manifest.json in bundle dir", file=sys.stderr)
            sys.exit(1)
        if bundle["skip_reason"]:
            print(f"SKIP: {bundle['skip_reason']}", file=sys.stderr)
            sys.exit(2)

        rows, metadata = run_screen_for_date(
            as_of_date=bundle["as_of_date"],
            clinical=bundle["clinical"],
            catalyst=bundle["catalyst"],
            coinvest=bundle["coinvest"],
            universe=universe,
            financial_records_by_ticker=financial_records_by_ticker,
            prices_df=prices_df,
            ruleset=ruleset,
            manifest=bundle["manifest"],
        )
        if rows:
            snap_dir = _write_snapshot(args.out_root, bundle["as_of_date"], rows, metadata)
            n_elig = sum(1 for r in rows if r.get("eligible") == "1")
            print(f"Done: {len(rows)} tickers, {n_elig} eligible → {snap_dir}")
        else:
            print("No tickers produced.")
        return

    # Batch mode
    summary = run_batch(
        bundle_root=args.bundle_root,
        out_root=args.out_root,
        ruleset=ruleset,
        universe=universe,
        financial_records_by_ticker=financial_records_by_ticker,
        prices_df=prices_df,
        pit_mode=args.pit_mode,
        date_from=args.date_from,
        date_to=args.date_to,
    )

    print(
        f"\nSummary: {summary['dates_processed']} processed, "
        f"{summary['dates_skipped']} skipped out of {summary['total_dates']} dates"
    )
    if summary["skip_reasons"]:
        for dt, reason in sorted(summary["skip_reasons"].items()):
            print(f"  SKIP {dt}: {reason}")

    if summary["dates_processed"] == 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
