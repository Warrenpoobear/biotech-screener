#!/usr/bin/env python3
"""
Rank-IC Backtest from Historical Snapshots

Measures whether higher-ranked securities in the screener actually deliver
higher forward returns, using existing snapshot rankings and Morningstar
HS793 total return index data (with price_history.csv fallback).

Key methodology:
- Investable-universe filtering: only tickers with valid forward returns
  are included, then re-ranked within that investable set. This eliminates
  rank-correlated missingness from IPO-after / no-coverage tickers.
- Chained returns: HS793 total return index primary, price_history.csv
  (close-to-close price return) fallback for tickers missing HS793.

Usable snapshots: 7 quarterly (2023-Q1 to 2024-Q3) — 2026 snapshots lack
sufficient forward return data (HS793 ends 2026-02-05).

Usage:
    python run_rank_ic_backtest.py
"""
from __future__ import annotations

import csv
import io
import json
import math
import re
import sys
import tarfile
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.ic_measurement import (
    IC_EXCELLENT, IC_GOOD, IC_WEAK,
    MIN_OBS_IC,
    calculate_ic,
    calculate_kendall,
    compute_quintile_spread,
    _classify_ic,
    _quantize,
    add_trading_days,
    ICCalculationEngine,
)
from backtest.returns_provider import BaseReturnsProvider, CSVReturnsProvider
from archive_snapshot import verify_archive


# =============================================================================
# PATHS
# =============================================================================

SNAPSHOT_DIR = PROJECT_ROOT / "data" / "snapshots"
ARCHIVE_DIR = PROJECT_ROOT / "data" / "archives"
RETURNS_JSON = PROJECT_ROOT / "production_data" / "morningstar_returns_history.json"
PRICE_CSV = PROJECT_ROOT / "production_data" / "price_history.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_JSON = OUTPUT_DIR / "rank_ic_backtest.json"

# Quarterly snapshots with sufficient forward return data
QUARTERLY_DATES = [
    "2023-01-15",
    "2023-04-15",
    "2023-07-15",
    "2023-10-15",
    "2024-01-15",
    "2024-04-15",
    "2024-07-15",
]

HORIZONS = {"20d": 20, "60d": 60}

# Regime: trailing 60 calendar day XBI return
REGIME_LOOKBACK_DAYS = 60


# =============================================================================
# REGIME INDICATOR
# =============================================================================

def compute_xbi_regime(csv_provider: CSVReturnsProvider, snapshot_dates: List[str]) -> Dict[str, str]:
    """
    Classify each snapshot date as RISK_ON or RISK_OFF based on
    trailing 60-calendar-day XBI price return.

    Returns {snapshot_date: "RISK_ON" | "RISK_OFF"}
    """
    regimes: Dict[str, str] = {}
    for snap_date in snapshot_dates:
        end_dt = date.fromisoformat(snap_date)
        start_dt = end_dt - timedelta(days=REGIME_LOOKBACK_DAYS)
        ret = csv_provider.get_forward_total_return("XBI", start_dt.isoformat(), end_dt.isoformat())
        if ret is not None:
            regimes[snap_date] = "RISK_ON" if float(ret) >= 0 else "RISK_OFF"
        else:
            regimes[snap_date] = "UNKNOWN"
    return regimes


# =============================================================================
# XBI BETA ESTIMATION & RESIDUAL RETURNS
# =============================================================================

# Trailing window for beta estimation (trading days ~ 1 year)
BETA_LOOKBACK_TRADING_DAYS = 252


def estimate_xbi_betas(
    csv_provider: CSVReturnsProvider,
    tickers: List[str],
    as_of_date: str,
    lookback_days: int = BETA_LOOKBACK_TRADING_DAYS,
) -> Dict[str, float]:
    """
    Estimate OLS beta of each ticker vs XBI using trailing daily returns.

    Uses simple OLS: stock_ret = alpha + beta * xbi_ret + epsilon.
    Only tickers with >= 60 overlapping daily observations get a beta.

    Returns {ticker: beta}.
    """
    MIN_OBS_BETA = 60

    end_dt = date.fromisoformat(as_of_date)
    # Approximate calendar days for lookback (trading days * 365/252)
    start_dt = end_dt - timedelta(days=int(lookback_days * 365 / 252) + 30)

    # Get XBI daily returns
    xbi_daily = csv_provider.get_daily_returns("XBI", start_dt, end_dt)
    if len(xbi_daily) < MIN_OBS_BETA:
        return {}

    xbi_by_date: Dict[date, float] = {d: r for d, r in xbi_daily}

    betas: Dict[str, float] = {}
    for ticker in tickers:
        stock_daily = csv_provider.get_daily_returns(ticker, start_dt, end_dt)
        if not stock_daily:
            continue

        # Align dates
        stock_by_date = {d: r for d, r in stock_daily}
        common_dates = sorted(set(stock_by_date.keys()) & set(xbi_by_date.keys()))

        if len(common_dates) < MIN_OBS_BETA:
            continue

        x = [xbi_by_date[d] for d in common_dates]
        y = [stock_by_date[d] for d in common_dates]

        # OLS: beta = cov(x,y) / var(x)
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        cov_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / n
        var_x = sum((xi - mean_x) ** 2 for xi in x) / n

        if var_x < 1e-12:
            continue

        beta = cov_xy / var_x
        betas[ticker.upper()] = round(beta, 6)

    return betas


def compute_residual_returns(
    raw_returns: Dict[str, float],
    xbi_return: Optional[float],
    betas: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute XBI-residual returns: resid = raw - beta * xbi_return.

    Tickers without a beta estimate keep their raw return (beta=0 assumption).
    If XBI return is unavailable, returns empty dict.
    """
    if xbi_return is None:
        return {}

    residuals: Dict[str, float] = {}
    for ticker, raw_ret in raw_returns.items():
        beta = betas.get(ticker, 0.0)
        residuals[ticker] = raw_ret - beta * xbi_return

    return residuals


# =============================================================================
# STRATIFIED IC
# =============================================================================

def compute_stratified_ic(
    rankings: Dict[str, int],
    fwd_returns: Dict[str, float],
    groups: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    """
    Split tickers by group label, re-rank within each group,
    and compute IC per group.

    Returns {group_label: {"ic": float, "n": int, "quality": str}}
    """
    # Partition tickers by group
    group_tickers: Dict[str, List[str]] = {}
    for t in rankings:
        g = groups.get(t, "unknown")
        group_tickers.setdefault(g, []).append(t)

    results: Dict[str, Dict[str, Any]] = {}
    for g, tickers in sorted(group_tickers.items()):
        # Re-rank within group (preserve original ordering)
        sorted_t = sorted(tickers, key=lambda t: rankings[t])
        g_rankings = {t: i + 1 for i, t in enumerate(sorted_t)}
        g_returns = {t: fwd_returns[t] for t in g_rankings if t in fwd_returns}

        ic_val = calculate_ic(g_rankings, g_returns)
        n = len(set(g_rankings) & set(g_returns))
        results[g] = {
            "ic": float(ic_val) if ic_val is not None else None,
            "n": n,
            "quality": _classify_ic(abs(ic_val)).value if ic_val is not None else "INSUFFICIENT",
        }
    return results


# =============================================================================
# PARTIAL CORRELATION & BUCKET SPREAD DIAGNOSTICS
# =============================================================================

def compute_partial_correlation(
    component_scores: Dict[str, Dict[str, float]],
    extra_controls: Dict[str, Dict[str, float]],
    fwd_returns: Dict[str, float],
    target_component: str,
) -> Optional[Dict[str, Any]]:
    """
    Partial correlation: residualize BOTH returns and target component
    against all other components + controls, then correlate the two
    residual vectors (Pearson and Spearman).

    This matches the OLS notion of "partial effect" without the
    rank-the-residuals-of-a-bucketed-thing pathology.
    """
    control_names: List[str] = []
    control_dicts: List[Dict[str, float]] = []
    for name in sorted(component_scores.keys()):
        if name == target_component:
            continue
        if component_scores[name]:
            control_names.append(name)
            control_dicts.append(component_scores[name])
    for name in sorted(extra_controls.keys()):
        if extra_controls[name]:
            control_names.append(name)
            control_dicts.append(extra_controls[name])

    target_dict = component_scores.get(target_component, {})
    if not target_dict or not control_names:
        return None

    common = set(target_dict.keys()) & set(fwd_returns.keys())
    for cd in control_dicts:
        common &= set(cd.keys())
    common_sorted = sorted(common)

    if len(common_sorted) < len(control_names) + 10:
        return None

    n = len(common_sorted)
    y_returns = [fwd_returns[t] for t in common_sorted]
    x_target = [target_dict[t] for t in common_sorted]

    # Build control matrix (standardized)
    ctrl_std = {name: _standardize([cd[t] for t in common_sorted])
                for name, cd in zip(control_names, control_dicts)}

    col_names = ["intercept"] + control_names
    X = [[1.0] + [ctrl_std[name][i] for name in control_names] for i in range(n)]

    # Residualize returns on controls
    reg_y = _ols(y_returns, X, col_names)
    if reg_y is None:
        return None
    beta_y = [reg_y["coefficients"][cn]["coef"] for cn in col_names]
    resid_y = [y_returns[i] - sum(X[i][j] * beta_y[j] for j in range(len(col_names)))
               for i in range(n)]

    # Residualize target component on controls
    reg_x = _ols(x_target, X, col_names)
    if reg_x is None:
        return None
    beta_x = [reg_x["coefficients"][cn]["coef"] for cn in col_names]
    resid_x = [x_target[i] - sum(X[i][j] * beta_x[j] for j in range(len(col_names)))
               for i in range(n)]

    # Pearson correlation of residuals
    pearson_r = _pairwise_corr(resid_x, resid_y)

    # Spearman on residual ranks
    def _rank_values(vals: List[float]) -> List[float]:
        """Average-rank with tie handling."""
        indexed = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(indexed):
            j = i
            while j < len(indexed) and vals[indexed[j]] == vals[indexed[i]]:
                j += 1
            avg_rank = (i + j + 1) / 2.0  # 1-based average rank
            for k in range(i, j):
                ranks[indexed[k]] = avg_rank
            i = j
        return ranks

    spearman_r = _pairwise_corr(_rank_values(resid_x), _rank_values(resid_y))

    return {
        "partial_pearson": round(pearson_r, 6),
        "partial_spearman": round(spearman_r, 6),
        "n": n,
        "r2_target_on_controls": reg_x["r_squared"],
        "r2_returns_on_controls": reg_y["r_squared"],
    }


def compute_bucket_spread(
    component_scores: Dict[str, Dict[str, float]],
    fwd_returns: Dict[str, float],
    target_component: str,
    stage_map: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    """
    For a bucketed component (e.g. Financial_Score with 4 levels),
    compute mean forward return per bucket level.

    Reports: per-bucket mean return, monotonicity, and top-bottom spread.
    Also splits by stage.
    """
    target_dict = component_scores.get(target_component, {})
    if not target_dict:
        return None

    # Group tickers by bucket value
    buckets: Dict[float, List[str]] = {}
    for t, val in target_dict.items():
        if val is not None and t in fwd_returns:
            buckets.setdefault(val, []).append(t)

    if len(buckets) < 2:
        return None

    sorted_levels = sorted(buckets.keys())
    bucket_data: List[Dict[str, Any]] = []
    for level in sorted_levels:
        tickers = buckets[level]
        rets = [fwd_returns[t] for t in tickers]
        bucket_data.append({
            "level": level,
            "n": len(rets),
            "mean_return": round(mean(rets), 6) if rets else None,
        })

    # Monotonicity: count adjacent pairs where higher level has higher return
    mono_up = 0
    mono_pairs = 0
    for i in range(len(bucket_data) - 1):
        if bucket_data[i]["mean_return"] is not None and bucket_data[i + 1]["mean_return"] is not None:
            mono_pairs += 1
            if bucket_data[i + 1]["mean_return"] > bucket_data[i]["mean_return"]:
                mono_up += 1
    monotonicity = mono_up / mono_pairs if mono_pairs > 0 else None

    # Top - bottom spread
    top_ret = bucket_data[-1]["mean_return"]
    bot_ret = bucket_data[0]["mean_return"]
    spread = round((top_ret - bot_ret) * 100, 2) if top_ret is not None and bot_ret is not None else None

    result: Dict[str, Any] = {
        "n_levels": len(sorted_levels),
        "levels": sorted_levels,
        "bucket_data": bucket_data,
        "monotonicity_up": monotonicity,
        "spread_top_minus_bottom_pct": spread,
    }

    # Stage split
    for stage_label in ["dev", "commercial"]:
        stage_buckets: Dict[float, List[float]] = {}
        for level in sorted_levels:
            for t in buckets[level]:
                if stage_map.get(t) == stage_label:
                    stage_buckets.setdefault(level, []).append(fwd_returns[t])
        if len(stage_buckets) >= 2:
            stage_data = []
            for level in sorted_levels:
                rets = stage_buckets.get(level, [])
                stage_data.append({
                    "level": level,
                    "n": len(rets),
                    "mean_return": round(mean(rets), 6) if rets else None,
                })
            top_r = stage_data[-1]["mean_return"]
            bot_r = stage_data[0]["mean_return"]
            result[f"stage_{stage_label}"] = {
                "bucket_data": stage_data,
                "spread_pct": round((top_r - bot_r) * 100, 2) if top_r is not None and bot_r is not None else None,
            }

    return result


def compute_vif(
    component_scores: Dict[str, Dict[str, float]],
    extra_controls: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """Variance Inflation Factor for each feature (VIF > 5 is concerning)."""
    feat_names: List[str] = []
    feat_dicts: List[Dict[str, float]] = []
    for name in sorted(component_scores.keys()):
        if component_scores[name]:
            feat_names.append(name)
            feat_dicts.append(component_scores[name])
    for name in sorted(extra_controls.keys()):
        if extra_controls[name]:
            feat_names.append(name)
            feat_dicts.append(extra_controls[name])

    if len(feat_names) < 2:
        return {}

    common = set(feat_dicts[0].keys())
    for fd in feat_dicts[1:]:
        common &= set(fd.keys())
    common_sorted = sorted(common)
    n = len(common_sorted)

    if n < len(feat_names) + 2:
        return {}

    feat_std = {name: _standardize([fd[t] for t in common_sorted])
                for name, fd in zip(feat_names, feat_dicts)}

    vifs: Dict[str, float] = {}
    for target in feat_names:
        others = [nm for nm in feat_names if nm != target]
        col_names = ["intercept"] + others
        y = feat_std[target]
        X = [[1.0] + [feat_std[o][i] for o in others] for i in range(n)]
        result = _ols(y, X, col_names)
        if result is not None and result["r_squared"] < 1.0:
            vifs[target] = round(1.0 / (1.0 - result["r_squared"]), 2)
    return vifs


# =============================================================================
# CROSS-SECTIONAL OLS (stdlib-only)
# =============================================================================

def _ols(y: List[float], X: List[List[float]], col_names: List[str]) -> Optional[Dict[str, Any]]:
    """
    Ordinary least squares: y = X @ beta + epsilon.
    X should already include an intercept column if desired.

    Returns dict with coefficients, t-stats, R², or None if singular.
    """
    n = len(y)
    k = len(X[0]) if X else 0
    if n < k + 2 or k == 0:
        return None

    # X'X
    XtX = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            XtX[i][j] = sum(X[row][i] * X[row][j] for row in range(n))

    # X'y
    Xty = [sum(X[row][i] * y[row] for row in range(n)) for i in range(k)]

    # Solve via Gaussian elimination with partial pivoting
    aug = [XtX[i][:] + [Xty[i]] for i in range(k)]
    for col in range(k):
        # Pivot
        max_row = max(range(col, k), key=lambda r: abs(aug[r][col]))
        aug[col], aug[max_row] = aug[max_row], aug[col]
        if abs(aug[col][col]) < 1e-12:
            return None  # singular
        for row in range(col + 1, k):
            factor = aug[row][col] / aug[col][col]
            for j in range(col, k + 1):
                aug[row][j] -= factor * aug[col][j]
    # Back-substitute
    beta = [0.0] * k
    for i in range(k - 1, -1, -1):
        beta[i] = (aug[i][k] - sum(aug[i][j] * beta[j] for j in range(i + 1, k))) / aug[i][i]

    # Residuals + R²
    y_hat = [sum(X[row][j] * beta[j] for j in range(k)) for row in range(n)]
    residuals = [y[row] - y_hat[row] for row in range(n)]
    ss_res = sum(r * r for r in residuals)
    y_mean = sum(y) / n
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Standard errors: se(beta_j) = sqrt(sigma² * (X'X)^{-1}_{jj})
    sigma2 = ss_res / (n - k) if n > k else 0.0

    # Invert X'X for variance of coefficients
    # Augmented identity for inversion
    inv_aug = [XtX[i][:] + [1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
    for col in range(k):
        max_row = max(range(col, k), key=lambda r: abs(inv_aug[r][col]))
        inv_aug[col], inv_aug[max_row] = inv_aug[max_row], inv_aug[col]
        if abs(inv_aug[col][col]) < 1e-12:
            return None
        pivot = inv_aug[col][col]
        for j in range(2 * k):
            inv_aug[col][j] /= pivot
        for row in range(k):
            if row == col:
                continue
            factor = inv_aug[row][col]
            for j in range(2 * k):
                inv_aug[row][j] -= factor * inv_aug[col][j]

    XtX_inv_diag = [inv_aug[i][k + i] for i in range(k)]

    se = [math.sqrt(max(0, sigma2 * XtX_inv_diag[i])) for i in range(k)]
    t_stats = [beta[i] / se[i] if se[i] > 1e-12 else 0.0 for i in range(k)]

    coefficients = {}
    for i, name in enumerate(col_names):
        coefficients[name] = {
            "coef": round(beta[i], 6),
            "se": round(se[i], 6),
            "t_stat": round(t_stats[i], 3),
        }

    return {
        "coefficients": coefficients,
        "r_squared": round(r_squared, 4),
        "n": n,
        "k": k,
        "sigma": round(math.sqrt(sigma2), 6) if sigma2 > 0 else 0.0,
    }


def _standardize(vals: List[float]) -> List[float]:
    """Standardize to mean=0, std=1 (returns zeros if no variance)."""
    n = len(vals)
    if n < 2:
        return [0.0] * n
    m = sum(vals) / n
    s = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    if s < 1e-12:
        return [0.0] * n
    return [(v - m) / s for v in vals]


def _pairwise_corr(a: List[float], b: List[float]) -> float:
    """Pearson correlation between two lists."""
    n = len(a)
    if n < 3:
        return 0.0
    ma = sum(a) / n
    mb = sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = math.sqrt(sum((v - ma) ** 2 for v in a))
    db = math.sqrt(sum((v - mb) ** 2 for v in b))
    if da < 1e-12 or db < 1e-12:
        return 0.0
    return num / (da * db)


def run_cross_sectional_regression(
    fwd_returns: Dict[str, float],
    component_scores: Dict[str, Dict[str, float]],
    extra_controls: Dict[str, Dict[str, float]],
    stage_map: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    """
    Run cross-sectional OLS: fwd_return ~ financial + clinical + momentum + controls.
    All features standardized before regression for comparable coefficients.

    Args:
        fwd_returns: {ticker: forward_return}
        component_scores: {score_name: {ticker: value}}
        extra_controls: {control_name: {ticker: value}} e.g. log_cash
        stage_map: {ticker: "commercial" | "dev"}

    Returns regression result dict or None.
    """
    # Collect features available for this snapshot
    feature_names = []
    feature_dicts: List[Dict[str, float]] = []

    # Component scores
    for name in sorted(component_scores.keys()):
        if component_scores[name]:
            feature_names.append(name)
            feature_dicts.append(component_scores[name])

    # Extra controls
    for name in sorted(extra_controls.keys()):
        if extra_controls[name]:
            feature_names.append(name)
            feature_dicts.append(extra_controls[name])

    if not feature_names:
        return None

    # Find common tickers (have return + all features)
    common = set(fwd_returns.keys())
    for fd in feature_dicts:
        common &= set(fd.keys())
    common_sorted = sorted(common)

    if len(common_sorted) < len(feature_names) + 5:
        return None

    # Build arrays
    y_raw = [fwd_returns[t] for t in common_sorted]
    feature_raw: Dict[str, List[float]] = {}
    for name, fd in zip(feature_names, feature_dicts):
        feature_raw[name] = [fd[t] for t in common_sorted]

    # Standardize all features
    feature_std: Dict[str, List[float]] = {}
    for name, vals in feature_raw.items():
        feature_std[name] = _standardize(vals)

    # Correlation matrix between features
    corr_matrix: Dict[str, Dict[str, float]] = {}
    for na in feature_names:
        corr_matrix[na] = {}
        for nb in feature_names:
            corr_matrix[na][nb] = round(_pairwise_corr(feature_std[na], feature_std[nb]), 3)

    # Build X matrix: intercept + standardized features
    col_names = ["intercept"] + feature_names
    n = len(common_sorted)
    X = [[1.0] + [feature_std[name][i] for name in feature_names] for i in range(n)]

    result = _ols(y_raw, X, col_names)
    if result is None:
        return None

    result["feature_correlations"] = corr_matrix
    result["tickers"] = common_sorted

    # Stage-conditional: subset to dev and commercial, re-run
    for stage_label in ["dev", "commercial"]:
        stage_tickers = [t for t in common_sorted if stage_map.get(t) == stage_label]
        if len(stage_tickers) < len(feature_names) + 5:
            continue
        idx_map = {t: i for i, t in enumerate(common_sorted)}
        sub_y = [y_raw[idx_map[t]] for t in stage_tickers]
        sub_X = [X[idx_map[t]] for t in stage_tickers]
        sub_result = _ols(sub_y, sub_X, col_names)
        if sub_result is not None:
            result[f"stage_{stage_label}"] = sub_result

    return result


# =============================================================================
# POOLED INTERACTION REGRESSION
# =============================================================================

def run_pooled_interaction_regression(
    per_snapshot: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Pool all snapshot observations and run interaction regression:

    fwd_ret ~ fin + clin + log_cash + I(dev) + fin×dev + clin×dev
              + I(risk_on) + fin×risk_on + clin×risk_on + snapshot_FEs

    All continuous vars standardized within each snapshot before pooling.
    Returns results keyed by horizon label.
    """
    results: Dict[str, Dict[str, Any]] = {}

    for horizon_label in HORIZONS:
        # Collect pooled observations across snapshots
        rows: List[Dict[str, float]] = []
        snapshot_ids: List[int] = []

        for snap_idx, snap in enumerate(per_snapshot):
            fwd_rets = snap.get(f"_fwd_returns_{horizon_label}", {})
            comp_scores = snap.get("_component_scores", {})
            controls = snap.get("_extra_controls", {})
            stage_map = snap.get("_stage_map", {})
            regime = snap.get("regime", "UNKNOWN")

            fin_scores = comp_scores.get("Financial_Score", {})
            clin_scores = comp_scores.get("Clinical_Score", {})
            log_cash_scores = controls.get("log_cash", {})

            # Common tickers: must have return + financial + clinical + log_cash
            common = (set(fwd_rets.keys()) & set(fin_scores.keys())
                      & set(clin_scores.keys()) & set(log_cash_scores.keys()))

            if len(common) < 20:
                continue

            common_sorted = sorted(common)

            # Standardize within this snapshot
            fin_raw = [fin_scores[t] for t in common_sorted]
            clin_raw = [clin_scores[t] for t in common_sorted]
            cash_raw = [log_cash_scores[t] for t in common_sorted]
            ret_raw = [fwd_rets[t] for t in common_sorted]

            fin_z = _standardize(fin_raw)
            clin_z = _standardize(clin_raw)
            cash_z = _standardize(cash_raw)

            is_risk_on = 1.0 if regime == "RISK_ON" else 0.0

            for i, t in enumerate(common_sorted):
                is_dev = 1.0 if stage_map.get(t) == "dev" else 0.0
                rows.append({
                    "y": ret_raw[i],
                    "fin": fin_z[i],
                    "clin": clin_z[i],
                    "log_cash": cash_z[i],
                    "dev": is_dev,
                    "fin_x_dev": fin_z[i] * is_dev,
                    "clin_x_dev": clin_z[i] * is_dev,
                    # risk_on main effect is absorbed by snapshot FEs;
                    # interactions are identified (vary within snapshot)
                    "fin_x_riskon": fin_z[i] * is_risk_on,
                    "clin_x_riskon": clin_z[i] * is_risk_on,
                })
                snapshot_ids.append(snap_idx)

        if len(rows) < 50:
            continue

        # Build design matrix: intercept + features + snapshot FEs
        # Note: risk_on main effect dropped (collinear with snapshot FEs)
        feature_names = [
            "fin", "clin", "log_cash",
            "dev", "fin_x_dev", "clin_x_dev",
            "fin_x_riskon", "clin_x_riskon",
        ]

        # Snapshot fixed effects (drop first as reference)
        unique_snaps = sorted(set(snapshot_ids))
        snap_fe_names = [f"snap_{s}" for s in unique_snaps[1:]]

        col_names = ["intercept"] + feature_names + snap_fe_names
        n = len(rows)
        y = [r["y"] for r in rows]
        X = []
        for i, r in enumerate(rows):
            row_x = [1.0]
            for fn in feature_names:
                row_x.append(r[fn])
            # Snapshot FE dummies
            for s in unique_snaps[1:]:
                row_x.append(1.0 if snapshot_ids[i] == s else 0.0)
            X.append(row_x)

        reg = _ols(y, X, col_names)
        if reg is None:
            continue

        # Compute derived effects
        coeffs = reg["coefficients"]
        def _coef(name: str) -> float:
            return coeffs.get(name, {}).get("coef", 0.0)

        derived = {
            "fin_commercial_riskoff": _coef("fin"),
            "fin_commercial_riskon": _coef("fin") + _coef("fin_x_riskon"),
            "fin_dev_riskoff": _coef("fin") + _coef("fin_x_dev"),
            "fin_dev_riskon": _coef("fin") + _coef("fin_x_dev") + _coef("fin_x_riskon"),
            "clin_commercial_riskoff": _coef("clin"),
            "clin_commercial_riskon": _coef("clin") + _coef("clin_x_riskon"),
            "clin_dev_riskoff": _coef("clin") + _coef("clin_x_dev"),
            "clin_dev_riskon": _coef("clin") + _coef("clin_x_dev") + _coef("clin_x_riskon"),
        }

        reg["derived_effects"] = {k: round(v, 6) for k, v in derived.items()}
        reg["n_snapshots_pooled"] = len(unique_snaps)

        # Strip snapshot FE coefficients from output (nuisance)
        for sn in snap_fe_names:
            reg["coefficients"].pop(sn, None)

        results[horizon_label] = reg

    return results


# =============================================================================
# MORNINGSTAR RETURNS PROVIDER
# =============================================================================

class MorningstarReturnsProvider(BaseReturnsProvider):
    """Compute forward returns from HS793 total return index."""

    def __init__(self, json_path: Path, date_tolerance_days: int = 3):
        self.date_tolerance_days = date_tolerance_days
        self._index: Dict[str, Dict[date, Decimal]] = {}
        self._load(json_path)

    def _load(self, json_path: Path) -> None:
        with open(json_path, "r") as f:
            data = json.load(f)
        records = data.get("records", {})
        for ticker, ticker_data in records.items():
            hs793 = ticker_data.get("HS793")
            if not hs793:
                continue
            ts = hs793.get("time_series", [])
            prices: Dict[date, Decimal] = {}
            for pt in ts:
                try:
                    dt = date.fromisoformat(pt["date"])
                    val = Decimal(pt["value"])
                    prices[dt] = val
                except (ValueError, KeyError):
                    continue
            if prices:
                self._index[ticker.upper()] = prices

    def _find_nearest(self, ticker: str, target: date) -> Optional[Decimal]:
        prices = self._index.get(ticker)
        if prices is None:
            return None
        if target in prices:
            return prices[target]
        for delta in range(1, self.date_tolerance_days + 1):
            for d in [target + timedelta(days=delta), target - timedelta(days=delta)]:
                if d in prices:
                    return prices[d]
        return None

    def get_forward_total_return(self, ticker: str, start_date: str, end_date: str) -> Optional[str]:
        ticker = ticker.upper()
        start_val = self._find_nearest(ticker, date.fromisoformat(start_date))
        end_val = self._find_nearest(ticker, date.fromisoformat(end_date))
        if start_val is None or end_val is None or start_val == 0:
            return None
        ret = (end_val / start_val) - Decimal("1")
        return str(ret.quantize(Decimal("0.000001")))

    def get_available_tickers(self) -> List[str]:
        return sorted(self._index.keys())

    def get_last_date(self) -> Optional[date]:
        """Return the latest date with data across all tickers."""
        last = None
        for prices in self._index.values():
            if prices:
                ticker_max = max(prices.keys())
                if last is None or ticker_max > last:
                    last = ticker_max
        return last


# =============================================================================
# CHAINED RETURNS PROVIDER (HS793 primary + price CSV fallback)
# =============================================================================

class ChainedReturnsProvider(BaseReturnsProvider):
    """
    Try HS793 total return index first; fall back to price_history.csv
    close-to-close return for tickers not covered by HS793.

    Caveat: CSV fallback is price return (no dividends), but biotech
    names rarely pay dividends so this is acceptable vs. dropping them.
    """

    def __init__(self, primary: MorningstarReturnsProvider, fallback: CSVReturnsProvider):
        self.primary = primary
        self.fallback = fallback
        self._fallback_hits: Dict[str, int] = {}  # track which source was used

    def get_forward_total_return(self, ticker: str, start_date: str, end_date: str) -> Optional[str]:
        ret = self.primary.get_forward_total_return(ticker, start_date, end_date)
        if ret is not None:
            return ret
        # Fallback to CSV price return
        ret = self.fallback.get_forward_total_return(ticker, start_date, end_date)
        if ret is not None:
            self._fallback_hits[ticker.upper()] = self._fallback_hits.get(ticker.upper(), 0) + 1
        return ret

    def get_available_tickers(self) -> List[str]:
        return sorted(set(self.primary.get_available_tickers()) |
                       set(self.fallback.get_available_tickers()))

    def get_last_date(self) -> Optional[date]:
        """Return the latest date with data across both providers."""
        d1 = self.primary.get_last_date()
        d2 = self.fallback.get_last_date()
        if d1 is None:
            return d2
        if d2 is None:
            return d1
        return max(d1, d2)

    def get_fallback_stats(self) -> Dict[str, Any]:
        return {
            "n_tickers_used_fallback": len(self._fallback_hits),
            "total_fallback_lookups": sum(self._fallback_hits.values()),
            "tickers": sorted(self._fallback_hits.keys()),
        }


# =============================================================================
# SNAPSHOT LOADER
# =============================================================================

SnapshotData = Tuple[
    Dict[str, int],                  # rankings
    Dict[str, float],                # scores (composite_score)
    Dict[str, Dict[str, float]],     # component_scores
    Dict[str, str],                  # stage_map
    Dict[str, Dict[str, float]],     # extra_controls
    bool,                            # higher_is_better
    Dict[str, Dict[str, float]],     # signal_scores: {"score_rank_pct": {...}, "score_z": {...}}
]


def load_snapshot_rankings(snapshot_dir: Path) -> SnapshotData:
    """
    Load rankings.csv from a snapshot directory.

    Returns:
        (rankings, scores, component_scores, stage_map, extra_controls, higher_is_better)
        - rankings: {ticker: rank}
        - scores: {ticker: composite_score}
        - component_scores: {component_name: {ticker: score}}
        - stage_map: {ticker: "commercial" | "dev"}
        - extra_controls: {control_name: {ticker: value}} e.g. log_cash
        - higher_is_better: True if higher component scores mean better (new format)
    """
    csv_path = snapshot_dir / "rankings.csv"
    if not csv_path.exists():
        return {}, {}, {}, {}, {}, False, {}

    rankings: Dict[str, int] = {}
    scores: Dict[str, float] = {}
    component_scores: Dict[str, Dict[str, float]] = {}
    stage_map: Dict[str, str] = {}
    extra_controls: Dict[str, Dict[str, float]] = {"log_cash": {}}
    signal_scores: Dict[str, Dict[str, float]] = {"score_rank_pct": {}, "score_z": {}}

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

        # Detect format: new format uses snake_case "composite_rank"
        is_new_format = "composite_rank" in headers

        # Identify component columns
        if is_new_format:
            component_cols = [h for h in headers if h.endswith("_score") and h != "composite_score"]
        else:
            component_cols = [h for h in headers if h in ("Clinical_Score", "Financial_Score", "Momentum_Score")]

        # In old format lower composite score = better rank; new format higher = better
        higher_is_better = is_new_format

        # Stage column name differs by format
        stage_col = "stage_bucket" if is_new_format else "Stage_Bucket"

        for col in component_cols:
            component_scores[col] = {}

        for row in reader:
            if is_new_format:
                ticker = row.get("ticker", "").strip().upper()
                rank_str = row.get("composite_rank", "").strip()
                score_str = row.get("composite_score", "").strip()
            else:
                ticker = row.get("Ticker", "").strip().upper()
                rank_str = row.get("Rank", "").strip()
                score_str = row.get("Composite_Score", "").strip()

            if not ticker or not rank_str:
                continue

            try:
                rankings[ticker] = int(rank_str)
            except ValueError:
                continue

            if score_str:
                try:
                    scores[ticker] = float(score_str)
                except ValueError:
                    pass

            # Stage: "commercial" stays commercial, everything else is "dev"
            raw_stage = row.get(stage_col, "").strip().lower()
            stage_map[ticker] = "commercial" if raw_stage == "commercial" else "dev"

            for col in component_cols:
                val_str = row.get(col, "").strip()
                if val_str:
                    try:
                        component_scores[col][ticker] = float(val_str)
                    except ValueError:
                        pass

            # Extract signal scores (new format only)
            for sig_field in ("score_rank_pct", "score_z"):
                sig_str = row.get(sig_field, "").strip()
                if sig_str:
                    try:
                        signal_scores[sig_field][ticker] = float(sig_str)
                    except ValueError:
                        pass

            # Extract Cash for log_cash control (old format: "Cash" column)
            # Cash==0 is missing data for large-cap names (AZN, BNTX, etc.)
            cash_str = row.get("Cash", "").strip()
            if cash_str:
                try:
                    cash_val = float(cash_str)
                    if cash_val > 0:
                        extra_controls["log_cash"][ticker] = math.log(cash_val)
                except ValueError:
                    pass

    # Drop empty control dicts
    extra_controls = {k: v for k, v in extra_controls.items() if v}
    # Drop empty signal dicts
    signal_scores = {k: v for k, v in signal_scores.items() if v}

    return rankings, scores, component_scores, stage_map, extra_controls, higher_is_better, signal_scores


# =============================================================================
# ARCHIVE READER
# =============================================================================

def read_rankings_from_archive(tar_path: Path) -> SnapshotData:
    """
    Read rankings.csv from a .tar.gz archive without extracting to disk.

    Returns the same SnapshotData tuple as load_snapshot_rankings.
    """
    rankings: Dict[str, int] = {}
    scores: Dict[str, float] = {}
    component_scores: Dict[str, Dict[str, float]] = {}
    stage_map: Dict[str, str] = {}
    extra_controls: Dict[str, Dict[str, float]] = {"log_cash": {}}
    signal_scores: Dict[str, Dict[str, float]] = {"score_rank_pct": {}, "score_z": {}}

    with tarfile.open(tar_path, "r:gz") as tar:
        # Find rankings.csv member
        csv_member = None
        for member in tar.getmembers():
            if member.name.endswith("/rankings.csv") or member.name == "rankings.csv":
                csv_member = member
                break

        if csv_member is None:
            return {}, {}, {}, {}, {}, False, {}

        f = io.TextIOWrapper(tar.extractfile(csv_member), encoding="utf-8")
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

        is_new_format = "composite_rank" in headers
        higher_is_better = is_new_format
        component_cols = [h for h in headers if h.endswith("_score") and h != "composite_score"]
        stage_col = "stage_bucket" if is_new_format else "Stage_Bucket"

        for col in component_cols:
            component_scores[col] = {}

        for row in reader:
            if is_new_format:
                ticker = row.get("ticker", "").strip().upper()
                rank_str = row.get("composite_rank", "").strip()
                score_str = row.get("composite_score", "").strip()
            else:
                ticker = row.get("Ticker", "").strip().upper()
                rank_str = row.get("Rank", "").strip()
                score_str = row.get("Composite_Score", "").strip()

            if not ticker or not rank_str:
                continue

            try:
                rankings[ticker] = int(rank_str)
            except ValueError:
                continue

            if score_str:
                try:
                    scores[ticker] = float(score_str)
                except ValueError:
                    pass

            raw_stage = row.get(stage_col, "").strip().lower()
            stage_map[ticker] = "commercial" if raw_stage == "commercial" else "dev"

            for col in component_cols:
                val_str = row.get(col, "").strip()
                if val_str:
                    try:
                        component_scores[col][ticker] = float(val_str)
                    except ValueError:
                        pass

            for sig_field in ("score_rank_pct", "score_z"):
                sig_str = row.get(sig_field, "").strip()
                if sig_str:
                    try:
                        signal_scores[sig_field][ticker] = float(sig_str)
                    except ValueError:
                        pass

            cash_str = row.get("Cash", "").strip()
            if cash_str:
                try:
                    cash_val = float(cash_str)
                    if cash_val > 0:
                        extra_controls["log_cash"][ticker] = math.log(cash_val)
                except ValueError:
                    pass

    extra_controls = {k: v for k, v in extra_controls.items() if v}
    signal_scores = {k: v for k, v in signal_scores.items() if v}

    return rankings, scores, component_scores, stage_map, extra_controls, higher_is_better, signal_scores


# Metadata fields to extract for failure attribution (new-format archives)
METADATA_FIELDS_NEW = [
    "confidence_overall", "catalyst_score", "smart_money_score",
    "archetype", "severity", "market_cap_bucket",
]
# Legacy format
METADATA_FIELDS_LEGACY = ["Momentum_Bucket"]


def extract_ticker_metadata_from_archive(tar_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Extract per-ticker metadata from an archive's rankings.csv.

    Returns {ticker: {field: value, ...}} for failure attribution.
    Numeric fields are cast to float; string fields kept as-is.
    """
    metadata: Dict[str, Dict[str, Any]] = {}

    with tarfile.open(tar_path, "r:gz") as tar:
        csv_member = None
        for member in tar.getmembers():
            if member.name.endswith("/rankings.csv") or member.name == "rankings.csv":
                csv_member = member
                break
        if csv_member is None:
            return {}

        f = io.TextIOWrapper(tar.extractfile(csv_member), encoding="utf-8")
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        is_new = "composite_rank" in headers
        fields = METADATA_FIELDS_NEW if is_new else METADATA_FIELDS_LEGACY

        for row in reader:
            ticker = row.get("ticker" if is_new else "Ticker", "").strip().upper()
            if not ticker:
                continue
            meta: Dict[str, Any] = {}
            for field in fields:
                val = row.get(field, "").strip()
                if not val:
                    continue
                try:
                    meta[field] = float(val)
                except ValueError:
                    meta[field] = val
            metadata[ticker] = meta

    return metadata


def extract_ticker_metadata_from_snapshot(snapshot_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Extract per-ticker metadata from a legacy snapshot's rankings.csv."""
    metadata: Dict[str, Dict[str, Any]] = {}
    csv_path = snapshot_dir / "rankings.csv"
    if not csv_path.exists():
        return {}

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row.get("Ticker", "").strip().upper()
            if not ticker:
                continue
            meta: Dict[str, Any] = {}
            for field in METADATA_FIELDS_LEGACY:
                val = row.get(field, "").strip()
                if val:
                    try:
                        meta[field] = float(val)
                    except ValueError:
                        meta[field] = val
            metadata[ticker] = meta

    return metadata


def build_failure_attribution(
    ic_rankings: Dict[str, int],
    fwd_returns: Dict[str, float],
    stage_map: Dict[str, str],
    component_scores: Dict[str, Dict[str, float]],
    ticker_metadata: Dict[str, Dict[str, Any]],
    n_show: int = 10,
) -> Optional[Dict[str, Any]]:
    """
    Build failure attribution: identify top and bottom quintile tickers
    with their metadata to explain where signal works / breaks.

    Returns dict with top_quintile and bottom_quintile lists, each entry
    containing ticker, rank, return, stage, component scores, and metadata.
    """
    common = sorted(
        [t for t in ic_rankings if t in fwd_returns],
        key=lambda t: ic_rankings[t]
    )
    n = len(common)
    if n < 20:
        return None

    q_size = n // 5

    def _build_entries(tickers: List[str]) -> List[Dict[str, Any]]:
        entries = []
        for t in tickers:
            entry: Dict[str, Any] = {
                "ticker": t,
                "rank": ic_rankings[t],
                "return_pct": round(fwd_returns[t] * 100, 4),
                "stage": stage_map.get(t, "?"),
            }
            # Component scores
            for comp_name, comp_vals in component_scores.items():
                if t in comp_vals and comp_vals[t] is not None:
                    short_name = comp_name.replace("_Score", "").replace("_score", "")
                    entry[short_name] = round(comp_vals[t], 2)
            # Extra metadata
            meta = ticker_metadata.get(t, {})
            for k, v in meta.items():
                entry[k] = round(v, 4) if isinstance(v, float) else v
            entries.append(entry)
        return entries

    # Sort each quintile by return (worst first for bottom, best first for top)
    top_q = common[:q_size]
    bottom_q = common[-q_size:]

    top_sorted = sorted(top_q, key=lambda t: fwd_returns[t], reverse=True)[:n_show]
    bottom_sorted = sorted(bottom_q, key=lambda t: fwd_returns[t])[:n_show]

    top_entries = _build_entries(top_sorted)
    bottom_entries = _build_entries(bottom_sorted)

    # Stage distribution
    top_stages = {"dev": 0, "commercial": 0}
    bottom_stages = {"dev": 0, "commercial": 0}
    for t in common[:q_size]:
        s = stage_map.get(t, "dev")
        top_stages[s] = top_stages.get(s, 0) + 1
    for t in common[-q_size:]:
        s = stage_map.get(t, "dev")
        bottom_stages[s] = bottom_stages.get(s, 0) + 1

    return {
        "n_per_quintile": q_size,
        "top_quintile": {
            "stage_mix": top_stages,
            "mean_return_pct": round(mean([fwd_returns[t] for t in common[:q_size]]) * 100, 4),
            "tickers": top_entries,
        },
        "bottom_quintile": {
            "stage_mix": bottom_stages,
            "mean_return_pct": round(mean([fwd_returns[t] for t in common[-q_size:]]) * 100, 4),
            "tickers": bottom_entries,
        },
    }


def discover_archives(
    archive_dir: Path,
    start: Optional[str] = None,
    end: Optional[str] = None,
    max_n: Optional[int] = None,
) -> List[Tuple[str, Path]]:
    """
    Discover archive tarballs in archive_dir, parse dates from filenames,
    filter by [start, end] inclusive, sort ascending.

    Returns list of (date_str, tar_path) tuples.
    """
    pattern = re.compile(r"(\d{4}-\d{2}-\d{2})\.tar\.gz$")
    results: List[Tuple[str, Path]] = []

    for p in sorted(archive_dir.glob("*.tar.gz")):
        m = pattern.search(p.name)
        if not m:
            continue
        d = m.group(1)
        if start and d < start:
            continue
        if end and d > end:
            continue
        results.append((d, p))

    results.sort(key=lambda x: x[0])
    if max_n is not None:
        results = results[:max_n]
    return results


# =============================================================================
# ARCHIVE VERIFICATION
# =============================================================================

def verify_archive_for_backtest(tar_path: Path) -> Dict[str, Any]:
    """
    Verify archive integrity before using it in backtest.

    Checks:
      1. archive_snapshot.verify_archive() — SHA-256 of input files vs manifest
      2. rankings.csv exists and is non-empty inside the tarball

    Returns dict with:
      - verified: bool
      - input_files_ok: bool
      - rankings_ok: bool
      - error: str or None
    """
    result: Dict[str, Any] = {
        "archive": str(tar_path),
        "verified": False,
        "input_files_ok": False,
        "rankings_ok": False,
        "error": None,
    }

    # 1. Input file hash verification (reuses archive_snapshot logic)
    try:
        # Suppress stdout from verify_archive (it prints OK/FAIL)
        import io as _io
        import contextlib
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            input_ok = verify_archive(tar_path)
        result["input_files_ok"] = input_ok
    except Exception as e:
        result["error"] = f"verify_archive failed: {e}"
        return result

    # 2. Check rankings.csv exists and has content
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            csv_member = None
            for member in tar.getmembers():
                if member.name.endswith("/rankings.csv") or member.name == "rankings.csv":
                    csv_member = member
                    break
            if csv_member is None:
                result["error"] = "rankings.csv not found in archive"
                return result
            if csv_member.size == 0:
                result["error"] = "rankings.csv is empty"
                return result
            result["rankings_ok"] = True
    except Exception as e:
        result["error"] = f"rankings.csv check failed: {e}"
        return result

    result["verified"] = result["input_files_ok"] and result["rankings_ok"]
    return result


# =============================================================================
# AS-OF FENCE
# =============================================================================

def compute_as_of_fence(
    snapshot_dates: List[str],
    last_return_date: Optional[date],
    max_horizon_days: int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Determine which snapshots have sufficient forward return data
    and which must be skipped.

    Returns:
        (usable_dates, skipped_records)

    Each skipped record has:
        snapshot_date, skipped_reason, needed_end_date, last_return_date
    """
    if last_return_date is None:
        return [], [
            {
                "snapshot_date": d,
                "skipped_reason": "no_return_data_loaded",
                "needed_end_date": None,
                "last_return_date": None,
            }
            for d in snapshot_dates
        ]

    usable: List[str] = []
    skipped: List[Dict[str, Any]] = []

    for snap_date in snapshot_dates:
        needed_end = add_trading_days(snap_date, max_horizon_days)
        needed_end_dt = date.fromisoformat(needed_end)

        if needed_end_dt > last_return_date:
            skipped.append({
                "snapshot_date": snap_date,
                "skipped_reason": "insufficient_forward_window",
                "needed_end_date": needed_end,
                "last_return_date": last_return_date.isoformat(),
                "shortfall_calendar_days": (needed_end_dt - last_return_date).days,
            })
        else:
            usable.append(snap_date)

    return usable, skipped


# =============================================================================
# RANK UNIQUENESS & SIGNAL CONVERSION
# =============================================================================

def assert_rank_unique(
    rankings: Dict[str, int],
    date_str: str,
    signal_scores: Dict[str, Dict[str, float]],
    allow_ties: bool = False,
) -> bool:
    """
    Assert composite_rank values are strictly unique per snapshot.
    Also warn if score_rank_pct is not monotonic with composite_rank.

    Returns True if ranks are unique (or ties allowed), False otherwise.
    """
    vals = list(rankings.values())
    unique_vals = set(vals)
    if len(vals) != len(unique_vals):
        dup_counts = {}
        for v in vals:
            dup_counts[v] = dup_counts.get(v, 0) + 1
        dups = {k: v for k, v in dup_counts.items() if v > 1}
        n_dup = sum(v - 1 for v in dups.values())
        sample = dict(list(dups.items())[:5])
        msg = f"  WARNING {date_str}: {n_dup} duplicate rank values (sample: {sample})"
        if allow_ties:
            print(msg + " [--allow-rank-ties: continuing]")
        else:
            print(msg)
            raise ValueError(f"Rank ties in {date_str}. Use --allow-rank-ties to proceed.")

    # Check score_rank_pct monotonicity with composite_rank
    rank_pct = signal_scores.get("score_rank_pct", {})
    if rank_pct:
        sorted_by_rank = sorted(rankings.items(), key=lambda x: x[1])
        prev_pct = None
        violations = 0
        for ticker, _rank in sorted_by_rank:
            pct = rank_pct.get(ticker)
            if pct is not None and prev_pct is not None:
                if pct > prev_pct:  # higher rank_pct should mean worse rank (lower is better)
                    violations += 1
            prev_pct = pct
        if violations > 0:
            print(f"  WARNING {date_str}: score_rank_pct non-monotonic with composite_rank "
                  f"({violations} violations out of {len(sorted_by_rank) - 1} pairs)")

    return True


def signal_to_rankings(
    signal_scores: Dict[str, float],
    higher_is_better: bool = True,
) -> Dict[str, int]:
    """
    Convert float signal values to dense integer ranks (1 = best).

    For score_rank_pct: lower value = better (higher_is_better=False)
    For score_z: higher value = better (higher_is_better=True)
    """
    sorted_tickers = sorted(
        signal_scores.keys(),
        key=lambda t: signal_scores[t],
        reverse=higher_is_better,
    )
    return {t: i + 1 for i, t in enumerate(sorted_tickers)}


# =============================================================================
# INVESTABLE UNIVERSE FILTERING
# =============================================================================

def filter_to_investable(
    rankings: Dict[str, int],
    scores: Dict[str, float],
    component_scores: Dict[str, Dict[str, float]],
    stage_map: Dict[str, str],
    extra_controls: Dict[str, Dict[str, float]],
    investable_tickers: set,
) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, Dict[str, float]], Dict[str, str], Dict[str, Dict[str, float]]]:
    """
    Restrict to tickers with valid forward returns and re-rank
    within the investable set, preserving original rank order.

    This eliminates rank-correlated missingness: every ticker in the
    resulting rankings has a measurable forward return.
    """
    # Sort investable tickers by their original rank (preserves ordering)
    investable_sorted = sorted(
        [t for t in rankings if t in investable_tickers],
        key=lambda t: rankings[t]
    )

    # Re-rank: dense 1..N within investable set
    new_rankings = {t: i + 1 for i, t in enumerate(investable_sorted)}

    # Filter scores, components, stage, and controls to same set
    new_scores = {t: scores[t] for t in new_rankings if t in scores}
    new_stage = {t: stage_map[t] for t in new_rankings if t in stage_map}
    new_components: Dict[str, Dict[str, float]] = {}
    for comp_name, comp_vals in component_scores.items():
        filtered = {t: comp_vals[t] for t in new_rankings if t in comp_vals}
        if filtered:
            new_components[comp_name] = filtered
    new_controls: Dict[str, Dict[str, float]] = {}
    for ctrl_name, ctrl_vals in extra_controls.items():
        filtered = {t: ctrl_vals[t] for t in new_rankings if t in ctrl_vals}
        if filtered:
            new_controls[ctrl_name] = filtered

    return new_rankings, new_scores, new_components, new_stage, new_controls


# =============================================================================
# HYGIENE / DIAGNOSTICS
# =============================================================================

def diagnose_missing_returns(
    provider: BaseReturnsProvider,
    rankings: Dict[str, int],
    fwd_returns: Dict[str, float],
    as_of_date: str,
    ms_provider: MorningstarReturnsProvider,
) -> Dict[str, Any]:
    """
    Diagnose why tickers are missing forward returns and whether
    the missingness is random or systematic (correlated with rank).
    """
    tickers = list(rankings.keys())
    n_total = len(tickers)
    missing_tickers = sorted(set(tickers) - set(fwd_returns.keys()))
    present_tickers = sorted(set(tickers) & set(fwd_returns.keys()))

    as_of_dt = date.fromisoformat(as_of_date)

    # Classify why each ticker is missing (check against both sources)
    no_data_anywhere = []      # not in HS793 or CSV
    series_starts_after = []   # data exists but starts after snapshot date
    no_start_price = []        # data exists but no price near start date
    no_end_price = []          # data exists but no price near end date

    for t in missing_tickers:
        t_upper = t.upper()
        hs793_prices = ms_provider._index.get(t_upper)

        if hs793_prices is None:
            # Check if CSV has it (via the chained provider)
            # If it's still missing from fwd_returns, CSV doesn't help either
            no_data_anywhere.append(t)
            continue

        min_date = min(hs793_prices.keys())
        if min_date > as_of_dt + timedelta(days=10):
            series_starts_after.append(t)
        elif ms_provider._find_nearest(t_upper, as_of_dt + timedelta(days=1)) is None:
            no_start_price.append(t)
        else:
            no_end_price.append(t)

    # Rank-quintile analysis of missing vs present
    quintile_missing = [0, 0, 0, 0, 0]  # Q1 (best) through Q5 (worst)
    quintile_total = [0, 0, 0, 0, 0]
    if n_total > 0:
        for t in tickers:
            rank = rankings[t]
            q = min(4, (rank - 1) * 5 // n_total)
            quintile_total[q] += 1
            if t in missing_tickers:
                quintile_missing[q] += 1

    quintile_pcts = []
    for i in range(5):
        pct = (quintile_missing[i] / quintile_total[i] * 100) if quintile_total[i] > 0 else 0.0
        quintile_pcts.append(round(pct, 1))

    max_quintile_miss = max(quintile_pcts) if quintile_pcts else 0
    min_quintile_miss = min(quintile_pcts) if quintile_pcts else 0
    miss_spread = max_quintile_miss - min_quintile_miss

    return {
        "n_ranked": n_total,
        "n_with_returns": len(present_tickers),
        "n_missing": len(missing_tickers),
        "miss_pct": round(len(missing_tickers) / n_total * 100, 1) if n_total else 0,
        "missing_reason": {
            "no_data_anywhere": len(no_data_anywhere),
            "series_starts_after_snapshot": len(series_starts_after),
            "no_start_price": len(no_start_price),
            "no_end_price": len(no_end_price),
        },
        "missing_tickers": missing_tickers,  # full list, not capped
        "rank_quintile_miss_pct": {
            f"Q{i+1}": quintile_pcts[i] for i in range(5)
        },
        "rank_quintile_miss_count": {
            f"Q{i+1}": f"{quintile_missing[i]}/{quintile_total[i]}" for i in range(5)
        },
        "missingness_spread_pp": round(miss_spread, 1),
        "missingness_verdict": (
            "RANDOM" if miss_spread < 10 else
            "MILD_BIAS" if miss_spread < 20 else
            "SYSTEMATIC"
        ),
    }


def build_hygiene_report(
    provider: BaseReturnsProvider,
    ms_provider: MorningstarReturnsProvider,
    snapshot_date: str,
    rankings: Dict[str, int],
    scores: Dict[str, float],
    component_scores: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """Build comprehensive data hygiene report for one snapshot."""
    tickers = list(rankings.keys())
    n_total = len(tickers)

    hygiene: Dict[str, Any] = {"date": snapshot_date, "n_ranked": n_total}

    for horizon_label, horizon_days in HORIZONS.items():
        fwd_returns = compute_forward_returns(provider, tickers, snapshot_date, horizon_days)
        diag = diagnose_missing_returns(provider, rankings, fwd_returns, snapshot_date, ms_provider)
        hygiene[f"returns_{horizon_label}"] = diag

    # Component score coverage (count only non-null values)
    comp_coverage: Dict[str, Dict[str, Any]] = {}
    for comp_name, comp_vals in component_scores.items():
        n_with_score = sum(1 for v in comp_vals.values() if v is not None)
        n_missing = n_total - n_with_score
        comp_coverage[comp_name] = {
            "n_with_score": n_with_score,
            "n_missing": n_missing,
            "miss_pct": round(n_missing / n_total * 100, 1) if n_total else 0,
        }
    hygiene["component_coverage"] = comp_coverage

    if scores:
        score_vals = sorted(scores.values())
        hygiene["composite_score_stats"] = {
            "min": round(min(score_vals), 2),
            "max": round(max(score_vals), 2),
            "mean": round(mean(score_vals), 2),
            "n_unique": len(set(score_vals)),
            "n_ties_at_min": sum(1 for v in score_vals if v == score_vals[0]),
        }

    return hygiene


def print_hygiene_report(
    hygiene_before: List[Dict[str, Any]],
    hygiene_after: List[Dict[str, Any]],
    fallback_stats: Dict[str, Any],
) -> None:
    """Print formatted hygiene/diagnostic table showing before/after filtering."""
    print()
    print("=" * 90)
    print("DATA HYGIENE REPORT")
    print("=" * 90)

    # Fallback stats
    print()
    fb = fallback_stats
    print(f"Chained returns: HS793 primary + price_history.csv fallback")
    print(f"  Fallback used for {fb['n_tickers_used_fallback']} tickers, "
          f"{fb['total_fallback_lookups']} total lookups")
    if fb['tickers']:
        print(f"  Fallback tickers: {', '.join(fb['tickers'][:20])}")
    print()

    # BEFORE filtering table
    print("BEFORE investable-universe filtering (raw snapshot):")
    header = f"{'Date':>12s}  {'Ranked':>6s}  {'Ret20d':>6s}  {'Miss%':>5s}  {'Ret60d':>6s}  {'Miss%':>5s}  {'Verdict20d':>12s}"
    print(f"  {header}")
    print(f"  {'-' * len(header)}")
    for h in hygiene_before:
        r20 = h.get("returns_20d", {})
        r60 = h.get("returns_60d", {})
        print(
            f"  {h['date']:>12s}  "
            f"{h['n_ranked']:>6d}  "
            f"{r20.get('n_with_returns', 0):>6d}  "
            f"{r20.get('miss_pct', 0):>4.1f}%  "
            f"{r60.get('n_with_returns', 0):>6d}  "
            f"{r60.get('miss_pct', 0):>4.1f}%  "
            f"{r20.get('missingness_verdict', '?'):>12s}"
        )
    print()

    # Detailed per-snapshot BEFORE
    for h in hygiene_before:
        print(f"  --- {h['date']} ---")
        for hlabel in ["returns_20d", "returns_60d"]:
            r = h.get(hlabel, {})
            if not r:
                continue
            reason = r.get("missing_reason", {})
            print(f"    {hlabel}: {r['n_with_returns']}/{r['n_ranked']} "
                  f"({r['miss_pct']}% missing)")
            print(f"      Why: "
                  f"no_data={reason.get('no_data_anywhere', 0)}, "
                  f"IPO_after={reason.get('series_starts_after_snapshot', 0)}, "
                  f"no_start={reason.get('no_start_price', 0)}, "
                  f"no_end={reason.get('no_end_price', 0)}")
            qm = r.get("rank_quintile_miss_pct", {})
            qc = r.get("rank_quintile_miss_count", {})
            print(f"      Q1={qm.get('Q1', 0):.0f}%({qc.get('Q1', '')}), "
                  f"Q2={qm.get('Q2', 0):.0f}%({qc.get('Q2', '')}), "
                  f"Q3={qm.get('Q3', 0):.0f}%({qc.get('Q3', '')}), "
                  f"Q4={qm.get('Q4', 0):.0f}%({qc.get('Q4', '')}), "
                  f"Q5={qm.get('Q5', 0):.0f}%({qc.get('Q5', '')})")
        print()

    # Cross-snapshot: always missing (using FULL lists now)
    all_missing_sets = []
    for h in hygiene_before:
        r20 = h.get("returns_20d", {})
        all_missing_sets.append(set(r20.get("missing_tickers", [])))

    if len(all_missing_sets) >= 2:
        always_missing = all_missing_sets[0]
        for s in all_missing_sets[1:]:
            always_missing = always_missing & s
        if always_missing:
            print(f"  Tickers missing in ALL {len(all_missing_sets)} snapshots: "
                  f"{len(always_missing)} tickers")
            print(f"    {', '.join(sorted(always_missing)[:25])}")
            if len(always_missing) > 25:
                print(f"    ... and {len(always_missing) - 25} more")
            print()

    # AFTER filtering table
    print("AFTER investable-universe filtering (IC computed on this set):")
    header = f"{'Date':>12s}  {'Investable':>10s}  {'Dropped':>7s}  {'Miss%20d':>8s}  {'Miss%60d':>8s}  {'Verdict':>10s}"
    print(f"  {header}")
    print(f"  {'-' * len(header)}")
    for h in hygiene_after:
        r20 = h.get("returns_20d", {})
        r60 = h.get("returns_60d", {})
        print(
            f"  {h['date']:>12s}  "
            f"{h['n_ranked']:>10d}  "
            f"{h.get('n_dropped', 0):>7d}  "
            f"{r20.get('miss_pct', 0):>7.1f}%  "
            f"{r60.get('miss_pct', 0):>7.1f}%  "
            f"{r20.get('missingness_verdict', '?'):>10s}"
        )
    print()
    print("=" * 90)


# =============================================================================
# MAIN IC COMPUTATION
# =============================================================================

def compute_forward_returns(
    provider: BaseReturnsProvider,
    tickers: List[str],
    as_of_date: str,
    horizon_days: int,
) -> Dict[str, float]:
    """Compute forward returns for all tickers at a given horizon."""
    start = add_trading_days(as_of_date, 1)  # next trading day
    end = add_trading_days(as_of_date, horizon_days)

    returns: Dict[str, float] = {}
    for ticker in tickers:
        ret = provider.get_forward_total_return(ticker, start, end)
        if ret is not None:
            returns[ticker] = float(ret)
    return returns


def run_backtest(
    provider: ChainedReturnsProvider,
    ms_provider: MorningstarReturnsProvider,
    csv_provider: CSVReturnsProvider,
    snapshot_dir: Optional[Path] = None,
    archives: Optional[List[Tuple[str, Path]]] = None,
    signal_field: str = "score_rank_pct",
    allow_rank_ties: bool = False,
    verify_archives: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Run the full rank-IC backtest across snapshots or archives.

    Returns:
        (ic_results, hygiene_before, hygiene_after)
    """
    snap_base = snapshot_dir or SNAPSHOT_DIR
    ic_engine = ICCalculationEngine()

    # Determine iteration source
    if archives is not None:
        snapshot_dates = [d for d, _ in archives]
    else:
        snapshot_dates = list(QUARTERLY_DATES)

    # ------------------------------------------------------------------
    # AS-OF FENCE: pre-filter snapshots with insufficient forward data
    # ------------------------------------------------------------------
    max_horizon = max(HORIZONS.values())
    last_return_date = provider.get_last_date()

    print(f"  Returns data ends: {last_return_date.isoformat() if last_return_date else 'NONE'}")
    print(f"  Max forward horizon: {max_horizon} trading days")

    usable_dates, fence_skipped = compute_as_of_fence(
        snapshot_dates, last_return_date, max_horizon
    )

    if fence_skipped:
        print(f"  As-of fence: {len(usable_dates)} usable, {len(fence_skipped)} skipped")
        for sk in fence_skipped:
            print(f"    SKIP {sk['snapshot_date']}: {sk['skipped_reason']} "
                  f"(need {sk.get('needed_end_date', '?')}, "
                  f"have {sk.get('last_return_date', '?')}, "
                  f"shortfall {sk.get('shortfall_calendar_days', '?')}d)")
    else:
        print(f"  As-of fence: all {len(usable_dates)} snapshots usable")
    print()

    # Build lookup for archive paths (if applicable)
    archive_by_date: Dict[str, Path] = {}
    if archives is not None:
        archive_by_date = {d: p for d, p in archives}

    # ------------------------------------------------------------------
    # ARCHIVE VERIFICATION (optional, default on)
    # ------------------------------------------------------------------
    verification_results: List[Dict[str, Any]] = []
    verify_skipped: List[Dict[str, Any]] = []

    if verify_archives and archives is not None:
        print("Verifying archive integrity...")
        for snap_date in usable_dates:
            tar_path = archive_by_date[snap_date]
            v = verify_archive_for_backtest(tar_path)
            verification_results.append(v)
            if not v["verified"]:
                verify_skipped.append({
                    "snapshot_date": snap_date,
                    "skipped_reason": "archive_integrity_failed",
                    "error": v.get("error"),
                    "input_files_ok": v["input_files_ok"],
                    "rankings_ok": v["rankings_ok"],
                })
                print(f"    FAIL {snap_date}: {v.get('error', 'hash mismatch')}")
            else:
                print(f"    OK   {snap_date}")
        # Remove failed archives from usable set
        failed_dates = {sk["snapshot_date"] for sk in verify_skipped}
        usable_dates = [d for d in usable_dates if d not in failed_dates]
        print()

    # Combine all skip records
    all_skipped = fence_skipped + verify_skipped

    # Pre-compute regime for usable snapshots only
    regimes = compute_xbi_regime(csv_provider, usable_dates)

    per_snapshot: List[Dict[str, Any]] = []
    hygiene_before: List[Dict[str, Any]] = []
    hygiene_after: List[Dict[str, Any]] = []

    for snapshot_date in usable_dates:
        # Load rankings from archive or legacy snapshot dir
        if archives is not None:
            tar_path = archive_by_date[snapshot_date]
            rankings, scores, component_scores, stage_map, extra_controls, higher_is_better, signal_scores_data = read_rankings_from_archive(tar_path)
            archive_path_str = str(tar_path)
        else:
            snap_dir = snap_base / snapshot_date
            if not snap_dir.exists():
                all_skipped.append({
                    "snapshot_date": snapshot_date,
                    "skipped_reason": "directory_not_found",
                    "path": str(snap_dir),
                })
                print(f"  SKIP {snapshot_date}: directory not found")
                continue
            rankings, scores, component_scores, stage_map, extra_controls, higher_is_better, signal_scores_data = load_snapshot_rankings(snap_dir)
            archive_path_str = None

        if not rankings:
            all_skipped.append({
                "snapshot_date": snapshot_date,
                "skipped_reason": "no_rankings_loaded",
            })
            print(f"  SKIP {snapshot_date}: no rankings loaded")
            continue

        # Extract per-ticker metadata for failure attribution
        if archives is not None:
            ticker_metadata = extract_ticker_metadata_from_archive(archive_by_date[snapshot_date])
        else:
            ticker_metadata = extract_ticker_metadata_from_snapshot(snap_base / snapshot_date)

        # Rank uniqueness check
        assert_rank_unique(rankings, snapshot_date, signal_scores_data, allow_ties=allow_rank_ties)

        tickers = list(rankings.keys())
        n_original = len(rankings)
        regime = regimes.get(snapshot_date, "UNKNOWN")

        # --- BEFORE filtering hygiene ---
        hygiene_b = build_hygiene_report(
            provider, ms_provider, snapshot_date, rankings, scores, component_scores
        )
        hygiene_before.append(hygiene_b)

        # --- Determine investable universe ---
        max_horizon = max(HORIZONS.values())
        fwd_returns_max = compute_forward_returns(provider, tickers, snapshot_date, max_horizon)
        investable = set(fwd_returns_max.keys())

        for horizon_days in HORIZONS.values():
            fwd = compute_forward_returns(provider, tickers, snapshot_date, horizon_days)
            investable &= set(fwd.keys())

        # --- Filter and re-rank ---
        inv_rankings, inv_scores, inv_components, inv_stage, inv_controls = filter_to_investable(
            rankings, scores, component_scores, stage_map, extra_controls, investable
        )
        n_investable = len(inv_rankings)

        # --- AFTER filtering hygiene ---
        hygiene_a = build_hygiene_report(
            provider, ms_provider, snapshot_date, inv_rankings, inv_scores, inv_components
        )
        hygiene_a["n_dropped"] = n_original - n_investable
        hygiene_after.append(hygiene_a)

        # --- Compute IC on investable universe ---
        inv_tickers = list(inv_rankings.keys())

        # Determine which rankings to use for IC: signal field or composite_rank
        if signal_field in ("score_rank_pct", "score_z") and signal_scores_data.get(signal_field):
            # Filter signal to investable tickers only
            inv_signal = {t: signal_scores_data[signal_field][t]
                          for t in inv_tickers if t in signal_scores_data[signal_field]}
            sig_higher_is_better = (signal_field == "score_z")
            ic_rankings = signal_to_rankings(inv_signal, higher_is_better=sig_higher_is_better)
        else:
            ic_rankings = inv_rankings

        snapshot_result: Dict[str, Any] = {
            "date": snapshot_date,
            "n_original": n_original,
            "n_investable": n_investable,
            "n_dropped": n_original - n_investable,
            "n_rows": n_original,
            "regime": regime,
            "signal_field": signal_field,
            "rank_unique": len(set(rankings.values())) == len(rankings),
        }
        if archive_path_str:
            snapshot_result["archive_path"] = archive_path_str

        # --- Estimate XBI betas for residual returns ---
        xbi_betas = estimate_xbi_betas(csv_provider, inv_tickers, snapshot_date)
        snapshot_result["n_with_beta"] = len(xbi_betas)

        for horizon_label, horizon_days in HORIZONS.items():
            fwd_returns = compute_forward_returns(provider, inv_tickers, snapshot_date, horizon_days)
            n_with_returns = len(fwd_returns)
            coverage = n_with_returns / len(inv_tickers) if inv_tickers else 0

            ic_result = ic_engine.calculate_ic_with_significance(ic_rankings, fwd_returns)

            snapshot_result[f"ic_{horizon_label}"] = float(ic_result.ic_value)
            snapshot_result[f"n_{horizon_label}"] = ic_result.n_observations
            snapshot_result[f"coverage_{horizon_label}"] = round(coverage * 100, 1)
            snapshot_result[f"quality_{horizon_label}"] = ic_result.ic_quality.value
            snapshot_result[f"tstat_{horizon_label}"] = float(ic_result.t_statistic) if ic_result.t_statistic else None
            snapshot_result[f"sig95_{horizon_label}"] = ic_result.is_significant_95

            # Kendall tau-b (more robust with ties)
            kendall_val = calculate_kendall(ic_rankings, fwd_returns)
            snapshot_result[f"kendall_{horizon_label}"] = float(kendall_val) if kendall_val is not None else None

            # Quintile spread
            qs = compute_quintile_spread(ic_rankings, fwd_returns)
            if qs is not None:
                snapshot_result[f"quintile_spread_{horizon_label}"] = qs

            # --- XBI-residual returns ---
            xbi_fwd_start = add_trading_days(snapshot_date, 1)
            xbi_fwd_end = add_trading_days(snapshot_date, horizon_days)
            xbi_ret_str = csv_provider.get_forward_total_return("XBI", xbi_fwd_start, xbi_fwd_end)
            xbi_fwd = float(xbi_ret_str) if xbi_ret_str is not None else None

            resid_returns = compute_residual_returns(fwd_returns, xbi_fwd, xbi_betas)
            if resid_returns:
                resid_ic = calculate_ic(ic_rankings, resid_returns)
                resid_kendall = calculate_kendall(ic_rankings, resid_returns)
                resid_qs = compute_quintile_spread(ic_rankings, resid_returns)

                snapshot_result[f"resid_ic_{horizon_label}"] = float(resid_ic) if resid_ic is not None else None
                snapshot_result[f"resid_kendall_{horizon_label}"] = float(resid_kendall) if resid_kendall is not None else None
                if resid_qs is not None:
                    snapshot_result[f"resid_quintile_spread_{horizon_label}"] = resid_qs
                snapshot_result[f"xbi_return_{horizon_label}"] = round(xbi_fwd * 100, 4) if xbi_fwd is not None else None

                # Residual stage-split IC
                resid_stage_ic = compute_stratified_ic(ic_rankings, resid_returns, inv_stage)
                snapshot_result[f"resid_stage_ic_{horizon_label}"] = resid_stage_ic

            # Failure attribution: top/bottom quintile with metadata
            attrib = build_failure_attribution(
                ic_rankings, fwd_returns, inv_stage, inv_components, ticker_metadata,
            )
            if attrib is not None:
                snapshot_result[f"_failure_attrib_{horizon_label}"] = attrib

            # Stratified IC: by stage
            stage_ic = compute_stratified_ic(ic_rankings, fwd_returns, inv_stage)
            snapshot_result[f"stage_ic_{horizon_label}"] = stage_ic

            # Store for decile spread + pooled regression
            snapshot_result[f"_fwd_returns_{horizon_label}"] = fwd_returns
            if horizon_label == "60d":
                snapshot_result["_rankings"] = ic_rankings
                snapshot_result["_returns_60d"] = fwd_returns

        # Store internal data for pooled regression
        snapshot_result["_component_scores"] = inv_components
        snapshot_result["_extra_controls"] = inv_controls
        snapshot_result["_stage_map"] = inv_stage

        # Component IC (20d horizon) on investable set
        fwd_returns_20d = compute_forward_returns(provider, inv_tickers, snapshot_date, 20)
        comp_ic: Dict[str, float] = {}
        for comp_name, comp_vals in inv_components.items():
            n_non_null = sum(1 for v in comp_vals.values() if v is not None)
            if n_non_null < MIN_OBS_IC:
                continue
            valid_tickers = [t for t in comp_vals if comp_vals[t] is not None]
            sorted_tickers = sorted(valid_tickers, key=lambda t: comp_vals[t], reverse=higher_is_better)
            comp_rankings = {t: i + 1 for i, t in enumerate(sorted_tickers)}
            comp_returns = {t: fwd_returns_20d[t] for t in comp_rankings if t in fwd_returns_20d}
            ic_val = calculate_ic(comp_rankings, comp_returns)
            if ic_val is not None:
                comp_ic[comp_name] = float(ic_val)

        snapshot_result["component_ic_20d"] = comp_ic

        # --- Partial correlation: residualize both returns and component ---
        for horizon_label, horizon_days in HORIZONS.items():
            fwd_returns = compute_forward_returns(provider, inv_tickers, snapshot_date, horizon_days)
            partial_corrs: Dict[str, Any] = {}
            for comp_name in inv_components:
                pc = compute_partial_correlation(
                    inv_components, inv_controls, fwd_returns, comp_name,
                )
                if pc is not None:
                    partial_corrs[comp_name] = pc
            if partial_corrs:
                snapshot_result[f"partial_corr_{horizon_label}"] = partial_corrs

            # Bucket spread for bucketed components
            bucket_spreads: Dict[str, Any] = {}
            for comp_name in inv_components:
                bs = compute_bucket_spread(inv_components, fwd_returns, comp_name, inv_stage)
                if bs is not None:
                    bucket_spreads[comp_name] = bs
            if bucket_spreads:
                snapshot_result[f"bucket_spread_{horizon_label}"] = bucket_spreads

        # --- VIF (same across horizons, compute once) ---
        vif = compute_vif(inv_components, inv_controls)
        if vif:
            snapshot_result["vif"] = vif

        # --- Cross-sectional regression per horizon ---
        for horizon_label, horizon_days in HORIZONS.items():
            fwd_returns = compute_forward_returns(provider, inv_tickers, snapshot_date, horizon_days)
            reg_result = run_cross_sectional_regression(
                fwd_returns, inv_components, inv_controls, inv_stage,
            )
            if reg_result is not None:
                snapshot_result[f"regression_{horizon_label}"] = reg_result

        per_snapshot.append(snapshot_result)
        print(f"  {snapshot_date} [{regime:8s}]: IC_20d={snapshot_result['ic_20d']:+.4f}  "
              f"IC_60d={snapshot_result['ic_60d']:+.4f}  "
              f"(n={snapshot_result['n_20d']}, dropped={snapshot_result['n_dropped']})")

    # Pooled interaction regression across all snapshots
    pooled_reg = run_pooled_interaction_regression(per_snapshot)

    agg = _aggregate(per_snapshot)
    if pooled_reg:
        agg["pooled_interaction_regression"] = pooled_reg

    # Attach as-of fence + verification metadata
    agg["returns_coverage"] = {
        "last_return_date": last_return_date.isoformat() if last_return_date else None,
        "max_forward_horizon_days": max_horizon,
    }
    agg["skipped_snapshots"] = all_skipped
    if verification_results:
        agg["archive_verification"] = {
            "enabled": True,
            "n_verified": sum(1 for v in verification_results if v["verified"]),
            "n_failed": sum(1 for v in verification_results if not v["verified"]),
            "details": verification_results,
        }

    return agg, hygiene_before, hygiene_after


# =============================================================================
# AGGREGATE STATISTICS
# =============================================================================

def _aggregate(per_snapshot: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate statistics from per-snapshot IC results."""
    n_snapshots = len(per_snapshot)
    if n_snapshots == 0:
        return {"error": "No usable snapshots"}

    result: Dict[str, Any] = {
        "n_snapshots": n_snapshots,
        "date_range": [per_snapshot[0]["date"], per_snapshot[-1]["date"]],
        "return_source": "HS793 total return index + price_history.csv fallback",
        "methodology": "investable-universe filtering with dense re-ranking",
        "per_snapshot": [],
    }

    # Clean per-snapshot for output (remove internal fields)
    for snap in per_snapshot:
        clean = {k: v for k, v in snap.items() if not k.startswith("_")}
        result["per_snapshot"].append(clean)

    # Aggregate per horizon
    for horizon_label in HORIZONS:
        ic_key = f"ic_{horizon_label}"
        ic_values = [s[ic_key] for s in per_snapshot if ic_key in s]

        if not ic_values:
            continue

        mean_ic = mean(ic_values)
        std_ic = stdev(ic_values) if len(ic_values) > 1 else 0.0
        t_stat = mean_ic / (std_ic / math.sqrt(len(ic_values))) if std_ic > 0 else float("inf")
        hit_rate = sum(1 for x in ic_values if x > 0) / len(ic_values)
        quality = _classify_ic(Decimal(str(abs(mean_ic)))).value

        # Kendall tau-b aggregate
        kendall_key = f"kendall_{horizon_label}"
        kendall_values = [s[kendall_key] for s in per_snapshot
                          if kendall_key in s and s[kendall_key] is not None]
        if kendall_values:
            mean_kendall = mean(kendall_values)
            std_kendall = stdev(kendall_values) if len(kendall_values) > 1 else 0.0
            t_kendall = mean_kendall / (std_kendall / math.sqrt(len(kendall_values))) if std_kendall > 0 else float("inf")
        else:
            mean_kendall = None
            std_kendall = None
            t_kendall = None

        # Quintile spread aggregate
        qs_key = f"quintile_spread_{horizon_label}"
        qs_spreads = [s[qs_key]["spread_pct"] for s in per_snapshot
                      if qs_key in s and s[qs_key] is not None]
        if qs_spreads:
            mean_qs = mean(qs_spreads)
            std_qs = stdev(qs_spreads) if len(qs_spreads) > 1 else 0.0
            t_qs = mean_qs / (std_qs / math.sqrt(len(qs_spreads))) if std_qs > 0 else float("inf")
            qs_hit = sum(1 for s in qs_spreads if s > 0) / len(qs_spreads)
            worst_idx = min(range(len(qs_spreads)), key=lambda i: qs_spreads[i])
            worst_snap = [s for s in per_snapshot if qs_key in s and s[qs_key] is not None][worst_idx]
        else:
            mean_qs = None
            std_qs = None
            t_qs = None
            qs_hit = None
            worst_snap = None

        result[f"aggregate_{horizon_label}"] = {
            "mean_ic": round(mean_ic, 6),
            "std_ic": round(std_ic, 6),
            "t_stat": round(t_stat, 2),
            "hit_rate": round(hit_rate * 100, 1),
            "quality": quality,
            "n_periods": len(ic_values),
            "mean_kendall": round(mean_kendall, 6) if mean_kendall is not None else None,
            "std_kendall": round(std_kendall, 6) if std_kendall is not None else None,
            "t_stat_kendall": round(t_kendall, 2) if t_kendall is not None else None,
            "quintile_spread": {
                "mean_spread_pct": round(mean_qs, 4) if mean_qs is not None else None,
                "std_spread_pct": round(std_qs, 4) if std_qs is not None else None,
                "t_stat": round(t_qs, 2) if t_qs is not None else None,
                "hit_rate": round(qs_hit * 100, 1) if qs_hit is not None else None,
                "n_periods": len(qs_spreads),
                "worst_spread_pct": round(min(qs_spreads), 4) if qs_spreads else None,
                "worst_date": worst_snap["date"] if worst_snap else None,
            } if qs_spreads else None,
        }

    # --- Residual (XBI-hedged) aggregate stats ---
    for horizon_label in HORIZONS:
        resid_ic_key = f"resid_ic_{horizon_label}"
        resid_ics = [s[resid_ic_key] for s in per_snapshot
                     if resid_ic_key in s and s[resid_ic_key] is not None]
        if not resid_ics:
            continue

        resid_mean = mean(resid_ics)
        resid_std = stdev(resid_ics) if len(resid_ics) > 1 else 0.0
        resid_t = resid_mean / (resid_std / math.sqrt(len(resid_ics))) if resid_std > 0 else float("inf")
        resid_hit = sum(1 for x in resid_ics if x > 0) / len(resid_ics)

        resid_kendall_key = f"resid_kendall_{horizon_label}"
        resid_kendalls = [s[resid_kendall_key] for s in per_snapshot
                          if resid_kendall_key in s and s[resid_kendall_key] is not None]
        rk_mean = mean(resid_kendalls) if resid_kendalls else None
        rk_t = None
        if resid_kendalls and len(resid_kendalls) > 1:
            rk_std = stdev(resid_kendalls)
            rk_t = rk_mean / (rk_std / math.sqrt(len(resid_kendalls))) if rk_std > 0 else float("inf")

        resid_qs_key = f"resid_quintile_spread_{horizon_label}"
        resid_qs_vals = [s[resid_qs_key]["spread_pct"] for s in per_snapshot
                         if resid_qs_key in s and s[resid_qs_key] is not None]
        rqs_mean = mean(resid_qs_vals) if resid_qs_vals else None
        rqs_t = None
        rqs_hit = None
        rqs_worst = None
        rqs_worst_date = None
        if resid_qs_vals:
            rqs_hit = sum(1 for s in resid_qs_vals if s > 0) / len(resid_qs_vals)
            if len(resid_qs_vals) > 1:
                rqs_std = stdev(resid_qs_vals)
                rqs_t = rqs_mean / (rqs_std / math.sqrt(len(resid_qs_vals))) if rqs_std > 0 else float("inf")
            worst_idx = min(range(len(resid_qs_vals)), key=lambda i: resid_qs_vals[i])
            rqs_worst = resid_qs_vals[worst_idx]
            rqs_worst_date = [s for s in per_snapshot if resid_qs_key in s and s[resid_qs_key] is not None][worst_idx]["date"]

        result[f"residual_aggregate_{horizon_label}"] = {
            "mean_ic": round(resid_mean, 6),
            "std_ic": round(resid_std, 6),
            "t_stat": round(resid_t, 2),
            "hit_rate": round(resid_hit * 100, 1),
            "quality": _classify_ic(Decimal(str(abs(resid_mean)))).value,
            "n_periods": len(resid_ics),
            "mean_kendall": round(rk_mean, 6) if rk_mean is not None else None,
            "t_stat_kendall": round(rk_t, 2) if rk_t is not None else None,
            "quintile_spread": {
                "mean_spread_pct": round(rqs_mean, 4) if rqs_mean is not None else None,
                "t_stat": round(rqs_t, 2) if rqs_t is not None else None,
                "hit_rate": round(rqs_hit * 100, 1) if rqs_hit is not None else None,
                "n_periods": len(resid_qs_vals),
                "worst_spread_pct": round(rqs_worst, 4) if rqs_worst is not None else None,
                "worst_date": rqs_worst_date,
            } if resid_qs_vals else None,
        }

    # Top/bottom decile spread (60d)
    spreads: List[float] = []
    for snap in per_snapshot:
        rankings = snap.get("_rankings", {})
        returns_60d = snap.get("_returns_60d", {})
        if not rankings or not returns_60d:
            continue

        common = sorted(
            [t for t in rankings if t in returns_60d],
            key=lambda t: rankings[t]
        )
        if len(common) < 20:
            continue

        decile_size = len(common) // 10
        if decile_size < 2:
            continue

        top_tickers = common[:decile_size]
        bottom_tickers = common[-decile_size:]

        top_ret = mean([returns_60d[t] for t in top_tickers])
        bottom_ret = mean([returns_60d[t] for t in bottom_tickers])
        spreads.append(top_ret - bottom_ret)

    if spreads:
        result["decile_spread_60d"] = {
            "mean_spread": round(mean(spreads) * 100, 2),
            "positive_pct": round(sum(1 for s in spreads if s > 0) / len(spreads) * 100, 1),
            "n_periods": len(spreads),
            "per_period": [round(s * 100, 2) for s in spreads],
        }

    # Aggregate component IC
    all_comp_names: set = set()
    for snap in per_snapshot:
        all_comp_names.update(snap.get("component_ic_20d", {}).keys())

    comp_agg: Dict[str, Dict[str, Any]] = {}
    for comp_name in sorted(all_comp_names):
        comp_ics = [
            snap["component_ic_20d"][comp_name]
            for snap in per_snapshot
            if comp_name in snap.get("component_ic_20d", {})
        ]
        if comp_ics:
            comp_mean = mean(comp_ics)
            comp_agg[comp_name] = {
                "mean_ic": round(comp_mean, 6),
                "quality": _classify_ic(Decimal(str(abs(comp_mean)))).value,
                "n_periods": len(comp_ics),
            }

    if comp_agg:
        result["component_ic_20d"] = comp_agg

    # --- Aggregate IC by regime (RISK_ON vs RISK_OFF) ---
    for horizon_label in HORIZONS:
        ic_key = f"ic_{horizon_label}"
        for regime_label in ["RISK_ON", "RISK_OFF"]:
            regime_ics = [
                s[ic_key] for s in per_snapshot
                if s.get("regime") == regime_label and ic_key in s
            ]
            if regime_ics:
                m = mean(regime_ics)
                result.setdefault("regime_ic", {})[f"{regime_label}_{horizon_label}"] = {
                    "mean_ic": round(m, 6),
                    "quality": _classify_ic(Decimal(str(abs(m)))).value,
                    "n_periods": len(regime_ics),
                    "hit_rate": round(sum(1 for x in regime_ics if x > 0) / len(regime_ics) * 100, 1),
                    "per_period": [round(x, 4) for x in regime_ics],
                }

    # --- Aggregate IC by stage (dev vs commercial) ---
    for horizon_label in HORIZONS:
        stage_key = f"stage_ic_{horizon_label}"
        for stage_label in ["commercial", "dev"]:
            stage_ics = []
            stage_ns = []
            for s in per_snapshot:
                stage_data = s.get(stage_key, {}).get(stage_label, {})
                ic_val = stage_data.get("ic")
                if ic_val is not None:
                    stage_ics.append(ic_val)
                    stage_ns.append(stage_data.get("n", 0))
            if stage_ics:
                m = mean(stage_ics)
                result.setdefault("stage_ic", {})[f"{stage_label}_{horizon_label}"] = {
                    "mean_ic": round(m, 6),
                    "quality": _classify_ic(Decimal(str(abs(m)))).value,
                    "n_periods": len(stage_ics),
                    "avg_n_per_period": round(mean(stage_ns)),
                    "hit_rate": round(sum(1 for x in stage_ics if x > 0) / len(stage_ics) * 100, 1),
                    "per_period": [round(x, 4) for x in stage_ics],
                }

    # --- Aggregate RESIDUAL IC by stage (dev vs commercial) ---
    for horizon_label in HORIZONS:
        resid_stage_key = f"resid_stage_ic_{horizon_label}"
        for stage_label in ["commercial", "dev"]:
            stage_ics = []
            stage_ns = []
            for s in per_snapshot:
                stage_data = s.get(resid_stage_key, {}).get(stage_label, {})
                ic_val = stage_data.get("ic")
                if ic_val is not None:
                    stage_ics.append(ic_val)
                    stage_ns.append(stage_data.get("n", 0))
            if stage_ics:
                m = mean(stage_ics)
                result.setdefault("resid_stage_ic", {})[f"{stage_label}_{horizon_label}"] = {
                    "mean_ic": round(m, 6),
                    "quality": _classify_ic(Decimal(str(abs(m)))).value,
                    "n_periods": len(stage_ics),
                    "avg_n_per_period": round(mean(stage_ns)),
                    "hit_rate": round(sum(1 for x in stage_ics if x > 0) / len(stage_ics) * 100, 1),
                    "per_period": [round(x, 4) for x in stage_ics],
                }

    # --- Failure attribution (top/bottom quintile analysis) ---
    for horizon_label in HORIZONS:
        attrib_key = f"_failure_attrib_{horizon_label}"
        attribs = [s[attrib_key] for s in per_snapshot if attrib_key in s]
        if not attribs:
            continue

        # Aggregate stage mix across snapshots
        agg_top_dev = sum(a["top_quintile"]["stage_mix"].get("dev", 0) for a in attribs)
        agg_top_com = sum(a["top_quintile"]["stage_mix"].get("commercial", 0) for a in attribs)
        agg_bot_dev = sum(a["bottom_quintile"]["stage_mix"].get("dev", 0) for a in attribs)
        agg_bot_com = sum(a["bottom_quintile"]["stage_mix"].get("commercial", 0) for a in attribs)

        top_rets = [a["top_quintile"]["mean_return_pct"] for a in attribs]
        bot_rets = [a["bottom_quintile"]["mean_return_pct"] for a in attribs]

        result.setdefault("failure_attribution", {})[horizon_label] = {
            "n_snapshots": len(attribs),
            "top_quintile": {
                "mean_return_pct": round(mean(top_rets), 4),
                "stage_mix_total": {"dev": agg_top_dev, "commercial": agg_top_com},
                "dev_pct": round(agg_top_dev / (agg_top_dev + agg_top_com) * 100, 1) if (agg_top_dev + agg_top_com) > 0 else 0,
            },
            "bottom_quintile": {
                "mean_return_pct": round(mean(bot_rets), 4),
                "stage_mix_total": {"dev": agg_bot_dev, "commercial": agg_bot_com},
                "dev_pct": round(agg_bot_dev / (agg_bot_dev + agg_bot_com) * 100, 1) if (agg_bot_dev + agg_bot_com) > 0 else 0,
            },
        }

        # Include per-snapshot detail in output (moved from internal _fields)
        per_snap_attribs = []
        for s in per_snapshot:
            a = s.get(attrib_key)
            if a is not None:
                per_snap_attribs.append({
                    "date": s["date"],
                    **a,
                })
        result.setdefault("failure_attribution_detail", {})[horizon_label] = per_snap_attribs

    # --- Aggregate regression results ---
    for horizon_label in HORIZONS:
        reg_key = f"regression_{horizon_label}"
        # Collect all feature names across snapshots
        all_features: set = set()
        for s in per_snapshot:
            reg = s.get(reg_key)
            if reg and "coefficients" in reg:
                all_features.update(k for k in reg["coefficients"] if k != "intercept")

        if not all_features:
            continue

        reg_agg: Dict[str, Any] = {"features": {}}
        for feat in sorted(all_features):
            coefs = []
            tstats = []
            for s in per_snapshot:
                reg = s.get(reg_key)
                if reg and feat in reg.get("coefficients", {}):
                    coefs.append(reg["coefficients"][feat]["coef"])
                    tstats.append(reg["coefficients"][feat]["t_stat"])
            if coefs:
                mean_coef = mean(coefs)
                reg_agg["features"][feat] = {
                    "mean_coef": round(mean_coef, 6),
                    "std_coef": round(stdev(coefs), 6) if len(coefs) > 1 else 0.0,
                    "mean_tstat": round(mean(tstats), 3),
                    "hit_rate_positive": round(sum(1 for c in coefs if c > 0) / len(coefs) * 100, 1),
                    "n_periods": len(coefs),
                    "per_period_coef": [round(c, 6) for c in coefs],
                    "per_period_tstat": [round(t, 3) for t in tstats],
                }

        # Aggregate R²
        r2s = [s[reg_key]["r_squared"] for s in per_snapshot if reg_key in s]
        if r2s:
            reg_agg["mean_r_squared"] = round(mean(r2s), 4)

        # Feature correlations from first available snapshot
        for s in per_snapshot:
            reg = s.get(reg_key)
            if reg and "feature_correlations" in reg:
                reg_agg["feature_correlations"] = reg["feature_correlations"]
                break

        # Regime-split regression: collect coefs by regime
        for feat in sorted(all_features):
            for regime_label in ["RISK_ON", "RISK_OFF"]:
                coefs = []
                for s in per_snapshot:
                    if s.get("regime") != regime_label:
                        continue
                    reg = s.get(reg_key)
                    if reg and feat in reg.get("coefficients", {}):
                        coefs.append(reg["coefficients"][feat]["coef"])
                if coefs:
                    reg_agg["features"][feat][f"mean_coef_{regime_label}"] = round(mean(coefs), 6)
                    reg_agg["features"][feat][f"n_{regime_label}"] = len(coefs)

        # Stage-split regression: aggregate sub-regressions
        for stage_label in ["dev", "commercial"]:
            stage_key_inner = f"stage_{stage_label}"
            stage_feats: Dict[str, List[float]] = {}
            for s in per_snapshot:
                reg = s.get(reg_key)
                if reg and stage_key_inner in reg:
                    sub = reg[stage_key_inner]
                    for feat_name, feat_data in sub.get("coefficients", {}).items():
                        if feat_name == "intercept":
                            continue
                        stage_feats.setdefault(feat_name, []).append(feat_data["coef"])
            if stage_feats:
                stage_agg = {}
                for feat_name, coef_list in stage_feats.items():
                    stage_agg[feat_name] = {
                        "mean_coef": round(mean(coef_list), 6),
                        "n_periods": len(coef_list),
                    }
                reg_agg[f"stage_{stage_label}"] = stage_agg

        result[f"regression_{horizon_label}"] = reg_agg

    # --- Aggregate partial correlations ---
    for horizon_label in HORIZONS:
        pc_key = f"partial_corr_{horizon_label}"
        all_pc_comps: set = set()
        for s in per_snapshot:
            all_pc_comps.update(s.get(pc_key, {}).keys())

        pc_agg: Dict[str, Any] = {}
        for comp_name in sorted(all_pc_comps):
            pearson_vals = []
            spearman_vals = []
            for s in per_snapshot:
                pc = s.get(pc_key, {}).get(comp_name, {})
                p = pc.get("partial_pearson")
                sp = pc.get("partial_spearman")
                if p is not None:
                    pearson_vals.append(p)
                if sp is not None:
                    spearman_vals.append(sp)
            if pearson_vals:
                mp = mean(pearson_vals)
                ms = mean(spearman_vals) if spearman_vals else None
                pc_agg[comp_name] = {
                    "mean_partial_pearson": round(mp, 6),
                    "mean_partial_spearman": round(ms, 6) if ms is not None else None,
                    "hit_rate_positive_pearson": round(
                        sum(1 for x in pearson_vals if x > 0) / len(pearson_vals) * 100, 1),
                    "n_periods": len(pearson_vals),
                    "per_period_pearson": [round(x, 4) for x in pearson_vals],
                    "per_period_spearman": [round(x, 4) for x in spearman_vals],
                }
                # Split by regime
                for regime_label in ["RISK_ON", "RISK_OFF"]:
                    rv = [
                        s.get(pc_key, {}).get(comp_name, {}).get("partial_pearson")
                        for s in per_snapshot
                        if s.get("regime") == regime_label
                        and s.get(pc_key, {}).get(comp_name, {}).get("partial_pearson") is not None
                    ]
                    if rv:
                        pc_agg[comp_name][f"mean_pearson_{regime_label}"] = round(mean(rv), 6)

        if pc_agg:
            result[f"partial_corr_{horizon_label}"] = pc_agg

    # --- Aggregate bucket spreads ---
    for horizon_label in HORIZONS:
        bs_key = f"bucket_spread_{horizon_label}"
        all_bs_comps: set = set()
        for s in per_snapshot:
            all_bs_comps.update(s.get(bs_key, {}).keys())

        bs_agg: Dict[str, Any] = {}
        for comp_name in sorted(all_bs_comps):
            spreads_list = []
            mono_list = []
            for s in per_snapshot:
                bs = s.get(bs_key, {}).get(comp_name, {})
                sp = bs.get("spread_top_minus_bottom_pct")
                mo = bs.get("monotonicity_up")
                if sp is not None:
                    spreads_list.append(sp)
                if mo is not None:
                    mono_list.append(mo)
            if spreads_list:
                ms = mean(spreads_list)
                bs_agg[comp_name] = {
                    "mean_spread_pct": round(ms, 2),
                    "hit_rate_positive": round(
                        sum(1 for x in spreads_list if x > 0) / len(spreads_list) * 100, 1),
                    "mean_monotonicity_up": round(mean(mono_list), 2) if mono_list else None,
                    "n_periods": len(spreads_list),
                    "per_period_spread": [round(x, 2) for x in spreads_list],
                }
                # Stage splits
                for stage_label in ["dev", "commercial"]:
                    stage_spreads = []
                    for s in per_snapshot:
                        bs = s.get(bs_key, {}).get(comp_name, {})
                        st = bs.get(f"stage_{stage_label}", {})
                        sp = st.get("spread_pct")
                        if sp is not None:
                            stage_spreads.append(sp)
                    if stage_spreads:
                        bs_agg[comp_name][f"mean_spread_{stage_label}"] = round(mean(stage_spreads), 2)

        if bs_agg:
            result[f"bucket_spread_{horizon_label}"] = bs_agg

    # --- Aggregate VIF ---
    vif_samples: Dict[str, List[float]] = {}
    for s in per_snapshot:
        for feat, val in s.get("vif", {}).items():
            vif_samples.setdefault(feat, []).append(val)
    if vif_samples:
        result["mean_vif"] = {k: round(mean(v), 2) for k, v in sorted(vif_samples.items())}

    return result


# =============================================================================
# FORMATTED OUTPUT
# =============================================================================

def print_report(results: Dict[str, Any]) -> None:
    """Print formatted backtest report to stdout."""
    print()
    print("=" * 60)
    print("RANK-IC BACKTEST RESULTS")
    print("=" * 60)

    # Handle zero-snapshot case
    if "error" in results:
        print(f"  {results['error']}")
        skipped = results.get("skipped_snapshots", [])
        if skipped:
            print(f"  ({len(skipped)} snapshot(s) skipped — see skipped_snapshots in output JSON)")
        rc = results.get("returns_coverage", {})
        if rc:
            print(f"  Returns data ends: {rc.get('last_return_date', '?')}")
        print("=" * 60)
        return

    print(f"Snapshots:  {results['n_snapshots']} ({results['date_range'][0]} to {results['date_range'][1]})")
    print(f"Method:     {results.get('methodology', 'N/A')}")

    per_snap = results.get("per_snapshot", [])
    avg_n = mean([s.get("n_20d", 0) for s in per_snap]) if per_snap else 0
    print(f"Universe:   ~{int(avg_n)} investable tickers per snapshot")
    print(f"Return src: {results['return_source']}")
    print()

    # Per-snapshot detail
    for horizon_label in HORIZONS:
        # Check if any snapshot has residual data
        has_resid = any(f"resid_ic_{horizon_label}" in s for s in results.get("per_snapshot", []))
        resid_hdr = f"  {'Resid IC':>9s}  {'Resid Qs':>9s}" if has_resid else ""
        resid_bar = f"  {'─' * 9}  {'─' * 9}" if has_resid else ""

        print(f"Per-Snapshot ({horizon_label} forward):")
        print(f"  {'Date':>12s}  {'Spearman':>9s}  {'Kendall':>9s}  {'Q spread':>9s}{resid_hdr}  {'n':>5s}  {'drop':>5s}  {'Quality'}")
        print(f"  {'─' * 12}  {'─' * 9}  {'─' * 9}  {'─' * 9}{resid_bar}  {'─' * 5}  {'─' * 5}  {'─' * 12}")
        for snap in results.get("per_snapshot", []):
            ic = snap.get(f"ic_{horizon_label}", 0)
            kendall = snap.get(f"kendall_{horizon_label}")
            qs = snap.get(f"quintile_spread_{horizon_label}", {})
            qs_spread = qs.get("spread_pct") if qs else None
            n = snap.get(f"n_{horizon_label}", 0)
            quality = snap.get(f"quality_{horizon_label}", "?")
            dropped = snap.get("n_dropped", 0)
            k_str = f"{kendall:+.4f}" if kendall is not None else "    n/a"
            qs_str = f"{qs_spread:+.2f}%" if qs_spread is not None else "     n/a"

            resid_cols = ""
            if has_resid:
                ric = snap.get(f"resid_ic_{horizon_label}")
                rqs = snap.get(f"resid_quintile_spread_{horizon_label}", {})
                rqs_spread = rqs.get("spread_pct") if rqs else None
                ric_str = f"{ric:+.4f}" if ric is not None else "    n/a"
                rqs_str = f"{rqs_spread:+.2f}%" if rqs_spread is not None else "     n/a"
                resid_cols = f"  {ric_str:>9s}  {rqs_str:>9s}"

            print(f"  {snap['date']:>12s}  {ic:+9.4f}  {k_str:>9s}  {qs_str:>9s}{resid_cols}  {n:>5d}  {dropped:>5d}  {quality}")
        print()

    # Aggregate
    print("Aggregate:")
    for horizon_label in HORIZONS:
        agg = results.get(f"aggregate_{horizon_label}")
        if agg:
            # Spearman
            print(
                f"  Spearman ({horizon_label}):  mean={agg['mean_ic']:+.4f}  "
                f"t={agg['t_stat']:.2f}  "
                f"hit={agg['hit_rate']:.0f}%  "
                f"-> {agg['quality']}"
            )
            # Kendall
            mk = agg.get("mean_kendall")
            tk = agg.get("t_stat_kendall")
            if mk is not None:
                line = f"  Kendall  ({horizon_label}):  mean={mk:+.4f}"
                if tk is not None:
                    line += f"  t={tk:.2f}"
                print(line)
            # Quintile spread
            qs = agg.get("quintile_spread")
            if qs and qs.get("mean_spread_pct") is not None:
                print(
                    f"  Q spread ({horizon_label}):  mean={qs['mean_spread_pct']:+.2f}%  "
                    f"t={qs['t_stat']:.2f}  "
                    f"hit={qs['hit_rate']:.0f}%  "
                    f"worst={qs['worst_spread_pct']:+.2f}% ({qs['worst_date']})"
                )

        # Residual aggregate
        ragg = results.get(f"residual_aggregate_{horizon_label}")
        if ragg:
            print(
                f"  Resid IC ({horizon_label}):  mean={ragg['mean_ic']:+.4f}  "
                f"t={ragg['t_stat']:.2f}  "
                f"hit={ragg['hit_rate']:.0f}%  "
                f"-> {ragg['quality']}"
            )
            rmk = ragg.get("mean_kendall")
            if rmk is not None:
                line = f"  Resid τ  ({horizon_label}):  mean={rmk:+.4f}"
                rtk = ragg.get("t_stat_kendall")
                if rtk is not None:
                    line += f"  t={rtk:.2f}"
                print(line)
            rqs = ragg.get("quintile_spread")
            if rqs and rqs.get("mean_spread_pct") is not None:
                print(
                    f"  Resid Qs ({horizon_label}):  mean={rqs['mean_spread_pct']:+.2f}%  "
                    f"t={rqs['t_stat']:.2f}  "
                    f"hit={rqs['hit_rate']:.0f}%  "
                    f"worst={rqs['worst_spread_pct']:+.2f}% ({rqs['worst_date']})"
                )
    print()

    # Decile spread (legacy 60d)
    spread = results.get("decile_spread_60d")
    if spread:
        print("Top/Bottom Decile Spread (60d):")
        print(f"  Mean: {spread['mean_spread']:+.1f}%   Positive: {spread['positive_pct']:.0f}% of periods")
        print(f"  Per-period: {', '.join(f'{s:+.1f}%' for s in spread['per_period'])}")
        print()

    # Component IC
    comp = results.get("component_ic_20d")
    if comp:
        print("Component IC (20d, averaged across snapshots):")
        for name, data in comp.items():
            label = name.replace("_Score", "").replace("_score", "").lower()
            print(f"  {label:20s}: {data['mean_ic']:+.4f}  {data['quality']}  (n_periods={data['n_periods']})")
        print()

    # Regime-stratified IC
    regime_ic = results.get("regime_ic")
    if regime_ic:
        print("IC by Regime (XBI trailing 60d return):")
        for key in sorted(regime_ic.keys()):
            data = regime_ic[key]
            regime_label, horizon = key.rsplit("_", 1)
            per = ", ".join(f"{x:+.4f}" for x in data.get("per_period", []))
            print(f"  {regime_label:10s} {horizon:>3s}: mean={data['mean_ic']:+.4f}  "
                  f"hit={data['hit_rate']:.0f}%  {data['quality']}  "
                  f"(n={data['n_periods']})  [{per}]")
        print()

    # Stage-stratified IC
    stage_ic = results.get("stage_ic")
    if stage_ic:
        print("IC by Stage (dev vs commercial, re-ranked within group):")
        for key in sorted(stage_ic.keys()):
            data = stage_ic[key]
            stage_label, horizon = key.rsplit("_", 1)
            per = ", ".join(f"{x:+.4f}" for x in data.get("per_period", []))
            print(f"  {stage_label:12s} {horizon:>3s}: mean={data['mean_ic']:+.4f}  "
                  f"hit={data['hit_rate']:.0f}%  {data['quality']}  "
                  f"(n={data['n_periods']}, ~{data['avg_n_per_period']}/period)  [{per}]")
        print()

    # Residual stage-stratified IC
    resid_stage_ic = results.get("resid_stage_ic")
    if resid_stage_ic:
        print("Residual IC by Stage (XBI-hedged, re-ranked within group):")
        for key in sorted(resid_stage_ic.keys()):
            data = resid_stage_ic[key]
            stage_label, horizon = key.rsplit("_", 1)
            per = ", ".join(f"{x:+.4f}" for x in data.get("per_period", []))
            print(f"  {stage_label:12s} {horizon:>3s}: mean={data['mean_ic']:+.4f}  "
                  f"hit={data['hit_rate']:.0f}%  {data['quality']}  "
                  f"(n={data['n_periods']}, ~{data['avg_n_per_period']}/period)  [{per}]")
        print()

    # Failure attribution summary
    fa = results.get("failure_attribution")
    if fa:
        print("Failure Attribution (top vs bottom quintile):")
        for horizon_label, data in sorted(fa.items()):
            top = data["top_quintile"]
            bot = data["bottom_quintile"]
            print(f"  {horizon_label}:  top Q1 mean={top['mean_return_pct']:+.2f}%  "
                  f"(dev {top['dev_pct']:.0f}%)   "
                  f"bottom Q5 mean={bot['mean_return_pct']:+.2f}%  "
                  f"(dev {bot['dev_pct']:.0f}%)")
        print()

        # Per-snapshot detail for 20d (the most actionable)
        fa_detail = results.get("failure_attribution_detail", {}).get("20d", [])
        if fa_detail:
            print("Bottom Quintile Detail (20d, worst performers per snapshot):")
            for snap_attrib in fa_detail:
                snap_date = snap_attrib["date"]
                bot = snap_attrib["bottom_quintile"]
                print(f"  --- {snap_date} (Q5 mean={bot['mean_return_pct']:+.2f}%, "
                      f"dev={bot['stage_mix'].get('dev', 0)}, "
                      f"com={bot['stage_mix'].get('commercial', 0)}) ---")
                for t in bot["tickers"][:5]:
                    meta_parts = []
                    for mk in ["catalyst_score", "smart_money_score", "confidence_overall"]:
                        if mk in t:
                            short = mk.replace("_score", "").replace("_overall", "")
                            meta_parts.append(f"{short}={t[mk]:.2f}")
                    meta_str = "  " + ", ".join(meta_parts) if meta_parts else ""
                    print(f"    {t['ticker']:6s}  rank={t['rank']:>3d}  "
                          f"ret={t['return_pct']:+7.2f}%  {t['stage']:>4s}{meta_str}")
            print()

    # Cross-sectional regression
    for horizon_label in HORIZONS:
        reg = results.get(f"regression_{horizon_label}")
        if not reg:
            continue
        print(f"Cross-Sectional OLS ({horizon_label} forward return ~ components + controls):")
        print(f"  Mean R²: {reg.get('mean_r_squared', 0):.4f}")
        print()

        # Feature coefficients table
        feats = reg.get("features", {})
        if feats:
            print(f"  {'Feature':20s}  {'MeanCoef':>10s}  {'StdCoef':>10s}  {'MeanTstat':>10s}  "
                  f"{'Hit%+':>6s}  {'N':>3s}  {'RISK_ON':>10s}  {'RISK_OFF':>10s}")
            print(f"  {'-' * 95}")
            for feat_name, fd in feats.items():
                risk_on = fd.get("mean_coef_RISK_ON")
                risk_off = fd.get("mean_coef_RISK_OFF")
                ro_str = f"{risk_on:+.6f}" if risk_on is not None else "n/a"
                rf_str = f"{risk_off:+.6f}" if risk_off is not None else "n/a"
                print(
                    f"  {feat_name:20s}  {fd['mean_coef']:+10.6f}  {fd['std_coef']:10.6f}  "
                    f"{fd['mean_tstat']:+10.3f}  {fd['hit_rate_positive']:5.0f}%  "
                    f"{fd['n_periods']:>3d}  {ro_str:>10s}  {rf_str:>10s}"
                )
            print()

            # Per-period detail
            print(f"  Per-period coefficients:")
            dates = [s["date"] for s in results.get("per_snapshot", [])]
            for feat_name, fd in feats.items():
                coefs_str = ", ".join(f"{c:+.4f}" for c in fd.get("per_period_coef", []))
                print(f"    {feat_name:20s}: [{coefs_str}]")
            print()

        # Feature correlations
        corrs = reg.get("feature_correlations")
        if corrs:
            feat_list = sorted(corrs.keys())
            # Only print if there are cross-correlations above 0.3
            notable = [(a, b, corrs[a][b]) for a in feat_list for b in feat_list
                       if a < b and abs(corrs[a][b]) > 0.2]
            if notable:
                print(f"  Notable feature correlations (|r| > 0.2):")
                for a, b, r in sorted(notable, key=lambda x: -abs(x[2])):
                    print(f"    {a} <-> {b}: {r:+.3f}")
                print()

        # Stage-split coefficients
        for stage_label in ["dev", "commercial"]:
            stage_data = reg.get(f"stage_{stage_label}")
            if stage_data:
                items = ", ".join(
                    f"{fn}={sd['mean_coef']:+.4f}(n={sd['n_periods']})"
                    for fn, sd in sorted(stage_data.items())
                )
                print(f"  {stage_label:12s} subset: {items}")
        if reg.get("stage_dev") or reg.get("stage_commercial"):
            print()

    # Partial correlation
    for horizon_label in HORIZONS:
        pc = results.get(f"partial_corr_{horizon_label}")
        if not pc:
            continue
        print(f"Partial Correlation ({horizon_label}, both Y and X residualized vs controls):")
        for comp_name, pd_data in pc.items():
            label = comp_name.replace("_Score", "").replace("_score", "").lower()
            ro = pd_data.get("mean_pearson_RISK_ON")
            rf = pd_data.get("mean_pearson_RISK_OFF")
            ro_str = f"  RISK_ON={ro:+.4f}" if ro is not None else ""
            rf_str = f"  RISK_OFF={rf:+.4f}" if rf is not None else ""
            pear = ", ".join(f"{x:+.4f}" for x in pd_data.get("per_period_pearson", []))
            print(f"  {label:20s}: pearson={pd_data['mean_partial_pearson']:+.4f}  "
                  f"spearman={pd_data['mean_partial_spearman']:+.4f}  "
                  f"hit+={pd_data['hit_rate_positive_pearson']:.0f}%"
                  f"{ro_str}{rf_str}")
            print(f"    per-period pearson: [{pear}]")
        print()

    # Bucket spread
    for horizon_label in HORIZONS:
        bs = results.get(f"bucket_spread_{horizon_label}")
        if not bs:
            continue
        print(f"Bucket Spread ({horizon_label}, mean return by component level, top-bottom):")
        for comp_name, bd in bs.items():
            label = comp_name.replace("_Score", "").replace("_score", "").lower()
            per = ", ".join(f"{x:+.1f}%" for x in bd.get("per_period_spread", []))
            mono = bd.get("mean_monotonicity_up")
            mono_str = f"  mono_up={mono:.0%}" if mono is not None else ""
            dev_sp = bd.get("mean_spread_dev")
            com_sp = bd.get("mean_spread_commercial")
            stage_str = ""
            if dev_sp is not None:
                stage_str += f"  dev={dev_sp:+.1f}%"
            if com_sp is not None:
                stage_str += f"  comm={com_sp:+.1f}%"
            print(f"  {label:20s}: mean={bd['mean_spread_pct']:+.1f}%  "
                  f"hit+={bd['hit_rate_positive']:.0f}%{mono_str}{stage_str}")
            print(f"    per-period: [{per}]")
        print()

    # VIF
    vif_data = results.get("mean_vif")
    if vif_data:
        print("Mean VIF (>5 = concerning multicollinearity):")
        for feat, val in vif_data.items():
            label = feat.replace("_Score", "").replace("_score", "").lower()
            flag = " ***" if val > 5 else ""
            print(f"  {label:20s}: {val:.2f}{flag}")
        print()

    # =====================================================================
    # SIDE-BY-SIDE DIAGNOSTIC: IC vs OLS vs Partial Corr vs Bucket Spread
    # =====================================================================
    snapshots = results.get("per_snapshot", [])
    if snapshots and any("regression_20d" in s for s in snapshots):
        print("DIAGNOSTIC: Per-Snapshot IC vs OLS Coef vs Partial Corr (side-by-side)")
        print("-" * 130)
        print(f"  {'Date':>12s} {'Regime':>8s} | {'IC_20d':>7s} {'IC_60d':>7s} | "
              f"{'Fin_IC':>7s} {'Cln_IC':>7s} | "
              f"{'OLS_f20':>8s} {'OLS_c20':>8s} {'OLS_f60':>8s} {'OLS_c60':>8s} | "
              f"{'PC_f20':>7s} {'PC_c20':>7s} {'PC_f60':>7s} {'PC_c60':>7s}")
        print(f"  {'-' * 126}")

        def _fmt(v: Optional[float], w: int = 7) -> str:
            return f"{v:+{w}.4f}" if v is not None else f"{'n/a':>{w}s}"

        for s in snapshots:
            regime = s.get("regime", "?")[:8]
            ic20 = s.get("ic_20d")
            ic60 = s.get("ic_60d")
            fin_ic = s.get("component_ic_20d", {}).get("Financial_Score")
            clin_ic = s.get("component_ic_20d", {}).get("Clinical_Score")

            reg20 = s.get("regression_20d", {}).get("coefficients", {})
            reg60 = s.get("regression_60d", {}).get("coefficients", {})
            ols_f20 = reg20.get("Financial_Score", {}).get("coef")
            ols_c20 = reg20.get("Clinical_Score", {}).get("coef")
            ols_f60 = reg60.get("Financial_Score", {}).get("coef")
            ols_c60 = reg60.get("Clinical_Score", {}).get("coef")

            pc20 = s.get("partial_corr_20d", {})
            pc60 = s.get("partial_corr_60d", {})
            pc_f20 = pc20.get("Financial_Score", {}).get("partial_pearson")
            pc_c20 = pc20.get("Clinical_Score", {}).get("partial_pearson")
            pc_f60 = pc60.get("Financial_Score", {}).get("partial_pearson")
            pc_c60 = pc60.get("Clinical_Score", {}).get("partial_pearson")

            print(
                f"  {s['date']:>12s} {regime:>8s} | "
                f"{_fmt(ic20)} {_fmt(ic60)} | "
                f"{_fmt(fin_ic)} {_fmt(clin_ic)} | "
                f"{_fmt(ols_f20, 8)} {_fmt(ols_c20, 8)} {_fmt(ols_f60, 8)} {_fmt(ols_c60, 8)} | "
                f"{_fmt(pc_f20)} {_fmt(pc_c20)} {_fmt(pc_f60)} {_fmt(pc_c60)}"
            )
        print()

    # Pooled interaction regression
    pooled = results.get("pooled_interaction_regression", {})
    if pooled:
        print("=" * 80)
        print("POOLED INTERACTION REGRESSION (all snapshots, snapshot FEs, within-snap standardized)")
        print("=" * 80)
        for horizon_label, reg in pooled.items():
            n = reg.get("n", 0)
            r2 = reg.get("r_squared", 0)
            n_snaps = reg.get("n_snapshots_pooled", 0)
            print(f"\n  {horizon_label}: N={n} obs across {n_snaps} snapshots, R²={r2:.4f}")
            print()
            print(f"  {'Variable':20s}  {'Coef':>10s}  {'SE':>10s}  {'t-stat':>8s}")
            print(f"  {'-' * 52}")
            for var_name in ["fin", "clin", "log_cash", "dev",
                             "fin_x_dev", "clin_x_dev",
                             "fin_x_riskon", "clin_x_riskon",
                             "intercept"]:
                cd = reg.get("coefficients", {}).get(var_name)
                if cd:
                    sig = "*" if abs(cd["t_stat"]) > 1.96 else " "
                    print(f"  {var_name:20s}  {cd['coef']:+10.6f}  {cd['se']:10.6f}  {cd['t_stat']:+8.3f}{sig}")
            print()

            # Derived effects: 2x2 table
            derived = reg.get("derived_effects", {})
            if derived:
                print(f"  Derived marginal effects (fin/clin by stage × regime):")
                print(f"  {'':20s}  {'RISK_OFF':>12s}  {'RISK_ON':>12s}")
                print(f"  {'-' * 48}")
                for comp in ["fin", "clin"]:
                    for stage in ["commercial", "dev"]:
                        key_off = f"{comp}_{stage}_riskoff"
                        key_on = f"{comp}_{stage}_riskon"
                        v_off = derived.get(key_off)
                        v_on = derived.get(key_on)
                        label = f"{comp}({stage})"
                        v_off_s = f"{v_off:+12.6f}" if v_off is not None else f"{'n/a':>12s}"
                        v_on_s = f"{v_on:+12.6f}" if v_on is not None else f"{'n/a':>12s}"
                        print(f"  {label:20s}  {v_off_s}  {v_on_s}")
                print()

    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Rank-IC backtest from archives or legacy snapshots")
    # Archive mode (default)
    parser.add_argument("--archive-dir", type=Path, default=ARCHIVE_DIR,
                        help=f"Archive directory (default: {ARCHIVE_DIR})")
    parser.add_argument("--archive", type=Path, default=None,
                        help="Single archive tarball to use")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date filter for archives (YYYY-MM-DD, inclusive)")
    parser.add_argument("--end", type=str, default=None,
                        help="End date filter for archives (YYYY-MM-DD, inclusive)")
    parser.add_argument("--max-archives", type=int, default=None,
                        help="Maximum number of archives to use")
    parser.add_argument("--signal", type=str, default="score_rank_pct",
                        choices=["score_rank_pct", "score_z", "composite_score"],
                        help="Signal field for IC computation (default: score_rank_pct)")
    parser.add_argument("--allow-rank-ties", action="store_true",
                        help="Allow duplicate composite_rank values instead of failing")
    # Archive verification
    parser.add_argument("--verify-archives", action="store_true", default=True,
                        dest="verify_archives",
                        help="Verify archive integrity before reading (default: on)")
    parser.add_argument("--no-verify-archives", action="store_false",
                        dest="verify_archives",
                        help="Skip archive integrity verification")
    # Legacy mode
    parser.add_argument("--use-live-snapshots", action="store_true",
                        help="Use legacy snapshot directories instead of archives")
    parser.add_argument("--snapshot-dir", type=Path, default=SNAPSHOT_DIR,
                        help=f"Snapshot directory for legacy mode (default: {SNAPSHOT_DIR})")
    parser.add_argument("--output", type=Path, default=OUTPUT_JSON,
                        help=f"Output JSON path (default: {OUTPUT_JSON})")
    args = parser.parse_args()
    output_json = args.output

    print("Loading HS793 total return index...")
    ms_provider = MorningstarReturnsProvider(RETURNS_JSON)
    print(f"  HS793: {len(ms_provider.get_available_tickers())} tickers")

    print("Loading price_history.csv fallback...")
    csv_provider = CSVReturnsProvider(PRICE_CSV, price_col="close")
    print(f"  CSV:   {len(csv_provider.get_available_tickers())} tickers")

    provider = ChainedReturnsProvider(ms_provider, csv_provider)
    print(f"  Combined: {len(provider.get_available_tickers())} tickers")
    print()

    # Determine source: archives (default) or legacy snapshots
    archives_list = None
    source_label = "snapshots"

    if args.use_live_snapshots:
        print(f"Computing rank-IC across quarterly snapshots (legacy mode)...")
        print(f"  Snapshot dir: {args.snapshot_dir}")
    elif args.archive:
        # Single archive mode
        if not args.archive.exists():
            print(f"ERROR: Archive not found: {args.archive}")
            sys.exit(1)
        m = re.search(r"(\d{4}-\d{2}-\d{2})\.tar\.gz$", args.archive.name)
        if not m:
            print(f"ERROR: Cannot parse date from archive filename: {args.archive.name}")
            sys.exit(1)
        archives_list = [(m.group(1), args.archive)]
        source_label = "archives"
        print(f"Computing rank-IC from single archive: {args.archive}")
    else:
        # Default: discover archives
        if not args.archive_dir.exists():
            print(f"ERROR: Archive directory not found: {args.archive_dir}")
            sys.exit(1)
        archives_list = discover_archives(
            args.archive_dir, start=args.start, end=args.end, max_n=args.max_archives
        )
        if not archives_list:
            print(f"ERROR: No archives found in {args.archive_dir}"
                  + (f" (start={args.start}, end={args.end})" if args.start or args.end else ""))
            sys.exit(1)
        source_label = "archives"
        print(f"Computing rank-IC from {len(archives_list)} archives in {args.archive_dir}")
        if args.start or args.end:
            print(f"  Date filter: {args.start or '*'} to {args.end or '*'}")
        print(f"  Archives: {archives_list[0][0]} to {archives_list[-1][0]}")

    signal_field = args.signal
    signal_defaulted = (signal_field == "score_rank_pct" and "--signal" not in sys.argv)
    print(f"  Signal field: {signal_field}" + (" (default)" if signal_defaulted else ""))
    print()

    results, hygiene_before, hygiene_after = run_backtest(
        provider, ms_provider, csv_provider,
        snapshot_dir=args.snapshot_dir if args.use_live_snapshots else None,
        archives=archives_list,
        signal_field=signal_field,
        allow_rank_ties=args.allow_rank_ties,
        verify_archives=args.verify_archives,
    )

    # Add provenance to results
    results["source"] = source_label
    results["signal_field"] = signal_field
    results["signal_field_defaulted"] = signal_defaulted
    results["verify_archives"] = args.verify_archives
    if archives_list:
        results["archive_dir"] = str(args.archive_dir if not args.archive else args.archive.parent)
        results["archives_discovered"] = [d for d, _ in archives_list]
        # archives_used = discovered minus skipped
        skipped_dates = {sk["snapshot_date"] for sk in results.get("skipped_snapshots", [])}
        results["archives_used"] = [d for d, _ in archives_list if d not in skipped_dates]

    # Print hygiene report (before/after comparison)
    print_hygiene_report(hygiene_before, hygiene_after, provider.get_fallback_stats())

    # Print IC report
    print_report(results)

    # Save JSON
    results["hygiene_before_filtering"] = hygiene_before
    results["hygiene_after_filtering"] = hygiene_after
    results["fallback_stats"] = provider.get_fallback_stats()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Strip full missing ticker lists from JSON (verbose); keep counts
    for h_list in [results["hygiene_before_filtering"], results["hygiene_after_filtering"]]:
        for h in h_list:
            for key in ["returns_20d", "returns_60d"]:
                r = h.get(key, {})
                tickers_list = r.pop("missing_tickers", [])
                r["missing_tickers_count"] = len(tickers_list)
                r["missing_tickers_sample"] = tickers_list[:15]

    # Strip verbose ticker lists from per-snapshot regression results
    for snap in results.get("per_snapshot", []):
        for hlabel in HORIZONS:
            reg = snap.get(f"regression_{hlabel}")
            if reg:
                reg.pop("tickers", None)
                for stage_key in ["stage_dev", "stage_commercial"]:
                    sub = reg.get(stage_key)
                    if sub:
                        sub.pop("tickers", None)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {output_json}")


if __name__ == "__main__":
    main()
