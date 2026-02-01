"""
Fama–MacBeth cross-sectional regressions (pandas + numpy only).

Two-step procedure:
  1. For each date t, regress forward return on cross-sectional z-scored features.
  2. Average betas across dates; compute Newey–West t-stats on the beta time series.

No scipy or statsmodels required.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Cross-sectional OLS (single date)
# ---------------------------------------------------------------------------

def _ols_betas(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    OLS betas via least-squares (handles rank-deficient X).

    X should include an intercept column if desired.
    Returns beta vector of shape (k,).
    """
    try:
        betas, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        betas = np.full(X.shape[1], np.nan)
    return betas


def cross_sectional_regression(
    returns: np.ndarray,
    features: np.ndarray,
    add_intercept: bool = True,
) -> np.ndarray:
    """
    Run one cross-sectional OLS regression.

    Args:
        returns: (n,) array of forward returns
        features: (n, k) array of feature values
        add_intercept: prepend column of ones

    Returns:
        betas: (k,) feature betas (excluding intercept)
               or all NaN if regression fails
    """
    n, k = features.shape
    if n < k + 2:  # Need at least k+2 obs
        return np.full(k, np.nan)

    if add_intercept:
        X = np.column_stack([np.ones(n), features])
        betas = _ols_betas(X, returns)
        return betas[1:]  # Drop intercept
    else:
        return _ols_betas(features, returns)


# ---------------------------------------------------------------------------
# Newey–West HAC standard errors on a time series
# ---------------------------------------------------------------------------

def newey_west_se(series: np.ndarray, lags: int = 4) -> float:
    """
    Newey–West HAC standard error for the mean of a time series.

    Uses Bartlett kernel weights.
    """
    series = series[np.isfinite(series)]
    T = len(series)
    if T < 3:
        return np.nan

    mean = np.mean(series)
    demeaned = series - mean

    # Variance term (lag 0)
    gamma_0 = np.sum(demeaned ** 2) / T

    # Autocovariance terms with Bartlett weights
    nw_sum = gamma_0
    for j in range(1, min(lags, T - 1) + 1):
        weight = 1.0 - j / (lags + 1.0)
        gamma_j = np.sum(demeaned[j:] * demeaned[:-j]) / T
        nw_sum += 2.0 * weight * gamma_j

    # SE of the mean
    nw_var = max(nw_sum / T, 0.0)
    return math.sqrt(nw_var)


# ---------------------------------------------------------------------------
# z-score standardisation (cross-sectional, per date)
# ---------------------------------------------------------------------------

def zscore_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    date_col: str = "rebalance_date",
) -> pd.DataFrame:
    """
    Cross-sectionally z-score features within each date.

    Replaces feature columns in-place (on a copy).
    Columns with zero std get set to 0.
    """
    df = df.copy()
    for col in feature_cols:
        grouped = df.groupby(date_col)[col]
        means = grouped.transform("mean")
        stds = grouped.transform("std", ddof=1)
        stds = stds.replace(0, np.nan)
        df[col] = (df[col] - means) / stds
        df[col] = df[col].fillna(0.0)
    return df


# ---------------------------------------------------------------------------
# Feature pre-processing helpers
# ---------------------------------------------------------------------------

_CONSTANT_EPS = 1e-12


def drop_constant_features(
    slice_df: pd.DataFrame,
    feature_cols: List[str],
) -> List[str]:
    """
    Return subset of feature_cols that have cross-sectional std > eps.

    Removes constant / near-constant columns for a single date slice.
    """
    keep = []
    for col in feature_cols:
        std = slice_df[col].std(ddof=1)
        if np.isfinite(std) and std > _CONSTANT_EPS:
            keep.append(col)
    return keep


def prune_correlated_features(
    slice_df: pd.DataFrame,
    feature_cols: List[str],
    threshold: float = 0.999,
) -> List[str]:
    """
    Drop one of any pair with |corr| > threshold (deterministic: drop later name).

    Greedy scan over sorted column pairs.
    """
    if len(feature_cols) < 2:
        return feature_cols

    X = slice_df[feature_cols].values.astype(np.float64)
    n = X.shape[1]
    # Correlation matrix
    stds = np.std(X, axis=0, ddof=1)
    stds[stds == 0] = 1.0  # prevent div-by-zero (constant cols already filtered)
    X_z = (X - np.mean(X, axis=0)) / stds
    corr = (X_z.T @ X_z) / max(X.shape[0] - 1, 1)

    drop_set: set = set()
    for i in range(n):
        if feature_cols[i] in drop_set:
            continue
        for j in range(i + 1, n):
            if feature_cols[j] in drop_set:
                continue
            if abs(corr[i, j]) > threshold:
                # Drop the later column (alphabetically later since cols are sorted)
                drop_set.add(feature_cols[j])

    return [c for c in feature_cols if c not in drop_set]


# ---------------------------------------------------------------------------
# Full Fama–MacBeth procedure
# ---------------------------------------------------------------------------

def fama_macbeth(
    panel: pd.DataFrame,
    return_col: str,
    feature_cols: List[str],
    date_col: str = "rebalance_date",
    standardize: bool = True,
    nw_lags: int = 4,
    prune_collinear: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run full 2-step Fama–MacBeth.

    Args:
        panel: DataFrame with date_col, return_col, and feature_cols
        return_col: forward return column
        feature_cols: list of feature column names (sorted for determinism)
        date_col: date column
        standardize: z-score features cross-sectionally per date
        nw_lags: Newey–West lags for HAC standard errors
        prune_collinear: drop constant and near-perfectly-correlated features per date

    Returns:
        (summary_df, betas_df)
        summary_df: one row per feature with mean_beta, t_stat, p_value, etc.
        betas_df: one row per (date, feature) with that date's beta
    """
    feature_cols = sorted(feature_cols)

    # Drop rows with missing return
    df = panel[[date_col] + feature_cols + [return_col]].copy()
    df = df.dropna(subset=[return_col])

    # Standardize features
    if standardize:
        df = zscore_features(df, feature_cols, date_col)

    # Step 1: cross-sectional regressions per date
    dates = sorted(df[date_col].unique())
    beta_records: List[Dict[str, Any]] = []
    dropped_features_log: Dict[str, List[str]] = {}

    for dt in dates:
        slice_df = df[df[date_col] == dt].dropna(subset=feature_cols, how="any")
        n_obs = len(slice_df)

        # Determine usable features for this date
        usable = feature_cols
        if prune_collinear:
            usable = drop_constant_features(slice_df, usable)
            usable = prune_correlated_features(slice_df, usable)

        dropped = sorted(set(feature_cols) - set(usable))
        if dropped:
            dropped_features_log[dt] = dropped

        if n_obs < len(usable) + 2 or not usable:
            continue

        y = slice_df[return_col].values.astype(np.float64)
        X = slice_df[usable].values.astype(np.float64)
        betas = cross_sectional_regression(y, X, add_intercept=True)

        if np.any(np.isfinite(betas)):
            rec: Dict[str, Any] = {date_col: dt, "n_obs": n_obs}
            for j, col in enumerate(usable):
                rec[col] = float(betas[j])
            # NaN for features dropped this date
            for col in dropped:
                rec[col] = np.nan
            beta_records.append(rec)

    if not beta_records:
        empty_summary = pd.DataFrame(
            columns=["feature", "mean_beta", "t_stat", "p_value",
                     "n_weeks", "avg_n_stocks"]
        )
        empty_betas = pd.DataFrame(
            columns=[date_col, "n_obs"] + feature_cols
        )
        return empty_summary, empty_betas

    betas_df = pd.DataFrame(beta_records)
    betas_df = betas_df.sort_values(date_col).reset_index(drop=True)

    # Step 2: time-series averages + NW t-stats
    n_weeks = len(betas_df)
    avg_n_stocks = betas_df["n_obs"].mean()
    summary_rows: List[Dict[str, Any]] = []

    for col in feature_cols:
        if col not in betas_df.columns:
            continue
        beta_series = betas_df[col].values
        valid = beta_series[np.isfinite(beta_series)]
        n_valid = len(valid)
        if n_valid < 3:
            continue

        mean_beta = float(np.mean(valid))
        se = newey_west_se(valid, lags=nw_lags)
        if se > 0 and np.isfinite(se):
            t_stat = mean_beta / se
        else:
            t_stat = np.nan

        # Two-sided p-value approximation (normal)
        if np.isfinite(t_stat):
            p_value = _normal_two_sided_p(t_stat)
        else:
            p_value = np.nan

        n_dropped = sum(1 for d in dropped_features_log if col in dropped_features_log[d])
        summary_rows.append({
            "feature": col,
            "mean_beta": round(mean_beta, 6),
            "t_stat": round(t_stat, 4) if np.isfinite(t_stat) else np.nan,
            "p_value": round(p_value, 6) if np.isfinite(p_value) else np.nan,
            "n_weeks": n_weeks,
            "n_valid_weeks": n_valid,
            "n_dropped_dates": n_dropped,
            "avg_n_stocks": round(avg_n_stocks, 1),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("feature").reset_index(drop=True)

    return summary_df, betas_df


# ---------------------------------------------------------------------------
# Normal CDF approximation (no scipy)
# ---------------------------------------------------------------------------

def _normal_cdf(x: float) -> float:
    """Standard normal CDF via Abramowitz & Stegun approximation."""
    # Rational approximation (Abramowitz & Stegun 26.2.17)
    if x < -8:
        return 0.0
    if x > 8:
        return 1.0
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x_abs = abs(x) / math.sqrt(2)
    t = 1.0 / (1.0 + p * x_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x_abs * x_abs)
    return 0.5 * (1.0 + sign * y)


def _normal_two_sided_p(t_stat: float) -> float:
    """Two-sided p-value from normal approximation."""
    return 2.0 * (1.0 - _normal_cdf(abs(t_stat)))


# ---------------------------------------------------------------------------
# Convenience: write outputs
# ---------------------------------------------------------------------------

def write_fmb_outputs(
    summary_df: pd.DataFrame,
    betas_df: pd.DataFrame,
    outdir: Path,
    return_col: str,
    feature_cols: List[str],
    nw_lags: int,
    standardize: bool,
) -> Dict[str, str]:
    """Write FMB summary CSV + metadata JSON. Returns dict of output paths."""
    outdir.mkdir(parents=True, exist_ok=True)

    # Summary CSV
    summary_path = outdir / "fmb_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Betas CSV (gzipped to save space)
    betas_path = outdir / "fmb_betas_by_week.csv.gz"
    betas_df.to_csv(betas_path, index=False, compression="gzip")

    # Metadata
    content_hash = hashlib.sha256(
        summary_df.to_csv(index=False).encode()
    ).hexdigest()[:16]
    metadata = {
        "return_col": return_col,
        "feature_cols": sorted(feature_cols),
        "nw_lags": nw_lags,
        "standardize": standardize,
        "n_weeks": int(len(betas_df)),
        "n_features": len(feature_cols),
        "content_hash": content_hash,
    }
    meta_path = outdir / "fmb_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "summary": str(summary_path),
        "betas": str(betas_path),
        "metadata": str(meta_path),
    }
