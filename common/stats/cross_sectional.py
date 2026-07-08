"""Fama-MacBeth cross-sectional regression framework.

Provides monthly cross-sectional OLS, time-series aggregation of coefficients,
and Newey-West standard errors for robust inference.

Usage:
    from common.stats.cross_sectional import fama_macbeth, ols_regression

    # Run Fama-MacBeth across monthly snapshots
    result = fama_macbeth(
        snapshots,  # {date: [{ticker, fwd_ret, signal_1, signal_2, ...}, ...]}
        y_col="fwd_excess_xbi_63d",
        x_cols=["coinvest_score_z", "inst_delta_z", "new_signal"],
        nw_lags=3,
    )
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def ols_regression(
    y: np.ndarray,
    X: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """Ordinary least squares regression.

    Args:
        y: (n,) response vector
        X: (n, k) design matrix (intercept should be included if desired)
        feature_names: optional names for columns of X

    Returns:
        dict with coefficients, residuals, r_squared, std_errors, t_stats
    """
    n, k = X.shape
    if n <= k:
        return {"error": f"n={n} <= k={k}, underdetermined"}

    # OLS: beta = (X'X)^-1 X'y
    try:
        XtX = X.T @ X
        Xty = X.T @ y
        beta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        return {"error": "singular matrix"}

    residuals: Any = y - X @ beta
    sse = float(residuals @ residuals)
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - sse / sst if sst > 1e-12 else 0.0

    # Standard errors
    dof = n - k
    if dof <= 0:
        return {"error": "no degrees of freedom"}
    sigma2 = sse / dof
    try:
        var_beta = sigma2 * np.linalg.inv(XtX)
        std_errors = np.sqrt(np.diag(var_beta))
    except np.linalg.LinAlgError:
        std_errors = np.full(k, np.nan)

    t_stats = np.where(std_errors > 1e-12, beta / std_errors, np.nan)

    names = feature_names or [f"x{i}" for i in range(k)]

    return {
        "coefficients": {names[i]: float(beta[i]) for i in range(k)},
        "std_errors": {names[i]: float(std_errors[i]) for i in range(k)},
        "t_stats": {names[i]: float(t_stats[i]) for i in range(k)},
        "r_squared": r_squared,
        "n_obs": n,
        "n_features": k,
        "residuals": residuals,
    }


def newey_west_se(coef_series: np.ndarray, lags: int = 3) -> float:
    """Newey-West standard error for a time series of coefficient estimates.

    Accounts for autocorrelation up to `lags` periods.

    Args:
        coef_series: (T,) array of monthly coefficient estimates
        lags: number of autocorrelation lags to include

    Returns:
        Newey-West standard error of the mean
    """
    T = len(coef_series)
    if T < 3:
        return float(np.std(coef_series, ddof=1) / np.sqrt(T)) if T >= 2 else np.nan

    mean = np.mean(coef_series)
    demeaned = coef_series - mean

    # Variance component
    gamma_0 = float(np.mean(demeaned**2))

    # Autocovariance components with Bartlett kernel
    nw_sum = 0.0
    for lag in range(1, min(lags + 1, T)):
        weight = 1.0 - lag / (lags + 1)
        gamma_lag = float(np.mean(demeaned[lag:] * demeaned[:-lag]))
        nw_sum += 2 * weight * gamma_lag

    nw_var = gamma_0 + nw_sum
    return float(np.sqrt(max(0, nw_var) / T))


def fama_macbeth(
    snapshots: dict[str, list[dict]],
    y_col: str,
    x_cols: list[str],
    add_intercept: bool = True,
    nw_lags: int = 3,
    min_obs: int = 20,
    eligible_col: str | None = "eligible",
    zscore_x: bool = True,
) -> dict[str, Any]:
    """Run Fama-MacBeth cross-sectional regressions.

    For each snapshot date, runs an OLS regression of forward returns on
    candidate signals. Collects monthly coefficients and computes time-series
    statistics with Newey-West standard errors.

    Args:
        snapshots: {date: [row_dicts]} from research panel
        y_col: forward return column
        x_cols: signal columns to include as regressors
        add_intercept: whether to prepend an intercept column
        nw_lags: Newey-West lag count
        min_obs: minimum observations per snapshot
        eligible_col: if set, filter to rows where this column == 1.0
        zscore_x: if True, z-score each X column per snapshot

    Returns:
        dict with per-signal coefficient stats, monthly history, verdicts
    """
    feature_names = (["intercept"] if add_intercept else []) + list(x_cols)

    monthly_coefs: dict[str, list] = {name: [] for name in feature_names}
    monthly_dates = []
    monthly_n_obs = []
    monthly_r2 = []

    for snap_date in sorted(snapshots.keys()):
        rows = snapshots[snap_date]

        # Filter eligible
        if eligible_col:
            rows = [r for r in rows if _safe_float(r.get(eligible_col)) == 1.0]

        # Extract valid observations
        valid = []
        for r in rows:
            y_val = _safe_float(r.get(y_col))
            if y_val is None or (isinstance(y_val, float) and math.isnan(y_val)):
                continue
            x_vals = []
            skip = False
            for col in x_cols:
                v = _safe_float(r.get(col))
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    skip = True
                    break
                x_vals.append(v)
            if skip:
                continue
            valid.append((y_val, x_vals))

        if len(valid) < min_obs:
            continue

        y = np.array([v[0] for v in valid])
        X_raw = np.array([v[1] for v in valid])

        # Z-score X per snapshot
        if zscore_x:
            means = X_raw.mean(axis=0)
            stds = X_raw.std(axis=0)
            stds[stds < 1e-9] = 1.0
            X_raw = (X_raw - means) / stds

        # Add intercept
        if add_intercept:
            intercept = np.ones((len(valid), 1))
            X = np.hstack([intercept, X_raw])
        else:
            X = X_raw

        result = ols_regression(y, X, feature_names)
        if "error" in result:
            continue

        monthly_dates.append(snap_date)
        monthly_n_obs.append(result["n_obs"])
        monthly_r2.append(result["r_squared"])
        for name in feature_names:
            monthly_coefs[name].append(result["coefficients"][name])

    if not monthly_dates:
        return {"error": "no valid snapshots", "x_cols": x_cols, "y_col": y_col}

    # Time-series aggregation with Newey-West
    signal_results = {}
    for name in feature_names:
        coefs = np.array(monthly_coefs[name])
        T = len(coefs)
        mean_coef = float(np.mean(coefs))
        nw_se = newey_west_se(coefs, lags=nw_lags)
        naive_se = float(np.std(coefs, ddof=1) / np.sqrt(T)) if T >= 2 else np.nan

        nw_t = mean_coef / nw_se if nw_se > 1e-12 else np.nan
        naive_t = mean_coef / naive_se if naive_se > 1e-12 else np.nan

        # Two-sided p-value from NW t-stat (normal approximation)
        if not math.isnan(nw_t):
            from scipy.stats import norm

            p_value = float(2 * (1 - norm.cdf(abs(nw_t))))
        else:
            p_value = np.nan

        signal_results[name] = {
            "mean_coefficient": _round(mean_coef),
            "newey_west_se": _round(nw_se),
            "newey_west_t": _round(nw_t),
            "naive_se": _round(naive_se),
            "naive_t": _round(naive_t),
            "p_value": _round(p_value),
            "n_months": T,
            "monthly_coefficients": [_round(c) for c in coefs],
            "survives_controls": abs(nw_t) >= 1.96 if not math.isnan(nw_t) else False,
        }

    return {
        "y_col": y_col,
        "x_cols": x_cols,
        "n_snapshots": len(monthly_dates),
        "dates": monthly_dates,
        "mean_n_obs": _round(float(np.mean(monthly_n_obs))),
        "mean_r_squared": _round(float(np.mean(monthly_r2))),
        "signals": signal_results,
    }


def run_incremental_test(
    snapshots: dict[str, list[dict]],
    candidate_signal: str,
    control_signals: list[str],
    y_col: str = "fwd_excess_xbi_63d",
    nw_lags: int = 3,
) -> dict[str, Any]:
    """Test whether a candidate signal has incremental value beyond controls.

    Runs three models:
    1. Candidate only (univariate)
    2. Controls only
    3. Candidate + controls (incremental test)

    Returns comparison of all three.
    """
    univariate = fama_macbeth(
        snapshots,
        y_col,
        [candidate_signal],
        nw_lags=nw_lags,
    )
    controls_only = fama_macbeth(
        snapshots,
        y_col,
        control_signals,
        nw_lags=nw_lags,
    )
    full = fama_macbeth(
        snapshots,
        y_col,
        control_signals + [candidate_signal],
        nw_lags=nw_lags,
    )

    # Extract candidate signal stats from each model
    uni_stats = univariate.get("signals", {}).get(candidate_signal, {}) if "error" not in univariate else {}
    full_stats = full.get("signals", {}).get(candidate_signal, {}) if "error" not in full else {}

    return {
        "candidate": candidate_signal,
        "controls": control_signals,
        "y_col": y_col,
        "univariate": {
            "coefficient": uni_stats.get("mean_coefficient"),
            "nw_t": uni_stats.get("newey_west_t"),
            "p_value": uni_stats.get("p_value"),
            "n_months": uni_stats.get("n_months"),
        },
        "controls_only": {
            "r_squared": controls_only.get("mean_r_squared"),
            "n_months": controls_only.get("n_snapshots"),
        },
        "incremental": {
            "coefficient": full_stats.get("mean_coefficient"),
            "nw_t": full_stats.get("newey_west_t"),
            "p_value": full_stats.get("p_value"),
            "survives_controls": full_stats.get("survives_controls", False),
            "r_squared": full.get("mean_r_squared"),
            "n_months": full_stats.get("n_months"),
        },
        "verdict": ("INCREMENTAL" if full_stats.get("survives_controls", False) else "NOT_INCREMENTAL"),
    }


def _safe_float(v):
    if v is None or v == "":
        return None
    try:
        f = float(v)
        return f if not math.isnan(f) else None
    except (ValueError, TypeError):
        return None


def _round(v, d=4):
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return round(v, d)
