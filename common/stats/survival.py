"""Survival / hazard modeling scaffold for biotech event timing.

Provides Kaplan-Meier estimation and Cox proportional hazards for
modeling catalyst timing, trial execution changes, and event arrival.

This is a research scaffold — not production-ready.

Usage:
    from common.stats.survival import kaplan_meier, cox_ph_simple

    km = kaplan_meier(durations, events)
    cox = cox_ph_simple(durations, events, covariates)
"""

from __future__ import annotations

from typing import Any

import numpy as np


def kaplan_meier(
    durations: np.ndarray | list[float],
    events: np.ndarray | list[int],
    label: str = "overall",
) -> dict[str, Any]:
    """Kaplan-Meier survival estimate.

    Args:
        durations: time-to-event or censoring time
        events: 1 = event observed, 0 = censored
        label: name for this group

    Returns:
        dict with survival table, median survival, summary
    """
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)
    n = len(durations)

    if n == 0:
        return {"error": "empty data"}

    # Sort by duration
    order = np.argsort(durations)
    t = durations[order]
    e = events[order]

    # Unique event times
    unique_times = np.unique(t[e == 1])
    at_risk = n
    survival = 1.0

    table: list = []
    for time in unique_times:
        # Count events and censorings before this time
        n_events = int(np.sum((t == time) & (e == 1)))
        n_censored = int(np.sum((t < time) & (e == 0) & (t >= (table[-1]["time"] if table else 0))))
        at_risk -= n_censored
        if at_risk <= 0:
            break
        survival *= 1 - n_events / at_risk
        table.append(
            {
                "time": float(time),
                "n_events": n_events,
                "at_risk": at_risk,
                "survival": _round(survival),
            }
        )
        at_risk -= n_events

    # Median survival
    median = None
    for row in table:
        if row["survival"] <= 0.5:
            median = row["time"]
            break

    return {
        "label": label,
        "n_obs": n,
        "n_events": int(np.sum(events)),
        "n_censored": n - int(np.sum(events)),
        "median_survival": _round(median),
        "survival_table": table,
    }


def cox_ph_simple(
    durations: np.ndarray | list[float],
    events: np.ndarray | list[int],
    covariates: np.ndarray,
    feature_names: list[str] | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """Simple Cox proportional hazards via Newton-Raphson.

    Minimal implementation for research scaffold. For production use,
    consider lifelines or statsmodels.

    Args:
        durations: time-to-event array
        events: event indicator (1=event, 0=censored)
        covariates: (n, p) design matrix
        feature_names: names for covariates
        max_iter: Newton-Raphson iterations
        tol: convergence tolerance

    Returns:
        dict with coefficients, hazard ratios, standard errors
    """
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)
    X = np.asarray(covariates, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n, p = X.shape
    names = feature_names or [f"x{i}" for i in range(p)]

    # Sort by duration (descending for risk set computation)
    order = np.argsort(-durations)
    e = events[order]
    X_sorted = X[order]

    # Newton-Raphson for partial likelihood
    beta = np.zeros(p)

    for iteration in range(max_iter):
        # Compute risk set quantities
        exp_xb = np.exp(X_sorted @ beta)
        gradient = np.zeros(p)
        hessian = np.zeros((p, p))

        risk_sum = 0.0
        risk_x_sum = np.zeros(p)
        risk_xx_sum = np.zeros((p, p))

        # Process from longest duration to shortest (risk sets grow)
        for i in range(n - 1, -1, -1):
            risk_sum += exp_xb[i]
            risk_x_sum += X_sorted[i] * exp_xb[i]
            risk_xx_sum += np.outer(X_sorted[i], X_sorted[i]) * exp_xb[i]

            if e[i] == 1:
                gradient += X_sorted[i] - risk_x_sum / risk_sum
                w = risk_x_sum / risk_sum
                hessian -= risk_xx_sum / risk_sum - np.outer(w, w)

        # Newton step
        try:
            step = np.linalg.solve(-hessian, gradient)
        except np.linalg.LinAlgError:
            return {"error": "singular Hessian", "iteration": iteration}

        beta += step
        if np.max(np.abs(step)) < tol:
            break

    # Standard errors from inverse Hessian
    try:
        var_beta = np.linalg.inv(-hessian)
        se = np.sqrt(np.diag(var_beta))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)

    z_stats = np.where(se > 1e-12, beta / se, np.nan)
    hazard_ratios = np.exp(beta)

    results = {}
    for i in range(p):
        results[names[i]] = {
            "coefficient": _round(float(beta[i])),
            "hazard_ratio": _round(float(hazard_ratios[i])),
            "std_error": _round(float(se[i])),
            "z_stat": _round(float(z_stats[i])),
            "significant": bool(abs(z_stats[i]) >= 1.96) if not np.isnan(z_stats[i]) else False,
        }

    # Concordance index (C-index)
    concordant = 0
    discordant = 0
    tied = 0
    risk_scores = X @ beta
    for i in range(n):
        if events[i] == 0:
            continue
        for j in range(n):
            if durations[j] <= durations[i]:
                continue
            if risk_scores[i] > risk_scores[j]:
                concordant += 1
            elif risk_scores[i] < risk_scores[j]:
                discordant += 1
            else:
                tied += 1

    total_pairs = concordant + discordant + tied
    c_index = (concordant + 0.5 * tied) / total_pairs if total_pairs > 0 else 0.5

    return {
        "n_obs": n,
        "n_events": int(np.sum(events)),
        "n_features": p,
        "converged": iteration < max_iter - 1,
        "iterations": iteration + 1,
        "features": results,
        "c_index": _round(c_index),
    }


def stratified_kaplan_meier(
    durations: np.ndarray | list[float],
    events: np.ndarray | list[int],
    groups: np.ndarray | list[str],
) -> dict[str, Any]:
    """Kaplan-Meier stratified by group variable.

    Args:
        durations, events: as in kaplan_meier
        groups: group labels for each observation

    Returns:
        dict with per-group KM estimates and log-rank test
    """
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)
    groups = np.asarray(groups)

    unique_groups = np.unique(groups)
    km_results = {}
    for g in unique_groups:
        mask = groups == g
        km = kaplan_meier(durations[mask], events[mask], label=str(g))
        km_results[str(g)] = km

    # Simple log-rank test (two-group case)
    if len(unique_groups) == 2:
        g1, g2 = unique_groups
        m1, m2 = groups == g1, groups == g2
        lr = _log_rank_test(
            durations[m1],
            events[m1],
            durations[m2],
            events[m2],
        )
    else:
        lr = {"note": "log-rank only for 2 groups"}

    return {
        "n_groups": len(unique_groups),
        "groups": km_results,
        "log_rank": lr,
    }


def _log_rank_test(dur1, ev1, dur2, ev2):
    """Two-sample log-rank test."""
    all_dur = np.concatenate([dur1, dur2])
    all_ev = np.concatenate([ev1, ev2])
    all_group = np.concatenate([np.zeros(len(dur1)), np.ones(len(dur2))])

    order = np.argsort(all_dur)
    all_dur = all_dur[order]
    all_ev = all_ev[order]
    all_group = all_group[order]

    unique_times = np.unique(all_dur[all_ev == 1])

    O1 = 0.0  # observed events in group 1
    E1 = 0.0  # expected events in group 1
    V = 0.0  # variance

    for t in unique_times:
        at_risk_1 = np.sum((all_dur >= t) & (all_group == 0))
        at_risk_2 = np.sum((all_dur >= t) & (all_group == 1))
        at_risk = at_risk_1 + at_risk_2
        d = np.sum((all_dur == t) & (all_ev == 1))
        d1 = np.sum((all_dur == t) & (all_ev == 1) & (all_group == 0))

        if at_risk <= 1:
            continue

        e1 = d * at_risk_1 / at_risk
        O1 += d1
        E1 += e1
        V += e1 * (1 - at_risk_1 / at_risk) * (at_risk - d) / (at_risk - 1)

    if V <= 0:
        return {"chi2": None, "p_value": None}

    chi2 = (O1 - E1) ** 2 / V

    from scipy.stats import chi2 as chi2_dist

    p_value = 1 - chi2_dist.cdf(chi2, df=1)

    return {
        "chi2": _round(float(chi2)),
        "p_value": _round(float(p_value)),
        "significant": p_value < 0.05,
    }


def _round(v, d=4):
    if v is None:
        return None
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    return round(v, d)
