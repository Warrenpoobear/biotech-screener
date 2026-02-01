"""
Milestone 1 IC Metrics — Spearman Rank IC + Summary

Deterministic, stdlib-first. Uses scipy.stats.spearmanr if available,
otherwise falls back to rank + Pearson implementation.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Spearman rank IC
# ---------------------------------------------------------------------------

def _rank_array(values: Sequence[float]) -> List[float]:
    """Assign average ranks (1-based) to values, handling ties."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and values[indexed[j + 1]] == values[indexed[j]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation coefficient (no deps)."""
    n = len(x)
    if n < 3:
        return float("nan")
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    syy = sum((yi - my) ** 2 for yi in y)
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    denom = math.sqrt(sxx * syy)
    if denom == 0:
        return float("nan")
    return sxy / denom


def spearman_rank_ic(
    scores: Sequence[float],
    returns: Sequence[float],
) -> float:
    """
    Compute Spearman rank IC between score and return arrays.

    Returns NaN if fewer than 3 valid pairs.
    """
    # Filter NaN pairs
    pairs = [
        (s, r)
        for s, r in zip(scores, returns)
        if math.isfinite(s) and math.isfinite(r)
    ]
    if len(pairs) < 3:
        return float("nan")
    sc, ret = zip(*pairs)

    # Try scipy first (much faster for large n)
    try:
        from scipy.stats import spearmanr  # type: ignore
        corr, _ = spearmanr(sc, ret)
        return float(corr)
    except ImportError:
        pass

    return _pearson(_rank_array(list(sc)), _rank_array(list(ret)))


# ---------------------------------------------------------------------------
# IC time-series builder
# ---------------------------------------------------------------------------

def compute_ic_timeseries(
    panel_rows: List[Dict[str, Any]],
    score_cols: List[str],
    return_cols: List[str],
    date_col: str = "rebalance_date",
) -> List[Dict[str, Any]]:
    """
    For each (date, score_col, return_col) triple, compute Spearman IC.

    Returns list of dicts:
        {date_col, score_col, return_col, ic, n_obs}
    """
    # Group rows by date (deterministic order)
    from collections import defaultdict
    by_date: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in panel_rows:
        by_date[row[date_col]].append(row)

    results: List[Dict[str, Any]] = []
    for dt in sorted(by_date):
        rows = by_date[dt]
        for scol in sorted(score_cols):
            for rcol in sorted(return_cols):
                scores = []
                rets = []
                for r in rows:
                    sv = r.get(scol)
                    rv = r.get(rcol)
                    if sv is not None and rv is not None:
                        try:
                            sf = float(sv)
                            rf = float(rv)
                            if math.isfinite(sf) and math.isfinite(rf):
                                scores.append(sf)
                                rets.append(rf)
                        except (TypeError, ValueError):
                            continue
                ic_val = spearman_rank_ic(scores, rets)
                results.append({
                    date_col: dt,
                    "score_col": scol,
                    "return_col": rcol,
                    "ic": ic_val,
                    "n_obs": len(scores),
                })
    return results


# ---------------------------------------------------------------------------
# IC summary
# ---------------------------------------------------------------------------

def summarize_ic(
    ic_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Summarise IC time-series into one row per (score_col, return_col).

    Returns list of dicts:
        {score_col, return_col, mean_ic, hit_rate, n_dates, median_ic}
    """
    from collections import defaultdict
    groups: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in ic_rows:
        key = (row["score_col"], row["return_col"])
        ic = row["ic"]
        if math.isfinite(ic):
            groups[key].append(ic)

    summaries: List[Dict[str, Any]] = []
    for (scol, rcol) in sorted(groups):
        vals = groups[(scol, rcol)]
        n = len(vals)
        if n == 0:
            continue
        mean_ic = sum(vals) / n
        hit_rate = sum(1 for v in vals if v > 0) / n
        sorted_vals = sorted(vals)
        if n % 2 == 1:
            median_ic = sorted_vals[n // 2]
        else:
            median_ic = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
        summaries.append({
            "score_col": scol,
            "return_col": rcol,
            "mean_ic": round(mean_ic, 6),
            "median_ic": round(median_ic, 6),
            "hit_rate": round(hit_rate, 4),
            "n_dates": n,
        })
    return summaries
