#!/usr/bin/env python3
"""Forward-return evaluation across snapshot dates.

Walk-forward evaluation: Spearman IC, top-K portfolio gross/net return,
turnover, benchmark-relative excess, long-short decile spread,
beta-hedged portfolio return, and skip tracking.

Strict PIT enforcement via metadata.json.

Outputs: summary.json, summary.md, by_date.csv, skips.json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
import sys
from dataclasses import asdict, dataclass, field, replace as dc_replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.ranking_utils import backfill_columns, safe_float as _ranking_safe_float
from decision_engine import DecisionRuleset, compute_actionable_sort_key

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_HORIZONS = [5, 20, 63]
DEFAULT_TOP_K = 20
DEFAULT_COST_BPS = 30
DEFAULT_MIN_PRICE_COVERAGE = 0.50

# Map component score columns to their signal-presence flag columns.
# When the flag is "False" or absent, the ticker is excluded from component IC.
SIGNAL_FLAG_MAP = {
    "coinvest_score_z": "has_coinvest_signal",
    "inst_delta_z": "has_inst_delta",
    "catalyst_decay_w": "has_catalyst_signal",
}


# ---------------------------------------------------------------------------
# Logistic decay (copied from decision_engine.py:435-440 for eval-time use)
# ---------------------------------------------------------------------------

def _logistic_decay(days: float, midpoint: float = 150.0, scale: float = 30.0) -> float:
    """Continuous proximity weight: 1.0 (imminent) -> 0.0 (distant). Midpoint -> 0.5."""
    if days is None or days < 0:
        return 0.0
    from math import exp
    return 1.0 / (1.0 + exp((days - midpoint) / max(scale, 1e-6)))


# ---------------------------------------------------------------------------
# Price loader  (lightweight, no pandas dependency)
# ---------------------------------------------------------------------------

def load_price_series(csv_path: Path) -> Dict[str, Dict[str, float]]:
    """Load price_history.csv -> {ticker: {date_str: close}}."""
    prices: Dict[str, Dict[str, float]] = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = (row.get("ticker") or "").strip().upper()
            close_str = (row.get("close") or "").strip()
            date_str = (row.get("date") or "").strip()
            if not ticker or not close_str or not date_str:
                continue
            try:
                close = float(close_str)
            except (ValueError, TypeError):
                continue
            prices.setdefault(ticker, {})[date_str] = close
    return prices


def _trading_days_after(all_dates: List[str], start: str, n: int) -> Optional[str]:
    """Return the trading date *n* trading days after *start*, or None."""
    try:
        idx = all_dates.index(start)
    except ValueError:
        return None
    target = idx + n
    if target < len(all_dates):
        return all_dates[target]
    return None


def resolve_trade_date(
    sorted_dates: List[str],
    snap_date: str,
    anchor_mode: str,
) -> Optional[str]:
    """Map a calendar as-of date to an actual trading date.

    Args:
        sorted_dates: All trading dates in price history, sorted ascending.
        snap_date: Calendar date from the snapshot (may be weekend/holiday).
        anchor_mode: One of "exact", "next_trading_day", "prev_trading_day".

    Returns:
        The resolved trading date, or None if not found.
    """
    if anchor_mode == "exact":
        return snap_date if snap_date in sorted_dates else None

    if anchor_mode == "next_trading_day":
        # First date strictly AFTER snap_date (PIT-safe: trade after info date)
        for d in sorted_dates:
            if d > snap_date:
                return d
        return None

    if anchor_mode == "prev_trading_day":
        # Last date ON or BEFORE snap_date
        result = None
        for d in sorted_dates:
            if d <= snap_date:
                result = d
            else:
                break
        return result

    return None


def compute_forward_return(
    prices: Dict[str, float],
    sorted_dates: List[str],
    snap_date: str,
    horizon: int,
) -> Optional[float]:
    """Simple forward return = P(t+h)/P(t) - 1."""
    p0 = prices.get(snap_date)
    if p0 is None or p0 <= 0:
        return None
    end_date = _trading_days_after(sorted_dates, snap_date, horizon)
    if end_date is None:
        return None
    p1 = prices.get(end_date)
    if p1 is None or p1 <= 0:
        return None
    return p1 / p0 - 1.0


# ---------------------------------------------------------------------------
# Spearman IC (manual, no scipy)
# ---------------------------------------------------------------------------

def _avg_ranks(values: List[float]) -> List[float]:
    """Average-rank with tie handling (1-based)."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and values[indexed[j + 1]] == values[indexed[j]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg
        i = j + 1
    return ranks


def spearman_ic(signal: List[float], returns: List[float]) -> Optional[float]:
    """Rank correlation (Pearson of ranks). Returns None if n < 3."""
    n = len(signal)
    if n < 3:
        return None
    rx = _avg_ranks(signal)
    ry = _avg_ranks(returns)
    mx = statistics.mean(rx)
    my = statistics.mean(ry)
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    sx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    sy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if sx == 0.0 or sy == 0.0:
        return None
    return cov / (sx * sy)


# ---------------------------------------------------------------------------
# Portfolio return + turnover
# ---------------------------------------------------------------------------

def top_k_portfolio_return(
    tickers_ranked: List[str],
    fwd_rets: Dict[str, float],
    k: int,
) -> Tuple[Optional[float], int]:
    """Equal-weight top-K gross return. Returns (mean_return, n_held)."""
    held = [t for t in tickers_ranked[:k] if t in fwd_rets]
    if not held:
        return None, 0
    ret = statistics.mean(fwd_rets[t] for t in held)
    return ret, len(held)


def bottom_k_portfolio_return(
    tickers_ranked: List[str],
    fwd_rets: Dict[str, float],
    k: int,
) -> Tuple[Optional[float], int]:
    """Equal-weight bottom-K gross return. Returns (mean_return, n_held)."""
    held = [t for t in tickers_ranked[-k:] if t in fwd_rets]
    if not held:
        return None, 0
    ret = statistics.mean(fwd_rets[t] for t in held)
    return ret, len(held)


def compute_turnover(prev_set: List[str], curr_set: List[str]) -> float:
    """Turnover = 0.5 * |symmetric difference| / max(len(prev), len(curr), 1)."""
    if not prev_set and not curr_set:
        return 0.0
    s_prev = set(prev_set)
    s_curr = set(curr_set)
    diff = len(s_prev ^ s_curr)
    denom = max(len(s_prev), len(s_curr), 1)
    return 0.5 * diff / denom


def net_return(gross: float, turnover: float, cost_bps: float) -> float:
    """Net return after transaction cost haircut.

    turnover is one-way (0.5 * |sym_diff| / n).  Each rebalance has two
    legs (sell exits + buy entries), so total cost = 2 * turnover * cost.
    """
    return gross - 2 * turnover * cost_bps / 10_000


# ---------------------------------------------------------------------------
# Long-short decile spread
# ---------------------------------------------------------------------------

def _decile_spread(
    tickers_ranked: List[str],
    fwd_rets: Dict[str, float],
) -> Optional[float]:
    """Top-decile minus bottom-decile equal-weight return.

    Uses the full ranked list (not just top-K). Decile = top/bottom 10%.
    """
    eligible = [t for t in tickers_ranked if t in fwd_rets]
    n = len(eligible)
    if n < 10:
        return None
    d_size = max(1, n // 10)
    top = eligible[:d_size]
    bottom = eligible[-d_size:]
    top_ret = statistics.mean(fwd_rets[t] for t in top)
    bottom_ret = statistics.mean(fwd_rets[t] for t in bottom)
    return top_ret - bottom_ret


# ---------------------------------------------------------------------------
# Beta-hedged portfolio return
# ---------------------------------------------------------------------------

def _avg_field(
    tickers: List[str],
    fwd_rets: Dict[str, float],
    field_map: Dict[str, float],
) -> Optional[float]:
    """Average field value for tickers that have forward returns."""
    vals = [field_map[t] for t in tickers if t in fwd_rets and t in field_map]
    if not vals:
        return None
    return statistics.mean(vals)


def _simple_ols(x: List[float], y: List[float]) -> Tuple[float, float, float]:
    """Simple OLS: y = alpha + beta * x.

    Returns (alpha, beta, r_squared).
    """
    n = len(x)
    if n < 3:
        return 0.0, 0.0, 0.0
    mx = statistics.mean(x)
    my = statistics.mean(y)
    cov = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    var_x = sum((x[i] - mx) ** 2 for i in range(n))
    if var_x == 0:
        return my, 0.0, 0.0
    beta = cov / var_x
    alpha = my - beta * mx
    ss_res = sum((y[i] - alpha - beta * x[i]) ** 2 for i in range(n))
    ss_tot = sum((y[i] - my) ** 2 for i in range(n))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return alpha, beta, r2


def compute_residual_alpha(
    date_results: List["DateResult"],
    horizon: int,
) -> Dict[str, Any]:
    """Regress top-bottom returns on benchmark return to extract residual alpha.

    Returns dict with alpha, beta, r2, alpha_t_stat, n.
    """
    pairs = []
    for dr in date_results:
        if dr.horizon != horizon or dr.skipped:
            continue
        if (dr.gross_return is not None and dr.bottom_k_return is not None
                and dr.benchmark_return is not None):
            ls = dr.gross_return - dr.bottom_k_return
            pairs.append((dr.benchmark_return, ls))

    if len(pairs) < 5:
        return {"n": len(pairs), "alpha": None, "beta": None, "r2": None}

    bm_rets = [p[0] for p in pairs]
    ls_rets = [p[1] for p in pairs]
    alpha, beta, r2 = _simple_ols(bm_rets, ls_rets)

    # Residuals for alpha t-stat
    residuals = [ls_rets[i] - alpha - beta * bm_rets[i] for i in range(len(pairs))]
    se_resid = math.sqrt(sum(r ** 2 for r in residuals) / max(len(pairs) - 2, 1))
    # Standard error of alpha
    var_x = sum((bm_rets[i] - statistics.mean(bm_rets)) ** 2
                for i in range(len(pairs)))
    se_alpha = se_resid * math.sqrt(1.0 / len(pairs) +
                                     statistics.mean(bm_rets) ** 2 / max(var_x, 1e-12))
    alpha_t = alpha / se_alpha if se_alpha > 0 else 0.0

    return {
        "n": len(pairs),
        "alpha": round(alpha, 6),
        "beta": round(beta, 4),
        "r2": round(r2, 4),
        "alpha_t_stat": round(alpha_t, 3),
    }


def _beta_hedged_return(
    held_tickers: List[str],
    fwd_rets: Dict[str, float],
    benchmark_ret: float,
    beta_by_ticker: Dict[str, float],
) -> Optional[float]:
    """Compute beta-hedged top-K return.

    hedged = portfolio_return - beta_portfolio * benchmark_return
    where beta_portfolio = weight-avg beta of held names.
    """
    betas = []
    for t in held_tickers:
        if t in fwd_rets and t in beta_by_ticker:
            betas.append(beta_by_ticker[t])
    if not betas:
        return None
    avg_beta = statistics.mean(betas)
    port_ret = statistics.mean(fwd_rets[t] for t in held_tickers if t in fwd_rets)
    return port_ret - avg_beta * benchmark_ret


# ---------------------------------------------------------------------------
# Decile curve + monotonicity diagnostics
# ---------------------------------------------------------------------------

def compute_decile_curve(
    tickers_ranked: List[str],
    fwd_rets: Dict[str, float],
) -> Optional[List[Dict[str, Any]]]:
    """Compute 10-bin decile curve: mean return per decile.

    Decile 1 = best-ranked tickers, decile 10 = worst-ranked.
    Returns list of dicts: [{decile, n, mean_return}, ...].
    """
    eligible = [t for t in tickers_ranked if t in fwd_rets]
    n = len(eligible)
    if n < 10:
        return None
    d_size = n // 10
    remainder = n % 10
    curve: List[Dict[str, Any]] = []
    pos = 0
    for d in range(10):
        bucket_size = d_size + (1 if d < remainder else 0)
        bucket = eligible[pos:pos + bucket_size]
        pos += bucket_size
        if bucket:
            mean_ret = statistics.mean(fwd_rets[t] for t in bucket)
            curve.append({
                "decile": d + 1,
                "n": len(bucket),
                "mean_return": round(mean_ret, 6),
            })
    return curve


def _monotonic_slope(curve: Optional[List[Dict[str, Any]]]) -> Optional[float]:
    """Correlation between signal-strength proxy and decile mean return.

    Signal proxy: (11 - decile_index), so decile 1 (best) -> proxy 10.
    Positive slope = returns increase with signal strength (consistent with IC > 0).
    """
    if not curve or len(curve) < 3:
        return None
    xs = [float(11 - c["decile"]) for c in curve]
    ys = [c["mean_return"] for c in curve]
    return spearman_ic(xs, ys)


def _share_counts(
    tickers: List[str],
    by_ticker: Dict[str, str],
    fwd_rets: Dict[str, float],
) -> Dict[str, int]:
    """Count categories for tickers that have forward returns."""
    counts: Dict[str, int] = {}
    for t in tickers:
        if t in fwd_rets:
            val = by_ticker.get(t, "unknown")
            counts[val] = counts.get(val, 0) + 1
    return counts


def _write_sign_mismatch(
    out_dir: Path,
    snap_date: str,
    horizon: int,
    data: Dict[str, Any],
) -> None:
    """Write sign mismatch diagnostic JSON."""
    path = out_dir / f"sign_mismatch_{snap_date}_{horizon}d.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Multi-variate OLS
# ---------------------------------------------------------------------------

def _multi_ols(
    x_cols: List[List[float]],
    y: List[float],
) -> Tuple[List[float], float]:
    """Multi-variate OLS: y = b0 + b1*x1 + b2*x2 + ...

    Args:
        x_cols: List of column vectors (each same length as y).
        y: Response vector.

    Returns: (coefficients [b0, b1, ...], r_squared).
    """
    n = len(y)
    p = len(x_cols) + 1  # +1 for intercept
    if n < p + 2:
        return [0.0] * p, 0.0

    cols = [[1.0] * n] + x_cols

    def dot(a: List[float], b: List[float]) -> float:
        return sum(a[i] * b[i] for i in range(n))

    # Normal equations: X^T X beta = X^T y
    xtx = [[dot(cols[i], cols[j]) for j in range(p)] for i in range(p)]
    xty = [dot(cols[i], y) for i in range(p)]

    # Gaussian elimination with partial pivoting
    aug = [xtx[i][:] + [xty[i]] for i in range(p)]
    for i in range(p):
        max_row = max(range(i, p), key=lambda r: abs(aug[r][i]))
        aug[i], aug[max_row] = aug[max_row], aug[i]
        if abs(aug[i][i]) < 1e-12:
            return [0.0] * p, 0.0
        for j in range(i + 1, p):
            factor = aug[j][i] / aug[i][i]
            for k in range(i, p + 1):
                aug[j][k] -= factor * aug[i][k]

    # Back-substitution
    beta = [0.0] * p
    for i in range(p - 1, -1, -1):
        beta[i] = aug[i][p]
        for j in range(i + 1, p):
            beta[i] -= aug[i][j] * beta[j]
        beta[i] /= aug[i][i]

    y_mean = statistics.mean(y)
    ss_tot = sum((y[i] - y_mean) ** 2 for i in range(n))
    y_pred = [sum(beta[j] * cols[j][i] for j in range(p)) for i in range(n)]
    ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(n))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return beta, r2


def compute_residual_alpha_multi(
    date_results: List["DateResult"],
    horizon: int,
) -> Dict[str, Any]:
    """Multi-regressor residual alpha: L/S ~ benchmark + beta_diff + drawdown_diff.

    Returns dict with alpha, betas for each regressor, r2, alpha_t_stat.
    """
    rows: List[Tuple[float, float, float, float]] = []
    for dr in date_results:
        if dr.horizon != horizon or dr.skipped:
            continue
        if (dr.gross_return is not None and dr.bottom_k_return is not None
                and dr.benchmark_return is not None
                and dr.topk_avg_beta is not None and dr.bottomk_avg_beta is not None
                and dr.topk_avg_drawdown is not None
                and dr.bottomk_avg_drawdown is not None):
            ls = dr.gross_return - dr.bottom_k_return
            beta_diff = dr.topk_avg_beta - dr.bottomk_avg_beta
            dd_diff = dr.topk_avg_drawdown - dr.bottomk_avg_drawdown
            rows.append((dr.benchmark_return, beta_diff, dd_diff, ls))

    if len(rows) < 8:
        return {"n": len(rows), "alpha": None}

    bm = [r[0] for r in rows]
    bd = [r[1] for r in rows]
    dd = [r[2] for r in rows]
    ls = [r[3] for r in rows]

    beta, r2 = _multi_ols([bm, bd, dd], ls)
    alpha = beta[0]

    n = len(rows)
    p = 4
    y_pred = [beta[0] + beta[1] * bm[i] + beta[2] * bd[i] + beta[3] * dd[i]
              for i in range(n)]
    residuals = [ls[i] - y_pred[i] for i in range(n)]
    se_resid = math.sqrt(sum(r ** 2 for r in residuals) / max(n - p, 1))
    se_alpha = se_resid / math.sqrt(n) if n > 0 else 0.0
    alpha_t = alpha / se_alpha if se_alpha > 0 else 0.0

    return {
        "n": n,
        "alpha": round(alpha, 6),
        "beta_benchmark": round(beta[1], 4),
        "beta_beta_diff": round(beta[2], 4),
        "beta_dd_diff": round(beta[3], 4),
        "r2": round(r2, 4),
        "alpha_t_stat": round(alpha_t, 3),
    }


# ---------------------------------------------------------------------------
# Walk-forward split assignment
# ---------------------------------------------------------------------------

def assign_splits(
    snap_dates: List[str],
    mode: str = "none",
    train_end: Optional[str] = None,
    test_start: Optional[str] = None,
    wf_train_months: int = 0,
    wf_test_months: int = 0,
) -> Dict[str, Tuple[int, bool, bool]]:
    """Assign train/test labels to snapshot dates.

    Returns: {date: (split_id, is_train, is_test)}

    Modes:
      none: all dates evaluated, no split labels.
      fixed: dates <= train_end are train, dates >= test_start are test.
      walk_forward: first wf_train_months dates are train-only,
          remaining dates are test, grouped into chunks of wf_test_months.
    """
    result: Dict[str, Tuple[int, bool, bool]] = {}

    if mode == "none":
        for d in snap_dates:
            result[d] = (0, False, False)
        return result

    if mode == "fixed":
        for d in snap_dates:
            is_train = d <= train_end if train_end else False
            is_test = d >= test_start if test_start else False
            sid = 1 if (is_train or is_test) else 0
            result[d] = (sid, is_train, is_test)
        return result

    if mode == "walk_forward":
        n = len(snap_dates)
        train_count = min(wf_train_months, n)
        for i in range(train_count):
            result[snap_dates[i]] = (0, True, False)
        split = 0
        pos = train_count
        while pos < n:
            split += 1
            end = min(pos + wf_test_months, n)
            for i in range(pos, end):
                result[snap_dates[i]] = (split, False, True)
            pos = end
        return result

    return {d: (0, False, False) for d in snap_dates}


# ---------------------------------------------------------------------------
# Snapshot discovery + loading
# ---------------------------------------------------------------------------

def discover_snapshot_dates(
    snapshot_root: Path,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> List[str]:
    """Return sorted list of YYYY-MM-DD snapshot dates."""
    dates: List[str] = []
    if not snapshot_root.exists():
        return dates
    for d in sorted(snapshot_root.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        # Must be YYYY-MM-DD format
        if len(name) != 10 or name[4] != "-" or name[7] != "-":
            continue
        try:
            date.fromisoformat(name)
        except ValueError:
            continue
        if date_from and name < date_from:
            continue
        if date_to and name > date_to:
            continue
        dates.append(name)
    return dates


def load_rankings(snapshot_dir: Path) -> List[Dict[str, str]]:
    """Load rankings.csv from a snapshot directory."""
    csv_path = snapshot_dir / "rankings.csv"
    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_positions(snapshot_dir: Path) -> List[Dict[str, str]]:
    """Load portfolio_positions.csv from a snapshot directory."""
    csv_path = snapshot_dir / "portfolio_positions.csv"
    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_pit_metadata(snapshot_dir: Path, snap_date: str) -> Tuple[bool, str]:
    """Validate PIT enforcement via metadata.json."""
    meta_path = snapshot_dir / "metadata.json"
    if not meta_path.exists():
        return False, "metadata.json missing"
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return False, f"metadata.json unreadable: {e}"
    as_of = meta.get("as_of_date", "")
    if as_of != snap_date:
        return False, f"as_of_date={as_of} != {snap_date}"
    return True, ""


# ---------------------------------------------------------------------------
# Coinvest eval-mode rescoring
# ---------------------------------------------------------------------------

def rescore_rankings(
    rankings: List[Dict[str, str]],
    coinvest_eval_mode: str,
    ruleset: DecisionRuleset,
) -> List[Dict[str, str]]:
    """Re-sort rankings with modified coinvest signal using the full production sort key.

    Uses ``compute_actionable_sort_key()`` from ``decision_engine`` via a
    temporarily modified ``DecisionRuleset``, matching the pattern in
    ``scripts/research/rerank_snapshots.py:rerank()``.

    This replaced the old simplified 3-tuple sort key (pre-v1.8) which
    omitted prefix ordering, catalyst priority, missingness, etc. and
    overstated coinvest-off IC improvements by 83-128%.

    Mode semantics:
        - "off":    ``enable_coinvest_sort_signal=False`` — coinvest has zero effect.
        - "contra": negate ``coinvest_sort_weight`` and disable ``coinvest_positive_only``
                    so the sort key truly inverts the signal (positive z penalises,
                    negative z boosts).

    Args:
        rankings: list of ranking dicts (from rankings.csv).
        coinvest_eval_mode: "off" or "contra".
        ruleset: DecisionRuleset to use as the base for rescoring.

    Returns:
        rankings list re-sorted with updated actionable_rank.
    """
    # Build modified ruleset for this eval mode
    if coinvest_eval_mode == "off":
        rs = dc_replace(ruleset, enable_coinvest_sort_signal=False)
    elif coinvest_eval_mode == "contra":
        # True sign-flip: negate the weight and disable positive-only clamp
        # so that originally-positive z values penalise and originally-negative
        # z values boost.  No value mutation needed.
        rs = dc_replace(
            ruleset,
            coinvest_sort_weight=-abs(ruleset.coinvest_sort_weight),
            coinvest_positive_only=False,
        )
    else:
        rs = ruleset

    # Backfill missing columns FIRST (before any sort) so that
    # compute_actionable_sort_key() sees all required fields and
    # no downstream backfill can overwrite eval-mode mutations.
    backfill_columns(rankings)

    # Sort using production-identical logic (same pattern as rerank_snapshots.py:rerank())
    rankings.sort(key=lambda r: compute_actionable_sort_key(
        decision_fields=r,
        archetype=r.get("archetype", ""),
        optionality=_ranking_safe_float(r.get("clinical_optionality_pct_dev")),
        composite_rank=r.get("composite_rank"),
        ticker=r.get("ticker", ""),
        catalyst_event_type=r.get("catalyst_event_type", ""),
        catalyst_source=r.get("catalyst_source", ""),
        ruleset=rs,
        tiebreaker_pct=(
            _ranking_safe_float(r.get("alpha_cohort_pct"))
            if rs.sort_anchor == "alpha_cohort"
            else (_ranking_safe_float(r.get("commercial_quality_pct"))
                  if r.get("archetype", "").startswith("commercial_")
                  else _ranking_safe_float(r.get("clinical_optionality_pct_dev")))
        ),
    ))

    # Assign actionable_rank for eligible rows
    rank = 1
    for r in rankings:
        if r.get("eligible") == "1" or r.get("eligible", "").lower() == "true":
            r["actionable_rank"] = str(rank)
            rank += 1
        else:
            r["actionable_rank"] = ""

    return rankings


# ---------------------------------------------------------------------------
# Coinvest signal diagnostics
# ---------------------------------------------------------------------------

def _coinvest_signal_diagnostics(
    rankings: List[Dict[str, str]],
    top_k: int,
) -> Dict[str, Any]:
    """Compute per-date coinvest signal existence diagnostics.

    Returns:
        Dict with pct_with_signal, tier1_distribution, n_eligible, and
        the top-K ticker set for Jaccard comparison.
    """
    eligible = [r for r in rankings
                if r.get("eligible", "").lower() in ("true", "1")]
    n_eligible = len(eligible)

    tier1_vals: List[int] = []
    for r in eligible:
        raw = r.get("sponsor_tier1_count", "") or r.get("de_tier1_count", "")
        try:
            tier1_vals.append(int(float(raw)) if raw else 0)
        except (ValueError, TypeError):
            tier1_vals.append(0)

    n_nonzero = sum(1 for v in tier1_vals if v > 0)
    pct_with_signal = (n_nonzero / n_eligible * 100) if n_eligible else 0.0

    # Distribution percentiles (manual, no numpy)
    if tier1_vals:
        sorted_vals = sorted(tier1_vals)
        n = len(sorted_vals)

        def _pctl(p: float) -> int:
            idx = int(p / 100.0 * (n - 1))
            return sorted_vals[min(idx, n - 1)]

        distribution = {
            "p50": _pctl(50),
            "p90": _pctl(90),
            "p99": _pctl(99),
            "max": sorted_vals[-1] if sorted_vals else 0,
            "n_nonzero": n_nonzero,
        }
    else:
        distribution = {"p50": 0, "p90": 0, "p99": 0, "max": 0, "n_nonzero": 0}

    # Capture top-K set for Jaccard comparison
    def _safe_rank(r: Dict[str, str]) -> int:
        try:
            return int(r.get("actionable_rank") or 9999)
        except (ValueError, TypeError):
            return 9999

    ranked = sorted(eligible, key=_safe_rank)
    topk_tickers = set(r.get("ticker", "") for r in ranked[:top_k])

    return {
        "pct_with_signal": round(pct_with_signal, 2),
        "tier1_distribution": distribution,
        "n_eligible": n_eligible,
        "_topk_set": topk_tickers,
        "_raw_tier1_vals": tier1_vals,
    }


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

@dataclass
class DateResult:
    date: str
    horizon: int
    trade_date: str = ""
    ic: Optional[float] = None
    gross_return: Optional[float] = None
    net_return: Optional[float] = None
    turnover: Optional[float] = None
    n_signal: int = 0
    n_held: int = 0
    coverage: float = 0.0
    skipped: bool = False
    skip_reason: str = ""
    # Bottom-K for sign checks
    bottom_k_return: Optional[float] = None
    # Benchmark-relative fields
    benchmark_return: Optional[float] = None
    excess_return: Optional[float] = None
    # Long-short spread
    ls_return: Optional[float] = None
    ls_net_return: Optional[float] = None
    # Beta-hedged
    hedged_return: Optional[float] = None
    # Exposure diagnostics
    topk_avg_beta: Optional[float] = None
    bottomk_avg_beta: Optional[float] = None
    topk_avg_drawdown: Optional[float] = None
    bottomk_avg_drawdown: Optional[float] = None
    # Monotonicity diagnostics
    monotonic_slope: Optional[float] = None
    # Walk-forward split
    split_id: int = 0
    is_train: bool = False
    is_test: bool = False
    # Eval universe diagnostics
    n_universe_eval: int = 0
    n_missing_anchor: int = 0
    n_missing_forward: int = 0


@dataclass
class EvalSummary:
    horizons: List[int] = field(default_factory=list)
    top_k: int = DEFAULT_TOP_K
    cost_bps: float = DEFAULT_COST_BPS
    n_dates: int = 0
    n_evaluated: int = 0
    n_skipped: int = 0
    by_horizon: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    use_positions: bool = False
    anchor_mode: str = "exact"
    benchmark: str = "none"
    split_mode: str = "none"
    universe_mode: str = "current"
    coinvest_signal_diagnostics: Optional[Dict[str, Any]] = None
    data_freshness: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _staleness_status(summary: EvalSummary) -> Dict[str, Any]:
    """Check forward-return data freshness from summary diagnostics.

    Returns {"stale": bool, "message": str}.
    """
    df = summary.data_freshness
    if df is None:
        return {"stale": False, "message": "no data_freshness block (cold start)"}

    stale = df.get("fwd_returns_stale", False)
    if not stale:
        return {"stale": False, "message": "forward returns up to date"}

    parts = []
    if df.get("price_end_date"):
        parts.append(f"price_end_date={df['price_end_date']}")
    if df.get("max_snapshot_date"):
        parts.append(f"max_snapshot_date={df['max_snapshot_date']}")
    if df.get("price_gap_days") is not None:
        parts.append(f"price_gap_days={df['price_gap_days']}")
    return {"stale": True, "message": f"forward returns stale: {'; '.join(parts)}"}


def evaluate(
    snapshot_root: Path,
    price_csv: Path,
    horizons: List[int],
    top_k: int = DEFAULT_TOP_K,
    cost_bps: float = DEFAULT_COST_BPS,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    pit_mode: str = "strict",
    use_positions: bool = False,
    min_price_coverage: float = DEFAULT_MIN_PRICE_COVERAGE,
    anchor_mode: str = "exact",
    benchmark: str = "none",
    long_short_deciles: bool = False,
    split_mode: str = "none",
    train_end: Optional[str] = None,
    test_start: Optional[str] = None,
    wf_train_months: int = 0,
    wf_test_months: int = 0,
    out_dir: Optional[Path] = None,
    universe_mode: str = "current",
    component_eval: bool = False,
    rebalance_buffer_ranks: int = 0,
    turnover_cap: float = 0.0,
    top_quantile: float = 0.0,
    coinvest_eval_mode: str = "default",
    catalyst_eval_mode: str = "default",
    ruleset: Optional[DecisionRuleset] = None,
    preflight: bool = False,
    preflight_strict: bool = False,
) -> Tuple[EvalSummary, List[DateResult], List[Dict[str, str]]]:
    """Run the full walk-forward evaluation.

    Args:
        anchor_mode: "exact" (require price on snap_date),
            "next_trading_day" (first date > snap_date, PIT-safe),
            "prev_trading_day" (last date <= snap_date).
        benchmark: "XBI", "SPY", or "none" — compute excess returns.
        long_short_deciles: compute top-decile minus bottom-decile spread.
        catalyst_eval_mode: "default" (use CSV values) or "logistic" (recompute from catalyst_days).
        split_mode: "none", "fixed", or "walk_forward".
        train_end: last date in training set (fixed split).
        test_start: first date in test set (fixed split).
        wf_train_months: number of initial dates for training (walk_forward).
        wf_test_months: dates per test fold (walk_forward).
        out_dir: output directory for sign mismatch dumps.
        universe_mode: "current" (use rankings as-is) or "price_available"
            (restrict eval universe to tickers with anchor+forward prices).
        component_eval: compute per-component IC and L/S.
        rebalance_buffer_ranks: buffer zone around top-K for rebalance.
        turnover_cap: max turnover fraction per rebalance (0=unlimited).
        top_quantile: use top N% instead of top-K (0=use top_k).

    Returns (summary, date_results, skips).
    """
    snap_dates = discover_snapshot_dates(snapshot_root, date_from, date_to)

    # Assign train/test splits
    split_map = assign_splits(
        snap_dates, split_mode,
        train_end=train_end, test_start=test_start,
        wf_train_months=wf_train_months, wf_test_months=wf_test_months,
    )

    prices = load_price_series(price_csv)

    # Build sorted trading dates from all price data
    all_dates_set: set = set()
    for ticker_prices in prices.values():
        all_dates_set.update(ticker_prices.keys())
    sorted_dates = sorted(all_dates_set)

    # Benchmark prices
    benchmark_ticker = benchmark.upper() if benchmark not in ("none", "") else None
    benchmark_prices = prices.get(benchmark_ticker, {}) if benchmark_ticker else {}

    date_results: List[DateResult] = []
    skips: List[Dict[str, str]] = []

    # Per-horizon accumulators
    horizon_ics: Dict[int, List[float]] = {h: [] for h in horizons}
    horizon_gross: Dict[int, List[float]] = {h: [] for h in horizons}
    horizon_bottom: Dict[int, List[float]] = {h: [] for h in horizons}
    horizon_net: Dict[int, List[float]] = {h: [] for h in horizons}
    horizon_turnover: Dict[int, List[float]] = {h: [] for h in horizons}
    horizon_excess: Dict[int, List[float]] = {h: [] for h in horizons}
    horizon_ls: Dict[int, List[float]] = {h: [] for h in horizons}
    horizon_ls_net: Dict[int, List[float]] = {h: [] for h in horizons}
    horizon_hedged: Dict[int, List[float]] = {h: [] for h in horizons}
    prev_holdings: Dict[int, List[str]] = {h: [] for h in horizons}
    sign_mismatches: Dict[int, int] = {h: 0 for h in horizons}

    # Component eval accumulators
    COMPONENT_COLS = [
        "clinical_score_z_tier", "coinvest_score_z", "catalyst_decay_w",
        "clinical_alpha_z", "inst_delta_z", "alpha_cohort_raw",
        "de_alpha_60d",
    ]
    component_results: List[Dict[str, Any]] = []  # per-date per-horizon per-component
    coinvest_audit_rows: List[Dict[str, Any]] = []  # per-date ticker-level audit

    # Coinvest signal diagnostics accumulators
    coinvest_diag_per_date: List[Dict[str, Any]] = []
    coinvest_topk_default_sets: List[set] = []  # top-K sets before rescore
    coinvest_topk_rescored_sets: List[set] = []  # top-K sets after rescore

    n_evaluated = 0

    for snap_date in snap_dates:
        snap_dir = snapshot_root / snap_date

        # PIT check
        if pit_mode == "strict":
            ok, reason = check_pit_metadata(snap_dir, snap_date)
            if not ok:
                for h in horizons:
                    dr = DateResult(date=snap_date, horizon=h, skipped=True,
                                    skip_reason=f"PIT: {reason}")
                    date_results.append(dr)
                skips.append({"date": snap_date, "reason": f"PIT: {reason}"})
                continue

        # Preflight check
        if preflight or preflight_strict:
            from tools.snapshot_preflight import run_preflight as _run_preflight
            pf = _run_preflight(snap_dir, snap_date)
            skip_pf = (pf.status == "FAIL"
                       or (preflight_strict and pf.status == "WARN"))
            if skip_pf:
                details = "; ".join(
                    c.detail for c in pf.checks if c.status != "PASS"
                )
                tag = f"preflight_{pf.status}"
                for h in horizons:
                    dr = DateResult(date=snap_date, horizon=h, skipped=True,
                                    skip_reason=f"{tag}: {details}")
                    date_results.append(dr)
                skips.append({"date": snap_date,
                              "reason": f"{tag}: {details}"})
                continue

        # Resolve trade_date from anchor_mode
        trade_date = resolve_trade_date(sorted_dates, snap_date, anchor_mode)
        if trade_date is None:
            for h in horizons:
                dr = DateResult(date=snap_date, horizon=h, skipped=True,
                                skip_reason="ANCHOR_DATE_NOT_FOUND")
                date_results.append(dr)
            skips.append({"date": snap_date,
                          "reason": f"ANCHOR_DATE_NOT_FOUND (mode={anchor_mode})"})
            continue

        # Load rankings or positions
        if use_positions:
            positions = load_positions(snap_dir)
            tickers_ranked = [r["ticker"] for r in positions
                              if r.get("ticker")]
            eligible_set: set = set(tickers_ranked)
        else:
            rankings = load_rankings(snap_dir)
            if not rankings:
                for h in horizons:
                    dr = DateResult(date=snap_date, horizon=h, skipped=True,
                                    skip_reason="EMPTY_RANKINGS")
                    date_results.append(dr)
                skips.append({"date": snap_date, "reason": "EMPTY_RANKINGS"})
                continue
            # Coinvest signal diagnostics (before any rescore)
            diag = _coinvest_signal_diagnostics(rankings, top_k)
            coinvest_diag_per_date.append(diag)
            default_topk = diag["_topk_set"]

            # Re-sort with modified coinvest signal if requested
            if coinvest_eval_mode != "default" and ruleset is not None:
                coinvest_topk_default_sets.append(default_topk)
                rankings = rescore_rankings(rankings, coinvest_eval_mode, ruleset)
                # Capture rescored top-K for Jaccard comparison
                diag_post = _coinvest_signal_diagnostics(rankings, top_k)
                coinvest_topk_rescored_sets.append(diag_post["_topk_set"])

            # Sort by actionable_rank ascending (per-row safe parse)
            def _safe_rank(r):
                try:
                    return int(r.get("actionable_rank") or 9999)
                except (ValueError, TypeError):
                    return 9999
            rankings.sort(key=_safe_rank)
            tickers_ranked = [r["ticker"] for r in rankings if r.get("ticker")]
            eligible_set = {r["ticker"] for r in rankings
                           if r.get("ticker")
                           and r.get("actionable_rank", "").strip()}

        # Build per-ticker lookups from rankings for hedged returns + diagnostics
        beta_by_ticker: Dict[str, float] = {}
        drawdown_by_ticker: Dict[str, float] = {}
        archetype_by_ticker: Dict[str, str] = {}
        catalyst_mode_by_ticker: Dict[str, str] = {}
        if not use_positions:
            for r in rankings:
                t = r.get("ticker", "")
                if not t:
                    continue
                # Beta
                b = r.get("de_beta_xbi_60d", "")
                if b:
                    try:
                        beta_by_ticker[t] = float(b)
                    except (ValueError, TypeError):
                        pass
                # Drawdown
                dd = r.get("de_drawdown", "")
                if dd:
                    try:
                        drawdown_by_ticker[t] = float(dd)
                    except (ValueError, TypeError):
                        pass
                # Archetype
                arch = r.get("archetype", "")
                if arch:
                    archetype_by_ticker[t] = arch
                # Catalyst mode
                cm = r.get("catalyst_mode", "")
                if cm:
                    catalyst_mode_by_ticker[t] = cm

        # Build component score lookups (for --component-eval)
        comp_by_ticker: Dict[str, Dict[str, float]] = {}
        # Signal flags per ticker (for sparse signal filtering)
        signal_flags_by_ticker: Dict[str, Dict[str, str]] = {}
        # Coinvest staleness diagnostic
        _coinvest_age_vals: List[float] = []
        if component_eval and not use_positions:
            for r in rankings:
                t = r.get("ticker", "")
                if not t:
                    continue
                scores: Dict[str, float] = {}
                for col in COMPONENT_COLS:
                    v = r.get(col, "")
                    if v:
                        try:
                            scores[col] = float(v)
                        except (ValueError, TypeError):
                            pass
                # Logistic catalyst override: recompute catalyst_decay_w from catalyst_days
                if catalyst_eval_mode == "logistic":
                    days_str = r.get("catalyst_days", "")
                    if days_str and str(days_str).strip():
                        try:
                            days_val = float(days_str)
                            scores["catalyst_decay_w"] = round(
                                _logistic_decay(days_val), 4
                            )
                        except (ValueError, TypeError):
                            pass
                if scores:
                    comp_by_ticker[t] = scores
                # Capture signal flags
                flags: Dict[str, str] = {}
                for flag_col in SIGNAL_FLAG_MAP.values():
                    fv = r.get(flag_col, "")
                    if fv:
                        flags[flag_col] = fv
                if flags:
                    signal_flags_by_ticker[t] = flags
                # Coinvest staleness
                _age_str = r.get("coinvest_filing_age_days", "")
                if _age_str and str(_age_str).strip():
                    try:
                        _coinvest_age_vals.append(float(_age_str))
                    except (ValueError, TypeError):
                        pass

        date_evaluated = False
        for h in horizons:
            # Compute forward returns for all tickers with prices
            # Track eval universe: anchor available + forward available
            n_missing_anchor = 0
            n_missing_forward = 0
            fwd_rets: Dict[str, float] = {}
            for ticker in tickers_ranked:
                if ticker not in prices:
                    n_missing_anchor += 1
                    continue
                anchor_price = prices[ticker].get(trade_date)
                if anchor_price is None or anchor_price <= 0:
                    n_missing_anchor += 1
                    continue
                ret = compute_forward_return(prices[ticker], sorted_dates, trade_date, h)
                if ret is not None:
                    fwd_rets[ticker] = ret
                else:
                    n_missing_forward += 1

            n_universe_eval = len(fwd_rets)

            coverage = n_universe_eval / max(len(tickers_ranked), 1)
            if coverage < min_price_coverage:
                dr = DateResult(date=snap_date, horizon=h, trade_date=trade_date,
                                skipped=True, skip_reason="LOW_COVERAGE",
                                coverage=round(coverage, 4),
                                n_universe_eval=n_universe_eval,
                                n_missing_anchor=n_missing_anchor,
                                n_missing_forward=n_missing_forward)
                date_results.append(dr)
                skips.append({"date": snap_date, "horizon": str(h),
                              "trade_date": trade_date,
                              "reason": f"LOW_COVERAGE ({coverage:.1%})"})
                continue

            # In price_available mode, restrict the ranked list to eval universe
            if universe_mode == "price_available":
                eval_tickers = [t for t in tickers_ranked if t in fwd_rets]
            else:
                eval_tickers = tickers_ranked

            # Signal: negative actionable_rank (lower rank = better = higher signal)
            signal_tickers = [t for t in eval_tickers if t in fwd_rets]
            signal_vals = [-float(i + 1) for i, t in enumerate(eval_tickers)
                           if t in fwd_rets]
            return_vals = [fwd_rets[t] for t in signal_tickers]

            ic = spearman_ic(signal_vals, return_vals)

            # Determine effective top-K: use top_quantile if set
            eff_k = top_k
            if top_quantile > 0:
                eff_k = max(1, int(len(eval_tickers) * top_quantile))

            # Top-K portfolio with optional rebalance buffer
            if rebalance_buffer_ranks > 0 and prev_holdings[h]:
                # Buffer: keep existing holdings unless they fall out of top_k + buffer
                prev_set = set(prev_holdings[h])
                candidate_tickers = eval_tickers[:eff_k + rebalance_buffer_ranks]
                # Keep prev holdings that are still in the buffer zone
                kept = [t for t in prev_holdings[h] if t in set(candidate_tickers)]
                # Add new tickers from top-K that aren't already held
                new_slots = eff_k - len(kept)
                if new_slots > 0:
                    for t in eval_tickers[:eff_k]:
                        if t not in set(kept):
                            kept.append(t)
                            if len(kept) >= eff_k:
                                break
                top_k_tickers = kept[:eff_k]
            else:
                top_k_tickers = eval_tickers[:eff_k]

            gross, n_held = top_k_portfolio_return(top_k_tickers, fwd_rets, eff_k)

            # Bottom-K portfolio (eligible only — ineligible tickers have
            # no signal; including them makes spread vs IC incomparable)
            eligible_eval = [t for t in eval_tickers if t in eligible_set]
            bottom_gross, _ = bottom_k_portfolio_return(eligible_eval, fwd_rets, eff_k)

            # Turnover (with optional cap)
            turn = compute_turnover(prev_holdings[h], top_k_tickers)
            if turnover_cap > 0 and turn > turnover_cap and prev_holdings[h]:
                # Scale trades: blend prev and new holdings
                scale = turnover_cap / turn if turn > 0 else 1.0
                prev_set = set(prev_holdings[h])
                new_set = set(top_k_tickers)
                # Tickers to drop = in prev but not in new
                to_drop = [t for t in prev_holdings[h] if t not in new_set]
                # Tickers to add = in new but not in prev
                to_add = [t for t in top_k_tickers if t not in prev_set]
                n_trades = max(1, int(len(to_drop) * scale))
                # Keep most of prev, only swap n_trades names
                capped = [t for t in prev_holdings[h] if t not in set(to_drop[:n_trades])]
                capped.extend(to_add[:n_trades])
                top_k_tickers = capped[:eff_k]
                gross, n_held = top_k_portfolio_return(top_k_tickers, fwd_rets, eff_k)
                turn = compute_turnover(prev_holdings[h], top_k_tickers)
            prev_holdings[h] = top_k_tickers

            net_ret = None
            if gross is not None:
                net_ret = net_return(gross, turn, cost_bps)

            # Decile curve + monotonicity diagnostics
            curve = compute_decile_curve(eval_tickers, fwd_rets)
            slope = _monotonic_slope(curve)

            # Benchmark return (computed before mismatch check so dump can include it)
            bm_ret = None
            excess_ret = None
            if benchmark_prices:
                bm_ret = compute_forward_return(benchmark_prices, sorted_dates, trade_date, h)
                if bm_ret is not None and gross is not None:
                    excess_ret = gross - bm_ret

            # Enhanced sign consistency check
            # Uses winsorized top-bottom to avoid false alerts from extreme
            # return outliers (e.g. small-cap binary events in bottom bucket).
            # See output/sign_mismatch_triage.md for analysis.
            mismatch = False
            top_minus_bottom = None
            top_minus_bottom_winsorized = None
            if ic is not None and gross is not None and bottom_gross is not None:
                top_minus_bottom = gross - bottom_gross
                # Winsorize bucket returns at p5/p95 before mismatch check
                top_k_rets = [fwd_rets[t] for t in tickers_ranked[:top_k] if t in fwd_rets]
                bot_k_rets = [fwd_rets[t] for t in tickers_ranked[-top_k:] if t in fwd_rets]
                all_bucket_rets = top_k_rets + bot_k_rets
                if len(all_bucket_rets) >= 10:
                    s = sorted(all_bucket_rets)
                    p5 = s[max(0, len(s) // 20)]
                    p95 = s[min(len(s) - 1, 19 * len(s) // 20)]
                    top_w = [max(p5, min(p95, r)) for r in top_k_rets]
                    bot_w = [max(p5, min(p95, r)) for r in bot_k_rets]
                    if top_w and bot_w:
                        top_minus_bottom_winsorized = (
                            sum(top_w) / len(top_w) - sum(bot_w) / len(bot_w)
                        )
                tmb_check = top_minus_bottom_winsorized if top_minus_bottom_winsorized is not None else top_minus_bottom
                if (ic > 0.02 and tmb_check < -0.005) or \
                   (ic < -0.02 and tmb_check > 0.005):
                    mismatch = True
            if ic is not None and slope is not None:
                if (ic > 0.02 and slope < -0.02) or \
                   (ic < -0.02 and slope > 0.02):
                    mismatch = True
            if mismatch:
                msg = (
                    f"  ::warning::SIGN_MISMATCH {snap_date} h={h}: "
                    f"IC={ic:.4f}"
                )
                if top_minus_bottom is not None:
                    msg += f" top-bottom={top_minus_bottom:.4f}"
                if slope is not None:
                    msg += f" slope={slope:.4f}"
                print(msg, file=sys.stderr)
                # Write mismatch JSON dump
                if out_dir is not None:
                    top_detail = [
                        {"ticker": t, "rank": i + 1,
                         "fwd_return": round(fwd_rets.get(t, float("nan")), 6)}
                        for i, t in enumerate(tickers_ranked[:top_k])
                        if t in fwd_rets
                    ]
                    bottom_detail = [
                        {"ticker": t, "rank": len(tickers_ranked) - top_k + i + 1,
                         "fwd_return": round(fwd_rets.get(t, float("nan")), 6)}
                        for i, t in enumerate(tickers_ranked[-top_k:])
                        if t in fwd_rets
                    ]
                    _write_sign_mismatch(out_dir, snap_date, h, {
                        "as_of_date": snap_date,
                        "trade_date": trade_date,
                        "horizon": h,
                        "ic": _round_opt(ic, 4),
                        "top_k_return": _round_opt(gross, 6),
                        "bottom_k_return": _round_opt(bottom_gross, 6),
                        "top_minus_bottom": _round_opt(top_minus_bottom, 6),
                        "top_minus_bottom_winsorized": _round_opt(top_minus_bottom_winsorized, 6),
                        "monotonic_slope": _round_opt(slope, 4),
                        "decile_curve": curve,
                        "top_tickers": top_detail,
                        "bottom_tickers": bottom_detail,
                        "benchmark_return": _round_opt(bm_ret, 6),
                    })

            # Long-short decile spread
            ls_ret = None
            ls_net_ret = None
            if long_short_deciles:
                ls_ret = _decile_spread(eval_tickers, fwd_rets)
                if ls_ret is not None:
                    # LS has two portfolios (long + short), each with buy+sell legs
                    ls_net_ret = ls_ret - 4 * turn * cost_bps / 10_000

            # Beta-hedged return
            hedged_ret = None
            if benchmark_prices and beta_by_ticker and bm_ret is not None:
                held_with_rets = [t for t in top_k_tickers if t in fwd_rets]
                if held_with_rets:
                    hedged_ret = _beta_hedged_return(
                        held_with_rets, fwd_rets, bm_ret, beta_by_ticker
                    )

            # Exposure diagnostics: avg beta/drawdown for top-K vs bottom-K
            topk_beta = _avg_field(top_k_tickers, fwd_rets, beta_by_ticker)
            bottomk_beta = _avg_field(eval_tickers[-eff_k:], fwd_rets, beta_by_ticker)
            topk_dd = _avg_field(top_k_tickers, fwd_rets, drawdown_by_ticker)
            bottomk_dd = _avg_field(eval_tickers[-eff_k:], fwd_rets, drawdown_by_ticker)

            # Component eval: per-component IC and L/S
            if component_eval and comp_by_ticker:
                for col in COMPONENT_COLS:
                    comp_signal = []
                    comp_returns = []
                    _flag_col = SIGNAL_FLAG_MAP.get(col)
                    _n_with_signal = 0
                    for t in signal_tickers:
                        if t in comp_by_ticker and col in comp_by_ticker[t]:
                            # Filter by signal flag when available
                            if _flag_col:
                                _fv = signal_flags_by_ticker.get(t, {}).get(_flag_col, "")
                                if _fv == "True":
                                    _n_with_signal += 1
                                else:
                                    continue  # skip tickers without real signal
                            comp_signal.append(comp_by_ticker[t][col])
                            comp_returns.append(fwd_rets[t])

                    # Dispersion diagnostics
                    comp_stdev = None
                    comp_pct_ties = None
                    flat_signal = True  # assume flat until proven otherwise
                    n_sig = len(comp_signal)
                    if n_sig >= 2:
                        comp_stdev = statistics.stdev(comp_signal)
                        # Count ties: values that appear more than once
                        from collections import Counter as _Counter
                        val_counts = _Counter(comp_signal)
                        n_tied = sum(c for c in val_counts.values() if c > 1)
                        comp_pct_ties = n_tied / n_sig
                        flat_signal = comp_stdev < 1e-9 or comp_pct_ties > 0.95

                    comp_ic = spearman_ic(comp_signal, comp_returns)

                    # Component L/S: only compute if signal has real dispersion
                    comp_ls = None
                    comp_top_decile_ret = None
                    comp_bot_decile_ret = None
                    comp_mono_slope = None
                    decile_means = []
                    if n_sig >= 10 and not flat_signal:
                        # Sort tickers by component score descending
                        comp_sorted = sorted(
                            [(t, comp_by_ticker[t][col]) for t in signal_tickers
                             if t in comp_by_ticker and col in comp_by_ticker[t]],
                            key=lambda x: -x[1]
                        )
                        d_size = max(1, len(comp_sorted) // 10)
                        top_comp = [t for t, _ in comp_sorted[:d_size]]
                        bot_comp = [t for t, _ in comp_sorted[-d_size:]]
                        top_ret = statistics.mean(fwd_rets[t] for t in top_comp) if top_comp else None
                        bot_ret = statistics.mean(fwd_rets[t] for t in bot_comp) if bot_comp else None
                        if top_ret is not None and bot_ret is not None:
                            comp_ls = top_ret - bot_ret
                        comp_top_decile_ret = top_ret
                        comp_bot_decile_ret = bot_ret
                        # Monotonic slope: split into 10 deciles, regress mean return vs decile index
                        n_comp = len(comp_sorted)
                        comp_d_size = n_comp // 10
                        comp_remainder = n_comp % 10
                        decile_means = []
                        pos = 0
                        for d_idx in range(10):
                            bucket_sz = comp_d_size + (1 if d_idx < comp_remainder else 0)
                            bucket_tickers = [t for t, _ in comp_sorted[pos:pos + bucket_sz]]
                            pos += bucket_sz
                            bucket_rets = [fwd_rets[t] for t in bucket_tickers if t in fwd_rets]
                            if bucket_rets:
                                decile_means.append(statistics.mean(bucket_rets))
                        if len(decile_means) >= 3:
                            # Signal proxy: decile 1 (best) → 10, decile 10 → 1
                            xs = [float(len(decile_means) - i) for i in range(len(decile_means))]
                            comp_mono_slope = spearman_ic(xs, decile_means)
                    _comp_row = {
                        "date": snap_date, "horizon": h, "component": col,
                        "ic": _round_opt(comp_ic, 4),
                        "ls_return": _round_opt(comp_ls, 6),
                        "n_signal": n_sig,
                        "n_with_signal": _n_with_signal if _flag_col else n_sig,
                        "mean_return_top_decile": _round_opt(comp_top_decile_ret, 6),
                        "mean_return_bottom_decile": _round_opt(comp_bot_decile_ret, 6),
                        "monotonic_slope": _round_opt(comp_mono_slope, 4),
                        "signal_stdev": _round_opt(comp_stdev, 4),
                        "pct_ties": _round_opt(comp_pct_ties, 4),
                        "flat_signal": flat_signal,
                    }
                    # Store full decile curve (d1=best-ranked .. d10=worst-ranked)
                    for _di, _dv in enumerate(decile_means if not flat_signal else [], 1):
                        _comp_row[f"d{_di}"] = _round_opt(_dv, 6)
                    # Coinvest staleness diagnostic
                    if col == "coinvest_score_z" and _coinvest_age_vals:
                        _comp_row["coinvest_staleness_days"] = _round_opt(
                            statistics.mean(_coinvest_age_vals), 1
                        )
                    component_results.append(_comp_row)

            # Coinvest audit: ticker-level view of top-20 for coinvest forensics
            if component_eval and not use_positions:
                audit_k = min(20, len(eval_tickers))
                for idx_a, t_a in enumerate(eval_tickers[:audit_k]):
                    audit_row: Dict[str, Any] = {
                        "date": snap_date,
                        "horizon": h,
                        "actionable_rank": idx_a + 1,
                        "ticker": t_a,
                        "coinvest_score_z": comp_by_ticker.get(t_a, {}).get("coinvest_score_z"),
                        "clinical_score_z_tier": comp_by_ticker.get(t_a, {}).get("clinical_score_z_tier"),
                        "alpha_cohort_raw": comp_by_ticker.get(t_a, {}).get("alpha_cohort_raw"),
                        f"fwd_return_{h}d": _round_opt(fwd_rets.get(t_a), 6),
                    }
                    coinvest_audit_rows.append(audit_row)

            # Split assignment for this date
            s_id, s_train, s_test = split_map.get(snap_date, (0, False, False))

            dr = DateResult(
                date=snap_date, horizon=h, trade_date=trade_date,
                ic=_round_opt(ic, 4),
                gross_return=_round_opt(gross, 6),
                net_return=_round_opt(net_ret, 6),
                turnover=round(turn, 4),
                n_signal=len(signal_tickers), n_held=n_held,
                coverage=round(coverage, 4),
                bottom_k_return=_round_opt(bottom_gross, 6),
                benchmark_return=_round_opt(bm_ret, 6),
                excess_return=_round_opt(excess_ret, 6),
                ls_return=_round_opt(ls_ret, 6),
                ls_net_return=_round_opt(ls_net_ret, 6),
                hedged_return=_round_opt(hedged_ret, 6),
                topk_avg_beta=_round_opt(topk_beta, 4),
                bottomk_avg_beta=_round_opt(bottomk_beta, 4),
                topk_avg_drawdown=_round_opt(topk_dd, 4),
                bottomk_avg_drawdown=_round_opt(bottomk_dd, 4),
                monotonic_slope=_round_opt(slope, 4),
                split_id=s_id, is_train=s_train, is_test=s_test,
                n_universe_eval=n_universe_eval,
                n_missing_anchor=n_missing_anchor,
                n_missing_forward=n_missing_forward,
            )
            date_results.append(dr)
            date_evaluated = True

            if ic is not None:
                horizon_ics[h].append(ic)
            if gross is not None:
                horizon_gross[h].append(gross)
            if bottom_gross is not None:
                horizon_bottom[h].append(bottom_gross)
            if net_ret is not None:
                horizon_net[h].append(net_ret)
            horizon_turnover[h].append(turn)
            # Track sign mismatches (from the enhanced check above)
            if mismatch:
                sign_mismatches[h] += 1
            if excess_ret is not None:
                horizon_excess[h].append(excess_ret)
            if ls_ret is not None:
                horizon_ls[h].append(ls_ret)
            if ls_net_ret is not None:
                horizon_ls_net[h].append(ls_net_ret)
            if hedged_ret is not None:
                horizon_hedged[h].append(hedged_ret)

        if date_evaluated:
            n_evaluated += 1

    # Build summary
    summary = EvalSummary(
        horizons=horizons, top_k=top_k, cost_bps=cost_bps,
        n_dates=len(snap_dates), n_evaluated=n_evaluated,
        n_skipped=len(snap_dates) - n_evaluated,
        use_positions=use_positions,
        anchor_mode=anchor_mode,
        benchmark=benchmark,
        split_mode=split_mode,
        universe_mode=universe_mode,
    )

    # Aggregate coinvest signal diagnostics
    if coinvest_diag_per_date:
        pct_vals = [d["pct_with_signal"] for d in coinvest_diag_per_date]

        # Pool raw tier1_count values across all dates for true percentiles
        pooled_tier1: List[int] = []
        for d in coinvest_diag_per_date:
            pooled_tier1.extend(d.get("_raw_tier1_vals", []))

        if pooled_tier1:
            pooled_sorted = sorted(pooled_tier1)
            n_pooled = len(pooled_sorted)

            def _pooled_pctl(p: float) -> int:
                idx = int(p / 100.0 * (n_pooled - 1))
                return pooled_sorted[min(idx, n_pooled - 1)]

            cross_date_dist = {
                "p50": _pooled_pctl(50),
                "p90": _pooled_pctl(90),
                "p99": _pooled_pctl(99),
                "max": pooled_sorted[-1],
                "n_total": n_pooled,
                "n_nonzero": sum(1 for v in pooled_tier1 if v > 0),
            }
        else:
            cross_date_dist = {"p50": 0, "p90": 0, "p99": 0, "max": 0, "n_total": 0, "n_nonzero": 0}

        diag_summary: Dict[str, Any] = {
            "mean_pct_with_signal": round(statistics.mean(pct_vals), 2) if pct_vals else 0.0,
            "tier1_distribution_across_dates": cross_date_dist,
        }

        # Top-K impact: Jaccard overlap when rescore is active
        if coinvest_topk_default_sets and coinvest_topk_rescored_sets:
            n_changed = 0
            overlaps: List[float] = []
            for default_set, rescored_set in zip(
                coinvest_topk_default_sets, coinvest_topk_rescored_sets
            ):
                if default_set != rescored_set:
                    n_changed += 1
                union = default_set | rescored_set
                inter = default_set & rescored_set
                jaccard = len(inter) / len(union) if union else 1.0
                overlaps.append(jaccard)
            diag_summary["topk_impact"] = {
                "changed_dates": n_changed,
                "total_dates": len(coinvest_topk_default_sets),
                "mean_overlap": round(statistics.mean(overlaps), 4) if overlaps else 1.0,
            }

        summary.coinvest_signal_diagnostics = diag_summary

    # Write component eval outputs if enabled
    if component_eval and component_results and out_dir is not None:
        _write_component_eval(component_results, horizons, out_dir)
        if coinvest_audit_rows:
            _write_coinvest_audit(coinvest_audit_rows, horizons, out_dir)

    for h in horizons:
        ics = horizon_ics[h]
        gross_list = horizon_gross[h]
        bottom_list = horizon_bottom[h]
        net_list = horizon_net[h]
        turn_list = horizon_turnover[h]
        excess_list = horizon_excess[h]
        ls_list = horizon_ls[h]
        ls_net_list = horizon_ls_net[h]
        hedged_list = horizon_hedged[h]

        bh: Dict[str, Any] = {
            "n_dates": len(ics),
            "mean_ic": _round_opt(_safe_mean(ics), 4),
            "median_ic": _round_opt(_safe_median(ics), 4),
            "std_ic": _round_opt(_safe_std(ics), 4),
            "mean_gross_return": _round_opt(_safe_mean(gross_list), 6),
            "mean_bottom_k_return": _round_opt(_safe_mean(bottom_list), 6),
            "mean_net_return": _round_opt(_safe_mean(net_list), 6),
            "cumulative_gross": _round_opt(_cumulative(gross_list), 6),
            "cumulative_net": _round_opt(_cumulative(net_list), 6),
            "mean_turnover": _round_opt(_safe_mean(turn_list), 4),
            "sign_mismatches": sign_mismatches[h],
        }

        # IC t-stat (if enough dates)
        if len(ics) >= 8:
            ic_std = _safe_std(ics)
            if ic_std and ic_std > 0:
                bh["ic_t_stat"] = _round_opt(
                    _safe_mean(ics) / (ic_std / math.sqrt(len(ics))), 3
                )

        # Benchmark-relative
        if excess_list:
            bh["mean_excess_return"] = _round_opt(_safe_mean(excess_list), 6)
            bh["cumulative_excess"] = _round_opt(_cumulative(excess_list), 6)

        # Long-short
        if ls_list:
            bh["mean_ls_return"] = _round_opt(_safe_mean(ls_list), 6)
            bh["mean_ls_net_return"] = _round_opt(_safe_mean(ls_net_list), 6)
            bh["cumulative_ls"] = _round_opt(_cumulative(ls_list), 6)
            if len(ls_list) >= 8:
                ls_std = _safe_std(ls_list)
                if ls_std and ls_std > 0:
                    bh["ls_t_stat"] = _round_opt(
                        _safe_mean(ls_list) / (ls_std / math.sqrt(len(ls_list))), 3
                    )

        # Beta-hedged
        if hedged_list:
            bh["mean_hedged_return"] = _round_opt(_safe_mean(hedged_list), 6)
            bh["cumulative_hedged"] = _round_opt(_cumulative(hedged_list), 6)

        # Residual alpha (regression of top-bottom on benchmark)
        resid = compute_residual_alpha(date_results, h)
        if resid["alpha"] is not None:
            bh["residual_alpha"] = resid["alpha"]
            bh["residual_beta"] = resid["beta"]
            bh["residual_r2"] = resid["r2"]
            bh["residual_alpha_t"] = resid["alpha_t_stat"]

        # Multi-regressor residual alpha (benchmark + beta_diff + drawdown_diff)
        resid_multi = compute_residual_alpha_multi(date_results, h)
        if resid_multi.get("alpha") is not None:
            bh["resid_multi_alpha"] = resid_multi["alpha"]
            bh["resid_multi_beta_bm"] = resid_multi["beta_benchmark"]
            bh["resid_multi_beta_bd"] = resid_multi["beta_beta_diff"]
            bh["resid_multi_beta_dd"] = resid_multi["beta_dd_diff"]
            bh["resid_multi_r2"] = resid_multi["r2"]
            bh["resid_multi_alpha_t"] = resid_multi["alpha_t_stat"]

        # Train/test split metrics (if split_mode != none)
        if split_mode != "none":
            train_dr = [dr for dr in date_results
                        if dr.horizon == h and not dr.skipped and dr.is_train]
            test_dr = [dr for dr in date_results
                       if dr.horizon == h and not dr.skipped and dr.is_test]
            train_ics = [dr.ic for dr in train_dr if dr.ic is not None]
            test_ics = [dr.ic for dr in test_dr if dr.ic is not None]
            bh["train_n"] = len(train_ics)
            bh["train_mean_ic"] = _round_opt(_safe_mean(train_ics), 4)
            bh["test_n"] = len(test_ics)
            bh["test_mean_ic"] = _round_opt(_safe_mean(test_ics), 4)
            test_gross = [dr.gross_return for dr in test_dr
                          if dr.gross_return is not None]
            bh["test_mean_gross"] = _round_opt(_safe_mean(test_gross), 6)
            bh["test_cumulative_gross"] = _round_opt(_cumulative(test_gross), 6)
            test_ls = [dr.ls_return for dr in test_dr
                       if dr.ls_return is not None]
            if test_ls:
                bh["test_mean_ls"] = _round_opt(_safe_mean(test_ls), 6)

        summary.by_horizon[h] = bh

    # --- Data freshness diagnostics ---
    price_end_str = sorted_dates[-1] if sorted_dates else None
    max_snap_str = snap_dates[-1] if snap_dates else None
    max_h = max(horizons) if horizons else 0
    fwd_stale = False
    gap_days = None
    if price_end_str and max_snap_str:
        try:
            gap_days = (
                datetime.strptime(price_end_str, "%Y-%m-%d").date()
                - datetime.strptime(max_snap_str, "%Y-%m-%d").date()
            ).days
        except ValueError:
            pass
        td = resolve_trade_date(sorted_dates, max_snap_str, anchor_mode)
        if td is not None:
            fwd_end = _trading_days_after(sorted_dates, td, max_h)
            fwd_stale = fwd_end is None
        else:
            fwd_stale = True
    summary.data_freshness = {
        "price_end_date": price_end_str,
        "max_snapshot_date": max_snap_str,
        "price_gap_days": gap_days,
        "max_horizon": max_h,
        "fwd_returns_stale": fwd_stale,
    }

    return summary, date_results, skips


def _write_component_eval(
    component_results: List[Dict[str, Any]],
    horizons: List[int],
    out_dir: Path,
) -> None:
    """Write component eval CSV and summary markdown."""
    # by_date CSV
    csv_path = out_dir / "component_eval_by_date.csv"
    fieldnames = ["date", "horizon", "component", "ic", "ls_return", "n_signal",
                  "n_with_signal",
                  "mean_return_top_decile", "mean_return_bottom_decile", "monotonic_slope",
                  "signal_stdev", "pct_ties", "flat_signal",
                  "coinvest_staleness_days"] + [f"d{i}" for i in range(1, 11)]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for row in component_results:
            writer.writerow(row)

    # Summary: aggregate per component per horizon
    from collections import defaultdict
    agg: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in component_results:
        key = (row["horizon"], row["component"])
        agg[key].append(row)

    md_path = out_dir / "component_eval_summary.md"
    lines = [
        "# Component Attribution Summary",
        "",
        "Per-component Spearman IC and decile L/S spread.",
        "",
        "| Horizon | Component | N dates | Mean IC | Median IC | Mean L/S "
        "| Mean Top-D | Mean Bot-D | Mean Slope | Mean Stdev | Mean %Ties | Flat? |",
        "|---------|-----------|---------|---------|-----------|----------"
        "|------------|------------|------------|------------|------------|-------|",
    ]
    for h in horizons:
        # Get all unique components for this horizon
        comps = sorted(set(r["component"] for r in component_results if r["horizon"] == h))
        for comp in comps:
            rows = agg[(h, comp)]
            ics = [r["ic"] for r in rows if r["ic"] is not None]
            ls_rets = [r["ls_return"] for r in rows if r["ls_return"] is not None]
            top_d = [r["mean_return_top_decile"] for r in rows if r.get("mean_return_top_decile") is not None]
            bot_d = [r["mean_return_bottom_decile"] for r in rows if r.get("mean_return_bottom_decile") is not None]
            slopes = [r["monotonic_slope"] for r in rows if r.get("monotonic_slope") is not None]
            stdevs = [r["signal_stdev"] for r in rows if r.get("signal_stdev") is not None]
            pct_ties_list = [r["pct_ties"] for r in rows if r.get("pct_ties") is not None]
            n_flat = sum(1 for r in rows if r.get("flat_signal"))
            mean_ic = statistics.mean(ics) if ics else None
            median_ic = statistics.median(ics) if ics else None
            mean_ls = statistics.mean(ls_rets) if ls_rets else None
            mean_top_d = statistics.mean(top_d) if top_d else None
            mean_bot_d = statistics.mean(bot_d) if bot_d else None
            mean_slope = statistics.mean(slopes) if slopes else None
            mean_stdev = statistics.mean(stdevs) if stdevs else None
            mean_pct_ties = statistics.mean(pct_ties_list) if pct_ties_list else None
            flat_flag = f"{n_flat}/{len(rows)}" if rows else "\u2014"
            lines.append(
                f"| {h}d | {comp} | {len(ics)} "
                f"| {_fmt(mean_ic)} | {_fmt(median_ic)} "
                f"| {_fmt_pct(mean_ls)} "
                f"| {_fmt_pct(mean_top_d)} | {_fmt_pct(mean_bot_d)} "
                f"| {_fmt(mean_slope)} "
                f"| {_fmt(mean_stdev)} | {_fmt_pct(mean_pct_ties)} "
                f"| {flat_flag} |"
            )
    lines.append("")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # --- Decile curve summary (mean return per decile, averaged across dates) ---
    decile_path = out_dir / "component_decile_curves.md"
    dclines = [
        "# Component Decile Return Curves",
        "",
        "Mean return per decile (D1=best-ranked → D10=worst-ranked), averaged across eval dates.",
        "Only shown for components with real signal dispersion (non-flat).",
        "",
    ]
    for h in horizons:
        comps = sorted(set(r["component"] for r in component_results if r["horizon"] == h))
        has_decile = [
            comp for comp in comps
            if any(f"d1" in r for r in agg[(h, comp)])
        ]
        if not has_decile:
            continue
        dclines.append(f"## Horizon {h}d")
        dclines.append("")
        header = "| Component | D1 (best) | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 (worst) | D1−D10 |"
        sep    = "|-----------|-----------|----|----|----|----|----|----|----|----|-------------|--------|"
        dclines += [header, sep]
        for comp in has_decile:
            rows_with = [r for r in agg[(h, comp)] if "d1" in r]
            if not rows_with:
                continue
            means = []
            for di in range(1, 11):
                vals = [r[f"d{di}"] for r in rows_with if r.get(f"d{di}") is not None]
                means.append(statistics.mean(vals) if vals else None)
            d1_d10 = (means[0] - means[9]) if (means[0] is not None and means[9] is not None) else None
            cells = " | ".join(_fmt_pct(m) for m in means)
            dclines.append(f"| {comp} | {cells} | {_fmt_pct(d1_d10)} |")
        dclines.append("")
    with open(decile_path, "w", encoding="utf-8") as f:
        f.write("\n".join(dclines))


def _write_coinvest_audit(
    audit_rows: List[Dict[str, Any]],
    horizons: List[int],
    out_dir: Path,
) -> None:
    """Write coinvest_audit.csv — ticker-level view of top-20 per date."""
    csv_path = out_dir / "coinvest_audit.csv"
    # Build fieldnames: fixed cols + per-horizon fwd return cols
    base_fields = ["date", "horizon", "actionable_rank", "ticker",
                   "coinvest_score_z", "clinical_score_z_tier", "alpha_cohort_raw"]
    fwd_fields = [f"fwd_return_{h}d" for h in horizons]
    fieldnames = base_fields + fwd_fields
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in audit_rows:
            writer.writerow(row)


def _round_opt(v: Optional[float], d: int) -> Optional[float]:
    return round(v, d) if v is not None else None


def _safe_mean(vals: List[float]) -> Optional[float]:
    return statistics.mean(vals) if vals else None


def _safe_median(vals: List[float]) -> Optional[float]:
    return statistics.median(vals) if vals else None


def _safe_std(vals: List[float]) -> Optional[float]:
    if len(vals) < 2:
        return None
    return statistics.stdev(vals)


def _cumulative(returns: List[float]) -> Optional[float]:
    if not returns:
        return None
    cum = 1.0
    for r in returns:
        cum *= (1.0 + r)
    return cum - 1.0


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_summary_json(summary: EvalSummary, out_dir: Path) -> Path:
    path = out_dir / "summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2, default=str)
    return path


def write_summary_md(summary: EvalSummary, out_dir: Path) -> Path:
    path = out_dir / "summary.md"
    lines = [
        "# Forward-Return Evaluation Summary",
        "",
        f"- **Dates evaluated**: {summary.n_evaluated} / {summary.n_dates}",
        f"- **Skipped**: {summary.n_skipped}",
        f"- **Top-K**: {summary.top_k}",
        f"- **Cost (bps)**: {summary.cost_bps}",
        f"- **Anchor mode**: {summary.anchor_mode}",
        f"- **Benchmark**: {summary.benchmark}",
        f"- **Positions mode**: {summary.use_positions}",
        f"- **Universe mode**: {summary.universe_mode}",
        "",
        "## Execution Convention",
        "",
        f"Signal computed as of calendar date D (the snapshot `as_of_date`).",
    ]
    if summary.anchor_mode == "next_trading_day":
        lines.append(
            "Trades executed on the **first trading day after D** (PIT-safe: "
            "no lookahead since information is known at D, execution is next open)."
        )
    elif summary.anchor_mode == "prev_trading_day":
        lines.append(
            "Trades executed on the **last trading day on/before D** "
            "(assumes signal known at market close on D)."
        )
    else:
        lines.append(
            "Trades anchored at **exact date D** (requires D to be a trading day)."
        )
    lines.append("")

    # Core table
    lines.append("## By Horizon")
    lines.append("")
    lines.append(
        "| Horizon | N | Mean IC | Median IC | Std IC | IC t | "
        "Mean Gross | Mean Net | Cum Gross | Cum Net | Mean Turn |"
    )
    lines.append(
        "|---------|---|---------|-----------|--------|------|"
        "-----------|----------|-----------|---------|-----------|"
    )
    for h in summary.horizons:
        bh = summary.by_horizon.get(h, {})
        lines.append(
            f"| {h}d | {bh.get('n_dates', 0)} "
            f"| {_fmt(bh.get('mean_ic'))} "
            f"| {_fmt(bh.get('median_ic'))} "
            f"| {_fmt(bh.get('std_ic'))} "
            f"| {_fmt(bh.get('ic_t_stat'))} "
            f"| {_fmt_pct(bh.get('mean_gross_return'))} "
            f"| {_fmt_pct(bh.get('mean_net_return'))} "
            f"| {_fmt_pct(bh.get('cumulative_gross'))} "
            f"| {_fmt_pct(bh.get('cumulative_net'))} "
            f"| {_fmt(bh.get('mean_turnover'))} |"
        )
    lines.append("")

    # Sign consistency check table
    lines.append("## Sign Consistency (Top-K vs Bottom-K)")
    lines.append("")
    lines.append(
        "| Horizon | Mean Top-K | Mean Bottom-K | Top-Bottom | Mean IC | "
        "Consistent | Mismatches |"
    )
    lines.append(
        "|---------|------------|---------------|------------|---------|"
        "------------|------------|"
    )
    for h in summary.horizons:
        bh = summary.by_horizon.get(h, {})
        top_k_ret = bh.get("mean_gross_return")
        bot_k_ret = bh.get("mean_bottom_k_return")
        mean_ic = bh.get("mean_ic")
        tmb = None
        if top_k_ret is not None and bot_k_ret is not None:
            tmb = top_k_ret - bot_k_ret
        consistent = ""
        if tmb is not None and mean_ic is not None:
            if (mean_ic >= 0 and tmb >= 0) or (mean_ic < 0 and tmb < 0):
                consistent = "YES"
            else:
                consistent = "**NO**"
        lines.append(
            f"| {h}d | {_fmt_pct(top_k_ret)} "
            f"| {_fmt_pct(bot_k_ret)} "
            f"| {_fmt_pct(tmb)} "
            f"| {_fmt(mean_ic)} "
            f"| {consistent} "
            f"| {bh.get('sign_mismatches', 0)} |"
        )
    lines.append("")

    # Benchmark table (if present)
    has_excess = any(
        "mean_excess_return" in summary.by_horizon.get(h, {}) for h in summary.horizons
    )
    if has_excess:
        lines.append(f"## Benchmark-Relative ({summary.benchmark})")
        lines.append("")
        lines.append("| Horizon | Mean Excess | Cum Excess |")
        lines.append("|---------|-------------|------------|")
        for h in summary.horizons:
            bh = summary.by_horizon.get(h, {})
            lines.append(
                f"| {h}d | {_fmt_pct(bh.get('mean_excess_return'))} "
                f"| {_fmt_pct(bh.get('cumulative_excess'))} |"
            )
        lines.append("")

    # Long-short table (if present)
    has_ls = any(
        "mean_ls_return" in summary.by_horizon.get(h, {}) for h in summary.horizons
    )
    if has_ls:
        lines.append("## Long-Short Decile Spread")
        lines.append("")
        lines.append("| Horizon | Mean L/S | Mean L/S Net | Cum L/S | L/S t |")
        lines.append("|---------|----------|--------------|---------|-------|")
        for h in summary.horizons:
            bh = summary.by_horizon.get(h, {})
            lines.append(
                f"| {h}d | {_fmt_pct(bh.get('mean_ls_return'))} "
                f"| {_fmt_pct(bh.get('mean_ls_net_return'))} "
                f"| {_fmt_pct(bh.get('cumulative_ls'))} "
                f"| {_fmt(bh.get('ls_t_stat'))} |"
            )
        lines.append("")

    # Beta-hedged table (if present)
    has_hedged = any(
        "mean_hedged_return" in summary.by_horizon.get(h, {}) for h in summary.horizons
    )
    if has_hedged:
        lines.append("## Beta-Hedged Portfolio")
        lines.append("")
        lines.append("| Horizon | Mean Hedged | Cum Hedged |")
        lines.append("|---------|-------------|------------|")
        for h in summary.horizons:
            bh = summary.by_horizon.get(h, {})
            lines.append(
                f"| {h}d | {_fmt_pct(bh.get('mean_hedged_return'))} "
                f"| {_fmt_pct(bh.get('cumulative_hedged'))} |"
            )
        lines.append("")

    # Residual alpha table (if present)
    has_resid = any(
        "residual_alpha" in summary.by_horizon.get(h, {}) for h in summary.horizons
    )
    if has_resid:
        lines.append("## Residual Alpha (Top-Bottom regressed on Benchmark)")
        lines.append("")
        lines.append(
            "| Horizon | Alpha (ann.) | Beta | R² | Alpha t |"
        )
        lines.append(
            "|---------|-------------|------|-----|---------|"
        )
        for h in summary.horizons:
            bh = summary.by_horizon.get(h, {})
            alpha = bh.get("residual_alpha")
            alpha_ann = f"{alpha * 12:.2%}" if alpha is not None else "\u2014"
            lines.append(
                f"| {h}d | {alpha_ann} "
                f"| {_fmt(bh.get('residual_beta'))} "
                f"| {_fmt(bh.get('residual_r2'))} "
                f"| {_fmt(bh.get('residual_alpha_t'))} |"
            )
        lines.append("")
        lines.append(
            "> Residual alpha = intercept from OLS: (TopK - BottomK) = α + β × BenchmarkReturn. "
            "Annualized by × 12 (monthly)."
        )
        lines.append("")

    # Multi-regressor attribution (if present)
    has_multi = any(
        "resid_multi_alpha" in summary.by_horizon.get(h, {}) for h in summary.horizons
    )
    if has_multi:
        lines.append("## Attribution (Multi-Regressor)")
        lines.append("")
        lines.append(
            "| Horizon | Alpha (ann.) | β(BM) | β(Δbeta) | β(Δdd) | R² | Alpha t |"
        )
        lines.append(
            "|---------|-------------|-------|----------|--------|-----|---------|"
        )
        for h in summary.horizons:
            bh = summary.by_horizon.get(h, {})
            alpha = bh.get("resid_multi_alpha")
            alpha_ann = f"{alpha * 12:.2%}" if alpha is not None else "\u2014"
            lines.append(
                f"| {h}d | {alpha_ann} "
                f"| {_fmt(bh.get('resid_multi_beta_bm'))} "
                f"| {_fmt(bh.get('resid_multi_beta_bd'))} "
                f"| {_fmt(bh.get('resid_multi_beta_dd'))} "
                f"| {_fmt(bh.get('resid_multi_r2'))} "
                f"| {_fmt(bh.get('resid_multi_alpha_t'))} |"
            )
        lines.append("")
        lines.append(
            "> L/S = α + β₁·BenchmarkRet + β₂·(TopBeta−BotBeta) + β₃·(TopDD−BotDD). "
            "Annualized by × 12."
        )
        lines.append("")

    # Train/test split metrics (if present)
    has_split = any(
        "test_n" in summary.by_horizon.get(h, {}) for h in summary.horizons
    )
    if has_split:
        lines.append(f"## Walk-Forward Split ({summary.split_mode})")
        lines.append("")
        lines.append(
            "| Horizon | Train N | Train IC | Test N | Test IC | "
            "Test Gross | Test Cum | Test L/S |"
        )
        lines.append(
            "|---------|---------|----------|--------|---------|"
            "-----------|----------|----------|"
        )
        for h in summary.horizons:
            bh = summary.by_horizon.get(h, {})
            lines.append(
                f"| {h}d | {bh.get('train_n', 0)} "
                f"| {_fmt(bh.get('train_mean_ic'))} "
                f"| {bh.get('test_n', 0)} "
                f"| {_fmt(bh.get('test_mean_ic'))} "
                f"| {_fmt_pct(bh.get('test_mean_gross'))} "
                f"| {_fmt_pct(bh.get('test_cumulative_gross'))} "
                f"| {_fmt_pct(bh.get('test_mean_ls'))} |"
            )
        lines.append("")

    # Interpretation Notes
    lines.append("## Interpretation Notes")
    lines.append("")
    lines.append("### Signal Direction")
    lines.append(
        "- **Signal**: `signal = -actionable_rank`. Rank 1 (best) maps to highest signal."
    )
    lines.append(
        "- **IC > 0**: better-ranked tickers tend to have higher forward returns."
    )
    lines.append(
        "- **Top-K**: first K tickers by actionable_rank (rank 1..K)."
    )
    lines.append(
        "- **Bottom-K**: last K tickers by actionable_rank."
    )
    lines.append(
        "- **L/S (decile spread)**: Top 10% return minus Bottom 10% return."
    )
    lines.append("")
    lines.append("### Anchor Convention")
    if summary.anchor_mode == "next_trading_day":
        lines.append(
            "- **next_trading_day**: calendar as-of date D maps to the "
            "first trading day after D. PIT-safe: information available at D, "
            "execution at next market open."
        )
    lines.append("")
    lines.append("### SIGN_MISMATCH")
    lines.append(
        "- Fired when IC > 0 but TopK < BottomK, or when IC and monotonic slope "
        "disagree. Indicates non-monotonic signal-return relationship or "
        "exposure confound at the tails."
    )
    lines.append("")
    lines.append("### PIT Enforcement")
    lines.append(
        "- All bundles use PIT-filtered data: CTgov trials `first_posted <= as_of_date`, "
        "13F filings `filing_date <= as_of_date`, catalyst events `disclosed_at <= as_of_date`."
    )
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def write_by_date_csv(date_results: List[DateResult], out_dir: Path) -> Path:
    path = out_dir / "by_date.csv"
    fieldnames = [
        "date", "horizon", "trade_date", "ic", "gross_return", "bottom_k_return",
        "net_return", "turnover", "n_signal", "n_held", "coverage",
        "skipped", "skip_reason",
        "benchmark_return", "excess_return",
        "ls_return", "ls_net_return", "hedged_return",
        "topk_avg_beta", "bottomk_avg_beta",
        "topk_avg_drawdown", "bottomk_avg_drawdown",
        "monotonic_slope",
        "split_id", "is_train", "is_test",
        "n_universe_eval", "n_missing_anchor", "n_missing_forward",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for dr in date_results:
            writer.writerow(asdict(dr))
    return path


def write_skips_json(skips: List[Dict[str, str]], out_dir: Path) -> Path:
    path = out_dir / "skips.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(skips, f, indent=2)
    return path


def _fmt(v: Optional[float]) -> str:
    return f"{v:.4f}" if v is not None else "\u2014"


def _fmt_pct(v: Optional[float]) -> str:
    return f"{v:.2%}" if v is not None else "\u2014"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Forward-return evaluation across snapshot dates",
    )
    parser.add_argument(
        "--snapshot-root", type=Path,
        default=PROJECT_ROOT / "data" / "snapshots",
        help="Root dir containing YYYY-MM-DD snapshot dirs",
    )
    parser.add_argument(
        "--price-csv", type=Path,
        default=PROJECT_ROOT / "production_data" / "price_history.csv",
        help="Path to price_history.csv",
    )
    parser.add_argument("--date-from", type=str, default=None)
    parser.add_argument("--date-to", type=str, default=None)
    parser.add_argument(
        "--horizons", type=str, default="5,20,63",
        help="Comma-separated forward-return horizons in trading days",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument(
        "--pit-mode", choices=["strict", "lenient"], default="strict",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=PROJECT_ROOT / "output" / "forward_eval",
    )
    parser.add_argument("--use-positions", action="store_true", default=False)
    parser.add_argument(
        "--min-price-coverage", type=float, default=DEFAULT_MIN_PRICE_COVERAGE,
    )
    parser.add_argument(
        "--anchor-mode",
        choices=["exact", "next_trading_day", "prev_trading_day"],
        default="exact",
        help="How to resolve non-trading as-of dates to trade dates",
    )
    parser.add_argument(
        "--benchmark",
        default="none",
        help="Benchmark ticker for excess returns (e.g. XBI, SPY, none)",
    )
    parser.add_argument(
        "--long-short-deciles", action="store_true", default=False,
        help="Compute top-decile minus bottom-decile spread",
    )
    parser.add_argument(
        "--split-mode",
        choices=["none", "fixed", "walk_forward"],
        default="none",
        help="Walk-forward split mode: none, fixed, or walk_forward",
    )
    parser.add_argument(
        "--train-end", type=str, default=None,
        help="Last date in training set (fixed split)",
    )
    parser.add_argument(
        "--test-start", type=str, default=None,
        help="First date in test set (fixed split)",
    )
    parser.add_argument(
        "--wf-train-months", type=int, default=0,
        help="Number of initial dates for training (walk_forward)",
    )
    parser.add_argument(
        "--wf-test-months", type=int, default=0,
        help="Dates per test fold (walk_forward)",
    )
    parser.add_argument(
        "--universe-mode",
        choices=["current", "price_available"],
        default="current",
        help="Eval universe: current (rankings as-is) or price_available (filter by price)",
    )
    parser.add_argument(
        "--component-eval", action="store_true", default=False,
        help="Compute per-component IC and L/S",
    )
    parser.add_argument(
        "--rebalance-buffer-ranks", type=int, default=0,
        help="Buffer zone around top-K for rebalance (research-only)",
    )
    parser.add_argument(
        "--turnover-cap", type=float, default=0.0,
        help="Max turnover fraction per rebalance, 0=unlimited (research-only)",
    )
    parser.add_argument(
        "--top-quantile", type=float, default=0.0,
        help="Use top N%% instead of top-K, 0=use top_k (research-only)",
    )
    parser.add_argument(
        "--coinvest-eval-mode",
        choices=["default", "off", "contra"],
        default="default",
        help="Coinvest signal mode: default (as-is), off (zero), contra (flip sign)",
    )
    parser.add_argument(
        "--catalyst-eval-mode",
        choices=["default", "logistic"],
        default="default",
        help="Catalyst decay: default (CSV values) or logistic (recompute from catalyst_days)",
    )
    parser.add_argument(
        "--ruleset", type=Path, default=None,
        help=(
            "DecisionRuleset JSON. Affects: (1) rebalance_buffer_ranks inheritance, "
            "(2) coinvest rescoring when --coinvest-eval-mode != default. "
            "NOTE: sort-weight changes (tiebreak, clinical, calendar alpha) require "
            "--rerank in run_audited_backtest.py — passing --ruleset alone does NOT re-sort."
        ),
    )
    parser.add_argument(
        "--fail-if-stale", action="store_true", default=False,
        help="Exit 2 if forward-return price data is stale",
    )
    parser.add_argument(
        "--no-warn-if-stale", action="store_true", default=False,
        help="Suppress staleness warning when price data is short",
    )
    parser.add_argument(
        "--preflight", action="store_true", default=False,
        help="Run snapshot preflight; skip dates that FAIL structural checks",
    )
    parser.add_argument(
        "--preflight-strict", action="store_true", default=False,
        help="Run snapshot preflight; skip dates that FAIL or WARN",
    )
    args = parser.parse_args()

    horizons = [int(h.strip()) for h in args.horizons.split(",")]

    # Load ruleset if provided or required
    eval_ruleset: Optional[DecisionRuleset] = None
    if args.ruleset:
        eval_ruleset = DecisionRuleset.from_json(str(args.ruleset))
    elif args.coinvest_eval_mode != "default":
        print("ERROR: --ruleset is required when --coinvest-eval-mode is not 'default'.",
              file=sys.stderr)
        print("  The eval script cannot infer which ruleset to use for faithful rescore.",
              file=sys.stderr)
        print("  Provide the ruleset JSON that matches the snapshots being evaluated.",
              file=sys.stderr)
        sys.exit(1)

    # Inherit portfolio mechanics from ruleset when CLI uses defaults
    eff_buffer = args.rebalance_buffer_ranks
    if eval_ruleset and args.rebalance_buffer_ranks == 0 and eval_ruleset.rebalance_buffer_ranks > 0:
        eff_buffer = eval_ruleset.rebalance_buffer_ranks

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating snapshots in {args.snapshot_root}")
    print(f"  Horizons: {horizons}, Top-K: {args.top_k}, Cost: {args.cost_bps} bps")
    print(f"  Anchor: {args.anchor_mode}, Benchmark: {args.benchmark}")
    if args.split_mode != "none":
        print(f"  Split: {args.split_mode}")
    if args.universe_mode != "current":
        print(f"  Universe: {args.universe_mode}")
    if args.component_eval:
        print("  Component eval: enabled")
    if eff_buffer > 0:
        print(f"  Rebalance buffer: {eff_buffer} ranks"
              + (" (from ruleset)" if eff_buffer != args.rebalance_buffer_ranks else ""))
    if args.turnover_cap > 0:
        print(f"  Turnover cap: {args.turnover_cap:.0%}")
    if args.top_quantile > 0:
        print(f"  Top quantile: {args.top_quantile:.0%}")
    if args.coinvest_eval_mode != "default":
        print(f"  Coinvest eval mode: {args.coinvest_eval_mode}")
        if eval_ruleset:
            print(f"  Ruleset: {args.ruleset} (ID: {eval_ruleset.ruleset_id})")
    if args.catalyst_eval_mode != "default":
        print(f"  Catalyst eval mode: {args.catalyst_eval_mode}")

    summary, date_results, skips = evaluate(
        snapshot_root=args.snapshot_root,
        price_csv=args.price_csv,
        horizons=horizons,
        top_k=args.top_k,
        cost_bps=args.cost_bps,
        date_from=args.date_from,
        date_to=args.date_to,
        pit_mode=args.pit_mode,
        use_positions=args.use_positions,
        min_price_coverage=args.min_price_coverage,
        anchor_mode=args.anchor_mode,
        benchmark=args.benchmark,
        long_short_deciles=args.long_short_deciles,
        split_mode=args.split_mode,
        train_end=args.train_end,
        test_start=args.test_start,
        wf_train_months=args.wf_train_months,
        wf_test_months=args.wf_test_months,
        out_dir=args.out_dir,
        universe_mode=args.universe_mode,
        component_eval=args.component_eval,
        rebalance_buffer_ranks=eff_buffer,
        turnover_cap=args.turnover_cap,
        top_quantile=args.top_quantile,
        coinvest_eval_mode=args.coinvest_eval_mode,
        catalyst_eval_mode=args.catalyst_eval_mode,
        ruleset=eval_ruleset,
        preflight=args.preflight,
        preflight_strict=args.preflight_strict,
    )

    write_summary_json(summary, args.out_dir)
    write_summary_md(summary, args.out_dir)
    write_by_date_csv(date_results, args.out_dir)
    write_skips_json(skips, args.out_dir)

    print(f"\nResults -> {args.out_dir}")
    print(f"  {summary.n_evaluated}/{summary.n_dates} dates evaluated, "
          f"{summary.n_skipped} skipped")
    for h in horizons:
        bh = summary.by_horizon.get(h, {})
        parts = [
            f"IC={_fmt(bh.get('mean_ic'))}",
            f"Gross={_fmt_pct(bh.get('mean_gross_return'))}",
            f"Net={_fmt_pct(bh.get('mean_net_return'))}",
            f"Turn={_fmt(bh.get('mean_turnover'))}",
        ]
        if "mean_excess_return" in bh:
            parts.append(f"Excess={_fmt_pct(bh.get('mean_excess_return'))}")
        if "mean_ls_return" in bh:
            parts.append(f"L/S={_fmt_pct(bh.get('mean_ls_return'))}")
        if "mean_hedged_return" in bh:
            parts.append(f"Hedged={_fmt_pct(bh.get('mean_hedged_return'))}")
        print(f"  {h}d: {', '.join(parts)}")

    # --- Staleness guardrail ---
    status = _staleness_status(summary)
    if status["stale"]:
        if args.fail_if_stale:
            print(f"FAIL: {status['message']}", file=sys.stderr)
            sys.exit(2)
        elif not args.no_warn_if_stale:
            log.warning(status["message"])


if __name__ == "__main__":
    main()
