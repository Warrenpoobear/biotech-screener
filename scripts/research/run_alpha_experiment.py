#!/usr/bin/env python3
"""Alpha experiment runner: one-command baseline vs exposure-neutralized evaluation.

RESEARCH ONLY — not for production use.

Runs a baseline forward-return evaluation and optionally an exposure-neutralized
variant, then generates a comparison report with IC, returns, exposure shifts,
and a promote/keep recommendation.

Exposure neutralization works by:
1. Extracting the anchor signal (alpha_cohort_pct or composite_rank) per row
2. Regressing anchor ~ intercept + beta + drawdown + vol + mcap (cross-sectional OLS)
3. Taking residuals (signal unexplained by exposures)
4. Percentile-ranking residuals → new tiebreaker_pct
5. Re-ranking with the production sort key using neutralized signal

Examples:
    # Baseline only (last 3 months)
    python scripts/research/run_alpha_experiment.py \\
        --experiment-name baseline_check \\
        --mode baseline \\
        --date-from 2026-01-15

    # Exposure-neutralized + comparison report
    python scripts/research/run_alpha_experiment.py \\
        --experiment-name beta_neutral_v1 \\
        --mode compare \\
        --exposures beta,drawdown \\
        --date-from 2026-01-15

    # Full neutralization with all exposures
    python scripts/research/run_alpha_experiment.py \\
        --experiment-name full_neutral \\
        --mode compare \\
        --exposures beta,drawdown,vol,mcap \\
        --date-from 2025-10-31 --date-to 2026-02-24

    # Custom top-K and horizons
    python scripts/research/run_alpha_experiment.py \\
        --experiment-name topk60_neutral \\
        --mode compare \\
        --exposures beta,drawdown \\
        --top-k 60 --horizons 5,20 \\
        --date-from 2026-01-15
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.ranking_utils import backfill_columns, safe_float
from decision_engine import DecisionRuleset, compute_actionable_sort_key
from dataclasses import replace as dc_replace

# Reuse building blocks from eval_forward_returns (import at function scope
# to avoid heavy top-level import if module isn't on sys.path yet)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_HORIZONS = [5, 20, 63]
DEFAULT_TOP_K = 20
DEFAULT_COST_BPS = 30

# Map short exposure names → rankings.csv column names
EXPOSURE_MAP = {
    "beta": "de_beta_xbi_60d",
    "drawdown": "de_drawdown",
    "vol": "de_rsi_14d",
    "mcap": "market_cap_bucket",
}

# Ordinal encoding for market_cap_bucket
MCAP_ORDINAL = {"xs": 1.0, "small": 2.0, "mid": 3.0, "large": 4.0}


# ---------------------------------------------------------------------------
# Data loading (lightweight, no pandas)
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


def load_rankings(snapshot_dir: Path) -> List[Dict[str, str]]:
    """Load rankings.csv from a snapshot directory."""
    csv_path = snapshot_dir / "rankings.csv"
    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def discover_snapshot_dates(
    snapshot_root: Path,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> List[str]:
    """Return sorted list of YYYY-MM-DD snapshot dates with rankings.csv."""
    dates: List[str] = []
    if not snapshot_root.exists():
        return dates
    for d in sorted(snapshot_root.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if len(name) != 10 or name[4] != "-" or name[7] != "-":
            continue
        if not (d / "rankings.csv").exists():
            continue
        dates.append(name)
    if date_from:
        dates = [d for d in dates if d >= date_from]
    if date_to:
        dates = [d for d in dates if d <= date_to]
    return dates


# ---------------------------------------------------------------------------
# Forward returns & IC (minimal reimplementation of eval_forward_returns)
# ---------------------------------------------------------------------------

def _trading_days_after(all_dates: List[str], start: str, n: int) -> Optional[str]:
    """Return the trading date n trading days after start, or None."""
    try:
        idx = all_dates.index(start)
    except ValueError:
        return None
    target = idx + n
    if target < len(all_dates):
        return all_dates[target]
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


def top_k_portfolio_return(
    tickers_ranked: List[str],
    fwd_rets: Dict[str, float],
    k: int,
) -> Tuple[Optional[float], int, List[str]]:
    """Equal-weight top-K gross return. Returns (mean_return, n_held, held_tickers)."""
    held = [t for t in tickers_ranked[:k] if t in fwd_rets]
    if not held:
        return None, 0, []
    ret = statistics.mean(fwd_rets[t] for t in held)
    return ret, len(held), held


def compute_turnover(prev_set: List[str], curr_set: List[str]) -> float:
    """Turnover = 0.5 * |symmetric difference| / max(len(prev), len(curr), 1)."""
    if not prev_set and not curr_set:
        return 0.0
    s_prev = set(prev_set)
    s_curr = set(curr_set)
    diff = len(s_prev ^ s_curr)
    denom = max(len(s_prev), len(s_curr), 1)
    return 0.5 * diff / denom


# ---------------------------------------------------------------------------
# OLS (deterministic, no numpy — identical to eval_forward_returns._multi_ols)
# ---------------------------------------------------------------------------

def multi_ols(
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
            for k_idx in range(i, p + 1):
                aug[j][k_idx] -= factor * aug[i][k_idx]

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


# ---------------------------------------------------------------------------
# Exposure extraction
# ---------------------------------------------------------------------------

def _extract_exposure(row: Dict[str, str], exposure_name: str) -> Optional[float]:
    """Extract a single exposure value from a rankings row."""
    col = EXPOSURE_MAP.get(exposure_name)
    if col is None:
        return None

    if exposure_name == "mcap":
        # Ordinal encoding for market_cap_bucket
        bucket = (row.get(col) or "").lower().strip()
        return MCAP_ORDINAL.get(bucket)

    raw = row.get(col, "")
    return safe_float(raw)


def _extract_anchor(row: Dict[str, str], ruleset: DecisionRuleset) -> Optional[float]:
    """Extract the anchor signal value from a row (higher = better)."""
    if ruleset.sort_anchor in ("optionality_pct", "alpha_cohort"):
        return safe_float(row.get("alpha_cohort_pct"))
    else:
        # composite_rank: lower rank = better → negate
        cr = safe_float(row.get("composite_rank"))
        return -cr if cr is not None else None


# ---------------------------------------------------------------------------
# Exposure neutralization
# ---------------------------------------------------------------------------

def neutralize_exposures(
    rows: List[Dict[str, str]],
    exposure_names: List[str],
    ruleset: DecisionRuleset,
) -> Tuple[List[Dict[str, str]], Optional[float], List[float]]:
    """Neutralize the anchor signal against exposure factors.

    For each eligible row:
    1. Extract anchor signal and exposure values
    2. Cross-sectional OLS: anchor ~ intercept + exposures
    3. Residual = anchor - predicted
    4. Percentile-rank residuals → new alpha_cohort_pct

    Args:
        rows: Rankings rows (will be deep-copied and modified).
        exposure_names: List of exposure short names (e.g., ["beta", "drawdown"]).
        ruleset: DecisionRuleset for anchor extraction.

    Returns:
        (modified_rows, r_squared, ols_coefficients)
    """
    rows = copy.deepcopy(rows)

    # Collect eligible rows with valid anchor + all exposures
    valid_indices: List[int] = []
    anchors: List[float] = []
    exposure_vals: List[List[float]] = [[] for _ in exposure_names]

    for i, row in enumerate(rows):
        if row.get("eligible") != "1":
            continue

        anchor = _extract_anchor(row, ruleset)
        if anchor is None:
            continue

        exps = []
        skip = False
        for j, exp_name in enumerate(exposure_names):
            v = _extract_exposure(row, exp_name)
            if v is None:
                skip = True
                break
            exps.append(v)
        if skip:
            continue

        valid_indices.append(i)
        anchors.append(anchor)
        for j, v in enumerate(exps):
            exposure_vals[j].append(v)

    if len(valid_indices) < len(exposure_names) + 3:
        # Not enough data for OLS — return unmodified
        return rows, None, []

    # Run OLS
    beta, r2 = multi_ols(exposure_vals, anchors)

    # Compute residuals
    n = len(valid_indices)
    p = len(exposure_names) + 1
    residuals = []
    for idx in range(n):
        predicted = beta[0]  # intercept
        for j in range(len(exposure_names)):
            predicted += beta[j + 1] * exposure_vals[j][idx]
        residuals.append(anchors[idx] - predicted)

    # Percentile-rank residuals (0-1, higher = better)
    ranked = _avg_ranks(residuals)
    pct = [(r - 1.0) / max(n - 1, 1) for r in ranked]

    # Inject neutralized percentile as alpha_cohort_pct
    for idx in range(n):
        row_idx = valid_indices[idx]
        rows[row_idx]["alpha_cohort_pct"] = str(round(pct[idx], 6))

    # For rows not in valid set (ineligible or missing data), leave as-is
    return rows, r2, beta


# ---------------------------------------------------------------------------
# Faithful reranking (same pattern as rerank_snapshots.py)
# ---------------------------------------------------------------------------

def rerank_faithful(
    rows: List[Dict[str, str]],
    ruleset: DecisionRuleset,
) -> List[Dict[str, str]]:
    """Re-sort rows using production-identical sort key and assign fresh ranks."""
    backfill_columns(rows)

    rows.sort(key=lambda r: compute_actionable_sort_key(
        decision_fields=r,
        archetype=r.get("archetype", ""),
        optionality=safe_float(r.get("clinical_optionality_pct_dev")),
        composite_rank=r.get("composite_rank"),
        ticker=r.get("ticker", ""),
        catalyst_event_type=r.get("catalyst_event_type", ""),
        catalyst_source=r.get("catalyst_source", ""),
        ruleset=ruleset,
        tiebreaker_pct=(
            safe_float(r.get("alpha_cohort_pct"))
            if ruleset.sort_anchor == "alpha_cohort"
            else (safe_float(r.get("commercial_quality_pct"))
                  if r.get("archetype", "").startswith("commercial_")
                  else safe_float(r.get("clinical_optionality_pct_dev")))
        ),
    ))

    rank = 1
    for r in rows:
        if r.get("eligible") == "1":
            r["actionable_rank"] = str(rank)
            rank += 1
        else:
            r["actionable_rank"] = ""

    return rows


# ---------------------------------------------------------------------------
# Per-date evaluation
# ---------------------------------------------------------------------------

@dataclass
class DateMetrics:
    """Metrics for a single snapshot date."""
    date: str
    n_eligible: int = 0
    n_valid_for_ols: int = 0
    ics: Dict[int, Optional[float]] = field(default_factory=dict)
    gross_returns: Dict[int, Optional[float]] = field(default_factory=dict)
    n_held: Dict[int, int] = field(default_factory=dict)
    top_k_tickers: List[str] = field(default_factory=list)
    exposure_means_topk: Dict[str, Optional[float]] = field(default_factory=dict)
    ols_r2: Optional[float] = None
    ols_coefficients: List[float] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""


def evaluate_single_date(
    snap_date: str,
    rankings: List[Dict[str, str]],
    prices: Dict[str, Dict[str, float]],
    sorted_dates: List[str],
    horizons: List[int],
    top_k: int,
    exposure_names: List[str],
) -> DateMetrics:
    """Evaluate a single snapshot date: IC, top-K return, exposure means."""
    metrics = DateMetrics(date=snap_date)

    # Count eligible
    eligible_rows = [r for r in rankings if r.get("eligible") == "1"]
    metrics.n_eligible = len(eligible_rows)

    if not eligible_rows:
        metrics.skipped = True
        metrics.skip_reason = "NO_ELIGIBLE_ROWS"
        return metrics

    # Build ranked ticker list (by actionable_rank)
    def _rank_key(r: Dict[str, str]) -> int:
        try:
            return int(r.get("actionable_rank") or 9999)
        except (ValueError, TypeError):
            return 9999

    ranked_eligible = sorted(eligible_rows, key=_rank_key)
    tickers_ranked = [r.get("ticker", "") for r in ranked_eligible]

    # Resolve trade date (prev_trading_day for PIT safety)
    trade_date = None
    if snap_date in sorted_dates:
        trade_date = snap_date
    else:
        # Find last trading date on or before snap_date
        for d in sorted_dates:
            if d <= snap_date:
                trade_date = d
            else:
                break

    if trade_date is None:
        metrics.skipped = True
        metrics.skip_reason = "NO_TRADE_DATE"
        return metrics

    # Per-horizon evaluation
    for h in horizons:
        # Build forward returns
        fwd_rets: Dict[str, float] = {}
        for t in tickers_ranked:
            if t not in prices:
                continue
            ret = compute_forward_return(prices[t], sorted_dates, trade_date, h)
            if ret is not None:
                fwd_rets[t] = ret

        # IC: signal = -position (lower rank = better = higher signal)
        signal_tickers = [t for t in tickers_ranked if t in fwd_rets]
        if not signal_tickers:
            metrics.ics[h] = None
            metrics.gross_returns[h] = None
            metrics.n_held[h] = 0
            continue

        signal_vals = [-float(i + 1) for i, t in enumerate(tickers_ranked)
                       if t in fwd_rets]
        return_vals = [fwd_rets[t] for t in signal_tickers]
        metrics.ics[h] = spearman_ic(signal_vals, return_vals)

        # Top-K return
        ret, n, held = top_k_portfolio_return(tickers_ranked, fwd_rets, top_k)
        metrics.gross_returns[h] = ret
        metrics.n_held[h] = n

        # Store top-K tickers (from first horizon only)
        if not metrics.top_k_tickers:
            metrics.top_k_tickers = held

    # Exposure means of top-K
    topk_set = set(tickers_ranked[:top_k])
    topk_rows = [r for r in ranked_eligible if r.get("ticker", "") in topk_set]
    for exp_name in exposure_names:
        vals = []
        for r in topk_rows:
            v = _extract_exposure(r, exp_name)
            if v is not None:
                vals.append(v)
        metrics.exposure_means_topk[exp_name] = (
            statistics.mean(vals) if vals else None
        )

    return metrics


# ---------------------------------------------------------------------------
# Experiment result
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """Aggregated result of a full experiment run."""
    name: str
    mode: str
    n_dates: int = 0
    n_skipped: int = 0
    horizons: List[int] = field(default_factory=list)
    top_k: int = DEFAULT_TOP_K
    exposure_names: List[str] = field(default_factory=list)
    ruleset_id: str = ""
    date_metrics: List[DateMetrics] = field(default_factory=list)

    # Aggregated
    mean_ic: Dict[int, Optional[float]] = field(default_factory=dict)
    mean_gross: Dict[int, Optional[float]] = field(default_factory=dict)
    mean_exposure_topk: Dict[str, Optional[float]] = field(default_factory=dict)
    mean_ols_r2: Optional[float] = None

    def aggregate(self) -> None:
        """Compute aggregate statistics from date_metrics."""
        valid = [m for m in self.date_metrics if not m.skipped]
        self.n_dates = len(valid)
        self.n_skipped = sum(1 for m in self.date_metrics if m.skipped)

        for h in self.horizons:
            ics = [m.ics[h] for m in valid if m.ics.get(h) is not None]
            self.mean_ic[h] = statistics.mean(ics) if ics else None

            gross = [m.gross_returns[h] for m in valid
                     if m.gross_returns.get(h) is not None]
            self.mean_gross[h] = statistics.mean(gross) if gross else None

        for exp in self.exposure_names:
            vals = [m.exposure_means_topk[exp] for m in valid
                    if m.exposure_means_topk.get(exp) is not None]
            self.mean_exposure_topk[exp] = statistics.mean(vals) if vals else None

        r2s = [m.ols_r2 for m in valid if m.ols_r2 is not None]
        self.mean_ols_r2 = statistics.mean(r2s) if r2s else None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "name": self.name,
            "mode": self.mode,
            "n_dates": self.n_dates,
            "n_skipped": self.n_skipped,
            "horizons": self.horizons,
            "top_k": self.top_k,
            "exposure_names": self.exposure_names,
            "ruleset_id": self.ruleset_id,
            "mean_ic": {str(k): _round_opt(v, 4) for k, v in self.mean_ic.items()},
            "mean_gross": {str(k): _round_opt(v, 6) for k, v in self.mean_gross.items()},
            "mean_exposure_topk": {
                k: _round_opt(v, 4) for k, v in self.mean_exposure_topk.items()
            },
            "mean_ols_r2": _round_opt(self.mean_ols_r2, 4),
            "by_date": [
                {
                    "date": m.date,
                    "skipped": m.skipped,
                    "skip_reason": m.skip_reason,
                    "n_eligible": m.n_eligible,
                    "ics": {str(k): _round_opt(v, 4) for k, v in m.ics.items()},
                    "gross_returns": {str(k): _round_opt(v, 6) for k, v in m.gross_returns.items()},
                    "top_k_tickers": m.top_k_tickers[:10],
                    "exposure_means_topk": {
                        k: _round_opt(v, 4) for k, v in m.exposure_means_topk.items()
                    },
                    "ols_r2": _round_opt(m.ols_r2, 4),
                }
                for m in self.date_metrics
            ],
        }
        return d


def _round_opt(v: Optional[float], n: int) -> Optional[float]:
    if v is None:
        return None
    return round(v, n)


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    experiment_name: str,
    mode: str,
    snapshot_root: Path,
    price_csv: Path,
    ruleset: DecisionRuleset,
    horizons: List[int],
    top_k: int,
    exposure_names: List[str],
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    min_cols: int = 50,
) -> Tuple[ExperimentResult, Optional[ExperimentResult]]:
    """Run the experiment: baseline, and optionally neutralized.

    Args:
        mode: "baseline" (baseline only), "neutralize" (neutralized only),
              or "compare" (both baseline + neutralized).

    Returns:
        (baseline_result, neutralized_result_or_None)
    """
    # Discover dates
    snap_dates = discover_snapshot_dates(snapshot_root, date_from, date_to)
    print(f"Discovered {len(snap_dates)} snapshot dates")

    # Load prices
    print(f"Loading prices from {price_csv} ...")
    prices = load_price_series(price_csv)
    all_dates_set: set = set()
    for tp in prices.values():
        all_dates_set.update(tp.keys())
    sorted_dates = sorted(all_dates_set)
    print(f"  {len(prices)} tickers, {len(sorted_dates)} trading dates")

    # For neutralization, force sort_anchor to alpha_cohort so we can
    # inject the neutralized percentile as tiebreaker_pct
    if mode in ("neutralize", "compare") and exposure_names:
        neutral_ruleset = dc_replace(ruleset, sort_anchor="alpha_cohort")
    else:
        neutral_ruleset = ruleset

    run_baseline = mode in ("baseline", "compare")
    run_neutral = mode in ("neutralize", "compare") and exposure_names

    baseline = ExperimentResult(
        name=f"{experiment_name}_baseline",
        mode="baseline",
        horizons=horizons,
        top_k=top_k,
        exposure_names=exposure_names,
        ruleset_id=ruleset.ruleset_id,
    )
    neutralized = ExperimentResult(
        name=f"{experiment_name}_neutralized",
        mode="neutralized",
        horizons=horizons,
        top_k=top_k,
        exposure_names=exposure_names,
        ruleset_id=ruleset.ruleset_id,
    ) if run_neutral else None

    for snap_date in snap_dates:
        snap_dir = snapshot_root / snap_date
        raw_rows = load_rankings(snap_dir)

        if not raw_rows or len(raw_rows[0]) < min_cols:
            skip_metrics = DateMetrics(
                date=snap_date, skipped=True,
                skip_reason=f"SKIP ({len(raw_rows[0]) if raw_rows else 0} cols < {min_cols})",
            )
            if run_baseline:
                baseline.date_metrics.append(skip_metrics)
            if neutralized:
                neutralized.date_metrics.append(copy.deepcopy(skip_metrics))
            continue

        # --- Baseline ---
        if run_baseline:
            base_rows = copy.deepcopy(raw_rows)
            backfill_columns(base_rows)
            # Use existing actionable_rank from production (already faithful)
            base_metrics = evaluate_single_date(
                snap_date, base_rows, prices, sorted_dates,
                horizons, top_k, exposure_names,
            )
            baseline.date_metrics.append(base_metrics)
            print(f"  {snap_date} baseline: "
                  f"eligible={base_metrics.n_eligible}, "
                  f"IC(20d)={_round_opt(base_metrics.ics.get(20), 3)}")

        # --- Neutralized ---
        if neutralized:
            neut_rows = copy.deepcopy(raw_rows)
            backfill_columns(neut_rows)

            # Neutralize exposures
            neut_rows, r2, coeffs = neutralize_exposures(
                neut_rows, exposure_names, ruleset,
            )

            # Rerank with neutralized signal
            neut_rows = rerank_faithful(neut_rows, neutral_ruleset)

            neut_metrics = evaluate_single_date(
                snap_date, neut_rows, prices, sorted_dates,
                horizons, top_k, exposure_names,
            )
            neut_metrics.ols_r2 = r2
            neut_metrics.ols_coefficients = coeffs
            neutralized.date_metrics.append(neut_metrics)
            print(f"  {snap_date} neutral:  "
                  f"eligible={neut_metrics.n_eligible}, "
                  f"IC(20d)={_round_opt(neut_metrics.ics.get(20), 3)}, "
                  f"R²={_round_opt(r2, 3)}")

    # Aggregate
    if run_baseline:
        baseline.aggregate()
    if neutralized:
        neutralized.aggregate()

    return baseline, neutralized


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

def generate_comparison_report(
    baseline: ExperimentResult,
    neutralized: Optional[ExperimentResult],
    exposure_names: List[str],
) -> str:
    """Generate a markdown comparison report."""
    lines: List[str] = []

    lines.append(f"# Alpha Experiment: {baseline.name.replace('_baseline', '')}")
    lines.append("")
    lines.append(f"- **Ruleset**: {baseline.ruleset_id}")
    lines.append(f"- **Dates evaluated**: {baseline.n_dates} ({baseline.n_skipped} skipped)")
    lines.append(f"- **Top-K**: {baseline.top_k}")
    lines.append(f"- **Horizons**: {baseline.horizons}")
    if exposure_names:
        lines.append(f"- **Exposures neutralized**: {', '.join(exposure_names)}")
    if neutralized and neutralized.mean_ols_r2 is not None:
        lines.append(f"- **Mean OLS R²**: {neutralized.mean_ols_r2:.4f}")
    lines.append("")

    # --- IC Table ---
    lines.append("## Information Coefficient (Spearman IC)")
    lines.append("")
    if neutralized:
        lines.append("| Horizon | Baseline IC | Neutralized IC | Delta | Improvement |")
        lines.append("|---------|-------------|----------------|-------|-------------|")
        for h in baseline.horizons:
            b_ic = baseline.mean_ic.get(h)
            n_ic = neutralized.mean_ic.get(h)
            b_str = f"{b_ic:.4f}" if b_ic is not None else "—"
            n_str = f"{n_ic:.4f}" if n_ic is not None else "—"
            if b_ic is not None and n_ic is not None:
                delta = n_ic - b_ic
                # Relative improvement
                if abs(b_ic) > 1e-6:
                    pct = delta / abs(b_ic) * 100
                    imp_str = f"{pct:+.1f}%"
                else:
                    imp_str = "—"
                d_str = f"{delta:+.4f}"
            else:
                d_str = "—"
                imp_str = "—"
            lines.append(f"| {h:>3d}d    | {b_str:>11s} | {n_str:>14s} | {d_str:>5s} | {imp_str:>11s} |")
    else:
        lines.append("| Horizon | IC     |")
        lines.append("|---------|--------|")
        for h in baseline.horizons:
            b_ic = baseline.mean_ic.get(h)
            b_str = f"{b_ic:.4f}" if b_ic is not None else "—"
            lines.append(f"| {h:>3d}d    | {b_str:>6s} |")
    lines.append("")

    # --- Returns Table ---
    lines.append("## Top-K Gross Returns (mean per date)")
    lines.append("")
    if neutralized:
        lines.append("| Horizon | Baseline Return | Neutralized Return | Delta   |")
        lines.append("|---------|-----------------|--------------------|---------| ")
        for h in baseline.horizons:
            b_ret = baseline.mean_gross.get(h)
            n_ret = neutralized.mean_gross.get(h)
            b_str = f"{b_ret:+.4%}" if b_ret is not None else "—"
            n_str = f"{n_ret:+.4%}" if n_ret is not None else "—"
            if b_ret is not None and n_ret is not None:
                d_str = f"{(n_ret - b_ret):+.4%}"
            else:
                d_str = "—"
            lines.append(f"| {h:>3d}d    | {b_str:>15s} | {n_str:>18s} | {d_str:>7s} |")
    else:
        lines.append("| Horizon | Return  |")
        lines.append("|---------|---------|")
        for h in baseline.horizons:
            b_ret = baseline.mean_gross.get(h)
            b_str = f"{b_ret:+.4%}" if b_ret is not None else "—"
            lines.append(f"| {h:>3d}d    | {b_str:>7s} |")
    lines.append("")

    # --- Exposure Shift Table ---
    if exposure_names and neutralized:
        lines.append("## Top-K Exposure Shift")
        lines.append("")
        lines.append("| Exposure | Baseline Mean | Neutralized Mean | Delta   |")
        lines.append("|----------|---------------|------------------|---------|")
        for exp in exposure_names:
            b_val = baseline.mean_exposure_topk.get(exp)
            n_val = neutralized.mean_exposure_topk.get(exp)
            b_str = f"{b_val:.4f}" if b_val is not None else "—"
            n_str = f"{n_val:.4f}" if n_val is not None else "—"
            if b_val is not None and n_val is not None:
                d_str = f"{(n_val - b_val):+.4f}"
            else:
                d_str = "—"
            lines.append(f"| {exp:<8s} | {b_str:>13s} | {n_str:>16s} | {d_str:>7s} |")
        lines.append("")

    # --- Per-date IC (20d) ---
    ref_h = 20 if 20 in baseline.horizons else baseline.horizons[0]
    lines.append(f"## Per-Date IC ({ref_h}d)")
    lines.append("")
    if neutralized:
        lines.append("| Date       | Baseline | Neutralized | Delta   |")
        lines.append("|------------|----------|-------------|---------|")
        for bm, nm in zip(baseline.date_metrics, neutralized.date_metrics):
            if bm.skipped:
                continue
            b_ic = bm.ics.get(ref_h)
            n_ic = nm.ics.get(ref_h)
            b_str = f"{b_ic:.4f}" if b_ic is not None else "—"
            n_str = f"{n_ic:.4f}" if n_ic is not None else "—"
            if b_ic is not None and n_ic is not None:
                d_str = f"{(n_ic - b_ic):+.4f}"
            else:
                d_str = "—"
            lines.append(f"| {bm.date} | {b_str:>8s} | {n_str:>11s} | {d_str:>7s} |")
    else:
        lines.append("| Date       | IC     |")
        lines.append("|------------|--------|")
        for bm in baseline.date_metrics:
            if bm.skipped:
                continue
            b_ic = bm.ics.get(ref_h)
            b_str = f"{b_ic:.4f}" if b_ic is not None else "—"
            lines.append(f"| {bm.date} | {b_str:>6s} |")
    lines.append("")

    # --- OLS Diagnostics ---
    if neutralized and neutralized.mean_ols_r2 is not None:
        lines.append("## OLS Diagnostics")
        lines.append("")
        lines.append(f"Mean R² across dates: **{neutralized.mean_ols_r2:.4f}**")
        lines.append("")
        # Show coefficients from last non-skipped date
        valid_dates = [m for m in neutralized.date_metrics
                       if not m.skipped and m.ols_coefficients]
        if valid_dates:
            last = valid_dates[-1]
            lines.append(f"Coefficients (last date {last.date}):")
            lines.append("")
            names = ["intercept"] + exposure_names
            for i, name in enumerate(names):
                if i < len(last.ols_coefficients):
                    lines.append(f"- **{name}**: {last.ols_coefficients[i]:.6f}")
        lines.append("")

    # --- Recommendation ---
    lines.append("## Recommendation")
    lines.append("")
    if neutralized:
        # Compare 20d IC (or first horizon)
        b_ic = baseline.mean_ic.get(ref_h)
        n_ic = neutralized.mean_ic.get(ref_h)
        if b_ic is not None and n_ic is not None:
            delta = n_ic - b_ic
            if delta > 0.02:
                verdict = "PROMOTE"
                reason = f"Neutralization improves {ref_h}d IC by {delta:+.4f} (>{0.02})"
            elif delta > 0:
                verdict = "INVESTIGATE"
                reason = f"Small IC improvement ({delta:+.4f}); needs more dates to confirm"
            elif delta > -0.02:
                verdict = "KEEP BASELINE"
                reason = f"IC change negligible ({delta:+.4f})"
            else:
                verdict = "REJECT"
                reason = f"Neutralization hurts {ref_h}d IC by {delta:+.4f}"
            lines.append(f"**{verdict}**: {reason}")
        else:
            lines.append("**INCONCLUSIVE**: Not enough data to compute IC comparison")
    else:
        lines.append("Baseline-only run — no neutralization to compare.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-K overlap between baseline and neutralized
# ---------------------------------------------------------------------------

def _topk_overlap(baseline: ExperimentResult, neutralized: ExperimentResult, k: int) -> Dict[str, Any]:
    """Compute per-date top-K Jaccard overlap between baseline and neutralized."""
    overlaps = []
    for bm, nm in zip(baseline.date_metrics, neutralized.date_metrics):
        if bm.skipped or nm.skipped:
            continue
        b_set = set(bm.top_k_tickers[:k])
        n_set = set(nm.top_k_tickers[:k])
        if not b_set and not n_set:
            continue
        jac = len(b_set & n_set) / len(b_set | n_set) if (b_set | n_set) else 1.0
        overlaps.append(jac)
    return {
        "mean_jaccard": statistics.mean(overlaps) if overlaps else None,
        "min_jaccard": min(overlaps) if overlaps else None,
        "n_dates": len(overlaps),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Alpha experiment runner: baseline vs exposure-neutralized evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--experiment-name", required=True,
                        help="Name for this experiment (used in output paths)")
    parser.add_argument("--mode", choices=["baseline", "neutralize", "compare"],
                        default="compare",
                        help="baseline: production only; neutralize: neutralized only; compare: both")
    parser.add_argument("--snapshot-root", type=Path,
                        default=PROJECT_ROOT / "data" / "snapshots")
    parser.add_argument("--price-csv", type=Path,
                        default=PROJECT_ROOT / "production_data" / "price_history.csv")
    parser.add_argument("--ruleset", type=Path, default=None,
                        help="Path to ruleset JSON (default: latest snapshot's decision_ruleset.json)")
    parser.add_argument("--exposures", type=str, default="beta,drawdown",
                        help="Comma-separated exposure names: beta,drawdown,vol,mcap")
    parser.add_argument("--horizons", type=str, default="5,20,63")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--date-from", type=str, default=None)
    parser.add_argument("--date-to", type=str, default=None)
    parser.add_argument("--out-root", type=Path, default=None,
                        help="Output directory (default: output/alpha_experiments/{name})")
    parser.add_argument("--min-cols", type=int, default=50,
                        help="Skip snapshots with fewer columns")
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    args = parser.parse_args()

    # Parse list args
    exposure_names = [e.strip() for e in args.exposures.split(",") if e.strip()]
    horizons = [int(h.strip()) for h in args.horizons.split(",")]

    # Validate exposure names
    for exp in exposure_names:
        if exp not in EXPOSURE_MAP:
            print(f"ERROR: Unknown exposure '{exp}'. Valid: {list(EXPOSURE_MAP.keys())}")
            sys.exit(1)

    # Load ruleset
    if args.ruleset:
        ruleset = DecisionRuleset.from_json(str(args.ruleset))
    else:
        dates = sorted([
            p.name for p in args.snapshot_root.iterdir()
            if p.is_dir() and len(p.name) == 10 and p.name[4] == "-"
            and (p / "decision_ruleset.json").exists()
        ])
        if not dates:
            print("ERROR: No snapshot with decision_ruleset.json found")
            sys.exit(1)
        rs_path = args.snapshot_root / dates[-1] / "decision_ruleset.json"
        ruleset = DecisionRuleset.from_json(str(rs_path))
    print(f"Ruleset: {ruleset.ruleset_id}")

    # Output directory
    out_root = args.out_root or (PROJECT_ROOT / "output" / "alpha_experiments" / args.experiment_name)
    out_root.mkdir(parents=True, exist_ok=True)

    # Run experiment
    t0 = time.time()
    baseline, neutralized = run_experiment(
        experiment_name=args.experiment_name,
        mode=args.mode,
        snapshot_root=args.snapshot_root,
        price_csv=args.price_csv,
        ruleset=ruleset,
        horizons=horizons,
        top_k=args.top_k,
        exposure_names=exposure_names,
        date_from=args.date_from,
        date_to=args.date_to,
        min_cols=args.min_cols,
    )
    elapsed = time.time() - t0

    # Print summary
    print(f"\n{'='*60}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Mode: {args.mode}")
    print(f"Elapsed: {elapsed:.1f}s")
    print()

    if baseline.n_dates > 0:
        print("Baseline:")
        for h in horizons:
            ic = baseline.mean_ic.get(h)
            ret = baseline.mean_gross.get(h)
            ic_s = f"{ic:.4f}" if ic is not None else "N/A"
            ret_s = f"{ret:+.4%}" if ret is not None else "N/A"
            print(f"  {h:>3d}d: IC={ic_s:>8s}  Gross={ret_s:>10s}")

    if neutralized and neutralized.n_dates > 0:
        print("Neutralized:")
        for h in horizons:
            ic = neutralized.mean_ic.get(h)
            ret = neutralized.mean_gross.get(h)
            ic_s = f"{ic:.4f}" if ic is not None else "N/A"
            ret_s = f"{ret:+.4%}" if ret is not None else "N/A"
            print(f"  {h:>3d}d: IC={ic_s:>8s}  Gross={ret_s:>10s}")
        if neutralized.mean_ols_r2 is not None:
            print(f"  Mean OLS R²: {neutralized.mean_ols_r2:.4f}")

    # Write outputs
    # 1. JSON results
    if baseline.n_dates > 0:
        with open(out_root / "baseline.json", "w", encoding="utf-8") as f:
            json.dump(baseline.to_dict(), f, indent=2, default=str)

    if neutralized and neutralized.n_dates > 0:
        with open(out_root / "neutralized.json", "w", encoding="utf-8") as f:
            json.dump(neutralized.to_dict(), f, indent=2, default=str)

    # 2. Comparison report
    report = generate_comparison_report(baseline, neutralized, exposure_names)
    report_path = out_root / "compare.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport: {report_path}")

    # 3. Top-K overlap (if both modes ran)
    if neutralized and baseline.n_dates > 0 and neutralized.n_dates > 0:
        overlap = _topk_overlap(baseline, neutralized, args.top_k)
        jac = overlap.get("mean_jaccard")
        jac_s = f"{jac:.3f}" if jac is not None else "N/A"
        print(f"Top-{args.top_k} Jaccard overlap: mean={jac_s}")
        with open(out_root / "overlap.json", "w", encoding="utf-8") as f:
            json.dump(overlap, f, indent=2)


if __name__ == "__main__":
    main()
