#!/usr/bin/env python3
"""Signal robustness backtest: clinical / catalyst / alpha cohort.

Iterates archived snapshots, computes per-date cross-sectional Spearman IC
and top-minus-bottom spread for each signal, then aggregates across dates.

Alpha is evaluated **out-of-sample**: for each eval date D, a temporary
cohort table is built from all usable archive dates strictly before D,
then rows on date D are scored using that rolling table.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import statistics
import sys
import tarfile
from collections import defaultdict
from datetime import date as _date_cls, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from run_rank_ic_backtest import (          # noqa: E402
    ARCHIVE_DIR,
    PRICE_CSV,
    RETURNS_JSON,
    ChainedReturnsProvider,
    MorningstarReturnsProvider,
    compute_forward_returns,
    discover_archives,
)
from backtest.returns_provider import CSVReturnsProvider  # noqa: E402
from module_5_alpha_cohort import (         # noqa: E402
    compute_alpha_cohort_key,
    compute_alpha_raw,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_SHRINK_K = 50
ALPHA_CLIP_MIN = -0.10
ALPHA_CLIP_MAX = 0.10
MIN_TRAIN_DATES = 2  # need at least this many prior dates to score alpha

# ---------------------------------------------------------------------------
# Copied helpers (build_alpha_cohort_table.py, small & self-contained)
# ---------------------------------------------------------------------------
_DD_ARCHETYPES = {"drug_developer"}


def parse_train_mode(mode_str: str) -> Tuple[str, Optional[int]]:
    """Parse --train-mode value.

    Returns (mode, param) where mode is one of 'expanding', 'trailing', 'decay'
    and param is the integer argument (None for expanding).
    """
    mode_str = mode_str.strip().lower()
    if mode_str == "expanding":
        return ("expanding", None)
    if mode_str.startswith("trailing-"):
        try:
            n = int(mode_str.split("-", 1)[1])
        except (ValueError, IndexError):
            raise ValueError(f"Invalid train mode: {mode_str!r} — expected trailing-N")
        if n < 1:
            raise ValueError(f"trailing window must be >= 1, got {n}")
        return ("trailing", n)
    if mode_str.startswith("decay-"):
        try:
            h = int(mode_str.split("-", 1)[1])
        except (ValueError, IndexError):
            raise ValueError(f"Invalid train mode: {mode_str!r} — expected decay-H")
        if h < 1:
            raise ValueError(f"decay half-life must be >= 1, got {h}")
        return ("decay", h)
    raise ValueError(f"Unknown train mode: {mode_str!r}")


def build_weighted_alpha_table(
    train_cache: List[Dict[str, Any]],
    weights: List[float],
    shrink_k: float = DEFAULT_SHRINK_K,
) -> Dict[str, Any]:
    """Build a cohort table with exponentially-weighted cell means.

    Each entry in train_cache is paired with a weight from *weights*.
    Weighted mean: sum(w_i * ret_i) / sum(w_i) per cell.
    Effective n = sum(weights) for shrinkage (lower weights → more shrinkage).
    """
    cell_w_sums: Dict[str, float] = defaultdict(float)
    cell_wr_sums: Dict[str, float] = defaultdict(float)
    cell_w_totals: Dict[str, float] = defaultdict(float)

    for entry, w in zip(train_cache, weights):
        if w <= 0.0:
            continue
        rows = entry["rows"]
        excess = entry["excess"]
        for row in rows:
            tk = (row.get("ticker") or "").strip()
            if tk not in excess:
                continue
            key = compute_alpha_cohort_key(row)
            cell_wr_sums[key] += w * excess[tk]
            cell_w_sums[key] += w
            cell_w_totals[key] += w

    cells: Dict[str, Dict[str, Any]] = {}
    for key in cell_w_sums:
        ws = cell_w_sums[key]
        if ws > 0:
            cells[key] = {
                "mean_excess_ret_6m": round(cell_wr_sums[key] / ws, 6),
                "n": cell_w_totals[key],  # effective n (sum of weights)
            }

    return {
        "cells": cells,
        "shrink_k_default": shrink_k,
        "alpha_clip": {"min": ALPHA_CLIP_MIN, "max": ALPHA_CLIP_MAX},
    }


def compute_cell_diagnostics(
    date_cache: List[Dict[str, Any]],
    cutoff: str = "2025-01-01",
) -> List[Dict[str, Any]]:
    """Per-cell mean excess return before/after cutoff + delta.

    Returns list of dicts sorted by delta ascending (most deteriorated first).
    Cells with only one period get NaN delta (sorted to end).
    """
    pre_rets: Dict[str, List[float]] = defaultdict(list)
    post_rets: Dict[str, List[float]] = defaultdict(list)

    for entry in date_cache:
        date_str = entry["date"]
        rows = entry["rows"]
        excess = entry["excess"]
        bucket = pre_rets if date_str < cutoff else post_rets
        for row in rows:
            tk = (row.get("ticker") or "").strip()
            if tk not in excess:
                continue
            key = compute_alpha_cohort_key(row)
            bucket[key].append(excess[tk])

    all_keys = sorted(set(pre_rets.keys()) | set(post_rets.keys()))
    results: List[Dict[str, Any]] = []
    for key in all_keys:
        pre = pre_rets.get(key, [])
        post = post_rets.get(key, [])
        mean_pre = statistics.mean(pre) if pre else float("nan")
        mean_post = statistics.mean(post) if post else float("nan")
        if pre and post:
            delta = mean_post - mean_pre
        else:
            delta = float("nan")
        results.append({
            "cell": key,
            "mean_excess_pre": mean_pre,
            "mean_excess_post": mean_post,
            "delta": delta,
            "n_pre": len(pre),
            "n_post": len(post),
        })

    # Sort by delta ascending; NaN sorts to end
    results.sort(key=lambda r: (math.isnan(r["delta"]), r["delta"]))
    return results


def load_rankings_dicts(tar_path: Path) -> List[Dict[str, str]]:
    """Read rankings.csv from a tar.gz archive as raw string dicts."""
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith("/rankings.csv"):
                f = io.TextIOWrapper(tar.extractfile(member), encoding="utf-8")
                return list(csv.DictReader(f))
    return []


def backfill_clinical_z_tier(rows: List[Dict[str, str]]) -> None:
    """Backfill clinical_score_z_tier in-place using ddof=0 z-scoring.

    Drug developers: z-score clinical_score within each tier_dev group.
    Commercial / platform / other: set to 0.0.
    """
    dd_by_tier: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for i, row in enumerate(rows):
        arch = (row.get("archetype") or "").strip()
        if arch not in _DD_ARCHETYPES:
            row["clinical_score_z_tier"] = "0.0"
            continue
        tier = (row.get("tier_dev") or "").strip()
        if not tier:
            row["clinical_score_z_tier"] = "0.0"
            continue
        raw = (row.get("clinical_score") or "").strip()
        if not raw:
            row["clinical_score_z_tier"] = "0.0"
            continue
        try:
            score = float(raw)
        except (ValueError, TypeError):
            row["clinical_score_z_tier"] = "0.0"
            continue
        dd_by_tier[tier].append((i, score))

    for tier, members in dd_by_tier.items():
        if len(members) < 2:
            for idx, _ in members:
                rows[idx]["clinical_score_z_tier"] = "0.0"
            continue
        scores = [s for _, s in members]
        mu = statistics.mean(scores)
        std = statistics.pstdev(scores)
        if std > 0:
            for idx, s in members:
                z = round((s - mu) / std, 4)
                rows[idx]["clinical_score_z_tier"] = str(z)
        else:
            for idx, _ in members:
                rows[idx]["clinical_score_z_tier"] = "0.0"


# ---------------------------------------------------------------------------
# Spearman rank correlation (no scipy)
# ---------------------------------------------------------------------------

def _avg_ranks(values: List[float]) -> List[float]:
    """Assign average ranks (1-based) with tie handling."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and values[indexed[j + 1]] == values[indexed[j]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_rank_corr(x: List[float], y: List[float]) -> float:
    """Rank both, then Pearson on ranks. Return 0.0 if n < 3 or zero std."""
    n = len(x)
    if n < 3:
        return 0.0
    rx = _avg_ranks(x)
    ry = _avg_ranks(y)
    mx = statistics.mean(rx)
    my = statistics.mean(ry)
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    sx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    sy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if sx == 0.0 or sy == 0.0:
        return 0.0
    return cov / (sx * sy)


# ---------------------------------------------------------------------------
# Spread computation
# ---------------------------------------------------------------------------

def compute_spread(
    signals: List[float],
    excess_rets: List[float],
    min_per_quintile: int = 5,
) -> float:
    """Top-20% minus bottom-20% mean excess return, sorted by signal desc.

    Returns 0.0 if fewer than min_per_quintile in each quintile.
    """
    n = len(signals)
    q_size = n // 5
    if q_size < min_per_quintile:
        return 0.0
    paired = sorted(zip(signals, excess_rets), key=lambda p: p[0], reverse=True)
    top_mean = statistics.mean([p[1] for p in paired[:q_size]])
    bot_mean = statistics.mean([p[1] for p in paired[-q_size:]])
    return top_mean - bot_mean


# ---------------------------------------------------------------------------
# Rank-residualization (partial IC)
# ---------------------------------------------------------------------------

def residualize_ranks(x: List[float], z: List[float]) -> List[float]:
    """Residualize x ranks vs z ranks via OLS on ranks.

    Returns residuals (x_rank - predicted_x_rank).
    If n < 3 or z has zero variance, returns x ranks unchanged.
    """
    n = len(x)
    if n < 3:
        return list(_avg_ranks(x))
    rx = _avg_ranks(x)
    rz = _avg_ranks(z)
    mz = statistics.mean(rz)
    var_z = sum((rz[i] - mz) ** 2 for i in range(n))
    if var_z == 0.0:
        return list(rx)
    mx = statistics.mean(rx)
    cov_xz = sum((rx[i] - mx) * (rz[i] - mz) for i in range(n))
    b = cov_xz / var_z
    a = mx - b * mz
    return [rx[i] - (a + b * rz[i]) for i in range(n)]


# ---------------------------------------------------------------------------
# Double-sort spread
# ---------------------------------------------------------------------------

def compute_double_sort_spread(
    sort1_signals: List[float],
    sort2_signals: List[float],
    excess_rets: List[float],
    n_groups: int = 3,
    min_per_group: int = 10,
) -> float:
    """Double-sort: split into n_groups by sort1, then within each group
    compute top-half vs bottom-half spread on sort2.  Average across groups.

    Returns 0.0 if any group has fewer than min_per_group members.
    """
    n = len(sort1_signals)
    if n < n_groups * min_per_group:
        return 0.0
    # Sort by sort1 and assign group indices
    order = sorted(range(n), key=lambda i: sort1_signals[i])
    group_size = n // n_groups
    spreads: List[float] = []
    for g in range(n_groups):
        start = g * group_size
        end = start + group_size if g < n_groups - 1 else n
        members = order[start:end]
        if len(members) < min_per_group:
            return 0.0
        # Within this group, sort by sort2 and split top/bottom half
        members_sorted = sorted(members, key=lambda i: sort2_signals[i], reverse=True)
        half = len(members_sorted) // 2
        if half == 0:
            return 0.0
        top_mean = statistics.mean([excess_rets[i] for i in members_sorted[:half]])
        bot_mean = statistics.mean([excess_rets[i] for i in members_sorted[half:]])
        spreads.append(top_mean - bot_mean)
    return statistics.mean(spreads) if spreads else 0.0


# ---------------------------------------------------------------------------
# Rolling out-of-sample alpha cohort table
# ---------------------------------------------------------------------------

def build_rolling_alpha_table(
    train_cache: List[Dict[str, Any]],
    shrink_k: float = DEFAULT_SHRINK_K,
) -> Dict[str, Any]:
    """Build a cohort table from cached train-date data.

    Each entry in train_cache is a dict with keys:
        rows: List[Dict[str, str]]  (already backfilled)
        excess: Dict[str, float]    (ticker -> excess return)

    Returns a table dict compatible with compute_alpha_raw().
    """
    cell_returns: Dict[str, List[float]] = defaultdict(list)
    for entry in train_cache:
        rows = entry["rows"]
        excess = entry["excess"]
        for row in rows:
            tk = (row.get("ticker") or "").strip()
            if tk not in excess:
                continue
            key = compute_alpha_cohort_key(row)
            cell_returns[key].append(excess[tk])

    cells: Dict[str, Dict[str, Any]] = {}
    for key, rets in cell_returns.items():
        cells[key] = {
            "mean_excess_ret_6m": round(statistics.mean(rets), 6) if rets else 0.0,
            "n": len(rets),
        }

    return {
        "cells": cells,
        "shrink_k_default": shrink_k,
        "alpha_clip": {"min": ALPHA_CLIP_MIN, "max": ALPHA_CLIP_MAX},
    }


def score_alpha_oos(
    rows: List[Dict[str, str]],
    table: Dict[str, Any],
    shrink_k: float = DEFAULT_SHRINK_K,
) -> Dict[str, float]:
    """Score each row's alpha using a pre-built cohort table.

    Returns {ticker: alpha_raw} for rows with non-zero alpha.
    """
    result: Dict[str, float] = {}
    for row in rows:
        tk = (row.get("ticker") or "").strip()
        if not tk:
            continue
        key = compute_alpha_cohort_key(row)
        alpha = compute_alpha_raw(key, table, shrink_k, ALPHA_CLIP_MIN, ALPHA_CLIP_MAX)
        result[tk] = alpha
    return result


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------

def _extract_clinical(row: Dict[str, str]) -> Optional[float]:
    """Return clinical_score_z_tier for drug developers, else None."""
    arch = (row.get("archetype") or "").strip()
    if arch not in _DD_ARCHETYPES:
        return None
    raw = (row.get("clinical_score_z_tier") or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


def _extract_catalyst(row: Dict[str, str]) -> Optional[float]:
    """Return inverse-days catalyst signal; 0.0 for non-actionable modes."""
    mode = (row.get("catalyst_mode") or "").strip()
    if mode in ("specific_days", "far_window", "blended_window"):
        days_str = (row.get("catalyst_days") or "").strip()
        if not days_str:
            return 0.0
        try:
            days_int = int(days_str)
        except (ValueError, TypeError):
            return 0.0
        return 1.0 / (1.0 + max(0, days_int))
    return 0.0


# ---------------------------------------------------------------------------
# Price extension
# ---------------------------------------------------------------------------

def extend_price_csv(
    csv_path: Path,
    through_date: str,
) -> Dict[str, Any]:
    """Extend price_history.csv through *through_date* for all tickers.

    For each ticker, fetches only the missing window (max_existing_date+1 ..
    through_date) via yfinance.  Appends new rows; no duplicates because the
    start is anchored to each ticker's existing max date.

    Returns stats: {n_extended, n_rows_appended, n_failed, failed_tickers,
                    n_already_current, n_tickers_total}.
    """
    import yfinance as yf  # lazy — not needed unless --extend-prices

    # Find max date per ticker in existing CSV
    max_dates: Dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = (row.get("ticker") or "").strip().upper()
            d = (row.get("date") or "").strip()[:10]
            if t and d:
                if t not in max_dates or d > max_dates[t]:
                    max_dates[t] = d

    needs_ext = {t: d for t, d in max_dates.items() if d < through_date}
    n_total = len(max_dates)
    if not needs_ext:
        return {"n_extended": 0, "n_rows_appended": 0, "n_failed": 0,
                "failed_tickers": [], "n_already_current": n_total,
                "n_tickers_total": n_total}

    # yfinance end is exclusive — add 1 day to include through_date
    end_yf = (_date_cls.fromisoformat(through_date) + timedelta(days=1)).isoformat()

    all_rows: List[Dict[str, Any]] = []
    failed: List[str] = []
    for ticker, last_date in sorted(needs_ext.items()):
        start = (_date_cls.fromisoformat(last_date) + timedelta(days=1)).isoformat()
        try:
            hist = yf.Ticker(ticker).history(start=start, end=end_yf,
                                             auto_adjust=False)
        except Exception as exc:
            log.warning("  %s: yfinance fetch failed (%s)", ticker, exc)
            failed.append(ticker)
            continue
        if hist.empty:
            continue
        for idx, row in hist.iterrows():
            dt = idx.strftime("%Y-%m-%d")
            close = row.get("Close")
            if close is None or close != close:  # NaN guard
                continue
            all_rows.append({
                "date": dt, "ticker": ticker, "close": str(close),
                "open": str(row["Open"]) if row.get("Open") == row.get("Open") else "",
                "high": str(row["High"]) if row.get("High") == row.get("High") else "",
                "low": str(row["Low"]) if row.get("Low") == row.get("Low") else "",
                "volume": str(int(row["Volume"])) if row.get("Volume") == row.get("Volume") else "",
            })

    if all_rows:
        fieldnames = ["date", "ticker", "close", "open", "high", "low", "volume"]
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            for row in all_rows:
                writer.writerow(row)

    n_ext = len(needs_ext) - len(failed)
    return {"n_extended": n_ext, "n_rows_appended": len(all_rows),
            "n_failed": len(failed), "failed_tickers": failed,
            "n_already_current": n_total - len(needs_ext),
            "n_tickers_total": n_total}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_backtest(
    archive_dir: Path,
    out_dir: Path,
    horizon: int,
    start: Optional[str],
    end: Optional[str],
    shrink_k: float = DEFAULT_SHRINK_K,
    use_regime: bool = False,
    train_mode: str = "expanding",
    cutoff: str = "2025-01-01",
    min_clinical_coverage: float = 0.0,
    min_clinical_pos_share: float = 0.05,
    max_clinical_pos_share: float = 0.95,
    min_fwd_coverage: float = 0.0,
    extend_prices: bool = False,
    prices_through: Optional[str] = None,
) -> Dict[str, Any]:
    """Run signal robustness backtest and write outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    archives = discover_archives(archive_dir, start=start, end=end)
    log.info("Found %d archives in [%s, %s]", len(archives), start or "*", end or "*")
    if not archives:
        log.warning("No archives found — exiting")
        return {}

    # Optionally extend price data before building the provider
    if extend_prices:
        through = prices_through or _date_cls.today().isoformat()
        log.info("Extending price_history.csv through %s …", through)
        ext = extend_price_csv(PRICE_CSV, through)
        log.info("Price extension: %d tickers extended, %d rows appended, "
                 "%d already current, %d failed",
                 ext["n_extended"], ext["n_rows_appended"],
                 ext["n_already_current"], ext["n_failed"])
        if ext["failed_tickers"]:
            log.warning("  Failed tickers: %s", ", ".join(ext["failed_tickers"]))

    # Build returns provider (reads potentially-updated CSV)
    ms = MorningstarReturnsProvider(RETURNS_JSON)
    csv_prov = CSVReturnsProvider(PRICE_CSV, price_col="close")
    provider = ChainedReturnsProvider(ms, csv_prov)
    price_end_date = provider.get_last_date()  # Optional[date]

    # -----------------------------------------------------------------------
    # Pass 1: Load all archives, compute forward returns, cache per-date data
    # -----------------------------------------------------------------------
    price_ticker_set = set(t.upper() for t in provider.get_available_tickers())
    n_price_universe = len(price_ticker_set)

    date_cache: List[Dict[str, Any]] = []  # ordered by date
    date_diagnostics: List[Dict[str, Any]] = []
    for date_str, tar_path in archives:
        log.info("Loading %s …", date_str)
        rows = load_rankings_dicts(tar_path)
        if not rows:
            log.warning("  empty rankings in %s — skipping", tar_path.name)
            date_diagnostics.append({
                "date": date_str, "n_rows": 0, "n_price_rows": 0,
                "n_fwd_rets": 0, "fwd_ret_coverage": 0.0,
                "skip_reason": "EMPTY_RANKINGS", "included": False,
            })
            continue

        backfill_clinical_z_tier(rows)

        all_tickers = [r["ticker"] for r in rows if r.get("ticker")]
        n_rows = len(all_tickers)
        n_price_rows = sum(1 for t in all_tickers if t.upper() in price_ticker_set)
        fwd_rets = compute_forward_returns(provider, all_tickers, date_str, horizon)
        n_fwd = len(fwd_rets)
        # Coverage vs priceable tickers (not total rows) — avoids penalizing
        # dates whose archives contain tickers absent from the price provider
        coverage = n_fwd / n_price_rows if n_price_rows > 0 else 0.0

        skip_reason = ""
        if n_fwd == 0:
            skip_reason = "NO_FWD_RET"
        elif min_fwd_coverage > 0.0 and coverage < min_fwd_coverage:
            skip_reason = "LOW_COVERAGE"

        date_diagnostics.append({
            "date": date_str, "n_rows": n_rows, "n_price_rows": n_price_rows,
            "n_fwd_rets": n_fwd, "fwd_ret_coverage": round(coverage, 4),
            "skip_reason": skip_reason, "included": skip_reason == "",
        })

        if skip_reason:
            log.warning("  %s: %d/%d fwd returns (%.1f%% of priceable) — skipping (%s)",
                        date_str, n_fwd, n_price_rows, coverage * 100, skip_reason)
            continue

        median_ret = statistics.median(fwd_rets.values())
        excess = {t: ret - median_ret for t, ret in fwd_rets.items()}

        # Clinical coverage ratio for training filter (based on raw clinical_score,
        # not backfilled z-tier, since backfill masks missing data as "0.0")
        n_dd = sum(1 for r in rows if (r.get("archetype") or "").strip() in _DD_ARCHETYPES)
        n_raw_clin = sum(
            1 for r in rows
            if (r.get("archetype") or "").strip() in _DD_ARCHETYPES
            and (r.get("clinical_score") or "").strip() != ""
        )
        clin_ratio = n_raw_clin / n_dd if n_dd > 0 else 0.0

        # Clinical sign dispersion: share of DDs with z_tier > 0 (after backfill)
        # If pos_share ≈ 0 or ≈ 1, the sign dimension of the cohort key is homogeneous
        n_pos = 0
        n_valid_clin = 0
        for r in rows:
            z = _extract_clinical(r)
            if z is None:
                continue
            n_valid_clin += 1
            if z > 0.0:
                n_pos += 1
        pos_share = n_pos / n_valid_clin if n_valid_clin > 0 else 0.0

        date_cache.append({
            "date": date_str,
            "rows": rows,
            "fwd_rets": fwd_rets,
            "excess": excess,
            "clinical_coverage": clin_ratio,
            "clinical_pos_share": pos_share,
        })

    # Write date_diagnostics.csv (all dates, including skipped)
    _write_csv(out_dir / "date_diagnostics.csv", date_diagnostics,
               ["date", "n_rows", "n_price_rows", "n_fwd_rets",
                "fwd_ret_coverage", "skip_reason", "included"])
    n_included = sum(1 for d in date_diagnostics if d["included"])
    n_skipped = sum(1 for d in date_diagnostics if not d["included"])
    log.info("Wrote date_diagnostics.csv (%d dates, %d included, %d skipped, "
             "threshold=%.2f, price_universe=%d)",
             len(date_diagnostics), n_included, n_skipped,
             min_fwd_coverage, n_price_universe)

    log.info("Usable dates after forward-return filter: %d", len(date_cache))
    if not date_cache:
        log.warning("No usable dates — exiting")
        return {}

    # -----------------------------------------------------------------------
    # Regime conditioning (optional)
    # -----------------------------------------------------------------------
    regime_by_date: Dict[str, str] = {}
    if use_regime:
        from backtest.regime import (
            load_price_history,
            compute_regime_series,
            assign_regime_to_rebalance_dates,
        )
        price_df = load_price_history(PRICE_CSV)
        regime_series = compute_regime_series(price_df, "XBI")
        eval_dates = [e["date"] for e in date_cache]
        regime_by_date = assign_regime_to_rebalance_dates(regime_series, eval_dates)
        log.info("Regime labels assigned: %s", {v: sum(1 for x in regime_by_date.values() if x == v) for v in set(regime_by_date.values())})

    # Parse training mode
    tm_mode, tm_param = parse_train_mode(train_mode)

    # -----------------------------------------------------------------------
    # Pass 2: Evaluate signals per date (alpha uses rolling OOS table)
    # -----------------------------------------------------------------------
    ic_rows: List[Dict[str, Any]] = []
    spread_rows: List[Dict[str, Any]] = []
    alpha_skipped = 0

    for eval_idx, entry in enumerate(date_cache):
        date_str = entry["date"]
        rows = entry["rows"]
        excess = entry["excess"]
        fwd_rets = entry["fwd_rets"]
        log.info("Evaluating %s (%d/%d) …", date_str, eval_idx + 1, len(date_cache))

        # --- Clinical (DD only, gated on valid clinical_score_z_tier) ---
        clin_pairs: List[Tuple[float, float]] = []
        n_dd_total = 0
        for row in rows:
            tk = (row.get("ticker") or "").strip()
            if (row.get("archetype") or "").strip() in _DD_ARCHETYPES:
                n_dd_total += 1
            sig = _extract_clinical(row)
            if sig is not None and tk in excess:
                clin_pairs.append((sig, excess[tk]))

        ic_clinical = spearman_rank_corr(
            [p[0] for p in clin_pairs], [p[1] for p in clin_pairs]
        ) if len(clin_pairs) >= 3 else float("nan")

        spread_clinical = compute_spread(
            [p[0] for p in clin_pairs], [p[1] for p in clin_pairs]
        ) if clin_pairs else float("nan")

        # --- Catalyst (all rows) ---
        cat_pairs_all: List[Tuple[float, float]] = []
        cat_pairs_nonzero: List[Tuple[float, float]] = []
        for row in rows:
            tk = (row.get("ticker") or "").strip()
            sig = _extract_catalyst(row)
            if sig is not None and tk in excess:
                cat_pairs_all.append((sig, excess[tk]))
                if sig > 0.0:
                    cat_pairs_nonzero.append((sig, excess[tk]))

        # IC on all rows (including zeros)
        ic_catalyst = spearman_rank_corr(
            [p[0] for p in cat_pairs_all], [p[1] for p in cat_pairs_all]
        ) if len(cat_pairs_all) >= 3 else float("nan")

        # Spread on non-zero catalyst only
        spread_catalyst = compute_spread(
            [p[0] for p in cat_pairs_nonzero], [p[1] for p in cat_pairs_nonzero]
        ) if cat_pairs_nonzero else float("nan")

        # --- Alpha (out-of-sample rolling) ---
        ic_alpha = float("nan")
        spread_alpha = float("nan")
        train_dates = date_cache[:eval_idx]  # strictly before eval date

        # Filter out dates with broken clinical data from alpha training:
        # 1. Raw clinical_score coverage below threshold
        # 2. Sign homogeneity (pos_share ≈ 0 or ≈ 1 → no sign variation)
        if (min_clinical_coverage > 0.0) or (min_clinical_pos_share > 0.0) or (max_clinical_pos_share < 1.0):
            train_dates = [
                d for d in train_dates
                if d["clinical_coverage"] >= min_clinical_coverage
                and min_clinical_pos_share <= d["clinical_pos_share"] <= max_clinical_pos_share
            ]

        # Apply training mode dispatch
        if tm_mode == "trailing" and tm_param is not None:
            train_dates = train_dates[max(0, len(train_dates) - tm_param):]

        if len(train_dates) >= MIN_TRAIN_DATES:
            if tm_mode == "decay" and tm_param is not None:
                # Exponential decay with true half-life: w = 2^(-distance/H)
                # i.e. exp(-ln(2) * distance / H), so at distance=H, w=0.5
                n_train = len(train_dates)
                ln2 = math.log(2)
                weights = [
                    math.exp(-ln2 * (n_train - 1 - i) / tm_param)
                    for i in range(n_train)
                ]
                table = build_weighted_alpha_table(train_dates, weights, shrink_k=shrink_k)
            else:
                table = build_rolling_alpha_table(train_dates, shrink_k=shrink_k)
            alpha_scores = score_alpha_oos(rows, table, shrink_k=shrink_k)

            alpha_pairs: List[Tuple[float, float]] = []
            for tk, alpha_val in alpha_scores.items():
                if tk in excess:
                    alpha_pairs.append((alpha_val, excess[tk]))

            if len(alpha_pairs) >= 3:
                ic_alpha = spearman_rank_corr(
                    [p[0] for p in alpha_pairs], [p[1] for p in alpha_pairs]
                )
            spread_alpha = compute_spread(
                [p[0] for p in alpha_pairs], [p[1] for p in alpha_pairs]
            ) if alpha_pairs else float("nan")
        else:
            alpha_skipped += 1
            log.info("  alpha skipped: only %d train dates (need %d)",
                     len(train_dates), MIN_TRAIN_DATES)

        # --- Incremental IC: alpha residualized vs catalyst ---
        ic_alpha_incr = float("nan")
        spread_alpha_double = float("nan")
        n_incr = 0

        if not math.isnan(ic_alpha):
            incr_triples: List[Tuple[float, float, float]] = []
            for row in rows:
                tk = (row.get("ticker") or "").strip()
                if tk in excess and tk in alpha_scores:
                    cat_sig = _extract_catalyst(row)
                    if cat_sig is not None:
                        incr_triples.append((alpha_scores[tk], cat_sig, excess[tk]))

            n_incr = len(incr_triples)
            if n_incr >= 10:
                alpha_vals = [p[0] for p in incr_triples]
                cat_vals = [p[1] for p in incr_triples]
                excess_vals = [p[2] for p in incr_triples]
                residuals = residualize_ranks(alpha_vals, cat_vals)
                ic_alpha_incr = spearman_rank_corr(residuals, excess_vals)
                spread_alpha_double = compute_double_sort_spread(
                    cat_vals, alpha_vals, excess_vals,
                )

        regime_label = regime_by_date.get(date_str, "")
        ic_rows.append({
            "date": date_str,
            "horizon": horizon,
            "regime": regime_label,
            "n_all": len(fwd_rets),
            "n_dd": n_dd_total,
            "n_clinical_valid": len(clin_pairs),
            "clinical_pos_share": round(entry["clinical_pos_share"], 4),
            "n_cat_nonzero": len(cat_pairs_nonzero),
            "n_train_dates": len(train_dates),
            "ic_clinical": ic_clinical,
            "ic_catalyst": ic_catalyst,
            "ic_alpha": ic_alpha,
            "ic_alpha_incr": ic_alpha_incr,
            "n_incr": n_incr,
        })
        spread_rows.append({
            "date": date_str,
            "horizon": horizon,
            "regime": regime_label,
            "spread_clinical": spread_clinical,
            "spread_catalyst": spread_catalyst,
            "spread_alpha": spread_alpha,
            "spread_alpha_double": spread_alpha_double,
        })

    # -----------------------------------------------------------------------
    # Cell diagnostics (always computed)
    # -----------------------------------------------------------------------
    cell_diags = compute_cell_diagnostics(date_cache, cutoff=cutoff)
    if cell_diags:
        _write_csv(
            out_dir / "cell_diagnostics.csv", cell_diags,
            ["cell", "mean_excess_pre", "mean_excess_post", "delta", "n_pre", "n_post"],
        )
        log.info("Wrote cell_diagnostics.csv (%d cells)", len(cell_diags))

    # Write CSV outputs
    _write_csv(out_dir / "ic_timeseries.csv", ic_rows,
               ["date", "horizon", "regime", "n_all", "n_dd", "n_clinical_valid",
                "clinical_pos_share", "n_cat_nonzero", "n_train_dates",
                "ic_clinical", "ic_catalyst", "ic_alpha", "ic_alpha_incr", "n_incr"])
    _write_csv(out_dir / "spread_timeseries.csv", spread_rows,
               ["date", "horizon", "regime", "spread_clinical", "spread_catalyst", "spread_alpha",
                "spread_alpha_double"])

    # Build summary
    max_archive_date = archives[-1][0] if archives else None
    summary = _build_summary(
        ic_rows, spread_rows, horizon, alpha_skipped,
        cell_diags=cell_diags, train_mode=train_mode, cutoff=cutoff,
        min_clinical_coverage=min_clinical_coverage,
        min_clinical_pos_share=min_clinical_pos_share,
        max_clinical_pos_share=max_clinical_pos_share,
        date_diagnostics=date_diagnostics,
        min_fwd_coverage=min_fwd_coverage,
        n_price_universe=n_price_universe,
        price_end_date=price_end_date.isoformat() if price_end_date else None,
        max_archive_date=max_archive_date,
    )
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Wrote outputs to %s", out_dir)
    return summary


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: _fmt(row.get(k)) for k in fieldnames})


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return f"{v:.6f}"
    return str(v)


def _safe_vals(series: List[float]) -> List[float]:
    """Filter out NaN values."""
    return [v for v in series if not math.isnan(v)]


def _build_summary(
    ic_rows: List[Dict[str, Any]],
    spread_rows: List[Dict[str, Any]],
    horizon: int,
    alpha_skipped: int,
    cell_diags: Optional[List[Dict[str, Any]]] = None,
    train_mode: str = "expanding",
    cutoff: str = "2025-01-01",
    min_clinical_coverage: float = 0.0,
    min_clinical_pos_share: float = 0.05,
    max_clinical_pos_share: float = 0.95,
    date_diagnostics: Optional[List[Dict[str, Any]]] = None,
    min_fwd_coverage: float = 0.0,
    n_price_universe: int = 0,
    price_end_date: Optional[str] = None,
    max_archive_date: Optional[str] = None,
) -> Dict[str, Any]:
    n_dates = len(ic_rows)
    if n_dates == 0:
        return {"horizon": horizon, "n_dates": 0, "alpha_skipped": alpha_skipped, "signals": {}}

    n_alpha_eval = sum(1 for r in ic_rows if not math.isnan(r["ic_alpha"]))

    def signal_stats(ic_key: str, spread_key: str) -> Dict[str, Any]:
        ics = _safe_vals([r[ic_key] for r in ic_rows])
        spreads = _safe_vals([r[spread_key] for r in spread_rows])
        result: Dict[str, Any] = {}
        if ics:
            result["mean_ic"] = round(statistics.mean(ics), 4)
            result["median_ic"] = round(statistics.median(ics), 4)
            result["stderr_ic"] = round(
                statistics.stdev(ics) / math.sqrt(len(ics)), 4
            ) if len(ics) >= 2 else 0.0
            result["n_dates"] = len(ics)
        if spreads:
            result["mean_spread"] = round(statistics.mean(spreads), 4)
            result["median_spread"] = round(statistics.median(spreads), 4)
        return result

    # Subperiod split (keyed off cutoff, not midpoint)
    def subperiod_stats(ic_key: str, spread_key: str, ic_slice: list, sp_slice: list) -> Dict[str, Any]:
        ics = _safe_vals([r[ic_key] for r in ic_slice])
        spreads = _safe_vals([r[spread_key] for r in sp_slice])
        result: Dict[str, Any] = {}
        if ics:
            result["mean_ic"] = round(statistics.mean(ics), 4)
            result["n"] = len(ics)
        if spreads:
            result["mean_spread"] = round(statistics.mean(spreads), 4)
        return result

    ic_pre = [r for r in ic_rows if r["date"] < cutoff]
    ic_post = [r for r in ic_rows if r["date"] >= cutoff]
    sp_pre = [r for r in spread_rows if r["date"] < cutoff]
    sp_post = [r for r in spread_rows if r["date"] >= cutoff]

    subperiod: Dict[str, Any] = {}
    for label, ic_sl, sp_sl in [
        ("pre_cutoff", ic_pre, sp_pre),
        ("post_cutoff", ic_post, sp_post),
    ]:
        subperiod[label] = {
            "clinical": subperiod_stats("ic_clinical", "spread_clinical", ic_sl, sp_sl),
            "catalyst": subperiod_stats("ic_catalyst", "spread_catalyst", ic_sl, sp_sl),
            "alpha": subperiod_stats("ic_alpha", "spread_alpha", ic_sl, sp_sl),
            "alpha_incremental": subperiod_stats("ic_alpha_incr", "spread_alpha_double", ic_sl, sp_sl),
        }

    # Alpha incremental stats
    incr_ics = _safe_vals([r["ic_alpha_incr"] for r in ic_rows])
    incr_spreads = _safe_vals([r["spread_alpha_double"] for r in spread_rows])
    alpha_incr_stats: Dict[str, Any] = {}
    if incr_ics:
        alpha_incr_stats["mean_ic"] = round(statistics.mean(incr_ics), 4)
        alpha_incr_stats["median_ic"] = round(statistics.median(incr_ics), 4)
        alpha_incr_stats["stderr_ic"] = round(
            statistics.stdev(incr_ics) / math.sqrt(len(incr_ics)), 4
        ) if len(incr_ics) >= 2 else 0.0
    if incr_spreads:
        alpha_incr_stats["mean_double_sort_spread"] = round(statistics.mean(incr_spreads), 4)
        alpha_incr_stats["median_double_sort_spread"] = round(statistics.median(incr_spreads), 4)
    alpha_incr_stats["n_dates"] = len(incr_ics)

    # Regime-conditioned stats (only if regime labels present)
    regime_stats: Optional[Dict[str, Any]] = None
    regimes_present = set(r.get("regime", "") for r in ic_rows) - {""}
    if regimes_present:
        regime_stats = {}
        for regime_label in sorted(regimes_present):
            ic_sl = [r for r in ic_rows if r.get("regime") == regime_label]
            sp_sl = [r for r in spread_rows if r.get("regime") == regime_label]
            regime_stats[regime_label] = {
                "clinical": subperiod_stats("ic_clinical", "spread_clinical", ic_sl, sp_sl),
                "catalyst": subperiod_stats("ic_catalyst", "spread_catalyst", ic_sl, sp_sl),
                "alpha": subperiod_stats("ic_alpha", "spread_alpha", ic_sl, sp_sl),
                "alpha_incremental": subperiod_stats("ic_alpha_incr", "spread_alpha_double", ic_sl, sp_sl),
            }

    # Cell diagnostics summary
    cell_summary: Optional[Dict[str, Any]] = None
    if cell_diags:
        valid_deltas = [c for c in cell_diags if not math.isnan(c["delta"])]
        n_positive = sum(1 for c in valid_deltas if c["delta"] > 0)
        n_negative = sum(1 for c in valid_deltas if c["delta"] < 0)
        top5_worst = [
            {"cell": c["cell"], "delta": round(c["delta"], 6)}
            for c in valid_deltas[:5]
        ]
        cell_summary = {
            "n_cells": len(cell_diags),
            "n_positive_delta": n_positive,
            "n_negative_delta": n_negative,
            "top5_worst": top5_worst,
        }

    result: Dict[str, Any] = {
        "horizon": horizon,
        "train_mode": train_mode,
        "cutoff": cutoff,
        "min_clinical_coverage": min_clinical_coverage,
        "min_clinical_pos_share": min_clinical_pos_share,
        "max_clinical_pos_share": max_clinical_pos_share,
        "n_dates_total": n_dates,
        "n_dates_alpha_eval": n_alpha_eval,
        "n_dates_alpha_skipped_due_to_insufficient_train": alpha_skipped,
        "signals": {
            "clinical": signal_stats("ic_clinical", "spread_clinical"),
            "catalyst": signal_stats("ic_catalyst", "spread_catalyst"),
            "alpha": signal_stats("ic_alpha", "spread_alpha"),
            "alpha_incremental": alpha_incr_stats,
        },
        "subperiod": subperiod,
    }
    if regime_stats is not None:
        result["regime"] = regime_stats
    if cell_summary is not None:
        result["cell_diagnostics"] = cell_summary

    # Forward-return coverage diagnostics
    if date_diagnostics:
        included_dates = [d for d in date_diagnostics if d["included"]]
        included_coverages = [d["fwd_ret_coverage"] for d in included_dates]
        coverage_stats: Dict[str, Any] = {}
        if included_coverages:
            coverage_stats = {
                "min": round(min(included_coverages), 4),
                "median": round(statistics.median(included_coverages), 4),
                "max": round(max(included_coverages), 4),
            }

        # Bottom 5 included dates by coverage (weakest data)
        bottom_5_included = sorted(included_dates, key=lambda d: d["fwd_ret_coverage"])[:5]
        # Top 5 skipped dates by coverage (closest to threshold)
        skipped_low = [d for d in date_diagnostics if d["skip_reason"] == "LOW_COVERAGE"]
        top_5_skipped = sorted(skipped_low, key=lambda d: d["fwd_ret_coverage"], reverse=True)[:5]

        # Data freshness: detect when price data is too short for the horizon.
        # price_gap_days = (price_end_date - max_archive_date).days:
        #   positive → prices extend past latest archive (healthy)
        #   zero     → prices end on the same day as latest archive
        #   negative → prices end BEFORE latest archive (stale)
        freshness: Dict[str, Any] = {
            "price_end_date": price_end_date,
            "max_archive_date": max_archive_date,
        }
        if price_end_date and max_archive_date:
            price_end_dt = _date_cls.fromisoformat(price_end_date)
            max_arch_dt = _date_cls.fromisoformat(max_archive_date)
            freshness["price_gap_days"] = (price_end_dt - max_arch_dt).days
        else:
            freshness["price_gap_days"] = None
        # Ground-truth staleness: newest archive date is not evaluable
        # (derived from actual skip_reason, not from price_gap_days)
        last_diag = max(date_diagnostics, key=lambda d: d["date"])
        freshness["fwd_returns_stale"] = (
            last_diag["skip_reason"] in ("NO_FWD_RET", "LOW_COVERAGE")
        )

        result["fwd_return_diagnostics"] = {
            "n_archives_total": len(date_diagnostics),
            "n_dates_included": len(included_dates),
            "n_dates_skipped_no_fwd": sum(
                1 for d in date_diagnostics if d["skip_reason"] == "NO_FWD_RET"
            ),
            "n_dates_skipped_low_coverage": len(skipped_low),
            "n_dates_skipped_empty_rankings": sum(
                1 for d in date_diagnostics if d["skip_reason"] == "EMPTY_RANKINGS"
            ),
            "min_fwd_coverage_threshold": min_fwd_coverage,
            "n_price_universe": n_price_universe,
            "coverage_stats": coverage_stats,
            "data_freshness": freshness,
            "bottom_5_included": [
                {"date": d["date"], "fwd_ret_coverage": d["fwd_ret_coverage"]}
                for d in bottom_5_included
            ],
            "top_5_skipped_low_coverage": [
                {"date": d["date"], "fwd_ret_coverage": d["fwd_ret_coverage"]}
                for d in top_5_skipped
            ],
        }

    return result


def _get_freshness(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Extract data_freshness from a summary dict.

    Returns the freshness sub-dict if present, otherwise a safe default
    with fwd_returns_stale=None (unknown).
    """
    diag = summary.get("fwd_return_diagnostics")
    if not diag:
        return {"fwd_returns_stale": None, "price_end_date": None,
                "max_archive_date": None, "price_gap_days": None}
    return diag.get("data_freshness", {"fwd_returns_stale": None,
                                        "price_end_date": None,
                                        "max_archive_date": None,
                                        "price_gap_days": None})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Signal robustness backtest")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "output" / "signal_backtests")
    parser.add_argument("--horizon", type=int, default=126)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--shrink-k", type=float, default=DEFAULT_SHRINK_K)
    parser.add_argument("--regime", action="store_true", help="Add regime (BULL/BEAR/CHOP) labels")
    parser.add_argument("--train-mode", type=str, default="expanding",
                        help="Training mode: expanding (default), trailing-N, decay-H")
    parser.add_argument("--cutoff", type=str, default="2025-01-01",
                        help="Cutoff date for pre/post splits (cell diagnostics, subperiod)")
    parser.add_argument("--min-clinical-coverage", type=float, default=0.0,
                        help="Min clinical_valid/n_dd ratio to include a date in alpha training (0=off)")
    parser.add_argument("--min-clinical-pos-share", type=float, default=0.05,
                        help="Min clinical pos_share to include a date in alpha training (default 0.05)")
    parser.add_argument("--max-clinical-pos-share", type=float, default=0.95,
                        help="Max clinical pos_share to include a date in alpha training (default 0.95)")
    parser.add_argument("--min-fwd-coverage", type=float, default=0.0,
                        help="Min fwd_ret_coverage to include a date (0.0=only skip zero-return dates)")
    parser.add_argument("--extend-prices", action="store_true", default=False,
                        help="Fetch missing price data via yfinance before running")
    parser.add_argument("--prices-through", type=str, default=None,
                        help="Pin price extension end date (YYYY-MM-DD) for PIT reproducibility")
    parser.add_argument("--fail-if-stale", action="store_true", default=False,
                        help="Exit with code 2 if forward returns are stale")
    parser.add_argument("--no-warn-if-stale", action="store_true", default=False,
                        help="Suppress the default staleness warning")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    summary = run_backtest(
        archive_dir=ARCHIVE_DIR,
        out_dir=args.out_dir,
        horizon=args.horizon,
        start=args.start,
        end=args.end,
        shrink_k=args.shrink_k,
        use_regime=args.regime,
        train_mode=args.train_mode,
        cutoff=args.cutoff,
        min_clinical_coverage=args.min_clinical_coverage,
        min_clinical_pos_share=args.min_clinical_pos_share,
        max_clinical_pos_share=args.max_clinical_pos_share,
        min_fwd_coverage=args.min_fwd_coverage,
        extend_prices=args.extend_prices,
        prices_through=args.prices_through,
    )
    if summary:
        print(json.dumps(summary, indent=2))

        # Staleness guardrail (CLI path only)
        fresh = _get_freshness(summary)
        if fresh["fwd_returns_stale"] is True:
            msg = (
                "Forward returns stale: price_end_date=%s "
                "max_archive_date=%s price_gap_days=%s"
            )
            fields = (fresh["price_end_date"], fresh["max_archive_date"],
                      fresh["price_gap_days"])
            if args.fail_if_stale:
                log.error(msg, *fields)
                raise SystemExit(2)
            elif not args.no_warn_if_stale:
                log.warning(msg, *fields)


if __name__ == "__main__":
    main()
