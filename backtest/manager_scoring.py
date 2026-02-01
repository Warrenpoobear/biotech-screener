"""
Tier-1 overlap manager scoring — weekly raw returns.

Builds weekly equal-weight portfolios per manager from 13F holdings snapshots,
computes forward returns, and produces a manager leaderboard.

PIT-safe: holdings are assigned to weeks only AFTER the filing date.
Deterministic: stable manager/ticker/date ordering throughout.
"""
from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load manager registry (Tier-1 elite_core)
# ---------------------------------------------------------------------------

def load_manager_registry(
    registry_path: Path,
    tier: str = "elite_core",
) -> Dict[str, str]:
    """
    Load manager registry and return {cik: name} for the specified tier.
    """
    with open(registry_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    managers = data.get(tier, [])
    return {m["cik"]: m["name"] for m in managers if "cik" in m and "name" in m}


# ---------------------------------------------------------------------------
# Build manager holdings panel from holdings_snapshots.json
# ---------------------------------------------------------------------------

def build_manager_holdings_panel(
    holdings_path: Path,
    manager_ciks: Dict[str, str],
    filing_metadata_cutoff: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a panel of (manager_cik, manager_name, quarter_end, ticker, shares, value_kusd, filed_at).

    PIT safety: only include holdings where filed_at <= filing_metadata_cutoff (if provided).
    Otherwise include all filings.

    Returns DataFrame sorted by (manager_cik, quarter_end, ticker).
    """
    with open(holdings_path, "r", encoding="utf-8") as f:
        snapshots = json.load(f)

    records: List[Dict[str, Any]] = []

    for ticker in sorted(snapshots.keys()):
        tdata = snapshots[ticker]
        filings_meta = tdata.get("filings_metadata", {})

        for period in ["current", "prior"]:
            holdings = tdata.get("holdings", {}).get(period, {})
            for cik in sorted(holdings.keys()):
                if cik not in manager_ciks:
                    continue

                info = holdings[cik]
                shares = info.get("shares", 0)
                value_kusd = info.get("value_kusd", 0)
                state = info.get("state", "")
                quarter_end = info.get("quarter_end", "")

                # Skip if not KNOWN or no position
                if state != "KNOWN" or shares <= 0:
                    continue

                # Get filing date from metadata
                meta = filings_meta.get(cik, {})
                filed_at = meta.get("filed_at", "")
                if filed_at:
                    filed_at = filed_at[:10]  # Truncate to YYYY-MM-DD

                # PIT filter
                if filing_metadata_cutoff and filed_at:
                    if filed_at > filing_metadata_cutoff:
                        continue

                records.append({
                    "manager_cik": cik,
                    "manager_name": manager_ciks[cik],
                    "quarter_end": quarter_end,
                    "ticker": ticker,
                    "shares": shares,
                    "value_kusd": value_kusd,
                    "filed_at": filed_at,
                })

    df = pd.DataFrame(records)
    if df.empty:
        return df
    df = df.sort_values(
        ["manager_cik", "quarter_end", "ticker"]
    ).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Map manager holdings to weekly rebalance dates (PIT-safe)
# ---------------------------------------------------------------------------

def assign_holdings_to_weeks(
    holdings_df: pd.DataFrame,
    rebalance_dates: List[str],
) -> pd.DataFrame:
    """
    For each rebalance date, assign the most recent filing per manager
    where filed_at <= rebalance_date.

    PIT safety: a manager's holdings only become usable AFTER the filing
    date (filed_at). This means no portfolios exist before the earliest
    filing date in the dataset.

    Returns DataFrame with columns:
        rebalance_date, manager_cik, manager_name, ticker, shares, value_kusd
    """
    if holdings_df.empty:
        return pd.DataFrame(columns=[
            "rebalance_date", "manager_cik", "manager_name",
            "ticker", "shares", "value_kusd",
        ])

    # Get unique filings per (manager_cik, quarter_end)
    filings = holdings_df.groupby(
        ["manager_cik", "quarter_end"]
    ).agg({"filed_at": "first", "manager_name": "first"}).reset_index()

    # Log PIT boundary: earliest filing date
    earliest_filing = filings["filed_at"].min()
    n_pre_filing = sum(1 for d in rebalance_dates if d < earliest_filing)
    n_post_filing = len(rebalance_dates) - n_pre_filing
    logger.info(
        f"  PIT boundary: earliest filing={earliest_filing}, "
        f"{n_pre_filing} dates before (no holdings), "
        f"{n_post_filing} dates with potential holdings"
    )

    # For each (rebalance_date, manager_cik), find the best (most recent)
    # quarter_end where filed_at <= rebalance_date
    best_quarter: Dict[tuple, str] = {}
    for _, filing in filings.iterrows():
        cik = filing["manager_cik"]
        qe = filing["quarter_end"]
        filed = filing["filed_at"]
        if not filed:
            continue
        for dt in rebalance_dates:
            if filed > dt:
                continue
            key = (dt, cik)
            if key not in best_quarter or qe > best_quarter[key]:
                best_quarter[key] = qe

    # Build final records from best_quarter mapping
    final_records: List[Dict[str, Any]] = []
    for (dt, cik), qe in sorted(best_quarter.items()):
        mask = (
            (holdings_df["manager_cik"] == cik)
            & (holdings_df["quarter_end"] == qe)
        )
        for _, row in holdings_df[mask].iterrows():
            final_records.append({
                "rebalance_date": dt,
                "manager_cik": cik,
                "manager_name": row["manager_name"],
                "ticker": row["ticker"],
                "shares": row["shares"],
                "value_kusd": row["value_kusd"],
            })

    result = pd.DataFrame(final_records)
    if result.empty:
        return result
    return result.sort_values(
        ["rebalance_date", "manager_cik", "ticker"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Compute manager portfolio returns
# ---------------------------------------------------------------------------

def compute_manager_returns(
    weekly_holdings: pd.DataFrame,
    panel: pd.DataFrame,
    return_col: str = "fwd_21d",
    date_col: str = "rebalance_date",
) -> pd.DataFrame:
    """
    Compute equal-weight portfolio return for each (rebalance_date, manager).

    Merges holdings with panel returns and computes equal-weight average.

    Returns DataFrame with:
        rebalance_date, manager_cik, manager_name, portfolio_return,
        n_holdings, n_with_returns, universe_mean_return
    """
    if weekly_holdings.empty:
        return pd.DataFrame(columns=[
            date_col, "manager_cik", "manager_name", "portfolio_return",
            "n_holdings", "n_with_returns", "universe_mean_return",
        ])

    # Merge holdings with panel returns
    merged = weekly_holdings.merge(
        panel[[date_col, "ticker", return_col]],
        on=[date_col, "ticker"],
        how="left",
    )

    # Compute universe mean return per date
    universe_means = panel.dropna(subset=[return_col]).groupby(date_col)[return_col].mean()

    results: List[Dict[str, Any]] = []
    groups = merged.groupby([date_col, "manager_cik", "manager_name"])

    for (dt, cik, name), group in sorted(groups):
        n_holdings = len(group)
        valid = group.dropna(subset=[return_col])
        n_with_returns = len(valid)

        if n_with_returns == 0:
            port_ret = np.nan
        else:
            port_ret = float(valid[return_col].mean())

        univ_mean = float(universe_means.get(dt, np.nan))

        results.append({
            date_col: dt,
            "manager_cik": cik,
            "manager_name": name,
            "portfolio_return": port_ret,
            "n_holdings": n_holdings,
            "n_with_returns": n_with_returns,
            "universe_mean_return": univ_mean,
        })

    df = pd.DataFrame(results)
    return df.sort_values([date_col, "manager_cik"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Newey–West t-stat for a return series
# ---------------------------------------------------------------------------

def _nw_tstat(series: np.ndarray, lags: int = 4) -> float:
    """Newey–West t-stat for the mean of a time series."""
    valid = series[np.isfinite(series)]
    T = len(valid)
    if T < 3:
        return np.nan
    mean = np.mean(valid)
    demeaned = valid - mean

    gamma_0 = np.sum(demeaned ** 2) / T
    nw_sum = gamma_0
    for j in range(1, min(lags, T - 1) + 1):
        w = 1.0 - j / (lags + 1.0)
        gamma_j = np.sum(demeaned[j:] * demeaned[:-j]) / T
        nw_sum += 2.0 * w * gamma_j

    se = math.sqrt(max(nw_sum / T, 0.0))
    if se == 0:
        return np.nan
    return mean / se


# ---------------------------------------------------------------------------
# Manager leaderboard
# ---------------------------------------------------------------------------

def build_manager_leaderboard(
    manager_returns: pd.DataFrame,
    min_weeks: int = 3,
    nw_lags: int = 4,
    date_col: str = "rebalance_date",
) -> pd.DataFrame:
    """
    Build manager leaderboard with summary statistics.

    Returns DataFrame with:
        manager_cik, manager_name, avg_return, volatility, t_stat,
        hit_rate, avg_excess_vs_universe, n_weeks, avg_holdings
    """
    if manager_returns.empty:
        return pd.DataFrame(columns=[
            "manager_cik", "manager_name", "avg_return", "volatility",
            "t_stat", "hit_rate", "avg_excess_vs_universe",
            "n_weeks", "avg_holdings",
        ])

    rows: List[Dict[str, Any]] = []
    groups = manager_returns.groupby(["manager_cik", "manager_name"])

    for (cik, name), group in sorted(groups):
        rets = group["portfolio_return"].dropna().values
        n_weeks = len(rets)

        if n_weeks < min_weeks:
            continue

        avg_ret = float(np.mean(rets))
        vol = float(np.std(rets, ddof=1)) if n_weeks > 1 else np.nan
        t_stat = _nw_tstat(rets, lags=nw_lags)
        hit_rate = float(np.sum(rets > 0) / n_weeks) if n_weeks > 0 else 0.0

        # Excess vs universe
        excess = group["portfolio_return"].values - group["universe_mean_return"].values
        excess_valid = excess[np.isfinite(excess)]
        avg_excess = float(np.mean(excess_valid)) if len(excess_valid) > 0 else np.nan

        avg_holdings = float(group["n_with_returns"].mean())

        rows.append({
            "manager_cik": cik,
            "manager_name": name,
            "avg_return": round(avg_ret, 6),
            "volatility": round(vol, 6) if np.isfinite(vol) else np.nan,
            "t_stat": round(t_stat, 4) if np.isfinite(t_stat) else np.nan,
            "hit_rate": round(hit_rate, 4),
            "avg_excess_vs_universe": round(avg_excess, 6) if np.isfinite(avg_excess) else np.nan,
            "n_weeks": n_weeks,
            "avg_holdings": round(avg_holdings, 1),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Sort by t_stat descending, then avg_return descending
    df = df.sort_values(
        ["t_stat", "avg_return"],
        ascending=[False, False],
    ).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


# ---------------------------------------------------------------------------
# Convenience: write outputs
# ---------------------------------------------------------------------------

def write_manager_outputs(
    leaderboard: pd.DataFrame,
    manager_returns: pd.DataFrame,
    outdir: Path,
) -> Dict[str, str]:
    """Write manager leaderboard and detailed returns. Returns paths."""
    outdir.mkdir(parents=True, exist_ok=True)

    ranking_path = outdir / "manager_ranking.csv"
    leaderboard.to_csv(ranking_path, index=False)

    scorecard_path = outdir / "manager_scorecard.csv"
    manager_returns.to_csv(scorecard_path, index=False)

    return {
        "ranking": str(ranking_path),
        "scorecard": str(scorecard_path),
    }
