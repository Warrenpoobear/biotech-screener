"""
Milestone 1 Panel Builder — builds (rebalance_date, ticker, scores…, fwd returns)

PIT-safe: calls run_screening_pipeline with explicit as_of_date and pit_mode.
Deterministic: stable date/ticker ordering, stable column set.
"""
from __future__ import annotations

import csv
import gzip
import json
import logging
import math
import sys
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Component score extraction
# ---------------------------------------------------------------------------

# Fields that are metadata / not numeric scores (skip in auto-discovery)
_SKIP_KEYS = {
    "ticker", "composite_rank", "market_cap_bucket", "stage_bucket",
    "severity", "rankable", "coinvest_overlap_count", "coinvest_quarter",
    "coinvest_usable", "missing_subfactor_pct", "flags", "determinism_hash",
    "schema_version", "normalization_method", "cohort_key",
    "score_breakdown", "component_scores", "effective_weights",
    "coinvest_holders", "monotonic_caps_applied",
}


def _safe_float(val: Any) -> float:
    """Convert to float or NaN."""
    if val is None:
        return float("nan")
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def extract_component_scores(record: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract numeric component scores from a ranked_securities entry.

    Handles both v2 (flat fields) and v3 (nested component_scores list) formats.
    Returns dict of {field_name: float_value}.
    """
    out: Dict[str, float] = {}

    # V3 format: component_scores is a list of dicts with name/raw/normalized/contribution
    comp_list = record.get("component_scores")
    if isinstance(comp_list, list):
        for comp in comp_list:
            if not isinstance(comp, dict):
                continue
            name = comp.get("name", "")
            if not name:
                continue
            # Extract raw and normalized scores
            raw_val = comp.get("raw")
            norm_val = comp.get("normalized")
            contrib_val = comp.get("contribution")
            conf_val = comp.get("confidence")
            if raw_val is not None:
                out[f"{name}_raw"] = _safe_float(raw_val)
            if norm_val is not None:
                out[f"{name}_normalized"] = _safe_float(norm_val)
            if contrib_val is not None:
                out[f"{name}_contribution"] = _safe_float(contrib_val)
            if conf_val is not None:
                out[f"{name}_confidence"] = _safe_float(conf_val)

    # Top-level fields: composite_score, uncertainty_penalty, etc.
    for key in ("composite_score", "uncertainty_penalty"):
        val = record.get(key)
        if val is not None:
            out[key] = _safe_float(val)

    # V2 fallback: flat fields like clinical_dev_raw, financial_raw, etc.
    for key in ("clinical_dev_raw", "clinical_dev_normalized",
                "financial_raw", "financial_normalized",
                "catalyst_raw", "catalyst_normalized"):
        if key not in out:
            val = record.get(key)
            if val is not None:
                out[key] = _safe_float(val)

    return out


def discover_score_columns(records: List[Dict[str, Any]]) -> List[str]:
    """
    Scan first batch of ranked_securities to discover available score columns.
    Uses extract_component_scores to be format-agnostic (v2/v3).
    Returns sorted list of column names.
    """
    found: Set[str] = set()
    for rec in records[:50]:  # Sample first 50 records
        scores = extract_component_scores(rec)
        found.update(scores.keys())
    return sorted(found)


# ---------------------------------------------------------------------------
# Weekly schedule generation
# ---------------------------------------------------------------------------

_WEEKDAY_MAP = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}


def generate_rebalance_dates(
    start: date,
    end: date,
    weekday: str = "FRI",
    trading_days: Optional[Set[str]] = None,
) -> List[date]:
    """
    Generate weekly rebalance dates between start and end (inclusive).

    If a candidate falls on a non-trading day (per trading_days set)
    or a weekend, roll back to the previous trading day.
    """
    target_wd = _WEEKDAY_MAP.get(weekday.upper())
    if target_wd is None:
        raise ValueError(f"Unknown weekday: {weekday}")

    # Find first target weekday >= start
    d = start
    while d.weekday() != target_wd:
        d += timedelta(days=1)

    dates: List[date] = []
    while d <= end:
        adjusted = _roll_to_trading_day(d, trading_days)
        if adjusted >= start:
            dates.append(adjusted)
        d += timedelta(days=7)

    # Deduplicate (two Fridays could roll to same Thursday) and sort
    return sorted(set(dates))


def _roll_to_trading_day(d: date, trading_days: Optional[Set[str]]) -> date:
    """Roll d backwards to the nearest trading day."""
    if trading_days is not None:
        # Roll back up to 7 days to find a trading day
        for _ in range(7):
            if d.isoformat() in trading_days:
                return d
            d -= timedelta(days=1)
        # Fallback: return original minus weekends
        return d
    # No calendar — just skip weekends
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


# ---------------------------------------------------------------------------
# Price loading + forward returns
# ---------------------------------------------------------------------------

def load_price_history(
    filepath: Path,
    close_col: str = "close",
) -> Dict[str, Dict[str, float]]:
    """
    Load price history CSV into {ticker: {date_str: close_price}}.

    Handles both 'close' and 'adj_close' column names.
    """
    prices: Dict[str, Dict[str, float]] = {}
    with open(filepath, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header in {filepath}")

        # Normalise column names
        cols = {c.strip().lower(): c for c in reader.fieldnames}
        date_key = cols.get("date", "date")
        ticker_key = cols.get("ticker", cols.get("symbol", "ticker"))

        # Pick best close column
        close_key = None
        for candidate in ("adj_close", "closeadj", "close_adj", "close"):
            if candidate in cols:
                close_key = cols[candidate]
                break
        if close_key is None:
            close_key = close_col

        for row in reader:
            ticker = row.get(ticker_key, "").strip().upper()
            dt = row.get(date_key, "").strip()
            raw_price = row.get(close_key, "").strip()
            if not ticker or not dt or not raw_price:
                continue
            try:
                price = float(raw_price)
                if price <= 0:
                    continue
            except ValueError:
                continue
            prices.setdefault(ticker, {})[dt] = price

    return prices


def build_trading_day_index(
    prices: Dict[str, Dict[str, float]],
) -> List[str]:
    """Return sorted union of all trading dates across all tickers."""
    all_dates: Set[str] = set()
    for ticker_prices in prices.values():
        all_dates.update(ticker_prices.keys())
    return sorted(all_dates)


def compute_forward_return(
    ticker_prices: Dict[str, float],
    trading_days: List[str],
    as_of_date: str,
    horizon: int,
) -> Optional[float]:
    """
    Compute forward return from as_of_date over `horizon` trading days.

    Returns (P_{t+h} / P_t) - 1, or None if data is missing.
    """
    if as_of_date not in ticker_prices:
        return None

    # Find index of as_of_date in trading_days
    try:
        idx = trading_days.index(as_of_date)
    except ValueError:
        return None

    target_idx = idx + horizon
    if target_idx >= len(trading_days):
        return None

    target_date = trading_days[target_idx]
    target_price = ticker_prices.get(target_date)
    if target_price is None:
        return None

    base_price = ticker_prices[as_of_date]
    if base_price == 0:
        return None

    return (target_price / base_price) - 1.0


# ---------------------------------------------------------------------------
# Pipeline caller
# ---------------------------------------------------------------------------

def run_pipeline_for_date(
    as_of_date: str,
    data_dir: Path,
    pit_mode: str = "degrade",
    cache_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """
    Run the screening pipeline for a single as_of_date.

    Uses cached results if available under cache_dir.
    Returns the full pipeline result dict, or None on failure.
    """
    # Check cache first
    if cache_dir is not None:
        cache_file = cache_dir / f"pipeline_{as_of_date}.json"
        if cache_file.exists():
            logger.info(f"  Cache hit: {as_of_date}")
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

    # Import pipeline function
    try:
        # Add project root to path if needed
        project_root = Path(__file__).resolve().parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from run_screen import run_screening_pipeline
    except ImportError as e:
        logger.error(f"Cannot import pipeline: {e}")
        return None

    try:
        logger.info(f"  Running pipeline for {as_of_date} (pit_mode={pit_mode})...")
        result = run_screening_pipeline(
            as_of_date=as_of_date,
            data_dir=data_dir,
            pit_mode=pit_mode,
            # Disable heavy optional features for backtest speed
            enable_coinvest=False,
            enable_enhancements=False,
            enable_short_interest=False,
            enable_smart_money=False,
            enable_clustering=False,
            apply_defensive_multiplier=False,
            enable_position_sizing=False,
            no_clinical_filter=True,  # Keep full universe for IC
            pipeline_timeout=300,
        )
    except Exception as e:
        logger.warning(f"  Pipeline failed for {as_of_date}: {e}")
        return None

    # Cache result
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"pipeline_{as_of_date}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(result, f, default=_json_default)
        except Exception as e:
            logger.warning(f"  Cache write failed: {e}")

    return result


def _json_default(obj: Any) -> Any:
    """JSON serializer for Decimal / date objects."""
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# Panel builder
# ---------------------------------------------------------------------------

def build_panel(
    rebalance_dates: List[date],
    data_dir: Path,
    prices: Dict[str, Dict[str, float]],
    trading_days: List[str],
    pit_mode: str = "degrade",
    cache_dir: Optional[Path] = None,
    max_dates: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Build the full panel dataset.

    Returns (rows, score_columns) where rows are dicts ready for CSV output.
    """
    dates_to_process = rebalance_dates
    if max_dates is not None:
        dates_to_process = dates_to_process[:max_dates]

    all_rows: List[Dict[str, Any]] = []
    score_cols: Optional[List[str]] = None

    for i, rebal_date in enumerate(dates_to_process):
        as_of_str = rebal_date.isoformat()
        logger.info(f"[{i+1}/{len(dates_to_process)}] Processing {as_of_str}")

        result = run_pipeline_for_date(
            as_of_date=as_of_str,
            data_dir=data_dir,
            pit_mode=pit_mode,
            cache_dir=cache_dir,
        )
        if result is None:
            logger.warning(f"  Skipping {as_of_str} (pipeline returned None)")
            continue

        # Extract ranked securities from Module 5
        m5 = result.get("module_5_composite", {})
        ranked = m5.get("ranked_securities", [])
        if not ranked:
            logger.warning(f"  No ranked securities for {as_of_str}")
            continue

        # Discover score columns from first batch
        if score_cols is None:
            score_cols = discover_score_columns(ranked)
            logger.info(f"  Score columns discovered: {score_cols}")

        # Build rows for this date (sorted by ticker)
        for rec in sorted(ranked, key=lambda r: r.get("ticker", "")):
            ticker = rec.get("ticker", "")
            if not ticker:
                continue

            ticker_prices = prices.get(ticker, {})
            row: Dict[str, Any] = {
                "rebalance_date": as_of_str,
                "ticker": ticker,
            }

            # Add score columns
            scores = extract_component_scores(rec)
            for col in score_cols:
                val = scores.get(col)
                if val is None:
                    # Try direct access from record
                    raw = rec.get(col)
                    if raw is not None:
                        try:
                            val = float(raw)
                        except (TypeError, ValueError):
                            val = float("nan")
                    else:
                        val = float("nan")
                row[col] = val

            # Composite rank
            row["composite_rank"] = rec.get("composite_rank")

            # Forward returns
            for horizon, col_name in [(5, "fwd_5d"), (21, "fwd_21d"), (63, "fwd_63d"), (126, "fwd_126d")]:
                ret = compute_forward_return(
                    ticker_prices, trading_days, as_of_str, horizon
                )
                row[col_name] = ret

            all_rows.append(row)

    return all_rows, score_cols or []


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_panel_csv_gz(
    rows: List[Dict[str, Any]],
    score_cols: List[str],
    output_path: Path,
) -> int:
    """
    Write panel rows to a gzipped CSV with stable column ordering.

    Column order: rebalance_date, ticker, [score_cols alphabetical],
                  composite_rank, fwd_21d, fwd_63d, fwd_126d
    Returns number of rows written.
    """
    if not rows:
        logger.warning("No rows to write")
        return 0

    # Stable column order
    meta_cols = ["rebalance_date", "ticker"]
    rank_cols = ["composite_rank"]
    return_cols = ["fwd_5d", "fwd_21d", "fwd_63d", "fwd_126d"]
    columns = meta_cols + sorted(score_cols) + rank_cols + return_cols

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # Format floats to 6 decimals, leave None/NaN as empty
            formatted = {}
            for col in columns:
                val = row.get(col)
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    formatted[col] = ""
                elif isinstance(val, float):
                    formatted[col] = f"{val:.6f}"
                else:
                    formatted[col] = val
            writer.writerow(formatted)

    return len(rows)
