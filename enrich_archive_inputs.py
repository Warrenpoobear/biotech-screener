#!/usr/bin/env python3
"""
Enrich Archive Inputs — Compute missing decision engine inputs from raw data.

For each archive date, computes per-ticker:
  A. Defensive features from prices: beta_xbi_60d, drawdown, rsi_14d, return_60d
  B. Momentum alpha_60d = return_60d - beta * xbi_return_60d
  C. Catalyst from trial_records.json + pdufa_dates.json (PIT-filtered)
  D. Coinvest tier1_count from holdings_detailed.json + elite_managers registry

Writes a sidecar `decision_inputs.json` into each archive's inputs/ directory.

PIT safety:
  - Prices: only use rows where date <= as_of_date
  - Trials: only use records where first_posted <= as_of_date AND
    primary_completion_date in (as_of_date, as_of_date + TRIAL_PCD_WINDOW_DAYS]
  - PDUFA: only use entries where pdufa_date > as_of_date
  - Holdings: static Q3 2025 (known limitation)
"""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import sys
import tarfile
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

ARCHIVE_DIR = PROJECT_ROOT / "data" / "archives"
PRICE_CSV = PROJECT_ROOT / "production_data" / "price_history.csv"

# Trial PCD look-ahead window (calendar days from as_of_date).
# 365d captures Phase 2/3 trials with PCD up to 1 year out.
# The prior 180d cutoff caused 2024 archives to fail the calibration
# DQ gate (catalyst_missing_pct > 80% among eligible dev tickers).
TRIAL_PCD_WINDOW_DAYS = 365


# =============================================================================
# PRICE DATA LOADER
# =============================================================================

def load_price_history(csv_path: Path) -> Dict[str, List[Tuple[date, float]]]:
    """Load price_history.csv into {ticker: [(date, close), ...]} sorted by date.

    Skips rows with missing close price.
    """
    prices: Dict[str, List[Tuple[date, float]]] = {}
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row.get("ticker", "").strip().upper()
            close_str = row.get("close", "").strip()
            date_str = row.get("date", "").strip()
            if not ticker or not close_str or not date_str:
                continue
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d").date()
                close = float(close_str)
            except (ValueError, TypeError):
                continue
            if ticker not in prices:
                prices[ticker] = []
            prices[ticker].append((dt, close))

    # Sort each ticker's prices by date
    for ticker in prices:
        prices[ticker].sort(key=lambda x: x[0])

    return prices


def _filter_prices(
    series: List[Tuple[date, float]], as_of: date
) -> List[Tuple[date, float]]:
    """Return prices up to and including as_of_date (PIT filter)."""
    return [(d, p) for d, p in series if d <= as_of]


# =============================================================================
# DEFENSIVE FEATURES FROM PRICES
# =============================================================================

def _compute_returns(closes: List[float]) -> List[float]:
    """Compute daily log returns from a list of closing prices."""
    rets = []
    for i in range(1, len(closes)):
        if closes[i - 1] > 0 and closes[i] > 0:
            rets.append(math.log(closes[i] / closes[i - 1]))
        else:
            rets.append(0.0)
    return rets


def _ols_beta(y: List[float], x: List[float]) -> Optional[float]:
    """Simple OLS slope: beta = cov(y,x) / var(x). Requires >= 30 obs."""
    n = min(len(y), len(x))
    if n < 30:
        return None
    y = y[:n]
    x = x[:n]
    mean_y = sum(y) / n
    mean_x = sum(x) / n
    cov_xy = sum((y[i] - mean_y) * (x[i] - mean_x) for i in range(n)) / n
    var_x = sum((x[i] - mean_x) ** 2 for i in range(n)) / n
    if var_x < 1e-12:
        return None
    beta = cov_xy / var_x
    return max(0.0, min(3.0, beta))  # Clamp [0, 3]


def _compute_rsi(closes: List[float], period: int = 14) -> Optional[float]:
    """Wilder RSI from a list of closing prices. Needs >= period+1 prices."""
    if len(closes) < period + 1:
        return None

    # Use last period+1 prices to compute initial average, then smooth
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

    # Seed: simple average of first `period` changes
    gains = [max(0, d) for d in deltas[:period]]
    losses = [max(0, -d) for d in deltas[:period]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    # Wilder smoothing for remaining
    for d in deltas[period:]:
        if d >= 0:
            avg_gain = (avg_gain * (period - 1) + d) / period
            avg_loss = (avg_loss * (period - 1) + 0) / period
        else:
            avg_gain = (avg_gain * (period - 1) + 0) / period
            avg_loss = (avg_loss * (period - 1) + (-d)) / period

    if avg_loss < 1e-12:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_drawdown(closes: List[float], lookback: int = 252) -> Optional[float]:
    """Max peak-to-trough drawdown over lookback trading days.

    Returns negative float (e.g. -0.35 means -35% drawdown).
    """
    window = closes[-lookback:] if len(closes) > lookback else closes
    if len(window) < 2:
        return None
    peak = window[0]
    max_dd = 0.0
    for p in window:
        if p > peak:
            peak = p
        if peak > 0:
            dd = (p - peak) / peak
            if dd < max_dd:
                max_dd = dd
    return max_dd


def compute_defensive_and_momentum(
    ticker_prices: List[Tuple[date, float]],
    xbi_prices: List[Tuple[date, float]],
    as_of: date,
) -> Dict[str, Any]:
    """Compute defensive features and momentum alpha for one ticker."""
    result: Dict[str, Any] = {}

    # PIT filter
    tp = _filter_prices(ticker_prices, as_of)
    xp = _filter_prices(xbi_prices, as_of)

    if len(tp) < 2:
        return result

    closes = [p for _, p in tp]

    # RSI from last 60 prices (enough for 14-day RSI with smoothing)
    rsi_window = closes[-60:] if len(closes) > 60 else closes
    rsi = _compute_rsi(rsi_window, 14)
    if rsi is not None:
        result["rsi_14d"] = round(rsi, 1)

    # Drawdown (252-day lookback)
    dd = _compute_drawdown(closes, 252)
    if dd is not None:
        result["drawdown"] = round(dd, 4)

    # XBI drawdown + relative drawdown (ticker vs XBI)
    xbi_closes = [p for _, p in xp]
    xbi_dd = _compute_drawdown(xbi_closes, 252) if len(xbi_closes) >= 126 else None
    if xbi_dd is not None:
        result["drawdown_xbi"] = round(xbi_dd, 4)
    if dd is not None and xbi_dd is not None:
        result["drawdown_rel_xbi"] = round(dd - xbi_dd, 4)

    # 60-day return
    if len(closes) >= 61:
        ret_60d = (closes[-1] / closes[-61]) - 1.0
        result["return_60d"] = round(ret_60d, 4)
    elif len(closes) >= 2:
        ret_60d = (closes[-1] / closes[0]) - 1.0
        result["return_60d"] = round(ret_60d, 4)

    # Beta and alpha require XBI alignment
    if len(xp) >= 61 and len(tp) >= 61:
        # Build date-aligned daily returns for last 60 trading days
        # Use date index for alignment
        xbi_map = {d: p for d, p in xp}
        ticker_map = {d: p for d, p in tp}
        common_dates = sorted(set(xbi_map.keys()) & set(ticker_map.keys()))

        # Take last 61 common dates for 60 daily returns
        common_dates = common_dates[-61:]
        if len(common_dates) >= 31:  # Need at least 30 returns
            t_rets = _compute_returns([ticker_map[d] for d in common_dates])
            x_rets = _compute_returns([xbi_map[d] for d in common_dates])

            beta = _ols_beta(t_rets, x_rets)
            if beta is not None:
                result["beta_xbi_60d"] = round(beta, 4)

                # Alpha = ticker_return_60d - beta * xbi_return_60d
                xbi_closes = [xbi_map[d] for d in common_dates]
                xbi_ret_60d = (xbi_closes[-1] / xbi_closes[0]) - 1.0
                ticker_closes = [ticker_map[d] for d in common_dates]
                ticker_ret_60d = (ticker_closes[-1] / ticker_closes[0]) - 1.0
                alpha = ticker_ret_60d - beta * xbi_ret_60d
                result["alpha_60d"] = round(alpha, 4)

    # If beta unavailable but return available, use raw return as alpha proxy
    if "alpha_60d" not in result and "return_60d" in result:
        result["alpha_60d"] = result["return_60d"]

    return result


# =============================================================================
# CATALYST FROM TRIALS + PDUFA
# =============================================================================

def _sec_event_days(event: Dict[str, Any], as_of: date) -> Optional[int]:
    """Compute days-to-catalyst from a SEC 8-K or ADCOM event.

    For DAY precision (HIGH confidence): use event_date directly.
    For QUARTER precision (MED confidence): use start of quarter (conservative).
    Skips HALF_YEAR (LOW confidence) and events with no date.

    Returns positive days if event is in the future, else None.
    """
    precision = event.get("date_precision", "DAY")
    confidence = event.get("confidence", "")

    # Skip LOW confidence (vague "mid-year" / "year-end" language)
    if confidence == "LOW":
        return None

    date_str = event.get("event_date", "")
    if not date_str:
        return None

    try:
        event_date = datetime.strptime(str(date_str)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None

    days_away = (event_date - as_of).days
    if days_away <= 0:
        return None

    return days_away


def compute_catalyst(
    ticker: str,
    trials: List[Dict[str, Any]],
    pdufa_entries: List[Dict[str, Any]],
    as_of: date,
    trial_window_days: int = TRIAL_PCD_WINDOW_DAYS,
    sec_events: Optional[List[Dict[str, Any]]] = None,
    adcom_events: Optional[List[Dict[str, Any]]] = None,
    fda_regulatory_events: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute catalyst proximity for one ticker from trials + PDUFA + SEC/ADCOM + FDA regulatory.

    PIT rules:
      - Trials: first_posted <= as_of AND primary_completion_date in (as_of, as_of+trial_window_days]
      - PDUFA: pdufa_date > as_of
      - SEC 8-K: filing_date <= as_of AND event_date > as_of (enforced by collector)
      - ADCOM: publication_date <= as_of AND event_date > as_of (enforced by collector)
      - FDA regulatory: publication_date <= as_of AND event_date > as_of (enforced by collector)
    """
    result: Dict[str, Any] = {}
    nearest_days: Optional[int] = None
    nearest_source: Optional[str] = None
    nearest_type: Optional[str] = None

    # Phase 2/3 trials with upcoming primary completion dates
    for trial in trials:
        if trial.get("ticker", "").upper() != ticker.upper():
            continue
        phase = str(trial.get("phase", "")).upper()
        if phase not in ("PHASE2", "PHASE3", "PHASE 2", "PHASE 3",
                         "PHASE2/PHASE3", "PHASE 2/PHASE 3"):
            continue
        fp_str = trial.get("first_posted")
        pcd_str = trial.get("primary_completion_date")
        if not fp_str or not pcd_str:
            continue
        try:
            fp_date = datetime.strptime(str(fp_str)[:10], "%Y-%m-%d").date()
            pcd_date = datetime.strptime(str(pcd_str)[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue

        # PIT filter: trial must be publicly known before as_of
        if fp_date > as_of:
            continue
        # Upcoming: completion date in (as_of, as_of + trial_window_days]
        days_away = (pcd_date - as_of).days
        if days_away <= 0 or days_away > trial_window_days:
            continue

        if nearest_days is None or days_away < nearest_days:
            nearest_days = days_away
            nearest_source = "trial_pcd"
            nearest_type = "DATA_READOUT"

    # PDUFA dates
    for entry in pdufa_entries:
        if entry.get("ticker", "").upper() != ticker.upper():
            continue
        pdufa_str = entry.get("pdufa_date")
        if not pdufa_str:
            continue
        try:
            pdufa_date = datetime.strptime(str(pdufa_str)[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        days_away = (pdufa_date - as_of).days
        if days_away <= 0:
            continue
        if nearest_days is None or days_away < nearest_days:
            nearest_days = days_away
            nearest_source = "pdufa"
            nearest_type = "FDA_DECISION"

    # SEC 8-K timing events (PDUFA dates, data readouts from filings)
    if sec_events:
        for event in sec_events:
            if event.get("ticker", "").upper() != ticker.upper():
                continue
            days_away = _sec_event_days(event, as_of)
            if days_away is None:
                continue
            if nearest_days is None or days_away < nearest_days:
                nearest_days = days_away
                nearest_source = "sec_8k"
                nearest_type = event.get("event_type", "DATA_READOUT")

    # FDA Advisory Committee events
    if adcom_events:
        for event in adcom_events:
            if event.get("ticker", "").upper() != ticker.upper():
                continue
            days_away = _sec_event_days(event, as_of)
            if days_away is None:
                continue
            if nearest_days is None or days_away < nearest_days:
                nearest_days = days_away
                nearest_source = "adcom"
                nearest_type = "FDA_ADCOM"

    # FDA regulatory notices (Federal Register: approvals, CRLs, RTFs, warning letters)
    if fda_regulatory_events:
        for event in fda_regulatory_events:
            if event.get("ticker", "").upper() != ticker.upper():
                continue
            days_away = _sec_event_days(event, as_of)
            if days_away is None:
                continue
            if nearest_days is None or days_away < nearest_days:
                nearest_days = days_away
                nearest_source = "federal_register"
                nearest_type = event.get("event_type", "FDA_APPROVAL")

    if nearest_days is not None:
        result["days_to_catalyst"] = nearest_days
        result["in_optimal_window"] = 14 <= nearest_days <= 60
        result["catalyst_source"] = nearest_source
        result["catalyst_event_type"] = nearest_type

    return result


# =============================================================================
# COINVEST TIER1 COUNT
# =============================================================================

def compute_tier1_counts(
    holdings_data: Dict[str, Any],
    tier1_ciks: set,
) -> Dict[str, int]:
    """Compute per-ticker tier1_count from holdings + elite manager CIKs.

    holdings_data["tickers"][TICKER]["holdings"]["current"] has CIK keys.
    tier1_ciks is a set of CIK strings (with leading zeros, 10-digit format).
    """
    result: Dict[str, int] = {}
    tickers_data = holdings_data.get("tickers", {})

    for ticker, entry in tickers_data.items():
        holdings = entry.get("holdings", {})
        current = holdings.get("current", {})
        count = 0
        for cik_key in current.keys():
            # Normalize both to compare: strip leading zeros then re-pad to 10
            cik_norm = cik_key.lstrip("0").zfill(10)
            if cik_norm in tier1_ciks:
                count += 1
        result[ticker.upper()] = count

    return result


# =============================================================================
# PER-ARCHIVE ENRICHMENT
# =============================================================================

CATALYST_KEYS = frozenset({
    "days_to_catalyst", "in_optimal_window", "catalyst_source",
    "catalyst_event_type",
})


def enrich_archive(
    tar_path: Path,
    all_prices: Dict[str, List[Tuple[date, float]]],
    dry_run: bool = False,
    trial_window_days: int = TRIAL_PCD_WINDOW_DAYS,
    catalyst_only: bool = False,
    output_tar_path: Path | None = None,
    sec_events: Optional[List[Dict[str, Any]]] = None,
    adcom_events: Optional[List[Dict[str, Any]]] = None,
    fda_regulatory_events: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute and write decision_inputs.json sidecar for one archive.

    When catalyst_only=True, only recompute catalyst fields in the existing
    decision_inputs.json, preserving defensive features (drawdown, beta, etc.).
    """
    date_str = tar_path.name.replace(".tar.gz", "")
    result: Dict[str, Any] = {"date": date_str, "status": "ok"}

    try:
        as_of = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        result["status"] = "error"
        result["error"] = f"Cannot parse date from {tar_path.name}"
        return result

    tmp_dir = tempfile.mkdtemp(prefix=f"enrich_{date_str}_")
    try:
        # Extract archive
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(tmp_dir)

        archive_root = Path(tmp_dir) / date_str
        if not archive_root.exists():
            subdirs = [d for d in Path(tmp_dir).iterdir() if d.is_dir()]
            if subdirs:
                archive_root = subdirs[0]
            else:
                result["status"] = "error"
                result["error"] = "no archive root dir found"
                return result

        inputs_dir = archive_root / "inputs"
        if not inputs_dir.exists():
            result["status"] = "error"
            result["error"] = "no inputs/ directory"
            return result

        # ---- Load archive-local data ----

        # Trial records
        trials_path = inputs_dir / "trial_records.json"
        trials: List[Dict[str, Any]] = []
        if trials_path.exists():
            with open(trials_path, "r") as f:
                trials = json.load(f)

        # PDUFA dates
        pdufa_path = inputs_dir / "pdufa_dates.json"
        pdufa_entries: List[Dict[str, Any]] = []
        if pdufa_path.exists():
            with open(pdufa_path, "r") as f:
                pdufa_entries = json.load(f)

        # Holdings
        holdings_path = inputs_dir / "holdings_detailed.json"
        holdings_data: Dict[str, Any] = {}
        if holdings_path.exists():
            with open(holdings_path, "r") as f:
                holdings_data = json.load(f)

        # Rankings CSV for ticker list
        csv_path = archive_root / "rankings.csv"
        tickers: List[str] = []
        if csv_path.exists():
            with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    t = row.get("ticker", "").strip().upper()
                    if t:
                        tickers.append(t)

        # ---- Tier1 CIKs from elite_managers ----
        from elite_managers import get_elite_ciks
        raw_ciks = get_elite_ciks()
        tier1_ciks = {cik.lstrip("0").zfill(10) for cik in raw_ciks}

        # ---- Coinvest tier1 counts ----
        tier1_counts = compute_tier1_counts(holdings_data, tier1_ciks)

        # ---- XBI prices for beta/alpha computation ----
        xbi_prices = all_prices.get("XBI", [])

        # ---- Load existing sidecar if catalyst_only ----
        existing_inputs: Dict[str, Dict[str, Any]] = {}
        sidecar_path = inputs_dir / "decision_inputs.json"
        if catalyst_only and sidecar_path.exists():
            with open(sidecar_path, "r") as f:
                existing_inputs = json.load(f)

        # ---- Compute per-ticker ----
        decision_inputs: Dict[str, Dict[str, Any]] = {}
        stats = {"n_beta": 0, "n_catalyst": 0, "n_tier1": 0}

        for ticker in tickers:
            if catalyst_only:
                # Start from existing entry, strip old catalyst keys
                entry = dict(existing_inputs.get(ticker, {}))
                for k in CATALYST_KEYS:
                    entry.pop(k, None)
            else:
                entry = {}

            if not catalyst_only:
                # A+B: Defensive + momentum from prices
                ticker_prices = all_prices.get(ticker, [])
                if ticker_prices:
                    features = compute_defensive_and_momentum(
                        ticker_prices, xbi_prices, as_of
                    )
                    entry.update(features)
                    if "beta_xbi_60d" in features:
                        stats["n_beta"] += 1

            # C: Catalyst from trials + PDUFA + SEC/ADCOM
            cat = compute_catalyst(ticker, trials, pdufa_entries, as_of,
                                   trial_window_days=trial_window_days,
                                   sec_events=sec_events,
                                   adcom_events=adcom_events,
                                   fda_regulatory_events=fda_regulatory_events)
            if cat:
                entry.update(cat)
                stats["n_catalyst"] += 1

            if not catalyst_only:
                # D: Coinvest tier1_count
                t1 = tier1_counts.get(ticker)
                if t1 is not None:
                    entry["tier1_count"] = t1
                    if t1 > 0:
                        stats["n_tier1"] += 1

            if entry:
                decision_inputs[ticker] = entry

        result["n_tickers"] = len(tickers)
        result["n_enriched"] = len(decision_inputs)
        result["stats"] = stats

        if dry_run:
            result["status"] = "dry_run"
            return result

        # Write sidecar JSON
        sidecar_path = inputs_dir / "decision_inputs.json"
        with open(sidecar_path, "w") as f:
            json.dump(decision_inputs, f, indent=2, default=str)

        # Repack archive
        dest = output_tar_path or tar_path
        new_tar_path = dest.with_suffix(".tar.gz.tmp")
        with tarfile.open(new_tar_path, "w:gz") as tar:
            tar.add(str(archive_root), arcname=date_str)

        shutil.move(str(new_tar_path), str(dest))

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return result


# =============================================================================
# MAIN
# =============================================================================

def _collect_sec_events_for_date(
    as_of: date,
    universe_entries: List[Dict[str, Any]],
    sec_mode: str,
    adcom_mode: str,
    data_dir: Path,
    multi_form_mode: str = "off",
    fda_regulatory_mode: str = "off",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Collect SEC 8-K, multi-form, ADCOM, and FDA regulatory events for a single as_of_date.

    Returns (sec_events, adcom_events, multi_form_events, fda_regulatory_events) lists.
    """
    sec_events: List[Dict[str, Any]] = []
    adcom_events: List[Dict[str, Any]] = []
    multi_form_events: List[Dict[str, Any]] = []
    fda_regulatory_events: List[Dict[str, Any]] = []

    if sec_mode == "live":
        try:
            from wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector import (
                collect_8k_timing_events,
            )
            sec_events = collect_8k_timing_events(
                universe=universe_entries,
                as_of_date=as_of,
                cache_dir=PROJECT_ROOT / "wake_robin_data_pipeline" / "cache" / "sec" / "8k_catalysts",
                lookback_days=365,
            )
        except Exception as e:
            print(f"    SEC 8-K collection error: {e}")

    if multi_form_mode == "live":
        try:
            from wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector import (
                collect_sec_filing_events,
            )
            multi_form_events = collect_sec_filing_events(
                universe=universe_entries,
                as_of_date=as_of,
                cache_dir=PROJECT_ROOT / "wake_robin_data_pipeline" / "cache" / "sec" / "8k_catalysts",
                lookback_days=365,
            )
        except Exception as e:
            print(f"    SEC multi-form collection error: {e}")

    if adcom_mode == "live":
        try:
            from wake_robin_data_pipeline.collectors.fda_adcom_collector import (
                collect_fda_adcom_events,
                build_product_ticker_map,
            )
            product_map = build_product_ticker_map(data_dir)
            ticker_set = {e.get("ticker", "").upper() for e in universe_entries if e.get("ticker")}
            adcom_events = collect_fda_adcom_events(
                drug_to_ticker=product_map,
                as_of_date=as_of,
                cache_dir=PROJECT_ROOT / "wake_robin_data_pipeline" / "cache" / "fda",
                lookback_days=365,
                universe_tickers=ticker_set,
            )
        except Exception as e:
            print(f"    ADCOM collection error: {e}")

    if fda_regulatory_mode == "live":
        try:
            from wake_robin_data_pipeline.collectors.fda_adcom_collector import (
                collect_fda_regulatory_notices,
                build_product_ticker_map,
            )
            product_map = build_product_ticker_map(data_dir)
            fda_regulatory_events = collect_fda_regulatory_notices(
                product_ticker_map=product_map,
                as_of_date=as_of,
                cache_dir=PROJECT_ROOT / "wake_robin_data_pipeline" / "cache" / "fda",
                lookback_days=365,
            )
        except Exception as e:
            print(f"    FDA regulatory collection error: {e}")

    return sec_events, adcom_events, multi_form_events, fda_regulatory_events


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Enrich archives with decision engine inputs"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute but don't modify archives"
    )
    parser.add_argument(
        "--archive", type=str, default=None,
        help="Process a single archive date (e.g. 2025-06-30)"
    )
    parser.add_argument(
        "--trial-window", type=int, default=None,
        help=f"Trial PCD look-ahead window in days (default: {TRIAL_PCD_WINDOW_DAYS})"
    )
    parser.add_argument(
        "--catalyst-only", action="store_true",
        help="Only update catalyst fields; preserve existing defensive features"
    )
    parser.add_argument(
        "--sec-8k-mode", type=str, default="off", choices=["off", "live"],
        help="SEC 8-K catalyst mode: off (default) or live (fetch from EDGAR)"
    )
    parser.add_argument(
        "--adcom-mode", type=str, default="off", choices=["off", "live"],
        help="FDA ADCOM mode: off (default) or live (fetch from Federal Register + EDGAR)"
    )
    parser.add_argument(
        "--sec-multi-form-mode", type=str, default="off", choices=["off", "live"],
        help="SEC multi-form (10-Q, 10-K, 6-K) mode: off (default) or live"
    )
    parser.add_argument(
        "--fda-regulatory-mode", type=str, default="off", choices=["off", "live"],
        help="FDA regulatory notices (Federal Register) mode: off (default) or live"
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory for enriched archives (default: modify in-place). "
             "Source archives are copied then enriched, preserving originals."
    )
    parser.add_argument(
        "--start", type=str, default=None,
        help="Only process archives on or after this date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="Only process archives on or before this date (YYYY-MM-DD)"
    )
    args = parser.parse_args()

    if not ARCHIVE_DIR.exists():
        print(f"ERROR: Archive directory not found: {ARCHIVE_DIR}")
        sys.exit(1)
    if not PRICE_CSV.exists():
        print(f"ERROR: Price history not found: {PRICE_CSV}")
        sys.exit(1)

    # Resolve output directory
    out_dir: Optional[Path] = None
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    archives = sorted(ARCHIVE_DIR.glob("*.tar.gz"))
    if args.archive:
        archives = [a for a in archives if a.name.replace(".tar.gz", "") == args.archive]
        if not archives:
            print(f"ERROR: Archive not found for date: {args.archive}")
            sys.exit(1)

    # Date range filter
    if args.start:
        archives = [a for a in archives if a.name.replace(".tar.gz", "") >= args.start]
    if args.end:
        archives = [a for a in archives if a.name.replace(".tar.gz", "") <= args.end]

    trial_window = args.trial_window if args.trial_window is not None else TRIAL_PCD_WINDOW_DAYS
    use_sec = args.sec_8k_mode != "off"
    use_adcom = args.adcom_mode != "off"
    use_multi_form = args.sec_multi_form_mode != "off"
    use_fda_regulatory = args.fda_regulatory_mode != "off"

    if out_dir:
        print(f"Output directory: {out_dir}")
    print(f"Loading price history from {PRICE_CSV} ...")
    all_prices = load_price_history(PRICE_CSV)
    print(f"  {len(all_prices)} tickers, XBI has {len(all_prices.get('XBI', []))} days")

    # Load universe for SEC/ADCOM collection (needs ticker + CIK)
    universe_entries: List[Dict[str, Any]] = []
    data_dir = PROJECT_ROOT / "production_data"
    if use_sec or use_adcom or use_multi_form or use_fda_regulatory:
        universe_path = data_dir / "universe.json"
        if universe_path.exists():
            with open(universe_path, "r") as f:
                universe_entries = json.load(f)
            print(f"  Universe: {len(universe_entries)} tickers (for SEC/ADCOM lookup)")
        else:
            print(f"  WARNING: universe.json not found, SEC/ADCOM will have no CIK mapping")

    print()

    mode_parts = []
    if args.catalyst_only:
        mode_parts.append("catalyst-only")
    else:
        mode_parts.append("full")
    if use_sec:
        mode_parts.append(f"sec-8k={args.sec_8k_mode}")
    if use_adcom:
        mode_parts.append(f"adcom={args.adcom_mode}")
    if use_multi_form:
        mode_parts.append(f"sec-multi-form={args.sec_multi_form_mode}")
    if use_fda_regulatory:
        mode_parts.append(f"fda-regulatory={args.fda_regulatory_mode}")
    mode_label = ", ".join(mode_parts)

    print(f"Enriching {len(archives)} archives ({mode_label}, trial window: {trial_window}d)")
    if args.dry_run:
        print("  (dry run — no files will be modified)")
    print()

    for tar_path in archives:
        date_str = tar_path.name.replace(".tar.gz", "")
        print(f"  {tar_path.name} ...", end=" ", flush=True)

        # Collect SEC/ADCOM events for this date if enabled
        sec_events: Optional[List[Dict[str, Any]]] = None
        adcom_events_list: Optional[List[Dict[str, Any]]] = None
        multi_form_events: Optional[List[Dict[str, Any]]] = None
        fda_reg_events: Optional[List[Dict[str, Any]]] = None
        if use_sec or use_adcom or use_multi_form or use_fda_regulatory:
            try:
                as_of = datetime.strptime(date_str, "%Y-%m-%d").date()
                sec_events, adcom_events_list, multi_form_events, fda_reg_events = _collect_sec_events_for_date(
                    as_of, universe_entries, args.sec_8k_mode, args.adcom_mode, data_dir,
                    multi_form_mode=args.sec_multi_form_mode,
                    fda_regulatory_mode=args.fda_regulatory_mode,
                )
            except ValueError:
                pass  # Bad date format, enrich_archive will handle it

        # Determine output path (copy-on-write if --out-dir)
        output_tar: Optional[Path] = None
        if out_dir:
            output_tar = out_dir / tar_path.name
            if not args.dry_run:
                shutil.copy2(tar_path, output_tar)

        enrich_path = output_tar if output_tar and not args.dry_run else tar_path
        # Combine 8-K and multi-form SEC events (same schema, different source labels)
        combined_sec = list(sec_events) if sec_events else []
        if multi_form_events:
            combined_sec.extend(multi_form_events)
        r = enrich_archive(enrich_path, all_prices, dry_run=args.dry_run,
                           trial_window_days=trial_window,
                           catalyst_only=args.catalyst_only,
                           sec_events=combined_sec or None,
                           adcom_events=adcom_events_list,
                           fda_regulatory_events=fda_reg_events)
        s = r.get("stats", {})
        if r["status"] in ("ok", "dry_run"):
            tag = "DRY" if r["status"] == "dry_run" else "OK"
            sec_str = ""
            if sec_events is not None:
                sec_str = f", sec={len(sec_events)}"
            if adcom_events_list is not None:
                sec_str += f", adcom={len(adcom_events_list)}"
            if fda_reg_events is not None:
                sec_str += f", fda_reg={len(fda_reg_events)}"
            print(
                f"{tag} ({r.get('n_enriched', 0)}/{r.get('n_tickers', 0)} tickers, "
                f"beta={s.get('n_beta', 0)}, cat={s.get('n_catalyst', 0)}, "
                f"t1={s.get('n_tier1', 0)}{sec_str})"
            )
        else:
            print(f"ERROR: {r.get('error', 'unknown')}")

    print("\nDone.")


if __name__ == "__main__":
    main()
