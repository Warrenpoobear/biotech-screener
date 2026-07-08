#!/usr/bin/env python3
"""Biotech portfolio hedge report — weekly IC-ready analysis.

Recommends optimal XBI/IBB options structures to hedge systematic
biotech risk in a long portfolio.  Read-only analysis tool; does NOT
modify scoring, ranking, or execution logic.

Usage:
    python tools/biotech_hedge_report.py --as-of-date 2026-03-17
    python tools/biotech_hedge_report.py \
        --as-of-date 2026-03-17 \
        --portfolio-csv production_data/portfolio_positions.csv \
        --hedge-notional 1000000
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Repo root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from common.massive_chain_analytics import (
    compute_atm_straddle,
    find_25delta_contracts,
    find_atm_contracts,
    load_chain_snapshot,
    save_chain_snapshot,
)
from common.options_greeks import black_scholes_greeks

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("biotech_hedge_report")

SCHEMA_VERSION = "hedge_report.v2"
HEDGE_ETFS = ["XBI", "IBB"]
DEFAULT_RISK_FREE = 0.05
TRADING_DAYS_PER_YEAR = 252
OPTIONS_SOURCE_AUTO = "auto"
OPTIONS_SOURCE_MASSIVE = "massive"
OPTIONS_SOURCE_TASTY = "tasty"


# ---------------------------------------------------------------------------
# Options source selection
# ---------------------------------------------------------------------------


def _tasty_available() -> bool:
    """Check if Tastytrade credentials are configured."""
    try:
        from common.options_diagnostics import _has_credentials

        return _has_credentials()
    except ImportError:
        return False


def _massive_available() -> bool:
    """Check if Massive API key is configured."""
    return bool(os.environ.get("MASSIVE_API_KEY"))


def select_options_source(preference: str) -> Tuple[str, str]:
    """Select the best available options source.

    Returns (source_name, reason).
    """
    if preference == OPTIONS_SOURCE_TASTY:
        if _tasty_available():
            return OPTIONS_SOURCE_TASTY, "user requested tasty; credentials available"
        return (
            OPTIONS_SOURCE_MASSIVE if _massive_available() else "realized_vol"
        ), "user requested tasty but credentials missing; fell back"

    if preference == OPTIONS_SOURCE_MASSIVE:
        if _massive_available():
            return OPTIONS_SOURCE_MASSIVE, "user requested massive; API key available"
        return "realized_vol", "user requested massive but API key missing; fell back"

    # Auto mode: prefer Tastytrade (richer live IV/skew), then Massive, then proxy
    if _tasty_available():
        return OPTIONS_SOURCE_TASTY, "auto: tastytrade selected (live IV, skew, term structure)"
    if _massive_available():
        return OPTIONS_SOURCE_MASSIVE, "auto: massive selected (per-strike chain data)"
    return "realized_vol", "auto: no options API credentials; using realized vol proxy"


def fetch_tasty_diagnostics(
    etf_tickers: List[str],
    as_of_date: str,
) -> Dict[str, Dict[str, Any]]:
    """Fetch Tastytrade diagnostics for hedge ETFs.

    Returns {ticker: {opt_has_data, opt_atm_iv, opt_front_iv, opt_back_iv,
    opt_term_slope, opt_put_call_skew, opt_rr_25d, opt_nearest_expiry,
    opt_dte, opt_iv_regime, opt_event_premium, ...}}.
    """
    try:
        from common.options_diagnostics import fetch_options_diagnostics

        return fetch_options_diagnostics(etf_tickers, as_of_date)
    except Exception as exc:
        logger.warning("Tastytrade fetch failed: %s", exc)
        return {}


def _safe_float(val: Any) -> Optional[float]:
    """Convert to float or return None."""
    if val is None or val == "":
        return None
    try:
        f = float(val)
        return f if not math.isnan(f) else None
    except (ValueError, TypeError):
        return None


def _safe_int(val: Any) -> Optional[int]:
    """Convert to int or return None."""
    if val is None or val == "":
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Phase 0 — Price data helpers
# ---------------------------------------------------------------------------


def load_price_history(
    price_csv: Path,
) -> Dict[str, Dict[str, float]]:
    """Load price_history.csv → {ticker: {date_str: close}}."""
    prices: Dict[str, Dict[str, float]] = {}
    if not price_csv.exists():
        logger.warning("price_history.csv not found at %s", price_csv)
        return prices
    with open(price_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tk = row.get("ticker", "").strip()
            dt = row.get("date", "").strip()
            cl = row.get("close", "")
            if tk and dt and cl:
                try:
                    prices.setdefault(tk, {})[dt] = float(cl)
                except ValueError:
                    continue
    return prices


def fetch_etf_prices(
    tickers: List[str],
    price_csv: Path,
    existing: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Fetch missing ETF price history via yfinance and append to CSV."""
    missing = [t for t in tickers if t not in existing or len(existing[t]) < 60]
    if not missing:
        return existing
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed — cannot fetch ETF prices")
        return existing

    logger.info("Fetching price history for %s via yfinance...", missing)
    end_dt = date.today() + timedelta(days=1)
    start_dt = end_dt - timedelta(days=400)  # ~1.1 years
    try:
        data = yf.download(
            missing,
            start=start_dt.isoformat(),
            end=end_dt.isoformat(),
            progress=False,
            threads=True,
        )
    except Exception as exc:
        logger.error("yfinance download failed: %s", exc)
        return existing

    if data.empty:
        logger.warning("No data returned from yfinance for %s", missing)
        return existing

    new_rows: List[Dict[str, str]] = []
    # yfinance returns MultiIndex columns even for single tickers when
    # passed as a list.  Always use tuple indexing.
    has_multi = hasattr(data.columns, "levels") or (len(data.columns) > 0 and isinstance(data.columns[0], tuple))
    for dt_idx in data.index:
        dt_str = dt_idx.strftime("%Y-%m-%d")
        for ticker in missing:
            try:
                if has_multi:
                    close = float(data.loc[dt_idx, ("Close", ticker)])
                else:
                    close = float(data.loc[dt_idx, "Close"])
                if close != close or close <= 0:  # NaN check
                    continue
                existing.setdefault(ticker, {})[dt_str] = close
                new_rows.append({"date": dt_str, "ticker": ticker, "close": str(close)})
            except (KeyError, TypeError, ValueError):
                continue

    # Append to price_history.csv
    if new_rows and price_csv.exists():
        try:
            with open(price_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["date", "ticker", "close", "open", "high", "low", "volume"],
                )
                for r in new_rows:
                    writer.writerow(
                        {
                            "date": r["date"],
                            "ticker": r["ticker"],
                            "close": r["close"],
                            "open": "",
                            "high": "",
                            "low": "",
                            "volume": "",
                        }
                    )
            logger.info("Appended %d rows for %s to price_history.csv", len(new_rows), missing)
        except Exception as exc:
            logger.warning("Failed to append to price_history.csv: %s", exc)

    return existing


# ---------------------------------------------------------------------------
# Phase 1 — Portfolio exposure and correlation
# ---------------------------------------------------------------------------


def resolve_portfolio_csv(
    provided: Optional[Path],
    *,
    snapshots_root: Path,
) -> Path:
    """Resolve --portfolio-csv to a real existing file. Fail closed.

    Spec 087 B1a — replaces the old rankings.csv stub fallback (root cause of
    the 2026-05-03 cron-misescalation incident).

    - If ``provided`` is given: return it iff it exists. Else SystemExit.
      An explicit operator-supplied path is honored as-is; we do not
      silently auto-discover a different file when the explicit one is
      missing. This protects the Friday-cron case where Friday's portfolio
      file must be present, not some prior day's.
    - If ``provided`` is None: auto-discover the latest
      ``data/snapshots/YYYY-MM-DD/portfolio_positions.csv``. SystemExit if
      no snapshot CSV exists anywhere.

    Never falls back to rankings.csv as a portfolio source.
    """
    if provided is not None:
        if provided.exists() and provided.is_file():
            return provided
        raise SystemExit(f"--portfolio-csv {provided} does not exist; refusing to emit hedge_report")
    if not snapshots_root.exists() or not snapshots_root.is_dir():
        raise SystemExit(
            f"--portfolio-csv not provided and snapshots root {snapshots_root} "
            "does not exist; refusing to emit hedge_report"
        )
    candidates = sorted(
        snapshots_root.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]/portfolio_positions.csv"),
        reverse=True,
    )
    if not candidates:
        raise SystemExit(
            f"--portfolio-csv not provided and no portfolio_positions.csv found under "
            f"{snapshots_root}/YYYY-MM-DD/; refusing to emit hedge_report"
        )
    resolved = candidates[0]
    logger.info("Auto-discovered --portfolio-csv: %s", resolved)
    return resolved


def load_portfolio_weights(portfolio_csv: Path) -> Tuple[Dict[str, float], str]:
    """Load portfolio weights from a guaranteed-existing CSV.

    Caller is responsible for resolving and validating ``portfolio_csv`` —
    use :func:`resolve_portfolio_csv`. Does NOT fall back to rankings.csv
    as a portfolio source (Spec 087 B1a; the rankings fallback was the
    2026-05-03 cron-misescalation root cause).

    Returns ({ticker: weight}, source_desc). Raises SystemExit if no
    usable weight column is present.
    """
    weights: Dict[str, float] = {}
    with open(portfolio_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if rows and "weight" in rows[0]:
        for r in rows:
            tk = r.get("ticker", "").strip()
            w = r.get("weight", "")
            if tk and w:
                try:
                    weights[tk] = float(w)
                except ValueError:
                    continue
    elif rows and "market_value" in rows[0]:
        total = 0.0
        for r in rows:
            tk = r.get("ticker", "").strip()
            mv = r.get("market_value", "")
            if tk and mv:
                try:
                    val = float(mv)
                    weights[tk] = val
                    total += val
                except ValueError:
                    continue
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
    elif rows and "target_weight_pct" in rows[0]:
        for r in rows:
            tk = r.get("ticker", "").strip()
            w = r.get("target_weight_pct", "")
            if tk and w:
                try:
                    weights[tk] = float(w) / 100.0
                except ValueError:
                    continue
    if not weights:
        raise SystemExit(
            f"--portfolio-csv {portfolio_csv} has no usable weight column "
            "(expected one of: weight, market_value, target_weight_pct); "
            "refusing to emit hedge_report"
        )
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    return weights, f"portfolio file ({portfolio_csv.name})"


def compute_log_returns(
    prices: Dict[str, float],
    dates: List[str],
) -> List[Optional[float]]:
    """Compute daily log returns for sorted dates.  Returns list aligned to dates[1:]."""
    rets: List[Optional[float]] = []
    for i in range(1, len(dates)):
        p0 = prices.get(dates[i - 1])
        p1 = prices.get(dates[i])
        if p0 and p1 and p0 > 0 and p1 > 0:
            rets.append(math.log(p1 / p0))
        else:
            rets.append(None)
    return rets


def compute_beta_stats(
    portfolio_weights: Dict[str, float],
    all_prices: Dict[str, Dict[str, float]],
    etf_ticker: str,
    as_of_date: str,
    window: int = 60,
) -> Dict[str, Any]:
    """OLS regression of portfolio returns on ETF returns."""
    etf_prices = all_prices.get(etf_ticker, {})
    if not etf_prices:
        return {"beta": None, "r_squared": None, "correlation": None, "etf_price": None, "realized_vol": None}

    # Build sorted date list for ETF
    all_dates = sorted(d for d in etf_prices if d <= as_of_date)
    if len(all_dates) < window + 1:
        logger.warning("Only %d dates for %s (need %d)", len(all_dates), etf_ticker, window + 1)
        window = max(len(all_dates) - 1, 10)
    dates = all_dates[-(window + 1) :]

    # ETF returns
    etf_rets = compute_log_returns(etf_prices, dates)

    # Portfolio returns (weighted sum)
    port_rets: List[Optional[float]] = []
    for i in range(len(etf_rets)):
        dt = dates[i + 1]
        port_r = 0.0
        total_w = 0.0
        for tk, wt in portfolio_weights.items():
            tk_prices = all_prices.get(tk, {})
            prev_dt = dates[i]
            p0 = tk_prices.get(prev_dt)
            p1 = tk_prices.get(dt)
            if p0 and p1 and p0 > 0 and p1 > 0:
                port_r += wt * math.log(p1 / p0)
                total_w += wt
        if total_w > 0:
            port_rets.append(port_r / total_w * sum(portfolio_weights.values()))
        else:
            port_rets.append(None)

    # Filter to days where both are available
    pairs = [(p, e) for p, e in zip(port_rets, etf_rets) if p is not None and e is not None]
    if len(pairs) < 10:
        return {"beta": None, "r_squared": None, "correlation": None, "etf_price": None, "realized_vol": None}

    py = [p[0] for p in pairs]
    ex = [p[1] for p in pairs]

    # OLS: beta = cov(port, etf) / var(etf)
    n = len(py)
    mean_p = sum(py) / n
    mean_e = sum(ex) / n
    cov_pe = sum((py[i] - mean_p) * (ex[i] - mean_e) for i in range(n)) / n
    var_e = sum((ex[i] - mean_e) ** 2 for i in range(n)) / n
    var_p = sum((py[i] - mean_p) ** 2 for i in range(n)) / n

    beta = cov_pe / var_e if var_e > 0 else None
    alpha = mean_p - beta * mean_e if beta is not None else None

    # R-squared
    r_sq = None
    if var_p > 0 and var_e > 0 and beta is not None:
        ss_res = sum((py[i] - alpha - beta * ex[i]) ** 2 for i in range(n))
        ss_tot = sum((py[i] - mean_p) ** 2 for i in range(n))
        r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

    # Correlation
    corr = None
    if var_p > 0 and var_e > 0:
        corr = cov_pe / math.sqrt(var_p * var_e)

    # Realized vol (annualized)
    port_vol = math.sqrt(var_p * TRADING_DAYS_PER_YEAR) if var_p > 0 else None
    etf_vol = math.sqrt(var_e * TRADING_DAYS_PER_YEAR) if var_e > 0 else None

    # Current ETF price
    etf_price = etf_prices.get(as_of_date) or etf_prices.get(dates[-1])

    return {
        "beta": round(beta, 4) if beta is not None else None,
        "r_squared": round(r_sq, 4) if r_sq is not None else None,
        "correlation": round(corr, 4) if corr is not None else None,
        "etf_price": round(etf_price, 2) if etf_price else None,
        "portfolio_realized_vol": round(port_vol, 4) if port_vol is not None else None,
        "etf_realized_vol": round(etf_vol, 4) if etf_vol is not None else None,
        "n_observations": n,
    }


def compute_concentration_metrics(
    portfolio_weights: Dict[str, float],
    rankings_csv: Optional[Path],
) -> Dict[str, Any]:
    """Compute portfolio concentration risk metrics."""
    sorted_w = sorted(portfolio_weights.items(), key=lambda x: -x[1])
    top5 = sorted_w[:5]
    herfindahl = sum(w * w for _, w in portfolio_weights.items())

    result: Dict[str, Any] = {
        "n_positions": len(portfolio_weights),
        "top_5": [(tk, round(w, 4)) for tk, w in top5],
        "herfindahl": round(herfindahl, 4),
    }

    # Join to rankings for catalyst and phase data
    if rankings_csv and rankings_csv.exists():
        with open(rankings_csv, newline="", encoding="utf-8") as f:
            rank_rows = {r["ticker"].strip(): r for r in csv.DictReader(f)}

        hard_cat_weight = 0.0
        phase_weights: Dict[str, float] = {}
        for tk, wt in portfolio_weights.items():
            rr = rank_rows.get(tk, {})
            cat_days = rr.get("catalyst_days", "")
            cat_fam = rr.get("catalyst_family", "")
            try:
                cd = float(cat_days)
                if cd <= 90 and cat_fam in ("REGULATORY", "CLINICAL"):
                    hard_cat_weight += wt
            except (ValueError, TypeError):
                pass
            phase = rr.get("lead_program_phase", rr.get("archetype", "unknown"))
            phase_weights[phase] = phase_weights.get(phase, 0.0) + wt

        result["hard_catalyst_weight_90d"] = round(hard_cat_weight, 4)
        result["phase_weights"] = {k: round(v, 4) for k, v in sorted(phase_weights.items(), key=lambda x: -x[1])[:5]}

    return result


# ---------------------------------------------------------------------------
# Phase 2 — Options surface analysis
# ---------------------------------------------------------------------------


def compute_realized_vol(
    prices: Dict[str, float],
    as_of_date: str,
    window: int = 30,
) -> Optional[float]:
    """Trailing N-day annualized realized vol from daily close prices."""
    sorted_dates = sorted(d for d in prices if d <= as_of_date)
    if len(sorted_dates) < window + 1:
        return None
    dates = sorted_dates[-(window + 1) :]
    rets = []
    for i in range(1, len(dates)):
        p0 = prices.get(dates[i - 1], 0)
        p1 = prices.get(dates[i], 0)
        if p0 > 0 and p1 > 0:
            rets.append(math.log(p1 / p0))
    if len(rets) < window // 2:
        return None
    mean_r = sum(rets) / len(rets)
    var = sum((r - mean_r) ** 2 for r in rets) / len(rets)
    return math.sqrt(var * TRADING_DAYS_PER_YEAR)


def get_expiries_for_analysis(
    chain: List[Dict[str, Any]],
    as_of_date: str,
) -> List[Dict[str, Any]]:
    """Select monthly and quarterly expiries for analysis.

    Returns list of {expiry, dte, label} sorted by DTE.
    """
    ref = date.fromisoformat(as_of_date)
    raw_expiries = sorted(set(c.get("expiration_date", "") for c in chain if c.get("expiration_date", "") > as_of_date))

    results = []
    monthly_count = 0
    for exp in raw_expiries:
        try:
            exp_dt = date.fromisoformat(exp)
        except ValueError:
            continue
        dte = (exp_dt - ref).days
        if dte < 7:
            continue
        if dte <= 45 and monthly_count < 2:
            results.append({"expiry": exp, "dte": dte, "label": f"near_{monthly_count + 1}"})
            monthly_count += 1
        elif 45 < dte <= 95:
            results.append({"expiry": exp, "dte": dte, "label": "mid"})
        elif 95 < dte <= 200:
            results.append({"expiry": exp, "dte": dte, "label": "far"})
    return results[:4]  # cap at 4 expiries


def analyze_options_surface(
    etf_ticker: str,
    chain: List[Dict[str, Any]],
    etf_price: float,
    as_of_date: str,
    realized_vol_30d: Optional[float],
    all_prices: Dict[str, Dict[str, float]],
    tasty_diag: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Analyze IV term structure, skew, VRP for an ETF chain.

    If tasty_diag is provided (from Tastytrade), uses its live IV, skew,
    and term structure data as the primary surface source.  Chain data
    from Massive is still used for per-strike pricing in structure eval.
    """
    # --- Tastytrade path: richer live diagnostics ---
    if tasty_diag and tasty_diag.get("opt_has_data") == "1":
        atm_iv = _safe_float(tasty_diag.get("opt_atm_iv"))
        front_iv = _safe_float(tasty_diag.get("opt_front_iv"))
        back_iv = _safe_float(tasty_diag.get("opt_back_iv"))
        term_slope = _safe_float(tasty_diag.get("opt_term_slope"))
        skew = _safe_float(tasty_diag.get("opt_put_call_skew"))
        rr_25d = _safe_float(tasty_diag.get("opt_rr_25d"))
        nearest_exp = tasty_diag.get("opt_nearest_expiry", "")
        dte = _safe_int(tasty_diag.get("opt_dte"))

        term_struct = []
        if front_iv is not None:
            term_struct.append({"expiry": nearest_exp, "dte": dte or 0, "label": "near_1", "atm_iv": front_iv})
        if back_iv is not None:
            term_struct.append({"expiry": "", "dte": 0, "label": "near_2", "atm_iv": back_iv})

        vrp = round(atm_iv - realized_vol_30d, 4) if atm_iv is not None and realized_vol_30d is not None else None
        vrp_pct = _compute_vrp_percentile(
            all_prices.get(etf_ticker, {}),
            as_of_date,
            atm_iv,
            realized_vol_30d,
        )

        # Straddle from Tastytrade IV (theoretical, not chain prices)
        straddles = []
        if atm_iv is not None and dte and dte > 0:
            T = dte / 365.0
            imp_move_pct = atm_iv * math.sqrt(T)
            rv_equiv = compute_realized_vol(all_prices.get(etf_ticker, {}), as_of_date, window=dte)
            real_move = rv_equiv * math.sqrt(dte / TRADING_DAYS_PER_YEAR) if rv_equiv else None
            regime = "fair"
            if real_move and real_move > 0:
                ratio = imp_move_pct / real_move
                regime = "expensive" if ratio > 1.5 else "cheap" if ratio < 0.8 else "fair"
            straddles.append(
                {
                    "expiry": nearest_exp,
                    "dte": dte,
                    "straddle_price": None,  # no chain-priced straddle from TT
                    "implied_move_pct": round(imp_move_pct, 4),
                    "realized_move_pct": round(real_move, 4) if real_move else None,
                    "cost_regime": regime,
                }
            )

        pricing_field = "tasty_mark_iv"
        return {
            "ticker": etf_ticker,
            "data_source": "tastytrade",
            "pricing_field_used": pricing_field,
            "etf_price": etf_price,
            "atm_iv_near": atm_iv,
            "realized_vol_30d": round(realized_vol_30d, 4) if realized_vol_30d else None,
            "term_structure": term_struct,
            "term_slope": term_slope,
            "skew_25d": skew,
            "skew_10d": None,  # not available from TT metrics
            "rr_25d": rr_25d,
            "vrp": vrp,
            "vrp_percentile": vrp_pct,
            "iv_regime": tasty_diag.get("opt_iv_regime", ""),
            "event_premium": tasty_diag.get("opt_event_premium", ""),
            "liquidity_ok": tasty_diag.get("opt_liquidity_ok", ""),
            "straddles": straddles,
        }

    # --- Massive chain / realized vol path ---
    result: Dict[str, Any] = {
        "ticker": etf_ticker,
        "data_source": "massive_chain" if chain else "realized_vol_proxy",
        "pricing_field_used": "day_close" if chain else "realized_vol_proxy",
        "etf_price": etf_price,
    }

    if not chain:
        result["atm_iv_near"] = realized_vol_30d
        result["realized_vol_30d"] = realized_vol_30d
        result["vrp"] = 0.0
        result["vrp_percentile"] = None
        result["skew_25d"] = None
        result["skew_10d"] = None
        result["term_structure"] = []
        result["straddles"] = []
        return result

    expiries = get_expiries_for_analysis(chain, as_of_date)
    if not expiries:
        result["data_source"] = "realized_vol_proxy"
        result["atm_iv_near"] = realized_vol_30d
        result["realized_vol_30d"] = realized_vol_30d
        result["vrp"] = 0.0
        result["term_structure"] = []
        result["straddles"] = []
        return result

    # Term structure: ATM IV per expiry
    term_struct = []
    for exp_info in expiries:
        exp = exp_info["expiry"]
        atm_call, atm_put = find_atm_contracts(chain, etf_price, exp)
        atm_iv = None
        if atm_call and atm_call.get("implied_volatility"):
            atm_iv = atm_call["implied_volatility"]
        elif atm_put and atm_put.get("implied_volatility"):
            atm_iv = atm_put["implied_volatility"]
        term_struct.append(
            {
                "expiry": exp,
                "dte": exp_info["dte"],
                "label": exp_info["label"],
                "atm_iv": round(atm_iv, 4) if atm_iv else None,
            }
        )
    result["term_structure"] = term_struct

    near_iv = next((t["atm_iv"] for t in term_struct if t["atm_iv"]), None)
    result["atm_iv_near"] = near_iv

    # Term slope
    ivs_with_dte = [(t["dte"], t["atm_iv"]) for t in term_struct if t["atm_iv"]]
    if len(ivs_with_dte) >= 2:
        far_iv = ivs_with_dte[-1][1]
        near_iv_val = ivs_with_dte[0][1]
        dt_years = (ivs_with_dte[-1][0] - ivs_with_dte[0][0]) / 365.0
        result["term_slope"] = round((far_iv - near_iv_val) / dt_years, 4) if dt_years > 0 else None
    else:
        result["term_slope"] = None

    # Put skew (nearest expiry with data)
    best_exp = expiries[0]["expiry"] if expiries else None
    if best_exp:
        put_25d, call_25d = find_25delta_contracts(chain, best_exp)
        if put_25d and near_iv:
            put_25d_iv = put_25d.get("implied_volatility")
            result["skew_25d"] = round(put_25d_iv - near_iv, 4) if put_25d_iv else None
        else:
            result["skew_25d"] = None
        # 10-delta put: find contract closest to -0.10 delta
        puts_exp = [
            c
            for c in chain
            if c.get("expiration_date") == best_exp and c.get("contract_type") == "put" and c.get("delta") is not None
        ]
        put_10d = min(puts_exp, key=lambda c: abs(c["delta"] - (-0.10)), default=None) if puts_exp else None
        if put_10d and near_iv:
            p10_iv = put_10d.get("implied_volatility")
            result["skew_10d"] = round(p10_iv - near_iv, 4) if p10_iv else None
        else:
            result["skew_10d"] = None
    else:
        result["skew_25d"] = None
        result["skew_10d"] = None

    # VRP
    result["realized_vol_30d"] = round(realized_vol_30d, 4) if realized_vol_30d else None
    if near_iv is not None and realized_vol_30d is not None:
        result["vrp"] = round(near_iv - realized_vol_30d, 4)
    else:
        result["vrp"] = None

    # VRP percentile (trailing 1yr of daily VRP proxy)
    result["vrp_percentile"] = _compute_vrp_percentile(
        all_prices.get(etf_ticker, {}), as_of_date, near_iv, realized_vol_30d
    )

    # Straddle / implied move per expiry
    straddles = []
    for exp_info in expiries:
        strad = compute_atm_straddle(chain, etf_price, exp_info["expiry"])
        imp_move = strad.get("actual_implied_move")
        # Compare to realized move over equivalent period
        rv_equiv = compute_realized_vol(all_prices.get(etf_ticker, {}), as_of_date, window=exp_info["dte"])
        realized_move = None
        if rv_equiv is not None:
            realized_move = rv_equiv * math.sqrt(exp_info["dte"] / TRADING_DAYS_PER_YEAR)

        regime = "fair"
        if imp_move and realized_move and realized_move > 0:
            ratio = imp_move / realized_move
            if ratio > 1.5:
                regime = "expensive"
            elif ratio < 0.8:
                regime = "cheap"

        straddles.append(
            {
                "expiry": exp_info["expiry"],
                "dte": exp_info["dte"],
                "straddle_price": strad.get("straddle_price"),
                "implied_move_pct": round(imp_move, 4) if imp_move else None,
                "realized_move_pct": round(realized_move, 4) if realized_move else None,
                "cost_regime": regime,
            }
        )
    result["straddles"] = straddles

    return result


def _compute_vrp_percentile(
    etf_prices: Dict[str, float],
    as_of_date: str,
    current_iv: Optional[float],
    current_rv: Optional[float],
) -> Optional[float]:
    """VRP percentile vs trailing 1-year history."""
    if current_iv is None or current_rv is None:
        return None
    current_vrp = current_iv - current_rv

    sorted_dates = sorted(d for d in etf_prices if d <= as_of_date)
    if len(sorted_dates) < 280:
        return None

    year_dates = sorted_dates[-280:]
    vrps = []
    for i in range(30, len(year_dates)):
        window_dates = year_dates[i - 30 : i + 1]
        rets = []
        for j in range(1, len(window_dates)):
            p0 = etf_prices.get(window_dates[j - 1], 0)
            p1 = etf_prices.get(window_dates[j], 0)
            if p0 > 0 and p1 > 0:
                rets.append(math.log(p1 / p0))
        if len(rets) >= 15:
            # Use RV=0 as proxy (no historical IV available)
            vrps.append(0.0)

    if not vrps:
        return None

    # Rank current VRP among historical (using 0 as baseline since we
    # only have realized vol history, the percentile indicates how
    # unusual today's VRP is relative to a zero-VRP baseline)
    below = sum(1 for v in vrps if v <= current_vrp)
    return round(below / len(vrps), 2)


# ---------------------------------------------------------------------------
# Phase 3 — Hedge structure evaluation
# ---------------------------------------------------------------------------

STRIKE_OFFSETS = {
    "ATM": 0.0,
    "5% OTM": -0.05,
    "10% OTM": -0.10,
    "15% OTM": -0.15,
    "20% OTM": -0.20,
    "25% OTM": -0.25,
}


def _bs_put_price(S: float, K: float, T: float, sigma: float, r: float = DEFAULT_RISK_FREE) -> float:
    """BS put price wrapper."""
    g = black_scholes_greeks(S, K, T, r, sigma, "put")
    return g["price"]


def _bs_call_price(S: float, K: float, T: float, sigma: float, r: float = DEFAULT_RISK_FREE) -> float:
    """BS call price wrapper."""
    g = black_scholes_greeks(S, K, T, r, sigma, "call")
    return g["price"]


def compute_hedge_contracts(
    hedge_notional: float,
    beta: float,
    etf_price: float,
) -> int:
    """Number of option contracts to hedge notional given beta."""
    if etf_price <= 0:
        return 0
    return max(1, round(hedge_notional * abs(beta) / (etf_price * 100)))


def evaluate_structures(
    etf_ticker: str,
    etf_price: float,
    sigma: float,
    dte: int,
    expiry: str,
    beta: float,
    hedge_notional: float,
    chain: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Evaluate all hedge structures for one ETF at one expiry."""
    if etf_price <= 0 or sigma <= 0 or dte <= 0:
        return []

    T = dte / 365.0
    contracts = compute_hedge_contracts(hedge_notional, beta, etf_price)
    annualization = 365.0 / dte
    structures: List[Dict[str, Any]] = []

    # ------- Structure 1: Straight puts -------
    for label, offset in [("ATM", 0.0), ("5% OTM", -0.05), ("10% OTM", -0.10), ("15% OTM", -0.15)]:
        K = round(etf_price * (1.0 + offset), 2)
        price = _find_chain_price(chain, expiry, K, "put") or _bs_put_price(etf_price, K, T, sigma)
        if math.isnan(price) or price <= 0:
            continue
        greeks = black_scholes_greeks(etf_price, K, T, DEFAULT_RISK_FREE, sigma, "put")
        total_cost = price * contracts * 100
        ann_bps = (total_cost * annualization) / hedge_notional * 10000 if hedge_notional > 0 else 0
        structures.append(
            {
                "vehicle": etf_ticker,
                "structure": f"Straight put {label}",
                "legs": 1,
                "expiry": expiry,
                "dte": dte,
                "strike_1": K,
                "premium_per_contract": round(price, 2),
                "delta": round(greeks["delta"], 4),
                "contracts": contracts,
                "total_cost": round(total_cost, 0),
                "ann_cost_bps": round(ann_bps, 0),
                "protection_start_pct": round(offset * 100 + (price / etf_price * 100), 1),
                "max_protection_pct": 100.0,  # full downside below breakeven
                "breakeven": round(K - price, 2),
                "type": "straight_put",
            }
        )

    # ------- Structure 2: Put spreads -------
    spread_configs = [
        ("5/15 put spread", -0.05, -0.15),
        ("10/20 put spread", -0.10, -0.20),
    ]
    for label, buy_off, sell_off in spread_configs:
        K_buy = round(etf_price * (1.0 + buy_off), 2)
        K_sell = round(etf_price * (1.0 + sell_off), 2)
        p_buy = _find_chain_price(chain, expiry, K_buy, "put") or _bs_put_price(etf_price, K_buy, T, sigma)
        p_sell = _find_chain_price(chain, expiry, K_sell, "put") or _bs_put_price(etf_price, K_sell, T, sigma)
        if math.isnan(p_buy) or math.isnan(p_sell):
            continue
        net = p_buy - p_sell
        if net <= 0:
            continue
        max_prot = K_buy - K_sell
        total_cost = net * contracts * 100
        ann_bps = (total_cost * annualization) / hedge_notional * 10000 if hedge_notional > 0 else 0
        structures.append(
            {
                "vehicle": etf_ticker,
                "structure": f"Put spread {label}",
                "legs": 2,
                "expiry": expiry,
                "dte": dte,
                "strike_1": K_buy,
                "strike_2": K_sell,
                "premium_per_contract": round(net, 2),
                "delta": None,
                "contracts": contracts,
                "total_cost": round(total_cost, 0),
                "ann_cost_bps": round(ann_bps, 0),
                "protection_start_pct": round(buy_off * 100 + (net / etf_price * 100), 1),
                "max_protection_pct": round(max_prot / etf_price * 100, 1),
                "breakeven": round(K_buy - net, 2),
                "type": "put_spread",
            }
        )

    # ------- Structure 3: Collars -------
    collar_configs = [
        ("5% OTM put collar", -0.05),
        ("10% OTM put collar", -0.10),
    ]
    for label, put_off in collar_configs:
        K_put = round(etf_price * (1.0 + put_off), 2)
        p_put = _find_chain_price(chain, expiry, K_put, "put") or _bs_put_price(etf_price, K_put, T, sigma)
        if math.isnan(p_put) or p_put <= 0:
            continue
        # Solve for call strike that produces ~zero cost
        # Binary search for K_call where call_price ≈ put_price
        lo = etf_price * 1.01
        hi = etf_price * 1.30
        K_call = etf_price * 1.05  # fallback
        for _ in range(30):
            mid = (lo + hi) / 2.0
            c_price = _find_chain_price(chain, expiry, round(mid, 2), "call") or _bs_call_price(
                etf_price, mid, T, sigma
            )
            if math.isnan(c_price):
                break
            if c_price > p_put:
                lo = mid
            else:
                hi = mid
            K_call = mid
        K_call = round(K_call, 2)
        c_price = _find_chain_price(chain, expiry, K_call, "call") or _bs_call_price(etf_price, K_call, T, sigma)
        net = p_put - c_price if not math.isnan(c_price) else p_put
        total_cost = net * contracts * 100
        ann_bps = (total_cost * annualization) / hedge_notional * 10000 if hedge_notional > 0 else 0
        upside_cap = (K_call / etf_price - 1.0) * 100
        structures.append(
            {
                "vehicle": etf_ticker,
                "structure": f"Collar {label}",
                "legs": 2,
                "expiry": expiry,
                "dte": dte,
                "strike_1": K_put,
                "strike_2": K_call,
                "premium_per_contract": round(net, 2),
                "delta": None,
                "contracts": contracts,
                "total_cost": round(total_cost, 0),
                "ann_cost_bps": round(ann_bps, 0),
                "protection_start_pct": round(put_off * 100, 1),
                "max_protection_pct": 100.0,
                "upside_cap_pct": round(upside_cap, 1),
                "breakeven": round(K_put - net, 2) if net > 0 else K_put,
                "type": "collar",
            }
        )

    # ------- Structure 4: Put ratio spread -------
    K_buy_ratio = round(etf_price * 0.90, 2)  # 10% OTM
    K_sell_ratio = round(etf_price * 0.75, 2)  # 25% OTM
    p_buy_r = _find_chain_price(chain, expiry, K_buy_ratio, "put") or _bs_put_price(etf_price, K_buy_ratio, T, sigma)
    p_sell_r = _find_chain_price(chain, expiry, K_sell_ratio, "put") or _bs_put_price(etf_price, K_sell_ratio, T, sigma)
    if not math.isnan(p_buy_r) and not math.isnan(p_sell_r):
        net_ratio = p_buy_r - 2 * p_sell_r
        danger_pct = (K_sell_ratio / etf_price - 1.0) * 100
        if danger_pct < -30:  # only recommend if danger zone > 30% below
            total_cost = net_ratio * contracts * 100
            ann_bps = (total_cost * annualization) / hedge_notional * 10000 if hedge_notional > 0 else 0
            structures.append(
                {
                    "vehicle": etf_ticker,
                    "structure": "Put ratio 1x2 (10/25 OTM)",
                    "legs": 3,
                    "expiry": expiry,
                    "dte": dte,
                    "strike_1": K_buy_ratio,
                    "strike_2": K_sell_ratio,
                    "premium_per_contract": round(net_ratio, 2),
                    "delta": None,
                    "contracts": contracts,
                    "total_cost": round(total_cost, 0),
                    "ann_cost_bps": round(ann_bps, 0),
                    "protection_start_pct": -10.0,
                    "max_protection_pct": round(abs(danger_pct), 1),
                    "danger_zone_pct": round(danger_pct, 1),
                    "breakeven": round(K_buy_ratio - net_ratio, 2) if net_ratio > 0 else K_buy_ratio,
                    "type": "put_ratio",
                }
            )

    return structures


def _find_chain_price(
    chain: List[Dict[str, Any]],
    expiry: str,
    strike: float,
    contract_type: str,
) -> Optional[float]:
    """Find day_close price from chain for exact or nearest strike."""
    if not chain:
        return None
    candidates = [c for c in chain if c.get("expiration_date") == expiry and c.get("contract_type") == contract_type]
    if not candidates:
        return None
    # Find nearest strike
    best = min(candidates, key=lambda c: abs(c.get("strike_price", 0) - strike))
    price = best.get("day_close")
    if price is not None and price > 0:
        return price
    return None


def score_structures(structures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Score and rank hedge structures."""
    if not structures:
        return []

    # Normalize metrics for scoring
    costs = [s["ann_cost_bps"] for s in structures if s["ann_cost_bps"] is not None]
    max_cost = max(costs) if costs else 1
    min_cost = min(costs) if costs else 0

    for s in structures:
        cost = s.get("ann_cost_bps", 0) or 0
        # Cost score: lower is better (invert)
        if max_cost > min_cost:
            s["cost_score"] = round(100 * (1.0 - (cost - min_cost) / (max_cost - min_cost)), 1)
        else:
            s["cost_score"] = 50.0

        # Protection score: deeper/wider is better
        max_prot = abs(s.get("max_protection_pct", 0))
        s["protection_score"] = round(min(100, max_prot * 3), 1)  # 33% protection → 100

        # Simplicity: fewer legs better
        legs = s.get("legs", 1)
        s["simplicity_score"] = round({1: 100, 2: 70, 3: 40}.get(legs, 20), 1)

        # Tail score: protection below -20%
        prot_start = abs(s.get("protection_start_pct", 0))
        max_p = abs(s.get("max_protection_pct", 0))
        if max_p >= 20:
            s["tail_score"] = round(min(100, max_p * 2.5), 1)
        elif prot_start <= 5:
            s["tail_score"] = 60.0  # ATM-ish gets partial credit
        else:
            s["tail_score"] = round(max_p * 2, 1)

        s["hedge_score"] = round(
            0.35 * s["cost_score"]
            + 0.30 * s["protection_score"]
            + 0.15 * s["simplicity_score"]
            + 0.20 * s["tail_score"],
            1,
        )

    structures.sort(key=lambda s: -s["hedge_score"])
    for i, s in enumerate(structures):
        s["rank"] = i + 1
    return structures


# ---------------------------------------------------------------------------
# DTE bucketing + Greeks (Spec 029)
# ---------------------------------------------------------------------------

DTE_BUCKETS = [
    ("short", 21, 35),
    ("medium_short", 36, 60),
    ("medium", 61, 90),
    ("long", 91, 120),
]


def bucket_expiries_by_dte(
    chain: List[Dict[str, Any]],
    as_of_date: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """Group chain expiries into DTE buckets.

    Returns {bucket_label: [{expiry, dte}, ...]}.
    """
    ref = date.fromisoformat(as_of_date)
    raw = sorted(set(c.get("expiration_date", "") for c in chain if c.get("expiration_date", "") > as_of_date))

    buckets: Dict[str, List[Dict[str, Any]]] = {label: [] for label, _, _ in DTE_BUCKETS}
    for exp in raw:
        try:
            dte = (date.fromisoformat(exp) - ref).days
        except ValueError:
            continue
        for label, lo, hi in DTE_BUCKETS:
            if lo <= dte <= hi:
                buckets[label].append({"expiry": exp, "dte": dte})
                break
    return buckets


def evaluate_all_dte_buckets(
    etf_ticker: str,
    etf_price: float,
    sigma: float,
    beta: float,
    hedge_notional: float,
    chain: List[Dict[str, Any]],
    as_of_date: str,
) -> List[Dict[str, Any]]:
    """Evaluate structures across all DTE buckets for one ETF.

    Returns flat list of scored structures, each tagged with dte_bucket.
    """
    buckets = bucket_expiries_by_dte(chain, as_of_date)
    all_structs: List[Dict[str, Any]] = []

    for label, expiry_list in buckets.items():
        if not expiry_list:
            continue
        # Pick the expiry with most contracts (proxy for liquidity)
        best_exp = expiry_list[0]  # already sorted by date; pick first
        structs = evaluate_structures(
            etf_ticker,
            etf_price,
            sigma,
            best_exp["dte"],
            best_exp["expiry"],
            beta,
            hedge_notional,
            chain,
        )
        for s in structs:
            s["dte_bucket"] = label
            s["_ref_price"] = etf_price
        all_structs.extend(structs)

    return all_structs


def compute_structure_greeks(
    structure: Dict[str, Any],
    etf_price: float,
    sigma: float,
) -> Dict[str, Any]:
    """Compute full Greeks for a hedge structure.

    Returns per-leg Greeks, net structure Greeks, and hedge-position Greeks.
    """
    dte = structure.get("dte", 45)
    T = dte / 365.0 if dte > 0 else 0.001
    contracts = structure.get("contracts", 1)
    struct_type = structure["type"]

    legs: List[Dict[str, Any]] = []

    if struct_type == "straight_put":
        K = structure.get("strike_1", etf_price * 0.95)
        legs.append(_leg_greeks("buy_put", K, T, etf_price, sigma, "put", 1))

    elif struct_type == "put_spread":
        K1 = structure.get("strike_1", etf_price * 0.95)
        K2 = structure.get("strike_2", etf_price * 0.85)
        legs.append(_leg_greeks("buy_put", K1, T, etf_price, sigma, "put", 1))
        legs.append(_leg_greeks("sell_put", K2, T, etf_price, sigma, "put", -1))

    elif struct_type == "collar":
        K1 = structure.get("strike_1", etf_price * 0.95)
        K2 = structure.get("strike_2", etf_price * 1.05)
        legs.append(_leg_greeks("buy_put", K1, T, etf_price, sigma, "put", 1))
        legs.append(_leg_greeks("sell_call", K2, T, etf_price, sigma, "call", -1))

    elif struct_type == "put_ratio":
        K1 = structure.get("strike_1", etf_price * 0.90)
        K2 = structure.get("strike_2", etf_price * 0.75)
        legs.append(_leg_greeks("buy_put", K1, T, etf_price, sigma, "put", 1))
        legs.append(_leg_greeks("sell_put_2x", K2, T, etf_price, sigma, "put", -2))

    # Net structure Greeks (per contract)
    net = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    for leg in legs:
        qty = leg["quantity_sign"]
        for g in net:
            val = leg.get(g, 0) or 0
            if not math.isnan(val):
                net[g] += val * qty

    per_contract = {k: round(v, 6) for k, v in net.items()}

    # Hedge-position Greeks (scaled by contracts * 100)
    scale = contracts * 100
    position = {
        "position_delta": round(net["delta"] * scale, 2),
        "position_gamma": round(net["gamma"] * scale, 4),
        "position_vega": round(net["vega"] * scale, 2),
        "position_theta": round(net["theta"] * scale, 2),
        "position_rho": round(net["rho"] * scale, 2),
        "theta_per_day_dollars": round(net["theta"] * scale, 2),
        "vega_pnl_per_1vol_point_dollars": round(net["vega"] * scale, 2),
    }

    return {
        "per_leg_greeks": legs,
        "per_contract_net_greeks": per_contract,
        "hedge_position_greeks": position,
        "implied_vol": round(sigma, 4),
        "contracts": contracts,
    }


def _leg_greeks(
    label: str,
    strike: float,
    T: float,
    S: float,
    sigma: float,
    option_type: str,
    quantity_sign: int,
) -> Dict[str, Any]:
    """Compute Greeks for a single leg."""
    g = black_scholes_greeks(S, strike, T, DEFAULT_RISK_FREE, sigma, option_type)
    return {
        "label": label,
        "option_type": option_type,
        "strike": round(strike, 2),
        "quantity_sign": quantity_sign,
        "price": g.get("price"),
        "implied_vol": round(sigma, 4),
        "delta": g.get("delta"),
        "gamma": g.get("gamma"),
        "vega": g.get("vega"),
        "theta": g.get("theta"),
        "rho": g.get("rho"),
    }


def _structure_summary(s: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key fields from a structure for summary output."""
    if not s:
        return {}
    return {
        "rank": s.get("rank"),
        "vehicle": s.get("vehicle"),
        "structure": s.get("structure"),
        "expiry": s.get("expiry"),
        "dte": s.get("dte"),
        "dte_bucket": s.get("dte_bucket"),
        "ann_cost_bps": s.get("ann_cost_bps"),
        "hedge_score": s.get("hedge_score"),
    }


def rank_best_dte_candidates(
    bucket_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Select best structure per DTE bucket and overall best.

    Decision rule for best_overall (priority order):
    1. higher historical down-month payoff
    2. lower max drawdown
    3. lower annualized carry
    4. higher historical coverage
    5. higher hedge score
    """
    # Group by bucket
    by_bucket: Dict[str, List[Dict[str, Any]]] = {}
    for r in bucket_results:
        b = r.get("dte_bucket", "unknown")
        by_bucket.setdefault(b, []).append(r)

    # Pick winner per bucket (by hedge_score since backtest is expensive)
    bucket_winners: Dict[str, Dict[str, Any]] = {}
    for label, items in by_bucket.items():
        items.sort(key=lambda x: -(x.get("hedge_score", 0)))
        bucket_winners[label] = items[0]

    # Best overall = highest hedge score among bucket winners
    # (historical metrics are added in Phase 4 after backtest runs)
    all_winners = list(bucket_winners.values())
    if all_winners:
        all_winners.sort(key=lambda x: -(x.get("hedge_score", 0)))
        best_overall = all_winners[0]
    else:
        best_overall = {}

    return {
        "bucket_winners": bucket_winners,
        "best_overall": best_overall,
    }


# ---------------------------------------------------------------------------
# Phase 4 — Historical hedge effectiveness backtest
# ---------------------------------------------------------------------------


def backtest_structure(
    structure: Dict[str, Any],
    etf_prices: Dict[str, float],
    portfolio_returns_monthly: List[Dict[str, Any]],
    sigma: float,
    hedge_notional: float,
) -> Dict[str, Any]:
    """Simulate monthly hedge P&L for a structure over trailing year."""
    struct_type = structure["type"]
    offsets = _infer_strike_offsets(structure)

    months = portfolio_returns_monthly
    results = []
    total_hedge_pnl = 0.0

    for m in months:
        month_start = m["start_date"]
        month_end = m["end_date"]
        port_ret = m["portfolio_return"]

        etf_start = etf_prices.get(month_start, 0)
        etf_end = etf_prices.get(month_end, 0)
        if etf_start <= 0 or etf_end <= 0:
            continue

        etf_ret = (etf_end - etf_start) / etf_start
        dte = max((date.fromisoformat(month_end) - date.fromisoformat(month_start)).days, 1)
        T = dte / 365.0

        hedge_pnl = _simulate_structure_pnl(
            struct_type, offsets, etf_start, etf_end, T, sigma, hedge_notional, structure.get("contracts", 1)
        )

        results.append(
            {
                "month_start": month_start,
                "month_end": month_end,
                "portfolio_return": round(port_ret, 4),
                "etf_return": round(etf_ret, 4),
                "hedge_pnl": round(hedge_pnl, 0),
            }
        )
        total_hedge_pnl += hedge_pnl

    if not results:
        return {"months": [], "summary": {}}

    # Summary stats
    port_rets = [r["portfolio_return"] for r in results]
    hedged_rets = [r["portfolio_return"] + r["hedge_pnl"] / hedge_notional for r in results]

    payoff_months = sum(1 for r in results if r["portfolio_return"] < 0 and r["hedge_pnl"] > 0)
    cost_months = sum(1 for r in results if r["portfolio_return"] > 0 and r["hedge_pnl"] < 0)

    def _max_dd(rets: List[float]) -> float:
        peak = 1.0
        max_dd = 0.0
        cum = 1.0
        for r in rets:
            cum *= 1 + r
            peak = max(peak, cum)
            dd = (peak - cum) / peak
            max_dd = max(max_dd, dd)
        return max_dd

    def _sharpe(rets: List[float]) -> Optional[float]:
        if len(rets) < 3:
            return None
        m = sum(rets) / len(rets)
        v = sum((r - m) ** 2 for r in rets) / len(rets)
        if v <= 0:
            return None
        return m / math.sqrt(v) * math.sqrt(12)  # annualized from monthly

    summary = {
        "total_months": len(results),
        "payoff_months": payoff_months,
        "cost_months": cost_months,
        "total_hedge_pnl": round(total_hedge_pnl, 0),
        "total_return_unhedged": round(sum(port_rets), 4),
        "total_return_hedged": round(sum(hedged_rets), 4),
        "worst_month_unhedged": round(min(port_rets), 4),
        "worst_month_hedged": round(min(hedged_rets), 4),
        "max_drawdown_unhedged": round(_max_dd(port_rets), 4),
        "max_drawdown_hedged": round(_max_dd(hedged_rets), 4),
        "sharpe_unhedged": round(_sharpe(port_rets), 2) if _sharpe(port_rets) else None,
        "sharpe_hedged": round(_sharpe(hedged_rets), 2) if _sharpe(hedged_rets) else None,
    }

    return {"months": results, "summary": summary}


def _infer_strike_offsets(structure: Dict[str, Any]) -> Dict[str, float]:
    """Infer strike offsets from structure type."""
    st = structure["type"]
    name = structure["structure"]
    if st == "straight_put":
        if "ATM" in name:
            return {"put": 0.0}
        elif "5%" in name:
            return {"put": -0.05}
        elif "10%" in name:
            return {"put": -0.10}
        elif "15%" in name:
            return {"put": -0.15}
    elif st == "put_spread":
        if "5/15" in name:
            return {"buy_put": -0.05, "sell_put": -0.15}
        elif "10/20" in name:
            return {"buy_put": -0.10, "sell_put": -0.20}
    elif st == "collar":
        if "5%" in name:
            return {"put": -0.05, "call": 0.05}
        elif "10%" in name:
            return {"put": -0.10, "call": 0.05}
    elif st == "put_ratio":
        return {"buy_put": -0.10, "sell_put": -0.25}
    return {"put": -0.05}


def _simulate_structure_pnl(
    struct_type: str,
    offsets: Dict[str, float],
    etf_start: float,
    etf_end: float,
    T: float,
    sigma: float,
    hedge_notional: float,
    contracts: int,
) -> float:
    """Simulate hedge P&L for one period."""
    if struct_type == "straight_put":
        K = etf_start * (1 + offsets.get("put", -0.05))
        entry = _bs_put_price(etf_start, K, T, sigma)
        exit_val = max(K - etf_end, 0)
        return (exit_val - entry) * contracts * 100

    elif struct_type == "put_spread":
        K_buy = etf_start * (1 + offsets.get("buy_put", -0.05))
        K_sell = etf_start * (1 + offsets.get("sell_put", -0.15))
        entry_buy = _bs_put_price(etf_start, K_buy, T, sigma)
        entry_sell = _bs_put_price(etf_start, K_sell, T, sigma)
        exit_buy = max(K_buy - etf_end, 0)
        exit_sell = max(K_sell - etf_end, 0)
        return ((exit_buy - entry_buy) - (exit_sell - entry_sell)) * contracts * 100

    elif struct_type == "collar":
        K_put = etf_start * (1 + offsets.get("put", -0.05))
        K_call = etf_start * (1 + offsets.get("call", 0.05))
        entry_put = _bs_put_price(etf_start, K_put, T, sigma)
        entry_call = _bs_call_price(etf_start, K_call, T, sigma)
        exit_put = max(K_put - etf_end, 0)
        exit_call = max(etf_end - K_call, 0)
        return ((exit_put - entry_put) - (exit_call - entry_call)) * contracts * 100

    elif struct_type == "put_ratio":
        K_buy = etf_start * (1 + offsets.get("buy_put", -0.10))
        K_sell = etf_start * (1 + offsets.get("sell_put", -0.25))
        entry_buy = _bs_put_price(etf_start, K_buy, T, sigma)
        entry_sell = _bs_put_price(etf_start, K_sell, T, sigma)
        exit_buy = max(K_buy - etf_end, 0)
        exit_sell = max(K_sell - etf_end, 0)
        return ((exit_buy - entry_buy) - 2 * (exit_sell - entry_sell)) * contracts * 100

    return 0.0


def compute_monthly_returns(
    portfolio_weights: Dict[str, float],
    all_prices: Dict[str, Dict[str, float]],
    as_of_date: str,
    months: int = 12,
) -> List[Dict[str, Any]]:
    """Compute monthly portfolio returns for trailing N months."""
    ref = date.fromisoformat(as_of_date)
    results = []

    for m in range(months):
        end_dt = ref - timedelta(days=30 * m)
        start_dt = end_dt - timedelta(days=30)

        # Find closest trading dates
        sample_ticker = next(iter(portfolio_weights), "")
        sample_prices = all_prices.get(sample_ticker, {})
        all_dates = sorted(sample_prices.keys())
        if not all_dates:
            continue

        start_str = _nearest_date(all_dates, start_dt.isoformat())
        end_str = _nearest_date(all_dates, end_dt.isoformat())
        if not start_str or not end_str or start_str >= end_str:
            continue

        # Portfolio return
        port_ret = 0.0
        total_w = 0.0
        for tk, wt in portfolio_weights.items():
            tk_prices = all_prices.get(tk, {})
            p0 = tk_prices.get(start_str)
            p1 = tk_prices.get(end_str)
            if p0 and p1 and p0 > 0 and p1 > 0:
                port_ret += wt * (p1 / p0 - 1.0)
                total_w += wt
        if total_w > 0:
            port_ret = port_ret / total_w
        results.append(
            {
                "start_date": start_str,
                "end_date": end_str,
                "portfolio_return": port_ret,
            }
        )

    return list(reversed(results))


def _nearest_date(sorted_dates: List[str], target: str) -> Optional[str]:
    """Find nearest date <= target in a sorted list."""
    best = None
    for d in sorted_dates:
        if d <= target:
            best = d
        else:
            break
    return best


def compute_regime_analysis(
    backtest_results: List[Dict[str, Any]],
    etf_prices: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Split months into up/flat/down regimes."""
    regimes: Dict[str, List[Dict[str, Any]]] = {"up": [], "flat": [], "down": []}

    for m in backtest_results:
        etf_ret = m.get("etf_return", 0)
        if etf_ret > 0.03:
            regimes["up"].append(m)
        elif etf_ret < -0.03:
            regimes["down"].append(m)
        else:
            regimes["flat"].append(m)

    results = []
    for label, months in [
        ("Up (>3%)", regimes["up"]),
        ("Flat (±3%)", regimes["flat"]),
        ("Down (<-3%)", regimes["down"]),
    ]:
        n = len(months)
        avg_pnl = sum(m.get("hedge_pnl", 0) for m in months) / n if n > 0 else 0
        avg_port = sum(m.get("portfolio_return", 0) for m in months) / n if n > 0 else 0
        results.append(
            {
                "regime": label,
                "months": n,
                "avg_hedge_pnl": round(avg_pnl, 0),
                "avg_portfolio_return": round(avg_port, 4),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Phase 5 — Report generation
# ---------------------------------------------------------------------------


def generate_markdown_report(report_data: Dict[str, Any]) -> str:
    """Generate the IC-ready markdown report."""
    d = report_data
    lines: List[str] = []

    lines.append(f"# Biotech Portfolio Hedge Report — {d['as_of_date']}\n")

    # IC Decision Banner
    ic = d.get("ic_decision", {})
    if ic:
        action = ic.get("policy_action", "WATCH")
        action_emoji = {"HEDGE NOW": ">>", "WATCH": "--", "DEFER": "!!"}
        # Get top structure Greeks for banner
        top_greeks = d.get("structure_greeks", {}).get(1, {})
        top_pos = top_greeks.get("hedge_position_greeks", {})
        top_ranked = d.get("ranked_structures", [{}])
        top_dte = top_ranked[0].get("dte", "") if top_ranked else ""
        top_expiry = top_ranked[0].get("expiry", "") if top_ranked else ""
        vega_str = f"${top_pos.get('vega_pnl_per_1vol_point_dollars', 0):+,.0f}/+1vol" if top_pos else ""
        theta_str = f"${top_pos.get('theta_per_day_dollars', 0):,.0f}/day" if top_pos else ""

        banner = (
            f"**[{action_emoji.get(action, '--')} {action}]** | "
            f"Primary: **{ic.get('primary_hedge', 'none')}** | "
            f"Expiry: {top_expiry} ({top_dte} DTE) | "
            f"Vega: {vega_str} | Theta: {theta_str} | "
            f"Confidence: **{ic.get('confidence', 'N/A')}** | "
            f"{ic.get('change_reason', '')}"
        )
        lines.append(banner)
        lines.append("")
        if ic.get("policy_reasons"):
            for reason in ic["policy_reasons"]:
                lines.append(f"- {reason}")
        if ic.get("secondary_hedge") and ic["secondary_hedge"] != "none":
            sec_line = (
                f"- Secondary alternative: {ic['secondary_hedge']} " f"({ic.get('secondary_cost_bps', 0):.0f} bps)"
            )
            lines.append(sec_line)
        lines.append("")

    # Portfolio summary
    lines.append("## Portfolio Summary\n")
    ps = d.get("portfolio_summary", {})
    lines.append(f"- **Portfolio source**: {d.get('portfolio_source', 'unknown')}")
    lines.append(f"- **Hedge notional**: ${d.get('hedge_notional', 0):,.0f}")
    lines.append(f"- **Positions**: {ps.get('n_positions', 0)}")
    top5 = ps.get("top_5", [])
    if top5:
        top5_str = ", ".join(f"{tk} ({w:.1%})" for tk, w in top5)
        lines.append(f"- **Top 5**: {top5_str}")
    lines.append(f"- **Herfindahl**: {ps.get('herfindahl', 0):.4f}")
    if ps.get("hard_catalyst_weight_90d") is not None:
        lines.append(f"- **Weight in hard-catalyst names (90d)**: {ps['hard_catalyst_weight_90d']:.1%}")
    lines.append("")

    # Hedge vehicle comparison
    lines.append("## Hedge Vehicle Comparison\n")
    lines.append("| Metric | XBI | IBB |")
    lines.append("|---|---|---|")
    xbi = d.get("beta_stats", {}).get("XBI", {})
    ibb = d.get("beta_stats", {}).get("IBB", {})
    xs = d.get("surface", {}).get("XBI", {})
    is_ = d.get("surface", {}).get("IBB", {})

    def _fmt(v: Any, fmt: str = ".4f") -> str:
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return f"{v:{fmt}}"
        return str(v)

    lines.append(f"| Beta to portfolio | {_fmt(xbi.get('beta'))} | {_fmt(ibb.get('beta'))} |")
    lines.append(f"| R-squared | {_fmt(xbi.get('r_squared'))} | {_fmt(ibb.get('r_squared'))} |")
    lines.append(f"| Correlation | {_fmt(xbi.get('correlation'))} | {_fmt(ibb.get('correlation'))} |")
    lines.append(f"| Current price | {_fmt(xbi.get('etf_price'), '.2f')} | {_fmt(ibb.get('etf_price'), '.2f')} |")
    lines.append(f"| ATM IV (near) | {_fmt_pct(xs.get('atm_iv_near'))} | {_fmt_pct(is_.get('atm_iv_near'))} |")
    lines.append(
        f"| 30d realized vol | {_fmt_pct(xs.get('realized_vol_30d'))} | {_fmt_pct(is_.get('realized_vol_30d'))} |"
    )
    lines.append(f"| VRP | {_fmt_pct(xs.get('vrp'))} | {_fmt_pct(is_.get('vrp'))} |")
    lines.append(
        f"| VRP percentile (1y) | {_fmt_ptile(xs.get('vrp_percentile'))} | {_fmt_ptile(is_.get('vrp_percentile'))} |"
    )

    # Implied move from straddles
    xbi_strad = xs.get("straddles", [{}])
    ibb_strad = is_.get("straddles", [{}])
    xbi_imp = xbi_strad[0].get("implied_move_pct") if xbi_strad else None
    ibb_imp = ibb_strad[0].get("implied_move_pct") if ibb_strad else None
    xbi_regime = xbi_strad[0].get("cost_regime", "N/A") if xbi_strad else "N/A"
    ibb_regime = ibb_strad[0].get("cost_regime", "N/A") if ibb_strad else "N/A"
    lines.append(f"| Implied move (nearest) | {_fmt_pct(xbi_imp)} | {_fmt_pct(ibb_imp)} |")
    lines.append(f"| Protection cost regime | {xbi_regime} | {ibb_regime} |")

    # Show skew/RR if available (from Tastytrade or chain)
    xbi_skew = xs.get("skew_25d")
    ibb_skew = is_.get("skew_25d")
    xbi_rr = xs.get("rr_25d")
    ibb_rr = is_.get("rr_25d")
    if xbi_skew is not None or ibb_skew is not None:
        lines.append(f"| Put skew (25d) | {_fmt_pct(xbi_skew)} | {_fmt_pct(ibb_skew)} |")
    if xbi_rr is not None or ibb_rr is not None:
        lines.append(f"| RR 25d | {_fmt(xbi_rr)} | {_fmt(ibb_rr)} |")

    # Data source flags
    xbi_src = xs.get("data_source", "unknown")
    ibb_src = is_.get("data_source", "unknown")
    lines.append(f"| Data source | {xbi_src} | {ibb_src} |")
    lines.append("")

    # Best vehicle
    best_etf = d.get("best_hedge_vehicle", "XBI")
    lines.append(f"**Better hedge vehicle**: {best_etf} (higher R-squared)\n")

    # Primary recommendation
    ranked = d.get("ranked_structures", [])
    if ranked:
        top = ranked[0]
        lines.append(f"## Recommended Primary Hedge: {top['structure']}\n")
        lines.append(f"- **Vehicle**: {top['vehicle']}")
        lines.append(f"- **Structure**: {top['structure']}")
        lines.append(f"- **Expiry**: {top.get('expiry', 'N/A')} ({top.get('dte', 0)} DTE)")
        lines.append(f"- **Contracts**: {top.get('contracts', 0)}")
        lines.append(
            f"- **Total cost**: ${top.get('total_cost', 0):,.0f} ({top.get('ann_cost_bps', 0):.0f} bps annualized)"
        )
        lines.append(
            f"- **Protection starts**: {top.get('protection_start_pct', 0):.1f}% below spot (breakeven at ${top.get('breakeven', 0):,.2f})"
        )
        lines.append(f"- **Breakeven**: ETF at ${top.get('breakeven', 0):,.2f}")
        lines.append(f"- **Hedge score**: {top.get('hedge_score', 0):.1f}")
        if top.get("type") == "collar":
            lines.append(f"- **Key risk**: upside capped at {top.get('upside_cap_pct', 0):.1f}%")
        elif top.get("type") == "put_ratio":
            lines.append(
                f"- **Key risk**: naked below ${top.get('strike_2', 0):,.2f} ({top.get('danger_zone_pct', 0):.0f}%)"
            )
        else:
            lines.append("- **Key risk**: premium decay if market stays flat/up")
        lines.append("")

    # All structures table
    lines.append("## All Structures Evaluated\n")
    lines.append("| Rank | Vehicle | Structure | Expiry | Cost (bps ann) | Strike(s) | Score |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in ranked:
        strike_info = f"${s.get('strike_1', 0):,.0f}"
        if s.get("strike_2"):
            strike_info += f"/${s['strike_2']:,.0f}"
        lines.append(
            f"| {s['rank']} | {s['vehicle']} | {s['structure']} | {s.get('expiry', '')} "
            f"| {s.get('ann_cost_bps', 0):.0f} "
            f"| {strike_info} "
            f"| {s.get('hedge_score', 0):.1f} |"
        )
    lines.append("")

    # Historical effectiveness
    bt = d.get("backtest", {})
    bs = bt if "total_months" in bt else bt.get("summary", {})
    if bs:
        hist_m = bs.get("historical_months", 0)
        total_m = bs.get("total_months", 0)
        bt_mode = bs.get("backtest_pricing", "bs")

        header = "## Historical Effectiveness (trailing 12mo)"
        if hist_m > 0:
            pct_hist = hist_m / total_m * 100 if total_m > 0 else 0
            header += f" — {hist_m}/{total_m} months from actual option closes ({pct_hist:.0f}%)"
        lines.append(header + "\n")

        lines.append("| Metric | Unhedged | Hedged (top pick) |")
        lines.append("|---|---|---|")
        lines.append(f"| Return | {bs.get('total_return_unhedged', 0):.2%} | {bs.get('total_return_hedged', 0):.2%} |")
        lines.append(
            f"| Max drawdown | {bs.get('max_drawdown_unhedged', 0):.2%} | {bs.get('max_drawdown_hedged', 0):.2%} |"
        )
        lines.append(
            f"| Worst month | {bs.get('worst_month_unhedged', 0):.2%} | {bs.get('worst_month_hedged', 0):.2%} |"
        )
        sh_u = bs.get("sharpe_unhedged")
        sh_h = bs.get("sharpe_hedged")
        lines.append(f"| Sharpe | {sh_u:.2f} | {sh_h:.2f} |" if sh_u and sh_h else "| Sharpe | N/A | N/A |")
        lines.append("")

        # BS vs Historical comparison if both are available
        bs_bt = d.get("bs_backtest_comparison")
        if bs_bt and bt_mode in ("historical", "mixed"):
            lines.append("### BS vs Historical Backtest Comparison\n")
            lines.append("| Metric | BS-only | Historical |")
            lines.append("|---|---|---|")
            lines.append(
                f"| Hedged return | {bs_bt.get('total_return_hedged', 0):.2%} "
                f"| {bs.get('total_return_hedged', 0):.2%} |"
            )
            lines.append(
                f"| Hedged max DD | {bs_bt.get('max_drawdown_hedged', 0):.2%} "
                f"| {bs.get('max_drawdown_hedged', 0):.2%} |"
            )
            lines.append(
                f"| Hedged worst month | {bs_bt.get('worst_month_hedged', 0):.2%} "
                f"| {bs.get('worst_month_hedged', 0):.2%} |"
            )
            bs_sh = bs_bt.get("sharpe_hedged")
            lines.append(
                f"| Hedged Sharpe | {bs_sh:.2f} | {sh_h:.2f} |" if bs_sh and sh_h else "| Hedged Sharpe | N/A | N/A |"
            )
            lines.append(
                f"| Total hedge P&L | ${bs_bt.get('total_hedge_pnl', 0):,.0f} "
                f"| ${bs.get('total_hedge_pnl', 0):,.0f} |"
            )
            lines.append("")
            lines.append(
                "> **IC note**: BS-based hedge backtests likely understate realized "
                "payoff of OTM put hedges due to the market volatility risk premium "
                "(VRP). Historical option closes capture this premium and produce a "
                "more accurate picture of actual hedge effectiveness."
            )
            lines.append("")

    # Top-N backtest comparison
    top_n = d.get("top_n_backtests", [])
    if len(top_n) > 1:
        lines.append("### Top Structure Backtest Comparison\n")
        lines.append("| Rank | Vehicle | Structure | Pricing | Hedged Return | Max DD | Hedge P&L | Sharpe |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for entry in top_n:
            s = entry.get("backtest_summary", {})
            pm = entry.get("pricing_mode", "bs")
            sh = s.get("sharpe_hedged")
            sh_str = f"{sh:.2f}" if sh else "N/A"
            lines.append(
                f"| {entry['rank']} | {entry['vehicle']} | {entry['structure']} "
                f"| {pm} "
                f"| {s.get('total_return_hedged', 0):.2%} "
                f"| {s.get('max_drawdown_hedged', 0):.2%} "
                f"| ${s.get('total_hedge_pnl', 0):,.0f} "
                f"| {sh_str} |"
            )
        lines.append("")

    # Best DTE Comparison (Spec 029)
    dte_summary = d.get("best_dte_summary", {})
    bw = dte_summary.get("bucket_winners", {})
    if bw:
        lines.append("## Best DTE Comparison\n")
        lines.append("| Bucket | Vehicle | Structure | Expiry | DTE | Carry (bps) | Score |")
        lines.append("|---|---|---|---|---|---|---|")
        for label in ["short", "medium_short", "medium", "long"]:
            w = bw.get(label, {})
            if w:
                lines.append(
                    f"| {label} | {w.get('vehicle', '')} | {w.get('structure', '')} "
                    f"| {w.get('expiry', '')} | {w.get('dte', '')} "
                    f"| {w.get('ann_cost_bps', 0):.0f} | {w.get('hedge_score', 0):.1f} |"
                )
        best_o = dte_summary.get("best_overall", {})
        if best_o:
            lines.append(
                f"| **overall** | {best_o.get('vehicle', '')} | {best_o.get('structure', '')} "
                f"| {best_o.get('expiry', '')} | {best_o.get('dte', '')} "
                f"| {best_o.get('ann_cost_bps', 0):.0f} | {best_o.get('hedge_score', 0):.1f} |"
            )
        lines.append("")

    # Greeks Summary (Spec 029)
    sg = d.get("structure_greeks", {})
    ranked_for_greeks = d.get("ranked_structures", [])[:5]
    if ranked_for_greeks and sg:
        lines.append("## Greeks Summary (hedge-position level)\n")
        lines.append("| Rank | Vehicle | Structure | DTE | Delta | Gamma | Vega ($/+1vol) | Theta ($/day) | IV |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for s in ranked_for_greeks:
            g = sg.get(s["rank"], {}).get("hedge_position_greeks", {})
            iv = sg.get(s["rank"], {}).get("implied_vol", 0)
            lines.append(
                f"| {s['rank']} | {s['vehicle']} | {s['structure']} "
                f"| {s.get('dte', '')} "
                f"| {g.get('position_delta', 0):,.0f} "
                f"| {g.get('position_gamma', 0):.2f} "
                f"| ${g.get('vega_pnl_per_1vol_point_dollars', 0):,.0f} "
                f"| ${g.get('theta_per_day_dollars', 0):,.0f} "
                f"| {iv:.1%} |"
            )
        lines.append("")

    # Shadow efficacy diagnostic
    se = d.get("shadow_efficacy", {})
    if se.get("status") == "ok":
        lines.append("## Static vs Historical Efficacy (shadow diagnostic)\n")
        if se.get("agree"):
            lines.append("Static scorer and historical efficacy **agree**: " f"**{se['static_winner']}**")
        else:
            lines.append("Static scorer and historical efficacy **disagree**:")
            lines.append("")
            lines.append("| | Static Winner | Efficacy Winner |")
            lines.append("|---|---|---|")
            lines.append(f"| Structure | {se['static_winner']} | {se['efficacy_winner']} |")
            lines.append(f"| Hedge score | {se['static_hedge_score']:.1f} | {se['efficacy_hedge_score']:.1f} |")
            lines.append(f"| Carry (bps ann) | {se['static_ann_cost_bps']:.0f} | {se['efficacy_ann_cost_bps']:.0f} |")
            s_dd = se.get("static_max_dd_hedged")
            e_dd = se.get("efficacy_max_dd_hedged")
            lines.append(
                f"| Max DD (hedged) | {s_dd:.2%} | {e_dd:.2%} |"
                if s_dd is not None and e_dd is not None
                else "| Max DD (hedged) | N/A | N/A |"
            )
            lines.append("")
            lines.append(
                f"DD gap: {se.get('dd_reduction_delta', 0):+.2%} | "
                f"Carry gap: {se.get('carry_delta_bps', 0):+.0f} bps"
            )
        lines.append("")

    # Regime preference (shadow classifier)
    rp = d.get("regime_preference", {})
    if rp.get("regime_preference"):
        pref = rp["regime_preference"]
        conf = rp.get("regime_confidence", "low")
        pref_display = {
            "collar_preferred": "Collar preferred",
            "otm_put_preferred": "OTM put preferred",
            "ambiguous": "Ambiguous",
        }.get(pref, pref)
        lines.append("## Regime Structure Preference (shadow)\n")
        lines.append(f"**{pref_display}** (confidence: {conf})\n")
        for reason in rp.get("regime_reasons", []):
            lines.append(f"- {reason}")
        lines.append("")

    # Regime analysis
    regimes = d.get("regime_analysis", [])
    if regimes:
        lines.append("## Regime Cost Profile\n")
        lines.append("| Market regime | Months | Avg hedge P&L | Avg portfolio return |")
        lines.append("|---|---|---|---|")
        for r in regimes:
            lines.append(
                f"| {r['regime']} | {r['months']} "
                f"| ${r['avg_hedge_pnl']:,.0f} "
                f"| {r['avg_portfolio_return']:.2%} |"
            )
        lines.append("")

    # Caveats
    lines.append("## Data Sources & Caveats\n")
    ph = d.get("price_history_range", {})
    lines.append(f"- **Price history**: {ph.get('start', 'N/A')} to {ph.get('end', 'N/A')}")

    # Mixed-source clarity: spell out what each source contributed
    src = d.get("options_source_used", "N/A")
    lines.append(f"- **Options source (primary)**: {src}")
    lines.append(f"- **Source selection reason**: {d.get('source_selection_reason', 'N/A')}")
    dtrace = d.get("decision_trace", {})
    for etf in HEDGE_ETFS:
        surf_src = dtrace.get("surface_source_by_etf", {}).get(etf, "N/A")
        price_src = dtrace.get("structure_pricing_source_by_etf", {}).get(etf, "N/A")
        lines.append(f"- **{etf}**: surface IV/skew from *{surf_src}*, structure pricing from *{price_src}*")
    fallbacks = dtrace.get("fallbacks_triggered", [])
    if fallbacks:
        lines.append(f"- **Fallbacks triggered**: {', '.join(fallbacks)}")
    if src == OPTIONS_SOURCE_TASTY:
        lines.append(
            "- **Note**: Tastytrade provides live IV, skew, and term structure. "
            "Contract-level pricing for hedge structures uses Massive chain "
            "day\\_close or BS theoretical prices — not Tastytrade quotes."
        )
    lines.append("- **Beta window**: 60 trading days")
    bt_pricing = dtrace.get("backtest_pricing", "bs")
    if bt_pricing == "historical":
        lines.append(
            "- **Backtest**: monthly rebalance, **historical option closes** (Massive day aggs), no transaction costs"
        )
    elif bt_pricing == "mixed":
        lines.append(
            "- **Backtest**: monthly rebalance, historical option closes + BS fallback (mixed), no transaction costs"
        )
    else:
        lines.append("- **Backtest**: monthly rebalance, BS pricing, no transaction costs")
    lines.append("- **This report is for IC discussion, not execution**")
    lines.append("")

    return "\n".join(lines)


def _fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    return f"{v:.1%}"


def _fmt_ptile(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    return f"{v:.0%}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


BACKTEST_AUTO = "auto"
BACKTEST_HISTORICAL = "historical"
BACKTEST_BS = "bs"


def _compute_shadow_efficacy(
    top_n_backtests: List[Dict[str, Any]],
    ranked: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compare static scorer winner vs historical-efficacy winner.

    Shadow diagnostic — does not change any ranking or recommendation.
    Surfaces whether the static score is systematically missing the
    better realized hedge structure.
    """
    if not top_n_backtests or len(top_n_backtests) < 2:
        return {"status": "insufficient_data", "n_candidates": len(top_n_backtests)}

    # Static winner = rank 1 by hedge_score
    static_winner = top_n_backtests[0]

    # Historical-efficacy winner = best by drawdown reduction
    # (lowest max_drawdown_hedged among candidates with historical data)
    candidates_with_bt = [e for e in top_n_backtests if e.get("backtest_summary", {}).get("total_months", 0) > 0]
    if not candidates_with_bt:
        return {"status": "no_backtest_data", "n_candidates": len(top_n_backtests)}

    efficacy_winner = min(
        candidates_with_bt,
        key=lambda e: e.get("backtest_summary", {}).get("max_drawdown_hedged", 1.0),
    )

    static_name = f"{static_winner.get('vehicle', '')} {static_winner.get('structure', '')}"
    efficacy_name = f"{efficacy_winner.get('vehicle', '')} {efficacy_winner.get('structure', '')}"
    agree = static_name == efficacy_name

    s_bt = static_winner.get("backtest_summary", {})
    e_bt = efficacy_winner.get("backtest_summary", {})

    return {
        "status": "ok",
        "n_candidates": len(top_n_backtests),
        "static_winner": static_name,
        "static_hedge_score": static_winner.get("hedge_score", 0),
        "static_ann_cost_bps": static_winner.get("ann_cost_bps", 0),
        "static_max_dd_hedged": s_bt.get("max_drawdown_hedged"),
        "static_down_avg_pnl": s_bt.get("total_hedge_pnl"),
        "efficacy_winner": efficacy_name,
        "efficacy_hedge_score": efficacy_winner.get("hedge_score", 0),
        "efficacy_ann_cost_bps": efficacy_winner.get("ann_cost_bps", 0),
        "efficacy_max_dd_hedged": e_bt.get("max_drawdown_hedged"),
        "efficacy_down_avg_pnl": e_bt.get("total_hedge_pnl"),
        "agree": agree,
        "dd_reduction_delta": (
            round((s_bt.get("max_drawdown_hedged", 0) or 0) - (e_bt.get("max_drawdown_hedged", 0) or 0), 4)
            if not agree
            else 0
        ),
        "carry_delta_bps": (
            round((static_winner.get("ann_cost_bps", 0) or 0) - (efficacy_winner.get("ann_cost_bps", 0) or 0), 0)
            if not agree
            else 0
        ),
    }


# Policy thresholds
CARRY_EXPENSIVE_BPS = 400  # above this, defer
CARRY_CHEAP_BPS = 50  # below this, hedge now (protection is cheap)
MIN_HISTORICAL_COVERAGE = 0.50  # at least 50% months from actual closes
CONFIDENCE_HIGH_COVERAGE = 0.75  # 75%+ historical for high confidence


def compute_ic_decision(report_data: Dict[str, Any]) -> Dict[str, Any]:
    """Compute IC decision block: recommendation banner, confidence, policy trigger.

    Returns dict with:
        primary_hedge, secondary_hedge, recommendation_changed, change_reason,
        confidence, confidence_drivers, policy_action, policy_reasons
    """
    ranked = report_data.get("ranked_structures", [])
    bt_summary = report_data.get("backtest", {})
    diff = report_data.get("weekly_diff")

    result: Dict[str, Any] = {}

    # --- Recommendation banner ---
    if ranked:
        top = ranked[0]
        result["primary_hedge"] = f"{top['vehicle']} {top['structure']}"
        result["primary_cost_bps"] = top.get("ann_cost_bps", 0)
        result["primary_score"] = top.get("hedge_score", 0)
    else:
        result["primary_hedge"] = "none"
        result["primary_cost_bps"] = 0
        result["primary_score"] = 0

    if len(ranked) >= 2:
        sec = ranked[1]
        result["secondary_hedge"] = f"{sec['vehicle']} {sec['structure']}"
        result["secondary_cost_bps"] = sec.get("ann_cost_bps", 0)
    else:
        result["secondary_hedge"] = "none"

    # Change vs prior week
    if diff:
        result["recommendation_changed"] = diff.get("structure_changed", False)
        if diff.get("structure_changed"):
            result["change_reason"] = (
                f"changed from {diff.get('top_structure_prior', '?')} " f"to {diff.get('top_structure_current', '?')}"
            )
        elif diff.get("vehicle_changed"):
            result["change_reason"] = (
                f"vehicle changed from {diff.get('best_vehicle_prior', '?')} "
                f"to {diff.get('best_vehicle_current', '?')}"
            )
        else:
            result["change_reason"] = "no change vs prior week"
    else:
        result["recommendation_changed"] = None
        result["change_reason"] = "no prior report for comparison"

    # --- Confidence ---
    drivers: List[str] = []
    conf_score = 0.0

    # Historical coverage quality (0-40 points)
    hist_m = bt_summary.get("historical_months", 0)
    total_m = bt_summary.get("total_months", 0)
    hist_pct = hist_m / total_m if total_m > 0 else 0
    if hist_pct >= CONFIDENCE_HIGH_COVERAGE:
        conf_score += 40
        drivers.append(f"historical coverage {hist_pct:.0%} (high)")
    elif hist_pct >= MIN_HISTORICAL_COVERAGE:
        conf_score += 25
        drivers.append(f"historical coverage {hist_pct:.0%} (adequate)")
    elif hist_pct > 0:
        conf_score += 10
        drivers.append(f"historical coverage {hist_pct:.0%} (low)")
    else:
        drivers.append("no historical option data (BS-only)")

    # Source quality (0-30 points)
    source = report_data.get("options_source_used", "")
    if source == OPTIONS_SOURCE_TASTY:
        conf_score += 30
        drivers.append("live Tastytrade IV/skew")
    elif source == OPTIONS_SOURCE_MASSIVE:
        conf_score += 20
        drivers.append("Massive chain data")
    else:
        conf_score += 5
        drivers.append("realized vol proxy only")

    # Stability vs prior week (0-30 points)
    if diff:
        if not diff.get("structure_changed") and not diff.get("vehicle_changed"):
            conf_score += 30
            drivers.append("stable vs prior week")
        elif not diff.get("structure_changed"):
            conf_score += 20
            drivers.append("structure stable, vehicle changed")
        else:
            conf_score += 5
            drivers.append("recommendation changed from prior week")
    else:
        conf_score += 15  # first run, neutral
        drivers.append("first report (no stability data)")

    if conf_score >= 80:
        confidence = "HIGH"
    elif conf_score >= 50:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    result["confidence"] = confidence
    result["confidence_score"] = round(conf_score, 0)
    result["confidence_drivers"] = drivers

    # --- Policy trigger ---
    cost_bps = result.get("primary_cost_bps", 0) or 0
    reasons: List[str] = []

    if confidence == "LOW":
        action = "DEFER"
        reasons.append("low confidence — insufficient data quality for hedge sizing")
    elif cost_bps > CARRY_EXPENSIVE_BPS:
        action = "WATCH"
        reasons.append(f"carry {cost_bps:.0f} bps > {CARRY_EXPENSIVE_BPS} bps threshold — protection expensive")
    elif cost_bps <= CARRY_CHEAP_BPS:
        action = "HEDGE NOW"
        reasons.append(f"carry {cost_bps:.0f} bps <= {CARRY_CHEAP_BPS} bps — protection is cheap")
    elif hist_pct < MIN_HISTORICAL_COVERAGE and total_m > 0:
        action = "WATCH"
        reasons.append(f"historical coverage {hist_pct:.0%} below {MIN_HISTORICAL_COVERAGE:.0%} minimum")
    elif confidence == "HIGH":
        action = "HEDGE NOW"
        reasons.append("high confidence, reasonable carry")
    else:
        action = "WATCH"
        reasons.append("medium confidence — review top-3 comparison before sizing")

    result["policy_action"] = action
    result["policy_reasons"] = reasons

    return result


# ---------------------------------------------------------------------------
# Weekly archive + diff
# ---------------------------------------------------------------------------


def _find_prior_report(
    archive_dir: Path,
    current_date: str,
) -> Optional[Dict[str, Any]]:
    """Find the most recent archived report before current_date."""
    if not archive_dir.exists():
        return None
    candidates = sorted(archive_dir.glob("hedge_report_*.json"), reverse=True)
    for path in candidates:
        try:
            report_date = path.stem.replace("hedge_report_", "")
            if report_date < current_date:
                return json.loads(path.read_text())
        except (ValueError, json.JSONDecodeError):
            continue
    return None


def _compute_weekly_diff(
    prior: Dict[str, Any],
    current: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute week-over-week changes between two bioshort reports."""
    diff: Dict[str, Any] = {
        "prior_date": prior.get("as_of_date", ""),
        "current_date": current.get("as_of_date", ""),
    }

    # Vehicle change
    diff["best_vehicle_prior"] = prior.get("best_hedge_vehicle", "")
    diff["best_vehicle_current"] = current.get("best_hedge_vehicle", "")
    diff["vehicle_changed"] = diff["best_vehicle_prior"] != diff["best_vehicle_current"]

    # Top structure change
    prior_ranked = prior.get("ranked_structures", [])
    current_ranked = current.get("ranked_structures", [])
    p_top = prior_ranked[0] if prior_ranked else {}
    c_top = current_ranked[0] if current_ranked else {}
    diff["top_structure_prior"] = f"{p_top.get('vehicle', '')} {p_top.get('structure', '')}"
    diff["top_structure_current"] = f"{c_top.get('vehicle', '')} {c_top.get('structure', '')}"
    diff["structure_changed"] = diff["top_structure_prior"] != diff["top_structure_current"]

    # Annualized carry
    diff["ann_cost_bps_prior"] = p_top.get("ann_cost_bps", 0)
    diff["ann_cost_bps_current"] = c_top.get("ann_cost_bps", 0)

    # Historical down-month payoff
    p_regime = prior.get("regime_analysis", [])
    c_regime = current.get("regime_analysis", [])
    p_down: dict = next((r for r in p_regime if "Down" in r.get("regime", "")), {})
    c_down: dict = next((r for r in c_regime if "Down" in r.get("regime", "")), {})
    diff["down_month_avg_pnl_prior"] = p_down.get("avg_hedge_pnl", 0)
    diff["down_month_avg_pnl_current"] = c_down.get("avg_hedge_pnl", 0)

    # Source mix
    p_trace = prior.get("decision_trace", {})
    c_trace = current.get("decision_trace", {})
    diff["source_prior"] = prior.get("options_source_used", "")
    diff["source_current"] = current.get("options_source_used", "")
    diff["backtest_pricing_prior"] = p_trace.get("backtest_pricing", "")
    diff["backtest_pricing_current"] = c_trace.get("backtest_pricing", "")

    # Beta changes
    for etf in HEDGE_ETFS:
        p_beta = prior.get("beta_stats", {}).get(etf, {})
        c_beta = current.get("beta_stats", {}).get(etf, {})
        diff[f"{etf}_beta_prior"] = p_beta.get("beta")
        diff[f"{etf}_beta_current"] = c_beta.get("beta")

    return diff


def _render_diff_markdown(diff: Dict[str, Any]) -> str:
    """Render the weekly diff as a markdown section."""
    if not diff.get("prior_date"):
        return ""

    lines: List[str] = []
    lines.append(f"## Week-over-Week Changes (vs {diff['prior_date']})\n")

    # Flags
    flags = []
    if diff.get("vehicle_changed"):
        flags.append(f"Hedge vehicle changed: {diff['best_vehicle_prior']} → {diff['best_vehicle_current']}")
    if diff.get("structure_changed"):
        flags.append(f"Top structure changed: {diff['top_structure_prior']} → {diff['top_structure_current']}")
    if flags:
        for f in flags:
            lines.append(f"- **{f}**")
        lines.append("")

    # Comparison table
    lines.append("| Metric | Prior | Current | Delta |")
    lines.append("|---|---|---|---|")

    cost_p = diff.get("ann_cost_bps_prior", 0) or 0
    cost_c = diff.get("ann_cost_bps_current", 0) or 0
    lines.append(f"| Annualized carry (bps) | {cost_p:.0f} | {cost_c:.0f} | {cost_c - cost_p:+.0f} |")

    down_p = diff.get("down_month_avg_pnl_prior", 0) or 0
    down_c = diff.get("down_month_avg_pnl_current", 0) or 0
    lines.append(f"| Down-month avg hedge P&L | ${down_p:,.0f} | ${down_c:,.0f} | ${down_c - down_p:+,.0f} |")

    for etf in HEDGE_ETFS:
        bp = diff.get(f"{etf}_beta_prior")
        bc = diff.get(f"{etf}_beta_current")
        if bp is not None and bc is not None:
            lines.append(f"| {etf} beta | {bp:.3f} | {bc:.3f} | {bc - bp:+.3f} |")

    lines.append(f"| Options source | {diff.get('source_prior', 'N/A')} | {diff.get('source_current', 'N/A')} | |")
    lines.append(
        f"| Backtest pricing | {diff.get('backtest_pricing_prior', 'N/A')} "
        f"| {diff.get('backtest_pricing_current', 'N/A')} | |"
    )
    lines.append("")

    return "\n".join(lines)


def run_hedge_report(
    as_of_date: str,
    portfolio_csv: Optional[Path],
    price_csv: Path,
    hedge_notional: float,
    output_dir: Path,
    snap_dir: Optional[Path] = None,
    options_source: str = OPTIONS_SOURCE_AUTO,
    backtest_mode: str = BACKTEST_AUTO,
    research_mode: bool = False,
) -> Dict[str, Any]:
    """Run the full hedge report pipeline.  Returns the report data dict."""
    mode_label = "[RESEARCH_BACKFILL]" if research_mode else ""
    logger.info("=== Biotech Portfolio Hedge Report — %s %s ===", as_of_date, mode_label)

    # Spec 087 B1a — resolve --portfolio-csv first; fail closed before any work.
    portfolio_csv = resolve_portfolio_csv(
        portfolio_csv,
        snapshots_root=REPO_ROOT / "data" / "snapshots",
    )

    # Resolve snapshot dir (rankings.csv is consumed only for catalyst/phase
    # metadata join in compute_concentration_metrics — never as a portfolio source).
    if snap_dir is None:
        snap_dir = REPO_ROOT / "data" / "snapshots" / as_of_date
    rankings_csv = snap_dir / "rankings.csv" if snap_dir.exists() else None

    # --- Phase 0: Price data ---
    logger.info("Phase 0: Loading price data...")
    all_prices = load_price_history(price_csv)
    all_prices = fetch_etf_prices(HEDGE_ETFS, price_csv, all_prices)

    # Price history date range
    etf_dates = sorted(all_prices.get("XBI", {}).keys())
    price_range = {"start": etf_dates[0] if etf_dates else "N/A", "end": etf_dates[-1] if etf_dates else "N/A"}

    # --- Phase 1: Portfolio exposure ---
    logger.info("Phase 1: Portfolio exposure analysis...")
    weights, portfolio_source = load_portfolio_weights(portfolio_csv)
    logger.info("Loaded %d positions from %s", len(weights), portfolio_source)

    # Beta stats for each ETF
    beta_stats: Dict[str, Dict[str, Any]] = {}
    for etf in HEDGE_ETFS:
        beta_stats[etf] = compute_beta_stats(weights, all_prices, etf, as_of_date)
        logger.info(
            "  %s: beta=%.3f, R²=%.3f, corr=%.3f",
            etf,
            beta_stats[etf].get("beta") or 0,
            beta_stats[etf].get("r_squared") or 0,
            beta_stats[etf].get("correlation") or 0,
        )

    # Best vehicle = higher R-squared
    best_etf = max(HEDGE_ETFS, key=lambda e: beta_stats[e].get("r_squared") or 0)

    # Concentration metrics
    concentration = compute_concentration_metrics(weights, rankings_csv)

    # --- Phase 2: Options surface ---
    logger.info("Phase 2: Options surface analysis...")
    source_used, source_reason = select_options_source(options_source)
    logger.info("  Options source: %s (%s)", source_used, source_reason)

    # Fetch Tastytrade diagnostics for ETFs if that source was selected
    tasty_diags: Dict[str, Dict[str, Any]] = {}
    if source_used == OPTIONS_SOURCE_TASTY:
        tasty_diags = fetch_tasty_diagnostics(HEDGE_ETFS, as_of_date)
        if not any(d.get("opt_has_data") == "1" for d in tasty_diags.values()):
            logger.warning("  Tastytrade returned no data; falling back")
            source_used = OPTIONS_SOURCE_MASSIVE if _massive_available() else "realized_vol"
            source_reason += " → tastytrade returned no data, fell back"

    surface: Dict[str, Dict[str, Any]] = {}

    for etf in HEDGE_ETFS:
        # Load Massive chain (used for per-strike pricing in structure eval
        # even when Tastytrade is the primary surface source)
        chain: List[Dict[str, Any]] = []
        if snap_dir and snap_dir.exists():
            chain = load_chain_snapshot(snap_dir, etf)
            if chain:
                logger.info("  %s: loaded cached chain (%d contracts)", etf, len(chain))

        if not chain and (source_used != OPTIONS_SOURCE_TASTY or not tasty_diags.get(etf, {}).get("opt_has_data")):
            try:
                from common.options_history_massive import fetch_chain_snapshot as _fetch

                chain = _fetch(etf, limit=250)
                if chain:
                    logger.info("  %s: fetched live chain (%d contracts)", etf, len(chain))
                    if snap_dir:
                        snap_dir.mkdir(parents=True, exist_ok=True)
                        save_chain_snapshot(chain, snap_dir, etf)
            except Exception as exc:
                logger.info("  %s: live chain fetch failed: %s", etf, exc)

        etf_price = beta_stats[etf].get("etf_price") or 0
        rv_30d = compute_realized_vol(all_prices.get(etf, {}), as_of_date, window=30)

        # Pass Tastytrade diag if available for this ETF
        tt_diag = tasty_diags.get(etf) if source_used == OPTIONS_SOURCE_TASTY else None
        surface[etf] = analyze_options_surface(
            etf,
            chain,
            etf_price,
            as_of_date,
            rv_30d,
            all_prices,
            tasty_diag=tt_diag,
        )

    # --- Phase 3: Hedge structures (multi-DTE, Spec 029) ---
    logger.info("Phase 3: Evaluating hedge structures across DTE buckets...")
    all_structures: List[Dict[str, Any]] = []

    for etf in HEDGE_ETFS:
        etf_price = beta_stats[etf].get("etf_price") or 0
        beta = beta_stats[etf].get("beta") or 1.0
        surf = surface.get(etf, {})
        sigma = surf.get("atm_iv_near") or surf.get("realized_vol_30d") or 0.30

        # Try loading chain for multi-DTE evaluation
        chain = []
        if snap_dir and snap_dir.exists():
            chain = load_chain_snapshot(snap_dir, etf)

        if chain:
            # Multi-DTE: evaluate across all buckets
            structs = evaluate_all_dte_buckets(
                etf,
                etf_price,
                sigma,
                beta,
                hedge_notional,
                chain,
                as_of_date,
            )
        else:
            # Fallback: single synthetic expiry
            target_expiry = (date.fromisoformat(as_of_date) + timedelta(days=45)).isoformat()
            structs = evaluate_structures(
                etf,
                etf_price,
                sigma,
                45,
                target_expiry,
                beta,
                hedge_notional,
                [],
            )
            for s in structs:
                s["dte_bucket"] = "medium_short"
                s["_ref_price"] = etf_price

        all_structures.extend(structs)

    ranked = score_structures(all_structures)

    # Compute Greeks for all ranked structures
    structure_greeks: Dict[int, Dict[str, Any]] = {}
    for s in ranked:
        etf_price_s = s.get("_ref_price") or beta_stats.get(s["vehicle"], {}).get("etf_price", 0)
        sigma_s = surface.get(s["vehicle"], {}).get("atm_iv_near") or 0.30
        greeks = compute_structure_greeks(s, etf_price_s, sigma_s)
        s["greeks"] = greeks
        structure_greeks[s["rank"]] = greeks

    # DTE bucket ranking
    best_dte = rank_best_dte_candidates(ranked)
    bucket_winners = best_dte.get("bucket_winners", {})
    n_buckets = sum(1 for v in bucket_winners.values() if v)
    logger.info(
        "  Evaluated %d structures across %d DTE buckets, top: %s",
        len(ranked),
        n_buckets,
        ranked[0]["structure"] if ranked else "none",
    )

    # --- Phase 4: Historical backtest ---
    logger.info("Phase 4: Historical backtest...")
    monthly_rets = compute_monthly_returns(weights, all_prices, as_of_date, months=12)

    TOP_N_BACKTEST = 3
    backtest = {}
    bs_comparison = {}
    regime_analysis = []
    backtest_pricing_mode = "bs"
    top_n_backtests: List[Dict[str, Any]] = []

    if ranked and monthly_rets:
        use_historical = backtest_mode in (BACKTEST_AUTO, BACKTEST_HISTORICAL)

        # Backtest top N structures
        for idx, struct in enumerate(ranked[:TOP_N_BACKTEST]):
            etf_bt = struct["vehicle"]
            offsets_bt = _infer_strike_offsets(struct)
            sigma_bt = (
                surface.get(etf_bt, {}).get("atm_iv_near") or surface.get(etf_bt, {}).get("realized_vol_30d") or 0.30
            )

            # BS backtest (always)
            bs_bt = backtest_structure(
                struct,
                all_prices.get(etf_bt, {}),
                monthly_rets,
                sigma_bt,
                hedge_notional,
            )

            # Historical backtest (if available)
            hist_bt = None
            if use_historical:
                try:
                    from common.historical_hedge_backtest import run_historical_backtest

                    hist_bt = run_historical_backtest(
                        struct,
                        offsets_bt,
                        etf_bt,
                        all_prices.get(etf_bt, {}),
                        monthly_rets,
                        hedge_notional,
                        struct.get("contracts", 1),
                    )
                    hs = hist_bt.get("summary", {})
                    if hs.get("historical_months", 0) == 0:
                        hist_bt = None
                except Exception:
                    hist_bt = None

            # Pick the better result
            best_bt = hist_bt if hist_bt else bs_bt
            best_summary = best_bt.get("summary", {})

            entry = {
                "rank": struct["rank"],
                "vehicle": struct["vehicle"],
                "structure": struct["structure"],
                "hedge_score": struct.get("hedge_score", 0),
                "ann_cost_bps": struct.get("ann_cost_bps", 0),
                "backtest_summary": best_summary,
                "bs_summary": bs_bt.get("summary", {}),
                "pricing_mode": best_summary.get("backtest_pricing", "bs"),
            }
            top_n_backtests.append(entry)

            if idx == 0:
                logger.info(
                    "  #%d %s %s: %s, hedge P&L=$%s",
                    struct["rank"],
                    struct["vehicle"],
                    struct["structure"],
                    entry["pricing_mode"],
                    f"{best_summary.get('total_hedge_pnl', 0):,.0f}",
                )

        # Primary backtest = top-ranked structure
        primary = top_n_backtests[0] if top_n_backtests else {}
        if primary:
            # Find the full backtest result for rank #1
            struct_0 = ranked[0]
            etf_0 = struct_0["vehicle"]
            offsets_0 = _infer_strike_offsets(struct_0)
            sigma_0 = (
                surface.get(etf_0, {}).get("atm_iv_near") or surface.get(etf_0, {}).get("realized_vol_30d") or 0.30
            )

            # Re-run to get full months array for regime analysis
            if use_historical:
                try:
                    from common.historical_hedge_backtest import run_historical_backtest

                    hist_bt = run_historical_backtest(
                        struct_0,
                        offsets_0,
                        etf_0,
                        all_prices.get(etf_0, {}),
                        monthly_rets,
                        hedge_notional,
                        struct_0.get("contracts", 1),
                    )
                    hs = hist_bt.get("summary", {})
                    if hs.get("historical_months", 0) > 0:
                        backtest = hist_bt
                        backtest_pricing_mode = hs.get("backtest_pricing", "mixed")
                        bs_bt_0 = backtest_structure(
                            struct_0,
                            all_prices.get(etf_0, {}),
                            monthly_rets,
                            sigma_0,
                            hedge_notional,
                        )
                        bs_comparison = bs_bt_0.get("summary", {})
                except Exception:
                    pass

            if not backtest:
                backtest = backtest_structure(
                    struct_0,
                    all_prices.get(etf_0, {}),
                    monthly_rets,
                    sigma_0,
                    hedge_notional,
                )
                backtest_pricing_mode = "bs"

        if backtest.get("months"):
            etf_for_regime = ranked[0]["vehicle"] if ranked else "XBI"
            regime_analysis = compute_regime_analysis(backtest["months"], all_prices.get(etf_for_regime, {}))

    # --- Phase 5: Assemble report data ---
    # Build decision trace for auditability
    fallbacks: List[str] = []
    surface_source_by_etf: Dict[str, str] = {}
    structure_pricing_by_etf: Dict[str, str] = {}
    for etf in HEDGE_ETFS:
        surf = surface.get(etf, {})
        ds = surf.get("data_source", "unknown")
        surface_source_by_etf[etf] = ds
        # Structure pricing comes from chain (Massive) or BS theoretical
        if load_chain_snapshot(snap_dir, etf) if snap_dir and snap_dir.exists() else []:
            structure_pricing_by_etf[etf] = "massive_chain_day_close"
        else:
            structure_pricing_by_etf[etf] = "bs_theoretical"
            if ds != "realized_vol_proxy":
                fallbacks.append(f"{etf}: no chain for structure pricing, used BS theoretical")
    if source_used == "realized_vol":
        fallbacks.append("no options API credentials; all surface data from realized vol")

    decision_trace: Dict[str, Any] = {
        "surface_source_by_etf": surface_source_by_etf,
        "structure_pricing_source_by_etf": structure_pricing_by_etf,
        "backtest_pricing": backtest_pricing_mode,
        "fallbacks_triggered": fallbacks,
    }

    report_data: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "as_of_date": as_of_date,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hedge_notional": hedge_notional,
        "portfolio_source": portfolio_source,
        "portfolio_summary": concentration,
        "beta_stats": beta_stats,
        "best_hedge_vehicle": best_etf,
        "surface": surface,
        "options_source_used": source_used,
        "source_selection_reason": source_reason,
        "decision_trace": decision_trace,
        "ranked_structures": ranked,
        "best_dte_summary": {
            "bucket_winners": {k: _structure_summary(v) for k, v in bucket_winners.items()},
            "best_overall": _structure_summary(best_dte.get("best_overall", {})),
        },
        "structure_greeks": structure_greeks,
        "backtest": backtest.get("summary", {}),
        "backtest_months": backtest.get("months", []),
        "bs_backtest_comparison": bs_comparison if bs_comparison else None,
        "top_n_backtests": top_n_backtests,
        "regime_analysis": regime_analysis,
        "price_history_range": price_range,
    }

    # Shadow efficacy comparison (Spec 029 research diagnostic)
    shadow_efficacy = _compute_shadow_efficacy(top_n_backtests, ranked)
    report_data["shadow_efficacy"] = shadow_efficacy

    # Shadow regime preference classifier
    try:
        from common.hedge_regime_classifier import classify_hedge_regime

        best_surf = surface.get(best_etf, {})
        best_strad = best_surf.get("straddles", [{}])
        regime_class = classify_hedge_regime(
            vrp=best_surf.get("vrp"),
            vrp_percentile=best_surf.get("vrp_percentile"),
            cost_regime=best_strad[0].get("cost_regime", "") if best_strad else "",
            implied_move_pct=best_strad[0].get("implied_move_pct") if best_strad else None,
            r_squared=beta_stats.get(best_etf, {}).get("r_squared"),
            skew_25d=best_surf.get("skew_25d"),
        )
        report_data["regime_preference"] = regime_class
        logger.info(
            "  Regime preference: %s (%s) — collar=%d, put=%d",
            regime_class["regime_preference"],
            regime_class["regime_confidence"],
            regime_class["collar_score"],
            regime_class["put_score"],
        )
    except Exception as _rp_exc:
        logger.debug("Regime classifier skipped: %s", _rp_exc)

    # Compute IC decision block (needs full report_data)
    ic_decision = compute_ic_decision(report_data)
    report_data["ic_decision"] = ic_decision

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    # Markdown
    md_path = output_dir / f"hedge_report_{as_of_date}.md"
    md_content = generate_markdown_report(report_data)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".md",
        dir=str(output_dir),
        delete=False,
    ) as tmp:
        tmp.write(md_content)
        tmp_name = tmp.name
    os.replace(tmp_name, str(md_path))
    logger.info("Phase 5: Wrote %s", md_path)

    # JSON
    json_path = output_dir / f"hedge_report_{as_of_date}.json"
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        dir=str(output_dir),
        delete=False,
    ) as tmp:
        json.dump(report_data, tmp, indent=2, default=str)
        tmp_name = tmp.name
    os.replace(tmp_name, str(json_path))
    logger.info("Phase 5: Wrote %s", json_path)

    # --- Weekly archive + diff (isolation in research mode) ---
    if research_mode:
        archive_dir = output_dir / "archive"
        report_data["mode"] = "research_backfill"
    else:
        archive_dir = REPO_ROOT / "output" / "hedge_report" / "archive"

    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_json = archive_dir / f"hedge_report_{as_of_date}.json"
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        dir=str(archive_dir),
        delete=False,
    ) as tmp:
        json.dump(report_data, tmp, indent=2, default=str)
        tmp_name = tmp.name
    os.replace(tmp_name, str(archive_json))

    # Generate week-over-week diff if prior report exists
    prior_report = _find_prior_report(archive_dir, as_of_date)
    if prior_report:
        diff = _compute_weekly_diff(prior_report, report_data)
        report_data["weekly_diff"] = diff
        # Append diff section to markdown
        diff_md = _render_diff_markdown(diff)
        if diff_md:
            md_content_with_diff = md_content.rstrip() + "\n\n" + diff_md
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".md",
                dir=str(output_dir),
                delete=False,
            ) as tmp:
                tmp.write(md_content_with_diff)
                tmp_name = tmp.name
            os.replace(tmp_name, str(md_path))
        # Re-write JSON with diff included
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            dir=str(output_dir),
            delete=False,
        ) as tmp:
            json.dump(report_data, tmp, indent=2, default=str)
            tmp_name = tmp.name
        os.replace(tmp_name, str(json_path))
        logger.info("  Weekly diff vs %s", prior_report.get("as_of_date", "unknown"))

    logger.info("  Archived to %s", archive_json)

    # --- Verdict artifact ---
    _write_verdict(report_data, output_dir, as_of_date, research_mode=research_mode)

    return report_data


def _write_verdict(
    report_data: Dict[str, Any],
    output_dir: Path,
    as_of_date: str,
    research_mode: bool = False,
) -> None:
    """Write governed BIOSHORT_VERDICT.json and .md. In research mode, writes only to output_dir."""
    ic = report_data.get("ic_decision", {})
    if not ic:
        return

    bt = report_data.get("backtest", {})
    diff = report_data.get("weekly_diff")

    # --- JSON ---
    verdict_doc: Dict[str, Any] = {
        "schema": "bioshort_verdict.v1",
        "as_of_date": as_of_date,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": "research_backfill" if research_mode else "operational",
        "verdict": ic.get("policy_action", "WATCH"),
        "recommendation": ic.get("primary_hedge", "none"),
        "secondary": ic.get("secondary_hedge", "none"),
        "confidence": ic.get("confidence", "LOW"),
        "confidence_score": ic.get("confidence_score", 0),
        "thresholds": {
            "carry_expensive_bps": CARRY_EXPENSIVE_BPS,
            "carry_cheap_bps": CARRY_CHEAP_BPS,
            "min_historical_coverage": MIN_HISTORICAL_COVERAGE,
            "confidence_high_coverage": CONFIDENCE_HIGH_COVERAGE,
        },
        "evidence": {
            "primary_cost_bps": ic.get("primary_cost_bps", 0),
            "primary_score": ic.get("primary_score", 0),
            "historical_months": bt.get("historical_months", 0),
            "total_months": bt.get("total_months", 0),
            "backtest_pricing": bt.get("backtest_pricing", "bs"),
            "options_source": report_data.get("options_source_used", ""),
            "best_vehicle": report_data.get("best_hedge_vehicle", ""),
            "best_vehicle_r_squared": (
                report_data.get("beta_stats", {}).get(report_data.get("best_hedge_vehicle", "XBI"), {}).get("r_squared")
            ),
        },
        "confidence_drivers": ic.get("confidence_drivers", []),
        "policy_reasons": ic.get("policy_reasons", []),
        "recommendation_changed": ic.get("recommendation_changed"),
        "change_reason": ic.get("change_reason", ""),
        "prior_date": diff.get("prior_date") if diff else None,
    }

    verdict_json_path = output_dir / "BIOSHORT_VERDICT.json"
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        dir=str(output_dir),
        delete=False,
    ) as tmp:
        json.dump(verdict_doc, tmp, indent=2, default=str)
        tmp_name = tmp.name
    os.replace(tmp_name, str(verdict_json_path))

    # --- Markdown ---
    v = verdict_doc["verdict"]
    lines = [
        f"# Bioshort Verdict: **{v}**",
        "",
        f"*{as_of_date}*",
        "",
        f"**Recommendation**: {verdict_doc['recommendation']}",
        f"**Confidence**: {verdict_doc['confidence']} ({verdict_doc['confidence_score']:.0f}/100)",
        f"**Status vs prior**: {verdict_doc['change_reason']}",
        "",
        "## Evidence",
        "",
        f"- Carry: {verdict_doc['evidence']['primary_cost_bps']:.0f} bps annualized",
        f"- Hedge score: {verdict_doc['evidence']['primary_score']:.1f}",
        f"- Backtest: {verdict_doc['evidence']['historical_months']}/{verdict_doc['evidence']['total_months']} "
        f"months historical ({verdict_doc['evidence']['backtest_pricing']})",
        f"- Options source: {verdict_doc['evidence']['options_source']}",
        f"- Best vehicle: {verdict_doc['evidence']['best_vehicle']} "
        f"(R²={verdict_doc['evidence']['best_vehicle_r_squared']})",
        "",
        "## Confidence Drivers",
        "",
    ]
    for driver in verdict_doc["confidence_drivers"]:
        lines.append(f"- {driver}")
    lines.extend(["", "## Policy Reasons", ""])
    for reason in verdict_doc["policy_reasons"]:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "## Thresholds",
            "",
            f"- Carry expensive: >{verdict_doc['thresholds']['carry_expensive_bps']} bps",
            f"- Carry cheap: <{verdict_doc['thresholds']['carry_cheap_bps']} bps",
            f"- Min historical coverage: {verdict_doc['thresholds']['min_historical_coverage']:.0%}",
            f"- High confidence coverage: {verdict_doc['thresholds']['confidence_high_coverage']:.0%}",
            "",
        ]
    )

    if verdict_doc.get("secondary") and verdict_doc["secondary"] != "none":
        lines.append(f"**Secondary alternative**: {verdict_doc['secondary']}")
        lines.append("")

    verdict_md_path = output_dir / "BIOSHORT_VERDICT.md"
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".md",
        dir=str(output_dir),
        delete=False,
    ) as tmp:
        tmp.write("\n".join(lines))
        tmp_name = tmp.name
    os.replace(tmp_name, str(verdict_md_path))

    logger.info("  Verdict: %s → %s", v, verdict_json_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Biotech portfolio hedge report — weekly IC-ready analysis",
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=date.today().isoformat(),
        help="Report date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--portfolio-csv",
        type=Path,
        default=None,
        help="Portfolio positions CSV (ticker, weight or market_value)",
    )
    parser.add_argument(
        "--price-csv",
        type=Path,
        default=REPO_ROOT / "production_data" / "price_history.csv",
        help="Price history CSV",
    )
    parser.add_argument(
        "--hedge-notional",
        type=float,
        default=1_000_000,
        help="Notional to hedge ($)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "hedge_report",
        help="Output directory",
    )
    parser.add_argument(
        "--snap-dir",
        type=Path,
        default=None,
        help="Snapshot directory (default: data/snapshots/{as_of_date})",
    )
    parser.add_argument(
        "--options-source",
        type=str,
        choices=["auto", "massive", "tasty"],
        default="auto",
        help="Options data source: auto (best available), massive, or tasty",
    )
    parser.add_argument(
        "--backtest-mode",
        type=str,
        choices=["auto", "historical", "bs"],
        default="auto",
        help="Backtest pricing: auto (historical if S3 creds, else BS), historical, or bs",
    )
    parser.add_argument(
        "--research-mode",
        action="store_true",
        default=False,
        help="Isolation mode for research backfill (redirects archive writes to output_dir only, no production mutations)",
    )
    args = parser.parse_args()

    report = run_hedge_report(
        as_of_date=args.as_of_date,
        portfolio_csv=args.portfolio_csv,
        price_csv=args.price_csv,
        hedge_notional=args.hedge_notional,
        output_dir=args.output_dir,
        snap_dir=args.snap_dir,
        options_source=args.options_source,
        backtest_mode=args.backtest_mode,
        research_mode=args.research_mode,
    )

    if report.get("error"):
        logger.error("Report failed: %s", report["error"])
        return 1

    ranked = report.get("ranked_structures", [])
    if ranked:
        top = ranked[0]
        logger.info(
            "\n=== RECOMMENDATION: %s %s — $%s (%s bps ann), score=%.1f ===",
            top["vehicle"],
            top["structure"],
            f"{top.get('total_cost', 0):,.0f}",
            f"{top.get('ann_cost_bps', 0):.0f}",
            top.get("hedge_score", 0),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
