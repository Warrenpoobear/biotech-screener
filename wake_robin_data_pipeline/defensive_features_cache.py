#!/usr/bin/env python3
"""Offline Defensive Features Cache Builder (PIT-safe).

v2.0.0: Multi-horizon blended risk metrics for monthly position sizing.
  - Blended volatility (63d/252d)
  - Max drawdown (252d/504d)
  - Shrunk beta/correlation (daily + weekly)
  - Data state tracking (FULL/PARTIAL/NONE)
  - confidence_risk field
v1.1.0: Added partial ticker diagnostics.
"""
import argparse, csv, hashlib, json, math
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

CACHE_VERSION = "2.0.0"


def load_prices(filepath: str, as_of: str) -> Dict[str, List[Tuple[str, float]]]:
    """Load prices from CSV, filtered to <= as_of date. Returns {ticker: [(date, close), ...]}."""
    prices = defaultdict(list)
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date, ticker, close = row["date"], row["ticker"], row.get("close")
            if date > as_of:
                continue  # PIT: skip future data
            if close:
                try:
                    prices[ticker].append((date, float(close)))
                except ValueError:
                    pass
    # Sort by date
    for ticker in prices:
        prices[ticker].sort(key=lambda x: x[0])
    return dict(prices)

def compute_returns(closes: List[float]) -> List[float]:
    """Compute log returns from close prices."""
    if len(closes) < 2:
        return []
    return [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]

def vol(returns: List[float], window: int) -> Optional[float]:
    """Annualized volatility from last `window` returns."""
    if len(returns) < window:
        return None
    recent = returns[-window:]
    mean = sum(recent) / len(recent)
    var = sum((r - mean) ** 2 for r in recent) / (len(recent) - 1)
    return math.sqrt(var) * math.sqrt(252)

def ret_cumulative(returns: List[float], window: int) -> Optional[float]:
    """Cumulative return over last `window` days."""
    return sum(returns[-window:]) if len(returns) >= window else None

def drawdown_current(closes: List[float], window: int = 60) -> Optional[float]:
    """Current drawdown from rolling high over last `window` days."""
    if len(closes) < 2:
        return None
    recent = closes[-window:] if len(closes) >= window else closes
    high = max(recent)
    return (closes[-1] / high) - 1.0 if high > 0 else None

def correlation(r1: List[float], r2: List[float], window: int) -> Optional[float]:
    """Correlation between two return series over last `window` days."""
    n = min(len(r1), len(r2), window)
    if n < window:
        return None
    a, b = r1[-n:], r2[-n:]
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / (n - 1)
    va = sum((x - ma) ** 2 for x in a) / (n - 1)
    vb = sum((x - mb) ** 2 for x in b) / (n - 1)
    if va <= 0 or vb <= 0:
        return None
    return cov / (math.sqrt(va) * math.sqrt(vb))


def beta(r1: List[float], r2: List[float], window: int) -> Optional[float]:
    """Beta of r1 vs r2 (market) over last `window` days."""
    n = min(len(r1), len(r2), window)
    if n < 20:
        return None
    a, b = r1[-n:], r2[-n:]
    mb = sum(b) / n
    cov = sum((a[i] - sum(a) / n) * (b[i] - mb) for i in range(n)) / (n - 1)
    vb = sum((x - mb) ** 2 for x in b) / (n - 1)
    return cov / vb if vb > 0 else None


# =============================================================================
# V2: Multi-horizon risk metric functions
# =============================================================================

def resample_weekly(dated_prices: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """Resample daily prices to weekly (last observation per ISO week).

    Args:
        dated_prices: [(date_str, close), ...] sorted by date ascending.

    Returns:
        [(week_end_date, close), ...] with one entry per ISO week.
    """
    if not dated_prices:
        return []
    weeks: Dict[Tuple[int, int], Tuple[str, float]] = {}
    for date_str, close in dated_prices:
        # Parse ISO year, week from date string YYYY-MM-DD
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue
        iso_year, iso_week, _ = dt.isocalendar()
        key = (iso_year, iso_week)
        # Keep latest date in each week (sorted input means last wins)
        weeks[key] = (date_str, close)
    # Sort by date and return
    result = sorted(weeks.values(), key=lambda x: x[0])
    return result


def vol_with_min_obs(returns: List[float], window: int, min_obs: int) -> Optional[float]:
    """Annualized volatility with minimum observation requirement.

    Like vol() but returns None if fewer than min_obs observations in window.
    """
    recent = returns[-window:]
    if len(recent) < min_obs:
        return None
    mean = sum(recent) / len(recent)
    var = sum((r - mean) ** 2 for r in recent) / (len(recent) - 1)
    return math.sqrt(var) * math.sqrt(252)


def blended_volatility(returns: List[float]) -> Tuple[float, Optional[float], Optional[float], str]:
    """Multi-horizon blended volatility.

    Returns:
        (blended_value, vol_63d, vol_252d, data_state)
        data_state: "FULL" (both windows), "PARTIAL" (one window), "NONE" (prior used)
    """
    vol_63d = vol_with_min_obs(returns, window=63, min_obs=45)
    vol_252d = vol_with_min_obs(returns, window=252, min_obs=180)

    if vol_63d is not None and vol_252d is not None:
        blended = 0.60 * vol_252d + 0.40 * vol_63d
        return (blended, vol_63d, vol_252d, "FULL")
    elif vol_252d is not None:
        return (vol_252d, vol_63d, vol_252d, "PARTIAL")
    elif vol_63d is not None:
        return (vol_63d, vol_63d, vol_252d, "PARTIAL")
    else:
        return (0.85, None, None, "NONE")


def max_drawdown_window(closes: List[float], window: int, min_obs: int) -> Optional[float]:
    """Max peak-to-trough drawdown over last `window` closes.

    Scans the window, tracking running peak, finds worst trough.
    Returns negative float (e.g., -0.45 for 45% drawdown) or None.
    """
    recent = closes[-window:]
    if len(recent) < min_obs:
        return None
    peak = recent[0]
    worst_dd = 0.0
    for price in recent:
        if price > peak:
            peak = price
        if peak > 0:
            dd = (price / peak) - 1.0
            if dd < worst_dd:
                worst_dd = dd
    return worst_dd


def blended_max_drawdown(closes: List[float]) -> Tuple[float, Optional[float], Optional[float], str]:
    """Multi-horizon blended max drawdown.

    Returns:
        (blended_value, dd_252d, dd_2y, data_state)
        data_state: "FULL", "PARTIAL", or "NONE"
    """
    dd_252d = max_drawdown_window(closes, window=252, min_obs=180)
    dd_2y = max_drawdown_window(closes, window=504, min_obs=360)

    if dd_252d is not None and dd_2y is not None:
        blended = 0.80 * dd_252d + 0.20 * dd_2y
        return (blended, dd_252d, dd_2y, "FULL")
    elif dd_252d is not None:
        return (dd_252d, dd_252d, dd_2y, "PARTIAL")
    elif dd_2y is not None:
        return (dd_2y, dd_252d, dd_2y, "PARTIAL")
    else:
        return (-0.70, None, None, "NONE")


def compute_dated_returns(dated_prices: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """Compute log returns preserving date labels.

    Args:
        dated_prices: [(date_str, close), ...] sorted by date.

    Returns:
        [(date_str, return), ...] with one fewer entry than input.
    """
    if len(dated_prices) < 2:
        return []
    result = []
    for i in range(1, len(dated_prices)):
        prev_close = dated_prices[i - 1][1]
        cur_close = dated_prices[i][1]
        if prev_close > 0:
            ret = math.log(cur_close / prev_close)
            result.append((dated_prices[i][0], ret))
    return result


def compute_weekly_returns(dated_prices: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """Resample to weekly then compute log returns between consecutive weeks.

    Returns:
        [(date_str, weekly_return), ...]
    """
    weekly = resample_weekly(dated_prices)
    return compute_dated_returns(weekly)


def aligned_overlap(
    dated_r1: List[Tuple[str, float]],
    dated_r2: List[Tuple[str, float]],
) -> Tuple[List[float], List[float]]:
    """Inner join two dated return series on date string.

    Returns:
        (list_r1, list_r2) with matched dates only.
    """
    # Build lookup from second series
    r2_by_date = {d: r for d, r in dated_r2}
    list_r1 = []
    list_r2 = []
    for d, r in dated_r1:
        if d in r2_by_date:
            list_r1.append(r)
            list_r2.append(r2_by_date[d])
    return list_r1, list_r2


def pearson_and_beta(stock_rets: List[float], bench_rets: List[float]) -> Optional[Tuple[float, float]]:
    """Compute correlation and beta in a single pass (shared mean/var).

    Returns:
        (corr, beta) or None if insufficient variance.
    """
    n = len(stock_rets)
    if n < 2:
        return None
    ms = sum(stock_rets) / n
    mb = sum(bench_rets) / n
    cov = sum((stock_rets[i] - ms) * (bench_rets[i] - mb) for i in range(n)) / (n - 1)
    vs = sum((x - ms) ** 2 for x in stock_rets) / (n - 1)
    vb = sum((x - mb) ** 2 for x in bench_rets) / (n - 1)
    if vs <= 0 or vb <= 0:
        return None
    corr = cov / (math.sqrt(vs) * math.sqrt(vb))
    beta_val = cov / vb
    return (corr, beta_val)


def shrunk_beta_corr(
    daily_stock_dated: List[Tuple[str, float]],
    daily_bench_dated: List[Tuple[str, float]],
    weekly_stock_dated: List[Tuple[str, float]],
    weekly_bench_dated: List[Tuple[str, float]],
) -> Tuple[float, float, str]:
    """Compute shrinkage-blended beta and correlation from daily + weekly returns.

    Shrinkage: 0.70 * daily_252d + 0.30 * weekly_~104w.
    Clamps beta to [0, 3], corr to [-1, 1].
    Fallback: beta=1.0, corr=0.60, state=NONE.

    Returns:
        (beta, corr, data_state)
    """
    # Daily 252d
    daily_stock = daily_stock_dated[-252:]
    daily_bench = daily_bench_dated[-252:]
    d_s, d_b = aligned_overlap(daily_stock, daily_bench)
    daily_result = pearson_and_beta(d_s, d_b) if len(d_s) >= 180 else None

    # Weekly ~104w
    weekly_stock = weekly_stock_dated[-104:]
    weekly_bench = weekly_bench_dated[-104:]
    w_s, w_b = aligned_overlap(weekly_stock, weekly_bench)
    weekly_result = pearson_and_beta(w_s, w_b) if len(w_s) >= 78 else None

    if daily_result is not None and weekly_result is not None:
        d_corr, d_beta = daily_result
        w_corr, w_beta = weekly_result
        corr = 0.70 * d_corr + 0.30 * w_corr
        beta_val = 0.70 * d_beta + 0.30 * w_beta
        state = "FULL"
    elif daily_result is not None:
        corr, beta_val = daily_result
        state = "PARTIAL"
    elif weekly_result is not None:
        corr, beta_val = weekly_result
        state = "PARTIAL"
    else:
        return (1.0, 0.60, "NONE")

    # Clamp
    beta_val = max(0.0, min(3.0, beta_val))
    corr = max(-1.0, min(1.0, corr))
    return (beta_val, corr, state)


def compute_confidence_risk(vol_state: str, dd_state: str, beta_state: str) -> float:
    """Compute confidence in risk estimates based on data states.

    Returns:
        0.90 if all FULL, 0.65 if >=2 are not NONE, 0.35 otherwise.
    """
    states = [vol_state, dd_state, beta_state]
    if all(s == "FULL" for s in states):
        return 0.90
    non_none_count = sum(1 for s in states if s != "NONE")
    if non_none_count >= 2:
        return 0.65
    return 0.35


# =============================================================================
# Updated compute_features (v2)
# =============================================================================

def compute_features(
    ticker: str,
    closes: List[float],
    xbi_returns: List[float],
    dated_closes: Optional[List[Tuple[str, float]]] = None,
    xbi_dated_closes: Optional[List[Tuple[str, float]]] = None,
) -> Tuple[Dict, List[str]]:
    """Compute all defensive features for a ticker.

    V2: When dated_closes and xbi_dated_closes are provided, computes
    multi-horizon blended metrics alongside legacy fields.

    Returns (features, skipped_fields).
    """
    returns = compute_returns(closes)
    features = {}
    skipped = []

    # --- Legacy fields (backward compat with defensive_multiplier) ---
    v60, v20 = vol(returns, 60), vol(returns, 20)
    if v60 is not None:
        features["vol_60d"] = f"{v60:.6f}"
    else:
        skipped.append("vol_60d")
    if v20 is not None:
        features["vol_20d"] = f"{v20:.6f}"
    if v60 and v20:
        features["vol_ratio"] = f"{v20 / v60:.6f}"

    r21 = ret_cumulative(returns, 21)
    if r21 is not None:
        features["ret_21d"] = f"{r21:.6f}"

    dd = drawdown_current(closes)
    if dd is not None:
        features["drawdown_current"] = f"{dd:.6f}"

    corr = correlation(returns, xbi_returns, 120)
    if corr is not None:
        features["corr_xbi_120d"] = f"{corr:.6f}"
    else:
        skipped.append("corr_xbi_120d")

    b = beta(returns, xbi_returns, 60)
    if b is not None:
        features["beta_xbi_60d"] = f"{b:.6f}"

    # --- V2 multi-horizon fields ---
    if dated_closes is not None and xbi_dated_closes is not None:
        # Blended volatility
        vol_blended, vol_63d, vol_252d, vol_state = blended_volatility(returns)
        features["vol_blended"] = f"{vol_blended:.6f}"
        if vol_63d is not None:
            features["vol_63d"] = f"{vol_63d:.6f}"
        if vol_252d is not None:
            features["vol_252d"] = f"{vol_252d:.6f}"
        features["risk_data_state_vol"] = vol_state

        # Backward-compat alias: vol_60d → vol_63d value when available
        if vol_63d is not None and "vol_60d" not in features:
            features["vol_60d"] = f"{vol_63d:.6f}"

        # Blended max drawdown
        dd_blended, dd_252d, dd_2y, dd_state = blended_max_drawdown(closes)
        features["max_drawdown_blended"] = f"{dd_blended:.6f}"
        if dd_252d is not None:
            features["max_drawdown_252d"] = f"{dd_252d:.6f}"
        if dd_2y is not None:
            features["max_drawdown_2y"] = f"{dd_2y:.6f}"
        features["risk_data_state_drawdown"] = dd_state

        # Backward-compat alias: drawdown_current → max_drawdown_252d when available
        if dd_252d is not None and "drawdown_current" not in features:
            features["drawdown_current"] = f"{dd_252d:.6f}"

        # Shrunk beta/correlation
        daily_stock_dated = compute_dated_returns(dated_closes)
        daily_bench_dated = compute_dated_returns(xbi_dated_closes)
        weekly_stock_dated = compute_weekly_returns(dated_closes)
        weekly_bench_dated = compute_weekly_returns(xbi_dated_closes)

        beta_shrunk, corr_shrunk, beta_state = shrunk_beta_corr(
            daily_stock_dated, daily_bench_dated,
            weekly_stock_dated, weekly_bench_dated,
        )
        features["beta_xbi_shrunk"] = f"{beta_shrunk:.6f}"
        features["corr_xbi_shrunk"] = f"{corr_shrunk:.6f}"
        features["risk_data_state_beta"] = beta_state

        # Backward-compat aliases
        if "corr_xbi_120d" not in features:
            features["corr_xbi_120d"] = f"{corr_shrunk:.6f}"
        if "beta_xbi_60d" not in features:
            features["beta_xbi_60d"] = f"{beta_shrunk:.6f}"

        # Confidence
        confidence = compute_confidence_risk(vol_state, dd_state, beta_state)
        features["confidence_risk"] = f"{confidence:.2f}"

    return features, skipped


def build_cache(price_file: str, as_of: str, tickers: List[str] = None) -> Dict:
    """Build defensive features cache from price file."""
    prices = load_prices(price_file, as_of)

    # Get XBI data for correlation/beta
    xbi_dated = prices.get("XBI", [])
    xbi_closes = [p[1] for p in xbi_dated]
    xbi_returns = compute_returns(xbi_closes)

    # Pre-compute XBI weekly returns once (shared across all tickers)
    xbi_weekly_returns = compute_weekly_returns(xbi_dated) if xbi_dated else []

    features_by_ticker = {}
    partial_tickers = {}
    warnings = []

    # Diagnostics counters for v2 metrics
    state_counts = {
        "vol": {"FULL": 0, "PARTIAL": 0, "NONE": 0},
        "drawdown": {"FULL": 0, "PARTIAL": 0, "NONE": 0},
        "beta": {"FULL": 0, "PARTIAL": 0, "NONE": 0},
    }
    priors_used = {"vol": 0, "drawdown": 0, "beta": 0}
    beta_clamped = 0
    corr_clamped = 0
    low_confidence_count = 0

    target_tickers = tickers if tickers else [t for t in prices if t != "XBI"]

    for ticker in sorted(target_tickers):
        if ticker not in prices:
            warnings.append(f"{ticker}: no price data")
            continue
        dated_closes = prices[ticker]
        closes = [p[1] for p in dated_closes]
        if len(closes) < 21:
            warnings.append(f"{ticker}: insufficient data ({len(closes)} days)")
            continue

        features, skipped = compute_features(
            ticker, closes, xbi_returns,
            dated_closes=dated_closes,
            xbi_dated_closes=xbi_dated,
        )
        if features:
            # Add deterministic skip reason for partial tickers (used downstream for def_notes)
            if "corr_xbi_120d" in skipped and len(closes) < 120:
                features["cache_skip_reason"] = "insufficient_rows_120"
            features_by_ticker[ticker] = features

            if skipped:
                partial_tickers[ticker] = {"rows": len(closes), "skipped": skipped}

            # Track v2 diagnostics
            for metric in ("vol", "drawdown", "beta"):
                state_key = f"risk_data_state_{metric}"
                state = features.get(state_key)
                if state in state_counts[metric]:
                    state_counts[metric][state] += 1
                    if state == "NONE":
                        priors_used[metric] += 1

            # Track clamped beta/corr
            beta_raw = features.get("beta_xbi_shrunk")
            if beta_raw is not None:
                bv = float(beta_raw)
                if bv <= 0.0 or bv >= 3.0:
                    beta_clamped += 1
            corr_raw = features.get("corr_xbi_shrunk")
            if corr_raw is not None:
                cv = float(corr_raw)
                if cv <= -1.0 or cv >= 1.0:
                    corr_clamped += 1

            conf = features.get("confidence_risk")
            if conf is not None and float(conf) < 0.65:
                low_confidence_count += 1

    # Build sorted vol list for diagnostics
    vol_entries = []
    for t, f in features_by_ticker.items():
        vb = f.get("vol_blended")
        if vb is not None:
            vol_entries.append((t, float(vb)))
    vol_entries.sort(key=lambda x: x[1])

    diagnostics = {
        "lowest_vol_5": vol_entries[:5] if vol_entries else [],
        "highest_vol_5": vol_entries[-5:] if vol_entries else [],
        "beta_clamped_count": beta_clamped,
        "corr_clamped_count": corr_clamped,
        "low_confidence_count": low_confidence_count,
    }

    computed_count = len(features_by_ticker)

    return {
        "as_of_date": as_of,
        "cache_version": CACHE_VERSION,
        "total_tickers": len(target_tickers),
        "computed_count": computed_count,
        "partial_count": len(partial_tickers),
        "features_by_ticker": features_by_ticker,
        "partial_tickers": partial_tickers,
        "warnings": warnings[:20],
        # V2 metadata
        "data_state_summary": state_counts,
        "priors_used": priors_used,
        "diagnostics": diagnostics,
    }

def write_cache(data: Dict, output_path: str) -> str:
    """Write cache file with sha256 integrity. Returns hash."""
    stable = json.dumps(data, sort_keys=True, separators=(",", ":"))
    integrity = hashlib.sha256(stable.encode()).hexdigest()

    output = {
        "cached_at": datetime.now().isoformat(),
        "integrity": integrity,
        "data": data,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)
    return integrity

def main():
    parser = argparse.ArgumentParser(description="Build defensive features cache")
    parser.add_argument("--as-of", required=True, help="As-of date (YYYY-MM-DD)")
    parser.add_argument("--price-file", required=True, help="Price history CSV")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers (default: all)")
    args = parser.parse_args()

    print(f"Building cache for as_of={args.as_of}")
    data = build_cache(args.price_file, args.as_of, args.tickers)
    integrity = write_cache(data, args.output)

    print(f"  Computed: {data['computed_count']}/{data['total_tickers']} tickers")
    if data["partial_count"]:
        print(f"  Partial (missing 120d): {data['partial_count']} tickers")
        for t, d in list(data["partial_tickers"].items())[:5]:
            print(f"    {t}: {d['rows']} rows, skipped={d['skipped']}")

    # V2 diagnostics
    state_summary = data.get("data_state_summary", {})
    if state_summary:
        print(f"  V2 risk metrics:")
        for metric, counts in state_summary.items():
            print(f"    {metric}: FULL={counts.get('FULL', 0)} PARTIAL={counts.get('PARTIAL', 0)} NONE={counts.get('NONE', 0)}")
        priors = data.get("priors_used", {})
        print(f"  Priors used: vol={priors.get('vol', 0)} dd={priors.get('drawdown', 0)} beta={priors.get('beta', 0)}")
        diag = data.get("diagnostics", {})
        print(f"  Low confidence (<0.65): {diag.get('low_confidence_count', 0)}")
        if diag.get("beta_clamped_count", 0):
            print(f"  Beta clamped: {diag['beta_clamped_count']}")

    print(f"  Output: {args.output}")
    print(f"  Integrity: {integrity[:16]}...")
    if data["warnings"]:
        print(f"  Warnings: {len(data['warnings'])}")

if __name__ == "__main__":
    main()
