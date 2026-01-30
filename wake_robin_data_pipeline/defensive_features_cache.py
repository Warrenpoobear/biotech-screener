#!/usr/bin/env python3
"""Offline Defensive Features Cache Builder (PIT-safe)."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

CACHE_VERSION = "1.0.0"


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
    if len(returns) < window:
        return None
    return sum(returns[-window:])


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


def compute_features(ticker: str, closes: List[float], xbi_returns: List[float]) -> Dict:
    """Compute all defensive features for a ticker."""
    returns = compute_returns(closes)
    features = {}

    v60 = vol(returns, 60)
    v20 = vol(returns, 20)
    if v60 is not None:
        features["vol_60d"] = f"{v60:.6f}"
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

    b = beta(returns, xbi_returns, 60)
    if b is not None:
        features["beta_xbi_60d"] = f"{b:.6f}"

    return features


def build_cache(price_file: str, as_of: str, tickers: List[str] = None) -> Dict:
    """Build defensive features cache from price file."""
    prices = load_prices(price_file, as_of)

    # Get XBI returns for correlation/beta
    xbi_closes = [p[1] for p in prices.get("XBI", [])]
    xbi_returns = compute_returns(xbi_closes)

    features_by_ticker = {}
    warnings = []

    target_tickers = tickers if tickers else [t for t in prices if t != "XBI"]

    for ticker in sorted(target_tickers):
        if ticker not in prices:
            warnings.append(f"{ticker}: no price data")
            continue
        closes = [p[1] for p in prices[ticker]]
        if len(closes) < 21:
            warnings.append(f"{ticker}: insufficient data ({len(closes)} days)")
            continue
        features = compute_features(ticker, closes, xbi_returns)
        if features:
            features_by_ticker[ticker] = features

    return {
        "as_of_date": as_of,
        "cache_version": CACHE_VERSION,
        "total_tickers": len(target_tickers),
        "computed_count": len(features_by_ticker),
        "features_by_ticker": features_by_ticker,
        "warnings": warnings[:20],  # Cap warnings
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
    print(f"  Output: {args.output}")
    print(f"  Integrity: {integrity[:16]}...")
    if data["warnings"]:
        print(f"  Warnings: {len(data['warnings'])}")


if __name__ == "__main__":
    main()
