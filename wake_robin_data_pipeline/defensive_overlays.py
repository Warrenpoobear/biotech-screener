"""
defensive_overlays.py - Deterministic, PIT-safe feature computation

Takes existing time_series data (prices, returns, volumes) and computes:
- Volatility features (vol_60d, vol_20d, vol_ratio)
- Correlation/beta to XBI benchmark (corr_xbi_120d, beta_xbi_60d)
- Drawdown (drawdown_current)
- Technical indicators (rsi_14d, ret_21d, skew_60d)
- Gates and flags (liquidity, drawdown, RSI, volatility expansion)

All calculations are:
- Deterministic (same inputs → same outputs)
- PIT-safe (only use data already filtered to as_of date)
- CCFT-compliant (Decimal arithmetic, proper quantization)
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple
import math

TRADING_DAYS_PER_YEAR = 252
Q = Decimal("0.0001")  # 4 decimal places for features

def _dq(x: float | int | str | Decimal) -> Decimal:
    """Quantize to 4 decimal places."""
    return Decimal(str(x)).quantize(Q, rounding=ROUND_HALF_UP)

def _clamp(x: float, lo: float, hi: float) -> float:
    """Clamp value between lo and hi."""
    return lo if x < lo else hi if x > hi else x

def _to_floats(xs: List[float | int | str | Decimal]) -> List[float]:
    """Convert list to floats for calculations."""
    return [float(x) for x in xs]

def realized_vol(returns: List[float], window: int, annualize: bool = True) -> Optional[Decimal]:
    """
    Calculate realized volatility from returns.
    
    Args:
        returns: Daily log returns
        window: Lookback window (e.g., 60 days)
        annualize: If True, annualize by sqrt(252)
    
    Returns:
        Annualized volatility as Decimal, or None if insufficient data
    """
    if returns is None or len(returns) < window or window < 2:
        return None
    
    r = returns[-window:]
    mean = sum(r) / window
    var = sum((x - mean) ** 2 for x in r) / (window - 1)
    std = math.sqrt(var)
    
    if annualize:
        std *= math.sqrt(TRADING_DAYS_PER_YEAR)
    
    return _dq(std)

def pearson_corr(a: List[float], b: List[float], window: int) -> Optional[Decimal]:
    """
    Calculate Pearson correlation between two return series.
    
    Args:
        a: First return series (e.g., stock returns)
        b: Second return series (e.g., XBI returns)
        window: Lookback window (e.g., 120 days)
    
    Returns:
        Correlation coefficient [-1, 1], or None if insufficient data
    """
    if a is None or b is None or len(a) < window or len(b) < window or window < 2:
        return None
    
    x = a[-window:]
    y = b[-window:]
    n = window
    
    mx = sum(x) / n
    my = sum(y) / n
    
    cov = sum((x[i] - mx) * (y[i] - my) for i in range(n)) / (n - 1)
    sx = math.sqrt(sum((v - mx) ** 2 for v in x) / (n - 1))
    sy = math.sqrt(sum((v - my) ** 2 for v in y) / (n - 1))
    
    if sx == 0 or sy == 0:
        return None
    
    c = cov / (sx * sy)
    return _dq(_clamp(c, -1.0, 1.0))

def beta(stock: List[float], bench: List[float], window: int) -> Optional[Decimal]:
    """
    Calculate beta (stock sensitivity to benchmark).
    
    Args:
        stock: Stock return series
        bench: Benchmark return series (e.g., XBI)
        window: Lookback window (e.g., 60 days)
    
    Returns:
        Beta coefficient, or None if insufficient data
    """
    if stock is None or bench is None or len(stock) < window or len(bench) < window or window < 2:
        return None
    
    s = stock[-window:]
    b = bench[-window:]
    n = window
    
    ms = sum(s) / n
    mb = sum(b) / n
    
    cov = sum((s[i] - ms) * (b[i] - mb) for i in range(n)) / (n - 1)
    vb = sum((x - mb) ** 2 for x in b) / (n - 1)
    
    if vb == 0:
        return None
    
    return _dq(cov / vb)

def drawdown(prices: List[float], window: int = 252) -> Optional[Decimal]:
    """
    Calculate current drawdown from rolling peak.
    
    Args:
        prices: Price series
        window: Lookback window for peak (default 252 = 1 year)
    
    Returns:
        Drawdown as negative decimal (e.g., -0.25 = -25%), or None
    """
    if prices is None or len(prices) < 2:
        return None
    
    look = prices[-window:] if len(prices) >= window else prices
    hi = max(look)
    cur = prices[-1]
    
    if hi == 0:
        return None
    
    return _dq((cur / hi) - 1.0)

def rsi_simple(prices: List[float], period: int = 14) -> Optional[Decimal]:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        prices: Price series
        period: Lookback period (default 14)
    
    Returns:
        RSI value [0, 100], or None if insufficient data
    """
    if prices is None or len(prices) < period + 1:
        return None
    
    chg = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    recent = chg[-period:]
    
    gains = [c if c > 0 else 0.0 for c in recent]
    losses = [-c if c < 0 else 0.0 for c in recent]
    
    ag = sum(gains) / period
    al = sum(losses) / period
    
    if al == 0:
        return Decimal("100").quantize(Q)
    
    rs = ag / al
    return _dq(100.0 - (100.0 / (1.0 + rs)))

def momentum(prices: List[float], period: int = 21) -> Optional[Decimal]:
    """
    Calculate momentum (return over period).
    
    Args:
        prices: Price series
        period: Lookback period (default 21 = 1 month)
    
    Returns:
        Return as decimal (e.g., 0.10 = 10%), or None
    """
    if prices is None or len(prices) < period + 1:
        return None
    
    past = prices[-(period + 1)]
    cur = prices[-1]
    
    if past == 0:
        return None
    
    return _dq((cur / past) - 1.0)

def skewness(returns: List[float], window: int = 60) -> Optional[Decimal]:
    """
    Calculate return skewness.
    
    Args:
        returns: Return series
        window: Lookback window (default 60)
    
    Returns:
        Skewness coefficient, or None if insufficient data
    """
    if returns is None or len(returns) < window or window < 3:
        return None
    
    r = returns[-window:]
    n = window
    m = sum(r) / n
    
    m2 = sum((x - m) ** 2 for x in r) / n
    if m2 == 0:
        return None
    
    m3 = sum((x - m) ** 3 for x in r) / n
    sk = m3 / (m2 ** 1.5)
    
    return _dq(sk)

def compute_defensive_features(
    sec_ts: Dict,
    xbi_ts: Dict,
) -> Dict[str, Optional[Decimal]]:
    """
    Compute all defensive features from time-series data.
    
    Args:
        sec_ts: Security time-series dict with prices/returns/volumes
        xbi_ts: XBI benchmark time-series dict
    
    Returns:
        Dict of feature_name -> Decimal value (or None if cannot compute)
    """
    # Convert to floats for calculations
    prices = _to_floats(sec_ts.get("prices") or [])
    rets = _to_floats(sec_ts.get("returns") or [])
    xbi_rets = _to_floats((xbi_ts.get("returns") or []))

    # Align series to shortest length (still PIT-safe)
    min_len = min(len(rets), len(xbi_rets))
    if min_len >= 2:
        rets = rets[-min_len:]
        xbi_rets = xbi_rets[-min_len:]

    # Compute all features
    vol_60d = realized_vol(rets, 60, annualize=True)
    vol_20d = realized_vol(rets, 20, annualize=True)
    corr_xbi_120d = pearson_corr(rets, xbi_rets, 120)
    beta_xbi_60d = beta(rets, xbi_rets, 60)
    dd_252d = drawdown(prices, 252)
    rsi_14d = rsi_simple(prices, 14)
    ret_21d = momentum(prices, 21)
    skew_60d = skewness(rets, 60)

    # Volatility ratio (vol_20d / vol_60d)
    vol_ratio = None
    if vol_20d is not None and vol_60d is not None and vol_60d != 0:
        vol_ratio = (vol_20d / vol_60d).quantize(Q, rounding=ROUND_HALF_UP)

    return {
        "vol_60d": vol_60d,
        "vol_20d": vol_20d,
        "corr_xbi_120d": corr_xbi_120d,
        "beta_xbi_60d": beta_xbi_60d,
        "drawdown_current": dd_252d,
        "rsi_14d": rsi_14d,
        "ret_21d": ret_21d,
        "skew_60d": skew_60d,
        "vol_ratio": vol_ratio,
    }

def apply_gates_and_flags(
    features: Dict[str, Optional[Decimal]],
    adv_20d_usd: Optional[Decimal],
) -> Tuple[bool, List[str]]:
    """
    Apply hard gates and generate warning flags.
    
    Hard gates (exclude from long portfolio):
    - ADV < $500K (illiquid)
    - Drawdown < -40% (severe decline)
    
    Warning flags (for review):
    - RSI > 80 (overbought)
    - RSI < 20 (oversold)
    - Vol ratio > 1.5 (volatility expansion)
    
    Args:
        features: Dict of computed features
        adv_20d_usd: Average daily dollar volume
    
    Returns:
        (include_in_portfolio, list_of_flags)
    """
    flags: List[str] = []
    include = True

    # Hard gate: Liquidity
    if adv_20d_usd is None or adv_20d_usd < Decimal("500000"):
        include = False
        flags.append("ILLQ_ADV_20D_LT_500K")

    # Hard gate: Drawdown
    dd = features.get("drawdown_current")
    if dd is not None and dd < Decimal("-0.40"):
        include = False
        flags.append("EXCLUDE_DRAWDOWN_LT_40PCT")

    # Warning: RSI overbought/oversold
    rsi = features.get("rsi_14d")
    if rsi is not None and rsi > Decimal("80"):
        flags.append("RSI_OVERBOUGHT")
    if rsi is not None and rsi < Decimal("20"):
        flags.append("RSI_OVERSOLD")

    # Warning: Volatility expansion
    vr = features.get("vol_ratio")
    if vr is not None and vr > Decimal("1.5"):
        flags.append("VOL_EXPANSION")

    return include, flags

def enrich_universe_with_defensive_overlays(
    records: List[Dict],
    xbi_benchmark_record: Dict = None,
) -> None:
    """
    Enrich each security record with defensive features and gates.
    
    Mutates records in-place, adding:
    - defensive_features: Dict of computed features
    - include_long: Boolean (True if passes all hard gates)
    - position_flags: List of warning/exclusion flags
    
    Args:
        records: List of security records from collect_universe_data.py
        xbi_benchmark_record: XBI benchmark record with time_series data
    
    Example:
        records, quality = collect_universe_data.main()
        xbi_record = next((r for r in records if r['ticker'] == '_XBI_BENCHMARK_'), None)
        enrich_universe_with_defensive_overlays(records, xbi_record)
    """
    # Extract XBI time-series
    if xbi_benchmark_record is None:
        print("WARNING: No XBI benchmark provided, correlation/beta features will be None")
        xbi_ts = {"prices": [], "returns": [], "volumes": []}
    else:
        xbi_ts = xbi_benchmark_record.get("time_series") or {}
    
    # Process each security
    for sec in records:
        # Skip the benchmark itself
        if sec.get("ticker") == "_XBI_BENCHMARK_":
            continue
        
        # Get security time-series
        ts = sec.get("time_series") or {}
        
        # Compute defensive features
        feats = compute_defensive_features(ts, xbi_ts)
        
        # Get ADV
        adv_raw = ts.get("adv_20d")
        adv = _dq(adv_raw) if adv_raw is not None else None
        
        # Apply gates and flags
        include, flags = apply_gates_and_flags(feats, adv)
        
        # Add to record (JSON-safe strings)
        sec["defensive_features"] = {
            k: (str(v) if v is not None else None) 
            for k, v in feats.items()
        }
        sec["include_long"] = include
        sec["position_flags"] = flags

def print_defensive_summary(records: List[Dict]) -> None:
    """Print summary of defensive overlay results."""
    total = len([r for r in records if r.get("ticker") != "_XBI_BENCHMARK_"])
    
    included = sum(1 for r in records if r.get("include_long") == True)
    excluded = total - included
    
    flag_counts = {}
    for rec in records:
        for flag in rec.get("position_flags", []):
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    
    print("\n" + "="*60)
    print("DEFENSIVE OVERLAY SUMMARY")
    print("="*60)
    print(f"\nTotal securities: {total}")
    print(f"  ✓ Included in portfolio: {included} ({included/total*100:.1f}%)")
    print(f"  ✗ Excluded by gates: {excluded} ({excluded/total*100:.1f}%)")
    
    if flag_counts:
        print("\nPosition flags:")
        for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
            print(f"  • {flag}: {count} securities")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    """Test with sample data."""
    print("Testing defensive_overlays.py...")
    
    # Sample data
    sample_sec = {
        "ticker": "TEST",
        "time_series": {
            "prices": [100 + i*0.5 for i in range(252)],  # Uptrend
            "returns": [0.005] * 251,  # Constant returns
            "adv_20d": 1000000,  # $1M ADV
        }
    }
    
    sample_xbi = {
        "ticker": "XBI",
        "time_series": {
            "prices": [100 + i*0.3 for i in range(252)],
            "returns": [0.003] * 251,
        }
    }
    
    # Compute features
    feats = compute_defensive_features(
        sample_sec["time_series"],
        sample_xbi["time_series"]
    )
    
    print("\nComputed features:")
    for k, v in feats.items():
        print(f"  {k}: {v}")
    
    # Apply gates
    adv = Decimal(str(sample_sec["time_series"]["adv_20d"]))
    include, flags = apply_gates_and_flags(feats, adv)
    
    print(f"\nInclude in portfolio: {include}")
    print(f"Flags: {flags}")
    
    print("\n✓ Test complete!")
