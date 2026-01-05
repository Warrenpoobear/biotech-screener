"""
Backtest Metrics Module

PIT-safe metrics computation for validating screener performance:
- IC (Spearman correlation)
- Adaptive bucket returns (terciles for small N, quintiles for larger)
- Hit rate
- Cohort-stratified analysis

Forward returns start NEXT trading day after as_of_date.
Horizons use trading day counts internally (63d/126d/252d → 3m/6m/12m display).
"""
from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Callable, Dict, List, Optional, Tuple
from statistics import mean, median
import hashlib
import json

from common.provenance import compute_hash

METRICS_VERSION = "1.0.1"

# Trading day horizons (internal representation)
HORIZON_TRADING_DAYS = {
    "63d": 63,
    "126d": 126,
    "252d": 252,
}

# Display name mapping
HORIZON_DISPLAY_NAMES = {
    "63d": "3m",
    "126d": "6m",
    "252d": "12m",
}

# Reverse mapping for input convenience
HORIZON_FROM_DISPLAY = {
    "3m": "63d",
    "6m": "126d",
    "12m": "252d",
}

# Minimum observations for reliable metrics
MIN_OBS_IC = 10

# Adaptive bucket sizing rules
BUCKET_THRESHOLDS = {
    "tercile": 25,
    "relaxed_quintile": 50,
}
MIN_OBS_QUINTILE_RELAXED = 2
MIN_OBS_QUINTILE_STANDARD = 5
MIN_OBS_TERCILE = 3

MIN_COHORT_N = 10

# Type alias
ReturnProvider = Callable[[str, str, str], Optional[str]]


# -----------------------------
# Helpers
# -----------------------------

def _decimal(x: Any) -> Decimal:
    if isinstance(x, Decimal):
        return x
    if x is None:
        return Decimal("0")
    return Decimal(str(x))


def _quantize_6dp(d: Decimal) -> str:
    return str(d.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))


def _parse_date(d: str) -> date:
    return date.fromisoformat(d)


def _format_date(d: date) -> str:
    return d.isoformat()


def _normalize_horizon(horizon: str) -> str:
    """Normalize horizon to internal representation (63d/126d/252d)."""
    if horizon in HORIZON_TRADING_DAYS:
        return horizon
    if horizon in HORIZON_FROM_DISPLAY:
        return HORIZON_FROM_DISPLAY[horizon]
    raise ValueError(f"Unknown horizon: {horizon}")


# -----------------------------
# Trading Calendar (MVP)
# -----------------------------

def next_trading_day(d: str) -> str:
    dt = _parse_date(d) + timedelta(days=1)
    while dt.weekday() >= 5:
        dt = dt + timedelta(days=1)
    return _format_date(dt)


def add_trading_days(d: str, trading_days: int) -> str:
    dt = _parse_date(d)
    days_added = 0
    while days_added < trading_days:
        dt = dt + timedelta(days=1)
        if dt.weekday() < 5:
            days_added += 1
    return _format_date(dt)


def compute_forward_windows(as_of_date: str, horizons: List[str] = None) -> Dict[str, Dict[str, str]]:
    if horizons is None:
        horizons = list(HORIZON_TRADING_DAYS.keys())
    
    start = next_trading_day(as_of_date)
    windows = {}
    for h in horizons:
        h_norm = _normalize_horizon(h)
        days = HORIZON_TRADING_DAYS[h_norm]
        end = add_trading_days(start, days)
        windows[h_norm] = {
            "start": start,
            "end": end,
            "display": HORIZON_DISPLAY_NAMES.get(h_norm, h_norm),
            "trading_days": days,
        }
    return windows


# -----------------------------
# Spearman IC
# -----------------------------

def _rank_data(values: List[Decimal]) -> List[float]:
    n = len(values)
    if n == 0:
        return []
    indexed = [(v, i) for i, v in enumerate(values)]
    indexed.sort(key=lambda x: (x[0], x[1]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and indexed[j][0] == indexed[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        for k in range(i, j):
            ranks[indexed[k][1]] = avg_rank
        i = j
    return ranks


def compute_spearman_ic(scores: List[Decimal], returns: List[Decimal]) -> Optional[Decimal]:
    n = len(scores)
    if n < MIN_OBS_IC:
        return None
    if len(returns) != n:
        raise ValueError("scores and returns must have same length")
    
    score_ranks = _rank_data(scores)
    return_ranks = _rank_data(returns)
    mean_s = mean(score_ranks)
    mean_r = mean(return_ranks)
    
    numerator = sum((s - mean_s) * (r - mean_r) for s, r in zip(score_ranks, return_ranks))
    denom_s = sum((s - mean_s) ** 2 for s in score_ranks) ** 0.5
    denom_r = sum((r - mean_r) ** 2 for r in return_ranks) ** 0.5
    
    if denom_s == 0 or denom_r == 0:
        return Decimal("0")
    
    ic = numerator / (denom_s * denom_r)
    return Decimal(str(ic))


# -----------------------------
# Adaptive Bucket Assignment
# -----------------------------

def _get_bucket_config(n_obs: int) -> Dict[str, Any]:
    """
    Determine bucket configuration based on observation count.
    - n_obs < 25: terciles (3 buckets)
    - n_obs < 50: quintiles with MIN_OBS=2
    - n_obs >= 50: quintiles with MIN_OBS=5
    """
    if n_obs < BUCKET_THRESHOLDS["tercile"]:
        return {"n_buckets": 3, "bucket_type": "tercile", "min_obs_per_bucket": MIN_OBS_TERCILE}
    elif n_obs < BUCKET_THRESHOLDS["relaxed_quintile"]:
        return {"n_buckets": 5, "bucket_type": "quintile_relaxed", "min_obs_per_bucket": MIN_OBS_QUINTILE_RELAXED}
    else:
        return {"n_buckets": 5, "bucket_type": "quintile_standard", "min_obs_per_bucket": MIN_OBS_QUINTILE_STANDARD}


def assign_buckets(ranked_securities: List[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """Assign buckets with adaptive sizing. Returns (ticker_to_bucket, config)."""
    n = len(ranked_securities)
    if n == 0:
        return {}, {"n_buckets": 0, "bucket_type": "empty", "min_obs_per_bucket": 0}
    
    config = _get_bucket_config(n)
    n_buckets = config["n_buckets"]
    sorted_secs = sorted(ranked_securities, key=lambda x: x["composite_rank"])
    
    buckets = {}
    for i, sec in enumerate(sorted_secs):
        pct = (i + 1) / n
        bucket = min(n_buckets, int(pct * n_buckets) + 1)
        buckets[sec["ticker"]] = bucket
    
    return buckets, config


def assign_quintiles(ranked_securities: List[Dict[str, Any]]) -> Dict[str, int]:
    """Assign quintiles (always 5 buckets for backward compatibility)."""
    n = len(ranked_securities)
    if n == 0:
        return {}
    
    sorted_secs = sorted(ranked_securities, key=lambda x: x["composite_rank"])
    quintiles = {}
    for i, sec in enumerate(sorted_secs):
        pct = (i + 1) / n
        if pct <= 0.2: q = 1
        elif pct <= 0.4: q = 2
        elif pct <= 0.6: q = 3
        elif pct <= 0.8: q = 4
        else: q = 5
        quintiles[sec["ticker"]] = q
    return quintiles


def compute_bucket_returns(ranked_securities: List[Dict[str, Any]], returns: Dict[str, Decimal]) -> Dict[str, Any]:
    """Compute bucket returns with adaptive sizing."""
    buckets, config = assign_buckets(ranked_securities)
    n_buckets = config["n_buckets"]
    min_obs = config["min_obs_per_bucket"]
    
    if n_buckets == 0:
        return {"bucket_type": "empty", "n_buckets": 0, "bucket_returns": {}, "bucket_counts": {}, "top_minus_bottom": None, "monotonic": False, "min_obs_per_bucket": 0}
    
    bucket_returns: Dict[int, List[Decimal]] = {i: [] for i in range(1, n_buckets + 1)}
    for sec in ranked_securities:
        ticker = sec["ticker"]
        if ticker in returns and ticker in buckets:
            bucket_returns[buckets[ticker]].append(returns[ticker])
    
    bucket_means: Dict[int, Optional[Decimal]] = {}
    for b in range(1, n_buckets + 1):
        if len(bucket_returns[b]) >= min_obs:
            bucket_means[b] = Decimal(str(mean([float(r) for r in bucket_returns[b]])))
        else:
            bucket_means[b] = None
    
    spread = None
    if bucket_means.get(n_buckets) is not None and bucket_means.get(1) is not None:
        spread = bucket_means[n_buckets] - bucket_means[1]
    
    monotonic = True
    for i in range(1, n_buckets):
        if bucket_means.get(i) is not None and bucket_means.get(i + 1) is not None:
            if bucket_means[i] > bucket_means[i + 1] + Decimal("0.001"):
                monotonic = False
                break
        else:
            monotonic = False
            break
    
    return {
        "bucket_type": config["bucket_type"],
        "n_buckets": n_buckets,
        "bucket_returns": {b: _quantize_6dp(bucket_means[b]) if bucket_means[b] is not None else None for b in range(1, n_buckets + 1)},
        "bucket_counts": {b: len(bucket_returns[b]) for b in range(1, n_buckets + 1)},
        "top_minus_bottom": _quantize_6dp(spread) if spread is not None else None,
        "monotonic": monotonic,
        "min_obs_per_bucket": min_obs,
    }


def compute_quintile_returns(ranked_securities: List[Dict[str, Any]], returns: Dict[str, Decimal]) -> Dict[str, Any]:
    """Compute quintile returns (backward compatible)."""
    quintiles = assign_quintiles(ranked_securities)
    n_obs = len(ranked_securities)
    config = _get_bucket_config(n_obs)
    min_obs = config["min_obs_per_bucket"]
    
    q_returns: Dict[int, List[Decimal]] = {i: [] for i in range(1, 6)}
    for sec in ranked_securities:
        ticker = sec["ticker"]
        if ticker in returns and ticker in quintiles:
            q_returns[quintiles[ticker]].append(returns[ticker])
    
    q_means: Dict[int, Optional[Decimal]] = {}
    for q in range(1, 6):
        if len(q_returns[q]) >= min_obs:
            q_means[q] = Decimal(str(mean([float(r) for r in q_returns[q]])))
        else:
            q_means[q] = None
    
    spread = None
    if q_means.get(5) is not None and q_means.get(1) is not None:
        spread = q_means[5] - q_means[1]
    
    monotonic = True
    for i in range(1, 5):
        if q_means.get(i) is not None and q_means.get(i + 1) is not None:
            if q_means[i] > q_means[i + 1] + Decimal("0.001"):
                monotonic = False
                break
        else:
            monotonic = False
            break
    
    return {
        "q1_mean_return": _quantize_6dp(q_means[1]) if q_means[1] is not None else None,
        "q2_mean_return": _quantize_6dp(q_means[2]) if q_means[2] is not None else None,
        "q3_mean_return": _quantize_6dp(q_means[3]) if q_means[3] is not None else None,
        "q4_mean_return": _quantize_6dp(q_means[4]) if q_means[4] is not None else None,
        "q5_mean_return": _quantize_6dp(q_means[5]) if q_means[5] is not None else None,
        "q5_minus_q1": _quantize_6dp(spread) if spread is not None else None,
        "monotonic": monotonic,
        "q_counts": {q: len(q_returns[q]) for q in range(1, 6)},
    }


# -----------------------------
# Hit Rate
# -----------------------------

def compute_hit_rate(ranked_securities: List[Dict[str, Any]], returns: Dict[str, Decimal]) -> Dict[str, Any]:
    """Hit rate for top quintile (return > cross-section median)."""
    quintiles = assign_quintiles(ranked_securities)
    all_returns = [returns[sec["ticker"]] for sec in ranked_securities if sec["ticker"] in returns]
    
    if not all_returns:
        return {"hit_rate_q5": None, "q5_hits": 0, "q5_total": 0, "cross_section_median": None}
    
    cs_median = Decimal(str(median([float(r) for r in all_returns])))
    q5_tickers = [t for t, q in quintiles.items() if q == 5]
    q5_hits = sum(1 for t in q5_tickers if t in returns and returns[t] > cs_median)
    q5_total = sum(1 for t in q5_tickers if t in returns)
    
    hit_rate = Decimal(str(q5_hits / q5_total)) if q5_total > 0 else None
    return {
        "hit_rate_q5": _quantize_6dp(hit_rate) if hit_rate is not None else None,
        "q5_hits": q5_hits,
        "q5_total": q5_total,
        "cross_section_median": _quantize_6dp(cs_median),
    }


# -----------------------------
# Cohort IC
# -----------------------------

def compute_cohort_ic(ranked_securities: List[Dict[str, Any]], returns: Dict[str, Decimal], min_cohort_n: int = MIN_COHORT_N) -> Dict[str, Any]:
    cohorts: Dict[str, List[Dict[str, Any]]] = {}
    for sec in ranked_securities:
        key = f"{sec['stage_bucket']}_{sec['market_cap_bucket']}"
        if key not in cohorts:
            cohorts[key] = []
        cohorts[key].append(sec)
    
    result = {}
    for cohort_key, secs in cohorts.items():
        valid = [(sec, returns[sec["ticker"]]) for sec in secs if sec["ticker"] in returns]
        if len(valid) < min_cohort_n:
            result[cohort_key] = {"ic_spearman": None, "n_obs": len(valid)}
            continue
        scores = [_decimal(sec["composite_score"]) for sec, _ in valid]
        rets = [r for _, r in valid]
        ic = compute_spearman_ic(scores, rets)
        result[cohort_key] = {"ic_spearman": _quantize_6dp(ic) if ic is not None else None, "n_obs": len(valid)}
    return result


# -----------------------------
# Period Metrics
# -----------------------------

def compute_period_metrics(module5_snapshot: Dict[str, Any], return_provider: ReturnProvider, horizons: List[str] = None, min_cohort_n: int = MIN_COHORT_N) -> Dict[str, Any]:
    if horizons is None:
        horizons = ["63d", "126d", "252d"]
    
    as_of_date = module5_snapshot["as_of_date"]
    ranked_securities = module5_snapshot["ranked_securities"]
    normalized_horizons = [_normalize_horizon(h) for h in horizons]
    windows = compute_forward_windows(as_of_date, normalized_horizons)
    
    result = {"as_of_date": as_of_date, "n_ranked": len(ranked_securities), "horizons": {}}
    
    for h_norm in normalized_horizons:
        if h_norm not in windows:
            continue
        
        window = windows[h_norm]
        returns: Dict[str, Decimal] = {}
        missing_tickers: List[str] = []
        
        for sec in ranked_securities:
            ticker = sec["ticker"]
            ret = return_provider(ticker, window["start"], window["end"])
            if ret is not None:
                returns[ticker] = _decimal(ret)
            else:
                missing_tickers.append(ticker)
        
        n_total = len(ranked_securities)
        n_with_returns = len(returns)
        coverage_pct = n_with_returns / n_total if n_total > 0 else 0
        
        ic = None
        if len(returns) >= MIN_OBS_IC:
            valid_secs = [sec for sec in ranked_securities if sec["ticker"] in returns]
            scores = [_decimal(sec["composite_score"]) for sec in valid_secs]
            rets = [returns[sec["ticker"]] for sec in valid_secs]
            ic = compute_spearman_ic(scores, rets)
        
        bucket_metrics = compute_bucket_returns(ranked_securities, returns)
        quintile_metrics = compute_quintile_returns(ranked_securities, returns)
        hit_rate_metrics = compute_hit_rate(ranked_securities, returns)
        cohort_ic = compute_cohort_ic(ranked_securities, returns, min_cohort_n)
        
        result["horizons"][h_norm] = {
            "window": window,
            "display_name": window["display"],
            "ic_spearman": _quantize_6dp(ic) if ic is not None else None,
            "n_obs": n_with_returns,
            "coverage_pct": _quantize_6dp(Decimal(str(coverage_pct))),
            "n_missing_returns": len(missing_tickers),
            "missing_return_examples": missing_tickers[:5],
            "bucket_metrics": bucket_metrics,
            **quintile_metrics,
            **hit_rate_metrics,
            "cohort_ic": cohort_ic,
        }
    
    return result


# -----------------------------
# Aggregation
# -----------------------------

def aggregate_metrics(all_period_metrics: Dict[str, Dict[str, Any]], horizons: List[str] = None) -> Dict[str, Any]:
    if horizons is None:
        horizons = ["63d", "126d", "252d"]
    
    result = {}
    for h in horizons:
        h_norm = _normalize_horizon(h)
        ic_values: List[Decimal] = []
        spreads: List[Decimal] = []
        bucket_spreads: List[Decimal] = []
        monotonic_count = 0
        bucket_monotonic_count = 0
        total_dates = 0
        
        for period_data in all_period_metrics.values():
            if "horizons" not in period_data or h_norm not in period_data["horizons"]:
                continue
            h_data = period_data["horizons"][h_norm]
            total_dates += 1
            
            if h_data.get("ic_spearman") is not None:
                ic_values.append(_decimal(h_data["ic_spearman"]))
            if h_data.get("q5_minus_q1") is not None:
                spreads.append(_decimal(h_data["q5_minus_q1"]))
            if h_data.get("monotonic"):
                monotonic_count += 1
            
            bucket_metrics = h_data.get("bucket_metrics", {})
            if bucket_metrics.get("top_minus_bottom") is not None:
                bucket_spreads.append(_decimal(bucket_metrics["top_minus_bottom"]))
            if bucket_metrics.get("monotonic"):
                bucket_monotonic_count += 1
        
        agg = {
            "display_name": HORIZON_DISPLAY_NAMES.get(h_norm, h_norm),
            "n_dates": total_dates,
            "ic_mean": None, "ic_median": None, "ic_pos_frac": None, "ic_tstat_newey_west": None,
            "spread_mean": None, "spread_pos_frac": None, "monotonicity_rate": None,
            "bucket_spread_mean": None, "bucket_monotonicity_rate": None,
        }
        
        if ic_values:
            ic_floats = [float(v) for v in ic_values]
            agg["ic_mean"] = _quantize_6dp(Decimal(str(mean(ic_floats))))
            agg["ic_median"] = _quantize_6dp(Decimal(str(median(ic_floats))))
            agg["ic_pos_frac"] = _quantize_6dp(Decimal(str(sum(1 for v in ic_values if v > 0) / len(ic_values))))
        
        if spreads:
            spread_floats = [float(v) for v in spreads]
            agg["spread_mean"] = _quantize_6dp(Decimal(str(mean(spread_floats))))
            agg["spread_pos_frac"] = _quantize_6dp(Decimal(str(sum(1 for v in spreads if v > 0) / len(spreads))))
        
        if total_dates > 0:
            agg["monotonicity_rate"] = _quantize_6dp(Decimal(str(monotonic_count / total_dates)))
        
        if bucket_spreads:
            agg["bucket_spread_mean"] = _quantize_6dp(Decimal(str(mean([float(v) for v in bucket_spreads]))))
        
        if total_dates > 0:
            agg["bucket_monotonicity_rate"] = _quantize_6dp(Decimal(str(bucket_monotonic_count / total_dates)))
        
        result[h_norm] = agg
    
    return result


# -----------------------------
# Full Suite
# -----------------------------

def run_metrics_suite(module5_snapshots: List[Dict[str, Any]], return_provider: ReturnProvider, run_id: str, horizons: List[str] = None, min_cohort_n: int = MIN_COHORT_N) -> Dict[str, Any]:
    if horizons is None:
        horizons = ["63d", "126d", "252d"]
    
    normalized_horizons = [_normalize_horizon(h) for h in horizons]
    all_period_metrics: Dict[str, Dict[str, Any]] = {}
    
    for snapshot in module5_snapshots:
        as_of_date = snapshot["as_of_date"]
        period_metrics = compute_period_metrics(snapshot, return_provider, normalized_horizons, min_cohort_n)
        all_period_metrics[as_of_date] = period_metrics
    
    aggregate = aggregate_metrics(all_period_metrics, normalized_horizons)
    
    config = {"horizons": normalized_horizons, "min_cohort_n": min_cohort_n, "horizon_trading_days": HORIZON_TRADING_DAYS}
    config_hash = compute_hash(config)
    
    ruleset_versions = list(set(s.get("provenance", {}).get("ruleset_version", "unknown") for s in module5_snapshots))
    
    return {
        "run_id": run_id,
        "horizons": normalized_horizons,
        "horizons_display": [HORIZON_DISPLAY_NAMES.get(h, h) for h in normalized_horizons],
        "period_metrics": all_period_metrics,
        "aggregate_metrics": aggregate,
        "provenance": {"metrics_version": METRICS_VERSION, "module_ruleset_versions": sorted(ruleset_versions), "config_hash": config_hash},
    }


def generate_attribution_frame(module5_snapshot: Dict[str, Any], return_provider: ReturnProvider, horizon: str) -> List[Dict[str, Any]]:
    """Generate attribution frame for debugging IC."""
    as_of_date = module5_snapshot["as_of_date"]
    ranked_securities = module5_snapshot["ranked_securities"]
    h_norm = _normalize_horizon(horizon)
    windows = compute_forward_windows(as_of_date, [h_norm])
    if h_norm not in windows:
        return []
    
    window = windows[h_norm]
    quintiles = assign_quintiles(ranked_securities)
    buckets, bucket_config = assign_buckets(ranked_securities)
    
    frame = []
    for sec in ranked_securities:
        ticker = sec["ticker"]
        ret = return_provider(ticker, window["start"], window["end"])
        frame.append({
            "ticker": ticker,
            "composite_score": sec["composite_score"],
            "composite_rank": sec["composite_rank"],
            "quintile": quintiles.get(ticker),
            "bucket": buckets.get(ticker),
            "bucket_type": bucket_config["bucket_type"],
            "cohort": f"{sec['stage_bucket']}_{sec['market_cap_bucket']}",
            "forward_return": ret,
            "severity": sec.get("severity"),
            "uncertainty_penalty": sec.get("uncertainty_penalty"),
            "flags": sec.get("flags", []),
        })
    return frame
