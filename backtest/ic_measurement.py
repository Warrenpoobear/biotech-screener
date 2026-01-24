"""
IC Measurement and Tracking System

Comprehensive Information Coefficient (IC) measurement, tracking, and analysis
for evaluating screener signal quality over time.

Features:
1. Forward Return Calculation Engine
   - Multi-horizon returns (5d, 10d, 20d, 30d, 60d, 90d)
   - Benchmark-adjusted returns (XBI, equal-weight universe)

2. IC Calculation Methodology
   - Spearman rank correlation
   - Proper handling of ties
   - Sign convention: negative IC is GOOD (lower rank = higher return)

3. IC Stability Analysis
   - Rolling 12-week IC calculation
   - Regime-conditional IC (BULL/BEAR separately)
   - Sector-conditional IC (rare disease, oncology, CNS)
   - Market cap quintile analysis

4. Statistical Significance Testing
   - T-statistic (target: >2.0 for 95% confidence)
   - Bootstrap confidence intervals (1000 iterations)
   - Out-of-sample validation framework

IC Benchmarks for Biotech:
- IC > 0.05: Excellent signal, institutional-grade
- IC 0.03-0.05: Good signal, tradeable
- IC 0.01-0.03: Weak signal, needs enhancement
- IC < 0.01: No predictive power, abandon

Design Philosophy:
- Deterministic: No datetime.now(), reproducible random with explicit seeds
- Stdlib-only: No scipy/numpy in core calculations
- Decimal arithmetic for precision
- Full audit trail for all computations

Author: Wake Robin Capital Management
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from statistics import mean, median, stdev
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

__version__ = "1.0.0"
SCHEMA_VERSION = "v1.0"

# Type aliases
ReturnProvider = Callable[[str, str, str], Optional[str]]
DateLike = Union[str, date]


# =============================================================================
# CONSTANTS
# =============================================================================

# Precision for quantization
DECIMAL_6DP = Decimal("0.000001")
DECIMAL_4DP = Decimal("0.0001")
DECIMAL_2DP = Decimal("0.01")

# Forward return horizons (trading days)
HORIZON_TRADING_DAYS = {
    "5d": 5,
    "10d": 10,
    "20d": 20,
    "30d": 30,
    "60d": 60,
    "90d": 90,
}

# Display names for horizons
HORIZON_DISPLAY_NAMES = {
    "5d": "1w",
    "10d": "2w",
    "20d": "1m",
    "30d": "1.5m",
    "60d": "3m",
    "90d": "4.5m",
}

# Minimum observations for reliable metrics
MIN_OBS_IC = 10
MIN_OBS_TSTAT = 20
MIN_OBS_BOOTSTRAP = 30
MIN_ROLLING_WINDOW = 12  # weeks

# IC quality benchmarks
IC_EXCELLENT = Decimal("0.05")
IC_GOOD = Decimal("0.03")
IC_WEAK = Decimal("0.01")
IC_NOISE = Decimal("0.00")

# T-stat threshold for significance
TSTAT_THRESHOLD_95 = Decimal("2.0")
TSTAT_THRESHOLD_99 = Decimal("2.58")

# Bootstrap parameters
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_CI_LEVELS = [0.90, 0.95, 0.99]


# =============================================================================
# ENUMS
# =============================================================================

class ICQuality(str, Enum):
    """IC quality classification."""
    EXCELLENT = "EXCELLENT"  # IC > 0.05
    GOOD = "GOOD"            # IC 0.03-0.05
    WEAK = "WEAK"            # IC 0.01-0.03
    NOISE = "NOISE"          # IC < 0.01
    NEGATIVE = "NEGATIVE"    # IC < 0 (inverted signal)


class MarketCapBucket(str, Enum):
    """Market cap classification for stratified analysis."""
    MICRO = "MICRO"     # < $300M
    SMALL = "SMALL"     # $300M - $1B
    MID = "MID"         # $1B - $5B
    LARGE = "LARGE"     # > $5B


class SectorCategory(str, Enum):
    """Biotech sector categories."""
    ONCOLOGY = "ONCOLOGY"
    RARE_DISEASE = "RARE_DISEASE"
    CNS = "CNS"
    IMMUNOLOGY = "IMMUNOLOGY"
    INFECTIOUS = "INFECTIOUS"
    CARDIOVASCULAR = "CARDIOVASCULAR"
    OTHER = "OTHER"


class RegimeType(str, Enum):
    """Market regime classification."""
    BULL = "BULL"
    BEAR = "BEAR"
    NEUTRAL = "NEUTRAL"


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class ForwardReturn:
    """Forward return calculation result."""
    ticker: str
    horizon: str
    start_date: str
    end_date: str
    raw_return: Optional[Decimal]
    benchmark_return: Optional[Decimal]
    excess_return: Optional[Decimal]
    data_status: str  # "complete", "missing_ticker", "missing_benchmark"


@dataclass
class ICResult:
    """IC calculation result."""
    ic_value: Decimal
    ic_quality: ICQuality
    n_observations: int
    t_statistic: Optional[Decimal]
    p_value: Optional[Decimal]
    is_significant_95: bool
    is_significant_99: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ic_value": str(self.ic_value),
            "ic_quality": self.ic_quality.value,
            "n_observations": self.n_observations,
            "t_statistic": str(self.t_statistic) if self.t_statistic else None,
            "p_value": str(self.p_value) if self.p_value else None,
            "is_significant_95": self.is_significant_95,
            "is_significant_99": self.is_significant_99,
        }


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval result."""
    point_estimate: Decimal
    ci_90_lower: Decimal
    ci_90_upper: Decimal
    ci_95_lower: Decimal
    ci_95_upper: Decimal
    ci_99_lower: Decimal
    ci_99_upper: Decimal
    bootstrap_std: Decimal
    n_iterations: int
    seed: int


@dataclass
class RollingICResult:
    """Rolling IC result for a specific window."""
    window_end_date: str
    ic_value: Decimal
    n_observations: int
    regime: Optional[str]
    ic_quality: ICQuality


@dataclass
class StratifiedIC:
    """IC stratified by category."""
    category: str
    ic_value: Decimal
    n_observations: int
    ic_quality: ICQuality


@dataclass
class ICReport:
    """Comprehensive IC report for a screening period."""
    as_of_date: str
    horizon: str
    overall_ic: ICResult
    rolling_ic_12w: List[RollingICResult]
    regime_conditional_ic: Dict[str, ICResult]
    sector_ic: Dict[str, ICResult]
    market_cap_ic: Dict[str, ICResult]
    bootstrap_ci: Optional[BootstrapCI]
    trend_analysis: Dict[str, Any]
    provenance: Dict[str, Any]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _decimal(x: Any) -> Decimal:
    """Convert to Decimal safely."""
    if isinstance(x, Decimal):
        return x
    if x is None:
        return Decimal("0")
    return Decimal(str(x))


def _quantize(d: Decimal, precision: Decimal = DECIMAL_6DP) -> Decimal:
    """Quantize decimal to specified precision."""
    return d.quantize(precision, rounding=ROUND_HALF_UP)


def _parse_date(d: DateLike) -> date:
    """Parse date from string or return date object."""
    if isinstance(d, date):
        return d
    return date.fromisoformat(d)


def _format_date(d: date) -> str:
    """Format date to ISO string."""
    return d.isoformat()


def _classify_ic(ic: Decimal) -> ICQuality:
    """Classify IC value into quality bucket."""
    if ic < IC_NOISE:
        return ICQuality.NEGATIVE
    if ic < IC_WEAK:
        return ICQuality.NOISE
    if ic < IC_GOOD:
        return ICQuality.WEAK
    if ic < IC_EXCELLENT:
        return ICQuality.GOOD
    return ICQuality.EXCELLENT


def _classify_market_cap(market_cap_mm: Decimal) -> MarketCapBucket:
    """Classify market cap into buckets."""
    if market_cap_mm < Decimal("300"):
        return MarketCapBucket.MICRO
    if market_cap_mm < Decimal("1000"):
        return MarketCapBucket.SMALL
    if market_cap_mm < Decimal("5000"):
        return MarketCapBucket.MID
    return MarketCapBucket.LARGE


# =============================================================================
# TRADING CALENDAR
# =============================================================================

def next_trading_day(d: DateLike) -> str:
    """Get next trading day (skip weekends)."""
    dt = _parse_date(d) + timedelta(days=1)
    while dt.weekday() >= 5:
        dt = dt + timedelta(days=1)
    return _format_date(dt)


def add_trading_days(d: DateLike, trading_days: int) -> str:
    """Add N trading days to a date."""
    dt = _parse_date(d)
    days_added = 0
    while days_added < trading_days:
        dt = dt + timedelta(days=1)
        if dt.weekday() < 5:
            days_added += 1
    return _format_date(dt)


def subtract_trading_days(d: DateLike, trading_days: int) -> str:
    """Subtract N trading days from a date."""
    dt = _parse_date(d)
    days_subtracted = 0
    while days_subtracted < trading_days:
        dt = dt - timedelta(days=1)
        if dt.weekday() < 5:
            days_subtracted += 1
    return _format_date(dt)


def compute_forward_windows(
    as_of_date: DateLike,
    horizons: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compute forward return measurement windows for each horizon.

    Returns start (next trading day) and end dates for each horizon.
    """
    if horizons is None:
        horizons = list(HORIZON_TRADING_DAYS.keys())

    start = next_trading_day(as_of_date)
    windows = {}

    for h in horizons:
        if h not in HORIZON_TRADING_DAYS:
            raise ValueError(f"Unknown horizon: {h}")
        days = HORIZON_TRADING_DAYS[h]
        end = add_trading_days(start, days)
        windows[h] = {
            "start": start,
            "end": end,
            "display": HORIZON_DISPLAY_NAMES.get(h, h),
            "trading_days": days,
        }
    return windows


# =============================================================================
# SPEARMAN RANK CORRELATION
# =============================================================================

def _compute_ranks(values: List[Decimal]) -> List[float]:
    """
    Compute ranks for a list of values, handling ties with average rank.

    Lower values get lower ranks (rank 1 = smallest value).
    """
    n = len(values)
    if n == 0:
        return []

    # Create (value, original_index) pairs and sort by value
    indexed = [(v, i) for i, v in enumerate(values)]
    indexed.sort(key=lambda x: (x[0], x[1]))

    ranks = [0.0] * n
    i = 0
    while i < n:
        # Find all items with same value (ties)
        j = i
        while j < n and indexed[j][0] == indexed[i][0]:
            j += 1

        # Assign average rank to all tied items
        avg_rank = (i + 1 + j) / 2
        for k in range(i, j):
            ranks[indexed[k][1]] = avg_rank
        i = j

    return ranks


def compute_spearman_ic(
    rankings: List[int],
    forward_returns: List[Decimal],
    negate: bool = True
) -> Optional[Decimal]:
    """
    Calculate Spearman rank correlation between rankings and forward returns.

    Args:
        rankings: Security rankings (1 = best, higher = worse)
        forward_returns: Forward returns for each security
        negate: If True, negate result so that lower rank → higher return = positive IC

    Returns:
        Spearman correlation coefficient, or None if insufficient data.

    Note:
        Negative IC is GOOD (lower rank = higher return) when negate=True
        This matches convention: IC > 0 means signal works as intended
    """
    n = len(rankings)
    if n < MIN_OBS_IC or len(forward_returns) != n:
        return None

    # Convert rankings and returns to Decimal lists
    rank_decimals = [Decimal(str(r)) for r in rankings]

    # Get ranks of both series
    rank_ranks = _compute_ranks(rank_decimals)
    return_ranks = _compute_ranks(forward_returns)

    # Compute Pearson correlation on ranks
    mean_r = mean(rank_ranks)
    mean_ret = mean(return_ranks)

    numerator = sum((r - mean_r) * (ret - mean_ret)
                    for r, ret in zip(rank_ranks, return_ranks))

    denom_r = sum((r - mean_r) ** 2 for r in rank_ranks) ** 0.5
    denom_ret = sum((ret - mean_ret) ** 2 for ret in return_ranks) ** 0.5

    if denom_r == 0 or denom_ret == 0:
        return Decimal("0")

    correlation = numerator / (denom_r * denom_ret)

    # Negate: we want low rank (good) → high return → positive IC
    if negate:
        correlation = -correlation

    return _quantize(Decimal(str(correlation)))


def calculate_ic(
    rankings: Dict[str, int],
    forward_returns: Dict[str, float],
    method: str = "spearman"
) -> Optional[Decimal]:
    """
    Calculate IC from ranking and return dictionaries.

    This is the main entry point for IC calculation.

    Args:
        rankings: Dict of ticker -> rank (lower = better)
        forward_returns: Dict of ticker -> forward return
        method: Correlation method (only "spearman" supported)

    Returns:
        IC value where positive = signal working as intended

    Target IC benchmarks for biotech:
    - IC > 0.05: Excellent signal, institutional-grade
    - IC 0.03-0.05: Good signal, tradeable
    - IC 0.01-0.03: Weak signal, needs enhancement
    - IC < 0.01: No predictive power, abandon
    """
    if method != "spearman":
        raise ValueError(f"Unknown method: {method}. Only 'spearman' supported.")

    # Get common tickers
    common_tickers = sorted(set(rankings.keys()) & set(forward_returns.keys()))

    if len(common_tickers) < MIN_OBS_IC:
        return None

    rank_list = [rankings[t] for t in common_tickers]
    return_list = [_decimal(forward_returns[t]) for t in common_tickers]

    return compute_spearman_ic(rank_list, return_list, negate=True)


# =============================================================================
# FORWARD RETURN CALCULATION ENGINE
# =============================================================================

class ForwardReturnEngine:
    """
    Engine for calculating forward returns at multiple horizons.

    Supports:
    - Raw returns
    - Benchmark-adjusted returns (XBI, equal-weight universe)
    - Multi-horizon calculation (5d, 10d, 20d, 30d, 60d, 90d)
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        return_provider: ReturnProvider,
        benchmark_ticker: str = "XBI",
    ):
        """
        Initialize forward return engine.

        Args:
            return_provider: Callable that returns forward returns
            benchmark_ticker: Ticker to use as benchmark (default: XBI)
        """
        self.return_provider = return_provider
        self.benchmark_ticker = benchmark_ticker

    def calculate_forward_returns(
        self,
        tickers: List[str],
        as_of_date: DateLike,
        horizons: Optional[List[str]] = None,
        include_benchmark_adjustment: bool = True,
    ) -> Dict[str, Dict[str, ForwardReturn]]:
        """
        Calculate forward returns for all tickers at all horizons.

        Args:
            tickers: List of tickers to calculate returns for
            as_of_date: Date to calculate from
            horizons: List of horizons (default: all)
            include_benchmark_adjustment: Include excess returns vs benchmark

        Returns:
            Dict of horizon -> ticker -> ForwardReturn
        """
        if horizons is None:
            horizons = list(HORIZON_TRADING_DAYS.keys())

        windows = compute_forward_windows(as_of_date, horizons)
        results: Dict[str, Dict[str, ForwardReturn]] = {}

        for horizon in horizons:
            window = windows[horizon]
            results[horizon] = {}

            # Get benchmark return for this horizon
            benchmark_return = None
            if include_benchmark_adjustment:
                bench_ret = self.return_provider(
                    self.benchmark_ticker,
                    window["start"],
                    window["end"]
                )
                if bench_ret is not None:
                    benchmark_return = _decimal(bench_ret)

            # Calculate return for each ticker
            for ticker in tickers:
                raw_ret = self.return_provider(
                    ticker,
                    window["start"],
                    window["end"]
                )

                if raw_ret is None:
                    results[horizon][ticker] = ForwardReturn(
                        ticker=ticker,
                        horizon=horizon,
                        start_date=window["start"],
                        end_date=window["end"],
                        raw_return=None,
                        benchmark_return=benchmark_return,
                        excess_return=None,
                        data_status="missing_ticker",
                    )
                else:
                    raw_decimal = _decimal(raw_ret)
                    excess = None
                    status = "complete"

                    if benchmark_return is not None:
                        excess = raw_decimal - benchmark_return
                    else:
                        status = "missing_benchmark"

                    results[horizon][ticker] = ForwardReturn(
                        ticker=ticker,
                        horizon=horizon,
                        start_date=window["start"],
                        end_date=window["end"],
                        raw_return=raw_decimal,
                        benchmark_return=benchmark_return,
                        excess_return=excess,
                        data_status=status,
                    )

        return results

    def calculate_equal_weight_benchmark(
        self,
        returns_by_ticker: Dict[str, Decimal],
    ) -> Optional[Decimal]:
        """
        Calculate equal-weight universe return as alternative benchmark.

        Args:
            returns_by_ticker: Dict of ticker -> return

        Returns:
            Equal-weight average return, or None if no data
        """
        valid_returns = [r for r in returns_by_ticker.values() if r is not None]
        if not valid_returns:
            return None
        return _quantize(Decimal(str(mean([float(r) for r in valid_returns]))))


# =============================================================================
# IC CALCULATION ENGINE
# =============================================================================

class ICCalculationEngine:
    """
    Engine for IC calculation with statistical significance testing.
    """

    VERSION = "1.0.0"

    def calculate_ic_with_significance(
        self,
        rankings: Dict[str, int],
        forward_returns: Dict[str, float],
    ) -> ICResult:
        """
        Calculate IC with t-statistic and significance testing.

        Args:
            rankings: Dict of ticker -> rank
            forward_returns: Dict of ticker -> return

        Returns:
            ICResult with IC value, quality classification, and significance
        """
        ic = calculate_ic(rankings, forward_returns)

        if ic is None:
            return ICResult(
                ic_value=Decimal("0"),
                ic_quality=ICQuality.NOISE,
                n_observations=0,
                t_statistic=None,
                p_value=None,
                is_significant_95=False,
                is_significant_99=False,
            )

        common_tickers = set(rankings.keys()) & set(forward_returns.keys())
        n = len(common_tickers)

        # Calculate t-statistic: t = IC * sqrt(n-2) / sqrt(1 - IC^2)
        t_stat = None
        p_value = None
        is_sig_95 = False
        is_sig_99 = False

        if n >= MIN_OBS_TSTAT:
            ic_squared = ic * ic
            # Handle edge case: perfect correlation (IC = +/- 1.0)
            # t-stat approaches infinity, so we cap at large value
            if ic_squared >= Decimal("0.9999"):
                # Perfect or near-perfect correlation
                sign = 1 if ic > 0 else -1
                t_stat = Decimal(str(sign * 100))  # Cap at +/- 100
                p_value = Decimal("0.0001")
                is_sig_95 = True
                is_sig_99 = True
            elif ic_squared < Decimal("1"):
                t_stat = _quantize(
                    ic * Decimal(str(math.sqrt(n - 2))) /
                    Decimal(str(math.sqrt(float(Decimal("1") - ic_squared)))),
                    DECIMAL_4DP
                )

                # Approximate p-value using normal distribution for large n
                # For t-distribution with df >= 30, normal is good approximation
                if n >= 30:
                    abs_t = abs(float(t_stat))
                    # Two-tailed p-value approximation
                    # Using error function approximation
                    p_approx = 2 * (1 - _normal_cdf(abs_t))
                    p_value = _quantize(Decimal(str(p_approx)), DECIMAL_4DP)

                is_sig_95 = abs(t_stat) >= TSTAT_THRESHOLD_95
                is_sig_99 = abs(t_stat) >= TSTAT_THRESHOLD_99

        return ICResult(
            ic_value=ic,
            ic_quality=_classify_ic(ic),
            n_observations=n,
            t_statistic=t_stat,
            p_value=p_value,
            is_significant_95=is_sig_95,
            is_significant_99=is_sig_99,
        )


def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF using error function."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

class BootstrapEngine:
    """
    Bootstrap resampling for IC confidence intervals.

    Uses deterministic random state for reproducibility.
    """

    VERSION = "1.0.0"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def _seeded_random(self, seed: int, iteration: int) -> float:
        """
        Generate deterministic pseudo-random number.

        Uses hash-based PRNG for reproducibility without random module.
        """
        # Create deterministic hash from seed and iteration
        hash_input = f"{seed}:{iteration}".encode()
        hash_bytes = hashlib.sha256(hash_input).digest()
        # Convert first 8 bytes to float in [0, 1)
        int_val = int.from_bytes(hash_bytes[:8], byteorder='big')
        return int_val / (2**64)

    def _bootstrap_sample_indices(
        self,
        n: int,
        iteration: int
    ) -> List[int]:
        """
        Generate bootstrap sample indices deterministically.
        """
        indices = []
        for i in range(n):
            r = self._seeded_random(self.seed + iteration, i)
            indices.append(int(r * n))
        return indices

    def calculate_bootstrap_ci(
        self,
        rankings: Dict[str, int],
        forward_returns: Dict[str, float],
        n_iterations: int = BOOTSTRAP_ITERATIONS,
    ) -> Optional[BootstrapCI]:
        """
        Calculate bootstrap confidence intervals for IC.

        Args:
            rankings: Dict of ticker -> rank
            forward_returns: Dict of ticker -> return
            n_iterations: Number of bootstrap iterations

        Returns:
            BootstrapCI with point estimate and confidence intervals
        """
        common_tickers = sorted(set(rankings.keys()) & set(forward_returns.keys()))
        n = len(common_tickers)

        if n < MIN_OBS_BOOTSTRAP:
            return None

        # Point estimate
        point_ic = calculate_ic(rankings, forward_returns)
        if point_ic is None:
            return None

        # Bootstrap iterations
        bootstrap_ics: List[float] = []

        for iteration in range(n_iterations):
            # Get bootstrap sample indices
            indices = self._bootstrap_sample_indices(n, iteration)

            # Build resampled data
            sampled_rankings = {}
            sampled_returns = {}
            for idx, sample_idx in enumerate(indices):
                ticker = common_tickers[sample_idx]
                key = f"{ticker}_{idx}"  # Unique key for resampled data
                sampled_rankings[key] = rankings[ticker]
                sampled_returns[key] = forward_returns[ticker]

            # Calculate IC for this bootstrap sample
            ic = calculate_ic(sampled_rankings, sampled_returns)
            if ic is not None:
                bootstrap_ics.append(float(ic))

        if len(bootstrap_ics) < n_iterations * 0.9:
            return None  # Too many failed samples

        # Sort for percentile calculation
        bootstrap_ics.sort()

        # Calculate percentiles
        def percentile(data: List[float], p: float) -> float:
            k = (len(data) - 1) * p
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return data[int(k)]
            d0 = data[int(f)] * (c - k)
            d1 = data[int(c)] * (k - f)
            return d0 + d1

        return BootstrapCI(
            point_estimate=point_ic,
            ci_90_lower=_quantize(Decimal(str(percentile(bootstrap_ics, 0.05)))),
            ci_90_upper=_quantize(Decimal(str(percentile(bootstrap_ics, 0.95)))),
            ci_95_lower=_quantize(Decimal(str(percentile(bootstrap_ics, 0.025)))),
            ci_95_upper=_quantize(Decimal(str(percentile(bootstrap_ics, 0.975)))),
            ci_99_lower=_quantize(Decimal(str(percentile(bootstrap_ics, 0.005)))),
            ci_99_upper=_quantize(Decimal(str(percentile(bootstrap_ics, 0.995)))),
            bootstrap_std=_quantize(Decimal(str(stdev(bootstrap_ics)))) if len(bootstrap_ics) > 1 else Decimal("0"),
            n_iterations=len(bootstrap_ics),
            seed=self.seed,
        )


# =============================================================================
# IC STABILITY ANALYSIS
# =============================================================================

class ICStabilityAnalyzer:
    """
    Analyze IC stability over time and across market conditions.
    """

    VERSION = "1.0.0"

    def __init__(self, ic_engine: ICCalculationEngine):
        self.ic_engine = ic_engine

    def calculate_rolling_ic(
        self,
        weekly_results: List[Dict[str, Any]],
        window_weeks: int = MIN_ROLLING_WINDOW,
    ) -> List[RollingICResult]:
        """
        Calculate rolling IC over specified window.

        Args:
            weekly_results: List of weekly ranking/return results
            window_weeks: Rolling window size in weeks

        Returns:
            List of RollingICResult for each window
        """
        if len(weekly_results) < window_weeks:
            return []

        rolling_results = []

        for i in range(window_weeks - 1, len(weekly_results)):
            # Aggregate data for this window
            window_data = weekly_results[i - window_weeks + 1:i + 1]

            # Combine rankings and returns
            all_rankings = {}
            all_returns = {}

            for week_idx, week in enumerate(window_data):
                for ticker, rank in week.get("rankings", {}).items():
                    key = f"{ticker}_w{week_idx}"
                    all_rankings[key] = rank
                    if ticker in week.get("forward_returns", {}):
                        all_returns[key] = week["forward_returns"][ticker]

            # Calculate IC
            ic = calculate_ic(all_rankings, all_returns)

            if ic is not None:
                rolling_results.append(RollingICResult(
                    window_end_date=week.get("as_of_date", ""),
                    ic_value=ic,
                    n_observations=len(all_rankings),
                    regime=week.get("regime"),
                    ic_quality=_classify_ic(ic),
                ))

        return rolling_results

    def calculate_regime_conditional_ic(
        self,
        weekly_results: List[Dict[str, Any]],
    ) -> Dict[str, ICResult]:
        """
        Calculate IC separately for different market regimes.

        Args:
            weekly_results: List of weekly ranking/return results with regime labels

        Returns:
            Dict of regime -> ICResult
        """
        regime_data: Dict[str, Tuple[Dict[str, int], Dict[str, float]]] = {
            "BULL": ({}, {}),
            "BEAR": ({}, {}),
            "NEUTRAL": ({}, {}),
        }

        for week_idx, week in enumerate(weekly_results):
            regime = week.get("regime", "NEUTRAL")
            if regime not in regime_data:
                regime = "NEUTRAL"

            rankings, returns = regime_data[regime]

            for ticker, rank in week.get("rankings", {}).items():
                key = f"{ticker}_w{week_idx}"
                rankings[key] = rank
                if ticker in week.get("forward_returns", {}):
                    returns[key] = week["forward_returns"][ticker]

        results = {}
        for regime, (rankings, returns) in regime_data.items():
            if rankings:
                results[regime] = self.ic_engine.calculate_ic_with_significance(
                    rankings, returns
                )
            else:
                results[regime] = ICResult(
                    ic_value=Decimal("0"),
                    ic_quality=ICQuality.NOISE,
                    n_observations=0,
                    t_statistic=None,
                    p_value=None,
                    is_significant_95=False,
                    is_significant_99=False,
                )

        return results

    def calculate_sector_conditional_ic(
        self,
        weekly_results: List[Dict[str, Any]],
        ticker_sectors: Dict[str, str],
    ) -> Dict[str, ICResult]:
        """
        Calculate IC separately for different sectors.

        Args:
            weekly_results: List of weekly ranking/return results
            ticker_sectors: Dict of ticker -> sector category

        Returns:
            Dict of sector -> ICResult
        """
        sectors = ["ONCOLOGY", "RARE_DISEASE", "CNS", "IMMUNOLOGY", "INFECTIOUS", "OTHER"]
        sector_data: Dict[str, Tuple[Dict[str, int], Dict[str, float]]] = {
            s: ({}, {}) for s in sectors
        }

        for week_idx, week in enumerate(weekly_results):
            for ticker, rank in week.get("rankings", {}).items():
                sector = ticker_sectors.get(ticker, "OTHER")
                if sector not in sector_data:
                    sector = "OTHER"

                key = f"{ticker}_w{week_idx}"
                sector_data[sector][0][key] = rank

                if ticker in week.get("forward_returns", {}):
                    sector_data[sector][1][key] = week["forward_returns"][ticker]

        results = {}
        for sector, (rankings, returns) in sector_data.items():
            if rankings:
                results[sector] = self.ic_engine.calculate_ic_with_significance(
                    rankings, returns
                )

        return results

    def calculate_market_cap_ic(
        self,
        weekly_results: List[Dict[str, Any]],
        ticker_market_caps: Dict[str, Decimal],
    ) -> Dict[str, ICResult]:
        """
        Calculate IC separately for market cap quintiles.

        Args:
            weekly_results: List of weekly ranking/return results
            ticker_market_caps: Dict of ticker -> market_cap_mm

        Returns:
            Dict of market_cap_bucket -> ICResult
        """
        buckets = ["MICRO", "SMALL", "MID", "LARGE"]
        bucket_data: Dict[str, Tuple[Dict[str, int], Dict[str, float]]] = {
            b: ({}, {}) for b in buckets
        }

        for week_idx, week in enumerate(weekly_results):
            for ticker, rank in week.get("rankings", {}).items():
                mcap = ticker_market_caps.get(ticker)
                if mcap is None:
                    continue

                bucket = _classify_market_cap(mcap).value
                key = f"{ticker}_w{week_idx}"
                bucket_data[bucket][0][key] = rank

                if ticker in week.get("forward_returns", {}):
                    bucket_data[bucket][1][key] = week["forward_returns"][ticker]

        results = {}
        for bucket, (rankings, returns) in bucket_data.items():
            if rankings:
                results[bucket] = self.ic_engine.calculate_ic_with_significance(
                    rankings, returns
                )

        return results


# =============================================================================
# IC TREND ANALYSIS
# =============================================================================

def analyze_ic_trend(rolling_ics: List[RollingICResult]) -> Dict[str, Any]:
    """
    Analyze IC trend over time.

    Returns:
        Dict with trend metrics:
        - trend_direction: "IMPROVING", "DECLINING", "STABLE"
        - slope: Linear regression slope
        - r_squared: R-squared of trend fit
        - recent_vs_historical: Comparison of recent 4w vs rest
        - volatility: IC standard deviation
    """
    if len(rolling_ics) < 4:
        return {
            "trend_direction": "INSUFFICIENT_DATA",
            "slope": None,
            "recent_vs_historical": None,
            "volatility": None,
        }

    ic_values = [float(r.ic_value) for r in rolling_ics]
    n = len(ic_values)

    # Linear regression for trend
    x_mean = (n - 1) / 2
    y_mean = mean(ic_values)

    numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(ic_values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))

    slope = numerator / denominator if denominator > 0 else 0

    # R-squared
    y_pred = [y_mean + slope * (i - x_mean) for i in range(n)]
    ss_res = sum((y - yp) ** 2 for y, yp in zip(ic_values, y_pred))
    ss_tot = sum((y - y_mean) ** 2 for y in ic_values)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Classify trend
    if abs(slope) < 0.001:
        trend_direction = "STABLE"
    elif slope > 0:
        trend_direction = "IMPROVING"
    else:
        trend_direction = "DECLINING"

    # Recent vs historical
    recent_4w = ic_values[-4:]
    historical = ic_values[:-4] if len(ic_values) > 4 else ic_values
    recent_mean = mean(recent_4w)
    historical_mean = mean(historical)

    # IC volatility
    ic_volatility = stdev(ic_values) if len(ic_values) > 1 else 0

    return {
        "trend_direction": trend_direction,
        "slope": _quantize(Decimal(str(slope)), DECIMAL_4DP),
        "r_squared": _quantize(Decimal(str(r_squared)), DECIMAL_4DP),
        "recent_mean_4w": _quantize(Decimal(str(recent_mean))),
        "historical_mean": _quantize(Decimal(str(historical_mean))),
        "recent_vs_historical_delta": _quantize(Decimal(str(recent_mean - historical_mean))),
        "volatility": _quantize(Decimal(str(ic_volatility)), DECIMAL_4DP),
        "n_observations": n,
    }


# =============================================================================
# OUT-OF-SAMPLE VALIDATION
# =============================================================================

class OutOfSampleValidator:
    """
    Out-of-sample validation framework.

    Supports train/test split validation for IC consistency.
    """

    VERSION = "1.0.0"

    def __init__(self, ic_engine: ICCalculationEngine):
        self.ic_engine = ic_engine

    def validate_train_test_split(
        self,
        weekly_results: List[Dict[str, Any]],
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
    ) -> Dict[str, Any]:
        """
        Validate IC consistency between train and test periods.

        Args:
            weekly_results: Full historical weekly results
            train_start: Training period start date
            train_end: Training period end date
            test_start: Test period start date
            test_end: Test period end date

        Returns:
            Dict with train IC, test IC, and consistency metrics
        """
        # Split data into train and test
        train_data = [
            w for w in weekly_results
            if train_start <= w.get("as_of_date", "") <= train_end
        ]
        test_data = [
            w for w in weekly_results
            if test_start <= w.get("as_of_date", "") <= test_end
        ]

        # Aggregate train period
        train_rankings = {}
        train_returns = {}
        for week_idx, week in enumerate(train_data):
            for ticker, rank in week.get("rankings", {}).items():
                key = f"{ticker}_w{week_idx}"
                train_rankings[key] = rank
                if ticker in week.get("forward_returns", {}):
                    train_returns[key] = week["forward_returns"][ticker]

        # Aggregate test period
        test_rankings = {}
        test_returns = {}
        for week_idx, week in enumerate(test_data):
            for ticker, rank in week.get("rankings", {}).items():
                key = f"{ticker}_w{week_idx}"
                test_rankings[key] = rank
                if ticker in week.get("forward_returns", {}):
                    test_returns[key] = week["forward_returns"][ticker]

        # Calculate IC for each period
        train_ic = self.ic_engine.calculate_ic_with_significance(
            train_rankings, train_returns
        )
        test_ic = self.ic_engine.calculate_ic_with_significance(
            test_rankings, test_returns
        )

        # Calculate consistency metrics
        ic_decay = None
        if train_ic.n_observations > 0 and test_ic.n_observations > 0:
            ic_decay = test_ic.ic_value - train_ic.ic_value

        # Assess overall consistency
        consistency = "UNKNOWN"
        if train_ic.n_observations >= MIN_OBS_IC and test_ic.n_observations >= MIN_OBS_IC:
            train_val = float(train_ic.ic_value)
            test_val = float(test_ic.ic_value)

            if train_val > 0 and test_val > 0:
                if test_val >= train_val * 0.7:
                    consistency = "CONSISTENT"
                else:
                    consistency = "DEGRADED"
            elif train_val > 0 and test_val <= 0:
                consistency = "INVERTED"
            elif train_val <= 0 and test_val <= 0:
                consistency = "CONSISTENTLY_WEAK"
            else:
                consistency = "IMPROVED"

        return {
            "train_period": {"start": train_start, "end": train_end},
            "test_period": {"start": test_start, "end": test_end},
            "train_ic": train_ic.to_dict(),
            "test_ic": test_ic.to_dict(),
            "ic_decay": str(ic_decay) if ic_decay is not None else None,
            "consistency": consistency,
            "train_n": train_ic.n_observations,
            "test_n": test_ic.n_observations,
        }


# =============================================================================
# WEEKLY IC REPORT GENERATOR
# =============================================================================

class WeeklyICReportGenerator:
    """
    Automated weekly IC report generation with trend analysis.
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        return_provider: ReturnProvider,
        benchmark_ticker: str = "XBI",
        bootstrap_seed: int = 42,
    ):
        self.forward_return_engine = ForwardReturnEngine(return_provider, benchmark_ticker)
        self.ic_engine = ICCalculationEngine()
        self.bootstrap_engine = BootstrapEngine(bootstrap_seed)
        self.stability_analyzer = ICStabilityAnalyzer(self.ic_engine)

    def generate_report(
        self,
        rankings: Dict[str, int],
        as_of_date: DateLike,
        horizon: str = "20d",
        historical_results: Optional[List[Dict[str, Any]]] = None,
        ticker_sectors: Optional[Dict[str, str]] = None,
        ticker_market_caps: Optional[Dict[str, Decimal]] = None,
        current_regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive weekly IC report.

        Args:
            rankings: Current week rankings (ticker -> rank)
            as_of_date: As-of date for the report
            horizon: Forward return horizon
            historical_results: Previous weeks' results for rolling analysis
            ticker_sectors: Ticker to sector mapping
            ticker_market_caps: Ticker to market cap mapping
            current_regime: Current market regime

        Returns:
            Complete IC report as dict
        """
        tickers = list(rankings.keys())
        as_of_str = _format_date(_parse_date(as_of_date))

        # Calculate forward returns
        returns_by_horizon = self.forward_return_engine.calculate_forward_returns(
            tickers, as_of_date, [horizon]
        )

        # Extract returns for target horizon
        horizon_returns = returns_by_horizon.get(horizon, {})
        forward_returns = {
            t: float(fr.raw_return)
            for t, fr in horizon_returns.items()
            if fr.raw_return is not None
        }

        # Calculate overall IC with significance
        overall_ic = self.ic_engine.calculate_ic_with_significance(rankings, forward_returns)

        # Bootstrap confidence intervals
        bootstrap_ci = None
        if len(forward_returns) >= MIN_OBS_BOOTSTRAP:
            bootstrap_ci = self.bootstrap_engine.calculate_bootstrap_ci(
                rankings, forward_returns
            )

        # Rolling IC analysis (if historical data available)
        rolling_ic = []
        trend_analysis = {}
        if historical_results:
            # Add current week to historical
            current_week = {
                "as_of_date": as_of_str,
                "rankings": rankings,
                "forward_returns": forward_returns,
                "regime": current_regime,
            }
            all_weeks = historical_results + [current_week]

            rolling_ic = self.stability_analyzer.calculate_rolling_ic(all_weeks)
            trend_analysis = analyze_ic_trend(rolling_ic)

        # Regime-conditional IC
        regime_ic = {}
        if historical_results:
            current_week = {
                "as_of_date": as_of_str,
                "rankings": rankings,
                "forward_returns": forward_returns,
                "regime": current_regime,
            }
            regime_ic = self.stability_analyzer.calculate_regime_conditional_ic(
                historical_results + [current_week]
            )

        # Sector-conditional IC
        sector_ic = {}
        if ticker_sectors and historical_results:
            current_week = {
                "as_of_date": as_of_str,
                "rankings": rankings,
                "forward_returns": forward_returns,
            }
            sector_ic = self.stability_analyzer.calculate_sector_conditional_ic(
                historical_results + [current_week], ticker_sectors
            )

        # Market cap IC
        market_cap_ic = {}
        if ticker_market_caps and historical_results:
            current_week = {
                "as_of_date": as_of_str,
                "rankings": rankings,
                "forward_returns": forward_returns,
            }
            market_cap_ic = self.stability_analyzer.calculate_market_cap_ic(
                historical_results + [current_week], ticker_market_caps
            )

        # Coverage statistics
        coverage_stats = {
            "total_ranked": len(rankings),
            "with_returns": len(forward_returns),
            "coverage_pct": len(forward_returns) / len(rankings) if rankings else 0,
            "missing_tickers": [t for t in rankings if t not in forward_returns],
        }

        # Compile report
        report = {
            "report_version": self.VERSION,
            "schema_version": SCHEMA_VERSION,
            "as_of_date": as_of_str,
            "horizon": horizon,
            "horizon_display": HORIZON_DISPLAY_NAMES.get(horizon, horizon),
            "overall_ic": overall_ic.to_dict(),
            "ic_quality_assessment": _get_ic_quality_assessment(overall_ic),
            "bootstrap_ci": {
                "point_estimate": str(bootstrap_ci.point_estimate),
                "ci_90": [str(bootstrap_ci.ci_90_lower), str(bootstrap_ci.ci_90_upper)],
                "ci_95": [str(bootstrap_ci.ci_95_lower), str(bootstrap_ci.ci_95_upper)],
                "ci_99": [str(bootstrap_ci.ci_99_lower), str(bootstrap_ci.ci_99_upper)],
                "bootstrap_std": str(bootstrap_ci.bootstrap_std),
                "n_iterations": bootstrap_ci.n_iterations,
            } if bootstrap_ci else None,
            "rolling_ic_12w": [
                {
                    "window_end_date": r.window_end_date,
                    "ic_value": str(r.ic_value),
                    "n_observations": r.n_observations,
                    "regime": r.regime,
                    "ic_quality": r.ic_quality.value,
                }
                for r in rolling_ic
            ],
            "trend_analysis": {
                k: str(v) if isinstance(v, Decimal) else v
                for k, v in trend_analysis.items()
            } if trend_analysis else {},
            "regime_conditional_ic": {
                k: v.to_dict() for k, v in regime_ic.items()
            },
            "sector_ic": {
                k: v.to_dict() for k, v in sector_ic.items()
            },
            "market_cap_ic": {
                k: v.to_dict() for k, v in market_cap_ic.items()
            },
            "coverage_stats": coverage_stats,
            "current_regime": current_regime,
            "provenance": {
                "module_version": self.VERSION,
                "schema_version": SCHEMA_VERSION,
                "benchmark_ticker": self.forward_return_engine.benchmark_ticker,
                "bootstrap_seed": self.bootstrap_engine.seed,
            },
        }

        return report


def _get_ic_quality_assessment(ic_result: ICResult) -> Dict[str, Any]:
    """Generate human-readable IC quality assessment."""
    quality = ic_result.ic_quality
    ic_val = float(ic_result.ic_value)

    if quality == ICQuality.EXCELLENT:
        assessment = "Excellent signal - institutional-grade, strong predictive power"
        recommendation = "Maintain current strategy, consider increasing allocation"
    elif quality == ICQuality.GOOD:
        assessment = "Good signal - tradeable, moderate predictive power"
        recommendation = "Strategy is working, monitor for consistency"
    elif quality == ICQuality.WEAK:
        assessment = "Weak signal - limited predictive power"
        recommendation = "Consider enhancements: additional factors, regime adaptation"
    elif quality == ICQuality.NOISE:
        assessment = "No predictive power - signal is noise"
        recommendation = "Review methodology, consider alternative signals"
    else:
        assessment = "Negative IC - signal may be inverted"
        recommendation = "URGENT: Investigate signal, consider reversal or abandonment"

    return {
        "quality": quality.value,
        "ic_value": str(ic_result.ic_value),
        "assessment": assessment,
        "recommendation": recommendation,
        "is_significant": ic_result.is_significant_95,
        "actionable": quality in (ICQuality.EXCELLENT, ICQuality.GOOD) and ic_result.is_significant_95,
    }


# =============================================================================
# TIME-SERIES DATABASE (RANKING/RETURN STORAGE)
# =============================================================================

class ICTimeSeriesDatabase:
    """
    Simple time-series storage for rankings and returns.

    Stores data in CSV format for each week:
    - rankings_YYYYMMDD.csv
    - returns_YYYYMMDD.csv
    """

    VERSION = "1.0.0"

    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def _format_filename_date(self, d: DateLike) -> str:
        """Format date for filename."""
        dt = _parse_date(d)
        return dt.strftime("%Y%m%d")

    def store_ranking(
        self,
        as_of_date: DateLike,
        rankings: Dict[str, int],
        scores: Optional[Dict[str, Decimal]] = None,
    ) -> str:
        """
        Store weekly ranking to CSV file.

        Returns filename of stored file.
        """
        import csv
        from pathlib import Path

        date_str = self._format_filename_date(as_of_date)
        filename = f"rankings_{date_str}.csv"
        filepath = Path(self.base_dir) / filename

        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ticker', 'rank', 'score'])

            for ticker in sorted(rankings.keys()):
                score = scores.get(ticker, "") if scores else ""
                writer.writerow([ticker, rankings[ticker], str(score)])

        return str(filepath)

    def store_returns(
        self,
        as_of_date: DateLike,
        returns: Dict[str, Dict[str, Optional[Decimal]]],  # horizon -> ticker -> return
    ) -> str:
        """
        Store forward returns to CSV file.

        Returns filename of stored file.
        """
        import csv
        from pathlib import Path

        date_str = self._format_filename_date(as_of_date)
        filename = f"returns_{date_str}.csv"
        filepath = Path(self.base_dir) / filename

        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Get all horizons and tickers
        horizons = sorted(returns.keys())
        all_tickers = set()
        for horizon_returns in returns.values():
            all_tickers.update(horizon_returns.keys())

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['ticker'] + [f'return_{h}' for h in horizons]
            writer.writerow(header)

            for ticker in sorted(all_tickers):
                row = [ticker]
                for h in horizons:
                    ret = returns.get(h, {}).get(ticker)
                    row.append(str(ret) if ret is not None else "")
                writer.writerow(row)

        return str(filepath)

    def load_ranking(self, as_of_date: DateLike) -> Dict[str, int]:
        """Load ranking from CSV file."""
        import csv
        from pathlib import Path

        date_str = self._format_filename_date(as_of_date)
        filename = f"rankings_{date_str}.csv"
        filepath = Path(self.base_dir) / filename

        if not filepath.exists():
            return {}

        rankings = {}
        with open(filepath, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rankings[row['ticker']] = int(row['rank'])

        return rankings

    def load_returns(
        self,
        as_of_date: DateLike
    ) -> Dict[str, Dict[str, Optional[Decimal]]]:
        """Load returns from CSV file."""
        import csv
        from pathlib import Path

        date_str = self._format_filename_date(as_of_date)
        filename = f"returns_{date_str}.csv"
        filepath = Path(self.base_dir) / filename

        if not filepath.exists():
            return {}

        returns: Dict[str, Dict[str, Optional[Decimal]]] = {}

        with open(filepath, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ticker = row['ticker']
                for col in row:
                    if col.startswith('return_'):
                        horizon = col.replace('return_', '')
                        if horizon not in returns:
                            returns[horizon] = {}
                        val = row[col]
                        returns[horizon][ticker] = Decimal(val) if val else None

        return returns

    def list_available_dates(self) -> List[str]:
        """List all available dates with stored data."""
        from pathlib import Path

        base = Path(self.base_dir)
        if not base.exists():
            return []

        dates = set()
        for f in base.glob("rankings_*.csv"):
            date_str = f.stem.replace("rankings_", "")
            try:
                # Convert YYYYMMDD to YYYY-MM-DD
                year = date_str[:4]
                month = date_str[4:6]
                day = date_str[6:8]
                dates.add(f"{year}-{month}-{day}")
            except (ValueError, IndexError):
                continue

        return sorted(dates)


# =============================================================================
# MAIN IC MEASUREMENT SYSTEM
# =============================================================================

class ICMeasurementSystem:
    """
    Main orchestrator for IC measurement and tracking.

    Combines all components for comprehensive IC analysis.
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        return_provider: ReturnProvider,
        data_dir: str,
        benchmark_ticker: str = "XBI",
        bootstrap_seed: int = 42,
    ):
        self.return_provider = return_provider
        self.data_dir = data_dir
        self.benchmark_ticker = benchmark_ticker

        # Initialize components
        self.forward_return_engine = ForwardReturnEngine(return_provider, benchmark_ticker)
        self.ic_engine = ICCalculationEngine()
        self.bootstrap_engine = BootstrapEngine(bootstrap_seed)
        self.stability_analyzer = ICStabilityAnalyzer(self.ic_engine)
        self.oos_validator = OutOfSampleValidator(self.ic_engine)
        self.report_generator = WeeklyICReportGenerator(
            return_provider, benchmark_ticker, bootstrap_seed
        )
        self.database = ICTimeSeriesDatabase(data_dir)

    def process_weekly_screening(
        self,
        rankings: Dict[str, int],
        scores: Dict[str, Decimal],
        as_of_date: DateLike,
        horizons: Optional[List[str]] = None,
        ticker_sectors: Optional[Dict[str, str]] = None,
        ticker_market_caps: Optional[Dict[str, Decimal]] = None,
        current_regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process weekly screening results and generate IC report.

        This is the main entry point for weekly IC analysis.

        Args:
            rankings: Dict of ticker -> rank
            scores: Dict of ticker -> composite score
            as_of_date: Screening date
            horizons: Forward return horizons to analyze
            ticker_sectors: Optional sector mapping
            ticker_market_caps: Optional market cap mapping
            current_regime: Current market regime

        Returns:
            Complete IC analysis report
        """
        if horizons is None:
            horizons = ["5d", "10d", "20d", "30d", "60d", "90d"]

        tickers = list(rankings.keys())
        as_of_str = _format_date(_parse_date(as_of_date))

        # Store ranking
        self.database.store_ranking(as_of_date, rankings, scores)

        # Calculate forward returns for all horizons
        all_returns = self.forward_return_engine.calculate_forward_returns(
            tickers, as_of_date, horizons
        )

        # Store returns
        returns_for_storage: Dict[str, Dict[str, Optional[Decimal]]] = {}
        for horizon in horizons:
            returns_for_storage[horizon] = {
                t: fr.raw_return for t, fr in all_returns.get(horizon, {}).items()
            }
        self.database.store_returns(as_of_date, returns_for_storage)

        # Load historical data for rolling analysis
        historical_dates = self.database.list_available_dates()
        historical_results = []
        for hist_date in historical_dates:
            if hist_date >= as_of_str:
                continue
            hist_rankings = self.database.load_ranking(hist_date)
            hist_returns = self.database.load_returns(hist_date)
            if hist_rankings and hist_returns:
                historical_results.append({
                    "as_of_date": hist_date,
                    "rankings": hist_rankings,
                    "forward_returns": {
                        t: float(r)
                        for t, r in hist_returns.get("20d", {}).items()
                        if r is not None
                    },
                })

        # Generate IC reports for each horizon
        horizon_reports = {}
        for horizon in horizons:
            # Extract returns for this horizon
            horizon_returns = {
                t: float(fr.raw_return)
                for t, fr in all_returns.get(horizon, {}).items()
                if fr.raw_return is not None
            }

            # Calculate IC with significance
            ic_result = self.ic_engine.calculate_ic_with_significance(
                rankings, horizon_returns
            )

            # Bootstrap CI
            bootstrap_ci = None
            if len(horizon_returns) >= MIN_OBS_BOOTSTRAP:
                bootstrap_ci = self.bootstrap_engine.calculate_bootstrap_ci(
                    rankings, horizon_returns
                )

            horizon_reports[horizon] = {
                "ic": ic_result.to_dict(),
                "quality_assessment": _get_ic_quality_assessment(ic_result),
                "bootstrap_ci": {
                    "point_estimate": str(bootstrap_ci.point_estimate),
                    "ci_95": [str(bootstrap_ci.ci_95_lower), str(bootstrap_ci.ci_95_upper)],
                    "bootstrap_std": str(bootstrap_ci.bootstrap_std),
                } if bootstrap_ci else None,
                "n_with_returns": len(horizon_returns),
            }

        # Rolling IC analysis (using 20d horizon as primary)
        rolling_ic = []
        trend_analysis = {}
        if historical_results:
            current_week = {
                "as_of_date": as_of_str,
                "rankings": rankings,
                "forward_returns": {
                    t: float(r)
                    for t, r in returns_for_storage.get("20d", {}).items()
                    if r is not None
                },
                "regime": current_regime,
            }
            all_weeks = historical_results + [current_week]
            rolling_ic = self.stability_analyzer.calculate_rolling_ic(all_weeks)
            trend_analysis = analyze_ic_trend(rolling_ic)

        # Stratified analysis (using 20d horizon)
        primary_returns = {
            t: float(r)
            for t, r in returns_for_storage.get("20d", {}).items()
            if r is not None
        }

        regime_ic = {}
        sector_ic = {}
        market_cap_ic = {}

        if historical_results:
            current_week = {
                "as_of_date": as_of_str,
                "rankings": rankings,
                "forward_returns": primary_returns,
                "regime": current_regime,
            }
            all_weeks = historical_results + [current_week]

            regime_ic = {
                k: v.to_dict()
                for k, v in self.stability_analyzer.calculate_regime_conditional_ic(all_weeks).items()
            }

            if ticker_sectors:
                sector_ic = {
                    k: v.to_dict()
                    for k, v in self.stability_analyzer.calculate_sector_conditional_ic(
                        all_weeks, ticker_sectors
                    ).items()
                }

            if ticker_market_caps:
                market_cap_ic = {
                    k: v.to_dict()
                    for k, v in self.stability_analyzer.calculate_market_cap_ic(
                        all_weeks, ticker_market_caps
                    ).items()
                }

        # Compile complete report
        return {
            "_metadata": {
                "system_version": self.VERSION,
                "schema_version": SCHEMA_VERSION,
                "as_of_date": as_of_str,
                "benchmark_ticker": self.benchmark_ticker,
                "n_securities": len(rankings),
            },
            "horizon_analysis": horizon_reports,
            "primary_horizon": {
                "horizon": "20d",
                "horizon_display": "1m",
            },
            "rolling_ic_12w": [
                {
                    "window_end_date": r.window_end_date,
                    "ic_value": str(r.ic_value),
                    "n_observations": r.n_observations,
                    "regime": r.regime,
                    "ic_quality": r.ic_quality.value,
                }
                for r in rolling_ic
            ],
            "trend_analysis": {
                k: str(v) if isinstance(v, Decimal) else v
                for k, v in trend_analysis.items()
            } if trend_analysis else {},
            "stratified_analysis": {
                "by_regime": regime_ic,
                "by_sector": sector_ic,
                "by_market_cap": market_cap_ic,
            },
            "current_regime": current_regime,
            "coverage": {
                "total_ranked": len(rankings),
                "horizons_analyzed": horizons,
            },
            "provenance": {
                "module_version": self.VERSION,
                "schema_version": SCHEMA_VERSION,
                "data_dir": self.data_dir,
                "bootstrap_seed": self.bootstrap_engine.seed,
            },
        }

    def run_out_of_sample_validation(
        self,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
    ) -> Dict[str, Any]:
        """
        Run out-of-sample validation between train and test periods.

        Args:
            train_start: Training period start (e.g., "2020-01-01")
            train_end: Training period end (e.g., "2022-12-31")
            test_start: Test period start (e.g., "2023-01-01")
            test_end: Test period end (e.g., "2025-12-31")

        Returns:
            Validation results with train/test IC comparison
        """
        # Load historical data
        available_dates = self.database.list_available_dates()

        weekly_results = []
        for date_str in available_dates:
            rankings = self.database.load_ranking(date_str)
            returns = self.database.load_returns(date_str)

            if rankings and returns:
                weekly_results.append({
                    "as_of_date": date_str,
                    "rankings": rankings,
                    "forward_returns": {
                        t: float(r)
                        for t, r in returns.get("20d", {}).items()
                        if r is not None
                    },
                })

        return self.oos_validator.validate_train_test_split(
            weekly_results, train_start, train_end, test_start, test_end
        )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("IC MEASUREMENT SYSTEM - DEMONSTRATION")
    print("=" * 70)
    print()

    # Example: Calculate IC from sample data
    sample_rankings = {
        "ACME": 1, "BETA": 2, "GAMMA": 3, "DELTA": 4, "EPSILON": 5,
        "ZETA": 6, "ETA": 7, "THETA": 8, "IOTA": 9, "KAPPA": 10,
        "LAMBDA": 11, "MU": 12, "NU": 13, "XI": 14, "OMICRON": 15,
    }

    # Simulated returns (lower rank = higher return = good signal)
    sample_returns = {
        "ACME": 0.15, "BETA": 0.12, "GAMMA": 0.10, "DELTA": 0.08, "EPSILON": 0.06,
        "ZETA": 0.04, "ETA": 0.02, "THETA": 0.00, "IOTA": -0.02, "KAPPA": -0.04,
        "LAMBDA": -0.06, "MU": -0.08, "NU": -0.10, "XI": -0.12, "OMICRON": -0.14,
    }

    print("Sample Data:")
    print(f"  Rankings: {len(sample_rankings)} securities")
    print(f"  Returns: Simulated perfect signal (rank 1 = highest return)")
    print()

    # Calculate IC
    ic = calculate_ic(sample_rankings, sample_returns)
    print(f"Calculated IC: {ic}")
    print(f"IC Quality: {_classify_ic(ic).value}")
    print()

    # IC with significance
    ic_engine = ICCalculationEngine()
    ic_result = ic_engine.calculate_ic_with_significance(sample_rankings, sample_returns)
    print("IC with Significance Testing:")
    print(f"  IC Value: {ic_result.ic_value}")
    print(f"  T-Statistic: {ic_result.t_statistic}")
    print(f"  P-Value: {ic_result.p_value}")
    print(f"  Significant (95%): {ic_result.is_significant_95}")
    print(f"  Significant (99%): {ic_result.is_significant_99}")
    print()

    # Bootstrap CI
    print("Bootstrap Confidence Intervals:")
    bootstrap_engine = BootstrapEngine(seed=42)
    bootstrap_ci = bootstrap_engine.calculate_bootstrap_ci(sample_rankings, sample_returns)
    if bootstrap_ci:
        print(f"  Point Estimate: {bootstrap_ci.point_estimate}")
        print(f"  95% CI: [{bootstrap_ci.ci_95_lower}, {bootstrap_ci.ci_95_upper}]")
        print(f"  Bootstrap Std: {bootstrap_ci.bootstrap_std}")
    print()

    # IC benchmarks
    print("IC Quality Benchmarks:")
    print("  IC > 0.05: EXCELLENT - institutional-grade")
    print("  IC 0.03-0.05: GOOD - tradeable")
    print("  IC 0.01-0.03: WEAK - needs enhancement")
    print("  IC < 0.01: NOISE - no predictive power")
    print()
