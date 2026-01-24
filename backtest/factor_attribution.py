#!/usr/bin/env python3
"""
factor_attribution.py - Factor Attribution Analysis for Backtesting

P2 Enhancement: Decomposes portfolio returns by signal source to understand
which factors are driving performance.

Key Features:
1. IC by factor type (clinical, financial, catalyst, momentum, etc.)
2. Marginal contribution analysis
3. Factor decay curves (IC vs signal age)
4. Sector-adjusted returns
5. Regime-conditional performance

Design Philosophy:
- DETERMINISTIC: All calculations use Decimal or explicit floating point
- AUDITABLE: Full provenance for all metrics
- PIT-SAFE: Uses forward returns from signal date

Author: Wake Robin Capital Management
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from statistics import mean, median
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

__version__ = "1.0.0"

logger = logging.getLogger(__name__)

# Type alias for return provider
ReturnProvider = Callable[[str, str, str], Optional[str]]


@dataclass
class FactorIC:
    """Information Coefficient for a single factor."""
    factor_name: str
    ic_value: Optional[Decimal]
    t_statistic: Optional[Decimal]
    n_observations: int
    is_significant: bool  # |t| > 2
    direction: str  # "POSITIVE", "NEGATIVE", "NEUTRAL"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor_name": self.factor_name,
            "ic_value": str(self.ic_value) if self.ic_value else None,
            "t_statistic": str(self.t_statistic) if self.t_statistic else None,
            "n_observations": self.n_observations,
            "is_significant": self.is_significant,
            "direction": self.direction,
        }


@dataclass
class FactorContribution:
    """Marginal contribution of a factor to overall IC."""
    factor_name: str
    standalone_ic: Optional[Decimal]
    marginal_ic: Optional[Decimal]  # Improvement over baseline
    contribution_pct: Optional[Decimal]  # % of total IC explained
    is_additive: bool  # Does it improve composite IC?

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor_name": self.factor_name,
            "standalone_ic": str(self.standalone_ic) if self.standalone_ic else None,
            "marginal_ic": str(self.marginal_ic) if self.marginal_ic else None,
            "contribution_pct": str(self.contribution_pct) if self.contribution_pct else None,
            "is_additive": self.is_additive,
        }


@dataclass
class DecayPoint:
    """IC at a specific signal age."""
    age_days: int
    ic_value: Optional[Decimal]
    n_observations: int


@dataclass
class FactorDecayCurve:
    """IC decay over signal age."""
    factor_name: str
    decay_points: List[DecayPoint]
    half_life_days: Optional[int]  # Days until IC halves
    decay_rate: Optional[Decimal]  # Per-day decay

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor_name": self.factor_name,
            "half_life_days": self.half_life_days,
            "decay_rate": str(self.decay_rate) if self.decay_rate else None,
            "decay_points": [
                {"age_days": p.age_days, "ic": str(p.ic_value) if p.ic_value else None}
                for p in self.decay_points
            ],
        }


@dataclass
class RegimePerformance:
    """Performance metrics for a specific regime."""
    regime: str
    ic_mean: Optional[Decimal]
    ic_median: Optional[Decimal]
    hit_rate: Optional[Decimal]  # % of positive IC periods
    n_periods: int
    spread_q5_q1: Optional[Decimal]  # Top vs bottom quintile return

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime,
            "ic_mean": str(self.ic_mean) if self.ic_mean else None,
            "ic_median": str(self.ic_median) if self.ic_median else None,
            "hit_rate": str(self.hit_rate) if self.hit_rate else None,
            "n_periods": self.n_periods,
            "spread_q5_q1": str(self.spread_q5_q1) if self.spread_q5_q1 else None,
        }


@dataclass
class FactorAttributionResult:
    """Complete factor attribution analysis result."""
    as_of_date: str
    horizon: str
    factor_ics: Dict[str, FactorIC]
    factor_contributions: Dict[str, FactorContribution]
    composite_ic: Optional[Decimal]
    best_factor: str
    worst_factor: str
    regime_performance: Dict[str, RegimePerformance]
    decay_curves: Dict[str, FactorDecayCurve]
    diagnostics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "as_of_date": self.as_of_date,
            "horizon": self.horizon,
            "composite_ic": str(self.composite_ic) if self.composite_ic else None,
            "best_factor": self.best_factor,
            "worst_factor": self.worst_factor,
            "factor_ics": {k: v.to_dict() for k, v in self.factor_ics.items()},
            "factor_contributions": {k: v.to_dict() for k, v in self.factor_contributions.items()},
            "regime_performance": {k: v.to_dict() for k, v in self.regime_performance.items()},
            "decay_curves": {k: v.to_dict() for k, v in self.decay_curves.items()},
            "diagnostics": self.diagnostics,
        }


def _to_decimal(value: Any, default: Optional[Decimal] = None) -> Optional[Decimal]:
    """Safely convert value to Decimal."""
    if value is None:
        return default
    try:
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))
    except Exception:
        return default


def _quantize(value: Decimal, precision: str = "0.0001") -> Decimal:
    """Quantize decimal to specified precision."""
    return value.quantize(Decimal(precision), rounding=ROUND_HALF_UP)


def _rank_data(values: List[Decimal]) -> List[float]:
    """Compute ranks with average tie-breaking."""
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


def compute_spearman_ic(
    scores: List[Decimal],
    returns: List[Decimal],
) -> Optional[Decimal]:
    """
    Compute Spearman rank correlation (Information Coefficient).

    Args:
        scores: Factor scores
        returns: Forward returns

    Returns:
        IC value or None if insufficient data
    """
    n = len(scores)
    if n < 10 or len(returns) != n:
        return None

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
    return _quantize(Decimal(str(ic)))


def compute_t_statistic(
    ic: Decimal,
    n: int,
) -> Optional[Decimal]:
    """Compute t-statistic for IC."""
    if n < 10:
        return None

    # t = IC * sqrt(n-2) / sqrt(1 - IC^2)
    ic_float = float(ic)
    if abs(ic_float) >= 1.0:
        return None

    import math
    try:
        t = ic_float * math.sqrt(n - 2) / math.sqrt(1 - ic_float ** 2)
        return _quantize(Decimal(str(t)), "0.01")
    except (ValueError, ZeroDivisionError):
        return None


class FactorAttributionEngine:
    """
    Engine for computing factor attribution analysis.

    Decomposes portfolio performance by factor to understand
    which signals are driving returns.

    Usage:
        engine = FactorAttributionEngine()
        result = engine.compute_attribution(
            scores_by_factor={
                "clinical": [(ticker, score), ...],
                "financial": [(ticker, score), ...],
            },
            returns={ticker: return_decimal, ...},
            composite_scores={ticker: composite_score, ...},
            as_of_date=date(2026, 1, 15),
            horizon="63d"
        )
    """

    VERSION = "1.0.0"

    # Minimum observations for reliable IC
    MIN_OBS_IC = 10

    # Decay curve age buckets (days)
    DECAY_AGE_BUCKETS = [7, 14, 30, 60, 90]

    def __init__(
        self,
        min_obs_ic: int = 10,
    ):
        """
        Initialize factor attribution engine.

        Args:
            min_obs_ic: Minimum observations for IC calculation
        """
        self.min_obs_ic = min_obs_ic

    def compute_factor_ic(
        self,
        factor_name: str,
        scores: Dict[str, Decimal],
        returns: Dict[str, Decimal],
    ) -> FactorIC:
        """
        Compute IC for a single factor.

        Args:
            factor_name: Name of the factor
            scores: Dict of ticker -> factor score
            returns: Dict of ticker -> forward return

        Returns:
            FactorIC with IC value and statistics
        """
        # Align scores and returns
        common_tickers = set(scores.keys()) & set(returns.keys())
        if len(common_tickers) < self.min_obs_ic:
            return FactorIC(
                factor_name=factor_name,
                ic_value=None,
                t_statistic=None,
                n_observations=len(common_tickers),
                is_significant=False,
                direction="NEUTRAL",
            )

        aligned_scores = [scores[t] for t in sorted(common_tickers)]
        aligned_returns = [returns[t] for t in sorted(common_tickers)]

        ic = compute_spearman_ic(aligned_scores, aligned_returns)
        t_stat = compute_t_statistic(ic, len(common_tickers)) if ic else None

        # Determine significance and direction
        is_significant = t_stat is not None and abs(t_stat) > Decimal("2.0")

        if ic is None:
            direction = "NEUTRAL"
        elif ic > Decimal("0.03"):
            direction = "POSITIVE"
        elif ic < Decimal("-0.03"):
            direction = "NEGATIVE"
        else:
            direction = "NEUTRAL"

        return FactorIC(
            factor_name=factor_name,
            ic_value=ic,
            t_statistic=t_stat,
            n_observations=len(common_tickers),
            is_significant=is_significant,
            direction=direction,
        )

    def compute_marginal_contribution(
        self,
        factor_name: str,
        factor_ic: Optional[Decimal],
        composite_ic: Optional[Decimal],
        all_factor_ics: Dict[str, Optional[Decimal]],
    ) -> FactorContribution:
        """
        Compute marginal contribution of a factor.

        Args:
            factor_name: Name of the factor
            factor_ic: IC of this factor
            composite_ic: IC of composite score
            all_factor_ics: ICs of all factors

        Returns:
            FactorContribution with marginal analysis
        """
        if factor_ic is None or composite_ic is None:
            return FactorContribution(
                factor_name=factor_name,
                standalone_ic=factor_ic,
                marginal_ic=None,
                contribution_pct=None,
                is_additive=False,
            )

        # Compute average IC of other factors
        other_ics = [ic for name, ic in all_factor_ics.items()
                     if name != factor_name and ic is not None]

        if not other_ics:
            return FactorContribution(
                factor_name=factor_name,
                standalone_ic=factor_ic,
                marginal_ic=factor_ic,
                contribution_pct=Decimal("100"),
                is_additive=True,
            )

        avg_other_ic = Decimal(str(mean([float(ic) for ic in other_ics])))
        marginal_ic = _quantize(composite_ic - avg_other_ic)

        # Contribution percentage
        total_ic_sum = sum(abs(ic) for ic in all_factor_ics.values() if ic is not None)
        if total_ic_sum > Decimal("0"):
            contribution_pct = _quantize(abs(factor_ic) / total_ic_sum * Decimal("100"), "0.1")
        else:
            contribution_pct = Decimal("0")

        # Is this factor additive (does it improve composite)?
        is_additive = factor_ic > Decimal("0") and marginal_ic > Decimal("0")

        return FactorContribution(
            factor_name=factor_name,
            standalone_ic=factor_ic,
            marginal_ic=marginal_ic,
            contribution_pct=contribution_pct,
            is_additive=is_additive,
        )

    def compute_decay_curve(
        self,
        factor_name: str,
        historical_ics: List[Tuple[int, Decimal]],  # (age_days, ic)
    ) -> FactorDecayCurve:
        """
        Compute IC decay over signal age.

        Args:
            factor_name: Name of the factor
            historical_ics: List of (signal_age_days, ic_value) tuples

        Returns:
            FactorDecayCurve with decay analysis
        """
        if not historical_ics:
            return FactorDecayCurve(
                factor_name=factor_name,
                decay_points=[],
                half_life_days=None,
                decay_rate=None,
            )

        # Bucket ICs by age
        buckets: Dict[int, List[Decimal]] = {age: [] for age in self.DECAY_AGE_BUCKETS}

        for age_days, ic in historical_ics:
            for bucket_age in self.DECAY_AGE_BUCKETS:
                if age_days <= bucket_age:
                    buckets[bucket_age].append(ic)
                    break

        # Compute average IC per bucket
        decay_points = []
        for age in self.DECAY_AGE_BUCKETS:
            ics = buckets[age]
            if ics:
                avg_ic = _quantize(Decimal(str(mean([float(ic) for ic in ics]))))
                decay_points.append(DecayPoint(
                    age_days=age,
                    ic_value=avg_ic,
                    n_observations=len(ics),
                ))
            else:
                decay_points.append(DecayPoint(
                    age_days=age,
                    ic_value=None,
                    n_observations=0,
                ))

        # Estimate half-life
        half_life = None
        decay_rate = None

        valid_points = [(p.age_days, p.ic_value) for p in decay_points
                        if p.ic_value is not None and p.ic_value > Decimal("0")]

        if len(valid_points) >= 2:
            first_age, first_ic = valid_points[0]
            last_age, last_ic = valid_points[-1]

            if first_ic > last_ic and last_ic > Decimal("0"):
                # Simple exponential decay estimate
                import math
                try:
                    # IC(t) = IC(0) * exp(-decay_rate * t)
                    # decay_rate = -ln(IC(t)/IC(0)) / t
                    ratio = float(last_ic / first_ic)
                    if ratio > 0:
                        decay_rate_val = -math.log(ratio) / (last_age - first_age)
                        decay_rate = _quantize(Decimal(str(decay_rate_val)), "0.0001")

                        # half_life = ln(2) / decay_rate
                        if decay_rate_val > 0:
                            half_life = int(math.log(2) / decay_rate_val)
                except (ValueError, ZeroDivisionError):
                    pass

        return FactorDecayCurve(
            factor_name=factor_name,
            decay_points=decay_points,
            half_life_days=half_life,
            decay_rate=decay_rate,
        )

    def compute_regime_performance(
        self,
        regime: str,
        period_ics: List[Decimal],
        period_spreads: List[Optional[Decimal]],
    ) -> RegimePerformance:
        """
        Compute performance metrics for a regime.

        Args:
            regime: Regime name
            period_ics: IC values for periods in this regime
            period_spreads: Q5-Q1 spreads for periods

        Returns:
            RegimePerformance with metrics
        """
        if not period_ics:
            return RegimePerformance(
                regime=regime,
                ic_mean=None,
                ic_median=None,
                hit_rate=None,
                n_periods=0,
                spread_q5_q1=None,
            )

        ic_floats = [float(ic) for ic in period_ics]
        ic_mean = _quantize(Decimal(str(mean(ic_floats))))
        ic_median = _quantize(Decimal(str(median(ic_floats))))
        hit_rate = _quantize(Decimal(str(sum(1 for ic in period_ics if ic > 0) / len(period_ics))), "0.01")

        valid_spreads = [s for s in period_spreads if s is not None]
        spread_mean = None
        if valid_spreads:
            spread_mean = _quantize(Decimal(str(mean([float(s) for s in valid_spreads]))))

        return RegimePerformance(
            regime=regime,
            ic_mean=ic_mean,
            ic_median=ic_median,
            hit_rate=hit_rate,
            n_periods=len(period_ics),
            spread_q5_q1=spread_mean,
        )

    def compute_attribution(
        self,
        scores_by_factor: Dict[str, Dict[str, Decimal]],
        returns: Dict[str, Decimal],
        composite_scores: Dict[str, Decimal],
        as_of_date: date,
        horizon: str = "63d",
        regime: Optional[str] = None,
    ) -> FactorAttributionResult:
        """
        Compute complete factor attribution analysis.

        Args:
            scores_by_factor: Dict of factor_name -> {ticker: score}
            returns: Dict of ticker -> forward return
            composite_scores: Dict of ticker -> composite score
            as_of_date: Analysis date
            horizon: Return horizon
            regime: Current market regime (optional)

        Returns:
            FactorAttributionResult with full analysis
        """
        # Compute IC for each factor
        factor_ics: Dict[str, FactorIC] = {}
        for factor_name, scores in scores_by_factor.items():
            factor_ics[factor_name] = self.compute_factor_ic(factor_name, scores, returns)

        # Compute composite IC
        composite_ic = None
        if composite_scores:
            composite_result = self.compute_factor_ic("composite", composite_scores, returns)
            composite_ic = composite_result.ic_value

        # Extract IC values for contribution analysis
        ic_values = {name: ic.ic_value for name, ic in factor_ics.items()}

        # Compute marginal contributions
        factor_contributions: Dict[str, FactorContribution] = {}
        for factor_name in scores_by_factor.keys():
            factor_contributions[factor_name] = self.compute_marginal_contribution(
                factor_name,
                ic_values.get(factor_name),
                composite_ic,
                ic_values,
            )

        # Find best and worst factors
        valid_ics = [(name, ic.ic_value) for name, ic in factor_ics.items()
                     if ic.ic_value is not None]

        if valid_ics:
            best_factor = max(valid_ics, key=lambda x: x[1])[0]
            worst_factor = min(valid_ics, key=lambda x: x[1])[0]
        else:
            best_factor = "unknown"
            worst_factor = "unknown"

        # Build diagnostics
        diagnostics = {
            "total_tickers_scored": len(composite_scores),
            "total_tickers_with_returns": len(returns),
            "coverage_pct": str(_quantize(
                Decimal(str(len(set(composite_scores.keys()) & set(returns.keys())) / max(1, len(composite_scores)))) * 100,
                "0.1"
            )),
            "factors_analyzed": list(scores_by_factor.keys()),
            "regime": regime,
        }

        return FactorAttributionResult(
            as_of_date=as_of_date.isoformat(),
            horizon=horizon,
            factor_ics=factor_ics,
            factor_contributions=factor_contributions,
            composite_ic=composite_ic,
            best_factor=best_factor,
            worst_factor=worst_factor,
            regime_performance={},  # Populated by multi-period analysis
            decay_curves={},  # Populated by historical analysis
            diagnostics=diagnostics,
        )


if __name__ == "__main__":
    # Demonstration
    print("=" * 70)
    print("FACTOR ATTRIBUTION ENGINE - DEMONSTRATION")
    print("=" * 70)

    engine = FactorAttributionEngine()

    # Generate sample data
    tickers = [f"TICK{i:02d}" for i in range(30)]

    # Sample factor scores (higher = better)
    clinical_scores = {t: Decimal(str(50 + i * 1.5)) for i, t in enumerate(tickers)}
    financial_scores = {t: Decimal(str(60 - i * 0.8)) for i, t in enumerate(tickers)}
    catalyst_scores = {t: Decimal(str(40 + (i % 10) * 3)) for i, t in enumerate(tickers)}

    # Sample returns (correlated with clinical scores for demo)
    returns = {t: Decimal(str(0.05 + i * 0.003 + (i % 5) * 0.01)) for i, t in enumerate(tickers)}

    # Composite scores
    composite_scores = {
        t: (clinical_scores[t] * Decimal("0.4") +
            financial_scores[t] * Decimal("0.35") +
            catalyst_scores[t] * Decimal("0.25"))
        for t in tickers
    }

    result = engine.compute_attribution(
        scores_by_factor={
            "clinical": clinical_scores,
            "financial": financial_scores,
            "catalyst": catalyst_scores,
        },
        returns=returns,
        composite_scores=composite_scores,
        as_of_date=date(2026, 1, 15),
        horizon="63d",
        regime="BULL",
    )

    print(f"\nAnalysis Date: {result.as_of_date}")
    print(f"Horizon: {result.horizon}")
    print(f"Composite IC: {result.composite_ic}")
    print(f"Best Factor: {result.best_factor}")
    print(f"Worst Factor: {result.worst_factor}")

    print("\nFactor ICs:")
    for name, ic in result.factor_ics.items():
        print(f"  {name}: IC={ic.ic_value}, t={ic.t_statistic}, sig={ic.is_significant}")

    print("\nFactor Contributions:")
    for name, contrib in result.factor_contributions.items():
        print(f"  {name}: standalone={contrib.standalone_ic}, marginal={contrib.marginal_ic}, additive={contrib.is_additive}")
