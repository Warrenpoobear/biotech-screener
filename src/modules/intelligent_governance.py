"""
Intelligent Governance Layer for Biotech Screener

This module provides a governance-preserving intelligence layer that sits on top of
the deterministic feature/data layers. It enables adaptive optimization while
maintaining full auditability and IC-defensibility.

Architecture:
    INTELLIGENCE LAYER (This Module)
    - Sharpe-optimized weight learning
    - Non-linear interaction effects with business logic
    - Regime-adaptive weight orchestration
    - Ensemble ranking (multiple perspectives)

    FEATURE LAYER (Governed - Existing Modules)
    - Institutional signals (13F)
    - Financial health (SEC)
    - Clinical catalysts (CT.gov)
    - Momentum (Yahoo Finance)

    DATA LAYER (Deterministic - Existing)
    - Raw filings, trials, prices
    - SHA256 integrity checks
    - Point-in-time discipline

Key Design Principles:
    1. DETERMINISTIC: Same inputs always produce identical outputs
    2. AUDITABLE: Every decision has a machine-readable explanation
    3. BOUNDED: Optimizations are constrained to prevent runaway behavior
    4. SHRINKAGE: All estimates shrink toward stable priors
    5. FAILSAFE: Degrades gracefully to base weights on insufficient data

Author: Wake Robin Capital Management
Version: 1.0.0
Created: 2026-01-18
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Tuple, Callable, Set, TypeVar, Union

# Type variable for coalesce function
T = TypeVar("T")

# Type aliases for common parameter types
ScoresDict = Dict[str, Decimal]
WeightsDict = Dict[str, Decimal]
MetadataDict = Dict[str, Union[Decimal, int, float, str, None]]
TickerDataDict = Dict[str, Union[str, Decimal, int, float, MetadataDict, None]]

__version__ = "1.0.0"
__author__ = "Wake Robin Capital Management"


# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

WEIGHT_PRECISION = Decimal("0.0001")
SCORE_PRECISION = Decimal("0.01")
EPS = Decimal("0.000001")

# Sharpe optimization constraints
SHARPE_OPT_MIN_WEIGHT = Decimal("0.02")  # No component < 2%
SHARPE_OPT_MAX_WEIGHT = Decimal("0.60")  # No component > 60%
SHARPE_OPT_MIN_PERIODS = 12  # Minimum months for optimization
SHARPE_OPT_SHRINKAGE = Decimal("0.70")  # Shrinkage toward base weights
SHARPE_OPT_SMOOTHING = Decimal("0.80")  # Smoothing toward previous weights
SHARPE_OPT_EMBARGO_MONTHS = 1  # PIT embargo for forward returns

# Interaction effect bounds
INTERACTION_MAX_ADJUSTMENT = Decimal("3.0")  # Max ±3 points
INTERACTION_SYNERGY_CAP = Decimal("2.0")  # Max synergy bonus
INTERACTION_DISTRESS_CAP = Decimal("2.0")  # Max distress penalty

# Ensemble configuration
ENSEMBLE_COMPOSITE_WEIGHT = Decimal("0.50")
ENSEMBLE_MOMENTUM_WEIGHT = Decimal("0.25")
ENSEMBLE_VALUE_WEIGHT = Decimal("0.25")

# Regime thresholds for weight adaptation
REGIME_MAX_WEIGHT_DELTA = Decimal("0.15")  # Max weight change per regime

# V3 Production Base Weights (from config/v3_production_integration.py)
V3_PRODUCTION_WEIGHTS: Dict[str, Decimal] = {
    "clinical": Decimal("0.28"),
    "financial": Decimal("0.25"),
    "catalyst": Decimal("0.17"),
    "pos": Decimal("0.15"),
    "momentum": Decimal("0.10"),
    "valuation": Decimal("0.05"),
}


# =============================================================================
# ENUMS
# =============================================================================

class OptimizationMethod(str, Enum):
    """Weight optimization methods."""
    SHARPE_RATIO = "sharpe_ratio"
    IC_WEIGHTED = "ic_weighted"
    EQUAL_WEIGHT = "equal_weight"
    BASE_WEIGHTS = "base_weights"
    FALLBACK = "fallback"


class InteractionType(str, Enum):
    """Types of factor interactions."""
    SYNERGY = "synergy"  # Factors reinforce each other
    CONFLICT = "conflict"  # Factors work against each other
    AMPLIFICATION = "amplification"  # One factor amplifies another
    DAMPENING = "dampening"  # One factor dampens another


class RankingMethod(str, Enum):
    """Ensemble ranking methodologies."""
    COMPOSITE = "composite"  # Weighted factor combination
    MOMENTUM = "momentum"  # Pure momentum + institutional
    VALUE = "value"  # Financials + catalyst PoS
    QUALITY = "quality"  # Clinical + financial health


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class SharpeOptimizationResult:
    """Result of Sharpe-ratio weight optimization."""
    optimized_weights: Dict[str, Decimal]
    base_weights: Dict[str, Decimal]
    historical_sharpe: Decimal
    component_contributions: Dict[str, Decimal]
    optimization_method: OptimizationMethod
    training_periods: int
    confidence: Decimal

    # Governance fields
    shrinkage_applied: Decimal
    smoothing_applied: Decimal
    weights_clamped: bool
    l1_change_from_base: Decimal

    # Audit trail
    audit_hash: str
    provenance: Dict[str, Union[str, int, float]] = field(default_factory=dict)


@dataclass
class InteractionEffect:
    """A single interaction effect between factors."""
    name: str
    interaction_type: InteractionType
    factors_involved: List[str]
    adjustment: Decimal
    triggered: bool
    trigger_conditions: Dict[str, str]
    business_logic: str  # Human-readable explanation


@dataclass
class InteractionEffectsResult:
    """Aggregated interaction effects for a ticker."""
    ticker: str
    effects: List[InteractionEffect]
    total_adjustment: Decimal
    net_synergy: Decimal
    net_conflict: Decimal
    flags: List[str]
    confidence: Decimal


@dataclass
class EnsembleRank:
    """Ensemble ranking result for a ticker."""
    ticker: str

    # Individual method ranks
    composite_rank: int
    momentum_rank: int
    value_rank: int

    # Ensemble rank (weighted average)
    ensemble_rank: Decimal
    final_rank: int

    # Agreement metrics
    rank_agreement: Decimal  # How much methods agree (0-1)
    max_rank_divergence: int  # Largest disagreement

    # Method contributions
    method_contributions: Dict[str, Decimal]


@dataclass
class IntelligentGovernanceResult:
    """Complete result from the intelligent governance layer."""
    ticker: str

    # Optimized scoring
    base_score: Decimal
    optimized_score: Decimal
    score_delta: Decimal

    # Weight optimization
    sharpe_result: SharpeOptimizationResult
    effective_weights: Dict[str, Decimal]

    # Interaction effects
    interaction_result: InteractionEffectsResult

    # Ensemble ranking
    ensemble_rank: Optional[EnsembleRank]

    # Governance
    governance_flags: List[str]
    audit_hash: str
    schema_version: str = "1.0.0"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _to_decimal(
    value: Union[Decimal, int, float, str, None],
    default: Optional[Decimal] = None
) -> Optional[Decimal]:
    """Convert various types to Decimal with safe handling."""
    if value is None:
        return default
    try:
        if isinstance(value, Decimal):
            return value
        if isinstance(value, (int, float)):
            return Decimal(str(value))
        if isinstance(value, str):
            stripped = value.strip()
            return Decimal(stripped) if stripped else default
        return default
    except (InvalidOperation, ValueError):
        return default


def _coalesce(*vals: Optional[T], default: Optional[T] = None) -> Optional[T]:
    """Return first non-None value, avoiding truthiness bugs with 0/Decimal('0')."""
    for v in vals:
        if v is not None:
            return v
    return default


def _quantize_weight(value: Decimal) -> Decimal:
    """Quantize to weight precision."""
    return value.quantize(WEIGHT_PRECISION, rounding=ROUND_HALF_UP)


def _quantize_score(value: Decimal) -> Decimal:
    """Quantize to score precision."""
    return value.quantize(SCORE_PRECISION, rounding=ROUND_HALF_UP)


def _clamp(value: Decimal, min_val: Decimal, max_val: Decimal) -> Decimal:
    """Clamp value to range."""
    return max(min_val, min(max_val, value))


def _compute_l1_distance(w1: Dict[str, Decimal], w2: Dict[str, Decimal]) -> Decimal:
    """Compute L1 distance between two weight vectors."""
    all_keys = set(w1.keys()) | set(w2.keys())
    total = Decimal("0")
    for k in all_keys:
        v1 = w1.get(k, Decimal("0"))
        v2 = w2.get(k, Decimal("0"))
        total += abs(v1 - v2)
    return total


def _normalize_weights(weights: Dict[str, Decimal]) -> Dict[str, Decimal]:
    """Normalize weights to sum to 1.0."""
    total = sum(weights.values())
    if total <= EPS:
        # Equal weights as fallback
        n = len(weights)
        return {k: Decimal("1") / Decimal(n) for k in weights}
    return {k: _quantize_weight(v / total) for k, v in weights.items()}


def _compute_audit_hash(data: Dict[str, Union[str, int, float, Dict[str, str], None]]) -> str:
    """Compute deterministic audit hash."""
    def serialize(obj: Union[Decimal, date, Enum]) -> str:
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, date):
            return obj.isoformat()
        if isinstance(obj, Enum):
            return str(obj.value)
        raise TypeError(f"Cannot serialize {type(obj)}")

    canonical = json.dumps(data, sort_keys=True, default=serialize)
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"


# =============================================================================
# SHARPE-RATIO WEIGHT OPTIMIZER
# =============================================================================

class SharpeWeightOptimizer:
    """
    Sharpe-ratio based weight optimization with governance constraints.

    This optimizer learns optimal component weights by maximizing historical
    Sharpe ratio while maintaining:
    - PIT safety (embargo period for forward returns)
    - Weight bounds (min/max per component)
    - Shrinkage toward base weights
    - Smoothing toward previous period weights
    - L1 change limits to prevent dramatic shifts

    The optimization uses a coordinate descent approach on the Sharpe ratio
    surface, which is deterministic and does not require scipy.

    Usage:
        optimizer = SharpeWeightOptimizer()
        result = optimizer.optimize(
            historical_scores=scores,
            forward_returns=returns,
            base_weights={"clinical": Decimal("0.40"), ...},
            as_of_date=date(2026, 1, 15)
        )
    """

    def __init__(
        self,
        min_weight: Decimal = SHARPE_OPT_MIN_WEIGHT,
        max_weight: Decimal = SHARPE_OPT_MAX_WEIGHT,
        min_periods: int = SHARPE_OPT_MIN_PERIODS,
        shrinkage_lambda: Decimal = SHARPE_OPT_SHRINKAGE,
        smooth_gamma: Decimal = SHARPE_OPT_SMOOTHING,
        embargo_months: int = SHARPE_OPT_EMBARGO_MONTHS,
    ):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_periods = min_periods
        self.shrinkage_lambda = shrinkage_lambda
        self.smooth_gamma = smooth_gamma
        self.embargo_months = embargo_months

    def optimize(
        self,
        historical_scores: List[Dict[str, Any]],
        forward_returns: Dict[Tuple[date, str], Decimal],
        base_weights: Dict[str, Decimal],
        as_of_date: date,
        prev_weights: Optional[Dict[str, Decimal]] = None,
        lookback_months: int = 12,
    ) -> SharpeOptimizationResult:
        """
        Optimize weights to maximize historical Sharpe ratio.

        Args:
            historical_scores: List of dicts with ticker, component scores, as_of_date
            forward_returns: Dict keyed by (as_of_date, ticker) -> forward return
            base_weights: Prior weights to shrink toward
            as_of_date: Current date for PIT cutoff
            prev_weights: Previous period weights for smoothing
            lookback_months: Months of history to use

        Returns:
            SharpeOptimizationResult with optimized weights and diagnostics
        """
        # Extract component names
        components = list(base_weights.keys())

        # Validate data sufficiency
        valid_periods = self._count_valid_periods(
            historical_scores, forward_returns, as_of_date, lookback_months
        )

        if valid_periods < self.min_periods:
            return self._fallback_result(
                base_weights,
                reason=f"insufficient_periods_{valid_periods}",
                training_periods=valid_periods
            )

        # Build period-by-period data matrix
        period_data = self._build_period_data(
            historical_scores, forward_returns, as_of_date, lookback_months, components
        )

        if not period_data:
            return self._fallback_result(
                base_weights,
                reason="no_valid_period_data",
                training_periods=0
            )

        # Coordinate descent optimization
        optimized_weights, sharpe, contributions = self._coordinate_descent(
            period_data, components, base_weights
        )

        # Apply shrinkage toward base weights
        shrunk_weights = self._apply_shrinkage(
            optimized_weights, base_weights, self.shrinkage_lambda
        )

        # Apply smoothing toward previous weights
        if prev_weights:
            smoothed_weights = self._apply_smoothing(
                shrunk_weights, prev_weights, self.smooth_gamma
            )
            smoothing_applied = self.smooth_gamma
        else:
            smoothed_weights = shrunk_weights
            smoothing_applied = Decimal("0")

        # Apply weight bounds
        bounded_weights, was_clamped = self._apply_bounds(smoothed_weights)

        # Normalize to sum to 1.0
        final_weights = _normalize_weights(bounded_weights)

        # Compute L1 change from base
        l1_change = _compute_l1_distance(final_weights, base_weights)

        # Compute confidence based on sample size and optimization quality
        confidence = self._compute_confidence(
            valid_periods, sharpe, l1_change
        )

        # Build provenance
        provenance = {
            "as_of_date": as_of_date.isoformat(),
            "lookback_months": lookback_months,
            "embargo_months": self.embargo_months,
            "training_periods": valid_periods,
            "optimization_method": OptimizationMethod.SHARPE_RATIO.value,
        }

        # Compute audit hash
        audit_data = {
            "weights": {k: str(v) for k, v in final_weights.items()},
            "sharpe": str(sharpe),
            "provenance": provenance,
        }
        audit_hash = _compute_audit_hash(audit_data)

        return SharpeOptimizationResult(
            optimized_weights=final_weights,
            base_weights=base_weights,
            historical_sharpe=_quantize_score(sharpe),
            component_contributions=contributions,
            optimization_method=OptimizationMethod.SHARPE_RATIO,
            training_periods=valid_periods,
            confidence=_quantize_weight(confidence),
            shrinkage_applied=self.shrinkage_lambda,
            smoothing_applied=smoothing_applied,
            weights_clamped=was_clamped,
            l1_change_from_base=_quantize_weight(l1_change),
            audit_hash=audit_hash,
            provenance=provenance,
        )

    def _count_valid_periods(
        self,
        historical_scores: List[Dict[str, Any]],
        forward_returns: Dict[Tuple[date, str], Decimal],
        as_of_date: date,
        lookback_months: int,
    ) -> int:
        """Count periods with sufficient data for optimization."""
        lookback_cutoff = as_of_date - timedelta(days=lookback_months * 30)
        embargo_cutoff = as_of_date - timedelta(days=self.embargo_months * 30)

        # Group by date
        dates_with_data: Set[date] = set()
        for rec in historical_scores:
            rec_date = self._parse_date(rec.get("as_of_date"))
            if rec_date is None:
                continue
            if rec_date < lookback_cutoff or rec_date > embargo_cutoff:
                continue

            ticker = rec.get("ticker")
            if (rec_date, ticker) in forward_returns:
                dates_with_data.add(rec_date)

        return len(dates_with_data)

    def _build_period_data(
        self,
        historical_scores: List[Dict[str, Any]],
        forward_returns: Dict[Tuple[date, str], Decimal],
        as_of_date: date,
        lookback_months: int,
        components: List[str],
    ) -> List[Dict[str, Any]]:
        """Build period-by-period data for optimization."""
        lookback_cutoff = as_of_date - timedelta(days=lookback_months * 30)
        embargo_cutoff = as_of_date - timedelta(days=self.embargo_months * 30)

        # Group scores by date
        scores_by_date: Dict[date, List[Dict[str, Any]]] = {}
        for rec in historical_scores:
            rec_date = self._parse_date(rec.get("as_of_date"))
            if rec_date is None:
                continue
            if rec_date < lookback_cutoff or rec_date > embargo_cutoff:
                continue
            if rec_date not in scores_by_date:
                scores_by_date[rec_date] = []
            scores_by_date[rec_date].append(rec)

        # Build period data with composite scores and returns
        period_data = []
        for score_date in sorted(scores_by_date.keys()):
            records = scores_by_date[score_date]

            period_scores = []
            period_returns = []

            for rec in records:
                ticker = rec.get("ticker")
                ret_key = (score_date, ticker)
                if ret_key not in forward_returns:
                    continue

                # Extract component scores (use _coalesce to avoid 0-as-falsy bug)
                comp_scores = {}
                for comp in components:
                    score = _to_decimal(_coalesce(
                        rec.get(f"{comp}_normalized"),
                        rec.get(comp),
                        rec.get(f"{comp}_score"),
                    ))
                    if score is not None:
                        comp_scores[comp] = score

                if len(comp_scores) >= len(components) * 0.5:  # At least 50% coverage
                    period_scores.append({
                        "ticker": ticker,
                        "scores": comp_scores,
                    })
                    period_returns.append(forward_returns[ret_key])

            if len(period_scores) >= 10:  # Minimum cross-section
                period_data.append({
                    "date": score_date,
                    "scores": period_scores,
                    "returns": period_returns,
                })

        return period_data

    def _coordinate_descent(
        self,
        period_data: List[Dict[str, Any]],
        components: List[str],
        initial_weights: Dict[str, Decimal],
    ) -> Tuple[Dict[str, Decimal], Decimal, Dict[str, Decimal]]:
        """
        Coordinate descent optimization for Sharpe ratio.

        Iteratively adjusts each weight while holding others fixed,
        moving in the direction that improves Sharpe ratio.
        """
        weights = initial_weights.copy()
        best_sharpe = self._compute_sharpe(period_data, weights, components)

        step_size = Decimal("0.05")
        min_step = Decimal("0.005")
        max_iterations = 50

        for iteration in range(max_iterations):
            improved = False

            for comp in components:
                # Try increasing weight
                weights_plus = weights.copy()
                weights_plus[comp] = weights[comp] + step_size
                weights_plus = _normalize_weights(weights_plus)
                sharpe_plus = self._compute_sharpe(period_data, weights_plus, components)

                # Try decreasing weight
                weights_minus = weights.copy()
                weights_minus[comp] = max(Decimal("0.01"), weights[comp] - step_size)
                weights_minus = _normalize_weights(weights_minus)
                sharpe_minus = self._compute_sharpe(period_data, weights_minus, components)

                # Pick best direction
                if sharpe_plus > best_sharpe and sharpe_plus >= sharpe_minus:
                    weights = weights_plus
                    best_sharpe = sharpe_plus
                    improved = True
                elif sharpe_minus > best_sharpe:
                    weights = weights_minus
                    best_sharpe = sharpe_minus
                    improved = True

            if not improved:
                step_size = step_size / Decimal("2")
                if step_size < min_step:
                    break

        # Compute component contributions
        contributions = self._compute_contributions(period_data, weights, components)

        return weights, best_sharpe, contributions

    def _compute_sharpe(
        self,
        period_data: List[Dict[str, Any]],
        weights: Dict[str, Decimal],
        components: List[str],
    ) -> Decimal:
        """
        Compute Sharpe ratio for given weights using long-short portfolio.

        For each period:
        1. Compute composite scores using weights
        2. Rank tickers by composite score
        3. Long top tercile, short bottom tercile
        4. Compute portfolio return

        Final Sharpe = mean(returns) / std(returns)
        """
        portfolio_returns: List[Decimal] = []

        for period in period_data:
            scores = period["scores"]
            returns = period["returns"]

            # Compute composite scores
            composites = []
            for i, rec in enumerate(scores):
                composite = Decimal("0")
                for comp in components:
                    if comp in rec["scores"] and comp in weights:
                        composite += rec["scores"][comp] * weights[comp]
                composites.append((composite, returns[i], rec["ticker"]))

            # Sort by composite (deterministic tiebreak by ticker)
            composites.sort(key=lambda x: (x[0], x[2]), reverse=True)

            n = len(composites)
            if n < 6:
                continue

            # Long top tercile, short bottom tercile
            tercile_size = n // 3
            long_return = sum(c[1] for c in composites[:tercile_size]) / Decimal(tercile_size)
            short_return = sum(c[1] for c in composites[-tercile_size:]) / Decimal(tercile_size)

            portfolio_return = long_return - short_return
            portfolio_returns.append(portfolio_return)

        if len(portfolio_returns) < 3:
            return Decimal("-999")  # Invalid

        # Compute Sharpe ratio
        mean_return = sum(portfolio_returns) / Decimal(len(portfolio_returns))

        variance = sum((r - mean_return) ** 2 for r in portfolio_returns) / Decimal(len(portfolio_returns))
        if variance <= EPS:
            return Decimal("0")

        std_return = variance.sqrt()
        sharpe = mean_return / std_return if std_return > EPS else Decimal("0")

        return sharpe

    def _compute_contributions(
        self,
        period_data: List[Dict[str, Any]],
        weights: Dict[str, Decimal],
        components: List[str],
    ) -> Dict[str, Decimal]:
        """Compute relative contribution of each component to returns."""
        contributions: Dict[str, List[Decimal]] = {c: [] for c in components}

        for period in period_data:
            scores = period["scores"]
            returns = period["returns"]

            for i, rec in enumerate(scores):
                for comp in components:
                    if comp in rec["scores"]:
                        # Approximate contribution: score * return
                        contrib = rec["scores"][comp] * returns[i]
                        contributions[comp].append(contrib)

        # Average and normalize
        avg_contrib = {}
        for comp, contribs in contributions.items():
            if contribs:
                avg_contrib[comp] = sum(contribs) / Decimal(len(contribs))
            else:
                avg_contrib[comp] = Decimal("0")

        total = sum(abs(v) for v in avg_contrib.values())
        if total > EPS:
            return {k: _quantize_weight(v / total) for k, v in avg_contrib.items()}

        return {c: Decimal("0") for c in components}

    def _apply_shrinkage(
        self,
        weights: Dict[str, Decimal],
        base_weights: Dict[str, Decimal],
        shrinkage: Decimal,
    ) -> Dict[str, Decimal]:
        """Shrink weights toward base weights."""
        result = {}
        for k in set(weights.keys()) | set(base_weights.keys()):
            w = weights.get(k, Decimal("0"))
            b = base_weights.get(k, Decimal("0"))
            result[k] = (Decimal("1") - shrinkage) * w + shrinkage * b
        return result

    def _apply_smoothing(
        self,
        weights: Dict[str, Decimal],
        prev_weights: Dict[str, Decimal],
        gamma: Decimal,
    ) -> Dict[str, Decimal]:
        """Smooth weights toward previous period."""
        result = {}
        for k in set(weights.keys()) | set(prev_weights.keys()):
            w = weights.get(k, Decimal("0"))
            p = prev_weights.get(k, w)
            result[k] = (Decimal("1") - gamma) * w + gamma * p
        return result

    def _apply_bounds(
        self,
        weights: Dict[str, Decimal],
    ) -> Tuple[Dict[str, Decimal], bool]:
        """Apply min/max weight bounds."""
        was_clamped = False
        result = {}
        for k, v in weights.items():
            clamped = _clamp(v, self.min_weight, self.max_weight)
            if clamped != v:
                was_clamped = True
            result[k] = clamped
        return result, was_clamped

    def _compute_confidence(
        self,
        n_periods: int,
        sharpe: Decimal,
        l1_change: Decimal,
    ) -> Decimal:
        """Compute confidence in optimization result."""
        # Base confidence from sample size
        if n_periods >= 24:
            size_conf = Decimal("0.8")
        elif n_periods >= 12:
            size_conf = Decimal("0.6")
        elif n_periods >= 6:
            size_conf = Decimal("0.4")
        else:
            size_conf = Decimal("0.2")

        # Sharpe quality
        if sharpe >= Decimal("1.0"):
            sharpe_conf = Decimal("0.9")
        elif sharpe >= Decimal("0.5"):
            sharpe_conf = Decimal("0.7")
        elif sharpe >= Decimal("0.2"):
            sharpe_conf = Decimal("0.5")
        else:
            sharpe_conf = Decimal("0.3")

        # Penalize large weight changes (might be overfitting)
        if l1_change < Decimal("0.10"):
            change_conf = Decimal("1.0")
        elif l1_change < Decimal("0.20"):
            change_conf = Decimal("0.8")
        else:
            change_conf = Decimal("0.6")

        return (size_conf + sharpe_conf + change_conf) / Decimal("3")

    def _fallback_result(
        self,
        base_weights: Dict[str, Decimal],
        reason: str,
        training_periods: int,
    ) -> SharpeOptimizationResult:
        """Return fallback result when optimization fails."""
        return SharpeOptimizationResult(
            optimized_weights=base_weights.copy(),
            base_weights=base_weights,
            historical_sharpe=Decimal("0"),
            component_contributions={k: Decimal("0") for k in base_weights},
            optimization_method=OptimizationMethod.FALLBACK,
            training_periods=training_periods,
            confidence=Decimal("0.1"),
            shrinkage_applied=Decimal("1.0"),
            smoothing_applied=Decimal("0"),
            weights_clamped=False,
            l1_change_from_base=Decimal("0"),
            audit_hash=_compute_audit_hash({"reason": reason}),
            provenance={"fallback_reason": reason},
        )

    def _parse_date(self, value: Union[date, str, None]) -> Optional[date]:
        """Parse date from various formats."""
        if value is None:
            return None
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value[:10])
            except ValueError:
                return None
        return None


# =============================================================================
# BUSINESS LOGIC INTERACTION EFFECTS
# =============================================================================

class InteractionEffectsEngine:
    """
    Non-linear interaction effects with explicit business logic.

    Each interaction has a clear, auditable business justification:

    SYNERGIES (factors reinforce each other):
    1. Strong institutional + near catalyst = amplification
       "Smart money positioning for known event"
    2. Strong clinical + strong runway = conviction quality
       "Good science with time to execute"
    3. Positive momentum + institutional buying = confirmation
       "Price action validated by smart money"

    CONFLICTS (factors work against each other):
    1. Strong momentum + weak fundamentals = fade signal
       "Momentum without substance, mean reversion risk"
    2. High institutional + deteriorating financials = warning
       "Smart money may be trapped or late to exit"
    3. Near catalyst + high short interest = squeeze/dump risk
       "Binary event with crowded positioning"

    All effects use smooth ramps (not hard thresholds) to prevent rank churn.
    """

    def __init__(self, max_adjustment: Decimal = INTERACTION_MAX_ADJUSTMENT):
        self.max_adjustment = max_adjustment

    def compute_effects(
        self,
        ticker: str,
        scores: ScoresDict,
        metadata: MetadataDict,
    ) -> InteractionEffectsResult:
        """
        Compute all applicable interaction effects for a ticker.

        Args:
            ticker: Security ticker
            scores: Dict of normalized scores (0-100) by component
            metadata: Additional data (runway_months, days_to_catalyst, etc.)

        Returns:
            InteractionEffectsResult with all effects and total adjustment
        """
        effects: List[InteractionEffect] = []
        flags: List[str] = []

        # Extract commonly used values
        clinical = scores.get("clinical", scores.get("clinical_dev", Decimal("50")))
        financial = scores.get("financial", Decimal("50"))
        catalyst = scores.get("catalyst", Decimal("50"))
        momentum = scores.get("momentum", Decimal("50"))
        institutional = scores.get("institutional", scores.get("smart_money", Decimal("50")))

        runway_months = _to_decimal(metadata.get("runway_months"))  # No default - missing data != 24 months
        days_to_catalyst = metadata.get("days_to_catalyst")
        short_interest_pct = _to_decimal(metadata.get("short_interest_pct"), Decimal("0"))

        # =====================================================================
        # SYNERGY 1: Institutional + Near Catalyst = Amplification
        # =====================================================================
        if institutional > Decimal("70") and days_to_catalyst is not None:
            if 0 < days_to_catalyst <= 60:  # Within 60 days
                proximity_factor = self._smooth_ramp(
                    Decimal(60 - days_to_catalyst), Decimal("0"), Decimal("60")
                )
                inst_factor = self._smooth_ramp(
                    institutional, Decimal("60"), Decimal("80")
                )
                adjustment = proximity_factor * inst_factor * Decimal("1.5")

                effects.append(InteractionEffect(
                    name="institutional_catalyst_amplification",
                    interaction_type=InteractionType.AMPLIFICATION,
                    factors_involved=["institutional", "catalyst"],
                    adjustment=_quantize_score(adjustment),
                    triggered=adjustment > Decimal("0.1"),
                    trigger_conditions={
                        "institutional": f">{institutional}",
                        "days_to_catalyst": f"{days_to_catalyst}d"
                    },
                    business_logic="Smart money positioning for imminent catalyst event"
                ))
                if adjustment > Decimal("0.5"):
                    flags.append("institutional_catalyst_synergy")

        # =====================================================================
        # SYNERGY 2: Clinical + Runway = Quality Conviction
        # =====================================================================
        if runway_months is not None and clinical > Decimal("65") and runway_months > Decimal("18"):
            clinical_factor = self._smooth_ramp(
                clinical, Decimal("60"), Decimal("80")
            )
            runway_factor = self._smooth_ramp(
                runway_months, Decimal("12"), Decimal("36")
            )
            adjustment = clinical_factor * runway_factor * Decimal("1.2")

            effects.append(InteractionEffect(
                name="clinical_runway_conviction",
                interaction_type=InteractionType.SYNERGY,
                factors_involved=["clinical", "runway_months"],
                adjustment=_quantize_score(adjustment),
                triggered=adjustment > Decimal("0.1"),
                trigger_conditions={
                    "clinical": f">{clinical}",
                    "runway_months": f"{runway_months}mo"
                },
                business_logic="Strong clinical progress with runway to execute"
            ))
            if adjustment > Decimal("0.5"):
                flags.append("quality_conviction")

        # =====================================================================
        # SYNERGY 3: Momentum + Institutional Buying = Confirmation
        # =====================================================================
        inst_buying = metadata.get("institutional_net_change", Decimal("0"))
        if momentum > Decimal("60") and inst_buying > Decimal("0"):
            momentum_factor = self._smooth_ramp(
                momentum, Decimal("55"), Decimal("75")
            )
            buying_factor = min(inst_buying / Decimal("10"), Decimal("1.0"))
            adjustment = momentum_factor * buying_factor * Decimal("1.0")

            effects.append(InteractionEffect(
                name="momentum_institutional_confirmation",
                interaction_type=InteractionType.SYNERGY,
                factors_involved=["momentum", "institutional"],
                adjustment=_quantize_score(adjustment),
                triggered=adjustment > Decimal("0.1"),
                trigger_conditions={
                    "momentum": f">{momentum}",
                    "inst_net_change": f"+{inst_buying}%"
                },
                business_logic="Price momentum confirmed by smart money accumulation"
            ))
            if adjustment > Decimal("0.5"):
                flags.append("momentum_confirmed")

        # =====================================================================
        # CONFLICT 1: Momentum + Weak Fundamentals = Fade Signal
        # =====================================================================
        if momentum > Decimal("70") and financial < Decimal("40"):
            momentum_excess = self._smooth_ramp(
                momentum, Decimal("65"), Decimal("85")
            )
            financial_weakness = self._smooth_ramp_inverted(
                financial, Decimal("30"), Decimal("50")
            )
            adjustment = -(momentum_excess * financial_weakness * Decimal("1.5"))

            effects.append(InteractionEffect(
                name="momentum_fundamental_conflict",
                interaction_type=InteractionType.CONFLICT,
                factors_involved=["momentum", "financial"],
                adjustment=_quantize_score(adjustment),
                triggered=abs(adjustment) > Decimal("0.1"),
                trigger_conditions={
                    "momentum": f">{momentum}",
                    "financial": f"<{financial}"
                },
                business_logic="Strong momentum without fundamental support - mean reversion risk"
            ))
            if adjustment < Decimal("-0.5"):
                flags.append("fade_momentum_signal")

        # =====================================================================
        # CONFLICT 2: Institutional + Deteriorating Financials = Warning
        # =====================================================================
        if institutional > Decimal("65") and financial < Decimal("35"):
            inst_factor = self._smooth_ramp(
                institutional, Decimal("60"), Decimal("80")
            )
            financial_distress = self._smooth_ramp_inverted(
                financial, Decimal("25"), Decimal("45")
            )
            adjustment = -(inst_factor * financial_distress * Decimal("1.2"))

            effects.append(InteractionEffect(
                name="institutional_financial_warning",
                interaction_type=InteractionType.CONFLICT,
                factors_involved=["institutional", "financial"],
                adjustment=_quantize_score(adjustment),
                triggered=abs(adjustment) > Decimal("0.1"),
                trigger_conditions={
                    "institutional": f">{institutional}",
                    "financial": f"<{financial}"
                },
                business_logic="High institutional ownership with deteriorating runway - trapped capital risk"
            ))
            if adjustment < Decimal("-0.5"):
                flags.append("institutional_trapped_warning")

        # =====================================================================
        # CONFLICT 3: Catalyst + High Short Interest = Binary Risk
        # =====================================================================
        if days_to_catalyst is not None and short_interest_pct > Decimal("15"):
            if days_to_catalyst <= 30:
                proximity = self._smooth_ramp(
                    Decimal(30 - days_to_catalyst), Decimal("0"), Decimal("30")
                )
                short_factor = self._smooth_ramp(
                    short_interest_pct, Decimal("10"), Decimal("30")
                )
                adjustment = -(proximity * short_factor * Decimal("1.0"))

                effects.append(InteractionEffect(
                    name="catalyst_short_interest_risk",
                    interaction_type=InteractionType.CONFLICT,
                    factors_involved=["catalyst", "short_interest"],
                    adjustment=_quantize_score(adjustment),
                    triggered=abs(adjustment) > Decimal("0.1"),
                    trigger_conditions={
                        "days_to_catalyst": f"{days_to_catalyst}d",
                        "short_interest": f"{short_interest_pct}%"
                    },
                    business_logic="Near-term catalyst with crowded short positioning - binary outcome risk"
                ))
                if adjustment < Decimal("-0.5"):
                    flags.append("binary_event_risk")

        # =====================================================================
        # DAMPENING: High Volatility Dampens Catalyst Signal
        # =====================================================================
        vol = _to_decimal(metadata.get("annualized_vol"), Decimal("0.50"))
        if vol > Decimal("0.80") and catalyst > Decimal("60"):
            vol_excess = self._smooth_ramp(vol, Decimal("0.60"), Decimal("1.20"))
            catalyst_excess = (catalyst - Decimal("50")) / Decimal("50")
            dampening = vol_excess * catalyst_excess * Decimal("0.20") * catalyst
            adjustment = -dampening

            effects.append(InteractionEffect(
                name="volatility_catalyst_dampening",
                interaction_type=InteractionType.DAMPENING,
                factors_involved=["volatility", "catalyst"],
                adjustment=_quantize_score(adjustment),
                triggered=abs(adjustment) > Decimal("0.1"),
                trigger_conditions={
                    "annualized_vol": f"{vol:.0%}",
                    "catalyst": f"{catalyst}"
                },
                business_logic="High volatility reduces reliability of catalyst signal"
            ))
            if abs(adjustment) > Decimal("0.5"):
                flags.append("vol_dampened_catalyst")

        # Aggregate effects
        triggered_effects = [e for e in effects if e.triggered]
        total_adjustment = sum((e.adjustment for e in triggered_effects), Decimal("0"))

        # Apply global cap
        total_adjustment = _clamp(total_adjustment, -self.max_adjustment, self.max_adjustment)

        # Compute net synergy vs conflict
        net_synergy = sum(
            (e.adjustment for e in triggered_effects
             if e.interaction_type in (InteractionType.SYNERGY, InteractionType.AMPLIFICATION)),
            Decimal("0")
        )
        net_conflict = sum(
            (e.adjustment for e in triggered_effects
             if e.interaction_type in (InteractionType.CONFLICT, InteractionType.DAMPENING)),
            Decimal("0")
        )

        # Confidence based on number and agreement of effects
        if len(triggered_effects) == 0:
            confidence = Decimal("0.5")  # Neutral
        elif abs(net_synergy + net_conflict) > Decimal("1.0"):
            confidence = Decimal("0.7")  # Clear signal
        else:
            confidence = Decimal("0.4")  # Mixed signals

        return InteractionEffectsResult(
            ticker=ticker,
            effects=triggered_effects,
            total_adjustment=_quantize_score(total_adjustment),
            net_synergy=_quantize_score(net_synergy),
            net_conflict=_quantize_score(abs(net_conflict)),
            flags=flags,
            confidence=confidence,
        )

    def _smooth_ramp(
        self, value: Decimal, low: Decimal, high: Decimal
    ) -> Decimal:
        """Compute smooth ramp from 0 at low to 1 at high."""
        if value <= low:
            return Decimal("0")
        if value >= high:
            return Decimal("1")
        return (value - low) / (high - low)

    def _smooth_ramp_inverted(
        self, value: Decimal, low: Decimal, high: Decimal
    ) -> Decimal:
        """Compute inverted smooth ramp from 1 at low to 0 at high."""
        if value <= low:
            return Decimal("1")
        if value >= high:
            return Decimal("0")
        return (high - value) / (high - low)


# =============================================================================
# ENSEMBLE RANKING
# =============================================================================

class EnsembleRanker:
    """
    Ensemble ranking combining multiple methodologies.

    Combines three independent ranking perspectives:
    1. COMPOSITE: Standard weighted factor combination
    2. MOMENTUM: Pure momentum + institutional (trend following)
    3. VALUE: Financials + catalyst PoS (value/catalyst)

    Final rank is a weighted average of ranks (not scores), which is more
    robust than averaging scores directly.

    This approach:
    - Diversifies across methodologies
    - Reduces single-model overfitting
    - Each sub-model is still fully explainable
    """

    def __init__(
        self,
        composite_weight: Decimal = ENSEMBLE_COMPOSITE_WEIGHT,
        momentum_weight: Decimal = ENSEMBLE_MOMENTUM_WEIGHT,
        value_weight: Decimal = ENSEMBLE_VALUE_WEIGHT,
    ):
        self.composite_weight = composite_weight
        self.momentum_weight = momentum_weight
        self.value_weight = value_weight

    def compute_ranks(
        self,
        ticker_data: List[Dict[str, Any]],
    ) -> List[EnsembleRank]:
        """
        Compute ensemble ranks for all tickers.

        Args:
            ticker_data: List of dicts with ticker, scores by component

        Returns:
            List of EnsembleRank for each ticker
        """
        if not ticker_data:
            return []

        # Compute individual method scores
        composite_scores = self._compute_composite_scores(ticker_data)
        momentum_scores = self._compute_momentum_scores(ticker_data)
        value_scores = self._compute_value_scores(ticker_data)

        # Convert to ranks (with deterministic tiebreak)
        tickers = [d.get("ticker", "") for d in ticker_data]
        composite_ranks = self._scores_to_ranks(composite_scores, tickers)
        momentum_ranks = self._scores_to_ranks(momentum_scores, tickers)
        value_ranks = self._scores_to_ranks(value_scores, tickers)

        # Compute ensemble ranks (weighted average of ranks)
        results = []
        for i, ticker in enumerate(tickers):
            c_rank = composite_ranks[i]
            m_rank = momentum_ranks[i]
            v_rank = value_ranks[i]

            ensemble_rank = (
                Decimal(c_rank) * self.composite_weight +
                Decimal(m_rank) * self.momentum_weight +
                Decimal(v_rank) * self.value_weight
            )

            # Compute agreement metrics
            ranks = [c_rank, m_rank, v_rank]
            max_div = max(ranks) - min(ranks)

            # Agreement: 1 - (std(ranks) / n), using pure Decimal for determinism
            dec_ranks = [Decimal(r) for r in ranks]
            mean_rank = sum(dec_ranks) / Decimal("3")
            variance = sum((r - mean_rank) ** 2 for r in dec_ranks) / Decimal("3")
            std_rank = variance.sqrt() if variance > Decimal("0") else Decimal("0")
            agreement = Decimal("1") - (std_rank / Decimal(len(tickers)))
            agreement = _clamp(agreement, Decimal("0"), Decimal("1"))

            results.append(EnsembleRank(
                ticker=ticker,
                composite_rank=c_rank,
                momentum_rank=m_rank,
                value_rank=v_rank,
                ensemble_rank=_quantize_score(ensemble_rank),
                final_rank=0,  # Will be assigned after sorting
                rank_agreement=_quantize_weight(agreement),
                max_rank_divergence=max_div,
                method_contributions={
                    "composite": self.composite_weight,
                    "momentum": self.momentum_weight,
                    "value": self.value_weight,
                },
            ))

        # Sort by ensemble rank and assign final ranks
        results.sort(key=lambda x: (x.ensemble_rank, x.ticker))
        for i, result in enumerate(results):
            result.final_rank = i + 1

        return results

    def _compute_composite_scores(
        self, ticker_data: List[Dict[str, Any]]
    ) -> List[Decimal]:
        """Compute standard composite scores."""
        scores = []
        for d in ticker_data:
            score = Decimal("0")
            # Standard weights
            score += _to_decimal(d.get("clinical"), Decimal("50")) * Decimal("0.35")
            score += _to_decimal(d.get("financial"), Decimal("50")) * Decimal("0.30")
            score += _to_decimal(d.get("catalyst"), Decimal("50")) * Decimal("0.20")
            score += _to_decimal(d.get("momentum"), Decimal("50")) * Decimal("0.15")
            scores.append(score)
        return scores

    def _compute_momentum_scores(
        self, ticker_data: List[Dict[str, Any]]
    ) -> List[Decimal]:
        """Compute momentum-focused scores."""
        scores = []
        for d in ticker_data:
            score = Decimal("0")
            # Momentum + institutional focus
            score += _to_decimal(d.get("momentum"), Decimal("50")) * Decimal("0.50")
            score += _to_decimal(d.get("institutional"), d.get("smart_money", Decimal("50"))) * Decimal("0.30")
            score += _to_decimal(d.get("catalyst"), Decimal("50")) * Decimal("0.20")
            scores.append(score)
        return scores

    def _compute_value_scores(
        self, ticker_data: List[Dict[str, Any]]
    ) -> List[Decimal]:
        """Compute value-focused scores."""
        scores = []
        for d in ticker_data:
            score = Decimal("0")
            # Financials + clinical + PoS focus
            score += _to_decimal(d.get("financial"), Decimal("50")) * Decimal("0.35")
            score += _to_decimal(d.get("clinical"), Decimal("50")) * Decimal("0.30")
            score += _to_decimal(d.get("pos"), Decimal("50")) * Decimal("0.20")
            score += _to_decimal(d.get("valuation"), Decimal("50")) * Decimal("0.15")
            scores.append(score)
        return scores

    def _scores_to_ranks(
        self,
        scores: List[Decimal],
        tickers: List[str],
    ) -> List[int]:
        """Convert scores to ranks with deterministic tiebreak."""
        indexed = [(score, ticker, i) for i, (score, ticker) in enumerate(zip(scores, tickers))]
        indexed.sort(key=lambda x: (-x[0], x[1]))  # Higher score = better = lower rank

        ranks = [0] * len(scores)
        for rank_idx, (_, _, orig_idx) in enumerate(indexed):
            ranks[orig_idx] = rank_idx + 1

        return ranks


# =============================================================================
# REGIME-ADAPTIVE WEIGHT ORCHESTRATOR
# =============================================================================

class RegimeAdaptiveOrchestrator:
    """
    Orchestrates weight adaptation across market regimes.

    Combines:
    1. Base weights (from configuration)
    2. Sharpe-optimized adjustments
    3. Regime-specific multipliers

    Maintains governance by:
    - Capping regime-driven changes
    - Logging all weight modifications
    - Providing full audit trail
    """

    # Regime-specific multipliers (from research)
    REGIME_MULTIPLIERS = {
        "BULL": {
            "clinical": Decimal("1.0"),
            "financial": Decimal("0.85"),
            "catalyst": Decimal("1.20"),
            "momentum": Decimal("1.25"),
            "pos": Decimal("1.0"),
            "valuation": Decimal("0.80"),
        },
        "BEAR": {
            "clinical": Decimal("1.0"),
            "financial": Decimal("1.30"),
            "catalyst": Decimal("0.80"),
            "momentum": Decimal("0.60"),
            "pos": Decimal("1.10"),
            "valuation": Decimal("1.15"),
        },
        "VOLATILITY_SPIKE": {
            "clinical": Decimal("0.90"),
            "financial": Decimal("1.40"),
            "catalyst": Decimal("0.70"),
            "momentum": Decimal("0.50"),
            "pos": Decimal("1.0"),
            "valuation": Decimal("1.0"),
        },
        "NEUTRAL": {
            "clinical": Decimal("1.0"),
            "financial": Decimal("1.0"),
            "catalyst": Decimal("1.0"),
            "momentum": Decimal("1.0"),
            "pos": Decimal("1.0"),
            "valuation": Decimal("1.0"),
        },
    }

    def __init__(self, max_weight_delta: Decimal = REGIME_MAX_WEIGHT_DELTA):
        self.max_weight_delta = max_weight_delta

    def adapt_weights(
        self,
        base_weights: Dict[str, Decimal],
        regime: str,
        sharpe_weights: Optional[Dict[str, Decimal]] = None,
        sharpe_confidence: Decimal = Decimal("0"),
    ) -> Tuple[Dict[str, Decimal], Dict[str, Any]]:
        """
        Adapt weights based on regime and optimization.

        Args:
            base_weights: Starting weights
            regime: Current market regime
            sharpe_weights: Optional Sharpe-optimized weights
            sharpe_confidence: Confidence in Sharpe optimization

        Returns:
            Tuple of (adapted_weights, diagnostics)
        """
        diagnostics = {
            "base_weights": {k: str(v) for k, v in base_weights.items()},
            "regime": regime,
            "adaptations_applied": [],
        }

        # Start with base weights
        weights = base_weights.copy()

        # Apply Sharpe optimization (if confident)
        if sharpe_weights and sharpe_confidence >= Decimal("0.40"):
            # Blend based on confidence
            blend = sharpe_confidence * Decimal("0.5")  # Max 50% Sharpe influence
            weights = self._blend_weights(weights, sharpe_weights, blend)
            diagnostics["adaptations_applied"].append(f"sharpe_blend_{blend}")

        # Apply regime multipliers
        multipliers = self.REGIME_MULTIPLIERS.get(regime, self.REGIME_MULTIPLIERS["NEUTRAL"])
        regime_weights = {}
        for k, v in weights.items():
            mult = multipliers.get(k, Decimal("1.0"))
            regime_weights[k] = v * mult

        # Normalize
        regime_weights = _normalize_weights(regime_weights)

        # Cap changes from base
        final_weights = {}
        for k in base_weights:
            base_val = base_weights[k]
            regime_val = regime_weights.get(k, base_val)
            delta = regime_val - base_val
            capped_delta = _clamp(delta, -self.max_weight_delta, self.max_weight_delta)
            final_weights[k] = base_val + capped_delta

        # Normalize again
        final_weights = _normalize_weights(final_weights)

        diagnostics["regime_multipliers"] = {k: str(v) for k, v in multipliers.items()}
        diagnostics["final_weights"] = {k: str(v) for k, v in final_weights.items()}
        diagnostics["adaptations_applied"].append(f"regime_{regime}")

        return final_weights, diagnostics

    def _blend_weights(
        self,
        w1: Dict[str, Decimal],
        w2: Dict[str, Decimal],
        alpha: Decimal,
    ) -> Dict[str, Decimal]:
        """Blend two weight sets: result = (1-alpha)*w1 + alpha*w2."""
        result = {}
        all_keys = set(w1.keys()) | set(w2.keys())
        for k in all_keys:
            v1 = w1.get(k, Decimal("0"))
            v2 = w2.get(k, v1)
            result[k] = (Decimal("1") - alpha) * v1 + alpha * v2
        return result


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

class IntelligentGovernanceLayer:
    """
    Main orchestrator for the intelligent governance layer.

    Combines:
    - Sharpe weight optimization
    - Interaction effects
    - Regime adaptation
    - Ensemble ranking

    Provides complete scoring with full auditability.

    Usage:
        layer = IntelligentGovernanceLayer()

        result = layer.compute(
            ticker="ACME",
            scores={
                "clinical": Decimal("72"),
                "financial": Decimal("65"),
                "catalyst": Decimal("58"),
                "momentum": Decimal("70"),
            },
            metadata={
                "runway_months": Decimal("24"),
                "days_to_catalyst": 30,
            },
            base_weights={...},
            regime="BULL",
        )
    """

    def __init__(
        self,
        enable_sharpe_optimization: bool = True,
        enable_interaction_effects: bool = True,
        enable_regime_adaptation: bool = True,
        smartness: Decimal = Decimal("0.5"),
    ):
        """
        Initialize the intelligent governance layer.

        Args:
            enable_sharpe_optimization: Enable Sharpe-ratio weight optimization
            enable_interaction_effects: Enable non-linear interaction effects
            enable_regime_adaptation: Enable regime-based weight adaptation
            smartness: Control knob from 0 (conservative/governed) to 1 (aggressive/smart)
                      - 0.0: Max shrinkage, min interactions, strict missing data handling
                      - 0.5: Balanced defaults (recommended for production)
                      - 1.0: Min shrinkage, max interactions, looser caps
        """
        self.enable_sharpe_optimization = enable_sharpe_optimization
        self.enable_interaction_effects = enable_interaction_effects
        self.enable_regime_adaptation = enable_regime_adaptation
        self.smartness = _clamp(smartness, Decimal("0"), Decimal("1"))

        # Compute smartness-dependent parameters
        # Interaction cap: 2.0 at smartness=0 → 4.0 at smartness=1
        interaction_cap = Decimal("2.0") + self.smartness * Decimal("2.0")

        # Sharpe blend max: 25% at smartness=0 → 60% at smartness=1
        sharpe_blend_factor = Decimal("0.25") + self.smartness * Decimal("0.35")

        # Regime delta cap: 10% at smartness=0 → 20% at smartness=1
        regime_delta_cap = Decimal("0.10") + self.smartness * Decimal("0.10")

        # Shrinkage: 90% at smartness=0 → 50% at smartness=1 (lower = less governed)
        shrinkage = Decimal("0.90") - self.smartness * Decimal("0.40")

        self.sharpe_optimizer = SharpeWeightOptimizer(shrinkage_lambda=shrinkage)
        self.interaction_engine = InteractionEffectsEngine(max_adjustment=interaction_cap)
        self.regime_orchestrator = RegimeAdaptiveOrchestrator(max_weight_delta=regime_delta_cap)
        self.ensemble_ranker = EnsembleRanker()

        # Store derived parameters for governance logging
        self._smartness_params = {
            "smartness": str(self.smartness),
            "interaction_cap": str(interaction_cap),
            "sharpe_blend_factor": str(sharpe_blend_factor),
            "regime_delta_cap": str(regime_delta_cap),
            "shrinkage": str(shrinkage),
        }

    def compute(
        self,
        ticker: str,
        scores: Dict[str, Decimal],
        metadata: Dict[str, Any],
        base_weights: Dict[str, Decimal],
        regime: str = "NEUTRAL",
        historical_scores: Optional[List[Dict[str, Any]]] = None,
        forward_returns: Optional[Dict[Tuple[date, str], Decimal]] = None,
        as_of_date: Optional[date] = None,
        prev_weights: Optional[Dict[str, Decimal]] = None,
    ) -> IntelligentGovernanceResult:
        """
        Compute intelligent governance result for a single ticker.

        Args:
            ticker: Security ticker
            scores: Dict of normalized scores by component
            metadata: Additional data (runway, catalyst timing, etc.)
            base_weights: Base weight configuration
            regime: Current market regime
            historical_scores: For Sharpe optimization
            forward_returns: For Sharpe optimization
            as_of_date: Current date
            prev_weights: Previous period weights

        Returns:
            IntelligentGovernanceResult with complete analysis
        """
        governance_flags: List[str] = []

        # Step 1: Sharpe weight optimization
        if (self.enable_sharpe_optimization and
            historical_scores and forward_returns and as_of_date):
            sharpe_result = self.sharpe_optimizer.optimize(
                historical_scores=historical_scores,
                forward_returns=forward_returns,
                base_weights=base_weights,
                as_of_date=as_of_date,
                prev_weights=prev_weights,
            )
            if sharpe_result.optimization_method == OptimizationMethod.SHARPE_RATIO:
                governance_flags.append("sharpe_optimized")
        else:
            sharpe_result = self.sharpe_optimizer._fallback_result(
                base_weights, "optimization_disabled", 0
            )

        # Step 2: Regime-adaptive weight orchestration
        if self.enable_regime_adaptation:
            effective_weights, regime_diag = self.regime_orchestrator.adapt_weights(
                base_weights=base_weights,
                regime=regime,
                sharpe_weights=sharpe_result.optimized_weights,
                sharpe_confidence=sharpe_result.confidence,
            )
            governance_flags.append(f"regime_adapted_{regime}")
        else:
            effective_weights = base_weights.copy()

        # Step 3: Handle missing scores - drop and renormalize (don't impute 50)
        available_weights: Dict[str, Decimal] = {}
        missing_components: List[str] = []

        for comp, weight in effective_weights.items():
            score_val = scores.get(comp)
            if score_val is not None:
                available_weights[comp] = weight
            else:
                missing_components.append(comp)

        # Log missing data governance flags
        if missing_components:
            coverage_pct = int(100 * len(available_weights) / len(effective_weights))
            governance_flags.append(f"missing_{len(missing_components)}_components_coverage_{coverage_pct}pct")
            for missing in missing_components:
                governance_flags.append(f"excluded_{missing}_missing")

        # Renormalize weights to available components only
        if available_weights:
            scoring_weights = _normalize_weights(available_weights)
        else:
            # Fallback: if ALL components missing, use effective weights with 50 imputation
            scoring_weights = effective_weights
            governance_flags.append("all_components_missing_imputed_50")

        # Compute base score using only available data
        base_score = Decimal("0")
        for comp, weight in scoring_weights.items():
            comp_score = scores.get(comp, Decimal("50"))  # 50 only for edge case above
            base_score += comp_score * weight
        base_score = _quantize_score(base_score)

        # Step 4: Interaction effects
        if self.enable_interaction_effects:
            interaction_result = self.interaction_engine.compute_effects(
                ticker=ticker,
                scores=scores,
                metadata=metadata,
            )
            governance_flags.extend(interaction_result.flags)
        else:
            interaction_result = InteractionEffectsResult(
                ticker=ticker,
                effects=[],
                total_adjustment=Decimal("0"),
                net_synergy=Decimal("0"),
                net_conflict=Decimal("0"),
                flags=[],
                confidence=Decimal("0.5"),
            )

        # Step 5: Compute optimized score
        optimized_score = base_score + interaction_result.total_adjustment
        optimized_score = _clamp(optimized_score, Decimal("0"), Decimal("100"))
        optimized_score = _quantize_score(optimized_score)

        score_delta = optimized_score - base_score

        # Step 6: Compute audit hash
        audit_data = {
            "ticker": ticker,
            "base_score": str(base_score),
            "optimized_score": str(optimized_score),
            "weights": {k: str(v) for k, v in effective_weights.items()},
            "interaction_adjustment": str(interaction_result.total_adjustment),
            "regime": regime,
        }
        audit_hash = _compute_audit_hash(audit_data)

        return IntelligentGovernanceResult(
            ticker=ticker,
            base_score=base_score,
            optimized_score=optimized_score,
            score_delta=score_delta,
            sharpe_result=sharpe_result,
            effective_weights=effective_weights,
            interaction_result=interaction_result,
            ensemble_rank=None,  # Set during batch processing
            governance_flags=governance_flags,
            audit_hash=audit_hash,
        )

    def compute_batch(
        self,
        ticker_data: List[Dict[str, Any]],
        base_weights: Dict[str, Decimal],
        regime: str = "NEUTRAL",
        historical_scores: Optional[List[Dict[str, Any]]] = None,
        forward_returns: Optional[Dict[Tuple[date, str], Decimal]] = None,
        as_of_date: Optional[date] = None,
        prev_weights: Optional[Dict[str, Decimal]] = None,
    ) -> Tuple[List[IntelligentGovernanceResult], List[EnsembleRank]]:
        """
        Compute intelligent governance for a batch of tickers.

        Returns both individual results and ensemble rankings.
        """
        results = []

        # Compute individual results
        for data in ticker_data:
            ticker = data.get("ticker", "")
            scores = {k: _to_decimal(v, Decimal("50")) for k, v in data.items()
                     if k not in ("ticker", "metadata")}
            metadata = data.get("metadata", {})

            result = self.compute(
                ticker=ticker,
                scores=scores,
                metadata=metadata,
                base_weights=base_weights,
                regime=regime,
                historical_scores=historical_scores,
                forward_returns=forward_returns,
                as_of_date=as_of_date,
                prev_weights=prev_weights,
            )
            results.append(result)

        # Compute ensemble ranks
        ensemble_data = []
        for data, result in zip(ticker_data, results):
            ensemble_entry = {"ticker": data.get("ticker", "")}
            for k, v in data.items():
                if k != "ticker":
                    ensemble_entry[k] = _to_decimal(v, Decimal("50"))
            ensemble_data.append(ensemble_entry)

        ensemble_ranks = self.ensemble_ranker.compute_ranks(ensemble_data)

        # Attach ensemble ranks to results
        rank_by_ticker = {r.ticker: r for r in ensemble_ranks}
        for result in results:
            result.ensemble_rank = rank_by_ticker.get(result.ticker)

        return results, ensemble_ranks


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstration() -> None:
    """Demonstrate the intelligent governance layer."""
    print("=" * 70)
    print("INTELLIGENT GOVERNANCE LAYER - DEMONSTRATION")
    print("=" * 70)
    print()

    # Create layer
    layer = IntelligentGovernanceLayer(
        enable_sharpe_optimization=False,  # Needs historical data
        enable_interaction_effects=True,
        enable_regime_adaptation=True,
    )

    # Sample ticker data
    ticker_data = [
        {
            "ticker": "ACME",
            "clinical": Decimal("75"),
            "financial": Decimal("68"),
            "catalyst": Decimal("62"),
            "momentum": Decimal("72"),
            "institutional": Decimal("78"),
            "metadata": {
                "runway_months": Decimal("30"),
                "days_to_catalyst": 25,
                "short_interest_pct": Decimal("8"),
            }
        },
        {
            "ticker": "BETA",
            "clinical": Decimal("65"),
            "financial": Decimal("35"),
            "catalyst": Decimal("70"),
            "momentum": Decimal("80"),
            "institutional": Decimal("45"),
            "metadata": {
                "runway_months": Decimal("8"),
                "short_interest_pct": Decimal("22"),
            }
        },
        {
            "ticker": "GAMMA",
            "clinical": Decimal("82"),
            "financial": Decimal("72"),
            "catalyst": Decimal("55"),
            "momentum": Decimal("48"),
            "institutional": Decimal("85"),
            "metadata": {
                "runway_months": Decimal("36"),
                "institutional_net_change": Decimal("5"),
            }
        },
    ]

    base_weights = {
        "clinical": Decimal("0.30"),
        "financial": Decimal("0.25"),
        "catalyst": Decimal("0.20"),
        "momentum": Decimal("0.15"),
        "institutional": Decimal("0.10"),
    }

    # Compute in BULL regime
    print("REGIME: BULL")
    print("-" * 70)

    results, ensemble_ranks = layer.compute_batch(
        ticker_data=ticker_data,
        base_weights=base_weights,
        regime="BULL",
    )

    for result in results:
        print(f"\n{result.ticker}:")
        print(f"  Base Score: {result.base_score}")
        print(f"  Optimized Score: {result.optimized_score}")
        print(f"  Score Delta: {result.score_delta:+}")
        print(f"  Governance Flags: {result.governance_flags}")

        if result.interaction_result.effects:
            print(f"  Interaction Effects:")
            for effect in result.interaction_result.effects:
                print(f"    - {effect.name}: {effect.adjustment:+} ({effect.business_logic})")

        if result.ensemble_rank:
            print(f"  Ensemble Rank: #{result.ensemble_rank.final_rank}")
            print(f"    Composite: #{result.ensemble_rank.composite_rank}")
            print(f"    Momentum: #{result.ensemble_rank.momentum_rank}")
            print(f"    Value: #{result.ensemble_rank.value_rank}")

    print()
    print("=" * 70)
    print("Ensemble Rankings (Final)")
    print("-" * 70)
    for rank in sorted(ensemble_ranks, key=lambda x: x.final_rank):
        print(f"#{rank.final_rank} {rank.ticker}: "
              f"Ensemble={rank.ensemble_rank:.2f}, "
              f"Agreement={rank.rank_agreement:.2f}, "
              f"MaxDiv={rank.max_rank_divergence}")

    print()


if __name__ == "__main__":
    demonstration()
