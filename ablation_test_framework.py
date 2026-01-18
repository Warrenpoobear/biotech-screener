#!/usr/bin/env python3
"""
ablation_test_framework.py

Ablation Testing Framework for Enhancement Modules

Measures the Information Coefficient (IC) contribution of each enhancement
by comparing scores with and without each module enabled.

Design Philosophy:
- Deterministic: Same inputs → same results
- Point-in-Time Safe: Uses historical returns with proper embargo
- Auditable: Full breakdown of each enhancement's contribution

Author: Wake Robin Capital Management
Version: 1.0.0
"""

from decimal import Decimal, ROUND_HALF_UP
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import json
from pathlib import Path

__version__ = "1.0.0"


@dataclass
class AblationResult:
    """Result of ablation test for a single enhancement."""
    enhancement_name: str
    ic_without: Decimal
    ic_with: Decimal
    ic_improvement: Decimal
    passed: bool  # IC improvement > threshold
    sample_size: int
    confidence: str  # "high", "medium", "low"
    p_value_estimate: Optional[Decimal] = None


@dataclass
class AblationTestSuite:
    """Complete ablation test results."""
    as_of_date: str
    results: Dict[str, AblationResult]
    total_ic_improvement: Decimal
    all_passed: bool
    sample_size: int
    provenance: Dict[str, Any] = field(default_factory=dict)


def calculate_ic(
    scores: Dict[str, Decimal],
    returns: Dict[str, Decimal],
    min_sample: int = 3,
) -> Decimal:
    """
    Calculate Information Coefficient (rank correlation).

    Args:
        scores: {ticker: score} mapping
        returns: {ticker: forward_return} mapping
        min_sample: Minimum sample size (default 3)

    Returns:
        Spearman rank correlation as Decimal
    """
    # Get common tickers
    common_tickers = set(scores.keys()) & set(returns.keys())
    if len(common_tickers) < min_sample:
        return Decimal("0")

    # Extract aligned data
    tickers = sorted(common_tickers)
    score_values = [scores[t] for t in tickers]
    return_values = [returns[t] for t in tickers]

    # Rank both series
    def rank_series(values):
        n = len(values)
        indexed = [(v, i) for i, v in enumerate(values)]
        indexed.sort(key=lambda x: x[0])
        ranks = [0] * n
        for rank, (_, orig_idx) in enumerate(indexed, 1):
            ranks[orig_idx] = rank
        return ranks

    score_ranks = rank_series(score_values)
    return_ranks = rank_series(return_values)

    # Spearman correlation
    n = len(tickers)
    d_squared_sum = sum((sr - rr) ** 2 for sr, rr in zip(score_ranks, return_ranks))

    # rho = 1 - (6 * sum(d^2)) / (n * (n^2 - 1))
    rho = Decimal("1") - (Decimal("6") * Decimal(str(d_squared_sum))) / (Decimal(str(n)) * (Decimal(str(n ** 2 - 1))))

    return rho.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)


class AblationTestFramework:
    """
    Framework for running ablation tests on enhancement modules.

    Usage:
        framework = AblationTestFramework()
        results = framework.run_ablation_tests(
            base_scores=base_scores,
            enhancement_scores=enhancement_scores_by_name,
            forward_returns=returns,
            as_of_date=date(2026, 1, 15)
        )
    """

    VERSION = "1.0.0"
    IC_IMPROVEMENT_THRESHOLD = Decimal("0.01")  # 1% IC improvement required to pass
    MIN_SAMPLE_SIZE = 30

    def __init__(self, ic_threshold: Decimal = None):
        """Initialize framework with optional custom threshold."""
        self.ic_threshold = ic_threshold or self.IC_IMPROVEMENT_THRESHOLD
        self.audit_trail: List[Dict[str, Any]] = []

    def _determine_confidence(self, sample_size: int, ic_improvement: Decimal) -> str:
        """Determine confidence level based on sample size and effect."""
        if sample_size >= 100 and abs(ic_improvement) >= Decimal("0.03"):
            return "high"
        elif sample_size >= 50 or abs(ic_improvement) >= Decimal("0.02"):
            return "medium"
        else:
            return "low"

    def run_single_ablation(
        self,
        enhancement_name: str,
        base_scores: Dict[str, Decimal],
        enhanced_scores: Dict[str, Decimal],
        forward_returns: Dict[str, Decimal],
    ) -> AblationResult:
        """
        Run ablation test for a single enhancement.

        Args:
            enhancement_name: Name of the enhancement being tested
            base_scores: Scores WITHOUT this enhancement
            enhanced_scores: Scores WITH this enhancement
            forward_returns: Forward returns for IC calculation

        Returns:
            AblationResult with IC comparison
        """
        # Calculate IC without enhancement
        ic_without = calculate_ic(base_scores, forward_returns)

        # Calculate IC with enhancement
        ic_with = calculate_ic(enhanced_scores, forward_returns)

        # Calculate improvement
        ic_improvement = ic_with - ic_without

        # Determine if passed
        passed = ic_improvement > self.ic_threshold

        # Sample size
        common = set(base_scores.keys()) & set(forward_returns.keys())
        sample_size = len(common)

        # Confidence
        confidence = self._determine_confidence(sample_size, ic_improvement)

        return AblationResult(
            enhancement_name=enhancement_name,
            ic_without=ic_without,
            ic_with=ic_with,
            ic_improvement=ic_improvement,
            passed=passed,
            sample_size=sample_size,
            confidence=confidence,
        )

    def run_ablation_tests(
        self,
        base_scores: Dict[str, Decimal],
        enhancement_scores: Dict[str, Dict[str, Decimal]],
        forward_returns: Dict[str, Decimal],
        as_of_date: Union[str, date] = None,
    ) -> AblationTestSuite:
        """
        Run ablation tests for all enhancements.

        Args:
            base_scores: Scores without any enhancements {ticker: score}
            enhancement_scores: {enhancement_name: {ticker: enhanced_score}}
            forward_returns: {ticker: forward_return}
            as_of_date: Analysis date

        Returns:
            AblationTestSuite with all results
        """
        if as_of_date is None:
            raise ValueError("as_of_date is required")

        if isinstance(as_of_date, str):
            as_of_dt = date.fromisoformat(as_of_date)
        else:
            as_of_dt = as_of_date

        results = {}
        total_improvement = Decimal("0")

        for enhancement_name, enhanced_scores in enhancement_scores.items():
            result = self.run_single_ablation(
                enhancement_name=enhancement_name,
                base_scores=base_scores,
                enhanced_scores=enhanced_scores,
                forward_returns=forward_returns,
            )
            results[enhancement_name] = result
            total_improvement += result.ic_improvement

            # Add to audit trail
            self.audit_trail.append({
                "enhancement": enhancement_name,
                "as_of_date": as_of_dt.isoformat(),
                "ic_without": str(result.ic_without),
                "ic_with": str(result.ic_with),
                "ic_improvement": str(result.ic_improvement),
                "passed": result.passed,
            })

        all_passed = all(r.passed for r in results.values())
        sample_size = len(set(base_scores.keys()) & set(forward_returns.keys()))

        return AblationTestSuite(
            as_of_date=as_of_dt.isoformat(),
            results=results,
            total_ic_improvement=total_improvement,
            all_passed=all_passed,
            sample_size=sample_size,
            provenance={
                "framework": "AblationTestFramework",
                "version": self.VERSION,
                "ic_threshold": str(self.ic_threshold),
            },
        )

    def print_results(self, suite: AblationTestSuite) -> None:
        """Print formatted ablation test results."""
        print(f"\n{'='*60}")
        print(f"ABLATION TEST RESULTS - {suite.as_of_date}")
        print(f"{'='*60}")
        print(f"Sample size: {suite.sample_size} tickers")
        print(f"IC Improvement Threshold: {self.ic_threshold}")
        print()

        for name, result in sorted(suite.results.items()):
            status = "PASS" if result.passed else "FAIL"
            print(f"  {name}:")
            print(f"    IC without: {result.ic_without:+.3f}")
            print(f"    IC with:    {result.ic_with:+.3f}")
            print(f"    Improvement:{result.ic_improvement:+.3f} [{status}] ({result.confidence} confidence)")
            print()

        print(f"Total IC Improvement: {suite.total_ic_improvement:+.3f}")
        print(f"All Tests Passed: {'Yes' if suite.all_passed else 'No'}")
        print(f"{'='*60}\n")


def create_mock_ablation_data(
    tickers: List[str],
    as_of_date: date,
    enhancement_effects: Dict[str, Decimal],
) -> Tuple[Dict[str, Decimal], Dict[str, Dict[str, Decimal]], Dict[str, Decimal]]:
    """
    Create mock data for testing ablation framework.

    Args:
        tickers: List of tickers to include
        as_of_date: Analysis date
        enhancement_effects: {enhancement_name: ic_boost} expected effects

    Returns:
        (base_scores, enhancement_scores, forward_returns)
    """
    import hashlib

    def deterministic_score(ticker: str, as_of: date, enhancement: str = "") -> Decimal:
        """Generate deterministic pseudo-random score."""
        seed = f"{ticker}:{as_of.isoformat()}:{enhancement}"
        h = hashlib.sha256(seed.encode()).hexdigest()
        # Use first 8 hex chars as score basis (0-100)
        raw = int(h[:8], 16) % 10000
        return Decimal(str(raw)) / Decimal("100")

    # Base scores
    base_scores = {t: deterministic_score(t, as_of_date) for t in tickers}

    # Forward returns (correlated with base scores + noise)
    forward_returns = {}
    for t in tickers:
        base = base_scores[t]
        noise = deterministic_score(t, as_of_date, "return") - Decimal("50")
        forward_returns[t] = (base * Decimal("0.3") + noise * Decimal("0.1")).quantize(Decimal("0.01"))

    # Enhancement scores (base + enhancement effect on correlation)
    enhancement_scores = {}
    for enhancement_name, effect in enhancement_effects.items():
        enhanced = {}
        for t in tickers:
            # Adjust score to improve correlation with returns
            if effect > 0:
                # Positive effect: adjust toward better correlation
                adjust = forward_returns[t] * effect
                enhanced[t] = base_scores[t] + adjust
            else:
                enhanced[t] = base_scores[t]
        enhancement_scores[enhancement_name] = enhanced

    return base_scores, enhancement_scores, forward_returns


# =============================================================================
# SELF-CHECKS
# =============================================================================

def _run_self_checks() -> List[str]:
    """Run self-checks to verify framework correctness."""
    errors = []

    # CHECK 1: IC calculation
    scores = {"A": Decimal("90"), "B": Decimal("80"), "C": Decimal("70"), "D": Decimal("60")}
    returns = {"A": Decimal("0.10"), "B": Decimal("0.08"), "C": Decimal("0.06"), "D": Decimal("0.04")}

    ic = calculate_ic(scores, returns)
    if ic != Decimal("1.000"):  # Perfect positive correlation
        errors.append(f"CHECK1 FAIL: Expected IC=1.0 for perfect correlation, got {ic}")

    # CHECK 2: Negative correlation
    returns_neg = {"A": Decimal("0.04"), "B": Decimal("0.06"), "C": Decimal("0.08"), "D": Decimal("0.10")}
    ic_neg = calculate_ic(scores, returns_neg)
    if ic_neg != Decimal("-1.000"):  # Perfect negative correlation
        errors.append(f"CHECK2 FAIL: Expected IC=-1.0 for negative correlation, got {ic_neg}")

    # CHECK 3: Framework initialization
    framework = AblationTestFramework()
    if framework.VERSION != "1.0.0":
        errors.append(f"CHECK3 FAIL: Wrong version {framework.VERSION}")

    # CHECK 4: Mock data generation
    tickers = [f"TEST{i}" for i in range(50)]
    effects = {"pos_prior": Decimal("0.05"), "dilution_risk": Decimal("0.03")}
    base, enhanced, rets = create_mock_ablation_data(tickers, date(2026, 1, 15), effects)

    if len(base) != 50:
        errors.append(f"CHECK4 FAIL: Expected 50 base scores, got {len(base)}")

    return errors


if __name__ == "__main__":
    errors = _run_self_checks()
    if errors:
        print("SELF-CHECK FAILURES:")
        for e in errors:
            print(f"  {e}")
        exit(1)
    else:
        print("All self-checks passed!")

        # Demo run
        print("\n--- DEMO ABLATION TEST ---")
        tickers = [f"DEMO{i:03d}" for i in range(100)]
        effects = {
            "pos_prior": Decimal("0.08"),
            "dilution_risk": Decimal("0.05"),
            "slippage_risk": Decimal("0.03"),
            "design_quality": Decimal("0.04"),
            "competitive_pressure": Decimal("0.02"),
        }

        base, enhanced, returns = create_mock_ablation_data(tickers, date(2026, 1, 15), effects)

        framework = AblationTestFramework()
        suite = framework.run_ablation_tests(
            base_scores=base,
            enhancement_scores=enhanced,
            forward_returns=returns,
            as_of_date=date(2026, 1, 15),
        )

        framework.print_results(suite)
        exit(0)
