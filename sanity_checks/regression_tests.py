"""
Query 7.7: Regression Testing Against Known Cases

Creates test suite of historical scenarios that must pass:

1. Golden Test Cases (Must Rank Top 20)
2. Negative Test Cases (Must Rank Bottom 50%)
3. Edge Case Validation

This ensures the model captures known winners and filters known losers.

Author: Wake Robin Capital Management
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional

from sanity_checks.types import (
    CheckCategory,
    FlagSeverity,
    GoldenTestCase,
    RankingSnapshot,
    SanityCheckResult,
    SanityFlag,
    SecurityContext,
    ThresholdConfig,
    DEFAULT_THRESHOLDS,
)

logger = logging.getLogger(__name__)


@dataclass
class RegressionTestResult:
    """Result of a single regression test case."""
    test_case: GoldenTestCase
    actual_rank: Optional[int]
    passed: bool
    message: str


# Pre-defined golden test cases
GOLDEN_POSITIVE_CASES = [
    GoldenTestCase(
        ticker="ARWR",
        as_of_date="2024-01-15",
        expected_outcome="top_20",
        case_type="positive",
        description="Arrowhead - amvuttra approval momentum, siRNA success story",
        threshold_rank=20,
    ),
    GoldenTestCase(
        ticker="IONS",
        as_of_date="2024-11-15",
        expected_outcome="top_20",
        case_type="positive",
        description="Ionis - tofersen approval path, antisense pioneer",
        threshold_rank=20,
    ),
    GoldenTestCase(
        ticker="BGNE",
        as_of_date="2023-06-15",
        expected_outcome="top_20",
        case_type="positive",
        description="BeiGene - zanubrutinib launch trajectory, BTK success",
        threshold_rank=20,
    ),
]

GOLDEN_NEGATIVE_CASES = [
    GoldenTestCase(
        ticker="SAVA",
        as_of_date="2022-06-15",
        expected_outcome="bottom_50",
        case_type="negative",
        description="Cassava - questionable data integrity concerns pre-collapse",
        threshold_rank=50,  # Should be ranked below 50th percentile
    ),
]

EDGE_CASES = [
    GoldenTestCase(
        ticker="PLACEHOLDER_PREREVENUE",
        as_of_date="2024-01-15",
        expected_outcome="below_100",
        case_type="edge",
        description="Pre-revenue, pre-Phase 2, <$100M cap - should not rank highly",
        threshold_rank=100,
    ),
]


class RegressionTestRunner:
    """
    Regression test runner for screening model.

    Validates model against known historical outcomes.
    """

    def __init__(
        self,
        golden_positive: Optional[List[GoldenTestCase]] = None,
        golden_negative: Optional[List[GoldenTestCase]] = None,
        edge_cases: Optional[List[GoldenTestCase]] = None,
        config: Optional[ThresholdConfig] = None,
    ) -> None:
        self.golden_positive = golden_positive or GOLDEN_POSITIVE_CASES
        self.golden_negative = golden_negative or GOLDEN_NEGATIVE_CASES
        self.edge_cases = edge_cases or EDGE_CASES
        self.config = config or DEFAULT_THRESHOLDS

    def run_all_tests(
        self,
        current_snapshot: RankingSnapshot,
        universe_size: int = 100,
    ) -> SanityCheckResult:
        """
        Run all regression tests.

        Args:
            current_snapshot: Current ranking snapshot to validate
            universe_size: Total universe size for percentile calculations

        Returns:
            SanityCheckResult with all flags
        """
        flags: List[SanityFlag] = []
        test_results: List[RegressionTestResult] = []

        # Build lookup
        rank_lookup = {s.ticker: s.rank for s in current_snapshot.securities if s.rank}

        # 1. Golden Positive Tests (must rank top N)
        positive_results = self._run_positive_tests(rank_lookup)
        test_results.extend(positive_results)

        for result in positive_results:
            if not result.passed:
                flags.append(SanityFlag(
                    severity=FlagSeverity.CRITICAL,
                    category=CheckCategory.REGRESSION,
                    ticker=result.test_case.ticker,
                    check_name="golden_positive_failed",
                    message=f"Validated winner {result.test_case.ticker} ranked #{result.actual_rank} (expected top {result.test_case.threshold_rank})",
                    details={
                        "description": result.test_case.description,
                        "expected_outcome": result.test_case.expected_outcome,
                        "actual_rank": result.actual_rank,
                        "threshold": result.test_case.threshold_rank,
                    },
                    recommendation="Missing validated winner patterns - investigate scoring",
                ))

        # 2. Golden Negative Tests (must NOT rank highly)
        negative_results = self._run_negative_tests(rank_lookup, universe_size)
        test_results.extend(negative_results)

        for result in negative_results:
            if not result.passed:
                flags.append(SanityFlag(
                    severity=FlagSeverity.CRITICAL,
                    category=CheckCategory.REGRESSION,
                    ticker=result.test_case.ticker,
                    check_name="golden_negative_failed",
                    message=f"Known loser {result.test_case.ticker} ranked #{result.actual_rank} (should be bottom 50%)",
                    details={
                        "description": result.test_case.description,
                        "expected_outcome": result.test_case.expected_outcome,
                        "actual_rank": result.actual_rank,
                    },
                    recommendation="Not filtering losers - investigate risk scoring",
                ))

        # 3. Edge Case Tests
        edge_results = self._run_edge_tests(rank_lookup, current_snapshot)
        test_results.extend(edge_results)

        for result in edge_results:
            if not result.passed:
                flags.append(SanityFlag(
                    severity=FlagSeverity.HIGH,
                    category=CheckCategory.REGRESSION,
                    ticker=result.test_case.ticker,
                    check_name="edge_case_failed",
                    message=result.message,
                    details={
                        "description": result.test_case.description,
                        "expected_outcome": result.test_case.expected_outcome,
                        "actual_rank": result.actual_rank,
                    },
                    recommendation="Edge case handling issue",
                ))

        # Calculate metrics
        metrics = self._calculate_metrics(test_results)

        # Pass only if all tests pass
        passed = all(r.passed for r in test_results)

        return SanityCheckResult(
            check_name="regression_tests",
            category=CheckCategory.REGRESSION,
            passed=passed,
            flags=flags,
            metrics=metrics,
        )

    def _run_positive_tests(
        self,
        rank_lookup: Dict[str, int],
    ) -> List[RegressionTestResult]:
        """Run golden positive test cases."""
        results = []

        for test_case in self.golden_positive:
            actual_rank = rank_lookup.get(test_case.ticker)

            if actual_rank is None:
                # Ticker not in current universe
                result = RegressionTestResult(
                    test_case=test_case,
                    actual_rank=None,
                    passed=False,  # Consider not found as failure
                    message=f"{test_case.ticker} not found in current universe",
                )
            elif actual_rank <= test_case.threshold_rank:
                result = RegressionTestResult(
                    test_case=test_case,
                    actual_rank=actual_rank,
                    passed=True,
                    message=f"{test_case.ticker} correctly ranked #{actual_rank} (within top {test_case.threshold_rank})",
                )
            else:
                result = RegressionTestResult(
                    test_case=test_case,
                    actual_rank=actual_rank,
                    passed=False,
                    message=f"{test_case.ticker} ranked #{actual_rank}, expected top {test_case.threshold_rank}",
                )

            # Update test case with actual outcome
            test_case.actual_outcome = f"rank_{actual_rank}" if actual_rank else "not_found"
            test_case.passed = result.passed

            results.append(result)

        return results

    def _run_negative_tests(
        self,
        rank_lookup: Dict[str, int],
        universe_size: int,
    ) -> List[RegressionTestResult]:
        """Run golden negative test cases."""
        results = []
        median_rank = universe_size // 2

        for test_case in self.golden_negative:
            actual_rank = rank_lookup.get(test_case.ticker)

            if actual_rank is None:
                # Not found is good for negative cases
                result = RegressionTestResult(
                    test_case=test_case,
                    actual_rank=None,
                    passed=True,
                    message=f"{test_case.ticker} correctly excluded from universe",
                )
            elif actual_rank > median_rank:
                # Ranked in bottom half is good
                result = RegressionTestResult(
                    test_case=test_case,
                    actual_rank=actual_rank,
                    passed=True,
                    message=f"{test_case.ticker} correctly ranked #{actual_rank} (bottom half)",
                )
            else:
                # Ranked too highly is bad
                result = RegressionTestResult(
                    test_case=test_case,
                    actual_rank=actual_rank,
                    passed=False,
                    message=f"{test_case.ticker} ranked #{actual_rank}, should be bottom 50%",
                )

            test_case.actual_outcome = f"rank_{actual_rank}" if actual_rank else "excluded"
            test_case.passed = result.passed

            results.append(result)

        return results

    def _run_edge_tests(
        self,
        rank_lookup: Dict[str, int],
        snapshot: RankingSnapshot,
    ) -> List[RegressionTestResult]:
        """Run edge case tests based on security characteristics."""
        results = []

        # Test: Pre-revenue, pre-Phase 2, micro-cap should not rank top 100
        for sec in snapshot.securities:
            if (sec.lead_phase in ("Preclinical", "Phase 1") and
                sec.is_micro_cap and
                sec.rank is not None and
                sec.rank <= 20):

                test_case = GoldenTestCase(
                    ticker=sec.ticker,
                    as_of_date=snapshot.as_of_date,
                    expected_outcome="below_100",
                    case_type="edge",
                    description=f"Pre-Phase 2 micro-cap {sec.ticker} should not rank top 20",
                    threshold_rank=20,
                )

                result = RegressionTestResult(
                    test_case=test_case,
                    actual_rank=sec.rank,
                    passed=False,
                    message=f"Pre-Phase 2 micro-cap {sec.ticker} ranked #{sec.rank}",
                )
                results.append(result)

        # Test: Cash runway <3 months should have max penalty
        for sec in snapshot.securities:
            if (sec.runway_months is not None and
                sec.runway_months < Decimal("3") and
                sec.rank is not None and
                sec.rank <= 10):

                test_case = GoldenTestCase(
                    ticker=sec.ticker,
                    as_of_date=snapshot.as_of_date,
                    expected_outcome="below_50",
                    case_type="edge",
                    description=f"Near-zero runway {sec.ticker} should have max dilution penalty",
                    threshold_rank=10,
                )

                result = RegressionTestResult(
                    test_case=test_case,
                    actual_rank=sec.rank,
                    passed=False,
                    message=f"Near-zero runway ({sec.runway_months:.1f} months) {sec.ticker} ranked #{sec.rank}",
                )
                results.append(result)

        # Test: Zero institutional ownership should have conviction penalty
        for sec in snapshot.securities:
            if (sec.total_13f_holders is not None and
                sec.total_13f_holders == 0 and
                sec.rank is not None and
                sec.rank <= 20):

                test_case = GoldenTestCase(
                    ticker=sec.ticker,
                    as_of_date=snapshot.as_of_date,
                    expected_outcome="below_50",
                    case_type="edge",
                    description=f"Zero institutional ownership {sec.ticker} should have conviction penalty",
                    threshold_rank=20,
                )

                result = RegressionTestResult(
                    test_case=test_case,
                    actual_rank=sec.rank,
                    passed=False,
                    message=f"Zero 13F holders {sec.ticker} ranked #{sec.rank}",
                )
                results.append(result)

        return results

    def _calculate_metrics(
        self,
        results: List[RegressionTestResult],
    ) -> Dict[str, Any]:
        """Calculate test metrics."""
        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed)

        by_type = {
            "positive": {
                "total": sum(1 for r in results if r.test_case.case_type == "positive"),
                "passed": sum(1 for r in results if r.test_case.case_type == "positive" and r.passed),
            },
            "negative": {
                "total": sum(1 for r in results if r.test_case.case_type == "negative"),
                "passed": sum(1 for r in results if r.test_case.case_type == "negative" and r.passed),
            },
            "edge": {
                "total": sum(1 for r in results if r.test_case.case_type == "edge"),
                "passed": sum(1 for r in results if r.test_case.case_type == "edge" and r.passed),
            },
        }

        return {
            "total_tests": len(results),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(results) if results else 0,
            "by_type": by_type,
            "all_passed": failed == 0,
        }


def add_golden_test_case(
    ticker: str,
    as_of_date: str,
    expected_outcome: str,
    case_type: str,
    description: str,
    threshold_rank: Optional[int] = None,
) -> GoldenTestCase:
    """
    Factory function to create a golden test case.

    Args:
        ticker: Stock ticker
        as_of_date: Reference date for the test
        expected_outcome: Expected outcome (e.g., "top_20", "bottom_50")
        case_type: "positive", "negative", or "edge"
        description: Human-readable description
        threshold_rank: Rank threshold for the test

    Returns:
        GoldenTestCase instance
    """
    return GoldenTestCase(
        ticker=ticker,
        as_of_date=as_of_date,
        expected_outcome=expected_outcome,
        case_type=case_type,
        description=description,
        threshold_rank=threshold_rank,
    )


def generate_historical_validation_report(
    test_results: List[RegressionTestResult],
) -> Dict[str, Any]:
    """
    Generate comprehensive historical validation report.

    Args:
        test_results: List of regression test results

    Returns:
        Detailed report dict
    """
    report = {
        "summary": {
            "total_tests": len(test_results),
            "passed": sum(1 for r in test_results if r.passed),
            "failed": sum(1 for r in test_results if not r.passed),
        },
        "positive_tests": [],
        "negative_tests": [],
        "edge_tests": [],
        "failures": [],
    }

    for result in test_results:
        test_entry = {
            "ticker": result.test_case.ticker,
            "description": result.test_case.description,
            "expected_outcome": result.test_case.expected_outcome,
            "actual_rank": result.actual_rank,
            "passed": result.passed,
            "message": result.message,
        }

        if result.test_case.case_type == "positive":
            report["positive_tests"].append(test_entry)
        elif result.test_case.case_type == "negative":
            report["negative_tests"].append(test_entry)
        else:
            report["edge_tests"].append(test_entry)

        if not result.passed:
            report["failures"].append(test_entry)

    # Calculate pass rates
    report["pass_rates"] = {
        "overall": report["summary"]["passed"] / report["summary"]["total_tests"]
        if report["summary"]["total_tests"] > 0 else 0,
        "positive": sum(1 for t in report["positive_tests"] if t["passed"]) / len(report["positive_tests"])
        if report["positive_tests"] else 0,
        "negative": sum(1 for t in report["negative_tests"] if t["passed"]) / len(report["negative_tests"])
        if report["negative_tests"] else 0,
    }

    return report
