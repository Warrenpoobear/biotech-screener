"""
Tier 4 Audit Runner - Testing, Validation & Regression Prevention.

Orchestrates all Tier 4 validators:
- Query 4.1: Test Coverage & Regression Suite
- Query 4.2: Backtesting Validation Framework
- Query 4.3: Model Explainability & Audit Trail
"""

import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from audit_framework.types import (
    AuditMetrics,
    AuditTier,
    ComplianceGrade,
    PassCriteria,
    TierResult,
)

from audit_framework.tier4_testing.coverage import (
    validate_test_coverage,
)
from audit_framework.tier4_testing.backtesting import (
    validate_backtest_capability,
)
from audit_framework.tier4_testing.explainability import (
    validate_explainability,
)


@dataclass
class Tier4Result(TierResult):
    """Extended result for Tier 4 with specific metrics."""

    test_coverage_passed: bool = False
    backtest_passed: bool = False
    explainability_passed: bool = False

    total_tests: int = 0
    estimated_coverage: Decimal = Decimal("0")
    backtest_readiness: int = 0
    explainability_score: int = 0


def run_tier4_audit(
    codebase_path: str,
    criteria: Optional[PassCriteria] = None,
) -> Tier4Result:
    """
    Run complete Tier 4 audit.

    Args:
        codebase_path: Root of codebase
        criteria: Pass/fail criteria

    Returns:
        Tier4Result with findings and metrics
    """
    if criteria is None:
        criteria = PassCriteria()

    start_time = time.time()

    # Initialize result
    result = Tier4Result(
        tier=AuditTier.TIER_4_TESTING,
        grade=ComplianceGrade.F,
        passed=False,
        metrics=AuditMetrics(),
    )

    # Run Query 4.1: Test Coverage
    cov_result = validate_test_coverage(codebase_path)
    result.findings.extend(cov_result.findings)
    result.test_coverage_passed = cov_result.passed
    result.total_tests = cov_result.metrics.get("total_tests", 0)
    result.estimated_coverage = Decimal(
        cov_result.metrics.get("estimated_coverage", "0")
    )
    result.metrics.test_count = result.total_tests
    result.metrics.line_coverage = result.estimated_coverage

    # Run Query 4.2: Backtesting
    bt_result = validate_backtest_capability(codebase_path)
    result.findings.extend(bt_result.findings)
    result.backtest_passed = bt_result.passed
    result.backtest_readiness = bt_result.metrics.get("readiness_score", 0)

    # Run Query 4.3: Explainability
    exp_result = validate_explainability(codebase_path)
    result.findings.extend(exp_result.findings)
    result.explainability_passed = exp_result.passed
    result.explainability_score = exp_result.metrics.get("explainability_score", 0)

    # Calculate execution time
    execution_time = Decimal(str(time.time() - start_time))
    result.execution_time_seconds = execution_time.quantize(Decimal("0.001"))

    # Determine overall pass/fail
    result.passed = (
        result.test_coverage_passed
        and result.backtest_passed
        and result.explainability_passed
    )

    # Grade calculation
    critical_count = result.critical_count
    high_count = result.high_count

    if result.passed and critical_count == 0 and high_count == 0:
        result.grade = ComplianceGrade.A
    elif result.passed and critical_count == 0:
        result.grade = ComplianceGrade.B
    elif critical_count == 0:
        result.grade = ComplianceGrade.C
    elif critical_count <= 3:
        result.grade = ComplianceGrade.D
    else:
        result.grade = ComplianceGrade.F

    # Generate summary
    result.summary = _generate_tier4_summary(result)

    return result


def _generate_tier4_summary(result: Tier4Result) -> str:
    """Generate executive summary for Tier 4 results."""
    lines = [
        "## Tier 4: Testing, Validation & Regression Prevention",
        "",
        f"**Overall Status:** {'PASSED' if result.passed else 'FAILED'}",
        f"**Grade:** {result.grade.value}",
        "",
        "### Component Results:",
        "",
        f"- Test Coverage: {'PASSED' if result.test_coverage_passed else 'FAILED'}",
        f"  - Total tests: {result.total_tests}",
        f"  - Estimated coverage: {result.estimated_coverage*100:.1f}%",
        "",
        f"- Backtesting: {'PASSED' if result.backtest_passed else 'FAILED'}",
        f"  - Readiness score: {result.backtest_readiness}/100",
        "",
        f"- Explainability: {'PASSED' if result.explainability_passed else 'FAILED'}",
        f"  - Score: {result.explainability_score}/100",
        "",
        "### Findings Summary:",
        "",
        f"- Critical: {result.critical_count}",
        f"- High: {result.high_count}",
        f"- Total: {len(result.findings)}",
        "",
        f"Execution time: {result.execution_time_seconds}s",
    ]

    return "\n".join(lines)
