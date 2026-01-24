"""
Tier 3 Audit Runner - Performance, Scalability & Operational Robustness.

Orchestrates all Tier 3 validators:
- Query 3.1: Performance Profiling
- Query 3.2: Error Handling & Resilience
- Query 3.3: Dependency & Supply Chain Security
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

from audit_framework.tier3_performance.profiling import (
    validate_performance,
)
from audit_framework.tier3_performance.resilience import (
    validate_resilience,
)
from audit_framework.tier3_performance.dependencies import (
    validate_dependencies,
)


@dataclass
class Tier3Result(TierResult):
    """Extended result for Tier 3 with specific metrics."""

    performance_passed: bool = False
    resilience_passed: bool = False
    dependencies_passed: bool = False

    resilience_score: int = 0
    external_deps_count: int = 0
    stdlib_only: bool = False


def run_tier3_audit(
    codebase_path: str,
    data_dir: str = "production_data",
    criteria: Optional[PassCriteria] = None,
) -> Tier3Result:
    """
    Run complete Tier 3 audit.

    Args:
        codebase_path: Root of codebase
        data_dir: Data directory name
        criteria: Pass/fail criteria

    Returns:
        Tier3Result with findings and metrics
    """
    if criteria is None:
        criteria = PassCriteria()

    start_time = time.time()

    # Initialize result
    result = Tier3Result(
        tier=AuditTier.TIER_3_PERFORMANCE,
        grade=ComplianceGrade.F,
        passed=False,
        metrics=AuditMetrics(),
    )

    # Run Query 3.1: Performance Profiling
    perf_result = validate_performance(codebase_path, data_dir)
    result.findings.extend(perf_result.findings)
    result.performance_passed = perf_result.passed

    # Run Query 3.2: Resilience
    res_result = validate_resilience(codebase_path)
    result.findings.extend(res_result.findings)
    result.resilience_passed = res_result.passed
    result.resilience_score = res_result.metrics.get("resilience_score", 0)

    # Run Query 3.3: Dependencies
    dep_result = validate_dependencies(codebase_path)
    result.findings.extend(dep_result.findings)
    result.dependencies_passed = dep_result.passed
    result.external_deps_count = dep_result.metrics.get("external_count", 0)
    result.stdlib_only = dep_result.metrics.get("stdlib_only", False)

    # Calculate execution time
    execution_time = Decimal(str(time.time() - start_time))
    result.execution_time_seconds = execution_time.quantize(Decimal("0.001"))

    # Determine overall pass/fail
    result.passed = (
        result.performance_passed
        and result.resilience_passed
        and result.dependencies_passed
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
    result.summary = _generate_tier3_summary(result)

    return result


def _generate_tier3_summary(result: Tier3Result) -> str:
    """Generate executive summary for Tier 3 results."""
    lines = [
        "## Tier 3: Performance, Scalability & Operational Robustness",
        "",
        f"**Overall Status:** {'PASSED' if result.passed else 'FAILED'}",
        f"**Grade:** {result.grade.value}",
        "",
        "### Component Results:",
        "",
        f"- Performance Profiling: {'PASSED' if result.performance_passed else 'FAILED'}",
        "",
        f"- Resilience: {'PASSED' if result.resilience_passed else 'FAILED'}",
        f"  - Score: {result.resilience_score}/100",
        "",
        f"- Dependencies: {'PASSED' if result.dependencies_passed else 'FAILED'}",
        f"  - External deps: {result.external_deps_count}",
        f"  - Stdlib only: {result.stdlib_only}",
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
