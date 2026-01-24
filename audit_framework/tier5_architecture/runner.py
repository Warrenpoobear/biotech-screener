"""
Tier 5 Audit Runner - Architecture & Code Quality.

Orchestrates all Tier 5 validators:
- Query 5.1: Architecture Review for Maintainability
- Query 5.2: Security & Access Control
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

from audit_framework.tier5_architecture.maintainability import (
    validate_maintainability,
)
from audit_framework.tier5_architecture.security import (
    validate_security,
)


@dataclass
class Tier5Result(TierResult):
    """Extended result for Tier 5 with specific metrics."""

    maintainability_passed: bool = False
    security_passed: bool = False

    maintainability_score: int = 0
    security_score: int = 0
    critical_vulnerabilities: int = 0


def run_tier5_audit(
    codebase_path: str,
    criteria: Optional[PassCriteria] = None,
) -> Tier5Result:
    """
    Run complete Tier 5 audit.

    Args:
        codebase_path: Root of codebase
        criteria: Pass/fail criteria

    Returns:
        Tier5Result with findings and metrics
    """
    if criteria is None:
        criteria = PassCriteria()

    start_time = time.time()

    # Initialize result
    result = Tier5Result(
        tier=AuditTier.TIER_5_ARCHITECTURE,
        grade=ComplianceGrade.F,
        passed=False,
        metrics=AuditMetrics(),
    )

    # Run Query 5.1: Maintainability
    maint_result = validate_maintainability(codebase_path)
    result.findings.extend(maint_result.findings)
    result.maintainability_passed = maint_result.passed
    result.maintainability_score = maint_result.metrics.get("maintainability_score", 0)

    # Run Query 5.2: Security
    sec_result = validate_security(codebase_path)
    result.findings.extend(sec_result.findings)
    result.security_passed = sec_result.passed
    result.security_score = sec_result.metrics.get("security_score", 0)
    result.critical_vulnerabilities = sec_result.metrics.get("critical_count", 0)
    result.metrics.critical_vulnerabilities = result.critical_vulnerabilities

    # Calculate execution time
    execution_time = Decimal(str(time.time() - start_time))
    result.execution_time_seconds = execution_time.quantize(Decimal("0.001"))

    # Determine overall pass/fail
    result.passed = (
        result.maintainability_passed
        and result.security_passed
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
    result.summary = _generate_tier5_summary(result)

    return result


def _generate_tier5_summary(result: Tier5Result) -> str:
    """Generate executive summary for Tier 5 results."""
    lines = [
        "## Tier 5: Architecture & Code Quality",
        "",
        f"**Overall Status:** {'PASSED' if result.passed else 'FAILED'}",
        f"**Grade:** {result.grade.value}",
        "",
        "### Component Results:",
        "",
        f"- Maintainability: {'PASSED' if result.maintainability_passed else 'FAILED'}",
        f"  - Score: {result.maintainability_score}/100",
        "",
        f"- Security: {'PASSED' if result.security_passed else 'FAILED'}",
        f"  - Score: {result.security_score}/100",
        f"  - Critical vulnerabilities: {result.critical_vulnerabilities}",
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
