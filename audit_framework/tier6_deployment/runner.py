"""
Tier 6 Audit Runner - Deployment & Operational Readiness.

Orchestrates all Tier 6 validators:
- Query 6.1: Production Deployment Checklist
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

from audit_framework.tier6_deployment.readiness import (
    validate_deployment_readiness,
)


@dataclass
class Tier6Result(TierResult):
    """Extended result for Tier 6 with specific metrics."""

    deployment_passed: bool = False
    readiness_score: int = 0
    items_passed: int = 0
    items_total: int = 0


def run_tier6_audit(
    codebase_path: str,
    criteria: Optional[PassCriteria] = None,
) -> Tier6Result:
    """
    Run complete Tier 6 audit.

    Args:
        codebase_path: Root of codebase
        criteria: Pass/fail criteria

    Returns:
        Tier6Result with findings and metrics
    """
    if criteria is None:
        criteria = PassCriteria()

    start_time = time.time()

    # Initialize result
    result = Tier6Result(
        tier=AuditTier.TIER_6_DEPLOYMENT,
        grade=ComplianceGrade.F,
        passed=False,
        metrics=AuditMetrics(),
    )

    # Run Query 6.1: Deployment Readiness
    deploy_result = validate_deployment_readiness(codebase_path)
    result.findings.extend(deploy_result.findings)
    result.deployment_passed = deploy_result.passed
    result.readiness_score = deploy_result.metrics.get("readiness_score", 0)
    result.items_passed = deploy_result.metrics.get("items_passed", 0)
    result.items_total = deploy_result.metrics.get("items_total", 0)

    # Calculate execution time
    execution_time = Decimal(str(time.time() - start_time))
    result.execution_time_seconds = execution_time.quantize(Decimal("0.001"))

    # Determine overall pass/fail
    result.passed = result.deployment_passed

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
    result.summary = _generate_tier6_summary(result)

    return result


def _generate_tier6_summary(result: Tier6Result) -> str:
    """Generate executive summary for Tier 6 results."""
    lines = [
        "## Tier 6: Deployment & Operational Readiness",
        "",
        f"**Overall Status:** {'PASSED' if result.passed else 'FAILED'}",
        f"**Grade:** {result.grade.value}",
        "",
        "### Component Results:",
        "",
        f"- Deployment Readiness: {'PASSED' if result.deployment_passed else 'FAILED'}",
        f"  - Score: {result.readiness_score}%",
        f"  - Items: {result.items_passed}/{result.items_total} passed",
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
