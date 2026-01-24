"""
Tier 1 Audit Runner - Determinism & Reproducibility Validation.

Orchestrates all Tier 1 validators:
- Query 1.1: Decimal Arithmetic Compliance
- Query 1.2: Reproducibility Stress Test
- Query 1.3: Point-in-Time Data Integrity
"""

import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from audit_framework.types import (
    AuditMetrics,
    AuditResult,
    AuditSeverity,
    AuditTier,
    ComplianceGrade,
    PassCriteria,
    TierResult,
)

from audit_framework.tier1_determinism.decimal_compliance import (
    validate_decimal_compliance,
)
from audit_framework.tier1_determinism.reproducibility import (
    run_reproducibility_stress_test,
)
from audit_framework.tier1_determinism.pit_integrity import (
    validate_pit_integrity,
)


@dataclass
class Tier1Result(TierResult):
    """Extended result for Tier 1 with specific metrics."""

    decimal_compliance_passed: bool = False
    reproducibility_passed: bool = False
    pit_integrity_passed: bool = False

    float_operations_count: int = 0
    non_deterministic_sources: int = 0
    pit_violations: int = 0


def run_tier1_audit(
    codebase_path: str,
    criteria: Optional[PassCriteria] = None,
    run_stress_test: bool = False,
    as_of_date: Optional[str] = None,
) -> Tier1Result:
    """
    Run complete Tier 1 audit.

    Args:
        codebase_path: Root of codebase
        criteria: Pass/fail criteria
        run_stress_test: Whether to run full pipeline stress test
        as_of_date: Date to use for stress test

    Returns:
        Tier1Result with findings and metrics
    """
    if criteria is None:
        criteria = PassCriteria()

    start_time = time.time()

    # Initialize result
    result = Tier1Result(
        tier=AuditTier.TIER_1_DETERMINISM,
        grade=ComplianceGrade.F,
        passed=False,
        metrics=AuditMetrics(),
    )

    # Run Query 1.1: Decimal Compliance
    decimal_result = validate_decimal_compliance(codebase_path)
    result.findings.extend(decimal_result.findings)
    result.decimal_compliance_passed = decimal_result.passed
    result.float_operations_count = decimal_result.metrics.get("float_operations_count", 0)
    result.metrics.float_operations_count = result.float_operations_count

    # Run Query 1.2: Reproducibility (scan only, not full stress test)
    repro_result = run_reproducibility_stress_test(
        codebase_path=codebase_path,
        num_runs=criteria.determinism_runs_required if run_stress_test else 0,
        as_of_date=as_of_date,
    )
    result.findings.extend(repro_result.findings)
    result.reproducibility_passed = repro_result.passed
    result.non_deterministic_sources = repro_result.metrics.get(
        "non_deterministic_sources_count", 0
    )
    result.metrics.datetime_now_usages = result.non_deterministic_sources

    # Run Query 1.3: PIT Integrity
    pit_result = validate_pit_integrity(codebase_path)
    result.findings.extend(pit_result.findings)
    result.pit_integrity_passed = pit_result.passed
    result.pit_violations = pit_result.metrics.get("violations_count", 0)

    # Calculate execution time
    execution_time = Decimal(str(time.time() - start_time))
    result.execution_time_seconds = execution_time.quantize(Decimal("0.001"))

    # Determine overall pass/fail and grade
    result.passed = (
        result.decimal_compliance_passed
        and result.reproducibility_passed
        and result.pit_integrity_passed
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
    result.summary = _generate_tier1_summary(result)

    return result


def _generate_tier1_summary(result: Tier1Result) -> str:
    """Generate executive summary for Tier 1 results."""
    lines = [
        "## Tier 1: Determinism & Reproducibility Validation",
        "",
        f"**Overall Status:** {'PASSED' if result.passed else 'FAILED'}",
        f"**Grade:** {result.grade.value}",
        "",
        "### Component Results:",
        "",
        f"- Decimal Arithmetic Compliance: {'PASSED' if result.decimal_compliance_passed else 'FAILED'}",
        f"  - Float operations detected: {result.float_operations_count}",
        "",
        f"- Reproducibility Scan: {'PASSED' if result.reproducibility_passed else 'FAILED'}",
        f"  - Non-deterministic sources: {result.non_deterministic_sources}",
        "",
        f"- Point-in-Time Integrity: {'PASSED' if result.pit_integrity_passed else 'FAILED'}",
        f"  - PIT violations: {result.pit_violations}",
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
