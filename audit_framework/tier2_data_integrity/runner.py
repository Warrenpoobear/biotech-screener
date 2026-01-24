"""
Tier 2 Audit Runner - Data Integrity & Provenance.

Orchestrates all Tier 2 validators:
- Query 2.1: Provenance Lock Validation
- Query 2.2: Data Quality & Coverage Analysis
- Query 2.3: Failure Mode Catalog
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

from audit_framework.tier2_data_integrity.provenance import (
    validate_provenance,
)
from audit_framework.tier2_data_integrity.coverage import (
    validate_data_coverage,
)
from audit_framework.tier2_data_integrity.failure_modes import (
    validate_failure_modes,
)


@dataclass
class Tier2Result(TierResult):
    """Extended result for Tier 2 with specific metrics."""

    provenance_passed: bool = False
    coverage_passed: bool = False
    failure_modes_passed: bool = False

    provenance_coverage: Decimal = Decimal("0")
    overall_data_coverage: Decimal = Decimal("0")
    failure_mode_pass_rate: Decimal = Decimal("0")


def run_tier2_audit(
    codebase_path: str,
    data_dir: str = "production_data",
    criteria: Optional[PassCriteria] = None,
    as_of_date: Optional[str] = None,
) -> Tier2Result:
    """
    Run complete Tier 2 audit.

    Args:
        codebase_path: Root of codebase
        data_dir: Data directory name
        criteria: Pass/fail criteria
        as_of_date: Reference date

    Returns:
        Tier2Result with findings and metrics
    """
    if criteria is None:
        criteria = PassCriteria()

    start_time = time.time()

    # Initialize result
    result = Tier2Result(
        tier=AuditTier.TIER_2_DATA_INTEGRITY,
        grade=ComplianceGrade.F,
        passed=False,
        metrics=AuditMetrics(),
    )

    # Run Query 2.1: Provenance Validation
    prov_result = validate_provenance(codebase_path, data_dir)
    result.findings.extend(prov_result.findings)
    result.provenance_passed = prov_result.passed
    result.provenance_coverage = Decimal(
        prov_result.metrics.get("provenance_coverage", "0")
    )
    result.metrics.provenance_coverage = result.provenance_coverage

    # Run Query 2.2: Data Coverage
    cov_result = validate_data_coverage(codebase_path, data_dir, as_of_date)
    result.findings.extend(cov_result.findings)
    result.coverage_passed = cov_result.passed
    result.overall_data_coverage = Decimal(
        cov_result.metrics.get("overall_coverage", "0")
    )
    result.metrics.data_quality_score = result.overall_data_coverage

    # Run Query 2.3: Failure Modes
    failure_result = validate_failure_modes(codebase_path, data_dir)
    result.findings.extend(failure_result.findings)
    result.failure_modes_passed = failure_result.passed
    result.failure_mode_pass_rate = Decimal(
        failure_result.metrics.get("pass_rate", "0")
    )

    # Calculate execution time
    execution_time = Decimal(str(time.time() - start_time))
    result.execution_time_seconds = execution_time.quantize(Decimal("0.001"))

    # Determine overall pass/fail
    result.passed = (
        result.provenance_passed
        and result.coverage_passed
        and result.failure_modes_passed
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
    result.summary = _generate_tier2_summary(result)

    return result


def _generate_tier2_summary(result: Tier2Result) -> str:
    """Generate executive summary for Tier 2 results."""
    lines = [
        "## Tier 2: Data Integrity & Provenance",
        "",
        f"**Overall Status:** {'PASSED' if result.passed else 'FAILED'}",
        f"**Grade:** {result.grade.value}",
        "",
        "### Component Results:",
        "",
        f"- Provenance Validation: {'PASSED' if result.provenance_passed else 'FAILED'}",
        f"  - Coverage: {result.provenance_coverage*100:.1f}%",
        "",
        f"- Data Coverage: {'PASSED' if result.coverage_passed else 'FAILED'}",
        f"  - Overall: {result.overall_data_coverage*100:.1f}%",
        "",
        f"- Failure Mode Testing: {'PASSED' if result.failure_modes_passed else 'FAILED'}",
        f"  - Pass rate: {result.failure_mode_pass_rate*100:.1f}%",
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
