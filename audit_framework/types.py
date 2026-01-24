"""
Core types for the Institutional-Grade Audit Framework.

Provides type definitions, enums, and data structures for audit validation,
compliance grading, and institutional reporting.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class AuditSeverity(Enum):
    """Severity levels for audit findings."""

    CRITICAL = "critical"  # Blocks institutional deployment
    HIGH = "high"          # Requires remediation before production
    MEDIUM = "medium"      # Should be addressed within sprint
    LOW = "low"            # Improvement opportunity
    INFO = "info"          # Informational finding


class AuditTier(Enum):
    """Audit framework tiers matching institutional requirements."""

    TIER_1_DETERMINISM = "tier_1_determinism"
    TIER_2_DATA_INTEGRITY = "tier_2_data_integrity"
    TIER_3_PERFORMANCE = "tier_3_performance"
    TIER_4_TESTING = "tier_4_testing"
    TIER_5_ARCHITECTURE = "tier_5_architecture"
    TIER_6_DEPLOYMENT = "tier_6_deployment"


class ComplianceGrade(Enum):
    """
    Institutional compliance grading scale.

    Pass criteria for "Institutional Grade":
    - A: 100% determinism, 95%+ provenance, zero critical issues
    - B: 98%+ determinism, 90%+ provenance, critical issues documented
    - C: 95%+ determinism, 80%+ provenance, remediation plan exists
    - D: Below thresholds, significant remediation required
    - F: Fundamental issues blocking institutional use
    """

    A = "A"  # Institutional-grade, Big 4 audit ready
    B = "B"  # Near institutional-grade, minor remediation
    C = "C"  # Acceptable with documented gaps
    D = "D"  # Significant improvements needed
    F = "F"  # Not suitable for institutional use


class ValidationCategory(Enum):
    """Categories of validation checks."""

    DETERMINISM = "determinism"
    REPRODUCIBILITY = "reproducibility"
    PIT_SAFETY = "pit_safety"
    PROVENANCE = "provenance"
    DATA_QUALITY = "data_quality"
    COVERAGE = "coverage"
    PERFORMANCE = "performance"
    SCALABILITY = "scalability"
    ERROR_HANDLING = "error_handling"
    SECURITY = "security"
    TEST_COVERAGE = "test_coverage"
    ARCHITECTURE = "architecture"
    DOCUMENTATION = "documentation"
    DEPLOYMENT = "deployment"


@dataclass(frozen=True)
class ValidationFinding:
    """
    A single audit finding with full context for remediation.

    Attributes:
        finding_id: Unique identifier for tracking
        severity: Impact level of the finding
        category: Classification of the issue
        title: Brief description of the finding
        description: Detailed explanation
        location: File path and line number(s)
        evidence: Specific code or data demonstrating the issue
        remediation: Recommended fix
        compliance_impact: Regulatory or compliance implications
    """

    finding_id: str
    severity: AuditSeverity
    category: ValidationCategory
    title: str
    description: str
    location: str
    evidence: str
    remediation: str
    compliance_impact: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "finding_id": self.finding_id,
            "severity": self.severity.value,
            "category": self.category.value,
            "title": self.title,
            "description": self.description,
            "location": self.location,
            "evidence": self.evidence,
            "remediation": self.remediation,
            "compliance_impact": self.compliance_impact,
        }


@dataclass
class PassCriteria:
    """
    Institutional pass/fail criteria for audit tiers.

    These thresholds align with Big 4 audit and CTO review requirements.
    """

    # Tier 1: Determinism
    determinism_runs_required: int = 10
    determinism_variance_threshold: Decimal = Decimal("0")
    determinism_hash_identity_required: bool = True

    # Tier 2: Data Integrity
    provenance_coverage_min: Decimal = Decimal("0.95")  # 95%
    data_quality_score_min: Decimal = Decimal("0.90")   # 90%
    pit_compliance_min: Decimal = Decimal("1.0")        # 100%

    # Tier 3: Performance
    runtime_320_tickers_max_seconds: int = 300  # 5 minutes
    runtime_1000_tickers_max_seconds: int = 1800  # 30 minutes
    memory_leak_tolerance_mb: int = 100

    # Tier 4: Testing
    unit_test_coverage_min: Decimal = Decimal("0.80")  # 80%
    scoring_logic_coverage_min: Decimal = Decimal("0.90")  # 90%
    integration_test_count_min: int = 3

    # Tier 5: Architecture
    cyclomatic_complexity_max: int = 10
    lines_per_module_max: int = 500
    security_vulnerabilities_critical_max: int = 0

    # Tier 6: Deployment
    documentation_coverage_min: Decimal = Decimal("0.80")  # 80%
    monitoring_required: bool = True
    backup_strategy_required: bool = True


@dataclass
class AuditMetrics:
    """Quantitative metrics collected during audit."""

    # Determinism metrics
    total_runs: int = 0
    identical_hash_runs: int = 0
    score_variance: Decimal = Decimal("0")
    ranking_correlation: Decimal = Decimal("1")
    float_operations_count: int = 0
    datetime_now_usages: int = 0
    random_usages: int = 0

    # Data integrity metrics
    provenance_coverage: Decimal = Decimal("0")
    data_quality_score: Decimal = Decimal("0")
    pit_compliance_rate: Decimal = Decimal("0")
    stale_data_count: int = 0

    # Coverage metrics
    tickers_with_financials: int = 0
    tickers_with_pos: int = 0
    tickers_with_catalyst: int = 0
    tickers_with_13f: int = 0

    # Performance metrics
    runtime_seconds: Decimal = Decimal("0")
    peak_memory_mb: Decimal = Decimal("0")

    # Testing metrics
    test_count: int = 0
    test_pass_count: int = 0
    line_coverage: Decimal = Decimal("0")
    branch_coverage: Decimal = Decimal("0")

    # Security metrics
    secrets_exposed: int = 0
    critical_vulnerabilities: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with Decimal handling."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Decimal):
                result[key] = str(value)
            else:
                result[key] = value
        return result


@dataclass
class TierResult:
    """
    Result of a single tier's audit execution.

    Attributes:
        tier: Which audit tier was executed
        grade: Compliance grade achieved
        passed: Whether tier met institutional criteria
        findings: List of validation findings
        metrics: Quantitative metrics collected
        execution_time_seconds: How long the audit took
        summary: Executive summary of results
    """

    tier: AuditTier
    grade: ComplianceGrade
    passed: bool
    findings: List[ValidationFinding] = field(default_factory=list)
    metrics: AuditMetrics = field(default_factory=AuditMetrics)
    execution_time_seconds: Decimal = Decimal("0")
    summary: str = ""

    @property
    def critical_count(self) -> int:
        """Count of critical findings."""
        return sum(1 for f in self.findings if f.severity == AuditSeverity.CRITICAL)

    @property
    def high_count(self) -> int:
        """Count of high severity findings."""
        return sum(1 for f in self.findings if f.severity == AuditSeverity.HIGH)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "tier": self.tier.value,
            "grade": self.grade.value,
            "passed": self.passed,
            "findings": [f.to_dict() for f in self.findings],
            "metrics": self.metrics.to_dict(),
            "execution_time_seconds": str(self.execution_time_seconds),
            "summary": self.summary,
            "critical_count": self.critical_count,
            "high_count": self.high_count,
        }


@dataclass
class AuditResult:
    """
    Complete result from a validation check.

    Used by individual validators to report their findings.
    """

    check_name: str
    passed: bool
    findings: List[ValidationFinding] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    details: str = ""

    def add_finding(
        self,
        severity: AuditSeverity,
        category: ValidationCategory,
        title: str,
        description: str,
        location: str,
        evidence: str,
        remediation: str,
        compliance_impact: str = "",
    ) -> None:
        """Add a finding to this result."""
        finding_id = f"{self.check_name}_{len(self.findings) + 1:03d}"
        self.findings.append(ValidationFinding(
            finding_id=finding_id,
            severity=severity,
            category=category,
            title=title,
            description=description,
            location=location,
            evidence=evidence,
            remediation=remediation,
            compliance_impact=compliance_impact,
        ))


@dataclass
class AuditReport:
    """
    Complete institutional audit report.

    Suitable for investment committee review, Big 4 audit, or CTO assessment.
    """

    report_id: str
    generated_at: str  # ISO timestamp
    codebase_version: str
    audit_framework_version: str
    overall_grade: ComplianceGrade
    overall_passed: bool
    tier_results: List[TierResult] = field(default_factory=list)
    executive_summary: str = ""
    recommendations: List[str] = field(default_factory=list)

    # Pass criteria used
    pass_criteria: PassCriteria = field(default_factory=PassCriteria)

    @property
    def total_findings(self) -> int:
        """Total findings across all tiers."""
        return sum(len(tr.findings) for tr in self.tier_results)

    @property
    def critical_findings(self) -> List[ValidationFinding]:
        """All critical findings across tiers."""
        findings = []
        for tr in self.tier_results:
            findings.extend(f for f in tr.findings if f.severity == AuditSeverity.CRITICAL)
        return findings

    @property
    def high_findings(self) -> List[ValidationFinding]:
        """All high severity findings across tiers."""
        findings = []
        for tr in self.tier_results:
            findings.extend(f for f in tr.findings if f.severity == AuditSeverity.HIGH)
        return findings

    def get_tier_result(self, tier: AuditTier) -> Optional[TierResult]:
        """Get result for a specific tier."""
        for tr in self.tier_results:
            if tr.tier == tier:
                return tr
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "codebase_version": self.codebase_version,
            "audit_framework_version": self.audit_framework_version,
            "overall_grade": self.overall_grade.value,
            "overall_passed": self.overall_passed,
            "tier_results": [tr.to_dict() for tr in self.tier_results],
            "executive_summary": self.executive_summary,
            "recommendations": self.recommendations,
            "total_findings": self.total_findings,
            "critical_count": len(self.critical_findings),
            "high_count": len(self.high_findings),
        }

    def generate_markdown(self) -> str:
        """Generate markdown report suitable for documentation."""
        lines = [
            f"# Institutional Audit Report",
            f"",
            f"**Report ID:** {self.report_id}",
            f"**Generated:** {self.generated_at}",
            f"**Codebase Version:** {self.codebase_version}",
            f"**Overall Grade:** {self.overall_grade.value}",
            f"**Status:** {'PASSED' if self.overall_passed else 'FAILED'}",
            f"",
            f"## Executive Summary",
            f"",
            self.executive_summary,
            f"",
            f"## Tier Results",
            f"",
        ]

        for tr in self.tier_results:
            status = "✅ PASSED" if tr.passed else "❌ FAILED"
            lines.extend([
                f"### {tr.tier.value.replace('_', ' ').title()} - Grade: {tr.grade.value} {status}",
                f"",
                tr.summary,
                f"",
                f"- Critical Findings: {tr.critical_count}",
                f"- High Findings: {tr.high_count}",
                f"",
            ])

        if self.recommendations:
            lines.extend([
                f"## Recommendations",
                f"",
            ])
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        if self.critical_findings:
            lines.extend([
                f"## Critical Findings Requiring Immediate Attention",
                f"",
            ])
            for f in self.critical_findings:
                lines.extend([
                    f"### {f.finding_id}: {f.title}",
                    f"",
                    f"**Location:** {f.location}",
                    f"",
                    f"{f.description}",
                    f"",
                    f"**Evidence:**",
                    f"```",
                    f.evidence,
                    f"```",
                    f"",
                    f"**Remediation:** {f.remediation}",
                    f"",
                    f"**Compliance Impact:** {f.compliance_impact}",
                    f"",
                ])

        return "\n".join(lines)


# Convenience type aliases
FindingList = List[ValidationFinding]
MetricsDict = Dict[str, Any]
