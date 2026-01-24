"""
Audit Orchestrator - Central coordination for institutional-grade audits.

Provides:
- Full audit execution across all tiers
- Individual tier execution
- Report generation
- Compliance grading
"""

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

from audit_framework.types import (
    AuditReport,
    AuditSeverity,
    AuditTier,
    ComplianceGrade,
    PassCriteria,
    TierResult,
)

from audit_framework.tier1_determinism.runner import run_tier1_audit
from audit_framework.tier2_data_integrity.runner import run_tier2_audit
from audit_framework.tier3_performance.runner import run_tier3_audit
from audit_framework.tier4_testing.runner import run_tier4_audit
from audit_framework.tier5_architecture.runner import run_tier5_audit
from audit_framework.tier6_deployment.runner import run_tier6_audit


class AuditOrchestrator:
    """
    Central orchestrator for institutional-grade technical audits.

    Coordinates execution of all audit tiers and generates comprehensive
    compliance reports suitable for Big 4 audit or CTO review.

    Usage:
        orchestrator = AuditOrchestrator("/path/to/codebase")
        report = orchestrator.run_full_audit()
        report.generate_markdown()  # For documentation
    """

    FRAMEWORK_VERSION = "1.0.0"

    def __init__(
        self,
        codebase_path: str,
        data_dir: str = "production_data",
        criteria: Optional[PassCriteria] = None,
        as_of_date: Optional[str] = None,
    ):
        """
        Initialize the audit orchestrator.

        Args:
            codebase_path: Root path of the codebase to audit
            data_dir: Data directory name (relative to codebase)
            criteria: Custom pass/fail criteria
            as_of_date: Reference date for point-in-time checks
        """
        self.codebase_path = Path(codebase_path)
        self.data_dir = data_dir
        self.criteria = criteria or PassCriteria()
        self.as_of_date = as_of_date or datetime.now().date().isoformat()

    def _generate_report_id(self) -> str:
        """Generate unique report ID."""
        timestamp = datetime.now().isoformat()
        content = f"{self.codebase_path}:{timestamp}:{self.FRAMEWORK_VERSION}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _get_codebase_version(self) -> str:
        """Get codebase version from git or pyproject.toml."""
        # Try git
        git_dir = self.codebase_path / ".git"
        if git_dir.exists():
            head_file = git_dir / "HEAD"
            if head_file.exists():
                try:
                    with open(head_file, "r") as f:
                        ref = f.read().strip()
                    if ref.startswith("ref:"):
                        ref_path = git_dir / ref.split(": ")[1]
                        if ref_path.exists():
                            with open(ref_path, "r") as f:
                                return f.read().strip()[:8]
                except Exception:
                    pass

        # Try pyproject.toml
        pyproject = self.codebase_path / "pyproject.toml"
        if pyproject.exists():
            try:
                with open(pyproject, "r") as f:
                    content = f.read()
                import re
                match = re.search(r'version\s*=\s*"([^"]+)"', content)
                if match:
                    return match.group(1)
            except Exception:
                pass

        return "unknown"

    def _calculate_overall_grade(
        self,
        tier_results: List[TierResult],
    ) -> ComplianceGrade:
        """Calculate overall compliance grade from tier results."""
        # Count severity across all tiers
        total_critical = sum(tr.critical_count for tr in tier_results)
        total_high = sum(tr.high_count for tr in tier_results)

        # Count passed tiers
        passed_tiers = sum(1 for tr in tier_results if tr.passed)
        total_tiers = len(tier_results)

        # Grade logic
        if passed_tiers == total_tiers and total_critical == 0 and total_high == 0:
            return ComplianceGrade.A
        elif passed_tiers == total_tiers and total_critical == 0:
            return ComplianceGrade.B
        elif passed_tiers >= total_tiers * 0.7 and total_critical == 0:
            return ComplianceGrade.C
        elif total_critical <= 3:
            return ComplianceGrade.D
        else:
            return ComplianceGrade.F

    def _generate_executive_summary(
        self,
        tier_results: List[TierResult],
        overall_grade: ComplianceGrade,
    ) -> str:
        """Generate executive summary for the report."""
        passed_count = sum(1 for tr in tier_results if tr.passed)
        total_findings = sum(len(tr.findings) for tr in tier_results)
        critical_count = sum(tr.critical_count for tr in tier_results)
        high_count = sum(tr.high_count for tr in tier_results)

        lines = [
            "This institutional-grade technical audit evaluates Wake Robin Capital Management's",
            "biotech-screener system across six compliance tiers covering determinism, data",
            "integrity, performance, testing, architecture, and deployment readiness.",
            "",
            f"**Overall Grade: {overall_grade.value}**",
            "",
            f"- Tiers Passed: {passed_count}/{len(tier_results)}",
            f"- Total Findings: {total_findings}",
            f"- Critical Issues: {critical_count}",
            f"- High Priority Issues: {high_count}",
            "",
        ]

        if overall_grade == ComplianceGrade.A:
            lines.append(
                "The system demonstrates institutional-grade quality suitable for "
                "Big 4 audit review and production deployment."
            )
        elif overall_grade == ComplianceGrade.B:
            lines.append(
                "The system is near institutional-grade with minor issues requiring "
                "remediation before full production deployment."
            )
        elif overall_grade == ComplianceGrade.C:
            lines.append(
                "The system is acceptable but has documented gaps that should be "
                "addressed according to the remediation recommendations."
            )
        else:
            lines.append(
                "The system requires significant improvements before institutional "
                "deployment. See critical findings for immediate action items."
            )

        return "\n".join(lines)

    def _generate_recommendations(
        self,
        tier_results: List[TierResult],
    ) -> List[str]:
        """Generate prioritized recommendations."""
        recommendations = []

        # Collect all critical and high findings
        critical_findings = []
        high_findings = []

        for tr in tier_results:
            for f in tr.findings:
                if f.severity == AuditSeverity.CRITICAL:
                    critical_findings.append(f)
                elif f.severity == AuditSeverity.HIGH:
                    high_findings.append(f)

        # Generate recommendations
        if critical_findings:
            recommendations.append(
                f"IMMEDIATE: Address {len(critical_findings)} critical findings "
                "before any production deployment"
            )

        if high_findings:
            recommendations.append(
                f"SHORT-TERM: Remediate {len(high_findings)} high-priority issues "
                "within the current sprint"
            )

        # Tier-specific recommendations
        for tr in tier_results:
            if not tr.passed:
                tier_name = tr.tier.value.replace("_", " ").title()
                recommendations.append(
                    f"REQUIRED: Complete {tier_name} remediation to achieve passing grade"
                )

        if not recommendations:
            recommendations.append(
                "MAINTENANCE: Continue regular audit cycles to maintain compliance"
            )

        return recommendations

    def run_tier(self, tier: AuditTier) -> TierResult:
        """
        Run a single audit tier.

        Args:
            tier: Which tier to execute

        Returns:
            TierResult with findings
        """
        if tier == AuditTier.TIER_1_DETERMINISM:
            return run_tier1_audit(
                str(self.codebase_path),
                self.criteria,
                run_stress_test=False,
                as_of_date=self.as_of_date,
            )
        elif tier == AuditTier.TIER_2_DATA_INTEGRITY:
            return run_tier2_audit(
                str(self.codebase_path),
                self.data_dir,
                self.criteria,
                self.as_of_date,
            )
        elif tier == AuditTier.TIER_3_PERFORMANCE:
            return run_tier3_audit(
                str(self.codebase_path),
                self.data_dir,
                self.criteria,
            )
        elif tier == AuditTier.TIER_4_TESTING:
            return run_tier4_audit(
                str(self.codebase_path),
                self.criteria,
            )
        elif tier == AuditTier.TIER_5_ARCHITECTURE:
            return run_tier5_audit(
                str(self.codebase_path),
                self.criteria,
            )
        elif tier == AuditTier.TIER_6_DEPLOYMENT:
            return run_tier6_audit(
                str(self.codebase_path),
                self.criteria,
            )
        else:
            raise ValueError(f"Unknown tier: {tier}")

    def run_full_audit(
        self,
        tiers: Optional[List[AuditTier]] = None,
    ) -> AuditReport:
        """
        Run complete audit across all (or specified) tiers.

        Args:
            tiers: List of tiers to run, or None for all

        Returns:
            Complete AuditReport
        """
        if tiers is None:
            tiers = list(AuditTier)

        tier_results: List[TierResult] = []

        for tier in tiers:
            result = self.run_tier(tier)
            tier_results.append(result)

        # Calculate overall grade
        overall_grade = self._calculate_overall_grade(tier_results)
        overall_passed = all(tr.passed for tr in tier_results)

        # Generate summaries
        executive_summary = self._generate_executive_summary(tier_results, overall_grade)
        recommendations = self._generate_recommendations(tier_results)

        return AuditReport(
            report_id=self._generate_report_id(),
            generated_at=datetime.now().isoformat(),
            codebase_version=self._get_codebase_version(),
            audit_framework_version=self.FRAMEWORK_VERSION,
            overall_grade=overall_grade,
            overall_passed=overall_passed,
            tier_results=tier_results,
            executive_summary=executive_summary,
            recommendations=recommendations,
            pass_criteria=self.criteria,
        )

    def save_report(
        self,
        report: AuditReport,
        output_path: Optional[str] = None,
        format: str = "json",
    ) -> str:
        """
        Save audit report to file.

        Args:
            report: The report to save
            output_path: Output file path (default: auto-generated)
            format: "json" or "markdown"

        Returns:
            Path to saved file
        """
        if output_path is None:
            reports_dir = self.codebase_path / "audit_framework" / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = "json" if format == "json" else "md"
            output_path = str(reports_dir / f"audit_report_{timestamp}.{ext}")

        output_file = Path(output_path)

        if format == "json":
            with open(output_file, "w") as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
        else:
            with open(output_file, "w") as f:
                f.write(report.generate_markdown())

        return str(output_file)


def run_full_audit(
    codebase_path: str,
    data_dir: str = "production_data",
    criteria: Optional[PassCriteria] = None,
    as_of_date: Optional[str] = None,
) -> AuditReport:
    """
    Convenience function to run full audit.

    Args:
        codebase_path: Root of codebase
        data_dir: Data directory name
        criteria: Pass/fail criteria
        as_of_date: Reference date

    Returns:
        Complete AuditReport
    """
    orchestrator = AuditOrchestrator(
        codebase_path=codebase_path,
        data_dir=data_dir,
        criteria=criteria,
        as_of_date=as_of_date,
    )
    return orchestrator.run_full_audit()


def run_tier_audit(
    codebase_path: str,
    tier: AuditTier,
    data_dir: str = "production_data",
    criteria: Optional[PassCriteria] = None,
    as_of_date: Optional[str] = None,
) -> TierResult:
    """
    Convenience function to run single tier audit.

    Args:
        codebase_path: Root of codebase
        tier: Which tier to run
        data_dir: Data directory name
        criteria: Pass/fail criteria
        as_of_date: Reference date

    Returns:
        TierResult with findings
    """
    orchestrator = AuditOrchestrator(
        codebase_path=codebase_path,
        data_dir=data_dir,
        criteria=criteria,
        as_of_date=as_of_date,
    )
    return orchestrator.run_tier(tier)
