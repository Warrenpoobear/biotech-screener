"""
Query 6.1 - Production Deployment Checklist.

Validates system readiness for institutional production use.

Checks:
- Environment setup documentation
- Monitoring & alerting capability
- Disaster recovery planning
- Change management procedures
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from audit_framework.types import (
    AuditResult,
    AuditSeverity,
    ValidationCategory,
)


@dataclass
class ChecklistItem:
    """A single deployment checklist item."""

    category: str
    item: str
    passed: bool
    evidence: str = ""
    recommendation: str = ""


@dataclass
class DeploymentReport:
    """Complete deployment readiness report."""

    checklist: List[ChecklistItem] = field(default_factory=list)
    environment_ready: bool = False
    monitoring_ready: bool = False
    backup_ready: bool = False
    change_mgmt_ready: bool = False
    readiness_score: int = 0  # 0-100
    passed: bool = False

    @property
    def items_passed(self) -> int:
        return sum(1 for i in self.checklist if i.passed)

    @property
    def items_total(self) -> int:
        return len(self.checklist)


class DeploymentValidator:
    """
    Validates deployment and operational readiness.

    Checks:
    - Python version and dependencies documented
    - Installation scripts
    - Monitoring infrastructure
    - Backup and recovery procedures
    - Version control practices
    """

    # Required documentation files
    REQUIRED_DOCS: Dict[str, str] = {
        "README.md": "Project overview and setup",
        "CLAUDE.md": "AI assistant guide",
        "pyproject.toml": "Package configuration",
    }

    # Configuration files
    CONFIG_FILES: Dict[str, str] = {
        "config.yml": "Pipeline configuration",
        "pyproject.toml": "Dependencies",
    }

    def __init__(self, codebase_path: str):
        """Initialize validator."""
        self.codebase_path = Path(codebase_path)

    def check_environment_setup(self) -> List[ChecklistItem]:
        """Check environment setup documentation."""
        items = []

        # Check for pyproject.toml
        pyproject = self.codebase_path / "pyproject.toml"
        items.append(ChecklistItem(
            category="environment",
            item="pyproject.toml exists",
            passed=pyproject.exists(),
            evidence=str(pyproject) if pyproject.exists() else "",
            recommendation="Add pyproject.toml for package configuration",
        ))

        # Check for Python version specification
        if pyproject.exists():
            try:
                with open(pyproject, "r") as f:
                    content = f.read()
                has_python_version = "python" in content.lower()
                items.append(ChecklistItem(
                    category="environment",
                    item="Python version specified",
                    passed=has_python_version,
                    evidence="Found python version requirement" if has_python_version else "",
                    recommendation="Specify python version in pyproject.toml",
                ))
            except Exception:
                pass

        # Check for requirements.txt or dependencies
        has_deps = (
            (self.codebase_path / "requirements.txt").exists()
            or pyproject.exists()
        )
        items.append(ChecklistItem(
            category="environment",
            item="Dependencies documented",
            passed=has_deps,
            evidence="requirements.txt or pyproject.toml",
            recommendation="Add requirements.txt or pyproject.toml dependencies",
        ))

        # Check for setup instructions
        readme = self.codebase_path / "README.md"
        has_setup = False
        if readme.exists():
            try:
                with open(readme, "r") as f:
                    content = f.read().lower()
                has_setup = "install" in content or "setup" in content
            except Exception:
                pass

        items.append(ChecklistItem(
            category="environment",
            item="Installation instructions documented",
            passed=has_setup,
            evidence="Found in README.md" if has_setup else "",
            recommendation="Add installation instructions to README.md",
        ))

        return items

    def check_monitoring(self) -> List[ChecklistItem]:
        """Check monitoring and alerting capability."""
        items = []

        # Check for logging infrastructure
        has_logging = False
        logging_patterns = [r"logging\.", r"logger\.", r"log\."]

        for root, _, files in os.walk(self.codebase_path):
            if any(skip in root for skip in ["test", "deprecated", "venv"]):
                continue

            for file in files:
                if not file.endswith(".py"):
                    continue

                try:
                    with open(Path(root) / file, "r") as f:
                        content = f.read()
                    if any(re.search(p, content) for p in logging_patterns):
                        has_logging = True
                        break
                except Exception:
                    continue
            if has_logging:
                break

        items.append(ChecklistItem(
            category="monitoring",
            item="Logging infrastructure present",
            passed=has_logging,
            evidence="logging module usage found" if has_logging else "",
            recommendation="Implement structured logging throughout",
        ))

        # Check for health check endpoints or monitoring
        has_health = False
        health_patterns = [r"health", r"status", r"heartbeat", r"monitor"]

        for root, _, files in os.walk(self.codebase_path):
            if any(skip in root for skip in ["test", "deprecated", "venv"]):
                continue

            for file in files:
                if any(re.search(p, file, re.IGNORECASE) for p in health_patterns):
                    has_health = True
                    break
            if has_health:
                break

        items.append(ChecklistItem(
            category="monitoring",
            item="Health monitoring capability",
            passed=has_health,
            evidence="Health check file found" if has_health else "",
            recommendation="Add health check endpoint or monitoring script",
        ))

        # Check for audit logging
        audit_log = self.codebase_path / "governance" / "audit_log.py"
        items.append(ChecklistItem(
            category="monitoring",
            item="Audit logging infrastructure",
            passed=audit_log.exists(),
            evidence=str(audit_log) if audit_log.exists() else "",
            recommendation="Implement audit logging for compliance",
        ))

        return items

    def check_disaster_recovery(self) -> List[ChecklistItem]:
        """Check disaster recovery planning."""
        items = []

        # Check for data directory
        data_dir = self.codebase_path / "production_data"
        items.append(ChecklistItem(
            category="backup",
            item="Production data directory exists",
            passed=data_dir.exists(),
            evidence=str(data_dir) if data_dir.exists() else "",
            recommendation="Create production_data directory for data storage",
        ))

        # Check for state management
        state_mgmt = self.codebase_path / "state_management.py"
        items.append(ChecklistItem(
            category="backup",
            item="State management implemented",
            passed=state_mgmt.exists(),
            evidence=str(state_mgmt) if state_mgmt.exists() else "",
            recommendation="Implement state management for recovery",
        ))

        # Check for output regeneration capability
        run_screen = self.codebase_path / "run_screen.py"
        items.append(ChecklistItem(
            category="backup",
            item="Output regeneration capability",
            passed=run_screen.exists(),
            evidence=str(run_screen) if run_screen.exists() else "",
            recommendation="Ensure outputs can be regenerated from inputs",
        ))

        return items

    def check_change_management(self) -> List[ChecklistItem]:
        """Check change management procedures."""
        items = []

        # Check for git
        git_dir = self.codebase_path / ".git"
        items.append(ChecklistItem(
            category="change_mgmt",
            item="Version control (Git) used",
            passed=git_dir.exists(),
            evidence="Git repository found" if git_dir.exists() else "",
            recommendation="Initialize git repository for version control",
        ))

        # Check for tests directory
        tests_dir = self.codebase_path / "tests"
        items.append(ChecklistItem(
            category="change_mgmt",
            item="Test suite present",
            passed=tests_dir.exists() and any(tests_dir.glob("test_*.py")),
            evidence=str(tests_dir) if tests_dir.exists() else "",
            recommendation="Add test suite for regression prevention",
        ))

        # Check for CI/CD configuration
        ci_files = [
            ".github/workflows",
            ".gitlab-ci.yml",
            "Jenkinsfile",
            ".circleci/config.yml",
        ]
        has_ci = any((self.codebase_path / ci).exists() for ci in ci_files)
        items.append(ChecklistItem(
            category="change_mgmt",
            item="CI/CD configuration present",
            passed=has_ci,
            evidence="CI/CD config found" if has_ci else "",
            recommendation="Add CI/CD pipeline for automated testing",
        ))

        # Check for schema versioning
        schema_registry = self.codebase_path / "governance" / "schema_registry.py"
        items.append(ChecklistItem(
            category="change_mgmt",
            item="Schema versioning implemented",
            passed=schema_registry.exists(),
            evidence=str(schema_registry) if schema_registry.exists() else "",
            recommendation="Implement schema versioning for backwards compatibility",
        ))

        return items

    def run_audit(self) -> DeploymentReport:
        """
        Run complete deployment readiness audit.

        Returns:
            DeploymentReport with findings
        """
        all_items: List[ChecklistItem] = []

        # Run all checks
        env_items = self.check_environment_setup()
        all_items.extend(env_items)
        environment_ready = all(i.passed for i in env_items)

        monitoring_items = self.check_monitoring()
        all_items.extend(monitoring_items)
        monitoring_ready = sum(1 for i in monitoring_items if i.passed) >= 2

        backup_items = self.check_disaster_recovery()
        all_items.extend(backup_items)
        backup_ready = all(i.passed for i in backup_items)

        change_items = self.check_change_management()
        all_items.extend(change_items)
        change_mgmt_ready = sum(1 for i in change_items if i.passed) >= 2

        # Calculate score
        passed_count = sum(1 for i in all_items if i.passed)
        total_count = len(all_items)
        score = int((passed_count / total_count) * 100) if total_count > 0 else 0

        passed = score >= 70 and environment_ready

        return DeploymentReport(
            checklist=all_items,
            environment_ready=environment_ready,
            monitoring_ready=monitoring_ready,
            backup_ready=backup_ready,
            change_mgmt_ready=change_mgmt_ready,
            readiness_score=score,
            passed=passed,
        )


def validate_deployment_readiness(codebase_path: str) -> AuditResult:
    """
    Run complete deployment readiness validation.

    Args:
        codebase_path: Root of codebase

    Returns:
        AuditResult with findings
    """
    validator = DeploymentValidator(codebase_path)
    report = validator.run_audit()

    result = AuditResult(
        check_name="deployment_readiness",
        passed=report.passed,
        metrics={
            "readiness_score": report.readiness_score,
            "items_passed": report.items_passed,
            "items_total": report.items_total,
            "environment_ready": report.environment_ready,
            "monitoring_ready": report.monitoring_ready,
            "backup_ready": report.backup_ready,
            "change_mgmt_ready": report.change_mgmt_ready,
        },
        details=f"Deployment readiness: {report.readiness_score}% "
                f"({report.items_passed}/{report.items_total} items)",
    )

    # Add findings for failed items
    for item in report.checklist:
        if not item.passed:
            severity = (
                AuditSeverity.HIGH if item.category in ["environment", "backup"]
                else AuditSeverity.MEDIUM
            )

            result.add_finding(
                severity=severity,
                category=ValidationCategory.DEPLOYMENT,
                title=f"Deployment check failed: {item.item}",
                description=f"Category: {item.category}",
                location=item.evidence or "N/A",
                evidence=item.evidence or "Not found",
                remediation=item.recommendation,
                compliance_impact="Missing deployment requirements increase operational risk",
            )

    return result
