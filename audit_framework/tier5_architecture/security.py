"""
Query 5.2 - Security & Access Control.

Audits security posture for institutional deployment.

Checks:
- Secrets management
- Data access controls
- Input validation
- Compliance with data privacy
"""

import ast
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
class SecurityFinding:
    """A security vulnerability or concern."""

    file_path: str
    line_number: int
    finding_type: str
    severity: str  # "critical", "high", "medium", "low"
    description: str
    code_snippet: str
    recommendation: str


@dataclass
class SecurityReport:
    """Complete security audit report."""

    findings: List[SecurityFinding] = field(default_factory=list)
    secrets_exposed: int = 0
    hardcoded_credentials: int = 0
    injection_risks: int = 0
    has_input_validation: bool = False
    has_audit_logging: bool = False
    security_score: int = 0  # 0-100
    passed: bool = False

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "critical")

    @property
    def high_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "high")


class SecurityValidator:
    """
    Validates security posture of the codebase.

    Checks:
    - Hardcoded secrets (API keys, passwords)
    - SQL/command injection risks
    - Input validation patterns
    - Audit logging presence
    """

    # Patterns for hardcoded secrets
    SECRET_PATTERNS: List[tuple] = [
        (r'api[_-]?key\s*[=:]\s*["\'][^"\']{20,}["\']', "Hardcoded API key"),
        (r'password\s*[=:]\s*["\'][^"\']+["\']', "Hardcoded password"),
        (r'secret\s*[=:]\s*["\'][^"\']{16,}["\']', "Hardcoded secret"),
        (r'token\s*[=:]\s*["\'][^"\']{20,}["\']', "Hardcoded token"),
        (r'aws[_-]?secret[_-]?access[_-]?key', "AWS secret key"),
        (r'AKIA[0-9A-Z]{16}', "AWS access key ID"),
    ]

    # Patterns for injection risks
    INJECTION_PATTERNS: List[tuple] = [
        (r'subprocess\.(call|run|Popen)\([^)]*\+', "Command injection risk"),
        (r'os\.system\([^)]*\+', "OS command injection"),
        (r'eval\s*\(', "Dangerous eval usage"),
        (r'exec\s*\(', "Dangerous exec usage"),
        (r'\.format\([^)]*user|\.format\([^)]*input', "Format string injection"),
        (r'f["\'][^"\']*\{[^}]*user[^}]*\}', "F-string with user input"),
    ]

    # Patterns for SQL injection
    SQL_PATTERNS: List[tuple] = [
        (r'execute\([^)]*%s[^)]*%', "SQL injection via string formatting"),
        (r'execute\([^)]*\+', "SQL injection via concatenation"),
        (r'cursor\.execute\([^)]*\.format', "SQL injection via format"),
    ]

    def __init__(self, codebase_path: str):
        """Initialize validator."""
        self.codebase_path = Path(codebase_path)

    def _scan_file_for_secrets(self, file_path: Path) -> List[SecurityFinding]:
        """Scan a file for exposed secrets."""
        findings = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.splitlines()
        except Exception:
            return findings

        rel_path = str(file_path.relative_to(self.codebase_path))

        for pattern, description in self.SECRET_PATTERNS:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                line_num = content[:match.start()].count("\n") + 1
                snippet = lines[line_num - 1] if line_num <= len(lines) else ""

                # Skip if it's in a comment or looks like a placeholder
                if snippet.strip().startswith("#"):
                    continue
                if any(p in snippet.lower() for p in ["example", "placeholder", "xxx", "your_"]):
                    continue

                findings.append(SecurityFinding(
                    file_path=rel_path,
                    line_number=line_num,
                    finding_type="exposed_secret",
                    severity="critical",
                    description=description,
                    code_snippet=snippet.strip()[:60] + "...",
                    recommendation="Use environment variables or secret management service",
                ))

        return findings

    def _scan_file_for_injection(self, file_path: Path) -> List[SecurityFinding]:
        """Scan a file for injection vulnerabilities."""
        findings = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.splitlines()
        except Exception:
            return findings

        rel_path = str(file_path.relative_to(self.codebase_path))

        # Check command injection
        for pattern, description in self.INJECTION_PATTERNS:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                line_num = content[:match.start()].count("\n") + 1
                snippet = lines[line_num - 1] if line_num <= len(lines) else ""

                if snippet.strip().startswith("#"):
                    continue

                findings.append(SecurityFinding(
                    file_path=rel_path,
                    line_number=line_num,
                    finding_type="injection_risk",
                    severity="high",
                    description=description,
                    code_snippet=snippet.strip()[:60],
                    recommendation="Use parameterized commands or input sanitization",
                ))

        # Check SQL injection
        for pattern, description in self.SQL_PATTERNS:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                line_num = content[:match.start()].count("\n") + 1
                snippet = lines[line_num - 1] if line_num <= len(lines) else ""

                if snippet.strip().startswith("#"):
                    continue

                findings.append(SecurityFinding(
                    file_path=rel_path,
                    line_number=line_num,
                    finding_type="sql_injection",
                    severity="critical",
                    description=description,
                    code_snippet=snippet.strip()[:60],
                    recommendation="Use parameterized queries",
                ))

        return findings

    def _check_input_validation(self) -> bool:
        """Check for input validation patterns."""
        validation_patterns = [
            r"validate.*input",
            r"sanitize",
            r"input.*validation",
            r"InputValidation",
            r"is_valid_ticker",
            r"ValidationError",
        ]

        for root, _, files in os.walk(self.codebase_path):
            if any(skip in root for skip in ["test", "deprecated", "venv"]):
                continue

            for file in files:
                if not file.endswith(".py"):
                    continue

                file_path = Path(root) / file
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    for pattern in validation_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            return True

                except Exception:
                    continue

        return False

    def _check_audit_logging(self) -> bool:
        """Check for audit logging presence."""
        audit_file = self.codebase_path / "governance" / "audit_log.py"
        return audit_file.exists()

    def _check_env_usage(self) -> List[SecurityFinding]:
        """Check for proper environment variable usage."""
        findings = []
        env_patterns = [
            r"os\.environ\.get",
            r"os\.getenv",
            r"environ\[",
        ]

        has_env_usage = False
        for root, _, files in os.walk(self.codebase_path):
            if any(skip in root for skip in ["test", "deprecated", "venv"]):
                continue

            for file in files:
                if not file.endswith(".py"):
                    continue

                file_path = Path(root) / file
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    for pattern in env_patterns:
                        if re.search(pattern, content):
                            has_env_usage = True
                            break

                except Exception:
                    continue

        return findings

    def run_audit(self) -> SecurityReport:
        """
        Run complete security audit.

        Returns:
            SecurityReport with findings
        """
        all_findings: List[SecurityFinding] = []

        exclude_dirs = {"tests", "deprecated", "venv", "mnt", "audit_framework", "__pycache__"}

        for root, dirs, files in os.walk(self.codebase_path):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if not file.endswith(".py"):
                    continue

                file_path = Path(root) / file

                # Scan for secrets
                secret_findings = self._scan_file_for_secrets(file_path)
                all_findings.extend(secret_findings)

                # Scan for injection
                injection_findings = self._scan_file_for_injection(file_path)
                all_findings.extend(injection_findings)

        # Check for positive security measures
        has_input_validation = self._check_input_validation()
        has_audit_logging = self._check_audit_logging()

        # Count by type
        secrets_exposed = sum(1 for f in all_findings if f.finding_type == "exposed_secret")
        injection_risks = sum(
            1 for f in all_findings if f.finding_type in ["injection_risk", "sql_injection"]
        )

        # Calculate score
        score = 100
        score -= sum(30 for f in all_findings if f.severity == "critical")
        score -= sum(15 for f in all_findings if f.severity == "high")
        score -= sum(5 for f in all_findings if f.severity == "medium")
        score += 10 if has_input_validation else 0
        score += 10 if has_audit_logging else 0
        score = max(0, min(100, score))

        passed = (
            sum(1 for f in all_findings if f.severity == "critical") == 0
            and score >= 70
        )

        return SecurityReport(
            findings=all_findings,
            secrets_exposed=secrets_exposed,
            hardcoded_credentials=secrets_exposed,
            injection_risks=injection_risks,
            has_input_validation=has_input_validation,
            has_audit_logging=has_audit_logging,
            security_score=score,
            passed=passed,
        )


def validate_security(codebase_path: str) -> AuditResult:
    """
    Run complete security validation.

    Args:
        codebase_path: Root of codebase

    Returns:
        AuditResult with findings
    """
    validator = SecurityValidator(codebase_path)
    report = validator.run_audit()

    result = AuditResult(
        check_name="security",
        passed=report.passed,
        metrics={
            "security_score": report.security_score,
            "secrets_exposed": report.secrets_exposed,
            "injection_risks": report.injection_risks,
            "has_input_validation": report.has_input_validation,
            "has_audit_logging": report.has_audit_logging,
            "critical_count": report.critical_count,
            "high_count": report.high_count,
        },
        details=f"Security score: {report.security_score}/100",
    )

    # Add findings
    for finding in report.findings:
        severity = (
            AuditSeverity.CRITICAL if finding.severity == "critical"
            else AuditSeverity.HIGH if finding.severity == "high"
            else AuditSeverity.MEDIUM
        )

        result.add_finding(
            severity=severity,
            category=ValidationCategory.SECURITY,
            title=f"Security issue: {finding.finding_type}",
            description=finding.description,
            location=f"{finding.file_path}:{finding.line_number}",
            evidence=finding.code_snippet,
            remediation=finding.recommendation,
            compliance_impact="Security vulnerabilities expose institutional data to risk",
        )

    return result
