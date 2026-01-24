"""
Query 3.2 - Error Handling & Resilience Audit.

Validates production-grade error handling across the pipeline.

Scenarios:
- Network failure handling
- Data corruption scenarios
- Resource exhaustion
- Exception handling patterns
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
class ResiliencePattern:
    """A detected resilience pattern (good or bad)."""

    file_path: str
    line_number: int
    pattern_type: str  # "retry", "timeout", "circuit_breaker", etc.
    is_positive: bool  # True for good patterns, False for anti-patterns
    description: str
    code_snippet: str


@dataclass
class ExceptionHandling:
    """Exception handling analysis for a file."""

    file_path: str
    try_blocks: int
    bare_except: int
    specific_except: int
    logged_exceptions: int
    reraised_exceptions: int


@dataclass
class ResilienceReport:
    """Complete resilience audit report."""

    patterns_found: List[ResiliencePattern] = field(default_factory=list)
    exception_analysis: List[ExceptionHandling] = field(default_factory=list)
    anti_patterns: List[str] = field(default_factory=list)
    resilience_score: int = 0  # 0-100
    passed: bool = False

    @property
    def positive_patterns(self) -> List[ResiliencePattern]:
        return [p for p in self.patterns_found if p.is_positive]

    @property
    def negative_patterns(self) -> List[ResiliencePattern]:
        return [p for p in self.patterns_found if not p.is_positive]


class ResilienceValidator:
    """
    Validates error handling and resilience patterns.

    Checks:
    - Retry mechanisms with exponential backoff
    - Timeout configurations
    - Graceful degradation
    - Exception handling quality
    """

    # Positive resilience patterns
    POSITIVE_PATTERNS: List[tuple] = [
        (r"retry|Retry|@retry", "retry_mechanism", "Retry mechanism implemented"),
        (r"timeout\s*=|Timeout", "timeout", "Timeout configuration"),
        (r"exponential_backoff|exp_backoff", "backoff", "Exponential backoff"),
        (r"circuit_breaker|CircuitBreaker", "circuit_breaker", "Circuit breaker pattern"),
        (r"fallback|Fallback", "fallback", "Fallback mechanism"),
        (r"graceful|Graceful", "graceful_degradation", "Graceful degradation"),
    ]

    # Anti-patterns
    ANTI_PATTERNS: List[tuple] = [
        (r"except\s*:\s*\n\s*pass", "silent_swallow", "Silent exception swallowing"),
        (r"except\s+Exception\s*:\s*\n\s*pass", "broad_swallow", "Broad exception swallowing"),
        (r"except\s*:\s*\n\s*continue", "silent_continue", "Silent continue on error"),
        (r"raise\s+Exception\(['\"]", "generic_raise", "Raising generic Exception"),
    ]

    def __init__(self, codebase_path: str):
        """Initialize validator."""
        self.codebase_path = Path(codebase_path)

    def _analyze_exception_handling(self, file_path: Path) -> Optional[ExceptionHandling]:
        """Analyze exception handling in a single file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            return None

        try_blocks = len(re.findall(r"\btry\s*:", content))
        bare_except = len(re.findall(r"\bexcept\s*:", content))
        specific_except = len(re.findall(r"\bexcept\s+\w+", content)) - bare_except
        logged = len(re.findall(r"(logger|logging)\.\w+\(.*(?:error|exception|warning)", content, re.IGNORECASE))
        reraised = len(re.findall(r"\braise\b", content))

        return ExceptionHandling(
            file_path=str(file_path.relative_to(self.codebase_path)),
            try_blocks=try_blocks,
            bare_except=bare_except,
            specific_except=specific_except,
            logged_exceptions=logged,
            reraised_exceptions=reraised,
        )

    def _find_patterns(self, file_path: Path) -> List[ResiliencePattern]:
        """Find resilience patterns in a file."""
        patterns = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.splitlines()
        except Exception:
            return patterns

        rel_path = str(file_path.relative_to(self.codebase_path))

        # Check positive patterns
        for pattern, pattern_type, description in self.POSITIVE_PATTERNS:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                line_num = content[:match.start()].count("\n") + 1
                snippet = lines[line_num - 1] if line_num <= len(lines) else ""

                patterns.append(ResiliencePattern(
                    file_path=rel_path,
                    line_number=line_num,
                    pattern_type=pattern_type,
                    is_positive=True,
                    description=description,
                    code_snippet=snippet.strip()[:80],
                ))

        # Check anti-patterns
        for pattern, pattern_type, description in self.ANTI_PATTERNS:
            for match in re.finditer(pattern, content, re.MULTILINE):
                line_num = content[:match.start()].count("\n") + 1
                snippet = lines[line_num - 1] if line_num <= len(lines) else ""

                patterns.append(ResiliencePattern(
                    file_path=rel_path,
                    line_number=line_num,
                    pattern_type=pattern_type,
                    is_positive=False,
                    description=description,
                    code_snippet=snippet.strip()[:80],
                ))

        return patterns

    def check_retry_implementations(self) -> List[str]:
        """Check for proper retry implementations."""
        findings = []

        for root, _, files in os.walk(self.codebase_path):
            if any(skip in root for skip in ["test", "deprecated", "venv", "mnt", "audit_framework"]):
                continue

            for file in files:
                if not file.endswith(".py"):
                    continue

                file_path = Path(root) / file
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Look for network calls without retry
                    if re.search(r"requests\.(get|post|put|delete)\(", content):
                        if not re.search(r"retry|Retry|@retry", content, re.IGNORECASE):
                            rel_path = str(file_path.relative_to(self.codebase_path))
                            findings.append(f"{rel_path}: HTTP requests without retry mechanism")

                    # Look for file operations without error handling
                    if re.search(r"open\([^)]+\)", content):
                        # Check if open is in try block context
                        try_count = len(re.findall(r"\btry\s*:", content))
                        open_count = len(re.findall(r"\bopen\s*\(", content))
                        if open_count > try_count * 2:  # Rough heuristic
                            rel_path = str(file_path.relative_to(self.codebase_path))
                            findings.append(f"{rel_path}: File operations may lack error handling")

                except Exception:
                    continue

        return findings

    def calculate_resilience_score(
        self,
        patterns: List[ResiliencePattern],
        exception_analysis: List[ExceptionHandling],
    ) -> int:
        """Calculate overall resilience score (0-100)."""
        score = 50  # Start at neutral

        # Positive adjustments
        positive_count = sum(1 for p in patterns if p.is_positive)
        score += min(positive_count * 5, 25)  # Max +25 for positive patterns

        # Check exception handling quality
        for analysis in exception_analysis:
            if analysis.bare_except > 0:
                score -= analysis.bare_except * 2
            if analysis.logged_exceptions > 0:
                score += min(analysis.logged_exceptions, 5)

        # Negative adjustments
        negative_count = sum(1 for p in patterns if not p.is_positive)
        score -= negative_count * 10  # -10 per anti-pattern

        return max(0, min(100, score))

    def run_audit(self) -> ResilienceReport:
        """
        Run complete resilience audit.

        Returns:
            ResilienceReport with findings
        """
        all_patterns: List[ResiliencePattern] = []
        all_exception_analysis: List[ExceptionHandling] = []

        for root, _, files in os.walk(self.codebase_path):
            if any(skip in root for skip in ["test", "deprecated", "venv", "mnt", "audit_framework"]):
                continue

            for file in files:
                if not file.endswith(".py"):
                    continue

                file_path = Path(root) / file

                # Find patterns
                patterns = self._find_patterns(file_path)
                all_patterns.extend(patterns)

                # Analyze exception handling
                analysis = self._analyze_exception_handling(file_path)
                if analysis and analysis.try_blocks > 0:
                    all_exception_analysis.append(analysis)

        # Check retry implementations
        anti_patterns = self.check_retry_implementations()

        # Calculate score
        score = self.calculate_resilience_score(all_patterns, all_exception_analysis)

        # Determine pass/fail (score >= 60 to pass)
        passed = score >= 60 and len([p for p in all_patterns if not p.is_positive]) < 5

        return ResilienceReport(
            patterns_found=all_patterns,
            exception_analysis=all_exception_analysis,
            anti_patterns=anti_patterns,
            resilience_score=score,
            passed=passed,
        )


def validate_resilience(codebase_path: str) -> AuditResult:
    """
    Run complete resilience validation.

    Args:
        codebase_path: Root of codebase

    Returns:
        AuditResult with findings
    """
    validator = ResilienceValidator(codebase_path)
    report = validator.run_audit()

    result = AuditResult(
        check_name="resilience",
        passed=report.passed,
        metrics={
            "resilience_score": report.resilience_score,
            "positive_patterns": len(report.positive_patterns),
            "negative_patterns": len(report.negative_patterns),
            "anti_patterns": len(report.anti_patterns),
            "files_with_exception_handling": len(report.exception_analysis),
        },
        details=f"Resilience score: {report.resilience_score}/100",
    )

    # Add findings for anti-patterns
    for pattern in report.negative_patterns:
        result.add_finding(
            severity=AuditSeverity.HIGH,
            category=ValidationCategory.ERROR_HANDLING,
            title=f"Error handling anti-pattern: {pattern.pattern_type}",
            description=pattern.description,
            location=f"{pattern.file_path}:{pattern.line_number}",
            evidence=pattern.code_snippet,
            remediation="Replace with proper exception handling, logging, and recovery",
            compliance_impact="Poor error handling can cause silent failures and data loss",
        )

    for anti_pattern in report.anti_patterns[:5]:
        result.add_finding(
            severity=AuditSeverity.MEDIUM,
            category=ValidationCategory.ERROR_HANDLING,
            title="Missing resilience mechanism",
            description=anti_pattern,
            location=anti_pattern.split(":")[0],
            evidence=anti_pattern,
            remediation="Add retry logic with exponential backoff",
            compliance_impact="Missing retries can cause data collection failures",
        )

    return result
