"""
Query 5.1 - Architecture Review for Maintainability.

Senior software architect code review for:
- Modularity assessment
- Code complexity metrics
- Design patterns
- Documentation quality
"""

import ast
import os
import re
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from audit_framework.types import (
    AuditResult,
    AuditSeverity,
    ValidationCategory,
)


@dataclass
class ComplexityMetrics:
    """Complexity metrics for a file."""

    file_path: str
    total_lines: int
    function_count: int
    class_count: int
    avg_function_length: int
    max_function_length: int
    max_nesting_depth: int
    cyclomatic_complexity: int = 0


@dataclass
class ArchitectureIssue:
    """An architecture concern."""

    file_path: str
    issue_type: str
    description: str
    severity: str
    recommendation: str


@dataclass
class MaintainabilityReport:
    """Complete maintainability audit report."""

    file_metrics: List[ComplexityMetrics] = field(default_factory=list)
    architecture_issues: List[ArchitectureIssue] = field(default_factory=list)
    total_lines: int = 0
    total_functions: int = 0
    total_classes: int = 0
    avg_complexity: Decimal = Decimal("0")
    has_proper_modularity: bool = False
    has_documentation: bool = False
    maintainability_score: int = 0  # 0-100
    passed: bool = False


class MaintainabilityValidator:
    """
    Validates code maintainability and architecture quality.

    Checks:
    - Module independence and testability
    - Cyclomatic complexity
    - Code organization
    - Documentation coverage
    """

    MAX_FUNCTION_LENGTH = 100
    MAX_FILE_LENGTH = 500
    MAX_COMPLEXITY = 10
    MAX_NESTING = 4

    def __init__(self, codebase_path: str):
        """Initialize validator."""
        self.codebase_path = Path(codebase_path)

    def _calculate_nesting_depth(self, node: ast.AST, depth: int = 0) -> int:
        """Calculate maximum nesting depth."""
        max_depth = depth

        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                child_depth = self._calculate_nesting_depth(child, depth + 1)
                max_depth = max(max_depth, child_depth)
            else:
                child_depth = self._calculate_nesting_depth(child, depth)
                max_depth = max(max_depth, child_depth)

        return max_depth

    def _estimate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Estimate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.comprehension,)):
                complexity += 1

        return complexity

    def _analyze_file(self, file_path: Path) -> Optional[ComplexityMetrics]:
        """Analyze a single Python file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.splitlines()

            tree = ast.parse(content)
        except Exception:
            return None

        rel_path = str(file_path.relative_to(self.codebase_path))

        function_lengths = []
        function_complexities = []
        class_count = 0
        function_count = 0
        max_nesting = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_count += 1
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function_count += 1

                # Calculate function length
                if hasattr(node, "end_lineno"):
                    length = node.end_lineno - node.lineno
                    function_lengths.append(length)

                # Calculate complexity
                complexity = self._estimate_cyclomatic_complexity(node)
                function_complexities.append(complexity)

                # Calculate nesting
                nesting = self._calculate_nesting_depth(node)
                max_nesting = max(max_nesting, nesting)

        avg_length = sum(function_lengths) // len(function_lengths) if function_lengths else 0
        max_length = max(function_lengths) if function_lengths else 0
        avg_complexity = sum(function_complexities) // len(function_complexities) if function_complexities else 0

        return ComplexityMetrics(
            file_path=rel_path,
            total_lines=len(lines),
            function_count=function_count,
            class_count=class_count,
            avg_function_length=avg_length,
            max_function_length=max_length,
            max_nesting_depth=max_nesting,
            cyclomatic_complexity=avg_complexity,
        )

    def _check_modularity(self) -> Tuple[bool, List[ArchitectureIssue]]:
        """Check for proper module separation."""
        issues = []
        has_modularity = True

        # Check for clear module structure
        expected_dirs = ["common", "governance", "backtest"]
        for dir_name in expected_dirs:
            if not (self.codebase_path / dir_name).exists():
                issues.append(ArchitectureIssue(
                    file_path=dir_name,
                    issue_type="missing_module",
                    description=f"Expected module directory '{dir_name}' not found",
                    severity="medium",
                    recommendation=f"Create {dir_name}/ directory for better organization",
                ))

        # Check for god objects (files > 1000 lines)
        for py_file in self.codebase_path.glob("*.py"):
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                if len(lines) > 1000:
                    has_modularity = False
                    issues.append(ArchitectureIssue(
                        file_path=str(py_file.relative_to(self.codebase_path)),
                        issue_type="god_object",
                        description=f"File has {len(lines)} lines (> 1000)",
                        severity="high",
                        recommendation="Split into smaller, focused modules",
                    ))

            except Exception:
                continue

        return has_modularity, issues

    def _check_documentation(self) -> Tuple[bool, List[ArchitectureIssue]]:
        """Check for proper documentation."""
        issues = []
        has_docs = False

        # Check for README
        readme = self.codebase_path / "README.md"
        if readme.exists():
            has_docs = True
        else:
            issues.append(ArchitectureIssue(
                file_path="README.md",
                issue_type="missing_readme",
                description="No README.md found",
                severity="medium",
                recommendation="Add README with project overview and setup instructions",
            ))

        # Check for CLAUDE.md (project-specific)
        claude_md = self.codebase_path / "CLAUDE.md"
        if claude_md.exists():
            has_docs = True

        # Check for docstrings in main modules
        module_files = list(self.codebase_path.glob("module_*.py"))
        modules_with_docstrings = 0

        for module_file in module_files:
            try:
                with open(module_file, "r", encoding="utf-8") as f:
                    content = f.read()

                if content.strip().startswith('"""') or content.strip().startswith("'''"):
                    modules_with_docstrings += 1

            except Exception:
                continue

        if module_files and modules_with_docstrings < len(module_files) * 0.8:
            issues.append(ArchitectureIssue(
                file_path="module_*.py",
                issue_type="missing_docstrings",
                description=f"Only {modules_with_docstrings}/{len(module_files)} modules have docstrings",
                severity="low",
                recommendation="Add module-level docstrings explaining purpose",
            ))

        return has_docs, issues

    def run_audit(self) -> MaintainabilityReport:
        """
        Run complete maintainability audit.

        Returns:
            MaintainabilityReport with findings
        """
        file_metrics: List[ComplexityMetrics] = []
        all_issues: List[ArchitectureIssue] = []

        exclude_dirs = {"tests", "deprecated", "venv", "mnt", "audit_framework", "__pycache__"}

        for root, dirs, files in os.walk(self.codebase_path):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if not file.endswith(".py"):
                    continue

                file_path = Path(root) / file
                metrics = self._analyze_file(file_path)
                if metrics:
                    file_metrics.append(metrics)

                    # Check for issues
                    if metrics.max_function_length > self.MAX_FUNCTION_LENGTH:
                        all_issues.append(ArchitectureIssue(
                            file_path=metrics.file_path,
                            issue_type="long_function",
                            description=f"Function with {metrics.max_function_length} lines",
                            severity="medium",
                            recommendation="Break into smaller functions",
                        ))

                    if metrics.cyclomatic_complexity > self.MAX_COMPLEXITY:
                        all_issues.append(ArchitectureIssue(
                            file_path=metrics.file_path,
                            issue_type="high_complexity",
                            description=f"Cyclomatic complexity of {metrics.cyclomatic_complexity}",
                            severity="medium",
                            recommendation="Simplify logic or extract helper functions",
                        ))

                    if metrics.max_nesting_depth > self.MAX_NESTING:
                        all_issues.append(ArchitectureIssue(
                            file_path=metrics.file_path,
                            issue_type="deep_nesting",
                            description=f"Nesting depth of {metrics.max_nesting_depth}",
                            severity="medium",
                            recommendation="Use early returns or extract functions",
                        ))

        # Check modularity
        has_modularity, modularity_issues = self._check_modularity()
        all_issues.extend(modularity_issues)

        # Check documentation
        has_docs, doc_issues = self._check_documentation()
        all_issues.extend(doc_issues)

        # Calculate totals
        total_lines = sum(m.total_lines for m in file_metrics)
        total_functions = sum(m.function_count for m in file_metrics)
        total_classes = sum(m.class_count for m in file_metrics)

        avg_complexity = Decimal("0")
        if file_metrics:
            complexities = [m.cyclomatic_complexity for m in file_metrics if m.cyclomatic_complexity > 0]
            if complexities:
                avg_complexity = Decimal(sum(complexities)) / Decimal(len(complexities))
                avg_complexity = avg_complexity.quantize(Decimal("0.01"))

        # Calculate score
        score = 100
        score -= len([i for i in all_issues if i.severity == "high"]) * 15
        score -= len([i for i in all_issues if i.severity == "medium"]) * 5
        score -= len([i for i in all_issues if i.severity == "low"]) * 2
        score = max(0, score)

        passed = score >= 60 and has_modularity

        return MaintainabilityReport(
            file_metrics=file_metrics,
            architecture_issues=all_issues,
            total_lines=total_lines,
            total_functions=total_functions,
            total_classes=total_classes,
            avg_complexity=avg_complexity,
            has_proper_modularity=has_modularity,
            has_documentation=has_docs,
            maintainability_score=score,
            passed=passed,
        )


def validate_maintainability(codebase_path: str) -> AuditResult:
    """
    Run complete maintainability validation.

    Args:
        codebase_path: Root of codebase

    Returns:
        AuditResult with findings
    """
    validator = MaintainabilityValidator(codebase_path)
    report = validator.run_audit()

    result = AuditResult(
        check_name="maintainability",
        passed=report.passed,
        metrics={
            "maintainability_score": report.maintainability_score,
            "total_lines": report.total_lines,
            "total_functions": report.total_functions,
            "total_classes": report.total_classes,
            "avg_complexity": str(report.avg_complexity),
            "has_proper_modularity": report.has_proper_modularity,
            "has_documentation": report.has_documentation,
            "issue_count": len(report.architecture_issues),
        },
        details=f"Maintainability score: {report.maintainability_score}/100",
    )

    # Add findings for architecture issues
    for issue in report.architecture_issues:
        severity = (
            AuditSeverity.HIGH if issue.severity == "high"
            else AuditSeverity.MEDIUM if issue.severity == "medium"
            else AuditSeverity.LOW
        )

        result.add_finding(
            severity=severity,
            category=ValidationCategory.ARCHITECTURE,
            title=f"Architecture issue: {issue.issue_type}",
            description=issue.description,
            location=issue.file_path,
            evidence=issue.description,
            remediation=issue.recommendation,
            compliance_impact="Maintainability issues increase technical debt",
        )

    return result
