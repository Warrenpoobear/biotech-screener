"""
Query 4.1 - Test Coverage & Regression Suite.

Audits testing infrastructure for institutional-grade quality.

Metrics:
- Unit test coverage
- Integration test coverage
- Edge case coverage
- Regression test presence
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
class TestFile:
    """Information about a test file."""

    file_path: str
    test_count: int
    test_names: List[str] = field(default_factory=list)
    fixtures_used: List[str] = field(default_factory=list)
    has_golden_test: bool = False
    has_edge_case: bool = False


@dataclass
class ModuleCoverage:
    """Coverage analysis for a source module."""

    module_name: str
    source_functions: int
    tested_functions: int
    coverage_percent: Decimal
    untested_functions: List[str] = field(default_factory=list)


@dataclass
class TestCoverageReport:
    """Complete test coverage report."""

    total_test_files: int
    total_tests: int
    test_files: List[TestFile] = field(default_factory=list)
    module_coverage: List[ModuleCoverage] = field(default_factory=list)
    has_integration_tests: bool = False
    has_regression_tests: bool = False
    has_golden_tests: bool = False
    estimated_coverage: Decimal = Decimal("0")
    passed: bool = False


class TestCoverageValidator:
    """
    Validates test coverage and testing infrastructure.

    Checks:
    - Test file count and organization
    - Test function patterns
    - Golden output tests for determinism
    - Edge case coverage
    - Integration test presence
    """

    # Patterns indicating different test types
    GOLDEN_TEST_PATTERNS = [
        r"golden",
        r"baseline",
        r"snapshot",
        r"regression",
        r"determinism",
        r"hash.*equal",
        r"identical.*output",
    ]

    EDGE_CASE_PATTERNS = [
        r"edge",
        r"boundary",
        r"extreme",
        r"zero",
        r"empty",
        r"null",
        r"invalid",
        r"malformed",
        r"corrupt",
    ]

    INTEGRATION_PATTERNS = [
        r"integration",
        r"e2e",
        r"end_to_end",
        r"full_pipeline",
        r"workflow",
    ]

    def __init__(self, codebase_path: str):
        """Initialize validator."""
        self.codebase_path = Path(codebase_path)
        self.tests_dir = self.codebase_path / "tests"

    def _analyze_test_file(self, file_path: Path) -> Optional[TestFile]:
        """Analyze a single test file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
        except Exception:
            return None

        rel_path = str(file_path.relative_to(self.codebase_path))
        test_names = []
        fixtures_used = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith("test_"):
                    test_names.append(node.name)

                # Check for fixture decorators
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Attribute):
                        if decorator.attr == "fixture":
                            fixtures_used.append(node.name)

        content_lower = content.lower()
        has_golden = any(
            re.search(p, content_lower) for p in self.GOLDEN_TEST_PATTERNS
        )
        has_edge = any(
            re.search(p, content_lower) for p in self.EDGE_CASE_PATTERNS
        )

        return TestFile(
            file_path=rel_path,
            test_count=len(test_names),
            test_names=test_names,
            fixtures_used=fixtures_used,
            has_golden_test=has_golden,
            has_edge_case=has_edge,
        )

    def _get_source_functions(self, module_path: Path) -> List[str]:
        """Get all public function names from a source module."""
        try:
            with open(module_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
        except Exception:
            return []

        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not node.name.startswith("_"):
                    functions.append(node.name)

        return functions

    def _estimate_function_coverage(
        self,
        source_path: Path,
        test_content: str,
    ) -> Tuple[int, List[str]]:
        """Estimate how many source functions are tested."""
        source_functions = self._get_source_functions(source_path)
        tested = 0
        untested = []

        for func in source_functions:
            # Check if function name appears in test content
            if re.search(rf"\b{func}\b", test_content):
                tested += 1
            else:
                untested.append(func)

        return tested, untested

    def analyze_module_coverage(self) -> List[ModuleCoverage]:
        """Analyze coverage for each source module."""
        coverage = []

        # Collect all test content
        all_test_content = ""
        if self.tests_dir.exists():
            for test_file in self.tests_dir.glob("**/test_*.py"):
                try:
                    with open(test_file, "r", encoding="utf-8") as f:
                        all_test_content += f.read() + "\n"
                except Exception:
                    continue

        # Analyze each module
        module_patterns = [
            "module_*.py",
            "*_engine.py",
            "*_adapter.py",
            "*_scoring.py",
        ]

        for pattern in module_patterns:
            for module_path in self.codebase_path.glob(pattern):
                if "test" in str(module_path).lower():
                    continue

                source_functions = self._get_source_functions(module_path)
                if not source_functions:
                    continue

                tested, untested = self._estimate_function_coverage(
                    module_path, all_test_content
                )

                cov_percent = Decimal("0")
                if source_functions:
                    cov_percent = (
                        Decimal(tested) / Decimal(len(source_functions))
                    ).quantize(Decimal("0.0001"))

                coverage.append(ModuleCoverage(
                    module_name=module_path.name,
                    source_functions=len(source_functions),
                    tested_functions=tested,
                    coverage_percent=cov_percent,
                    untested_functions=untested[:10],
                ))

        return coverage

    def check_integration_tests(self) -> bool:
        """Check for presence of integration tests."""
        integration_dir = self.tests_dir / "integration"
        if integration_dir.exists():
            return any(integration_dir.glob("*.py"))

        # Also check for integration patterns in test names
        for test_file in self.tests_dir.glob("**/test_*.py"):
            try:
                with open(test_file, "r", encoding="utf-8") as f:
                    content = f.read().lower()

                if any(re.search(p, content) for p in self.INTEGRATION_PATTERNS):
                    return True
            except Exception:
                continue

        return False

    def run_audit(self) -> TestCoverageReport:
        """
        Run complete test coverage audit.

        Returns:
            TestCoverageReport with findings
        """
        test_files: List[TestFile] = []
        total_tests = 0

        if self.tests_dir.exists():
            for test_file in self.tests_dir.glob("**/test_*.py"):
                analysis = self._analyze_test_file(test_file)
                if analysis:
                    test_files.append(analysis)
                    total_tests += analysis.test_count

        # Analyze module coverage
        module_coverage = self.analyze_module_coverage()

        # Check for special test types
        has_integration = self.check_integration_tests()
        has_golden = any(tf.has_golden_test for tf in test_files)
        has_regression = any(
            "regression" in tf.file_path.lower() for tf in test_files
        )

        # Estimate overall coverage
        if module_coverage:
            total_funcs = sum(mc.source_functions for mc in module_coverage)
            tested_funcs = sum(mc.tested_functions for mc in module_coverage)
            estimated_cov = Decimal("0")
            if total_funcs > 0:
                estimated_cov = (
                    Decimal(tested_funcs) / Decimal(total_funcs)
                ).quantize(Decimal("0.0001"))
        else:
            estimated_cov = Decimal("0")

        # Determine pass/fail
        passed = (
            total_tests >= 50
            and estimated_cov >= Decimal("0.60")
            and has_integration
        )

        return TestCoverageReport(
            total_test_files=len(test_files),
            total_tests=total_tests,
            test_files=test_files,
            module_coverage=module_coverage,
            has_integration_tests=has_integration,
            has_regression_tests=has_regression,
            has_golden_tests=has_golden,
            estimated_coverage=estimated_cov,
            passed=passed,
        )


def validate_test_coverage(codebase_path: str) -> AuditResult:
    """
    Run complete test coverage validation.

    Args:
        codebase_path: Root of codebase

    Returns:
        AuditResult with findings
    """
    validator = TestCoverageValidator(codebase_path)
    report = validator.run_audit()

    result = AuditResult(
        check_name="test_coverage",
        passed=report.passed,
        metrics={
            "total_test_files": report.total_test_files,
            "total_tests": report.total_tests,
            "estimated_coverage": str(report.estimated_coverage),
            "has_integration_tests": report.has_integration_tests,
            "has_regression_tests": report.has_regression_tests,
            "has_golden_tests": report.has_golden_tests,
        },
        details=f"Found {report.total_tests} tests across {report.total_test_files} files, "
                f"estimated coverage: {report.estimated_coverage*100:.1f}%",
    )

    # Add findings for coverage gaps
    for mc in report.module_coverage:
        if mc.coverage_percent < Decimal("0.50"):
            result.add_finding(
                severity=AuditSeverity.MEDIUM,
                category=ValidationCategory.TEST_COVERAGE,
                title=f"Low test coverage: {mc.module_name}",
                description=f"Module has {mc.coverage_percent*100:.0f}% estimated coverage",
                location=mc.module_name,
                evidence=f"Untested functions: {', '.join(mc.untested_functions[:5])}",
                remediation="Add unit tests for untested functions",
                compliance_impact="Low test coverage increases regression risk",
            )

    if not report.has_integration_tests:
        result.add_finding(
            severity=AuditSeverity.HIGH,
            category=ValidationCategory.TEST_COVERAGE,
            title="Missing integration tests",
            description="No integration tests found",
            location="tests/integration/",
            evidence="Integration test directory missing or empty",
            remediation="Add end-to-end pipeline integration tests",
            compliance_impact="Missing integration tests may miss system-level issues",
        )

    if not report.has_golden_tests:
        result.add_finding(
            severity=AuditSeverity.HIGH,
            category=ValidationCategory.TEST_COVERAGE,
            title="Missing golden/determinism tests",
            description="No golden output or determinism tests found",
            location="tests/",
            evidence="No test files with golden/baseline/determinism patterns",
            remediation="Add golden output tests to verify deterministic behavior",
            compliance_impact="Missing determinism tests cannot guarantee reproducibility",
        )

    return result
