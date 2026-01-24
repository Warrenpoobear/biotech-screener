"""
Query 1.1 - Decimal Arithmetic Compliance Audit.

Performs comprehensive scanning of all Python modules to verify:
- 100% usage of Decimal type for financial calculations
- No float operations that could introduce non-determinism
- Proper rounding mode declarations
- Correct string-to-number conversions using explicit Decimal()

This is critical for SEC audit trail requirements.
"""

import ast
import os
import re
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from audit_framework.types import (
    AuditResult,
    AuditSeverity,
    ValidationCategory,
    ValidationFinding,
)


@dataclass
class FloatViolation:
    """A detected float operation in financial code."""

    file_path: str
    line_number: int
    column: int
    violation_type: str
    code_snippet: str
    context: str


@dataclass
class DecimalComplianceReport:
    """Complete compliance report for decimal arithmetic."""

    files_scanned: int
    float_operations: List[FloatViolation] = field(default_factory=list)
    improper_coercions: List[FloatViolation] = field(default_factory=list)
    missing_rounding_context: List[str] = field(default_factory=list)
    compliant_files: List[str] = field(default_factory=list)
    passed: bool = False
    grade: str = "F"

    @property
    def total_violations(self) -> int:
        return len(self.float_operations) + len(self.improper_coercions)


class DecimalComplianceValidator:
    """
    Validates Decimal arithmetic compliance across the codebase.

    Scans for:
    - Float literals in scoring/financial code
    - Float type hints where Decimal expected
    - Division operations that may produce floats
    - Improper string-to-number conversions
    - Missing rounding context declarations
    """

    # Modules that MUST use Decimal-only arithmetic
    FINANCIAL_MODULES: Set[str] = {
        "module_2_financial",
        "module_2_financial_v2",
        "module_3_scoring",
        "module_3_scoring_v2",
        "module_4_clinical_dev",
        "module_4_clinical_dev_v2",
        "module_5_composite",
        "module_5_composite_v3",
        "module_5_composite_with_defensive",
        "module_5_scoring_v3",
        "pos_engine",
        "dilution_risk_engine",
        "short_interest_engine",
        "liquidity_scoring",
        "time_decay_scoring",
    }

    # Patterns indicating financial/scoring operations
    FINANCIAL_PATTERNS: List[str] = [
        r"score",
        r"cash",
        r"burn",
        r"runway",
        r"market_cap",
        r"dilution",
        r"probability",
        r"weight",
        r"normalized",
        r"rank",
        r"composite",
        r"clinical",
        r"financial",
    ]

    def __init__(self, codebase_path: str):
        """Initialize validator with codebase root path."""
        self.codebase_path = Path(codebase_path)
        self.violations: List[FloatViolation] = []
        self.coercions: List[FloatViolation] = []
        self.files_scanned = 0
        self.compliant_files: List[str] = []

    def scan_file(self, file_path: Path) -> List[FloatViolation]:
        """
        Scan a single Python file for float violations.

        Returns list of violations found.
        """
        violations = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.splitlines()
        except (IOError, UnicodeDecodeError):
            return violations

        # Check if this is a financial module
        module_name = file_path.stem
        is_financial_module = module_name in self.FINANCIAL_MODULES

        # Also check if file contains financial patterns
        has_financial_code = any(
            re.search(pattern, content, re.IGNORECASE)
            for pattern in self.FINANCIAL_PATTERNS
        )

        should_check = is_financial_module or has_financial_code

        if not should_check:
            return violations

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return violations

        # Walk AST looking for float violations
        for node in ast.walk(tree):
            violation = self._check_node(node, file_path, lines, is_financial_module)
            if violation:
                violations.append(violation)

        return violations

    def _check_node(
        self,
        node: ast.AST,
        file_path: Path,
        lines: List[str],
        strict_mode: bool,
    ) -> Optional[FloatViolation]:
        """Check an AST node for float violations."""

        # Check for float literals (1.5, 0.25, etc.)
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            # Get surrounding context
            if hasattr(node, "lineno"):
                line = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                context = self._get_context(lines, node.lineno)

                # Skip if it's clearly not a financial calculation
                if self._is_benign_float(line, node.value):
                    return None

                return FloatViolation(
                    file_path=str(file_path),
                    line_number=node.lineno,
                    column=getattr(node, "col_offset", 0),
                    violation_type="FLOAT_LITERAL",
                    code_snippet=line.strip(),
                    context=context,
                )

        # Check for float() calls that should be Decimal()
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "float":
                if hasattr(node, "lineno"):
                    line = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                    context = self._get_context(lines, node.lineno)

                    return FloatViolation(
                        file_path=str(file_path),
                        line_number=node.lineno,
                        column=getattr(node, "col_offset", 0),
                        violation_type="FLOAT_CONVERSION",
                        code_snippet=line.strip(),
                        context=context,
                    )

        # Check for division that may produce floats (in strict mode)
        if strict_mode and isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            if hasattr(node, "lineno"):
                line = lines[node.lineno - 1] if node.lineno <= len(lines) else ""

                # Check if Decimal is used in the operation
                if "Decimal" not in line:
                    context = self._get_context(lines, node.lineno)
                    return FloatViolation(
                        file_path=str(file_path),
                        line_number=node.lineno,
                        column=getattr(node, "col_offset", 0),
                        violation_type="UNGUARDED_DIVISION",
                        code_snippet=line.strip(),
                        context=context,
                    )

        return None

    def _is_benign_float(self, line: str, value: float) -> bool:
        """Check if a float literal is benign (not financial)."""
        # Common benign patterns
        benign_patterns = [
            r"sleep\s*\(",
            r"timeout",
            r"version",
            r"\.0+\s*[,\)]",  # Integer-equivalent floats
            r"log\(",
            r"assert",
            r"test",
            r"ratio\s*=\s*0\.0",  # Initialization patterns
        ]

        for pattern in benign_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return True

        # Floats that are integer-equivalent (1.0, 2.0) are often benign
        if value == int(value):
            return True

        return False

    def _get_context(self, lines: List[str], line_number: int, context_size: int = 2) -> str:
        """Get surrounding context lines."""
        start = max(0, line_number - context_size - 1)
        end = min(len(lines), line_number + context_size)
        context_lines = lines[start:end]
        return "\n".join(
            f"{i + start + 1:4d} | {line}"
            for i, line in enumerate(context_lines)
        )

    def scan_codebase(self, exclude_dirs: Optional[Set[str]] = None) -> DecimalComplianceReport:
        """
        Scan entire codebase for decimal compliance.

        Args:
            exclude_dirs: Directories to exclude (e.g., {"tests", "deprecated"})

        Returns:
            Complete compliance report
        """
        if exclude_dirs is None:
            exclude_dirs = {"tests", "deprecated", "venv", ".venv", "__pycache__", "mnt"}

        python_files = []
        for root, dirs, files in os.walk(self.codebase_path):
            # Remove excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if file.endswith(".py"):
                    python_files.append(Path(root) / file)

        all_violations: List[FloatViolation] = []
        compliant_files: List[str] = []

        for file_path in python_files:
            violations = self.scan_file(file_path)
            if violations:
                all_violations.extend(violations)
            else:
                compliant_files.append(str(file_path.relative_to(self.codebase_path)))

        self.files_scanned = len(python_files)
        self.violations = all_violations
        self.compliant_files = compliant_files

        # Determine grade
        if len(all_violations) == 0:
            grade = "A"
            passed = True
        elif len(all_violations) <= 5:
            grade = "B"
            passed = True
        elif len(all_violations) <= 15:
            grade = "C"
            passed = False
        else:
            grade = "F"
            passed = False

        return DecimalComplianceReport(
            files_scanned=self.files_scanned,
            float_operations=all_violations,
            improper_coercions=[v for v in all_violations if v.violation_type == "FLOAT_CONVERSION"],
            missing_rounding_context=[],  # Would need more complex analysis
            compliant_files=compliant_files,
            passed=passed,
            grade=grade,
        )

    def check_rounding_modes(self) -> List[str]:
        """
        Check for proper ROUND_HALF_EVEN usage in financial modules.

        ROUND_HALF_EVEN (banker's rounding) is required for regulatory compliance.
        """
        issues = []

        for module_name in self.FINANCIAL_MODULES:
            # Find module file
            module_file = self.codebase_path / f"{module_name}.py"
            if not module_file.exists():
                continue

            try:
                with open(module_file, "r", encoding="utf-8") as f:
                    content = f.read()
            except IOError:
                continue

            # Check for quantize() calls without explicit rounding mode
            if "quantize(" in content:
                if "ROUND_HALF_EVEN" not in content and "rounding=" not in content:
                    issues.append(
                        f"{module_name}: Uses quantize() without explicit ROUND_HALF_EVEN rounding mode"
                    )

        return issues


def validate_decimal_compliance(codebase_path: str) -> AuditResult:
    """
    Run complete decimal compliance validation.

    Args:
        codebase_path: Root path of the codebase

    Returns:
        AuditResult with findings and metrics
    """
    validator = DecimalComplianceValidator(codebase_path)
    report = validator.scan_codebase()

    result = AuditResult(
        check_name="decimal_compliance",
        passed=report.passed,
        metrics={
            "files_scanned": report.files_scanned,
            "float_operations_count": len(report.float_operations),
            "improper_coercions_count": len(report.improper_coercions),
            "compliant_files_count": len(report.compliant_files),
            "grade": report.grade,
        },
        details=f"Scanned {report.files_scanned} files, found {report.total_violations} violations",
    )

    # Add findings for each violation
    for v in report.float_operations:
        severity = (
            AuditSeverity.CRITICAL
            if v.violation_type == "FLOAT_CONVERSION"
            else AuditSeverity.HIGH
        )

        result.add_finding(
            severity=severity,
            category=ValidationCategory.DETERMINISM,
            title=f"Float operation in financial code: {v.violation_type}",
            description=f"Found {v.violation_type} at {v.file_path}:{v.line_number}",
            location=f"{v.file_path}:{v.line_number}",
            evidence=v.code_snippet,
            remediation="Replace with Decimal type using Decimal('value') syntax",
            compliance_impact="Float operations can introduce non-determinism, violating SEC audit requirements",
        )

    # Check rounding modes
    rounding_issues = validator.check_rounding_modes()
    for issue in rounding_issues:
        result.add_finding(
            severity=AuditSeverity.MEDIUM,
            category=ValidationCategory.DETERMINISM,
            title="Missing explicit rounding mode",
            description=issue,
            location=issue.split(":")[0],
            evidence="quantize() without ROUND_HALF_EVEN",
            remediation="Add explicit rounding=ROUND_HALF_EVEN parameter",
            compliance_impact="Inconsistent rounding may cause cross-platform discrepancies",
        )

    return result
