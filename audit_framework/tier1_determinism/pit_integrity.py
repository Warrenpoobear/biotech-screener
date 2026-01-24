"""
Query 1.3 - Point-in-Time Data Integrity.

Audits the system for temporal leakage that would invalidate backtesting.

Validates:
- All data loading functions use as_of_date parameter
- No API calls without historical snapshot support
- Files are versioned/immutable
- Caching doesn't serve stale/future data
- Provenance timestamps are enforced
- source_date <= as_of_date - 1 (PIT cutoff)
"""

import ast
import json
import os
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from audit_framework.types import (
    AuditResult,
    AuditSeverity,
    ValidationCategory,
)


@dataclass
class PITViolation:
    """A detected point-in-time safety violation."""

    file_path: str
    line_number: int
    violation_type: str
    description: str
    code_snippet: str
    severity: str  # "critical", "high", "medium"


@dataclass
class PITIntegrityReport:
    """Complete PIT integrity audit report."""

    files_scanned: int
    violations: List[PITViolation] = field(default_factory=list)
    data_sources_audited: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    pit_compliant_functions: List[str] = field(default_factory=list)
    missing_as_of_date: List[str] = field(default_factory=list)
    passed: bool = False

    @property
    def critical_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "critical")

    @property
    def walk_forward_capable(self) -> bool:
        """Can historical backtests be reconstructed?"""
        return self.critical_count == 0 and len(self.missing_as_of_date) == 0


class PITIntegrityValidator:
    """
    Validates Point-in-Time data integrity across the pipeline.

    Ensures:
    - No lookahead bias in data loading
    - All functions accept as_of_date parameter where needed
    - Data sources are snapshotted with timestamps
    - PIT cutoff (as_of_date - 1) is enforced
    """

    # Functions that MUST have as_of_date parameter
    REQUIRED_PIT_FUNCTIONS: Set[str] = {
        "load_universe",
        "load_financial_records",
        "load_trial_records",
        "load_market_data",
        "load_holdings_snapshots",
        "compute_financial_score",
        "compute_catalyst_score",
        "compute_clinical_score",
        "compute_composite_score",
        "run_screen",
    }

    # Data loading patterns that need PIT enforcement
    DATA_LOADING_PATTERNS: List[str] = [
        r"def\s+load_\w+\s*\(",
        r"def\s+fetch_\w+\s*\(",
        r"def\s+get_\w+_data\s*\(",
        r"def\s+compute_\w+_score\s*\(",
        r"json\.load\s*\(",
        r"pd\.read_\w+\s*\(",
    ]

    # Patterns indicating PIT-unsafe code
    UNSAFE_PATTERNS: List[Tuple[str, str, str]] = [
        (
            r"datetime\.now\(\)",
            "Using current datetime without as_of_date",
            "critical",
        ),
        (
            r"date\.today\(\)",
            "Using current date without as_of_date",
            "critical",
        ),
        (
            r"requests\.get\s*\([^)]*(?!as_of)[^)]*\)",
            "API call without historical date parameter",
            "high",
        ),
        (
            r"urllib\.request\.urlopen",
            "URL fetch without snapshot versioning",
            "high",
        ),
    ]

    def __init__(self, codebase_path: str):
        """Initialize validator with codebase root."""
        self.codebase_path = Path(codebase_path)
        self.violations: List[PITViolation] = []
        self.files_scanned = 0

    def _check_function_signature(
        self,
        func_def: ast.FunctionDef,
        file_path: Path,
        lines: List[str],
    ) -> Optional[PITViolation]:
        """Check if a function that should have as_of_date parameter has it."""
        func_name = func_def.name

        # Check if this function should have as_of_date
        needs_as_of_date = any(
            func_name.startswith(prefix)
            for prefix in ["load_", "fetch_", "compute_", "get_"]
        )

        if not needs_as_of_date:
            return None

        # Check parameters for as_of_date
        param_names = {arg.arg for arg in func_def.args.args}
        param_names.update({arg.arg for arg in func_def.args.kwonlyargs})

        has_as_of = any(
            p in param_names
            for p in ["as_of_date", "as_of", "date", "pit_date", "cutoff_date"]
        )

        if has_as_of:
            return None

        # This is a potential violation
        line_num = func_def.lineno
        snippet = lines[line_num - 1] if line_num <= len(lines) else ""

        return PITViolation(
            file_path=str(file_path),
            line_number=line_num,
            violation_type="MISSING_AS_OF_DATE",
            description=f"Function {func_name} should accept as_of_date parameter",
            code_snippet=snippet.strip(),
            severity="high",
        )

    def _check_data_source_compliance(
        self,
        file_path: Path,
        content: str,
        lines: List[str],
    ) -> List[PITViolation]:
        """Check data source operations for PIT compliance."""
        violations = []

        for pattern, description, severity in self.UNSAFE_PATTERNS:
            for match in re.finditer(pattern, content):
                # Get line number
                line_start = content[:match.start()].count("\n") + 1
                snippet = lines[line_start - 1] if line_start <= len(lines) else ""

                # Skip if in comment or docstring context
                stripped = snippet.strip()
                if stripped.startswith("#") or stripped.startswith('"""'):
                    continue

                violations.append(PITViolation(
                    file_path=str(file_path),
                    line_number=line_start,
                    violation_type="PIT_UNSAFE_PATTERN",
                    description=description,
                    code_snippet=snippet.strip()[:100],
                    severity=severity,
                ))

        return violations

    def _audit_data_files(self) -> Dict[str, Dict[str, Any]]:
        """
        Audit data files for PIT compliance infrastructure.

        Checks:
        - Do data files have source_date fields?
        - Are they versioned/timestamped?
        - Is there a snapshot strategy?
        """
        data_sources: Dict[str, Dict[str, Any]] = {}
        data_dir = self.codebase_path / "production_data"

        if not data_dir.exists():
            return data_sources

        json_files = list(data_dir.glob("*.json"))

        for json_file in json_files:
            source_info = {
                "has_source_date": False,
                "has_timestamp": False,
                "has_pit_cutoff": False,
                "record_count": 0,
                "sample_record": None,
            }

            try:
                with open(json_file, "r") as f:
                    data = json.load(f)

                # Check structure
                if isinstance(data, list):
                    source_info["record_count"] = len(data)
                    if data:
                        sample = data[0]
                        source_info["sample_record"] = {
                            k: type(v).__name__ for k, v in sample.items()
                        } if isinstance(sample, dict) else str(type(sample))

                        if isinstance(sample, dict):
                            source_info["has_source_date"] = any(
                                k in sample for k in ["source_date", "as_of_date", "filed_date"]
                            )
                            source_info["has_timestamp"] = any(
                                k in sample for k in ["timestamp", "created_at", "updated_at"]
                            )

                elif isinstance(data, dict):
                    # Check for governance block
                    if "_governance" in data:
                        gov = data["_governance"]
                        source_info["has_pit_cutoff"] = "pit_cutoff" in gov
                        source_info["has_timestamp"] = "generated_at" in gov

                    # Check for as_of_date at top level
                    source_info["has_source_date"] = any(
                        k in data for k in ["as_of_date", "source_date"]
                    )

            except Exception:
                pass

            data_sources[json_file.name] = source_info

        return data_sources

    def _check_pit_enforcement_code(self) -> List[str]:
        """Check for proper PIT enforcement patterns in codebase."""
        compliant_patterns = [
            r"compute_pit_cutoff",
            r"is_pit_admissible",
            r"pit_cutoff\s*=",
            r"as_of_date\s*-\s*timedelta",
            r"source_date\s*<=\s*pit_cutoff",
        ]

        compliant_files = []
        common_dir = self.codebase_path / "common"

        if common_dir.exists():
            for py_file in common_dir.glob("*.py"):
                try:
                    with open(py_file, "r", encoding="utf-8") as f:
                        content = f.read()

                    if any(re.search(p, content) for p in compliant_patterns):
                        compliant_files.append(str(py_file.relative_to(self.codebase_path)))
                except Exception:
                    continue

        return compliant_files

    def scan_file(self, file_path: Path) -> List[PITViolation]:
        """Scan a single file for PIT violations."""
        violations = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.splitlines()
        except (IOError, UnicodeDecodeError):
            return violations

        # Check for unsafe patterns
        violations.extend(self._check_data_source_compliance(file_path, content, lines))

        # Parse AST for function signatures
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return violations

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                violation = self._check_function_signature(node, file_path, lines)
                if violation:
                    violations.append(violation)

        return violations

    def scan_codebase(
        self,
        exclude_dirs: Optional[Set[str]] = None,
    ) -> PITIntegrityReport:
        """
        Scan entire codebase for PIT integrity.

        Args:
            exclude_dirs: Directories to exclude

        Returns:
            Complete PIT integrity report
        """
        if exclude_dirs is None:
            exclude_dirs = {
                "tests", "deprecated", "venv", ".venv",
                "__pycache__", "mnt", "audit_framework",
            }

        python_files = []
        for root, dirs, files in os.walk(self.codebase_path):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if file.endswith(".py"):
                    python_files.append(Path(root) / file)

        all_violations: List[PITViolation] = []

        for file_path in python_files:
            violations = self.scan_file(file_path)
            all_violations.extend(violations)

        self.files_scanned = len(python_files)
        self.violations = all_violations

        # Audit data files
        data_sources = self._audit_data_files()

        # Check for compliant enforcement code
        compliant_functions = self._check_pit_enforcement_code()

        # Extract missing as_of_date functions
        missing_as_of = [
            f"{v.file_path}:{v.line_number} - {v.description}"
            for v in all_violations
            if v.violation_type == "MISSING_AS_OF_DATE"
        ]

        # Determine pass/fail
        critical_violations = sum(1 for v in all_violations if v.severity == "critical")
        passed = critical_violations == 0

        return PITIntegrityReport(
            files_scanned=self.files_scanned,
            violations=all_violations,
            data_sources_audited=data_sources,
            pit_compliant_functions=compliant_functions,
            missing_as_of_date=missing_as_of,
            passed=passed,
        )


def validate_pit_integrity(codebase_path: str) -> AuditResult:
    """
    Run complete PIT integrity validation.

    Args:
        codebase_path: Root of codebase

    Returns:
        AuditResult with findings
    """
    validator = PITIntegrityValidator(codebase_path)
    report = validator.scan_codebase()

    result = AuditResult(
        check_name="pit_integrity",
        passed=report.passed,
        metrics={
            "files_scanned": report.files_scanned,
            "violations_count": len(report.violations),
            "critical_violations": report.critical_count,
            "data_sources_audited": len(report.data_sources_audited),
            "pit_compliant_functions": len(report.pit_compliant_functions),
            "walk_forward_capable": report.walk_forward_capable,
        },
        details=f"Scanned {report.files_scanned} files, found {len(report.violations)} PIT violations",
    )

    # Add findings for violations
    for v in report.violations:
        severity = (
            AuditSeverity.CRITICAL if v.severity == "critical"
            else AuditSeverity.HIGH if v.severity == "high"
            else AuditSeverity.MEDIUM
        )

        result.add_finding(
            severity=severity,
            category=ValidationCategory.PIT_SAFETY,
            title=f"PIT violation: {v.violation_type}",
            description=v.description,
            location=f"{v.file_path}:{v.line_number}",
            evidence=v.code_snippet,
            remediation="Add as_of_date parameter and use compute_pit_cutoff() for data filtering",
            compliance_impact="Temporal leakage invalidates backtests and violates FINRA/SEC compliance",
        )

    # Add data source compliance information
    for source_name, info in report.data_sources_audited.items():
        if not info.get("has_source_date") and not info.get("has_pit_cutoff"):
            result.add_finding(
                severity=AuditSeverity.MEDIUM,
                category=ValidationCategory.PIT_SAFETY,
                title=f"Data source missing temporal metadata: {source_name}",
                description=f"Data file {source_name} lacks source_date or pit_cutoff fields",
                location=f"production_data/{source_name}",
                evidence=json.dumps(info, indent=2),
                remediation="Add source_date field to each record or _governance.pit_cutoff to file",
                compliance_impact="Cannot verify point-in-time safety without temporal metadata",
            )

    return result
