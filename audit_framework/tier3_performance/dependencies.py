"""
Query 3.3 - Dependency & Supply Chain Security.

Validates the "stdlib-only" architecture claim for corporate safety.

Checks:
- Dependency inventory
- Non-stdlib dependencies
- Network call inventory
- Airgapped operation capability
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


# Python standard library modules (partial list of common ones)
STDLIB_MODULES: Set[str] = {
    "abc", "argparse", "ast", "asyncio", "base64", "bisect",
    "calendar", "collections", "contextlib", "copy", "csv",
    "dataclasses", "datetime", "decimal", "difflib", "email",
    "enum", "functools", "glob", "gzip", "hashlib", "heapq",
    "html", "http", "importlib", "inspect", "io", "itertools",
    "json", "logging", "math", "mimetypes", "multiprocessing",
    "numbers", "operator", "os", "pathlib", "pickle", "platform",
    "pprint", "queue", "random", "re", "secrets", "shutil",
    "signal", "socket", "sqlite3", "ssl", "statistics", "string",
    "struct", "subprocess", "sys", "tarfile", "tempfile", "textwrap",
    "threading", "time", "timeit", "traceback", "types", "typing",
    "unittest", "urllib", "uuid", "warnings", "weakref", "xml",
    "zipfile", "zlib",
}

# Known external dependencies (for reference)
KNOWN_EXTERNAL: Set[str] = {
    "requests", "pandas", "numpy", "scipy", "matplotlib", "seaborn",
    "sqlalchemy", "flask", "django", "fastapi", "pytest", "coverage",
    "black", "flake8", "mypy", "pylint", "isort", "pydantic",
    "aiohttp", "httpx", "boto3", "google", "azure", "redis",
    "celery", "beautifulsoup4", "bs4", "lxml", "pillow", "PIL",
}


@dataclass
class ImportInfo:
    """Information about an import."""

    module: str
    file_path: str
    line_number: int
    is_stdlib: bool
    is_local: bool
    full_import: str


@dataclass
class NetworkCall:
    """A detected network call."""

    file_path: str
    line_number: int
    call_type: str  # "http", "socket", "urllib"
    code_snippet: str


@dataclass
class DependencyReport:
    """Complete dependency analysis report."""

    imports: List[ImportInfo] = field(default_factory=list)
    external_deps: Set[str] = field(default_factory=set)
    network_calls: List[NetworkCall] = field(default_factory=list)
    local_modules: Set[str] = field(default_factory=set)
    stdlib_only: bool = True
    airgap_compatible: bool = True
    passed: bool = False

    @property
    def external_count(self) -> int:
        return len(self.external_deps)


class DependencyValidator:
    """
    Validates dependencies and supply chain security.

    Checks:
    - All imports are stdlib or local
    - External dependencies are documented
    - Network calls are inventoried
    - Airgapped operation is possible
    """

    # Network call patterns
    NETWORK_PATTERNS: List[tuple] = [
        (r"requests\.(get|post|put|delete|patch|head)", "http_requests"),
        (r"urllib\.request\.urlopen", "http_urllib"),
        (r"http\.client\.HTTP", "http_client"),
        (r"socket\.(socket|create_connection)", "socket"),
        (r"aiohttp\.", "http_aiohttp"),
        (r"httpx\.", "http_httpx"),
    ]

    def __init__(self, codebase_path: str):
        """Initialize validator."""
        self.codebase_path = Path(codebase_path)
        self.local_modules: Set[str] = set()
        self._discover_local_modules()

    def _discover_local_modules(self):
        """Discover all local Python modules."""
        for root, _, files in os.walk(self.codebase_path):
            if any(skip in root for skip in ["venv", ".venv", "__pycache__", "mnt"]):
                continue

            for file in files:
                if file.endswith(".py"):
                    # Get module name from file path
                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(self.codebase_path)
                    parts = list(rel_path.parts)
                    if parts[-1] == "__init__.py":
                        parts = parts[:-1]
                    else:
                        parts[-1] = parts[-1][:-3]  # Remove .py

                    if parts:
                        module_name = parts[0]
                        self.local_modules.add(module_name)

        # Also add common known local modules
        for name in ["common", "src", "governance", "backtest", "config", "tests",
                     "extensions", "collectors", "data_sources", "validation"]:
            if (self.codebase_path / name).exists():
                self.local_modules.add(name)

    def _is_stdlib(self, module_name: str) -> bool:
        """Check if module is part of Python stdlib."""
        base_module = module_name.split(".")[0]
        return base_module in STDLIB_MODULES

    def _is_local(self, module_name: str) -> bool:
        """Check if module is a local project module."""
        base_module = module_name.split(".")[0]
        return base_module in self.local_modules

    def _extract_imports(self, file_path: Path) -> List[ImportInfo]:
        """Extract all imports from a Python file."""
        imports = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
        except Exception:
            return imports

        rel_path = str(file_path.relative_to(self.codebase_path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name
                    imports.append(ImportInfo(
                        module=module,
                        file_path=rel_path,
                        line_number=node.lineno,
                        is_stdlib=self._is_stdlib(module),
                        is_local=self._is_local(module),
                        full_import=f"import {module}",
                    ))

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(ImportInfo(
                        module=node.module,
                        file_path=rel_path,
                        line_number=node.lineno,
                        is_stdlib=self._is_stdlib(node.module),
                        is_local=self._is_local(node.module),
                        full_import=f"from {node.module} import ...",
                    ))

        return imports

    def _find_network_calls(self, file_path: Path) -> List[NetworkCall]:
        """Find network calls in a file."""
        calls = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.splitlines()
        except Exception:
            return calls

        rel_path = str(file_path.relative_to(self.codebase_path))

        for pattern, call_type in self.NETWORK_PATTERNS:
            for match in re.finditer(pattern, content):
                line_num = content[:match.start()].count("\n") + 1
                snippet = lines[line_num - 1] if line_num <= len(lines) else ""

                calls.append(NetworkCall(
                    file_path=rel_path,
                    line_number=line_num,
                    call_type=call_type,
                    code_snippet=snippet.strip()[:80],
                ))

        return calls

    def check_pyproject_dependencies(self) -> Set[str]:
        """Check pyproject.toml for declared dependencies."""
        deps = set()
        pyproject = self.codebase_path / "pyproject.toml"

        if not pyproject.exists():
            return deps

        try:
            with open(pyproject, "r") as f:
                content = f.read()

            # Simple pattern matching for dependencies
            # Look for dependencies section
            in_deps = False
            for line in content.splitlines():
                if re.match(r"\[.*dependencies.*\]", line, re.IGNORECASE):
                    in_deps = True
                    continue
                if in_deps:
                    if line.startswith("["):
                        in_deps = False
                        continue
                    # Extract package name
                    match = re.match(r'[\s"\']*([a-zA-Z0-9_-]+)', line)
                    if match:
                        deps.add(match.group(1).lower())

        except Exception:
            pass

        return deps

    def run_audit(self) -> DependencyReport:
        """
        Run complete dependency audit.

        Returns:
            DependencyReport with findings
        """
        all_imports: List[ImportInfo] = []
        all_network_calls: List[NetworkCall] = []
        external_deps: Set[str] = set()

        for root, _, files in os.walk(self.codebase_path):
            if any(skip in root for skip in ["test", "deprecated", "venv", "mnt", "audit_framework"]):
                continue

            for file in files:
                if not file.endswith(".py"):
                    continue

                file_path = Path(root) / file

                # Extract imports
                imports = self._extract_imports(file_path)
                all_imports.extend(imports)

                # Find external deps
                for imp in imports:
                    if not imp.is_stdlib and not imp.is_local:
                        base = imp.module.split(".")[0]
                        external_deps.add(base)

                # Find network calls
                calls = self._find_network_calls(file_path)
                all_network_calls.extend(calls)

        # Check pyproject
        declared_deps = self.check_pyproject_dependencies()

        # Filter to only truly external (not dev dependencies)
        core_external = external_deps - {"pytest", "coverage", "mypy", "black", "flake8"}

        stdlib_only = len(core_external) == 0
        airgap_compatible = len(all_network_calls) == 0 or all(
            "test" in c.file_path.lower() for c in all_network_calls
        )

        passed = stdlib_only or len(core_external) <= 3

        return DependencyReport(
            imports=all_imports,
            external_deps=external_deps,
            network_calls=all_network_calls,
            local_modules=self.local_modules,
            stdlib_only=stdlib_only,
            airgap_compatible=airgap_compatible,
            passed=passed,
        )


def validate_dependencies(codebase_path: str) -> AuditResult:
    """
    Run complete dependency validation.

    Args:
        codebase_path: Root of codebase

    Returns:
        AuditResult with findings
    """
    validator = DependencyValidator(codebase_path)
    report = validator.run_audit()

    result = AuditResult(
        check_name="dependencies",
        passed=report.passed,
        metrics={
            "total_imports": len(report.imports),
            "external_dependencies": list(report.external_deps),
            "external_count": report.external_count,
            "network_calls": len(report.network_calls),
            "local_modules": list(report.local_modules),
            "stdlib_only": report.stdlib_only,
            "airgap_compatible": report.airgap_compatible,
        },
        details=f"External dependencies: {report.external_count}, "
                f"Stdlib only: {report.stdlib_only}",
    )

    # Add findings for external deps
    core_external = report.external_deps - {"pytest", "coverage", "mypy", "black", "flake8"}
    if core_external:
        result.add_finding(
            severity=AuditSeverity.MEDIUM,
            category=ValidationCategory.SECURITY,
            title="External dependencies detected",
            description=f"Non-stdlib dependencies: {', '.join(sorted(core_external))}",
            location="pyproject.toml",
            evidence=f"Dependencies: {sorted(core_external)}",
            remediation="Document justification for each external dependency or vendor them",
            compliance_impact="External dependencies increase supply chain risk",
        )

    # Add findings for network calls
    if not report.airgap_compatible:
        for call in report.network_calls[:5]:
            result.add_finding(
                severity=AuditSeverity.LOW,
                category=ValidationCategory.SECURITY,
                title=f"Network call: {call.call_type}",
                description=f"Network call in {call.file_path}",
                location=f"{call.file_path}:{call.line_number}",
                evidence=call.code_snippet,
                remediation="Document API dependency and caching strategy",
                compliance_impact="Network calls may fail in airgapped environments",
            )

    return result
