#!/usr/bin/env python3
"""Static hardening checks for biotech agent workflows.

These checks are intentionally narrow and deterministic. They protect agent
operating surfaces without touching screening, ranking, selector, or scoring
behavior.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PIN_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)==([^\s\\#]+)")
WALL_CLOCK_PATTERNS = ("datetime.now", "datetime.utcnow")
DATE_ARTIFACT_RE = re.compile(r"\bartifacts/[^\"'\s]*\d{4}-\d{2}-\d{2}[^\"'\s]*")

LOCAL_IMPORT_ROOTS = {
    "adapters",
    "agents",
    "backtest",
    "common",
    "decision_engine",
    "governance",
    "mcp_server",
    "module_3_schema",
    "module_5_composite",
    "module_5_scoring",
    "ranker_engine",
    "scripts",
    "scientific_cartography",
    "selector_engine",
    "tools",
    "wake_robin_data_pipeline",
}

THIRD_PARTY_IMPORT_ALIASES = {
    "yaml": "pyyaml",
}

NETWORK_SYMBOLS = (
    "requests.get",
    "requests.post",
    "requests.request",
    "urllib.request.urlopen",
    "httpx.get",
    "httpx.post",
    "yfinance.Ticker",
    "yf.Ticker",
)

APPROVAL_RISK_PATTERNS = (
    "APPROVED_FOR_PRODUCTION",
    "PRODUCTION_DEPLOYMENT_APPROVED",
    '"production_deployment_approved": true',
    '"automation_approval": true',
    "automation_approval = True",
    "human_approved_by_automation",
)

TIER_PATH_RULES: tuple[tuple[int, tuple[str, ...]], ...] = (
    (
        4,
        (
            "governance/AGENT_ROUTING_POLICY.md",
            "HASH_ROTATIONS.md",
            "production_data/decision_rulesets/",
        ),
    ),
    (
        3,
        (
            "decision_engine.py",
            "selector_engine.py",
            "ranker_engine.py",
            "run_screen.py",
            "run_phase2",
            "common/",
            "production_data/",
        ),
    ),
    (
        2,
        (
            "tools/",
            "scripts/",
            "scientific_cartography/",
            "tests/scientific_cartography/",
        ),
    ),
    (
        1,
        (
            "docs/",
            ".github/",
            ".cursor/",
            "tests/test_agent_workflow_hardening.py",
        ),
    ),
)

DEFAULT_WALL_CLOCK_PATHS = (
    "scientific_cartography/langgraph_review",
    "tools/agent_preflight.py",
    "tools/check_agent_workflow.py",
    "tools/run_scientific_cartography_scheduled_review.py",
)

DEFAULT_APPROVAL_LANGUAGE_PATHS = (
    "scientific_cartography/langgraph_review",
    "tools/run_scientific_cartography_scheduled_review.py",
)


@dataclass(frozen=True)
class CheckResult:
    """Result for one deterministic workflow check."""

    name: str
    ok: bool
    message: str


def _parse_pinned_packages(path: Path) -> dict[str, str]:
    packages: dict[str, str] = {}
    if not path.exists():
        return packages

    for line in path.read_text(encoding="utf-8").splitlines():
        match = PIN_RE.match(line)
        if match:
            packages[match.group(1).lower()] = match.group(2)
    return packages


def _normalize_package_name(name: str) -> str:
    return name.lower().replace("_", "-")


def _parse_pyproject_dependencies(path: Path) -> set[str]:
    if not path.exists():
        return set()

    names: set[str] = set()
    in_dependencies = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "dependencies = [":
            in_dependencies = True
            continue
        if in_dependencies and line == "]":
            break
        if in_dependencies:
            match = re.search(r'"([A-Za-z0-9_.-]+)', line)
            if match:
                names.add(_normalize_package_name(match.group(1)))
    return names


def _declared_packages(requirements_path: Path, pyproject_path: Path) -> set[str]:
    packages = {_normalize_package_name(name) for name in _parse_pinned_packages(requirements_path)}
    packages.update(_parse_pyproject_dependencies(pyproject_path))
    return packages


def _is_stdlib_import(name: str) -> bool:
    root = name.split(".", 1)[0]
    stdlib_names = getattr(sys, "stdlib_module_names", set())
    return root in stdlib_names


def _import_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return roots

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def check_dependency_lock(requirements_path: Path, lock_path: Path) -> CheckResult:
    """Ensure every pinned runtime requirement is represented in requirements.lock."""
    requirements = _parse_pinned_packages(requirements_path)
    locked = _parse_pinned_packages(lock_path)

    if not requirements_path.exists():
        return CheckResult("dependency_lock", False, f"Missing requirements file: {requirements_path}")
    if not lock_path.exists():
        return CheckResult("dependency_lock", False, f"Missing lock file: {lock_path}")

    missing = sorted(name for name in requirements if name not in locked)
    if missing:
        return CheckResult(
            "dependency_lock",
            False,
            "Pinned packages missing from requirements.lock: " + ", ".join(missing),
        )

    mismatched = sorted(name for name, version in requirements.items() if locked.get(name) != version)
    if mismatched:
        return CheckResult(
            "dependency_lock",
            False,
            "Pinned package versions differ from requirements.lock: " + ", ".join(mismatched),
        )

    return CheckResult("dependency_lock", True, "All pinned runtime requirements are locked.")


def _iter_python_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(p for p in path.rglob("*.py") if p.is_file()))
    return files


def check_wall_clock_usage(paths: list[Path]) -> CheckResult:
    """Block wall-clock calls in deterministic agent/review workflow paths."""
    offenders: list[str] = []
    for path in _iter_python_files(paths):
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            offenders.append(f"{path}: syntax error while scanning wall-clock usage: {exc}")
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in {"now", "utcnow"}:
                continue
            receiver = node.func.value
            if isinstance(receiver, ast.Name) and receiver.id == "datetime":
                offenders.append(f"{path}:{node.lineno}: datetime.{node.func.attr}")

    if offenders:
        return CheckResult(
            "wall_clock_usage",
            False,
            "Wall-clock calls found in deterministic workflow paths: " + "; ".join(offenders),
        )

    return CheckResult("wall_clock_usage", True, "No wall-clock calls in deterministic workflow paths.")


def check_langgraph_declared(requirements_path: Path, lock_path: Path, repo_root: Path = REPO_ROOT) -> CheckResult:
    """Ensure LangGraph imports have explicit dependency declarations."""
    langgraph_paths = [
        repo_root / "scientific_cartography" / "langgraph_review",
        repo_root / "tools" / "run_scientific_cartography_langgraph_review.py",
    ]
    uses_langgraph = any(
        "langgraph" in path.read_text(encoding="utf-8") for path in _iter_python_files(langgraph_paths)
    )
    if not uses_langgraph:
        return CheckResult("langgraph_dependency", True, "No LangGraph usage detected.")

    requirements = _parse_pinned_packages(requirements_path)
    locked = _parse_pinned_packages(lock_path)
    missing = []
    if "langgraph" not in requirements:
        missing.append("requirements.txt")
    if "langgraph" not in locked:
        missing.append("requirements.lock")

    if missing:
        return CheckResult(
            "langgraph_dependency",
            False,
            "LangGraph is imported but missing from: " + ", ".join(missing),
        )

    return CheckResult("langgraph_dependency", True, "LangGraph dependency is declared and locked.")


def check_agent_change_manifest(path: Path) -> CheckResult:
    """Require the operator-facing change manifest template."""
    if not path.exists():
        return CheckResult("agent_change_manifest", False, f"Missing agent change manifest template: {path}")

    text = path.read_text(encoding="utf-8")
    required_terms = (
        "governance tier",
        "scoring impact",
        "pit impact",
        "tests run",
        "known gaps",
    )
    missing = [term for term in required_terms if term not in text.lower()]
    if missing:
        return CheckResult(
            "agent_change_manifest",
            False,
            "Agent change manifest missing required fields: " + ", ".join(missing),
        )

    return CheckResult("agent_change_manifest", True, "Agent change manifest template is present.")


def check_scheduled_review_defaults(path: Path) -> CheckResult:
    """Ensure scheduled review does not approve human-review continuation by default."""
    text = path.read_text(encoding="utf-8")
    if "auto_approve: bool = False" not in text:
        return CheckResult(
            "scheduled_review_defaults",
            False,
            "run_scheduled_review must default auto_approve to False.",
        )
    if 'default=False,\n        help="Auto-approve review' not in text:
        return CheckResult(
            "scheduled_review_defaults",
            False,
            "--auto-approve CLI flag must be opt-in with default=False.",
        )
    return CheckResult("scheduled_review_defaults", True, "Scheduled review approval is opt-in.")


def classify_path_tier(path: str) -> int:
    """Classify one changed path by highest matching governance tier."""
    normalized = path.replace("\\", "/").lstrip("./")
    for tier, prefixes in TIER_PATH_RULES:
        for prefix in prefixes:
            if normalized == prefix.rstrip("/") or normalized.startswith(prefix):
                return tier
    return 1


def classify_changed_files(paths: list[str]) -> CheckResult:
    """Summarize the governance tier required for changed files."""
    if not paths:
        return CheckResult("changed_file_tier", True, "No changed files supplied.")

    classified = [(path, classify_path_tier(path)) for path in paths]
    max_tier = max(tier for _, tier in classified)
    details = ", ".join(f"{path}=Tier {tier}" for path, tier in classified)
    return CheckResult(
        "changed_file_tier",
        True,
        f"Maximum governance tier: Tier {max_tier}. {details}",
    )


def check_third_party_imports(paths: list[Path], requirements_path: Path, pyproject_path: Path) -> CheckResult:
    """Ensure third-party imports in workflow paths are declared dependencies."""
    declared = _declared_packages(requirements_path, pyproject_path)
    missing: list[str] = []

    for path in _iter_python_files(paths):
        for root in sorted(_import_roots(path)):
            if root in LOCAL_IMPORT_ROOTS or root.startswith("_") or _is_stdlib_import(root):
                continue
            package_name = THIRD_PARTY_IMPORT_ALIASES.get(root, root)
            if _normalize_package_name(package_name) not in declared:
                missing.append(f"{path}: {root}")

    if missing:
        return CheckResult(
            "third_party_imports",
            False,
            "Third-party imports missing dependency declarations: " + "; ".join(missing),
        )

    return CheckResult("third_party_imports", True, "Workflow third-party imports are declared.")


def check_approval_language(paths: list[Path]) -> CheckResult:
    """Block risky approval terms that collapse human, automation, and production states."""
    offenders: list[str] = []
    for path in _iter_python_files(paths):
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        for pattern in APPROVAL_RISK_PATTERNS:
            if pattern.lower() in lowered:
                offenders.append(f"{path}: {pattern}")

    if offenders:
        return CheckResult(
            "approval_language",
            False,
            "Risky approval language found: " + "; ".join(offenders),
        )

    return CheckResult(
        "approval_language", True, "Approval language keeps review, deployment, and automation separate."
    )


def check_artifact_schema_registry(path: Path) -> CheckResult:
    """Validate the review/diagnostic artifact schema registry exists and has required fields."""
    if not path.exists():
        return CheckResult("artifact_schema_registry", False, f"Missing artifact schema registry: {path}")

    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return CheckResult("artifact_schema_registry", False, f"Invalid schema registry JSON: {exc}")

    artifacts = registry.get("artifacts", {})
    required_artifacts = {
        "hermes_recursive_self_improvement_queue",
        "scientific_cartography_langgraph_review_summary",
        "scientific_cartography_langgraph_human_decision",
        "scientific_cartography_lg3_scheduled_review_cron_execution",
    }
    missing_artifacts = sorted(required_artifacts - set(artifacts))
    if missing_artifacts:
        return CheckResult(
            "artifact_schema_registry",
            False,
            "Missing artifact schemas: " + ", ".join(missing_artifacts),
        )

    missing_fields: list[str] = []
    for name in required_artifacts:
        fields = set(artifacts[name].get("required_fields", []))
        for field in ("artifact_type", "schema_version", "governance"):
            if field not in fields:
                missing_fields.append(f"{name}.{field}")

    if missing_fields:
        return CheckResult(
            "artifact_schema_registry",
            False,
            "Artifact schemas missing required fields: " + ", ".join(missing_fields),
        )

    return CheckResult("artifact_schema_registry", True, "Artifact schema registry is complete for agent workflows.")


def check_stale_artifact_references(paths: list[Path]) -> CheckResult:
    """Detect hard-coded dated artifact paths in docs/tests that can silently go stale."""
    offenders: list[str] = []
    for path in paths:
        if not path.exists() or path.is_dir():
            continue
        text = path.read_text(encoding="utf-8")
        for match in DATE_ARTIFACT_RE.finditer(text):
            offenders.append(f"{path}: {match.group(0)}")

    if offenders:
        return CheckResult(
            "stale_artifact_references",
            False,
            "Hard-coded dated artifact references found: " + "; ".join(offenders),
        )

    return CheckResult("stale_artifact_references", True, "No hard-coded dated artifact references found.")


def check_network_tests_marked(paths: list[Path]) -> CheckResult:
    """Require network-marked tests for files that appear to make live HTTP calls."""
    offenders: list[str] = []
    for path in _iter_python_files(paths):
        text = path.read_text(encoding="utf-8")
        if "pytest.mark.network" in text:
            continue
        if "patch(" in text or "monkeypatch" in text:
            continue
        if any(symbol in text for symbol in NETWORK_SYMBOLS):
            offenders.append(str(path))

    if offenders:
        return CheckResult(
            "network_test_markers",
            False,
            "Potential live-network tests missing pytest.mark.network: " + ", ".join(offenders),
        )

    return CheckResult("network_test_markers", True, "Network-looking tests are marked or mocked.")


def run_checks(repo_root: Path = REPO_ROOT) -> list[CheckResult]:
    """Run all static workflow checks."""
    requirements_path = repo_root / "requirements.txt"
    lock_path = repo_root / "requirements.lock"
    pyproject_path = repo_root / "pyproject.toml"
    wall_clock_paths = [repo_root / path for path in DEFAULT_WALL_CLOCK_PATHS]
    approval_language_paths = [repo_root / path for path in DEFAULT_APPROVAL_LANGUAGE_PATHS]

    return [
        check_dependency_lock(requirements_path, lock_path),
        check_langgraph_declared(requirements_path, lock_path, repo_root),
        check_wall_clock_usage(wall_clock_paths),
        check_third_party_imports(wall_clock_paths, requirements_path, pyproject_path),
        check_approval_language(approval_language_paths),
        check_agent_change_manifest(repo_root / "docs" / "templates" / "AGENT_CHANGE_MANIFEST.md"),
        check_scheduled_review_defaults(repo_root / "tools" / "run_scientific_cartography_scheduled_review.py"),
        check_artifact_schema_registry(repo_root / "docs" / "agent_artifact_schemas.json"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic agent workflow hardening checks")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--changed-file", action="append", default=[], help="Changed file path to classify by tier")
    parser.add_argument(
        "--check-network-tests",
        action="store_true",
        help="Also scan tests for unmarked live-network patterns",
    )
    parser.add_argument(
        "--check-stale-artifact-refs",
        action="store_true",
        help="Also scan docs/tests for hard-coded dated artifact references",
    )
    args = parser.parse_args()

    results = run_checks(args.repo_root)
    if args.changed_file:
        results.append(classify_changed_files(args.changed_file))
    if args.check_network_tests:
        results.append(check_network_tests_marked([args.repo_root / "tests"]))
    if args.check_stale_artifact_refs:
        results.append(check_stale_artifact_references([args.repo_root / "docs" / "AGENT_WORKFLOW_HARDENING.md"]))
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"{status} {result.name}: {result.message}")

    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
