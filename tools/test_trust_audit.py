#!/usr/bin/env python3
"""Read-only test-trust auditor (v1: L0 + L1, advisory only).

Usage:
    python -m tools.test_trust_audit --mode static --out reports/
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

SCHEMA_VERSION = "test_trust_audit.v1"
DISABLED_STUBS = {
    "T8": "stale-golden detector disabled in v1",
    "T9": "PIT-leakage detector disabled in v1",
    "T10": "coverage cross-check (L2) disabled in v1",
    "L3": "mutation probe (L3) disabled in v1",
}

SEVERITY_ORDER = {
    "CRITICAL": 0,
    "HIGH": 1,
    "MEDIUM": 2,
    "LOW-MEDIUM": 3,
    "UNCERTAIN": 4,
}
SEVERITY_WEIGHTS = {
    "CRITICAL": 5.0,
    "HIGH": 3.0,
    "MEDIUM": 2.0,
    "LOW-MEDIUM": 1.5,
    "UNCERTAIN": 2.5,
}

GENERIC_SUBJECT_TOKENS = {
    "test",
    "with",
    "without",
    "when",
    "then",
    "should",
    "and",
    "or",
    "for",
    "mock",
    "mocks",
    "works",
    "behavior",
    "integration",
    "regression",
    "case",
}

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WARNING_SOURCES = (
    ROOT_DIR / "full_test_results.txt",
    ROOT_DIR / "pytest_warnings.txt",
)


@dataclass
class Finding:
    finding_id: str
    file: str
    line: int
    test: str
    detector: str
    severity: str
    claimed_behavior: str
    why_hollow: str
    model_path: bool
    report_only: bool
    suggested_remedy: str
    source_layer: str
    model_refs: list[str] = field(default_factory=list)

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FrozenPolicy:
    markers: list[str]
    evidence_files: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only test-trust auditor (v1: L0+L1).")
    parser.add_argument("--mode", choices=["static"], default="static")
    parser.add_argument("--out", default="reports/", help="Output directory for markdown/json reports.")
    parser.add_argument("--tests-root", default="tests", help="Root directory for test AST scan.")
    parser.add_argument("--as-of", dest="as_of", default=None, help="Audit date (YYYY-MM-DD).")
    parser.add_argument(
        "--warnings-input",
        default=None,
        help="Optional warning text file to parse for L0 (T1/T11).",
    )
    parser.add_argument("--enable-t8", action="store_true", help="Stub only in v1 (disabled).")
    parser.add_argument("--enable-t9", action="store_true", help="Stub only in v1 (disabled).")
    parser.add_argument("--enable-t10", action="store_true", help="Stub only in v1 (disabled).")
    parser.add_argument("--enable-l3", action="store_true", help="Stub only in v1 (disabled).")
    return parser.parse_args()


def _run_command(cmd: list[str], cwd: Path) -> str:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    return (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")


def derive_default_as_of_date(repo_root: Path) -> str:
    source_date_epoch = _read_source_date_epoch()
    if source_date_epoch:
        return source_date_epoch
    cmd = ["git", "show", "-s", "--format=%cs", "HEAD"]
    output = _run_command(cmd, repo_root).strip().splitlines()
    if output:
        first = output[0].strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", first):
            return first
    return "1970-01-01"


def _read_source_date_epoch() -> Optional[str]:
    raw = None
    try:
        raw = __import__("os").environ.get("SOURCE_DATE_EPOCH")
    except Exception:
        raw = None
    if not raw:
        return None
    try:
        ts = int(raw)
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
    except (ValueError, OSError):
        return None


def validate_as_of(value: str) -> str:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError(f"--as-of must be YYYY-MM-DD, got: {value}")
    return value


def load_frozen_policy(repo_root: Path) -> FrozenPolicy:
    evidence_files: list[str] = []
    texts: list[str] = []

    claude_path = repo_root / "CLAUDE.md"
    if claude_path.exists():
        texts.append(claude_path.read_text(encoding="utf-8"))
        evidence_files.append(str(claude_path.relative_to(repo_root)))

    governance_dir = repo_root / "governance"
    if governance_dir.exists():
        for path in sorted(governance_dir.rglob("*.md")):
            texts.append(path.read_text(encoding="utf-8"))
            evidence_files.append(str(path.relative_to(repo_root)))

    corpus = "\n".join(texts).lower()
    marker_candidates = {
        "decision_engine",
        "selector",
        "ranker",
        "sizing",
        "final_score",
        "portfolio",
        "snapshot",
        "scoring",
        "rankings.csv",
    }
    markers = {m for m in marker_candidates if m in corpus}
    for token in re.findall(r"[a-z0-9_./-]+\.(?:py|csv)", corpus):
        if any(x in token for x in ("decision_engine", "selector", "ranker", "snapshot", "portfolio", "score")):
            markers.add(token)
            if token.endswith(".py"):
                markers.add(token[:-3])
    return FrozenPolicy(markers=sorted(markers), evidence_files=evidence_files)


def dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = dotted_name(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    return ""


def split_identifier(value: str) -> list[str]:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    tokens = re.split(r"[^a-zA-Z0-9]+", value.lower())
    return [t for t in tokens if t]


def claim_from_test_name(test_name: str) -> str:
    parts = [p for p in split_identifier(test_name.replace("::", "_")) if p != "test"]
    if not parts:
        return "unspecified behavior"
    return " ".join(parts[:12])


def normalize_test_ref(file_path: str, test: str) -> str:
    return f"{file_path}::{test}"


def collect_test_functions(tree: ast.Module) -> list[tuple[Optional[str], ast.AST]]:
    out: list[tuple[Optional[str], ast.AST]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            out.append((None, node))
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and member.name.startswith("test_"):
                    out.append((node.name, member))
    return out


def build_test_index(tests_root: Path, repo_root: Path) -> dict[str, int]:
    index: dict[str, int] = {}
    for file_path in sorted(tests_root.rglob("test_*.py")):
        rel = str(file_path.relative_to(repo_root))
        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=rel)
        except SyntaxError:
            continue
        for class_name, fn in collect_test_functions(tree):
            test_name = f"{class_name}::{fn.name}" if class_name else fn.name
            index[normalize_test_ref(rel, test_name)] = fn.lineno
            index[normalize_test_ref(rel, fn.name)] = fn.lineno
    return index


def iter_assert_calls(fn: ast.AST) -> Iterable[ast.Call]:
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            fname = dotted_name(node.func)
            if ".assert" in fname or fname.startswith("assert"):
                yield node


def has_pytest_raises(fn: ast.AST) -> bool:
    for node in ast.walk(fn):
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                call = item.context_expr
                if isinstance(call, ast.Call):
                    fname = dotted_name(call.func)
                    if fname.endswith("pytest.raises") or fname.endswith("raises"):
                        return True
    return False


def has_fail_call(fn: ast.AST) -> bool:
    """True if the test uses pytest.fail()/self.fail() as an explicit failure primitive.

    These are effective assertions -- they fail the test on the negative path -- even
    though they are neither ``assert`` statements, ``assert*`` calls, nor
    ``pytest.raises`` blocks. Without this, contract tests that branch and then call
    ``pytest.fail(...)`` (sort-order contracts, import barriers, golden-baseline
    mismatch) are misreported as T2 no-effective-assert.
    """
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            fname = dotted_name(node.func)
            if fname in {"pytest.fail", "self.fail", "fail"}:
                return True
    return False


def is_unconditionally_skipped(fn: ast.AST) -> bool:
    """True if the test carries an unconditional skip/xfail (or skipif(True)) decorator.

    Such a test makes no passing claim, so T2/T3 -- which detect tests that run and
    fake a pass -- do not apply. T7 already accounts for the suppressed execution, so
    routing these to T7 only avoids double-flagging an honestly-declared gap.
    """
    for dec in getattr(fn, "decorator_list", []):
        dec_name = dotted_name(dec.func) if isinstance(dec, ast.Call) else dotted_name(dec)
        if dec_name.endswith(".skip") or dec_name.endswith(".xfail"):
            return True
        if isinstance(dec, ast.Call) and dec_name.endswith(".skipif") and dec.args:
            cond = dec.args[0]
            if isinstance(cond, ast.Constant) and cond.value is True:
                return True
    return False


def is_constant_truthy_assert(expr: ast.AST) -> bool:
    if isinstance(expr, ast.Constant):
        return bool(expr.value) is True
    if isinstance(expr, ast.Compare) and len(expr.ops) == 1 and len(expr.comparators) == 1:
        if isinstance(expr.ops[0], ast.Eq):
            return ast.dump(expr.left, include_attributes=False) == ast.dump(
                expr.comparators[0], include_attributes=False
            )
    return False


def _assignment_map(fn: ast.AST) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            assignments[node.targets[0].id] = ast.dump(node.value, include_attributes=False)
    return assignments


def detect_t3_tautological_assertions(
    rel_path: str,
    test_name: str,
    fn: ast.AST,
) -> list[Finding]:
    findings: list[Finding] = []
    assigns = _assignment_map(fn)
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assert):
            continue
        expr = node.test
        if not (isinstance(expr, ast.Compare) and len(expr.ops) == 1 and isinstance(expr.ops[0], ast.Eq)):
            continue
        if len(expr.comparators) != 1:
            continue
        left = expr.left
        right = expr.comparators[0]
        left_dump = ast.dump(left, include_attributes=False)
        right_dump = ast.dump(right, include_attributes=False)
        same_structure = left_dump == right_dump
        same_recomputed = False
        if isinstance(left, ast.Name) and assigns.get(left.id) == right_dump:
            same_recomputed = True
        if isinstance(right, ast.Name) and assigns.get(right.id) == left_dump:
            same_recomputed = True
        if isinstance(left, ast.Name) and isinstance(right, ast.Name):
            same_recomputed = assigns.get(left.id) == assigns.get(right.id) and assigns.get(left.id) is not None
        if not (same_structure or same_recomputed):
            continue
        findings.append(
            Finding(
                finding_id="",
                file=rel_path,
                line=node.lineno,
                test=test_name,
                detector="T3",
                severity="HIGH",
                claimed_behavior=claim_from_test_name(test_name),
                why_hollow="Assertion compares structurally equivalent values/recomputed equivalent expression.",
                model_path=False,
                report_only=False,
                suggested_remedy="Assert against an independently-derived oracle or fixture-backed expected value.",
                source_layer="L1",
            )
        )
    return findings


def _extract_patch_target(call: ast.Call) -> Optional[str]:
    fname = dotted_name(call.func)
    if not (
        fname.endswith(".patch")
        or fname == "patch"
        or fname.endswith(".patch.object")
        or fname.endswith("setattr")
        or fname.endswith(".Mock")
        or fname.endswith(".MagicMock")
    ):
        return None
    if call.args:
        first = call.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value
        if isinstance(first, (ast.Name, ast.Attribute)):
            return dotted_name(first)
    return None


def _derive_subject_tokens(rel_path: str, test_name: str) -> set[str]:
    file_stem = Path(rel_path).stem
    file_stem = file_stem[5:] if file_stem.startswith("test_") else file_stem
    tokens = set(split_identifier(file_stem))
    tokens.update(split_identifier(test_name))
    return {t for t in tokens if len(t) >= 4 and t not in GENERIC_SUBJECT_TOKENS}


def _path_from_patch_target(target: str) -> Optional[str]:
    if "/" in target or target.endswith(".py"):
        return target
    module = target.split(":", 1)[0].split(".", 1)[0]
    if not module:
        return None
    return f"{module}.py"


def detect_t4_mock_of_subject(rel_path: str, test_name: str, fn: ast.AST) -> list[Finding]:
    findings: list[Finding] = []
    tokens = _derive_subject_tokens(rel_path, test_name)
    if not tokens:
        return findings
    seen: set[tuple[str, int]] = set()

    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            target = _extract_patch_target(node)
            if not target:
                continue
            target_l = target.lower()
            if not any(token in target_l for token in tokens):
                continue
            key = (target, node.lineno)
            if key in seen:
                continue
            seen.add(key)
            ref_path = _path_from_patch_target(target)
            model_refs = [target]
            if ref_path:
                model_refs.append(ref_path)
            findings.append(
                Finding(
                    finding_id="",
                    file=rel_path,
                    line=node.lineno,
                    test=test_name,
                    detector="T4",
                    severity="CRITICAL",
                    claimed_behavior=claim_from_test_name(test_name),
                    why_hollow=f"Patch/Mock target overlaps subject under test ({target}).",
                    model_path=False,
                    report_only=False,
                    suggested_remedy="Patch external dependencies only; instantiate/execute the real subject module.",
                    source_layer="L1",
                    model_refs=model_refs,
                )
            )
    return findings


def _contains_assert(nodes: list[ast.stmt]) -> Optional[int]:
    for stmt in nodes:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Assert):
                return node.lineno
    return None


def _is_bare_or_broad_exception(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    if isinstance(handler.type, ast.Name):
        return handler.type.id in {"Exception", "BaseException"}
    if isinstance(handler.type, ast.Tuple):
        names = {elt.id for elt in handler.type.elts if isinstance(elt, ast.Name)}
        return bool(names.intersection({"Exception", "BaseException"}))
    return False


def detect_t5_swallowed_failure(rel_path: str, test_name: str, fn: ast.AST) -> list[Finding]:
    findings: list[Finding] = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Try):
            assert_line = _contains_assert(node.body)
            if assert_line is not None:
                for handler in node.handlers:
                    if _is_bare_or_broad_exception(handler):
                        suppresses = not any(isinstance(x, ast.Raise) for x in ast.walk(handler))
                        if suppresses:
                            findings.append(
                                Finding(
                                    finding_id="",
                                    file=rel_path,
                                    line=handler.lineno,
                                    test=test_name,
                                    detector="T5",
                                    severity="HIGH",
                                    claimed_behavior=claim_from_test_name(test_name),
                                    why_hollow="Broad/bare except can swallow AssertionError from asserted branch.",
                                    model_path=False,
                                    report_only=False,
                                    suggested_remedy="Catch specific expected exceptions and re-raise/assert on unexpected failures.",
                                    source_layer="L1",
                                )
                            )
        if isinstance(node, ast.If) and isinstance(node.test, ast.Constant) and isinstance(node.test.value, bool):
            if node.test.value is False:
                line = _contains_assert(node.body)
                if line is not None:
                    findings.append(
                        Finding(
                            finding_id="",
                            file=rel_path,
                            line=line,
                            test=test_name,
                            detector="T5",
                            severity="HIGH",
                            claimed_behavior=claim_from_test_name(test_name),
                            why_hollow="Assertion is unreachable (guarded by constant-false branch).",
                            model_path=False,
                            report_only=False,
                            suggested_remedy="Move assertions onto reachable execution paths with explicit preconditions.",
                            source_layer="L1",
                        )
                    )
            if node.test.value is True:
                line = _contains_assert(node.orelse)
                if line is not None:
                    findings.append(
                        Finding(
                            finding_id="",
                            file=rel_path,
                            line=line,
                            test=test_name,
                            detector="T5",
                            severity="HIGH",
                            claimed_behavior=claim_from_test_name(test_name),
                            why_hollow="Assertion is unreachable (in else-branch of constant-true guard).",
                            model_path=False,
                            report_only=False,
                            suggested_remedy="Remove dead branches and keep assertions in executed control flow.",
                            source_layer="L1",
                        )
                    )
    return findings


def detect_t6_vacuous_parametrize(rel_path: str, test_name: str, fn: ast.AST) -> list[Finding]:
    findings: list[Finding] = []
    decorators = getattr(fn, "decorator_list", [])
    for dec in decorators:
        if not isinstance(dec, ast.Call):
            continue
        dname = dotted_name(dec.func)
        if not dname.endswith("parametrize"):
            continue
        if len(dec.args) < 2:
            continue
        argvalues = dec.args[1]
        if isinstance(argvalues, (ast.List, ast.Tuple, ast.Set)):
            values = list(argvalues.elts)
            if len(values) == 0:
                reason = "Parametrize argvalues is empty."
            elif len(values) == 1:
                reason = "Parametrize argvalues has a single case (degenerate)."
            else:
                dumps = [ast.dump(v, include_attributes=False) for v in values]
                reason = "Parametrize argvalues collapse to repeated equivalent values." if len(set(dumps)) == 1 else ""
            if reason:
                findings.append(
                    Finding(
                        finding_id="",
                        file=rel_path,
                        line=dec.lineno,
                        test=test_name,
                        detector="T6",
                        severity="MEDIUM",
                        claimed_behavior=claim_from_test_name(test_name),
                        why_hollow=reason,
                        model_path=False,
                        report_only=False,
                        suggested_remedy="Add diverse, behavior-distinguishing parameter sets.",
                        source_layer="L1",
                    )
                )
    return findings


def detect_t7_silent_skip(rel_path: str, test_name: str, fn: ast.AST) -> list[Finding]:
    findings: list[Finding] = []
    decorators = getattr(fn, "decorator_list", [])
    for dec in decorators:
        dec_name = dotted_name(dec.func) if isinstance(dec, ast.Call) else dotted_name(dec)
        dec_line = getattr(dec, "lineno", getattr(fn, "lineno", 1))
        if dec_name.endswith(".skip") or dec_name.endswith(".xfail"):
            findings.append(
                Finding(
                    finding_id="",
                    file=rel_path,
                    line=dec_line,
                    test=test_name,
                    detector="T7",
                    severity="MEDIUM",
                    claimed_behavior=claim_from_test_name(test_name),
                    why_hollow="Unconditional skip/xfail decorator suppresses behavioral execution.",
                    model_path=False,
                    report_only=False,
                    suggested_remedy="Gate skip/xfail with explicit temporary condition and expiry ticket.",
                    source_layer="L1",
                )
            )
        if isinstance(dec, ast.Call) and dec_name.endswith(".skipif") and dec.args:
            cond = dec.args[0]
            if isinstance(cond, ast.Constant) and cond.value is True:
                findings.append(
                    Finding(
                        finding_id="",
                        file=rel_path,
                        line=dec_line,
                        test=test_name,
                        detector="T7",
                        severity="MEDIUM",
                        claimed_behavior=claim_from_test_name(test_name),
                        why_hollow="skipif(True) unconditionally suppresses test execution.",
                        model_path=False,
                        report_only=False,
                        suggested_remedy="Use a concrete, temporary condition and annotate removal criteria.",
                        source_layer="L1",
                    )
                )
    return findings


def _is_len_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and dotted_name(node.func) == "len"


def _is_type_or_isinstance_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    name = dotted_name(node.func)
    return name in {"type", "isinstance"}


def is_weak_assert_expr(expr: ast.AST) -> bool:
    if isinstance(expr, ast.Compare):
        if len(expr.ops) != 1 or len(expr.comparators) != 1:
            return False
        op = expr.ops[0]
        right = expr.comparators[0]
        if isinstance(op, ast.IsNot) and isinstance(right, ast.Constant) and right.value is None:
            return True
        if isinstance(op, (ast.Gt, ast.GtE, ast.NotEq, ast.Eq, ast.Lt, ast.LtE)):
            return _is_len_call(expr.left) or _is_len_call(right) or _is_type_or_isinstance_call(expr.left)
    if _is_type_or_isinstance_call(expr):
        return True
    return False


def detect_t12_broad_snapshot(rel_path: str, test_name: str, fn: ast.AST) -> list[Finding]:
    assert_nodes = [n for n in ast.walk(fn) if isinstance(n, ast.Assert)]
    assert_calls = list(iter_assert_calls(fn))
    if has_pytest_raises(fn):
        return []
    if not assert_nodes and not assert_calls:
        return []

    weak_count = 0
    strong_count = 0
    for node in assert_nodes:
        if is_constant_truthy_assert(node.test):
            continue
        if is_weak_assert_expr(node.test):
            weak_count += 1
        else:
            strong_count += 1

    weak_call_methods = {"assertIsNotNone", "assertIsInstance"}
    for call in assert_calls:
        fname = dotted_name(call.func)
        method = fname.split(".")[-1] if fname else ""
        if method in weak_call_methods:
            weak_count += 1
        elif method.startswith("assert"):
            strong_count += 1

    if weak_count > 0 and strong_count == 0:
        line = assert_nodes[0].lineno if assert_nodes else assert_calls[0].lineno
        return [
            Finding(
                finding_id="",
                file=rel_path,
                line=line,
                test=test_name,
                detector="T12",
                severity="LOW-MEDIUM",
                claimed_behavior=claim_from_test_name(test_name),
                why_hollow="Assertions only validate non-null/len/type shape, not semantic content.",
                model_path=False,
                report_only=False,
                suggested_remedy="Add field/value-level assertions tied to expected behavioral invariants.",
                source_layer="L1",
            )
        ]
    return []


def detect_t2_no_effective_assert(rel_path: str, test_name: str, fn: ast.AST) -> list[Finding]:
    asserts = [n for n in ast.walk(fn) if isinstance(n, ast.Assert)]
    calls = list(iter_assert_calls(fn))
    has_raises = has_pytest_raises(fn)
    has_fail = has_fail_call(fn)
    if not asserts and not calls and not has_raises and not has_fail:
        return [
            Finding(
                finding_id="",
                file=rel_path,
                line=getattr(fn, "lineno", 1),
                test=test_name,
                detector="T2",
                severity="HIGH",
                claimed_behavior=claim_from_test_name(test_name),
                why_hollow="No effective assertion primitive found.",
                model_path=False,
                report_only=False,
                suggested_remedy="Add behavior-verifying assertions or explicit pytest.raises expectations.",
                source_layer="L1",
            )
        ]

    non_constant_asserts = [a for a in asserts if not is_constant_truthy_assert(a.test)]
    if not non_constant_asserts and not calls and not has_raises and not has_fail and asserts:
        return [
            Finding(
                finding_id="",
                file=rel_path,
                line=asserts[0].lineno,
                test=test_name,
                detector="T2",
                severity="HIGH",
                claimed_behavior=claim_from_test_name(test_name),
                why_hollow="Only constant-truth assertions found.",
                model_path=False,
                report_only=False,
                suggested_remedy="Replace placeholder asserts with checks against expected outcomes.",
                source_layer="L1",
            )
        ]
    return []


def analyze_source_text(
    source: str,
    rel_path: str,
    frozen_markers: Optional[list[str]] = None,
) -> tuple[list[Finding], int]:
    frozen_markers = frozen_markers or []
    findings: list[Finding] = []
    try:
        tree = ast.parse(source, filename=rel_path)
    except SyntaxError as exc:
        findings.append(
            Finding(
                finding_id="",
                file=rel_path,
                line=exc.lineno or 1,
                test="<module>",
                detector="UNCERTAIN",
                severity="UNCERTAIN",
                claimed_behavior="unparsed test module",
                why_hollow=f"Unable to parse test file: {exc.msg}",
                model_path=False,
                report_only=False,
                suggested_remedy="Fix syntax/import-time parse errors so static analysis can fail closed.",
                source_layer="L1",
            )
        )
        return findings, 0

    test_fns = collect_test_functions(tree)
    for class_name, fn in test_fns:
        test_name = f"{class_name}::{fn.name}" if class_name else fn.name
        if not is_unconditionally_skipped(fn):
            findings.extend(detect_t2_no_effective_assert(rel_path, test_name, fn))
            findings.extend(detect_t3_tautological_assertions(rel_path, test_name, fn))
        findings.extend(detect_t4_mock_of_subject(rel_path, test_name, fn))
        findings.extend(detect_t5_swallowed_failure(rel_path, test_name, fn))
        findings.extend(detect_t6_vacuous_parametrize(rel_path, test_name, fn))
        findings.extend(detect_t7_silent_skip(rel_path, test_name, fn))
        findings.extend(detect_t12_broad_snapshot(rel_path, test_name, fn))

    apply_model_path_flags(findings, frozen_markers)
    return dedupe_findings(findings), len(test_fns)


def apply_model_path_flags(findings: list[Finding], frozen_markers: list[str]) -> None:
    markers = [m.lower() for m in frozen_markers]
    for finding in findings:
        refs = list(finding.model_refs)
        if not finding.file.startswith("tests/"):
            refs.append(finding.file)
        text = " ".join(refs).lower()
        finding.model_path = any(marker in text for marker in markers) if markers else False
        finding.report_only = finding.model_path


def dedupe_findings(findings: list[Finding]) -> list[Finding]:
    seen: set[tuple[str, str, int, str, str]] = set()
    deduped: list[Finding] = []
    for finding in findings:
        key = (finding.detector, finding.file, finding.line, finding.test, finding.why_hollow)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(finding)
    return deduped


def parse_l0_warnings(
    warning_text: str,
    test_index: dict[str, int],
    frozen_markers: list[str],
) -> list[Finding]:
    findings: list[Finding] = []
    lines = warning_text.splitlines()
    nodeid_re = re.compile(r"^(tests/[^\s:]+::[^\s]+)$")
    return_re = re.compile(r"but (tests/[^\s]+::[^\s]+) returned")
    pending_nodeids: list[str] = []

    for line in lines:
        stripped = line.strip()
        nodeid_match = nodeid_re.match(stripped)
        if nodeid_match:
            pending_nodeids.append(nodeid_match.group(1))
            continue

        if "PytestReturnNotNoneWarning" in line:
            nodeid = None
            m = return_re.search(line)
            if m:
                nodeid = m.group(1)
            elif pending_nodeids:
                nodeid = pending_nodeids[-1]
            if nodeid:
                rel_path, test_name = split_nodeid(nodeid)
                line_no = test_index.get(normalize_test_ref(rel_path, test_name), 1)
                findings.append(
                    Finding(
                        finding_id="",
                        file=rel_path,
                        line=line_no,
                        test=test_name,
                        detector="T1",
                        severity="HIGH",
                        claimed_behavior=claim_from_test_name(test_name),
                        why_hollow="PytestReturnNotNoneWarning indicates test returned a value instead of asserting behavior.",
                        model_path=False,
                        report_only=False,
                        suggested_remedy="Replace return values with explicit assertions over expected outcomes.",
                        source_layer="L0",
                    )
                )
            pending_nodeids = []
            continue

        if "RuntimeWarning" in line and "was never awaited" in line:
            grouped = sorted(set(pending_nodeids))
            if grouped:
                rel_path, test_name = split_nodeid(grouped[0])
                line_no = test_index.get(normalize_test_ref(rel_path, test_name), 1)
                why = "Coroutine/AsyncMock was never awaited in warning capture."
                if len(grouped) > 1:
                    why += f" Warning group references {len(grouped)} tests."
                findings.append(
                    Finding(
                        finding_id="",
                        file=rel_path,
                        line=line_no,
                        test=test_name if len(grouped) == 1 else f"{test_name} (+{len(grouped)-1} grouped)",
                        detector="T11",
                        severity="MEDIUM",
                        claimed_behavior=claim_from_test_name(test_name),
                        why_hollow=why,
                        model_path=False,
                        report_only=False,
                        suggested_remedy="Await coroutine-producing mocks and assert awaited call contracts explicitly.",
                        source_layer="L0",
                    )
                )
            else:
                findings.append(
                    Finding(
                        finding_id="",
                        file="<warnings>",
                        line=1,
                        test="<unknown>",
                        detector="T11",
                        severity="MEDIUM",
                        claimed_behavior="async coroutine contract",
                        why_hollow="Detected never-awaited coroutine warning with no resolvable nodeid context.",
                        model_path=False,
                        report_only=False,
                        suggested_remedy="Capture nodeids in warning harvest and enforce awaited async interactions.",
                        source_layer="L0",
                    )
                )
            pending_nodeids = []
            continue

        if stripped and not stripped.startswith("/home/") and not stripped.startswith("tests/"):
            pending_nodeids = []

    apply_model_path_flags(findings, frozen_markers)
    return dedupe_findings(findings)


def split_nodeid(nodeid: str) -> tuple[str, str]:
    parts = nodeid.split("::")
    file_path = parts[0]
    test_name = "::".join(parts[1:]) if len(parts) > 1 else "<unknown>"
    return file_path, test_name


def load_warning_text(repo_root: Path, explicit: Optional[str]) -> tuple[str, str]:
    if explicit:
        path = Path(explicit)
        text = path.read_text(encoding="utf-8")
        return text, str(path)

    for source in DEFAULT_WARNING_SOURCES:
        if source.exists():
            return source.read_text(encoding="utf-8"), str(source.relative_to(repo_root))

    cmd = ["pytest", "--collect-only", "-q", "-rw"]
    output = _run_command(cmd, repo_root)
    return output, "pytest --collect-only -q -rw"


def analyze_tests_ast(tests_root: Path, repo_root: Path, frozen_markers: list[str]) -> tuple[list[Finding], int]:
    findings: list[Finding] = []
    total_tests = 0
    for file_path in sorted(tests_root.rglob("test_*.py")):
        rel_path = str(file_path.relative_to(repo_root))
        source = file_path.read_text(encoding="utf-8")
        file_findings, test_count = analyze_source_text(source, rel_path, frozen_markers=frozen_markers)
        findings.extend(file_findings)
        total_tests += test_count
    return dedupe_findings(findings), total_tests


def sort_findings(findings: list[Finding]) -> list[Finding]:
    return sorted(
        findings,
        key=lambda f: (
            SEVERITY_ORDER.get(f.severity, 99),
            f.file,
            f.line,
            f.detector,
            f.test,
        ),
    )


def assign_finding_ids(findings: list[Finding]) -> None:
    counters: dict[str, int] = defaultdict(int)
    for finding in findings:
        counters[finding.detector] += 1
        finding.finding_id = f"{finding.detector}-{counters[finding.detector]:04d}"


def compute_trust_score(findings: list[Finding], tests_analyzed: int) -> tuple[float, float]:
    weighted = sum(SEVERITY_WEIGHTS.get(f.severity, 2.0) for f in findings)
    density = weighted / max(tests_analyzed, 1)
    trust_score = max(0.0, round(100.0 - (density * 100.0), 2))
    return trust_score, round(density, 6)


def compute_findings_sha(findings: list[Finding]) -> str:
    canonical = [f.as_json() for f in findings]
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def severity_counts(findings: list[Finding]) -> dict[str, int]:
    counts = Counter(f.severity for f in findings)
    return {name: counts.get(name, 0) for name in ["CRITICAL", "HIGH", "MEDIUM", "LOW-MEDIUM", "UNCERTAIN"]}


def locate_prior_json(out_dir: Path, current_name: str) -> Optional[Path]:
    candidates = sorted(out_dir.glob("test_trust_audit_*.json"))
    for candidate in reversed(candidates):
        if candidate.name != current_name:
            return candidate
    return None


def read_prior_delta(prior_path: Optional[Path], current_total: int, current_trust: float) -> dict[str, Any]:
    if not prior_path or not prior_path.exists():
        return {
            "prior_file": None,
            "delta_findings": None,
            "delta_trust_score": None,
        }
    try:
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"prior_file": str(prior_path), "delta_findings": None, "delta_trust_score": None}
    prior_total = int(prior.get("summary", {}).get("total_findings", 0))
    prior_trust = float(prior.get("summary", {}).get("trust_score", 0.0))
    return {
        "prior_file": str(prior_path.name),
        "delta_findings": current_total - prior_total,
        "delta_trust_score": round(current_trust - prior_trust, 2),
    }


def render_markdown(
    out_path: Path,
    as_of_date: str,
    findings: list[Finding],
    counts: dict[str, int],
    trust_score: float,
    delta: dict[str, Any],
    model_report_only_count: int,
) -> None:
    reportable = [f for f in findings if not f.report_only]
    model_only = [f for f in findings if f.report_only]
    lines = [
        "TEST_TRUST_AUDIT_EXECUTIVE",
        f"as_of_date: {as_of_date}",
        f"total_findings: {len(findings)}",
        (
            "severity_counts: "
            f"CRITICAL={counts['CRITICAL']} HIGH={counts['HIGH']} MEDIUM={counts['MEDIUM']} "
            f"LOW-MEDIUM={counts['LOW-MEDIUM']} UNCERTAIN={counts['UNCERTAIN']}"
        ),
        f"trust_score: {trust_score:.2f}",
        (
            "delta_vs_prior: "
            + (
                "n/a"
                if delta["delta_findings"] is None
                else f"findings={delta['delta_findings']:+d}, trust_score={delta['delta_trust_score']:+.2f}"
            )
        ),
        f"model_path_report_only_findings: {model_report_only_count}",
        "",
        "## Findings Ledger (severity sorted)",
        "",
        "| finding_id | file:line | test | detector | severity | claimed_behavior | why_hollow | model_path | suggested_remedy |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for finding in reportable:
        lines.append(
            "| {fid} | {loc} | {test} | {det} | {sev} | {claim} | {why} | {model} | {fix} |".format(
                fid=finding.finding_id,
                loc=f"{finding.file}:{finding.line}",
                test=finding.test,
                det=finding.detector,
                sev=finding.severity,
                claim=finding.claimed_behavior.replace("|", "/"),
                why=finding.why_hollow.replace("|", "/"),
                model=str(finding.model_path).lower(),
                fix=finding.suggested_remedy.replace("|", "/"),
            )
        )
    lines.extend(
        [
            "",
            "## Report-only Findings on Frozen Model Paths",
            "",
            "These are advisory findings on frozen model paths. They are report-only in v1.",
            "",
            "| finding_id | file:line | test | detector | severity | claimed_behavior | why_hollow | model_path | suggested_remedy |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for finding in model_only:
        lines.append(
            "| {fid} | {loc} | {test} | {det} | {sev} | {claim} | {why} | {model} | {fix} |".format(
                fid=finding.finding_id,
                loc=f"{finding.file}:{finding.line}",
                test=finding.test,
                det=finding.detector,
                sev=finding.severity,
                claim=finding.claimed_behavior.replace("|", "/"),
                why=finding.why_hollow.replace("|", "/"),
                model=str(finding.model_path).lower(),
                fix=finding.suggested_remedy.replace("|", "/"),
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_json_payload(
    *,
    as_of_date: str,
    mode: str,
    findings: list[Finding],
    tests_analyzed: int,
    trust_score: float,
    density: float,
    findings_sha: str,
    delta: dict[str, Any],
    warning_source: str,
    frozen_policy: FrozenPolicy,
) -> dict[str, Any]:
    counts = severity_counts(findings)
    model_report_only = sum(1 for f in findings if f.report_only)
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "as_of_date": as_of_date,
        "disabled_stubs": DISABLED_STUBS,
        "summary": {
            "total_findings": len(findings),
            "severity_counts": counts,
            "tests_analyzed": tests_analyzed,
            "weighted_hollow_density": density,
            "trust_score": trust_score,
            "model_path_report_only_findings": model_report_only,
            "delta_vs_prior": delta,
        },
        "sources": {
            "l0_warning_source": warning_source,
            "frozen_policy_evidence_files": frozen_policy.evidence_files,
            "frozen_policy_markers": frozen_policy.markers,
        },
        "findings": [f.as_json() for f in findings],
        "findings_sha256": findings_sha,
    }


def ensure_stub_flags_disabled(args: argparse.Namespace) -> None:
    requested = []
    if args.enable_t8:
        requested.append("T8")
    if args.enable_t9:
        requested.append("T9")
    if args.enable_t10:
        requested.append("T10")
    if args.enable_l3:
        requested.append("L3")
    if requested:
        requested_str = ", ".join(requested)
        raise SystemExit(f"{requested_str} requested, but disabled in v1 (L0 + L1 only).")


def run_audit(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any]]:
    ensure_stub_flags_disabled(args)
    repo_root = ROOT_DIR
    tests_root = (repo_root / args.tests_root).resolve()
    if not tests_root.exists():
        raise SystemExit(f"tests root does not exist: {tests_root}")

    as_of_date = validate_as_of(args.as_of) if args.as_of else derive_default_as_of_date(repo_root)
    out_dir = (repo_root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    frozen_policy = load_frozen_policy(repo_root)
    test_index = build_test_index(tests_root, repo_root)
    l1_findings, total_tests = analyze_tests_ast(tests_root, repo_root, frozen_policy.markers)
    warning_text, warning_source = load_warning_text(repo_root, args.warnings_input)
    l0_findings = parse_l0_warnings(warning_text, test_index, frozen_policy.markers)

    findings = sort_findings(dedupe_findings(l0_findings + l1_findings))
    assign_finding_ids(findings)
    trust_score, density = compute_trust_score(findings, total_tests)
    findings_sha = compute_findings_sha(findings)

    json_name = f"test_trust_audit_{as_of_date}.json"
    md_name = f"TEST_TRUST_AUDIT_{as_of_date}.md"
    prior = locate_prior_json(out_dir, current_name=json_name)
    delta = read_prior_delta(prior, len(findings), trust_score)

    payload = build_json_payload(
        as_of_date=as_of_date,
        mode=args.mode,
        findings=findings,
        tests_analyzed=total_tests,
        trust_score=trust_score,
        density=density,
        findings_sha=findings_sha,
        delta=delta,
        warning_source=warning_source,
        frozen_policy=frozen_policy,
    )
    json_path = out_dir / json_name
    md_path = out_dir / md_name
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    render_markdown(
        md_path,
        as_of_date=as_of_date,
        findings=findings,
        counts=severity_counts(findings),
        trust_score=trust_score,
        delta=delta,
        model_report_only_count=payload["summary"]["model_path_report_only_findings"],
    )
    return md_path, json_path, payload


def main() -> int:
    args = parse_args()
    md_path, json_path, payload = run_audit(args)
    print(
        json.dumps(
            {
                "markdown_report": str(md_path),
                "json_report": str(json_path),
                "total_findings": payload["summary"]["total_findings"],
                "trust_score": payload["summary"]["trust_score"],
                "findings_sha256": payload["findings_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
