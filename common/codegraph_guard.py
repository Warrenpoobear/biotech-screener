"""Codegraph acceptance-gate wrapper for Hermes agents.

Implements the five acceptance conditions required before codegraph is
registered as a Hermes MCP dependency (see docs/CODEGRAPH_RUNBOOK.md):

  1. Dynamic-dispatch break  → warn + fallback, do not halt silently
  2. Ambiguous symbol        → require file-qualified disambiguation
  3. File-path literal       → automatic grep/read fallback recommendation
  4. Cron/shell boundary     → explicit non-graph verification warning
  5. Partial graph proof     → emit [PARTIAL PROOF] warning; no hallucinated edges

Usage (Hermes agent):

    from common.codegraph_guard import CodegraphGuard

    cg = CodegraphGuard()
    result = cg.callers("save_validation_snapshot")
    if not result.is_trustworthy:
        # heed result.warnings before drawing conclusions
        ...
    print(result.output)
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FILE_PATH_RE = re.compile(
    r"[/\\]|"
    r"\.(py|json|csv|yaml|yml|txt|md|sh|toml|cfg|ini|lock)$",
    re.IGNORECASE,
)

_DYNAMIC_DISPATCH_PHRASES = [
    "dynamic dispatch",
    "dynamic-dispatch",
    "cannot trace",
    "break point",
    "callback",
    "indirect call",
]

_CRON_SHELL_PHRASES = [
    "subprocess",
    "os.system",
    "os.popen",
    "crontab",
    "shell=true",
    "shlex",
    "popen",
]

# Production surfaces: if impact reaches these, gate fires.
TIER3_SURFACES = frozenset(
    [
        "selector_engine",
        "ranker_engine",
        "decision_engine",
        "final_score",
        "rankings.csv",
        "run_screen_columns",
        "save_validation_snapshot",
        "_write_snapshot",
    ]
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class ProofConfidence(Enum):
    FULL = "full"
    PARTIAL = "partial"
    UNVERIFIED = "unverified"


@dataclass
class CodegraphResult:
    query: str
    command: str
    output: str
    confidence: ProofConfidence
    warnings: list[str] = field(default_factory=list)
    fallback_instructions: list[str] = field(default_factory=list)
    exit_code: int = 0

    @property
    def is_trustworthy(self) -> bool:
        return self.confidence == ProofConfidence.FULL and not self.warnings

    def format(self) -> str:
        lines = [
            f"=== codegraph_guard [{self.command}] '{self.query}' "
            f"[{self.confidence.value.upper()}] ==="
        ]
        for w in self.warnings:
            lines.append(f"  ⚠  {w}")
        if self.fallback_instructions:
            lines.append("  Fallback steps:")
            for instr in self.fallback_instructions:
                lines.append(f"    → {instr}")
        if self.output.strip():
            lines.append(self.output)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Guard class
# ---------------------------------------------------------------------------


class CodegraphGuard:
    """Wrapper around the codegraph CLI that enforces all five acceptance gates."""

    def __init__(self, timeout: int = 30):
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, symbol: str, file_hint: Optional[str] = None) -> CodegraphResult:
        """Search for a symbol definition. Gates: file-path literal, ambiguous."""
        path_warn = self._gate_file_path(symbol)
        if path_warn:
            return CodegraphResult(
                query=symbol,
                command="query",
                output="",
                confidence=ProofConfidence.UNVERIFIED,
                warnings=path_warn,
                fallback_instructions=[f"rg '{symbol}' --type py"],
            )

        args = ["query", symbol]
        if file_hint:
            args += ["--filter", file_hint]
        output, code = self._run(args)

        warnings: list[str] = []
        confidence = ProofConfidence.FULL

        ambig_warn = self._gate_ambiguous(output, symbol)
        if ambig_warn:
            warnings += ambig_warn
            confidence = ProofConfidence.PARTIAL

        if "not found" in output.lower() and code == 0:
            warnings.append(
                f"[PARTIAL PROOF] Symbol '{symbol}' not found in index. "
                "Check spelling, confirm the index is current ('codegraph sync'), "
                "or the symbol may be dynamically generated."
            )
            confidence = ProofConfidence.PARTIAL

        result = CodegraphResult(
            query=symbol, command="query", output=output,
            confidence=confidence, warnings=warnings, exit_code=code,
        )
        if confidence == ProofConfidence.PARTIAL:
            result.fallback_instructions.append(
                f"Re-run with file_hint='<specific_file.py>' to disambiguate."
            )
        return result

    def callers(self, symbol: str, file_hint: Optional[str] = None) -> CodegraphResult:
        """Find upstream callers. Gates: file-path, dynamic-dispatch, cron/shell, partial."""
        path_warn = self._gate_file_path(symbol)
        if path_warn:
            return CodegraphResult(
                query=symbol, command="callers", output="",
                confidence=ProofConfidence.UNVERIFIED, warnings=path_warn,
                fallback_instructions=[f"rg '{symbol}' --type py"],
            )

        args = ["callers", symbol]
        if file_hint:
            args += ["--filter", file_hint]
        output, code = self._run(args)

        warnings: list[str] = []
        warnings += self._gate_dynamic_dispatch(output)
        warnings += self._gate_cron_shell(symbol, output)
        warnings += self._gate_ambiguous(output, symbol)

        confidence = ProofConfidence.PARTIAL if warnings else ProofConfidence.FULL

        if "not found" in output.lower():
            warnings.append(
                f"[PARTIAL PROOF] No callers found for '{symbol}'. "
                "This may mean: (a) it is a top-level entry point, "
                "(b) it is called via dynamic dispatch, or "
                "(c) the index is stale — run 'codegraph sync'."
            )
            confidence = ProofConfidence.PARTIAL

        return CodegraphResult(
            query=symbol, command="callers", output=output,
            confidence=confidence, warnings=warnings, exit_code=code,
        )

    def callees(self, symbol: str) -> CodegraphResult:
        """Find downstream callees. Gates: file-path, dynamic-dispatch."""
        path_warn = self._gate_file_path(symbol)
        if path_warn:
            return CodegraphResult(
                query=symbol, command="callees", output="",
                confidence=ProofConfidence.UNVERIFIED, warnings=path_warn,
            )

        output, code = self._run(["callees", symbol])
        warnings = self._gate_dynamic_dispatch(output)
        confidence = ProofConfidence.PARTIAL if warnings else ProofConfidence.FULL

        return CodegraphResult(
            query=symbol, command="callees", output=output,
            confidence=confidence, warnings=warnings, exit_code=code,
        )

    def impact(self, symbol: str, depth: int = 2) -> CodegraphResult:
        """Blast-radius analysis. Gates: file-path, dynamic-dispatch, partial proof."""
        path_warn = self._gate_file_path(symbol)
        if path_warn:
            return CodegraphResult(
                query=symbol, command="impact", output="",
                confidence=ProofConfidence.UNVERIFIED, warnings=path_warn,
            )

        output, code = self._run(["impact", "--depth", str(depth), symbol])
        warnings = self._gate_dynamic_dispatch(output)

        if warnings:
            warnings.append(
                "[PARTIAL PROOF] Impact count is a lower bound — dynamic dispatch "
                "gaps mean actual blast radius may be larger. "
                "Inspect break points manually before concluding this change is safe."
            )

        confidence = ProofConfidence.PARTIAL if warnings else ProofConfidence.FULL

        return CodegraphResult(
            query=symbol, command="impact", output=output,
            confidence=confidence, warnings=warnings, exit_code=code,
        )

    def tier3_gate(self, symbol: str) -> tuple[bool, list[str]]:
        """Run impact and return (hit_tier3, list_of_tier3_surfaces_reached).

        Use before any Tier 2+ edit: if hit_tier3 is True, the change
        requires operator approval (BLOCKED per governance policy).
        """
        result = self.impact(symbol, depth=2)
        lower = result.output.lower()
        hit: list[str] = [s for s in TIER3_SURFACES if s in lower]
        return bool(hit), hit

    # ------------------------------------------------------------------
    # Internal gates
    # ------------------------------------------------------------------

    def _gate_file_path(self, symbol: str) -> list[str]:
        if _FILE_PATH_RE.search(symbol):
            return [
                f"[FILE-PATH LITERAL] '{symbol}' looks like a file path or extension. "
                "codegraph cannot trace string literals. "
                "Use text search instead: "
                f"  rg '{symbol}' --type py"
            ]
        return []

    def _gate_ambiguous(self, output: str, symbol: str) -> list[str]:
        # Multiple definition lines in query output indicate ambiguity.
        # Definition lines contain a % relevance score.
        def_lines = [ln for ln in output.splitlines() if "%" in ln]
        if len(def_lines) > 1:
            return [
                f"[AMBIGUOUS SYMBOL] '{symbol}' matches {len(def_lines)} definitions. "
                "Provide file context to disambiguate before drawing conclusions. "
                "Example: guard.callers('symbol', file_hint='specific_module.py')"
            ]
        return []

    def _gate_dynamic_dispatch(self, output: str) -> list[str]:
        lower = output.lower()
        for phrase in _DYNAMIC_DISPATCH_PHRASES:
            if phrase in lower:
                return [
                    "[DYNAMIC DISPATCH] Static graph has a break point here. "
                    "Inspect the break point with codegraph_node + grep/read. "
                    "Do NOT infer missing edges across the gap."
                ]
        return []

    def _gate_cron_shell(self, symbol: str, output: str) -> list[str]:
        combined = (symbol + " " + output).lower()
        for phrase in _CRON_SHELL_PHRASES:
            if phrase in combined:
                return [
                    f"[CRON/SHELL BOUNDARY] '{symbol}' may involve subprocess or cron "
                    "execution. Static graph has no edges across these boundaries. "
                    "Verify separately: inspect subprocess calls manually, "
                    "check 'crontab -l' for cron wiring."
                ]
        return []

    # ------------------------------------------------------------------
    # CLI runner
    # ------------------------------------------------------------------

    def _run(self, args: list[str]) -> tuple[str, int]:
        try:
            proc = subprocess.run(
                ["codegraph"] + args,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            return (proc.stdout + proc.stderr).strip(), proc.returncode
        except FileNotFoundError:
            return "[ERROR] codegraph not found in PATH. Run: npm install -g @colbymchenry/codegraph@0.9.6", 1
        except subprocess.TimeoutExpired:
            return f"[ERROR] codegraph timed out after {self._timeout}s", 1
