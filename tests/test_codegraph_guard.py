"""Unit tests for common/codegraph_guard.py — the Hermes acceptance-gate wrapper.

Each test exercises one of the five acceptance gates:
  1. File-path literal gate
  2. Ambiguous symbol gate
  3. Dynamic-dispatch gate
  4. Cron/shell boundary gate
  5. Partial-proof gate (symbol not found)

Also tests tier3_gate() and the ProofConfidence trustworthiness contract.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from common.codegraph_guard import TIER3_SURFACES, CodegraphGuard, ProofConfidence

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_guard(mock_output: str = "", exit_code: int = 0) -> tuple[CodegraphGuard, object]:
    """Return (guard, mock) with _run stubbed to return (mock_output, exit_code)."""
    guard = CodegraphGuard()
    patcher = patch.object(guard, "_run", return_value=(mock_output, exit_code))
    mock = patcher.start()
    return guard, mock


# ---------------------------------------------------------------------------
# Gate 3: File-path literal
# ---------------------------------------------------------------------------


class TestFilePathLiteralGate:
    def test_csv_extension_triggers_gate(self):
        guard, _ = _make_guard()
        result = guard.query("rankings.csv")
        assert result.confidence == ProofConfidence.UNVERIFIED
        assert any("FILE-PATH LITERAL" in w for w in result.warnings)
        assert any("rg" in fi for fi in result.fallback_instructions)

    def test_py_extension_triggers_gate(self):
        guard, _ = _make_guard()
        result = guard.callers("run_screen.py")
        assert result.confidence == ProofConfidence.UNVERIFIED

    def test_absolute_path_triggers_gate(self):
        guard, _ = _make_guard()
        result = guard.impact("/workspace/run_screen.py")
        assert result.confidence == ProofConfidence.UNVERIFIED

    def test_plain_symbol_does_not_trigger_gate(self):
        guard, mock = _make_guard('Callers of "save_validation_snapshot" (1):\nfunction main  run_screen.py:1')
        result = guard.callers("save_validation_snapshot")
        mock.assert_called_once()
        assert result.confidence == ProofConfidence.FULL


# ---------------------------------------------------------------------------
# Gate 2: Ambiguous symbol
# ---------------------------------------------------------------------------


class TestAmbiguousSymbolGate:
    MULTI_MATCH_OUTPUT = (
        'Search Results for "main":\n'
        "function    main (95%)\n"
        "  run_screen.py:11358\n"
        "function    main (91%)\n"
        "  scripts/run_screen_from_bundle.py:1\n"
        "function    main (88%)\n"
        "  tools/run_daily_production.py:1\n"
    )

    def test_multiple_definitions_trigger_ambiguity_warning(self):
        guard, _ = _make_guard(self.MULTI_MATCH_OUTPUT)
        result = guard.query("main")
        assert result.confidence == ProofConfidence.PARTIAL
        assert any("AMBIGUOUS SYMBOL" in w for w in result.warnings)
        assert any("file_hint" in fi for fi in result.fallback_instructions)

    def test_single_definition_is_full_confidence(self):
        guard, _ = _make_guard(
            'Search Results for "final_score":\n' "variable    final_score (99%)\n" "  decision_engine.py:42\n"
        )
        result = guard.query("final_score")
        assert result.confidence == ProofConfidence.FULL
        assert not result.warnings


# ---------------------------------------------------------------------------
# Gate 1: Dynamic-dispatch
# ---------------------------------------------------------------------------


class TestDynamicDispatchGate:
    def test_dynamic_dispatch_in_output_warns(self):
        guard, _ = _make_guard(
            'Callees of "run_screen":\n' "  dynamic dispatch detected at line 200 — cannot trace further\n"
        )
        result = guard.callees("run_screen")
        assert result.confidence == ProofConfidence.PARTIAL
        assert any("DYNAMIC DISPATCH" in w for w in result.warnings)

    def test_impact_with_dynamic_dispatch_adds_lower_bound_warning(self):
        guard, _ = _make_guard("Impact: 3 symbols affected\n" "  dynamic dispatch break point at compositor\n")
        result = guard.impact("composite_score")
        assert result.confidence == ProofConfidence.PARTIAL
        assert any("PARTIAL PROOF" in w for w in result.warnings)
        assert any("lower bound" in w for w in result.warnings)

    def test_clean_output_has_full_confidence(self):
        guard, _ = _make_guard('Impact of changing "foo" — 2 affected symbols:\n  bar\n  baz')
        result = guard.impact("foo")
        assert result.confidence == ProofConfidence.FULL
        assert not result.warnings


# ---------------------------------------------------------------------------
# Gate 4: Cron/shell boundary
# ---------------------------------------------------------------------------


class TestCronShellBoundaryGate:
    def test_subprocess_in_symbol_warns(self):
        guard, _ = _make_guard('Callees of "subprocess_runner":\n  nothing\n')
        result = guard.callers("subprocess_runner")
        assert any("CRON/SHELL BOUNDARY" in w for w in result.warnings)

    def test_subprocess_in_output_warns(self):
        guard, _ = _make_guard('Callers of "warm_caches":\n' "  subprocess.run called at warm_caches.py:88\n")
        result = guard.callers("warm_caches")
        assert any("CRON/SHELL BOUNDARY" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Gate 5: Partial proof (symbol not found)
# ---------------------------------------------------------------------------


class TestPartialProofGate:
    def test_not_found_message_degrades_confidence(self):
        guard, _ = _make_guard('ℹ Symbol "ghost_function" not found')
        result = guard.callers("ghost_function")
        assert result.confidence == ProofConfidence.PARTIAL
        assert any("PARTIAL PROOF" in w for w in result.warnings)

    def test_not_found_in_query_warns(self):
        guard, _ = _make_guard('Symbol "ghost" not found')
        result = guard.query("ghost")
        assert result.confidence == ProofConfidence.PARTIAL


# ---------------------------------------------------------------------------
# Tier 3 gate
# ---------------------------------------------------------------------------


class TestTier3Gate:
    def test_hits_selector_engine(self):
        guard, _ = _make_guard(
            'Impact of changing "coinvest_score_z" — 5 affected symbols:\n'
            "  selector_engine.py:42\n"
            "  run_screen_columns.py:100\n"
        )
        hit, surfaces = guard.tier3_gate("coinvest_score_z")
        assert hit is True
        assert "selector_engine" in surfaces

    def test_no_tier3_hit(self):
        guard, _ = _make_guard(
            'Impact of changing "format_report" — 1 affected symbol:\n' "  tools/report_formatter.py:10\n"
        )
        hit, surfaces = guard.tier3_gate("format_report")
        assert hit is False
        assert surfaces == []

    def test_tier3_surfaces_frozenset(self):
        assert isinstance(TIER3_SURFACES, frozenset)
        assert "selector_engine" in TIER3_SURFACES
        assert "ranker_engine" in TIER3_SURFACES
        assert "final_score" in TIER3_SURFACES


# ---------------------------------------------------------------------------
# CodegraphResult contracts
# ---------------------------------------------------------------------------


class TestCodegraphResultContract:
    def test_full_confidence_no_warnings_is_trustworthy(self):
        guard, _ = _make_guard('Callers of "foo" (1):\n  bar  baz.py:10')
        result = guard.callers("foo")
        assert result.is_trustworthy

    def test_partial_confidence_is_not_trustworthy(self):
        guard, _ = _make_guard('Symbol "foo" not found')
        result = guard.callers("foo")
        assert not result.is_trustworthy

    def test_format_includes_confidence_tag(self):
        guard, _ = _make_guard("some output")
        result = guard.query("my_symbol")
        formatted = result.format()
        assert "my_symbol" in formatted
        assert "FULL" in formatted or "PARTIAL" in formatted or "UNVERIFIED" in formatted

    def test_format_includes_warnings(self):
        guard, _ = _make_guard('Symbol "x" not found')
        result = guard.query("x")
        formatted = result.format()
        assert "PARTIAL PROOF" in formatted or "not found" in formatted.lower()


# ---------------------------------------------------------------------------
# CLI not found (gate: missing binary)
# ---------------------------------------------------------------------------


class TestMissingBinary:
    def test_missing_binary_returns_error_message(self):
        guard = CodegraphGuard()
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = guard.query("any_symbol")
        assert result.exit_code == 1
        assert "not found in PATH" in result.output
