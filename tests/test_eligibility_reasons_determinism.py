"""
Unit tests for eligibility reason canonicalization and summary sidecar.

Tests:
1. canonicalize_reasons: list input, string input, None, dedup, ordering
2. Eligibility summary: counts, examples, stable serialization

Run: pytest tests/test_eligibility_reasons_determinism.py -v
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from decision_engine_codes import canonicalize_reasons, REASON_ORDER


# =============================================================================
# CANONICALIZE REASONS — LIST INPUT
# =============================================================================


class TestCanonicalizeReasonsList:
    """Tests for canonicalize_reasons() with list inputs."""

    def test_empty_list_returns_empty(self):
        assert canonicalize_reasons([]) == []

    def test_none_returns_empty(self):
        assert canonicalize_reasons(None) == []

    def test_single_reason_returned(self):
        assert canonicalize_reasons(["deep_drawdown"]) == ["deep_drawdown"]

    def test_duplicates_removed(self):
        result = canonicalize_reasons(["adv_fail", "adv_fail", "sev3"])
        assert result == ["sev3", "adv_fail"]
        assert len(result) == 2

    def test_sorted_by_reason_order(self):
        """Known codes should be sorted in REASON_ORDER gate-evaluation order."""
        result = canonicalize_reasons(["adv_fail", "fundamental_red_flag", "financials_missing"])
        assert result == ["financials_missing", "fundamental_red_flag", "adv_fail"]

    def test_unknown_codes_after_known(self):
        """Unknown codes should sort alphabetically after all known codes."""
        result = canonicalize_reasons(["zzz_unknown", "deep_drawdown", "aaa_unknown"])
        assert result == ["deep_drawdown", "aaa_unknown", "zzz_unknown"]

    def test_whitespace_stripped(self):
        result = canonicalize_reasons(["  deep_drawdown  ", " adv_fail"])
        assert result == ["deep_drawdown", "adv_fail"]

    def test_empty_strings_dropped(self):
        result = canonicalize_reasons(["", "adv_fail", "", ""])
        assert result == ["adv_fail"]

    def test_all_known_codes_in_order(self):
        """All REASON_ORDER codes, shuffled, should come back in canonical order."""
        shuffled = list(reversed(REASON_ORDER))
        result = canonicalize_reasons(shuffled)
        assert result == REASON_ORDER


# =============================================================================
# CANONICALIZE REASONS — STRING INPUT
# =============================================================================


class TestCanonicalizeReasonsString:
    """Tests for canonicalize_reasons() with string inputs."""

    def test_pipe_separated(self):
        result = canonicalize_reasons("deep_drawdown|adv_fail")
        assert result == ["deep_drawdown", "adv_fail"]

    def test_comma_separated(self):
        result = canonicalize_reasons("adv_fail, deep_drawdown, adv_fail")
        assert result == ["deep_drawdown", "adv_fail"]

    def test_empty_string(self):
        assert canonicalize_reasons("") == []

    def test_whitespace_only(self):
        assert canonicalize_reasons("  ") == []

    def test_mixed_separators(self):
        """Pipe and comma mixed should both work."""
        result = canonicalize_reasons("adv_fail|sev3,deep_drawdown")
        assert result == ["sev3", "deep_drawdown", "adv_fail"]


# =============================================================================
# DETERMINISM
# =============================================================================


class TestDeterminism:
    """Ensure identical inputs always produce identical outputs."""

    def test_repeated_calls_identical(self):
        reasons = ["adv_fail", "fundamental_red_flag", "deep_drawdown", "sev3"]
        r1 = canonicalize_reasons(reasons)
        r2 = canonicalize_reasons(reasons)
        assert r1 == r2

    def test_input_order_independent(self):
        """Different input orderings should produce the same canonical list."""
        a = canonicalize_reasons(["adv_fail", "sev3", "deep_drawdown"])
        b = canonicalize_reasons(["deep_drawdown", "adv_fail", "sev3"])
        c = canonicalize_reasons(["sev3", "deep_drawdown", "adv_fail"])
        assert a == b == c

    def test_json_serialization_stable(self):
        """sort_keys JSON of summary should be stable across calls."""
        reasons = ["adv_fail", "deep_drawdown", "fundamental_red_flag"]
        c = canonicalize_reasons(reasons)
        j1 = json.dumps({"reasons": c}, sort_keys=True)
        j2 = json.dumps({"reasons": c}, sort_keys=True)
        assert j1 == j2


# =============================================================================
# ELIGIBILITY SUMMARY COMPUTATION
# =============================================================================


class TestEligibilitySummaryCounts:
    """Test summary computation from synthetic rows."""

    def _compute_summary(self, csv_rows):
        """Replicate the summary logic from run_screen.py."""
        counts = {}
        examples = {}
        n_eligible = 0
        n_ineligible = 0
        for row in csv_rows:
            raw = row.get("ineligible_reasons", "")
            codes = canonicalize_reasons(raw)
            if codes:
                n_ineligible += 1
                for c in codes:
                    counts[c] = counts.get(c, 0) + 1
                    examples.setdefault(c, []).append(row.get("ticker", ""))
            else:
                n_eligible += 1
        for c in examples:
            examples[c] = sorted(examples[c])[:10]
        return {
            "n_total": len(csv_rows),
            "n_eligible": n_eligible,
            "n_ineligible": n_ineligible,
            "counts_by_reason": counts,
            "examples_by_reason": examples,
        }

    def test_all_eligible(self):
        rows = [
            {"ticker": "AAA", "ineligible_reasons": ""},
            {"ticker": "BBB", "ineligible_reasons": ""},
        ]
        s = self._compute_summary(rows)
        assert s["n_total"] == 2
        assert s["n_eligible"] == 2
        assert s["n_ineligible"] == 0
        assert s["counts_by_reason"] == {}

    def test_mixed_eligibility(self):
        rows = [
            {"ticker": "AAA", "ineligible_reasons": ""},
            {"ticker": "BBB", "ineligible_reasons": "deep_drawdown"},
            {"ticker": "CCC", "ineligible_reasons": "deep_drawdown|adv_fail"},
            {"ticker": "DDD", "ineligible_reasons": "adv_fail"},
        ]
        s = self._compute_summary(rows)
        assert s["n_total"] == 4
        assert s["n_eligible"] == 1
        assert s["n_ineligible"] == 3
        assert s["counts_by_reason"]["deep_drawdown"] == 2
        assert s["counts_by_reason"]["adv_fail"] == 2

    def test_examples_bounded_and_sorted(self):
        """Examples list should be alphabetically sorted and capped at 10."""
        rows = [{"ticker": f"T{i:03d}", "ineligible_reasons": "sev3"} for i in range(15)]
        s = self._compute_summary(rows)
        assert len(s["examples_by_reason"]["sev3"]) == 10
        assert s["examples_by_reason"]["sev3"] == sorted(s["examples_by_reason"]["sev3"])

    def test_examples_deterministic_across_calls(self):
        rows = [
            {"ticker": "ZZZ", "ineligible_reasons": "adv_fail"},
            {"ticker": "AAA", "ineligible_reasons": "adv_fail"},
            {"ticker": "MMM", "ineligible_reasons": "adv_fail"},
        ]
        s1 = self._compute_summary(rows)
        s2 = self._compute_summary(rows)
        assert s1["examples_by_reason"]["adv_fail"] == ["AAA", "MMM", "ZZZ"]
        assert s1 == s2

    def test_eligible_consistent_with_reasons(self):
        """eligible=0 iff ineligible_reasons is non-empty."""
        rows = [
            {"ticker": "AAA", "eligible": "1", "ineligible_reasons": ""},
            {"ticker": "BBB", "eligible": "0", "ineligible_reasons": "sev3"},
        ]
        s = self._compute_summary(rows)
        assert s["n_eligible"] == 1
        assert s["n_ineligible"] == 1
