"""Tests for data_integrity_audit.py — invariant checks + price-derived cross-validation.

Validates:
  1. check_invariants catches eligible/ineligible mismatches
  2. check_invariants catches catalyst window violations
  3. check_invariants catches range violations
  4. check_universe_coverage detects missing tickers
  5. _violation_severity classifies correctly
  6. _verdict returns correct labels
  7. _safe_float handles edge cases
  8. recompute_price_fields detects mismatches
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.corporate_actions import load_actions  # noqa: E402
from tools.data_integrity_audit import (  # noqa: E402
    _safe_float,
    _split_adjust_prices,
    _verdict,
    _violation_severity,
    check_invariants,
    check_universe_coverage,
    recompute_price_fields,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(**kw):
    """Build a single-row DataFrame for check_invariants."""
    defaults = {
        "ticker": "TEST",
        "eligible": "1",
        "ineligible_reasons": "",
        "actionable_rank": "5",
        "catalyst_in_window": "",
        "catalyst_days": "",
        "catalyst_mode": "",
        "missingness_penalty": "0",
        "missing_components": "",
        "tier_any": "",
        "tier_any_reason": "",
        "risk_flags": "",
        "de_drawdown": "",
        "de_rsi_14d": "",
        "de_beta_xbi_60d": "",
    }
    defaults.update(kw)
    return pd.DataFrame([defaults])


# ---------------------------------------------------------------------------
# A) _violation_severity
# ---------------------------------------------------------------------------


class TestViolationSeverity:
    def test_critical_rules(self):
        assert _violation_severity("eligible_reasons_mismatch") == "critical"
        assert _violation_severity("ineligible_has_rank") == "critical"

    def test_warn_rules(self):
        assert _violation_severity("catalyst_window_no_days") == "warn"
        assert _violation_severity("tier_no_reason") == "warn"

    def test_range_rules_are_info(self):
        assert _violation_severity("range_de_drawdown") == "info"
        assert _violation_severity("range_score_rank_pct") == "info"

    def test_unknown_defaults_to_warn(self):
        assert _violation_severity("some_new_rule_xyz") == "warn"


# ---------------------------------------------------------------------------
# B) _verdict
# ---------------------------------------------------------------------------


class TestVerdict:
    def test_both_missing(self):
        assert _verdict(None, None, 0.01) == "BOTH_MISSING"

    def test_stored_missing(self):
        assert _verdict(None, 0.5, 0.01) == "STORED_MISSING"

    def test_recomp_missing(self):
        assert _verdict(0.5, None, 0.01) == "RECOMP_MISSING"

    def test_ok_within_tolerance(self):
        assert _verdict(0.50, 0.51, 0.02) == "OK"

    def test_fail_outside_tolerance(self):
        assert _verdict(0.50, 0.60, 0.02) == "FAIL"

    def test_exact_match(self):
        assert _verdict(1.0, 1.0, 0.001) == "OK"


# ---------------------------------------------------------------------------
# C) _safe_float
# ---------------------------------------------------------------------------


class TestSafeFloat:
    def test_valid_number(self):
        assert _safe_float("3.14") == 3.14

    def test_none(self):
        assert _safe_float(None) is None

    def test_nan_float(self):
        assert _safe_float(float("nan")) is None

    def test_nan_string(self):
        assert _safe_float("nan") is None

    def test_empty_string(self):
        assert _safe_float("") is None

    def test_none_string(self):
        assert _safe_float("None") is None

    def test_integer(self):
        assert _safe_float(42) == 42.0


# ---------------------------------------------------------------------------
# D) check_invariants
# ---------------------------------------------------------------------------


class TestCheckInvariants:
    def test_clean_row_no_violations(self):
        df = _row()
        violations = check_invariants(df)
        assert len(violations) == 0

    def test_eligible_with_ineligible_reasons(self):
        df = _row(eligible="1", ineligible_reasons="low_volume")
        violations = check_invariants(df)
        rules = [v["rule"] for v in violations]
        assert "eligible_reasons_mismatch" in rules

    def test_ineligible_with_rank(self):
        df = _row(eligible="0", actionable_rank="3")
        violations = check_invariants(df)
        rules = [v["rule"] for v in violations]
        assert "ineligible_has_rank" in rules

    def test_ineligible_no_rank_is_clean(self):
        df = _row(eligible="0", actionable_rank="", ineligible_reasons="no_data")
        violations = check_invariants(df)
        rules = [v["rule"] for v in violations]
        assert "ineligible_has_rank" not in rules

    def test_catalyst_in_window_no_days(self):
        df = _row(catalyst_in_window="True", catalyst_days="")
        violations = check_invariants(df)
        rules = [v["rule"] for v in violations]
        assert "catalyst_window_no_days" in rules

    def test_catalyst_in_window_negative_days(self):
        df = _row(catalyst_in_window="True", catalyst_days="-5")
        violations = check_invariants(df)
        rules = [v["rule"] for v in violations]
        assert "catalyst_window_negative_days" in rules

    def test_catalyst_in_window_valid(self):
        df = _row(catalyst_in_window="True", catalyst_days="30")
        violations = check_invariants(df)
        rules = [v["rule"] for v in violations]
        assert "catalyst_window_no_days" not in rules
        assert "catalyst_window_negative_days" not in rules

    def test_specific_days_no_days(self):
        df = _row(catalyst_mode="specific_days", catalyst_days="")
        violations = check_invariants(df)
        rules = [v["rule"] for v in violations]
        assert "specific_days_no_days" in rules

    def test_specific_days_non_integer(self):
        df = _row(catalyst_mode="specific_days", catalyst_days="30.5")
        violations = check_invariants(df)
        rules = [v["rule"] for v in violations]
        assert "specific_days_non_integer" in rules

    def test_penalty_without_components(self):
        df = _row(missingness_penalty="0.5", missing_components="")
        violations = check_invariants(df)
        rules = [v["rule"] for v in violations]
        assert "penalty_no_components" in rules

    def test_penalty_with_components_ok(self):
        df = _row(missingness_penalty="0.5", missing_components="catalyst")
        violations = check_invariants(df)
        rules = [v["rule"] for v in violations]
        assert "penalty_no_components" not in rules

    def test_unknown_missing_component(self):
        df = _row(missingness_penalty="0.5", missing_components="bogus_component")
        violations = check_invariants(df)
        rules = [v["rule"] for v in violations]
        assert "unknown_missing_component" in rules

    def test_tier_without_reason(self):
        df = _row(tier_any="A", tier_any_reason="")
        violations = check_invariants(df)
        rules = [v["rule"] for v in violations]
        assert "tier_no_reason" in rules

    def test_tier_with_reason_ok(self):
        df = _row(tier_any="A", tier_any_reason="clinical_quality")
        violations = check_invariants(df)
        rules = [v["rule"] for v in violations]
        assert "tier_no_reason" not in rules

    def test_deep_drawdown_flag_no_value(self):
        df = _row(risk_flags="deep_drawdown", de_drawdown="")
        violations = check_invariants(df)
        rules = [v["rule"] for v in violations]
        assert "deep_dd_no_value" in rules

    def test_overbought_rsi_flag_no_value(self):
        df = _row(risk_flags="overbought_rsi", de_rsi_14d="")
        violations = check_invariants(df)
        rules = [v["rule"] for v in violations]
        assert "rsi_flag_no_value" in rules

    def test_range_violation(self):
        # score_rank_pct must be in [0, 1] per SANITY_RANGES
        df = _row(score_rank_pct="1.5")
        violations = check_invariants(df)
        rules = [v["rule"] for v in violations]
        assert "range_score_rank_pct" in rules

    def test_range_ok(self):
        df = _row(score_rank_pct="0.75")
        violations = check_invariants(df)
        rules = [v["rule"] for v in violations]
        range_violations = [r for r in rules if r.startswith("range_score_rank_pct")]
        assert len(range_violations) == 0

    def test_multiple_violations_same_row(self):
        df = _row(
            eligible="1",
            ineligible_reasons="test",
            tier_any="B",
            tier_any_reason="",
        )
        violations = check_invariants(df)
        rules = [v["rule"] for v in violations]
        assert "eligible_reasons_mismatch" in rules
        assert "tier_no_reason" in rules


# ---------------------------------------------------------------------------
# E) check_universe_coverage
# ---------------------------------------------------------------------------


class TestUniverseCoverage:
    def test_all_present(self, tmp_path):
        universe = [{"ticker": "AAPL"}, {"ticker": "GOOG"}]
        uni_path = tmp_path / "universe.json"
        uni_path.write_text(json.dumps(universe))

        df = pd.DataFrame({"ticker": ["AAPL", "GOOG", "MSFT"]})
        violations = check_universe_coverage(df, uni_path)
        assert len(violations) == 0

    def test_missing_ticker(self, tmp_path):
        universe = [{"ticker": "AAPL"}, {"ticker": "GOOG"}, {"ticker": "TSLA"}]
        uni_path = tmp_path / "universe.json"
        uni_path.write_text(json.dumps(universe))

        df = pd.DataFrame({"ticker": ["AAPL", "GOOG"]})
        violations = check_universe_coverage(df, uni_path)
        assert len(violations) == 1
        assert violations[0]["ticker"] == "TSLA"
        assert violations[0]["rule"] == "universe_missing"

    def test_no_universe_file(self, tmp_path):
        df = pd.DataFrame({"ticker": ["AAPL"]})
        violations = check_universe_coverage(df, tmp_path / "missing.json")
        assert len(violations) == 0

    def test_skips_underscore_tickers(self, tmp_path):
        universe = [{"ticker": "AAPL"}, {"ticker": "_INTERNAL"}]
        uni_path = tmp_path / "universe.json"
        uni_path.write_text(json.dumps(universe))

        df = pd.DataFrame({"ticker": ["AAPL"]})
        violations = check_universe_coverage(df, uni_path)
        assert len(violations) == 0

    def test_string_universe_format(self, tmp_path):
        universe = ["AAPL", "GOOG"]
        uni_path = tmp_path / "universe.json"
        uni_path.write_text(json.dumps(universe))

        df = pd.DataFrame({"ticker": ["AAPL"]})
        violations = check_universe_coverage(df, uni_path)
        assert len(violations) == 1
        assert violations[0]["ticker"] == "GOOG"


# ---------------------------------------------------------------------------
# F) Split-adjusted price recompute (MLTX 10:1 forward split false-positive)
# ---------------------------------------------------------------------------


AS_OF = "2026-07-02"
SPLIT_DATE = "2026-03-16"  # a business day partway through the window


def _split_registry(tmp_path):
    """Registry with a single 10:1 forward split (factor 0.1) for SPLT."""
    payload = {
        "actions": [
            {
                "ticker": "SPLT",
                "action": "forward_split",
                "effective_date": SPLIT_DATE,
                "ratio": "10:1",
                "factor": 0.1,
                "notes": "test fixture (mirrors MLTX 2025-09-29)",
            }
        ]
    }
    p = tmp_path / "corporate_actions.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return load_actions(p)


def _split_prices():
    """Raw (unadjusted) price history for a ticker with a 10:1 forward split.

    Pre-split: flat 60.0 (raw). On the split date the raw price divides by 10.
    Post-split: 6.0 -> peak 6.6 -> ends 5.0. 140 business days total (> MIN_BARS_DD).
    Split-adjusted-to-as_of drawdown = 5.0/6.6 - 1 = -0.2424.
    Raw (buggy) drawdown = 5.0/60.0 - 1 = -0.9167.
    """
    dates = pd.bdate_range(end=AS_OF, periods=140).strftime("%Y-%m-%d").tolist()
    split_idx = next(i for i, d in enumerate(dates) if d >= SPLIT_DATE)
    rows = []
    for i, d in enumerate(dates):
        if i < split_idx:
            close = 60.0  # pre-split raw
        else:
            # post-split raw: 6.0 rising to 6.6 then falling to 5.0
            n_post = len(dates) - split_idx
            j = i - split_idx
            if j <= n_post // 2:
                close = 6.0 + 0.6 * (j / max(1, n_post // 2))
            else:
                close = 6.6 - 1.6 * ((j - n_post // 2) / max(1, n_post - n_post // 2))
        rows.append({"ticker": "SPLT", "date": d, "close": round(close, 4)})
    df = pd.DataFrame(rows)
    df["date"] = df["date"].astype(str)
    return df


class TestSplitAdjustedRecompute:
    def test_raw_prices_falsely_flag_split_ticker(self, tmp_path):
        """Without split adjustment, a split ticker's stored (adjusted) drawdown
        diverges wildly from the raw recompute — the false-positive we're fixing."""
        prices = _split_prices()
        rankings = pd.DataFrame([{"ticker": "SPLT", "de_drawdown": "-0.2424"}])
        diffs = recompute_price_fields(rankings, prices, AS_OF)
        entry = diffs[0]
        assert entry["dd_verdict"] == "FAIL"
        assert abs(entry["dd_recomputed"] - (-0.9167)) < 0.01

    def test_split_adjust_prices_fixes_recompute(self, tmp_path):
        """After applying corporate_actions split factors, the recompute matches
        the stored split-adjusted value (OK, no false positive)."""
        registry = _split_registry(tmp_path)
        prices = _split_adjust_prices(_split_prices(), AS_OF, registry)
        rankings = pd.DataFrame([{"ticker": "SPLT", "de_drawdown": "-0.2424"}])
        diffs = recompute_price_fields(rankings, prices, AS_OF)
        entry = diffs[0]
        assert entry["dd_verdict"] == "OK", entry
        assert abs(entry["dd_recomputed"] - (-0.2424)) < 0.02

    def test_split_adjust_leaves_unsplit_ticker_untouched(self, tmp_path):
        """A ticker with no corporate action must be returned byte-identical."""
        registry = _split_registry(tmp_path)
        prices = pd.DataFrame(
            [
                {"ticker": "NOSP", "date": "2026-01-05", "close": 10.0},
                {"ticker": "NOSP", "date": "2026-06-30", "close": 12.0},
            ]
        )
        out = _split_adjust_prices(prices.copy(), AS_OF, registry)
        pd.testing.assert_frame_equal(out.reset_index(drop=True), prices.reset_index(drop=True))

    def test_split_adjust_empty_registry_is_noop(self):
        """No registry / no actions -> prices returned unchanged."""
        from common.corporate_actions import CorporateActionRegistry

        prices = _split_prices()
        out = _split_adjust_prices(prices.copy(), AS_OF, CorporateActionRegistry())
        pd.testing.assert_frame_equal(out.reset_index(drop=True), prices.reset_index(drop=True))
