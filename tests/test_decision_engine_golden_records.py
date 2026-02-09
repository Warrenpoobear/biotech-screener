#!/usr/bin/env python3
"""
test_decision_engine_golden_records.py - Golden Record Tests for Decision Engine v1

Tests 3 synthetic rec fixtures covering all catalyst modes:
  1. specific_days  -> Tier A (high opt + catalyst near)
  2. blended_window -> Tier A (high opt + blended proximity)
  3. missing        -> Tier C (moderate opt + no catalyst data)

Also tests:
  4. no_upcoming    -> Tier B (high opt + pipeline ran, nothing actionable)
  5. ineligible     -> Tier D (fundamental_red_flag)

These are the authoritative golden records referenced in decision_engine_v1.md
Section 9.5. If any assertion fails, the code change is real and must be
intentional.

Usage:
    python tests/test_decision_engine_golden_records.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from decision_engine import (
    VERSION,
    RULESET_ID,
    DECISION_COLUMNS,
    DecisionRuleset,
    compute_decision_fields,
)


# =============================================================================
# SYNTHETIC REC FIXTURES
# =============================================================================

def _base_rec(ticker, **overrides):
    """Minimal rec dict with sane defaults. Override any field."""
    rec = {
        "ticker": ticker,
        "component_scores": [
            {"name": "clinical", "normalized": "40.0", "raw": "60.0",
             "weight_base": "0.26", "weight_effective": "0.29",
             "confidence": "0.80", "contribution": "5.0",
             "decay_factor": None, "notes": []},
        ],
        "score_breakdown": {
            "enhancements": {
                "momentum": {"alpha_60d": 0.02},  # neutral
            },
        },
        "smart_money_signal": {
            "tier1_holders": ["FundA", "FundB", "FundC"],
            "holders_increasing": ["FundA"],
            "holders_decreasing": [],
            "overlap_count": 2,
        },
        "coinvest": {"tier1_count": 3},
        "catalyst_decay": {
            "days_to_catalyst": None,
            "factor": "0.5",
            "in_optimal_window": False,
        },
        "defensive_features": {
            "vol_60d": 0.65,
            "beta_xbi_60d": 1.1,
            "drawdown": -0.15,
            "rsi_14d": 48.0,
        },
        "severity": "NONE",
        "confidence_overall": 0.72,
        "fundamental_red_flag": False,
        "fundamental_red_flag_reasons": [],
        "flags": [],
        "attn_flags": [],
        "momentum_signal": {"alpha_60d": 0.02},
    }
    # Apply overrides (supports nested dicts via top-level key replacement)
    rec.update(overrides)
    return rec


# --- Fixture 1: specific_days -> Tier A ---
REC_SPECIFIC_DAYS = _base_rec(
    "SYNTH_SPEC",
    catalyst_decay={
        "days_to_catalyst": 45,
        "factor": "0.85",
        "in_optimal_window": True,
    },
)

# --- Fixture 2: blended_window -> Tier A ---
REC_BLENDED_WINDOW = _base_rec(
    "SYNTH_BLEND",
    catalyst_decay={
        "days_to_catalyst": 0,
        "factor": "0.70",
        "in_optimal_window": True,
    },
)

# --- Fixture 3: missing -> Tier C ---
# No catalyst_decay key at all
REC_MISSING = _base_rec(
    "SYNTH_MISS",
    catalyst_decay=None,  # will be treated as {} via `or {}`
    coinvest={"tier1_count": 1},  # low sponsor to keep sizing reasonable
)

# --- Fixture 4: no_upcoming -> Tier B ---
REC_NO_UPCOMING = _base_rec(
    "SYNTH_NOUP",
    catalyst_decay={
        "days_to_catalyst": None,
        "factor": "0.5",
        "in_optimal_window": False,
    },
)

# --- Fixture 5: ineligible -> Tier D ---
REC_INELIGIBLE = _base_rec(
    "SYNTH_INEL",
    fundamental_red_flag=True,
    fundamental_red_flag_reasons=["cash_runway_lt_6m"],
    severity="SEV3",
    catalyst_decay={
        "days_to_catalyst": 30,
        "factor": "0.90",
        "in_optimal_window": True,
    },
)


# --- Fixture 6: soft drawdown breach -> stays eligible, size reduced ---
REC_DRAWDOWN_SOFT = _base_rec(
    "SYNTH_DDSOFT",
    defensive_features={
        "vol_60d": 0.65,
        "beta_xbi_60d": 1.1,
        "drawdown": -0.50,
        "rsi_14d": 48.0,
    },
    catalyst_decay={
        "days_to_catalyst": 45,
        "factor": "0.85",
        "in_optimal_window": True,
    },
)

SOFT_DRAWDOWN_RULESET = DecisionRuleset(drawdown_gate_mode="soft")


# =============================================================================
# TESTS
# =============================================================================

def test_versioning():
    """Version and ruleset ID are present and stable."""
    result = compute_decision_fields(REC_SPECIFIC_DAYS, "drug_developer", 0.75)
    assert result["decision_engine_version"] == VERSION
    assert result["decision_engine_ruleset_id"] == RULESET_ID
    assert len(RULESET_ID) == 8  # sha256[:8]


def test_all_columns_present():
    """Every DECISION_COLUMNS key is emitted."""
    result = compute_decision_fields(REC_SPECIFIC_DAYS, "drug_developer", 0.75)
    for col in DECISION_COLUMNS:
        assert col in result, f"Missing column: {col}"


def test_specific_days_tier_a():
    """specific_days catalyst + high optionality -> Tier A."""
    result = compute_decision_fields(REC_SPECIFIC_DAYS, "drug_developer", 0.75)
    assert result["eligible"] == "1"
    assert result["catalyst_mode"] == "specific_days"
    assert result["catalyst_days"] == 45
    assert result["catalyst_in_window"] == "1"
    assert result["tier_dev"] == "A"
    assert result["tier_reason"] == "high_opt+catalyst_near"


def test_blended_window_tier_a():
    """blended_window catalyst + high optionality -> Tier A."""
    result = compute_decision_fields(REC_BLENDED_WINDOW, "drug_developer", 0.75)
    assert result["eligible"] == "1"
    assert result["catalyst_mode"] == "blended_window"
    assert result["catalyst_days"] == 0
    assert result["catalyst_in_window"] == "1"
    assert result["tier_dev"] == "A"
    assert result["tier_reason"] == "high_opt+catalyst_window"


def test_missing_catalyst_tier_c():
    """Missing catalyst data + moderate optionality -> Tier C (degraded)."""
    result = compute_decision_fields(REC_MISSING, "drug_developer", 0.45)
    assert result["eligible"] == "1"
    assert result["catalyst_mode"] == "missing"
    assert result["catalyst_days"] == ""
    assert result["catalyst_in_window"] == ""
    assert result["tier_dev"] == "C"
    assert result["tier_reason"] == "mod_opt+no_catalyst_data"


def test_no_upcoming_tier_b():
    """Pipeline ran, nothing actionable + high optionality -> Tier B (capped)."""
    result = compute_decision_fields(REC_NO_UPCOMING, "drug_developer", 0.75)
    assert result["eligible"] == "1"
    assert result["catalyst_mode"] == "no_upcoming"
    assert result["catalyst_days"] == ""  # days_to_catalyst was None -> blank
    assert result["catalyst_in_window"] == "0"
    assert result["tier_dev"] == "B"
    assert result["tier_reason"] == "high_opt+no_catalyst_data"


def test_ineligible_tier_d():
    """Fundamental red flag -> ineligible -> Tier D, size XS."""
    result = compute_decision_fields(REC_INELIGIBLE, "drug_developer", 0.80)
    assert result["eligible"] == "0"
    assert "fundamental_red_flag" in result["ineligible_reasons"]
    assert result["tier_dev"] == "D"
    assert result["tier_reason"] == "ineligible"
    assert result["size_band"] == "XS"
    assert result["runway_bucket"] == "critical"


def test_soft_drawdown_stays_eligible():
    """Soft mode: drawdown=-0.50 breaches -0.40 gate but stays eligible, tier A/B, size reduced."""
    # Under hard mode (default): ineligible
    hard_result = compute_decision_fields(REC_DRAWDOWN_SOFT, "drug_developer", 0.75)
    assert hard_result["eligible"] == "0"
    assert "deep_drawdown" in hard_result["ineligible_reasons"]

    # Under soft mode: eligible, good tier, but sizing penalized
    soft_result = compute_decision_fields(
        REC_DRAWDOWN_SOFT, "drug_developer", 0.75, ruleset=SOFT_DRAWDOWN_RULESET,
    )
    assert soft_result["eligible"] == "1"
    assert soft_result["tier_dev"] in ("A", "B")
    assert "drawdown_penalty" in soft_result["size_reasons"]
    # Deep drawdown risk flag should still fire (overlay layer independent)
    assert "deep_drawdown" in soft_result["risk_flags"]


def test_non_dev_archetype():
    """Non-dev archetype -> tier_dev and tier_reason are blank."""
    result = compute_decision_fields(REC_SPECIFIC_DAYS, "commercial_biotech", 0.75)
    assert result["tier_dev"] == ""
    assert result["tier_reason"] == ""
    assert result["eligible"] == "1"


def test_blank_vs_zero_sponsorship():
    """Blank-when-missing: no coinvest data -> blank, not zero."""
    rec = _base_rec("SYNTH_NOSPON", coinvest={}, smart_money_signal={})
    result = compute_decision_fields(rec, "drug_developer", 0.50)
    assert result["sponsor_tier1_count"] == ""
    assert result["sponsor_overlap_count"] == ""
    assert result["sponsor_net_buying"] == ""


def test_zero_sponsorship():
    """Zero is real data: coinvest ran but found 0 tier-1."""
    rec = _base_rec(
        "SYNTH_ZEROSPON",
        coinvest={"tier1_count": 0},
        smart_money_signal={"overlap_count": 0, "holders_increasing": [], "holders_decreasing": []},
    )
    result = compute_decision_fields(rec, "drug_developer", 0.50)
    assert result["sponsor_tier1_count"] == 0
    assert result["sponsor_overlap_count"] == 0
    assert result["sponsor_net_buying"] == "neutral"


def test_sizing_orthogonal_to_tier():
    """Sizing reflects risk/confirmation signals, not tier directly.

    A Tier C name with strong sponsors + tailwind can reach L.
    """
    rec = _base_rec(
        "SYNTH_CL",
        catalyst_decay=None,
        coinvest={"tier1_count": 4},
        smart_money_signal={
            "tier1_holders": ["A", "B", "C", "D"],
            "holders_increasing": ["A", "B"],
            "holders_decreasing": [],
            "overlap_count": 3,
        },
    )
    rec["score_breakdown"]["enhancements"]["momentum"] = {"alpha_60d": 0.10}
    result = compute_decision_fields(rec, "drug_developer", 0.25)
    assert result["tier_dev"] == "C"
    assert result["size_band"] == "L"  # sponsor + momentum push past M


# =============================================================================
# RUNNER
# =============================================================================

def _run_all():
    """Run all test_ functions and report results."""
    import inspect
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            # Get the failing line
            tb = sys.exc_info()[2]
            lineno = tb.tb_next.tb_lineno if tb.tb_next else tb.tb_lineno
            print(f"  FAIL  {name} (line {lineno}): {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")

    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests")
    return failed == 0


if __name__ == "__main__":
    print(f"Decision Engine {VERSION} (ruleset {RULESET_ID})")
    print(f"Running golden record tests...\n")
    success = _run_all()
    sys.exit(0 if success else 1)
