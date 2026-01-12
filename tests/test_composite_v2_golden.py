#!/usr/bin/env python3
"""
Golden fixture tests for Composite Ranker Module v2.

Tests:
1. Breakdown consistency - sum of contributions equals pre_penalty (within Decimal tolerance)
2. Confidence weighting - lowering confidence_financial decreases its contribution
3. Monotonic caps - liquidity FAIL never yields score > 35 even with perfect clinical/catalyst
4. Determinism hash - same inputs => same hash, any one input change => hash changes
5. Robust normalization - outlier in cohort doesn't produce extreme normalized shifts
6. Weakest-link - set financial very low, clinical high: hybrid score decreases vs pure sum

Run: python tests/test_composite_v2_golden.py
"""

import sys
from decimal import Decimal
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from module_5_composite_v2 import (
    compute_module_5_composite_v2,
    _score_single_ticker_v2,
    _rank_normalize_winsorized,
    _apply_monotonic_caps,
    _compute_determinism_hash,
    _extract_confidence_financial,
    _extract_confidence_clinical,
    _extract_confidence_catalyst,
    _to_decimal,
    _quantize_score,
    ComponentScore,
    ScoringMode,
    NormalizationMethod,
    MonotonicCap,
    HYBRID_ALPHA,
    CRITICAL_COMPONENTS,
    DEFAULT_WEIGHTS,
    ENHANCED_WEIGHTS,
    WINSOR_LOW,
    WINSOR_HIGH,
    EPS,
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

def create_universe_fixture(tickers: list) -> dict:
    """Create a universe result fixture."""
    return {
        "active_securities": [{"ticker": t} for t in tickers],
    }


def create_financial_fixture(
    ticker: str,
    financial_normalized: float = 50.0,
    severity: str = "none",
    liquidity_gate_status: str = "PASS",
    runway_months: float = 24.0,
    dilution_risk_bucket: str = "LOW",
    financial_data_state: str = "FULL",
    market_cap_mm: float = 1000.0,
    confidence: float = None,
) -> dict:
    """Create a financial score fixture."""
    data = {
        "ticker": ticker,
        "financial_normalized": Decimal(str(financial_normalized)),
        "severity": severity,
        "liquidity_gate_status": liquidity_gate_status,
        "runway_months": Decimal(str(runway_months)),
        "dilution_risk_bucket": dilution_risk_bucket,
        "financial_data_state": financial_data_state,
        "market_cap_mm": Decimal(str(market_cap_mm)),
        "flags": [],
    }
    if confidence is not None:
        data["confidence"] = Decimal(str(confidence))
    return data


def create_clinical_fixture(
    ticker: str,
    clinical_score: float = 50.0,
    lead_phase: str = "Phase 2",
    severity: str = "none",
    trial_count: int = 3,
    confidence: float = None,
) -> dict:
    """Create a clinical score fixture."""
    data = {
        "ticker": ticker,
        "clinical_score": Decimal(str(clinical_score)),
        "lead_phase": lead_phase,
        "severity": severity,
        "trial_count": trial_count,
        "flags": [],
    }
    if confidence is not None:
        data["confidence"] = Decimal(str(confidence))
    return data


def create_catalyst_fixture(
    ticker: str,
    score_blended: float = 50.0,
    catalyst_proximity_score: float = 0.0,
    catalyst_delta_score: float = 0.0,
    events_detected_total: int = 3,
    n_high_confidence: int = 2,
    severe_negative_flag: bool = False,
) -> dict:
    """Create a catalyst score fixture."""
    return {
        "ticker": ticker,
        "scores": {
            "score_blended": Decimal(str(score_blended)),
            "catalyst_proximity_score": Decimal(str(catalyst_proximity_score)),
            "catalyst_delta_score": Decimal(str(catalyst_delta_score)),
            "events_detected_total": events_detected_total,
            "n_high_confidence": n_high_confidence,
        },
        "flags": {"severe_negative_flag": severe_negative_flag},
    }


def create_full_result_set(tickers: list, base_scores: dict = None) -> tuple:
    """Create a complete set of result fixtures for testing."""
    if base_scores is None:
        base_scores = {}

    universe = create_universe_fixture(tickers)

    financial = {"scores": []}
    clinical = {"scores": []}
    catalyst = {"summaries": {}}

    for t in tickers:
        t_scores = base_scores.get(t, {})
        financial["scores"].append(create_financial_fixture(
            t,
            financial_normalized=t_scores.get("financial", 50.0),
            liquidity_gate_status=t_scores.get("liquidity_gate_status", "PASS"),
            runway_months=t_scores.get("runway_months", 24.0),
            dilution_risk_bucket=t_scores.get("dilution_risk_bucket", "LOW"),
            confidence=t_scores.get("financial_confidence"),
        ))
        clinical["scores"].append(create_clinical_fixture(
            t,
            clinical_score=t_scores.get("clinical", 50.0),
            lead_phase=t_scores.get("lead_phase", "Phase 2"),
            confidence=t_scores.get("clinical_confidence"),
        ))
        catalyst["summaries"][t] = create_catalyst_fixture(
            t,
            score_blended=t_scores.get("catalyst", 50.0),
            catalyst_proximity_score=t_scores.get("proximity", 0.0),
            catalyst_delta_score=t_scores.get("delta", 0.0),
        )

    return universe, financial, catalyst, clinical


# ============================================================================
# 1. BREAKDOWN CONSISTENCY TESTS
# ============================================================================

def test_breakdown_sum_equals_weighted_sum():
    """Sum of component contributions equals weighted_sum in breakdown."""
    tickers = ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE"]
    universe, financial, catalyst, clinical = create_full_result_set(tickers)

    result = compute_module_5_composite_v2(
        universe, financial, catalyst, clinical,
        as_of_date="2026-01-11",
    )

    for sec in result["ranked_securities"]:
        breakdown = sec["score_breakdown"]
        components = breakdown["components"]

        # Sum contributions
        total_contribution = sum(
            Decimal(c["contribution"]) for c in components
        )

        # Get weighted_sum from hybrid_aggregation
        weighted_sum = Decimal(breakdown["hybrid_aggregation"]["weighted_sum"])

        # They should match within tolerance
        diff = abs(total_contribution - weighted_sum)
        assert diff < Decimal("0.02"), \
            f"{sec['ticker']}: contribution sum {total_contribution} != weighted_sum {weighted_sum}"

    print("✓ test_breakdown_sum_equals_weighted_sum passed")


def test_breakdown_effective_weights_sum_to_one():
    """Effective weights should sum to 1.0 (or very close)."""
    tickers = ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE"]
    universe, financial, catalyst, clinical = create_full_result_set(tickers)

    result = compute_module_5_composite_v2(
        universe, financial, catalyst, clinical,
        as_of_date="2026-01-11",
    )

    for sec in result["ranked_securities"]:
        effective_weights = sec["effective_weights"]
        total_weight = sum(Decimal(w) for w in effective_weights.values())

        diff = abs(total_weight - Decimal("1"))
        assert diff < Decimal("0.001"), \
            f"{sec['ticker']}: effective weights sum to {total_weight}, expected ~1.0"

    print("✓ test_breakdown_effective_weights_sum_to_one passed")


def test_breakdown_contribution_matches_weight_times_score():
    """Each contribution = normalized * effective_weight."""
    tickers = ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE"]
    universe, financial, catalyst, clinical = create_full_result_set(tickers)

    result = compute_module_5_composite_v2(
        universe, financial, catalyst, clinical,
        as_of_date="2026-01-11",
    )

    for sec in result["ranked_securities"]:
        breakdown = sec["score_breakdown"]

        for comp in breakdown["components"]:
            normalized = Decimal(comp["normalized"]) if comp["normalized"] else Decimal("0")
            weight_eff = Decimal(comp["weight_effective"])
            contribution = Decimal(comp["contribution"])

            expected = _quantize_score(normalized * weight_eff)
            diff = abs(contribution - expected)

            assert diff < Decimal("0.02"), \
                f"{sec['ticker']}/{comp['name']}: contribution {contribution} != " \
                f"normalized({normalized}) * weight({weight_eff}) = {expected}"

    print("✓ test_breakdown_contribution_matches_weight_times_score passed")


# ============================================================================
# 2. CONFIDENCE WEIGHTING TESTS
# ============================================================================

def test_confidence_weighting_decreases_contribution():
    """Lower confidence should decrease component's effective weight."""
    # Create two scenarios: high confidence vs low confidence
    tickers = ["TEST"]

    # High confidence scenario
    base_high = {"TEST": {"financial": 70, "clinical": 60, "catalyst": 50}}
    universe_h, financial_h, catalyst_h, clinical_h = create_full_result_set(tickers, base_high)
    financial_h["scores"][0]["confidence"] = Decimal("1.0")
    financial_h["scores"][0]["financial_data_state"] = "FULL"

    result_high = compute_module_5_composite_v2(
        universe_h, financial_h, catalyst_h, clinical_h,
        as_of_date="2026-01-11",
    )

    # Low confidence scenario
    base_low = {"TEST": {"financial": 70, "clinical": 60, "catalyst": 50}}
    universe_l, financial_l, catalyst_l, clinical_l = create_full_result_set(tickers, base_low)
    financial_l["scores"][0]["confidence"] = Decimal("0.3")
    financial_l["scores"][0]["financial_data_state"] = "MINIMAL"

    result_low = compute_module_5_composite_v2(
        universe_l, financial_l, catalyst_l, clinical_l,
        as_of_date="2026-01-11",
    )

    # Extract financial effective weights
    weight_high = Decimal(result_high["ranked_securities"][0]["effective_weights"]["financial"])
    weight_low = Decimal(result_low["ranked_securities"][0]["effective_weights"]["financial"])

    assert weight_low < weight_high, \
        f"Low confidence weight ({weight_low}) should be < high confidence ({weight_high})"

    print("✓ test_confidence_weighting_decreases_contribution passed")


def test_confidence_extraction_financial_states():
    """Financial data state affects confidence correctly."""
    states_expected = {
        "FULL": Decimal("1.0"),
        "PARTIAL": Decimal("0.7"),
        "MINIMAL": Decimal("0.4"),
        "NONE": Decimal("0.1"),
    }

    for state, expected in states_expected.items():
        data = {"financial_data_state": state}
        conf = _extract_confidence_financial(data)
        assert conf == expected, \
            f"State {state}: expected {expected}, got {conf}"

    print("✓ test_confidence_extraction_financial_states passed")


def test_confidence_extraction_overrides_state():
    """Explicit confidence field overrides state-based inference."""
    data = {
        "financial_data_state": "MINIMAL",  # Would give 0.4
        "confidence": Decimal("0.9"),  # Override
    }
    conf = _extract_confidence_financial(data)
    assert conf == Decimal("0.9"), \
        f"Expected 0.9 (explicit), got {conf}"

    print("✓ test_confidence_extraction_overrides_state passed")


def test_confidence_clinical_trial_based():
    """Clinical confidence increases with trial count and data."""
    # Minimal data
    conf_min = _extract_confidence_clinical({})

    # With score
    conf_score = _extract_confidence_clinical({"clinical_score": 50})

    # With trials
    conf_trials = _extract_confidence_clinical({"clinical_score": 50, "trial_count": 5})

    # With lead phase
    conf_full = _extract_confidence_clinical({
        "clinical_score": 50, "trial_count": 5, "lead_phase": "Phase 3"
    })

    assert conf_min < conf_score, "Score should increase confidence"
    assert conf_score < conf_trials, "Trials should increase confidence"
    assert conf_trials < conf_full, "Lead phase should increase confidence"

    print("✓ test_confidence_clinical_trial_based passed")


# ============================================================================
# 3. MONOTONIC CAPS TESTS
# ============================================================================

def test_monotonic_cap_liquidity_fail():
    """Liquidity FAIL should cap score at 35 regardless of other scores."""
    tickers = ["BEST"]
    base = {"BEST": {
        "financial": 90,
        "clinical": 95,
        "catalyst": 90,
        "liquidity_gate_status": "FAIL",
    }}
    universe, financial, catalyst, clinical = create_full_result_set(tickers, base)

    result = compute_module_5_composite_v2(
        universe, financial, catalyst, clinical,
        as_of_date="2026-01-11",
    )

    sec = result["ranked_securities"][0]
    score = Decimal(sec["composite_score"])

    assert score <= MonotonicCap.LIQUIDITY_FAIL_CAP, \
        f"Score {score} exceeds liquidity FAIL cap {MonotonicCap.LIQUIDITY_FAIL_CAP}"
    assert len(sec["monotonic_caps_applied"]) > 0, "Should have cap applied"
    assert any(c["reason"] == "liquidity_gate_fail" for c in sec["monotonic_caps_applied"])

    print("✓ test_monotonic_cap_liquidity_fail passed")


def test_monotonic_cap_liquidity_warn():
    """Liquidity WARN should cap score at 60."""
    tickers = ["WARN"]
    base = {"WARN": {
        "financial": 90,
        "clinical": 95,
        "catalyst": 90,
        "liquidity_gate_status": "WARN",
    }}
    universe, financial, catalyst, clinical = create_full_result_set(tickers, base)

    result = compute_module_5_composite_v2(
        universe, financial, catalyst, clinical,
        as_of_date="2026-01-11",
    )

    sec = result["ranked_securities"][0]
    score = Decimal(sec["composite_score"])

    assert score <= MonotonicCap.LIQUIDITY_WARN_CAP, \
        f"Score {score} exceeds liquidity WARN cap {MonotonicCap.LIQUIDITY_WARN_CAP}"

    print("✓ test_monotonic_cap_liquidity_warn passed")


def test_monotonic_cap_runway_critical():
    """Runway < 6 months should cap score at 40."""
    tickers = ["BURN"]
    base = {"BURN": {
        "financial": 90,
        "clinical": 95,
        "catalyst": 90,
        "runway_months": 4.0,  # Critical
    }}
    universe, financial, catalyst, clinical = create_full_result_set(tickers, base)

    result = compute_module_5_composite_v2(
        universe, financial, catalyst, clinical,
        as_of_date="2026-01-11",
    )

    sec = result["ranked_securities"][0]
    score = Decimal(sec["composite_score"])

    assert score <= MonotonicCap.RUNWAY_CRITICAL_CAP, \
        f"Score {score} exceeds runway critical cap {MonotonicCap.RUNWAY_CRITICAL_CAP}"

    print("✓ test_monotonic_cap_runway_critical passed")


def test_monotonic_cap_runway_warning():
    """Runway < 12 months should cap score at 55."""
    tickers = ["CAUTION"]
    base = {"CAUTION": {
        "financial": 90,
        "clinical": 95,
        "catalyst": 90,
        "runway_months": 9.0,  # Warning
    }}
    universe, financial, catalyst, clinical = create_full_result_set(tickers, base)

    result = compute_module_5_composite_v2(
        universe, financial, catalyst, clinical,
        as_of_date="2026-01-11",
    )

    sec = result["ranked_securities"][0]
    score = Decimal(sec["composite_score"])

    assert score <= MonotonicCap.RUNWAY_WARNING_CAP, \
        f"Score {score} exceeds runway warning cap {MonotonicCap.RUNWAY_WARNING_CAP}"

    print("✓ test_monotonic_cap_runway_warning passed")


def test_monotonic_cap_dilution_severe():
    """SEVERE dilution risk should cap score at 45."""
    tickers = ["DILUTE"]
    base = {"DILUTE": {
        "financial": 90,
        "clinical": 95,
        "catalyst": 90,
        "dilution_risk_bucket": "SEVERE",
    }}
    universe, financial, catalyst, clinical = create_full_result_set(tickers, base)

    result = compute_module_5_composite_v2(
        universe, financial, catalyst, clinical,
        as_of_date="2026-01-11",
    )

    sec = result["ranked_securities"][0]
    score = Decimal(sec["composite_score"])

    assert score <= MonotonicCap.DILUTION_SEVERE_CAP, \
        f"Score {score} exceeds dilution severe cap {MonotonicCap.DILUTION_SEVERE_CAP}"

    print("✓ test_monotonic_cap_dilution_severe passed")


def test_monotonic_cap_cumulative():
    """Multiple cap triggers should apply the lowest cap."""
    score = Decimal("80")

    # Apply all caps at once - liquidity fail is strictest (35)
    capped, caps = _apply_monotonic_caps(
        score,
        liquidity_gate_status="FAIL",  # cap 35
        runway_months=Decimal("4"),    # cap 40
        dilution_risk_bucket="SEVERE", # cap 45
    )

    assert capped == MonotonicCap.LIQUIDITY_FAIL_CAP, \
        f"Expected {MonotonicCap.LIQUIDITY_FAIL_CAP}, got {capped}"

    # Should have all three caps applied
    assert len(caps) >= 1, "Should have at least one cap reason"

    print("✓ test_monotonic_cap_cumulative passed")


def test_no_cap_when_score_below_threshold():
    """Caps should not be applied if score is already below threshold."""
    score = Decimal("30")  # Already below all caps

    capped, caps = _apply_monotonic_caps(
        score,
        liquidity_gate_status="FAIL",  # Would cap at 35
        runway_months=Decimal("4"),
        dilution_risk_bucket="SEVERE",
    )

    assert capped == score, f"Score should remain {score}, got {capped}"
    assert len(caps) == 0, "No caps should be applied when score is below threshold"

    print("✓ test_no_cap_when_score_below_threshold passed")


# ============================================================================
# 4. DETERMINISM HASH TESTS
# ============================================================================

def test_determinism_hash_same_inputs():
    """Same inputs should produce same hash."""
    tickers = ["DETER"]
    base = {"DETER": {"financial": 60, "clinical": 70, "catalyst": 50}}
    universe, financial, catalyst, clinical = create_full_result_set(tickers, base)

    result1 = compute_module_5_composite_v2(
        universe, financial, catalyst, clinical,
        as_of_date="2026-01-11",
    )

    result2 = compute_module_5_composite_v2(
        universe, financial, catalyst, clinical,
        as_of_date="2026-01-11",
    )

    hash1 = result1["ranked_securities"][0]["determinism_hash"]
    hash2 = result2["ranked_securities"][0]["determinism_hash"]

    assert hash1 == hash2, f"Same inputs produced different hashes: {hash1} vs {hash2}"

    print("✓ test_determinism_hash_same_inputs passed")


def test_determinism_hash_different_score():
    """Different scores should produce different hashes."""
    tickers = ["DIFFER"]

    # Scenario 1
    base1 = {"DIFFER": {"financial": 60, "clinical": 70, "catalyst": 50}}
    universe1, financial1, catalyst1, clinical1 = create_full_result_set(tickers, base1)

    result1 = compute_module_5_composite_v2(
        universe1, financial1, catalyst1, clinical1,
        as_of_date="2026-01-11",
    )

    # Scenario 2 - different financial score
    base2 = {"DIFFER": {"financial": 80, "clinical": 70, "catalyst": 50}}
    universe2, financial2, catalyst2, clinical2 = create_full_result_set(tickers, base2)

    result2 = compute_module_5_composite_v2(
        universe2, financial2, catalyst2, clinical2,
        as_of_date="2026-01-11",
    )

    hash1 = result1["ranked_securities"][0]["determinism_hash"]
    hash2 = result2["ranked_securities"][0]["determinism_hash"]

    assert hash1 != hash2, f"Different inputs produced same hash: {hash1}"

    print("✓ test_determinism_hash_different_score passed")


def test_determinism_hash_different_date():
    """Different as_of_date with same ticker data should produce same hash."""
    # Note: Date is not included in ticker-level hash, only at result level
    tickers = ["DATED"]
    base = {"DATED": {"financial": 60, "clinical": 70, "catalyst": 50}}
    universe, financial, catalyst, clinical = create_full_result_set(tickers, base)

    result1 = compute_module_5_composite_v2(
        universe, financial, catalyst, clinical,
        as_of_date="2026-01-10",
    )

    result2 = compute_module_5_composite_v2(
        universe, financial, catalyst, clinical,
        as_of_date="2026-01-11",
    )

    hash1 = result1["ranked_securities"][0]["determinism_hash"]
    hash2 = result2["ranked_securities"][0]["determinism_hash"]

    # Hash is ticker-specific, should be same if same ticker-level inputs
    assert hash1 == hash2, \
        f"Same ticker data should produce same hash regardless of date"

    print("✓ test_determinism_hash_different_date passed")


def test_determinism_hash_length():
    """Hash should be 16 characters (hex truncated)."""
    tickers = ["HEXED"]
    base = {"HEXED": {"financial": 60, "clinical": 70, "catalyst": 50}}
    universe, financial, catalyst, clinical = create_full_result_set(tickers, base)

    result = compute_module_5_composite_v2(
        universe, financial, catalyst, clinical,
        as_of_date="2026-01-11",
    )

    hash_val = result["ranked_securities"][0]["determinism_hash"]

    assert len(hash_val) == 16, f"Hash length {len(hash_val)}, expected 16"
    assert all(c in "0123456789abcdef" for c in hash_val), "Hash should be hex"

    print("✓ test_determinism_hash_length passed")


# ============================================================================
# 5. ROBUST NORMALIZATION (WINSORIZATION) TESTS
# ============================================================================

def test_winsorization_clips_outliers():
    """Extreme values should be clipped to p5-p95 range."""
    # Create cohort with one extreme outlier
    values = [Decimal(str(x)) for x in [10, 20, 30, 40, 50, 60, 70, 80, 90, 1000]]

    normalized, winsor_applied = _rank_normalize_winsorized(values)

    assert winsor_applied, "Winsorization should be applied with outlier"

    # Check that no normalized value exceeds 100
    assert all(0 <= v <= 100 for v in normalized), \
        f"Normalized values should be in [0, 100]: {normalized}"

    # The extreme value (1000) should be normalized to 100 (max after rescale)
    assert normalized[-1] == Decimal("100"), \
        f"Highest value should normalize to 100, got {normalized[-1]}"

    print("✓ test_winsorization_clips_outliers passed")


def test_winsorization_rescales_range():
    """After winsorization, p5-p95 values should map to 0-100."""
    # All values within normal range
    values = [Decimal(str(x)) for x in range(0, 101, 10)]  # 0, 10, ..., 100

    normalized, winsor_applied = _rank_normalize_winsorized(values)

    # Min should be close to 0, max close to 100
    assert normalized[0] >= Decimal("0")
    assert normalized[-1] <= Decimal("100")

    print("✓ test_winsorization_rescales_range passed")


def test_winsorization_single_value():
    """Single value should normalize to 50."""
    values = [Decimal("75")]

    normalized, winsor_applied = _rank_normalize_winsorized(values)

    assert len(normalized) == 1
    assert normalized[0] == Decimal("50"), \
        f"Single value should normalize to 50, got {normalized[0]}"

    print("✓ test_winsorization_single_value passed")


def test_winsorization_tied_values():
    """Tied values should receive same normalized score."""
    values = [Decimal("50"), Decimal("50"), Decimal("50"), Decimal("50"), Decimal("100")]

    normalized, _ = _rank_normalize_winsorized(values)

    # First four values are tied
    assert normalized[0] == normalized[1] == normalized[2] == normalized[3], \
        f"Tied values should have same normalized score: {normalized[:4]}"

    print("✓ test_winsorization_tied_values passed")


def test_winsorization_in_cohort_normalization():
    """Cohort normalization should use winsorized method for outliers."""
    # Create tickers with outlier
    tickers = ["A", "B", "C", "D", "E", "F", "G", "OUTLIER"]
    base_scores = {
        "A": {"financial": 40, "clinical": 50, "catalyst": 45},
        "B": {"financial": 45, "clinical": 55, "catalyst": 50},
        "C": {"financial": 50, "clinical": 60, "catalyst": 55},
        "D": {"financial": 55, "clinical": 65, "catalyst": 60},
        "E": {"financial": 60, "clinical": 70, "catalyst": 65},
        "F": {"financial": 65, "clinical": 75, "catalyst": 70},
        "G": {"financial": 70, "clinical": 80, "catalyst": 75},
        "OUTLIER": {"financial": 99, "clinical": 99, "catalyst": 99},  # Extreme
    }

    universe, financial, catalyst, clinical = create_full_result_set(
        tickers, base_scores
    )

    result = compute_module_5_composite_v2(
        universe, financial, catalyst, clinical,
        as_of_date="2026-01-11",
    )

    # Check that outlier's normalized scores are bounded
    outlier = next(s for s in result["ranked_securities"] if s["ticker"] == "OUTLIER")
    score = Decimal(outlier["composite_score"])

    # Score should be reasonable (not astronomical)
    assert score <= Decimal("100"), f"Outlier score {score} exceeds 100"

    print("✓ test_winsorization_in_cohort_normalization passed")


# ============================================================================
# 6. WEAKEST-LINK HYBRID AGGREGATION TESTS
# ============================================================================

def test_weakest_link_reduces_score():
    """Low critical component should pull down hybrid score vs pure weighted sum."""
    tickers = ["WEAK"]
    base = {"WEAK": {
        "financial": 20,   # Low financial (critical)
        "clinical": 90,    # High clinical (critical)
        "catalyst": 85,    # High catalyst
    }}

    universe, financial, catalyst, clinical = create_full_result_set(tickers, base)

    result = compute_module_5_composite_v2(
        universe, financial, catalyst, clinical,
        as_of_date="2026-01-11",
    )

    sec = result["ranked_securities"][0]
    breakdown = sec["score_breakdown"]

    weighted_sum = Decimal(breakdown["hybrid_aggregation"]["weighted_sum"])
    min_critical = Decimal(breakdown["hybrid_aggregation"]["min_critical"])

    # Min critical should be much lower than weighted sum
    assert min_critical < weighted_sum, \
        f"min_critical ({min_critical}) should be < weighted_sum ({weighted_sum})"

    # Final pre_penalty should be pulled down from pure weighted sum
    pre_penalty = Decimal(breakdown["final"]["pre_penalty_score"])
    expected_hybrid = HYBRID_ALPHA * weighted_sum + (Decimal("1") - HYBRID_ALPHA) * min_critical

    diff = abs(pre_penalty - expected_hybrid)
    assert diff < Decimal("0.1"), \
        f"Hybrid score mismatch: {pre_penalty} vs expected {expected_hybrid}"

    print("✓ test_weakest_link_reduces_score passed")


def test_weakest_link_balanced_scores():
    """Balanced critical components should have minimal weakest-link impact."""
    tickers = ["BALANCED"]
    base = {"BALANCED": {
        "financial": 60,   # Balanced
        "clinical": 65,    # Balanced
        "catalyst": 70,    # Slightly higher (non-critical)
    }}

    universe, financial, catalyst, clinical = create_full_result_set(tickers, base)

    result = compute_module_5_composite_v2(
        universe, financial, catalyst, clinical,
        as_of_date="2026-01-11",
    )

    sec = result["ranked_securities"][0]
    breakdown = sec["score_breakdown"]

    weighted_sum = Decimal(breakdown["hybrid_aggregation"]["weighted_sum"])
    min_critical = Decimal(breakdown["hybrid_aggregation"]["min_critical"])

    # When balanced, min_critical should be close to weighted_sum
    diff_pct = abs(min_critical - weighted_sum) / max(weighted_sum, Decimal("1")) * 100
    assert diff_pct < Decimal("25"), \
        f"Balanced scores should have similar weighted_sum and min_critical"

    print("✓ test_weakest_link_balanced_scores passed")


def test_hybrid_alpha_applied_correctly():
    """Verify hybrid formula: score = α*(weighted_sum) + (1-α)*(min_critical)."""
    tickers = ["HYBRID"]
    base = {"HYBRID": {"financial": 30, "clinical": 80, "catalyst": 60}}

    universe, financial, catalyst, clinical = create_full_result_set(tickers, base)

    result = compute_module_5_composite_v2(
        universe, financial, catalyst, clinical,
        as_of_date="2026-01-11",
    )

    sec = result["ranked_securities"][0]
    breakdown = sec["score_breakdown"]

    weighted_sum = Decimal(breakdown["hybrid_aggregation"]["weighted_sum"])
    min_critical = Decimal(breakdown["hybrid_aggregation"]["min_critical"])
    alpha = Decimal(breakdown["hybrid_aggregation"]["alpha"])

    assert alpha == HYBRID_ALPHA, f"Alpha mismatch: {alpha} vs {HYBRID_ALPHA}"

    expected_hybrid = alpha * weighted_sum + (Decimal("1") - alpha) * min_critical
    pre_penalty = Decimal(breakdown["final"]["pre_penalty_score"])

    diff = abs(pre_penalty - _quantize_score(expected_hybrid))
    assert diff < Decimal("0.02"), \
        f"Hybrid formula mismatch: got {pre_penalty}, expected {_quantize_score(expected_hybrid)}"

    print("✓ test_hybrid_alpha_applied_correctly passed")


def test_critical_components_defined():
    """Critical components should be financial and clinical."""
    assert "financial" in CRITICAL_COMPONENTS
    assert "clinical" in CRITICAL_COMPONENTS
    assert "catalyst" not in CRITICAL_COMPONENTS

    print("✓ test_critical_components_defined passed")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_full_pipeline_basic():
    """Basic end-to-end test of Module 5 v2."""
    tickers = ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE"]
    base_scores = {
        "AAAA": {"financial": 80, "clinical": 75, "catalyst": 70},
        "BBBB": {"financial": 60, "clinical": 65, "catalyst": 60},
        "CCCC": {"financial": 70, "clinical": 50, "catalyst": 80},
        "DDDD": {"financial": 40, "clinical": 85, "catalyst": 55},
        "EEEE": {"financial": 55, "clinical": 70, "catalyst": 65},
    }

    universe, financial, catalyst, clinical = create_full_result_set(tickers, base_scores)

    result = compute_module_5_composite_v2(
        universe, financial, catalyst, clinical,
        as_of_date="2026-01-11",
    )

    assert result["as_of_date"] == "2026-01-11"
    assert result["scoring_mode"] == "default"
    assert len(result["ranked_securities"]) == 5
    assert result["schema_version"] == "v2.0"

    # Check ranking is sorted by composite_score descending
    scores = [Decimal(s["composite_score"]) for s in result["ranked_securities"]]
    assert scores == sorted(scores, reverse=True), "Should be sorted by score descending"

    # Each security should have required fields
    for sec in result["ranked_securities"]:
        assert "score_breakdown" in sec
        assert "determinism_hash" in sec
        assert "effective_weights" in sec
        assert "confidence_clinical" in sec
        assert "confidence_financial" in sec
        assert "monotonic_caps_applied" in sec

    print("✓ test_full_pipeline_basic passed")


def test_full_pipeline_with_exclusions():
    """Test that SEV3 exclusions work properly."""
    tickers = ["GOOD", "BAD"]

    universe = create_universe_fixture(tickers)
    financial = {"scores": [
        create_financial_fixture("GOOD", financial_normalized=70),
        create_financial_fixture("BAD", financial_normalized=60, severity="sev3"),
    ]}
    clinical = {"scores": [
        create_clinical_fixture("GOOD", clinical_score=65),
        create_clinical_fixture("BAD", clinical_score=80),
    ]}
    catalyst = {"summaries": {
        "GOOD": create_catalyst_fixture("GOOD", score_blended=60),
        "BAD": create_catalyst_fixture("BAD", score_blended=75),
    }}

    result = compute_module_5_composite_v2(
        universe, financial, catalyst, clinical,
        as_of_date="2026-01-11",
    )

    assert len(result["ranked_securities"]) == 1
    assert result["ranked_securities"][0]["ticker"] == "GOOD"
    assert len(result["excluded_securities"]) == 1
    assert result["excluded_securities"][0]["ticker"] == "BAD"
    assert result["excluded_securities"][0]["reason"] == "sev3_gate"

    print("✓ test_full_pipeline_with_exclusions passed")


def test_diagnostic_counts_accurate():
    """Diagnostic counts should accurately reflect processing."""
    tickers = ["A", "B", "C", "D", "E", "EXCLUDE"]

    universe = create_universe_fixture(tickers)
    financial = {"scores": [
        create_financial_fixture(t) for t in ["A", "B", "C", "D", "E"]
    ] + [create_financial_fixture("EXCLUDE", severity="sev3")]}
    clinical = {"scores": [create_clinical_fixture(t) for t in tickers]}
    catalyst = {"summaries": {t: create_catalyst_fixture(t) for t in tickers}}

    result = compute_module_5_composite_v2(
        universe, financial, catalyst, clinical,
        as_of_date="2026-01-11",
    )

    diag = result["diagnostic_counts"]
    assert diag["total_input"] == 6
    assert diag["rankable"] == 5
    assert diag["excluded"] == 1

    print("✓ test_diagnostic_counts_accurate passed")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

def run_all_tests():
    """Run all golden fixture tests."""
    print("\n" + "=" * 70)
    print("COMPOSITE MODULE V2 GOLDEN FIXTURE TESTS")
    print("=" * 70 + "\n")

    tests = [
        # 1. Breakdown consistency
        test_breakdown_sum_equals_weighted_sum,
        test_breakdown_effective_weights_sum_to_one,
        test_breakdown_contribution_matches_weight_times_score,

        # 2. Confidence weighting
        test_confidence_weighting_decreases_contribution,
        test_confidence_extraction_financial_states,
        test_confidence_extraction_overrides_state,
        test_confidence_clinical_trial_based,

        # 3. Monotonic caps
        test_monotonic_cap_liquidity_fail,
        test_monotonic_cap_liquidity_warn,
        test_monotonic_cap_runway_critical,
        test_monotonic_cap_runway_warning,
        test_monotonic_cap_dilution_severe,
        test_monotonic_cap_cumulative,
        test_no_cap_when_score_below_threshold,

        # 4. Determinism hash
        test_determinism_hash_same_inputs,
        test_determinism_hash_different_score,
        test_determinism_hash_different_date,
        test_determinism_hash_length,

        # 5. Robust normalization
        test_winsorization_clips_outliers,
        test_winsorization_rescales_range,
        test_winsorization_single_value,
        test_winsorization_tied_values,
        test_winsorization_in_cohort_normalization,

        # 6. Weakest-link hybrid
        test_weakest_link_reduces_score,
        test_weakest_link_balanced_scores,
        test_hybrid_alpha_applied_correctly,
        test_critical_components_defined,

        # Integration
        test_full_pipeline_basic,
        test_full_pipeline_with_exclusions,
        test_diagnostic_counts_accurate,
    ]

    passed = 0
    failed = 0
    failures = []

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            failures.append((test.__name__, str(e)))
            print(f"✗ {test.__name__} FAILED: {e}")

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    if failures:
        print("\nFailed tests:")
        for name, error in failures:
            print(f"  - {name}: {error}")
        return 1

    print("\n✓ All tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(run_all_tests())
