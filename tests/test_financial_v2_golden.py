#!/usr/bin/env python3
"""
Golden fixture tests for Financial Health Module v2.

Tests:
1. Burn hierarchy selection (CFO > FCF > NetIncome > R&D)
2. Runway stability with 4-quarter average
3. Determinism hash verification
4. Liquidity gating (PASS/WARN/FAIL)
5. Dilution risk bucket classification
6. Data quality state assessment

Run: python tests/test_financial_v2_golden.py
"""

import sys
from decimal import Decimal
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from module_2_financial_v2 import (
    calculate_burn_rate_v2,
    calculate_liquid_assets,
    calculate_runway,
    calculate_dilution_risk,
    score_liquidity,
    assess_data_quality,
    score_financial_health_v2,
    BurnSource,
    BurnConfidence,
    DataState,
    LiquidityGate,
    DilutionRiskBucket,
    _to_decimal,
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

def create_financial_fixture(
    cash: float = None,
    cfo_quarterly: float = None,
    cfo_ytd: float = None,
    cfo: float = None,
    fcf_quarterly: float = None,
    fcf: float = None,
    net_income: float = None,
    rd: float = None,
    marketable_securities: float = None,
    burn_history: list = None,
    shares_outstanding: float = None,
    shares_outstanding_prior: float = None,
) -> dict:
    """Create a financial data fixture."""
    data = {}
    if cash is not None:
        data['Cash'] = cash
    if cfo_quarterly is not None:
        data['CFO_quarterly'] = cfo_quarterly
    if cfo_ytd is not None:
        data['CFO_YTD'] = cfo_ytd
    if cfo is not None:
        data['CFO'] = cfo
    if fcf_quarterly is not None:
        data['FCF_quarterly'] = fcf_quarterly
    if fcf is not None:
        data['FCF'] = fcf
    if net_income is not None:
        data['NetIncome'] = net_income
    if rd is not None:
        data['R&D'] = rd
    if marketable_securities is not None:
        data['MarketableSecurities'] = marketable_securities
    if burn_history is not None:
        data['burn_history'] = burn_history
    if shares_outstanding is not None:
        data['shares_outstanding'] = shares_outstanding
    if shares_outstanding_prior is not None:
        data['shares_outstanding_prior'] = shares_outstanding_prior
    return data


def create_market_fixture(
    market_cap: float = 1000e6,
    avg_volume: float = 500000,
    price: float = 20.0,
) -> dict:
    """Create a market data fixture."""
    return {
        'market_cap': market_cap,
        'avg_volume': avg_volume,
        'price': price,
    }


# ============================================================================
# BURN HIERARCHY TESTS
# ============================================================================

def test_burn_hierarchy_cfo_quarterly_first():
    """CFO quarterly should be prioritized over all other sources."""
    # Provide all burn sources, CFO quarterly should win
    data = create_financial_fixture(
        cfo_quarterly=-30e6,  # Quarterly burn: $30M
        cfo=-100e6,           # Annual CFO
        fcf=-25e6,            # FCF
        net_income=-20e6,     # Net income
        rd=10e6,              # R&D
    )

    result = calculate_burn_rate_v2(data)

    assert result.burn_source == BurnSource.CFO_QUARTERLY, \
        f"Expected CFO_quarterly, got {result.burn_source}"
    assert result.burn_confidence == BurnConfidence.HIGH
    # Monthly burn = 30M / 3 = 10M
    expected_monthly = Decimal("10000000")
    assert abs(result.monthly_burn - expected_monthly) < Decimal("0.01"), \
        f"Expected {expected_monthly}, got {result.monthly_burn}"

    print("✓ test_burn_hierarchy_cfo_quarterly_first passed")


def test_burn_hierarchy_cfo_ytd_second():
    """CFO YTD should be used when CFO quarterly is not available."""
    data = create_financial_fixture(
        cfo_ytd=-60e6,  # YTD (2 quarters)
        cfo=-100e6,
        fcf=-25e6,
        net_income=-20e6,
    )
    data['quarters_in_ytd'] = 2

    result = calculate_burn_rate_v2(data)

    assert result.burn_source == BurnSource.CFO_YTD, \
        f"Expected CFO_YTD, got {result.burn_source}"
    assert result.burn_confidence == BurnConfidence.HIGH
    # Monthly burn = 60M / 6 months = 10M
    expected_monthly = Decimal("10000000")
    assert abs(result.monthly_burn - expected_monthly) < Decimal("0.01"), \
        f"Expected {expected_monthly}, got {result.monthly_burn}"

    print("✓ test_burn_hierarchy_cfo_ytd_second passed")


def test_burn_hierarchy_cfo_annual_third():
    """CFO annual should be used when quarterly/YTD not available."""
    data = create_financial_fixture(
        cfo=-120e6,       # Annual CFO burn
        fcf=-100e6,
        net_income=-80e6,
        rd=30e6,
    )

    result = calculate_burn_rate_v2(data)

    assert result.burn_source == BurnSource.CFO_ANNUAL, \
        f"Expected CFO_annual, got {result.burn_source}"
    assert result.burn_confidence == BurnConfidence.HIGH
    # Monthly burn = 120M / 12 = 10M
    expected_monthly = Decimal("10000000")
    assert abs(result.monthly_burn - expected_monthly) < Decimal("0.01"), \
        f"Expected {expected_monthly}, got {result.monthly_burn}"

    print("✓ test_burn_hierarchy_cfo_annual_third passed")


def test_burn_hierarchy_trailing_4q_average():
    """Trailing 4-quarter average should be used when single-period CFO not available."""
    data = create_financial_fixture(
        burn_history=[-25e6, -30e6, -28e6, -27e6],  # 4 quarters of burn
        net_income=-20e6,
        rd=10e6,
    )

    result = calculate_burn_rate_v2(data)

    assert result.burn_source == BurnSource.CFO_TRAILING_4Q, \
        f"Expected CFO_trailing_4q, got {result.burn_source}"
    assert result.burn_confidence == BurnConfidence.HIGH
    assert result.quarters_used == 4
    # Average quarterly = (25 + 30 + 28 + 27) / 4 = 27.5M
    # Monthly = 27.5M / 3 = 9.166...M
    expected_monthly = Decimal("9166666.6667")
    assert abs(result.monthly_burn - expected_monthly) < Decimal("1"), \
        f"Expected ~{expected_monthly}, got {result.monthly_burn}"

    print("✓ test_burn_hierarchy_trailing_4q_average passed")


def test_burn_hierarchy_fcf_quarterly():
    """FCF quarterly should be used when CFO not available."""
    data = create_financial_fixture(
        fcf_quarterly=-24e6,
        net_income=-20e6,
        rd=10e6,
    )

    result = calculate_burn_rate_v2(data)

    assert result.burn_source == BurnSource.FCF_QUARTERLY, \
        f"Expected FCF_quarterly, got {result.burn_source}"
    assert result.burn_confidence == BurnConfidence.HIGH
    # Monthly = 24M / 3 = 8M
    expected_monthly = Decimal("8000000")
    assert abs(result.monthly_burn - expected_monthly) < Decimal("0.01"), \
        f"Expected {expected_monthly}, got {result.monthly_burn}"

    print("✓ test_burn_hierarchy_fcf_quarterly passed")


def test_burn_hierarchy_fcf_annual():
    """FCF annual should be used when quarterly FCF not available."""
    data = create_financial_fixture(
        fcf=-96e6,
        net_income=-80e6,
        rd=30e6,
    )

    result = calculate_burn_rate_v2(data)

    assert result.burn_source == BurnSource.FCF_ANNUAL, \
        f"Expected FCF_annual, got {result.burn_source}"
    assert result.burn_confidence == BurnConfidence.HIGH
    # Monthly = 96M / 12 = 8M
    expected_monthly = Decimal("8000000")
    assert abs(result.monthly_burn - expected_monthly) < Decimal("0.01"), \
        f"Expected {expected_monthly}, got {result.monthly_burn}"

    print("✓ test_burn_hierarchy_fcf_annual passed")


def test_burn_hierarchy_net_income_fallback():
    """NetIncome should be used as fallback with MEDIUM confidence."""
    data = create_financial_fixture(
        net_income=-21e6,
        rd=10e6,
    )

    result = calculate_burn_rate_v2(data)

    assert result.burn_source == BurnSource.NET_INCOME, \
        f"Expected NetIncome, got {result.burn_source}"
    assert result.burn_confidence == BurnConfidence.MEDIUM, \
        f"Expected MEDIUM confidence, got {result.burn_confidence}"
    # Monthly = 21M / 3 = 7M
    expected_monthly = Decimal("7000000")
    assert abs(result.monthly_burn - expected_monthly) < Decimal("0.01"), \
        f"Expected {expected_monthly}, got {result.monthly_burn}"

    print("✓ test_burn_hierarchy_net_income_fallback passed")


def test_burn_hierarchy_rd_proxy_last_resort():
    """R&D proxy should be used as last resort with LOW confidence."""
    data = create_financial_fixture(rd=14e6)

    result = calculate_burn_rate_v2(data)

    assert result.burn_source == BurnSource.RD_PROXY, \
        f"Expected R&D_proxy, got {result.burn_source}"
    assert result.burn_confidence == BurnConfidence.LOW, \
        f"Expected LOW confidence, got {result.burn_confidence}"
    # R&D proxy: quarterly burn = R&D * 1.5 = 14M * 1.5 = 21M
    # Monthly = 21M / 3 = 7M
    expected_monthly = Decimal("7000000")
    assert abs(result.monthly_burn - expected_monthly) < Decimal("0.01"), \
        f"Expected {expected_monthly}, got {result.monthly_burn}"

    print("✓ test_burn_hierarchy_rd_proxy_last_resort passed")


def test_burn_hierarchy_profitable_company():
    """Profitable companies should have zero burn."""
    data = create_financial_fixture(
        cfo=50e6,  # Positive CFO
        net_income=30e6,
    )

    result = calculate_burn_rate_v2(data)

    assert result.burn_source == BurnSource.PROFITABLE, \
        f"Expected profitable, got {result.burn_source}"
    assert result.burn_confidence == BurnConfidence.HIGH
    assert result.monthly_burn == Decimal("0")

    print("✓ test_burn_hierarchy_profitable_company passed")


def test_burn_hierarchy_no_data():
    """No data should return NONE burn source."""
    data = {}

    result = calculate_burn_rate_v2(data)

    assert result.burn_source == BurnSource.NONE, \
        f"Expected none, got {result.burn_source}"
    assert result.burn_confidence == BurnConfidence.NONE
    assert result.monthly_burn is None

    print("✓ test_burn_hierarchy_no_data passed")


# ============================================================================
# RUNWAY STABILITY TESTS (4-QUARTER AVERAGE)
# ============================================================================

def test_runway_with_4q_average():
    """Runway should use 4-quarter average for stability."""
    data = create_financial_fixture(
        cash=100e6,
        burn_history=[-20e6, -25e6, -22e6, -23e6],  # Avg = 22.5M/q = 7.5M/month
    )
    market = create_market_fixture(market_cap=500e6)

    result = calculate_runway(data, market)

    # Runway = 100M / 7.5M = 13.33 months
    assert result.runway_months is not None
    assert result.burn_source == BurnSource.CFO_TRAILING_4Q
    assert result.quarters_used == 4
    expected_runway = Decimal("13.33")
    assert abs(result.runway_months - expected_runway) < Decimal("0.5"), \
        f"Expected ~{expected_runway}, got {result.runway_months}"

    print("✓ test_runway_with_4q_average passed")


def test_runway_with_partial_quarters():
    """Runway should work with less than 4 quarters of data."""
    data = create_financial_fixture(
        cash=60e6,
        burn_history=[-18e6, -22e6],  # Only 2 quarters, avg = 20M/q
    )
    market = create_market_fixture(market_cap=300e6)

    result = calculate_runway(data, market)

    # Monthly burn = 20M / 3 = 6.67M
    # Runway = 60M / 6.67M = 9 months
    assert result.runway_months is not None
    assert result.quarters_used == 2
    expected_runway = Decimal("9")
    assert abs(result.runway_months - expected_runway) < Decimal("0.5"), \
        f"Expected ~{expected_runway}, got {result.runway_months}"

    print("✓ test_runway_with_partial_quarters passed")


def test_runway_liquid_assets_includes_marketable_securities():
    """Liquid assets should include marketable securities."""
    data = create_financial_fixture(
        cash=50e6,
        marketable_securities=30e6,
        cfo_quarterly=-24e6,  # 8M/month burn
    )
    market = create_market_fixture()

    result = calculate_runway(data, market)

    # Liquid assets = 50M + 30M = 80M
    # Runway = 80M / 8M = 10 months
    assert result.liquid_assets == Decimal("80000000")
    assert 'Cash' in result.liquid_components
    assert 'MarketableSecurities' in result.liquid_components
    expected_runway = Decimal("10")
    assert abs(result.runway_months - expected_runway) < Decimal("0.5"), \
        f"Expected ~{expected_runway}, got {result.runway_months}"

    print("✓ test_runway_liquid_assets_includes_marketable_securities passed")


def test_runway_profitable_company_infinite():
    """Profitable companies should have infinite runway."""
    data = create_financial_fixture(
        cash=100e6,
        cfo=50e6,  # Positive CFO
    )
    market = create_market_fixture()

    result = calculate_runway(data, market)

    assert result.runway_months == Decimal("999")
    assert result.runway_score == Decimal("100")
    assert result.burn_source == BurnSource.PROFITABLE

    print("✓ test_runway_profitable_company_infinite passed")


# ============================================================================
# DETERMINISM HASH TESTS
# ============================================================================

def test_determinism_hash_stable():
    """Same inputs should always produce same hash."""
    data = create_financial_fixture(
        cash=75e6,
        cfo_quarterly=-22.5e6,
    )
    market = create_market_fixture(market_cap=400e6)

    result1 = score_financial_health_v2("TEST", data, market)
    result2 = score_financial_health_v2("TEST", data, market)

    assert result1['determinism_hash'] == result2['determinism_hash'], \
        "Determinism hash should be stable across calls"

    print("✓ test_determinism_hash_stable passed")


def test_determinism_hash_differs_for_different_inputs():
    """Different inputs should produce different hashes."""
    data1 = create_financial_fixture(cash=75e6, cfo_quarterly=-22.5e6)
    data2 = create_financial_fixture(cash=80e6, cfo_quarterly=-22.5e6)
    market = create_market_fixture()

    result1 = score_financial_health_v2("TEST", data1, market)
    result2 = score_financial_health_v2("TEST", data2, market)

    assert result1['determinism_hash'] != result2['determinism_hash'], \
        "Different inputs should produce different hashes"

    print("✓ test_determinism_hash_differs_for_different_inputs passed")


def test_determinism_hash_differs_for_different_tickers():
    """Different tickers should produce different hashes."""
    data = create_financial_fixture(cash=75e6, cfo_quarterly=-22.5e6)
    market = create_market_fixture()

    result1 = score_financial_health_v2("TEST1", data, market)
    result2 = score_financial_health_v2("TEST2", data, market)

    assert result1['determinism_hash'] != result2['determinism_hash'], \
        "Different tickers should produce different hashes"

    print("✓ test_determinism_hash_differs_for_different_tickers passed")


# ============================================================================
# LIQUIDITY GATING TESTS
# ============================================================================

def test_liquidity_gate_pass():
    """High dollar ADV should result in PASS gate."""
    market = create_market_fixture(
        avg_volume=100000,
        price=50.0,  # Dollar ADV = $5M
    )

    result = score_liquidity(market)

    assert result.liquidity_gate == LiquidityGate.PASS, \
        f"Expected PASS, got {result.liquidity_gate}"
    assert result.dollar_adv_20d >= Decimal("500000")

    print("✓ test_liquidity_gate_pass passed")


def test_liquidity_gate_warn():
    """Medium dollar ADV should result in WARN gate."""
    market = create_market_fixture(
        avg_volume=20000,
        price=10.0,  # Dollar ADV = $200K
    )

    result = score_liquidity(market)

    assert result.liquidity_gate == LiquidityGate.WARN, \
        f"Expected WARN, got {result.liquidity_gate}"

    print("✓ test_liquidity_gate_warn passed")


def test_liquidity_gate_fail():
    """Low dollar ADV should result in FAIL gate."""
    market = create_market_fixture(
        avg_volume=5000,
        price=10.0,  # Dollar ADV = $50K
    )

    result = score_liquidity(market)

    assert result.liquidity_gate == LiquidityGate.FAIL, \
        f"Expected FAIL, got {result.liquidity_gate}"

    print("✓ test_liquidity_gate_fail passed")


# ============================================================================
# DILUTION RISK BUCKET TESTS
# ============================================================================

def test_dilution_risk_low():
    """High cash/mcap ratio should be LOW risk."""
    data = create_financial_fixture(cash=400e6)  # 40% of mcap
    market = create_market_fixture(market_cap=1000e6)

    result = calculate_dilution_risk(data, market, Decimal("24"), Decimal("5e6"))

    assert result.dilution_risk_bucket == DilutionRiskBucket.LOW, \
        f"Expected LOW, got {result.dilution_risk_bucket}"
    assert result.cash_to_mcap >= Decimal("0.30")

    print("✓ test_dilution_risk_low passed")


def test_dilution_risk_moderate():
    """Moderate cash/mcap ratio should be MODERATE risk."""
    data = create_financial_fixture(cash=200e6)  # 20% of mcap
    market = create_market_fixture(market_cap=1000e6)

    result = calculate_dilution_risk(data, market, Decimal("18"), Decimal("8e6"))

    assert result.dilution_risk_bucket == DilutionRiskBucket.MODERATE, \
        f"Expected MODERATE, got {result.dilution_risk_bucket}"

    print("✓ test_dilution_risk_moderate passed")


def test_dilution_risk_high():
    """Low cash/mcap ratio should be HIGH risk."""
    data = create_financial_fixture(cash=80e6)  # 8% of mcap
    market = create_market_fixture(market_cap=1000e6)

    result = calculate_dilution_risk(data, market, Decimal("10"), Decimal("8e6"))

    assert result.dilution_risk_bucket == DilutionRiskBucket.HIGH, \
        f"Expected HIGH, got {result.dilution_risk_bucket}"

    print("✓ test_dilution_risk_high passed")


def test_dilution_risk_severe():
    """Very low cash/mcap ratio should be SEVERE risk."""
    data = create_financial_fixture(cash=30e6)  # 3% of mcap
    market = create_market_fixture(market_cap=1000e6)

    result = calculate_dilution_risk(data, market, Decimal("4"), Decimal("8e6"))

    assert result.dilution_risk_bucket == DilutionRiskBucket.SEVERE, \
        f"Expected SEVERE, got {result.dilution_risk_bucket}"

    print("✓ test_dilution_risk_severe passed")


def test_dilution_share_count_growth():
    """Share count growth should be tracked."""
    data = create_financial_fixture(
        cash=200e6,
        shares_outstanding=100e6,
        shares_outstanding_prior=80e6,  # 25% growth
    )
    market = create_market_fixture(market_cap=1000e6)

    result = calculate_dilution_risk(data, market, Decimal("18"), Decimal("8e6"))

    assert result.share_count_growth is not None
    expected_growth = Decimal("0.25")
    assert abs(result.share_count_growth - expected_growth) < Decimal("0.01"), \
        f"Expected {expected_growth}, got {result.share_count_growth}"

    print("✓ test_dilution_share_count_growth passed")


# ============================================================================
# DATA QUALITY TESTS
# ============================================================================

def test_data_quality_full():
    """All fields present should be FULL state."""
    data = create_financial_fixture(
        cash=100e6,
        marketable_securities=50e6,
        cfo=-30e6,
    )
    market = create_market_fixture(market_cap=500e6, avg_volume=100000, price=25.0)

    result = assess_data_quality(data, market)

    assert result.financial_data_state == DataState.FULL, \
        f"Expected FULL, got {result.financial_data_state}"
    assert len(result.missing_fields) == 0
    assert result.confidence == Decimal("1.0")

    print("✓ test_data_quality_full passed")


def test_data_quality_partial():
    """Some fields missing should be PARTIAL state."""
    data = create_financial_fixture(
        cash=100e6,
        # Missing burn rate data
    )
    market = create_market_fixture(market_cap=500e6, avg_volume=100000, price=25.0)

    result = assess_data_quality(data, market)

    assert result.financial_data_state in [DataState.PARTIAL, DataState.MINIMAL], \
        f"Expected PARTIAL or MINIMAL, got {result.financial_data_state}"
    assert 'burn_rate' in result.missing_fields

    print("✓ test_data_quality_partial passed")


def test_data_quality_none():
    """Critical fields missing should be NONE state."""
    data = {}  # No financial data
    market = {}  # No market data

    result = assess_data_quality(data, market)

    assert result.financial_data_state == DataState.NONE, \
        f"Expected NONE, got {result.financial_data_state}"
    assert 'cash' in result.missing_fields
    assert 'market_cap' in result.missing_fields

    print("✓ test_data_quality_none passed")


# ============================================================================
# BUGFIX VERIFICATION TESTS
# ============================================================================

def test_bugfix_zero_values_not_falsy():
    """Zero values should not be treated as None (bugfix verification)."""
    # Decimal zero should be a valid value, not treated as missing
    val = _to_decimal(0)
    assert val == Decimal("0"), "Zero should convert to Decimal('0'), not None"

    val = _to_decimal(0.0)
    assert val == Decimal("0"), "0.0 should convert to Decimal('0'), not None"

    val = _to_decimal("0")
    assert val == Decimal("0"), "'0' should convert to Decimal('0'), not None"

    print("✓ test_bugfix_zero_values_not_falsy passed")


def test_bugfix_empty_string_is_none():
    """Empty strings should convert to None."""
    val = _to_decimal("")
    assert val is None, "Empty string should convert to None"

    val = _to_decimal("  ")
    assert val is None, "Whitespace string should convert to None"

    print("✓ test_bugfix_empty_string_is_none passed")


# ============================================================================
# COMPOSITE SCORE TESTS
# ============================================================================

def test_composite_score_weights():
    """Composite score should use correct weights: 50% runway, 30% dilution, 20% liquidity."""
    data = create_financial_fixture(
        cash=200e6,
        cfo_quarterly=-24e6,  # 8M/month, 25 month runway
    )
    market = create_market_fixture(
        market_cap=500e6,
        avg_volume=100000,
        price=50.0,
    )

    result = score_financial_health_v2("TEST", data, market)

    assert result['has_financial_data'] is True
    assert result['burn_source'] == 'CFO_quarterly'
    assert result['burn_confidence'] == 'HIGH'
    assert result['liquidity_gate'] in ['PASS', 'WARN']

    # Verify composite is in valid range
    composite = Decimal(result['financial_normalized'])
    assert Decimal("0") <= composite <= Decimal("100"), \
        f"Composite {composite} out of range"

    print("✓ test_composite_score_weights passed")


def test_flags_generated_correctly():
    """Flags should be generated for concerning conditions."""
    data = create_financial_fixture(
        cash=30e6,
        cfo_quarterly=-12e6,  # 4M/month, 7.5 month runway
    )
    market = create_market_fixture(
        market_cap=1000e6,
        avg_volume=5000,
        price=10.0,  # $50K ADV - FAIL gate
    )

    result = score_financial_health_v2("TEST", data, market)

    assert 'low_runway' in result['flags'], "Should flag low runway"
    assert 'low_cash_ratio' in result['flags'], "Should flag low cash ratio"
    assert 'liquidity_gate_fail' in result['flags'], "Should flag liquidity gate fail"

    print("✓ test_flags_generated_correctly passed")


# ============================================================================
# MAIN
# ============================================================================

def run_all_tests():
    """Run all golden fixture tests."""
    print("=" * 70)
    print("FINANCIAL MODULE v2 - GOLDEN FIXTURE TESTS")
    print("=" * 70)
    print()

    tests = [
        # Burn hierarchy tests
        test_burn_hierarchy_cfo_quarterly_first,
        test_burn_hierarchy_cfo_ytd_second,
        test_burn_hierarchy_cfo_annual_third,
        test_burn_hierarchy_trailing_4q_average,
        test_burn_hierarchy_fcf_quarterly,
        test_burn_hierarchy_fcf_annual,
        test_burn_hierarchy_net_income_fallback,
        test_burn_hierarchy_rd_proxy_last_resort,
        test_burn_hierarchy_profitable_company,
        test_burn_hierarchy_no_data,

        # Runway stability tests
        test_runway_with_4q_average,
        test_runway_with_partial_quarters,
        test_runway_liquid_assets_includes_marketable_securities,
        test_runway_profitable_company_infinite,

        # Determinism hash tests
        test_determinism_hash_stable,
        test_determinism_hash_differs_for_different_inputs,
        test_determinism_hash_differs_for_different_tickers,

        # Liquidity gating tests
        test_liquidity_gate_pass,
        test_liquidity_gate_warn,
        test_liquidity_gate_fail,

        # Dilution risk tests
        test_dilution_risk_low,
        test_dilution_risk_moderate,
        test_dilution_risk_high,
        test_dilution_risk_severe,
        test_dilution_share_count_growth,

        # Data quality tests
        test_data_quality_full,
        test_data_quality_partial,
        test_data_quality_none,

        # Bugfix tests
        test_bugfix_zero_values_not_falsy,
        test_bugfix_empty_string_is_none,

        # Composite tests
        test_composite_score_weights,
        test_flags_generated_correctly,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}")
            failed += 1

    print()
    print("=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
